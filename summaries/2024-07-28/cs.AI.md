New uploads on arXiv(cs.CV)

### Sparse vs Contiguous Adversarial Pixel Perturbations in Multimodal Models: An Empirical Analysis (https://arxiv.org/abs/2407.18251)
- **What's New**: 이 논문은 멀티모달 모델의 견고성을 평가하기 위해, 사전 처리된 입력 이미지에 L0-norm 섭동 공격 (perturbation attacks)을 적용하는 새로운 방법을 제안합니다. 이 방법은 이미지 영역의 0.04% 미만을 섭동시키면서, 섭동된 픽셀의 공간적 분포 (sparse positioning, contiguous shapes)를 다양하게 조절하여 공격 효과를 극대화합니다. 특히, 이 연구는 ALIGN, AltCLIP, GroupViT와 같은 최첨단 멀티모달 모델의 견고성을 sparse와 contiguous 픽셀 분포 섭동에 대해 처음으로 평가했습니다. 또한, CNN 기반 이미지 인코더 (Image Encoder)를 사용하는 모델이 ViT 기반 모델보다 섭동 공격에 취약하다는 것을 발견했습니다.



### Trajectory-aligned Space-time Tokens for Few-shot Action Recognition (https://arxiv.org/abs/2407.18249)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 움직임과 외형 표현을 분리하여 극소량의 데이터를 이용한 동작 인식(few-shot action recognition)을 위한 효과적이고 간단한 방법을 제안합니다. 특히, 최근 트래킹 기술의 발전(특히 포인트 트래젝토리와 자기 지도 학습)을 활용하여 움직임과 외형 정보를 포착하는 'Trajectory-Aligned Tokens(TAT)'를 구축합니다. 이를 통해 데이터 요구량을 크게 줄이면서도 중요한 정보를 유지할 수 있습니다. 이렇게 얻어진 정보를 처리하기 위해, 마스크 공간-시간 트랜스포머(Masked Space-Time Transformer)를 사용하여 극소량의 데이터로 동작 인식을 수행하는 데 필요한 정보를 효과적으로 집계합니다. 다양한 데이터셋에서 최첨단 성능을 달성했습니다.



### RegionDrag: Fast Region-Based Image Editing with Diffusion Models (https://arxiv.org/abs/2407.18247)
Comments:
          ECCV 2024, Project page: this https URL

- **What's New**: 이 논문은 기존의 Point-drag 방식 이미지 편집 방법의 단점을 해결하기 위해 Region-based copy-and-paste dragging 방법인 RegionDrag를 제안합니다. RegionDrag는 사용자가 handle과 target 영역을 지정하여 편집 명령을 표현할 수 있도록 하여, 더 정확한 제어가 가능하고 모호성을 줄일 수 있습니다. 또한, RegionDrag는 한 번의 반복으로 편집을 완료하여 Point-drag 방식보다 훨씬 빠릅니다. 이 논문은 또한 편집 중 안정성을 높이기 위해 attention-swapping 기법을 사용했습니다. 



### VGGHeads: A Large-Scale Synthetic Dataset for 3D Human Heads (https://arxiv.org/abs/2407.18245)
- **What's New**: 이 논문은 synthetic data를 사용하여 편향, 개인 정보 보호 및 윤리적 문제를 줄인 대규모 합성 데이터셋 VGGHeads를 소개합니다. 이 데이터셋은 100만 개 이상의 고해상도 이미지로 구성되며 각 이미지는 자세한 3D 머리 메시, 얼굴 랜드마크 및 바운딩 박스로 주석이 달려 있습니다. 또한 이 논문은 단일 이미지에서 머리 감지와 머리 메시 재구성을 동시에 수행할 수 있는 새로운 모델 아키텍처를 소개합니다.



### RefMask3D: Language-Guided Transformer for 3D Referring Segmentation (https://arxiv.org/abs/2407.18244)
Comments:
          ACM MM 2024, Code: this https URL

- **What's New**: 이 논문은 3D 참조 분할(3D referring segmentation)을 위한 새로운 접근 방식인 RefMask3D를 제안합니다. 이 방법은 기존의 두 단계 접근 방식인 분할 후 매칭(segmentation-then-matching) 방식을 벗어나 언어 정보를 전체적으로 활용하는 통합적이고 효율적인 엔드 투 엔드 파이프라인을 제안합니다. RefMask3D는 언어와 시각의 상호 작용과 이해를 개선하기 위해 기하 강화 그룹-단어 주의 (Geometry-Enhanced Group-Word Attention) 와 언어 원시 생성 (Linguistic Primitives Construction), 그리고 객체 클러스터 모듈 (Object Cluster Module) 을 사용합니다.



### BIV-Priv-Seg: Locating Private Content in Images Taken by People With Visual Impairments (https://arxiv.org/abs/2407.18243)
- **What's New**: 시각 장애인(BLV)이 촬영한 사진에 포함된 개인 정보를 보호할 수 있는 기술 개발을 위한 새로운 로컬라이제이션 데이터셋 BIV-Priv-Seg를 소개합니다. 이 데이터셋은 1,028개의 이미지와 16가지 개인 정보 객체에 대한 분할 주석(segmentation annotations)을 포함하고 있습니다. 이 논문은 BIV-Priv-Seg 데이터셋의 특징을 분석하고, 최신 모델들의 개인 정보를 포함한 객체 위치 파악 성능을 평가합니다.



### LION: Linear Group RNN for 3D Object Detection in Point Clouds (https://arxiv.org/abs/2407.18232)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 3D 객체 탐지에서 장거리 관계 (long-range relationship) 모델링을 위한 새로운 윈도우 기반 프레임워크인 LION (LInear grOup RNN)을 제안합니다. LION은 선형 그룹 RNN (linear group RNN)을 사용하여 장거리 관계를 효과적으로 모델링할 수 있으며, 기존의 트랜스포머 기반 방법보다 훨씬 큰 그룹에서 특징 상호 작용 (feature interaction) 을 허용합니다.



### Geometry Fidelity for Spherical Images (https://arxiv.org/abs/2407.18207)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 논문에서는 기존의 2D 이미지를 위한 FID(Fréchet Inception Distance)를 spherical image에 직접 적용하는 것이 geometric fidelity 측면에서 부족하다는 것을 보여줍니다. 이를 해결하기 위해, spherical image의 geometric constraints를 고려하는 새로운 두 가지 지표인 OmniFID(Omnidirectional FID)와 DS(Discontinuity Score)를 제안합니다. OmniFID는 cubemap projection을 이용하여 spherical image format의 field-of-view 요구사항을 추가적으로 고려하여 FID를 확장한 것입니다. DS는 spherical image의 2D 표현의 경계에서 continuity를 측정하는 kernel-based seam alignment score입니다. 실험 결과 OmniFID와 DS는 FID로는 감지되지 않는 geometry fidelity 문제를 정량화할 수 있는 것을 보여줍니다.

- **Technical Details**: OmniFID는 cubemap projection을 이용하여 FID를 확장한 것입니다. cubemap projection은 spherical image의 geometric constraints를 고려하여 field-of-view 요구사항을 더 잘 만족시키도록 합니다. DS는 spherical image의 2D 표현의 경계에서 continuity를 측정하는 kernel-based seam alignment score입니다. 이는 spherical image의 2D 표현에서 발생할 수 있는 seams를 감지하고 평가하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, OmniFID와 DS는 FID로는 감지되지 않는 geometry fidelity 문제를 정량화할 수 있는 것을 보여줍니다. 특히, OmniFID는 field-of-view reduction을 잘 포착하며, DS는 seam alignment를 정확하게 측정할 수 있습니다. 이는 spherical image generation model을 평가하는 데 매우 유용한 지표입니다.



### PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations (https://arxiv.org/abs/2407.18178)
- **What's New**: 이 논문에서는 인터넷 데이터로부터 피아노 연주 로봇 에이전트를 훈련하는 새로운 프레임워크인 PianoMime을 소개합니다. 기존 연구와 달리 PianoMime은 Youtube에서 다양한 피아노 연주 영상을 활용하여 일반적인 연주 에이전트를 훈련합니다. 이 프레임워크는 데이터 준비, 정책 학습, 정책 증류의 세 단계로 나뉩니다. 각 단계는 Youtube 영상에서 유용한 정보 추출, 특정 곡에 대한 전문가 정책 학습, 그리고 전문가 정책들을 하나의 일반화된 에이전트로 통합하는 과정을 포함합니다. PianoMime은 다양한 정책 설계를 실험하고 훈련 데이터 양이 에이전트의 일반화 능력에 미치는 영향을 평가합니다. 결과적으로, PianoMime은 학습된 에이전트가 데이터셋에 포함되지 않은 새로운 곡을 최대 56%의 F1 점수로 연주할 수 있음을 보여줍니다.



### Taxonomy-Aware Continual Semantic Segmentation in Hyperbolic Spaces for Open-World Perception (https://arxiv.org/abs/2407.18145)
- **What's New**: This paper introduces TOPICS (Taxonomy-Oriented Poincaré-regularized Incremental-Class Segmentation), a novel approach for class-incremental semantic segmentation that leverages hyperbolic space and taxonomy-tree structures to address the challenge of catastrophic forgetting while enabling the model to learn new classes effectively. TOPICS addresses limitations of existing methods by providing plasticity for old classes and incorporating pseudo-labeling of the background and relational constraints to ensure a robust structure for combating forgetting.



### XS-VID: An Extremely Small Video Object Detection Datas (https://arxiv.org/abs/2407.18137)
- **What's New**: 이 논문은 매우 작은 객체를 포함하는 새로운 비디오 객체 검출 (SVOD) 데이터셋인 XS-VID를 제안한다. 이 데이터셋은 기존 데이터셋과 비교하여 훨씬 더 다양한 크기의 작은 객체와 다양한 장면을 포함하며, 특히 매우 작은 객체 (es, 0~12^2 픽셀)의 수가 매우 많다.  또한, 매우 작은 객체 검출에 특화된 YOLOFT라는 새로운 SVOD 모델을 제안한다. YOLOFT는  optical flow를 기반으로 한 recurrent all-pairs Field Transforms를 YOLOv8에 통합하여  local feature associations를 강화하고  temporal motion features를 통합함으로써  SVOD의 정확도와 안정성을 높인다.  



### $\mathbb{X}$-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs (https://arxiv.org/abs/2407.18134)
- **What's New**: 본 논문은 표준적인 contrastive loss를 수정하여 각 샘플이 다른 샘플들과 어떻게 연관되는지 명시적으로 인코딩하는 새로운 방법을 제안합니다. 기존 contrastive loss는 하나의 샘플만 positive 샘플로 취급하여 샘플 간의 유사성을 무시했지만, 본 논문에서 제안하는 새로운 목표 함수는 샘플 간 유사성을 명시적으로 고려합니다.



### Estimating Earthquake Magnitude in Sentinel-1 Imagery via Ranking (https://arxiv.org/abs/2407.18128)
- **What's New**: 이 논문은 지진의 크기를 추정하는 새로운 접근 방식을 제안하여, 지진 크기를 추정하는 문제를 metric learning (메트릭 학습) 문제로 재정의합니다. 기존의 회귀 (regression) 만 사용하는 방법과 달리, 제안된 방법은 Sentinel-1 위성 이미지에서 지진 크기를 예측할 뿐만 아니라, 서로 다른 크기의 샘플들을 순위를 매기도록 모델을 훈련합니다. 이러한 이중 목표 훈련 (dual-objective training) 은 제한된 데이터에서 모델의 일반화 능력을 향상시켜 정확성과 견고성을 개선합니다.  



### Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images (https://arxiv.org/abs/2407.18125)
- **What's New**: 본 연구는 의료 영상에서 랜드 마크 검출을 위한 확산 모델 기반의 새로운 자기 지도 학습 전처리 프로토콜을 소개합니다. 이 프로토콜은 제한된 수의 레이블이 있는 훈련 이미지 (최대 50개) 를 사용하여 정확한 랜드 마크 검출을 가능하게 하며, ImageNet 지도 학습 전처리 및 최첨단 자기 지도 학습 전처리 성능을 능가합니다. 본 연구는 확산 모델을 자기 지도 학습 방식으로 랜드 마크 검출에 적용한 첫 번째 시도이며, 데이터 부족 문제를 완화하기 위한 유용한 전처리 접근 방식을 제공합니다.



### Efficient Inference of Vision Instruction-Following Models with Elastic Cach (https://arxiv.org/abs/2407.18121)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이 논문은 지식 종속 가능성 (KDA) 라는 새로운 자동 평가 지표를 제안하여 MCQ가 학생의 지식을 제대로 평가하는지 측정합니다. 또한, KDA를 근사하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하여 pretrained language model을 이용해 학생의 문제 해결 방식을 모방합니다.



### Keypoint Promptable Re-Identification (https://arxiv.org/abs/2407.18112)
- **What's New**: 본 논문은 기존 occluded person ReID (재식별) 방법들이 간과했던 다중 인물 모호성 (MPA) 문제를 해결하기 위해 Keypoint Promptable ReID (KPR) 라는 새로운 방법론을 제안합니다. KPR은 입력 bounding box에 특정 대상을 가리키는 semantic keypoints (의미적 핵심 지점) 를 추가하여 ReID 문제를 재정의합니다. 또한, 기존 ReID 데이터셋에 없는 keypoints 라벨을 갖춘 새로운 ReID 데이터셋인 Occluded-PoseTrack ReID (Occ-PTrack) 를 소개합니다.  



### DINOv2 Rocks Geological Image Analysis: Classification, Segmentation, and Interpretability (https://arxiv.org/abs/2407.18100)
- **What's New**: 본 논문에서는 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 MCQ가 대상 사실을 얼마나 잘 반영하는지를 측정합니다. KDA는 학생의 응답을 분석하여 측정할 수 있으며, 이를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.



### SSTD: Stripe-Like Space Target Detection using Single-Point Supervision (https://arxiv.org/abs/2407.18097)
- **What's New**: 본 논문은 우주 상황 인식 (Space Situational Awareness) 및 우주선 행동 평가에 중요한 역할을 하는 스트라이프형 우주 목표물 탐지 (Stripe-like Space Target Detection, SSTD)를 위한 새로운 데이터셋 'AstroStripeSet'과 새로운 의사 레이블 진화 (pseudo-label evolution) 티처-스튜던트 프레임워크를 제안합니다. AstroStripeSet은 공개적으로 제공되는 데이터셋 부족, 빛과 별의 간섭, 스트라이프형 목표물의 변동성으로 인한 픽셀 단위 주석 (pixel-level annotation)의 어려움 등 SSTD 분야의 3가지 과제를 해결하기 위해 고안되었습니다. 또한, 단일 지점 감독 (single-point supervision)을 사용하는 의사 레이블 진화 티처-스튜던트 프레임워크는 Segment Anything Model (SAM)의 제로-샷 능력을 활용하여 초기 의사 레이블을 생성하고 이를 반복적으로 개선합니다. 이 프레임워크에서 미세 조정된 StripeSAM은 티처 역할을 하고, 새로 개발된 StripeNet은 학생 역할을 맡아 의사 레이블의 질을 향상시켜 분할 성능을 지속적으로 개선합니다. 또한 스트라이프형 목표물의 선형 특징에 맞게 새롭게 고안된 손실 함수 'GeoDice'를 도입했습니다. 광범위한 실험 결과, 제안된 방법의 성능이 모든 평가 지표에서 완전 감독 방법과 동일하게 나타났으며, 새로운 최첨단 (SOTA) 벤치마크를 설정했습니다. 데이터셋과 코드는 공개적으로 제공될 예정입니다.



### HVM-1: Large-scale video models pretrained with nearly 5000 hours of human-like video data (https://arxiv.org/abs/2407.18067)
Comments:
          10 pages, 5 figures, 1 table; code & models available from this https URL

- **What's New**: 이 논문은 5,000시간 가까운 인간과 유사한 비디오 데이터로 학습된 대규모 비디오 모델 HVM-1을 소개합니다. HVM-1은 ST-MAE 알고리즘을 사용하여 학습되었으며, 224x224 및 448x448 픽셀의 공간 해상도로 학습된 두 개의 6억 3천 3백만 매개변수 모델이 공개되었습니다. 이 모델은 짧은 액션 지향 비디오 클립으로 학습된 Kinetics-700 모델과 비교하여 하위 작업(few-shot video 및 image recognition)에서 경쟁력 있는 성능을 보여줍니다. 또한 HVM-1 모델은 동일한 데이터로 이미지 기반 MAE 알고리즘으로 학습된 모델에 비해 더 정확하고 견고한 객체 표현을 학습하여 자연 비디오에서 시간적 규칙성을 예측하는 학습이 더 나은 객체 표현을 학습하는 데 기여한다는 가능성을 보여줍니다.



### GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution (https://arxiv.org/abs/2407.18046)
Comments:
          13 pages, 12 figures

- **What's New**: 이 논문은 Implicit Neural Representations (INRs) 기반의 Arbitrary-Scale Super-Resolution (ASSR) 기술에서 2D Gaussian Splatting (2DGS)을 활용하여 기존 방식의 한계를 극복한 새로운 방법인 GaussianSR을 제안한다. 기존 방법들은 픽셀을 discrete point로 취급하는 반면, GaussianSR은 픽셀을 continuous Gaussian field로 표현하여 개선된 성능을 제공한다.



### TiCoSS: Tightening the Coupling between Semantic Segmentation and Stereo Matching within A Joint Learning Framework (https://arxiv.org/abs/2407.18038)
- **What's New**: 이 논문은 semantic segmentation과 stereo matching을 결합하는 새로운 학습 프레임워크인 TiCoSS를 소개하며, 두 가지 태스크를 더욱 긴밀히 연결하기 위한 세 가지 혁신적인 기술을 제시한다. 1) Selective Inheritance Gates (SIGs)를 사용하여 이전 레이어의 유용한 정보를 현재 레이어로 전달하는 Tightly-Coupled, Gated Feature Fusion (TGF) 전략, 2) 가장 풍부한 local spatial details를 가진 fused features를 사용하여 각 분기에서 deep supervision을 안내하는 Hierarchical Deep Supervision (HDS) 전략, 3) stereo matching loss, Semantic Consistency-Guided (SCG) loss, Disparity Inconsistency-Aware (DIA) loss, Deep Supervision Consistency Constraint (DSCC) loss를 결합한 Coupling Tightening (CT) loss.



### AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild (https://arxiv.org/abs/2407.18034)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 자동으로 다중 선택 질문(MCQ)을 생성하는 새로운 방법을 제안하며, 기존의 평가 지표인 BLEU, ROUGE, METEOR와 달리 MCQ가 교육적 가치를 얼마나 갖는지 측정하는 새로운 지표인 Knowledge Dependent Answerability(KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하는 지표입니다. 이 논문은 또한 KDA를 자동화하기 위한 두 가지 새로운 지표인 KDA_disc와 KDA_cont를 제안하고, 이러한 지표가 실제 교육 환경에서 유용성과 높은 상관관계를 갖는다는 것을 실험적으로 보여줍니다.



### Investigation to answer three key questions concerning plant pest identification and development of a practical identification framework (https://arxiv.org/abs/2407.18000)
Comments:
          40 pages, 10 figures

- **What's New**: 이 논문은 이미지 기반 식물 해충 식별 시스템을 위한 실용적이고 견고한 자동 진단 시스템 개발을 위한 3가지 핵심 연구 질문 (RQ)을 제시하며 이를 기반으로 정확하고 견고하며 빠른 식물 해충 식별 프레임워크를 개발했습니다. 이 프레임워크는 27개 농장에서 촬영된 오이, 토마토, 딸기, 가지의 잎 앞면, 잎 뒷면, 열매, 꽃 등 4가지 식물 부분의 78가지 조합과 20가지 해충 종을 포함하여 334,000장의 이미지로 구성됩니다. 이를 통해, (1) 모델의 적절한 평가를 위해 테스트 데이터는 훈련 이미지가 수집된 현장의 이미지를 포함하지 않거나, 테스트 세트의 다양성을 높이기 위한 다른 고려 사항을 고려해야 함, (2) 잎, 열매와 같은 ROI의 사전 추출은 식별 정확도를 높이는 데 도움이 됨, (3) 동일한 방제 방법을 사용하는 밀접하게 관련된 종과 동일한 해충에 대한 교차 작물 훈련 방법을 통합하는 것이 효과적임을 확인했습니다.



### Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography (https://arxiv.org/abs/2407.17996)
- **What's New**: 이 논문에서는 모바일 장치에 작은 분광계를 통합하여 이미지 품질을 향상시키고 새로운 다운스트림 작업을 용이하게 하는 RGB-Spectral 분해 모델 기반의 향상 프레임워크를 제안합니다. 이 프레임워크는 RGB와 저해상도 다중 스펙트럼 이미지(Lr-MSI)의 상호 보완성을 활용하여 그림자, 반사율 및 재료 의미적 사전 정보를 예측합니다. 이러한 사전 정보는 HDRNet에 통합되어 동적 범위 향상, 색상 매핑 및 그리드 전문가 학습을 촉진합니다. 또한 모바일 스펙 데이터셋을 구축하여 연구를 지원하고 Lr-MSI가 톤 향상 작업에 효과적임을 실험적으로 입증합니다. 이 논문은 모바일 사진에서 스펙트럼 비전을 발전시키기 위한 견고한 기반을 마련하고자 합니다.



### SaccadeDet: A Novel Dual-Stage Architecture for Rapid and Accurate Detection in Gigapixel Images (https://arxiv.org/abs/2407.17956)
Comments:
          This paper is accepted to ECML-PKDD 2024

- **What's New**: 본 논문에서는 기존의 MCQ 생성 평가 지표가 MCQ의 교육적 가치를 고려하지 않는다는 문제점을 지적하며, 새로운 자동 평가 지표인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가하는 능력을 평가합니다. KDA를 측정하기 위해 Human Survey를 기반으로 한 KDA와  Pre-trained Language Model을 활용하여 학생들의 문제 해결 행동을 모방한 KDA_disc 및 KDA_cont라는 두 가지 자동 평가 지표를 제시합니다. 실제 강의실 환경에서 Human evaluation을 통해 KDA_disc와 KDA_cont는 실제 강의실 환경에서 사용성과 강한 상관관계를 가지고 있음을 보여줍니다. 또한, n-gram 기반 유사도 지표와 함께 사용될 경우, KDA_disc와 KDA_cont는 전문가가 평가한 다양한 MCQ 품질 척도에 대한 예측력이 높음을 보여줍니다.



### Scaling Training Data with Lossy Image Compression (https://arxiv.org/abs/2407.17954)
Comments:
          21 pages, 27 figures

- **What's New**: 본 논문은 기존의 데이터 크기와 모델 파라미터를 고려한 scaling law에서 더 나아가 데이터 저장 공간을 고려한 **storage scaling law**를 제안합니다. 이는 이미지와 같은 아날로그 데이터를 디지털 방식으로 저장할 때 손실 압축(lossy compression)을 사용하면 저장 공간을 줄일 수 있지만, 압축으로 인해 모델 성능 저하가 발생할 수 있다는 점에서 착안했습니다.



### BetterDepth: Plug-and-Play Diffusion Refiner for Zero-Shot Monocular Depth Estimation (https://arxiv.org/abs/2407.17952)
- **What's New**: 본 논문은 단일 카메라 깊이 추정(MDE) 분야에서 깊이 추정의 정확성을 높이는 새로운 방법인 BetterDepth를 제안합니다. 기존의 대규모 데이터셋 기반 MDE 방법들은 현실 환경에서 견고한 성능을 보이지만, 세부적인 정보를 충분히 포착하지 못하는 단점이 있습니다. 반면, 최근 확산 기반 MDE 접근 방식들은 세부 정보 추출 능력이 뛰어나지만, 다양한 데이터셋에서 견고한 기하학적 사전 정보를 얻는 데 어려움을 겪어 기하학적으로 복잡한 장면에서는 여전히 부족한 모습을 보입니다. BetterDepth는 이러한 두 가지 접근 방식의 장점을 결합하여 기하학적으로 정확한 어파인 불변 MDE 성능을 효율적으로 달성하면서도 세밀한 정보까지 포착합니다. 특히, BetterDepth는 사전 훈련된 MDE 모델의 예측을 깊이 조건으로 사용하여 전역 깊이 컨텍스트를 포착하고, 입력 이미지를 기반으로 세부 정보를 반복적으로 개선하는 조건부 확산 기반 개선기입니다. BetterDepth의 훈련을 위해, 전역 사전 정렬 및 로컬 패치 마스킹 방법을 제안하여 깊이 조건에 대한 정확성을 유지하면서도 세밀한 장면 정보를 포착하도록 학습합니다. BetterDepth는 소규모 합성 데이터셋에서 효율적인 훈련을 통해 다양한 공개 데이터셋과 실제 환경에서 최첨단 제로샷 MDE 성능을 달성합니다. 더욱이 BetterDepth는 추가적인 재훈련 없이도 다른 MDE 모델의 성능을 향상시킬 수 있는 플러그 앤 플레이 방식으로 사용할 수 있습니다.



### Real Time American Sign Language Detection Using Yolo-v9 (https://arxiv.org/abs/2407.17950)
Comments:
          11 pages, 13 figures, 1 table

- **What's New**: 본 논문은 실시간 미국 수어 검출에 초점을 맞추고 있습니다. YOLO는 2015년 처음 출시된 CNN 기반 모델로, 실시간 검출 능력으로 인해 최근 몇 년 동안 인기를 얻었습니다. 본 연구는 특히 2024년에 출시된 YOLO-v9 모델을 대상으로 합니다. YOLO-v9는 새롭게 출시된 모델이기 때문에, 수어 검출에 관한 연구가 많지 않으며, 특히 이 모델에 대한 연구는 많지 않습니다. 본 논문은 YOLO-v9가 어떻게 작동하는지, 그리고 이전 모델보다 뛰어난지에 대한 심층적인 통찰력을 제공합니다.



### Segmentation by registration-enabled SAM prompt engineering using five reference images (https://arxiv.org/abs/2407.17933)
Comments:
          Accepted to the 11th International Workshop on Biomedical Image Registration (WBIR 2024)

- **What's New**: 이 논문은 의료 영상 분할을 위해 SAM (Segment Anything Model) 을 활용하는 새로운 prompt engineering framework를 제안한다. 이 방법은 이미지 레지스트레이션 (image registration) 알고리즘을 사용하여 새로운 이미지와 소량의 참조 이미지들을 정렬하여, 분할 레이블 없이도 SAM을 활용할 수 있도록 한다. 특히, 연골 수술 후 나타나는 새로운 영상 패턴을 SAM이 학습할 수 있도록 하는데 초점을 맞춘다.



### Guided Latent Slot Diffusion for Object-Centric Learning (https://arxiv.org/abs/2407.17929)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 생성된 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다.  KDA_disc와 KDA_cont는 pre-trained language model을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화한 자동 평가 지표입니다.  KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계가 있으며, n-gram 기반 유사성 지표와 결합하면 다양한 전문가 평가 MCQ 품질 척도에 대한 강력한 예측력을 보입니다.



### Invariance of deep image quality metrics to affine transformations (https://arxiv.org/abs/2407.17927)
Comments:
          12 pages 13 figures

- **What's New**: 이 논문은 기존 이미지 품질 평가 지표들이 인간의 시각적 인지 능력을 제대로 반영하지 못한다는 점을 지적하고, affine transformation (회전, 이동, 크기 조정, 스펙트럼 조명 변화) 에 대한 인간의 시각적 불변성 (invariance) 을 고려한 새로운 평가 방법을 제안한다. 이 방법은 특정 지표의 invisibility threshold (눈에 띄지 않는 변환의 한계) 를 정의하고, 인간의 시각적 인지와 비교하여 평가한다.



### Separating Novel Features for Logical Anomaly Detection: A Straightforward yet Effective Approach (https://arxiv.org/abs/2407.17909)
- **What's New**: 본 논문은 기존의 지식 증류 기반의 논리적 이상 탐지 방법에서 발생하는 오탐 문제를 해결하기 위한 새로운 방법을 제안합니다. 특히, EfficientAD (Batzner et al., 2023) 를 기반으로 하여, 학습 과정에서 제약 조건을 추가하여 글로벌 특징 추출기가 서로 다른 특징을 유지하도록 합니다. 이를 통해 오탐을 줄이고 MVTec LOCO AD 데이터셋에서 92.0%의 AUROC를 달성합니다. (AUROC (Area Under the Receiver Operating Characteristic Curve))



### Amortized Posterior Sampling with Diffusion Prior Distillation (https://arxiv.org/abs/2407.17907)
- **What's New**: 이 논문은 역문제를 푸는 데 필요한 후방 분포(posterior distribution)에서 샘플을 추출하기 위한 변분 추론(variational inference) 방식을 제안합니다. 사전 훈련된 확산 모델(diffusion model)에서, 제안된 방식은 조건부 흐름 모델(conditional flow model)을 훈련하여 제안된 변분 분포(proposal variational distribution)와 확산 모델을 통해 암묵적으로 정의된 후방 분포 사이의 차이를 최소화합니다. 일단 훈련되면, 흐름 모델은 측정값에 대해 비용을 지불하면서 한 번의 NFE(neural function evaluation)로 후방 분포에서 샘플을 추출할 수 있습니다. 제안된 방법은 효율적인 후방 샘플링을 위해 확산 사전(diffusion prior)을 증류하는 새로운 방법을 제시합니다. 이 논문에서는 제안된 방법이 유클리드 공간(Euclidean space)의 표준 신호뿐만 아니라 다양체(manifold)의 신호에도 적용될 수 있음을 보여줍니다.



### Hierarchical Object Detection and Recognition Framework for Practical Plant Disease Diagnosis (https://arxiv.org/abs/2407.17906)
Comments:
          6 pages, 3 figures

- **What's New**: 본 논문은 식물 질병 진단을 위한 계층적 객체 탐지 및 인식 프레임워크(HODRF)를 제안한다. HODRF는 객체 탐지(OD)와 분류(CL)의 장점을 결합하여 기존 방법의 단점을 보완한다.



### StreamMOS: Streaming Moving Object Segmentation with Multi-View Perception and Dual-Span Memory (https://arxiv.org/abs/2407.17905)
Comments:
          8 pages, 7 figures

- **What's New**: 이 논문은 기존의 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)가 단순히 문장 유사도만 평가하는 한계를 지적하며, MCQ의 교육적 가치를 측정하는 새로운 지표인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안합니다. KDA는 MCQ가 실제로 학생의 지식을 평가할 수 있는지 측정하는 지표입니다. 또한, 이 논문은 KDA를 자동으로 계산할 수 있는 두 가지 지표(KDA_disc, KDA_cont)를 제안하고, Human Evaluation을 통해 이 지표들이 실제 교육 환경에서의 유용성과 높은 상관관계를 보인다는 것을 증명했습니다. 



### Exploring the Effect of Dataset Diversity in Self-Supervised Learning for Surgical Computer Vision (https://arxiv.org/abs/2407.17904)
Comments:
          accepted - Data Engineering in Medical Imaging (DEMI) Workshop @ MICCAI2024

- **What's New**: 본 논문은 수술 영상에서의 딥러닝 모델 robustness를 높이기 위한 새로운 self-supervised learning (SSL) 기법을 제안합니다. 기존 방법들은 데이터셋의 spurious correlation에 영향을 받았던 반면, 본 연구에서는 여러 개의 counterfactual을 생성하고 집합적 의사 결정 (collective decisions)을 통해 robust하게 단어들의 인과관계를 파악합니다. 이는 모델 bias를 줄이고 다양한 측면에서 향상된 성능을 제공합니다.



### Advancing 3D Point Cloud Understanding through Deep Transfer Learning: A Comprehensive Survey (https://arxiv.org/abs/2407.17877)
Comments:
          55 pages, 9 tables, and 15 figures

- **What's New**: 이 논문은 3D 포인트 클라우드 (3DPC) 데이터를 이해하기 위한 딥 전이 학습 (DTL) 및 도메인 적응 (DA)에 대한 최초의 리뷰를 제공한다.



### Mew: Multiplexed Immunofluorescence Image Analysis through an Efficient Multiplex Network (https://arxiv.org/abs/2407.17857)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 기존의 MCQ 평가 메트릭의 한계를 극복하기 위해 **지식 종속 가능성 (Knowledge Dependent Answerability, KDA)** 이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하는 데 초점을 맞춥니다. 또한, KDA를 근사화하기 위해 **KDA_disc** 와 **KDA_cont** 라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.



### FlexiEdit: Frequency-Aware Latent Refinement for Enhanced Non-Rigid Editing (https://arxiv.org/abs/2407.17850)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 교육적 가치를 고려하지 못하는 기존의 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR) 대신,  MCQ의 대답 가능성(answerability)을 측정하는 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안한다. KDA는 대상 사실에 대한 학생의 지식을 평가하는 MCQ의 능력을 측정하며, KDA_disc와 KDA_cont는 전이 학습(transfer learning) 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 추정한다.



### Move and Act: Enhanced Object Manipulation and Background Integrity for Image Editing (https://arxiv.org/abs/2407.17847)
- **What's New**: 본 논문은 자동 MCQ 생성 시스템의 교육적 가치를 평가하는 새로운 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 학생의 지식을 정확히 평가할 수 있는지 측정하며, 기존의 n-gram 기반 유사성 메트릭(BLEU, ROUGE, METEOR)의 한계를 극복합니다. 또한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 학생의 문제 해결 능력을 모방하는 사전 훈련된 언어 모델을 활용합니다. 인간 평가를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보이는 것으로 확인되었습니다.



### DragText: Rethinking Text Embedding in Point-based Image Editing (https://arxiv.org/abs/2407.17843)
Comments:
          22 pages, 18 figures

- **What's New**: 이 논문은 이미지 편집 과정에서 텍스트 임베딩의 역할을 심층적으로 분석하고, DragText라는 새로운 방법을 제안하여 기존 텍스트 기반 이미지 편집 방법의 효율성을 향상시킵니다. 특히, 이미지 편집 과정에서 텍스트 임베딩이 이미지 임베딩과의 불일치 문제로 인해 발생하는 드래그 중단 (drag halting) 현상을 해결하는 데 중점을 둡니다. 



### UMono: Physical Model Informed Hybrid CNN-Transformer Framework for Underwater Monocular Depth Estimation (https://arxiv.org/abs/2407.17838)
- **What's New**: 본 논문에서는 수중 환경의 특성을 고려하여 단일 이미지에서 수중 깊이를 추정하는 새로운 End-to-end 학습 프레임워크인 UMono를 제안합니다. UMono는 수중 이미지 형성 모델의 특성을 네트워크 구조에 통합하고, 수중 이미지의 지역적 특징과 전역적 특징을 효과적으로 활용합니다.



### Towards the Spectral bias Alleviation by Normalizations in Coordinate Networks (https://arxiv.org/abs/2407.17834)
- **What's New**: 이 논문은 좌표 네트워크에서 스펙트럼 편향 (spectral bias) 문제를 해결하기 위해 normalization 기술을 적용한 새로운 방법을 제안합니다. 기존 좌표 네트워크는 고주파 성분을 학습하는 데 어려움을 겪었지만, 이 논문에서 제안된 normalization 기술은 NTK의 고유값 분포를 개선하여 이 문제를 해결합니다. 특히, BN (Batch Normalization)과 LN (Layer Normalization)을 조합한 두 가지 새로운 normalization 기술인 GN (Global Normalization)과 CN (Cross Normalization)을 제안합니다.



### Image Segmentation via Divisive Normalization: dealing with environmental diversity (https://arxiv.org/abs/2407.17829)
- **What's New**: 이 논문은 이미지 분할(segmentation)에 대한 생물학적으로 영감을 받은 Divisive Normalization (분할 정규화)의 효과를 체계적으로 분석했습니다. 특히, 다양한 데이터 소스와 환경 요인(조도, 명암비, 스펙트럼 조명)에 따른 성능을 조사했습니다.



### Enhancing Model Performance: Another Approach to Vision-Language Instruction Tuning (https://arxiv.org/abs/2407.17813)
- **What's New**: 이 논문은 멀티모달 LLM 프레임워크를 위한 새로운 접근 방식인 Bottleneck Adapter를 소개합니다. 이 접근 방식은 이미지 인코더와 LLM을 연결하는 경량 어댑터를 사용하여 큰 신경망을 필요로 하지 않고 멀티모달 LLM 프레임워크 전체를 함께 최적화할 수 있도록 합니다. 기존의 모듈식 학습 방식과 달리 이 접근 방식은 끝단 최적화 체계를 채택하여 어댑터와 함께 훨씬 작은 매개변수 집합을 사용하여 전체 멀티모달 LLM을 함께 최적화할 수 있도록 합니다. 본 논문의 접근 방식은 90.12%의 정확도로 강력한 성능을 보여주며, 인간 수준의 성능(88.4%)과 LaVIN-7B(89.41%)를 능가합니다.



### A Unified Understanding of Adversarial Vulnerability Regarding Unimodal Models and Vision-Language Pre-training Models (https://arxiv.org/abs/2407.17797)
Comments:
          14 pages, 9 figures, published in ACMMM2024(oral)

- **What's New**: 이 논문은 VLP(Vision-Language Pre-training) 모델에 대한 새로운 적대적 공격 방법인 FGA(Feature Guidance Attack)를 제안합니다. FGA는 텍스트 표현을 사용하여 깨끗한 이미지의 왜곡을 유도하여 적대적 이미지를 생성합니다. 또한, FGA는 텍스트 공격을 통합하여 VLP 모델에 대한 공격 효과를 향상시키는 FGA-T(Feature Guidance with Text Attack)를 구축합니다.



### Harnessing Temporal Causality for Advanced Temporal Action Detection (https://arxiv.org/abs/2407.17792)
Comments:
          1st in Moment Queries track at the Ego4D Challenge 2024; 1st in Action Recognition, Action Detection, and Audio-Based Interaction Detection tracks at the EPIC-Kitchens Challenge 2024

- **What's New**: 본 논문에서는  'CausalTAD' 라는 새로운 모델을 제안하여 Temporal Action Detection (TAD) 작업에서 최첨단 성능을 달성했습니다. CausalTAD는 Causal Attention과 Causal Mamba를 결합하여  시간적 인과 관계를 활용하여  동작 경계 변화의 원인을 효과적으로 모델링합니다.



### Topology-Preserving Downsampling of Binary Images (https://arxiv.org/abs/2407.17786)
Comments:
          Accepted to The 18th European Conference on Computer Vision (ECCV) 2024

- **What's New**: 이 논문은 이진 이미지 다운샘플링(downsampling)을 위한 새로운 이산 최적화(discrete optimization) 기반 방법을 제시하여, 원본 이미지의 토폴로지(topology)를 보존하면서도 IoU(Intersection over Union)와 Dice score와 같은 지표를 통해 원본 이미지와 유사성을 유지하는 방법을 소개합니다. 기존의 이진 이미지 다운샘플링 방법은 이러한 토폴로지 보존 보장 기능이 없었습니다.



### How Lightweight Can A Vision Transformer B (https://arxiv.org/abs/2407.17783)
- **What's New**: 이 논문에서는 Vision Transformer를 간소화하는 Mixture-of-Experts (MoE) 전략을 제안합니다. 각 MoE 레이어의 전문가는 V와 W2를 공유하는 SwiGLU 피드포워드 네트워크입니다. 복잡한 어텐션이나 컨볼루션 메커니즘은 사용되지 않습니다. 깊이별 스케일링을 적용하여 히든 레이어의 크기를 점진적으로 줄이고 전문가 수는 단계별로 증가시킵니다. 그룹화된 쿼리 어텐션을 사용합니다. 소규모 데이터셋에 대한 사전 학습 유무와 관계없이 제안된 접근 방식을 연구하고 전이 학습이 이 규모에서 작동하는지 조사했습니다. 이 아키텍처는 0.67M 파라미터 크기에서도 경쟁력이 있음을 발견했습니다.



### DAC: 2D-3D Retrieval with Noisy Labels via Divide-and-Conquer Alignment and Correction (https://arxiv.org/abs/2407.17779)
Comments:
          accepted by ACM MM 2024

- **What's New**: 이 논문은 2D/3D 데이터에서의 오류 있는 라벨 문제를 해결하기 위해 DAC(Divide-and-conquer 2D-3D cross-modal Alignment and Correction) framework을 제안합니다. DAC는 MDD(Multimodal Dynamic Division)와 AAC(Adaptive Alignment and Correction) 두 가지 전략으로 구성되어 있습니다. MDD는 멀티모달 손실 분포를 기반으로 각 샘플의 신뢰도를 동적으로 모델링하여 샘플을 정확하게 분류합니다. AAC는 다른 전략을 통해 각 샘플을 활용하여 의미적 일관성을 향상시키고 노이즈 라벨로 인한 과적합을 완화합니다. 또한 200k 개의 샘플과 1156개의 현실적인 노이즈 라벨로 구성된 Objaverse-N200 벤치마크를 새롭게 제시하며, 다양한 벤치마크에서 실험을 통해 DAC의 우수성을 입증합니다.



### Mpox Detection Advanced: Rapid Epidemic Response Through Synthetic Data (https://arxiv.org/abs/2407.17762)
Comments:
          8 pages, 4 figures, 1 table

- **What's New**: 이 연구는 Mpox 병변을 감지하는 포괄적인 컴퓨터 비전 모델을 구축하기 위해 합성 데이터를 사용하는 새로운 접근 방식을 제시합니다. 이 연구는 합성 데이터를 활용하여 의료 응급 상황(예: 전염병, 생물 테러)에 신속하게 대응할 수 있는 질병 감지 모델 개발을 위한 새로운 방법을 제시합니다. 기존의 데이터 수집 방법은 이러한 상황에서 너무 느리기 때문에 최소한의 데이터로부터 신속하고 신뢰할 수 있는 모델을 생성하기 위한 혁신적인 방법이 필요합니다. SynthVision이라고 불리는 이 새로운 방법은 Fitzpatrick 척도(밝은 피부, 갈색 피부, 어두운 피부)에 따라 다양한 피부색의 신체 부위(얼굴, 등, 가슴, 다리, 목, 팔)에 있는 Mpox 병변을 나타내는 다양한 합성 이미지 세트를 생성하는 확산 모델을 사용합니다. 그런 다음 이 합성 데이터셋으로 비전 모델을 훈련 및 테스트하여 확산 모델이 고품질 훈련 데이터를 생성하는 효과와 비전 모델의 의료 이미지 인식 성능에 미치는 영향을 평가했습니다. 결과는 유망했습니다. 비전 모델은 Mpox 사례에 대해 96%의 정밀도와 재현율로 97%의 정확도를 달성했으며, 정상 및 기타 피부 질환 사례에 대해서도 마찬가지로 높은 지표를 보여주어 진짜 양성을 올바르게 식별하고 가짜 양성을 최소화할 수 있는 능력을 입증했습니다. 모델은 Mpox 사례에 대해 96%의 F1 점수를, 정상 및 기타 피부 질환에 대해서는 98%의 F1 점수를 달성하여 균형 잡힌 정밀도-재현율 관계를 반영하며, 따라서 예측의 신뢰성과 견고성을 보장합니다. 이 연구에서 제안된 SynthVision 방법론은 향후 의료 응급 상황에 대한 최소한의 데이터 입력으로 정확한 컴퓨터 비전 모델을 개발할 수 있는 잠재력을 보여줍니다.



### CRASH: Crash Recognition and Anticipation System Harnessing with Context-Aware and Temporal Focus Attentions (https://arxiv.org/abs/2407.17757)
- **What's New**: 본 논문에서는 자율주행 시스템을 위한 새로운 사고 예측 프레임워크인 CRASH를 제안합니다. CRASH는 객체 감지, 특징 추출, 객체 인식 모듈, 컨텍스트 인식 모듈, 다층 융합을 포함하는 5가지 구성 요소를 통합합니다.



### Enhancing Eye Disease Diagnosis with Deep Learning and Synthetic Data Augmentation (https://arxiv.org/abs/2407.17755)
Comments:
          18 pages, 7 figures, 2 Tables

- **What's New**: 이 논문은 당뇨병성 망막증 (Diabetic Retinopathy, DR) 의 조기 진단을 위한 새로운 앙상블 학습 기법을 제안합니다. 이 기법은 두 개의 기본 모델, DenseNet121과 InceptionV3를 사용하여 더 높은 정확도를 달성합니다. 또한 데이터 전처리 단계에서 다중 레이블 포맷을 사용하여 소수 클래스를 오버샘플링하여 균형 잡힌 데이터셋을 생성합니다.



### Balancing Complementarity and Consistency via Delayed Activation in Incomplete Multi-view Clustering (https://arxiv.org/abs/2407.17744)
- **What's New**: 본 논문은 불완전한 멀티뷰 클러스터링(IMC)에서 다른 뷰에서 얻는 유용한 보완 정보가 무시되는 문제를 해결하기 위해 Complementarity와 Consistency 정보를 효과적으로 균형을 맞추는 새로운 프레임워크인 CoCo-IMC를 제안합니다. 특히, 다른 뷰 간의 보완성과 일관성을 균형있게 맞추기 위한 지연 활성화(delayed activation) 이중 네트워크를 설계합니다. 지연 활성화는 일관성 학습 중에 무시되었던 보완 정보를 풍부하게 해줍니다. 그런 다음 조건부 엔트로피를 최소화하고 서로 다른 뷰 간의 상호 정보를 극대화하여 불완전한 정보를 복구하고 일관성 학습을 향상시킵니다. 이는 불완전한 데이터 복구에 지연 활성화를 통합하고 보완성과 일관성의 균형을 이루려는 최초의 이론적 시도입니다. 4개의 공개적으로 사용 가능한 데이터 세트에 대한 12개의 최첨단 기준선과의 광범위한 비교 실험에서 CoCo-IMC의 효과를 증명했습니다.



### Enhancing Fine-grained Object Detection in Aerial Images via Orthogonal Mapping (https://arxiv.org/abs/2407.17738)
- **What's New**: 본 논문은 항공 이미지 분석에서 세밀한 객체 검출 (Fine-Grained Object Detection, FGOD) 을 위한 효과적인 방법으로 직교 매핑 (Orthogonal Mapping, OM) 을 제안합니다. OM은 마지막 분류 레이어의 특징을 클래스별 직교 벡터 기저로 분리하여 특징 공간에서 직교 제약 조건을 도입합니다. 이를 통해 의미적 혼란 (semantic confusion) 을 완화하고 분류 정확도를 향상시킵니다. 또한 OM은 주요 객체 검출기 (object detectors) 에 쉽게 통합할 수 있습니다.



### ALMRR: Anomaly Localization Mamba on Industrial Textured Surface with Feature Reconstruction and Refinemen (https://arxiv.org/abs/2407.17705)
- **What's New**: 본 논문은 산업용 텍스처 이미지에서 비지도 학습 기반의 이상치 (Anomaly) 지역화 (Localization)를 위한 새로운 방법론인 ALMRR(Anomaly Localization method based on Mamba with Feature Reconstruction and Refinement)을 제안합니다. 기존의 이미지 재구성 기반 방법론의 한계점 (과도한 일반화)과 특징 재구성 기반 방법론의 한계점 (특징 구조의 중복성 및 이상치 정보 부족)을 극복하고자 합니다.



### SAM-MIL: A Spatial Contextual Aware Multiple Instance Learning Approach for Whole Slide Image Classification (https://arxiv.org/abs/2407.17689)
Comments:
          accepted by ACM Multimedia 2024

- **What's New**: 이 논문은 WSI 분류를 위한 새로운 MIL 프레임워크인 SAM-MIL을 제안합니다. SAM-MIL은 WSI에서 공간적 맥락 정보를 추출하고 이를 MIL 모델 학습에 명시적으로 통합하여 기존 MIL 모델의 성능을 향상시킵니다. 특히, 이 논문은 SAM을 사용하여 WSI에서 공간적 맥락을 추출하고, 이를 기반으로 SG2M (SAM-Guided Group Masking) 전략을 통해 인스턴스 불균형 문제를 완화하는 방식을 제안합니다. 또한, SAM-MIL은 인스턴스를 나누어 추가적인 가짜 백 (pseudo-bags)을 생성하여 학습 데이터를 보강하고, 가짜 백들 사이의 공간적 맥락 일관성을 유지함으로써 모델 성능을 더욱 향상시킵니다.



### CRASAR-U-DROIDs: A Large Scale Benchmark Dataset for Building Alignment and Damage Assessment in Georectified sUAS Imagery (https://arxiv.org/abs/2407.17673)
Comments:
          16 Pages, 7 Figures, 6 Tables

- **What's New**: 이 논문은 CRASAR-U-DROIDs라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 소형 무인 항공기 (sUAS)에서 수집한 지형 공간 이미지 (geospatial imagery)를 사용하여 건물 피해 평가와 공간 정렬을 수행합니다. 이 데이터셋은 재난 대응에 sUAS의 사용이 증가하고, 고해상도 지형 공간 sUAS 이미지를 기반으로 한 머신러닝과 컴퓨터 비전 모델에서 이전 연구가 부족하다는 점을 고려하여 만들어졌습니다. 또한, sUAS와 위성 이미지 사이의 추가적인 연구를 가능하게 하고, 운영적인 사용 사례와의 일관성을 유지하기 위한 목표를 가지고 있습니다. CRASAR-U-DROIDs는 10개의 연방 재해 (허리케인 이안, 허리케인 아이다, 허리케인 하비, 허리케인 이달리아, 허리케인 로라, 허리케인 마이클, 머셋 베이 화재, 메이필드 토네이도, 킬라우에아 폭발, 샹플랭 타워 붕괴)에서 수집된 52개의 오쏘모자이크 (orthomosaic)로 구성되어 있으며, 67.98 평방 킬로미터 (26.245 평방 마일)를 포괄합니다. 또한, 이 데이터셋은 21,716개의 건물 다각형과 피해 라벨, 그리고 7,880개의 조정 주석을 포함하고 있습니다. 이미지는 타일링 처리되어 130명의 주석자에게 제공되었으며, 주석자들은 공동 피해 척도 (Joint Damage Scale)에 따라 건물 다각형의 피해에 대한 인간 판단을 제공했습니다. 이러한 주석은 건물 다각형 피해 라벨을 개별적으로 검토한 후, 위원회에서 다시 검토하는 2단계 검토 과정을 통해 검토되었습니다. 또한, 건물 다각형은 더욱 성능이 좋은 머신러닝 모델을 학습할 수 있도록 이미지와 정확히 겹치도록 공간적으로 정렬되었습니다. CRASAR-U-DROIDs는 sUAS 오쏘모자이크 이미지의 가장 큰 라벨링된 데이터셋인 것으로 보입니다. 



### Unsqueeze [CLS] Bottleneck to Learn Rich Representations (https://arxiv.org/abs/2407.17671)
- **What's New**: 이 논문은 이미지의 의미 정보를 더 풍부하게 유지하면서도 압축된 표현을 얻을 수 있는 Unsqueezed Distillation-based Self-supervised Learning (UDI)라는 새로운 방법을 제안합니다. UDI는 여러 단계에서 추출된 정보를 종합적으로 활용하여 다중 모드 예측을 수행하며, 이를 통해 이미지의 의미 정보를 더 풍부하게 유지하는 데 기여합니다.



### SDLNet: Statistical Deep Learning Network for Co-Occurring Object Detection and Identification (https://arxiv.org/abs/2407.17664)
Comments:
          8 pages, 3 figures, ICMLT-2024. arXiv admin note: text overlap with arXiv:2403.17223

- **What's New**: 본 논문은 딥러닝 기반 기술의 발전과 함께, 보안 및 감시와 같은 다양한 분야에서 활용되는 동시 발생 객체 (co-occurring objects)의 탐지 및 식별을 위한 새로운 프레임워크인 SDLNet을 제안합니다. SDLNet은 다중 레이블 객체 카테고리에서 기본 객체 (base object)와 함께 동시 발생 객체를 식별합니다. SDLNet은 다중 레이블 탐지기 (multilabel detectors)를 사용하여 레이블을 발견하는 첫 번째 단계와, 동시 발생 매트릭스 분석 (co-occurrence matrix analysis)을 수행하는 두 번째 단계로 구성됩니다. 동시 발생 매트릭스 분석 단계에서는 기본 클래스와 자주 발생하는 클래스를 설정하여 동시 발생 통계를 학습하고, 이를 기반으로 연관 규칙 (association rules)을 구축하고 빈번한 패턴 (frequent patterns)을 생성합니다. SDLNet의 핵심은 기본 클래스를 인식하고 동시 발생 클래스를 고려하는 것입니다. 최종적으로 생성된 동시 발생 매트릭스는 기본 클래스와 해당 동시 발생 클래스를 보여줍니다.



### Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness (https://arxiv.org/abs/2407.17630)
- **What's New**: 본 논문은 기존 자동 MCQ 생성 평가 지표 (BLEU, ROUGE, METEOR)가 생성된 MCQ의 교육적 가치를 고려하지 않고, 골드 샘플과의 단어 유사성만 비교한다는 점을 지적합니다. 이를 해결하기 위해, MCQ의 대답 가능성(answerability)을 측정하는 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 제대로 평가하는지 측정합니다. 본 논문에서는 KDA를 측정하는 방법을 설명하고, 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안하여 KDA를 근사화합니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 방식을 모방합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.



### PEEKABOO: Hiding parts of an image for unsupervised object localization (https://arxiv.org/abs/2407.17628)
- **What's New**: 이 논문은 unsupervised object localization을 위한 새로운 single-stage learning framework인 PEEKABOO를 제안합니다. PEEKABOO는 이미지의 일부를 가리고 나머지 이미지 정보를 활용하여 객체의 위치를 추론하는 방식으로, pixel-level과 shape-level에서 context-based representation을 학습합니다.



### CoMoTo: Unpaired Cross-Modal Lesion Distillation Improves Breast Lesion Detection in Tomosynthesis (https://arxiv.org/abs/2407.17620)
Comments:
          ADSMI @ MICCAI 2024

- **What's New**: 이 논문은 Digital Breast Tomosynthesis (DBT) 영상에서 병변 탐지 정확도를 향상시키기 위한 새로운 프레임워크인 CoMoTo를 제안합니다. 기존 맘모그래피 데이터를 활용하여 DBT 모델의 학습을 향상시키는 방법을 제시합니다. 특히, Lesion-specific Knowledge Distillation (LsKD)과 Intra-modal Point Alignment (ImPA)라는 두 가지 새로운 구성 요소를 제안합니다.



### Quality Assured: Rethinking Annotation Strategies in Imaging AI (https://arxiv.org/abs/2407.17596)
Comments:
          Accepted at ECCV 2024, preprint, Computer Vision, Data Annotation

- **What's New**: 이 논문은 AI 기반 이미지 분석을 위한 신뢰할 수 있는 벤치마킹과 실제 응용을 위한 필수적인 기반인 고품질 참조 주석을 생성하는 문제를 연구했습니다. 이전 연구는 주석을 아웃소싱하는 수단으로 크라우드소싱에 초점을 맞추었지만, 주석 회사의 내부 품질 보증(QA) 프로세스에 대해서는 거의 주목하지 않았습니다. 따라서 이 연구는 주석 회사가 사용하는 QA가 주석 품질에 미치는 영향을 평가하고 데이터 주석 효율성을 극대화하기 위한 방법론을 고안하는 데 목표를 두었습니다. 연구팀은 4개의 주석 회사와 Amazon Mechanical Turk(MTurk)에서 924명의 주석자와 34명의 QA 작업자로부터 얻은 총 57,648개의 인스턴스 분할 이미지를 기반으로 다음과 같은 통찰력을 얻었습니다. (1) 주석 회사는 널리 사용되는 플랫폼인 MTurk에 비해 수량과 품질 측면에서 모두 더 나은 성능을 보입니다. (2) 주석 회사의 내부 QA는 미미하거나 전혀 개선을 제공하지 않습니다. 그러나 QA에 투자하는 대신 라벨링 지침을 개선하면 주석 성능을 크게 향상시킬 수 있습니다. (3) 내부 QA의 이점은 특정 이미지 특성에 따라 다릅니다. 이 연구는 연구자들이 고정된 주석 예산에서 훨씬 더 많은 가치를 얻고 주석 회사가 내부 QA를 수행하는 방식을 바꿀 수 있도록 합니다.



### S-E Pipeline: A Vision Transformer (ViT) based Resilient Classification Pipeline for Medical Imaging Against Adversarial Attacks (https://arxiv.org/abs/2407.17587)
- **What's New**: 본 논문에서는 의료 영상에서 정확한 질병 진단을 자동화하는 데 널리 사용되는 비전 트랜스포머(ViT)가 적대적 공격에 취약하다는 점에 주목하여, 이러한 공격의 영향을 줄이고 ViT를 더 강력하게 만드는 새로운 이미지 분류 파이프라인인 S-E 파이프라인을 제안합니다. S-E 파이프라인은 여러 전처리 단계를 통해 ViT가 중요한 특징에 집중하도록 훈련시키는 방법을 사용합니다. 특히, CLAHE, UM, HFE와 같은 이미지 향상 기법과 세분화 기술을 사용하여 적대적 공격 후에도 그대로 유지되는 중요한 특징을 식별합니다.



### CityX: Controllable Procedural Content Generation for Unbounded 3D Cities (https://arxiv.org/abs/2407.17572)
Comments:
          5 figures

- **What's New**: 이 논문은 여러 레이아웃 조건(OSM, 의미론적 지도, 위성 이미지 등)을 기반으로 현실적인, 무한한 3D 도시 생성을 가능하게 하는 새로운 멀티모달 제어 가능한 프로시저럴 콘텐츠 생성 방식인 CityX를 제안합니다. CityX는 다양한 PCG 플러그인을 통합하는 일반적인 프로토콜과 명령어를 실행 가능한 Blender 액션으로 변환하는 멀티 에이전트 프레임워크를 제공합니다. CityX는 생성된 자산의 품질과 산업 요구 사항 간의 차이를 줄임으로써 3D 장면 생성을 위한 혁신적인 생태계를 구축할 수 있는 가능성을 보여줍니다.



### Diffusion Models for Multi-Task Generative Modeling (https://arxiv.org/abs/2407.17571)
Comments:
          Published as a conference paper at ICLR 2024

- **What's New**: 본 논문에서는 멀티 모달 데이터를 위한 새로운 확산 모델(Diffusion Model) 프레임워크인 'MT-Diffusion'을 제안합니다. MT-Diffusion은 이미지와 레이블과 같은 다양한 유형의 데이터를 동시에 생성하고, 다중 작업 학습 손실(multi-task learning loss)을 통합하여 멀티 모달 생성 모델링을 일반화합니다.  이는 기존의 단일 모달 생성 모델(single-modal generative model)의 한계를 극복하고, 멀티 모달 데이터를 효과적으로 활용하는 새로운 방법을 제시합니다.



### Learning Instance-Specific Parameters of Black-Box Models Using Differentiable Surrogates (https://arxiv.org/abs/2407.17530)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문은 이미지 잡음 제거를 위한 블랙박스 알고리즘(BM3D)의 매개변수를 학습하기 위한 새로운 방법을 제안합니다. 기존 방법들은 매개변수 공간에서 무작위 샘플링이나 격자 샘플링에 의존하는 반면, 본 논문에서는 입력에 특화된 매개변수를 학습할 수 있는 새로운 방법을 제시합니다.



### StreamTinyNet: video streaming analysis with spatial-temporal TinyML (https://arxiv.org/abs/2407.17524)
Comments:
          this paper has been accepted and presented at the WCCI24 conference

- **What's New**: 이 논문은 TinyML(Tiny Machine Learning) 장치에서 다중 프레임 비디오 스트리밍 분석(VSA, Video Streaming Analysis)을 가능하게 하는 최초의 아키텍처인 StreamTinyNet을 제안합니다. 기존의 TinyML 솔루션들은 제한된 메모리와 연산 능력으로 인해 프레임 단위로 분석을 수행했지만, StreamTinyNet은 여러 프레임을 동시에 분석하여 공간-시간 패턴을 파악할 수 있습니다.

- **Technical Details**: StreamTinyNet은 TinyML 장치에 적합하도록 기존 CNN 아키텍처를 개선하여 메모리 사용량과 연산량을 줄였습니다. 특히, 이 연구에서는 TinyML 장치의 제한된 리소스를 고려하여 메모리 사용량과 연산량을 최소화하는 동시에 정확도를 높이는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, StreamTinyNet은 공개 데이터셋에서 기존 솔루션보다 우수한 성능을 보였으며, 실제 TinyML 장치인 Arduino Nicla Vision에 적용하여 효율성을 입증했습니다. 이는 제스처 인식과 같은 공간-시간 분석이 필요한 다양한 TinyML 애플리케이션에 활용될 수 있는 잠재력을 보여줍니다.



### CORT: Class-Oriented Real-time Tracking for Embedded Systems (https://arxiv.org/abs/2407.17521)
- **What's New**: 이 논문은 복잡한 도시 환경에서 여러 유형의 물체(자동차, 트럭, 자전거, 보행자 등)를 추적하는 데 효과적인 다중 클래스 객체 추적(MOT)에 대한 새로운 접근 방식을 제안합니다. 이 새로운 방식은 Hungarian 매칭 알고리즘의 복잡성을 줄이고, 딥러닝 재식별 모델을 사용하여 더 작은 수의 요소에 대해서만 재식별 단계를 수행합니다. 이를 통해 추적 성능 저하 없이 실행 시간을 줄이고 더 예측 가능한 시간을 달성합니다.



### PatchEX: High-Quality Real-Time Temporal Supersampling through Patch-based Parallel Extrapolation (https://arxiv.org/abs/2407.17501)
- **What's New**: 이 논문은 PatchEX라고 불리는 새로운 프레임 외삽 (extrapolation) 기법을 제안하여 기존 방법보다 품질은 높이고 속도는 빠르게 프레임을 생성합니다. 기존 방법들은 외삽 작업을 하나의 작업으로 수행했지만, PatchEX는 외삽 작업을 여러 하위 작업으로 나누어 병렬 처리하여 속도를 향상시킵니다. 또한, 패치 기반 인페인팅 (inpainting) 및 맞춤형 그림자 예측 (shadow prediction) 방법을 사용하여 생성된 하위 프레임을 결합하여 품질을 유지하면서 지연 시간을 크게 줄입니다.



### ReDiFine: Reusable Diffusion Finetuning for Mitigating Degradation in the Chain of Diffusion (https://arxiv.org/abs/2407.17493)
Comments:
          27 page

- **What's New**: 본 논문에서는 pretrained text-to-image diffusion models을 이용하여 여러 번 fine-tuning을 반복하는 ‘Chain of Diffusion’에서 발생하는 model collapse 문제를 해결하기 위한 새로운 방법인 ‘Reusable Diffusion Finetuning (ReDiFine)’을 제안합니다. ReDiFine은 condition drop finetuning과 CFG scheduling을 결합하여 여러 반복을 거쳐도 이미지 품질이 저하되지 않도록 합니다.



### Robust Adaptation of Foundation Models with Black-Box Visual Prompting (https://arxiv.org/abs/2407.17491)
Comments:
          Extended work from the CVPR'23 paper: arXiv:2303.14773; This paper has been submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) for possible publication

- **What's New**: 본 논문은  pre-trained model(PTM)을 Black-box API로 이용하는 시나리오에서  PTM을 효율적으로 적용할 수 있는 새로운 방법인 'Black-box Visual Prompting (BlackVIP)'를 제안합니다. BlackVIP는 모델 구조 및 파라미터에 대한 정보 없이도 PTM을 적응시킬 수 있습니다. 또한, BlackVIP는 모델 파라미터 접근 없이도 gradient estimation을 위한 대용량 메모리 요구사항을 줄일 수 있습니다.



### Learning from Memory: Non-Parametric Memory Augmented Self-Supervised Learning of Visual Features (https://arxiv.org/abs/2407.17486)
Comments:
          To appear in ICML 2024. Code at this https URL

- **What's New**: 이 논문에서는 과거에 학습한 이미지의 표현을 저장하는 메모리 모듈을 추가하여 self-supervised learning (SSL)의 학습 안정성을 향상시키는 새로운 방법을 제안합니다. 제안된 방법은 신경망에 메모리 모듈을 추가하여 현재 이미지와 이전에 학습한 이미지를 비교하는 방식으로 동작합니다. 또한, stochastic memory blocks를 사용하여 학습 과정을 규제하고 이미지 간의 일관성을 강화합니다.  이 방법은 linear probing, transfer learning, few-shot classification, image retrieval 등 다양한 비전 태스크에서 효과를 보입니다.



### Universal Approximation Theory: The basic theory for deep learning-based computer vision models (https://arxiv.org/abs/2407.17480)
- **What's New**: 이 논문은 컴퓨터 비전 분야에서 널리 사용되는 Convolutional Neural Network (CNN) 및 Transformer 모델을 Universal Approximation Theorem (UAT)의 틀 안에서 통합하여 설명하는 새로운 접근 방식을 제시합니다. UAT를 사용하여 CNN과 Transformer가 이미지 처리에서 어떻게 작동하는지에 대한 기본적인 질문에 대한 이론적 기초를 제공합니다.



### Real-Time Automated donning and doffing detection of PPE based on Yolov4-tiny (https://arxiv.org/abs/2407.17471)
- **What's New**: 이 연구는 MCQ 생성에 대한 새로운 자동 평가 지표인 지식 종속 가능성(KDA)을 제안합니다. 이 지표는 MCQ의 대답 가능성을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 실제 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표로 근사화됩니다. 이러한 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA 및 전문가에 의해 표시된 실제 강의실 환경에서의 사용성과 높은 상관관계를 보입니다. 또한, n-gram 기반 유사성 지표와 결합하면 KDA_disc와 KDA_cont는 다양한 전문가가 표시한 MCQ 품질 측정 지표를 강력하게 예측하는 것으로 나타났습니다.



### CodedVO: Coded Visual Odometry (https://arxiv.org/abs/2407.18240)
Comments:
          7 pages, 4 figures, IEEE ROBOTICS AND AUTOMATION LETTERS

- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 기존의 n-gram 기반 유사도 지표는 교육적 가치를 고려하지 않고, 생성된 MCQ와 골드 샘플의 유사성만 평가했던 반면, KDA는 학생들의 지식 수준을 평가할 수 있는 MCQ의 능력을 측정합니다. KDA는 학생들의 응답을 통해 측정할 수 있으며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 통해 근사화할 수 있습니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 방식을 모방하여 KDA를 근사화합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보였습니다. 또한 n-gram 기반 유사도 지표와 결합하여 전문가가 평가한 다양한 MCQ 품질 측정 기준에 대한 높은 예측력을 보였습니다.



### Quasar-ViT: Hardware-Oriented Quantization-Aware Architecture Search for Vision Transformers (https://arxiv.org/abs/2407.18175)
Comments:
          Accepted by ICS 2024

- **What's New**: 본 논문은 ViT (Vision Transformer)를 위한 하드웨어 지향적인 양자화 인식 아키텍처 검색 프레임워크인 Quasar-ViT를 제안합니다. 이 프레임워크는 하드웨어 구현에 적합한 효율적인 ViT 모델을 설계하여 정확도를 유지하는 동시에 효율성을 높이는 데 중점을 둡니다. Quasar-ViT는 슈퍼넷을 훈련하는 데 사용되는 혁신적인 방법들을 제시하며, FPGA 플랫폼에서의 모델 적응형 설계를 통해 이론적 계산 감소와 실제 추론 속도 향상 간의 차이를 줄입니다.



### Multi-Resolution Histopathology Patch Graphs for Ovarian Cancer Subtyping (https://arxiv.org/abs/2407.18105)
Comments:
          Initially submitted version of a paper which has been accepted in the GRAIL workshop at MICCAI 2024

- **What's New**: 이 논문은 다양한 해상도에서 조직 패치들의 공간적 관계를 이용하여 각 패치의 맥락(context)을 학습하는 다중 해상도 그래프 모델을 사용하여 난소 상피암 아형 분류를 위한 가장 철저한 검증(validation)을 수행합니다. 이 연구는 7개의 모델을 조정하고 Leeds Teaching Hospitals NHS Trust에서 치료받은 434명의 환자로부터 얻은 1864개의 전체 슬라이드 이미지(WSI) 세트에서 5중 교차 검증(five-fold cross-validation)을 사용하여 훈련했습니다. 교차 검증 모델은 앙상블(ensemble) 방식으로 만들어졌으며, 30명의 환자로부터 얻은 100개의 WSI로 구성된 균형(balanced) 홀드아웃 테스트 세트와 Transcanadian Study에서 얻은 80명의 환자로부터 얻은 80개의 WSI로 구성된 외부 검증 세트를 사용하여 평가되었습니다.



### CSWin-UNet: Transformer UNet with Cross-Shaped Windows for Medical Image Segmentation (https://arxiv.org/abs/2407.18070)
- **What's New**: 본 논문은 의료 영상 분할을 위한 새로운 U-모양의 인코더-디코더 네트워크 구조인 CSWin-UNet을 제안합니다. 이 구조는 특히 의료 영상 분할에 맞춤화된 CSWin Transformer 블록을 사용합니다. CSWin 자기 주의 메커니즘은 수평 및 수직 스트라이프 자기 주의 학습을 구현하기 위해 통합되었으며, 이는 각 토큰의 집중 영역을 크게 확장하여 더 포괄적인 분석과 맥락 통합을 가능하게 합니다. 디코더에서는 CARAFE(Content-Aware ReAssembly of FEatures) 레이어가 기존 전치 합성곱 또는 보간 전략을 대신하여 업샘플링에 사용되었으며, 이는 픽셀 수준의 분할 마스크를 더 정확하게 생성합니다.



### LKCell: Efficient Cell Nuclei Instance Segmentation with Large Convolution Kernels (https://arxiv.org/abs/2407.18054)
- **What's New**: LKCell, a high-accuracy and efficient cell segmentation method, utilizes large convolution kernels to achieve a large receptive field while maintaining computational efficiency. This approach significantly reduces the number of parameters and FLOPs compared to previous methods that relied on stacking small convolution kernels or global receptive fields offered by Vision Transformers (ViT).



### YOCO: You Only Calibrate Once for Accurate Extrinsic Parameter in LiDAR-Camera Systems (https://arxiv.org/abs/2407.18043)
Comments:
          IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT

- **What's New**: 본 논문은 LiDAR-카메라 시스템을 위한 새로운 자동 외재 보정(extrinsic calibration) 방법을 제안하며, 기존 방법과 달리 대응점(corresponding points) 매칭이 필요하지 않습니다. 특히, 면 점구름(plane point clouds)의 방향을 계산하고 거리 및 밀도 기반 임계값을 적용하여 관련 없는 점들을 효과적으로 제거하는 새로운 알고리즘을 제시합니다. 또한, 추출된 점들의 투영 과정에 LiDAR와 카메라 간 외재 매개변수를 도입하고 공면 제약 조건(co-planar constraints)을 사용하여 대응점 매칭을 피합니다. 이 매개변수들을 최적화하여 외재 매개변수를 계산합니다.  



### Segmentation-guided MRI reconstruction for meaningfully diverse reconstructions (https://arxiv.org/abs/2407.18026)
Comments:
          Accepted at DGM4MICCAI 2024

- **What's New**: 본 논문에서는 기존의 MCQ 평가 지표들이 교육적 가치를 제대로 반영하지 못한다는 점을 지적하고, 지식 종속 가능성(KDA)이라는 새로운 지표를 제안합니다. KDA는 학생의 지식 수준을 평가하는 MCQ의 능력을 측정하며, 실제 학생들의 응답 데이터를 기반으로 계산됩니다. 또한, KDA를 자동으로 추정할 수 있는 KDA_disc와 KDA_cont라는 두 가지 지표를 제안하고, 이들이 실제 교육 환경에서의 사용성과 높은 상관관계를 보인다는 것을 실험적으로 확인했습니다. 

- **Technical Details**: 본 논문에서는 pre-trained language model을 활용하여 학생들의 문제 해결 방식을 모방함으로써 KDA를 자동으로 추정하는 두 가지 새로운 지표를 제안합니다. KDA_disc는 MCQ의 난이도를 기반으로 KDA를 추정하고, KDA_cont는 MCQ의 내용과 관련된 지식의 연관성을 기반으로 KDA를 추정합니다. 

- **Performance Highlights**: 본 논문의 실험 결과, KDA_disc와 KDA_cont는 실제 전문가들이 평가한 MCQ의 품질 지표와 강한 상관관계를 보였으며, n-gram 기반 유사도 지표와 함께 사용할 경우 MCQ의 품질을 더 정확하게 예측하는 데 효과적인 것으로 나타났습니다. 



### Network Inversion of Convolutional Neural Nets (https://arxiv.org/abs/2407.18002)
- **What's New**: 이 논문은 신경망의 의사 결정 과정을 투명하게 보여주는 네트워크 역전(Network Inversion) 기법을 제시합니다.  기존의 방법들은 네트워크를 역전시키는 데 어려움을 겪거나 생성된 입력의 다양성이 부족했지만, 이 논문에서는 조건화된 생성기를 사용하여 훈련된 신경망의 입력 공간 데이터 분포를 학습하고, 원하는 출력을 내는 입력을 재구성하는 방법을 제안합니다.



### Lightweight Language-driven Grasp Detection using Conditional Consistency Mod (https://arxiv.org/abs/2407.17967)
Comments:
          Accepted at IROS 2024

- **What's New**: 본 논문에서는 교육적 가치를 고려하는 새로운 MCQ 평가 지표인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 해당 사실에 대한 학생의 지식을 평가할 수 있는 능력을 측정합니다. 이를 위해 Human evaluation과 Pre-trained language model을 활용한 KDA_disc, KDA_cont 두 가지 자동 평가 지표를 제안합니다. Human evaluation 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여줍니다. 또한, n-gram 기반 유사성 지표와 함께 사용하면 다양한 전문가 평가 MCQ 품질 측정 지표를 예측할 수 있는 능력이 뛰어남을 보여줍니다.



### Analyzing Brain Tumor Connectomics using Graphs and Persistent Homology (https://arxiv.org/abs/2407.17938)
Comments:
          15 Pages, 7 Figures, 2 Tables, TGI3-MICCAI Workshop

- **What's New**: 본 논문은 뇌 종양 서브타입을 구별하기 위해, Diffusion-weighted MRI (DWI) 를 이용한 전뇌 커넥톰 분석에 지속적 호몰로지와 그래프 이론을 적용했습니다. 기존의 연구들은 주로 종양 분할이나 종양 등급 분류에 초점을 맞췄지만, 본 연구는 종양으로 인한 전뇌 구조적 연결성 변화를 조사하는 데 중점을 두었습니다.



### ReCorD: Reasoning and Correcting Diffusion for HOI Generation (https://arxiv.org/abs/2407.17911)
Comments:
          Accepted by ACM MM 2024. Project website: this https URL

- **What's New**: 이 논문은 HOI(Human-Object Interaction)를 정확하게 묘사하는 이미지 생성을 위한 새로운 학습-없는 방법인 ReCorD를 제안합니다. ReCorD는 Latent Diffusion Model(LDM)과 Visual Language Model(VLM)을 결합하여 HOI 생성 과정을 개선합니다.  특히, 이미지 내용을 이해하는 VLM의 능력을 활용하여  정확한 자세와 개체 배치를  가지고 있는 후보를 선택하고, 상호작용하는  시나리오를  정확하게 이해할 수 있는 상호 작용 인식 추론 모듈을 제안합니다.  또한, 이미지에서 인간의 자세를 유지하면서 개체의 위치를 조정하는 상호 작용 수정 모듈을 도입하여 더욱 정확한 HOI 생성을 수행합니다.  ReCorD는 이미지를 생성하는 동안 인간과 개체 간의 어텐션 맵 겹침을 방지하여 최종 이미지의 충실도를 높입니다.



### Investigating learning-independent abstract reasoning in artificial neural networks (https://arxiv.org/abs/2407.17791)
- **What's New**: 본 논문에서는 교육적 가치를 고려하지 않는 기존 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)의 문제점을 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 지표를 제안한다. KDA는 MCQ의 답변 가능성을 측정하고, 해당 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다.



### HF-Fed: Hierarchical based customized Federated Learning Framework for X-Ray Imaging (https://arxiv.org/abs/2407.17780)
- **What's New**: 본 논문에서는 X-ray 영상에서 개인 맞춤형 FL(Federated Learning)을 위한 계층적 구조 기반 연합 학습(HF-Fed) 방식을 제안합니다. HF-Fed는 X-ray 영상 최적화 문제를 각 병원의 데이터 적응 및 전체적인 X-ray 영상 처리 문제로 나누어 접근합니다. 각 병원마다 특화된 계층적 구조와 네트워크의 네트워크(NoN)라고 불리는 공유된 일반 영상 네트워크를 사용하여 다양한 데이터 분포에서 안정적인 특징을 추출합니다. 계층적 하이퍼 네트워크는 각 병원별 하이퍼 파라미터를 추출하여 NoN을 조건화하여 맞춤형 X-ray 영상 재구성을 수행합니다. HF-Fed는 데이터 공유 없이 X-ray 영상을 향상시키는 유망한 해결책을 제공하며 실험 결과를 통해 경쟁력 있는 성능을 보여줍니다.



### Multi-modal Data Binding for Survival Analysis Modeling with Incomplete Data and Annotations (https://arxiv.org/abs/2407.17726)
Comments:
          Accepted by MICCAI 2024

- **What's New**: 이 논문은 여러 모달리티의 데이터와 함께 censored된 생존 라벨을 처리하는 새로운 생존 분석 프레임워크를 제안합니다. 이 프레임워크는 여러 모달리티를 하나의 공통된 표현 공간에 통합하여, 각 모달리티를 별도로 처리할 수 있게 합니다. 또한 pseudo label과 불확실성을 도입하여 생존 예측의 정확성을 높입니다.



### SV4D: Dynamic 3D Content Generation with Multi-Frame and Multi-View Consistency (https://arxiv.org/abs/2407.17470)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 동적 3D 콘텐츠를 생성하기 위해 멀티프레임(multi-frame) 및 멀티뷰(multi-view) 일관성을 갖춘 잠재적 비디오 확산 모델인 'Stable Video 4D (SV4D)'를 제안합니다. 기존의 비디오 생성 및 새로운 뷰 합성을 위해 별도로 훈련된 생성 모델에 의존하는 방법과 달리, 이 논문은 동적 3D 객체의 새로운 뷰 비디오를 생성하는 통합 확산 모델을 설계합니다.



### CSCPR: Cross-Source-Context Indoor RGB-D Place Recognition (https://arxiv.org/abs/2407.17457)
- **What's New**: 이 논문에서는 RGB-D 실내 장소 인식 (indoor place recognition)을 위한 새로운 알고리즘, 크로스 소스 컨텍스트 장소 인식 (CSCPR)을 소개합니다. CSCPR은 전역 검색과 재순위 지정을 하나의 end-to-end 모델로 통합하여, RGB 영역에만 집중하는 기존 방법과 차별화됩니다. CSCPR은 RGB-D 데이터를 처리하도록 설계되었으며, 잡음이 있는 컬러 포인트 클라우드를 처리하기 위해 CoCs (Context-of-Clusters)를 확장하고,  두 개의 새로운 재순위 지정 모듈, SCC (Self-Context Cluster)와 CSCC (Cross Source Context Cluster)를 소개합니다. SCC와 CSCC는 각각 로컬 특징을 기반으로 특징 표현을 개선하고 쿼리-데이터베이스 쌍을 매칭합니다. 또한 ScanNetIPR과 ARKitIPR이라는 두 개의 새로운 데이터셋을 소개합니다. 실험 결과, CSCPR은 ScanNet-PR 데이터셋에서 Recall@1 기준으로 최소 36.5%, 새로운 데이터셋에서 44% 향상된 성능을 보이며 최첨단 모델보다 뛰어난 성능을 보입니다. 코드와 데이터셋은 공개될 예정입니다.



### $VILA^2$: VILA Augmented VILA (https://arxiv.org/abs/2407.17453)
- **What's New**: 이 논문은 MCQ 생성에 대한 새로운 자동 평가 지표인 지식 종속 가능성(KDA)을 제안한다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 수준을 평가하는 능력을 파악한다. 기존의 n-gram 기반 유사도 지표와 달리, KDA는 학생의 지식에 대한 이해를 기반으로 MCQ의 질을 평가한다. 이 연구는 KDA를 측정하기 위한 두 가지 자동 평가 지표, KDA_disc와 KDA_cont를 제안하며, 이는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방한다.



### AHMF: Adaptive Hybrid-Memory-Fusion Model for Driver Attention Prediction (https://arxiv.org/abs/2407.17442)
- **What's New**: 이 연구는 드라이버의 주의 예측을 위한 새로운 Adaptive Hybrid-Memory-Fusion (AHMF) 모델을 제안하여 사람과 유사한 예측을 달성합니다. 이 모델은 드라이버의 작업 기억(working memory)과 장기 기억(long-term memory)을 활용하여 현장 이해와 경험 회상을 통해 주의 예측을 개선합니다. 특히, 모델은 현재 상황의 특정 위험 자극에 대한 정보를 인코딩하여 작업 기억을 형성하고, 유사한 상황 경험을 장기 기억에서 적응적으로 검색하여 최종 예측을 수행합니다. 또한, 도메인 적응 기술을 사용하여 다양한 데이터셋에서 병렬 훈련을 수행하여 장기 기억 모듈 내에 축적된 운전 경험을 풍부하게 합니다.



### HumanVid: Demystifying Training Data for Camera-controllable Human Image Animation (https://arxiv.org/abs/2407.17438)
Comments:
          camera controllable human image animation, a dataset and a baseline

- **What's New**: 이 논문은 자동으로 MCQ를 생성할 때 기존 평가 지표가 교육적 가치를 고려하지 않고 있다는 문제점을 제기하며, 지식 종속 가능성(KDA)이라는 새로운 평가 지표를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지를 측정합니다. 논문은 또한 KDA를 근사하기 위해 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다.  Human evaluation을 통해 KDA_disc와 KDA_soft가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.  KDA_disc와 KDA_cont는 n-gram 기반 유사성 지표와 결합하여 다양한 전문가가 평가한 MCQ 품질 지표를 예측하는 데 강력한 능력을 보여주었습니다. (This paper addresses the problem of existing evaluation metrics for automatically generated MCQs not considering educational value, proposing a novel evaluation metric called Knowledge Dependent Answerability (KDA). KDA measures how well an MCQ assesses students' knowledge of the target fact. The paper also proposes two automatic evaluation metrics, KDA_disc and KDA_cont, which leverage pre-trained language models to mimic students' problem-solving behavior to approximate KDA. Human evaluations demonstrate a strong correlation between KDA_disc and KDA_soft and usability in real classroom settings. When combined with n-gram based similarity metrics, KDA_disc and KDA_cont exhibit strong predictive power for various expert-labeled MCQ quality measures.)



### Vision Language Model-Empowered Contract Theory for AIGC Task Allocation in Teleoperation (https://arxiv.org/abs/2407.17428)
Comments:
          11 pages, 10 figures

- **What's New**: 이 논문은 야간 원격 조작 (teleoperation) 에서의 저조도 이미지 향상 기술을 개선하기 위해 AI 생성 콘텐츠 (AIGC) 모델을 활용하는 방법을 제안합니다. 특히, AIGC 모델은 계산량이 많기 때문에 풍부한 계산 자원을 가진 에지 서버에 AIGC 작업을 할당해야 합니다. 다양한 크기의 데이터셋으로 훈련된 AIGC 모델의 비용과 AIGC 작업의 서로 다른 수요를 고려하여, 원격 조작자와 에지 서버의 유틸리티를 동시에 최적화하는 차등 가격 책정 전략을 수립하는 것이 중요합니다. 하지만, 가격 책정 전략 수립은 정보 비대칭 (information asymmetry) 아래에서 이루어지며, 즉 AIGC 작업의 수요 (예: AIGC 작업의 난이도 수준 및 분포) 는 에지 서버에 대한 숨겨진 정보입니다. 또한, AIGC 작업의 난이도 수준을 수동으로 평가하는 것은 원격 조작자에게 지루하고 불필요합니다. 이를 위해, 우리는 Vision Language Model (VLM) 을 활용한 계약 이론 (contract theory) 에 의해 지원되는 AIGC 작업 할당 프레임워크를 고안했으며, 이 프레임워크에는 VLM 지원 난이도 평가와 계약 이론 지원 AIGC 작업 할당의 두 가지 구성 요소가 포함됩니다. 첫 번째 구성 요소는 자동적이고 정확한 AIGC 작업 난이도 평가를 가능하게 합니다. 두 번째 구성 요소는 정보 비대칭 하에서 에지 서버에 대한 가격 책정 전략을 수립할 수 있으며, 이를 통해 에지 서버와 원격 조작자 모두의 유틸리티를 최적화합니다. 시뮬레이션 결과에 따르면, 제안된 프레임워크는 원격 조작자와 에지 서버의 평균 유틸리티를 각각 10.88~12.43% 와 1.4~2.17% 개선할 수 있습니다. 코드와 데이터는 이 URL 에서 확인할 수 있습니다.



### On selection of centroids of fuzzy clusters for color classification (https://arxiv.org/abs/2407.17423)
- **What's New**: 본 논문은 퍼지 c-평균(FCM) 알고리즘의 새로운 초기화 방법을 제안하며, 이는 색상 클러스터링 문제에 대한 해결책을 제공합니다. 제안된 초기화 방법은 주어진 색상 포인트 세트에서 가장 선명하고 구별 가능한 색상인 지배적인 색상을 추출합니다. FCM에서 초기 중심으로 선택되는 색상 포인트는 지배적인 색상에 가장 가까운 색상 포인트입니다. 지배적인 색상과 가장 가까운 색상 포인트를 얻기 위해, 참조 색상을 도입하고 색상 포인트와 참조 색상 간의 퍼지 멤버십 모델을 정의합니다.



### 3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities (https://arxiv.org/abs/2407.17418)
- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS)의 다양한 활용과 관련 기술에 대한 포괄적인 분석을 제공하는 최신 서베이 논문이다. 기존 3DGS 관련 연구들을 체계적으로 정리하고 다양한 관점에서 분석하여 3DGS의 발전 현황을 상세히 다루고 있다. 이를 통해 3DGS 연구의 다양한 기술 및 과제를 이해하고 새로운 연구 방향을 제시하는 것을 목표로 한다.



### (PASS) Visual Prompt Locates Good Structure Sparsity through a Recurrent HyperNetwork (https://arxiv.org/abs/2407.17412)
Comments:
          Under review

- **What's New**: 본 논문은 **지식 종속 가능성 (Knowledge Dependent Answerability, KDA)** 라는 새로운 자동 평가 지표를 제안합니다. KDA는 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생 응답을 기반으로 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하여 KDA를 근사합니다. 이 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.



### Generation of Training Data from HD Maps in the Lanelet2 Framework (https://arxiv.org/abs/2407.17409)
- **What's New**: 본 논문에서는 자동화된 주행 시스템에서 널리 사용되는 HD 맵 프레임워크인 Lanelet2에 대한 확장판인 lanelet2_ml_converter를 제안한다. 이 확장판은 단일 소스의 맵 데이터와 포맷에서 맵 기반 자동 주행, 머신 러닝 추론 및 훈련을 통합한다. 이를 통해 맵 기반 자동 주행 시스템에 필요한 표준화된 HD 맵 프레임워크를 제공한다.



### Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising (https://arxiv.org/abs/2407.17399)
- **What's New**: 이 논문은 Gaussian noise 제거에 특화된 딥 신경망이 추가 훈련 없이도 실제 이미지의 잡음 제거에 효율적으로 활용될 수 있음을 보여줍니다. 이를 위해, 저자들은 Variance-Stabilizing Transform (VST)를 활용하는 Noise2VST라는 알고리즘을 제안합니다. Noise2VST는 입력 잡음 이미지와 기존의 Gaussian denoiser만을 사용하여 모델-프리 VST를 학습합니다.



### 3D Question Answering for City Scene Understanding (https://arxiv.org/abs/2407.17398)
- **What's New**: This paper proposes a novel automatic evaluation metric called **Knowledge Dependent Answerability (KDA)** for evaluating Multiple Choice Questions (MCQ) generation. This metric focuses on the MCQ's ability to assess a student's understanding of the target fact, addressing the shortcomings of existing metrics like BLEU, ROUGE, and METEOR, which only consider n-gram similarity.



### PrevPredMap: Exploring Temporal Modeling with Previous Predictions for Online Vectorized HD Map Construction (https://arxiv.org/abs/2407.17378)
- **What's New**: 본 논문에서는 이전 예측을 활용하여 온라인 벡터화 HD 맵을 생성하는 획기적인 시간적 모델링 프레임워크인 PrevPredMap을 소개합니다. PrevPredMap은 이전 예측 기반 쿼리 생성기와 동적 위치 쿼리 디코더라는 두 가지 필수 모듈로 구성되어 있습니다. 이전 예측 기반 쿼리 생성기는 이전 예측에서 다양한 유형의 정보를 별도로 인코딩하도록 설계되었으며, 이는 동적 위치 쿼리 디코더에 의해 효과적으로 활용되어 현재 예측을 생성합니다. 또한, 단일 프레임 모드와 시간적 모드 모두에서 PrevPredMap의 강력한 성능을 보장하기 위해 이중 모드 전략을 개발했습니다.



### ViPer: Visual Personalization of Generative Models via Individual Preference Learning (https://arxiv.org/abs/2407.17365)
Comments:
          Project page at this https URL

- **What's New**: 이 논문은 사용자의 시각적 선호도를 학습하여 개인화된 이미지를 생성하는 새로운 방법을 제안합니다. 기존의 텍스트-이미지 생성 모델들은 다양한 사용자들의 선호도를 만족시키도록 훈련되지만, 개별 사용자의 선호도에 맞춰 이미지를 생성하는 기능은 부족했습니다. 이 연구에서는 사용자가 이미지를 보고 좋아하거나 싫어하는 이유를 설명하는 짧은 댓글을 통해 사용자의 시각적 선호도를 학습하고, 이를 바탕으로 개인화된 이미지를 생성합니다. 이는 사용자가 반복적으로 프로ンプ트를 수정하는 기존 방법보다 효율적이며, 개인화된 이미지 생성에 대한 사용자의 만족도를 높입니다.



### MuST: Multi-Scale Transformers for Surgical Phase Recognition (https://arxiv.org/abs/2407.17361)
- **What's New**: 본 논문은 수술 단계 인식(phase recognition)을 위한 새로운 Transformer 기반 접근 방식인 MuST(Multi-Scale Transformers for Surgical Phase Recognition)를 제안합니다. MuST는 Multi-Term Frame Encoder와 Temporal Consistency Module을 결합하여 수술 영상의 다양한 시간 규모(temporal scales) 정보를 포착합니다. Multi-Term Frame Encoder는 관심 프레임 주변에서 다양한 stride로 시퀀스를 샘플링하여 시간 규모 계층(hierarchy)에서 상호 의존성(interdependencies)을 계산합니다. 또한, 프레임 임베딩에 대한 장기 Transformer 인코더를 사용하여 장기 추론(long-term reasoning)을 강화합니다.  



### Deep Spherical Superpixels (https://arxiv.org/abs/2407.17354)
- **What's New**: 본 논문에서는 기존 MCQ 생성 평가 지표의 한계를 극복하기 위해 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 지표를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다.



### Preliminary study on artificial intelligence methods for cybersecurity threat detection in computer networks based on raw data packets (https://arxiv.org/abs/2407.17339)
Comments:
          Submitted to Computer Science Journal

- **What's New**: 본 논문은 네트워크 트래픽 내에서 원시 패킷 데이터로부터 실시간으로 공격을 감지할 수 있는 딥러닝 기법을 제안합니다. 이는 기존의 트래픽 흐름 특징을 기반으로 하는 방법과 달리 원시 패킷 데이터로부터 직접 특징과 패턴을 추출하는 딥러닝 알고리즘의 잠재력을 활용합니다. 또한, 추가 소프트웨어 구성 요소에 대한 의존성을 제거하고, 실시간 모니터링을 가능하게 합니다. 



### Cascaded Light Propagation Volumes using Spherical Radial Basis Functions (https://arxiv.org/abs/2407.17336)
- **What's New**: 이 논문은 동적 장면에서 간접 조명을 시뮬레이션하는 최신 방법 중 하나인 계단형 광 전파 볼륨 (cascaded light propagation volumes)에 대한 기여를 소개합니다. 본 논문의 기여는 구면 조화 함수 (Spherical Harmonic) 대신 구면 방사 기저 함수 (Spherical Radial Basis Functions)를 사용하는 것입니다. 구면 방사 기저 함수는 많은 계수가 사용될 때 구면 조화 함수보다 훨씬 나은 결과를 얻기 때문입니다. 저자는 계단형 광 전파 볼륨에 구면 방사 기저 함수를 통합하는 방법을 설명하고 동일한 구현 방식이지만 구면 조화 함수를 사용한 경우와 비교하여 기술을 평가합니다.



### Multi-label Cluster Discrimination for Visual Representation Learning (https://arxiv.org/abs/2407.17331)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 연구에서는 CLIP 모델의 시각적 표현 능력을 강화하기 위해 멀티-레이블 클러스터 차별 (MLCD) 방법을 제안합니다. 기존의 CLIP 모델은 이미지-텍스트 대조 학습을 통해 우수한 특징 표현을 제공하지만, 인스턴스 차별 방법은 훈련 데이터의 의미 구조를 효과적으로 인코딩하지 못한다는 한계를 가지고 있습니다. 이러한 한계를 해결하기 위해, MLCD는 이미지에 여러 레이블을 할당하여 더 풍부한 의미 정보를 활용합니다. MLCD는 이미지 클러스터링 단계에서 기존의 임베딩 특징을 사용하여 1백만 개의 클러스터 센터를 생성합니다. 그리고 각 이미지에 여러 개의 가까운 센터를 선택하여 보조 클래스 레이블로 사용합니다. 이러한 멀티-레이블 레이블을 사용하여 훈련 데이터의 의미 구조를 더 잘 파악할 수 있습니다. 또한, MLCD는 새로운 멀티-레이블 분류 손실 함수를 설계하여 양성 클래스와 음성 클래스의 손실을 분리하고 의사 결정 경계의 모호성을 줄입니다. 이러한 방식으로, MLCD는 CLIP 모델의 시각적 표현 능력을 향상시킵니다.



### DarSwin-Unet: Distortion Aware Encoder-Decoder Architectur (https://arxiv.org/abs/2407.17328)
- **What's New**: 본 논문에서는 넓은 시야각을 가진 fisheye 이미지의 왜곡(distortions)을 고려한 새로운 인코더-디코더 모델인 DarSwin-Unet을 제안합니다. 기존 모델들은 넓은 시야각 이미지의 왜곡을 무시하거나 픽셀 단위 작업에 적합하지 않았습니다. DarSwin-Unet은 radial transformer 아키텍처를 기반으로 하여 왜곡된 이미지를 픽셀 단위로 처리할 수 있습니다. 또한 입력 토큰을 생성하기 위해 이미지를 샘플링할 때 스파스성(sparsity)을 최소화하는 새로운 전략을 제안하여 픽셀 단위 작업 성능을 향상시켰습니다.



### Physical Adversarial Attack on Monocular Depth Estimation via Shape-Varying Patches (https://arxiv.org/abs/2407.17312)
- **What's New**: 본 논문은 단안 깊이 추정(MDE) 시스템에 대한 물리 기반 적대적 공격을 제안하며, 이는 ASP(Attack with Shape-Varying Patches)라는 프레임워크를 사용하여 효과를 극대화하기 위해 패치 내용, 모양 및 위치를 최적화하는 것을 목표로 한다.  기존의 패치 기반 적대적 공격은 패치 주변에 제한되어 전체 대상에 영향을 미치기 어려웠지만, 이 논문에서는 다양한 모양의 마스크(사각형, 직사각형, 원형)를 도입하여 공격의 유연성과 효율성을 향상시켰다. 또한 패치 영향을 겹치는 영역을 넘어 확장하기 위해 새로운 손실 함수를 제안한다.



### LangOcc: Self-Supervised Open Vocabulary Occupancy Estimation via Volume Rendering (https://arxiv.org/abs/2407.17310)
- **What's New**: 이 논문에서는 LangOcc라는 새로운 오픈 보캐뷸러리 오큐판시 (occupancy) 추정 접근 방식을 제시합니다. 이 접근 방식은 카메라 이미지만을 사용하여 훈련되며, 비전-언어 정렬을 통해 임의의 의미를 감지할 수 있습니다. 특히, LangOcc는 미분 가능한 볼륨 렌더링을 통해 강력한 비전-언어 정렬 인코더인 CLIP의 지식을 3D 오큐판시 모델로 증류합니다. LangOcc는 이미지만 사용하여 3D 복셀 그리드에서 비전-언어 정렬된 특징을 추정합니다. 또한, 2D 공간으로 추정을 다시 렌더링하여 자기 지도 방식으로 훈련됩니다. 이 훈련 메커니즘은 명시적인 기하학적 감독 없이도 장면 기하학을 자동으로 감독합니다. LangOcc는 오픈 보캐뷸러리 오큐판시에서 LiDAR 감독 경쟁자를 훨씬 능가하며, 비전 기반 훈련에만 의존합니다. 또한, LangOcc는 Occ3D-nuScenes 데이터셋에서 자기 지도 방식의 의미적 오큐판시 추정에서 최첨단 결과를 달성합니다. 즉, 특정 범주 세트에 국한되지 않고, 제안된 비전-언어 훈련의 효과성을 보여줍니다.



### DenseTrack: Drone-based Crowd Tracking via Density-aware Motion-appearance Synergy (https://arxiv.org/abs/2407.17272)
- **What's New**: 본 논문에서는 드론 기반 군중 추적의 정확도를 높이기 위해 밀도 인식 추적(DenseTrack) 프레임워크를 제시합니다. DenseTrack은 군중 수를 계산하여 정확한 객체 위치를 파악하고, 시각 및 운동 신호를 결합하여 소형 객체의 추적을 개선합니다. 특히 프레임 간 이동 문제를 해결하여 추적 정확도와 신뢰성을 높입니다. DenseTrack은 군중 밀도 추정을 사용하여 비디오 프레임 내에서 정확한 객체 위치를 지정합니다. 이러한 추정치는 추적 네트워크의 운동 및 위치 정보와 결합되며, 운동 오프셋은 주요 추적 신호 역할을 합니다. 또한 DenseTrack은 시각-언어 모델의 통찰력을 사용하여 소형 객체를 구분하는 기능을 강화하고, 외관을 운동 신호와 통합합니다. 이 프레임워크는 헝가리 알고리즘을 사용하여 프레임 간 개인의 정확한 매칭을 보장합니다. DroneCrowd 데이터셋에서 보여진 바와 같이, 본 접근 방식은 탁월한 성능을 보여주며, 드론에 의해 캡처된 시나리오에서 효과적임을 확인합니다.



### M4: Multi-Proxy Multi-Gate Mixture of Experts Network for Multiple Instance Learning in Histopathology Image Analysis (https://arxiv.org/abs/2407.17267)
Comments:
          25pages,5figures

- **What's New**: 이 논문은 컴퓨터 병리학에서 전체 슬라이드 이미지(WSI) 분석에 성공적으로 적용된 다중 인스턴스 학습(MIL)을 기반으로, 단일 작업 학습(single-task learning)이 아닌 다중 작업 학습(multi-task learning)을 통해 효율성을 높이고 작업 간의 연관성을 고려한 새로운 모델을 제안합니다. 이는 WSI에서 여러 유전자 돌연변이를 동시에 예측하는 데 사용될 수 있습니다.



### SCIsegV2: A Universal Tool for Segmentation of Intramedullary Lesions in Spinal Cord Injury (https://arxiv.org/abs/2407.17265)
Comments:
          Accepted at MICCAI AMAI 2024 workshop

- **What's New**: 이 논문은  SCI (Spinal Cord Injury)  영상에서 병변 (lesion)을 자동으로 분할 (segmentation) 하고,  tissue bridge를 계산하는 새로운 방법론,  'SCIsegV2'를 제안한다. 이 도구는 7개의 서로 다른 병원에서 수집된 다양한 SCI 단계 (급성, 아급성, 만성) 와 원인 (외상성 SCI, 허혈성 SCI, 퇴행성 경추 척수증) 의 환자 데이터를 사용하여 훈련 및 검증되었다. 이 도구는  expert에 의해 수동으로 계산된  tissue bridge 와 큰 차이가 없어  MRI 바이오마커 (biomarker) 를 자동으로 도출하는 데 사용될 수 있음을 보여준다. SCIsegV2와 자동  tissue bridge 계산은 오픈 소스이며  Spinal Cord Toolbox  (v6.4 이상) 에서  `sct_deepseg -task seg_sc_lesion_t2w_sci`  와  `sct_analyze_lesion`  함수를 통해 사용할 수 있다.



### Embedding-Free Transformer with Inference Spatial Reduction for Efficient Semantic Segmentation (https://arxiv.org/abs/2407.17261)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문은 새로운 멀티플 초이스 질문(MCQ) 생성 평가 지표인 '지식 종속 가능성'(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가할 수 있는 능력을 측정합니다. 기존 지표인 BLEU, ROUGE, METEOR는 MCQ의 교육적 가치를 고려하지 않고 문장 유사성에만 집중했던 반면, KDA는 MCQ의 대답 가능성 (answerability)에 초점을 맞춥니다. KDA는 학생 설문조사 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 수업 환경에서의 사용성과 높은 상관관계를 보였습니다.



### LPGen: Enhancing High-Fidelity Landscape Painting Generation through Diffusion Mod (https://arxiv.org/abs/2407.17229)
- **What's New**: LPGen은 이미지 프롬프트를 확산 모델(diffusion model)에 통합한 새로운 멀티 모달 프레임워크를 도입하여 고품질의 제어 가능한 풍경화 생성 모델입니다. LPGen은 캔니 엣지를 사용하여 이미지에서 엣지와 윤곽을 추출하고, 자연어 텍스트 프롬프트, 그림 스타일 참조와 함께 잠재 확산 모델에 조건으로 제공합니다. LPGen은 이미지와 텍스트 프롬프트 간의 호환성을 보장하는 분리된 교차 주의 전략을 구현하여 멀티 모달 이미지 생성을 가능하게 합니다. 디코더는 최종 이미지를 생성합니다.



### Graph Neural Networks: A suitable Alternative to MLPs in Latent 3D Medical Image Classification? (https://arxiv.org/abs/2407.17219)
Comments:
          Accepted at MICCAI 2024 - GRAIL Workshop

- **What's New**: 본 논문에서는 3차원 의료 영상 분류 작업에서 GNN(Graph Neural Network)이 MLP(Multi-Layer Perceptron) 예측 헤드와 비교하여 얼마나 효과적인지 조사하며, MLP 헤드 대신 GNN을 사용할 수 있는 가능성을 제시합니다. 특히, 3차원 의료 영상 데이터에 대한 제로 샷(Zero-Shot) 환경에서 자연 영상 기반 모델을 강력한 특징 추출기로 활용하는 연구들이 증가하는 가운데, MLP 헤드 대신 GNN을 사용하여 예측 작업을 수행하는 방식을 제안합니다.



### Nonverbal Immediacy Analysis in Education: A Multimodal Computational Mod (https://arxiv.org/abs/2407.17209)
Comments:
          12 pages, 3 figures. Camera-ready version for the SAB 2024: 17th International Conference on the Simulation of Adaptive Behavior

- **What's New**: 이 논문은 교육 환경에서 비언어적 사회적 행동을 분석하기 위한 새로운 컴퓨팅 접근 방식을 소개합니다. 얼굴 표정, 제스처 강도, 공간 역학과 같은 다중 모드 행동 신호를 통합하여 모델은 RGB 교실 비디오에서 교사의 비언어적 즉각성(NVI)을 평가합니다. 독일 교실에서 400개의 30초 비디오 세그먼트 데이터셋이 모델 학습 및 검증을 위해 구성되었습니다. 제스처 강도 회귀 모델은 0.84의 상관관계를, 인식 거리 회귀 모델은 0.55의 상관관계를, NVI 모델은 0.44의 상관관계를 보였습니다. 이 모델은 개별 인간 평가자의 정확도에 근접하여 비언어적 행동 평가에 귀중한 지원을 제공할 수 있는 잠재력을 보여줍니다. 설문 조사 데이터와 훈련된 관찰자 평가 모두에 대해 검증된 모델은 관련 교육 결과와 중간에서 강한 상관관계를 보여주어 효과적인 교수 행동을 반영하는 효과를 보여줍니다. 이 연구는 비언어적 의사 소통 행동의 객관적인 평가를 발전시켜 교육 연구를 위한 새로운 경로를 열어줍니다.



### ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only (https://arxiv.org/abs/2407.17197)
- **What's New**: 이 논문은 3D 객체 탐지 모델을 훈련하는데 필요한 정밀한 3D 어노테이션을 대체하기 위한 새로운 약지도 학습 방법을 제안한다. 이 방법은 이미지의 2D 바운딩 박스 어노테이션과 크기 정보만을 사용하여 3D 객체 탐지를 위한 3D 어노테이션을 생성한다. 2D 바운딩 박스만을 사용하여 3D 탐지 모델을 훈련하는 것은 3D 포즈의 모호성으로 인해 신뢰할 수 없다는 문제를 해결하기 위해, 이 논문은 3D 프록시 객체를 생성하여 훈련 데이터셋에 추가하는 방법을 제안한다. 또한, 2D 감독을 3D 탐지와 더 잘 맞추기 위해 새로운 2D 손실 표현을 사용하여 깊이 불변성(depth invariance)을 보장한다. 마지막으로, 더 어려운 인스턴스를 탐지하기 위해 오프라인 의사 라벨링(pseudo-labeling) 방식을 사용하여 3D 의사 라벨을 점차 개선한다.



### Unpaired Photo-realistic Image Deraining with Energy-informed Diffusion Mod (https://arxiv.org/abs/2407.17193)
- **What's New**: 이 논문은 비(雨)가 낀 사진에서 비를 제거하여 깨끗한 사진을 복원하는 데에 딥러닝 기반 확산 모델(diffusion model)을 적용한 새로운 방법을 제시합니다. 기존의 비제거 모델들은 깨끗한 사진과 비가 낀 사진의 쌍(pair)을 학습 데이터로 사용했지만, 이 논문에서는 쌍 데이터가 없는 상황에서도 효과적인 비제거 모델을 구축하는 것을 목표로 합니다.



### Domain Generalized Recaptured Screen Image Identification Using SWIN Transformer (https://arxiv.org/abs/2407.17170)
Comments:
          11 pages, 10 figures, 9 tables

- **What's New**: 이 논문은 이미지 재방송 및 재캡처(rebroadcast and recapturing) 문제를 해결하기 위해 캐스케이드 데이터 증강과 SWIN 트랜스포머 도메인 일반화 프레임워크(DAST-DG)를 제안한다. 이 프레임워크는 도메인 변화, 특히 도메인 간 및 도메인 간 크기 변동으로 인해 악화되는 상황에서도 잘 작동하도록 설계되었다.



### Context-aware Multi-task Learning for Pedestrian Intent and Trajectory Prediction (https://arxiv.org/abs/2407.17162)
- **What's New**: 본 논문은 자동 MCQ 생성 평가를 위한 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정하는 지표입니다. 기존의 BLEU, ROUGE, METEOR와 같은 메트릭은 MCQ의 교육적 가치를 고려하지 않고 단어의 유사성만 비교했기 때문에 KDA는 교육적 가치를 측정하는 데 더 적합합니다. KDA는 실제 학생들의 답변을 분석하여 계산하며, KDA_disc와 KDA_cont는 각각 KDA를 근사화하기 위한 자동 평가 지표입니다. 이 연구는 human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 MCQ 사용성과 강한 상관관계를 가지고 있음을 보여줍니다. 또한 KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 함께 사용될 때 전문가가 평가한 다양한 MCQ 품질 지표를 예측하는 데 강력한 힘을 발휘합니다.



### Establishing Truly Causal Relationship Between Whole Slide Image Predictions and Diagnostic Evidence Subregions in Deep Learning (https://arxiv.org/abs/2407.17157)
- **What's New**: 이 논문은 '지식 종속 가능성(KDA)'이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생의 응답을 기반으로 측정되며, 이는 기존 n-gram 기반 유사성 메트릭으로는 측정할 수 없는 교육적 가치를 평가할 수 있다는 것을 의미합니다. 또한, KDA를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 새로운 메트릭이 제안되었으며, 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 연구 결과, KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지고 있으며, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가가 평가한 MCQ 품질 척도를 효과적으로 예측할 수 있음이 밝혀졌습니다.



### FIIH: Fully Invertible Image Hiding for Secure and Robus (https://arxiv.org/abs/2407.17155)
- **What's New**: 이 논문은 완전히 역전 가능한 이미지 숨기기 아키텍처를 제안하여, 데이터와 네트워크 모두에서 역전 가능한 이미지 숨기기를 구현하는 것을 목표로 합니다. 이 아키텍처는 딥 러닝 기반 이미지 스테가노그래피 분석을 견딜 수 있습니다. 또한, 전송 중 간섭 후 스테고 이미지의 강력성을 향상시키는 새로운 방법을 제안합니다.



### XMeCap: Meme Caption Generation with Sub-Image Adaptability (https://arxiv.org/abs/2407.17152)
Comments:
          Accepted to MM 2024

- **What's New**: 이 논문은 기존 MCQ 평가 지표인 BLEU, ROUGE, METEOR가 MCQ의 교육적 가치를 고려하지 않고 단순히 골드 샘플과의 유사성만 평가한다는 점을 지적합니다. 따라서, MCQ의 대답 가능성 (answerability)을 측정하여 MCQ가 학생의 지식을 실제로 평가할 수 있는지 여부를 판단하는 새로운 지표, Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생 설문조사 결과를 기반으로 계산되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 KDA를 근사화합니다.



### RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer (https://arxiv.org/abs/2407.17140)
- **What's New**: RT-DETRv2는 실시간 객체 감지 (object detection) 성능을 향상시키는 새로운 트랜스포머 기반 (Transformer-based) 모델입니다. 기존 RT-DETR를 기반으로 하여 유연성과 실용성을 높이는 'Bag-of-Freebies'와 향상된 성능을 위한 최적화된 학습 전략 (training strategy)을 도입했습니다.



### A Self-Supervised Image Registration Approach for Measuring Local Response Patterns in Metastatic Ovarian Cancer (https://arxiv.org/abs/2407.17114)
- **What's New**: 본 논문은 고등장액난소암 (HGSOC) 치료 중 종양 부담 변화를 정량화하기 위한 새로운 자가 지도 변형 이미지 등록 알고리즘을 제안합니다. 이는  neoadjuvant chemotherapy (NACT) 전후에 촬영된 CT 이미지를 공동등록하는데 일반적인 이미지 인코더를 활용합니다.



### PiPa++: Towards Unification of Domain Adaptive Semantic Segmentation via Self-supervised Learning (https://arxiv.org/abs/2407.17101)
Comments:
          This study is under IEEE TMM review. arXiv admin note: substantial text overlap with arXiv:2211.07609

- **What's New**: 이 논문은 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 실제 강의실에서의 사용성과의 강한 상관관계를 보여주는 Human Evaluation을 통해 KDA_disc와 KDA_soft의 효과를 입증했습니다.



### MemBench: Memorized Image Trigger Prompt Dataset for Diffusion Models (https://arxiv.org/abs/2407.17095)
- **What's New**: 본 논문은 텍스트-이미지 생성 모델의 이미지 메모리 현상을 완화하기 위한 새로운 벤치마크인 MemBench를 제안합니다. MemBench는 Stable Diffusion 모델에서 메모리화된 이미지 트리거 프롬프트를 대량으로 제공하며, 이는 기존 연구에서 사용했던 몇몇 샘플을 넘어서기 때문에 더욱 견고한 평가를 가능하게 합니다. 또한 MemBench는 트리거 프롬프트뿐만 아니라 일반 프롬프트에 대한 평가 지표도 제공하여 이미지 메모리 완화 방법이 일반적인 프롬프트에 대한 성능 저하 없이 메모리화 문제를 해결하는지 여부를 확인할 수 있도록 합니다. 이는 기존 연구에서 간과했던 실제적인 응용 분야를 고려할 때 중요한 발전입니다.



### OVR: A Dataset for Open Vocabulary Temporal Repetition Counting in Videos (https://arxiv.org/abs/2407.17085)
- **What's New**: 본 논문은 비디오에서의 반복적인 시간적 패턴을 표시하는 새로운 데이터셋, OVR(Over 발음)을 소개합니다. OVR은 72,000개 이상의 비디오를 포함하며, 각 어노테이션은 반복 횟수, 반복 시작 및 종료 시간, 그리고 반복되는 내용에 대한 자유 형식 설명을 제공합니다. 어노테이션은 Kinetics와 Ego4D에서 가져온 비디오에 대해 제공되며, 따라서 Exo와 Ego 보기 조건을 모두 포함하고 다양한 액션과 활동을 보여줍니다. 또한 OVR은 이전 비디오 반복 데이터셋보다 훨씬 규모가 큽니다. 본 논문에서는 최대 320 프레임 길이의 비디오에서 반복을 찾고 계산할 수 있는 기준 트랜스포머 기반 계산 모델인 OVRCounter를 제안합니다. 모델은 OVR 데이터셋에서 훈련 및 평가되었으며, 텍스트를 사용하여 계산할 대상 클래스를 지정하는 경우와 그렇지 않은 경우 성능이 평가되었습니다. 또한 이전 반복 계산 모델과 비교하여 성능이 평가되었습니다. 데이터셋은 [this https URL](this https URL)에서 다운로드 가능합니다.



### When Text and Images Don't Mix: Bias-Correcting Language-Image Similarity Scores for Anomaly Detection (https://arxiv.org/abs/2407.17083)
- **What's New**: 이 논문은 CLIP (Contrastive Language-Image Pre-training) 모델에서 발생하는 '텍스트 클러스터링 효과'와 '유사성 편향'을 분석하고 이를 해결하기 위한 새로운 방법인 BLISS (Bias-corrected Language Image Similarity Scoring) 를 제안합니다. CLIP은 이미지와 텍스트 입력 임베딩을 정렬하여 다양한 하위 작업에서 뛰어난 성능을 보여주며, 이상 탐지에도 유망합니다. 그러나 본 연구에서는 이미지 임베딩에서 멀리 떨어져 텍스트 입력 임베딩이 예상치 못하게 밀집되어 있는 현상을 발견했습니다. 이는 이미지-텍스트 입력 쌍을 정렬하는 모델의 대조 학습 목표와 배치됩니다. 이러한 현상으로 인해 '유사성 편향'이 발생하며, 이미지와 정상 레이블 텍스트 임베딩 간의 유사성에 대한 편향으로 인해 오탐과 누락 탐지 오류가 발생합니다. BLISS는 보조 외부 텍스트 입력 집합을 사용하여 이러한 유사성 편향을 직접 고려하는 방법입니다. BLISS는 간단하고, 이상 행동에 대한 강력한 귀납적 편향이나 비용이 많이 드는 훈련 과정이 필요하지 않으며, 벤치 마크 이미지 데이터 세트에서 기준 방법을 능가하며, 정상 데이터에 대한 액세스가 매우 제한적인 경우에도 뛰어난 성능을 보입니다.



### AI-based Density Recognition (https://arxiv.org/abs/2407.17064)
- **What's New**: 본 논문은 이미지 기반 물체 인식에 물리적 특성을 할당하는 새로운 AI 기반 개념을 소개합니다. 특히, 이미지에서 물체의 밀도를 추론하여 물체의 무게, 재질, 작용하는 힘 등을 추론하는 것을 목표로 합니다. 이를 통해 밀도 정보를 활용하여 물체의 환경에 대한 영향을 더 정확하게 예측하고, 상황에 맞는 행동을 수행하는 로봇 및 모빌리티 시스템 개발에 기여할 수 있습니다.



### DiffCD: A Symmetric Differentiable Chamfer Distance for Neural Implicit Surface Fitting (https://arxiv.org/abs/2407.17058)
- **What's New**: 본 논문은 Neural implicit surfaces를 이용하여 불완전한 점 구름 (point clouds)에서 정확한 3D 기하학을 복원하는 기술을 연구합니다. 기존 기술들은 단방향 챔퍼 거리 (Chamfer distance) 의 근사값을 최소화하는 방식으로 작동했기 때문에, 점 구름이 표면에 가깝지만 표면이 점 구름에 가까운 것은 보장하지 못했습니다. 이로 인해 기존 방법들은 부정확한 재구성 결과를 만들어낼 수 있습니다. 이러한 문제를 해결하기 위해 본 논문은 DiffCD라는 새로운 손실 함수를 제안하며, 이는 대칭적인 챔퍼 거리에 해당합니다. DiffCD는 기존 기술과 달리, 표면이 점 구름에 가까운 것을 보장하며, 이는 추가적인 정규화 없이 부정확한 표면을 제거할 수 있게 합니다. 본 논문은 실험적으로 DiffCD가 다양한 표면 복잡성과 노이즈 레벨에서 기존 기술보다 훨씬 뛰어난 성능을 보이며, 높은 수준의 형태 디테일을 정확하게 복원한다는 것을 보여줍니다.



### Q-Ground: Image Quality Grounding with Large Multi-modality Models (https://arxiv.org/abs/2407.17035)
Comments:
          ACM Multimedia 2024 (Oral)

- **What's New**: 본 논문에서는  "Q-Ground" 라는 새로운 프레임워크를 소개하여 LMM과 상세한 시각적 품질 분석을 결합하여 fine-scale visual quality grounding을 해결합니다. Q-Ground는 새로운 데이터셋인 QGround-100K를 이용하는데, 해당 데이터셋은 100,000개의 (이미지, 품질 텍스트, 왜곡 세분화) 삼중항 (triplets)을 포함하여 시각적 품질에 대한 심층적인 연구를 가능하게 합니다. QGround-100K는 사람이 레이블을 붙인 정확한 품질 평가용 데이터와 GPT4V와 같은 LMM이 자동으로 레이블을 붙인 데이터로 구성되어 있습니다. 이를 통해 모델 훈련의 안정성을 향상시키면서 데이터 수집 비용을 줄일 수 있습니다. QGround-100K 데이터셋을 기반으로 논문에서는 멀티스케일 특징 학습 (multi-scale feature learning) 기능을 갖춘 LMM 기반 방법을 제안하여 텍스트 프롬프트 (text prompts)를 기반으로 이미지 품질 답변과 왜곡 세분화를 수행할 수 있는 모델을 학습합니다. 이러한 이중 능력 접근 방식은 모델의 영역 인식 이미지 품질 이해를 개선할 뿐만 아니라 이미지 품질과 특정 왜곡에 대한 복잡한 텍스트 기반 질문에 대화식으로 응답할 수 있도록 합니다. Q-Ground는 보다 미세한 규모로 정교한 시각적 품질 분석을 향해 나아가고 있으며, 이 분야의 향후 연구를 위한 새로운 벤치마크를 구축합니다. 코드와 데이터셋은 [링크]에서 이용 가능합니다.



### Enhancing Environmental Monitoring through Multispectral Imaging: The WasteMS Dataset for Semantic Segmentation of Lakeside Was (https://arxiv.org/abs/2407.17028)
- **What's New**: 이 논문은 호숫가 녹지 공간의 폐기물을 구분하기 위한 새로운 멀티 스펙트럼 데이터셋인 WasteMS를 소개합니다. 이 데이터셋은 다양한 조명 조건에서 촬영된 다양한 폐기물 유형을 포함하고 있으며, 정확한 주석 작업을 거쳤습니다. 이를 통해 호숫가 잔디밭에서 폐기물을 분류하는 작업의 어려움과 WasteMS 데이터셋의 활용 가능성을 보여줍니다.



### EAFormer: Scene Text Segmentation with Edge-Aware Transformers (https://arxiv.org/abs/2407.17020)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 Scene Text Segmentation에서 text edge를 활용하여 정확도를 높이는 새로운 모델인 Edge-Aware Transformers (EAFormer)를 제안한다. EAFormer는 text edge를 추출하고 이를 encoder에 활용하여 모델이 text edge에 더 집중하도록 한다. 또한, 기존 COCO_TS 및 MLT_S 데이터셋의 annotation 품질이 좋지 않다는 점을 인지하고, 이를 재-annotation하여 더 정확한 실험 결과를 도출하였다.



### Progressive Query Refinement Framework for Bird's-Eye-View Semantic Segmentation from Surrounding Images (https://arxiv.org/abs/2407.17003)
Comments:
          IROS 2024

- **What's New**: 이 논문은 자율주행 시스템에서 BEV (Bird's-Eye-View) 의미 분할을 위한 새로운 View Transformation (VT) 모델을 제안합니다. 이 모델은 다중 해상도(MR) BEV 쿼리 맵을 사용하여 주행 환경의 전역적 및 국지적 특징을 효과적으로 포착합니다. 이를 위해, 모델은 다양한 해상도의 쿼리 맵을 순차적으로 업데이트하고 병합하여 목표 해상도의 최종 쿼리 맵을 생성합니다. 또한, 이미지 간 및 특징 레벨 간의 특징 상호 작용을 촉진하는 시각적 특징 상호 작용 네트워크를 제안합니다. 해당 모델은 대규모 실제 데이터셋에 대한 실험을 통해 SOTA 모델보다 뛰어난 성능을 보여주었습니다.



### LoFormer: Local Frequency Transformer for Image Deblurring (https://arxiv.org/abs/2407.16993)
- **What's New**: 본 논문에서는 이미지 디블러링 (deblurring) 작업을 위해 Local Frequency Transformer (LoFormer)라는 새로운 방법을 제안합니다. LoFormer는 fine-grained detail을 유지하면서 long-range dependency를 효과적으로 모델링하기 위해 설계되었습니다. 특히, LoFormer는 Frequency domain-Local Channel-wise Self-Attention (Freq-LC)를 사용하여 저주파 (low-frequency) 및 고주파 (high-frequency) 지역 윈도우 내에서의 상관 관계 (cross-covariance)를 동시에 포착합니다. 이를 통해 LoFormer는 1) coarse-grained 구조와 fine-grained detail 모두에 대한 학습 기회를 동등하게 제공하고, 2) coarse-grained global SA 방법에 비해 더 넓은 범위의 표현 특성 (representational properties)을 탐색할 수 있습니다. 또한, LoFormer는 Freq-LC와 보완적인 MLP Gating 메커니즘을 도입하여 관련 없는 특징을 필터링하고 글로벌 학습 기능을 향상시킵니다. 



### DreamCar: Leveraging Car-specific Prior for in-the-wild 3D Car Reconstruction (https://arxiv.org/abs/2407.16988)
Comments:
          Projet Page: this https URL

- **What's New**: 이 논문에서는 자율 주행 데이터셋에서 몇 장의 이미지만으로도 고품질 3D 자동차 모델을 재구성할 수 있는 새로운 방법인 DreamCar를 제안합니다. DreamCar는 기존의 방법들이 가진 몇 가지 한계를 극복하기 위해 다양한 기술을 활용합니다: (1)  자동차의 거울 대칭(mirror symmetry)을 활용하여 데이터를 두 배로 늘리고, (2) Car360이라는 새로운 자동차 데이터셋을 구축하여 기존 생성 모델(generative model)의 자동차에 대한 일반화 성능(generalization)을 향상시키고, (3) 카메라 위치(pose) 오류를 수정하는 새로운 방법을 사용하여 텍스처(texture) 정렬 문제를 해결합니다. 



### Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Mod (https://arxiv.org/abs/2407.16982)
- **What's New**: 본 논문은 텍스트 지시만으로 이미지에 새로운 객체를 추가하는 중요한 문제를 다룹니다. 이 과제는 조명, 질감, 공간적 위치와 같은 일관된 시각적 컨텍스트를 유지하면서 새로운 객체를 이미지에 매끄럽게 통합해야 하기 때문에 어려움을 겪습니다. 기존의 텍스트 기반 이미지 인페인팅 (inpainting) 방법들은 객체를 추가할 수 있지만 배경 일관성을 유지하지 못하거나, 바운딩 박스를 지정하거나 사용자가 스크래치한 마스크를 사용하는 번거로운 인간 개입을 포함합니다. 이러한 과제를 해결하기 위해 본 연구에서는 텍스트 기반 제어만으로 텍스트 기반 객체 추가를 용이하게 하는 Text-to-Image (T2I) 모델인 Diffree를 소개합니다. 이를 위해 본 연구에서는 고급 이미지 인페인팅 기법을 사용하여 객체를 제거한 고급 합성 데이터셋인 OABench를 구축했습니다. OABench는 원본 이미지, 객체가 제거된 인페인팅 이미지, 객체 마스크, 객체 설명의 74,000개의 실제 세계 튜플로 구성됩니다. 추가 마스크 예측 모듈을 갖춘 Stable Diffusion 모델을 사용하여 OABench에서 학습된 Diffree는 새 객체의 위치를 고유하게 예측하고 텍스트만으로 안내를 받아 객체를 추가합니다. 광범위한 실험 결과, Diffree는 배경 일관성, 공간적 적절성, 객체 관련성 및 품질을 유지하면서 높은 성공률로 새로운 객체를 추가하는 데 탁월한 성능을 보여줍니다.



### Case-Enhanced Vision Transformer: Improving Explanations of Image Similarity with a ViT-based Similarity Metric (https://arxiv.org/abs/2407.16981)
- **What's New**: 본 논문은 이미지 데이터의 유사성 평가에 대한 설명 가능성을 향상시키기 위한 새로운 유사성 측정 방법인 케이스 향상 비전 트랜스포머 (CEViT)에 대한 초기 연구를 제시합니다. CEViT는 k-Nearest Neighbor (k-NN) 분류에 통합되어 최첨단 컴퓨터 비전 모델과 비교할 만한 분류 정확도를 달성하면서 클래스 간 차이를 보여주는 기능을 추가합니다. CEViT 설명은 이전 케이스의 영향을 받아 해당 케이스와 관련된 유사성의 측면을 보여줄 수 있습니다.



### Selective Vision-Language Subspace Projection for Few-shot CLIP (https://arxiv.org/abs/2407.16977)
Comments:
          Accepted to ACM MultiMedia 2024

- **What's New**: CLIP (Contrastive Language-Image Pre-training) 모델의 few-shot 학습 성능을 향상시키기 위해, modality gap 문제를 해결하는 새로운 방법인 SSP (Selective Vision-Language Subspace Projection)를 제안합니다. SSP는 local image features를 활용하여 image와 text features 사이의 alignment를 개선합니다.



### Pose Estimation from Camera Images for Underwater Inspection (https://arxiv.org/abs/2407.16961)
Comments:
          Submitted to IEEE Journal of Oceanic Engineering

- **What's New**: 본 논문은 수중 재검사 임무에서의 정밀한 위치 추정을 위한 딥러닝 기반 시각적 위치 추정(visual localization) 방법론을 제안합니다. 이 방법론은 기존의 비용이 많이 드는 방법(관성항법시스템, 도플러 속도 기록기, 음향 위치 시스템) 대신 이미 장착된 카메라를 활용하여 주변 환경의 이미지에서 위치를 추정합니다. 특히, 기존에 매핑된 장면을 기반으로 훈련된 모델을 사용하여 효율적인 재위치 지정 (relocalization) 을 수행하는 딥러닝 기반 위치 추정 방법론을 수중 환경에 적용하는 효과를 연구합니다. 또한, 이 논문은 훈련 데이터를 증강하기 위해 새로운 뷰 합성 (Novel View Synthesis) 모델을 도입하여 미탐험 지역에서의 위치 추정 성능을 향상시키고, 위치 추정기 출력을 센서 데이터와 결합하여 위치 추정 정확도를 향상시키는 방법을 제안합니다. 



### Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Remova (https://arxiv.org/abs/2407.16957)
Comments:
          Accepted to ECCV2024, dataset and benchmark at: \url{this https URL}

- **What's New**: 본 논문에서는 기존 빗방울 제거 데이터셋의 한계를 극복한 새로운 대규모 실제 데이터셋인 Raindrop Clarity를 소개합니다. Raindrop Clarity는 15,186개의 고품질 이미지 쌍/삼중 세트(빗방울, 흐릿함, 배경)로 구성되어 있으며, 빗방울이 있는 이미지와 해당 깨끗한 배경 이미지를 포함합니다. 특히, Raindrop Clarity는 주간 및 야간 빗방울 시나리오를 모두 포함하며, 배경에 초점을 맞춘 이미지와 빗방울에 초점을 맞춘 이미지를 모두 포함하여 빗방울 제거 연구에 새로운 가능성을 제공합니다.  



### DVPE: Divided View Position Embedding for Multi-View 3D Object Detection (https://arxiv.org/abs/2407.16955)
- **What's New**: 이 논문은 3D 객체 탐지에 적용 가능한 새로운 프레임워크인 DVPE(Divided View Position Embedding)를 제안합니다. DVPE는 3D 공간을 여러 개의 가상 공간으로 분할하고 각 공간 내에서 가시성 크로스 어텐션을 수행하여 비 관련 기능의 간섭을 줄이고 카메라 포즈와 위치에서 위치 임베딩을 분리하여 학습 난이도를 완화합니다. 또한 2D 시각 정보가 강화된 객체 중심 시간 모델링 방식을 통해 3D 기능을 강화하고, 3D 객체 탐지 프레임워크 훈련에 1:N 할당 전략을 적용하여 디코더의 훈련 안정성을 향상시킵니다.



### Open Challenges on Fairness of Artificial Intelligence in Medical Imaging Applications (https://arxiv.org/abs/2407.16953)
Comments:
          Published as part of the book "Trustworthy AI in Medical Imaging" (Elsevier, 2024) available at this https URL

- **What's New**: 이 논문은 의료 영상 분석에서 AI의 공정성(fairness) 문제를 다루며, 데이터 수집, 모델 훈련, 임상 배포 단계에서 발생할 수 있는 다양한 편향(bias)의 원인과 영향을 분석합니다. 또한, 편향된 평가 지표(metrics) 사용, 수준 하락 효과(leveling down effect), 하위 집단 간 작업 난이도 차이, 미지의 모집단에서 편향 발견, 표준 인구통계적 속성을 넘어 편향 설명 등의 주제를 다루며 연구자와 실무자들이 주목해야 할 핵심 과제들을 제시합니다.



### Affective Behaviour Analysis via Progressive Learning (https://arxiv.org/abs/2407.16945)
Comments:
          Techical Report for 7th ABAW Competition

- **What's New**: 이 논문에서는 교사의 시간을 절약하기 위해 자동으로 MCQ를 생성하는 방법을 연구하고 있습니다. 기존의 평가 지표들은 MCQ의 질을 제대로 평가하지 못한다는 문제점이 있었는데, 이 논문에서는 새로운 지표인 KDA를 제안하여 MCQ의 답변 가능성과 학생의 지식 수준을 평가하는 능력을 측정합니다. KDA는 학생들이 실제로 답변한 데이터를 바탕으로 측정되며, KDA_disc와 KDA_cont라는 자동 평가 지표도 함께 제안합니다. 실제 교육 환경에서 KDA_disc와 KDA_cont는 전문가에 의해 평가된 사용성과 높은 상관관계를 보였습니다. 또한, 기존 지표와 결합했을 때 MCQ의 질을 예측하는 능력이 뛰어났습니다.



### McGAN: Generating Manufacturable Designs by Embedding Manufacturing Rules into Conditional Generative Adversarial Network (https://arxiv.org/abs/2407.16943)
- **What's New**: 본 논문에서는 MCQ 생성 시스템의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭인 'Knowledge Dependent Answerability (KDA)'를 제안합니다. KDA는 생성된 MCQ가 대상 사실에 대한 학생의 지식을 실제로 평가할 수 있는지 여부를 측정합니다. KDA는 학생 설문 조사를 통해 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되어 pre-trained language model을 활용하여 학생들의 문제 해결 행동을 모방합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 강의실 상황에서의 사용성과 높은 상관관계를 보이며, n-gram 기반 유사성 메트릭과 함께 사용하면 다양한 전문가 평가 MCQ 품질 측정 지표를 예측하는 데 뛰어난 능력을 보여줍니다.  



### SAR to Optical Image Translation with Color Supervised Diffusion Mod (https://arxiv.org/abs/2407.16921)
- **What's New**: 이 논문은 Synthetic Aperture Radar (SAR) 이미지를 더 이해하기 쉽도록 optical 이미지로 변환하는 새로운 생성 모델을 제안합니다. 이 모델은 최신 diffusion model을 기반으로 하며, SAR 이미지를 조건으로 활용하여 샘플링 과정을 진행합니다. 또한, 색상 변화 문제를 해결하기 위해 색상 supervision을 통합합니다.



### PathwayBench: Assessing Routability of Pedestrian Pathway Networks Inferred from Multi-City Imagery (https://arxiv.org/abs/2407.16875)
Comments:
          arXiv admin note: text overlap with arXiv:2303.02323

- **What's New**: 본 논문은 보행자 경로 그래프 추출 문제를 위한 첫 번째 표준 벤치마크를 제시합니다. 이 벤치마크는 8개 도시의 3,000 km^2 면적을 커버하는 수동으로 검증된 그라운드 트루스 주석과 함께 제공되는 가장 큰 데이터셋을 포함하며, 이동성 애플리케이션의 유용성에 중점을 둔 평가 지표 모음을 제공합니다. 이 벤치마크는 개별 교차로 규모의 다각형으로 데이터를 분할하여 지역적 이동 가능성을 효율적인 전역 이동 가능성 대리자로 계산합니다. 이러한 지표를 사용하여 본 벤치마크가 이전 작업에서 사용된 단일 영역 데이터셋에 대한 간단한 에지 계산 지표로 숨겨진 기존 방법의 강점과 약점을 드러낼 수 있음을 보여줍니다. 즉, 컴퓨터 비전 및 기계 학습에서 어려운 고 영향력 문제를 나타냅니다. (**routability**, **global routability**, **local routability**, **ground truth annotation**).



### A Multi-Level Hierarchical Framework for the Classification of Weather Conditions and Hazard Prediction (https://arxiv.org/abs/2407.16834)
Comments:
          6 pages

- **What's New**: 이 논문은 날씨 상태와 위험 예측을 위한 다층적 계층적 프레임워크를 제시합니다. 이 프레임워크는 이미지 데이터를 사용하여 11가지 특정 유형의 날씨 이미지를 분류하고 실시간 날씨 정보를 제공합니다. 특히, 이 프레임워크는 전통적인 날씨 예보가 부정확한 상황, 예를 들어 위험한 날씨에서 자율 주행 자동차의 안전 운행을 보장하는 데 유용합니다. 

- **Technical Details**: 이 논문은 이미지 데이터를 사용하여 11가지 날씨 이미지(이슬, 서리, 결빙, 서리, 눈, 우박, 비, 번개, 무지개, 모래 폭풍)를 분류하고 실시간 날씨 정보를 제공하는 다층적 계층적 프레임워크를 제시합니다. 이 프레임워크는 이미지를 11가지 날씨 범주로 분류하는 능력을 갖추고 있으며, 0.9329의 정확도로 실시간 날씨 정보를 제공합니다. 

- **Performance Highlights**: 이 프레임워크는 11가지 날씨 범주로 이미지를 분류하고 실시간 날씨 정보를 제공하며, 0.9329의 정확도를 달성했습니다. 



### SINDER: Repairing the Singular Defects of DINOv2 (https://arxiv.org/abs/2407.16826)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 비전 트랜스포머 모델에서 나타나는 "Singular Defect" 라는 새로운 현상을 분석하고, 이를 완화하기 위한 새로운 fine-tuning 방법을 제안합니다. 특히 기존 방법들은 모델 전체를 재학습하는데 반해, 이 논문에서는 작은 데이터셋만 사용하여 효율적으로 모델을 개선할 수 있습니다.



### AI-Enhanced 7-Point Checklist for Melanoma Detection Using Clinical Knowledge Graphs and Data-Driven Quantification (https://arxiv.org/abs/2407.16822)
- **What's New**: 이 논문은 7점 체크리스트 (7PCL) 을 이용한 악성 흑색종 진단의 정확성을 높이기 위해 새로운 진단 방법을 제안합니다. 7PCL은 7가지 특징 (attribute) 에 점수를 부여하는데, 주요 특징은 각각 2점, 부차적 특징은 각각 1점입니다. 총 3점 이상이면 생검 등 추가적인 검사를 진행합니다. 기존 방법은 모든 특징에 같은 가중치를 부여하여 정확성이 떨어지고 특징 간의 연관성을 무시한다는 한계점이 있습니다. 또한, 기존 딥러닝 연구는 악성 흑색종을 예측하는 것과 특징을 예측하는 것을 동일하게 중요하게 다루어 악성 흑색종 진단에 필요한 특징의 중요성을 간과했습니다. 본 논문에서는 이러한 한계점을 해결하기 위해 Clinical Knowledge-Based Topological Graph (CKTG) 와 Gradient Diagnostic Strategy with Data-Driven Weighting Standards (GD-DDW) 두 가지 혁신적인 요소를 통합한 새로운 진단 방법을 제안합니다.



### Fusion and Cross-Modal Transfer for Zero-Shot Human Action Recognition (https://arxiv.org/abs/2407.16803)
- **What's New**: 이 논문은 인간의 움직임과 행동을 이해하기 위해 시각과 관성 센서 데이터 간의 지식 전이 (cross-modal transfer) 를 위한 새로운 방법, FACT (Fusion and Cross-modal Transfer) 를 제안합니다. FACT는 시각 데이터로 학습된 모델이 관성 센서 데이터만으로 인간 행동을 인식 (Human Action Recognition, HAR) 할 수 있도록 합니다. 특히, 기존의 cross-modal transfer 학습 방식과 달리, FACT는 학습 중에 관성 센서 데이터에 대한 라벨을 필요로 하지 않고, 테스트 시에만 관성 센서 데이터를 사용합니다. 이는 zero-shot cross-modal transfer 학습을 가능하게 합니다. 또한, FACT는 시간 정보를 효과적으로 처리하기 위해 시간 연속적인 버전인 T-FACT도 제시합니다.



### Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels (https://arxiv.org/abs/2407.16802)
- **What's New**: 본 논문은 기존 MCQ 평가 지표들이 교육적 가치를 제대로 반영하지 못한다는 문제점을 지적하며 새로운 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지를 측정합니다. 또한, KDA를 자동화하기 위한 KDA_disc와 KDA_cont 두 가지 지표를 제안합니다. 이러한 지표들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 높은 상관관계를 가지고 있음을 보여주었습니다. 더 나아가, KDA_disc와 KDA_cont는 n-gram 기반 유사도 지표와 함께 사용될 때, 전문가가 평가한 다양한 MCQ 품질 척도를 예측하는 데 강력한 힘을 가지고 있습니다.



### What Matters in Range View 3D Object Detection (https://arxiv.org/abs/2407.16789)
- **What's New**: 본 논문은 Lidar 기반의 3D 물체 탐지 모델에서 Range-view 표현을 사용하여 최첨단 성능을 달성한 새로운 모델을 제안합니다. 기존 연구에서 제안된 다양한 기법 없이도 최첨단 성능을 달성했으며, 이는 Range-view 3D 물체 탐지 모델의 중요한 진전입니다. 또한 Argoverse 2와 Waymo Open 데이터셋에서의 실험을 통해 핵심적인 통찰력을 얻었으며, 이는 Range-view 3D 물체 탐지 모델의 설계 및 성능 향상에 중요한 지침을 제공합니다. 특히, Input feature dimensionality의 중요성, 3D 공간 근접성 기반 분류 손실의 효용성, Range subsampling 기법의 효과성 등을 강조하며 기존 연구에서 제안된 복잡한 기법 없이도 최고의 성능을 달성할 수 있음을 보여줍니다. 본 논문은 Range-view 3D 물체 탐지 모델의 새로운 기준을 제시하며, 자율 주행 분야의 발전에 기여할 것으로 예상됩니다.



### Occlusion-Aware 3D Motion Interpretation for Abnormal Behavior Detection (https://arxiv.org/abs/2407.16788)
- **What's New**: 본 논문에서는 3D 포즈 기반 이상 자세 추정을 위한 새로운 방법인 OAD2D를 제안합니다. 이 방법은 단안 비디오에서 메쉬 버텍스와 인간 관절의 3D 좌표를 재구성하여 이상 동작을 구별합니다. OAD2D는 흐름(optical flow)을 활용하여 비디오 스트림에서 동작 선행 정보를 포착하고, 폐색된 인간 움직임에 대한 정보를 풍부하게 하여 포즈의 시간적-공간적 정렬을 보장합니다.



### A Dataset for Crucial Object Recognition in Blind and Low-Vision Individuals' Navigation (https://arxiv.org/abs/2407.16777)
Comments:
          16 pages, 4 figures

- **What's New**: 이 논문에서는 시각장애인과 저시력자(BLV)의 탐색 작업을 지원하기 위해 실시간 객체 인식 시스템을 개선하기 위한 데이터셋을 소개합니다. 이 데이터셋은 BLV 개인이 야외 공간을 탐색하는 21개의 비디오와 포커스 그룹 연구를 통해 개선된 BLV 탐색에 중요한 90개의 객체 분류를 포함합니다. 또한 21개 비디오에서 생성된 31개 비디오 세그먼트에 걸쳐 90개 객체에 대한 객체 레이블을 제공합니다. 심층 분석 결과, 컴퓨터 비전 모델 훈련에 사용되는 대부분의 최신 데이터셋에는 데이터셋의 분류 체계 중 일부만 포함되어 있습니다. 데이터셋에 대한 최첨단 컴퓨터 비전 모델의 예비 평가는 BLV 탐색과 관련된 핵심 객체를 정확하게 감지하는 데 있어서의 단점을 강조하며 특수 데이터셋의 필요성을 강조합니다. 저희는 데이터셋을 공개적으로 제공하여 BLV 개인을 위한 보다 포괄적인 탐색 시스템을 개발하는 데 귀중한 리소스를 제공합니다.  



### A study of animal action segmentation algorithms across supervised, unsupervised, and semi-supervised learning paradigms (https://arxiv.org/abs/2407.16727)
Comments:
          33 pages, 15 figures

- **What's New**: 본 연구는 기존의 MCQ 생성 평가 지표가 교육적 가치를 고려하지 않는 문제를 해결하기 위해, '지식 종속 가능성(KDA)'이라는 새로운 평가 지표를 제안합니다. KDA는 MCQ의 답변 가능성(answerability)을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 또한, KDA를 근사화하기 위해, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 'KDA_disc'와 'KDA_cont'라는 두 가지 자동 평가 지표를 제안합니다. 이러한 지표들은 실제 강의실 환경에서의 사용성과 강한 상관관계를 보이며, 기존의 n-gram 기반 유사성 지표와 함께 사용될 경우 다양한 전문가가 평가한 MCQ 품질 지표에 대한 예측력이 뛰어납니다.



### Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions (https://arxiv.org/abs/2407.16725)
Comments:
          Accepted by 37th Conference on Neural Information Processing Systems (NeurIPS 2023)

- **What's New**: 이 논문은 vision-language 모델(CLIP)을 이용하여 OOD(Out-of-Distribution) 탐지를 위한 새로운 프레임워크를 제안하며, 특히 unseen categories에 대한 정확한 범주 설명을 위해 두 가지 계층적 맥락(perceptual context, spurious context)을 사용한다. Perceptual context는 현재 분류 작업에서 범주 간 차이(예: 고양이 vs 사과)를 인식하고, spurious context는 각 범주에 대해 잘못된 (유사하지만 정확히 일치하지 않는) OOD 샘플(예: 고양이 vs 팬더, 사과 vs 복숭아)을 추가로 식별한다.



### SoNIC: Safe Social Navigation with Adaptive Conformal Inference and Constrained Reinforcement Learning (https://arxiv.org/abs/2407.17460)
Comments:
          Project website: this https URL

- **What's New**: 본 논문에서는 MCQ 생성의 교육적 가치를 평가하는 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 학생의 문제 해결 능력을 모방한 사전 훈련된 언어 모델을 사용하여 KDA를 근사화합니다. 인간 연구를 통해 KDA_disc와 KDA_soft가 KDA와 전문가가 평가한 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였다는 것을 보여줍니다. 또한, n-gram 기반 유사성 메트릭과 결합하면 KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 측정에 대한 강력한 예측력을 보이는 것으로 나타났습니다.



### Looking at Model Debiasing through the Lens of Anomaly Detection (https://arxiv.org/abs/2407.17449)
Comments:
          15 pages, 7 figures

- **What's New**: 본 논문에서는 모델 바이어스를 탐지하고 완화하는 새로운 접근 방식을 제안한다. 이 접근 방식은 바이어스가 있는 모델의 특징 공간에서 바이어스가 없는 샘플을 이상치로 간주하여 바이어스가 있는 샘플과 바이어스가 없는 샘플을 구분하는 이상 탐지 방법을 사용한다.



### 2D and 3D Deep Learning Models for MRI-based Parkinson's Disease Classification: A Comparative Analysis of Convolutional Kolmogorov-Arnold Networks, Convolutional Neural Networks, and Graph Convolutional Networks (https://arxiv.org/abs/2407.17380)
Comments:
          19 Pages, 5 figures

- **What's New**: 본 논문은 파킨슨병 (PD) 진단을 위한 MRI 기반 딥러닝 아키텍처를 비교 분석하여 컨볼루션 레이어와 적응형 스플라인 기반 활성화 함수를 결합한 새로운 접근 방식인 컨볼루션 콜모고로프-아놀드 네트워크 (ConvKANs)의 최초 3차원 (3D) 구현을 소개합니다. Convolutional Neural Networks (CNNs), ConvKANs, Graph Convolutional Networks (GCNs)를 총 142명의 참가자 (PD 75명, 나이가 일치하는 건강한 대조군 67명)를 포함하는 3개의 오픈 소스 데이터셋을 사용하여 평가했습니다. 2D 분석의 경우 각 T1 가중 스캔에서 중뇌를 중심으로 100개의 축면 슬라이스를 추출했습니다. 3D 분석의 경우 전체 볼륨 스캔을 사용했습니다. ConvKANs는 학습 가능한 B-스플라인 함수를 컨볼루션 레이어와 통합합니다. GCNs는 MRI 데이터를 그래프로 표현하여 기존 접근 방식에서 간과할 수 있는 구조적 관계를 이론적으로 포착합니다. 최초의 ConvKAN 스플라인 활성화 맵과 그래프 노드 임베딩의 투영을 포함한 해석 가능성 시각화가 묘사되었습니다. ConvKANs는 데이터셋과 차원에 걸쳐 높은 성능을 보여주었으며, 한 데이터셋에서 최고의 2D AUROC (0.98)를 달성하고 CNN 피크 3D 성능 (1.00)과 일치했습니다. CNN 모델은 잘 수행되었지만 GCN 모델은 3D 분석에서 최대 0.97 AUROC에 도달하여 개선되었습니다. 3D 구현은 모든 모델에서 2D와 비교하여 더 높은 AUROC 값을 산출했습니다. ConvKAN 구현은 특히 조기 진단 측면에서 PD 분류에서 MRI 분석에 대한 가능성을 보여줍니다. 3D 분석의 개선은 미세한 PD 관련 변화를 포착하는 데 있어 볼륨 데이터의 가치를 강조합니다. MRI는 현재 PD 진단에 사용되지 않지만 이러한 결과는 특히 조기 발견을 위해 다중 모드 진단 접근 방식의 구성 요소로서의 잠재력을 시사합니다.



### Enhanced Deep Learning Methodologies and MRI Selection Techniques for Dementia Diagnosis in the Elderly Population (https://arxiv.org/abs/2407.17324)
- **What's New**: 이 논문은 MRI 슬라이스를 선택적으로 처리하여 뇌의 가장 중요한 영역에 집중하고 정보가 적은 부분을 제외하는 새로운 방법을 소개합니다. 이 방법은 세 가지 맞춤형 딥 러닝 모델(Dem3D ResNet, Dem3D CNN, Dem3D EfficientNet)로 구성된 신뢰 기반 분류 위원회를 통해 보완됩니다. 이러한 모델은 각 모델의 강점을 활용하여 의사 결정 정확도를 높이기 위해 협력적으로 작동합니다.



### Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation (https://arxiv.org/abs/2407.17274)
Comments:
          Work in progress

- **What's New**: 본 논문에서는 Text-to-Image Retrieval(텍스트-이미지 검색)을 위한 새로운 Generative Cross-Modal Retrieval(생성형 교차 모달 검색) 방법인 AVG(Autoregressive Voken Generation)를 제안한다. AVG는 이미지를 'voken'(시각적 토큰)으로 토큰화하여 텍스트-이미지 검색을 토큰-보켄 생성 문제로 바꾼다. 이는 기존의 이미지 ID와 같은 단순한 문자열 식별자 대신 시각적 정보와 의미를 담은 vokens를 사용하여 더 정확한 검색을 가능하게 한다.



### Trans2Unet: Neural fusion for Nuclei Semantic Segmentation (https://arxiv.org/abs/2407.17181)
Comments:
          ICCAIS 2022

- **What's New**: 본 논문은 Unet과 TransUnet을 결합한 새로운 두 가지 분기(branch) 아키텍처를 제안하여 핵 분할(nuclei segmentation) 과제를 해결합니다. 이 아키텍처는 Trans2Unet이라고 불리며, 입력 이미지는 먼저 마지막 합성곱 레이어(convolution layer)가 제거된 Unet 분기에 전달됩니다. 이 분기는 네트워크가 입력 이미지의 서로 다른 공간 영역에서 특징을 결합하여 관심 영역을 더 정확하게 찾아낼 수 있도록 합니다. 또한 입력 이미지는 두 번째 분기에도 전달됩니다. 이 분기는 TransUnet 분기라고 불리며, 입력 이미지는 이미지 패치로 나뉩니다. 비전 트랜스포머(Vision transformer, ViT)를 아키텍처에 적용하여, TransUnet은 의료 이미지 분할 작업을 위한 강력한 인코더 역할을 수행하고 국소화된 공간 정보를 복원하여 이미지 세부 정보를 향상시킵니다. Trans2Unet의 효율성과 성능을 높이기 위해, 논문에서는 "워터폴"(Waterfall) 아트루스 공간 풀링(Atrous Spatial Pooling, WASP) 모듈에서 영감을 얻어 "워터폴" 아트루스 공간 풀링 with 스킵 연결(Skip Connection, WASP-KC) 모듈을 통합합니다. 2018 Data Science Bowl 벤치마크에 대한 실험 결과는 이전 분할 모델과 비교하여 제안된 아키텍처의 효율성과 성능을 보여줍니다.



### Toward an Integrated Decision Making Framework for Optimized Stroke Diagnosis with DSA and Treatment under Uncertainty (https://arxiv.org/abs/2407.16962)
- **What's New**: 본 논문에서는 불확실성 속에서 뇌졸중 진단 및 치료의 어려움을 해결하기 위한 새로운 접근 방식을 제시합니다. 이는 동맥류(aneurysm), 뇌동맥 기형(AVM), 혈관 폐쇄(occlusion)와 같은 뇌졸중 조건의 빠른 진행과 심각한 결과를 고려할 때 매우 중요한 문제입니다. 디지털 혈관 조영술(DSA)과 같은 기존 진단 방법은 비용이 많이 들고 침습적이라는 단점이 있습니다. 이러한 문제를 해결하기 위해, 부분 관측 마르코프 결정 프로세스(POMDP) 프레임워크를 사용하는 새로운 접근 방식을 제안합니다. 본 모델은 고급 진단 도구와 치료 접근 방식을 뇌졸중 진단의 고유한 불확실성을 고려하는 의사 결정 알고리즘과 통합합니다. 본 접근 방식은 CT 스캔, Siriraj 점수, DSA 보고서의 노이즈가 많은 관찰 결과를 결합하여 후속 치료 옵션을 알립니다. 트리 검색 방법과 입자 필터를 사용하는 온라인 솔버 DESPOT을 활용하여 잠재적인 미래 시나리오를 시뮬레이션하고 전략을 안내합니다. 결과는 POMDP 프레임워크가 진단 및 치료 목표 간의 균형을 맞추고, DSA와 같은 침습적 절차를 통한 정확한 뇌졸중 식별의 필요성과 병원 내 또는 가정 관찰과 같은 비용 효율적인 전략의 제약(의료 자원 제한) 간의 절충을 이룬다는 것을 나타냅니다. 본 연구는 뇌졸중에 대한 진단 및 치료 프로세스를 최적으로 통합하고 다양한 불확실성을 고려하여 뇌졸중 관리의 치료 및 결과를 개선하는 체계적인 프레임워크를 제시함으로써 중요한 기여를 합니다.



### Vision-Based Adaptive Robotics for Autonomous Surface Crack Repair (https://arxiv.org/abs/2407.16874)
Comments:
          20 pages, 13 figures, submitted to Automation in Construction

- **What's New**: 본 논문에서는 로봇을 이용한 표면 균열 탐지 및 수리 시스템을 제안하며, 이는 RGB-D 카메라, 레이저 스캐너, 압출기 및 펌프를 활용하여 자율적으로 균열을 감지하고 수리합니다. 또한 3D 프린팅된 균열 시편으로 실제 균열을 모방하여 반복성 있는 검증 절차를 수행합니다. 본 연구는 적응형 시스템이 고정 속도 방식보다 더 효율적이고 효과적임을 보여주는 실험 결과를 통해 정밀성과 일관성을 입증합니다. 이 연구는 다재다능하고 신뢰할 수 있는 로봇 인프라 유지 보수를 위한 길을 열어줍니다. 



### PlantTrack: Task-Driven Plant Keypoint Tracking with Zero-Shot Sim2Real Transfer (https://arxiv.org/abs/2407.16829)
- **What's New**: 본 논문은 MCQ (Multiple Choice Questions) 자동 생성에 대한 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안하며, MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다. 또한, KDA_disc와 KDA_cont 두 가지 자동 평가 지표를 제안하여 KDA를 근사화하고 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.



### Exploring The Neural Burden In Pruned Models: An Insight Inspired By Neuroscienc (https://arxiv.org/abs/2407.16716)
- **What's New**: 이 논문은 교육적 가치를 고려한 새로운 자동 MCQ 평가 지표, Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 대상 사실에 대한 학생의 지식 수준을 평가하는 MCQ의 능력을 측정하는 지표입니다. 이 논문은 KDA를 Human evaluation과 Pre-trained Language Model을 이용하여 측정하는 두 가지 자동 평가 방법, KDA_disc와 KDA_cont를 제안합니다.



### Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions (https://arxiv.org/abs/2407.16698)
Comments:
          ECCV 2024. Code: this https URL Project page: this https URL

- **What's New**: 본 논문은 단일 이미지 깊이 추정 작업에서 복잡하고 분포에서 벗어난 데이터 (out-of-distribution data) 에 대한 새로운 접근 방식을 제시합니다. 이 접근 방식은 불리한 요소가 없는 이미지로 시작하여 사용자가 정의한 다양한 어려움과 해당 깊이 정보를 가진 새로운 장면을 체계적으로 생성합니다. 이는 텍스트에서 이미지로의 확산 모델 (text-to-image diffusion model) 을 활용하여 고품질 이미지 콘텐츠를 생성하고, 생성된 이미지와 원본 이미지 간의 3D 구조 일관성을 유지하는 방식으로 이루어집니다. 이후 어떠한 단안 깊이 네트워크의 미세 조정은 자체 증류 프로토콜을 통해 수행되는데, 이 프로토콜은 생성된 이미지와 간단하고 어려움이 없는 장면에 대한 깊이 예측을 고려합니다. 목적에 맞게 제작된 벤치마크에서 수행된 실험은 제안된 방식의 효율성과 다용성을 보여줍니다.

- **Technical Details**: 본 논문에서 제안된 방식은 텍스트에서 이미지로의 확산 모델을 활용하여 다양한 어려움을 가진 이미지를 생성하는 새로운 깊이 추정 방법입니다. 이는 기존 단안 깊이 네트워크를 훈련하고 미세 조정하는 데 사용할 수 있는 새로운 데이터셋을 생성하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과는 제안된 방식이 다양한 어려움을 가진 데이터셋에서 기존 단안 깊이 네트워크의 성능을 개선함을 보여줍니다. 특히, 비바람, 밤, 비반사 표면과 같은 까다로운 상황에서 효과적임을 보여줍니다. 또한, 이 방법은 기존 방법들과 비교하여 경쟁력있는 결과를 달성하는 것을 보여줍니다.



### AbdomenAtlas: A Large-Scale, Detailed-Annotated, & Multi-Center Dataset for Efficient Transfer Learning and Open Algorithmic Benchmarking (https://arxiv.org/abs/2407.16697)
Comments:
          Published in Medical Image Analysis

- **What's New**: 이 논문에서는 교육적 가치를 고려한 새로운 MCQ 자동 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 이 지표는 학생들의 응답을 분석하여 자동으로 MCQ의 품질을 평가합니다.



### PartGLEE: A Foundation Model for Recognizing and Parsing Any Objects (https://arxiv.org/abs/2407.16696)
Comments:
          Accepted by ECCV2024, homepage: this https URL

- **What's New**: PartGLEE는 이미지에서 객체와 부분을 모두 찾고 식별하기 위한 부분 수준 기반 모델을 제시합니다. PartGLEE는 통합 프레임워크를 통해 개방형 세계 시나리오에서 모든 입자도(granularity)에서 인스턴스를 감지, 분할, 접지(grounding)합니다. 특히, Q-Former를 제안하여 객체와 부분 간의 계층적 관계를 구축하고 모든 객체를 해당 의미적 부분으로 구문 분석합니다. 많은 양의 객체 수준 데이터를 통합함으로써 계층적 관계를 확장하여 PartGLEE가 다양한 부분을 인식할 수 있도록 할 수 있습니다.



### SAM-CP: Marrying SAM with Composable Prompts for Versatile Segmentation (https://arxiv.org/abs/2407.16682)
- **What's New**: 이 논문은 SAM-CP라고 하는 새로운 접근 방식을 제시하며, SAM (Segment Anything Model)을 이용하여 다양한 영역에서 뛰어난 성능을 보이는 세분화(segmentation)를 수행할 수 있도록 돕는다.  SAM-CP는 SAM이 생성한 패치(patch)를 넘어 두 가지 유형의 프롬프트(prompt)를 도입하여 다재다능한 세분화를 가능하게 한다. 특히, 입력으로 주어진 클래스 집합(텍스트)과 SAM 패치 집합을 이용하여 Type-I 프롬프트는 SAM 패치가 텍스트 레이블과 일치하는지 판단하고, Type-II 프롬프트는 동일한 텍스트 레이블을 가진 두 개의 SAM 패치가 동일한 인스턴스에 속하는지 판단한다. SAM-CP는 의미, 인스턴스, 패노라마 세분화(panoptic segmentation)를 모두 수행할 수 있으며, 특히 오픈 보캐뷸러리(open-vocabulary) 세분화에서 최첨단 성능을 보인다.



### FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process (https://arxiv.org/abs/2407.16670)
Comments:
          Will appear at ACM Multimedia 2024 (MM 2024), 13 pages, 15 figures

- **What's New**: 본 논문은 짧은 형식의 비디오 공유 플랫폼에서의 가짜 뉴스 탐지를 위한 새로운 접근 방식을 제시합니다. 기존 방법들은 비디오의 내용만 분석하는데 반해, 본 논문은 뉴스 비디오 제작 과정의 관점에서 가짜 뉴스를 분석합니다. 특히, 가짜 뉴스 비디오의 특징을 파악하기 위해 비디오 제작 과정에서의 소재 선택과 편집 방식을 분석했습니다.



### A Framework for Pupil Tracking with Event Cameras (https://arxiv.org/abs/2407.16665)
- **What's New**: 본 논문은 기존의 BLEU, ROUGE, METEOR 등의 평가 지표가 단순히 생성된 MCQ의 문장 유사성만을 고려하기 때문에 교육적 가치를 제대로 반영하지 못한다는 점을 지적하고, 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가하는 데 중점을 둡니다. 또한, KDA를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 해당 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 과정을 모방함으로써 KDA를 근사화합니다. 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 보이는 것으로 확인되었습니다.



### EgoCVR: An Egocentric Benchmark for Fine-Grained Composed Video Retrieva (https://arxiv.org/abs/2407.16658)
Comments:
          ECCV 2024

- **What's New**: 본 논문은 컴퓨터 비전 분야에서 새롭게 제시되는  'Composed Video Retrieval' 작업을 위한 새로운 평가 기준인 EgoCVR를 소개합니다. EgoCVR은 대규모 egocentric 비디오 데이터셋을 활용하여 시간적 흐름을 정확히 이해하는 데 중점을 둡니다. 특히 기존의 Composed Video Retrieval 프레임워크는 EgoCVR에서 요구되는 정확한 시간적 이해 능력이 부족하다는 문제점을 지적합니다. 이러한 문제를 해결하기 위해, 본 논문에서는 훈련 과정이 필요 없는 간단한 방법과  Composed Video Retrieval을 위한 일반적인 재랭킹 프레임워크를 제안하며, EgoCVR에서 뛰어난 성능을 보여줍니다.



### MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequenc (https://arxiv.org/abs/2407.16655)
Comments:
          23 pages, 18 figures

- **What's New**: 이 논문은 MovieDreamer라는 새로운 계층적 프레임워크를 소개하여 오토리그레시브 모델과 확산 기반 렌더링을 결합하여 장기간 내러티브 일관성과 단기 시각적 충실도를 조화시킵니다. 이 방법은 생성된 비디오 콘텐츠의 지속 시간을 수천 개의 키프레임으로 크게 확장합니다.



### Aggregated Attributions for Explanatory Analysis of 3D Segmentation Models (https://arxiv.org/abs/2407.16653)
Comments:
          Added Acknowledgments

- **What's New**: 본 논문은 MCQ 생성의 교육적 가치를 고려한 새로운 평가 지표인 지식 의존 가능성(KDA, Knowledge Dependent Answerability)을 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표들은 단순히 MCQ가 얼마나 정확하게 생성되었는지에만 초점을 맞추는 반면, KDA는 MCQ가 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 이는 학생들의 실제 응답을 기반으로 하며, 학생들의 문제 해결 방식을 모방하는 사전 훈련된 언어 모델을 활용하여 자동 평가 지표 KDA_disc와 KDA_cont를 제시합니다.



### Deformable Convolution Based Road Scene Semantic Segmentation of Fisheye Images in Autonomous Driving (https://arxiv.org/abs/2407.16647)
- **What's New**: 이 연구는 자동 운전 시나리오에서 어안 렌즈 이미지에 대한 의미론적 분할 작업에서 최신 변형 합성곱 신경망(DCNNs)의 효과를 조사합니다. 넓은 시야를 제공하는 어안 렌즈 이미지는 객체 속성의 동적 변화로 인해 공간 및 기하 정보를 추출하는 데 고유한 과제를 제시합니다. 실험은 WoodScape 어안 렌즈 이미지 데이터셋을 10개의 고유한 클래스로 분할하는 데 중점을 두며, 변형 네트워크가 복잡한 공간 관계를 포착하고 분할 정확도를 개선할 수 있는 능력을 평가합니다. 또한 클래스 불균형 문제를 해결하기 위해 다양한 손실 함수를 탐색하고 기존 CNN 아키텍처와 변형 합성곱 기반 CNN(Vanilla U-Net 및 Residual U-Net 아키텍처 포함)의 성능을 비교합니다. 변형 CNN을 통합하여 얻은 mIoU 점수의 상당한 개선은 어안 렌즈 영상에 존재하는 기하 왜곡을 처리하는 데 효과적임을 보여주며, 기존 CNN 아키텍처의 성능을 능가합니다. 이는 어안 렌즈 영상의 의미론적 분할 성능을 향상시키는 데 있어 변형 합성곱의 중요한 역할을 강조합니다.



### Unveiling and Mitigating Bias in Audio Visual Segmentation (https://arxiv.org/abs/2407.16638)
Comments:
          Accepted by ACM MM 24 (ORAL)

- **What's New**: 본 논문은 오디오-비주얼 분할(AVS) 모델의 성능을 향상시키기 위한 새로운 방법론을 제시합니다. 기존의 AVS 모델은 오디오-비주얼 grounding이 복잡하기 때문에, 단순한 시각 정보에 의존하여 오류가 발생하는 현상을 보이는데, 이를 '오디오 priming bias'와 'visual prior'로 분류하여 분석합니다.  



### DHGS: Decoupled Hybrid Gaussian Splatting for Driving Scen (https://arxiv.org/abs/2407.16600)
Comments:
          12 pages, 12 figures, conference

- **What's New**: 이 논문은 Decoupled Hybrid Gaussian Splatting (DHGS)라는 새로운 방법을 제안하여 주행 장면의 Novel View Synthesis 품질을 향상시킵니다. DHGS는 주행 장면을 도로와 비도로 영역으로 분리하고 각 영역에 대해 별도의 Gaussian 모델을 사용하여 최적화하는 방식을 사용합니다. 특히, DHGS는 Signed Distance Field (SDF)를 사용하여 도로 표면을 명시적으로 표현하고, 이를 통해 도로 영역의 기하학적 특성을 정확하게 모델링합니다.



### Timeliness-Fidelity Tradeoff in 3D Scene Representations (https://arxiv.org/abs/2407.16575)
Comments:
          This paper has been accepted for publication by the IEEE International Conference on Computer Communications (INFOCOM) Workshops 2024

- **What's New**: 본 논문에서는 실시간 3차원(3D) 장면 표현에서 시간성(timeliness)과 충실도(fidelity) 사이의 절충(tradeoff)을 조사합니다. 특히, 여러 카메라가 에지 서버와 통신하는 실제 시나리오를 모니터링하여 통신 지연(communication delay)이 절충에 미치는 영향을 평가하기 위한 프레임워크를 구축합니다.



### COALA: A Practical and Vision-Centric Federated Learning Platform (https://arxiv.org/abs/2407.16560)
Comments:
          ICML'24

- **What's New**: 이 논문은 COALA라는 비전 중심의 연합 학습(FL) 플랫폼과 실제 FL 시나리오를 위한 벤치마크 세트를 제시합니다. 이는 작업, 데이터, 모델의 세 가지 수준으로 분류됩니다. 작업 수준에서 COALA는 간단한 분류에서 객체 감지, 분할, 자세 추정 등 15가지 컴퓨터 비전 작업까지 지원을 확장합니다. 또한 연합 다중 작업 학습을 용이하게 하여 클라이언트가 여러 작업을 동시에 처리할 수 있도록 합니다. 데이터 수준에서 COALA는 지도 FL을 넘어 반지도 FL과 비지도 FL을 모두 벤치마크합니다. 또한 일반적으로 고려되는 레이블 분포 이동 외에도 특징 분포 이동을 벤치마크합니다. 정적 데이터를 처리하는 것 외에도 실제 시나리오에서 지속적으로 변화하는 데이터에 대한 연합 지속 학습을 지원합니다. 모델 수준에서 COALA는 분할된 모델과 서로 다른 클라이언트의 서로 다른 모델을 사용한 FL을 벤치마크합니다. COALA 플랫폼은 구성 사용자 지정, 구성 요소 사용자 지정, 워크플로우 사용자 지정을 포함하여 이러한 실제 FL 시나리오를 위해 세 가지 수준의 사용자 지정을 제공합니다. 저자는 실제 FL 시나리오에 대한 체계적인 벤치마킹 실험을 수행하고 FL에서 더 발전할 수 있는 잠재적인 기회를 강조합니다.



### MicroEmo: Time-Sensitive Multimodal Emotion Recognition with Micro-Expression Dynamics in Video Dialogues (https://arxiv.org/abs/2407.16552)
- **What's New**: 이 논문은 비디오에서 시각, 청각, 언어적 맥락을 통합하여 인간의 감정 상태를 인식하는 다중 모달 대규모 언어 모델(MLLM)을 위한 새로운 방법, MicroEmo를 제안합니다. MicroEmo는 특히 미세 표현의 시간적 동적 특징에 대한 주의를 집중하고 발화 인식 비디오 클립의 맥락적 의존성을 활용하는 데 중점을 둡니다.



### QPT V2: Masked Image Modeling Advances Visual Scoring (https://arxiv.org/abs/2407.16541)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문에서는 MCQ 생성의 교육적 가치를 고려한 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안했습니다. KDA는 MCQ가 해당 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 기존의 MCQ 생성 평가 지표인 BLEU, ROUGE, METEOR는 단어 유사도만 비교하기 때문에 교육적 가치를 고려하지 않았습니다. 본 연구에서는 인간 설문 조사를 통해 KDA를 측정하는 방법을 보여주고, 사전 학습된 언어 모델을 활용하여 KDA를 근사하는 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안했습니다.



### Is 3D Convolution with 5D Tensors Really Necessary for Video Analysis? (https://arxiv.org/abs/2407.16514)
- **What's New**: 이 논문은 2D 및/또는 1D convolution을 사용하여 5D tensor가 아닌 4D 및/또는 3D tensor로 3D convolutional block을 구현하는 새로운 기술을 제안합니다. 3D convolution은 계산량이 많아 로봇과 같은 실시간 애플리케이션에서 사용되는 엣지 장치에서는 지원되지 않을 수 있습니다. 기존 방법은 3D 커널을 공간 영역과 시간 영역으로 분리하여 이 문제를 완화하지만 구현에는 여전히 5D tensor를 사용하는 3D convolution이 필요합니다. 본 논문은 4D/3D tensor 재구성과 공간 및 시간 분할을 위한 새로운 결합 기법을 도입하여 이 문제를 해결합니다.



### DreamVTON: Customizing 3D Virtual Try-on with Personalized Diffusion Models (https://arxiv.org/abs/2407.16511)
- **What's New**: 이 논문에서는 이미지 기반 3D 가상 피팅 (VTON)을 위한 새로운 확산 기반 3D 인간 생성 프레임워크인 **DreamVTON**을 제안합니다. DreamVTON은 여러 개의 사람 이미지, 옷 이미지, 텍스트 프롬프트를 입력으로 받아 3D 인간을 생성합니다. DreamVTON은 개인화된 Stable Diffusion (SD) 모델과 여러 개념 및 노멀 스타일 LoRA (Low-Rank Adaptation)를 사용하여 3D 인간 최적화를 위한 강력한 생성 사전 (generative prior)을 제공합니다. 또한 DreamVTON은 SDS 손실 (Score Distillation Sampling loss)과 템플릿 기반 최적화 메커니즘을 함께 사용하여 고품질 3D 인간 생성을 달성합니다. 마지막으로 DreamVTON은 노멀 스타일 LoRA를 개인화된 SD에 추가하여 더 부드러운 기하학적 모델링을 가능하게 합니다.  



### ToDER: Towards Colonoscopy Depth Estimation and Reconstruction with Geometry Constraint Adaptation (https://arxiv.org/abs/2407.16508)
- **What's New**: 이 논문은 딥러닝 기반 콜로노스코피 영상 재구성 모델 ToDER를 제안하며, 이 모델은 양방향 적응 아키텍처를 사용하여 정확한 깊이 정보를 추정한다. ToDER는 기하학적 제약을 강제하는 TNet 모듈을 사용하여 깊이 추정 성능을 향상시킨다. 깊이 정보를 활용하여 콜로노스코피 영상을 시각화하는데 사용된다.



### HDRSplat: Gaussian Splatting for High Dynamic Range 3D Scene Reconstruction from Raw Images (https://arxiv.org/abs/2407.16503)
- **What's New**: HDRSplat은 실시간 3D 시각 재구성에 사용되는 3D Gaussian Splatting (3DGS) 방법을 14-bit 선형 HDR 이미지에 적용하여 야간 및 저조도 환경에서의 3D 시각 재구성 성능을 향상시키는 새로운 방법입니다. 기존 3DGS 방법은 8-bit LDR 이미지에 의존하여 저조도 환경에서 정확한 재구성에 어려움을 겪었습니다. HDRSplat은 HDR 공간에 적합한 새로운 손실 함수를 제안하여 노이즈가 있는 어두운 영역과 포화된 밝은 영역에서 정보를 효과적으로 추출하고, 뷰 의존성 색상을 처리합니다. 또한, 3DGS의 포인트 클라우드 초기화에 대한 의존성을 줄여 텍스처가 적거나 깊이가 깊고 조명이 부족한 영역에서도 정확한 재구성을 가능하게 합니다. HDRSplat은 기존의 RawNeRF보다 30배 빠른 속도로 14-bit HDR 3D 시각 재구성을 수행하며, 초당 120 프레임 이상의 빠른 추론 속도를 보여줍니다. 또한, HDRSplat은 합성 핀트 흐림, 밀집된 깊이 맵 추출, 노출 제어, 톤 매핑, 뷰 포인트 제어 등 다양한 애플리케이션에 활용될 수 있습니다.



### Dynamic Retraining-Updating Mean Teacher for Source-Free Object Detection (https://arxiv.org/abs/2407.16497)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 자동 다중 선택형 질문(MCQ) 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 기존의 n-gram 기반 유사성 메트릭과 달리, KDA는 MCQ가 목표 사실에 대한 학생의 지식을 평가할 수 있는지 측정합니다. 또한 KDA를 자동으로 추정하는 두 가지 메트릭인 KDA_disc와 KDA_cont를 제시합니다. 이러한 메트릭은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. 인간 평가 연구를 통해 KDA_disc와 KDA_cont가 KDA와 강의실에서의 사용성 모두와 강한 상관관계를 보인다는 것을 입증했습니다. 이는 MCQ 생성을 위한 교육적 가치를 평가하는 새로운 방법을 제시합니다.



### qMRI Diffusor: Quantitative T1 Mapping of the Brain using a Denoising Diffusion Probabilistic Mod (https://arxiv.org/abs/2407.16477)
Comments:
          Accepted by Deep Generative Models workshop at MICCAI 2024

- **What's New**: 본 논문에서는 qMRI Diffusor라는 새로운 qMRI 방법론을 제안하여 딥 제너레이티브 모델을 이용합니다. 특히, T1 정량화를 위한 잡음 제거 확산 확률 모델(DDPM)을 구현하여 정량적 지도 추정을 조건부 생성 작업으로 규정했습니다. 제안된 방법론은 팬텀 및 생체 내 데이터에서 잔차 신경망(ResNet) 및 순환 추론 머신(RIM)과 비교됩니다. 결과는 제안된 방법이 파라미터 추정에서 개선된 정확도와 정밀도를 달성하며, 뛰어난 시각적 성능을 보여줍니다. 또한, 이 방법은 본질적으로 확률성을 통합하여 불확실성을 간편하게 정량화할 수 있습니다. 따라서, 본 연구에서 제안된 방법은 정량적 MR 매핑에 상당한 가능성을 제시합니다.



### Lymphoid Infiltration Assessment of the Tumor Margins in H&E Slides (https://arxiv.org/abs/2407.16464)
Comments:
          Published in Medical Optical Imaging and Virtual Microscopy Image Analysis (MOVI) at MICCAI 2024

- **What's New**: 본 논문은 종양 주변 림프구 침투 평가를 위한 새로운 H&E 염색 기반 접근 방식을 제안한다. 이는 공개 데이터 세트에 훈련된 림프구 분할 모델을 사용하여 CD3+ 및 CD20+ 림프구를 정확하게 탐지하고 종양 경계선을 정확하게 식별하는 데 도움이 된다. 이 방법은 전통적인 IHC 염색과 비교하여 정확도가 뛰어나며, 특히 종양 경계선에서 림프구 밀도 곡선 생성에 유용하다. 또한, 이 방법은 튜링 테스트를 통해 검증되었으며, 병리학자의 맹검 평가를 통해 H&E 슬라이드와 IHC 슬라이드에서 얻은 곡선의 유사성을 확인하였다. 이는 암 관리 및 면역 요법 계획 개선을 위한 새로운 가능성을 제시한다.



### MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection (https://arxiv.org/abs/2407.16448)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 기존의 monocular 3D object detection 모델이 주로 맑은 날씨에 대한 성능만을 고려해 왔다는 한계를 지적하며, 흐린 날씨에도 robust하게 작동하는 새로운 모델, MonoWAD를 제안합니다. MonoWAD는 weather codebook과 weather-adaptive diffusion model이라는 두 가지 핵심 구성 요소를 통해 다양한 날씨 조건에서도 3D object detection 성능을 유지할 수 있습니다.



### Rethinking Out-of-Distribution Detection on Imbalanced Data Distribution (https://arxiv.org/abs/2407.16430)
Comments:
          N/A

- **What's New**: 이 논문은 불균형 데이터 분포에서 OOD 탐지 문제를 해결하기 위한 새로운 프레임워크인 ImOOD를 제안합니다. ImOOD는 불균형 데이터 분포에서 OOD 탐지 문제를 정식화하고, 균형과 불균형 OOD 탐지 간의 클래스 인식 편향 항목이 존재한다는 이론적 분석을 제시합니다. 이러한 분석 결과를 기반으로, 본 논문은 훈련 시간 규제 기법을 통해 OOD 탐지 성능을 향상시키는 방법을 제안합니다.



### ESOD: Efficient Small Object Detection on High-Resolution Images (https://arxiv.org/abs/2407.16424)
Comments:
          N/A

- **What's New**: 본 논문은 고해상도 이미지에서 작은 물체를 효율적으로 탐지하는 새로운 프레임워크인 ESOD를 제안합니다. ESOD는 기존 객체 검출기의 백본을 재활용하여 특징 수준에서 객체를 찾고 패치를 슬라이싱하여 불필요한 특징 추출을 피하고 계산 비용을 줄입니다. 또한, 희소 검출 헤드를 통합하여 고해상도 입력 (예: 1080P 또는 그 이상)에서 작은 물체를 감지하여 성능을 향상시킵니다. ESOD는 CNN 기반 및 ViT 기반 검출기에 모두 적용할 수 있는 일반적인 프레임워크입니다.



### Hi-EF: Benchmarking Emotion Forecasting in Human-interaction (https://arxiv.org/abs/2407.16406)
- **What's New**: 이 연구는 심리학의 미래 감정 예측(Affective Forecasting) 분야를 딥러닝 문제로 변환하여 두 사람의 상호작용을 기반으로 한 감정 예측 패러다임을 설계합니다. 특히, 이 연구는 개인의 감정이 다른 사람과의 상호작용 중에 전달되는 감정이나 다른 정보에 의해 쉽게 영향을 받는다는 이론에 근거하여 새로운 감정 예측(Emotion Forecasting, EF) 작업을 제안합니다. 이를 위해 연구진은 두 사람의 상호 작용에 대한 다층적 맥락 정보 샘플(Multilayered-Contextual Interaction Samples, MCIS)을 포함하는 Human-interaction-based Emotion Forecasting (Hi-EF) 데이터셋을 개발했습니다. Hi-EF는 EF 작업의 실행 가능성을 보여줄 뿐만 아니라 그 잠재력을 강조합니다. 또한, 연구진은 EF 작업에 대한 기본적인 참조 모델을 확립하는 방법론을 제안하고 광범위한 실험을 제공합니다.

- **Technical Details**: 이 연구는 감정 예측을 딥러닝 문제로 변환하고 두 사람의 상호작용을 기반으로 한 새로운 감정 예측 패러다임을 제안합니다. 이를 위해 연구진은 Human-interaction-based Emotion Forecasting (Hi-EF) 데이터셋을 개발했으며, Hi-EF는 다층적 맥락 정보 샘플(Multilayered-Contextual Interaction Samples, MCIS)을 포함하고 있습니다. Hi-EF는 얼굴 표정, 대화 내용, 음성 톤 등 다양한 정보를 포함하며, 두 사람의 감정 변화를 예측하는 데 도움을 줍니다.

- **Performance Highlights**: 이 연구는 Hi-EF 데이터셋과 기본 모델을 통해 감정 예측 작업의 실행 가능성을 보여주고, 향후 개인 감정 모델링, 인간형 감정 생성 등 다양한 응용 분야에 활용될 수 있음을 제시합니다.



### Learning Unsigned Distance Functions from Multi-view Images with Volume Rendering Priors (https://arxiv.org/abs/2407.16396)
Comments:
          Accepted by ECCV 2024. Project page: this https URL

- **What's New**: 이 논문은 기존의 handcrafted 방식 대신 데이터 기반 학습 방식으로 훈련된 신경망을 사용하는 새로운 differentiable renderer를 제안합니다. 이를 통해 UDFs (Unsigned Distance Functions)를 더 정확하게 추론할 수 있는 '볼륨 렌더링 prior (volume rendering priors)'라는 지식을 얻게 됩니다.  



### SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieva (https://arxiv.org/abs/2407.16394)
Comments:
          Accepted to ACM International Conference on Multimedia (MM) 2024

- **What's New**: 본 연구에서는 기존의 비디오 검색과 달리, 수어 검색은 영상 클립에 담긴 인간 행동의 의미 정보를 이해하는 데 더 치우쳐져 있습니다. 이전 연구는 일반적으로 RGB 영상을 인코딩하여 고수준 의미 특징을 얻는 데 집중했기 때문에, 많은 시각 정보 중복으로 인해 지역적인 행동 세부 정보가 묻히는 문제가 있었습니다. 또한, 기존의 RGB 기반 수어 검색 연구는 밀집된 시각 데이터 임베딩을 사용하여 end-to-end 훈련 시 막대한 메모리 비용이 발생하며, 오프라인 RGB 인코더를 사용하여 최적의 특징 표현을 얻지 못했습니다. 이러한 문제를 해결하기 위해, 본 연구는 수어 영상의 지역적 및 전역적 정보를 표현하기 위해 포즈 및 RGB 모달리티를 통합한 '의미적으로 향상된 이중 스트림 인코더 (Semantically Enhanced Dual-Stream Encoder, SEDS)'라는 새로운 수어 표현 프레임워크를 제안합니다. 특히, 포즈 인코더는 인간 관절에 해당하는 키포인트의 좌표를 임베딩하여 세부적인 행동 특징을 효과적으로 포착합니다. 두 가지 영상 모달리티를 더 잘 융합하기 위해, 본 연구는 모달리티 내 및 모달리티 간 유사한 의미 정보를 가진 인접 클립 특징을 집계하는 '교차 글로스 주의 융합 (Cross Gloss Attention Fusion, CGAF)' 모듈을 제안합니다. 또한, 미세한 이중 스트림 특징의 문맥 일치를 통해 집계된 융합 특징을 향상시키기 위한 '포즈-RGB 미세 입자 일치 목표 (Pose-RGB Fine-grained Matching Objective)'가 개발되었습니다. 오프라인 RGB 인코더 외에, 전체 프레임워크는 학습 가능한 경량 네트워크만 포함하고 있으며 end-to-end로 훈련할 수 있습니다. 광범위한 실험 결과, 본 연구의 프레임워크는 다양한 데이터셋에서 최첨단 방법을 능가하는 것으로 나타났습니다.



### A Multitask Deep Learning Model for Classification and Regression of Hyperspectral Images: Application to the large-scale datas (https://arxiv.org/abs/2407.16384)
- **What's New**: 본 논문은 hyperspectral 이미지에서 여러 분류(classification) 및 회귀(regression) 작업을 동시에 수행하도록 설계된 다중 작업 딥 러닝 모델을 제안합니다. 특히, 대규모 hyperspectral 데이터셋인 TAIGA를 사용하여 모델을 검증했습니다. 이 데이터셋은 3개의 범주형 변수와 10개의 연속 변수를 포함하여 13개의 숲 변수를 포함하고 있습니다. 또한, 본 연구에서는 다중 스케일 컨텍스트 정보를 추출하고 작업별 특징을 우선시하여 선택적 정보 처리를 가능하게 하기 위해 밀집된 atrous pyramid pooling 레이어와 어텐션 네트워크를 통합했습니다.



### FCNR: Fast Compressive Neural Representation of Visualization Images (https://arxiv.org/abs/2407.16369)
- **What's New**: 이 논문은 FCNR이라는 새로운 압축 신경 표현 방법을 제안하며, 다양한 시점과 시간 단계에서 수만 장의 시각화 이미지를 빠르게 압축하는 방법을 제공합니다. 기존의 NeRVI 솔루션은 높은 압축률을 제공하지만, 인코딩 및 디코딩 속도가 느립니다. FCNR은 최근 스테레오 이미지 압축 기술 발전을 기반으로 스테레오 컨텍스트 모듈과 공동 컨텍스트 전송 모듈을 통합하여 이미지 쌍을 압축합니다. 이 솔루션은 높은 재구성 품질과 압축률을 유지하면서 인코딩 및 디코딩 속도를 크게 향상시킵니다. 효과를 입증하기 위해 FCNR을 E-NeRV, HNeRV, NeRVI, ECSIC을 포함한 최첨단 신경 압축 방법과 비교했습니다. 소스 코드는 [링크]에서 확인할 수 있습니다.



### Navigating Uncertainty in Medical Image Segmentation (https://arxiv.org/abs/2407.16367)
Comments:
          Published in the conference proceedings of the 21st IEEE International Symposium on Biomedical Imaging (ISBI 2024)

- **What's New**: 이 논문은 의료 영상에서 불확실한 분할 방법 (uncertain segmentation) 의 선택 및 평가를 다룹니다. 특히 전립선 분할과 폐 병변 분할 두 가지 사례 연구를 통해, 최소한의 주석자 변화 (annotator variation) 가 있는 경우에는 단순한 결정론적 모델 (deterministic models) 만으로도 충분하지만, 폐 병변 분할에서는 일반화된 에너지 거리 (GED) 의 모델 선택 제한 사항을 강조합니다. 이 연구는 알레아토릭 (aleatoric) 과 인식론적 (epistemic) 요소를 통합한 불확실한 분할 모델을 정확하게 선택하고 개발하기 위한 지침을 제시합니다. 이 지침은 연구자와 실무자가 불확실한 분할 방법을 더 잘 개발, 선택 및 평가할 수 있도록 하여 실제로 분할 불확실성을 더 쉽게 채택하고 효과적으로 적용할 수 있도록 합니다.



### Harmonizing Visual Text Comprehension and Generation (https://arxiv.org/abs/2407.16364)
- **What's New**: 본 논문에서는 TextHarmony라는 새로운 멀티모달 생성 모델을 소개합니다. 이 모델은 이미지와 텍스트를 동시에 이해하고 생성하는 능력을 갖추고 있습니다. 이전의 모델들은 비전과 언어 모드 사이의 불일치로 인해 성능 저하를 보였지만, TextHarmony는 Slide-LoRA를 사용하여 이러한 문제를 해결했습니다. Slide-LoRA는 모드별 LoRA 전문가를 동적으로 통합하여 멀티모달 생성 공간을 부분적으로 분리합니다. 또한, 고품질 이미지 캡션 데이터셋인 DetailedTextCaps-100K를 개발하여 비주얼 텍스트 생성 능력을 더욱 향상시켰습니다.



### Strike a Balance in Continual Panoptic Segmentation (https://arxiv.org/abs/2407.16354)
- **What's New**: 본 논문은 기존의 MCQ 평가 지표들이 MCQ의 교육적 가치를 고려하지 않고, 단순히 텍스트 유사도만 평가한다는 한계점을 지적하며 새로운 평가 지표인 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)를 제안합니다. KDA는 MCQ가 실제로 학생의 지식을 측정할 수 있는지 평가하는 지표이며, human evaluation을 통해 KDA와 높은 상관관계를 보이는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 이러한 지표들은 n-gram 기반 유사도 지표와 함께 사용하여 다양한 전문가가 평가한 MCQ 품질 척도를 예측하는 데 뛰어난 능력을 보여줍니다.



### SOAP: Enhancing Spatio-Temporal Relation and Motion Information Capturing for Few-Shot Action Recognition (https://arxiv.org/abs/2407.16344)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 본 논문은 High Frame-Rate (HFR) 비디오에서 액션 인식 (action recognition)을 위한 새로운 플러그 앤 플레이 아키텍처인 Spatio-tempOral frAme tuPle enhancer (SOAP)를 제안합니다. SOAP는 기존의 FSAR (Few-shot action recognition) 모델의 한계를 극복하고, HFR 비디오의 특징인 미세한 액션 표현과 낮은 공간-시간적 관계 및 모션 정보 밀도를 효과적으로 처리하는 것을 목표로 합니다. SOAP는 3D 컨볼루션을 사용하여 공간-시간적 관계를 구축하는 3DEM (3-Dimension Enhancement Module)과 채널별 특징 응답을 적응적으로 보정하는 CWEM (Channel-Wise Enhancement Module) 그리고 다양한 프레임 수의 프레임 튜플 (frame tuples)을 결합하여 광범위한 시각을 제공하는 HMEM (Hybrid Motion Enhancement Module)을 사용합니다. 이러한 모듈들을 결합하여 SOAP는 기존 방법들에 비해 공간-시간적 관계를 보다 효과적으로 구축하고, 다양한 프레임 튜플을 통해 포괄적인 모션 정보를 포착합니다.



### Motion Capture from Inertial and Vision Sensors (https://arxiv.org/abs/2407.16341)
Comments:
          17 pages,9 figures

- **What's New**: 이 논문에서는 저렴하고 사용하기 쉬운 개인용 모션 캡처 솔루션 개발을 위해 모션 캡처 데이터셋인 MINIONS를 공개했습니다. MINIONS 데이터셋은 단일 카메라와 소수의 관성 측정 장치(IMU)를 사용하여 수집되었으며, 다양한 종류의 액션을 포함하고 있습니다. 이 데이터셋은 저렴한 가격으로 다양한 액션을 캡처할 수 있는 가능성을 보여주며, 향후 연구 및 개발에 중요한 자료가 될 것으로 예상됩니다.



### A new visual quality metric for Evaluating the performance of multidimensional projections (https://arxiv.org/abs/2407.16309)
Comments:
          19 pages, 10 figures

- **What's New**: 본 논문에서는 다차원 데이터를 2차원으로 시각화하는 기법인 다차원 투영(MP)의 품질을 평가하기 위해 새로운 시각적 품질 메트릭을 제안합니다. 이 새로운 메트릭은 기존의 실루엣 계수(silhouette coefficient), 이웃 유지(neighborhood preservation), 실루엣 비율(silhouette ratio) 세 가지 메트릭을 결합합니다. 또한, LAMP(Local Affine Multidimensional Projection)라는 기존의 다차원 투영 방법의 한계를 극복하기 위한 새로운 알고리즘을 제시합니다. LAMP는 제어점과 시각 공간 상의 대응 지점 간 스케일이 유사해야 한다는 제약을 가지고 있는데, 제안된 알고리즘은 이러한 문제를 해결하기 위해 노력합니다.



### SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging (https://arxiv.org/abs/2407.16308)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 HDR 이미지 생성을 위한 효율적인 Selective Alignment Fusion Network (SAFNet)을 제안한다. SAFNet은 피라미드 특징을 추출하고 공유 디코더를 사용하여 선택된 영역에서 가치 있는 영역 마스크와 크로스 노출 모션을 동시에 개선한 다음, 고품질 HDR 이미지를 명시적으로 융합한다. 이러한 접근 방식은 모델이 가치 있는 영역을 찾는 데 집중하는 동시에 쉽게 감지되고 의미 있는 모션을 추정할 수 있다. 또한, 경량화된 개선 모듈을 도입하여 이전 광학 흐름, 선택 마스크 및 초기 예측의 이점을 활용한다. 더 나아가, 큰 모션이 있는 샘플에 대한 학습을 용이하게 하기 위해 훈련 중에 새로운 창 분할 자르기 방법이 제시된다. 공개 및 새롭게 개발된 어려운 데이터셋에서의 실험 결과는 제안된 SAFNet이 기존 SOTA 경쟁자보다 양적 및 질적으로 뛰어날 뿐만 아니라 훨씬 빠르게 실행된다는 것을 보여준다. 코드와 데이터셋은 [링크]에서 이용 가능하다.



### DeepClean: Integrated Distortion Identification and Algorithm Selection for Rectifying Image Corruptions (https://arxiv.org/abs/2407.16302)
Comments:
          7 pages, 3 figures

- **What's New**: 이 논문은 이미지와 비디오에서 왜곡 (distortion)을 식별하고 수정하기 위한 새로운 방법을 제안합니다. 기존의 고정적인 이미지 처리 파이프라인 (trial-and-error based image processing pipeline) 대신, 두 단계의 순차적 계획 (sequential planning) 접근 방식을 사용하여 자동으로 이미지 왜곡을 분류하고 수정합니다. 상위 레벨에서는 입력 이미지에 존재하는 왜곡 유형 (corruption class)을 감지하고, 하위 레벨에서는 외부에서 제공된 후보 알고리즘 집합에서 적용할 특정 알고리즘을 선택합니다. 이 두 단계 시스템은 추론 (inference) 중에 단일 전달 (forward pass) 형태로 실행되며 원본 이미지를 복구할 때까지 반복적으로 쿼리됩니다.  



### TAPTRv2: Attention-based Position Update Improves Tracking Any Poin (https://arxiv.org/abs/2407.16291)
- **What's New**: TAPTRv2는 Tracking Any Point (TAP) task를 해결하기 위한 Transformer 기반 접근 방식으로, TAPTR을 기반으로 하며 DETR (DEtection TRansformer)의 디자인을 차용하여 각 추적 지점을 포인트 쿼리로 표현함으로써 DETR과 유사한 알고리즘에서 잘 알려진 연산을 활용할 수 있도록 합니다. TAPTRv2는 cost-volume에 대한 의존성과 관련된 TAPTR의 중요한 문제를 해결함으로써 TAPTR을 개선합니다. cost-volume은 포인트 쿼리의 콘텐츠 특징을 오염시키고 가시성 예측과 cost-volume 계산에 부정적인 영향을 미칩니다. TAPTRv2에서는 새로운 주의 기반 위치 업데이트(APU) 연산을 제안하고, 키 인식 디포멀러블 어텐션 (key-aware deformable attention)을 사용하여 구현합니다. 각 쿼리에 대해 이 연산은 키 인식 어텐션 가중치를 사용하여 해당 디포멀러블 샘플링 위치를 결합하여 새로운 쿼리 위치를 예측합니다. 이 디자인은 로컬 어텐션 (local attention)이 기본적으로 cost-volume과 동일하며 둘 다 쿼리와 주변 특징 사이의 내적을 통해 계산된다는 관찰에 기반합니다. 이 새로운 연산을 도입함으로써 TAPTRv2는 cost-volume 계산의 부담을 제거할 뿐만 아니라 성능을 크게 향상시킵니다. TAPTRv2는 TAPTR을 능가하고 여러 가지 어려운 데이터셋에서 최첨단 성능을 달성하여 우수성을 보여줍니다.



### Federated Learning for Face Recognition via Intra-subject Self-supervised Learning (https://arxiv.org/abs/2407.16289)
Comments:
          Accepted at the The 35th British Machine Vision Conference 2024 (BMVC 2024), Glasgow, UK. Youngjun Kwak is corresponding author

- **What's New**: 본 논문은 기존의 Federated Learning(FL) 기반 얼굴 인식 모델의 한계를 극복하기 위해 개인화된 얼굴 인식 모델을 훈련하는 새로운 FedFS(Federated Learning for personalized Face recognition via intra-subject Self-supervised learning framework) 아키텍처를 제안합니다. FedFS는 개인 데이터 유출 없이 사용자 기기에서 개인화된 얼굴 인식 모델을 훈련할 수 있습니다.



### When, Where, and What? An Novel Benchmark for Accident Anticipation and Localization with Large Language Models (https://arxiv.org/abs/2407.16277)
- **What's New**: 본 논문에서는 자율 주행 시스템의 안전을 강화하기 위해 대규모 언어 모델(LLM)을 통합한 새로운 프레임워크를 제안합니다. 이 프레임워크는 사고가 발생할 수 있는 시간, 위치, 종류를 정확하게 예측하는 데 초점을 맞춥니다. 또한, 복잡한 주행 환경에서 고위험 요소를 우선시하는 역동적인 체인 기반 주의 메커니즘을 도입했습니다. 이 메커니즘은 소규모 모델의 출력을 다중 모드 입력으로 처리하는 3단계 모델과 결합되어 LLM이 교통 역학을 더욱 정확하게 이해할 수 있도록 합니다. DAD, CCD, A3D 데이터셋에 대한 실험 결과는 평균 정밀도(AP)와 사고까지의 평균 시간(mTTA) 측면에서 우수한 성능을 보여주었으며, 이는 사고 예측 기술의 새로운 기준을 제시합니다. 이 연구는 자율 주행 시스템의 안전성을 향상시키는 기술적 기반을 마련할 뿐만 아니라, 인간-AI 상호 작용을 개선하여 자율 주행 시스템이 생성한 예측적 통찰력을 더 직관적이고 실행 가능하게 만듭니다.



### HyTAS: A Hyperspectral Image Transformer Architecture Search Benchmark and Analysis (https://arxiv.org/abs/2407.16269)
Comments:
          The paper is accepted at ECCV2024

- **What's New**: 본 연구는 교사의 학습 평가 시간을 크게 줄일 수 있는 자동 MCQ 생성의 새로운 평가 지표인 **Knowledge Dependent Answerability (KDA)**를 제안합니다. 기존 지표들은 MCQ가 학생의 지식을 평가하는 능력을 고려하지 않았지만, KDA는 학생들의 답변을 바탕으로 MCQ의 답변 가능성을 측정합니다. KDA를 자동화하기 위해, **KDA_disc** 와 **KDA_cont** 두 가지 지표를 제시합니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.



### Image Classification using Fuzzy Pooling in Convolutional Kolmogorov-Arnold Networks (https://arxiv.org/abs/2407.16268)
Comments:
          The paper has been submitted to IEEE SCIS ISIS 2024 for consideration

- **What's New**: 본 논문에서는 MCQ 생성 평가 지표로 '지식 종속 가능성(KDA)'을 제안하여 기존 BLEU, ROUGE, METEOR와 같은 지표가 갖는 교육적 가치를 고려하지 않는 문제를 해결합니다. KDA는 MCQ의 답변 가능성(answerability)을 측정하여 학생의 지식 평가 능력을 측정하며, 학생들의 응답을 이용하여 KDA를 계산하는 방법과 더불어, 사전 학습된 언어 모델을 활용하여 KDA를 자동으로 계산하는 두 가지 지표(KDA_disc와 KDA_cont)를 제시합니다.



### Masks and Manuscripts: Advancing Medical Pre-training with End-to-End Masking and Narrative Structuring (https://arxiv.org/abs/2407.16264)
Comments:
          Accepted in MICCAI-24

- **What's New**: 본 논문은 의학적 대조 학습(contrastive learning)에서 나타나는 의미의 불일치와 샘플 쌍의 형태적 차이를 해결하기 위해 새로운 접근 방식을 제안합니다. 특히, 텍스트 보고서를 표준화된 삼중항 형식으로 변환하여 '관찰(observations)'과 '판결(verdicts)'이라는 개념을 도입합니다. 또한, 의료 이미지의 지역적 맥락을 나타내는 특징에 초점을 맞춘 Meijering 기반 마스킹을 사용하여 비주얼 프리트레이닝을 개선합니다. 이러한 접근 방식을 통해 모델은 다중 모달 대조 학습 프레임워크에서 교차 모달 표현을 향상시키고 의료 이미지 분석 분야에서 새로운 벤치마크를 설정합니다.



### DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors (https://arxiv.org/abs/2407.16260)
Comments:
          ECCV 2024. Project page: this https URL

- **What's New**: 본 논문은 텍스트에서 3D 오브젝트를 생성하는 기술에서 다수의 독립적인 오브젝트를 생성하고 상호작용을 가능하게 하는 'DreamDissector' 기술을 소개합니다. 이 기술은 기존의 텍스트-3D 생성 모델과 달리, 독립적인 오브젝트의 생성과 함께 공간적으로 타당한 상호작용을 제공합니다.  



### Spatiotemporal Graph Guided Multi-modal Network for Livestreaming Product Retrieva (https://arxiv.org/abs/2407.16248)
Comments:
          9 pages, 12 figures

- **What's New**: 본 연구는 livestreaming에서 판매되는 상품을 정확하게 식별하는 새로운 방법, Spatiotemporal Graphing Multi-modal Network (SGMN)을 제안합니다. SGMN은 판매자의 음성 정보를 활용하여 상품에 대한 주의를 집중시키고, 장기간 공간-시간 그래프 네트워크를 통해 비디오-이미지 이질성을 해결하며, 다중 모달 하드 샘플 마이닝을 통해 유사한 상품을 구별하는 데 도움을 줍니다.  



### HSVLT: Hierarchical Scale-Aware Vision-Language Transformer for Multi-Label Image Classification (https://arxiv.org/abs/2407.16244)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 다중 레이블 이미지 분류 (multi-label image classification)를 위한 새로운 방법인 HSVLT (Hierarchical Scale-Aware Vision-Language Transformer) 를 제안한다. 이 방법은 다양한 크기와 외형의 객체를 인식하기 위해 계층적 다중 스케일 아키텍처와 상호 작용하는 시각 언어 어텐션 (Interactive Visual-Linguistic Attention, IVLA) 을 사용한다. 또한, 다중 스케일 정보를 통합하는 크로스 스케일 집계 (Cross-Scale Aggregation, CSA) 모듈을 제안한다.



### Chameleon: Images Are What You Need For Multimodal Learning Robust To Missing Modalities (https://arxiv.org/abs/2407.16243)
- **What's New**: 본 논문에서는 기존의 Multimodal Learning의 Multi-Branch 구조에서 벗어나, 텍스트 정보를 시각 정보로 변환하여 하나의 포맷으로 통합하는 새로운 방법인 Chameleon을 제안합니다. 이를 통해 modality-specific branch 없이 modality-independent multimodal representation을 학습할 수 있고, 특정 modality가 부족하더라도 robust한 성능을 유지할 수 있습니다.



### Channel-Partitioned Windowed Attention And Frequency Learning for Single Image Super-Resolution (https://arxiv.org/abs/2407.16232)
Comments:
          Version 1, BMVC 2024

- **What's New**: 본 논문은 이미지 슈퍼 해상도(SISR)를 위한 새로운 아키텍처인 채널 분할 주의 트랜스포머(CPAT)를 제안합니다. CPAT는 장거리 의존성을 더 잘 포착하기 위해 윈도우를 높이와 너비를 따라 순차적으로 확장하는 새로운 채널 분할 윈도우 자기 주의 메커니즘(CPWin-SA)을 사용합니다. 또한, 공간 및 주파수 영역의 정보를 통합하여 특징 맵에서 더 포괄적인 정보를 제공하는 공간-주파수 상호 작용 모듈(SFIM)을 설계했습니다. 이를 통해 주파수 콘텐츠에 대한 정보를 포함하고 전체 이미지에 걸쳐 수용 영역을 향상시킵니다. 



### OutfitAnyone: Ultra-high Quality Virtual Try-On for Any Clothing and Any Person (https://arxiv.org/abs/2407.16224)
Comments:
          10 pages, 13 figures

- **What's New**: 이 논문에서는 Virtual Try-On (VTON) 문제를 해결하기 위해 'OutfitAnyone'이라는 새로운 방법을 제안합니다. OutfitAnyone은 기존 VTON 방법의 한계인 고품질, 디테일 유지, 다양한 옷과 사람에 대한 적용성 문제를 해결하기 위해 Two-stream conditional diffusion model을 사용합니다. 특히, Pose, 체형, 이미지 유형 (애니메이션, 실제 사진) 등에 대한 확장성을 고려하여 다양한 상황에서 높은 성능을 보여줍니다. OutfitAnyone은 Zero-shot Try-on Network와 Post-hoc Refiner 두 가지 요소로 구성되어 있으며, 처음에는 옷의 외형을 생성하고, 이후에는 옷과 피부의 질감을 개선하는 역할을 합니다.



### Diff-Shadow: Global-guided Diffusion Model for Shadow Remova (https://arxiv.org/abs/2407.16214)
- **What's New**: 이 연구는 고품질 그림자 제거를 위한 새로운 확산 모델인 Diff-Shadow를 제안한다. 기존 트랜스포머 기반 접근 방식은 그림자 영역과 비 그림자 영역을 연결하기 위해 전역 정보를 활용하지만, 합성 능력이 제한되어 명확한 경계가 있는 이미지를 복구하는 데 어려움이 있다. 반대로, 확산 기반 방법은 더 나은 콘텐츠를 생성할 수 있지만, 전역 정보를 무시하여 조명이 불일치하는 결과를 초래한다. 이 연구에서는 그림자 없는 복원을 실현하기 위해 확산 모델의 장점과 전역 안내를 결합한다. 특히, 두 개의 병렬 UNets 아키텍처를 제안한다. 1) 지역 분기는 확산 과정에서 패치 기반 노이즈 추정을 수행하고, 2) 전역 분기는 저해상도 그림자 없는 이미지를 복구한다. 비 그림자 영역의 전역 맥락 정보를 지역 분기에 통합하기 위해 Reweight Cross Attention (RCA) 모듈을 설계했다. 또한 복원된 이미지에서 그림자 영역과 비 그림자 영역 전반에 걸쳐 일관된 조명을 보장하고 패치 경계 문제를 완화하는 Global-guided Sampling Strategy (GSS)를 설계했다.



### CLII: Visual-Text Inpainting via Cross-Modal Predictive Interaction (https://arxiv.org/abs/2407.16204)
- **What's New**: 이 논문에서는 자동 MCQ 생성 평가에 있어 교육적 가치를 고려하는 새로운 지표, 지식 종속 가능성 (KDA) 을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. KDA는 학생들의 설문 조사 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표는 사전 학습된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 연구 결과, KDA_disc와 KDA_cont는 인간 평가와 높은 상관 관계를 보였고 실제 강의실 환경에서의 사용성과도 높은 상관 관계를 보였습니다.



### INF-LLaVA: Dual-perspective Perception for High-Resolution Multimodal Large Language Mod (https://arxiv.org/abs/2407.16198)
- **What's New**: 이 논문은 기존 MCQ 생성 평가 메트릭의 한계를 극복하고 교육적 가치를 고려한 새로운 자동 평가 메트릭, '지식 종속 가능성(KDA)'을 제안합니다. KDA는 학생의 지식 수준을 평가하는 MCQ의 능력을 측정하며, 실제 학생 응답을 기반으로 측정됩니다. 이 논문은 KDA를 근사화하기 위해 사전 학습된 언어 모델을 이용한 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가에 의한 실제 강의실 설정에서의 사용성과 높은 상관관계를 보인다는 것을 입증합니다. 또한, 이 메트릭은 기존 n-gram 기반 유사성 메트릭과 함께 사용하면 전문가가 평가한 다양한 MCQ 품질 측정 지표를 예측하는 능력이 뛰어납니다.



### LiCROcc: Teach Radar for Accurate Semantic Occupancy Prediction using LiDAR and Camera (https://arxiv.org/abs/2407.16197)
- **What's New**: 이 논문에서는 자동 MCQ 생성의 교육적 가치를 평가하기 위한 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고 학생의 지식 수준을 평가하는 데 중점을 둡니다. 또한, KDA를 자동으로 측정하기 위한 두 가지 새로운 지표인 KDA_disc와 KDA_cont를 제시합니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 추정합니다.



### CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation (https://arxiv.org/abs/2407.16193)
Comments:
          32 pages; Accepted to ECCV2024

- **What's New**: 이 논문은 3D point cloud의 noisy data를 다루기 위한 새로운 test-time input adaptation 방법인 CloudFixer를 제안합니다. CloudFixer는 pre-trained diffusion model을 활용하여 test-time에 input data를 source domain으로 변환하는 방식으로, 기존의 2D 이미지에 적용된 DDA (Diffusion-based Domain Adaptation)를 3D point cloud에 최적화한 것입니다. CloudFixer는 point cloud의 기하학적 특징을 고려한 optimization objective와 효율적인 계산 방법을 사용하며, online model adaptation 전략을 통해 더욱 향상된 성능을 보여줍니다.



### EIANet: A Novel Domain Adaptation Approach to Maximize Class Distinction with Neural Collapse Principles (https://arxiv.org/abs/2407.16189)
Comments:
          12 pages, 3 figures. Accepted by BMVC2024

- **What's New**: 이 논문은 출처 없는 도메인 적응(SFDA)에서 각 클래스의 프로토타입을 분리하기 위한 새로운 ETF-Informed Attention Network (EIANet)를 소개한다. EIANet은 attention mechanism과 neural collapse principle을 활용하여 유사한 샘플 임베딩을 가진 클래스를 구분한다. 특히 EIANet은 simplex Equiangular Tight Frame (ETF) 분류기를 attention mechanism과 결합하여 차별적인 특징에 집중하고 클래스 프로토타입 간의 최대 분리를 보장한다.



### No Re-Train, More Gain: Upgrading Backbones with Diffusion Model for Few-Shot Segmentation (https://arxiv.org/abs/2407.16182)
Comments:
          7 figures

- **What's New**: 본 논문에서는 새로운 Few-Shot Segmentation (FSS) 방법인 DiffUp을 제안하여, 다양한 유형의 주석 (예: 스크리블, 바운딩 박스, 마스크, 텍스트)을 통합적으로 처리하고, 주석의 양에 따라 유연하게 적응할 수 있는 능력을 향상시킵니다. 또한, DiffUp은 백본 업그레이드를 위해 재훈련이 필요하지 않고, 다양한 유형의 주석을 통합적으로 처리할 수 있습니다.



### Integrating Meshes and 3D Gaussians for Indoor Scene Reconstruction with SAM Mask Guidanc (https://arxiv.org/abs/2407.16173)
- **What's New**: 본 논문은 3D 실내 장면 재구성을 위해 3D Gaussian Splatting (3DGS)와 메쉬 표현을 결합하는 새로운 방법을 제시합니다. 벽, 천장, 바닥과 같은 실내 장면의 방 배치에는 메쉬를 사용하고 다른 물체에는 3D Gaussian을 사용합니다. 이 하이브리드 접근 방식은 두 표현 방식의 장점을 활용하여 유연성을 향상시키고 편집을 용이하게 합니다. 그러나 메쉬와 3D Gaussian을 함께 학습하는 것은 어려운데, 어떤 기본 요소가 렌더링된 이미지의 어떤 부분에 영향을 미쳐야 하는지 명확하지 않기 때문입니다. 특히 방 배치가 텍스처가 없을 때, 방 배치에 가까운 물체는 학습 중에 어려움을 겪을 수 있으며, 이로 인해 잘못된 최적화 및 불필요한 3D Gaussian으로 이어질 수 있습니다. 이러한 문제를 해결하기 위해 본 연구는 Segment Anything Model (SAM)을 사용하여 기본 요소 선택을 안내합니다. SAM 마스크 손실은 각 인스턴스가 Gaussian 또는 메쉬로 표현되도록 강제하여 명확한 분리 및 안정적인 학습을 보장합니다. 또한 표준 밀집화 후 불투명도를 재설정하지 않는 추가 밀집화 단계를 도입했습니다. 이 단계는 표준 밀집화 후 3D Gaussian의 수가 제한되어 발생하는 이미지 품질 저하를 완화합니다. 



### Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality (https://arxiv.org/abs/2407.16171)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 연구에서는 **단일 모달이 누락된 경우에도 강력한 성능을 유지하는 새로운 AVQA 프레임워크**를 제안합니다. 기존 AVQA 모델은 오디오 또는 비주얼 정보가 누락되면 성능이 크게 저하되는 문제가 있었는데, 이 연구에서는 **관계 인식 모달 생성기(RMM)**와 **오디오-비주얼 관계 인식 확산 모델(AVR)**을 활용하여 누락된 모달 정보를 재구성하고 정확한 답변을 제공하는 방식을 제시합니다. 이를 통해 실제 환경에서 발생할 수 있는 모달 누락 문제에 대한 해결책을 제시합니다.



### 3D-UGCN: A Unified Graph Convolutional Network for Robust 3D Human Pose Estimation from Monocular RGB Images (https://arxiv.org/abs/2407.16137)
Comments:
          Proceedings of IEEE AICON2024

- **What's New**: 본 논문은 단일 시점 비디오에서 누락된 인간 자세 골격 시퀀스 문제를 해결하기 위해 공간-시간 그래프 합성 네트워크(UGCN) 기반의 개선된 방법을 제안합니다. 3D 인간 자세 데이터를 처리하고 3D 인간 자세 골격 시퀀스를 개선할 수 있는 개선된 UGCN을 제시하여 폐색 문제를 해결합니다.



### Open-Set Biometrics: Beyond Good Closed-Set Models (https://arxiv.org/abs/2407.16133)
Comments:
          Published at ECCV 2024

- **What's New**: 이 논문은 오픈셋 바이오메트릭(open-set biometrics) 에서의 성능을 향상시키기 위한 새로운 손실 함수(loss function)를 제안합니다. 오픈셋 바이오메트릭은 탐색 대상(probe)이 갤러리(gallery)에 존재할 수도 있고 존재하지 않을 수도 있는 실제 응용 분야에서 중요한 문제입니다. 기존의 손실 함수들은 탐색 대상과 갤러리의 유사성 점수(similarity score)를 대칭적으로 처리하여 오픈셋 평가와 일치하지 않습니다. 이 논문에서는 새로운 손실 함수인 (1) identification-detection loss와 (2) relative threshold minimization을 제안하여 오픈셋 성능을 향상시키고 탐색 대상과 갤러리의 유사성 점수 차이를 더 잘 반영합니다.



### FoRA: Low-Rank Adaptation Model beyond Multimodal Siamese Network (https://arxiv.org/abs/2407.16129)
- **What's New**: 이 논문에서는 기존 MCQ 평가 지표의 단점을 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 지표를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지를 측정하는 지표입니다.  KDA는 학생 설문 조사를 통해 측정되며,  KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 방식을 모방하여 자동으로 계산됩니다. 

- **Technical Details**: KDA는 학생 설문 조사를 통해 측정되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 방식을 모방하여 자동으로 계산됩니다.  KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보여줍니다. 

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 n-gram 기반 유사도 지표와 결합하여 전문가가 평가한 다양한 MCQ 품질 지표에 대한 예측력을 높여줍니다. 



### Advancing Brain Imaging Analysis Step-by-step via Progressive Self-paced Learning (https://arxiv.org/abs/2407.16128)
Comments:
          miccai-2024

- **What's New**: 이 논문은 Brain imaging 분석을 위한 새로운 Curriculum Learning 프레임워크인 Progressive Self-Paced Distillation (PSPD)를 제안합니다. PSPD는 모델의 과거 상태와 현재 상태를 모두 활용하여 적응적이고 점진적인 학습 과정을 통해 generalization 능력을 향상시키고 기존 지식을 잊는 것을 방지합니다.



### MxT: Mamba x Transformer for Image Inpainting (https://arxiv.org/abs/2407.16126)
- **What's New**: 이 논문은 이미지 인페인팅 (image inpainting)을 위한 새로운 모델 MxT를 제안합니다. MxT는 Mamba와 Transformer를 결합한 하이브리드 모듈을 사용하여 픽셀 수준과 패치 수준에서의 상호작용 학습을 가능하게 합니다. 이는 이미지를 고품질로 복원하고 컨텍스트 정확도를 향상시키는 데 도움이 됩니다.



### Diffusion Prior-Based Amortized Variational Inference for Noisy Inverse Problems (https://arxiv.org/abs/2407.16125)
Comments:
          ECCV 2024; 41 pages, 19 figures

- **What's New**: 이 논문은 교육적 가치를 고려하는 새로운 MCQ 자동 평가 지표인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'를 제안합니다. 기존 지표들은 단어 유사도에만 집중했지만, KDA는 MCQ가 학생의 지식을 제대로 평가할 수 있는지 측정합니다. 이를 위해, 연구진은 학생들의 응답 데이터를 사용하여 KDA를 측정하고, 사전 훈련된 언어 모델을 활용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 개발했습니다.



### Fr\'echet Video Motion Distance: A Metric for Evaluating Motion Consistency in Videos (https://arxiv.org/abs/2407.16124)
- **What's New**: 본 논문에서는 비디오 생성 모델에서의 모션 일관성 (temporal and motion consistency)을 평가하기 위한 새로운 메트릭인 Fréchet Video Motion Distance (FVMD)를 제안합니다. 기존 FID-VID, FVD, VBench와 같은 메트릭은 비디오의 시각적 품질 또는 일관성을 평가하지만, 복잡한 모션 패턴을 평가하는 데는 부족함이 있었습니다. FVMD는 key point tracking을 통해 명시적인 모션 feature를 추출하고, Fréchet distance를 이용하여 feature 간의 유사성을 측정합니다. 이를 통해 생성된 비디오의 움직임이 실제 물리 법칙에 맞게 자연스러운지 여부를 평가합니다.



### Augmented Efficiency: Reducing Memory Footprint and Accelerating Inference for 3D Semantic Segmentation through Hybrid Vision (https://arxiv.org/abs/2407.16102)
Comments:
          18 pages, 3 figures, 3 tables

- **What's New**: 본 논문에서는 2D semantic segmentation 기술을 3D semantic segmentation에 적용하여 모델의 효율성과 경량화를 꾀하는 새로운 방법을 제안합니다. 3D semantic segmentation은 memory와 latency 측면에서 제약이 있는데, 이를 해결하기 위해 2D와 3D 기술을 결합하여 3D semantic segmentation을 효율적으로 수행합니다. 먼저 RGB 이미지에 대한 2D semantic segmentation을 수행하고, 이 결과를 3D point cloud로 확장합니다. 이를 통해 3D point cloud의 subspace를 줄이고, 효율적인 3D semantic segmentation을 수행합니다. DeepViewAgg 모델을 기반으로 3D point cloud 전체에 대한 실험을 진행하고, IoU (Intersection over Union) 정확도, inference time latency, memory consumption을 측정했습니다.



### PLayerTV: Advanced Player Tracking and Identification for Automatic Soccer Highlight Clips (https://arxiv.org/abs/2407.16076)
- **What's New**: 본 논문에서는 축구 영상에서 선수 추적 및 식별을 위한 혁신적인 프레임워크인 PlayerTV를 제안합니다. PlayerTV는 객체 감지 및 추적, 광학 문자 인식 (OCR), 색상 분석을 통합하여 축구 영상에서 선수별 하이라이트 클립을 자동으로 생성합니다. PlayerTV는 전통적으로 이러한 작업과 관련된 수작업을 크게 줄일 수 있습니다.

- **Technical Details**: PlayerTV는 객체 감지 및 추적, OCR, 색상 분석을 사용하여 축구 영상에서 선수를 자동으로 추적하고 식별합니다. 이러한 기술을 통합하여 PlayerTV는 광범위한 경기 영상에서 선수별 하이라이트 클립을 생성할 수 있습니다.

- **Performance Highlights**: 노르웨이 Eliteserien 리그 데이터셋에서 수행한 PlayerTV 핵심 파이프라인 평가의 예비 결과는 PlayerTV가 팀과 선수를 정확하고 효율적으로 식별할 수 있음을 나타냅니다. PlayerTV는 사용자 친화적인 그래픽 사용자 인터페이스 (GUI)를 제공하여 사용자가 PlayerTV 기능을 쉽게 사용할 수 있도록 지원합니다.



### Pavement Fatigue Crack Detection and Severity Classification Based on Convolutional Neural Network (https://arxiv.org/abs/2407.16021)
Comments:
          10 pages, 14 figures, 3 tables

- **What's New**: 이 논문은 아스팔트 포장 도로의 균열을 분류하기 위한 새로운 딥 컨볼루션 신경망 (Deep Convolutional Neural Network) 을 제안합니다. 이 신경망은 두 가지 목표를 달성합니다: 첫째, 포장 표면 이미지를 기반으로 피로 균열(Fatigue cracking, 또는 악어 균열)의 존재를 분류합니다. 둘째, 손상 식별 매뉴얼 (Distress Identification Manual, DIM) 표준을 기반으로 피로 균열의 심각도 수준을 분류합니다.



### EfficientCD: A New Strategy For Change Detection Based With Bi-temporal Layers Exchanged (https://arxiv.org/abs/2407.15999)
- **What's New**: 이 논문에서는 효율적이고 정확한 원격 감지 영상 변화 탐지(CD)를 위해 새롭고 효율적인 딥러닝 프레임워크인 EfficientCD를 제안합니다. 이 프레임워크는 특징 추출을 위해 EfficientNet을 백본 네트워크로 사용합니다.  EfficientCD는 또한 ChangeFPN(Change Feature Pyramid Network)이라는 새로운 모듈을 도입하여 양쪽 시간 영상 특징 맵(bi-temporal image feature map) 간의 정보 교환을 향상시킵니다. 또한, 디코딩 단계에서 다층 특징 맵을 최대한 활용하기 위해 레이어별 특징 업샘플링 모듈을 유클리드 거리와 결합하여 디코딩 단계에서 특징 융합 및 재구성을 개선했습니다.



### FDWST: Fingerphoto Deblurring using Wavelet Style Transfer (https://arxiv.org/abs/2407.15964)
Comments:
          Accepted by IJCB 2024

- **What's New**: 본 논문에서는 지식 종속 가능성(KDA)이라는 새로운 자동 평가 지표를 제안하여 MCQ의 대답 가능성(answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생들의 반응을 통해 계산되며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용하여 학생들의 문제 해결 능력을 모방하여 KDA를 추정합니다. 

- **Technical Details**: KDA는 학생들의 반응을 통해 계산되며, KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 이용하여 학생들의 문제 해결 능력을 모방하여 KDA를 추정합니다. 

- **Performance Highlights**: 본 논문에서 제안된 KDA_disc와 KDA_cont는 human evaluation을 통해 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었으며, n-gram 기반 유사성 지표와 결합하여 다양한 전문가가 평가한 MCQ 품질 척도에 대한 높은 예측력을 보여줍니다.



### Test-Time Low Rank Adaptation via Confidence Maximization for Zero-Shot Generalization of Vision-Language Models (https://arxiv.org/abs/2407.15913)
Comments:
          Main paper: 11 pages, Supplementary material: 5 pages

- **What's New**: 이 논문에서는 대규모 비전-언어 모델 (VLMs) 의 제로-샷 일반화를 위한 프롬프트 튜닝 대신 테스트 시간 낮은 순위 적응 (TTL) 을 제안합니다. TTL은 큰 언어 모델의 효율적인 미세 조정의 최근 발전에서 영감을 받아 예측 신뢰도를 극대화하여 트랜스포머 인코더의 주의 가중치를 업데이트하는 테스트 시간 매개변수 효율적인 적응 접근 방식을 제공합니다. 자기 감독 신뢰도 극대화 목표는 보강된 샘플의 예측 간 일관성을 강제하는 가중 엔트로피 손실을 사용하여 지정됩니다. TTL은 프롬프트와 백본을 동결시키면서 모델 공간에서 낮은 순위 어댑터에 대해 소량의 학습 가능한 매개변수만 소개합니다. 다양한 자연 분포 및 교차 도메인 작업에 대한 광범위한 실험 결과, TTL은 엄격한 제로-샷 설정에서 VLM의 테스트 시간 최적화를 위해 다른 기술을 능가할 수 있음을 보여줍니다. 특히, TTL은 평균적으로 상당한 개선을 통해 테스트 시간 프롬프트 튜닝 기준선을 능가합니다.



### Craft: Cross-modal Aligned Features Improve Robustness of Prompt Tuning (https://arxiv.org/abs/2407.15894)
Comments:
          15pages

- **What's New**: 이 논문에서는 자동 MCQ 생성 평가 메트릭으로 지식 종속 가능성(KDA)을 제안하여 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 자동으로 KDA를 측정하는 새로운 메트릭입니다. Human evaluation 결과, KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 높은 상관관계를 보이며, n-gram 기반 유사성 메트릭과 함께 사용하면 전문가가 평가한 MCQ 품질 척도를 예측하는 데 높은 정확성을 보여줍니다.



### CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models (https://arxiv.org/abs/2407.15886)
Comments:
          10 pages, 9 figures, 4 tables

- **What's New**: 기존의 Virtual Try-on (VTON) 방법들은 ReferenceNet 또는 추가적인 이미지 인코더를 사용하여 이미지 처리를 수행하여 높은 트레이닝 및 추론 비용을 발생시켰다. 본 논문에서는 ReferenceNet과 이미지 인코더의 필요성을 재고하고, CatVTON이라는 간단하고 효율적인 VTON diffusion model을 제안하여 옷과 사람의 상호 작용을 혁신한다. CatVTON은 옷과 사람 이미지를 공간 차원으로 연결하여 입력으로 받아, 어떤 종류의 옷이든 목표 대상에 쉽게 적용할 수 있다.



### A Novel Method to Improve Quality Surface Coverage in Multi-View Captur (https://arxiv.org/abs/2407.15883)
Comments:
          submitted version 1

- **What's New**: 본 논문에서는 3D 메쉬와 카메라 위치 정보를 활용하여 표면 면적을 최적화하는 새로운 방법을 제안한다. 이 방법은 EM 알고리즘과 k-view 알고리즘을 통해 각 카메라의 초점 거리를 결정하여 표면 커버리지를 최적화한다. 기존의 단일 뷰 방식에 비해 표면 면적을 1550 cm^2 및 1780 cm^2 증가시키며, 비용을 각각 24%와 28% 절감하는 효과를 보인다.



### BSH for Collision Detection in Point Cloud models (https://arxiv.org/abs/2407.15852)
- **What's New**: 이 논문은 대규모 포인트 클라우드 모델을 위한 새로운 충돌 감지 알고리즘을 제시합니다. 이 알고리즘은 폭셀(voxel), 옥트리(octree), 경계 구(Bounding Sphere) 계층 구조(BSH)를 사용하여 충돌을 감지합니다. 이 알고리즘은 먼저 장면을 폭셀로 분할하고 각 폭셀의 객체를 옥트리로 구성합니다. 그런 다음, 옥트리의 각 비어 있지 않은 셀(cell)을 R-tree 계층 구조와 같은 구조를 기반으로 하는 경계 구 계층 구조로 구성합니다. BSH 계층 구조는 인접한 포인트를 그룹화하고 다른 모델과 상호 작용하지 않는 객체 부분을 매우 빠르게 필터링하는 데 사용됩니다.  이 알고리즘은 레이저 스캔 데이터에서 파생된 포인트가 일반적으로 분할되지 않고 임의의 공간 해상도를 가질 수 있어 계산 및 모델링 문제를 발생시키는 문제를 해결합니다.  



### A Survey on Trustworthiness in Foundation Models for Medical Image Analysis (https://arxiv.org/abs/2407.15851)
- **What's New**: 이 논문은 의료 영상 분석에서의 기반 모델(foundation models) 신뢰성에 대한 새로운 서베이를 제공합니다. 기반 모델의 신뢰성은 의료 분야에서 매우 중요하며 개인 정보 보호, 견고성, 신뢰성, 설명 가능성, 공정성 등을 포함합니다. 이 논문은 특히 의료 영상 세분화, 의료 보고서 생성, 의료 질문과 답변 (Q&A), 질병 진단 등 다양한 응용 분야에서 기반 모델의 신뢰성에 대한 연구를 검토하고 분석합니다.



### AutoRG-Brain: Grounded Report Generation for Brain MRI (https://arxiv.org/abs/2407.16684)
- **What's New**: 본 논문은 의료 영상 보고서 자동 생성 시스템인 AutoRG-Brain을 제안하며, 뇌 MRI 해석을 지원하는 시스템으로 뇌 구조 분할, 이상 부위 국재화, 잘 정돈된 소견 생성을 지원합니다. 이 시스템은 뇌 MRI 데이터셋 RadGenome-Brain MRI를 공개하여 연구 및 개발을 촉진합니다. AutoRG-Brain은 픽셀 수준의 시각적 단서를 기반으로 하는 최초의 뇌 MRI 보고서 생성 시스템입니다. 또한, 정량적 평가와 사람에 의한 평가를 통해 시스템의 신뢰성과 정확성을 입증했습니다.



### Velocity Driven Vision: Asynchronous Sensor Fusion Birds Eye View Models for Autonomous Vehicles (https://arxiv.org/abs/2407.16636)
- **What's New**: 본 연구에서는 기존의 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)의 단점을 보완하여 새로운 지표인 "Knowledge Dependent Answerability (KDA)"를 제안합니다. 이 지표는 MCQ가 대상 사실에 대한 학생의 지식을 평가할 수 있는지 여부를 측정합니다. 또한, KDA를 자동으로 측정할 수 있는 두 가지 새로운 지표, "KDA_disc"와 "KDA_cont"를 제안합니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.



### Knowledge-driven AI-generated data for accurate and interpretable breast ultrasound diagnoses (https://arxiv.org/abs/2407.16634)
- **What's New**: 본 논문은 희귀 케이스의 정확도를 높이기 위해 장기 분포 데이터를 사용하는 진단 모델 성능 향상을 위한 새로운 파이프라인, TAILOR를 제안합니다. TAILOR는 지식 기반 생성 모델을 활용하여 희귀 케이스를 위한 맞춤형 합성 데이터를 생성합니다. 특히, 3,749개의 병변을 소스 데이터로 사용하여 수백만 개의 유방 초음파 이미지를 생성하며, 특히 오류 발생률이 높은 희귀 케이스에 대한 데이터 생성에 초점을 맞춥니다. 생성된 데이터는 정확하고 해석 가능한 진단을 위한 진단 모델을 구축하는 데 사용됩니다. 



### Deep Bayesian segmentation for colon polyps: Well-calibrated predictions in medical imaging (https://arxiv.org/abs/2407.16608)
Comments:
          comments are welcome. 43 pages

- **What's New**: 이 논문은 자동 MCQ 생성을 위한 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. 기존의 BLEU, ROUGE, METEOR 등은 MCQ의 교육적 가치를 고려하지 않고 단어 유사도만 비교했던 반면, KDA는 MCQ의 대답 가능성 (answerability)을 측정하여 학생의 지식을 평가하는 능력을 평가합니다. KDA는 인간 설문 조사를 통해 측정되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 학생들의 문제 해결 행동을 모방하여 자동으로 KDA를 측정합니다.



### Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization (https://arxiv.org/abs/2407.16554)
Comments:
          9pages, 3figures. This paper has been accepted for ACM MM 2024

- **What's New**: 본 논문에서는 오디오 편집 탐지 및 위치 지정을 위해 프레임 레벨 탐지 네트워크(FDN)와 제안 개선 네트워크(PRN)를 통합한 새로운 '거친-미세 제안 개선 프레임워크(CFPRF)'를 제안합니다. FDN은 실제 및 위조 프레임 사이의 차이를 통해 위조 영역을 대략적으로 알려주는 정보를 얻고, PRN은 FDN에서 파생된 거친 제안을 개선하기 위해 신뢰도 점수와 회귀 오프셋을 예측합니다.



### Accelerating Learned Video Compression via Low-Resolution Representation Learning (https://arxiv.org/abs/2407.16418)
- **What's New**: 이 논문은 MCQ 생성 모델의 교육적 가치를 측정하는 새로운 평가 지표인 지식 의존성 대답 가능성(KDA)을 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표들은 생성된 MCQ와 기존 MCQ의 단어 유사성만을 측정했지만, KDA는 학생의 지식을 평가할 수 있는 능력을 측정하는 데 초점을 맞춥니다. KDA는 학생 설문조사를 통해 계산할 수 있으며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 자동으로 KDA를 추정합니다.



### Low Complexity Regularized Phase Retrieva (https://arxiv.org/abs/2407.16413)
- **What's New**: 본 논문은 MCQ 생성의 교육적 가치를 고려하는 새로운 자동 평가 지표 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 답변 가능성을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA는 학생 응답을 활용하여 계산되며, 이를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.



### On Differentially Private 3D Medical Image Synthesis with Controllable Latent Diffusion Models (https://arxiv.org/abs/2407.16405)
- **What's New**: 본 논문에서는 3차원 심장 MRI 이미지 생성을 위한 새로운 개념의 차별적 프라이버시 (Differential Privacy, DP) 기반 확산 모델을 제안합니다. 이 모델은 개인 정보 보호를 보장하면서 의료 속성에 따라 합성 이미지를 생성합니다. 이는 3차원 의료 이미지 생성에 DP를 적용하고 정량화한 최초의 연구입니다.

- **Technical Details**: 본 연구에서는 잠재 확산 모델 (LDMs)을 사용하여 공개 데이터로 사전 훈련을 수행하고, UK Biobank 데이터셋에 DP를 적용하여 미세 조정을 수행했습니다. 3차원 이미지를 처리하기 위해 기존 2차원 LDM 아키텍처를 3차원으로 확장했고, 의료 속성에 따라 이미지 생성을 조절할 수 있도록 교차 주의 메커니즘 (cross-attention mechanism)을 추가했습니다. DP-SGD를 통해 DP를 구현했으며, 압축 모델은 공개 데이터로 훈련하고, 노이즈 예측 네트워크만 DP로 훈련했습니다.

- **Performance Highlights**: 실험 결과, 사전 훈련은 모델 성능을 크게 향상시켰으며, ϵ=10에서 프리셰 인셉션 디스턴스 (FID) 26.77을 달성했습니다. 사전 훈련이 없는 모델의 FID는 92.52였습니다. 또한, DP를 적용한 훈련을 통해 생성된 합성 심장 MRI 이미지의 품질이 상당히 향상되었지만, 의학적 현실성을 일관되게 유지하는 데는 여전히 어려움이 있다는 사실을 확인했습니다.



### Improving multidimensional projection quality with user-specific metrics and optimal scaling (https://arxiv.org/abs/2407.16328)
Comments:
          10 Pages, 4 figures

- **What's New**: 본 연구에서는 사용자의 선호도를 반영하는 맞춤형 다차원 투영(MP) 프레임워크를 제안합니다. 기존 MP 방법과 달리 개별 사용자의 품질 기준을 고려하여 투영 해석성을 향상시키는 것을 목표로 합니다. 제안된 프레임워크는 스트레스, 이웃 보존, 실루엣 점수 등 세 가지 시각적 품질 지표를 통합하여 MP 평가를 위한 복합 지표를 생성합니다. 이후 복합 지표 값을 극대화하여 투영 규모를 최적화합니다. 실험은 두 명의 사용자를 대상으로 진행되었으며, 각 사용자는 다른 투영 선호도를 가지고 있습니다. 두 명의 사용자는 t-SNE, UMAP, LAMP를 사용하여 투영을 생성하고, 자신의 기준에 따라 투영을 평가하여 두 개의 학습 집합을 생성합니다. 각 집합에 대한 최적 가중치를 도출하고 다른 데이터 세트에 적용하여 사용자별 최적 투영을 결정합니다.



### Understanding Impacts of Electromagnetic Signal Injection Attacks on Object Detection (https://arxiv.org/abs/2407.16327)
Comments:
          2024 IEEE International Conference on Multimedia and Expo (ICME), July 15 - July 19, 2024, Niagra Falls, Ontario, Canada

- **What's New**: 본 논문은 이미지 센서의 하드웨어 특징을 악용하여 전자기 간섭 (electromagnetic interference)을 주입함으로써 이미지를 조작하는 사이버 물리 공격 (cyber-physical attacks)이 최첨단 객체 탐지 모델에 미치는 영향을 분석하고 정량화합니다. 이는 실제 애플리케이션에서 캡처된 이미지에 영향을 미치는 다양한 요인을 고려한 연구입니다.



### Deep Learning for Pancreas Segmentation: a Systematic Review (https://arxiv.org/abs/2407.16313)
- **What's New**: 이 논문은 췌장 분할을 위한 딥 러닝 모델의 최근 발전을 포괄적으로 다루는 체계적인 검토를 제공합니다. 기존 연구에서 간과된 연구를 포함하여 2013년부터 2023년까지 출판된 130개의 연구를 검토했습니다.

- **Technical Details**: 이 연구는 PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-analyses) 지침을 따릅니다. PubMed, Web of Science, Scopus 및 IEEE Xplore 데이터베이스에서 관련 연구를 수집했습니다. 연구는 췌장 실질, 종양, 낭포 및 염증 분할에 중점을 두었습니다. 흔히 사용되는 네트워크 아키텍처, 공개적으로 사용 가능한 데이터 세트, 평가 지표 및 손실 함수를 요약했습니다.

- **Performance Highlights**: 본 검토는 췌장 분할을 위한 딥 러닝 기술의 발전을 체계적으로 분석하여 췌장 분할 기술의 현재 상태를 요약합니다. 다양한 연구에서 사용된 딥 러닝 아키텍처, 데이터 세트, 학습 전략 및 손실 함수를 비교하여 췌장 분할의 과제와 향후 연구 방향을 제시합니다. 특히 췌장 종양 및 염증 분할에 대한 최신 연구 동향을 소개하며, 임상 적용을 위한 미래 방향을 제시합니다.



### EffiSegNet: Gastrointestinal Polyp Segmentation through a Pre-Trained EfficientNet-based Network with a Simplified Decoder (https://arxiv.org/abs/2407.16298)
Comments:
          To be published in IEEE Engineering in Medicine and Biology (EMBC) 2024 conference proceedings

- **What's New**: 본 논문에서는 효율적인 세그먼테이션 프레임워크인 EffiSegNet을 소개합니다. 이 프레임워크는 사전 훈련된 CNN 분류기를 백본으로 사용하는 전이 학습을 활용합니다. EffiSegNet은 기존의 대칭 U-shape 아키텍처와 달리 디코더를 간소화하고 전체 규모의 특징 융합을 사용하여 계산 비용과 매개변수 수를 최소화합니다. 



### Probabilistic Parameter Estimators and Calibration Metrics for Pose Estimation from Image Features (https://arxiv.org/abs/2407.16223)
Comments:
          Accepted at DASC '24. 9 pages, 4 figures

- **What's New**: 이 논문에서는 실시간 측정 불확실성이 있는 상황에서 확률적 파라미터 추정 문제를 다룹니다. 특히, 자율 비주얼 착륙 시스템을 위한 자세 추정(pose estimation)에 적용합니다. 세 가지 확률적 파라미터 추정기를 제시합니다: 최소 제곱 샘플링 접근 방식, 선형 근사 방법, 확률적 프로그래밍 추정기. 이 추정기를 평가하기 위해 다변량 정규 분포를 위한 보정(calibration)과 선명도(sharpness)를 측정하는 새로운 폐쇄형 공식을 소개합니다. 실험 연구에서는 다양한 잡음 조건에서 세 가지 추정기를 비교합니다. 선형 근사 추정기가 다른 방법보다 훨씬 빠르게 선명하고 잘 보정된 자세 예측을 생성할 수 있지만 특정 시나리오에서는 과도한 자신감 예측을 할 수 있다는 것을 보여줍니다. 또한, 이러한 추정기를 칼만 필터(Kalman filter)와 통합하여 활주로 접근 중 지속적인 자세 추정을 할 수 있음을 보여주었습니다. 여기서 보정은 유지하면서 선명도가 50% 향상되었습니다. 이 연구는 데이터 기반 컴퓨터 비전 모델을 복잡한 안전 중요 항공 시스템에 통합하는 데 기여하고 이러한 시스템에 대한 엄격한 인증 지침을 개발하기 위한 기반을 제공합니다.



### Pixel Embedding: Fully Quantized Convolutional Neural Network with Differentiable Lookup Tab (https://arxiv.org/abs/2407.16174)
- **What's New**: 이 논문은 입력 데이터를 벡터로 변환하는 Pixel Embedding이라는 새로운 기법을 제안하여 첫 번째 컨볼루션 계층에서의 양자화(quantization) 손실을 줄이는 방법을 소개합니다. 이는 자연어 처리 분야의 단어 임베딩(word embedding)에서 영감을 받았습니다. Pixel Embedding은 입력 픽셀을 룩업 테이블(lookup table)을 사용하여 양자화된 값들의 벡터로 대체합니다. 이 룩업 테이블은 미분 가능하며 역전파(backpropagation)를 통해 학습될 수 있습니다.



### Advanced AI Framework for Enhanced Detection and Assessment of Abdominal Trauma: Integrating 3D Segmentation with 2D CNN and RNN Models (https://arxiv.org/abs/2407.16165)
Comments:
          6 Pages

- **What's New**: 본 연구는 복부 외상 진단을 위한 인공지능 기반 모델을 제시합니다. 3D 세분화, 2D 합성곱 신경망 (CNN), 순환 신경망 (RNN)을 결합하여 복부 CT 스캔을 실시간으로 처리하여 정확한 진단을 제공합니다. 이 모델은 기존 진단 방법보다 뛰어난 성능을 보여주며 외상 진단의 자동화에 새로운 기준을 제시합니다.



### Representation Magnitude has a Liability to Privacy Vulnerability (https://arxiv.org/abs/2407.16164)
Comments:
          Accepted in the AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2024

- **What's New**: 본 논문은 기존의 MCQ 생성 평가 지표인 BLEU, ROUGE, METEOR가 MCQ의 교육적 가치를 고려하지 않고, 단순히 생성된 MCQ와 기존 샘플의 유사성을 비교한다는 문제점을 지적합니다.  따라서, 본 논문은 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 이라는 새로운 자동 평가 지표를 제안하여, MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. KDA는 실제 학생들의 응답 데이터를 기반으로 측정될 수 있으며, 논문에서는 이를 근사화하기 위한 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델 (pre-trained language model)을 사용하여 학생들의 문제 해결 방식을 모방하는 방식으로 설계되었습니다.  인간 평가를 통해 KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성 간의 높은 상관관계를 보여주었으며, 기존의 n-gram 기반 유사성 지표와 결합했을 때 전문가가 평가한 다양한 MCQ 품질 지표를 예측하는 능력이 우수한 것으로 나타났습니다.



### Cross-Domain Separable Translation Network for Multimodal Image Change Detection (https://arxiv.org/abs/2407.16158)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 다양한 센서 유형의 이미지 변화를 추적할 수 있는 멀티 모달 변화 감지 (MCD)에 대한 새로운 방법을 제시한다. 특히, 서로 다른 스타일과 통계적 특징을 가진 지형 공간 객체에서 이미지를 비교하는 데 어려움을 겪는 기존 MCD 방법의 한계를 극복하기 위해, 이 논문은 영역 내 자기 재구성(self-reconstruction)과 영역 간 이미지 변환 및 사이클 재구성(cycle-reconstruction) 워크플로우를 변화 감지 제약 조건(constraints)과 통합하는 새로운 비지도 교차 영역 분리 변환 네트워크(CSTN)를 제안한다. 모델은 이미지 변환과 MCD 작업을 동시에 수행하여 멀티 모달 이미지에서 학습된 특징의 비교 가능성을 보장하도록 최적화된다. 특히, 단순하면서도 효율적인 이중 분기 합성곱 아키텍처를 사용하여 멀티 모달 이미지의 콘텐츠와 스타일 정보를 분리한다. 이 과정은 스타일과 무관한 콘텐츠 비교 가능한 특징 공간을 생성하여 센서 변화가 심한 경우에도 정확한 변화 감지를 달성하는 데 중요하다. 광범위한 실험 결과는 제안된 방법의 효과를 보여주며, MCD의 정확성과 효율성 측면에서 최첨단 방법보다 뛰어난 성능을 보여준다.



### Improved Few-Shot Image Classification Through Multiple-Choice Questions (https://arxiv.org/abs/2407.16145)
- **What's New**: 이 논문은 기존의 VQA 모델을 이용하여 이미지 분류를 위한 새로운 few-shot 학습 방법을 제안합니다. 이 방법은 다양한 시각적 특징을 가진 데이터셋에서 기존의 few-shot 학습 방법보다 뛰어난 성능을 보여줍니다.



### Diffusion Models as Optimizers for Efficient Planning in Offline RL (https://arxiv.org/abs/2407.16142)
Comments:
          The paper was accepted by ECCV2024

- **What's New**: 이 논문은 MCQ 생성의 교육적 가치를 평가하는 새로운 지표인 지식 종속 가능성(KDA)을 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표들은 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 고려하지 않습니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 평가 능력을 평가하는 데 도움을 줄 수 있습니다. 본 연구에서는 Human Evaluation을 통해 KDA_disc와 KDA_cont가 실제 교육 환경에서 사용성과 높은 상관관계를 가지고 있음을 확인했습니다. 또한, n-gram 기반 유사도 지표와 결합하여 여러 전문가가 평가한 MCQ 품질 지표를 예측하는 데 효과적임을 보여줍니다.



### LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies (https://arxiv.org/abs/2407.16067)
Comments:
          ICML 2024 Oral Presentation; Project Page: this https URL

- **What's New**: 본 논문은 OOD (Out-of-Distribution) 데이터를 사용하지 않고 ID (In-Distribution) 측정을 통해 모델의 OOD 성능을 예측하는 새로운 방법을 제안합니다. 기존의 "Effective Robustness" 평가 방식은 다양한 supervision 및 distribution (예: 이미지넷의 Vision Model (VM), LAION의 Visual-Language Model (VLM)의 클래스 레이블과 텍스트 설명)으로 학습된 모델에 한계를 보였습니다. 특히 VLM은 VM과 비슷하거나 낮은 ID 성능에도 불구하고 OOD 데이터에 대해 더 잘 일반화됩니다. 본 논문에서는 이러한 문제를 해결하기 위해 LCA-on-the-Line 프레임워크를 도입하여 ID 측정으로부터 모델의 OOD 성능 예측을 개선합니다. 이 방법은 WordNet과 같은 사전 정의된 클래스 계층 구조 내에서 레이블과 예측 간의 계층적 거리를 측정하는 LCA (Lowest Common Ancestor) 거리 개념을 재해석합니다.



### Wallcamera: Reinventing the Wheel? (https://arxiv.org/abs/2407.16015)
- **What's New**: 이 논문은 자동 MCQ 생성을 위한 새로운 자동 평가 지표인 **지식 종속 가능성(Knowledge Dependent Answerability, KDA)**를 제안한다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가할 수 있는 능력을 측정한다. 기존의 평가 지표들은 MCQ의 교육적 가치를 고려하지 않고 단어 유사성만 비교했지만, KDA는 MCQ가 학생의 지식을 실제로 평가하는 능력을 평가한다.



### Memory Management for Real-Time Appearance-Based Loop Closure Detection (https://arxiv.org/abs/2407.15890)
Comments:
          6 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2407.15304

- **What's New**: 이 논문에서는 대규모 장기간 SLAM (Simultaneous Localization and Mapping) (동시 위치 인식 및 매핑) 에서 실시간 루프 클로저 탐지를 위한 새로운 접근 방식을 제시합니다. 이 방법은 각 새 관찰에 대한 계산 시간을 고정된 한계 내로 유지하는 메모리 관리 방식을 기반으로 합니다.

- **Technical Details**: 본 논문에서는 새로 획득된 위치를 Working Memory (WM)에 저장하고 가장 자주 관찰되는 위치를 Long-Term Memory (LTM)으로 옮겨 메모리를 관리하는 새로운 접근 방식을 제시합니다. 이 방법은 로봇의 이동 속도와 위치 획득률에 따라 Short-Term Memory (STM) 크기를 고정하고, STM의 크기 제한에 도달하면 가장 오래된 위치를 WM으로 이동합니다. 이는 WM에서 일정한 수의 키 위치를 유지하고, 실시간 제약 조건을 충족시키면서 전체 맵의 위치에 액세스할 수 있도록 합니다. 또한, 위치가 WM에서 LTM으로 이동되면 해당 위치가 더 이상 루프 클로저 탐지에 사용되지 않습니다. 그러나 루프 클로저가 감지되면 인접 위치를 가져와 WM으로 되돌려 루프 클로저 탐지에 고려할 수 있습니다.

- **Performance Highlights**: 이 논문에서는 네 개의 표준 데이터 세트를 사용하여 실험을 수행했으며, 다양한 조건에서 제안된 접근 방식의 실시간 성능을 보여줍니다. 특히 실시간 제약 조건을 충족하면서 매핑된 환경의 크기에 관계없이 실시간 성능을 유지한다는 것을 보여줍니다.



### Shapley Pruning for Neural Network Compression (https://arxiv.org/abs/2407.15875)
- **What's New**: 본 논문은 기존의 딥러닝 모델의 robusteness가 부족한 문제를 해결하기 위해 contrastive learning과 counterfactual augmentation을 사용한 새로운 접근 방식을 제안합니다. 기존 방식들은 사람이 직접 counterfactual을 추가하거나 모델이 데이터셋에서 유사한 counterfactual을 찾는 방식을 사용했지만, spurious correlation의 영향을 받았습니다. 본 논문에서는 “여러 개의” counterfactual을 합성하고, 집합적 의사 결정 (collective decisions)을 통해 단어들의 인과관계를 더 robust하게 파악하는 방법을 제안합니다. 



### CIC: Circular Image Compression (https://arxiv.org/abs/2407.15870)
- **What's New**: 이 논문은 **Circular Image Compression (CIC)** 라는 새로운 압축 방법을 제안하며, 기존의 **Serial Image Compression (SIC)** 방식의 한계를 극복하고 **Closed-loop** 구조를 활용하여 훈련과 테스트 이미지 사이의 차이를 줄여서 성능을 향상시킵니다.  특히 **Out-of-sample**, **Out-of-distribution**, 또는 **Out-of-domain** 테스트 이미지에서도  성능 저하를 최소화 합니다. 

- **Technical Details**: CIC는 **Nonlinear loop equation** 을 설정하여 재구성된 이미지와 원본 이미지 간의 **Steady-state error** 를 Taylor series expansion을 통해 0에 가깝게 만드는 방식을 사용합니다. 또한 **Post-Training** 및 **Plug-and-play** 특성을 가지므로 기존의 첨단 SIC 방법에 쉽게 적용 가능합니다.

- **Performance Highlights**: 5개의 공개 이미지 압축 데이터셋에서 CIC는 기존의 5개 SIC 알고리즘보다 **재구성 능력** 측면에서 우수한 성능을 보였습니다. 특히 어두운 배경, 뚜렷한 가장자리, 높은 대비, 그리드 모양, 복잡한 패턴을 가진 **Out-of-sample** 테스트 이미지에서도 효과적인 결과를 보였습니다.



### Adversarial Attacks and Defenses on Text-to-Image Diffusion Models: A Survey (https://arxiv.org/abs/2407.15861)
- **What's New**: 이 논문은 텍스트-이미지 확산 모델의 견고성(robustness)과 안전성(safety)에 대한 심층적인 분석을 제공하며, 이를 위한 다양한 적대적 공격(adversarial attack)과 방어(defense) 방법들을 살펴봅니다. 특히, 기존 연구에서 간과되었던 문법적으로 부정확한 프롬프트(grammatically incorrect prompt)에 대한 견고성 강화와 악의적인 프롬프트(malicious prompt)에 대한 안전성 강화에 초점을 맞춥니다. 또한, 기존 공격 및 방어 방법의 한계점을 분석하고 잠재적인 해결책을 논의합니다.  



### AutoAD-Zero: A Training-Free Framework for Zero-Shot Audio Description (https://arxiv.org/abs/2407.15850)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문은 영상에 대한 오디오 설명(AD)을 생성하기 위해 VLMs(Vision-Language Model)과 LLMs(Large Language Model)을 사용하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 훈련이 필요하지 않고, VLM을 이용하여 영상에 나오는 등장인물을 파악하고, LLM을 이용하여 상세한 설명을 요약하여 AD 문장을 생성합니다. 특히, TV 시리즈에 대한 AD 생성을 위한 새로운 데이터셋을 제시합니다.



### BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes (https://arxiv.org/abs/2407.15848)
Comments:
          SIGGRAPH 2024 Conference Papers. Project page: this https URL

- **What's New**: 본 논문은 MVS 기반 NeRF 모델의 렌더링 품질을 향상시키는 새로운 방법인 BoostMVSNeRFs를 제안합니다. 기존 MVS 기반 NeRF 모델의 제한된 뷰포트 커버리지와 부족한 입력 뷰로 인한 아티팩트 문제를 해결하기 위해, 본 논문은 렌더링 과정에서 여러 개의 비용 볼륨을 선택하고 결합하는 새로운 방법을 제안합니다. BoostMVSNeRFs는 별도의 학습 없이도 뛰어난 렌더링 품질 향상을 제공하며, 특정 장면에 대한 미세 조정 (fine-tuning)을 통해 더욱 향상된 성능을 보입니다. 대규모 데이터셋에서의 실험 결과를 통해, BoostMVSNeRFs가 대규모 장면과 무한한 야외 환경에서 렌더링 품질을 크게 향상시킨다는 사실이 확인되었습니다. BoostMVSNeRFs의 소스 코드는 [이곳](https://github.com/google/nerfies)에서 공개됩니다.



### HandDGP: Camera-Space Hand Mesh Prediction with Differentiable Global Positioning (https://arxiv.org/abs/2407.15844)
Comments:
          To be presented at ECCV 2024

- **What's New**: 본 논문은 단일 RGB 이미지에서 카메라 공간 손 메시(camera-space hand meshes)를 예측하는 데 있어, 기존의 2단계 방식(손 이미지를 자르고 상대 좌표계에서 메시를 예측한 후 카메라 공간으로 변환)의 한계를 극복하기 위해 2D-3D 대응 문제(correspondence problem)를 해결하는 end-to-end 솔루션을 제안합니다. 이는 카메라 공간 출력에서 네트워크의 다른 부분으로 역전파(back-propagation)를 가능하게 하는 새로운 미분 가능한 전역 위치 모듈(differentiable global positioning module)을 사용합니다. 또한, 동일한 카메라로 촬영된 것처럼 학습 데이터셋과 입력 이미지를 조화시키는 이미지 정정 단계(image rectification step)를 도입하여 문제의 고유한 척도-깊이 모호성(scale-depth ambiguity)을 완화합니다.



### CarFormer: Self-Driving with Learned Object-Centric Representations (https://arxiv.org/abs/2407.15843)
Comments:
          Accepted to ECCV 2024, code and the pre-trained models can be found at this https URL

- **What's New**: 본 논문에서는 MCQ 생성의 교육적 가치를 고려하는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 평가합니다. 특히, KDA를 Human Survey를 통해 측정하는 방법을 보여주고, Pre-trained Language Model을 활용하여 KDA를 근사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제시합니다. Human Evaluation 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 상황에서의 사용성과 강한 상관관계를 보입니다. 또한, n-gram 기반 유사성 메트릭과 결합하여 KDA_disc와 KDA_cont는 전문가가 평가한 다양한 MCQ 품질 측정 지표에 대한 강력한 예측력을 보여줍니다.



### Artist: Aesthetically Controllable Text-Driven Stylization without Training (https://arxiv.org/abs/2407.15842)
Comments:
          WIP,webpage: this https URL

- **What's New**: 이 논문은 MCQ 생성의 교육적 가치를 평가하는 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안한다. KDA는 MCQ의 대답 가능성을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다. 이 논문은 KDA를 계산하는 방법과 함께 인공지능 모델을 활용하여 학생의 문제 해결 행동을 모방한 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안한다. 실제 교실 환경에서의 KDA_disc와 KDA_cont의 유용성에 대한 연구를 통해 이들이 KDA 및 전문가가 평가한 MCQ 품질 측정 지표와 강한 상관관계를 갖는 것을 보여주었다.  



### SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models (https://arxiv.org/abs/2407.15841)
Comments:
          Technical report

- **What's New**: SF-LLaVA (SlowFast-LLaVA)라는 새로운 training-free Video LLM이 제안되었습니다. SF-LLaVA는 SlowFast 아키텍처를 사용하여 영상의 공간적 의미 (spatial semantics)와 장기적 시간적 맥락 (long-range temporal context)을 동시에 포착합니다. 이는 Slow pathway는 저속 프레임 (low frame rate)에서 고해상도 공간적 정보를 추출하는 반면, Fast pathway는 고속 프레임 (high frame rate)에서 저해상도 공간적 정보를 추출하여 움직임 정보 (motion cues)에 집중하는 방식으로 구현됩니다. 이를 통해 SF-LLaVA는 기존 training-free 방법들보다 다양한 영상 작업 (Open-Ended VideoQA, Multiple Choice VideoQA, Text Generation)에서 더 우수한 성능을 보입니다. 특히, 일부 벤치마크에서는 영상 데이터셋에서 fine-tuned된 최첨단 Video LLM과 비슷하거나 더 나은 성능을 달성합니다.



### MMInstruct: A High-Quality Multi-Modal Instruction Tuning Dataset with Extensive Diversity (https://arxiv.org/abs/2407.15838)
Comments:
          18 pages, 8 figures, technical report

- **What's New**: 이 논문은 VLLM (Vision Large Language Model)의 성능을 향상시키기 위해 고품질의 다양한 시각적 지시 사항(visual instruction) 튜닝 데이터셋인 MMInstruct를 제안합니다. 기존 데이터셋의 한계점 (제한된 이미지 다양성, 불완전한 주석, 제한적인 지시 유형)을 해결하기 위해, MMInstruct는 24개 도메인에서 973K 개의 지시를 포함합니다. MMInstruct는 GPT-4V와 GPT-3.5를 활용하여 반자동으로 저렴하게 지시를 생성하는 데이터 엔진을 사용합니다. 또한 다양한 질문 유형 (판단, 객관식, 긴 시각적 질문 답변, 짧은 시각적 질문 답변)과 다양한 도메인 (인식, 추론, 다중 라운드 긴 시각적 질문 답변)을 포함합니다.



### Towards Latent Masked Image Modeling for Self-Supervised Visual Representation Learning (https://arxiv.org/abs/2407.15837)
- **What's New**: 본 연구는  "지식 종속 가능성(KDA)" 라는 새로운 자동 평가 지표를 제안하여 MCQ 생성의 교육적 가치를 평가합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제시하며, 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 설정에서의 사용성 모두와 강한 상관관계를 보이는 것으로 나타났습니다.



### Accelerating Pre-training of Multimodal LLMs via Chain-of-Sigh (https://arxiv.org/abs/2407.15819)
- **What's New**: 본 논문에서는 Chain-of-Sight라는 비전-언어 브릿지 모듈을 소개하여 멀티모달 대규모 언어 모델 (MLLM)의 사전 훈련을 가속화하는 방법을 제시합니다. 이 방법은 다양한 공간 규모(spacial scales)에서 시각적 세부 정보를 포착하는 시각적 리샘플러(visual resampler) 시퀀스를 사용합니다. Chain-of-Sight는 사전 훈련 단계에서 시각 토큰(visual token)의 수를 줄여 사전 훈련 시간을 최대 73% 단축하며, 훈련 후 토큰 수를 최대 16배까지 늘릴 수 있는 복합 토큰 스케일링 전략(compound token scaling strategy)을 통해 시각적 맥락을 효과적으로 활용합니다.



### Efficient and generalizable prediction of molecular alterations in multiple cancer cohorts using H&E whole slide images (https://arxiv.org/abs/2407.15816)
- **What's New**: 이 논문에서는 H&E 염색 이미지로부터 여러 DNA 변이를 동시에 예측하기 위해 다중 작업 학습을 사용하여 모델을 훈련시킨 새로운 접근 방식을 제안합니다. 이는 단일 슬라이드에서 여러 가지 실행 가능한 예측을 제공하는 임상적으로 유용한 알고리즘을 개발하는 데 유망한 방법입니다.



### Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budg (https://arxiv.org/abs/2407.15811)
Comments:
          41 pages, 28 figures, 5 tables

- **What's New**: 본 논문에서는 이미지 생성 모델(Text-to-Image, T2I) 학습의 비용을 줄이기 위해 새로운 방법을 제안합니다. 기존의 대규모 이미지 생성 모델은 막대한 계산 자원을 필요로 하여 연구 및 개발에 있어 진입 장벽이 높았습니다. 본 논문에서는 이미지 패치의 일부를 마스킹하여 학습 과정의 계산 비용을 크게 줄이는 방법을 제시합니다. 특히, 패치 믹서(patch-mixer)를 사용하여 마스킹 전에 모든 패치를 사전 처리하는 방식으로, 기존의 모델 축소(model downscaling) 방식보다 효과적으로 비용을 절감하면서 성능 저하를 최소화합니다. 또한, 혼합 전문가 계층(mixture-of-experts layers) 등 최신 트랜스포머 아키텍처 개선 사항을 도입하여 성능을 향상시키고, 합성 이미지를 사용하는 것의 중요성을 강조합니다.  결과적으로, 11억 개의 매개변수를 가진 스파스 트랜스포머 모델을 3700만 개의 실제 및 합성 이미지를 사용하여 1,890 달러의 저렴한 비용으로 학습시켰으며, COCO 데이터셋에서 12.7 FID를 달성했습니다. 이는 Stable Diffusion 모델보다 118배, 최첨단 모델보다 14배 저렴한 비용으로 경쟁력 있는 성능을 달성한 것입니다. 본 논문의 목표는 이러한 연구 결과를 바탕으로 저렴한 비용으로 대규모 확산 모델을 학습할 수 있는 훈련 파이프라인을 공개하여 이미지 생성 모델 연구 및 개발의 접근성을 높이는 것입니다.



### Breaking the Global North Stereotype: A Global South-centric Benchmark Dataset for Auditing and Mitigating Biases in Facial Recognition Systems (https://arxiv.org/abs/2407.15810)
Comments:
          This work has been accepted for publication at AAAI/ACM AIES 2024

- **What's New**: 이 연구는 전 세계 8개국 출신 운동선수 6,579명의 얼굴 데이터셋을 공개하여, 특히 글로벌 사우스(Global South) 국가들의 데이터 부족 문제를 해결하고자 합니다. 데이터셋의 50% 이상이 글로벌 사우스 출신이며, 다양한 인구 통계학적 특징을 반영합니다. 또한, 각 이미지에 4가지의 적대적 변형 (adversarial variants)을 추가하여 총 40,000장 이상의 이미지를 제공하여, 적대적 감사(adversarial audits)와 강력한 모델 훈련에 도움을 줍니다. 연구진은 5가지 인기 있는 얼굴 인식 시스템(FRS) (상업용 및 오픈 소스)을 사용하여 성별 예측 (그리고 예시로 한 오픈 소스 모델에 대한 국가 예측) 작업을 수행했습니다.



### Robust Facial Reactions Generation: An Emotion-Aware Framework with Modality Compensation (https://arxiv.org/abs/2407.15798)
- **What's New**: 본 논문에서는 MCQ 생성 평가 지표로 '지식 종속 가능성(KDA)'을 제안한다. KDA는 MCQ의 답변 가능성을 측정하여 학생이 해당 대상 사실에 대한 지식을 얼마나 잘 평가할 수 있는지 측정하는 새로운 지표이다. 기존의 BLEU, ROUGE, METEOR와 같은 지표는 단어 유사성에만 초점을 맞추었지만, KDA는 MCQ의 교육적 가치를 측정한다.



### MILAN: Milli-Annotations for Lidar Semantic Segmentation (https://arxiv.org/abs/2407.15797)
- **What's New**: 본 연구는 자율 주행 시스템에 사용되는 라이더 포인트 클라우드 데이터를 효율적으로 주석 처리하는 새로운 방법을 제안합니다. 특히, 최근 발전한 자기 지도 학습(self-supervised learning) 기반 라이더 스캔 표현 모델을 활용하여 주석 비용을 크게 줄일 수 있다는 것을 보여줍니다.



### AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection (https://arxiv.org/abs/2407.15795)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문에서는 MCQ 생성의 교육적 가치를 평가하는 새로운 자동 평가 메트릭인 '지식 종속 가능성(KDA)'를 제안합니다. 기존 메트릭들은 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 고려하지 않고 단순히 골드 샘플과의 유사성만을 측정했지만, KDA는 학생의 지식을 기반으로 MCQ의 대답 가능성을 측정합니다.  KDA를 자동화하기 위해 두 가지 새로운 메트릭(KDA_disc, KDA_cont)을 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다.



### Disentangling spatio-temporal knowledge for weakly supervised object detection and segmentation in surgical video (https://arxiv.org/abs/2407.15794)
Comments:
          13 pages, 6 figures, 8 tables

- **What's New**: 본 논문은 약한 감독 비디오 객체 분할(WSVOS)을 위한 새로운 프레임워크인 비디오 공간-시간 분리 네트워크(VDST-Net)를 제안합니다. VDST-Net은 반-분리된 지식 증류(semi-decoupled knowledge distillation)를 사용하여 고품질 클래스 활성화 맵(CAMs)을 예측하고, 비디오에서 객체의 위치와 타이밍에 대한 명확한 정보가 제공되지 않을 때 발생하는 시간적 충돌(temporal conflicts)을 해결하도록 설계된 교사 네트워크(teacher network)와 시간 의존성을 활용하여 시간 정보를 통합하는 학생 네트워크(student network)로 구성됩니다. 



### CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning (https://arxiv.org/abs/2407.15793)
Comments:
          15 pages, 1 figure. Accepted at the The 35th British Machine Vision Conference 2024 (BMVC 2024), Glasgow, UK

- **What's New**: 본 논문에서는 기존의 MCQ 평가 지표인 BLEU, ROUGE, METEOR 등이 MCQ의 교육적 가치를 고려하지 않고 단순히 문장 유사도만 측정하는 문제점을 지적하고, MCQ의 대답 가능성(answerability)을 측정하여 교육적 가치를 반영하는 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안한다. KDA는 학생들의 응답을 기반으로 계산되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표로 구현되었다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화한다. 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 설정에서의 사용성과 높은 상관관계를 가지고 있음을 보여주었다. 또한, n-gram 기반 유사도 지표와 함께 사용하면 다양한 전문가가 평가한 MCQ 품질 지표에 대한 예측력이 높아지는 것으로 나타났다.



### RADA: Robust and Accurate Feature Learning with Domain Adaptation (https://arxiv.org/abs/2407.15791)
- **What's New**: 자동 MCQ 생성 분야에서 교육적 가치를 고려한 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 사람의 평가 결과와 높은 상관관계를 보이며, 실제 강의실 환경에서의 사용성과도 높은 상관관계를 보입니다. 또한, 기존 n-gram 기반 유사성 메트릭과 결합했을 때, KDA_disc와 KDA_cont는 전문가가 평가한 다양한 MCQ 품질 척도에 대한 예측력이 뛰어납니다.



### Unsupervised Mastoidectomy for Cochlear CT Mesh Reconstruction Using Highly Noisy Data (https://arxiv.org/abs/2407.15787)
- **What's New**: 본 논문에서는  '지식 종속 가능성(KDA)' 이라는 새로운 평가 지표를 제안합니다. 이 지표는 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생들의 답변을 통해 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 개발되어 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.  



### Towards Open-World Object-based Anomaly Detection via Self-Supervised Outlier Synthesis (https://arxiv.org/abs/2407.15763)
Comments:
          35 pages, 21 figures, includes supplementary material, accepted at ECCV 2024

- **What's New**: 본 논문에서는 오픈월드(open-world) 환경에서 객체 수준의 이상 탐지를 위한 새로운 방법을 제안합니다. 기존의 객체 수준 이상 탐지 방법은 특정 클래스의 정보에 의존하는 반면, 본 논문에서 제안하는 방법은 클래스 정보 없이도 이상 탐지를 수행할 수 있다는 장점이 있습니다. 이는 가상 이상치 합성(virtual outlier synthesis)을 통해 오픈월드 객체 검출기와 이상 탐지기를 결합하는 새로운 접근 방식을 활용하기 때문입니다. 가상 이상치 합성은 객체 검출기 백본의 특징을 사용하여 자기 지도 학습(self-supervision)을 통해 객체의 의사 클래스(pseudo-classes)를 학습하는 데 사용됩니다. 이러한 의사 클래스는 이상 탐지 헤드(OoD head)에 의해 분류되는 이상 특징의 클래스 조건부 가상 이상치 샘플링(class-conditional virtual outlier sampling)을 위한 기반 역할을 합니다. 따라서 본 논문의 접근 방식은 클래스 레이블에 의존하지 않고도 이상 인식 특징 표현을 학습하여 오픈월드 객체 이상 탐지를 가능하게 합니다.



### Diffusion for Out-of-Distribution Detection on Road Scenes and Beyond (https://arxiv.org/abs/2407.15739)
Comments:
          ECCV 2024 - Benchmark page: this https URL

- **What's New**: 본 논문은 기존의 MCQ 평가 메트릭이 교육적 가치를 고려하지 않고, 단순히 문장의 유사성만 평가한다는 문제를 지적하며 새로운 메트릭인 "지식 종속 가능성 (KDA)"를 제안합니다. KDA는 MCQ가 학생의 지식을 평가하는 능력을 측정합니다.



### Zero-Shot Embeddings Inform Learning and Forgetting with Vision-Language Encoders (https://arxiv.org/abs/2407.15731)
- **What's New**: 본 논문에서는 비전-언어 기반 모델의 미세 조정 (fine-tuning) 후 학습 및 잊어버림 (forgetting) 결과를 예측하는 새로운 방법인 IIMM (Inter-Intra Modal Measure)을 제안합니다. IIMM은 이미지 임베딩 간 유사성과 잘못된 이미지-레이블 임베딩 쌍 간 유사성을 측정하는 항을 결합하여 미세 조정 후 성능 변화를 강력하게 예측하는 지표로 사용됩니다.  



### Beyond Size and Class Balance: Alpha as a New Dataset Quality Metric for Deep Learning (https://arxiv.org/abs/2407.15724)
Comments:
          11 pages, 5 figures, 3 tables

- **What's New**: 본 논문에서는 의료 영상에서 딥 러닝 성능 향상을 위해 데이터셋 다양성을 측정하는 포괄적인 프레임워크를 제안합니다. 기존의 데이터셋 크기와 클래스 균형 대신, '큰 알파(big alpha)'로 표현되는 일반화된 엔트로피 측정 지표를 사용하여 이미지 간 유사성을 고려하여 데이터셋의 유효한 이미지-클래스 쌍 수를 측정합니다. 이 프레임워크는 데이터셋 크기와 클래스 균형을 특수한 경우로 포함합니다. 

- **Technical Details**: 본 연구는 다양한 의료 데이터셋 (초음파, X선, CT, 병리 이미지)에서 수천 개의 서브셋을 분석하여 성능과의 상관관계를 조사했습니다. 그 결과, 데이터셋 크기나 클래스 균형보다 '큰 알파' 측정 지표가 성능과 더 강한 상관관계를 보였습니다. 특히, A_0은 모든 서브셋에서 균형 정확도의 67%를 설명하는 반면, 클래스 균형은 54%, 크기는 39%를 설명했습니다. 가장 좋은 조합은 크기와 A_1 (79%)이었으며, 크기와 클래스 균형 (74%)보다 더 나은 성능을 보였습니다. '큰 알파' 측정 지표는 개별 데이터셋뿐만 아니라 모든 데이터셋에서 가장 우수한 성능을 보여주었으며, 이러한 결과의 일반성을 뒷받침합니다. 

- **Performance Highlights**: 본 연구는 '큰 알파' 측정 지표가 데이터셋 크기나 클래스 균형보다 딥 러닝 모델 성능과 더 강한 상관관계를 보임을 확인했습니다. 특히, A_0는 균형 정확도의 67%를 설명하는 반면, 클래스 균형은 54%, 크기는 39%를 설명했습니다. 이는 '큰 알파' 측정 지표가 의료 영상에서 딥 러닝 모델 성능 향상에 기여할 수 있음을 시사합니다. 



### GFE-Mamba: Mamba-based AD Multi-modal Progression Assessment via Generative Feature Extraction from MCI (https://arxiv.org/abs/2407.15719)
Comments:
          35 pages, 4 figures

- **What's New**: 이 논문은 Generative Feature Extraction (GFE) 기반의 새로운 분류기인 GFE-Mamba를 소개하며 이는 MCI에서 AD로의 전환을 예측하는 데 효과적입니다.



### Harmonizing Flows: Leveraging normalizing flows for unsupervised and source-free MRI harmonization (https://arxiv.org/abs/2407.15717)
- **What's New**: 본 논문은 MRI 이미지를 정규화하여 여러 기관 및 장비에서 획득된 MRI 이미지의 불균일성 문제를 해결하는 새로운 비지도 학습 기반 조화화 프레임워크를 제안합니다. 이 프레임워크는 정규화 흐름을 활용하여 소스 도메인의 분포를 모방합니다.



### Mamba meets crack segmentation (https://arxiv.org/abs/2407.15714)
Comments:
          32 pages, 8 figures. Preprint submitted to Elsevier

- **What's New**: 이 논문은 균열 분할 (crack segmentation) 모델에 Mamba를 사용하는 새로운 방법을 제안합니다. Mamba는 선형적인 공간 및 연산 복잡성과 강력한 전역 인식 (global perception) 능력으로 인해 주목받고 있습니다. 특히, 이 논문은 Mamba와 어텐션 메커니즘 (attention mechanism) 사이의 관계를 밝혀내고, 어텐션 블록의 원리를 따르는 새로운 Mamba 모듈인 CrackMamba를 개발합니다.



### SwinSF: Image Reconstruction from Spatial-Temporal Spike Streams (https://arxiv.org/abs/2407.15708)
- **What's New**: 본 논문은 스파이크 카메라에서 획득한 스파이크 스트림에서 동적인 장면을 재구성하기 위한 새로운 모델인 Swin Spikeformer (SwinSF)를 소개합니다. SwinSF는 스파이크 특징 추출, 공간-시간 특징 추출 및 최종 재구성 모듈로 구성되어 있으며, 이동 창 자기 주의(shifted window self-attention)와 제안된 시간 스파이크 주의(temporal spike attention)를 결합하여 공간 및 시간 역학을 포괄적으로 추출하여 스파이크 스트림을 더욱 강력하고 정확하게 재구성합니다.



### Predicting the Best of N Visual Trackers (https://arxiv.org/abs/2407.15707)
- **What's New**: 이 논문은 다양한 비디오 특성과 데이터셋에서 최첨단 시각 추적기의 성능이 놀랍도록 다르게 나타나는 것을 관찰합니다. 모든 추적 특성과 데이터셋에서 최고의 성능을 유지하는 단일 추적기는 없습니다. 이러한 차이를 해소하기 위해, 주어진 비디오 시퀀스에 대해 "N개의 추적기 중 최고"를 예측하는, BofN 메타 추적기를 제안합니다. 핵심적으로, 추적 성능 예측 네트워크(TP2N)는 초기 프레임 몇 개만 사용하여 주어진 비디오 시퀀스에 대해 예측된 최고 성능 시각 추적기를 선택합니다. 또한, 정기적인 시간 간격 후 최고의 성능을 예측하는 프레임 수준 BofN 메타 추적기를 소개합니다. TP2N은 자기 지도 학습 아키텍처인 MocoV2, SwAv, BT, DINO를 기반으로 하며, 실험 결과 ViT-S를 백본으로 사용하는 DINO가 가장 좋은 성능을 보입니다. 비디오 수준 BofN 메타 추적기는 LaSOT, TrackingNet, GOT-10K, VOT2019, VOT2021, VOT2022, UAV123, OTB100, WebUAV-3M 등 9개의 표준 벤치마크에서 기존 최첨단 추적기를 훨씬 능가합니다. 프레임 수준 BofN 메타 추적기는 긴 시퀀스 내 추적 시나리오의 변화를 효과적으로 처리하여 더욱 개선된 성능을 보입니다. 예를 들어 GOT-10k에서 BofN 메타 추적기의 평균 중첩은 비디오 수준 설정에서 각각 88.7% 및 프레임 수준 설정에서 91.1%입니다. 최고의 성능을 보이는 추적기인 RTS는 85.20% AO를 달성합니다. VOT2022에서 BofN의 예상 평균 중첩은 비디오 수준 설정에서 각각 67.88% 및 프레임 수준 설정에서 70.98%로, 최고의 성능을 보이는 ARTrack인 64.12%보다 높습니다. 이 연구는 또한 모든 일반적으로 사용되는 벤치마크에서 경쟁적인 추적 방법을 광범위하게 평가하고, 해당 프로토콜을 따릅니다. 코드, 훈련된 모델 및 결과는 곧 https URL에서 공개적으로 제공될 예정입니다.



### Multi-Modality Co-Learning for Efficient Skeleton-based Action Recognition (https://arxiv.org/abs/2407.15706)
- **What's New**: 본 논문에서는 skeleton 정보만 이용하는 기존 skeleton-based action recognition 모델의 제한적인 성능을 개선하기 위해 **multi-modality co-learning (MMCL)** 프레임워크를 제안합니다. MMCL은 **multimodal large language model (LLM)**을 활용하여 효율적인 skeleton-based action recognition을 가능하게 합니다. 특히, training 단계에서 multimodal 정보를 활용하여 co-learning을 수행하고 inference 단계에서는 간결한 skeleton 정보만 사용하여 효율성을 높입니다.



### Enhancing Transferability of Targeted Adversarial Examples: A Self-Universal Perspectiv (https://arxiv.org/abs/2407.15683)
Comments:
          8 pages and 9 figures

- **What's New**: 이 논문에서는 자율 학습(self-supervised learning)과 역추론 증강(counterfactual augmentation)을 활용하여 딥 러닝 모델의 강건성을 개선하는 새로운 접근 방식을 제시합니다. 기존의 증강 방법은 사람이 수동으로 데이터셋에 역추론을 추가하거나, 모델이 데이터셋에서 유사한 역추론을 찾아야 했지만, 여전히 허위 상관관계(spurious correlation)에 영향을 받는 문제가 있었습니다. 이 논문에서는 "여러 개의" 역추론을 합성하고, 집합적 의사 결정(collective decisions)을 통해 각 단어의 인과관계를 더 정확하게 파악하는 방법을 제안합니다.



### HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning (https://arxiv.org/abs/2407.15680)
Comments:
          Accepted as a main conference paper at ECCV 2024 (this https URL)

- **What's New**: 이 논문은 Vision-Language Model (VLM)의 환각 (hallucination) 문제를 해결하기 위해 HaloQuest라는 새로운 데이터셋을 소개합니다. HaloQuest는 실제 이미지와 합성 이미지를 모두 사용하여 다양한 환각 유발 시나리오를 포함하며, VLM의 환각 문제를 평가하고 완화하기 위한 새로운 벤치마크 역할을 합니다. 또한, VLM의 답변을 평가하는 새로운 자동 평가 (Auto-Eval) 메커니즘을 제안합니다.



### Flow-guided Motion Prediction with Semantics and Dynamic Occupancy Grid Maps (https://arxiv.org/abs/2407.15675)
Comments:
          Accepted for publication at the 27th IEEE International Conference on Intelligent Transportation Systems (ITSC) (ITSC 2024)

- **What's New**: 본 논문은  Occupancy Grid Maps (OGMs)를 기반으로 미래 자동차의 semantic grid와 flow를 예측하는 새로운 다중 작업 프레임워크를 제안합니다. 기존 방법은 scene의 진화 예측이나 복잡한 행동 학습에 집중했지만, scene의 흐름이나 속도 벡터 예측은 고려하지 않았습니다. 반면, 이 논문에서 제시된 프레임워크는 semantic flow 정보를 활용하여 warped semantic grid를 생성하여 scene의 dynamic vehicle을 더 정확하게 유지합니다.



### SLVideo: A Sign Language Video Moment Retrieval Framework (https://arxiv.org/abs/2407.15668)
Comments:
          5 pages, 3 figures, 1 table

- **What's New**: 본 논문은 수화 영상 검색을 위한 소프트웨어인 SLVideo를 소개한다. SLVideo는 손과 얼굴 둘 다를 인식하여 수화 비디오 검색을 가능하게 한다. 기존 시스템들은 수화 인식 알고리즘에 의존하지만, 얼굴 표정 인식은 포함하지 않는다는 한계가 있었다.  SLVideo는 손과 얼굴 표정을 모두 인식하여 수화의 표현력을 풍부하게 해주고, 문맥에 따라 의미가 변하는 수화를 정확하게 인식할 수 있도록 돕는다.



### MSSPlace: Multi-Sensor Place Recognition with Visual and Text Semantics (https://arxiv.org/abs/2407.15663)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 기존의 자동 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)가 단어 유사도에만 초점을 맞추고 교육적 가치를 고려하지 못한다는 점을 지적하며, MCQ의 대답 가능성 (answerability) 을 측정하는 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 대상 사실에 대한 학생의 지식을 평가하는 MCQ의 능력을 측정하는데 초점을 맞춥니다. 논문에서는 KDA를 계산하기 위한 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제시하며, 이러한 지표들이 실제 강의실 환경에서의 사용성과 높은 상관관계를 보인다는 것을 실험적으로 증명합니다.



### DriveDiTFit: Fine-tuning Diffusion Transformers for Autonomous Driving (https://arxiv.org/abs/2407.15661)
- **What's New**: 이 논문에서는 Diffusion Transformer (DiT) 를 미세 조정하여 자율 주행 데이터를 효율적으로 생성하는 새로운 방법인 DriveDiTFit을 제안합니다. DriveDiTFit은 미세 조정된 DiT의 생성 결과가 실제 데이터와 차이가 나는 부분들을 조절하여 DiT의 일부 파라미터만을 선택적으로 조정합니다. 또한, 날씨와 조명 조건을 나타내는 embedding module을 사용하여 다양한 환경의 데이터를 생성합니다.



### TreeSBA: Tree-Transformer for Self-Supervised Sequential Brick Assembly (https://arxiv.org/abs/2407.15648)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 3D 오브젝트를 조립하는 순차적인 조립 동작을 예측하기 위한 새로운 접근 방식을 제안합니다. 특히, 계층 간 연결을 고려하여 BFS (Breadth-First Search) 기반의 LEGO-Tree 구조를 사용하여 계산 효율성을 높였습니다. 또한, 클래스에 독립적인 트리 변환기 (tree-transformer) 프레임워크를 설계하여 다중 뷰 이미지로부터 순차적 조립 동작을 예측합니다.  실제 데이터에서 동작 라벨링의 어려움을 해결하기 위해 합성 데이터에서 실제 데이터로의 전이 학습 (transfer learning)을 활용합니다.  특히, 모델은 먼저 합성 데이터에서 완벽한 라벨링을 통해 사전 훈련됩니다. 그리고 실제 데이터에서 동작 라벨을 사용하지 않고 실루엣 프로젝션 (silhouette projection)을 사용하여 자기 지도 학습을 수행합니다.



### SS-SFR: Synthetic Scenes Spatial Frequency Response on Virtual KITTI and Degraded Automotive Simulations for Object Detection (https://arxiv.org/abs/2407.15646)
Comments:
          8 pages, 2 figures, 2 tables

- **What's New**: 본 논문은 자동 MCQ 생성의 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 대상 사실에 대한 학생의 지식 수준을 평가할 수 있도록 설계되었습니다. 기존 지표들은 MCQ의 텍스트 유사성에만 초점을 맞췄지만 KDA는 학생의 실제 답변을 고려하여 교육적 가치를 더 잘 반영합니다.



### Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models (https://arxiv.org/abs/2407.15642)
Comments:
          Project webpage: this https URL

- **What's New**: 이 논문은 이미지 애니메이션을 위한 새로운 방법인 Cinemo를 제안하여 더 나은 모션 제어, 일관성 및 부드러움을 제공합니다. Cinemo는 diffusion 모델을 기반으로 하며 세 가지 핵심 전략을 사용합니다: (1) 모션 잔차(motion residual) 분포를 학습하여 입력 이미지의 세부 사항을 유지합니다. (2) SSIM(Structural Similarity Index) 기반 전략을 사용하여 모션 강도를 미세하게 조절합니다. (3) DCTInit을 도입하여 초기 추론 노이즈를 개선하고 갑작스러운 모션 변화를 완화합니다.



### Reinforcement Learning Meets Visual Odometry (https://arxiv.org/abs/2407.15626)
- **What's New**: 이 논문은 Visual Odometry(VO)를 순차적 의사 결정 과제로 재구성하고, Reinforcement Learning(RL)을 적용하여 VO 프로세스를 동적으로 적응시키는 새로운 접근 방식을 제시합니다. 기존 VO 방법은 여러 주간의 수동적인 하이퍼파라미터 튜닝을 필요로 했지만, 이 방법은 VO 파이프라인 내에서 에이전트 역할을 하는 신경망을 도입하여 실시간 조건에 따라 키프레임 및 그리드 크기 선택과 같은 의사 결정을 내립니다. RL 프레임워크는 VO 시스템과 이미지 시퀀스를 환경으로 취급하며, 에이전트는 키포인트, 맵 통계 및 이전 포즈로부터 관측값을 받습니다.  



### Norface: Improving Facial Expression Analysis by Identity Normalization (https://arxiv.org/abs/2407.15617)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문은 딥러닝 기반의 얼굴 표현 분석(FEA) 분야에서 **타스크와 무관한 노이즈(task-irrelevant noise)**를 제거하기 위해 **Norface**라는 새로운 프레임워크를 제안합니다. Norface는 **정규화 네트워크(normalization network)**와 **분류 네트워크(classification network)**로 구성되며, **정규화 네트워크**는 얼굴 표현 일관성(expression consistency)를 유지하면서 모든 원본 이미지를 일관된 자세(pose)와 배경(background)을 가진 공통적인 신원(identity)으로 정규화하여 **타스크와 무관한 노이즈**를 제거합니다. **정규화된 이미지(normalized image)**는 **분류 네트워크(classification network)**에 입력되며, **분류 네트워크**는 **전문가 혼합(Mixture of Experts)**을 사용하여 잠재 표현(latent representation)을 개선하고 여러 AU(Action Unit) 또는 감정 라벨을 처리합니다.

- **Technical Details**: Norface는 **정규화 네트워크**와 **분류 네트워크**로 구성됩니다. **정규화 네트워크**는 **Masked AutoEncoder(MAE)**를 사용하여 얼굴 특징을 추출하고, **Expression Merging Module(EMM)**을 통해 원본 얼굴의 표현 특징을 대상 얼굴에 적용합니다. 또한, 표현 일관성을 유지하기 위해 **표현 손실(expression loss)**과 **눈썹 손실(eyebrow loss)**을 적용합니다. **분류 네트워크**는 **전문가 혼합(Mixture of Experts)**을 사용하여 입력과 출력에 대한 전문가를 활용하여 잠재 표현을 개선하고 다중 AU 또는 감정 라벨을 처리합니다.

- **Performance Highlights**: Norface는 **AU 감지(AU detection), AU 강도 추정(AU intensity estimation), 감정 인식(FER)** 등 다양한 얼굴 표현 분석 태스크에서 SOTA(State-of-the-Art) 성능을 보여줍니다. 특히 **데이터셋 간 전이(cross-dataset) 태스크**에서도 뛰어난 성능을 보이며, **타스크와 무관한 노이즈** 제거에 효과적임을 입증합니다.



### Visual-Semantic Decomposition and Partial Alignment for Document-based Zero-Shot Learning (https://arxiv.org/abs/2407.15613)
Comments:
          Accepted to ACM International Conference on Multimedia (MM) 2024

- **What's New**: 자동 MCQ 생성을 위한 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 또한, KDA를 근사화하기 위한 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안합니다.



### Probing Fine-Grained Action Understanding and Cross-View Generalization of Foundation Models (https://arxiv.org/abs/2407.15605)
- **What's New**: 자동 MCQ 생성 평가를 위한 새로운 지식 종속 가능성(KDA) 지표 제안. 기존의 BLEU, ROUGE, METEOR와 같은 지표들은 단어 유사성만 비교하지만, KDA는 학생의 지식 수준을 평가하는 MCQ의 능력을 측정한다. KDA를 근사화하는 두 가지 자동 지표, KDA_disc와 KDA_cont를 제안. 인간 연구를 통해 두 지표는 KDA와 실제 강의실에서의 사용성과 높은 상관관계를 보임을 확인. KDA_disc와 KDA_cont는 단어 유사성 지표와 함께 사용될 경우 다양한 전문가 평가 MCQ 품질 측정 지표에 대한 예측력이 높아짐을 보여준다.



### Learning Where to Look: Self-supervised Viewpoint Selection for Active Localization using Geometrical Information (https://arxiv.org/abs/2407.15593)
Comments:
this http URL

- **What's New**: 이 논문은 활성 위치 추정 (active localization) 분야를 탐구하며, 특히 위치 추정 정확도를 높이기 위한 시점 선택의 중요성을 강조합니다. 이 연구는 실시간 작동을 위해 설계된 간단한 아키텍처, 자기 지도 학습 데이터 훈련 방법, 실제 로봇 애플리케이션에 맞춤화된 계획 프레임워크에 지속적으로 맵을 통합하는 기능을 갖춘 데이터 기반 접근 방식을 사용합니다. 



### All rivers run into the sea: Unified Modality Brain-like Emotional Central Mechanism (https://arxiv.org/abs/2407.15590)
- **What's New**: 본 논문에서는 교육적 가치를 고려한 MCQ 생성 평가 메트릭인 KDA를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 제대로 평가하는지 여부를 판단합니다. 기존의 BLEU, ROUGE, METEOR와 같은 메트릭은 단어 유사성만을 비교했기 때문에 교육적 가치를 반영하지 못했지만, KDA는 학생 응답을 기반으로 MCQ의 질을 평가합니다. 또한, KDA를 자동으로 계산할 수 있는 KDA_disc와 KDA_cont 메트릭을 제안하며, 실제 강의실 환경에서의 사용성과 높은 상관관계를 보인다는 것을 실험적으로 입증했습니다. (The paper proposes a novel evaluation metric for MCQ generation called KDA, which considers educational value by measuring the answerability of an MCQ and assessing its ability to evaluate student knowledge. Existing metrics like BLEU, ROUGE, and METEOR only focus on word similarity, neglecting educational value. KDA, however, evaluates the quality of MCQs based on student responses. It also proposes automatic KDA metrics, KDA_disc and KDA_cont, which are shown to have strong correlations with usability in actual classroom settings through experiments.)



### Exploring the Effectiveness of Object-Centric Representations in Visual Question Answering: Comparative Insights with Foundation Models (https://arxiv.org/abs/2407.15589)
- **What's New**: 본 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하는 데 중점을 둡니다. 또한, KDA를 근사화하기 위해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안합니다. 

- **Technical Details**: KDA는 학생 설문 조사를 통해 얻은 학생 응답을 기반으로 측정됩니다. KDA_disc와 KDA_cont는 학생의 문제 해결 행동을 모방하도록 사전 훈련된 언어 모델을 활용하여 KDA를 근사화합니다. 

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성 모두와 강한 상관관계를 갖는다는 것을 보여주었습니다. 또한, N-gram 기반 유사성 지표와 결합하여, KDA_disc와 KDA_cont는 다양한 전문가가 라벨링한 MCQ 품질 측정에 대한 강력한 예측력을 보여줍니다.



### Not All Pairs are Equal: Hierarchical Learning for Average-Precision-Oriented Video Retrieva (https://arxiv.org/abs/2407.15566)
- **What's New**: 이 논문은 비디오 검색 작업을 위한 새로운 학습 프레임워크인 **HAP-VR (Hierarchical learning framework for Average-Precision-oriented Video Retrieval)**을 제안합니다. 이 프레임워크는 AP (Average Precision) 기반의 평가 지표를 효과적으로 최적화하여 기존 비디오 검색 방법들의 한계를 극복하고 성능을 향상시키는 데 목표를 두고 있습니다.



### Decomposition of Neural Discrete Representations for Large-Scale 3D Mapping (https://arxiv.org/abs/2407.15554)
Comments:
          ECCV 2024

- **What's New**: 본 논문은 3차원 신경 매핑에서 효율적인 로컬 특징 표현을 학습하는 새로운 방법인 분해 기반 신경 매핑(DNMap)을 제안합니다. DNMap은 각 이산 임베딩을 임베딩 공간 전체에서 공유되는 구성 요소 벡터로 분해하여 반복적이고 대표적인 모양 패턴을 효율적으로 포착하는 분해 전략을 사용합니다. 이 분해 전략을 통해 DNMap은 전체 이산 임베딩이 아닌 구성 요소 벡터 세트를 최적화하고 이산 임베딩 색인 대신 구성을 학습합니다. 또한 매핑 품질을 보완하기 위해 작은 저장 공간을 필요로 하는 저해상도 연속 임베딩을 추가로 학습합니다. DNMap은 이러한 표현을 얕은 신경망과 효율적인 옥트리 기반 특징 볼륨과 결합하여 부호화된 거리 함수를 성공적으로 근사하고 특징 볼륨을 압축하면서 매핑 품질을 유지합니다. 본 논문의 소스 코드는 [링크]에서 확인할 수 있습니다.



### Differentiable Product Quantization for Memory Efficient Camera Relocalization (https://arxiv.org/abs/2407.15540)
Comments:
          Accepted to the European Conference on Computer Vision (ECCV) 2024

- **What's New**: 이 논문에서는 3D 모델 기반 카메라 재현지 (relocalization) 시스템의 메모리 효율을 높이기 위한 새로운 방법을 제안합니다. 기존의 압축 방식은 3D 포인트를 제거하고 descriptor를 양자화 (quantization)하는 방식을 사용했지만, 정보 손실로 인해 성능 저하 문제가 발생했습니다. 이 연구에서는 scene-specific auto-encoder 네트워크를 사용하여 descriptor의 양자화-역양자화 (dequantization) 과정을 end-to-end differentiable하게 학습함으로써, product quantization centroid와 네트워크 파라미터를 동시에 업데이트하는 방식을 제안합니다. 또한 descriptor 재구성 뿐만 아니라, margin-based metric loss function을 통해 descriptor 매칭 성능을 유지하도록 네트워크를 학습시킵니다.  



### Double Deep Learning-based Event Data Coding and Classification (https://arxiv.org/abs/2407.15531)
- **What's New**: 이 논문에서는 이벤트 카메라 데이터의 효율적인 코딩 및 분류를 위한 새로운 이중 딥 러닝 아키텍처를 제안하며, 이벤트를 포인트 클라우드 기반으로 표현합니다. 이 아키텍처는 이벤트를 포인트 클라우드로 변환하고 다시 이벤트로 변환하는 과정을 포함하며, 이는 압축 및 분류 성능에 중요한 영향을 미칩니다. 



### Synthetic Image Learning: Preserving Performance and Preventing Membership Inference Attacks (https://arxiv.org/abs/2407.15526)
- **What's New**: 본 논문은 지식 재활용(Knowledge Recycling, KR) 파이프라인을 소개하여 의료 분야와 같은 데이터 부족 및 프라이버시 문제 해결에 도움을 주는 인공지능 기반 합성 데이터 생성 및 활용을 개선합니다. 이 파이프라인의 핵심은 생성 지식 증류(Generative Knowledge Distillation, GKD) 기술로, 합성 데이터셋 재생성 및 소프트 라벨링 메커니즘을 통해 분류기가 합성 데이터로부터 얻을 수 있는 정보의 질과 유용성을 향상시킵니다.



### SpotDiffusion: A Fast Approach For Seamless Panorama Generation Over Tim (https://arxiv.org/abs/2407.15507)
- **What's New**: 이 논문은 여러 개의 denoising predictions를 생성하고 평균을 내는 기존 방법의 비효율성을 해결하는 새로운 접근 방식을 제시합니다. 이 방법은 시간에 따라 겹치지 않는 denoising windows를 이동시켜 한 시간 단계에서 생긴 이음새(seam)를 다음 시간 단계에서 수정하여 더 적은 단계로 일관성 있는 고해상도 이미지를 생성합니다.



### WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation (https://arxiv.org/abs/2407.15502)
Comments:
          Accepted at ECCV 2024. The dataset and code can be accessed at this https URL

- **What's New**: 본 논문에서는 웹 디자인 자동화를 위한 새로운 작업인 웹 렌더링 매개변수 생성(WebRPG)을 소개한다. WebRPG는 HTML 코드를 기반으로 웹 페이지의 시각적 표현을 자동으로 생성하는 것을 목표로 한다. 이 작업은 더 빠른 웹 개발 워크플로우에 기여할 수 있다.



### TextureCrop: Enhancing Synthetic Image Detection through Texture-based Cropping (https://arxiv.org/abs/2407.15500)
Comments:
          17 pages, 6 images

- **What's New**: 이 논문은 자동 MCQ 생성을 위한 새로운 평가 지표인 지식 의존 가능성(Knowledge Dependent Answerability, KDA)를 제안합니다. KDA는 MCQ가 실제 학생의 지식을 평가하는 능력을 측정하기 위해, 대상 사실에 대한 지식을 고려하여 MCQ의 답변 가능성을 평가합니다. 기존의 BLEU, ROUGE, METEOR와 같은 평가 지표들은 단순히 생성된 MCQ와 데이터셋에 있는 골드 샘플의 n-gram 유사성에만 초점을 맞추었기 때문에 교육적 가치를 고려하지 못했습니다. 따라서 KDA는 MCQ의 실제 교육적 가치를 평가하는 데 유용한 도구가 될 것입니다.



### DiffX: Guide Your Layout to Cross-Modal Generative Modeling (https://arxiv.org/abs/2407.15488)
- **What's New**: 본 논문에서는 일반적인 레이아웃 기반의 크로스 모달 "RGB+X" 생성을 위한 새로운 확산 모델인 DiffX를 소개합니다. 특히, DiffX는 모달리티 공유 잠재 공간에서 확산 및 잡음 제거 프로세스를 수행하는 간단하지만 효과적인 크로스 모달 생성 모델 파이프라인을 제시하며, 이는 이중 경로 변이형 오토 인코더(DP-VAE)에 의해 가능해집니다. 또한, 레이아웃과 텍스트 조건을 연결하기 위해 게이트된 크로스 어텐션 메커니즘을 통합하여 긴 캡션을 임베딩하여 사용자 안내를 강화합니다. 긴 캡션 임베딩은 Long-CLIP을 활용합니다.



### In-Context Learning Improves Compositional Understanding of Vision-Language Models (https://arxiv.org/abs/2407.15487)
- **What's New**: 본 논문은 Vision-Language Models(VLMs)의 Compositional Image Understanding 능력을 향상시키기 위한 새로운 방법을 제안합니다. 이 방법은 In-Context Learning(ICL)을 활용하여 VLMs의 복잡한 추론 및 이미지 이해 능력을 향상시킵니다.



### 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Mod (https://arxiv.org/abs/2407.15484)
Comments:
          Project page: this https URL Accepted to ECCV 2024

- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS) 모델을 이용하여 target RGB 이미지의 카메라 포즈 (camera pose) 를 추정하는 6DGS 방법을 제안한다. 6DGS는 기존의 analysis-by-synthesis 방법 (예: iNeRF) 과 같이 반복적인 과정을 거치지 않으며, 카메라 포즈 초기화가 필요하지 않다. 6DGS는 3DGS 렌더링 과정을 반대로 이용하여 6DoF 포즈를 추정한다. 6DGS 모델을 구성하는 각 타원체 (ellipsoid) 에서 시작하여 균일하게 광선 (rays) 를 생성하는 Ellicell을 정의한다. 각 Ellicell 광선은 각 타원체의 렌더링 파라미터와 연결되며, 이는 target 이미지 픽셀과 생성된 광선을 연결하는 데 사용된다. 이 연결들은 점수를 기준으로 순위가 매겨지고, 가장 높은 점수를 가진 광선 묶음 (bundle of rays) 이 선택된다. 이 묶음의 교차점은 카메라 중심을 나타내고, 카메라 회전은 이를 이용하여 계산된다. 6DGS는 초기화를 위한 '사전' 포즈가 필요하지 않으며, 반복적인 과정 없이 6DoF 포즈 추정을 닫힌 형태로 해결한다.



### Diverse Image Harmonization (https://arxiv.org/abs/2407.15481)
- **What's New**: 이 논문에서는 이미지 조화 (image harmonization) 분야에서 기존 방법들의 한계를 극복하는 새로운 방법을 제안한다. 기존 방법들은 복합 이미지 (composite image)에 대해 단일한 조화 결과만을 생성했지만, 여러 가능한 반사율 (reflectance) 때문에 여러 가능한 조화 결과가 존재할 수 있다는 점을 간과했다. 이 논문에서는 반사율 기반 조화 네트워크 (reflectance-guided harmonization network)를 제안하여, 실제 반사율 정보를 이용하여 더 나은 성능을 달성한다. 또한, 여러 가능한 반사율을 예측하는 다양한 반사율 생성 네트워크 (diverse reflectance generation network)를 설계하여, 여러 가능한 조화 결과를 생성한다. 다양한 데이터셋에서 수행된 실험 결과들은 제안된 방법의 효과를 보여준다.



### Affordance Labeling and Exploration: A Manifold-Based Approach (https://arxiv.org/abs/2407.15479)
Comments:
          17 Pages, 3 Figures, 3 Tables

- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 이 논문은 KDA를 계산하는 방법을 제시하고, 사전 훈련된 언어 모델을 활용하여 KDA를 근사하는 두 가지 자동 평가 지표(KDA_disc 및 KDA_cont)를 소개합니다.  



### Learning deep illumination-robust features from multispectral filter array images (https://arxiv.org/abs/2407.15472)
- **What's New**: 본 논문에서는 raw MS 이미지에서 illumination-robust하고 discriminant한 특징을 학습하는 새로운 방법을 제안합니다. 이 방법은 raw spectral constancy를 활용하여 illumination의 영향을 최소화하고, MSFA-preserving 변환을 통해 다양한 raw texture를 학습하는 DNN을 training합니다. 또한, raw-mixing을 통해 raw 이미지에서 discriminant한 spatio-spectral interaction을 포착합니다.



### Domain-Adaptive 2D Human Pose Estimation via Dual Teachers in Extremely Low-Light Conditions (https://arxiv.org/abs/2407.15451)
Comments:
          18 pages, 3 figure. Accepted by ECCV24

- **What's New**: 이 논문은 극도로 어두운 조명 환경에서의 2D 인간 자세 추정 (human pose estimation)을 위한 새로운 도메인 적응 (domain adaptation) 방법을 소개합니다. 이 방법은 기존 방법과 달리 어두운 조명 환경에서의 데이터 라벨이 필요하지 않으며, 밝은 조명 환경에서 수집된 라벨 데이터만을 활용합니다. 특히, 이 연구는 두 가지 보완적인 교사 네트워크 (complementary-teacher networks) 를 활용하여 더 신뢰할 수 있는 의사 라벨 (pseudo labels) 을 생성하여 극도로 어두운 조명 환경에서도 뛰어난 성능을 달성합니다. 또한, 사람별 저조도 증강 (Person-specific Degradation Augmentation, PDA) 기술을 통해 학생 모델이 교사 모델보다 더 나은 성능을 발휘하도록 합니다. 



### SIGMA: Sinkhorn-Guided Masked Video Modeling (https://arxiv.org/abs/2407.15447)
Comments:
          Accepted at ECCV 24

- **What's New**: 이 논문은 기존의 영상 모델링 방법이 픽셀과 같은 낮은 수준의 정보를 재구성하는데 초점을 맞추어 고수준 의미를 포착하는 데 어려움을 겪는다는 점을 지적합니다. 이를 해결하기 위해 SIGMA (Sinkhorn-guided Masked Video Modelling)라는 새로운 영상 사전 학습 방법을 제안합니다. SIGMA는 영상 모델과 함께 투영 네트워크를 사용하여 목표 특징 공간을 학습합니다. 또한, L2 재구성 손실을 사용하면 단순한 수정으로 인해 두 네트워크가 동시에 최적화될 때 사소한 해결책으로 이어질 수 있는 문제를 해결하기 위해, SIGMA는 시공간 튜브의 특징을 제한된 수의 학습 가능한 클러스터에 균등하게 분산시키는 최적 전송 문제 (optimal transport problem)를 활용합니다. 이를 통해 생성된 특징의 엔트로피를 높이고, 특징 공간에 의미적 및 시간적 의미를 부여합니다. 생성된 클러스터 할당은 영상 모델이 투영 네트워크의 클러스터 할당을 예측하고 그 반대로도 예측하는 대칭적 예측 작업의 목표로 사용됩니다.



### Text2Place: Affordance-aware Text Guided Human Placemen (https://arxiv.org/abs/2407.15446)
Comments:
          ECCV 2024, Project Page: this https URL

- **What's New**: 자동 MCQ 생성 평가를 위한 새로운 메트릭인 KDA(Knowledge Dependent Answerability)를 제안했습니다. 기존 메트릭(BLEU, ROUGE, METEOR)은 단어 유사성만 고려했지만 KDA는 MCQ가 대상 사실에 대한 학생의 지식을 제대로 평가하는지 측정합니다. KDA를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 메트릭을 제안했습니다. 

- **Technical Details**: KDA는 인간 설문조사를 통해 학생들의 응답을 기반으로 측정됩니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보이는 것을 확인했습니다. 또한, n-gram 기반 유사성 메트릭과 함께 사용하면 다양한 전문가 평가 MCQ 품질 척도에 대한 예측력이 높아지는 것을 확인했습니다.



### Enhancement of 3D Gaussian Splatting using Raw Mesh for Photorealistic Recreation of Architectures (https://arxiv.org/abs/2407.15435)
- **What's New**: 이 논문은 3D Gaussian Splatting(3DGS)을 이용한 건축물의 3D 재구성에 기존의 3D 모델을 활용하는 새로운 방법을 제안합니다. 3DGS는 사진만으로 3D 모델을 생성하는 기술이지만, SfM(Structure-from-Motion) 프로세스를 통해 계산된 기하학적 파라미터에 크게 의존합니다. 이 논문에서는 3DGS의 초기 샘플링을 위한 추가적인 정보로써 3D 모델에서 추출한 메쉬를 활용합니다. 이를 통해 3DGS는 건축물의 기본적인 형태를 정확하게 파악하고, 사진이 체계적으로 촬영되지 않은 경우에도 더욱 시각적으로 풍부한 텍스처와 디테일을 생성할 수 있습니다.  

- **Technical Details**: 이 논문에서 제안하는 방법은 3DGS의 초기 샘플링을 위해 기존 3D 모델(coarse mesh)을 활용합니다. 3D 모델은 COLMAP으로 계산된 희소 포인트 클라우드와 정렬되어 3DGS에 대한 추가적인 형태 정보를 제공합니다. 이는 3DGS가 건축물의 기본적인 형태를 정확하게 파악할 수 있도록 돕고, 사진의 촬영 조건이 제한적인 경우에도 더욱 상세한 형태와 텍스처를 생성할 수 있도록 합니다.  

- **Performance Highlights**: 이 논문에서 제안하는 방법은 4개의 건축물 모델에 적용되었으며, vanilla 3DGS에 비해 형태와 텍스처 품질이 크게 향상되었습니다. 특히 사진 촬영 조건이 제한적인 경우에도 3DGS의 렌더링 품질을 크게 향상시킬 수 있음을 보여줍니다.



### Learning at a Glance: Towards Interpretable Data-limited Continual Semantic Segmentation via Semantic-Invariance Modelling (https://arxiv.org/abs/2407.15429)
- **What's New**: 이 논문은 지식 종속 가능성 (KDA)이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 이 메트릭은 기존의 BLEU, ROUGE, METEOR 등의 평가 메트릭과 달리 교육적 가치를 고려합니다.



### YOLO-pdd: A Novel Multi-scale PCB Defect Detection Method Using Deep Representations with Sequential Images (https://arxiv.org/abs/2407.15427)
- **What's New**: 본 논문은 PCB 결함 검출을 위한 고정밀, 강력하고 실시간 엔드-투-엔드 방식을 제안합니다. 기존 방식들은 낮은 정확도와 적용 가능성에 제한이 있었지만, 이 논문에서는 YOLOv5와 다중 스케일 모듈을 결합하여 계층적 잔차 연결 (hierarchical residual-like connections) 을 이용한 새로운 접근 방식을 제안합니다. 특히 YOLOv5 모델은 실시간 처리와 정확한 객체 검출 능력을 제공하며, 다중 스케일 모듈은 단일 블록 내에서 계층적 잔차 연결을 통합하여 다중 스케일 특징 추출을 가능하게 하여 다양한 크기와 복잡성의 결함을 식별하는 데 도움을 줍니다.  본 연구에서는 다중 스케일 아키텍처를 통해 특징 추출, 결함 위치 찾기 및 분류를 통합한 네트워크를 구축했습니다. 대규모 PCB 데이터셋을 사용한 실험 결과, 기존 방법들과 비교하여 정확도, 재현율 및 F1 점수가 크게 향상되었음을 보여줍니다. 본 연구는 PCB 결함 검출을 위한 컴퓨터 비전 검사를 발전시키고, PCB 제조 산업에서 고정밀, 강력하고 실시간이며 도메인 적응형 결함 검출을 위한 신뢰할 수 있는 솔루션을 제공합니다. 

- **Technical Details**: YOLOv5, 다중 스케일 모듈, 계층적 잔차 연결, 다중 스케일 특징 추출

- **Performance Highlights**: 기존 방법들과 비교하여 정확도, 재현율 및 F1 점수가 크게 향상됨을 보여줍니다.



### Bidirectional skip-frame prediction for video anomaly detection with intra-domain disparity-driven attention (https://arxiv.org/abs/2407.15424)
Comments:
          11 pages,7 figures, 4 tables

- **What's New**: 이 논문은 비디오 이상 감지 (VAD) 분야에서 일반적인 이벤트와 비정상 이벤트 간의 차이를 확대하여 성능을 향상시키기 위해 새로운 양방향 스킵 프레임 예측 (BiSP) 네트워크를 제안합니다. BiSP는 듀얼 스트림 오토인코더 기반으로 다양한 기능 간의 도메인 내 차이를 학습하는 관점에서 설계되었습니다.



### Local All-Pair Correspondence for Point Tracking (https://arxiv.org/abs/2407.15420)
Comments:
          ECCV 2024. Project page: this https URL Code: this https URL

- **What's New**: LocoTrack, a novel point tracking model that utilizes local 4D correlation for highly accurate and efficient point correspondence across video sequences.



### Chronologically Accurate Retrieval for Temporal Grounding of Motion-Language Models (https://arxiv.org/abs/2407.15408)
Comments:
          To appear at ECCV 2024. Project page: this https URL

- **What's New**: 이 논문은 3D 인간 모션과 언어 간의 시간적 정렬 (temporal alignment) 문제를 다룬다. 기존의 모션-언어 모델은 복합적인 행동(compound actions)에서 시간적 순서를 제대로 이해하지 못하는 경우가 많았다는 점을 지적한다. 이 문제를 해결하기 위해, 이 논문은 새로운 평가 지표인 CAR(Chronologically Accurate Retrieval)을 제안한다. CAR는 모션-언어 모델이 복합적인 행동에 대한 언어적 설명의 시간적 순서를 올바르게 이해하는지 평가한다. 또한, 시간적 정렬 성능을 향상시키기 위해, 이 논문은 모션-언어 모델 학습 과정에서 이벤트 순서를 섞은 음성 샘플(negative samples)을 사용하는 방법을 제안한다.



### Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation (https://arxiv.org/abs/2407.15396)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 Scene Graph Generation (SGG) 분야에서 '단일 술어 (predicate)'의 다양한 의미 (semantic diversity) 를 고려한 새로운 프레임워크를 제안한다. 기존 SGG 모델들은 각 주어-목적어 쌍에 대해 단일 술어만 예측하도록 학습되어, 술어의 다양한 의미를 간과하고 편향된 예측을 할 수 있었다. 본 논문의 DPL (Semantic Diversity-aware Prototype-based Learning) 프레임워크는 각 술어에 대한 프로토타입 (prototype)을 이용하여 술어의 의미 공간 (semantic space) 을 학습하고, 다양한 의미를 구분하여 보다 정확하고 편향되지 않은 예측을 가능하게 한다.



### Towards Robust Vision Transformer via Masked Adaptive Ensemb (https://arxiv.org/abs/2407.15385)
Comments:
          9 pages

- **What's New**: 본 논문에서는 Vision Transformers (ViT)의 robustness를 향상시키기 위한 새로운 ViT 아키텍처를 제안합니다. 이 아키텍처는 탐지기(detector)와 분류기(classifier)를 적응형 앙상블(adaptive ensemble)로 연결하여 구성됩니다. 탐지기는 adversarial examples를 감지하고, 분류기는 깨끗한 이미지와 adversarial examples에서 추출한 시각적 표현을 각각 사용하여 분류를 수행합니다. 적응형 앙상블은 두 엔코더에서 추출한 시각적 표현의 비율을 조정하여 정확한 분류를 수행합니다.



### Is user feedback always informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data (https://arxiv.org/abs/2407.15383)
Comments:
          Accepted to ECCV 2024, Project page: this https URL

- **What's New**: 이 논문은 사용자 피드백을 이용하여 소스 모델을 타겟 환경에 적응시키는 새로운 방법을 제안합니다. 기존의 Semi-supervised Domain Adaptation (SemiSDA) 방법들은 실제 사용자 피드백을 바로 사용할 경우 적응 성능이 저하되는 문제점이 있었습니다. 이는 사용자 피드백이 모델의 예측이 잘못되었을 때 더 많이 발생하는 경향이 있기 때문입니다. 이 현상을 'Negatively Biased Feedback (NBF)' 이라고 부릅니다. 이러한 문제를 해결하기 위해 이 논문은 'Retrieval Latent Defending'이라는 새로운 접근 방식을 제안합니다. 이 방법은 기존의 SemiSDA 방법들에 적용하여, 적응 과정 동안 'latent defending samples'를 활용하여 균형 잡힌 supervised signal을 제공합니다. 이 논문은 NBF로 인한 문제점을 보여주고, 다양한 벤치마크에서 제안된 방법의 효과를 입증합니다.



### Sparse Prior Is Not All You Need: When Differential Directionality Meets Saliency Coherence for Infrared Small Target Detection (https://arxiv.org/abs/2407.15369)
Comments:
          Submitted to IEEE TIM, Minor Revision

- **What's New**: 이 논문에서는 적외선 소형 표적 탐지(infrared small target detection) 성능을 향상시키기 위한 새로운 Sparse Differential Directionality (SDD) 프레임워크를 제안합니다. SDD는 적외선 소형 표적의 고유한 방향 특징 (directional characteristics)을 활용하여 배경과 구분하고, 튜커 분해 (Tucker decomposition)에서 유도된 차분 방향 이미지 (differential directional images)와 시간적 요소의 연속성 차이 행렬 (continuity difference matrix)에 혼합 스파스 제약 (mixed sparse constraints)을 적용합니다. 또한, 계층적 분해 (hierarchical decomposition) 과정에서 표적과 배경의 대비를 강화하는 뚜렷함 일관성 전략 (saliency coherence strategy)을 통해 표적 탐지 가능성을 향상시킵니다. 제안된 모델은 근접 교대 최소화 (Proximal Alternating Minimization, PAM) 알고리즘을 사용하여 효율적으로 해결됩니다.



### A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Mod (https://arxiv.org/abs/2407.15362)
Comments:
          44 pages, 9 figures

- **What's New**: 본 논문은 조직 병리학 (Computational Pathology, CPath) 분야에서 다양한 임상 작업의 성능을 향상시키는 작업-독립적인 기반 모델 (task-agnostic foundation model)을 위한 새로운 방법을 제시합니다. 이 방법은 기존의 비전-전용 (vision-only) 또는 비전-캡션 (vision-captions) 데이터만 사용하는 것과 달리, 병리 보고서 (pathology reports)와 유전자 발현 프로필 (gene expression profiles)과 같은 귀중한 다중 모달 (multimodal) 데이터를 통합합니다. 또한, 기존의 CPath 모델들이 패치 수준 (patch level)에 초점을 맞춘 것과 달리, 본 논문은 전체 슬라이드 수준 (whole-slide level)에서 다중 모달 정보를 활용하여 모델을 사전 훈련하는 새로운 패러다임인 mSTAR (Multimodal Self-TAught PRetraining)를 제안합니다. 이를 통해 CPath 모델은 전체 슬라이드 수준의 맥락을 이해할 수 있게 됩니다.



### X-Recon: Learning-based Patient-specific High-Resolution CT Reconstruction from Orthogonal X-Ray Images (https://arxiv.org/abs/2407.15356)
- **What's New**: 이 연구는 흉부 X선 영상을 이용하여 CT 영상을 초저밀도로 재구성하는 새로운 학습 기반 네트워크인 X-Recon을 제안합니다. X-Recon은 여러 스케일의 융합 렌더링 모듈(MFusionRen)을 갖춘 생성기와 판별기에서 기존의 합성곱 계층을 대체하는 3D 좌표 합성곱 계층을 포함하는 생성적 적대적 네트워크(GAN)을 사용합니다.  X-Recon은 기존의 방법보다 훨씬 높은 재구성 해상도를 달성하며, 이는 초저밀도 3D 단층 영상 재구성 분야에서 새로운 최첨단 수준을 설정합니다. 또한, 연구진은 재구성된 CT 이미지의 품질을 평가하기 위해 영상 처리 기법과 딥 러닝 모델을 결합한 영점 폐탈(Zero-shot) 흉막 탈출(pneumothorax) 분할 파이프라인인 PTX-Seg를 제안했습니다. 



### Attention Beats Linear for Fast Implicit Neural Representation Generation (https://arxiv.org/abs/2407.15355)
Comments:
          Accept by ECCV 2024

- **What's New**: 본 논문에서는 Attention-based Localized Implicit Neural Representation (ANR)이라는 새로운 방법을 제안하여, 데이터 표현의 효율성과 정확성을 향상시키는 방법을 제시합니다. 기존의 Multi-Layer Perceptrons (MLP) 기반 INR 방식은 불연속적인 신호 모델링에 어려움을 겪고, 많은 파라미터를 필요로 한다는 문제점을 가지고 있습니다. ANR은 Localized Attention Layer (LAL)과 Global MLP를 결합하여, 좌표 특징과 데이터 특징을 통합하고 의미 있는 출력으로 변환합니다. 또한, Transformer-like Hyper-network를 이용하여 데이터 인스턴스를 압축된 벡터로 표현하는 인스턴스 표현 프레임워크를 설계했습니다. ANR은 Instance-specific Representation Vector와 Instance-agnostic ANR Parameter를 이용하여, 대상 신호를 연속적인 함수로 정확하게 재구성합니다. 또한, Super-Resolution Inference 결과에서 발생하는 Aliasing Artifacts를 해결하기 위한 방법도 제시합니다.  



### Learning High-resolution Vector Representation from Multi-Camera Images for 3D Object Detection (https://arxiv.org/abs/2407.15354)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: 이 논문은 고해상도 벡터 표현을 사용하는 카메라 기반 3D 객체 검출기인 VectorFormer를 소개합니다. 이것은 기존의 BEV 격자 표현 방식이 공간 해상도가 높아짐에 따라 연산 비용이 제곱으로 증가하는 문제를 해결하기 위한 것입니다.  VectorFormer는 고해상도 벡터 표현을 저해상도 BEV 표현과 결합하여 두 개의 새로운 모듈인 벡터 산란(vector scattering) 및 집합(gathering)을 통해 다중 카메라 이미지에서 고해상도로 3D 기하학을 효율적으로 활용합니다. 이를 통해 풍부한 장면 컨텍스트를 가진 학습된 벡터 표현은 최종 예측을 위한 디코딩 쿼리 역할을 합니다.



### WTS: A Pedestrian-Centric Traffic Video Dataset for Fine-grained Spatial-Temporal Understanding (https://arxiv.org/abs/2407.15350)
Comments:
          ECCV24. Website: this https URL

- **What's New**: 이 논문에서는 자율 주행과 안전을 위해 필수적인 교통 시나리오에서의 세밀한 비디오 이벤트 이해(fine-grained video event understanding) 문제를 다룹니다. 기존 데이터셋은 운전자 또는 차량 행동에 초점을 맞추고 보행자 관점은 종종 무시합니다. 이러한 간극을 메우기 위해, 이 논문은 수백 개의 교통 시나리오에서 1,200개 이상의 비디오 이벤트에서 차량과 보행자의 상세한 행동을 강조하는 WTS 데이터셋을 소개합니다. WTS는 차량-인프라 협력 환경에서 차량 자체 및 고정 오버헤드 카메라의 다양한 관점을 통합하여 보행자 분석에 중점을 둔 동기화된 2D/3D 보기를 위해 포괄적인 텍스트 설명과 고유한 3D Gaze 데이터로 풍부하게 합니다. 또한 5,000개의 공개적으로 출처를 밝힌 보행자 관련 교통 비디오에 대한 주석을 제공합니다. 또한 추론 캡션을 지상 진실과 일치시키기 위한 LLM 기반 평가 지표인 LLMScorer를 소개합니다. WTS를 사용하여 비디오에서 텍스트로의 밀집형 작업(dense video-to-text tasks)에 대한 벤치 마크를 설정하고, 인스턴스 인식 VideoLLM 방법을 기준으로 최첨단 비전-언어 모델을 탐구합니다. WTS는 교통 안전과 자율 주행 개발을 향상시키는 세밀한 비디오 이벤트 이해를 발전시키는 것을 목표로 합니다.



### RoadPainter: Points Are Ideal Navigators for Topology transformER (https://arxiv.org/abs/2407.15349)
Comments:
          17 pages, 5 figures, Accepted by ECCV 2024

- **What's New**: 이 논문에서는 RoadPainter라는 새로운 접근 방식을 제시합니다. 이는 다중 뷰 이미지를 사용하여 차선 중심선의 위상 (topology)을 감지하고 추론합니다. RoadPainter의 핵심은 각 차선 중심선 마스크에서 포인트 세트를 추출하여 차선 중심선 예측의 정확성을 향상시키는 것입니다.



### ThermalNeRF: Thermal Radiance Fields (https://arxiv.org/abs/2407.15337)
Comments:
          Presented at ICCP 2024; project page at this https URL

- **What's New**: 본 논문은 LWIR (Long-Wave Infrared) 이미지의 낮은 해상도와 제한된 특징으로 인해 어려움을 겪는 3D 열 이미지 복원을 위한 통합 프레임워크를 제안합니다. 이 프레임워크는 가시광선과 적외선 카메라 모두에서 관찰되는 장면을 표현하는 다중 스펙트럼 복사장 (multispectral radiance field)을 사용하여 가시광선과 적외선 스펙트럼의 정보를 활용합니다.  RGB와 적외선 카메라는 간단한 교정 타겟 (calibration target)을 사용하여 사전 처리 단계에서 서로 교정됩니다.



### Explore the LiDAR-Camera Dynamic Adjustment Fusion for 3D Object Detection (https://arxiv.org/abs/2407.15334)
- **What's New**: 이 논문은 카메라와 LiDAR 데이터의 분포 차이를 줄이고 효과적인 모달 표현을 학습하여 융합 성능을 향상시키기 위한 새로운 기술을 제안합니다. 특히 3D 객체 탐지 성능 향상에 초점을 맞춥니다.



### Iterative Ensemble Training with Anti-Gradient Control for Mitigating Memorization in Diffusion Models (https://arxiv.org/abs/2407.15328)
Comments:
          To appear in ECCV 2024, 20 pages with 7 figures

- **What's New**: 본 연구는 Diffusion 모델의 데이터 암기 문제를 해결하기 위해, 시각적 모드에서 새로운 훈련 프레임워크를 제안합니다. 기존 방법들은 텍스트 모드나 데이터 증강 전략에만 초점을 맞춘 반면, 본 연구는 더 일반적이고 근본적인 접근 방식을 제시합니다.



### Open-CD: A Comprehensive Toolbox for Change Detection (https://arxiv.org/abs/2407.15317)
Comments:
          9 pages

- **What's New**: 이 논문에서는 교사의 학습 평가 시간을 줄이기 위해 자동으로 MCQ를 생성하는 방법을 연구했으며, MCQ 생성의 교육적 가치를 고려하는 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안한다. KDA는 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하는 지표이다. 이 논문에서는 KDA를 측정하는 방법과 KDA를 자동으로 측정하는 두 가지 지표인 KDA_disc와 KDA_cont를 제안한다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었다.



### FMDNN: A Fuzzy-guided Multi-granular Deep Neural Network for Histopathological Image Classification (https://arxiv.org/abs/2407.15312)
Comments:
          This paper has been accepted by IEEE Transactions on Fuzzy Systems for publication. Permission from IEEE must be obtained for all other uses, in any current or future media. The final version is available at [doi: https://doi.org/10.1109/TFUZZ.2024.3410929]

- **What's New**: 자동 MCQ 생성의 교육적 가치를 고려한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다.



### VideoGameBunny: Towards vision assistants for video games (https://arxiv.org/abs/2407.15295)
- **What's New**: 이 논문은 비디오 게임 이미지 이해를 위해 LLaVA 스타일 모델인 VideoGameBunny를 개발한 것을 소개합니다. 이 모델은 Bunny를 기반으로 하며, 413개 게임에서 수집된 185,259개의 이미지 데이터셋과 389,565개의 이미지-명령어 쌍 (이미지 캡션, 질문-답변 쌍, 136,974개 이미지의 16개 요소를 포함하는 JSON 표현)을 사용하여 훈련되었습니다.



### Enhancing Retinal Disease Classification from OCTA Images via Active Learning Techniques (https://arxiv.org/abs/2407.15293)
Comments:
          10 pages, 2 figures, 3 tables, Published at Data Engineering in Medical Imaging (DEMI) workshop @ MICCAI 2024

- **What's New**: 본 논문은 안과 질환 진단을 위해 Optical Coherence Tomography Angiography (OCTA) 이미지를 사용하여 딥러닝 기반 예측 모델을 개발하는 데 있어, 활성 학습 (Active Learning) 기법을 활용하여 데이터셋을 효과적으로 선택하는 방법을 제안합니다. 특히, 기존의 딥러닝 모델이 데이터 부족으로 인해 일반화 성능이 저조한 문제를 해결하기 위해, 활성 학습을 통해 모델 학습에 가장 효과적인 데이터를 선별하는 방법을 제시합니다.



### Point Transformer V3 Extreme: 1st Place Solution for 2024 Waymo Open Dataset Challenge in Semantic Segmentation (https://arxiv.org/abs/2407.15282)
Comments:
          1st Place Solution for 2024 Waymo Open Dataset Challenge in Semantic Segmentation

- **What's New**: 이 기술 보고서는 2024 Waymo Open Dataset Challenge의 의미론적 분할 트랙에서 1위를 차지한 솔루션을 자세히 설명합니다. 최첨단 플러그 앤 플레이 훈련 및 추론 기술을 구현하여 Waymo 벤치마크에서 Point Transformer V3의 성능을 크게 향상시켰습니다. 특히, 고급 버전인 Point Transformer V3 Extreme은 멀티 프레임 훈련과 클리핑 지점 정책을 사용하여 기존 PTv3 성능을 뛰어넘는 상당한 성과를 거두었습니다. 또한, 간단한 모델 앙상블 전략을 사용하여 결과를 더욱 향상시켰습니다. 이 접근 방식은 Waymo Open Dataset 의미론적 분할 리더보드에서 1위를 차지하여 다른 참가자들을 능가했습니다. 



### MIBench: Evaluating Multimodal Large Language Models over Multiple Images (https://arxiv.org/abs/2407.15272)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문은 Multi-Image (여러 이미지) 시나리오에서 MLLM(Multimodal Large Language Model, 다중 모달 대규모 언어 모델)의 성능을 포괄적으로 평가하기 위한 새로운 벤치마크인 MIBench를 제안합니다. MIBench는 Multi-Image Instruction (MII), Multimodal Knowledge-Seeking (MKS), Multimodal In-Context Learning (MIC)의 세 가지 시나리오로 나누어져 총 13개의 태스크와 13,000개의 주석이 달린 샘플로 구성됩니다.



### Weakly SSM : On the Viability of Weakly Supervised Segmentations for Statistical Shape Modeling (https://arxiv.org/abs/2407.15260)
- **What's New**: 이 논문은 여러 개의 counterfactual을 생성하여 집합적 의사 결정 (collective decisions)을 통해 단어들의 인과관계를 더 robust하게 파악하는 새로운 방법을 제안한다. 기존의 방법들은 사람이 counterfactual을 만들거나 모델이 데이터셋에서 counterfactual 비슷한 것들을 찾아야 했지만, spurious correlation에 영향을 받는다는 문제가 있었다. 이 논문에서 제안된 방법은 이러한 문제를 해결하여 counterfactual robustness, cross-domain generalization, scarce data에서의 generalization 등 다양한 측면에서 개선된 성능을 보여준다. (Existing methods either require humans to add counterfactuals to the dataset or machines to automatically matches near-counterfactuals already in the dataset. Unlike existing augmentation is affected by spurious correlations, ours, by synthesizing “a set” of counterfactuals, and making a collective decision on the distribution of predictions on this set, can robustly supervise the causality of each term.)



### An Adaptive System for Wearable Devices to Detect Stress Using Physiological Signals (https://arxiv.org/abs/2407.15252)
- **What's New**: 본 논문은 개인 맞춤형 스트레스 감지 (personalized stress detection)을 위한 적응형 프레임워크를 소개합니다. PPG와 EDA 신호를 사용하여 각 사용자에게 맞춤형 모델을 제공하여 스트레스 감지 정확도를 높입니다. 기존 방법들은 모든 사용자에게 동일한 모델을 사용하여 도메인 차이 (domain shifts)로 인해 성능이 저하될 수 있는 반면, 본 프레임워크는 개인 맞춤형 모델을 제공합니다. 



### BIGbench: A Unified Benchmark for Social Bias in Text-to-Image Generative Models Based on Multi-modal LLM (https://arxiv.org/abs/2407.15240)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.17814

- **What's New**: 본 논문은 이미지 생성 모델의 bias를 측정하는 통합 벤치마크인 BIGbench를 소개합니다. BIGbench는 기존 벤치마크와 달리, bias를 4가지 차원 (manifestation, visibility, acquired attributes, protected attributes)으로 분류하고, 다양한 이미지 생성 모델을 평가합니다. 또한, 다중 모달 대규모 언어 모델(MLLM)을 활용하여 자동화된 평가 시스템을 구축했습니다.



### CGB-DM: Content and Graphic Balance Layout Generation with Transformer-based Diffusion Mod (https://arxiv.org/abs/2407.15233)
- **What's New**: 이 논문에서는 교사의 학습 평가 시간을 단축시킬 수 있는 자동 MCQ 생성 분야에서 교육적 가치를 고려하지 않는 기존 평가 지표의 한계를 지적하고, 새로운 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 제대로 평가하는지 측정하며, Human evaluation과 더불어 자동 평가 지표인 KDA_disc와 KDA_cont를 제시하여 MCQ 생성 모델의 질을 더 정확하게 평가할 수 있도록 합니다. KDA_disc와 KDA_cont는 pre-trained language model을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다.



### 3D Reconstruction of the Human Colon from Capsule Endoscope Video (https://arxiv.org/abs/2407.15228)
Comments:
          11 pages, 12 figures

- **What's New**: 본 논문은 무선 캡슐 내시경 비디오에서 얻은 이미지 시퀀스를 사용하여 인간 결장의 전체 섹션의 3D 모델을 구축하는 가능성을 조사하여 위장병 전문의에게 향상된 시청 환경을 제공합니다. 캡슐 내시경 이미지에는 많은 3D 재구성 알고리즘에 적합하지 않은 왜곡과 아티팩트가 포함되어 있어 이 문제를 해결하는 데 어려움이 있습니다. 그러나 최근 왜곡과 아티팩트를 활성화하거나 비활성화할 수 있는 인간 위장 시스템의 가상 그래픽 기반 모델 개발을 통해 문제를 '해부'할 수 있게 되었습니다. 그래픽 모델은 또한 3D 재구성 방법에 의해 도입된 기하학적 왜곡 계산을 가능하게 하는 기준 진실을 제공합니다. 이 논문에서는 대부분의 왜곡과 아티팩트를 제외하여 기존 방법으로 인간 위장 시스템의 전체 섹션을 재구성할 수 있는지 여부를 확인합니다. 연구 결과는 동시 위치 확인 및 매핑을 사용하여 3D 재구성이 가능함을 보여줍니다. 또한 밀도가 크게 다른 결과 점 구름에서 위장 벽 표면을 재구성하기 위해 Poisson 표면 재구성은 좋은 선택입니다. 연구 결과는 유망하며, 이 문제에 대한 추가 연구를 장려합니다.



### Efficient Visual Transformer by Learnable Token Merging (https://arxiv.org/abs/2407.15219)
- **What's New**: 이 논문은 학습 가능한 토큰 병합(LTM, Learnable Token Merging)을 사용한 새로운 트랜스포머 블록인 LTM-트랜스포머를 제안합니다. LTM-트랜스포머는 학습 가능한 방식으로 토큰 병합을 수행하여 기존 비주얼 트랜스포머의 연산량(FLOPs)과 추론 시간을 줄이면서 정확도를 유지하거나 개선합니다.



### Surfel-based Gaussian Inverse Rendering for Fast and Relightable Dynamic Human Reconstruction from Monocular Video (https://arxiv.org/abs/2407.15212)
Comments:
          Under Review; Project Page: this https URL

- **What's New**: 이 논문은 단일 비디오에서 재조명 가능하고 동적인 옷을 입은 아바타를 효율적이고 정확하게 재구성하는 SGIA (Surfel-based Gaussian Inverse Avatar) 방법을 소개합니다. SGIA는 이전 Gaussian Avatar 방법들을 발전시켜 옷을 입은 아바타에 대한 PBR (Physically-Based Rendering) 특성을 포괄적으로 모델링하여 다양한 조명 조건에서 새로운 포즈로 아바타를 조작할 수 있도록 합니다. 특히, 이 접근 방식은 기존 암묵적 기반 기술의 성능을 능가하는 빠른 광선 계산을 위해 사전 통합 및 이미지 기반 조명을 통합합니다.  재료 조명 분리 및 정확한 기하학적 재구성과 관련된 과제를 해결하기 위해, 이 논문은 혁신적인 폐색 근사 전략과 점진적 훈련 방식을 제안합니다. 광범위한 실험 결과, SGIA는 매우 정확한 물리적 특성을 달성할 뿐만 아니라 동적 인간 아바타의 사실적인 재조명을 크게 향상시켜 상당한 속도 이점을 제공하는 것으로 나타났습니다.  



### Mask Guided Gated Convolution for Amodal Content Completion (https://arxiv.org/abs/2407.15203)
Comments:
          6 pages, 4 figures

- **What's New**: 본 논문에서는 가려진 객체의 부분적으로 보이는 부분을 재구성하는 모델을 제안합니다. 이 모델은 가중 마스크(weighted mask)를 입력으로 받아 가려진 객체의 보이는 픽셀에 더 많은 가중치를 부여하고, 배경 픽셀은 무시하여 가려진 영역의 특징을 추출합니다. 가시 영역에서 더 많은 주의를 끌어내어 기존 모델보다 더 효과적으로 보이지 않는 패치를 예측할 수 있습니다. 특히 균일한 텍스처를 가진 객체에 대해 더 효과적입니다. 이 모델은 COCOA 데이터셋과 두 개의 하위 데이터셋에서 자기 지도 학습 방식으로 학습되었습니다. 실험 결과, 이 모델은 기존 모델보다 더 고품질의 텍스처가 풍부한 출력을 생성하는 것으로 나타났습니다. 코드는 this https URL 에서 확인할 수 있습니다.



### Multiple Object Detection and Tracking in Panoramic Videos for Cycling Safety Analysis (https://arxiv.org/abs/2407.15199)
- **What's New**: 본 연구는 파노라마 사이클링 영상에서 자동 도로 사용자 분석을 위한 새로운 방법론을 제시합니다. 이 방법론은 기존 컴퓨터 비전 모델의 성능을 향상시키고, 특히 파노라마 영상의 왜곡, 작은 객체, 경계 연속성 문제를 해결하는 데 초점을 맞춥니다.



### HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions (https://arxiv.org/abs/2407.15187)
Comments:
          Homepage: this https URL

- **What's New**: 본 논문은 텍스트 기반 3D 장면 생성을 위한 새로운 프레임워크인 HoloDreamer를 소개합니다. HoloDreamer는 텍스트 기반 이미지 확산 모델을 활용하여 고해상도 파노라마를 생성하고, 이를 3D Gaussian Splatting(3D-GS)으로 렌더링하여 시각적으로 일관성 있고 완전한 3D 장면을 생성합니다. 기존의 방법들은 점진적으로 이미지를 확장하는 방식으로 3D 장면을 생성하여 시각적 일관성 문제가 발생했으나, HoloDreamer는 하나의 전체 파노라마를 생성하여 이 문제를 해결합니다.



### Rethinking Domain Adaptation and Generalization in the Era of CLIP (https://arxiv.org/abs/2407.15173)
- **What's New**: 본 논문은 CLIP 기반 zero-shot 인식을 위한 새로운 unsupervised domain adaptation 방법을 제안합니다. 기존 CLIP은 다양한 데이터셋에서 zero-shot 인식 능력을 보여주지만, 특정 도메인에 대한 학습은 부족합니다. 이 논문에서는 domain prior (예: 'infograph', 'clipart', 'quickdraw')와 같은 간단한 정보를 추가하여 CLIP의 zero-shot 성능을 향상시키는 방법을 제안합니다. 또한, 본 논문은 CLIP의 사전 훈련 데이터셋이 이미 다양한 도메인 데이터를 포함하고 있기 때문에 기존 UDA 설정에서 source domain 데이터의 필요성이 크게 줄어든다는 점을 강조합니다. 즉, CLIP은 pseudo-labeling 기반 self-training을 통해 target domain 데이터에서 학습하는 것이 더 효과적입니다. 마지막으로, 본 논문은 다수의 unlabeled source domain에서 CLIP의 task generalization 능력을 향상시키는 방법을 제안합니다. 이는 다양한 unlabeled 데이터에서 common knowledge를 학습하여 CLIP의 일반화 능력을 향상시키는 것을 목표로 합니다.



### Assessing Sample Quality via the Latent Space of Generative Models (https://arxiv.org/abs/2407.15171)
Comments:
          Accepted paper - ECCV 2024

- **What's New**: 이 논문은 MCQ 생성 모델의 교육적 가치를 평가할 수 있는 새로운 자동 평가 지표, '지식 종속 가능성(KDA)'을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 제대로 평가하는지 측정합니다. 또한,  KDA를 자동으로 측정하기 위한 KDA_disc와 KDA_cont라는 두 가지 지표를 제안하고, 인간 평가를 통해 이 지표들이 실제 강의실 환경에서 유용하게 사용될 수 있음을 보여줍니다.



### Semi-Supervised Pipe Video Temporal Defect Interval Localization (https://arxiv.org/abs/2407.15170)
Comments:
          13 pages, 3 figures

- **What's New**: 이 연구는 하수관 CCTV 검사에서 정확한 시간적 결함 위치를 파악하는 새로운 반지도 학습 방식인 PipeSPO를 제안한다. 기존 방식들은 시간 간격 표시가 필요하지 않은 반면, PipeSPO는 시간 간격 표시 정보를 활용하여 보다 정확한 위치 파악을 가능하게 한다. 또한, 시각적 오도메트리(visual Odometry) 기능을 사용하여 카메라 위치 정보를 포착하고, 비지도 전이 학습(unsupervised pretext tasks)을 통해 레이블이 없는 데이터를 최대한 활용한다. 



### The VEP Booster: A Closed-Loop AI System for Visual EEG Biomarker Auto-generation (https://arxiv.org/abs/2407.15167)
Comments:
          19 pages, 6 figures

- **What's New**: 본 논문에서는 시각 자극 프로토콜 하에서 신뢰할 수 있고 안정적인 EEG 바이오마커를 생성하는 새로운 폐쇄 루프 AI 프레임워크인 VEP Booster를 제안합니다. 이 시스템은 이미지 생성기를 사용하여 실시간으로 사람의 EEG 신호에서 받은 피드백을 기반으로 자극 이미지를 개선하며, 이를 통해 시각 피질(V1) 뉴런의 선호도에 맞게 시각 자극을 생성하고 자극에 가장 잘 반응하는 뉴런을 효과적으로 타겟팅할 수 있습니다.



### HERGen: Elevating Radiology Report Generation with Longitudinal Data (https://arxiv.org/abs/2407.15158)
Comments:
          ECCV 2024

- **What's New**: 이 논문은  **History Enhanced Radiology Report Generation (HERGen)** 이라는 새로운 프레임워크를 제안하여 의료 이미지 데이터와 환자의 병력을 통합하여 정확한 방사선 보고서를 자동 생성하는 것을 목표로 합니다. HERGen은 **group causal transformer**를 사용하여 여러 번의 진료 기록을 효율적으로 통합하고, 이를 통해 방사선 보고서 생성의 정확도와 효율성을 높입니다.



### Distilling Vision-Language Foundation Models: A Data-Free Approach via Prompt Diversification (https://arxiv.org/abs/2407.15155)
Comments:
          Accepted by ACMMM 2023

- **What's New**: 본 논문은 자동 MCQ 생성 평가 메트릭으로, KDA (Knowledge Dependent Answerability) 라는 새로운 지표를 제안합니다. 기존의 BLEU, ROUGE, METEOR와 달리, KDA는 MCQ가 학생의 지식을 측정하는 능력을 평가합니다. KDA는 human survey를 통해 측정되며, KDA_disc와 KDA_cont라는 자동 평가 지표를 통해 근사화됩니다.  KDA_disc와 KDA_cont는 pre-trained language model을 활용하여 학생의 문제 해결 방식을 모방합니다. Human study를 통해, KDA_disc와 KDA_cont가 실제 강의실 상황에서의 사용성과 강한 상관관계를 보인다는 것을 확인했습니다.



### Anchored Diffusion for Video Face Reenactmen (https://arxiv.org/abs/2407.15153)
- **What's New**: 본 논문은 'Anchored Diffusion'이라는 새로운 방법을 소개하여 상대적으로 길고 매끄러운 비디오 합성을 가능하게 합니다. 이 방법은 Diffusion Transformers (DiTs)를 확장하여 시간 정보를 통합하고 짧은 비디오 세그먼트를 생성하기 위해 sequence-DiT (sDiT) 모델을 만듭니다. 기존 연구와 달리, 이 논문에서는 무작위 비균일 시간 간격을 가진 비디오 시퀀스에서 모델을 훈련하고 외부 가이드를 통해 시간 정보를 통합하여 유연성을 높여 단기 및 장기 관계를 모두 포착할 수 있도록 합니다. 또한 추론 중에 Transformer 아키텍처를 활용하여 확산 프로세스를 수정하여 공통 프레임에 고정된 비균일 시퀀스 배치를 생성하여 시간 거리에 관계없이 일관성을 보장합니다.



### Rethinking Feature Backbone Fine-tuning for Remote Sensing Object Detection (https://arxiv.org/abs/2407.15143)
Comments:
          Under Review

- **What's New**: 본 논문에서는 원격 감지 객체 탐지 (Remote Sensing Object Detection) 에서 feature backbone fine-tuning을 위한 새로운 방법인 DBF (Dynamic Backbone Freezing)를 제안합니다. 이 방법은 'Freezing Scheduler' 모듈을 통해 backbone feature의 업데이트를 동적으로 관리하여, backbone이 낮은 수준의 일반적인 특징 (low-level generic features)을 추출할지, 또는 원격 감지 도메인에 대한 특정 지식 (specific knowledge)을 갖추어야 할지의 문제를 해결합니다. 



### D$^4$M: Dataset Distillation via Disentangled Diffusion Mod (https://arxiv.org/abs/2407.15138)
Comments:
          Accepted to CVPR 2024

- **What's New**: 본 논문에서는 MCQ 생성 평가를 위해 기존의 n-gram 기반 유사도 측정에서 벗어나 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이는 MCQ가 대상 사실에 대한 학생의 지식을 실제로 평가할 수 있는지 여부를 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사화합니다.  



### D$^4$-VTON: Dynamic Semantics Disentangling for Differential Diffusion based Virtual Try-On (https://arxiv.org/abs/2407.15111)
Comments:
          ECCV2024

- **What's New**: 본 논문은 이미지 기반 가상 피팅(VTON)을 위한 혁신적인 솔루션인 D$^4$-VTON을 소개합니다.  기존 연구에서 제기된 옷 변형 전후의 의미적 불일치 및 정적 주석 기반 옷 파서 의존성과 같은 문제점들을 해결합니다.  또한, 인페인팅 및 잡음 제거와 같은 동시 작업을 처리할 때 확산 기반 VTON 모델의 복잡성을 다룹니다.  D$^4$-VTON은 동적 의미 분리 모듈(DSDM)과 차등 정보 추적 경로(DITP)라는 두 가지 핵심 기술을 활용합니다.  DSDM은 옷에서 추상적인 의미 정보를 추출하여 별개의 로컬 플로우를 생성하여 자체 발견 방식으로 정확한 옷 변형을 개선합니다.  DITP를 통합함으로써, 불완전한 피팅 입력과 완전한 버전 사이의 차등 정보를 포착하여 네트워크가 여러 가지 퇴화를 독립적으로 처리할 수 있도록 하여 학습 모호성을 최소화하고 최소한의 오버헤드로 사실적인 결과를 얻을 수 있습니다.  광범위한 실험을 통해 D$^4$-VTON이 정량적 지표와 질적 평가 모두에서 기존 방법을 크게 능가하여 사실적인 이미지를 생성하고 의미적 일관성을 보장하는 능력을 보여줍니다.



### Navigation Instruction Generation with BEV Perception and Large Language Models (https://arxiv.org/abs/2407.15087)
Comments:
          ECCV 2024; Project Page: this https URL

- **What's New**: 이 논문은 3D 환경의 기하학적 정보와 객체 의미를 고려하여 탐색 경로를 설명하는 탐색 지시 생성 (navigation instruction generation)을 위한 새로운 방법, BEVInstructor를 제안합니다. BEVInstructor는 Bird's Eye View (BEV) 특징을 Multi-Modal Large Language Models (MLLMs)에 통합하여 3D 환경에 대한 이해를 향상시킵니다. 특히, BEVInstructor는 BEV 및 원근 특징을 융합하여 PerspectiveBEVVisual Encoder를 구성합니다. 또한, MLLMs의 강력한 언어 능력을 활용하기 위해 융합된 표현을 MLLMs에 대한 시각적 프롬프트로 사용하고, 매개변수 효율적인 업데이트를 위해 Perspective-BEV 프롬프트 조정 (prompt tuning)을 제안합니다. BEVInstructor는 Perspective-BEV 프롬프트를 기반으로, 지시를 점진적으로 개선하는 인스턴스 기반 반복적 개선 파이프라인을 채택합니다. BEVInstructor는 R2R, REVERIE, UrbanWalk과 같은 다양한 데이터셋에서 뛰어난 성능을 보여줍니다.



### Learn to Preserve and Diversify: Parameter-Efficient Group with Orthogonal Regularization for Domain Generalization (https://arxiv.org/abs/2407.15085)
- **What's New**: 이 논문에서는 기존의 Domain Generalization (DG) 방법론이 pre-trained 모델의 일반화 능력을 제대로 활용하지 못한다는 점을 지적하며, Parameter-Efficient Group with Orthogonal Regularization (PEGO)라는 새로운 프레임워크를 제안합니다. PEGO는 pre-trained Vision Transformer (ViT) 모델에 trainable Low-Rank Adaptation (LoRA) 모듈을 추가하고, orthogonal regularization loss를 적용하여 pre-trained 모델의 일반화 능력을 보존하고 다양한 지식을 학습하도록 설계되었습니다.



### 3D Gaussian Parametric Head Mod (https://arxiv.org/abs/2407.15070)
Comments:
          project page: this https URL

- **What's New**: 이 논문은 3D Gaussian Parametric Head Model을 제안하여, 3D Gaussian을 이용해 사람의 머리 모델을 표현하고 photo-realistic rendering과 real-time rendering 속도를 달성합니다. 또한 잘 설계된 학습 전략을 통해 모델이 안정적으로 학습되도록 하고, 풍부한 외모 정보와 복잡한 표현을 효율적으로 학습하도록 합니다. 3D Gaussian Parametric Head Model은 단일 이미지에서 상세하고 고품질의 얼굴 아바타를 생성할 수 있으며, 표현과 신원 편집 기능도 제공합니다.



### VoxDepth: Rectification of Depth Images on Edge Devices (https://arxiv.org/abs/2407.15067)
- **What's New**: 이 연구는 딥 러닝 기반의 자율주행 로봇, 특히 드론 및 산업용 로봇에서 자주 발생하는 깊이 이미지의 오류 문제 해결을 위한 새로운 솔루션, VoxDepth를 제안합니다. VoxDepth는 에지 장치에서 빠르고 정확하게 실행될 수 있는 획기적인 방법입니다. 기존의 ML 기반 방법들은 에지 장치의 제한된 연산 능력으로 인해 사용하기 어려웠으며, 비-ML 기반 방법들은 속도는 빠르지만 정확성이 떨어졌습니다.  VoxDepth는 3D 포인트 클라우드 생성과 융합을 통해 오류가 있는 깊이 이미지를 수정할 수 있는 템플릿을 만드는 새로운 기술을 활용합니다.  



### LSReGen: Large-Scale Regional Generator via Backward Guidance Framework (https://arxiv.org/abs/2407.15066)
- **What's New**: 이 논문은 MCQ(Multiple Choice Question) 생성을 위한 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표는 생성된 MCQ가 데이터셋의 골드 샘플과 얼마나 유사한지를 측정하지만, 학생의 지식을 평가하는 능력을 고려하지 않습니다. KDA는 대상 사실에 대한 학생의 지식을 고려하여 MCQ의 대답 가능성을 측정합니다. 이 논문에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하며, 이 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사합니다. Human evaluation 결과 KDA_disc와 KDA_cont는 실제 강의실 설정에서 사용성과 강한 상관관계가 있음을 보여줍니다.



### Prior Knowledge Integration via LLM Encoding and Pseudo Event Regulation for Video Moment Retrieva (https://arxiv.org/abs/2407.15051)
Comments:
          Accepted to ACM Multimedia 2024

- **What's New**: 이 논문은 비디오 모멘트 검색(VMR) 모델에서 일반 지식을 통합하고 의사 이벤트(pseudo-events)를 시간적 콘텐츠 분포의 사전 정보(priors)로 활용하는 대규모 언어 모델(LLM)의 가능성을 조사한다. 기존 LLM은 텍스트 설명을 생성하기 위한 디코더로 사용되어 왔는데, 이는 연속적인 출력 (예: 중요도 점수, 프레임 간 관계를 포착하는 프레임 간 임베딩)에는 직접적으로 적용되지 못한다는 제한점이 있었다. 이러한 제한을 극복하기 위해, 이 논문에서는 디코더가 아닌 LLM 인코더를 활용하는 방법을 제안한다.  



### MedSAGa: Few-shot Memory Efficient Medical Image Segmentation using Gradient Low-Rank Projection in SAM (https://arxiv.org/abs/2407.15042)
- **What's New**: 본 논문은 MedSAGa (Medical Segment Anything Model with Galore)를 소개합니다. MedSAGa는 SAM (Segment Anything Model)의 이미지 인코더 파라미터에 GaLore (Gradient Low-Rank Projection)를 적용하여 메모리 효율적인 few-shot 의료 이미지 분할을 달성합니다. 이 모델은 기존의 의료 이미지 분할 모델보다 적은 데이터로 높은 성능을 보여주며, 특히 자원이 부족한 환경에서 효과적입니다.



### Self-training Room Layout Estimation via Geometry-aware Ray-casting (https://arxiv.org/abs/2407.15041)
Comments:
          Accepted to ECCV-2024

- **What's New**: 이 논문은 레이 캐스팅(ray-casting) 공식을 사용하여 다양한 시점에서 생성된 여러 추정치를 집계함으로써, 라벨링되지 않은 데이터를 사용하여 보이지 않는 장면에서 룸 레이아웃(room layout) 추정 모델을 위한 새로운 지오메트리 인식 자기 학습 프레임워크를 소개한다. 이를 통해 자기 학습을 위한 신뢰할 수 있는 의사 라벨(pseudo-label)을 계산할 수 있다.



### ViT LoS V2X: Vision Transformers for Environment-aware LoS Blockage Prediction for 6G Vehicular Networks (https://arxiv.org/abs/2407.15023)
- **What's New**: 이 논문은 6G 차량 네트워크에서 장애물 (blockage) 을 예측하기 위해 CNN과 ViT를 결합한 딥러닝 기반 접근 방식을 제안합니다. 이 방식은 이미지와 빔 벡터를 포함한 시계열 다중 모드 데이터 (multimodal data) 에서 특징을 추출하기 위해 CNN과 ViT의 장점을 활용합니다. 또한, GRU 기반 아키텍처를 사용하여 추출된 특징과 미래 시간 단계의 장애물 상태 간의 시간 의존성을 포착합니다.  



### GreenStableYolo: Optimizing Inference Time and Image Quality of Text-to-Image Generation (https://arxiv.org/abs/2407.14982)
Comments:
          This paper is published in the SSBSE Challenge Track 2024

- **What's New**: 이 연구는 Stable Diffusion의 파라미터와 프롬프트를 최적화하여 GPU 추론 시간을 줄이고 이미지 생성 품질을 향상시키는 새로운 방법인 GreenStableYolo를 제안합니다. GreenStableYolo는 NSGA-II와 Yolo를 사용하여 Stable Diffusion의 파라미터와 프롬프트를 최적화합니다. 이를 통해 이미지 품질에 대한 약간의 손실(18%)을 감수하면서도, 추론 시간을 크게 단축(266% 감소)하고, 하이퍼볼륨(Hypervolume)을 526% 향상시켜 텍스트-이미지 생성 분야의 최첨단 기술을 발전시켰습니다.



### RGB2Point: 3D Point Cloud Generation from Single RGB Images (https://arxiv.org/abs/2407.14979)
- **What's New**: RGB2Point라는, Transformer 기반의 unposed (정렬되지 않은) 단일 뷰 RGB 이미지에서 3D 점 구름을 생성하는 새로운 방법을 소개합니다. RGB2Point는 객체의 입력 이미지를 받아 밀집된 3D 점 구름을 생성합니다. CNN 계층과 확산 잡음 제거 방식에 기반한 이전 작업과 달리, RGB2Point는 빠르고 고품질의 점 구름을 생성하며, 사용 가능한 범주에 걸쳐 일관된 품질을 유지하는 사전 훈련된 Transformer 계층을 사용합니다.



### ARoFace: Alignment Robustness to Improve Low-Quality Face Recognition (https://arxiv.org/abs/2407.14972)
Comments:
          European Conference on Computer Vision (ECCV 2024)

- **What's New**: 본 논문은 저품질 얼굴 인식(FR) 모델의 견고성을 향상시키기 위해, 얼굴 정렬 오류(FAE)를 고려한 새로운 훈련 방법을 제안합니다. 기존 연구들은 일반적인 저품질 요소(예: 대기 난류, 해상도 등)에 초점을 맞추었지만, 본 연구는 FAE를 FR에 특화된 저품질 요소로 간주합니다.



### Base and Exponent Prediction in Mathematical Expressions using Multi-Output CNN (https://arxiv.org/abs/2407.14967)
Comments:
          4 pages, 9 figures

- **What's New**: 이 논문은 다중 출력 Convolutional Neural Network (CNN)을 사용하여 수학식 이미지에서 밑과 지수를 예측하는 단순하지만 효과적인 접근 방식을 제시합니다. 

- **Technical Details**: 모델은 실제 조건을 모방하기 위해 임의 노이즈, 글꼴 크기 변형, 흐림 강도를 통합한 10,900개의 합성 이미지를 사용하여 훈련되었습니다. 제안된 CNN 모델은 효율적인 훈련 시간으로 강력한 성능을 보여줍니다. 

- **Performance Highlights**: 실험 결과는 모델이 밑과 지수 값을 예측하는 데 높은 정확도를 달성하여 노이즈가 있고 다양한 입력 이미지를 처리하는 이 접근 방식의 효율성을 입증합니다.



### Temporal Residual Jacobians For Rig-free Motion Transfer (https://arxiv.org/abs/2407.14958)
Comments:
          15 pages, 6 figures

- **What's New**: 이 논문은 데이터 기반 모션 전이를 위한 새로운 표현 방식인 Temporal Residual Jacobians를 소개합니다. 이 접근 방식은 리깅이나 중간 형태 키 프레임에 대한 접근을 가정하지 않고, 기하학적으로 시간적으로 일관된 모션을 생성하며, 긴 모션 시퀀스를 전송하는 데 사용할 수 있습니다. 이 방법의 핵심은 로컬 기하학적 및 시간적 변화를 개별적으로 예측하는 두 개의 결합된 신경망으로, 이후 최종 애니메이션 메시를 생성하기 위해 공간적 및 시간적으로 통합됩니다. 두 신경망은 공동으로 학습되고 공간 및 시간 신호를 생성하는 데 상호 보완적이며 3D 위치 정보로 직접 감독됩니다. 추론 중에 키 프레임이 없는 경우 이 방법은 본질적으로 모션 외삽 문제를 해결합니다. 이 연구는 다양한 메시(합성 및 스캔된 모양)에서 이 방법을 테스트하여 SoTA 대안에 비해 익숙하지 않은 신체 모양에 대한 사실적이고 자연스러운 애니메이션을 생성하는 데 탁월함을 보여줍니다. 추가 비디오 및 코드는 https URL에서 제공됩니다.



### Automatic Generation of Fashion Images using Prompting in Generative Machine Learning Models (https://arxiv.org/abs/2407.14944)
- **What's New**: 본 논문에서는 두 가지 대규모 언어 모델과 패션 이미지 생성을 위한 Stable Diffusion 모델을 사용하여 맞춤형 패션 설명을 생성하는 방법론을 조사합니다. AI 기반 패션 창의성에서의 적응성을 강조하기 위해 기존 접근 방식에서 벗어나 제로샷 및 퓨샷 학습, 사고 연쇄(CoT)와 같은 프롬프트 기술에 중점을 둡니다. 이는 다양한 색상과 질감을 생성하여 출력의 다양성을 높입니다. 우리 방법론의 핵심은 검색 증강 생성(RAG)으로, 패션 출처에서 얻은 통찰력으로 모델을 풍부하게 하여 현대적 표현을 보장합니다. 평가는 CLIPscore와 같은 정량적 지표와 질적 인간 판단을 결합하여 다양한 스타일에서 창의성, 일관성 및 미적 매력을 강조합니다. 참가자 중 RAG와 퓨샷 학습 기술은 더 관련성 있고 매력적인 패션 설명을 생성할 수 있는 능력으로 선호됩니다. 우리 코드는 [URL]에서 제공됩니다.



### RayFormer: Improving Query-Based Multi-Camera 3D Object Detection via Ray-Centric Strategies (https://arxiv.org/abs/2407.14923)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 이 논문에서는 카메라 광선 기반 쿼리 (RayFormer)를 활용하여 3D 객체 탐지의 정확도를 높이는 새로운 방법을 제안합니다. 기존의 쿼리 기반 멀티 카메라 3D 객체 탐지는 3D 공간에 쿼리를 초기화한 후, 투시 이미지에서 특징을 추출하여 쿼리를 반복적으로 개선합니다. 이러한 방식에서는 같은 카메라 광선 근처의 쿼리 포인트들이 매우 가까운 픽셀에서 유사한 특징을 추출하여 쿼리 특징이 모호해지고 탐지 정확도가 떨어지는 문제가 있습니다. RayFormer는 이러한 문제를 해결하기 위해 카메라의 광학적 특징을 활용하여 쿼리 초기화 및 특징 추출을 수행합니다. RayFormer는 투시 이미지 특징을 lift-splat-shoot 방법을 사용하여 조감도 (BEV) 로 변환하고, 카메라 광선을 기준으로 BEV 맵을 섹터로 분할합니다. 이후 각 카메라 광선을 따라 쿼리를 균일하고 희소하게 초기화하여 서로 다른 쿼리가 이미지의 서로 다른 영역에 투영되도록 하여 독립적인 특징을 추출할 수 있도록 합니다. 또한, 2D 객체 탐지 박스로부터 추가적인 쿼리를 추가하여 이미지의 인스턴스 정보를 활용하여 균일하게 초기화된 쿼리를 보완합니다. 그리고, 서로 다른 쿼리에 맞는 고유한 객체 수준의 특징을 추출하기 위해 이미지와 조감도 모두에서 특징 샘플링 포인트의 분포를 적절히 구성하는 광선 샘플링 방법을 설계합니다.



### RoIPoly: Vectorized Building Outline Extraction Using Vertex and Logit Embeddings (https://arxiv.org/abs/2407.14920)
- **What's New**: 본 논문은 공중 또는 위성 이미지에서 건물 윤곽 추출을 위한 새로운 RoI(Region-of-Interest) 기반 쿼리 접근 방식인 RoIPoly를 제안한다. RoIPoly는 각 꼭짓점을 쿼리로 표현하여 잠재적인 건물의 가장 관련성 있는 영역에 쿼리 주의를 제한함으로써 계산 오버헤드를 줄이고 꼭짓점 수준의 상호 작용을 더 효율적으로 만든다. 또한, 학습 가능한 로짓 임베딩을 도입하여 어텐션 맵에서 꼭짓점 분류를 용이하게 하므로, 중복 꼭짓점 제거를 위한 후처리가 필요하지 않다.



### PolyR-CNN: R-CNN for end-to-end polygonal building outline extraction (https://arxiv.org/abs/2407.14912)
- **What's New**: 이 논문은 PolyR-CNN이라는 새로운 end-to-end framework를 제안합니다. 이 프레임워크는 원격 감지 이미지에서 직접 벡터화된 건물 다각형과 경계 상자를 예측합니다. 또한, PolyR-CNN은 RoI(Region of Interest) 특징만을 활용하여 복잡한 설계가 필요하지 않습니다. 이 논문은 다각형 꼭지점 좌표에서 상세한 윤곽 정보를 추출하여 RoI 특징을 안내하는 새로운 방식을 제안합니다. 이를 통해 PolyR-CNN은 간단한 후처리 방법을 통해 Inria 데이터셋에서 구멍이 있는 건물을 처리할 수 있습니다.



### Self-supervised transformer-based pre-training method with General Plant Infection datas (https://arxiv.org/abs/2407.14911)
Comments:
          14 pages, 5 figures, 4 tables, 3 formulas

- **What's New**: 본 연구는 농업 분야의 해충 및 질병 분류 문제를 해결하기 위해 대규모 다양한 데이터셋을 구축하고, 대조 학습(Contrastive Learning)과 마스크 이미지 모델링(MIM)을 결합한 고급 네트워크 아키텍처를 제안합니다. 이 데이터셋은 다양한 식물 종과 해충 종류를 포함하고 있어 해당 분야에서 가장 크고 다양한 데이터셋 중 하나입니다. 제안된 네트워크 아키텍처는 식물 해충 및 질병 인식 작업에 효과적임을 보여주었으며, 뛰어난 검출 정확도를 달성했습니다. 이 접근 방식은 농업 생산 비용을 줄이는 빠르고 효율적이며 비용 효율적인 식물 해충 및 질병 검출을 위한 실행 가능한 솔루션을 제공합니다.



### Visual Geo-Localization from images (https://arxiv.org/abs/2407.14910)
Comments:
          18 pages, 8 figures,

- **What's New**: 본 논문에서는 GPS 데이터에 의존하지 않고 이미지에서 장소(건물과 도로 교차로)의 지리적 위치를 결정할 수 있는 시각적 지오 로컬라이제이션 시스템을 제시합니다. 이 시스템은 장소 인식을 위한 SIFT(Scale-Invariant Feature Transform), 도로 교차로 유형 식별을 위한 전통적인 이미지 처리, 그리고 도로 교차로 분류를 위한 VGG16 모델을 사용하는 딥 러닝을 결합합니다.



### Automated Patient Positioning with Learned 3D Hand Gestures (https://arxiv.org/abs/2407.14903)
- **What's New**: 이 논문은 MRI 스캐닝과 같은 의료 절차에서 환자 위치를 자동으로 조절하는 시스템을 제안합니다. 이 시스템은 카메라를 사용하여 기술자의 손동작을 인식하고 해석하여 의료 장비를 정확하게 움직여 환자를 원하는 위치에 배치합니다. 기존 방식은 수동으로 환자 지지대를 조절해야 했기 때문에 시간이 오래 걸리고 부정확할 수 있었지만, 이 시스템은 기술자의 손동작을 이용하여 자동으로 환자를 위치시키기 때문에 정확성과 효율성을 높일 수 있습니다.



### AGLLDiff: Guiding Diffusion Models Towards Unsupervised Training-free Real-world Low-light Image Enhancemen (https://arxiv.org/abs/2407.14900)
Comments:
          21 pages, 9 figures

- **What's New**: 이 논문은 실제 환경에서의 저조도 이미지 향상(LIE)을 위한 새로운 훈련 없는 방법인 특성 안내 확산 프레임워크(AGLLDiff)를 제안합니다. AGLLDiff는 왜곡/깨끗한 이미지 쌍의 수집이 어려운 실제 LIE의 문제점을 해결하기 위해 왜곡 프로세스를 명시적으로 정의하는 대신 이미지 노출, 구조, 색상과 같은 정상 광 이미지의 원하는 특성을 모델링하는 방식을 채택합니다.



### A New Dataset and Framework for Real-World Blurred Images Super-Resolution (https://arxiv.org/abs/2407.14880)
- **What's New**: 본 논문은 **흐릿한 이미지**에 대한 새로운 **Super-Resolution (SR) 데이터셋**인 ReBlurSR을 제안합니다. 또한,  PBaSR이라는 흐릿한 이미지에 최적화된 새로운 BSR 프레임워크를 제시합니다. PBaSR은 Cross Disentanglement Module (CDM)과 Cross Fusion Module (CFM)으로 구성됩니다. CDM은 흐릿함과 일반 이미지 데이터를 분리하여 최적화를 수행하며, CFM은 모델 보간을 통해 두 데이터 도메인에서 얻은 최적화된 정보를 효율적으로 결합합니다.



### Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts (https://arxiv.org/abs/2407.14872)
Comments:
          ECCV 2024 camera-ready

- **What's New**: 이 논문은 로봇이 다양한 환경에서 다양한 지시 사항을 수행할 수 있도록 일반화된 보상 함수를 만드는 새로운 방법을 제안합니다. 이 방법은 CLIP과 같은 비전-언어 모델을 활용하며, 소량의 데이터만 사용하여 새로운 환경과 지시에 대한 일반화를 달성합니다. 또한, 실패 영상의 패턴을 파악하여 모델의 성공/실패 구분 능력을 향상시킵니다.



### Dual High-Order Total Variation Model for Underwater Image Restoration (https://arxiv.org/abs/2407.14868)
Comments:
          13 pages, 10 figures

- **What's New**: 본 논문은 수중 이미지 개선 및 복원 (UIER)을 위한 새로운 변형 프레임워크를 제안합니다. 특히, 기존의 대부분 UIER 방법들이 대비 향상과 탈색 (dehazing)에 집중하여 이미지 내부의 조명 변화로 인한 국소 조명 차이를 고려하지 못한 반면, 이 논문에서는 확장된 수중 이미지 형성 모델 (UIFM)을 기반으로 국소 조명 차이를 고려한 변형 프레임워크를 제안합니다.



### An Explainable Fast Deep Neural Network for Emotion Recognition (https://arxiv.org/abs/2407.14865)
Comments:
          37 pages, 3 figures, 7 tables

- **What's New**: 본 연구는 비디오 분석을 통한 감정 분류에서 이진 딥 신경망의 설명 가능성 기술을 탐구합니다. 특히, 개선된 Integrated Gradients 설명 가능성 방법을 사용하여 얼굴 랜드마크 탐지를 통해 감정 인식을 위한 이진 분류기의 입력 특징을 최적화합니다. 본 논문의 주요 기여는 딥 러닝 기반 감정 분류기의 성능을 향상시키기 위해 감정 느낌 동안 중요한 얼굴 랜드마크 움직임을 이해하기 위해 혁신적인 설명 가능한 인공 지능 알고리즘을 사용하는 데 있습니다. 설명 가능성을 통해 우리는 얼굴 감정 인식을 위한 입력 특징으로 사용되는 얼굴 랜드마크의 수와 위치를 최적화하여 노이즈가 많은 랜드마크의 영향을 줄이고 개발된 모델의 정확도를 높일 수 있습니다. 제안된 접근 방식의 효과를 테스트하기 위해, 처음에는 완전한 얼굴 랜드마크 세트로 훈련된 감정 분류를 위한 이진 딥 모델 세트를 고려했으며, 적절한 최적화 절차에 따라 점차 줄여 나갔습니다. 얻어진 결과는 다양한 감정에 대한 다른 얼굴 점의 관련성을 이해하는 측면에서 제안된 설명 가능한 접근 방식의 견고성을 입증하며, 분류 정확도를 향상시키고 계산 비용을 줄입니다.



### CBCTLiTS: A Synthetic, Paired CBCT/CT Dataset For Segmentation And Style Transfer (https://arxiv.org/abs/2407.14853)
Comments:
          Accepted at VCBM 2024 - this https URL

- **What's New**: CBCTLiTS: 새로운 합성 CBCT 이미지 데이터셋을 소개합니다. 이 데이터셋은 다양한 수준의 품질 (5단계) 를 가진 CBCT 이미지와 고품질 CT 이미지를 제공하며, 단순 간 세분화부터 복잡한 간 종양 세분화까지 다양한 연구 시나리오에 활용 가능합니다. 특히, 프로젝션 수와 아티팩트 수준을 조절하여 품질을 조절할 수 있어 CBCT 이미지 품질이 연구의 자유 변수가 될 수 있다는 장점이 있습니다.



### A Tale of Single-channel Electroencephalogram: Devices, Datasets, Signal Processing, Applications, and Future Directions (https://arxiv.org/abs/2407.14850)
- **What's New**: This paper proposes a novel automatic evaluation metric, coined **Knowledge Dependent Answerability (KDA)**, to assess the educational value of Multiple Choice Questions (MCQ) generated by AI models. KDA evaluates the MCQ's ability to assess a student's knowledge of the target fact, unlike existing metrics that focus on n-gram similarity with gold samples. The authors introduce two automatic approximations of KDA, **KDA_disc** and **KDA_cont**, leveraging pre-trained language models to mimic student problem-solving behavior.



### Realistic Surgical Image Dataset Generation Based On 3D Gaussian Splatting (https://arxiv.org/abs/2407.14846)
Comments:
          This paper has already been accepted by INTERNATIONAL CONFERENCE ON MEDICAL IMAGE COMPUTING AND COMPUTER ASSISTED INTERVENTION (MICCAI 2024)

- **What's New**: 이 논문은 3D Gaussian Splatting을 사용하여 합성 수술 데이터셋을 생성하는 새로운 방법을 소개합니다. 3D Gaussian 표현을 추출하고 결합하여 고품질 합성 수술 시나리오를 생성하는 방법을 제안합니다.



### Text-based Talking Video Editing with Cascaded Conditional Diffusion (https://arxiv.org/abs/2407.14841)
- **What's New**: 본 논문에서는 Text-based talking-head video editing을 위해 효율적인 계단식 조건부 확산 기반 프레임워크를 제안합니다. 이 프레임워크는 오디오를 밀집 랜드마크 모션으로 변환하는 첫 번째 단계와 모션을 비디오로 변환하는 두 번째 단계로 구성됩니다. 첫 번째 단계에서 동적 가중치 인 컨텍스트 확산 모듈은 편집된 오디오를 입력으로 받아 밀집 랜드마크 모션을 합성합니다. 두 번째 단계에서 워핑 기반 조건부 확산 모듈은 편집 구간의 시작과 끝 프레임 사이를 보간하여 부드러운 중간 프레임을 생성합니다. 그런 다음 오디오-투-밀집 모션 이미지의 도움을 받아 이러한 중간 프레임을 왜곡하여 거친 중간 프레임을 얻습니다. 왜곡된 중간 프레임을 조건으로 하여 확산 모델이 채택되어 세부적이고 고해상도의 대상 프레임을 생성하여 일관성 있고 신원이 유지된 전환을 보장합니다. 계단식 조건부 확산 모델은 복잡한 토킹 편집 작업을 두 가지 유연한 생성 작업으로 분해하여 일반적인 토킹 페이스 표현, 매끄러운 오디오-비주얼 전환 및 소규모 데이터 세트의 신원이 유지된 페이스를 제공합니다.



### Can VLMs be used on videos for action recognition? LLMs are Visual Reasoning Coordinators (https://arxiv.org/abs/2407.14834)
Comments:
          LLMs, VLMs, Action Recognition

- **What's New**: 본 논문은 여러 개의 VLMs을 LLM을 통해 효율적으로 통합하는 'Cola Framework'를 제안하며, 이를 통해 각 VLM의 강점을 활용하여 상호 보완적인 능력을 발휘할 수 있음을 보여준다. 특히 A-OKVQA 데이터셋에서 뛰어난 성능을 확인하였으며, 이러한 통합 프레임워크를 감시 영상에서의 액션 인식 (action recognition)에 적용하는 가능성을 조사한다. 즉, LLM이 여러 개의 VLMs과 함께, 제한적인 프레임과 시간 정보만으로도 영상의 액션을 정확하게 추론할 수 있는지 탐구한다.



### Toward Efficient Convolutional Neural Networks With Structured Ternary Patterns (https://arxiv.org/abs/2407.14831)
Comments:
          Published in: IEEE Transactions on Neural Networks and Learning Systems Code: this https URL ImageNet-16 Dataset: this https URL

- **What's New**: 본 연구는 기존의 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)의 한계를 극복하고, MCQ의 교육적 가치를 평가하는 새로운 지표인 "지식 종속 가능성(Knowledge Dependent Answerability, KDA)"를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가할 수 있는지를 측정하는 지표입니다.



### CrossDehaze: Scaling Up Image Dehazing with Cross-Data Vision Alignment and Augmentation (https://arxiv.org/abs/2407.14823)
Comments:
          A cross-dataset vision alignment and augmentation technology is proposed to boost generalizable feature learning in the de-hazing task

- **What's New**: 이 논문에서는 이미지 dehazing 방법론을 개선하기 위해 내부 및 외부 데이터 증강을 제안합니다. (internal and external data augmentation). 외부 증강은 서로 다른 도메인에서 샘플을 가져와 모델이 더 강력하고 일반화된 특징을 학습하도록 돕습니다. (cross-data external augmentor). 내부 증강은 이미지 내의 로컬 정보를 활용하여 더 많은 이미지 디테일을 얻습니다.



### Blind Image Deconvolution by Generative-based Kernel Prior and Initializer via Latent Encoding (https://arxiv.org/abs/2407.14816)
Comments:
          ECCV@2024. Code: this https URL

- **What's New**: 본 논문에서는 Blind Image Deconvolution (BID) 문제를 해결하기 위해 Deep Image Prior (DIP) 기반의 새로운 프레임워크를 제안합니다. DIP는 최근 BID에서 뛰어난 성능을 보여주었지만, 비선형 최적화 과정으로 인해 초기화된 커널 (kernel)에 매우 민감한 문제점이 있었습니다. 이 문제를 해결하기 위해 본 논문은 생성적 적대 신경망 (Generative Adversarial Network, GAN) 기반의 커널 생성기 (kernel generator)와 커널 초기화기 (kernel initializer)를 사용합니다. 커널 생성기는 커널의 사전 정보 (prior)를 정확하게 특징화하고, 커널 초기화기는 잠재 공간 (latent space) 인코딩을 통해 블러 커널을 잘 초기화합니다. 이러한 사전 학습된 커널 생성기와 초기화기를 통해 더 나은 품질의 블러 커널 초기화를 얻을 수 있으며, 잠재 커널 매니폴드 (latent kernel manifold) 내에서 최적화를 수행할 수 있습니다. 이러한 프레임워크는 기존 DIP 기반 BID 방법보다 성능이 훨씬 향상되었습니다. 다양한 데이터셋에 대한 실험 결과는 제안된 방법의 효과를 보여줍니다.



### GaitMA: Pose-guided Multi-modal Feature Fusion for Gait Recognition (https://arxiv.org/abs/2407.14812)
Comments:
          Accepted to ICME 2024

- **What's New**: 본 연구는 기존의 실루엣 기반 gait recognition (보행 인식) 방법과 skeleton 기반 방법의 한계를 극복하기 위해, 두 modality (모달리티)를 결합하는 Gait Multi-model Aggregation Network (GaitMA)를 제안합니다. GaitMA는 silhouettes와 skeletons의 특징을 각각 CNN으로 추출하고, co-attention alignment module을 통해 특징을 정렬합니다. 또한, mutual learning module을 이용하여 cross-attention을 통해 feature fusion (특징 융합)을 수행하고 Wasserstein loss를 적용하여 두 modality의 효과적인 융합을 보장합니다.



### Decoupled Prompt-Adapter Tuning for Continual Activity Recognition (https://arxiv.org/abs/2407.14811)
- **What's New**: 본 논문은 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 답변 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다. KDA는 인간 설문 조사를 통해 학생의 응답을 기반으로 측정되며, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제시한다. 

- **Technical Details**: KDA는 MCQ가 주어진 대상 사실을 얼마나 잘 평가하는지 측정하는 메트릭으로, 학생들의 응답을 기반으로 측정된다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화한 자동 평가 메트릭이다. 

- **Performance Highlights**: 본 논문은 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 환경에서 전문가가 평가한 사용성과 높은 상관관계를 갖는다는 것을 보여준다. 또한, n-gram 기반 유사성 지표와 결합했을 때, KDA_disc와 KDA_cont가 전문가가 평가한 다양한 MCQ 품질 측정 지표에 대한 강력한 예측력을 갖는 것으로 나타났다.



### FairViT: Fair Vision Transformer via Adaptive Masking (https://arxiv.org/abs/2407.14799)
Comments:
          20 pages, The European Conference on Computer Vision (ECCV 2024)

- **What's New**: 본 논문은 비전 트랜스포머(ViT) 모델의 공정성 문제를 해결하기 위해 새로운 알고리즘인 FairViT를 제안합니다. FairViT는 정확성을 유지하면서 공정성을 향상시키는 것을 목표로 합니다.



### PASSION: Towards Effective Incomplete Multi-Modal Medical Image Segmentation with Imbalanced Missing Rates (https://arxiv.org/abs/2407.14796)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이 논문은 의료 이미지에서 불완전한 다중 모드 이미지 분할 문제를 다룹니다. 특히, 모드가 임의로 누락될 수 있는 실제 의료 환경에서 발생하는 불균형 누락률 문제를 해결하는 것을 목표로 합니다. 이는 기존의 완전한 모드 데이터가 학습 과정에 사용되는 방식과는 차이가 있습니다.



### FedPartWhole: Federated domain generalization via consistent part-whole hierarchies (https://arxiv.org/abs/2407.14792)
- **What's New**: 본 논문은 Federated Domain Generalization(FedDG) 문제를 해결하기 위해 백본 모델 아키텍처 관점에서 새로운 접근 방식을 제시합니다. FedDG는 다양한 클라이언트에서 오는 여러 도메인의 데이터를 중앙 집중식으로 저장하는 것을 방지하는 데이터 프라이버시 제약을 고려하면서 테스트 시간에 보이지 않는 도메인으로 일반화하는 과제를 해결하는 것을 목표로 합니다. 기존 접근 방식은 도메인 정렬, 데이터 조작, 학습 전략 및 모델 집계 가중치 최적화의 네 가지 그룹으로 크게 분류할 수 있습니다. 본 논문은 백본 모델 아키텍처 관점에서 문제를 해결하는 새로운 Federated Domain Generalization 접근 방식을 제안합니다. 핵심 원칙은 객체가 상당한 도메인 이동과 모양 변화에도 불구하고 부분과 전체의 일관된 계층 구조를 유지한다는 것입니다. 예를 들어, 개의 사진과 스케치는 머리, 몸통, 다리 등으로 구성된 동일한 계층적 구성을 공유합니다. 도입된 아키텍처는 이미지 구문 트리에 대한 기능 표현을 명시적으로 통합합니다. 이는 모델 아키텍처 관점에서 Federated Domain Generalization을 해결하는 최초의 연구입니다. 본 논문에서 제안한 접근 방식은 더 적은 매개변수를 사용함에도 불구하고 비슷한 크기의 합성곱 아키텍처보다 12% 이상 성능이 뛰어납니다. 또한 CNN의 블랙박스 특성과 달리 본질적으로 해석 가능하여 예측에 대한 신뢰를 높여주며, 이는 연합 학습에서 중요한 자산입니다.



### Intelligent Artistic Typography: A Comprehensive Review of Artistic Text Design and Generation (https://arxiv.org/abs/2407.14774)
Comments:
          GitHub Page: this https URL

- **What's New**: 자동 MCQ 생성 평가 지표인 KDA(Knowledge Dependent Answerability)를 새롭게 제안, 기존 n-gram 기반 지표의 단점을 보완하여 MCQ의 교육적 가치를 평가한다. KDA는 학생 응답 데이터를 기반으로 계산되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 개발하여 실제 KDA를 근사화한다. 

- **Technical Details**: KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하는 지표로, 학생 응답 데이터를 기반으로 계산된다. KDA_disc와 KDA_cont는 각각 차별적(discriminative) 및 연속적(continuous) 접근 방식을 활용하여 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방한다. 

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 강한 상관관계를 보이고, 전문가에 의해 평가된 실제 강의실 환경에서의 사용성과도 상관관계를 나타낸다는 것을 확인했다. 또한 n-gram 기반 유사도 지표와 결합하여 다양한 전문가 평가 MCQ 품질 지표에 대한 예측력이 뛰어남을 확인했다.



### Subgraph Clustering and Atom Learning for Improved Image Classification (https://arxiv.org/abs/2407.14772)
- **What's New**: 본 연구는 기존 CNN 기반 이미지 분류 모델의 한계를 극복하기 위해 새로운 하이브리드 모델인 Graph Sub-Graph Network (GSN)를 제안합니다. GSN은 이미지를 그래프로 표현하고, k-means clustering을 이용하여 그래프 노드를 클러스터링하여 서브그래프를 생성합니다. 이 서브그래프들을 사용하여 sparse하고 class-distinguishable한 특징들을 추출하는 dictionary learning을 수행합니다. 이러한 통합된 접근 방식은 의료 영상과 같이 미묘한 특징 차이를 구별해야 하는 분야에 특히 유용합니다.  



### DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-level Contro (https://arxiv.org/abs/2407.14758)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 지식 의존 가능성(Knowledge Dependent Answerability, KDA)라는 새로운 자동 평가 지표를 제안하여 MCQ 생성 모델의 품질을 평가합니다. KDA는 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 또한, 기존 n-gram 기반 유사성 지표(BLEU, ROUGE, METEOR)를 보완하여 MCQ의 교육적 가치를 더 정확하게 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: KDA는 학생 응답 데이터를 활용하여 계산되며, 두 가지 자동 평가 지표, KDA_disc와 KDA_cont가 제안되었습니다. 이 지표들은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: 본 논문에서 제시된 KDA_disc와 KDA_cont는 인간 평가 결과와 높은 상관관계를 보였으며, 실제 강의실 환경에서의 사용성과도 강한 연관성을 보여주었습니다. 또한, n-gram 기반 유사성 지표와 함께 사용하면 다양한 전문가 평가 지표에 대한 예측력이 높아지는 것으로 나타났습니다.



### Enhancing Skin Disease Classification Leveraging Transformer-based Deep Learning Architectures and Explainable AI (https://arxiv.org/abs/2407.14757)
Comments:
          Submitted to Computers in Biology and Medicine

- **What's New**: 이 논문은 피부병 분류를 위한 새로운 딥 러닝 모델을 제안하며, 다양한 비전 트랜스포머 (Vision Transformers), 스윈 트랜스포머 (Swin Transformers), 그리고 디노V2 (DivoV2)를 비교 분석합니다. 또한 기존의 컨볼루션 기반 아키텍처와 비교하여 성능을 평가합니다.



### Difflare: Removing Image Lens Flare with Latent Diffusion Mod (https://arxiv.org/abs/2407.14746)
Comments:
          Accepted by BMVC 2024

- **What's New**: 본 논문은 렌즈 플레어(lens flare)로 인해 손상된 이미지에서 고품질 이미지를 복구하는 새로운 방법인 Difflare를 소개합니다. Difflare는 사전 훈련된 확산 모델(PTDM)에서 학습된 생성적 사전(generative prior)을 활용하여 렌즈 플레어 제거 과정을 안내합니다. 또한 렌즈 플레어 제거에 관련된 물리적 사전(physical prior)을 고려하여 정보 손실을 줄이고 효율적인 훈련을 가능하게 합니다.



### A Comprehensive Review of Few-shot Action Recognition (https://arxiv.org/abs/2407.14744)
Comments:
          22 pages

- **What's New**: 본 논문은 Few-shot Action Recognition (FASR) 에 대한 포괄적인 서베이를 제공합니다. FASR은 각 클래스에 대한 라벨링 된 데이터가 적은 상황에서 비디오 데이터 내의 인간 행동을 정확하게 분류하는 것을 목표로 합니다.  특히 이미지 데이터와 달리, 비디오 데이터는 시간적 순서와 풍부한 의미 정보를 모델링해야 하는 과제를 가지고 있습니다. 이를 극복하기 위해 많은 연구들이 FASR 분야에서 큰 발전을 이루었고, 따라서 이러한 발전을 정리하는 서베이가 필요했습니다.



### Early Detection of Coffee Leaf Rust Through Convolutional Neural Networks Trained on Low-Resolution Images (https://arxiv.org/abs/2407.14737)
- **What's New**: 이 논문은 Deep Learning을 활용한 초기 커피 잎 녹병 탐지 (early coffee leaf rust detection) 기술을 제안하며, 리소스 제약 환경에서 모델 효율성을 향상시키기 위한 전처리 기법을 제안한다.



### MetaAug: Meta-Data Augmentation for Post-Training Quantization (https://arxiv.org/abs/2407.14726)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 연구는 기존의 Post-Training Quantization (PTQ) 방법론에서 발생하는 과적합 문제를 해결하기 위해 새로운 메타 학습 기반 접근 방식을 제안합니다. 이전 PTQ 방식은 학습 과정에서 검증 없이 원본 보정 데이터 세트만 사용하여 양자화된 모델을 훈련했기 때문에 과적합 문제가 발생했지만, 본 연구에서는 두 가지 다른 이미지 세트를 사용하여 양자화된 모델을 훈련 및 검증합니다. 특히, 메타 학습 기반 접근 방식을 통해 변환 네트워크와 양자화된 모델을 이중 수준 최적화를 통해 함께 최적화합니다. 변환 네트워크는 원본 보정 데이터를 수정하며, 수정된 데이터는 양자화된 모델이 원본 보정 데이터에서 좋은 성능을 달성하도록 훈련하는 데 사용됩니다.



### CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting (https://arxiv.org/abs/2407.14725)
- **What's New**: 본 논문에서는 CrowdMAC이라는 새로운 프레임워크를 제안하여 부분적으로 마스킹된 과거 군중 밀도 맵에서 미래 군중 밀도 맵을 예측하는 동시에 마스킹된 관측 맵을 재구성합니다.  CrowdMAC은 부분적으로 마스킹된 과거 군중 밀도 맵에서 미래 군중 밀도 맵을 예측하는 동시에 마스킹된 관측 맵을 재구성하여 훈련됩니다.  이를 통해 CrowdMAC은 보행자 누락으로 인해 불완전한 과거 밀도 맵에 강인한 모델을 만드는데 도움이 됩니다. 또한, 본 논문은 군중 밀도 맵의 희소성 (sparsity)과 예측 작업에 대한 후속 프레임의 정보량을 고려하여 관측된 군중 밀도 맵에서 토큰을 비균일하게 마스킹하는 Temporal-Density-aware Masking (TDM)을 제안합니다.  마지막으로, 훈련 효율성을 높이기 위해 multi-task masking을 도입했습니다.



### $\infty$-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions (https://arxiv.org/abs/2407.14709)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: 본 논문에서는 대규모 이미지 도메인에서 정교하고 도메인 특정 정보로부터 고해상도 이미지를 합성하는 새로운 조건부 확산 모델인  `∞`-Brush를 제안한다. 이 모델은 함수 공간(function space)에서 조건 지정을 가능하게 하는 크로스 어텐션 신경 연산자(cross-attention neural operator)를 사용하여 기존의 유한 차원 확산 모델과 패치 기반 방법의 제약을 극복한다.  `∞`-Brush는  4096x4096 픽셀까지 임의 해상도에서 이미지를 생성할 수 있는 최초의 함수 공간 조건부 확산 모델이다. (This paper introduces a novel conditional diffusion model in infinite dimensions, `∞`-Brush for controllable large image synthesis. The model employs a cross-attention neural operator to enable conditioning in function space, overcoming limitations of traditional finite-dimensional diffusion models and patch-based methods.  `∞`-Brush is the first conditional diffusion model in function space, capable of generating images at arbitrary resolutions up to 4096x4096 pixels.)



### On Learning Discriminative Features from Synthesized Data for Self-Supervised Fine-Grained Visual Recognition (https://arxiv.org/abs/2407.14676)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 MCQ (Multiple Choice Questions) 자동 생성을 위한 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성(answerability)을 측정하며, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 이 지표는 기존의 n-gram 기반 유사성 지표 (BLEU, ROUGE, METEOR)의 단점을 보완하고 MCQ의 교육적 가치를 더 잘 반영합니다.



### A New Lightweight Hybrid Graph Convolutional Neural Network -- CNN Scheme for Scene Classification using Object Detection Inferenc (https://arxiv.org/abs/2407.14658)
- **What's New**: 본 논문은 자동으로 MCQ를 생성하는 과정에서 교육적인 가치를 고려하는 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 수준을 평가하는 데 도움을 줍니다.  기존 평가 지표들은 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 측정에 집중했지만, 실제 학습 과정에서 MCQ의 효용성은 고려하지 않았습니다.  KDA는 학생들의 응답을 분석하여 MCQ의 질을 평가하고,  KDA_disc 및 KDA_cont와 같은 자동 평가 지표를 통해  실제 강의실 환경에서의 사용성을 측정합니다.  본 연구에서는 KDA_disc 및 KDA_cont가 KDA와 높은 상관관계를 가지며, 전문가가 평가한 MCQ 품질 척도와 높은 예측력을 보여주는 것을 확인했습니다.



### LORTSAR: Low-Rank Transformer for Skeleton-based Action Recognition (https://arxiv.org/abs/2407.14655)
Comments:
          12 pages

- **What's New**: 본 논문에서는 Skeleton-based Action Recognition에서 Transformer 모델의 크기를 줄이기 위해 Singular Value Decomposition (SVD)를 활용한 LORTSAR (LOw-Rank Transformer for Skeleton-based Action Recognition)을 제안합니다. 이 방법은 모델의 효율성을 높이고 인식 정확도는 유지하거나 심지어 향상시킵니다. SVD와 사후 압축 미세 조정은 인간 행동 인식에서 더 지속 가능하고 가볍고 고성능 기술을 가능하게 합니다.



### The Collection of a Human Robot Collaboration Dataset for Cooperative Assembly in Glovebox Environments (https://arxiv.org/abs/2407.14649)
- **What's New**: 본 논문은 산업 현장에서 사람과 로봇의 협업을 위한 손과 장갑 분할 데이터셋인 HAGS를 제안합니다. 기존 데이터셋은 주로 주거 또는 상업 환경에 초점을 맞추거나, 실제 환경에 적용하기 어려운 합성 데이터를 사용했으며, 안전한 협업에 필수적인 불확실성 추정 값이 부족했습니다. HAGS는 1200개의 도전적인 예제를 제공하여 실제 산업 현장에서 사람과 로봇의 협업 시나리오에서 손과 장갑 분할 애플리케이션을 구축할 수 있도록 합니다. 또한, 그린 스크린 증강을 통해 생성된 분포 외 이미지를 평가하여 ML 분류기의 견고성을 파악할 수 있습니다.



### Advancing Melanoma Diagnosis with Self-Supervised Neural Networks: Evaluating the Effectiveness of Different Techniques (https://arxiv.org/abs/2407.14628)
- **What's New**: 본 논문은 멜라닌 종양 패치를 분류하도록 훈련된 딥 러닝 모델의 정확성을 향상시키기 위한 자기 지도 학습(self-supervision)의 가능성을 조사합니다. 회전 예측, 누락된 패치 예측, 손상 제거와 같은 다양한 자기 지도 학습 기법들이 구현 및 평가되어 컨볼루션 신경망(CNN)의 성능에 미치는 영향을 분석합니다.



### The Research of Group Re-identification from Multiple Cameras (https://arxiv.org/abs/2407.14620)
- **What's New**: 이 논문에서는 그룹 재식별(Group Re-identification)이라는 새로운 문제를 제시합니다. 그룹 재식별은 여러 카메라에서 사람들을 재식별하는 기존의 방식과 달리, 그룹 단위로 사람들을 재식별하는 문제입니다. 이러한 문제는 기존의 재식별 작업에서 발생하는 시점 변화 및 사람의 자세 변화뿐만 아니라 그룹 레이아웃 변화 및 그룹 구성원 변화로 인한 어려움을 안고 있습니다.



### ESCAPE: Energy-based Selective Adaptive Correction for Out-of-distribution 3D Human Pose Estimation (https://arxiv.org/abs/2407.14605)
Comments:
          32 pages, 8 figures

- **What's New**: 이 논문에서는 인간 자세 추정(HPE) 분야에서 오류를 줄이고, 특히 손목과 발목 같은 먼 키포인트(distal keypoints)에 대한 오류를 줄이는 새로운 방법인 ESCAPE를 제안합니다. ESCAPE는 경량화된 교정 및 선택적 적응 프레임워크로, 대부분의 데이터에 대한 빠른 전방 패스 교정을 적용하고, OOD(out-of-distribution) 데이터에 대해서는 비용이 많이 드는 TTA(Test-Time Adaptation)를 예약하는 방법입니다. ESCAPE는 OOD 샘플을 분리하기 위해 자유 에너지 함수를 사용하며, 사전 학습된 백본 HPE 예측의 먼 키포인트 오류를 추정하는 교정 네트워크를 학습합니다. OOD 샘플의 경우, 먼 키포인트와 가까운 키포인트(어깨, 엉덩이) 사이의 제약 관계를 활용하는 두 번째 '역' 네트워크를 통해 교정 네트워크를 업데이트하는 새로운 자기 일관성 적응 손실을 제안합니다.



### A Comparative Study of Transfer Learning for Emotion Recognition using CNN and Modified VGG16 Models (https://arxiv.org/abs/2407.14576)
Comments:
          5 Pages, 9 figures

- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 해당 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생들의 응답을 분석하여 측정되며, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 사용하여 자동으로 측정할 수 있습니다.



### Are handcrafted filters helpful for attributing AI-generated images? (https://arxiv.org/abs/2407.14570)
Comments:
          9 pages, 5 figures

- **What's New**: 본 논문에서는 AI 생성 이미지를 식별하기 위한 새로운 방법을 제안합니다. 이 방법은 다 방향 하이 패스 필터(Multi-Directional High-Pass Filters, MHFs)를 사용하여 다양한 방향에서 미묘한 지문을 추출하여 이미지 생성 모델의 고유한 특징을 식별하는 데 효과적임을 보여줍니다. 또한, 방향 강화 특징 학습 네트워크(Directional Enhanced Feature Learning network, DEFL)를 도입하여 MHFs와 랜덤 초기화 필터를 모두 고려하여 컴팩트한 지문을 생성하고 이를 사용하여 다양한 이미지 생성 모델을 구분할 수 있도록 듀얼 마진 대조 손실(Dual-Margin Contrastive, DMC)을 사용했습니다. 마지막으로, 참조 기반 지문 분류 체계를 통해 이미지 속성을 효과적으로 수행합니다.



### Learning Visual Grounding from Generative Vision and Language Mod (https://arxiv.org/abs/2407.14563)
- **What's New**: 이 논문은 대규모 시각적 지면 데이터의 텍스트 주석을 확장하기 위해 이미지-텍스트 데이터로 주로 학습된 생성적 VLMs (Visual Language Models)을 활용하는 방법을 연구합니다. 생성적 VLMs는 적절한 프롬프트를 통해 지면 지식을 추출할 수 있습니다.



### Reconstructing Training Data From Real World Models Trained with Transfer Learning (https://arxiv.org/abs/2407.15845)
- **What's New**: 이 논문은 이미지 임베딩 공간에서 데이터 재구성을 통해 대규모 사전 훈련된 모델(DINO-ViT, CLIP 등)에서 전이 학습으로 훈련된 모델에 대한 데이터 재구성을 가능하게 하는 새로운 방법을 제시합니다. 이 방법은 기존 방법보다 훨씬 현실적인 환경에서 적용 가능하며, 이미지 해상도 및 학습 데이터 크기에 대한 제약이 적습니다. 또한, 클러스터링 기반 방법을 사용하여 수천 개의 후보 중에서 좋은 재구성 결과를 식별하여 기존 방법의 한계를 극복했습니다.



### Enhancing Cell Instance Segmentation in Scanning Electron Microscopy Images via a Deep Contour Closing Operator (https://arxiv.org/abs/2407.15817)
Comments:
          13 pages, 8 figures, 2 tables

- **What's New**: 본 연구에서는 SEM 이미지에서 세포 경계를 개선하기 위한 새로운 AI 기반 접근 방식을 제시하여 인스턴스 기반 세포 분할을 개선하고, 잔여 수동 보정의 필요성을 줄입니다. CNN COp-Net이라는 새로운 네트워크를 도입하여 세포 윤곽의 빈틈을 채우고, 부족하거나 없는 정보가 있는 영역을 효과적으로 채웁니다. 이 네트워크는 잠재적으로 불충분하거나 누락된 정보가 있는 세포 윤곽 확률 맵을 입력으로 받아 보정된 세포 윤곽 묘사를 출력합니다. 교육 데이터 부족 문제는 맞춤형 PDE를 사용하여 낮은 무결성 확률 맵을 생성하여 해결했습니다. 이 연구는 PDX 간세포종 조직의 개인 SEM 이미지와 공개적으로 사용 가능한 이미지 데이터 세트를 사용하여 세포 경계 정밀도 향상에 대한 접근 방식의 효능을 보여줍니다. 제안된 세포 윤곽 폐쇄 연산자는 테스트된 데이터 세트에서 주목할 만한 개선을 보여주며, 최첨단 방법과 비교하여 정확하게 묘사된 세포 비율에서 각각 약 50%(개인 데이터) 및 10%(공개 데이터) 증가했습니다. 또한 수동 보정의 필요성이 크게 감소하여 전반적인 디지털화 프로세스를 용이하게 합니다. 이 결과는 특히 이미지 품질이 세포 경계의 무결성을 손상시키고 빈틈을 채울 필요가 있는 매우 어려운 영역에서 세포 인스턴스 분할 정확도가 눈에 띄게 향상되었음을 보여줍니다. 따라서 이 연구는 궁극적으로 종양 조직 생체 구조의 연구를 촉진해야 합니다.  



### Learning to Manipulate Anywhere: A Visual Generalizable Framework For Reinforcement Learning (https://arxiv.org/abs/2407.15815)
Comments:
          Webpage: this https URL

- **What's New**: 본 논문에서는 다양한 시각적 방해 요소들을 결합하여 학습된 로봇 정책이 일반화될 수 있도록 하는 시각적 강화 학습에 최적화된 일반화 가능한 프레임워크인 **Maniwhere**를 제안합니다. 여러 시점 간의 공유된 의미 정보와 대응 관계를 포착하기 위해 Spatial Transformer Network (STN) 모듈과 융합된 다중 시점 표현 학습 접근 방식을 소개합니다. 또한, RL 학습 과정을 안정시키고 시각적 일반화 능력을 강화하기 위해 커리큘럼 기반 랜덤화 및 증강 접근 방식을 사용합니다. Maniwhere의 효과를 보여주기 위해, 우리는 명료한 객체, 양손 조작 및 숙련된 손 조작 작업을 포함한 8가지 작업을 설계하고 3개의 하드웨어 플랫폼에서 Maniwhere의 강력한 시각적 일반화 및 sim2real 전이 능력을 보여줍니다. 실험 결과 Maniwhere는 기존 최첨단 방법보다 훨씬 뛰어난 성능을 보여줍니다.



### Adaptive Extensions of Unbiased Risk Estimators for Unsupervised Magnetic Resonance Image Denoising (https://arxiv.org/abs/2407.15799)
- **What's New**: 본 논문에서는 의료 영상에서 널리 나타나는 복잡한 노이즈 상황에서의 딥러닝 기반 이미지 노이즈 제거 (image denoising)를 위해  새로운 unsupervised 학습 전략을 제시하고 이를 벤치마킹한다. 특히, 이 논문에서는 Stein's Unbiased Risk Estimator (SURE)와 그 확장판 (eSURE) 그리고 새로운 구현 방식인 Extended Poisson Unbiased Risk Estimator (ePURE)을 소개한다.



### STAMP: Outlier-Aware Test-Time Adaptation with Stable Memory Replay (https://arxiv.org/abs/2407.15773)
Comments:
          Accepted by ECCV 2024

- **What's New**: 기존 MCQ 생성 평가 지표는 교육적 가치를 고려하지 않고 단어 유사도만 비교하여, 학생의 지식 평가 능력을 제대로 평가하지 못했습니다. 본 논문에서는 지식 종속 가능성(KDA)이라는 새로운 평가 지표를 제안하여 MCQ의 대답 가능성 (answerability)을 측정합니다. 이 지표는 학생의 대답을 통해 측정 가능하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표로 구현되어 학생의 문제 해결 능력을 모방하는 사전 훈련된 언어 모델을 활용합니다. 실제 강의실 환경에서 전문가들이 평가한 결과, KDA_disc와 KDA_cont는 KDA와 사용성과 높은 상관관계를 보였습니다. 또한, 이 지표들은 n-gram 기반 유사도 지표와 함께 사용될 때, 다양한 전문가 평가 MCQ 품질 측정 지표를 예측하는 능력이 높다는 것을 보여주었습니다. 



### Neural-based Video Compression on Solar Dynamics Observatory Images (https://arxiv.org/abs/2407.15730)
- **What's New**: 이 논문은 SDO (Solar Dynamics Observatory) 임무에서 수집된 이미지 데이터를 위한 새로운 신경망 기반 비디오 압축 기술을 제안합니다. 이 방법은 데이터의 시간적 및 공간적 중복성을 활용하여 기존의 H.264 및 H.265 코덱보다 높은 압축 비율을 달성합니다. 특히, Transformer 모델을 기반으로 한 아키텍처를 사용하여 입력 이미지의 지역 및 전역 정보를 효율적으로 포착합니다. 또한, 엔트로피 모델을 사용하여 잠재 표현의 확률 분포를 정확하게 모델링하고 엔트로피 디코딩 단계의 속도를 높입니다. 이 엔트로피 모델은 채널 의존적 접근 방식을 사용하며 체커보드 모양의 지역 및 전역 공간 컨텍스트를 활용합니다.



### SAM2CLIP2SAM: Vision Language Model for Segmentation of 3D CT Scans for Covid-19 Detection (https://arxiv.org/abs/2407.15728)
- **What's New**: 이 논문에서는 의료 이미지 분할 (segmentation) 을 위한 새로운 접근 방식을 제시하며, 이는 코로나19 검출을 위한 3D 흉부 CT 스캔 분류 (classification) 모델과 방법에 통합될 수 있습니다. 이 접근 방식은 CT 스캔을 분할하는 시각 언어 모델 (vision-language models) 과 코로나19 검출을 위한 RACNet이라는 심층 신경망 아키텍처 (deep neural architecture) 를 결합합니다. 특히, SAM2CLIP2SAM이라는 새로운 프레임워크가 도입되어, Segment Anything Model (SAM)과 Contrastive Language-Image Pre-Training (CLIP)의 장점을 활용하여 CT 스캔에서 오른쪽 및 왼쪽 폐를 정확하게 분할하고, 이 분할된 결과를 RACNet에 전달하여 COVID-19 및 비 COVID-19 사례를 분류합니다. 먼저 SAM은 CT 스캔의 각 슬라이스에 대한 여러 부분 기반 분할 마스크 (part-based segmentation masks) 를 생성합니다. 그런 다음 CLIP은 관심 영역 (ROIs), 즉 오른쪽 및 왼쪽 폐와 관련된 마스크만 선택합니다. 마지막으로 SAM은 이러한 ROIs를 프롬프트 (prompts) 로 사용하여 폐에 대한 최종 분할 마스크를 생성합니다. 두 개의 코로나19 주석이 달린 데이터베이스에 대한 실험 결과는, 이 방법이 CT 스캔 분할에 사용되었을 때 개선된 성능을 보여줍니다.



### YOLOv10 for Automated Fracture Detection in Pediatric Wrist Trauma X-rays (https://arxiv.org/abs/2407.15689)
Comments:
          The code will soon be made publicly available

- **What's New**: 본 논문은 YOLOv10을 이용하여 소아 손목 골절을 검출하는 최초의 연구입니다. 이 연구는 모델 복잡도, 아키텍처 확장, 이중 레이블 할당 전략의 변화가 검출 성능을 어떻게 향상시키는지 조사합니다.



### Differentiable Convex Polyhedra Optimization from Multi-view Images (https://arxiv.org/abs/2407.15686)
Comments:
          ECCV2024 this https URL

- **What's New**: 본 논문은 implicit field supervision에 의존하는 최근 방법의 한계를 해결하기 위해 볼록 다면체의 미분 가능한 렌더링을 위한 새로운 접근 방식을 제시합니다. 이 기술은 dual 변환을 통한 초평면 교차의 비미분 가능한 계산과 세 평면 교차를 통한 정점 위치 지정을 위한 미분 가능한 최적화를 결합하는 전략을 도입하여 3D 암시적 필드 없이도 그래디언트 기반 최적화를 가능하게 합니다. 이를 통해 형태 파싱에서 컴팩트 메쉬 재구성에 이르기까지 다양한 응용 프로그램에서 효율적인 형태 표현이 가능합니다. 이 연구는 이전 접근 방식의 과제를 극복할 뿐만 아니라 볼록 다면체로 형태를 표현하는 새로운 기준을 제시합니다.



### A Diffusion Model for Simulation Ready Coronary Anatomy with Morpho-skeletal Contro (https://arxiv.org/abs/2407.15631)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이 논문은 coronary artery에 device를 배치하는 physics-based simulation을 가능하게 하는 virtual intervention에 대한 연구로, Latent Diffusion Models (LDMs)을 이용하여 다양한 해부학적 구조에 동일한 device를 배치하는 counterfactual reasoning을 가능하게 하는 새로운 방법을 제안한다.



### Increasing the Robustness of Model Predictions to Missing Sensors in Earth Observation (https://arxiv.org/abs/2407.15512)
Comments:
          Accepted at the MACLEAN workshop in the ECML/PKDD 2024

- **What's New**: 이 논문은 다중 센서 지구 관측 (EO) 모델에서 누락된 데이터 문제를 해결하기 위한 두 가지 새로운 방법, 즉 입력 센서 드롭아웃 (ISensD) 및 앙상블 센서 불변 (ESensI)을 제안합니다. 이러한 방법은 모델 예측의 로버스트성을 향상시키는 데 효과적입니다.



### Subthalamic Nucleus segmentation in high-field Magnetic Resonance data. Is space normalization by template co-registration necessary? (https://arxiv.org/abs/2407.15485)
- **What's New**: 본 논문은 Deep Learning (DL) 모델을 이용하여 Parkinson's Disease (PD) 환자의 뇌 영상에서 Subthalamic Nucleus (STN)를 자동으로 분할하는 두 가지 방법을 비교 분석합니다. 하나는 뇌 템플릿에 맞추어 영상을 변환한 후 분할하는 방식이고, 다른 하나는 영상의 원래 공간 (native space) 에서 직접 분할하는 방식입니다.  두 방법 모두 nnUNet을 사용하였지만 데이터 전처리 및 후처리 과정은 다릅니다.



### Iterative approach to reconstructing neural disparity fields from light-field data (https://arxiv.org/abs/2407.15380)
Comments:
          12 pages, 7 figures

- **What's New**: 이 연구는 신경 필드 (Neural field)를 기반으로 한 암묵적이고 연속적인 장면 불일치 표현 (implicit, continuous representation of scene disparity)인 신경 불일치 필드 (NDF)를 제안하며, 라이트 필드 데이터에서 NDF 재구성의 역문제 (inverse problem)를 해결하기 위한 반복적 접근 방식을 제시한다. NDF는 3차원 장면에서 불일치 변화 (disparity variations)를 매끄럽고 정확하게 특성화할 수 있으며, 샘플링 오류 및 보간 부정확성 (interpolation inaccuracies)에 취약한 기존 불일치 맵 (disparity map)의 한계를 극복하여 임의의 해상도에서 불일치를 이산화 (discretize) 할 수 있다. 제안된 NDF 네트워크 아키텍처는 멀티 레이어 퍼셉트론 (multilayer perceptrons)과 결합된 해시 인코딩 (hash encoding)을 사용하여 텍스처 수준의 자세한 불일치를 포착하여 복잡한 장면의 기하학적 정보를 표현하는 능력을 향상시킨다. 라이트 필드 데이터에 내재된 공간-각도 일관성 (spatial-angular consistency)을 활용하여 라이트 필드 데이터에서 중앙 보기 이미지 (central view image)를 생성하는 미분 가능한 순방향 모델 (differentiable forward model)을 개발한다. 순방향 모델을 기반으로 미분 가능한 전파 연산자 (differentiable propagation operators)를 사용하여 NDF 재구성의 역문제를 위한 최적화 방식 (optimization scheme)을 구축한다. 또한, 최적화 방식에서 NDF를 재구성하기 위해 반복적 해법 방법 (iterative solution method)을 채택하여, 훈련 데이터 세트를 필요로 하지 않으며 다양한 획득 방법으로 캡처된 라이트 필드 데이터에 적용된다. 실험 결과는 제안된 방법을 사용하여 라이트 필드 데이터에서 고품질 NDF를 재구성할 수 있음을 보여준다. NDF를 통해 고해상도 불일치를 효과적으로 복구할 수 있으며, 장면 불일치를 암묵적이고 연속적으로 표현하는 능력을 입증한다.



### Efficient Multi-disparity Transformer for Light Field Image Super-resolution (https://arxiv.org/abs/2407.15329)
- **What's New**: 본 논문에서는 Multi-scale Disparity Transformer (MDT)를 소개합니다. MDT는 light field image super-resolution (LFSR)을 위해 특별히 고안된 Transformer로 기존 방식에서 발생하는 sub-aperture 이미지의 무분별한 처리로 인한 계산 중복 및 disparity entanglement 문제를 해결합니다. MDT는 각각의 disparity range에 특화된 독립적인 disparity self-attention (DSA)를 사용하는 multi-branch 구조를 특징으로 하여 계산 복잡성을 줄이고 disparity를 효과적으로 분리합니다. 이 구조를 기반으로 효율적인 LFSR 네트워크인 LF-MDTNet을 제시합니다. 실험 결과, LF-MDTNet은 기존 최첨단 방식보다 2x 및 4x 스케일에서 PSNR 기준 각각 0.37 dB 및 0.41 dB 향상된 성능을 보여주며, 더 적은 매개변수와 더 빠른 속도로 우수한 성능을 달성했습니다.



### Hierarchical Homogeneity-Based Superpixel Segmentation: Application to Hyperspectral Image Analysis (https://arxiv.org/abs/2407.15321)
- **What's New**: 본 논문은 Hyperspectral Image (HI) 분석을 위한 새로운 Multiscale Superpixel 방법을 제안하며, 이는 기존의 SLIC (Simple Linear Iterative Clustering) oversegmentation 알고리즘을 계층적으로 확장하여 특정 HI 특징을 고려합니다. 기존 Superpixel 방법은 HI 데이터의 높은 Spectral Dimension을 고려하지 않아 문제가 있었지만, 본 논문에서는 Spectral Homogeneity를 높이는 새로운 Robust Homogeneity Testing을 통해 가변적인 크기의 Superpixel을 생성합니다.  



### Appearance-Based Loop Closure Detection for Online Large-Scale and Long-Term Operation (https://arxiv.org/abs/2407.15304)
Comments:
          12 pages, 11 figures

- **What's New**: 이 논문에서는 대규모 및 장기간 작동을 위한 온라인 루프 클로저 탐지 (loop closure detection) 접근 방식을 제시합니다. 이 접근 방식은 메모리 관리 방법을 기반으로 하며, 루프 클로저 탐지를 위한 위치 수를 제한하여 실시간 제약 조건 내에서 계산 시간을 유지합니다.  



### MedEdit: Counterfactual Diffusion-based Image Editing on Brain MRI (https://arxiv.org/abs/2407.15270)
Comments:
          Accepted at MICCAI24 Simulation and Synthesis in Medical Imaging (SASHIMI) workshop

- **What's New**: 이 논문은 의료 이미지 편집을 위한 새로운 조건부 확산 모델인 MedEdit을 제안합니다. MedEdit은 뇌 위축과 같은 간접적인 병리학적 영향을 모델링하면서 원본 스캔의 무결성을 유지하는 동시에 특정 영역에 병리를 유도합니다.  



### Back-in-Time Diffusion: Unsupervised Detection of Medical Deepfakes (https://arxiv.org/abs/2407.15169)
- **What's New**: 본 논문에서는 의료 영상에서 조작된 콘텐츠를 감지하는 데 사용될 수 있는 새로운 이상 감지 모델을 제안합니다. 이 모델은 확산 모델(Diffusion Model)을 기반으로 하며, 조작된 의료 이미지를 감지하기 위해 모델이 의심스러운 이미지의 확산을 역전시키는 방식으로 작동합니다. 기존의 이미지 조작 감지 기법과 달리, 의료 이미지의 고유한 포렌식 특징을 고려하여 개발되었습니다.



### AsyCo: An Asymmetric Dual-task Co-training Model for Partial-label Learning (https://arxiv.org/abs/2407.15036)
Comments:
          15 pages, accepted by Science China, Information Science

- **What's New**: 본 논문에서는 부분 라벨 학습(PLL) 모델에서 발생하는 에러 누적 문제를 해결하기 위해 비대칭 듀얼 태스크 코트레이닝 PLL 모델인 AsyCo를 제안합니다. AsyCo는 두 개의 네트워크, 즉 해소 네트워크(disambiguation network)와 보조 네트워크(auxiliary network)가 서로 다른 뷰에서 명확하게 학습하도록 강제하는 방식으로 작동합니다. 해소 네트워크는 자기 학습 PLL 태스크를 통해 라벨 신뢰도(label confidence)를 학습하고, 보조 네트워크는 학습된 라벨 신뢰도를 기반으로 생성된 잡음이 있는 쌍방향 유사성 라벨을 통해 지도 학습 방식으로 학습합니다. 마지막으로 정보 증류(information distillation)와 신뢰도 개선(confidence refinement)을 통해 에러 누적 문제를 완화합니다.



### Non-Reference Quality Assessment for Medical Imaging: Application to Synthetic Brain MRIs (https://arxiv.org/abs/2407.14994)
Comments:
          MICCAI 2024 workshop on Deep Generative Models

- **What's New**: 이 논문은 의료 영상, 특히 뇌 MRI의 품질을 평가하기 위한 새로운 참조 이미지가 필요 없는(non-reference) 딥 러닝 기반 방법을 제안합니다. 이 방법은 3D ResNet을 학습하여 MRI 스캔에서 흔히 발생하는 여섯 가지 종류의 아티팩트(artifact)를 측정합니다. 또한, 다양한 데이터셋으로 학습된 확산 모델을 사용하여 고품질의 합성 3D 이미지를 생성합니다. 이 방법은 여러 데이터셋을 사용하여 학습하고, 실제 이미지와 합성 이미지 모두에 대한 최첨단 품질 평가 지표와 비교하여 평가합니다.  



### Deep Learning CT Image Restoration using System Blur and Noise Models (https://arxiv.org/abs/2407.14983)
- **What's New**: 이 논문에서는 컴퓨터 단층 촬영(CT) 이미지와 같은 의료 영상 분야에서 블러 및 노이즈로 인해 손상된 이미지를 복원하는 새로운 방법을 제안합니다. 기존의 접근 방식에서는 블러 및 노이즈를 모델링하고 예측하여 복원에 활용했지만, 딥 러닝 접근 방식은 이미지 입력만 사용하여 맹목적으로 복원을 시도하는 경우가 많았습니다. 본 논문에서는 손상된 이미지 입력과 시스템 블러 및 노이즈 특성을 모두 활용하는 방법을 제시하여 모델링 및 딥 러닝 접근 방식을 결합합니다. 또한, 보조 입력을 CNN 아키텍처에 쉽게 통합할 수 있도록 입력 변형 및 가중치 변형 방식을 제시합니다. 제안된 모델은 보조 입력이 없는 기준 모델에 비해 우수한 성능을 보여줍니다.



### CoCoG-2: Controllable generation of visual stimuli for understanding human concept representation (https://arxiv.org/abs/2407.14949)
- **What's New**: 본 논문은 MCQ 생성 시 MCQ의 교육적 가치를 평가하는 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 특정 사실에 대한 학생의 지식을 얼마나 잘 평가하는지를 측정합니다. 이 지표는 실제 학생들의 응답을 기반으로 산출되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하여 KDA를 근사화합니다. 이 지표들은 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 방식을 모방합니다.



### Hyperspectral Unmixing Under Endmember Variability: A Variational Inference Framework (https://arxiv.org/abs/2407.14899)
- **What's New**: 본 논문에서는  Endmember Variability (HU-EV)가 존재하는 상황에서 hyperspectral unmixing (고해상도 분광 영상 분석)을 위한 variational inference (VI) 프레임워크를 제안합니다.  Endmember Variability를 고려한 noisy linear mixture model (LMM)이 사용되며, outlier (이상치) 역시 모델에 포함됩니다. Marginalized maximum likelihood (MML) 원칙을 따르는 VI 알고리즘 구조를 설계하여 HU-EV에 대한 확률적 추론을 수행합니다. 특히, Patch-wise static endmember 가정을 통해 spatial smoothness를 활용하여 HU-EV 문제의 ill-posed nature (잘못 설정된 문제)를 극복하려고 합니다.  이 설계는 다양한 Endmember priors 하에서 가벼운 연속 최적화 기반 업데이트를 가능하게 합니다.  Beta prior와 같은 일부 priors는 이전에 계산량이 많은 샘플 기반 확률적 HU-EV 방법에서 사용되었습니다. 제안된 프레임워크의 효과는 합성 데이터, 반실제 데이터, 실제 데이터 실험을 통해 입증되었습니다.



### MedMAE: A Self-Supervised Backbone for Medical Imaging Tasks (https://arxiv.org/abs/2407.14784)
- **What's New**: 본 논문은 의료 영상 분야의 학습 데이터 부족 문제를 해결하기 위해, 대규모 비지도 학습 데이터셋을 구축하고, 이를 기반으로 자기 지도 학습 기법인 Masked autoencoder를 활용하여 의료 영상 데이터에 특화된 사전 학습 모델을 제안합니다. 이 모델은 다양한 유형의 의료 영상 데이터를 학습하여 의료 영상 분야에서 사용할 수 있는 기반 모델로 사용될 수 있습니다.



### Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures (https://arxiv.org/abs/2407.14754)
- **What's New**: 이 논문은 긴 튜브 형태의 구조를 정확하게 분할하기 위한 새로운 딥 러닝 모델을 제안합니다. 이 모델은 프랙탈 특징을 활용하여 튜브 구조의 위상학적 자기 유사성 (topological self-similarity)을 고려합니다. 특히, 프랙탈 차원 (fractal dimension, FD)을 슬라이딩 윈도우 기법을 통해 픽셀 수준으로 확장하여 프랙탈 특징 맵 (fractal feature maps, FFMs)을 생성합니다. 이 FFMs은 모델의 입력 및 손실 함수의 가중치로 사용되어 분할 성능을 향상시킵니다. 또한, 이 논문은 U-Net 아키텍처를 확장하여 에지 디코더 (edge decoder)와 스켈레톤 디코더 (skeleton decoder)를 통합하여 경계 정확도와 스켈레톤 연속성을 개선합니다.



### ECRTime: Ensemble Integration of Classification and Retrieval for Time Series Classification (https://arxiv.org/abs/2407.14735)
- **What's New**: 기존 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ와 골드 샘플의 유사성만 평가하며, 교육적 가치를 고려하지 않습니다. 본 연구는 **지식 종속 가능성(KDA)**이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성과 학생의 지식 평가 능력을 측정합니다.

- **Technical Details**: KDA는 Human evaluation을 통해 측정되지만, 두 가지 자동 평가 메트릭인 **KDA_disc**와 **KDA_cont**는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다.

- **Performance Highlights**: Human evaluation 결과 **KDA_disc**와 **KDA_cont**는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보입니다. 또한, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가 평가 MCQ 품질 측정치에 대한 예측력이 강력해집니다.



### FedDM: Enhancing Communication Efficiency and Handling Data Heterogeneity in Federated Diffusion Models (https://arxiv.org/abs/2407.14730)
Comments:
          13 pages,3 figures, 2 algorithms, 3 tables

- **What's New**: 본 논문에서는 FedDM이라는 새로운 연합 학습 프레임워크를 소개하며, 연합 환경에서 확산 모델(diffusion model)의 학습 수렴을 위한 이론적 분석을 제공합니다. FedDM은 연합 학습을 위한 확산 모델 훈련에 적합하며, 수렴을 보장하는 구체적인 조건을 제시합니다. 논문은 U-Net 아키텍처를 백본으로 사용하는 다양한 훈련 알고리즘을 제안합니다. 여기에는 기본 연합 평균(Federated Averaging) 변형인 FedDM-vanilla, 클라이언트 간 데이터 이질성(heterogeneity)을 처리하기 위한 FedDM-prox, 그리고 모델 업데이트 크기를 줄여 연합 네트워크 간의 통신 효율성을 높이는 양자화 모듈(quantization module)을 통합한 FedDM-quant가 포함됩니다.



### Universal Medical Imaging Model for Domain Generalization with Data Privacy (https://arxiv.org/abs/2407.14719)
- **What's New**: 이 논문에서는 교육적 가치를 고려하는 새로운 자동 평가 메트릭, KDA를 제안하여 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생의 응답을 통해 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다.



### Productivity profile of CNPq scholarship researchers in computer science from 2017 to 2021 (https://arxiv.org/abs/2407.14690)
Comments:
          15 pages, 14 figures

- **What's New**: 이 논문은 브라질의 과학기술연구위원회(CNPq)에서 수여하는 연구 생산성 장학금(PQ) 수혜자 185명의 연구 생산성을 평가한 결과를 제시합니다. 연구 생산성은 2017년부터 2021년까지 5년 동안 학술지와 학회에 발표된 논문을 기반으로 분석되었습니다. 논문 수와 품질, 지역, 대학교, 연구 시설 분포, 공동 저자 네트워크 등을 분석했습니다.



### Improving Representation of High-frequency Components for Medical Foundation Models (https://arxiv.org/abs/2407.14651)
- **What's New**: 본 논문은 의료 영상 분야에서 고주파 성분과 미세한 디테일을 효과적으로 표현하기 위해 새로운 사전 학습 전략인 Frepa(Frequency-advanced Representation Autoencoder)를 제안합니다. Frepa는 고주파 마스킹과 저주파 섭동을 적용하여 인코더가 이미지 임베딩에서 고주파 성분을 효과적으로 표현하고 유지하도록 합니다. 또한, 기존의 Masked Autoencoder 방식을 ViT 뿐만 아니라 Swin Transformer와 Convolutional Network와 같은 다른 아키텍처로 확장하는 히스토그램 평활 이미지 마스킹 전략을 새롭게 도입합니다.



### Deep Learning-based 3D Coronary Tree Reconstruction from Two 2D Non-simultaneous X-ray Angiography Projections (https://arxiv.org/abs/2407.14616)
Comments:
          16 pages, 13 figures, 3 tables

- **What's New**: 이 연구는 2개의 비동시적 X선 촬영 투사로부터 3D 관상동맥 나무 재구성을 달성하기 위해 딥러닝을 활용하는 최초의 연구입니다. 특히, 워터스테인 조건부 생성적 적대 신경망 (Wasserstein conditional generative adversarial network) 에 기울기 페널티 (gradient penalty), 잠재적 합성곱 변환기 레이어 (latent convolutional transformer layers), 그리고 동적 스네이크 합성곱 비평가 (dynamic snake convolutional critic) 을 사용하여 비강성 운동 (non-rigid motion) 을 보완하고 3D 관상동맥 나무를 재구성합니다.  



### Detecting and Characterising Mobile App Metamorphosis in Google Play Stor (https://arxiv.org/abs/2407.14565)
Comments:
          15 pages, 14 figures

- **What's New**: 본 논문에서는 앱의 변신(app metamorphosis)이라는 새로운 현상을 정의하고, 앱 마켓에서의 변화를 효과적으로 파악하기 위한 새로운 멀티모달 검색 방법론을 제안한다. 앱 변신은 앱이 기능 개선이나 버그 수정을 위한 점진적인 업데이트가 아닌, 사용 사례나 시장 포지셔닝을 크게 바꾸는 경우를 말한다.



### APS-USCT: Ultrasound Computed Tomography on Sparse Data via AI-Physic Synergy (https://arxiv.org/abs/2407.14564)
Comments:
          MICCAI

- **What's New**: 이 논문에서는 sparse data를 효율적으로 활용하여 USCT 이미지 재구성을 향상시키는 새로운 USCT 방법인 APS-USCT를 제안합니다. APS-USCT는 sparse data를 dense waveform으로 변환하여 재구성 전 샘플 밀도를 높이는 APS-wave와 속도를 직접 재구성하는 APS-FWI라는 두 가지 주요 구성 요소로 구성됩니다. 또한 SE 블록과 소스 인코딩 기술을 추가하여 모델 성능을 향상시켰습니다.



New uploads on arXiv(cs.AI)

### Personalized and Context-aware Route Planning for Edge-assisted Vehicles (https://arxiv.org/abs/2407.17980)
- **What's New**: 이 연구는 자율 주행 시대에 개인 맞춤형 경로 계획을 제공하기 위해 그래프 신경망 (GNN)과 심층 강화 학습 (DRL)을 결합한 새로운 프레임워크를 제안합니다. 기존의 경로 계획 서비스는 모든 운전자에게 동일한 경로를 제공하는 반면, 이 연구는 개별 운전자의 선호도를 고려하여 개인화된 경로 계획을 제공하는 데 중점을 둡니다. 



### Pruning Boolean d-DNNF Circuits Through Tseitin-Awareness (https://arxiv.org/abs/2407.17951)
Comments:
          submitted to ICTAI 2024

- **What's New**: 이 연구에서는 d-DNNF 회로 컴파일 과정에서 발생하는 **Tseitin 아티팩트** (Tseitin artifacts)라는 무관한 하위 회로를 제거하는 방법을 제안합니다. Tseitin 아티팩트는 d-DNNF 컴파일러에서 CNF 변환을 위해 사용하는 **Tseitin 변환** (Tseitin transformation) 과정에서 발생합니다. Tseitin 아티팩트를 제거하여 더 간결한 회로를 생성하고, 확률적 추론 (probabilistic inference) 등 하위 작업의 성능을 향상시킵니다.



### Long-term Fairness in Ride-Hailing Platform (https://arxiv.org/abs/2407.17839)
Comments:
          Accepted by ECML PKDD 2024

- **What's New**: 이 논문은 기존의 자동 MCQ 생성 평가 메트릭이 생성된 MCQ의 교육적 가치를 고려하지 않는다는 문제를 해결하기 위해 새로운 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안한다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정한다. 이 논문에서는 KDA를 측정하는 방법을 제시하고, KDA를 근사화하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안한다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방한다.



### Investigating learning-independent abstract reasoning in artificial neural networks (https://arxiv.org/abs/2407.17791)
- **What's New**: 본 논문에서는 교육적 가치를 고려하지 않는 기존 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)의 문제점을 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 지표를 제안한다. KDA는 MCQ의 답변 가능성을 측정하고, 해당 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다.



### A process algebraic framework for multi-agent dynamic epistemic systems (https://arxiv.org/abs/2407.17537)
- **What's New**: This paper proposes a new framework for modeling and analyzing multi-agent, knowledge-based, dynamic systems by combining the classical model of labeled transition systems (LTSs) with the epistemic model for reasoning about knowledge. It introduces Kripke labeled transition systems (KLTSs), which associate a Kripke model with each state of an LTS, enabling the representation of both dynamic temporal behaviors and what agents know or do not know. This framework includes a logic with dynamic and epistemic modalities and a high-level, process-algebraic specification language for modeling agent-oriented concurrent systems.



### LAMBDA: A Large Model Based Data Agen (https://arxiv.org/abs/2407.17535)
Comments:
          30 pages, 21 figures and 5 tables

- **What's New**: 이 논문에서는 LAMBDA라는 새로운 오픈소스, 코드 없는 멀티 에이전트 데이터 분석 시스템을 소개합니다. LAMBDA는 대규모 모델의 힘을 활용하여 복잡한 데이터 기반 애플리케이션에서 데이터 분석 과제를 해결하도록 설계되었습니다. LAMBDA는 자연어를 사용하여 반복적으로 생성적으로 작동하는 혁신적으로 설계된 데이터 에이전트를 사용합니다. LAMBDA의 핵심에는 프로그래머와 검사관이라는 두 가지 주요 에이전트 역할이 있습니다. 프로그래머는 사용자의 지시 사항과 도메인 특정 지식을 기반으로 코드를 생성하고 고급 모델로 향상시킵니다. 한편, 검사관은 필요할 때 코드를 디버깅합니다. LAMBDA는 강력한 기능을 제공하고 불리한 시나리오를 처리하기 위해 운영 루프에서 직접 사용자 개입이 가능한 사용자 인터페이스를 제공합니다. 또한 LAMBDA는 지식 통합 메커니즘을 통해 외부 모델과 알고리즘을 유연하게 통합하여 맞춤형 데이터 분석 요구 사항을 충족합니다. LAMBDA는 다양한 머신 러닝 데이터셋에서 강력한 성능을 보여주었습니다. LAMBDA는 인간과 인공 지능을 원활하게 통합하여 데이터 과학 실무와 분석 패러다임을 개선하여 다양한 배경을 가진 개인이 더 쉽게 접근하고 효과적이고 효율적으로 사용할 수 있도록 합니다. 여러 사례 연구에서 데이터 과학 문제 해결에 있어 LAMBDA의 강력한 성능이 입증되었으며, 이는 [링크](this https URL)에서 제공됩니다.



### Driving pattern interpretation based on action phases clustering (https://arxiv.org/abs/2407.17518)
- **What's New**: 이 논문은 Action phase를 이용하여 무감독 방식으로 운전 패턴을 분류하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Resampling and Downsampling Method (RDM)을 사용하여 Action phase의 길이를 표준화한 후, Feature Selection, Clustering Analysis, Difference/Similarity Evaluation, Action phases Re-extraction을 포함하는 클러스터링 보정 프로세스를 반복적으로 적용하여 모든 클러스터 간의 차이와 클러스터 내의 유사성이 사전에 결정된 기준에 도달할 때까지 수행합니다.  



### Differentiable Quantum Architecture Search in Asynchronous Quantum Reinforcement Learning (https://arxiv.org/abs/2407.18202)
Comments:
          Accepted by IEEE International Conference on Quantum Computing and Engineering - QCE 2024

- **What's New**: 이 논문은 교육적 가치를 고려하여 MCQ (Multiple Choice Question) 의 품질을 평가하는 새로운 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. 기존의 지표인 BLEU, ROUGE, METEOR는 생성된 MCQ가 원본 문장과 얼마나 유사한지를 단어 수준에서 비교하지만, KDA는 MCQ가 실제로 학생의 지식을 평가하는 데 얼마나 효과적인지 측정합니다. KDA는 Human evaluation을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 측정 지표를 제안하여 실제 교육 환경에서의 MCQ의 유용성과 강한 상관관계를 보여줍니다.



### Gene Regulatory Network Inference from Pre-trained Single-Cell Transcriptomics Transformer with Joint Graph Learning (https://arxiv.org/abs/2407.18181)
Comments:
          Accepted into the ICML 2024 AI for Science workshop

- **What's New**: 본 연구에서는 scRNA-seq 데이터에서 Gene Regulatory Networks (GRNs)를 추론하는 새로운 방법을 제시합니다. 이 방법은 사전 훈련된 단일 세포 BERT 기반 트랜스포머 모델 (scBERT)과 기존 GRNs의 구조화된 생물학적 지식을 결합하여 작동합니다. scBERT는 방대한 양의 라벨 없는 scRNA-seq 데이터로 훈련되어 단일 세포의 gene expression 패턴과 상호 작용을 학습합니다. 그런 다음, scBERT는 사용자 지정 scRNA-seq 데이터셋으로 미세 조정되어 배치 효과를 완화하고 잠재적인 gene-gene 상호 작용을 포착합니다. 이렇게 훈련된 scBERT를 기반으로 GRN의 구조화된 지식과 결합하는 scTransNet을 개발했습니다. scTransNet은 scBERT로부터 학습한 gene 표현과 GRNs로부터 파생된 그래프 표현을 결합하여 통합된 컨텍스트 인식 및 지식 인식 표현을 생성합니다. 이 공동 학습 방식을 통해 scRNA-seq 데이터에서 제공하는 gene expression 수준 제약 조건과 GRNs에 내재된 구조화된 생물학적 지식을 효과적으로 추론할 수 있습니다.



### PianoMime: Learning a Generalist, Dexterous Piano Player from Internet Demonstrations (https://arxiv.org/abs/2407.18178)
- **What's New**: 이 논문에서는 인터넷 데이터로부터 피아노 연주 로봇 에이전트를 훈련하는 새로운 프레임워크인 PianoMime을 소개합니다. 기존 연구와 달리 PianoMime은 Youtube에서 다양한 피아노 연주 영상을 활용하여 일반적인 연주 에이전트를 훈련합니다. 이 프레임워크는 데이터 준비, 정책 학습, 정책 증류의 세 단계로 나뉩니다. 각 단계는 Youtube 영상에서 유용한 정보 추출, 특정 곡에 대한 전문가 정책 학습, 그리고 전문가 정책들을 하나의 일반화된 에이전트로 통합하는 과정을 포함합니다. PianoMime은 다양한 정책 설계를 실험하고 훈련 데이터 양이 에이전트의 일반화 능력에 미치는 영향을 평가합니다. 결과적으로, PianoMime은 학습된 에이전트가 데이터셋에 포함되지 않은 새로운 곡을 최대 56%의 F1 점수로 연주할 수 있음을 보여줍니다.



### Quasar-ViT: Hardware-Oriented Quantization-Aware Architecture Search for Vision Transformers (https://arxiv.org/abs/2407.18175)
Comments:
          Accepted by ICS 2024

- **What's New**: 본 논문은 ViT (Vision Transformer)를 위한 하드웨어 지향적인 양자화 인식 아키텍처 검색 프레임워크인 Quasar-ViT를 제안합니다. 이 프레임워크는 하드웨어 구현에 적합한 효율적인 ViT 모델을 설계하여 정확도를 유지하는 동시에 효율성을 높이는 데 중점을 둡니다. Quasar-ViT는 슈퍼넷을 훈련하는 데 사용되는 혁신적인 방법들을 제시하며, FPGA 플랫폼에서의 모델 적응형 설계를 통해 이론적 계산 감소와 실제 추론 속도 향상 간의 차이를 줄입니다.



### Taxonomy-Aware Continual Semantic Segmentation in Hyperbolic Spaces for Open-World Perception (https://arxiv.org/abs/2407.18145)
- **What's New**: This paper introduces TOPICS (Taxonomy-Oriented Poincaré-regularized Incremental-Class Segmentation), a novel approach for class-incremental semantic segmentation that leverages hyperbolic space and taxonomy-tree structures to address the challenge of catastrophic forgetting while enabling the model to learn new classes effectively. TOPICS addresses limitations of existing methods by providing plasticity for old classes and incorporating pseudo-labeling of the background and relational constraints to ensure a robust structure for combating forgetting.



### Maximum Entropy On-Policy Actor-Critic via Entropy Advantage Estimation (https://arxiv.org/abs/2407.18143)
- **What's New**: 이 논문은  MaxEnt RL (Maximum Entropy Reinforcement Learning) 알고리즘을  on-policy actor-critic 설정에서 쉽게 구현할 수 있도록 하는 새로운 방법을 제안합니다. 특히, entropy reward를 따로 관리하여  MaxEnt RL의 장점을 활용하면서도 기존 알고리즘과의 호환성을 유지합니다.  Entropy Advantage Policy Optimisation (EAPO) 라는 새로운 방법을 제안하며, 기존의 PPO와 TRPO 알고리즘을 MaxEnt framework 안에서 확장하여 성능을 향상시킵니다.  



### Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images (https://arxiv.org/abs/2407.18125)
- **What's New**: 본 연구는 의료 영상에서 랜드 마크 검출을 위한 확산 모델 기반의 새로운 자기 지도 학습 전처리 프로토콜을 소개합니다. 이 프로토콜은 제한된 수의 레이블이 있는 훈련 이미지 (최대 50개) 를 사용하여 정확한 랜드 마크 검출을 가능하게 하며, ImageNet 지도 학습 전처리 및 최첨단 자기 지도 학습 전처리 성능을 능가합니다. 본 연구는 확산 모델을 자기 지도 학습 방식으로 랜드 마크 검출에 적용한 첫 번째 시도이며, 데이터 부족 문제를 완화하기 위한 유용한 전처리 접근 방식을 제공합니다.



### MapTune: Advancing ASIC Technology Mapping via Reinforcement Learning Guided Library Tuning (https://arxiv.org/abs/2407.18110)
Comments:
          IEEE/ACM International Conference on Computer-Aided Design (ICCAD '24), October 27--31, 2024

- **What's New**: 본 논문에서는 Reinforcement Learning을 활용하여 기술 매핑 성능을 최적화하는 새로운 프레임워크인 MapTune을 제안합니다. MapTune은 기술 라이브러리의 일부만을 선택적으로 사용하여 검색 공간을 줄이고, 디자인 특성에 맞는 셀 선택을 통해 매핑 품질을 향상시킵니다.



### Multi-Resolution Histopathology Patch Graphs for Ovarian Cancer Subtyping (https://arxiv.org/abs/2407.18105)
Comments:
          Initially submitted version of a paper which has been accepted in the GRAIL workshop at MICCAI 2024

- **What's New**: 이 논문은 다양한 해상도에서 조직 패치들의 공간적 관계를 이용하여 각 패치의 맥락(context)을 학습하는 다중 해상도 그래프 모델을 사용하여 난소 상피암 아형 분류를 위한 가장 철저한 검증(validation)을 수행합니다. 이 연구는 7개의 모델을 조정하고 Leeds Teaching Hospitals NHS Trust에서 치료받은 434명의 환자로부터 얻은 1864개의 전체 슬라이드 이미지(WSI) 세트에서 5중 교차 검증(five-fold cross-validation)을 사용하여 훈련했습니다. 교차 검증 모델은 앙상블(ensemble) 방식으로 만들어졌으며, 30명의 환자로부터 얻은 100개의 WSI로 구성된 균형(balanced) 홀드아웃 테스트 세트와 Transcanadian Study에서 얻은 80명의 환자로부터 얻은 80개의 WSI로 구성된 외부 검증 세트를 사용하여 평가되었습니다.



### Privacy Threats and Countermeasures in Federated Learning for Internet of Things: A Systematic Review (https://arxiv.org/abs/2407.18096)
- **What's New**: 본 논문은 IoT 환경에서 FL의 프라이버시 위협을 체계적으로 분석하고 이러한 위협을 완화하는 데 사용할 수 있는 방어 메커니즘을 평가합니다. 이는 IoT 기기의 제한된 특성으로 인해 FL이 프라이버시 및 보안 문제를 야기할 수 있기 때문입니다. 본 논문에서는 SLR(Systematic Literature Review) 방법을 사용하여 2017년부터 2024년 4월까지 발표된 관련 논문 49편을 분석했습니다. 이를 통해 추론 공격, 오염 공격 및 도청과 같은 다양한 프라이버시 위협을 파악하고, 차등 프라이버시 및 안전한 다자간 컴퓨팅과 같은 방어 메커니즘을 평가했습니다. 이러한 방어 메커니즘은 IoT 설정에서 FL의 기능적 무결성을 손상시키지 않고 프라이버시를 보호하는 효율성을 평가했습니다. 분석 결과, IoT 환경에 맞는 강력하고 효율적인 프라이버시 보호 전략의 필요성이 강조됩니다. 특히, 재생, 회피 및 모델 도용 공격에 대한 전략이 필요합니다. 경량 방어 메커니즘 및 블록체인과 같은 신기술을 탐색하면 IoT에서 FL의 프라이버시를 향상시킬 수 있습니다. 이는 변동하는 네트워크 조건에서 작동하는 FL 모델을 만드는 데 도움이 됩니다.



### GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution (https://arxiv.org/abs/2407.18046)
Comments:
          13 pages, 12 figures

- **What's New**: 이 논문은 Implicit Neural Representations (INRs) 기반의 Arbitrary-Scale Super-Resolution (ASSR) 기술에서 2D Gaussian Splatting (2DGS)을 활용하여 기존 방식의 한계를 극복한 새로운 방법인 GaussianSR을 제안한다. 기존 방법들은 픽셀을 discrete point로 취급하는 반면, GaussianSR은 픽셀을 continuous Gaussian field로 표현하여 개선된 성능을 제공한다.



### Peak-Controlled Logits Poisoning Attack in Federated Distillation (https://arxiv.org/abs/2407.18039)
Comments:
          arXiv admin note: text overlap with arXiv:2401.03685

- **What's New**: 이 논문은 연합 증류(FD) 시스템에서 로그릿 공격에 대한 새로운 방법인 PCFDLA를 제안합니다. PCFDLA는 로그릿의 신뢰도를 조절하여 모델이 부정확하지만 그럴듯한 결과를 선택하도록 유도하여 공격을 은밀하게 수행합니다. 또한, 공격 전후에 공격자와 피해자 모델의 정확도를 측정하는 새로운 평가 지표를 제안하여 공격의 효과를 더욱 명확하고 포괄적으로 보여줍니다.



### AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild (https://arxiv.org/abs/2407.18034)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 자동으로 다중 선택 질문(MCQ)을 생성하는 새로운 방법을 제안하며, 기존의 평가 지표인 BLEU, ROUGE, METEOR와 달리 MCQ가 교육적 가치를 얼마나 갖는지 측정하는 새로운 지표인 Knowledge Dependent Answerability(KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하는 지표입니다. 이 논문은 또한 KDA를 자동화하기 위한 두 가지 새로운 지표인 KDA_disc와 KDA_cont를 제안하고, 이러한 지표가 실제 교육 환경에서 유용성과 높은 상관관계를 갖는다는 것을 실험적으로 보여줍니다.



### Learning mental states estimation through self-observation: a developmental synergy between intentions and beliefs representations in a deep-learning model of Theory of Mind (https://arxiv.org/abs/2407.18022)
- **What's New**: 이 논문은 지식 기반 답변 가능성(KDA)이라는 새로운 자동 평가 지표를 제안하여 MCQ의 답변 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생들의 응답을 기반으로 측정되며, 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화합니다.



### Quadratic Advantage with Quantum Randomized Smoothing Applied to Time-Series Analysis (https://arxiv.org/abs/2407.18021)
Comments:
          Accepted at the IEEE International Conference on Quantum Computing and Engineering (QCE)

- **What's New**: 이 논문은 양자 랜덤 스무딩(Quantum Randomized Smoothing)의 강건성(Robustness) 분석과 데이터 인코딩 및 섭동 모델링 방식을 일치시켜 유의미한 강건성 인증을 얻는 방법을 제시합니다. 특히 Grover 알고리즘을 통합하여 기존의 고전적 랜덤 스무딩보다 2배 빠른 샘플링 이점을 얻습니다. 이 전략은 기저 상태 인코딩(Basis State Encoding)을 필요로 하므로 허용되는 섭동의 범위가 제한됩니다. 이 연구에서는 제한된 k-거리 해밍 가중치 섭동(Constrained k-distant Hamming weight Perturbations)이 적절한 잡음 분포임을 보여주고, 양자 컴퓨터에서 이를 구현하는 방법을 설명합니다. 제안된 프레임워크의 효과는 Bag-of-Words 전처리 솔루션을 사용한 시계열 분류 작업에서 입증되었습니다. 특히 샘플 수가 많은 경우 2배의 샘플 감소 이점을 얻을 수 있습니다. 이를 통해 양자 컴퓨터는 고전적 방법으로는 불가능한 더 복잡한 작업에 랜덤 스무딩을 효율적으로 확장할 수 있습니다.



### A Sensitivity Analysis of Cellular Automata and Heterogeneous Topology Networks: Partially-Local Cellular Automata and Homogeneous Homogeneous Random Boolean Networks (https://arxiv.org/abs/2407.18017)
- **What's New**: 이 논문은 MCQ 생성 모델의 교육적 가치를 평가하기 위한 새로운 지표인 '지식 종속 가능성(KDA)'을 제안합니다. KDA는 생성된 MCQ가 대상 사실에 대한 학생의 지식을 제대로 평가할 수 있는지 측정합니다. 기존 지표인 BLEU, ROUGE, METEOR는 단어 유사도에만 집중했던 반면, KDA는 학생 응답을 통해 MCQ의 질을 더 정확하게 판단합니다. 또한,  KDA를 자동으로 계산하는 두 가지 지표인 KDA_disc와 KDA_cont가 제시됩니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보여주며, 기존 지표와 함께 사용될 경우 MCQ 품질을 더욱 정확하게 예측할 수 있습니다.



### Relating the Seemingly Unrelated: Principled Understanding of Generalization for Generative Models in Arithmetic Reasoning Tasks (https://arxiv.org/abs/2407.17963)
- **What's New**: 본 논문은 MCQ 생성 모델의 교육적 가치를 평가하는 새로운 지표인 "지식 종속 가능성(Knowledge Dependent Answerability, KDA)"를 제안합니다. KDA는 MCQ가 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 기존 지표들은 MCQ의 문법적 유사성만 평가했지만, KDA는 실제 학생의 답변을 활용하여 교육적 가치를 더 정확하게 평가할 수 있습니다. 본 논문은 KDA를 근사화하는 자동 평가 지표인 KDA_disc와 KDA_cont를 제안하며, 실제 강의실에서의 사용성과 높은 상관관계를 보입니다.



### Real Time American Sign Language Detection Using Yolo-v9 (https://arxiv.org/abs/2407.17950)
Comments:
          11 pages, 13 figures, 1 table

- **What's New**: 본 논문은 실시간 미국 수어 검출에 초점을 맞추고 있습니다. YOLO는 2015년 처음 출시된 CNN 기반 모델로, 실시간 검출 능력으로 인해 최근 몇 년 동안 인기를 얻었습니다. 본 연구는 특히 2024년에 출시된 YOLO-v9 모델을 대상으로 합니다. YOLO-v9는 새롭게 출시된 모델이기 때문에, 수어 검출에 관한 연구가 많지 않으며, 특히 이 모델에 대한 연구는 많지 않습니다. 본 논문은 YOLO-v9가 어떻게 작동하는지, 그리고 이전 모델보다 뛰어난지에 대한 심층적인 통찰력을 제공합니다.



### Comparison of different Artificial Neural Networks for Bitcoin price forecasting (https://arxiv.org/abs/2407.17930)
Comments:
          9 pages, 8 figures, 2 tables

- **What's New**: 이 논문은 다양한 시퀀스 길이가 인공 신경망(ANN)을 사용한 암호화폐 수익률 예측의 정확도에 미치는 영향을 조사합니다. 평균 절대 오차(MAE)를 임계 기준으로 사용하여 소규모 수익률과 관련된 오류를 완화하여 작은 수익률을 제외함으로써 예측 정확도를 높이는 것을 목표로 합니다. 그 후 평가는 이 임계값을 초과하는 예측된 수익률의 정확도에 집중합니다. 4가지 시퀀스 길이(각각 168시간(7일), 72시간(3일), 24시간, 12시간)를 비교하고 각각 2시간의 수익률 예측 간격을 사용합니다. 연구 결과는 시퀀스 길이가 예측 정확도에 미치는 영향을 보여주고 금융 예측 모델에서 최적화된 시퀀스 구성의 가능성을 강조합니다.



### Invariance of deep image quality metrics to affine transformations (https://arxiv.org/abs/2407.17927)
Comments:
          12 pages 13 figures

- **What's New**: 이 논문은 기존 이미지 품질 평가 지표들이 인간의 시각적 인지 능력을 제대로 반영하지 못한다는 점을 지적하고, affine transformation (회전, 이동, 크기 조정, 스펙트럼 조명 변화) 에 대한 인간의 시각적 불변성 (invariance) 을 고려한 새로운 평가 방법을 제안한다. 이 방법은 특정 지표의 invisibility threshold (눈에 띄지 않는 변환의 한계) 를 정의하고, 인간의 시각적 인지와 비교하여 평가한다.



### The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models (https://arxiv.org/abs/2407.17915)
- **What's New**: 이 논문은 대규모 언어 모델 (LLM)의 함수 호출 기능에서 발생하는 새로운 보안 취약점을 발견하여 '탈옥 함수' 공격 방법을 소개한다. 이 공격은 LLM의 정렬 불일치, 사용자 강제 및 엄격한 안전 필터 부재를 악용한다.

- **Technical Details**: 이 연구는 GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-pro를 포함한 6가지 최첨단 LLM에서 이 공격의 평균 성공률이 90% 이상임을 밝혀냈다. 또한 함수 호출이 이러한 공격에 취약한 이유와 방어적 프롬프트 사용과 같은 방어 전략을 제안한다.

- **Performance Highlights**: 실험 결과, 제안된 '탈옥 함수' 공격은 평균 90% 이상의 성공률을 보였으며, 이는 LLM의 함수 호출 기능에서의 심각한 보안 문제를 시사한다. 또한, 방어적 프롬프트를 사용하는 방어 전략이 효과적임을 보여주었다.



### ReCorD: Reasoning and Correcting Diffusion for HOI Generation (https://arxiv.org/abs/2407.17911)
Comments:
          Accepted by ACM MM 2024. Project website: this https URL

- **What's New**: 이 논문은 HOI(Human-Object Interaction)를 정확하게 묘사하는 이미지 생성을 위한 새로운 학습-없는 방법인 ReCorD를 제안합니다. ReCorD는 Latent Diffusion Model(LDM)과 Visual Language Model(VLM)을 결합하여 HOI 생성 과정을 개선합니다.  특히, 이미지 내용을 이해하는 VLM의 능력을 활용하여  정확한 자세와 개체 배치를  가지고 있는 후보를 선택하고, 상호작용하는  시나리오를  정확하게 이해할 수 있는 상호 작용 인식 추론 모듈을 제안합니다.  또한, 이미지에서 인간의 자세를 유지하면서 개체의 위치를 조정하는 상호 작용 수정 모듈을 도입하여 더욱 정확한 HOI 생성을 수행합니다.  ReCorD는 이미지를 생성하는 동안 인간과 개체 간의 어텐션 맵 겹침을 방지하여 최종 이미지의 충실도를 높입니다.



### Causal Deepsets for Off-policy Evaluation under Spatial or Spatio-temporal Interferences (https://arxiv.org/abs/2407.17910)
- **What's New**: 이 논문은 기존 Off-Policy Evaluation (OPE) 방법론에서 흔히 사용되는 Mean-Field 가정을 완화하는 새로운 Causal Deepset Framework를 제안한다. 특히 spatio-temporal interference (공간-시간 간섭) 을 처리할 때, 이러한 가정은 실제 환경에서 자주 충족되지 못하기 때문에, 기존 OPE 방법론의 효과성을 제한하는 요인이었다. 이 논문은 대신 Permutation Invariance (PI) 가정을 도입하여, 데이터 기반으로 Mean-Field 함수를 학습할 수 있게 하고, 기존 평균화 방법보다 유연한 추정 방식을 제공한다. 또한, PI 가정을 OPE에 적용하는 새로운 알고리즘을 제시하고, 이론적 기반을 자세히 살펴본다.



### 3D Hole Filling using Deep Learning Inpainting (https://arxiv.org/abs/2407.17896)
Comments:
          20 pages, 12 figures, to be submitted to Computers & Graphics Journal

- **What's New**: 본 논문은 3D 디지털화 기술에서 획득된 3D 표면의 빈 공간을 채우기 위한 새로운 방법론을 제시하며, 2D inpainting 기반 신경망을 활용하여 3D 표면을 효과적으로 재구성합니다. 특히, 100만 개 이상의 곡률 이미지 데이터셋으로 훈련된 맞춤형 신경망을 사용하며, 이러한 이미지는 정점의 곡률을 2D 평면 표현으로 보여줍니다. 또한, 정밀도를 높이고 표면 적응성을 보장하기 위해 표면 변형 기법을 사용합니다. 이는 시스템이 입력 데이터에서 패턴을 학습하고 일반화하여 정확하고 포괄적인 3D 표면을 생성할 수 있도록 합니다.



### An Iterative Approach to Topic Modelling (https://arxiv.org/abs/2407.17892)
- **What's New**: 이 논문은 기존의 topic modelling 방식의 한계를 극복하기 위해 반복적인 (iterative) topic modelling 방법을 제안합니다. 기존 방법들은 한 번의 실행 (one-shot)으로 topic을 생성하고 결과를 평가하는 데 어려움을 겪는 반면, 이 논문에서 제안하는 방법은 반복적인 과정을 통해 topic을 개선해 나가며 최적의 결과를 얻을 수 있다는 장점을 가지고 있습니다. 특히 BERTopic 패키지를 이용하여 COVIDSenti-A 데이터셋의 일부를 이용하여 반복적인 프로세스를 통해 최적화된 topic set을 도출하는 예시를 보여줍니다. 이러한 결과는 반복적인 topic modelling 접근 방식이 다른 topic modelling 알고리즘에도 적용될 수 있는 가능성을 보여줍니다.



### Unraveling the Never-Ending Story of Lifecycles and Vitalizing Processes (https://arxiv.org/abs/2407.17881)
- **What's New**: 이 논문은 기존 BPM (Business Process Management)  기법들이 특정 목표를 지향하는 프로세스 (teleological process) 분석에 초점을 맞춰왔기 때문에 엔티티의 수명 주기(lifecycle)를 다루는 프로세스를 분석하는 데 제한적이라는 점을 지적합니다.  이 논문은 엔티티의 수명 주기를 목표로 하는 "활성화 비즈니스 프로세스 (vitalizing business processes)" 라는 새로운 개념을 제시합니다.  활성화 프로세스는 엔티티의 상태를 유지하거나 개선하는 일련의 작업을 포함하며,  엔티티의 수명 주기와 밀접하게 연관되어 있습니다.  이 논문은 활성화 프로세스를 분석하기 위한 요구사항을 제시하고 수명 주기와 활성화 프로세스의 개념적 모델을 정의합니다.  



### HG-PIPE: Vision Transformer Acceleration with Hybrid-Grained Pipelin (https://arxiv.org/abs/2407.17879)
Comments:
          Accepted by ICCAD 2024

- **What's New**: 이 논문에서는 기존의 MCQ 생성 평가 지표들이 단어 유사성만을 고려하는 단점을 지적하고, 학습 목표에 대한 이해도를 측정하는 새로운 지표인 '지식 종속 가능성(KDA)'을 제안합니다. KDA는 학생들의 응답을 통해 측정 가능하며, 연구진은 이를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 지표를 개발했습니다. Human evaluation 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 유용성과 높은 상관관계를 보였으며, 기존의 단어 유사성 지표와 결합하여 MCQ 품질 평가에 대한 예측력을 높였습니다.



### Mew: Multiplexed Immunofluorescence Image Analysis through an Efficient Multiplex Network (https://arxiv.org/abs/2407.17857)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 기존의 MCQ 평가 메트릭의 한계를 극복하기 위해 **지식 종속 가능성 (Knowledge Dependent Answerability, KDA)** 이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하는 데 초점을 맞춥니다. 또한, KDA를 근사화하기 위해 **KDA_disc** 와 **KDA_cont** 라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.



### DragText: Rethinking Text Embedding in Point-based Image Editing (https://arxiv.org/abs/2407.17843)
Comments:
          22 pages, 18 figures

- **What's New**: 이 논문은 이미지 편집 과정에서 텍스트 임베딩의 역할을 심층적으로 분석하고, DragText라는 새로운 방법을 제안하여 기존 텍스트 기반 이미지 편집 방법의 효율성을 향상시킵니다. 특히, 이미지 편집 과정에서 텍스트 임베딩이 이미지 임베딩과의 불일치 문제로 인해 발생하는 드래그 중단 (drag halting) 현상을 해결하는 데 중점을 둡니다. 



### On the Opportunities of (Re)-Exploring Atmospheric Science by Foundation Models: A Case Study (https://arxiv.org/abs/2407.17842)
Comments:
          28 pages, 12 figures

- **What's New**: 이 논문은 GPT-4o라는 최첨단 멀티모달 기반 모델이 기후 과학에서 다양한 작업을 수행하는 방법을 탐구한다. 기후 데이터 처리, 물리적 진단, 예측 및 예측, 적응 및 완화의 네 가지 주요 카테고리로 나누어 GPT-4o의 성능을 평가한다.



### UMono: Physical Model Informed Hybrid CNN-Transformer Framework for Underwater Monocular Depth Estimation (https://arxiv.org/abs/2407.17838)
- **What's New**: 본 논문에서는 수중 환경의 특성을 고려하여 단일 이미지에서 수중 깊이를 추정하는 새로운 End-to-end 학습 프레임워크인 UMono를 제안합니다. UMono는 수중 이미지 형성 모델의 특성을 네트워크 구조에 통합하고, 수중 이미지의 지역적 특징과 전역적 특징을 효과적으로 활용합니다.



### NC-NCD: Novel Class Discovery for Node Classification (https://arxiv.org/abs/2407.17816)
Comments:
          Accepted by CIKM'24

- **What's New**: 이 연구는 기존의 Novel Class Discovery (NCD) 방법들이 새로운 카테고리를 발견하는 동안 이전에 학습한 카테고리의 성능을 유지하는 데 어려움을 겪는다는 문제를 해결하기 위해 새로운 NC-NCD (Node Classification Novel Class Discovery)  설정을 제안합니다. NC-NCD는 기존 카테고리 데이터 없이 새로운 카테고리의 레이블되지 않은 노드를 사용하여 모델을 계속 학습하여 모든 카테고리를 분류할 수 있도록 합니다. 이를 위해, 이 연구는 프로토타입 재생 및 지식 증류를 사용하는 자기 학습 프레임워크인 SWORD를 제안합니다. SWORD는 새로운 카테고리를 학습하고 기존 카테고리에 대한 성능을 유지하기 위해 자기 학습 전략을 사용합니다. 새로운 카테고리를 학습하면서 기존 카테고리에 대한 정보를 잊어버리는 것을 방지하기 위해 프로토타입과 지식 증류를 함께 사용합니다.



### Enhancing Model Performance: Another Approach to Vision-Language Instruction Tuning (https://arxiv.org/abs/2407.17813)
- **What's New**: 이 논문은 멀티모달 LLM 프레임워크를 위한 새로운 접근 방식인 Bottleneck Adapter를 소개합니다. 이 접근 방식은 이미지 인코더와 LLM을 연결하는 경량 어댑터를 사용하여 큰 신경망을 필요로 하지 않고 멀티모달 LLM 프레임워크 전체를 함께 최적화할 수 있도록 합니다. 기존의 모듈식 학습 방식과 달리 이 접근 방식은 끝단 최적화 체계를 채택하여 어댑터와 함께 훨씬 작은 매개변수 집합을 사용하여 전체 멀티모달 LLM을 함께 최적화할 수 있도록 합니다. 본 논문의 접근 방식은 90.12%의 정확도로 강력한 성능을 보여주며, 인간 수준의 성능(88.4%)과 LaVIN-7B(89.41%)를 능가합니다.



### EEG-SSM: Leveraging State-Space Model for Dementia Detection (https://arxiv.org/abs/2407.17801)
- **What's New**: 본 논문은 EEG-SSM이라는 새로운 state-space model 기반 접근 방식을 제안하여 EEG 데이터를 사용한 치매 분류를 수행합니다. EEG-SSM은 두 가지 혁신적인 구성 요소를 가지고 있습니다: EEG-SSM temporal과 EEG-SSM spectral입니다. temporal 요소는 다양한 길이의 EEG 시퀀스를 효율적으로 처리하도록 설계되었으며, spectral 요소는 EEG 신호의 주파수 영역 정보를 통합하여 모델을 향상시킵니다. 이러한 요소들의 시너지는 EEG-SSM이 다변수 EEG 데이터의 복잡성을 능숙하게 관리하여 다양한 시간 해상도에서 정확성과 안정성을 크게 향상시킬 수 있도록 합니다.



### A Unified Understanding of Adversarial Vulnerability Regarding Unimodal Models and Vision-Language Pre-training Models (https://arxiv.org/abs/2407.17797)
Comments:
          14 pages, 9 figures, published in ACMMM2024(oral)

- **What's New**: 이 논문은 VLP(Vision-Language Pre-training) 모델에 대한 새로운 적대적 공격 방법인 FGA(Feature Guidance Attack)를 제안합니다. FGA는 텍스트 표현을 사용하여 깨끗한 이미지의 왜곡을 유도하여 적대적 이미지를 생성합니다. 또한, FGA는 텍스트 공격을 통합하여 VLP 모델에 대한 공격 효과를 향상시키는 FGA-T(Feature Guidance with Text Attack)를 구축합니다.



### Very Large-Scale Multi-Agent Simulation in AgentScop (https://arxiv.org/abs/2407.17789)
Comments:
          We have released code on this https URL

- **What's New**: 본 논문에서는 대규모 시뮬레이션을 지원하기 위해 사용자 친화적인 멀티 에이전트 플랫폼인 AgentScope에 새로운 기능과 구성 요소를 추가하여 AgentScope의 편의성과 유연성을 향상시켰습니다. 구체적으로, 대규모 확장성과 높은 효율성을 위한 기반 기술 인프라로 액터 기반 분산 메커니즘을 제안하고 다양한 실제 시나리오를 시뮬레이션하기 위한 유연한 환경 지원을 제공하여 여러 에이전트의 병렬 실행, 중앙 집중식 워크플로우 오케스트레이션, 에이전트 간 상호 작용 및 에이전트-환경 상호 작용을 가능하게 합니다. 또한, AgentScope에 사용하기 쉬운 구성 가능한 도구와 자동 백그라운드 생성 파이프라인을 통합하여 다양하지만 세부적인 백그라운드 설정을 갖춘 에이전트를 만드는 프로세스를 간소화합니다. 마지막으로, 여러 장치에 배포될 수 있는 많은 수의 에이전트를 편리하게 모니터링하고 관리하기 위한 웹 기반 인터페이스를 제공합니다.



### HC-GST: Heterophily-aware Distribution Consistency based Graph Self-training (https://arxiv.org/abs/2407.17787)
Comments:
          accepted by CIKM 2024

- **What's New**: 본 논문에서는 Graph self-training (GST) 방법론이 Heterophilic graph에서 homophily ratio distribution shift를 일으켜 training bias를 발생시키는 문제를 제기하고, 이 문제를 해결하기 위한 novel framework인 HC-GST를 제안합니다. HC-GST는 soft label을 활용하여 homophily ratio를 추정하고, pseudo-node selection vector를 최적화하여 global homophily ratio distribution과 alignment를 맞추는 방식으로 training bias를 줄입니다. 또한, heterophilic node에 대해 multi-hop neighbor를 활용하여 pseudo-labeling accuracy를 높이고, dual-head GNN 모델을 통해 consistent하지 않은 고품질 node를 활용하여 feature extractor를 최적화합니다.



### How Lightweight Can A Vision Transformer B (https://arxiv.org/abs/2407.17783)
- **What's New**: 이 논문에서는 Vision Transformer를 간소화하는 Mixture-of-Experts (MoE) 전략을 제안합니다. 각 MoE 레이어의 전문가는 V와 W2를 공유하는 SwiGLU 피드포워드 네트워크입니다. 복잡한 어텐션이나 컨볼루션 메커니즘은 사용되지 않습니다. 깊이별 스케일링을 적용하여 히든 레이어의 크기를 점진적으로 줄이고 전문가 수는 단계별로 증가시킵니다. 그룹화된 쿼리 어텐션을 사용합니다. 소규모 데이터셋에 대한 사전 학습 유무와 관계없이 제안된 접근 방식을 연구하고 전이 학습이 이 규모에서 작동하는지 조사했습니다. 이 아키텍처는 0.67M 파라미터 크기에서도 경쟁력이 있음을 발견했습니다.



### Advancing Multi-Modal Sensing Through Expandable Modality Alignmen (https://arxiv.org/abs/2407.17777)
- **What's New**: 본 논문에서는 Babel framework을 소개하여, 부분적으로 쌍을 이루는 멀티 모달 데이터를 활용하는 방법을 제시합니다. Babel framework는 Wi-Fi, mmWave, IMU, LiDAR, 비디오, 깊이 등 6가지 감각 모달리티를 통합하는 확장 가능한 멀티 모달 감각 신경망입니다.  이 framework은 N-모달리티 정렬을 일련의 2-모달리티 정렬로 변환하여 완벽하게 쌍을 이룬 데이터의 부족 문제를 해결합니다. Babel은 또한 기존의 단일 모달 네트워크를 활용하는 사전 훈련된 모달 타워와 새로 통합된 모달리티의 기여도를 이전에 확립된 모달리티 정렬과 균형을 맞추는 적응형 훈련 전략을 포함합니다. 



### Mpox Detection Advanced: Rapid Epidemic Response Through Synthetic Data (https://arxiv.org/abs/2407.17762)
Comments:
          8 pages, 4 figures, 1 table

- **What's New**: 이 연구는 Mpox 병변을 감지하는 포괄적인 컴퓨터 비전 모델을 구축하기 위해 합성 데이터를 사용하는 새로운 접근 방식을 제시합니다. 이 연구는 합성 데이터를 활용하여 의료 응급 상황(예: 전염병, 생물 테러)에 신속하게 대응할 수 있는 질병 감지 모델 개발을 위한 새로운 방법을 제시합니다. 기존의 데이터 수집 방법은 이러한 상황에서 너무 느리기 때문에 최소한의 데이터로부터 신속하고 신뢰할 수 있는 모델을 생성하기 위한 혁신적인 방법이 필요합니다. SynthVision이라고 불리는 이 새로운 방법은 Fitzpatrick 척도(밝은 피부, 갈색 피부, 어두운 피부)에 따라 다양한 피부색의 신체 부위(얼굴, 등, 가슴, 다리, 목, 팔)에 있는 Mpox 병변을 나타내는 다양한 합성 이미지 세트를 생성하는 확산 모델을 사용합니다. 그런 다음 이 합성 데이터셋으로 비전 모델을 훈련 및 테스트하여 확산 모델이 고품질 훈련 데이터를 생성하는 효과와 비전 모델의 의료 이미지 인식 성능에 미치는 영향을 평가했습니다. 결과는 유망했습니다. 비전 모델은 Mpox 사례에 대해 96%의 정밀도와 재현율로 97%의 정확도를 달성했으며, 정상 및 기타 피부 질환 사례에 대해서도 마찬가지로 높은 지표를 보여주어 진짜 양성을 올바르게 식별하고 가짜 양성을 최소화할 수 있는 능력을 입증했습니다. 모델은 Mpox 사례에 대해 96%의 F1 점수를, 정상 및 기타 피부 질환에 대해서는 98%의 F1 점수를 달성하여 균형 잡힌 정밀도-재현율 관계를 반영하며, 따라서 예측의 신뢰성과 견고성을 보장합니다. 이 연구에서 제안된 SynthVision 방법론은 향후 의료 응급 상황에 대한 최소한의 데이터 입력으로 정확한 컴퓨터 비전 모델을 개발할 수 있는 잠재력을 보여줍니다.



### TwIPS: A Large Language Model Powered Texting Application to Simplify Conversational Nuances for Autistic Users (https://arxiv.org/abs/2407.17760)
- **What's New**: 자폐증 개인의 텍스트 기반 의사 소통을 위한 새로운 앱인 TwIPS가 소개되었으며, 이 앱은 대규모 언어 모델(LLM)을 활용하여 텍스트 메시지의 어조, 뉘앙스, 의도를 해석하고 사용자에게 피드백을 제공한다.



### Overcome the Difficulties of NSGA-II via Truthful Crowding Distance with Theoretical Guarantees (https://arxiv.org/abs/2407.17687)
- **What's New**: 이 논문은 NSGA-II의 다중 목표 최적화 문제에 대한 효율성을 향상시키기 위해 새로운 군집 거리(crowding distance) 계산 방법인 "진실된 군집 거리(truthful crowding distance)"를 제안합니다. 진실된 군집 거리는 각 목표 함수에 대한 해의 상대적인 위치를 더 정확하게 반영하여 다중 목표 최적화 문제에서 더 균등한 해 분포를 제공합니다.



### CRASAR-U-DROIDs: A Large Scale Benchmark Dataset for Building Alignment and Damage Assessment in Georectified sUAS Imagery (https://arxiv.org/abs/2407.17673)
Comments:
          16 Pages, 7 Figures, 6 Tables

- **What's New**: 이 논문은 CRASAR-U-DROIDs라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 소형 무인 항공기 (sUAS)에서 수집한 지형 공간 이미지 (geospatial imagery)를 사용하여 건물 피해 평가와 공간 정렬을 수행합니다. 이 데이터셋은 재난 대응에 sUAS의 사용이 증가하고, 고해상도 지형 공간 sUAS 이미지를 기반으로 한 머신러닝과 컴퓨터 비전 모델에서 이전 연구가 부족하다는 점을 고려하여 만들어졌습니다. 또한, sUAS와 위성 이미지 사이의 추가적인 연구를 가능하게 하고, 운영적인 사용 사례와의 일관성을 유지하기 위한 목표를 가지고 있습니다. CRASAR-U-DROIDs는 10개의 연방 재해 (허리케인 이안, 허리케인 아이다, 허리케인 하비, 허리케인 이달리아, 허리케인 로라, 허리케인 마이클, 머셋 베이 화재, 메이필드 토네이도, 킬라우에아 폭발, 샹플랭 타워 붕괴)에서 수집된 52개의 오쏘모자이크 (orthomosaic)로 구성되어 있으며, 67.98 평방 킬로미터 (26.245 평방 마일)를 포괄합니다. 또한, 이 데이터셋은 21,716개의 건물 다각형과 피해 라벨, 그리고 7,880개의 조정 주석을 포함하고 있습니다. 이미지는 타일링 처리되어 130명의 주석자에게 제공되었으며, 주석자들은 공동 피해 척도 (Joint Damage Scale)에 따라 건물 다각형의 피해에 대한 인간 판단을 제공했습니다. 이러한 주석은 건물 다각형 피해 라벨을 개별적으로 검토한 후, 위원회에서 다시 검토하는 2단계 검토 과정을 통해 검토되었습니다. 또한, 건물 다각형은 더욱 성능이 좋은 머신러닝 모델을 학습할 수 있도록 이미지와 정확히 겹치도록 공간적으로 정렬되었습니다. CRASAR-U-DROIDs는 sUAS 오쏘모자이크 이미지의 가장 큰 라벨링된 데이터셋인 것으로 보입니다. 



### Spiking Neural Networks in Vertical Federated Learning: Performance Trade-offs (https://arxiv.org/abs/2407.17672)
- **What's New**: 본 논문은 Vertical Federated Learning (VFL) 환경에서 Spiking Neural Networks (SNNs)를 사용한 새로운 방법을 제안하여 기존의 Artificial Neural Networks (ANNs)와 비교하여 에너지 효율성을 높이는 동시에 비슷한 정확도를 달성합니다. SNNs의 에너지 효율성은 edge computing과 같은 제한된 자원 환경에서 특히 유용합니다. 이 연구는 VFL에서 SNNs의 효용성을 보여주며 기존의 Federated Learning 연구를 확장합니다.



### SMA-Hyper: Spatiotemporal Multi-View Fusion Hypergraph Learning for Traffic Accident Prediction (https://arxiv.org/abs/2407.17642)
- **What's New**: 본 논문에서는 교사의 시간을 절약할 수 있는 MCQ 자동 생성의 새로운 평가 지표인 **지식 종속 가능성(KDA)**를 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표는 MCQ의 교육적 가치를 고려하지 않았지만, KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 이를 위해, 인간 설문 조사를 통해 KDA를 측정하고 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하여 KDA를 근사화합니다. 이 지표들은 **사전 훈련된 언어 모델**을 사용하여 학생의 문제 해결 행동을 모방합니다. 인간 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 설정에서의 사용성과 높은 상관관계를 보였습니다.



### CoMoTo: Unpaired Cross-Modal Lesion Distillation Improves Breast Lesion Detection in Tomosynthesis (https://arxiv.org/abs/2407.17620)
Comments:
          ADSMI @ MICCAI 2024

- **What's New**: 이 논문은 Digital Breast Tomosynthesis (DBT) 영상에서 병변 탐지 정확도를 향상시키기 위한 새로운 프레임워크인 CoMoTo를 제안합니다. 기존 맘모그래피 데이터를 활용하여 DBT 모델의 학습을 향상시키는 방법을 제시합니다. 특히, Lesion-specific Knowledge Distillation (LsKD)과 Intra-modal Point Alignment (ImPA)라는 두 가지 새로운 구성 요소를 제안합니다.



### Quality Assured: Rethinking Annotation Strategies in Imaging AI (https://arxiv.org/abs/2407.17596)
Comments:
          Accepted at ECCV 2024, preprint, Computer Vision, Data Annotation

- **What's New**: 이 논문은 AI 기반 이미지 분석을 위한 신뢰할 수 있는 벤치마킹과 실제 응용을 위한 필수적인 기반인 고품질 참조 주석을 생성하는 문제를 연구했습니다. 이전 연구는 주석을 아웃소싱하는 수단으로 크라우드소싱에 초점을 맞추었지만, 주석 회사의 내부 품질 보증(QA) 프로세스에 대해서는 거의 주목하지 않았습니다. 따라서 이 연구는 주석 회사가 사용하는 QA가 주석 품질에 미치는 영향을 평가하고 데이터 주석 효율성을 극대화하기 위한 방법론을 고안하는 데 목표를 두었습니다. 연구팀은 4개의 주석 회사와 Amazon Mechanical Turk(MTurk)에서 924명의 주석자와 34명의 QA 작업자로부터 얻은 총 57,648개의 인스턴스 분할 이미지를 기반으로 다음과 같은 통찰력을 얻었습니다. (1) 주석 회사는 널리 사용되는 플랫폼인 MTurk에 비해 수량과 품질 측면에서 모두 더 나은 성능을 보입니다. (2) 주석 회사의 내부 QA는 미미하거나 전혀 개선을 제공하지 않습니다. 그러나 QA에 투자하는 대신 라벨링 지침을 개선하면 주석 성능을 크게 향상시킬 수 있습니다. (3) 내부 QA의 이점은 특정 이미지 특성에 따라 다릅니다. 이 연구는 연구자들이 고정된 주석 예산에서 훨씬 더 많은 가치를 얻고 주석 회사가 내부 QA를 수행하는 방식을 바꿀 수 있도록 합니다.



### Unified Prediction Model for Employability in Indian Higher Education System (https://arxiv.org/abs/2407.17591)
Comments:
          9 pages

- **What's New**: 본 논문에서는 인도 전역의 17개 주에서 수집한 공학/기술 학사 학위와 컴퓨터 응용 학사 학위 학생들의 데이터를 사용하여 학생 고용 가능성을 예측하는 통합 예측 모델을 개발했습니다. 이 모델은 다양한 문화적 배경과 교육 과정 구조를 가진 인도 전역의 다양한 주와 기관에 적용될 수 있습니다. 또한, 본 연구에서는 인도 교육 시스템에서 학생 고용 가능성 예측과 관련하여 주별로 유의미한 차이가 없다는 것을 통계적으로 입증했습니다.



### Is computational creativity flourishing on the dead internet? (https://arxiv.org/abs/2407.17590)
Comments:
          6 pages

- **What's New**: 이 논문은 소셜 미디어에서 인기를 얻고 있는 '죽은 인터넷 이론'(Dead Internet Theory)을 탐구하며, 특히 사람처럼 행동하는 인공지능 기반 봇(AI influencer)의 행동을 '계산적 창의성'(Computational Creativity)의 관점에서 분석합니다. 이러한 봇들은 실제 사람처럼 보이지만, 콘텐츠를 대량으로 생성하고 소셜 미디어에서의 참여(Engagement)를 극대화하도록 설계되었습니다.



### Quelle {\'e}thique pour quelle IA ? (https://arxiv.org/abs/2407.17585)
Comments:
          in French language. Workshop Ethique et Morale de la Chaire IA Responsable, Nathalie Nevejans, May 2021, Distanciel, France

- **What's New**: 본 논문에서는 인공지능 윤리에 관련된 다양한 윤리적 접근 방식을 분석하고, 이들의 관심사와 한계를 규명합니다. 저자는 현대 사회에서 윤리의 필요성과 의미를 소개하고, 다른 규범적 레지스터와 구분하여 형식화의 부적절성을 강조합니다. 이후, 저자는 윤리 철학의 범위를 포괄하는 윤리 이론의 지도를 제시하며, 메타윤리, 규범 윤리, 응용 윤리를 명확히 구분합니다. 이러한 개요를 통해 저자는 윤리와 인공지능의 관계를 질문합니다. 분석은 특히 서구 민주주의에서 디지털 윤리와 인공지능의 수행 방식에 영향을 미친 주요 윤리적 흐름에 초점을 맞춥니다. 저자는 오늘날 특정 패턴으로 결정되는 것처럼 보이는 윤리적 관행이 인공지능 윤리에 대한 우리의 필요를 충족할 만큼 충분하고 만족스러운 답변인지 질문합니다. 본 연구는 인공지능의 인간 윤리가 맥락 윤리의 실용적인 실천에 기반해야 하는 이유, 즉 인간에게 제기되는 윤리적 문제에 대한 어떠한 형식화나 자동화된 처리에도 불구하고, 맥락 윤리는 필수적이고 축소될 수 없다는 것을 반영하며 결론을 내립니다.



### CityX: Controllable Procedural Content Generation for Unbounded 3D Cities (https://arxiv.org/abs/2407.17572)
Comments:
          5 figures

- **What's New**: 이 논문은 여러 레이아웃 조건(OSM, 의미론적 지도, 위성 이미지 등)을 기반으로 현실적인, 무한한 3D 도시 생성을 가능하게 하는 새로운 멀티모달 제어 가능한 프로시저럴 콘텐츠 생성 방식인 CityX를 제안합니다. CityX는 다양한 PCG 플러그인을 통합하는 일반적인 프로토콜과 명령어를 실행 가능한 Blender 액션으로 변환하는 멀티 에이전트 프레임워크를 제공합니다. CityX는 생성된 자산의 품질과 산업 요구 사항 간의 차이를 줄임으로써 3D 장면 생성을 위한 혁신적인 생태계를 구축할 수 있는 가능성을 보여줍니다.



### MathViz-E: A Case-study in Domain-Specialized Tool-Using Agents (https://arxiv.org/abs/2407.17544)
- **What's New**: 본 논문에서는 교육 분야에서 자동화된 수학 시각화 시스템인 MathViz-E를 제안합니다. 이 시스템은 학생들이 수학적 개념을 시각화하고 상호 작용할 수 있도록 돕는 핵심 도구인 그래프 생성을 자동화합니다. 음성 명령을 받아 수학적 표현으로 변환하고 Desmos 그래프 계산기를 사용하여 시각화합니다. 이를 통해 교사는 수업 흐름을 방해하지 않고 수업에 수학 시각화 기술을 더 쉽게 통합할 수 있습니다.



### Dataset Distribution Impacts Model Fairness: Single vs. Multi-Task Learning (https://arxiv.org/abs/2407.17543)
Comments:
          Submitted to MICCAI 2024

- **What's New**: 이 연구는 피부 병변 분류에서 ResNet 기반 CNN을 사용하여 성별 차이가 있는 훈련 데이터 세트를 사용한 모델 성능을 평가하고, 세 가지 학습 전략의 효과를 조사했습니다. 특히 선형 프로그래밍 방법을 사용하여 성별과 클래스 레이블을 조절한 데이터 세트를 생성하여 모델의 공정성 (fairness)을 연구했습니다.



### SFPrompt: Communication-Efficient Split Federated Fine-Tuning for Large Pre-Trained Models over Resource-Limited Devices (https://arxiv.org/abs/2407.17533)
- **What's New**: 본 논문은 개인정보 보호 문제로 인해 다운스트림 데이터에 접근할 수 없는 상황에서도 대규모 사전 훈련 모델의 기능을 효과적으로 활용할 수 있도록, 분산 환경에서의 프롬프트 학습을 위한 새로운 프라이버시 보호 미세 조정 방법인 SFPrompt를 제안합니다. SFPrompt는 분할 학습(split learning)과 연합 학습(federated learning)을 적절히 결합하여 이러한 과제를 해결합니다.



### StreamTinyNet: video streaming analysis with spatial-temporal TinyML (https://arxiv.org/abs/2407.17524)
Comments:
          this paper has been accepted and presented at the WCCI24 conference

- **What's New**: 이 논문은 TinyML(Tiny Machine Learning) 장치에서 다중 프레임 비디오 스트리밍 분석(VSA, Video Streaming Analysis)을 가능하게 하는 최초의 아키텍처인 StreamTinyNet을 제안합니다. 기존의 TinyML 솔루션들은 제한된 메모리와 연산 능력으로 인해 프레임 단위로 분석을 수행했지만, StreamTinyNet은 여러 프레임을 동시에 분석하여 공간-시간 패턴을 파악할 수 있습니다.

- **Technical Details**: StreamTinyNet은 TinyML 장치에 적합하도록 기존 CNN 아키텍처를 개선하여 메모리 사용량과 연산량을 줄였습니다. 특히, 이 연구에서는 TinyML 장치의 제한된 리소스를 고려하여 메모리 사용량과 연산량을 최소화하는 동시에 정확도를 높이는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, StreamTinyNet은 공개 데이터셋에서 기존 솔루션보다 우수한 성능을 보였으며, 실제 TinyML 장치인 Arduino Nicla Vision에 적용하여 효율성을 입증했습니다. 이는 제스처 인식과 같은 공간-시간 분석이 필요한 다양한 TinyML 애플리케이션에 활용될 수 있는 잠재력을 보여줍니다.



### Quality Diversity for Robot Learning: Limitations and Future Directions (https://arxiv.org/abs/2407.17515)
Comments:
          Accepted to GECCO 2024

- **What's New**: 본 논문에서는 기존의 Quality Diversity (QD) 방법론이 가진 한계를 지적하고,  새로운 접근 방식을 제시한다. 기존 QD 방법은 목표 지점까지 이동하는 다양한 에이전트를 학습하는 데 초점을 맞췄지만, 본 논문에서는 단일 목표 조건 정책(goal-conditioned policy)과 고전적 플래너(classical planner)를 결합하여 다양한 목표 지점으로의 이동을 효율적으로 달성할 수 있는 방법을 제안한다. 이 방법은 O(1) 공간 복잡도를 가지며, 새로운 환경에 대한 일반화(generalization) 성능도 뛰어나다.



### Artificial Intelligence Based Navigation in Quasi Structured Environmen (https://arxiv.org/abs/2407.17508)
Comments:
          10 pages, 8 figures

- **What's New**: This paper proposes a novel modified Floyd-Warshall with ACO algorithm for transportation route planning. The proposed algorithm demonstrated better results with less time complexity compared to the general Floyd-Warshall algorithm when applied on quasi-structured points.



### AI in Remote Patient Monitoring (https://arxiv.org/abs/2407.17494)
Comments:
          A chapter for an upcoming Springer book titled "Transformation in Health Care"

- **What's New**: 본 논문에서는 MCQ 생성을 위한 새로운 자동 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 대상 사실에 대한 지식 평가 능력을 평가합니다. KDA는 학생 설문 조사에서 얻은 반응을 기반으로 측정됩니다. 또한, KDA를 근사화하기 위해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, n-gram 기반 유사성 지표와 결합하면 KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에 대한 강력한 예측력을 보여줍니다.



### ReDiFine: Reusable Diffusion Finetuning for Mitigating Degradation in the Chain of Diffusion (https://arxiv.org/abs/2407.17493)
Comments:
          27 page

- **What's New**: 본 논문에서는 pretrained text-to-image diffusion models을 이용하여 여러 번 fine-tuning을 반복하는 ‘Chain of Diffusion’에서 발생하는 model collapse 문제를 해결하기 위한 새로운 방법인 ‘Reusable Diffusion Finetuning (ReDiFine)’을 제안합니다. ReDiFine은 condition drop finetuning과 CFG scheduling을 결합하여 여러 반복을 거쳐도 이미지 품질이 저하되지 않도록 합니다.



### Unraveling Molecular Structure: A Multimodal Spectroscopic Dataset for Chemistry (https://arxiv.org/abs/2407.17492)
Comments:
          14 pages, submited to conference, code available at: this https URL

- **What's New**: 본 논문에서는 다양한 분광학 기술에서 얻은 스펙트럼 데이터를 포함하는 다중 모달 데이터셋을 소개합니다. 이 데이터셋은 기존의 단일 모달 기반 접근 방식의 한계를 극복하고, 여러 분광학 모달에서 정보를 통합할 수 있는 기반 모델을 개발하는데 사용될 수 있습니다. 특히, 79만 개의 분자에 대한 IR, 1H-NMR, 13C-NMR, HSQC-NMR, 양이온 모드 MS/MS, 음이온 모드 MS/MS 스펙트럼 데이터를 제공하며, 분광학 기반 구조 규명을 위한 벤치마크 모델도 함께 제공합니다. 데이터셋과 벤치마크 코드는  [link] 에서 확인할 수 있습니다.



### AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents (https://arxiv.org/abs/2407.17490)
- **What's New**: 자동 MCQ 생성 평가를 위한 새로운 지식 종속 가능성 (KDA) 메트릭을 제안합니다. KDA는 MCQ가 특정 사실에 대한 학생의 지식을 평가할 수 있는지 측정합니다.  KDA는 학생 응답을 기반으로 측정되고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 방식을 모방하여 KDA를 추정합니다.



### Collective Attention in Human-AI Teams (https://arxiv.org/abs/2407.17489)
- **What's New**: 이 연구는 기존의 MCQ 생성 평가 지표가 교육적 가치를 무시하고 골드 샘플과의 유사성에만 초점을 맞춘다는 문제점을 지적하며, 새로운 지식 의존 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하는 메트릭입니다. 또한, KDA를 자동으로 계산하기 위해 KDA_disc와 KDA_cont라는 두 가지 새로운 메트릭을 제안합니다. 이러한 메트릭은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 추정합니다. 연구팀은 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 환경에서의 사용성 간에 강한 상관관계를 가지고 있음을 확인했습니다.



### A Survey of Accessible Explainable Artificial Intelligence Research (https://arxiv.org/abs/2407.17484)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문에서는 시각 장애인을 포함한 모든 사용자를 위한 AI 기반 의사결정 설명의 접근성에 대한 체계적인 문헌 분석을 제시합니다. 이 연구는 접근 가능한 설명이 디지털 포용을 촉진하고 모든 사람이 신체적, 감각적 또는 인지적 능력에 관계없이 기술을 효과적으로 사용할 수 있도록 하기 때문에 매우 중요합니다.  



### Universal Approximation Theory: The basic theory for deep learning-based computer vision models (https://arxiv.org/abs/2407.17480)
- **What's New**: 이 논문은 컴퓨터 비전 분야에서 널리 사용되는 Convolutional Neural Network (CNN) 및 Transformer 모델을 Universal Approximation Theorem (UAT)의 틀 안에서 통합하여 설명하는 새로운 접근 방식을 제시합니다. UAT를 사용하여 CNN과 Transformer가 이미지 처리에서 어떻게 작동하는지에 대한 기본적인 질문에 대한 이론적 기초를 제공합니다.



### ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems (https://arxiv.org/abs/2407.17476)
- **What's New**: 본 논문에서는 기존의 Cognitive Diagnosis Model (CDM)들이 학생들의 지식 수준을 너무 유사하게 학습하는 "oversmoothing" 문제를 해결하기 위해 새로운 Over-smoothing Resistant Cognitive Diagnosis Framework (ORCDF)를 제안합니다. ORCDF는 학습 과정에서 응답 신호를 활용하여 기존의 CDM을 개선하며, 특히 응답 그래프 (response graph)와 Response-Aware Graph Convolution Network (RGC)를 도입하여 학생들의 응답 패턴을 효과적으로 분석합니다. 또한, 추측과 실수 (guess and slip) 문제를 줄이기 위해 학습 단계에서 응답 그래프의 엣지를 뒤집는 (flip) 방법을 적용합니다.



### An Approach to Detect Abnormal Submissions for CodeWorkout Datas (https://arxiv.org/abs/2407.17475)
- **What's New**: 이 논문은 프로그래밍 학습 환경에서 학생들의 로그 데이터를 분석하여 부정행위를 탐지하는 새로운 방법을 제안합니다. 기존의 부정행위 탐지 방법들은 코드 표절만 탐지할 수 있었지만, 이 논문에서는 여러 번의 유사한 솔루션을 제출하는 등의 다른 이상 행위를 탐지하는 방법을 제시합니다. 이 논문은 특히 CWO (CodeWorkout) 프로그래밍 데이터셋을 사용하여 분석을 수행하였으며,  MOSS (Measure of Software Similarity)와 같은 기존 방법을 비교하여 분석했습니다.



### "My Kind of Woman": Analysing Gender Stereotypes in AI through The Averageness Theory and EU Law (https://arxiv.org/abs/2407.17474)
Comments:
          presented at IAIL 2024 the Imagining the AI Landscape After the AI ACT, in conjunction with HHAI2024, Malmö, Sweden, June 10, 2024

- **What's New**: 본 연구는 성별 분류 시스템에서 사회적 고정관념과 알고리즘 결정 간의 상호 작용을 탐구하여 인간 편견이 인공 지능(AI) 시스템에 전파될 가능성을 살펴봅니다. 인간의 얼굴 매력도와 성별 식별 능력 간의 관계를 시사하는 '평균성 이론'을 바탕으로, AI 모델 Stable Diffusion 2.1을 사용하여 다양한 매력도 수준을 포함하는 데이터셋을 생성하여 인간 인지에서 관찰되는 매력도와 성별 분류 정확도 간의 상관관계가 AI에서도 지속되는지 여부를 조사했습니다. 연구 결과, 인간과 유사하게 AI 시스템은 매력도에 따라 성별 분류 정확도에 차이를 보이며, 알고리즘 결정에 사회적 편견과 고정관념을 반영하는 것으로 나타났습니다. 이러한 발견은 AI 개발 및 AI 데이터 훈련에 다학제적이고 교차적인 접근 방식의 필요성을 강조하며, AI 훈련에 사용되는 데이터가 AI 법과 GDPR의 범위 내에서 성별 다양성과 공정성을 촉진할 수 있는 방법을 살펴봅니다.



### Automated Explanation Selection for Scientific Discovery (https://arxiv.org/abs/2407.17454)
Comments:
          Composite AI Workshop at ECAI 2024 (accepted for publication)

- **What's New**: 본 논문은 자동 MCQ 생성의 교육적 가치를 평가하기 위해 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가할 수 있는 능력을 측정합니다. 이 지표는 인간 학생의 응답을 기반으로 하며, KDA_disc와 KDA_cont라는 두 가지 자동화된 지표로 구현되었습니다. 이러한 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 인간 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 수업 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 지표와 결합하면 다양한 전문가 평가 MCQ 품질 측정에 대한 예측력이 강력하게 나타났습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하기 위해 인간 학생의 응답을 사용하여 측정됩니다. 이를 자동화하기 위해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont가 개발되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 수업 환경에서의 사용성과 강한 상관관계를 보였습니다. 또한, n-gram 기반 유사성 지표와 결합하면 다양한 전문가 평가 MCQ 품질 측정에 대한 예측력이 강력하게 나타났습니다.



### Grammar-based Game Description Generation using Large Language Models (https://arxiv.org/abs/2407.17404)
- **What's New**: 본 논문은 제한적인 데이터를 가진 자동 게임 디자인 문제를 해결하기 위해 대규모 언어 모델(LLM)의 컨텍스트 학습(in-context learning)을 활용하는 새로운 방법을 제시한다. LLM은 몇 가지 예시를 통해 작업의 특징을 파악하고 사전 훈련 과정에서 습득한 능력을 적용할 수 있다. 특히, 게임 디자인 공간을 효과적으로 구조화하는 게임 설명의 문법(grammar)을 LLM의 추론 과정에 도입하여 LLM이 게임 설명 생성이라는 복잡한 작업의 특성을 파악하도록 돕는다. 또한, 문법을 활용하여 생성된 출력을 반복적으로 개선하는 디코딩 방법을 제안한다. 이러한 접근 방식은 게임 설명 생성에서 뛰어난 성능을 보여준다.



### Systematic Reasoning About Relational Domains With Graph Neural Networks (https://arxiv.org/abs/2407.17396)
Comments:
          10+16 pages, 2+7 figures, 4+9 tables. Preprint under review. Comments welcome

- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 생성된 MCQ의 대답 가능성을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA를 측정하기 위해, 학생 설문 조사를 통해 얻은 응답을 활용하는 방법과 사전 훈련된 언어 모델을 활용하여 KDA를 자동으로 근사하는 두 가지 방법 (KDA_disc와 KDA_cont)을 제안합니다.



### Testing Large Language Models on Driving Theory Knowledge and Skills for Connected Autonomous Vehicles (https://arxiv.org/abs/2407.17211)
- **What's New**: 본 논문에서는 자율 주행 시스템에 LLMs (Large Language Models)를 적용하는 새로운 방법을 제시하고, LLMs의 주행 이론 및 기술 이해를 평가하여 CAV (Connected and Automated Vehicle)에 안전에 중요한 역할을 할 수 있는지 확인합니다. 특히, LLM이 주행 이론과 기술에 대한 이해가 있는지 평가하기 위한 주행 이론 시험을 설계하고 실시했습니다.



### Toward an Integrated Decision Making Framework for Optimized Stroke Diagnosis with DSA and Treatment under Uncertainty (https://arxiv.org/abs/2407.16962)
- **What's New**: 본 논문에서는 불확실성 속에서 뇌졸중 진단 및 치료의 어려움을 해결하기 위한 새로운 접근 방식을 제시합니다. 이는 동맥류(aneurysm), 뇌동맥 기형(AVM), 혈관 폐쇄(occlusion)와 같은 뇌졸중 조건의 빠른 진행과 심각한 결과를 고려할 때 매우 중요한 문제입니다. 디지털 혈관 조영술(DSA)과 같은 기존 진단 방법은 비용이 많이 들고 침습적이라는 단점이 있습니다. 이러한 문제를 해결하기 위해, 부분 관측 마르코프 결정 프로세스(POMDP) 프레임워크를 사용하는 새로운 접근 방식을 제안합니다. 본 모델은 고급 진단 도구와 치료 접근 방식을 뇌졸중 진단의 고유한 불확실성을 고려하는 의사 결정 알고리즘과 통합합니다. 본 접근 방식은 CT 스캔, Siriraj 점수, DSA 보고서의 노이즈가 많은 관찰 결과를 결합하여 후속 치료 옵션을 알립니다. 트리 검색 방법과 입자 필터를 사용하는 온라인 솔버 DESPOT을 활용하여 잠재적인 미래 시나리오를 시뮬레이션하고 전략을 안내합니다. 결과는 POMDP 프레임워크가 진단 및 치료 목표 간의 균형을 맞추고, DSA와 같은 침습적 절차를 통한 정확한 뇌졸중 식별의 필요성과 병원 내 또는 가정 관찰과 같은 비용 효율적인 전략의 제약(의료 자원 제한) 간의 절충을 이룬다는 것을 나타냅니다. 본 연구는 뇌졸중에 대한 진단 및 치료 프로세스를 최적으로 통합하고 다양한 불확실성을 고려하여 뇌졸중 관리의 치료 및 결과를 개선하는 체계적인 프레임워크를 제시함으로써 중요한 기여를 합니다.



### Networks of Networks: Complexity Class Principles Applied to Compound AI Systems Design (https://arxiv.org/abs/2407.16831)
- **What's New**: 본 논문에서는 여러 개의 언어 모델 추론 호출로 구성된 복합 인공지능 시스템(Compound AI Systems)이 증가하는 상황에서, 제안된 답변을 생성하고 정확성을 검증하는 기능을 분리하여 구성된 ‘네트워크의 네트워크(Networks of Networks, NoNs)’라는 시스템을 소개합니다. 특히 복잡성 이론(Complexity Theory)에서 언급되는 생성 복잡도(Generation Complexity)와 검증 복잡도(Verification Complexity) 개념을 언어 모델(Language Models, LMs)에도 적용하여 실험적으로 검증합니다. NoNs는 K개의 생성기(Generator)와 검증기(Verifier)로 구성되며, 생성기는 답변과 이유를 생성하고, 검증기는 생성된 답변과 이유를 검증하는 방식으로 작동합니다.



### Infinite Ends from Finite Samples: Open-Ended Goal Inference as Top-Down Bayesian Filtering of Bottom-Up Proposals (https://arxiv.org/abs/2407.16770)
Comments:
          Accepted for publication at CogSci 2024. 6 pages, 4 figures. (Appendix: 5 pages, 6 figures, 2 tables)

- **What's New**: 이 논문은 MCQ 자동 생성 평가 메트릭의 한계점을 지적하고, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 메트릭을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 또한, KDA를 자동화하기 위한 두 가지 방법 (KDA_disc, KDA_cont)을 제시하며, 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여줍니다.



### SoNIC: Safe Social Navigation with Adaptive Conformal Inference and Constrained Reinforcement Learning (https://arxiv.org/abs/2407.17460)
Comments:
          Project website: this https URL

- **What's New**: 본 논문에서는 MCQ 생성의 교육적 가치를 평가하는 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 학생의 문제 해결 능력을 모방한 사전 훈련된 언어 모델을 사용하여 KDA를 근사화합니다. 인간 연구를 통해 KDA_disc와 KDA_soft가 KDA와 전문가가 평가한 실제 강의실 설정에서의 사용성과 강한 상관관계를 보였다는 것을 보여줍니다. 또한, n-gram 기반 유사성 메트릭과 결합하면 KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 측정에 대한 강력한 예측력을 보이는 것으로 나타났습니다.



### HumanVid: Demystifying Training Data for Camera-controllable Human Image Animation (https://arxiv.org/abs/2407.17438)
Comments:
          camera controllable human image animation, a dataset and a baseline

- **What's New**: 이 논문은 자동으로 MCQ를 생성할 때 기존 평가 지표가 교육적 가치를 고려하지 않고 있다는 문제점을 제기하며, 지식 종속 가능성(KDA)이라는 새로운 평가 지표를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지를 측정합니다. 논문은 또한 KDA를 근사하기 위해 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다.  Human evaluation을 통해 KDA_disc와 KDA_soft가 실제 강의실에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.  KDA_disc와 KDA_cont는 n-gram 기반 유사성 지표와 결합하여 다양한 전문가가 평가한 MCQ 품질 지표를 예측하는 데 강력한 능력을 보여주었습니다. (This paper addresses the problem of existing evaluation metrics for automatically generated MCQs not considering educational value, proposing a novel evaluation metric called Knowledge Dependent Answerability (KDA). KDA measures how well an MCQ assesses students' knowledge of the target fact. The paper also proposes two automatic evaluation metrics, KDA_disc and KDA_cont, which leverage pre-trained language models to mimic students' problem-solving behavior to approximate KDA. Human evaluations demonstrate a strong correlation between KDA_disc and KDA_soft and usability in real classroom settings. When combined with n-gram based similarity metrics, KDA_disc and KDA_cont exhibit strong predictive power for various expert-labeled MCQ quality measures.)



### AIR-Bench 2024: A Safety Benchmark Based on Risk Categories from Regulations and Policies (https://arxiv.org/abs/2407.17436)
- **What's New**: 이 논문은 기존 MCQ 생성 평가 지표들이 교육적 가치를 제대로 반영하지 못한다는 점을 지적하며,  MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정하는 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 실제 학생들의 응답을 기반으로 측정되며, 논문에서는 KDA를 추정할 수 있는 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제시합니다.



### How Do Students Interact with an LLM-powered Virtual Teaching Assistant in Different Educational Settings? (https://arxiv.org/abs/2407.17429)
Comments:
          Accepted in the Seventeenth International Conference on Educational Data Mining (EDM) Workshop: Leveraging LLMs for Next Generation Educational Technologies, July 2024

- **What's New**: 이 논문에서는 교육용 챗봇인 Jill Watson을 이용하여 학생들의 질문 유형과 복잡성을 분석한 결과를 제시합니다. 특히, Jill은 다양한 수준의 인지 능력을 요구하는 질문에 대응하며 학생들이 고차원적인 사고를 요구하는 질문을 하도록 장려하는 효과를 보여줍니다.  Bloom's Revised Taxonomy를 사용하여 분석을 진행했습니다.  



### Vision Language Model-Empowered Contract Theory for AIGC Task Allocation in Teleoperation (https://arxiv.org/abs/2407.17428)
Comments:
          11 pages, 10 figures

- **What's New**: 이 논문은 야간 원격 조작 (teleoperation) 에서의 저조도 이미지 향상 기술을 개선하기 위해 AI 생성 콘텐츠 (AIGC) 모델을 활용하는 방법을 제안합니다. 특히, AIGC 모델은 계산량이 많기 때문에 풍부한 계산 자원을 가진 에지 서버에 AIGC 작업을 할당해야 합니다. 다양한 크기의 데이터셋으로 훈련된 AIGC 모델의 비용과 AIGC 작업의 서로 다른 수요를 고려하여, 원격 조작자와 에지 서버의 유틸리티를 동시에 최적화하는 차등 가격 책정 전략을 수립하는 것이 중요합니다. 하지만, 가격 책정 전략 수립은 정보 비대칭 (information asymmetry) 아래에서 이루어지며, 즉 AIGC 작업의 수요 (예: AIGC 작업의 난이도 수준 및 분포) 는 에지 서버에 대한 숨겨진 정보입니다. 또한, AIGC 작업의 난이도 수준을 수동으로 평가하는 것은 원격 조작자에게 지루하고 불필요합니다. 이를 위해, 우리는 Vision Language Model (VLM) 을 활용한 계약 이론 (contract theory) 에 의해 지원되는 AIGC 작업 할당 프레임워크를 고안했으며, 이 프레임워크에는 VLM 지원 난이도 평가와 계약 이론 지원 AIGC 작업 할당의 두 가지 구성 요소가 포함됩니다. 첫 번째 구성 요소는 자동적이고 정확한 AIGC 작업 난이도 평가를 가능하게 합니다. 두 번째 구성 요소는 정보 비대칭 하에서 에지 서버에 대한 가격 책정 전략을 수립할 수 있으며, 이를 통해 에지 서버와 원격 조작자 모두의 유틸리티를 최적화합니다. 시뮬레이션 결과에 따르면, 제안된 프레임워크는 원격 조작자와 에지 서버의 평균 유틸리티를 각각 10.88~12.43% 와 1.4~2.17% 개선할 수 있습니다. 코드와 데이터는 이 URL 에서 확인할 수 있습니다.



### $A^*$ for Graphs of Convex Sets (https://arxiv.org/abs/2407.17413)
- **What's New**: 이 논문은 기존의 convex-programming 기반 접근 방식에 휴리스틱 정보를 결합하여 그래프의 볼록 집합에서 최단 경로 문제(SPP-GCS)의 최적성 보장(optimality guarantees)과 거의 최적의 경로(near-optimal paths)를 찾는 새로운 알고리즘을 제시합니다. 이 방법은 A*에서 영감을 받아 정해진 정점의 하위 집합에서 최상위 탐색(best-first-like) 프로세스를 시작하고 추가적인 성장이 가능하거나 유익하지 않을 때까지 반복적으로 확장합니다.



### (PASS) Visual Prompt Locates Good Structure Sparsity through a Recurrent HyperNetwork (https://arxiv.org/abs/2407.17412)
Comments:
          Under review

- **What's New**: 본 논문은 **지식 종속 가능성 (Knowledge Dependent Answerability, KDA)** 라는 새로운 자동 평가 지표를 제안합니다. KDA는 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생 응답을 기반으로 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하여 KDA를 근사합니다. 이 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.



### Co-designing an AI Impact Assessment Report Template with AI Practitioners and AI Compliance Experts (https://arxiv.org/abs/2407.17374)
Comments:
          11 pages, 7 figures

- **What's New**: 본 논문은 AI 시스템의 실제 사용에 대한 영향 평가 보고서 템플릿을 제안합니다. 이 템플릿은 EU AI Act, NIST의 AI Risk Management Framework, ISO 42001 AI Management System을 기반으로 합니다.



### MuST: Multi-Scale Transformers for Surgical Phase Recognition (https://arxiv.org/abs/2407.17361)
- **What's New**: 본 논문은 수술 단계 인식(phase recognition)을 위한 새로운 Transformer 기반 접근 방식인 MuST(Multi-Scale Transformers for Surgical Phase Recognition)를 제안합니다. MuST는 Multi-Term Frame Encoder와 Temporal Consistency Module을 결합하여 수술 영상의 다양한 시간 규모(temporal scales) 정보를 포착합니다. Multi-Term Frame Encoder는 관심 프레임 주변에서 다양한 stride로 시퀀스를 샘플링하여 시간 규모 계층(hierarchy)에서 상호 의존성(interdependencies)을 계산합니다. 또한, 프레임 임베딩에 대한 장기 Transformer 인코더를 사용하여 장기 추론(long-term reasoning)을 강화합니다.  



### Preliminary study on artificial intelligence methods for cybersecurity threat detection in computer networks based on raw data packets (https://arxiv.org/abs/2407.17339)
Comments:
          Submitted to Computer Science Journal

- **What's New**: 본 논문은 네트워크 트래픽 내에서 원시 패킷 데이터로부터 실시간으로 공격을 감지할 수 있는 딥러닝 기법을 제안합니다. 이는 기존의 트래픽 흐름 특징을 기반으로 하는 방법과 달리 원시 패킷 데이터로부터 직접 특징과 패턴을 추출하는 딥러닝 알고리즘의 잠재력을 활용합니다. 또한, 추가 소프트웨어 구성 요소에 대한 의존성을 제거하고, 실시간 모니터링을 가능하게 합니다. 



### Enhanced Deep Learning Methodologies and MRI Selection Techniques for Dementia Diagnosis in the Elderly Population (https://arxiv.org/abs/2407.17324)
- **What's New**: 이 논문은 MRI 슬라이스를 선택적으로 처리하여 뇌의 가장 중요한 영역에 집중하고 정보가 적은 부분을 제외하는 새로운 방법을 소개합니다. 이 방법은 세 가지 맞춤형 딥 러닝 모델(Dem3D ResNet, Dem3D CNN, Dem3D EfficientNet)로 구성된 신뢰 기반 분류 위원회를 통해 보완됩니다. 이러한 모델은 각 모델의 강점을 활용하여 의사 결정 정확도를 높이기 위해 협력적으로 작동합니다.



### Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation (https://arxiv.org/abs/2407.17274)
Comments:
          Work in progress

- **What's New**: 본 논문에서는 Text-to-Image Retrieval(텍스트-이미지 검색)을 위한 새로운 Generative Cross-Modal Retrieval(생성형 교차 모달 검색) 방법인 AVG(Autoregressive Voken Generation)를 제안한다. AVG는 이미지를 'voken'(시각적 토큰)으로 토큰화하여 텍스트-이미지 검색을 토큰-보켄 생성 문제로 바꾼다. 이는 기존의 이미지 ID와 같은 단순한 문자열 식별자 대신 시각적 정보와 의미를 담은 vokens를 사용하여 더 정확한 검색을 가능하게 한다.



### SCIsegV2: A Universal Tool for Segmentation of Intramedullary Lesions in Spinal Cord Injury (https://arxiv.org/abs/2407.17265)
Comments:
          Accepted at MICCAI AMAI 2024 workshop

- **What's New**: 이 논문은  SCI (Spinal Cord Injury)  영상에서 병변 (lesion)을 자동으로 분할 (segmentation) 하고,  tissue bridge를 계산하는 새로운 방법론,  'SCIsegV2'를 제안한다. 이 도구는 7개의 서로 다른 병원에서 수집된 다양한 SCI 단계 (급성, 아급성, 만성) 와 원인 (외상성 SCI, 허혈성 SCI, 퇴행성 경추 척수증) 의 환자 데이터를 사용하여 훈련 및 검증되었다. 이 도구는  expert에 의해 수동으로 계산된  tissue bridge 와 큰 차이가 없어  MRI 바이오마커 (biomarker) 를 자동으로 도출하는 데 사용될 수 있음을 보여준다. SCIsegV2와 자동  tissue bridge 계산은 오픈 소스이며  Spinal Cord Toolbox  (v6.4 이상) 에서  `sct_deepseg -task seg_sc_lesion_t2w_sci`  와  `sct_analyze_lesion`  함수를 통해 사용할 수 있다.



### Sublinear Regret for An Actor-Critic Algorithm in Continuous-Time Linear-Quadratic Reinforcement Learning (https://arxiv.org/abs/2407.17226)
Comments:
          42 pages, 4 figures

- **What's New**: 이 논문에서는, 주어진 모델 파라미터를 사용하거나 추정하는 대신, 상태 과정의 변동성이 상태 및 제어 변수에 모두 의존하는 확산에 대한 연속 시간 선형-2차(LQ) 제어 문제의 한 종류를 위한 강화 학습(RL)을 연구한다. 모델 파라미터를 학습하거나 추정하지 않고 모델 프리 접근 방식을 적용하여 최적 정책 파라미터를 직접 학습하기 위한 액터-크리틱 알고리즘을 고안한다. 주요 기여는 새로운 탐색 일정(exploration schedule)을 도입하고 제안된 알고리즘의 후회 분석을 수행하는 것이다. 최적 파라미터로의 정책 파라미터 수렴 속도를 제공하고 알고리즘이 로그 계수까지 O(N^(3/4))의 후회 경계(regret bound)를 달성한다는 것을 증명한다. 시뮬레이션 연구를 통해 이론적 결과를 검증하고 제안된 알고리즘의 효율성과 신뢰성을 보여준다. 또한 상태 및 제어 의존적 변동성 설정에 적응된 최근 모델 기반 확률적 LQ RL 연구의 방법과 우리의 방법 사이에 수치적 비교를 수행하여 후회 경계 측면에서 전자의 성능이 더 우수하다는 것을 보여준다.



### Nonverbal Immediacy Analysis in Education: A Multimodal Computational Mod (https://arxiv.org/abs/2407.17209)
Comments:
          12 pages, 3 figures. Camera-ready version for the SAB 2024: 17th International Conference on the Simulation of Adaptive Behavior

- **What's New**: 이 논문은 교육 환경에서 비언어적 사회적 행동을 분석하기 위한 새로운 컴퓨팅 접근 방식을 소개합니다. 얼굴 표정, 제스처 강도, 공간 역학과 같은 다중 모드 행동 신호를 통합하여 모델은 RGB 교실 비디오에서 교사의 비언어적 즉각성(NVI)을 평가합니다. 독일 교실에서 400개의 30초 비디오 세그먼트 데이터셋이 모델 학습 및 검증을 위해 구성되었습니다. 제스처 강도 회귀 모델은 0.84의 상관관계를, 인식 거리 회귀 모델은 0.55의 상관관계를, NVI 모델은 0.44의 상관관계를 보였습니다. 이 모델은 개별 인간 평가자의 정확도에 근접하여 비언어적 행동 평가에 귀중한 지원을 제공할 수 있는 잠재력을 보여줍니다. 설문 조사 데이터와 훈련된 관찰자 평가 모두에 대해 검증된 모델은 관련 교육 결과와 중간에서 강한 상관관계를 보여주어 효과적인 교수 행동을 반영하는 효과를 보여줍니다. 이 연구는 비언어적 의사 소통 행동의 객관적인 평가를 발전시켜 교육 연구를 위한 새로운 경로를 열어줍니다.



### Take a Step and Reconsider: Sequence Decoding for Self-Improved Neural Combinatorial Optimization (https://arxiv.org/abs/2407.17206)
Comments:
          Accepted at ECAI-2024

- **What's New**: 이 논문은 NCO(Neural Combinatorial Optimization)에서 self-improved learning 방법을 위한 간단하고 문제와 독립적인 sequence decoding 방법을 제안합니다. 이 방법은 sampling without replacement을 사용하여 sequence를 순차적으로 decoding합니다. 또한, 이미 선택된 sequence는 무시하도록 policy를 수정하여 policy가 다양한 해를 고려하도록 강제합니다.



### ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only (https://arxiv.org/abs/2407.17197)
- **What's New**: 이 논문은 3D 객체 탐지 모델을 훈련하는데 필요한 정밀한 3D 어노테이션을 대체하기 위한 새로운 약지도 학습 방법을 제안한다. 이 방법은 이미지의 2D 바운딩 박스 어노테이션과 크기 정보만을 사용하여 3D 객체 탐지를 위한 3D 어노테이션을 생성한다. 2D 바운딩 박스만을 사용하여 3D 탐지 모델을 훈련하는 것은 3D 포즈의 모호성으로 인해 신뢰할 수 없다는 문제를 해결하기 위해, 이 논문은 3D 프록시 객체를 생성하여 훈련 데이터셋에 추가하는 방법을 제안한다. 또한, 2D 감독을 3D 탐지와 더 잘 맞추기 위해 새로운 2D 손실 표현을 사용하여 깊이 불변성(depth invariance)을 보장한다. 마지막으로, 더 어려운 인스턴스를 탐지하기 위해 오프라인 의사 라벨링(pseudo-labeling) 방식을 사용하여 3D 의사 라벨을 점차 개선한다.



### Robust Deep Hawkes Process under Label Noise of Both Event and Occurrenc (https://arxiv.org/abs/2407.17164)
Comments:
          ECAI2024

- **What's New**: 이 논문에서는 딥 Hawkes 프로세스 모델이 라벨 노이즈 (label noise)에 취약한 문제를 해결하기 위해 **Robust Deep Hawkes Process (RDHP)**라는 새로운 프레임워크를 제안합니다. RDHP는 이벤트 (event)와 타이밍 (timing) 모두에서 발생하는 라벨 노이즈의 영향을 최소화하여 Hawkes 모델의 강도 함수 (intensity function)의 정확성을 높입니다. 특히, RDHP는 이벤트 유형과 발생 시기에 모두 영향을 미치는 라벨 노이즈에 대해서도 효과적으로 작동합니다.



### XMeCap: Meme Caption Generation with Sub-Image Adaptability (https://arxiv.org/abs/2407.17152)
Comments:
          Accepted to MM 2024

- **What's New**: 이 논문은 기존 MCQ 평가 지표인 BLEU, ROUGE, METEOR가 MCQ의 교육적 가치를 고려하지 않고 단순히 골드 샘플과의 유사성만 평가한다는 점을 지적합니다. 따라서, MCQ의 대답 가능성 (answerability)을 측정하여 MCQ가 학생의 지식을 실제로 평가할 수 있는지 여부를 판단하는 새로운 지표, Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 학생 설문조사 결과를 기반으로 계산되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 KDA를 근사화합니다.



### Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspectiv (https://arxiv.org/abs/2407.17120)
- **What's New**: 본 논문은 Neural Tangent Kernel (NTK) 이론을 활용하여 Parameter-Efficient Fine-Tuning for Continual Learning (PEFT-CL) 모델의 성능을 분석하고 개선합니다. PEFT-CL는 기존 모델에 새로운 데이터를 학습시키면서 이전 데이터에 대한 지식을 유지하는 기술입니다. NTK 이론을 사용하여 PEFT-CL 모델의 성능 저하 원인을 분석하고, 이를 해결하기 위한 새로운 방법론인 NTK-CL을 제안합니다. NTK-CL은 각 샘플의 표현 범위를 확장하고, Adaptive Exponential Moving Average (EMA) 기반의 지식 유지 메커니즘을 사용하여 새로운 데이터 학습 과정에서 기존 지식이 손실되는 문제를 해결합니다. 또한, NTK-CL은 Task-feature orthogonality constraint를 적용하여 각 태스크에 대한 특징을 명확히 구분하여 이전 태스크의 지식을 간섭하는 것을 방지합니다.



### EverAdapt: Continuous Adaptation for Dynamic Machine Fault Diagnosis Environments (https://arxiv.org/abs/2407.17117)
- **What's New**: 본 논문에서는 지속적으로 변화하는 환경에서 모델 적응력을 향상시키기 위해 새로운 EverAdapt 프레임워크를 제안합니다. 특히, EverAdapt는 다양한 도메인에 걸쳐 특징 표현을 표준화하기 위해 소스 도메인 통계를 참조점으로 활용하는 새로운 연속 배치 정규화(CBN)를 도입합니다. EverAdapt는 이전 도메인의 통계 정보를 유지하는 동시에 새로운 상황에 효과적으로 적응합니다. CBN을 보완하여, 목표 도메인을 효과적으로 통합하기 위한 클래스 조건 도메인 정렬 모듈과 메모리 유지를 강화하는 샘플 효율적인 재생 전략을 설계했습니다.



### Neural Dueling Bandits (https://arxiv.org/abs/2407.17112)
Comments:
          Accepted at ICML 2024 Workshop on Foundations of Reinforcement Learning and Control

- **What's New**: This paper proposes novel algorithms for contextual dueling bandits (preference-based bandits) using neural networks to estimate non-linear reward functions. The algorithms are based on Upper Confidence Bound (UCB) and Thompson Sampling (TS) and achieve sub-linear regret guarantees.  The paper also extends the theoretical results to contextual bandit problems with binary feedback.



### PiPa++: Towards Unification of Domain Adaptive Semantic Segmentation via Self-supervised Learning (https://arxiv.org/abs/2407.17101)
Comments:
          This study is under IEEE TMM review. arXiv admin note: substantial text overlap with arXiv:2211.07609

- **What's New**: 이 논문은 지식 종속 가능성 (Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 실제 강의실에서의 사용성과의 강한 상관관계를 보여주는 Human Evaluation을 통해 KDA_disc와 KDA_soft의 효과를 입증했습니다.



### Towards Robust Knowledge Tracing Models via k-Sparse Attention (https://arxiv.org/abs/2407.17097)
Comments:
          Accepted at SIGIR'2023 (revised version with additional results)

- **What's New**: 본 논문은 Knowledge Tracing (KT) 모델의 Robustness 및 Generalization을 향상시키기 위해 Sparse Attention 기법을 적용한 새로운 프레임워크인 SparseKT를 제안합니다. SparseKT는 학생의 과거 학습 데이터에서 중요한 상호작용만 선택하여 모델의 성능을 향상시킵니다.



### OVR: A Dataset for Open Vocabulary Temporal Repetition Counting in Videos (https://arxiv.org/abs/2407.17085)
- **What's New**: 본 논문은 비디오에서의 반복적인 시간적 패턴을 표시하는 새로운 데이터셋, OVR(Over 발음)을 소개합니다. OVR은 72,000개 이상의 비디오를 포함하며, 각 어노테이션은 반복 횟수, 반복 시작 및 종료 시간, 그리고 반복되는 내용에 대한 자유 형식 설명을 제공합니다. 어노테이션은 Kinetics와 Ego4D에서 가져온 비디오에 대해 제공되며, 따라서 Exo와 Ego 보기 조건을 모두 포함하고 다양한 액션과 활동을 보여줍니다. 또한 OVR은 이전 비디오 반복 데이터셋보다 훨씬 규모가 큽니다. 본 논문에서는 최대 320 프레임 길이의 비디오에서 반복을 찾고 계산할 수 있는 기준 트랜스포머 기반 계산 모델인 OVRCounter를 제안합니다. 모델은 OVR 데이터셋에서 훈련 및 평가되었으며, 텍스트를 사용하여 계산할 대상 클래스를 지정하는 경우와 그렇지 않은 경우 성능이 평가되었습니다. 또한 이전 반복 계산 모델과 비교하여 성능이 평가되었습니다. 데이터셋은 [this https URL](this https URL)에서 다운로드 가능합니다.



### When Text and Images Don't Mix: Bias-Correcting Language-Image Similarity Scores for Anomaly Detection (https://arxiv.org/abs/2407.17083)
- **What's New**: 이 논문은 CLIP (Contrastive Language-Image Pre-training) 모델에서 발생하는 '텍스트 클러스터링 효과'와 '유사성 편향'을 분석하고 이를 해결하기 위한 새로운 방법인 BLISS (Bias-corrected Language Image Similarity Scoring) 를 제안합니다. CLIP은 이미지와 텍스트 입력 임베딩을 정렬하여 다양한 하위 작업에서 뛰어난 성능을 보여주며, 이상 탐지에도 유망합니다. 그러나 본 연구에서는 이미지 임베딩에서 멀리 떨어져 텍스트 입력 임베딩이 예상치 못하게 밀집되어 있는 현상을 발견했습니다. 이는 이미지-텍스트 입력 쌍을 정렬하는 모델의 대조 학습 목표와 배치됩니다. 이러한 현상으로 인해 '유사성 편향'이 발생하며, 이미지와 정상 레이블 텍스트 임베딩 간의 유사성에 대한 편향으로 인해 오탐과 누락 탐지 오류가 발생합니다. BLISS는 보조 외부 텍스트 입력 집합을 사용하여 이러한 유사성 편향을 직접 고려하는 방법입니다. BLISS는 간단하고, 이상 행동에 대한 강력한 귀납적 편향이나 비용이 많이 드는 훈련 과정이 필요하지 않으며, 벤치 마크 이미지 데이터 세트에서 기준 방법을 능가하며, 정상 데이터에 대한 액세스가 매우 제한적인 경우에도 뛰어난 성능을 보입니다.



### Curriculum Negative Mining For Temporal Networks (https://arxiv.org/abs/2407.17070)
- **What's New**: 이 논문은 TGNN(Temporal Graph Neural Network)의 학습 과정에서 발생하는 negative sampling 문제를 해결하기 위한 새로운 방법론, Curriculum Negative Mining (CurNM)을 제안합니다. 기존의 negative sampling 방법은 positive sparsity (각 타임스탬프에서 positive 샘플이 매우 적음)와 positive shift (타임스탬프에 따라 positive 샘플의 분포가 달라짐) 문제를 해결하지 못했습니다. CurNM은 이러한 문제를 해결하기 위해 model-aware curriculum learning framework을 사용하여 negative 샘플의 난이도를 조절합니다. 



### PatchFinder: A Two-Phase Approach to Security Patch Tracing for Disclosed Vulnerabilities in Open-Source Softwar (https://arxiv.org/abs/2407.17065)
Comments:
          to appear at ISSTA 2024

- **What's New**: 본 논문은 오픈소스 소프트웨어 (OSS) 취약점 (vulnerability)을 위한 패치 추적 (patch tracing) 시스템인 PatchFinder를 제안합니다. PatchFinder는 기존 시스템의 한계를 극복하기 위해 두 단계로 구성된 프레임워크를 사용하며, 각 단계에서 코드 변화와 CVE 설명을 이용하여 유사한 커밋을 추출하고 재랭킹 (re-ranking) 합니다. 특히, PatchFinder는 CVE 설명과 커밋 간의 의미적 상관관계를 학습하는 End-to-End 아키텍처를 통해 정확도를 높이고 계산 비용을 낮춥니다.



### Time Series Missing Imputation with Multivariate Radial Basis Function Neural Network (https://arxiv.org/abs/2407.17040)
- **What's New**: 이 논문은 RBFNN(Radial Basis Functions Neural Network)을 기반으로 시간 시계열 데이터의 결측값을 처리하는 새로운 방법을 제시합니다. 제안된 MIM-RBFNN 모델은 시간 정보를 활용하여 연속적인 함수를 생성하며, 결측값의 시간 간격까지 고려하여 학습합니다. 또한, MIRNN-CF 모델은 MIM-RBFNN으로 생성된 연속 함수를 이용하여 재귀 신경망을 확장하여 시간 정보를 더 효과적으로 활용합니다.



### Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inferenc (https://arxiv.org/abs/2407.17033)
- **What's New**: 이 논문에서는 DGP(Deep Gaussian Process) 모델의 inducing point에 대한 후방 분포 (posterior distribution) 추론을 위한 새로운 방법인 DDVI(Denoising Diffusion Variational Inference)를 제안합니다. DDVI는 denoising diffusion SDE(Stochastic Differential Equation)를 사용하여 inducing point의 후방 샘플을 생성하고 score matching method를 통해 신경망을 사용하여 score function을 근사합니다. 이를 통해 DGP의 marginal likelihood function에 대한 새로운 명시적 variational lower bound를 도출합니다. 기존의 variational inference 방법은 편향 (bias) 문제가 발생할 수 있지만, DDVI는 이러한 문제를 해결하고 정확한 후방 추론을 가능하게 합니다.



### Enhancing Environmental Monitoring through Multispectral Imaging: The WasteMS Dataset for Semantic Segmentation of Lakeside Was (https://arxiv.org/abs/2407.17028)
- **What's New**: 이 논문은 호숫가 녹지 공간의 폐기물을 구분하기 위한 새로운 멀티 스펙트럼 데이터셋인 WasteMS를 소개합니다. 이 데이터셋은 다양한 조명 조건에서 촬영된 다양한 폐기물 유형을 포함하고 있으며, 정확한 주석 작업을 거쳤습니다. 이를 통해 호숫가 잔디밭에서 폐기물을 분류하는 작업의 어려움과 WasteMS 데이터셋의 활용 가능성을 보여줍니다.



### Pensieve Discuss: Scalable Small-Group CS Tutoring System with AI (https://arxiv.org/abs/2407.17007)
Comments:
          6 pages, 7 figures, 4 tables, 1 page of references

- **What's New**: 이 논문은 소그룹 튜터링 세션에서 학생 협업과 경험을 향상시키기 위해 설계된 Pensieve Discuss라는 소프트웨어 플랫폼을 소개합니다. 이 플랫폼은 scaffolding된 프로그래밍 문제에 대한 동시 편집(synchronous editing), 온라인 인간 튜터, AI 튜터를 통합합니다.



### SepsisLab: Early Sepsis Prediction with Uncertainty Quantification and Active Sensing (https://arxiv.org/abs/2407.16999)
Comments:
          To be published in KDD 2024

- **What's New**: 이 논문은 기존의 MCQ 생성 평가 지표가 단어 유사성에만 초점을 맞추고 실제 교육적 가치를 고려하지 않는다는 문제점을 제기합니다. 새로운 지표인 'Knowledge Dependent Answerability (KDA)'를 제안하여 MCQ가 학생의 지식 수준을 정확히 평가할 수 있는지 측정합니다. 또한, KDA를 자동으로 계산하는 KDA_disc와 KDA_cont 지표를 개발하여, 이 지표들이 Human evaluation과 강한 상관관계를 가지는 것을 확인합니다.



### Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Mod (https://arxiv.org/abs/2407.16982)
- **What's New**: 본 논문은 텍스트 지시만으로 이미지에 새로운 객체를 추가하는 중요한 문제를 다룹니다. 이 과제는 조명, 질감, 공간적 위치와 같은 일관된 시각적 컨텍스트를 유지하면서 새로운 객체를 이미지에 매끄럽게 통합해야 하기 때문에 어려움을 겪습니다. 기존의 텍스트 기반 이미지 인페인팅 (inpainting) 방법들은 객체를 추가할 수 있지만 배경 일관성을 유지하지 못하거나, 바운딩 박스를 지정하거나 사용자가 스크래치한 마스크를 사용하는 번거로운 인간 개입을 포함합니다. 이러한 과제를 해결하기 위해 본 연구에서는 텍스트 기반 제어만으로 텍스트 기반 객체 추가를 용이하게 하는 Text-to-Image (T2I) 모델인 Diffree를 소개합니다. 이를 위해 본 연구에서는 고급 이미지 인페인팅 기법을 사용하여 객체를 제거한 고급 합성 데이터셋인 OABench를 구축했습니다. OABench는 원본 이미지, 객체가 제거된 인페인팅 이미지, 객체 마스크, 객체 설명의 74,000개의 실제 세계 튜플로 구성됩니다. 추가 마스크 예측 모듈을 갖춘 Stable Diffusion 모델을 사용하여 OABench에서 학습된 Diffree는 새 객체의 위치를 고유하게 예측하고 텍스트만으로 안내를 받아 객체를 추가합니다. 광범위한 실험 결과, Diffree는 배경 일관성, 공간적 적절성, 객체 관련성 및 품질을 유지하면서 높은 성공률로 새로운 객체를 추가하는 데 탁월한 성능을 보여줍니다.



### Case-Enhanced Vision Transformer: Improving Explanations of Image Similarity with a ViT-based Similarity Metric (https://arxiv.org/abs/2407.16981)
- **What's New**: 본 논문은 이미지 데이터의 유사성 평가에 대한 설명 가능성을 향상시키기 위한 새로운 유사성 측정 방법인 케이스 향상 비전 트랜스포머 (CEViT)에 대한 초기 연구를 제시합니다. CEViT는 k-Nearest Neighbor (k-NN) 분류에 통합되어 최첨단 컴퓨터 비전 모델과 비교할 만한 분류 정확도를 달성하면서 클래스 간 차이를 보여주는 기능을 추가합니다. CEViT 설명은 이전 케이스의 영향을 받아 해당 케이스와 관련된 유사성의 측면을 보여줄 수 있습니다.



### Stochastic Variance-Reduced Iterative Hard Thresholding in Graph Sparsity Optimization (https://arxiv.org/abs/2407.16968)
- **What's New**: 이 논문은 그래프 스파스(graph sparsity) 최적화를 위한 새로운 확률적 분산 감소 경사 하강법(Stochastic variance-reduced gradient-based methods)인 GraphSVRG-IHT와 GraphSCSG-IHT를 제안합니다. 이 방법들은 그래프 스파스 최적화 문제에 적용 가능한 일반적인 이론적 분석 프레임워크를 제공하며, 선형 수렴 속도를 보여줍니다.  



### Cheems: Wonderful Matrices More Efficient and More Effective Architectur (https://arxiv.org/abs/2407.16958)
- **What's New**: 이 논문은 긴 텍스트를 처리하는 데 있어서 효율적이고 효과적인 언어 모델을 구축하기 위해, 선택적 상태 공간 모델(SSM) 알고리즘과 쿼드라틱 셀프 어텐션(Quadratic Self-Attention) 알고리즘을 결합한 새로운 하이브리드 아키텍처 Cheems를 제안합니다. 특히, 다양한 위치 인코딩 기법(position encoding)의 효과를 조사하고, SSM과 어텐션(Attention)을 결합하는 효과적인 방법을 제시하며, 다양한 분야의 지식을 학습할 수 있는 크로스 도메인 밀리언 믹스드 익스퍼트(CDMMOE)를 도입합니다.



### Synthetic Trajectory Generation Through Convolutional Neural Networks (https://arxiv.org/abs/2407.16938)
Comments:
          To appear in the proceedings of the 21st Annual International Conference on Privacy, Security & Trust (PST 2024)

- **What's New**: 본 논문은 CNN 기반 모델에 적용 가능하도록 경로 데이터를 변환하는 **Reversible Trajectory-to-CNN Transformation (RTCT)** 를 제안합니다.  이 변환을 통해 기존 CNN 기반 GAN 모델 (DCGAN)을 사용하여 경로 데이터를 생성하는 PoC (proof-of-concept) 모델을 구축하고, RNN 기반 경로 GAN 모델과 비교 분석했습니다.  RTCT는 CNN 모델을 사용하여 경로 데이터를 생성하는 새로운 가능성을 제시합니다.



### Synthetic Data, Similarity-based Privacy Metrics, and Regulatory (Non-)Complianc (https://arxiv.org/abs/2407.16929)
Comments:
          Accepted to the 2nd Workshop on Generative AI and Law (GenLaw 2023), part of ICML 2024

- **What's New**: 이 논문은 synthetic data의 regulatory compliance를 보장하기 위해 similarity-based privacy metrics가 충분하지 않다는 것을 주장합니다. 이는 similarity-based privacy metrics가 singling out과 linkability를 보호하지 못하고 motivated intruder test를 완전히 무시하기 때문입니다.



### Assessing the role of clinical summarization and patient chart review within communications, medical management, and diagnostics (https://arxiv.org/abs/2407.16905)
- **What's New**: 본 논문은 EHR(Electronic Health Records) 데이터에서 비정형 환자 데이터를 효과적으로 요약하는 것이 정확한 진단과 효율적인 환자 치료에 필수적이지만, 임상의들은 종종 정보 과부하와 시간 제약에 어려움을 겪는다는 점을 강조합니다. 이 리뷰는 의사소통, 진단 및 관리에 대한 환자 차트 검토의 중요한 영향과 뛰어난 문제점을 다룬 최신 문헌과 사례 연구를 탐구합니다. 또한, 인공 지능(AI)을 임상 요약 작업에 통합하기 위한 최근 노력과 행정적 부담 감소 및 환자 중심 치료 개선을 포함하되 이에 국한되지 않는 임상의의 잠재력에 대한 혁신적인 영향을 논의합니다.



### US-China perspectives on extreme AI risks and global governanc (https://arxiv.org/abs/2407.16903)
- **What's New**: 본 논문은 중국과 미국 AI 전문가들이 인공지능의 안전 및 보안 위협과 관련된 국제 협력에 대해 어떻게 생각하는지 조사했습니다. 인공 일반 지능(AGI)과 같은 첨단 인공지능 기술이 국가 및 세계 안보에 미칠 수 있는 영향에 초점을 맞춰 공개적으로 발표된 미국과 중국의 주요 기술 및 정책 리더들의 발언을 수집했습니다.

- **Technical Details**: 이 연구는 미국과 중국의 주요 기술 및 정책 리더들의 발언을 수집하여 인공지능의 안전 및 보안 위협에 대한 인식, 특히 AGI와 같은 첨단 인공지능 기술이 가져올 수 있는 극단적인 위험, 그리고 국제 협력 가능성을 분석했습니다.

- **Performance Highlights**: 두 국가의 전문가들은 AGI로 인한 위험, 지능 폭발로 인한 위험, 인간의 통제를 벗어난 AI 시스템으로 인한 위험에 대해 우려를 표명했습니다. 두 국가 모두 안전 기준과 위험 관리 관행에 대한 국제 협력을 촉진하기 위한 초기 노력을 시작했습니다.

- **Findings**: 본 연구는 중국과 미국에서 첨단 AI와 관련된 안전 및 보안 위협에 대한 우려가 높음을 보여주었고, 국제 협력을 통한 위협 완화에 대한 필요성을 강조했습니다. 연구 결과는 AI 기술 개발과 관련된 안전 및 보안 문제에 대한 정책 결정자들의 인식을 높이고 국제 협력을 위한 토대를 마련할 수 있습니다.

- **Limitations**: 본 연구는 공개적으로 발표된 자료만을 대상으로 하였으며, 모든 관련 발언을 포함하지는 못했습니다.

- **Future Work**: 본 연구는 중국과 미국 AI 전문가들의 인식을 심층적으로 분석하고 국제 협력을 위한 구체적인 방안을 모색하는 추가 연구의 필요성을 보여줍니다.



### The Potential and Perils of Generative Artificial Intelligence for Quality Improvement and Patient Safety (https://arxiv.org/abs/2407.16902)
- **What's New**: This paper proposes a novel automatic evaluation metric, coined Knowledge Dependent Answerability (KDA), for evaluating the quality of Multiple Choice Questions (MCQ) generated by AI models. Existing metrics like BLEU, ROUGE, and METEOR focus on n-gram similarity, neglecting the educational value of MCQs. KDA measures the MCQ's answerability by considering the student's knowledge of the target fact. The authors also present two automatic metrics, KDA_disc and KDA_cont, which leverage pre-trained language models to mimic student problem-solving behavior.



### Regulating AI Adaptation: An Analysis of AI Medical Device Updates (https://arxiv.org/abs/2407.16900)
- **What's New**: 이 논문은 FDA 승인을 받은 AI 의료기기의 업데이트 빈도와 종류를 체계적으로 분석하여 AI 모델 업데이트와 관련된 규제적 고려 사항을 심층적으로 살펴본다.



### (Unfair) Norms in Fairness Research: A Meta-Analysis (https://arxiv.org/abs/2407.16895)
- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하는 새로운 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다. KDA는 MCQ의 대답 가능성(answerability)을 측정하여 학생의 지식을 얼마나 잘 평가하는지 확인합니다. 이 메트릭은 학생들의 응답을 분석하여 계산되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭으로 근사화됩니다. 이 두 메트릭은 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 행동을 모방합니다.



### Estimating the Increase in Emissions caused by AI-augmented Search (https://arxiv.org/abs/2407.16894)
- **What's New**: 본 논문은 AI 기반 검색 결과 요약의 에너지 소비 증가를 분석하고 기존 검색 엔진과 비교하여 환경적 영향을 강조한다. 특히 BLOOM 모델과 ChatGPT와 같은 대규모 언어 모델의 에너지 소비량을 분석하여 기존 검색과 비교하여 에너지 소비가 60-70배 증가한다는 결론을 내린다. 또한, ChatGPT의 훈련 과정보다 실제 사용으로 인한 에너지 소비가 더 크다는 점을 강조한다.



### Why Machines Can't Be Moral: Turing's Halting Problem and the Moral Limits of Artificial Intelligenc (https://arxiv.org/abs/2407.16890)
- **What's New**: 이 연구에서는 자동 MCQ 생성 평가에 사용되는 기존 메트릭 (BLEU, ROUGE, METEOR)의 단점을 지적하고, 교육적 가치를 고려한 새로운 자동 평가 메트릭을 제안합니다. 새 메트릭은 KDA (Knowledge Dependent Answerability)라고 불리며, MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 새로운 메트릭을 제안하며, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 과정을 모방합니다.



### A Nested Model for AI Design and Validation (https://arxiv.org/abs/2407.16888)
- **What's New**: 이 논문은 교사의 시간을 줄일 수 있는 MCQ (Multiple Choice Questions) 생성을 위한 새로운 평가 지표인 '지식 종속 가능성 (Knowledge Dependent Answerability, KDA)'를 제안합니다. 기존 평가 지표인 BLEU, ROUGE, METEOR는 단어 유사도만을 측정하지만, KDA는 MCQ가 학생의 지식을 측정하는 능력을 평가합니다.



### Comprehensive AI Assessment Framework: Enhancing Educational Evaluation with Ethical AI Integration (https://arxiv.org/abs/2407.16887)
Comments:
          13 Pages, 2 figures, 1 Framework

- **What's New**: 이 논문은 교육 평가에 인공지능(AI)을 윤리적으로 통합하는 것을 목표로 하는 포괄적인 AI 평가 프레임워크(CAIAF)를 제시합니다. CAIAF는 Perkins, Furze, Roe, MacVaugh가 개발한 AI 평가 척도(AIAS)를 발전시킨 것입니다. 이는 CAIAF가 교육 수준과 실시간 상호 작용 및 개인화된 지원과 같은 고급 AI 기능에 따라 명확한 구분을 두면서 엄격한 윤리적 지침을 통합하는 부분에서 차이가 있습니다. 이 프레임워크는 사용자 친화성을 높이는 색상 그라디언트를 사용하여 매우 직관적으로 사용할 수 있습니다. 방법론적으로 이 프레임워크는 철저한 문헌 검토 및 주제에 대한 실질적인 통찰력을 통해 개발되었으며 다양한 교육 환경에서 사용할 수 있는 역동적인 도구가 되었습니다. 이 프레임워크는 더 나은 학습 결과를 보장하고 학업의 완전성을 유지하며 AI의 책임감 있는 사용을 촉진하여 현대 교육 실습에서 이 프레임워크가 필요합니다. 



### Cluster Model for parsimonious selection of variables and enhancing Students Employability Prediction (https://arxiv.org/abs/2407.16884)
Comments:
          8 pages

- **What's New**: 본 논문은 교육 데이터 마이닝(EDM) 분야에서 학생 고용 가능성 예측을 위한 클러스터 기반 모델을 제안합니다. 교육 데이터는 일반적으로 대량, 다차원적이며 불균형적 특징을 가지고 있습니다. 이러한 데이터에서 지식을 추출하는 과정은 자체적으로 여러 문제점을 가지고 있으며 매우 복잡한 작업입니다. 본 논문은 인도 전역의 여러 대학 및 기관의 공학 및 MCA(컴퓨터 응용 분야 석사) 학생 데이터를 수집하여 분석했습니다. 이 데이터셋은 대규모, 불균형 및 다차원적입니다. 제안된 클러스터 기반 모델은 전처리 단계에서 적용되어 변수 선택을 간소화하고 예측 알고리즘의 성능을 향상시킵니다. 따라서 학생 고용 가능성 예측을 더욱 정확하게 수행할 수 있습니다.



### Comparative Analysis Vision of Worldwide AI Courses (https://arxiv.org/abs/2407.16881)
Comments:
          9 pages, 6 figures

- **What's New**: 본 연구는 전 세계 대학의 인공지능(AI) 교육 커리큘럼 구조를 조사합니다. 주요 대학의 커리큘럼을 조사함으로써 연구는 전 세계적 규모의 AI 교육에 대한 심층적인 이해에 기여하여 AI 환경의 진화하는 요구에 맞춰 교육 관행을 조정하는 데 도움이 됩니다. 이 연구는 주요 대학의 다양한 과정 구조를 탐구하고, 최신 트렌드와 우선 순위를 조사하여 AI 교육의 미묘한 접근 방식을 밝혀냅니다. 또한, CS2023 커리큘럼 지침과의 일치 및 불일치를 파악하기 위해 자주 가르치는 핵심 AI 주제와 학습 내용을 조사합니다. 또한, 이 연구는 여러 국가의 대학들이 AI 교육에 접근하는 방식을 조사하여 교육 목표, 우선 순위, 잠재적 직업, 방법론을 분석하여 AI 교육의 세계적 규모와 영향을 이해합니다.



### Balanced Multi-Relational Graph Clustering (https://arxiv.org/abs/2407.16863)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 본 논문은 Multi-relational graph clustering에서의 view imbalance 문제를 해결하기 위해 새로운 지표인 Aggregation Class Distance (ACD)를 제안한다. 또한, unsupervised dominant view mining과 dual signal guided representation learning을 결합한 Balanced Multi-Relational Graph Clustering (BMGC) 알고리즘을 제안한다. BMGC는 학습 과정에서 dominant view를 동적으로 찾아내고, 이를 이용하여 representation learning을 개선한다.



### Synth4Kws: Synthesized Speech for User Defined Keyword Spotting in Low Resource Environments (https://arxiv.org/abs/2407.16840)
Comments:
          5 pages, 5 figures, 2 tables The paper is accepted in Interspeech SynData4GenAI 2024 Workshop - this https URL

- **What's New**: 이 논문에서는 MCQ 생성의 교육적 가치를 고려하여 새로운 자동 평가 지표인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'를 제안합니다. KDA는 학생의 지식 수준을 정확하게 평가할 수 있는 MCQ의 능력을 측정합니다.



### A Multi-Level Hierarchical Framework for the Classification of Weather Conditions and Hazard Prediction (https://arxiv.org/abs/2407.16834)
Comments:
          6 pages

- **What's New**: 이 논문은 날씨 상태와 위험 예측을 위한 다층적 계층적 프레임워크를 제시합니다. 이 프레임워크는 이미지 데이터를 사용하여 11가지 특정 유형의 날씨 이미지를 분류하고 실시간 날씨 정보를 제공합니다. 특히, 이 프레임워크는 전통적인 날씨 예보가 부정확한 상황, 예를 들어 위험한 날씨에서 자율 주행 자동차의 안전 운행을 보장하는 데 유용합니다. 

- **Technical Details**: 이 논문은 이미지 데이터를 사용하여 11가지 날씨 이미지(이슬, 서리, 결빙, 서리, 눈, 우박, 비, 번개, 무지개, 모래 폭풍)를 분류하고 실시간 날씨 정보를 제공하는 다층적 계층적 프레임워크를 제시합니다. 이 프레임워크는 이미지를 11가지 날씨 범주로 분류하는 능력을 갖추고 있으며, 0.9329의 정확도로 실시간 날씨 정보를 제공합니다. 

- **Performance Highlights**: 이 프레임워크는 11가지 날씨 범주로 이미지를 분류하고 실시간 날씨 정보를 제공하며, 0.9329의 정확도를 달성했습니다. 



### AI-Enhanced 7-Point Checklist for Melanoma Detection Using Clinical Knowledge Graphs and Data-Driven Quantification (https://arxiv.org/abs/2407.16822)
- **What's New**: 이 논문은 7점 체크리스트 (7PCL) 을 이용한 악성 흑색종 진단의 정확성을 높이기 위해 새로운 진단 방법을 제안합니다. 7PCL은 7가지 특징 (attribute) 에 점수를 부여하는데, 주요 특징은 각각 2점, 부차적 특징은 각각 1점입니다. 총 3점 이상이면 생검 등 추가적인 검사를 진행합니다. 기존 방법은 모든 특징에 같은 가중치를 부여하여 정확성이 떨어지고 특징 간의 연관성을 무시한다는 한계점이 있습니다. 또한, 기존 딥러닝 연구는 악성 흑색종을 예측하는 것과 특징을 예측하는 것을 동일하게 중요하게 다루어 악성 흑색종 진단에 필요한 특징의 중요성을 간과했습니다. 본 논문에서는 이러한 한계점을 해결하기 위해 Clinical Knowledge-Based Topological Graph (CKTG) 와 Gradient Diagnostic Strategy with Data-Driven Weighting Standards (GD-DDW) 두 가지 혁신적인 요소를 통합한 새로운 진단 방법을 제안합니다.



### In Search for Architectures and Loss Functions in Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2407.16807)
Comments:
          20 pages, 10 figures, 3 tables

- **What's New**: 이 논문은 Multi-objective Reinforcement Learning(MORL)에 대한 새로운 접근 방식을 제안합니다. 특히, 다양한 목표 함수들 사이의 균형을 유지하는 데 중점을 두어 다양한 실제 RL 문제에 대한 적용 가능성을 높였습니다. 기존의 MORL 연구는 대부분 가치 기반 손실 함수에 초점을 맞추었지만, 이 논문에서는 모델-프리 정책 학습 손실 함수와 다양한 아키텍처 선택에 대한 영향을 실험적으로 분석했습니다.



### Multimodal Machine Learning in Mental Health: A Survey of Data, Algorithms, and Challenges (https://arxiv.org/abs/2407.16804)
- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 해당 대상 사실에 대한 학생의 지식을 평가할 수 있는 능력을 측정합니다. 기존의 BLEU, ROUGE, METEOR와 같은 평가 지표는 n-gram 유사성에만 초점을 맞춘 반면, KDA는 MCQ의 교육적 가치를 고려합니다. KDA를 측정하기 위해, 본 연구는 학생들의 답변을 기반으로 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 인간 연구 결과, KDA_disc와 KDA_cont가 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.



### Fusion and Cross-Modal Transfer for Zero-Shot Human Action Recognition (https://arxiv.org/abs/2407.16803)
- **What's New**: 이 논문은 인간의 움직임과 행동을 이해하기 위해 시각과 관성 센서 데이터 간의 지식 전이 (cross-modal transfer) 를 위한 새로운 방법, FACT (Fusion and Cross-modal Transfer) 를 제안합니다. FACT는 시각 데이터로 학습된 모델이 관성 센서 데이터만으로 인간 행동을 인식 (Human Action Recognition, HAR) 할 수 있도록 합니다. 특히, 기존의 cross-modal transfer 학습 방식과 달리, FACT는 학습 중에 관성 센서 데이터에 대한 라벨을 필요로 하지 않고, 테스트 시에만 관성 센서 데이터를 사용합니다. 이는 zero-shot cross-modal transfer 학습을 가능하게 합니다. 또한, FACT는 시간 정보를 효과적으로 처리하기 위해 시간 연속적인 버전인 T-FACT도 제시합니다.



### Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels (https://arxiv.org/abs/2407.16802)
- **What's New**: 본 논문은 기존 MCQ 평가 지표들이 교육적 가치를 제대로 반영하지 못한다는 문제점을 지적하며 새로운 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지를 측정합니다. 또한, KDA를 자동화하기 위한 KDA_disc와 KDA_cont 두 가지 지표를 제안합니다. 이러한 지표들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 높은 상관관계를 가지고 있음을 보여주었습니다. 더 나아가, KDA_disc와 KDA_cont는 n-gram 기반 유사도 지표와 함께 사용될 때, 전문가가 평가한 다양한 MCQ 품질 척도를 예측하는 데 강력한 힘을 가지고 있습니다.



### What Matters in Range View 3D Object Detection (https://arxiv.org/abs/2407.16789)
- **What's New**: 본 논문은 Lidar 기반의 3D 물체 탐지 모델에서 Range-view 표현을 사용하여 최첨단 성능을 달성한 새로운 모델을 제안합니다. 기존 연구에서 제안된 다양한 기법 없이도 최첨단 성능을 달성했으며, 이는 Range-view 3D 물체 탐지 모델의 중요한 진전입니다. 또한 Argoverse 2와 Waymo Open 데이터셋에서의 실험을 통해 핵심적인 통찰력을 얻었으며, 이는 Range-view 3D 물체 탐지 모델의 설계 및 성능 향상에 중요한 지침을 제공합니다. 특히, Input feature dimensionality의 중요성, 3D 공간 근접성 기반 분류 손실의 효용성, Range subsampling 기법의 효과성 등을 강조하며 기존 연구에서 제안된 복잡한 기법 없이도 최고의 성능을 달성할 수 있음을 보여줍니다. 본 논문은 Range-view 3D 물체 탐지 모델의 새로운 기준을 제시하며, 자율 주행 분야의 발전에 기여할 것으로 예상됩니다.



### PrISM-Observer: Intervention Agent to Help Users Perform Everyday Procedures Sensed using a Smartwatch (https://arxiv.org/abs/2407.16785)
Comments:
          conditionally accepted to ACM UIST 2024

- **What's New**: 본 논문에서는 PrISM-Observer라는 스마트워치 기반의 실시간 개입 시스템을 소개합니다. 이 시스템은 사용자의 행동을 관찰하고 오류를 예방하여 일상적인 작업을 지원하도록 설계되었습니다. 사용자는 시스템에 정보를 직접 입력하지 않고도, PrISM-Observer는 다양한 센서를 통해 사용자 행동을 실시간으로 감지하고 분석하여 적시에 개입합니다.



### PLM-Net: Perception Latency Mitigation Network for Vision-Based Lateral Control of Autonomous Vehicles (https://arxiv.org/abs/2407.16740)
Comments:
          13 pages excluding the appendixes. 19 pages including appendixes

- **What's New**: 자동 MCQ 생성 평가를 위한 새로운 지식 의존 가능성 (KDA) 메트릭이 제안되었습니다. 이 메트릭은 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 자동 평가 메트릭입니다.  



### Theoretical Analysis of Privacy Leakage in Trustworthy Federated Learning: A Perspective from Linear Algebra and Optimization Theory (https://arxiv.org/abs/2407.16735)
- **What's New**: 본 논문에서는 연합 학습에서의 개인 정보 유출을 선형 대수와 최적화 이론의 두 가지 관점에서 분석하여 개인 정보 보호를 강화하는 연합 학습 알고리즘을 설계하기 위한 이론적 기반을 마련했습니다. 특히, 배치 데이터의 Jacobian 행렬이 full rank가 아닐 때, 동일한 모델 업데이트를 생성하는 다른 배치 데이터가 존재하여 개인 정보 보호 수준을 보장할 수 있다는 점을 증명했습니다. 또한 최적화 이론 관점에서 배치 크기, 왜곡 정도 및 기타 요인에 따라 개인 정보 유출에 대한 상한을 설정했습니다.



### PyBench: Evaluating LLM Agent on various real-world coding tasks (https://arxiv.org/abs/2407.16732)
Comments:
          9 pages

- **What's New**: PyBench, a new benchmark, is proposed to evaluate the real-world coding capabilities of LLM agents. It's designed to assess how LLMs can reason, write executable Python code, and utilize code results in a practical context, unlike existing benchmarks focused on simplistic or extremely complex coding tasks.



### PateGail: A Privacy-Preserving Mobility Trajectory Generator with Imitation Learning (https://arxiv.org/abs/2407.16729)
- **What's New**: 이 논문에서는 PateGail이라는 새로운 프라이버시 보호 모형을 제안하여 개인 정보 유출 위험 없이 이동 경로를 생성합니다. 이 모형은 사용자 기기에서 분산된 이동 데이터를 기반으로 학습되며, 개인 식별자 (discriminator)는 각 기기에서 로컬하게 학습되어 실제 이동 경로와 생성된 이동 경로를 구분하고 보상합니다. 개인 식별자가 보상하는 정보만 서버와 기기 사이에 공유되고, 이 정보는 차등 프라이버시 (differential privacy) 를 만족하는 섭동 메커니즘을 사용하여 추가적인 개인 정보 보호를 제공합니다. 또한, 사람의 의사 결정 과정을 더 잘 모형화하기 위해 개인 식별자에서 얻은 보상을 집계하는 새로운 메커니즘을 제안했습니다. 이 연구는 이러한 메커니즘을 사용하여 모델이 사용자의 할인된 총 보상의 하한을 최대화한다는 것을 이론적으로 증명했습니다.



### Distributed Difference of Convex Optimization (https://arxiv.org/abs/2407.16728)
Comments:
          9 pages, 7 figures

- **What's New**: 본 논문은 모든 에이전트 i에 대한 지역적 목적 함수가 두 개의 볼록 함수 f_i와 g_i의 차이로 주어지는 (difference-of-convex (DC) 형태) 분산 최적화 문제의 한 종류를 해결하는 데 중점을 둡니다. 여기서 f_i와 g_i는 잠재적으로 비미분 가능합니다. 에이전트는 n개의 노드를 포함하는 방향 그래프를 통해 통신합니다. 우리는 f_i와 g_i 함수의 부드러운 근사를 만들고, 부드러운 서로게이트의 기울기와 유한 시간 근사 합의 프로토콜을 활용하는 분산 알고리즘을 개발합니다. 우리는 이 알고리즘을 DDC-Consensus라고 부릅니다. 개발된 DDC-Consensus 알고리즘은 비대칭 방향 그래프 토폴로지를 허용하며 분산적으로 합성될 수 있습니다. 우리는 DDC-Consensus 알고리즘이 비볼록 분산 최적화 문제의 정지점으로 수렴함을 입증합니다. DDC-Consensus 알고리즘의 성능은 비볼록 DC-정규화된 분산 최소 제곱 문제를 해결하기 위한 시뮬레이션 연구를 통해 평가됩니다. 숫자 결과는 제안된 알고리즘의 효과를 입증합니다.



### Topology Reorganized Graph Contrastive Learning with Mitigating Semantic Drif (https://arxiv.org/abs/2407.16726)
- **What's New**: 본 논문은 그래프 콘트라스티브 학습(GCL)에서 노드 표현 학습을 향상시키기 위해 두 가지 새로운 데이터 증강 방법을 제안합니다. 첫 번째는 특징 공간(feature space)에서 노드 간의 의미적 상관 관계를 분석하여 데이터 증강을 수행하고, 두 번째는 인접 행렬(adjacency matrix)의 대수적 특성을 이용하여 고유 분해(eigen-decomposition)를 통해 위상(topology)을 특징화합니다. 이러한 방법들을 통해 더 나은 뷰(view)를 구축하기 위해 중요한 에지를 유지합니다. 또한, 잘못된 음성 샘플을 필터링하여 의미적 드리프트(semantic drift)를 줄이기 위해 프로토타입 기반 음성 쌍 선택(prototype-based negative pair selection)을 설계했습니다.



### Research on Adverse Drug Reaction Prediction Model Combining Knowledge Graph Embedding and Deep Learning (https://arxiv.org/abs/2407.16715)
Comments:
          12 pages, 4 figures, 9 tables

- **What's New**: 이 논문은 약물 부작용 예측 모델을 개발하여 의사의 약물 처방 결정을 돕는 것을 목표로 합니다. 기존 연구에서 특징 벡터가 고차원이고 희소하며 각 부작용마다 독립적인 예측 모델을 만들어야 하는 문제를 해결하기 위해 지식 그래프 임베딩(knowledge graph embedding)과 딥러닝을 기반으로 하는 통합적인 부작용 예측 모델을 개발했습니다. 이 모델은 실험 결과를 예측하고 약물 부작용을 통합적으로 예측할 수 있습니다. 지식 그래프 임베딩 기술은 약물 간의 연관 정보를 결합하여 특징 행렬의 고차원 희소성 문제를 해결하며, 딥러닝의 효율적인 학습 능력은 모델의 예측 정확도를 향상시킵니다.



### Masked Graph Learning with Recurrent Alignment for Multimodal Emotion Recognition in Conversation (https://arxiv.org/abs/2407.16714)
Comments:
          15 pages, 9 figures

- **What's New**: 이 논문은 대화에서의 다중 모달 감정 인식 (MERC)을 위한 새로운 접근 방식인 마스킹 그래프 학습과 재귀적 정렬 (MGLRA)을 제안합니다. MGLRA는 다중 모달 특징을 정렬하기 위해 메모리 기능이 있는 반복적 모듈을 사용하고, 다중 모달 특징 융합을 위해 마스킹 GCN (Graph Convolutional Network) 을 사용합니다.



### Multi-Scale Simulation of Complex Systems: A Perspective of Integrating Knowledge and Data (https://arxiv.org/abs/2306.10275)
- **What's New**: 자동 MCQ 생성을 위한 새로운 평가 지표인 **Knowledge Dependent Answerability (KDA)**가 제안되었으며, 이는 MCQ의 대답 가능성(answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다.

- **Technical Details**: KDA는 인간 설문 조사에서 수집된 학생 응답을 기반으로 측정하며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 (1) KDA와 (2) 전문가가 평가한 실제 강의실 설정에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 지표와 결합하면 다양한 전문가가 평가한 MCQ 품질 측정에 대한 강력한 예측력을 보여줍니다.



### HAPFI: History-Aware Planning based on Fused Information (https://arxiv.org/abs/2407.16533)
Comments:
          7 pages, 3 figures, published to ICRA 2024

- **What's New**: 본 논문은 embodied instruction following(EIF)에서 멀티모달 정보를 이용하여 과거 정보를 효과적으로 활용하는 새로운 프레임워크인 HAPFI(History-Aware Planning based on Fused Information)를 제안합니다. HAPFI는 RGB 관측값, 바운딩 박스, 하위 목표, 자연어 지시 등 다양한 모달리티의 정보와 그 이력을 통합하여 상황을 더 잘 이해하고 다음 단계의 행동을 정확하게 예측할 수 있습니다. 이는 특히 장기간에 걸쳐 이루어지는 작업에서 유용합니다.



### Virtue Ethics For Ethically Tunable Robotic Assistants (https://arxiv.org/abs/2407.16361)
Comments:
          Accepted for EUMAS24

- **What's New**: 이 논문은 기존의 로봇 윤리 프레임워크의 한계를 극복하기 위해 **덕 윤리** (Virtue Ethics) 에 기반한 새로운 계산적 접근 방식을 제시한다. 이 방법은 로봇의 **성격** (Character) 을 조정하여 특정 환경의 윤리적 요구 사항에 맞출 수 있도록 설계되었다.  특히, 로봇의 성격을 조정하여 환경에 맞는 윤리적 행동을 수행하도록 하는 새로운 방식을 소개하고, 노인 간병 환경 시뮬레이션을 통해 이러한 조정이 로봇의 행동에 미치는 영향을 분석한다.



### On The Expressive Power of Knowledge Graph Embedding Methods (https://arxiv.org/abs/2407.16326)
Comments:
          11 pages, 1 figure

- **What's New**: 본 논문은 지식 그래프 임베딩(KGE) 방법들의 추론 능력을 비교하기 위한 수학적 프레임워크를 제안합니다. 기존 KGE 방법들은 추론 능력에 한계가 있었으며, 본 논문은 STransE가 TransComplEx보다 더 높은 추론 능력을 가지고 있음을 보여줍니다. 또한, STransE의 공간 복잡성을 줄이면서 STransE를 개선한 새로운 STransCoRe 방법을 제시합니다.



### Efficient Detection of Commutative Factors in Factor Graphs (https://arxiv.org/abs/2407.16280)
Comments:
          Accepted to the Proceedings of the 12th Conference on Probabilistic Graphical Models (PGM 2024)

- **What's New**: 본 논문은 probabilistic graphical model에서 대칭성을 활용하여  domain size에 관계없이 효율적인 probabilistic inference를 가능하게 하는 lifted probabilistic inference 방법을 연구합니다. 특히, factor graph에서 대칭성을 찾기 위해 commutative factor를 효율적으로 찾는 새로운 알고리즘인 DECOR (Detection of Commutative Factors)를 제안합니다. 



### ODGR: Online Dynamic Goal Recognition (https://arxiv.org/abs/2407.16220)
Comments:
          8 pages, 1 figure, RLC workshop, WAHT workshop

- **What's New**: 본 논문은 기존의 강화 학습(Reinforcement Learning, RL) 방식에서 벗어나, 다른 에이전트의 정책을 학습하여 실시간으로 에이전트의 목표를 인식하는 새로운 접근 방식을 제안합니다. 기존의 목표 인식(Goal Recognition, GR)은 관찰된 행동을 기반으로 에이전트의 목표를 인식하는 계획 문제로 여겨졌습니다. 최근의 연구에서는 강화 학습을 GR 파이프라인에 사용하는 방법을 보여주었지만, 사전에 정의된 목표만 인식할 수 있고, 목표 공간이 큰 영역에서는 확장성이 부족했습니다. 본 논문에서는 이러한 제한 사항을 해결하기 위한 첫 단계로 '온라인 동적 목표 인식'(Online Dynamic Goal Recognition, ODGR) 문제를 새롭게 정의합니다. 기여 사항은 표준 GR 문제 정의에 동적 목표 개념을 도입하고, ODGR을 사용하여 기존 접근 방식을 재정의하고, 전이 학습을 사용하여 탐색 영역에서 ODGR을 해결할 수 있음을 보여주는 것입니다. 이러한 새로운 공식은 변화하고 확장되는 실시간 환경에 강력한 기존의 전이 학습 기반 GR 방법의 미래 확장을 위한 길을 열어줍니다.



### MCTS Based Dispatch of Autonomous Vehicles under Operational Constraints for Continuous Transportation (https://arxiv.org/abs/2407.16200)
Comments:
          International Conference on Automation Science and Engineering (CASE), 2024

- **What's New**: 이 논문은 기존의 자동 MCQ 생성 평가 지표의 단점을 보완하기 위해 새로운 지표인 지식 종속 가능성(KDA)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정하며, 학생 응답을 기반으로 한 실험 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 유용성과 높은 상관관계를 보입니다. 즉, MCQ의 교육적 가치를 측정하는 데 효과적인 새로운 방법을 제시합니다.



### Artificial Intelligence-based Decision Support Systems for Precision and Digital Health (https://arxiv.org/abs/2407.16062)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2203.02605

- **What's New**: 이 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 생성된 MCQ가 대상 사실에 대한 학생의 지식을 정확하게 평가하는지를 측정합니다.



### KAN or MLP: A Fairer Comparison (https://arxiv.org/abs/2407.16674)
Comments:
          Technical Report

- **What's New**: 본 논문은 기존의 MLP (Multi-Layer Perceptron) 모델의 대안으로 제시된 KAN (Kolmogorov–Arnold Networks) 모델을 여러 가지 작업에 걸쳐 더 공정하고 포괄적인 비교를 제공합니다. 특히, 논문에서는 매개변수 수와 FLOPs를 제어하여 KAN과 MLP의 성능을 비교합니다. 본 연구는 기호 공식 표현 작업을 제외하고는 MLP가 일반적으로 KAN보다 뛰어나다는 것을 발견했습니다. 또한, KAN에 대한 ablation study를 수행한 결과, 기호 공식 표현에서 KAN의 장점이 주로 B-spline activation function에서 비롯되었다는 것을 알아냈습니다. B-spline을 MLP에 적용하면 기호 공식 표현에서의 성능이 크게 향상되어 KAN의 성능을 능가하거나 일치합니다. 그러나 MLP가 이미 KAN보다 뛰어난 다른 작업에서 B-spline은 MLP의 성능을 크게 향상시키지 못했습니다. 또한, KAN의 forgetting issue가 일반적인 class-incremental continual learning 환경에서 MLP보다 심각하다는 것을 발견했는데, 이는 KAN 논문에서 보고된 결과와 다릅니다. 이러한 결과가 KAN 및 기타 MLP 대안에 대한 미래 연구에 대한 통찰력을 제공하기를 바랍니다.



### A Framework for Pupil Tracking with Event Cameras (https://arxiv.org/abs/2407.16665)
- **What's New**: 본 논문은 기존의 BLEU, ROUGE, METEOR 등의 평가 지표가 단순히 생성된 MCQ의 문장 유사성만을 고려하기 때문에 교육적 가치를 제대로 반영하지 못한다는 점을 지적하고, 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가하는 데 중점을 둡니다. 또한, KDA를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 해당 지표는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 과정을 모방함으로써 KDA를 근사화합니다. 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서의 사용성과 강한 상관관계를 보이는 것으로 확인되었습니다.



### A Geometry-Aware Algorithm to Learn Hierarchical Embeddings in Hyperbolic Spac (https://arxiv.org/abs/2407.16641)
- **What's New**: 이 논문은 기존의 BLEU, ROUGE, METEOR 등의 MCQ 평가 지표들이 단어 유사성에만 집중하여 교육적 가치를 제대로 평가하지 못한다는 한계를 지적하고, 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 이라는 새로운 평가 지표를 제안한다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하는 데 초점을 맞춘다. 이를 위해, 논문은 KDA를 Human Survey를 통해 측정하는 방법을 보여주고, 또한 KDA_disc와 KDA_cont 라는 두 가지 자동 평가 지표를 제안한다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사화한다. Human Studies 결과, KDA_disc와 KDA_cont는 Human Evaluation에 의한 KDA와 실제 강의실 설정에서의 사용성과 높은 상관관계를 보여주었고, n-gram 기반 유사성 지표와 결합했을 때 다양한 MCQ 품질 척도에 대한 예측력이 뛰어남을 보여주었다.



### Knowledge-driven AI-generated data for accurate and interpretable breast ultrasound diagnoses (https://arxiv.org/abs/2407.16634)
- **What's New**: 본 논문은 희귀 케이스의 정확도를 높이기 위해 장기 분포 데이터를 사용하는 진단 모델 성능 향상을 위한 새로운 파이프라인, TAILOR를 제안합니다. TAILOR는 지식 기반 생성 모델을 활용하여 희귀 케이스를 위한 맞춤형 합성 데이터를 생성합니다. 특히, 3,749개의 병변을 소스 데이터로 사용하여 수백만 개의 유방 초음파 이미지를 생성하며, 특히 오류 발생률이 높은 희귀 케이스에 대한 데이터 생성에 초점을 맞춥니다. 생성된 데이터는 정확하고 해석 가능한 진단을 위한 진단 모델을 구축하는 데 사용됩니다. 



### Implementing engrams from a machine learning perspective: the relevance of a latent spac (https://arxiv.org/abs/2407.16616)
Comments:
          6 pages, 2 figures

- **What's New**: 이 논문에서는 뇌에서의 engram이 recurrent neural network를 기반으로 하는 autoencoder로 구현될 수 있다는 가설을 제안하고, 이러한 autoencoder의 latent space의 중요성에 대해 조사합니다. 특히, autoencoder의 차원과 암호화되는 정보의 복잡성 사이의 관계를 살펴보고, 종들 간의 connectome 차이가 그들의 인지 능력과 어떻게 관련되는지 논의합니다. 마지막으로 인간 인지는 뇌 구조에 의해 제한될 가능성이 크지만, 이러한 제한은 기계 학습 시스템에는 적용되지 않는다는 점을 강조합니다.



### No-brainer: Morphological Computation driven Adaptive Behavior in Soft Robots (https://arxiv.org/abs/2407.16613)
Comments:
          Accepted to the From Animals to Animats: 17th International Conference on the Simulation of Adaptive Behavior (SAB 2024) conference

- **What's New**: 이 논문은 별도의 뇌가 없는 로봇의 형태 (morphology) 에서 지능적인 행동을 생성하는 방법을 연구합니다. 특히, 환경 자극에 따라 모양을 변화시키는 간단한 반응성 재료를 사용하여 복셀 기반 가상 소프트 로봇에서 적응력 있고 복잡한 행동을 생성할 수 있음을 보여줍니다. 이러한 접근 방식은 '폐쇄 루프 형태 계산 (closed-loop morphological computation)'의 개념을 증명합니다. 즉, 로봇의 몸체 내에서 발생하는 계산을 통해 별도의 뇌 없이 지능적인 행동을 구현하는 것입니다.



### Local vs Global continual learning (https://arxiv.org/abs/2407.16611)
Comments:
          (10 pages, Will appear in the proceedings of CoLLAs 2024)

- **What's New**: 이 논문은 기존의 continual learning 알고리즘들을 multi-task loss approximation (다중 작업 손실 근사) 관점에서 분석하고, local approximation과 global approximation 두 가지 전략을 비교 분석합니다. 또한, local polynomial approximation (지역 다항식 근사) 상황에서 최적의 continual learning 목표 함수를 연구하고, 최적 목표 함수를 구현하는 기존 알고리즘을 예시로 제시합니다.



### Deep Bayesian segmentation for colon polyps: Well-calibrated predictions in medical imaging (https://arxiv.org/abs/2407.16608)
Comments:
          comments are welcome. 43 pages

- **What's New**: 이 논문은 자동 MCQ 생성을 위한 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. 기존의 BLEU, ROUGE, METEOR 등은 MCQ의 교육적 가치를 고려하지 않고 단어 유사도만 비교했던 반면, KDA는 MCQ의 대답 가능성 (answerability)을 측정하여 학생의 지식을 평가하는 능력을 평가합니다. KDA는 인간 설문 조사를 통해 측정되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 학생들의 문제 해결 행동을 모방하여 자동으로 KDA를 측정합니다.



### Functional Acceleration for Policy Mirror Descen (https://arxiv.org/abs/2407.16602)
- **What's New**: 이 논문은 정책 거울 하강(PMD) 알고리즘의 일반적인 계열에 기능적 가속을 적용하여 강화 학습(RL)에서 다양한 새로운 기본 방법을 다룹니다. 듀얼을 활용하여 모멘텀 기반 PMD 업데이트를 제안합니다. 기능적 접근 방식을 통해 이 방법은 정책 매개변수화와 독립적이며 이전의 정책 매개변수 수준에서 모멘텀 적용을 특별한 경우로 포함합니다. 이 방법의 몇 가지 속성을 이론적으로 분석하고, 이 공간에서 다양한 알고리즘 설계 선택과 관련하여 값 다면체에서 정책 최적화 역학을 보여주는 숫자적 삭제 연구를 보완합니다. 또한 기능적 가속에 관련된 문제 설정의 몇 가지 기능을 숫자적으로 특징짓고, 마지막으로 근사의 학습 메커니즘에 미치는 영향을 조사합니다.



### A Faster Branching Algorithm for the Maximum $k$-Defective Clique Problem (https://arxiv.org/abs/2407.16588)
Comments:
          The accepted paper of confernece ECAI-2024 as well as the appendix

- **What's New**: 이 논문에서는 기존 알고리즘의 단점을 해결하고, 더 효율적인 **k-defective clique** 찾기 알고리즘을 제안합니다. 기존 알고리즘은 최대 크기의 **k-defective clique**를 찾기 위해 모든 가능한 조합을 따져보는 방식으로, 시간 복잡도가 높았습니다. 본 논문에서는 **k-defective clique**의 구조적 특성을 활용하여 **maximum clique** 알고리즘을 서브루틴으로 사용하는 새로운 분기 알고리즘을 제시합니다. 이를 통해 기존 알고리즘보다 더 나은 점근 시간 복잡도를 달성했습니다. 또한, **conflict relationship**을 활용한 새로운 상한 경계 기법을 제안하여 더욱 효율적인 검색을 가능하게 합니다. **Conflict relationship**은 그래프 문제에서 흔히 볼 수 있는 개념이기 때문에, 이 기법은 다른 그래프 문제에도 적용될 수 있을 것으로 기대됩니다.



### Audio Prompt Adapter: Unleashing Music Editing Abilities for Text-to-Music with Lightweight Finetuning (https://arxiv.org/abs/2407.16564)
Comments:
          Accepted by the 25th International Society for Music Information Retrieval (ISMIR)

- **What's New**: 본 논문에서는 기존 Text-to-Music 모델에 추가할 수 있는 Audio Prompt Adapter(AP-Adapter)를 제안합니다. AP-Adapter는 AudioMAE를 사용하여 입력 오디오에서 특징을 추출하고, 이를 AudioLDM2의 내부 레이어에 공급하기 위한 어텐션 기반 어댑터를 구성합니다. 이를 통해 사용자는 입력 오디오와 짧은 텍스트를 사용하여 음악의 전반적인 특징(예: 장르, 음색)과 세부적인 특징(예: 멜로디)을 제어할 수 있습니다.



### Patched RTC: evaluating LLMs for diverse software development tasks (https://arxiv.org/abs/2407.16557)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 다양한 소프트웨어 개발 작업(특히 버그 수정, 코드 검토, 문서 업데이트와 같은 '외부 루프' 작업)에 대한 새로운 평가 기술인 Patched Round-Trip Correctness (Patched RTC)를 소개합니다. Patched RTC는 원래 Round-Trip Correctness 방법을 확장하여 모든 LLM과 다운스트림 작업에 적용할 수 있으며, 인간의 개입 없이 모델 응답의 일관성과 강력성을 측정하는 자체 평가 프레임워크를 제공합니다. 이 연구는 Patched RTC 점수와 작업 특정 정확도 지표 간의 상관관계를 보여주며, 오픈 도메인 작업 평가에 대한 LLM-as-Judge 패러다임의 대안으로 제시합니다. Patched RTC는 patchwork라는 오픈 소스 프레임워크에 구현되어 다양한 패치 플로우에서 추론 중 투명한 평가를 가능하게 합니다. 다양한 소프트웨어 개발 작업에서 GPT-3.5 및 GPT-4 모델을 비교하는 실험을 통해 Patched RTC가 모델 성능과 작업 난이도를 효과적으로 구별한다는 사실을 밝혀냈습니다. 또한, 이 논문은 일관성 프롬프트가 모델 정확도 향상에 미치는 영향을 살펴보고 Patched RTC가 복잡한 소프트웨어 개발 워크플로우에 대한 프롬프트 개선 및 모델 선택을 안내할 수 있음을 시사합니다.



### Is 3D Convolution with 5D Tensors Really Necessary for Video Analysis? (https://arxiv.org/abs/2407.16514)
- **What's New**: 이 논문은 2D 및/또는 1D convolution을 사용하여 5D tensor가 아닌 4D 및/또는 3D tensor로 3D convolutional block을 구현하는 새로운 기술을 제안합니다. 3D convolution은 계산량이 많아 로봇과 같은 실시간 애플리케이션에서 사용되는 엣지 장치에서는 지원되지 않을 수 있습니다. 기존 방법은 3D 커널을 공간 영역과 시간 영역으로 분리하여 이 문제를 완화하지만 구현에는 여전히 5D tensor를 사용하는 3D convolution이 필요합니다. 본 논문은 4D/3D tensor 재구성과 공간 및 시간 분할을 위한 새로운 결합 기법을 도입하여 이 문제를 해결합니다.



### Articulation Work and Tinkering for Fairness in Machine Learning (https://arxiv.org/abs/2407.16496)
- **What's New**: 본 논문은 알고리즘 편향 문제 해결을 위한 새로운 접근 방식으로 사회적 관점과 학제적 (SOI) 관점을 강조하는 '공정한 AI' (fair AI) 연구의 긴장 관계를 분석한다.



### Learning General Continuous Constraint from Demonstrations via Positive-Unlabeled Learning (https://arxiv.org/abs/2407.16485)
- **What's New**: 이 논문은 기존의 제약 조건 추론 방법들의 한계를 극복하기 위해 PU learning 방식을 이용한 새로운 방법을 제안합니다. 특히, 기존 방법들이 선형 제약 조건이나 알려진 매개변수를 가진 비선형 제약 조건에만 집중했던 반면, 이 논문에서는 연속적이고 임의적이며 비선형적인 제약 조건을 추론할 수 있도록 합니다.



### BONES: a Benchmark fOr Neural Estimation of Shapley values (https://arxiv.org/abs/2407.16482)
Comments:
          6 pages

- **What's New**: 이 논문은 Shapley Value의 신경 추정을 위한 새로운 벤치마크인 BONES를 소개합니다. 이는 연구자들에게 최신 신경 및 기존 추정기 모음, 일반적으로 사용되는 벤치마크 데이터셋, 블랙박스 모델을 훈련하기 위한 특별한 모듈, 그리고 가장 인기 있는 평가 지표를 쉽게 계산하고 결과를 시각화하기 위한 특정 기능을 제공합니다.  BONES의 목표는 XAI 모델의 사용, 평가 및 비교를 간소화하는 것입니다.



### Side-Channel Analysis of OpenVINO-based Neural Network Models (https://arxiv.org/abs/2407.16467)
- **What's New**: 본 논문은 OpenVINO 프레임워크에서 구현된 양자화된 모델 (quantized models)의 취약성을 분석하고, SCA(Side-Channel Analysis) 공격을 통해 모델 매개변수 (model parameters)를 복구하는 것이 가능함을 보여줍니다. 특히, 임베디드 기기에서의 신경망 배포를 위한 임베디드 프레임워크인 OpenVINO에서 구현된 양자화된 모델에 대한 SCA 공격의 가능성을 조사합니다. 이 연구는 OpenVINO에서 구현된 양자화된 모델이 SCA 공격에 취약함을 보여주고, 고정밀도로 모델 매개변수를 복구할 수 있음을 입증합니다. 복구된 모델은 원래 모델과 매우 유사한 성능을 보이며, GoogleNet v1에 대한 실험 결과 Top 1 정확도는 1% 차이, Top 5 정확도는 0.64% 차이를 보였습니다.



### On ADMM in Heterogeneous Federated Learning: Personalization, Robustness, and Fairness (https://arxiv.org/abs/2407.16397)
Comments:
          arXiv admin note: text overlap with arXiv:2311.06756

- **What's New**: 본 논문에서는 FLAME이라는 새로운 PFL 프레임워크를 제안하여, ADMM(Alternating Direction Method of Multipliers)을 활용하여 개인화된 모델과 글로벌 모델을 동시에 학습하는 방식을 채택합니다. 이는 기존 PFL 프레임워크에서 개인화된 모델에만 집중하고 글로벌 모델을 무시했던 문제점을 해결합니다. FLAME은 ADMM을 활용하여 더 빠르고 정확하게 수렴하며,  다양한 종류의 이기종 데이터 (heterogeneous data)에 더 강력하고 공정하게 적용될 수 있다는 장점을 가지고 있습니다. 또한,  FLAME은 기존 PFL 및 FL 프레임워크를 일반화할 수 있으며,  이론적으로 수렴성과 공정성, 견고성을 분석하여,  기존 방법보다 더 견고하고 공정함을 보여줍니다.



### Ranking protein-protein models with large language models and graph neural networks (https://arxiv.org/abs/2407.16375)
Comments:
          14 pages. Detailed protocol to use our DeepRank-GNN-esm software to analyse models of protein-protein complexes

- **What's New**: 이 논문은 단백질-단백질 상호 작용(PPI) 모델링에서 좋은 모델(near-native PPI conformations)을 선별하기 위한 새로운 방법인 DeepRank-GNN-esm을 소개합니다. 이 방법은 그래프 기반 딥 러닝 알고리즘을 사용하며, 단백질 언어 모델의 강력한 기능을 활용하여 PPI 구조 모델들을 순위 매깁니다.



### SOAP: Enhancing Spatio-Temporal Relation and Motion Information Capturing for Few-Shot Action Recognition (https://arxiv.org/abs/2407.16344)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 본 논문은 High Frame-Rate (HFR) 비디오에서 액션 인식 (action recognition)을 위한 새로운 플러그 앤 플레이 아키텍처인 Spatio-tempOral frAme tuPle enhancer (SOAP)를 제안합니다. SOAP는 기존의 FSAR (Few-shot action recognition) 모델의 한계를 극복하고, HFR 비디오의 특징인 미세한 액션 표현과 낮은 공간-시간적 관계 및 모션 정보 밀도를 효과적으로 처리하는 것을 목표로 합니다. SOAP는 3D 컨볼루션을 사용하여 공간-시간적 관계를 구축하는 3DEM (3-Dimension Enhancement Module)과 채널별 특징 응답을 적응적으로 보정하는 CWEM (Channel-Wise Enhancement Module) 그리고 다양한 프레임 수의 프레임 튜플 (frame tuples)을 결합하여 광범위한 시각을 제공하는 HMEM (Hybrid Motion Enhancement Module)을 사용합니다. 이러한 모듈들을 결합하여 SOAP는 기존 방법들에 비해 공간-시간적 관계를 보다 효과적으로 구축하고, 다양한 프레임 튜플을 통해 포괄적인 모션 정보를 포착합니다.



### PhenoFlow: A Human-LLM Driven Visual Analytics System for Exploring Large and Complex Stroke Datasets (https://arxiv.org/abs/2407.16329)
Comments:
          11 pages, 5 figures, paper to appear in IEEE Transactions on Visualization and Computer Graphics (TVCG) (Proc. IEEE VIS 2024)

- **What's New**: 이 논문은 자동 MCQ 생성 시스템 평가를 위해 **지식 종속 가능성 (Knowledge Dependent Answerability, KDA)** 라는 새로운 메트릭을 제안합니다. 이는 MCQ가 대상 사실 (target fact)에 대한 학생의 지식을 실제로 평가할 수 있는지 여부를 측정하는 데 중점을 둡니다. 기존 메트릭 (BLEU, ROUGE, METEOR)은 MCQ가 골드 샘플과 얼마나 유사한지에만 초점을 맞췄지만, KDA는 MCQ가 학생의 지식을 측정하는 데 얼마나 효과적인지 측정합니다. KDA를 자동으로 측정하기 위해 **KDA_disc**와 **KDA_cont**라는 두 가지 새로운 메트릭이 제안되었으며, 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 연구 결과, KDA_disc와 KDA_cont는 실제 강의실에서의 MCQ 사용성과 높은 상관관계를 보였으며, 기존 n-gram 기반 유사성 메트릭과 함께 사용하면 MCQ 품질을 더 잘 예측할 수 있음을 보여주었습니다.



### MOMAland: A Set of Benchmarks for Multi-Objective Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2407.16312)
- **What's New**: 본 논문은 다중 목표 다중 에이전트 강화 학습 (MOMARL, Multi-Objective Multi-Agent Reinforcement Learning)을 위한 표준 환경 모음인 "MOMAland"를 소개합니다. MOMARL은 다양한 독립적인 의사 결정자 (DM, Decision-Makers)들의 행동을 조정하고 여러 상충되는 목표를 균형 있게 유지해야 하는 복잡한 의사 결정 과정이 필요한 교통 시스템, 전력망, 공급망과 같은 어려운 작업을 공식화하고 해결하기 위한 관점입니다. MOMAland는 다양한 환경을 제공하여 MOMARL 분야의 발전을 위한 포괄적인 벤치마킹을 제공합니다. 이 환경들은 에이전트의 수, 상태 표현, 보상 구조 및 유틸리티 고려 사항이 다릅니다.  



### Quantum Computing for Climate Resilience and Sustainability Challenges (https://arxiv.org/abs/2407.16296)
- **What's New**: 이 논문은 기후 변화 예측 및 지속 가능한 개발을 위한 양자 컴퓨팅 (QC) 의 응용에 대해 논의하며, 기존 컴퓨터의 한계를 극복하기 위해 QC가 제공하는 강력한 컴퓨팅 능력을 강조한다. 또한 양자 머신 러닝 (QML) 과 최적화 기술이 기후 변화 예측 및 지속 가능한 개발에 어떻게 활용될 수 있는지 탐구한다.



### Visual Stereotypes of Autism Spectrum in DALL-E, Stable Diffusion, SDXL, and Midjourney (https://arxiv.org/abs/2407.16292)
- **What's New**: 본 연구는 텍스트-이미지 모델이 훈련 데이터의 내재된 편견으로 인해 고정관념을 확산시키는 잠재력을 조사했습니다. 특히, 자폐증과 관련된 구체적인 객체와 추상적인 개념을 시각화하는 53개의 프롬프트를 기반으로 이미지를 생성하는 방식으로 자폐증에 대한 비합리적인 믿음을 어떻게 무의식적으로 영속시키는지 조사했습니다. 연구 프로토콜은 DALL-E, Stable Diffusion, SDXL, Midjourney의 네 가지 모델(N=249)에 걸쳐 자폐증과 관련된 구체적인 객체와 추상적인 개념을 시각화하는 53개의 프롬프트를 기반으로 이미지를 생성하는 것을 포함했습니다. 결과에 대한 전문가 평가는 자폐증 커뮤니티에서 논란이 되는 일반적인 고정관념의 존재와 공간적 강도를 나타내는 10개의 연역적 코드 프레임워크를 통해 수행되었으며, 순서척 척도로 정량화되었고 평가자 간 신뢰도와 효과 크기의 통계적 분석을 거쳤습니다. 모델은 자주 논란이 되는 주제와 상징을 사용했으며, 이는 고르게 분포되지 않았지만, 피부색, 성별, 나이 측면에서 놀라울 정도로 동질적이었습니다. 자폐증을 가진 개인은 고립된 활동에 참여하고, 사람이 아닌 사물과 상호 작용하고, 창백함, 분노 또는 슬픔과 같은 전형적인 감정 표현을 보여주는 것으로 묘사되었습니다. 둘째, 우리는 위 결과를 반증하기 위한 방향성 있는 프롬프트에도 불구하고 자폐증 이미지에 대한 표현적 무감각성을 관찰했습니다. 또한 DALL-E는 고정관념을 영속시키는 것을 명시적으로 부인했습니다. 우리는 이를 ANN이 사람의 인지 아키텍처를 반영하는 것으로 해석하며, 이는 사람의 자폐증 관련 고정관념에 대한 이전 연구에 의해 뒷받침됩니다.



### Federated Learning for Face Recognition via Intra-subject Self-supervised Learning (https://arxiv.org/abs/2407.16289)
Comments:
          Accepted at the The 35th British Machine Vision Conference 2024 (BMVC 2024), Glasgow, UK. Youngjun Kwak is corresponding author

- **What's New**: 본 논문은 기존의 Federated Learning(FL) 기반 얼굴 인식 모델의 한계를 극복하기 위해 개인화된 얼굴 인식 모델을 훈련하는 새로운 FedFS(Federated Learning for personalized Face recognition via intra-subject Self-supervised learning framework) 아키텍처를 제안합니다. FedFS는 개인 데이터 유출 없이 사용자 기기에서 개인화된 얼굴 인식 모델을 훈련할 수 있습니다.



### A deeper look at depth pruning of LLMs (https://arxiv.org/abs/2407.16286)
- **What's New**: 본 논문은 Large Language Model (LLM) 의 효율적인 배포를 위한 새로운 블록 중요도 측정 방법과 블록 삭제 (pruning) 및 성능 회복 기법을 제안합니다. 기존의 정적 (static) 블록 중요도 측정 방법 대신, 샤플리 값 (Shapley value) 과 같은 적응적 (adaptive) 측정 방법을 사용하여 블록의 중요도를 더 정확하게 평가합니다. 또한, 개별 self-attention 과 feed-forward 레이어에 대한 분석을 통해 self-attention 레이어가 feed-forward 레이어보다 pruning에 더 강인함을 보여줍니다. 마지막으로, 블록 삭제로 인한 성능 저하를 최소화하기 위해, 가볍게 학습된 additive bias (emulated update) 나 저랭크 선형 어댑터 (low-rank linear adapter) 와 같은 간단한 성능 회복 기법을 제안합니다.



### Comparative Analysis of AES, Blowfish, Twofish, Salsa20, and ChaCha20 for Image Encryption (https://arxiv.org/abs/2407.16274)
- **What's New**: 이 연구는 인터넷 보안을 위한 암호화 알고리즘 (encryption algorithms)의 속도와 효율성을 비교 분석했습니다. 특히 AES, Blowfish, Twofish, Salsa20, ChaCha20 등의 알고리즘을 이용하여 이미지 데이터를 암호화/복호화 (encryption/decryption) 하는 데 걸리는 시간과 처리량 (throughput)을 비교했습니다.



### Self-Reasoning Assistant Learning for non-Abelian Gauge Fields Design (https://arxiv.org/abs/2407.16255)
- **What's New**: 이 논문에서는 비아벨 게이지 필드(Non-Abelian gauge field)를 직접 생성할 수 있는 자기 추론(self-reasoning) 보조 학습 프레임워크를 제안합니다. 이 프레임워크는 전방 확산(forward diffusion) 과정을 사용하여 지속적인 변환을 통해 대상 분포에 내재된 복잡한 패턴과 세부 사항을 캡처하고 재현합니다. 그런 다음 역방향 확산(reverse diffusion) 과정을 사용하여 생성된 데이터를 원래 상황의 분포에 더 가깝게 만듭니다. 이를 통해 강력한 자기 추론 능력을 갖추게 되어 데이터셋에서 특징 표현을 자동으로 발견하고 더 미묘한 관계를 포착할 수 있습니다. 게다가 자기 추론은 수동 특징 엔지니어링이 필요 없으며 모델 구축 프로세스를 간소화합니다. 이 프레임워크는 복잡한 물리적 프로세스를 파싱하고 대규모 데이터셋에서 패턴을 자동으로 발견하는 혁신적인 패러다임 변화를 제공합니다.



### HSVLT: Hierarchical Scale-Aware Vision-Language Transformer for Multi-Label Image Classification (https://arxiv.org/abs/2407.16244)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 다중 레이블 이미지 분류 (multi-label image classification)를 위한 새로운 방법인 HSVLT (Hierarchical Scale-Aware Vision-Language Transformer) 를 제안한다. 이 방법은 다양한 크기와 외형의 객체를 인식하기 위해 계층적 다중 스케일 아키텍처와 상호 작용하는 시각 언어 어텐션 (Interactive Visual-Linguistic Attention, IVLA) 을 사용한다. 또한, 다중 스케일 정보를 통합하는 크로스 스케일 집계 (Cross-Scale Aggregation, CSA) 모듈을 제안한다.



### OriGen:Enhancing RTL Code Generation with Code-to-Code Augmentation and Self-Reflection (https://arxiv.org/abs/2407.16237)
- **What's New**: OriGen, a fully open-source framework for RTL code generation with self-reflection capabilities and a novel code-to-code dataset augmentation methodology, is introduced. This framework utilizes knowledge distillation to enhance the quality of open-source RTL code datasets and incorporates a self-reflection process to correct syntactic errors by leveraging compiler feedback.



### Comparison of Static Application Security Testing Tools and Large Language Models for Repo-level Vulnerability Detection (https://arxiv.org/abs/2407.16235)
- **What's New**: 이 논문에서는 SAST 툴과 대규모 언어 모델(LLM)을 비교하여 소프트웨어 취약점을 자동으로 감지하는 방법을 제안합니다.  특히, 기존의 함수 단위(function-level) 취약점 감지에서 벗어나 전체 저장소(repo-level) 단위의 취약점 감지를 수행합니다.  또한, SAST 툴과 LLM의 장단점을 분석하고, 두 가지 접근 방식을 결합하여 더 효과적인 취약점 감지 시스템을 구축하는 가능성을 탐구합니다.



### Strategy and Skill Learning for Physics-based Table Tennis Animation (https://arxiv.org/abs/2407.16210)
Comments:
          SIGGRAPH 2024

- **What's New**: 이 논문은 자동 MCQ 생성의 교육적 가치를 측정하기 위한 새로운 평가 지표인 '지식 종속 가능성(KDA)'을 제안합니다. KDA는 MCQ가 학생의 지식을 얼마나 잘 평가할 수 있는지 측정하며, 기존의 단어 기반 유사도 측정 지표(BLEU, ROUGE, METEOR)의 단점을 보완합니다. 또한, 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안하여 KDA를 근사화합니다. 이러한 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 방식을 모방합니다.



### INF-LLaVA: Dual-perspective Perception for High-Resolution Multimodal Large Language Mod (https://arxiv.org/abs/2407.16198)
- **What's New**: 이 논문은 기존 MCQ 생성 평가 메트릭의 한계를 극복하고 교육적 가치를 고려한 새로운 자동 평가 메트릭, '지식 종속 가능성(KDA)'을 제안합니다. KDA는 학생의 지식 수준을 평가하는 MCQ의 능력을 측정하며, 실제 학생 응답을 기반으로 측정됩니다. 이 논문은 KDA를 근사화하기 위해 사전 학습된 언어 모델을 이용한 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA 및 전문가에 의한 실제 강의실 설정에서의 사용성과 높은 상관관계를 보인다는 것을 입증합니다. 또한, 이 메트릭은 기존 n-gram 기반 유사성 메트릭과 함께 사용하면 전문가가 평가한 다양한 MCQ 품질 측정 지표를 예측하는 능력이 뛰어납니다.



### Automatic Environment Shaping is the Next Frontier in RL (https://arxiv.org/abs/2407.16186)
Comments:
          ICML 2024 Position Track; Website at this https URL

- **What's New**: 이 논문은 자동 MCQ 생성 평가 메트릭으로 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해 인간 설문 조사를 통해 KDA를 측정하고, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다.



### Pixel Embedding: Fully Quantized Convolutional Neural Network with Differentiable Lookup Tab (https://arxiv.org/abs/2407.16174)
- **What's New**: 이 논문은 입력 데이터를 벡터로 변환하는 Pixel Embedding이라는 새로운 기법을 제안하여 첫 번째 컨볼루션 계층에서의 양자화(quantization) 손실을 줄이는 방법을 소개합니다. 이는 자연어 처리 분야의 단어 임베딩(word embedding)에서 영감을 받았습니다. Pixel Embedding은 입력 픽셀을 룩업 테이블(lookup table)을 사용하여 양자화된 값들의 벡터로 대체합니다. 이 룩업 테이블은 미분 가능하며 역전파(backpropagation)를 통해 학습될 수 있습니다.



### Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality (https://arxiv.org/abs/2407.16171)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 연구에서는 **단일 모달이 누락된 경우에도 강력한 성능을 유지하는 새로운 AVQA 프레임워크**를 제안합니다. 기존 AVQA 모델은 오디오 또는 비주얼 정보가 누락되면 성능이 크게 저하되는 문제가 있었는데, 이 연구에서는 **관계 인식 모달 생성기(RMM)**와 **오디오-비주얼 관계 인식 확산 모델(AVR)**을 활용하여 누락된 모달 정보를 재구성하고 정확한 답변을 제공하는 방식을 제시합니다. 이를 통해 실제 환경에서 발생할 수 있는 모달 누락 문제에 대한 해결책을 제시합니다.



### Representation Magnitude has a Liability to Privacy Vulnerability (https://arxiv.org/abs/2407.16164)
Comments:
          Accepted in the AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2024

- **What's New**: 본 논문은 기존의 MCQ 생성 평가 지표인 BLEU, ROUGE, METEOR가 MCQ의 교육적 가치를 고려하지 않고, 단순히 생성된 MCQ와 기존 샘플의 유사성을 비교한다는 문제점을 지적합니다.  따라서, 본 논문은 지식 종속 가능성 (Knowledge Dependent Answerability, KDA) 이라는 새로운 자동 평가 지표를 제안하여, MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. KDA는 실제 학생들의 응답 데이터를 기반으로 측정될 수 있으며, 논문에서는 이를 근사화하기 위한 두 가지 자동 평가 지표인 KDA_disc와 KDA_cont를 제안합니다. KDA_disc와 KDA_cont는 사전 학습된 언어 모델 (pre-trained language model)을 사용하여 학생들의 문제 해결 방식을 모방하는 방식으로 설계되었습니다.  인간 평가를 통해 KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성 간의 높은 상관관계를 보여주었으며, 기존의 n-gram 기반 유사성 지표와 결합했을 때 전문가가 평가한 다양한 MCQ 품질 지표를 예측하는 능력이 우수한 것으로 나타났습니다.



### Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis (https://arxiv.org/abs/2407.16150)
Comments:
          10 pages, 6 figures, 2 tables, 2024 8th International Conference on Cloud and Big Data Computing

- **What's New**: 이 논문은 주식 시장의 동향 예측을 위해 뉴스 데이터와 주식 가격 정보를 결합한 새로운 딥 러닝 모델 FinBERT-LSTM을 제안합니다. 기존 모델들과 달리 FinBERT-LSTM은 금융 뉴스를 분석하는 전문적인 FinBERT 모델과 주식 시장 구조 계층 (market, industry, stock 관련 뉴스 카테고리) 정보를 통합하여 예측 정확도를 높입니다.



### Diffusion Models as Optimizers for Efficient Planning in Offline RL (https://arxiv.org/abs/2407.16142)
Comments:
          The paper was accepted by ECCV2024

- **What's New**: 이 논문은 MCQ 생성의 교육적 가치를 평가하는 새로운 지표인 지식 종속 가능성(KDA)을 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표들은 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 고려하지 않습니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 평가 능력을 평가하는 데 도움을 줄 수 있습니다. 본 연구에서는 Human Evaluation을 통해 KDA_disc와 KDA_cont가 실제 교육 환경에서 사용성과 높은 상관관계를 가지고 있음을 확인했습니다. 또한, n-gram 기반 유사도 지표와 결합하여 여러 전문가가 평가한 MCQ 품질 지표를 예측하는 데 효과적임을 보여줍니다.



### FoRA: Low-Rank Adaptation Model beyond Multimodal Siamese Network (https://arxiv.org/abs/2407.16129)
- **What's New**: 이 논문에서는 기존 MCQ 평가 지표의 단점을 해결하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 지표를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지를 측정하는 지표입니다.  KDA는 학생 설문 조사를 통해 측정되며,  KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 방식을 모방하여 자동으로 계산됩니다. 

- **Technical Details**: KDA는 학생 설문 조사를 통해 측정되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 방식을 모방하여 자동으로 계산됩니다.  KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보여줍니다. 

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 n-gram 기반 유사도 지표와 결합하여 전문가가 평가한 다양한 MCQ 품질 지표에 대한 예측력을 높여줍니다. 



### Advancing Brain Imaging Analysis Step-by-step via Progressive Self-paced Learning (https://arxiv.org/abs/2407.16128)
Comments:
          miccai-2024

- **What's New**: 이 논문은 Brain imaging 분석을 위한 새로운 Curriculum Learning 프레임워크인 Progressive Self-Paced Distillation (PSPD)를 제안합니다. PSPD는 모델의 과거 상태와 현재 상태를 모두 활용하여 적응적이고 점진적인 학습 과정을 통해 generalization 능력을 향상시키고 기존 지식을 잊는 것을 방지합니다.



### Towards Effective Fusion and Forecasting of Multimodal Spatio-temporal Data for Smart Mobility (https://arxiv.org/abs/2407.16123)
Comments:
          4 pages

- **What's New**: 이 논문은 자동 MCQ 생성 평가를 위한 새로운 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 지표를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식에 대한 평가 능력을 평가하는 것을 목표로 합니다.

- **Technical Details**: KDA는 Human evaluation을 기반으로 측정됩니다. 즉, 학생들이 MCQ에 대한 답변을 분석하여 KDA를 계산합니다. 또한 KDA_disc 및 KDA_cont라는 두 가지 자동 지표를 제안하여 KDA를 근사화합니다. 이 지표들은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 KDA와 강한 상관관계를 보이는 것으로 나타났습니다. 또한 실제 강의실 환경에서 전문가에 의해 평가된 사용성과도 높은 상관관계가 확인되었습니다. KDA_disc와 KDA_cont는 n-gram 기반 유사성 지표와 함께 사용될 때 전문가에 의해 평가된 다양한 MCQ 품질 지표에 대한 예측력이 강력한 것으로 나타났습니다.



### Uncertainty-Aware Deep Neural Representations for Visual Analysis of Vector Field Data (https://arxiv.org/abs/2407.16119)
Comments:
          Accepted for publication at IEEE Visualization 2024

- **What's New**: 이 논문은 벡터 필드 데이터를 효과적으로 모델링하기 위해 불확실성 인식 (uncertainty-aware) 암묵적 신경 표현 (implicit neural representations)을 개발했습니다. 또한 두 가지 핵심적인 딥 불확실성 추정 기법인 (1) 딥 앙상블 (Deep Ensemble)과 (2) 몬테 카를로 드롭아웃 (Monte Carlo Dropout)의 효과를 포괄적으로 평가하여 불확실성에 대한 시각적 분석을 가능하게 했습니다.



### Transformer-based Graph Neural Networks for Battery Range Prediction in AIoT Battery-Swap Services (https://arxiv.org/abs/2407.16115)
Comments:
          9pages, 6figures, accepted by IEEE ICWS 2024 The International Conference on Web Services

- **What's New**: 이 논문에서는 SEB(Sharing E-Bike Battery)의 잔여 배터리 범위를 예측하기 위해 SEB-Transformer라는 새로운 구조적 Transformer 기반 모델을 제안합니다. SEB 시나리오는 사용자와 자전거 간의 상호 작용을 포괄적으로 나타내는 동적 이종 그래프로 구성됩니다. 또한 그래프 구조를 SEB-Transformer에 통합하여 평균 구조적 유사성과 함께 e-bike 배터리 잔여 범위 추정을 용이하게 하여 예측 정확도를 향상시킵니다. 모델의 예측을 활용하여 충전소의 전략적 위치를 고려하면서 실시간으로 사용자의 최적 주행 경로를 동적으로 조정하여 사용자 경험을 최적화합니다.



### Faster Optimal Coalition Structure Generation via Offline Coalition Selection and Graph-Based Search (https://arxiv.org/abs/2407.16092)
- **What's New**: 본 논문은 다중 에이전트 시스템에서의 연합 형성 (coalition formation) 문제에 대한 새로운 알고리즘 SMART를 제안합니다. SMART는 세 가지 혁신적인 기법의 결합으로 만들어졌습니다. 첫째, 두 가지 기법은 동적 프로그래밍 (dynamic programming) 을 기반으로 하며, 평가를 위해 선택된 연합과 알고리즘 성능 사이의 강력한 연결성을 보여줍니다. 이 알고리즘은 오프라인 단계를 사용하여 평가할 연합을 최적화합니다. 둘째, 브랜치 앤 바운드 (branch-and-bound) 와 정수 분할 그래프 검색 (integer partition graph search) 을 사용하여 솔루션 공간을 탐색합니다.  이러한 기법들은 새로운 방식으로 문제에 접근하고, 새로운 수준의 정확성을 제공합니다.  



### Modelling brain connectomes networks: Solv is a worthy competitor to hyperbolic geometry! (https://arxiv.org/abs/2407.16077)
Comments:
          Full version of our paper accepted to ECAI 2024

- **What's New**: 이 논문은 MCQ 생성 평가에 새로운 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 답변 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가하여 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사도 측정 방식의 한계를 극복합니다.



### LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies (https://arxiv.org/abs/2407.16067)
Comments:
          ICML 2024 Oral Presentation; Project Page: this https URL

- **What's New**: 본 논문은 OOD (Out-of-Distribution) 데이터를 사용하지 않고 ID (In-Distribution) 측정을 통해 모델의 OOD 성능을 예측하는 새로운 방법을 제안합니다. 기존의 "Effective Robustness" 평가 방식은 다양한 supervision 및 distribution (예: 이미지넷의 Vision Model (VM), LAION의 Visual-Language Model (VLM)의 클래스 레이블과 텍스트 설명)으로 학습된 모델에 한계를 보였습니다. 특히 VLM은 VM과 비슷하거나 낮은 ID 성능에도 불구하고 OOD 데이터에 대해 더 잘 일반화됩니다. 본 논문에서는 이러한 문제를 해결하기 위해 LCA-on-the-Line 프레임워크를 도입하여 ID 측정으로부터 모델의 OOD 성능 예측을 개선합니다. 이 방법은 WordNet과 같은 사전 정의된 클래스 계층 구조 내에서 레이블과 예측 간의 계층적 거리를 측정하는 LCA (Lowest Common Ancestor) 거리 개념을 재해석합니다.



### Generalizing Teacher Networks for Effective Knowledge Distillation Across Student Architectures (https://arxiv.org/abs/2407.16040)
Comments:
          Accepted by the BMVC-24

- **What's New**: 이 논문에서는, 다양한 student 모델들로 구성된 pool에서, 어떤 student 모델이 주어지더라도 효과적인 지식 전달이 가능한 generic teacher model을 학습하는 새로운 방법론인 GTN (Generic Teacher Network)을 제안합니다. 이를 통해, teacher model을 다양한 student model에 맞게 여러 번 재학습시키는 비용을 줄이고, 다양한 하드웨어 제약 조건에 맞는 다양한 student model에 적용 가능한 generic teacher model을 만들 수 있습니다.



### KWT-Tiny: RISC-V Accelerated, Embedded Keyword Spotting Transformer (https://arxiv.org/abs/2407.16026)
Comments:
          6 pages, 7 figures, accepted to be published in the IEEE SOCC 2024 conference

- **What's New**: 본 논문은 Edge 장치에서 Transformer 기반 모델의 적용을 연구하며, ARM Keyword Transformer (KWT) 모델을 RISC-V 플랫폼에서 양자화 및 하드웨어 가속화를 통해 구현했습니다. KWT-1 모델은 369배 더 작아졌고, 출력 클래스를 35에서 2로 줄였지만 정확도는 10%만 손실되었습니다. 재학습 및 양자화를 통해 모델 크기는 2.42MB에서 1.65KB로 감소했습니다. GELU 및 SoftMax 연산을 가속화하는 RISC-V 명령어를 추가하여 추론 속도가 5배 빨라졌으며, 이를 통해 전력 소비가 약 5배 감소했습니다. 추론에 필요한 클록 사이클 수는 2,600만에서 550만으로 감소했으며, 약 29%의 작은 면적 오버헤드가 발생했습니다. 이 결과는 저전력 IoT 장치에서 Transformer 기반 모델을 포팅하고 가속화하는 실행 가능한 방법을 보여줍니다.



### Exploring and Addressing Reward Confusion in Offline Preference Learning (https://arxiv.org/abs/2407.16025)
- **What's New**: This paper proposes a novel automatic evaluation metric, Knowledge Dependent Answerability (KDA), for evaluating Multiple Choice Question (MCQ) generation. KDA measures the MCQ's answerability given knowledge of the target fact, focusing on the educational value rather than just n-gram based similarity. Two automatic evaluation metrics, KDA_disc and KDA_cont, are introduced to approximate KDA by leveraging pre-trained language models to simulate student problem-solving behavior. 

- **Technical Details**: KDA is calculated based on student responses from a human survey, while KDA_disc and KDA_cont approximate KDA using pre-trained language models. These metrics are designed to assess the MCQ's ability to evaluate students' understanding of the target fact.

- **Performance Highlights**: Human studies demonstrate that KDA_disc and KDA_cont have strong correlations with both KDA and usability in an actual classroom setting. Combining these metrics with n-gram based similarity metrics shows strong predictive power for various expert-labeled MCQ quality measures.



### AIDE: Antithetical, Intent-based, and Diverse Example-Based Explanations (https://arxiv.org/abs/2407.16010)
- **What's New**: 이 논문은 블랙박스 모델의 예측을 설명하기 위해 'AIDE'라는 새로운 방법을 제안합니다. AIDE는 사용자의 의도에 따라 다양한 각도에서 모델의 추론을 보여주는 대조적인(contrastive) 설명을 제공합니다. AIDE는 사용자가 예측을 해석하거나, 잘못된 예측을 조사하거나, 모호한 예측을 명확히 하려는 세 가지 의도를 구분합니다.

- **Technical Details**: AIDE는 각 의도에 따라 예측을 직접적으로 또는 대조적으로 뒷받침하거나 반박하는 적절한 훈련 샘플을 선택합니다. AIDE는 훈련 데이터의 중복을 피하고 범위를 넓히기 위해 다양성을 고려한 샘플링을 사용하여 간결한 요약을 제공합니다.

- **Performance Highlights**: AIDE는 이미지 및 텍스트 분류 작업에서 정확성 및 연속성, AIDE와 다른 예제 기반 방법의 일화적 증거 비교, 사용자 연구를 통해 AIDE의 다양한 측면을 평가하는 세 가지 방법으로 효과를 보여줍니다. 결과는 AIDE가 기존 방법의 한계를 해결하고 설명 가능성 방법에 대한 바람직한 특징을 보여준다는 것을 보여줍니다.



### AI for Handball: predicting and explaining the 2024 Olympic Games tournament with Deep Learning and Large Language Models (https://arxiv.org/abs/2407.15987)
- **What's New**: 본 논문에서는 2024년 파리 올림픽 핸드볼 토너먼트의 결과를 예측하기 위해 딥 러닝 모델을 활용합니다. 이 모델은 XAI 기술을 사용하여 각 경기 결과에 영향을 미치는 주요 요소를 분석하고, 경기 정보, 선수 성적 등의 요소가 예측에 어떻게 기여하는지 이해하는 데 도움을 줄 수 있습니다. 또한, LLM을 통합하여 경기 결과에 영향을 미치는 가장 중요한 요소를 강조하는 인간 친화적인 설명을 생성합니다. 이러한 인간 중심적인 설명을 제공함으로써 AI 예측에 대한 더 깊은 이해를 제공하여, 코치와 분석가가 더 효과적으로 활용할 수 있도록 합니다.



### A Survey of Explainable Artificial Intelligence (XAI) in Financial Time Series Forecasting (https://arxiv.org/abs/2407.15909)
Comments:
          35 pages, This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in a journal soon

- **What's New**: 본 논문은 금융 시계열을 예측하는 XAI 접근 방식을 분류하여 XAI가 금융 분야에서 어떻게 사용되는지 포괄적으로 조사했습니다. XAI 분야는 AI 모델을 더 이해하기 쉽게 만들기 위해 노력합니다.



### An Ad-hoc graph node vector embedding algorithm for general knowledge graphs using Kinetica-Graph (https://arxiv.org/abs/2407.15906)
Comments:
          11 pages, 16 figures, 16 references

- **What's New**: 본 논문은 일반 지식 그래프 (knowledge graph) 표현에서 일반 그래프 노드 임베딩 (graph node embedding)을 생성하는 방법을 제안합니다. 임베딩 공간 (embedded space)은 로컬 친화성 (local affinity)과 원격 구조적 관련성 (remote structural relevance)을 모방하기 위해 여러 개의 서브 피처 (sub-feature)로 구성됩니다. 이러한 서브 피처 차원은 홉 기반 토폴로지 패턴 (hop-based topological patterns), 중첩 레이블 수 (number of overlapping labels), 전이 확률 (transitional probabilities), 그리고 재귀 스펙트럼 이분 (Recursive Spectral Bisection, RSB) 알고리즘에 의해 계산된 클러스터 인덱스 (cluster indices)와 같은 노드 유사성을 포착할 수 있다고 가정되는 몇 가지 지표 (indicators)에 의해 정의됩니다. 이러한 지표는 유사한 노드를 찾기 위해 전체 벡터 유사성 함수 세트 (set of vector similarity functions)를 사용할 수 있도록 1차원 벡터 공간에 해당 서브 구성 요소 범위 (sub-component ranges)로 평평하게 만들어집니다. 오류 (error)는 임베딩된 것과 기본 진실 추정치 (ground truth estimates) 사이의 그래프 노드의 무작위로 선택된 샘플에서 쌍별 제곱 차이 (pairwise square differences)의 합으로 정의되며, 이것은 우리의 새로운 손실 함수 (loss function)입니다. 기본 진실은 쌍별 자카드 유사성 (pairwise Jaccard similarity)과 중첩 레이블 수 (number of overlapping labels)의 조합으로 추정됩니다. 마지막으로, 우리는 무작위 샘플링 논리 (random sampling logic)를 사용하여 평균 오류 (average error)를 최소화하기 위해 서브 벡터 공간 간의 가중치 요소 (weighing factors)를 계산하는 다변량 확률적 경사 하강 (Stochastic Gradient Descent, SGD) 알고리즘을 보여줍니다.



### Enhancing Cognitive Workload Classification Using Integrated LSTM Layers and CNNs for fNIRS Data Analysis (https://arxiv.org/abs/2407.15901)
Comments:
          conference

- **What's New**: 본 논문은 fNIRS 데이터 분석에 LSTM 레이어를 통합하여 CNN 모델의 성능을 향상시키는 방법을 제시한다. 기존의 CNN 모델은 공간적 특징에 대한 과적합 (overfitting) 문제와 시간 의존성 부족 문제를 가지고 있었다. LSTM 레이어를 통합함으로써, 모델은 fNIRS 데이터에서 시간적 의존성을 포착하여 더욱 포괄적인 인지 상태를 이해할 수 있다.



### Spatial-Temporal Cross-View Contrastive Pre-training for Check-in Sequence Representation Learning (https://arxiv.org/abs/2407.15899)
Comments:
          This paper has been accepted as a regular paper at IEEE TKDE

- **What's New**: 본 논문은 체크인 시퀀스 표현 학습을 위한 새로운 공간-시간 크로스뷰 대조 표현(STCCR) 프레임워크를 제안합니다. STCCR은 "공간 토픽" 및 "시간적 의도" 뷰에서 자기 지도 학습을 활용하여 공간 및 시간 정보를 의미 수준에서 효과적으로 융합합니다. 또한 STCCR은 대조적 클러스터링을 활용하여 다양한 이동 활동에서 사용자의 공유된 공간 토픽을 밝혀내는 동시에 각도 모멘텀 대조를 활용하여 시간적 불확실성과 노이즈의 영향을 완화합니다.



### Cascaded two-stage feature clustering and selection via separability and consistency in fuzzy decision systems (https://arxiv.org/abs/2407.15893)
Comments:
          This paper has been accepted by IEEE Transactions on Fuzzy Systems for publication. Permission from IEEE must be obtained for all other uses, in any current or future media. The final version is available at [https://doi.org/10.1109/TFUZZ.2024.3420963]

- **What's New**: 이 논문은 퍼지 의사 결정 시스템에서 특징 클러스터링 및 선택을 위한 계단식 2단계 알고리즘을 제안한다. 특징의 중복성을 해결하고 검색 공간을 줄이기 위해 특징을 그룹으로 클러스터링한다. 이 알고리즘은 클러스터링 기반 순차적 순방향 선택 알고리즘으로 특징 그룹에서 특징을 선택하고, 관련 없는 특징 간 상호작용을 고려한다. 또한 퍼지 의사 결정 시스템에서 특징의 중요성을 평가하기 위한 융합 지표를 제안한다. 이 지표는 전역 분리성과 지역 일관성을 평가하여 특징의 중요성을 평가하는데, 전역 분리성은 퍼지 멤버십에 기반한 클래스 내 응집력과 클래스 간 분리를 평가하고, 지역 일관성은 퍼지 근접 거칠기 집합 모델을 사용하여 데이터의 불확실성을 포착한다.



### CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models (https://arxiv.org/abs/2407.15886)
Comments:
          10 pages, 9 figures, 4 tables

- **What's New**: 기존의 Virtual Try-on (VTON) 방법들은 ReferenceNet 또는 추가적인 이미지 인코더를 사용하여 이미지 처리를 수행하여 높은 트레이닝 및 추론 비용을 발생시켰다. 본 논문에서는 ReferenceNet과 이미지 인코더의 필요성을 재고하고, CatVTON이라는 간단하고 효율적인 VTON diffusion model을 제안하여 옷과 사람의 상호 작용을 혁신한다. CatVTON은 옷과 사람 이미지를 공간 차원으로 연결하여 입력으로 받아, 어떤 종류의 옷이든 목표 대상에 쉽게 적용할 수 있다.



### Diff4VS: HIV-inhibiting Molecules Generation with Classifier Guidance Diffusion for Virtual Screening (https://arxiv.org/abs/2407.15880)
- **What's New**: 본 논문에서는 새로운 HIV 억제제 발견을 위한 새로운 방법론인 Diff4VS를 제안하며, 이는 Classifier Guidance Diffusion 모델과 리간드 기반 가상 스크리닝 전략을 결합한 모델이다. 특히, HIV 분자 데이터셋을 이용하여 훈련된 분류기의 그래디언트를 활용하여 Diffusion 모델을 안내하여 HIV 억제제 분자를 생성한다. 이 방법은 기존 방법들보다 더 많은 후보 HIV 억제제 분자를 생성할 수 있음을 실험적으로 확인했다. 또한, 리간드 기반 가상 스크리닝에서 영감을 받아 새로운 지표인 DrugIndex를 제안했으며, 이는 생성된 분자 중 후보 약물 분자 비율과 학습 데이터셋의 후보 약물 분자 비율의 비율로 정의된다. DrugIndex는 제약 관점에서 분자 생성 모델의 발전을 평가하는 새로운 방법을 제시한다. 더 나아가, 가상 스크리닝을 위해 분자 생성 모델을 사용할 때 관찰되는 새로운 현상, 즉 생성된 분자는 실제 분자에 비해 알려진 약물 분자와 매우 유사한 비율이 낮다는 '분자 생성 저하' 현상을 보고한다. 이러한 현상은 생성 모델에서 특정 구조를 가진 분자를 생성하는 어려움에서 비롯될 수 있다는 분석 결과를 제시하며, 이 연구는 방법론, 지표, 현상 분석을 통해 약물 설계에서 생성 모델의 응용에 기여한다.



### Decentralized Federated Anomaly Detection in Smart Grids: A P2P Gossip Approach (https://arxiv.org/abs/2407.15879)
- **What's New**: 이 논문은 스마트 그리드 환경에서 데이터 보안과 개인정보 보호 문제를 해결하기 위해 분산형 연합 학습(decentralized federated learning) 기반의 이상 탐지 시스템을 제안한다. 이 시스템은 분산형 환경에서 데이터 공유 없이 협업적인 학습을 가능하게 하며, 기존의 연합 학습 방식과 달리 중앙 집중식 에이전트에 대한 의존도를 줄이고 모델 업데이트 전송 과정에서 개인정보 유출 위험을 최소화한다.



### CRMSP: A Semi-supervised Approach for Key Information Extraction with Class-Rebalancing and Merged Semantic Pseudo-Labeling (https://arxiv.org/abs/2407.15873)
- **What's New**: This paper proposes a novel automatic evaluation metric called **Knowledge Dependent Answerability (KDA)** for Multiple Choice Question (MCQ) generation. KDA measures the MCQ's ability to assess a student's knowledge of the target fact, rather than simply focusing on n-gram similarity to a gold standard. This metric is designed to address the limitations of existing evaluation metrics like BLEU, ROUGE, and METEOR, which fail to capture the educational value of generated MCQs.



### Semantic Prototypes: Enhancing Transparency Without Black Boxes (https://arxiv.org/abs/2407.15871)
- **What's New**: 이 논문은 기존 prototype 방법의 한계를 극복하기 위해 semantic description을 이용한 새로운 prototype framework을 제안한다. 기존 방법들은 raw data를 기반으로 prototype을 정의하여 이해하기 어려운 latent space를 생성하는 반면, 이 논문에서는 concept-based semantic description을 활용하여 데이터를 semantic level에서 clustering하고 prototype을 정의한다. 따라서 prototype은 데이터의 기본적인 특징을 직관적으로 나타내며 해석하기 쉽다.



### Long Input Sequence Network for Long Time Series Forecasting (https://arxiv.org/abs/2407.15869)
Comments:
          9 pages

- **What's New**: 기존의 자동 MCQ 생성 평가 메트릭은 단어 유사성만 평가하여 교육적 가치를 고려하지 못했지만, 이 논문에서는 MCQ의 대답 가능성을 측정하는 새로운 지식 종속 가능성(KDA) 메트릭을 제안한다. KDA는 실제 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont는 이를 자동으로 근사화하기 위해 사전 훈련된 언어 모델을 사용한다. 이 메트릭들은 Human evaluation을 통해 KDA 및 실제 강의실 환경에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 함께 사용될 때 다양한 MCQ 품질 측정 지표에 대한 강력한 예측력을 가진다.



### SmartQuant: CXL-based AI Model Store in Support of Runtime Configurable Weight Quantization (https://arxiv.org/abs/2407.15866)
- **What's New**: 이 논문은 생성적 AI 모델의 추론 과정에서 가중치의 중요성이 맥락에 따라 크게 달라지는 것을 발견했습니다. 이는 가중치 양자화를 적응적으로 구성하여 생성적 AI 모델의 추론 효율을 향상시킬 수 있는 가능성을 보여줍니다. 가변 정밀도 산술을 위한 하드웨어 지원을 활용하여 구성 가능한 가중치 양자화를 통해 AI 모델의 메모리 액세스 속도와 에너지 효율을 향상시키는 방법은 거의 연구되지 않았습니다. 이 논문은 빠르게 성장하는 CXL 생태계를 기반으로 이러한 간극을 메우기 위한 CXL 기반 설계 솔루션을 개발합니다. 핵심은 CXL 메모리 컨트롤러가 런타임 구성 가능한 가중치 양자화를 지원하고 활용하는 데 적극적인 역할을 하도록 하는 것입니다. 대표적인 생성적 AI 모델인 트랜스포머를 사용하여 실험을 수행한 결과, 제안된 설계 솔루션의 효과를 잘 보여주었습니다.



### A Survey of AI-Powered Mini-Grid Solutions for a Sustainable Future in Rural Communities (https://arxiv.org/abs/2407.15865)
- **What's New**: 이 논문은 지속 가능한 에너지 접근성을 강화하기 위한 AI 기반 미니 그리드 솔루션을 종합적으로 조사합니다. 미니 그리드는 독립적으로 또는 국가 전력망과 함께 작동하여 원격 지역 사회에 안정적이고 저렴한 전력을 공급할 수 있습니다. 태양열 및 풍력과 같은 재생 에너지원의 본질적인 불확실성을 감안하여 정확한 에너지 예측 및 관리의 필요성을 논의하고, 에너지 공급 및 수요 예측, 그리드 운영 최적화, 지속 가능한 에너지 분배를 보장하는 데 있어 고급 AI 기술의 역할을 강조합니다. 이 논문은 단기 및 장기 예측 모두에 대한 효과를 평가하여 통계적 방법, 기계 학습 알고리즘 및 하이브리드 접근 방식을 포함한 다양한 예측 모델을 검토합니다. 또한 모델 구현 및 검증을 위한 Prophet, NeuralProphet 및 N-BEATS와 같은 공개 데이터 세트 및 도구를 살펴봅니다. 이 설문 조사는 실제 응용 프로그램에 대한 모델 적응 및 최적화의 과제를 해결하여 미래 연구에 대한 권장 사항으로 마무리됩니다.



### Adversarial Attacks and Defenses on Text-to-Image Diffusion Models: A Survey (https://arxiv.org/abs/2407.15861)
- **What's New**: 이 논문은 텍스트-이미지 확산 모델의 견고성(robustness)과 안전성(safety)에 대한 심층적인 분석을 제공하며, 이를 위한 다양한 적대적 공격(adversarial attack)과 방어(defense) 방법들을 살펴봅니다. 특히, 기존 연구에서 간과되었던 문법적으로 부정확한 프롬프트(grammatically incorrect prompt)에 대한 견고성 강화와 악의적인 프롬프트(malicious prompt)에 대한 안전성 강화에 초점을 맞춥니다. 또한, 기존 공격 및 방어 방법의 한계점을 분석하고 잠재적인 해결책을 논의합니다.  



### A Survey on Trustworthiness in Foundation Models for Medical Image Analysis (https://arxiv.org/abs/2407.15851)
- **What's New**: 이 논문은 의료 영상 분석에서의 기반 모델(foundation models) 신뢰성에 대한 새로운 서베이를 제공합니다. 기반 모델의 신뢰성은 의료 분야에서 매우 중요하며 개인 정보 보호, 견고성, 신뢰성, 설명 가능성, 공정성 등을 포함합니다. 이 논문은 특히 의료 영상 세분화, 의료 보고서 생성, 의료 질문과 답변 (Q&A), 질병 진단 등 다양한 응용 분야에서 기반 모델의 신뢰성에 대한 연구를 검토하고 분석합니다.



### On shallow planning under partial observability (https://arxiv.org/abs/2407.15820)
Comments:
          Presented at deployable RL (RLC conference 2024)

- **What's New**: 이 논문은 강화 학습 문제에서 할인율(discount factor)이 에이전트의 계획 수평선(planning horizon)을 정의하고, 이것이 편향-분산 트레이드오프(bias-variance trade-off)에 미치는 영향을 연구합니다. 특히, 부분 관측 가능성(partial observability) 환경에서 짧은 계획 수평선이 유리할 수 있음을 보여줍니다.



### Explaining Decisions in ML Models: a Parameterized Complexity Analysis (https://arxiv.org/abs/2407.15780)
Comments:
          A short version of the paper has been accepted at the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR 2024)

- **What's New**: 본 논문은 다양한 기계 학습(ML) 모델에서 설명 문제의 매개변수 복잡성에 대한 포괄적인 이론적 조사를 제시합니다. 블랙박스 인식과는 달리 이 연구는 투명한 내부 메커니즘을 가진 모델에 중점을 둡니다. 우리는 연역적 (abductive) 및 대조적 (contrastive)의 두 가지 주요 유형의 설명 문제를 다루며, 둘 다 지역적 및 글로벌 변형에서 다룹니다. 우리의 분석은 의사 결정 트리, 의사 결정 세트, 의사 결정 목록, 순서형 이진 결정 다이어그램, 랜덤 포레스트, 부울 회로 및 이들의 앙상블을 포함한 다양한 ML 모델을 포괄하며, 각 모델은 고유한 설명 과제를 제공합니다. 이 연구는 이러한 모델에 대한 설명을 생성하는 복잡성에 대한 기본적인 이해를 제공하여 설명 가능한 AI(XAI)에서 중요한 격차를 메웁니다. 이 연구는 XAI 분야의 추가 연구에 중요한 통찰력을 제공하여 AI 시스템의 투명성과 책임성에 대한 광범위한 논의에 기여합니다.



### TaskGen: A Task-Based, Memory-Infused Agentic Framework using StrictJSON (https://arxiv.org/abs/2407.15734)
Comments:
          53 pages

- **What's New**: TaskGen, an open-sourced agentic framework that utilizes an Agent to solve arbitrary tasks by breaking them down into subtasks. Each subtask is assigned to an Equipped Function or another Agent for execution. This framework prioritizes concise output for reduced token usage and increased efficiency.



### Problems in AI, their roots in philosophy, and implications for science and society (https://arxiv.org/abs/2407.15671)
- **What's New**: 본 논문은 MCQ 생성 평가 지표로 새로운 지표인 '지식 종속 가능성(Knowledge Dependent Answerability, KDA)'을 제안합니다. 이 지표는 MCQ의 대답 가능성(answerability)을 측정하며, 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 기존 지표(BLEU, ROUGE, METEOR)는 MCQ의 교육적 가치를 고려하지 않고 단어 유사성만 비교했던 반면, KDA는 MCQ가 실제로 학생의 지식을 평가하는 능력을 측정합니다.



### Interpretable Concept-Based Memory Reasoning (https://arxiv.org/abs/2407.15527)
- **What's New**: 이 연구는 MCQ 자동 생성 평가 지표로서 지식 종속 가능성(KDA, Knowledge Dependent Answerability)를 제안하며, 이는 MCQ가 대상 사실(target fact)에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 기존 지표인 BLEU, ROUGE, METEOR는 MCQ의 교육적 가치를 고려하지 않고 단순히 문장 유사성만 평가합니다. KDA는 인간 설문 조사를 기반으로 측정되는데, 연구진은 이를 자동화하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 개발했습니다. 이 지표들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 방식을 모방하여 KDA를 근사합니다.



### Algebraic anti-unification (https://arxiv.org/abs/2407.15510)
- **What's New**: 이 논문은 자동 MCQ 생성을 위한 새로운 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하며, 기존 지표인 BLEU, ROUGE, METEOR와 달리 단어 유사도가 아닌 교육적 가치를 중시합니다. KDA를 자동으로 계산하기 위해 KDA_disc와 KDA_cont라는 두 가지 지표를 제안하며, 이는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.  



### Odyssey: Empowering Agents with Open-World Skills (https://arxiv.org/abs/2407.15325)
- **What's New**: 본 논문에서는 **ODYSSEY**라는 새로운 프레임워크를 소개하며, 대규모 언어 모델 (LLM) 기반 에이전트에게 Minecraft 세계를 탐험할 수 있는 오픈월드 스킬을 부여합니다. ODYSSEY는 40개의 기본 스킬과 183개의 복합 스킬로 구성된 오픈월드 스킬 라이브러리를 갖춘 에이전트, Minecraft 위키에서 추출한 39만 개 이상의 지시 사항이 포함된 질의 응답 데이터셋으로 미세 조정된 LLaMA-3 모델, 장기 계획 작업, 동적 즉각 계획 작업, 자율 탐험 작업으로 구성된 새로운 오픈월드 벤치마크로 구성됩니다.



### New Rules for Causal Identification with Background Knowledg (https://arxiv.org/abs/2407.15259)
- **What's New**: 본 논문은 MCQ(Multiple Choice Questions) 생성을 위한 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 제안합니다. 기존의 BLEU, ROUGE, METEOR는 단어 유사성만 고려하여 MCQ의 교육적 가치를 평가하지 못했지만, KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. KDA는 인간의 응답을 기반으로 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 KDA를 근사합니다.



### Explaining Decisions of Agents in Mixed-Motive Games (https://arxiv.org/abs/2407.15255)
- **What's New**: 본 논문은 협력과 경쟁이 동시에 존재하는 상황에서, 에이전트의 의사 결정 과정을 설명하는 새로운 방법을 제안합니다. 기존의 설명 방법들은 협력적인 환경에만 적용 가능했지만, 이 논문은 협력과 경쟁이 혼재된 상황에서도 에이전트의 행동을 설명할 수 있습니다. 특히, 'cheap-talk' (저비용 대화)나 행동을 통한 암묵적인 소통 (implicit communication) 등을 설명할 수 있다는 장점이 있습니다.



### Text-Augmented Multimodal LLMs for Chemical Reaction Condition Recommendation (https://arxiv.org/abs/2407.15141)
- **What's New**: 본 논문에서는 MCQ (Multiple Choice Question) 생성을 위한 새로운 자동 평가 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하여, 기존의 BLEU, ROUGE, METEOR와 같은 지표들이 교육적 가치를 고려하지 못하는 문제를 해결합니다.



### A Measure for Level of Autonomy Based on Observable System Behavior (https://arxiv.org/abs/2407.14975)
Comments:
          9 pages, 1 figure, 3 tables

- **What's New**: 본 논문에서는 자율 시스템의 자율성 수준을 측정하는 새로운 방법을 제안한다. 이 방법은 시스템의 관찰 가능한 행동을 기반으로 자율성 수준을 예측하는 것을 목표로 한다. 또한 이러한 측정 방법을 통합한 알고리즘을 제시한다. 이 방법은 실행 중에 자율 시스템을 비교하는 데 유용하며, 방어 시스템에도 적용할 수 있다. 특히, 반-자율 시스템은 자율 시스템을 정확히 식별해야 하기 때문에 이 방법이 중요하다.



### TraveLLM: Could you plan my new public transit route in face of a network disruption? (https://arxiv.org/abs/2407.14926)
- **What's New**: 이 논문은 사용자의 선호도와 지하철 운행 중단과 같은 예외 상황을 고려한 대중교통 경로 추천을 위해 LLMs(Large Language Models)을 활용하는 TraveLLM이라는 프로토타입 시스템을 제안한다. 이는 기존 내비게이션 앱들이 예외 상황에서 적절한 경로를 추천하지 못하는 문제점을 해결하기 위한 시도이다.



### Unveiling the Decision-Making Process in Reinforcement Learning with Genetic Programming (https://arxiv.org/abs/2407.14714)
Comments:
          Accepted at: The Fifteenth International Conference on Swarm Intelligence (ICSI'2024)

- **What's New**: 이 논문은 기존의 MCQ 생성 평가 지표들이 단어 유사성에만 초점을 맞춰 교육적 가치를 고려하지 못한다는 문제점을 지적하고, 지식 종속 가능성(KDA)이라는 새로운 자동 평가 지표를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. 또한, 이 논문은 KDA를 근사화하는 두 가지 자동 평가 지표, KDA_disc와 KDA_cont를 제안하고, 이들이 실제 강의실에서 사용 가능성과 높은 상관관계를 가진다는 것을 실험적으로 보여줍니다.



### Relational Composition in Neural Networks: A Survey and Call to Action (https://arxiv.org/abs/2407.14662)
- **What's New**: 본 논문은 신경망이 데이터를 표현하는 방식에 대한 새로운 관점을 제시하며, 특징 벡터(feature vector)의 조합을 통해 복잡한 관계를 나타내는 '관계적 구성(relational composition)'에 대한 연구의 필요성을 강조합니다. 기존 연구에서 성공적으로 사용되었던 특징 벡터 발견 알고리즘은 이러한 관계적 구성을 고려하지 않아 신경망의 해석 가능성에 대한 이해를 제한합니다.  



### Towards consistency of rule-based explainer and black box model -- fusion of rule induction and XAI-based feature importanc (https://arxiv.org/abs/2407.14543)
- **What's New**: 본 논문은 블랙 박스 모델의 의사 결정 과정을 규칙 기반 모델로 설명하는 새로운 방법을 제안합니다. 기존 방법들은 블랙 박스 모델을 규칙 기반 모델로 근사화하는 데 집중했지만, 두 모델의 의사 결정 과정이 일치하는지 여부는 연구되지 않았습니다. 본 논문에서는 규칙 기반 모델이 블랙 박스 모델의 성능을 모방하도록 하는 새로운 방법을 제안합니다. 해당 방법은 규칙 생성과 XAI 방법을 사용하여 블랙 박스 모델의 중요한 특성을 고려하는 설명 융합(explanation fusion)을 수행합니다. 이를 통해 전역 및 지역 규칙 기반 설명을 얻을 수 있습니다.



### WayEx: Waypoint Exploration using a Single Demonstration (https://arxiv.org/abs/2407.15849)
Comments:
          ICRA 2024

- **What's New**: 새로운 방법인 WayEx를 제안하여 단일 데모(demonstration)에서 복잡한 목표 조건형 로봇 작업을 학습합니다. WayEx는 기존의 모방 학습(imitation learning) 방법과 달리 전문가 예제를 적게 필요로 하고 데모 중에 수행된 액션에 대한 정보가 필요하지 않습니다. 이는 새로운 보상 함수를 도입하고 지식 확장 기술을 사용하여 가능합니다. WayEx는 다양한 환경에서 6가지 다양한 작업에 효율성을 보여줍니다.



### LLMmap: Fingerprinting For Large Language Models (https://arxiv.org/abs/2407.15847)
Comments:
          version 0.1 (added missing refs)

- **What's New**: 본 논문은 LLMmap을 소개합니다. 이것은 LLM 기반 애플리케이션을 타겟으로 하는 최초의 지문 공격 방식입니다. LLMmap은 적극적인 지문 방식을 사용하여 애플리케이션에 신중하게 구성된 쿼리를 보내고 응답을 분석하여 사용되는 특정 LLM 모델을 식별합니다.



### Reconstructing Training Data From Real World Models Trained with Transfer Learning (https://arxiv.org/abs/2407.15845)
- **What's New**: 이 논문은 이미지 임베딩 공간에서 데이터 재구성을 통해 대규모 사전 훈련된 모델(DINO-ViT, CLIP 등)에서 전이 학습으로 훈련된 모델에 대한 데이터 재구성을 가능하게 하는 새로운 방법을 제시합니다. 이 방법은 기존 방법보다 훨씬 현실적인 환경에서 적용 가능하며, 이미지 해상도 및 학습 데이터 크기에 대한 제약이 적습니다. 또한, 클러스터링 기반 방법을 사용하여 수천 개의 후보 중에서 좋은 재구성 결과를 식별하여 기존 방법의 한계를 극복했습니다.



### HandDGP: Camera-Space Hand Mesh Prediction with Differentiable Global Positioning (https://arxiv.org/abs/2407.15844)
Comments:
          To be presented at ECCV 2024

- **What's New**: 본 논문은 단일 RGB 이미지에서 카메라 공간 손 메시(camera-space hand meshes)를 예측하는 데 있어, 기존의 2단계 방식(손 이미지를 자르고 상대 좌표계에서 메시를 예측한 후 카메라 공간으로 변환)의 한계를 극복하기 위해 2D-3D 대응 문제(correspondence problem)를 해결하는 end-to-end 솔루션을 제안합니다. 이는 카메라 공간 출력에서 네트워크의 다른 부분으로 역전파(back-propagation)를 가능하게 하는 새로운 미분 가능한 전역 위치 모듈(differentiable global positioning module)을 사용합니다. 또한, 동일한 카메라로 촬영된 것처럼 학습 데이터셋과 입력 이미지를 조화시키는 이미지 정정 단계(image rectification step)를 도입하여 문제의 고유한 척도-깊이 모호성(scale-depth ambiguity)을 완화합니다.



### CarFormer: Self-Driving with Learned Object-Centric Representations (https://arxiv.org/abs/2407.15843)
Comments:
          Accepted to ECCV 2024, code and the pre-trained models can be found at this https URL

- **What's New**: 본 논문에서는 MCQ 생성의 교육적 가치를 고려하는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 평가합니다. 특히, KDA를 Human Survey를 통해 측정하는 방법을 보여주고, Pre-trained Language Model을 활용하여 KDA를 근사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제시합니다. Human Evaluation 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 상황에서의 사용성과 강한 상관관계를 보입니다. 또한, n-gram 기반 유사성 메트릭과 결합하여 KDA_disc와 KDA_cont는 전문가가 평가한 다양한 MCQ 품질 측정 지표에 대한 강력한 예측력을 보여줍니다.



### Importance Sampling-Guided Meta-Training for Intelligent Agents in Highly Interactive Environments (https://arxiv.org/abs/2407.15839)
- **What's New**: 본 논문은 자동 MCQ 생성을 위한 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ의 답변 가능성을 측정하여 학생의 지식 수준을 평가하는 능력을 측정합니다. 이 지표는 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 지표와 달리 교육적 가치를 고려합니다. 또한 본 논문에서는 KDA를 자동으로 측정하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 이러한 지표는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. Human evaluation을 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다.



### Towards Latent Masked Image Modeling for Self-Supervised Visual Representation Learning (https://arxiv.org/abs/2407.15837)
- **What's New**: 본 연구는  "지식 종속 가능성(KDA)" 라는 새로운 자동 평가 지표를 제안하여 MCQ 생성의 교육적 가치를 평가합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제시하며, 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 설정에서의 사용성 모두와 강한 상관관계를 보이는 것으로 나타났습니다.



### Learning to Manipulate Anywhere: A Visual Generalizable Framework For Reinforcement Learning (https://arxiv.org/abs/2407.15815)
Comments:
          Webpage: this https URL

- **What's New**: 본 논문에서는 다양한 시각적 방해 요소들을 결합하여 학습된 로봇 정책이 일반화될 수 있도록 하는 시각적 강화 학습에 최적화된 일반화 가능한 프레임워크인 **Maniwhere**를 제안합니다. 여러 시점 간의 공유된 의미 정보와 대응 관계를 포착하기 위해 Spatial Transformer Network (STN) 모듈과 융합된 다중 시점 표현 학습 접근 방식을 소개합니다. 또한, RL 학습 과정을 안정시키고 시각적 일반화 능력을 강화하기 위해 커리큘럼 기반 랜덤화 및 증강 접근 방식을 사용합니다. Maniwhere의 효과를 보여주기 위해, 우리는 명료한 객체, 양손 조작 및 숙련된 손 조작 작업을 포함한 8가지 작업을 설계하고 3개의 하드웨어 플랫폼에서 Maniwhere의 강력한 시각적 일반화 및 sim2real 전이 능력을 보여줍니다. 실험 결과 Maniwhere는 기존 최첨단 방법보다 훨씬 뛰어난 성능을 보여줍니다.



### Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budg (https://arxiv.org/abs/2407.15811)
Comments:
          41 pages, 28 figures, 5 tables

- **What's New**: 본 논문에서는 이미지 생성 모델(Text-to-Image, T2I) 학습의 비용을 줄이기 위해 새로운 방법을 제안합니다. 기존의 대규모 이미지 생성 모델은 막대한 계산 자원을 필요로 하여 연구 및 개발에 있어 진입 장벽이 높았습니다. 본 논문에서는 이미지 패치의 일부를 마스킹하여 학습 과정의 계산 비용을 크게 줄이는 방법을 제시합니다. 특히, 패치 믹서(patch-mixer)를 사용하여 마스킹 전에 모든 패치를 사전 처리하는 방식으로, 기존의 모델 축소(model downscaling) 방식보다 효과적으로 비용을 절감하면서 성능 저하를 최소화합니다. 또한, 혼합 전문가 계층(mixture-of-experts layers) 등 최신 트랜스포머 아키텍처 개선 사항을 도입하여 성능을 향상시키고, 합성 이미지를 사용하는 것의 중요성을 강조합니다.  결과적으로, 11억 개의 매개변수를 가진 스파스 트랜스포머 모델을 3700만 개의 실제 및 합성 이미지를 사용하여 1,890 달러의 저렴한 비용으로 학습시켰으며, COCO 데이터셋에서 12.7 FID를 달성했습니다. 이는 Stable Diffusion 모델보다 118배, 최첨단 모델보다 14배 저렴한 비용으로 경쟁력 있는 성능을 달성한 것입니다. 본 논문의 목표는 이러한 연구 결과를 바탕으로 저렴한 비용으로 대규모 확산 모델을 학습할 수 있는 훈련 파이프라인을 공개하여 이미지 생성 모델 연구 및 개발의 접근성을 높이는 것입니다.



### CLIP with Generative Latent Replay: a Strong Baseline for Incremental Learning (https://arxiv.org/abs/2407.15793)
Comments:
          15 pages, 1 figure. Accepted at the The 35th British Machine Vision Conference 2024 (BMVC 2024), Glasgow, UK

- **What's New**: 본 논문에서는 기존의 MCQ 평가 지표인 BLEU, ROUGE, METEOR 등이 MCQ의 교육적 가치를 고려하지 않고 단순히 문장 유사도만 측정하는 문제점을 지적하고, MCQ의 대답 가능성(answerability)을 측정하여 교육적 가치를 반영하는 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안한다. KDA는 학생들의 응답을 기반으로 계산되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표로 구현되었다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화한다. 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 설정에서의 사용성과 높은 상관관계를 가지고 있음을 보여주었다. 또한, n-gram 기반 유사도 지표와 함께 사용하면 다양한 전문가가 평가한 MCQ 품질 지표에 대한 예측력이 높아지는 것으로 나타났다.



### Concept-Based Interpretable Reinforcement Learning with Limited to No Human Labels (https://arxiv.org/abs/2407.15786)
Comments:
          23 pages, 6 figures, 9 tables

- **What's New**: 이 논문은 교육적 가치를 고려하는 새로운 자동 MCQ 평가 지표인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 수준을 평가하는 데 도움을 주는 지표입니다. KDA는 학생들의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 KDA를 근사합니다. 연구 결과, KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 다른 n-gram 기반 유사성 지표와 함께 사용했을 때 MCQ의 품질을 예측하는 데 강력한 힘을 발휘했습니다. 

- **Technical Details**: KDA는 학생들의 응답을 기반으로 측정되는 지표이며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용하여 KDA를 근사합니다. KDA_disc와 KDA_cont는 Human evaluation을 통해 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, MCQ의 품질을 예측하는 데 강력한 힘을 발휘했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, 다른 n-gram 기반 유사성 지표와 함께 사용했을 때 MCQ의 품질을 예측하는 데 강력한 힘을 발휘했습니다.



### Diffusion Model Based Resource Allocation Strategy in Ultra-Reliable Wireless Networked Control Systems (https://arxiv.org/abs/2407.15784)
Comments:
          5 pages, 4 figures

- **What's New**: 이 논문은 무선 네트워크 제어 시스템 (WNCS)의 리소스 할당 문제에 대한 새로운 확산 모델 기반 접근 방식을 제안합니다. 특히, DDPM(Denoising Diffusion Probabilistic Model)을 사용하여 채널 상태 정보 (CSI)를 조건으로 사용하여 최적의 블록 길이 값을 생성하여 총 전력 소비를 최소화합니다.



### Local Occupancy-Enhanced Object Grasping with Multiple Triplanar Projection (https://arxiv.org/abs/2407.15771)
- **What's New**: 본 논문은 일반적인 물체를 잡는 로봇의 문제를 해결합니다.  기존 연구와 마찬가지로 이 작업은 깊이 카메라에서 캡처한 단일 뷰 3D 관찰(즉, 포인트 클라우드)을 입력으로 받습니다. 물체 잡기의 성공은 장면 내 물체의 모양을 포괄적으로 이해하는 데 크게 의존합니다. 그러나 단일 뷰 관찰은 특히 복잡한 혼잡한 장면에서 포인트 클라우드에 간격을 만드는 폐색(자체 폐색 및 객체 간 폐색 포함)으로 인해 종종 불완전합니다. 이는 객체 모양에 대한 불완전한 인식을 초래하고 객체 잡기 중에 실패 또는 부정확한 자세 추정을 자주 발생시킵니다. 본 논문에서는 로컬 점유 예측을 통해 잡기에 관련된 장면 영역을 완성하는 효과적이지만 간단한 솔루션을 통해 이 문제를 해결합니다. 제안된 모델은 먼저 장면에서 가장 가능성이 높은 잡기 지점을 여러 개 제안하여 실행됩니다. 각 잡기 지점 주변에는 모듈이 설계되어 해당 이웃의 모든 복셀을 비어 있는지 아니면 어떤 객체에 의해 점유되었는지 추론합니다. 중요한 점은 점유 맵이 로컬 큐와 글로벌 큐를 융합하여 추론된다는 것입니다. 장거리 맥락 정보를 효율적으로 집계하기 위해 멀티 그룹 삼면 체계를 구현합니다. 모델은 로컬 점유 향상된 객체 모양 정보를 활용하여 6-DoF 잡기 자세를 추가로 추정하고 상위 순위의 잡기 제안을 반환합니다. 대규모 GraspNet-1Billion 벤치마크와 실제 로봇 암 모두에 대한 종합적인 실험은 제안된 방법이 혼잡하고 폐색된 장면에서 관찰되지 않은 부분을 효과적으로 완성할 수 있음을 보여줍니다. 점유 향상된 기능의 이점을 통해 모델은 잡기 평균 정밀도와 같은 다양한 성능 지표에서 다른 경쟁 방법을 명확히 능가합니다.



### Model editing for distribution shifts in uranium oxide morphological analysis (https://arxiv.org/abs/2407.15756)
Comments:
          Presented at CV4MS @ CVPR 2024

- **What's New**: 본 논문은 UOC 합성 조건을 분류하는 딥러닝 모델에 대한 모델 에디팅 (Model Editing) 기법을 제시하며, 이를 통해 특정 도메인에서 발생하는 분포 변화 (distribution shifts)에 대한 일반화 성능 (generalization)을 향상시킬 수 있음을 보여줍니다. 특히, 모델 에디팅은 두 가지 커레이션된 데이터셋(습도 챔버에서 노화된 U$_{3}$O$_{8}$의 마이크로그래프(micrographs)와 다른 주사 전자 현미경 (scanning electron microscopes)으로 획득한 마이크로그래프)에서 미세 조정 (fine-tuning)보다 뛰어난 성능을 보여줍니다.



### Diffusion for Out-of-Distribution Detection on Road Scenes and Beyond (https://arxiv.org/abs/2407.15739)
Comments:
          ECCV 2024 - Benchmark page: this https URL

- **What's New**: 본 논문은 기존의 MCQ 평가 메트릭이 교육적 가치를 고려하지 않고, 단순히 문장의 유사성만 평가한다는 문제를 지적하며 새로운 메트릭인 "지식 종속 가능성 (KDA)"를 제안합니다. KDA는 MCQ가 학생의 지식을 평가하는 능력을 측정합니다.



### Parallel Split Learning with Global Sampling (https://arxiv.org/abs/2407.15738)
- **What's New**: 이 논문은 분산 딥 러닝(DDL) 시스템에서 발생하는 문제, 특히 병렬 분할 학습(PSL)에서 발생하는 대량의 유효 배치 크기, 비IID 데이터 분포, 지연 효과를 해결하기 위해 유니폼 글로벌 샘플링(UGS)과 잠재 디리클레 샘플링(LDS) 메서드를 제안합니다.  



### GFE-Mamba: Mamba-based AD Multi-modal Progression Assessment via Generative Feature Extraction from MCI (https://arxiv.org/abs/2407.15719)
Comments:
          35 pages, 4 figures

- **What's New**: 이 논문은 Generative Feature Extraction (GFE) 기반의 새로운 분류기인 GFE-Mamba를 소개하며 이는 MCI에서 AD로의 전환을 예측하는 데 효과적입니다.



### Mamba meets crack segmentation (https://arxiv.org/abs/2407.15714)
Comments:
          32 pages, 8 figures. Preprint submitted to Elsevier

- **What's New**: 이 논문은 균열 분할 (crack segmentation) 모델에 Mamba를 사용하는 새로운 방법을 제안합니다. Mamba는 선형적인 공간 및 연산 복잡성과 강력한 전역 인식 (global perception) 능력으로 인해 주목받고 있습니다. 특히, 이 논문은 Mamba와 어텐션 메커니즘 (attention mechanism) 사이의 관계를 밝혀내고, 어텐션 블록의 원리를 따르는 새로운 Mamba 모듈인 CrackMamba를 개발합니다.



### SwinSF: Image Reconstruction from Spatial-Temporal Spike Streams (https://arxiv.org/abs/2407.15708)
- **What's New**: 본 논문은 스파이크 카메라에서 획득한 스파이크 스트림에서 동적인 장면을 재구성하기 위한 새로운 모델인 Swin Spikeformer (SwinSF)를 소개합니다. SwinSF는 스파이크 특징 추출, 공간-시간 특징 추출 및 최종 재구성 모듈로 구성되어 있으며, 이동 창 자기 주의(shifted window self-attention)와 제안된 시간 스파이크 주의(temporal spike attention)를 결합하여 공간 및 시간 역학을 포괄적으로 추출하여 스파이크 스트림을 더욱 강력하고 정확하게 재구성합니다.



### Predicting the Best of N Visual Trackers (https://arxiv.org/abs/2407.15707)
- **What's New**: 이 논문은 다양한 비디오 특성과 데이터셋에서 최첨단 시각 추적기의 성능이 놀랍도록 다르게 나타나는 것을 관찰합니다. 모든 추적 특성과 데이터셋에서 최고의 성능을 유지하는 단일 추적기는 없습니다. 이러한 차이를 해소하기 위해, 주어진 비디오 시퀀스에 대해 "N개의 추적기 중 최고"를 예측하는, BofN 메타 추적기를 제안합니다. 핵심적으로, 추적 성능 예측 네트워크(TP2N)는 초기 프레임 몇 개만 사용하여 주어진 비디오 시퀀스에 대해 예측된 최고 성능 시각 추적기를 선택합니다. 또한, 정기적인 시간 간격 후 최고의 성능을 예측하는 프레임 수준 BofN 메타 추적기를 소개합니다. TP2N은 자기 지도 학습 아키텍처인 MocoV2, SwAv, BT, DINO를 기반으로 하며, 실험 결과 ViT-S를 백본으로 사용하는 DINO가 가장 좋은 성능을 보입니다. 비디오 수준 BofN 메타 추적기는 LaSOT, TrackingNet, GOT-10K, VOT2019, VOT2021, VOT2022, UAV123, OTB100, WebUAV-3M 등 9개의 표준 벤치마크에서 기존 최첨단 추적기를 훨씬 능가합니다. 프레임 수준 BofN 메타 추적기는 긴 시퀀스 내 추적 시나리오의 변화를 효과적으로 처리하여 더욱 개선된 성능을 보입니다. 예를 들어 GOT-10k에서 BofN 메타 추적기의 평균 중첩은 비디오 수준 설정에서 각각 88.7% 및 프레임 수준 설정에서 91.1%입니다. 최고의 성능을 보이는 추적기인 RTS는 85.20% AO를 달성합니다. VOT2022에서 BofN의 예상 평균 중첩은 비디오 수준 설정에서 각각 67.88% 및 프레임 수준 설정에서 70.98%로, 최고의 성능을 보이는 ARTrack인 64.12%보다 높습니다. 이 연구는 또한 모든 일반적으로 사용되는 벤치마크에서 경쟁적인 추적 방법을 광범위하게 평가하고, 해당 프로토콜을 따릅니다. 코드, 훈련된 모델 및 결과는 곧 https URL에서 공개적으로 제공될 예정입니다.



### A Life-long Learning Intrusion Detection System for 6G-Enabled IoV (https://arxiv.org/abs/2407.15700)
- **What's New**: 본 논문에서는 6G 네트워크의 엄격한 신뢰성과 보안 요구사항을 충족하는 동적이고 다양한 IoV 환경을 위한 새로운 침입 탐지 시스템을 제안합니다. 이 시스템은 6G의 도입으로 인해 IoV가 겪게 될 다양한 사이버 위협에 대응하기 위해 지속적인 학습 (life-long learning) 패러다임을 활용합니다.



### AI-Driven Fast and Early Detection of IoT Botnet Threats: A Comprehensive Network Traffic Analysis Approach (https://arxiv.org/abs/2407.15688)
- **What's New**: 본 논문은 IoT 보트넷 트래픽을 조기에 탐지하기 위한 새로운 방법론을 제안합니다. 특히 공격을 앞서고 조직하는 스텔스 보트 통신 (stealth bot communication)을 탐지하는 데 초점을 맞춥니다. 이 방법론은 단방향과 양방향 데이터 흐름을 고려하여 IoT 네트워크 트래픽을 분석하는 방법을 제안하고, 패킷 포맷까지 분석합니다. 또한, 다양한 반지도 학습 기술을 활용하여 트래픽을 모델링합니다.



### HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning (https://arxiv.org/abs/2407.15680)
Comments:
          Accepted as a main conference paper at ECCV 2024 (this https URL)

- **What's New**: 이 논문은 Vision-Language Model (VLM)의 환각 (hallucination) 문제를 해결하기 위해 HaloQuest라는 새로운 데이터셋을 소개합니다. HaloQuest는 실제 이미지와 합성 이미지를 모두 사용하여 다양한 환각 유발 시나리오를 포함하며, VLM의 환각 문제를 평가하고 완화하기 위한 새로운 벤치마크 역할을 합니다. 또한, VLM의 답변을 평가하는 새로운 자동 평가 (Auto-Eval) 메커니즘을 제안합니다.



### Flow-guided Motion Prediction with Semantics and Dynamic Occupancy Grid Maps (https://arxiv.org/abs/2407.15675)
Comments:
          Accepted for publication at the 27th IEEE International Conference on Intelligent Transportation Systems (ITSC) (ITSC 2024)

- **What's New**: 본 논문은  Occupancy Grid Maps (OGMs)를 기반으로 미래 자동차의 semantic grid와 flow를 예측하는 새로운 다중 작업 프레임워크를 제안합니다. 기존 방법은 scene의 진화 예측이나 복잡한 행동 학습에 집중했지만, scene의 흐름이나 속도 벡터 예측은 고려하지 않았습니다. 반면, 이 논문에서 제시된 프레임워크는 semantic flow 정보를 활용하여 warped semantic grid를 생성하여 scene의 dynamic vehicle을 더 정확하게 유지합니다.



### SLVideo: A Sign Language Video Moment Retrieval Framework (https://arxiv.org/abs/2407.15668)
Comments:
          5 pages, 3 figures, 1 table

- **What's New**: 본 논문은 수화 영상 검색을 위한 소프트웨어인 SLVideo를 소개한다. SLVideo는 손과 얼굴 둘 다를 인식하여 수화 비디오 검색을 가능하게 한다. 기존 시스템들은 수화 인식 알고리즘에 의존하지만, 얼굴 표정 인식은 포함하지 않는다는 한계가 있었다.  SLVideo는 손과 얼굴 표정을 모두 인식하여 수화의 표현력을 풍부하게 해주고, 문맥에 따라 의미가 변하는 수화를 정확하게 인식할 수 있도록 돕는다.



### How to Shrink Confidence Sets for Many Equivalent Discrete Distributions? (https://arxiv.org/abs/2407.15662)
- **What's New**: 이 논문은 학습자가 공통 알파벳  \(\mathcal{X}\) 위에 정의된 알려지지 않은 이산 분포  \((p_k)\)_{k\in\mathcal{K}}  집합을 가지고 있을 때, 각 분포  \(p_k\)에 대해  \(n_k\)개의 관찰 값을 샘플링하여 고확률 신뢰 집합을 구축할 수 있는 상황을 고려합니다.  \((p_k)\)_{k\in\mathcal{K}}  집합은 구조화되어 있으며, 각 분포  \(p_k\)는 알려지지 않은 공통 분포  \(q\)에서  \(\mathcal{X}\)에 알려지지 않은 순열을 적용하여 얻어집니다.  이를 \emph{순열 동등성(permutation-equivalence)} 이라고 부릅니다.  목표는 이러한 구조적 특성을 \emph{활용}하여 정밀화된 신뢰 집합을 구축하는 것입니다.  다른 인기 있는 구조 개념(Lipschitz smoothness, Linearity 등)과 마찬가지로, 순열 동등성은 기계 학습 문제에서 자연스럽게 나타나며, 잠재적인 이점을 활용하려면 특정 접근 방식이 필요합니다.



### Evaluation of Reinforcement Learning for Autonomous Penetration Testing using A3C, Q-learning and DQN (https://arxiv.org/abs/2407.15656)
- **What's New**: 본 논문은 침투 테스트(penetration testing) 자동화를 위해 강화 학습(reinforcement learning) 에이전트를 훈련하는 새로운 방법을 제시합니다. 기존 침투 테스트는 전문가가 수동으로 수행해야 했지만, 본 연구에서는 NASim이라는 환경에서 에이전트가 3가지 시나리오(exploitation, post-exploitation, wiretapping)를 해결하도록 훈련했습니다. 

- **Technical Details**: 본 연구에서는 Q-learning, DQN, A3C 알고리즘을 사용하여 에이전트를 훈련했습니다. 그 중 A3C는 모든 시나리오를 해결하고 일반화(generalization)에 성공했습니다. 더욱이, A3C는 기존의 자동화된 침투 테스트보다 더 적은 액션으로 시나리오를 해결했습니다. 

- **Performance Highlights**: 본 연구에서는 비록 작은 규모의 시나리오와 작은 상태 및 액션 공간으로 훈련되었지만, 강화 학습 에이전트가 성공적으로 침투 테스트를 수행할 수 있음을 보여줍니다.



### Norface: Improving Facial Expression Analysis by Identity Normalization (https://arxiv.org/abs/2407.15617)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문은 딥러닝 기반의 얼굴 표현 분석(FEA) 분야에서 **타스크와 무관한 노이즈(task-irrelevant noise)**를 제거하기 위해 **Norface**라는 새로운 프레임워크를 제안합니다. Norface는 **정규화 네트워크(normalization network)**와 **분류 네트워크(classification network)**로 구성되며, **정규화 네트워크**는 얼굴 표현 일관성(expression consistency)를 유지하면서 모든 원본 이미지를 일관된 자세(pose)와 배경(background)을 가진 공통적인 신원(identity)으로 정규화하여 **타스크와 무관한 노이즈**를 제거합니다. **정규화된 이미지(normalized image)**는 **분류 네트워크(classification network)**에 입력되며, **분류 네트워크**는 **전문가 혼합(Mixture of Experts)**을 사용하여 잠재 표현(latent representation)을 개선하고 여러 AU(Action Unit) 또는 감정 라벨을 처리합니다.

- **Technical Details**: Norface는 **정규화 네트워크**와 **분류 네트워크**로 구성됩니다. **정규화 네트워크**는 **Masked AutoEncoder(MAE)**를 사용하여 얼굴 특징을 추출하고, **Expression Merging Module(EMM)**을 통해 원본 얼굴의 표현 특징을 대상 얼굴에 적용합니다. 또한, 표현 일관성을 유지하기 위해 **표현 손실(expression loss)**과 **눈썹 손실(eyebrow loss)**을 적용합니다. **분류 네트워크**는 **전문가 혼합(Mixture of Experts)**을 사용하여 입력과 출력에 대한 전문가를 활용하여 잠재 표현을 개선하고 다중 AU 또는 감정 라벨을 처리합니다.

- **Performance Highlights**: Norface는 **AU 감지(AU detection), AU 강도 추정(AU intensity estimation), 감정 인식(FER)** 등 다양한 얼굴 표현 분석 태스크에서 SOTA(State-of-the-Art) 성능을 보여줍니다. 특히 **데이터셋 간 전이(cross-dataset) 태스크**에서도 뛰어난 성능을 보이며, **타스크와 무관한 노이즈** 제거에 효과적임을 입증합니다.



### Sustainable broadcasting in Blockchain Network with Reinforcement Learning (https://arxiv.org/abs/2407.15616)
Comments:
          7 pages, 4 figures

- **What's New**: 본 논문은 블록체인 네트워크의 블록 전파 방식을 개선하기 위해 강화 학습 기반의 효율적인 접근 방식을 제안합니다. 기존의 블록체인 방식보다 네트워크 역학을 효과적으로 처리하고 더 나은 성능을 달성합니다. 또한, 시뮬레이터와 RL 환경의 통합은 RL 또는 다른 머신 러닝 기법을 활용하는 새로운 체계 및 프로토콜에 대한 추가 연구를 위한 완벽한 솔루션으로 활용될 수 있습니다.



### A Pairwise Comparison Relation-assisted Multi-objective Evolutionary Neural Architecture Search Method with Multi-population Mechanism (https://arxiv.org/abs/2407.15600)
- **What's New**: 본 논문은 기존의 MCQ 생성 평가 지표들이 교육적 가치를 고려하지 못한다는 문제를 제기하고,  학생의 지식에 의존하여 MCQ의 대답 가능성을 평가하는 새로운 지표인 '지식 의존 가능성 (Knowledge Dependent Answerability, KDA)'을 제안합니다. 특히 KDA를 측정하는 방법과  KDA_disc, KDA_cont라는 두 가지 자동 평가 지표를 제안하여 pre-trained language model을 활용하여 학생의 문제 해결 방식을 모방합니다.  



### Discrete Flow Matching (https://arxiv.org/abs/2407.15595)
- **What's New**: 이 논문은 기존 MCQ 평가 지표의 한계를 지적하고, 학습 내용에 대한 이해도를 평가하는 새로운 지표인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 학생들의 답변을 기반으로 계산되며, 이를 자동화하기 위해 KDA_disc와 KDA_cont 두 가지 지표를 제시합니다.  KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 과정을 모방하여 KDA를 근사합니다. 

- **Technical Details**: KDA는 MCQ의 대답 가능성(answerability)을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 과정을 모방하여 KDA를 근사합니다. 

- **Performance Highlights**: 사람에 의한 평가 결과, KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 지표와 함께 사용하면 다양한 전문가 평가 MCQ 품질 지표를 예측하는 데 강력한 힘을 발휘하는 것으로 나타났습니다.



### Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks (https://arxiv.org/abs/2407.15532)
Comments:
          37 pages, 7 figures, v1

- **What's New**: 본 연구는 주식 시장에서 투자자들이 자산의 개별 성과를 평가하는 것 외에도 포트폴리오로서 기업 집합의 집합적 성과를 고려해야 한다는 점에 주목합니다. 기존의 Markowitz 기반 평균-분산 포트폴리오가 널리 사용되지만, 네트워크 기반 최적화 기법은 이러한 개발을 기반으로 구축되었습니다. 그러나 대부분의 연구는 부도 위험이 있는 기업을 포함하지 않고 특정 기간 동안 지수에서 제외된 모든 기업을 제거합니다. 이는 위험 기업을 통합하고 포트폴리오 최적화에 모든 기업을 사용하는 최초의 연구입니다. 이 연구에서는 그래프 신경망(GNN)의 하위 집합인 그래프 어텐션 네트워크(GAT)를 활용하는 새로운 방법을 제안하고 실증적으로 검증합니다. GNN은 딥 러닝 기반 모델로서 네트워크 데이터를 활용하여 비선형 관계를 파악할 수 있습니다. 고차원 특징을 처리하고 특정 목적에 맞게 사용자 지정 레이어를 수용할 수 있는 기능은 중소형 주식 포트폴리오 최적화와 같은 대규모 문제에 특히 매력적입니다. 이 연구는 중소형 주식에 대한 30년간의 데이터를 활용하여 거리 상관 관계와 삼각형 최대 필터링 그래프 접근 방식을 사용하여 기업의 그래프를 만듭니다. 이러한 그래프는 GAT 모델에 대한 입력이며, 이 모델은 가중치 및 할당 제약 조건을 부과하는 사용자 지정 레이어와 샤프 비율에서 파생된 손실 함수를 사용하여 훈련됩니다. 따라서 포트폴리오 위험 조정 수익을 직접 극대화합니다. 이 새로운 모델은 네트워크 특성 기반 포트폴리오, 평균 분산 기반 포트폴리오 및 동일 가중 포트폴리오와 비교됩니다. 결과는 GAT 기반 모델이 생성한 포트폴리오가 모든 벤치마크를 능가하고 장기간에 걸쳐 다른 전략보다 일관되게 우수하며 시장 역학을 잘 나타낸다는 것을 보여줍니다.



### Synthetic Image Learning: Preserving Performance and Preventing Membership Inference Attacks (https://arxiv.org/abs/2407.15526)
- **What's New**: 본 논문은 지식 재활용(Knowledge Recycling, KR) 파이프라인을 소개하여 의료 분야와 같은 데이터 부족 및 프라이버시 문제 해결에 도움을 주는 인공지능 기반 합성 데이터 생성 및 활용을 개선합니다. 이 파이프라인의 핵심은 생성 지식 증류(Generative Knowledge Distillation, GKD) 기술로, 합성 데이터셋 재생성 및 소프트 라벨링 메커니즘을 통해 분류기가 합성 데이터로부터 얻을 수 있는 정보의 질과 유용성을 향상시킵니다.



### TOM: A Development Platform For Wearable Intelligent Assistants (https://arxiv.org/abs/2407.15523)
Comments:
          14 pages, 6 figures, 2 tables

- **What's New**: 이 논문은 사용자의 작업 수행 능력을 향상시키기 위해 맥락에 맞는 사용자 및 환경 인식 기능을 갖춘 지능형 웨어러블 어시스턴트 개발을 지원하는 TOM이라는 개념적 아키텍처와 소프트웨어 플랫폼을 소개합니다. TOM은 사용자, 연구자 및 개발자의 요구 사항을 파악하여 개발되었으며, 일상 활동에 대한 지능형 보조 AR 애플리케이션 생성을 용이하게 하고 사용자 상호 작용 기록 및 분석, 새로운 장치 통합 및 다양한 활동에 대한 지원 제공을 지원합니다. 또한 몇 가지 개념 증명 보조 서비스를 보여주고 이러한 서비스를 개발하는 데 따른 과제를 논의합니다.



### Future-Proofing Mobile Networks: A Digital Twin Approach to Multi-Signal Managemen (https://arxiv.org/abs/2407.15520)
Comments:
          A shortened version of this paper is currently under review for publication in an IEEE magazine. If accepted, the copyright will be transferred to IEEE

- **What's New**: 이 논문은 네트워크 관리를 향상시키는 데 있어 Digital Twin (DT)의 잠재력을 강조하며, 특히 다양한 네트워크 접근 기술을 활용하여 성능과 관리를 개선하는 DT 프레임워크를 제안합니다. 이 프레임워크는 캠퍼스 네트워크 환경에서 다양한 데이터 소스를 통합하여 실시간 네트워크 성능 및 환경 감지에 대한 통합적인 인사이트를 제공합니다. 또한, 기존 분석 방식을 발전시켜 생성형 AI (GenAI)와 같은 새로운 AI 모델을 활용하고, 현재 분석 기능을 활용하여 분석 프로세스를 단순화합니다. 이는 기술을 통합하여 기술적, 진단적, 예측적, 처방적 분석을 가능하게 하는 고급 ML 모델을 활용할 수 있습니다. 마지막으로, 이 논문은 DT 기술의 발전과 진화된 AI 통합을 조화시키는 상호 운용성 측면과 관련된 특정 연구 기회를 제시합니다.



### Increasing the Robustness of Model Predictions to Missing Sensors in Earth Observation (https://arxiv.org/abs/2407.15512)
Comments:
          Accepted at the MACLEAN workshop in the ECML/PKDD 2024

- **What's New**: 이 논문은 다중 센서 지구 관측 (EO) 모델에서 누락된 데이터 문제를 해결하기 위한 두 가지 새로운 방법, 즉 입력 센서 드롭아웃 (ISensD) 및 앙상블 센서 불변 (ESensI)을 제안합니다. 이러한 방법은 모델 예측의 로버스트성을 향상시키는 데 효과적입니다.



### In-Context Learning Improves Compositional Understanding of Vision-Language Models (https://arxiv.org/abs/2407.15487)
- **What's New**: 본 논문은 Vision-Language Models(VLMs)의 Compositional Image Understanding 능력을 향상시키기 위한 새로운 방법을 제안합니다. 이 방법은 In-Context Learning(ICL)을 활용하여 VLMs의 복잡한 추론 및 이미지 이해 능력을 향상시킵니다.



### A Multi-Level Corroborative Approach for Verification and Validation of Autonomous Robotic Swarms (https://arxiv.org/abs/2407.15475)
Comments:
          15 pages, 11 figures

- **What's New**: 이 논문은 자율 로봇 군집(swarm)의 신뢰성을 높이기 위해 다층 모델링(multi-level modeling) 기반의 공식 검증 및 검증(V&V) 접근 방식을 제안합니다. 이 방법은 거시적 공식 모델링(macroscopic formal modeling), 저 충실도 시뮬레이션(low-fidelity simulation), 고 충실도 시뮬레이션(high-fidelity simulation), 실제 로봇(real-robot) 수준에서 군집의 동작을 모델링하여 분석합니다. 특히, 공식적 거시적 모델은 실제 시뮬레이션에서 얻은 데이터로 구성되어, 다양한 시스템 모델 간의 정확성과 추적성(traceability)을 보장합니다.  더욱이, 이 연구는 공식 검증과 실제 로봇을 이용한 실험 검증을 결합하여 V&V 방법론을 상호 보완적으로 활용하고, 이를 통해 얻은 증거에 대한 신뢰도를 높입니다.



### Pre-Training and Prompting for Few-Shot Node Classification on Text-Attributed Graphs (https://arxiv.org/abs/2407.15431)
Comments:
          Accepted to KDD'24

- **What's New**: 본 논문에서는 텍스트와 그래프 정보를 함께 활용하는 새로운 few-shot node classification 프레임워크인 P2TAG를 제안합니다. P2TAG는 그래프 프리트레이닝과 프롬프팅을 사용하여 텍스트-속성 그래프 (TAG)에서 few-shot node classification을 수행합니다. 기존 방법들은 전처리된 노드 특징에만 의존했지만, P2TAG는 텍스트 정보를 활용하여 few-shot 학습 성능을 향상시킵니다.



### Decoding BACnet Packets: A Large Language Model Approach for Packet Interpretation (https://arxiv.org/abs/2407.15428)
Comments:
          12 pages

- **What's New**: 본 논문은 산업 제어 시스템(ICS) 환경에서 사용되는 다양한 통신 프로토콜을 이해하기 쉽게 요약해주는 소프트웨어 솔루션을 제안합니다. 특히, BACnet 프로토콜에 집중하여, 패킷 파일 데이터를 처리하고 매핑 데이터베이스를 사용하여 컨텍스트를 추출하고, Retrieval Augmented Generation (RAG)을 위한 최신 컨텍스트 검색 방법을 활용합니다. 처리된 패킷 정보와 추출된 컨텍스트는 LLM (Large Language Model)에 입력되고, 사용자를 위한 간결한 패킷 파일 요약을 생성합니다. 이 소프트웨어는 네트워크 활동을 명확하고 일관성 있게 요약하여 SOC 분석가가 제어 시스템의 현재 상태를 더 잘 평가할 수 있도록 지원합니다.



### YOLO-pdd: A Novel Multi-scale PCB Defect Detection Method Using Deep Representations with Sequential Images (https://arxiv.org/abs/2407.15427)
- **What's New**: 본 논문은 PCB 결함 검출을 위한 고정밀, 강력하고 실시간 엔드-투-엔드 방식을 제안합니다. 기존 방식들은 낮은 정확도와 적용 가능성에 제한이 있었지만, 이 논문에서는 YOLOv5와 다중 스케일 모듈을 결합하여 계층적 잔차 연결 (hierarchical residual-like connections) 을 이용한 새로운 접근 방식을 제안합니다. 특히 YOLOv5 모델은 실시간 처리와 정확한 객체 검출 능력을 제공하며, 다중 스케일 모듈은 단일 블록 내에서 계층적 잔차 연결을 통합하여 다중 스케일 특징 추출을 가능하게 하여 다양한 크기와 복잡성의 결함을 식별하는 데 도움을 줍니다.  본 연구에서는 다중 스케일 아키텍처를 통해 특징 추출, 결함 위치 찾기 및 분류를 통합한 네트워크를 구축했습니다. 대규모 PCB 데이터셋을 사용한 실험 결과, 기존 방법들과 비교하여 정확도, 재현율 및 F1 점수가 크게 향상되었음을 보여줍니다. 본 연구는 PCB 결함 검출을 위한 컴퓨터 비전 검사를 발전시키고, PCB 제조 산업에서 고정밀, 강력하고 실시간이며 도메인 적응형 결함 검출을 위한 신뢰할 수 있는 솔루션을 제공합니다. 

- **Technical Details**: YOLOv5, 다중 스케일 모듈, 계층적 잔차 연결, 다중 스케일 특징 추출

- **Performance Highlights**: 기존 방법들과 비교하여 정확도, 재현율 및 F1 점수가 크게 향상됨을 보여줍니다.



### Integrating IP Broadcasting with Audio Tags: Workflow and Challenges (https://arxiv.org/abs/2407.15423)
Comments:
          Submitted to DCASE 2024 Workshop

- **What's New**: 이 논문은 IP 방송 환경에서 실시간 오디오 태깅 모델을 마이크로 서비스로 패키징하는 방법을 제시합니다. 마이크로 서비스는 다양한 네트워크 환경에 통합 가능한 작고 분리된 코드 모듈로, 자동 캡션 생성, 원치 않는 소리 이벤트 식별 등 다양한 활용 가능성을 제공합니다. 이를 통해 소규모 제작부터 대기업까지 모든 규모의 방송 워크플로우에 원활하게 배포할 수 있는 모듈식, 접근 가능하고 유연한 도구를 개발하는 것을 목표로 합니다.



### Planning behavior in a recurrent neural network that plays Sokoban (https://arxiv.org/abs/2407.15421)
Comments:
          Mechanistic Interpretability workshop, ICML 2024

- **What's New**: 이 논문은 Sokoban 게임을 하는 딥 리커런트 뉴럴 네트워크(RNN)를 통해 생각하는 시간의 중요성과 기계 학습에서 계획 능력(planning capabilities)을 촉진하는 방법을 연구합니다. 특히, RNN이 생각하는 시간을 늘리면서 계획을 더 잘 수행하는 것을 발견하고 이 현상을 '페이싱(pacing)'이라고 부릅니다. 또한, 이 모델의 작은 크기(1.29M 파라미터)와 흥미로운 행동은 메커니즘 해석(mechanistic interpretability)을 위한 훌륭한 모델 유기체(model organism)로 만들어줍니다.



### Automated Road Safety: Enhancing Sign and Surface Damage Detection with AI (https://arxiv.org/abs/2407.15406)
Comments:
          16 pages, 10 figures

- **What's New**: 이 논문은 도로 표지판과 도로 표면 손상을 감지하고 분류하여 도로 안전을 향상시키는 혁신적인 접근 방식을 제시합니다. 이 통합 시스템은 이탈리아 경제 성장부(MIMIT)가 지원하는 Casa delle Tecnologie Emergenti (House of Emergent Technologies) Molise (Molise CTE) 연구 프로젝트의 일환으로 개발되었습니다. 이 시스템은 클라우드 컴퓨팅과 GPU 활용을 통한 고성능 컴퓨팅과 같은 최첨단 기술을 활용합니다. 이 시스템은 지자체에 귀중한 도구가 되어 이상 현상을 빠르게 감지하고 유지 보수 작업을 신속하게 조직할 수 있도록 지원합니다.



### Offline Imitation Learning Through Graph Search and Retrieva (https://arxiv.org/abs/2407.15403)
Comments:
          Robotics: Science and Systems (RSS) 2024

- **What's New**: 이 논문은 MCQ 자동 생성 평가 메트릭으로 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 제대로 평가하는지 측정합니다. KDA는 학생들의 응답을 통해 측정되고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭으로 근사화됩니다. 이 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 방식을 모방합니다. 실험 결과 KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성 간에 강력한 상관관계가 있는 것으로 나타났습니다.



### Tackling Selfish Clients in Federated Learning (https://arxiv.org/abs/2407.15402)
Comments:
          10 pages, 16 figures. European Conference on Artificial Intelligence (ECAI) 2024

- **What's New**: 이 논문은 MCQ 생성의 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가할 수 있는지 측정합니다. KDA는 Human evaluation을 통해 측정할 수 있으며, KDA_disc와 KDA_cont는 pretrained language model을 사용하여 KDA를 자동으로 근사화합니다. KDA_disc와 KDA_cont는 Human evaluation과 실제 강의실에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가 평가된 MCQ 품질 척도를 예측하는데 강력한 성능을 보입니다.



### Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation (https://arxiv.org/abs/2407.15396)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 Scene Graph Generation (SGG) 분야에서 '단일 술어 (predicate)'의 다양한 의미 (semantic diversity) 를 고려한 새로운 프레임워크를 제안한다. 기존 SGG 모델들은 각 주어-목적어 쌍에 대해 단일 술어만 예측하도록 학습되어, 술어의 다양한 의미를 간과하고 편향된 예측을 할 수 있었다. 본 논문의 DPL (Semantic Diversity-aware Prototype-based Learning) 프레임워크는 각 술어에 대한 프로토타입 (prototype)을 이용하여 술어의 의미 공간 (semantic space) 을 학습하고, 다양한 의미를 구분하여 보다 정확하고 편향되지 않은 예측을 가능하게 한다.



### Towards Robust Vision Transformer via Masked Adaptive Ensemb (https://arxiv.org/abs/2407.15385)
Comments:
          9 pages

- **What's New**: 본 논문에서는 Vision Transformers (ViT)의 robustness를 향상시키기 위한 새로운 ViT 아키텍처를 제안합니다. 이 아키텍처는 탐지기(detector)와 분류기(classifier)를 적응형 앙상블(adaptive ensemble)로 연결하여 구성됩니다. 탐지기는 adversarial examples를 감지하고, 분류기는 깨끗한 이미지와 adversarial examples에서 추출한 시각적 표현을 각각 사용하여 분류를 수행합니다. 적응형 앙상블은 두 엔코더에서 추출한 시각적 표현의 비율을 조정하여 정확한 분류를 수행합니다.



### A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Mod (https://arxiv.org/abs/2407.15362)
Comments:
          44 pages, 9 figures

- **What's New**: 본 논문은 조직 병리학 (Computational Pathology, CPath) 분야에서 다양한 임상 작업의 성능을 향상시키는 작업-독립적인 기반 모델 (task-agnostic foundation model)을 위한 새로운 방법을 제시합니다. 이 방법은 기존의 비전-전용 (vision-only) 또는 비전-캡션 (vision-captions) 데이터만 사용하는 것과 달리, 병리 보고서 (pathology reports)와 유전자 발현 프로필 (gene expression profiles)과 같은 귀중한 다중 모달 (multimodal) 데이터를 통합합니다. 또한, 기존의 CPath 모델들이 패치 수준 (patch level)에 초점을 맞춘 것과 달리, 본 논문은 전체 슬라이드 수준 (whole-slide level)에서 다중 모달 정보를 활용하여 모델을 사전 훈련하는 새로운 패러다임인 mSTAR (Multimodal Self-TAught PRetraining)를 제안합니다. 이를 통해 CPath 모델은 전체 슬라이드 수준의 맥락을 이해할 수 있게 됩니다.



### X-Recon: Learning-based Patient-specific High-Resolution CT Reconstruction from Orthogonal X-Ray Images (https://arxiv.org/abs/2407.15356)
- **What's New**: 이 연구는 흉부 X선 영상을 이용하여 CT 영상을 초저밀도로 재구성하는 새로운 학습 기반 네트워크인 X-Recon을 제안합니다. X-Recon은 여러 스케일의 융합 렌더링 모듈(MFusionRen)을 갖춘 생성기와 판별기에서 기존의 합성곱 계층을 대체하는 3D 좌표 합성곱 계층을 포함하는 생성적 적대적 네트워크(GAN)을 사용합니다.  X-Recon은 기존의 방법보다 훨씬 높은 재구성 해상도를 달성하며, 이는 초저밀도 3D 단층 영상 재구성 분야에서 새로운 최첨단 수준을 설정합니다. 또한, 연구진은 재구성된 CT 이미지의 품질을 평가하기 위해 영상 처리 기법과 딥 러닝 모델을 결합한 영점 폐탈(Zero-shot) 흉막 탈출(pneumothorax) 분할 파이프라인인 PTX-Seg를 제안했습니다. 



### Robust personalized pricing under uncertainty of purchase probabilities (https://arxiv.org/abs/2407.15332)
- **What's New**: 본 논문은 개인화된 가격 책정 모델이 예상 수익을 극대화하기 위해 노력하는 데 중점을 둡니다. 각 고객의 구매 가능성을 예측하는 것은 개인화된 가격 책정에 필수적이지만, 예측된 값은 본질적으로 피할 수 없는 오류를 포함하고 있으며, 이는 실제 수익에 부정적인 영향을 미칠 수 있습니다. 이러한 문제를 해결하기 위해 본 논문은 불확실성 하에서 최적화 문제에 대한 신뢰할 수 있는 솔루션을 제공하는 견고한 최적화 기법에 중점을 둡니다. 특히, 본 논문은 예측된 구매 가능성의 불확실성을 고려하는 개인화된 가격 책정을 위한 견고한 최적화 모델을 제안합니다. 이 모델은 혼합 정수 선형 최적화 문제로 공식화할 수 있으며, 수학적 최적화 해결사를 사용하여 정확하게 해결할 수 있습니다. 또한 본 논문은 대규모 최적화 문제에 대한 고품질 솔루션을 효율적으로 찾기 위해 라그랑주 분해 알고리즘과 선형 검색을 결합한 방법을 개발했습니다. 실험 결과는 견고한 최적화 모델의 효과를 보여주며, 라그랑주 분해 알고리즘의 계산 효율성과 솔루션 품질 측면에서 유용성을 강조합니다.



### Edge Graph Intelligence: Reciprocally Empowering Edge Networks with Graph Intelligenc (https://arxiv.org/abs/2407.15320)
Comments:
          38 pages, 14 figures

- **What's New**: 이 논문은 그래프 지능 (GI, Graph Intelligence) 모델과 엣지 네트워크 사이의 상호 작용, 즉 엣지 GI (EGI, Edge Graph Intelligence) 라는 새로운 분야를 소개합니다. EGI는 그래프 표현 학습과 엣지 네트워크 간의 상호 작용을 통해 엣지 컴퓨팅의 잠재력을 최대한 발휘할 수 있는 유망한 솔루션으로 주목받고 있습니다.



### AI as a Tool for Fair Journalism: Case Studies from Malta (https://arxiv.org/abs/2407.15316)
Comments:
          Accepted as a full paper in the proceedings of the IEEE 2024 Conference on Artificial Intelligence

- **What's New**: 이 논문은 멀타의 미디어 시장을 중심으로 두 가지 사례 연구를 통해 AI가 사회적 관점과 언론의 진실성을 형성하는 역할을 보여줍니다. 이 두 프로젝트는 미디어 모니터링에 초점을 맞추며 뉴스 기사와 텔레비전 뉴스 세그먼트의 잠재적인 편향을 분석하도록 설계된 도구를 제공합니다. 첫 번째 프로젝트는 컴퓨터 비전 및 자연어 처리 기술을 사용하여 뉴스 기사의 이미지와 해당 캡션, 헤드라인 및 기사 본문 간의 일관성을 분석합니다. 두 번째 프로젝트는 뉴스 비디오에서 개인의 화면 시간 또는 시각적 노출을 추적하는 컴퓨터 비전 기술을 사용하여 쿼리 가능한 데이터를 제공합니다. 이러한 이니셔티브는 언론인과 대중 모두에게 편향을 식별할 수 있는 수단을 제공함으로써 사회에 기여하는 것을 목표로 합니다. 또한, 편향을 감지하고 줄이기 위한 강력한 도구를 제공함으로써 언론 매체의 신뢰성을 향상시키기 위해 언론인이 이러한 도구를 사용할 수 있도록 합니다.



### FMDNN: A Fuzzy-guided Multi-granular Deep Neural Network for Histopathological Image Classification (https://arxiv.org/abs/2407.15312)
Comments:
          This paper has been accepted by IEEE Transactions on Fuzzy Systems for publication. Permission from IEEE must be obtained for all other uses, in any current or future media. The final version is available at [doi: https://doi.org/10.1109/TFUZZ.2024.3410929]

- **What's New**: 자동 MCQ 생성의 교육적 가치를 고려한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다.



### Enhancing Hardware Fault Tolerance in Machines with Reinforcement Learning Policy Gradient Algorithms (https://arxiv.org/abs/2407.15283)
- **What's New**: 이 논문은 강화 학습 기반의 로봇 제어를 활용하여 기계의 하드웨어 오류 허용 능력을 향상시키는 새로운 방법을 제안합니다. 기존의 오류 허용 방법은 구성 요소를 복제하고 오류 발생 시 알고리즘적으로 재구성하는 방식이었지만, 이 연구는 강화 학습을 활용하여 기계가 오류에 적응할 수 있도록 합니다.



### Unifying Invariant and Variant Features for Graph Out-of-Distribution via Probability of Necessity and Sufficiency (https://arxiv.org/abs/2407.15273)
- **What's New**: 본 논문은 그래프 데이터에 대한 Out-of-Distribution (OOD) 문제를 해결하기 위해 Probability of Necessity and Sufficiency (PNS)를 활용하여 불변적인 서브 그래프를 추출하는 새로운 방법을 제안합니다. 특히, 그래프 데이터 생성 프로세스를 고려하여 필요 충분 조건을 만족하는 불변적인 서브 그래프를 추출하는 방법을 제시합니다. 이를 위해, PNS 이론을 기반으로 상한선을 최소화하여 불변적인 서브 그래프를 추출합니다. 이론과 알고리즘을 연결하기 위해, SNIGL (Sufficiency and Necessity Inspired Graph Learning) 모델을 설계하여 잠재적인 필요 충분 조건을 만족하는 불변적인 서브 그래프를 기반으로 불변적인 서브 그래프 분류기를 앙상블하고, 테스트 도메인에 특화된 도메인 변형 서브 그래프 분류기를 사용하여 일반화 성능을 향상시킵니다.



### Genetic Algorithm to Optimize Design of Micro-Surgical Scissors (https://arxiv.org/abs/2407.15243)
Comments:
          Accepted for presentation at the International Conference on Manipulation, Automation and Robotics at Small Scales (MARSS) 2024, Delft, Netherlands

- **What's New**: 본 논문에서는 미세 외과 수술용 가위의 성능을 향상시키기 위해 유전 알고리즘을 사용한 새로운 설계를 제안합니다. 특히, 자석의 최적 위치와 자기 모멘트의 방향을 결정하여 더 강력한 절단 힘을 생성합니다. 이를 통해 기존 설계보다 1.65배 향상된 58 mN의 절단 힘을 달성했습니다.



### Variational Potential Flow: A Novel Probabilistic Framework for Energy-Based Generative Modelling (https://arxiv.org/abs/2407.15238)
- **What's New**: 본 논문은 기존 에너지 기반 모델(EBM)의 단점인 불안정하고 시간 소모적인 MCMC 샘플링을 없애고, 잠재 에너지 함수의 기울기(흐름)를 사용하여 사전 샘플을 데이터 분포에 맞춰 진화시키는 새로운 VAPO(Variational Potential Flow) 프레임워크를 제안한다. VAPO는 데이터 분포와 사전 분포의 쿨백-라이블러 발산을 최소화하는 방식으로 학습하며, 이를 통해 안정적이고 빠른 이미지 생성을 가능하게 한다.



### PUFFLE: Balancing Privacy, Utility, and Fairness in Federated Learning (https://arxiv.org/abs/2407.15224)
- **What's New**: 본 논문에서는 Federated Learning (FL) 환경에서 신뢰성 (trustworthiness)을 위한 새로운 접근 방식인 PUFFLE를 제안합니다. PUFFLE은 FL에서 효용성 (utility), 프라이버시 (privacy), 공정성 (fairness)의 균형을 조절하는 데 도움이 되는 고수준 매개변수화된 접근 방식입니다. 이는 기존 연구에서 흔히 간과되었던 세 가지 요소를 모두 고려하여, 신뢰성 있는 머신러닝 모델을 개발하는 데 중요한 의미를 갖습니다.



### Explainability Paths for Sustained Artistic Practice with AI (https://arxiv.org/abs/2407.15216)
Comments:
          In Proceedings of Explainable AI for the Arts Workshop 2024 (XAIxArts 2024) arXiv:2406.14485

- **What's New**: 이 논문은 AI 기반 생성 오디오의 설명 가능성을 향상시키는 방법을 탐구합니다. 특히, 생성 오디오 모델의 교육 및 구현에서 연구-창작 실무를 기반으로 합니다.



### Flow as the Cross-Domain Manipulation Interfac (https://arxiv.org/abs/2407.15208)
- **What's New**: 본 논문은 다양한 데이터 소스에서 로봇이 조작 기술을 습득할 수 있도록 하는 확장 가능한 학습 프레임워크인 Im2Flow2Act를 소개합니다. Im2Flow2Act의 핵심 아이디어는 객체 흐름 (object flow)을 조작 인터페이스로 사용하여 서로 다른 구현 (예: 사람과 로봇) 및 훈련 환경 (예: 실제 세계 및 시뮬레이션) 간의 도메인 차이를 해소하는 것입니다. Im2Flow2Act는 흐름 생성 네트워크 (flow generation network)와 흐름 조건 정책 (flow-conditioned policy)의 두 가지 구성 요소로 구성됩니다. 사람의 시연 비디오로 훈련된 흐름 생성 네트워크는 작업 설명을 조건으로 하여 초기 장면 이미지에서 객체 흐름을 생성합니다. 시뮬레이션된 로봇 플레이 데이터로 훈련된 흐름 조건 정책은 생성된 객체 흐름을 로봇 동작에 매핑하여 원하는 객체 움직임을 실현합니다. 흐름을 입력으로 사용함으로써 이 정책은 실제 세계에 최소한의 시뮬레이션-실제 차이로 직접 배포될 수 있습니다. 실제 세계 사람 비디오와 시뮬레이션된 로봇 플레이 데이터를 활용함으로써 실제 세계에서 실제 로봇을 원격 조작하는 어려움을 해결하여 다양한 작업을 위한 확장 가능한 시스템을 구축합니다. 본 연구에서는 강체, 관절형 및 변형 가능한 객체의 조작을 포함한 다양한 실제 세계 작업에서 Im2Flow2Act의 기능을 입증합니다.



### Exploiting Pre-trained Models for Drug Target Affinity Prediction with Nearest Neighbors (https://arxiv.org/abs/2407.15202)
Comments:
          Accepted by 33rd ACM International Conference on Information and Knowledge Management 2024 (CIKM 2024)

- **What's New**: 이 논문은 약물-표적 결합 친화력 (DTA) 예측을 위한 새로운 방법인 kNN-DTA를 제안합니다. 이 방법은 기존의 DTA 예측 모델에 기반하여, 비모수적 임베딩 기반 검색 방법을 사용하여 성능을 향상시키는 것을 목표로 합니다. kNN-DTA는 임베딩 공간과 레이블 공간에서 두 가지 이웃 집계 방식을 통합하여 기존 방법과 차별화됩니다. 특히, 쌍별 검색을 통한 레이블 집계와 점별 검색을 통한 표현 집계를 제안합니다. 이 방법은 추론 단계에서 실행되며 훈련 비용 없이 효율적으로 DTA 예측 성능을 향상시킬 수 있습니다. 또한, 경량 학습을 통한 인스턴스별 적응형 집계 방식인 Ada-kNN-DTA를 확장하여 제안합니다.



### HyperbolicLR: Epoch insensitive learning rate scheduler (https://arxiv.org/abs/2407.15200)
Comments:
          26 pages, 7 figures

- **What's New**: 이 논문은 훈련 에포크 수가 달라져도 일관성 있는 학습 곡선을 유지하는 두 가지 새로운 학습률 스케줄러 (learning rate scheduler)인 하이퍼볼릭 학습률 스케줄러(HyperbolicLR)와 지수 하이퍼볼릭 학습률 스케줄러(ExpHyperbolicLR)를 제안합니다. 기존 스케줄러들은 훈련 에포크 수가 변경되면 학습 곡선이 일관성을 잃는 문제가 있었는데, 하이퍼볼릭 곡선의 점근적 특성을 활용하여 이러한 문제를 해결합니다. HyperbolicLR은 에포크-학습률 공간에 직접 이러한 특성을 적용하는 반면, ExpHyperbolicLR은 에포크와 학습률의 지수 공간에 이러한 개념을 매핑합니다. 이러한 스케줄러의 성능을 평가하기 위해, 먼저 소수의 에포크에서 각 스케줄러에 대한 최적의 하이퍼파라미터를 찾았으며, 이러한 값을 고정한 후 에포크 수가 증가할 때 성능을 비교했습니다.



### Error Detection and Constraint Recovery in Hierarchical Multi-Label Classification without Prior Knowledg (https://arxiv.org/abs/2407.15192)
- **What's New**: 본 논문은 MCQ 생성 평가 메트릭의 한계를 극복하기 위해 **지식 종속 가능성(KDA)**이라는 새로운 자동 평가 메트릭을 제안한다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하며, 학생 응답을 기반으로 계산된다. 또한, KDA를 근사화하기 위해 두 가지 자동 평가 메트릭인 **KDA_disc**와 **KDA_cont**를 제시한다. 이들은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 추정한다.



### TADA: Temporal Adversarial Data Augmentation for Time Series Data (https://arxiv.org/abs/2407.15174)
- **What's New**: 이 논문에서는 기존의 MCQ 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성 (answerability)을 측정하여 학생의 지식을 평가하는 능력을 측정합니다. 이를 위해 학생 응답을 이용한 KDA 측정 방법과 pre-trained language model을 활용하여 KDA를 근사한 두 개의 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제시합니다.  



### Mitigating Deep Reinforcement Learning Backdoors in the Neural Activation Spac (https://arxiv.org/abs/2407.15168)
Comments:
          11 Pages, 12 figures

- **What's New**: 본 논문은 딥 강화 학습(DRL) 에이전트 정책에서 백도어의 위협을 조사하고 런타임에서 백도어를 탐지하기 위한 새로운 방법을 제안합니다. 연구는 분포 내에 숨겨진 백도어 트리거에 초점을 맞춥니다. 이러한 트리거는 백도어 에이전트의 행동을 유도하도록 설계되었지만 탐지를 피하기 위해 예상 데이터 분포에 혼합됩니다. 아타리 브레이크아웃 환경에서 수행된 실험을 통해, 현재 위생 방법이 이러한 트리거에 직면했을 때 제한 사항을 보여주고 왜 이러한 방법이 어려운 방어 문제를 제시하는지 조사합니다. 그런 다음 백도어 트리거가 DRL 에이전트의 정책 네트워크의 신경 활성화 공간에서 탐지하기 더 쉬울 수 있다는 가설을 평가합니다. 통계 분석 결과, 트리거가 환경에 얼마나 잘 숨겨져 있든 관계없이 에이전트의 정책 네트워크의 활성화 패턴이 트리거가 있을 때 분명히 다르다는 것을 보여줍니다. 이를 바탕으로 깨끗한 환경 샘플로 훈련된 분류기를 사용하고 비정상적인 활성화를 탐지하는 새로운 방어 접근 방식을 제안합니다. 결과는 가벼운 분류기조차도 상당한 정확도로 악의적인 행동을 효과적으로 방지할 수 있음을 보여주며, 이는 정교한 적대자에 대항하여도 이 연구 방향의 잠재력을 나타냅니다.



### FFHFlow: A Flow-based Variational Approach for Multi-fingered Grasp Synthesis in Real Tim (https://arxiv.org/abs/2407.15161)
Comments:
          First two authors contributed equally, whose ordering decided via coin-tossing

- **What's New**: 본 논문은 다중 손가락 로봇 손으로 다양하고 정확한 파지(grasp)를 생성하는 문제를 다룹니다. 기존의 Generative Model 기반 접근 방식들은 다중 모드(multi-modal), 고차원 파지 분포를 정확하게 포착하는 데 한계가 있었습니다. 본 논문에서는 Normalizing Flows (NFs)를 기반으로 한 Deep Generative Model (DGM)을 활용하여 이 문제를 해결합니다. NFs는 복잡한 확률 분포를 학습하는 데 유용한 모델입니다. 특히, 불완전한 점 구름을 조건으로 파지 분포를 학습하기 위해 단일 조건부 NFs (cNFs)를 적용한 FFHFlow-cnf 모델을 제안하며, 이를 통해 다양성이 향상되는 것을 확인했습니다. 그러나 잠재 공간의 표현력 제한으로 인해 성능 향상은 제한적이었습니다. 이러한 문제를 해결하기 위해, 본 논문에서는 새로운 Flow 기반 Deep Latent Variable Model (DLVM)인 FFHFlow-lvm을 제안합니다. 이 모델은 더 합리적인 잠재 특징을 생성하여 새로운 객체에 대해 다양하고 정확한 파지를 생성할 수 있도록 지원합니다.  Variational Autoencoders (VAEs)와 달리 제안된 DLVM은 사전 및 우도 분포에 대한 두 개의 cNFs를 활용하여 모드 붕괴(mode collapse) 및 잘못된 사전 설정 문제를 해결합니다. 이러한 cNFs는 일반적으로 등방성 가우시안으로 제한됩니다. 시뮬레이션 및 실제 로봇 환경에서의 광범위한 실험 결과, 제안된 방법이 VAE 기반 모델보다 더 정확하고 다양한 파지를 생성하는 것으로 나타났습니다. 또한, 실시간 애플리케이션을 위한 높은 잠재력을 보여주는 실행 시간 비교가 수행되었습니다.



### Distilling Vision-Language Foundation Models: A Data-Free Approach via Prompt Diversification (https://arxiv.org/abs/2407.15155)
Comments:
          Accepted by ACMMM 2023

- **What's New**: 본 논문은 자동 MCQ 생성 평가 메트릭으로, KDA (Knowledge Dependent Answerability) 라는 새로운 지표를 제안합니다. 기존의 BLEU, ROUGE, METEOR와 달리, KDA는 MCQ가 학생의 지식을 측정하는 능력을 평가합니다. KDA는 human survey를 통해 측정되며, KDA_disc와 KDA_cont라는 자동 평가 지표를 통해 근사화됩니다.  KDA_disc와 KDA_cont는 pre-trained language model을 활용하여 학생의 문제 해결 방식을 모방합니다. Human study를 통해, KDA_disc와 KDA_cont가 실제 강의실 상황에서의 사용성과 강한 상관관계를 보인다는 것을 확인했습니다.



### Rethinking Feature Backbone Fine-tuning for Remote Sensing Object Detection (https://arxiv.org/abs/2407.15143)
Comments:
          Under Review

- **What's New**: 본 논문에서는 원격 감지 객체 탐지 (Remote Sensing Object Detection) 에서 feature backbone fine-tuning을 위한 새로운 방법인 DBF (Dynamic Backbone Freezing)를 제안합니다. 이 방법은 'Freezing Scheduler' 모듈을 통해 backbone feature의 업데이트를 동적으로 관리하여, backbone이 낮은 수준의 일반적인 특징 (low-level generic features)을 추출할지, 또는 원격 감지 도메인에 대한 특정 지식 (specific knowledge)을 갖추어야 할지의 문제를 해결합니다. 



### Proximal Policy Distillation (https://arxiv.org/abs/2407.15134)
- **What's New**: 본 논문에서는 학생 중심 증류와 Proximal Policy Optimization (PPO)를 통합하여 샘플 효율성을 높이고 학생 정책이 증류 과정에서 수집한 추가 보상을 활용하는 새로운 정책 증류 방법인 Proximal Policy Distillation (PPD)를 소개합니다. 이 방법의 효과를 평가하기 위해 다양한 강화 학습 환경(ATARI, Mujoco, Procgen)에서 PPD를 학생 중심 증류와 교사 중심 증류와 비교했습니다. 각 환경과 방법에 대해 교사 네트워크보다 작거나, 동일하거나(자기 증류), 크기가 큰 다양한 목표 학생 신경망 세트에 대한 증류를 수행했습니다. 연구 결과, PPD는 기존 정책 증류 방법보다 샘플 효율성을 높이고 더 나은 학생 정책을 생성하는 것으로 나타났습니다. 또한, PPD는 불완전한 시연에서 정책을 증류할 때 다른 방법보다 뛰어난 견고성을 보여줍니다. 이 논문의 코드는 stable-baselines3을 기반으로 구축된 새로운 Python 라이브러리인 `sb3-distill`의 일부로 공개됩니다.



### Learning Physics for Unveiling Hidden Earthquake Ground Motions via Conditional Generative Modeling (https://arxiv.org/abs/2407.15089)
- **What's New**: 이 논문에서는 지진의 지반 운동(ground motion)을 예측하기 위해 AI 기반 시뮬레이터인 CGM-GM을 제안합니다. CGM-GM은 지진 규모와 지리적 좌표를 입력으로 받아 복잡한 파동 물리(wave physics)와 지구 이질성(Earth heterogeneities)을 학습합니다. 이를 위해 시간-주파수 도메인(time-frequency domain)의 잠재적 분포(latent distributions)를 포착하는 확률적 오토인코더(probabilistic autoencoder)와 사전 및 사후 분포(prior and posterior distributions)를 위한 변분 순차 모델(variational sequential models)을 사용합니다.  



### MaxMI: A Maximal Mutual Information Criterion for Manipulation Concept Discovery (https://arxiv.org/abs/2407.15086)
- **What's New**: 이 논문은 MCQ(객관식 문제) 자동 생성을 위한 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지 측정합니다. 기존 지표인 BLEU, ROUGE, METEOR는 단순히 문장 유사성만을 비교하기 때문에 교육적 가치를 충분히 반영하지 못합니다.



### Learning to Compile Programs to Neural Networks (https://arxiv.org/abs/2407.15078)
- **What's New**: 본 논문은 MCQ 자동 생성 평가 메트릭으로 KDA (Knowledge Dependent Answerability)를 제안하여 MCQ의 답변 가능성을 측정하고, 학생의 대상 사실에 대한 지식을 평가하는 능력을 평가합니다. 기존의 BLEU, ROUGE, METEOR 등의 메트릭은 MCQ의 교육적 가치를 무시하고, 단어 유사도만 비교합니다. KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다. 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계가 있는 것으로 나타났습니다. 또한, n-gram 기반 유사성 측정 지표와 결합하면 다양한 전문가가 평가한 MCQ 품질 측정 지표에 대한 강력한 예측력을 갖는 것으로 나타났습니다.



### MusiConGen: Rhythm and Chord Control for Transformer-Based Text-to-Music Generation (https://arxiv.org/abs/2407.15060)
Comments:
          Accepted by the 25th International Society for Music Information Retrieval (ISMIR)

- **What's New**: 본 논문은 MusiConGen을 소개합니다. MusiConGen은 MusicGen 프레임워크를 기반으로 한, 시간적 조건 (temporally-conditioned) Transformer 기반 텍스트-투-뮤직 모델로, 자동 추출된 리듬과 코드를 조건 신호 (condition signal)로 통합하는 효율적인 미세 조정 (finetuning) 메커니즘을 도입합니다.



### Arondight: Red Teaming Large Vision Language Models with Auto-generated Multi-modal Jailbreak Prompts (https://arxiv.org/abs/2407.15050)
Comments:
          To be published in ACM MM 2024

- **What's New**: 이 논문은 **Arondight**라는 VLM을 위한 새로운 red teaming 프레임워크를 제안합니다. 이 프레임워크는 VLM의 안전성을 평가하는 데 필요한 다양한 이미지와 텍스트 테스트 케이스를 생성하여 VLM의 취약성을 찾아내는 데 목표를 두고 있습니다.



### MedSAGa: Few-shot Memory Efficient Medical Image Segmentation using Gradient Low-Rank Projection in SAM (https://arxiv.org/abs/2407.15042)
- **What's New**: 본 논문은 MedSAGa (Medical Segment Anything Model with Galore)를 소개합니다. MedSAGa는 SAM (Segment Anything Model)의 이미지 인코더 파라미터에 GaLore (Gradient Low-Rank Projection)를 적용하여 메모리 효율적인 few-shot 의료 이미지 분할을 달성합니다. 이 모델은 기존의 의료 이미지 분할 모델보다 적은 데이터로 높은 성능을 보여주며, 특히 자원이 부족한 환경에서 효과적입니다.



### Self-training Room Layout Estimation via Geometry-aware Ray-casting (https://arxiv.org/abs/2407.15041)
Comments:
          Accepted to ECCV-2024

- **What's New**: 이 논문은 레이 캐스팅(ray-casting) 공식을 사용하여 다양한 시점에서 생성된 여러 추정치를 집계함으로써, 라벨링되지 않은 데이터를 사용하여 보이지 않는 장면에서 룸 레이아웃(room layout) 추정 모델을 위한 새로운 지오메트리 인식 자기 학습 프레임워크를 소개한다. 이를 통해 자기 학습을 위한 신뢰할 수 있는 의사 라벨(pseudo-label)을 계산할 수 있다.



### AsyCo: An Asymmetric Dual-task Co-training Model for Partial-label Learning (https://arxiv.org/abs/2407.15036)
Comments:
          15 pages, accepted by Science China, Information Science

- **What's New**: 본 논문에서는 부분 라벨 학습(PLL) 모델에서 발생하는 에러 누적 문제를 해결하기 위해 비대칭 듀얼 태스크 코트레이닝 PLL 모델인 AsyCo를 제안합니다. AsyCo는 두 개의 네트워크, 즉 해소 네트워크(disambiguation network)와 보조 네트워크(auxiliary network)가 서로 다른 뷰에서 명확하게 학습하도록 강제하는 방식으로 작동합니다. 해소 네트워크는 자기 학습 PLL 태스크를 통해 라벨 신뢰도(label confidence)를 학습하고, 보조 네트워크는 학습된 라벨 신뢰도를 기반으로 생성된 잡음이 있는 쌍방향 유사성 라벨을 통해 지도 학습 방식으로 학습합니다. 마지막으로 정보 증류(information distillation)와 신뢰도 개선(confidence refinement)을 통해 에러 누적 문제를 완화합니다.



### Benchmarking End-To-End Performance of AI-Based Chip Placement Algorithms (https://arxiv.org/abs/2407.15026)
Comments:
          A comprehensive benchmark for AI-based chip placement algorithms using end-to-end performance metrics

- **What's New**: 이 논문은 VLSI (Very-Large-Scale Integration) 디자인에서 Chip Placement (칩 배치) 작업을 위한 AI 기반 알고리즘의 효과를 평가할 수 있는 벤치마크인 ChiPBench를 소개합니다. 기존 연구는 중간 단계의 대리 지표 (surrogate metrics)를 사용하여 AI 알고리즘을 평가했으나, ChiPBench는 최종 디자인 PPA (Performance, Power, Area) 지표를 직접 평가할 수 있도록 설계되었습니다.  



### ViT LoS V2X: Vision Transformers for Environment-aware LoS Blockage Prediction for 6G Vehicular Networks (https://arxiv.org/abs/2407.15023)
- **What's New**: 이 논문은 6G 차량 네트워크에서 장애물 (blockage) 을 예측하기 위해 CNN과 ViT를 결합한 딥러닝 기반 접근 방식을 제안합니다. 이 방식은 이미지와 빔 벡터를 포함한 시계열 다중 모드 데이터 (multimodal data) 에서 특징을 추출하기 위해 CNN과 ViT의 장점을 활용합니다. 또한, GRU 기반 아키텍처를 사용하여 추출된 특징과 미래 시간 단계의 장애물 상태 간의 시간 의존성을 포착합니다.  



### Encouraging Responsible Use of Generative AI in Education: A Reward-Based Learning Approach (https://arxiv.org/abs/2407.15022)
Comments:
          9 pages, 4 figures

- **What's New**: 본 연구는 ChatGPT와 같은 생성 AI를 활용하여 빠른 해답보다는 구조화된 학습을 촉진하는 혁신적인 수학 학습 방법을 소개합니다. 챗봇 기능과 생성 AI를 결합하여 상호 작용적인 문제 해결 연습을 제공하여 다양한 문제에 대한 단계별 접근 방식을 통해 학습을 향상시키고 교육에서 AI의 책임감 있는 사용을 옹호합니다. 본 연구는 ChatGPT의 즉각적인 답변이 실제 학습을 방해할 수 있다는 점을 강조합니다. 학생들이 최종 답변을 받기 위해 수학 문제를 효과적으로 풀도록 하는 보상 시스템을 도입했습니다. 이는 기본 문제에서 복잡한 문제로의 점진적인 학습 경로를 장려하고, 최종 솔루션으로 마스터리에 대한 보상을 제공합니다. 목표는 학생들이 빠른 해결책을 찾는 것에서 포괄적인 학습 경험에 적극적으로 참여하는 것으로 전환하는 것입니다.



### Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning (https://arxiv.org/abs/2407.15007)
- **What's New**: 이 논문에서는 지식 종속 가능성(KDA)이라고 불리는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. KDA는 학생들의 응답을 이용한 측정 방식과, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 지표(KDA_disc, KDA_cont)로 나뉩니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하기 위해 학생 응답을 분석하고, 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방하는 방식으로 자동화합니다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었습니다. 또한, 기존 n-gram 기반 유사도 지표와 결합하면 KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 측정 지표에 대한 예측력이 강력함을 보여줍니다.



### Enhancing Microgrid Performance Prediction with Attention-based Deep Learning Models (https://arxiv.org/abs/2407.14984)
Comments:
          2024 11th International Conference on Information Technology, Computer, and Electrical Engineering (ICITACEE)

- **What's New**: 본 연구에서는 마이크로그리드 시스템의 운영상 문제점, 특히 그리드 불안정성을 유발하는 전력 진동 문제를 해결하기 위한 노력을 기울였습니다. 기존의 convolution과 GRU(Gated Recurrent Unit) 레이어의 장점을 결합한 통합 전략을 제시합니다. 이 전략은 에너지 데이터셋에서 시간적 데이터를 효과적으로 추출하여 마이크로그리드 동작 예측의 정확도를 향상시키는 데 중점을 둡니다. 또한 어텐션 레이어를 사용하여 시간 시리즈 데이터 내의 중요한 특징을 강조하여 예측 과정을 최적화합니다. 이 프레임워크는 MLP(Multi-Layer Perceptron) 모델을 기반으로 하며, 포괄적인 부하 예측과 비정상적인 그리드 동작을 식별하는 임무를 수행합니다. 본 연구의 방법론은 마이크로그리드 요금 평가 도구 데이터셋을 사용하여 엄격한 평가를 거쳤으며, RMSE(Root Mean Square Error), MAE(Mean Absolute Error) 및 결정 계수(r2-score)가 주요 지표로 사용되었습니다. 이 방법은 부하 예측에서 MAE 0.39, RMSE 0.28 및 r2-score 98.89%의 우수한 성능을 보였으며, 거의 완벽한 제로 상태 예측 정확도(약 99.9%)를 보였습니다. 지원 벡터 회귀 및 랜덤 포레스트 회귀와 같은 기존 머신 러닝 모델보다 훨씬 뛰어난 성능을 보이는 본 모델의 간소화된 아키텍처는 실시간 애플리케이션에 특히 적합하여 더 효과적이고 안정적인 마이크로그리드 관리를 가능하게 합니다.



### GreenStableYolo: Optimizing Inference Time and Image Quality of Text-to-Image Generation (https://arxiv.org/abs/2407.14982)
Comments:
          This paper is published in the SSBSE Challenge Track 2024

- **What's New**: 이 연구는 Stable Diffusion의 파라미터와 프롬프트를 최적화하여 GPU 추론 시간을 줄이고 이미지 생성 품질을 향상시키는 새로운 방법인 GreenStableYolo를 제안합니다. GreenStableYolo는 NSGA-II와 Yolo를 사용하여 Stable Diffusion의 파라미터와 프롬프트를 최적화합니다. 이를 통해 이미지 품질에 대한 약간의 손실(18%)을 감수하면서도, 추론 시간을 크게 단축(266% 감소)하고, 하이퍼볼륨(Hypervolume)을 526% 향상시켜 텍스트-이미지 생성 분야의 최첨단 기술을 발전시켰습니다.



### Out of spuriousity: Improving robustness to spurious correlations without group annotations (https://arxiv.org/abs/2407.14974)
- **What's New**: 자동 MCQ 생성을 위한 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)가 제안되었습니다. 기존의 BLEU, ROUGE, METEOR와 같은 지표들은 MCQ의 질적 가치를 제대로 평가하지 못했지만, KDA는 MCQ가 학생의 지식을 평가하는 능력을 측정합니다. KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 제안되어 사람의 문제 해결 능력을 모방하는 사전 훈련된 언어 모델을 활용합니다. 

- **Technical Details**: KDA는 학생의 지식 수준과 MCQ의 대답 가능성 간의 관계를 고려하여 MCQ의 질을 측정합니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 KDA를 근사화합니다. 이러한 모델은 학생의 문제 해결 행동을 모방하여 MCQ의 질을 평가합니다.

- **Performance Highlights**: 인간 평가 연구를 통해 KDA_disc와 KDA_cont가 KDA와 강한 상관관계를 가지며 실제 강의실 환경에서의 사용성과도 강한 상관관계를 가짐을 보여주었습니다. 또한, N-gram 기반 유사성 지표와 결합했을 때, KDA_disc와 KDA_cont는 다양한 전문가가 평가한 MCQ 품질 측정 지표를 예측하는 데 강력한 예측력을 보여주었습니다.



### POGEMA: A Benchmark Platform for Cooperative Multi-Agent Navigation (https://arxiv.org/abs/2407.14931)
Comments:
          27 pages, 9 figures

- **What's New**: 본 논문은 MARL(Multi-agent Reinforcement Learning) 기반의 다중 로봇 시스템의 문제를 해결하고자 했고, 이를 위한 폭넓은 도구 모음인 **POGEMA**를 제안했습니다. POGEMA는 다양한 환경에서 로봇이 이동하고 장애물을 피하는 문제를 해결하는 데 사용할 수 있는, **고속 학습 환경**, **문제 생성기**, **사전 정의된 문제 모음**, **시각화 툴킷**, **벤치마킹 도구**로 구성되어 있습니다. 또한, 성공률과 경로 길이를 포함한 다양한 지표를 바탕으로 **평가 프로토콜**을 정의하여, MARL, 탐색 기반, 하이브리드 방법 등 다양한 방법을 **공정하게 비교할 수 있도록 했습니다.**



### Visual Geo-Localization from images (https://arxiv.org/abs/2407.14910)
Comments:
          18 pages, 8 figures,

- **What's New**: 본 논문에서는 GPS 데이터에 의존하지 않고 이미지에서 장소(건물과 도로 교차로)의 지리적 위치를 결정할 수 있는 시각적 지오 로컬라이제이션 시스템을 제시합니다. 이 시스템은 장소 인식을 위한 SIFT(Scale-Invariant Feature Transform), 도로 교차로 유형 식별을 위한 전통적인 이미지 처리, 그리고 도로 교차로 분류를 위한 VGG16 모델을 사용하는 딥 러닝을 결합합니다.



### Inferring Ingrained Remote Information in AC Power Flows Using Neuromorphic Modality Regim (https://arxiv.org/abs/2407.14883)
- **What's New**: 본 논문은 스파이크 신경망(SNN)을 에지 프로세서로 사용하여 AC 전력 흐름에서 내재된 원격 정보를 추론하고, 전력 전자 변환기의 효율적인 조정을 수행하는 새로운 방법을 제안합니다. 이 연구는 에너지 효율적인 뉴로모픽 처리 및 의미론 이론을 사용하여 스파이크 형태의 다중 모드 체제를 통해 전력과 정보를 통합하여 데이터 정규화의 수단으로 활용합니다. 먼저 각 에지에서 동기식 실수값 측정을 구성하고 비동기식 스파이크 기반 이벤트로 변환하여 각 에지에서 SNN 훈련을 위한 희소 데이터를 수집합니다. 오류 의존적 지도 학습 이론에 의존하는 대신, 지연 기반 비지도 헤브 규칙을 활용하여 전력 전자 변환기의 스위칭을 위한 변조 펄스를 얻습니다. 이러한 철학은 사이버 계층을 무시함으로써 사이버 공격자에 대한 외생적 경로 도착을 차단할 뿐만 아니라 시스템 재구성 및 매개변수 불일치 문제에 대한 변환기 적응을 수반합니다. 수정된 IEEE 14-버스 시스템에서 다양한 시나리오와 실험 조건에서 에너지 효율적이고 효과적인 온라인 학습 성능을 검증하여 이 연구를 마무리합니다.



### Reduced Effectiveness of Kolmogorov-Arnold Networks on Functions with Nois (https://arxiv.org/abs/2407.14882)
- **What's New**: 본 논문은 교육적 가치를 고려하지 않는 기존 MCQ 생성 평가 메트릭의 한계를 지적하고, 새로운 지식 종속 가능성(KDA, Knowledge Dependent Answerability) 메트릭을 제안합니다. KDA는 MCQ가 대상 사실을 제대로 평가할 수 있는 능력을 측정합니다. KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하고, Human evaluation을 통해 이러한 메트릭들이 실제 강의실에서 사용 가능한 MCQ를 효과적으로 평가할 수 있음을 보여줍니다.



### Preictal Period Optimization for Deep Learning-Based Epileptic Seizure Prediction (https://arxiv.org/abs/2407.14876)
- **What's New**: 본 논문은 딥러닝 모델을 활용하여 약물 내성 간질 환자의 발작 예측을 위한 새로운 방법론을 제시하며, 특히 발작 예측 작업에서 예측 성능을 종합적으로 평가하는 방법론을 제시합니다. 이를 위해, 연구자들은 CNN-Transformer 딥러닝 모델을 도입하여 발작 전 시공간적 패턴을 감지하고, 최적의 발작 전 기간 (OPP)을 결정하기 위한 새로운 연속 입력-출력 성능 비율 (CIOPR) 지표를 제안합니다. 

- **Technical Details**: 본 연구는 CNN-Transformer 딥러닝 모델을 사용하여 발작 전 시공간적 역동성 (spatiotemporal dynamics)을 감지하고, 새로운 CIOPR 지표를 도입하여 발작 전 기간 (OPP)을 결정하는 데 집중합니다. 이 방법론은 발작 예측 작업에서 예측 성능을 종합적으로 평가하는 방법을 제공하며, 특히 발작 전 기간의 정의가 예측 시간, 정확도, 출력 안정성, 발작 간 및 발작 전 상태 간 전환 시간에 미치는 영향을 정량적으로 평가할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과는 CHB-MIT 데이터셋의 19명의 소아 환자를 대상으로 개별적으로 수행되었습니다. 이 모델은 각 환자의 OPP를 사용하여 발작 전 및 발작 간 구간을 평균적으로 99.31%의 민감도, 95.34%의 특이도, 99.35%의 AUC, 97.46%의 F1-점수로 정확하게 식별했으며, 예측 시간은 평균적으로 발작 시작 76.8분 전이었습니다. 특히, 새로운 CIOPR 지표는 발작 전 기간 정의가 예측 시간, 정확도, 출력 안정성, 발작 간 및 발작 전 상태 간 전환 시간에 미치는 영향을 종합적이고 정량적으로 파악할 수 있었고, 발작 예측에서 환자 간 및 환자 내 변동성을 고려하는 것이 중요하다는 점을 강조했습니다.



### Retrieval Augmented Generation Integrated Large Language Models in Smart Contract Vulnerability Detection (https://arxiv.org/abs/2407.14838)
Comments:
          17 pages, 3 figures, 4 tables

- **What's New**: 본 논문은 스마트 컨트랙트 감사를 위해 Retrieval-Augmented Generation (RAG)과 GPT-4-1106 모델을 결합한 새로운 접근 방식을 제안합니다. 830개의 알려진 취약한 컨트랙트를 벡터 저장소에 저장하고, LangChain을 사용하여 RAG-LLM 파이프라인을 구축합니다.  



### Toward Efficient Convolutional Neural Networks With Structured Ternary Patterns (https://arxiv.org/abs/2407.14831)
Comments:
          Published in: IEEE Transactions on Neural Networks and Learning Systems Code: this https URL ImageNet-16 Dataset: this https URL

- **What's New**: 본 연구는 기존의 MCQ 생성 평가 지표(BLEU, ROUGE, METEOR)의 한계를 극복하고, MCQ의 교육적 가치를 평가하는 새로운 지표인 "지식 종속 가능성(Knowledge Dependent Answerability, KDA)"를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가할 수 있는지를 측정하는 지표입니다.



### CrossDehaze: Scaling Up Image Dehazing with Cross-Data Vision Alignment and Augmentation (https://arxiv.org/abs/2407.14823)
Comments:
          A cross-dataset vision alignment and augmentation technology is proposed to boost generalizable feature learning in the de-hazing task

- **What's New**: 이 논문에서는 이미지 dehazing 방법론을 개선하기 위해 내부 및 외부 데이터 증강을 제안합니다. (internal and external data augmentation). 외부 증강은 서로 다른 도메인에서 샘플을 가져와 모델이 더 강력하고 일반화된 특징을 학습하도록 돕습니다. (cross-data external augmentor). 내부 증강은 이미지 내의 로컬 정보를 활용하여 더 많은 이미지 디테일을 얻습니다.



### Decoupled Prompt-Adapter Tuning for Continual Activity Recognition (https://arxiv.org/abs/2407.14811)
- **What's New**: 본 논문은 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 답변 가능성을 측정하고, 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다. KDA는 인간 설문 조사를 통해 학생의 응답을 기반으로 측정되며, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제시한다. 

- **Technical Details**: KDA는 MCQ가 주어진 대상 사실을 얼마나 잘 평가하는지 측정하는 메트릭으로, 학생들의 응답을 기반으로 측정된다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하여 KDA를 근사화한 자동 평가 메트릭이다. 

- **Performance Highlights**: 본 논문은 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 실제 강의실 환경에서 전문가가 평가한 사용성과 높은 상관관계를 갖는다는 것을 보여준다. 또한, n-gram 기반 유사성 지표와 결합했을 때, KDA_disc와 KDA_cont가 전문가가 평가한 다양한 MCQ 품질 측정 지표에 대한 강력한 예측력을 갖는 것으로 나타났다.



### PASSION: Towards Effective Incomplete Multi-Modal Medical Image Segmentation with Imbalanced Missing Rates (https://arxiv.org/abs/2407.14796)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이 논문은 의료 이미지에서 불완전한 다중 모드 이미지 분할 문제를 다룹니다. 특히, 모드가 임의로 누락될 수 있는 실제 의료 환경에서 발생하는 불균형 누락률 문제를 해결하는 것을 목표로 합니다. 이는 기존의 완전한 모드 데이터가 학습 과정에 사용되는 방식과는 차이가 있습니다.



### Do Generative AI Models Output Harm while Representing Non-Western Cultures: Evidence from A Community-Centered Approach (https://arxiv.org/abs/2407.14779)
Comments:
          This is the pre-peer reviewed version, which has been accepted at the 7th AAAI ACM Conference on AI, Ethics, and Society, Oct. 21, 2024, California, USA

- **What's New**: 본 연구는 인도 문화를 중심으로 이미지 생성 AI(T2I) 모델이 비서구 문화를 어떻게 표현하는지 조사한다. T2I는 이미지 생성에 있어 혁신적이지만, 편향이 발생하여 잘못된 표현과 소외를 야기할 수 있다는 우려가 제기되었다. 연구팀은 인도의 다양한 하위 문화 5개 그룹을 대상으로 포커스 그룹 분석을 진행하여 T2I가 영어 프롬프트에 대한 결과물을 통해 인도 문화와 하위 문화를 어떻게 묘사하는지 조사했으며, 이국화 (exoticism) 및 문화적 도용과 같은 새로운 표현적 피해 (representational harm)를 발견했다. 본 연구는 포커스 그룹 참여자들이 스테레오타입, 특정 하위 문화의 삭제 및 폄하, 서구적 맥락과 비교한 낮은 서비스 품질 등의 기존 표현적 피해를 인지하는 것 외에도, 두 가지 새로운 문화적 피해를 발견했다. 즉, 이국화는 특정 문화의 특징이나 품질을 과장하거나 과도하게 표현하여 문화적으로 정확한 세부 사항을 무시하는 현상을 말하며, 문화적 도용은 문화적으로 특정한 맥락에 대한 세부 사항을 묘사할 때 해당 맥락에 맞지 않는 다른 문화나 하위 문화의 세부 사항을 잘못 포함하는 것을 의미한다. 연구팀은 인도의 다양한 지역 사회 구성원들이 T2I Stable Diffusion 모델의 실제 출력 결과에서 인지한 피해를 문서화하고, 섬세한 문화적 상호 작용과 경험을 정확하게 표현할 수 있는 보다 포괄적이고 문화적으로 민감한 T2I 개발을 위한 설계 원칙을 제시한다.



### Intelligent Artistic Typography: A Comprehensive Review of Artistic Text Design and Generation (https://arxiv.org/abs/2407.14774)
Comments:
          GitHub Page: this https URL

- **What's New**: 자동 MCQ 생성 평가 지표인 KDA(Knowledge Dependent Answerability)를 새롭게 제안, 기존 n-gram 기반 지표의 단점을 보완하여 MCQ의 교육적 가치를 평가한다. KDA는 학생 응답 데이터를 기반으로 계산되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 개발하여 실제 KDA를 근사화한다. 

- **Technical Details**: KDA는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하는 지표로, 학생 응답 데이터를 기반으로 계산된다. KDA_disc와 KDA_cont는 각각 차별적(discriminative) 및 연속적(continuous) 접근 방식을 활용하여 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방한다. 

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA와 강한 상관관계를 보이고, 전문가에 의해 평가된 실제 강의실 환경에서의 사용성과도 상관관계를 나타낸다는 것을 확인했다. 또한 n-gram 기반 유사도 지표와 결합하여 다양한 전문가 평가 MCQ 품질 지표에 대한 예측력이 뛰어남을 확인했다.



### Teach Harder, Learn Poorer: Rethinking Hard Sample Distillation for GNN-to-MLP Knowledge Distillation (https://arxiv.org/abs/2407.14768)
- **What's New**: 이 논문은 Graph Neural Networks (GNNs)와 Multi-Layer Perceptron (MLPs) 간의 차이를 줄이기 위해 GNN-to-MLP Knowledge Distillation (KD)을 제안합니다. 이 방법은 잘 훈련된 teacher GNN에서 student MLP로 지식을 증류합니다. 또한, 이 논문은 teacher GNNs에서의 knowledge sample (노드)을 hardness (난이도)의 관점에서 재검토하고, hard sample distillation이 기존 graph KD 알고리즘의 주요 성능 병목 현상이 될 수 있음을 밝힙니다. GNN-to-MLP KD에는 두 가지 유형의 hardness가 존재하며, 하나는 student-free knowledge hardness로 GNN 지식의 본질적인 복잡성을 나타내고, 다른 하나는 student-dependent distillation hardness로 teacher-to-student distillation의 어려움을 나타냅니다. 하지만 대부분의 기존 연구는 이러한 측면 중 하나에만 집중하거나 두 가지를 동일하게 여겼습니다. 이 논문은 간단하면서도 효과적인 Hardness-aware GNN-to-MLP Distillation (HGMD) 프레임워크를 제안하여 두 가지 hardness를 분리하고 비모수적 접근 방식을 사용하여 추정합니다. 마지막으로, HGMD-weight와 HGMD-mixup이라는 두 가지 hardness-aware distillation scheme을 제안하여 teacher GNNs의 hardness-aware 지식을 student MLPs의 해당 노드로 증류합니다. 비모수적 증류로서, HGMD는 student MLPs 이외에는 학습 가능한 매개변수를 포함하지 않지만, 여전히 최첨단 경쟁자 대부분을 능가합니다.



### Implementing Fairness: the view from a FairDream (https://arxiv.org/abs/2407.14766)
- **What's New**: 본 논문에서는 AI 분류에서의 공정성 문제를 실험적으로 조사합니다. 소득 예측을 사례 연구로 사용하여 AI 모델을 훈련하고 공정성 패키지 FairDream을 개발하여 불평등을 감지하고 수정합니다. 우리의 실험은 알고리즘이 그룹 간 양성을 동등하게 만드는 작업을 수행하도록 설정된 경우에도 FairDream의 특성이 기본 진실 (Equalized Odds)을 조건으로 하는 공정성 목표를 달성하는 데 있다는 것을 보여줍니다. 이는 이상으로 여겨질 수 있지만, 우리는 이러한 특성을 Demographic Parity를 Equalized Odds의 비용으로 강제할 수 있는 관련 공정성 방법 (GridSearch)과의 접근 방식을 비교하여 설명합니다. 우리는 True label을 조건으로 하는 공정성 지표가 공정성에 도달하기 위한 충분한 기준을 제공하지 않는다는 것을 인정하지만, 적어도 Demographic Parity를 신중하게 구현하기 위한 필요 조건을 제공한다고 주장합니다. 또한 Equal Calibration과 Equal Precision이 분류에서 관련 공정성 기준으로 적합하지 않은 이유를 설명합니다. 불리한 비율에 대한 의사 결정자의 경고를 제한하는 데 대한 한계를 해결하기 위해 Equalized Odds는 엄격한 보수주의의 위험을 피하면서 알고리즘을 통한 자원의 전체적인 재분배라는 유토피아를 멀리합니다.



### Data Augmentation in Graph Neural Networks: The Role of Generated Synthetic Graphs (https://arxiv.org/abs/2407.14765)
- **What's New**: 본 논문은 그래프 데이터 증강을 위한 새로운 방법을 제시합니다. 이 방법은 그래프 생성 모델을 사용하여 실제 그래프 데이터에 추가할 새로운 그래프를 생성하며, 생성된 그래프의 수량을 조절하여 그래프 분류 성능을 향상시키는 방식입니다. 특히, 그래프의 크기에 따라 다른 생성기 (generator)를 사용하여 확장성과 품질의 균형을 맞추는 전략을 제시합니다.



### Flatness-aware Sequential Learning Generates Resilient Backdoors (https://arxiv.org/abs/2407.14738)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는  Backdoor 공격에 대한 방어를 위해,  **지속적인 학습(Continual Learning, CL)** 기법을 활용하는 새로운 프레임워크 **Sequential Backdoor Learning (SBL)**을 제안합니다. SBL은 Backdoor poisoning 프로세스를 두 가지 작업으로 분리하여  fine-tuning 방어에 강력한  Backdoor를 만드는 것을 목표로 합니다.  첫 번째 작업은 Backdoor 모델을 학습하고, 두 번째 작업은 CL 원칙을 기반으로  Backdoor 지역을 fine-tuning에 강력하게 만드는 것입니다. 또한,  framework 내에서 **Sharpness-aware minimizer**를 통해  Backdoor 지역을 더 평평하게 만들어  Backdoor의 내구성을 높입니다.



### ECRTime: Ensemble Integration of Classification and Retrieval for Time Series Classification (https://arxiv.org/abs/2407.14735)
- **What's New**: 기존 MCQ 생성 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ와 골드 샘플의 유사성만 평가하며, 교육적 가치를 고려하지 않습니다. 본 연구는 **지식 종속 가능성(KDA)**이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성과 학생의 지식 평가 능력을 측정합니다.

- **Technical Details**: KDA는 Human evaluation을 통해 측정되지만, 두 가지 자동 평가 메트릭인 **KDA_disc**와 **KDA_cont**는 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다.

- **Performance Highlights**: Human evaluation 결과 **KDA_disc**와 **KDA_cont**는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보입니다. 또한, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가 평가 MCQ 품질 측정치에 대한 예측력이 강력해집니다.



### CrowdMAC: Masked Crowd Density Completion for Robust Crowd Density Forecasting (https://arxiv.org/abs/2407.14725)
- **What's New**: 본 논문에서는 CrowdMAC이라는 새로운 프레임워크를 제안하여 부분적으로 마스킹된 과거 군중 밀도 맵에서 미래 군중 밀도 맵을 예측하는 동시에 마스킹된 관측 맵을 재구성합니다.  CrowdMAC은 부분적으로 마스킹된 과거 군중 밀도 맵에서 미래 군중 밀도 맵을 예측하는 동시에 마스킹된 관측 맵을 재구성하여 훈련됩니다.  이를 통해 CrowdMAC은 보행자 누락으로 인해 불완전한 과거 밀도 맵에 강인한 모델을 만드는데 도움이 됩니다. 또한, 본 논문은 군중 밀도 맵의 희소성 (sparsity)과 예측 작업에 대한 후속 프레임의 정보량을 고려하여 관측된 군중 밀도 맵에서 토큰을 비균일하게 마스킹하는 Temporal-Density-aware Masking (TDM)을 제안합니다.  마지막으로, 훈련 효율성을 높이기 위해 multi-task masking을 도입했습니다.



### Differential Privacy of Cross-Attention with Provable Guaran (https://arxiv.org/abs/2407.14717)
- **What's New**: 본 논문에서는 교사의 MCQ 생성 시간을 줄이는 자동 MCQ 생성 시스템의 교육적 가치를 평가하는 새로운 지표인 KDA (Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 정확히 평가하는 능력을 측정합니다. 이 연구에서는 KDA를 측정하기 위해 인간 설문 조사 결과를 활용한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안합니다. 이러한 지표들은 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다. 인간 연구 결과, KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보였습니다. 또한, 기존의 n-gram 기반 유사성 지표와 결합하여 다양한 전문가 평가 기준에 대한 예측력을 향상시킵니다.



### Value Internalization: Learning and Generalizing from Social Reward (https://arxiv.org/abs/2407.14681)
Comments:
          Reinforcement Learning Conference (RLC) 2024 & Cognitive Science Conference Oral

- **What's New**: 본 연구는 MCQ 생성의 교육적 가치를 측정하는 새로운 자동 평가 지표인 **Knowledge Dependent Answerability (KDA)**를 제안합니다. 기존 지표들은 BLEU, ROUGE, METEOR와 같이 단어의 유사성만 측정했지만, KDA는 MCQ가 학생의 지식을 제대로 평가할 수 있는지를 측정합니다.



### Is $F_1$ Score Suboptimal for Cybersecurity Models? Introducing $C_{score}$, a Cost-Aware Alternative for Model Assessmen (https://arxiv.org/abs/2407.14664)
- **What's New**: 이 논문은 기존 MCQ 자동 생성 평가 메트릭의 한계점을 지적하고 새로운 평가 메트릭, Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성 (answerability) 측면에서 MCQ의 교육적 가치를 측정합니다. 또한 KDA를 근사화하기 위해 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.



### A New Lightweight Hybrid Graph Convolutional Neural Network -- CNN Scheme for Scene Classification using Object Detection Inferenc (https://arxiv.org/abs/2407.14658)
- **What's New**: 본 논문은 자동으로 MCQ를 생성하는 과정에서 교육적인 가치를 고려하는 새로운 평가 지표인 Knowledge Dependent Answerability (KDA)를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식 수준을 평가하는 데 도움을 줍니다.  기존 평가 지표들은 BLEU, ROUGE, METEOR와 같은 n-gram 기반 유사성 측정에 집중했지만, 실제 학습 과정에서 MCQ의 효용성은 고려하지 않았습니다.  KDA는 학생들의 응답을 분석하여 MCQ의 질을 평가하고,  KDA_disc 및 KDA_cont와 같은 자동 평가 지표를 통해  실제 강의실 환경에서의 사용성을 측정합니다.  본 연구에서는 KDA_disc 및 KDA_cont가 KDA와 높은 상관관계를 가지며, 전문가가 평가한 MCQ 품질 척도와 높은 예측력을 보여주는 것을 확인했습니다.



### Improving Representation of High-frequency Components for Medical Foundation Models (https://arxiv.org/abs/2407.14651)
- **What's New**: 본 논문은 의료 영상 분야에서 고주파 성분과 미세한 디테일을 효과적으로 표현하기 위해 새로운 사전 학습 전략인 Frepa(Frequency-advanced Representation Autoencoder)를 제안합니다. Frepa는 고주파 마스킹과 저주파 섭동을 적용하여 인코더가 이미지 임베딩에서 고주파 성분을 효과적으로 표현하고 유지하도록 합니다. 또한, 기존의 Masked Autoencoder 방식을 ViT 뿐만 아니라 Swin Transformer와 Convolutional Network와 같은 다른 아키텍처로 확장하는 히스토그램 평활 이미지 마스킹 전략을 새롭게 도입합니다.



### Two new feature selection methods based on learn-heuristic techniques for breast cancer prediction: A comprehensive analysis (https://arxiv.org/abs/2407.14631)
Comments:
          36 pages, 3 figures, 12 tables

- **What's New**: 본 연구는 유방암 진단 모델의 효율성을 향상시키기 위해 ICA(Imperialist Competitive Algorithm)와 BA(Bat Algorithm)를 기반으로 하는 두 가지 새로운 특징 선택(FS) 방법을 제안합니다. 이러한 방법은 ML 알고리즘과 결합되어 의사들이 더 정확하고 신뢰할 수 있는 진단을 내릴 수 있도록 돕습니다.



### ESCAPE: Energy-based Selective Adaptive Correction for Out-of-distribution 3D Human Pose Estimation (https://arxiv.org/abs/2407.14605)
Comments:
          32 pages, 8 figures

- **What's New**: 이 논문에서는 인간 자세 추정(HPE) 분야에서 오류를 줄이고, 특히 손목과 발목 같은 먼 키포인트(distal keypoints)에 대한 오류를 줄이는 새로운 방법인 ESCAPE를 제안합니다. ESCAPE는 경량화된 교정 및 선택적 적응 프레임워크로, 대부분의 데이터에 대한 빠른 전방 패스 교정을 적용하고, OOD(out-of-distribution) 데이터에 대해서는 비용이 많이 드는 TTA(Test-Time Adaptation)를 예약하는 방법입니다. ESCAPE는 OOD 샘플을 분리하기 위해 자유 에너지 함수를 사용하며, 사전 학습된 백본 HPE 예측의 먼 키포인트 오류를 추정하는 교정 네트워크를 학습합니다. OOD 샘플의 경우, 먼 키포인트와 가까운 키포인트(어깨, 엉덩이) 사이의 제약 관계를 활용하는 두 번째 '역' 네트워크를 통해 교정 네트워크를 업데이트하는 새로운 자기 일관성 적응 손실을 제안합니다.



### Regression prediction algorithm for energy consumption regression in cloud computing based on horned lizard algorithm optimised convolutional neural network-bidirectional gated recurrent un (https://arxiv.org/abs/2407.14575)
- **What's New**: 본 논문에서는 Convolutional Neural Networks-Bi-Directional Gated Recurrent Units (CNN-Bi-GRU) 기반의 혼(Horny Lizard) 최적화 알고리즘으로 데이터 회귀 알고리즘을 최적화하여 클라우드 컴퓨팅 에너지 소비 예측 연구를 수행했습니다. 기존 연구와 차별점은 다음과 같습니다.  1. CPU 사용률, 메모리 사용량, 네트워크 트래픽, 전력 소비량, 실행된 명령어 수, 실행 시간, 에너지 효율 등의 상관관계 분석을 통해 전력 소비량과 에너지 효율 간의 가장 높은 양의 상관관계, CPU 사용률과 에너지 효율 간의 가장 높은 음의 상관관계를 발견했습니다. 2. 혼 최적화 알고리즘을 기반으로 한 최적화 모델과 랜덤 포레스트 모델을 도입하여 실험을 진행했습니다. 3.  혼 최적화 알고리즘이 랜덤 포레스트 모델보다 더 나은 예측 결과를 보여주었고, 특히 평균 제곱 오차(MSE)는 0.01 작고, 평균 절대 오차(MAE)는 0.01 작게 나타났습니다.  



### Operating System And Artificial Intelligence: A Systematic Review (https://arxiv.org/abs/2407.14567)
Comments:
          14 pages,5 figures

- **What's New**: 본 논문은 MCQ 생성을 위한 새로운 자동 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 MCQ가 대상 사실에 대한 학생의 지식을 제대로 평가할 수 있는지 측정합니다. 기존 지표는 단어 유사도만 비교했지만 KDA는 MCQ의 실질적인 교육적 가치를 평가합니다. KDA는 인간 설문 조사를 통해 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표를 제안하여 학생의 문제 해결 능력을 모방하는 사전 훈련된 언어 모델을 활용합니다.



### Detecting and Characterising Mobile App Metamorphosis in Google Play Stor (https://arxiv.org/abs/2407.14565)
Comments:
          15 pages, 14 figures

- **What's New**: 본 논문에서는 앱의 변신(app metamorphosis)이라는 새로운 현상을 정의하고, 앱 마켓에서의 변화를 효과적으로 파악하기 위한 새로운 멀티모달 검색 방법론을 제안한다. 앱 변신은 앱이 기능 개선이나 버그 수정을 위한 점진적인 업데이트가 아닌, 사용 사례나 시장 포지셔닝을 크게 바꾸는 경우를 말한다.



### APS-USCT: Ultrasound Computed Tomography on Sparse Data via AI-Physic Synergy (https://arxiv.org/abs/2407.14564)
Comments:
          MICCAI

- **What's New**: 이 논문에서는 sparse data를 효율적으로 활용하여 USCT 이미지 재구성을 향상시키는 새로운 USCT 방법인 APS-USCT를 제안합니다. APS-USCT는 sparse data를 dense waveform으로 변환하여 재구성 전 샘플 밀도를 높이는 APS-wave와 속도를 직접 재구성하는 APS-FWI라는 두 가지 주요 구성 요소로 구성됩니다. 또한 SE 블록과 소스 인코딩 기술을 추가하여 모델 성능을 향상시켰습니다.



### NNsight and NDIF: Democratizing Access to Foundation Model Internals (https://arxiv.org/abs/2407.14561)
Comments:
          Code at this https URL

- **What's New**: 이 논문은 대규모 언어 모델에 대한 연구 접근성을 높이기 위해 NNsight라는 오픈 소스 Python 패키지와 NDIF라는 연구 플랫폼을 소개합니다. NNsight는 PyTorch 모델에 대한 간단하고 유연한 API를 제공하여 모델 내부에 대한 접근과 조작을 가능하게 합니다. NDIF는 NNsight API를 통해 연구자들이 대규모 언어 모델에 접근할 수 있는 플랫폼으로, 연구자들은 대규모 모델에 대한 연구를 보다 쉽게 수행할 수 있습니다. 



### Automated and Holistic Co-design of Neural Networks and ASICs for Enabling In-Pixel Intelligenc (https://arxiv.org/abs/2407.14560)
Comments:
          18 pages, 17 figures

- **What's New**: 본 논문은 MCQ 자동 생성 평가를 위한 새로운 지식 의존적 대답 가능성(Knowledge Dependent Answerability, KDA) 지표를 제안합니다. 이 지표는 MCQ가 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정하여 기존의 n-gram 기반 유사도 지표가 놓치는 교육적 가치를 평가합니다. KDA는 학생 설문 조사를 통해 측정되며,  KDA_disc와 KDA_cont라는 두 가지 자동 평가 지표가 제시되어 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 방식을 모방합니다.



### Predicting Star Scientists in the Field of Artificial Intelligence: A Machine Learning Approach (https://arxiv.org/abs/2407.14559)
Comments:
          21 pages, 4 figures

- **What's New**: 본 논문에서는 잠재적인 스타 과학자들을 초기 단계에 예측하는 모델을 제안하며, 이 모델은 인공지능 분야에서의 성공과 관련된 특징을 강조합니다. 특히, 떠오르는 스타들은 비슷한 수준의 연구자들과 비교했을 때 초기 경력 단계에서 다른 패턴을 보이는 경향이 있습니다. 논문에서는 성별, 인종 다양성이 과학적 협업에 중요한 역할을 한다는 사실을 발견했으며, 이는 연구자의 경력 발전 및 성공에 큰 영향을 미칠 수 있습니다.



### Risks of uncertainty propagation in Al-augmented security pipelines (https://arxiv.org/abs/2407.14540)
- **What's New**: 본 논문은 AI 기반 서브시스템이 포함된 자동화 파이프라인의 불확실성을 정량화하는 새로운 방법을 제안합니다. 이는 항공 분야와 같은 안전에 중요한 분야에서 특히 중요합니다.



### FuncEvalGMN: Evaluating Functional Correctness of SQL via Graph Matching Network (https://arxiv.org/abs/2407.14530)
- **What's New**: 본 논문에서는 SQL 코드 생성의 기능적 정확성을 평가하기 위한 새로운 그래프 기반 방법론을 제안합니다. 기존의 SQL 코드 생성 평가 지표 (예: 일치 기반, 실행 기반 방법)는 두 가지 주요 제한 사항이 있습니다. 첫째, 서로 다른 SQL 쿼리가 동일한 기능을 가질 수 있기 때문에 일치 기반 방법은 기능적 정확성을 효과적으로 평가할 수 없습니다. 둘째, 실행 기반 방법은 평가에서 오탐(false positive) 샘플을 생성할 수 있습니다. 제안된 평가 방법인 FuncEvalGMN은 테스트 데이터의 충분한 준비에 의존하지 않으며 코드의 기능적 정확성을 정확하게 테스트할 수 있습니다. 첫째, 논리적 실행 관점에서 풍부한 의미 정보를 포함하는 Relnode라는 관계 연산자 트리(ROT)를 사용하여 SQL을 파싱합니다. 그런 다음, 생성된 SQL의 기능적 정확성을 예측하기 위한 GNN 기반 접근 방식을 소개합니다. 이 접근 방식은 기존 그래프 매칭 프레임워크에서 위상 정보 손실로 인한 제한 사항을 해결하기 위해 전역 위치 임베딩을 통합합니다. 보조 기여로서, Relnode 부분 일치(RelPM)라는 규칙 기반 매칭 알고리즘을 기준선으로 제안합니다. 마지막으로, 훈련 세트와 두 개의 테스트 세트로 구성된 데이터셋 Pair-Aug-Spider를 제공하며, 각 세트는 다양한 SQL 코드 평가 시나리오를 시뮬레이션하기 위해 SQL 코드 쌍으로 구성됩니다. 훈련 세트와 한 테스트 데이터셋은 대규모 언어 모델(LLM)을 사용한 코드 생성에 중점을 두고, 다른 데이터셋은 SQL 동등성 재작성에 중점을 둡니다.



### Addressing Imbalance for Class Incremental Learning in Medical Image Classification (https://arxiv.org/abs/2407.13768)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이 논문은 교육적 가치를 고려하는 새로운 자동 MCQ 평가 지표인 "지식 종속 가능성(Knowledge Dependent Answerability, KDA)"를 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 대상 사실에 대한 학생의 지식을 평가하는 능력을 측정합니다. 또한 이 논문은 KDA를 자동으로 계산하기 위한 두 가지 지표인 KDA_disc와 KDA_cont를 제안합니다.



### Learning Neural Network Classifiers with Low Model Complexity (https://arxiv.org/abs/1707.09933)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문은 MCQ(Multiple Choice Questions) 자동 생성 과제를 위한 새로운 평가 지표인 KDA(Knowledge Dependent Answerability)를 제안합니다. KDA는 생성된 MCQ의 대답 가능성을 측정하여 교육적 가치를 평가합니다. 기존 지표인 BLEU, ROUGE, METEOR는 단어 유사성만을 측정했지만, KDA는 학생들의 지식 수준에 따라 MCQ의 적합성을 측정합니다. KDA를 측정하기 위해 인간 설문조사와 더불어, 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 능력을 모방하는 자동 평가 지표인 KDA_disc와 KDA_cont를 제안합니다.

- **Technical Details**: KDA는 인간 설문조사를 통해 얻은 학생들의 응답을 기반으로 계산됩니다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 사용하여 학생들의 문제 해결 과정을 모방하여 KDA를 추정합니다. 이러한 지표들은 기존 지표인 BLEU, ROUGE, METEOR와 함께 사용되어 다양한 전문가 평가 지표를 예측하는데 유용합니다.

- **Performance Highlights**: 본 논문에서 제안된 KDA_disc와 KDA_cont는 인간 설문조사 결과와 실제 강의실 환경에서의 사용성과 높은 상관관계를 보였습니다. 이러한 결과는 KDA_disc와 KDA_cont가 MCQ 생성의 질을 효과적으로 평가할 수 있음을 시사합니다.



### KoMA: Knowledge-driven Multi-agent Framework for Autonomous Driving with Large Language Models (https://arxiv.org/abs/2407.14239)
Comments:
          13 pages, 18 figures

- **What's New**: 이 논문에서는 여러 개의 LLM 기반 에이전트를 사용하여 복잡한 다중 에이전트 주행 환경에서 고급 의사 결정을 촉진하는 지식 기반 자율 주행 프레임워크인 KoMA를 제안합니다. KoMA는 다중 에이전트 상호 작용, 다단계 계획, 공유 메모리, 순위 기반 반성 모듈을 통합하여 에이전트의 성능을 향상시킵니다.



### The Cardinality of Identifying Code Sets for Soccer Ball Graph with Application to Remote Sensing (https://arxiv.org/abs/2407.14120)
Comments:
          22 pages, 5 figures, preprint

- **What's New**: 본 논문은 교육적 가치를 고려하지 않고 n-gram 기반 유사성만 평가하는 기존 MCQ 평가 지표의 한계를 지적하고, 새로운 지표인 '지식 종속 가능성(KDA)'을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하여 학생의 지식을 평가하는 능력을 측정합니다.  KDA를 근사화하기 위해 KDA_disc 및 KDA_cont라는 두 가지 자동 평가 지표를 제안하고, 사람의 평가를 통해 이 지표들이 실제 교육 환경에서의 유용성과 높은 상관관계를 가지고 있음을 보여줍니다.



### Optimizing Agricultural Order Fulfillment Systems: A Hybrid Tree Search Approach (https://arxiv.org/abs/2407.13968)
- **What's New**: 이 논문은 농업 산업에서 종자 공급망의 계절적 특성으로 인해 효율적인 주문 이행이 중요하다는 점을 강조하며, 특히 예측 불가능한 종자 재고 도착과 엄격한 주문 마감일을 고려하여 중앙 창고에서 주문이 파동으로 처리되는 종자 주문 이행을 최적화하는 문제를 다룹니다.  파동 스케줄링 문제를 마르코프 의사 결정 과정으로 모델링하고, 몬테카를로 트리 검색과 도메인 특정 지식을 결합한 적응형 하이브리드 트리 검색 알고리즘을 제안하여 복잡하고 역동적인 종자 유통 환경을 효율적으로 탐색합니다.  이 논문은 기존 솔루션 방법을 계산적으로 처리할 수 없게 만드는 큰 상태 및 작업 공간을 처리하기 위해 각 의사 결정 단계에서 후보 작업의 수를 동적으로 줄이는 문제 특정 부가 정보를 통해 몬테카를로 트리 검색 알고리즘을 확장할 수 있다는 핵심 아이디어를 제시합니다.  실제 매개변수(다양한 제품, 대량 주문 및 실제 계절 기간 포함)를 사용한 광범위한 시뮬레이션은 제안된 접근 방식이 기존 산업 표준 방법보다 훨씬 뛰어남을 보여줍니다.



### Assurance of AI Systems From a Dependability Perspectiv (https://arxiv.org/abs/2407.13948)
- **What's New**: 이 논문은 컴퓨터 기반 시스템의 안전성을 보장하는 고전적인 원칙을 살펴보고, 이 원칙을 인공지능(AI) 및 머신러닝(ML) 시스템에 적용하는 방법을 제시합니다. AI/ML 시스템의 안전성을 확보하기 위해서는 핵심 구성 요소의 동작을 완벽하게 이해해야 하지만, AI/ML 시스템의 경우 이는 현실적으로 불가능합니다. 따라서 이 논문은 AI/ML 요소에 대한 신뢰를 최소화하는 데 중점을 두고, 기존에 안전성이 확보된 시스템을 계층적으로 사용하여 방어하는 방법을 제시합니다. 이는 AI/ML 요소 자체의 안전성을 확보하려는 '신뢰할 수 있는' 관점과 대조적입니다. 자율 주행 자동차와 같은 사이버 물리 시스템에서는 환경을 인식하기 위해 AI/ML에 의존해야 하는 경우가 많기 때문에, 두 관점 모두 필요하며, 두 관점 사이에는 연속적인 스펙트럼이 존재합니다. 이 논문은 스펙트럼의 안전성 측면에 초점을 맞추고, 다른 연구자들이 스펙트럼의 다른 지점을 고려하도록 권장합니다. AI/ML을 사용하여 인식을 수행하는 시스템에 대해서는 AI/ML 요소에 대한 신뢰를 최소화하는 다양한 방법을 제시합니다. 이러한 방법에는 다양성, 방어 심화, 설명 가능성, 마이크로 이상치 탐지(micro-ODDs) 등이 있습니다. 또한, 이 논문은 세계 모델을 기반으로 허용 가능한 동작을 강제하는 방법을 살펴봅니다. 이러한 방법에는 고전적인 사이버 물리 계산 및 봉투, 포괄적인 원칙, 헌법, 윤리 또는 평판을 기반으로 하는 규범적 규칙 등이 포함됩니다. 이 논문은 자율 시스템, 특정 기능을 위한 AI 시스템, 대규모 언어 모델과 같은 일반적인 AI, 인공 일반 지능(AGI)에 대한 우리의 관점을 적용하고, 현재 최선의 사례와 연구 과제를 제안합니다.

- **Technical Details**: 이 논문은 AI/ML 시스템의 안전성을 보장하는 방법론을 제시합니다. 기존의 안전성 확보 방식을 AI/ML 시스템에 적용하는 어려움을 인지하고, AI/ML 요소에 대한 신뢰를 최소화하는 방어 심화 전략을 제시합니다. 이는 AI/ML 요소 자체의 안전성 확보보다는, 외부 시스템을 통해 안전성을 보장하는 방식입니다. 또한, AI/ML을 사용하는 시스템에서 신뢰를 최소화하기 위한 구체적인 방법들을 제시하고, 세계 모델을 기반으로 허용 가능한 동작을 강제하는 방식을 소개합니다. 

- **Performance Highlights**: 이 논문은 AI/ML 시스템의 안전성을 확보하는 다양한 전략과 방법론을 제시하며, AI/ML 시스템의 안전성에 대한 연구 방향을 제시합니다. 이는 안전성이 중요한 AI/ML 시스템 개발에 중요한 기여를 할 것으로 예상됩니다.



### LinSATNet: The Positive Linear Satisfiability Neural Networks (https://arxiv.org/abs/2407.13917)
Comments:
          This is a revised version of our ICML'23 publication that fixes a minor issue in Eq (11). In Proceedings of the 40th International Conference on Machine Learning (ICML'23)

- **What's New**: 본 논문은 신경망에 인기있는 선형 만족성(positive linear satisfiability)을 도입하는 방법을 연구합니다. 여러 집합의 주변 분포를 함께 인코딩하기 위한 고전적인 Sinkhorn 알고리즘의 확장을 기반으로 첫 번째 미분 가능한 만족성 레이어를 제안합니다. 또한 여러 주변에 대한 Sinkhorn 알고리즘의 수렴 특성을 이론적으로 특징짓습니다. 강화 학습 기반 솔버와 같은 순차적 의사 결정과 달리, 이 기술은 i) 최적 솔루션의 감독 없이 학습된 신경 라우팅 솔버, ii) 양쪽에 일치하지 않는 이상치가 있는 그래프를 처리하는 부분 그래프 일치 네트워크, iii) 연속적 제약이 있는 금융 포트폴리오에 대한 예측 네트워크를 포함하여 한 번에 신경망으로 제약(특히 만족성) 문제를 해결하는 데 활용합니다. 우리가 알기로, 이러한 시나리오를 만족성 문제로 공식화했을 때 한 번에 신경 솔버가 존재하지 않습니다. 소스 코드는 이 https URL에서 제공됩니다.



### DEPICT: Diffusion-Enabled Permutation Importance for Image Classification Tasks (https://arxiv.org/abs/2407.14509)
Comments:
          36 pages, 18 figures, 9 tables, to be published in ECCV 2024

- **What's New**: 본 논문에서는 이미지 분류기의 설명을 위한 새로운 순열 기반 방법을 제안합니다. 기존의 활성화 맵과 같은 이미지 모델 설명 방법은 픽셀 공간에서 인스턴스 기반 설명에 국한되어 모델의 전역적 동작을 이해하는 데 어려움이 있었습니다. 반면, 표 형식 데이터 분류기의 순열 기반 설명은 특징을 순열하기 전후의 모델 성능을 비교하여 특징 중요도를 측정합니다. 본 연구에서는 텍스트 공간에서 특정 개념(예: 캡션)으로 레이블된 이미지 데이터셋이 주어졌을 때, 데이터셋 이미지에서 해석 가능한 개념을 순열하고 텍스트 조건부 확산 모델을 통해 이미지를 생성하는 이미지 기반 모델에 대한 설명 방법을 제안합니다. 특징 중요도는 순열되지 않은 데이터에 대한 모델 성능 변화를 통해 반영됩니다. 이 방법을 개념 집합에 적용하면 특징 중요도 순위가 생성됩니다. 본 연구에서는 이 방법이 합성 및 실제 이미지 분류 작업에서 기본 모델 특징 중요도를 복구함을 보여줍니다.



### Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery (https://arxiv.org/abs/2407.14499)
Comments:
          40 pages, 21 figures, 6 tables, European Conference on Computer Vision (ECCV) 2024

- **What's New**: 본 논문은 딥 러닝 모델의 '블랙박스' 문제를 해결하기 위해 제안된 Concept Bottleneck Model (CBM)을 개선한 Discover-then-Name-CBM (DN-CBM)을 제안합니다. DN-CBM은 기존의 CBM과 달리 downstream classification task에 기반하여 concepts을 미리 선택하는 대신 sparse autoencoder를 사용하여 모델이 학습한 concepts를 먼저 발견하고, 그 다음에 이름을 지정하고 linear probes를 훈련하여 classification을 수행합니다. 이는 모델이 이미 알고 있는 concepts를 사용하기 때문에 효율적입니다.



### Explainable Post hoc Portfolio Management Financial Policy of a Deep Reinforcement Learning agen (https://arxiv.org/abs/2407.14486)
- **What's New**: 본 연구는 **설명 가능한 심층 강화 학습(XDRL)**을 도입하여 포트폴리오 관리에 대한 투명성을 높인 새로운 접근 방식을 제시합니다. XDRL은 **근접 정책 최적화(PPO)**와 **모델 비 의존적 설명 가능 기술(feature importance, SHAP, LIME)**을 통합하여 예측 시간에 대한 투명성을 높였습니다. 이를 통해 **에이전트 행동을 해석하고 투자 정책 요구 사항을 충족하는지 또는 에이전트 제안을 따르는 위험을 평가할 수 있습니다.**



### The Extrapolation Power of Implicit Models (https://arxiv.org/abs/2407.14430)
Comments:
          Accepted at the Workshop on Explainable Artificial Intelligence (XAI) at IJCAI 2024

- **What's New**: 이 논문은 암묵적 딥 러닝 모델(Implicit Deep Learning Model)의 외삽 능력(Extrapolation Capability)을 조사하여 관찰되지 않은 데이터(Unobserved Data)를 처리하는 방법을 제시합니다. 암묵적 모델은 레이어 깊이(Layer Depth)를 조절하고 계산 그래프(Computational Graph)에 피드백(Feedback)을 통합할 수 있어 기존의 딥 뉴럴 네트워크(Deep Neural Network)가 어려움을 겪는 다양한 외삽 시나리오(Extrapolation Scenario)에서 뛰어난 성능을 보여줍니다. 특히, 암묵적 모델은 각 작업에 맞는 정교한 아키텍처 설계(Architectural Design)에 의존하지 않고도 복잡한 모델 구조를 학습할 수 있다는 장점이 있습니다. 이는 암묵적 모델이 보이지 않는 데이터를 처리하는 데 있어서 강력한 능력을 가지고 있음을 의미합니다. 



### Mixture of Experts with Mixture of Precisions for Tuning Quality of Servic (https://arxiv.org/abs/2407.14417)
- **What's New**: 본 논문은 대규모 혼합 전문가(MoE) 모델을 제한된 리소스 환경에 배포하는 데 대한 증가하는 요구 사항을 해결하기 위해 효율적인 접근 방식을 제시합니다. 특히, 다양한 사용자 정의 제약 조건을 갖춘 작업과 다중 테넌트 환경에서 사용 가능한 리소스가 시간이 지남에 따라 변화하기 때문에 유연한 구성 공간을 제공하는 접근 방식을 설계해야 합니다. 이 논문은 전문가의 부분 양자화를 활용하여 MoE 모델의 효율적인 배포를 위한 적응형 서빙 접근 방식을 제시합니다. 양자화된 전문가의 수와 CPU 및 GPU에 대한 분포를 동적으로 결정함으로써 이 접근 방식은 Pareto 프론티어를 탐색하고 처리량과 모델 품질을 조정하기 위한 다양한 구성 범위를 제공합니다. 



### DEAL: Disentangle and Localize Concept-level Explanations for VLMs (https://arxiv.org/abs/2407.14412)
Comments:
          In Proceedings of the European Conference on Computer Vision (ECCV), 2024

- **What's New**: 본 연구는 자동으로 생성된 MCQ (Multiple Choice Question, 객관식 문제)의 질을 평가하기 위한 새로운 지표, Knowledge Dependent Answerability (KDA, 지식 종속 가능성)를 제안합니다. KDA는 MCQ가 학생의 지식을 제대로 평가하는지 측정합니다. 기존의 MCQ 평가 지표는 BLEU, ROUGE, METEOR 등이 있는데, 이들은 단어의 유사성만을 측정할 뿐 교육적 가치를 고려하지 않습니다.



### On the Impact of PRB Load Uncertainty Forecasting for Sustainable Open RAN (https://arxiv.org/abs/2407.14400)
- **What's New**: 이 논문은 오픈 무선 접속 네트워크(O-RAN)의 지속 가능한 구현을 위한 새로운 자원 관리 접근 방식을 제안합니다. 특히, 물리적 자원 블록(PRB) 활용을 예측하는 데 중점을 둡니다. PRB 부하를 예측하기 위해 확률적 예측 기술을 사용합니다. O-RAN 아키텍처와 구성 요소에 대한 배경 정보를 제공하고 지속 가능한 구현을 위한 에너지/전력 소비 모델의 중요성을 강조합니다. 논문은 자원 할당 및 전력 효율성을 최적화하기 위해 정확한 PRB 부하 예측이 필요하다는 점을 강조합니다. 또한 Simple-Feed-Forward(SFF), DeepAR 및 트랜스포머와 같은 확률적 예측 기술을 조사하고 이러한 기술의 가능성 모델 가정에 대해 논의합니다. 시뮬레이션 결과는 DeepAR 추정기가 SFF 및 트랜스포머 기반 모델에 비해 더 적은 불확실성으로 PRB를 예측하고 데이터셋의 시간적 의존성을 효과적으로 포착하여 전력 절약으로 이어진다는 것을 보여줍니다. 또한 다른 백분위 수 선택은 전력 절약을 증가시킬 수 있지만 과도한/부족한 프로비저닝의 비용이 발생합니다. 동시에, 장단기 메모리(LSTM)의 성능은 모든 오류 메트릭에 대해 확률적 추정기에 비해 열등한 것으로 나타났습니다. 마지막으로, 이 논문은 지속 가능한 O-RAN 구현을 위한 확률적 예측 기반 특성화의 중요성을 강조하고 미래 연구 방향을 제시합니다.




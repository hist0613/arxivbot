### MaGGIe: Masked Guided Gradual Human Instance Matting (https://arxiv.org/abs/2404.16035)
Comments: CVPR 2024. Project link: this https URL

- **What's New**: MaGGIe (Masked Guided Gradual Human Instance Matting)는 입력 속의 인간 포그라운드를 추출하는 이미지 및 비디오 처리의 기본 작업인 인간 마팅에 사용되는 새로운 프레임워크입니다. 이는 transformer attention과 sparse convolution을 포함한 최신 아키텍처를 활용하여 여러 인스턴스의 알파 매트를 동시에 출력하는 것을 가능케 합니다. 복잡한 계산 비용과 일관성을 유지하면서도, 비디오 입력에 적합한 효율적인 아키텍처를 제안합니다.

- **Technical Details**: MaGGIe는 AOT에서 영감을 받은 마스크 가이던스 임베딩을 사용하여 입력 크기를 상수로 줄이고, transformer attention을 통해 한 번의 포워드 패스에서 인스턴스 매트를 예측합니다. 이는 인스턴스 간의 상호작용을 attention 메커니즘을 통해 처리하고, 이로 인한 복잡한 정제(refinement)를 대체합니다. 또한, sparse convolution을 사용하여 효율적인 refinement를 진행하며, 비디오 입력에 대한 높은 효율과 정밀도를 보장하는 동시에, 여러 스케일에서 점진적인 정제를 적용하여 계산 비용을 절감합니다.

- **Performance Highlights**: 이 프레임워크는 다중 인스턴스 시나리오에서 일관된 추론 비용을 유지하면서도, 제안된 합성 벤치마크에서 강인하고 다양한 성능을 보여줍니다. 높은 품질의 이미지와 비디오 마팅 벤치마크를 통해 모델의 일반화 능력이 실제 시나리오에서 향상되었습니다. 또한, MaGGIe는 인스턴스 간 상호작용을 한 번의 포워드 패스로 처리하고 비디오에서 매트의 시간적 일관성을 유지하는 새로운 접근 방식을 제안합니다.



### Cantor: Inspiring Multimodal Chain-of-Thought of MLLM (https://arxiv.org/abs/2404.16033)
Comments: The project page is available at this https URL

- **What's New**: 이 논문은 시각적 추론 문제를 해결하기 위해 멀티모달 연쇄 사고 (CoT: Chain-of-Thought) 방법론을 활용한 새로운 프레임워크, 칸토르(Cantor)를 제안합니다. 이는 기존의 시각적 정보와 논리적 추론을 통합하여 복잡한 시각적 추론 작업을 해결하는 구조로 설계되었습니다.

- **Technical Details**: 칸토르 프레임워크는 두 부분, 결정 생성(Decision-Generation) 부분과 실행(Execution) 부분으로 나뉩니다. 처음에는 MLLM 혹은 LLM을 사용하여 문제의 시각적 및 텍스트 맥락을 동시에 처리하고 복잡한 추론 과정을 거칩니다. 이후 여러 '전문가' 역할을 하는 하나의 MLLM에 의해 수행됩니다. 전문가들은 각자 다른 역할과 요구사항을 가지고 참여하여 높은 수준의 정보를 제공합니다.

- **Performance Highlights**: 칸토르 프레임워크는 ScinceQA 및 Mathvista 데이터셋에서 상태-의-기술(SOTA: State-of-the-Art) 성능을 달성했습니다. Gemini를 사용할 때 각각 4.11%, 5.9%의 정확도 향상을 보였고, GPT-3.5를 사용했을 때는 2.24%, 9.2%의 정확도 향상을 보였습니다. 이는 기존 방법들을 크게 앞서는 결과입니다.



### MoDE: CLIP Data Experts via Clustering (https://arxiv.org/abs/2404.16030)
Comments: IEEE CVPR 2024 Camera Ready. Code Link: this https URL

- **What's New**: 이 연구에서는 웹에서 수집한 데이터 셋의 잡음 문제를 집중적으로 다루고, 특히 CLIP 학습에 있어 흔히 발생하는 거짓 음성(false negatives)에 대한 해결 방안을 제시합니다. Mixture of Data Experts (MoDE)라는 새로운 프레임워크를 도입하여, 데이터 클러스터링을 통해 다수의 데이터 전문가 시스템을 학습하고, 추론 시점에서는 작업 메타데이터(task metadata)와 클러스터 조건 간의 상관관계를 통해 동적으로 데이터 전문가들을 앙상블합니다.

- **Technical Details**: MoDE 프레임워크는 데이터를 클러스터링한 뒤, 각 클러스터를 기반으로 별도의 CLIP 데이터 전문가 모델을 훈련합니다. 이러한 접근 방식은 클러스터 내에서 의미론적으로 유사한 캡션을 사용하여 대조적 학습을 수행함으로써, 훈련 중 거짓 음성의 영향을 줄이고 훈련 효율을 높입니다. 추론 시에는 각 데이터 전문가의 출력을 우선 순위를 두고 결합하여 최종 분류 결과를 도출합니다.

- **Performance Highlights**: MoDE는 여러 벤치마크에서 최신 기술 대비 우수한 성능을 보여 주었습니다. 예를 들어, 이미지 분류에서 CLIP 벤치마크 기준 3.7% 향상됐으며, 이미지-텍스트 및 텍스트-이미지 검색에서 각각 3.3% 및 2.7% 향상을 달성했습니다. 또한, MoDE는 새로운 데이터 전문가를 유연하게 포함할 수 있고, 대규모의 이미지-캡션 쌍 데이터셋을 효율적으로 훈련할 수 있는 장점을 보유하고 있습니다.



### Editable Image Elements for Controllable Synthesis (https://arxiv.org/abs/2404.16029)
Comments: Project page: this https URL

- **What's New**: 이 작업에서는 사용자가 제공한 이미지를 편집하는 것이 여전히 어려운 문제임에도 불구하고, 디퓨전 모델(diffusion models)을 이용하여 입력 이미지의 공간적 편집을 촉진하는 이미지 표현을 제안합니다. 입력을 '이미지 요소(image elements)'로 인코딩하여 입력 이미지를 정확히 재구성하고, 이 요소들을 사용자가 직관적으로 편집할 수 있게 하여, 디퓨전 모델로 사실적인 이미지를 디코드할 수 있습니다.

- **Technical Details**: 이미지 요소는 사용자가 직관적으로 조작할 수 있는 형태로 인코딩되며, 디퓨전 모델을 통해 디코드될 때 사실적이고 자연스러운 이미지로 재현됩니다. 디퓨전 모델의 고차원 잡음 입력 공간(high dimensional noise input space)을 이용하지 않고, '이미지 요소'라는 새로운 형태의 이미지 표현을 통해 다양한 이미지 편집 작업에 효과성을 입증합니다.

- **Performance Highlights**: 제안된 방법은 객체의 크기 조정(object resizing), 재배열(rearrangement), 드래그(dragging), 가리기 해제(de-occlusion), 제거(removal), 변형(variation), 및 이미지 구성(image composition)과 같은 다양한 이미지 편집 작업에서 효과를 보였습니다. 이 표현은 사용자가 이미지를 더 쉽고 직관적으로 편집할 수 있게 해주며, 결과 이미지는 사실적이고 자연스러운 외관을 유지합니다.



### PuLID: Pure and Lightning ID Customization via Contrastive Alignmen (https://arxiv.org/abs/2404.16022)
Comments: Tech Report. Codes and models will be available at this https URL

- **What's New**: 우리는 PuLID라는 신개념의 튜닝이 필요 없는 신원(ID) 맞춤 방식을 제안합니다. 이 방법은 표준 확산 분기(diffusion branch)와 함께 Lightning T2I(branch)를 도입하여 원본 모델에 미치는 영향을 최소화하고 ID 충실도(fidelity)를 높이는 것을 목표로 합니다. PuLID는 ID 정보의 삽입(insertion)이 주변 환경(background), 조명(lightning), 구성(composition), 스타일(style)에 영향을 주지 않도록 설계되었습니다.

- **Technical Details**: PuLID는 두 가지 주요 기술적 접근법을 사용합니다. 첫째, Lightning T2I 분기는 노이즈에서 고품질 이미지를 생성할 수 있는 빠른 샘플링 방법을 활용합니다. 또한, contrastive alignment loss와 accurate ID loss를 도입하여 ID가 삽입된 이미지와 삽입되지 않은 이미지 간의 UNet 특징을 의미론적으로 정렬합니다. 둘째, 이 분기를 통해 생성된 고품질 이미지에서 추출된 얼굴 임베디딩(face embedding)을 사용하여 정확한 ID 손실을 계산하여 ID 유사도를 높입니다.

- **Performance Highlights**: PuLID는 ID fidelity와 editability 모두에서 SOTA(state-of-the-art) 성능을 달성했습니다. 실험 결과, 기존 방법들과 비교할 때 PuLID는 모델에 미치는 간섭이 적고, 실용적인 응용에서의 유연성을 더 많이 제공하는 것으로 나타났습니다.



### RetinaRegNet: A Versatile Approach for Retinal Image Registration (https://arxiv.org/abs/2404.16017)
- **What's New**: 새로운 RetinaRegNet 모델은 홍채 이미지 등록(registration) 작업에서 최첨단(state-of-the-art) 성능을 달성할 수 있습니다. 이 모델은 홍채 이미지 학습 없이도 사용할 수 있으며, SIFT (Scale-Invariant Feature Transform) 알고리즘을 통해 이동 이미지의 특징 점을 선택하고 확률적 포인트 샘플링을 사용합니다. 우리는 또한 복잡한 이미지 데이터를 처리할 때 일관된 결과를 보장하고 계산 시간을 절약하는 새로운 이상치 탐지기(outlier detector)를 개발했습니다.

- **Technical Details**: RetinaRegNet은 최초로 두 홍채 이미지 간의 점 대응을 설정하기 위하여 확산 모델에서 파생된 이미지 특징을 사용합니다. 점 대응 평가에 있어서 역 일관성 제약(inverse consistency constraint)을 적용하고 변환 기반 이상치 감지기를 사용하였습니다. 뿐만 아니라, 큰 변형을 처리하기 위해 두 단계의 이미지 등록 프레임워크를 사용했습니다. 첫 번째 단계에서는 배경 변환(homography transformation)을 사용하고 두 번째 단계에서는 보다 정확한 3차 다항식 변환을 사용합니다.

- **Performance Highlights**: RetinaRegNet은 색각 막상 이미지(color fundus images), 형광 안저촬영 이미지(fluorescein angiography images), 레이저 스페클 유동도 영상(laser speckle flowgraphy images) 등 세 가지 홍채 이미지 데이터셋에서 현재 최고의 방법들을 능가했습니다. 특히 큰 변위와 스케일링 변형이 있는 이미지 쌍을 등록하는 데 효과적이었습니다. 이 모델은 홍채 이미지 분석의 다양한 응용 프로그램에 유망한 혁신을 제공합니다.



### GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with  Audio-Driven 3D Gaussian Splatting (https://arxiv.org/abs/2404.16012)
Comments: Project Page: this https URL

- **What's New**: GaussianTalker는 실시간으로 포즈 제어가 가능한 토킹 헤드(Talking Head)를 생성하는 새로운 프레임워크를 제안합니다. 이는 3D Gaussian Splatting(3DGS)의 빠른 렌더링 능력을 활용하며, 음성 오디오로 직접 3DGS를 제어하는 데 있어 과제를 해결합니다. GaussianTalker는 표준 3DGS 표현을 구성하고 오디오와 동기화하여 변형합니다.

- **Technical Details**: GaussianTalker는 3D Gaussian 속성을 공유된 암시적 특성(implicit feature) 표현으로 인코딩하고, 이를 오디오 특성과 결합하여 각 Gaussian 속성을 조작합니다. 이 설계는 공간적 인식 기능을 이용하고 인접 포인트 간 상호 작용을 강제합니다. 특성 임베딩은 공간-오디오 주의 모듈(spatial-audio attention module)에 공급되어 각 Gaussian의 속성에 대한 프레임별 오프셋을 예측합니다. 이 모듈은 이전의 연결(concatenation)이나 곱셈(multiplication) 방식보다 더 안정적인 방법을 제공합니다.

- **Performance Highlights**: GaussianTalker는 얼굴의 충실도, 입술 동기화 정확도 및 렌더링 속도에서 이전 방법들과 비교하여 우수한 성능을 보였습니다. 특히 놀라운 120 FPS의 렌더링 속도를 달성하여 이전 벤치마크를 초과합니다. 이와 관련된 코드는 제공된 URL에서 확인할 수 있습니다.



### MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large  Vision-Language Models Towards Multitask AGI (https://arxiv.org/abs/2404.16006)
Comments: 77 pages, 41 figures

- **What's New**: MMT-Bench 소개, LVLMs 제풼모델을 위한 새로운 포괄적인 벤치마크입니다. 이는 다양한 전문 지식을 요구하며, 복잡한 시각 인지, 위치 파악, 추론 및 계획과 같은 과업을 포함한 방대한 멀티모달 과업을 평가할 수 있는 도구로 설계되었습니다.

- **Technical Details**: MMT-Bench는 차량 운전, 실체화 탐색(embodied navigation)과 같은 여러 시나리오에서 31,325개의 멀티-초이스 시각 질문을 다룹니다. 이 벤치마크는 32개의 핵심 메타태스크(core meta-tasks)와 162개의 서브태스크(sub-tasks)를 포함하여 멀티모달 이해(multimodal understanding)를 평가합니다.

- **Performance Highlights**: 30개의 LVLMs를 평가한 결과, MMT-Bench가 제기하는 상당한 도전을 강조합니다. GPT-4V, GeminiProVision, 그리고 open-sourced InternVL-Chat과 같은 모델들이 포함된 테스트로 이 벤치마크가 어떺게 LVLMs의 발전에 도움을 줄 수 있는지를 보여줍니다.



### A comprehensive and easy-to-use multi-domain multi-task medical imaging  meta-dataset (MedIMeta) (https://arxiv.org/abs/2404.16000)
- **What's New**: MedIMeta, 새로운 의료 이미지 메타데이터셋을 소개합니다. 이 데이터셋은 10개의 다양한 도메인(domain)에서 19개의 의료 이미지 데이터셋을 아우르고, 총 54개의 다양한 의료 작업(medical tasks)을 포함하고 있습니다. 모든 데이터는 표준화되어 PyTorch 및 기타 ML 프레임워크에서 즉시 사용할 수 있도록 준비되어 있습니다. 이것은 특별히 크로스-도메인 퓨샷 학습(CD-FSL, Cross-Domain Few-Shot Learning)에 있어서 큰 진전을 나타냅니다.

- **Technical Details**: MedIMeta는 모든 이미지를 224×224 픽셀로 표준화하고, 사전 훈련된 모델(pre-trained models)에서 일반적으로 사용되는 이미지 크기와 일치시킵니다. 또한, 데이터를 사용하기 쉽게 사전 설정된 분할(pre-made splits)을 제공하여 표준화된 벤치마킹을 보장합니다. 파이썬 패키지를 통해 PyTorch에서 이미지를 직접 로드할 수 있게 함으로써 사용자 친화적 접근성을 제공합니다.

- **Performance Highlights**: 기술 검증을 통해 MedIMeta의 유틸리티를 입증했습니다. 전적으로 감독된 학습(fully supervised learning) 및 CD-FSL기준을 사용하여 데이터셋의 신뢰성과 강건성을 확인하고, 의료 이미지 분석에서 ML 연구를 위한 신뢰할 수 있는 벤치마크로서 자리매김하고 있습니다. 다양한 작업과 도메인에서 퓨샷 학습 기술을 연구하고 개발할 수 있는 탁월한 기회를 제공합니다.



### HDDGAN: A Heterogeneous Dual-Discriminator Generative Adversarial  Network for Infrared and Visible Image Fusion (https://arxiv.org/abs/2404.15992)
- **What's New**: 새롭게 개발된 이기종 이중 판별자 생성적 적대 신경망 (HDDGAN)는 적외선 및 가시 광선 이미지 융합 (IVIF; Infrared and Visible Image Fusion) 문제를 다룹니다. 이 모델은 특히 다양한 규모의 기능을 추출하고 융합하기 위해 다중 규모의 건너뛰기 연결 구조를 사용하며 주목 메커니즘을 활용하여 정보 융합 계층을 구축합니다. 또한, 적외선 및 가시 광선 이미지 각각에 최적화된 두 개의 구조가 다른 판별자를 도입하여 모델이 보다 효율적으로 중요 기능을 학습할 수 있도록 설계되었습니다.

- **Technical Details**: HDDGAN은 생성기(generator)에 다중 규모의 건너뛰기 연결 구조를 적용함으로써 중요 정보의 검출 및 통합 수행을 개선합니다. 주목 메커니즘(attention mechanism)을 사용하여 적외선 및 가시 광선 이미지 간의 차이를 활용하여 융합 계층을 구성합니다. 또한, 글로벌 판별자(global discriminator)와 마르코프 판별자(Markovian discriminator)라는 두 가지 다른 구조의 판별자를 설계함으로써 각각 적외선 영역의 특징과 가시 이미지의 상세 정보 학습에 초점을 맞춥니다. 이 구조는 학습 과정 중에 서로 다른 이미지 소스에서 정보를 효과적으로 추출하고 학습할 수 있도록 지원합니다.

- **Performance Highlights**: 다양한 공개 데이터셋에 대한 광범위한 실험을 통해 HDDGAN이 다른 최신 기술(SOTA; State-of-the-Art) 알고리즘들보다 우수함을 입증하였습니다. 특히, HDDGAN은 감소 저항성(degradation resistance), 다운스트림 애플리케이션 분석, 확장된 실험에서도 뛰어난 성능을 보여줍니다. 이는 HDDGAN이 실제 응용 프로그램에서 매우 유망한 기술임을 시사합니다.




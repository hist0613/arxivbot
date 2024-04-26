### The Third Monocular Depth Estimation Challeng (https://arxiv.org/abs/2404.16831)
Comments: To appear in CVPRW2024

- **What's New**: 이 논문은 단안(모노큘러) 깊이 추정 챌린지(Monocular Depth Estimation Challenge, MDEC)의 세 번째 판에 대한 결과를 논의하고 있습니다. 본 챌린지는 자연 및 실내 환경에서 복잡한 장면을 특징으로 하는 SYNS-Patches 데이터셋에 대한 제로샷 일반화(zero-shot generalization)에 중점을 두고 있습니다.

- **Technical Details**: 이번 챌린지는 이전과 마찬가지로 감독형(supervised) 또는 자기감독형(self-supervised)의 모든 형태의 감독을 사용할 수 있습니다. 총 19개의 제출물이 기준 베이스라인을 초과하는 성능을 보였으며, 그 중 10개는 접근 방법을 설명하는 보고서를 제출했습니다. 이 중 'Depth Anything'과 같은 기초 모델(foundational models)의 사용이 널리 퍼진 것으로 강조되었습니다.

- **Performance Highlights**: 챌린지 우승자들은 3D F-Score 성능을 17.51%에서 23.72%로 크게 개선할 수 있었습니다. 이것은 복잡한 장면에서의 깊이 추정의 정확성을 향상시키는 데 중요한 발전을 나타내며, 실내외 환경에서의 깊이 인식 애플리케이션에 유의미한 기여를 제공할 것입니다.



### Make-it-Real: Unleashing Large Multimodal Model's Ability for Painting  3D Objects with Realistic Materials (https://arxiv.org/abs/2404.16829)
Comments: Project Page: this https URL

- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs), 특히 GPT-4V를 이용하여 3D 오브젝트에 현실적 재료 속성을 자동 할당하는 새로운 접근법 'Make-it-Real'을 제시합니다. 이 방식은 3D 자산의 현실감을 개선하고자 하는 새로운 기술적 진보를 보여줍니다.

- **Technical Details**: GPT-4V를 사용하여 재료를 인식하고 묘사할 수 있으며, 이는 자세한 재료 라이브러리의 구축을 가능하게 합니다. 또한, 시각적 단서와 계층적인 텍스트 프롬프트를 결합하여 3D 객체의 해당 구성 요소에 재료를 정확하게 식별하고 매치합니다. 이렇게 식별된 재료는 오리지널 diffuse map (디퓨즈 맵)에 따라 새로운 SVBRDF (Spatially Varying Bidirectional Reflectance Distribution Function) 재료 생성을 위한 참조로 세심하게 적용됩니다.

- **Performance Highlights**: Make-it-Real은 3D 컨텐츠 생성 워크플로우에 손쉽게 통합되며, 3D 자산 개발자에게 필수 도구로서의 유틸리티를 입증합니다. 복잡하고 시간이 많이 소요되는 수동 재료 할당 작업을 효과적으로 간소화하고, 3D 자산의 시각적 진위성을 크게 향상시킬 수 있습니다.



### Made to Order: Discovering monotonic temporal changes via  self-supervised video ordering (https://arxiv.org/abs/2404.16828)
Comments: Project page: this https URL

- **What's New**: 새로운 연구에서, 일련의 이미지에서 시간이 지나면서 단조롭게 변하는 변화를 발견하고 지역화하는 것이 목표입니다. 이를 위해 혼합된 이미지 순서를 배열하는 간단한 프록시 작업을 활용하며, '시간'을 감독 신호로 사용하여 시간에 따라 단조로운 변화만이 올바른 순서를 낳을 수 있습니다.

- **Technical Details**: 본 연구는 임의의 길이의 이미지 시퀀스에 대한 일반적인 목적의 정렬 (ordering)과 함께 내장된 속성 맵(attribution maps)을 제공하는 유연한 트랜스포머 기반 모델(transformer-based model)을 소개합니다. 이 모델은 교육 후 단조로운 변화를 성공적으로 발견하고 지역화하면서 주기적(cyclic) 및 확률적(stochastic) 변화는 무시합니다.

- **Performance Highlights**: 이 모델은 다양한 비디오 설정에서 시험되었으며, 객체 수준과 환경 변화를 미리 보지 않은 시퀀스에서 발견하는 데 성공했습니다. 또한, 주목 기반 속성 맵은 변화하는 영역을 분할하는 데 효과적인 프롬프트로 기능하며, 학습된 표현은 하류 애플리케이션(downstream applications)에 사용될 수 있습니다. 최종적으로, 이 모델은 이미지 세트 정렬에 대한 표준 벤치마크에서 최고의 성능(state of the art)을 달성했습니다.



### ResVR: Joint Rescaling and Viewport Rendering of Omnidirectional Images (https://arxiv.org/abs/2404.16825)
- **What's New**: 이 연구는 가상현실 기술의 발전으로 보다 실현화된 전망을 제공합니다. ResVR은 전 세계적 이미지 (ODI: omnidirectional image) 재조정 및 뷰포트 렌더링을 동시에 처리할 수 있는 최초의 포괄적 프레임워크를 제시합니다. 기존의 ODI 재조정 방법들이 equirectangular projection (ERP) 이미지의 품질 향상에 집중했던 것과 다르게, ResVR은 사용자가 헤드 마운트 디스플레이 (HMD)에서 실제로 경험하는 뷰포트의 비주얼 품질을 강조합니다.

- **Technical Details**: ResVR은 전송을 위해 저해상도의 ERP 이미지를 얻으면서 사용자가 HMD에서 고품질의 뷰포트를 볼 수 있게 하는 기술을 개발하였습니다. 이를 위해 디스크리트 픽셀 샘플링(discrete pixel sampling) 전략을 개발하여 뷰포트와 ERP 간의 복잡한 매핑을 해결하였고, 이는 ResVR 파이프라인의 end-to-end 트레이닝을 가능하게 합니다. 또한, 구면 차별화(spherical differentiation)에서 파생된 구면 픽셀 형태 표현 기술을 혁신적으로 도입하여 렌더링된 뷰포트의 시각적 품질을 크게 개선했습니다.

- **Performance Highlights**: ResVR은 다양한 시야각(field of view), 해상도, 및 시청 방향에서 기존 방법들보다 뷰포트 렌더링 작업에서 우수한 성능을 입증하였습니다. 또한 전송 오버헤드를 낮게 유지하면서 이러한 성능을 달성하였습니다. 전반적으로, ResVR은 ODI를 처리하는 새로운 접근 방식을 제공하면서, 가상현실 환경에서 사용자 경험을 크게 향상시키는 데 기여할 것입니다.



### V2A-Mark: Versatile Deep Visual-Audio Watermarking for Manipulation  Localization and Copyright Protection (https://arxiv.org/abs/2404.16824)
- **What's New**: AI가 생성하는 비디오는 짧은 비디오 제작, 영화 제작 및 맞춤형 미디어에서 혁명을 일으켰으며, 이로 인해 비디오 로컬 편집(Video Local Editing)이 필수 도구가 되었습니다. 그러나 이러한 진보는 또한 현실과 허구의 경계를 흐리게 만들어 멀티미디어 포렌식(Multimedia Forensics) 분야에서 도전을 야기합니다. V2A-Mark는 현재의 비디오 변조 포렌식의 제한을 해결하기 위해 제안되었습니다. 이 방법은 일반화 능력이 떨어지고, 단일 기능에 집중되며, 단일 모달리티에만 초점을 맞춘 기존의 접근법들을 개선합니다.

- **Technical Details**: V2A-Mark는 비디오 내 비디오 스테가노그래피(Video-into-Video Steganography)의 취약성과 깊은 강인한 워터마킹(Deep Robust Watermarking)을 결합하여 원본 비디오 프레임과 오디오에 보이지 않는 시각-청각 로컬라이제이션 워터마크와 저작권 워터마크를 삽입할 수 있습니다. 이를 통해 정확한 조작 로컬라이제이션(Manipulation Localization)과 저작권 보호가 가능합니다. 또한, 로컬라이제이션 정확도와 디코딩 강인성을 향상시키기 위해 시간 정렬 및 퓨전 모듈(Temporal Alignment and Fusion Module)과 디그레이드 프롬프트 학습(Degradation Prompt Learning)을 설계했습니다. 그리고, 오디오와 비디오 프레임의 정보를 결합하는 샘플 레벨 오디오 로컬라이제이션 방법과 크로스-모달 저작권 추출 메커니즘(Cross-Modal Copyright Extraction Mechanism)을 도입했습니다.

- **Performance Highlights**: V2A-Mark의 효과는 시각-청각 변조 데이터셋(Visual-Audio Tampering Dataset)에서 검증되었으며, 로컬라이제이션 정밀성과 저작권 정확성에서 뛰어난 우수성을 강조했습니다. 이는 AIGC 비디오 시대에서 비디오 편집의 지속 가능한 발전에 매우 중요합니다.



### How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal  Models with Open-Source Suites (https://arxiv.org/abs/2404.16821)
Comments: Technical report

- **What's New**: InternVL 1.5에서는 MLLM (multimodal large language model)을 향상하여 오픈 소스와 상업적인 모델 사이의 기능 격차를 해소합니다. 세 가지 주요 개선 점을 도입하였는데, 첫 번째로 강력한 시각 인코더(Strong Vision Encoder), 동적 고해상도(Dynamic High-Resolution), 그리고 고품질 이중 언어 데이터셋(High-Quality Bilingual Dataset)입니다.

- **Technical Details**: InternVL 1.5는 세 개의 기술적 개선을 포함합니다. 첫 번째, '강력한 시각 인코더'(Strong Vision Encoder)는 대규모 시각 기반 모델인 InternViT-6B를 지속적으로 학습시켜 시각적 이해 능력을 향상시켰습니다. 두 번째, '동적 고해상도'(Dynamic High-Resolution)는 입력 이미지의 종횡비와 해상도에 따라 이미지를 1부터 40개의 448×448 픽셀 타일로 분할하여 최대 4K 해상도 입력을 지원합니다. 세 번째, '고품질 이중 언어 데이터셋'(High-Quality Bilingual Dataset)은 일반적인 장면과 문서 이미지를 상세히 수집하고, 영어와 중국어의 질문-답변 쌍으로 주석을 달아 OCR 및 중국어 관련 작업 성능을 크게 향상시켰습니다.

- **Performance Highlights**: InternVL 1.5는 오픈 소스 및 상업적인 모델과 비교하여 경쟁력 있는 성능을 보여주었으며, 18개의 벤치마크 중 8개에서 최고의 결과(State-of-the-art)를 달성하였습니다. 또한, 이번 모델의 코드는 공개적으로 공유되었습니다.



### Revisiting Text-to-Image Evaluation with Gecko: On Metrics, Prompts, and  Human Ratings (https://arxiv.org/abs/2404.16820)
Comments: Data and code will be released at: this https URL

- **What's New**: 본 논문에서는 텍스트-투-이미지(text-to-image, T2I) 생성 모델이 주어진 프롬프트와 일치하는 이미지를 생성하지 않는 문제를 다룹니다. 이 문제에 대응하기 위해 새로운 기술적 접근 방식과 벤치마크를 제안하며, 특히 T2I 모델의 성능을 정밀하게 평가할 수 있는 새로운 QA(Question-Answering)-기반 자동 평가 메트릭(auto-eval metric)을 도입합니다.

- **Technical Details**: 이 연구에서는 100,000개 이상의 주석(annotation)을 포함하여 네 가지 T2I 모델과 네 가지 인간 평가 템플릿(human templates)을 사용하여 인간 평가를 수집하였습니다. 또한, 제안된 '기술 기반 벤치마크'(skills-based benchmark)는 프롬프트를 하위 기술(sub-skills)로 분류하여, 어떤 기술이 어려운지, 그리고 어떤 복잡성 수준에서 기술이 도전적인지를 정확히 지적할 수 있게 합니다.

- **Performance Highlights**: 새로운 QA기반 자동 평가 메트릭은 기존의 메트릭보다 인간의 평가와 더 높은 상관성을 보이며, TIFA160(TIFA160 dataset)을 포함한 다양한 휴먼 템플릿(human templates)과 T2I 모델에 걸쳐 성능이 향상되었습니다.



### Boosting Unsupervised Semantic Segmentation with Principal Mask  Proposals (https://arxiv.org/abs/2404.16818)
Comments: Code: this https URL

- **What's New**: PriMaPs - Principal Mask Proposals - 는 이미지를 시맨틱적(semantic)으로 의미있는 마스크(mask)로 분해하고, PriMaPs-EM (PriMaPs with a stochastic expectation-maximization algorithm)을 통해 무감독(unsupervised) 시맨틱 분할(semantic segmentation)을 구현합니다. DINO와 DINOv2와 같은 다양한 사전 학습된 모델(pre-trained models) 및 Cityscapes, COCO-Stuff, Potsdam-3와 같은 데이터셋에서 경쟁력 있는 결과를 제공합니다.

- **Technical Details**: PriMaPs는 무감독 시맨틱 분할을 위해 자기 감독 학습(self-supervised learning)에서 얻은 전 세계 범주(global categories)를 식별하여 이미지를 자동으로 파티션(partition)합니다. PriMaPs-EM은 기대값 최대화 알고리즘(expectation-maximization algorithm)을 사용하여 클래스 프로토타입(class prototypes)을 PriMaPs에 적합하게 만듭니다.

- **Performance Highlights**: 이 방법은 개념적으로 단순함에도 불구하고, 다양한 사전 훈련된 백본 모델 및 데이터셋에서 경쟁력 있는 결과를 낼 수 있습니다. 또한, 최신의 무감독 시맨틱 분할 파이프라인에 수직적으로 적용되어 결과를 향상시킬 수 있는 능력을 가집니다.



### Meta-Transfer Derm-Diagnosis: Exploring Few-Shot Learning and Transfer  Learning for Skin Disease Classification in Long-Tail Distribution (https://arxiv.org/abs/2404.16814)
Comments: 17 pages, 5 figures, 6 tables, submitted to IEEE Journal of Biomedical and Health Informatics

- **What's New**: 본 연구에서는 희귀 피부 질환을 대상으로 한장 길이 데이터(distributions)의 문제를 해결하기 위하여 적은 수의 사례 데이터로도 효과적인 모델을 구축할 수 있는 새로운 접근 방식을 제시합니다. 특히 에피소드(Episodic)와 전통적인 교육 방법론을 비교 분석하면서, 소수샷 학습(Few-shot learning)과 전이 학습(Transfer learning)을 결합하여 효율적인 방안을 모색하였습니다.

- **Technical Details**: 연구는 ISIC2018, Derm7pt 및 SD-198 데이터셋을 사용하여 모델을 평가했습니다. 주요 실험은 DenseNet121과 MobileNetV2 모델에 ImageNet에서 사전 훈련된 모델을 사용하여 클래스 내 유사성을 증가시키는 방식으로 특징을 표현하는 능력이 향상되었다는 것을 확인했습니다. 또한, 데이터 증강(Data augmentation) 기술을 활용하여 전이 학습 기반 모델의 성능을 개선시키는 실험을 진행하였습니다.

- **Performance Highlights**: 본 모델은 2-way에서 5-way 분류, 최대 10개의 예제를 사용한 실험에서 전통적인 전이 학습 방법이 예제 수가 증가함에 따라 성공률이 높아지는 것을 보였습니다. 특히 SD-198과 ISIC2018 데이터셋에서 기존 방법들보다 더 높은 성능을 보이며, 적은 레이블의 예시에서도 높은 정보 획득과 성능 향상을 실현하였습니다. 연구와 관련된 모든 소스 코드는 곧 공개될 예정입니다.



### AAPL: Adding Attributes to Prompt Learning for Vision-Language Models (https://arxiv.org/abs/2404.16804)
Comments: Accepted to CVPR 2024 Workshop on Prompting in Vision, Project Page: this https URL

- **What's New**: 본 논문은 'AAPL(Adding Attributes to Prompt Learning)'이라는 새로운 메커니즘을 제안하여, 기존의 프롬프트 학습 방식에 대한 한계를 극복하고자 합니다. AAPL은 learnable prompts(학습 가능 프롬프트)에 고급 특성을 추가함으로써, 클래스 특성을 더 효과적으로 이해하고 추출하는 데 중점을 두었습니다. 이 방법은 특히 보이지 않는(seen) 클래스에 대해 일반화 성능을 크게 향상시키는 것을 목표로 합니다.

- **Technical Details**: AAPL은 learnable contexts(학습 가능한 문맥)에 'delta meta token'을 도입하여, 이 token이 특정 클래스 관련 세밀한 특성(attribute-specific) 정보를 담을 수 있도록 합니다. 또한, AAPL은 AdTriplet loss를 사용하여 이미지 증강(augmentation)이 추가된 learnable prompt에 대한 조건부 bias를 강화하고 안정화합니다. 이러한 접근법은 특히 zero-shot learning(제로샷 학습)과 few-shot learning(퓨샷 학습), cross-dataset(교차 데이터셋) 태스크 및 도메인 일반화(domain generalization) 작업에서 유리합니다.

- **Performance Highlights**: AAPL은 11개의 데이터셋을 거쳐 실험을 진행하였으며, 일반적으로 기존 방법들에 비해 유리한 성능을 보였습니다. 특히, few-shot learning, zero-shot learning, cross-dataset 및 도메인 일반화 작업에서의 성능 향상을 입증하였습니다. 이는 AAPL이 고급 특성에 초점을 맞춤으로써, 더 정교하고 일반화된 텍스트 특성 추출이 가능함을 보여줍니다.




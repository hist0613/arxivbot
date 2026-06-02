New uploads on arXiv(cs.LG)

### Remote Sensing Semantic Segmentation Quality Assessment based on Vision Language Mod (https://arxiv.org/abs/2502.13990)
Comments:
          16 pages,6 figures

- **What's New**: 본 논문은 원격 감지 이미지를 위한 비지도 품질 평가 모델인 RS-SQA를 제안합니다. 기존의 평가 메트릭스가 전문가 레이블에 의존하는 반면, RS-SQA는 Vision Language Model(VLM)을 활용하여 보다 클래스 간의 불일치를 측정할 수 있게 됩니다. RS-SQA는 CLIP-RS라는 대규모 사전 훈련된 VLM을 통해 세분화 품질에 대한 보고서를 작성하며, 이는 다양한 원격 감지 파라미터로의 적용 가능성을 보여줍니다.

- **Technical Details**: RS-SQA는 세분화 정보가 포함된 중간 레이어 특성과 CLIP-RS에서 제공하는 의미적 특징을 결합한 이중 분기 네트워크 구조입니다. CLIP-RS는 의미적 유사성을 기반으로 한 데이터 정제 전략을 통해 텍스트 노이즈를 감소시켜, 원격 감지에서의 강인한 의미적 이해 능력을 향상시켰습니다. 또한 RS-SQED라는 고유의 데이터셋을 구축하여 8개의 대표 세분화 모델의 정확도 점수로 주석이 달려 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 RS-SQA는 기존의 품질 평가 모델들을 초월한 성능을 보입니다. 실험 결과는 RS-SQA가 세분화 품질을 예측하고, 최고의 모델을 추천하는 데 있어 73%의 정확도를 달성함을 보여줍니다. 이는 원격 감지 이미지의 세분화 분석을 위한 효율성을 증대시키며, 실제 어플리케이션에서의 활용 가치를 높입니다.



### Benchmarking Automatic Speech Recognition coupled LLM Modules for Medical Diagnostics (https://arxiv.org/abs/2502.13982)
- **What's New**: 이번 연구에서는 자동 음성 인식(ASR)과 대형 언어 모델(LLM) 기반의 헬스케어 시스템이 도입되어 효율적이고 접근 가능한 환자 지원을 가능하게 하는 새로운 모델을 제시합니다. 의료통화 녹음을 기반으로 한 세분화된 모델을 분석하며, 특히 노이즈와 클리핑에 강한 새로운 오디오 전처리 전략을 개발하였습니다. 이러한 전략은 환자의 통화/녹음 조건에 관계없이 일관된 퍼포먼스를 제공할 수 있도록 돕습니다.

- **Technical Details**: 시스템은 두 단계로 운영됩니다: 첫 번째로 ASR이 음성을 텍스트로 전사하고, 두 번째로 LLM이 전사된 텍스트를 기반으로 문맥 인식 응답을 생성합니다. 노이즈와 클리핑을 효과적으로 처리하기 위해 고주파 필터 및 저주파 필터를 포함한 평형화(equalization) 기법을 활용하여 오디오 신호를 개선하고 있습니다. 이러한 방법은 낮은 품질의 통화 녹음으로 인한 정확도 손실을 방지하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 연구 결과, Whisper 모델이 실시간으로 다양한 환경에서 우수한 전사 성능을 보여주었으며, ASR 시스템의 정확도를 상당히 향상시켰습니다. Whisper는 적은 수의 훈련만으로도 의료 용어와 같은 도메인 특화 작업에 적합하도록 조정할 수 있어, wav2vec 2.0과 비교해 더 나은 결과를 보였습니다. 선택된 Qwen2 모델은 속도와 성능면에서 훌륭하여 전체 시스템의 효율성을 극대화합니다.



### Regularização, aprendizagem profunda e interdisciplinaridade em problemas inversos mal-postos (https://arxiv.org/abs/2502.13976)
Comments:
          200 pages, in Portuguese language, 54 figures

- **What's New**: 이번 논문에서는 ill-posed problems에 관한 내용을 다루며, 이러한 문제를 해결하기 위한 regularization method의 중요성을 강조합니다. 다양한 분야에서의 regularization의 유사성과 차이점에 대해 질문과 답변 형식으로 설명하고 있습니다. 이 책은 기계학습(Machine Learning)과 통계학(Statistics) 및 심층학습(Deep Learning)과 같은 분야에서의 정규화의 미래를 탐구합니다.

- **Technical Details**: 회귀(regression) 문제는 주어진 법칙에 대한 불완전한 지식으로 사건을 발견하려고 시도하는 것이 아니라, 사건을 알고 법칙을 찾고자 하는 경우로 분류됩니다. 이 책은 pnorm과 같은 함수의 특성과 Euclidean norm, infinity norm 등 여러 종류의 norm에 대해 논의하고, 각각의 거리 측정법을 설명합니다. 이러한 norm들은 벡터 간의 거리를 측정하는 데 사용되며, 0 < p < 1인 경우에는 triangle inequality가 성립하지 않음을 지적합니다.

- **Performance Highlights**: 본 논문은 sparsity(희소성)와 관련된 norm의 중요성에 대해 여러 가지 시각화를 제공하며, ℓ1과 ℓ0 norm 간의 해결책이 조건에 따라 일치할 수 있음을 강조합니다. 또한, ℓ2 norm과 ℓ1 norm의 결과를 비교하여 최적화(minimization) 문제에서의 차이를 시각적으로 도식화합니다. 이 결과들을 통해 regularization method가 실험적 방법의 핵심 문제로 작용할 수 있음을 알 수 있습니다.



### Herglotz-NET: Implicit Neural Representation of Spherical Data with Harmonic Positional Encoding (https://arxiv.org/abs/2502.13777)
Comments:
          Keywords: Herglotz, spherical harmonics, spectral analysis, implicit neural representation. Remarks: 4 pages + 1 reference page, 4 figures (submitted to SAMPTA2025)

- **What's New**: 최근 업데이트된 연구 내용은 Herglotz-NET(HNET)이라는 새로운 암시적 신경 표현(implicit neural representation, INR) 아키텍처를 도입함으로써 구면 데이터 처리의 정확성 및 안정성을 높이는 것입니다. 기존의 SPH-SIREN 방법과는 달리, HNET은 구면 조화 함수(spherical harmonics)를 명시적으로 평가할 필요가 없이 Herglotz 매핑을 기반으로 하는 조화적 위치 인코딩(harmonic positional encoding)을 사용합니다. 이는 구면에서 잘 정의된 신호 표현을 가능하게 하여, 신뢰할 수 있는 스펙트럼 속성을 유지합니다.

- **Technical Details**: HNET은 지구와 같은 구면의 고유한 기하학을 고려하여 설계된 새로운 구조로, 위치 인코딩에 Herglotz 매핑을 사용하여 다루고 있습니다. 또한, HNET은 Guated 개수에 따라 예측 가능한 스펙트럼 확장을 보여주는 보편적인 표현력 분석(unified expressivity analysis)을 통해 깊이에 따라 성능이 안정적으로 향상되는 특성을 보입니다. 이 연구는 잠재적으로 깊이 깊은 신경망에 대해 조화적 위치 인코딩을 접목시킨 비유클리드 기하학적 공간에서의 활용 가능성을 제시하고 있습니다.

- **Performance Highlights**: 실험을 통해 HNET은 두 가지 응용 프로그램, 즉 초해상도(super-resolution) 작업과 연속적인 구면 라플라시안(continuous spherical Laplacian) 맵 추정에서 SPH-SIREN과 유사한 성능을 입증했습니다. HNET는 피처 인코딩의 간단함에도 불구하고, 비구면 SIREN보다 훨씬 더 높은 정확도를 자랑하는 것으로 나타났습니다. 이러한 결과는 HNET이 구면 데이터를 정확하게 모델링할 수 있는 확장 가능하고 유연한 프레임워크임을 보여줍니다.



### Linear Diffusion Networks: Harnessing Diffusion Processes for Global Interactions (https://arxiv.org/abs/2502.12381)
- **What's New**: 이번 연구에서는 선형 확산 네트워크(Linear Diffusion Networks, LDN)를 소개하는데, 이는 순차적 데이터 처리를 하나의 통합된 확산(process) 프로세스으로 재해석한 새로운 아키텍처이다. LDN은 적응형 확산 모듈(adaptive diffusion modules)과 국소 비선형 업데이트(localized nonlinear updates), 그리고 확산 기반 어텐션 메커니즘(diffusion-inspired attention mechanism)을 통합하여 전 세계 정보의 효율적인 전파를 가능하게 한다. 이를 통해 기존의 RNN 및 변환기(transformer) 모델의 제한을 극복하며, 다중 스케일 타임 표현을 지원한다.

- **Technical Details**: LDN은 고유한 확산 업데이트를 사용하여 모든 타임 스텝을 통합적으로 업데이트하며, 각 입력 시퀀스의 토큰이 시간적으로 일관된 정보를 공유하도록 설계되었다. 이를 위해, LDN은 다중 스케일 업데이트(multi-scale updates)와 혁신적인 어텐션 모듈을 결합하여 일반적인 RNN 모델의 한계를 극복한다. LDN은 각 레이어에서 서로 다른 확산 속도(diffusion rates)와 커널 매개변수를 채택하여 다양한 시간을 스케일에서의 역동성을 모델링할 수 있다.

- **Performance Highlights**: 벤치마크 시퀀스 모델링 작업에서 실험을 통해 LDN은 전통적인 RNN 및 최신 transformer 모델에 비해 성능과 확장성에서 우수한 결과를 보였다. 이 모델은 글로벌 상호작용(global interaction)을 효과적으로 캡처하면서도 세밀한 시계열 정보를 유지하게 설계되었다. LDN은 효율적인 계산과 강력한 표현 학습 간의 격차를 메우는 중요한 한 단계로 간주된다.




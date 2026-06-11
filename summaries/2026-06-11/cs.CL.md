New uploads on arXiv(cs.AI)

### Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation (https://arxiv.org/abs/2606.12006)
Comments:
          Accepted for publication at International Conference on AI in Healthcare 2026

- **What's New**: 이 연구에서는 생존 예측(survival prediction) 분야에 새로운 방향성을 제시하고 있습니다. 전통적인 survival 분석과는 달리, 사전 훈련된 tabular foundation models를 활용하여 보다 효율적으로 환자의 생존 결과를 예측할 수 있는 경량화된 접근 방식을 제안합니다. 이를 통해 기존의 딥러닝 모델들이 요구하는 대규모 labeled data 없이도 임상 데이터를 다룰 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 연구에서는 TabPFN, TabDPT, TabICL과 같은 전형적인 구조를 연구하고, 각각에 대해 survival-aware head를 통한 파라미터 효율적인 적응 방법을 소개합니다. 이를 통해 오른쪽 검열(right-censoring)이 있는 생존 결과를 모델링할 수 있는 방법을 탐구합니다. 또한 다중 작업 로지스틱 회귀(MTLR) 헤드를 사용하여 질병 예측 모델을 개발하였으며, 다양한 공공 생존 기준 벤치마크와 두 개의 대규모 ICU 집단 데이터셋에 대해 평가합니다.

- **Performance Highlights**: 연구 결과, Tabular foundation models를 사용하여 생존 예측을 수행했을 때 기존의 강력한 베이스라인 모델에 비해 경쟁력 있는 성과를 달성했습니다. 예를 들어, MIMIC-IV 데이터셋에서는 TabDPT-FT-MTLR 모델이 C-index 0.856을 달성하여 가장 우수한 비FM 베이스라인과 비교하여 +1.4%의 상대적 개선을 보였습니다. 또 다른 데이터셋인 eICU에서는 TabICL-FT-MTLR 모델이 C-index 0.797을 달성하여 이전 베이스라인 대비 +1.7%의 개선을 나타냈습니다.



### Time-Series Foundation Model Embeddings for Remaining Useful Life Estimation (https://arxiv.org/abs/2606.11990)
Comments:
          Accepted to EUSIPCO 2026, 4 pages, 2 figures

- **What's New**: 이 연구에서는 Remaining Useful Life (RUL) 예측을 위한 새로운 경량 학습 접근 방식을 소개합니다. 기존의 대규모 라벨링 데이터셋이나 복잡한 feature engineering에 의존하지 않고, frozen pretrained time-series foundation model (TSFM)인 Chronos-2를 활용하여 RUL을 추정합니다. 이를 통해, 멀티 변수 센서 스트림에서의 예측 정확성을 높이면서도 데이터 효율성을 유지합니다. Chronos-2의 고유한 특징을 통해 재현성을 확보할 수 있음을 보여줍니다.

- **Technical Details**: RUL 예측을 위해, 연구진은 완전한 멀티 변수 센서 데이터의 이력 정보를 처리합니다. Chronos-2를 활용하여 frozen backbone을 유지한 채 고차원 기능을 추출하고, 이를 바탕으로 간단한 회귀 네트워크로 RUL을 예측하는 모델로 개발하였습니다. 데이터 전처리 과정에서는 중첩된 센서 데이터를 정규화하고, NaN 값을 포함한 오류 측정치를 걸러내며, 이상치를 전역 클리핑으로 다루었습니다. 마지막으로, 훈련은 특정 RUL 레이블을 기반으로 수행됩니다.

- **Performance Highlights**: 실험 결과, Chronos-2 기반의 모델이 전통적인 재귀(network), 합성곱(convolutional), Transformer 기반 모델 및 그래디언트 부스팅(gradient-boosting) 기반의 기준선보다 성능이 우수하게 나타났습니다. 특히, RUL 예측에서 80 스텝의 긴 이력을 반영하였을 때, 성능이 2배 이상 향상되었으며 MAE가 눈에 띌 정도로 감소했습니다. 이러한 결과는 TSFM이 산업 환경에서 RUL 예측을 위한 효과적이고 데이터 효율적인 대안이 될 수 있음을 시사합니다.



### Exploration Structure in LLM Agents for Multi-File Change Localization (https://arxiv.org/abs/2606.11976)
- **What's New**: 이 논문에서는 소프트웨어 문제를 해결하기 위한 LLM(Large Language Model) 기반 에이전트의 파일 로컬라이제이션(file localization) 접근 방식을 재조명합니다. 특히, 현재의 대부분의 AI 에이전트가 시스템 전역을 선형적으로 탐색하는 방식이 멀티파일 변경에 적합하지 않다는 점을 제기합니다. 이 연구는 비선형, 도메인 범위의 병렬 탐색 방법을 제안하며, 향후 소프트웨어 문제 해결 효율성을 높일 수 있는 잠재력을 보입니다.

- **Technical Details**: 연구진은 SWE Bench Pro를 기준으로, ansible을 예시로 하여 GitHub 문제(GitHub issues)의 지속적 세션 평가 방법을 구축하였습니다. 여기서 비선형 도메인-에이전트 파일 탐색 시스템을 소개하고, 기존의 LLM 시스템인 plain LLM, 단일 에이전트 Recursive Language Model(RLM), 그리고 외부 CLI 기존 시스템과 성능을 비교합니다. 이 과정에서 작은 Haiku 클래스 모델을 사용하여 도메인 범위의 병렬 에이전트 스폰이 가장 높은 마이크로 F1 점수를 달성한 것으로 나타났습니다.

- **Performance Highlights**: 효율적인 로컬라이제이션을 위해 비선형 탐색 방식이 더 효율적이며, 테스트 세트에서 Codex 5.5 High를 사용한 외부 CLI 기준과 경쟁력 있는 성능을 보였습니다. 또한, 문서의 공진화가 일반적인 한계로 남아 있다는 점과, 단순 파일 시스템 접근이 오히려 정확도를 저하시킬 수 있다는 점, 다수의 에이전트를 강제로 함께 작업하게 하는 것이 정확도를 높이지 않는다는 세 가지 주요 발견도 제시하였습니다.



### Categorical Prior Lock-in: Why In-Context Learning Fails for Structured Data (https://arxiv.org/abs/2606.11961)
Comments:
          9 pages, 5 figures. Empirical study of in-context learning and LoRA fine-tuning for synthetic tabular data generation, introducing the phenomenon of categorical prior lock-in. Under review

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 컨디셔널 생성자(conditional generators)로서 구조화된 데이터를 생성하는 과정에서의 인컨텍스트 학습(in-context learning, ICL)의 제한을 조사합니다. 연구자는 ICL이 고차원 범주형 데이터를 다룰 때 발생하는 구조적 실패 모드인 'categorical prior lock-in'을 발견했습니다. 이로 인해 모델이 사전 학습(pre-training)에서 상속된 토큰 분포를 업데이트하지 못하고, 희귀 클래스(represent rare classes) 생성에 실패하는 문제를 다룹니다.

- **Technical Details**: 자세히 살펴보면, 상이한 입력-출력 매핑을 조건부로 근사할 수 있는 ICL의 성능은 조합된 예제의 수와 상관없이 두드러진 한계를 보입니다. 반면에 파라미터 효율적인 미세 조정(parametric-efficient fine-tuning)인 LoRA는 ICL이 수행할 수 없는 전역 분포 조정을 통해 마진(marginal) 및 조인트(joint) 정확도를 크게 개선할 수 있습니다. 그러나 이러한 개선은 데이터 누출(data leakage) 및 프라이버시 위험을 초래할 수 있는 메모리제이션(memorization) 위험도 함께 동반합니다.

- **Performance Highlights**: 연구 결과, 구조적 신뢰성은 생성 시도 중 검증 파이프라인을 통과하지 못하는 비율로 측정되었으며, 이는 배포 효율성의 핵심 지표입니다. 통계적 신뢰성은 합성 데이터와 실제 데이터 간의 총 변동 거리(Total Variation Distance, TVD)를 통해 평가되었으며, 이 결과는 저자들이 제안한 모델의 분포적 적합도(distributional fidelity)와 관련된 몇 가지 중요한 메트릭을 강조합니다. 또한, 모형의 학습 기록과의 근접성을 비교하기 위한 DCR 비율(DCR Ratio)과 같이 메모리제이션 위험을 정량화하는 방법도 포함되었습니다.



### Lung-SRAD: Spectral-Aware Regularized Audio DASS with Dual-Axis Patch-Mix Contrastive Learning for Respiratory Sound Classification (https://arxiv.org/abs/2606.11922)
Comments:
          Accepted to Interspeech 2026

- **What's New**: 이 연구에서는 최근의 음성 소리 분류 방법(RSC)이 Audio Spectrogram Transformer (AST)와 같은 CLS-token 기반의 self-attention 아키텍처에 크게 의존하고 있다는 점을 지적합니다. 이는 전반적으로 효과적일 수 있지만, 저주파 필터링 행동이 특정 이상 패턴을 감지하는 민감도를 저하시킬 수 있다는 분석이 이루어졌습니다. 따라서 본 논문에서는 음성 상태 공간 모델(State Space Models, SSMs)을 RSC의 대안적 백본으로 소개하며, DASS(법모형화된 오디오 상태 공간 모델)를 실험하여 중간 표현을 분석하고 강화된 성능을 보여줍니다.

- **Technical Details**: 본 연구에서는 5.5시간의 녹음과 6,898개의 호흡 주기로 이루어진 ICBHI 데이터셋을 사용하여 세 가지 비정상 폐음인 크랙클, 천명음, 및 둘 다를 포함한 분류 문제에 접근합니다. 모든 모델은 Adam 옵티마이저를 사용하여 훈련되며, 5×10⁻⁵의 학습률을 갖고 있습니다. DASS는 병렬 분석을 통해 중간 계층에서의 주파수 반응을 분석하고, 선택된 계층에 Gaussian convolution을 적용하여 스펙트럼 인식 레귤러리제이션을 제안합니다.

- **Performance Highlights**: 총체적으로 DASS는 ICBHI 데이터셋에서 Score 64.48%를 달성하였으며, 이는 기존의 AST 기준보다 5% 더 높은 성능입니다. 이러한 성과는 도메인에 특화된 dual-axis patch-mix 대조 학습 방식 덕분에 가능하였습니다. 이 연구는 RSC 분야에서 새로운 가능성을 열어주는 중요한 기여를 하고 있습니다.



### Characterizing Software Aging in GPU-Based LLM Serving Systems (https://arxiv.org/abs/2606.11916)
Comments:
          7 pages

- **What's New**: 이 연구에서는 GPU 기반 LLM(대형 언어 모델) 제공 시스템의 소프트웨어 노화(software aging)를 연구하기 위한 실증적 방법론(empirical methodology)을 제안합니다. 기존의 노화 연구는 CPU 중심 소프트웨어에 초점을 맞추고 있으며, 상대적으로 일정한 워크로드(with regular workloads)가 특징입니다. 하지만 LLM 제공 시스템은 Python 호스트와 CUDA 장치가 겹쳐 있는 복잡한 구조를 가지며, 요청의 비용이 큰 차이를 보이고 빠르게 발전하는 소프트웨어 스택에 의존합니다.

- **Technical Details**: 이 연구에서는 동일한 스트레스 조건하에 6개의 동시 배치에서 216시간 동안 캠페인을 운영하며, 호스트, 장치, 클라이언트의 메트릭(metrics)을 동시에 모니터링합니다. 우리는 34개의 지표(indicators)를 추적하여 시스템, 프로세스, GPU 및 클라이언트 측 메트릭을 포함한 통계적 파이프라인(statistical pipeline)을 적용하였습니다. 이 방법은 비모수 트렌드 감지(non-parametric trend detection)와 자가상관(serial-correlation) 보정을 통해 설계되었습니다.

- **Performance Highlights**: 조사 결과, 모든 배치에서 통계적으로 중요한 메모리 노화(memory aging)가 발생하며, 이는 서빙 런타임과 배치 구성에 크게 의존합니다. 세 가지 엔진(vLLM standalone, Triton-wrapped vLLM, naive PyTorch + HuggingFace)이 고유한 노화 패턴을 보이며, 메모리 리크는 최적화된 엔진의 정도와는 상관없이 발생하는 것을 발견했습니다. 최종적으로, 연구의 방법론은 재현 가능(reproducible)하며, 소프트웨어 노화가 현대 LLM 제공에서 실재함을 강력히 입증합니다.



### Quality Adaptive Angular Margin Learning for Respiratory Sound Classification (https://arxiv.org/abs/2606.11915)
Comments:
          Accepted to Interspeech 2026

- **What's New**: 이번 연구에서는 intra-class compactness와 inter-class separability를 강화하여 feature 일반화를 개선하는 quality-adaptive angular-margin learning 프레임워크인 QLung을 제안합니다. 이 프레임워크는 스펙트럼 엔트로피(spectral entropy)와 RMS 에너지를 기반으로 한 no-reference audio quality margin을 도입하여 녹음 품질에 따라 angular margin을 적응적으로 조정합니다. 또한, severe class imbalance 하에서도 훈련을 안정화시키는 log-scaled angular margin을 제안합니다.

- **Technical Details**: 본 연구에서는 unit hypersphere 상에서 feature를 정규화하고 class weight를 사용하여 각 클래스에 대해 일관된 margin penalty를 적용하는 angular classifier를 설계하였습니다. 이로써 정상 호흡과 비정상 호흡 신호 간의 미세한 차이를 효과적으로 분리할 수 있게 되었습니다. 품질 점수(Audio Quality Score, AQS)를 정의하여 각 녹음의 품질을 평가하고, class frequency 기반의 log-scaled class-imbalance margin을 적용하여 불균형한 클래스 데이터에서도 훈련 안정성을 확보했습니다.

- **Performance Highlights**: ICBHI 데이터셋에서 cross-entropy 기본값에 비해 2.46% 성능 향상을 달성하였고, SPRSound 데이터셋에서는 이전 최첨단 방법들보다 더 뛰어난 out-of-distribution 성능을 보였습니다. QLung은 호흡음 분류(RSC) 분야에서 처음으로 angular margin 기반 프레임워크를 적용하여 비정상 호흡 신호의 분류 문제를 효과적으로 해결하고 있습니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### DuoBench: A Reproducible Benchmark for Bimanual Manipulation in Simulation and the Real World (https://arxiv.org/abs/2606.11901)
- **What's New**: DuoBench는 이중 팔 조작을 위한 새로운 벤치마크 프레임워크로, FR3 Duo 플랫폼에서 작동합니다. 이 시스템은 11개의 작업을 통해 4가지 조정 범주를 포괄하며, 시뮬레이션과 현실 세계에서의 재현 가능성을 결합합니다. 또한 이 프레임워크는 섬세한 의미적 실패 분석을 지원하며, 인간 원격 조작 데이터를 제공합니다. 이는 여러 두 팔 정책의 성능을 평가하는 데 중요한 진전을 가져올 것으로 기대됩니다.

- **Technical Details**: DuoBench는 Robot Control Stack (RCS) 생태계 위에 구축되었으며, 각 작업은 Markov Decision Process (MDP)를 기반으로 구성됩니다. 환경은 다양한 포장을 통해 새로운 환경으로 래핑되어 다양한 상태 및 행동 공간을 생성합니다. 이 시스템은 인간 원격 조작자 또는 학습된 정책을 포함한 확률적 에이전트를 사용할 수 있으며, 데이터 레코더 래퍼를 통해 작업 진행 상황을 기록합니다. 이는 각 작업이 인간 및 로봇의 협력적 조작을 평가하기 위한 체계적인 접근 방식을 제공하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 현재 정책들은 이중 팔 조작에서 여전히 도전 과제를 안고 있으며, 특히 초기 상호작용 단계와 팔의 병렬 실행, 시뮬레이션과 현실 세계 간의 전이에 어려움을 겪고 있습니다. DuoBench는 이러한 실패 모드를 진단하고, 이중 팔 정책 학습을 위한 향후 방법을 연구하는 데 도움을 줄 수 있는 재현 가능성 있는 테스트베드를 제공합니다. 벤치마크 결과는 조정 문제의 천착적인 분석을 통해 모델의 약점을 드러내고, 표적 발전을 지원하는 데 중요한 통찰력을 제공합니다.



### Agents All the Way Down; A Methodology for Building Custom AI Agents from Substrate to Production (https://arxiv.org/abs/2606.11869)
- **What's New**: 본 논문에서는 기존의 일반적인 AI 에이전트와 구별되는 커스텀 AI 에이전트(Custom AI agents)의 작성 방법론을 제시합니다. 이 에이전트는 특정 작업을 수행하도록 설계되며, 유지보수를 담당하는 엔지니어가 이를 구축합니다. 또한 현재 실무에서 어떤 커스텀 에이전트를 작성하는지에 대한 명확한 지침이 없음을 지적합니다.

- **Technical Details**: 제안된 방법론은 두 가지 전제조건(P1, P2)과 세 가지 실천(P3, P4, P5)으로 나뉘어 있습니다. P1은 LLM을 도구, 시스템, 메세지의 형태로 프레이밍하며, P2는 기능 호출(function calling), Model Context Protocol(MCP), CLI 오케스트레이션(Orchestration)과 같은 빌딩 블록을 포함합니다. P3에서는 일반 목적 에이전트를 통해 프로토타입을 제작하고, P4에서는 결과를 CLI로 배포하며, P5에서는 일반 목적 에이전트가 행동 시나리오를 통해 새로운 에이전트를 테스트하는 구조입니다.

- **Performance Highlights**: 이 방법론은 특정 언어나 프레임워크에 의존하지 않으며, 커스텀 에이전트를 개발하는 데 있어 보편적으로 적용 가능한 실천 방법으로 제안됩니다. 연구자는 이 방법론을 오픈 소스 LAMB 플랫폼을 위한 AAC(Autonomous Agent with Collaboration)이라는 커스텀 에이전트를 10일 만에 개발한 사례를 통해 입증하였습니다. 멀티 에이전트 오케스트레이션은 CLI 조합으로 단순화될 수 있다는 결론도 제시하고 있습니다.



### Towards Data-free and Training-free Compression for Speech Foundation Models Using Parameter Clustering (https://arxiv.org/abs/2606.11836)
Comments:
          Accepted by Interspeech 2026

- **What's New**: 이 논문은 음성 기초 모델을 위한 새로운 데이터 프리(data-free) 및 훈련 프리(training-free) 압축 방법을 제안합니다. 이 방법은 k-means를 통한 채널별 클러스터링(channel-wise clustering)을 사용하여 구성되며, 층별로 다양한 수의 매개변수 클러스터를 도입하는 미세한 혼합 희소성(pruning)도 탐구합니다.

- **Technical Details**: 제안된 방법은 기존의 중요도 기반 압축 방식과는 달리, 매개변수 간의 유사성을 고려하여 클러스터링과 융합(fusion)을 기반으로 합니다. 이 방법은 HuBERT-large 및 Whisper-large-v3 모델을 대상으로 실험하였으며, 제안된 혼합 희소성 전략을 통해 각 레이어에서 매개변수 클러스터 수를 다르게 설정했습니다. 이를 통해 데이터 없이도 고성능의 압축 모델을 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: HuBERT-large 모델에 대해 50%의 희소성을 적용했을 때, 기존의 magnitude-based pruning과 비교하여 절대 WER(Word Error Rate)가 각각 27.73%와 18.61% 감소하였습니다. Whisper-large-v3에서는 10%의 모델 압축으로 절대 WER가 각각 2.86%와 5.02% 개선되어, 비압축 기준에 비해 성능 저하가 발생하지 않았습니다.



### Designing AI-Supported Focus Groups: A Role x Modality Playbook (https://arxiv.org/abs/2606.11835)
- **What's New**: 본 논문은 디자인 연구에 있어 참여자들의 경험을 수집하는 방법으로서 포커스 그룹의 중요성을 강조합니다. 특히, 참여자들이 서로의 의견에 반응함으로써 비교, 이견 및 집단적 이해를 촉진하는 점에서 포커스 그룹의 가치를 제시합니다. 이와 함께 Generative AI 기술이 포커스 그룹 내의 대화를 지원할 수 있는 방법을 정리한 새로운 플레이북을 제공합니다.

- **Technical Details**: 연구는 AI가 실시간 대화에서 어떻게 여러 역할(도구/tool, 공동 진행자/co-host, 진행자/host) 및 방식(텍스트/text, 음성/voice, 체현/embodied)을 통해 지원할 수 있는지를 설명합니다. 또한, AI의 지원을 받아 진행될 경우 포커스 그룹에서 발생할 수 있는 상호작용적 무역-offs(interactional trade-offs)와 방법론적 리스크를 검토합니다.

- **Performance Highlights**: AI의 적용이 포커스 그룹에서 참여자간의 상호작용 및 대화를 어떻게 변화시키는지에 대한 새로운 통찰력이 제시되었습니다. 특히, AI 도구가 주제를 효과적으로 관리하고 참여를 균형 있게 할 수 있으며, 참여자들의 심리적 안전을 유지하는 데 도움을 줄 수 있다는 점이 강조되었습니다.



### From Uniform to Learned Graph Priors: Diffusion for Structure Discovery (https://arxiv.org/abs/2606.11831)
Comments:
          15 pages, 3 figures, Accepted by KDD 2026

- **What's New**: 이번 논문에서는 구조적 발견의 신뢰성을 저하시켰던 기존의 비현실적인 그래프 사전( prior )을 대체하는 새로운 방법인 Diff-prior를 제안합니다. 기존의 NRI 방법들은 독립적인 에지( edge ) 모델링에 의존했으나, Diff-prior는 확산(diffusion) 과정을 통해 그래프 구조를 더 일관성 있게 조정합니다. 이는 학습 가능한 노이즈 제거(calibration) 스타일의 변환을 통해 scattered하고 불확실한 에지 후방 분포를 정리하는 혁신적인 접근 방식입니다. Diff-prior는 NRI 모델의 성능을 향상시키는 동시에 더 결정적인 에지 후방 분포를 생성할 수 있음을 실험적으로 입증하였습니다.

- **Technical Details**: Diff-prior는 전체 에지-로짓(edge-logit) 구성에 대해 비인과적(non-factorized) 구조 사전을 매개변수화하는 확산 기반(differentiable) 프레임워크입니다. 이는 전통적인 에지 정규화의 한계를 넘어, 교차-에지 종속성(cross-edge dependencies)을 고려하여 전반적인 그래프 일관성을 시행합니다. Diff-prior는 노이즈가 있는 표현을 입력으로 받아들이고, 학습된 구조 패턴에 맞는 정제된 결과를 산출합니다. 이는 확산 모델의 생성력을 활용하여 에지 간의 복잡한 통계적 의존성을 내재화하며 수동으로 설계된 제약을 우회합니다.

- **Performance Highlights**: Diff-prior는 NRI 계열의 여러 테스트에서 성능 개선을 보여주었으며, ablation 실험을 통해 이러한 개선이 매개변수의 수나 훈련 방식에 의한 것이 아니라 사전 모델링에서 기인함을 입증하였습니다. 실험 결과는 Diff-prior가 개선된 샤프하고 더 잘 조정된 에지 신뢰도를 생성함을 나타내며, 이는 엔트로피(entropy)와 ECE(expected calibration error)로 정량화되었습니다. 새로운 프레임워크는 구조적 불확실성을 정교하게 조정하여 더욱 견고한 구조 발견을 가능하게 합니다.



### Feature-Aligned Speech Watermarking for Robustness to Reconstruction Distortions (https://arxiv.org/abs/2606.11828)
Comments:
          Accepted by ICME2026

- **What's New**: 이번 논문에서는 스피치(일본식 말이라면 스피치) 신호의 특징 분포에 맞춰 워터마크를 정렬하는 특징 정렬 워터마킹 기법을 제안합니다. 이는 기존 워터마킹 방법의 내구성과 충실도 간의 트레이드오프 문제를 해결하는 데 중점을 두고 있습니다. 이 방법은 고유한 스피치 특징 분포와의 일치를 통해 높은 에너지의 워터마크를 사용할 수 있으면서도 인지성이 유지됩니다. 실험 결과, 제안된 방법은 기존 방법들과 유사한 수준의 인지성을 유지하면서도 스피치 재구성 모델에 대한 내구성을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 워터마크 임베더와 디코더로 구성된 아키텍처를 가지고 있습니다. 워터마크 임베더는 사전 훈련된 스피치 코덱의 잠재 공간에 워터마크를 삽입하여 기존 스피치와 잘 정렬된 의사 스피치 워터마크를 생성합니다. 이 후, VAD 손실과 인지적 손실을 결합하여 워터마크를 음성 신호의 주파수 영역에서 효과적으로 임베드합니다. 이를 통해 워터마크가 인지되지 않으면서도 높은 에너지를 달성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 ABX 청취 테스트와 VISQOL MOS 결과에서 기존의 임베딩 기반 접근법과 유사한 수준의 인지성을 보이며, 생성 기반 방법들보다 우수한 성능을 보입니다. 다양한 스피치 재구성 모델에 대한 실험을 통해, 훈련 동안 보지 못한 모델에 대해서도 높은 내구성을 지속적으로 달성했습니다. 이러한 성과는 현대 스피치 애플리케이션에서 오디오 워터마킹의 신뢰성을 크게 향상시킵니다.



### Sparsified Kolmogorov-Arnold Networks for Interpretable Quantum State Tomography (https://arxiv.org/abs/2606.11814)
- **What's New**: 이번 연구에서는 Kolmogorov-Arnold Networks (KAN)을 활용하여 양자 상태 톰그래피에서 고 유사성 재구성과 더불어 이 모델의 내부 구조를 검토할 수 있는 방법을 제시합니다. 특히, 3-qubit GHZ 가족을 이용한 벤치마크를 통해, KAN이 기존의 복잡한 MLP 모델 대신 단순화된 방식으로 물리적인 구조와의 일치성을 유지하여 신뢰성 있는 재구성을 수행할 수 있는지 조사하였습니다.

- **Technical Details**: 스파르시파이드 KAN은 Pauli 기대값을 기반으로 하여 물리적 변수들을 재구성하는 네트워크입니다. 이를 통해 훈련된 KAN은 63개의 Pauli 입력 채널을 통해 복잡한 상태를 효율적으로 재구성하며, 각 경로는 Pauli 관측 가능성과 출력 변수 간의 관계를 명확히 나타낼 수 있도록 구성됩니다.

- **Performance Highlights**: 이 연구에서는 KAN의 경로 수준 구조 해석 가능성을 강조하며, 훈련된 모델이 어떤 Pauli 측정을 활용하는지, 그리고 이들이 GHZ 관계와 얼마나 잘 일치하는지를 분석하였습니다. 결과적으로, KAN은 고정밀도의 재구성을 제공하는 동시에 물리적 구조와의 비교 분석을 통해 투명성을 확보하였습니다.



### Multimodal Ordinal Modeling of Alzheimer's Disease Severity Using Structural MRI and Clinical Data (https://arxiv.org/abs/2606.11794)
Comments:
          18 pages. Submitted to journal for review

- **What's New**: 이번 연구에서는 알츠하이머병(Alzheimer's Disease, AD)의 질병 정도를 자동화하고 해석 가능한 방법으로 평가하기 위해 주의 기반의 다중 모달(machine learning framework)을 제안합니다. 기존의 임상 단계는 시간 소모가 크고 변동성이 있기 때문에, 이에 대한 대안으로서 T1-가중치(MRI)와 인구 통계 및 유전적 변수의 통합을 통해 AD의 심각성을 평가할 수 있는 프레임워크를 개발했습니다.

- **Technical Details**: 이 프레임워크는 T1-weighted MRI 데이터와 인구통계학적(demographic) 및 유전적(genetic) 변수를 통합하여, 단일 모달(unimodal) 및 다중 모달(multimodal) 아키텍처를 비교합니다. 훈련과 검증은 ADNI, AIBL, NIFD 데이터셋을 기반으로 한 코호트(stratified) 분할을 통해 이루어졌으며, 데이터 누수를 방지하기 위해 주제 수준(suject-level) 분할이 적용되었습니다.

- **Performance Highlights**: 연구 결과, 단일 모달 접근법 중 T1-weighted MRI 모델이 약간 높은 인접 단계 정확도(0.963)를 달성하였고, 다중 모달 모델에서는 최고 인접 단계 정확도(0.970) 및 임상 단계와의 가장 강한 일치를 보였습니다(QWK 0.549). 이러한 결과는 계층적 구조를 잘 포착하는 정렬(ordinal) 포뮬레이션이 임상적 예측과 더 일치하는 것으로 나타났습니다.



### AI4Land: Scalable Deep Learning for Global High-Resolution Land Use Reconstruction (https://arxiv.org/abs/2606.11793)
- **What's New**: AI4Land는 고해상도 과거 복원 및 미래 예측을 위해 데이터 기반 프레임워크를 제안합니다. 이 프레임워크는 U-Net 아키텍처를 사용하여 연간 토지 이용 및 토지 피복을 재구성함으로써 기후 모델의 불확실성을 줄이는데 기여하고자 합니다. AI4Land는 현실적인 지표를 제공함으로써 직면하고 있는 기후 예측의 주요 한계를 극복할 수 있습니다.

- **Technical Details**: AI4Land의 첫 번째 단계는 1km 해상도의 고해상도 토지 이용(Terrestrial Carbon Cycle) 맵을 생성하는 것입니다. 이를 위해 고 해상도 HILDA+ 데이터셋과 낮은 해상도의 시나리오 데이터를 통합합니다. 입력 데이터는 LUH2 데이터 및 정적 지형 특성으로 구성되며, U-Net 아키텍처는 고해상도 맵을 생성하기 위해 사용됩니다.

- **Performance Highlights**: AI4Land는 MareNostrum5에서 훈련되어 GPU 가속 HPC 인프라를 통해 글로벌 기후 AI 파이프라인을 실현합니다. DDP (Distributed Data Parallelism)를 통해 모델의 확장성을 입증했으며, 최대 8개 노드에서 실질적으로 선형 확장을 유지합니다. 이 시스템은 최대 32개의 H100 GPU를 통해 초당 약 300개의 샘플을 처리할 수 있습니다.



### Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning (https://arxiv.org/abs/2606.11767)
Comments:
          23 pages, 6 figures

- **What's New**: 본 논문에서는 촉각 기반의 맹목적 그립(Blind Grasping) 기술을 물리적 다관절 로봇 손에서 구현하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Real2Sim 촉각 보정 절차를 통해 촉각 신호의 정확도를 높이고, 레이아웃 인식 촉각 인코더를 활용하여 희소한 촉각 관측의 표현력을 개선합니다. 또한, 강화 학습과 확산 정책 학습을 결합하여 다양한 객체 기하형태에 대한 그립 전략을 수립합니다.

- **Technical Details**: 이 연구에서는 시뮬레이터와 실제 로봇 손의 촉각 신호 간의 차이를 줄이기 위한 Real2Sim 보정 절차를 도입합니다. 이 방법은 시뮬레이션된 촉각 이벤트를 실제 발생하는 촉각 이벤트와 정렬시켜줍니다. 또한 각 센서의 3D 위치를 반영한 레이아웃 인식 촉각 인코더를 통해 희소한 촉각 정보에서 객체 정보를 우선적으로 추출할 수 있는 방법을 제공합니다. 마지막으로, 성공적인 촉각 그립 궤적을 수집하여 촉각 조건에 맞춘 확산 정책을 학습합니다.

- **Performance Highlights**: LEAP 핸드를 이용한 실제 실험에서 제안된 정책은 20개의 다양한 객체에 대해 27%의 실제 그립 성공률을 기록했습니다. 이는 실제 그립 시연이나 시각적 입력 없이도 가능하다는 점에서 주목할 만합니다. 시뮬레이션 실험에서는 레이아웃 인식 촉각 사전 학습이 그립 성능을 개선했다는 결과를 보여줍니다. 또한 Real2Sim 보정은 시뮬레이션과 하드웨어 간의 촉각 접촉 사건의 일관성을 높인 것으로 확인되었습니다.



### T2S: A Rehearsal-Based Approach for Extraction-Resistant Model Watermarking (https://arxiv.org/abs/2606.11698)
- **What's New**: 이 논문은 AI 모델의 지적 재산권을 보호하기 위해 수중에 고유한 행동 서명을 유도하는 독특한 지식을 삽입하는 모델 워터마킹을 다룹니다. 특히 모델 추출 공격에 대한 워터마크의 강인성을 높이기 위한 리허설 기반의 워터마크 삽입 프레임워크를 제안합니다. 기존의 접근 방식과 달리, 이 방법은 시뮬레이션 된 도난 모델의 손실을 훈련 신호로 활용하여 타겟 모델 내 워터마크 지식을 미세 조정합니다.

- **Technical Details**: 제안된 방법은 주로 타겟 모델의 워터마크 지식을 시뮬레이션 된 도난 모델의 워터마킹 손실을 통해 직접적으로 미세 조정하는 구조를 가지고 있습니다. 또한, 세 가지 유형의 트리거 세트로 이 프레임워크를 구현하여 그 일반성과 다양성을 입증합니다. 논문에서는 이 방법이 모델 추출 공격 및 이후의 워터마크 제거 공격에 대한 저항력을 크게 향상시킨다는 것을 여러 실험을 통해 확인합니다.

- **Performance Highlights**: 실험 결과는 이 방법이 도난 모델에서도 높은 워터마크 감지율을 보이며, 잘못된 긍정율(FPR)이 거의 없는 것을 보여줍니다. 제안된 리허설 기반 워터마킹 접근법은 보안에 중요한 기여를 하며, 저자들은 실험을 통해 다양한 모델 추출 공격 및 워터마크 제거 작업에 대한 강한 저항성을 입증하였습니다.



### Noise-Aware Framework for Correcting Corrupted Labels (https://arxiv.org/abs/2606.11695)
- **What's New**: CANOLA는 데이터셋 내의 부정확한 레이블을 수정하기 위한 새로운 프레임워크로, 노이즈 인식 학습(noise-aware learning)과 반복적인 레이블 정제(iterative label refinement)를 통해 문제를 해결합니다. 기존의 방법들과는 달리, CANOLA는 모델 학습 과정에서 노이즈 분포(noise distribution)를 명확히 추정하여 보다 신뢰할 수 있는 패턴에 집중하도록 합니다. 이를 통해 모델의 강건성과 일반화 능력을 향상시키는 데 기여합니다.

- **Technical Details**: CANOLA는 특정 노이즈 특성을 통합하여 신뢰할 수 없는 감독 신호를 줄이고, 모델의 예측을 안정적으로 정제하는 방법론을 사용합니다. 이 과정은 모델 학습 후, 학습이 안정된 상태에서 이루어져 예측의 신뢰성을 극대화합니다. 레이블 정제는 모델의 예측과 관찰된 레이블을 블렌딩하여 수행하며, 이를 통해 데이터를 점진적으로 안정적이고 조절된 방식으로 복구합니다.

- **Performance Highlights**: 실험 결과, CANOLA는 여섯 개의 데이터셋에서 일관되게 SOTA(label correction) 방법들을 능가하며, 전반적으로 원래 데이터셋 오류율을 약 25% 줄였고, 높은 노이즈 조건에서 더욱 효과적인 개선을 보였습니다. CANOLA로 수정된 데이터셋을 사용해 학습한 모델은 모델 중심의 방법들보다 최대 67%까지 성능이 향상되는 결과를 나타냈습니다.



### Can Open-Source LLM Agents Replace Static Application Security Testing Tools? An Empirical Assessmen (https://arxiv.org/abs/2606.11672)
Comments:
          Keywords: Agentic AI, Cybersecurity, Large Language Models, Static Application Security Testing, Model performance evaluation

- **What's New**: 이 논문은 사이버 보안 목적으로 에이전틱(Agentic) AI 도구의 가치를 탐구합니다. 특히, 다양한 Ollama 기반의 오픈 소스 모델로 구동되는 일반 목적의 GenAI 대형 언어 모델(LLM) 기반 에이전트의 효율성을 평가합니다. 기존의 검증된 정적 애플리케이션 보안 테스트(SAST) 도구인 Bandit와 비교하여 에이전트의 성능을 정밀도(precision), 재현율(recall), 허위 양성(false positive) 수치 등을 사용해 평가한 결과, 현대의 오픈 소스 GenAI LLM 기반 에이전트가 현실적인 조건에서 SAST 스캔에 적합하지 않음을 발견했습니다.

- **Technical Details**: 2023년 이후 LLM과 인공지능 분야는 폭발적인 발전을 겪었습니다. 이 기술은 3년 전과 비교하여 거의 알아볼 수 없는 수준으로 향상되었습니다. 에이전틱 AI 시스템은 목표 지향적 행동, 동적 적응 및 자기 개선 기능을 특징으로 하며, 이는 Generative AI의 발전 덕분에 가능해졌습니다. 이 연구에서는 LLM 기반 사이버 보안 스캐닝 에이전트가 기존 도구들과 비교했을 때 성능을 평가하기 위한 실험적 데이터를 제공합니다.

- **Performance Highlights**: 실험 결과, 생성된 에이전트는 기존 SAST 도구와 비교할 때 여러 측면에서 부족함을 보였으며, 이는 규칙적인 상황에서의 스캔 작업에 부적합함을 증명합니다. 연구 결과는 에이전틱 AI 도구가 아직은 전문적인 사이버 보안 테스트 작업에 대해 충분한 효과를 발휘하지 못하고 있다는 것을 보여줍니다. 이러한 발견은 개발자들 및 중소 팀들이 에이전틱 보안 스캐너의 힘과 단순함을 활용할 수 있는 가능성을 언급하지만, 실제로는 명확한 성과를 제공하지 못함을 입증했습니다.



### Runtime Skill Audit: Targeted Runtime Probing for Agent Skill Security (https://arxiv.org/abs/2606.11671)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 에이전트의 스킬이 지침, 자원, 도구 및 워크플로우를 재사용할 수 있도록 해주지만, 악의적인 행동을 숨길 새로운 공간을 만든다고 설명합니다. 연구진은 스킬이 실제로 특정 사용자 요청이나 다단계 도구 상호작용에서 호출될 때만 해를 끼칠 수 있음을 강조하며, 그리하여 정적 검증(static vetting) 방법의 취약성을 지적합니다.

- **Technical Details**: 이 연구에서 제안된 러untime Skill Audit(RSA)는 특정 런타임 조건 하에서 스킬 중재 에이전트가 실제로 수행하는 작업을 감사(audit)하는 동적 분석 방법입니다. RSA는 리스크 관련 인터페이스를 프로파일링하고, 이들을 실행하는 데 필요한 컨텍스트를 준비하며, 그 결과에서 보안 레이블(security labels)을 할당합니다.

- **Performance Highlights**: RSA는 OpenClaw에서 구현되었으며, 대표적인 정적 기준선(static baselines)과의 비교를 통해 100개의 스킬을 평가했습니다. RSA는 90.0%의 정확도로 88.0%의 진짜 양성률(true positive rate)과 8.0%의 가짜 양성률(false positive rate)을 달성하였으며, 최고 정적 기준선보다 정확도를 13.0% 포인트 개선했습니다. 정적 감지기가 한두 번의 공격 후 실패하는 반면, RSA는 연속적인 공격에서도 20가지 악의적인 스킬을 모두 탐지할 수 있었습니다.



### Sparse probes and murky physics: a case study of interpretability challenges in a foundation model for continuum dynamics (https://arxiv.org/abs/2606.11657)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 강력한 이론과 벤치마크, 물리적 직관이 이미 존재하는 과학 분야에서 생성적 AI 에뮬레이터의 사용이 증가하는 이유를 다루고 있습니다. Polymathic의 Walrus라는 기초 모델을 연구의 대상으로 삼아, 물리 원칙에 기반한 메커니즘 해석 가능성(mechanistic interpretability)을 조사합니다. 이 연구는 모델이 알려진 연속 역학을 재현하는 경우, 그 내부 메커니즘이 어떻게 작동하는지를 평가하는 질문을 제기합니다.

- **Technical Details**: 우리는 희소 오토인코더(sparse autoencoder, SAE)를 사용하여 Walrus의 특정 층을 탐색하고, 20,000개 이상의 특징 집합 중에서 enstrophy란 물리적 지표를 기반으로 한 실제적인 도전 과제를 해결합니다. 이 연구는 전단 흐름을 중심으로 여러 시뮬레이션 설정 전반에서의 특징의 채택 과정을 비교합니다. 예를 들어, 우리가 조사한 시뮬레이션의 매개변수 값에 따라 특징 일관성을 분석하며, 이 결론은 전형적인 물리적 해체와 완전히 일치하지 않는 구조를 발견합니다.

- **Performance Highlights**: Walrus는 물리적 시나리오를 아우르는 1.3B 매개변수 변환기 기반 모델로, 이를 통해 예측 정확성이나 계산 효율성 면에서 이전의 기초 모델들과 경쟁하는 성과를 보여주었습니다. 그러나 직접적인 수치 시뮬레이션과 에뮬레이터 간의 시스템적 출력 차이가 존재함을 발견하였으며, 이는 SAE 특징 사용의 변화와 관련이 있습니다. 이 연구는 기초 모델의 과학적 활용에서 기능적으로 의미 있는 특징을 추출하는 방법, 분석 아티팩트와 안정적 구조를 구분하는 방법, 그리고 기존 벤치마크를 사용하여 모델의 내부 표현이 유용한지를 결정하는 방법에 대한 자료를 제공합니다.



### TAROT: Task-Adaptive Refinement of LLM-prior Graphs for Few-shot Tabular Learning (https://arxiv.org/abs/2606.11640)
- **What's New**: 본 논문에서는 TAROT라는 새로운 GNN(Graphic Neural Network) 기반 프레임워크를 제안하여 구조적 및 의미적 사전 정보를 활용해 적은 수의 샘플로 학습할 수 있는 방법을 제안합니다. TAROT는 의미적 그래프(Semantic Graph)를 생성하고 이를 정제함으로써 예측 성능을 향상시키며, 이는 기존의 몇 가지 문제점을 해결하는 데 중점을 두고 있습니다. 기존 방법들이 큰 데이터에 의존하고 있었다면, TAROT는 적은 데이터 환경에서도 의미 있는 피처 상호작용을 모델링할 수 있음을 강조합니다.

- **Technical Details**: TAROT는 통합된 의미적 테이브 노드 인코더(USTNE)를 사용하여 다양한 테이블 데이터를 통합된 노드 의미 표현으로 인코딩합니다. 이후 LLM(대형 언어 모델)은 작업 설명 및 피처 이름에 기반하여 의미적 관계를 추론하여 초기 의미적 그래프를 구성합니다. TAROT는 이후 작업 적응형 의미적 그래프 정제를 통해 무관한 엣지를 제거하고 중요한 엣지를 추가하여 구조적 노이즈를 줄이고, 최종적으로 정제된 그래프에서 메시지 패싱을 수행합니다.

- **Performance Highlights**: TAROT는 다양한 적은 샷 테이블 학습 벤치마크에서 기존의 최첨단 성능을 지속적으로 초과하는 결과를 보여주었습니다. 특히, 11개의 실제 데이터셋에 대한 실험으로 TAROT의 성능이 기존 방법들보다 뛰어난 것을 확인할 수 있습니다. 이를 통해 TAROT가 적은 수의 샘플로도 강력한 예측을 가능하게 하는 혁신적인 접근법임을 입증하였습니다.



### Are LLMs Bad at Moral Reasoning? (https://arxiv.org/abs/2606.11635)
- **What's New**: 이 논문은 AI 시스템이 도덕적 이유를 이해하고 반응할 수 있는 능력, 즉 도덕적 능력(moral competence)의 평가에 대해 새로운 관점을 제시합니다. 최근의 연구들은 현재의 AI 모델이 도덕적 추론에서 부족하다는 비관적인 결과를 내놓고 있지만, 저자들은 MoReBench 데이터세트를 새롭게 활용하여 LLM들이 기존의 평가 기준보다 더 높은 도덕적 능력을 보여준다고 주장합니다. 이를 통해 AI의 도덕적 추론 능력에 대한 보다 긍정적인 그림을 그릴 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 LLM이 도덕적 문제들을 분석하기 위하여 작성한 루브릭(rubric)의 품질을 평가하며, 모델들이 생성한 루브릭이 인간 전문가의 루브릭보다 더 많은 도덕적 고려사항(moral considerations)을 포함하고 있다고 합니다. 연구에 따르면, 모델들이 작성한 루브릭은 인간 루브릭을 기준으로 약 83%에서 89%의 고려사항을 포착하고, 인간에 비해 2.26배 이상의 독창적인 도덕적 고려사항을 포함하고 있어 LLM의 도덕적 분석 능력을 입증합니다. 이러한 연구 결과는 모델들의 도덕적 추론이 더 뛰어나다는 주장을 뒷받침하는 것입니다.

- **Performance Highlights**: 논문은 LLM의 도덕적 판단이 더 많은 도덕적 고려를 포함하며, 정량적 성과가 인간 수준에 근접함을 보여줍니다. 모델들은 다양한 도덕적 문제에 대해 더욱 까다로운 루브릭을 작성할 수 있으며, 이전 연구에서 나타난 부족한 점들이 개선된다는 것을 발견했습니다. 이로 인해 LLM의 도덕적 추론 능력에 대한 신뢰도가 높아지며, 향후 더 나은 평가 기준 수립을 위한 기초를 제공합니다.



### Sovereign Assurance Boundary: Certificate-Bound Admission for Agentic Infrastructur (https://arxiv.org/abs/2606.11632)
Comments:
          12 pages, 1 figure, 13 tables

- **What's New**: 이 논문은 Sovereign Assurance Boundary (SAB)라는 새로운 인증 기반의 런타임 입장 레이어를 도입합니다. 이는 고위험 프로덕션 리소스에 대한 비결정적(reasoning) 시스템의 돌연변이를 제안할 수 있는 문제를 해결합니다. 기존의 접근 방식들이 정적이고 맥락 인식이 부족한 권한을 강제하거나, 실행 후 행동만 기록하는 것에 그치는 반면, SAB는 제안된 행동이 실행될 수 있는지 결정하는 새로운 메커니즘을 제공합니다.

- **Technical Details**: SAB는 에이전트 프로포절을 보증 공기 잠금장치에서 가로채고, 이들을 타입화된 실행 계약으로 정리하여, 암호화된 증거 다이제스트와 정책 버전에 바인딩합니다. 이는 제안된 행동이 특정 실행 권한, 철회 시점 및 유효성 창에 국한된 서명된 Sovereign Assurance Certificate를 발급받도록 합니다. SAB는 개별 행위와 시스템 수준의 보증을 분리하여, 직접적인 주 상태 변경을 방지합니다.

- **Performance Highlights**: 초기 Go 프로토타입을 개발하여 2,500회의 입장 시도를 평가했고, 입장 대기 시간, 브로커 검증, 철회 전파, 루팅 정확성, 재생 완전성 및 증명 오버헤드에 대한 초기 측정치를 보고했습니다. 이 연구는 에이전트 시스템에 대한 안전한 인증 메커니즘을 강화하는 방향으로 큰 기여를 할 것으로 기대됩니다.



### LUCID: Learning Embodiment-Agnostic Intent Models from Unstructured Human Videos for Scalable Dexterous Robot Skill Acquisition (https://arxiv.org/abs/2606.11628)
- **What's New**: 이번 논문에서는 구조화되지 않은 인간 비디오에서 로봇의 태스크 의도를 학습하는 LUCID라는 두 단계의 프레임워크를 제안합니다. 기존 로봇 학습 파이프라인은 로봇 시연이나 구조화된 인간 데이터를 기반으로 한 반면, LUCID는 인터넷 규모의 비디오 데이터에서 다양한 조작 시연을 활용합니다. 이를 통해 로봇 제어를 대규모 시뮬레이션에서 학습하며, 서로 다른 로봇 구현에도 동일한 의도 모델을 적용할 수 있습니다.

- **Technical Details**: LUCID는 두 가지 디자인 선택을 통해 의도(what should change in the scene)와 제어(how the robot achieves it)를 분리합니다. 첫째, 의도 모델은 비디오에서 예측한 짧은 각도에서의 의도 정보와 팔 자세(palm pose) 정보를 제공하며, 두 번째로 이러한 예측을 폐쇄 루프(closed loop) 시스템으로 연속적으로 갱신합니다. 결과적으로, 실행 시에 대형 비디오 데이터를 필요로 하지 않습니다.

- **Performance Highlights**: LUCID는 다섯 가지 실제 조작 태스크에서 평가되었으며, 웹 비디오로 감독된 작업에서 평균 73%의 성공률을 달성하였습니다. 기존 오픈 루프(open-loop) 기반 접근법에 비해 큰 개선을 보여주었고, 동일한 의도 모델로 다양한 로봇 구현에서 유사한 성공률을 보였습니다. 이러한 성과는 훈련된 비디오 데이터의 양에 따라 예측 가능하게 향상된다는 것을 보여줍니다.



### When Context Returns: Toward Robust Internalization in On-Policy Distillation (https://arxiv.org/abs/2606.11627)
- **What's New**: 최근 연구에서는 on-policy distillation을 통해 학생 모델이 시스템 프롬프트나 태스크 힌트와 같은 특권적 맥락(priviliged context)을 내재화할 수 있음을 보여주었다. 이 방법은 학생 모델의 무맥락(no-context) 성능을 향상시켰으나, 흥미롭게도 원래의 특권적 맥락을 다시 도입하면 오히려 성능이 저하되는 현상을 발견하였다. 우리는 이를 맥락 유발 저하(context-induced degradation)라고 명명하였다.

- **Technical Details**: 이 연구는 학생 모델이 특권적 정보를 내재화하기 위해 context removability라는 추가적인 특성이 필요하다고 주장한다. 연구팀은 No-Context Anchoring (NCA)이라는 간단한 일관성(Consistency) 정규화기를 제안하였으며, 이 정규화기는 학생 모델의 무맥락 출력을 고정시키고 맥락조건 출력이 이를 이탈하지 않도록 페널티를 부여한다. 이 과정은 훈련 단계마다 단지 한 번의 추가적인 forward pass를 필요로 한다.

- **Performance Highlights**: 연구 결과, NCA는 12개의 다양한 설정에서 대부분의 경우 맥락조건 정확도를 향상시켰으며, 12개 설정 중 11개에서는 맥락 유발 저하를 감소시키고 응답 길이의 비대칭성을 효과적으로 제거하였다. 기계적 분석 또한 NCA가 출력 수준뿐만 아니라 표현 수준에서도 맥락 제거 가능성을 달성한다는 것을 확인하였다. 이러한 결과는 NCA가 명확한 설정에서도 효과적임을 보여준다.



### Physics-Distilled Neural Network enabled by Large Language Models for Manufacturing Process-Property Predictive Modeling (https://arxiv.org/abs/2606.11605)
Comments:
          Under review, Journal of Computing and Information Science in Engineering

- **What's New**: 이 논문에서는 제조 공정-속성 관계 예측의 어려움을 해결하기 위해 새로운 지식 증류(knowledge distillation) 프레임워크를 제안합니다. 이 프레임워크는 데이터가 부족한 상황에서도 높은 정확도로 예측할 수 있도록 설계되었습니다. 특히, 과학 문헌에서 체계적으로 추출된 물리학적 분석 사전(analytical physics priors)을 이용하여 특권 교사 모델(privileged teacher model)에 통합합니다.

- **Technical Details**: 프레임워크는 입력 변수 간의 복잡한 물리적 의존성을 캡처하기 위해 Graph-Masked Attention 레이어를 사용합니다. 이는 엄격한 목표 지점(setpoints) 및 정적 또는 고주파 신호의 조합을 포함한 입력 변수를 반영합니다. 지식이 증류되어 경량의 학생 예측기(student predictor)가 생성되며, 이는 예측 및 추론에 활용됩니다.

- **Performance Highlights**: 다양한 5가지 제조 공정을 대상으로 한 포괄적인 실험을 통해 프레임워크의 타당성과 견고함이 평가됩니다. 결과적으로, 해당 프레임워크는 평가된 모든 도메인에서 높은 예측 정확도를 지속적으로 달성합니다. 특히, 유연한 추론 빈도(inference frequency)를 통해 최대 6000 Hz를 초과하여 실시간 엣지 배포(edge deployment)가 가능하다는 점이 강조됩니다.



### Model-Based and Data-Driven Hierarchical Control and Topology Co-Design for Robust Networked Systems (https://arxiv.org/abs/2606.11596)
Comments:
          To be submitted to Automatica

- **What's New**: 본 논문에서는 선형 서브시스템으로 구성된 네트워크 시스템을 고려하고, 이를 위한 새로운 제어 설계 전략을 제안합니다. 기존의 모델 기반 접근법에 의해 서브시스템에 대한 로컬 분산 제어기를 설계하고, 이를 통해 전역적인 제어 시스템의 지속 가능성을 보장하도록 합니다. 또한, 데이터 기반 접근법을 제시하여 실제 시스템에서 동적 정보가 부족할 경우에도 적용 가능하도록 하였습니다.

- **Technical Details**: 본 연구는 두 가지 접근법을 통해 네트워크 시스템의 제어를 다룹니다. 첫 번째는 모델 기반 계층 제어 설계(MPC, Model Predictive Control)로, 각 서브시스템의 동적 특성을 이용하여 로컬 제어기를 설계합니다. 두 번째는 데이터 기반 계층 제어 설계로, 이는 서브시스템의 입력-상태-출력 궤적 데이터만을 사용하여 제어기를 설계할 수 있도록 구성되어 있습니다.

- **Performance Highlights**: 제안된 제어 설계 전략은 DC 마이크로그리드(networked system) 사례를 통해 효과성을 입증하였습니다. 이 시스템은 강건한 전압 조정 및 전류 분배를 보장하도록 설계되어, 전반적인 효율성을 높이는 동시에 불확실성을 줄이는 데 기여합니다. 이를 통해 로컬 및 글로벌 지속 가능한 특성을 보장하며, 시스템의 비용 최적화도 가능합니다.



### ConsistencyPlanner: Real-time Planning with Fast-Sampling Consistency Models (https://arxiv.org/abs/2606.11569)
- **What's New**: 이번 논문에서는 상호작용이 있는 복잡한 실제 주행 시나리오를 위한 닫힌 루프 계획(closed-loop planning)의 새로운 접근법인 Consistency Planner를 제안합니다. 이는 다중 모드 행동( multimodal behavior) 표현과 실시간 계획(real-time planning)의 균형을 맞추기 위한 것으로, 빠른 샘플링(consistency models) 기법을 사용하여 다양한 예측 궤적들을 효율적으로 생성합니다. 또한, 주행 시나리오에서 복잡한 다양한 입력 특징들을 동적으로 통합하는 주의력 강화 디코더(attention-enhanced decoder)를 도입하여 계획의 견고성을 강화합니다.

- **Technical Details**: Consistency Planner는 전통적인 규칙 기반 방법과 학습 기반 방법의 장점을 결합한 프레임워크입니다. 이 시스템은 빠른 샘플링(consistency models)을 통해 실제 주행 데이터셋에 적합한 네트워크 아키텍처를 개발하고, 서로 다른 입력 특징(예: 도로 정보, 차량 상태)을 효율적으로 융합하여 최적의 행동 계획을 수립합니다. 이를 통해 모델은 다수의 가능성 있는 미래 경로를 신속하게 탐색할 수 있으며, 기존의 반복적 생성 방법에서 발생하는 계산 병목 현상을 극복합니다.

- **Performance Highlights**: Waymax 시뮬레이터에서의 실험 결과, Consistency Planner는 기존 학습 기반 기법들에 비해 안전성 메트릭에서 우수한 성능을 기록했습니다. 특히 동적인 시나리오에서 강력한 결과를 보여주었습니다. 이 연구에서 제안된 방법은 닫힌 루프 계획의 효율성을 극대화하며, 안전-critical 시스템으로서 자율 주행에 적합한 실시간 응답성을 유지합니다.



### LLMs+Graphs: Toward Graph-Native, Synergistic AI Systems (https://arxiv.org/abs/2606.11560)
Comments:
          10 pages, Accepted at PAKDD 2066 Tutorial

- **What's New**: 이 논문은 최근의 대형 언어 모델(LLMs)과 그래프 기반 데이터 처리 기술의 융합을 다루고 있습니다. LLMs의 효율성을 향상시키기 위해 그래프 계산을 이용한 정보 검색과 추론이 어떻게 결합될 수 있는지를 탐구하며, 지식 그래프(KGs)와 LLM 간의 상호작용을 통해 세부적인 제약과 일관성을 유지하는 방법을 설명합니다. 이는 데이터 과학 및 데이터 마이닝 연구자들에게 혁신적인 그래프-네이티브 AI 시스템의 필요성을 강조합니다.

- **Technical Details**: 논문에서는 LLMs가 자연어 질문을 처리하는 능력을 활용하여 그래프 쿼리와 그래프 마이닝 작업을 수행하는 방식을 다룹니다. LLMs는 그래프 구조를 통해 복잡한 데이터 문제를 해결하고, GNN(그래프 신경망)과의 통합을 통해 제로샷 reasoning이라는 새로운 가능성을 창출합니다. 또한, 그래프 기반 검색 증강 생성(Graph RAG)을 통해 문서에서 구조적이고 관계가 풍부한 컨텍스트를 효과적으로 활용하는 방법에 대해서도 설명하고 있습니다.

- **Performance Highlights**: 이 연구는 LLMs와 KGs가 상호 보완적으로 작용하여 정보 검색과 의사결정을 개선하는 여러 방법을 보여주고 있습니다. 예를 들어, Microsoft의 GraphRAG와 ArchRAG는 사용자 쿼리를 보강하고 정확도를 높이는 방향으로 설계되었습니다. 이러한 기법들은 의료, 교육, 전자 상거래와 같은 도메인에서도 응용될 수 있어, LLM-KG 융합의 잠재력을 강하게 시사합니다.



### Privacy-Preserving Federated Autoencoder for ECG Anomaly Detection on Edge Devices (https://arxiv.org/abs/2606.11556)
Comments:
          9 pages, 4 figures, 6 tables. Preprint prepared in IEEE conference format. Submitted to: FLTA 2026

- **What's New**: 이 논문은 연속적인 심전도(ECG) 모니터링 시스템의 개발을 통해 리듬 비정상을 조기에 감지하여 심혈관 사건으로의 발전을 방지할 수 있는 새로운 접근법을 제안합니다. 이 시스템은 법률 기준의 프라이버시(GDPR, HIPAA)를 준수하면서도 실시간으로 비정상을 탐지할 수 있어야 하며, 비독립적(non-IID) 병원 데이터에서도 높은 탐지 품질을 유지해야 합니다. 제안된 연합 학습(federated learning) 기반의 시스템은 비지도 학습(unsupervised) 환경에서도 성능을 저하 없이 실시간으로 동작할 수 있도록 설계되었습니다.

- **Technical Details**: 이 시스템은 VanillaAE, ConvAE 및 VAE 등 세 가지 오토인코더(autoencoder) 계열을 조합하여 PTB-XL 데이터셋에서 비정상 탐지를 수행합니다. 연합 평균(FedAvg)과 차별적 프라이버시(differential privacy)에 기반한 확률적 경량화(Stochastic Gradient Descent, DP-SGD)를 적용하고, 라즈베리 파이 4의 성능을 기반으로 8비트 정수(INT8) 양자화(post-training quantization)를 수행하여 모델 크기를 줄이고 처리 속도를 높입니다. 특히 DP와 양자화의 성능 저하가 독립적임을 확인하여, 강력한 프라이버시 보장을 유지하면서도 경량화된 모델을 사용할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 연합 학습 및 DP 적용 시스템이 중앙 집중형 기반라인 대비 모든 구조에서 동등하거나 높은 성능을 보였으며, ConvAE의 경우 수신자 조작 특성 곡선(AUROC)에서 0.782에 도달했습니다. INT8 양자화를 통해 모델 크기를 약 50% 줄이고 라즈베리 파이 4의 지연 시간을 44%까지 단축하였으며, AUROC 손실은 0.12% 미만으로 유지되었습니다. 이러한 성능 결과는 ECG 데이터의 비지도 복원 기반 탐지에서 독보적인 성공 사례로 기록될 것입니다.



### End-to-End Machine Learning for Depressive State Classification via EEG and fNIRS (https://arxiv.org/abs/2606.11555)
Comments:
          4 pages, 4 figures, Accepted for publication in the Proc. 48th Annu. Int. Conf. IEEE EMBS (EMBC 2026), Toronto, Canada, July 20-24, 2026

- **What's New**: 정신 건강 관리에 대한 수요가 증가하면서 전통적인 정신과 진단의 한계가 강조되고 있습니다. 기존의 진단 방법이 주로 임상 인터뷰와 환자의 자기 보고에 의존하는데, 이는 주관적 편향과 의사 개인의 판단에 영향을 받기 쉽습니다. 이에 따라, 생물학적 신호 기반의 진단 방법, 특히 EEG와 fNIRS를 활용한 방법이 유망한 대안으로 떠오르고 있습니다.

- **Technical Details**: 본 연구에서는 11명의 건강한 대학생을 대상으로 EEG와 fNIRS를 동시 기록하여 우울 증상 조기 감지를 위한 프레임워크를 제안합니다. 실험 참가자들은 감정 자극을 기억하는 작업을 수행하였고, 이를 통해 신경 반응을 분석했습니다. Beck Depression Inventory(BDI) 점수를 사용하여 다중 모달 신호를 레이블링하고, SincShallowNet이라는 신경망을 통해 분석의 타당성을 평가했습니다.

- **Performance Highlights**: 이 연구의 예비 결과는 건강한 피험자 집단에서도 미세한 우울 경향을 감지할 수 있는 가능성을 보여줍니다. EEG와 fNIRS의 통합 사용은 정신 상태를 평가하는 데 있어 전통적인 방법보다 더 객관적이고 신뢰할 수 있는 결과를 제공합니다. 이러한 접근 방식은 정신 건강 진단의 자동화 및 객관화를 위한 기초적인 단계로 평가됩니다.



### AI Researchers Must Help Lead Arms Control to Mitigate Military AI Risks (https://arxiv.org/abs/2606.11533)
Comments:
          9 pages, 1 figure, ICML 2026 Position Paper

- **What's New**: 이번 논문은 AI의 군사적 응용에 대한 규제가 필요한 긴급한 과제임을 강조합니다. AI 기술의 발전은 군사 구조에도 깊은 영향을 미치고 있으며, 이에 따라 학계와 군사 전문가들 간의 협력이 필수적입니다. 특히, AI 연구자들은 군사적 AI 시스템의 안정성을 정의하고 완화하는 기술 연구를 선도해야 한다고 주장하고 있습니다.

- **Technical Details**: AI 안전성 프레임워크는 AI 시스템과 인간의 가치 간의 정렬을 보장하는 필수적인 가이드라인을 제공합니다. 하지만 현재의 안전성 방법론은 군사적 맥락에서 배치되는 최신 AI 모델의 리스크를 신뢰성 있게 완화하지 못하고 있습니다. 군사 AI의 새로운 리스크를 해결하기 위해서는 다국적 대화가 필요하며, 특정 연구 방향에 대한 제약을 설정하고 잠재적 적국 간의 외교적 합의를 개발해야 합니다.

- **Performance Highlights**: AI 군비 통제는 공공 안전, 전략적 억지력, 그리고 전 세계의 권력 균형을 해치는 군사 AI 응용의 개발과 배치를 제한하는 외교적 프레임워크로 정의됩니다. 특히, 상호 확신 AI 오작동(MAIM)과 같은 새로운 억제 체제가 제안되어 전통적인 핵 억제 이론과 유사한 방식으로 국가 간의 AI 지배권 경쟁을 방지하려는 노력이 이루어지고 있습니다.



### SirenFNO: Efficient and Full Frequency Learning of Fourier Neural Operators (https://arxiv.org/abs/2606.11518)
Comments:
          9 pages, accepted by IJCAI 2026

- **What's New**: 이번 논문은 Fourier Neural Operator(FNO)의 한계점을 극복하기 위해 새로운 프레임워크인 SirenFNO를 제안합니다. 기존 FNO는 주파수 절단(frequency truncation)에 의존하여 낮은 주파수 정보에 편향된 경향이 있었고, 이러한 특성이 고주파 신호를 캡처하는데 어려움을 겪게 했습니다. SirenFNO는 SIREN(Sinusoidal Representation Networks)을 활용하여 중첩 신경망(hypernetwork)를 통해 효율적인 학습을 가능하게 합니다.

- **Technical Details**: SirenFNO는 SIREN을 사용하여 주파수 모드 별로 커널 매개변수를 동적으로 생성하여 주파수 절단을 피하고, 이러한 접근법은 다양한 격자 해상도에서의 학습을 가능하게 합니다. SIREN의 사인 함수 활성화는 고주파와 저주파 정보를 동시에 효과적으로 포함시켜, 특히 주파수 편향을 줄이는 데 유리합니다. 또한, 본 논문에서는 기능적 텐서 분해를 통해 매개변수 효율성을 향상시키고 있습니다.

- **Performance Highlights**: 실험 결과 SirenFNO는 기존 FNO와 비교하여 약 4배에서 15배의 매개변수 감소를 달성하면서도 격자 불변 특성을 유지합니다. 기능적 분해 방식의 변형에서는 최대 73배 적은 매개변수를 사용하여 성능이 개선됨을 보여주었습니다. 이러한 성능 개선은 여러 PDE 벤치마크에서도 일관되게 나타났습니다.



### CRUMB: Efficient Prior Fitted Network Inference via Distributionally Matched Context Batching (https://arxiv.org/abs/2606.11473)
Comments:
          26 pages, 13 figures

- **What's New**: CRUMB (Clustered Retrieval Using Minimised-MMD Batching)은 Prior-fitted networks (PFNs)의 비효율적인 추론 문제를 해결하기 위해 제안된 새로운 방법입니다. 이 방법은 테스트 쿼리를 클러스터로 묶고, 각 클러스터에 대해 MMD(최대 평균 차이)를 최소화하여 분포적으로 일치하는 작은 훈련 서브셋을 선택합니다. CRUMB는 아키텍처에 구애받지 않으며 재학습이 필요 없습니다.

- **Technical Details**: CRUMB는 세 가지 단계의 추론 래퍼로 구성됩니다: (i) 테스트 쿼리 클러스터링, (ii) MMD를 최소화하여 클러스터별 훈련 서브셋 선택, (iii) 각 축소된 컨텍스트 배치에 대한 PFN 추론 실행입니다. 이 접근 방식은 테스트 쿼리 포인트에 대한 훈련 컨텍스트의 정렬을 이루면서도 효율적인 배치 추론을 가능하게 합니다.

- **Performance Highlights**: 51개의 데이터셋으로 구성된 TabArena 벤치마크에서 CRUMB는 세 가지 PFN 아키텍처에서 평가되었으며, 각 쿼리별 kkNN 성능과 유사한 성과를 보이면서도 고정된 수의 포워드 패스를 요구하는 점이 강조됩니다. CRUMB는 동일한 컨텍스트 예산에서 Mixture of In-context Prompters (MICP) 기법과 균일 샘플링을 크게 초월한 성능을 나타내며, 공변량 시프트가 심해질수록 MICP에 대한 CRUMB의 이점이 증가하는 경향을 보입니다.



### LSTM-Based Detection of Structural Breaks in Property Insurance Loss Reserving: A Climate-Informed Approach (https://arxiv.org/abs/2606.11463)
Comments:
          15 pages, 0 figures, whitepaper YC

- **What's New**: 이 연구 논문에서는 LSTM(장기 단기 메모리) 신경망이 기존의 Chain Ladder, Bornhuetter Ferguson, Cape Cod 방법보다 기후 변화에 따른 손실 데이터의 구조적 변화에 신속하게 적응할 수 있는지를 평가하는 연구 프로그램을 제안합니다. 기후 변화에 의해 빈번해진 재해와 손실 데이터의 불안정성이 손해준비금의 정확성에 미치는 영향이 크기 때문에, 이러한 새로운 접근 방식의 중요성이 강조됩니다.

- **Technical Details**: LSTM 네트워크는 단기 및 장기 메모리를 유지할 수 있는 구조로 설계되어 있으며, 이는 과거의 손실 개발 패턴이 미래 예측에 더 이상 유효하지 않을 때 효과적입니다. 이 모델은 입력 게이트와 잊기 게이트를 사용하여 중요한 정보를 선택적으로 유지하고 오래된 정보를 버리며, 양방향 레이어를 통해 손실 데이터의 맥락을 이해할 수 있습니다. 이러한 메커니즘 덕분에 LSTM은 재해 발생 패턴 변화에 대한 신속한 반응이 가능합니다.

- **Performance Highlights**: 본 연구는 플로리다와 루이지애나의 15년 이상의 데이터를 사용하여 기후 변화로 인해 재해 위험이 높은 연도의 손해준비금 정확성이 LSTM의 적용으로 15-20% 개선될 것이라고 예상합니다. 제한된 재해 데이터에도 불구하고 LSTM이 전통적인 방법보다 빠른 속도로 진짜 손실 개발에 수렴할 수 있음을 수학적으로 증명하였습니다. 이러한 결과는 전통적인 손해준비 기법의 한계를 극복하고, 더욱 효과적인 손해 준비를 가능하게 합니다.



### The Power of Test-Time Training for Approximate Sampling (https://arxiv.org/abs/2606.11437)
- **What's New**: 이번 논문에서는 복잡한 확률 분포에서 효율적으로 샘플링하는 문제를 다룹니다. 특히, test-time training (TTT)이라는 새로운 접근법을 제안하며, 이는 주어진 확률 측정 $d^$에서 샘플을 생성하는 과정에서 모델의 가중치를 업데이트합니다. 이 연구의 주요 초점은 TTT의 기초 이론을 발전시키고 이를 확률적 샘플링 문제에 formalize하는 것입니다.

- **Technical Details**: TTT는 특정 문제에 대한 피드백을 통해 모델의 가중치를 조정하는 방식입니다. 각 샘플링 절차는 언어 모델과 관련된 특별한 구조를 가지고 있으며, 이 연구에서는 주어진 오라클 $d$를 기반으로 정확한 밀도 추정치를 제공하는 방법을 제시합니다. 또한, 샘플링의 쿼리 복잡도에서의 준수성 한계를 수학적으로 검증하여 이를 통해 TTT의 이론적 기반을 형성합니다.

- **Performance Highlights**: TTT 접근법은 기존 샘플링 방법보다 월등한 성능을 기록함을 보여줍니다. 예를 들어, AlphaProof와 같은 모델이 TTT를 통해 IMO에서 첫 번째 실버 레벨 성과를 달성했습니다. 언어 모델을 통한 문제 해결이 점점 더 복잡해지는 가운데, TTT는 이러한 성공을 이끌 수 있는 핵심 기술로 부상하고 있습니다.



### Towards a Bridge Layer Between Bibliographic and Formalized Mathematical Knowledg (https://arxiv.org/abs/2606.11430)
- **What's New**: 이 논문에서는 MathSciNet과 zbMATH Open과 같은 서지 데이터베이스와 Lean mathlib와 같은 형식 증명 라이브러리 간의 연결 부족 문제를 해결하기 위해 관계형 브리지 데이터베이스를 제안합니다. 이 데이터베이스는 출판 메타데이터와 형식화 아티팩트를 정렬하여 수학 문헌과 기계 검증 가능한 증명 간의 상호 운용 계층을 제공합니다. 또한, 출판물이 공식 시스템에서 얼마나 커버되는지를 측정하는 논문 수준의 형식화 점수 (formalization score)를 도입하였습니다.

- **Technical Details**: 논문에서 제안하는 관계형 브리지 데이터베이스는 MathSciNet과 zbMATH Open의 기록을 Lean과 같은 형식 아티팩트와 연결하는 상호 운용 가능 계층을 구축합니다. 이 시스템은 수학적 성명서(Paper)의 구조화된 모음을 다루며, 각 성명서는 동일하게 취급되어 formalized 여부에 따라 점수가 부여됩니다. 점수 산정 단계에서는 정의된 조건을 만족하는 경우에만 가산되며, 순수 명제(pure statements)와 증명된 명제(proved statements)에 따른다.

- **Performance Highlights**: 사례 연구에서는 Google Gemini 모델을 사용하여 여러 수학 논문에 대한 형식화 점수를 독립적으로 계산할 수 있는지 평가하였습니다. 이 과정에서 모델은 미세 조정 없이 고정된 세 부분 입력을 사용하여 문서 간 형식화 정렬 작업을 수행했습니다. 이를 통해 논문의 형식화 점수가 어떻게 계산될 수 있는지에 대한 실험적 결과를 도출하였습니다.



### JailbreakOPT: Tool-Assisted Iterative Jailbreak Prompt Optimization (https://arxiv.org/abs/2606.11425)
- **What's New**: JailbreakOPT라는 새로운 툴 지원 프레임워크가 제안되었습니다. 이 시스템은 반복적인 단일 턴(jailbreak prompt optimization)의 성능을 향상시키는 데 중점을 두고 있습니다. 기존의 방법들과 비교해 JailbreakOPT는 공격 프롬프트를 더 강력하게 개선할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: JailbreakOPT는 다양한 원자적(jailbreak) 프롬프트를 공격 툴 라이브러리에 조직하고, 이를 통합된 에피소드 내부 최적화 추상화(intra-episode optimization)로 조합합니다. 또한 JailbreakOPT는 과거의 결과를 기반으로 탐색과 활용을 유도하기 위해 문맥적(contextual) 밴딧 문제로 툴 선택을 프레이밍(framing)합니다. 이 과정에서 문맥적 톰슨 샘플링(contextual Thompson sampling)을 적용합니다.

- **Performance Highlights**: 다수의 LLM(target LLMs)과 공격 목표에서 실험을 수행한 결과, JailbreakOPT는 공격 성공률(ASR)을 개선하고 성공하기까지 필요한 공격 횟수(No.A)를 줄였습니다. 기존의 원자적 단일 턴 공격 및 반복 최적화 기반의 방법들과 비교하여 성능이 현저하게 향상되었습니다.



### Signed Compression Progress on a Sealed Audit is Goodhart-Resistan (https://arxiv.org/abs/2606.11417)
Comments:
          16 pages, 7 figures. Lean 4 (Mathlib) mechanized core and ARC-TGI experiment code: this https URL

- **What's New**: 이 논문에서는 내재적 동기(intrinsic motivation)가 경험을 압축(compress)하거나 예측하는 모델의 개선을 통해 보상하는 메커니즘을 수학적으로 정밀하게 정의하고 입증했다. 고전적인 '좋은 하트 법칙(Goodhart's Law)'을 바탕으로, 에이전트가 학습하는 동안 실제 감사(audit) 성능이 저하되지 않도록 보상 시스템을 설계한다. 이 메커니즘은 보상이 에이전트의 모델 개선에 실제적으로 연관되어 있음을 보장하도록 설계되었다.

- **Technical Details**: 저자들은 고정된 감사를 기반으로 한 통계적 보상 측정 프레임을 정의하였다. 세 가지 주요 이론을 제시하며, 첫 번째로는 예산화된 좋은 하트 저항(budgeted Goodhart resistance)을 정의한다. 두 번째는 Lean 4를 사용하여 구조적 핵심을 기계화(mechanization)한 점이며, 세 번째는 보상 신호와 스케줄러를 분리하여 감사를 통해 제공되는 신뢰할 수 있는 보상을 명확히 한다.

- **Performance Highlights**: 실험 결과, 감사 편차(audit deviation)는 n^{-0.527}으로 축소될 수 있음을 보여주었고, 서명된 압축 진행(signed compression progress)은 클리핑(clipping), 스트림 유출(stream leakage) 및 잡음 TV(curiosity) 공격에서 저항력을 유지한다. 또한 재사용 가능한 감사(audit) 메커니즘은 블랙박스 스칼라 피드백(black-box scalar feedback) 공격에 취약하며, 일반적인 방어 메커니즘은 2 Delta_n 임계값 아래로 공격을 유지할 수 있다.



### MPC-Patch-Bench: Security-Aware LLM Code Patch for Multi-Party Computation (https://arxiv.org/abs/2606.11416)
Comments:
          preprint

- **What's New**: 이번 논문에서는 Secure Multi-Party Computation (MPC) 소프트웨어의 코드 수리를 위한 새로운 벤치마크인 MPC-Patch-Bench를 소개합니다. 기존의 일반 목적 벤치마크가 세 가지 구조적인 문제로 인해 MPC 코드 수리 평가에 적합하지 않음을 지적하고, MPC 의식 데이터 큐레이션과 보안 요구 사항에 맞춘 검증자를 도입하여 이러한 문제를 해결했습니다. 이는 LLM(대형 언어 모델)이 실제 MPC 코드 수리를 수행할 수 있도록 돕기 위한 중요한 발전을 나타냅니다.

- **Technical Details**: MPC-Patch-Bench는 두 가지 프레임워크로 구성됩니다. 첫째, 데이터 큐레이션 프레임워크는 MPC 고유 semantics에 해당하는 요소만을 필터링하여 PR(pull requests)을 유지하며, 인간-AI 협업 엔진을 통해 문제가 발생한 테스트를 합성합니다. 둘째, MPC Verifier는 동적 차별 테스트(dynamic differential testing)와 정적 분석(static analysis)을 결합하여 보안성을 유지하며, MPC 프로그램의 특정한 요구 사항을 충족하도록 검증을 수행합니다.

- **Performance Highlights**: 실험 결과에 따르면, 가장 강력한 LLM조차도 MPC-Patch-Bench의 과제 중 22.9%만을 기능적으로 해결할 수 있었고, MPC Verifier의 추가 검증 후에는 이 비율이 17.1%로 감소했습니다. 최대 40%의 기능적으로 패스한 패치가 cryptographic 혹은 numerical-fidelity 위반으로 거부되었습니다. 이러한 결과는 LLM이 보안이 중요한 소프트웨어에서 신뢰성을 과대 평가하고 있음을 시사하며, 더 나은 안전 보장을 위한 평가 기준의 필요성을 강조합니다.



### Risk Under Pressure: Compute-Aware Evaluation of Adversarial Robustness in Language Models (https://arxiv.org/abs/2606.11409)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 적대적 강인성(e.g., adversarial robustness)을 평가하는 새로운 프레임워크를 제안합니다. 기존의 평가 방식이 모든 공격을 동일한 비용으로 간주하는 한계가 있으므로, 우리는 각 공격 전략의 계산 비용(computational cost)을 고려한 방법을 개발했습니다. 이를 통해 공격의 리스크(risk)를 정량화하고, 공격 성공을 위한 평균적 계산 압력(pressure)을 도출하는 지표를 소개합니다.

- **Technical Details**: 제안된 프레임워크는 누적 부동 소수점 연산(FLOPs)으로 측정되는 계산 압력을 바탕으로 적대적 공격을 모델링합니다. 각 공격 단계에서 제안된 프롬프트(p)와 그에 대한 모델의 응답(y)을 평가하는 안전 심사자(safety judge)가 포함되며, 계산 비용은 FLOPs로 변환됩니다. 이를 통해 공격 비용을 평가하고, 다양한 공격 전략을 비교할 수 있는 통합된 분석을 제공합니다.

- **Performance Highlights**: 연구 결과, 세 가지 공격 전략(gradient-based, iterative refinement, template-based)을 사용하여 10개 모델을 평가한 결과 다양한 패턴이 발견되었습니다. 특히, 모델 크기를 확장하는 것이 gradient 기반 공격의 효과는 줄이지만, template 기반 공격에 미치는 영향은 제한적이라는 점이 강조되었습니다. 이 연구는 실제 공격자들이 계산 비용을 고려하여 공격 방식을 선택한다는 점을 명확히 하며, 안전성을 높이기 위한 새로운 평가 방법의 필요성을 제기합니다.



### Steering Where to Listen: Instruction-Based Activation Steering Redirects Temporal Attention in Large Audio-Language Models (https://arxiv.org/abs/2606.11400)
- **What's New**: 이 논문은 대규모 오디오-언어 모델(LALMs)이 오디오 이해에서 뛰어나지만, 신호의 특정 부분에 집중하는 방법을 명확히 드러내지 않는 문제를 다룹니다. 새로운 접근법인 instruction-based vector steering을 소개하며, 이는 고정된 오디오를 유지하면서 서로 다른 프롬프트에서 활성화를 대비하여 조정 벡터를 생성합니다. 이러한 방법은 표준 프롬프트나 오디오 기반 조정과 달리, 시간적으로 오디오 토큰에게 할당된 주의를 효과적으로 재배치하는 것을 보여줍니다.

- **Technical Details**: 이 연구에서 시스템적으로 LALM의 주의를 조사하여, 조정으로 인한 주의 집중 변화가 오디오 신호의 특정 부분에서 일어난다는 것을 발견하였습니다. 특히, 주의 집중을 최대화하는 시점의 위치를 읽어내는 방법을 통해 훈련 없이도 쿼리된 소리 이벤트의 위치를 회복할 수 있음을 입증하였습니다. 이 과정에서 오디오 관련 영역에 주의를 집중시키며, 모델이 인코딩한 잠재적 시간 구조에 대한 훈련 없는 프로브가 가능하다고 강조하였습니다.

- **Performance Highlights**: 세 가지 이벤트가 있는 통제된 환경에서, 주의 집중 변화 지점을 읽어낸 결과 Qwen2-Audio와 Audio Flamingo 3에서 각각 60.87%와 68.72%의 실제 간격과의 겹침(overlap)을 기록했습니다. 이는 직접 프롬프트(31.84%, 46.75%)나 무작위 기준치(27.74%)와 비교할 때 현저히 높은 성과입니다. 이러한 결과는 LALMs에서 instruction-based steering의 메커니즘 특성을Characterizing하고, 훈련 없이도 효과적으로 모델의 잠재 구조를 탐지할 수 있는 방법을 제시합니다.



### TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs (https://arxiv.org/abs/2606.11357)
Comments:
          13 pages excluding reference, 11 figures

- **What's New**: 이 논문에서는 AMD XDNA2 NPUs를 위한 TileFuse라는 혼합 정밀도(mixed-precision) 커널 라이브러리를 소개합니다. 이 라이브러리는 양자화된 대형 언어 모델(LLM) 추론에서 transformer 선형 층을 목표로 하며, 기존의 양자화 방식을 변경하지 않고도 AWQ 스타일의 저비트 형식을 사용할 수 있도록 설계되었습니다. TileFuse는 언패킹(unpacking), 양자화 해제(dequantization), GEMM/GEMV 실행을 단일 커널 흐름으로 통합하여 효율성을 제공합니다.

- **Technical Details**: TileFuse의 주요 기술적 기여는 AWQ 스타일의 W4A16 및 W8A16 가중치를 직접 소비하는 융합 커널을 설계하고 구현하는 것입니다. 이 과정에서 저비트 가중치 변환이 GEMM/GEMV 커널 내로 통합되어, 계산 코어에 도달하기 전까지 가중치는 압축된 INT4/INT8 형식으로 유지됩니다. 또한, 타일 구조를 활용하여 32K까지의 GEMM 차원이 지원되는 새로운 interleaved pre-tiling 레이아웃을 도입하고, GEMV 데이터 흐름에 대한 공동 설계를 통해 성능을 최적화하였습니다.

- **Performance Highlights**: TileFuse를 통해 GEMM에서 최대 121.6%, GEMV에서 281%의 성능 향상을 달성하였으며, 유사한 환경의 iGPU와 비교하여 2배 이상의 성능과 에너지 효율성 향상을 보여주었습니다. 특히, Ryzen AI 노트북에서의 LLM 실험에서는 최대 2.0배 낮은 프리필링 지연(latency)과 64.6% 이상의 에너지 절감을 기록했습니다. 이러한 결과는 XDNA2가 AWQ 스타일의 엣지 LLM 추론을 위한 실용적인 타겟임을 입증하고, 현장 배치에서 NPU의 유용성을 크게 향상시킬 수 있음을 시사합니다.



### Quantized Stochastic Primal-Dual Methods for Distributed Optimization under Relaxed Global Geometry (https://arxiv.org/abs/2606.11339)
Comments:
          Accepted to UAI

- **What's New**: 본 논문에서는 랜덤(비편향) 양자화에 의해 모델링된 확률적 그래디언트를 가진 분산 최적화 문제를 다루고 있으며, 이를 위한 양자화된 확률적 프라이멀-듀얼 방법인 q-PDGD를 제안하고 분석합니다. 이 방법은 제한된 전역 기하학 하에서 동작하며, 그래디언트 노이즈, 양자화 왜곡, 네트워크 연결성에 따라 결정되는 이웃 영역으로의 선형 수렴을 보입니다. 상수 스텝 크기와 점진적으로 감소하는 스텝 크기에 대해 다양한 수렴 보장을 제공합니다.

- **Technical Details**: q-PDGD는 제한된 시컨트 불평등(restricted secant inequality, RSI)과 폴리악-로자예비치(Polyak-Lojasiewicz, PL) 불평등 하에서 다양한 스텝 크기의 경우에 대한 수렴을 보장합니다. 특히, 상수 스텝 크기의 경우, 노이즈와 양자화, 네트워크에 따라 결정되는 이웃으로의 선형 수렴을 달성하며, 점진적으로 감소하는 스텝 크기 하에서는 O(1/k) 수렴 속도를 보입니다. 또한, 실험을 통해 양자화 수준, 스텝 크기 선택, 그래프 구조 간의 예측된 트레이드오프를 실증적으로 보여줍니다.

- **Performance Highlights**: 이 연구는 기존의 중앙 집중식 확률적 최적화 속도와 최악의 복잡성을 일치시키며, 여러 시뮬레이션을 통해 저자들이 예측한 양자화 수준과 스텝 크기 선택의 상관 관계를 확인하고 있습니다. 다수의 성능 실험을 통해, 제안된 방법이 기존의 서버 기반 아키텍처의 제한을 뛰어넘어 분산 환경에서도 잘 작동할 수 있음을 보였습니다. 이로 인해 q-PDGD는 실제 대규모 네트워크의 효율적인 최적화에 기여할 수 있습니다.



### Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models (https://arxiv.org/abs/2606.11324)
Comments:
          Embodied R1.5 technical report. Project page: this https URL

- **What's New**: Embodied-R1.5는 복합적 의식(reasoning) 능력을 통합한 통합형 Embodied Foundation Model (EFM)으로, 15B 이상의 토큰을 포함한 대규모 데이터 시스템을 통해 일반적인 물리적 지능을 목표로 설계되었습니다. 이 모델은 Planner-Grounder-Corrector (PGC) 폐쇄형 프레임워크를 통해 장기적인 작업을 자율적으로 실행하고 수정할 수 있으며, 이를 통해 동작 및 오류 수정의 자동화를 가능하게 합니다. Embodied-R1.5는 8B의 파라미터만으로 24개의 embodied VLM 벤치마크 중 16개에서 SOTA(State Of The Art)를 달성했습니다.

- **Technical Details**: EFM 모델은 한 가지 아키텍처 내에서 지각(perception), 추론(reasoning), 실행(execution)의 모든 기능을 통합하여, 점진적인 추론 체인을 형성합니다. 모델 아키텍처의 세 가지 주된 차원은 공간 인지(cognition), 작업 계획(planning), 수정(correction)으로 구성되며, 이는 약속된 정보 흐름을 통해 외부 통신 없이도 원활한 작업을 지원합니다. 이를 위해 세 가지 자동화된 데이터 생성 파이프라인을 사용하여 15B 이상의 토큰을 포함한 대규모 데이터 코퍼스를 구축했습니다.

- **Performance Highlights**: Embodied-R1.5는 SOTA 벤치마크에서 평균 70.4%의 정확도를 달성하며, Gemini-Robotics-ER-1.5 및 GPT-5.4를 각각 17.0% 및 21.7% 초과합니다. 특정 테스트 벤치마크에서 92.4%의 성능을 보여주며, 이는 π0.5 및 ManipLLM과 같은 강력한 VLA 모델을 능가합니다. 또한, 제로샷(real-robot) 실험을 통해 다양한 실제 작업 수행에서 강력한 일반화를 보여주고 있습니다.



### FreeBridge: Variational Schrödinger Bridges for Cellular Transition Dynamics (https://arxiv.org/abs/2606.11286)
Comments:
          Accepted to MICCAI 2026 (early accept). Project page: this https URL

- **What's New**: 이 논문에서는 화학적 고정으로 인해 개별 세포의 연속적인 경로를 관측할 수 없던 문제를 새로운 접근 방식인 	extbf{FreeBridge}를 통해 해결하고자 한다. FreeBridge는 단일 세포 전이 모델링을 위한 슈뢰딩거 브릿지(Schrödinger Bridge) 형식을 사용하여, 엔드포인트(Endpoint)만으로 제한된 감독 하에서도 세포 다이나믹스를 모델링한다. 이 방법은 수치적 지원 제약을 통해 고정된 세포 기하학 내에서 확률적 이동을 학습하는 방식을 기반으로 한다.

- **Technical Details**: FreeBridge는 세포 엔진(Cell Engine)이라는 구조로 단일 세포의 상태를 명확히 정의하고, 이 상태 내에서 확률적 이동을 모델링한다. 상태 유닛(State Unit)은 유효한 단일 세포 표현의 허용된 기하학을 정의하고, 전이 규칙(Transition Rule)은 해당 고정 기하학 내에서의 엔트로피 정규화된 확률적 이동을 모델링한다. 이 과정에서 저자들은 데이터 기반의 상태 비용을 추가하여 중간 경로를 현실의 세포 형태와 연결하도록 강하게 하고, FreeBridge가 독립적인 세포 모델을 어떻게 구성하는지를 보여준다.

- **Performance Highlights**: FreeBridge는 BBBC021, RxRx1 및 JUMP 데이터셋에서 기존 상태 정렬 방법에 비해 경쟁력 있는 또는 향상된 생성 성능을 보여준다. 특히 BBBC021에서는 중간 지원 위반을 줄이는 성과를 보였다. 이러한 평가들은 생물학적으로 해석 가능한 교란 다이나믹스에 대한 기하학적 기반의 중요성을 강조한다.



### RoVE: Rotary Value Embeddings Attention for Relative Position-dependent Value Pathways (https://arxiv.org/abs/2606.11275)
- **What's New**: 이 논문에서 제안하는 RoVE는 Rotary Position Embeddings (RoPE) 기법을 개선한 것이다. RoPE는 주의(attention) 점수를 위치 상대적으로 만드는데, 값(value) 경로는 위치에 둔감하다. RoVE는 값을 키와 동시에 회전시켜 값을 위치 감지 하도록 변경함으로써, RoPE의 단점을 보완한다. 이러한 관점에서 RoVE는 컴퓨터 비전, 로보틱스 및 최신 LLM 아키텍처 전반에 걸쳐 동일한 작업에 대한 여러 독립적인 형식을 통합한다.

- **Technical Details**: RoVE는 값 경로의 상대 위치 감도를 제고해준다. 기존 RoPE는 주의 점수를 상대적 위치에 따라 조정하지만, 값을 동일하게 처리해 도출된 결과는 상대적 거리와 무관하게 같은 메시지를 전달했다. RoVE에서는 각 값을 집계 전에 쿼리의 참조 프레임으로 회전시키며, 이를 통해 값 경로가 단일 공유 맵이 아닌 상대적인 위치 정보를 포함하는 매핑으로 전환된다. 이는 기존의 RoPE를 attentive convolution으로 재구성하여, 구조적 특성을 명확히 한다.

- **Performance Highlights**: RoVE는 124M 및 354M 매개변수를 가진 GPT-2 모델에서 실증적인 성과를 보였다. RoVE는 몇 Shot의 인 상황 학습(in-context learning)과 긴 맥락 검색(long-context retrieval)에서 RoPE보다 일관된 성능 향상을 보였으며, 특히 긴 거리 집계가 필요한 작업에서 뚜렷한 개선을 나타냈다. 이러한 성능 향상은 RoVE가 기존 RoPE의 한계를 극복하고 더 강력한 모델링 기능을 제공함을 입증한다.



### Federated continual learning: A comprehensive survey on lifelong and privacy-preserving learning over distributed and non-stationary data (https://arxiv.org/abs/2606.11272)
Comments:
          77 pages, 8 figures

- **What's New**: 이 논문에서는 연속적인 데이터 분포에서의 학습을 가능하게 하는 Federated Continual Learning (FCL)의 개념을 소개합니다. FCL은 기존의 Federated Learning (FL)과 Continual Learning (CL)를 결합하여 데이터가 분산되고 비정상적일 때 적응적이고 개인 정보를 보호하면서 학습할 수 있는 방법을 제시합니다. 점점 증가하는 데이터의 비정상적인 특성을 감안할 때, FCL은 더욱 필요해지는 학습 패러다임이 되어가고 있습니다.

- **Technical Details**: FCL은 데이터의 비정상성을 고려하며, 클라이언트의 데이터가 분산되어 있고 시간이 지남에 따라 변화하는 상황을 처리합니다. 이 접근 방식은 모델이 이전에 학습한 지식을 잃지 않으면서도 새로운 정보를 효과적으로 학습할 수 있도록 돕습니다. 논문에서는 FCL 방법론을 다차원 분류체계로 체계적으로 정리하여 다양한 응용 문제와 데이터 양식을 고려합니다.

- **Performance Highlights**: FCL에 대한 기존 연구는 산재해 있으며, 다양한 평가 기준과 실험적 접근 방식이 있어 종합적인 성과 평가가 어려웠습니다. 논문은 FCL의 주요 도전 과제를 다루고, 성능, 기억 유지, 통신 효율성 및 메모리 요구 사항을 포함하여 장기 성능에 대한 표준 평가를 강조합니다. FCL의 효과적인 실현을 위해서는 프라이버시를 유지하면서도 역동적인 데이터 변화에 적응할 수 있는 기념 메커니즘과 같은 해결해야 할 주요 과제들이 남아있습니다.



### When Poison Fails After Retrieval: Revisiting Corpus Poisoning under Chunking and Reranking Pipelines (https://arxiv.org/abs/2606.11265)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 코퍼스 중독(corpus poisoning) 공격을 재검토합니다. 기존 연구들은 단순화된 검색 설정에서 중독 공격을 평가하는 데 집중했으나, 본 연구는 실제 다단계 검색 파이프라인을 고려하여 이러한 공격의 한계를 분석합니다. 특히, 문서 수준의 적대적 신호가 조각화(fragmentation)되면서 효과가 감소하는 문제를 지적하고, 새로운 공격 프레임워크인 Chunk-aware and Rerank-Consistent Poisoning (CRCP)를 제안합니다.

- **Technical Details**: CRCP는 검색의 관련성(retrieval relevance), 재순위의 일관성(reranker consistency), 그리고 청크 경계의 강건성(chunk-boundary robustness)을 동시에 최적화하는 프레임워크입니다. 기존의 문서 수준 유사성(global document-level similarity)을 최적화하는 대신, CRCP는 청크화(chunking) 변환을 모델링하여 독립적인 적대적 구절을 생성합니다. 실험을 통해 CRCP가 다양한 검색 파이프라인을 통해 강력한 공격 성공률과 내구성을 제공함을 보여줍니다.

- **Performance Highlights**: CRCP는 기존의 중독 공격 방법들보다 실제 검색 환경에서 더욱 일관된 성능을 보여줍니다. 실험 결과에 따르면, 기존의 방법은 청크 크기(chunk size)와 재순위 전략(reranking strategies)에 매우 민감하여 성능이 급격히 저하되지만, CRCP는 다양한 청크 및 재순위 구성에서 또한 강한 방어력을 발휘합니다. 본 연구는 현재 RAG 보안 평가에서 나타나는 중요한 현실적 격차를 강조하며, 중독의 문제를 단순한 검색 문제로 보기보다 다단계 검색 일관성 문제로 접근해야 함을 제안합니다.



### OmniBioTwin: A System-of-Twinned-Systems Framework for Health Digital Twins (https://arxiv.org/abs/2606.11264)
- **What's New**: 이 논문에서는 OmniBioTwin이라는 새로운 시스템-온-트윈드-시스템(System-of-Twinned-Systems, SoTS) 프레임워크를 제안합니다. OmniBioTwin은 통합적인 환자 맞춤형 건강 디지털 트윈(Health Digital Twins, HDT)을 구축할 수 있도록 설계된 7개의 계층으로 구성된 모듈형 네트워크 아키텍처입니다. 이 프레임워크는 다양한 생물학적 스케일을 결합하고, 역동적으로 진화하는 상호작용을 포착할 수 있는 가능성을 보여줍니다.

- **Technical Details**: OmniBioTwin은 데이터 통합, 자율 트윈 모델링, 크로스 스케일 결합, 시간 동기화, 인간-주도 의사결정 지원을 포함하는 7개의 협조적인 계층으로 조직됩니다. 이 구조는 생물학적 의미를 유지하면서도 모듈성을 제공하며, 독립적인 검증과 개별 구성 요소의 확장을 용이하게 합니다. 각 트윈은 독립적인 상태, 관찰, 동적 및 불확실성 표현을 유지하며, 특정 생물학적 구성요소나 프로세스를 모델링합니다.

- **Performance Highlights**: 이 프레임워크는 알츠하이머병(Alzheimer's Disease)에서 GLP-1 신호 경로의 다중 스케일 트윈을 구현하여 그 효과를 입증하였습니다. OmniBioTwin의 아키텍처는 다양한 데이터 소스를 통합하고 동적인 결정 프로세스를 수용하면서 생물학적 해석 가능성과 모듈식 확장성을 보장합니다. 이 시스템은 환자 맞춤형의 닫힌 루프 디지털 트윈으로 지속적으로 환자에게서 정보들을 수집하고 임상 환경에 예측 결과를 반환합니다.



### PermDoRA -- Understanding Adapter Interference in Language Models: Limits of Parameter-Space Geometry (https://arxiv.org/abs/2606.11262)
Comments:
          18 Pages, COLM 2026

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 접근 제어를 위한 모듈화된 메커니즘을 제안합니다. DoRA-RBAC라는 새로운 계층형 어댑터 구성 프레임워크를 사용하여 여러 도메인 간 간섭을 줄이는 방법을 탐구하고 있습니다. 기존 접근 방식은 기하학적으로 인지된 병합 전략을 사용하여 Euclidean 병합과 비교하는 방법론을 설정합니다. 여러 QA 벤치마크에서의 평가 결과는 기하학적 병합이 항상 우수한 성능을 제공하지 않는다는 것을 입증합니다.

- **Technical Details**: 이 연구에서는 어댑터 상호작용의 보다 정교한 분석을 위해 업데이트를 방향과 크기로 분해하는 DoRA(low-rank adaptation) 방법론을 채택합니다. 각 도메인은 관리 어댑터에 의해 업데이트가 저장되며 사용자가 액세스할 수 있는 도메인에 따라 선택적으로 활성화됩니다. 두 가지 병합 전략인 Euclidean병합과 Riemannian 병합을 비교하며, 후자는 방향적 정규화 평균을 사용하여 Fréchet 평균을 근사합니다.

- **Performance Highlights**: 테스트 결과, 기하학적 병합 방식은 다중 도메인 설정에서 Euclidean 병합 기법에 비해 일관된 성능 향상을 보여주지 않았습니다. 특정 도메인 어댑터를 사용하는 경우 성능이 개선되지만, 모듈화된 합성이 강한 개인정보 보호 보장을 제공하지 않음을 시사합니다. 결과적으로 DoRA-RBAC는 엄격한 격리를 위한 메커니즘이 아닌 모듈화된 구성 프레임워크로 간주되어야 합니다.



### RAIL: Rethinking Auditory Intelligence in Large Audio-Language Models with a CHC-Grounded Benchmark (https://arxiv.org/abs/2606.11260)
- **What's New**: 이 논문은 RAIL이라는 새로운 human-centric evaluation paradigm을 제안합니다. 이는 Cattell-Horn-Carroll (CHC) 인지 프레임워크에 기반하여, 인간의 청각 인지를 다섯 가지 핵심 능력으로 정형화하고 이를 평가하는 구조화된 작업을 개발합니다. RAIL은 모델이 청각 정보를 처리하고 통합하는 방식을 체계적으로 포착하는 데 초점을 맞추고 있습니다.

- **Technical Details**: RAIL에서는 총 5,306개의 다양한 작업 형식을 가진 음성 샘플 데이터를 수집하고, 이 데이터를 통해 26개의 최신 LALM 모델을 평가하였습니다. 각 모델의 성능은 청각 인지 능력에 따라 다르게 나타났으며, 특히 모델 간의 성능 차이가 뚜렷했습니다. 이러한 결과는 기존의 작업 중심 평가 방식이 모델의 인지 능력을 충분히 반영하지 못하고 있음을 보여줍니다.

- **Performance Highlights**: 결과적으로 LALMs는 언어 기반의 사전 훈련에서 강력한 지식을 나타내지만, 청각 인지(예: 청각 지각, 추론 및 기억)에서는 상대적으로 약한 성능을 보였습니다. 특히, 장애물로 여겨지는 청각 처리와 처리 효율성에서 인간과 비교했을 때 큰 격차가 존재함을 확인했습니다. 따라서 이 연구는 청각 기초 및 효율성을 개선하기 위한 방향성을 제시합니다.



### Physics-informed generative AI for semiconductor manufacturing: Enforcing hard physical constraints in generative models by construction (https://arxiv.org/abs/2606.11247)
- **What's New**: 이 논문에서는 반도체 제조가 제약 조건이 있는 물리적 도메인(physical domains)에서 생성 AI의 활용에 관한 새로운 시각을 제시합니다. 생성 모델이 고유의 물리적 제약을 준수해야 하며, 후처리(Post-hoc filtering)가 아닌 설계 단계에서부터 물리 기반 정보(physics-informed)를 반영해야 한다고 주장하고 있습니다. 이를 통해 물리적 유효성(physical validity)에 대한 새로운 기준을 세우고 있습니다.

- **Technical Details**: 논문에서 검토된 기술적 요소로는 물리 기반 확산(physics-informed diffusion), PDE 제약 변분 모델(PDE-constrained variational models), 신경 연산자 사전(neural-operator priors), 보존 법칙을 준수하는 생성 네트워크(conservation-law-respecting generative networks) 등이 있습니다. 이러한 아키텍처는 차별화 가능한 리소그래피(differentiable lithography), TCAD, 프로세스 시뮬레이션(process simulation), 자율 실험(autonomous experimentation)과 연결되어 있습니다. 논문은 생성 모델과 물리 기반 시뮬레이터 간의 네 가지 통합 패턴(integration patterns)을 제시합니다.

- **Performance Highlights**: 저자들은 물리적 유효성이 성공의 기준이 되는 환경에서, 설계 단계에서부터 이를 강제하는 아키텍처가 후처리 방식으로 필터링하는 아키텍처보다 우수할 것이라고 주장합니다. 특히 반도체 제조지는 이러한 차별점이 가장 두드러지는 장소로 제시됩니다. 또한 물리 기준 평가(physics-fidelity benchmarks), 차별화 가능한 시뮬레이터 인프라(differentiable simulator infrastructure), 물리적 설계 및 제조를 위한 다중 모달 기초 모델(multimodal foundation models) 중심의 연구 의제를 제안합니다.



### SPEAR: A System for Post-Quantization Error-Adaptive Recovery Enabling Efficient Low-Bit LLM Serving (https://arxiv.org/abs/2606.11244)
- **What's New**: 이번 논문은 SPEAR 시스템을 제안하여 포스트-양자화(post-quantization)에서의 오류 보정을 개선함으로써 저비트 LLM 서비스의 품질을 높이는 방법을 설명합니다. 기존의 고정적 보정 방식들이 다양한 입력에 대한 변동성을 고려하지 않아 발생하는 품질 격차를 해결합니다. SPEAR는 각 토큰의 오류에 적응하는 경량의 오류 보상기를 사용하여 더 어려운 토큰에 보정량을 집중시키는 혁신적인 방식을 도입합니다.

- **Technical Details**: SPEAR 시스템은 입력에 따라 보정을 조절하는 오류 보상기(Error Compensators, ECs)를 도입하여 양자화 오류를 최소화합니다. ECs는 CKA(Centered Kernel Alignment) 기반 진단을 통해 수익성이 높은 모듈에만 선택적으로 배치됩니다. 이를 통해 모델의 품질을 회복하면서도 과도한 래텐시나 메모리 오버헤드를 피합니다.

- **Performance Highlights**: SPEAR는 W4(4-bit)와 FP16 사이에서 56-75%의 혼란도(perplexity) 격차를 회복하며, 1% 이하의 추가 모델 메모리 오버헤드를 요구합니다. 성능 면에서 SPEAR는 표준 4비트 인퍼런스에 비해 수용 가능한 처리량과 래텐시를 유지하며, 기존의 정적 보상 방법들보다 효율성과 효과성의 균형을 검증합니다. 또한, SPEAR는 단일 GPU 및 텐서 병렬 설정에서도 저비트 서비스의 효율성을 유지합니다.



### Artificial Intelligence in Ship Finance: Applications, Opportunities, and a Case Study in AI-Augmented Loan Origination (https://arxiv.org/abs/2606.11238)
Comments:
          9 pages, 1 figure

- **What's New**: 이 논문은 선박 금융(ship finance) 분야에서의 인공지능(AI) 활용 가능성을 탐구합니다. 특히, 대형 언어 모델(LLM) 기반 시스템을 통해 문서 이해(document comprehension), 정보 추출(information extraction), 그리고 워크플로우 자동화(workflow automation)에 주목합니다. LLM을 사용한 모듈형 에이전트 구조를 설계하여 선박 금융의 대출 신청 프로세스를 지원하는 시스템을 제안합니다.

- **Technical Details**: 선박 금융은 데이터 중심(data-intensive)으로 문서가 많은(asset-heavy) 대출(segment)로, 다양한 비구조적(unstructured) 정보의 통합이 필요합니다. 최근의 환경 규제와 ESG(환경, 사회, 지배구조) 보고 요구 사항은 근거 기반의 대출 신청을 더욱 복잡하게 만듭니다. AI는 이러한 정보의 처리 및 분석을 향상시킬 기회를 제공합니다. 논문은 LLM 기반의 정보 추출 모듈, 재무 분석 컴포넌트, 외부 해양 데이터 서비스, 문서 생성 모듈 및 챗봇 인터페이스를 결합하여 표준화된 대출 신청서를 준비하는 방법론을 설명합니다.

- **Performance Highlights**: AI 기반 시스템 활용 시, 선박 금융 전문가들은 복잡한 정보 관리 및 보고 요구 사항에 효과적으로 대처할 수 있습니다. 데이터 품질이 높을수록 대출자들은 보수적인 가정을 줄일 수 있으며, 이는 자본의 조달, 만기 연장, 협약의 공간을 정당화하는 데 유리합니다. AI의 발전은 문서 기반의 금융 프로세스를 더욱 효율적으로 만들어 시장의 경쟁력을 높이는 데 기여할 것으로 보입니다.



### An Ethical eValuation Agent (EeVA): Results of a Proof-of-Concept Test on a Prototype Agentic-like Workflow to Assist Ethical Deliberations (https://arxiv.org/abs/2606.11218)
- **What's New**: 이 논문에서는 EeVA라는 윤리적 성찰을 지원하는 LLM 기반의 에이전틱(agentic) 워크플로를 개발하였다는 점이 새롭습니다. EeVA는 단순한 정답을 제공하는 것이 아니라 비교 윤리적 성찰을 촉진하는데 초점을 맞추고 있습니다. 이를 위해 n8n에서 세 가지 상호 연결된 워크플로(스타터, 워커, 이미터)를 사용하여 프로그램되었습니다.

- **Technical Details**: EeVA는 업로드된 사례를 10개의 윤리적 프레임워크와 비교하여 평가하는 구조를 가지고 있습니다. 평가자는 인공지능 모델의 프로프트(prompts)를 통해 여러 사례를 평가하며, 이 과정에서 구체적인 프레임워크에 대한 구조적 평가 및 통합된 종합 분석을 제공합니다. 본 연구는 도시 이동성, 피어 투 피어 에너지 거래, 사회 서비스 자원 배분의 세 가지 사례를 사용하여 개념 증명(proof-of-concept) 테스트를 진행했습니다.

- **Performance Highlights**: EeVA는 사례 분석에서 프레임워크 간의 공통점 및 차이점을 식별하고, 정렬을 높이기 위한 수정 사항을 제안하며, 지속적인 윤리적 긴장 관계를 강조했습니다. 비전문가에게도 이해하기 쉽게 구성된 결과를 제공하며, 단순한 정답에서 설계 조건과 안전 장치, 그리고 서로 다른 프레임워크 간의 전면적 동의가 불가능한 영역으로 시선을 전환합니다. 이 연구 결과는 윤리 전문가와 비전문가 간의 의사소통 격차를 메울 수 있는 유용한 작업 흐름을 제공한다고 제안합니다.



### Preregistration for Experiments with AI Agents (https://arxiv.org/abs/2606.11217)
Comments:
          Accepted at ICML 2026 as a Spotlight (Top 5%) Position Paper

- **What's New**: 이번 연구는 대규모 언어 모델(LLM) 및 자율 AI 에이전트의 확산으로 인한 "in silico" 행동 실험 방법론의 발전을 다루고 있습니다. AI 에이전트를 활용한 연구는 기존의 인간 주자 연구와 비교하여 효율성 및 통제력을 제공하면서도, 연구자 자유도와 같은 잠재적인 취약점을 포함하고 있습니다. 연구는 이러한 문제를 해결하기 위해 선행 등록(preregistration) 절차의 적용이 필요하다고 주장합니다.

- **Technical Details**: 이 논문은 AI 에이전트에 대한 실험에서 발생할 수 있는 다양한 연구자 자유도를 체계적으로 분류합니다. 모델 선택, 프롬프트 구문 구성, 재실행 정책 등 다차원적인 선택이 존재하며, 이러한 선택의 저렴한 비용이 연구 결과의 신뢰성에 영향을 미칠 수 있습니다. 논문은 AI 에이전트 실험을 위한 맞춤형 선행 등록 템플릿을 제공하고, 이러한 템플릿이 과학적 탐사의 유연성을 유지하면서도 새로운 자유도를 다루어야 한다고 강조합니다.

- **Performance Highlights**: AI 에이전트를 활용한 실험은 실질적으로 전달된 행동 패턴이 인간의 행동을 유사하게 반영할 수 있음을 보여줍니다. 그러나 연구자들이 유도할 수 있는 결과의 편향 문제가 존재하며, 특히 AI 에이전트를 사용하여 반복적으로 실험을 진행할 경우 신뢰성 있는 결과를 얻는 것이 어려워질 수 있습니다. 이 연구는 AI 분야에서의 신뢰성 구축을 위한 선행 등록을 표준화하는 것의 중요성을 강조합니다.



### The Environmental Cost of LLMs in AIED: Reporting and Practices (https://arxiv.org/abs/2606.11215)
- **What's New**: 최근 인공지능 교육(AIED) 커뮤니티에서 대규모 언어 모델(LLM)의 사용이 급증하고 있으며, 이에 따라 LLM 사용으로 인한 컴퓨팅 및 환경 비용에 대한 문제도 부각되고 있습니다. 이 논문에서는 AIED 2025 컨퍼런스에서 발표된 연구들을 분석하여 LLM 사용에 대한 보고의 불일치 상태를 조명하고, 이를 해결하기 위한 표준화된 보고 방안을 제안합니다. 저자들은 AIED 연구 커뮤니티에서 LLM 사용 관련 숨겨진 비용을 투명하게 보고할 수 있는 오픈 소스 방법론을 제공하고자 합니다.

- **Technical Details**: AIED 커뮤니티 내 LLM의 자원 사용을 측정하기 위해 저자들은 문헌_review를 통해 2025 AIED 회의 발표 논문 396편을 분석하였습니다. LLM의 사용 여부와 사용하는 방식, 계산 비용 및 환경 문제에 대한 언급을 조사했습니다. 그 결과, 257편의 논문에서 LLM이 사용되었으나, 컴퓨팅 비용이 명시된 논문은 85편에 불과하고, 지속 가능성에 관한 논의는 57편에서만 발견되었습니다.

- **Performance Highlights**: LLM의 사용은 다양한 방식으로 진행되고 있으며, 대부분의 사례에서 OpenAI 모델(GPT-4/4o) 및 Anthropic Claude가 주로 사용되었습니다. LLM을 사용한 연구들은 주로 실험 대상, 데이터 분석, 콘텐츠 설계, 사용자 상호작용 등 네 가지 역할로 나뉘며, 이러한 역할은 종종 조합되어 나타납니다. 하지만 AIED 커뮤니티의 LLM 사용에 대한 환경적 고려는 여전히 미비하며, 이에 대한 표준화된 측정 및 보고 방안이 필요합니다.



### From Awareness to Action: Understanding and Overcoming the Research-Practice Gap in Algorithmic Fairness for Public Health (https://arxiv.org/abs/2606.11214)
Comments:
          Extended version of an accepted IASEAI'26 paper; includes technical appendices. 22 pages, 2 figures

- **What's New**: 이 논문은 공공 보건 연구에서 기계 학습(ML) 기반 알고리즘의 공정성(algorithmic fairness)이 중요하다는 점을 강조하며, 이를 위한 방법론적, 조직적, 체계적 차원의 통합된 접근법인 Fairness-to-Action (F2A) 프레임워크를 제안합니다. 연구자들이 공정성을 이해하고 실행하는 방식에 관한 차이를 조사하기 위해 전문가 인터뷰와 온라인 설문조사를 포함한 혼합 방법 연구를 진행하였습니다.

- **Technical Details**: 연구는 ML 라이프 사이클에서의 공정성 개념화, 인식, 평가, 설계 및 적용을 포함한 다섯 가지 차원에서 연구하였습니다. 또한, 기존의 이론들인 Knowledge-Practice Gap, Knowledge-to-Action Cycle, Knowing-Doing Gap을 통해 공정성 연구와 실제 적용 간의 격차를 해석했습니다. 이를 통해 우리가 제안하는 F2A 프레임워크는 공정성 지식의 전이가 중단되는 지점을 파악하고 더 효과적인 적용 조건을 제시합니다.

- **Performance Highlights**: 이 연구는 공공 보건 연구에 있어 공정성이 제대로 제도화되지 않았으며, 시스템 수준의 우선순위가 공정성보다는 정확성에 치중하고 있다는 중요한 통찰을 제공합니다. 연구결과에 따르면, ML 기반 공공 보건 연구에서 공정성을 강화하는 데 있어 접근 장애가 외부에서 유도되며, 공정성의 이해와 실천 간의 연계가 부족함을 드러내었습니다. 이로 인해 공정한 ML 기반 공공 보건 실천의 발전을 위한 중요한 레버리지 포인트가 제시됩니다.



### From Consumption to Reflection: Designing Human-AI Relations for Stable Reasoning (https://arxiv.org/abs/2606.11195)
- **What's New**: 이번 논문에서는 Relational Reflective Intelligence (RRI)라는 새로운 개념을 소개합니다. RRI는 대형 언어 모델(Large Language Models, LLMs)과 인간 간의 반영(reflection) 과정을 auditable reasoning loops를 통해 운영하는 인퍼런스 타임 거버넌스 레이어입니다. 이 연구는 LLM이 인간의 사고처럼 인지적 취약성(cognitive vulnerabilities)을 상속받으며, 이는 직관적 단축(intuitive shortcuts), 자아와 현실의 혼동(confusion between representation and reality) 등을 포함합니다.

- **Technical Details**: RRI는 모델 내부가 아닌 주변에서 작동하여 인간과 LLM 간의 안정적이고 auditable한 reasoning 구조를 제공합니다. RRI는 Rose-Frame, Architect's Pen, inference-time workflow의 세 가지 구성 요소로 이루어져 있습니다. Rose-Frame은 Reasoning에서 발생할 수 있는 오류를 식별하고, Architect's Pen은 중요한 순간에 반영 단계를 삽입하며, inference-time workflow는 모델 재훈련 없이 이러한 단계를 포함합니다.

- **Performance Highlights**: 이러한 구성 요소들은 인간과 AI의 상호작용을 공동 reasoning 시스템으로 변화시킵니다. RRI는 명시적인 체크포인트, 갈등 표출(conflict surfacing), 가정의 auditable한 추적을 통해 서로의 한계를 보완하는 구조화된 상호작용을 만듭니다. 이 연구는 AI 안전성을 인지 아키텍처 문제로 재구성함으로써, 신뢰할 수 있는 결정을 내리기 위해 상호작용 과정에 반영을 직접적으로 주입하는 것이 어떻게 중요한지를 강조합니다.



New uploads on arXiv(cs.RO)

### World Pilot: Steering Vision-Language-Action Models with World-Action Priors (https://arxiv.org/abs/2606.12403)
Comments:
          Project Website: this https URL

- **What's New**: 이번 논문에서는 World Pilot라는 새로운 Vision-Language-Action (VLA) 프레임워크를 소개합니다. 이 프레임워크는 기존의 이미지-텍스트 쌍을 기반으로 한 정적인 세심한 grounding을 넘어, World-Action Model (WAM)에서 가져온 priors를 정책에 통합하여 작동합니다. 이를 통해 로봇 조작 작업에서 체계적이고 일관된 Scene Dynamics를 보다 잘 표현할 수 있게 됩니다.

- **Technical Details**: World Pilot는 Latent Steering과 Action Steering 두 가지 보완 경로를 통해 WAM의 출력을 정책에 연결합니다. Latent Steering은 장면 진화(latent evolution)를 인식 계층에 주입하여 공간-시간 동역학(spatiotemporal dynamics)을 예측하도록 돕고, Action Steering은 예상 궤적(trajectory)을 단일 prefix token으로 압축하여 행동 생성기(action generator)에 의도와 함께 제공하게 됩니다. 이를 통해 VLA는 세부적인 동역학 정보를 유지하면서도 정책을 유연하게 조정할 수 있게 됩니다.

- **Performance Highlights**: World Pilot는 LIBERO-Plus zero-shot OOD 벤치마크에서 84.7%라는 최첨단의 성공률(total success rate)을 달성했으며, 네 가지 조작 작업(real-robot tasks)에서 모든 실제 로봇 설정에서 가장 높은 성공률을 기록했습니다. 이러한 결과는 시점(viewpoint), 기하학(geometry), 변형 상태(deformable state) 및 자세(pose) 변화에 대한 저항력이 강하다는 것을 보여주었으며, 각 경로가 독립적으로 기여함을 확인했습니다.



### UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning (https://arxiv.org/abs/2606.12372)
Comments:
          Project page: this https URL

- **What's New**: 새로운 연구에서는 HiL-RL(인간-참여 강화 학습) 분야에 대한 개선으로 UniIntervene 모델을 제안합니다. 이 모델은 비생산적 탐색을 감지하고, 인간 운영자가 개입할 필요 없이 정책을 자율적으로 고가치 상태로 복구합니다. 기존의 HiL-RL 프레임워크가 가진 인력 의존성을 줄임으로써, 노동 비용과 실제 환경에서의 확장성을 증가시킬 수 있는 가능성을 제공합니다.

- **Technical Details**: UniIntervene는 미래조건부(action-value estimation) 행동 가치 예측을 통해 현재 행동의 잠재적인 결과를 예측하고, 이에 따라 유도된 가치를 평가합니다. 이 정보를 바탕으로 시간적 가치 리스크 비평가(temporal value-risk critic)가 최근 가치 동역학을 집계하고, 지속적인 정체나 저하가 감지될 경우 개입을 유도합니다. 이러한 과정을 통해 UniIntervene는 비생산적 상호작용을 회복 가능하고 정보가 풍부한 경로로 전환시킵니다.

- **Performance Highlights**: 다양한 실제 조작 작업에 대한 실험 결과, UniIntervene는 최신 HiL-RL 기법 대비 평균 성공률을 8.6% 향상시키고, 인간 개입 수를 57% 감소시키는 성과를 보였습니다. 이는 HiL-RL에서의 인간 개입 비용을 크게 줄이면서도 효율적인 학습을 가능하게 합니다.



### APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies (https://arxiv.org/abs/2606.12366)
- **What's New**: 본 논문에서는 Vision-Language-Action (VLA) 모델을 개선하기 위한 새로운 방법인 APT(Action expert Pre-Training)를 제안합니다. APT는 행동 전문가(action expert)를 언어가 없는 Vision-Action (VA) 우선순위로 사전 훈련(pretraining)한 후 언어 토큰을 주입하는 두 단계의 훈련 방법입니다. 이 접근법은 VLA 데이터에서의 구조적 불균형을 해소하고, 기존 VLA 아키텍처의 언어 일반화 성능을 향상시키는 데 기여합니다.

- **Technical Details**: APT는 VLA 정책을 베이지안 관점에서 분해하여, VA 우선순위(πp)와 언어 조건부 VLA 가능성(ℒ)으로 나눕니다. 첫 번째 단계에서는 정지된 VLM으로부터 시각-행동 쌍을 기반으로 행동 전문가를 사전 훈련하고, 두 번째 단계에서는 새로운 주의(attention) 레이어를 통해 언어 토큰을 주입하여 트레이닝합니다. 이 과정을 통해 VLM의 시각 모터 제어 능력을 보존하면서 행동 전문가가 언어 훈련을 통해 능력을 강화하도록 합니다.

- **Performance Highlights**: 종합 실험 결과, APT는 보이지 않는 명령 및 조합적 작업에서 일관된 성장을 달성하며, VLA 모델의 언어 일반화 성능을 현저하게 향상시킵니다. 다양한 메인스트림 연속-행동 VLA 아키텍처에 적용 가능하며, 이러한 구조는 시뮬레이션 및 실제 환경에서의 성능 개선을 입증합니다. APT의 제안된 두 단계 훈련 방법은 효과적인 언어 일반화와 더불어 기존 기술들보다 두드러진 성과를 보여줍니다.



### Traceable Virtual Sea Trials in the Marine Robotics Unity Simulator for Manoeuvring Assessment of Unmanned Surface Vehicles (https://arxiv.org/abs/2606.12349)
- **What's New**: 이번 연구는 Marine Robotics Unity Simulator (MARUS)를 확장하여 자동화된 TC(회전성 동작) 및 ZZ(지그재그 동작) 시험을 위한 표준화된 Virtual Sea Trial 프레임워크를 도입했습니다. 이 프레임워크는 명확한 명령 실행 로그와 시스템 식별(SI)에 초점을 맞춘 데이터 조정을 통해 IMO 및 ITTC에 따른 성능 지표의 자동 추출을 지원합니다. 주요 기여로는 TC 및 ZZ 데이터를 수집하고 후처리하는 파이프라인이 포함되어, 시뮬레이터 기반 동작의 반복 가능성과 감사 가능성을 향상시킵니다.

- **Technical Details**: 연구의 핵심은 차별 추진(differential thrust) 조정 방법을 명시적으로 분리하고, 이를 통해 명령 입력을 기록하고 실제 실행을 추적하는 것입니다. 수집된 데이터는 IMO 및 ITTC의 기준을 만족하는 동작 메트릭으로 가공되며, 이는 USV의 유체역학적 파라미터 추정 및 디지털 트윈(Digital Twin) 교정에 유용합니다. TC 테스트에서는 정규화된 전진이 포트와 스타보드 사이에서 약 3.9% 다르고, 전술 직경은 약 4.6%에서 4.7% 다릅니다.

- **Performance Highlights**: TC 및 ZZ 시험의 사례 연구 결과, 두 시험 모두 반복 가능한 성능을 보여줍니다. ZZ 시험에서의 첫 번째와 두 번째 과도 초과 각도가 +10도 및 -10도 동작에서 각각 1도를 초과하지 않아 IMO 기준을 만족했습니다. 전반적으로 이 프레임워크는 USV를 위한 신뢰할 수 있는 가상 해양 시험 환경을 제공하며, 유체역학적인 파라미터 추정과 디지털 트윈 교정을 지원합니다.



### UGV-Conditioned Multi-UAV Informative Planning on a Shared Exposure Belief (https://arxiv.org/abs/2606.12306)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문은 위험이 추가된 넓은 환경에서 지상 차량(UGV)의 안전한 내비게이션을 위한 새로운 접근 방식을 제시합니다. 기존의 공중 정찰 시스템이 환경을 Mapping 하는 데 중점을 두는 반면, 우리는 무인 항공기(UAV) 팀과 UGV 간의 협업을 통해 안전성을 향상시키는 데 중점을 두었습니다. 이 연구는 실시간으로 업데이트되는 공동 노출 신념(shared exposure belief)을 통해 공중 감지를 지상 차량에 가장 관련성이 높은 지역으로 직접 유도합니다.

- **Technical Details**: 연구에서는 UGV가 미지의 위협 지역을 탐색하는 동안 UAV 팀이 조정하여 중복 감지를 피합니다. 이 시스템은 공중 관측 데이터를 통해 공동 노출 신념을 업데이트하며, 이를 통해 UAV 팀과 UGV가 동시에 안전을 극대화할 수 있게 합니다. 또한, 해당 지역에 대한 공간적 할당(spatial region assignment)을 통해 비효율적인 공중 감시를 줄이는 방법을 사용합니다.

- **Performance Highlights**: 시뮬레이션 실험 결과, 제안한 접근 방식은 위험 수준을 고려하지 않는 시스템에 비해 UGV의 누적 노출을 38% 줄였습니다. 또한, 다수의 UAV 조정을 통해 중복되는 공중 감시를 38.8%에서 3.7%로 대폭 줄이는 성과를 얻었습니다. 이 결과는 UAV와 UGV 간의 효율적 협업이 지상 차량의 안전성을 얼마나 극대화할 수 있는지를 잘 보여줍니다.



### Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering (https://arxiv.org/abs/2606.12299)
Comments:
          22 pages, 14 tables, 14 figures

- **What's New**: 이 논문에서는 Vision-Language-Action (VLA) 모델의 성능을 향상시키기 위한 반응형 언어 피드백 정책(Language Feedback Policy, LFP)을 제안합니다. 이 접근법은 언어 세quences를 상호작용적으로 검색하여 닫힌 루프(closed-loop) 성능을 향상시킵니다. VLA 모델이 주어진 언어 지침에 따라 일관되게 성공적인 행동을 끌어낼 수 있도록 하며, 원래 지침으로 되돌아갈 수 있도록 배웁니다.

- **Technical Details**: 제안된 방법은 우선 로봇 행동의 내레이터 비디오를 사용하여 언어 시퀀스에 대한 제안 분포를 생성합니다. 그 후, 이 제안 시퀀스를 사용하여 닫힌 루프 VLA 롤아웃을 통해 성공적인 행동을 유도하는 언어 수정을 평가합니다. 또한, 성능 향상을 예측하는 개선 헤드(improvement head)를 학습하여 분포 변동 상황에서도 유해한 스티어링 개입을 방지합니다.

- **Performance Highlights**: 시뮬레이션 및 하드웨어 실험을 통해, 제안된 LFP는 기본 VLA 성능을 시뮬레이션에서 24.7%, 하드웨어에서 65.0% 향상시킨다는 것을 보여주었습니다. 우리의 접근법은 언어 스티어링을 통해 회복 행동을 유도할 수 있는 강력한 무해 보장을 제공합니다. 이는 기존의 오픈 루프 프롬프트 재구성 전략과는 차별화된 성능입니다.



### PEBRE: An Open-Hardware Compute and Perception Add-On for the Pepper Robo (https://arxiv.org/abs/2606.12112)
- **What's New**: 본 논문에서는 Pepper 로봇에 대한 빠른 소프트웨어 개발을 위해 설계된 오픈 하드웨어인 PEBRE를 소개합니다. PEBRE는 Jetson Orin Nano, Logitech BRIO, Intel RealSense D435i와 같은 외부 컴포넌트를 통합하여 Pepper의 계산 및 인식 능력을 크게 향상시킵니다. 이 개발은 커뮤니티에 기여하여 Pepper 로봇의 기능을 연장하고 빠른 소프트웨어 개발을 촉진할 수 있는 기반을 마련합니다.

- **Technical Details**: PEBRE의 설계는 Pepper의 센서 및 계산 능력을 현대 HRI 및 사회 로봇 응용 프로그램의 요구에 맞추어 확장하는 데 초점을 맞추었습니다. 새로운 센서와 컴퓨팅 모듈을 유연하게 통합할 수 있도록 하여 연구 요구에 따라 시스템을 적응할 수 있습니다. NVIDIA Jetson Orin Nano를 선택하여 새로운 센서와 온보드 센서의 데이터를 처리하며, 외부 컴퓨터 및 클라우드 서비스와의 통신을 지원합니다.

- **Performance Highlights**: PEBRE는 Pepper의 오디오 및 비주얼 인식 능력을 개선하기 위해 경량의 고성능 주변장치를 설계하여 통합했습니다. Samson UB1 및 RØDE VideoMicro II 마이크와 Logitech BRIO 및 Intel RealSense D435i 카메라의 장착을 통해 Pepper의 인식 능력을 극대화했습니다. 이러한 업그레이드는 Pepper가 다양한 환경에서 더 향상된 상호작용과 반응성을 제공할 수 있도록 돕습니다.



### Fibration Trees: A Unified Approach to Multi-Robot Motion Planning (https://arxiv.org/abs/2606.12070)
Comments:
          23 pages, 12 figures

- **What's New**: 이번 논문에서는 고차원 다중 로봇 모션 플래닝 문제를 해결하기 위한 통일된 프레임워크인 'fibration trees'를 소개합니다. 이는 상태 공간을 노드로, fibration을 엣지로 구성한 트리 구조를 갖고 있으며, fibration은 고차원 공간에서 저차원 공간으로의 투영을 모델링합니다. 이를 통해 기존의 우선순위 계획, 병렬 분해, 그리고 작업 공간 투영을 하나의 일관된 형식으로 통합합니다.

- **Technical Details**: Fibration trees는 상태 공간의 특정 종류의 분해 및 투영을 단일 프레임워크로 결합하는 방법으로, 기존 방법들은 주로 따로 다루어진 반면, 본 방법은 이를 통합하여 보다 효율적인 모션 플래닝을 가능하게 합니다. Fibration-RRT는 이와 같은 fibration trees를 기반으로 하여 확장된 샘플링 기반 모션 플래너입니다. 이 알고리즘은 사용자가 정의한 fibration trees에서 작동하며, 확률적으로 완전함이 증명되었습니다.

- **Performance Highlights**: Fibration-RRT는 32개의 서로 다른 시나리오에서 다중 로봇 팀과 최대 96도의 자유도를 대상으로 한 실험에서 평가되었습니다. 그 결과, Fibration-RRT는 사용자 정의 fibration trees를 효과적으로 이용하여 고차원 문제를 효율적으로 해결할 수 있는 가능성을 보여주었습니다. 이로 인해 다중 로봇 모션 플래닝을 위한 강력하고 통합된 프레임워크로서 fibration trees의 위상이 정립되었습니다.



### Point Cloud Segmentation for Autonomous Clip Positioning in Laparoscopic Cholecystectomy on a Phantom (https://arxiv.org/abs/2606.12048)
Comments:
          8 pages, 5 figures, accepted to IEEE Robotics and Automation Letters (RAL)

- **What's New**: 이 논문에서는 로봇 보조 수술(Robot-Assisted Surgery)에서의 첫 번째 자율 클립 위치 지정 시스템을 제시합니다. 이 시스템은 단일 카메라에서 색이 없는 포인트 클라우드를 분할하여 자율적으로 목표 위치를 정하고, 이 과정에서 한명의 인간 조작자가 위치를 조정할 수 있도록 하는 기능을 갖추고 있습니다. 이 접근은 특히 일반 수술에서 가장 흔하게 시행되는 복강경 담낭절제술(laparoscopic cholecystectomy)에서 매우 중요한 성과입니다.

- **Technical Details**: 시스템은 60개의 수작업으로 라벨이 붙은 데이터가 부족한 의료 환경을 고려하여, 128,000개의 합성 포인트 클라우드로 미리 훈련되었습니다. 여기에 두 가지 새로운 데이터 증강 기법을 도입하여 성능을 향상시켰습니다: VariableJitter와 RandomPatches를 포함합니다. 이 시스템은 한 단계에서 목표 위치의 정확성을 0.75mm로 유지하며, 95%의 성공률을 기록하고 있습니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 모든 클립 위치 지정 작업에 대해 100%의 성공률을 달성했습니다. 이를 통해 더욱 안전하고 신뢰할 수 있는 수술 지원을 가능하게 하며, 기계의 운동 경로가 인간 조작자에 의해 사전에 확인될 수 있어 해석 가능성을 높입니다. 또한 로봇 시스템의 피드백 기능을 통해 인간 조작자는 실시간으로 위치를 수정하고 확인할 수 있습니다.



### KinematicRL: A Sim-to-Real Reinforcement Learning Framework For Social Navigation With Kinodynamic Feasibility (https://arxiv.org/abs/2606.12042)
Comments:
          Accepted by IEEE Transactions on Automation Science and Engineering (T-ASE)

- **What's New**: 이 논문은 소셜 내비게이션에서 심각한 sim-to-real 격차를 해소하기 위한 통합된 프레임워크를 제시합니다. 특히, 로봇의 동적 피할 수 있는 경로를 생성하기 위해 고차 제어 입력을 사용하는 것을 정당화하는 이론적 분석을 제공합니다. 또한, 2D LiDAR 기반의 군집 추적 파이프라인을 도입하여, 카메라와 LiDAR의 융합으로 인한 복잡성 문제를 해결합니다.

- **Technical Details**: 논문에서는 고차 제어 입력이 로봇의 시뮬레이션과 실제 위치 간의 추적 오류를 지수적으로 감소시키는 방법을 제안합니다. 또한, 차별적 구동 로봇에 적합한 두 번째 제어 형식을 개발하고, 이를 통해 스토캐스틱 반복 선형 2차 조절기(iLQR)를 사용하여 정책을 사전 훈련합니다. 군집 기반 인간 추적 방식은 2D LiDAR만을 이용하여 위치를 추적하며, 인접한 사람들의 거리를 기반으로 신뢰할 수 있는 속도 추정을 제공합니다.

- **Performance Highlights**: KinematicRL 정책은 모든 기계적 성능 지표에서 기초 모델을 지속적으로 초과 달성합니다. 이 정책은 제안된 추적 파이프라인과 결합되어, 최소한의 수정으로 실제 차별적 구동 로봇에 배포될 수 있습니다. 실험 결과, KinematicRL은 다양한 탐지된 인간의 수에도 유연하게 적응하며, 실제 환경에서도 안정적으로 동작합니다.



### VICX: Generalizable Robot Manipulation via Video Generation and In-Context Operator Network (https://arxiv.org/abs/2606.12028)
Comments:
          The first two authors contributed equally to this work

- **What's New**: 이번 연구는 **VICX (Video generation and In-Context eXecution)**라는 새로운 조작 프레임워크를 소개합니다. 이 프레임워크는 **비디오 생성 모델**을 활용하여 높은 수준의 시각적 계획을 수행하고, **비디오-경로 (Video-to-Trajectory)** 인터페이스를 통해 이러한 계획을 실제 로봇 상태로 변환합니다. 두 가지 모듈을 통합하여 상호작용 성능을 높이고, 로봇이 다양한 작업을 수행할 수 있도록 지원합니다.

- **Technical Details**: VICX는 고급 시각적 계획과 저급 상태 실행을 두 단계로 나누어 수행합니다. 먼저 **비디오 생성 모델**이 미래 실행 비디오를 예측한 후, **V2T-ICON**이 이 시각적 계획을 실행 가능한 로봇 상태 경로로 변환합니다. V2T-ICON은 훈련 세트에서 유사한 이미지-상태 쌍을 검색하고 이를 사용하여 로봇 상태를 예측하는데, 이는 로봇의 다양한 관점, 배경 및 조명 조건에서 일반화하기 용이하게 만듭니다.

- **Performance Highlights**: 실험 결과, VICX는 **Meta-World** 환경에서 놀라운 일반화 성능을 보여주었습니다. V2T-ICON이 세 가지 원천 작업에 대해 훈련된 경우에도 VICX는 아홉 가지 조작 작업에 대한 일반화를 성공적으로 달성했습니다. 또한, 폐쇄 루프 자기 수정(closed-loop self-correction) 및 로봇 구조에 따른 전이(transfer) 능력을 구현하여, 생성을 통한 왜곡 및 외부 방해 요소로 인한 오류를 줄이는 데 효과적임을 입증했습니다.



### Learning Unions of Convex Sets via Invertible Latent Decomposition for Path Planning (https://arxiv.org/abs/2606.12027)
- **What's New**: 이번 연구에서는 ILD(Invertible Latent Decomposition)라는 새로운 프레임워크를 제안합니다. 이는 변형 가능한 매핑과 명시적인 볼륨들의 집합을 공동으로 학습하여, 효율적인 충돌 회피 경로 계획을 가능하게 합니다. 특히, 이 방법은 복잡한 기하학적 환경에서도 효과적으로 작동하며 기존 방식들과 비교하여 더 나은 성능을 보입니다.

- **Technical Details**: 이 프레임워크는 invertible neural network를 이용하여 원래의 구성 공간과 잠재 공간 간의 일대일 대응을 보장합니다. ILD는 잠재 공간에서의 경로 계획을 통해 충돌 회피 경로를 생성하며, Visibility-Guided Sampling (VGS)를 통해 다각형의 연결성을 향상시킵니다. 이러한 구성 덕분에 경로 계획의 최적화를 위해 명시적 제약조건을 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: 실험 결과 ILD는 2D 내비게이션, 6자유도 및 14자유도 작업 환경에서 기존 방법들보다 최우수 성능을 기록하였습니다. 특히 테스팅 시 정밀도가 1.0에 달하며, False Positive는 관찰되지 않아 안정성을 입증하였습니다. 이러한 성과는 ILD가 실시간으로 충돌 회피 계획을 수행할 수 있게 해줍니다.



### MPPI-based Informative Trajectory Planning for Search and Capture of Drifting Targets with ASVs (https://arxiv.org/abs/2606.12019)
- **What's New**: 본 논문에서는 자율 수면 차량(ASV)이 동적 환경에서 떠다니는 쓰레기와 같은 여러 이동 대상의 검색 및 포획 문제를 해결하기 위한 하이브리드 계획 프레임워크를 제안합니다. 이 접근법은 기존의 간단한 의사결정 방식과는 달리, 예측 길이를 고려하며, 복수의 목적을 최적화하는 지속 가능한 경로를 생성하여 탐색과 추적 간의 균형을 유지합니다. 또한 새로운 공간-시간 정보 계획 방식인 모델 예측 경로 적분(MPPI) 제어를 채택하여, 자율 차량이 탐색 및 포획을 동시에 수행할 수 있도록 합니다.

- **Technical Details**: 제안된 계획 방법은 MPPI를 기반으로 하여, 자율 수면 차량이 긴 예측 호에 걸쳐 연속적인 경로를 최적화하고, 안전하고 실행 가능한 경로를 보장하는 복합 목표 비용을 균형있게 설계합니다. 특정 목표가 탐지되면, 차량이 물리적으로 잡을 수 있도록 순수 추적 지침 제어기로 전환합니다. 이러한 과정을 통해 드리프트하는 대상에 대한 즉각적이고 효과적인 추적이 가능합니다.

- **Performance Highlights**: 실험 결과, 제안한 계획자는 선택한 계획 기준선보다 우수한 성능을 보여주었습니다. 필드 실험을 통해 이 프레임워크가 실제 모니터링 시나리오에서 효과적임을 입증하였습니다. 또한 시뮬레이션에서 적응형 및 비적응형 계획 기준선에 비해 향상된 결과를 보였고, 각 비용 구성요소의 효과도 검증되었습니다.



### Deformable In-Hand Slip-Aware Tactile Sensor with Integrated Velocity, Force/Torque, and Pressure Map Sensing (https://arxiv.org/abs/2606.11952)
- **What's New**: 이 논문은 슬립 인식 제어(slide-aware control)를 위한 새로운 촉각 센서를 소개합니다. 이 센서는 속도(velocity), 힘/토크(force/torque), 그리고 압력 맵(pressure map) 센싱을 하나의 장치와 변형 가능한 접촉 패드(deformable contact pad)로 통합하고 있습니다. 기존 센서와의 차별점은 여러 센싱 모달리티를 단일 구조에 결합했다는 것입니다.

- **Technical Details**: 이 촉각 센서는 여러 센싱 모달리티를 통합하여 슬라이딩 속도, 6자유도 힘 및 토크, 접촉 압력 맵을 측정할 수 있도록 설계되었습니다. 센서는 탈착 가능한 접촉 패드를 통해 객체와 상호작용하며, 다층으로 구성되어 있어 신속한 프로토타이핑 기법을 사용하여 제작할 수 있습니다. 각 센서 구성 요소는 USB-C 인터페이스와 통신하기 위해 마이크로컨트롤러를 포함하고 있습니다.

- **Performance Highlights**: 이 센서의 성능 평가를 통해 객체의 속도, 힘 및 압력 분포를 안정적으로 추적할 수 있는 능력을 입증했습니다. 새로운 접촉 패드는 향상된 접촉 동역학과 접촉 토크 권한을 제공하여 곡면 객체에 대한 조작을 보다 부드럽고 안정적으로 수행할 수 있게 합니다. 이 모든 측정값은 강인한 그립 상태 및 객체 속성과 외부 접촉을 추정하는 데 기여합니다.



### DuoBench: A Reproducible Benchmark for Bimanual Manipulation in Simulation and the Real World (https://arxiv.org/abs/2606.11901)
- **What's New**: DuoBench는 이중 팔 조작을 위한 새로운 벤치마크 프레임워크로, FR3 Duo 플랫폼에서 작동합니다. 이 시스템은 11개의 작업을 통해 4가지 조정 범주를 포괄하며, 시뮬레이션과 현실 세계에서의 재현 가능성을 결합합니다. 또한 이 프레임워크는 섬세한 의미적 실패 분석을 지원하며, 인간 원격 조작 데이터를 제공합니다. 이는 여러 두 팔 정책의 성능을 평가하는 데 중요한 진전을 가져올 것으로 기대됩니다.

- **Technical Details**: DuoBench는 Robot Control Stack (RCS) 생태계 위에 구축되었으며, 각 작업은 Markov Decision Process (MDP)를 기반으로 구성됩니다. 환경은 다양한 포장을 통해 새로운 환경으로 래핑되어 다양한 상태 및 행동 공간을 생성합니다. 이 시스템은 인간 원격 조작자 또는 학습된 정책을 포함한 확률적 에이전트를 사용할 수 있으며, 데이터 레코더 래퍼를 통해 작업 진행 상황을 기록합니다. 이는 각 작업이 인간 및 로봇의 협력적 조작을 평가하기 위한 체계적인 접근 방식을 제공하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 현재 정책들은 이중 팔 조작에서 여전히 도전 과제를 안고 있으며, 특히 초기 상호작용 단계와 팔의 병렬 실행, 시뮬레이션과 현실 세계 간의 전이에 어려움을 겪고 있습니다. DuoBench는 이러한 실패 모드를 진단하고, 이중 팔 정책 학습을 위한 향후 방법을 연구하는 데 도움을 줄 수 있는 재현 가능성 있는 테스트베드를 제공합니다. 벤치마크 결과는 조정 문제의 천착적인 분석을 통해 모델의 약점을 드러내고, 표적 발전을 지원하는 데 중요한 통찰력을 제공합니다.



### Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation (https://arxiv.org/abs/2606.11891)
Comments:
          Accepted at the ICRA 2026 Workshop on Reinforcement Learning for Imitation Learning (RL4IL), Vienna, Austria. 4 pages, 2 figures

- **What's New**: 이 논문은 로봇의 보행과 조작을 동시에 조정하는 다목적 강화 학습(multi-objective reinforcement learning)의 중요성을 강조합니다. 특히, 단일 크리틱(unified critic)과 이중 크리틱(dual critic) 구조 간의 비교를 통해, 이중 크리틱이 더 빠르고 효율적인 성능을 보여준다는 점에서 의미가 있습니다. 각기 다른 보상 신호를 갖는 별개의 크리틱을 사용하는 것이 학습 효율성을 크게 끌어올릴 수 있음을 입증했습니다.

- **Technical Details**: 이 연구에서 우리는 Unitree G1 휴머노이드 로봇을 NVIDIA Isaac Lab에서 훈련시키면서, 13단계의 커리큘럼을 통해 보행과 조작 정책을 연속적으로 훈련했습니다. 두 가지 크리틱 아키텍처, 즉 단일 크리틱과 이중 크리틱 구조를 사용하였고, 각 구조에서 효율성을 측정했습니다. 결과적으로 이중 크리틱 구조가 3.5배 빠른 도달 속도를 기록하며, 2배 높은 처리량을 달성했습니다.

- **Performance Highlights**: 표준화된 평가에서 이중 크리틱 정책은 목표 도달 속도가 3.5배 빨랐고(6.5와 22.6 시뮬레이션 스텝), 1,000스텝당 14.3번의 검증된 도달 횟수로 처리량이 2배 더 높았습니다. 이는 각각의 구조에서 보상 설계나 커리큘럼 진행과는 별개로, 단순히 아키텍처의 차이가 큰 차이를 만들어 낸다는 것을 의미합니다. 또한 추가적인 리워드 해킹(anti-gaming) 메커니즘은 효율성을 향상시키지 못하는 것으로 나타났습니다.



### Modular Anthropomorphic Hand Design via Multi-Parameter Finger Benchmarking and Selection (https://arxiv.org/abs/2606.11826)
Comments:
          14 pages, 13 figures. Submitted to an IEEE journal for possible publication

- **What's New**: 이 연구는 손가락의 모듈화를 통해 전체 로봇 손의 성능을 향상시킬 수 있는 플랫폼을 제안합니다. 기존의 손 디자인 접근법이 아닌, 각 손가락을 모듈화하여 성능 지표를 기반으로 최적화하는 새로운 방법을 포함하고 있습니다. 또한, 이를 통해 신속한 프로토타입 스크리닝을 수행하여 최적의 손가락 디자인을 선정하는 방법을 소개합니다.

- **Technical Details**: 모듈식 손 플랫폼은 다양한 손가락 구조 및 소프트 스킨 디자인을 체계적으로 평가하기 위해 개발되었습니다. 손가락 모듈은 원위 간관절(DIP) 및 근위 간관절(PIP)로 구성되며, 각 손가락 모듈은 핸드에 고정되어 독립적으로 테스트할 수 있습니다. 이 연구에서는 다양한 뼈 및 관절 디자인과 교환 가능한 스킨을 개발하여, 손가락 디자인의 효과를 정량적으로 평가하고자 하였습니다.

- **Performance Highlights**: 최적화된 JCPB 손은 기준 모델에 비해 83.3% 더 높은 측면 리프팅 용량을 달성했으며, 전체 손 잡기 힘이 약 32% 증가하였습니다. 또한, 노브 회전 범위는 87.5% 더 넓어졌습니다. 이러한 성능 향상은 최적화된 손가락 모듈과 스킨 디자인에 기인하였습니다.



### Human-Guided Co-Manipulation of Carbon Fiber Plies (https://arxiv.org/abs/2606.11818)
Comments:
          Accepted to the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026)

- **What's New**: 이 논문에서는 탄소 섬유 층(carbon fiber plies)의 공동 조작(co-manipulation)을 위한 다양한 제어 방법을 소개하고 분석합니다. 이러한 방법들은 각각의 장단점을 평가하며, 인지 및 조작 면에서의 도전 과제를 해결하기 위한 효과적인 해결책을 모색합니다. 특히, 음성 명령(speech commands), 비전 기반 손목 추적(wrist-tracking), 그리고 힘의 조절(compliant control)을 조합한 다중 모달 조작이 작업의 직관적인 제어를 위한 최적의 솔루션을 제공할 것이라고 제안합니다.

- **Technical Details**: 본 연구의 시스템 아키텍처는 감지(perception), 제어 아키텍처(control architecture), 로봇(robots) 세 가지 주요 블록으로 구성됩니다. 감지 블록은 환경으로부터 오디오 및 비주얼 데이터를 수신하여 처리한 후, 이를 제어 블록에 전달합니다. 제어된 포스 컨트롤(force control) 및 카르테시안 임피던스 제어(Cartesian impedance controller)의 통합을 통해, 인간 운영자는 로봇의 동작을 원활하게 제어할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 다중 모달 조작을 통한 공동 조작 방식이 단일 경로에 기반한 전통적인 방법보다 더 높은 조작 효율성을 나타내며, 더욱 안전한 작업 환경을 제공하는 것으로 나타났습니다. 특히, 인간과 로봇 간의 상호작용을 중시한 기법들이 생산성 향상에 기여한다는 점에서 큰 의의를 가집니다. 본 연구는 조작을 위한 복잡한 경로를 요구하는 상황에서도 성공적으로 적용될 수 있는 가능성을 제시합니다.



### Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning (https://arxiv.org/abs/2606.11767)
Comments:
          23 pages, 6 figures

- **What's New**: 본 논문에서는 촉각 기반의 맹목적 그립(Blind Grasping) 기술을 물리적 다관절 로봇 손에서 구현하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Real2Sim 촉각 보정 절차를 통해 촉각 신호의 정확도를 높이고, 레이아웃 인식 촉각 인코더를 활용하여 희소한 촉각 관측의 표현력을 개선합니다. 또한, 강화 학습과 확산 정책 학습을 결합하여 다양한 객체 기하형태에 대한 그립 전략을 수립합니다.

- **Technical Details**: 이 연구에서는 시뮬레이터와 실제 로봇 손의 촉각 신호 간의 차이를 줄이기 위한 Real2Sim 보정 절차를 도입합니다. 이 방법은 시뮬레이션된 촉각 이벤트를 실제 발생하는 촉각 이벤트와 정렬시켜줍니다. 또한 각 센서의 3D 위치를 반영한 레이아웃 인식 촉각 인코더를 통해 희소한 촉각 정보에서 객체 정보를 우선적으로 추출할 수 있는 방법을 제공합니다. 마지막으로, 성공적인 촉각 그립 궤적을 수집하여 촉각 조건에 맞춘 확산 정책을 학습합니다.

- **Performance Highlights**: LEAP 핸드를 이용한 실제 실험에서 제안된 정책은 20개의 다양한 객체에 대해 27%의 실제 그립 성공률을 기록했습니다. 이는 실제 그립 시연이나 시각적 입력 없이도 가능하다는 점에서 주목할 만합니다. 시뮬레이션 실험에서는 레이아웃 인식 촉각 사전 학습이 그립 성능을 개선했다는 결과를 보여줍니다. 또한 Real2Sim 보정은 시뮬레이션과 하드웨어 간의 촉각 접촉 사건의 일관성을 높인 것으로 확인되었습니다.



### TacCoRL: Integrating Tactile Feedback into VLA via Simulation (https://arxiv.org/abs/2606.11743)
- **What's New**: 본 논문에서는 TacCoRL이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Vision-Language-Action (VLA) 모델에 Tactile feedback을 통합하여 로봇 매니퓰레이션을 향상시키는 방법을 소개합니다. TacCoRL은 대규모 촉각 사전 훈련이나 복잡한 실제 접촉 탐색 없이 시뮬레이션 기반 강화 학습(simulation-based reinforcement learning)을 통해 VLA 정책을 개선합니다.

- **Technical Details**: TacCoRL은 두 단계의 sim-to-real 파이프라인을 사용하여 촉각 피드백을 통합합니다. 초기 단계에서는 혼합된 시뮬레이션-실제 훈련을 통해 촉각 조건의 행동 우선 사항을 설정합니다. 이후, 드문 사례에 대한 정책 수정을 위해 희박 보상 강화 학습을 적용하여 실제 데이터 정규화를 통해 정책을 개선합니다.

- **Performance Highlights**: 최종적으로 TacCoRL을 통해 개발된 visuo-tactile 정책은 4가지 이인 접촉이 많은 작업에서 평균 72.5%의 성공률을 달성하였으며, 이는 기준선인 50.0%에 비해 크게 향상된 결과입니다. 이성공적인 결과는 TacCoRL의 프레임워크가 기존의 비전 및 모방 기반 방법들보다 우수한 성능을 제공함을 나타냅니다.



### Explore From Sketch: Accelerating UAV Exploration in Large-scale Environments with Prior Maps (https://arxiv.org/abs/2606.11708)
Comments:
          25 pages, 22 figures

- **What's New**: 이 논문은 LiDAR 기반의 UAV 탐사를 위한 새로운 탐사 프레임워크를 제안합니다. 기존 방안과 달리, 불완전하고 서로 다르며 정렬되지 않은 2D 이전 지도들을 효과적으로 이용하여 드론의 탐사 효율성을 향상시킬 수 있습니다. 이를 통해 UAV가 복잡한 환경에서 보다 효율적으로 탐사를 할 수 있도록 지원합니다.

- **Technical Details**: 논문에서는 LiDAR 관측치를 2D 이전 지도와 정렬하기 위한 강력한 2D-3D 포인트 클라우드 정합 파이프라인을 소개합니다. 이 파이프라인은 단일 프레임 후보 검색을 위한 GeoContext descriptor, 다중 프레임 검증 메커니즘, 그리고 정밀한 보정을 위한 Scale-ICP 알고리즘을 결합합니다. 이러한 정합 모듈은 전반적인 탐사 계획을 지원하기 위해 불확실성을 고려한 계층적 시점 계획 전략을 개발하여, 여러 정합 가설을 생성합니다.

- **Performance Highlights**: 본 논문에서 제안한 방법은 최신 방법과 비교하여 탐사 효율성을 최대 34.2% 향상시키고 비행 거리에서는 37.9% 감소되는 성과를 보였습니다. 다양한 시뮬레이션 및 현장 실험을 통해 결함이 있는 이전 지도의 불완전성 및 변형에 대한 강인성을 입증했습니다. 이러한 결과는 UAV 탐사 기술의 실용화에 기여할 것으로 기대됩니다.



### Improving Human Diving Endurance with a Field-Deployable, Untethered Exoskeleton (https://arxiv.org/abs/2606.11704)
- **What's New**: 본 연구에서는 DiveMate라는 최초의 비접속식(exoskeleton) 잠수 보조 기기를 소개합니다. DiveMate는 현장 실험이 가능하며, 실제 수중 환경에서 인간의 잠수 지구력(dive endurance)을 향상시키기 위해 설계되었습니다. 이 장비는 자연적인 킥(kick) 스타일에 적응하여 작동하며, 사용자가 스쿠버 장비를 착용한 상태에서도 쉽게 사용할 수 있습니다.

- **Technical Details**: DiveMate는 최대 30 Nm의 토크(torque)를 제공하며, 4.86 kg의 중량이 균형 잡힌 방식으로 허리에 위치하게 설계되었습니다. 이 장비의 주요 구성 요소는 양측 조인트 액추에이터(joint actuator), 허리 구조, 양측 다리 구조 및 배터리로 이루어져 있습니다. 또한, 브러시리스(statotor) 및 로터(rotor)를 포함하는 맞춤형 조인트 액추에이터는 수중에서도 작동할 수 있도록 방수 설계가 되어 있습니다.

- **Performance Highlights**: DiveMate는 수중에서 주어진 에너지로 이동 거리를 42.9% 증가시키고, 호흡 가스 소비율을 줄임으로써 잠수 시간을 54.9% 연장합니다. 또한, 근육 활성화(muscle activation)가 크게 감소하여 생리적 부담이 경감되며, 가스 소비율(net gas consumption rate)도 47.0% 감소합니다. 이러한 결과는 DiveMate가 잠수 지구력을 개선하고 인간의 수중 탐사 능력을 향상시키는 데 효과적임을 보여줍니다.



### SAFER-Nav: Enhancing Safety for Visual Robot Navigation via Segmentation-Aware Fine-Tuning (https://arxiv.org/abs/2606.11636)
- **What's New**: 이 연구에서는 SAFER-Nav라는 새로운 navigation model을 제안했습니다. 이 모델은 RGB 기반 백본을 통해 주어진 환경에서 장애물 경계를 명시적으로 나타내며 안전성을 향상시키기 위해 segmentation을 활용합니다. 기존의 방법들이 외부 모듈을 사용하는 데 반해, SAFER-Nav는 정책 자체에 안전 정보를 내재화하여 장애물 회피 성능을 개선합니다.

- **Technical Details**: SAFER-Nav는 RGB-goal encoder, segmentation encoder, representation-level fusion module, 그리고 이중 action prediction pathways로 구성됩니다. 이 모델은 RGB 관측 데이터와 목표 이미지뿐만 아니라, 장애물과 통과 가능한 공간을 구분하는 이진 segmentation mask를 활용합니다. 또한, 이 모델은 학습 과정에서 안전성 신호를 시맨틱 정보로 내재화하여, 정책이 장애물 인식 능력을 갖추도록 합니다.

- **Performance Highlights**: 여러 로봇 플랫폼과 다양한 장애물 시나리오에서 실험한 결과, SAFER-Nav는 ViNT 및 NoMaD 모델과 비교할 때 충돌 빈도를 현저히 감소시켰습니다. 이는 실내 환경에서도 안전성을 유지하면서 목표에 도달할 수 있는 성능을 보여주었습니다. 또한, SAFER-Nav는 조명과 시각적 표현이 다른 두 개의 실내 환경에서 모두 좋은 성능을 발휘했습니다.



### LUCID: Learning Embodiment-Agnostic Intent Models from Unstructured Human Videos for Scalable Dexterous Robot Skill Acquisition (https://arxiv.org/abs/2606.11628)
- **What's New**: 이번 논문에서는 구조화되지 않은 인간 비디오에서 로봇의 태스크 의도를 학습하는 LUCID라는 두 단계의 프레임워크를 제안합니다. 기존 로봇 학습 파이프라인은 로봇 시연이나 구조화된 인간 데이터를 기반으로 한 반면, LUCID는 인터넷 규모의 비디오 데이터에서 다양한 조작 시연을 활용합니다. 이를 통해 로봇 제어를 대규모 시뮬레이션에서 학습하며, 서로 다른 로봇 구현에도 동일한 의도 모델을 적용할 수 있습니다.

- **Technical Details**: LUCID는 두 가지 디자인 선택을 통해 의도(what should change in the scene)와 제어(how the robot achieves it)를 분리합니다. 첫째, 의도 모델은 비디오에서 예측한 짧은 각도에서의 의도 정보와 팔 자세(palm pose) 정보를 제공하며, 두 번째로 이러한 예측을 폐쇄 루프(closed loop) 시스템으로 연속적으로 갱신합니다. 결과적으로, 실행 시에 대형 비디오 데이터를 필요로 하지 않습니다.

- **Performance Highlights**: LUCID는 다섯 가지 실제 조작 태스크에서 평가되었으며, 웹 비디오로 감독된 작업에서 평균 73%의 성공률을 달성하였습니다. 기존 오픈 루프(open-loop) 기반 접근법에 비해 큰 개선을 보여주었고, 동일한 의도 모델로 다양한 로봇 구현에서 유사한 성공률을 보였습니다. 이러한 성과는 훈련된 비디오 데이터의 양에 따라 예측 가능하게 향상된다는 것을 보여줍니다.



### Distortion-Resilient Robotic Imitation Learning for Autonomous Cable Routing (https://arxiv.org/abs/2606.11577)
- **What's New**: 이번 연구에서는 이미지 왜곡이 있는 환경에서도 높은 성능을 유지할 수 있도록 설계된 새로운 로봇 모방 학습 프레임워크를 제안합니다. 제안된 프레임워크는 이미지 품질 평가 모듈과 신뢰 기반 학습 메커니즘을 포함하여 결정-making 모듈의 효과를 극대화합니다. 또한 MyRMB 데이터셋을 구축하여 로봇 조작 행동을 통해 이미지 품질을 정량화하고, 로봇 결정-making 시스템에 필수적인 사전 정보를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 이미지 품질 정보(IQA)를 추출하고, 신뢰 기반 학습 메커니즘을 통해 학습 효과를 높이는 구조로 되어 있습니다. IQA 모듈은 입력된 이미지 관찰에서 의사 결정 관련 시각 품질 정보를 추출하며, 신뢰 기반 학습 메커니즘은 훈련 어려움에 따라 샘플의 우선순위를 조정합니다. 이 방법은 특히 디지털 왜곡이 발생하는 상황에서 로봇의 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 이미지 왜곡이 있는 경우에도 의사 결정 모듈의 전반적인 성능을 향상시키는 것으로 나타났습니다. 로봇의 결정-making 시스템은 다양한 시각적 충실도 조건에서도 향상된 결정 정확도를 달성했습니다. 이러한 결과는 지능형 제어의 문제를 해결하는 새로운 방안을 제공하며, 복잡한 환경에서 로봇을 더욱 자율적이고 지능적으로 만들어줄 수 있습니다.



### ConsistencyPlanner: Real-time Planning with Fast-Sampling Consistency Models (https://arxiv.org/abs/2606.11569)
- **What's New**: 이번 논문에서는 상호작용이 있는 복잡한 실제 주행 시나리오를 위한 닫힌 루프 계획(closed-loop planning)의 새로운 접근법인 Consistency Planner를 제안합니다. 이는 다중 모드 행동( multimodal behavior) 표현과 실시간 계획(real-time planning)의 균형을 맞추기 위한 것으로, 빠른 샘플링(consistency models) 기법을 사용하여 다양한 예측 궤적들을 효율적으로 생성합니다. 또한, 주행 시나리오에서 복잡한 다양한 입력 특징들을 동적으로 통합하는 주의력 강화 디코더(attention-enhanced decoder)를 도입하여 계획의 견고성을 강화합니다.

- **Technical Details**: Consistency Planner는 전통적인 규칙 기반 방법과 학습 기반 방법의 장점을 결합한 프레임워크입니다. 이 시스템은 빠른 샘플링(consistency models)을 통해 실제 주행 데이터셋에 적합한 네트워크 아키텍처를 개발하고, 서로 다른 입력 특징(예: 도로 정보, 차량 상태)을 효율적으로 융합하여 최적의 행동 계획을 수립합니다. 이를 통해 모델은 다수의 가능성 있는 미래 경로를 신속하게 탐색할 수 있으며, 기존의 반복적 생성 방법에서 발생하는 계산 병목 현상을 극복합니다.

- **Performance Highlights**: Waymax 시뮬레이터에서의 실험 결과, Consistency Planner는 기존 학습 기반 기법들에 비해 안전성 메트릭에서 우수한 성능을 기록했습니다. 특히 동적인 시나리오에서 강력한 결과를 보여주었습니다. 이 연구에서 제안된 방법은 닫힌 루프 계획의 효율성을 극대화하며, 안전-critical 시스템으로서 자율 주행에 적합한 실시간 응답성을 유지합니다.



### Adversarial Attacks on Learned Policies for Surgical Robotic Tasks (https://arxiv.org/abs/2606.11535)
- **What's New**: 이 논문은 로봇 수술에서 학습 기반 정책을 겨냥한 적대적 공격의 첫 번째 연구를 제시합니다. 논문에서는 수술 기계 조작의 안전성에 대한 의문을 제기하며, 특히 우리가 사용하는 로봇 시스템이 치료 중 악의적 공격에 어떤 영향을 받을 수 있는지를 explores 합니다. 기존의 연구에서는 패치 공격(patch attack)이나 언어 기반 공격(language-based attacks)과 같은 방법이 사용되었지만, 본 연구는 비가시적 perturbations를 사용하여 적대적 공격을 수행합니다.

- **Technical Details**: 적대적 공격에는 두 가지 공격 모드가 포함되어 있습니다: (a) disruptive attacks는 미세한 시각적 변화가 정책 실행을 방해하고, (b) steering attacks는 그런 변화가 정책의 행동을 공격자가 설정한 방향으로 유도합니다. 연구에서는 세 가지 적대적 공격 방법과 다양한 정책 아키텍처(ACT, Diffusion Policy, π0)를 대상으로 합니다. 새로운 클래스의 photometric 적대적 공격도 소개되며, 이는 조명 변화와 같은 자연스러운 시각적 변화를 모방하여 현실적인 perturbations을 생성합니다.

- **Performance Highlights**: 560회의 물리 실험을 통해 최신 정책들이 이러한 공격에 크게 영향받을 수 있음을 증명하였습니다. 평균적으로 61%의 수술 하위 작업 성공률 감소가 관찰되었습니다. 특히 steering attacks는 관측당 밀리초 내에 생성될 수 있으며, 작은 조작을 위험한 대작업으로 확대할 수 있는 가능성을 보여줍니다.



### Learning Object Manipulation from Scratch via Contrastive Interaction (https://arxiv.org/abs/2606.11525)
- **What's New**: 본 논문은 Contrastive Reinforcement Learning (CRL)의 한계를 극복하기 위한 연구로, 상호작용 기반의 조작 문제 해결에 중점을 두고 있습니다. 특히, 객체 중심 상호작용이 동적 모드의 변화를 초래하여 CRL의 성능에 영향을 미친다는 점을 강조합니다. 이를 해결하기 위해, Interaction-weighted Resampling (IWR) 기법을 제안하며, 이 기법은 다양한 상호작용 환경에서 샘플 효율성을 높이고 성능을 개선하는 데 기여합니다.

- **Technical Details**: 조작 동역학을 조각별 매끄러운 마르코프 프로세스로 형식화하고, 상호작용이 유도하는 모드 변화를 통해 다중 모드 및 조각별 비선형 도달 가능성을 포착하는 방법에 대한 분석을 제공합니다. IWR은 상호작용에 대한 인식을 바탕으로 샘플링을 수행하여 훈련 분포를 조정하고, 상호작용에 따른 모드 경계를 보존하도록 장려합니다. 이는 기존의 CRL 에너지 함수로는 표현하기 힘든 복잡한 동역학 구조를 더 잘 모델링할 수 있게 합니다.

- **Performance Highlights**: IWR 기법은 2D 동적 제어, 로봇 조작 및 로봇 에어 하키와 같은 상호작용 중심 환경에서 이전 CRL 방법에 비해 평균 19.8%의 성능 향상을 가져왔습니다. 또한 IWR로 훈련된 정책을 사용하여 최초로 실제 환경에서 목표 조건 로봇 에어 하키 에이전트를 구현하였으며, 이는 성공률을 25%에서 60%로 개선하는 성과를 달성했습니다.



### Steering Multirobot Behavior via Closed-Loop Affine Activation Editing (https://arxiv.org/abs/2606.11489)
- **What's New**: 본 논문은 CLAE(Closed-Loop Affine Activation Editing)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 로봇의 학습된 정책을 수정하지 않고도 중간 활성화를 편집하여 동작을 조정할 수 있게 해줍니다. CLAE는 로봇의 상태, 환경, 목표 동작 및 다중 로봇 컨텍스트에 따라 온라인으로 정책을 수정할 수 있는 닫힌 루프 문제로 접근합니다.

- **Technical Details**: CLAE는 고정된 로봇 정책의 중간 활성화를 편집하기 위해 희소 오토인코더(Sparse Autoencoder)와 경량 강화 학습 기반의 조정 정책을 사용합니다. 각 정책 단계에서, CLAE는 중간 활성화를 입력받아 행동 관련 활성화를 선택하고, 이에 대해 아핀 편집(affine edits)을 적용합니다. 이 과정에서 기본 정책의 가중치나 구조를 수정하지 않고도 다양한 행동을 생성할 수 있습니다.

- **Performance Highlights**: CLAE는 멀티 쿼드로터(navigation policy)를 대상으로 한 세 가지 행동 조정 작업에서 평가되었습니다. 이 실험을 통해 CLAE는 로봇의 속도 프로파일 조정, 다중 로봇 형성 제어 및 감시 카메라 회피 등 새로운 행동을 생성하는 데 성공적으로 작동함을 입증했습니다. 이러한 결과는 다양한 로봇 환경에서 성능 저하 없이 동작을 조정할 수 있는 가능성을 보여줍니다.



### Bridging the sim2real gap in the table tennis robot with a transformer-based ball states predictor (https://arxiv.org/abs/2606.11464)
- **What's New**: 본 논문에서는 테이블 테니스 공 상태 예측을 위한 트랜스포머 기반 프레임워크를 제안합니다. Attention (어텐션) 메커니즘을 활용하여 과거 관측에서 직접적으로 장기적인 시간적 상관관계를 모델링하며, 분석적인 비행 또는 튕김 모델에 의존하지 않습니다. 이를 통해 물리 기반 및 학습 기반의 기존 방법들보다 더욱 정확하고 효율적인 예측을 가능하게 합니다.

- **Technical Details**: 제안된 예측기는 인코더-디코더 구조를 사용하며, 멀티 헤드 어텐션과 확장-병목 컨볼루션 블록을 포함하고 있습니다. 입력으로는 과거의 공 위치와 스핀을 받아서 미래의 상태(위치, 속도 및 스핀)를 예측합니다. 또한, 100,000개 이상의 다양한 기술 수준의 플레이어와 볼 캐논 구성의 실제 데이터셋을 수집하여 강한 일반화를 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 예측기는 물리 기반 및 기계 학습 기반의 기존 기준보다 특히 장기 예측에서 우수한 성능을 보였습니다. 또한, SPAD(Swap Predictor at Deployment)라는 간단한 시뮬레이션-리얼 전이 전략을 도입하여 훈련된 정책을 별도의 재훈련 없이 실제 환경에 적용할 수 있음을 보여주었습니다. 이러한 접근법은 시뮬레이션 기반 훈련의 효율성을 유지하면서도 실제 환경에서의 성능을 향상시킵니다.



### A Modular Dual-Camera Pipeline for Micro-Inspection Using Aerial Robots (https://arxiv.org/abs/2606.11419)
- **What's New**: 이번 논문에서는 드론 기반의 미세 검사(micro-inspection) 시스템을 다루고 있습니다. 기존 시스템이 드론과 목표물 간의 근접 비행을 요구하거나 복잡한 비행 경로를 따르는 것에 비해, 본 연구는 PX4 기반 드론과 이중 카메라 시스템을 활용한 새로운 접근 방식을 제시하고 있습니다. 이를 통해 보다 안전하게 고해상도 이미징을 수행할 수 있으며, 비 구조물(target) 검사도 가능하게 합니다.

- **Technical Details**: 제안된 시스템은 넓은 시야각의 스테레오 내비게이션 카메라와 세부 검사를 위한 줌이 가능한 짐벌 장착 카메라로 구성됩니다. 내부 알고리즘은 검사 중 드론 이동을 보상하며, 실시간으로 표면 기하학을 추정합니다. 주요 기여 중 하나는 고정밀 검사 부위를 구분하는 파라메트릭 스위핑 전략과 시각 기반 피드백 루프를 통한 안정성입니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험 결과, 제안된 파이프라인은 드론 방해 요소에서의 견고한 커버리지를 보여주었으며, 오크 행렬 및 알 유충 탐지에서 효과적인 성능을 나타냈습니다. 특히, 도서관 검사와 온실 검사에서 높은 세부 이미지 결과를 달성하였으며, 이 모든 과정은 오픈 소스로 제공되므로 다양한 응용 프로그램에 적응할 수 있습니다.



### Dynamic Execution Horizon Prediction for Chunk-based Robot Policies (https://arxiv.org/abs/2606.11408)
- **What's New**: 이번 연구에서는 로봇 정책에서 고정된 실행 지평선(fixed execution horizon)의 필요성을 극복하기 위한 새로운 방법, 동적 실행 지평선 예측(Dynamic Execution Horizon Prediction, DEHP)을 제안합니다. DEHP는 온라인 강화 학습(online reinforcement learning)을 통해 경량의 실행 지평선 예측 모듈을 훈련시킵니다. 이 방식은 기존의 정책 그대로 유지하면서, 로봇이 작업의 각 단계에 따라 적절한 지평선을 조정할 수 있도록 도와줍니다.

- **Technical Details**: DEHP는 반마르코프 결정 과정(semi-Markov decision process)으로 동적 실행 지평선 예측을 공식화하고, 희소 이진 보상(sparse binary rewards)을 사용하여 지평선 예측기를 최적화합니다. 이 방법은 사전 훈련된 청크 정책을 수정하지 않고도 다양한 로봇 작업에 적용될 수 있으며, 정책의 깊은 변경 없이 단 하나의 예측 통과(forward pass)만을 추가합니다. DEHP는 또한 단계별로 서로 다른 반응성을 요구하는 작업에 따라 실행 지평선을 조정하는 방법을 학습합니다.

- **Performance Highlights**: DEHP는 여러 고정 지평선 기준선에 비해 조립 및 정밀 삽입 작업에서 일관되게 성공률을 높였습니다. 다양한 작업 단계 동안 DEHP는 짧은 실행 지평선을 예측하여 더 높은 반응성을 보이며, 여유 공간에서의 동작에는 긴 지평선을 적용하여 효율성을 극대화합니다. 이 과정에서 DEHP는 제어 안정성을 향상시키고, 작업 진행 상황에 맞게 실행 지평선을 효과적으로 조정하는 능력을 배양합니다.



### PLUME: Probabilistic Latent Unified World Modeling and Parameter Estimation for Multi-Finger Manipulation (https://arxiv.org/abs/2606.11396)
Comments:
          16 pages, 5 figures

- **What's New**: 이번 연구에서는 Probabilistic Latent Unified world Modeling and parameter Estimation (PLUME)이라는 새로운 세계 모델을 제안합니다. PLUME는 물리적 매개변수에 대한 신념을 진화시키고, 이러한 매개변수에 조건화하여 시스템 역학을 학습할 수 있도록 설계되었습니다. 이는 기존의 도메인 무작위화 기법들보다 더 높은 정밀도가 요구되는 조작 과제에 적합합니다.

- **Technical Details**: PLUME는 다양한 물리적 조건에서 생성된 데이터셋을 통해 세계 모델을 학습합니다. 이를 위해, PLUME는 관찰 및 행동에 따라 잠재 매개변수에 대한 신념을 추정하며, 이러한 신념을 바탕으로 세계 모델링이 이루어집니다. 특히, flow matching 기법을 사용하여 다양한 물리적 매개변수를 통합적으로 학습하는 고유한 잠재 표현을 개발했습니다.

- **Performance Highlights**: 제안된 PLUME 방법의 성능은 시뮬레이션된 드라이버 불량 조작, 밸브 조작, 버킷 들어올리기 및 디스크 플리킹 등의 여러 작업에서 평가되었습니다. PLUME는 최신 오프라인 강화 학습 및 세계 모델 기반 행동 복제 기법보다 우수한 성과를 보였으며, 하드웨어 드라이버 조작 작업에서도 제로샷 전이를 성공적으로 수행하였습니다.



### HiPi: Reproducible High-Fidelity Piezoresistive Sensors for Robotic Manipulation (https://arxiv.org/abs/2606.11372)
- **What's New**: 이번 연구에서는 로봇 조작을 위한 재현 가능한 고충실도(piezoresistive) 압력 센싱 시스템인 HiPi를 제안합니다. HiPi는 복잡한 하드웨어 구조를 단순화하고, 상업적 PCB 제작이 가능하도록 설계하였습니다. 이로 인해 기존의 재현 가능한 설계와 고충실도 읽기 아키텍처 사이의 간극을 메우는 데 기여하고 있습니다.

- **Technical Details**: HiPi 시스템은 저크로스타크(low-crosstalk) 읽기 원리를 기반으로 하며, 통합된 하드웨어 구조를 통해 재현성과 배치 용이성을 주 목표로 하고 있습니다. 이 시스템은 수동 납땜을 배제한 컴팩트한 읽기 PCB와 STM32 기반의 저비용 MCU 모듈을 포함하고 있습니다. 전체 작동 주파수는 네 개의 고밀도(tactile arrays) 배열을 이용하여 220Hz를 달성하고 있습니다.

- **Performance Highlights**: HiPi는 실험을 통해 기존 시스템보다 압접 면적의 기하학적 보존이 훨씬 뛰어남을 보여주며, 평균 IoU는 0.428에서 0.797로, 평균 Dice 점수는 0.539에서 0.886으로 개선되었습니다. 이러한 성과는 HiPi가 다수의 압력 센서를 사용하는 로봇 시스템에 있어 실용적이고 신뢰할 수 있는 솔루션이 될 것임을 보여줍니다.



### Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models (https://arxiv.org/abs/2606.11324)
Comments:
          Embodied R1.5 technical report. Project page: this https URL

- **What's New**: Embodied-R1.5는 복합적 의식(reasoning) 능력을 통합한 통합형 Embodied Foundation Model (EFM)으로, 15B 이상의 토큰을 포함한 대규모 데이터 시스템을 통해 일반적인 물리적 지능을 목표로 설계되었습니다. 이 모델은 Planner-Grounder-Corrector (PGC) 폐쇄형 프레임워크를 통해 장기적인 작업을 자율적으로 실행하고 수정할 수 있으며, 이를 통해 동작 및 오류 수정의 자동화를 가능하게 합니다. Embodied-R1.5는 8B의 파라미터만으로 24개의 embodied VLM 벤치마크 중 16개에서 SOTA(State Of The Art)를 달성했습니다.

- **Technical Details**: EFM 모델은 한 가지 아키텍처 내에서 지각(perception), 추론(reasoning), 실행(execution)의 모든 기능을 통합하여, 점진적인 추론 체인을 형성합니다. 모델 아키텍처의 세 가지 주된 차원은 공간 인지(cognition), 작업 계획(planning), 수정(correction)으로 구성되며, 이는 약속된 정보 흐름을 통해 외부 통신 없이도 원활한 작업을 지원합니다. 이를 위해 세 가지 자동화된 데이터 생성 파이프라인을 사용하여 15B 이상의 토큰을 포함한 대규모 데이터 코퍼스를 구축했습니다.

- **Performance Highlights**: Embodied-R1.5는 SOTA 벤치마크에서 평균 70.4%의 정확도를 달성하며, Gemini-Robotics-ER-1.5 및 GPT-5.4를 각각 17.0% 및 21.7% 초과합니다. 특정 테스트 벤치마크에서 92.4%의 성능을 보여주며, 이는 π0.5 및 ManipLLM과 같은 강력한 VLA 모델을 능가합니다. 또한, 제로샷(real-robot) 실험을 통해 다양한 실제 작업 수행에서 강력한 일반화를 보여주고 있습니다.



### Model-based Optimization of Anguilliform Swimming Gaits for Soft Robotic Applications (https://arxiv.org/abs/2606.11278)
- **What's New**: 이번 논문에서는 부드러운 램프레이에서 영감을 받은 이중 환경 로봇(SLIDER)의 설계 및 최적화 절차를 소개합니다. Lighthill의 이론을 기반으로 액체 환경에서의 유체-구조 상호작용을 구현하며, 비선형 모델을 사용하여 로봇의 구조적 설계 매개변수를 신속하게 개발합니다. 실험과 컴퓨터 시뮬레이션 결과의 비교를 통해, 저주파수 수영은 저항 환경 힘에 의해 지배되고 고주파수 수영은 관성 유체 힘의 영향을 받음을 발견했습니다.

- **Technical Details**: SLIDER의 설계 최적화를 위해, 중간 난이도의 비선형 유한 요소 모델을 개발하여 유체 역학과 구조 물리를 결합하였습니다. 이 모델은 Euler-Bernoulli 빔 이론을 기반으로 하며, SLIDER의 대한 운동을 설명하기 위해 Euler-Lagrangian 이중 좌표계를 사용합니다. 유선형 비디오 및 물리적 실험을 통해 매개변수를 검증하고, 이러한 발전된 모델을 통해 실시간 설계 및 제어 최적화를 지원합니다.

- **Performance Highlights**: 슬라이더는 21.7 +/- 0.4 cm/s의 속도로 수영할 수 있으며, 이는 로봇의 climbing 설정에 맞춘 지식적 제어 패턴과 caudal fin 디자인의 공동 최적화를 통해 달성되었습니다. 구조가 climbing에 최적화된 여러 부분에도 불구하고, SLIDER는 뛰어난 수영 성능을 발휘하여 다중 모달(멀티모달) 로봇 설계에서 조정의 중요성을 보여줍니다. 이 연구는 서로 다른 환경에서 로봇의 형태적 설계를 최적화하는 데 효과적인 새로운 접근 방식을 제시합니다.



### MASK: Multi-Agent Semantic K-Scheduling for Risk-Sensitive 6G Robotics (https://arxiv.org/abs/2606.11249)
- **What's New**: 6G 연결 로봇의 비전을 실현하려면 고성능 협력 제어와 물리적 무선 채널의 엄격한 스펙트럼 제한을 조화시켜야 합니다. 이를 위해 Multi-Agent Semantic K-Scheduling (MASK)라는 새로운 제어 아키텍처를 제안하며, Arbiter-Assisted Semantic Information Gating (A-SIG) 메커니즘을 통해 로컬로 계산된 의미론적 중요도 점수를 기반으로 전송 에이전트를 엄선합니다. 이 접근 방식은 자원의 제약이 있는 6G 시스템에서의 안정적인 조정을 지원합니다.

- **Technical Details**: MASK는 여러 에이전트가 제한된 물리적 자원 블록을 공유하는 네트워크 환경에서 협력적 멀티 에이전트 작업을 간소화하기 위한 Decentralized Partially Observable Markov Decision Process (Dec-POMDP) 모델로 구성됩니다. 각 에이전트는 로컬 데이터와 다른 에이전트의 메시지를 받아서 환경 행동과 통신 행동을 선택합니다. A-SIG 모듈을 통해 중앙 조정자가 각 에이전트의 중요도를 평가하여 채널 접근을 조율합니다.

- **Performance Highlights**: MASK는 대규모 에이전트 집합에서 채널 접근이 제한되더라도 기존의 통신 비제한 기준과 같은 성능을 보여줍니다. 특히, 이 프레임워크는 임의의 패킷 삭제에 대한 내성을 보이며, 실제 6G 로봇 네트워크의 불안정한 특성에서도 우수한 성능을 유지합니다. MASK의 시스템은 데이터 효율성을 최적화하면서도 안전하게 작동할 수 있는 구조를 갖추고 있습니다.



### Fast-SDE: Efficient Single-Microphone Sound Source Distance Estimation in Reverberant Environments (https://arxiv.org/abs/2606.12339)
Comments:
          To appear in the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)

- **What's New**: 이번 논문에서는 로봇 플랫폼에서의 적용을 위해 경량의 단일 마이크로폰 기반 음원 거리 추정(Sound Source Distance Estimation, SDE) 프레임워크인 Fast-SDE를 제안합니다. 기존의 다중 마이크로폰 시스템이 지닌 하드웨어 동기화 및 계산 자원 요구의 문제를 극복하면서, Fast-SDE는 서브밴드(subband) 기반의 구조를 통해 주파수 축을 여러 개의 서브밴드로 분해하여 거리 추정을 수행합니다.

- **Technical Details**: Fast-SDE는 서브밴드 인코더를 이용하여 각 서브밴드를 압축된 잠재 표현(latent representation)으로 매핑하고, 저렴한 계산 비용으로 음향 구조와 시-주파수(time-frequency) 패턴 간의 관계를 학습합니다. 이 접근 방식을 통해 모델의 복잡성을 줄이고 추론 지연(inference latency)을 최소화하며, 실시간 응용 프로그램에 적합한 정확성과 효율성을 달성합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 환경에서의 실험 결과, Fast-SDE는 기존 방법들에 비해 경쟁력 있는 거리 추정 정확도를 제공하며, 필요한 파라미터 수와 실행 시간 비용이 적습니다. 이로 인해 Fast-SDE는 메모리와 컴퓨터 자원이 제한된 로봇 플랫폼에서의 사용에 적합합니다.



### Fourier Features Let Agents Learn High Precision Policies with Imitation Learning (https://arxiv.org/abs/2606.12334)
Comments:
          Published as a conference paper at ICML 2026

- **What's New**: 이 논문은 로봇 조작의 높은 정밀도를 위해 점 구름(point clouds)에서 고차원 푸리에 공간(Fourier space)으로 변환하는 새로운 접근 방식을 제안합니다. 이를 통해 정책은 고주파 요소(high-frequency features)에 직접 접근함으로써 공간 추론(spatial reasoning)을 개선할 수 있습니다. 실험 결과, 푸리에 특성(Fourier features)은 다양한 인코더 아키텍처와 벤치마크에서 상당한 성능 향상을 보여주었습니다.

- **Technical Details**: 로봇 비주얼 모터 컨트롤에서 고주파 정보의 중요성을 강조하는 이 논문은 다중 모달 컬렉션(multi-modal action distributions)을 다루기 위해 확산 기반 모방 학습(Diffusion-based Imitation Learning) 프레임워크를 사용합니다. 3차원 표현(3D representations)은 복잡한 동작을 수행하는 데 도움을 주지만, 기존의 3D 모달리티는 작업에 따라 성능 편차가 큽니다. 이 연구는 포인트 클라우드 아키텍처에서 스펙트럼 편향(spectral bias)을 보완하기 위해 푸리에 특성을 효과적으로 포함시킵니다.

- **Performance Highlights**: 푸리에 특성을 사용함으로써 RoboCasa와 ManiSkill3 벤치마크에서 각각 20%와 7%의 성공률 향상 효과를 검증했습니다. 또한, 44개의 실제 세계 작업에서 정규화된 점수를 14.8%에서 40.2%로 증가시켰습니다. 연구 결과는 푸리에 매핑을 활용하여 세밀한 조작이 요구되는 로봇 제어 작업에서 매끄럽고 정밀한 동작을 나타냅니다.



### Energy-Conserved Neural Pipelines: Attenuating Error Propagation in Modular Neural Networks via Physical Conservation Constraints (https://arxiv.org/abs/2606.11341)
Comments:
          22 pages, 2 figures, 7 tables, 25 references

- **What's New**: 이 논문은 모듈형 신경망 아키텍처에서 발생하는 오류 전파 문제를 해결하기 위해 에너지 보존을 적용하는 새로운 접근 방식을 제안합니다. 에너지 보존은 각 모듈 경계에서 활성화 에너지가 정확히 보존되도록 강제하며, 이는 기존의 소프트 에너지 패널티와 다르게 절대적인 물리 법칙으로 작용합니다. 이를 통해 신경망은 유연하게 뉴런 간 에너지를 재분배하되, 에너지를 생성하거나 파괴할 수 없습니다.

- **Technical Details**: 저자들은 신경망 계층에 의해 생성된 활성화 벡터의 에너지를 일반적으로 물리학 및 신호 처리에서 사용하는 정의에 따라 정의합니다. 모듈 간 정보 흐름을 제어하기 위해 에너지 보존 연산자를 도입하여, 각 모듈의 출력 활성화 에너지가 입력 활성화 에너지와 같도록 보존합니다. 이 과정은 정량적 안정성을 위해 수치적 조정을 포함하며, 에너지 관리 운영자인 ℬ​(𝐱~,E0)가 입력 노이즈 에너지보다 항상 보존된 노이즈 에너지가 적음을 증명합니다.

- **Performance Highlights**: CIFAR-10을 대상으로 한 실험에서, 에너지 보존이 도입된 네트워크는 노이즈가 있는 환경에서도 높은 정확도를 유지할 수 있는 것을 보여주었습니다. 특히, 에너지 보존을 적용한 경우, 노이즈가 σ=0.2일 때 77.4%의 정확도를 유지했으며, 기존 모델에 비해 현저한 성능 향상을 나타냈습니다. 또한, 심층 모듈 파이프라인에서는 동일한 성능을 유지하면서도 에너지 보존의 이점이 적용되어 노이즈의 영향을 최소화했습니다.



New uploads on arXiv(cs.MA)

### Phi-Actor-Critic: Steering General-Sum Games to Pareto-Efficient Correlated Equilibria (https://arxiv.org/abs/2606.11284)
Comments:
          Accepted to IJCAI 2026

- **What's New**: 본 논문에서는 일반합 게임(general-sum games)에서 사회적으로 효율적인 결과를 달성하기 위한 새로운 접근법인 Φ-Actor-Critic (Φ-AC)를 제안합니다. 기존의 deep multi-agent reinforcement learning (MARL) 방법들은 Nash equilibrium(NE) 도달에 어려움을 겪고 있으며, Φ-AC는 swap regret 최소화를 이용하여 효율적인 상관 균형(correlated equilibria)을 학습하는 데 중점을 둡니다. 이를 통해 다양한 혼합 동기 설정에서 높은 사회적 복지를 달성할 수 있는 획기적인 방법론을 제공하고 있습니다.

- **Technical Details**: Φ-AC 프레임워크는 중앙 집중형 비평가(centralized attention critic)를 활용하여 벡터 값을 가진 regret를 단일 전방통과(forward pass) 방식으로 예측하게 합니다. 이러한 접근은 계산 비용이 높은 counterfactual 시뮬레이션을 피할 수 있게 하여, 고차원 환경에서도 효율적인 regret 추정을 가능하게 합니다. 또한, Lagrangian 기반의 균형 선택 메커니즘을 도입하여 사회적 복지를 최적화하면서도 안정성을 유지하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과는 Φ-AC가 matrix 게임, Multi-Agent Particle Environments (MPE), 및 Melting Pot Harvest 시나리오에서 높은 사회적 복지와 경쟁의 공정성을 유지하면서도 효율적이고 안정적인 조정 전략을 학습함을 보여줍니다. 특히, Φ-AC는 기존 MARL 방법들과 비교하여 더 나은 지속 가능성과 공정성을 달성하며, 복잡한 Sequential Social Dilemmas (SSDs)에서도 강력한 성능을 발휘합니다.



### Multi-agent rendezvous in fluid flows via reinforcement learning (https://arxiv.org/abs/2606.11274)
- **What's New**: 본 연구에서는 다중 에이전트 시스템에서의 rendezvous 문제를 다루고 있습니다. 특히, 유체 환경에서의 rendezvous 전략을 물리학적으로 기반으로 한 다중 에이전트 강화 학습(MARL) 접근법을 통해 개발하였습니다. MARL 전략은 비대칭 상태-행동 맵을 활용하여 개별 에이전트들이 분리된 소용돌이에 갇히는 것을 방지하며, 성공적인 rendezvous율을 크게 향상시킵니다.

- **Technical Details**: 에이전트들은 점질량(point-mass)으로 모델링되어 서로의 충돌을 무시할 수 있으며, 모든 에이전트의 전역 관찰이 가능합니다. 다중 에이전트 강화 학습(MARL)을 통해 각 에이전트는 주변 유체 신호를 활용하여 프로펠링 방향을 조정하며, Deep Set Proximal Policy Optimization(DS-PPO) 알고리즘을 사용하여 rendezvous 문제를 탐구합니다. 연구에서는 제어성과 관련된 JJ 기준(QQ-criterion)과 함께 유체 동역학의 복잡성을 반영하였습니다.

- **Performance Highlights**: MARL 전략은 약해진 소용돌이 지역에서 예상되는 rendezvous 성능을 높이고, 전통적인 나이브 전략에 비해 우수한 결과를 보입니다. 또한, 내장된 휴리스틱 전략은 나이브 전략을 초월하는 성능을 보여주어나, 유체의 변형이 rendezvous 프로세스를 방해한다는 점은 이론적으로도 입증되었습니다. 연구 결과는 유체-에이전트 상호작용의 중요성을 강조하며 복잡한 유동 환경에서 MARL의 가능성을 보여줍니다.



### Evaluation of Alternative-Based Information Systems for Deliberative Polling using an Agentic Simulator (https://arxiv.org/abs/2606.11692)
- **What's New**: 이번 논문에서는 합리적인 집단 의사결정을 위한 방안을 제시하고 있습니다. 대표적인 의견 표본을 보장하기 위한 커버리지 문제를 해결하기 위해 LLM 기반의 Agentic Bipolar Argumentation Simulator (ABAS)를 도입하였습니다. 이 시뮬레이터는 자율적으로 행동하는 주주 대리인을 N명 시뮬레이션하여, 각각의 대리인이 어떤 자격을 갖고 투표하는지를 측정합니다. 이를 통해 추천 시스템의 성공을 평가하는 기준으로 커버리지를 사용하고 있습니다.

- **Technical Details**: ABAS는 엄격한 수학적 프레임워크에 기초하여 주주들의 투표와 주장을 시뮬레이션합니다. 논문에서 제안된 문제는 NP-hard인 Subsuming Justification Problem (SJP)이며, 이 시스템은 선호도 기반 추천 시스템을 구현하여 투표자들에게 제공되는 추천들의 커버리지를 극대화하는 방법을 모색합니다. 유사한 의견과 상반된 의견을 탐색하면서도 주주들 사이의 관계를 평가하는 그래프 구조를 적용하였습니다.

- **Performance Highlights**: 실험 결과에서는 창의성 비율(pown), 추천 크기(K), 주장의 밀도(plinks), 인구 크기(N)가 커버리지와 적극적인 다양성에 미치는 영향을 분석했습니다. 독립된 주주들이 참여하는 상황에서, 조정된 전략적 투표 공격은 커버리지를 감소시키지만, 저자 수에 기반한 관계 가중치는 일반적인 가중치보다 훨씬 더 효과적으로 이러한 홍수 공격에 저항할 수 있음을 보여주었습니다.



### Sovereign Assurance Boundary: Certificate-Bound Admission for Agentic Infrastructur (https://arxiv.org/abs/2606.11632)
Comments:
          12 pages, 1 figure, 13 tables

- **What's New**: 이 논문은 Sovereign Assurance Boundary (SAB)라는 새로운 인증 기반의 런타임 입장 레이어를 도입합니다. 이는 고위험 프로덕션 리소스에 대한 비결정적(reasoning) 시스템의 돌연변이를 제안할 수 있는 문제를 해결합니다. 기존의 접근 방식들이 정적이고 맥락 인식이 부족한 권한을 강제하거나, 실행 후 행동만 기록하는 것에 그치는 반면, SAB는 제안된 행동이 실행될 수 있는지 결정하는 새로운 메커니즘을 제공합니다.

- **Technical Details**: SAB는 에이전트 프로포절을 보증 공기 잠금장치에서 가로채고, 이들을 타입화된 실행 계약으로 정리하여, 암호화된 증거 다이제스트와 정책 버전에 바인딩합니다. 이는 제안된 행동이 특정 실행 권한, 철회 시점 및 유효성 창에 국한된 서명된 Sovereign Assurance Certificate를 발급받도록 합니다. SAB는 개별 행위와 시스템 수준의 보증을 분리하여, 직접적인 주 상태 변경을 방지합니다.

- **Performance Highlights**: 초기 Go 프로토타입을 개발하여 2,500회의 입장 시도를 평가했고, 입장 대기 시간, 브로커 검증, 철회 전파, 루팅 정확성, 재생 완전성 및 증명 오버헤드에 대한 초기 측정치를 보고했습니다. 이 연구는 에이전트 시스템에 대한 안전한 인증 메커니즘을 강화하는 방향으로 큰 기여를 할 것으로 기대됩니다.



### MASK: Multi-Agent Semantic K-Scheduling for Risk-Sensitive 6G Robotics (https://arxiv.org/abs/2606.11249)
- **What's New**: 6G 연결 로봇의 비전을 실현하려면 고성능 협력 제어와 물리적 무선 채널의 엄격한 스펙트럼 제한을 조화시켜야 합니다. 이를 위해 Multi-Agent Semantic K-Scheduling (MASK)라는 새로운 제어 아키텍처를 제안하며, Arbiter-Assisted Semantic Information Gating (A-SIG) 메커니즘을 통해 로컬로 계산된 의미론적 중요도 점수를 기반으로 전송 에이전트를 엄선합니다. 이 접근 방식은 자원의 제약이 있는 6G 시스템에서의 안정적인 조정을 지원합니다.

- **Technical Details**: MASK는 여러 에이전트가 제한된 물리적 자원 블록을 공유하는 네트워크 환경에서 협력적 멀티 에이전트 작업을 간소화하기 위한 Decentralized Partially Observable Markov Decision Process (Dec-POMDP) 모델로 구성됩니다. 각 에이전트는 로컬 데이터와 다른 에이전트의 메시지를 받아서 환경 행동과 통신 행동을 선택합니다. A-SIG 모듈을 통해 중앙 조정자가 각 에이전트의 중요도를 평가하여 채널 접근을 조율합니다.

- **Performance Highlights**: MASK는 대규모 에이전트 집합에서 채널 접근이 제한되더라도 기존의 통신 비제한 기준과 같은 성능을 보여줍니다. 특히, 이 프레임워크는 임의의 패킷 삭제에 대한 내성을 보이며, 실제 6G 로봇 네트워크의 불안정한 특성에서도 우수한 성능을 유지합니다. MASK의 시스템은 데이터 효율성을 최적화하면서도 안전하게 작동할 수 있는 구조를 갖추고 있습니다.




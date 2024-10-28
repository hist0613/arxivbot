New uploads on arXiv(cs.CL)

### Rethinking Visual Dependency in Long-Context Reasoning for Large Vision-Language Models (https://arxiv.org/abs/2410.19732)
- **What's New**: 이 연구에서는 LVLMs(Long Vision-Language Models)가 긴 맥락에서의 추론에서 성능 저하를 겪는 원인을 분석하고, 비주얼 의존성을 높이는 새로운 텍스트 정보 선택적 제거 방법을 제안했습니다. 이로 인해 LVLMs의 긴 맥락에서의 성능이 향상됩니다.

- **Technical Details**: LVLMs는 비전 인코더와 LLM(Long Language Model)을 포함하여 이미지와 텍스트를 교차 모달(inter-modal) 상호작용을 통해 처리합니다. 연구에서는 훈련 없이 긴 맥락에서 덜 중요한 텍스트 정보를 제거하는 'context pruning' 방법을 통해 비주얼 의존성을 증가시켰습니다.

- **Performance Highlights**: 제안된 방법은 SVIT 데이터셋을 기반으로 한 긴 맥락 데이터셋에서 여러 LVLMs에 대해 검증되었으며, 실험 결과 다양한 토큰 프루닝(token pruning) 전략의 강건성도 입증되었습니다.



### Counting Ability of Large Language Models and Impact of Tokenization (https://arxiv.org/abs/2410.19730)
- **What's New**: 이 연구는 Large Language Models (LLMs)에서 토큰화(tokenization)가 카운팅(counting) 능력에 미치는 영향을 조명하고 있습니다. 기존 연구에서는 Transformer 구조의 한계를 이야기했으나, 토큰화 과정이 이들 모델의 이론적 계산 능력에 미치는 영향은 상대적으로 적게 다뤄졌던 문제입니다.

- **Technical Details**: 저자들은 토큰화 선택이 LLM의 이론적 카운팅 능력을 크게 저하시킬 수 있음을 입증하기 위해 모델 중립적인 접근법을 사용했습니다. 이를 통해 Chain of Thought (CoT) 방식으로 수행된 카운팅 실험을 진행하였으며, 적절한 토큰화 선택이 모델의 이론적 카운팅 능력을 최대로 드러내는데 필수적이라는 점을 강조하고 있습니다.

- **Performance Highlights**: 토큰화의 선택에 따라 최대 80%까지 정확도가 떨어질 수 있으며, 다양한 모델 간에도 토큰화가 카운팅 작업에 미치는 영향이 다르게 나타났습니다. 연구 결과는 LLM의 카운팅 능력을 향상시키기 위한 새로운 토큰화 방법 설계에 영감을 줄 수 있을 것으로 보입니다.



### 2D-DPO: Scaling Direct Preference Optimization with 2-Dimensional Supervision (https://arxiv.org/abs/2410.19720)
Comments:
          The first four authors contributed equally, 25 pages

- **What's New**: Direct Preference Optimization (DPO)의 발전을 통해 대규모 언어 모델(LLMs)과 인간의 선호를 보다 정교하게 정렬할 수 있는 방법이 제안되었습니다. 새로운 2차원 감시 데이터셋인 HelpSteer-2D를 도입하여 응답을 세그먼트와 측면으로 나누고 이 두 차원을 기반으로 한 DPO(2D-DPO)를 소개합니다.

- **Technical Details**: 2D-DPO는 응답을 문장으로 나누고 각 세그먼트에 점수를 할당하여 다차원적인 인간의 선호를 포착합니다. 이를 위해 2차원 점수 행렬을 사용하여 각 세그먼트의 여러 측면을 평가합니다. 2D-DPO는 총 목표를 다중 세그먼트 및 다중 측면 목표로 분해하여 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과, 2D-DPO는 스칼라 또는 1차원 선호를 최적화하는 기존 방법보다 더 나은 성능을 보였습니다. 이는 모델 훈련에서 다차원적인 피드백의 중요성을 잘 반영하고 있습니다.



### Less is More: Extreme Gradient Boost Rank-1 Adaption for Efficient Finetuning of LLMs (https://arxiv.org/abs/2410.19694)
Comments:
          19 pages

- **What's New**: 본 논문에서는 eXtreme Gradient Boosting LoRA (XGBLoRA)라는 혁신적인 프레임워크를 제안합니다. XGBLoRA는 앙상블 학습의 힘을 활용하여 Low-Rank Adaptation (LoRA)의 이론적 최적성과 실제 성능 간의 격차를 해소하는 방법을 제시합니다.

- **Technical Details**: XGBLoRA는 Gradient Boosting의 원리를 바탕으로 연속적인 LoRA 적응을 학습하고 이를 결합하여 모델 예측을 개선합니다. 이 프레임워크는 적응 행렬의 랭크가 낮아도 성능 저하 없이 높은 예측 품질을 유지합니다. XGBLoRA의 이론적 분석을 통해 수렴 보장과 예측 경계가 확립되었으며, 낮은 랭크를 유지하면서도 양질의 성능을 얻을 수 있는 방법이 설명되었습니다.

- **Performance Highlights**: 실험 결과, XGBLoRA는 기존의 LoRA 및 전체 파인튜닝에 비해 더 나은 성능을 지속적으로 보여주었습니다. XGBLoRA는 더 적은 수의 훈련 가능한 매개변수를 사용하면서도 상위 성능을 달성, 예를 들어 LLaMA3-8B 모델에서 NVIDIA RTX 4090에서 실행이 가능했으며, LoRA는 A100 GPU를 필요로 했습니다.



### AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions for Conversational Search with LLMs (https://arxiv.org/abs/2410.19692)
Comments:
          23 pages

- **What's New**: AGENT-CQ라는 새로운 프레임워크가 발표되었습니다. 이 시스템은 대화형 검색 시스템에서 다양한 기다리는 질문(clarifying questions)을 자동 생성하고 평가하는 기능을 제공합니다. 기존 방법들의 한계를 극복하기 위해 LLM 기반(end-to-end LLM-based) 접근 방식을 채택했습니다.

- **Technical Details**: AGENT-CQ는 두 단계로 구성되어 있습니다. 첫 번째 단계는 생성 단계(generation stage)로, LLM 프롬프트(prompting) 전략을 사용하여 기다리는 질문을 생성합니다. 두 번째 단계는 평가 단계(evaluation stage)로, CrowdLLM을 사용하여 다수의 LLM 인스턴스를 통해 생성된 질문과 답변의 품질을 종합적인 품질 메트릭(classic quality metrics)에 따라 평가합니다.

- **Performance Highlights**: ClariQ 데이터셋을 기반으로 한 실험에서 CrowdLLM의 평가가 질문과 답변의 품질을 높이는 데 매우 효과적임을 입증하였습니다. AGENT-CQ의 생성 단계는 다양한 질문과 답변 품질 측면에서 기준선(baselines)을 지속적으로 능가했습니다. 검색 기반 평가에서는 LLM이 생성한 질문이 인간이 생성한 질문에 비해 BM25 및 크로스 인코더(cross-encoder) 모델 모두에서 검색 효과를 크게 향상시키는 것으로 나타났습니다.



### ProvocationProbe: Instigating Hate Speech Dataset from Twitter (https://arxiv.org/abs/2410.19687)
- **What's New**: 본 논문에서는 새로운 데이터셋인 ProvocationProbe를 소개하고, 이는 일반적인 혐오발언(hate speech)과 증오를 조장하는 발언(instigating hate speech)의 차이점을 탐구하기 위해 설계되었습니다. 이 연구에서는 트위터에서 수집된 약 20,000개의 트윗을 포함하여, 인종차별, 정치, 종교와 같은 다양한 글로벌 논란을 다루고 있습니다.

- **Technical Details**: ProvocationProbe 데이터셋은 세 가지 카테고리로 나뉘어 있습니다: Non Hateful (NH), Instigating Hate (IH), Non Instigating Hate (NIH) 발언. IH 발언은 특정 개인이나 집단에 대한 정체성 공격이 지속적으로 포함된다는 특징이 있으며, 이를 통해 IH 발언이 NIH 발언과 어떻게 다른지 분석하고 있습니다. NMF 주제 모델링(NMF topic modeling)을 이용하여 특정 주제에 대한 표적성을 식별하였습니다.

- **Performance Highlights**: IH 발언의 분석이 이루어지며, 이는 온라인 혐오발언 확산을 완화하려는 노력에 기여할 것입니다. IH 발언은 충분히 연구되지 않은 주제이며, 본 데이터셋을 통해 IH 발언과 NIH 발언 간의 차이점을 이해하는 데 도움을 줄 것입니다.



### A distributional simplicity bias in the learning dynamics of transformers (https://arxiv.org/abs/2410.19637)
Comments:
          10 pages, 5 figures, Accepted at NeurIPS 2024

- **What's New**: 이번 연구는 자연어 데이터에 대해 훈련된 Transformer 모델이 'simplicity bias'를 보여준다는 것을 입증했습니다. 특히, 이 모델들이 입력 토큰 간의 상호작용을 순차적으로 학습하며 차원에 따른 포화점을 발견하였습니다.

- **Technical Details**: 연구진은 자연어 처리(NLP) 데이터셋에 대한 상호작용을 캡처하기 위해 'clones'라는 기법을 개발하였습니다. 이를 통해 토큰 간의 상호작용을 제어하는 새로운 Transformer 아키텍처를 제안합니다. 이 아키텍처는 MLM(Masked Language Modeling) 방식을 사용하여 훈련되며, 깊이에 따라 상호작용의 정도를 조정할 수 있습니다.

- **Performance Highlights**: BERT-like Transformer 인코더를 사용한 실험에서, 낮은 차수의 상호작용만 포함된 클론에서의 테스트 손실이 약 60 SGD 에포크 이후 포화 상태에 이르는 반면, 높은 차수의 상호작용을 갖는 클론에서는 손실이 계속 개선되는 경향을 보여주었습니다. 이는 처음에는 낮은 차수의 상호작용을 학습하고 이후에 높은 차수의 상호작용을 활용하게 됨을 의미합니다.



### OpenWebVoyager: Building Multimodal Web Agents via Iterative Real-World Exploration, Feedback and Optimization (https://arxiv.org/abs/2410.19609)
- **What's New**: 새롭게 소개된 OpenWebVoyager는 멀티모달 웹 에이전트를 개발하기 위한 오픈 소스 프레임워크입니다. 이 프레임워크는 자율적으로 실제 환경을 탐색하고 스스로 개선할 수 있는 웹 에이전트를 만들 수 있도록 돕습니다.

- **Technical Details**: OpenWebVoyager는 맨 처음에 imitation learning (모방 학습)으로 기본 능력을 익히고, 이후 웹을 탐색하며 피드백을 수집합니다. 이 과정에서 잘 수행된 궤적을 다른 일반 목적 모델을 통해 학습하여 정책을 개선합니다. 이러한 exploration-feedback-optimization (탐색-피드백-최적화) 사이클은 여러 번 반복될 수 있습니다.

- **Performance Highlights**: 실험 결과, OpenWebVoyager는 반복할수록 성능이 향상되며, WebVoyager 테스트 세트에서 작업 성공률이 19.9%에서 25.8%로, Mind2Web 크로스 과제 세트에서는 6.3%에서 19.6%로 증가하였습니다. 또한, Mind2Web 크로스 웹 세트에서도 성과가 개선되었습니다.



### ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems (https://arxiv.org/abs/2410.19572)
- **What's New**: 이 논문은 LLM 기반 청크 필터링(ChunkRAG) 프레임워크를 제안하여 기존의 Retrieval-Augmented Generation (RAG) 시스템에서 무관한 정보를 제거함으로써 정확도를 개선하는 방법을 설명합니다.

- **Technical Details**: ChunkRAG는 문서 수준이 아닌 청크 수준에서 정보를 평가하고 필터링하는 방식으로 운영됩니다. 이 시스템은 세멘틱 청크(semantic chunking)를 통해 문서를 고유한 섹션으로 나누고, LLM 기반 관련성 점수를 적용하여 각 청크가 사용자 쿼리와 얼마나 일치하는지를 평가합니다.

- **Performance Highlights**: 실험 결과, ChunkRAG는 기존의 RAG 모델보다 높은 정확성을 달성했으며, 특히 사실 확인 및 다단계 추론과 같은 정밀한 정보 검색이 필요한 작업에서 효과적입니다.



### Detection of Human and Machine-Authored Fake News in Urdu (https://arxiv.org/abs/2410.19517)
- **What's New**: 최근 소셜 미디어의 확산과 함께 생성된 기계 뉴스(machien news)가 가짜 뉴스 검출 체계에 미치는 영향을 분석한 연구가 발표되었습니다. 이 연구에서 저자들은 우르두어를 포함한 저자원 언어(low-resource languages)에서의 가짜 뉴스 감지의 중요성을 강조하고, 기계 생성 뉴스와 진짜 뉴스의 차이를 구별하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 기존의 이진 분류 방법을 넘어서 우르두어에서의 기계 생성 가짜 뉴스와 진짜 뉴스를 포함한 새로운 네 가지 레이블 데이터셋을 개발하였습니다. 또한, 원래의 네 클래스 문제를 기계 생성 텍스트 감지와 가짜 뉴스 감지의 두 개의 하위 작업으로 나누는 계층적 방법(hierarchical method)을 제안하여 성능 개선을 도모합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 기존의 RoBERTa 기법 보다 정확도(accuracy)와 F1 점수에서 우수한 성능을 보이는 것으로 나타났습니다. 이는 다양한 데이터셋과 환경에서의 효과성과 강인성을 강조하며, 우르두어의 가짜 뉴스 검출 연구의 새로운 기반이 될 것으로 기대됩니다.



### SWITCH: Studying with Teacher for Knowledge Distillation of Large Language Models (https://arxiv.org/abs/2410.19503)
- **What's New**: 이번 연구에서는 Knowledge Distillation (KD)의 효율성을 높이기 위해 SWITCH라는 새로운 접근 방식을 제안합니다. SWITCH는 학생 모델의 시퀀스 생성 과정에서 교사 모델을 전략적으로 통합하여 불일치 문제를 해결합니다.

- **Technical Details**: SWITCH는 교사와 학생 모델 간의 토큰 확률 불일치를 식별하여 교사가 선택적으로 개입할 수 있도록 합니다. 이 방법은 특히 긴 시퀀스 생성을 할 때 발생할 수 있는 노이즈 및 편향 문제를 완화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 세 가지 모델 패밀리와 다섯 개의 지시 준수 데이터셋을 통해, SWITCH는 전통적인 Knowledge Distillation 방법보다 뛰어난 성능을 보여주며, 특히 긴 시퀀스 데이터 생성에 강점을 발휘합니다.



### Introducing MAPO: Momentum-Aided Gradient Descent Prompt Optimization (https://arxiv.org/abs/2410.19499)
- **What's New**: 제안된 Momentum-Aided Prompt Optimization (MAPO)는 Large Language Models (LLMs)의 효율성을 높이는 새로운 접근법입니다. 기존의 ProTeGi를 개선하여, MAPO는 자연어 'gradient'와 모멘텀 기반의 확장을 사용하여 프롬프트 최적화를 자동화합니다.

- **Technical Details**: MAPO는 긍정적인 자연어 gradient를 생성하고, 이를 기반으로 풍부한 후보들을 탐색하여 최적의 프롬프트를 선택합니다. 이 과정에서 Beam Search와 Upper Confidence Bound (UCB) 알고리즘을 활용하여 후보 프롬프트의 선택과 확대를 효율적으로 수행합니다. 또한 gradient 이력을 추적하여 지역 최솟값(local minima)과 진동(oscillation)을 피합니다.

- **Performance Highlights**: MAPO는 ProTeGi에 비해 더 빠른 수렴 시간(convergence time)과 적은 API 호출, 더 높은 F1 점수를 달성했습니다. 이로 인해 MAPO는 LLMs의 자동화된 프롬프트 엔지니어링을 위한 강력하고 확장 가능한 솔루션임을 입증하였습니다.



### Graph Linearization Methods for Reasoning on Graphs with Large Language Models (https://arxiv.org/abs/2410.19494)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)을 그래프 기계 학습(Graph Machine Learning) 작업에 적합하도록 변형하는 방법을 제안합니다. 특히, 그래프를 직선 시퀀스 토큰으로 변환하는 프로세스인 그래프 리니어라이제이션(Graph Linearization)에 초점을 맞추고 있으며, 이는 LLM이 그래프를 자연스럽게 처리할 수 있도록 돕습니다.

- **Technical Details**: 연구에서는 그래프 중심성(centrality) 및 노드 재라벨링(node relabeling) 방법을 기반으로 다양한 그래프 리니어화 방법을 개발하였습니다. 이를 통해 그래프의 지역적 의존(local dependency) 및 전역 정렬(global alignment) 속성을 반영하여 LLM의 성능을 개선하도록 했습니다.

- **Performance Highlights**: 인공 그래프를 대상으로 한 실험에서, 제안된 방법들이 임의의 리니어라이제이션 기반보다 뛰어난 성능을 보였습니다. Llama 3 Instruct 모델을 사용한 실험 결과에서는 노드 수 세기, 최대 차수 계산 및 형태 분류와 같은 기초적인 그래프 추론 작업이 포함되어 있어, 제안 방법의 효과성을 입증하였습니다.



### A Debate-Driven Experiment on LLM Hallucinations and Accuracy (https://arxiv.org/abs/2410.19485)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서의 환각(hallucination) 현상을 새로운 실험 프레임워크를 통해 탐구하며, 여러 개의 GPT-4o-Mini 모델이 사실 확인 문제에 대한 토론을 진행하도록 하여 정보의 정확성과 강건성을 향상시키기 위한 기회를 제시합니다.

- **Technical Details**: 연구는 각각 하나의 Saboteur(교란자)와 Truthful(정확한) 모델들로 구성된 여러 개의 GPT-4o-Mini 모델 인스턴스를 활용하여 진행되었습니다. Saboteur 모델은 거짓이지만 그럴듯한 답변을 생성하도록 지시받고, 다른 모델들은 사실에 기반한 답변을 제공합니다. 이 토론은 TruthfulQA 데이터셋을 기반으로 하며 모델 간의 상호작용을 분석하는 데 중점을 둡니다.

- **Performance Highlights**: Saboteur가 있는 경우 전체 평균 정확도는 78.72%로, 기본 정확도 61.94%에서 유의미하게 개선되었습니다. 특히 역사 및 날씨 카테고리에서 100% 정확도를 달성한 반면, 심리학(16.67%) 및 혼동 유형(0.00%)에서 낮은 정확도가 나타났습니다. 이러한 결과는 모델이 명확하고 사실적인 질문에 대해 더 나은 성능을 보인다는 것을 시사합니다.



### ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Contrastive Framework (https://arxiv.org/abs/2410.19453)
Comments:
          23 pages, 11 figures

- **What's New**: ShifCon은 비우세 언어(non-dominant language)의 성능을 개선하기 위해 설계된 Shift-based Contrastive 프레임워크입니다. 이 프레임워크는 비우세 언어의 표현(representation)을 주 언어(dominant language)와 정렬하는 방법을 제시합니다.

- **Technical Details**: ShifCon은 두 개의 모듈로 구성됩니다: 1) Shift Projection은 비우세 언어의 표현을 주 언어의 서브스페이스(subspace)로 매핑하여 주 언어와 유사한 표현을 생성하고, 이러한 표현을 생성 전 원래 언어 서브스페이스로 다시 이동시킵니다. 2) Multilingual Contrastive Learning은 비우세 언어의 주 언어 유사 표현과 주 언어의 표현 간의 정렬을 더욱 강화합니다. 이를 통해 모델 파라미터에 인코딩된 더 많은 정보를 접근할 수 있게 됩니다.

- **Performance Highlights**: ShifCon 프레임워크는 Llama-27B 모델에서 MGSM(Multilingual Generalized Supervised Metric) 작업에 대해 18.9%의 성능 향상을 보였으며, 저자들은 이 프레임워크가 비우세 언어의 성능을 크게 향상시킨다고 주장합니다.



### Intelligent Understanding of Large Language Models in Traditional Chinese Medicine Based on Prompt Engineering Framework (https://arxiv.org/abs/2410.19451)
- **What's New**: 이 논문은 전통 중국 의학(Traditional Chinese Medicine, TCM) 영역에서 대형 언어 모델(large language models, LLMs)의 성능을 향상시키기 위한 프롬프트 엔지니어링(prompt engineering)의 응용을 탐구합니다. TCM-Prompt라는 프레임워크를 제안하며, 이는 다양한 사전 훈련 언어 모델(pre-trained language models, PLMs), 템플릿, 토크나이제이션(tokenization), 및 언어화(verbalization) 방법을 통합하여 연구자들이 TCM 관련 작업에 특화된 모델을 쉽게 구성하고 조정할 수 있도록 합니다.

- **Technical Details**: TCM-Prompt 프레임워크는 질병 분류(disease classification), 증후군 식별(syndrome identification), 한방 약 추천(herbal medicine recommendation), 그리고 일반 NLP 작업(general NLP tasks)에서 실험을 수행하였습니다. 이 연구는 기존의 방법론과 비교할 때 우리의 접근법의 효과성과 우수성을 입증합니다.

- **Performance Highlights**: 연구 결과는 프롬프트 엔지니어링이 TCM과 같은 전문 영역에서 LLMs의 성능 향상을 위한 유망한 기술이라는 것을 시사하며, 디지털화(digitalization), 현대화(modernization), 그리고 개인화된 의학(personalized medicine) 분야에서의 잠재적 응용 가능성을 보여줍니다.



### KAHANI: Culturally-Nuanced Visual Storytelling Pipeline for Non-Western Cultures (https://arxiv.org/abs/2410.19419)
Comments:
          Under review

- **What's New**: 이번 논문은 비서구 문화에 맞춰진 시각적 이야기 생성을 위한 KAHANI라는 스토리텔링 파이프라인을 개발하였다는 점에서 주목받고 있습니다. 기존의 Language Models는 북반구(Global North)의 감성에 맞춰져 있었으나, KAHANI는 다양한 문화적 배경을 포함하여 더 포괄적인 이야기를 생성합니다.

- **Technical Details**: KAHANI는 JSON 기준으로 사용자의 프롬프트에서 문화적 맥락을 캡처하기 위해 Chain of Thought (CoT)와 Text-To-Image (T2I) 프롬팅 기법을 이용합니다. 이 과정에서 GPT-4 Turbo와 Stable Diffusion XL (SDXL) 모델을 활용하여 캐릭터와 장면 구성에 대한 생생한 설명이 생성됩니다.

- **Performance Highlights**: 사용자 연구 결과에 따르면 KAHANI는 ChatGPT-4(및 DALL-E3)와 비교했을 때 더 많은 Cultural Specific Items (CSIs)를 포함하여 문화적 관련성을 강화했습니다. 36개의 비교 중 27건에서 KAHANI가 ChatGPT-4보다 문화적 역량과 시각적 이야기 생성 품질 모두에서 우수한 성능을 발휘했습니다.



### Investigating the Role of Prompting and External Tools in Hallucination Rates of Large Language Models (https://arxiv.org/abs/2410.19385)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 환각(hallucination) 감소를 위한 다양한 프롬프트(prompt) 전략과 프레임워크에 대한 포괄적인 실증 평가를 제공합니다. 또한, 외부 도구가 추가된 LLM 에이전트의 환각 생성 비율에 미치는 영향을 조사합니다.

- **Technical Details**: LLM은 다양한 자연어 처리(NLP) 작업을 수행할 수 있는 강력한 계산 모델입니다. 이 연구에서는 다양한 프롬프트 기법을 넓은 벤치마크 데이터셋에 적용하여 각 방법의 정확도와 환각 비율을 평가하고, 환각의 맥락 의존성을 탐구합니다. 온도(temperature) 설정, 양자화(quantization) 등과 같은 모델 압축 기법도 논의됩니다.

- **Performance Highlights**: 단순한 프롬프트 기법이 더 복잡한 방법보다 환각을 줄이는 데 종종 더 효과적임을 보여주며, LLM 에이전트는 외부 도구 사용으로 인해 환각 비율이 현저하게 증가할 수 있음을 입증했습니다.



### Interleaving Text and Number Embeddings to Solve Mathemathics Problems (https://arxiv.org/abs/2410.19353)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 수치 예측 및 산술 연산 성능 향상을 위해 표현력이 풍부한 숫자 임베딩과 텍스트와 숫자 간의 구분을 위한 라우팅 레이어를 소개합니다. 이는 기존의 숫자 토큰화 방법의 한계를 극복하는 데 기여합니다.

- **Technical Details**: 연구에서는 MLP (Multi-Layer Perceptron)를 사용하여 각각의 숫자에 독특한 벡터 방향을 할당하고, Mixture-of-Experts (MoE) 모델에서 영감을 받은 라우팅 레이어를 통해 텍스트와 숫자 임베딩 간의 구조를 구분합니다. 이를 통해 숫자와 텍스트 간의 분포를 다르게 하여 산술 연산을 가능하게 합니다.

- **Performance Highlights**: 45M 파라미터의 인코더-디코더 아키텍처에서 우리의 방법은 $R^2$=0.9988을 달성하였으며, 여러 가지 수치 비편향과 아티팩트 개선을 관찰했습니다. 실험 결과, 우리의 방법은 숫자와 텍스트가 혼합된 복잡한 문제에서 기존 모델 대비 월등한 성능을 보였습니다.



### AgentSense: Benchmarking Social Intelligence of Language Agents through Interactive Scenarios (https://arxiv.org/abs/2410.19346)
- **What's New**: 본 논문은 AgentSense라는 새로운 벤치마크를 소개하며, 이는 다양한 상호작용 시나리오를 통해 언어 에이전트의 사회적 지능을 평가하는 데 중점을 둡니다. 이전 연구들은 충분한 시나리오 다양성과 복잡성을 결핍하고 있었지만, AgentSense는 이를 해결하기 위해 1,225개의 다양한 사회적 시나리오를 제작했습니다.

- **Technical Details**: AgentSense는 Dramaturgical Theory(드라마 이론)을 바탕으로 하여 극복적 접근(합성적으로) 방식을 사용하여 사회적 목표와 개인 정보를 포함한 시나리오를 설계합니다. 실험은 멀티 턴 인터랙션을 수반하며, 참가자들의 목표 달성 여부(Goal Completion)와 타인의 개인 정보를 추론할 수 있는 능력(Implicit Reasoning)을 측정합니다. ERG 이론을 바탕으로 하여 목표를 분석합니다.

- **Performance Highlights**: 실험 결과, LLMs는 복잡한 사회적 시나리오 특히 고급 성장 필요에 대한 목표를 달성하는 데 어려움을 겪는다는 것을 발견했습니다. 이와 함께 GPT-4o조차도 개인 정보 추론에 있어서 개선이 필요하다는 결과를 도출했습니다.



### Two are better than one: Context window extension with multi-grained self-injection (https://arxiv.org/abs/2410.19318)
Comments:
          The code is available at this https URL

- **What's New**: 이 논문은 SharedLLM이라는 새로운 접근 방식을 제안합니다. SharedLLM은 다중 세분화된 컨텍스트 압축 및 쿼리 인지 정보 검색의 설계 철학에 기반하여 두 개의 짧은 컨텍스트 LLM으로 구성됩니다. 이 모델은 압축된 정보를 전달하기 위해 하부 모델과 상부 모델 간의 효율적인 연결을 최적화합니다.

- **Technical Details**: SharedLLM은 LLaMA-2와 같은 두 개의 짧은 컨텍스트 LLM으로 구성됩니다. 하부 모델은 컨텍스트 정보를 압축하고, 상부 모델은 이를 기반으로 하는 언어 모델링을 수행합니다. 또한, 그들은 특화된 트리 형태의 데이터 구조를 도입하여 다양한 레벨의 정보를 효율적으로 인코딩, 저장, 검색합니다. 정보를 주입하는 단계에서는 청크-레벨의 위치 ID를 정의하여 쿼리의 상대적인 위치를 인식할 수 있도록 합니다.

- **Performance Highlights**: SharedLLM은 8K 토큰의 텍스트로 훈련되며, 128K 토큰 시퀀스를 처리할 수 있는 뛰어난 외삽 능력을 나타냅니다. 200K 이상의 최대 길이에 대한 모든 실험은 단일 A800 80G GPU에서 수행할 수 있으며, SharedLLM은 상대적으로 낮은 메모리 소비로 모든 기준 모델보다 몇 배 더 빠른 속도를 보여줍니다.



### FairMT-Bench: Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs (https://arxiv.org/abs/2410.19317)
- **What's New**: 대화가 복잡한 멀티턴(Multi-Turn) 환경에서의 공정성(Fairness) 문제를 다루는 포괄적인 벤치마크, FairMT-Bench를 제안하며, 기존의 단일 대화(turn) 벤치마크를 넘어서 더 현실적인 상황에서의 공정성을 평가합니다.

- **Technical Details**: 논문에서는 FairMT-Bench라는 멀티턴 대화 시나리오에서 LLM(대형 언어 모델)의 공정성을 평가하기 위한 종합 벤치마크를 소개하며, 3단계(컨텍스트 이해, 사용자 상호작용, 지시 교환)의 태스크 분류(Taxonomy)를 통해 공정성 측정능력을 제고하고, FairMT-10K라는 데이터셋을 통해 다양한 편향(Bias) 유형을 측정합니다.

- **Performance Highlights**: 실험 결과, 현재 LLM들은 멀티턴 대화 시나리오에서 편향된 응답을 생성할 가능성이 높으며, 다양한 태스크와 모델 간의 성능 차이가 두드러진다는 것을 보여주었습니다. 또한 FairMT-1K라는 더욱 도전적인 데이터셋을 활용하여 15개 최첨단 LLM의 성능을 평가한 결과, 공정성 확보가 여전히 큰 도전 과제임을 강조하였습니다.



### Any Other Thoughts, Hedgehog? Linking Deliberation Chains in Collaborative Dialogues (https://arxiv.org/abs/2410.19301)
Comments:
          Accepted at Findings of EMNLP 2024

- **What's New**: 이번 연구에서는 공동 대화에서 질문하는 행위의 중요성을 재조명하고, 이와 관련된 원인 관계의 모델링을 통해 'probing questions' 즉, 상대방의 응답을 명시적으로 유도하는 질문의 중요성을 강조합니다.

- **Technical Details**: 연구는 'deliberation chains'라는 그래프 기반 프레임워크를 도입하여 대화 중의 발언들이 어떻게 'probing utterances'로 이어지는지를 모델링합니다. 이러한 체인은 대화의 흐름 속에서 원인과 그에 따른 질문들 간의 관련성을 함께 모델링합니다.

- **Performance Highlights**: DeliData와 Weights Task라는 두 개의 복잡한 협업 작업 데이터셋에서 평가했으며, 기존의 방법들에 비해 이론적 근거가 있는 접근방식이 효과적임을 입증하였습니다.



### Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning (https://arxiv.org/abs/2410.19290)
- **What's New**: 이번 연구에서는 LLM(Large Language Model)의 허위 응답 원인 중 하나로 사전 학습(pre-training)과 미세 조정(fine-tuning) 과정 간의 지식 불일치(knowledge inconsistency)를 지적하고, 이를 해결하기 위해 Prereq-Tune이라 불리는 새로운 미세 조정 전략을 제안합니다.

- **Technical Details**: Prereq-Tune은 두 가지 단계를 포함하는 혁신적인 미세 조정 방법으로, 첫 번째 단계에서는 LLM이 Supervised Fine-Tuning (SFT)에서 필요한 지식을 학습하는 선행 학습(prerequisite learning)을 수행합니다. 두 번째 단계에서는 이전 단계에서 고정된 지식을 기반으로 기술(기능)을 학습하게 됩니다. 이를 통해 모델은 지식 불일치 문제를 극복하고 정확한 기술을 배우게 됩니다.

- **Performance Highlights**: Experiments show that Prereq-Tune이 기존의 기본 모델보다 짧은 Q&A 및 긴 형식 생성을 포함한 작업에서 LLM의 사실성(factuality)을 유의미하게 개선하며, 허위 응답(hallucination) 감소에서 기존 최첨단 기술을 능가함을 입증합니다. 또한, Prereq-Tune은 새로운 지식 제어 생성 가능성을 열어줍니다.



### Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning (https://arxiv.org/abs/2410.19258)
Comments:
          18pages,submitted to ICLR 2025

- **What's New**: 이번 논문에서는 HeadKV라는 새로운 head-level KV 캐시 압축 방법을 제안합니다. 이는 각 attention head의 중요도를 평가하여 KV 캐시의 예산을 효율적으로 할당하는 방법으로, 이전 연구에서 제안된 단순 layer-level 압축 방식보다 성능이 우수합니다.

- **Technical Details**: HeadKV는 각 attention head의 retrieval과 reasoning 능력을 평가하여 KV 캐시의 적절한 할당을 결정합니다. 이 방법은 LongBench와 LooGLE과 같은 다양한 검증 벤치마크를 통한 실험을 통해 각기 다른 모델 아키텍처(Llama-3-8B-Instruct, Mistral-7B-Instruct)에서 효과를 입증했습니다.

- **Performance Highlights**: HeadKV를 사용했을 때 KV 캐시의 1.5%만을 유지하면서도 전체 KV 캐시 모델에서 97%에 해당하는 성능을 유지할 수 있었고, 특히 낮은 자원 환경에서 강력한 기준선(base line)을 초과하는 성능을 보여주었습니다.



### The Reopening of Pandora's Box: Analyzing the Role of LLMs in the Evolving Battle Against AI-Generated Fake News (https://arxiv.org/abs/2410.19250)
- **What's New**: 이번 연구는 대학 차원에서 개최된 대회에서 LLM(대형 언어 모델)을 이용한 가짜 뉴스 생성 및 탐지에 관한 결과를 제시합니다. 총 110명의 참가자가 252개의 독창적인 가짜 뉴스 이야기를 생성하였으며, 84명의 참가자가 탐지 작업에 참여했습니다. LLM은 인간보다 약 68% 더 효과적으로 진짜 뉴스를 탐지하지만, 가짜 뉴스 탐지에 있어서는 LLM과 인간의 성능이 유사한 결과 (~60% 정확도)를 보였습니다.

- **Technical Details**: 연구는 두 단계로 진행되었습니다. 첫 번째 단계에서는 참가자들이 LLM을 사용하여 가짜 뉴스를 생성하였으며, 두 번째 단계에서는 다른 참가자들이 이 가짜 뉴스와 진짜 뉴스를 구별하는 작업을 수행했습니다. 연구 결과는 시각적 요소가 가짜 뉴스 탐지 정확도에 미치는 영향을 분석하고, 가짜 뉴스 생성자가 사용하는 다양한 전략에 대해서도 조사했습니다.

- **Performance Highlights**: LLM은 진짜 뉴스 탐지에서 인간보다 약 68% 더 효과적이며, 가짜 뉴스에 대해서는 유사한 성능을 보였습니다. 특히, 일부 뉴스 주제에 대한 가짜 뉴스 탐지가 어려운 점과 혼합된 프롬프트 전략 사용이 탐지를 더욱 복잡하게 만들었다는 점이 강조되었습니다.



### Developing a Tutoring Dialog Dataset to Optimize LLMs for Educational Us (https://arxiv.org/abs/2410.19231)
- **What's New**: 최근의 큰 언어 모델(LLMs)의 발전은 교육 분야에서의 확장 가능한 응용 가능성을 보여주었지만, 대화 기반 튜터링 시스템에서의 사용은 효율적인 교수법( pedagogical strategies) 필요성과 전문가가 정제한 데이터셋의 높은 비용으로 인해 여전히 도전적인 과제입니다. 본 연구는 읽기 이해 문제 해결에 있어 1:1 튜터링을 위해 더 작고 저렴한 LLM을 활용하는 방식을 탐구합니다.

- **Technical Details**: 우리는 합성 튜터링 대화 데이터셋( synthetic tutoring dialog dataset)을 개발하고, 이를 통해 소형 LLM을 미세 조정(fine-tuning)하였습니다. 더 나아가, 실제 튜터링 시나리오에서 미세 조정된 모델과 대형 모델의 성능을 비교하는 인터랙티브 실험을 진행하였습니다. 연구에 사용된 모델은 Transformer 기반의 혼합 전문가(Mixture of Experts) LLM인 Mistral ‘8x7b’와 ‘7b’ 모델입니다.

- **Performance Highlights**: 연구 결과는 미세 조정된 모델이 대형 모델과 동등한 성능을 보이며, 비용 측면에서도 우수하다는 것을 보여주었습니다. 즉, 더 작은 모델이 저렴한 비용으로 교육 환경에서 LLM 기반 튜터링 시스템을 구현하는데 유효하고 비용 효율적인 접근법임을 증명하였습니다.



### Can Stories Help LLMs Reason? Curating Information Space Through Narrativ (https://arxiv.org/abs/2410.19221)
- **What's New**: 이 논문에서는 내러티브 요소를 통합하여 대형 언어 모델(LLM)이 복잡한 문제를 보다 효과적으로 해결할 수 있도록 돕는 새로운 접근법인 'Thought의 이야기(Story of Thought, SoT)'를 제안합니다.

- **Technical Details**: SoT는 문제 진술 주위에 내러티브를 구성하고 관련 정보를 식별하고 조직하는 프레임워크를 생성하여 문제 해결을 위한 프롬프트 기술에 내러티브 구조를 통합하는 방법입니다. 이 방법은 질문 명확화, 내러티브 생성 및 문제 해결의 세 단계를 포함합니다.

- **Performance Highlights**: 실험 결과, SoT를 사용한 LLM들은 GPQA 및 JEEBench 데이터셋의 물리학, 화학, 수학 및 생물학 문제에서 기존의 프롬프트 기술보다 일관되게 우수한 성능을 보였습니다.



### Label Set Optimization via Activation Distribution Kurtosis for Zero-shot Classification with Generative Models (https://arxiv.org/abs/2410.19195)
- **What's New**: 본 연구는 제로샷 분류(zero-shot classification)에서 레이블 옵션(label option)이 ICL 성능에 미치는 영향을 처음으로 포괄적으로 조사한 경험적 연구이다. 특히 레이블 이름의 선택, 순서 및 세부 정보가 모델 성능에 미치는 영향을 분석하였다.

- **Technical Details**: 이 연구는 다양한 레이블 변형(lexical variants)의 영향을 조사하기 위해 스탠스 분류(stance classification) 작업을 기반으로 하였다. 또한, 모델 내부 상태의 분석을 통해 최적의 레이블 이름은 피드포워드 네트워크(feed forward network)에서 아웃라이어 신경 세포를 적게 활성화시키는 경향이 있음을 보여주었다. LOADS라는 후처리(post-hoc) 방법을 제안하여, 100개의 비라벨 샘플만으로도 다양한 모델과 언어에서 효과를 발휘함을 입증하였다.

- **Performance Highlights**: LOADS 방법은 100개의 라벨 없는 샘플만으로도 제로샷 ICL 성능을 개선할 수 있으며, 다양한 모델 유형과 크기, 언어에서 통계적으로 유의미한 성능 향상을 보여준다. 본 연구는 ICL 모델의 성능을 향상시키기 위한 레이블 설계에 대한 유용한 권고사항을 제공한다.



### Enriching GNNs with Text Contextual Representations for Detecting Disinformation Campaigns on Social Media (https://arxiv.org/abs/2410.19193)
Comments:
          Work in progress

- **What's New**: 본 연구는 Graph Neural Networks(GNNs)에 텍스트 정보를 통합하여 가짜 뉴스 탐지의 성능을 개선하는 방법을 조사합니다. 최근 Transformer 기반 언어 모델의 진전을 활용하여, 정적 표현(static representations)과 비교해 9.3%의 성능 향상을 나타냈습니다.

- **Technical Details**: 연구에서는 사용자의 프로필 및 상호 작용에서 얻은 텍스트 정보를 GNN의 노드에 통합하여, 정보 전파 네트워크에서 가짜 뉴스를 분류하는 모델을 설계하였습니다. GNNs는 메시지 전송(message passing)을 통해 그래프 내 의존 관계를 포착하며, 이 연구에서는 Graph Attention Networks(GATs)를 사용하여 노드 임베딩(node embeddings)을 생성합니다.

- **Performance Highlights**: 텍스트 표현을 GNN에 통합한 결과, 정적 표현에 비해 9.3%의 성능 향상과 텍스트가 없는 GNN에 비해 33.8%의 향상을 달성했습니다. 그러나 노이즈가 포함된 데이터 증강 방법은 성능 저하와 불안정을 초래하였습니다.



### No Argument Left Behind: Overlapping Chunks for Faster Processing of Arbitrarily Long Legal Texts (https://arxiv.org/abs/2410.19184)
Comments:
          To appear at 15th STIL @ BRACIS'24

- **What's New**: 브라질의 사법 시스템이 수백만 건의 사건을 처리하는 데 어려움을 겪고 있는 가운데, 법률 텍스트를 효과적으로 분석하기 위한 효율적인 방법이 필요해졌습니다. 이를 위해 uBERT라는 하이브리드 모델이 개발되었습니다. uBERT는 Transformer와 Recurrent Neural Network 아키텍처를 결합하여 긴 법률 텍스트를 처리할 수 있습니다.

- **Technical Details**: uBERT 모델은 입력된 전체 텍스트를 길이에 관계없이 처리할 수 있으며, 적절한 계산 오버헤드를 유지합니다. uBERT는 BERT+LSTM 및 ULMFiT와의 비교 실험을 통해, 특히 오버랩(Overlapping) 입력을 사용할 때 우수한 성능을 보이는 것으로 확인되었습니다. 또한 ULMFiT는 긴 문서를 처리하는 데 더 뛰어난 성능을 보였지만, uBERT보다 4배 느리다는 결과가 나왔습니다.

- **Performance Highlights**: uBERT는 BERT+LSTM을 약간 초과하는 성능을 보이며 긴 법률 문서를 처리하는 데에 있어 빠른 속도를 자랑합니다. 실험 결과, 주어진 데이터 세트에서 법률적인 판단 예측 작업에서 유의미한 성능 개선을 보여주었습니다.



### Lived Experience Not Found: LLMs Struggle to Align with Experts on Addressing Adverse Drug Reactions from Psychiatric Medication Us (https://arxiv.org/abs/2410.19155)
Comments:
          27 pages, 8 figures, 15 tables

- **What's New**: 이번 연구에서는 정신과 약물과 관련된 부작용(Adverse Drug Reactions, ADR)을 탐지하고, LLM(대형 언어 모델)의 성능을 평가하기 위한 Psych-ADR 벤치마크 및 ADRA(Adverse Drug Reaction Response Assessment) 프레임워크를 소개하고 있습니다. 기존 연구와 다른 점은 정신과 약물의 ADR 탐지 및 대응 전략에 LLM의 효용성을 체계적으로 평가했다는 것입니다.

- **Technical Details**: 연구에서는 Reddit에서 수집한 데이터를 바탕으로 LLM이 ADR을 탐지하고, 전문가와의 전략적 정합성을 평가했습니다. Psych-ADR 벤치마크는 239개의 Reddit 포스트와 전문적인 응답을 포함하고 있으며, LLM의 응답을 네 가지 축으로 평가합니다: (a) 텍스트 가독성, (b) 감정 및 어조 표현, (c) 해로운 결과 저감 전략의 정합성, (d) 전략의 실행 가능성입니다.

- **Performance Highlights**: 연구 결과, LLM은 ADR 탐지 및 분류에 어려움을 겪고 있으며, 전문가와의 일치는 70.86%에 불과하고, 평균적으로 12.32% 덜 실용적인 조언을 제공합니다. 이러한 결과는 의료 질환 및 약물 간의 상호작용을 다루는 과제에서 LLM의 성능을 평가하는 데 중요한 벤치마크가 될 것입니다.



### AlignCap: Aligning Speech Emotion Captioning to Human Preferences (https://arxiv.org/abs/2410.19134)
Comments:
          Accepted to EMNLP2024 main conference

- **What's New**: 본 논문에서는 Speech Emotion Captioning (SEC)에서의 최신 접근 방식인 AlignCap을 제안합니다. AlignCap은 대규모 언어 모델(LLM)을 기반으로 하여 음성의 감정 설명을 생성하고, 인간의 선호를 반영하는 정교한 캡션을 생성하는 것을 목표로 합니다.

- **Technical Details**: AlignCap은 두 가지 주요 특성을 가지고 있습니다: 1) Speech-Text Alignment, 즉 음성과 텍스트 입력에 대한 LLM의 응답 예측 분포의 차이를 최소화하는 지식 증류(knowledge distillation, KD) 정규화를 사용합니다. 2) Human Preference Alignment, 여기서는 사실성과 충실성의 환각을 제거하기 위한 Preference Optimization (PO) 정규화를 설계했습니다. 또한, 정교한 정보를 풍부하게 하기 위해 감정 단서를 프롬프트로 추출합니다.

- **Performance Highlights**: 실험 결과, AlignCap은 제로-샷 SEC 작업에서 다른 최신 방법들에 비해 우수한 성능을 보였습니다. 기존 SEC 모델들이 새로운 음성에 대해 저조한 일반화 성능을 보이는 반면, AlignCap은 감정 설명의 풍부함과 일관성을 유지하며 더 나은 성능을 달성합니다.



### Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback (https://arxiv.org/abs/2410.19133)
Comments:
          Code in this https URL, MultiPref dataset in this https URL

- **What's New**: 본 연구에서는 언어 모델(LMs)과 사람의 피드백을 결합한 라우팅 프레임워크를 제안하여 인간 주석의 전반적인 비용을 줄이면서도 더 나은 주석 품질을 달성하고자 합니다.

- **Technical Details**: 이 접근법의 핵심은 인간 주석으로 이익을 볼 수 있는 선호 인스턴스( preference instances )를 식별하는 것입니다. 우리는 이 문제를 최적화 문제로 모델링하고, MultiPref라는 새로운 선호 데이터셋을 사용하여 성능 예측 모델을 훈련시킵니다. 이 모델은 인간 및 LM 주석의 조합에서 보상 모델(reward model)의 성능을 예측합니다.

- **Performance Highlights**: 제안된 라우팅 프레임워크를 사용하여 선택된 LM과 인간의 선호 하이브리드 혼합이 독점적으로 사용할 때보다 보상 모델 성능을 향상시키는 것을 보여줍니다. 또한 세 가지 다른 데이터셋을 사용하여 선택적 인간 선호 수집을 시뮬레이션하고, 우리의 방법이 모든 세 데이터셋에 잘 일반화된다는 것을 입증합니다.



### Retrieving Implicit and Explicit Emotional Events Using Large Language Models (https://arxiv.org/abs/2410.19128)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 감정 검색 능력을 심층적으로 평가하며, 특히 암시적(implicit) 및 명시적(explicit) 감정 검색을 다룹니다. 이전 연구에서 다루어지지 않은 감정 검색의 한계를 탐색하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구팀은 LLM의 감정 검색 성능을 검증하기 위해 감독 기반 대조 프로빙(supervised contrastive probing) 방법을 제안했습니다. 이 방법은 주어진 감정에 대해 관련된 감정 사건을 검색하는 능력을 평가하는 데 사용되며, 다른 감정 범주에 대한 다양한 감정 사건을 추출하는 데 중점을 두고 있습니다. 평가 데이터셋은 C3KG 데이터셋에서 수집된 감정 원인 흐름(emotion-cause flow)을 사용합니다.

- **Performance Highlights**: 연구 결과, LLM은 특히 명시적 감정 사건 검색에서 뛰어난 성능을 보였으나, 암시적 감정 사건 검색에서는 한계가 있음을 보여주었습니다. 이러한 결과는 LLM이 감정 검색 작업에서 강점과 한계를 가지고 있음을 시사합니다.



### Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design (https://arxiv.org/abs/2410.19123)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문은 기존의 복잡한 Mixture-of-Experts (MoE) 모델을 사전 훈련된 조밀한 대규모 언어 모델(LLM)에서 더 작은 MoE 모델로 변환하는 새로운 프레임워크인 Read-ME를 제안합니다. 이는 전통적으로 처음부터 MoE를 훈련하는 것보다 비용 효율적인 방식입니다.

- **Technical Details**: Read-ME는 activation sparsity를 활용하여 전문가를 추출하고, 기존의 layer-wise router 디자인의 중복성을 분석하여 pre-gating router를 도입합니다. 이 router는 MoE 구조와 분리되어 있으며, 시스템의 전처리에서 더 효율적인 pre-computing과 lookahead scheduling을 가능하게 하여 전문가를 인식하는 배치 및 캐싱을 향상시켜 줍니다.

- **Performance Highlights**: Read-ME는 MMLU에서 최대 10.1% 개선을 달성하며, 평균 엔드 투 엔드 대기시간을 6.1% 감소시킵니다. 이는 적은 추가 훈련 비용으로도 인기 있는 오픈소스 조밀한 모델보다 월등한 성능을 보여줍니다.



### LLM Tree Search (https://arxiv.org/abs/2410.19117)
- **What's New**: 본 연구는 AlphaGo 패러다임에서 영감을 얻은 새로운 시퀀스 생성 방법을 제안하며, 이를 대형 언어 모델(LLMs)에 적응하는 방법을 탐구합니다. 제안한 접근 방식은 다양한 가능한 완성을 탐색하는 검색 트리를 만드는 것과 모델의 신뢰도를 기반으로 완성을 평가하는 것을 포함합니다.

- **Technical Details**: 이 텍스트 생성 패러다임은 검색 트리, 신뢰도 기반 샘플링 및 반복적 개선을 결합하여 다양하고 고퀄리티의 완성을 생성합니다. 검색 트리는 초기 토큰을 포함하는 루트 노드에서 시작하여 각 노드에 대해 가장 가능성 높은 다음 토큰을 예측합니다. 샘플링 과정에서 각 리프노드는 신뢰도 점수를 부여받아 가중치에 따라 샘플링됩니다. 이 방법은 다양한 가능성을 탐색하면서도 주목할만한 경로를 집중적으로 탐색할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 출력 품질의 향상, 오류 감소 및 창의적이고 다양한 완성 생성 등을 포함하여 LLM의 전반적인 효과성을 높일 수 있는 잠재력을 가지고 있습니다. 또한, 반복적 문제 해결 및 자기 훈련을 지원하여, 최종 결과물의 품질이 계속해서 향상될 것으로 기대됩니다.



### GCoder: Improving Large Language Model for Generalized Graph Problem Solving (https://arxiv.org/abs/2410.19084)
- **What's New**: 본 논문에서는 GCoder라는 코드 기반의 LLM을 소개하며, 이는 일반화된 그래프 계산 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. GraphWild라는 대규모 훈련 데이터셋을 구축하여 다양한 그래프 형식과 알고리즘을 포함시켰습니다.

- **Technical Details**: GCoder는 다단계 훈련 과정(Supervised Fine-Tuning, Reinforcement Learning from Compiler Feedback)을 이용해 모델의 능력을 정제합니다. 계량적 추론의 한계(예: 검증할 수 없는 중간 단계, 긴 거리 추론의 한계, 불만족스러운 일반화)를 극복하기 위해 코드 기반 접근 방식을 도입하여, 더 높은 정확성과 효율성을 달성합니다.

- **Performance Highlights**: 실험 결과, GCoder는 다양한 그래프 계산 문제에서 평균 16.42%의 정확도 향상을 보이며 최신 LLM인 GPT-4o를 초과하는 성능을 발휘합니다. GCoder는 또한 수백만 노드를 포함한 대규모 그래프를 처리하고 다양한 입력 형식을 지원하는 능력을 보여줍니다.



### FISHNET: Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert Swarms, and Task Planning (https://arxiv.org/abs/2410.19727)
Comments:
          Accepted at the 5th ACM International Conference on AI in Finance (ICAIF '24)

- **What's New**: 이번 연구에서는 전통적인 금융 데이터 분석의 방법론을 넘어, FISHNET (Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert swarming, and Task planning)라는 새로운 에이전트 기반 아키텍처를 제안합니다. 이 시스템은 다양한 규제 제출 문서에서 복잡한 재무 정보를 생성하는 데 뛰어난 성능을 발휘합니다.

- **Technical Details**: FISHNET은 Swarming Large Language Models (LLMs)과 Agent-Based Modeling (ABM)을 통합하여 다양한 규제 filings를 효과적으로 분석합니다. 이 시스템은 동시 분석과 데이터 조정, 작업 계획 및 전문가 에이전트를 활용하여 98,034개의 미국 규제 제출 문서에서 금융 통찰력을 생성합니다.

- **Performance Highlights**: FISHNET은 61.8%의 성공률을 기록하고 기존 기법들과 비교 시 현저한 성과를 보여주며, 특히 Retrieval Augmented Generation (RAG)와 Generative Routing과의 성능 비교에서 우수한 결과를 자랑합니다.



### IPPON: Common Sense Guided Informative Path Planning for Object Goal Navigation (https://arxiv.org/abs/2410.19697)
- **What's New**: 본 논문은 3D 객체 확률 맵핑 및 유용한 경로 계획을 위한 IPPON이라는 새로운 접근 방식을 도입합니다. 이 방법은 의미적 세분화 및 베이지안 필터를 활용하여 탐색 중인 객체의 확률을 계산하고, 대규모 언어 모델(Large Language Model)에서의 상식 근거를 바탕으로 탐색을 안내합니다.

- **Technical Details**: IPPON은 온라인 정보 경로 계획(Informative Path Planning, IPP) 프레임워크에 기반하여 객체 목표 탐색 문제를 해결합니다. 이 방법은 각 시점에서 OOI(Object of Interest)를 포함할 가능성이 높은 voxel들의 집계 확률을 이득으로 정의하여, 3D 객체 확률 맵핑 알고리즘을 통해 일반 객체와 OOI의 확률을 추정합니다.

- **Performance Highlights**: 본 연구의 제안된 방법은 Habitat ObjectNav Challenge 2023에서 성공 비율 및 경로 길이에 가중치(Success weighted by Path Length, SPL) 측정에서 20% 이상 다른 방법들보다 우수한 성능을 기록하였으며, 실제 로봇에서도 효과를 검증하였습니다.



### Mirror Matrix on the Wall: coding and vector notation as tools for introspection (https://arxiv.org/abs/2410.19549)
Comments:
          22 pages, 1 figure (3 subfigures)

- **What's New**: 이 논문은 GNU Octave의 벡터 표기법의 중요성을 탐구하며, 이를 통해 프로그래밍 언어가 수학적 표기법과 얼마나 잘 정렬될 수 있는지를 보여줍니다.

- **Technical Details**: 벡터 표기법은 인덱싱(indexing), 브로드캐스팅(broadcasting), 함수 핸들(function handles)과 같은 기본 개념을 사용하여 코드를 더 효율적이고 우아하게 만드는 데 기여합니다. GNU Octave는 이러한 기능을 통해 수학자, 과학자 및 엔지니어들이 복잡한 문제를 보다 효과적이고 직관적으로 표현하고 해결할 수 있도록 돕습니다.

- **Performance Highlights**: GNU Octave의 벡터 프로그래밍 패러다임을 통해 절차적 프로그래밍 방법에서 벗어나 더 높은 수준의 추상화와 간결함을 제공하며, 코드의 효율성이 크게 향상됩니다.



### Revealing and Reducing Gender Biases in Vision and Language Assistants (VLAs) (https://arxiv.org/abs/2410.19314)
- **What's New**: 본 연구에서는 22개의 인기 있는 오픈 소스 Vision-Language Assistants (VLAs)에서 성별 편향(gender bias)을 평가하였습니다. VLAs는 성격 특성(personality traits), 기술(skills), 직업(occupations) 측면에서 인간 편향을 재현하며, 성별에 따른 기술 및 긍정적인 성격 특성의 할당에서 불균형이 나타났습니다.

- **Technical Details**: VLAs의 성별 편향을 조사하기 위해 FairFace, MIAP, Phase, PATA 데이터셋에서 성별 균형이 잡힌 이미지 서브셋을 만든 후, Visual Question Answering (VQA) 형식으로 VLAs에 이미지를 제시하여 응답을 분석했습니다. 성별 편향 제거를 위한 다수의 디바이싱(debiasing) 기법, 즉 Finetuning, Prompt Tuning 및 Pruning 등을 적용하여 성별 편향을 줄이면서도 다운스트림 작업의 정확성을 유지하는 방법을 모색했습니다.

- **Performance Highlights**: Finetuning 방법이 성별 편향 감소와 성능 유지 간의 최적의 균형을 이룬다는 점이 확인되었습니다. 전반적인 성능을 유지하면서도 성별 편향을 효과적으로 줄이는 다양한 실험을 통해 이를 입증하였습니다.



### Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors (https://arxiv.org/abs/2410.19230)
Comments:
          26 pages

- **What's New**: 이번 연구에서는 사람의 글쓰기와 유사한 결과를 생성하는 대형 언어 모델(LLMs)의 탐지기를 스트레스 테스트하기 위해 프록시 공격(proxy attack) 전략을 소개합니다.

- **Technical Details**: 이 연구는 강화 학습(reinforcement learning, RL)으로 미세 조정된 인간화된 소형 언어 모델(small language model, SLM)을 사용하여 소스 모델을 공격하고 텍스트 생성 과정의 해독 단계에서 이를 활용합니다.

- **Performance Highlights**: 프록시 공격 전략은 Llama2-13B, Llama3-70B, Mixtral-8*7B와 같은 오픈 소스를 활용하여 여러 데이터 세트에서 주도적인 탐지기를 속이는 데 성공했으며, 평균 AUROC 점수가 70.4% 감소했습니다.



### Inference time LLM alignment in single and multidomain preference spectrum (https://arxiv.org/abs/2410.19206)
- **What's New**: 본 논문에서는 선호 조정(Preference Tuning)을 위한 새로운 접근법인 Inference-Time Model Alignment을 소개합니다. 이는 사용자 맞춤형 출력을 제공하면서도 연산 비용을 절감할 수 있는 Alignment Vectors (AVs)라는 개념을 활용합니다.

- **Technical Details**: 이 연구는 모델 편집(Model Editing) 기술을 통해 Alignment Vectors를 도입하여 LLM의 반응 수준을 세밀하게 조정할 수 있는 방법을 제시합니다. AV는 기본 모델과 조정된 모델 간의 차이를 계산하여 나타내며, 이를 통해 단순한 선형 연산만으로도 동적으로 모델 동작을 조정할 수 있습니다.

- **Performance Highlights**: 이 새로운 접근법은 인퍼런스 시간 동안 사용자가 출력을 쉽게 조정할 수 있도록 하여, 전통적인 프롬프트 엔지니어링 기법에 비해 인퍼런스 비용을 절반으로 줄일 수 있습니다. 특히 AV는 다양한 도메인에서 전이 가능하며, 모델의 견고성을 높이는 데 기여합니다.



### Making Social Platforms Accessible: Emotion-Aware Speech Generation with Integrated Text Analysis (https://arxiv.org/abs/2410.19199)
- **What's New**: 이 연구에서는 시각 장애인(BVIP)과 저소득 환경에서의 접근성 문제를 해결하기 위해 정서 인식 Text-to-Speech(TTS) 시스템을 제안합니다. 기존 TTS 시스템의 문제점을 보완하고 감정적인 음성을 생성하는 데 초점을 맞춘 혁신적인 기법을 개발하였습니다.

- **Technical Details**: 제안된 시스템은 텍스트 입력에서 전달되는 감정을 도출하고, 자연스럽고 표현력이 풍부한 음성을 합성하는 end-to-end Context-aware TTS 시스템입니다. 감정 예측을 위한 transformer 기반 모델을 활용하고, 음성 합성을 위한 FastSpeech2 아키텍처를 수정하여 다중 화자 및 다중 감정을 처리할 수 있도록 하였습니다.

- **Performance Highlights**: 제안된 시스템은 현실 세계의 음성 패턴을 복제하고 자연어 처리(NLP) 기술과 음성 합성 기법을 통합하여 실시간 접근성 애플리케이션에 적합한 competitive inference time 성능을 보여줍니다. 시스템 성능은 기존의 최첨단 TTS 모델들과 비교하였을 때 매우 경쟁력이 있습니다.



### Indication Finding: a novel use case for representation learning (https://arxiv.org/abs/2410.19174)
- **What's New**: 이번 논문에서는 자연어 처리(Natural Language Processing) 및 실제 데이터(Real-World Data)를 활용하여 작용 메커니즘(Mechanism of Action, MoA)에 대한 새로운 적응증 발견 접근 방식을 제시합니다. 기존의 임상 실험이나 의견 리더들의 지침에 기대지 않고, 임상 이벤트의 표현을 학습(Representation Learning)하여 임상에 적합한 적응증을 우선 순위화합니다.

- **Technical Details**: 이 연구에서는 SPPMI(Symmetric Positive Pointwise Mutual Information) 방법을 사용하여 임상 이벤트의 임베딩을 생성합니다. 이를 통해 실제 데이터의 상관성을 기반으로 새로운 치료 적응증을 탐색할 수 있는 가능성을 열었습니다. 또한, 결과의 신뢰성을 보장하기 위해 임베딩 품질 평가 프레임워크를 개발하였습니다.

- **Performance Highlights**: 이 새로운 방법론은 임상 시험에서 효과가 입증된 진단을 높은 순위로 평가하며, 기존의 접근 방식에 비해 더 나은 성공 가능성을 제공하는 것으로 나타났습니다. 구체적으로, 연구에서 anti-IL-17A의 케이스 스터디를 가지고 실제 적용 결과를 통해 우수한 성능을 입증하였습니다.



### MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark (https://arxiv.org/abs/2410.19168)
Comments:
          Project Website: this https URL

- **What's New**: MMAU는 다중 모드 오디오 이해(audio understanding) 모델을 평가하기 위한 혁신적인 벤치마크로, 전문 지식과 복잡한 추론 능력을 요구하는 작업을 포함하고 있습니다. 이 벤치마크는 10,000개의 오디오 클립과 관련된 자연어 질문 및 답변을 포함하며, 27개의 독특한 기술을 요구하는 제작물들로 구성되어 있습니다. 이는 기존 벤치마크들과는 차별화되는 점으로, 전문가들이 수행하는 작업과 유사한 과제를 모델에게 도전하게 합니다. 

- **Technical Details**: MMAU는 미각(adaptive) 오디오 인식(Understanding) 및 추론(Reasoning)을 평가하기 위해 설계되었습니다. 이 벤치마크는 오디오 클립에 초점을 맞추고 있으며 정보 추출(information extraction) 및 추론 질문을 포함하여 모델들이 27개의 고유 기술을 시연하도록 요구합니다. 다중 음성 역할 매핑(multi-speaker role mapping), 감정 변화 감지(emotional shift detection), 및 시간적 음향 이벤트 분석(temporal acoustic event analysis)과 같은 고급 추론이 요구되는 작업이 포함됩니다. 이를 통해 오디오 콘텐츠와 텍스트를 공동으로 처리하고, 적절한 지식을 회상하며, 복잡한 추론을 통해 문제를 해결해야 합니다.

- **Performance Highlights**: 현재 18개의 오픈 소스 및 상용 (Large) Audio-Language 모델들이 MMAU에서 평가되었으며, Gemini Pro v1.5는 52.97%의 정확도, Qwen2-Audio는 52.50%에 불과하여 많은 향상 여지가 있음을 보여줍니다. 이러한 결과는 현재 오디오 이해 모델들이 인간이 쉽게 수행하는 과제를 해결하는 데 어려움을 겪고 있다는 것을 시사합니다.



### Adversarial Attacks on Large Language Models Using Regularized Relaxation (https://arxiv.org/abs/2410.19160)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문에서는 기존의 적대적 공격 기법이 가진 한계를 극복하기 위해, 정규화된 기울기를 활용한 새로운 적대적 공격 기법을 제안합니다. 이 방법은 연속적인 최적화 (continuous optimization) 기법을 사용하며, LLMs에서 공격 성공률을 크게 향상시킵니다.

- **Technical Details**: 제안된 기법은 연속 임베딩 공간 (continuous embedding space)에서 relaxed token embeddings를 최적화하고, 이를 통해 유효한 디스크리트 토큰 (discrete tokens)을 생성할 수 있습니다. 기존의 greedy coordinate gradient-based 방식에 비해, 두 배 이상의 속도로 작동하며, 다섯 개의 최신 LLMs에서 효과성을 입증했습니다.

- **Performance Highlights**: 새로운 공격 기법은 공격 성공률이 높으며, 기존의 최첨단 공격 기법들에 비해 효율성과 효과성에서 우수한 성능을 보여줍니다. 그 결과, 저자들은 제안하는 기법이 특정 유사한 결과를 생성하는 데 필요한 계산 비용을 최소화할 수 있음을 강조합니다.



### A Test of Time: Predicting the Sustainable Success of Online Collaboration in Wikipedia (https://arxiv.org/abs/2410.19150)
- **What's New**: 본 연구에서는 온라인 협업의 지속 가능한 성공(Sustainable Success)을 측정하는 새로운 지표를 제안합니다. 이는 협업이 시간에 따라 품질을 유지할 수 있는 능력을 평가합니다. 특히, 4만 개 이상의 위키백과 기사 데이터를 포함하는 SustainPedia 데이터셋을 사용하여 성공적인 협업의 지속 가능성을 예측하기 위해 머신러닝 모델을 개발했습니다.

- **Technical Details**: SustainPedia 데이터셋은 위키백과의 4만 개 이상의 기사와 각 기사에 대한 지속 가능한 성공 지표 및 300개 이상의 설명적 특성(예: 수정 이력, 사용자 경험, 팀 구성)으로 이루어져 있습니다. 또한, 최고 성능의 머신러닝 모델은 평균 AU-ROC 점수 0.88을 달성했습니다.

- **Performance Highlights**: 연구 결과, 기사가 고품질로 인정받기까지 걸리는 시간이 길어질수록 그 상태를 지속할 가능성이 높아진다는 발견을 했습니다. 사용자 경험은 지속 가능성을 예측하는 가장 중요한 요소로 나타났습니다. 이러한 통찰은 위키백과를 넘어서는 보다 넓은 집단 행동에도 적용될 수 있습니다.



### Visual Text Matters: Improving Text-KVQA with Visual Text Entity Knowledge-aware Large Multimodal Assistan (https://arxiv.org/abs/2410.19144)
Comments:
          Accepted to EMNLP (Main) 2024

- **What's New**: 이 연구는 최신 대형 다중모달 모델(large multimodal models, LMMs)의 발전을 기반으로 텍스트 기반 시각 질문 응답(knowledge-aware text-based visual question answering, Text-KVQA)을 재조명하였습니다. 주요 기여는 VisTEL이라는 시각 텍스트 엔티티 링크 모듈과 KaLMA라는 지식 인식 대형 다중모달 어시스턴트를 제안하여, 보다 정확한 답변을 제공하는 것입니다.

- **Technical Details**: VisTEL은 이미지 내 시각 텍스트 엔티티를 인식하고 이를 지식 기반에 연결하는 방법론으로, 최신 OCR 기반 텍스트 인식 엔진과 LMM의 기능을 결합하여 텍스트와 시각적 맥락을 동시에 활용하여 링크를 수행합니다. KaLMA는 이러한 정보로 LMM을 보강하여 시각 질문에 대한 응답 정확도를 높입니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, 제안한 방법론은 Text-KVQA에 대하여 기존 최상위 접근 방식에 비해 23.3%의 성능 개선을 이루어냈으며, 장면, 책 표지 및 영화 포스터 데이터셋의 각각 18.2%, 19.6%, 32.2% 개선을 보여주었습니다. 이로써 새로운 최첨단 결과를 달성하였습니다.



### RSA-Control: A Pragmatics-Grounded Lightweight Controllable Text Generation Framework (https://arxiv.org/abs/2410.19109)
Comments:
          Accepted to EMNLP 2024 (main conference)

- **What's New**: 본 논문에서는 RSA-Control이라는 훈련 없는 (training-free) 제어 가능한 텍스트 생성 프레임워크를 소개합니다. 이 프레임워크는 발화자(발언하는 사람)와 청자(듣는 사람) 간의 반복적 추론을 통해 정보의 해석을 조정하여 특정 속성(attribute)이 청자에 의해 정확하게 인식될 가능성을 높입니다.

- **Technical Details**: RSA-Control은 주어진 컨텍스트에 따라 제어 강도를 자동으로 조정할 수 있는 자기 조정 합리성 매개변수를 도입합니다. 이 메커니즘은 PLMs(Pre-trained Language Models)의 의사소통 과정을 명확히 하고, 발화자가 원하는 속성을 반영하여 청자의 이해를 보장합니다. RSA-Control은 발화자와 청자 모듈의 상호작용을 모델링하여 설명적이고 실용적인 발화를 생성합니다.

- **Performance Highlights**: RSA-Control은 두 가지 작업 유형과 두 가지 유형의 PLMs를 활용한 실험에서 강력한 속성 제어를 실현하면서 언어 유창성과 내용 일관성을 유지했습니다. 실험 결과, GPT2 모델로 오픈 엔드 생성에서 독성(toxicity) 및 고정관념(bias)을 줄이고, Llama-2-7b-chat 모델로 가독성에 초점을 맞춘 요약 생성을 성공적으로 수행했습니다.



### Watermarking Large Language Models and the Generated Content: Opportunities and Challenges (https://arxiv.org/abs/2410.19096)
Comments:
          invited paper to Asilomar Conference on Signals, Systems, and Computers

- **What's New**: 본 논문은 생성적 대규모 언어 모델(LLMs)의 저작권 보호를 위한 워터마킹(watermarking) 기법에 대한 새로운 기회와 도전 과제를 분석하였습니다. LLMs의 사용이 증가하면서, 지적 재산권 침해 및 기계 생성 허위 정보 문제에 대한 해결책으로서 워터마킹 기술의 필요성이 더욱 강조되고 있습니다.

- **Technical Details**: LLMs에 대한 워터마킹 기술은 다양한 위협 모델(threat model) 및 사용 사례(scenario)에 따라 구현됩니다. 특히, LLM이 생성한 콘텐츠를 위한 워터마킹 기법을 조사하고, 이러한 기법의 효율성(efficiency) 및 강인성(robustness)을 평가합니다. 효율성을 높이기 위해 하드웨어 가속(hardware acceleration)을 포함한 다양한 방법론을 탐색하며, 이론적 기준으로는 효과성(effectiveness), 충실도(fidelity), 효율성(efficiency), 강인성(robustness), 탐지 불가능성(undetectability)을 설정합니다.

- **Performance Highlights**: 연구 결과, 다양한 워터마킹 알고리즘과 도메인별 애플리케이션에 대한 분석을 바탕으로, 현재의 한계점을 논의하며 향후 연구 방향을 제시하고 있습니다. 특히, 손실 없는 성능을 유지하면서도 워터마킹 과정의 속도를 높이는 방안으로 소프트웨어-하드웨어 공동 설계(software-hardware co-design) 접근법을 제안합니다.



### Infogent: An Agent-Based Framework for Web Information Aggregation (https://arxiv.org/abs/2410.19054)
Comments:
          Preprint

- **What's New**: 웹 내 정보 집합을 위한 새로운 모듈형 프레임워크 Infogent를 소개합니다. 이 프레임워크는 내비게이터(Navigator), 추출기(Extractor), 집계기(Aggregator)라는 세 가지 주요 구성 요소로 구성되어 있습니다.

- **Technical Details**: Infogent는 두 가지 정보 접근 시나리오인 직접 API 기반 접근(Direct API-Driven Access)과 인터랙티브 비주얼 접근(Interactive Visual Access)을 지원합니다. 내비게이터(Navigator)는 웹을 탐색하고 적절한 웹사이트를 찾으며, 추출기(Extractor)는 선정된 웹페이지에서 관련 정보를 찾고, 마지막으로 집계기(Aggregator)는 추출된 정보를 선택적으로 유지하고 가장 최종 집계 결과에 포함할 내용을 결정합니다.

- **Performance Highlights**: 실험 결과, Infogent는 FRAMES에서 기존의 다중 에이전트 검색 프레임워크 대비 7% 우수한 성능을 보였고, AssistantBench에서 기존 정보 탐색 웹 에이전트보다 4.3% 더 개선된 성과를 얻었습니다.



### O1 Replication Journey: A Strategic Progress Report -- Part 1 (https://arxiv.org/abs/2410.18982)
- **What's New**: 이 논문에서는 OpenAI의 O1 모델에 대한 투명하고 실시간으로 진행되는 복제 탐험인 O1 Replication Journey를 소개합니다. 이 연구에서는 현업 AI 연구의 여러 문제를 해결하고, 시도와 실패를 기록하여 오픈 사이언스를 촉진하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 ‘journey learning’ 패러다임을 제안하며, 모델이 단순한 단축키를 배우는 것이 아니라 Trial and Error, Reflection 및 Backtracking을 포괄하는 완전한 탐색 과정을 학습하도록 장려합니다. 이 방법론은 327개의 훈련 샘플만 가지고도 기존의 supervised learning보다 8% 이상 성능 향상을 보여주었습니다.

- **Performance Highlights**: O1 Replication Journey에서는 O1 모델의 특성을 복제하고 학습하는 과정에서 성공적인 사례와 실패 사례를 모두 기록하여 오픈 사이언스를 촉진하며, AI 연구 커뮤니티에 중요한 기여를 할 것으로 기대합니다.



### Stick-breaking Attention (https://arxiv.org/abs/2410.17980)
- **What's New**: 본 논문에서는 기존의 softmax 기반 self-attention 메커니즘을 대체할 수 있는 stick-breaking attention 메커니즘을 제안합니다. 이 접근법은 재귀적이지 않으며 최근성 편향을 자연스럽게 통합합니다.

- **Technical Details**: Stick-breaking attention은 각 토큰에 대해 break point $eta_{i,j}$를 계산하여 현재 토큰에 allocation하는 비율을 결정합니다. 이 과정을 반복하여 attention 가중치의 시퀀스를 생성합니다. 구현 시, numerically stable stick-breaking attention을 적용하고 Flash Attention을 조정하여 이 메커니즘을 수용하도록 합니다.

- **Performance Highlights**: Stick-breaking attention은 길이 일반화(length generalisation)와 여러 다운스트림 작업에서 기존 softmax+RoPE 시스템과 경쟁력 있는 성과를 보였습니다. 특히, $2^{11}$의 context window로 학습된 모델이 $2^{14}$에서 덜 혼란스러운 퍼플렉서티(perplexity) 향상을 보여주었습니다.



New uploads on arXiv(cs.IR)

### pEBR: A Probabilistic Approach to Embedding Based Retrieva (https://arxiv.org/abs/2410.19349)
- **What's New**: 본 논문은 임베딩 기반 검색(embedding based retrieval)에서 전통적인 빈도주의(frequentist) 접근 방식을 벗어나 확률적(probabilistic) 접근 방식을 제안합니다. 특히, 코사인 유사도(threshold)를 유동적으로 계산할 수 있는 기법을 도입하여 다양한 쿼리에 대해 동적인 검색이 가능하도록 하였습니다.

- **Technical Details**: 제안하는 방법은 각 쿼리에 대한 항목 분포를 학습하여, 확률적 누적 분포 함수(probabilistic cumulative distribution function, CDF) 값을 활용하여 코사인 유사도(threshold)를 동적으로 계산합니다. 이로써 고정된 개수의 항목이나 고정된 코사인 임계값을 사용하는 기존 방법보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안한 방법(pEBR)은 검색 정밀도(precision)와 재현율(recall) 모두를 유의미하게 개선했습니다. 또한, ablation study를 통해 두 가지 쿼리(헤드 쿼리와 테일 쿼리) 간의 차이를 잘 포착하는 확률적 접근 방식의 효과를 검증하였습니다.



### TEARS: Textual Representations for Scrutable Recommendations (https://arxiv.org/abs/2410.19302)
- **What's New**: TEARS(TExtuAl Representations for Scrutable Recommendations) 시스템은 사용자의 관심사를 높은 차원 잠재 임베딩 대신 자연어 텍스트로 표현하여 투명성을 제공하고 사용자가 직접 수정할 수 있도록 합니다. 이를 통해 추천 시스템의 투명성과 사용자의 제어력을 강화합니다.

- **Technical Details**: TEARS는 두 개의 인코더를 사용하여 전통적인 블랙박스 모델이 처리한 역사적 상호작용을 기반으로 숫자 블랙박스 임베딩을 생성하고, 사용자가 편집할 수 있는 자연어 요약을 수집하여 요약 기반 임베딩으로 변환합니다. 이 두 임베딩을 최적 운송(optimal transport) 절차를 통해 정렬하고, 혼합 계수를 통해 추천을 조정할 수 있습니다.

- **Performance Highlights**: TEARS는 현대의 LLM(대규모 언어 모델)을 통해 생성된 사용자 요약을 사용하여 추천 성능을 개선하고 있으며, 사용자 요약을 편집함으로써 추천 결과를 효과적으로 조정할 수 있는 유연성을 제공합니다. 세 가지 시뮬레이션된 사용자 작업을 통해 TEARS의 제어력이 검증되었으며, 기존 VAE(변분 오토인코더) 모델들의 성능을 초과하는 결과를 나타냈습니다.



### Learning ID-free Item Representation with Token Crossing for Multimodal Recommendation (https://arxiv.org/abs/2410.19276)
Comments:
          11 pages,6 figures

- **What's New**: 현재 멀티모달 추천 모델들은 멀티모달 정보를 효과적으로 활용하고 있지만, ID 임베딩(ID embeddings)에 의존함으로써 성능 저하를 겪고 있습니다. 본 논문에서는 MOTOR라는 ID-free 멀티모달 토큰 표현 기법을 제안하여, 각 아이템을 학습 가능한 멀티모달 토큰으로 표현하고 이를 공유 토큰을 통해 연결합니다.

- **Technical Details**: MOTOR는 제품 양자화를 통해 각 아이템의 멀티모달 특성을 이산 토큰 ID로 변환하며, 이러한 토큰 ID에 해당하는 임베딩을 암묵적 아이템 특성으로 해석합니다. Token Cross Network(TCN)을 통해 이 토큰 간의 상호작용 패턴을 캡처하여, 기존의 ID 기반 멀티모달 추천 모델을 ID-free 시스템으로 변환합니다.

- **Performance Highlights**: MOTOR는 9개의 주요 모델에 대한 광범위한 실험을 통해 좋은 성능 향상을 보여주며, 특히 차가운 시작 문제(cold-start)와 긴 꼬리 아이템을 다루는 데 효과적임을 입증했습니다. MOTOR는 ID 임베딩 없이도 추천 시스템의 성능을 상당히 향상시키는 것을 목표로 하며, 오픈소스 프레임워크로서 다양한 하위 멀티모달 추천 모델과 호환됩니다.



### Taxonomy-guided Semantic Indexing for Academic Paper Search (https://arxiv.org/abs/2410.19218)
Comments:
          EMNLP'24

- **What's New**: 이번 연구에서는 학술 논문 검색에서의 효율성을 높이기 위해 Taxonomy-guided Semantic Indexing (TaxoIndex) 프레임워크를 제안합니다. 이는 핵심 개념을 추출하고 이를 학술 분류 체계에 기반한 의미 색인으로 구성하여 쿼리와 문서 간의 효과적인 학술 개념 매칭을 가능하게 합니다.

- **Technical Details**: TaxoIndex는 자연어 처리(NLP) 및 기계 학습(ML) 기법을 사용하여 논문에서 핵심 개념을 추출합니다. 이 프레임워크는 주제 수준과 구절 수준의 두 가지 세부 사항으로 분류된 의미 색인을 활용하여 문서를 보다 정확하게 표현합니다. 또한, 인덱스 학습(index learning)을 통해 서로 다른 용어로 표현된 쿼리와 관련된 개념을 자동으로 찾도록 모델을 훈련합니다.

- **Performance Highlights**: Extensive experiments show that TaxoIndex는 제한된 학습 데이터에도 불구하고 매우 효과적인 성능 개선을 가져옵니다. 기존의 Dense retrievers 보다 향상된 검색 품질을 제공하며, 특정 학술 개념 매칭의 해석 가능성을 개선하는 데에도 큰 기여를 합니다.



### FISHNET: Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert Swarms, and Task Planning (https://arxiv.org/abs/2410.19727)
Comments:
          Accepted at the 5th ACM International Conference on AI in Finance (ICAIF '24)

- **What's New**: 이번 연구에서는 전통적인 금융 데이터 분석의 방법론을 넘어, FISHNET (Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert swarming, and Task planning)라는 새로운 에이전트 기반 아키텍처를 제안합니다. 이 시스템은 다양한 규제 제출 문서에서 복잡한 재무 정보를 생성하는 데 뛰어난 성능을 발휘합니다.

- **Technical Details**: FISHNET은 Swarming Large Language Models (LLMs)과 Agent-Based Modeling (ABM)을 통합하여 다양한 규제 filings를 효과적으로 분석합니다. 이 시스템은 동시 분석과 데이터 조정, 작업 계획 및 전문가 에이전트를 활용하여 98,034개의 미국 규제 제출 문서에서 금융 통찰력을 생성합니다.

- **Performance Highlights**: FISHNET은 61.8%의 성공률을 기록하고 기존 기법들과 비교 시 현저한 성과를 보여주며, 특히 Retrieval Augmented Generation (RAG)와 Generative Routing과의 성능 비교에서 우수한 결과를 자랑합니다.



### AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions for Conversational Search with LLMs (https://arxiv.org/abs/2410.19692)
Comments:
          23 pages

- **What's New**: AGENT-CQ라는 새로운 프레임워크가 발표되었습니다. 이 시스템은 대화형 검색 시스템에서 다양한 기다리는 질문(clarifying questions)을 자동 생성하고 평가하는 기능을 제공합니다. 기존 방법들의 한계를 극복하기 위해 LLM 기반(end-to-end LLM-based) 접근 방식을 채택했습니다.

- **Technical Details**: AGENT-CQ는 두 단계로 구성되어 있습니다. 첫 번째 단계는 생성 단계(generation stage)로, LLM 프롬프트(prompting) 전략을 사용하여 기다리는 질문을 생성합니다. 두 번째 단계는 평가 단계(evaluation stage)로, CrowdLLM을 사용하여 다수의 LLM 인스턴스를 통해 생성된 질문과 답변의 품질을 종합적인 품질 메트릭(classic quality metrics)에 따라 평가합니다.

- **Performance Highlights**: ClariQ 데이터셋을 기반으로 한 실험에서 CrowdLLM의 평가가 질문과 답변의 품질을 높이는 데 매우 효과적임을 입증하였습니다. AGENT-CQ의 생성 단계는 다양한 질문과 답변 품질 측면에서 기준선(baselines)을 지속적으로 능가했습니다. 검색 기반 평가에서는 LLM이 생성한 질문이 인간이 생성한 질문에 비해 BM25 및 크로스 인코더(cross-encoder) 모델 모두에서 검색 효과를 크게 향상시키는 것으로 나타났습니다.



### Knowledge Graph Enhanced Language Agents for Recommendation (https://arxiv.org/abs/2410.19627)
- **What's New**: 최근 언어 에이전트는 추천 시스템에서 인간 행동 및 사용자-아이템(interaction) 상호작용을 시뮬레이션하는 데 사용되고 있습니다. 하지만 기존의 시뮬레이션은 사용자와 아이템 간의 관계를 이해하지 못해 부정확한 사용자 프로필과 비효율적인 추천을 초래합니다.

- **Technical Details**: 이 연구에서는 사용자와 아이템 간의 광범위하고 신뢰할 수 있는 관계를 포함하는 지식 그래프(Knowledge Graphs, KGs)의 유용성을 탐구합니다. 우리의 주요 통찰력은 KG의 경로들이 사용자와 아이템 간의 복잡한 관계를 포착하여 사용자 선호의 근본적인 이유를 끌어내고 사용자 프로필을 풍부하게 만든다는 것입니다. 우리는 KGLA(지식 그래프 강화 언어 에이전트)라는 프레임워크를 제안하여 추천 시스템을 위한 언어 에이전트와 KG를 통합합니다.

- **Performance Highlights**: 실험 결과, KGLA는 추천 성능을 상당히 향상시켰으며(세 가지 널리 사용되는 벤치마크 중 NDCG@1에서 33%-95%의 증가) 이전의 최적 기준 방법과 비교하여 뚜렷한 성과를 보였습니다.



### Sentiment-Driven Community Detection in a Network of Perfume Preferences (https://arxiv.org/abs/2410.19177)
- **What's New**: 이 연구는 향수 네트워크에서 커뮤니티 탐지(community detection)를 적용하여 사용자 선호에 기반한 향수 클러스터를 분류하는 새로운 접근법을 제시합니다.

- **Technical Details**: Persian 리테일 플랫폼 "Atrafshan"에서 사용자 리뷰를 기반으로 하여 사용자와 향수를 노드로, 긍정적인 댓글을 엣지로 하는 이분 bipartite 네트워크를 구축했습니다. 향수 동선호 네트워크(Perfume Co-Preference Network)로 변환하여 같은 사용자가 좋아하는 향수를 연결했습니다. 이 연구에서는 이모티콘(emojis)과 사용자 투표 시스템을 통합하여 감정 분석(sentiment analysis)의 정확도를 높였습니다. 엣지 가중치는 60:40 비율로 인접 값(adjacency values)과 사용자 평가(user ratings)를 조합하여 조정되었습니다.

- **Performance Highlights**: 이 연구는 향수 그룹화의 정확성을 높이며, 향수 네트워크의 커뮤니티 탐지 기법을 혁신적으로 사용하여 소비자 선호에 대한 새로운 통찰력을 제공합니다. 향수 추천 및 마케팅 전략 최적화를 위한 실행 가능한 인사이트를 제공합니다.



New uploads on arXiv(cs.CV)

### Model merging with SVD to tie the Knots (https://arxiv.org/abs/2410.19735)
- **What's New**: 최근 LoRA 모델병합 기법의 한계를 극복하기 위해 KnOTS를 제안합니다. KnOTS는 다양한 LoRA 모델의 작업 업데이트를 정렬된 공간으로 변환해 기존의 모델 병합 기법을 효과적으로 사용 가능하도록 합니다.

- **Technical Details**: KnOTS는 특이값 분해(SVD)를 활용하여 LoRA 모델의 작업 업데이트를 공유 공간으로 변환합니다. 이 기법은 서로 다른 LoRA 모델 간의 정렬을 향상시켜 모델 병합의 효과를 증가시킵니다. 또한, 새로운 벤치마크를 도입하여 병합된 모델의 일반성(e.g., general models)을 평가합니다.

- **Performance Highlights**: KnOTS는 여러 비전 및 언어 벤치마크에서 최대 4.3%의 성능 향상을 달성했습니다. 새롭게 만든 평가 설정에서는, 모든 데이터셋의 입력과 레이블을 통합하여 어떤 이미지를 모든 가능한 레이블로 올바르게 분류할 수 있는 모델의 능력을 측정했습니다.



### TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning (https://arxiv.org/abs/2410.19702)
- **What's New**: 이번 논문에서는 짧은 비디오 이해에 뛰어난 성능을 보인 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 장기 비디오 이해에 적용하기 위해 TimeSuite라는 새로운 디자인을 제안합니다. 이 프레임워크는 긴 비디오 시퀀스를 처리하는 간단하면서도 효율적인 방법과 고품질 비디오 데이터셋, 그리고 기존 질문 답변 형식(QA format)에서 그라운딩(supervision) 감독을 명시적으로 포함하는 훈련 과제를 포함합니다.

- **Technical Details**: TimeSuite는 긴 비디오 이해를 위해 설계되었으며, 다음과 같은 주요 요소를 포함합니다: 1) Token Shuffle 기법을 통한 비주얼 토큰 압축, 2) Temporal Adaptive Position Encoding (TAPE)을 사용한 시각적 표현의 시간 인식 향상, 3) Temporal Grounded Caption 훈련 과제를 통한 세부적인 비디오 설명 생성. 이 외에도 TimePro라는 복합적인 그라운딩 중심 인스트럭션 튜닝 데이터셋이 포함되어 있습니다.

- **Performance Highlights**: TimeSuite는 긴 비디오 이해에서 Egoschema와 VideoMME 기준에서 각각 5.6%와 6.8% 성능 향상을 달성하였으며, VideoChat-T는 다른 최신 MLLMs에 비해 뛰어난 제로샷(zero-shot) 시간 그라운딩 능력을 보여 주었습니다. 실험 결과, VideoChat-T는 전통적인 전문가 모델과 동등한 성능을 기록했습니다.



### Deep Learning for Classification of Inflammatory Bowel Disease Activity in Whole Slide Images of Colonic Histopathology (https://arxiv.org/abs/2410.19690)
- **What's New**: 이번 연구에서는 염증성 장 질환(IBD) 활동을 효과적으로 분류할 수 있는 딥 러닝 모델을 개발하였다. 이 모델은 전통적인 병리학적 평가 방식에서의 자원 제약 및 관찰자 간 변동성을 극복하는 데 도움을 줄 수 있다.

- **Technical Details**: 본 연구에서는 2018년과 2019년에 Dartmouth-Hitchcock Medical Center에서 치료받은 636명의 환자로부터 얻은 2,077개의 전 슬라이드 이미지(Whole Slide Images, WSIs)를 사용하였다. 보드 인증을 받은 위장관 병리학자들이 WSIs를 비활동, 경미한 활동, 중등도 활동, 심각한 활동의 네 가지 등급으로 분류하였다. 모델은 Transformer 기반 구조로 개발되었으며, 다섯 번의 교차 검증(five-fold cross-validation)을 통해 검증되었다. HoVerNet을 사용하여 활동 등급에 따른 호중구(neutrophil) 분포를 분석하였고, 주목도 맵(attention maps)도 생성되었다.

- **Performance Highlights**: 모델의 IBD 활동 분류 성능은 곡선 아래 면적(Area Under the Curve)에서 0.871(95% 신뢰 구간: 0.860-0.883), 정밀도(precision)에서 0.695(95% CI: 0.674-0.715), 재현율(recall)에서 0.697(95% CI: 0.678-0.716), F1-점수(F1-score)에서 0.695(95% CI: 0.674-0.714)으로 나타났다. 또한 호중구 분포는 활동 등급 간에 유의미한 차이를 보였고, 병리학자에 의한 주목도 맵의 질적 평가는 해석 가능성 개선의 잠재력을 시사한다.



### Inferring Neural Signed Distance Functions by Overfitting on Single Noisy Point Clouds through Finetuning Data-Driven based Priors (https://arxiv.org/abs/2410.19680)
Comments:
          Accepted by NeurlPS 2024. Project page: this https URL

- **What's New**: 본 연구에서는 노이즈가 많고 어려운 상황에서도 더 나은 일반화(generalization), 빠른 추론(inference), 그리고 높은 정확성(accuracy)을 달성하기 위해 데이터 기반(data-driven) 및 과적합(overfitting) 기반 방법의 장점을 결합한 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 단일 포인트 클라우드(point cloud)로부터 SDF를 학습하는데, 이를 위해 데이터 기반 우선(datar-driven prior)을 미세 조정(finetune)하며, 이를 위해 새로운 손실 함수(loss function)를 도입합니다. 이 손실 함수는 서명 거리 감독(signed distance supervision) 없이도 데이터 기반 우선을 조정할 수 있게 해주며, 로컬 지역에서 통계적 추론(statistical reasoning)을 수행합니다.

- **Performance Highlights**: 제안된 방법은 표면 재구성(surface reconstruction) 및 포인트 클라우드 디노이징(point cloud denoising) 작업에서 최신 방법들보다 우수한 성능을 보여주었습니다. 또한, 여러 공통적으로 사용되는 벤치마크(shape and scene benchmarks)에서 성능을 비교하여 본 방법의 우수성을 입증했습니다.



### DiffGS: Functional Gaussian Splatting Diffusion (https://arxiv.org/abs/2410.19657)
Comments:
          Accepted by NeurIPS 2024. Project page: this https URL

- **What's New**: DiffGS는 효율적으로 높은 품질의 Gaussian primitives를 생성할 수 있는 새로운 diffusion 기반 generative model을 제안합니다. 3D Gaussian Splatting(3DGS)의 복잡성과 비구조성을 해결하기 위해, Gaussian Probability Function, Gaussian Color Function, Gaussian Transform Function이라는 세 가지 새로운 함수를 통해 Gaussian Splatting을 분리된 방식으로 표현합니다.

- **Technical Details**: DiffGS는 Gaussian VAE 모델을 사용하여 Gaussian Splatting 기능을 압축된 표현으로 생성합니다. 또한, octree-guided sampling 및 최적화를 통해 생성된 함수에서 임의의 개수의 Gaussians를 추출하는 discretization 알고리즘을 도입합니다. 이렇게 함으로써 Gaussian Splatting 기능을 불완전한 조건과 완전한 조건 모두에서 생성할 수 있게 됩니다.

- **Performance Highlights**: DiffGS는 다양한 작업에서 뛰어난 성능을 보입니다. 조건 없는 생성, 텍스트 및 이미지로부터의 조건부 생성, 부분 3DGS로부터의 생성 및 Point-to-Gaussian 생성 등에서 훌륭한 결과를 보여주며, ShapeNet 데이터셋과 DeepFashion3D 데이터셋에서 이전 최고 성능 모델들보다 비약적인 개선을 달성하였습니다.



### Frozen-DETR: Enhancing DETR with Image Understanding from Frozen Foundation Models (https://arxiv.org/abs/2410.19635)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 최근 비전 기초 모델(vision foundation models)은 다양한 작업에서 인상적인 능력을 보여주고 있지만, 객체 탐지(object detection)에서의 활용은 간과되고 있었다. 본 논문에서는 얼려진 기초 모델(frozen foundation models)이 객체 탐지에 있어 유용한 필드 엔핸서(feature enhancer)로 활용될 수 있음을 보여준다.

- **Technical Details**: 연구팀은 기초 모델의 높은 이미지 이해 능력을 탐지기로 직접 전이하는 두 가지 접근 방식을 탐구했다. 첫째, 클래스 토큰(class token)은 감지기의 디코더(decoder)에서 객체 쿼리(object queries)를 디코딩하는 데 필요한 맥락(context)을 제공함으로써 복잡한 장면을 이해하는 데 기여한다. 둘째, 패치 토큰(patch tokens)은 탐지기의 인코더(encoder)에 의미론적 세부정보(semantic details)를 더하여 성능을 향상시킨다. 이러한 방식으로 Frozen-DETR는 SOTA 쿼리 기반 탐지기인 DINO의 성능을 COCO 벨리데이션 세트에서 49.0% AP에서 51.9% AP (+2.9% AP)로 끌어올렸다.

- **Performance Highlights**: Frozen-DETR은 COCO 데이터셋에서 53.2% AP (+2.8%)로 성능을 보였으며, LVIS 데이터셋에서는 6.6% AP 향상을 기록했다. 또한, 클래스 불균형 문제를 완화하는 가능성을 보여주며, 오픈 보캐뷸러리(open-vocabulary) 시나리오에서 8.8% novel AP 증가도 달성했다.



### Multi-modal Motion Prediction using Temporal Ensembling with Learning-based Aggregation (https://arxiv.org/abs/2410.19606)
Comments:
          IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024), accepted by IROS2024

- **What's New**: 이 논문에서는 궤적 예측에서의 결측 행동 문제를 해결하기 위해 Temporal Ensembling with Learning-based Aggregation이라는 메타 알고리즘을 제안합니다. 이 방법은 인접한 프레임에서 얻은 예측을 활용하여 공간적인 범위와 예측의 다양성을 향상시킵니다.

- **Technical Details**: Temporal Ensembling은 기존의 모델 앙상블 기법과 유사하게 여러 프레임의 예측을 통합하여 최종 예측을 만듭니다. 학습 기반 집합 방법(Learning-based Aggregation)에서는 DETR와 유사한 아키텍처를 활용하여 여러 프레임의 예측 특성을 이용하고 교통 컨텍스트를 고려합니다. 이를 통해 시간적 앙상블을 가능하게 했습니다.

- **Performance Highlights**: Argoverse 2 데이터셋에서 검증한 결과 minADE에서 4%, minFDE에서 5%, 그리고 miss rate에서 1.16%의 개선을 보이며, 가장 강력한 기준선인 QCNet과 비교해 이 방법의 유효성과 잠재력을 강조합니다.



### Microplastic Identification Using AI-Driven Image Segmentation and GAN-Generated Ecological Contex (https://arxiv.org/abs/2410.19604)
Comments:
          6 pages one figure

- **What's New**: 이번 연구에서는 미세 플라스틱을 자동으로 식별하기 위해 딥 러닝 기반의 세분화(segmentation) 모델을 제안했습니다. 기존의 비싼 분석 기법 대신 저렴하고 효율적인 방법으로, GAN(Generative Adversarial Network)을 사용하여 다양한 훈련 데이터를 생성하였습니다.

- **Technical Details**: 본 모델은 Moore Institute for Plastic Pollution Research의 이미지를 라벨링하고, 환경 다양한 데이터 세트에서 미세 플라스틱을 자동으로 식별하도록 훈련되었습니다. 기존의 데이터를 기반으로 GAN을 활용하여 새로운 이미지 데이터를 생성하여 모델의 정확성을 향상시켰습니다.

- **Performance Highlights**: 세분화 모델은 다양하게 생성된 데이터를 포함하여 훈련되었고, F1 점수는 0.91로 향상되었습니다. 이는 생성된 데이터 없이 훈련된 모델의 0.82와 비교하여 현저한 개선을 보여줍니다.



### MonoDGP: Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors (https://arxiv.org/abs/2410.19590)
- **What's New**: MonoDGP는 3D 객체 탐지를 위해 새로운 Transformer 기반의 접근 방식을 제안합니다. 기존의 기하학적 깊이를 재정의하고 여러 깊이 예측 대신 기하학적 오류 예측을 채택하여 모델의 성능을 향상시킵니다.

- **Technical Details**: MonoDGP는 관점 불변 기하학적 오류를 사용하여 투영 공식을 수정하며, 2D 특징에만 의존하는 분리된 2D 디코더를 통해 2D 프라이어(priors) 및 객체 쿼리 초기화를 수행합니다. 또한, RSH(Region Segment Head)를 도입하여 개선된 특징과 분할 임베딩을 생성합니다.

- **Performance Highlights**: MonoDGP는 KITTI 3D 객체 탐지 벤치마크에서 추가 데이터 없이도 최고 성능(SOTA)을 달성하였습니다.



### FastPCI: Motion-Structure Guided Fast Point Cloud Frame Interpolation (https://arxiv.org/abs/2410.19573)
Comments:
          To appear in ECCV 2024

- **What's New**: FastPCI는 새로운 Pyramid Convolution-Transformer 아키텍처를 도입하여 빠르고 정확한 Point Cloud Frame Interpolation을 실현합니다. 이 기술은 기존의 Motion Estimators에 의존하지 않고 효율성을 극대화합니다.

- **Technical Details**: FastPCI는 Dual-Direction Motion-Structure Block을 사용하여 bidirectional input에서 motion과 구조를 함께 추정합니다. 또한, Pyramid 구조를 통해 다층 기능을 제공하며 계산 부담을 줄입니다. 이 모델은 Transformer의 주의 메커니즘을 활용하여 전역 reasoning 능력을 강화합니다.

- **Performance Highlights**: FastPCI는 Chamfer Distance에서 KITTI 데이터셋에서 PointINet과 NeuralPCI보다 각각 26.6% 및 18.3% 개선된 성능을 보이며, 속도 측면에서도 각각 10배 및 600배 더 빠릅니다.



### Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning (https://arxiv.org/abs/2410.19560)
- **What's New**: C-JEPA(Contrastive-JEPA)라는 새로운 프레임워크를 소개하며, 이는 JEPA(Join-Embedding Predictive Architecture)와 VICReg(Variance-Invariance-Covariance Regularization)를 통합하여 시각적 표현 학습의 한계를 극복하고자 합니다.

- **Technical Details**: C-JEPA는 I-JEPA의 Exponential Moving Average(EMA) 및 예측 메커니즘의 한계를 해결하고, 다양한 뷰 간의 불변성을 유지하여 전체 붕괴를 방지하는 방법으로 설계되었습니다. 이 연구는 VICReg 전략을 결합하여 모델의 안정성과 표현 품질을 크게 향상시킴을 보여줍니다.

- **Performance Highlights**: C-JEPA는 ImageNet-1K 데이터셋에서 사전 훈련 시 선형 프로빙과 파인 튜닝 성능 지표 모두에서 빠르고 향상된 수렴을 보여주며, 비지도 visual representation learning 분야의 새로운 기준을 설정할 가능성이 있습니다.



### On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes (https://arxiv.org/abs/2410.19553)
Comments:
          This paper was accepted to NeurIPS 2023 Dataset And Benchmark Track. It also showcases: Hinton's Islands of Agreement on realistic datasets which were previously hypothesized in his GLOM paper

- **What's New**: 이 논문은 비디오 액션 탐지에서 가리개(occlusion)의 영향을 탐구하고 있으며, 정적/동적 가리개를 포함한 다섯 개의 새로운 벤치마크 데이터셋을 소개합니다. 이 연구는 가리개가 비디오 인식 모델의 성능에 미치는 영향을 정량화하고, Transformer 모델이 CNN보다 더 뛰어난 성능을 보일 수 있음을 확인합니다.

- **Technical Details**: 연구에서는 O-UCF, O-JHMDB, OVIS-UCF, OVIS-JHMDB, Real-OUCF의 다섯 개 데이터셋을 사용하여 다양한 유형의 가리개를 분석하였고, 가리개의 세기가 증가할수록 모델 성능이 감소함을 발견했습니다. 특히 Transformer 모델은 가리개를 데이터 증강 형태로 사용한 CNN 모델과 비교했을 때 더 우수한 성능을 보여주었습니다. 또한, 실제 사례를 바탕으로 사전 학습의 중요성이 강조되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 모델이 O-UCF, O-JHMDB, Real-OUCF에서 각각 32.3%, 32.7%, 2.6%의 성능 향상을 기록하였고, 비디오 액션 탐지 분야에서 가리개에 대한 강건성을 높이는 몇 가지 간단하고 효과적인 트레이닝 레시피를 제안했습니다.



### GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing (https://arxiv.org/abs/2410.19552)
Comments:
          14 pages, 5 figures, 3 tables

- **What's New**: 이 논문은 시각 언어 모델(Visual Language Models, VLMs)의 한계를 극복하기 위해 비디오 프레임 쌍으로 구성된 주석 데이터셋을 소개하며, 지역적 변화를 시간에 따라 추적할 수 있도록 설계되었습니다. 주요 기법으로는 Low-Rank Adaptation(LoRA) 및 Quantized LoRA(QLoRA)를 활용하여 VLM의 성능을 향상시켰습니다.

- **Technical Details**: 논문에서는 VLM에 대한 세분화된 주석 데이터셋을 도입하며, 이는 비디오 프레임 쌍을 사용하여 시간적인 변화를 추적합니다. 이를 통해 VLM은 특정 시점 사이의 변화를 설명하는 작업을 수행할 수 있습니다. 또한, LoRA, QLoRA와 같은 효율적인 미세 조정 기법과 모델 가지치기를 통해 자원 사용의 최적화를 추구합니다.

- **Performance Highlights**: 결과적으로, 연구에서 VLM의 성능이 크게 향상되어 BERT 점수 0.864 및 ROUGE-1 점수 0.576을 달성했습니다. 이는 토지 사용 변화를 설명하는 데 있어 우수한 정확도를 보여줍니다.



### MM-WLAuslan: Multi-View Multi-Modal Word-Level Australian Sign Language Recognition Datas (https://arxiv.org/abs/2410.19488)
- **What's New**: 이 논문은 호주 수화( Auslan )의 첫 번째 대규모 워드 레벨 인식 데이터셋인 MM-WLAuslan을 구축하여 ISLR( Isolated Sign Language Recognition ) 연구에 기여합니다. 이 데이터셋은 광범위한 어휘와 다양한 카메라 뷰로 구성되어 있습니다.

- **Technical Details**: MM-WLAuslan은 282K개 이상의 수화 비디오를 포함하고 있으며, 73명의 수화 기호자에 의해 제공된 3,215개의 일반적으로 사용되는 Auslan 글로스를 기록합니다. 비디오 녹화는 RGB-D 카메라로 수행되며, 다중 뷰와 다중 모달 데이터를 관련 연구에 있습니다. 그들은 지역 커뮤니티 간의 의사소통을 지원하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과 MM-WLAuslan 데이터셋은 ISLR의 도전적인 데이터셋임을 보여줍니다. 이 데이터셋은 기존의 최신 방법들과의 벤치마크 결과를 제공하며, 특히 다중 카메라 및 뷰를 통한 성능 평가를 통해 향후 연구 및 개발에 기여할 수 있는 잠재성을 지니고 있습니다.



### x-RAGE: eXtended Reality -- Action & Gesture Events Datas (https://arxiv.org/abs/2410.19486)
- **What's New**: 이번 연구에서는 XR(확장 현실) 환경과 웨어러블 장치에서 제스처 인식을 가능하게 하는 최초의 이벤트 기반 에고센트릭(egocentric) 제스처 데이터셋을 제시합니다. 이 데이터셋은 Prophesee EVK4 카메라를 사용하여 수집되었으며, 빠른 움직임을 포착하고 조명에 영향을 받지 않는 장점을 가지고 있습니다.

- **Technical Details**: 이 데이터셋은 36개의 독특한 제스처로 구성되어 있으며, 각각은 다양한 환경에서 6명의 피험자에 의해 6번 수행되었습니다. 제스처는 정적 및 동적 환경을 포함하여, 다양한 조명 조건을 시뮬레이션하여 실제 상황을 반영합니다. 이를 통해 메모리 대역폭과 계산 요구 사항을 해결할 수 있는 가능성이 열립니다.

- **Performance Highlights**: 이 데이터셋은 XR 기기용 제스처 및 액션 인식을 위해 설계되었으며, 다양한 실세계 시나리오에서 작동할 수 있는 강력한 제스처 인식 시스템 개발에 유용합니다. 특히, 낮은 지연 시간과 높은 동적 범위 특성 덕분에 스마트 안경 및 헤드 마운트 디스플레이와 같은 웨어러블 장치에서의 응용 가능성이 높습니다.



### Content-Aware Radiance Fields: Aligning Model Complexity with Scene Intricacy Through Learned Bitwidth Quantization (https://arxiv.org/abs/2410.19483)
Comments:
          accepted by ECCV2024

- **What's New**: 본 연구에서는 콘텐츠-감지 반사 필드(content-aware radiance fields)를 소개합니다. 이 모델은 각 장면의 복잡성에 맞춰 비트폭(bitwidth)을 조정하여 훈련합니다. 기존의 모델이 모든 장면을 단일 픽스드 스케일(fixed scale)로 압축하는 것과는 달리, 본 방법은 특정 장면의 특성과 요구 사항에 따라 차별화된 비트폭을 제공합니다.

- **Technical Details**: 제안된 방법은 A-CAQ(Adversarial Content-Aware Quantization) 알고리즘을 통해 구현됩니다. 이 알고리즘에서는 모델 파라미터의 비트폭을 학습 가능하게 설정하여, 복잡한 장면은 더 높은 비트폭을 사용하고, 간단한 장면은 낮은 비트폭을 사용하도록 만듭니다. 이를 통해 계산 복잡성을 줄이는 동시에 재구성 및 렌더링 품질을 유지합니다.

- **Performance Highlights**: 실험 결과에 따르면, 콘텐츠-감지 반사 필드는 다양한 데이터 세트에서 계산 복잡성이 상당히 감소하는 한편, 복원 및 렌더링 품질을 유지하는 데 성공했습니다. 이로써 반사 필드 모델의 실제 배포에 있어 유용성을 입증하였습니다.



### Peter Parker or Spiderman? Disambiguating Multiple Class Labels (https://arxiv.org/abs/2410.19479)
Comments:
          Accepted to Neural Information Processing Systems (NeurIPS 2024). ATTRIB Workshop

- **What's New**: 이 논문에서는 다중 예측을 하는 심층 신경망(Deep Neural Networks)의 예측 결과를 해석하는 새로운 방법론을 제시합니다. 특히 top-k 예측에서 예측의 의미를 명확히 하기 위한 프레임워크를 개발하여, 두 개의 예측이 서로 다른 엔티티를 기반으로 하는지, 또는 동일한 엔티티에 대한 두 개의 예측인지 확인할 수 있습니다.

- **Technical Details**: 제안된 방법론은 먼저 입력 이미지를 세그먼트화(Segmentation)한 후, 각 레이블에 대해 세그먼트별 입력 기여도(Input Attribution) 점수를 할당합니다. 이 점수는 두 개의 레이블 예측이 동일한 엔티티 집합을 가리키는지 아니면 다른 집합을 가리키는지를 결정하는 데 사용됩니다. 또한, 이 방법론은 재실행 없이 주어진 주장(클래스 레이블 쌍이 서로 다른 엔티티 타입인지 단일 엔티티 타입인지)의 유효성을 증명할 수 있는 반증(Counterfactual Proof)을 제공합니다.

- **Performance Highlights**: ImageNet 검증 세트의 여러 샘플을 통해 이 방법론이 여러 모델에서 잘 작동함을 보여주었습니다. 이 연구는 현재의 해석 가능성 연구에서 주목받지 못했던 예측의 두 가지 해석을 구분하는 데 기여합니다.



### Fusion-then-Distillation: Toward Cross-modal Positive Distillation for Domain Adaptive 3D Semantic Segmentation (https://arxiv.org/abs/2410.19446)
- **What's New**: 본 논문에서는 cross-modal positive distillation과 fusion-then-distillation (FtD++) 방법을 제안하여 소스 도메인과 타겟 도메인 간의 3D semantic segmentation을 개선합니다. 이 방법은 각 도메인에서의 class probability distribution의 일관성을 유지하며, cross-modal learning에서 얻는 보완적 이점을 활용합니다.

- **Technical Details**: FtD++ 방법은 다음과 같은 세 가지 주요 요소로 구성됩니다. 첫째, model-agnostic feature fusion module (MFFM)을 통해 cross-modal fusion representation을 생성합니다. 둘째, cross-modal positive distillation을 통해 소스 도메인의 정보와 타겟 도메인의 스타일을 결합하여 domain-modality alignment를 달성합니다. 셋째, cross-modal debiased pseudo-labeling (xDPL)을 설계하여 pseudo-label의 불확실성을 모델링합니다.

- **Performance Highlights**: 종합적인 실험 결과, 본 방법은 Day→Night, USA→Sing., vKITTI→sKITTI, A2D2→sKITTI와 같은 여러 도메인 적응 시나리오에서 state-of-the-art 성능을 달성하였습니다. 또한, 타겟 도메인에서 주석이 달린 데이터 몇 개를 입력할 수 있는 semi-supervised domain adaptation (SSDA) 방식을 통해 성능을 더욱 향상시켰습니다.



### Balancing the Scales: Enhancing Fairness in Facial Expression Recognition with Latent Alignmen (https://arxiv.org/abs/2410.19444)
- **What's New**: 이 연구는 Facial Expression Recognition (FER) 분야에서 편향(bias)을 완화하는 새로운 방법론을 제안하며, 이를 통해 공정성(fairness)과 정확도를 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 접근법은 새로운 latent alignment 기술을 통해 편향을 완화하고, Variational Autoencoder (VAE) 및 적대적 판별기(adversarial discriminator)를 활용하여 학습 과정에서 편향을 줄이는 손실 함수(loss function)를 개발했습니다. 또한, CNN(Convolutional Neural Network)을 기반으로 한 모델을 사용하여 감정 인식을 구현합니다.

- **Performance Highlights**: 이 연구는 RAF-DB와 CelebA 두 개의 데이터셋에서 생긴 모든 성별, 인종 및 연령에 대한 편향을 감소시키며, 최첨단 결과와 경쟁력 있는 성능을 달성했습니다.



### Transductive Learning for Near-Duplicate Image Detection in Scanned Photo Collections (https://arxiv.org/abs/2410.19437)
Comments:
          Published in ICDAR 2023

- **What's New**: 이 논문은 실제 문서 관리 환경에서의 near-duplicate 이미지 탐지 기술을 비교 연구한 결과를 제시합니다. 특히, 대량의 스캔된 사진을 수동으로 주석 달기 위해 의뢰받은 문서 관리 회사에서의 사용 사례를 다루고 있습니다.

- **Technical Details**: 이 연구에서는 convolutional neural networks (CNNs)와 Vision Transformers (ViTs)와 같은 최신 deep learning 아키텍처를 활용한 transductive learning 접근법을 제안합니다. 대규모 데이터셋에서 사전 훈련된 모델을 unlabeled target collection에 대해 self-supervised learning으로 fine-tuning하여 성능을 향상시킬 수 있었습니다.

- **Performance Highlights**: 제안된 접근법은 UKBench 및 내부 개인 데이터셋에서 진행된 near-duplicate 이미지 탐지 작업에서 기존 방법보다 우수한 성능을 보였음을 입증하였습니다.



### Paint Bucket Colorization Using Anime Character Color Design Sheets (https://arxiv.org/abs/2410.19424)
Comments:
          Extension of arXiv:2403.18342; Project page at this https URL

- **What's New**: 이 논문에서는 'inclusion matching'이라는 새로운 접근 방식을 도입하여 구간과 세그먼트 간의 포함 관계를 이해하고, 이를 활용해 색상화를 보다 정확하게 수행하는 기법을 제안합니다.

- **Technical Details**: 제안된 'inclusion matching'은 목표 프레임의 각 세그먼트가 참조 프레임 내 특정 색으로 포함될 가능성을 계산합니다. 또한, 세그먼트 파싱 모듈과 색상 왜곡(color warping) 모듈을 통합하여 키프레임 색상화 및 연속 프레임 색상화의 성능을 크게 개선합니다. PaintBucket-Character라는 독특한 데이터셋을 개발하여 캐릭터의 윤곽선 및 색상화 버전, 음영 주석을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 정확하고 일관된 색상화를 보여주었으며, 기존의 수작업 애니메이션에서 제시된 벤치마크에서도 우수한 성능을 발휘했습니다.



### Unified Cross-Modal Image Synthesis with Hierarchical Mixture of Product-of-Experts (https://arxiv.org/abs/2410.19378)
Comments:
          Manuscript under review

- **What's New**: MMHVAE(멀티모달 계층 변분 오토인코더)는 서로 다른 모달리티의 이미지로부터 누락된 이미지를 합성하는 새로운 접근 방식을 제안합니다. 이 방법은 불완전한 이미지 세트를 입력으로 사용하여 통합된 교차 모달 이미지 합성을 수행하는 위계를 가진 모델을 포함합니다.

- **Technical Details**: 제안된 모델은 Product-of-Experts의 혼합으로 변분 사후 분포를 모델링하며, 제시된 방법은 관찰된 정보를 인코딩할 뿐만 아니라 이미지 합성을 위한 누락된 정보를 추정하도록 장려합니다. 또한, 훈련 데이터가 항상 완전하지 않을 것이라는 가정을 하고 있으며, 데이터셋 수준의 정보를 활용하는 적대적 전략을 도입합니다.

- **Performance Highlights**: 다양한 실험을 통해 다중 매개변수 자기 공명 영상 및 초음파 데이터 간의 교차 모달 이미지 합성 문제에서 기존 방법보다 더 나은 이미지 합성 성능을 보여주며, 계산 비용이 덜 듭니다.



### Capsule Endoscopy Multi-classification via Gated Attention and Wavelet Transformations (https://arxiv.org/abs/2410.19363)
Comments:
          Capsule Vision 2024 Challenge

- **What's New**: 이 연구는 비디오 캡슐 내시경(VCE) 이미지에서 위장관 이상을 자동으로 분류할 새로운 모델을 개발하고 평가하는 과정을 제시합니다.

- **Technical Details**: 모델 아키텍처에 Omni Dimensional Gated Attention (OGA) 메커니즘과 Wavelet 변환을 통합하여, 중요한 영역에 집중하고 노이즈 및 불필요한 특징을 감소시켰습니다. 이는 VCE 이미지의 텍스처와 색상에서 높은 변동성이 존재할 때 특히 유리합니다. Stationary Wavelet Transform과 Discrete Wavelet Transform에서 추출된 특징들은 다중 스케일 특징을 포착하기 위해 채널-wise로 연결됩니다.

- **Performance Highlights**: 제안된 모델은 훈련 정확도 92.76%, 검증 정확도 91.19%를 기록하였으며, Balanced Accuracy는 94.81%, AUC는 87.49%, F1-score는 91.11%, precision은 91.17%, recall은 91.19%, specificity는 98.44%입니다. 또한, VGG16 및 ResNet50 모델과 비교하여 위장관 이상을 신속하고 정확하게 식별하는 능력이 향상되었음을 보여주었습니다.



### FasterCache: Training-Free Video Diffusion Model Acceleration with High Quality (https://arxiv.org/abs/2410.19355)
- **What's New**: 이 논문에서는 비디오 확산 모델(video diffusion models)의 추론(inference) 속도를 높이기 위한 새로운 훈련 없는 전략인 FasterCache를 제안합니다. 기존 캐시 기반 방법들에서 인접 단계(feature) 재사용이 비디오 품질을 저하시키는 문제를 해결하고, classifier-free guidance (CFG)와 조건부 및 비조건부 feature 간의 중복을 밝혀냈습니다.

- **Technical Details**: FasterCache는 주목(attention) 모듈의 feature 재사용을 동적으로 조정하여 인접 단계 간의 특성을 유지하며, CFG-Cache를 통해 조건부 및 비조건부 출력의 잔여물(residual)을 저장합니다. 이로써 비디오 품질을 유지하면서 추론 속도를 향상시키는 방법론을 설명합니다.

- **Performance Highlights**: FasterCache는 Vchitect-2.0 모델에서 1.67배의 속도 향상을 달성하며, 비디오 품질은 기존 기준과 비슷합니다. 전반적으로 FasterCache는 추론 속도와 비디오 품질 모두에서 기존 방법들을 일관되게 초월하는 성능을 보여줍니다.



### DECADE: Towards Designing Efficient-yet-Accurate Distance Estimation Modules for Collision Avoidance in Mobile Advanced Driver Assistance Systems (https://arxiv.org/abs/2410.19336)
Comments:
          8 pages, 17 figures, 4 tables

- **What's New**: 본 연구에서는 모든 사용자가 접근할 수 있는 Advanced Driver Assistance Systems (ADAS)를 위한 모바일 애플리케이션 개발을 위한 새로운 거리 추정 모델, DECADE를 제안합니다. 이 모델은 객체 탐지 결과를 기반으로 한 경량의 DNN 구조를 사용하여 픽셀 단위가 아닌 탐지 단위로 거리 추정을 수행합니다.

- **Technical Details**: DECADE 모델은 Pose Estimation DNN을 사용하여 탐지된 객체의 방향 정보를 보완하여 거리 추정 DNN의 예측을 강화합니다. YOLO 객체 탐지기와의 결합을 통해 두 개의 신경망을 활용한 방식으로, 주어진 거리 범위에서 높은 정확도를 유지합니다. 이 연구는 KITTI 3D Object Detection 데이터셋을 사용하여 모델의 성능을 평가합니다.

- **Performance Highlights**: DECADE 모델은 평균 절대 오차(MAE) 1.38m, 평균 상대 오차(MRE) 7.3%를 달성했으며, 특히 0-70m 거리 범위에서의 정확도 향상이 두드러집니다. 이 연구는 크기 변동에 대한 객체 인식 강도를 평가하고, 최종적으로 저비용으로 효율적인 거리 추정 기술의 가능성을 보여줍니다.



### Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion (https://arxiv.org/abs/2410.19324)
- **What's New**: 이 논문은 Latent diffusion (잠재 확산) 모델 대신 Pixel-space diffusion (픽셀 공간 확산) 모델이 고해상도 이미지 생성에서 품질과 효율성 면에서 경쟁력이 있음을 보여줍니다. 특히, Pixel-space 모델을 위한 새로운 접근 방식을 통해 Simple Diffusion v2 (SiD2)라는 새로운 모델 가족을 제안하고 있습니다.

- **Technical Details**: 주요 기술적 기여는 다음과 같습니다: 1) Sigmoid loss (시그모이드 손실)를 사용하여 hyper-parameters (하이퍼 파라미터)를 조정했습니다. 2) skip-connections를 최소화한 간소화된 메모리 효율 아키텍처를 사용했습니다. 3) 고해상도 처리에 최적화된 모델 크기를 조정했습니다. 이러한 조합을 통해 ImageNet512에서 FID 1.50을 달성하였으며, ImageNet128 및 ImageNet256에서 새로운 SOTA 결과를 얻었습니다.

- **Performance Highlights**: Pixel-space diffusion 모델의 성능이 ImageNet512에서 2.65에서 1.50으로 개선되었습니다. 또한, ImageNet128 및 ImageNet256에서의 전체 성능에서도 SOTA를 달성하였습니다.



### Flow Generator Matching (https://arxiv.org/abs/2410.19310)
- **What's New**: 본 논문은 Flow Generator Matching (FGM)이라는 혁신적인 접근 방식을 제시하여, flow-matching 모델의 샘플링 속도를 가속화하고 원래의 성능을 유지하는 방법을 제안합니다.

- **Technical Details**: FGM은 확률적(distillation) 프레임워크로, flow 모델의 샘플링 과정을 단일 단계 생성기(one-step generator)로 간소화합니다. 이를 통해 컴퓨팅 자원을 보다 효율적으로 활용할 수 있습니다. 또한 CIFAR10 데이터셋에서 새로운 Fréchet Inception Distance (FID) 점수 3.08을 기록하며 성능을 확인했습니다.

- **Performance Highlights**: 이 결과는 특히 MM-DiT 아키텍처에 기반한 Stable Diffusion 3 모델의 distillation을 통해 생성된 MM-DiT-FGM 모델이 GenEval 벤치마크에서 뛰어난 생성 품질을 보여주는 것으로 나타났습니다.



### Semi-supervised Chinese Poem-to-Painting Generation via Cycle-consistent Adversarial Networks (https://arxiv.org/abs/2410.19307)
- **What's New**: 본 연구는 고전 중국 시(詩)와 그림(畫)의 번역을 위한 반감독 학습(framework for semi-supervised learning) 접근 방식을 제안합니다. 이를 통해 제한된 쌍 데이터를 활용하여 시와 그림 간의 의미적 정렬을 학습합니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 cycle-consistent adversarial networks를 기반으로 하여, 시와 그림을 매치하기 위한 쌍 및 비쌍 데이터 모두로부터 학습합니다. 모델은 시 및 그림 인코더(encoders)를 도입하여 공유된 의미 공간(semantic space)으로 매핑하며, 생성기(generators)는 이러한 공간으로부터 복원합니다.

- **Performance Highlights**: 제안된 모델은 새로운 Chinese Painting Description Dataset(CPDD)에서 이전 방법들보다 뛰어난 성능을 보여주었고, 고품질의 시-그림 쌍을 생성하는 데 성공하였습니다.



### Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting (https://arxiv.org/abs/2410.19294)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 새로운 연구 FROLIC은 레이블이 없는 데이터로 제로샷 성능을 향상시키기 위한 프로프트 분포 학습 및 편향 수정 프레임워크를 제안합니다. 이는 기존의 레이블 요구 없이 다양한 시각 표현을 효과적으로 캡처합니다.

- **Technical Details**: Frolic은 프로프트 프로토타입들 간의 분포를 학습하여 다양한 시각적 표현을 캡처합니다. 기존 CLIP 모델과의 적응형 융합을 통해 신뢰도 매칭을 수행하고, 레이블 없는 로짓 조정을 통해 레이블 편향을 수정합니다. 이 과정에서 하이퍼파라미터 조정이 필요하지 않으며, 훈련이 필요 없는 장점이 있습니다.

- **Performance Highlights**: Frolic은 16개 데이터셋에서 실시한 실험을 통해 기존 최첨단 기술보다 평균 2.6% 향상된 성능을 보였습니다. CLIP ViT-B/16 모델을 사용한 결과, ImageNet 및 다섯 개 분포 이동에서 평균 1.5% 향상된 성과를 달성했습니다.



### Prompting Continual Person Search (https://arxiv.org/abs/2410.19239)
Comments:
          ACM MM 2024

- **What's New**: 이 논문에서는 상황에 따라 달라지는 다양한 영역에서 연속적으로 학습하는 지속적(Prompt-based Continual Person Search, PoPS) 모델을 소개합니다. 이를 통해 실시간으로 증가하는 데이터에 적응하고, 각 도메인에서 사람을 검색할 수 있는 기능을 구성하고 있습니다.

- **Technical Details**: PoPS 모델은 compositional person search transformer를 기반으로 하여, 대규모 데이터로부터의 사전 학습 없이도 효과적인 사전 학습된 transformer를 구축합니다. 또한, 도메인별 정보와 속성을 매칭하기 위한 다양한 속성 프로젝션과 프로토타입 임베딩을 포함한 도메인 증가 프롬프트 풀을 설계하여 입력 이미지와 학습된 속성을 매칭할 수 있도록 합니다.

- **Performance Highlights**: 제안된 PoPS 모델은 여러 차례의 실험을 통해 지속적인 사람 검색 문제의 유효성을 입증하였으며, 기존 데이터에 대한 재학습 없이도 새로운 검색 작업에 적응할 수 있는 능력을 보여줍니다.



### Prototypical Hash Encoding for On-the-Fly Fine-Grained Category Discovery (https://arxiv.org/abs/2410.19213)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 실시간으로 알려진 및 알려지지 않은 클래스에 속하는 새로운 스트림 데이터를 발견하는 On-the-fly Category Discovery (OCD) 작업을 연구합니다. 기존의 해시 기반 방법들이 가지는 불리한 점을 극복하기 위해 새로운 Prototypical Hash Encoding (PHE) 프레임워크를 제안합니다.

- **Technical Details**: PHE 프레임워크는 Category-aware Prototype Generation (CPG)과 Discriminative Category Encoding (DCE)으로 구성되어 있습니다. CPG는 각 카테고리를 다수의 프로토타입으로 표현하여 범주 내부의 다양성을 포착합니다. DCE는 카테고리 프로토타입의 도움이 되는 해시 코드의 구별 능력을 향상시킵니다. 두 가지 구성 요소가 상호 이익을 가지도록 최적화 됩니다.

- **Performance Highlights**: 다수의 미세한 데이터셋에서 진행된 실험 결과, PHE는 이전 해시 기반 OCD 방법들보다 평균 5.3% 향상된 ALL ACC를 기록하여 성능이 크게 우수함을 입증했습니다.



### Classifying Bicycle Infrastructure Using On-Bike Street-Level Images (https://arxiv.org/abs/2410.19194)
Comments:
          8 pages, 6 figures, presented at ITSC 2024

- **What's New**: 본 연구는 자전거에 장착된 스마트폰 카메라 데이터만을 사용하여 자전거 인프라를 분류하는 최초의 시스템을 제안합니다. 이 시스템은 이미지 시퀀스를 입력받아 시간적으로 분석하여 표지판의 희소성(sparsity)을 고려합니다.

- **Technical Details**: 제안된 모델은 ConvNeXt-V2 아키텍처를 기반으로 하여 이미지 시퀀스에서 특징을 추출하고, 이를 통해 생성된 잠재 표현(latent representation)을 시간적으로 분석하는 Temporal extractor를 포함합니다. 이를 통해 5개의 주요 클래스와 13개의 하위 클래스로 자전거 인프라를 분류합니다.

- **Performance Highlights**: 모델은 95.38%의 정확도를 기록하였으며, 비시간적(non-temporal) 모델보다 7.55% 향상되었습니다. 또한, 90%의 이미지가 빈 이미지로 대체된 경우에도 정확도가 6.6%만 감소하는 robust한 성능을 보여줍니다.



### Review of wavelet-based unsupervised texture segmentation, advantage of adaptive wavelets (https://arxiv.org/abs/2410.19191)
- **What's New**: 본 논문에서는 이미지의 텍스처(segmentation)를 구분하기 위한 새로운 접근법으로 최근 도입된 Empirical Wavelet Transform (EWT)를 제안합니다. EWT를 활용하여 기존의 classical wavelet들보다 향상된 결과를 도출함을 보여줍니다.

- **Technical Details**: 텍스처 세분화에 있어, 원본 이미지를 분석하고 Parallell 기법을 통해 텍스처의 특징 벡터(feature vector)를 추출하며, 클러스터링(clustering) 기법을 적용합니다. 이는 카툰(cartoon) 및 텍스처(texture) 분해 단계를 지나며, EWT 및 다양한 wavelet 기반 특징들이 이용됩니다.

- **Performance Highlights**: 여섯 가지의 일반적인 벤치마크 데이터셋에서의 테스트 결과, Empirical Wavelet를 활용한 접근법이 기존의 wavelet 방법보다 우수한 성능을 발휘함을 확인했습니다. 이는 주로 텍스처의 전반적인 정보에 집중할 수 있도록 돕습니다.



### Noise Adaption Network for Morse Code Image Classification (https://arxiv.org/abs/2410.19180)
Comments:
          8 pages, 3 figures

- **What's New**: 논문에서는 Morse code 이미지 분류를 위한 새로운 두 단계 접근법인 Noise Adaptation Network (NANet)를 제안하고 있습니다. 기존 방법들이 단일 잡음 유형에 국한되어 있던 점을 개선하고 있습니다.

- **Technical Details**: NANet은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 U형 네트워크를 통해 표현 특징을 학습하고 이미지를 denoise하며, 두 번째 단계에서는 깊은 합성곱 신경망(Deep Convolutional Neural Network)을 이용하여 분류를 수행합니다. 이 과정에서 첫 번째 단계의 denoising 모듈을 활용하여 정확도 및 강건성을 향상시킵니다.

- **Performance Highlights**: 보고된 결과는 다양한 잡음 데이터셋, 즉 Gaussian, salt-and-pepper, uniform noise 변형에 대해 NANet이 기존 접근법보다 우수한 성능을 보인다는 것을 입증합니다.



### DCT-HistoTransformer: Efficient Lightweight Vision Transformer with DCT Integration for histopathological image analysis (https://arxiv.org/abs/2410.19166)
Comments:
          7 pages, 5 figures, Accepted for 2024 9th International Iranian Conference on Biomedical Engineering (ICBME)

- **What's New**: 본 연구에서는 대규모 데이터셋 없이 효과적으로 작동하는 경량형 유방암 분류 접근법을 제안합니다. Discrete Cosine Transform (DCT) Attention과 MobileConv를 포함한 병렬 처리 경로를 통해 이미지 데이터를 공간 영역에서 주파수 영역으로 변환하여 높은 주파수를 필터링함으로써 계산 비용을 줄이는 장점을 활용합니다.

- **Technical Details**: 제안된 모델은 DCT-Conv 블록을 이용하여 두 개의 병렬 분기를 구성하며, DCT-Attention과 MobileConv 각기 다른 전처리를 수행합니다. 이러한 접근은 입력 데이터의 크기를 줄여 자체 주의(self-attention) 가중치 계산에 대한 계산 요구사항을 완화하고, 모바일 및 에지 장치에서의 처리를 수월하게 합니다.

- **Performance Highlights**: 이 모델은 이진 분류에서 96.00% ± 0.48%의 정확도를, 다중 클래스 분류에서 87.85% ± 0.93%의 정확도를 달성하였으며, 이는 최신 모델들과 동등한 성능을 보이면서도 계산 비용을 크게 줄이는 효율적인 솔루션입니다.



### HUE Dataset: High-Resolution Event and Frame Sequences for Low-Light Vision (https://arxiv.org/abs/2410.19164)
Comments:
          18 pages, 4 figures. Has been accepted for publication at the European Conference on Computer Vision Workshops (ECCVW), Milano, 2024. The project page can be found at this https URL

- **What's New**: 이번 연구에서는 다양한 저조도 환경에서 촬영된 고해상도 이벤트(event) 및 프레임(frame) 시퀀스를 포함하는 HUE(dataset)를 소개합니다. HUE 데이터셋은 저조도 환경에서의 이미지 향상 방법들의 성능을 평가하기 위해 106가지 시퀀스를 포함하고 있습니다.

- **Technical Details**: HUE 데이터셋은 1280x720 해상도로 이벤트 데이터를 캡처하며, 실내 및 도시 풍경, 황혼, 야간 등 다양한 환경에서 촬영되었습니다. 이 데이터셋은 비동축(non-coaxial) 카메라 세팅을 사용하여 고해상도 이벤트 데이터와 보완적인 프레임 데이터를 수집하였습니다.

- **Performance Highlights**: 이 연구의 평가 결과에 따르면, 이벤트 기반 방법들은 특정 지표에서 우수한 성과를 보였지만, 실제 애플리케이션에서는 잘못된 긍정(true positive)을 발생시키는 경우가 있음을 보여주었습니다. 이러한 HUE 데이터셋과 포괄적인 분석은 저조도 비전 및 하이브리드 카메라 시스템의 향후 연구에 귀중한 통찰을 제공합니다.



### Visual Text Matters: Improving Text-KVQA with Visual Text Entity Knowledge-aware Large Multimodal Assistan (https://arxiv.org/abs/2410.19144)
Comments:
          Accepted to EMNLP (Main) 2024

- **What's New**: 이 연구는 최신 대형 다중모달 모델(large multimodal models, LMMs)의 발전을 기반으로 텍스트 기반 시각 질문 응답(knowledge-aware text-based visual question answering, Text-KVQA)을 재조명하였습니다. 주요 기여는 VisTEL이라는 시각 텍스트 엔티티 링크 모듈과 KaLMA라는 지식 인식 대형 다중모달 어시스턴트를 제안하여, 보다 정확한 답변을 제공하는 것입니다.

- **Technical Details**: VisTEL은 이미지 내 시각 텍스트 엔티티를 인식하고 이를 지식 기반에 연결하는 방법론으로, 최신 OCR 기반 텍스트 인식 엔진과 LMM의 기능을 결합하여 텍스트와 시각적 맥락을 동시에 활용하여 링크를 수행합니다. KaLMA는 이러한 정보로 LMM을 보강하여 시각 질문에 대한 응답 정확도를 높입니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, 제안한 방법론은 Text-KVQA에 대하여 기존 최상위 접근 방식에 비해 23.3%의 성능 개선을 이루어냈으며, 장면, 책 표지 및 영화 포스터 데이터셋의 각각 18.2%, 19.6%, 32.2% 개선을 보여주었습니다. 이로써 새로운 최첨단 결과를 달성하였습니다.



### MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision (https://arxiv.org/abs/2410.19115)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 MoGe라는 강력한 모델을 소개하며, 단일 이미지를 통해 3D 지오메트리를 복원할 수 있는 새로운 접근 방식을 제안합니다. 특히 이 모델은 affine-invariant representation을 사용하여 진정한 전역 스케일과 이동에 구애받지 않는 3D 포인트 맵을 직접 예측합니다.

- **Technical Details**: MoGe는 monocular open-domain 이미지에서 3D 포인트 맵을 직접 예측하도록 설계되었습니다. 모델은 global scaling factor와 translation을 통해 예측 결과를 ground truth와 정렬하는 Robust, Optimal, Efficient (ROE) global alignment solver를 사용합니다. 또한, multi-scale local geometry loss를 도입하여 3D 포인트 클라우드의 지역적인 차이에 대한 페널티를 부과하여 정확성을 높입니다.

- **Performance Highlights**: 모델은 다양한 기존 데이터셋에서 학습되었으며, 테스트 결과 다양한 unseen datasets에서 state-of-the-art 방법들에 비해 35% 이상의 오류 감소 성능을 보여주었습니다. 기존의 monocular geometric estimation(MGE), monocular depth estimation(MDE) 및 camera FoV 추정에서도 20%~30%의 오류 감소를 기록하며 모든 평가 메트릭에서 탁월한 성과를 얻었습니다.



### VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks (https://arxiv.org/abs/2410.19100)
- **What's New**: 이번 논문에서는 VideoWebArena (VideoWA)라는 새로운 벤치마크를 소개하며, 이는 장기 맥락(long-context) 비디오 이해를 위한 멀티모달 에이전트의 능력을 평가하는 데 중점을 두고 있습니다. 이 벤치마크는 수작업으로 제작된 비디오 튜토리얼에 기반한 2,021개의 에이전트 작업으로 구성되어 있으며, 총 4시간에 달하는 콘텐츠를 포함하고 있습니다.

- **Technical Details**: VideoWebArena는 기술 유지를 평가하는 1,621개의 작업과 정보 유지를 평가하는 400개의 작업으로 나뉘어 있습니다. 기술 유지 작업은 에이전트가 주어진 인간 시연을 사용하여 작업을 효율적으로 수행할 수 있는지를 평가하며, 정보 유지 작업은 에이전트가 비디오에서 작업 수행에 관련된 정보를 검색할 수 있는지를 평가합니다. 최신 모델인 GPT-4o와 Gemini 1.5 Pro가 이 벤치마크에서 평가되었습니다.

- **Performance Highlights**: 본 연구에서 최고의 모델은 정보 유지 작업에서 13.3%의 성공률을, 기술 유지 작업에서는 45.8%의 성공률을 기록했습니다. 이는 각각 인간 성과의 73.9% 및 79.3%에 비해 훨씬 낮은 수치입니다. 또한 URL에서 제공되는 현 시스템의 코드 및 문서에 대한 접근성도 강조되었습니다.



### A Counterexample in Cross-Correlation Template Matching (https://arxiv.org/abs/2410.19085)
- **What's New**: 이 논문은 신호 및 이미지 처리에서 샘플링(sampling) 및 양자화(quantization)의 이론적 이해의 부족함을 다루고 있습니다. 특히, 노이즈가 있는 데이터 시퀀스의 정합 및 세분화(segmentation)에 관한 문제를 제시하며, 전통적인 교차 상관(cross-correlation) 기법의 한계를 보여주고 새로운 정합 기법을 제안합니다.

- **Technical Details**: 하나의 차원적 공간 제한(piecewise constant) 함수의 샘플링과 관련된 이론적 문제를 다루고 있습니다. 논문에서는 노이즈가 있는 두 세트의 관측치를 기반으로 차이 시퀀스(difference sequences), 임계값(thresholding), 동적 프로그래밍(dynamic programming) 기법을 사용하여 데이터 시퀀스를 정합하고 세분화하는 방법을 제시합니다. 이 기술들이 노이즈 많은 데이터 시퀀스를 align하는 데 어떻게 사용될 수 있는지를 보여 줍니다.

- **Performance Highlights**: 기존의 cross-correlation 기법은 노이즈가 있는 샘플에서 잘 작동하지 않지만, 제안된 방법은 특정 조건 하에 최적의 정합과 세분화를 제공할 수 있습니다. 저자들은 노이즈의 가정 아래에서도 근본 함수의 추정이 가능하다는 점을 이론적으로 입증하였습니다.



### BIFR\"OST: 3D-Aware Image compositing with Language Instructions (https://arxiv.org/abs/2410.19079)
Comments:
          NeurIPS 2024, Code Available: this https URL. arXiv admin note: text overlap with arXiv:2307.09481 by other authors

- **What's New**: 본 논문은 Bifröst라는 새로운 3D 인식 프레임워크를 소개하면서, 기존의 2D 레벨 이미지 컴포지팅(영상 합성) 방법의 한계를 극복하고 2.5D 객체 위치 예측을 통해 복잡한 공간 관계를 잘 처리하는 방법을 제시합니다.

- **Technical Details**: Bifröst는 MLLM(Multi-modal Large Language Model)을 2.5D 위치 예측기로 훈련시키고, 이미지 생성 과정에서 깊이 맵(depth map)을 추가적인 조건으로 통합하여 2D와 3D의 갭을 줄이는 방식으로 작동합니다. 이 방법은 여러 종류의 입력 특성을 처리할 수 있도록 설계된 이미지 컴포지팅 모델을 포함하여, 오클루전(occlusion), 깊이 블러(depth blur) 및 이미지 조화(image harmonization)를 고려한 고품질 이미지 합성을 제공합니다.

- **Performance Highlights**: Bifröst는 기존 방법들과 비교해 시각적으로 우수한 결과를 보여줍니다. 포괄적인 정성적 및 정량적 평가를 통해 Bifröst가 정교한 공간 이해가 필요한 시나리오에서 현실감 있는 이미지를 생성하는 데 강력한 해결책을 제공함을 입증하였습니다.



### Generative Topology for Shape Synthesis (https://arxiv.org/abs/2410.18987)
- **What's New**: 본 논문에서는 Euler Characteristic Transform (ECT)의 역전환 방법을 학습하여 포인트 클라우드(point clouds)에서의 형태 생성(shape generation) 작업을 위한 새로운 프레임워크를 개발하였다. 이 모델은 재구성 및 생성 작업에서 높은 품질을 자랑하며, 효율적인 잠재 공간(latent space) 보간(interpolation)을 제공하고, 기존 방법들보다 몇 배 빠른 성능을 보여준다.

- **Technical Details**: ECT는 형상을 다양한 방향과 규모에서 여러 방향으로 연구하는 아이디어를 바탕으로 하여, 포인트 클라우드 및 단순 복합체(simplicial complexes)와 같은 다양한 데이터 형태를 처리하는 일반적인 프레임워크를 제공한다. 이 논문에서는 ECT를 역전환하는 새로운 딥러닝(deep learning) 모델을 개발하였으며, ECT는 본질적으로 순열 불변(permutation-invariant)이며 데이터로부터 회전을 학습할 수 있는 능력을 갖춘 표현이다.

- **Performance Highlights**: 본 연구의 결과, ECT를 활용한 생성 모델이 재구성 및 생성 품질, 그리고 계산 성능 측면에서 기존 모델들을 초월하는 것으로 나타났다. 특히, ECT는 아키텍처의 요구사항을 상당히 간소화시키며, 일반적인 아키텍처와 호환 가능하다는 장점이 있다.



### VehicleSDF: A 3D generative model for constrained engineering design via surrogate modeling (https://arxiv.org/abs/2410.18986)
Comments:
          9 pages, 14 figures, NeurIPS 2024 workshop

- **What's New**: 이번 연구는 차량 개발 맥락에서 설계 공간을 탐색하는 3D 생성 모델을 탐구하며, 엔지니어링 제약 조건을 추정하고 강제 적용하는 방법을 제안합니다. 이 모델은 특정 기하학적 사양을 충족하는 다양한 3D 자동차 모델을 생성하여, 기계적 성능과 미적 디자인을 동시에 고려합니다.

- **Technical Details**: 이 연구에서는 ShapeNet 데이터셋을 사용하여 VehicleSDF라는 DeepSDF 기반 모델을 훈련합니다. 이 모델은 잠재 공간(latent space)에서 잠재 벡터를 최적화하여 기하학적 특성에 맞춘 3D 모델을 생성할 수 있습니다. 또한, 서그리게이트 모델을 훈련하여 3D 모델에서 엔지니어링 매개변수를 빠르게 추정합니다. 마지막으로, StableDiffusion과 ControlNet을 사용하여 사진 같은 스타일의 차량 렌더링을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기하학적 매개변수를 충족하는 다양한 3D 모델을 생성하는 데 성공적이며, 공기 저항 계수(Cd)와 같은 성능 매개변수를 효율적으로 추정할 수 있음을 보여주었습니다. 이러한 과정은 디자인 초기 단계에서 기계적 성능을 검증하는 데 도움을 주고, 이후 수정 작업을 줄일 수 있는 가능성을 가지고 있습니다.



### Very High-Resolution Bridge Deformation Monitoring Using UAV-based Photogrammetry (https://arxiv.org/abs/2410.18984)
- **What's New**: UAV 기반 구조 건강 모니터링(Structural Health Monitoring, SHM)의 효용성을 조사하였으며, 하중 하의 기하학적 변형(geometric deformation)을 집중적으로 분석함.

- **Technical Details**: 연구에 사용된 강화 콘크리트 교량은 지반 앵커를 통해 미리 설정된 하중에 노출되었으며, 고해상도 이미지 블록을 촬영하여 변형을 모니터링하였다. 다양한 센싱 기술(Displacement Transducers, Tachymetry, Laser Profiling)을 적용하여 이미지 기반 변형 결과를 평가하였다. UAV 플랫폼으로는 DJI Matrice 600 Pro를 사용하였고, PhaseOne iXM-100 카메라를 장착하였다.

- **Performance Highlights**: 하중 적용 전후의 이미지 분석 결과, 변위 전송 장치(Displacement Transducers)와 비교 시 1 mm 미만의 차이를 보여주었다. UAV 기반 모니터링을 통해 전체 영역의 변형을 정량화할 수 있음을 입증하였다.



### Rethinking Visual Dependency in Long-Context Reasoning for Large Vision-Language Models (https://arxiv.org/abs/2410.19732)
- **What's New**: 이 연구에서는 LVLMs(Long Vision-Language Models)가 긴 맥락에서의 추론에서 성능 저하를 겪는 원인을 분석하고, 비주얼 의존성을 높이는 새로운 텍스트 정보 선택적 제거 방법을 제안했습니다. 이로 인해 LVLMs의 긴 맥락에서의 성능이 향상됩니다.

- **Technical Details**: LVLMs는 비전 인코더와 LLM(Long Language Model)을 포함하여 이미지와 텍스트를 교차 모달(inter-modal) 상호작용을 통해 처리합니다. 연구에서는 훈련 없이 긴 맥락에서 덜 중요한 텍스트 정보를 제거하는 'context pruning' 방법을 통해 비주얼 의존성을 증가시켰습니다.

- **Performance Highlights**: 제안된 방법은 SVIT 데이터셋을 기반으로 한 긴 맥락 데이터셋에서 여러 LVLMs에 대해 검증되었으며, 실험 결과 다양한 토큰 프루닝(token pruning) 전략의 강건성도 입증되었습니다.



### VARS: Vision-based Assessment of Risk in Security Systems (https://arxiv.org/abs/2410.19642)
- **What's New**: 본 연구는 비디오 콘텐츠의 위험 수준을 예측하기 위한 통합적인 머신 러닝 및 딥 러닝 프레임워크를 개발하고, 다양한 모델의 성능을 비교하여 가장 신뢰할 수 있는 방법을 식별합니다.

- **Technical Details**: 우리는 지원 벡터 머신(SVM), 신경망(Neural Networks), 변환기(transformer) 기반 모델을 포함하는 여러 프레임워크를 구현하였습니다. 각 모델은 CLIP와 GPT 임베딩을 결합하여 비디오 위험 예측의 정확도를 높이는 방법을 사용합니다.

- **Performance Highlights**: 비교 분석을 통해, 정확성(accuracy), F1-스코어(F1-score), 평균 절대 오차(MAE)를 활용하여 성능을 평가하고, 비디오 기반 위험 감지의 일반화 가능성을 높이는 데 기여합니다.



### Toward Generalizable Multiple Sclerosis Lesion Segmentation Models (https://arxiv.org/abs/2410.19623)
- **What's New**: 다양한 평가 데이터셋에서 일반화된 MS 병변(segmentation) 분할 모델을 개발하여 실제 임상 시나리오를 반영하는 것이 본 연구의 핵심입니다.

- **Technical Details**: 본 연구에서는 고품질의 공개적으로 이용 가능한 MS 병변 분할 데이터셋을 활용하여 최첨단 UNet++ 아키텍처를 체계적으로 훈련시켰습니다. 우리는 MRI 강도에 대한 양적 정규화(quantile normalization) 기법을 이용하여 모델의 일반화 가능성을 높였습니다.

- **Performance Highlights**: 결과 모델은 잔여 테스트 데이터셋에서 일관된 성능을 보여주며, 더 크고 이질적인 데이터셋이 보다 우수한 모델 결과로 이어졌습니다. 특히, MSSEG2016-train, ISBI2015, 3D-MR-MS 데이터셋을 결합하여 훈련된 모델은 MICCAI-2016 대회의 우승자를 초과하는 성능을 보였습니다.



### Diverse Sign Language Translation (https://arxiv.org/abs/2410.19586)
- **What's New**: 이 논문은 Sign Language Translation (SLT) 모델의 단일 참조 및 데이터의 양에 의한 제약을 극복하기 위해 Diverse Sign Language Translation (DivSLT)라는 새로운 작업을 제안합니다. 이 작업은 손가락 언어 비디오에 대해 다양한 번역 결과를 생성하는 것을 목표로 하고 있습니다.

- **Technical Details**: 본 연구에서는 대형 언어 모델(LLM)을 활용하여 두 개의 주요 손언어 번역 데이터셋씩, 즉 CSL-Daily와 PHOENIX14T에서 다수의 참조 결과를 생성합니다. 모델의 학습 효율성을 높이기 위해 다단계 훈련 패러다임을 도입하고, 강화 학습(max-reward-driven reinforcement learning) 방법을 채택하여 번역 결과의 정확도를 최적화합니다.

- **Performance Highlights**: 확장된 데이터셋에서 실험을 통해, DivSLT 방법은 기존의 SLT 모델에 비해 번역 성능과 다양성이 모두 향상되었음을 보여줍니다. 특히, 풍부한 참고 문헌을 바탕으로 모델이 더 정확하고 다양한 번역 결과를 생성할 수 있음을 강조합니다.



### Prediction of microstructural representativity from a single imag (https://arxiv.org/abs/2410.19568)
- **What's New**: 이번 연구에서는 재료의 단일 이미지(2D 또는 3D)에서 관찰된 위상 분율의 대표성을 예측하는 방법을 제안합니다. 전통적인 접근 방식은 대규모 데이터셋과 광범위한 통계 분석을 요구하는 반면, 우리의 방법은 Two-Point Correlation function을 활용하여 단일 이미지에서 직접 변동성을 추정합니다.

- **Technical Details**: 제안된 모델인 ImageRep은 MicroLib 데이터베이스를 사용하여 개발되었으며, Generative Adversarial Network(GAN)를 통해 다양한 크기의 표본을 합성할 수 있게 해줍니다. TPC를 이용한 빠른 근사값 계산은 단일 이미지를 통해 가능하며, 이는 위상 분율의 신뢰도 추정에 직접 연결됩니다.

- **Performance Highlights**: 이 방법은 다양한 마이크로구조의 개방형 데이터 세트를 사용하여 유효성을 검증하였으며, 제한적인 마이크로구조 데이터로 작업하는 재료 과학자와 엔지니어들에게 실용적인 도구가 될 수 있습니다. 또한, www.imagerep.io에서 제안된 모델을 쉽게 접근할 수 있는 웹 애플리케이션을 제공하여 빠르고 간편하며 유익한 사용을 가능하게 합니다.



### Utilizing Image Transforms and Diffusion Models for Generative Modeling of Short and Long Time Series (https://arxiv.org/abs/2410.19538)
Comments:
          Accepted to NeurIPS 2024; The first two authors contributed equally

- **What's New**: 최근 생성 모델링(generative modeling) 분야에서 시간 시계열 데이터(time series data)의 생성에 대한 관심이 급증하고 있습니다. 본 연구는 시간 시계열을 이미지로 변환하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 invertible transforms(가역 변환)인 delay embedding(지연 임베딩)과 short-time Fourier transform(단기 푸리에 변환)을 사용하여 시간 시계열을 이미지로 변환합니다. 이를 통해 단기 및 장기 입력을 동일한 프레임워크 내에서 처리할 수 있는 장점을 가집니다.

- **Performance Highlights**: 우리는 여러 작업을 통해 제안한 방법의 효과를 검증하였고, 특히 무조건적 생성(unconditional generation) 작업에서 이전 diffusion 모델에 비해 평균 58.17%의 성능 향상을 보여주었습니다. 또한, 매우 긴 시퀀스의 경우 132.61% 향상을 달성했습니다.



### Detection of Emerging Infectious Diseases in Lung CT based on Spatial Anomaly Patterns (https://arxiv.org/abs/2410.19535)
- **What's New**: 이 논문은 새로운 질병의 발생을 탐지하기 위한 혁신적인 접근 방식을 소개합니다. 기존의 지역적 이상 탐지 방법들이 새로운 질병의 확산을 인식하지 못하는 경우가 많다는 점을 고려하여, 세 가지 단계의 방법론을 제안합니다.

- **Technical Details**: 제안된 방법은 CT 영상의 이상 탐지를 통해 질병 패턴을 세그먼트하며, 이를 통해 환자의 지속적이고 새롭게 수집된 데이터를 비교합니다. 3차원 컨볼루션 신경망(convolutional neural network)에서 파생된 그래미트릭(gram-matrix) 기반의 표현을 사용하여 새로운 질병 클러스터를 식별합니다. 이상 탐지는 영상에서 병변의 공간 분포(pattern)를 분석하는 데 필수적입니다.

- **Performance Highlights**: 이 접근 방식은 환자 데이터의 누적 증거를 활용하여 새로운 질병을 조기에 탐지할 수 있으며, 질병의 전염 속도가 빨라질수록 탐지의 소요 시간이 단축됩니다. 실험 결과에 따르면, 질병 부하가 변하지 않더라도 새로운 질병을 효과적으로 식별하는 데 성공하였습니다.



### Conditional Hallucinations for Image Compression (https://arxiv.org/abs/2410.19493)
- **What's New**: 이 연구는 이미지 압축에서 왜곡(distortion)과 지각(perception) 간의 최적 균형을 자동으로 조정하는 최초의 방법론을 제안합니다. 기존의 압축 모델들은 세부 사항을 허구화하거나 분포 밖 샘플을 생성하는 문제에 직면해 있어, 이미지의 내용에 따라 허구를 도입해야 할 필요가 있습니다.

- **Technical Details**: 이 알고리즘은 이미지 내용에 따라 사용자 선호도를 예측할 수 있는 모델을 학습하여, GAN 기반의 손실 함수에서 지각적 가중치를 조정하는 방식으로 작동합니다. 제안하는 모델인 Conditionally Hallucinating Compression Model (ConHa)은 각 이미지의 내용에 따라 허구화 수준을 동적으로 조절하여, 이상적인 분포 내 샘플을 생성합니다.

- **Performance Highlights**: ConHa는 최신 이미지 압축 방법들과 비교하여 높은 성능을 보여주며, 특히 텍스트, 직선 및 작은 얼굴을 포함한 이미지의 경우 왜곡을 최소화하는 반면, 잔디와 같은 텍스처를 포함한 경우에는 허구화를 통해 질감을 재현하는데 집중합니다.



### Evaluation of strategies for efficient rate-distortion NeRF streaming (https://arxiv.org/abs/2410.19459)
- **What's New**: 이번 논문은 Neural Radiance Fields (NeRF)의 스트리밍 기술을 개선하기 위해 두 가지 다양한 스트리밍 전략인 픽셀 기반(pixel-based)과 신경망(neural network) 파라미터 기반 스트리밍을 조사합니다.

- **Technical Details**: NeRF는 3D 포인트를 색상과 불투명도로 매핑하는 볼류메트릭(functional representation) 표현을 사용하여 이미지의 비어 있는 지점에서 포토리얼리스틱(view synthesis) 재구성을 가능하게 합니다. 본 논문에서는 데이터 전송 과정에서의 효율성을 고려하여 픽셀 기반 및 NN 파라미터 기반 스트리밍 전략의 rate-distortion 성능을 평가합니다.

- **Performance Highlights**: 신경망 파라미터 기반 스트리밍 전략이 일반적으로 더 높은 효율성을 제공하며, 이는 한 개의 데이터 소스로부터 여러 목적지로 전송되는 시나리오에 적합하다는 것을 보여줍니다.



### NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction (https://arxiv.org/abs/2410.19452)
Comments:
          NeurIPS 2024 Oral

- **What's New**: NeuroClips는 fMRI를 사용하여 고화질의 매끄러운 비디오를 재구성하는 혁신적인 프레임워크입니다. 이 방법은 두 가지 주요 구성 요소인 Perception Reconstructor와 Semantics Reconstructor를 통해 구현됩니다.

- **Technical Details**: NeuroClips는 비디오 키프레임을 재구성하기 위해 semantics reconstructor를 활용하고, 영상의 매끄러움을 보장하기 위해 퍼셉션 세부 사항을 캡처하는 perception reconstructor를 채택합니다. 이 논문은 pre-trained T2V diffusion model을 통해 fMRI에서 비디오를 재구성하는 것을 정교하게 수행합니다.

- **Performance Highlights**: NeuroClips는 공개된 fMRI-비디오 데이터셋에서 최대 6초 길이의 비디오를 8FPS로 매끄럽게 재구성하는 성과를 보여주며, SSIM에서 128% 향상, 시공간 메트릭에서도 81% 개선된 결과를 기록했습니다.



### Integration of Communication and Computational Imaging (https://arxiv.org/abs/2410.19415)
- **What's New**: 이번 논문에서는 통신(communication)과 계산 영상(computational imaging)의 경계를 허물기 위한 새로운 프레임워크인 ICCI(Integrative Communication and Computational Imaging)를 제안합니다. 이 프레임워크는 원거리 인식(remote perception)을 위한 통합 솔루션을 제공하며, 정보 손실을 최소화하는 최적화를 목표로 합니다.

- **Technical Details**: ICCI 프레임워크는 원거리 비주얼 정보의 감지(sensing)와 전송(transmitting)을 동시에 고려하여 전체 링크 정보 전송 최적화(full-link information transfer optimization)를 구현합니다. 이는 통신 시스템과 스냅샷 압축 영상 시스템(snapshot compressive imaging systems)의 통합을 통해 나타납니다.

- **Performance Highlights**: ICCI 방식은 채널 노이즈(channel noise)와 손상에 대한 강인성을 보여주며, 높은 데이터 압축(data compression)을 달성합니다. 또한, 80 km 거리에서 30 fps의 속도로 27밴드 하이퍼스펙트럴 비디오(hyperspectral video) 인식을 실험적으로 성취했습니다.



### Context-Based Visual-Language Place Recognition (https://arxiv.org/abs/2410.19341)
- **What's New**: 이 논문에서는 로봇의 시각 기반 위치 인식(Visual Place Recognition, VPR) 문제를 해결하기 위한 새로운 방법론을 제안합니다. 기존의 방법들의 한계를 극복하고, 추가 학습이 필요 없는 방법으로 환경 변화에 강인한 VPR 방식을 구현했습니다.

- **Technical Details**: 제안하는 방법은 제로샷(zero-shot) 언어 기반 의미 분할(semantic segmentation) 모델을 활용하여 픽셀 수준의 언어 임베딩을 추출하고 이를 바탕으로 의미 있는 이미지 서술어를 생성합니다. 이 서술어는 비주얼-언어(vocabulary) 인식 모듈에서 사용되어 복잡한 환경에서도 장소를 효과적으로 인식할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존의 비학습 기반 이미지 표현 기법과 일반적인 CNN(Convolutional Neural Network) 서술어들보다 뛰어난 성능을 보였으며, 다양한 환경에서 재방문한 위치를 정확히 인식하는 것이 가능함을 보여주었습니다.



### Beyond Point Annotation: A Weakly Supervised Network Guided by Multi-Level Labels Generated from Four-Point Annotation for Thyroid Nodule Segmentation in Ultrasound Imag (https://arxiv.org/abs/2410.19332)
- **What's New**: 이 논문에서는 여러 포인트 주석을 기반으로 다단계 라벨을 생성하는 약한 감독 네트워크를 제안하여 갑상선 결절(segmentation of thyroid nodules) 분할을 개선했습니다. 이 네트워크는 결절과 배경의 섬세한 특성 차이를 나타내는 데 어려움을 겪었던 기존의 약한 감독 방법들보다 우수한 성능을 발휘합니다.

- **Technical Details**: 제안된 네트워크는 Distance-Similarity Fusion Prior를 도입하여 주석된 포인트와의 거리 및 그레이스케일 차이를 기반으로 정보를 필터링합니다. 주석으로부터 생성된 경계 상자(bounding box) 및 순수 전경/배경(labels) 라벨은 결절의 위치 및 공간 분포를 보장합니다. 이를 통해 구분된 특성을 학습하고 피험 데이터셋에서 우수한 성능을 보였습니다.

- **Performance Highlights**: TN3K 데이터셋에서 Jaccard 계수 82.36%, Dice 계수 89.85%를 기록하였고, DDTI 데이터셋에서는 Jaccard 계수 84.33%, Dice 계수 91.37%로 기존 약한 감독 네트워크보다 성능이 크게 향상되었습니다.



### A Flow-based Truncated Denoising Diffusion Model for Super-resolution Magnetic Resonance Spectroscopic Imaging (https://arxiv.org/abs/2410.19288)
Comments:
          Accepted by Medical Image Analysis (MedIA)

- **What's New**: 본 논문은 Magnetic Resonance Spectroscopic Imaging (MRSI)를 위한 새로운 Flow-based Truncated Denoising Diffusion Model (FTDDM)을 제안합니다. 이 모델은 낮은 해상도의 MRSI 데이터를 이용하여 고해상도의 이미지를 생성하는 데 중점을 두고 있으며, 기존의 생성 모델들보다 우수한 성능을 보입니다.

- **Technical Details**: FTDDM은 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하여, 확산 과정을 단축시키기 위해 이론적 목적으로 정상화 흐름(normalizing flow) 네트워크를 사용합니다. 또한, Conditional Instance Normalization을 통해 업스케일링 인자에 조건을 주어 멀티-스케일(super-resolution) 처리를 수행합니다.

- **Performance Highlights**: 실험 결과, FTDDM은 기존의 GAN 및 flow 기반 모델을 능가하며, DDPM에 비해 샘플링 과정을 9배 이상 가속화했습니다. 신경영상의학 전문가들은 제안된 방법의 임상적 효용성을 확인했습니다.



### ST-NeRP: Spatial-Temporal Neural Representation Learning with Prior Embedding for Patient-specific Imaging Study (https://arxiv.org/abs/2410.19283)
Comments:
          14 pages with 10 figures and 6 tables

- **What's New**: 이번 논문에서는 환자 맞춤형 이미징 연구를 위한 공간-시간 신경 표현 학습 방법(ST-NeRP)을 제안합니다.

- **Technical Details**: ST-NeRP 모델은 참조 시점에서 이미지를 인코딩하기 위해 암시적 신경 표현(Implicit Neural Representation, INR) 네트워크를 사용하며, 또 다른 INR 네트워크를 통해 공간-시간 지속적인 변형 함수(deformation function)를 학습합니다.

- **Performance Highlights**: ST-NeRP 모델은 4D CT 및 흉부와 복부 이미징을 포함한 다양한 순차적 이미지 시리즈에 적용되어 생기는 해부학적 변화를 효과적으로 모니터링할 수 있는 상당한 잠재력을 보여줍니다.



### Non-rigid Relative Placement through 3D Dense Diffusion (https://arxiv.org/abs/2410.19247)
Comments:
          Conference on Robot Learning (CoRL), 2024

- **What's New**: 본 논문에서는 ''relative placement'' (상대 배치)의 개념을 변화하는 물체에 대한 기하학적 관계로 확장하는 ''cross-displacement'' (교차 변위)를 제안합니다. 이를 통해 비틀림이 가능한 물체를 대상으로 하는 새로운 시각 기반 방법을 소개합니다.

- **Technical Details**: 이 방법은 dense diffusion (밀집 확산)을 통해 cross-displacement 를 학습하며, 이전의 연구에서는 다루지 못했던 비형상 물체의 기하학적 관계를 모형화합니다. 이는 물체의 변형(transformations)과 같은 복잡한 상황에서도 효과적인 학습이 가능함을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 다양한 비형상 작업에서 일반적인 실제 환경과 시뮬레이션에서의 새로운 객체 인스턴스, 분포 외(scene configurations) 장면에 대한 일반화를 달성했습니다. 여러 가지 다중 모드(multimodal) 목표에서도 뛰어난 성능을 입증하였습니다.



### VisionCoder: Empowering Multi-Agent Auto-Programming for Image Processing with Hybrid LLMs (https://arxiv.org/abs/2410.19245)
- **What's New**: 이 논문은 자동 프로그래밍 분야에서 대형 언어 모델(LLMs)을 활용한 혁신적인 다중 에이전트 프레임워크인 VisionCoder를 소개한다. 이 프레임워크는 여러 LLM을 조합하여 복잡한 이미지 처리 작업을 자동화하는 데 중점을 둔다.

- **Technical Details**: VisionCoder는 Tree of Thoughts(ToT) 접근 방식을 채택하여, 프로젝트, 모듈, 함수 레벨에서 생각의 계층 구조를 설정해 각 에이전트가 특정 역할을 수행하도록 한다. 이 시스템은 GPT-4o와 deeepseek-coder-7b-instruct-v1.5를 하이브리드 모델로 활용하여 에이전트 간의 협력을 통해 자동화된 코드 생성을 수행한다. 또한, Retrieval-Augmented Generation(RAG) 기법을 도입해 작업 비대칭을 줄이고, Pair Programming을 통해 최종 코드의 정확성을 높인다.

- **Performance Highlights**: 실험 결과, VisionCoder는 이미지 처리 자동 프로그래밍 작업에서 기존 방법보다 상당히 뛰어난 성능을 보였다. 이 프레임워크는 효율적이고 비용 효과적인 솔루션을 제공하며, 향후 연구를 위한 자동 프로그래밍 벤치마크 데이터를 개발하였다.



### The Empirical Watershed Wav (https://arxiv.org/abs/2410.19187)
- **What's New**: 이 논문에서는 기존 2D 경험적 웨이브렛 변환의 제한을 극복하기 위해 주파수 영역에서 임의의 분할을 사용하여 2D 경험적 웨이브렛 필터를 구성하는 방법을 제안했습니다. 새로운 알고리즘을 통해 이미지 스펙트럼에서 주요 조화 모드의 위치를 추정할 수 있게 되었습니다.

- **Technical Details**: 제안된 알고리즘은 스케일-스페이스 표현(scale-space representation)과 수조 변환(watershed transform)을 결합하여 여러 지원(support)을 자동으로 감지하는 가능성을 열어줍니다. 이를 통해 경험적 수조 웨이브렛 변환(empirical watershed wavelet transform, EWWT)을 정의하였습니다.

- **Performance Highlights**: EWWT의 성능을 토이 이미지와 비지도 텍스처 분할 및 이미지 디콘볼루션(nonblind deconvolution) 응용 프로그램에서 시각적으로 평가하였으며, 기존 방법보다 우수한 성능을 보여주었습니다.



### CapsuleNet: A Deep Learning Model To Classify GI Diseases Using EfficientNet-b7 (https://arxiv.org/abs/2410.19151)
Comments:
          Capsule Vision 2024 Challenge

- **What's New**: 이 논문에서는 Capsule Endoscopy (CE)에서 사용하기 위해 개발한 딥러닝 모델인 CapsuleNet을 소개합니다. 이 모델은 10개의 다양한 위장 질환(Gastrointestinal diseases) 이상을 분류하기 위해 설계되었습니다.

- **Technical Details**: CapsuleNet은 EfficientNet-b7 백본을 활용하여 사전 훈련된 모델로, PReLU 활성화 함수를 통해 최적화되었습니다. 데이터 불균형 문제를 해결하기 위해 다수의 데이터 증강(data augmentation) 기법이 사용되었고, 최종적으로 클래스당 1500개의 샘플로 줄였습니다.

- **Performance Highlights**: 모델은 검증 데이터에서 마이크로 정확도(micro accuracy) 84.5%를 달성하여 VGG16 기준 모델을 대부분의 클래스에서 초과했습니다. 특히 Erythema 클래스에서의 성능 지표는 저조하여, 추가적인 연구가 필요하다는 결과를 도출했습니다.



### Teach Multimodal LLMs to Comprehend Electrocardiographic Images (https://arxiv.org/abs/2410.19008)
- **What's New**: 이번 연구에서는 ECG(심전도) 이미지 해석을 위한 새로운 Datasets인 ECGInstruct를 소개합니다. 이 데이터셋은 100만 개 이상의 ECG 이미지-텍스트 샘플로 구성되어 있으며, 다양한 ECG 관련 작업을 포함하고 있습니다.

- **Technical Details**: ECGInstruct는 현실적 이미지 합성과 임상 전문가의 통찰을 반영한 다양한 ECG 관련 작업을 특징으로 하며, PULSE라는 MLLM(Multimodal Large Language Model) 모델을 개발하였습니다. 또한, ECGBench라는 새로운 평가 벤치마크를 설정하였고, 이를 통해 ECG 해석 성능을 측정합니다.

- **Performance Highlights**: PULSE는 다양한 벤치마크에서 평균 15%에서 30%의 정확도 개선을 달성하며 기존의 일반 MLLM들을 능가하여 심전도 해석의 새로운 최첨단 성능을 설정하였습니다.



New uploads on arXiv(cs.AI)

### The Potential and Value of AI Chatbot in Personalized Cognitive Training (https://arxiv.org/abs/2410.19733)
- **What's New**: 최근 전 세계 인구의 급격한 고령화가 치매와 같은 인지 장애의 증가를 초래했습니다. 특히, 아미로이드(beta-amyloid) 기반 알츠하이머 치료법이 없는 상황에서, 인지 훈련과 같은 예방 및 조기 개입이 중요해졌습니다. 본 연구에서는 AI 챗봇인 ReMe를 통해 개인화된 인지 훈련을 지원하기 위한 웹 기반 프레임워크를 소개합니다.

- **Technical Details**: ReMe 프레임워크는 3개의 주요 구성요소인 퍼즐 엔진(puzzle engine), 라이프 로깅 모듈(life-logging module), 훈련 사용자 인터페이스(training user interface)로 구성됩니다. 퍼즐 엔진은 인지 훈련 과제를 생성하고 관리하며, 라이프 로깅 모듈은 사용자가 일상 활동과 경험을 기록할 수 있도록 돕습니다. 사용자는 멀티모달 입력(text, image, voice)을 통해 챗봇과 상호작용하며, 사용자 맞춤형 피드백을 받을 수 있습니다.

- **Performance Highlights**: ReMe는 개인의 삶의 기록을 바탕으로 사용자를 참여시키기 위해 설계된 인지 훈련 퍼즐을 통해 시나리오 기반 기억 훈련의 효과를 높일 수 있음을 사례 연구를 통해 입증했습니다. 그러나 대규모 연구를 통한 훈련 효과의 검증이 필요합니다. ReMe는 비약물적 개입에 대한 수요 증가에 발 맞춘 개인화된 인지 훈련 접근 방식을 제공합니다.



### FISHNET: Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert Swarms, and Task Planning (https://arxiv.org/abs/2410.19727)
Comments:
          Accepted at the 5th ACM International Conference on AI in Finance (ICAIF '24)

- **What's New**: 이번 연구에서는 전통적인 금융 데이터 분석의 방법론을 넘어, FISHNET (Financial Intelligence from Sub-querying, Harmonizing, Neural-Conditioning, Expert swarming, and Task planning)라는 새로운 에이전트 기반 아키텍처를 제안합니다. 이 시스템은 다양한 규제 제출 문서에서 복잡한 재무 정보를 생성하는 데 뛰어난 성능을 발휘합니다.

- **Technical Details**: FISHNET은 Swarming Large Language Models (LLMs)과 Agent-Based Modeling (ABM)을 통합하여 다양한 규제 filings를 효과적으로 분석합니다. 이 시스템은 동시 분석과 데이터 조정, 작업 계획 및 전문가 에이전트를 활용하여 98,034개의 미국 규제 제출 문서에서 금융 통찰력을 생성합니다.

- **Performance Highlights**: FISHNET은 61.8%의 성공률을 기록하고 기존 기법들과 비교 시 현저한 성과를 보여주며, 특히 Retrieval Augmented Generation (RAG)와 Generative Routing과의 성능 비교에서 우수한 결과를 자랑합니다.



### VARS: Vision-based Assessment of Risk in Security Systems (https://arxiv.org/abs/2410.19642)
- **What's New**: 본 연구는 비디오 콘텐츠의 위험 수준을 예측하기 위한 통합적인 머신 러닝 및 딥 러닝 프레임워크를 개발하고, 다양한 모델의 성능을 비교하여 가장 신뢰할 수 있는 방법을 식별합니다.

- **Technical Details**: 우리는 지원 벡터 머신(SVM), 신경망(Neural Networks), 변환기(transformer) 기반 모델을 포함하는 여러 프레임워크를 구현하였습니다. 각 모델은 CLIP와 GPT 임베딩을 결합하여 비디오 위험 예측의 정확도를 높이는 방법을 사용합니다.

- **Performance Highlights**: 비교 분석을 통해, 정확성(accuracy), F1-스코어(F1-score), 평균 절대 오차(MAE)를 활용하여 성능을 평가하고, 비디오 기반 위험 감지의 일반화 가능성을 높이는 데 기여합니다.



### Planning-Aware Diffusion Networks for Enhanced Motion Forecasting in Autonomous Driving (https://arxiv.org/abs/2410.19639)
Comments:
          Accepted by CoRL Workshop Leap 2024

- **What's New**: 본 논문에서는 다중 에이전트 환경에서의 상호작용을 고려한 Planning-Integrated Forecasting Model (PIFM)이라는 새로운 프레임워크를 제안합니다. 이 모델은 예측의 정확성과 해석 가능성을 높이기 위해 도로 구조, 교통 규칙 및 주변 차량의 행동을 통합합니다.

- **Technical Details**: PIFM은 신경망의 의사결정 및 다중 에이전트 조정 메커니즘에서 영감을 받아 설계되었으며, 확산 기반 아키텍처를 채택하여 모든 에이전트의 미래 궤적을 예측합니다. 이 모델은 외부 자극과 다른 에이전트의 행동에 따라 예측을 동적으로 조정하는 방법과 유사하여 투명성을 높입니다.

- **Performance Highlights**: PIFM은 안전하고 효율적인 자율 주행 시스템을 위한 해석 가능한 솔루션을 제공하며, 매우 적은 수의 파라미터로도 우수한 성능을 발휘하는 것으로 실험을 통해 검증되었습니다. 기존 모델 대비 80% 이상의 계산 복잡도 감소를 달성하였습니다.



### Knowledge Graph Enhanced Language Agents for Recommendation (https://arxiv.org/abs/2410.19627)
- **What's New**: 최근 언어 에이전트는 추천 시스템에서 인간 행동 및 사용자-아이템(interaction) 상호작용을 시뮬레이션하는 데 사용되고 있습니다. 하지만 기존의 시뮬레이션은 사용자와 아이템 간의 관계를 이해하지 못해 부정확한 사용자 프로필과 비효율적인 추천을 초래합니다.

- **Technical Details**: 이 연구에서는 사용자와 아이템 간의 광범위하고 신뢰할 수 있는 관계를 포함하는 지식 그래프(Knowledge Graphs, KGs)의 유용성을 탐구합니다. 우리의 주요 통찰력은 KG의 경로들이 사용자와 아이템 간의 복잡한 관계를 포착하여 사용자 선호의 근본적인 이유를 끌어내고 사용자 프로필을 풍부하게 만든다는 것입니다. 우리는 KGLA(지식 그래프 강화 언어 에이전트)라는 프레임워크를 제안하여 추천 시스템을 위한 언어 에이전트와 KG를 통합합니다.

- **Performance Highlights**: 실험 결과, KGLA는 추천 성능을 상당히 향상시켰으며(세 가지 널리 사용되는 벤치마크 중 NDCG@1에서 33%-95%의 증가) 이전의 최적 기준 방법과 비교하여 뚜렷한 성과를 보였습니다.



### Shared Control with Black Box Agents using Oracle Queries (https://arxiv.org/abs/2410.19612)
- **What's New**: 이번 연구에서는 로봇이 인간과 협동 학습을 할 때, 질의(query)를 해 궁극적으로 공유 제어 정책의 효율성을 높이는 방안에 대해 제안합니다. 기존의 접근법은 실행을 통해서만 블랙 박스(black box)와 상호작용할 수 있다고 가정했지만, 본 연구에서는 이를 확장하여 협력 에이전트에 직접적으로 질문할 수 있는 새로운 프레임워크를 제시합니다.

- **Technical Details**: 제안된 모델은 Multi-Agent Markov Decision Process (MA-MDP)를 기반으로 하며, 두 가지 종류의 오라클(oracle) 응답을 고려합니다: 첫째, 가장 좋은 행동을 제공할 수 있는 지식이 가득한 오라클, 둘째, 단지 자신의 시스템의 일부분에 대한 지식만을 가진 제한된 오라클입니다. 이러한 질의 응답을 통해 훈련 중에 발생할 수 있는 오류를 줄이고, 효율적인 정책 학습을 도모합니다.

- **Performance Highlights**: 두 개의 환경에서 실험한 결과, 질의를 활용함으로써 공유 제어 정책의 학습 성능이 유의미하게 향상됨을 보여주었고, 제안된 세 가지 휴리스틱(heuristics) 접근법의 성능 차이를 분석하여, 언제 질의를 할 것인지에 대한 중요한 통찰을 제공했습니다.



### Bongard in Wonderland: Visual Puzzles that Still Make AI Go Mad? (https://arxiv.org/abs/2410.19546)
- **What's New**: 최근 OpenAI의 GPT-4o와 같은 최신 Vision-Language Models (VLMs)이 등장하였으며, 이 모델들은 텍스트와 이미지의 융합된 인식 능력을 보여주고 있습니다. 그러나 이러한 발전이 실제로 얼마나 깊이 있는지는 아직 잘 알려져 있지 않습니다.

- **Technical Details**: 이 연구에서는 Bongard 문제라는 고전적인 시각적 추론 퍼즐을 활용하여 VLM의 성능을 평가했습니다. 이 문제들은 사람과 유사한 패턴 인식 및 추상적 사고 능력을 요구하며, VLM은 일부 개념을 식별하고 문제를 해결할 수 있지만, 종종 이를 실패합니다. 특히, 단순한 나선형과 같은 초보 개념조차 VLM에게 큰 도전 과제가 됩니다.

- **Performance Highlights**: VLM은 명백히 특정 개념에 집중하여 분석하라는 요청을 받더라도, 기본적인 시각 개념을 이해하지 못하고, 보지 못한 개념에 일반화하는 능력도 부족함을 보여줍니다. 이는 VLM의 현재 한계를 강조하며, 인간과 유사한 시각적 추론과 기계 인지 간의 상당한 격차가 여전히 존재함을 시사합니다.



### EDGE: Enhanced Grounded GUI Understanding with Enriched Multi-Granularity Synthetic Data (https://arxiv.org/abs/2410.19461)
- **What's New**: 본 논문에서는 다양한 어플리케이션의 그래픽 사용자 인터페이스(GUI)에서 작동하는 자율 에이전트의 중요성을 강조하고, 기존의 낮은 품질의 훈련 데이터를 다루기 위해 EDGE라는 데이터 합성 프레임워크를 제안합니다.

- **Technical Details**: EDGE는 웹 페이지에서 대규모 다중 세분형 훈련 데이터를 자동으로 생성하여 LVLMs의 GUI 이해 및 상호작용 능력을 향상시키는 데이터 기반 접근 방식을 채택합니다. 이 프레임워크는 요소 수준의 레이블 및 풍부한 잠재적 요소를 추출하고, 다중 세분형 QA 작업을 통합하여 복잡한 웹 페이지와 액션 그라운딩 시나리오에서 필요한 글로벌 의미 이해 및 추론 능력을 커버합니다.

- **Performance Highlights**: EDGE를 통해 생성된 데이터셋으로 훈련된 모델은 GUI 벤치마크에서 뛰어난 성능을 보였으며, 이는 기존에 보지 못한 데스크탑 및 모바일 환경으로 쉽게 전이될 수 있습니다. 이 접근 방식은 수동 주석에 대한 의존도를 크게 줄이고, 연구자들이 웹에서 활용 가능한 방대한 공공 자원을 활용해 새로운 연구를 진전시킬 수 있도록 지원합니다.



### Offline-to-Online Multi-Agent Reinforcement Learning with Offline Value Function Memory and Sequential Exploration (https://arxiv.org/abs/2410.19450)
- **What's New**: 오프라인에서 온라인으로의 강화학습(Offline-to-Online Reinforcement Learning, O2O RL)은 오프라인 데이터를 초기화에 활용하고, 온라인 단계에서 조정하여 샘플 효율성과 성능을 향상시키는데 강력한 패러다임으로 부상했습니다. 본 논문에서는 O2O MARL(Offline-to-Online Multi-Agent Reinforcement Learning)의 새로운 프레임워크인 OVMSE(Offline Value Function Memory with Sequential Exploration)를 제안하고, 기존의 단일 에이전트 환경에서 벗어나 다중 에이전트 문제를 다루고 있습니다.

- **Technical Details**: OVMSE는 두 가지 핵심 구성 요소로 이루어져 있습니다: 오프라인 가치 함수 메모리(OVM)와 분산된 순차 탐색(Sequential Exploration, SE) 전략입니다. OVM은 목표 Q-value를 계산하기 위한 새로운 메커니즘을 도입하여, 오프라인 훈련 중 얻은 지식을 보존하고 부드러운 전환을 보장합니다. SE는 O2O MARL에 맞게 설계된 탐색 전략으로, 사전에 훈련된 오프라인 정책을 활용하여 탐색의 효율성을 높입니다.

- **Performance Highlights**: StarCraft Multi-Agent Challenge(SMAC)에서 수행된 광범위한 실험에서 OVMSE는 기존의 기준선과 비교하여 샘플 효율성과 전반적인 성능에서 현저한 개선을 보여주었습니다. OVMSE는 더 높은 샘플 효율성과 빠른 온라인 조정을 달성하며, 복잡한 다중 에이전트 환경에서의 실용성을 강조합니다.



### Expose Before You Defend: Unifying and Enhancing Backdoor Defenses via Exposed Models (https://arxiv.org/abs/2410.19427)
Comments:
          19 pages

- **What's New**: 새로운 연구에서는 Expose Before You Defend (EBYD)라는 새로운 두 단계 방어 프레임워크를 도입하여 기존의 백도어 방어 방법을 통합하여 성능을 향상시킵니다. EBYD는 우선 백도어 모델의 백도어 기능을 드러내는 backdoor exposure 단계로 시작하여, 이를 통해 탐지 및 제거 방법을 적용하는 방식으로 진행됩니다.

- **Technical Details**: EBYD는 두 단계로 구성되며, 첫 번째 단계인 backdoor exposure에서는 Clean Unlearning (CUL)이라는 새로운 기술을 소개하여 백도어 모델에서 청정 기능을 'unlearn'하여 숨겨진 백도어 기능을 드러냅니다. 이 과정에서 모델 수정 기술인 fine-tuning, model sparsification, weight perturbation 등을 탐색합니다. 이 백도어를 노출한 모델은 후속 방어 작업에 있어 더욱 유리한 출발점을 제공합니다.

- **Performance Highlights**: EBYD는 CIFAR-10 및 ImageNet subset과 같은 이미지 데이터셋과 SST-2, IMDB, Twitter, AG's News와 같은 언어 데이터셋에서 10개의 이미지 공격과 6개의 텍스트 공격에 대해 광범위한 실험을 진행하였고, 결과는 EBYD가 현재의 최첨단 방법들보다 상당한 성능 개선을 달성함을 보여줍니다.



### Learning Neural Strategy-Proof Matching Mechanism from Examples (https://arxiv.org/abs/2410.19384)
- **What's New**: 본 논문에서는 전략 증명(Strategy-Proof) 매칭 메커니즘을 위한 새로운 매개변수화된 패밀리를 제안하며, 기존의 직렬 독재(Serial Dictatorship) 방법을 확장하여 NeuralSD라는 주목 기반의 신경망을 개발했습니다.

- **Technical Details**: NeuralSD는 공공 맥락 정보를 포함한 데이터셋에서 학습하여 전략 증명 매칭 메커니즘을 구축합니다. 이 모델은 텐서(tensor) 연산을 통해 직렬 독재를 미분 가능하게 만들어 최적의 매칭 순서를 계산합니다. 또한, 주목 기반의 하위 네트워크를 사용하여 에이전트의 맥락에서 SD의 순서를 추정합니다.

- **Performance Highlights**: NeuralSD는 다양한 에이전트 수에 대한 매칭 예제를 학습하여, 기존의 RSD(무작위 직렬 독재)와 비교하여 매칭 에러 및 기타 메트릭에서 우수한 성능을 입증했습니다. 실험 결과, NeuralSD는 맞춤형 매칭에서 더 나은 결과를 내며, 공공 맥락을 반영하여 사회적 가치를 향상시키는 데 기여합니다.



### Engineering Trustworthy AI: A Developer Guide for Empirical Risk Minimization (https://arxiv.org/abs/2410.19361)
- **What's New**: 이 논문은 신뢰할 수 있는 AI(system) 구축을 위한 주요 요구 사항이 경험적 위험 최소화(empirical risk minimization, ERM) 구성 요소에 어떻게 적용될 수 있는지를 논의합니다.

- **Technical Details**: 이 연구는 AI 시스템에서 정확성을 중시하는 기존 ERM 접근 방식의 한계를 지적하고, 신뢰성을 고려한 설계 선택이 어떻게 이루어져야 하는지를 설명합니다. 주요 요구 사항으로는 편향(bias), 불투명성(opacity) 감소가 포함됩니다.

- **Performance Highlights**: 의사 결정 및 개인적, 사회적 영역에서 AI의 영향을 고려할 때, 신뢰할 수 있는 AI 구축을 위한 실용적인 지침을 제공하여 최신 AI 신뢰성 기준을 충족하는 시스템 개발에 기여하고자 합니다.



### LArctan-SKAN: Simple and Efficient Single-Parameterized Kolmogorov-Arnold Networks using Learnable Trigonometric Function (https://arxiv.org/abs/2410.19360)
Comments:
          7 pages, 3 figures, experiment code is available at this https URL

- **What's New**: 본 논문은 삼각 함수로 구성된 Single-Parameterized Function (SFunc)을 활용하여 Single-Parameterized Kolmogorov-Arnold Networks (SKAN)의 새로운 설계 방법을 제안합니다. 세 가지 새로운 SKAN 변형인 LSin-SKAN, LCos-SKAN 및 LArctan-SKAN이 개발되었습니다.

- **Technical Details**: 이 연구는 MNIST 데이터 세트에서 LArctan-SKAN이 정확성과 계산 효율성 모두에서 뛰어나다는 것을 입증했습니다. 특히, LArctan-SKAN은 FourierKAN, LSS-SKAN, 그리고 Spl-KAN과 같은 순수 KAN 변형 모델을 모두 초월합니다. LArctan-SKAN의 훈련 속도는 MLP+rKAN 및 MLP+fKAN과 비교할 때 각각 535.01% 및 49.55% 향상되었습니다.

- **Performance Highlights**: LArctan-SKAN은 모든 순수 KAN 네트워크의 성능을 초과할 뿐만 아니라 MLP 기반 혼합 모델인 MLP+rKAN 및 MLP+fKAN보다도 뛰어난 정확성을 기록했습니다. LArctan-SKAN은 FourierKAN 대비 0.93%, LSS-SKAN 대비 0.53%, Spl-KAN 대비 1.94% 등의 개선을 보여주었습니다. 이러한 결과는 주어진 데이터 세트에서 SKAN의 효과와 가능성을 뒷받침합니다.



### A prescriptive theory for brain-like inferenc (https://arxiv.org/abs/2410.19315)
- **What's New**: 이번 연구에서는 포아송(Poisson) 가정을 기반으로 하는 ELBO(증거 하한)를 최대화하여 스파이킹 뉴럴 네트워크(spiking neural network)를 도출하는 방법을 제시합니다. 이를 통해 뇌 기능과 기계 학습의 통합 이론을 가능하게 합니다.

- **Technical Details**: 새롭게 제안된 모델인 iterative Poisson VAE(iP-VAE)는 신경세포 의사소통 방식인 스파이크를 통해 베이지안 사후 추론(Bayesian posterior inference)을 수행합니다. iP-VAE는 반복적 추론(iterative inference)을 통해 뇌의 생물학적 뉴런과 밀접하게 연결되어 있습니다. 모델은 메모리 전위(dynamic potential) 변화를 기반으로 예측 코딩(predicitive coding)의 효과를 보입니다.

- **Performance Highlights**: iP-VAE는 희소 표현(sparse representation) 습득 및 일반화 성능에서 기존의 변동 VAE보다 우수한 성능을 보입니다. 특히, iP-VAE는 OOD(out-of-distribution) 샘플에서 뛰어난 일반화 능력을 보여 주며, 에너지 효율적인 하드웨어 구현에 적합합니다.



### Autonomous Building Cyber-Physical Systems Using Decentralized Autonomous Organizations, Digital Twins, and Large Language Mod (https://arxiv.org/abs/2410.19262)
Comments:
          40 pages, 22 figures

- **What's New**: 본 논문에서는 자율 건물 연구에서의 한계를 해결하기 위해 새로운 방식의 분산 자율 건물 사이버-물리적 시스템 프레임워크를 소개합니다. 이 시스템은 분산 자율 조직(Decentralized Autonomous Organizations), 대규모 언어 모델(Large Language Models) 및 디지털 트윈(Digital Twins)을 통합하여 스마트하고 자율적으로 운영되는 건물 인프라를 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 건물 인프라의 분산 거버넌스를 촉진하기 위해 전체 스택 분산 애플리케이션을 개발하였으며, LLM 기반의 인공지능 어시스턴트를 통해 블록체인 및 건물 운영 관리와 관련된 작업을 직관적으로 수행할 수 있도록 지원합니다. 이 시스템은 차세대 AI를 이용한 자율 건물 운영을 가능하게 합니다.

- **Performance Highlights**: 여섯 가지 실제 시나리오를 통해 자율 건물 시스템의 유용성을 평가한 결과, 수익 및 비용 관리, AI 지원 시설 제어, 자율 시스템 조정 등의 작업을 성공적으로 수행하여 프레임워크의 적합성을 입증하였습니다.



### Designing LLM-Agents with Personalities: A Psychometric Approach (https://arxiv.org/abs/2410.19238)
- **What's New**: 이 연구는 대형 언어 모델 기반의 에이전트(Agents)에 Big Five 성격 이론을 적용하여 측정 가능하고 조절 가능한 성격을 부여하는 새로운 방법론을 제안합니다. 이는 인간 대상 연구의 제약을 극복하고 사회 과학 연구를 위한 접근 가능한 도구로서 에이전트를 사용하려고 합니다.

- **Technical Details**: 이 연구는 네 가지 연구를 통해 에이전트에 심리 측정적으로 유효한 성격 특성을 할당하는 가능성을 보여줍니다. 첫 번째 연구는 LLM의 의미적 공간 내에서 성격 구성과 성격 검사의 이해를 설정합니다. 이후 두 개의 연구는 경험적 및 시뮬레이션 데이터를 사용하여 에이전트를 생성하는 과정을 설명하고, 성격 검사에 대한 인간과 에이전트의 응답 간 강한 일치를 통해 결과를 검증합니다.

- **Performance Highlights**: 마지막 연구는 에이전트를 사용하여 위험 감수 및 윤리적 딜레마 상황에서 성격 특성과 의사 결정 행동 간의 알려진 인간 간 상관관계를 반복함으로써 이러한 일치를 추가적으로 확인합니다. 이로써 에이전트 설계에 대한 심리 측정적 접근의 효과성과 사회 및 행동적 연구에의 적용 가능성을 검증합니다.



### Integrating Large Language Models with Internet of Things Applications (https://arxiv.org/abs/2410.19223)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 IoT(사물 인터넷) 네트워크의 지능화 및 반응성을 향상시키는 데 어떻게 기여할 수 있는지를 세 가지 사례 연구를 통해 분석합니다. 주요 내용으로는 DDoS 공격 탐지, IoT 시스템을 위한 매크로 프로그래밍, 센서 데이터 처리 등이 있습니다.

- **Technical Details**: LLM인 GPT 모델은 few-shot learning을 통해 87.6%의 탐지 정확도를 기록하였고, fine-tuning을 통해 94.9%로 증가했습니다. 또한, LLM은 매크로 프로그래밍 프레임워크를 바탕으로 다양한 스크립트를 작성할 수 있으며, 방대한 양의 센서 데이터를 빠르고 질 높은 응답으로 처리하는 능력을 보입니다.

- **Performance Highlights**: LLM은 DDoS 공격 탐지에 효과적이며, 다양한 IoT 장치에서 데이터를 통합하여 시스템의 상태를 종합적으로 관리할 수 있습니다. 또한, LLM을 통해 더욱 직관적인 인터페이스와 사용자 친화적인 API를 제공하여 복잡한 IoT 시스템을 자연어 쿼리를 통해 쉽게 조작할 수 있는 가능성을 보여주었습니다.



### MAP: Multi-Human-Value Alignment Pa (https://arxiv.org/abs/2410.19198)
- **What's New**: 이 논문에서는 인간 가치 정렬(human value alignment)의 새로운 접근 방식인 Multi-Human-Value Alignment Palette (MAP)를 제안합니다. 이 방법은 여러 인간 가치를 구조적이고 신뢰할 수 있는 방법으로 정렬할 수 있도록 돕습니다.

- **Technical Details**: MAP는 사용자 정의 제약 조건을 사용하여 최적화 작업으로 정렬 문제를 공식화합니다. 이 접근법은 primal-dual 방식으로 효율적으로 해결되며, 사용자 정의 정렬 목표의 달성 가능성과 방법을 결정합니다. MAP의 이론적 분석은 가치 간의 균형(trade-off), 제약에 대한 민감도(sensitivity), 다중 가치 정렬과 순차적 정렬(sequential alignment) 간의 근본적인 연결 등을 정량화합니다.

- **Performance Highlights**: MAP는 다중 값을 원칙적으로 정렬할 수 있는 능력을 보여주며, 다양한 작업(task)에서 강력한 경험적 성능을 발휘합니다.



### Tailored-LLaMA: Optimizing Few-Shot Learning in Pruned LLaMA Models with Task-Specific Prompts (https://arxiv.org/abs/2410.19185)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)에서 기존의 파라미터를 잘라내고(프루닝) 특정 작업에 적합하게 조정할 수 있는 새로운 방법을 소개합니다. 특히 Tailored LLaMA라는 접근법을 통해 성능을 유지하면서 모델 크기를 크게 줄이는 방안을 제안합니다.

- **Technical Details**: Tailored LLaMA는 구조적 프루닝(Structural Pruning)을 통해 원래 7B 파라미터 모델에서 5B 및 4B 파라미터로 줄이는 방법을 사용합니다. 이후, LoRA 방법(Low-Rank Adaptation)을 활용하여 특정 작업에 맞춘 프롬프트(Prompt)를 적용하고 모델의 성능을 신속하게 향상시킵니다.

- **Performance Highlights**: Tailored LLaMA를 통해 50%로 압축된 모델이 평균 재분류 작업 정확도를 95.68%로, 50% 압축시 86.54%로 유지할 수 있었으며, 이 과정에서 최적의 프롬프트를 사용하여 더욱 효과적인 성능을 달성하였습니다.



### Can Self Supervision Rejuvenate Similarity-Based Link Prediction? (https://arxiv.org/abs/2410.19183)
- **What's New**: 최근의 링크 예측(link prediction, LP) 방법에서 Self-Supervised Similarity-based LP (3SLP)라는 새로운 접근법이 제안되었습니다. 이 방법은 전통적인 유사성 기반 LP 방식에 자기 지도 학습(self-supervised learning, SSL) 기술을 통합하여 노드 특성의 표현을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 3SLP는 DCNRL(Dual-View Contrastive Node Representation Learning)을 도입하여, 데이터 증강(data augmentation)과 노드 표현 학습(node representation learning)을 통해 노드의 정보를 더욱 풍부하게 생성합니다. 이 방법은 유사성 기반 LP에서의 노드 특성을 개선하여 링크 예측 성능을 향상시키는 데 기여합니다. 주요 기술로는 그래프 확산(graph diffusion)과 크로스 스케일 대비(cross-scale contrast)가 결합되어 있습니다.

- **Performance Highlights**: 기존의 전통적인 유사성 기반 LP 기법과 비교할 때, 3SLP는 최대 21.2%의 AUC(Area Under the Curve) 성능 향상을 보였습니다. 이는 제안된 방법이 전통적인 방법들에 비해 우수한 성능을 나타낸다는 것을 의미합니다.



### PDL: A Declarative Prompt Programming Languag (https://arxiv.org/abs/2410.19135)
- **What's New**: 이 논문은 Prompt Declaration Language (PDL)을 소개하며, 이는 YAML 기반의 간단하고 선언적인 데이터 지향 언어입니다. PDL은 기존의 고급 프레임워크가 갖는 복잡함과 사용자의 제어 부족 문제를 해결하는 것을 목표로 합니다.

- **Technical Details**: PDL은 YAML 형식을 기반으로 하여 데이터 직렬화 포매트를 인간이 읽기 쉬우면서도 구조적인 방법으로 제공합니다. 사용자는 PDL을 통해 다양한 LLM 플랫폼과 상호작용하는 응용 프로그램을 쉽게 작성할 수 있으며, 반복문, 조건문 및 함수와 파일 포함과 같은 제어 구조를 지원합니다.

- **Performance Highlights**: PDL은 LLM을 호출하며 상호작용하는 응용 프로그램 개발을 쉽게 만들어 주며, 챗봇, RAG (Retrieval-Augmented Generation), 에이전트 등 일반적인 사용 사례 구현에 효율적입니다. 또한, PDL은 여러 LLM과 여러 공급자가 제공하는 LLM에 대한 지원을 통해 유연성을 제공합니다.



### RSA-Control: A Pragmatics-Grounded Lightweight Controllable Text Generation Framework (https://arxiv.org/abs/2410.19109)
Comments:
          Accepted to EMNLP 2024 (main conference)

- **What's New**: 본 논문에서는 RSA-Control이라는 훈련 없는 (training-free) 제어 가능한 텍스트 생성 프레임워크를 소개합니다. 이 프레임워크는 발화자(발언하는 사람)와 청자(듣는 사람) 간의 반복적 추론을 통해 정보의 해석을 조정하여 특정 속성(attribute)이 청자에 의해 정확하게 인식될 가능성을 높입니다.

- **Technical Details**: RSA-Control은 주어진 컨텍스트에 따라 제어 강도를 자동으로 조정할 수 있는 자기 조정 합리성 매개변수를 도입합니다. 이 메커니즘은 PLMs(Pre-trained Language Models)의 의사소통 과정을 명확히 하고, 발화자가 원하는 속성을 반영하여 청자의 이해를 보장합니다. RSA-Control은 발화자와 청자 모듈의 상호작용을 모델링하여 설명적이고 실용적인 발화를 생성합니다.

- **Performance Highlights**: RSA-Control은 두 가지 작업 유형과 두 가지 유형의 PLMs를 활용한 실험에서 강력한 속성 제어를 실현하면서 언어 유창성과 내용 일관성을 유지했습니다. 실험 결과, GPT2 모델로 오픈 엔드 생성에서 독성(toxicity) 및 고정관념(bias)을 줄이고, Llama-2-7b-chat 모델로 가독성에 초점을 맞춘 요약 생성을 성공적으로 수행했습니다.



### ReasonAgain: Using Extractable Symbolic Programs to Evaluate Mathematical Reasoning (https://arxiv.org/abs/2410.19056)
- **What's New**: 본 연구에서는 기존의 수학 데이터셋을 사용한 평가 방식을 개선하기 위해 symbolic programs (상징적 프로그램)을 사용하여 대형 언어 모델(LLMs)의 수학적 추론 능력을 자동으로 평가하는 방법을 제안합니다.

- **Technical Details**: 우리는 Python 코드를 사용하여 GSM8K와 MATH 데이터셋의 질문에 대한 프로그램을 추출합니다. 이를 통해 다양한 입력-출력 쌍을 생성하고 LLM의 수학적 추론 과정을 검증합니다. 우리의 접근 방식은 프로그램을 통한 일관된 정답 생성을 검토하여 모델의 추론 품질을 평가합니다.

- **Performance Highlights**: 실험 결과, 기존 LLM들은 제안한 평가 방법에서 성능이 10%에서 20%까지 떨어지는 것을 관찰했습니다. 이는 LLM들이 수학적 추론에서 상당한 취약점을 가지고 있음을 나타냅니다. 전통적인 정적 데이터 평가 방법과는 달리, \



### Infogent: An Agent-Based Framework for Web Information Aggregation (https://arxiv.org/abs/2410.19054)
Comments:
          Preprint

- **What's New**: 웹 내 정보 집합을 위한 새로운 모듈형 프레임워크 Infogent를 소개합니다. 이 프레임워크는 내비게이터(Navigator), 추출기(Extractor), 집계기(Aggregator)라는 세 가지 주요 구성 요소로 구성되어 있습니다.

- **Technical Details**: Infogent는 두 가지 정보 접근 시나리오인 직접 API 기반 접근(Direct API-Driven Access)과 인터랙티브 비주얼 접근(Interactive Visual Access)을 지원합니다. 내비게이터(Navigator)는 웹을 탐색하고 적절한 웹사이트를 찾으며, 추출기(Extractor)는 선정된 웹페이지에서 관련 정보를 찾고, 마지막으로 집계기(Aggregator)는 추출된 정보를 선택적으로 유지하고 가장 최종 집계 결과에 포함할 내용을 결정합니다.

- **Performance Highlights**: 실험 결과, Infogent는 FRAMES에서 기존의 다중 에이전트 검색 프레임워크 대비 7% 우수한 성능을 보였고, AssistantBench에서 기존 정보 탐색 웹 에이전트보다 4.3% 더 개선된 성과를 얻었습니다.



### O1 Replication Journey: A Strategic Progress Report -- Part 1 (https://arxiv.org/abs/2410.18982)
- **What's New**: 이 논문에서는 OpenAI의 O1 모델에 대한 투명하고 실시간으로 진행되는 복제 탐험인 O1 Replication Journey를 소개합니다. 이 연구에서는 현업 AI 연구의 여러 문제를 해결하고, 시도와 실패를 기록하여 오픈 사이언스를 촉진하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 ‘journey learning’ 패러다임을 제안하며, 모델이 단순한 단축키를 배우는 것이 아니라 Trial and Error, Reflection 및 Backtracking을 포괄하는 완전한 탐색 과정을 학습하도록 장려합니다. 이 방법론은 327개의 훈련 샘플만 가지고도 기존의 supervised learning보다 8% 이상 성능 향상을 보여주었습니다.

- **Performance Highlights**: O1 Replication Journey에서는 O1 모델의 특성을 복제하고 학습하는 과정에서 성공적인 사례와 실패 사례를 모두 기록하여 오픈 사이언스를 촉진하며, AI 연구 커뮤니티에 중요한 기여를 할 것으로 기대합니다.



### Counting Ability of Large Language Models and Impact of Tokenization (https://arxiv.org/abs/2410.19730)
- **What's New**: 이 연구는 Large Language Models (LLMs)에서 토큰화(tokenization)가 카운팅(counting) 능력에 미치는 영향을 조명하고 있습니다. 기존 연구에서는 Transformer 구조의 한계를 이야기했으나, 토큰화 과정이 이들 모델의 이론적 계산 능력에 미치는 영향은 상대적으로 적게 다뤄졌던 문제입니다.

- **Technical Details**: 저자들은 토큰화 선택이 LLM의 이론적 카운팅 능력을 크게 저하시킬 수 있음을 입증하기 위해 모델 중립적인 접근법을 사용했습니다. 이를 통해 Chain of Thought (CoT) 방식으로 수행된 카운팅 실험을 진행하였으며, 적절한 토큰화 선택이 모델의 이론적 카운팅 능력을 최대로 드러내는데 필수적이라는 점을 강조하고 있습니다.

- **Performance Highlights**: 토큰화의 선택에 따라 최대 80%까지 정확도가 떨어질 수 있으며, 다양한 모델 간에도 토큰화가 카운팅 작업에 미치는 영향이 다르게 나타났습니다. 연구 결과는 LLM의 카운팅 능력을 향상시키기 위한 새로운 토큰화 방법 설계에 영감을 줄 수 있을 것으로 보입니다.



### Sparse Decomposition of Graph Neural Networks (https://arxiv.org/abs/2410.19723)
- **What's New**: 이번 논문에서는 그래프 신경망(Graph Neural Networks, GNN)의 효율적인 추론(inference) 성능을 개선하기 위한 새로운 방법론인 Sparse Decomposition for Graph Neural Networks (SDGNN)을 제안합니다. SDGNN은 노드의 특성을 동적으로 반영할 수 있는 온라인 예측(online prediction) 설정에서 GNN의 추론 시간 복잡성을 선형으로 줄일 수 있는 접근법을 제공합니다.

- **Technical Details**: SDGNN은 확장된 이웃(neighbourhood) 내에서 선택된 노드 집합의 선형 변환된 특성에 대한 가중합(weighted sum)을 통해 노드 표현을 근사합니다. 이 방법은 평균 노드 차수(average node degree)와 GNN의 레이어(layer) 수에 대해 선형 복잡도를 보장하며, 최적의 파라미터를 계산하기 위한 알고리즘도 포함되어 있어 원래 GNN 모델에 대한 정확한 근사를 보장합니다.

- **Performance Highlights**: SDGNN은 노드 분류(node classification) 및 시공간 예측(spatio-temporal forecasting) 작업에 대한 광범위한 실험을 통해 GNN 추론 속도를 높이기 위한 기존 최첨단 모델들보다 높은 정확성을 달성하며, 비슷한 추론 시간 내에서 성능 향상을 입증했습니다.



### 2D-DPO: Scaling Direct Preference Optimization with 2-Dimensional Supervision (https://arxiv.org/abs/2410.19720)
Comments:
          The first four authors contributed equally, 25 pages

- **What's New**: Direct Preference Optimization (DPO)의 발전을 통해 대규모 언어 모델(LLMs)과 인간의 선호를 보다 정교하게 정렬할 수 있는 방법이 제안되었습니다. 새로운 2차원 감시 데이터셋인 HelpSteer-2D를 도입하여 응답을 세그먼트와 측면으로 나누고 이 두 차원을 기반으로 한 DPO(2D-DPO)를 소개합니다.

- **Technical Details**: 2D-DPO는 응답을 문장으로 나누고 각 세그먼트에 점수를 할당하여 다차원적인 인간의 선호를 포착합니다. 이를 위해 2차원 점수 행렬을 사용하여 각 세그먼트의 여러 측면을 평가합니다. 2D-DPO는 총 목표를 다중 세그먼트 및 다중 측면 목표로 분해하여 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과, 2D-DPO는 스칼라 또는 1차원 선호를 최적화하는 기존 방법보다 더 나은 성능을 보였습니다. 이는 모델 훈련에서 다차원적인 피드백의 중요성을 잘 반영하고 있습니다.



### Arabic Music Classification and Generation using Deep Learning (https://arxiv.org/abs/2410.19719)
- **What's New**: 이 논문은 고전 및 현대 이집트 음악을 작곡가에 따라 분류하고 유사한 새로운 음악을 생성하는 기계 학습 접근법을 제안합니다. 이 시스템은 음악 분류를 위해 합성곱 신경망 (CNN)을 사용하고, 새로운 음악 생성을 위해 CNN 오토인코더를 활용합니다.

- **Technical Details**: 제안된 시스템은 다양한 작곡가가 작곡한 고전 및 현대 이집트 음악으로 구성된 데이터셋을 사용합니다. 각 샘플은 멜 스펙트로그램으로 변환되어 CNN 모델의 입력 특징으로 사용됩니다. 모델은 작곡가 레이블을 출력 클래스로 하여 학습됩니다. 이 모델은 81.4%의 정확도로 음악을 분류하며, CNN 오토인코더를 통해 새로운 음악을 생성합니다.

- **Performance Highlights**: 제안된 시스템은 음악 추천 시스템, 음악 제작 및 음악 교육 등 다양한 음악 응용 분야에 적용될 수 있는 효과적인 방법을 제공합니다.



### Evolving Neural Networks Reveal Emergent Collective Behavior from Minimal Agent Interactions (https://arxiv.org/abs/2410.19718)
Comments:
          25 pages, 9 figures

- **What's New**: 이 연구에서는 다중 에이전트 시스템에서 emergent behaviors의 메커니즘을 이해하는 것이 swarm robotics 및 인공지능 분야 발전에 얼마나 중요한지를 살펴보았습니다. 신경망의 비선형성 정도가 emergent behaviors의 복잡성과 상관 관계가 있음을 보여주며, 환경 파라미터가 비선형 네트워크의 진화에 미치는 영향을 분석하였습니다.

- **Technical Details**: 본 연구에서는 최소한의 지능을 갖춘 에이전트 모델을 제안하고, 이들이 진화 알고리즘을 통해 근접성을 기반으로 행동 방향을 선택하도록 학습합니다. 설정된 환경 파라미터는 적당한 노이즈, 넓은 시야, 낮은 에이전트 밀도를 포함하여 더욱 진화적이고 복잡한 복합 행동을 유도합니다.

- **Performance Highlights**: 이 연구의 결과는 최소한의 조건에서 자율적인 집합 지능이 어떻게 발전할 수 있는지를 보여주며, 자율 swarm을 최적화하는 데 새로운 경로를 제시합니다. 이를 통해 다중 에이전트 시스템에서의 자연적인 자가 조직화의 이해를 깊게 해줄 뿐만 아니라, 실제 응용에 있어 중요한 통찰을 제공합니다.



### Adversarial Environment Design via Regret-Guided Diffusion Models (https://arxiv.org/abs/2410.19715)
Comments:
          38th Conference on Neural Information Processing Systems

- **What's New**: 이번 연구에서는 Deep Reinforcement Learning (RL)에서 환경 변화에 강한 에이전트를 훈련하기 위한 새로운 알고리즘, Adversarial Environment Design via Regret-guided Diffusion Models (ADD)를 제안합니다. 이 방법은 에이전트의 후회(regret)를 이용하여 도전적인 환경을 생성하여 정책 학습을 개선하도록 설계되었습니다.

- **Technical Details**: 이 알고리즘은 확산 모델(diffusion models)의 표현력을 활용하여 환경 생성을 직접적으로 수행합니다. ADD는 환경 생성자(generator)가 에이전트의 후회를 정교화 하도록 유도하며, 이를 통해 두 가지 주요 접근 방식인 학습 기반 방법과 재생 기반 방법의 장점을 결합하였습니다. 후회 추정을 위해 새로운 방식을 도입하여 후회를 미분 가능하게 만들었습니다.

- **Performance Highlights**: 실험 결과 ADD는 기존의 UED 기반선(linear regression baseline)보다 높은 제로샷 제너럴리제이션(zero-shot generalization) 성능을 달성하였으며, 다양한 난이도의 환경을 포함하는 교육 커리큘럼을 성공적으로 생성함으로써 에이전트가 일반화 능력이 뛰어난 정책을 학습하도록 지원했습니다.



### TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning (https://arxiv.org/abs/2410.19702)
- **What's New**: 이번 논문에서는 짧은 비디오 이해에 뛰어난 성능을 보인 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 장기 비디오 이해에 적용하기 위해 TimeSuite라는 새로운 디자인을 제안합니다. 이 프레임워크는 긴 비디오 시퀀스를 처리하는 간단하면서도 효율적인 방법과 고품질 비디오 데이터셋, 그리고 기존 질문 답변 형식(QA format)에서 그라운딩(supervision) 감독을 명시적으로 포함하는 훈련 과제를 포함합니다.

- **Technical Details**: TimeSuite는 긴 비디오 이해를 위해 설계되었으며, 다음과 같은 주요 요소를 포함합니다: 1) Token Shuffle 기법을 통한 비주얼 토큰 압축, 2) Temporal Adaptive Position Encoding (TAPE)을 사용한 시각적 표현의 시간 인식 향상, 3) Temporal Grounded Caption 훈련 과제를 통한 세부적인 비디오 설명 생성. 이 외에도 TimePro라는 복합적인 그라운딩 중심 인스트럭션 튜닝 데이터셋이 포함되어 있습니다.

- **Performance Highlights**: TimeSuite는 긴 비디오 이해에서 Egoschema와 VideoMME 기준에서 각각 5.6%와 6.8% 성능 향상을 달성하였으며, VideoChat-T는 다른 최신 MLLMs에 비해 뛰어난 제로샷(zero-shot) 시간 그라운딩 능력을 보여 주었습니다. 실험 결과, VideoChat-T는 전통적인 전문가 모델과 동등한 성능을 기록했습니다.



### Enhancing Resilience and Scalability in Travel Booking Systems: A Microservices Approach to Fault Tolerance, Load Balancing, and Service Discovery (https://arxiv.org/abs/2410.19701)
Comments:
          18 pages, 3 figures

- **What's New**: 이 논문은 항공사 예약 시스템 개발에 마이크로서비스 아키텍처의 통합을 조사합니다. 전통적인 예약 시스템은 경직되어 있고 중앙 집중화되어 있어 병목 현상(bottlenecks) 및 단일 실패 지점(single point of failure)에 취약합니다.

- **Technical Details**: 본 연구는 Circuit Breaker Pattern을 기반으로 하여 외부 리소스(예: 비행 API 및 결제 시스템)를 소비할 때 결함 허용(fault tolerance)을 유지합니다. 이를 통해 시스템에서 실패 전파(failure propagation)를 60% 줄였으며, 트래픽 리라우팅(traffic rerouting)을 통해 99.95% 이상의 가동 시간(uptime)을 보장했습니다. 또한 로드 밸런싱(load balancing)에는 라운드 로빈(Round-Robin) 방법이 사용되어 사용자 요청을 서비스 인스턴스에 균등하게 분배함으로써 성능을 35% 향상시켰습니다.

- **Performance Highlights**: 마이크로서비스를 사용하는 경우 시스템 확장성(scalability)이 40% 증가하고, 다운타임(downtime)은 50% 감소하며, 30% 더 많은 동시 사용자(concurrent users)를 지원할 수 있었습니다. 이러한 결과는 변화에 민감하고 외부 시스템 장애에서 복구할 수 있는 강력하고 유연한 항공권 예약 시스템 개발에 있어 마이크로서비스의 능력을 확인시켜 줍니다.



### IPPON: Common Sense Guided Informative Path Planning for Object Goal Navigation (https://arxiv.org/abs/2410.19697)
- **What's New**: 본 논문은 3D 객체 확률 맵핑 및 유용한 경로 계획을 위한 IPPON이라는 새로운 접근 방식을 도입합니다. 이 방법은 의미적 세분화 및 베이지안 필터를 활용하여 탐색 중인 객체의 확률을 계산하고, 대규모 언어 모델(Large Language Model)에서의 상식 근거를 바탕으로 탐색을 안내합니다.

- **Technical Details**: IPPON은 온라인 정보 경로 계획(Informative Path Planning, IPP) 프레임워크에 기반하여 객체 목표 탐색 문제를 해결합니다. 이 방법은 각 시점에서 OOI(Object of Interest)를 포함할 가능성이 높은 voxel들의 집계 확률을 이득으로 정의하여, 3D 객체 확률 맵핑 알고리즘을 통해 일반 객체와 OOI의 확률을 추정합니다.

- **Performance Highlights**: 본 연구의 제안된 방법은 Habitat ObjectNav Challenge 2023에서 성공 비율 및 경로 길이에 가중치(Success weighted by Path Length, SPL) 측정에서 20% 이상 다른 방법들보다 우수한 성능을 기록하였으며, 실제 로봇에서도 효과를 검증하였습니다.



### Less is More: Extreme Gradient Boost Rank-1 Adaption for Efficient Finetuning of LLMs (https://arxiv.org/abs/2410.19694)
Comments:
          19 pages

- **What's New**: 본 논문에서는 eXtreme Gradient Boosting LoRA (XGBLoRA)라는 혁신적인 프레임워크를 제안합니다. XGBLoRA는 앙상블 학습의 힘을 활용하여 Low-Rank Adaptation (LoRA)의 이론적 최적성과 실제 성능 간의 격차를 해소하는 방법을 제시합니다.

- **Technical Details**: XGBLoRA는 Gradient Boosting의 원리를 바탕으로 연속적인 LoRA 적응을 학습하고 이를 결합하여 모델 예측을 개선합니다. 이 프레임워크는 적응 행렬의 랭크가 낮아도 성능 저하 없이 높은 예측 품질을 유지합니다. XGBLoRA의 이론적 분석을 통해 수렴 보장과 예측 경계가 확립되었으며, 낮은 랭크를 유지하면서도 양질의 성능을 얻을 수 있는 방법이 설명되었습니다.

- **Performance Highlights**: 실험 결과, XGBLoRA는 기존의 LoRA 및 전체 파인튜닝에 비해 더 나은 성능을 지속적으로 보여주었습니다. XGBLoRA는 더 적은 수의 훈련 가능한 매개변수를 사용하면서도 상위 성능을 달성, 예를 들어 LLaMA3-8B 모델에서 NVIDIA RTX 4090에서 실행이 가능했으며, LoRA는 A100 GPU를 필요로 했습니다.



### MILES: Making Imitation Learning Easy with Self-Supervision (https://arxiv.org/abs/2410.19693)
Comments:
          Published at the Conference on Robot Learning (CoRL) 2024

- **What's New**: MILES라는 완전 자율적인 자기 지도 학습(data collection paradigm)을 통한 데이터 수집 접근 방식을 제안합니다. 단 하나의 Demonstration과 환경 리셋 없이도 효율적인 정책 학습이 가능하다는 점이 혁신적입니다.

- **Technical Details**: MILES는 자기 주도적으로 데이터 수집을 수행하며, Behavioral Cloning (BC)에 기반하여 정책을 학습합니다. 기존의 모방 학습(imitation learning) 방법들은 수백 개의 Demonstration이 필요했지만, MILES는 하나의 Demonstration만으로도 로봇이 작업을 수행할 수 있도록 합니다.

- **Performance Highlights**: MILES는 다양한 실시간 작업에서 뛰어난 성능을 보이며, 기존의 강화 학습(reinforcement learning) 및 리플레이 기반 모방 학습 방법들보다 우수한 결과를 낸 것으로 확인되었습니다. 실험적으로는 하나의 Demonstration과 약 30분의 자기 지도 학습으로 여러 작업을 수행할 수 있었습니다.



### AGENT-CQ: Automatic Generation and Evaluation of Clarifying Questions for Conversational Search with LLMs (https://arxiv.org/abs/2410.19692)
Comments:
          23 pages

- **What's New**: AGENT-CQ라는 새로운 프레임워크가 발표되었습니다. 이 시스템은 대화형 검색 시스템에서 다양한 기다리는 질문(clarifying questions)을 자동 생성하고 평가하는 기능을 제공합니다. 기존 방법들의 한계를 극복하기 위해 LLM 기반(end-to-end LLM-based) 접근 방식을 채택했습니다.

- **Technical Details**: AGENT-CQ는 두 단계로 구성되어 있습니다. 첫 번째 단계는 생성 단계(generation stage)로, LLM 프롬프트(prompting) 전략을 사용하여 기다리는 질문을 생성합니다. 두 번째 단계는 평가 단계(evaluation stage)로, CrowdLLM을 사용하여 다수의 LLM 인스턴스를 통해 생성된 질문과 답변의 품질을 종합적인 품질 메트릭(classic quality metrics)에 따라 평가합니다.

- **Performance Highlights**: ClariQ 데이터셋을 기반으로 한 실험에서 CrowdLLM의 평가가 질문과 답변의 품질을 높이는 데 매우 효과적임을 입증하였습니다. AGENT-CQ의 생성 단계는 다양한 질문과 답변 품질 측면에서 기준선(baselines)을 지속적으로 능가했습니다. 검색 기반 평가에서는 LLM이 생성한 질문이 인간이 생성한 질문에 비해 BM25 및 크로스 인코더(cross-encoder) 모델 모두에서 검색 효과를 크게 향상시키는 것으로 나타났습니다.



### Deep learning-based identification of patients at increased risk of cancer using routine laboratory markers (https://arxiv.org/abs/2410.19646)
- **What's New**: 본 논문은 혈액 기반 위험 평가를 통해 암의 조기 screening 방법을 제안합니다. 이는 혈액 검사 결과를 이용해 암 위험이 높은 사람을 식별하고, 진단 검사를 받도록 유도하는 접근 방식을 다룹니다.

- **Technical Details**: 'Deep Profiler'라는 심층 학습 모델을 활용하여, 연령, 성별 및 일반적인 혈액 biomarkers를 입력으로 받아 12개월 내 암 발생 가능성을 계산합니다. 이 모델은 쌍방향 변분 오토인코더(variational autoencoder, VAE)를 통해 결측 데이터를 보완한 뒤, 암별 위험 예측 모델을 학습합니다.

- **Performance Highlights**: 제안된 방법은 대장암, 간암 및 폐암에 대해 ROC 곡선 아래 영역(areas under the ROC curve) 값이 각각 0.76, 0.85, 0.78에 도달하며, 기존 방법들에 비해 더 나은 성능을 보입니다.



### Impact of Leakage on Data Harmonization in Machine Learning Pipelines in Class Imbalance Across Sites (https://arxiv.org/abs/2410.19643)
- **What's New**: 이 연구에서는 데이터 조화를 위해 기존의 ComBat 기반 방법들이 사이트 간 클래스 불균형 상황에서 데이터 누수(data leakage) 문제에 직면한다는 점을 발견하였으며, 이를 해결하기 위해 'PrettYharmonize'라는 새로운 방법을 제안하였다.

- **Technical Details**: PrettYharmonize는 데이터 조화 과정에서 실제 목표 레이블을 사용하는 대신 가상의 레이블을 활용하는 방식을 채택하여 데이터 누수를 방지한다. 이 방법은 neuroHarmonize 모델을 기반으로 하며, 예측 모델과 스택 모델의 조합을 통해 데이터를 조화화한다.

- **Performance Highlights**: 실제 MRI 및 임상 데이터를 사용하여 데이터 누수가 발생할 위험이 있는 기존 방법들과 PrettYharmonize를 비교한 결과, PrettYharmonize가 동일한 성능을 달성하면서도 데이터 누수를 피함으로써 더욱 신뢰할 수 있는 결과를 만들어 내었다.



### OpenWebVoyager: Building Multimodal Web Agents via Iterative Real-World Exploration, Feedback and Optimization (https://arxiv.org/abs/2410.19609)
- **What's New**: 새롭게 소개된 OpenWebVoyager는 멀티모달 웹 에이전트를 개발하기 위한 오픈 소스 프레임워크입니다. 이 프레임워크는 자율적으로 실제 환경을 탐색하고 스스로 개선할 수 있는 웹 에이전트를 만들 수 있도록 돕습니다.

- **Technical Details**: OpenWebVoyager는 맨 처음에 imitation learning (모방 학습)으로 기본 능력을 익히고, 이후 웹을 탐색하며 피드백을 수집합니다. 이 과정에서 잘 수행된 궤적을 다른 일반 목적 모델을 통해 학습하여 정책을 개선합니다. 이러한 exploration-feedback-optimization (탐색-피드백-최적화) 사이클은 여러 번 반복될 수 있습니다.

- **Performance Highlights**: 실험 결과, OpenWebVoyager는 반복할수록 성능이 향상되며, WebVoyager 테스트 세트에서 작업 성공률이 19.9%에서 25.8%로, Mind2Web 크로스 과제 세트에서는 6.3%에서 19.6%로 증가하였습니다. 또한, Mind2Web 크로스 웹 세트에서도 성과가 개선되었습니다.



### CoqPilot, a plugin for LLM-based generation of proofs (https://arxiv.org/abs/2410.19605)
Comments:
          Published in the proceedings of the ASE'24 Tool Demonstrations Track

- **What's New**: CoqPilot는 Coq 증명의 작성 자동화를 위한 VS Code 확장 프로그램입니다. 이 플러그인은 Coq 파일에서 admit 전술로 표시된 증명의 부분을 수집하여 LLM과 비기계 학습 방법을 결합하여 증명 후보를 생성합니다.

- **Technical Details**: CoqPilot은 VS Code용 플러그인으로, Coq의 admit 전술로 표시된 증명의 빈 부분을 탐색하고, LLM 및 기타 방법을 통한 증명 후보를 생성합니다. 개발 중에 Coq 언어 서버와의 통합, 여러 모델의 성능 비교를 위한 벤치마킹 프레임워크를 구현하였습니다.

- **Performance Highlights**: CoqPilot은 다양한 LLM 모델을 사용하여 Coq 증명을 자동으로 생성하고 검증하여 사용자가 더 나은 증명 작성을 할 수 있도록 지원합니다. 벤치마킹 결과, CoqPilot이 기존 접근 방식보다 더 효과적인 증명 생성을 가능하게 함을 확인하였습니다.



### Take Caution in Using LLMs as Human Surrogates: Scylla Ex Machina (https://arxiv.org/abs/2410.19599)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 인간의 행동을 모방할 수 있다고 주장하는 많은 기존 논문과 달리, LLMs의 사고 깊이가 인간 행동을 재현하는 데 실패한다는 것을 보여줍니다. 특히 실험을 통해 LLM의 반응이 인간 참여자의 유사성에서 크게 벗어남을 입증하였으며, 이는 LLM을 사회 과학 연구의 인간 대체자로 사용하는 것에 대해 경각심을 불러일으킵니다.

- **Technical Details**: 11-20 머니 요청 게임을 사용하여 LLM의 사고 깊이를 평가했으며, 8개의 인기 있는 LLM(GPT-4, GPT-3.5 등)에 대해 각 1,000개의 세션을 기록하고 인간 참여자의 응답 분포와 비교했습니다. 다양한 접근 방식 중에서도 명확하게 인간의 행동과 유사한 반응을 생성하는 것은 불가능했습니다. 특정한 프레임에 따라 LLM의 반응이 더욱 불안정해지고, 그 결과는 다양한 언어와 지침에 따라 달라지는 경향을 보였습니다.

- **Performance Highlights**: 대부분의 최첨단 접근법이 인간 행동 분포를 복제하는 데 실패한 것으로 나타났으며, 단 한 가지 사례(GPT-4의 파인튜닝)를 제외하고 모든 접근법은 LLM이 인간 참여자와의 통계적 유사성을 발휘하는 데 성공하지 못했습니다. 마지막으로, LLM의 반응은 프롬프트의 변화에 따라 큰 차이를 보이며, 이는 LLM의 출력의 불안정성을 나타냅니다.



### Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning (https://arxiv.org/abs/2410.19560)
- **What's New**: C-JEPA(Contrastive-JEPA)라는 새로운 프레임워크를 소개하며, 이는 JEPA(Join-Embedding Predictive Architecture)와 VICReg(Variance-Invariance-Covariance Regularization)를 통합하여 시각적 표현 학습의 한계를 극복하고자 합니다.

- **Technical Details**: C-JEPA는 I-JEPA의 Exponential Moving Average(EMA) 및 예측 메커니즘의 한계를 해결하고, 다양한 뷰 간의 불변성을 유지하여 전체 붕괴를 방지하는 방법으로 설계되었습니다. 이 연구는 VICReg 전략을 결합하여 모델의 안정성과 표현 품질을 크게 향상시킴을 보여줍니다.

- **Performance Highlights**: C-JEPA는 ImageNet-1K 데이터셋에서 사전 훈련 시 선형 프로빙과 파인 튜닝 성능 지표 모두에서 빠르고 향상된 수렴을 보여주며, 비지도 visual representation learning 분야의 새로운 기준을 설정할 가능성이 있습니다.



### On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes (https://arxiv.org/abs/2410.19553)
Comments:
          This paper was accepted to NeurIPS 2023 Dataset And Benchmark Track. It also showcases: Hinton's Islands of Agreement on realistic datasets which were previously hypothesized in his GLOM paper

- **What's New**: 이 논문은 비디오 액션 탐지에서 가리개(occlusion)의 영향을 탐구하고 있으며, 정적/동적 가리개를 포함한 다섯 개의 새로운 벤치마크 데이터셋을 소개합니다. 이 연구는 가리개가 비디오 인식 모델의 성능에 미치는 영향을 정량화하고, Transformer 모델이 CNN보다 더 뛰어난 성능을 보일 수 있음을 확인합니다.

- **Technical Details**: 연구에서는 O-UCF, O-JHMDB, OVIS-UCF, OVIS-JHMDB, Real-OUCF의 다섯 개 데이터셋을 사용하여 다양한 유형의 가리개를 분석하였고, 가리개의 세기가 증가할수록 모델 성능이 감소함을 발견했습니다. 특히 Transformer 모델은 가리개를 데이터 증강 형태로 사용한 CNN 모델과 비교했을 때 더 우수한 성능을 보여주었습니다. 또한, 실제 사례를 바탕으로 사전 학습의 중요성이 강조되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 모델이 O-UCF, O-JHMDB, Real-OUCF에서 각각 32.3%, 32.7%, 2.6%의 성능 향상을 기록하였고, 비디오 액션 탐지 분야에서 가리개에 대한 강건성을 높이는 몇 가지 간단하고 효과적인 트레이닝 레시피를 제안했습니다.



### DeMuVGN: Effective Software Defect Prediction Model by Learning Multi-view Software Dependency via Graph Neural Networks (https://arxiv.org/abs/2410.19550)
- **What's New**: 이번 연구는 소프트웨어 결함 예측을 위한 새로운 모델 DeMuVGN을 제안합니다. 이 모델은 다중 뷰 소프트웨어 의존성 그래프(Multi-view Software Dependency Graph, MSDG)를 활용하여 소프트웨어 결함 정보를 보다 포괄적으로 제공합니다.

- **Technical Details**: DeMuVGN은 코드 의존성 그래프(Code Dependency Graph, CDG)와 개발자 의존성 그래프(Developer Dependency Graph, DDG)를 통합한 MSDG를 기반으로 삼습니다. 또한, BiGGNN(Bidirectional Gated Graph Neural Network)을 활용하고, Synthetic Minority Oversampling Technique (SMOTE)를 도입하여 클래스 불균형 문제를 해결합니다.

- **Performance Highlights**: DeMuVGN은 20개의 버전에 걸쳐 8개의 오픈소스 프로젝트에서 평가한 결과, 단일 뷰 모델에 비해 F1 점수가 11.1%에서 12.1% 향상되었고, 내부 프로젝트에서 17.4%에서 45.8%, 외부 프로젝트에서 17.9%에서 41.0%까지 성능이 개선되었습니다. 또한, 소프트웨어 진화에 있어 후반 버전에서의 성능 향상이 두드러졌습니다.



### PMM-Net: Single-stage Multi-agent Trajectory Prediction with Patching-based Embedding and Explicit Modal Modulation (https://arxiv.org/abs/2410.19544)
- **What's New**: 이 논문은 다중 에이전트 궤적 예측(MATP)을 위한 새로운 접근 방식을 소개합니다. 특히, 패칭 기반의 시간적 특성 추출 모듈과 그래프 기반의 사회적 특성 추출 모듈을 제안하여 효과적인 특성 추출과 교차 시나리오 일반화를 가능하게 합니다.

- **Technical Details**: 논문에서는 새롭게 설계된 단일 단계 프레임워크인 PMM-Net을 제안하며, 이는 시간적 및 사회적 특성을 분리하여 처리함으로써 전체적인 계산 복잡성을 줄이고, 소셜 데이터를 활용한 교차 집중 기반 명시적 모달리티 조정을 통해 다중 모달 예측을 달성합니다.

- **Performance Highlights**: 공공 벤치마크 데이터셋인 Stanford Drone 및 ETH/UCY에서 실험한 결과, 제안된 모델이 기존 최첨단 방법들보다 우수한 성능을 보였으며, 다중 모달성을 정확하게 예측할 수 있음을 입증했습니다.



### Brain-like Functional Organization within Large Language Models (https://arxiv.org/abs/2410.19542)
- **What's New**: 본 연구는 인공 신경망(ANNs)과 인간의 뇌 기능적 뇌 네트워크(FBN) 간의 직접적인 연결을 통해, 대형 언어 모델(LLMs) 내에서의 뇌와 유사한 기능적 구조를 탐구합니다. 특히, BERT와 Llama 1-3 모델의 인공 신경세포(ANs)가 인간 뇌의 FBN과 유사한 조직 패턴을 나타내는 것을 밝혔습니다.

- **Technical Details**: 인공 신경세포(ANs)의 시간적 반응 패턴을 추출한 후, 이를 고정 회귀자(fixed regressor)로 활용하여, voxel-wise encoding 모델을 구축했습니다. 이 모델은 기능적 자기 공명 영상(fMRI)으로 기록된 뇌 활동을 예측하는 데 사용됩니다. 연구 결과, LLMs의 AN 하위 그룹은 잘 확립된 FBN과 밀접하게 연관되어 있음이 드러났습니다.

- **Performance Highlights**: LLM의 뇌와 유사한 기능적 구조는 성능이 향상됨에 따라 더욱 두드러지며, 이는 계산 행동의 다양성과 기능적 특화의 일관성 간의 균형을 개선하는 결과를 가져옵니다. 이 연구는 LLMs 내 뇌유사 기능 조직 탐구의 첫 사례로, 인공 일반 지능(AGI)의 발전에 기여할 수 있는 중요한 통찰을 제공합니다.



### CloserMusicDB: A Modern Multipurpose Dataset of High Quality Music (https://arxiv.org/abs/2410.19540)
- **What's New**: 이번 논문에서는 CloserMusicDB라는 고품질 음악 데이터셋을 소개합니다. 이 데이터셋은 전문가들이 주석을 달아 완전한 길이의 스튜디오 트랙을 포함하고 있으며, 훅 탐지(hook detection), 맥락 태깅(contextual tagging), 아티스트 식별(artist identification)과 같은 여러 작업을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: CloserMusicDB에는 106개의 고품질 풀 길이 트랙이 포함되어 있으며, 모든 트랙은 전문가에 의해 녹음, 프로듀싱, 믹싱 및 마스터링되었습니다. 파일 형식은 압축되지 않은 스테레오 WAV이며, 샘플링 주파수는 44,100Hz, 비트 깊이는 16비트입니다. 데이터셋은 후크 시작 및 종료 타임스탬프, 비트당 분(BPM)과 같은 메타데이터 필드를 포함하고 있습니다.

- **Performance Highlights**: 모델 성능에서, 훅 탐지 작업의 정확도는 41.5%였고, 맥락 태깅 작업에서 Jaccard 점수는 0.2998±0.0767로 나타났습니다. 아티스트 식별 작업에서는 60.22%의 정확도를 얻어, 다수의 아티스트를 식별할 수 있는 가능성을 보였습니다.



### DMT-HI: MOE-based Hyperbolic Interpretable Deep Manifold Transformation for Unspervised Dimensionality Reduction (https://arxiv.org/abs/2410.19504)
Comments:
          14 pages, 8 figures

- **What's New**: 이번 연구에서는 고차원 데이터의 차원 축소(Dimensionality Reduction, DR) 과정에서 성능과 해석 가능성 간의 균형을 맞추기 위해 MOE 기반의 하이퍼볼릭 해석 가능 심층 매니폴드 변환(DMT-HI) 방안을 제시합니다.

- **Technical Details**: DMT-HI는 하이퍼볼릭 임베딩(hyperbolic embeddings)과 전문가 혼합 모델(Mixture of Experts, MOE)을 결합하여 데이터의 계층적 구조를 효과적으로 캡처합니다. MOE 전략은 입력 특성에 따라 다양한 작업을 동적으로 할당하여 성능과 효율성을 향상시키고 모델 해석 가능성을 높입니다.

- **Performance Highlights**: DMT-HI는 드높은 차원 축소 정확도와 모델 해석 가능성을 달성하며, 복잡한 데이터 분석을 위한 견고한 솔루션으로 자리매김하였습니다.



### Peter Parker or Spiderman? Disambiguating Multiple Class Labels (https://arxiv.org/abs/2410.19479)
Comments:
          Accepted to Neural Information Processing Systems (NeurIPS 2024). ATTRIB Workshop

- **What's New**: 이 논문에서는 다중 예측을 하는 심층 신경망(Deep Neural Networks)의 예측 결과를 해석하는 새로운 방법론을 제시합니다. 특히 top-k 예측에서 예측의 의미를 명확히 하기 위한 프레임워크를 개발하여, 두 개의 예측이 서로 다른 엔티티를 기반으로 하는지, 또는 동일한 엔티티에 대한 두 개의 예측인지 확인할 수 있습니다.

- **Technical Details**: 제안된 방법론은 먼저 입력 이미지를 세그먼트화(Segmentation)한 후, 각 레이블에 대해 세그먼트별 입력 기여도(Input Attribution) 점수를 할당합니다. 이 점수는 두 개의 레이블 예측이 동일한 엔티티 집합을 가리키는지 아니면 다른 집합을 가리키는지를 결정하는 데 사용됩니다. 또한, 이 방법론은 재실행 없이 주어진 주장(클래스 레이블 쌍이 서로 다른 엔티티 타입인지 단일 엔티티 타입인지)의 유효성을 증명할 수 있는 반증(Counterfactual Proof)을 제공합니다.

- **Performance Highlights**: ImageNet 검증 세트의 여러 샘플을 통해 이 방법론이 여러 모델에서 잘 작동함을 보여주었습니다. 이 연구는 현재의 해석 가능성 연구에서 주목받지 못했던 예측의 두 가지 해석을 구분하는 데 기여합니다.



### Improving Inverse Folding for Peptide Design with Diversity-regularized Direct Preference Optimization (https://arxiv.org/abs/2410.19471)
Comments:
          Preprint. 10 pages plus appendices

- **What's New**: 이번 연구에서는 ProteinMPNN 모델을 다채롭고 구조적으로 일관된 펩타이드 서열을 생성하도록 직접 최적화(Direct Preference Optimization, DPO) 방식으로 미세 조정하였습니다.

- **Technical Details**: 저자들은 온라인 다양성 정규화(Online Diversity Regularization)와 도메인 특정 선행 정보(Domain-specific Priors)를 포함하여 DPO에 두 가지 강화 방법을 도출했습니다. 이 연구는 OpenFold에서 생성된 구조를 조건으로 하는 Fine-tuned 모델이 최소 8% 향상된 상태에서 동작함을 보여주었습니다.

- **Performance Highlights**: DPO의 정규화 방법은 기존 표준 DPO에 비해 최대 20% 더 높은 서열 다양성을 달성하면서도 구조적 유사성 점수에서 손실이 없습니다.



### Unified Causality Analysis Based on the Degrees of Freedom (https://arxiv.org/abs/2410.19469)
Comments:
          32 pages, 7 figures

- **What's New**: 이 논문은 동적 시스템 간의 근본적인 인과관계를 식별할 수 있는 통합 방법을 제시합니다. 이 방법은 관찰된 동적에 대한 미지의 숨겨진 요인의 존재와 영향도 파악하는 데 초점을 맞춥니다.

- **Technical Details**: 제안된 방법론은 시간에 의존하는 변수 간의 상호작용을 분석하며, 결정론적(dynamic)과 확률론적(stochastic) 시스템 모두에 적용 가능합니다. 이 연구는 degrees of freedom causality method(df-causality 또는 df-method)를 통해 인과관계 분석을 수행합니다.

- **Performance Highlights**: 이 통합 프레임워크는 이론적 모델과 시뮬레이션을 통해 검증되었으며, 강건성과 광범위한 응용 가능성을 입증합니다.



### LOCAL: Learning with Orientation Matrix to Infer Causal Structure from Time Series Data (https://arxiv.org/abs/2410.19464)
Comments:
          10 pages, 7 figures

- **What's New**: 본 논문에서는 시간 시계열 관측 데이터에서 동적 원인 구조를 복원하기 위한 새로운 방법, LOCAL(제약 없는 로컬 동적 원인 구조 학습 방법)을 제안합니다. 이는 기존의 방법론들이 갖고 있던 비효율성과 고차원 데이터 처리의 문제를 해결하기 위한 것입니다.

- **Technical Details**: LOCAL은 준 최댓값 우도 기반(score function) 프레임워크로, 동적 DAG(Directed Acyclic Graph)와의 근본적인 동일성을 보장합니다. 이 논문은 두 가지 적응형 모듈인 Asymptotic Causal Mask Learning (ACML)과 Dynamic Graph Parameter Learning (DGPL)을 통해 DAGs의 비대칭성을 향상시키고 고차원 데이터의 동적 원인 구조를 효과적으로 포착합니다.

- **Performance Highlights**: LOCAL은 합성 데이터와 실제 데이터셋에서 기존의 최첨단 방법들에 비해 유의미한 성능 향상을 보였으며, 안정적이고 효율적인 동적 원인 발견 방법으로서의 가능성을 보여주었습니다.



### Accelerating AI Performance using Anderson Extrapolation on GPUs (https://arxiv.org/abs/2410.19460)
Comments:
          6 pages, 6 figures, 1 table, Accepted by NeurIPS 2024 Workshop MLNCP this https URL

- **What's New**: 이 논문에서는 AI의 성능을 가속화하기 위해 Anderson extrapolation 기법을 활용하는 새로운 접근 방식을 소개합니다. 이 방법은 과거 반복의 윈도우를 기반으로 한 벡터 간 매핑 기법으로, 혼합 페널티가 발생하는 교차점에서 반복 횟수를 줄이고, 계산 집약적인 반복을 통한 수렴 속도를 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: Anderson extrapolation은 비선형 고정점 이터레이션을 가속화하기 위한 윈도우링 기법으로, GPU에서 효율적으로 구현됩니다. 이 기법은 이전 이터레이션 정보를 재사용함으로써 불필요한 그래디언트 계산을 피하고, 메모리 사용량을 줄이며, 고급 컴퓨팅 아키텍처에서 기대하는 이점을 제공합니다. 연구에서 고정점 이터레이션 문제를 해결하기 위해 두 가지 접근 방식이 사용되며, CIFAR10 데이터셋을 통해 벤치마킹하고 있습니다.

- **Performance Highlights**: AI 성능 가속화를 통해 GPU에서의 훈련 및 추론 속도가 향상되었습니다. 테스트 결과, Anderson extrapolation이 기존의 전방 이터레이션 방식에 비해 뛰어난 수렴 속도를 보였으며, 메모리 사용량을 최적화하여 AI의 에너지 소비를 줄일 수 있는 잠재력을 가지고 있습니다.



### NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction (https://arxiv.org/abs/2410.19452)
Comments:
          NeurIPS 2024 Oral

- **What's New**: NeuroClips는 fMRI를 사용하여 고화질의 매끄러운 비디오를 재구성하는 혁신적인 프레임워크입니다. 이 방법은 두 가지 주요 구성 요소인 Perception Reconstructor와 Semantics Reconstructor를 통해 구현됩니다.

- **Technical Details**: NeuroClips는 비디오 키프레임을 재구성하기 위해 semantics reconstructor를 활용하고, 영상의 매끄러움을 보장하기 위해 퍼셉션 세부 사항을 캡처하는 perception reconstructor를 채택합니다. 이 논문은 pre-trained T2V diffusion model을 통해 fMRI에서 비디오를 재구성하는 것을 정교하게 수행합니다.

- **Performance Highlights**: NeuroClips는 공개된 fMRI-비디오 데이터셋에서 최대 6초 길이의 비디오를 8FPS로 매끄럽게 재구성하는 성과를 보여주며, SSIM에서 128% 향상, 시공간 메트릭에서도 81% 개선된 결과를 기록했습니다.



### Intelligent Understanding of Large Language Models in Traditional Chinese Medicine Based on Prompt Engineering Framework (https://arxiv.org/abs/2410.19451)
- **What's New**: 이 논문은 전통 중국 의학(Traditional Chinese Medicine, TCM) 영역에서 대형 언어 모델(large language models, LLMs)의 성능을 향상시키기 위한 프롬프트 엔지니어링(prompt engineering)의 응용을 탐구합니다. TCM-Prompt라는 프레임워크를 제안하며, 이는 다양한 사전 훈련 언어 모델(pre-trained language models, PLMs), 템플릿, 토크나이제이션(tokenization), 및 언어화(verbalization) 방법을 통합하여 연구자들이 TCM 관련 작업에 특화된 모델을 쉽게 구성하고 조정할 수 있도록 합니다.

- **Technical Details**: TCM-Prompt 프레임워크는 질병 분류(disease classification), 증후군 식별(syndrome identification), 한방 약 추천(herbal medicine recommendation), 그리고 일반 NLP 작업(general NLP tasks)에서 실험을 수행하였습니다. 이 연구는 기존의 방법론과 비교할 때 우리의 접근법의 효과성과 우수성을 입증합니다.

- **Performance Highlights**: 연구 결과는 프롬프트 엔지니어링이 TCM과 같은 전문 영역에서 LLMs의 성능 향상을 위한 유망한 기술이라는 것을 시사하며, 디지털화(digitalization), 현대화(modernization), 그리고 개인화된 의학(personalized medicine) 분야에서의 잠재적 응용 가능성을 보여줍니다.



### Gradient Descent Efficiency Index (https://arxiv.org/abs/2410.19448)
Comments:
          12 Pages, 3 Figures

- **What's New**: 이 논문에서는 기울기 하강법(Gradient Descent)의 각 반복(iteration)의 효과성을 정량화하기 위한 새로운 효율성 지표, Ek을 소개합니다. 이 지표는 오류의 상대적 변화와 반복 간의 손실 함수의 안정성을 고려하여, 자원이 제한된 환경에서의 훈련 시간과 비용 절감에 기여할 수 있습니다.

- **Technical Details**: 기존의 기울기 하강법은 최적의 정지 지점을 결정하는 데 어려움이 있으며, 실험적으로 Ek이 여러 데이터셋과 모델에서 기울기 하강법의 수렴성을 평가하는 데 기존 성능 지표를 보완하는 데 사용되었습니다. 이 지표는 상대적인 기울기 변화, 초기 학습률, 학습률 감소율, 계수의 절대 변화 및 반복 횟수 등을 포함하는 새로운 비율로 정의됩니다.

- **Performance Highlights**: Ek 지표는 기울기 하강법의 효율성을 보다 체계적으로 평가하기 위한 새로운 접근 방식을 제공하며, 머신 러닝 응용 프로그램에서 최적화 알고리즘의 선택 및 조정을 위한 정보에 기반한 결정을 내리는 데 기여할 수 있는 잠재력을 가지고 있습니다.



### Robust Time Series Causal Discovery for Agent-Based Model Validation (https://arxiv.org/abs/2410.19412)
- **What's New**: 이 연구에서는 복잡하고 노이즈가 많은 시계열 데이터에 적용할 때 기존 인과 발견 방법의 정확성과 강건성을 향상시키기 위해 Robust Cross-Validation (RCV) 접근 방식을 제안합니다. 이는 Agent-Based Model (ABM) 검증을 위한 새로운 프레임워크를 통합하여 다양한 데이터 및 모델 구조를 처리할 수 있도록 설계되었습니다.

- **Technical Details**: RCV-VarLiNGAM과 RCV-PCMCI는 기존의 인과 발견 알고리즘인 VAR-LiNGAM과 PCMCI의 새로운 확장으로, 노이즈의 영향을 줄이고 높은 차원의 시간 의존적 데이터에서도 신뢰할 수 있는 인과 관계 결과를 제공합니다. 이 방법은 데이터셋 속성을 분석하고, 다양한 실험에서 성능을 평가합니다.

- **Performance Highlights**: 제안된 방법은 합성 데이터셋과 복잡한 시뮬레이션된 fMRI 데이터셋을 사용하여 평가되었으며, 기존의 인과 발견 방법들보다 더 높은 신뢰성을 보여주었습니다. 데이터셋의 특성(선형성, 노이즈 분포, 정상성 및 인과 구조 밀도)이 성능에 미치는 영향을 분석하여, RCV 방법이 다양한 상황에서 어떻게 작동하는지를 비교합니다.



### Analysis of Financial Risk Behavior Prediction Using Deep Learning and Big Data Algorithms (https://arxiv.org/abs/2410.19394)
- **What's New**: 이 논문은 복잡하고 동적인 금융 시장에서 전통적인 금융 리스크 예측 방법이 대량의 데이터셋과 복잡한 행동 패턴을 처리하는 데 어려움을 겪고 있다는 점에 주목합니다. 그리고 딥 러닝 (deep learning)과 빅데이터 (big data) 알고리즘을 활용한 금융 리스크 행동 예측의 가능성과 효과를 탐구합니다.

- **Technical Details**: 딥 러닝 기반의 빅데이터 리스크 예측 프레임워크가 설계되었으며, 실제 금융 데이터셋에 대한 실험적 검증이 이루어졌습니다. 본 연구에서는 딥 러닝과 빅데이터의 금융 분야에서의 적용 및 장점이 분석됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 금융 리스크 행동 예측의 정확성을 크게 향상시키며, 금융 기관의 리스크 관리에 귀중한 지원을 제공함을 보여주었습니다. 또한 딥 러닝 적용에 따른 도전 과제와 미래 연구 방향에 대한 논의도 포함되어 있습니다.



### CLAP. I. Resolving miscalibration for deep learning-based galaxy photometric redshift estimation (https://arxiv.org/abs/2410.19390)
Comments:
          22 + 6 pages, 9 + 5 figures

- **What's New**: 이 논문은 Photometric Redshift(포토메트릭 적색편이) 추정을 위한 새로운 방법인 CLAP(Contrastive Learning and Adaptive KNN for Photometric Redshift)를 제안합니다. 이 방법은 supervised contrastive learning(SCL)과 k-nearest neighbours(KNN)을 활용하여 향상된 확률 밀도 추정을 제공합니다.

- **Technical Details**: CLAP는 raw probability density estimates(원시 확률 밀도 추정)를 구성하고 보정하는 과정에서 supervised contrastive learning(SCL)과 k-nearest neighbours(KNN) 기술을 활용합니다. 마지막 추정치를 도출하기 위해 end-to-end discriminative models를 재적합(refitting)하도록 설계되었습니다.

- **Performance Highlights**: CLAP는 기존의 모델들에 비해 확률 밀도 추정의 보정(calibration)에서 뛰어난 성능을 보이며, 높은 정확도와 효율성을 유지합니다. 특히, 많은 우주 및 천체 물리학적 응용에 필요한 정확하게 보정된 포토메트릭 적색편이 확률 밀도를 얻기에 적합한 접근법으로 자리 잡을 것으로 기대됩니다.



### Investigating the Role of Prompting and External Tools in Hallucination Rates of Large Language Models (https://arxiv.org/abs/2410.19385)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 환각(hallucination) 감소를 위한 다양한 프롬프트(prompt) 전략과 프레임워크에 대한 포괄적인 실증 평가를 제공합니다. 또한, 외부 도구가 추가된 LLM 에이전트의 환각 생성 비율에 미치는 영향을 조사합니다.

- **Technical Details**: LLM은 다양한 자연어 처리(NLP) 작업을 수행할 수 있는 강력한 계산 모델입니다. 이 연구에서는 다양한 프롬프트 기법을 넓은 벤치마크 데이터셋에 적용하여 각 방법의 정확도와 환각 비율을 평가하고, 환각의 맥락 의존성을 탐구합니다. 온도(temperature) 설정, 양자화(quantization) 등과 같은 모델 압축 기법도 논의됩니다.

- **Performance Highlights**: 단순한 프롬프트 기법이 더 복잡한 방법보다 환각을 줄이는 데 종종 더 효과적임을 보여주며, LLM 에이전트는 외부 도구 사용으로 인해 환각 비율이 현저하게 증가할 수 있음을 입증했습니다.



### Multi-Agent Reinforcement Learning with Selective State-Space Models (https://arxiv.org/abs/2410.19382)
Comments:
          17 pages, 7 figures

- **What's New**: 이번 연구에서는 Multi-Agent Transformer (MAT) 대신 Mamba라는 새로운 State-Space Model (SSM)을 활용하여 Multi-Agent Reinforcement Learning (MARL)에서 성능과 효율성을 개선할 수 있는 가능성을 탐구합니다. Mamba는 기존의 Transformer 모델보다 더 나은 확장성과 계산 효율성을 제공합니다.

- **Technical Details**: MAM (Multi-Agent Mamba) 모델은 기본 및 양방향 Mamba 블록과 함께 새로운 'cross-attention' Mamba 블록을 포함한 MAT의 수정된 버전입니다. 이 모델은 기존 MAT 아키텍처에서 주의(attention) 메커니즘을 Mamba 블록으로 대체하여 성능을 비교합니다. Mamba는 빠른 추론(fast inference)과 선형적인 시퀀스 길이 확장(linear scaling)을 자랑합니다.

- **Performance Highlights**: MAM은 여러 표준 MARL 벤치마크 환경에서 MAT과 유사한 성능을 보이며, 많은 수의 에이전트로 확장할 때 더 높은 효율성을 제공합니다. 따라서 SSM이 Transformer를 대체할 수 있는 가능성을 제시하며, 큰 에이전트의 수를 지원하는 보다 효과적인 확장이 가능하다는 것을 나타냅니다.



### BitPipe: Bidirectional Interleaved Pipeline Parallelism for Accelerating Large Models Training (https://arxiv.org/abs/2410.19367)
Comments:
          10 pages, 13 figures

- **What's New**: 이번 논문에서는 대규모 모델 교육을 가속화하기 위한 새로운 기법인 BitPipe를 제안합니다. BitPipe는 양방향(interleaved) 파이프라인 병렬 처리 방식을 결합하여 컴퓨팅 시간을 줄이고 동시에 실행 가능한 장치 수를 두 배로 늘립니다.

- **Technical Details**: BitPipe는 양방향 파이프라인과 인터리브(interleaved) 파이프라인을 융합하여 하이브리드(pipeline) 방식을 도입합니다. V자형 스케줄을 사용하여 장치 간 통신을 줄이고 오버랩(overlap)하는 기법도 포함되어 있습니다. 이를 통해 각 단일 마이크로 배치(micro-batch)의 연산 시간을 줄이고, 장치 간 소통의 지연을 완화할 수 있습니다.

- **Performance Highlights**: BitPipe는 최대 1.28배의 향상을 달성하며, GPT 스타일 및 BERT 스타일 모델에 대해 SOTA 동기화 접근 방식과 비교하여 훈련 처리량(training throughput)이 1.05x-1.28x 개선되었습니다.



### Interleaving Text and Number Embeddings to Solve Mathemathics Problems (https://arxiv.org/abs/2410.19353)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 수치 예측 및 산술 연산 성능 향상을 위해 표현력이 풍부한 숫자 임베딩과 텍스트와 숫자 간의 구분을 위한 라우팅 레이어를 소개합니다. 이는 기존의 숫자 토큰화 방법의 한계를 극복하는 데 기여합니다.

- **Technical Details**: 연구에서는 MLP (Multi-Layer Perceptron)를 사용하여 각각의 숫자에 독특한 벡터 방향을 할당하고, Mixture-of-Experts (MoE) 모델에서 영감을 받은 라우팅 레이어를 통해 텍스트와 숫자 임베딩 간의 구조를 구분합니다. 이를 통해 숫자와 텍스트 간의 분포를 다르게 하여 산술 연산을 가능하게 합니다.

- **Performance Highlights**: 45M 파라미터의 인코더-디코더 아키텍처에서 우리의 방법은 $R^2$=0.9988을 달성하였으며, 여러 가지 수치 비편향과 아티팩트 개선을 관찰했습니다. 실험 결과, 우리의 방법은 숫자와 텍스트가 혼합된 복잡한 문제에서 기존 모델 대비 월등한 성능을 보였습니다.



### Interpreting Neural Networks through Mahalanobis Distanc (https://arxiv.org/abs/2410.19352)
Comments:
          11 pages, October 2024

- **What's New**: 이 논문은 신경망의 선형 레이어와 Mahalanobis 거리(Mahalanobis distance)를 연결하는 이론적 프레임워크를 소개합니다. 기존 연구들이 주로 성능 최적화를 위해 활성화 함수(activation functions)를 탐구했던 반면, 본 연구는 통계적 거리 척도를 통해 이러한 함수들을 해석하는 데 초점을 맞추고 있습니다. 이러한 연결을 통해 신경망 모델의 해석 가능성을 높일 수 있는 기반을 제공합니다.

- **Technical Details**: 이 논문은 절대값(Absolute Value, Abs) 활성화가 거리 기반 해석을 촉진하는 방법을 보여주고, Mahalanobis 거리 접근 방식을 근사화할 때 신경망이 학습할 가능성이 있는 해결 공간을 분석합니다. 비독점적인 화이트닝 변환(whitening transformations)의 비유일성(non-uniqueness)과 Abs 활성화된 선형 노드의 역할을 조사하며, 이를 통해 신경망 설계와 해석 가능성에 대한 더 광범위한 의미를 논의합니다.

- **Performance Highlights**: 이 연구는 신경망보다 해석할 수 있는 모델을 개발하기 위한 기초를 제공하며, 이는 투명성이 요구되는 응용 분야에서 중요한 역할을 합니다. 비록 이 연구가 이론적인 연구에 국한되어 있지만, 제안된 거리 기반 해석이 모델의 견고성을 향상시키고 일반화(generalization)를 개선하며 신경망 결정의 보다 직관적인 설명을 제공할 잠재력이 있습니다.



### pEBR: A Probabilistic Approach to Embedding Based Retrieva (https://arxiv.org/abs/2410.19349)
- **What's New**: 본 논문은 임베딩 기반 검색(embedding based retrieval)에서 전통적인 빈도주의(frequentist) 접근 방식을 벗어나 확률적(probabilistic) 접근 방식을 제안합니다. 특히, 코사인 유사도(threshold)를 유동적으로 계산할 수 있는 기법을 도입하여 다양한 쿼리에 대해 동적인 검색이 가능하도록 하였습니다.

- **Technical Details**: 제안하는 방법은 각 쿼리에 대한 항목 분포를 학습하여, 확률적 누적 분포 함수(probabilistic cumulative distribution function, CDF) 값을 활용하여 코사인 유사도(threshold)를 동적으로 계산합니다. 이로써 고정된 개수의 항목이나 고정된 코사인 임계값을 사용하는 기존 방법보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안한 방법(pEBR)은 검색 정밀도(precision)와 재현율(recall) 모두를 유의미하게 개선했습니다. 또한, ablation study를 통해 두 가지 쿼리(헤드 쿼리와 테일 쿼리) 간의 차이를 잘 포착하는 확률적 접근 방식의 효과를 검증하였습니다.



### High Resolution Seismic Waveform Generation using Denoising Diffusion (https://arxiv.org/abs/2410.19343)
- **What's New**: 이 연구는 고주파 지진파형 생성을 위한 효율적이고 확장 가능한 새로운 생성 모델을 도입합니다. 이 모델은 지진의 크기, 기록 거리, 사이트 조건 및 단층 유형과 같은 주요 입력 매개변수에 따라 조정된 잠재 표현(latent representation)을 생성하는 최첨단 확산 모델(difffusion model)을 활용합니다.

- **Technical Details**: 이 모델은 지진파형 데이터를 스펙트로그램(spectrogram) 형태로 변환하여 자동인코더(autoencoder)를 사용해 저차원 서브매니폴드(submanifold)로 축소합니다. 이 과정은 고주파 세부정보를 포함하여 최대 50Hz의 주파수 내용을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델의 성능은 실제 지진 기록과의 평균 경향 및 변동성을 정확하게 재현함으로써 검증되었습니다. 우리는 이 모델을 공개할 것을 주장하며, 커뮤니티의 지진파 모델 평가 노력에 통합될 수 있기를 희망합니다.



### Two are better than one: Context window extension with multi-grained self-injection (https://arxiv.org/abs/2410.19318)
Comments:
          The code is available at this https URL

- **What's New**: 이 논문은 SharedLLM이라는 새로운 접근 방식을 제안합니다. SharedLLM은 다중 세분화된 컨텍스트 압축 및 쿼리 인지 정보 검색의 설계 철학에 기반하여 두 개의 짧은 컨텍스트 LLM으로 구성됩니다. 이 모델은 압축된 정보를 전달하기 위해 하부 모델과 상부 모델 간의 효율적인 연결을 최적화합니다.

- **Technical Details**: SharedLLM은 LLaMA-2와 같은 두 개의 짧은 컨텍스트 LLM으로 구성됩니다. 하부 모델은 컨텍스트 정보를 압축하고, 상부 모델은 이를 기반으로 하는 언어 모델링을 수행합니다. 또한, 그들은 특화된 트리 형태의 데이터 구조를 도입하여 다양한 레벨의 정보를 효율적으로 인코딩, 저장, 검색합니다. 정보를 주입하는 단계에서는 청크-레벨의 위치 ID를 정의하여 쿼리의 상대적인 위치를 인식할 수 있도록 합니다.

- **Performance Highlights**: SharedLLM은 8K 토큰의 텍스트로 훈련되며, 128K 토큰 시퀀스를 처리할 수 있는 뛰어난 외삽 능력을 나타냅니다. 200K 이상의 최대 길이에 대한 모든 실험은 단일 A800 80G GPU에서 수행할 수 있으며, SharedLLM은 상대적으로 낮은 메모리 소비로 모든 기준 모델보다 몇 배 더 빠른 속도를 보여줍니다.



### COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training (https://arxiv.org/abs/2410.19313)
Comments:
          16 pages. 9 Figures. 8 Tables

- **What's New**: 이 논문은 COAT라는 새로운 FP8 훈련 프레임워크를 소개합니다. 이는 대규모 모델 훈련 시 메모리 사용량을 크게 줄이는 데 집중하여, 기존 방법의 한계를 극복합니다.

- **Technical Details**: COAT는 두 가지 주요 혁신을 통해 메모리 사용 최적화에 기여합니다: (1) Dynamic Range Expansion, 이는 옵티마이저 상태 분포를 FP8 표현范围에 맞추어 수량화 오류를 줄입니다. (2) Mixed-Granularity Activation Quantization, 이는 비선형 레이어에 대해 세분화된 수량화(fine-grained quantization)를 적용하여 활성화 메모리를 최적화합니다.

- **Performance Highlights**: COAT는 BF16에 비해 1.54배의 메모리 축소를 달성하고, Llama 7B 및 30B 모델에 대해 1.43배의 훈련 속도 향상을 보입니다. 또한, 분산 훈련 환경에서 배치 크기를 두 배로 증가시킬 수 있어 대규모 모델 훈련을 지원합니다.



### Flow Generator Matching (https://arxiv.org/abs/2410.19310)
- **What's New**: 본 논문은 Flow Generator Matching (FGM)이라는 혁신적인 접근 방식을 제시하여, flow-matching 모델의 샘플링 속도를 가속화하고 원래의 성능을 유지하는 방법을 제안합니다.

- **Technical Details**: FGM은 확률적(distillation) 프레임워크로, flow 모델의 샘플링 과정을 단일 단계 생성기(one-step generator)로 간소화합니다. 이를 통해 컴퓨팅 자원을 보다 효율적으로 활용할 수 있습니다. 또한 CIFAR10 데이터셋에서 새로운 Fréchet Inception Distance (FID) 점수 3.08을 기록하며 성능을 확인했습니다.

- **Performance Highlights**: 이 결과는 특히 MM-DiT 아키텍처에 기반한 Stable Diffusion 3 모델의 distillation을 통해 생성된 MM-DiT-FGM 모델이 GenEval 벤치마크에서 뛰어난 생성 품질을 보여주는 것으로 나타났습니다.



### Semantics in Robotics: Environmental Data Can't Yield Conventions of Human Behaviour (https://arxiv.org/abs/2410.19308)
- **What's New**: 이 논문에서는 로봇 공학과 AI에서의 의미론(semantics)에 대한 명확한 정의가 없음을 지적하며, 자율 에이전트를 위한 추가 데이터의 중요성을 강조합니다.

- **Technical Details**: 의미론이란 환경 데이터에서 직접 추출될 수 없는 추가 데이터를 의미하며, 이는 인간 행동의 관습(conventions of human behaviour)으로 구성된 데이터로 이해되어야 한다고 주장합니다. 여기에는 레이블(labels), 장소(places), 온톨로지(ontologies), 그리고 어포던스(affordances)가 포함됩니다.

- **Performance Highlights**: 특히 객체 어포던스는 물리학(physics)과 객체 조합(object combinations)에 대한 이해를 요구하며, 이는 인공지능이 인공 초지능(artificial superintelligence)을 이룰 경우를 구성할 수 있음을 보여줍니다.



### TEARS: Textual Representations for Scrutable Recommendations (https://arxiv.org/abs/2410.19302)
- **What's New**: TEARS(TExtuAl Representations for Scrutable Recommendations) 시스템은 사용자의 관심사를 높은 차원 잠재 임베딩 대신 자연어 텍스트로 표현하여 투명성을 제공하고 사용자가 직접 수정할 수 있도록 합니다. 이를 통해 추천 시스템의 투명성과 사용자의 제어력을 강화합니다.

- **Technical Details**: TEARS는 두 개의 인코더를 사용하여 전통적인 블랙박스 모델이 처리한 역사적 상호작용을 기반으로 숫자 블랙박스 임베딩을 생성하고, 사용자가 편집할 수 있는 자연어 요약을 수집하여 요약 기반 임베딩으로 변환합니다. 이 두 임베딩을 최적 운송(optimal transport) 절차를 통해 정렬하고, 혼합 계수를 통해 추천을 조정할 수 있습니다.

- **Performance Highlights**: TEARS는 현대의 LLM(대규모 언어 모델)을 통해 생성된 사용자 요약을 사용하여 추천 성능을 개선하고 있으며, 사용자 요약을 편집함으로써 추천 결과를 효과적으로 조정할 수 있는 유연성을 제공합니다. 세 가지 시뮬레이션된 사용자 작업을 통해 TEARS의 제어력이 검증되었으며, 기존 VAE(변분 오토인코더) 모델들의 성능을 초과하는 결과를 나타냈습니다.



### A Stock Price Prediction Approach Based on Time Series Decomposition and Multi-Scale CNN using OHLCT Images (https://arxiv.org/abs/2410.19291)
Comments:
          32 pages, 5 figures, 5 tables

- **What's New**: 본 논문에서는 중국 A주 시장에서 주가 변동 예측을 위해 Sequence-based Multiscale Fusion Regression Convolutional Neural Network (SMSFR-CNN)이라는 새로운 방법을 제안합니다. 이 모델은 CNN을 통해 순차적 특성을 학습하고 이미지 특성과 결합하여 예측 정확도를 높입니다.

- **Technical Details**: SMSFR-CNN은 두 가지 새로운 방법론을 제안합니다. 첫 번째는 거래량을 주식 회전율로 대체하여 보다 안정적인 특성을 얻는 것이고, 두 번째는 OHLCT(Opening price, Highest price, Lowest price, Closing price, Turnover rate) 이미지를 시간 구분자와 통합하여 새로운 이미지 특성 TS-OHLCT를 생성합니다. 이 모델은 긴 시퀀스 데이터를 여러 시간대에 따라 분해하고 중요도에 따라 다른 특성 가중치를 부여합니다.

- **Performance Highlights**: 제안한 SMSFR-CNN 모델은 긍정적 예측 가치(Positive Predictive Value) 61.15% 및 부정적 예측 가치(Negative Predictive Value) 63.37%를 달성하였으며, 5일 내 주가 추세 예측에 있어 165.09%의 총 이익을 기록했습니다.



### ST-NeRP: Spatial-Temporal Neural Representation Learning with Prior Embedding for Patient-specific Imaging Study (https://arxiv.org/abs/2410.19283)
Comments:
          14 pages with 10 figures and 6 tables

- **What's New**: 이번 논문에서는 환자 맞춤형 이미징 연구를 위한 공간-시간 신경 표현 학습 방법(ST-NeRP)을 제안합니다.

- **Technical Details**: ST-NeRP 모델은 참조 시점에서 이미지를 인코딩하기 위해 암시적 신경 표현(Implicit Neural Representation, INR) 네트워크를 사용하며, 또 다른 INR 네트워크를 통해 공간-시간 지속적인 변형 함수(deformation function)를 학습합니다.

- **Performance Highlights**: ST-NeRP 모델은 4D CT 및 흉부와 복부 이미징을 포함한 다양한 순차적 이미지 시리즈에 적용되어 생기는 해부학적 변화를 효과적으로 모니터링할 수 있는 상당한 잠재력을 보여줍니다.



### UbiHR: Resource-efficient Long-range Heart Rate Sensing on Ubiquitous Devices (https://arxiv.org/abs/2410.19279)
- **What's New**: UbiHR는 실시간 장거리 시공간 모델을 활용하여 노이즈에 독립적인 심박수 인식을 가능하게 하는 시스템입니다. 이 시스템은 일반 모바일 기기에서 즉각적인 시청각 데이터를 샘플링하고 전처리할 수 있는 메커니즘을 포함하고 있습니다.

- **Technical Details**: UbiHR는 세 가지 주요 모듈로 구성되어 있습니다. 첫째, 에너지 소비와 얼굴 키포인트 탐지의 즉시성을 조절할 수 있는 적응형 주기 얼굴 비디오 샘플링 모듈입니다. 둘째, 환경 조명 변화와 사용자 머리 움직임으로부터 노이즈를 제거하는 동적 노이즈 인식 얼굴 이미지 전처리 모듈입니다. 셋째, 경량화된 시공간 모델링을 위한 빠른 장거리 시공간 심박수를 인식하는 모듈입니다.

- **Performance Highlights**: UbiHR는 80명의 참가자와 다양한 실험을 통해 74.2%의 정확도 향상과 51.2%의 지연시간 감소를 기록하였습니다. 특히 긴 거리 비디오 스트림에서 42.8%의 정확도 향상이 있었습니다.



### Applying sparse autoencoders to unlearn knowledge in language models (https://arxiv.org/abs/2410.19278)
- **What's New**: 본 연구는 Sparse Autoencoders (SAEs)를 사용하여 언어 모델에서 지식을 제거하는 방법을 탐구합니다. 기존의 파인튜닝 기반 방법들과는 달리 SAEs를 통해 좀 더 해석가능한 방식으로 지식을 'unlearn' 할 수 있는 가능성을 모색합니다.

- **Technical Details**: 연구에서는 SAEs의 피처 활성화를 기반으로 언어 모델에 개입하며, 다수의 SAE 피처를 동시에 사용하여 다양한 주제를 'unlearn'합니다. 사용된 데이터셋은 생물학 관련 WMDP-bio로, 위험한 생물학적 지식을 평가하기 위해 선택되었습니다. 각 피처의 활성화를 고정된 음수값으로 설정하여 개입합니다.

- **Performance Highlights**: SAEs 기반의 방법이 기존 Representation Misdirection for Unlearning (RMU) 기법보다 성공적인 결과를 도출하였지만, 여전히 파인튜닝 기반 기술들과 비교할 때 개선이 필요합니다. 특히, SAEs의 개입이 복수 피처를 동시에 다룰 경우 원치 않는 부작용이 발생할 수 있다는 점이 중요한 발견으로 제시됩니다.



### Ripple: Accelerating LLM Inference on Smartphones with Correlation-Aware Neuron Managemen (https://arxiv.org/abs/2410.19274)
- **What's New**: 이번 연구에서는 모바일 기기에서의 대규모 언어 모델 (LLM) 추론을 가속화하기 위한 새로운 접근법인 Ripple을 제안합니다. Ripple은 플래시 메모리에서 뉴런 배치를 최적화하여 I/O (Input/Output) 성능을 향상시킵니다.

- **Technical Details**: Ripple은 뉴런 공동 활성화 (Neuron Co-Activation) 개념을 활용하여, 자주 함께 활성화되는 뉴런들을 연결하여 연속적인 읽기 접근을 용이하게 합니다. 이 방법은 두 단계로 이루어져 있으며, 오프라인 단계에서는 뉴런 배치를 재구성하고, 온라인 단계에서는 맞춤형 데이터 접근 및 캐싱 전략을 적용합니다.

- **Performance Highlights**: Ripple은 다양한 스마트폰 및 LLM에 대한 평가를 통해 I/O 대기 시간에서 최대 5.93배 개선을 달성하였으며, 기존 솔루션에 비해 성능 격차를 효과적으로 줄였습니다.



### Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning (https://arxiv.org/abs/2410.19258)
Comments:
          18pages,submitted to ICLR 2025

- **What's New**: 이번 논문에서는 HeadKV라는 새로운 head-level KV 캐시 압축 방법을 제안합니다. 이는 각 attention head의 중요도를 평가하여 KV 캐시의 예산을 효율적으로 할당하는 방법으로, 이전 연구에서 제안된 단순 layer-level 압축 방식보다 성능이 우수합니다.

- **Technical Details**: HeadKV는 각 attention head의 retrieval과 reasoning 능력을 평가하여 KV 캐시의 적절한 할당을 결정합니다. 이 방법은 LongBench와 LooGLE과 같은 다양한 검증 벤치마크를 통한 실험을 통해 각기 다른 모델 아키텍처(Llama-3-8B-Instruct, Mistral-7B-Instruct)에서 효과를 입증했습니다.

- **Performance Highlights**: HeadKV를 사용했을 때 KV 캐시의 1.5%만을 유지하면서도 전체 KV 캐시 모델에서 97%에 해당하는 성능을 유지할 수 있었고, 특히 낮은 자원 환경에서 강력한 기준선(base line)을 초과하는 성능을 보여주었습니다.



### Non-rigid Relative Placement through 3D Dense Diffusion (https://arxiv.org/abs/2410.19247)
Comments:
          Conference on Robot Learning (CoRL), 2024

- **What's New**: 본 논문에서는 ''relative placement'' (상대 배치)의 개념을 변화하는 물체에 대한 기하학적 관계로 확장하는 ''cross-displacement'' (교차 변위)를 제안합니다. 이를 통해 비틀림이 가능한 물체를 대상으로 하는 새로운 시각 기반 방법을 소개합니다.

- **Technical Details**: 이 방법은 dense diffusion (밀집 확산)을 통해 cross-displacement 를 학습하며, 이전의 연구에서는 다루지 못했던 비형상 물체의 기하학적 관계를 모형화합니다. 이는 물체의 변형(transformations)과 같은 복잡한 상황에서도 효과적인 학습이 가능함을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 다양한 비형상 작업에서 일반적인 실제 환경과 시뮬레이션에서의 새로운 객체 인스턴스, 분포 외(scene configurations) 장면에 대한 일반화를 달성했습니다. 여러 가지 다중 모드(multimodal) 목표에서도 뛰어난 성능을 입증하였습니다.



### Learning Diffusion Policies from Demonstrations For Compliant Contact-rich Manipulation (https://arxiv.org/abs/2410.19235)
- **What's New**: DIPCOM(Diffusion Policies For Compliant Manipulation)이라는 새로운 확산 기반 프레임워크를 소개하여 로봇이 복잡한 접촉 강조 조작을 배우는 데 도움을 줍니다. 이 연구는 다중 모드(distribution modeling)로 힘 조절을 향상시키고, 실제 작업에서 그 효과를 시연하였습니다.

- **Technical Details**: DIPCOM은 생성적 확산 모델(generative diffusion models)을 활용하여 카르테시안 최종 효과기(pose)를 예측하고 로봇 팔의 강성을 조정하여 작업에 필요한 힘을 유지합니다. 이 접근 방식은 강도 조절을 위한 정책을 개발하고, 로봇이 접촉이 빈번한 조작을 수행할 때 필요한 힘을 지속적으로 조절할 수 있도록 합니다.

- **Performance Highlights**: DIPCOM은 실제 세계 작업에서 강력한 성능을 보여주었으며, 기존 방법들과 비교하여 저희 접근 방식의 장점 및 최상의 모범 사례를 강조하였습니다. 특히, 장기 작업에서 힘을 조절하며 정밀한 제어를 유지하는 데 효과적임을 입증했습니다.



### Developing a Tutoring Dialog Dataset to Optimize LLMs for Educational Us (https://arxiv.org/abs/2410.19231)
- **What's New**: 최근의 큰 언어 모델(LLMs)의 발전은 교육 분야에서의 확장 가능한 응용 가능성을 보여주었지만, 대화 기반 튜터링 시스템에서의 사용은 효율적인 교수법( pedagogical strategies) 필요성과 전문가가 정제한 데이터셋의 높은 비용으로 인해 여전히 도전적인 과제입니다. 본 연구는 읽기 이해 문제 해결에 있어 1:1 튜터링을 위해 더 작고 저렴한 LLM을 활용하는 방식을 탐구합니다.

- **Technical Details**: 우리는 합성 튜터링 대화 데이터셋( synthetic tutoring dialog dataset)을 개발하고, 이를 통해 소형 LLM을 미세 조정(fine-tuning)하였습니다. 더 나아가, 실제 튜터링 시나리오에서 미세 조정된 모델과 대형 모델의 성능을 비교하는 인터랙티브 실험을 진행하였습니다. 연구에 사용된 모델은 Transformer 기반의 혼합 전문가(Mixture of Experts) LLM인 Mistral ‘8x7b’와 ‘7b’ 모델입니다.

- **Performance Highlights**: 연구 결과는 미세 조정된 모델이 대형 모델과 동등한 성능을 보이며, 비용 측면에서도 우수하다는 것을 보여주었습니다. 즉, 더 작은 모델이 저렴한 비용으로 교육 환경에서 LLM 기반 튜터링 시스템을 구현하는데 유효하고 비용 효율적인 접근법임을 증명하였습니다.



### Hierarchical Mixture of Experts: Generalizable Learning for High-Level Synthesis (https://arxiv.org/abs/2410.19225)
- **What's New**: 이번 연구에서는 하드웨어 지식이 없는 소프트웨어 개발자들이 FPGA 설계를 통해 HLS(High-Level Synthesis)에서 겪는 도전 과제를 해결하기 위해 두 단계의 계층적 전문가 혼합 모델(Mixture of Experts, MoE)을 제안합니다.

- **Technical Details**: 제안된 MoE 구조는 소스 코드의 세 가지 자연적 세분화, 즉 노드(node), 기본 블록(basic block), 그래프(graph)에서 작동하며, 최종 결정을 위해 이 세 가지 세분화를 집계하는 고수준 MoE를 포함합니다.

- **Performance Highlights**: 실험 결과, 계층적 MoE 사용 시 FPGA 설계에서 평균 26.6%의 성능 향상을 확인했습니다.



### Can Stories Help LLMs Reason? Curating Information Space Through Narrativ (https://arxiv.org/abs/2410.19221)
- **What's New**: 이 논문에서는 내러티브 요소를 통합하여 대형 언어 모델(LLM)이 복잡한 문제를 보다 효과적으로 해결할 수 있도록 돕는 새로운 접근법인 'Thought의 이야기(Story of Thought, SoT)'를 제안합니다.

- **Technical Details**: SoT는 문제 진술 주위에 내러티브를 구성하고 관련 정보를 식별하고 조직하는 프레임워크를 생성하여 문제 해결을 위한 프롬프트 기술에 내러티브 구조를 통합하는 방법입니다. 이 방법은 질문 명확화, 내러티브 생성 및 문제 해결의 세 단계를 포함합니다.

- **Performance Highlights**: 실험 결과, SoT를 사용한 LLM들은 GPQA 및 JEEBench 데이터셋의 물리학, 화학, 수학 및 생물학 문제에서 기존의 프롬프트 기술보다 일관되게 우수한 성능을 보였습니다.



### Robot Behavior Personalization from Sparse User Feedback (https://arxiv.org/abs/2410.19219)
- **What's New**: 이 논문에서는 서비스 로봇이 사용자 선호에 맞춰 다양한 작업을 수행할 수 있도록 적응하는 방법을 제안합니다. 이는 기존의 개별 사용자에 대한 작업 구체적 데이터를 필요로 하는 개인화 접근 방식을 대체하는 것입니다. 새로운 Task Adaptation using Abstract Concepts (TAACo) 프레임워크를 개발하여 로봇이 사용자 피드백을 기반으로 작업의 수행 방법을 예측할 수 있도록 합니다.

- **Technical Details**: TAACo는 사용자가 제공한 피드백을 반영하여 작업에 대한 사용자의 선호되는 지원 유형을 예측합니다. 이 모델은 다양한 추상 개념을 통해 사용자 선호를 설명하며, 소량의 사용자 피드백으로도 모든 가정 내 작업에 일반화할 수 있는 능력을 가집니다. 또한, TAACo는 데이터 공유의 필요성을 없애고, 사용자의 데이터가 지역적으로 저장되고 활용될 수 있게 합니다.

- **Performance Highlights**: TAACo는 5명의 사용자의 선호 데이터를 기반으로 평가되었으며, 예측 정확도에서 GPT-4보다 16% 더 높은 성능을 보였고, 규칙 기반 시스템보다 54% 더 우수한 결과를 나타냈습니다. 이 모델은 40개의 샘플 사용자 피드백을 통해 선호되는 지원 방식의 정확한 예측을 달성했습니다.



### Taxonomy-guided Semantic Indexing for Academic Paper Search (https://arxiv.org/abs/2410.19218)
Comments:
          EMNLP'24

- **What's New**: 이번 연구에서는 학술 논문 검색에서의 효율성을 높이기 위해 Taxonomy-guided Semantic Indexing (TaxoIndex) 프레임워크를 제안합니다. 이는 핵심 개념을 추출하고 이를 학술 분류 체계에 기반한 의미 색인으로 구성하여 쿼리와 문서 간의 효과적인 학술 개념 매칭을 가능하게 합니다.

- **Technical Details**: TaxoIndex는 자연어 처리(NLP) 및 기계 학습(ML) 기법을 사용하여 논문에서 핵심 개념을 추출합니다. 이 프레임워크는 주제 수준과 구절 수준의 두 가지 세부 사항으로 분류된 의미 색인을 활용하여 문서를 보다 정확하게 표현합니다. 또한, 인덱스 학습(index learning)을 통해 서로 다른 용어로 표현된 쿼리와 관련된 개념을 자동으로 찾도록 모델을 훈련합니다.

- **Performance Highlights**: Extensive experiments show that TaxoIndex는 제한된 학습 데이터에도 불구하고 매우 효과적인 성능 개선을 가져옵니다. 기존의 Dense retrievers 보다 향상된 검색 품질을 제공하며, 특정 학술 개념 매칭의 해석 가능성을 개선하는 데에도 큰 기여를 합니다.



### No Free Lunch: Fundamental Limits of Learning Non-Hallucinating Generative Models (https://arxiv.org/abs/2410.19217)
- **What's New**: 본 연구에서는 생성 모델에서 발생하는 'hallucinations'(할루시네이션, 환각 현상)에 대한 이론적 분석 프레임워크를 제시하였습니다. 이 논문은 비할루시네이션 생성 모델의 학습 가능성을 학습 이론적 관점에서 rigorously(엄밀하게) 분석합니다.

- **Technical Details**: 제시된 연구는 non-hallucinating learning(비할루시네이션 학습)이 학습 데이터셋에만 의존할 경우 통계적으로 불가능하다는 결론을 도출했습니다. 이 논문에서는 inductive biases(유도 편향)를 실제 사실에 맞춘 학습 과정에 통합하는 것이 필수적이라고 주장하며, 이러한 유도 편향을 결과적으로 VC-dimension(VC 차원)이 유한한 개념 클래스에 제한함으로써 달성할 수 있는 체계적인 접근 방식을 제공합니다.

- **Performance Highlights**: 연구 결과는 주로 개념적이지만, 생성 모델에서 hallucinations(환각 현상)의 문제를 접근하기 위한 원칙적인 접근의 첫 단계를 나타냅니다. 이 연구는 비할루시네이션 생성 모델의 학습 가능성을 다양한 학습 패러다임에서 분석하며, 데이터의 진실성 여부와 무관하게 비할루시네이션 학습 가능성을 보장하기 위한 구조적 가정들을 구별합니다.



### Equitable Federated Learning with Activation Clustering (https://arxiv.org/abs/2410.19207)
Comments:
          28 pages

- **What's New**: 본 논문에서는 Federated Learning (FL)에서 고객의 기술적, 문화적 및 기타 편향성을 고려한 새로운 클러스터링 기반의 공정성(fairness) 있는 프레임워크를 제안합니다. 이는 고객 간의 유사성을 기반으로 고객을 분류하여 알고리즘의 편향을 완화하는데 기여합니다.

- **Technical Details**: 제안된 방법은 activation vectors를 사용하여 유사성 행렬을 구성하며, 클러스터 기반 고객 가중 부여 메커니즘을 통해 각 클러스터의 중요도를 균등하게 배분합니다. 이로 인해 O(1/√K) 수렴 속도로 ϵ-정상 상태 솔루션에 도달할 수 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 MNIST, CIFAR-10, CIFAR-100 및 FEMNIST와 같은 인기 있는 데이터셋에서 기존 기준 알고리즘과 비교하여 높은 정확성을 제공하고, 고객 간의 불일치를 최소화함으로써 특정 그룹에 대한 알고리즘 편향을 완화한다는 결과를 보였습니다.



### An Inverse Modeling Constrained Multi-Objective Evolutionary Algorithm Based on Decomposition (https://arxiv.org/abs/2410.19203)
Comments:
          6 pages, 1 figure, 1 algorithm, and 2 tables

- **What's New**: 이번 논문에서는 제약이 있는 실제 최적화 문제를 해결하기 위해 분해 기반의 역 모델링 제약 다목적 진화 알고리즘(IM-C-MOEA/D)을 소개합니다. 이 알고리즘은 다목적 진화 알고리즘(MOEA)의 최신 발전을 바탕으로 하여, 제약 조건을 가진 문제 도메인에 역 모델을 적용하는 데의 격차를 극복하려고 합니다.

- **Technical Details**: 제안된 IM-C-MOEA/D는 다목적 최적화 문제를 해결하기 위해 제약 처리 기법을 통합합니다. 초기화, 하위 집단 생성, 역 모델, 제약 처리, 분해 기반의 글로벌 대체를 포함한 여러 구성 요소를 가지고 있습니다. 각 하위 집단은 목표 공간에서 결정 공간으로의 매핑을 수행하는 역 모델을 가집니다.

- **Performance Highlights**: 실험 결과, IM-C-MOEA/D는 다양한 실제 세계 문제(RWMOP1-35)에 대한 실험에서 기존의 최신 제약 다목적 진화 알고리즘(CMOEAs)에 비해 우수한 성능을 보였습니다. 이는 알고리즘의 강건성과 실제 최적화 시나리오에서의 적용 가능성을 강조합니다.



### Enriching GNNs with Text Contextual Representations for Detecting Disinformation Campaigns on Social Media (https://arxiv.org/abs/2410.19193)
Comments:
          Work in progress

- **What's New**: 본 연구는 Graph Neural Networks(GNNs)에 텍스트 정보를 통합하여 가짜 뉴스 탐지의 성능을 개선하는 방법을 조사합니다. 최근 Transformer 기반 언어 모델의 진전을 활용하여, 정적 표현(static representations)과 비교해 9.3%의 성능 향상을 나타냈습니다.

- **Technical Details**: 연구에서는 사용자의 프로필 및 상호 작용에서 얻은 텍스트 정보를 GNN의 노드에 통합하여, 정보 전파 네트워크에서 가짜 뉴스를 분류하는 모델을 설계하였습니다. GNNs는 메시지 전송(message passing)을 통해 그래프 내 의존 관계를 포착하며, 이 연구에서는 Graph Attention Networks(GATs)를 사용하여 노드 임베딩(node embeddings)을 생성합니다.

- **Performance Highlights**: 텍스트 표현을 GNN에 통합한 결과, 정적 표현에 비해 9.3%의 성능 향상과 텍스트가 없는 GNN에 비해 33.8%의 향상을 달성했습니다. 그러나 노이즈가 포함된 데이터 증강 방법은 성능 저하와 불안정을 초래하였습니다.



### No Argument Left Behind: Overlapping Chunks for Faster Processing of Arbitrarily Long Legal Texts (https://arxiv.org/abs/2410.19184)
Comments:
          To appear at 15th STIL @ BRACIS'24

- **What's New**: 브라질의 사법 시스템이 수백만 건의 사건을 처리하는 데 어려움을 겪고 있는 가운데, 법률 텍스트를 효과적으로 분석하기 위한 효율적인 방법이 필요해졌습니다. 이를 위해 uBERT라는 하이브리드 모델이 개발되었습니다. uBERT는 Transformer와 Recurrent Neural Network 아키텍처를 결합하여 긴 법률 텍스트를 처리할 수 있습니다.

- **Technical Details**: uBERT 모델은 입력된 전체 텍스트를 길이에 관계없이 처리할 수 있으며, 적절한 계산 오버헤드를 유지합니다. uBERT는 BERT+LSTM 및 ULMFiT와의 비교 실험을 통해, 특히 오버랩(Overlapping) 입력을 사용할 때 우수한 성능을 보이는 것으로 확인되었습니다. 또한 ULMFiT는 긴 문서를 처리하는 데 더 뛰어난 성능을 보였지만, uBERT보다 4배 느리다는 결과가 나왔습니다.

- **Performance Highlights**: uBERT는 BERT+LSTM을 약간 초과하는 성능을 보이며 긴 법률 문서를 처리하는 데에 있어 빠른 속도를 자랑합니다. 실험 결과, 주어진 데이터 세트에서 법률적인 판단 예측 작업에서 유의미한 성능 개선을 보여주었습니다.



### MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark (https://arxiv.org/abs/2410.19168)
Comments:
          Project Website: this https URL

- **What's New**: MMAU는 다중 모드 오디오 이해(audio understanding) 모델을 평가하기 위한 혁신적인 벤치마크로, 전문 지식과 복잡한 추론 능력을 요구하는 작업을 포함하고 있습니다. 이 벤치마크는 10,000개의 오디오 클립과 관련된 자연어 질문 및 답변을 포함하며, 27개의 독특한 기술을 요구하는 제작물들로 구성되어 있습니다. 이는 기존 벤치마크들과는 차별화되는 점으로, 전문가들이 수행하는 작업과 유사한 과제를 모델에게 도전하게 합니다. 

- **Technical Details**: MMAU는 미각(adaptive) 오디오 인식(Understanding) 및 추론(Reasoning)을 평가하기 위해 설계되었습니다. 이 벤치마크는 오디오 클립에 초점을 맞추고 있으며 정보 추출(information extraction) 및 추론 질문을 포함하여 모델들이 27개의 고유 기술을 시연하도록 요구합니다. 다중 음성 역할 매핑(multi-speaker role mapping), 감정 변화 감지(emotional shift detection), 및 시간적 음향 이벤트 분석(temporal acoustic event analysis)과 같은 고급 추론이 요구되는 작업이 포함됩니다. 이를 통해 오디오 콘텐츠와 텍스트를 공동으로 처리하고, 적절한 지식을 회상하며, 복잡한 추론을 통해 문제를 해결해야 합니다.

- **Performance Highlights**: 현재 18개의 오픈 소스 및 상용 (Large) Audio-Language 모델들이 MMAU에서 평가되었으며, Gemini Pro v1.5는 52.97%의 정확도, Qwen2-Audio는 52.50%에 불과하여 많은 향상 여지가 있음을 보여줍니다. 이러한 결과는 현재 오디오 이해 모델들이 인간이 쉽게 수행하는 과제를 해결하는 데 어려움을 겪고 있다는 것을 시사합니다.



### Adversarial Attacks on Large Language Models Using Regularized Relaxation (https://arxiv.org/abs/2410.19160)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문에서는 기존의 적대적 공격 기법이 가진 한계를 극복하기 위해, 정규화된 기울기를 활용한 새로운 적대적 공격 기법을 제안합니다. 이 방법은 연속적인 최적화 (continuous optimization) 기법을 사용하며, LLMs에서 공격 성공률을 크게 향상시킵니다.

- **Technical Details**: 제안된 기법은 연속 임베딩 공간 (continuous embedding space)에서 relaxed token embeddings를 최적화하고, 이를 통해 유효한 디스크리트 토큰 (discrete tokens)을 생성할 수 있습니다. 기존의 greedy coordinate gradient-based 방식에 비해, 두 배 이상의 속도로 작동하며, 다섯 개의 최신 LLMs에서 효과성을 입증했습니다.

- **Performance Highlights**: 새로운 공격 기법은 공격 성공률이 높으며, 기존의 최첨단 공격 기법들에 비해 효율성과 효과성에서 우수한 성능을 보여줍니다. 그 결과, 저자들은 제안하는 기법이 특정 유사한 결과를 생성하는 데 필요한 계산 비용을 최소화할 수 있음을 강조합니다.



### Lived Experience Not Found: LLMs Struggle to Align with Experts on Addressing Adverse Drug Reactions from Psychiatric Medication Us (https://arxiv.org/abs/2410.19155)
Comments:
          27 pages, 8 figures, 15 tables

- **What's New**: 이번 연구에서는 정신과 약물과 관련된 부작용(Adverse Drug Reactions, ADR)을 탐지하고, LLM(대형 언어 모델)의 성능을 평가하기 위한 Psych-ADR 벤치마크 및 ADRA(Adverse Drug Reaction Response Assessment) 프레임워크를 소개하고 있습니다. 기존 연구와 다른 점은 정신과 약물의 ADR 탐지 및 대응 전략에 LLM의 효용성을 체계적으로 평가했다는 것입니다.

- **Technical Details**: 연구에서는 Reddit에서 수집한 데이터를 바탕으로 LLM이 ADR을 탐지하고, 전문가와의 전략적 정합성을 평가했습니다. Psych-ADR 벤치마크는 239개의 Reddit 포스트와 전문적인 응답을 포함하고 있으며, LLM의 응답을 네 가지 축으로 평가합니다: (a) 텍스트 가독성, (b) 감정 및 어조 표현, (c) 해로운 결과 저감 전략의 정합성, (d) 전략의 실행 가능성입니다.

- **Performance Highlights**: 연구 결과, LLM은 ADR 탐지 및 분류에 어려움을 겪고 있으며, 전문가와의 일치는 70.86%에 불과하고, 평균적으로 12.32% 덜 실용적인 조언을 제공합니다. 이러한 결과는 의료 질환 및 약물 간의 상호작용을 다루는 과제에서 LLM의 성능을 평가하는 데 중요한 벤치마크가 될 것입니다.



### Visual Text Matters: Improving Text-KVQA with Visual Text Entity Knowledge-aware Large Multimodal Assistan (https://arxiv.org/abs/2410.19144)
Comments:
          Accepted to EMNLP (Main) 2024

- **What's New**: 이 연구는 최신 대형 다중모달 모델(large multimodal models, LMMs)의 발전을 기반으로 텍스트 기반 시각 질문 응답(knowledge-aware text-based visual question answering, Text-KVQA)을 재조명하였습니다. 주요 기여는 VisTEL이라는 시각 텍스트 엔티티 링크 모듈과 KaLMA라는 지식 인식 대형 다중모달 어시스턴트를 제안하여, 보다 정확한 답변을 제공하는 것입니다.

- **Technical Details**: VisTEL은 이미지 내 시각 텍스트 엔티티를 인식하고 이를 지식 기반에 연결하는 방법론으로, 최신 OCR 기반 텍스트 인식 엔진과 LMM의 기능을 결합하여 텍스트와 시각적 맥락을 동시에 활용하여 링크를 수행합니다. KaLMA는 이러한 정보로 LMM을 보강하여 시각 질문에 대한 응답 정확도를 높입니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, 제안한 방법론은 Text-KVQA에 대하여 기존 최상위 접근 방식에 비해 23.3%의 성능 개선을 이루어냈으며, 장면, 책 표지 및 영화 포스터 데이터셋의 각각 18.2%, 19.6%, 32.2% 개선을 보여주었습니다. 이로써 새로운 최첨단 결과를 달성하였습니다.



### Research on Key Technologies for Cross-Cloud Federated Training of Large Language Models (https://arxiv.org/abs/2410.19130)
- **What's New**: 본 연구에서는 크로스 클라우드 연합 훈련(Cross-cloud Federated Training)의 개념을 소개하고, 여러 클라우드의 자원을 활용하여 대규모 모델 훈련의 자원 병목 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: 크로스 클라우드 연합 훈련은 데이터 분할(Data Partitioning)과 배포(Distribution), 통신 최적화(Communication Optimization), 모델 집계(Model Aggregation) 알고리즘, 이종 클라우드 플랫폼의 호환성(Compatibility) 등의 핵심 기술을 포함합니다. 또한 데이터 보안(Data Security) 및 개인 정보 보호(Privacy Protection) 전략도 다루며, 데이터 암호화(Data Encryption) 및 차별적 프라이버시(Differential Privacy) 기법의 적용을 중점적으로 분석합니다.

- **Performance Highlights**: 실험 검증을 통해 제안된 기술 프레임워크는 훈련 효율성을 높이고 데이터 보안을 보장하며 훈련 비용을 감소시키는 것을 입증하였습니다. 이는 크로스 클라우드 연합 훈련의 광범위한 응용 가능성을 강조합니다.



### Bio2Token: All-atom tokenization of any biomolecular structure with Mamba (https://arxiv.org/abs/2410.19110)
- **What's New**: 본 논문에서는 고해상도의 3D 분자 구조를 효율적으로 인코딩하고 표현하기 위한 새로운 quantized auto-encoder를 개발했습니다. 이는 전체 단백질, RNA 및 소분자 구조의 원자 수준 tokenization을 학습할 수 있습니다.

- **Technical Details**: 제안된 모델은 Mamba state space 모델 아키텍처를 사용하며, 이는 훈련 데이터, 매개변수 및 컴퓨팅 자원의 일부만으로 경쟁력 있는 정확도를 달성할 수 있도록 합니다. 정확도는 1 Angstrom 이하에서 회복됩니다.

- **Performance Highlights**: 이 모델은 거의 100,000 원자까지 확장 가능하며, bio2token의 학습된 구조 토큰은 향후 모든 원자 모델에 대한 입력으로 활용될 수 있습니다.



### Conditional diffusions for neural posterior estimation (https://arxiv.org/abs/2410.19105)
- **What's New**: 이번 연구에서는 Neural Posterior Estimation (NPE)을 위한 새로운 접근 방식으로 conditional diffusion 모델을 제안합니다. 기존의 normalizing flow에 비해 안정성, 정확도, 훈련 속도가 개선된 성능을 보여주었습니다.

- **Technical Details**: NPE는 Bayes inference를 위한 시뮬레이션 기반 방법론으로, normalizing flows에 의존하는 기존 방법의 한계를 극복하기 위해 conditional diffusion 모델을 활용합니다. 이 모델은 다양한 벤치마크 문제에서 기존 모델보다 우수한 결과를 도출하였습니다. 또한, 다양한 encoder 또는 'summary network' 아키텍처에서도 일관된 성능을 보였습니다.

- **Performance Highlights**: 시장에 존재하는 다른 NPE 기법과 비교했을 때, conditional diffusion 모델은 훈련 안정성이 높고, 더 나은 정확도를 제공하며, 훈련 시간도 단축되었습니다. 이는 과학적 분석에 중요한 후행 분포의 특성을 잘 포착할 수 있게 해줍니다.



### VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks (https://arxiv.org/abs/2410.19100)
- **What's New**: 이번 논문에서는 VideoWebArena (VideoWA)라는 새로운 벤치마크를 소개하며, 이는 장기 맥락(long-context) 비디오 이해를 위한 멀티모달 에이전트의 능력을 평가하는 데 중점을 두고 있습니다. 이 벤치마크는 수작업으로 제작된 비디오 튜토리얼에 기반한 2,021개의 에이전트 작업으로 구성되어 있으며, 총 4시간에 달하는 콘텐츠를 포함하고 있습니다.

- **Technical Details**: VideoWebArena는 기술 유지를 평가하는 1,621개의 작업과 정보 유지를 평가하는 400개의 작업으로 나뉘어 있습니다. 기술 유지 작업은 에이전트가 주어진 인간 시연을 사용하여 작업을 효율적으로 수행할 수 있는지를 평가하며, 정보 유지 작업은 에이전트가 비디오에서 작업 수행에 관련된 정보를 검색할 수 있는지를 평가합니다. 최신 모델인 GPT-4o와 Gemini 1.5 Pro가 이 벤치마크에서 평가되었습니다.

- **Performance Highlights**: 본 연구에서 최고의 모델은 정보 유지 작업에서 13.3%의 성공률을, 기술 유지 작업에서는 45.8%의 성공률을 기록했습니다. 이는 각각 인간 성과의 73.9% 및 79.3%에 비해 훨씬 낮은 수치입니다. 또한 URL에서 제공되는 현 시스템의 코드 및 문서에 대한 접근성도 강조되었습니다.



### A Counterexample in Cross-Correlation Template Matching (https://arxiv.org/abs/2410.19085)
- **What's New**: 이 논문은 신호 및 이미지 처리에서 샘플링(sampling) 및 양자화(quantization)의 이론적 이해의 부족함을 다루고 있습니다. 특히, 노이즈가 있는 데이터 시퀀스의 정합 및 세분화(segmentation)에 관한 문제를 제시하며, 전통적인 교차 상관(cross-correlation) 기법의 한계를 보여주고 새로운 정합 기법을 제안합니다.

- **Technical Details**: 하나의 차원적 공간 제한(piecewise constant) 함수의 샘플링과 관련된 이론적 문제를 다루고 있습니다. 논문에서는 노이즈가 있는 두 세트의 관측치를 기반으로 차이 시퀀스(difference sequences), 임계값(thresholding), 동적 프로그래밍(dynamic programming) 기법을 사용하여 데이터 시퀀스를 정합하고 세분화하는 방법을 제시합니다. 이 기술들이 노이즈 많은 데이터 시퀀스를 align하는 데 어떻게 사용될 수 있는지를 보여 줍니다.

- **Performance Highlights**: 기존의 cross-correlation 기법은 노이즈가 있는 샘플에서 잘 작동하지 않지만, 제안된 방법은 특정 조건 하에 최적의 정합과 세분화를 제공할 수 있습니다. 저자들은 노이즈의 가정 아래에서도 근본 함수의 추정이 가능하다는 점을 이론적으로 입증하였습니다.



### From a Tiny Slip to a Giant Leap: An LLM-Based Simulation for Fake News Evolution (https://arxiv.org/abs/2410.19064)
- **What's New**: 본 연구는 진실 뉴스가 어떻게 점진적으로 가짜 뉴스로 변하는지를 이해하고 이에 대한 모델링 및 시뮬레이션을 제안합니다. 이를 위해 Fake News evolUtion Simulation framEwork (FUSE)를 개발하고 대규모 언어 모델(LLM)을 활용한 다양한 에이전트를 정의하여 사회적 상호작용을 시뮬레이션합니다.

- **Technical Details**: FUSE 프레임워크는 네 가지 유형의 에이전트를 정의합니다: 정보를 전파하는 'spreaders', 의견을 제공하는 'commentators', 정보를 검증하는 'verifiers', 수동적으로 관찰하는 'bystanders'. 에이전트는 신뢰 기반의 상호작용을 통해 진실 뉴스가 가짜 뉴스로 발전하는 과정을 모델링합니다. FUSE-EVAL 평가 프레임워크를 통해 뉴스의 진실에서의 편차를 정량적으로 측정합니다.

- **Performance Highlights**: FUSE의 시뮬레이션 실험 결과는 진실 뉴스가 어떻게 가짜 뉴스로 변형되는지를 이해하는 데 기여하며, 특히 클러스터링이 높은 네트워크에서 진실 뉴스의 발전이 더 빠르게 진행되는 것으로 나타났습니다. 또한, 정치 관련 가짜 뉴스는 그 외의 주제보다 더 빠르게 확산되는 경향을 보였습니다. 이러한 결과는 가짜 뉴스의 초기에 개입하여 상황을 저지하는 중요성을 강조합니다.



### Large Language Models for Financial Aid in Financial Time-series Forecasting (https://arxiv.org/abs/2410.19025)
Comments:
          GitHub link this https URL

- **What's New**: 이 논문은 금융 지원(Financial Aid) 데이터의 시계열 예측을 위해 사전 훈련된 대규모 언어 모델(LLMs)을 사용하는 새로운 접근 방식을 제안합니다. 기존의 전통적인 방법에 비해 LLM 기반 모델이 효과적으로 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 다양한 재무 데이터를 포함하는 여덟 가지 데이터 세트를 수집하고, 입력 데이터는 최근 L일의 과거 데이터를 사용하여 다음 τmax 일의 출력을 예측합니다. 주요 모델들로는 DLinear, iTransformer, TimeLLM과 같은 최신 LLM 기반 모델들이 있으며, 이 모델들은 메모리 효율성과 실행 시간의 균형을 맞추는 데 집중하고 있습니다.

- **Performance Highlights**: 논문에서는 LLM을 활용하여 전통적 시계열 모델을 초월하는 결과를 보여주었습니다. 특히 재무 지원 데이터에 대한 예측 정확도에서 MSE(Mean Squared Error)와 MAE(Mean Absolute Error) 측정값이 대폭 개선되었습니다.



### Dual Space Training for GANs: A Pathway to Efficient and Creative Generative Models (https://arxiv.org/abs/2410.19009)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문은 GAN(Generative Adversarial Networks)의 훈련 과정에서 새로운 최적화 접근 방안을 제안하며, 인버터블 매핑(invertible mappings)과 오토인코더(autoencoders)를 사용하여 초기 데이터의 이중 공간(dual space)에서 작동하도록 합니다. 이를 통해 GAN의 훈련을 더 효율적이고 자원 절약적으로 만들 수 있도록 합니다.

- **Technical Details**: 이중 공간의 개념을 도입하여 데이터의 중요한 특징을 캡슐화한 인코딩된 표현으로 GAN을 훈련합니다. 오토인코더가 데이터의 특성을 보존하면서 불필요한 정보를 제거하고 이를 통해 GAN이 더 나은 훈련과 생성 결과를 나타낼 수 있습니다. 인코더와 디코더는 각각 매핑과 역매핑 기능을 수행합니다.

- **Performance Highlights**: 제안된 방법은 개 이미지 생성 실험에서 기존 GAN 훈련 방법보다 약 100배 빠른 훈련 시간을 기록했습니다. 또한, 훈련 데이터에서 제외된 특정 개 품종을 생성할 수 있는 능력을 나타내어, 보다 근본적인 표현 방식으로부터 학습하여 초기 데이터 세트를 초과하는 출력 생성을 가능하게 하였습니다.



### Teach Multimodal LLMs to Comprehend Electrocardiographic Images (https://arxiv.org/abs/2410.19008)
- **What's New**: 이번 연구에서는 ECG(심전도) 이미지 해석을 위한 새로운 Datasets인 ECGInstruct를 소개합니다. 이 데이터셋은 100만 개 이상의 ECG 이미지-텍스트 샘플로 구성되어 있으며, 다양한 ECG 관련 작업을 포함하고 있습니다.

- **Technical Details**: ECGInstruct는 현실적 이미지 합성과 임상 전문가의 통찰을 반영한 다양한 ECG 관련 작업을 특징으로 하며, PULSE라는 MLLM(Multimodal Large Language Model) 모델을 개발하였습니다. 또한, ECGBench라는 새로운 평가 벤치마크를 설정하였고, 이를 통해 ECG 해석 성능을 측정합니다.

- **Performance Highlights**: PULSE는 다양한 벤치마크에서 평균 15%에서 30%의 정확도 개선을 달성하며 기존의 일반 MLLM들을 능가하여 심전도 해석의 새로운 최첨단 성능을 설정하였습니다.



### Whither Bias Goes, I Will Go: An Integrative, Systematic Review of Algorithmic Bias Mitigation (https://arxiv.org/abs/2410.19003)
Comments:
          forthcoming in Journal of Applied Psychology

- **What's New**: 본 논문은 머신 러닝(ML) 모델의 인사 평가 및 선발에서의 사용 증가에 따른 편견과 불평등 문제를 다루고 있습니다. 저자들은 ML 평가에서 공정성을 정의하고 알고리즘적 편견 완화 방법을 통합하기 위해 새로운 네 단계 모델을 제안하고 있습니다.

- **Technical Details**: 제안된 네 단계 모델은 1) 교육 데이터 생성(Generating the training data), 2) 모델 교육(Training the model), 3) 모델 테스트(Testing the model), 4) 모델 배포(Deploying the model)로 구성됩니다. 각 단계에서 발생할 수 있는 편견 및 불공정성의 잠재적 출처를 설명하고, 알고리즘적 편견의 정의 및 법적 요구 사항을 체계적으로 검토합니다.

- **Performance Highlights**: 이 연구는 알고리즘적 편견이 발생할 수 있는 메커니즘을 밝히고, 효과적이고 법적으로 허용되는 편견 완화 방법을 식별하여 향후 조직 연구와 컴퓨터 과학, 데이터 과학 간의 협업을 통해 해결해야 할 지식의 공백을 드러냅니다.



### TRIAGE: Ethical Benchmarking of AI Models Through Mass Casualty Simulations (https://arxiv.org/abs/2410.18991)
- **What's New**: TRIAGE Benchmark는 LLM(대형 언어 모델)의 윤리적 의사결정 능력을 테스트하는 새로운 기계 윤리(Machine Ethics, ME) 벤치마크입니다. 의료 전문가들이 설계한 실제 윤리적 딜레마를 이용하여 모델의 성능을 평가하며, 주로 무작위 추측보다 높은 성능을 보여 LLM이 외상 사태에서 의사결정을 지원할 수 있음을 시사합니다.

- **Technical Details**: TRIAGE Benchmark는 START와 jumpSTART 의료 분류 모델을 기반으로, 환자 상황 87개를 포함하여 네 개의 분류 범주에 따라 환자를 분류합니다. 각 모델의 성능은 다양한 프롬프트와 문맥에 따라 평가되었습니다. 프롬프트 유형에는 Deontology(의무론) 및 Utilitarianism(공리주의) 등이 포함되었습니다.

- **Performance Highlights**: 모델들은 TRIAGE 벤치마크에서 랜덤 추측을 일관되게 초과하는 성능을 보였으며, 대체로 중립적인 문구가 가장 우수한 성능을 기록했습니다. 고급 모델이 일반적으로 더 좋은 성능을 보였으나, 특정 문맥에서는 차이가 나타났습니다. 오픈 소스 모델은 주로 도덕적으로 심각한 오류가 발생하는 경향이 있었습니다.



### rECGnition_v1.0: Arrhythmia detection using cardiologist-inspired multi-modal architecture incorporating demographic attributes in ECG (https://arxiv.org/abs/2410.18985)
- **What's New**: 이 논문은 ECG 분석의 새로운 다중 모달(multi-modal) 방법론인 rECGnition_v1.0을 제안하며, 이는 환자의 특성과 ECG 형태 변화 간의 연관성을 명확히 이해하는 데 중점을 둡니다.

- **Technical Details**: XGBoost 모델을 사용하여 UCI Arrhythmia 데이터 세트를 분석하였으며, 환자의 특성과 ECG 형태 변화 간의 연결성을 탐구하였습니다. 모델은 87.75%의 신뢰도로 성별을 분류하였고, Squeeze and Excitation 기반의 Patient characteristic Encoding Network (SEPcEnet)를 도입하여 환자의 인구 통계학적 특성을 고려하였습니다.

- **Performance Highlights**: 기존 여러 알고리즘과 비교하여 MITDB에서 0.986의 F1-score를 달성하였고, LBBB, RBBB, Premature ventricular contraction beat, Atrial premature beat, Paced beat에 대해 ~0.99에 가까운 예측 점수를 기록하였습니다. 이 방법론은 INCARTDB, EDB 및 MITDB의 다양한 클래스 그룹에서 전이 학습(transfer learning)을 통해 검증되었습니다.



### Stick-breaking Attention (https://arxiv.org/abs/2410.17980)
- **What's New**: 본 논문에서는 기존의 softmax 기반 self-attention 메커니즘을 대체할 수 있는 stick-breaking attention 메커니즘을 제안합니다. 이 접근법은 재귀적이지 않으며 최근성 편향을 자연스럽게 통합합니다.

- **Technical Details**: Stick-breaking attention은 각 토큰에 대해 break point $eta_{i,j}$를 계산하여 현재 토큰에 allocation하는 비율을 결정합니다. 이 과정을 반복하여 attention 가중치의 시퀀스를 생성합니다. 구현 시, numerically stable stick-breaking attention을 적용하고 Flash Attention을 조정하여 이 메커니즘을 수용하도록 합니다.

- **Performance Highlights**: Stick-breaking attention은 길이 일반화(length generalisation)와 여러 다운스트림 작업에서 기존 softmax+RoPE 시스템과 경쟁력 있는 성과를 보였습니다. 특히, $2^{11}$의 context window로 학습된 모델이 $2^{14}$에서 덜 혼란스러운 퍼플렉서티(perplexity) 향상을 보여주었습니다.



### Scaling Law with Learning Rate Annealing (https://arxiv.org/abs/2408.11029)
Comments:
          Add more experiments to consolidate our scaling laws. 29 pages, 29 figures

- **What's New**: 본 연구에서는 신경 언어 모델의 cross-entropy (크로스 엔트로피) 손실 곡선이 학습률 (Learning Rate, LR) 퍼지 동안 스케일링 법칙에 기초한다는 점을 발견함. 제안된 공식은 손실 곡선을 각 학습 단계에서 나타낼 수 있도록 함.

- **Technical Details**: 제안된 손실 모델은 다음과 같은 두 가지 주요 요소를 포함함: 1) 전방 영역 (forward area) S1과 2) LR 퍼지 영역 (LR annealing area) S2. 이 식은 LR 스케쥴러와 관계없이 어떤 학습률 스케쥴에서든 훈련 단계에서 손실을 예측할 수 있게 해줌.

- **Performance Highlights**: 새로운 접근법은 치킨밀라 스케일링 법칙 (chinchilla scaling law) 대비 1% 미만의 계산 비용으로 정확한 손실 예측을 가능하게 하며, 대규모 언어 모델 개발에서 스케일링 법칙 피팅과 예측을 대폭 민주화함.




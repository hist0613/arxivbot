New uploads on arXiv(cs.CL)

### Fingerspelling within Sign Language Translation (https://arxiv.org/abs/2408.07065)
- **What's New**: This paper explores the impact of character-level tokenization and fingerspelling recognition data integration on American Sign Language (ASL) to English translation models, specifically focusing on the ability to understand and translate fingerspelling within sentences.

- **Technical Details**: The study uses FLEURS-ASL dataset, manually annotated for fingerspelling instances, to evaluate two approaches: 1) using ByT5, a model family with character-level tokenization, and 2) incorporating fingerspelling recognition data (FSboard) into training. The paper measures performance using BLEURT scores and character-error rate (CER) for fingerspelled phrases.

- **Performance Highlights**: Using ByT5 with character-level tokenization significantly improves overall translation quality (BLEURT score), particularly in sentences containing fingerspelling. However, integrating fingerspelling recognition data into training yielded mixed or negative results. The study suggests character-level tokenization as a promising approach for improving fingerspelling understanding in sign language translation models.



### LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs (https://arxiv.org/abs/2408.07055)
- **What's New**: 현재의 대규모 언어 모델(LLM)은 10만 토큰까지 입력을 처리할 수 있지만, 2,000단어를 넘는 출력을 생성하는 데 어려움을 겪고 있습니다. 본 연구에서는 제어된 실험을 통해 모델의 효과적인 생성 길이가 지도 학습(SFT) 중에 관찰한 샘플에 의해 본질적으로 제한된다는 것을 발견했습니다. 즉, 모델의 출력 제한은 기존 SFT 데이터셋에서 긴 출력 예제가 부족하기 때문입니다. 이 문제를 해결하기 위해, 본 연구에서는 초장문 생성 작업을 하위 작업으로 분해하는 에이전트 기반 파이프라인인 AgentWrite를 도입하여 기존의 LLM이 2만 단어를 넘는 일관성 있는 출력을 생성할 수 있도록 합니다. AgentWrite를 활용하여 2k에서 32k 단어에 이르는 출력 길이를 가진 6,000개의 SFT 데이터가 포함된 LongWriter-6k 데이터셋을 구축했습니다. 이 데이터셋을 모델 훈련에 통합함으로써 기존 모델의 출력 길이를 1만 단어 이상으로 확장하는 동시에 출력 품질을 유지하는 데 성공했습니다. 또한, 초장문 생성 기능을 평가하기 위한 포괄적인 벤치마크인 LongBench-Write를 개발했습니다. DPO를 통해 개선된 90억 매개변수 모델은 이 벤치마크에서 최첨단 성능을 달성하여 훨씬 더 큰 독점 모델을 능가했습니다. 전반적으로 본 연구는 기존의 장문맥 LLM이 더 긴 출력 창을 위한 잠재력을 이미 가지고 있음을 보여줍니다. 모델 정렬 중에 확장된 출력이 있는 데이터만 있으면 이 기능을 활용할 수 있습니다. 본 연구의 코드와 모델은 다음 주소에서 확인할 수 있습니다: (링크 주소).



### Generative AI for automatic topic labelling (https://arxiv.org/abs/2408.07003)
Comments:
          10 pages, 1 figure

- **What's New**: 본 논문은 주제 모델링 (Topic Modeling) 결과 해석을 위한 라벨링에 대규모 언어 모델 (LLM)의 효용성을 평가합니다. 구체적으로 Flan, GPT-4o, GPT-4 mini 모델을 활용하여, 스위스 생물학 교수 465명이 2008년부터 2020년까지 발표한 과학 논문 데이터셋 (34,797개)에서 추출된 주제들을 라벨링합니다.



### The advantages of context specific language models: the case of the Erasmian Language Mod (https://arxiv.org/abs/2408.06931)
Comments:
          12 pages, 3 figures, 1 table

- **What's New**: 본 논문은 기존 대규모 언어 모델(LLM)의 한계를 극복하기 위해 Erasmus 대학교 로테르담(Erasmus University Rotterdam)에 맞춰 특화된 소규모 언어 모델인 Erasmian Language Model (ELM)을 제시합니다. ELM은 9억 개의 파라미터를 가지며, Erasmus 대학교의 데이터로 사전 학습 및 미세 조정되었습니다. 특정 주제에 대한 전문성을 갖춘 소규모 모델을 활용함으로써 계산 자원 및 에너지 소비, 개인 정보 보호와 같은 문제를 완화하고자 합니다.



### Diagnosis extraction from unstructured Dutch echocardiogram reports using span- and document-level characteristic classification (https://arxiv.org/abs/2408.06930)
Comments:
          28 pages, 5 figures

- **What's New**: 본 연구는 네덜란드의 대규모 대학 병원인 UMCU에서 수집된 115,692개의 비정형 심장초음파 보고서를 사용하여 자동화된 스팬 및 문서 수준 진단 추출의 가능성을 조사했습니다. 이 연구는 심장초음파 보고서에서 11가지 주요 심장 특징의 발생 및 심각도를 자동으로 분류할 수 있는 새로운 딥러닝 기반 모델을 제시합니다.



### Evaluating Cultural Adaptability of a Large Language Model via Simulation of Synthetic Personas (https://arxiv.org/abs/2408.06929)
Comments:
          18 pages, 8 figures, Published as a conference paper at COLM 2024

- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 문화적 적응력(Cultural Adaptability)을 측정하기 위해 심리 실험 스타일의 설문지를 사용하여 LLM이 다양한 국적의 인간 프로필을 시뮬레이션하도록 했습니다. 특히, GPT-3.5를 사용하여 15개국의 7,286명 참가자의 설득력 있는 뉴스 기사에 대한 반응을 재현하고, 동일한 인구 통계적 특성을 가진 실제 참가자 데이터 세트와 비교했습니다. 이를 통해 LLM이 문화적 배경을 이해하는 데 어려움을 겪고 있으며, 국적 정보 제공 방법에 따라 모델의 성능이 크게 달라진다는 것을 발견했습니다.



### Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives (https://arxiv.org/abs/2408.06904)
Comments:
          Work in Progress

- **What's New**: This paper introduces the **Re-TASK framework**, a novel model that systematically analyzes LLM failures in domain-specific tasks and enhances their performance. Re-TASK utilizes principles from **Bloom's Taxonomy** and **Knowledge Space Theory** to revisit tasks from **capability, skill, and knowledge perspectives**.



### Leveraging Language Models for Emotion and Behavior Analysis in Education (https://arxiv.org/abs/2408.06874)
Comments:
          8 pages

- **What's New**: 본 논문에서는 학생의 감정과 행동을 분석하기 위해 대규모 언어 모델(LLM)과 프롬프트 엔지니어링을 활용하는 새로운 방법을 제안합니다. 기존 방법은 개인 정보 보호 문제와 확장성 문제를 야기하는 침입적인 시각 및 생리적 데이터 수집에 의존하는 경우가 많습니다. 본 연구에서는 학생의 텍스트 데이터를 분석하기 위해 맞춤형 프롬프트를 활용하는 프롬프트 엔지니어링을 통해 LLM을 활용합니다. 이를 통해 비침입적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 본 연구에서는 Qwen, ChatGPT, Claude2, GPT-4를 사용하여 실험을 수행했으며, 기존 모델과 chain-of-thought(CoT) 프롬프팅과 비교했습니다. 프롬프트는 학생의 텍스트 데이터에서 감정과 참여 상태를 추론하는 데 초점을 맞추도록 설계되었습니다. 이러한 프롬프트는 LLM이 텍스트 데이터에서 감정 상태와 참여 수준을 파악하도록 유도하고, 이를 통해 교육적 감정과 행동 분석을 위한 실용적이고 효과적인 도구를 제공할 가능성이 있습니다.

- **Performance Highlights**: 본 연구에서 제안된 방법은 기존 모델보다 정확도와 맥락 이해 측면에서 모두 뛰어난 성능을 보였습니다. 또한, 본 연구는 LLM과 프롬프트 엔지니어링을 결합하여 학생들의 감정과 행동을 분석하는 데 유용하다는 점을 보여줍니다.



### LoRA$^2$ : Multi-Scale Low-Rank Approximations for Fine-Tuning Large Language Models (https://arxiv.org/abs/2408.06854)
- **What's New**: 이 논문은 LoRA(Low-Rank Adaptation)의 확장판인 LoRA2를 제안합니다. LoRA2는 두 개의 직교 평면에 LoRA를 학습시키는 방법으로, 기존 LoRA의 학습 공간을 확장하여 다양한 하위 작업에 대한 적응력을 향상시킵니다. 또한, LoRA2는 AdaLoRA에서 사용된 중요도 점수 알고리즘을 개선하여 매개변수 민감도 점수 계산을 약 98.5% 줄였습니다.



### Layerwise Recurrent Router for Mixture-of-Experts (https://arxiv.org/abs/2408.06793)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 훈련 효율성을 높이기 위한 새로운 아키텍처인 계층별 순환 라우터(RMoE)를 제안합니다. 기존의 MoE 모델은 각 계층에서 독립적으로 토큰을 라우팅하는 방식으로, 각 토큰이 어떤 전문가(expert) 네트워크에 할당될지에 대한 최적화가 제한적이었습니다. 반면, RMoE는 각 계층의 라우팅 결과를 이전 계층의 결과에 의존하도록 함으로써, 토큰-전문가 조합을 최적화하고 모델 효율성을 향상시키는 것을 목표로 합니다. 이를 위해, RMoE는 GRU(Gated Recurrent Unit)를 사용하여 라우팅 정보를 계층간에 공유하고, 이 정보를 사용하여 각 토큰에 가장 적합한 전문가를 선택합니다.

- **Technical Details**: RMoE는 각 계층의 라우터가 이전 계층의 라우팅 결과를 참고하여 토큰을 전문가 네트워크에 할당하는 방식을 채택합니다. 이를 위해 GRU(Gated Recurrent Unit)가 사용됩니다. GRU는 입력 정보를 선택적으로 사용할 수 있는 기능을 가지고 있으며, 이를 통해 각 계층은 이전 계층의 라우팅 정보를 사용하여 더 나은 토큰-전문가 매칭을 수행할 수 있습니다. RMoE는 MoE 아키텍처의 기존 방법들과 호환 가능하도록 설계되었습니다.

- **Performance Highlights**: RMoE는 다양한 모델 크기, 아키텍처, 데이터셋 및 훈련 설정에서 기존 MoE 모델보다 우수한 성능을 보여줍니다. RMoE는 계층간 정보 공유를 통해 전문가 선택의 다양성을 향상시키고, 이를 통해 모델의 전체적인 성능을 향상시킬 수 있습니다. RMoE는 MoE 모델의 효율성과 성능을 향상시키는 데 유용한 새로운 기술로 평가됩니다.



### Unlock the Power of Frozen LLMs in Knowledge Graph Completion (https://arxiv.org/abs/2408.06787)
- **What's New**: 본 논문은 지식 그래프 완성(KGC)을 위해 효과적이고 효율적인 방법으로 대규모 언어 모델(LLM)을 활용하는 새로운 방법을 제시합니다. 이는 고정된(frozen) LLM의 잠재력을 활용하여 지식 그래프의 희소성 문제를 해결하고 뛰어난 성능을 달성하며 동시에 훈련 시간과 메모리 소비량을 최소화합니다. 



### Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors (https://arxiv.org/abs/2408.06778)
- **What's New**: 이 논문은 그래프 구조와 텍스트 설명을 통합하여 지식 그래프(KG)의 링크 예측을 위한 새로운 Transformer 기반 모델을 제안합니다. 이 모델은 Fast-and-Frugal Text-Graph (FnF-TG) Transformer라고 불리며, 기존의 리소스 집약적인 텍스트 인코더에 대한 의존성을 줄이면서 효율성과 확장성을 유지합니다.

- **Technical Details**: FnF-TG Transformer는 Transformer의 자기 주의 메커니즘을 활용하여 그래프 구조와 텍스트 설명을 효과적으로 통합합니다. 또한, 관계의 텍스트 설명으로부터 관계 임베딩을 계산하여 완전히 귀납적인 학습을 가능하게 합니다. 이 모델은 텍스트 설명과 그래프 구조를 효율적으로 통합하여 텍스트 인코더에 대한 의존성을 줄여줍니다.

- **Performance Highlights**: 세 가지 까다로운 KG 링크 예측 데이터 세트에서 수행된 실험 결과, FnF-TG Transformer는 기존의 최첨단 모델을 능가하는 성능을 보였습니다. 또한, FnF-TG Transformer는 기존의 텍스트 기반 모델에 비해 훈련 시간이 크게 단축되면서도 경쟁력 있는 성능을 유지했습니다.



### Multilingual Models for Check-Worthy Social Media Posts Detection (https://arxiv.org/abs/2408.06737)
- **What's New**: 본 연구는 소셜 미디어 게시글에서 검증 가능한 사실적 주장과 유해한 주장을 탐지하기 위한 트랜스포머 기반 NLP 모델에 대한 광범위한 연구를 제시합니다. 이 연구는 데이터셋 수집, 데이터셋 전처리, 아키텍처 선택, 설정 설정, 모델 훈련(파인튜닝), 모델 테스트 및 구현을 포함한 다양한 활동을 다룹니다. 본 연구는 다양한 모델에 대한 포괄적인 분석을 포함하며, 영어와 아랍어, 불가리아어, 네덜란드어, 폴란드어, 체코어, 슬로바키아어와 같은 저자원 언어에서 동일한 모델이 소셜 미디어 게시글을 처리할 수 있는 다국어 모델에 특히 중점을 둡니다. 연구에서 얻은 결과는 최첨단 모델에 대해 검증되었으며, 비교 결과는 제안된 모델의 견고성을 보여주었습니다. 이 연구의 참신성은 유해한 게시글과 검증 가능한 사실적 주장을 포함하는 게시글을 효율적으로 동시에 탐지할 수 있는 다중 레이블 다국어 분류 모델 개발에 있습니다.



### Exploring the anatomy of articulation rate in spontaneous English speech: relationships between utterance length effects and social factors (https://arxiv.org/abs/2408.06732)
Comments:
          Proceedings of Interspeech 2024. 5 pages, 4 figures

- **What's New**: 이 연구는 영어 연설 속도의 변화를 13개의 다른 영어 말뭉치 데이터를 이용하여 분석했습니다. 이를 통해 연설 속도에 미치는 요인들을 자세히 살펴보았습니다. 특히 연설 속도에 가장 큰 영향을 미치는 요인으로는 발화 길이, 즉 문장의 길이가 밝혀졌습니다.  또한 나이와 성별 또한 연설 속도에 영향을 미치지만, 그 영향은 발화 길이에 비해 상대적으로 작다는 것을 확인했습니다.



### Latin Treebanks in Review: An Evaluation of Morphological Tagging Across Tim (https://arxiv.org/abs/2408.06675)
- **What's New**: 이 논문은 다양한 시대와 장르에 걸쳐 544개의 라틴어 텍스트를 포함하는 라틴어 UD 트리뱅크에 대한 분석을 수행하고, 이들 트리뱅크의 메타데이터를 정확히 문서화하고, UD 트리뱅크와 LASLA 트리뱅크의 주석을 조정하여 라틴어 형태소 태깅의 도메인 간 정확도를 개선합니다.



### Pragmatic inference of scalar implicature by LLMs (https://arxiv.org/abs/2408.06673)
Comments:
          This research was presented at the Association for Computational Linguistics conference, held on August 11-16

- **What's New**: 이 연구는 BERT와 GPT-2와 같은 대규모 언어 모델(LLM)이 "some"과 같은 스칼라 함축(scalar implicature)의 실용적 추론(pragmatic inference)에 어떻게 참여하는지 조사합니다. 연구는 코사인 유사도와 다음 문장/토큰 예측을 사용한 두 가지 실험을 통해 진행되었습니다.



### Amuro & Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of Large Language Models (https://arxiv.org/abs/2408.06663)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 사전 훈련(Pre-training)과 미세 조정(Fine-tuning)의 상호 작용을 탐구하여 새로운 통찰력을 제공합니다. 기존의 LLM 훈련 패러다임에서 사전 훈련은 대량의 텍스트 데이터에서 수행되고 미세 조정은 후처리 단계로 여겨졌습니다. 이 논문에서는 사전 훈련의 중간 단계에서 여러 체크포인트를 미세 조정함으로써 이러한 두 단계 간의 관계를 심층적으로 분석합니다.

- **Technical Details**: 본 연구는 대규모 언어 모델(LLM)의 사전 훈련(Pre-training)과 미세 조정(Fine-tuning)의 상호 작용을 탐구합니다. 18개의 데이터셋에 대한 실험을 통해 다음과 같은 주요 결과를 얻습니다.
1. 지속적인 사전 훈련(Continual pre-training)은 미세 조정 후에만 드러나는 잠재적인 방식으로 모델을 개선합니다.
2. 추가 미세 조정을 통해 모델이 사전 훈련 단계에서 능력을 보이지 않던 데이터셋의 성능이 크게 향상됩니다.
3. 지도 학습(Supervised fine-tuning)을 통한 모델 성능 향상에도 불구하고 이전에 알고 있던 도메인 지식이나 미세 조정 중에 보지 못한 작업을 잊어버릴 수 있습니다.
4. 지도 학습으로 미세 조정된 모델은 평가 프롬프트(Evaluation prompt)에 대한 높은 민감성을 보이지만 더 많은 사전 훈련을 통해 이러한 민감성을 완화할 수 있습니다.

- **Performance Highlights**: 본 논문은 사전 훈련과 미세 조정의 상호 작용을 통해 모델의 능력을 탐구합니다. 특히, 지속적인 사전 훈련은 미세 조정 후에만 모델의 잠재적인 개선을 드러낸다는 점을 발견했습니다. 또한 모델은 미세 조정을 통해 새로운 능력을 얻지만, 이전에 학습한 도메인 지식이나 작업을 잊어버릴 수도 있습니다. 이러한 결과는 LLM 훈련에 대한 새로운 통찰력을 제공하며, 사전 훈련 및 미세 조정 방법을 개선하는 데 기여할 수 있습니다. 더욱이 본 연구는 최종 모델뿐만 아니라 훈련 동역학(Training dynamics)을 분석하는 것이 해석 가능성(Interpretability)을 높이는 중요한 측면임을 보여줍니다.



### IFShip: A Large Vision-Language Model for Interpretable Fine-grained Ship Classification via Domain Knowledge-Enhanced Instruction Tuning (https://arxiv.org/abs/2408.06631)
- **What's New**: 본 논문에서는 해상 원격 감지 데이터에서 선박을 정확하게 분류할 뿐만 아니라 분류 과정을 자연어로 설명하는 새로운 인공지능 모델 (LVLM)인 IFShip를 제안합니다. IFShip는 딥 러닝 기반의 기존 방법과 달리, 사람과 유사한 논리적 추론 과정을 통해 선박의 종류를 판단하고, 그 결과를 자세히 설명할 수 있습니다.



### Generalized knowledge-enhanced framework for biomedical entity and relation extraction (https://arxiv.org/abs/2408.06618)
- **What's New**: 이 연구는 바이오메디컬 엔티티 및 관계 추출을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 외부 지식을 활용하여 작업에 독립적이고 재사용 가능한 바이오메디컬 엔티티 및 관계 추출을 위한 배경 지식 그래프를 구축합니다. 이 모델은 인간이 도메인 특정 주제를 학습하는 방식에서 영감을 받았습니다. 특히 인간은 먼저 기초 지식을 구축하기 위해 필드에 대한 가장 기본적이고 일반적인 지식을 습득한 다음, 이를 기반으로 다양한 전문 분야 주제로 확장합니다. 이 프레임워크는 이러한 일반적인 지식 공유 메커니즘을 사용하여 다양한 도메인 특정 바이오메디컬 텍스트에 효과적으로 전이 학습이 가능한 일반적인 신경망 지식 그래프를 구축합니다.



### A Perspective on Large Language Models, Intelligent Machines, and Knowledge Acquisition (https://arxiv.org/abs/2408.06598)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 한계를 강조하며, LLM이 인간 지능과 비교하여 추상적 개념을 이해하고 추론하는 능력에 있어서 심각한 차이가 있다고 주장합니다. 특히, GPT-4를 이용한 과학, 수학, 상식 추론 질문에 대한 분석을 통해, GPT-4가 인간의 추론을 모방할 수 있지만 실제 이해는 부족하다는 점을 보여줍니다. LLM이 인간 지식 습득과 교육에 미치는 영향을 논의합니다.



### Biomedical Event Extraction via Structure-aware Generation (https://arxiv.org/abs/2408.06583)
Comments:
          8 pages, 4 figures, 6 tables

- **What's New**: GenBEE, a generative model with a structure-aware prefix for Biomedical Event Extraction (BEE), is proposed. It utilizes knowledge distilled from large language models (LLMs) to incorporate label semantics and argument dependency relationships. GenBEE further introduces a structural prefix learning module to enhance the generation process with structural features.



### OpenEP: Open-Ended Future Event Prediction (https://arxiv.org/abs/2408.06578)
- **What's New**: This paper introduces a new task called **OpenEP** (Open-Ended Future Event Prediction), which aims to generate more flexible and diverse predictions for real-world scenarios.  This contrasts with existing approaches which typically treat event prediction as classification tasks and limit outcomes to a fixed scope (e.g., yes/no, candidate set).

- **Technical Details**: The authors built **OpenEPBench**, an open-ended future event prediction dataset, to facilitate research on OpenEP. Key aspects of the dataset include:

* **Diverse Questions**: Questions are generated from seven perspectives (location, time, event development, event outcome, event impact, event response, and other) to provide a comprehensive understanding of event evolution.
* **Flexible Outcomes**: Outcomes are collected as free-form text, allowing for semantically complete and detailed responses.

They also proposed **StkFEP**, a stakeholder-enhanced future event prediction framework. StkFEP leverages the identification of stakeholders involved in an event to:

* Extend questions to gather diverse information.
* Retrieve relevant and similar historical events to reveal potential evolutionary patterns.

- **Performance Highlights**: Experimental results indicate that accurately predicting future events in open-ended settings is challenging for existing LLMs. This suggests that the task of OpenEP presents new challenges and opportunities for future research.



### CTISum: A New Benchmark Dataset For Cyber Threat Intelligence Summarization (https://arxiv.org/abs/2408.06576)
- **What's New**: 새로운 사이버 위협 정보 (CTI) 요약 벤치마크인 CTISum이 제시되었습니다. 이는 사이버 보안 영역에서 사이버 위협에 대한 빠른 탐지와 대응에 필요한 정보를 의사 결정자에게 제공하기 위해 중요한 역할을 합니다. CTISum은 CTI 보고서에서 사실, 분석적 통찰력, 공격 프로세스 등을 요약하는 핵심 과제인 CTI 요약 (CTIS) 과제와 공격 프로세스를 이해하는 데 도움이 되는 세분화된 하위 과제인 공격 프로세스 요약 (APS) 과제를 제공합니다.

- **Technical Details**: CTISum은 여러 사이버 보안 정보 출처에서 수집된 위협 정보를 기반으로 합니다. 데이터 수집, 파싱 및 정리, 프롬프트 스키마, 정보 요약의 다단계 주석 파이프라인을 통해 데이터를 구축합니다. 이 과정에서 대규모 언어 모델 (LLMs)을 활용하여 주석을 자동화하고, 전문가 검토를 통해 품질을 보장합니다.

- **Performance Highlights**: 실험 결과는 현재 최첨단 모델들이 CTISum에 적용될 때 한계를 보여주었으며, CTI 보고서를 간결하게 요약하는 자동화 시스템 개발의 어려움을 강조합니다. 특히, 기존의 추출적 방법들은 길고 복잡한 문서에서 중요한 정보를 식별하는 데 어려움을 겪으며 불완전한 요약을 생성하는 경향이 있습니다. 추상적 방법들은 일관성 있고 중복되지 않는 요약 생성 및 환각 방지에 어려움을 겪습니다. 두 방법 모두 ROUGE와 같은 자동 평가 지표에서 낮은 점수를 기록했습니다.

- **Dataset Characteristics**: CTISum은 평균 문서 길이가 약 2,865 단어인 대규모 데이터셋입니다. 이는 기존의 딥 러닝 모델들이 처리할 수 있는 길이를 넘어섭니다. 또한 높은 문서-요약 압축 비율 (14.32 및 22.23)을 가지고 있어 시스템이 길고 복잡한 문서에서 가장 관련성이 높은 사실만을 최소한의 단어로 요약해야 하는 어려움이 있습니다.

- **Future Directions**: CTISum은 사이버 보안 영역에서 CTI 요약을 위한 새로운 벤치마크를 제공하며, 미래 연구를 위한 방향을 제시합니다. 특히, 길고 복잡한 문서에서 중요한 정보를 추출하고 일관성 있고 정확한 요약을 생성할 수 있는 새로운 요약 기술 개발이 필요합니다.



### SparkRA: A Retrieval-Augmented Knowledge Service System Based on Spark Large Language Mod (https://arxiv.org/abs/2408.06574)
- **What's New**: iFLYTEK Spark LLM을 기반으로 과학 문헌 전용 대규모 언어 모델(LLM)인 SciLit-LLM을 개발했습니다. SciLit-LLM은 과학 문헌에 대한 사전 훈련과 지도 학습 미세 조정을 통해 과학 문헌 서비스에서 LLM의 성능을 향상시키도록 설계되었습니다.



### Social Debiasing for Fair Multi-modal LLMs (https://arxiv.org/abs/2408.06569)
- **What's New**: 본 논문은 멀티모달 대규모 언어 모델(MLLM)의 사회적 편향 문제를 해결하기 위해 두 가지 새로운 접근 방식을 제시합니다. 첫째, 다양한 사회적 개념을 포함하는 역설적 데이터 세트(Counterfactual dataset with Multiple Social Concepts, CMSC)를 소개합니다. CMSC는 기존 데이터 세트에 비해 더 다양하고 광범위한 훈련 세트를 제공합니다. 둘째, 반-고정관념 편향 제거 전략(Anti-Stereotype Debiasing, ASD)을 제안합니다. ASD는 MLLM 훈련 과정을 재검토하고, 자동 회귀 손실 함수를 조정하고, 데이터 샘플링 방법을 개선하여 편향을 제거합니다. 



### AquilaMoE: Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies (https://arxiv.org/abs/2408.06567)
- **What's New**: This paper introduces AquilaMoE, a bilingual 8*16B Mixture of Experts (MoE) language model trained using a novel method called EfficientScale, which significantly improves training efficiency and reduces data requirements compared to traditional methods.

- **Technical Details**: EfficientScale consists of three phases: Preparation, Scale-Up, and Scale-Out. In the Scale-Up phase, a smaller pre-trained dense model's weights are used to initialize a larger dense model, enabling knowledge transfer and continued training with less data. The Scale-Out phase further enhances performance by using a pre-trained dense model to initialize MoE experts. The paper proposes AKI-Pro, an improved version of Advanced Knowledge Initialization (AKI), for initializing the larger model, addressing limitations of the original AKI in expanding the depth and adapting to Group Query Attention (GQA).

- **Performance Highlights**: The authors successfully trained a 16B model and then the 8*16B AquilaMoE model using the optimal initialization scheme. Extensive validation experiments on smaller models (1.8B and 7B) demonstrate that these models maintain and further reduce loss during continuous pretraining, highlighting the effectiveness of EfficientScale.



### Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data (https://arxiv.org/abs/2408.06537)
- **What's New**: 이 논문은 기계 번역(MT) 모델 훈련을 위한 새로운 고품질 데이터셋인 NewsPaLM을 소개합니다. NewsPaLM은 대규모 언어 모델(LLM)인 PaLM-2를 사용하여 생성되었으며, 기존 훈련 데이터셋보다 성능이 뛰어나다는 것이 입증되었습니다. 특히, 이 데이터셋은 문장 수준과 다문장 수준의 병렬 데이터를 포함하고 있으며, Minimum Bayes Risk (MBR) 디코딩 및 Quality Estimation (QE) 재순위 지정 기술을 적용하여 생성되었습니다.



### Chain-of-Strategy Planning with LLMs: Aligning the Generation of Psychotherapy Dialogue with Strategy in Motivational Interviewing (https://arxiv.org/abs/2408.06527)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)을 사용하여 동기적 면접(MI)에 기반한 심리치료 대화를 생성하는 새로운 접근 방식을 제시합니다. 특히, Chain-of-Strategy(CoS) 계획을 사용하여 전략 인식 대화 생성을 제안하며, 이는 먼저 MI 전략을 추론으로 예측하고 이러한 전략을 사용하여 후속 대화 생성을 안내합니다. 이는 생성된 MI 대화를 치료 전략에 맞춤으로써 심리치료에서 제어 가능하고 설명 가능한 생성 가능성을 제공합니다.

- **Technical Details**: 본 연구에서는 Chain-of-Strategy (CoS) 계획을 기반으로 전략 인식 대화 생성을 제안합니다. CoS 계획은 MI 전략을 추론으로 예측하고 이러한 전략을 사용하여 후속 대화 생성을 안내하는 방식입니다. 이러한 방식은 생성된 MI 대화를 치료 전략에 맞춤으로써 심리치료에서 제어 가능하고 설명 가능한 생성 가능성을 제공합니다. 본 연구에서는 AnnoMI와 BiMISC의 두 가지 MI 데이터 세트를 사용하여 이러한 접근 방식을 평가합니다. 이러한 데이터 세트는 MI 대화와 MISC 스킴에 의해 주석 처리된 MI 전략을 포함하고 있습니다.

- **Performance Highlights**: 자동 및 인간 평가를 포함한 광범위한 실험을 통해 MI 전략의 효과를 검증했습니다. 연구 결과는 LLM이 전략적으로 정렬된 대화를 생성할 가능성을 보여주었으며 심리치료 환경에서 실용적인 응용을 위한 방향을 제시합니다.



### Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models (https://arxiv.org/abs/2408.06518)
- **What's New**: This paper identifies and characterizes a new phenomenon called "semantic leakage" in language models (LMs), where irrelevant information from the prompt leaks into the generation in unexpected ways. This is distinct from previously identified issues like hallucinations, sycophancy, and biases.



### Cross-Lingual Conversational Speech Summarization with Large Language Models (https://arxiv.org/abs/2408.06484)
- **What's New**: This paper introduces a novel dataset for cross-lingual conversational speech summarization, addressing the lack of resources in this area. It utilizes the Fisher and Callhome Spanish-English Speech Translation corpus, generating summaries from the English translations using GPT-4. This allows for the evaluation of summarization models in the presence of ASR and MT errors.

- **Technical Details**: The authors create a cascade-based system using Whisper-large-v3 for ASR and NLLB 1.3 Billion parameter dense model for machine translation. They experiment with various LLMs for summarization, including GPT-4 and Mistral-7B. They employ LoRA fine-tuning to adapt the models for the task. The dataset is constructed by splitting conversations into chunks and generating four reference summaries per chunk using GPT-4.

- **Performance Highlights**: The Mistral-7B model, adapted for this task, outperforms other off-the-shelf LLMs and achieves performance comparable to GPT-4. The results highlight the importance of fine-tuning for task adaptation and the potential of smaller, quantized models in this domain.



### TOGGL: Transcribing Overlapping Speech with Staggered Labeling (https://arxiv.org/abs/2408.06474)
Comments:
          5 pages

- **What's New**: 본 논문에서는 TOGGL 모델을 제안하여, 단일 디코더로 여러 명의 화자의 음성을 동시에 전사합니다. TOGGL 모델은 특수 출력 토큰을 사용하여, 각 화자의 음성을 할당합니다.  이는 기존 모델과 달리 별도의 디코더가 필요하지 않아 효율성을 높였습니다. 또한 TOGGL 모델은 2명의 화자에 대해 학습하더라도, 2명 이상의 화자에 대한 음성 전사에도 일반화될 수 있습니다.  본 연구에서는, 대화형 음성 데이터셋에서 경쟁적인 접근 방식에 비해 우수한 성능을 보여주었습니다.  더욱이 단일 화자 오디오에 대한 성능도 개선되었습니다.



### Evaluating Language Models on Entity Disambiguation in Tables (https://arxiv.org/abs/2408.06423)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)을 기반으로 한 **일반적인 테이블 이해 및 조작(GTUM)** 접근 방식이 **Semantic Table Interpretation(STI)** 작업, 특히 **Cell-Entity Annotation(CEA)** 작업에서 기존의 규칙 기반 접근 방식과 비교하여 어떻게 성능을 발휘하는지 분석합니다.



### Diversity Empowers Intelligence: Integrating Expertise of Software Engineering Agents (https://arxiv.org/abs/2408.07060)
- **What's New**: DEI (Diversity Empowered Intelligence), a new framework for enhancing software engineering (SWE) agent performance by leveraging their diverse expertise, is introduced. DEI functions as a meta-module that enables collaboration among agents, leading to improved problem-solving.

- **Technical Details**: DEI integrates with existing SWE agent frameworks, creating a multi-agent ensemble system with a re-ranking pipeline. This pipeline utilizes a multi-stage rating system to identify the most effective agent for resolving a specific issue. DEI employs a committee of agents to enhance performance, surpassing the capabilities of individual agents.

- **Performance Highlights**: DEI significantly improves the issue resolution rate on SWE-Bench Lite, a benchmark for evaluating SWE agent performance. For instance, a group of open-source agents, with a maximum individual resolve rate of 27.3%, achieved a 34.3% resolve rate with DEI, representing a 25% improvement. The best-performing group, guided by DEI, achieved a 55% resolve rate, securing the highest ranking on SWE-Bench Lite.



### A Survey on Model MoErging: Recycling and Routing Among Specialized Experts for Collaborative Learning (https://arxiv.org/abs/2408.07057)
Comments:
          26 pages

- **What's New**: 본 논문은 **모델 모어징 (MoErging)**, 즉 특정 도메인이나 작업에 특화된 사전 훈련된 전문가 모델을 재활용하여 성능이나 일반화 능력을 향상시키는 새로운 패러다임을 제시하는 최신 연구 동향을 소개합니다. MoErging은 기존의 **혼합 전문가 모델 (MoE)**과 **모델 병합 (Model Merging)** 접근 방식과 유사하지만, 전문가 모델이 중앙 집중식으로 훈련되는 것이 아니라 분산된 기여자들에 의해 독립적으로 훈련되고 공유되는 점이 특징입니다. 이 논문은 MoErging 방법을 정확히 비교 분석하고, **전문가 (experts)**, **라우팅 (routing)**, **애플리케이션 (application)**의 세 가지 범주로 MoErging 방법을 분류하는 새로운 분류 체계를 제시합니다. 또한 MoErging과 관련된 연구 분야, 이러한 접근 방식을 지원하는 도구, 미래 연구 방향을 논의합니다.



### The News Comment Gap and Algorithmic Agenda Setting in Online Forums (https://arxiv.org/abs/2408.07052)
- **What's New**: 본 논문은 뉴스 기사에 대한 독자들의 댓글에 대한 뉴스 기자와 독자 간의 차이, 즉 "뉴스 댓글 격차"(News Comment Gap)를 분석합니다. 이는 뉴스 가치에 대한 기자와 독자 간 차이인 "뉴스 격차"(News Gap)를 확장한 개념입니다. 특히 다양한 댓글 순위 알고리즘이 독자와 기자에게 어떻게 다른 방식으로 댓글을 보여주는지 분석하여 뉴스 토론의 대표성을 어떻게 형성하는지 조사합니다.

- **Technical Details**: 오스트리아 신문 Der Standard의 120만 개 이상의 댓글 데이터를 분석하여 기자와 독자의 댓글 선호도를 비교했습니다. 선호도를 분석하기 위해 "편집자 선택"(Editors’ Picks)과 사용자 투표를 기반으로 회귀 및 분류 모델을 사용했습니다. 또한 다양한 댓글 순위 알고리즘의 성능을 평가하기 위해 새로운 "특징 지향 순위 유틸리티 지표"(Feature-Oriented Ranking Utility Metric, FORUM)를 도입했습니다.

- **Performance Highlights**: 분석 결과, 기자들은 긍정적이고 시의적절하며 복잡하고 직접적인 댓글을 선호하는 반면 독자들은 기사 내용과 유사하며 엘리트 저자의 댓글을 선호하는 것으로 나타났습니다. 또한 FORUM을 통해 감정, 주제 관련성, 어휘 다양성, 가독성 등 다양한 측면에서 댓글 순위 알고리즘이 서로 다른 결과를 보여준다는 것을 발견했습니다. 즉, 댓글 순위 알고리즘은 뉴스 토론의 방향을 크게 바꿀 수 있는 힘을 가지고 있으며, 기자들은 큐레이션과 알고리즘을 통해 토론에 큰 영향력을 행사할 수 있습니다.



### TableGuard -- Securing Structured & Unstructured Data (https://arxiv.org/abs/2408.07045)
Comments:
          7 pages, 3 tables, 1 figure

- **What's New**: TableGuard는 관계형 데이터베이스를 위한 혁신적인 데이터 난독화 접근 방식으로, 컨텍스트 기반 난독화를 활용하여 API 호출이 난독화된 데이터만 반환하도록 함으로써 제3자와 데이터를 공유할 때 개인 정보를 보호합니다. TableGuard는 컨텍스트에 적합한 대안으로 민감한 데이터 요소를 대체하여 데이터의 관계적 무결성 및 일관성을 유지함으로써 인지 부조화 및 데이터 유출 위험을 완화합니다.



### Causal Agent based on Large Language Mod (https://arxiv.org/abs/2408.06849)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)이 인과적 추론(causal reasoning)을 수행할 수 있도록 인과적 에이전트(Causal Agent)라는 새로운 프레임워크를 제시합니다. 인과적 에이전트는 인과적 방법(causal methods)을 활용하여 LLM이 인과 관계를 이해하고 활용할 수 있도록 지원합니다. 이는 LLM이 인과적 문제를 해결하는 능력을 향상시키고 다양한 분야에서 활용 가능성을 확대할 수 있는 중요한 발전입니다.



### MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty (https://arxiv.org/abs/2408.06816)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 신뢰성을 향상시키기 위해 데이터 불확실성(aleatoric uncertainty) 하에서의 불확실성 정량화를 조사합니다. 이를 위해 다중 답변 질의응답 데이터셋(MAQA)을 새롭게 제안하여 여러 답변을 요구하는 질문을 통해 데이터 불확실성을 도입합니다. 또한, 다양한 화이트박스 및 블랙박스 LLM에 대해 5가지 불확실성 정량화 방법을 평가하여 데이터 불확실성 하에서의 LLM 신뢰성을 측정합니다.



### Sumotosima: A Framework and Dataset for Classifying and Summarizing Otoscopic Images (https://arxiv.org/abs/2408.06755)
Comments:
          Work in Progress

- **What's New**: 이 논문에서는 청각기 이미지를 이해하고 요약하는 새로운 딥 러닝 및 트랜스포머 기반 프레임워크인 Sumotosima(Otoscopic 이미지 요약기)를 제안합니다. Sumotosima는 환자에게 적합한 요약을 제공하여 청각기 이미지를 명확하고 효율적으로 이해하도록 돕는 것을 목표로 합니다. 이를 위해 삼중 손실(triplet loss)과 교차 엔트로피 손실(cross-entropy loss)의 조합을 사용하고, 텍스트 및 이미지 임베딩(embedding)을 결합한 지식 증강 다중 모달 BART(Knowledge Enhanced Multimodal BART)를 활용합니다. 데이터셋 부족 문제를 해결하기 위해 5가지 독특한 범주에 대한 이미지 500개와 이에 대한 요약이 포함된 OCASD(Otoscopic Classification And Summary Dataset) 데이터셋을 구축했습니다.



### Large language models can consistently generate high-quality content for election disinformation operations (https://arxiv.org/abs/2408.06731)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 선거 허위 정보 생성 능력을 평가하여 선거 허위 정보 작전의 자동화 가능성을 조사합니다. 연구는 DisElect 데이터셋을 사용하여 13개 LLM의 선거 허위 정보 생성 능력을 평가하고, LLM이 생성한 허위 정보가 인간이 작성한 것과 구별할 수 있는지 여부를 평가하는 실험을 수행합니다. 연구 결과, 대부분의 LLM은 선거 허위 정보 생성 지시에 따르며, 일부 모델은 인간 평가자가 LLM이 생성한 허위 정보와 인간이 작성한 허위 정보를 구별하는 데 어려움을 겪는 것으로 나타났습니다.



### Enhancing Visual Dialog State Tracking through Iterative Object-Entity Alignment in Multi-Round Conversations (https://arxiv.org/abs/2408.06725)
Comments:
          This article has been accepted in CAAI Transactions on Intelligence Technology! Article ID: CIT2_12370, Article DOI: https://doi.org/10.1049/cit2.12370

- **What's New**: This paper introduces a new model, Multi-round Dialogue State Tracking (MDST), for Visual Dialog (VD). MDST addresses the limitation of previous VD models that treat the entire dialog history as a simple text input, by leveraging the dialogue state learned from the dialog history to answer questions. MDST captures each round of dialog history, constructing internal dialogue state representations defined as 2-tuples of vision-language representations, which effectively ground the current question, enabling the generation of more accurate answers.



### Harnessing Earnings Reports for Stock Predictions: A QLoRA-Enhanced LLM Approach (https://arxiv.org/abs/2408.06634)
Comments:
          Accepted by 2024 6th International Conference on Data-driven Optimization of Complex Systems

- **What's New**: 이 논문은 실적 발표 후 주가 변동을 예측하기 위해 거대 언어 모델(LLM)을 사용하여 새로운 접근 방식을 제시합니다. 이 방법은 '기본 요인'(재무 지표 성장, 실적 발표 내용)과 '외부 요인'(최근 시장 지수 실적, 애널리스트 등급)을 통합하여 포괄적인 데이터셋을 구성하고, LLM을 미세 조정합니다.

- **Technical Details**: 본 연구는 S&P 500에 상장된 501개 기업의 데이터를 사용하며, 각 기업의 주식 성과, 시장 지수(SPY, QQQ, DOW) 성과, 애널리스트 등급, 실적 발표 내용, 실적 서프라이즈(earnings surprise)와 같은 요소들을 포함합니다. 이 데이터는 자연어 텍스트로 변환되어 LLM에 사용되며, 'Long' 또는 'Short'로 분류된 다음날 주식 성과를 예측하는 데 사용됩니다. 모델은 '지침 기반 기법'과 '양자화된 저랭크 적응(QLoRA)' 압축을 사용하여 미세 조정됩니다.

- **Performance Highlights**: llama-3-8b-Instruct-4bit 모델은 GPT-4와 같은 벤치마크에 비해 정확성, 가중 F1 점수, 매튜스 상관 계수(MCC) 측면에서 탁월한 예측 성능을 보여주었습니다. 이 모델은 기존 모델보다 상당한 개선을 보였습니다. 또한, 'Hold' 옵션을 추가하고 예측 기간을 확장하여 다양한 투자 스타일과 시간 프레임에 맞게 적용할 수 있는 잠재력을 제시합니다.



### EditScribe: Non-Visual Image Editing with Natural Language Verification Loops (https://arxiv.org/abs/2408.06632)
Comments:
          ASSETS 2024

- **What's New**: EditScribe, a prototype system that utilizes large multimodal models (LMMs) to make object-level image editing actions non-visually accessible to blind and low-vision (BLV) individuals, is introduced. This system addresses the accessibility challenges in image editing for BLV users by enabling them to comprehend image content, specify edit actions through natural language prompts, and receive verification feedback in natural language to ensure accurate edits.

- **Technical Details**: EditScribe leverages natural language verification loops. It provides four types of verification feedback: Summary of Visual Changes, AI Judgement, updated General and Object Descriptions, and follow-up questions for clarification. The system is tested with five object-level edit actions, including blurring, removing, color changing, brightness adjusting, and adding text to an object in an image. This object-level focus allows for precise image detail manipulation crucial for tasks commonly desired by BLV individuals like privacy protection and image enhancement.

- **Performance Highlights**: A study with 10 BLV participants demonstrated the effectiveness of EditScribe. The participants were able to perform most editing tasks using the system and adopted different prompting strategies, including detailed, succinct, and varied tones to facilitate the system's understanding. They also showed preferences for different types of verification feedback depending on the context. Overall, participants expressed confidence in the edited images produced by EditScribe and were willing to publish them based on context, but preferred a second check using sighted assistance or other AI services. The findings highlight the potential of natural language verification loops for non-visual content creation accessibility.



### WorldScribe: Towards Context-Aware Live Visual Descriptions (https://arxiv.org/abs/2408.06627)
Comments:
          UIST 2024

- **What's New**: WorldScribe는 사용자의 컨텍스트에 맞게 적응 가능한 실시간 시각적 설명을 생성하는 새로운 시스템입니다. 이 시스템은 사용자의 의도에 따라 설명을 우선순위를 정하고, 시각적 컨텍스트와 사운드 컨텍스트에 따라 설명 방식을 조정합니다.



### Towards Robust and Cost-Efficient Knowledge Unlearning for Large Language Models (https://arxiv.org/abs/2408.06621)
Comments:
          Preprint

- **What's New**: 본 논문에서는 LLM (Large Language Model)에서 민감한 데이터의 지식을 제거하는 효율적인 기계 학습 해제 프레임워크를 제안합니다. 기존의 Gradient Ascent (GA) 방식의 단점을 해결하기 위해, Inverted Hinge Loss (IHL)를 제안하여 더 빠르고 안정적인 학습 해제를 가능하게 합니다. 또한, LoRA (Low-Rank Adaptation)의 효율성을 높이기 위해, Fisher-weighted low-rank approximation 기반의 새로운 LoRA 초기화 방법을 제시합니다. 이러한 새로운 기술을 통해 LLM에서 원치 않는 데이터에 대한 지식을 효과적으로 제거하고, 원래 모델의 지식을 보존할 수 있습니다.



### CROME: Cross-Modal Adapters for Efficient Multimodal LLM (https://arxiv.org/abs/2408.06610)
- **What's New**: CROME은 비용 효율적인 멀티모달 대규모 언어 모델(MLLM) 학습을 위한 새로운 프레임워크입니다. 기존의 접근 방식과 달리 CROME은 언어 모델 재학습 비용을 줄이고 다양한 작업에 대한 적응력을 높입니다. 핵심은 '게이트 방식의 크로스 모달 어댑터(gated cross-modal adapter)'인데, 이는 이미지와 텍스트 표현을 효율적으로 결합하여 동결된 LLM에 입력합니다. 이 어댑터는 몇 가지 매개변수만으로 학습되므로 효율적인 크로스 모달 이해를 가능하게 합니다. 특히, CROME은 표준 시각 질문 답변 및 지시 추론 벤치마크에서 우수한 제로 샷(zero-shot) 성능을 보여줍니다. 또한, 매개변수 효율성이 뛰어난 미세 조정(fine-tuning)을 통해 특정 작업에 특화된 최첨단 방법과 경쟁합니다. CROME은 확장 가능하고 적응력이 뛰어나며 매개변수 효율적인 멀티모달 모델을 구축하기 위한 사전 LLM 정렬(pre-LM alignment)의 잠재력을 보여줍니다.



### Hierarchical in-Context Reinforcement Learning with Hindsight Modular Reflections for Planning (https://arxiv.org/abs/2408.06520)
- **What's New**: This paper introduces Hierarchical in-Context Reinforcement Learning (HCRL), a novel framework that utilizes Large Language Models (LLMs) to decompose complex tasks into sub-tasks for efficient robotic decision-making. Inspired by Hierarchical Reinforcement Learning (HRL), HCRL leverages a high-level policy (LLM) to generate sub-goals on-the-fly, which are then executed by a low-level policy.  Additionally, it incorporates Hindsight Modular Reflection (HMR) to improve multi-episode learning by reflecting on both sub-goal and full-trajectory levels, enhancing the agent's ability to identify and correct errors.

- **Technical Details**: HCRL employs a hierarchical structure with a high-level policy (LLM) responsible for task decomposition and a low-level policy for executing sub-tasks. The LLM uses structured prompt tags ([Goal], [Think], [Action], [Finish]) to guide its reasoning and decision-making. HMR enhances learning by providing two levels of reflection: low-level reflection on sub-trajectories and high-level reflection on the sequence of proposed goals, allowing for more efficient learning compared to reflecting on the entire trajectory.

- **Performance Highlights**: Evaluations across three diverse environments (ALFWorld, Webshop, and HotpotQA) demonstrate significant performance improvements over existing in-context learning methods. HCRL achieves a 9% improvement on ALFWorld, a 42% improvement on Webshop (establishing a new state-of-the-art), and a 10% improvement on HotpotQA in five episodes. These results highlight the effectiveness of HCRL in enhancing sample efficiency and generalization capabilities in language-guided RL.



### Towards Autonomous Agents: Adaptive-planning, Reasoning, and Acting in Language Models (https://arxiv.org/abs/2408.06458)
- **What's New**: 이 연구는 자율적인 의사 결정 언어 에이전트를 구축하기 위한 새로운 인 컨텍스트 학습 알고리즘을 제안합니다. 언어 에이전트는 작업이 실패할 때마다 스스로를 수정하여 동일한 작업을 지속적으로 해결하려고 합니다. 연구에서 사용된 언어 에이전트는 텍스트 기반 게임 환경에서 작업을 해결할 수 있는 능력을 보여줍니다. 연구 결과, 제안된 방법을 사용하는 gemma-2-9b-it 언어 모델은 처음 시도에서 실패한 여섯 가지 작업 중 두 가지를 성공적으로 완료할 수 있었습니다. 이는 자기 수정을 통한 단일 언어 모델의 문제 해결 능력을 향상시키는 데 있어 제안된 접근 방식의 효과를 강조하며, 더 발전된 자율 에이전트를 위한 길을 열어줍니다. 코드는 [링크](https://this https URL) 에서 공개적으로 이용 가능합니다.

- **Technical Details**: 이 연구는 ReAct (Reasoning and Acting) 프롬프팅을 기반으로 하며, 언어 모델이 텍스트 기반 게임 환경에서 작업을 해결하는 데 필요한 추론 과정을 학습하도록 돕습니다. ReAct 프롬프팅은 언어 모델에 사전 언어 설명을 제공하여 다양한 언어 추론 및 의사 결정 작업을 해결하기 위한 추론을 안내하고 외부 세계로부터 피드백을 받아 이러한 추론을 조정합니다. 본 연구의 핵심은 '자기 반성' (self-reflection) 단계를 추가하여 언어 에이전트가 이전 실패로부터 배우고 자체 정책을 개선할 수 있도록 하는 것입니다. 기존의 ReAct 프롬프팅과 달리, 이 연구는 하나의 언어 모델만을 사용하여 자율적인 언어 에이전트를 구축합니다. 이는 기존의 ReAct 프롬프팅이 복잡한 작업을 해결하는 데 어려움을 겪는다는 점을 해결하기 위한 시도입니다.

- **Performance Highlights**: 제안된 방법은 ALFWorld 환경에서 14가지 다른 의사 결정 작업에서, '자기 반성' 단계 없이 ReAct 프롬프팅만을 사용하는 에이전트가 해결할 수 없는 작업을 성공적으로 완료할 수 있었습니다. 이는 '자기 반성' 단계가 언어 에이전트의 성능을 향상시키는 데 효과적임을 보여줍니다. 본 연구는 '자기 반성'을 통해 단일 언어 모델의 자율성과 문제 해결 능력을 향상시킬 수 있음을 입증합니다. 또한, 제안된 방법은 기존의 ReAct 프롬프팅의 한계를 극복하고 더 복잡한 작업을 해결할 수 있는 잠재력을 가지고 있습니다.



### Evaluating Language Models for Efficient Code Generation (https://arxiv.org/abs/2408.06450)
- **What's New**: DPE (Differential Performance Evaluation), 새로운 프레임워크 제안! 코드 생성 능력을 효과적으로 평가하는 새로운 방식으로, 기존 벤치마크의 단점을 보완하여 LLMs의 코드 효율성을 더 정확하게 평가할 수 있습니다.

- **Technical Details**: DPE는 두 단계로 구성됩니다: 첫째, 기존 벤치마크에서 효율성을 요구하는 작업을 선택하고 연산량이 많은 입력을 생성하여 LLM 솔루션의 효율성을 평가합니다. 둘째, 새로운 솔루션을 프로파일링하고, 서로 다른 효율성 수준을 보이는 기준 솔루션과 비교하여 매칭된 수준을 효율성 점수로 정의합니다.  DPE는 'Synthesizing a Synthesizer (SaS)'를 이용하여 스케일 조절이 가능한 입력 샘플러를 생성합니다. 이 샘플러는 지수적 입력 샘플링을 통해 연산량이 많은 입력을 생성하도록 조정됩니다. 또한, DPE는 필터링 전략을 통해 성능 평가에 적합한 작업을 선택합니다. DPE는 새로운 솔루션과 기준 솔루션을 프로파일링하여 매칭된 성능 클러스터의 순위를 통해 효율성 점수를 결정합니다. 

- **Performance Highlights**: DPE를 사용하여 121개의 성능 중심 프로그래밍 작업으로 구성된 EvalPerf 벤치마크를 만들었습니다. EvalPerf는 다양한 플랫폼에서도 일관된 성능 평가를 제공하고, 기존 방법보다 4.8배 더 성능이 뛰어난 입력을 생성할 수 있음을 보여줍니다.

- **Key Findings**: 모델 크기, 명령어 튜닝, 프롬프팅이 코드 효율성에 미치는 영향을 조사한 결과,  모델 크기의 증가는 코드 효율성을 항상 개선하는 것은 아니지만, 일반적인 명령어 튜닝은 코드 정확성과 효율성 모두를 향상시키는 것으로 나타났습니다.

- **Availability**: EvalPerf는 github.com/evalplus/evalplus에서 오픈소스로 제공됩니다.



### ViC: Virtual Compiler Is All You Need For Assembly Code Search (https://arxiv.org/abs/2408.06385)
- **What's New**: 본 논문에서는 역 엔지니어링을 위한 어셈블리 코드 검색을 개선하기 위해 가상 컴파일러(Virtual Compiler, ViC)라는 새로운 접근 방식을 제시합니다. ViC는 대규모 언어 모델(LLM)을 사용하여 다양한 프로그래밍 언어의 소스 코드를 어셈블리 코드로 컴파일하는 기능을 제공합니다. 이를 통해 기존 컴파일러의 복잡성을 우회하고 다양한 프로그래밍 언어의 어셈블리 코드 데이터셋을 효과적으로 구축할 수 있습니다.



### Lyrics Transcription for Humans: A Readability-Aware Benchmark (https://arxiv.org/abs/2408.06370)
Comments:
          ISMIR 2024 camera-ready. 6 pages + references + supplementary material. Website this https URL Data this https URL Code this https URL. arXiv admin note: text overlap with arXiv:2311.13987

- **What's New**: 새로운 벤치마크 데이터셋 Jam-ALT를 소개합니다. 이 데이터셋은 음악 산업 표준에 맞춰 JamendoLyrics 데이터셋을 재구성한 것으로 가사 전사 및 형식 지정에 대한 규칙을 반영합니다. 기존 데이터셋과 달리 Jam-ALT는 구두점, 줄 바꿈, 대소문자 및 비언어적 발성 음 등의 세부 사항을 포함합니다.



### Large Language Model Agent in Financial Trading: A Survey (https://arxiv.org/abs/2408.06361)
- **What's New**: This survey reviews the current research on utilizing Large Language Models (LLMs) as agents in financial trading. It investigates their architecture, data inputs, performance, and challenges. This is the first survey dedicated to this domain.



### Deep Learning based Key Information Extraction from Business Documents: Systematic Literature Review (https://arxiv.org/abs/2408.06345)
Comments:
          52 pages, 7 figures, 9 tables; Submitted to ACM Computing Surveys

- **What's New**: 이 논문은 비즈니스 문서에서 핵심 정보를 추출하는 데 사용되는 최근의 딥 러닝 기반 접근 방식을 광범위하게 분석하고 요약합니다. 96개의 논문을 분석하여 Document Understanding (DU) 분야의 최신 연구를 살펴봅니다. 특히, Key Information Extraction (KIE)에 중점을 두고 딥 러닝 기법과 기술적 특징을 살펴봅니다. 또한, 비즈니스 프로세스 관점에서 KIE를 연구하고, 각 방법의 특징을 범주화하여 비교 분석합니다.

- **Technical Details**: 이 연구는 딥 러닝 기반 KIE 방법을 그래프 기반, 그리드 기반, 시퀀스 기반의 세 가지 주요 그룹으로 분류하고 각 그룹에 속하는 주요 기술들을 자세히 살펴봅니다. 그래프 기반 시스템은 문서 페이지를 레이아웃과 콘텐츠를 나타내는 그래프 구조로 변환합니다. 그리드 기반 시스템은 문서 이미지 픽셀을 기반으로 잘 정의된 연결을 갖는 그리드 구조를 만듭니다. 시퀀스 기반 시스템은 문서를 선형 텍스트 시퀀스로 처리합니다. 또한, 문서 이해 작업을 지각, 유도, 추론의 세 가지 하위 작업으로 분류하고 각 작업의 핵심 내용을 소개합니다. KIE 작업은 Named Entity Recognition (NER), Relation Extraction (RE) 등의 하위 작업으로 구성되며, 각 작업의 목적과 특징을 설명합니다.

- **Performance Highlights**: 이 연구는 딥 러닝 기반 KIE 방법의 다양한 성능 지표를 분석하고, 각 방법의 장단점을 비교 분석합니다. 특히, 정확도, 재현율, F1 스코어, 처리 시간 등을 비교하여 각 방법의 강점과 약점을 파악합니다. 또한, 각 방법의 적용 가능성을 다양한 비즈니스 문서 유형에 대한 분석을 통해 살펴봅니다.



### LOLgorithm: Integrating Semantic,Syntactic and Contextual Elements for Humor Classification (https://arxiv.org/abs/2408.06335)
- **What's New**: 본 논문은 자연어 처리(NLP)에서 계산적 방법보다 구문론적, 의미론적, 문맥적 특징을 우선시하여 유머 감지를 탐구합니다. Colbert라는 모델은 BERT 임베딩과 병렬 숨겨진 계층을 사용하여 문장 일관성을 포착합니다. SHAP 해석과 의사 결정 트리는 영향력 있는 특징을 식별하여 숨겨진 데이터에서 유머 감지 정확도를 향상시키는 포괄적인 접근 방식을 보여줍니다.



### FastFiD: Improve Inference Efficiency of Open Domain Question Answering via Sentence Selection (https://arxiv.org/abs/2408.06333)
Comments:
          ACL 2024 Main Conference

- **What's New**: 본 논문에서는 오픈 도메인 질의 응답 (ODQA) 시스템의 속도를 높이기 위한 새로운 방법인 FastFiD를 소개합니다. FastFiD는 인코더 출력에서 문장 선택을 수행하여 디코더에 필요한 컨텍스트 길이를 줄이고 추론 시간을 단축합니다. 즉, FastFiD는 여러 개의 문장으로 구성된 컨텍스트를 디코더에 제공하는 기존 FiD 모델과 달리, 가장 중요한 몇 개의 문장만 선택하여 디코더에 제공함으로써 추론 속도를 향상시킵니다. 실험 결과, FastFiD는 컨텍스트 길이를 최대 38배까지 줄이고 추론 속도를 2.3배에서 5.7배까지 향상시키면서도 기존 FiD 모델과 유사한 성능을 유지하는 것으로 나타났습니다.



### Animate, or Inanimate, That is the Question for Large Language Models (https://arxiv.org/abs/2408.06332)
- **What's New**: This paper investigates whether Large Language Models (LLMs) can process animacy (a key concept in human cognition, impacting memory, vision, and language) in a similar way to humans.  This is significant because LLMs are trained solely on text, unlike humans who also draw on visual and physical stimuli.



### Long-Form Answers to Visual Questions from Blind and Low Vision Peop (https://arxiv.org/abs/2408.06303)
Comments:
          COLM 2024

- **What's New**: This paper introduces VizWiz-LF, a dataset of long-form answers to visual questions posed by blind and low vision (BLV) users. This dataset provides 4.2k long-form answers to 600 visual questions, collected from human expert describers and six VQA models. The paper also proposes a new set of functional roles for long-form answers, going beyond just answering the question to include explanations and suggestions.

- **Technical Details**: The paper identifies different functional roles (e.g., answer, explanation, suggestion) within long-form answers and proposes a classifier to automatically identify these roles. They also analyze the information sources used in these answers (e.g., image content, image quality, external information). They assess the ability of VQA models to abstain from answering unanswerable questions using various prompting strategies.

- **Performance Highlights**: The paper finds that long-form answers generated by VQA models often hallucinate incorrect visual details, particularly for unanswerable questions. They demonstrate that GPT-4 achieves the highest recall for abstention (0.82), while QWEN (0.42) performs well with default settings. The paper also highlights the importance of evaluating LFVQA beyond just factual accuracy, considering factors like relevance and plausibility in user experience.

- **Contributions**: The work provides the first dataset with both short and long answers to visual questions, enabling the transfer from short-answer VQA tasks to long-answer tasks. It proposes a new set of functional roles for LFVQA, which can be used to improve and evaluate LFVQA systems. It highlights the importance of user experience evaluation for LFVQA, going beyond factual accuracy and considering factors like relevance and plausibility.



### Synthetic Patient-Physician Dialogue Generation from Clinical Notes Using LLM (https://arxiv.org/abs/2408.06285)
- **What's New**: SynDial, 새로운 의료 대화 데이터 생성 방식은 단일 LLM(Large Language Model)을 활용하여 제로 샷 프롬프팅(zero-shot prompting)과 피드백 루프를 통해 고품질 의료 대화 데이터를 생성합니다. 특히 SynDial은 유사성(similarity)과 추출성(extractiveness)에 대한 가중 평가 점수를 기반으로 피드백 루프를 통해 대화의 품질을 지속적으로 향상시킵니다. 이는 기존 방법과 차별화되는 중요한 특징이며, 더 나은 대화 데이터 생성을 가능하게 합니다.



### MovieSum: An Abstractive Summarization Dataset for Movie Screenplays (https://arxiv.org/abs/2408.06281)
Comments:
          ACL 2024 Findings

- **What's New**: 이 논문은 영화 각본 요약을 위한 새로운 데이터셋인 "MovieSum"을 소개합니다. MovieSum은 2200개의 영화 각본과 해당 위키피디아 줄거리 요약으로 구성되어 있습니다. 기존 데이터셋에 비해 MovieSum은 다음과 같은 특징을 가지고 있습니다: (1) TV 드라마보다 긴 영화 각본을 포함합니다. (2) 이전 영화 각본 데이터셋보다 두 배 더 큰 규모를 가지고 있습니다. (3) 추가적인 외부 정보 접근을 용이하게 하기 위해 IMDb ID가 포함된 메타데이터를 제공합니다.



### Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation (https://arxiv.org/abs/2408.06276)
- **What's New**: 이 논문은 EXP3RT라는 새로운 LLM 기반 추천 시스템을 소개하며, 사용자와 상품 리뷰에 담긴 풍부한 선호도 정보를 활용합니다. EXP3RT는 LLM을 통해 사용자 및 상품 프로필을 구축하고, 사용자/상품 프로필 및 상품 설명을 고려하여 단계별 추론과 함께 예상 평점을 생성하여 추천 시스템의 설명력을 높이는 것을 목표로 합니다.

- **Technical Details**: EXP3RT는 세 가지 주요 단계를 수행합니다. 첫째, 원시 리뷰에서 주관적인 선호도를 추출하고 캡슐화하여 사용자 및 상품 프로필을 만듭니다. 둘째, 사용자 및 상품 프로필과 상품 설명을 이용하여 단계별 추론을 생성합니다. 마지막으로, 추론된 내용을 바탕으로 예상 평점을 예측합니다. 이러한 단계를 통해 사용자의 선호도를 정확하게 파악하고, 예상 평점에 대한 근거를 제공합니다. EXP3RT는 사전 훈련된 LLM을 사용하여 튜닝되며, 기존의 추천 시스템에 통합하여 사용할 수 있습니다.

- **Performance Highlights**: EXP3RT는 다양한 추천 작업에서 기존 방법보다 우수한 성능을 보여줍니다. 특히, 평점 예측 및 상위 K 추천 작업에서 눈에 띄는 성능을 보이며, 추천 시스템의 설명력을 향상시킵니다. 또한, 기존의 CF 기반 추천 시스템과 함께 사용하여 다단계 순위 지정 파이프라인에서 후보 아이템을 효과적으로 재순위 지정합니다.

- **Contributions**: - EXP3RT는 리뷰 기반의 개인화된 추론을 통해 평점 예측 정확도를 효과적으로 향상시킵니다.
- EXP3RT는 리뷰에서 추출된 풍부한 선호도 정보를 활용하여 상세한 단계별 추론을 생성하고, 신뢰할 수 있고 논리적인 설명을 제공합니다.
- EXP3RT는 독립적인 추천 시스템으로 작동할 수 있으며, 다단계 순위 지정 파이프라인에서 후보 아이템 재순위 지정을 위한 기존 CF 기반 추천 시스템과 원활하게 통합될 수 있습니다.



### FuxiTranyu: A Multilingual Large Language Model Trained with Balanced Data (https://arxiv.org/abs/2408.06273)
- **What's New**: FuxiTranyu, a new open-source multilingual LLM (Large Language Model) that tackles the performance discrepancies between high- and low-resource languages, is introduced. The model is trained from scratch on a balanced dataset of 600 billion tokens across 43 natural languages and 16 programming languages.

- **Technical Details**: FuxiTranyu-8B, the base model with 8 billion parameters, is trained using a balanced multilingual data repository. Two instruction-tuned models, FuxiTranyu-8B-SFT and FuxiTranyu-8B-DPO, are fine-tuned on diverse multilingual instruction datasets and further refined with DPO (Decision-based Preference Optimization) for enhanced alignment.

- **Performance Highlights**: FuxiTranyu demonstrates competitive performance against existing multilingual LLMs like BLOOM-7B, PolyLM-13B, Llama-2-Chat-7B, and Mistral-7B-Instruct. It shows superior results on various multilingual benchmarks, including ARC, HellaSwag, MMLU, XWinograd, XCOPA, XStoryCloze, WMT, IWSLT, and XL-Sum. Interpretability analyses indicate that FuxiTranyu learns consistent multilingual representations across different languages.

- **Availability**: The base model and instruction-tuned models are publicly released on HuggingFace and Github along with 58 pretraining checkpoints.

- **Key Features**: Balanced multilingual data repository, from-scratch training, instruction tuning, DPO refinement, and competitive performance on multiple benchmarks.

- **Limitations**: Limited resources for extremely low-resource languages like Bengali and Tamil, leading to fewer neurons allocated for processing them.



### Context-aware Visual Storytelling with Visual Prefix Tuning and Contrastive Learning (https://arxiv.org/abs/2408.06259)
Comments:
          18 pages, 12 figures, accepted by INLG 2024

- **What's New**: 이 논문은 사전 훈련된 기반 모델의 일반화 능력을 활용하여 이미지 시퀀스에서 다중 문장 스토리를 생성하는 시각적 스토리텔링 시스템을 제안합니다. 이 시스템은 모달리티를 연결하는 경량 비전-언어 매핑 네트워크만 학습하고 일관성을 높이기 위해 컨텍스트를 통합합니다.

- **Technical Details**: 이 프레임워크는 비전-언어 매핑 네트워크의 컨텍스트 인식 기능을 강화하고 이전 스토리 문장을 통합하여 일관성을 강화합니다. 또한 시각적 관련성과 정보성을 개선하기 위해 다중 모달 대조 학습 목표를 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과는 자동 평가 지표와 인간 평가 모두에서 이 프레임워크에 의해 생성된 스토리가 다양하고 일관성 있으며 정보가 풍부하고 흥미롭다는 것을 보여줍니다.



### FLEURS-R: A Restored Multilingual Speech Corpus for Generation Tasks (https://arxiv.org/abs/2408.06227)
- **What's New**: FLEURS-R은 다양한 언어 (102개 언어)에 대한 음성 복원 (speech restoration) 데이터셋으로, FLEURS 데이터셋을 기반으로 합니다. FLEURS-R은 Miipher 음성 복원 모델을 적용하여 기존 FLEURS 데이터셋의 음질과 정확성을 개선했습니다.  FLEURS-R은 저자원 언어 (low-resource languages)에서 음성 기술 발전을 목표로 하며, 텍스트 음성 변환 (TTS) 및 기타 음성 생성 작업 연구를 활성화하기 위해 고안되었습니다.



### On Effects of Steering Latent Representation for Large Language Model Unlearning (https://arxiv.org/abs/2408.06223)
Comments:
          15 pages, 5 figures, 8 tables

- **What's New**: 이 논문은 Representation Misdirection for Unlearning (RMU)라는 효과적인 LLM 언러닝(unlearning) 방법의 작동 원리를 탐구합니다. RMU는 모델 표현을 중간 계층에서 임의의 표현으로 유도하여 언러닝을 수행합니다. 이 논문에서는 RMU의 효과를 이론적으로 분석하고, RMU가 언러닝에 미치는 영향을 설명합니다. 또한,  RMU의 한계점을 해결하기 위해 Adaptive RMU라는 새로운 방법을 제안합니다. Adaptive RMU는 RMU의 효과를 향상시키고, 대부분의 계층에서 효과적인 언러닝을 가능하게 합니다.



### Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers (https://arxiv.org/abs/2408.06195)
- **What's New**: rStar는  파인튜닝이나 우수한 모델 없이도 소형 언어 모델(SLM)의 추론 능력을 크게 향상시키는 셀프 플레이 상호 추론 방법입니다. rStar는 셀프 플레이 상호 생성-판별 프로세스를 통해 추론을 분리합니다. 먼저, 대상 SLM은  Monte Carlo Tree Search (MCTS)에 다양한 인간과 같은 추론 액션을 추가하여 더 고품질의 추론 경로를 구축합니다. 다음으로, 대상 SLM과 유사한 능력을 가진 또 다른 SLM은 대상 SLM이 생성한 각 경로를 검증하는 판별자 역할을 합니다. 상호 합의된 추론 경로는 상호 일관성이 있다고 간주되므로, 더 정확할 가능성이 높습니다.  다섯 개의 SLM에 걸친 광범위한 실험 결과, rStar는 GSM8K, GSM-Hard, MATH, SVAMP, StrategyQA를 포함한 다양한 추론 문제를 효과적으로 해결할 수 있음을 보여줍니다. 특히, rStar는 LLaMA2-7B의 GSM8K 정확도를 12.51%에서 63.91%로, Mistral-7B의 정확도를 36.46%에서 81.88%로, LLaMA3-8B-Instruct의 정확도를 74.53%에서 91.13%로 향상시킵니다. 코드는 [이 URL](https://...)에서 제공됩니다.



### Improving Structural Diversity of Blackbox LLMs via Chain-of-Specification Prompting (https://arxiv.org/abs/2408.06186)
- **What's New**: 이 논문은 텍스트 생성에서 **구조적 다양성 (structural diversity)** 개념을 도입하여 사용자가 특정 구조적 특징에 대한 다양성을 제어할 수 있도록 합니다. 또한 **사양 체인 (chain-of-specification, CoS)** 프롬프팅이라는 새로운 방법을 제시하여 다양한 구조적 특징을 가진 텍스트를 생성합니다. CoS 프롬프팅은 먼저 LLM이 구조적 특징의 예를 생성하도록 한 다음, 해당 특징을 만족하는 텍스트를 생성하도록 합니다.

- **Technical Details**: **구조적 다양성**은 사용자가 정의한 특징 매핑 함수 𝜙:𝒳→𝒮  (𝜙(x)∈𝒮) 를 통해 텍스트 (x∈𝒳)를 특징 벡터로 매핑하고, 이 벡터들의 엔트로피를 측정하여 다양성을 평가합니다. 𝜙 함수는 사용자가 원하는 구조적 특징 (예: 시의 운율, 코드의 패러다임 등)을 반영합니다. **사양 체인 프롬프팅**은 LLM이 먼저 사양 (specification)을 생성하고, 이 사양을 만족하는 텍스트를 생성하는 과정을 반복적으로 수행하여 다양성을 높입니다. 각 사양은 특정 구조적 특징을 나타냅니다.

- **Performance Highlights**: 실험 결과, CoS 프롬프팅은 시, 코드 생성, 코딩 문제 생성 등 다양한 분야에서 기존 다양성 향상 기법보다 구조적 다양성을 크게 향상시키는 것으로 나타났습니다. 특히, 기존 방법들이 n-gram이나 BERT 임베딩 기반의 다양성 측정에 초점을 맞춘 반면, 구조적 다양성은 사용자가 정의한 구조적 특징에 대한 다양성을 측정하여 기존 방법과는 다른 차원의 다양성을 포착하는 것으로 나타났습니다.



### LipidBERT: A Lipid Language Model Pre-trained on METiS de novo Lipid Library (https://arxiv.org/abs/2408.06150)
- **What's New**: METiS 연구팀은 1천만 개의 가상 지질 데이터베이스를 구축하고 이를 활용하여 LNP(Lipid Nanoparticle, 지질 나노 입자) 특성 예측 성능을 향상시키는 LipidBERT 모델을 개발했습니다.

- **Technical Details**: LipidBERT는 BERT와 유사한 모델로, MLM(Masked Language Model)과 다양한 보조 작업을 통해 사전 훈련되었습니다. LipidBERT는 실험실 데이터를 활용하여 LNP 특성 예측을 위한 미세 조정을 수행하며, 이는 실제 LNP 데이터로의 지식 전이를 가능하게 합니다.

- **Performance Highlights**: LipidBERT는 LNP 특성 예측에서 뛰어난 성능을 보여주며, 실험실 데이터를 활용하여 높은 R2 점수(0.9 이상)를 달성했습니다. 이는 가상 지질 데이터베이스를 활용한 사전 훈련된 언어 모델의 효과를 입증하는 것입니다.

- **Dataset**: 1천만 개의 가상 지질 데이터베이스는 METiS의 자체 개발 알고리즘과 가상 스크리닝 기술을 통해 생성되었으며, 이는 LipidBERT 모델의 사전 훈련, 지질 표현 학습, 하류 작업 지식 전이에 활용됩니다.

- **Applications**: LipidBERT는 새로운 LNP 라이브러리 생성, 새로운 LNP 후보 물질 발굴, 특히 기관 표적 LNP에 대한 생체 내 테스트 후보 발굴 등 다양한 작업에 적용될 수 있습니다.

- **Contributions**: 이 연구는 가상 지질 데이터베이스를 활용한 사전 훈련된 언어 모델의 LNP 특성 예측 효과를 처음으로 입증하였으며, 실험실 데이터와의 통합을 통해 인공 지능 기반 LNP 개발에 새로운 가능성을 제시합니다.



### Med42-v2: A Suite of Clinical LLMs (https://arxiv.org/abs/2408.06142)
- **What's New**: Med42-v2는 의료 분야에서 일반적인 모델의 한계를 해결하도록 설계된 의료 전문 대규모 언어 모델(LLM) 모음입니다. 이 모델은 Llama3 아키텍처를 기반으로 구축되었으며 전문 의료 데이터를 사용하여 미세 조정되었습니다. 이 모델은 자연스러운 프롬프트에 효과적으로 응답하도록 다단계 선호도 정렬을 거쳤습니다. 일반적인 모델은 종종 예방 조치로 의료 질문에 답변하는 것을 피하도록 선호도가 정렬되지만, Med42-v2는 특히 이러한 한계를 극복하도록 훈련되어 의료 환경에서 사용할 수 있습니다. Med42-v2 모델은 8B 및 70B 매개변수 구성에서 원래 Llama3 모델과 GPT-4를 능가하여 다양한 의료 벤치마크에서 뛰어난 성능을 보여줍니다. 이러한 LLM은 의료 질문을 이해하고, 추론 작업을 수행하며, 의료 환경에서 귀중한 지원을 제공하도록 개발되었습니다.

- **Technical Details**: Med42-v2는 Llama3 아키텍처를 기반으로 구축된 의료 전문 대규모 언어 모델(LLM) 모음입니다. Med42-v2는 의료 관련 쿼리에 대한 응답 능력을 향상시키기 위해 전문 의료 데이터를 사용하여 미세 조정되었습니다. 이 모델은 또한 사용자 기대에 맞게 출력을 조정하기 위해 다단계 선호도 정렬을 거쳤습니다. 이는 의료 관련 질문에 답변하기를 꺼리는 일반적인 LLM과 대조적입니다. 이러한 훈련 과정은 Med42-v2가 의료 분야에서 사용하기 적합하도록 만들었습니다.

- **Performance Highlights**: Med42-v2는 8B 및 70B 매개변수 구성에서 원래 Llama3 모델과 GPT-4를 능가하여 다양한 의료 벤치마크에서 뛰어난 성능을 보여줍니다. 이 모델은 의료 질문을 이해하고, 추론 작업을 수행하며, 의료 환경에서 귀중한 지원을 제공하도록 설계되었습니다.



### Utilize Transformers for translating Wikipedia category names (https://arxiv.org/abs/2408.06124)
Comments:
          5 pages, 1 figure

- **What's New**: 본 논문에서는 영어 위키피디아 카테고리를 베트남어로 자동 번역하는 언어 모델을 구축하여, 수동 카테고리 생성의 노력을 줄이고 번역 작업에 필요한 컴퓨팅 리소스를 줄이는 대안 솔루션을 제시합니다. 특히, 15,000개의 영어-베트남어 카테고리 쌍 데이터셋을 사용하여, sequence-to-sequence 아키텍처를 가진 작은 규모의 Transformer 사전 훈련 모델을 미세 조정했습니다. OPUS-MT-en-vi 모델이 가장 뛰어난 성능을 보여주었고, BLEU 점수가 0.73으로 다른 모델보다 높았으며, 모델 저장 공간도 작았습니다.



### How ChatGPT Changed the Media's Narratives on AI: A Semi-Automated Narrative Analysis Through Frame Semantics (https://arxiv.org/abs/2408.06120)
Comments:
          18 pages, 6 figures and 2 appendices (5 pages)

- **What's New**: 이 연구는 OpenAI의 챗봇 ChatGPT 출시를 중심으로 한 12개월 동안 AI 관련 뉴스 기사 5,846개에서 수집된 49,000개 이상의 문장을 분석하여 AI에 대한 미디어 담론의 변화를 조사했습니다. 분석 결과, ChatGPT 출시 이후 6개월 동안 AI에 대한 미디어 관심도가 기존의 높은 수준에서 10배나 증가했음을 보여줍니다.



### Building Decision Making Models Through Language Model Regim (https://arxiv.org/abs/2408.06087)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 일반화 능력을 활용하여 의사 결정 문제를 해결하는 새로운 접근 방식인 "Learning then Using (LTU)"를 제안합니다. LTU는 전통적인 의사 결정 모델 훈련 방식과 달리, 다양한 도메인과 맥락에서 얻은 지식을 통합하여 기초 의사 결정 모델을 구축하는 "학습(learning)" 단계와 특정 의사 결정 시나리오에 맞게 이 기초 모델을 미세 조정하는 "사용(using)" 단계로 구성됩니다. 기존의 지도 학습 방식과 달리, LTU는 광범위한 사전 훈련과 목표 지향적인 미세 조정을 결합하여 다재다능한 훈련 방법론을 제공합니다.

- **Technical Details**: LTU는 Llama-2-13b (LLM)를 기반으로 하며, 인과적 언어 모델링(CLM) 훈련 방식과 트랜스포머 아키텍처를 사용합니다. 학습 단계에서는 다양한 의사 결정 도메인과 맥락에서 얻은 지식을 활용하여 기초 의사 결정 모델을 훈련합니다. 사용 단계에서는 특정 의사 결정 작업에 맞게 이 기초 모델을 미세 조정합니다.

- **Performance Highlights**: 전자 상거래 도메인(광고 및 검색 최적화)에서 실험한 결과, LTU는 기존의 지도 학습 방식보다 의사 결정 능력과 일반화 성능이 뛰어난 것으로 나타났습니다. 특히, LTU는 다양한 의사 결정 작업에 대해 지속적으로 우수한 성능을 보였으며, 단일 단계 및 다단계 의사 결정 작업에 적용 가능합니다. LTU는 게임 및 로봇 도메인을 넘어 다양한 의사 결정 과제를 해결하는 데 유연하고 강력한 프레임워크를 제공합니다.



### An Investigation Into Explainable Audio Hate Speech Detection (https://arxiv.org/abs/2408.06065)
Comments:
          Accepted to SIGDIAL 2024

- **What's New**: This research introduces a novel task: **explainable audio hate speech detection**. It aims to identify hate speech in audio recordings while pinpointing the precise time intervals (audio frame-level rationales) that justify the classification.



### On Tables with Numbers, with Numbers (https://arxiv.org/abs/2408.06062)
- **What's New**: 이 논문은 숫자 테이블에 대한 현대 계산 언어학의 집착을 비판적으로 반추하고 있습니다. 이는 숫자 테이블의 인식론적 무관성, 환경적 영향, 사회적 불평등을 심화시키는 역할, 그리고 상업적 응용 및 이익 중심 연구와의 깊은 연관성을 근거로 주장합니다. 이 논문은 지난 10년간 계산 언어학 연구에 대한 메타 분석에서 얻은 경험적 증거로 주장을 뒷받침합니다.



### DiagESC: Dialogue Synthesis for Integrating Depression Diagnosis into Emotional Support Conversation (https://arxiv.org/abs/2408.06044)
Comments:
          Accepted by SIGDIAL 2024

- **What's New**: 이 논문은 진단 감정적 지원 대화(DiagESC)라는 새로운 과제를 소개합니다. 이 과제는 사용자에게 적절한 정신 건강 지원을 제공하는 동시에 우울증과 같은 정신 건강 문제를 조기에 진단하는 것을 목표로 합니다. 이 논문에서는 DiagESC 과제를 위한 새로운 데이터셋인 DESC를 제시합니다.



### Enhancing Dialogue Speech Recognition with Robust Contextual Awareness via Noise Representation Learning (https://arxiv.org/abs/2408.06043)
Comments:
          11 pages, 2 figures, Accepted to SIGDIAL2024

- **What's New**: 이 논문에서는 잡음이 많은 컨텍스트 내에서도 정확한 컨텍스트 정보를 인코딩하여 대화 음성 인식의 정확성을 높이는 새로운 방법인 컨텍스트 노이즈 표현 학습(CNRL)을 소개합니다. CNRL은 텍스트 기반 대화 데이터를 이용한 디코더 사전 훈련과 컨텍스트 인코더를 위한 노이즈 표현 학습을 통합하여 잡음이 많은 컨텍스트에 대한 로버스트성을 강화합니다. 특히 실제 환경에서 사용자 음성이 거의 들리지 않는 잡음 환경에서도 컨텍스트 정보를 활용하여 정확하게 입력을 전사할 수 있다는 강점을 보여줍니다.



### The Language of Trauma: Modeling Traumatic Event Descriptions Across Domains with Explainable AI (https://arxiv.org/abs/2408.05977)
- **What's New**: 본 연구는 다양한 온라인 컨텍스트에서 나타나는 심리적 트라우마를 포괄적으로 분석하기 위해 NLP(Natural Language Processing)와 XAI(Explainable Artificial Intelligence) 기법을 활용한 혁신적인 접근 방식을 제시합니다. 이는 기존 연구에서 주로 단일 트라우마 유형에 집중했던 한계를 극복하고, 다양한 트라우마 상황에 대한 일반화 가능성을 높이려는 시도입니다.



### ConvKGYarn: Spinning Configurable and Scalable Conversational Knowledge Graph QA datasets with Large Language Models (https://arxiv.org/abs/2408.05948)
- **What's New**: 본 논문은 대규모 언어 모델(LLM) 및 대화형 비서의 급속한 발전으로 인해 훈련 및 평가를 위한 역동적이고 확장 가능하며 구성 가능한 대화형 데이터 세트의 필요성이 증가했음을 강조합니다. 이러한 데이터 세트는 텍스트 및 음성을 포함한 다양한 사용자 상호 작용 모드를 수용해야 하며, 각 모드는 고유한 모델링 과제를 제시합니다. 구조적이고 진화하는 특성을 지닌 지식 그래프(KG)는 현재 및 정확한 지식에 대한 이상적인 기반을 제공합니다. 인간이 큐레이팅한 KG 기반 대화형 데이터 세트가 존재하지만, 급변하는 사용자 정보 요구 사항을 따라잡기 어려움이 있습니다. 이 논문에서는 최신의 구성 가능한 대화형 KGQA 데이터 세트를 생성하기 위한 확장 가능한 방법인 ConvKGYarn을 제시합니다. 질적 심리 측정 분석은 이 방법이 인기 있는 대화형 KGQA 데이터 세트에 필적하는 고품질 데이터 세트를 생성할 수 있음을 확인하면서 동시에 규모를 확장하고 다양한 인간 상호 작용 구성을 다룰 수 있습니다. 이 논문에서는 다양한 대화에서 LLM을 테스트하여 동일한 KG 팩트 세트를 기반으로 하는 다양한 구성을 갖춘 대화형 KGQA 세트에서 모델 동작을 탐구함으로써 유용성을 보여줍니다. 결과는 ConvKGYarn이 KGQA 기반을 개선하고 LLM의 매개 변수 지식을 평가할 수 있는 능력을 강조하여 대화형 비서의 끊임없이 진화하는 환경에 대한 강력한 솔루션을 제공합니다.



### A New Pipeline For Generating Instruction Dataset via RAG and Self Fine-Tuning (https://arxiv.org/abs/2408.05911)
Comments:
          5 pages, SCA 2024: The 7th IEEE International Workshop on Smart Computing & Applications

- **What's New**: 본 연구에서는 특정 도메인에 맞는 고품질 지침 데이터셋을 생성하는 파이프라인을 제안합니다. 이 파이프라인은 특정 도메인의 문서 컬렉션을 사용하여 LLM과 Retrieval-Augmented Generation(RAG) 프레임워크를 활용하여 지침을 생성합니다. 이를 통해 기존 수동 방식이나 웹 스크래핑으로 인한 오류 가능성을 제거하고 도메인 특화 모델을 효과적으로 구축할 수 있습니다.



### AdTEC: A Unified Benchmark for Evaluating Text Quality in Search Engine Advertising (https://arxiv.org/abs/2408.05906)
- **What's New**: This paper introduces AdTEC, the first public benchmark designed for evaluating the quality of advertisement text generated by Natural Language Generation (NLG) models in a real-world advertising setting.



### Creating Arabic LLM Prompts at Sca (https://arxiv.org/abs/2408.05882)
- **What's New**: This paper presents two novel methods for efficiently creating Arabic language prompts for training instruction-following Large Language Models (LLMs) in Arabic, a language where prompt generation research has been limited. The first method adapts existing English prompt datasets by translating them into Arabic and employing machine translation quality estimation to retain high-quality prompts. The second method leverages existing Arabic NLP datasets to generate prompts tailored to specific tasks. Using these methods, the authors created over 87 million Arabic prompts, covering diverse tasks such as summarization, question answering, grammar checking, and creative writing.

- **Technical Details**: The paper details the methods for generating Arabic prompts: 
1. **PromptSource Adaptation**:  Prompts are created for 78 Arabic NLP datasets using the PromptSource tool, generating over 67 million prompts. 
2. **English Prompt Translation**: Existing English prompt datasets (PromptSource and Super-NaturalInstructions) are translated into Arabic using the Opus-MT model and filtered using COMET-QE for quality estimation. Manual verification is employed to ensure high quality. This resulted in roughly 20 million prompts. 
The paper also describes the training process using a 7 billion parameter open LLM, Qwen2 7B, fine-tuned on the created prompts. Two training datasets (800,000 prompts and 8 million prompts) are used to assess the impact of data size on fine-tuning performance.

- **Performance Highlights**: The fine-tuned Qwen2 7B model outperforms even a larger 70 billion parameter model, Llama3-Instruct 70B, on Arabic prompts. The paper emphasizes the focus on instruction following ability, tested through diverse tasks, rather than knowledge-based benchmarks like MMLU and Hellaswag.



### LLM-Based Robust Product Classification in Commerce and Complianc (https://arxiv.org/abs/2408.05874)
Comments:
          11 pages

- **What's New**: 이 연구는 실제 세계 제품 분류의 어려움, 특히 제품 설명의 불완전성과 축약에 초점을 맞추어 진행되었습니다. 제품 분류의 정확성은 국제 무역에서 매우 중요하며, 잘못된 분류는 세금 및 관세 책임으로 이어질 수 있습니다. 이 연구는 GPT-4와 같은 강력한 대규모 언어 모델(LLM)을 사용하여 실제 세계 데이터 불완전성을 모방하는 데이터 왜곡(perturbation)을 만들어냅니다. 이러한 왜곡된 데이터는 LLM 기반 제품 분류 모델을 훈련하는 데 사용되어, 깨끗한 데이터에서만 훈련된 기존 감독 학습 모델보다 더 강력한 성능을 보여줍니다.



### Defining Boundaries: A Spectrum of Task Feasibility for Large Language Models (https://arxiv.org/abs/2408.05873)
Comments:
          20 pages, 9 tables, 15 Figures

- **What's New**: 이 연구는 대규모 언어 모델(LLM)이 자신의 능력을 넘어서는 작업을 수행할 때 거부하는 능력을 평가하는 새로운 벤치마크를 제시합니다. LLM이 처리할 수 없는 작업을 체계적으로 정의하고 분류하여 다양한 작업의 가능성을 평가합니다. 또한, 거부 능력을 향상시키는 훈련 방식을 탐구하고 벤치마크 데이터셋을 개발합니다.



### Iterative Improvement of an Additively Regularized Topic Mod (https://arxiv.org/abs/2408.05840)
Comments:
          A full draft of the second version of the article

- **What's New**: 이 논문은 이전에 발견된 모든 좋은 주제를 유지하여 이전 모델보다 좋거나 동등한 일련의 관련 주제 모델을 학습하는 주제 모델의 반복적 학습 방법을 제시합니다. 이 방법의 핵심은 각 후속 모델이 이전 모델보다 좋다는 것을 보장하기 위해 (즉, 이전에 발견된 모든 좋은 주제를 유지하기 위해) 각 후속 모델이 이전 모델보다 적어도 좋거나 동등한 일련의 관련 주제 모델을 학습하는 것입니다. 이러한 모델 간의 연결은 추가 정규화를 통해 달성됩니다. 이러한 반복적인 학습의 결과는 일련의 마지막 주제 모델이며, 이를 반복적으로 업데이트된 추가적으로 정규화된 주제 모델(ITAR)이라고 합니다.



### SAGA: A Participant-specific Examination of Story Alternatives and Goal Applicability for a Deeper Understanding of Complex Events (https://arxiv.org/abs/2408.05793)
Comments:
          Accepted to Findings of the Association for Computational Linguistics 2024

- **What's New**: 본 논문은 복잡한 사건을 이해하고 추론하는 데 필수적인 목표 지향적 행동을 해석하고 평가하는 새로운 접근 방식을 제시합니다. 이러한 이해에 필요한 지식을 획득하는 것은 어려운 일이지만, 본 논문은 참여자 달성 렌즈를 통해 이러한 지식을 얻을 수 있다고 주장합니다. 본 연구는 서사 속 참여자의 의도된 달성, 예상되는 미래 행동, 목표 달성 가능성을 분석하여 서사 속 복잡한 사건을 분석합니다. 연구진은 제안된 참여자 달성 렌즈를 반영하여 6,300개의 고품질 목표 및 행동 주석을 수집했으며, 평균 가중치 Fleiss-Kappa IAA는 80%입니다. 본 수집에는 각 서사의 대안 버전에 대한 주석이 포함되어 있습니다. 이러한 대안 버전은 '원본' 이야기와 미세하게 다르지만, 완전히 다른 추론을 허용할 수 있습니다. 연구 결과에 따르면 최신 대규모 언어 모델(LLMs)은 본 연구에서 다루는 목표 기반 지식을 어느 정도 반영할 수 있지만, 모델 사전 학습에 목표 지식을 추출한 데이터가 포함되어 있더라도, 공동 행동의 설계와 의도를 완전히 포착하는 데 어려움을 겪는 것으로 나타났습니다. 연구진은 본 데이터셋에서 미세 조정된 소규모 모델이 더 큰 모델보다 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구는 서사 속 참여자의 목표를 이해하고 추론하는 복잡한 작업을 단순화하기 위해 참여자가 취하는 또는 미래에 취할 수 있는 행동, 의도, 계획으로 분해합니다. 연구진은 여러 단계 주석 파이프라인을 사용하여 6,225개의 목표 주석 세트(886개의 '실제' 및 951개의 '대안' 이야기 포함)를 수집했습니다. 이 풍부한 데이터를 통해 연구진은 그림 2에 표시된 것처럼 어려움이 증가하는 여러 추론 작업을 공식화했습니다. 이 작업에는 참여자의 목표(작업 1 및 2), 목표 결과에 영향을 미치는 미래 행동(작업 3 및 4), 달성(작업 5)에 대해 추론할 수 있는 내용이 포함됩니다. 연구진은 GPT 버전 4, 3.5-Turbo 및 T5 모델 계열의 다양한 크기의 Flan-T5 및 T5 모델을 비교하여 이러한 작업에 대해 LLM을 조사했습니다. 연구진은 프롬프팅을 사용하고 수집된 데이터셋에서 모델을 미세 조정하여 두 옵션과 소규모 모델을 미세 조정하는 이점을 조사했습니다. 연구 결과 및 분석은 이러한 다양한 모델의 강점과 약점을 보여줍니다.

- **Performance Highlights**: 사전 훈련된 대규모 모델은 일반적으로 소규모 모델보다 성능이 좋으며 사실적 오류가 적습니다. 몇 가지 예시 프롬프팅은 특히 이야기 내 세부 정보가 필요한 작업에 유용합니다. 본 데이터셋의 일부 이야기는 Flan-T5에 대한 사전 학습 데이터의 일부이며, 연구진은 모델이 이러한 이야기의 미묘한 변화를 어떻게 처리하는지 조사했으며, 몇 가지 예시 프롬프팅 및 미세 조정이 이러한 오류를 수정하는 데 도움이 되는 것으로 나타났습니다. 전반적으로 연구진은 가장 큰 모델조차도 일련의 행동 뒤에 숨은 인간의 의도를 정확히 파악하는 데 어려움을 겪는다는 것을 발견했습니다. 미세 조정 및 몇 가지 예시 프롬프팅은 소규모 모델을 개선하여 더 큰 모델과 경쟁하거나 그 이상의 성능을 발휘하게 합니다.



### HiLight: A Hierarchy-aware Light Global Model with Hierarchical Local ConTrastive Learning (https://arxiv.org/abs/2408.05786)
- **What's New**: 본 논문에서는 계층적 텍스트 분류(HTC)를 위한 새로운 경량형 모델인 HiLight를 제안합니다. HiLight는 기존의 복잡한 구조 인코더(structure encoder) 없이 텍스트 인코더(text encoder)와 멀티 레이블 분류 헤드(multi-label classification head)만을 사용하며, HiLCL(Hierarchical Local Contrastive Learning)이라는 새로운 학습 전략을 도입하여 계층적 정보를 효과적으로 학습합니다. HiLCL은 LCL(Local Contrastive Learning)과 HiLearn(Hierarchical Learning) 전략으로 구성되어, 동일한 경로(path) 상의 분류기(classifier)가 유사한 방향으로 판별력을 향상시키도록 설계되었습니다.



### LI-TTA: Language Informed Test-Time Adaptation for Automatic Speech Recognition (https://arxiv.org/abs/2408.05769)
Comments:
          INTERSPEECH 2024

- **What's New**: This paper introduces **Language Informed Test-Time Adaptation (LI-TTA)**, a novel method for improving Automatic Speech Recognition (ASR) in scenarios with domain shifts. Unlike previous TTA methods that focused primarily on acoustic features, LI-TTA incorporates linguistic information from an external language model (LM).



### Reference-free Hallucination Detection for Large Vision-Language Models (https://arxiv.org/abs/2408.05767)
- **What's New**: 이 연구는 참조 없는 (reference-free) 방법이 LVLMs (Large Vision-Language Models)의 환각 (hallucination)을 효과적으로 감지할 수 있는지 조사합니다. 특히, 불확실성 기반 (uncertainty-based), 일관성 기반 (consistency-based), 그리고 감독된 불확실성 정량화 (Supervised Uncertainty Quantification, SUQ) 방법이라는 세 가지 유형의 기술을 사용하여 연구를 수행합니다.

- **Technical Details**: 연구에서는 네 가지 대표적인 LVLMs을 사용하여 두 가지 다른 작업 (Yes-and-No, Open-ended)에 대해 광범위한 실험을 수행했습니다. 불확실성 기반 방법에는 AvgProb, AvgEnt, MaxProb, MaxEnt의 네 가지 지표가 사용되었습니다. 일관성 기반 방법에는 BERTScore, 질문 답변 (Question Answering, QA), Unigram, 자연어 추론 (Natural Language Inference, NLI)의 네 가지 변형이 사용되었습니다. SUQ 방법은 모델의 내부 상태를 분석하여 진술의 신뢰성을 예측하는 분류기를 훈련합니다.

- **Performance Highlights**: 실험 결과는 SUQ 방법이 다른 접근 방식보다 환각 감지 성능이 뛰어나다는 것을 보여줍니다. 특히 SUQ 방법은 문장 및 구절 수준에서 모두 우수한 성능을 보였습니다. 일관성 기반 방법은 불확실성 기반 방법보다 뛰어났지만 SUQ 방법보다는 낮은 성능을 보였습니다.

- **Contributions**: 이 논문은 다음과 같은 주요 기여를 합니다: - 다양한 참조 없는 방법의 환각 감지 성능을 포괄적으로 측정합니다. - 감독된 불확실성 정량화 (SUQ) 방법이 다양한 설정에서 최상의 성능을 보임을 보여줍니다. - LLaVA-v1.5-7b를 사용하여 수동으로 주석이 달린 문장 수준의 데이터셋인 Image-Hallucination Annotation Dataset (IHAD)를 제공합니다.



### Language-Informed Beam Search Decoding for Multilingual Machine Translation (https://arxiv.org/abs/2408.05738)
Comments:
          ACL 2024 Findings

- **What's New**: 이 논문에서는 다국어 NMT 모델의 빔 검색 디코딩 과정에서 발생하는 '오프-타겟(off-target)' 번역 오류를 분석하고, 오류를 줄이기 위한 새로운 디코딩 알고리즘인 '언어 정보 빔 검색(Language-informed Beam Search, LiBS)'을 제안합니다. LiBS는 빔 검색 디코딩에 기존의 언어 식별(Language Identification, LiD) 모델을 통합하여 오프-타겟 번역을 줄이는 일반적인 디코딩 알고리즘입니다.

- **Technical Details**: LiBS는 NMT 모델과 독립적으로 작동하며 추가적인 병렬 데이터 없이 추론 시에 적용 가능한 알고리즘입니다. LiBS는 LiD 모델을 사용하여 빔 검색 디코딩 과정에서 생성된 각 후보 번역의 언어를 예측하고, 목표 언어와 일치하는 후보 번역을 더 높은 확률로 선택합니다.

- **Performance Highlights**: WMT 및 OPUS 데이터셋에서 실험한 결과, LiBS는 오프-타겟 비율을 각각 22.9%에서 7.7%, 65.8%에서 25.3%로 감소시켰으며, BLEU 점수는 각각 +1.1, +0.9 개선되었습니다. 이는 LiBS가 기존 다국어 모델에 추가적으로 적용되어 오프-타겟 번역 문제를 효과적으로 해결할 수 있음을 의미합니다.



### Training an NLP Scholar at a Small Liberal Arts College: A Backwards Designed Course Proposa (https://arxiv.org/abs/2408.05664)
Comments:
          9 pages, Presented at 6th Workshop on Teaching NLP

- **What's New**: 본 논문은 NLP 코스를 통해 양성할 수 있는 두 가지 유형의 학생을 제시합니다. 첫째, 다양한 작업을 위해 NLP의 새로운 기술을 유연하게 설계, 구축 및 적용할 수 있는 'NLP 엔지니어'입니다. 둘째, NLP와 사회의 관계에 대한 질문을 제기하고, 개선하고, 답변하고, 이러한 답변을 더 넓은 청중에게 효과적으로 전달할 수 있는 'NLP 학자'입니다. 이 두 가지 유형의 기술은 상호 배타적이지 않지만, NLP 엔지니어는 비판적으로 사고해야 하고, NLP 학자는 시스템을 구축할 수 있어야 합니다. 본 논문은 NLP 학자가 갖추어야 할 기술 유형을 명확히 하고, 이러한 기술 습득을 돕는 교육 과정 구성 요소를 제안합니다.



### WiDe-analysis: Enabling One-click Content Moderation Analysis on Wikipedia's Articles for Deletion (https://arxiv.org/abs/2408.05655)
Comments:
          System Demonstration

- **What's New**: 이 논문은 위키피디아 삭제 토론 데이터셋을 위한 새로운 Python 패키지인 wide-analysis를 소개합니다. wide-analysis는 데이터 수집 및 전처리, 토론 분석 (댓글 및 토론 수준), 일반 목적 시스템 (예: 감정 분석 또는 공격적 언어 감지) 및 위키피디아 관련 분석기 (정책 예측 또는 입장 감지)를 통한 콘텐츠 분석을 위한 기능을 제공합니다. 이 패키지는 위키피디아 콘텐츠 조절 연구를 가속화하기 위해 데이터, 모델 및 Python 패키지와 함께 출시됩니다.



### Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion (https://arxiv.org/abs/2408.05636)
- **What's New**: This paper introduces a new method called *Speculative Diffusion Decoding (SpecDiff)*, which combines speculative decoding with discrete diffusion models for faster and more efficient large language model inference.

- **Technical Details**: SpecDiff utilizes discrete diffusion models to generate draft sequences, enabling parallel processing of both drafting and verification steps, which significantly accelerates the inference process.  Unlike traditional speculative decoding, which relies on incremental token generation, SpecDiff leverages the ability of diffusion models to generate entire sequences in a single step.  This approach addresses the limitations of previous speculative decoding methods by improving both the speed and quality of the draft sequences.

- **Performance Highlights**: SpecDiff demonstrates a substantial speed-up compared to standard generation processes and existing speculative decoding techniques.  It achieves up to 8.7x faster inference compared to standard generation and up to 2.5x faster than traditional speculative decoding approaches.  This improvement is attributed to the parallel drafting and verification processes enabled by diffusion models.



### Document-Level Event Extraction with Definition-Driven ICL (https://arxiv.org/abs/2408.05566)
- **What's New**: 본 논문에서는 문서 수준 이벤트 추출에서 LLM의 프롬프트 엔지니어링 문제를 해결하기 위한 새로운 최적화 전략인 "정의 기반 문서 수준 이벤트 추출(DDEE)"를 제안합니다. 이 전략은 프롬프트 길이를 조정하고 휴리스틱의 명확성을 강화하여 LLM의 이벤트 추출 성능을 향상시키고, 데이터 균형 기술을 사용하여 롱테일 효과 문제를 해결함으로써 모델의 이벤트 유형에 대한 일반화 능력을 강화했습니다. 동시에, LLM의 프롬프트 스타일 민감도에 맞춰 간결하고 포괄적인 프롬프트를 개선했고, 구조화된 휴리스틱 방법과 엄격한 제한 조건을 도입하여 이벤트 및 인수 역할 추출의 정확성을 향상시켰습니다. 이러한 전략은 문서 수준 이벤트 추출에서 LLM의 프롬프트 엔지니어링 문제를 해결할 뿐만 아니라 이벤트 추출 기술의 발전을 촉진하고 NLP 분야의 다른 작업에 대한 새로운 연구 관점을 제공합니다.



### Large Language Model-based Role-Playing for Personalized Medical Jargon Extraction (https://arxiv.org/abs/2408.05555)
Comments:
          17 pages, 3 figures, 3 tables

- **What's New**: 본 연구는 대규모 언어 모델(LLM)에서 역할극(role-playing)을 적용하여 개인 맞춤형 의료 용어 추출 성능을 향상시킬 수 있는지 조사했습니다. 특히, 사회인구학적 배경에 따라 개인화된 의료 용어 추출을 위한 ChatGPT의 역할극 효과를 정량적으로 분석했습니다.



### Multi-layer Sequence Labeling-based Joint Biomedical Event Extraction (https://arxiv.org/abs/2408.05545)
Comments:
          13 pages, 3 figures, accepted by NLPCC2024

- **What's New**: MLSL, a novel method based on multi-layer sequence labeling for joint biomedical event extraction, is proposed in this paper. It utilizes the information of candidate trigger words explicitly for a simplified workflow with no prior knowledge or complex structures.



### P3: A Policy-Driven, Pace-Adaptive, and Diversity-Promoted Framework for Optimizing LLM Training (https://arxiv.org/abs/2408.05541)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 미세 조정을 개선하기 위해 작업별 데이터 가지치기(pruning) 및 선택에 중점을 둡니다. 제시된 새로운 프레임워크인 P3는 동적이고 적응적인 훈련 전략을 통해 LLM 성능을 향상시키는 것을 목표로 합니다. P3는 모델의 실시간 성능을 기반으로 데이터의 난이도를 측정하는 정책 기반 난이도 측정(Policy-driven Difficulty Measurement), 점진적으로 더 어려운 데이터를 선택하는 속도 적응형 선택(Pace-adaptive Selection) 및 샘플 내부 및 샘플 간 다양성을 촉진하는 Determinantal Point Process (DPP)를 통합한 다양성 촉진(Diversity Promotion)을 포함합니다.

- **Technical Details**: P3는 모델이 데이터를 얼마나 잘 처리하는지 측정하여 데이터의 난이도를 평가합니다. 모델이 힘들어하는 데이터는 더 어려운 것으로 간주됩니다. 그런 다음 모델은 난이도가 점차 증가하는 데이터를 선택하여 학습합니다. DPP는 훈련 세트의 다양성을 유지하여 모델이 다양한 유형의 데이터에서 일반화하도록 돕습니다.

- **Performance Highlights**: P3는 두 가지 LLM 데이터 세트인 APPS와 MATH에서 기존 방법보다 뛰어난 훈련 결과를 보였습니다. P3는 미세 조정 성능을 향상시키고 훈련 시간을 단축하는 효과적인 방법입니다.



### Context-Driven Index Trimming: A Data Quality Perspective to Enhancing Precision of RALMs (https://arxiv.org/abs/2408.05524)
- **What's New**: 이 논문은 Retrieval-Augmented Large Language Models (RALMs) 의 정확성을 향상시키는 새로운 방법인 Context-Driven Index Trimming (CDIT) 프레임워크를 제시합니다. CDIT는 벡터 데이터베이스에서 인덱스를 조정하여 컨텍스트와 일치하지 않는 검색 결과를 제거함으로써 RALMs의 응답 품질을 향상시킵니다.  기존의 vector-distance-based 검색 방법은  컨텍스트 일관성에 대한 고려 없이 유사한 벡터를 반환하는 한계를 가지고 있으며, 이는 RALMs의 정확성에 부정적인 영향을 미칩니다. CDIT는  Context Matching Dependencies (CMDs)를 사용하여  LLM(Large Language Model)의 의미 이해 능력을 활용하여 컨텍스트 일관성을 검증하고, 잘못된 검색 결과를 제거합니다.

- **Technical Details**: CDIT는 다음과 같은 기술적 특징을 가지고 있습니다:
1. **Context Matching Dependencies (CMDs):** 컨텍스트 일관성을 보장하는 논리적 데이터 품질 규칙.  
2. **Large Language Model (LLM):**  CMDs의  제약 조건을 평가하고 컨텍스트 일관성을 검사하는 데 사용됩니다.
3. **Vector Database:** 검색 결과의  인덱스를  조정하여 컨텍스트 일관성에 맞는 결과만 반환하도록 합니다.

- **Performance Highlights**: 실험 결과, CDIT는  다양한 언어 모델과 인덱싱 방법에서 평균 3.75%의 정확도 향상을 보였습니다. 또한, 기존의 IndexFlatL2, IndexHNSWFlat, IndexIVFFlat  인덱싱 방법에 비해 각각 3.44%, 4.07%, 3.75%의 정확도 향상을 보였습니다. 최대 15.21%의 성능 향상을 달성했습니다.



### SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning (https://arxiv.org/abs/2408.05517)
- **What's New**: SWIFT, an open-source framework for large model training and fine-tuning, has been developed to address the growing demand for efficient and adaptable solutions in this field.  It is the first training framework to provide systematic support for Multi-Modal Large Language Models (MLLMs).

- **Technical Details**: SWIFT offers a comprehensive infrastructure that integrates various training techniques and post-training processes. It supports over 300 LLMs and 50+ MLLMs, including models like Qwen-VL, GLM4-V, InternVL, and DiT. SWIFT leverages libraries such as PEFT and Optimum, and it incorporates features like lightweight training, quantization, and human alignment.

- **Performance Highlights**: Training with SWIFT on customized datasets achieved notable improvements on the ToolBench leaderboard, demonstrating the effectiveness of the framework. Results show an increase of 5.2%-21.8% in the Act.EM metric, a reduction in hallucination by 1.6%-14.1%, and an average performance improvement of 8%-17% over various baseline models.



### Your Context Is Not an Array: Unveiling Random Access Limitations in Transformers (https://arxiv.org/abs/2408.05506)
Comments:
          Published as a conference paper at COLM 2024

- **What's New**: 본 논문은 Transformer 기반 대규모 언어 모델이 알고리즘 작업에서 길이 일반화(length generalization)에 실패하는 근본적인 이유를 탐구합니다. 분석 결과, 이러한 실패는 모델이 컨텍스트 윈도우(context window) 내에서 랜덤 메모리 액세스(random memory access)를 수행하는 데 어려움을 겪는 것과 밀접하게 관련되어 있음을 밝혀냈습니다. 연구진은 이 가설을 뒷받침하기 위해 색인(indexing) 필요성을 우회하거나 콘텐츠 기반 주소 지정(content-based addressing)을 통해 간접적으로 랜덤 토큰 액세스를 가능하게 하는 방법론의 효과를 입증했습니다.

- **Technical Details**: 이 연구는 이진 패리티(binary parity) 작업을 사용하여 Transformer가 알고리즘 작업을 학습하는 과정을 심층적으로 분석했습니다. 패리티 작업은 컨텍스트 윈도우 내에서 랜덤 메모리 액세스를 요구하는 기본적인 작업이기 때문에 Transformer의 컴퓨팅 요구 사항을 연구하기에 적합합니다. 연구진은 다양한 위치 임베딩(positional embedding) 방법을 가진 모델에 대한 실증 연구를 통해 Transformer가 자연어를 사전 훈련하면서 콘텐츠 기반 주소 지정을 통해 토큰을 검색하는 능력을 학습하지만, 이는 랜덤 메모리 액세스에 의존하는 알고리즘 작업에서는 실패로 이어진다는 가설을 강력히 뒷받침했습니다.

- **Performance Highlights**: 연구진은 모델이 콘텐츠 기반 주소 지정을 활용하는 “mnemonics”를 추가하여 패리티 및 덧셈 작업에 대한 길이 일반화가 가능한 알고리즘을 학습할 수 있음을 보여주었습니다. 이러한 결과는 Transformer 모델에 효과적인 색인 기반 주소 지정 메커니즘을 제공하는 것이 길이 일반화가 가능한 알고리즘을 학습하는 데 중요한 역할을 할 수 있음을 시사합니다.



### MABR: A Multilayer Adversarial Bias Removal Approach Without Prior Bias Knowledg (https://arxiv.org/abs/2408.05497)
- **What's New**: 이 논문은 새로운 적대적 훈련 전략인 다층 적대적 편향 제거(MABR) 프레임워크를 소개하여 보호된 속성 라벨 없이 사회적 편향을 완화합니다. 이 방법은 각 레이어에서 주 모델의 인코더에 보조 분류기를 도입하여 편향을 감지하고 주 모델의 표현이 편향 분류기에 의해 식별된 편향에 대해 불변하도록 만듭니다.



### Investigating Instruction Tuning Large Language Models on Graphs (https://arxiv.org/abs/2408.05457)
Comments:
          COLM 2024

- **What's New**: 이 연구는 최신의 대규모 언어 모델(LLM)이 그래프 관련 작업에 적용될 수 있는 가능성을 살펴봅니다. 특히 LLM이 실제 그래프와 상호 작용하고 다양한 그래프 작업에서 일반화될 수 있는 방법에 대한 실증적인 통찰력을 제공하는 것을 목표로 합니다. 이를 위해 연구팀은 79개의 다양한 그래프 관련 작업(학문적 및 전자 상거래 도메인)을 포함하는 지침 조정 데이터 세트를 구축했습니다. 또한 연구팀은 LLM이 복잡한 그래프 구조를 이해하는 데 도움이 되는 최적의 그래프 표현 방식을 조사했습니다. 연구 결과, JSON 형식의 그래프 표현 방식이 다양한 LLM과 그래프 유형에서 자연어 및 코드 형식보다 일관되게 더 나은 성능을 보이는 것으로 나타났습니다. 마지막으로, 연구팀은 지침 조정된 LLM의 일반화 능력에 영향을 미치는 주요 요인을 도메인 내 및 도메인 외부 그래프 작업에 대한 성능을 평가하여 분석했습니다.



### Path-LLM: A Shortest-Path-based LLM Learning for Unified Graph Representation (https://arxiv.org/abs/2408.05456)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문은 **Path-LLM**이라는 새로운 그래프 표현 학습 모델을 제안합니다. Path-LLM은 **최단 경로(shortest path)**를 이용하여 그래프의 구조 정보를 효과적으로 학습하는 방식입니다. 특히,  **장단 최단 경로(L2SP)**를 이용하여 그래프 내에서 서로 다른 밀집 그룹 간 연결을 포착하는 새로운 메커니즘을 사용합니다. 이는 기존의 랜덤 워크(random walk) 방식보다 더 효과적으로 그래프 구조를 표현할 수 있습니다. 또한, Path-LLM은 **최단 경로 기반 텍스트화(path textualization)**를 통해 최단 경로 정보를 LLM이 이해할 수 있는 형태로 변환합니다. 이를 통해 **LLM의 강력한 의미 표현 능력(semantic representation capabilities)**을 활용하여 그래프 구조를 학습할 수 있습니다.



### Chain of Condition: Construct, Verify and Solve Conditions for Conditional Question Answering (https://arxiv.org/abs/2408.05442)
- **What's New**: 본 논문에서는 조건부 질의응답(Conditional Question Answering, CQA)을 위한 새로운 프롬프팅 방식인 '체인 오브 컨디션(Chain of Condition)'을 제안합니다. CQA는 사용자의 질문에 대한 답변을 찾고, 그 답변을 뒷받침하는 조건들을 파악하는 작업입니다. 기존 접근 방식들은 두 가지 주요 과제에 어려움을 겪고 있습니다. 첫째, 조건들을 정확하게 파악하고 그들 사이의 논리적 관계를 명확히 밝혀내는 것이 어렵습니다. 둘째, 조건들을 검증하고 해결하는 것이 어렵습니다. 제안된 체인 오브 컨디션은 이러한 과제를 해결하기 위해 다음과 같은 단계를 거칩니다. 1) 문서에서 모든 조건들을 파악하고 문서에 따라 그들의 논리적 관계를 명확히 구성합니다. 2) 이러한 조건들이 사용자에 의해 충족되었는지 검증합니다. 3) 도구를 사용하여 논리적 표현을 해결하고, 누락된 조건들을 표시하고, 해결된 조건들을 기반으로 답변을 생성합니다. 



### LaiDA: Linguistics-aware In-context Learning with Data Augmentation for Metaphor Components Identification (https://arxiv.org/abs/2408.05404)
Comments:
          This paper has been accepted by NLPCC 2024 Shared Tasks

- **What's New**: LaiDA (Linguistics-aware In-context Learning with Data Augmentation) - a novel LLM-based framework for Metaphor Components Identification (MCI) task, effectively recognizing metaphor components. It leverages ChatGPT for dataset construction and integrates linguistically similar examples into fine-tuning prompts, boosting performance.



### FiST-Financial Style Transfer with Hallucination and Creativity Control Framework (https://arxiv.org/abs/2408.05365)
Comments:
          8 pages, 13 figures, 5 tables, conference

- **What's New**: This paper proposes a novel two-stage fine-tuning process for large language models (LLMs) to improve financial report generation. This method addresses two major challenges: the lack of complex sentences and hallucinations in LLM-generated reports. The fine-tuning process involves training the LLM with pre-processed public domain financial reports and then fine-tuning it with simple prompts and tabular data inputs. The proposed method significantly reduces hallucinations and increases the number of correct answers, demonstrating its effectiveness in enhancing the quality of financial report generation.



### DataNarrative: Automated Data-Driven Storytelling with Visualizations and Texts (https://arxiv.org/abs/2408.05346)
- **What's New**: 본 논문에서는 데이터 스토리 생성을 위한 새로운 작업을 소개하고 다양한 출처에서 가져온 1,449개의 스토리로 구성된 벤치마크를 제시합니다.

- **Technical Details**: 본 논문에서는 데이터 이해 및 설명, 개요 및 내레이션 생성, 각 중간 단계에서의 검증을 수행하도록 설계된 두 개의 LLM 에이전트를 사용하는 멀티 에이전트 프레임워크를 제안합니다.

- **Performance Highlights**: 제안된 에이전트 기반 프레임워크는 모델 기반 및 인간 평가에서 일반적으로 비 에이전트 기반 프레임워크보다 우수한 성능을 보여줍니다. 하지만 데이터 스토리 생성에서 고유한 과제를 드러냅니다.



### From Text to Insight: Leveraging Large Language Models for Performance Evaluation in Managemen (https://arxiv.org/abs/2408.05328)
Comments:
          39 pages, 8 figures, 5 tables

- **What's New**: 본 연구는 GPT-4와 같은 대규모 언어 모델(LLM)이 조직 업무 수행 평가의 객관성을 향상시키는 데 어떻게 기여할 수 있는지 살펴봅니다. 2가지 연구에 걸친 비교 분석을 통해, 다양한 업무 성과 결과를 분석하여 LLM이 지식 기반 성과 출력(knowledge-based performance outputs)을 평가하는 데 있어 인간 평가자에 비해 신뢰성이 높고 뛰어난 대안이 될 수 있음을 보여줍니다. 지식 기반 성과 출력은 지식 노동자의 핵심 기여입니다.



### A Psychology-based Unified Dynamic Framework for Curriculum Learning (https://arxiv.org/abs/2408.05326)
- **What's New**: 본 논문은 **Psychology-based Unified Dynamic Framework for Curriculum Learning (PUDF)**라는 새로운 커리큘럼 학습 프레임워크를 제안합니다. PUDF는 **Item Response Theory (IRT)** 기반의 **Artificial Crowds (AC)**를 사용하여 학습 데이터의 난이도를 자동으로 추정하고, **Dynamic Data Selection via Model Ability Estimation (DDS-MAE)**를 사용하여 모델의 현재 능력에 맞춰 학습 데이터를 동적으로 선택합니다.



### MUSE: Multi-Knowledge Passing on the Edges, Boosting Knowledge Graph Completion (https://arxiv.org/abs/2408.05283)
- **What's New**: MUSE (Multi-Knowledge Embedding Space), a 새로운 지식 그래프 완성(KGC) 모델을 제안합니다. MUSE는 다양한 지식 표현 학습 메커니즘을 통해 누락된 관계 예측을 위한 맞춤형 임베딩 공간을 3차원으로 학습합니다.



### Large Model Strategic Thinking, Small Model Efficiency: Transferring Theory of Mind in Large Language Models (https://arxiv.org/abs/2408.05241)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문은 더 작고, 시뮬레이션에 적합한 에이전트를 미세 조정을 통해 만들 수 있는지 조사합니다. 20가지의 고유한 시나리오(각 시나리오는 사회적 맥락과 사회적 딜레마를 결합)를 통해 대규모 사전 훈련된 모델을 사용하여 답변을 기록하고 이를 동일 계열의 더 작은 모델에 대한 Q&A 미세 조정에 사용합니다.



### VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents (https://arxiv.org/abs/2408.06327)
- **What's New**: 본 논문에서는 **VisualAgentBench (VAB)**라는 새로운 벤치마크를 소개하며, 이는 **대규모 멀티모달 모델(LMM)**을 **시각적 기반 에이전트(Visual Foundation Agents)**로 훈련하고 평가하기 위한 포괄적인 벤치마크입니다. VAB는 **Embodied**, **Graphical User Interface**, **Visual Design**과 같은 다양한 시나리오에 걸쳐 LMM의 잠재력을 최대한 활용하고 복잡한 실제 환경에서 LMM의 이해 및 상호 작용 능력을 평가하기 위해 설계되었습니다.



### The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery (https://arxiv.org/abs/2408.06292)
- **What's New**: 본 논문은 인공지능 과학자(The AI Scientist)라는 새로운 프레임워크를 소개하며, 최첨단 대규모 언어 모델(large language models)이 독립적으로 과학 연구를 수행하고 그 결과를 전달할 수 있도록 합니다. 인공지능 과학자는 새로운 연구 아이디어를 생성하고, 코드를 작성하고, 실험을 실행하고, 결과를 시각화하며, 완전한 과학 논문을 작성하여 결과를 설명하고, 마지막으로 평가를 위한 시뮬레이션 검토 프로세스를 수행합니다.  본질적으로 이 과정은 인간 과학계처럼 끝없이 아이디어를 반복적으로 개발할 수 있습니다. 이러한 새로운 연구 프레임워크는 과학 연구 과정에서 인공지능 에이전트의 변혁적인 이점을 제공하며, 엄청난 비용과 시간이 소요되는 과학 연구에 대한 새로운 가능성을 열어줍니다.



### Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignmen (https://arxiv.org/abs/2408.06266)
- **What's New**: 이 논문은 AI 모델의 정렬(alignment) 성능을 향상시키기 위한 두 가지 새로운 방법인 CLAIR와 APO를 제시합니다. CLAIR는 최소한의 수정을 통해 선호도 쌍(preference pairs)을 생성하는 데이터 생성 방법이고, APO는 모델과 데이터 간의 관계를 명확하게 고려하는 제어 가능한 정렬 목표입니다.



### Quantum Algorithms for Compositional Text Processing (https://arxiv.org/abs/2408.06061)
Comments:
          In Proceedings QPL 2024, arXiv:2408.05113

- **What's New**: 이 논문은 자연어 처리에서 양자 컴퓨팅과 AI의 교차점을 다루는 새로운 양자 모델 QDisCoCirc를 제안합니다. 이 모델은 자연어를 위한 최근 제안된 DisCoCirc 프레임워크를 기반으로 하며, AI의 해석 가능성을 높이는 조합적 접근 방식을 채택합니다. 즉, 전체 모델의 동작을 부분 모델들의 동작과 그 조합 방식으로 이해할 수 있습니다. 특히, 텍스트 유사성이라는 모델 고유의 기본 연산에 초점을 맞추어, QDisCoCirc 내에서 질문-답변 작업을 해결하기 위한 내결함성 양자 컴퓨터를 위한 양자 알고리즘을 제시합니다. 또한, 이 작업이 BQP-hard임을 증명합니다. 이 모델은 단어 임베딩을 매개변수화된 양자 회로로 인코딩하고, 조합성은 텍스트의 언어적 구조에 따라 양자 회로가 조합되는 방식으로 구현됩니다. QDisCoCirc는 실용적인 양자 프로세서의 잠재력을 보여주는 데 유용할 수 있습니다.



### ARPA: A Novel Hybrid Model for Advancing Visual Word Disambiguation Using Large Language Models and Transformers (https://arxiv.org/abs/2408.06040)
- **What's New**: This paper proposes a novel architecture called ARPA for Visual Word Sense Disambiguation (V-WSD). ARPA combines large language models (LLMs) with transformer-based vision encoders and a custom Graph Neural Network (GNN) layer. This architecture is designed to learn intricate relationships between visual and linguistic features, enhancing the model's ability to disambiguate words based on both text and image context.

- **Technical Details**: ARPA uses RoBERTa (a BERT-based language model) for text processing and Swin Transformer for visual feature extraction. Both modalities are projected into a shared embedding space, and then passed through a GCN layer to capture multimodal relationships. The paper also explores data augmentation techniques to improve robustness.

- **Performance Highlights**: ARPA surpasses previous V-WSD methods by achieving 15% improvement in accuracy and 6-8% improvement in Mean Reciprocal Rank (MRR).



### Controlling Surprisal in Music Generation via Information Content Curve Matching (https://arxiv.org/abs/2408.06022)
Comments:
          8 pages, 4 figures, 2 tables, accepted at the 25th Int. Society for Music Information Retrieval Conf., San Francisco, USA, 2024

- **What's New**: 이 논문은 시퀀스 모델을 사용하여 음악 생성에서 놀라움(surprisal)을 제어하는 새로운 방법을 제안합니다. 새로운 지표인 순간 정보 내용(Instantaneous Information Content, IIC)을 정의하여 음악적 놀라움을 추정하고, 이를 사용하여 음악 생성 과정을 제어합니다. 특히, 빔 검색(beam search)을 사용하여 주어진 IIC 목표 곡선을 따라가는 IIC 곡선을 갖는 음악 자료를 생성합니다.



### GlyphPattern: An Abstract Pattern Recognition for Vision-Language Models (https://arxiv.org/abs/2408.05894)
- **What's New**: GlyphPattern, a novel dataset with 954 items, assesses abstract pattern recognition in Vision-Language Models (VLMs) by testing their understanding of natural language descriptions of visual patterns. This dataset leverages patterns from 40 writing systems with three visual presentation styles, aiming to push the limits of VLMs in recognizing abstract patterns.



### VQ-CTAP: Cross-Modal Fine-Grained Sequence Representation Learning for Speech Processing (https://arxiv.org/abs/2408.05758)
- **What's New**: VQ-CTAP (Vector Quantized Contrastive Token-Acoustic Pre-training) 모델을 제안하여 텍스트와 음성을 프레임 단위로 연결하는 방법을 제시합니다. 이 모델은 크로스 모달 정렬된 시퀀스 트랜스코더를 사용하여 텍스트와 음성을 공동 다중 모달 공간으로 가져옵니다.



### Metacognitive Myopia in Large Language Models (https://arxiv.org/abs/2408.05568)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 편향을 설명하기 위해 새로운 이론적 틀인 "메타인지 근시"(metacognitive myopia)를 제시합니다. 기존 연구들은 LLM의 편향을 주로 인간 어노테이터와 훈련 데이터 선택의 문제로 여겨왔지만, 이 논문은 LLM의 계산 과정 자체에서 발생하는 문제를 강조합니다. 메타인지 근시는 LLM이 메타인지의 두 구성 요소인 모니터링과 제어 기능이 부족하여 발생하는 현상으로, LLM이 정보를 잘못 처리하고 편향된 출력을 생성하는 이유를 설명합니다.



### Improving Whisper's Recognition Performance for Under-Represented Language Kazakh Leveraging Unpaired Speech and Tex (https://arxiv.org/abs/2408.05554)
Comments:
          Accepted by INTERSPEECH 2024;Minor typo correction

- **What's New**: This paper proposes a method to improve the performance of Whisper (a large-scale Automatic Speech Recognition (ASR) model) on low-resource languages, specifically focusing on Kazakh. The study leverages readily accessible unpaired speech and text data and integrates the language model GPT (Generative Pre-trained Transformer) with Whisper for improved recognition accuracy.



### GEM: Context-Aware Gaze EstiMation with Visual Search Behavior Matching for Chest Radiograph (https://arxiv.org/abs/2408.05502)
Comments:
          9 figures

- **What's New**: 본 논문은 의료 영상 해석 시 의사의 시선 추적 데이터를 활용하여 시각적 주의 패턴과 정보 처리 전략을 파악하는 새로운 접근 방식을 제시합니다. 특히, 의료 보고서와 영상 간의 맥락적 연결을 활용하여 정확한 시선 추정을 수행하는 '맥락 인식 시선 추정(Context-Aware Gaze EstiMation, GEM) 네트워크'를 제안합니다. GEM 네트워크는 다양한 모듈을 통해 의사의 시각적 탐색 행동 패턴을 정확하게 모사하고 의료 영상 해석 과정에 대한 통찰력을 제공합니다.



### Revisiting Multi-Modal LLM Evaluation (https://arxiv.org/abs/2408.05334)
- **What's New**: 본 연구는 기존의 비주얼 퀘스천 앤서링(VQA) 데이터셋의 한계를 극복하고 최신 멀티모달 대규모 언어 모델(MLLM)을 위한 새로운 평가 데이터셋을 제시합니다. 새로운 데이터셋은 다양한 질문 유형과 시각적 추론 능력을 평가할 수 있도록 설계되었으며, 특히 시각적 지각(visual grounding) 능력을 더욱 엄격하게 평가하는 데 중점을 둡니다.

- **Technical Details**: 본 연구에서는 다음과 같은 데이터셋을 사용합니다:

- **TDIUC**: 12가지 질문 유형을 포함하여 세분화된 분석을 가능하게 하는 VQA 데이터셋.
- **TallyQA**: 간단하고 복잡한 계산 질문을 포함하는 VQA 데이터셋.
- **DVQA**: 차트 이해를 위해 광학 문자 인식(OCR)을 필요로 하는 VQA 데이터셋.
- **VQDv1**: 주어진 질의에 맞는 모든 이미지 영역을 식별해야 하는 VQA 데이터셋. 

또한 다음과 같은 최신 MLLM 모델을 평가합니다:

- **LLaVA 1.5**, **LLaVA-NeXT**, **BLIP2**, **InstructBLIP**, **GPT-4V**, **GPT-4o**

- **Performance Highlights**: 본 연구에서는 새로운 평가 데이터셋을 사용하여 최신 MLLM의 성능을 분석한 결과, 기존의 데이터셋에서는 발견되지 않았던 새로운 약점들이 드러났습니다. 예를 들어, 일부 MLLM은 복잡한 시각적 추론 능력을 요구하는 VQDv1에서 저조한 성능을 보였습니다. 이는 이러한 모델들이 단일 객체 식별에 의존하는 기존의 데이터셋에 과도하게 적응되어 있기 때문일 수 있습니다.



### reCSE: Portable Reshaping Features for Sentence Embedding in Self-supervised Contrastive Learning (https://arxiv.org/abs/2408.04975)
- **What's New**: This paper proposes a novel contrastive learning sentence representation framework called reCSE.  Instead of using discrete data augmentation methods (like adding supplementary samples), reCSE focuses on **feature reshaping** to improve sentence representation. It reshapes the input features of the original sentence, capturing global information of each token, and alleviates the problems of representation polarity and excessive GPU memory consumption.



New uploads on arXiv(cs.IR)

### OpenResearcher: Unleashing AI for Accelerated Scientific Research (https://arxiv.org/abs/2408.06941)
- **What's New**: OpenResearcher는 AI 기반 연구 플랫폼으로, 다양한 과학적 질문에 답변하고 연구 프로세스를 가속화합니다. 특히, Retrieval-Augmented Generation (RAG) 기술을 사용하여 대규모 언어 모델(LLM)과 최신 도메인별 지식을 통합합니다.



### Diffusion Model for Slate Recommendation (https://arxiv.org/abs/2408.06883)
Comments:
          9 pages, 5 figures, 3 tables

- **What's New**: This paper introduces a novel slate recommendation approach using Diffusion Models (DMs) to address the challenge of combinatorial choice space, especially when users interact with multiple items simultaneously. Unlike traditional methods that focus on single-item engagement, DMSR generates high-quality slates considering user satisfaction and diversity.



### Reformulating Conversational Recommender Systems as Tri-Phase Offline Policy Learning (https://arxiv.org/abs/2408.06809)
Comments:
          Accepted at CIKM 2024

- **What's New**: 기존의 대화형 추천 시스템(CRS)은 주로 사용자 시뮬레이터를 활용하여 추천 정책을 학습하고 평가했습니다. 하지만 이러한 시뮬레이터는 정적인 아이템 속성에만 초점을 맞춰 사용자 상호 작용의 복잡성을 단순화하여 실제 사용자 행동을 특징짓는 풍부하고 진화하는 선호도를 무시했습니다. 이러한 제한으로 인해 시뮬레이션 환경에서는 잘 수행되지만 실제 배포 시에는 제대로 작동하지 않는 모델이 빈번하게 발생했습니다. 이러한 과제를 해결하기 위해 본 논문에서는 실시간 상호 작용에 대한 의존성을 크게 줄이고 기존 방식에 만연한 과적합 문제를 완화하는 삼단계 오프라인 정책 학습 기반 대화형 추천 시스템(TPCRS)을 소개합니다. TPCRS는 모델 기반 오프라인 학습 전략과 개인화되고 진화하는 사용자 선호도에 동적으로 맞춰지는 제어 가능한 사용자 시뮬레이션을 통합합니다. 종합적인 실험을 통해 TPCRS는 다양한 사용자 시나리오에서 향상된 견고성, 적응성 및 정확성을 보여주었으며 기존 CRS 모델보다 우수한 성능을 보였습니다. 이 접근 방식은 더 현실적인 평가 환경을 제공할 뿐만 아니라 사용자 행동 역학에 대한 심층적인 이해를 가능하게 하여 추천 프로세스를 개선합니다.

- **Technical Details**: TPCRS는 사용자 모델 학습, 정책 학습, 제어 가능한 사용자 시뮬레이션의 세 가지 주요 구성 요소로 구성됩니다. 사용자 모델 학습은 오프라인 데이터에서 사용자 선호도를 추정하고 사용자 선호도의 동적이고 개인화된 특성을 포착하는 반면, 정책 학습은 학습된 사용자 모델을 시뮬레이션 환경으로 사용하여 추천 정책의 학습을 용이하게 합니다. 마지막으로, 제어 가능한 사용자 시뮬레이션은 모델의 적응성을 다양한 사용자 시나리오에서 평가하기 위해 개인화된 선호도 매개변수와 선호도 진화율을 사용하여 역동적으로 사용자 상호 작용에 맞춰지는 블랙 박스 테스트 환경을 제공합니다.

- **Performance Highlights**: TPCRS는 다양한 사용자 시나리오에서 향상된 견고성, 적응성 및 정확성을 보여주었으며 기존 CRS 모델보다 우수한 성능을 보였습니다. TPCRS는 실제 사용자 환경에서 배포될 때 개선된 추천 성능을 보이는 데 기여하는 더 현실적인 평가 환경과 더 나은 사용자 행동 역학에 대한 이해를 제공합니다.



### Hierarchical Structured Neural Network for Retrieva (https://arxiv.org/abs/2408.06653)
Comments:
          9 pages

- **What's New**: HSNN (Hierarchical Structured Neural Network)를 소개합니다. 이 모델은 광고 추천 시스템의 검색 단계를 개선하기 위해,  hierarchical clustering과 neural network를 함께 학습하는 새로운 방법을 제시합니다.  기존의 Two Tower 아키텍처의 한계를 극복하고,  sophisticated interaction 및  ranking 시스템에서 일반적으로 사용되는 모델 아키텍처를 활용합니다.  HSNN은 sub-linear inference cost를 유지하면서  offline 평가에서 6.5% 개선 및 A/B 실험을 통해  1.22% online 개선을 달성했습니다.  또한,  HSNN은 광고 추천 시스템에 배포되어  현재 많은 트래픽을 처리하고 있습니다. 



### BMX: Entropy-weighted Similarity and Semantic-enhanced Lexical Search (https://arxiv.org/abs/2408.06643)
- **What's New**: BMX (BM25 확장)은 기존 BM25의 제한적인 성능을 개선하기 위해 엔트로피 가중 유사도와 의미 강화 기술을 도입합니다. BMX는 긴 컨텍스트와 실제 검색 벤치마크에서 BM25를 능가하고 심지어 PLM/LLM 기반 밀집 검색을 뛰어넘는 성능을 보여줍니다.  이 연구는 고전적인 어휘 검색과 현대적인 의미적 접근 방식 간의 격차를 해소하고 미래 정보 검색 연구를 위한 유망한 방향을 제시합니다.



### Prompt Tuning as User Inherent Profile Inference Machin (https://arxiv.org/abs/2408.06577)
- **What's New**: 이 논문에서는 사용자의 잠재적 프로필을 추론하는 데 있어 기존의 LLMs(Large Language Models)의 한계를 해결하기 위해 UserIP-Tuning 프레임워크를 제안합니다. 이 프레임워크는 사용자 프로필과 행동 시퀀스 간의 인과 관계를 LLMs의 프롬프트에 통합하고, Expectation Maximization 알고리즘을 사용하여 잠재적 프로필을 추론합니다.

- **Technical Details**: UserIP-Tuning은 다음과 같은 세 가지 주요 모듈로 구성됩니다. 
1. **UserIP 추론:** 프롬프트 템플릿 내에 훈련 가능한 소프트 토큰으로 사용자의 잠재적 프로필을 처리하여 LLMs로부터 정확한 사용자 행동 시퀀스를 유도합니다. 
2. **UserIP 양자화:** 훈련된 소프트 토큰을 훈련 가능한 코드북을 사용하여 희소 특징 ID로 변환합니다. 이러한 ID는 온라인 배포를 위해 잠재적 특징 뱅크에 저장됩니다. 
3. **UserIP 특징 뱅크:** 양자화된 소프트 토큰을 저장하여 온라인 추론 시 효율성을 높입니다. 
UserIP-Tuning은 훈련 가능한 소프트 토큰을 사용하고, Expectation Maximization(EM) 알고리즘을 적용하여 LLMs의 불안정한 지시 준수 문제를 해결하고, 텍스트 노이즈를 최소화합니다.

- **Performance Highlights**: UserIP-Tuning은 네 가지 공개 데이터 세트에서 최첨단 추천 알고리즘을 능가하는 성능을 보여줍니다. 추가 테스트 및 사례 연구를 통해 효율성, 강력성, 전이 가능성을 확인했습니다.



### Modality-Balanced Learning for Multimedia Recommendation (https://arxiv.org/abs/2408.06360)
Comments:
          ACM Multimedia 2024 (Oral)

- **What's New**: 이 논문에서는 **모달 불균형 (modal imbalance)** 문제를 해결하기 위해 **반사실적 지식 증류 (Counterfactual Knowledge Distillation)** 프레임워크를 제안합니다. 이 프레임워크는 다양한 모달의 정보를 효과적으로 활용하여 추천 모델의 성능을 향상시키는 데 중점을 둡니다.



### Deep Learning based Key Information Extraction from Business Documents: Systematic Literature Review (https://arxiv.org/abs/2408.06345)
Comments:
          52 pages, 7 figures, 9 tables; Submitted to ACM Computing Surveys

- **What's New**: 이 논문은 비즈니스 문서에서 핵심 정보를 추출하는 데 사용되는 최근의 딥 러닝 기반 접근 방식을 광범위하게 분석하고 요약합니다. 96개의 논문을 분석하여 Document Understanding (DU) 분야의 최신 연구를 살펴봅니다. 특히, Key Information Extraction (KIE)에 중점을 두고 딥 러닝 기법과 기술적 특징을 살펴봅니다. 또한, 비즈니스 프로세스 관점에서 KIE를 연구하고, 각 방법의 특징을 범주화하여 비교 분석합니다.

- **Technical Details**: 이 연구는 딥 러닝 기반 KIE 방법을 그래프 기반, 그리드 기반, 시퀀스 기반의 세 가지 주요 그룹으로 분류하고 각 그룹에 속하는 주요 기술들을 자세히 살펴봅니다. 그래프 기반 시스템은 문서 페이지를 레이아웃과 콘텐츠를 나타내는 그래프 구조로 변환합니다. 그리드 기반 시스템은 문서 이미지 픽셀을 기반으로 잘 정의된 연결을 갖는 그리드 구조를 만듭니다. 시퀀스 기반 시스템은 문서를 선형 텍스트 시퀀스로 처리합니다. 또한, 문서 이해 작업을 지각, 유도, 추론의 세 가지 하위 작업으로 분류하고 각 작업의 핵심 내용을 소개합니다. KIE 작업은 Named Entity Recognition (NER), Relation Extraction (RE) 등의 하위 작업으로 구성되며, 각 작업의 목적과 특징을 설명합니다.

- **Performance Highlights**: 이 연구는 딥 러닝 기반 KIE 방법의 다양한 성능 지표를 분석하고, 각 방법의 장단점을 비교 분석합니다. 특히, 정확도, 재현율, F1 스코어, 처리 시간 등을 비교하여 각 방법의 강점과 약점을 파악합니다. 또한, 각 방법의 적용 가능성을 다양한 비즈니스 문서 유형에 대한 분석을 통해 살펴봅니다.



### TableGuard -- Securing Structured & Unstructured Data (https://arxiv.org/abs/2408.07045)
Comments:
          7 pages, 3 tables, 1 figure

- **What's New**: TableGuard는 관계형 데이터베이스를 위한 혁신적인 데이터 난독화 접근 방식으로, 컨텍스트 기반 난독화를 활용하여 API 호출이 난독화된 데이터만 반환하도록 함으로써 제3자와 데이터를 공유할 때 개인 정보를 보호합니다. TableGuard는 컨텍스트에 적합한 대안으로 민감한 데이터 요소를 대체하여 데이터의 관계적 무결성 및 일관성을 유지함으로써 인지 부조화 및 데이터 유출 위험을 완화합니다.



### Generalized knowledge-enhanced framework for biomedical entity and relation extraction (https://arxiv.org/abs/2408.06618)
- **What's New**: 이 연구는 바이오메디컬 엔티티 및 관계 추출을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 외부 지식을 활용하여 작업에 독립적이고 재사용 가능한 바이오메디컬 엔티티 및 관계 추출을 위한 배경 지식 그래프를 구축합니다. 이 모델은 인간이 도메인 특정 주제를 학습하는 방식에서 영감을 받았습니다. 특히 인간은 먼저 기초 지식을 구축하기 위해 필드에 대한 가장 기본적이고 일반적인 지식을 습득한 다음, 이를 기반으로 다양한 전문 분야 주제로 확장합니다. 이 프레임워크는 이러한 일반적인 지식 공유 메커니즘을 사용하여 다양한 도메인 특정 바이오메디컬 텍스트에 효과적으로 전이 학습이 가능한 일반적인 신경망 지식 그래프를 구축합니다.



### Learned Ranking Function: From Short-term Behavior Predictions to Long-term User Satisfaction (https://arxiv.org/abs/2408.06512)
Comments:
          RecSys 24

- **What's New**: 이 논문은 YouTube에서 사용자 만족도를 극대화하기 위해 새로운 학습 기반 순위 함수 (Learned Ranking Function, LRF) 시스템을 제안합니다. 기존의 솔루션들이 휴리스틱 함수의 하이퍼파라미터를 최적화하는 데 초점을 맞춘 반면, LRF는 슬레이트 최적화 문제를 직접 모델링하여 장기적인 사용자 만족도를 극대화하는 데 목표를 두고 있습니다. LRF는 다중 목표 최적화의 안정성을 보장하는 새로운 제약 최적화 알고리즘도 포함합니다. 

- **Technical Details**: LRF는 사용자와 슬레이트(추천 목록)의 상호 작용을 캐스케이드 클릭 모델(cascade click model)로 모델링합니다. 이 모델은 사용자가 슬레이트를 순차적으로 검토하며, 특정 비디오를 클릭하거나, 슬레이트를 포기하거나, 또는 슬레이트의 끝까지 도달할 수 있다는 것을 가정합니다. LRF는 이러한 사용자 행동을 고려하여 슬레이트의 장기적인 가치를 최적화하는 알고리즘을 사용합니다. 또한, LRF는 다중 목표 최적화의 안정성을 보장하기 위해 동적 선형 스칼라화 (dynamic linear scalarization) 기반의 새로운 제약 최적화 알고리즘을 제안합니다. 

- **Performance Highlights**: LRF 시스템은 YouTube에 배포되어 실험 결과를 통해 효과가 입증되었습니다. LRF는 기존의 휴리스틱 기반 시스템보다 장기적인 사용자 만족도를 개선하는 것으로 나타났습니다. 또한, LRF는 다양한 목표 간의 균형을 유지하며 안정적인 성능을 보여주었습니다. 



### Accuracy and Political Bias of News Source Credibility Ratings by Large Language Models (https://arxiv.org/abs/2304.00228)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문은 세 가지 주요 제공업체(OpenAI, Google, Meta)의 널리 사용되는 8개의 대규모 언어 모델(LLM)을 감사하여 낮은 신뢰도의 정보 소스에서 신뢰할 수 있고 고품질의 정보 소스를 식별하는 능력을 평가합니다.



### The landscape of ontologies in materials science and engineering: A survey and evaluation (https://arxiv.org/abs/2408.06034)
- **What's New**: 본 논문은 재료 과학 및 엔지니어링 (MSE) 분야에서 사용되는 온톨로지 (Ontology)를 종합적으로 검토하고 분석하여, MSE 전문가들이 특정 목적에 가장 적합한 온톨로지를 선택하는 데 도움을 주는 가이드를 제공합니다. 특히, 60개의 온톨로지를 분석하고 비교하여 각 온톨로지의 강점과 약점을 파악하고, 온톨로지 재사용 통계 및 주요 지표를 제공합니다. 이를 통해 전문가들은 적합한 온톨로지를 선택하고 기존 리소스의 관련 용어를 통합할 수 있습니다.



### Optimizing RAG Techniques for Automotive Industry PDF Chatbots: A Case Study with Locally Deployed Ollama Models (https://arxiv.org/abs/2408.05933)
- **What's New**: 본 연구는 오프라인 PDF 챗봇의 자동차 산업 생산 환경에서의 수요 증가에 따라, 저성능 로컬 환경에서 대규모 언어 모델(LLMs, Large Language Models)의 효율적인 배포를 위한 연구를 다룹니다. 특히, 로컬 환경에 배포된 Ollama 모델을 사용하여 자동차 산업 문서를 처리하는 Retrieval-Augmented Generation (RAG) 기법의 성능 향상에 중점을 둡니다. Langchain 프레임워크를 기반으로, Ollama의 로컬 RAG 구현을 위한 다차원적 최적화 방안을 제시합니다. 본 연구는 자동차 산업 문서 처리의 핵심 과제인 다열 레이아웃(multi-column layouts)과 기술 사양(technical specifications) 처리에 초점을 맞추어, 자동차 산업 문서의 특성에 맞춰 PDF 처리, 검색 메커니즘 및 컨텍스트 압축(context compression)을 개선합니다. 또한, 임베딩 파이프라인(embedding pipelines)을 지원하는 맞춤형 클래스와 LangGraph 모범 사례를 기반으로 Self-RAG를 지원하는 에이전트를 설계합니다.



### GraphTransfer: A Generic Feature Fusion Framework for Collaborative Filtering (https://arxiv.org/abs/2408.05792)
- **What's New**: 본 논문에서는 그래프 특징(graph feature)과 보조 특징(auxiliary feature)을 효과적으로 결합하여 협업 필터링(collaborative filtering, CF) 성능을 향상시키는 새로운 그래프 전송(GraphTransfer) 프레임워크를 제안합니다. GraphTransfer는 GNN 기반의 CF 알고리즘을 위한 보편적인 기능 융합(feature fusion) 프레임워크이며, 그래프 특징과 보조 특징 사이의 의미적 차이(semantic gap)를 줄이기 위해 교차 융합 모듈(cross fusion module)을 사용합니다.



### Advancing Re-Ranking with Multimodal Fusion and Target-Oriented Auxiliary Tasks in E-Commerce Search (https://arxiv.org/abs/2408.05751)
- **What's New**: 이 연구는 전자상거래 검색 재순위 지정 모델에 텍스트 및 시각 정보를 결합하여 사용자 경험을 향상시키고 전환율을 높이는 새로운 접근 방식을 제시합니다. "ARMMT"라는 새로운 모델은 텍스트와 시각 정보를 통합하는 "Context-Aware Fusion Unit (CAFU)"와 순위와 관련된 보조 작업을 활용하여 품목 표현을 개선하고 개인 맞춤형 추천 기능을 강화합니다.



### Moment&Cross: Next-Generation Real-Time Cross-Domain CTR Prediction for Live-Streaming Recommendation at Kuaishou (https://arxiv.org/abs/2408.05709)
Comments:
          Work in progress

- **What's New**: 이 논문은 Kuaishou의 실시간 스트리밍 추천 시스템을 개선하기 위한 두 가지 새로운 기술인 Moment & Cross를 제안합니다. Moment는 실시간으로 발생하는 사용자 행동을 이용하여 실시간 CTR 추세를 파악하고, 이를 통해 '하이라이트' 실시간 스트리밍을 자동으로 발견하는 기술입니다. Cross는 사용자의 짧은 비디오 히스토리와 실시간 스트리밍 임베딩 공간을 정렬하여 사용자의 짧은 비디오 관심사를 실시간 스트리밍 추천에 활용하는 기술입니다.



### A Decoding Acceleration Framework for Industrial Deployable LLM-based Recommender Systems (https://arxiv.org/abs/2408.05676)
- **What's New**: 이 논문은 LLM 기반 추천 시스템에서 추천 지식 생성의 효율성을 높이는 **DARE (Decoding Acceleration Framework for LLM-based Recommendation)** 프레임워크를 제안합니다. DARE는 **Speculative Decoding** 기법을 활용하여 LLM의 **Autoregressive** 특성으로 인한 지연을 줄이고 효율성을 향상시킵니다. 특히, 추천 시스템의 특징을 고려하여 **Customized Retrieval Pool**과 **Relaxed Verification** 전략을 도입합니다.



### Exploring Applications of State Space Models and Advanced Training Techniques in Sequential Recommendations: A Comparative Study on Efficiency and Performanc (https://arxiv.org/abs/2408.05606)
Comments:
          arXiv admin note: text overlap with arXiv:2403.07691 by other authors

- **What's New**: 이 논문은 시퀀스 추천에서의 효율성과 성능을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히 State Space Model (SSM)을 기반으로 한 Mamba 모델을 사용하여 추천 시스템의 속도를 높이고 대규모 언어 모델 (LLM)을 통합하여 추천 품질을 향상시킵니다. 또한, 학습 과정을 가속화하고 비용을 줄이기 위한 적응형 배치 및 단계 크기 알고리즘을 도입합니다.

- **Technical Details**: 이 연구는 다음과 같은 주요 기술적 세부 사항을 다룹니다:

- **State Space Model (SSM):** SSM은 시퀀스 모델링을 위한 최신 프레임워크로, 선형 상미분 방정식을 사용하여 시퀀스의 변화를 모델링합니다. SSM은 Transformer 모델에 비해 낮은 메모리 및 추론 비용으로 우수한 성능을 제공합니다.
- **Mamba:** Mamba는 SSM의 확장판으로, 데이터 의존적인 선택 메커니즘을 추가하여 관련 정보에 집중하고 노이즈를 제거합니다. Mamba는 GPU를 사용하여 효율적으로 계산할 수 있습니다.
- **Universal Stochastic Gradient Method (USGM):** USGM은 기계 학습의 기반인 확률적 최적화 분야에서 주목할 만한 발전입니다. USGM은 기존의 Stochastic Gradient Descent (SGD)보다 더 빠른 수렴 속도를 제공합니다.
- **Adaptive Batch and Step Size Algorithms:** 이러한 알고리즘은 학습 과정에서 배치 크기와 단계 크기를 동적으로 조정하여 학습 속도를 높이고 비용을 절감합니다.

- **Performance Highlights**: 이 연구는 다음과 같은 주요 성능 향상을 강조합니다:

- **더 빠른 추론 속도:** SSM 기반 Mamba 모델은 Transformer 모델에 비해 훨씬 빠른 추론 속도를 제공합니다.
- **향상된 추천 품질:** LLM을 사용한 Monolithic Preference Optimization (ORPO)는 추천 품질을 향상시킵니다.
- **더 빠른 학습 과정:** 적응형 배치 및 단계 크기 알고리즘은 학습 과정을 가속화합니다.
- **낮은 메모리 및 추론 비용:** SSM은 Transformer 모델에 비해 낮은 메모리 및 추론 비용을 요구합니다.



### HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou (https://arxiv.org/abs/2408.05430)
Comments:
          Work in progress

- **What's New**: This paper presents **HoME (Hierarchy of Multi-gate Experts)**, a novel architecture for multi-task learning in short-video platforms like Kuaishou, addressing three key anomalies found in traditional Mixture-of-Experts (MoE) models: **Expert Collapse**, **Expert Degradation**, and **Expert Underfitting**.

- **Technical Details**: HoME employs three main techniques: **Expert Normalization & Swish Mechanism** to balance expert output distributions and avoid expert collapse, **Hierarchy Mask Mechanism** to reduce expert occupancy and prevent expert degradation, and **Feature-gate and Self-gate Mechanisms** to enhance the training of sparse-task experts.

- **Performance Highlights**: HoME achieves significant improvements in multi-task learning for short-video platforms, leading to better personalization and user experience by effectively capturing users' interests through various behavioral cues.



### Report on the 1st Workshop on Large Language Model for Evaluation in Information Retrieval (LLM4Eval 2024) at SIGIR 2024 (https://arxiv.org/abs/2408.05388)
Comments:
          LLM4Eval Workshop Report

- **What's New**: The first workshop on Large Language Models for Evaluation in Information Retrieval (LLM4Eval 2024) was held in July 2024, focusing on the use of LLMs for evaluating IR systems.

- **Technical Details**: The workshop explored various topics related to LLM-based evaluation in IR, including: LLM-based evaluation metrics, agreement between human and LLM labels, effectiveness and efficiency of LLMs for relevance labeling, LLM-based relevance estimators, automated evaluation of text generation systems, end-to-end evaluation of Retrieval Augmented Generation (RAG) systems, trustworthiness in LLM evaluation, prompt engineering in LLM evaluation, and the use of LLMs as ranking models.

- **Performance Highlights**: The workshop received 21 paper submissions, 7 of which were selected for presentation and publication in the CEUR-WS volume. The workshop also featured two invited keynote talks focusing on the use of LLMs in IR evaluation and their potential impact on the field.

- **Event Details**: The workshop was held as a full-day in-person event in Washington D.C., US on July 18, 2024, and included a poster session with accepted papers and a panel discussion.



### IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning (https://arxiv.org/abs/2408.05353)
- **What's New**: 이 논문은 Netflix 사용자 데이터에서 사용자의 잠재적인 의도를 추론하고 사용자의 다음 행동을 예측하는 새로운 권장 시스템인 IntentRec을 제안합니다. IntentRec은 계층적 다중 작업 신경망 아키텍처를 기반으로 하며, 단기 및 장기 암묵적 신호를 사용하여 사용자의 잠재적인 의도를 추정하고, 이를 사용하여 사용자가 다음에 참여할 가능성이 높은 항목을 예측합니다.



### Towards Scalable Topic Detection on Web via Simulating Levy Walks Nature of Topics in Similarity Spac (https://arxiv.org/abs/2408.05348)
- **What's New**: 본 논문에서는 소셜 미디어에서 인기 주제를 찾기 위한 새로운 방법을 제안합니다. 기존의 방법들은 소셜 미디어 데이터의 잡음(noise)으로 인해 효과적인 주제 발견에 어려움을 겪었습니다. 본 논문에서는 잡음 데이터에서도 인기 주제를 효과적으로 찾을 수 있는 Explore-Exploit (EE) 기반의 새로운 접근 방식을 제안합니다. 이 방법은 르비 워크(Lévy walks)의 특성을 이용하여 웹 페이지 간의 유사성을 분석하여 주제를 그룹화하는 방식입니다. EE 방법은 기존 방법들과 비교하여 효율성과 정확성 측면에서 모두 뛰어난 성능을 보였습니다.

- **Technical Details**: 본 논문에서 제안하는 EE 기반의 주제 클러스터링은 다음과 같은 특징을 가지고 있습니다.

* **르비 워크(Lévy walks)**: 르비 워크는 짧은 이동과 긴 이동을 반복적으로 수행하는 무작위 이동 모델입니다. 본 논문에서는 웹 페이지 간의 유사성을 르비 워크의 이동 거리에 비유하여 인기 주제를 찾는 방식을 제안합니다.
* **Explore-Exploit (EE) 접근 방식**: EE 접근 방식은 새로운 주제를 탐색(Explore)하고, 이미 발견된 주제를 더 자세히 조사(Exploit)하는 전략입니다. 본 논문에서는 EE 접근 방식을 사용하여 르비 워크를 시뮬레이션하고, 새로운 주제를 발견하고, 기존 주제를 확장합니다.
* **Poisson Deconvolution (PD)**: PD는 주제의 흥미도를 측정하는 방법입니다. 본 논문에서는 PD를 사용하여 EE 방법을 통해 발견된 주제 중 인기 주제를 선별합니다.


- **Performance Highlights**: 실험 결과, 본 논문에서 제안한 EE 기반의 주제 클러스터링 방법은 기존 방법들과 비교하여 다음과 같은 장점을 보였습니다.

* **효율성**: 기존 방법들보다 훨씬 빠르게 주제를 발견할 수 있습니다.
* **정확성**: 기존 방법들과 비교하여 뛰어난 정확도를 보였습니다.
* **일반화 능력**: 다양한 유형의 소셜 미디어 데이터에 적용 가능합니다.




### AI-assisted Coding with Cody: Lessons from Context Retrieval and Evaluation for Code Recommendations (https://arxiv.org/abs/2408.05344)
- **What's New**: 이 논문은 LLM(Large Language Model) 기반 코딩 어시스턴트의 핵심 요소인 컨텍스트 엔진(context engine)에 대한 심층적인 분석을 제공합니다. 특히, LLM의 컨텍스트 윈도우(context window) 크기 제한과 실제 코딩 환경에서의 컨텍스트 엔진의 중요성을 강조하며, 전통적인 추천 시스템(RecSys)과의 차이점 및 유사점을 비교 분석합니다.



### Neural Machine Unranking (https://arxiv.org/abs/2408.05330)
- **What's New**: 본 논문은 신경 정보 검색(Neural Information Retrieval, NIR)에서 특정 데이터 포인트를 선택적으로 삭제하는 새로운 머신 언러닝(Machine Unlearning) 기술, Neural Machine UnRanking (NuMuR)을 제시합니다. 기존의 머신 언러닝 방법들은 주로 분류 작업에 초점을 맞춰 설계되었으나, NIR의 고유한 특징 때문에 NuMuR 작업에서 효과적이지 못했습니다.  본 연구에서는 NuMuR에 적합한 Contrastive and Consistent Loss (CoCoL) 방법론을 개발하여 데이터 삭제와 모델 성능 유지를 효과적으로 조화시켰습니다.



### Perceptual Similarity for Measuring Decision-Making Style and Policy Diversity in Games (https://arxiv.org/abs/2408.06051)
Comments:
          TMLR 08/2024 this https URL

- **What's New**: 이 논문은 게임에서 플레이 스타일을 측정하는 방법인 'Playstyle Distance'를 개선한 'Playstyle Similarity'를 제안합니다. 기존 방법의 한계를 극복하기 위해 다중 스케일 분석, 심리 물리학적 커널, 교집합-합집합 비율(intersection-over-union) 방법을 도입했습니다. 이를 통해 플레이 스타일 분류 정확도를 향상시켰고, 게임 상황을 이해하는 새로운 관점을 제시합니다.



### ConvKGYarn: Spinning Configurable and Scalable Conversational Knowledge Graph QA datasets with Large Language Models (https://arxiv.org/abs/2408.05948)
- **What's New**: 본 논문은 대규모 언어 모델(LLM) 및 대화형 비서의 급속한 발전으로 인해 훈련 및 평가를 위한 역동적이고 확장 가능하며 구성 가능한 대화형 데이터 세트의 필요성이 증가했음을 강조합니다. 이러한 데이터 세트는 텍스트 및 음성을 포함한 다양한 사용자 상호 작용 모드를 수용해야 하며, 각 모드는 고유한 모델링 과제를 제시합니다. 구조적이고 진화하는 특성을 지닌 지식 그래프(KG)는 현재 및 정확한 지식에 대한 이상적인 기반을 제공합니다. 인간이 큐레이팅한 KG 기반 대화형 데이터 세트가 존재하지만, 급변하는 사용자 정보 요구 사항을 따라잡기 어려움이 있습니다. 이 논문에서는 최신의 구성 가능한 대화형 KGQA 데이터 세트를 생성하기 위한 확장 가능한 방법인 ConvKGYarn을 제시합니다. 질적 심리 측정 분석은 이 방법이 인기 있는 대화형 KGQA 데이터 세트에 필적하는 고품질 데이터 세트를 생성할 수 있음을 확인하면서 동시에 규모를 확장하고 다양한 인간 상호 작용 구성을 다룰 수 있습니다. 이 논문에서는 다양한 대화에서 LLM을 테스트하여 동일한 KG 팩트 세트를 기반으로 하는 다양한 구성을 갖춘 대화형 KGQA 세트에서 모델 동작을 탐구함으로써 유용성을 보여줍니다. 결과는 ConvKGYarn이 KGQA 기반을 개선하고 LLM의 매개 변수 지식을 평가할 수 있는 능력을 강조하여 대화형 비서의 끊임없이 진화하는 환경에 대한 강력한 솔루션을 제공합니다.



### Low-Rank Approximation, Adaptation, and Other Tales (https://arxiv.org/abs/2408.05883)
- **What's New**: 이 논문은 저랭크 근사와 적응의 개념을 명확히 하고, 다양한 분야에서 이 기술의 힘을 보여주는 포괄적인 안내서를 제공합니다. 특히, 새로운 저랭크 분해 및 적응 알고리즘을 소개하며, 이는 미래 연구자들에게 큰 잠재력을 가지고 있습니다.



### Iterative Improvement of an Additively Regularized Topic Mod (https://arxiv.org/abs/2408.05840)
Comments:
          A full draft of the second version of the article

- **What's New**: 이 논문은 이전에 발견된 모든 좋은 주제를 유지하여 이전 모델보다 좋거나 동등한 일련의 관련 주제 모델을 학습하는 주제 모델의 반복적 학습 방법을 제시합니다. 이 방법의 핵심은 각 후속 모델이 이전 모델보다 좋다는 것을 보장하기 위해 (즉, 이전에 발견된 모든 좋은 주제를 유지하기 위해) 각 후속 모델이 이전 모델보다 적어도 좋거나 동등한 일련의 관련 주제 모델을 학습하는 것입니다. 이러한 모델 간의 연결은 추가 정규화를 통해 달성됩니다. 이러한 반복적인 학습의 결과는 일련의 마지막 주제 모델이며, 이를 반복적으로 업데이트된 추가적으로 정규화된 주제 모델(ITAR)이라고 합니다.



### Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites (https://arxiv.org/abs/2408.05667)
- **What's New**: PhishLang, an open-source, lightweight Large Language Model (LLM) for phishing website detection, is introduced. It leverages contextual analysis of website source code to identify phishing patterns.

- **Technical Details**: PhishLang utilizes LLM's advanced language processing capabilities to learn granular features characteristic of phishing attacks. It employs a "sliding window" technique for efficient training and operates with minimal data preprocessing, making it faster and less resource-intensive compared to traditional deep learning models.

- **Performance Highlights**: PhishLang successfully identified approximately 26K phishing URLs, many undetected by popular anti-phishing blocklists, over a 3.5-month testing period. It's also robust against adversarial attacks and integrates with GPT-3.5 Turbo for "explainable blocklisting", providing users with contextual information on why a website was flagged as phishing.

- **Open-source Availability**: PhishLang is open-sourced and available as a Chromium-based browser extension and a URL scanning website, providing users with real-time protection against phishing threats.



### Document-Level Event Extraction with Definition-Driven ICL (https://arxiv.org/abs/2408.05566)
- **What's New**: 본 논문에서는 문서 수준 이벤트 추출에서 LLM의 프롬프트 엔지니어링 문제를 해결하기 위한 새로운 최적화 전략인 "정의 기반 문서 수준 이벤트 추출(DDEE)"를 제안합니다. 이 전략은 프롬프트 길이를 조정하고 휴리스틱의 명확성을 강화하여 LLM의 이벤트 추출 성능을 향상시키고, 데이터 균형 기술을 사용하여 롱테일 효과 문제를 해결함으로써 모델의 이벤트 유형에 대한 일반화 능력을 강화했습니다. 동시에, LLM의 프롬프트 스타일 민감도에 맞춰 간결하고 포괄적인 프롬프트를 개선했고, 구조화된 휴리스틱 방법과 엄격한 제한 조건을 도입하여 이벤트 및 인수 역할 추출의 정확성을 향상시켰습니다. 이러한 전략은 문서 수준 이벤트 추출에서 LLM의 프롬프트 엔지니어링 문제를 해결할 뿐만 아니라 이벤트 추출 기술의 발전을 촉진하고 NLP 분야의 다른 작업에 대한 새로운 연구 관점을 제공합니다.



### SocFedGPT: Federated GPT-based Adaptive Content Filtering System Leveraging User Interactions in Social Networks (https://arxiv.org/abs/2408.05243)
Comments:
          This research paper is submitted to ASONAM 2024 conference on Advances in Social Networks Analysis and Mining and going to be published in Springer

- **What's New**: 본 논문은 개인화된 GPT 및 컨텍스트 기반 소셜 미디어 LLM 모델을 사용하여 연합 학습(Federated Learning) 프레임워크를 통해 소셜 미디어 플랫폼의 사용자 상호 작용 및 콘텐츠 관련성을 향상시키는 다면적인 접근 방식을 제시합니다. 연합 학습은 사용자 데이터 보호 및 보안을 보장하기 위해 사용됩니다.



### FLASH: Federated Learning-Based LLMs for Advanced Query Processing in Social Networks through RAG (https://arxiv.org/abs/2408.05242)
Comments:
          This research paper is submitted to ASONAM 2024 conference on Advances in Social Networks Analysis and Mining and going to be published in Springer

- **What's New**: This paper presents a novel social media chatbot powered by Federated Learning GPT, enabling personalized information retrieval and user engagement.



New uploads on arXiv(cs.CV)

### KAN You See It? KANs and Sentinel for Effective and Explainable Crop Field Segmentation (https://arxiv.org/abs/2408.07040)
Comments:
          Accepted at ECCV 2024 CVPPA Workshop

- **What's New**: 본 논문에서는 작물 필드 분할에 Kolmogorov-Arnold 네트워크(KANs) 기반 U-Net 아키텍처(U-KAN)를 적용하고 성능 및 설명 가능성을 분석한 최초의 연구입니다. 기존 U-Net 모델 대비 U-KAN 모델은 더 적은 GFLOPs에서 IoU(Intersection-Over-Union) 지표에서 2% 향상된 성능을 보였습니다. 또한 기울기 기반 설명 기법을 통해 U-KAN 예측의 타당성을 확인하고, 모델이 작물 영역 자체보다는 경계에 더 집중하는 특징을 밝혀냈습니다.



### PathInsight: Instruction Tuning of Multimodal Datasets and Models for Intelligence Assisted Diagnosis in Histopathology (https://arxiv.org/abs/2408.07037)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문은 암 진단을 위한 혁신적인 멀티모달 (multimodal) 대규모 언어 모델(LLM)인 PathEnhanceDS를 제시합니다. PathEnhanceDS는 병리학적 이미지 분류, 캡션 생성, 질문 답변 등 다양한 작업을 수행할 수 있는 45,000개의 데이터셋으로 구성됩니다. 연구팀은 LLaVA, Qwen-VL, InternLM과 같은 기존 모델을 PathEnhanceDS로 미세 조정하여 병리학 영역에서 더욱 효과적인 성능을 발휘하도록 했습니다. 특히, 이미지 캡션 생성, 분류 및 질문 답변 작업에 대한 정량적 및 질적 평가를 통해 모델의 우수성을 확인했습니다.

- **Technical Details**: PathEnhanceDS는 병리학 분야의 다양한 데이터 소스를 결합하여 구성되었습니다. 이 데이터셋은 이미지 캡션 생성, 분류, 이미지 기반 질문 답변(VQA) 및 대화를 포함한 여러 작업을 수행할 수 있도록 설계되었습니다. 특히, Pathologist-level Dataset, OpenPath, PCam, CRC-VAL-HE-7K, PathVQA, LLaVA-Med-Instruct 데이터셋이 통합되었습니다. 연구팀은 LoRA와 전체 파라미터 미세 조정을 포함한 다양한 방법을 사용하여 멀티모달 모델을 PathEnhanceDS로 미세 조정했습니다.

- **Performance Highlights**: PathEnhanceDS로 미세 조정된 모델은 이미지 캡션 생성, 분류 및 질문 답변 작업에서 우수한 성능을 보였습니다. 특히, 경험이 풍부한 병리학자의 질적 평가 결과, 모델은 병리학적 질문에 대한 답변의 정확성과 신뢰성이 높다는 것을 확인했습니다. 이 연구는 멀티모달 LLM의 잠재력을 보여주며, 향후 병리학 교육 및 진단 분야에 크게 기여할 수 있을 것으로 기대됩니다.



### Efficient Human-Object-Interaction (EHOI) Detection via Interaction Label Coding and Conditional Decision (https://arxiv.org/abs/2408.07018)
- **What's New**: 이 논문은 **효율적인 인간-객체 상호 작용 (EHOI) 탐지기**를 제안하며, 이는 탐지 성능, 추론 복잡성 및 수학적 투명성 사이의 균형을 맞추기 위해 설계되었습니다. EHOI는 두 단계 방법으로 구성됩니다. 첫 번째 단계에서는 **고정된 객체 탐지기**를 활용하여 객체를 찾고 다양한 특징을 중간 출력으로 추출합니다. 두 번째 단계에서는 첫 번째 단계의 출력을 사용하여 **XGBoost 분류기**를 통해 상호 작용 유형을 예측합니다. 이 연구는 **오류 정정 코드 (ECC)**를 적용하여 드문 상호 작용 사례를 인코딩하여 두 번째 단계의 XGBoost 분류기의 모델 크기와 복잡성을 줄입니다. 또한 재라벨링 및 의사 결정 프로세스의 수학적 공식을 제공합니다. 아키텍처 외에도, 전방 모듈의 기능을 설명하는 질적 결과를 제시합니다. 실험 결과는 ECC로 인코딩된 상호 작용 레이블의 장점과 제안된 EHOI 방법의 탐지 성능과 복잡성 사이의 우수한 균형을 보여줍니다.



### Imagen 3 (https://arxiv.org/abs/2408.07009)
- **What's New**: Imagen 3은 텍스트 프롬프트에서 고품질 이미지를 생성하는 잠재적 확산 모델입니다. 이 논문은 Imagen 3의 품질 및 책임 평가를 설명합니다. Imagen 3은 평가 당시 다른 최첨단 (SOTA) 모델보다 선호됩니다. 또한, 안전 및 표현과 관련된 문제뿐만 아니라 모델의 잠재적인 피해를 최소화하기 위해 사용된 방법을 논의합니다.



### Low-Bitwidth Floating Point Quantization for Efficient High-Quality Diffusion Models (https://arxiv.org/abs/2408.06995)
- **What's New**: 본 논문은 딥 러닝 모델의 효율성을 높이기 위한 방법으로 널리 활용되는 양자화(quantization) 기법을, 이미지 생성 모델 중 하나인 확산 모델(diffusion model)에 적용하여 이미지 품질 저하 없이 효율적인 추론을 가능하게 하는 새로운 방법을 제시합니다. 특히 기존의 정수형 양자화(integer quantization)와 달리, 부동 소수점 양자화(floating-point quantization)를 사용하여 더욱 높은 이미지 품질을 달성합니다. 또한, 부동 소수점 양자화의 효율성을 높이기 위해 가중치 반올림 학습(weight rounding learning) 기법을 도입했습니다. 이러한 방법은 기존의 정수형 양자화 방법보다 뛰어난 성능을 보이며, 8비트 부동 소수점 양자화를 적용한 경우에는 32비트 부동 소수점 모델과 비교하여 이미지 품질 저하가 거의 없음을 보여줍니다.



### SpectralGaussians: Semantic, spectral 3D Gaussian splatting for multi-spectral scene representation, visualization and analysis (https://arxiv.org/abs/2408.06975)
- **What's New**: This paper presents a novel cross-spectral rendering framework based on 3D Gaussian Splatting (3DGS) that generates realistic and semantically meaningful splats from registered multi-view spectrum and segmentation maps. This framework enhances the representation of scenes with multiple spectra, providing insights into the underlying materials and segmentation.

- **Technical Details**: The paper introduces an improved physically-based rendering approach for Gaussian splats, estimating reflectance and lights per spectra, thereby enhancing accuracy and realism. It also proposes a new spectral 3DGS approach that extends the scene representation based on 3DGS to generate realistic and semantically meaningful splats from registered multi-view spectrum and segmentation maps.

- **Performance Highlights**: The proposed approach outperforms other recent learning-based spectral scene representation approaches like XNeRF and SpectralNeRF, as well as other non-spectral state-of-the-art learning-based approaches. It also showcases the potential of spectral scene understanding for precise scene editing techniques like style transfer, inpainting, and removal.

- **Datasets**: The authors generated two synthetic spectral datasets by extending the shiny Blender dataset and the synthetic NERF dataset in terms of their spectral properties. These datasets are expected to serve as valuable resources for researchers and practitioners, offering a diverse range of spectral scenes for experimentation, evaluation, and advancements in the field of image-based/multi-view spectral rendering.



### Prompt-Based Segmentation at Multiple Resolutions and Lighting Conditions using Segment Anything Model 2 (https://arxiv.org/abs/2408.06970)
- **What's New**: 이 논문은 2023년에 출시된 Segment Anything Model (SAM)과 업데이트된 버전인 SAM 2의 효과를 조사합니다. 이 두 모델은 제로 샷(zero-shot) 프롬프트 기반 이미지 분할을 위한 최신 기술이며, 기존의 컨볼루션 신경망(CNN)과 비교됩니다. SAM 2는 특히 포인트 프롬프트를 사용할 때 광원 조건이 좋지 않은 경우 SAM보다 더 나은 성능을 보여줍니다. 두 SAM 모델 모두 사용자 지정 박스 프롬프트를 사용했을 때 모든 시나리오에서 CNN보다 뛰어났습니다. 또한, YOLOv9 프롬프팅은 사용자 지정 포인트 프롬프팅보다 더 나은 성능을 보였습니다. 고해상도 이미지(최적 및 비최적 조명 조건 모두)에서 Eff-UNet은 YOLOv9 박스 프롬프트를 사용한 두 SAM 모델보다 뛰어났으며, 이는 Eff-UNet이 고해상도 데이터의 자동 분할에 적합한 모델임을 보여줍니다. 저해상도 데이터에서 사용자 지정 박스 프롬프트는 적절한 성능을 얻는 데 매우 중요한 것으로 나타났습니다.



### Breaking Class Barriers: Efficient Dataset Distillation via Inter-Class Feature Compensator (https://arxiv.org/abs/2408.06927)
- **What's New**: This paper introduces a new paradigm for dataset distillation called **Inter-class Feature Compensator (INFER)**, which addresses the limitations of the prevailing class-specific synthesis paradigm. This paradigm focuses on generating synthetic data instances that represent specific classes, leading to inefficient use of the distillation budget and overlooking the inter-class features crucial for generalization.



### SceneGPT: A Language Model for 3D Scene Understanding (https://arxiv.org/abs/2408.06926)
Comments:
          UBC Report

- **What's New**: 이 연구는 3D 사전 훈련 없이 사전 훈련된 언어 모델(LLM)의 지식을 3D 장면 이해에 활용할 수 있는 방법을 제시합니다. SceneGPT라는 프레임워크를 통해 3D 공간 추론을 위한 LLM의 가능성을 탐구하고, 이를 위한 적절한 프롬프팅 전략을 제안합니다.



### Divide and Conquer: Improving Multi-Camera 3D Perception with 2D Semantic-Depth Priors and Input-Dependent Queries (https://arxiv.org/abs/2408.06901)
Comments:
          Accepted by TIP 2024

- **What's New**: 이 논문은 멀티 카메라 이미지를 사용하여 3D 객체 감지 및 조감도(BEV) 분할과 같은 3D 인식 작업을 위한 새로운 트랜스포머 기반 프레임워크인 SDTR(Semantics and Depth as Priors)를 제안합니다. SDTR은 의미론적(semantic) 및 깊이(depth) 정보를 우선 정보로 사용하여, 객체 분류 및 위치 추정을 분리하여 더 정확한 결과를 얻습니다.



### EE3P3D: Event-based Estimation of Periodic Phenomena Frequency using 3D Correlation (https://arxiv.org/abs/2408.06899)
Comments:
          15 paper pages + 11 suppl pages, 15 figues, 4 tables

- **What's New**: 이 논문은 이벤트 카메라를 사용하여 주기적 현상(예: 회전, 깜빡임, 진동)의 주파수를 측정하는 새로운 방법을 제시합니다. 이벤트 카메라는 높은 시간 해상도로 독립적으로 작동하는 픽셀에서 밝기 변화를 비동기적으로 보고하는 장치입니다. 이 방법은 주기적 현상의 경우 특정 공간-시간 창 내에서 현상의 주기에 해당하는 시간 차이로 매우 유사한 이벤트 집합이 생성된다는 가정을 기반으로 합니다. 유사한 이벤트 집합은 이벤트 스트림 공간에서 3D 공간-시간 상관관계로 감지됩니다. 제안된 방법인 EE3P3D는 깜박이는 빛과 진동, 회전과 같은 12개의 주기적 현상 시퀀스 데이터 세트에서 평가되었으며 3.2Hz에서 2kHz(192~120,000RPM에 해당)의 범위를 포함합니다. EE3P3D는 이 데이터 세트에서 기존 방법을 크게 능가하며 평균 상대 오차 0.1%를 달성합니다.



### PBIR-NIE: Glossy Object Capture under Non-Distant Lighting (https://arxiv.org/abs/2408.06878)
- **What's New**: 본 논문은 PBIR-NIE라는 새로운 역 렌더링 프레임워크를 소개하며, 자연광 하에서 다중 뷰 이미지로부터 광택이 있는 객체의 기하학, 재질 속성 및 주변 조명을 포괄적으로 캡처하도록 설계되었습니다. 이 프레임워크는 표준 무한 거리 환경 맵보다 복잡한 시차 효과를 수용할 수 있는 새로운 시차 인식 비 원거리 환경 맵(parallax-aware non-distant environment map)을 제안하여 가볍고 효율적인 조명 표현을 제공합니다. 또한, 본 논문은 물리 기반 미분 렌더링을 통해 기본적인 부호화된 거리 필드(SDF)를 최적화하고, 신경 암시적 진화(NIE)를 통해 삼각형 메시와 SDF 사이의 표면 기울기를 원활하게 연결합니다. 고광택 BRDF의 미분 렌더링에서 복잡성을 해결하기 위해, 본 논문은 반대 샘플링 알고리즘을 통합하여 몬테카를로 기울기 추정기의 분산을 완화합니다. 이로 인해 프레임워크는 광택이 있는 객체 재구성을 처리하는 데 강력한 기능을 보여주며, 기하학, 재조명 및 재질 추정에서 뛰어난 품질을 선보입니다.



### A Comprehensive Survey on Synthetic Infrared Image synthesis (https://arxiv.org/abs/2408.06868)
Comments:
          Submitted in Journal of Infrared Physics & Technology

- **What's New**: This survey paper comprehensively reviews existing methods for synthetic infrared (IR) scene and target generation, encompassing both traditional mathematical modeling and deep learning approaches. It highlights the importance of synthetic IR data for training and testing various applications like remote sensing, surveillance, and target recognition, especially given the scarcity of real-world IR data. The paper emphasizes the need for further research in this field to enhance the efficiency and effectiveness of synthetic IR generation.



### Dynamic and Compressive Adaptation of Transformers From Images to Videos (https://arxiv.org/abs/2408.06840)
- **What's New**: 본 논문에서는 이미지-비디오 변환에서 효율성을 개선하기 위해, **동적 프레임 간 토큰 보간 (Inter-frame Token Interpolation)**을 사용하는 새로운 압축 이미지-비디오 변환 방법인 **InTI**를 소개합니다.

- **Technical Details**: InTI는 이웃 프레임의 동일한 위치에 있는 토큰 쌍을 선형적으로 결합하여 새로운 토큰을 생성합니다. 이때, 결합 가중치는 다중 스케일 컨텍스트 인식 네트워크에 의해 동적으로 생성됩니다. 이를 통해, 이웃 프레임의 정보를 공간-시간적 인식을 통해 지점별로 적응적으로 압축할 수 있으며, 처리되는 프레임 수를 매번 절반으로 효과적으로 줄일 수 있습니다.

- **Performance Highlights**: Kinetics-400 데이터셋에서 InTI는 기존 변환 방식과 비교하여 **GFLOPs를 37.5% 감소**시키면서 **정확도 87.1%**를 달성했습니다. InTI는 추가적인 시간적 모듈과 결합되었을 때 **GFLOPs를 37% 감소**시키면서 **정확도 87.6%**를 달성했습니다. 이와 유사한 결과는 다른 일반적인 데이터셋에서도 확인되었습니다. 



### GLGait: A Global-Local Temporal Receptive Field Network for Gait Recognition in the Wild (https://arxiv.org/abs/2408.06834)
Comments:
          Accepted by ACM MM2024

- **What's New**: 이 논문은 야외 환경에서 보행 인식 성능을 향상시키기 위해 'Global-Local Temporal Receptive Field Network (GLGait)'를 제안합니다. GLGait는 'Global-Local Temporal Module (GLTM)'을 사용하여 전역-지역 시간 수용 필드를 구축하고, 이는 'Pseudo Global Temporal Self-Attention (PGTA)'와 시간적 합성곱 연산으로 구성됩니다.

- **Technical Details**: PGTA는 다중 헤드 셀프 어텐션 (MHSA)에 비해 메모리와 계산 복잡성을 줄여 전역 시간 수용 필드를 구축합니다. 시간적 합성곱 연산은 지역 시간 수용 필드를 강화하고 전역 시간 수용 필드를 통합합니다. GLGait는 또한 훈련 과정에서 클래스 내 거리를 줄이고 양성 샘플을 확장하기 위해 'Center-Augmented Triplet Loss (CTL)'를 사용합니다.

- **Performance Highlights**: GLGait는 'Gait3D'와 'GREW'와 같은 야외 데이터셋에서 최첨단 성능을 달성했습니다.



### FlatFusion: Delving into Details of Sparse Transformer-based Camera-LiDAR Fusion for Autonomous Driving (https://arxiv.org/abs/2408.06832)
- **What's New**: 이 논문은 스파스 트랜스포머 기반 카메라-라이더 퓨전을 위한 새로운 설계 원칙과 구조를 제시하는 FlatFusion을 소개합니다. FlatFusion은 이미지-to-3D 및 라이더-to-2D 매핑, 어텐션 이웃 그룹화, 단일 모드 토크나이저, 트랜스포머의 미세 구조를 포함한 다양한 설계 선택 사항을 심층적으로 분석하고 비교하여 최적의 설계를 찾아냈습니다. 결과적으로 FlatFusion은 UniTR, CMT, SparseFusion과 같은 기존의 스파스 트랜스포머 기반 방법들을 능가하는 성능을 보여주며, nuScenes 검증 세트에서 73.7 NDS를 달성하고 PyTorch를 사용하여 10.1 FPS의 속도를 유지합니다.

- **Technical Details**: FlatFusion은 이미지 백본과 라이더 백본을 모두 활용하여 각 모드의 정보를 효율적으로 추출합니다. 특히, 이미지 백본으로는 ResNet18+FPN과 VoVNet+FPN을 사용하여 가벼우면서도 성능이 좋은 모델을 선택했습니다. 라이더 백본으로는 VoxelNet을 사용하여 포인트 클라우드 데이터를 3차원 공간으로 변환합니다. 두 모드의 정보를 융합하기 위해, FlatFusion은 2D 이미지 평면과 3D 공간의 두 가지 표현 공간을 사용합니다. 이미지-to-3D 매핑에는 부분 투영(partial projection)을 사용하여 2D 이미지 정보를 3D 공간으로 효율적으로 변환합니다. 반대로, 라이더-to-2D 매핑에는 Voxel 기반 3D-to-2D 퓨전을 사용하여 3D 포인트 클라우드 정보를 2D 이미지 평면으로 투영합니다. 이러한 퓨전 전략을 통해 두 모드의 정보를 효과적으로 결합할 수 있습니다. 또한, FlatFusion은 3D 위치 인코딩(PE)을 포함하는 PreNorm 트랜스포머 구조를 사용하여 위치 정보를 유지합니다. Window 파티션 알고리즘은 Flatten Window 파티션 알고리즘을 사용하여 계산 복잡성을 줄입니다.

- **Performance Highlights**: FlatFusion은 nuScenes 검증 세트에서 73.7 NDS를 달성하여 기존의 스파스 트랜스포머 기반 방법들보다 뛰어난 성능을 보여줍니다. 이는 기존 방법들보다 더 정확하고 효율적인 스파스 카메라-라이더 퓨전 방법을 제공한다는 것을 의미합니다. 또한, FlatFusion은 PyTorch를 사용하여 10.1 FPS의 처리 속도를 유지하며 실시간 애플리케이션에 적합합니다. FlatFusion은 다양한 성능 평가 지표에서 뛰어난 결과를 보여주며, 향후 자율 주행 시스템의 3D 객체 인식 성능을 향상시킬 수 있는 잠재력이 높습니다.



### Photometric Inverse Rendering: Shading Cues Modeling and Surface Reflectance Regularization (https://arxiv.org/abs/2408.06828)
Comments:
          Project page: https://jzbao03.site/projects/PIR/

- **What's New**: This paper proposes a new neural inverse rendering method for photometric images, using a point light source. It tackles the limitations of existing approaches by jointly optimizing the light source position to account for self-shadows, and using a differentiable rendering layer and importance sampling for inter-reflections. This method also introduces a novel regularization by distilling DINO features to improve material decomposition.



### Structure-preserving Planar Simplification for Indoor Environments (https://arxiv.org/abs/2408.06814)
- **What's New**: 이 논문은 시뮬레이션 및 실제 환경 모두를 위한 실내 장면 포인트 클라우드의 구조 유지 평면 단순화를 위한 새로운 접근 방식을 제시합니다. 구조화된 장면(벽, 천장, 바닥)과 비구조화된 장면(실내 물체)으로 장면 포인트 클라우드를 분할합니다. RANSAC 알고리즘을 활용하여 입력 포인트 클라우드에서 기본 평면을 추출하여 구조화된 장면의 분할 및 단순화를 용이하게 합니다. 그런 다음 최적의 벽 메시를 기본 평면에서 생성하고, 정점 이동 알고리즘을 사용하여 인접한 메시를 병합하여 메시 레이아웃을 보존합니다. 천장과 바닥을 정확하게 표현하기 위해, 벽 법선에 대한 천장과 바닥 메시를 클리핑하는 메시 클리핑 알고리즘을 사용합니다. 실내 장면의 경우, 표면 재구성 기술을 적용하여 충실도를 향상시킵니다. 이 논문은 다층 및 경사진 벽과 천장과 같은 복잡한 시나리오를 다루는 제안된 장면 단순화 방법론의 복잡한 단계에 중점을 둡니다. 또한 인기 있는 표면 재구성, 모양 근사 및 평면도 생성 접근 방식에 대한 정성적 및 정량적 성능 비교를 수행합니다.



### Oracle Bone Script Similiar Character Screening Approach Based on Simsiam Contrastive Learning and Supervised Learning (https://arxiv.org/abs/2408.06811)
- **What's New**: 본 프로젝트는 ResNet-50 자기지도 학습(self-supervised learning)과 RepVGG 지도 학습(supervised learning)을 융합하기 위해 퍼지 종합 평가 방법(fuzzy comprehensive evaluation method)을 활용하는 새로운 방법을 제안합니다. HWOBC 오라클(oracle) 이미지 데이터셋을 입력으로 사용하고, 타겟 이미지를 선택하며, 마지막으로 수동 개입 없이 가장 유사한 이미지를 출력합니다.  모달리티(modality)가 다른 이미지에는 동일한 특징 인코딩(feature encoding) 방법을 사용하지 않습니다. 모델 훈련 전에 이미지 데이터를 전처리하고, 랜덤 회전 처리(random rotation processing), 자기제곱 그래프 균등화 이론 알고리즘(self-square graph equalization theory algorithm), 감마 변환(gamma transform)을 통해 이미지를 향상시켜 핵심 특징 학습(key feature learning)을 효과적으로 향상시킵니다. 마지막으로, 퍼지 종합 평가 방법을 사용하여 지도 학습(supervised training)과 비지도 학습(unsupervised training) 결과를 결합하여 정량화하기 어려운 '가장 유사한' 문제를 더 잘 해결할 수 있습니다. 현재 많은 미지의 오라클 뼈 글자가 우리가 풀어야 할 숙제로 남아 있습니다. 글자와의 접촉은 해독을 위한 새로운 아이디어를 제공할 수 있습니다.



### Unmasking the Uniqueness: A Glimpse into Age-Invariant Face Recognition of Indigenous African Faces (https://arxiv.org/abs/2408.06806)
Comments:
          Keywords: Age-Invariant Face Recognition, CACD, FAGE_v2, VGGFace

- **What's New**: 이 논문은 아프리카 원주민 얼굴의 연령 불변 얼굴 인식(AIFR) 시스템을 개발하여 기존 연구에서 아프리카 민족을 잘못 대표하는 문제를 해결합니다. 연구진은 이 연구를 위해 수집한 5,000명의 아프리카 원주민 얼굴 데이터셋(FAGE_v2)에 대해 사전 훈련된 딥 러닝 모델(VGGFace)을 사용했습니다.



### Integrating Saliency Ranking and Reinforcement Learning for Enhanced Object Detection (https://arxiv.org/abs/2408.06803)
Comments:
          Resultant work from Dissertation, Department of AI, University of Malta. Code available at: this https URL

- **What's New**: 이 연구는 강화 학습(RL) 기반의 시각적 주의 방법과 급격성 순위(saliency ranking) 기술을 결합하여 투명하고 지속 가능한 솔루션을 탐구하는 일련의 실험을 수행합니다. 이 연구는 초기 경계 상자 예측을 위해 급격성 순위를 통합하고 여러 시간 단계에 걸쳐 유한한 작업 집합을 통해 이러한 예측을 미세 조정하기 위해 RL 기법을 적용하여 RL 객체 감지 정확도를 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 다양한 이미지 특징 추출 방법을 사용하고 다양한 심층 Q-네트워크(DQN) 아키텍처 변형을 탐구하여 심층 강화 학습 기반의 현지화 에이전트 훈련을 수행합니다. 또한 이 연구는 이전 RL 접근 방식에 없는 기능인 감지된 객체를 분류하는 기능을 통합하면서, 가볍고 빠른 모델을 우선시하여 모든 단계에서 감지 파이프라인을 최적화하는 데 중점을 둡니다.

- **Performance Highlights**: Pascal VOC 2007 데이터 세트를 사용하여 이러한 훈련된 에이전트의 성능을 평가함으로써 더 빠르고 최적화된 모델이 개발되었습니다. 특히 이 연구에서 달성된 최고의 평균 정밀도(mAP)는 51.4로, 문헌에서 RL 기반 단일 객체 검출기가 설정한 벤치 마크를 능가했습니다.



### Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning (https://arxiv.org/abs/2408.06798)
Comments:
          Accepted to ECCV2024

- **What's New**: 이 논문에서는 토큰 압축 (token compression)의 성능 저하 문제를 해결하기 위해 새로운 모델 산술 프레임워크 (model arithmetic framework)를 제안합니다. 이 프레임워크는 토큰 압축 비율을 학습 단계와 추론 단계에서 분리하여 성능 저하 문제를 해결합니다. 특히, 사전 학습된 모델에 작은 플러그인 모델인 ToCom (Token Compensator)을 추가하여, 압축 비율 간의 차이를 해소합니다. ToCom은 사전 학습된 모델에 직접 삽입하여 추가 학습 없이 다양한 압축 비율에서 범용적인 성능 향상을 제공합니다.



### Visual Neural Decoding via Improved Visual-EEG Semantic Consistency (https://arxiv.org/abs/2408.06788)
- **What's New**: 본 논문에서는 시각적 신경 해독(Visual neural decoding) 성능을 향상시키기 위해 시각 자극과 뇌 활동 사이의 의미적 일관성(Semantic consistency)을 강화하는 새로운 접근 방식인 시각-EEG 의미 분리 프레임워크(Visual-EEG Semantic Decouple Framework, VE-SDN)를 제안합니다. VE-SDN은 시각 이미지와 EEG 신호의 의미 관련 특징을 분리하고 이들을 직접 정렬하여 의미적 일관성을 높입니다. 또한, 이 프레임워크는 상호 정보 극대화-최소화 적대적 학습(Mutual information maximization-minimization adversarial learning) 기반 방법을 사용하여 모달리티에서 의미 관련 정보 추출을 유도합니다. 추가적으로, 본 논문에서는 시각-EEG 사이의 의미적 상관관계를 개선하기 위해 교차 모달 순환 재구성(Cross-modal cyclic reconstruction)을 활용합니다. 이러한 접근 방식은 학습 과정에서 도메인 특징의 퇴화(Degenerate) 사례를 방지하고, 시각 이미지와 EEG 특징 사이의 상호 정보(Mutual information, MI) 값을 정량화하여 성능 향상과의 강력한 상관관계를 확인합니다. 마지막으로, 본 논문은 인간의 시각적 개념 인코딩을 모방하여 시각적 샘플이 같은 클래스 내에서 일관성을 유지하도록 하는 기하학적 일관성 손실(Geometric consistency loss)을 도입하여, 시각-EEG 정렬의 견고성을 높입니다.



### Do Vision-Language Foundational models show Robust Visual Perception? (https://arxiv.org/abs/2408.06781)
Comments:
          UBC Report

- **What's New**: 본 연구는 비전-언어 기반 모델(vision-language foundational models)의 실제 환경에서의 견고성(robustness)을 평가합니다. 특히, 흔히 발생하는 이미지 왜곡(corruptions, 예: 모션 블러, 안개, 눈, 가우시안 노이즈)에 대한 모델의 성능 변화를 분석합니다.

- **Technical Details**: 본 연구는 다양한 비전-언어 기반 모델을 대상으로, 이미지 왜곡(corruptions)에 대한 견고성을 평가하기 위해 영상 분류(image classification) 작업을 수행합니다. 모델의 유형은 크게 지도 학습 비전 모델(Supervised Vision Models)과 다중 모드 기반 모델(Multimodal Foundational Models)로 분류되며, 다중 모드 기반 모델은 다시 대조 학습 기반 모델(Contrastive Multi-Encoder models), 인코더-디코더 생성 모델(Encoder-Decoder Generative models), 하이브리드 모델(Hybrid-Models)로 세분화됩니다.

- **Performance Highlights**: 연구 결과는 다중 모드 기반 모델이 지도 학습 비전 모델에 비해 이미지 왜곡에 대한 견고성이 높음을 보여줍니다. 특히, 대조 학습 기반 모델은 왜곡된 이미지에 대한 뛰어난 성능을 보이는 반면, 인코더-디코더 생성 모델은 왜곡된 이미지에 대한 성능 저하를 보입니다. 하이브리드 모델은 대조 학습과 언어 모델링을 결합하여 왜곡된 이미지에 대한 견고성을 향상시킵니다.



### ED$^4$: Explicit Data-level Debiasing for Deepfake Detection (https://arxiv.org/abs/2408.06779)
- **What's New**: 이 논문은 기존 딥페이크 탐지 모델의 일반화 성능을 저해하는 새로운 공간 편향 (Spatial Bias) 문제를 제시하고, 이를 해결하기 위한 효과적인 데이터 수준의 탈편향 전략 (Explicit Data-level Debiasing)인 ED$^4$를 소개합니다. ED$^4$는 기존의 암묵적인 탈편향 (Implicit Disentanglement) 방식 대신 명시적인 데이터 수준에서 편향을 해결하는 방법을 채택하여, 콘텐츠 편향 (Content Bias), 특정 위조 편향 (Specific-Forgery Bias) 및 공간 편향 (Spatial Bias)을 동시에 해결합니다.

- **Technical Details**: ED$^4$는 두 가지 핵심 모듈인 ClockMix와 AdvSCM을 통해 편향 문제를 해결합니다. ClockMix는 다양한 이미지를 시계 방향으로 혼합 (Clockwise Mixing)하여, 콘텐츠 편향과 특정 위조 편향을 해결합니다. AdvSCM은 공간 편향을 제거하기 위해, 공간적으로 불일치하는 이미지를 생성하여 탐지기가 공간적 불일치에도 일관된 위조 특징을 학습하도록 유도합니다.

- **Performance Highlights**: ED$^4$는 다양한 딥페이크 탐지 모델에 적용 가능한 플러그 앤 플레이 (Plug-and-Play) 방식으로, 기존 모델의 성능을 크게 향상시키는 것으로 입증되었습니다. 또한, ClockMix를 통해 다양한 얼굴 이미지를 혼합하여 데이터 증강 효과를 높이고, AdvSCM을 통해 탐지기의 공간적 편향을 제거하여 일반화 성능을 향상시키는 것으로 나타났습니다.



### Exploring Domain Shift on Radar-Based 3D Object Detection Amidst Diverse Environmental Conditions (https://arxiv.org/abs/2408.06772)
Comments:
          6 pages, 5 figures, 3 tables, accepted in IEEE International Conference on Intelligent Transportation Systems (ITSC) 2024

- **What's New**: 본 논문은 자율 주행 시스템에서 4D 레이더 기반 객체 탐지에 대한 도메인 이동(domain shift) 문제를 심층적으로 분석한 연구입니다. 기존 연구와 달리, 다양한 기상 조건(예: 비, 눈, 안개)과 도로 유형(고속도로, 도시)이 3D 객체 탐지 성능에 미치는 영향을 종합적으로 살펴보았습니다. 특히, 레이더 포인트 클라우드 생성 과정에서 발생하는 도메인 이동을 명확하게 보여주는 것이 핵심입니다.



### Cross-View Geolocalization and Disaster Mapping with Street-View and VHR Satellite Imagery: A Case Study of Hurricane IAN (https://arxiv.org/abs/2408.06761)
- **What's New**: 본 논문에서는 **CVDisaster**라는 새로운 재해 매핑 프레임워크를 제안합니다. 이 프레임워크는 **Street-View Imagery (SVI)**와 **Very High-Resolution 위성 이미지**를 사용하여 **지리 위치**와 **피해 인식**을 동시에 추정합니다. CVDisaster는 **CVDisaster-Geoloc** (지리 위치 모델)와 **CVDisaster-Est** (피해 인식 모델) 두 가지 모델로 구성되어 있습니다. CVDisaster-Geoloc은 **Siamese ConvNeXt 이미지 인코더**를 사용하는 **대조 학습** 기반 모델이고, CVDisaster-Est는 **Couple Global Context Vision Transformer (CGCViT)** 기반 모델입니다.



### Sumotosima: A Framework and Dataset for Classifying and Summarizing Otoscopic Images (https://arxiv.org/abs/2408.06755)
Comments:
          Work in Progress

- **What's New**: 이 논문에서는 청각기 이미지를 이해하고 요약하는 새로운 딥 러닝 및 트랜스포머 기반 프레임워크인 Sumotosima(Otoscopic 이미지 요약기)를 제안합니다. Sumotosima는 환자에게 적합한 요약을 제공하여 청각기 이미지를 명확하고 효율적으로 이해하도록 돕는 것을 목표로 합니다. 이를 위해 삼중 손실(triplet loss)과 교차 엔트로피 손실(cross-entropy loss)의 조합을 사용하고, 텍스트 및 이미지 임베딩(embedding)을 결합한 지식 증강 다중 모달 BART(Knowledge Enhanced Multimodal BART)를 활용합니다. 데이터셋 부족 문제를 해결하기 위해 5가지 독특한 범주에 대한 이미지 500개와 이에 대한 요약이 포함된 OCASD(Otoscopic Classification And Summary Dataset) 데이터셋을 구축했습니다.



### Detecting Audio-Visual Deepfakes with Fine-Grained Inconsistencies (https://arxiv.org/abs/2408.06753)
Comments:
          Accepted in BMVC 2024

- **What's New**: 이 논문에서는 영상 및 음성 데이터 간의 미세한 차이를 포착하는 미세 조정 메커니즘을 도입하여 오디오-비주얼 딥페이크 탐지 성능을 향상시키는 새로운 방법을 제시합니다. 기존 방법들은 고수준의 특징에 집중하여 딥페이크를 탐지하는 반면, 이 논문은 딥페이크 고유의 미세한 오디오-비주얼 아티팩트(artifact)를 더 효과적으로 탐지하기 위해 공간 및 시간 영역에서 미세 조정 메커니즘을 활용합니다.



### ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation (https://arxiv.org/abs/2408.06747)
Comments:
          Extended version of our CVPR 24 paper

- **What's New**: CLIP을 이용한 비지도 의미 분할 (Unsupervised Semantic Segmentation)에서 발생하는 공간 선호 편향 (Space-preference Bias)과 클래스 선호 편향 (Class-preference Bias)을 명시적으로 모델링하고 수정하여 성능을 향상시킨 ReCLIP++ 모델을 제안합니다. 특히, 학습 가능한 '참조(Reference)' 프롬프트를 사용하여 클래스 선호 편향을 인코딩하고 비전 트랜스포머 (Vision Transformer)의 위치 임베딩 (Positional Embedding)을 투영하여 공간 선호 편향을 인코딩합니다. 이러한 편향들은 독립적으로 인코딩되며, 두 가지 편향을 명시적으로 나타내는 편향 로짓 맵 (Bias Logit Map)을 생성하기 위해 행렬 곱셈 (Matrix Multiplication)을 수행합니다. 그런 다음 원래 CLIP의 로짓에서 편향 로짓을 빼서 편향을 수정합니다. 수정된 결과를 더 부드럽고 문맥적으로 만드는 마스크 디코더 (Mask Decoder)가 설계되어, CLIP의 특징과 수정된 로짓을 입력으로 받아 Gumbel-Softmax 연산을 통해 수정된 분할 마스크 (Rectified Segmentation Mask)를 출력합니다.



### Long-Tailed Out-of-Distribution Detection: Prioritizing Attention to Ta (https://arxiv.org/abs/2408.06742)
- **What's New**: This paper proposes a new method called Prioritizing Attention to Tail (PATT) to improve out-of-distribution (OOD) detection in long-tailed image classification. PATT addresses the issue of imbalanced data distribution by using implicit semantic augmentation contrastive learning (TISAC) and post-hoc feature calibration, which helps to enhance the distinction between in-distribution (ID) and OOD samples.



### Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspectiv (https://arxiv.org/abs/2408.06741)
- **What's New**: 본 논문은 합성 이미지 탐지(SID)에서 기존 훈련 방식의 두 가지 편향, 즉 약화된 아티팩트 특징과 과적합된 아티팩트 특징을 지적합니다. 또한, 합성 이미지의 영상 메커니즘이 픽셀 간의 높은 지역적 상관 관계를 야기한다는 점을 발견하고, 이를 감안하여 지역적 인식(local awareness)을 갖춘 탐지기를 제안합니다. 이를 위해 SAFE라는 가볍고 효과적인 탐지기를 제안하며, 이는 세 가지 간단한 이미지 변환을 통합합니다. 첫째, 약화된 아티팩트 특징을 해결하기 위해 이미지 전처리에서 다운샘플링 연산자를 크롭 연산자로 대체하여 아티팩트 왜곡을 방지합니다. 둘째, 과적합된 아티팩트 특징을 해결하기 위해 ColorJitter와 RandomRotation을 추가적인 데이터 증강으로 도입하여 제한된 훈련 샘플에서 색상 차이와 의미적 차이에 대한 무관한 편향을 완화합니다. 셋째, 지역적 인식을 위해 SID에 맞춤화된 패치 기반 랜덤 마스킹 전략을 제안하여 훈련 중 탐지기가 지역 영역에 집중하도록 합니다.



### DiffLoRA: Generating Personalized Low-Rank Adaptation Weights with Diffusion (https://arxiv.org/abs/2408.06740)
Comments:
          9 pages,8 figures

- **What's New**: 본 논문은 'DiffLoRA'라는 새로운 개념을 도입하여 기존 텍스트-이미지 생성 모델의 개인화 문제를 해결합니다. DiffLoRA는 이미지 참조를 기반으로 저랭크 적응(LoRA) 가중치를 예측하는 확산 모델을 활용합니다. 이를 통해 별도의 학습 없이 추론 단계에서 개인화를 수행하며, 기존 모델의 생성 성능을 유지하면서도 신원 일치성을 높일 수 있습니다.

- **Technical Details**: DiffLoRA는 LoRA 가중치를 예측하기 위한 하이퍼네트워크로 확산 모델을 활용합니다. LoRA 가중치를 압축 및 재구성하기 위한 LoRA 가중치 오토인코더와, 신원 추출 기능을 강화하기 위해 혼합 전문가(MoE)에서 영감을 받은 게이트 네트워크를 도입했습니다. 또한 DiffLoRA 학습을 위한 다양한 신원의 LoRA 가중치 데이터셋을 생성하는 파이프라인을 제안했습니다.

- **Performance Highlights**: 실험 결과, DiffLoRA는 기존 최첨단 방법들보다 텍스트-이미지 일관성, 신원 일치성, 생성 품질, 추론 비용 측면에서 뛰어난 성능을 보였습니다. DiffLoRA는 추론 단계에서 추가적인 계산 비용 없이 고품질 개인화된 초상화를 생성할 수 있으며, LoRA는 일반적인 미세 조정 방법이기 때문에 다른 PEFT(Parameter-Efficient Fine-Tuning) 방법과 통합하여 다양한 작업에 적용할 수 있습니다.



### Response Wide Shut: Surprising Observations in Basic Vision Language Model Capabilities (https://arxiv.org/abs/2408.06721)
Comments:
          Under Submission

- **What's New**: 이 논문은 최첨단 비전-언어 모델(VLMs)의 기본적인 시각적 이해 능력을 심층적으로 분석합니다. 기존의 성능 측정 방식을 넘어, 시각 인코더에서 추출된 특징을 사용하여 훈련된 프로브와 비전-언어 프로젝션을 통해 이미지 인코더와 LLM 디코더의 출력을 연결하는 중간 단계의 비전-언어 프로젝션을 비교하여 분석합니다. 이를 통해, VLMs의 응답에서 나타나는 초기적인 단점을 밝혀내고, 더 효과적인 VLM 모델을 훈련 및 개발하는 데 도움이 되는 중요한 관찰 결과를 제시합니다.

- **Technical Details**: 본 연구에서는 VLMs의 기본적인 시각적 이해 능력을 탐구하기 위해 다양한 공간(visual latent, vision-language shared latent, language response space)에 대한 분석을 수행합니다. 이러한 공간들은 VLMs 내에서 시각 정보를 처리하는 과정에서 정보를 나타내는 중간 단계를 나타냅니다.  VLMs의 성능을 분석하기 위해, 각 공간에서 객체 인식, 공간 배열 이해, 객체 인스턴스 구분 (계산)과 같은 기본적인 시각적 작업을 수행하는 프로브를 훈련합니다. 또한, VLMs의 설계 구성 요소 중 어떤 부분이 부족한지 파악하기 위해, VLMs의 설계 구성 요소별로 성능을 비교 분석합니다.

- **Performance Highlights**: 연구 결과, VLMs는 (1) 세분화된 카테고리보다는 일반적인 카테고리의 객체를 더 잘 인식하고, (2) 시각적 특징과 텍스트 잠재 공간에는 객체 계산에 필요한 정보가 존재하지만 응답 공간의 성능은 상대적으로 낮으며, (3) 시각 인코더는 공간 배열 정보를 잘 포착하지 못한다는 것을 밝혀냈습니다. 특히,  VLMs의 언어 디코더는 명령어 미세 조정을 통해 이러한 성능 격차를 어느 정도 해소하는 것으로 나타났습니다. VLMs의 성능 향상을 위해, 시각-언어 프로젝션 및 언어 디코더와 관련된 기술 향상과 시각 인코더의 공간적 정보 처리 능력을 개선하는 노력이 필요합니다.



### Multimodal Analysis of White Blood Cell Differentiation in Acute Myeloid Leukemia Patients using a \beta-Variational Autoencoder (https://arxiv.org/abs/2408.06720)
Comments:
          Accepted for publication at MICCAI 2024 workshop on AI for Imaging Genomics Learning (AIIG)

- **What's New**: 이 연구는 백혈병과 같은 백혈구 질환에 대한 이해를 향상시키기 위해 단일 세포 수준에서 형태학적 데이터와 전사체 데이터를 결합하는 새로운 방법을 제시합니다. 기존의 방법론은 형태학적 데이터와 전사체 데이터를 통합하는 데 어려움을 겪고 있어 세포 분화 역학을 포괄적으로 이해하는 데 큰 연구 격차를 남겼습니다. 이 연구는 이러한 두 가지 모달리티를 탐색하고 재구성하며 사람 말초 혈액 도말에서 백혈구의 다양한 하위 유형 간의 관계를 형태학 및 해당 전사체 측면에서 밝히는 비지도 학습 방법을 소개합니다.



### Towards Cross-Domain Single Blood Cell Image Classification via Large-Scale LoRA-based Segment Anything Mod (https://arxiv.org/abs/2408.06716)
- **What's New**: This paper presents a novel approach called BC-SAM (Blood Cell Segment Anything Model) for classifying blood cell images. BC-SAM utilizes the large-scale foundation model Segment Anything Model (SAM) and incorporates fine-tuning using LoRA (Low-Rank Adaptation) to extract general image embeddings from blood cell images. To improve BC-SAM's performance across different datasets, an unsupervised cross-domain autoencoder is introduced to learn intrinsic features while suppressing artifacts. BC-SAM outperforms existing state-of-the-art methods for blood cell classification.



### Review Learning: Advancing All-in-One Ultra-High-Definition Image Restoration Training Method (https://arxiv.org/abs/2408.06709)
- **What's New**: 이 논문은 **리뷰 학습(Review Learning)**이라는 새로운 학습 패러다임을 제안하여 **프롬프트(prompt) 없이도 여러 종류의 이미지 왜곡(degradation)을 처리할 수 있는 일반적인 이미지 복원 모델을 훈련합니다.** 이 방법은 여러 왜곡된 데이터셋으로 이미지 복원 모델을 순차적으로 훈련하고, 이전 왜곡된 데이터셋의 여러 클래스에 대한 모델의 기억을 향상시키는 리뷰 메커니즘을 결합합니다. 또한, 소비자급 GPU에서 4K(3840 × 2160) 해상도의 왜곡된 이미지를 효율적으로 처리할 수 있는 경량의 범용 이미지 복원 네트워크를 설계합니다.



### MAIR++: Improving Multi-view Attention Inverse Rendering with Implicit Lighting Representation (https://arxiv.org/abs/2408.06707)
- **What's New**: 이 논문은 MAIR(Multi-view Attention Inverse Rendering)의 확장판인 MAIR++를 제안합니다. MAIR++는 장면 레벨 멀티뷰 역 렌더링(Scene-level Multi-view Inverse Rendering)의 정확성과 현실감을 향상시키기 위해, 기존 MAIR의 제한점을 해결하는 새로운 방법들을 도입합니다.

- **Technical Details**: MAIR++는 다음과 같은 핵심적인 기술적 개선점을 가지고 있습니다:
- **Implicit Lighting Representation (ILR):** 각 픽셀의 전체 입사광을 특징 벡터로 표현하는 새로운 광원 표현 방식. ILR은 방향 디코더를 통해 환경 맵으로 디코딩되거나, 신경 렌더러를 통해 사실적인 이미지를 렌더링할 수 있습니다.
- **Directional Attention Module (DAM):** 멀티뷰 이미지에서 ILR 정보를 철저히 분석하기 위해, 새로운 방향 주의 모듈을 설계했습니다.
- **Albedo Fusion Module (AFM):** 단일 뷰와 멀티뷰 알베도를 통합하여 두 알베도 맵의 단점을 보완합니다.

- **Performance Highlights**: 실험 결과 MAIR++는 MAIR 및 단일 뷰 기반 방법보다 더 나은 성능을 달성했으며, 보이지 않는 실제 장면에서도 견고한 성능을 보여줍니다. 특히, MAIR++는 MAIR가 가진 균일 샘플링 렌더링, 기존 반사성 유지 실패, 사실적인 사과 음영 렌더링 부족 등의 문제점을 해결하여, 사실적인 재질 편집을 가능하게 합니다.



### SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields (https://arxiv.org/abs/2408.06697)
Comments:
          Accepted by ECCV 2024. Project website: this https URL

- **What's New**: SlotLifter, a novel object-centric radiance model for 3D scene reconstruction and decomposition, leverages slot-guided feature lifting to unite object-centric learning representations with image-based rendering methods. It achieves superior performance in both scene decomposition and novel-view synthesis compared to existing 3D object-centric models, especially on challenging real-world datasets like ScanNet and DTU.

- **Technical Details**: SlotLifter utilizes a slot-guided feature lifting mechanism that initializes 3D point features based on lifted 2D input-view features. This design improves detail granularity for novel-view synthesis and provides explicit guidance for slot learning.  The model employs a cross-attention-based transformer to predict volume rendering parameters by interacting lifted features with slot representations. It only relies on the reconstruction loss and requires fewer sampling overheads during training, leading to greater efficiency compared to existing models.

- **Performance Highlights**: SlotLifter significantly outperforms existing 3D object-centric models, achieving state-of-the-art performance in both scene decomposition (∼10+ ARI) and novel-view synthesis (∼2+ PSNR) on synthetic and real-world datasets. It demonstrates superior performance against image-based rendering methods on complex real-world datasets such as ScanNet and DTU. Ablative studies reveal the effectiveness of each module, highlighting SlotLifter's potential for object-centric learning and image-based rendering.



### DC3DO: Diffusion Classifier for 3D Objects (https://arxiv.org/abs/2408.06693)
- **What's New**: This paper presents **DC3DO**, a novel method for 3D object classification using **diffusion models** that achieves zero-shot classification without additional training. This approach leverages the density estimates from 3D diffusion models, demonstrating superior multimodal reasoning compared to discriminative methods.



### Masked Image Modeling: A Survey (https://arxiv.org/abs/2408.06687)
- **What's New**: 본 논문은 **마스크 이미지 모델링(Masked Image Modeling, MIM)**의 최신 연구들을 조사하여, 컴퓨터 비전 분야에서 강력한 **자기 지도 학습(self-supervised learning)** 기법으로 떠오르고 있는 MIM의 다양한 측면을 분석합니다. MIM은 이미지의 일부 정보 (픽셀, 패치, 또는 잠재 표현)를 가리고, 모델 (주로 오토인코더)이 보이는 부분의 컨텍스트를 사용하여 누락된 정보를 예측하도록 학습하는 기법입니다. 본 논문은 **재구성(Reconstruction)**과 **대조 학습(Contrastive Learning)** 두 가지 MIM 구현 방식을 공식화하고, 최근 주목할 만한 논문들을 분류하여 분석합니다. 또한, 계층적 클러스터링 알고리즘을 적용하여 수동 분류를 보완하고, 덴드로그램을 분석하여 관련 클러스터를 식별합니다. MIM 연구에 널리 사용되는 데이터셋을 살펴보고, 다양한 MIM 방법들의 성능 결과를 비교 분석하여, MIM의 연구 동향과 미래 방향을 제시합니다.



### Bi-directional Contextual Attention for 3D Dense Captioning (https://arxiv.org/abs/2408.06662)
Comments:
          Accepted to ECCV 2024 (Oral)

- **What's New**: This paper proposes BiCA, a novel transformer-based approach for 3D dense captioning that incorporates bi-directional contextual attention. This attention mechanism allows BiCA to generate object-aware contexts and context-aware objects, effectively addressing the limitations of prior methods by capturing both localized and global contextual information.

- **Technical Details**: BiCA utilizes a transformer encoder-decoder pipeline. It introduces two types of queries: instance queries for objects and context queries for non-object contexts. These queries enable the generation of object-aware contexts, which summarize the context relevant to each object, and context-aware objects, which aggregate objects relevant to the summarized contexts. By leveraging this bi-directional attention, BiCA effectively combines localized and global information, enhancing both localization accuracy and caption generation performance.

- **Performance Highlights**: Extensive experiments on widely-used 3D dense captioning datasets demonstrate that BiCA significantly outperforms previous methods. It achieves notable improvements in both localization and caption generation performance, highlighting the effectiveness of the proposed bi-directional contextual attention mechanism.



### Hybrid SD: Edge-Cloud Collaborative Inference for Stable Diffusion Models (https://arxiv.org/abs/2408.06646)
- **What's New**: 이 논문은 이미지 생성 모델인 Stable Diffusion 모델(SDM)을 위한 새로운 에지-클라우드 협업 추론 프레임워크인 Hybrid SD를 제시합니다. Hybrid SD는 클라우드 서버에 배포된 대규모 모델을 사용하여 확산 프로세스의 초기 단계를 처리하여 의미적 계획(semantic planning)을 향상시키고, 나중 단계에서 시각적 세부 정보를 개선하기 위해 에지 장치에 배포된 소형 효율적인 모델을 통합합니다.



### Unified-IoU: For High-Quality Object Detection (https://arxiv.org/abs/2408.06636)
- **What's New**: 이 논문에서는 기존 IoU 손실 함수의 한계를 극복하기 위해 새로운 IoU 손실 함수인 Unified-IoU(UIoU)를 제안합니다. UIoU는 다양한 품질의 예측 박스(prediction box)에 대한 가중치 할당(weight assignment)에 더 집중하여 모델의 성능을 향상시키는 것을 목표로 합니다. 이는 예측 박스의 품질에 따라 모델의 주의(attention)를 동적으로 이동시켜 고품질 박스에 더 집중하도록 유도하는 새로운 방식을 사용합니다. 특히 UIoU는 고정밀도 또는 집중적 데이터 세트(high-precision or intensive datasets)에서 모델의 검출 성능을 향상시키고 훈련 속도의 균형을 맞추는 데 효과적입니다.



### IDRetracor: Towards Visual Forensics Against Malicious Face Swapping (https://arxiv.org/abs/2408.06635)
- **What's New**: 이 논문은 딥페이크 기술을 이용한 페이스 스와핑으로 인한 개인 신원 보안의 심각한 위험에 대응하여 **페이스 리트레이싱** (Face Retracing)이라는 새로운 작업을 제안합니다. 이 작업은 **역 매핑** (Inverse Mapping)을 통해 주어진 가짜 얼굴에서 원래의 대상 얼굴을 추적하는 것을 목표로 합니다. 이를 위해 다양한 페이스 스와핑 방법으로 생성된 가짜 얼굴에서 임의의 원래 대상 신원을 추적할 수 있는 **IDRetracor**를 제안합니다.



### A lightweight YOLOv5-FFM model for occlusion pedestrian detection (https://arxiv.org/abs/2408.06633)
- **What's New**: 이 논문은 YOLOv5 모델을 기반으로 한 경량화된 보행자 검출 모델을 제안하여 폐색된 보행자 검출 정확도를 향상시키고 연산량을 줄였습니다. Ghost 모듈과 SE 블록을 도입하고, 폐색 문제를 해결하기 위해 지역 특징 융합 모듈 (FFM)을 설계했습니다.

- **Technical Details**: **주요 기술**: 

* **Ghost 모듈**: 기존의 Convolution 연산을 대체하여 파라미터 수와 FLOPs를 줄이는 경량화 기술
* **SE 블록**: 전역 정보를 고려하여 채널별 가중치를 부여하여 특징 정보를 강화하는 기술
* **FFM (Feature Fusion Module)**: 보행자의 머리와 다리 영역을 개별적으로 검출하고, 이를 융합하여 전체 보행자 예측 박스를 생성하는 모듈. 폐색 문제를 효과적으로 해결합니다.
* **WIoU Loss**:  예측 박스와 GT 박스 사이의 유사도를 더 정확하게 측정하는 손실 함수. 특히 여러 개체 부분이 포함된 경우 유용합니다.

- **Performance Highlights**: CityPersons 및 CUHK Occlusion 데이터셋에서 실험 결과, 제안된 모델은 기존 YOLOv5s 모델에 비해 평균 정밀도(AP)가 크게 향상되었으며, 파라미터 수는 27.9% 감소하고 FLOPs는 19.0% 감소했습니다.



### Fast Information Streaming Handler (FisH): A Unified Seismic Neural Network for Single Station Real-Time Earthquake Early Warning (https://arxiv.org/abs/2408.06629)
- **What's New**: 본 논문은 지진 조기 경보(EEW) 시스템에서 위상 선별(phase picking), 위치 추정(location estimation), 규모 추정(magnitude estimation)을 통합한 새로운 신경망 모델인 Fast Information Streaming Handler (FisH)를 제안합니다. FisH는 실시간 스트리밍 지진 데이터를 처리하고 이러한 작업을 단일 모델 내에 통합하여 전반적인 프로세스를 간소화하고 작업 간의 비선형 관계를 활용하여 성능을 향상시킵니다.

- **Technical Details**: FisH 모델은 RetNet을 백본으로 사용하여 훈련 중에 병렬 처리가 가능하고 추론 중에 순환 처리가 가능하게 하여 실시간 응용 프로그램에 적합합니다. Embedder 모듈은 입력 웨이브폼 데이터를 WaveEmbedding으로 변환합니다. Encoder 모듈은 RetNet 아키텍처를 사용하여 웨이브 임베딩 간의 상관 관계와 의존성을 추출하여 예측 임베딩을 생성합니다. Decoder 모듈은 예측 임베딩을 디코딩하여 위상 선별, 위치 추정, 규모 추정 결과를 생성합니다.

- **Performance Highlights**: STEAD 벤치마크 데이터셋을 사용한 광범위한 실험 결과 FisH 모델의 효율성을 입증했습니다. FisH는 위상 선별, 위치 추정, 규모 추정 작업에서 모두 뛰어난 성능을 보여줍니다. 특히, FisH는 0.99/0.96의 F1 점수를 달성했습니다. 또한 FisH는 위치 오차가 6.0km, 거리 오차가 2.6km, 역방향 각도 오차가 19°에 불과한 정확한 지진 위치 추정을 보여줍니다. 모델은 또한 0.14의 규모 오차만 있는 정확한 지진 규모 추정을 보여줍니다. 또한, FisH는 실시간으로 추정을 생성하여 P파가 도착한 후 불과 3초 만에 위치 오차 8.06km, 규모 오차 0.18의 위치 및 규모 추정을 제공합니다.



### DePatch: Towards Robust Adversarial Patch for Evading Person Detectors in the Real World (https://arxiv.org/abs/2408.06625)
- **What's New**: 본 논문에서는 기존 패치 기반 공격의 한계점인 ‘자체 결합 문제’를 해결하기 위해 새로운 패치 기반 공격 방식인 ‘DePatch’를 제안합니다. DePatch는 패치를 여러 블록으로 나누어 각 블록의 의존성을 줄이는 방식으로 설계되었습니다. 특히, 최적화 과정에서 일부 블록을 임의로 제거하여 각 블록의 독립성을 강화하는 전략을 사용합니다. 또한, 블록 경계를 이동시키는 ‘경계 이동 연산’과 블록 크기와 비율을 점진적으로 조정하는 ‘점진적 분리 전략’을 통해 공격 성능을 향상시킵니다.

- **Technical Details**: DePatch 공격은 기존 패치 기반 공격에서 나타나는 ‘자체 결합 문제’를 해결하기 위한 새로운 방법론입니다. 이 문제는 패치 내 모든 부분이 동시에 최적화되면서, 하나의 부분이 손상되면 전체 패치의 효과가 사라지는 현상을 말합니다. DePatch는 이 문제를 해결하기 위해 패치를 여러 블록으로 나누고, 각 블록을 독립적으로 최적화합니다. 구체적으로, 최적화 과정에서 일부 블록을 임의로 삭제하여 다른 블록의 영향을 최소화합니다. 이러한 블록 분리는 각 블록의 독립성을 강화하고, 실제 환경에서 발생할 수 있는 다양한 손상에 대한 강인성을 높여줍니다. 또한, DePatch는 블록 경계를 이동시키는 ‘경계 이동 연산’과 블록 크기와 비율을 점진적으로 조정하는 ‘점진적 분리 전략’을 통해 공격 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, DePatch는 기존 패치 기반 공격보다 뛰어난 성능을 보여주었습니다. 특히, 실제 환경에서 발생하는 다양한 손상에 대한 강인성이 크게 향상되었습니다. 예를 들어, 부분적인 가림, 자세 변화, 야외 환경, 조명 변화 등의 상황에서도 DePatch는 안정적인 성능을 유지했습니다. 이러한 결과는 DePatch가 실제 환경에서 더욱 효과적인 물리적 적대적 공격 방법임을 입증합니다.



### ActPrompt: In-Domain Feature Adaptation via Action Cues for Video Temporal Grounding (https://arxiv.org/abs/2408.06622)
Comments:
          9 pages, 5 figures

- **What's New**: This paper addresses the challenge of adapting pre-trained vision-language models (VLM) for video temporal grounding tasks, specifically for moment retrieval and highlight detection.  The authors propose two main contributions:

- **Technical Details**: 1. **In-Domain Fine-Tuning:**  They introduce a novel in-domain fine-tuning paradigm for VLMs using pretext tasks to adapt features to the specific video data domain. This approach reduces the computational overhead of traditional end-to-end fine-tuning. The pretext tasks include moment-query pairwise ranking and moment-query contrastive learning. 
2. **Action-Cue-Injected Temporal Prompt Learning (ActPrompt):** This framework aims to enhance the VLM's ability to capture action-sensitive patterns by injecting action cues into the image encoder. ActPrompt consists of two modules: 
    * **Action Cue Injection (ACI):** Uses action cues derived from video and text encoders to guide the image encoder towards action-sensitive regions within video frames. 
    * **Context-aware Temporal Prompt Learning (CTPL):** Extracts motion features from a sequence of selected visual regions, capturing temporal context. An adaptor is trained to generate temporal prompts from these motion features, incorporating temporal information into the static frame.

- **Performance Highlights**: ActPrompt outperforms various state-of-the-art (SOTA) methods on both moment retrieval and highlight detection tasks. The experiments demonstrate the effectiveness of the proposed in-domain fine-tuning and ActPrompt framework for improving video temporal grounding performance.



### ViMo: Generating Motions from Casual Videos (https://arxiv.org/abs/2408.06614)
- **What's New**: This paper introduces ViMo, a novel Video-to-Motion-Generation (ViMo) framework that leverages the vast amount of video data to generate diverse and realistic 3D human motions. This is a significant advancement from previous methods that heavily rely on limited and costly motion capture datasets.

- **Technical Details**: ViMo utilizes a diffusion-based model that takes 2D poses from multiple camera views as input to generate 3D motions. It bypasses the need for explicit camera position estimation, allowing it to handle videos with complex camera movements and occlusions. This approach overcomes the limitations of existing methods that struggle with such challenges.

- **Performance Highlights**: ViMo demonstrates remarkable results in generating natural motions from casual videos, even those with rapid movements, varying perspectives, and frequent occlusions. It enables three downstream applications: (1) generating a large-scale 3D dancing motion dataset from Chinese classic dancing videos, (2) performing few-shot style transfer in dancing generation, and (3) enabling video-guided motion completion and editing tasks.



### CROME: Cross-Modal Adapters for Efficient Multimodal LLM (https://arxiv.org/abs/2408.06610)
- **What's New**: CROME은 비용 효율적인 멀티모달 대규모 언어 모델(MLLM) 학습을 위한 새로운 프레임워크입니다. 기존의 접근 방식과 달리 CROME은 언어 모델 재학습 비용을 줄이고 다양한 작업에 대한 적응력을 높입니다. 핵심은 '게이트 방식의 크로스 모달 어댑터(gated cross-modal adapter)'인데, 이는 이미지와 텍스트 표현을 효율적으로 결합하여 동결된 LLM에 입력합니다. 이 어댑터는 몇 가지 매개변수만으로 학습되므로 효율적인 크로스 모달 이해를 가능하게 합니다. 특히, CROME은 표준 시각 질문 답변 및 지시 추론 벤치마크에서 우수한 제로 샷(zero-shot) 성능을 보여줍니다. 또한, 매개변수 효율성이 뛰어난 미세 조정(fine-tuning)을 통해 특정 작업에 특화된 최첨단 방법과 경쟁합니다. CROME은 확장 가능하고 적응력이 뛰어나며 매개변수 효율적인 멀티모달 모델을 구축하기 위한 사전 LLM 정렬(pre-LM alignment)의 잠재력을 보여줍니다.



### MV-DETR: Multi-modality indoor object detection by Multi-View DEtecton TRansformers (https://arxiv.org/abs/2408.06604)
- **What's New**: 이 논문은 효율적인 트랜스포머 기반 탐지 방법인 MV-DETR 파이프라인을 소개합니다. MV-DETR은 RGBD 데이터를 입력으로 받아, 기하학적 정보와 시각적 텍스처 정보를 분리하여 처리하는 방식으로 기존 방법의 한계를 극복합니다. 특히, 대규모 이미지 사전 학습 데이터셋에서 얻어진 강력한 시각적 텍스처 특징 추출기가 사용됩니다. 또한, 시각적 텍스처 인코더, 기하학적 인코더, VG 커넥터로 구성된 경량 VG 모듈을 설계하여 시각적 및 기하학적 특징을 효과적으로 결합합니다. ScanNetV2 데이터셋에서 실험 결과, MV-DETR은 기존 최첨단 방법인 V-DETR보다 뛰어난 성능을 보이며, 78% AP라는 새로운 최첨단 결과를 달성합니다. 이러한 결과는 MV-DETR이 RGBD 기반 객체 탐지 분야에서 혁신적인 성과를 제공함을 보여줍니다.



### GeoFormer: Learning Point Cloud Completion with Tri-Plane Integrated Transformer (https://arxiv.org/abs/2408.06596)
Comments:
          accepted by the 32nd ACM International Conference on Multimedia (MM'24)

- **What's New**: 이 논문에서는 부분적인 포인트 클라우드에서 정확한 전역 기하학적 구조를 복구하고 미세한 지역적 세부 사항을 보존하는 포인트 클라우드 완성(point cloud completion)을 위한 GeoFormer를 소개합니다. GeoFormer는 멀티 뷰 일관성이 있는 정규 좌표 맵(Canonical Coordinate Maps, CCM)에서 이미지 특징을 통합하여 포인트 특징과 정렬함으로써 전역 기하학적 특징을 향상시키는 CCM 특징 강화 포인트 생성기(CCM Feature Enhanced Point Generator)를 설계합니다. 또한 부분 입력에서 추출된 다중 스케일 특징과 이전에 추정된 포인트에서 파생된 특징 간의 교차 주의(cross attention)를 통해 점진적으로 지역적 세부 사항을 향상시키는 다중 스케일 기하 인식 업샘플러(Multi-scale Geometry-aware Upsampler) 모듈을 사용합니다.



### ActiveNeRF: Learning Accurate 3D Geometry by Active Pattern Projection (https://arxiv.org/abs/2408.06592)
Comments:
          18 pages, 10 figures

- **What's New**: ActiveNeRF는 기존 NeRF의 3D 기하 복원 정확도를 향상시키는 새로운 방법을 제시합니다. 이 방법은 카메라와 상대적인 위치를 유지하는 프로젝터를 사용하여 고주파 공간 패턴을 씬에 투사하여 이미지에서 얻을 수 있는 기하 정보의 양을 늘립니다. 이를 통해 기존의 정적 조명 환경에서 얻는 저주파 공간 정보의 한계를 극복하고, 더 정확한 기하 정보를 복원할 수 있습니다. ActiveNeRF는 이러한 능동적인 패턴 프로젝션을 활용하여 씬의 기하 구조와 능동적 패턴을 동시에 학습하는 학습 가능한 능동적 패턴 렌더링 파이프라인을 설계했습니다.



### HDRGS: High Dynamic Range Gaussian Splatting (https://arxiv.org/abs/2408.06543)
- **What's New**: This paper proposes a novel method called **High Dynamic Range Gaussian Splatting (HDR-GS)** for reconstructing 3D HDR radiance fields from multi-exposure LDR images. HDR-GS leverages the recent real-time 3D reconstruction technique called **Gaussian Splatting** and incorporates a differentiable, **asymmetric grid** for tone mapping, enabling efficient and accurate HDR scene recovery. Additionally, it introduces a **coarse-to-fine strategy** that accelerates model convergence and enhances robustness against sparse viewpoints and extreme exposure conditions.



### Benchmarking tree species classification from proximally-sensed laser scanning data: introducing the FOR-species20K datas (https://arxiv.org/abs/2408.06507)
- **What's New**: FOR-species20K, 레이저 스캐닝 데이터를 사용한 나무 종 분류를 위한 딥 러닝 모델 개발 및 벤치마킹을 위한 핵심 자원으로 20,000개 이상의 나무 포인트 클라우드(point cloud)를 포함하고 있습니다. 이 데이터 세트는 다양한 유럽 숲에서 획득한 지상 (TLS), 모바일 (MLS) 및 드론 레이저 스캐닝 (ULS)을 사용하여 33개 종의 나무 포인트 클라우드를 포함합니다. (TLS: Terrestrial Laser Scanning, MLS: Mobile Laser Scanning, ULS: Unmanned Laser Scanning)



### Prompt Recovery for Image Generation Models: A Comparative Study of Discrete Optimizers (https://arxiv.org/abs/2408.06502)
Comments:
          9 Pages, 4 Figures

- **What's New**: This paper presents a comprehensive comparison of recent discrete optimization techniques for image generation prompt inversion, which involves recovering the natural language prompt used to create an image. It analyzes the performance of various methods like Greedy Coordinate Gradients (GCG), PEZ, Random Search, AutoDAN, and BLIP2's image captioner, evaluating them based on the quality of inverted prompts and generated images.

- **Technical Details**: The paper employs a CLIP-guided diffusion model as a benchmark for evaluating the performance of the prompt inversion methods. It examines how each optimizer performs in recovering the natural language prompt using multiple metrics, including CLIP similarity, image quality, and coherence of the generated image using the inverted prompt. Notably, the study focuses on the comparison of discrete optimization techniques in the context of image generation, a task with significant practical implications for controlling the output of generative models and understanding their internal workings.

- **Performance Highlights**: The research found that focusing solely on CLIP similarity between the inverted prompts and the ground truth image is not a reliable indicator of the similarity between the generated images and the original image. While the discrete optimizers effectively minimize their objectives, simply using a well-trained image captioner often results in generated images that more closely resemble those produced by the original prompts. The study highlights the need for a more holistic evaluation approach that considers both prompt quality and generated image quality for prompt inversion tasks.



### Generalization Enhancement Strategies to Enable Cross-year Cropland Mapping with Convolutional Neural Networks Trained Using Historical Samples (https://arxiv.org/abs/2408.06467)
- **What's New**: 본 논문은 대규모 지역의 농경지 매핑 정확도를 향상시키기 위한 새로운 딥 러닝 (DL) 모델을 제시합니다. 특히, 매년 달라지는 농업 관행 및 환경 조건으로 인한 도메인 변화를 고려하여 연간 농경지 지도를 생성하기 위한 일반화 능력을 향상시키는 데 중점을 둡니다. 이 모델은 Tversky-focal loss (TFL)와 같은 영역 기반 손실 함수, 광도 증강 (photometric augmentation) 및 MC-dropout과 같은 다양한 증강 기법, 입력 정규화 전략을 결합하여 일반화 능력을 향상시킵니다.



### Advanced Vision Transformers and Open-Set Learning for Robust Mosquito Classification: A Novel Approach to Entomological Studies (https://arxiv.org/abs/2408.06457)
Comments:
          23 pages, 15 figures

- **What's New**: 이 연구는 최첨단 비전 트랜스포머(Vision Transformer)와 오픈셋 학습(Open-Set Learning) 기술을 활용하여 모기 분류를 위한 혁신적인 방법을 제시합니다. 본 연구에서 제시된 프레임워크는 트랜스포머 기반 딥러닝 모델을 포괄적인 데이터 증강 및 전처리 방법과 통합하여 10종의 모기 종을 강력하고 정확하게 식별할 수 있습니다.



### S-SAM: SVD-based Fine-Tuning of Segment Anything Model for Medical Image Segmentation (https://arxiv.org/abs/2408.06447)
Comments:
          Accepted in MICCAI 2024

- **What's New**: This paper introduces S-SAM, a novel adaptation method for the Segment Anything Model (SAM) that enables efficient and accurate medical image segmentation using simple text prompts (e.g., "tumor", "organ"). This approach significantly reduces the training burden by only tuning 0.4% of SAM's parameters, making it more practical for medical applications compared to previous methods that require expert-level prompts for every image.



### HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization (https://arxiv.org/abs/2408.06437)
Comments:
          Accepted to ECCV 2024

- **What's New**: This paper proposes a new framework named History-Augmented Anchor Transformer (HAT) for Online Temporal Action Localization (OnTAL). HAT effectively integrates historical context into the OnTAL process, enhancing the performance of anchor features for action classification and localization.



### Wavelet based inpainting detection (https://arxiv.org/abs/2408.06429)
- **What's New**: 이 논문은 DT-CWT(Dual-Tree Complex Wavelet Transform)와 계층적 특징 분할 및 노이즈 불일치 분석을 결합하여 이미지 인페인팅 위조를 감지하는 새로운 방법을 제시합니다. DT-CWT는 인페인팅 과정에서 발생하는 사소한 조작에 강력한 이동 불변성(shift-invariance)과 인페인팅에 의해 특정 주파수 대역 및 방향에서 도입된 미묘한 인공물을 포착하는 데 도움이 되는 방향 선택성(directional selectivity)을 제공합니다.



### Using deep learning to enhance electronic service quality: Application to real estate websites (https://arxiv.org/abs/2408.06364)
- **What's New**: 이 연구는 전자 서비스의 가시성과 효율성을 향상시키기 위해 시각적 특징과 기술적 특징을 통합하는 새로운 접근 방식을 제시합니다. 특히, 부동산 웹사이트의 예를 들어, 사용자 선호도에 맞는 부동산을 찾는 것은 시각적 필터(예: 부동산 손상 수준)가 부족하여 어려움을 겪고 있습니다. 이 연구에서는 Mask-RCNN (Mask Region-based Convolutional Neural Network)라는 딥 러닝 네트워크를 사용하여 부동산 이미지의 손상 수준을 추정하는 새로운 시각적 기술적 특징인 손상 수준을 소개합니다. 또한, 전자 부동산 서비스에서 손상 수준을 가시적 특징으로 통합하여 고객 경험을 향상시키는 모델을 개발했습니다.



### Algorithm Research of ELMo Word Embedding and Deep Learning Multimodal Transformer in Image Description (https://arxiv.org/abs/2408.06357)
- **What's New**: 본 논문에서는 기존의 임베딩 기반 제로 샘플 학습 방법론의 단점을 개선하여, 미지의 클래스를 벡터 공간에 포함시키는 새로운 방법을 제안합니다. 이는 카테고리 의미 유사성 측정을 활용하여, 현재 알려진 클래스와 동일한 의미를 가진 미지의 클래스를 벡터 공간에 통합하는 방식을 통해 가능합니다. 또한, 기존의 제로 샘플 학습 알고리즘들이 의료 이미지의 깊이 특징을 직접 입력으로 사용하는 반면, 본 연구에서는 ELMo-MCT를 활용하여 셀프 어텐션 메커니즘을 통해 원본 이미지와 관련된 다양한 시각적 특징을 추출하는 방식을 제안합니다. 이를 통해, 기존 방법보다 더욱 정확한 제로 샘플 학습 성능을 달성할 수 있습니다.



### Enhancing Ecological Monitoring with Multi-Objective Optimization: A Novel Dataset and Methodology for Segmentation Algorithms (https://arxiv.org/abs/2408.06356)
- **What's New**: This paper introduces a novel **semantic segmentation** dataset called **ALGSeg** which consists of **6,096 high-resolution aerial images** capturing indigenous and invasive grass species in **Bega Valley, New South Wales, Australia**. The dataset focuses on distinguishing grass from non-grass areas, a critical first step towards identifying invasive species like **African lovegrass (ALG)**.



### Automated Romberg Test: Leveraging a CNN and Centre of Mass Analysis for Sensory Ataxia Diagnosis (https://arxiv.org/abs/2408.06354)
- **What's New**: 이 논문은 감각성 운동실조(sensory ataxia)를 진단하기 위한 새로운 자동 롬베르크 테스트(Romberg Test) 방법을 제안합니다. 이 방법은 컨볼루션 신경망(Convolutional Neural Network)을 사용하여 관절 위치를 예측하고 이를 통해 중심 질량(center of mass) 및 다양한 관절 각도와 같은 생체 역학적 마커(bio-mechanical markers)를 계산합니다. 이 정보는 칼만 필터(Kalman Filter)와 같은 데이터 필터링 기술 및 중심 질량 분석과 결합되어 측면 및 전후 축(lateral and anterior-posterior axes)에서 상대적인 무게 분포(relative weight distribution)에 대한 정확한 추론을 가능하게 하며, 이 질환에 대한 객관적이고 수학적으로 근거한 진단을 제공합니다. 이 방법의 성능을 평가하기 위해 의료 환경에서 촬영된 사전 주석이 달린 진단 비디오와 이중 무게 측정(dual weight scales)을 사용하여 테스트를 수행했습니다. 이 두 가지 방법 모두 지면 표면에서의 실제 무게 분포를 정확하게 측정했으며 제안된 방법의 실제 정확도를 제공했습니다. 계산된 상대적인 무게 분포 차이의 평균 절대 오차(mean absolute error)는 0.2912%로 나타났으며, 진단 정확도는 83.33%였습니다.



### Automated Schizophrenia Detection from Handwriting Samples via Transfer Learning Convolutional Neural Networks (https://arxiv.org/abs/2408.06347)
Comments:
          5 pages, 8 figures

- **What's New**: 이 연구는 조현병 환자의 필적을 분석하여 조현병을 진단하고 모니터링하는 비침습적이고 객관적인 방법을 개발했습니다. 연구자들은 조현병 환자와 정상인의 필적 데이터를 사용하여 컨볼루션 신경망(CNN) 모델을 훈련했으며, InceptionV3 아키텍처 기반 모델이 92%의 정확도를 달성하여 두 유형의 필적을 구분하는 데 성공했습니다. 이 모델은 의료 전문가가 환자의 필적을 분석하고 조현병 여부를 판단하는 데 도움이 되는 웹사이트에 구축되었습니다. 



### Fingerspelling within Sign Language Translation (https://arxiv.org/abs/2408.07065)
- **What's New**: This paper explores the impact of character-level tokenization and fingerspelling recognition data integration on American Sign Language (ASL) to English translation models, specifically focusing on the ability to understand and translate fingerspelling within sentences.

- **Technical Details**: The study uses FLEURS-ASL dataset, manually annotated for fingerspelling instances, to evaluate two approaches: 1) using ByT5, a model family with character-level tokenization, and 2) incorporating fingerspelling recognition data (FSboard) into training. The paper measures performance using BLEURT scores and character-error rate (CER) for fingerspelled phrases.

- **Performance Highlights**: Using ByT5 with character-level tokenization significantly improves overall translation quality (BLEURT score), particularly in sentences containing fingerspelling. However, integrating fingerspelling recognition data into training yielded mixed or negative results. The study suggests character-level tokenization as a promising approach for improving fingerspelling understanding in sign language translation models.



### PSM: Learning Probabilistic Embeddings for Multi-scale Zero-Shot Soundscape Mapping (https://arxiv.org/abs/2408.07050)
Comments:
          Accepted at ACM MM 2024

- **What's New**: 이 논문은 전 세계의 사운드스케이프(soundscape)를 매핑하는 프레임워크를 제시합니다. 이 프레임워크는 다양한 공간적 규모에 걸쳐 소리 분포를 포함하는 사운드스케이프를 고려하여 다중 스케일 위성 이미지로 위치를 나타내고 이미지, 오디오, 텍스트 간의 공동 표현을 학습합니다. 특히, 위치의 사운드스케이프에 존재하는 고유한 불확실성을 포착하기 위해 확률적 표현 공간을 설계하고, 지리 위치, 시간, 데이터 소스 등의 광범위한 메타데이터를 통합하여 공간 및 시간적으로 동적 사운드스케이프 표현을 학습합니다. 또한, 이 연구에서는 30만 개 이상의 지리 태그가 지정된 오디오 샘플과 저해상도 및 고해상도 위성 이미지를 결합한 대규모 데이터셋인 GeoSound를 소개합니다. GeoSound와 SoundingEarth 데이터셋에서 제안된 방법이 기존 최첨단 기술보다 뛰어난 성능을 보임을 보여줍니다. 데이터셋과 코드는 [URL]에서 이용 가능합니다.



### Event-Stream Super Resolution using Sigma-Delta Neural Network (https://arxiv.org/abs/2408.06968)
Comments:
          ECCV: The 18th European Conference on Computer Vision ECCV 2024 NeVi Workshop

- **What's New**: 본 논문은 이벤트 카메라에서 포착한 휘도 변화를 기반으로 시간 이벤트 픽셀의 공간-시간 해상도를 향상시키는 새로운 방법을 소개합니다. 이벤트 카메라는 데이터를 수집하는 방식이 드물고 비동기적이기 때문에 낮은 해상도라는 고유한 과제를 제시합니다. 현재 이벤트 슈퍼 해상도 알고리즘은 이벤트 카메라가 생성하는 고유한 데이터 구조에 완벽하게 최적화되어 있지 않아 계산 복잡성을 개선하면서 시각 장면의 전체 역동성과 디테일을 포착하는 데 비효율성을 초래합니다. 이러한 간극을 해소하기 위해 이 연구에서는 이벤트 스트림의 공간 및 시간 분포를 동시에 학습하도록 설계된 공간-시간 제약 학습 메커니즘을 활용하여 이진 스파이크를 시그마 델타 신경망(SDNN)과 통합하는 방법을 제안합니다. 제안된 네트워크는 N-MNIST, CIFAR10-DVS, ASL-DVS 및 Event-NFS를 포함한 널리 알려진 벤치마크 데이터셋을 사용하여 평가됩니다. 포괄적인 평가 프레임워크가 사용되어 루트 평균 제곱 오차(RMSE)를 통한 정확성과 모델의 계산 효율성을 모두 평가합니다. 결과는 기존 최첨단 방법에 비해 상당한 개선을 보여줍니다. 특히 제안된 방법은 계산 효율성 측면에서 최첨단 성능을 능가하여 전통적인 인공 신경망에 비해 이벤트 스파스성이 17.04배, 시냅스 작동 효율성이 32.28배 향상되었으며, 스파이킹 신경망에 비해 2배 더 나은 성능을 보입니다.



### Automatic Feature Recognition and Dimensional Attributes Extraction From CAD Models for Hybrid Additive-Subtractive Manufacturing (https://arxiv.org/abs/2408.06891)
Comments:
          10 pages, 12 figures. This paper has been accepted for presentation at the ASME IDETC-CIE 2024 conference

- **What's New**: 본 논문은 CAD 모델에서 가산 및 감산 제조 공정에 모두 관련된 기능을 인식하는 새로운 접근 방식을 제시합니다. 기존의 AFR (Automatic Feature Recognition) 방법은 구멍, 필렛, 챔퍼, 포켓, 슬롯과 같은 감산 (기계 가공) 기능 식별에 중점을 두었지만, 가산 제조에 관련된 기능을 인식하는 데는 실패했습니다. 또한, 기존 방법은 효과적인 제조 공정 계획에 필수적인 기하학적 치수와 방향을 정확하게 추출하는 데 부족했습니다. 이 논문에서는 Python Open Cascade를 통해 가산 및 감산 가공에 관련된 기능을 포함하는 합성 CAD 데이터 세트를 생성하는 새로운 방법을 제시합니다. 계층적 그래프 합성곱 신경망 (HGCNN) 모델은 합성 CAD 데이터 세트 내에서 복합 가산-감산 기능을 정확하게 식별하기 위해 구현되었습니다. 제안된 방법론의 주요 참신성과 기여는 다양한 제조 기능을 인식하고, 치수, 방향 및 스톡 크기를 정확하게 추출할 수 있다는 것입니다.



### Membership Inference Attack Against Masked Image Modeling (https://arxiv.org/abs/2408.06825)
- **What's New**: 이 논문은 마스크 이미지 모델링(MIM)을 통해 사전 훈련된 이미지 인코더에 대한 첫 번째 멤버십 추론 공격을 제안합니다. 이 공격은 이미지가 MIM 사전 훈련 데이터 세트의 일부인지 여부를 판별하는 것을 목표로 합니다.

- **Technical Details**: 공격은 MIM의 사전 훈련 패러다임(즉, 이미지 마스킹 및 재구성)을 모방하고 재구성 오류를 얻어 작동합니다. 인코더가 훈련 세트에서 입력 이미지를 더 낮은 오류로 재구성할 수 있으므로 재구성 오류는 공격 목표를 달성하기 위한 멤버십 신호 역할을 합니다.

- **Performance Highlights**: 세 가지 모델 아키텍처와 세 가지 벤치 마크 데이터 세트에 대한 광범위한 평가 결과, 제안된 공격이 기준선 방법보다 뛰어난 성능을 보이는 것으로 나타났습니다. 또한, 공격의 성능에 영향을 미칠 수 있는 여러 요인을 분석하기 위해 심층적인 에블레이션 연구를 수행했습니다.



### Enhancing Diabetic Retinopathy Diagnosis: A Lightweight CNN Architecture for Efficient Exudate Detection in Retinal Fundus Images (https://arxiv.org/abs/2408.06784)
- **What's New**: 본 논문에서는 자동 망막 삼출물(exudate) 검출을 위해 경량화된 합성곱 신경망 아키텍처를 소개합니다. 이 아키텍처는 효율적이고 정확하게 삼출물을 식별하도록 설계되었습니다. 제한된 훈련 데이터 문제를 해결하기 위해 모델의 일반화(generalizability)를 향상시키는 도메인 특정 데이터 증강(domain-specific data augmentations)을 통합했습니다. 또한, 맞춤형 아키텍처 내에 다양한 규제 기법(regularization techniques)을 적용하여 진단 정확도를 높이고 계산 효율성을 최적화했습니다. 이 간소화된 모델은 기존의 ResNet-18 모델의 1169만 개 매개변수에 비해 약 60% 감소한 473만 개 매개변수만 포함하고 있습니다. 복잡성이 줄어든 것에도 불구하고 모델은 90%의 인상적인 F1 점수를 달성하여 망막 영상을 통한 당뇨병성 망막증의 조기 진단에 대한 효과를 입증합니다.



### Enhancing Visual Dialog State Tracking through Iterative Object-Entity Alignment in Multi-Round Conversations (https://arxiv.org/abs/2408.06725)
Comments:
          This article has been accepted in CAAI Transactions on Intelligence Technology! Article ID: CIT2_12370, Article DOI: https://doi.org/10.1049/cit2.12370

- **What's New**: This paper introduces a new model, Multi-round Dialogue State Tracking (MDST), for Visual Dialog (VD). MDST addresses the limitation of previous VD models that treat the entire dialog history as a simple text input, by leveraging the dialogue state learned from the dialog history to answer questions. MDST captures each round of dialog history, constructing internal dialogue state representations defined as 2-tuples of vision-language representations, which effectively ground the current question, enabling the generation of more accurate answers.



### How to Best Combine Demosaicing and Denoising? (https://arxiv.org/abs/2408.06684)
Comments:
          This paper was accepted by Inverse Problems and Imaging on October, 2023

- **What's New**: 이 논문은 노이즈가 있는 RAW 이미지에서 디모자이킹(demosaicing)과 디노이징(denoising)을 함께 수행하는 최적화된 파이프라인을 제안합니다. 이전의 접근 방식은 디모자이킹과 디노이징을 독립적으로 처리했지만 이 논문은 둘의 상호 작용을 고려하여 더 효과적인 방법을 제시합니다. 특히, 노이즈 수준에 따라 "먼저 디모자이킹, 그 다음 디노이징" 또는 "부분적인 CFA 디노이징 후 디모자이킹, 그리고 마지막으로 RGB 이미지에 대한 추가 디노이징" 과 같은 두 가지 전략을 제안합니다.



### Coherence Awareness in Diffractive Neural Networks (https://arxiv.org/abs/2408.06681)
- **What's New**: 이 논문은 빛의 공간적 응집성이 회절 신경망의 성능에 미치는 영향을 분석하고, 다양한 응집성 수준을 가진 조명에 대한 회절 신경망을 학습할 수 있는 새로운 프레임워크를 제시합니다. 특히, 회절 신경망은 영상 시스템과 달리 공간적 응집성에 매우 민감하며, 빛의 응집 길이가 시스템의 최소 해상도와 비슷한 경우, 완전한 응집성 또는 비응집성 근사 모두 적용할 수 없습니다.



### Specialized Change Detection using Segment Anything (https://arxiv.org/abs/2408.06644)
- **What's New**: 이 논문은 특정 객체의 사라짐(disappearance)을 감지하는 데 초점을 맞춘 새로운 변화 감지(change detection) 방법을 제안합니다. 기존의 변화 감지 방법은 모든 변화를 감지하는 반면, 이 방법은 건물과 같은 특정 객체의 사라짐을 우선적으로 감지하여 자연 재해나 다른 원인으로 인한 손실을 파악하는 데 사용됩니다.

- **Technical Details**: 이 방법은 Segment Anything Model(SAM)이라는 강력한 비전 기반 모델을 활용합니다. SAM은 이미지 내의 객체를 분할하는 데 탁월하며, 이 연구에서는 사전 변화 이미지에서 제공된 이진 마스크(binary mask)를 사용하여 후변화 이미지에서 해당 객체가 사라졌는지 감지합니다. SAM의 강력한 분할 능력을 활용하여 사전 변화 마스크로부터 프롬프트(prompt)를 생성하고 이 프롬프트를 사용하여 후변화 이미지를 분할하고 누락된 객체를 식별합니다. 이 방법은 비지도 학습 방식으로 작동하며, 건물 사라짐 감지에 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: 제안된 방법은 건물 사라짐 감지에 효과적인 것으로 입증되었으며, 특정 유형의 변화 감지에 적합합니다. 또한, 이 방법은 기존의 변화 감지 방법과 달리 사전 변화 이미지를 필요로 하지 않고, 사전 변화 마스크만을 사용하여 개인 정보 보호에도 유용합니다.



### Attention Based Feature Fusion Network for Monkeypox Skin Lesion Detection (https://arxiv.org/abs/2408.06640)
Comments:
          6 pages with 6 figures

- **What's New**: 본 연구는 효율적인 네트워크(EfficientNet)와 ResNet 아키텍처를 결합하여 가벼운 딥 러닝 모델을 개발하여 사람의 원숭이두창 질병을 분류했습니다. Squeeze-and-excitation(SE) 네트워크 모듈을 통합하여 기능 맵의 중요한 부분에 집중하여 원숭이두창 이미지 분류를 향상시켰습니다. 즉, 특징 맵 내에서 중요한 영역을 강조하여 채널 및 공간 주의를 제공합니다. 



### COD: Learning Conditional Invariant Representation for Domain Adaptation Regression (https://arxiv.org/abs/2408.06638)
Comments:
          Accepted to ECCV 2024 (oral)

- **What's New**: This paper proposes a novel Domain Adaptation Regression (DAR) method for continuous output scenarios. Unlike existing methods that focus on marginal distribution alignment, this work addresses the critical issue of conditional distribution alignment for continuous labels.

- **Technical Details**: The paper establishes the sufficiency theory for DAR, demonstrating that the generalization error can be upper-bounded by the conditional discrepancy. To measure this conditional discrepancy, a new metric called Conditional Operator Discrepancy (COD) is introduced. COD leverages the kernel embedding theory to characterize the conditional distributions as finite statistical moments in Hilbert space.

- **Performance Highlights**: The proposed COD-based conditional invariant representation learning model effectively minimizes the conditional discrepancy, improving the discriminability of the adaptation model. Extensive experiments on standard DAR datasets validate the theoretical results and demonstrate superior performance compared to state-of-the-art DAR methods.



### Deep Inertia $L_p$ Half-Quadratic Splitting Unrolling Network for Sparse View CT Reconstruction (https://arxiv.org/abs/2408.06600)
Comments:
          This paper was accepted by IEEE Signal Processing Letters on July 28, 2024

- **What's New**: 본 논문에서는 희소 뷰 컴퓨터 단층 촬영(CT) 재구성 문제를 해결하기 위해 새로운 알고리즘인 관성 Lp-노름 반-2차 분할 알고리즘(IHQSp)을 제안합니다. 이 알고리즘은 Lp-노름(0<p<1) 정규화를 사용하여 희소성을 유도하고 관성 단계를 도입하여 수렴 속도를 향상시킵니다. 또한, 딥러닝을 활용하여 공액 기울기(CG) 방법을 초기화하여 이론적 보장이 있는 딥 언롤링 네트워크를 구축합니다.

- **Technical Details**: IHQSp 알고리즘은 Lp-노름 정규화를 사용하여 희소성을 유도하고 관성 단계를 도입하여 수렴 속도를 향상시킵니다. 또한, 딥러닝을 활용하여 공액 기울기(CG) 방법을 초기화하여 이론적 보장이 있는 딥 언롤링 네트워크를 구축합니다. 이 네트워크는 CG에 초기값을 제공할 뿐 알고리즘의 수렴에 영향을 미치지 않습니다. 즉, IHQSp 알고리즘의 수렴이 보장되는 딥러닝 기반 알고리즘인 IHQSp-Net을 개발합니다.

- **Performance Highlights**: 본 논문에서 제안된 알고리즘은 기존 방법보다 뛰어난 성능을 보여주며, 특히 스캔 뷰 수가 적고 복잡한 노이즈 조건에서 탁월한 성능을 보입니다. 또한, 이론적 보장을 갖춘 딥러닝 기반 알고리즘인 IHQSp-Net을 통해 희소 뷰 CT 재구성 문제를 더 효과적으로 해결할 수 있습니다.



### What Color Scheme is More Effective in Assisting Readers to Locate Information in a Color-Coded Article? (https://arxiv.org/abs/2408.06494)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 문서 컬러 코딩(color-coding)을 자동화하는 데 사용될 수 있는 가능성을 탐구합니다. 이 연구는 LLM이 생성한 다양한 컬러 스킴(color scheme)이 정보 탐색에 미치는 영향을 조사했습니다. 특히, 아날로그가 아닌(non-analogous) 컬러 스킴과 노란색을 포함하는 컬러 스킴이 정보 탐색 성능을 향상시키는 것으로 나타났습니다.



### InfLocNet: Enhanced Lung Infection Localization and Disease Detection from Chest X-Ray Images Using Lightweight Deep Learning (https://arxiv.org/abs/2408.06459)
- **What's New**: 이 논문은 가슴 X선 영상을 이용하여 폐 질환을 효과적으로 감지하고 감염 영역을 국한하는 새로운 분할-분류 네트워크를 제시합니다. 이 네트워크는 UNet++ 구조 내에 개선된 스킵 연결을 통합하여 의미적 간격을 줄이고 분할 작업의 정밀도를 향상시킵니다. 또한, 이 모델은 인코더 블록의 끝에 분류 모듈을 통합하여 동시 분류 및 분할을 가능하게 합니다.



### From Diagnostic CT to DTI Tractography labels: Using Deep Learning for Corticospinal Tract Injury Assessment and Outcome Prediction in Intracerebral Haemorrhag (https://arxiv.org/abs/2408.06403)
Comments:
          Accepted to Miccai Switch Workshop

- **What's New**: 본 연구는 뇌출혈 환자의 CT 스캔만으로 corticospinal tract (CST)를 분할하는 딥러닝 모델을 제시합니다. 이 모델은 diffusion tensor tractography (DTI)를 사용하지 않고도 CST의 무결성을 평가할 수 있으며, 뇌출혈 후 운동 기능 회복 예측에 유용합니다. 또한, 수술적 혈종 제거가 필요한 환자를 선별하는 데 도움을 줄 수 있습니다.



### Synthetic Photography Detection: A Visual Guidance for Identifying Synthetic Images Created by AI (https://arxiv.org/abs/2408.06398)
Comments:
          27 pages, 25 figures

- **What's New**: 이 논문은 최근 인공지능(AI) 기술이 만들어내는 합성 이미지의 위험성을 경고합니다. 특히 실제 사진과 흡사한 합성 이미지는 사기꾼에서 국가기관까지 다양한 위협 요소들에게 사용되어 사람들을 속이고, 돈을 빼앗고, 오도할 수 있다는 점을 강조합니다. 이를 막기 위해서는 사진이 진짜인지 합성 이미지인지 판별하는 기술이 필수적입니다. 본 연구는 최신 합성 이미지 생성 모델의 한계를 분석하고, 생성된 이미지에 나타나는 특징적인 인공 흔적(artifact)을 통해 이미지의 진위 여부를 판별하는 방법을 제시합니다.



### Dilated Convolution with Learnable Spacings (https://arxiv.org/abs/2408.06383)
Comments:
          PhD Thesis

- **What's New**: 본 논문에서는 DCLS(Dilated Convolution with Learnable Spacings)라는 새로운 방법을 소개하고 평가합니다. DCLS는 기존의 컨볼루션 방법보다 더 나은 성능을 보여주는 새로운 방법이며, 컴퓨터 비전, 오디오, 음성 처리 분야의 다양한 지도 학습 실험에서 입증되었습니다.



### FedRobo: Federated Learning Driven Autonomous Inter Robots Communication For Optimal Chemical Sprays (https://arxiv.org/abs/2408.06382)
Comments:
          This research article is going to be submitted to a best-fit conference. We are looking for a conference

- **What's New**: 이 논문은 농업 자동화에 Federated Learning(연합 학습)을 적용하여, 로봇들이 중앙 집중식 데이터 수집 없이 서로의 경험에서 학습하여 작물 상태와 살충제 살포 효율성을 개선할 수 있는 방법을 제시합니다. 각 로봇은 독립적으로 작물 상태와 화학 살포 효과에 대한 모델을 유지하고, 이 모델은 주기적으로 다른 로봇과 공유되어 살포 전략을 지속적으로 개선하고, 낭비를 줄이고 생산량을 높이는 데 기여합니다. 특히, 논문은 클러스터 기반 연합 학습을 제안하여 글로벌 서버의 계산 부하를 줄이고 클라이언트 간 통신 오버헤드를 최소화합니다.



### Assessment of Cell Nuclei AI Foundation Models in Kidney Pathology (https://arxiv.org/abs/2408.06381)
- **What's New**: 이 연구는 세 가지 최첨단 (SOTA) 세포 핵 기반 모델(Cellpose, StarDist, CellViT)의 성능을 대규모로 평가하여 신장 병리학에서의 세포 핵 분할 성능을 측정했습니다. 2,542개의 신장 전체 슬라이드 이미지(WSI)로 구성된 다양한 평가 데이터 세트를 만들었습니다. 이는 인간과 설치류 출처의 다양한 조직 유형, 크기 및 염색 방법을 포함합니다. 이 연구는 신장 병리학에 특화된 기반 모델의 필요성을 강조하며 기반 모델 간의 일치 분석을 수행하여 합의된 실패 사례를 파악했습니다.



### Modality-Balanced Learning for Multimedia Recommendation (https://arxiv.org/abs/2408.06360)
Comments:
          ACM Multimedia 2024 (Oral)

- **What's New**: 이 논문에서는 **모달 불균형 (modal imbalance)** 문제를 해결하기 위해 **반사실적 지식 증류 (Counterfactual Knowledge Distillation)** 프레임워크를 제안합니다. 이 프레임워크는 다양한 모달의 정보를 효과적으로 활용하여 추천 모델의 성능을 향상시키는 데 중점을 둡니다.



### How good nnU-Net for Segmenting Cardiac MRI: A Comprehensive Evaluation (https://arxiv.org/abs/2408.06358)
- **What's New**: 이 연구에서는 심장 자기 공명 영상(MRI) 세분화를 위한 강력한 의료 영상 세분화 도구인 nnU-Net 프레임워크의 성능을 평가합니다. 연구자들은 다양한 nnU-Net 구성(2D, 3D 전체 해상도, 3D 저해상도, 3D 캐스케이드 및 앙상블 모델)을 사용하여 다섯 개의 심장 세분화 데이터 세트에서 성능을 벤치마킹했습니다.



### Moo-ving Beyond Tradition: Revolutionizing Cattle Behavioural Phenotyping with Pose Estimation Techniques (https://arxiv.org/abs/2408.06336)
- **What's New**: 본 논문은 인공지능(AI) 기반의 동물 자세 추정 기술을 이용하여 축산업, 특히 소 사육 산업의 효율성을 높이는 새로운 방식을 제시합니다. AI는 축산업의 자동화 및 효율성 증대에 중요한 역할을 수행하며, 특히 자세 추정 기술을 통해 동물의 움직임과 건강 상태를 정확하게 파악하고 모니터링할 수 있습니다.  



### HeLiMOS: A Dataset for Moving Object Segmentation in 3D Point Clouds From Heterogeneous LiDAR Sensors (https://arxiv.org/abs/2408.06328)
Comments:
          Proc. IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS) 2024

- **What's New**: 본 논문에서는 HeLiMOS라는 새로운 데이터셋을 소개합니다. HeLiMOS는 다양한 유형의 LiDAR 센서 (기계적으로 회전하는 센서, 솔리드 스테이트 센서 등)로 수집된 3D 포인트 클라우드 데이터에 대한 이동 객체 분할 (MOS) 라벨을 제공합니다. 이 데이터셋은 기존의 MOS 연구에서 주로 사용되는 기계적으로 회전하는 omnidirectional LiDAR 센서만을 사용한 데이터셋과 달리, 불규칙적인 스캔 패턴을 가진 솔리드 스테이트 LiDAR 센서 데이터를 포함하고 있어 MOS 알고리즘의 일반화 성능을 평가하는 데 유용합니다. 또한, 본 논문에서는 instance-aware static map building 방법과 tracking-based false label filtering을 활용한 새로운 자동 라벨링 방법을 제시합니다. 이를 통해 사람이 직접 라벨링하는 작업량을 크게 줄일 수 있습니다. 마지막으로, HeLiMOS 데이터셋을 사용하여 널리 사용되는 최첨단 MOS 접근 방식의 성능을 평가하고, 다양한 LiDAR 센서에 관계없이 작동하는 sensor-agnostic MOS에 대한 새로운 방향을 제시합니다.



### From SAM to SAM 2: Exploring Improvements in Meta's Segment Anything Mod (https://arxiv.org/abs/2408.06305)
- **What's New**: SAM (Segment Anything Model), Meta에서 2023년 4월에 소개된 혁신적인 도구로, 이미지에서 객체를 자동으로 분할하는 기능을 제공합니다. SAM은 텍스트, 클릭 또는 경계 상자와 같은 프롬프트를 기반으로 객체를 분할합니다. 특히 SAM은 제로샷(zero-shot) 성능이 뛰어나 추가 학습 없이 10억 개 이상의 이미지 마스크를 사용하여 보지 못한 객체도 분할할 수 있습니다. SAM 2는 비디오 기능을 확장하여 이전 및 이후 프레임의 메모리를 활용하여 전체 비디오에서 정확한 분할을 생성하여 실시간에 가까운 성능을 제공합니다. 이 비교는 다양한 애플리케이션에서 정밀하고 효율적인 분할에 대한 증가하는 요구를 충족하기 위해 SAM이 어떻게 진화했는지 보여줍니다. 이 연구는 SAM과 같은 모델의 미래 발전이 컴퓨터 비전 기술을 개선하는 데 중요함을 시사합니다.

- **Technical Details**: SAM은 이미지 인코더, 프롬프트 인코더, 마스크 디코더의 세 가지 구성 요소로 구성됩니다. 이미지 인코더는 이미지를 처리하여 이미지를 고차원 특징 공간으로 변환합니다. 프롬프트 인코더는 텍스트, 클릭 또는 경계 상자와 같은 프롬프트를 처리하여 마스크 디코더가 사용할 수 있는 정보를 제공합니다. 마스크 디코더는 이미지 및 텍스트 입력을 마스크로 변환하여 객체를 분할합니다. SAM2는 비디오 기능을 확장하여 이전 및 이후 프레임의 메모리를 활용하여 전체 비디오에서 정확한 분할을 생성합니다. 메모리 인코더는 이전 프레임의 정보를 저장하고, 마스크 디코더는 메모리 인코더에서 저장된 정보와 현재 프레임의 정보를 결합하여 마스크를 생성합니다.

- **Performance Highlights**: SAM은 10억 개 이상의 이미지 마스크와 1,100만 개의 이미지로 훈련되어 제로샷 성능을 제공합니다. SAM2는 실시간에 가까운 성능으로 전체 비디오에서 정확한 분할을 생성합니다.



### Mipmap-GS: Let Gaussians Deform with Scale-specific Mipmap for Anti-aliasing Rendering (https://arxiv.org/abs/2408.06286)
Comments:
          9 pages

- **What's New**: 3D Gaussian Splatting (3DGS)는 뛰어난 렌더링 효율성과 높은 충실도로 인해 새로운 뷰 합성 분야에서 큰 관심을 받고 있습니다. 하지만 훈련된 Gaussian은 단일 스케일 훈련으로 인해 조정할 수 없는 표현으로 인해 심각한 줌 품질 저하를 겪습니다. 일부 방법은 원시 객체에 대한 선택적 렌더링 또는 필터링 기술과 같은 후처리 기법을 통해 이 문제를 해결하려고 시도하지만, 스케일별 정보는 Gaussian에 포함되어 있지 않습니다. 본 논문에서는 원시 객체 속성(예: 색상, 모양 및 크기)과 분포(예: 위치)를 자체 조정하여 Gaussian을 임의 스케일에 적응적으로 만들 수 있는 통합 최적화 방법을 제안합니다. mipmap 기술에서 영감을 받아, 목표 스케일에 대한 의사 지상 진실을 설계하고 스케일 일관성 지도 손실을 제안하여 3D Gaussian에 스케일 정보를 주입합니다. 본 방법은 플러그인 모듈이며, 줌 인 및 줌 아웃 알리어싱을 해결하기 위해 모든 3DGS 모델에 적용될 수 있습니다. 광범위한 실험은 본 방법의 효과를 입증합니다. 특히, 본 방법은 NeRF Synthetic 데이터 세트에서 줌 인에 대해 9.25dB, 줌 아웃에 대해 10.40dB의 PSNR로 3DGS를 능가합니다.



### Latent Disentanglement for Low Light Image Enhancemen (https://arxiv.org/abs/2408.06245)
- **What's New**: 이 논문은 저조도 이미지 향상(LLIE) 작업을 위한 잠재적 분리 기반 향상 네트워크(LDE-Net)를 제안합니다. LDE-Net은 입력 이미지를 잠재 공간(latent space)에서 분리하여 분리된 콘텐츠 및 조명 구성 요소에 오류가 남지 않도록 합니다. LLIE 작업의 경우, 콘텐츠 기능을 사용하여 조명 구성 요소의 향상을 지시하는 콘텐츠 인식 임베딩(CAE) 모듈을 설계했습니다. 하류 작업(예: 야간 UAV 추적 및 저조도 객체 감지)의 경우, 잠재적 분리 프레임워크를 기반으로 효과적인 경량 향상기를 개발했습니다. 포괄적인 정량적 및 정성적 실험을 통해 제안된 LDE-Net이 다양한 LLIE 벤치마크에서 최첨단 방법을 능가함을 보여줍니다. 또한, 하류 작업에 프레임워크를 적용하여 얻은 탁월한 결과는 잠재적 분리 설계의 유용성을 보여줍니다.

- **Technical Details**: 제안된 LDE-Net은 입력 이미지를 잠재 공간에서 분리하는 분리 모듈, 조명 구성 요소만 향상시키는 향상 모듈, 향상된 이미지를 재구성하는 재구성 모듈로 구성됩니다. 분리 모듈은 입력 이미지를 콘텐츠(Content)와 조명(Illumination)으로 분리하고 CAE 모듈은 콘텐츠 기능을 사용하여 조명 구성 요소의 향상을 지시합니다. 또한, 잠재적 분리 설계를 활용하여 하류 작업의 성능을 향상시키는 경량 향상 네트워크를 구현할 수 있습니다.

- **Performance Highlights**: 제안된 LDE-Net은 다양한 LLIE 벤치마크에서 최첨단 방법을 능가합니다. 또한, 야간 UAV 추적 및 저조도 객체 감지와 같은 하류 작업에 적용하여 탁월한 결과를 얻었습니다. 이는 잠재적 분리 설계의 유용성을 보여줍니다.



### 3D Reconstruction of Protein Structures from Multi-view AFM Images using Neural Radiance Fields (NeRFs) (https://arxiv.org/abs/2408.06244)
- **What's New**: 본 논문은 단백질 복합체(PC)의 3D 구조 예측에 Atomic Force Microscopy (AFM)과 딥러닝을 결합한 새로운 방법을 제시합니다. AFM은 다양한 방향으로 PC의 높이 맵(height map)을 생성하여 신경망이 3D 구조를 예측하는 데 필요한 풍부한 정보를 제공합니다. 연구팀은 이 정보를 이용하여 3D 재구성을 위해 UpFusion 모델(새로운 뷰를 합성하는 조건부 확산 모델)을 사용했습니다.



### Correlation Weighted Prototype-based Self-Supervised One-Shot Segmentation of Medical Images (https://arxiv.org/abs/2408.06235)
Comments:
          Accepted to ICPR 2024

- **What's New**: 본 논문은 의료 영상 분할에서 제한적인 데이터 환경을 해결하기 위해, **프로토타입 기반의 셀프-슈퍼바이즈드 원-샷 러닝(self-supervised one-shot learning)** 프레임워크를 제안합니다. 특히, **슈퍼픽셀(superpixels)**을 이용하여 생성된 의사 레이블(pseudo-labels)을 활용하여 분할 작업 자체를 학습하는 방식을 사용합니다. 또한, 각 쿼리 픽셀에 대한 동적 프로토타입(dynamic prototype)을 생성하기 위해 **상관관계 기반 확률 점수(correlation-based probability score)**를 사용하며, 이는 문맥적으로 관련된 프로토타입에 더 높은 가중치를 부여하는 역할을 합니다. 마지막으로, **사분면 마스킹 전략(quadrant masking strategy)**을 통해 오류 양성(false positives)을 줄이는 전략을 추가합니다.



### FruitNeRF: A Unified Neural Radiance Field based Fruit Counting Framework (https://arxiv.org/abs/2408.06190)
Comments:
          Project Page: this https URL

- **What's New**: FruitNeRF는 3D에서 모든 종류의 과일을 직접 계산하기 위해 최첨단 뷰 합성 방법을 활용하는 통합된 새로운 과일 계산 프레임워크를 소개합니다. 이 프레임워크는 단안 카메라로 촬영한 순서 없는 포즈 이미지 집합을 입력으로 받아 각 이미지에서 과일을 분할합니다. 시스템이 과일 종류에 독립적이 되도록, 모든 과일에 대한 이진 분할 마스크를 생성하는 기반 모델을 사용합니다. RGB와 의미적(semantic) 모드를 모두 활용하여 의미적 신경 복사 필드(semantic neural radiance field)를 학습합니다. 암시적 Fruit Field를 균일하게 샘플링하여 과일만 있는 포인트 클라우드를 얻습니다. 추출된 포인트 클라우드에 캐스케이드 클러스터링을 적용하여 정확한 과일 개수를 계산합니다. 신경 복사 필드를 사용하면 객체 추적 또는 광학 흐름과 같은 기존 방법보다 상당한 이점이 있습니다. 계산 자체가 3D로 수행되기 때문입니다. 이 방법은 과일을 중복 계산하지 않고 관련 없는 과일(예: 떨어진 과일 또는 배경 과일)을 잘못 계산하지 않습니다. 실제 데이터와 합성 데이터로 구성된 데이터 세트를 사용하여 방법론을 평가했습니다. 실제 데이터 세트는 수동으로 계산된 기준 진실(ground truth)이 있는 세 개의 사과 나무, 한 줄의 사과와 기준 진실 과일 위치가 있는 벤치마크 사과 데이터 세트, 그리고 사과, 자두, 레몬, 배, 복숭아, 망고와 같은 다양한 과일 유형으로 구성된 합성 데이터 세트로 구성됩니다. 또한 U-Net과 비교하여 기반 모델을 사용한 과일 계산 성능을 평가했습니다.



### Blind-Match: Efficient Homomorphic Encryption-Based 1:N Matching for Privacy-Preserving Biometric Identification (https://arxiv.org/abs/2408.06167)
Comments:
          Accepted to CIKM 2024 (Applied Research Track)

- **What's New**: Blind-Match, a novel biometric identification system based on homomorphic encryption (HE), is proposed for efficient and privacy-preserving 1:N matching. This system is designed to enhance performance in real-world applications such as Naver Cloud's FaceSign.



### OmniCLIP: Adapting CLIP for Video Recognition with Spatial-Temporal Omni-Scale Feature Learning (https://arxiv.org/abs/2408.06158)
Comments:
          ECAI-2024

- **What's New**: This paper introduces OmniCLIP, a framework that adapts CLIP for video recognition by learning "omni-scale features" encompassing spatial, temporal, and dynamic spatial-temporal information. OmniCLIP utilizes a Parallel Temporal Adapter (PTA) and Self-Prompt Generator (SPG) to enable efficient temporal modeling and capture dynamic object spatial features, respectively.

- **Technical Details**: OmniCLIP utilizes two main components: 
1. **Parallel Temporal Adapter (PTA):** PTA integrates a temporal attention mechanism that tracks information across frames at consistent spatial locations, enhancing temporal modeling capabilities. It works parallel with the frozen spatial CLIP block, integrating spatial information through a learnable addition operation, balancing temporal adaptation with computational efficiency.
2. **Self-Prompt Generator (SPG):** SPG leverages average pooling and a learnable projector to extract multi-scale spatial information, enabling dynamic adaptation to varying spatial-temporal scales and handling object movement across frames. This approach capitalizes on CLIP's strong spatial capabilities.

- **Performance Highlights**: OmniCLIP demonstrates superior performance compared to existing methods in various settings, including supervised, few-shot, and zero-shot video recognition tasks. Notably, it achieves a top-1 accuracy of 74.30% on HMDB51 in a 16-shot setting, surpassing even the fully trained MotionPrompt approach. The authors highlight the synergy between PTA and SPG, allowing OmniCLIP to effectively discern spatial information across frames and assess object scales over time, leading to robust and efficient video recognition performance.



### Novel View Synthesis from a Single Image with Pretrained Diffusion Guidanc (https://arxiv.org/abs/2408.06157)
Comments:
          6 pages, 7 figures

- **What's New**: HawkI++는 단일 입력 이미지에서 카메라 제어를 통해 새로운 뷰를 생성하는 혁신적인 방법입니다. 이 방법은 3D 데이터나 광범위한 훈련 없이도 복잡하고 다양한 장면을 처리할 수 있습니다. HawkI++는 기존 3D NVS 모델을 약한 지침으로 사용하여 3D 없는 뷰 합성 접근 방식에 통합하여 효율적으로 원하는 결과를 얻습니다.



### Efficient and Scalable Point Cloud Generation with Sparse Point-Voxel Diffusion Models (https://arxiv.org/abs/2408.06145)
- **What's New**: 본 논문에서는 고품질 및 다양한 3D 형상을 생성하면서 빠른 생성 시간을 유지할 수 있는 3D 생성 모델링을 위한 새로운 포인트 클라우드 U-Net 확산 아키텍처를 제안합니다. 제안된 네트워크는 포인트의 고해상도 표현과 희소 폭셀(sparse voxels)의 계산 효율성을 결합하는 이중 분기 아키텍처를 사용합니다. 본 논문에서 제안된 네트워크의 가장 빠른 변형은 포인트 클라우드 생성 모델을 평가하기 위한 가장 인기 있는 벤치마크인 무조건적 형상 생성에서 모든 비 확산 생성 접근 방식을 능가하는 반면, 가장 큰 모델은 이전의 최첨단 PVD에 비해 약 70%의 실행 시간으로 확산 방법 중 최첨단 결과를 달성합니다. 무조건적 생성 외에도, ShapeNet의 모든 범주에서 조건부 생성을 포함하여 광범위한 평가를 수행하여 모델의 더 큰 데이터 세트로의 확장성을 입증하고, 네트워크가 더 적은 시간 단계에서 고품질 포인트 클라우드를 생성할 수 있도록 하는 암묵적 생성을 통해 생성 시간을 더욱 단축시킵니다. 마지막으로, 포인트 클라우드 완성 및 초해상도에서 아키텍처의 성능을 평가합니다. 제안된 모델은 모든 작업에서 탁월한 성능을 보여주며, 포인트 클라우드 생성 모델을 위한 최첨단 확산 U-Net으로 자리매김합니다. 코드는 [this https URL](https://github.com/AI-Growth-Lab/SPVD)에서 공개적으로 제공됩니다.



### MR3D-Net: Dynamic Multi-Resolution 3D Sparse Voxel Grid Fusion for LiDAR-Based Collective Perception (https://arxiv.org/abs/2408.06137)
Comments:
          Accepted at IEEE ITSC 2024

- **What's New**: 이 논문은 MR3D-Net을 소개하는데, 이는 LiDAR 기반의 집단 인식을 위한 동적 다중 해상도 3D 희소 폭셀 그리드 융합 백본 아키텍처입니다. 이는 통신 대역폭에 따라 해상도를 조정할 수 있는 3D 희소 폭셀 그리드를 사용하여 동적 환경 표현을 제공합니다.

- **Technical Details**: MR3D-Net은 다중 해상도 희소 폭셀 그리드를 사용하여 환경 정보를 융합할 수 있는 새로운 융합 아키텍처를 제시합니다. 희소 폭셀 그리드는 환경의 기하학적 표현을 제공하며, 원시 데이터를 교환하는 것보다 전송되는 데이터 양을 크게 줄입니다. 또한 교환 가능하고 표준화할 수 있습니다.

- **Performance Highlights**: MR3D-Net은 OPV2V 3D 객체 탐지 벤치마크에서 최첨단 성능을 달성하며, 초기 융합에 비해 필요한 대역폭을 최대 94%까지 줄입니다.



### DPDETR: Decoupled Position Detection Transformer for Infrared-Visible Object Detection (https://arxiv.org/abs/2408.06123)
- **What's New**: 본 논문에서는 적외선-가시광선 객체 탐지에서 모달리티 정렬 불일치 문제를 해결하기 위해 분리된 위치 탐지 트랜스포머(DPDETR)를 제안합니다. DPDETR은 객체의 카테고리, 가시광선 모달리티 위치, 적외선 모달리티 위치를 명시적으로 공식화하여 네트워크가 두 모달리티 모두에서 객체의 정확한 위치를 학습하도록 합니다.



### RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation (https://arxiv.org/abs/2408.06110)
Comments:
          ECCV 2024 (oral)

- **What's New**: 이 논문은 3D 점 구름 분류 및 분할을 위한 새로운 회전 불변 아키텍처를 제안합니다. 기존의 점별 연산 대신, 입력 데이터의 각 참조점 주변에 국소 삼각형 표면을 구성하여 더 자세한 표면 구조를 포착합니다. 이를 통해 높은 표현력을 가진 회전 불변 표면 특징을 추출할 수 있습니다. 이러한 특징은 RISurConv라는 주의력 증강 컨볼루션 연산자에 통합되어 자기 주의력 계층을 통해 개선된 주의력 특징을 생성합니다. RISurConv를 기반으로 3D 점 구름 분석을 위한 효과적인 신경망을 구축하여 임의의 회전에 대해 불변하면서도 높은 정확도를 유지합니다.



### Towards Robust Monocular Depth Estimation in Non-Lambertian Surfaces (https://arxiv.org/abs/2408.06083)
- **What's New**: 이 논문은 비-람베르트 표면(예: 투명 또는 거울 표면)의 깊이 추정을 위한 새로운 프레임워크를 제안합니다. 기존의 방법들은 종종 비-람베르트 표면에서 제대로 된 깊이를 예측하지 못했지만, 이 방법은 훈련 과정에서 비-람베르트 표면 영역 지도(non-Lambertian surface regional guidance)를 사용하여 모델이 특정 영역의 특징을 직접 학습하도록 합니다. 또한 조명(lighting)이 깊이 예측에 미치는 영향을 최소화하기 위해 무작위 톤 매핑 증강(random tone-mapping augmentation)을 사용합니다. 추가적으로, 여러 노출 이미지가 있는 경우, 변형 오토인코더(Variational Autoencoders, VAE)를 이용하여 최적의 이미지를 선택하는 선택적 이미지 융합 모듈(optional image fusion module)을 제안합니다.



### Towards Adversarial Robustness via Debiased High-Confidence Logit Alignmen (https://arxiv.org/abs/2408.06079)
- **What's New**: 이 논문에서는 인버스 적대적 공격(inverse adversarial attacks)을 사용한 적대적 훈련(adversarial training)에서 나타나는 편향된 특징 활성화 문제를 해결하는 새로운 방법인 **Debiased High-Confidence Adversarial Training (DHAT)**을 제안합니다. DHAT는 인버스 적대적 예제에서 얻은 편향된 고신뢰도 로짓(logit)을 조정하고, 포그라운드 로짓 직교성(Foreground Logit Orthogonal Enhancement)을 향상시켜 모델의 주의를 정상 상태로 복원하는 두 가지 기술을 사용합니다.

- **Technical Details**: DHAT는 두 가지 주요 기술을 사용합니다: **Debiased High-Confidence Logit Regularization (DHLR)**와 **Foreground Logit Orthogonal Enhancement (FLOE)**. DHLR는 배경 특징 활성화에 대한 편향을 정량화하고 인버스 적대적 예제에서 얻은 편향된 고신뢰도 로짓을 재조정합니다. 이러한 정규화는 적대적 예제의 로짓을 편향되지 않은 고신뢰도 로짓과 정렬하여 적대적 견고성을 향상시키고 허위 상관 관계 편향(spurious correlation bias)을 완화합니다. FLOE는 고신뢰도 로짓과 배경 특징 간의 상관 관계를 아핀 공간에서 줄여 모델의 주의를 정상 상태로 복원하는 데 도움이 됩니다.

- **Performance Highlights**: DHAT는 다양한 데이터 세트에서 최첨단 기술을 능가하여 적대적 공격에 대한 뛰어난 견고성과 일반화 성능을 제공합니다. 또한, DHAT는 기존의 고급 적대적 훈련 방법과 원활하게 통합될 수 있습니다.



### CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer (https://arxiv.org/abs/2408.06072)
- **What's New**: CogVideoX라는 대규모 확산 변환기 모델이 텍스트 프롬프트를 기반으로 비디오를 생성하기 위해 소개되었습니다. 이 모델은 3D 변이형 오토인코더(VAE)를 활용하여 공간 및 시간 차원 모두에서 비디오를 효율적으로 압축하는 새로운 방법을 제안합니다. 텍스트-비디오 정렬을 개선하기 위해, 전문가 적응형 LayerNorm을 사용하는 전문가 변환기가 두 모드 간의 심층적인 융합을 용이하게 하도록 설계되었습니다. CogVideoX는 점진적인 학습 기법을 사용하여 의미 있는 움직임이 특징인 일관성 있고 긴 지속 시간의 비디오를 생성하는 데 능숙합니다.  또한, 다양한 데이터 전처리 전략과 비디오 캡션 방법을 포함하는 효과적인 텍스트-비디오 데이터 처리 파이프라인을 개발했습니다. 이 파이프라인은 CogVideoX의 성능을 크게 향상시켜 생성 품질과 의미적 정렬을 모두 개선합니다.



### A-BDD: Leveraging Data Augmentations for Safe Autonomous Driving in Adverse Weather and Lighting (https://arxiv.org/abs/2408.06071)
- **What's New**: A-BDD, a large synthetically augmented image dataset based on BDD100K, is introduced.  A-BDD includes over 60,000 images with semantic segmentation and bounding box annotations, covering diverse adverse weather and lighting conditions like rain, fog, overcast, and sunglare/shadow at varying intensity levels. This dataset is the first large-scale publicly available resource specifically designed for benchmarking and improving perception models under challenging weather and lighting conditions.



### ControlNeXt: Powerful and Efficient Control for Image and Video Generation (https://arxiv.org/abs/2408.06070)
Comments:
          controllable generation

- **What's New**: 본 논문에서는 ControlNeXt를 제안하며, 이미지 및 비디오 생성을 위한 효율적이고 강력한 제어 가능한 방법을 선보입니다. 기존 방법과 달리, ControlNeXt는 추가적인 계산 자원 없이 기본 모델에 최소한의 비용만 추가하는 효율적인 아키텍처를 사용합니다. 이를 통해 ControlNeXt는 다른 LoRA 가중치와 원활하게 통합되어 추가 훈련 없이 스타일 변경이 가능합니다. 또한 ControlNeXt는 기존 방법 대비 훈련 가능한 매개변수를 최대 90%까지 줄입니다. 더 나아가, 본 논문은 Zero-Convolution을 대체하는 Cross Normalization (CN) 방법을 제안하여 빠르고 안정적인 훈련 수렴을 달성합니다. 이미지와 비디오에 걸쳐 다양한 기본 모델로 실험을 수행한 결과, ControlNeXt의 강력함을 확인했습니다.



### BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training (https://arxiv.org/abs/2408.06047)
- **What's New**: 이 논문은 복잡한 야생 시나리오에서도 마스크 없이 고품질 이미지 기반 가상 피팅을 가능하게 하는 새로운 학습 패러다임을 제안합니다. 기존의 방법들은 정확한 마스크에 크게 의존했지만, 이 논문에서는 기존의 피팅 모델을 사용하여 야생 데이터에서 마스크 없이 학습하는 방식을 제시합니다. 또한, 학습 데이터를 더욱 다양하게 하기 위해 레이어 확산(Layer Diffusion)을 사용하여 배경과 전경을 합성하는 데이터 증강 기법을 적용했습니다. 마스크 없이도 피팅 영역을 정확하게 인식하도록 돕는 피팅 위치 손실(try-on localization loss)도 새롭게 설계했습니다. 



### ARPA: A Novel Hybrid Model for Advancing Visual Word Disambiguation Using Large Language Models and Transformers (https://arxiv.org/abs/2408.06040)
- **What's New**: This paper proposes a novel architecture called ARPA for Visual Word Sense Disambiguation (V-WSD). ARPA combines large language models (LLMs) with transformer-based vision encoders and a custom Graph Neural Network (GNN) layer. This architecture is designed to learn intricate relationships between visual and linguistic features, enhancing the model's ability to disambiguate words based on both text and image context.

- **Technical Details**: ARPA uses RoBERTa (a BERT-based language model) for text processing and Swin Transformer for visual feature extraction. Both modalities are projected into a shared embedding space, and then passed through a GCN layer to capture multimodal relationships. The paper also explores data augmentation techniques to improve robustness.

- **Performance Highlights**: ARPA surpasses previous V-WSD methods by achieving 15% improvement in accuracy and 6-8% improvement in Mean Reciprocal Rank (MRR).



### Layer-Specific Optimization: Sensitivity Based Convolution Layers Basis Search (https://arxiv.org/abs/2408.06024)
Comments:
          A revived draft of an unpublished (and never-to-be-published) article. For the sake of history, memory, and old times

- **What's New**: 본 논문은 딥러닝 모델의 파라미터 수를 줄이기 위한 새로운 방법을 제안합니다. 이 방법은 **Convolutional Layer (합성곱 층)**의 가중치에 대한 **Matrix Decomposition (행렬 분해)**을 적용하는 새로운 방식입니다. 본 논문에서 제안하는 방법의 핵심은 모든 합성곱을 훈련하는 것이 아니라, 일부 합성곱만 훈련하고 (**Basis Convolution (기저 합성곱)**) 나머지 합성곱은 기저 합성곱의 선형 결합으로 표현하는 것입니다. 또한, **Matrix Decomposition (행렬 분해)**을 적용할 때 모델 성능 저하가 없는 네트워크 레이어를 선택하는 빠른 방법을 제안합니다.



### ClickAttention: Click Region Similarity Guided Interactive Segmentation (https://arxiv.org/abs/2408.06021)
- **What's New**: 이 논문은 Click Attention 알고리즘을 제안하여 기존의 Click-based Segmentation 방식에서의 문제점을 해결합니다. 기존 방식은 Sparse Click Map을 입력으로 사용하여 특정 목표 객체를 분할하는 데 초점을 맞추어 전체 객체에 대한 정보를 제한적으로 사용하며, 이는 더 많은 클릭을 필요로 한다는 단점을 가지고 있습니다. 또한, 기존 방식은 높은 성능과 적은 클릭 횟수 사이의 균형을 맞추는 데 어려움을 겪습니다. 이 논문에서는 Positive Click 영역과 전체 입력 이미지의 유사성을 기반으로 Positive Click의 영향 범위를 확장하는 Click Attention 알고리즘을 제안합니다. 또한, Positive와 Negative Click 영역 간의 상호 간섭을 방지하여 정확도 저하를 최소화하는 Discriminative Affinity Loss를 제안합니다. 

- **Technical Details**: 제안된 Click Attention 알고리즘은 입력 이미지를 여러 Patch로 나누고 각 Patch의 유사성을 기반으로 Click Point의 영향 범위를 확장합니다. 또한, Discriminative Affinity Loss를 통해 Positive Click은 목표 객체에 집중하고 Negative Click은 배경에 집중하도록 유도하여 정확도를 향상시킵니다. 

- **Performance Highlights**: 실험 결과, 제안된 방식은 기존의 방식보다 뛰어난 성능을 보였으며, 적은 Click 횟수로 최첨단 성능을 달성했습니다.



### HeadGAP: Few-shot 3D Head Avatar via Generalizable Gaussian Priors (https://arxiv.org/abs/2408.06019)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 소량의 실제 데이터에서도 고품질 및 애니메이션 가능한 3D 헤드 아바타를 생성하는 새로운 방식을 제시합니다. HeadGAP라는 새로운 프레임워크는 사전 학습(prior learning)과 아바타 생성(avatar creation) 단계를 포함합니다. 사전 학습 단계에서는 대규모 다중 뷰 데이터셋에서 3D 헤드 사전을 추출하고, 아바타 생성 단계에서는 이 사전을 사용하여 소량의 데이터로 개인화된 아바타를 생성합니다. HeadGAP는 Gaussian Splatting 기반 오토디코더 네트워크와 부분 기반 동적 모델링을 사용하여 효과적으로 사전 정보를 포착합니다. 또한 개별 아이덴티티(identity)에 대한 개인화된 잠재 코드(latent codes)를 사용하여 가우시안 기본(Gaussian primitives)의 속성을 학습합니다. 아바타 생성 단계에서는 역변환(inversion) 및 미세 조정(fine-tuning) 전략을 활용하여 빠른 헤드 아바타 개인화를 실현합니다. 실험 결과, 본 모델은 헤드 사전을 효과적으로 활용하여 소량의 데이터로 개인화를 성공적으로 수행하여, 사진처럼 사실적인 렌더링 품질, 다중 뷰 일관성 및 안정적인 애니메이션을 달성했습니다.



### DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation (https://arxiv.org/abs/2408.06010)
Comments:
          First two authors contributed equally

- **What's New**: DEEPTalk은 음성 입력에서 다양하고 감정적으로 풍부한 3D 얼굴 표현을 생성하는 새로운 접근 방식을 소개합니다. 이 모델은 음성과 얼굴 움직임 모두에 대한 공동 감정 임베딩 공간을 형성하는 확률적 대조 학습을 사용하는 DEE(Dynamic Emotion Embedding)를 사용합니다. 또한 TH-VQVAE(Temporally Hierarchical VQ-VAE)를 설계하여 표현력이 뛰어나고 견고한 동작 사전을 구축하여 VAE 및 VQ-VAE의 한계를 극복합니다. 이러한 강력한 사전을 활용하여 DEEPTalk는 비자동 회귀적으로 코드북 색인을 예측하여 새로운 감정 일관성 손실을 통합하여 동적 얼굴 움직임을 생성합니다.

- **Technical Details**: DEEPTalk는 두 가지 핵심 구성 요소를 기반으로 합니다. 첫째, DEE는 음성과 얼굴 움직임 간의 감정적 상관 관계를 포착하여 공동 임베딩 공간을 형성합니다. 둘째, TH-VQVAE는 시간적 계층적 구조를 사용하여 동적 얼굴 움직임의 다양한 시간적 주파수를 포착하는 표현력이 뛰어나고 견고한 동작 사전을 만듭니다. DEEPTalk는 이러한 사전을 활용하여 비자동 회귀적으로 코드북 색인을 예측하고 새로운 감정 일관성 손실을 통해 생성된 표현이 입력 음성 감정을 일관되게 반영하도록 합니다.

- **Performance Highlights**: 다양한 데이터 세트에 대한 광범위한 실험 결과 DEEPTalk가 정확한 입술 동기화를 유지하면서 다양하고 감정적으로 표현력이 뛰어난 말하는 얼굴을 만드는 데 효과적임을 보여줍니다. 또한, DEEPTalk는 입술 동기화 측면에서 최첨단 모델을 능가합니다.



### An Analysis for Image-to-Image Translation and Style Transfer (https://arxiv.org/abs/2408.06000)
- **What's New**: 이 논문은 이미지-투-이미지 변환(Image-to-Image Translation)과 스타일 전이(Style Transfer)의 차이점과 연관성을 자세히 설명하여 두 기술 간의 혼동을 해소하고 더 나은 이해를 돕는 것을 목표로 합니다. 특히, 두 기술의 개념, 형태, 학습 방식, 평가 과정, 시각화 결과 등을 비교 분석하여 차이점을 명확히 밝힙니다.



### Diffuse-UDA: Addressing Unsupervised Domain Adaptation in Medical Image Segmentation with Appearance and Structure Aligned Diffusion Models (https://arxiv.org/abs/2408.05985)
- **What's New**: Diffuse-UDA는 의료 영상 분할에서 비지도 도메인 적응(UDA) 문제를 해결하기 위해 확산 모델을 활용하는 새로운 방법입니다. 이 방법은 타겟 도메인 특성과 다양한 구조를 갖는 고품질 이미지-마스크 쌍을 생성하여 UDA 작업을 향상시킵니다.



### Unseen No More: Unlocking the Potential of CLIP for Generative Zero-shot HOI Detection (https://arxiv.org/abs/2408.05974)
Comments:
          Accepted by ACM MM 2024

- **What's New**: This paper proposes a novel generation-based method called HOIGen for zero-shot Human-Object Interaction (HOI) detection, leveraging CLIP for feature generation instead of just extraction. Unlike previous embedding-based approaches, HOIGen effectively addresses the seen-unseen confusion problem by jointly training seen and unseen classes.  It introduces a CLIP-injected feature generator based on a variational autoencoder (VAE) to create synthetic image features, which are then mixed with realistic features from seen samples. This allows for a more comprehensive understanding of both seen and unseen HOI categories.



### Freehand Sketch Generation from Mechanical Components (https://arxiv.org/abs/2408.05966)
Comments:
          Published at ACM Multimedia (ACM MM) 2024

- **What's New**: 본 논문에서는 기계 부품의 자유로운 손 그림 스케치를 생성하는 새로운 기술인 MSFormer를 제안합니다. 이 기술은 기존 기술의 한계인 손 그림 스타일 부족 및 생성 모델의 비효율성을 극복하여 인간의 스케치 행동 패턴을 모방하는 2단계 생성 프레임워크를 구축합니다. 특히 MSFormer는 인간 손 그림 스타일을 유지하면서 기계 부품의 필수 모델링 정보를 보존하는 데 초점을 맞춥니다.



### Target Detection of Safety Protective Gear Using the Improved YOLOv5 (https://arxiv.org/abs/2408.05964)
- **What's New**: 본 논문에서는 YOLO-EA라는 새로운 모델을 제안하여 고위험 철도 건설 현장에서 개인 보호 장비 모니터링의 정확성을 향상시킵니다. YOLO-EA는 ECA(Efficient Channel Attention)를 백본의 합성곱 층에 통합하여 헬멧과 같이 작은 물체를 더 잘 식별할 수 있도록 합니다. 또한, GIoU(Generalized Intersection over Union) 손실을 EIoU(Efficient Intersection over Union) 손실로 대체하여 폐색된 상황에서도 목표물 인식 정확도를 높입니다. YOLO-EA는 실제 철도 건설 현장 감시 영상으로부터 수집된 데이터셋을 사용하여 실험적으로 효과를 입증했습니다. YOLOv5보다 98.9%의 정밀도와 94.7%의 재현율을 달성하여 각각 2.5%와 0.5% 향상되었으며, 70.774 fps의 실시간 성능을 유지했습니다.



### Boosting Adverse Weather Crowd Counting via Multi-queue Contrastive Learning (https://arxiv.org/abs/2408.05956)
Comments:
          11 pages, 7 figures

- **What's New**: 본 논문에서는 악천후 상황에서의 군중 계산 모델의 강인성을 높이기 위해, 악천후 환경에 대한 인식 능력을 갖춘 모델을 개발하는 데 중점을 둡니다. 이를 위해, 악천후 이미지의 제한된 가용성으로 인해 발생하는 클래스 불균형 문제를 해결하기 위한 새로운 두 단계 방법인 MQCL(Multi-queue Contrastive Learning)을 제안합니다. MQCL은 첫 번째 단계에서 다중 큐 MoCo 대조 학습 전략을 사용하여 날씨 인식 표현을 학습하고, 두 번째 단계에서 대조 학습의 지침을 활용하여 날씨 인식 표현을 정상 날씨 영역으로 변환합니다. 이러한 전략은 모델의 복잡성을 크게 증가시키지 않고도 강인성을 크게 향상시킵니다.



### Probabilistic Vision-Language Representation for Weakly Supervised Temporal Action Localization (https://arxiv.org/abs/2408.05955)
Comments:
          Accepted to ACM MM 2024

- **What's New**: 이 논문은 약한 지도 학습 시간적 액션 로컬라이제이션 (WTAL)을 위한 새로운 프레임워크인 PVLR (Probabilistic Vision Language Representation)을 제안합니다. PVLR은 확률적 임베딩 공간에서 VLP 지식과 인간 액션 지식을 통합하여 미세한 움직임 모델링을 위한 시간적 역동성을 완전히 고려합니다. 또한, 분포 대조 학습을 통해 통계적 거리에 기반한 독특한 확률적 임베딩 공간을 구축합니다.



### A Simple Task-aware Contrastive Local Descriptor Selection Strategy for Few-shot Learning between inter class and intra class (https://arxiv.org/abs/2408.05953)
Comments:
          Submitted to ICANN 2024

- **What's New**: 본 논문은 Task-Aware Contrastive Discriminative Local Descriptor Selection Network (TCDSNet)을 제안하며, 이는 적은 데이터로 새로운 클래스를 분류하는 few-shot 이미지 분류 문제를 해결하기 위해 설계되었습니다. TCDSNet은 기존 방법과 달리 지원 클래스의 로컬 디스크립터에서 배경 특징을 활용하는 새로운 방법을 제시합니다. TCDSNet은 지원 클래스의 로컬 디스크립터에 대한 contrastive discriminative score를 계산하여 차별적 디스크립터를 선택하고, 이를 기반으로 쿼리 디스크립터를 선택하여 특정 작업에 맞게 적응력을 높입니다. 이를 통해 기존 방법의 단점을 보완하고, 일반적 및 미세 분류 데이터셋에서 우수한 성능을 보여줍니다.



### Optimizing Vision Transformers with Data-Free Knowledge Transfer (https://arxiv.org/abs/2408.05952)
- **What's New**: This paper introduces a novel approach to compress large Vision Transformers (ViTs) models using data-free knowledge distillation (DFKD) for efficient deployment on resource-constrained devices. The DFKD technique leverages synthetic data generation to transfer knowledge from a larger, pre-trained teacher model to a smaller student model without requiring access to the original training data. This addresses the limitations of traditional knowledge distillation methods that rely on real training data.



### MV2DFusion: Leveraging Modality-Specific Object Semantics for Multi-Modal 3D Detection (https://arxiv.org/abs/2408.05945)
- **What's New**: MV2DFusion이라는 새로운 멀티모달 3D 객체 감지 프레임워크를 소개합니다. 이 프레임워크는 카메라와 LiDAR 센서의 강점을 결합하여 보다 강력하고 정확한 객체 감지를 목표로 합니다. MV2DFusion은 이미지 쿼리 생성기와 포인트 클라우드 쿼리 생성기를 사용하여 모달리티별 객체 의미 정보를 효과적으로 통합합니다. 이는 기존 방법들과 달리 특정 모달리티에 편향되지 않고, 모달리티의 장점을 균형 있게 활용할 수 있도록 설계되었습니다.

- **Technical Details**: MV2DFusion은 쿼리 기반 퓨전 메커니즘을 사용하여 이미지 및 포인트 클라우드 정보를 통합합니다. 이미지 쿼리 생성기는 이미지 특징과 정렬되도록 설계되었으며, 포인트 클라우드 쿼리 생성기는 포인트 클라우드 데이터에서 객체 의미 정보를 추출합니다. 이러한 쿼리들은 희소 퓨전 프로세스를 통해 효율적이고 정확한 객체 감지를 가능하게 합니다. MV2DFusion은 다양한 이미지 및 포인트 클라우드 기반 탐지기와 통합될 수 있으며, 유연성이 뛰어나기 때문에 향후 발전 가능성이 높습니다.

- **Performance Highlights**: nuScenes와 Argoverse2 데이터셋에서 MV2DFusion은 최첨단 성능을 달성했습니다. 특히 장거리 탐지 시나리오에서 탁월한 성능을 보였습니다. 또한, MV2DFusion은 희소 퓨전 전략을 사용하여 메모리 소비 및 계산 비용을 최소화하면서도 효율적인 장거리 감지를 가능하게 합니다.



### Spb3DTracker: A Robust LiDAR-Based Person Tracker for Noisy Environmen (https://arxiv.org/abs/2408.05940)
Comments:
          17 pages, 5 figures

- **What's New**: 본 논문은 LiDAR 기반 사람 검출 및 추적 (PDT) 분야의 주요 과제를 다루며,  기존 TBD(Tracking-by-Detection) 방식의 한계를 극복하기 위한 SpbTrack 이라는 강력한 사람 추적 알고리즘을 제안합니다.  SpbTrack은 다양한 환경에서 뛰어난 성능을 보이며 특히 소음이 많은 데이터셋에서 탁월한 결과를 제공합니다. KITTI 데이터셋과 맞춤형 사무실 실내 데이터셋에서 최첨단 성능을 달성합니다.  



### UniPortrait: A Unified Framework for Identity-Preserving Single- and Multi-Human Image Personalization (https://arxiv.org/abs/2408.05939)
Comments:
          Tech report; Project page: this https URL

- **What's New**: 이 논문은 UniPortrait라는 혁신적인 인물 이미지 개인화 프레임워크를 소개하며, 높은 얼굴 정확도(fidelity), 광범위한 얼굴 편집 기능, 자유 형식 입력 설명, 다양한 레이아웃 생성을 통해 단일 ID 및 다중 ID 맞춤 설정을 통합합니다.



### Deep Geometric Moments Promote Shape Consistency in Text-to-3D Generation (https://arxiv.org/abs/2408.05938)
Comments:
          9 pages, 8 figures

- **What's New**: This paper introduces MT3D, a novel text-to-3D generative model that leverages a high-fidelity 3D object to overcome viewpoint bias and infuse geometric understanding into the generation process, leading to more consistent 3D representations.



### Multi-scale Contrastive Adaptor Learning for Segmenting Anything in Underperformed Scenes (https://arxiv.org/abs/2408.05936)
- **What's New**: MCA-SAM, a novel framework for enhancing the adaptability of the Segmenting Anything Model (SAM) in underperforming scenarios, is introduced. This framework leverages multi-scale contrastive adaptors (MC-adaptors) to improve SAM's discriminative ability and adaptability at both token and sample levels. The MC-adaptor comprises Token-level Contrastive adaptors (TC-adaptor) and Sample-level Contrastive adaptors (SC-adaptor) which are trained using contrastive losses, leading to more robust and effective representational learning.



### A Simple Early Exiting Framework for Accelerated Sampling in Diffusion Models (https://arxiv.org/abs/2408.05927)
Comments:
          ICML 2024

- **What's New**: 본 논문에서는 확산 모델의 샘플링 속도를 향상시키기 위해 적응형 점수 추정(Adaptive Score Estimation, ASE) 프레임워크를 제안합니다. ASE는 점수 추정 네트워크의 계산량을 시간 단계에 따라 적응적으로 할당하여 전체 샘플링 시간을 줄이는 것을 목표로 합니다. 이를 위해, 시간 의존적인 종료 스케줄(exit schedule)을 기반으로 점수 추정 네트워크의 일부 매개변수를 건너뛰는 조기 종료 방식(early-exiting scheme)을 도입합니다.

- **Technical Details**: ASE는 확산 모델의 샘플링 과정에서 시간에 따라 점수 추정의 복잡성이 다를 수 있다는 가정에서 출발합니다. 즉, 특정 시간 단계에서는 적은 수의 매개변수로도 정확한 점수 추정이 가능할 수 있다는 것입니다. ASE는 이러한 가정을 바탕으로 시간에 따른 블록 드롭(block-dropping) 스케줄을 활용하여 각 시간 단계마다 필요한 계산량을 조절합니다. ASE는 기존 확산 모델의 백본 아키텍처와 다양한 솔버(solver)와의 호환성을 갖추고 있으며, 샘플링 속도를 향상시키기 위해 다른 방법들과 함께 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, ASE는 이미지 합성 작업에서 기존 확산 모델의 샘플링 처리량을 크게 향상시키면서 이미지 품질은 유지했습니다. 또한, ASE는 다양한 유형의 솔버와의 통합을 통해 샘플링 속도를 더욱 향상시키는 것으로 나타났습니다.



### PAFormer: Part Aware Transformer for Person Re-identification (https://arxiv.org/abs/2408.05918)
Comments:
          34 pages, 8 figures

- **What's New**: 이 논문은 기존의 부분적인 사람 재식별(ReID) 방법의 해부학적 인식 부족 문제를 해결하기 위해, 자세 추정 기반 ReID 모델인 '부분 인식 트랜스포머(PAFormer)'를 제시합니다. PAFormer는 '포즈 토큰'이라는 학습 가능한 매개변수를 도입하여 이미지의 부분 영역과 각 신체 부위 간의 상관관계를 추정함으로써 부분 인식을 주입합니다. 또한 PAFormer는 학습 기반 가시성 예측기를 사용하여 각 신체 부위의 폐색 정도를 추정하고, 지상 진실 가시성 점수를 사용한 교사 강제 기법을 통해 가시적인 부분만으로 학습할 수 있도록 합니다.



### Deep Multimodal Collaborative Learning for Polyp Re-Identification (https://arxiv.org/abs/2408.05914)
Comments:
          Work in progress. arXiv admin note: text overlap with arXiv:2307.10625

- **What's New**: 이 연구에서는 대장 내시경 용종 재식별 (Polyp ReID)을 위한 딥 멀티모달 협업 학습 프레임워크인 DMCL을 제안합니다. DMCL은 시각적 정보와 텍스트 정보를 동시에 활용하여 모달 간 협업을 장려하고 의료 환경에서 일반화 능력을 강화합니다. 또한, 이 연구에서는 동적 멀티모달 특징 융합 전략을 도입하여 멀티모달 융합을 위한 최적화된 멀티모달 표현을 활용합니다.



### Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts (https://arxiv.org/abs/2408.05905)
Comments:
          Accepted by ACMMM2024

- **What's New**: This paper presents a novel method called STPrompt for weakly supervised video anomaly detection and localization (WSVADL). STPrompt leverages pre-trained vision-language models (VLMs) to learn spatio-temporal prompt embeddings for identifying specific local regions of anomalies, enabling accurate video anomaly detection while mitigating the influence of background information.



### HcNet: Image Modeling with Heat Conduction Equation (https://arxiv.org/abs/2408.05901)
- **What's New**: 이 논문은 이미지 모델링을 위해 열 전도 방정식을 사용하는 새로운 딥 러닝 아키텍처인 HcNet을 제안합니다. 기존 모델 아키텍처의 설계를 열 전도 방정식을 통해 해석하고, 이를 기반으로 새로운 모델 구성 요소를 제안합니다. 또한, 열 전도 방정식의 지식을 활용하여 모델 아키텍처 설계를 가이드하고 해석력을 향상시키는 것을 목표로 합니다. 이는 CNN (Convolutional Neural Networks) 및 ViT (Vision Transformers)와 같은 기존 딥 러닝 모델의 제한을 극복하는 새로운 방법을 제시합니다.



### Classifier Guidance Enhances Diffusion-based Adversarial Purification by Preserving Predictive Information (https://arxiv.org/abs/2408.05900)
Comments:
          Accepted by ECAI 2024

- **What's New**: 이 논문에서는 기존의 확산 모델 기반 적대적 정화 방법이 핵심 잡음 제거 과정에서 샘플 정보를 점차적으로 잃어버리는 문제점을 지적하고, 분류기 신뢰도를 이용하여 이러한 정보 손실을 억제하는 새로운 방법인 분류기 신뢰도 기반 정화(COUP) 알고리즘을 제안합니다. COUP는 적대적 예제를 정화하는 동시에 분류기 결정 경계에서 멀리 유지하여 레이블 이동 문제를 최소화합니다.



### GlyphPattern: An Abstract Pattern Recognition for Vision-Language Models (https://arxiv.org/abs/2408.05894)
- **What's New**: GlyphPattern, a novel dataset with 954 items, assesses abstract pattern recognition in Vision-Language Models (VLMs) by testing their understanding of natural language descriptions of visual patterns. This dataset leverages patterns from 40 writing systems with three visual presentation styles, aiming to push the limits of VLMs in recognizing abstract patterns.



### CMAB: A First National-Scale Multi-Attribute Building Dataset Derived from Open Source Data and GeoAI (https://arxiv.org/abs/2408.05891)
Comments:
          43 pages, 20 figures

- **What's New**: 본 논문은 중국 전역의 건물에 대한 다양한 속성을 빠르게 추출하는 지리공간 인텔리전스 프레임워크를 소개합니다. 이는 중국 전역의 3,667개 자연 도시를 포괄하는 최초의 국가 규모의 다중 속성 건물 데이터셋(CMAB)을 생성합니다. 이 데이터셋은 지붕, 높이, 기능, 나이, 품질 등 건물의 다양한 속성을 제공합니다.



### Enhancing 3D Transformer Segmentation Model for Medical Image with Token-level Representation Learning (https://arxiv.org/abs/2408.05889)
- **What's New**: 의료 이미지 분야에서, Swin Transformer는 픽셀 단위 예측에서 유망한 효과를 보였지만, 추가 데이터셋 없이 사전 훈련된 모델이 하위 작업인 의미론적 분할에서 성능을 더욱 향상시킬 수 있는지 여부는 아직 탐구되지 않았습니다. 기존의 표현 학습 방법은 3D 볼륨의 수가 제한적이고 계산 비용이 높아 적용에 어려움이 있었습니다. 또한, Transformer를 위해 특별히 설계된 대부분의 사전 훈련 작업은 Swin Transformer의 계층적 구조에는 적용할 수 없습니다. 본 연구는 볼륨 수준의 전역 특징 대신, 다른 증강된 뷰에서 추출된 토큰 임베딩 간의 일치도를 극대화하는 토큰 수준의 표현 학습 손실을 제안합니다. 또한, 새로운 손실에 의해 발생하는 잠재적인 표현 붕괴를 파악했습니다. 이 붕괴를 방지하기 위해, 입력 볼륨의 증강된 뷰를 회전하고 뒤집는 간단한 '회전 및 복원' 메커니즘을 개발했습니다. 이 메커니즘은 특징 맵에서 토큰의 순서를 복원합니다. 또한, 동일한 위치에 있지만 다른 볼륨에서 추출된 토큰 간의 차별화를 해결하기 위해 대조 손실을 수정했습니다. 두 개의 공개 의료 분할 데이터셋에서 사전 훈련 방식을 테스트한 결과, 하위 작업인 분할 작업에서 기존 최첨단 사전 훈련 방법보다 더 큰 개선을 보였습니다.



### LaWa: Using Latent Space for In-Generation Image Watermarking (https://arxiv.org/abs/2408.05868)
- **What's New**: LaWa, a new in-generation image watermarking method for Latent Diffusion Models (LDMs), effectively integrates watermarking into the image generation process itself, addressing the issue of malicious use of AI-generated images. This method utilizes a multi-scale embedding approach to modify the latent space of pre-trained autoencoders, achieving robustness against image transformations while preserving image quality.

- **Technical Details**: LaWa utilizes a coarse-to-fine multi-scale embedding module that modifies the frozen intermediate layers of the LDM decoder to ensure robustness against geometric attacks. The watermarking process utilizes a simple yet effective spatial coding scheme, improving the trade-off between robustness and perceptual quality. LaWa can be applied to any pre-trained LDM without further fine-tuning, allowing for versatile application in various image generation tasks.

- **Performance Highlights**: LaWa outperforms previous methods in terms of perceptual quality, robustness against attacks, and computational complexity. It also exhibits a very low false positive rate.

- **Advantages**: LaWa offers advantages over existing post-generation methods by improving the trade-off between quality and robustness through the use of the same latent features for image generation and watermarking. LaWa can be extended to work as a general post-generation image watermarking technique, making it versatile and adaptable.

- **Code**: The code is available here.

- **Keywords**: image watermarking, latent diffusion model, generative model, in-generation watermarking, multi-scale embedding, robustness, perceptual quality, false positive rate



### SABER-6D: Shape Representation Based Implicit Object Pose Estimation (https://arxiv.org/abs/2408.05867)
- **What's New**: SABER: A novel encoder-decoder architecture for learning 6D pose estimation of objects in an embedding space, specifically designed to handle object symmetry without requiring explicit symmetry labels. SABER leverages shape representation at different orientations to learn rotations, enabling it to handle symmetrical objects effectively. The approach utilizes DeepSDF for shape representation and is trained in a two-stage pipeline, first pre-training the decoder and then jointly training the encoder and decoder.



### Real-Time Drowsiness Detection Using Eye Aspect Ratio and Facial Landmark Detection (https://arxiv.org/abs/2408.05836)
- **What's New**: 이 논문은 실시간 졸음 감지 시스템을 소개합니다. 이 시스템은 눈의 개방 비율(EAR, Eye Aspect Ratio) 및 얼굴 랜드마크 검출 기술을 사용하여 졸음을 감지합니다.



### Robust Domain Generalization for Multi-modal Object Recognition (https://arxiv.org/abs/2408.05831)
Comments:
          6 pages, 2 figures. This is a preprint version of the article. The final version will be published in the proceedings of the IEEE conference

- **What's New**: 이 논문은 비전-언어 사전 훈련(vision-language pre-training) 모델인 CLIPood의 제한 사항을 해결하여 도메인 일반화(domain generalization) 성능을 향상시키는 새로운 방법을 제안합니다. CLIPood는 다양한 도메인에서 개념 인식(concept recognition)을 강화하는 비전-언어 쌍(visual-language pairs)의 감독을 활용하지만, 손실 함수(loss function) 활용, 백본(backbone)의 일반성 및 클래스 인식 시각 융합(class-aware visual fusion) 측면에서 제한점을 가지고 있습니다. 이 논문에서는 실제 손실(actual loss)을 추론하고, 더 큰 비전-언어 백본으로 평가를 확장하며, 클래스 인식 시각 융합을 강화하는 믹스업 손실(Mixup-CLIPood)을 소개합니다. 



### HySparK: Hybrid Sparse Masking for Large Scale Medical Image Pre-Training (https://arxiv.org/abs/2408.05815)
Comments:
          Early accept at MICCAI 2024

- **What's New**: 본 논문은 의료 이미지 분석을 위한 **하이브리드 아키텍처(hybrid architecture)** 기반의 **생성적 자기 지도 학습(generative self-supervised learning)** 전략인 **HySparK**를 제안합니다. HySparK는 CNN과 Transformer를 결합하여 국소 및 전역 표현을 동시에 학습하는 새로운 방법입니다. 특히, HySparK는 **하향식 마스크(bottom-up masking)**, **희소 합성곱(sparse convolution)**, 그리고 **계층적 디코더(hierarchical decoder)**를 통해 기존의 문제점들을 해결합니다.



### Egocentric Vision Language Planning (https://arxiv.org/abs/2408.05802)
- **What's New**: This paper introduces EgoPlan, a novel framework for building more general embodied agents that can perform long-horizon tasks in household scenarios from an egocentric perspective. EgoPlan leverages a diffusion model as a world model to simulate the dynamics between states and actions, while using a Large Multi-modal Model (LMM) as a planner to break down instructions into sub-goals and select actions.

- **Technical Details**: EgoPlan tackles the challenge of partially observable environments by using a diffusion model to predict the next state based on current observations and actions. To enhance generalization across different environments, the model incorporates style transfer (using LoRA) and optical flow techniques to capture motion patterns and adapt to variations in environment dynamics. The LMM serves as a planner, decomposing instructions into sub-goals and selecting actions that align with these sub-goals.

- **Performance Highlights**: EgoPlan demonstrates improvements in long-horizon task success rates compared to baselines across different household scenarios. The paper provides empirical evidence for the effectiveness of the world model in generating high-quality images and predicting accurate optical flow, ultimately aiding in decision-making. The framework’s generalization capabilities are validated through experiments in different environments.



### U-DECN: End-to-End Underwater Object Detection ConvNet with Improved DeNoising Training (https://arxiv.org/abs/2408.05780)
- **What's New**: 본 논문은 수중 환경에서 색상 왜곡 노이즈 (color cast noise) 문제를 해결하는 동시에 속도와 효율성을 높인 수중 객체 탐지 모델 'U-DECN'을 제안합니다. U-DECN은 기존 DETR (Detection Transformer)의 장점을 활용하면서 ConvNet (Convolutional Neural Network) 아키텍처를 사용하여 속도를 향상시켰습니다. 또한, 별도의 대조 학습 (Contrastive DeNoising) 전진 방법과 변형 가능한 합성곱 (Deformable Convolution)을 통합하여 수중 객체 탐지 성능을 향상시켰습니다. 특히, 수중 색상 왜곡 노이즈에 대한 일반화를 개선하기 위해 수중 색상 잡음 제거 쿼리를 도입했습니다.



### Seg-CycleGAN : SAR-to-optical image translation guided by a downstream task (https://arxiv.org/abs/2408.05777)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문은 선박 표적의 정확한 번역을 향상시키기 위해 사전 훈련된 의미론적 분할 모델에서 얻은 의미 정보를 활용하여 Seg-CycleGAN이라는 GAN 기반 SAR-to-optical 이미지 번역 방법을 제안합니다.



### Efficient Test-Time Prompt Tuning for Vision-Language Models (https://arxiv.org/abs/2408.05775)
- **What's New**: 이 논문에서는 Self-TPT라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 시각-언어 모델에서 **테스트 시간 프롬프트 튜닝(TPT)**에 대한 효율적인 접근 방식을 제시합니다. 특히, **자가 지도 학습(SSL)**을 이용하여 기존의 per-image 프롬프트 적응의 계산 비용을 줄이고 확장성을 높입니다. Self-TPT는 **대조 학습(Contrastive Learning)**을 기반으로 한 **대조 프롬프트 학습(CPT)**을 도입하여 새 클래스에 대한 적응 과정을 효율화합니다.

- **Technical Details**: Self-TPT는 크게 두 단계로 구성됩니다. 첫째, 소스 데이터를 사용하여 자가 지도 학습과 분류 작업을 공동 학습합니다. 둘째, 테스트 단계에서는 자가 지도 학습만을 사용하여 새 클래스에 대한 적응을 수행합니다. CPT는 클래스 내 거리를 최소화하고 클래스 간 구별성을 극대화하는 데 중점을 둡니다. 또한, 기울기 일치 손실(GM Loss)을 도입하여 CPT와 분류 작업의 기울기 유사성을 높입니다.

- **Performance Highlights**: Self-TPT는 세 가지 벤치마크에서 최첨단 성능을 달성했습니다. 특히, 기존의 TPT 기반 방법인 PromptAlign에 비해 25배 빠른 추론 속도를 보였으며 메모리 사용량은 30배 감소했습니다. 이러한 결과는 Self-TPT가 효율성과 효과성 간의 균형을 효과적으로 달성함을 보여줍니다.



### An analysis of HOI: using a training-free method with multimodal visual foundation models when only the test set is available, without the training s (https://arxiv.org/abs/2408.05772)
- **What's New**: 이 연구는 훈련 데이터셋 없이 테스트 데이터셋만 사용하여, 멀티모달 비주얼 기반 모델(Multimodal Visual Foundation Model)을 사용한 훈련 없는(Training-free) 방식으로 Human-Object Interaction (HOI)를 수행합니다.

- **Technical Details**:  세 가지 실험 설정을 사용합니다: (1) 기본 진실(ground truth)에서 ⟨human, object⟩ 쌍을 멀티모달 비주얼 기반 모델에 입력하여 다양한 동사를 포함한 텍스트 프롬프트와 비교하여 동사 확률 분포를 얻습니다. (2) 기본 진실의 쌍의 특징을 혼합하여 'human'은 모든 기본 진실의 human 바운딩 박스를 포함하고 'object'는 모든 바운딩 박스를 포함합니다. 이것들은 쿼리 모듈에 입력되어 첫 번째 설정과 유사한 동사 결과를 생성합니다. (3) 쌍이 없는 grounding DINO에서 추출된 바운딩 박스를 사용하여 두 번째 설정과 유사한 방식으로 동사 결과를 얻습니다. 이 연구는 희소한/비 희소한, 본/보지 못한 조합/객체/동사와 같은 차이점이 관련이 없음을 보여줍니다.

- **Performance Highlights**: 희소한 클래스는 human과 object의 임의 조합에 무감각한 반면, 비 희소한 클래스는 민감합니다.  RF-UC(Rare First Unseen Combinations) 설정에서 꼬리 HOI(희소한 클래스) 범주는 보지 못한 클래스로 지정되며, 보지 못한 클래스(희소한 클래스)는 임의 조합에 무감각한 반면, 본 클래스(비 희소한)는 민감합니다.  NF-UC(Non-rare First Unseen Combinations) 설정에서는 머리 HOI 범주(비 희소한)를 보지 못한 클래스로 지정하고, 보지 못한 클래스(비 희소한)는 임의 조합에 민감하고, 본 클래스(희소한)는 민감하지 않습니다. 보지 못한/본 객체 또는 동사가 포함된 실험에서 임의 조합에 대한 민감도는 보지 못한 및 본 분류 간에 일관성을 유지합니다.  이 연구는 멀티모달 비주얼 기반 모델의 제로 샷/퓨 샷 기능이 아직 완전히 구현되지 않았다는 것을 보여줍니다.



### PRECISe : Prototype-Reservation for Explainable Classification under Imbalanced and Scarce-Data Settings (https://arxiv.org/abs/2408.05754)
- **What's New**: PRECISe, a novel explainable-by-design deep learning model, is introduced to address the challenges of limited training data, class imbalance, and explainability in medical image classification.

- **Technical Details**: PRECISe is composed of three main components: an auto-encoder (encoder fitalic_f and decoder gitalic_g), a prototype-metric layer pitalic_p, and a linear classification layer witalic_w. The auto-encoder compresses input images into a lower-dimensional latent space, while the prototype-metric layer calculates the Euclidean distance between the encoded input and learned prototypes in the latent space, transforming it into a new metricitalic_m space. The final linear classification layer predicts the probability distribution over the classes based on this transformed vector.

- **Performance Highlights**: Evaluations on imbalanced medical image datasets demonstrate PRECISe's superior performance in data-efficient generalization to minority classes. It achieves an accuracy of ~87% in detecting pneumonia in chest x-rays with only <60 training images, outperforming current state-of-the-art methods. PRECISe's ability to generate easily interpretable predictions is showcased in a case study, highlighting its practical value and reliability for medical imaging tasks.

- **Key Contributions**: PRECISe, a novel explainable-by-design deep learning model, is introduced to address the challenges of limited training data, class imbalance, and explainability in medical image classification.

- **Benefits**: PRECISe demonstrates superior performance in data-efficient generalization to minority classes, outperforming current state-of-the-art methods. It provides easily interpretable predictions, enhancing trust and reliability in medical imaging tasks.



### RTF-Q: Unsupervised domain adaptation based retraining-free quantization network (https://arxiv.org/abs/2408.05752)
- **What's New**: 본 논문은 리소스 제약이 있는 에지 장치에서의 비지도 도메인 적응을 위한 새로운 "재훈련 없는 양자화(RTF-Q) 네트워크"를 제안합니다. RTF-Q 네트워크는 다양한 계산 예산에 따라 작동할 수 있는 다양한 계산 비용을 가진 양자화된 서브네트워크를 특징으로 합니다. RTF-Q는 네트워크 성능에 미치는 영향을 최소화하면서도 Imagenet-1K에서의 비용이 많이 드는 사전 훈련 없이도 공식 가중치 파일을 직접 로드할 수 있습니다. 또한, RTF-Q는 네트워크의 계산 부하와 메모리 사용량을 줄이기 위해 양자화 인식 훈련을 활용하여 전체 정밀도 네트워크의 BitOPs를 최소 1/16로 줄입니다.



### FADE: A Dataset for Detecting Falling Objects around Buildings in Video (https://arxiv.org/abs/2408.05750)
Comments:
          11 pages, 10 figures

- **What's New**: 본 논문은 건물 주변에서 떨어지는 물체를 감지하기 위한 새로운 대규모 데이터셋인 FADE (FAlling Object DEtection around Buildings)를 처음으로 제안합니다. FADE 데이터셋은 다양한 장면, 떨어지는 물체 종류, 날씨 조건, 해상도를 포함하며, 1,881개의 비디오와 164,314개의 프레임으로 구성됩니다. 또한, FADE 데이터셋에서 떨어지는 물체를 효과적으로 감지하기 위해, 움직임 정보를 활용하는 새로운 객체 감지 방법인 FADE-Net을 제안합니다. 본 논문에서는 FADE-Net을 기존 객체 감지 방법, 비디오 객체 감지 방법, 움직이는 객체 감지 방법과 비교하여 광범위하게 평가하고 분석합니다.



### Efficient and Versatile Robust Fine-Tuning of Zero-shot Models (https://arxiv.org/abs/2408.05749)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 논문은 대규모 이미지-텍스트 사전 훈련 모델을 하위 작업에 효율적으로 미세 조정하는 새로운 방법인 Robust Adapter(R-Adapter)를 제안합니다. R-Adapter는 경량 모듈을 사전 훈련된 모델에 통합하고 새로운 자기 앙상블 기법을 사용하여 OOD(Out-of-Distribution) 강인성을 높이고 저장 비용을 크게 줄입니다. 또한 비전-언어 하위 작업에 대한 미세 조정을 위해 설계된 MPM-NCE 손실 함수를 제안합니다. 이 손실 함수는 여러 이미지-텍스트 쌍의 정확한 정렬과 차별적인 특징 학습을 보장합니다.



### Improving Adversarial Transferability with Neighbourhood Gradient Information (https://arxiv.org/abs/2408.05745)
- **What's New**: 본 논문에서는 '인접 기울기 정보(Neighbourhood Gradient Information)'를 활용하여 블랙박스 공격 시나리오에서 적대적 예제의 전이성(Transferability)을 향상시키는 새로운 공격 기법인 'NGI-Attack'을 제안합니다. NGI-Attack은 '예제 역추적(Example Backtracking)' 및 '다중 마스크(Multiplex Mask)' 전략을 통합하여 기울기 정보를 효과적으로 활용합니다. 특히, 초기 단계에서 생성된 기울기 정보가 높은 전이성을 보유한다는 점에 착안하여, 해당 정보를 축적하고 활용하는 전략을 제시합니다.

- **Technical Details**: NGI-Attack은 다음과 같은 두 가지 핵심 전략을 사용합니다:

1. **예제 역추적(Example Backtracking):** 깨끗한 이미지 근처에서 생성된 기울기 정보를 '인접 기울기 정보'로 정의하고, 이를 축적하여 적대적 예제 생성의 초기 모멘텀(Momentum)으로 활용합니다.
2. **다중 마스크(Multiplex Mask):** 네트워크가 비판별적인 영역(Non-discriminative Regions)에 집중하도록 유도하는 '마스크(Mask)'를 사용하여 다중 경로(Multi-way) 공격 전략을 구현합니다. 이를 통해 더 풍부한 기울기 정보를 얻고, 제한된 반복 횟수 내에 충분한 섭동(Perturbation)을 달성합니다.

- **Performance Highlights**: NGI-Attack은 다양한 모델과 방어 기법에 대한 실험에서 우수한 성능을 보였습니다. 특히, 다양한 방어 모델에 대한 평균 공격 성공률이 95.8%에 달하며, 기존 방법보다 2.6%~19.3% (일반 모델) 및 2.4%~13.5% (적대적 학습 모델) 개선된 성능을 보여주었습니다. 또한, NGI-Attack은 추가적인 시간 비용 없이 기존 공격 기법에 쉽게 통합될 수 있다는 장점을 가지고 있습니다.



### Neural Architecture Search based Global-local Vision Mamba for Palm-Vein Recognition (https://arxiv.org/abs/2408.05743)
- **What's New**: 이 논문은 정맥 인식을 위한 새로운 Global-local Vision Mamba (GLVM) 네트워크를 제안합니다. GLVM은 CNN, Multi-head Mamba (MHMamba), Feature Interaction Unit (FIU) 세 가지 모듈을 통합하여 지역적 특징과 글로벌 표현을 학습합니다. 또한, 글로벌 및 지역적 검색 공간을 번갈아 가며 최적의 네트워크 아키텍처를 찾는 Global-local Alternating Neural Network Search (GLANAS) 알고리즘을 제안합니다.

- **Technical Details**: GLVM은 CNN, MHMamba, FIU 세 가지 모듈로 구성됩니다. CNN은 지역적 특징을 추출하고 MHMamba는 글로벌 의존성 표현을 포착하며, FIU는 CNN의 지역적 특징과 MHMamba의 글로벌 표현을 결합합니다. GLANAS는 네트워크 아키텍처를 자동으로 찾기 위해 진화 알고리즘을 사용하여 글로벌 및 지역적 하이퍼파라미터를 번갈아 가며 검색합니다.

- **Performance Highlights**: 실험 결과는 제안된 GLVM과 GLANAS가 세 가지 공개 손바닥 정맥 데이터베이스에서 최첨단 성능을 달성함을 보여줍니다. 특히, GLVM은 기존의 Mamba 모델보다 뛰어난 성능을 보여주며, GLANAS는 자동으로 GLVM의 아키텍처를 최적화하여 정확도를 향상시킵니다.



### A Training-Free Framework for Video License Plate Tracking and Recognition with Only One-Sho (https://arxiv.org/abs/2408.05729)
- **What's New**: OneShotLP는 비디오 기반 자동 번호판 탐지 및 인식을 위한 훈련 없는 프레임워크로, 대규모 사전 훈련된 모델의 일반화 능력을 활용하여 다양한 번호판 형식에 대한 적응성을 갖추었습니다.



### Deformable Image Registration with Multi-scale Feature Fusion from Shared Encoder, Auxiliary and Pyramid Decoders (https://arxiv.org/abs/2408.05717)
- **What's New**: 본 논문은 비지도 이미지 레지스트레이션(unsupervised image registration)을 위한 새로운 변형 가능한 피라미드 네트워크(deformable convolutional pyramid network)를 제안합니다. 이 네트워크는 이미지 쌍을 위한 추가적인 공유 보조 디코더(shared auxiliary decoder)를 추가하여 기존 피라미드 네트워크를 개선합니다. 이 디코더는 레지스트레이션 작업에 사용할 수 있도록 이미지 쌍의 다중 스케일 고수준 기능 정보(multi-scale high-level feature information)를 제공합니다. 또한, 레지스트레이션 과정에서 다중 스케일 기능 융합 블록(multi-scale feature fusion block, MSFB)을 설계하여 글로벌 및 로컬 컨텍스트에서 레지스트레이션 작업에 가장 유용한 기능을 추출합니다.



### SSL: A Self-similarity Loss for Improving Generative Image Super-resolution (https://arxiv.org/abs/2408.05713)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 본 논문에서는 기존의 GAN 및 DM 기반 Real-ISR 모델의 성능을 향상시키고, 인공물을 줄이고 더욱 사실적인 디테일을 생성하기 위해 새로운 훈련 손실 함수를 제안합니다. 이 손실 함수는 이미지 자체 유사성(Self-Similarity) 특징을 활용하여 훈련 과정을 지도하는 방식입니다.



### Contrastive masked auto-encoders based self-supervised hashing for 2D image and 3D point cloud cross-modal retrieva (https://arxiv.org/abs/2408.05711)
Comments:
          Accepted by ICME 2024

- **What's New**: 본 논문에서는 2D 이미지와 3D 포인트 클라우드 데이터 간의 크로스 모달 해싱을 위한 새로운 접근 방식인 CMAH (Contrastive Masked Autoencoders based Self-supervised Hashing)를 제안합니다. CMAH는 컨트라스티브 학습과 마스크된 오토인코더를 결합하여 이미지와 포인트 클라우드 데이터의 의미를 더 잘 이해하고, 모달 간 격차를 줄여 효과적인 해싱 코드를 생성합니다.

- **Technical Details**: CMAH는 이미지와 포인트 클라우드 데이터를 컨트라스티브 학습 방식으로 비교하여 모달 간 불변성을 확보합니다. 또한, 마스크된 오토인코더를 통해 이미지와 포인트 클라우드 데이터의 특징을 잡아내는 능력을 향상시킵니다. 이 과정에서 마스크된 데이터를 재구성하는 과제를 통해 모델은 더욱 국소적인 단서를 포착하도록 유도됩니다. 컨트라스티브 학습과 마스크된 오토인코더는 각각 전역적인 관계 유지와 국소적인 특징 포착에 탁월하여, 이를 결합함으로써 더욱 정확하고 의미 있는 해싱 코드를 생성합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 실험 결과, CMAH는 기존 방법들을 능가하는 뛰어난 크로스 모달 검색 성능을 보여주었습니다. CMAH는 이미지와 포인트 클라우드 데이터 간의 의미를 더 잘 이해하고, 모달 간 격차를 줄임으로써 차별화된 해싱 코드를 생성하여 뛰어난 검색 성능을 달성합니다.



### Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators (https://arxiv.org/abs/2408.05710)
Comments:
          ECCV 2024

- **What's New**: 본 논문은 특히 잡음 제거 확산 단계의 초기 단계에서 확산 변환기 모델의 자기 주의 메커니즘 내에서 쿼리-키 상호 작용에 상당한 중복성이 존재한다는 것을 밝혀냈습니다. 이러한 관찰에 대한 답변으로, 저희는 쿼리 및 키와 별도로 상호 작용하는 추가 중재자 토큰 집합을 통합한 새로운 확산 변환기 프레임워크를 제시합니다. 잡음 제거 생성 단계 동안 중재자 토큰의 수를 조절함으로써, 저희 모델은 정확하고 모호하지 않은 단계로 잡음 제거 프로세스를 시작하고 점차적으로 세부 정보가 풍부한 단계로 전환합니다. 동시에, 중재자 토큰을 통합하면 주의 모듈의 복잡성이 선형 규모로 단순화되어 글로벌 주의 프로세스의 효율성이 향상됩니다. 또한, 저희는 생성에 필요한 계산 FLOP를 더욱 줄이는 동시에 다양한 추론 예산 제약 내에서 고품질 이미지 생성을 용이하게 하는 시간 단계 동적 중재자 토큰 조정 메커니즘을 제안합니다. 광범위한 실험은 제안된 방법이 생성된 이미지 품질을 개선하는 동시에 확산 변환기의 추론 비용을 줄일 수 있음을 보여줍니다. 최근 연구인 SiT와 통합하면, 저희 방법은 2.01의 최첨단 FID 점수를 달성합니다.

- **Technical Details**: 저희는 쿼리 및 키와 상호 작용하는 추가 중재자 토큰 집합을 도입하여 확산 변환기 모델의 자기 주의 메커니즘의 중복성 문제를 해결했습니다. 중재자 토큰은 먼저 소프트맥스 주의를 사용하여 키에서 정보를 집계하여 압축된 표현을 형성합니다. 그런 다음, 압축된 정보는 또 다른 소프트맥스 주의를 통해 쿼리로 전파되어 최종 출력으로 사용됩니다. 또한, 저희는 시간 단계 간의 중복성 변화를 이용하여 각 시간 단계에서 중재자 토큰의 수를 조정하는 새로운 동적 전략을 도입했습니다. 구체적으로, 중복성이 뚜렷한 초기 단계에서는 중재자 토큰 수를 줄여 유사한 정보 집계를 효과적으로 줄입니다. 후반 단계에서 중복성이 점차 줄어들면, 더욱 상세하고 다양한 기능을 생성하기 위해 중재자 토큰 수를 동적으로 늘립니다.

- **Performance Highlights**: 광범위한 실험 결과는 저희 접근 방식이 우수한 생성 품질(낮은 FID로 표시됨)을 달성하고 생성 중에 계산 복잡성(FLOP로 측정됨)을 줄인다는 것을 보여줍니다. SiT-XL/2 모델과 결합하면 저희 방법은 최첨단 FID 점수를 달성합니다.



### Decoder Pre-Training with only Text for Scene Text Recognition (https://arxiv.org/abs/2408.05706)
Comments:
          Accepted by ACM MM 2024

- **What's New**: This paper introduces a novel STR pre-training method, Decoder Pre-training with only text for STR (DPTR), that leverages text embeddings from CLIP to derive potential representations of real images, addressing the domain gap between synthetic and real datasets. DPTR utilizes the CLIP text encoder to generate pseudo visual embeddings for decoder pre-training. To enhance feature diversity and prevent overfitting, the paper proposes an Offline Random Perturbation (ORP) strategy, adding natural image features as noise to the text embeddings. Furthermore, a Feature Merge Unit (FMU) is introduced to guide the model's attention towards the character foreground, enabling more efficient and accurate decoding.



### MacFormer: Semantic Segmentation with Fine Object Boundaries (https://arxiv.org/abs/2408.05699)
Comments:
          13 pages, 7 figures, submitted to TIP

- **What's New**: 본 논문에서는  **MacFormer**라는 새로운 세맨틱 세그멘테이션 아키텍처를 소개합니다. MacFormer는 **Mutual Agent Cross-Attention (MACA)** 메커니즘과 **Frequency Enhancement Module (FEM)**의 두 가지 핵심 구성 요소를 특징으로 합니다. MACA는 학습 가능한 에이전트 토큰을 사용하여 인코더와 디코더 레이어 간의 양방향 특징 통합을 효과적으로 촉진합니다. 이를 통해 디코딩 과정에서 초깃단계 경계와 같은 저수준 특징을 더 잘 보존할 수 있습니다. FEM은 디코더에서 고주파 및 저주파 구성 요소를 활용하여 주파수 영역에서 특징을 강화하여 계산 복잡도 증가를 최소화하면서 객체 경계를 향상시킵니다. MacFormer는 다양한 네트워크 아키텍처와 호환 가능하며, ADE20K 및 Cityscapes 벤치마크 데이터 세트에서 다양한 계산 제약 조건 하에 정확성과 효율성 측면에서 기존 방법을 능가합니다.



### A Novel Momentum-Based Deep Learning Techniques for Medical Image Classification and Segmentation (https://arxiv.org/abs/2408.05692)
Comments:
          8 pages

- **What's New**: 이 연구는 의료 영상 분석에서 학습 역학을 향상시키기 위해 잔여 블록 (Residual Block) 내에 모멘텀을 통합하는 새로운 기술을 소개합니다. 이 연구는 컴퓨터 지원 진단 및 개입 계획을 위해 CT 및 MRI 스캔에서 다양한 장기를 분할하고 질병을 분류하는 데 중점을 둡니다. 제안된 모멘텀 기반 접근 방식은 폐, 간 및 결장 데이터를 분할하고 복부 골반 CT 및 MRI 스캔을 분류하는 두 가지 별개의 작업에서 뛰어난 성능을 보여 주었습니다. 특히 폐 분할 데이터 세트에서 제안된 방법은 Dice Score에서 5.72% 증가, mIoU (Mean Intersection over Union)에서 5.04% 향상, 재현율에서 8.02% 향상 및 정밀도에서 4.42% 향상을 포함하여 TransNetR 모델보다 상당한 개선을 보였습니다. 이러한 결과는 모멘텀 통합이 의료 영상 분야에서 획기적인 발전을 나타내는 분할 및 분류 작업 모두에서 최첨단 성능을 달성한다는 것을 시사합니다.



### Single Image Dehazing Using Scene Depth Ordering (https://arxiv.org/abs/2408.05683)
Comments:
          14 pages, 15 figures

- **What's New**: This paper proposes a depth order guided single image dehazing method that preserves the original depth order in hazy images, resulting in dehazed images with similar depth perception.

- **Technical Details**: The method consists of two main components:
1. **Depth Order Extraction:** A strategy to extract depth order in hazy images without estimating specific depth values. This provides a reference for depth perception in hazy weather.
2. **Depth Order Embedded Transformation Model:** A transformation model for transmission estimation that incorporates the extracted depth order, ensuring an unchanged depth order in the dehazed results. This helps achieve similar depth perception in the dehazed images.

- **Performance Highlights**: The proposed method effectively removes haze while preserving the original depth order, resulting in higher quality dehazed images with better structure and color recovery. It also outperforms existing dehazing methods in terms of computational efficiency due to the absence of pre-processing and iteration steps.



### PS-TTL: Prototype-based Soft-labels and Test-Time Learning for Few-shot Object Detection (https://arxiv.org/abs/2408.05674)
Comments:
          Accepted to ACM MM 2024

- **What's New**: 이 논문은 **Few-Shot Object Detection (FSOD)** 에서 새로운 프레임워크 **Prototype-based Soft-labels and Test-Time Learning (PS-TTL)** 를 제안합니다. 이 프레임워크는 제한된 새 클래스 데이터를 보완하기 위해 **Test-Time Learning (TTL)** 모듈과 **Prototype-based Soft-labels (PS)** 전략을 활용합니다.

- **Technical Details**: **TTL 모듈**은 **mean-teacher network** 를 이용하여 테스트 데이터에서 새 인스턴스를 발견하고 self-training 을 수행하여 모델을 개선합니다. **PS 전략**은 저품질 pseudo-labels 의 잠재력을 끌어내기 위해 클래스 프로토타입과의 유사성을 기반으로 **soft-labels** 를 생성합니다.

- **Performance Highlights**: VOC 및 COCO 벤치마크에서 다양한 few-shot 설정에서 최첨단 성능을 달성하여 새 객체 감지의 효율성을 입증했습니다.



### StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Mod (https://arxiv.org/abs/2408.05669)
- **What's New**: 본 논문은 **AI-Generated Content Stealth (AIGC-S)** 라는 새로운 과제를 제시하며, 이는 인공지능이 생성한 이미지가 탐지 기술과 사람의 눈 모두를 속일 수 있도록 만드는 것을 목표로 합니다. 기존의 **adversarial attack** 방법들은 시각적으로 눈에 띄는 노이즈를 발생시키거나, **transferability**가 떨어지거나, **spectral difference**를 해결하지 못하는 단점을 가지고 있습니다. 이를 해결하기 위해 본 논문은 **StealthDiffusion**이라는 새로운 프레임워크를 제안합니다.



### Performance Evaluation of YOLOv8 Model Configurations, for Instance Segmentation of Strawberry Fruit Development Stages in an Open Field Environmen (https://arxiv.org/abs/2408.05661)
Comments:
          15 page, 18 figures

- **What's New**: 본 연구는 현장 환경에서 딸기 성숙 단계를 익은 것과 익지 않은 것으로 구분하는 인스턴스 분할(instance segmentation)을 위한 YOLOv8 모델 구성의 성능을 평가했습니다. 다른 YOLOv8 구성과 비교하여 YOLOv8n 모델이 평균 평균 정밀도(mAP) 80.9%로 뛰어난 분할 정확도를 보였습니다.



### Advancing Pavement Distress Detection in Developing Countries: A Novel Deep Learning Approach with Locally-Collected Datasets (https://arxiv.org/abs/2408.05649)
- **What's New**: 개발도상국의 도로 인프라 유지 보수는 자원 제약과 다양한 환경적 요인으로 인해 특별한 과제에 직면합니다. 이 연구는 이러한 지역에서 효율적이고 정확하며 지역 특성에 맞는 포장 노면 결함 감지 방법의 중요성을 다룹니다. 저희는 YOLO(You Only Look Once) 객체 감지 모델과 CBAM(Convolutional Block Attention Module)을 결합한 혁신적인 딥 러닝 기법을 제시하여 다양한 포장 노면 결함 유형을 동시에 감지하고 분류합니다. 이 모델은 0.46에서 0.93의 신뢰도 점수로 포트홀, 세로 균열, 악어 균열, 마모 등을 감지하고 분류하는 데 탁월한 성능을 보여줍니다. 복잡한 시나리오에서 오분류가 발생하지만 이는 개발도상국의 포장 노면 평가에서 나타나는 고유한 과제에 대한 통찰력을 제공합니다. 또한 저희는 이미지 및 비디오에서 실시간 결함 감지를 위한 웹 기반 애플리케이션을 개발했습니다. 이 연구는 자동화된 포장 노면 결함 감지 기술을 발전시키고 개발도상국에 맞춤형 솔루션을 제공하여 도로 안전을 향상시키고 유지 보수 전략을 최적화하며 지속 가능한 교통 인프라 개발에 기여할 수 있습니다.



### Visual SLAM with 3D Gaussian Primitives and Depth Priors Enabling Novel View Synthesis (https://arxiv.org/abs/2408.05635)
- **What's New**: 본 논문은 실시간 RGB-D SLAM 시스템을 제안하며, 3D 시나리오 표현과 자세 추정을 위해 3D Gaussian Splatting이라는 새로운 뷰 합성 기법을 통합합니다. 또한, 3D Gaussian Splatting을 통해 메쉬 재구성을 가능하게 하여 명시적인 3D 재구성을 수행합니다. 이 시스템은 정확한 카메라 자세 추정을 위해 역 최적화를 활용하여 회전과 이동을 분리하는 전략을 사용합니다. 이 시스템은 기존의 3D Gaussian 맵을 기반으로 광도 손실, 깊이 기하 손실 및 가시성 손실을 결합하여 최소화하는 방식으로 카메라 파라미터를 업데이트합니다. 3D Gaussian Splatting은 다중 뷰 불일치로 인해 표면을 정확하게 표현하는 데 어려움을 겪을 수 있으므로 카메라 자세 추정 및 장면 재구성 정확도가 감소될 수 있습니다. 이러한 문제를 해결하기 위해 본 논문은 깊이 사전 정보를 추가적인 규제로 사용하여 기하학적 제약 조건을 강화하고 자세 추정 및 3D 재구성 정확도를 향상시킵니다.



### PRTGaussian: Efficient Relighting Using 3D Gaussians with Precomputed Radiance Transfer (https://arxiv.org/abs/2408.05631)
- **What's New**: PRTGaussian은 3D 가우시안과 사전 계산된 복사 전달(Precomputed Radiance Transfer, PRT)을 결합하여 실시간 재조명 가능한 새로운 뷰 합성 방법을 제시합니다. 이 방법은 멀티뷰 OLAT 데이터에 재조명 가능한 가우시안을 적합시켜 실시간, 자유 시점 재조명을 가능하게 합니다. 고차 구면 조화(spherical harmonics)를 기반으로 복사 전달을 추정하여 자세한 재조명 효과를 포착하는 동시에 계산 효율성을 유지합니다.



### UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling (https://arxiv.org/abs/2408.05618)
- **What's New**: UrFound, a universal retinal foundation model that can process both Color Fundus Photography (CFP) and Optical Coherence Tomography (OCT) images while incorporating domain knowledge from expert annotations, is introduced. This model addresses the limitations of existing retinal foundation models, which are typically restricted to a single modality and may not fully utilize expert annotations.



### Non-Negative Reduced Biquaternion Matrix Factorization with Applications in Color Face Recognition (https://arxiv.org/abs/2408.05582)
- **What's New**: 본 논문은 색상 얼굴 인식 문제를 해결하기 위해 **비음수 RB 행렬 분해 (NRBMF)** 모델을 제안합니다. 이 모델은 전통적인 쿼터니언의 곱셈 특성 때문에 어려움을 겪는 **비음수 쿼터니언 행렬 분해** 모델의 문제를 해결하기 위해 **비음수 RB 행렬**의 개념을 도입합니다.



### Camera Perspective Transformation to Bird's Eye View via Spatial Transformer Model for Road Intersection Monitoring (https://arxiv.org/abs/2408.05577)
- **What's New**: 본 논문은 단일 카메라 관점에서 도로 교차로를 조류시점(Bird's Eye View, BEV)로 변환하는 새로운 딥 러닝 모델을 제안합니다. 이 모델은 실제 교차로 모니터링 및 제어 시스템에 BEV 시뮬레이션 기반 모델을 적용하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 제안된 모델은 Spatial-Transformer Double Decoder-UNet (SDD-UNet)이라고 하며, 기존 UNet 아키텍처를 기반으로 하지만 두 개의 디코더 브랜치를 포함합니다. 첫 번째 디코더는 UNet의 기존 구조를 따르며 두 번째 디코더는 Spatial Transformer Network를 통합합니다. 이를 통해 변환된 이미지 왜곡을 제거하고 차량의 정확한 위치를 추정할 수 있습니다.

- **Performance Highlights**: SDD-UNet 모델은 95% 이상의 평균 다이스 유사 계수(DSC)를 달성하여 기존 UNet 모델보다 40% 향상되었습니다. 평균 절대 오차(MAE)는 0.102이며, 예측된 마스크의 중심은 평균 0.14미터 이동하여 높은 정확도를 나타냅니다.



### What Matters in Autonomous Driving Anomaly Detection: A Weakly Supervised Horizon (https://arxiv.org/abs/2408.05562)
- **What's New**: 이 논문은 자율 주행 환경에서의 weakly-supervised 비디오 이상 감지(VAD) 연구를 진전시키기 위해 DoTA 데이터셋을 재구성하고 최신 weakly-supervised VAD 방법들을 움직이는 카메라 시나리오에 적용하여 검증합니다. 또한, 최첨단 방법에 대한 세부적인 분석을 통해 이상 감지 성능을 향상시킬 수 있는 수정 사항을 제시하고, 이를 위해 ‘특징 변환 블록(Feature Transformation Block, FTB)’을 제안합니다.

- **Technical Details**: 이 논문에서는 CLIP(Contrastive Language-Image Pre-training)을 특징 추출 백본으로 활용하고,  FTB를 도입하여 기존 weakly-supervised VAD 방법들의 성능을 개선합니다. FTB는 temporal saliency를 강화하여 temporal 모델링과 feature magnitude supervision을 향상시키는 역할을 합니다. 또한, DoTA 데이터셋을 weakly-supervised 학습에 적합하도록 재구성하여 WS-DoTA 데이터셋을 제시합니다. 이 데이터셋은 자율 주행 환경에서의 weakly-supervised VAD 연구를 위한 새로운 기반을 마련할 것으로 예상됩니다.

- **Performance Highlights**: 실험 결과, 제안된 FTB가 최첨단 weakly-supervised VAD 방법들의 성능을 WS-DoTA 데이터셋에서 크게 향상시키는 것을 보여줍니다. 이는 FTB가 weakly-supervised 환경에서의 이상 감지 성능을 향상시키는 데 효과적임을 입증합니다.



### Object Re-identification via Spatial-temporal Fusion Networks and Causal Identity Matching (https://arxiv.org/abs/2408.05558)
- **What's New**: 본 논문은 대규모 카메라 네트워크에서의 객체 재식별(ReID) 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 공간-시간 융합 네트워크(FusionNet)와 인과적 ID 매칭(CIM)을 활용하여 유사한 외관으로 인해 발생하는 재식별 성능 저하 문제를 해결하고 실제 환경에 적용 가능하도록 설계되었습니다.



### Evolutionary Neural Architecture Search for 3D Point Cloud Analysis (https://arxiv.org/abs/2408.05556)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 3차원 포인트 클라우드 데이터 분석에 특화된 효율적인 신경망 아키텍처를 자동으로 설계하는 새로운 진화형 신경망 아키텍처 검색(NAS) 프레임워크, SHSADE-PIDS를 제안합니다. SHSADE-PIDS는 이산 신경망 아키텍처를 연속 공간으로 인코딩하고 연속 공간에서 검색을 수행하여 효율적인 포인트 클라우드 신경망 아키텍처를 찾습니다. 이 방법은 기존 NAS 기법보다 높은 정확도로 더 효율적인 아키텍처를 발견합니다.



### PixelFade: Privacy-preserving Person Re-identification with Noise-guided Progressive Replacemen (https://arxiv.org/abs/2408.05543)
Comments:
          accepted by ACMMM24

- **What's New**: 이 논문은 보행자 이미지를 노이즈 이미지로 변환하여 복구 공격에 대한 저항성을 높이는 새로운 개인 정보 보호 보행자 재 식별 (PPPR) 방법인 PixelFade를 제안합니다.

- **Technical Details**: PixelFade는 보행자 이미지를 거의 정규 분포된 노이즈 이미지로 근사화하는 Noise-guided Objective Function과 사전 훈련된 Re-ID 모델의 특징 공간에서 보호된 이미지와 원본 이미지 간의 특징 거리에 제약을 두어 보호된 이미지의 유용성을 유지합니다. 이 비볼록 최적화 문제를 해결하기 위해 PixelFade는 제약 연산과 부분 교체 연산을 번갈아 수행하는 Progressive Pixel Fading이라는 휴리스틱 최적화 전략을 제안합니다. 제약 연산은 TypeI 공격을 활용하여 기울기를 도출하고, 부분 교체 연산은 산란된 픽셀의 일부만 노이즈로 대체합니다.

- **Performance Highlights**: PixelFade는 세 가지 널리 사용되는 Re-ID 데이터 세트에서 이전 PPPR 방법보다 복구 공격에 대한 저항성과 Re-ID 성능 측면에서 더 나은 결과를 얻었습니다. 또한 PixelFade는 다양한 Re-ID 네트워크 아키텍처와 다양한 Re-ID 시나리오(예: 텍스트-이미지 Re-ID, 가시적 적외선 Re-ID)에 쉽게 적용할 수 있어 확장성과 적용 가능성이 높습니다.



### Radiance Field Learners As UAV First-Person Viewers (https://arxiv.org/abs/2408.05533)
Comments:
          Accepted to ECCV 2024

- **What's New**: FPV-NeRF는 UAV에서 촬영한 비디오에서 First-Person-View(FPV) 영상을 합성하기 위한 새로운 프레임워크입니다. FPV는 복잡한 건물 구조를 탐색하는 흥미로운 방법을 제공하지만, 기존 NeRF(Neural Radiance Field) 방법은 UAV 비디오의 제한된 시야와 공간 규모 변화로 인해 다양한 규모에 걸쳐 충분한 세부 렌더링을 생성하는 데 어려움을 겪습니다. FPV-NeRF는 이러한 문제를 해결하기 위해 세 가지 주요 측면을 도입합니다: (1) 시간 일관성, (2) 글로벌 구조, (3) 로컬 세분화.



### CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM (https://arxiv.org/abs/2408.05526)
- **What's New**: This research introduces CryoBench, a suite of datasets and metrics designed to benchmark heterogeneous reconstruction in cryo-EM, a powerful technique for determining high-resolution 3D biomolecular structures.

- **Technical Details**: CryoBench comprises five datasets representing various sources of heterogeneity: conformational heterogeneity from antibody complexes, molecular dynamics simulations, and compositional heterogeneity from ribosome assembly states and common cellular complexes. CryoBench leverages synthetic data generation to provide ground truth structures and imaging parameters for quantitative evaluation.

- **Performance Highlights**: The research performs a comprehensive analysis of existing heterogeneous reconstruction tools, both neural and non-neural methods, to assess their sensitivity to noise. It proposes novel metrics for quantitative comparison, providing a foundational resource for analyzing existing methods and advancing algorithmic development in both the cryo-EM and machine learning communities.



### Long working distance portable smartphone microscopy for metallic mesh defect detection (https://arxiv.org/abs/2408.05518)
- **What's New**: 본 연구는 산업 현장에서의 실시간, 현장 검사를 위한 장거리 반사형 스마트폰 현미경(LD-RSM)을 제안합니다. LD-RSM은 기존 스마트폰 현미경의 단거리 작동 거리 및 산업 현장 검사에 적합하지 않은 투과 이미징의 한계를 극복하기 위해 설계되었습니다. 특히, 빔 스플리터(beam splitter)를 활용하여 광원과 이미징 시스템을 시료의 동일한 측면에 배치하여 반사 이미징을 구현하는 4f 광학 이미징 시스템을 구축했습니다.

- **Technical Details**: LD-RSM은 4.92µm의 광학 해상도와 최대 22.23mm의 작동 거리를 달성합니다. 또한, 본 연구는 다중 우선 순위 가중 로버스트 주성분 분석(DW-RPCA)을 도입하여 결함 감지를 수행합니다. DW-RPCA는 스펙트럼 필터 융합 및 허프 변환을 활용하여 다양한 결함 유형을 모델링하여 결함 식별의 정확성과 효율성을 향상시킵니다. 최적화된 임계값 분할 알고리즘과 결합된 DW-RPCA 방법은 84.8%의 픽셀 수준 정확도를 달성합니다.

- **Performance Highlights**: LD-RSM은 금속 메쉬의 결함 검사에서 높은 정확성과 효율성을 보여줍니다. 특히, DW-RPCA 방법은 기존의 스펙트럼 방법이나 통계적 방법에 비해 정확성과 효율성을 크게 향상시켜 산업 현장에서의 실시간 결함 검사에 적합합니다.



### PointMT: Efficient Point Cloud Analysis with Hybrid MLP-Transformer Architectur (https://arxiv.org/abs/2408.05508)
- **What's New**: 본 논문에서는 효율적인 포인트 클라우드 분석 아키텍처인 **PointMT**를 제안합니다. PointMT는 트랜스포머의 계산 복잡성을 줄이고 특징 표현 능력을 향상시켜, 실시간 처리 및 모바일 기기 배포를 가능하게 하는 것이 목표입니다. PointMT는 기존의 트랜스포머의 한계점을 해결하기 위해 세 가지 주요 혁신을 도입합니다.



### Disentangled Noisy Correspondence Learning (https://arxiv.org/abs/2408.05503)
- **What's New**: DisNCL (Disentanglement in Noisy Correspondence Learning), 새로운 정보 이론적 프레임워크를 소개하여, 훈련 데이터의 노이즈 대응 학습에서 특징 해 disentanglement (분리) 효율성을 개선합니다. 기존의 방법들과 달리, DisNCL은 모달 간 불변 정보 (MII)와 모달 고유 정보 (MEI)를 분리하는 방식으로, 노이즈가 포함된 데이터에서도 정확한 유사도 예측을 가능하게 합니다. 또한, 부드러운 매칭 타겟을 도입하여, 다중 모달 데이터의 다 대 다 관계를 모델링하여, 노이즈에 강인하고 정확한 모달 간 정렬을 수행합니다.



### GEM: Context-Aware Gaze EstiMation with Visual Search Behavior Matching for Chest Radiograph (https://arxiv.org/abs/2408.05502)
Comments:
          9 figures

- **What's New**: 본 논문은 의료 영상 해석 시 의사의 시선 추적 데이터를 활용하여 시각적 주의 패턴과 정보 처리 전략을 파악하는 새로운 접근 방식을 제시합니다. 특히, 의료 보고서와 영상 간의 맥락적 연결을 활용하여 정확한 시선 추정을 수행하는 '맥락 인식 시선 추정(Context-Aware Gaze EstiMation, GEM) 네트워크'를 제안합니다. GEM 네트워크는 다양한 모듈을 통해 의사의 시각적 탐색 행동 패턴을 정확하게 모사하고 의료 영상 해석 과정에 대한 통찰력을 제공합니다.



### ZePo: Zero-Shot Portrait Stylization with Faster Sampling (https://arxiv.org/abs/2408.05492)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 본 논문은 기존 텍스트 투 이미지 생성 모델 기반 초상화 스타일 변환 방식의 한계점을 극복하기 위해 인버전 없이 4단계 샘플링만으로 콘텐츠와 스타일 특징을 융합하는 새로운 초상화 스타일 변환 프레임워크인 ZePo를 제시합니다. 특히, Latent Consistency Model (LCM)을 활용하여 이미지 생성 속도를 높이고 DDIM 인버전 없이 노이즈 이미지에서 일관성 특징을 직접 추출하는 방법을 제시합니다. 또한, 스타일 강화 어텐션 제어 (SEAC) 기법을 도입하여 콘텐츠와 스타일 이미지의 일관성 특징을 효율적으로 융합하는 방법을 제시하여 어텐션 계산 속도를 향상시킵니다.



### ReToMe-VA: Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack (https://arxiv.org/abs/2408.05479)
- **What's New**: This paper proposes **ReToMe-VA** (Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack), a novel framework for generating imperceptible adversarial video clips with high transferability, addressing the limitations of existing diffusion-based unrestricted attacks on images.

- **Technical Details**: **ReToMe-VA** utilizes a **Timestep-wise Adversarial Latent Optimization (TALO)** strategy, optimizing perturbations in the latent space of diffusion models at each denoising step. This leads to more accurate updates and reduces memory consumption compared to previous methods. To achieve temporal imperceptibility, **ReToMe-VA** introduces a **Recursive Token Merging (ReToMe)** mechanism that aligns and merges tokens across frames in the self-attention module, resulting in temporally consistent adversarial videos. This also facilitates inter-frame interactions, leading to more robust and diverse gradients, and boosting adversarial transferability.

- **Performance Highlights**: Extensive experiments on various video recognition models demonstrate that **ReToMe-VA** significantly surpasses state-of-the-art attacks in adversarial transferability, achieving an average improvement of over 14.16%. The proposed approach proves its effectiveness by demonstrating strong performance even against defense mechanisms.



### Scene123: One Prompt to 3D Scene Generation via Video-Assisted and Consistency-Enhanced MAE (https://arxiv.org/abs/2408.05477)
Comments:
          arXiv admin note: text overlap with arXiv:2305.11588 by other authors

- **What's New**: Scene123 is a novel 3D scene generation model that can create realistic and consistent large-scale scenes from a single input (image or text). It leverages recent advancements in video generation models and implicit neural representations (NeRF) to address the challenge of ensuring consistency across extrapolated views.

- **Technical Details**: Scene123 uses a combination of techniques: 
1. **Masked Autoencoders (MAE)**: A MAE model is used to fill in unseen areas in newly generated views, ensuring consistency across views. 
2. **Implicit Neural Fields (NeRF)**: NeRF is employed to enhance the geometric consistency of the generated views. 
3. **Video Generation Model:** A video generation model is used to generate high-quality scene videos, enriching the scene with detail and texture. 
4. **Generative Adversarial Network (GAN)**: A GAN is used to further enhance the detail and texture fidelity of the generated views.

- **Performance Highlights**: Extensive experiments show that Scene123 significantly outperforms existing state-of-the-art methods in terms of realism, consistency, and texture fidelity. The generated scenes exhibit high-quality 3D representation, showcasing the effectiveness of Scene123 in both real and synthetically styled scenes.



### Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network (https://arxiv.org/abs/2408.05475)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 새로운 교차 뷰 이미지 지리 위치 확인 방법인 파노라마-BEV 공동 검색 네트워크를 제안합니다. 이 방법은 스트리트 뷰 파노라마 이미지를 BEV (Bird's Eye View) 뷰로 변환하여 스트리트 파노라마와 위성 이미지 간의 차이를 줄입니다. 이를 위해 BEV 및 위성 이미지 검색 분기를 도입하여 스트리트 뷰 파노라마 이미지와 위성 이미지를 공동으로 검색합니다. 또한, 스트리트 뷰 검색 분기를 유지하여 BEV 표현의 제한된 인식 범위 문제를 해결합니다. 이 네트워크는 스트리트 뷰 캡처 위치 주변의 글로벌 레이아웃과 로컬 세부 정보를 포괄적으로 인식할 수 있도록 합니다. 또한, 이 논문은 실제 시나리오에 가까운 글로벌 교차 뷰 데이터 세트인 CVGlobal을 소개합니다. 이 데이터 세트는 더 현실적인 설정을 채택하여 스트리트 뷰 방향이 위성 이미지와 정렬되지 않습니다. CVGlobal은 또한 교차 지역, 교차 시간 및 스트리트 뷰에서 지도 검색 테스트를 포함하여 알고리즘 성능을 포괄적으로 평가할 수 있도록 합니다. 제안된 방법은 CVUSA, CVACT, VIGOR 및 새로 도입된 CVGlobal과 같은 일반적인 교차 뷰 데이터 세트에서 다중 테스트에서 뛰어난 성능을 보이며 기존 최첨단 방법을 능가합니다.



### Multimodal generative semantic communication based on latent diffusion mod (https://arxiv.org/abs/2408.05455)
- **What's New**: 이 논문은 긴급 상황에서 시각 및 적외선 데이터를 효과적으로 전송하고 재구성하기 위해 다중 모드 생성적 의미 통신 프레임워크인 mm-GESCO를 소개합니다. mm-GESCO는 퓨전된 의미 분할 맵을 생성하여 압축하고, 수신기에서 이 맵을 사용하여 원래 다중 모드 이미지를 재구성합니다. 또한, 대조 학습을 기반으로 한 잠재 확산 모델을 사용하여 잠재 공간 내에서 다양한 모드 데이터를 정렬하여, mm-GESCO가 입력에서 제공되는 어떤 모드의 잠재 특성도 재구성할 수 있도록 합니다.



### EV-MGDispNet: Motion-Guided Event-Based Stereo Disparity Estimation Network with Left-Right Consistency (https://arxiv.org/abs/2408.05452)
- **What's New**: 본 논문에서는 EV-MGDispNet이라는 새로운 이벤트 기반 입체 불일치 추정 (stereo disparity estimation) 방법을 제안합니다. 이 방법은 이벤트 프레임과 모션 신뢰도 맵(motion confidence maps)을 융합하여 명확한 이벤트 표현을 생성하는 엣지 인식 집계(EAA) 모듈을 도입합니다. 또한, 모션 신뢰도 맵을 이용하여 변형 가능한 트랜스포머 인코더(deformable transformer encoder)를 활용하여 더 정확한 엣지를 갖는 특징 맵을 생성하는 모션 유도 주의(MGA) 모듈을 제안합니다. 마지막으로, 입체 이벤트 표현의 좌우 일관성을 향상시키기 위해 센서스 좌우 일관성 손실 함수(census left-right consistency loss function)를 추가합니다.



### Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness (https://arxiv.org/abs/2408.05446)
Comments:
          34 pages, 25 figures, appendix

- **What's New**: 본 논문에서는 딥 뉴럴 네트워크의 강건성, 신뢰성 및 정렬을 위협하는 적대적 예제 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 이는 다중 해상도 입력 표현(multi-resolution input representations)과 중간 계층 예측(intermediate layer predictions)의 동적 자기 앙상블(dynamic self-ensembling)을 활용하여 고품질 표현을 얻는 방식입니다. 중간 계층 예측은 전체 분류기를 속이기 위해 고안된 적대적 공격에 대한 내재적인 강건성을 보이고 있으며, 이러한 예측들을 효과적으로 통합하기 위해 'CrossMax'라는 새로운 Vickrey 옥션 기반 강건한 집계 메커니즘을 제안합니다. 다중 해상도 입력과 강건한 앙상블을 결합하여, 추가적인 적대적 훈련이나 데이터 없이 CIFAR-10 및 CIFAR-100 데이터 세트에서 상당한 적대적 강건성을 달성했습니다. RobustBench AutoAttack 슈트(L∞=8/255)에서 미세 조정된 ImageNet 사전 훈련된 ResNet152를 사용하여 CIFAR-10에서 약 72%, CIFAR-100에서 약 48%의 적대적 정확도를 달성했습니다. 이는 CIFAR-10의 상위 3개 모델과 비슷한 결과이며, CIFAR-100에 대한 최신 전용 접근 방식에 비해 +5% 향상된 수치입니다. 간단한 적대적 훈련을 추가하면 CIFAR-10에서 약 78%, CIFAR-100에서 약 51%를 달성하여 각각 5% 및 9%의 성능 향상을 보이며, 특히 더 어려운 데이터 세트에서 더 큰 이점을 얻을 수 있습니다. 이 논문에서는 광범위한 실험을 통해 이러한 접근 방식을 검증하고 적대적 강건성과 딥 표현의 계층적 특성 간의 상호 작용에 대한 통찰력을 제공합니다. 또한 간단한 경사 기반 공격을 통해 모델에 대한 해석 가능한 이미지 변경과 함께 목표 클래스에 대한 인간이 해석할 수 있는 이미지를 생성하고, 다중 해상도 사전을 사용하여 사전 훈련된 분류기와 CLIP 모델을 제어 가능한 이미지 생성기로 변환하여 대규모 비전 언어 모델에 대한 전이 가능한 공격을 개발합니다.



### Content-decoupled Contrastive Learning-based Implicit Degradation Modeling for Blind Image Super-Resolution (https://arxiv.org/abs/2408.05440)
- **What's New**: 본 논문은 새로운 **컨텐츠 분리 대조 학습 기반 블라인드 이미지 슈퍼 레졸루션 (CdCL)** 프레임워크를 제안합니다. 이 프레임워크는 **음성 없는 대조 학습 (negative-free contrastive learning)** 기술을 도입하여 암묵적인 **디그레이데이션 표현 (degradation representation)**을 모델링합니다. 이는 **순환적 시프트 샘플링 (cyclic shift sampling)** 전략을 통해 컨텐츠 특징과 디그레이데이션 특징을 분리함으로써, 학습된 암묵적 디그레이데이션 공간의 순도와 차별성을 향상시킵니다. 또한, **디테일 인식 암묵적 디그레이데이션 적응 모듈 (detail-aware implicit degradation adaption module)**을 설계하여, 채널 및 공간적 관점에서 디그레이데이션 정보를 LR 이미지에 적용합니다. 이 모듈은 낮은 복잡성으로 효율성과 효과를 높입니다. 



### A Methodological and Structural Review of Hand Gesture Recognition Across Diverse Data Modalities (https://arxiv.org/abs/2408.05436)
- **What's New**: 이 논문은 2014년부터 2024년까지 수행된 손 제스처 인식(HGR) 기술과 데이터 모달리티에 대한 포괄적인 검토를 제공합니다. 센서 기술과 컴퓨터 비전 분야의 발전을 탐구하며, RGB, Skeleton, Depth, Audio, EMG, EEG 및 다중 모달리티 접근 방식을 포함한 다양한 모달리티를 사용한 성과를 강조하고 추가 연구가 필요한 분야를 파악합니다. 이 논문은 데이터 수집, 데이터 설정 및 제스처 표현에 중점을 두어 저명한 데이터베이스에서 200개 이상의 논문을 검토했습니다. 이 검토는 HGR 시스템의 효율성을 인식 정확도를 통해 평가하고 연속적인 제스처 인식에 대한 연구 격차를 식별하여 향상된 비전 기반 제스처 시스템의 필요성을 강조합니다. 이 분야는 수작업 기능과 딥 러닝(DL) 기술의 발전을 포함하여 꾸준한 연구 진전을 보였습니다. 또한 이 논문은 HGR 방법과 다중 모달리티 접근 방식 분야에서 유망한 개발을 보고합니다. 이 설문 조사가 다양한 데이터 모달리티 기반 HGR 연구에 대한 잠재적인 지침이 되기를 바랍니다.



### SAM-FNet: SAM-Guided Fusion Network for Laryngo-Pharyngeal Tumor Detection (https://arxiv.org/abs/2408.05426)
- **What's New**: This paper proposes a novel dual-branch network, SAM-FNet, for laryngo-pharyngeal tumor detection, aiming to improve accuracy by integrating global and local (lesion) feature extraction.

- **Technical Details**: The SAM-FNet leverages the Segment Anything Model (SAM) for accurate lesion segmentation, introducing a GAN-like feature optimization (GFO) module to capture discriminative features between global and local branches, enhancing fusion feature complementarity.

- **Performance Highlights**: Extensive experiments on two datasets (FAHSYSU and SAHSYSU) show that SAM-FNet achieves competitive results, outperforming state-of-the-art methods in laryngo-pharyngeal tumor detection.

- **Datasets**: Two datasets were collected from the First Affiliated Hospital (FAHSYSU) and the Sixth Affiliated Hospital (SAHSYSU) of Sun Yat-sen University. FAHSYSU was used for training, while SAHSYSU was used for external evaluation.



### EPAM-Net: An Efficient Pose-driven Attention-guided Multimodal Network for Video Action Recognition (https://arxiv.org/abs/2408.05421)
- **What's New**: 본 연구에서는 비디오에서 행동 인식을 위한 효율적인 자세 기반 주의 유도 다중 모달 네트워크(EPAM-Net)을 제시합니다. EPAM-Net은 RGB 및 자세 스트림 모두에 X3D 네트워크를 적용하여 RGB 비디오 및 해당 골격 시퀀스에서 공간-시간적 특징을 포착합니다. 골격 특징은 시각적 네트워크 스트림이 공간-시간 주의 블록을 사용하여 핵심 프레임 및 해당 두드러진 공간 영역에 집중하는 데 도움이 됩니다. 마지막으로 제안된 네트워크의 두 스트림의 점수가 최종 분류를 위해 융합됩니다.



### High-fidelity and Lip-synced Talking Face Synthesis via Landmark-based Diffusion Mod (https://arxiv.org/abs/2408.05416)
Comments:
          submitted to IEEE Transactions on Image Processing(TIP)

- **What's New**: 본 논문은 고품질의 립싱크 동영상을 생성하기 위해 랜드마크 기반 확산 모델(Landmark-based diffusion model)을 제안합니다. 이 모델은 오디오와 비주얼 간의 매핑(Mapping)을 좀 더 명확하게 하기 위해 랜드마크를 중간 표현(Intermediate Representation)으로 사용하며, 각 단계를 통합적으로 최적화하여 오류 누적을 최소화합니다. 또한, TalkFormer라는 새로운 조건 모듈(Conditioning Module)을 도입하여 합성된 모션을 랜드마크에 표현된 모션과 일치시키고, 참조 이미지 특징을 대상 모션에 맞추는 방식을 사용하여 주체의 외관 세부 정보를 보존합니다.



### Style-Preserving Lip Sync via Audio-Aware Style Referenc (https://arxiv.org/abs/2408.05412)
Comments:
          submitted to IEEE Transactions on Image Processing(TIP)

- **What's New**: 본 논문에서는 새로운 오디오 인식 스타일 참조 방식(audio-aware style reference scheme)을 제안하여 개별 스타일을 유지하는 오디오 기반 입술 동기화(style-preserving audio-driven lip sync)를 개선했습니다. 이 방식은 입력 오디오와 스타일 참조 비디오의 참조 오디오 간의 관계를 활용하여 개별적인 발화 스타일을 효과적으로 포착합니다.



### How Does Audio Influence Visual Attention in Omnidirectional Videos? Database and Mod (https://arxiv.org/abs/2408.05411)
- **What's New**: 이 논문은 오디오와 비주얼 모드를 모두 고려한 입체 영상 (Omnidirectional Video, ODV) 의 주의 예측 연구를 위한 새로운 데이터베이스(AVS-ODV)를 제시합니다. AVS-ODV 데이터베이스는 162개의 ODV와 60명의 참가자로부터 수집된 시선 추적 데이터를 포함하고 있으며, 오디오 모드(무음, 모노, 앰비소닉)에 따른 주의 변화를 분석합니다. 또한, 오디오-비주얼 정보를 효과적으로 결합하여 ODV 주의 예측 성능을 향상시키는 새로운 딥러닝 모델(OmniAVS)을 제시합니다. OmniAVS는 U-Net 아키텍처를 기반으로 하며, ImageBind와 같은 다중 모달 기반 모델을 활용하여 오디오와 비주얼 특징을 효과적으로 통합합니다.

- **Technical Details**: OmniAVS 모델은 U-Net 아키텍처를 기반으로 하며, ImageBind와 같은 다중 모달 기반 모델을 활용하여 오디오와 비주얼 특징을 효과적으로 통합합니다. 이를 통해, OmniAVS는 ODV에서 오디오와 비주얼 정보를 효과적으로 결합하여 주의를 예측합니다. 또한, OmniAVS는 다양한 오디오 모드(무음, 모노, 앰비소닉)에 대한 주의 변화를 분석하기 위해 설계되었습니다.

- **Performance Highlights**: OmniAVS 모델은 AVS-ODV 데이터베이스에서 다른 최첨단 모델들에 비해 뛰어난 성능을 보여줍니다. 또한, OmniAVS는 기존의 오디오-비주얼 주의 예측 데이터베이스에서도 뛰어난 성능을 보이며, 일반화 성능이 뛰어남을 입증합니다.



### RSL-BA: Rolling Shutter Line Bundle Adjustmen (https://arxiv.org/abs/2408.05409)
- **What's New**: This paper introduces the first rolling shutter line-based bundle adjustment (RSL-BA) solution, a novel method for 3D reconstruction using lines as features in the context of rolling shutter cameras. Unlike previous RSBA methods that relied solely on sparse feature points, this approach leverages the inherent spatial structural information encoded in lines, offering greater robustness and accuracy, especially in challenging environments.

- **Technical Details**: The paper establishes a theoretical framework for rolling shutter line projection using Plücker line parameterization. It derives stable and efficient reprojection error formulations that address the challenges posed by the time-dependent exposure of rolling shutter cameras and the potential for degeneracy in optimization. The proposed method incorporates a strategy to prevent three common degeneracies in RSBA, including a new type of degeneracy identified in this research.

- **Performance Highlights**: Extensive experiments using both synthetic and real datasets demonstrate that RSL-BA achieves comparable efficiency and accuracy to existing point-based RSBA solutions. The results showcase the method's effectiveness in handling challenging scenarios where point-based methods may struggle due to limited feature availability or degeneracy issues.



### Mesh deformation-based single-view 3D reconstruction of thin eyeglasses frames with differentiable rendering (https://arxiv.org/abs/2408.05402)
- **What's New**: 본 논문은 단일 RGB 이미지로부터 고정밀 3D 안경테 모델을 복원하기 위한 새로운 메쉬 변형 기반 단일 뷰 3D 재구성 프레임워크를 제안합니다. 이 프레임워크는 안경테의 고유한 특성(예: 텍스처 특징 부족, 얇은 요소, 심각한 자체 폐색)을 고려하여 사전 지식과 도메인 특정 지식을 활용합니다. 특히, 합성 안경테 데이터 세트를 구축하여 미리 정의된 키 포인트가 있는 클래스별 안경테 템플릿을 정의합니다. 그런 다음 입력 안경테 이미지에서 얇은 구조와 몇 가지 텍스처 특징을 갖춘 이미지에서 정의된 키 포인트를 정확하게 탐지하기 위해 키 포인트 탐지기 및 개선기를 설계합니다. 이후, 미분 가능한 렌더링을 사용하여 템플릿 메쉬에 자유 형태 변형(FFD)을 수행하여 정확한 기하학을 생성하기 위한 새로운 최적화 접근 방식을 제안합니다. 렌더링된 결과와 해당 RGB 입력 간의 일관성을 강제하기 위해 고유한 구조, 실루엣, 키 포인트, 픽셀별 음영 정보 등의 제약 조건을 활용하여 일련의 손실 함수를 정의합니다.



### PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identifica (https://arxiv.org/abs/2408.05398)
- **What's New**: 이 논문은 최첨단의 **Masked Image Modeling (MIM)** 자기 지도 학습 방식을 사람 재식별 (Person ReID)에 도입하여 기존의 ViT (Vision Transformer)의 한계를 극복합니다. 기존 ViT는 전역 특징 추출에 능숙하지만, 사람 재식별은 미세한 지역 특징을 추출하는 것이 중요합니다. MIM은 이미지의 일부를 가리고 복원하는 학습 방식으로, 미세한 지역 특징을 효과적으로 추출하는 데 유용합니다.

- **Technical Details**: **PersonViT**는 MIM과 DINO (Distillation with No Labels) 대조 학습을 결합하여 대규모 비지도 사전 훈련을 수행하는 방법입니다. ViT를 기반으로 하며, LUPerson 데이터셋에서 비지도 사전 훈련을 거쳐 MSMT17, Market1501, DukeMTMC-reID, Occluded-Duke 등 다양한 사람 재식별 데이터셋에서 지도 학습 미세 조정을 수행합니다.

- **Performance Highlights**: 실험 결과, PersonViT는 여러 사람 재식별 데이터셋에서 최첨단 성능을 달성하며, 특히 **Occluded-Duke**와 같은 까다로운 데이터셋에서 뛰어난 성능을 보였습니다. 또한 사전 훈련된 모델을 시각화한 결과, PersonViT가 사람 신체 부위, 옷 패턴, 지역 부위 간의 연관성을 자동으로 파악하는 능력을 보여주었습니다.



### DeepSpeak Dataset v1.0 (https://arxiv.org/abs/2408.05366)
- **What's New**: This newsletter introduces "DeepSpeak", a novel, large-scale dataset designed to aid in the development and evaluation of deepfake detection systems. It consists of both real and deepfake videos featuring individuals engaging in speech and gestures. The dataset includes diverse demographics and various deepfake technologies, making it a valuable resource for the digital forensics community.

- **Technical Details**: DeepSpeak includes a vast collection of real videos from 220 individuals, spanning over 9 hours. The deepfake component incorporates various state-of-the-art techniques such as face-swap (FaceFusion, FaceSwap, DeepFaceLab) and lip-sync (DeepFakeVoice, FakeAV). The deepfake videos, totaling more than 25 hours, were generated using these methods, ensuring a comprehensive and representative representation of current deepfake technologies.

- **Performance Highlights**: The dataset was meticulously curated, ensuring diverse demographics, consistent recording conditions, and high-quality audio/video. It provides a rich resource for training and evaluating deepfake detection models. Furthermore, the authors plan to release updated versions of DeepSpeak periodically, incorporating new deepfake technologies and enhancing its comprehensiveness. This continuous update strategy ensures that the dataset remains relevant and useful for the evolving field of deepfake detection.



### Spherical World-Locking for Audio-Visual Localization in Egocentric Videos (https://arxiv.org/abs/2408.05364)
Comments:
          ECCV2024

- **What's New**: 이 논문은 에고센트릭 비디오의 멀티센서리 인식을 위한 일반적인 프레임워크인 Spherical World-Locking(SWL)을 제안합니다. SWL은 헤드 방향 측정에 따라 멀티센서리 스트림을 암시적으로 변환하여 2D 평면 시야의 기존 헤드 고정 에고센트릭 표현 방식과 비교하여 움직임에 의한 문제를 효과적으로 해결합니다. SWL은 멀티센서리 임베딩을 사용하여 이미지와 세계 좌표계 간의 비용이 많이 드는 프로젝션 없이도 장면 표현의 구형 구조를 유지하는 통합 인코더-디코더 트랜스포머 아키텍처를 설계합니다. 이러한 아키텍처는 에고센트릭 비디오 이해를 위한 오디오-비주얼 활성 스피커 위치 찾기, 청각 구형 소스 위치 찾기 및 일상 활동의 행동 예측을 포함한 다양한 벤치마크 작업에서 효과를 입증했습니다.



### AyE-Edge: Automated Deployment Space Search Empowering Accuracy yet Efficient Real-Time Object Detection on the Edg (https://arxiv.org/abs/2408.05363)
- **What's New**: AyE-Edge, a novel framework for Edge-OD (Edge-Object Detection) which optimizes for accuracy and power efficiency while meeting real-time requirements. It uses a multi-agent deep reinforcement learning (MARL) approach to coordinate three key techniques: keyframe selection, DNN pruning, and CPU-GPU configuration.

- **Technical Details**: AyE-Edge deploys a branch-and-bound methodology to explore the vast deployment space, leveraging T-Locality based keyframe selection, latency-restrained DNN pruning, and CPU core cluster selection. It also includes an Edge-OD performance collector for precise estimation of accuracy, power consumption, and latency.

- **Performance Highlights**: Experiments on the OnePlus 8T smartphone showcase AyE-Edge's effectiveness:  - Remarkable 96.7% reduction in power consumption compared to state-of-the-art (SOTA) methods.  - Outstanding real-time performance and detection accuracy.

- **Benefits**: AyE-Edge provides a powerful framework for optimizing Edge-OD deployments, leading to substantial improvements in power efficiency and accuracy while maintaining real-time capabilities.



### Enabling Quick, Accurate Crowdsourced Annotation for Elevation-Aware Flood Extent Mapping (https://arxiv.org/abs/2408.05350)
- **What's New**: FloodTrace, a crowdsourcing application for flood extent annotation, is introduced to efficiently train machine learning models for flood mapping. It utilizes elevation-guided annotation tools and 3D rendering to improve annotation accuracy and incorporates an uncertainty visualization framework for expert review and correction of crowdsourced annotations.



### CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization (https://arxiv.org/abs/2408.05341)
Comments:
          12 pages, 3 figures, 3 tables, accecpted by WBIR 2024

- **What's New**: 본 논문에서는 훈련 과정에서 다른 영상 대조(contrast)를 관찰하지 않고도 임의의 대조 영상에 일반화될 수 있는 대조 지각(contrast-agnostic) 변형 영상 등록 프레임워크(CAR)를 제안합니다. CAR는 임의의 대조를 모방하기 위한 랜덤 컨볼루션 기반 대조 증강 기법(random convolution-based contrast augmentation)과 대조 불변 표현 학습을 위한 대조 불변 잠재 정규화(contrast-invariant latent regularization)를 제안합니다. 훈련 과정에 사용되지 않은 영상 대조에서도 우수한 성능을 보여주며, 이는 CAR가 기존의 학습 기반 다중 대조 영상 등록 프레임워크와 비교하여 일반화 가능성과 적용 가능성을 크게 향상시킨다는 것을 의미합니다.



### VACoDe: Visual Augmented Contrastive Decoding (https://arxiv.org/abs/2408.05337)
Comments:
          10 pages, 7 figures

- **What's New**: This paper introduces VACoDe (Visual Augmented Contrastive Decoding), a novel method that enhances the performance of Large Vision-Language Models (LVLMs) by effectively addressing the issue of hallucination (generating incorrect outputs). Unlike previous methods that rely on a single augmentation, VACoDe leverages multiple image augmentations and adaptively selects the most contrastive augmentation for each task using a softmax distance metric.

- **Technical Details**: VACoDe works by providing various types of augmented images to the LVLMs and generating multiple outputs. The algorithm then assesses the difference between the original output distribution and the augmented output distributions. The augmentation with the largest distance gap, signifying the highest contrast, is identified and used to produce the final output through contrastive decoding. The core concept behind VACoDe is that different augmentations have distinct impacts on the output distribution of VLMs, and selecting the appropriate augmentation can significantly improve performance.

- **Performance Highlights**: Experimental results demonstrate that VACoDe outperforms existing decoding methods in various vision-language tasks. This method is universally applicable across different model types and sizes without the need for additional training or external models and data. VACoDe offers a robust and efficient way to enhance the accuracy and reliability of LVLMs, particularly in scenarios where hallucination poses a significant challenge.



### A Recurrent YOLOv8-based framework for Event-Based Object Detection (https://arxiv.org/abs/2408.05321)
- **What's New**: 이 연구는 기존 프레임 기반 객체 탐지 시스템에 시공간 모델링 기능을 추가하여 객체 탐지 성능을 향상시킨 새로운 프레임워크인 ReYOLOv8을 소개합니다. 특히, 이 연구에서는 이벤트 데이터를 효율적으로 인코딩하는 저지연, 메모리 효율적인 방법을 구현했으며, 이벤트 데이터의 특징을 활용하도록 설계된 새로운 데이터 증강 기법을 개발했습니다.



### VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents (https://arxiv.org/abs/2408.06327)
- **What's New**: 본 논문에서는 **VisualAgentBench (VAB)**라는 새로운 벤치마크를 소개하며, 이는 **대규모 멀티모달 모델(LMM)**을 **시각적 기반 에이전트(Visual Foundation Agents)**로 훈련하고 평가하기 위한 포괄적인 벤치마크입니다. VAB는 **Embodied**, **Graphical User Interface**, **Visual Design**과 같은 다양한 시나리오에 걸쳐 LMM의 잠재력을 최대한 활용하고 복잡한 실제 환경에서 LMM의 이해 및 상호 작용 능력을 평가하기 위해 설계되었습니다.



### EqNIO: Subequivariant Neural Inertial Odometry (https://arxiv.org/abs/2408.06321)
Comments:
          26 pages

- **What's New**: This paper proposes a novel subequivariant framework for inertial-only odometry (IO), which addresses the lack of symmetry in existing neural network models for IO. This framework enhances the robustness and generalizability of IO models by leveraging the inherent geometric symmetry in the environment.

- **Technical Details**: The proposed framework incorporates subequivariant layers that handle sequences of vectors and scalars to predict a subequivariant frame for IMU data. This frame is then used to extract invariant features, which are integrated with arbitrary network architectures. The invariant output is transformed by frame transformation to obtain equivariant displacements and covariances.

- **Performance Highlights**: The proposed method exhibits significant improvements in accuracy and reliability compared to existing techniques for inertial-only odometry. It outperforms baseline methods on various real-world datasets, including TLIO, Aria, RONIN, RIDI, and OxIOD, and demonstrates the effectiveness of integrating equivariance into the framework.

- **Key Contributions**: The paper makes several key contributions: (i) modeling subequivariance for IMU data, (ii) introducing a novel subequivariant framework for IO, and (iii) demonstrating state-of-the-art performance on multiple datasets.



### Long-Form Answers to Visual Questions from Blind and Low Vision Peop (https://arxiv.org/abs/2408.06303)
Comments:
          COLM 2024

- **What's New**: This paper introduces VizWiz-LF, a dataset of long-form answers to visual questions posed by blind and low vision (BLV) users. This dataset provides 4.2k long-form answers to 600 visual questions, collected from human expert describers and six VQA models. The paper also proposes a new set of functional roles for long-form answers, going beyond just answering the question to include explanations and suggestions.

- **Technical Details**: The paper identifies different functional roles (e.g., answer, explanation, suggestion) within long-form answers and proposes a classifier to automatically identify these roles. They also analyze the information sources used in these answers (e.g., image content, image quality, external information). They assess the ability of VQA models to abstain from answering unanswerable questions using various prompting strategies.

- **Performance Highlights**: The paper finds that long-form answers generated by VQA models often hallucinate incorrect visual details, particularly for unanswerable questions. They demonstrate that GPT-4 achieves the highest recall for abstention (0.82), while QWEN (0.42) performs well with default settings. The paper also highlights the importance of evaluating LFVQA beyond just factual accuracy, considering factors like relevance and plausibility in user experience.

- **Contributions**: The work provides the first dataset with both short and long answers to visual questions, enabling the transfer from short-answer VQA tasks to long-answer tasks. It proposes a new set of functional roles for LFVQA, which can be used to improve and evaluate LFVQA systems. It highlights the importance of user experience evaluation for LFVQA, going beyond factual accuracy and considering factors like relevance and plausibility.



### Finding Patterns in Ambiguity: Interpretable Stress Testing in the Decision~Boundary (https://arxiv.org/abs/2408.06302)
Comments:
          To be published in the Responsible Generative AI workshop at CVPR

- **What's New**: 이 논문은 딥 바이너리 분류기(deep binary classifiers)의 해석력(interpretability)을 높이는 새로운 방법을 제안합니다. 이 방법은 모델의 의사 결정 경계(decision boundary)에서 대표적인 샘플(prototypes)을 선택하고 후처리 설명 알고리즘(post-model explanation algorithms)을 적용하여 작동합니다. GASTeN(Generative Adversarial Networks for Stress Testing)을 사용하여 의사 결정 경계 근처에 있는 합성 데이터(synthetic data)를 생성하고, UMAP(Uniform Manifold Approximation and Projection)과 GMM(Gaussian Mixture Models)을 사용하여 패턴을 감지합니다. 각 클러스터(cluster)에서 대표적인 프로토타입(prototype)을 선택하고 2D 시각화(visualization)와 GradientSHAP 분석(analysis)을 사용하여 프로토타입과 의사 결정 경계를 시각화합니다.



### Context-aware Visual Storytelling with Visual Prefix Tuning and Contrastive Learning (https://arxiv.org/abs/2408.06259)
Comments:
          18 pages, 12 figures, accepted by INLG 2024

- **What's New**: 이 논문은 사전 훈련된 기반 모델의 일반화 능력을 활용하여 이미지 시퀀스에서 다중 문장 스토리를 생성하는 시각적 스토리텔링 시스템을 제안합니다. 이 시스템은 모달리티를 연결하는 경량 비전-언어 매핑 네트워크만 학습하고 일관성을 높이기 위해 컨텍스트를 통합합니다.

- **Technical Details**: 이 프레임워크는 비전-언어 매핑 네트워크의 컨텍스트 인식 기능을 강화하고 이전 스토리 문장을 통합하여 일관성을 강화합니다. 또한 시각적 관련성과 정보성을 개선하기 위해 다중 모달 대조 학습 목표를 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과는 자동 평가 지표와 인간 평가 모두에서 이 프레임워크에 의해 생성된 스토리가 다양하고 일관성 있으며 정보가 풍부하고 흥미롭다는 것을 보여줍니다.



### Rethinking Video with a Universal Event-Based Representation (https://arxiv.org/abs/2408.06248)
Comments:
          137 pages. PhD dissertation at the University of North Carolina, Chapel Hill

- **What's New**: 이 논문에서는 기존의 프레임 기반 비디오 시스템의 단점을 해결하기 위해 새로운 비디오 표현 및 시스템 프레임워크인 주소, 십진화, Δt 이벤트 표현(ADΔER)을 제시합니다. ADΔER는 다양한 프레임 및 이벤트 카메라 소스를 단일 이벤트 기반 표현으로 변환하여 소스 모델 기반 손실 압축과 기존 프레임 기반 애플리케이션과의 하위 호환성을 지원합니다. ADΔER는 높은 시간적 중복성이 있는 장면에 대해 최첨단 애플리케이션 속도와 압축 성능을 달성합니다. 특히 ADΔER는 컴퓨터 비전을 위한 완전히 새로운 제어 메커니즘을 제공합니다. 즉, 애플리케이션 속도는 장면 콘텐츠와 손실 압축 수준 모두와 상관관계를 가질 수 있습니다. 이 논문은 대규모 비디오 감시 및 자원 제약이 있는 감지에 대한 이벤트 기반 비디오의 의미를 논의합니다.



### Zero-shot 3D Segmentation of Abdominal Organs in CT Scans Using Segment Anything Model 2: Adapting Video Tracking Capabilities for 3D Medical Imaging (https://arxiv.org/abs/2408.06170)
Comments:
          16 pages, 6 figures (including 1 supplemental figure), 3 tables

- **What's New**: 본 연구는 Segment Anything Model 2 (SAM 2)의 영상 추적 기능을 활용하여 3차원 CT 영상의 복부 장기 분할에 대한 제로 샷(zero-shot) 성능을 평가했습니다. SAM 2는 기존에 학습된 데이터 없이 새로운 데이터에 적용될 수 있는 모델입니다.



### ACCELERATION: Sequentially-scanning DECT Imaging Using High Temporal Resolution Image Reconstruction And Temporal Extrapolation (https://arxiv.org/abs/2408.06163)
- **What's New**: This paper proposes a novel technique called ACCELERATION to address temporal inconsistency in sequentially-scanning dual-energy computed tomography (DECT) data sets. ACCELERATION utilizes high temporal resolution image reconstruction and temporal extrapolation to improve iodine quantification accuracy in sequentially-scanning DECT.

- **Technical Details**: ACCELERATION reconstructs several time-resolved images at high temporal resolution using short-scan data acquired at the first tube potential. Temporal extrapolation is then applied to obtain the image corresponding to the acquisition time of the second tube potential. The temporally consistent images at two tube potentials are used for iodine quantification. The method leverages implicit neural representation learning, where a multi-layer perceptron (MLP) is trained to express the mapping relationship between pixel coordinates and pixel values. This mapping is used for image reconstruction and temporal extrapolation. ACCELERATION is validated and evaluated using numerical simulation data sets generated from clinical human subject exams.

- **Performance Highlights**: Results demonstrate the improvement of iodine quantification accuracy using ACCELERATION, addressing the challenge posed by temporal inconsistency in sequentially-scanning DECT data sets.



### Palantir: Towards Efficient Super Resolution for Ultra-high-definition Live Streaming (https://arxiv.org/abs/2408.06152)
- **What's New**: Palantir는 기존의 프레임 단위 스케줄링 방식을 뛰어넘어 픽셀 패치 단위의 미세한 스케줄링을 도입한 최초의 신경망 기반 UHD 실시간 스트리밍 시스템입니다. Palantir은 픽셀 패치에 대한 안커/비안커(anchor/non-anchor) 판단을 효율적으로 수행하여 컴퓨팅 오버헤드를 최소화하고 UHD 화질을 유지합니다.



### Five Pitfalls When Assessing Synthetic Medical Images with Reference Metrics (https://arxiv.org/abs/2408.06075)
Comments:
          10 pages, 5 figures, accepted at Deep Generative Models workshop @ MICCAI 2024

- **What's New**: 이 논문은 의료 영상의 생성 모델 평가에 사용되는 참조 지표(reference metrics)의 5가지 함정(pitfalls)을 제시하고, 이러한 함정을 피하기 위한 전략을 논의합니다. 참조 지표는 영상의 품질을 객관적이고 정량적으로 비교하는 데 사용되지만, 의료 영상에서는 영상 내용, 영상 데이터 형식, 영상 해석에 대한 가정이 다르기 때문에 함정에 빠지기 쉽습니다. 특히 SSIM, PSNR, MAE와 같은 일반적으로 사용되는 지표는 모든 상황에 적합한 것은 아닙니다.



### Parallel transport on matrix manifolds and Exponential Action (https://arxiv.org/abs/2408.06054)
- **What's New**: 본 논문은 특정 유형의 매트릭스 리 군(matrix Lie group)에 대한 평행 이동(parallel transport)을 매트릭스 지수(matrix exponential)와 지수 작용(exponential action)을 사용하여 표현합니다. 또한 스티펠 다양체(Stiefel manifold)와 플래그 다양체(flag manifold)에 대한 평행 이동 공식을 제공하며, 특히 스티펠 다양체에 대해서는 $O(nd^2)$의 시간 복잡도로 계산 가능한 효율적인 알고리즘을 제시합니다. 이는 스티펠 다양체에 대한 평행 이동을 계산하는 오랜 난제를 해결하는 데 한 걸음 더 나아간 것입니다.



### Uncertainty-Informed Volume Visualization using Implicit Neural Representation (https://arxiv.org/abs/2408.06018)
Comments:
          To appear in IEEE Workshop on Uncertainty Visualization in conjunction with IEEE VIS 2024, Florida, USA

- **What's New**: 이 논문은 과학적 시각화 작업을 위한 불확실성 인식(Uncertainty-Aware) 암묵적 신경 표현(Implicit Neural Representation)을 제안합니다. 이 방법은 스칼라 필드 데이터 세트를 효과적으로 모델링하고 볼륨 시각화 작업에서 추정된 불확실성 정보의 효율성과 이점을 종합적으로 연구합니다. 특히, 두 가지 원칙적인 딥 불확실성 추정 기술인 딥 앙상블(Deep Ensemble)과 몬테카를로 드롭아웃(Monte Carlo Dropout, MCDropout)을 평가하여 스칼라 필드 데이터 세트에서 불확실성에 따른 볼륨 시각화를 가능하게 합니다.



### A Sharpness Based Loss Function for Removing Out-of-Focus Blur (https://arxiv.org/abs/2408.06014)
Comments:
          6 pages, IEEE MMSP

- **What's New**: 이 논문은 이미지 흐림 제거를 위한 딥 러닝 모델에 새로운 손실 함수(loss function)로 객관적인 선명도 지표(sharpness metric)인 Q를 사용하는 방법을 제시합니다. 또한 실제 흐릿한 이미지(out-of-focus blur) 데이터셋을 새로 구축하여 복원 모델을 평가합니다.



### Image Denoising Using Green Channel Prior (https://arxiv.org/abs/2408.05923)
Comments:
          arXiv admin note: text overlap with arXiv:2402.08235

- **What's New**: 이 논문은 이미지 노이즈 제거 분야에서 **녹색 채널 우선(Green Channel Prior, GCP)**을 활용하는 새로운 방식을 제안합니다. GCP는 녹색 채널이 일반적으로 다른 채널보다 높은 샘플링 비율을 갖는다는 사실을 활용하여 노이즈 제거 성능을 향상시킵니다. 기존의 패치 기반 노이즈 제거 프레임워크에 GCP를 통합하여 유사 패치를 찾는 데 도움을 주고, 변환 도메인에서 스파스성(sparsity)을 증가시켜 노이즈 제거 효과를 높입니다. 또한, GCP-ID는 **컨볼루션 신경망(Convolutional Neural Networks, CNNs)**을 사용하여 노이즈 추정 문제를 분류 작업으로 바꾸어 이미지 콘텐츠에 대한 적응력을 향상시킵니다.



### Polyp SAM 2: Advancing Zero shot Polyp Segmentation in Colorectal Cancer Detection (https://arxiv.org/abs/2408.05892)
- **What's New**: 본 연구는 폴립 분할에 대한 새롭고 유망한 접근 방식으로 Segment Anything Model 2 (SAM 2)를 조사합니다. SAM 2는 특히 제로 샷(zero-shot) 학습 능력이 뛰어나 별도의 훈련 없이도 폴립 이미지와 비디오를 효과적으로 분할할 수 있습니다.



### Deep Learning in Medical Image Registration: Magic or Mirage? (https://arxiv.org/abs/2408.05839)
- **What's New**: 이 논문은 고전적인 최적화 기반 방식과 학습 기반 방식, 2가지 주요 방식이 변형 이미지 등록(Deformable Image Registration, DIR)에서 보이는 장단점을 명확히 제시하고, 각 방식이 적합한 조건을 규명합니다. 특히, 고전적인 방식의 성능이 픽셀 강도와 레이블 분포의 상호 정보(Mutual Information)와 강한 상관관계가 있음을 밝히고, 학습 기반 방식의 아키텍처 설계는 이 상관관계에 영향을 미치지 못한다는 가설을 제시합니다. 또한, 약한 감독(Weak Supervision)을 활용한 학습 기반 방식은 고전적인 방식으로 불가능한 고정밀 강도 및 레이블 등록을 수행할 수 있음을 보여주며, 도메인 변화에 대한 민감성 문제도 지적합니다. 마지막으로, 이러한 관찰을 토대로 특정 등록 문제에 가장 적합한 방식을 선택하는 일반적인 가이드라인을 제시합니다.



### Sampling Foundational Transformer: A Theoretical Perspectiv (https://arxiv.org/abs/2408.05822)
- **What's New**: 본 논문은 다양한 데이터 유형(예: 포인트 클라우드, 그래프, 시퀀스)과 제약 조건(예: 회전 불변성)을 처리할 수 있는 샘플링 기반 기초 트랜스포머(Sampling Foundational Transformer, SFT)를 제안합니다. 이 모델은 여러 데이터 출처에서 작동하는 현대의 기초 모델링을 위해 중요하며, 샘플링 기반의 메커니즘을 통해 선형적 복잡도와 실제 추론 시간 개선을 달성합니다. 또한, 트랜스포머 계층의 의사 볼록(pseudoconvex) 공식화를 통해 모델의 수렴 속도를 높입니다.

- **Technical Details**: SFT는 샘플링 기반 메커니즘을 통해 다양한 데이터 유형에 적용될 수 있는 스파스 글로벌 어텐션(sparse global attention)을 구현합니다. 이를 위해 Gumbel-Softmax 재매개변수화 기반의 샘플링 없이 샘플링(sampling without replacement) 방법을 사용하고, 신경망 중요도 점수(neural importance score)를 계산하여 중요한 토큰에 주목하도록 합니다. 또한, maxout 네트워크(maxout network) 기반의 어텐션 비선형성(attention nonlinearity)을 설계하여 트랜스포머 계층의 의사 볼록 공식화를 가능하게 합니다. 이는 모델의 수렴 속도를 향상시키고 하이퍼파라미터 튜닝 과정을 간소화합니다.

- **Performance Highlights**: SFT는 포인트 클라우드, 그래프, 시퀀스 데이터 세트에서 다양한 벤치마크 작업에서 경쟁력 있는 결과를 보여주었습니다. 특히 포인트 클라우드 분류 및 의미적 분할(semantic segmentation) 작업에서 전통적인 방식과 회전 불변성 방식 모두에서 우수한 성능을 기록했습니다. 그래프 데이터 세트에서도 페пти드(Peptide) 데이터 세트와 컴퓨터 비전(Computer Vision) 데이터 세트에서 우수한 성능을 보였습니다. 시퀀스 데이터 세트에서도 장거리 아레나 벤치마크(long-range arena benchmark)에서 4개의 분류 작업과 1개의 검색 작업에서 우수한 결과를 보여주었습니다. 또한, SFT는 특수화된 모델에 비해 추론 속도가 빠르다는 장점을 가지고 있습니다.



### Prototype Learning Guided Hybrid Network for Breast Tumor Segmentation in DCE-MRI (https://arxiv.org/abs/2408.05803)
- **What's New**: 본 연구에서는 유방암 종양 분할을 위한 효율적인 인코더-디코더 기반 하이브리드 네트워크를 제안합니다. 이 네트워크는 컨볼루션 계층, 디컨볼루션 계층 및 트랜스포머 계층을 통합합니다. 하이브리드 네트워크는 3D CNN 인코더와 글로벌 의존성을 포착하기 위한 3D 트랜스포머 계층으로 구성됩니다. 효율적인 최적화를 위해 두 개의 서브 네트워크 인코더를 사용하고 디코더를 위해 세 개의 전치 컨볼루션 계층을 사용합니다. 트랜스포머와 인코더 서브 네트워크로부터 혼합된 특징 정보를 사용하는 디코더는 전체 해상도에서 분할 마스크를 생성합니다. 또한, 제안된 하이브리드 네트워크의 분할 성능을 향상시키기 위해 프로토타입 학습 기반 예측 모듈을 도입합니다. 이 모듈은 온라인 클러스터링을 사용하여 각 카테고리에 대한 프로토타입 특징을 계산합니다. 이러한 프로토타입 특징은 디코더의 정규화된 출력 특징과 유사성 맵을 계산하는 데 사용됩니다. 이 유사성 맵은 종양 복셀의 국지화 맵을 제공합니다. 또한, 디코더의 출력 특징을 재구성하기 위해 어텐션 기반 융합 전략을 설계합니다. 마지막으로, 프로토타입과 디코더의 재구성된 출력 특징과의 유사성 맵을 융합하여 최종 유방암 종양 마스크를 생성합니다. 이 접근 방식을 통해 하이브리드 네트워크는 글로벌 및 로컬 의미론적 단서를 더 효과적으로 포착할 수 있습니다.



### CURLing the Dream: Contrastive Representations for World Modeling in Reinforcement Learning (https://arxiv.org/abs/2408.05781)
Comments:
          Paper accepted for 24th International Conference on Control, Automation and Systems (ICCAS)

- **What's New**: Curled-Dreamer, a 새로운 강화 학습 알고리즘이 DreamerV3 프레임워크에 contrastive learning (대조 학습)을 통합하여 시각적 강화 학습 작업에서 성능을 향상시킵니다.



### Advancing Re-Ranking with Multimodal Fusion and Target-Oriented Auxiliary Tasks in E-Commerce Search (https://arxiv.org/abs/2408.05751)
- **What's New**: 이 연구는 전자상거래 검색 재순위 지정 모델에 텍스트 및 시각 정보를 결합하여 사용자 경험을 향상시키고 전환율을 높이는 새로운 접근 방식을 제시합니다. "ARMMT"라는 새로운 모델은 텍스트와 시각 정보를 통합하는 "Context-Aware Fusion Unit (CAFU)"와 순위와 관련된 보조 작업을 활용하여 품목 표현을 개선하고 개인 맞춤형 추천 기능을 강화합니다.



### Deep Learning with Data Privacy via Residual Perturbation (https://arxiv.org/abs/2408.05723)
- **What's New**: 이 논문에서는 개인정보 보호 딥 러닝(DL)을 위한 확률적 미분 방정식 기반 잔차 섭동(stochastic differential equation-based residual perturbation)을 제안합니다. 이 방법은 ResNets의 각 잔차 매핑(residual mapping)에 가우시안 노이즈를 주입하여 개인정보를 보호합니다. 이론적으로 잔차 섭동은 차등 프라이버시(DP)를 보장하고 DL의 일반화 격차(generalization gap)를 줄이는 것으로 증명되었습니다.



### TC-KANRecon: High-Quality and Accelerated MRI Reconstruction via Adaptive KAN Mechanisms and Intelligent Feature Scaling (https://arxiv.org/abs/2408.05705)
Comments:
          10 pages, 3 figures

- **What's New**: TC-KANRecon: This paper presents a novel conditional guided diffusion model, TC-KANRecon, for accelerating MRI reconstruction while preserving image quality. It incorporates two key modules: the Multi-Free U-KAN (MF-UKAN) module and a dynamic clipping strategy, aiming to address limitations in existing deep learning-based MRI reconstruction techniques.



### Evaluating BM3D and NBNet: A Comprehensive Study of Image Denoising Across Multiple Datasets (https://arxiv.org/abs/2408.05697)
- **What's New**: 이 논문은 이미지 잡음 제거 분야에서 기존의 비 학습 기반 기법(BM3D로 대표됨)과 최신 학습 기반 방법(NBNet으로 대표됨)을 비교 분석하여 다양한 잡음 환경에서 각 기법의 장단점을 밝혀냈습니다. 특히, 저조도, 과조도, 흐림과 같은 다양한 잡음에 대한 각 기법의 효과를 다양한 이미지 품질 평가 지표(IQA)와 객체 검출 성능 분석을 통해 비교했습니다.



### BeyondCT: A deep learning model for predicting pulmonary function from chest CT scans (https://arxiv.org/abs/2408.05645)
Comments:
          5 tables, 7 figures,22 pages

- **What's New**: 본 논문은 흉부 CT 스캔을 사용하여 폐 기능을 예측하는 딥 러닝 알고리즘인 BeyondCT를 개발 및 검증했습니다. 이 알고리즘은 3D CNN (Convolutional Neural Network)과 Vision Transformer (ViT) 아키텍처를 결합하여 폐 용량 (FVC) 및 1초 강제 호기량 (FEV1)을 예측합니다. 이는 기존의 PFT (폐 기능 검사)를 대체할 수 있는 가능성을 보여줍니다.



### Residual-INR: Communication Efficient On-Device Learning Using Implicit Neural Representation (https://arxiv.org/abs/2408.05617)
Comments:
          This paper has been accepted by ICCAD 2024

- **What's New**: Residual-INR는 엣지 디바이스 간 통신 효율성을 높이기 위해 잉크(Implicit Neural Representation)를 활용한 새로운 엣지 컴퓨팅 기반 학습 프레임워크입니다. 기존 엣지 디바이스 간 JPEG 이미지 전송 방식 대신, 잉크를 활용해 이미지를 압축하고, 엣지 디바이스에서 학습을 진행하는 방식을 제시합니다.

- **Technical Details**: Residual-INR은 이미지를 배경과 객체로 분리하여 각각 다른 크기의 잉크로 압축하는 기술입니다. 즉, 배경은 상대적으로 작은 잉크로 압축하고, 객체는 별도의 잉크를 통해 더 높은 품질로 압축하여 잉크 크기를 줄이면서도 객체의 정보는 유지하는 기술입니다. 이를 통해 데이터 전송량을 크게 줄이고, 엣지 디바이스에서의 연산량도 감소시킬 수 있습니다.

- **Performance Highlights**: Residual-INR은 기존 JPEG 압축 방식 대비 최대 12.1배 이미지 크기 감소, 데이터 전송량 최대 5.16배 감소, 학습 속도 최대 2.9배 향상을 달성했습니다. 또한, 객체 인식 정확도를 유지하면서도 잉크 크기 최적화를 통해 엣지 디바이스에서 CPU를 사용하지 않고 학습이 가능하도록 설계되었습니다.

- **Benefits**: Residual-INR은 엣지 컴퓨팅 환경에서 데이터 전송량을 줄이고, 학습 속도를 향상시키는 데 유용합니다. 특히, 이미지/영상 처리, 객체 인식, 자율 주행 등 데이터 전송량이 많은 분야에서 효과적인 기술입니다.

- **Limitations**: Residual-INR은 잉크 네트워크를 학습하는 데 시간이 오래 걸릴 수 있습니다. 또한, 잉크 네트워크의 크기가 커질수록 학습 및 추론에 더 많은 자원이 필요합니다.



### Sequential Representation Learning via Static-Dynamic Conditional Disentanglemen (https://arxiv.org/abs/2408.05599)
Comments:
          Accepted at ECCV 2024

- **What's New**: 이 논문은 비디오에서 시간에 의존하지 않는 요소와 시간에 따라 변하는 요소를 분리하는 데 중점을 두고 순차 데이터에서 자기 지도 학습된 분리된 표현 학습을 탐구합니다. 저자는 정적/동적 변수 간의 인과 관계를 명시적으로 고려하여 이러한 요소 간의 일반적인 독립성 가정을 깨뜨리는 새로운 모델을 제안합니다. 이는 Normalizing Flows(정규화 흐름)를 추가하여 모델 표현력을 향상시킵니다. 저자는 이러한 요소에 대한 공식적인 정의를 제안합니다. 이 형식주의는 기본 요소가 식별 가능하도록 충분한 조건을 유도하고 새로운 모델 프레임워크에 직접적이고 효율적으로 통합할 수 있는 이론적으로 근거 있는 새로운 분리 제약 조건을 도입합니다. 실험 결과, 제안된 접근 방식은 장면의 동역학이 해당 콘텐츠의 영향을 받는 시나리오에서 이전의 복잡한 최첨단 기술보다 뛰어난 성능을 보여줍니다.



### Impacts of Darwinian Evolution on Pre-trained Deep Neural Networks (https://arxiv.org/abs/2408.05563)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 딥 러닝 모델 훈련을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 진화론적 이론을 기반으로 하여, 백프로퍼게이션(Back-Propagation, BP)으로 훈련된 딥 신경망을 초기 종(primordial ancestor)으로 간주하고, 차등 진화(Differential Evolution)를 통해 이들을 진화시킵니다. 이를 통해 기존 BP 기반 훈련 방식의 단점인 과적합(overfitting) 문제를 완화하고, 훈련 시간을 크게 단축시키는 효과를 얻을 수 있습니다.

- **Technical Details**: 본 연구는 딥 신경망을 생물학적 종(species)에 비유하고, 이들의 훈련 과정을 진화론적 관점에서 해석합니다.  BP로 훈련된 딥 신경망은 초기 종으로 간주되고, 차등 진화를 통해 이들이 진화하면서 데이터셋, 환경, 모델과 살아있는 생물 종 사이의 상관관계를 조사합니다. 본 논문에서는 차등 진화(Differential Evolution)를 기반으로 하는 진화 알고리즘을 사용합니다.  차등 진화는 개체군 내에서의 차이를 이용하여 새로운 개체를 생성하는 방법으로, 기존 BP 방법과 달리 기울기 정보를 직접 사용하지 않고도 최적화를 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 과적합을 줄이고, 기존 BP 방법에 비해 훈련 시간을 10배 이상 단축시키는 효과를 보였습니다. 또한, 제안된 프레임워크는 딥 신경망 및 대규모 데이터셋에서도 효과적인 성능을 보였습니다. 



### DeepFace-Attention: Multimodal Face Biometrics for Attention Estimation with Application to e-Learning (https://arxiv.org/abs/2408.05523)
Comments:
          Article accepted in the IEEE Access journal. Accessible at this https URL

- **What's New**: 이 논문은 웹캠 비디오에 적용된 얼굴 분석 기술의 앙상블을 사용하여 주의 수준(인지 부하)을 추정하는 혁신적인 방법을 소개합니다. 이 방법은 특히 e-러닝 애플리케이션에서 유용하며, 연구팀은 공개 멀티모달 데이터베이스인 mEBAL2에서 접근 방식을 훈련, 평가 및 비교했습니다. mEBAL2는 8가지 다른 작업을 수행한 60명의 사용자 데이터를 포함하며, 작업은 난이도가 다양하여 인지 부하의 변화를 유발합니다. 이 논문의 접근 방식은 최첨단 얼굴 분석 기술을 적용하여 사용자의 인지 부하를 높거나 낮은 주의 수준으로 정량화합니다. 인지 부하와 관련된 여러 행동 신호와 생리적 과정, 예를 들어 눈 깜빡임, 심박수, 얼굴 액션 유닛(facial action units), 머리 자세 등을 사용합니다. 또한, 이 논문은 어떤 개별 기능이 더 나은 결과를 얻는지, 가장 효율적인 조합은 무엇인지, 지역적(local) 및 전역적(global) 기능을 탐구하고, 일시적인 시간 간격이 주의 수준 추정에 어떤 영향을 미치는지 등을 이해하기 위한 연구를 수행합니다. 연구 결과에 따르면 전역 얼굴 기능은 특히 시간 창이 증가함에 따라 점수 수준 융합(score-level fusion)을 사용하는 멀티모달 시스템에 더 적합하며, 반면 지역 기능은 점수 수준 융합 방식으로 신경망 훈련을 통한 융합에 더 적합합니다. 이 논문의 방법은 공개 벤치마크인 mEBAL2를 사용하여 기존의 최첨단 정확도를 능가합니다.



### Anticipation through Head Pose Estimation: a preliminary study (https://arxiv.org/abs/2408.05516)
Comments:
          Accepted at the workshop on advancing Group Understanding and robots' adaptive behavior (GROUND), held at the Robotics Science and Systems (RSS) Conference, 2024

- **What's New**: 이 논문은 인간-로봇 상호작용(HRI) 시나리오에서 로봇이 인간의 행동 목표를 예측하는 데 있어서 머리 자세를 시각적 단서로 활용하는 가능성을 탐구합니다. 특히, 이 연구는 인간의 팔을 뻗거나 물건을 운반하는 행동을 예측하는 데 초점을 맞추고 있습니다. 머리와 손, 물체의 공간 및 시간적 관계를 분석함으로써, 이 연구는 단기 예측이 가능함을 보여주고, 향후 인간-로봇 상호작용에 적용할 기반을 마련합니다.



### PointNCBW: Towards Dataset Ownership Verification for Point Clouds via Negative Clean-label Backdoor Watermark (https://arxiv.org/abs/2408.05500)
Comments:
          12 pages

- **What's New**: 본 논문은 포인트 클라우드 데이터셋의 저작권 보호를 위해 새로운 ‘스케일러블 클린-라벨 백도어 기반 데이터셋 워터마크(Scalable Clean-Label Backdoor-based Dataset Watermark)’를 제안합니다. 기존의 방법들은 많은 클래스를 가진 대규모 데이터셋에서 효과가 떨어지는 한계를 가지고 있었지만, 본 논문에서 제시된 방법은 모든 클래스에서 샘플을 워터마킹할 수 있어 대규모 데이터셋에도 효과적으로 적용될 수 있습니다.

- **Technical Details**: 본 논문에서 제시된 방법은 ‘네거티브 트리거 효과(negative trigger effects)’를 활용합니다. 먼저 비-타겟 클래스의 포인트 클라우드를 타겟 클래스의 포인트 클라우드와 유사하도록 변형시키고, 변형된 포인트 클라우드에 트리거 패턴을 삽입합니다. 이렇게 생성된 워터마크된 샘플은 타겟 클래스의 레이블과는 다르지만, 특징 공간에서는 타겟 클래스의 샘플과 유사합니다. 훈련된 DNN은 삽입된 트리거 패턴을 타겟 레이블을 예측하지 못하게 하는 신호로 인식하게 됩니다. 즉, 트리거 패턴이 나타나면 타겟 클래스에 대한 예측 확신도가 감소하게 됩니다.

- **Performance Highlights**: 본 논문에서 제시된 방법의 효과성과 잠재적인 제거 방법에 대한 저항성을 실험적으로 검증했습니다. 실험 결과, 제안된 PointNCBW (Negative Clean-Label Backdoor Watermark for Point Clouds)는 기존의 방법에 비해 뛰어난 성능을 보여주었으며, 대규모 데이터셋에 대한 확장성을 갖추고 있습니다. 또한, 잠재적인 공격에 대해 저항성을 가지는 것으로 확인되었습니다.



### Unidirectional imaging with partially coherent ligh (https://arxiv.org/abs/2408.05449)
Comments:
          25 Pages, 8 Figures

- **What's New**: 이 연구는 공간적으로 부분적으로 결맞는 빛(spatially partially coherent light) 아래에서 단방향 이미징(unidirectional imaging)을 구현하는 새로운 기술을 제시합니다. 이 기술은 한 방향(FOV A -> FOV B)으로만 이미지를 형성하고 반대 방향(FOV B -> FOV A)으로는 이미지 형성을 차단합니다. 이를 통해 전방향으로는 고품질 이미지를 얻으면서도 후방향으로는 이미지 형성을 왜곡시켜 에너지 효율을 높일 수 있습니다.



### PRISM Lite: A lightweight model for interactive 3D placenta segmentation in ultrasound (https://arxiv.org/abs/2408.05372)
- **What's New**: 본 연구는 3D 초음파 이미지에서 태반을 실시간으로 분할하는 경량의 상호 작용형 분할 모델을 제안합니다. 이 모델은 완전 자동화된 모델에서 분할을 초기화하는 방식을 채택하며, 반복적인 개선을 위해 인간-루프 방식으로 설계되었습니다. 특히, 이 모델은 기존의 모델과 비교하여 상당히 적은 매개변수를 사용하면서도 우수한 분할 성능을 보여줍니다. 또한, 제안된 모델은 추론 속도가 훨씬 빠르고, 초기 마스크가 부족한 경우에도 견고합니다. 코드는 [링크]에서 확인할 수 있습니다.



### GesturePrint: Enabling User Identification for mmWave-based Gesture Recognition Systems (https://arxiv.org/abs/2408.05358)
Comments:
          Accepted to the 44th IEEE International Conference on Distributed Computing Systems (ICDCS 2024)

- **What's New**: GesturePrint, 최초로 mmWave 기반 제스처 인식 시스템에 사용자 식별 기능을 추가한 솔루션입니다. mmWave 센서를 사용하여 제스처 인식과 사용자 식별을 동시에 수행하는 방법을 제시합니다.



### Revisiting Multi-Modal LLM Evaluation (https://arxiv.org/abs/2408.05334)
- **What's New**: 본 연구는 기존의 비주얼 퀘스천 앤서링(VQA) 데이터셋의 한계를 극복하고 최신 멀티모달 대규모 언어 모델(MLLM)을 위한 새로운 평가 데이터셋을 제시합니다. 새로운 데이터셋은 다양한 질문 유형과 시각적 추론 능력을 평가할 수 있도록 설계되었으며, 특히 시각적 지각(visual grounding) 능력을 더욱 엄격하게 평가하는 데 중점을 둡니다.

- **Technical Details**: 본 연구에서는 다음과 같은 데이터셋을 사용합니다:

- **TDIUC**: 12가지 질문 유형을 포함하여 세분화된 분석을 가능하게 하는 VQA 데이터셋.
- **TallyQA**: 간단하고 복잡한 계산 질문을 포함하는 VQA 데이터셋.
- **DVQA**: 차트 이해를 위해 광학 문자 인식(OCR)을 필요로 하는 VQA 데이터셋.
- **VQDv1**: 주어진 질의에 맞는 모든 이미지 영역을 식별해야 하는 VQA 데이터셋. 

또한 다음과 같은 최신 MLLM 모델을 평가합니다:

- **LLaVA 1.5**, **LLaVA-NeXT**, **BLIP2**, **InstructBLIP**, **GPT-4V**, **GPT-4o**

- **Performance Highlights**: 본 연구에서는 새로운 평가 데이터셋을 사용하여 최신 MLLM의 성능을 분석한 결과, 기존의 데이터셋에서는 발견되지 않았던 새로운 약점들이 드러났습니다. 예를 들어, 일부 MLLM은 복잡한 시각적 추론 능력을 요구하는 VQDv1에서 저조한 성능을 보였습니다. 이는 이러한 모델들이 단일 객체 식별에 의존하는 기존의 데이터셋에 과도하게 적응되어 있기 때문일 수 있습니다.



### The impact of internal variability on benchmarking deep learning climate emulators (https://arxiv.org/abs/2408.05288)
- **What's New**: 본 논문에서는 기존 기후 모델 에뮬레이터 벤치마크인 ClimateBench를 분석하여 선형 회귀 기반 에뮬레이터인 LPS(Linear Pattern Scaling)가 심층 학습 기반 에뮬레이터인 ClimaX보다 지역별 표면 온도, 강수량 및 극심한 강수량 예측에 더 우수한 성능을 보인다는 것을 발견했습니다. 이는 강수량이 비선형적 관계를 가지고 있다는 것을 고려했을 때 놀라운 결과입니다. 또한 ClimateBench에서 사용되는 3개의 시뮬레이션이 내부 변동성을 완전히 제거하기에 충분하지 않다는 점을 강조합니다.



New uploads on arXiv(cs.AI)

### LLMs can Schedu (https://arxiv.org/abs/2408.06993)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용하여 작업장 스케줄링 문제(JSSP)를 해결하는 새로운 접근 방식을 제시합니다. 연구자들은 LLM 훈련을 위해 특별히 설계된 최초의 지도 학습 데이터 세트(120k 개)를 소개하며, LLM 기반 스케줄링이 기존의 신경망 방식과 유사한 성능을 달성할 수 있음을 보여줍니다. 또한 JSSP에서 LLM의 효과를 향상시키는 샘플링 기법을 제안합니다.



### Multi-Agent Continuous Control with Generative Flow Networks (https://arxiv.org/abs/2408.06920)
- **What's New**: 본 논문에서는 다중 에이전트 연속 제어 문제에 대한 GFlowNets 확장을 목표로 하는 MACFN(Multi-Agent generative Continuous Flow Networks)를 제안합니다. MACFN은 각 에이전트가 분산된 개별 흐름 기반 정책을 중앙 집중식으로 학습하는 CTDE(Centralized Training with Decentralized Execution) 패러다임을 사용합니다. 중앙 집중식 훈련 과정에서 MACFN은 연속 흐름 분해 네트워크를 구축하여 전역 보상 정보를 활용하여 공동 흐름 함수를 에이전트별 흐름 함수로 분리합니다. 이를 통해 에이전트는 자신의 지역 흐름에만 의존하여 분산 방식으로 결정을 내릴 수 있으며, 에이전트 수 증가에 따른 차원의 저주 문제를 해결합니다. 또한, MACFN은 샘플링 기반 접근 방식을 사용하여 연속 흐름 일치에서의 적분을 근사화하고, 추정된 부모 노드의 예측 오류 하에서 제안된 샘플링 알고리즘의 수렴을 이론적으로 분석합니다.



### Automatic Feature Recognition and Dimensional Attributes Extraction From CAD Models for Hybrid Additive-Subtractive Manufacturing (https://arxiv.org/abs/2408.06891)
Comments:
          10 pages, 12 figures. This paper has been accepted for presentation at the ASME IDETC-CIE 2024 conference

- **What's New**: 본 논문은 CAD 모델에서 가산 및 감산 제조 공정에 모두 관련된 기능을 인식하는 새로운 접근 방식을 제시합니다. 기존의 AFR (Automatic Feature Recognition) 방법은 구멍, 필렛, 챔퍼, 포켓, 슬롯과 같은 감산 (기계 가공) 기능 식별에 중점을 두었지만, 가산 제조에 관련된 기능을 인식하는 데는 실패했습니다. 또한, 기존 방법은 효과적인 제조 공정 계획에 필수적인 기하학적 치수와 방향을 정확하게 추출하는 데 부족했습니다. 이 논문에서는 Python Open Cascade를 통해 가산 및 감산 가공에 관련된 기능을 포함하는 합성 CAD 데이터 세트를 생성하는 새로운 방법을 제시합니다. 계층적 그래프 합성곱 신경망 (HGCNN) 모델은 합성 CAD 데이터 세트 내에서 복합 가산-감산 기능을 정확하게 식별하기 위해 구현되었습니다. 제안된 방법론의 주요 참신성과 기여는 다양한 제조 기능을 인식하고, 치수, 방향 및 스톡 크기를 정확하게 추출할 수 있다는 것입니다.



### Decision-Focused Learning to Predict Action Costs for Planning (https://arxiv.org/abs/2408.06876)
- **What's New**: 이 논문은 자동 계획 분야에서 **의사 결정 중심 학습(Decision-Focused Learning, DFL)**을 사용하여 **액션 비용(action cost)**을 예측하는 최초의 연구입니다. DFL은 예측 정확도 대신 문제 해결 품질을 최적화하는 방식으로 조합 최적화 문제의 매개변수를 예측하는 데 성공적인 접근 방식입니다. 이 논문에서는 자동 계획에서 DFL을 구현하는 과제를 살펴보고, 특히 부정적인 액션 비용을 처리하고 학습 과정을 가속화하는 새로운 방법을 제안합니다.

- **Technical Details**: 자동 계획 문제에서 액션 비용은 종종 다양한 요소에 따라 달라지기 때문에 예측하기 어렵습니다. 예를 들어, 교통 네트워크에서 도로 구간을 통과하는 데 걸리는 시간은 실시간 교통 상황, 날씨, 요일 등의 요소에 의존합니다. DFL을 사용하여 액션 비용을 예측하려면 두 가지 주요 과제를 해결해야 합니다.
1. **부정적인 액션 비용 처리**: DFL은 경사 하강법(gradient descent)을 통해 학습하는 동안 계획 시스템을 호출하는데, 이때 부정적인 액션 비용이 발생할 수 있습니다. 이 문제를 해결하기 위해 이 논문은 부정적인 값을 수정하고 학습 과정에서 부정적인 값에 대한 페널티를 부과하는 새로운 방법을 제안합니다.
2. **계획 시간 최소화**: DFL은 훈련 과정에서 계획 시스템을 반복적으로 호출해야 하므로 계산 비용이 많이 소모될 수 있습니다. 이 문제를 완화하기 위해 이 논문은 계획 시스템에서 사용되는 다양한 기술을 활용하여 최적화된 계획 대신 근사적인 계획을 계산합니다. 또한, 캐싱 메커니즘을 도입하여 학습 중에 계획을 재사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 MSE(Mean Squared Error)를 최소화하는 것만을 목표로 하는 예측 방법에 비해 더 나은 계획 결과를 생성하는 것으로 나타났습니다. 또한, 캐싱을 통해 훈련 시간이 크게 단축되는 것을 확인했습니다. 이러한 결과는 DFL이 자동 계획에서 액션 비용을 예측하는 효과적인 방법임을 보여줍니다.



### Advancing Interactive Explainable AI via Belief Change Theory (https://arxiv.org/abs/2408.06875)
Comments:
          9 pages. To be published at KR 2024

- **What's New**: 이 논문은 AI 모델의 설명 가능성 (XAI)을 높이기 위해 사용자 피드백을 통한 대화형 XAI 시스템을 위한 새로운 틀을 제시합니다. 특히, 벨리프 변화 이론 (Belief Change Theory)을 사용하여 데이터 기반 분류기의 논리적 표현에 대한 사용자 피드백을 통합하는 연산자를 모델링합니다. 이 접근 방식은 대화형 설명을 체계적으로 개발하는 방법론을 제공하며, 설명 가능성과 책임성을 강화합니다.



### Causal Agent based on Large Language Mod (https://arxiv.org/abs/2408.06849)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)이 인과적 추론(causal reasoning)을 수행할 수 있도록 인과적 에이전트(Causal Agent)라는 새로운 프레임워크를 제시합니다. 인과적 에이전트는 인과적 방법(causal methods)을 활용하여 LLM이 인과 관계를 이해하고 활용할 수 있도록 지원합니다. 이는 LLM이 인과적 문제를 해결하는 능력을 향상시키고 다양한 분야에서 활용 가능성을 확대할 수 있는 중요한 발전입니다.



### Personalized Dynamic Difficulty Adjustment -- Imitation Learning Meets Reinforcement Learning (https://arxiv.org/abs/2408.06818)
Comments:
          2 pages, the code to our demo can be found here: this https URL

- **What's New**: 본 논문에서는 게임의 난이도를 플레이어의 현재 행동에 따라 조정하는 **개인화된 동적 난이도 조정(PDDA)** 기법을 제안합니다. 이는 플레이어의 행동을 모방하는 **모방 학습 에이전트(Imitation Learning Agent)**와 이 에이전트를 이기도록 훈련된 **강화 학습 에이전트(Reinforcement Learning Agent)**를 결합하여 구현됩니다.



### MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty (https://arxiv.org/abs/2408.06816)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 신뢰성을 향상시키기 위해 데이터 불확실성(aleatoric uncertainty) 하에서의 불확실성 정량화를 조사합니다. 이를 위해 다중 답변 질의응답 데이터셋(MAQA)을 새롭게 제안하여 여러 답변을 요구하는 질문을 통해 데이터 불확실성을 도입합니다. 또한, 다양한 화이트박스 및 블랙박스 LLM에 대해 5가지 불확실성 정량화 방법을 평가하여 데이터 불확실성 하에서의 LLM 신뢰성을 측정합니다.



### Enhancing Visual Dialog State Tracking through Iterative Object-Entity Alignment in Multi-Round Conversations (https://arxiv.org/abs/2408.06725)
Comments:
          This article has been accepted in CAAI Transactions on Intelligence Technology! Article ID: CIT2_12370, Article DOI: https://doi.org/10.1049/cit2.12370

- **What's New**: This paper introduces a new model, Multi-round Dialogue State Tracking (MDST), for Visual Dialog (VD). MDST addresses the limitation of previous VD models that treat the entire dialog history as a simple text input, by leveraging the dialogue state learned from the dialog history to answer questions. MDST captures each round of dialog history, constructing internal dialogue state representations defined as 2-tuples of vision-language representations, which effectively ground the current question, enabling the generation of more accurate answers.



### Simple but Effective Compound Geometric Operations for Temporal Knowledge Graph Completion (https://arxiv.org/abs/2408.06603)
- **What's New**: TCompoundE는 시간별 지식 그래프 (TKG)에서 누락된 사실을 추론하는 시간별 지식 그래프 완성을 위한 새로운 방법입니다. 기존 방법들은 일반적으로 지식을 연속 벡터 공간에 임베딩하고 기하학적 연산을 적용하여 TKG에서 잠재적인 패턴을 학습합니다. 그러나 이러한 방법들은 하나의 연산만 사용하여 TKG에 존재하는 복잡한 시간적 역동성을 포착하는 데 한계가 있을 수 있습니다. TCompoundE는 시간별 및 관계별 연산을 포함하는 두 가지 기하학적 연산으로 특별히 설계된 간단하지만 효과적인 방법입니다. 이 논문에서는 TCompoundE가 다양한 관계 패턴을 인코딩할 수 있는 능력을 보여주는 수학적 증명을 제공합니다. 실험 결과는 제안된 모델이 기존의 TKG 임베딩 모델보다 훨씬 뛰어난 성능을 보여줍니다.



### Value of Information and Reward Specification in Active Inference and POMDPs (https://arxiv.org/abs/2408.06542)
- **What's New**: 본 논문은 활성 추론(Active Inference)의 핵심 개념인 예상 자유 에너지(EFE)를 탐구하며, EFE가 보상 기반 강화 학습(RL) 에이전트와 비교했을 때 얼마나 효율적으로 의사 결정을 수행하는지 분석합니다. 특히 EFE가 정보 가치(Information Value)를 통해 베이즈 최적 RL 정책을 근사하는 것을 보여줍니다.



### Towards Autonomous Agents: Adaptive-planning, Reasoning, and Acting in Language Models (https://arxiv.org/abs/2408.06458)
- **What's New**: 이 연구는 자율적인 의사 결정 언어 에이전트를 구축하기 위한 새로운 인 컨텍스트 학습 알고리즘을 제안합니다. 언어 에이전트는 작업이 실패할 때마다 스스로를 수정하여 동일한 작업을 지속적으로 해결하려고 합니다. 연구에서 사용된 언어 에이전트는 텍스트 기반 게임 환경에서 작업을 해결할 수 있는 능력을 보여줍니다. 연구 결과, 제안된 방법을 사용하는 gemma-2-9b-it 언어 모델은 처음 시도에서 실패한 여섯 가지 작업 중 두 가지를 성공적으로 완료할 수 있었습니다. 이는 자기 수정을 통한 단일 언어 모델의 문제 해결 능력을 향상시키는 데 있어 제안된 접근 방식의 효과를 강조하며, 더 발전된 자율 에이전트를 위한 길을 열어줍니다. 코드는 [링크](https://this https URL) 에서 공개적으로 이용 가능합니다.

- **Technical Details**: 이 연구는 ReAct (Reasoning and Acting) 프롬프팅을 기반으로 하며, 언어 모델이 텍스트 기반 게임 환경에서 작업을 해결하는 데 필요한 추론 과정을 학습하도록 돕습니다. ReAct 프롬프팅은 언어 모델에 사전 언어 설명을 제공하여 다양한 언어 추론 및 의사 결정 작업을 해결하기 위한 추론을 안내하고 외부 세계로부터 피드백을 받아 이러한 추론을 조정합니다. 본 연구의 핵심은 '자기 반성' (self-reflection) 단계를 추가하여 언어 에이전트가 이전 실패로부터 배우고 자체 정책을 개선할 수 있도록 하는 것입니다. 기존의 ReAct 프롬프팅과 달리, 이 연구는 하나의 언어 모델만을 사용하여 자율적인 언어 에이전트를 구축합니다. 이는 기존의 ReAct 프롬프팅이 복잡한 작업을 해결하는 데 어려움을 겪는다는 점을 해결하기 위한 시도입니다.

- **Performance Highlights**: 제안된 방법은 ALFWorld 환경에서 14가지 다른 의사 결정 작업에서, '자기 반성' 단계 없이 ReAct 프롬프팅만을 사용하는 에이전트가 해결할 수 없는 작업을 성공적으로 완료할 수 있었습니다. 이는 '자기 반성' 단계가 언어 에이전트의 성능을 향상시키는 데 효과적임을 보여줍니다. 본 연구는 '자기 반성'을 통해 단일 언어 모델의 자율성과 문제 해결 능력을 향상시킬 수 있음을 입증합니다. 또한, 제안된 방법은 기존의 ReAct 프롬프팅의 한계를 극복하고 더 복잡한 작업을 해결할 수 있는 잠재력을 가지고 있습니다.



### Diversity Empowers Intelligence: Integrating Expertise of Software Engineering Agents (https://arxiv.org/abs/2408.07060)
- **What's New**: DEI (Diversity Empowered Intelligence), a new framework for enhancing software engineering (SWE) agent performance by leveraging their diverse expertise, is introduced. DEI functions as a meta-module that enables collaboration among agents, leading to improved problem-solving.

- **Technical Details**: DEI integrates with existing SWE agent frameworks, creating a multi-agent ensemble system with a re-ranking pipeline. This pipeline utilizes a multi-stage rating system to identify the most effective agent for resolving a specific issue. DEI employs a committee of agents to enhance performance, surpassing the capabilities of individual agents.

- **Performance Highlights**: DEI significantly improves the issue resolution rate on SWE-Bench Lite, a benchmark for evaluating SWE agent performance. For instance, a group of open-source agents, with a maximum individual resolve rate of 27.3%, achieved a 34.3% resolve rate with DEI, representing a 25% improvement. The best-performing group, guided by DEI, achieved a 55% resolve rate, securing the highest ranking on SWE-Bench Lite.



### Model Counting in the Wild (https://arxiv.org/abs/2408.07059)
Comments:
          Full version of conference paper accepted at KR 2024

- **What's New**: 이 논문은 다양한 응용 분야에서 모델 카운터의 확장성을 조사하여 실제 환경에서 모델 카운터의 성능이 어떻게 다르게 나타나는지 보여줍니다. 이 연구는 11개의 애플리케이션 도메인에서 수집한 2262개의 벤치마크 세트를 사용하여 6개의 최첨단 모델 카운터를 평가했습니다.

- **Technical Details**: 이 논문은 모델 카운터의 성능에 영향을 미치는 두 가지 주요 매개변수인 트리 너비(treewidth)와 독립적 지지 크기(independent support size)를 조사했습니다. 또한, 비-프로젝션된 인스턴스(non-projected instance)에 대해서는 SharpSAT-TD가 최고의 성능을 보였고, 프로젝션된 인스턴스(projected instance)에 대해서는 ApproxMC가 최고의 성능을 보였습니다.

- **Performance Highlights**: 결과에 따르면, 컴파일 기반(compilation-based) 모델 카운터는 트리 너비가 낮은 벤치마크 세트에서 뛰어난 성능을 보인 반면, 해싱 기반(hashing-based) 모델 카운터는 독립적 지지 크기가 큰 벤치마크 세트에서 뛰어난 성능을 보였습니다. 이러한 상호 보완적인 특성을 활용하여 가상 최상의 솔버(virtual best solver)를 구축한 결과, 2262개 벤치마크 중 2106개를 해결할 수 있었습니다.



### A Survey on Model MoErging: Recycling and Routing Among Specialized Experts for Collaborative Learning (https://arxiv.org/abs/2408.07057)
Comments:
          26 pages

- **What's New**: 본 논문은 **모델 모어징 (MoErging)**, 즉 특정 도메인이나 작업에 특화된 사전 훈련된 전문가 모델을 재활용하여 성능이나 일반화 능력을 향상시키는 새로운 패러다임을 제시하는 최신 연구 동향을 소개합니다. MoErging은 기존의 **혼합 전문가 모델 (MoE)**과 **모델 병합 (Model Merging)** 접근 방식과 유사하지만, 전문가 모델이 중앙 집중식으로 훈련되는 것이 아니라 분산된 기여자들에 의해 독립적으로 훈련되고 공유되는 점이 특징입니다. 이 논문은 MoErging 방법을 정확히 비교 분석하고, **전문가 (experts)**, **라우팅 (routing)**, **애플리케이션 (application)**의 세 가지 범주로 MoErging 방법을 분류하는 새로운 분류 체계를 제시합니다. 또한 MoErging과 관련된 연구 분야, 이러한 접근 방식을 지원하는 도구, 미래 연구 방향을 논의합니다.



### The News Comment Gap and Algorithmic Agenda Setting in Online Forums (https://arxiv.org/abs/2408.07052)
- **What's New**: 본 논문은 뉴스 기사에 대한 독자들의 댓글에 대한 뉴스 기자와 독자 간의 차이, 즉 "뉴스 댓글 격차"(News Comment Gap)를 분석합니다. 이는 뉴스 가치에 대한 기자와 독자 간 차이인 "뉴스 격차"(News Gap)를 확장한 개념입니다. 특히 다양한 댓글 순위 알고리즘이 독자와 기자에게 어떻게 다른 방식으로 댓글을 보여주는지 분석하여 뉴스 토론의 대표성을 어떻게 형성하는지 조사합니다.

- **Technical Details**: 오스트리아 신문 Der Standard의 120만 개 이상의 댓글 데이터를 분석하여 기자와 독자의 댓글 선호도를 비교했습니다. 선호도를 분석하기 위해 "편집자 선택"(Editors’ Picks)과 사용자 투표를 기반으로 회귀 및 분류 모델을 사용했습니다. 또한 다양한 댓글 순위 알고리즘의 성능을 평가하기 위해 새로운 "특징 지향 순위 유틸리티 지표"(Feature-Oriented Ranking Utility Metric, FORUM)를 도입했습니다.

- **Performance Highlights**: 분석 결과, 기자들은 긍정적이고 시의적절하며 복잡하고 직접적인 댓글을 선호하는 반면 독자들은 기사 내용과 유사하며 엘리트 저자의 댓글을 선호하는 것으로 나타났습니다. 또한 FORUM을 통해 감정, 주제 관련성, 어휘 다양성, 가독성 등 다양한 측면에서 댓글 순위 알고리즘이 서로 다른 결과를 보여준다는 것을 발견했습니다. 즉, 댓글 순위 알고리즘은 뉴스 토론의 방향을 크게 바꿀 수 있는 힘을 가지고 있으며, 기자들은 큐레이션과 알고리즘을 통해 토론에 큰 영향력을 행사할 수 있습니다.



### KAN You See It? KANs and Sentinel for Effective and Explainable Crop Field Segmentation (https://arxiv.org/abs/2408.07040)
Comments:
          Accepted at ECCV 2024 CVPPA Workshop

- **What's New**: 본 논문에서는 작물 필드 분할에 Kolmogorov-Arnold 네트워크(KANs) 기반 U-Net 아키텍처(U-KAN)를 적용하고 성능 및 설명 가능성을 분석한 최초의 연구입니다. 기존 U-Net 모델 대비 U-KAN 모델은 더 적은 GFLOPs에서 IoU(Intersection-Over-Union) 지표에서 2% 향상된 성능을 보였습니다. 또한 기울기 기반 설명 기법을 통해 U-KAN 예측의 타당성을 확인하고, 모델이 작물 영역 자체보다는 경계에 더 집중하는 특징을 밝혀냈습니다.



### PathInsight: Instruction Tuning of Multimodal Datasets and Models for Intelligence Assisted Diagnosis in Histopathology (https://arxiv.org/abs/2408.07037)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문은 암 진단을 위한 혁신적인 멀티모달 (multimodal) 대규모 언어 모델(LLM)인 PathEnhanceDS를 제시합니다. PathEnhanceDS는 병리학적 이미지 분류, 캡션 생성, 질문 답변 등 다양한 작업을 수행할 수 있는 45,000개의 데이터셋으로 구성됩니다. 연구팀은 LLaVA, Qwen-VL, InternLM과 같은 기존 모델을 PathEnhanceDS로 미세 조정하여 병리학 영역에서 더욱 효과적인 성능을 발휘하도록 했습니다. 특히, 이미지 캡션 생성, 분류 및 질문 답변 작업에 대한 정량적 및 질적 평가를 통해 모델의 우수성을 확인했습니다.

- **Technical Details**: PathEnhanceDS는 병리학 분야의 다양한 데이터 소스를 결합하여 구성되었습니다. 이 데이터셋은 이미지 캡션 생성, 분류, 이미지 기반 질문 답변(VQA) 및 대화를 포함한 여러 작업을 수행할 수 있도록 설계되었습니다. 특히, Pathologist-level Dataset, OpenPath, PCam, CRC-VAL-HE-7K, PathVQA, LLaVA-Med-Instruct 데이터셋이 통합되었습니다. 연구팀은 LoRA와 전체 파라미터 미세 조정을 포함한 다양한 방법을 사용하여 멀티모달 모델을 PathEnhanceDS로 미세 조정했습니다.

- **Performance Highlights**: PathEnhanceDS로 미세 조정된 모델은 이미지 캡션 생성, 분류 및 질문 답변 작업에서 우수한 성능을 보였습니다. 특히, 경험이 풍부한 병리학자의 질적 평가 결과, 모델은 병리학적 질문에 대한 답변의 정확성과 신뢰성이 높다는 것을 확인했습니다. 이 연구는 멀티모달 LLM의 잠재력을 보여주며, 향후 병리학 교육 및 진단 분야에 크게 기여할 수 있을 것으로 기대됩니다.



### Defining and Measuring Disentanglement for non-Independent Factors of Variation (https://arxiv.org/abs/2408.07016)
- **What's New**: 이 논문에서는 정보 이론을 기반으로 한 새로운 **disentanglement** (해석 가능한 표현) 정의를 제시합니다. 이 정의는 데이터에서 **factor of variation** (변이 요인)이 독립적이지 않은 경우에도 유효합니다. 기존의 정의와 측정 방법은 일반적으로 변이 요인이 독립적이라고 가정하지만, 실제 세계에서는 이러한 가정이 항상 성립하는 것은 아닙니다. 새로운 정의는 **Information Bottleneck Method** (정보 병목 방법)과 관련되어 있으며, 변이 요인이 독립적이지 않은 경우에도 해석 가능한 표현을 측정할 수 있는 방법을 제시합니다.  실험 결과는 제안된 방법이 다른 방법들보다 비독립적인 변이 요인의 경우 **disentanglement** (해석 가능한 표현)을 더 정확하게 측정할 수 있음을 보여줍니다.



### Casper: Prompt Sanitization for Protecting User Privacy in Web-Based Large Language Models (https://arxiv.org/abs/2408.07004)
- **What's New**: Casper, a lightweight browser extension, protects user privacy in online Large Language Model (LLM) services by sanitizing prompts before sending them to cloud-based LLMs.



### Generative AI for automatic topic labelling (https://arxiv.org/abs/2408.07003)
Comments:
          10 pages, 1 figure

- **What's New**: 본 논문은 주제 모델링 (Topic Modeling) 결과 해석을 위한 라벨링에 대규모 언어 모델 (LLM)의 효용성을 평가합니다. 구체적으로 Flan, GPT-4o, GPT-4 mini 모델을 활용하여, 스위스 생물학 교수 465명이 2008년부터 2020년까지 발표한 과학 논문 데이터셋 (34,797개)에서 추출된 주제들을 라벨링합니다.



### SpectralGaussians: Semantic, spectral 3D Gaussian splatting for multi-spectral scene representation, visualization and analysis (https://arxiv.org/abs/2408.06975)
- **What's New**: This paper presents a novel cross-spectral rendering framework based on 3D Gaussian Splatting (3DGS) that generates realistic and semantically meaningful splats from registered multi-view spectrum and segmentation maps. This framework enhances the representation of scenes with multiple spectra, providing insights into the underlying materials and segmentation.

- **Technical Details**: The paper introduces an improved physically-based rendering approach for Gaussian splats, estimating reflectance and lights per spectra, thereby enhancing accuracy and realism. It also proposes a new spectral 3DGS approach that extends the scene representation based on 3DGS to generate realistic and semantically meaningful splats from registered multi-view spectrum and segmentation maps.

- **Performance Highlights**: The proposed approach outperforms other recent learning-based spectral scene representation approaches like XNeRF and SpectralNeRF, as well as other non-spectral state-of-the-art learning-based approaches. It also showcases the potential of spectral scene understanding for precise scene editing techniques like style transfer, inpainting, and removal.

- **Datasets**: The authors generated two synthetic spectral datasets by extending the shiny Blender dataset and the synthetic NERF dataset in terms of their spectral properties. These datasets are expected to serve as valuable resources for researchers and practitioners, offering a diverse range of spectral scenes for experimentation, evaluation, and advancements in the field of image-based/multi-view spectral rendering.



### Neural Speech and Audio Coding (https://arxiv.org/abs/2408.06954)
Comments:
          Accepted for publication in IEEE Signal Processing Magazine

- **What's New**: 이 논문은 신경 음성 및 오디오 코딩 시스템에서 모델 기반 및 데이터 기반 접근 방식을 통합하는 것을 탐구합니다. 이 논문은 음성 및 오디오 코덱의 주관적 평가 과정에서 발생하는 과제를 강조하고, 종종 모델 기반 방법의 성능과 일치하기 위해 비효율적으로 큰 아키텍처를 요구하는 순전히 데이터 기반 접근 방식의 한계에 대해 논의합니다. 이 연구는 하이브리드 시스템을 실현 가능한 솔루션으로 제시하여 신중하게 선택된 설계 개선을 통해 기존 코덱의 성능을 크게 향상시킵니다. 구체적으로, 기존 코덱의 출력을 후처리하도록 설계된 신경망 기반 신호 강화기를 소개하며, 오토인코더 기반 엔드 투 엔드 모델 및 선형 예측 코딩(LPC)을 신경망과 결합한 LPCNet--하이브리드 시스템을 소개합니다. 또한, 이 논문은 사용자 지정 특징 공간(TF-Codec) 또는 사전 정의된 변환 도메인(MDCTNet) 내에서 작동하는 예측 모델을 살펴보고 엔드 투 엔드 신경 오디오 코덱을 훈련하기 위해 심리음향적으로 보정된 손실 함수를 사용하는 것을 조사합니다. 이러한 조사를 통해 이 논문은 하이브리드 시스템이 전통적인 모델 기반 접근 방식과 현대의 데이터 기반 기술 간의 간극을 메워 음성 및 오디오 코딩 분야를 발전시킬 잠재력을 입증합니다.

- **Technical Details**: 이 논문은 신경 음성 및 오디오 코딩(NSAC) 시스템에서 모델 기반 및 데이터 기반 접근 방식을 통합합니다. 이 논문은 기존 코덱의 성능을 향상시키기 위해 신중하게 선택된 설계 개선을 통해 기존 코덱의 성능을 크게 향상시키는 하이브리드 시스템을 소개합니다. 구체적으로 이 논문은 다음과 같은 하이브리드 시스템을 소개합니다.

* **신경망 기반 신호 강화기:** 기존 코덱의 출력을 후처리하도록 설계된 신경망 기반 신호 강화기.
* **오토인코더 기반 엔드 투 엔드 모델:** 오토인코더 기반 엔드 투 엔드 모델은 입력 신호를 직접 복구합니다.
* **LPCNet:** 선형 예측 코딩(LPC)을 신경망과 결합한 하이브리드 시스템.

이 논문은 또한 사용자 지정 특징 공간(TF-Codec) 또는 사전 정의된 변환 도메인(MDCTNet) 내에서 작동하는 예측 모델을 살펴보고 엔드 투 엔드 신경 오디오 코덱을 훈련하기 위해 심리음향적으로 보정된 손실 함수를 사용하는 것을 조사합니다.

- **Performance Highlights**: 하이브리드 시스템은 기존 코덱의 성능을 크게 향상시키는 것으로 나타났습니다. 예를 들어, LPCNet은 기존 LPC 코덱보다 훨씬 나은 음성 품질을 제공하는 것으로 나타났습니다. 또한, 심리음향적으로 보정된 손실 함수를 사용하여 훈련된 엔드 투 엔드 신경 오디오 코덱은 기존 코덱보다 훨씬 더 나은 주관적 음질을 제공하는 것으로 나타났습니다.



### Heavy-Ball Momentum Accelerated Actor-Critic With Function Approximation (https://arxiv.org/abs/2408.06945)
- **What's New**: 이 논문은 **모멘텀**을 사용하여 액터-크리틱(AC) 알고리즘의 수렴 속도를 개선하는 새로운 방법을 제안합니다. 특히, 헤비볼(HB) 모멘텀을 크리틱 재귀에 통합하여 **HB-A2C(Heavy-Ball based Advantage Actor-Critic)** 알고리즘을 제안합니다. 이 알고리즘은 크리틱이 선형 함수로 매개변수화되어 있고, 샘플 궤적이 마코브 의사 결정 과정(MDP)을 따를 때 가속화 능력을 보여줍니다.



### The advantages of context specific language models: the case of the Erasmian Language Mod (https://arxiv.org/abs/2408.06931)
Comments:
          12 pages, 3 figures, 1 table

- **What's New**: 본 논문은 기존 대규모 언어 모델(LLM)의 한계를 극복하기 위해 Erasmus 대학교 로테르담(Erasmus University Rotterdam)에 맞춰 특화된 소규모 언어 모델인 Erasmian Language Model (ELM)을 제시합니다. ELM은 9억 개의 파라미터를 가지며, Erasmus 대학교의 데이터로 사전 학습 및 미세 조정되었습니다. 특정 주제에 대한 전문성을 갖춘 소규모 모델을 활용함으로써 계산 자원 및 에너지 소비, 개인 정보 보호와 같은 문제를 완화하고자 합니다.



### Diagnosis extraction from unstructured Dutch echocardiogram reports using span- and document-level characteristic classification (https://arxiv.org/abs/2408.06930)
Comments:
          28 pages, 5 figures

- **What's New**: 본 연구는 네덜란드의 대규모 대학 병원인 UMCU에서 수집된 115,692개의 비정형 심장초음파 보고서를 사용하여 자동화된 스팬 및 문서 수준 진단 추출의 가능성을 조사했습니다. 이 연구는 심장초음파 보고서에서 11가지 주요 심장 특징의 발생 및 심각도를 자동으로 분류할 수 있는 새로운 딥러닝 기반 모델을 제시합니다.



### Temporal Variability and Multi-Viewed Self-Supervised Representations to Tackle the ASVspoof5 Deepfake Challeng (https://arxiv.org/abs/2408.06922)
- **What's New**: ASVspoof5, the latest iteration of the ASVspoof challenge, focuses on deepfake audio detection. The paper investigates various countermeasures (CMs) for open-domain audio deepfake detection, tackling the ASVspoof5 Track 1 open condition. They propose a novel data augmentation technique called Frequency Mask (Freqmask) to address the high-frequency gaps present in the ASVspoof5 dataset. This method masks specific frequency bands to improve CM robustness.

- **Technical Details**: The paper explores data expansion, data augmentation, and self-supervised learning (SSL) features for deepfake audio detection. They experiment with various datasets for co-training, including ASVspoof2019LA, MLAAD, and Codecfake. For data augmentation, they utilize traditional methods like low-pass filtering, high-pass filtering, pitch shifting, and time stretching, along with MUSAN and RIR noise augmentation. The Frequency Mask (Freqmask) method is proposed to address the high-frequency gap issue by randomly masking frequency bands in the spectrogram. They investigate the performance of various SSL features, such as WavLM and HuBERT, by freezing specific layers to improve efficiency.

- **Performance Highlights**: Their proposed approach achieves a minDCF of 0.0158 and an EER of 0.55% on the ASVspoof5 evaluation progress set. This performance is achieved by integrating seven CM methods at different scales through logits score fusion, considering both temporal information and diverse SSL categorical perspectives.



### Heterogeneous Space Fusion and Dual-Dimension Attention: A New Paradigm for Speech Enhancemen (https://arxiv.org/abs/2408.06911)
Comments:
          Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2024

- **What's New**: 본 논문은 시끄러운 환경에서 음성 신호의 선명도와 품질을 향상시키는 새로운 음성 향상 프레임워크인 HFSDA를 제안합니다. 이 프레임워크는 이종 공간 특징을 통합하고 이중 차원 주의 메커니즘을 도입하여 효율적인 음성 특징에 집중합니다. HFSDA는 자기 지도 학습 임베딩과 STFT 스펙트로그램 특징을 활용하여 고수준의 의미 정보와 세부적인 스펙트럼 데이터를 모두 포착합니다. 또한 ODConv 기술을 사용하여 스펙트로그램 입력 분기에서 다차원에 걸쳐 중요한 정보를 추출하고 통합합니다. 더 나아가 Conformer 모델을 개선하여 시간 차원뿐만 아니라 스펙트럼 영역에서도 특징 추출 기능을 향상시킵니다.



### VNet: A GAN-based Multi-Tier Discriminator Network for Speech Synthesis Vocoders (https://arxiv.org/abs/2408.06906)
Comments:
          Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2024

- **What's New**: VNet, a 새로운 GAN-기반 음성 합성 모델 제안, 전체 주파수 대역 멜 스펙트로그램 정보를 입력으로 사용하며, 다단계 판별기(Multi-Tier Discriminator, MTD)를 도입하여 고해상도 신호 생성



### Entendre, a Social Bot Detection Tool for Niche, Fringe, and Extreme Social Media (https://arxiv.org/abs/2408.06900)
Comments:
          6 pages

- **What's New**: Entendre, an open-access, scalable, and platform-agnostic bot detection framework for social media platforms, is introduced. It addresses the gap in bot detection tools for niche platforms like Parler, Gab, and Gettr, which often lack moderation and are prone to misinformation.

- **Technical Details**: Entendre utilizes a random forest classification approach to detect bots. It leverages common data features across platforms, such as user posts, approval, and bio, to achieve platform-agnostic detection. This allows for rapid extensibility but may compromise accuracy compared to platform-specific models.

- **Performance Highlights**: A study on Parler using Entendre identified 1,916 unique users (4.99%) exhibiting bot-like behavior. These bots amplified influential rhetoric and hashtags, highlighting their impact on the network.

- **Applications**: Entendre can be used to monitor bot activity on various platforms, especially those lacking moderation, and help mitigate the spread of misinformation and hate speech.



### BMFT: Achieving Fairness via Bias-based Weight Masking Fine-tuning (https://arxiv.org/abs/2408.06890)
Comments:
          Accepted by MICCAI 2024 FAIMI Workshop Oral

- **What's New**: 본 논문은 훈련 데이터에 접근하지 않고도 기존 모델의 공정성을 향상시키는 새로운 사후 처리 방법인 **Bias-based Weight Masking Fine-Tuning (BMFT)**를 제안합니다. BMFT는 편향된 예측에 가장 크게 기여하는 가중치를 효과적으로 식별하는 모델 매개변수 마스크를 생성합니다. BMFT는 **특징 추출기**를 먼저 조정한 다음 재초기화된 **분류 계층**을 조정하는 **2단계 탈 편향 전략**을 사용하여 차별적 성능을 유지합니다.



### Generative AI Tools in Academic Research: Applications and Implications for Qualitative and Quantitative Research Methodologies (https://arxiv.org/abs/2408.06872)
- **What's New**: 본 연구는 생성형 인공지능(GenAI)이 질적 및 양적 데이터 분석에 적용되는 방식을 중심으로 학술 연구에 미치는 영향을 조사합니다. GenAI 도구가 빠르게 발전하면서 연구 생산성을 높이고 복잡한 분석 과정을 민주화할 수 있는 새로운 가능성을 제공합니다. 그러나 학술 관행에 통합되는 것은 연구 무결성 및 보안, 저자, 그리고 학술적 작업의 변화하는 본질에 관한 중요한 질문을 제기합니다. 본 연구는 현재 기능과 잠재적인 미래 적용을 살펴봄으로써 연구자들이 GenAI 도구를 책임감 있고 윤리적으로 사용할 수 있는 방법에 대한 통찰력을 제공합니다. 다양한 연구 방법론에서 GenAI 적용을 보여주는 사례 연구를 제시하고, AI 지원 연구에서 재현성 및 일관성의 과제를 논의하며, 학계에서 AI 통합이 증가함에 따른 윤리적 함의를 고려합니다. 본 연구는 GenAI의 질적 및 양적 적용을 모두 살펴보고, 전사, 코딩, 주제 분석, 시각적 분석 및 통계 분석을 위한 도구를 강조합니다. 이러한 문제점을 해결함으로써, 우리는 AI가 학술 연구의 미래를 형성하는 데 있어 역할에 대한 지속적인 논의에 기여하고, 빠르게 진화하는 AI 지원 연구 도구 및 연구 환경을 탐구하는 연구자들에게 지침을 제공하고자 합니다.



### BSS-CFFMA: Cross-Domain Feature Fusion and Multi-Attention Speech Enhancement Network based on Self-Supervised Embedding (https://arxiv.org/abs/2408.06851)
Comments:
          Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2024

- **What's New**: This research introduces BSS-CFFMA, a novel speech enhancement network that effectively leverages self-supervised embeddings. The network combines a multi-scale cross-domain feature fusion (MSCFF) block and a residual hybrid multi-attention (RHMA) block. The MSCFF block integrates cross-domain features to extract rich acoustic information, while the RHMA block utilizes multiple attention mechanisms to capture diverse attention representations for high-quality speech estimation.

- **Technical Details**: BSS-CFFMA utilizes self-supervised learning (SSL) features extracted from Wav2vec2.0 and WavLM models, along with spectrograms, as input. The MSCFF block fuses these features at different scales to enhance information extraction. The RHMA block employs three distinct attention modules: a selective channel-time attention fusion module (SCTA) with self-attention, a convolutional attention module (CA), and a multi-head self-attention module (MHSA), enabling the network to capture varied attention representations.

- **Performance Highlights**: BSS-CFFMA achieved state-of-the-art (SOTA) performance on the VoiceBank-DEMAND dataset, surpassing previous methods. Notably, it also demonstrates promising results on the WHAMR! dataset for tasks like denoising only, dereverberation only, and simultaneous denoising and dereverberation. The research highlights the potential of self-supervised embeddings for speech enhancement in complex scenarios.

- **Availability**: A demo implementation of BSS-CFFMA is available online.



### AI Research is not Magic, it has to be Reproducible and Responsible: Challenges in the AI field from the Perspective of its PhD Students (https://arxiv.org/abs/2408.06847)
Comments:
          8 pages, 4 figures, 1 appendix (interview questions)

- **What's New**: 본 논문은 유럽의 28명의 AI 박사 과정 학생들을 대상으로 설문 조사를 실시하여 AI 연구 분야에서 직면하는 어려움을 파악했습니다. 연구 결과는 AI 리소스(데이터 세트, 모델, 실험)의 발견 및 품질, AI 논문의 실험 재현의 어려움, 신뢰성 및 학제 간 협력 부족의 세 가지 주요 영역에서 어려움을 강조합니다.



### Efficient Search for Customized Activation Functions with Gradient Descen (https://arxiv.org/abs/2408.06820)
Comments:
          10 pages, 1 figure, excluding references and appendix

- **What's New**: 본 논문은 기존의 활성화 함수 (activation function) 를 고정적으로 사용하는 대신, 딥러닝 모델의 성능을 향상시키기 위해 활성화 함수를 자동으로 설계하는 방법을 제안합니다. 특히, 최근 발전된  Gradient-based Neural Architecture Search (NAS) 기술을 활용하여 주어진 문제에 가장 적합한 활성화 함수를 효율적으로 찾아냅니다. 이를 통해 기존의 ReLU, GELU, SiLU와 같은 활성화 함수보다 더 나은 성능을 발휘하는 새로운 활성화 함수를 찾아낼 수 있습니다.

- **Technical Details**: 본 논문에서는 활성화 함수를 설계하기 위해 다양한 기본 수학 연산들을 조합하여 이루어진 fine-grained search cell 을 제안합니다. 이 search cell 은 다양한 활성화 함수를 모델링할 수 있으며, 이를 통해 새로운 활성화 함수를 탐색할 수 있도록 합니다. 본 논문에서는 Gradient-based one-shot NAS 방법을 적용하여 활성화 함수를 검색하며, 이는 기존의 블랙박스 기반 검색 방법보다 훨씬 효율적입니다.

- **Performance Highlights**: 본 연구에서 제안된 방법은 이미지 분류, 언어 모델링 등 다양한 딥러닝 모델에서 기존 활성화 함수보다 더 나은 성능을 보여주었습니다. 또한, 발견된 활성화 함수는 유사한 모델이나 새로운 데이터셋에도 효과적으로 적용될 수 있음을 확인했습니다. 본 논문에서 제시된 자동화된 활성화 함수 설계 방법은 기존의 수동 설계 방법보다 훨씬 효율적이며, 딥러닝 모델의 성능을 향상시키기 위한 실용적인 방법입니다.



### Unmasking the Uniqueness: A Glimpse into Age-Invariant Face Recognition of Indigenous African Faces (https://arxiv.org/abs/2408.06806)
Comments:
          Keywords: Age-Invariant Face Recognition, CACD, FAGE_v2, VGGFace

- **What's New**: 이 논문은 아프리카 원주민 얼굴의 연령 불변 얼굴 인식(AIFR) 시스템을 개발하여 기존 연구에서 아프리카 민족을 잘못 대표하는 문제를 해결합니다. 연구진은 이 연구를 위해 수집한 5,000명의 아프리카 원주민 얼굴 데이터셋(FAGE_v2)에 대해 사전 훈련된 딥 러닝 모델(VGGFace)을 사용했습니다.



### Deep Learning for Speaker Identification: Architectural Insights from AB-1 Corpus Analysis and Performance Evaluation (https://arxiv.org/abs/2408.06804)
Comments:
          Resultant work from Assignment, Department of AI, University of Malta. Code available at: this https URL

- **What's New**: 본 연구는 음성 인식 (SID)에서 멜 스펙트로그램과 멜 주파수 케프스트럼 계수 (MFCC)를 특징 추출에 사용하는 6가지의 서로 다른 모델 아키텍처를 평가하여 그 성능을 분석했습니다. 특히, AB-1 코퍼스 데이터셋에서 성별과 억양 정확성을 분석하여 모델의 편향성을 평가했습니다.



### Integrating Saliency Ranking and Reinforcement Learning for Enhanced Object Detection (https://arxiv.org/abs/2408.06803)
Comments:
          Resultant work from Dissertation, Department of AI, University of Malta. Code available at: this https URL

- **What's New**: 이 연구는 강화 학습(RL) 기반의 시각적 주의 방법과 급격성 순위(saliency ranking) 기술을 결합하여 투명하고 지속 가능한 솔루션을 탐구하는 일련의 실험을 수행합니다. 이 연구는 초기 경계 상자 예측을 위해 급격성 순위를 통합하고 여러 시간 단계에 걸쳐 유한한 작업 집합을 통해 이러한 예측을 미세 조정하기 위해 RL 기법을 적용하여 RL 객체 감지 정확도를 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 다양한 이미지 특징 추출 방법을 사용하고 다양한 심층 Q-네트워크(DQN) 아키텍처 변형을 탐구하여 심층 강화 학습 기반의 현지화 에이전트 훈련을 수행합니다. 또한 이 연구는 이전 RL 접근 방식에 없는 기능인 감지된 객체를 분류하는 기능을 통합하면서, 가볍고 빠른 모델을 우선시하여 모든 단계에서 감지 파이프라인을 최적화하는 데 중점을 둡니다.

- **Performance Highlights**: Pascal VOC 2007 데이터 세트를 사용하여 이러한 훈련된 에이전트의 성능을 평가함으로써 더 빠르고 최적화된 모델이 개발되었습니다. 특히 이 연구에서 달성된 최고의 평균 정밀도(mAP)는 51.4로, 문헌에서 RL 기반 단일 객체 검출기가 설정한 벤치 마크를 능가했습니다.



### Stunned by Sleeping Beauty: How Prince Probability updates his forecast upon their fateful encounter (https://arxiv.org/abs/2408.06797)
Comments:
          12 pages, 1 figure, all comments welcome!

- **What's New**: This paper provides a simple Bayesian probability argument to support the 'Thirder' position in the Sleeping Beauty Problem. It argues that Sleeping Beauty's awareness of being awake is crucial information and should lead her to update her belief in the outcome of the coin toss being 'Heads' to 1/3.



### Robust Deep Reinforcement Learning for Inverter-based Volt-Var Control in Partially Observable Distribution Networks (https://arxiv.org/abs/2408.06776)
- **What's New**: 이 논문에서는 DRL 기반 접근 방식에서 활성 분산 네트워크(ADNs)에서의 제한적인 측정 배치로 인한 부분 관측 상태 및 알 수 없는 보상 문제를 해결하는 강력한 DRL 접근 방식을 제안합니다. 제안된 방식은 보수적인 비평과 대리 보상을 사용합니다. 보수적인 비평은 분위수 회귀 기술을 활용하여 부분 관측 상태를 기반으로 보수적인 상태-작동 값 함수를 추정하며, 이는 강력한 정책을 훈련하는 데 도움이 됩니다. 전력 손실과 전압 위반의 대리 보상은 제한적인 측정값으로 계산할 수 있습니다. 제안된 방식은 측정 가능한 전압이 있는 버스의 전체 네트워크 전력 손실과 전압 프로파일을 최적화하는 동시에 다른 버스의 전압 프로파일을 간접적으로 개선합니다. 광범위한 시뮬레이션은 루트 버스의 활성 전력 주입과 10% 미만의 버스 전압만 측정 가능한 경우에도 다양한 제한된 측정 조건에서 강력한 DRL 방식의 효과를 검증합니다.



### Exploring Domain Shift on Radar-Based 3D Object Detection Amidst Diverse Environmental Conditions (https://arxiv.org/abs/2408.06772)
Comments:
          6 pages, 5 figures, 3 tables, accepted in IEEE International Conference on Intelligent Transportation Systems (ITSC) 2024

- **What's New**: 본 논문은 자율 주행 시스템에서 4D 레이더 기반 객체 탐지에 대한 도메인 이동(domain shift) 문제를 심층적으로 분석한 연구입니다. 기존 연구와 달리, 다양한 기상 조건(예: 비, 눈, 안개)과 도로 유형(고속도로, 도시)이 3D 객체 탐지 성능에 미치는 영향을 종합적으로 살펴보았습니다. 특히, 레이더 포인트 클라우드 생성 과정에서 발생하는 도메인 이동을 명확하게 보여주는 것이 핵심입니다.



### Cross-View Geolocalization and Disaster Mapping with Street-View and VHR Satellite Imagery: A Case Study of Hurricane IAN (https://arxiv.org/abs/2408.06761)
- **What's New**: 본 논문에서는 **CVDisaster**라는 새로운 재해 매핑 프레임워크를 제안합니다. 이 프레임워크는 **Street-View Imagery (SVI)**와 **Very High-Resolution 위성 이미지**를 사용하여 **지리 위치**와 **피해 인식**을 동시에 추정합니다. CVDisaster는 **CVDisaster-Geoloc** (지리 위치 모델)와 **CVDisaster-Est** (피해 인식 모델) 두 가지 모델로 구성되어 있습니다. CVDisaster-Geoloc은 **Siamese ConvNeXt 이미지 인코더**를 사용하는 **대조 학습** 기반 모델이고, CVDisaster-Est는 **Couple Global Context Vision Transformer (CGCViT)** 기반 모델입니다.



### Evaluating Research Quality with Large Language Models: An Analysis of ChatGPT's Effectiveness with Different Settings and Inputs (https://arxiv.org/abs/2408.06752)
- **What's New**: 본 연구는 ChatGPT가 학술 논문의 질 평가에 활용될 수 있는지에 대한 연구입니다. ChatGPT의 입력으로는 전체 텍스트, 제목 및 초록, 제목만을 사용하고, ChatGPT 모델과 시스템 프롬프트가 점수에 미치는 영향을 분석했습니다.



### DiffLoRA: Generating Personalized Low-Rank Adaptation Weights with Diffusion (https://arxiv.org/abs/2408.06740)
Comments:
          9 pages,8 figures

- **What's New**: 본 논문은 'DiffLoRA'라는 새로운 개념을 도입하여 기존 텍스트-이미지 생성 모델의 개인화 문제를 해결합니다. DiffLoRA는 이미지 참조를 기반으로 저랭크 적응(LoRA) 가중치를 예측하는 확산 모델을 활용합니다. 이를 통해 별도의 학습 없이 추론 단계에서 개인화를 수행하며, 기존 모델의 생성 성능을 유지하면서도 신원 일치성을 높일 수 있습니다.

- **Technical Details**: DiffLoRA는 LoRA 가중치를 예측하기 위한 하이퍼네트워크로 확산 모델을 활용합니다. LoRA 가중치를 압축 및 재구성하기 위한 LoRA 가중치 오토인코더와, 신원 추출 기능을 강화하기 위해 혼합 전문가(MoE)에서 영감을 받은 게이트 네트워크를 도입했습니다. 또한 DiffLoRA 학습을 위한 다양한 신원의 LoRA 가중치 데이터셋을 생성하는 파이프라인을 제안했습니다.

- **Performance Highlights**: 실험 결과, DiffLoRA는 기존 최첨단 방법들보다 텍스트-이미지 일관성, 신원 일치성, 생성 품질, 추론 비용 측면에서 뛰어난 성능을 보였습니다. DiffLoRA는 추론 단계에서 추가적인 계산 비용 없이 고품질 개인화된 초상화를 생성할 수 있으며, LoRA는 일반적인 미세 조정 방법이기 때문에 다른 PEFT(Parameter-Efficient Fine-Tuning) 방법과 통합하여 다양한 작업에 적용할 수 있습니다.



### Speculations on Uncertainty and Humane Algorithms (https://arxiv.org/abs/2408.06736)
- **What's New**: 이 논문은 인공지능 윤리에 있어서 위험과 불확실성의 중요성을 강조하며, 특히 고위험 시나리오에서 불확실성을 처리하는 것이 윤리적인 인공지능 개발에 필수적이라고 주장합니다. 이 논문은 알고리즘이 불확실성, 특히 인식적 불확실성을 처리하는 것이 인공지능이 해를 입히지 않고 신뢰할 수 있게 유지하며, 그들이 내리는 결정이 인간적이도록 보장하는 데 중요하다고 강조합니다.



### Large language models can consistently generate high-quality content for election disinformation operations (https://arxiv.org/abs/2408.06731)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 선거 허위 정보 생성 능력을 평가하여 선거 허위 정보 작전의 자동화 가능성을 조사합니다. 연구는 DisElect 데이터셋을 사용하여 13개 LLM의 선거 허위 정보 생성 능력을 평가하고, LLM이 생성한 허위 정보가 인간이 작성한 것과 구별할 수 있는지 여부를 평가하는 실험을 수행합니다. 연구 결과, 대부분의 LLM은 선거 허위 정보 생성 지시에 따르며, 일부 모델은 인간 평가자가 LLM이 생성한 허위 정보와 인간이 작성한 허위 정보를 구별하는 데 어려움을 겪는 것으로 나타났습니다.



### Adaptive Data Quality Scoring Operations Framework using Drift-Aware Mechanism for Industrial Applications (https://arxiv.org/abs/2408.06724)
Comments:
          17 pages

- **What's New**: 본 논문은 산업용 데이터 스트림의 동적 품질 차원 문제를 해결하기 위해 "적응형 데이터 품질 점수 운영 프레임워크(Adaptive Data Quality Scoring Operations Framework)"를 제안합니다. 이 프레임워크는 데이터 품질의 변화를 적극적으로 모니터링하고 이에 적응하는 동적 변화 감지 메커니즘을 통합하여 품질 점수의 적합성을 보장합니다.



### Computation-friendly Graph Neural Network Design by Accumulating Knowledge on Large Language Models (https://arxiv.org/abs/2408.06717)
- **What's New**: 본 논문은 Large Language Model (LLM)을 이용하여 그래프 신경망 (GNN) 아키텍처 설계를 자동화하는 DesiGNN 프레임워크를 제안합니다. 기존 자동화 알고리즘의 단점인 계산 비용 및 전문성 부족 문제를 해결하기 위해 LLM에 특화된 지식을 제공하는 방식을 제시합니다. DesiGNN은 데이터 특성, 모델 아키텍처, 성능 간의 연관성을 파악하여 LLM이 초기 모델 제안을 빠르게 생성하고,  세밀한 아키텍처 조정을 통해 최적화된 GNN을 찾도록 돕습니다. 이를 통해 GNN 설계 과정을 혁신적으로 단축하고 효율성을 높입니다.



### Variational Learning of Gaussian Process Latent Variable Models through Stochastic Gradient Annealed Importance Sampling (https://arxiv.org/abs/2408.06710)
- **What's New**: 본 논문에서는 **Annealed Importance Sampling (AIS)**를 활용하여 **Gaussian Process Latent Variable Model (GPLVM)**의 변분 추론(Variational Inference) 성능을 향상시키는 새로운 방법을 제안합니다. 이 방법은 기존의 중요도 샘플링 방식의 한계를 극복하여 고차원 데이터 및 복잡한 데이터셋에 대한 효과적인 추론을 가능하게 합니다. 특히, **AIS**를 통해 **GPLVM**의 후방 분포를 연속적인 중간 분포로 변환하여 **Sequential Monte Carlo (SMC)** 샘플러와 변분 추론의 장점을 결합했습니다.



### Information Geometry and Beta Link for Optimizing Sparse Variational Student-t Processes (https://arxiv.org/abs/2408.06699)
- **What's New**: This paper proposes the use of **natural gradients** (Amari 1998) in **Sparse Variational Student-t Processes (SVTPs)** (Xu and Zeng 2024) for more efficient and robust optimization. It leverages the geometry of the parameter space defined by the variational distribution. This approach utilizes the **Fisher information matrix** which is connected to the **Beta function** (Abramowitz and Stegun 1948), allowing for efficient computation of natural gradients. Additionally, a **mini-batch algorithm** is introduced for efficient computation of natural gradients. The paper presents a robust method for computing the Fisher information matrix of multivariate Student-t distributions with diagonal covariance matrices, establishing the ‘Beta Link’ between the Fisher information matrix and the Beta function.



### SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields (https://arxiv.org/abs/2408.06697)
Comments:
          Accepted by ECCV 2024. Project website: this https URL

- **What's New**: SlotLifter, a novel object-centric radiance model for 3D scene reconstruction and decomposition, leverages slot-guided feature lifting to unite object-centric learning representations with image-based rendering methods. It achieves superior performance in both scene decomposition and novel-view synthesis compared to existing 3D object-centric models, especially on challenging real-world datasets like ScanNet and DTU.

- **Technical Details**: SlotLifter utilizes a slot-guided feature lifting mechanism that initializes 3D point features based on lifted 2D input-view features. This design improves detail granularity for novel-view synthesis and provides explicit guidance for slot learning.  The model employs a cross-attention-based transformer to predict volume rendering parameters by interacting lifted features with slot representations. It only relies on the reconstruction loss and requires fewer sampling overheads during training, leading to greater efficiency compared to existing models.

- **Performance Highlights**: SlotLifter significantly outperforms existing 3D object-centric models, achieving state-of-the-art performance in both scene decomposition (∼10+ ARI) and novel-view synthesis (∼2+ PSNR) on synthetic and real-world datasets. It demonstrates superior performance against image-based rendering methods on complex real-world datasets such as ScanNet and DTU. Ablative studies reveal the effectiveness of each module, highlighting SlotLifter's potential for object-centric learning and image-based rendering.



### DC3DO: Diffusion Classifier for 3D Objects (https://arxiv.org/abs/2408.06693)
- **What's New**: This paper presents **DC3DO**, a novel method for 3D object classification using **diffusion models** that achieves zero-shot classification without additional training. This approach leverages the density estimates from 3D diffusion models, demonstrating superior multimodal reasoning compared to discriminative methods.



### Masked Image Modeling: A Survey (https://arxiv.org/abs/2408.06687)
- **What's New**: 본 논문은 **마스크 이미지 모델링(Masked Image Modeling, MIM)**의 최신 연구들을 조사하여, 컴퓨터 비전 분야에서 강력한 **자기 지도 학습(self-supervised learning)** 기법으로 떠오르고 있는 MIM의 다양한 측면을 분석합니다. MIM은 이미지의 일부 정보 (픽셀, 패치, 또는 잠재 표현)를 가리고, 모델 (주로 오토인코더)이 보이는 부분의 컨텍스트를 사용하여 누락된 정보를 예측하도록 학습하는 기법입니다. 본 논문은 **재구성(Reconstruction)**과 **대조 학습(Contrastive Learning)** 두 가지 MIM 구현 방식을 공식화하고, 최근 주목할 만한 논문들을 분류하여 분석합니다. 또한, 계층적 클러스터링 알고리즘을 적용하여 수동 분류를 보완하고, 덴드로그램을 분석하여 관련 클러스터를 식별합니다. MIM 연구에 널리 사용되는 데이터셋을 살펴보고, 다양한 MIM 방법들의 성능 결과를 비교 분석하여, MIM의 연구 동향과 미래 방향을 제시합니다.



### Leveraging Priors via Diffusion Bridge for Time Series Generation (https://arxiv.org/abs/2408.06672)
- **What's New**: This paper introduces TimeBridge, a novel framework for time series generation using diffusion models. TimeBridge leverages diffusion bridges to learn the transport between chosen prior and data distributions, enabling flexible synthesis for both unconditional and conditional time series generation.

- **Technical Details**: The key innovation lies in utilizing diverse prior distributions for time series synthesis. TimeBridge incorporates data- and time-dependent priors for unconditional generation and data-scale preserving priors for conditional generation. It utilizes diffusion bridges to learn the optimal transport between these priors and the data distribution, enabling more flexible and controlled time series generation.

- **Performance Highlights**: TimeBridge achieves state-of-the-art performance in both unconditional and conditional time series generation tasks. It demonstrates superior capabilities in generating diverse and high-quality time series data, particularly when compared to traditional diffusion models that rely on standard Gaussian priors.



### RW-NSGCN: A Robust Approach to Structural Attacks via Negative Sampling (https://arxiv.org/abs/2408.06665)
- **What's New**: 본 논문은 그래프 구조 네트워크에서 발생하는 토폴로지 변화 및 가중치 불안정성 문제를 해결하기 위해 랜덤 워크 기반 음성 샘플링 그래프 합성곱 네트워크(RW-NSGCN)라는 새로운 방법을 제안합니다. RW-NSGCN은 랜덤 워크 알고리즘을 기반으로 음성 샘플링 메커니즘을 도입하여 비 인접 노드 정보를 활용하고 네트워크 안정성을 높입니다. 특히, RW-NSGCN은 랜덤 워크 with restart (RWR) 및 PageRank (PGR) 알고리즘을 통합하여 전역 및 지역 정보를 활용하고, 노드 중요도를 재평가하여 토폴로지 안정성을 확보합니다. 또한, RW-NSGCN은 Determinantal Point Process (DPP) 기반 GCN을 사용하여 음성 샘플의 다양성을 보장하고 강력한 노드 임베딩을 생성합니다. 



### Amuro & Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of Large Language Models (https://arxiv.org/abs/2408.06663)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 사전 훈련(Pre-training)과 미세 조정(Fine-tuning)의 상호 작용을 탐구하여 새로운 통찰력을 제공합니다. 기존의 LLM 훈련 패러다임에서 사전 훈련은 대량의 텍스트 데이터에서 수행되고 미세 조정은 후처리 단계로 여겨졌습니다. 이 논문에서는 사전 훈련의 중간 단계에서 여러 체크포인트를 미세 조정함으로써 이러한 두 단계 간의 관계를 심층적으로 분석합니다.

- **Technical Details**: 본 연구는 대규모 언어 모델(LLM)의 사전 훈련(Pre-training)과 미세 조정(Fine-tuning)의 상호 작용을 탐구합니다. 18개의 데이터셋에 대한 실험을 통해 다음과 같은 주요 결과를 얻습니다.
1. 지속적인 사전 훈련(Continual pre-training)은 미세 조정 후에만 드러나는 잠재적인 방식으로 모델을 개선합니다.
2. 추가 미세 조정을 통해 모델이 사전 훈련 단계에서 능력을 보이지 않던 데이터셋의 성능이 크게 향상됩니다.
3. 지도 학습(Supervised fine-tuning)을 통한 모델 성능 향상에도 불구하고 이전에 알고 있던 도메인 지식이나 미세 조정 중에 보지 못한 작업을 잊어버릴 수 있습니다.
4. 지도 학습으로 미세 조정된 모델은 평가 프롬프트(Evaluation prompt)에 대한 높은 민감성을 보이지만 더 많은 사전 훈련을 통해 이러한 민감성을 완화할 수 있습니다.

- **Performance Highlights**: 본 논문은 사전 훈련과 미세 조정의 상호 작용을 통해 모델의 능력을 탐구합니다. 특히, 지속적인 사전 훈련은 미세 조정 후에만 모델의 잠재적인 개선을 드러낸다는 점을 발견했습니다. 또한 모델은 미세 조정을 통해 새로운 능력을 얻지만, 이전에 학습한 도메인 지식이나 작업을 잊어버릴 수도 있습니다. 이러한 결과는 LLM 훈련에 대한 새로운 통찰력을 제공하며, 사전 훈련 및 미세 조정 방법을 개선하는 데 기여할 수 있습니다. 더욱이 본 연구는 최종 모델뿐만 아니라 훈련 동역학(Training dynamics)을 분석하는 것이 해석 가능성(Interpretability)을 높이는 중요한 측면임을 보여줍니다.



### Hierarchical Structured Neural Network for Retrieva (https://arxiv.org/abs/2408.06653)
Comments:
          9 pages

- **What's New**: HSNN (Hierarchical Structured Neural Network)를 소개합니다. 이 모델은 광고 추천 시스템의 검색 단계를 개선하기 위해,  hierarchical clustering과 neural network를 함께 학습하는 새로운 방법을 제시합니다.  기존의 Two Tower 아키텍처의 한계를 극복하고,  sophisticated interaction 및  ranking 시스템에서 일반적으로 사용되는 모델 아키텍처를 활용합니다.  HSNN은 sub-linear inference cost를 유지하면서  offline 평가에서 6.5% 개선 및 A/B 실험을 통해  1.22% online 개선을 달성했습니다.  또한,  HSNN은 광고 추천 시스템에 배포되어  현재 많은 트래픽을 처리하고 있습니다. 



### EditScribe: Non-Visual Image Editing with Natural Language Verification Loops (https://arxiv.org/abs/2408.06632)
Comments:
          ASSETS 2024

- **What's New**: EditScribe, a prototype system that utilizes large multimodal models (LMMs) to make object-level image editing actions non-visually accessible to blind and low-vision (BLV) individuals, is introduced. This system addresses the accessibility challenges in image editing for BLV users by enabling them to comprehend image content, specify edit actions through natural language prompts, and receive verification feedback in natural language to ensure accurate edits.

- **Technical Details**: EditScribe leverages natural language verification loops. It provides four types of verification feedback: Summary of Visual Changes, AI Judgement, updated General and Object Descriptions, and follow-up questions for clarification. The system is tested with five object-level edit actions, including blurring, removing, color changing, brightness adjusting, and adding text to an object in an image. This object-level focus allows for precise image detail manipulation crucial for tasks commonly desired by BLV individuals like privacy protection and image enhancement.

- **Performance Highlights**: A study with 10 BLV participants demonstrated the effectiveness of EditScribe. The participants were able to perform most editing tasks using the system and adopted different prompting strategies, including detailed, succinct, and varied tones to facilitate the system's understanding. They also showed preferences for different types of verification feedback depending on the context. Overall, participants expressed confidence in the edited images produced by EditScribe and were willing to publish them based on context, but preferred a second check using sighted assistance or other AI services. The findings highlight the potential of natural language verification loops for non-visual content creation accessibility.



### WorldScribe: Towards Context-Aware Live Visual Descriptions (https://arxiv.org/abs/2408.06627)
Comments:
          UIST 2024

- **What's New**: WorldScribe는 사용자의 컨텍스트에 맞게 적응 가능한 실시간 시각적 설명을 생성하는 새로운 시스템입니다. 이 시스템은 사용자의 의도에 따라 설명을 우선순위를 정하고, 시각적 컨텍스트와 사운드 컨텍스트에 따라 설명 방식을 조정합니다.



### Generalized knowledge-enhanced framework for biomedical entity and relation extraction (https://arxiv.org/abs/2408.06618)
- **What's New**: 이 연구는 바이오메디컬 엔티티 및 관계 추출을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 외부 지식을 활용하여 작업에 독립적이고 재사용 가능한 바이오메디컬 엔티티 및 관계 추출을 위한 배경 지식 그래프를 구축합니다. 이 모델은 인간이 도메인 특정 주제를 학습하는 방식에서 영감을 받았습니다. 특히 인간은 먼저 기초 지식을 구축하기 위해 필드에 대한 가장 기본적이고 일반적인 지식을 습득한 다음, 이를 기반으로 다양한 전문 분야 주제로 확장합니다. 이 프레임워크는 이러한 일반적인 지식 공유 메커니즘을 사용하여 다양한 도메인 특정 바이오메디컬 텍스트에 효과적으로 전이 학습이 가능한 일반적인 신경망 지식 그래프를 구축합니다.



### Super-intelligence or Superstition? Exploring Psychological Factors Underlying Unwarranted Belief in AI Predictions (https://arxiv.org/abs/2408.06602)
- **What's New**: 이 연구는 개인 행동에 대한 AI 예측에 대한 믿음에 영향을 미치는 심리적 요인을 조사하여 점성술과 성격 기반 예측에 대한 믿음과 비교했습니다. 238명의 참가자를 대상으로 한 실험을 통해 인지 스타일, 초자연적 믿음, AI 태도, 성격 특성 및 기타 요인이 다양한 출처의 예측의 인식된 유효성, 신뢰성, 유용성 및 개인화에 어떻게 영향을 미치는지 조사했습니다.

- **Technical Details**: 실험은 238명의 참가자를 대상으로 진행되었으며, AI, 점성술 및 성격 심리학의 세 가지 출처에서 가상의 예측을 제시했습니다. 참가자들은 인식된 유효성, 신뢰성, 유용성 및 개인화를 평가했습니다. 예측을 위해 참가자들은 점성술 및 성격에 대한 설문지를 작성하고 tương tác을 분석하고 미래 투자 행동을 예측하는 것처럼 보이는 시뮬레이션 투자 게임에 참여했습니다. 참가자들은 긍정적인 (N = 119) 또는 부정적인 (N = 119) 예측 그룹에 무작위로 할당되었으며, 미래 투자 행동과 결과에 대한 예측을 각각 받았습니다.

- **Performance Highlights**: 결과는 AI 예측에 대한 믿음이 점성술과 성격 심리학에 근거한 예측에 대한 믿음과 양의 상관관계가 있음을 보여줍니다. 특히 초자연적 믿음과 긍정적인 AI 태도는 AI 예측의 인식된 유효성, 신뢰성, 유용성 및 개인화를 상당히 증가시켰습니다. 양심성은 모든 출처의 예측에 대한 믿음과 음의 상관관계가 있었고, 예측 주제에 대한 관심은 모든 예측에 대한 신뢰성을 높였습니다. 놀랍게도 인지 스타일은 예측에 대한 믿음에 유의미한 영향을 미치지 않았습니다. 이러한 결과는 AI에서 '합리적인 미신' 현상을 강조하는데, 이는 믿음이 비판적 평가보다는 정신적 휴리스틱(heuristics)과 직관에 의해 더 많이 추진된다는 것을 의미합니다. 이 연구는 AI 시스템과 신뢰와 회의주의를 촉진하는 의사 소통 전략을 설계하기 위한 시사점을 논의합니다. 이 연구는 인간-AI 상호 작용의 심리에 대한 이해에 기여하고 AI 시스템의 설계 및 배포에 대한 통찰력을 제공합니다.



### A Perspective on Large Language Models, Intelligent Machines, and Knowledge Acquisition (https://arxiv.org/abs/2408.06598)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 한계를 강조하며, LLM이 인간 지능과 비교하여 추상적 개념을 이해하고 추론하는 능력에 있어서 심각한 차이가 있다고 주장합니다. 특히, GPT-4를 이용한 과학, 수학, 상식 추론 질문에 대한 분석을 통해, GPT-4가 인간의 추론을 모방할 수 있지만 실제 이해는 부족하다는 점을 보여줍니다. LLM이 인간 지식 습득과 교육에 미치는 영향을 논의합니다.



### Social Debiasing for Fair Multi-modal LLMs (https://arxiv.org/abs/2408.06569)
- **What's New**: 본 논문은 멀티모달 대규모 언어 모델(MLLM)의 사회적 편향 문제를 해결하기 위해 두 가지 새로운 접근 방식을 제시합니다. 첫째, 다양한 사회적 개념을 포함하는 역설적 데이터 세트(Counterfactual dataset with Multiple Social Concepts, CMSC)를 소개합니다. CMSC는 기존 데이터 세트에 비해 더 다양하고 광범위한 훈련 세트를 제공합니다. 둘째, 반-고정관념 편향 제거 전략(Anti-Stereotype Debiasing, ASD)을 제안합니다. ASD는 MLLM 훈련 과정을 재검토하고, 자동 회귀 손실 함수를 조정하고, 데이터 샘플링 방법을 개선하여 편향을 제거합니다. 



### AquilaMoE: Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies (https://arxiv.org/abs/2408.06567)
- **What's New**: This paper introduces AquilaMoE, a bilingual 8*16B Mixture of Experts (MoE) language model trained using a novel method called EfficientScale, which significantly improves training efficiency and reduces data requirements compared to traditional methods.

- **Technical Details**: EfficientScale consists of three phases: Preparation, Scale-Up, and Scale-Out. In the Scale-Up phase, a smaller pre-trained dense model's weights are used to initialize a larger dense model, enabling knowledge transfer and continued training with less data. The Scale-Out phase further enhances performance by using a pre-trained dense model to initialize MoE experts. The paper proposes AKI-Pro, an improved version of Advanced Knowledge Initialization (AKI), for initializing the larger model, addressing limitations of the original AKI in expanding the depth and adapting to Group Query Attention (GQA).

- **Performance Highlights**: The authors successfully trained a 16B model and then the 8*16B AquilaMoE model using the optimal initialization scheme. Extensive validation experiments on smaller models (1.8B and 7B) demonstrate that these models maintain and further reduce loss during continuous pretraining, highlighting the effectiveness of EfficientScale.



### HDRGS: High Dynamic Range Gaussian Splatting (https://arxiv.org/abs/2408.06543)
- **What's New**: This paper proposes a novel method called **High Dynamic Range Gaussian Splatting (HDR-GS)** for reconstructing 3D HDR radiance fields from multi-exposure LDR images. HDR-GS leverages the recent real-time 3D reconstruction technique called **Gaussian Splatting** and incorporates a differentiable, **asymmetric grid** for tone mapping, enabling efficient and accurate HDR scene recovery. Additionally, it introduces a **coarse-to-fine strategy** that accelerates model convergence and enhances robustness against sparse viewpoints and extreme exposure conditions.



### Dynamic Exclusion of Low-Fidelity Data in Bayesian Optimization for Autonomous Beamline Alignmen (https://arxiv.org/abs/2408.06540)
Comments:
          12 pages, 6 figure sets

- **What's New**: 본 연구는 싱크로트론 광원에서 빔라인을 정렬하는 문제를 효율적으로 해결하기 위한 새로운 방법을 제시합니다. 이 방법은 Bayesian Optimization (베이지안 최적화) 기법을 사용하여 빔 품질을 최적화하지만, 센서 오류나 배경 노이즈로 인해 신뢰할 수 없는 데이터를 제거하는 기능을 추가했습니다.



### Chain-of-Strategy Planning with LLMs: Aligning the Generation of Psychotherapy Dialogue with Strategy in Motivational Interviewing (https://arxiv.org/abs/2408.06527)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)을 사용하여 동기적 면접(MI)에 기반한 심리치료 대화를 생성하는 새로운 접근 방식을 제시합니다. 특히, Chain-of-Strategy(CoS) 계획을 사용하여 전략 인식 대화 생성을 제안하며, 이는 먼저 MI 전략을 추론으로 예측하고 이러한 전략을 사용하여 후속 대화 생성을 안내합니다. 이는 생성된 MI 대화를 치료 전략에 맞춤으로써 심리치료에서 제어 가능하고 설명 가능한 생성 가능성을 제공합니다.

- **Technical Details**: 본 연구에서는 Chain-of-Strategy (CoS) 계획을 기반으로 전략 인식 대화 생성을 제안합니다. CoS 계획은 MI 전략을 추론으로 예측하고 이러한 전략을 사용하여 후속 대화 생성을 안내하는 방식입니다. 이러한 방식은 생성된 MI 대화를 치료 전략에 맞춤으로써 심리치료에서 제어 가능하고 설명 가능한 생성 가능성을 제공합니다. 본 연구에서는 AnnoMI와 BiMISC의 두 가지 MI 데이터 세트를 사용하여 이러한 접근 방식을 평가합니다. 이러한 데이터 세트는 MI 대화와 MISC 스킴에 의해 주석 처리된 MI 전략을 포함하고 있습니다.

- **Performance Highlights**: 자동 및 인간 평가를 포함한 광범위한 실험을 통해 MI 전략의 효과를 검증했습니다. 연구 결과는 LLM이 전략적으로 정렬된 대화를 생성할 가능성을 보여주었으며 심리치료 환경에서 실용적인 응용을 위한 방향을 제시합니다.



### Learned Ranking Function: From Short-term Behavior Predictions to Long-term User Satisfaction (https://arxiv.org/abs/2408.06512)
Comments:
          RecSys 24

- **What's New**: 이 논문은 YouTube에서 사용자 만족도를 극대화하기 위해 새로운 학습 기반 순위 함수 (Learned Ranking Function, LRF) 시스템을 제안합니다. 기존의 솔루션들이 휴리스틱 함수의 하이퍼파라미터를 최적화하는 데 초점을 맞춘 반면, LRF는 슬레이트 최적화 문제를 직접 모델링하여 장기적인 사용자 만족도를 극대화하는 데 목표를 두고 있습니다. LRF는 다중 목표 최적화의 안정성을 보장하는 새로운 제약 최적화 알고리즘도 포함합니다. 

- **Technical Details**: LRF는 사용자와 슬레이트(추천 목록)의 상호 작용을 캐스케이드 클릭 모델(cascade click model)로 모델링합니다. 이 모델은 사용자가 슬레이트를 순차적으로 검토하며, 특정 비디오를 클릭하거나, 슬레이트를 포기하거나, 또는 슬레이트의 끝까지 도달할 수 있다는 것을 가정합니다. LRF는 이러한 사용자 행동을 고려하여 슬레이트의 장기적인 가치를 최적화하는 알고리즘을 사용합니다. 또한, LRF는 다중 목표 최적화의 안정성을 보장하기 위해 동적 선형 스칼라화 (dynamic linear scalarization) 기반의 새로운 제약 최적화 알고리즘을 제안합니다. 

- **Performance Highlights**: LRF 시스템은 YouTube에 배포되어 실험 결과를 통해 효과가 입증되었습니다. LRF는 기존의 휴리스틱 기반 시스템보다 장기적인 사용자 만족도를 개선하는 것으로 나타났습니다. 또한, LRF는 다양한 목표 간의 균형을 유지하며 안정적인 성능을 보여주었습니다. 



### Fooling SHAP with Output Shuffling Attacks (https://arxiv.org/abs/2408.06509)
- **What's New**: This paper proposes a novel type of adversarial attack on Explainable AI (XAI) methods, specifically targeting Shapley value-based explanations. These attacks, called "shuffling attacks," are data-agnostic, meaning they don't require access to the underlying data distribution, making them more practical in real-world scenarios.

- **Technical Details**: The shuffling attacks modify the model output (e.g., scores or rankings) without altering the model's internal structure or training data. The attacks exploit the way Shapley values are calculated by changing the order of input features, effectively masking the influence of "protected features" on the model's output. The paper proves that Shapley values cannot detect these attacks, but estimation algorithms like linear SHAP and SHAP can detect them with varying degrees of effectiveness.

- **Performance Highlights**: The paper demonstrates the effectiveness of the shuffling attacks by comparing the performance of linear SHAP and SHAP using real-world datasets. The attacks effectively fooled the XAI methods, making it difficult to identify unfairness in the model's output.

- **Limitations**: The paper focuses on Shapley value-based explanations, leaving room for investigation on other XAI methods.

- **Keywords**: Explainable AI (XAI), Shapley values, Adversarial attacks, Data-agnostic attacks, Shuffling attacks, Model fairness, Protected features, Linear SHAP, SHAP



### Benchmarking tree species classification from proximally-sensed laser scanning data: introducing the FOR-species20K datas (https://arxiv.org/abs/2408.06507)
- **What's New**: FOR-species20K, 레이저 스캐닝 데이터를 사용한 나무 종 분류를 위한 딥 러닝 모델 개발 및 벤치마킹을 위한 핵심 자원으로 20,000개 이상의 나무 포인트 클라우드(point cloud)를 포함하고 있습니다. 이 데이터 세트는 다양한 유럽 숲에서 획득한 지상 (TLS), 모바일 (MLS) 및 드론 레이저 스캐닝 (ULS)을 사용하여 33개 종의 나무 포인트 클라우드를 포함합니다. (TLS: Terrestrial Laser Scanning, MLS: Mobile Laser Scanning, ULS: Unmanned Laser Scanning)



### Decentralized Cooperation in Heterogeneous Multi-Agent Reinforcement Learning via Graph Neural Network-Based Intrinsic Motivation (https://arxiv.org/abs/2408.06503)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문은 분산 환경에서 부분 관찰 가능성(partial observability)과 희소 보상(sparse reward) 문제를 가진 이종 에이전트(heterogeneous agent) 협업을 위한 새로운 알고리즘인 CoHet을 제안합니다. CoHet은 그래프 신경망(GNN) 기반의 고유한 내재적 보상 메커니즘을 사용하여 이종 에이전트 정책을 학습합니다. CoHet은 기존의 방법과 달리 에이전트의 지역 이웃 정보만을 사용하여 내재적 보상을 계산하며, 에이전트 이종성에 대한 사전 지식이 필요하지 않습니다.



### Cross-Lingual Conversational Speech Summarization with Large Language Models (https://arxiv.org/abs/2408.06484)
- **What's New**: This paper introduces a novel dataset for cross-lingual conversational speech summarization, addressing the lack of resources in this area. It utilizes the Fisher and Callhome Spanish-English Speech Translation corpus, generating summaries from the English translations using GPT-4. This allows for the evaluation of summarization models in the presence of ASR and MT errors.

- **Technical Details**: The authors create a cascade-based system using Whisper-large-v3 for ASR and NLLB 1.3 Billion parameter dense model for machine translation. They experiment with various LLMs for summarization, including GPT-4 and Mistral-7B. They employ LoRA fine-tuning to adapt the models for the task. The dataset is constructed by splitting conversations into chunks and generating four reference summaries per chunk using GPT-4.

- **Performance Highlights**: The Mistral-7B model, adapted for this task, outperforms other off-the-shelf LLMs and achieves performance comparable to GPT-4. The results highlight the importance of fine-tuning for task adaptation and the potential of smaller, quantized models in this domain.



### Multi-View Neural Differential Equations for Continuous-Time Stream Data in Long-Term Traffic Forecasting (https://arxiv.org/abs/2408.06445)
- **What's New**: 이 논문에서는 장기 교통 흐름 예측을 위한 새로운 딥 러닝 프레임워크인 멀티-뷰 뉴럴 디퍼렌셜 방정식(MNDE)을 제안합니다. MNDE는 NDE(Neural Differential Equations)를 기반으로 하여 지연된 전파, 동적 공간 의존성 및 국지적 트렌드 변화를 포함한 복잡한 교통 패턴을 포착합니다. 이 모델은 다중 NDE 모듈을 통해 연속 시간 내의 동적 교통 패턴을 파악하고, 지연된 전파와 동적 공간 의존성을 별도로 학습합니다. 또한, 그래디언트 자기 주의 메커니즘을 사용하여 국지적 트렌드 변화를 모델링합니다.

- **Technical Details**: MNDE는 다중 NDE 모듈을 사용하여 3가지 핵심적인 교통 흐름 특징을 포착합니다. 첫째, 지연된 전파를 나타내기 위해, 현재 상태와 지연된 상태를 별도의 NDE 모듈로 학습합니다. 둘째, 동적 공간 의존성을 모델링하기 위해, 시간 경과에 따른 위치 간 상관 관계를 계산하여 NDE 입력으로 사용합니다. 셋째, 국지적 트렌드 변화를 파악하기 위해, 그래디언트 자기 주의 메커니즘을 사용하여 연속 시간 내의 교통 흐름 그래디언트에 대한 자기 주의를 수행합니다.

- **Performance Highlights**: 실제 교통 데이터셋에 대한 광범위한 실험 결과, MNDE가 기존 방법보다 장기 교통 예측에서 우수한 성능을 보였으며, 특히 노이즈가 많은 입력 데이터에서 뛰어난 예측 정확도를 달성했습니다. MNDE는 장기 교통 예측에서 지연된 전파, 동적 공간 의존성 및 국지적 트렌드 변화를 효과적으로 고려함으로써 뛰어난 성능을 보여줍니다.



### Evaluating Language Models on Entity Disambiguation in Tables (https://arxiv.org/abs/2408.06423)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)을 기반으로 한 **일반적인 테이블 이해 및 조작(GTUM)** 접근 방식이 **Semantic Table Interpretation(STI)** 작업, 특히 **Cell-Entity Annotation(CEA)** 작업에서 기존의 규칙 기반 접근 방식과 비교하여 어떻게 성능을 발휘하는지 분석합니다.



### PhaGO: Protein function annotation for bacteriophages by integrating the genomic contex (https://arxiv.org/abs/2408.06402)
Comments:
          17 pages,6 figures

- **What's New**: PhaGO, a novel phage protein function annotation tool, leverages the modular genomic structure of phage genomes to improve accuracy in predicting protein functions, especially for proteins with uncommon functions and those lacking homology search results.



### Distributed Stackelberg Strategies in State-based Potential Games for Autonomous Decentralized Learning Manufacturing Systems (https://arxiv.org/abs/2408.06397)
Comments:
          This pre-print was submitted to IEEE Transactions on Systems, Man, and Cybernetics: Systems on July 31, 2024

- **What's New**: This paper introduces **Distributed Stackelberg Strategies in State-Based Potential Games (DS2-SbPG)**, a new game structure for autonomously optimizing decentralized manufacturing systems with multiple objectives. It combines **potential games** and **Stackelberg games** to improve cooperative trade-off capabilities and handle multi-objective optimization challenges in a distributed manner.



### Design Proteins Using Large Language Models: Enhancements and Comparative Analyses (https://arxiv.org/abs/2408.06396)
Comments:
          This paper has been accepted for presentation at Language and Molecules ACL 2024

- **What's New**: 본 논문은 Mistral-7B1, Llama-2-7B2, Llama-3-8B3, gemma-7B4와 같은 사전 훈련된 LLM(Large Language Model)을 사용하여 고품질 단백질 서열을 생성하는 방법을 제시합니다. 기존의 단백질 생성 모델과 달리, 이 연구는 42,000개의 인간 단백질 서열을 사용하여 훈련합니다. 이러한 LLM들은 생물학적으로 타당한 단백질 구조를 생성하도록 재훈련되었으며, ProGen, ProtGPT2, ProLLaMA와 같은 기존의 단백질 생성 모델과 유사한 성능을 보여줍니다. 이 연구에서는 pLDDT, RMSD, TM-score, REU 등의 지표를 사용하여 모델 성능을 평가하고, 훈련된 모델들을 공개하여 연구 및 협업을 장려합니다.



### Autoregressive Enzyme Function Prediction with Multi-scale Multi-modality Fusion (https://arxiv.org/abs/2408.06391)
- **What's New**: MAPred는 단백질의 EC 번호 (Enzyme Commission number)를 자동 회귀적으로 예측하도록 설계된 새로운 다중 모달 및 다중 스케일 모델입니다. 기존의 딥 러닝 방법들은 주로 서열 데이터 또는 구조 데이터에만 의존하여 EC 번호를 전체적으로 예측하는 경향이 있었고, EC 번호의 고유 계층적 구조를 무시했습니다. MAPred는 이러한 제한 사항을 해결하기 위해 단백질의 1차 아미노산 서열과 3차원 토큰 (3D tokens)을 모두 통합하고, 포괄적인 단백질 특성과 필수적인 지역 기능 부위 (local functional sites)를 포착하기 위해 이중 경로 접근 방식 (dual-pathway approach)을 사용합니다. 또한, MAPred는 EC 분류의 계층적 구성을 활용하여 EC 번호의 자릿수를 순차적으로 예측하는 자동 회귀 예측 네트워크를 사용합니다. New-392, Price, New-815를 포함한 벤치마크 데이터 세트 (benchmark datasets)에 대한 평가는 MAPred가 기존 모델보다 뛰어난 성능을 보여 주었으며, 생물정보학에서 단백질 기능 예측의 신뢰성과 세분성을 크게 향상시켰음을 나타냅니다.

- **Technical Details**: MAPred는 다음과 같은 핵심 기술을 사용합니다:

* **다중 모달 및 다중 스케일:** MAPred는 단백질의 1차 아미노산 서열과 3차원 토큰 (3D tokens)을 모두 통합하여 단백질의 다양한 특성을 포착합니다.
* **이중 경로 접근 방식:** MAPred는 단백질의 전반적인 특징을 파악하는 전역 특징 추출 경로 (global feature extraction pathway)와 기능적 부위 (functional sites)를 포착하는 지역 특징 추출 경로 (local feature extraction pathway)를 사용합니다.
* **자동 회귀 예측 네트워크:** MAPred는 EC 번호의 자릿수를 순차적으로 예측하는 자동 회귀 예측 네트워크를 사용하여 EC 번호의 계층적 구조를 활용합니다.

- **Performance Highlights**: MAPred는 New-392, Price, New-815를 포함한 벤치마크 데이터 세트에서 기존 모델보다 더 나은 성능을 보여 주었습니다. 이는 MAPred가 단백질 기능 예측의 신뢰성과 세분성을 향상시켰음을 의미합니다.



### ViC: Virtual Compiler Is All You Need For Assembly Code Search (https://arxiv.org/abs/2408.06385)
- **What's New**: 본 논문에서는 역 엔지니어링을 위한 어셈블리 코드 검색을 개선하기 위해 가상 컴파일러(Virtual Compiler, ViC)라는 새로운 접근 방식을 제시합니다. ViC는 대규모 언어 모델(LLM)을 사용하여 다양한 프로그래밍 언어의 소스 코드를 어셈블리 코드로 컴파일하는 기능을 제공합니다. 이를 통해 기존 컴파일러의 복잡성을 우회하고 다양한 프로그래밍 언어의 어셈블리 코드 데이터셋을 효과적으로 구축할 수 있습니다.



### Assessment of Cell Nuclei AI Foundation Models in Kidney Pathology (https://arxiv.org/abs/2408.06381)
- **What's New**: 이 연구는 세 가지 최첨단 (SOTA) 세포 핵 기반 모델(Cellpose, StarDist, CellViT)의 성능을 대규모로 평가하여 신장 병리학에서의 세포 핵 분할 성능을 측정했습니다. 2,542개의 신장 전체 슬라이드 이미지(WSI)로 구성된 다양한 평가 데이터 세트를 만들었습니다. 이는 인간과 설치류 출처의 다양한 조직 유형, 크기 및 염색 방법을 포함합니다. 이 연구는 신장 병리학에 특화된 기반 모델의 필요성을 강조하며 기반 모델 간의 일치 분석을 수행하여 합의된 실패 사례를 파악했습니다.



### Masked Graph Autoencoders with Contrastive Augmentation for Spatially Resolved Transcriptomics Data (https://arxiv.org/abs/2408.06377)
- **What's New**: 이 논문은 공간 해상도 전사체학 (SRT) 데이터 분석을 위해 새로운 방법인 대조적으로 강화된 마스크 그래프 오토인코더 (STMGAC)를 제안합니다. STMGAC는 도메인 식별을 위한 저차원 잠재 표현을 학습하며, 자기 증류를 통해 잠재 공간에서 지속적인 표현 신호를 얻어 자기 감독 일치를 유도합니다. 동시에, 트리플렛 학습을 사용하여 양성 및 음성 앵커 쌍을 구성하여 차별적 능력을 향상시킵니다.



### An Adaptive CSI Feedback Model Based on BiLSTM for Massive MIMO-OFDM Systems (https://arxiv.org/abs/2408.06359)
Comments:
          13 pages, 14 figures, 3 tables

- **What's New**: This paper introduces a novel Adaptive Bidirectional Long Short-Term Memory Network (ABLNet) for Channel State Information (CSI) feedback in massive Multiple-Input Multiple-Output Orthogonal Frequency Division Multiplexing (MIMO-OFDM) systems. This network is designed to handle various input CSI lengths and adjust the number of feedback bits accordingly. It incorporates a Feedback Bit Control Unit (FBCU) to enable flexible feedback bit number control. Furthermore, a Bit Number Adjusting (BNA) algorithm is proposed to optimize feedback performance by adapting the number of feedback bits.  To address the model protection challenge between different manufacturers, a separate training approach is devised.

- **Technical Details**: ABLNet employs BiLSTM (Bidirectional Long Short-Term Memory) to process different input CSI lengths effectively. The FBCU module controls the output length of feedback bits, allowing for flexible feedback bit number adjustments. The BNA algorithm optimizes feedback performance by adjusting the number of bits. The separate training approach enables model protection by training the encoder and decoder separately.

- **Performance Highlights**: The proposed ABLNet with FBCU can accommodate various input CSI lengths and feedback bit numbers. The BNA algorithm stabilizes CSI feedback performance by dynamically adjusting the feedback bit number. The separate training approach preserves feedback performance while reducing the complexity of the feedback model.



### Algorithm Research of ELMo Word Embedding and Deep Learning Multimodal Transformer in Image Description (https://arxiv.org/abs/2408.06357)
- **What's New**: 본 논문에서는 기존의 임베딩 기반 제로 샘플 학습 방법론의 단점을 개선하여, 미지의 클래스를 벡터 공간에 포함시키는 새로운 방법을 제안합니다. 이는 카테고리 의미 유사성 측정을 활용하여, 현재 알려진 클래스와 동일한 의미를 가진 미지의 클래스를 벡터 공간에 통합하는 방식을 통해 가능합니다. 또한, 기존의 제로 샘플 학습 알고리즘들이 의료 이미지의 깊이 특징을 직접 입력으로 사용하는 반면, 본 연구에서는 ELMo-MCT를 활용하여 셀프 어텐션 메커니즘을 통해 원본 이미지와 관련된 다양한 시각적 특징을 추출하는 방식을 제안합니다. 이를 통해, 기존 방법보다 더욱 정확한 제로 샘플 학습 성능을 달성할 수 있습니다.



### Using Large Language Models to Compare Explainable Models for Smart Home Human Activity Recognition (https://arxiv.org/abs/2408.06352)
Comments:
          Accepted for publication at UbiComp / ISWC 2024's XAIforU workshop

- **What's New**: 이 논문에서는 비전문가 사용자에게 가장 적합한 설명 가능한 인공지능(XAI) 접근 방식을 식별하기 위해 대규모 언어 모델(LLM)을 사용하는 자동 평가 방법을 제안합니다. 

- **Technical Details**: 본 연구에서는 스마트 홈 환경에서 수행되는 활동을 인식하는데 사용되는 설명 가능한 활동 인식(XAR) 시스템을 평가하는 새로운 방법을 제시합니다. 특히 여러 XAR 모델의 출력을 분석하여 비전문가 사용자에게 가장 적합한 모델을 선택하는데 LLM을 활용합니다. 이를 위해 두 가지 LLM 프롬프팅 전략을 제안하며, 이를 통해 XAR 모델이 생성하는 자연어 설명의 품질을 평가합니다.

- **Performance Highlights**: 본 논문의 초기 결과는 LLM 평가가 사용자 설문 조사와 일치하는 것을 시사합니다. LLM은 XAR 모델의 설명 품질을 효과적으로 평가할 수 있는 잠재력을 보여주며, 비용 및 시간이 많이 드는 사용자 설문 조사를 대체할 수 있는 가능성을 제시합니다. 

- **Keywords**: eXplainable AI (XAI), Human Activity Recognition (HAR), Large Language Models (LLMs),  explainable activity recognition (XAR), smart home,  Activities of Daily Living (ADLs), cognitive decline



### Closing the Affective Loop via Experience-Driven Reinforcement Learning Designers (https://arxiv.org/abs/2408.06346)
Comments:
          9 pages, 4 figures, 1 table

- **What's New**: 이 논문은 레이싱 게임에서 사용자의 특정 감정 반응을 유발하는 트랙을 생성하기 위한 새로운 강화 학습(RL) 프레임워크인 경험 기반 RL(EDRL) 프레임워크를 제안합니다. EDRL은 설계자가 지정한 감정 패턴을 목표로 하며, 사용자의 감정적 반응을 정확하게 예측할 수 있는 트랙을 생성하는 데 성공합니다.

- **Technical Details**: EDRL 프레임워크는 각 생성된 트랙의 감정 패턴을 평가하는 보상 함수를 사용합니다. 이 보상 함수는 게임 상태 시뮬레이션을 통해 생성된 트랙을 평가하는 데 사용되는 K-Nearest Neighbors(KNN) 알고리즘을 기반으로 합니다. EDRL은 유전자 알고리즘을 사용하여 트랙을 생성하는 경험 기반 PCG 방법과 비교되어, EDRL이 감정 기반 게임 레벨 생성에 더 효율적이고 안정적이라는 결과를 보여줍니다.

- **Performance Highlights**: EDRL은 다양한 플레이어 유형과 목표 감정 패턴에 따라 게임 레벨을 생성할 수 있으며, 특정 플레이어 유형에 대해서는 특정 감정 패턴을 생성하는 데 더 어려움을 겪는 것으로 나타났습니다. 예를 들어, 낮은 감정 수준을 보이는 플레이어의 경우 최대 자극 트랙을 생성하는 것이 최소 자극 트랙을 생성하는 것보다 쉬운 반면, 높은 자극 값을 가진 플레이어의 경우 EDRL이 자극 변화를 최소화하도록 설계된 경우 감정적 만족을 제공하기 어려웠습니다.



### VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents (https://arxiv.org/abs/2408.06327)
- **What's New**: 본 논문에서는 **VisualAgentBench (VAB)**라는 새로운 벤치마크를 소개하며, 이는 **대규모 멀티모달 모델(LMM)**을 **시각적 기반 에이전트(Visual Foundation Agents)**로 훈련하고 평가하기 위한 포괄적인 벤치마크입니다. VAB는 **Embodied**, **Graphical User Interface**, **Visual Design**과 같은 다양한 시나리오에 걸쳐 LMM의 잠재력을 최대한 활용하고 복잡한 실제 환경에서 LMM의 이해 및 상호 작용 능력을 평가하기 위해 설계되었습니다.



### Can We Rely on LLM Agents to Draft Long-Horizon Plans? Let's Take TravelPlanner as an Examp (https://arxiv.org/abs/2408.06318)
Comments:
          13 pages, 2 figures, 4 tables

- **What's New**: 본 논문은 대규모 언어 모델(LLM) 기반 에이전트가 복잡한 실제 세계 계획 작업에서 어떻게 작동하는지, 왜 실패할 수 있는지, 그리고 어떻게 개선할 수 있는지에 대한 연구를 제시합니다.

- **Technical Details**: 본 연구는 실제적인 벤치마크인 TravelPlanner를 사용하여 LLM 에이전트의 성능을 평가합니다. TravelPlanner는 여러 제약 조건을 충족해야 하는 정확한 계획을 생성하는 작업을 수행해야 합니다. 이 논문에서는 LLM 에이전트의 성능을 향상시키기 위한 새로운 방법인 피드백 인식 미세 조정(FAFT)을 제안합니다. FAFT는 긍정적 및 부정적 피드백을 활용하여 기존의 지도 학습 미세 조정(SFT)보다 훨씬 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, LLM은 긴 컨텍스트 내에서 중요한 정보를 찾아내는 데 어려움을 겪는다는 사실이 밝혀졌습니다. 또한, LLM은 긴 계획을 분석하고 정확한 피드백을 제공하는 데 어려움을 겪고 있습니다. 반면에, FAFT는 긍정적 및 부정적 피드백을 사용하여 계획 작업의 성능을 크게 향상시켰습니다. FAFT는 LLM 기반 에이전트의 성능을 향상시키는 데 유망한 방법임을 보여줍니다.



### OWL2Vec4OA: Tailoring Knowledge Graph Embeddings for Ontology Alignmen (https://arxiv.org/abs/2408.06310)
Comments:
          Submitted to a conference

- **What's New**: OWL2Vec4OA is introduced, an extension of the OWL2Vec* ontology embedding system specifically designed for ontology alignment.  It incorporates confidence values from seed mappings into the random walk strategy, leading to embeddings more suitable for alignment tasks.

- **Technical Details**: OWL2Vec* projects an ontology into a graph and uses random walks to generate sequences of entities. OWL2Vec4OA adds confidence values from seed mappings to bias these random walks, giving preference to edges with higher confidence. Seed mappings are provided by ontology matching systems like LogMap and AML.

- **Performance Highlights**: Experiments show that OWL2Vec4OA's embeddings are more effective for ontology alignment than those generated by the original OWL2Vec*.  Its embeddings also produce promising results in the OAEI's Bio-ML track for entity ranking.



### The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery (https://arxiv.org/abs/2408.06292)
- **What's New**: 본 논문은 인공지능 과학자(The AI Scientist)라는 새로운 프레임워크를 소개하며, 최첨단 대규모 언어 모델(large language models)이 독립적으로 과학 연구를 수행하고 그 결과를 전달할 수 있도록 합니다. 인공지능 과학자는 새로운 연구 아이디어를 생성하고, 코드를 작성하고, 실험을 실행하고, 결과를 시각화하며, 완전한 과학 논문을 작성하여 결과를 설명하고, 마지막으로 평가를 위한 시뮬레이션 검토 프로세스를 수행합니다.  본질적으로 이 과정은 인간 과학계처럼 끝없이 아이디어를 반복적으로 개발할 수 있습니다. 이러한 새로운 연구 프레임워크는 과학 연구 과정에서 인공지능 에이전트의 변혁적인 이점을 제공하며, 엄청난 비용과 시간이 소요되는 과학 연구에 대한 새로운 가능성을 열어줍니다.



### Strategy Game-Playing with Size-Constrained State Abstraction (https://arxiv.org/abs/2408.06202)
Comments:
          8 pages, to be published in Proceedings of the Conference on Games 2024, codes are open-sourced at this https URL

- **What's New**: 이 논문은 전략 게임에서의 **상태 추상화(state abstraction)** 문제를 해결하기 위한 새로운 접근 방식인 **크기 제한 상태 추상화(Size-Constrained State Abstraction, SCSA)**를 제안합니다. SCSA는 **상태 추상화의 품질이 낮아질 가능성을 완화하는 데 초점을 맞춥니다.** 기존의 **탄력적 MCTS(Elastic MCTS)**와 같은 방법은 추상화의 품질 저하를 방지하기 위해 **조기에 추상화를 포기하는 메커니즘을 사용했지만**, SCSA는 이러한 문제를 해결하기 위해 **특정 그룹 내 최대 노드 수를 제한하는 방식을 적용합니다.**

- **Technical Details**: SCSA는 각 노드 그룹의 최대 노드 수를 제한하여 **추상화를 포기하지 않고도 낮은 품질의 추상화 문제를 해결**합니다. 또한, SCSA는 탄력적 MCTS에 비해 **하이퍼파라미터에 대한 민감도가 낮아 설정이 더욱 용이합니다.**

- **Performance Highlights**: SCSA는 **3가지 전략 게임(Kill The King, Push Them All, Two Kingdoms)**에서 실험을 통해 기존 방법보다 우수한 성능을 보여주었습니다. 특히 **Kill The King과 Push Them All 게임에서 탁월한 성과를 달성**했으며, **Two Kingdoms 게임에서는 탄력적 MCTS와 유사한 수준의 성능을 보였습니다.** SCSA는 **메모리 사용량과 성능 간의 균형을 유지**하며, 더 낮은 압축률(compression rate)을 보여주었습니다.



### Dynamic Blocked Clause Elimination for Projected Model Counting (https://arxiv.org/abs/2408.06199)
Comments:
          LIPIcs, Volume 305, SAT 2024

- **What's New**: 본 논문은 **투영 모델 계산(Projected Model Counting)**을 위한 **차단 절(Blocked Clause)** 제거의 적용을 탐구합니다.  이것은 주어진 변수 집합 X를 존재적으로 제거한 후 명제 공식 Σ의 모델 수 ||∃X.Σ||를 결정하는 문제입니다. 차단 절 제거는 SAT 해결을 위한 잘 알려진 기법이지만, 일반적으로 모델 수를 변경하기 때문에 모델 계산에 직접 적용하기는 어렵습니다.  그러나 본 논문에서는 차단 절 검색 중에 투영된 변수에 집중함으로써 차단 절 제거를 적용하면서 정확한 모델 수를 유지할 수 있음을 보여줍니다.  모델 계산 중에 차단 절 제거를 효율적으로 활용하기 위해 새로운 데이터 구조와 관련 알고리즘이 소개됩니다.  제안된 접근 방식은 모델 계산기 d4에 구현되었습니다.  실험 결과는 투영 모델 계산을 위한 차단 절 제거 방법의 계산적 이점을 보여줍니다.



### Online Optimization of Curriculum Learning Schedules using Evolutionary Optimization (https://arxiv.org/abs/2408.06068)
Comments:
          8 pages including abstract, to be published in the Proceedings of the IEEE Conference on Games 2024

- **What's New**: RHEA CL algorithm: This research proposes a novel algorithm, RHEA CL, which combines Curriculum Learning (CL) with Rolling Horizon Evolutionary Algorithms (RHEA) to create dynamic and effective training curricula for reinforcement learning agents. This approach allows the algorithm to automatically optimize the curriculum during training, potentially leading to better performance and faster learning.



### Perceptual Similarity for Measuring Decision-Making Style and Policy Diversity in Games (https://arxiv.org/abs/2408.06051)
Comments:
          TMLR 08/2024 this https URL

- **What's New**: 이 논문은 게임에서 플레이 스타일을 측정하는 방법인 'Playstyle Distance'를 개선한 'Playstyle Similarity'를 제안합니다. 기존 방법의 한계를 극복하기 위해 다중 스케일 분석, 심리 물리학적 커널, 교집합-합집합 비율(intersection-over-union) 방법을 도입했습니다. 이를 통해 플레이 스타일 분류 정확도를 향상시켰고, 게임 상황을 이해하는 새로운 관점을 제시합니다.



### Exploring and Learning Structure: Active Inference Approach in Navigational Agents (https://arxiv.org/abs/2408.05982)
Comments:
          IWAI workshop 2024

- **What's New**: 본 논문은 동물의 탐색 전략에서 영감을 받아, 생물학적으로 영감을 받은 원칙에 기반한 새로운 탐색 및 매핑(mapping)을 위한 계산 모델을 제시합니다. 동물들은 복잡하고 애매한 환경을 효율적으로 탐색하기 위해 기억, 상상, 전략적 의사 결정을 사용하는 놀라운 탐색 능력을 보여줍니다. 본 연구는 이러한 통찰력을 바탕으로 기존 인지 매핑(cognitive mapping) 방식과 능동 추론 프레임워크(AIF, Active Inference Framework)를 통합하여 몇 단계만에 환경 구조를 학습합니다. 토폴로지 매핑(topological mapping)을 장기 기억에 통합하고, 탐색 계획 및 구조 학습에 AIF를 통합하여, 모델은 탐색 중에 예측된 믿음(belief)을 통해 환경 구조를 동적으로 이해하고 내부 지도를 확장할 수 있습니다. CSCG(Clone-Structured Graph) 모델과의 비교 실험을 통해, 본 모델이 단일 에피소드에서 최소한의 탐색 중복(navigation overlap)을 통해 환경 구조를 빠르게 학습할 수 있다는 점을 강조합니다. 이는 환경의 크기나 관찰 유형에 대한 사전 지식 없이 이루어지며, 애매한 환경을 탐색하는 데 있어 모델의 견고성과 효율성을 보여줍니다.



### Match Point AI: A Novel AI Framework for Evaluating Data-Driven Tennis Strategies (https://arxiv.org/abs/2408.05960)
Comments:
          4 pages, 1 page abstract, short paper, to be published in Proceedings of the IEEE Conference on Games 2024

- **What's New**: This paper introduces **Match Point AI**, a novel framework for simulating tennis matches. This framework empowers researchers to explore tennis strategies through AI agents by creating a realistic virtual environment.

- **Technical Details**: Match Point AI leverages a data-driven approach, utilizing real-world data from the **Match Charting Project** (2017-2023) to model player behavior. It represents a tennis match as a non-deterministic game where shots are encoded with direction, error probability, and winner probability, dependent on the current game state. This game state considers factors like previous shot direction, serve type, and player initiating the rally.

- **Performance Highlights**: Initial experiments demonstrate the capability of Match Point AI in generating realistic shot-by-shot data for simulated matches. This data closely resembles real-world data patterns. Furthermore, the framework enables the application of **Monte Carlo Tree Search (MCTS)** to identify effective shot direction strategies in tennis rallies, highlighting promising insights into optimal playstyles.

- **Future Extensions**: The framework is expected to be further developed by incorporating additional factors such as player positions, movement directions, and specialized shot types like slices, volleys, and lobs, further enhancing its realism and analytical power.



### Markov Senior -- Learning Markov Junior Grammars to Generate User-specified Conten (https://arxiv.org/abs/2408.05959)
Comments:
          8 pages, to be published in the Proceedings of the IEEE Conference on Games 2024, demo implementation can be found here: this https URL

- **What's New**: Markov Senior, a 새로운 알고리즘, Markov Junior의 규칙을 자동으로 학습하는 데 사용됩니다. Markov Junior는  프로시저럴 콘텐츠 제너레이션(PCG)에 사용되는 확률적 프로그래밍 언어입니다.  Markov Senior는 주어진 샘플에서 위치 및 거리 관계를 추출하여 Markov Junior에서 사용할 확률적 규칙을 구성합니다. Kullback-Leibler Divergence 기반 적합성 측정을 사용하여 주어진 샘플과 일관성 있는 콘텐츠를 생성하는 문법을 검색합니다.  



### BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation (https://arxiv.org/abs/2408.05926)
Comments:
          ECCV 2024

- **What's New**: This paper introduces BI-MDRG, a novel model for Multimodal Dialogue Response Generation (MDRG), which addresses the limitations of existing MDRG models by effectively integrating image history information into both textual and visual responses. This integration enhances image-grounded text responses and ensures consistency of objects in sequential image responses.



### Urban Region Pre-training and Prompting: A Graph-based Approach (https://arxiv.org/abs/2408.05920)
- **What's New**: 본 논문에서는 도시 지역 표현 학습을 위한 새로운 프레임워크인 GURPP (Graph-based Urban Region Pre-training and Prompting Framework)를 제안합니다. GURPP는 도시 지역의 공간적 구조와 기능적 레이아웃을 포착하는 그래프 기반의 사전 학습 및 프롬프팅 모델입니다.



### Leveraging Knowledge Graph-Based Human-Like Memory Systems to Solve Partially Observable Markov Decision Processes (https://arxiv.org/abs/2408.05861)
- **What's New**: 본 논문에서는 부분 관측 가능한 마르코프 의사 결정 프로세스(POMDP) 환경을 사용하여 인공 지능(AI) 에이전트가 장기 기억 시스템을 배우고 활용하는 방법을 조사했습니다. 이 환경은 에이전트가 미로를 탐색하면서 질문에 답해야 하는 지식 그래프(KG) 기반이며, 숨겨진 상태는 동적 KG로 구성됩니다. KG는 사람과 기계 모두 읽을 수 있기 때문에 에이전트가 무엇을 기억하고 잊는지 쉽게 알 수 있습니다.



### Root Cause Attribution of Delivery Risks via Causal Discovery with Reinforcement Learning (https://arxiv.org/abs/2408.05860)
- **What's New**: This paper proposes a novel approach to root cause attribution of delivery risks in supply chains by integrating causal discovery with reinforcement learning. This method aims to address the limitations of traditional methods that often identify spurious correlations and fail to capture intricate interrelationships in complex supply chains.



### The Cognitive Revolution in Interpretability: From Explaining Behavior to Interpreting Representations and Algorithms (https://arxiv.org/abs/2408.05859)
- **What's New**: 이 논문은 기계적 해석성(MI, Mechanistic Interpretability)을 인지 과학(Cognitive Science)의 맥락에서 이해하려고 시도합니다. 특히, 인간 두뇌와 같은 '블랙박스' 지능 시스템의 작동 방식을 이해하는 데 어려움을 겪고 있는 인지 과학에서 얻은 통찰력을 활용합니다.

- **Technical Details**: 논문은 MI 연구를 두 가지 범주로 분류합니다: 1) 의미 해석(Semantic Interpretation): 모델이 학습하고 사용하는 잠재적 표현(latent representations)을 연구하는 것, 2) 알고리즘 해석(Algorithmic Interpretation): 모델이 특정 행동을 수행하기 위해 표현에 대해 수행하는 연산과 알고리즘을 연구하는 것.  이 논문은 두 범주 내에서 다양한 접근 방식의 유사점과 차이점을 분석하고, 대표적인 연구의 강점과 약점을 분석하며, 기본 가정을 명확히 하고, 주요 과제를 제시하며, 공통 프레임워크 하에서 이러한 해석 방식을 통합할 가능성을 논의합니다.

- **Performance Highlights**: None



### Scaling Virtual World with Delta-Engin (https://arxiv.org/abs/2408.05842)
- **What's New**: 이 논문에서는 가상 세계의 진화하는 특성을 강조하며, 사용자의 성장과 세계에 영향을 미치는 동적 변화를 구현하기 위한 델타 엔진을 소개합니다.

- **Technical Details**: 델타 엔진은 기본 엔진과 신경 프록시(neural proxy)로 구성되며, 프록시는 기본 엔진을 기반으로 증분 예측(incremental prediction)을 통해 새로운 코드를 생성합니다. 특히, 대규모 언어 모델(LLM)을 프록시로 활용하여 자연어, 스크립트, 이미지 등 다양한 입력을 처리합니다.

- **Performance Highlights**: 델타 엔진은 세계 내 알려지지 않은 요소에 대한 확장성을 제공하며, 기본 엔진과 프록시의 협력적인 작동과 고품질 데이터 정렬을 통해 이루어집니다. 엔진 중심 미세 조정 방법(engine-oriented fine-tuning)을 통해 기본 엔진을 프록시에 통합하고, 인간과 AI의 협업 설계 프로세스를 통해 새로운 데이터를 효율적으로 생성합니다. 또한 델타 엔진의 성능을 평가하기 위해 나이브 평가, 증분 평가, 적대적 평가 등 3가지 원칙을 제시합니다.

- **Domain**: 가상 세계, 델타 엔진, 증분 예측, 대규모 언어 모델(LLM), 신경 프록시, 엔진 중심 미세 조정, 인간-AI 협업 설계, 나이브 평가, 증분 평가, 적대적 평가

- **Contributions**: 델타 엔진을 통한 가상 세계의 동적 진화 구현, 프록시를 위한 LLM 활용 및 증분 예측 방법, 고품질 데이터 정렬을 위한 인간-AI 협업 설계 프로세스, 델타 엔진의 성능을 평가하기 위한 3가지 원칙 제시



### HateSieve: A Contrastive Learning Framework for Detecting and Segmenting Hateful Content in Multimodal Memes (https://arxiv.org/abs/2408.05794)
Comments:
          8 pages overall, the accepted paper at the 3rd Workshop on Advances in Language and Vision Research (ALVR 2024) ACL workshops

- **What's New**: 본 논문에서는 **HateSieve**라는 새로운 프레임워크를 소개합니다. HateSieve는 **Confounder Memes (혼란스러운 밈)**에서 혐오 콘텐츠를 감지하고 분할하는 데 효과적인 방법입니다. 이 프레임워크는 **Contrastive Meme Generator (대조적 밈 생성기)**, **Triplet Dataset (삼중 데이터셋)**, **Image-Text Alignment (ITA, 이미지-텍스트 정렬) 모듈**을 사용하여 혐오 콘텐츠를 정확하게 파악하고 분리합니다.



### Neurosymbolic Methods for Rule Mining (https://arxiv.org/abs/2408.05773)
- **What's New**: 이 장에서는 규칙 마이닝 문제를 다루고, 규칙 품질 측정과 같은 필수적인 배경 정보를 시작으로 합니다. 그런 다음 귀납 논리 프로그래밍, 경로 샘플링 및 일반화, 선형 프로그래밍의 세 가지 그룹으로 분류된 다양한 규칙 마이닝 방법론을 살펴봅니다. 이어서 딥 러닝과 규칙의 통합, 임베딩을 사용한 규칙 학습, 대규모 언어 모델의 규칙 학습 적용과 같은 주제를 다루는 신경 기호 방법론을 살펴봅니다.



### Low-Dimensional Federated Knowledge Graph Embedding via Knowledge Distillation (https://arxiv.org/abs/2408.05748)
- **What's New**: 본 논문에서는 분산된 지식 그래프(KG)에서 여러 클라이언트 간의 협력적인 학습을 가능하게 하는 연합 지식 그래프 임베딩(FKGE)에 대한 새로운 경량화 방법인 FedKD를 제안합니다. FedKD는 기존 FKGE 방법에 보조적으로 작동하여 사전 학습된 고차원 티처 모델의 정보를 저차원 스튜던트 모델로 전송합니다. 이를 통해, FKGE의 성능을 유지하면서 임베딩 크기를 줄입니다. 기존의 지식 증류 방식과 달리, FedKD는 Adaptive Asymmetric Temperature Scaling 메커니즘을 사용하여 티처 모델의 오버컨피던스 문제를 해결합니다. 또한, 하드 라벨 손실과 소프트 라벨 손실의 최적화 목표 차이를 해소하기 위해 소프트 라벨 손실의 가중치를 동적으로 조정합니다.



### Top Pass: Improve Code Generation by Pass@k-Maximized Code Ranking (https://arxiv.org/abs/2408.05715)
Comments:
          Accepted by Frontier of Computer Science

- **What's New**: 본 논문은 코드 생성 시스템에서 사용자의 검토 및 테스트 시간을 줄이기 위한 새로운 코드 순위 알고리즘인 'Top Pass'를 제안합니다. Top Pass는 코드 생성 모델이 생성한 후보 프로그램들의 정확도를 예측하고, 정확한 솔루션을 우선 순위로 배치하는 방법입니다. 이를 통해 사용자는 가능한 한 적은 수의 후보 코드를 검토함으로써 정확한 솔루션을 빠르게 찾을 수 있습니다.



### Separate Generation and Evaluation for Parallel Greedy Best-First Search (https://arxiv.org/abs/2408.05682)
Comments:
          In Proceedings of ICAPS-2024 Workshop on Heuristics and Search for Domain-Independent Planning (HSDIP-24) this https URL

- **What's New**: 본 논문은 **Separate Generation and Evaluation (SGE)**라는 새로운 접근 방식을 제안하여 제약 조건이 있는 병렬 GBFS에서 상태 평가 속도를 크게 향상시켰습니다. SGE는 상태 확장과 평가를 분리하여 여러 스레드가 상태의 후계자를 평가하는 데 동시에 사용될 수 있도록 합니다.



### In-Context Exploiter for Extensive-Form Games (https://arxiv.org/abs/2408.05575)
- **What's New**: 이 논문은 **게임 이론**에서 널리 사용되는 해결책인 **내쉬 균형 (Nash Equilibrium)** 의 한계를 다룹니다. 내쉬 균형은 **안정성**이라는 장점이 있지만, 상대방이 내쉬 균형 전략을 따르지 않을 경우 항상 최상의 결과를 보장하지 못합니다. 따라서 이 논문에서는 게임 이론에서 **새로운 해결 문제**를 제시합니다: **내쉬 균형을 포함한 모든 상대방을 이용하여 자신의 이익을 극대화할 수 있는 모델을 학습할 수 있을까요?**



### Metacognitive Myopia in Large Language Models (https://arxiv.org/abs/2408.05568)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 편향을 설명하기 위해 새로운 이론적 틀인 "메타인지 근시"(metacognitive myopia)를 제시합니다. 기존 연구들은 LLM의 편향을 주로 인간 어노테이터와 훈련 데이터 선택의 문제로 여겨왔지만, 이 논문은 LLM의 계산 과정 자체에서 발생하는 문제를 강조합니다. 메타인지 근시는 LLM이 메타인지의 두 구성 요소인 모니터링과 제어 기능이 부족하여 발생하는 현상으로, LLM이 정보를 잘못 처리하고 편향된 출력을 생성하는 이유를 설명합니다.



### Structure and Reduction of MCTS for Explainable-AI (https://arxiv.org/abs/2408.05488)
Comments:
          ECAI 2024

- **What's New**: 이 논문은 알파제로(AlphaZero) 알고리즘과 같은 복잡한 순차적 의사 결정 계획 문제(sequential decision-making planning problems)의 설명 가능성(explainability)을 개선하기 위해 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 데이터 구조 내의 정보를 활용하는 새로운 방법을 제시합니다. 이 논문은 MCTS의 구조적 정보를 추출하고 간소화하는 정보 이론적 도구를 사용하여 인간이 이해할 수 있는 설명을 생성하는 방법을 다룹니다. 또한 이러한 설명은 MCTS의 크기를 줄이면서도 정보 손실을 최소화하기 위한 효율적인 트리 축소 알고리즘을 통해 더욱 간결해집니다.



### Multi-agent Planning using Visual Language Models (https://arxiv.org/abs/2408.05478)
- **What's New**: 이 논문에서는 자유형 영역(free-form domains)에서 구현된 에이전트(embodied agents)를 위한 기반 모델(Foundation Models, FMs) 기반 계획(planning) 시스템을 제안합니다. 이 시스템은 복잡한 구조화된 데이터 구조(structured data structures) 대신 이미지 단 한 장만 사용하여 상식 지식(commonsense knowledge)을 활용하여 계획을 수립합니다. 또한 계획의 질을 더 잘 평가하기 위한 새로운 자동 평가 절차(PG2S)를 소개합니다.



### SHIELD: LLM-Driven Schema Induction for Predictive Analytics in EV Battery Supply Chain Disruptions (https://arxiv.org/abs/2408.05357)
Comments:
          30 pages, 11 figures, Project: this https URL

- **What's New**: SHIELD (Schema-based Hierarchical Induction for EV supply chain Disruption) 시스템은 전기 자동차 (EV) 배터리 공급망의 위험 분석을 위해 대규모 언어 모델 (LLM)을 도메인 전문 지식과 통합합니다.

- **Technical Details**: SHIELD는 다음과 같은 기능을 통합합니다: (1) LLM 기반 스키마 학습을 통해 포괄적인 지식 라이브러리를 구축합니다. (2) 이벤트 추출을 위한 미세 조정된 언어 모델, 스키마 매칭을 위한 다차원 유사성 매칭, 예측을 위한 논리적 제약 조건이 있는 그래프 합성곱 신경망 (GCN)을 활용하는 중단 분석 시스템입니다. (3) 결과를 시각화하고 전문가 피드백을 통합하여 의사 결정을 개선하는 대화형 인터페이스입니다.

- **Performance Highlights**: 2022년에서 2023년 사이 365개 출처에서 12,070개의 단락을 평가한 결과, SHIELD는 기준 GCN 및 LLM+프롬프트 방법(예: GPT-4o)보다 중단 예측에서 뛰어난 성능을 보였습니다.

- **Keywords**: LLM, EV battery supply chain, risk assessment, schema learning, disruption analysis, Graph Convolutional Networks (GCN), fine-tuned RoBERTa, multi-dimensional similarity matching, human-in-the-loop



### Revisiting Multi-Modal LLM Evaluation (https://arxiv.org/abs/2408.05334)
- **What's New**: 본 연구는 기존의 비주얼 퀘스천 앤서링(VQA) 데이터셋의 한계를 극복하고 최신 멀티모달 대규모 언어 모델(MLLM)을 위한 새로운 평가 데이터셋을 제시합니다. 새로운 데이터셋은 다양한 질문 유형과 시각적 추론 능력을 평가할 수 있도록 설계되었으며, 특히 시각적 지각(visual grounding) 능력을 더욱 엄격하게 평가하는 데 중점을 둡니다.

- **Technical Details**: 본 연구에서는 다음과 같은 데이터셋을 사용합니다:

- **TDIUC**: 12가지 질문 유형을 포함하여 세분화된 분석을 가능하게 하는 VQA 데이터셋.
- **TallyQA**: 간단하고 복잡한 계산 질문을 포함하는 VQA 데이터셋.
- **DVQA**: 차트 이해를 위해 광학 문자 인식(OCR)을 필요로 하는 VQA 데이터셋.
- **VQDv1**: 주어진 질의에 맞는 모든 이미지 영역을 식별해야 하는 VQA 데이터셋. 

또한 다음과 같은 최신 MLLM 모델을 평가합니다:

- **LLaVA 1.5**, **LLaVA-NeXT**, **BLIP2**, **InstructBLIP**, **GPT-4V**, **GPT-4o**

- **Performance Highlights**: 본 연구에서는 새로운 평가 데이터셋을 사용하여 최신 MLLM의 성능을 분석한 결과, 기존의 데이터셋에서는 발견되지 않았던 새로운 약점들이 드러났습니다. 예를 들어, 일부 MLLM은 복잡한 시각적 추론 능력을 요구하는 VQDv1에서 저조한 성능을 보였습니다. 이는 이러한 모델들이 단일 객체 식별에 의존하는 기존의 데이터셋에 과도하게 적응되어 있기 때문일 수 있습니다.



### Can a Bayesian Oracle Prevent Harm from an Agent? (https://arxiv.org/abs/2408.05284)
- **What's New**: 이 논문은 기계 학습 방식을 기반으로 한 강력한 AI 시스템을 설계하는 방법에 대한 새로운 연구를 제시합니다. 특히, AI 시스템이 안전 규격을 위반할 가능성을 제한하는 확률적 안전 보장을 제공하는 방법을 살펴봅니다. 이 논문은 컨텍스트에 따라 안전 규격 위반 확률을 경계하는 방법을 제안하며, 이는 잠재적으로 위험한 AI 행동을 방지하기 위해 런타임에 수행될 수 있습니다.



### Large Language Model based Agent Framework for Electric Vehicle Charging Behavior Simulation (https://arxiv.org/abs/2408.05233)
Comments:
          7 pages,3 figures

- **What's New**: 이 논문은 사용자 선호도, 심리적 특징 및 환경 요인을 통합하여 전기 자동차(EV) 충전 프로세스를 최적화하는 새로운 LLM 기반 에이전트 프레임워크를 소개합니다. 이 프레임워크는 사용자 기대와 효율성 향상을 보장하기 위해 지속적인 반성과 메모리 업데이트를 통해 동적 의사 결정을 지원하는 여러 모듈로 구성됩니다. 개인화된 사용자 프로필과 실시간 의사 결정을 생성할 수 있는 프레임워크의 기능은 도시 EV 충전 관리에 상당한 발전을 가져옵니다. 향후 연구는 예측 정확도와 실용성을 높이기 위해 더 복잡한 시나리오를 통합하고 데이터 소스를 확장하는 데 집중할 수 있습니다.



### LOLgorithm: Integrating Semantic,Syntactic and Contextual Elements for Humor Classification (https://arxiv.org/abs/2408.06335)
- **What's New**: 본 논문은 자연어 처리(NLP)에서 계산적 방법보다 구문론적, 의미론적, 문맥적 특징을 우선시하여 유머 감지를 탐구합니다. Colbert라는 모델은 BERT 임베딩과 병렬 숨겨진 계층을 사용하여 문장 일관성을 포착합니다. SHAP 해석과 의사 결정 트리는 영향력 있는 특징을 식별하여 숨겨진 데이터에서 유머 감지 정확도를 향상시키는 포괄적인 접근 방식을 보여줍니다.



### Body Transformer: Leveraging Robot Embodiment for Policy Learning (https://arxiv.org/abs/2408.06316)
- **What's New**: 이 논문에서는 로봇의 신체 구조를 고려하여 트랜스포머 아키텍처를 개선한 **Body Transformer (BoT)**를 제안합니다. BoT는 로봇의 센서와 액추에이터를 그래프로 표현하고, 마스킹 된 어텐션(masked attention)을 사용하여 각 노드가 인접 노드의 정보만 참조하도록 합니다. 이를 통해 로봇 신체 구조에 대한 정보를 효과적으로 활용하여 학습 과정을 안내합니다.



### Synthetic Patient-Physician Dialogue Generation from Clinical Notes Using LLM (https://arxiv.org/abs/2408.06285)
- **What's New**: SynDial, 새로운 의료 대화 데이터 생성 방식은 단일 LLM(Large Language Model)을 활용하여 제로 샷 프롬프팅(zero-shot prompting)과 피드백 루프를 통해 고품질 의료 대화 데이터를 생성합니다. 특히 SynDial은 유사성(similarity)과 추출성(extractiveness)에 대한 가중 평가 점수를 기반으로 피드백 루프를 통해 대화의 품질을 지속적으로 향상시킵니다. 이는 기존 방법과 차별화되는 중요한 특징이며, 더 나은 대화 데이터 생성을 가능하게 합니다.



### MovieSum: An Abstractive Summarization Dataset for Movie Screenplays (https://arxiv.org/abs/2408.06281)
Comments:
          ACL 2024 Findings

- **What's New**: 이 논문은 영화 각본 요약을 위한 새로운 데이터셋인 "MovieSum"을 소개합니다. MovieSum은 2200개의 영화 각본과 해당 위키피디아 줄거리 요약으로 구성되어 있습니다. 기존 데이터셋에 비해 MovieSum은 다음과 같은 특징을 가지고 있습니다: (1) TV 드라마보다 긴 영화 각본을 포함합니다. (2) 이전 영화 각본 데이터셋보다 두 배 더 큰 규모를 가지고 있습니다. (3) 추가적인 외부 정보 접근을 용이하게 하기 위해 IMDb ID가 포함된 메타데이터를 제공합니다.



### Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignmen (https://arxiv.org/abs/2408.06266)
- **What's New**: 이 논문은 AI 모델의 정렬(alignment) 성능을 향상시키기 위한 두 가지 새로운 방법인 CLAIR와 APO를 제시합니다. CLAIR는 최소한의 수정을 통해 선호도 쌍(preference pairs)을 생성하는 데이터 생성 방법이고, APO는 모델과 데이터 간의 관계를 명확하게 고려하는 제어 가능한 정렬 목표입니다.



### Audio Enhancement for Computer Audition -- An Iterative Training Paradigm Using Sample Importanc (https://arxiv.org/abs/2408.06264)
- **What's New**: 본 논문에서는 음성 인식, 음성 명령 인식, 음성 감정 인식, 음향 장면 분류 등 다양한 오디오 작업에서 소음에 대한 모델의 견고성을 높이는 새로운 프레임워크를 제시합니다. 이 프레임워크에서는 음향 강화(AE) 모델과 대상 오디오 작업 모델을 함께 학습시켜 상호 작용을 강화합니다. 특히, 샘플별 성능 지표를 사용하여 AE 모듈을 특정 작업에 맞게 최적화하고 어려운 샘플을 극복합니다. 즉, AE 모듈은 각 작업에 필요한 정보(예: 음성 인식의 경우 음성 명료도, 음성 감정 인식의 경우 음성의 높낮이)를 보존하도록 학습됩니다. 또한, 각 샘플의 어려움을 나타내는 CAT 모델의 손실을 사용하여 어려운 샘플에 대한 AE 학습을 유도합니다. 이러한 접근 방식은 AE와 CAT 모델 간의 상호 의존적인 특성을 활용하여 시스템 전체의 성능을 최적화합니다.



### Open-Source Molecular Processing Pipeline for Generating Molecules (https://arxiv.org/abs/2408.06261)
Comments:
          Presented at the 2024 Molecular Machine Learning Conference (MoML 2024)

- **What's New**: 이 연구는 널리 사용되는 분자 머신 러닝 라이브러리인 DeepChem에 오픈소스 분자 생성 모델 인프라를 도입하여 전문가가 아닌 사용자도 쉽게 생성 모델을 활용할 수 있도록 지원합니다. 특히, 연구진은 MolGAN (Molecular Generative Adversarial Networks)과 Normalizing Flows 두 가지 유명한 분자 생성 모델을 PyTorch로 구현하여 DeepChem에 통합했습니다. 이 구현은 기존 연구 결과와 비교하여 뛰어난 성능을 보여줍니다.



### Decentralized Intelligence Health Network (DIHN) (https://arxiv.org/abs/2408.06240)
Comments:
          17 pages, 7 figures. arXiv admin note: substantial text overlap with arXiv:2407.02461

- **What's New**: DIHN (Decentralized Intelligence Health Network)은 보건 데이터 주권과 의료 AI 활용의 과제를 해결하기 위한 이론적 틀입니다. DIHN은 의료 서비스 제공자와 기관 간 데이터 단편화로 인해 발생하는 문제들을 해결하고자 합니다. DIHN은 주권 기반 의료 네트워크를 위한 전제 조건으로서 주권 의료 서비스 아키텍처를 구축하고 다양한 의료 데이터 소스에 대한 접근 장벽을 극복하여 효과적인 AI 활용을 가능하게 합니다.



### FLEURS-R: A Restored Multilingual Speech Corpus for Generation Tasks (https://arxiv.org/abs/2408.06227)
- **What's New**: FLEURS-R은 다양한 언어 (102개 언어)에 대한 음성 복원 (speech restoration) 데이터셋으로, FLEURS 데이터셋을 기반으로 합니다. FLEURS-R은 Miipher 음성 복원 모델을 적용하여 기존 FLEURS 데이터셋의 음질과 정확성을 개선했습니다.  FLEURS-R은 저자원 언어 (low-resource languages)에서 음성 기술 발전을 목표로 하며, 텍스트 음성 변환 (TTS) 및 기타 음성 생성 작업 연구를 활성화하기 위해 고안되었습니다.



### A Large-Scale Study of Model Integration in ML-Enabled Software Systems (https://arxiv.org/abs/2408.06226)
- **What's New**: 본 논문은 GitHub의 2,928개 오픈 소스 시스템을 분석하여 실제 ML 기반 소프트웨어 시스템의 특징을 규명한 첫 번째 대규모 연구 결과를 제시합니다. 이 연구는 ML 모델 재사용 관행과 시스템 아키텍처를 분석하여 ML 기반 시스템 개발의 실제 관행에 대한 통찰력을 제공합니다.

- **Technical Details**: ML 기반 시스템은 기존 소프트웨어와의 통합, ML 모델의 재사용, ML 모델의 아키텍처 통합, 그리고 ML 모델의 구현과 실행을 위한 관련 코드와 같은 측면에서 분석되었습니다. 연구는 ML 모델이 기존 소프트웨어 시스템에 어떻게 통합되는지, ML 모델이 시스템에서 어떻게 재사용되는지, 그리고 ML 기반 소프트웨어 시스템의 일반적인 특징이 무엇인지에 대한 통찰력을 얻기 위한 세 가지 연구 질문을 제기했습니다.

- **Performance Highlights**: ML 기반 시스템의 대부분은 전통적인 소스 코드가 대부분을 차지하는 것을 발견했습니다.  이 연구는 ML 모델 재사용 관행에 대한 통찰력을 제공하고, 사전 훈련된 모델에 대한 강한 의존성과 시스템 간 소스 코드 복사 관행을 강조합니다. 또한, ML 기능을 제공하기 위해 26개 응용 프로그램을 심층 분석하여 흔히 사용되는 ML 통합 패턴과 코딩 관행 목록을 제시합니다.



### On Effects of Steering Latent Representation for Large Language Model Unlearning (https://arxiv.org/abs/2408.06223)
Comments:
          15 pages, 5 figures, 8 tables

- **What's New**: 이 논문은 Representation Misdirection for Unlearning (RMU)라는 효과적인 LLM 언러닝(unlearning) 방법의 작동 원리를 탐구합니다. RMU는 모델 표현을 중간 계층에서 임의의 표현으로 유도하여 언러닝을 수행합니다. 이 논문에서는 RMU의 효과를 이론적으로 분석하고, RMU가 언러닝에 미치는 영향을 설명합니다. 또한,  RMU의 한계점을 해결하기 위해 Adaptive RMU라는 새로운 방법을 제안합니다. Adaptive RMU는 RMU의 효과를 향상시키고, 대부분의 계층에서 효과적인 언러닝을 가능하게 합니다.



### ACCELERATION: Sequentially-scanning DECT Imaging Using High Temporal Resolution Image Reconstruction And Temporal Extrapolation (https://arxiv.org/abs/2408.06163)
- **What's New**: This paper proposes a novel technique called ACCELERATION to address temporal inconsistency in sequentially-scanning dual-energy computed tomography (DECT) data sets. ACCELERATION utilizes high temporal resolution image reconstruction and temporal extrapolation to improve iodine quantification accuracy in sequentially-scanning DECT.

- **Technical Details**: ACCELERATION reconstructs several time-resolved images at high temporal resolution using short-scan data acquired at the first tube potential. Temporal extrapolation is then applied to obtain the image corresponding to the acquisition time of the second tube potential. The temporally consistent images at two tube potentials are used for iodine quantification. The method leverages implicit neural representation learning, where a multi-layer perceptron (MLP) is trained to express the mapping relationship between pixel coordinates and pixel values. This mapping is used for image reconstruction and temporal extrapolation. ACCELERATION is validated and evaluated using numerical simulation data sets generated from clinical human subject exams.

- **Performance Highlights**: Results demonstrate the improvement of iodine quantification accuracy using ACCELERATION, addressing the challenge posed by temporal inconsistency in sequentially-scanning DECT data sets.



### Palantir: Towards Efficient Super Resolution for Ultra-high-definition Live Streaming (https://arxiv.org/abs/2408.06152)
- **What's New**: Palantir는 기존의 프레임 단위 스케줄링 방식을 뛰어넘어 픽셀 패치 단위의 미세한 스케줄링을 도입한 최초의 신경망 기반 UHD 실시간 스트리밍 시스템입니다. Palantir은 픽셀 패치에 대한 안커/비안커(anchor/non-anchor) 판단을 효율적으로 수행하여 컴퓨팅 오버헤드를 최소화하고 UHD 화질을 유지합니다.



### Med42-v2: A Suite of Clinical LLMs (https://arxiv.org/abs/2408.06142)
- **What's New**: Med42-v2는 의료 분야에서 일반적인 모델의 한계를 해결하도록 설계된 의료 전문 대규모 언어 모델(LLM) 모음입니다. 이 모델은 Llama3 아키텍처를 기반으로 구축되었으며 전문 의료 데이터를 사용하여 미세 조정되었습니다. 이 모델은 자연스러운 프롬프트에 효과적으로 응답하도록 다단계 선호도 정렬을 거쳤습니다. 일반적인 모델은 종종 예방 조치로 의료 질문에 답변하는 것을 피하도록 선호도가 정렬되지만, Med42-v2는 특히 이러한 한계를 극복하도록 훈련되어 의료 환경에서 사용할 수 있습니다. Med42-v2 모델은 8B 및 70B 매개변수 구성에서 원래 Llama3 모델과 GPT-4를 능가하여 다양한 의료 벤치마크에서 뛰어난 성능을 보여줍니다. 이러한 LLM은 의료 질문을 이해하고, 추론 작업을 수행하며, 의료 환경에서 귀중한 지원을 제공하도록 개발되었습니다.

- **Technical Details**: Med42-v2는 Llama3 아키텍처를 기반으로 구축된 의료 전문 대규모 언어 모델(LLM) 모음입니다. Med42-v2는 의료 관련 쿼리에 대한 응답 능력을 향상시키기 위해 전문 의료 데이터를 사용하여 미세 조정되었습니다. 이 모델은 또한 사용자 기대에 맞게 출력을 조정하기 위해 다단계 선호도 정렬을 거쳤습니다. 이는 의료 관련 질문에 답변하기를 꺼리는 일반적인 LLM과 대조적입니다. 이러한 훈련 과정은 Med42-v2가 의료 분야에서 사용하기 적합하도록 만들었습니다.

- **Performance Highlights**: Med42-v2는 8B 및 70B 매개변수 구성에서 원래 Llama3 모델과 GPT-4를 능가하여 다양한 의료 벤치마크에서 뛰어난 성능을 보여줍니다. 이 모델은 의료 질문을 이해하고, 추론 작업을 수행하며, 의료 환경에서 귀중한 지원을 제공하도록 설계되었습니다.



### A Methodological Report on Anomaly Detection on Dynamic Knowledge Graphs (https://arxiv.org/abs/2408.06121)
- **What's New**: This paper presents a novel approach to anomaly detection in dynamic knowledge graphs (DKGs), particularly focusing on Kubernetes microservices environments. It explores three representations of DKGs: sequential data, one-hop graph structure, and two-hop graph structure, each capturing increasing levels of structural complexity. Different machine learning (ML) and deep learning (DL) models are applied to each representation, with an ensemble learning strategy employed to combine their strengths.



### Generalization capabilities of MeshGraphNets to unseen geometries for fluid dynamics (https://arxiv.org/abs/2408.06101)
- **What's New**: 이 연구는 MeshGraphNets(MGN) [Pfaff 등, 학습 기반 메시 시뮬레이션 그래프 네트워크. ICML 2021]의 일반화 능력을 유체 역학에서 보이지 않는 기하학(예: 훈련 데이터에 포함되지 않은 새로운 장애물 주변의 흐름 예측)에 대해 조사합니다. 이를 위해 DeepMind의 실린더 주변 흐름 데이터 세트를 확장하여 다른 모양과 여러 개체를 포함하는 데이터 기반 계산 유체 역학(CFD)을 위한 새로운 벤치 마크 데이터 세트를 만듭니다. 그런 다음 이 새로운 데이터 세트를 사용하여 DeepMind가 MGN에 대해 수행한 일반화 실험을 확장하여 MGN이 다른 모양으로 얼마나 잘 일반화될 수 있는지 테스트합니다. 수치 테스트에서 MGN이 하나의 장애물 모양 데이터 세트에서 훈련하고 다른 장애물 모양 데이터 세트에서 테스트하여 다양한 모양으로 잘 일반화될 수 있음을 보여줍니다.



### Building Decision Making Models Through Language Model Regim (https://arxiv.org/abs/2408.06087)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 일반화 능력을 활용하여 의사 결정 문제를 해결하는 새로운 접근 방식인 "Learning then Using (LTU)"를 제안합니다. LTU는 전통적인 의사 결정 모델 훈련 방식과 달리, 다양한 도메인과 맥락에서 얻은 지식을 통합하여 기초 의사 결정 모델을 구축하는 "학습(learning)" 단계와 특정 의사 결정 시나리오에 맞게 이 기초 모델을 미세 조정하는 "사용(using)" 단계로 구성됩니다. 기존의 지도 학습 방식과 달리, LTU는 광범위한 사전 훈련과 목표 지향적인 미세 조정을 결합하여 다재다능한 훈련 방법론을 제공합니다.

- **Technical Details**: LTU는 Llama-2-13b (LLM)를 기반으로 하며, 인과적 언어 모델링(CLM) 훈련 방식과 트랜스포머 아키텍처를 사용합니다. 학습 단계에서는 다양한 의사 결정 도메인과 맥락에서 얻은 지식을 활용하여 기초 의사 결정 모델을 훈련합니다. 사용 단계에서는 특정 의사 결정 작업에 맞게 이 기초 모델을 미세 조정합니다.

- **Performance Highlights**: 전자 상거래 도메인(광고 및 검색 최적화)에서 실험한 결과, LTU는 기존의 지도 학습 방식보다 의사 결정 능력과 일반화 성능이 뛰어난 것으로 나타났습니다. 특히, LTU는 다양한 의사 결정 작업에 대해 지속적으로 우수한 성능을 보였으며, 단일 단계 및 다단계 의사 결정 작업에 적용 가능합니다. LTU는 게임 및 로봇 도메인을 넘어 다양한 의사 결정 과제를 해결하는 데 유연하고 강력한 프레임워크를 제공합니다.



### Fully Bayesian Differential Gaussian Processes through Stochastic Differential Equations (https://arxiv.org/abs/2408.06069)
- **What's New**: 기존의 깊은 가우시안 프로세스(Deep Gaussian Processes)는 이산적인 계층 구조를 사용하여 데이터 진화를 모델링하는 반면, 미분 가우시안 프로세스(DIFFGPs)는 무한히 깊은 가우시안 프로세스로 진화를 표현합니다. 그러나 기존 DIFFGP 방법은 종종 커널 하이퍼파라미터의 불확실성을 무시하고 고정 및 시간 불변으로 간주하여 연속 시간 모델과 근사 추론 간의 독특한 시너지 효과를 활용하지 못했습니다. 이 연구에서는 커널 하이퍼파라미터를 랜덤 변수로 취급하고 결합된 확률 미분 방정식(Stochastic Differential Equations, SDEs)을 구성하여 해당 후방 분포와 유도 포인트를 학습하는 완전 베이지안 접근 방식을 제안합니다. 하이퍼파라미터에 대한 추정 불확실성을 통합함으로써, 우리의 방법은 모델의 유연성과 복잡한 역학에 대한 적응성을 향상시킵니다. 또한, 우리의 접근 방식은 SDE 방법을 사용하여 변수를 결합하여 시간 변화, 포괄적이고 현실적인 후방 근사를 제공합니다. 실험 결과는 기존 접근 방식에 비해 우리 방법의 장점을 보여주며 유연성, 정확성 및 기타 지표 측면에서 우수한 성능을 보여줍니다. 우리의 연구는 베이지안 추론 발전을 위한 흥미로운 연구 분야를 열어주고 연속 시간 가우시안 프로세스를 위한 강력한 모델링 도구를 제공합니다.



### An Investigation Into Explainable Audio Hate Speech Detection (https://arxiv.org/abs/2408.06065)
Comments:
          Accepted to SIGDIAL 2024

- **What's New**: This research introduces a novel task: **explainable audio hate speech detection**. It aims to identify hate speech in audio recordings while pinpointing the precise time intervals (audio frame-level rationales) that justify the classification.



### Understanding Byzantine Robustness in Federated Learning with A Black-box Server (https://arxiv.org/abs/2408.06042)
Comments:
          We have released code on this https URL

- **What's New**: 이 연구는 블랙박스 서버가 있는 연합 학습(FL) 시스템의 바이잔틴 내성을 심층적으로 분석합니다. 특히, 서버가 사용하는 집계 규칙을 공격자에게 공개하지 않아 바이잔틴 공격의 영향을 줄일 수 있는지 여부를 조사합니다. 연구 결과는 블랙박스 서버가 동적 방어 전략을 사용하여 바이잔틴 공격에 대한 내성을 향상시킬 수 있음을 보여줍니다. 또한, 블랙박스 서버가 최악의 공격 영향을 최대 수준에서 기대 수준으로 감소시킬 수 있음을 이론적으로 증명하고 실험적으로 확인합니다. 이는 블랙박스 서버가 공격자에게 방어 전략에 대한 유용한 정보를 제공하지 않아 공격자가 최적의 공격 전략을 찾는 것을 효과적으로 방지하기 때문입니다.



### Spacetime $E(n)$-Transformer: Equivariant Attention for Spatio-temporal Graphs (https://arxiv.org/abs/2408.06039)
- **What's New**: 이 논문에서는 시공간 그래프 데이터에 대한 E(n)-동변환(equivariant) 트랜스포머 아키텍처를 소개합니다. 공간과 시간 모두에서 회전, 병진, 순열 동변환 유도적 편향(inductive bias)을 부과하여, 공간 시간 E(n)-트랜스포머(SET)가 대칭성을 유지하는 특성이 없는 순수 공간 또는 시간 모델보다 우수한 성능을 발휘한다는 것을 보여줍니다. 이 논문에서는 SET를 복잡한 역동성을 가진 단순한 물리 시스템인 충전된 N-입자 문제(charged N-body problem)에 대해 위 모델들과 비교 평가했습니다. 기존 시공간 그래프 신경망은 순차적 모델링에 초점을 맞춘 반면, 이 연구는 그래프 상에서 동적 시스템을 모델링하기 위해 기본 도메인 대칭성을 활용하면 상당한 성능 향상을 얻을 수 있다는 것을 실험적으로 보여줍니다.

- **Technical Details**: E(n)-동변환(equivariant) 특성은 공간과 시간 모두에서 회전, 병진 및 순열에 대한 변환 불변성(invariance) 및 동변환(equivariance)을 나타냅니다. 이는 네트워크가 입력 데이터의 기하학적 변형(예: 회전 또는 병진)에 대해 일관성을 유지한다는 것을 의미합니다. E(n)-동변환(equivariant) 트랜스포머는 공간 및 시간 동변환(equivariance)을 통해 이러한 특성을 달성합니다. 트랜스포머는 시간 구성 요소에 사용되어 장기 종속성을 보존하고 시간 왜곡(time-warping)에 대한 불변성(invariance)을 유지합니다. 각 노드는 그래프의 충전된 입자를 나타내고 특징, 좌표 및 속도를 갖습니다. 따라서 신경망은 좌표에 작용하는 회전 및 병진 대칭 E(n)에 대해 동변환(equivariant)이어야 합니다. 또한, 속도에 작용하는 회전 대칭 SO(n)에 대해 동변환(equivariant)이어야 합니다. 마지막으로 노드는 순열 동변환(permutation equivariant)이어야 합니다.

- **Performance Highlights**: SET는 충전된 N-입자 문제(charged N-body problem)에서 기존 시공간 그래프 신경망보다 우수한 성능을 보여주었습니다. 이는 도메인 대칭성을 유지하는 것이 시공간 그래프 데이터를 모델링하는 데 매우 중요하다는 것을 시사합니다. 특히, SET는 다른 모델보다 장기 종속성(long-term dependencies)을 더 잘 포착할 수 있었습니다.



### Peaking into the Black-box: Prediction Intervals Give Insight into Data-driven Quadrotor Model Reliability (https://arxiv.org/abs/2408.06036)
Comments:
          Presented at AIAA SciTech Forum 2023 in National Harbor, MD, USA

- **What's New**: 이 논문에서는 회전익 드론(quadrotor) 모델의 예측 신뢰도와 유효성을 높이기 위해 예측 구간(Prediction Intervals, PIs)을 추정하는 방법을 제시합니다. 특히, 회전익 드론의 공기 역학 모델을 다항식(polynomial)과 인공 신경망(ANN)을 사용하여 예측 구간을 추정하는 방법을 연구했습니다. 기존의 두 가지 ANN 기반 PI 추정 기법인 부트스트랩(bootstrap) 방법과 품질 중심(quality-driven) 방법을 검증하고, 실제 회전익 드론 비행 데이터에 적용하여 유용성과 모델 보간(interpolation) 및 외삽(extrapolation)에 대한 민감성을 분석했습니다. 특히, 외삽 시 ANN 기반 PIs가 크게 넓어지는 반면, 보간 시에는 일정하거나 좁아지는 경향을 보였습니다.



### Controlling Surprisal in Music Generation via Information Content Curve Matching (https://arxiv.org/abs/2408.06022)
Comments:
          8 pages, 4 figures, 2 tables, accepted at the 25th Int. Society for Music Information Retrieval Conf., San Francisco, USA, 2024

- **What's New**: 이 논문은 시퀀스 모델을 사용하여 음악 생성에서 놀라움(surprisal)을 제어하는 새로운 방법을 제안합니다. 새로운 지표인 순간 정보 내용(Instantaneous Information Content, IIC)을 정의하여 음악적 놀라움을 추정하고, 이를 사용하여 음악 생성 과정을 제어합니다. 특히, 빔 검색(beam search)을 사용하여 주어진 IIC 목표 곡선을 따라가는 IIC 곡선을 갖는 음악 자료를 생성합니다.



### Uncertainty-Informed Volume Visualization using Implicit Neural Representation (https://arxiv.org/abs/2408.06018)
Comments:
          To appear in IEEE Workshop on Uncertainty Visualization in conjunction with IEEE VIS 2024, Florida, USA

- **What's New**: 이 논문은 과학적 시각화 작업을 위한 불확실성 인식(Uncertainty-Aware) 암묵적 신경 표현(Implicit Neural Representation)을 제안합니다. 이 방법은 스칼라 필드 데이터 세트를 효과적으로 모델링하고 볼륨 시각화 작업에서 추정된 불확실성 정보의 효율성과 이점을 종합적으로 연구합니다. 특히, 두 가지 원칙적인 딥 불확실성 추정 기술인 딥 앙상블(Deep Ensemble)과 몬테카를로 드롭아웃(Monte Carlo Dropout, MCDropout)을 평가하여 스칼라 필드 데이터 세트에서 불확실성에 따른 볼륨 시각화를 가능하게 합니다.



### Transfer learning of state-based potential games for process optimization in decentralized manufacturing systems (https://arxiv.org/abs/2408.05992)
Comments:
          This pre-print was submitted to Computers in Industry on May 02, 2024

- **What's New**: 이 논문은 제조 시스템에서 분산된 자기 최적화를 향상시키기 위한 상태 기반 잠재적 게임(SbPGs)에서 새로운 전이 학습 접근 방식(TL-SbPGs)을 제시합니다. 이 접근 방식은 유사한 행동을 하는 플레이어 간에 획득한 지식을 공유하고 전이하여 대규모 시스템에서 자기 학습 메커니즘을 향상시키는 실질적인 산업 환경에 중점을 둡니다. TL-SbPGs를 사용하면 획득한 지식을 다른 플레이어가 정책을 최적화하는 데 재사용할 수 있으므로 플레이어의 학습 결과를 개선하고 학습 과정을 가속화할 수 있습니다.



### Freehand Sketch Generation from Mechanical Components (https://arxiv.org/abs/2408.05966)
Comments:
          Published at ACM Multimedia (ACM MM) 2024

- **What's New**: 본 논문에서는 기계 부품의 자유로운 손 그림 스케치를 생성하는 새로운 기술인 MSFormer를 제안합니다. 이 기술은 기존 기술의 한계인 손 그림 스타일 부족 및 생성 모델의 비효율성을 극복하여 인간의 스케치 행동 패턴을 모방하는 2단계 생성 프레임워크를 구축합니다. 특히 MSFormer는 인간 손 그림 스타일을 유지하면서 기계 부품의 필수 모델링 정보를 보존하는 데 초점을 맞춥니다.



### Robust online reconstruction of continuous-time signals from a lean spike train ensemble cod (https://arxiv.org/abs/2408.05950)
Comments:
          22 pages, including a 9-page appendix, 8 figures. A GitHub link to the project implementation is embedded in the paper

- **What's New**: 이 논문은 연속 시간 신호를 생물학적으로 가능한 스파이크 트레인으로 결정론적으로 인코딩하는 신호 처리 프레임워크를 제시하고, 표현 가능한 신호 클래스와 재구성 경계에 대한 질문을 다룹니다. 이 프레임워크는 다양한 컨볼루션 커널(convolution kernel)을 사용하는 컨볼루션-후-임계값(convolve-then-threshold) 메커니즘을 통해 뉴런 집합으로 생성된 스파이크 트레인을 통한 신호 인코딩을 고려합니다. 일반화된 FRI(Finite Rate of Innovation) 신호 클래스의 희소 표현을 보장하는 이동된 커널 함수의 힐베르트 공간에서 스파이크 트레인에서 신호 재구성까지의 역 문제에 대한 폐쇄 형 해(closed-form solution)가 도출됩니다. 또한 생물학적 시스템에서 실시간 처리에서 영감을 받아, 덜 조절된 인코딩에 대한 기술의 견고성을 보장하는, 과거 스파이크의 유한한 창(finite window)만을 고려하는 최적 재구성의 효율적인 반복 버전이 공식화되었습니다. 최적 솔루션에 대한 창 기반 재구성의 수렴 보장이 제공됩니다.



### Multimodal Large Language Models for Phishing Webpage Detection and Identification (https://arxiv.org/abs/2408.05941)
Comments:
          To appear in eCrime 2024

- **What's New**: 본 논문에서는 피싱 웹페이지 탐지에 대한 새로운 접근 방식을 제시하며, 특히 대규모 언어 모델 (LLM, Large Language Model)을 활용하여 기존의 브랜드 기반 피싱 탐지 시스템의 단점을 해결합니다. 기존 시스템은 브랜드 이미지 데이터 셋 구축 및 유지 관리에 어려움을 겪었으나, 이 논문에서는 LLM의 강력한 텍스트 및 이미지 이해 능력을 활용하여 웹페이지의 브랜드를 식별하고 도메인 이름과 비교하여 피싱 공격을 탐지합니다.



### Spb3DTracker: A Robust LiDAR-Based Person Tracker for Noisy Environmen (https://arxiv.org/abs/2408.05940)
Comments:
          17 pages, 5 figures

- **What's New**: 본 논문은 LiDAR 기반 사람 검출 및 추적 (PDT) 분야의 주요 과제를 다루며,  기존 TBD(Tracking-by-Detection) 방식의 한계를 극복하기 위한 SpbTrack 이라는 강력한 사람 추적 알고리즘을 제안합니다.  SpbTrack은 다양한 환경에서 뛰어난 성능을 보이며 특히 소음이 많은 데이터셋에서 탁월한 결과를 제공합니다. KITTI 데이터셋과 맞춤형 사무실 실내 데이터셋에서 최첨단 성능을 달성합니다.  



### Optimizing RAG Techniques for Automotive Industry PDF Chatbots: A Case Study with Locally Deployed Ollama Models (https://arxiv.org/abs/2408.05933)
- **What's New**: 본 연구는 오프라인 PDF 챗봇의 자동차 산업 생산 환경에서의 수요 증가에 따라, 저성능 로컬 환경에서 대규모 언어 모델(LLMs, Large Language Models)의 효율적인 배포를 위한 연구를 다룹니다. 특히, 로컬 환경에 배포된 Ollama 모델을 사용하여 자동차 산업 문서를 처리하는 Retrieval-Augmented Generation (RAG) 기법의 성능 향상에 중점을 둡니다. Langchain 프레임워크를 기반으로, Ollama의 로컬 RAG 구현을 위한 다차원적 최적화 방안을 제시합니다. 본 연구는 자동차 산업 문서 처리의 핵심 과제인 다열 레이아웃(multi-column layouts)과 기술 사양(technical specifications) 처리에 초점을 맞추어, 자동차 산업 문서의 특성에 맞춰 PDF 처리, 검색 메커니즘 및 컨텍스트 압축(context compression)을 개선합니다. 또한, 임베딩 파이프라인(embedding pipelines)을 지원하는 맞춤형 클래스와 LangGraph 모범 사례를 기반으로 Self-RAG를 지원하는 에이전트를 설계합니다.



### Adapting a Foundation Model for Space-based Tasks (https://arxiv.org/abs/2408.05924)
- **What's New**: 본 논문은 우주 로봇 공학에 대한 적용을 위해 **기초 모델**(Foundation Model)을 개발하는 첫 번째 단계로, **AI4Mars 데이터셋**을 이용하여 시각-질문-답변 튜플의 언어 주석이 달린 데이터셋을 만듭니다. 또한, **LLaVA**를 이 데이터셋에 미세 조정하여 시각-언어 모델에 **우주 공간 추론과 탐색**(spatial reasoning and navigation) 능력을 부여합니다.



### Inverse design of Non-parameterized Ventilated Acoustic Resonator via Variational Autoencoder with Acoustic Response-encoded Latent Spac (https://arxiv.org/abs/2408.05917)
- **What's New**: 본 논문은 환기가 필요한 환경에서의 소음 감쇠를 위한 대안으로 떠오르고 있는 음향 메타물질인 환기형 음향 공명기(VAR, Ventilated Acoustic Resonator)의 효율적인 역설계(inverse design)를 위한 새로운 방법을 제시합니다. 기존 VAR 설계는 비선형 음향 응답 특성으로 인해 제한적인 매개변수화된 설계 공간 내에서만 이루어졌으며, 수치 시뮬레이션을 반복적으로 수행해야 하므로 많은 계산 시간과 자원이 소모되었습니다. 본 연구에서는 음향 응답을 암호화한 변형 오토인코더(AR-VAE, Acoustic Response-Encoded Variational Autoencoder)라는 새로운 변형 오토인코더 기반 생성 설계 모델을 제안하여 비매개변수화된 설계에서도 효율적이고 정확한 VAR 역설계를 가능하게 합니다.



### A New Pipeline For Generating Instruction Dataset via RAG and Self Fine-Tuning (https://arxiv.org/abs/2408.05911)
Comments:
          5 pages, SCA 2024: The 7th IEEE International Workshop on Smart Computing & Applications

- **What's New**: 본 연구에서는 특정 도메인에 맞는 고품질 지침 데이터셋을 생성하는 파이프라인을 제안합니다. 이 파이프라인은 특정 도메인의 문서 컬렉션을 사용하여 LLM과 Retrieval-Augmented Generation(RAG) 프레임워크를 활용하여 지침을 생성합니다. 이를 통해 기존 수동 방식이나 웹 스크래핑으로 인한 오류 가능성을 제거하고 도메인 특화 모델을 효과적으로 구축할 수 있습니다.



### Weakly Supervised Video Anomaly Detection and Localization with Spatio-Temporal Prompts (https://arxiv.org/abs/2408.05905)
Comments:
          Accepted by ACMMM2024

- **What's New**: This paper presents a novel method called STPrompt for weakly supervised video anomaly detection and localization (WSVADL). STPrompt leverages pre-trained vision-language models (VLMs) to learn spatio-temporal prompt embeddings for identifying specific local regions of anomalies, enabling accurate video anomaly detection while mitigating the influence of background information.



### Quantum Gradient Class Activation Map for Model Interpretability (https://arxiv.org/abs/2408.05899)
Comments:
          Submitted to IEEE SiPS 2024

- **What's New**: 이 연구는 양자 기계 학습 (QML) 모델의 투명성을 높이기 위해 변분 양자 회로 (VQC)를 활용한 활성화 맵핑 기법을 제안합니다. 이는 양자 기울기 클래스 활성화 맵 (QGrad-CAM)이라는 하이브리드 양자-클래식 컴퓨팅 프레임워크입니다. 이 프레임워크는 양자와 클래식 컴퓨팅의 장점을 활용하여 특징 맵의 중요성을 명시적으로 계산할 수 있게 해줍니다.



### Integrative Approaches in Cybersecurity and AI (https://arxiv.org/abs/2408.05888)
- **What's New**: 이 논문은 사이버 보안, 인공지능(AI), 데이터 관리의 융합이 현대 기술 생태계의 복잡성과 상호 의존성 증가로 인해 연구의 중요한 분야로 부상했다는 점을 다룹니다. 이 연구는 AI 기법을 활용하여 사이버 보안 프레임워크를 강화하고 데이터 관리 관행을 최적화하는 통합적 접근 방식을 포괄적으로 검토하고 분석합니다. 이러한 분야 간 시너지를 탐구함으로써 조직이 데이터를 보호, 분석 및 활용하는 방식에 혁명을 일으킬 잠재력을 가진 주요 트렌드, 과제 및 미래 방향을 파악합니다. 이 연구 결과는 AI 기반 자동화, 실시간 위협 탐지 및 고급 데이터 분석을 통합하여 더욱 탄력적이고 적응력이 뛰어난 보안 아키텍처를 구축할 필요성을 강조합니다.



### LLM-Based Robust Product Classification in Commerce and Complianc (https://arxiv.org/abs/2408.05874)
Comments:
          11 pages

- **What's New**: 이 연구는 실제 세계 제품 분류의 어려움, 특히 제품 설명의 불완전성과 축약에 초점을 맞추어 진행되었습니다. 제품 분류의 정확성은 국제 무역에서 매우 중요하며, 잘못된 분류는 세금 및 관세 책임으로 이어질 수 있습니다. 이 연구는 GPT-4와 같은 강력한 대규모 언어 모델(LLM)을 사용하여 실제 세계 데이터 불완전성을 모방하는 데이터 왜곡(perturbation)을 만들어냅니다. 이러한 왜곡된 데이터는 LLM 기반 제품 분류 모델을 훈련하는 데 사용되어, 깨끗한 데이터에서만 훈련된 기존 감독 학습 모델보다 더 강력한 성능을 보여줍니다.



### Real-Time Drowsiness Detection Using Eye Aspect Ratio and Facial Landmark Detection (https://arxiv.org/abs/2408.05836)
- **What's New**: 이 논문은 실시간 졸음 감지 시스템을 소개합니다. 이 시스템은 눈의 개방 비율(EAR, Eye Aspect Ratio) 및 얼굴 랜드마크 검출 기술을 사용하여 졸음을 감지합니다.



### Divide-and-Conquer Predictive Coding: a structured Bayesian inference algorithm (https://arxiv.org/abs/2408.05834)
Comments:
          22 pages, 5 figures, submitted to Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이 논문은 구조화된 생성 모델을 위한 새로운 예측 코딩 알고리즘인 "분할 및 정복 예측 코딩(Divide-and-Conquer Predictive Coding, DCPC)"를 소개합니다. DCPC는 기존 예측 코딩 알고리즘과 달리 생성 모델의 상관 구조를 존중하고, 생물학적 타당성을 희생하지 않고 모델 매개변수에 대한 최대 우도 업데이트를 수행하는 것이 특징입니다.



### Robust Domain Generalization for Multi-modal Object Recognition (https://arxiv.org/abs/2408.05831)
Comments:
          6 pages, 2 figures. This is a preprint version of the article. The final version will be published in the proceedings of the IEEE conference

- **What's New**: 이 논문은 비전-언어 사전 훈련(vision-language pre-training) 모델인 CLIPood의 제한 사항을 해결하여 도메인 일반화(domain generalization) 성능을 향상시키는 새로운 방법을 제안합니다. CLIPood는 다양한 도메인에서 개념 인식(concept recognition)을 강화하는 비전-언어 쌍(visual-language pairs)의 감독을 활용하지만, 손실 함수(loss function) 활용, 백본(backbone)의 일반성 및 클래스 인식 시각 융합(class-aware visual fusion) 측면에서 제한점을 가지고 있습니다. 이 논문에서는 실제 손실(actual loss)을 추론하고, 더 큰 비전-언어 백본으로 평가를 확장하며, 클래스 인식 시각 융합을 강화하는 믹스업 손실(Mixup-CLIPood)을 소개합니다. 



### A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals (https://arxiv.org/abs/2408.05804)
Comments:
          Code and videos: this https URL

- **What's New**: 본 논문은 단일 목표 상태만 제공되는 상황에서도 RL 에이전트가 목표에 도달하기 위한 다양한 기술을 성공적인 시도 이전에 학습하는 것을 보여줍니다. 이러한 기술들은 보상 함수, 시연 데이터, 수동으로 지정된 거리 측정 없이도 스스로 발달합니다. 이 방법은 기존 방법을 간단하게 수정한 것으로, 밀도 추정, 앙상블, 추가 하이퍼파라미터 없이 구현됩니다.



### Time Makes Space: Emergence of Place Fields in Networks Encoding Temporally Continuous Sensory Experiences (https://arxiv.org/abs/2408.05798)
- **What's New**: 이 연구는 시뮬레이션된 공간을 탐색하는 에이전트가 부분적이고 노이지가 있는 감각 정보를 받는 상황에서 시냅스 가소성을 통해 공간 정보가 자연스럽게 형성되는 것을 보여줍니다. 이러한 연구는 해마 CA3 영역의 재귀적 네트워크가 부분적인 단서로부터 에피소드 기억을 회상하는 데 어떻게 기여하는지, 그리고 장소 세포의 형성과 기능에 어떤 역할을 하는지에 대한 새로운 통찰력을 제공합니다. 특히, 해마 CA3를 재귀적 오토인코더로 모델링하여 공간 탐색 과정에서 경험을 재구성하고 기억을 회상하는 과정을 모사했으며, 이를 통해 장소 세포와 유사한 특징을 가진 신경 활동 패턴이 자연스럽게 나타나는 것을 관찰했습니다.



### A Meta-Engine Framework for Interleaved Task and Motion Planning using Topological Refinements (https://arxiv.org/abs/2408.05795)
Comments:
          To appear in ECAI 2024

- **What's New**: 이 논문은 로봇의 자율성을 향상시키는 데 중요한 역할을 하는 작업 및 모션 계획(TAMP)(Task And Motion Planning) 문제를 위한 일반적이고 오픈소스 프레임워크를 제안합니다. 특히, 이 프레임워크는 이동 에이전트와 다중 작업 상태 종속 장애물(task-state-dependent obstacles)이 포함된 TAMP 문제를 해결하기 위한 혁신적인 메타 기술을 도입합니다. 이 메타 기술은 기존의 작업 계획자와 모션 계획자를 활용하면서, 모션 계획자의 검색 공간에 대한 기하학적 분석을 통해 작업 계획자의 탐색을 가지치기하여 효율성을 높입니다.



### Continual Learning of Nonlinear Independent Representations (https://arxiv.org/abs/2408.05788)
Comments:
          9 pages, 5 Figures

- **What's New**: 이 논문은 **연속적 인과적 표현 학습(CCRL)** 이라는 새로운 개념을 제시하여 순차적으로 도착하는 분포에서 의미 있는 (식별 가능한) 표현을 학습할 수 있는 모델을 구현합니다. 특히 비선형 독립 성분 분석 (ICA) 프레임워크에 초점을 맞춰, 순차적으로 도착하는 분포에서 인과적 표현을 학습하는 새로운 방법을 제시합니다. 이 방법은 이전 분포의 정보를 유지하면서 새로 도착하는 분포의 정보를 활용하여 학습된 표현을 개선하는 방식으로 작동합니다. 이는 인간의 학습 메커니즘과 유사합니다.



### CURLing the Dream: Contrastive Representations for World Modeling in Reinforcement Learning (https://arxiv.org/abs/2408.05781)
Comments:
          Paper accepted for 24th International Conference on Control, Automation and Systems (ICCAS)

- **What's New**: Curled-Dreamer, a 새로운 강화 학습 알고리즘이 DreamerV3 프레임워크에 contrastive learning (대조 학습)을 통합하여 시각적 강화 학습 작업에서 성능을 향상시킵니다.



### Seg-CycleGAN : SAR-to-optical image translation guided by a downstream task (https://arxiv.org/abs/2408.05777)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문은 선박 표적의 정확한 번역을 향상시키기 위해 사전 훈련된 의미론적 분할 모델에서 얻은 의미 정보를 활용하여 Seg-CycleGAN이라는 GAN 기반 SAR-to-optical 이미지 번역 방법을 제안합니다.



### An analysis of HOI: using a training-free method with multimodal visual foundation models when only the test set is available, without the training s (https://arxiv.org/abs/2408.05772)
- **What's New**: 이 연구는 훈련 데이터셋 없이 테스트 데이터셋만 사용하여, 멀티모달 비주얼 기반 모델(Multimodal Visual Foundation Model)을 사용한 훈련 없는(Training-free) 방식으로 Human-Object Interaction (HOI)를 수행합니다.

- **Technical Details**:  세 가지 실험 설정을 사용합니다: (1) 기본 진실(ground truth)에서 ⟨human, object⟩ 쌍을 멀티모달 비주얼 기반 모델에 입력하여 다양한 동사를 포함한 텍스트 프롬프트와 비교하여 동사 확률 분포를 얻습니다. (2) 기본 진실의 쌍의 특징을 혼합하여 'human'은 모든 기본 진실의 human 바운딩 박스를 포함하고 'object'는 모든 바운딩 박스를 포함합니다. 이것들은 쿼리 모듈에 입력되어 첫 번째 설정과 유사한 동사 결과를 생성합니다. (3) 쌍이 없는 grounding DINO에서 추출된 바운딩 박스를 사용하여 두 번째 설정과 유사한 방식으로 동사 결과를 얻습니다. 이 연구는 희소한/비 희소한, 본/보지 못한 조합/객체/동사와 같은 차이점이 관련이 없음을 보여줍니다.

- **Performance Highlights**: 희소한 클래스는 human과 object의 임의 조합에 무감각한 반면, 비 희소한 클래스는 민감합니다.  RF-UC(Rare First Unseen Combinations) 설정에서 꼬리 HOI(희소한 클래스) 범주는 보지 못한 클래스로 지정되며, 보지 못한 클래스(희소한 클래스)는 임의 조합에 무감각한 반면, 본 클래스(비 희소한)는 민감합니다.  NF-UC(Non-rare First Unseen Combinations) 설정에서는 머리 HOI 범주(비 희소한)를 보지 못한 클래스로 지정하고, 보지 못한 클래스(비 희소한)는 임의 조합에 민감하고, 본 클래스(희소한)는 민감하지 않습니다. 보지 못한/본 객체 또는 동사가 포함된 실험에서 임의 조합에 대한 민감도는 보지 못한 및 본 분류 간에 일관성을 유지합니다.  이 연구는 멀티모달 비주얼 기반 모델의 제로 샷/퓨 샷 기능이 아직 완전히 구현되지 않았다는 것을 보여줍니다.



### Reference-free Hallucination Detection for Large Vision-Language Models (https://arxiv.org/abs/2408.05767)
- **What's New**: 이 연구는 참조 없는 (reference-free) 방법이 LVLMs (Large Vision-Language Models)의 환각 (hallucination)을 효과적으로 감지할 수 있는지 조사합니다. 특히, 불확실성 기반 (uncertainty-based), 일관성 기반 (consistency-based), 그리고 감독된 불확실성 정량화 (Supervised Uncertainty Quantification, SUQ) 방법이라는 세 가지 유형의 기술을 사용하여 연구를 수행합니다.

- **Technical Details**: 연구에서는 네 가지 대표적인 LVLMs을 사용하여 두 가지 다른 작업 (Yes-and-No, Open-ended)에 대해 광범위한 실험을 수행했습니다. 불확실성 기반 방법에는 AvgProb, AvgEnt, MaxProb, MaxEnt의 네 가지 지표가 사용되었습니다. 일관성 기반 방법에는 BERTScore, 질문 답변 (Question Answering, QA), Unigram, 자연어 추론 (Natural Language Inference, NLI)의 네 가지 변형이 사용되었습니다. SUQ 방법은 모델의 내부 상태를 분석하여 진술의 신뢰성을 예측하는 분류기를 훈련합니다.

- **Performance Highlights**: 실험 결과는 SUQ 방법이 다른 접근 방식보다 환각 감지 성능이 뛰어나다는 것을 보여줍니다. 특히 SUQ 방법은 문장 및 구절 수준에서 모두 우수한 성능을 보였습니다. 일관성 기반 방법은 불확실성 기반 방법보다 뛰어났지만 SUQ 방법보다는 낮은 성능을 보였습니다.

- **Contributions**: 이 논문은 다음과 같은 주요 기여를 합니다: - 다양한 참조 없는 방법의 환각 감지 성능을 포괄적으로 측정합니다. - 감독된 불확실성 정량화 (SUQ) 방법이 다양한 설정에서 최상의 성능을 보임을 보여줍니다. - LLaVA-v1.5-7b를 사용하여 수동으로 주석이 달린 문장 수준의 데이터셋인 Image-Hallucination Annotation Dataset (IHAD)를 제공합니다.



### VQ-CTAP: Cross-Modal Fine-Grained Sequence Representation Learning for Speech Processing (https://arxiv.org/abs/2408.05758)
- **What's New**: VQ-CTAP (Vector Quantized Contrastive Token-Acoustic Pre-training) 모델을 제안하여 텍스트와 음성을 프레임 단위로 연결하는 방법을 제시합니다. 이 모델은 크로스 모달 정렬된 시퀀스 트랜스코더를 사용하여 텍스트와 음성을 공동 다중 모달 공간으로 가져옵니다.



### MTSCI: A Conditional Diffusion Model for Multivariate Time Series Consistent Imputation (https://arxiv.org/abs/2408.05740)
Comments:
          10 pages, 5 figures, accepted by CIKM2024

- **What's New**: This paper introduces MTSCI, a novel conditional diffusion model specifically designed for multivariate time series imputation. MTSCI addresses a key limitation of existing methods: the lack of consideration for imputation consistency. MTSCI focuses on ensuring both intra-consistency (consistency between observed and imputed values) and inter-consistency (consistency between adjacent windows after imputation).

- **Technical Details**: MTSCI utilizes two key techniques:

1. **Contrastive Complementary Mask:** During the forward noising process, this technique generates dual views of the data, employing a contrastive loss function to maintain consistency between observed and imputed values.

2. **Mixup Mechanism:** This mechanism incorporates conditional information from adjacent windows during the denoising process, ensuring temporal consistency between imputed samples.

- **Performance Highlights**: Extensive experiments on multiple real-world datasets demonstrate that MTSCI achieves state-of-the-art performance in multivariate time series imputation, outperforming existing methods in terms of MAE, RMSE, and MAPE. MTSCI shows an average improvement of 17.88% in MAE, 15.09% in RMSE, and 13.64% in MAPE compared to baseline methods.



### Deformable Image Registration with Multi-scale Feature Fusion from Shared Encoder, Auxiliary and Pyramid Decoders (https://arxiv.org/abs/2408.05717)
- **What's New**: 본 논문은 비지도 이미지 레지스트레이션(unsupervised image registration)을 위한 새로운 변형 가능한 피라미드 네트워크(deformable convolutional pyramid network)를 제안합니다. 이 네트워크는 이미지 쌍을 위한 추가적인 공유 보조 디코더(shared auxiliary decoder)를 추가하여 기존 피라미드 네트워크를 개선합니다. 이 디코더는 레지스트레이션 작업에 사용할 수 있도록 이미지 쌍의 다중 스케일 고수준 기능 정보(multi-scale high-level feature information)를 제공합니다. 또한, 레지스트레이션 과정에서 다중 스케일 기능 융합 블록(multi-scale feature fusion block, MSFB)을 설계하여 글로벌 및 로컬 컨텍스트에서 레지스트레이션 작업에 가장 유용한 기능을 추출합니다.



### DeepAir: A Multi-Agent Deep Reinforcement Learning Based Scheme for an Unknown User Location Problem (https://arxiv.org/abs/2408.05712)
Comments:
          12 pages, 8 figures, 5 tables

- **What's New**: 이 논문은 인프라가 없는 환경에서 사용자 위치가 알려지지 않은 상황에서 무인 항공기(UAV)를 사용하여 사용자를 감지하고, 위치를 파악하고, 리소스를 할당하며, 작업 오프로딩을 위한 멀티 액세스 에지 컴퓨팅(MEC)을 수행하는 새로운 딥 강화 학습(DRL) 기반 시스템인 DeepAir를 제안합니다. DeepAir는 검출 UAV와 서비스 UAV의 두 가지 유형의 UAV를 사용합니다. 검출 UAV는 DRL 에이전트로서 감지, 위치 확인, 리소스 할당을 수행하는 반면, 서비스 UAV는 MEC 기능을 제공합니다. DeepAir는 기존 방법에 비해 환경에 적은 수의 검출 UAV를 배포하여 높은 작업 성공률을 달성합니다. DeepAir는 사용자의 품질(QoS) 요구 사항을 충족하고 허용 가능한 지연 시간을 넘지 않도록 합니다. 특히, DeepAir는  (1) 감지, (2) 위치 확인, (3) 리소스 할당, (4) UAV 지원 MEC의 네 가지 범주를 모두 고려합니다.



### TC-KANRecon: High-Quality and Accelerated MRI Reconstruction via Adaptive KAN Mechanisms and Intelligent Feature Scaling (https://arxiv.org/abs/2408.05705)
Comments:
          10 pages, 3 figures

- **What's New**: TC-KANRecon: This paper presents a novel conditional guided diffusion model, TC-KANRecon, for accelerating MRI reconstruction while preserving image quality. It incorporates two key modules: the Multi-Free U-KAN (MF-UKAN) module and a dynamic clipping strategy, aiming to address limitations in existing deep learning-based MRI reconstruction techniques.



### A Novel Momentum-Based Deep Learning Techniques for Medical Image Classification and Segmentation (https://arxiv.org/abs/2408.05692)
Comments:
          8 pages

- **What's New**: 이 연구는 의료 영상 분석에서 학습 역학을 향상시키기 위해 잔여 블록 (Residual Block) 내에 모멘텀을 통합하는 새로운 기술을 소개합니다. 이 연구는 컴퓨터 지원 진단 및 개입 계획을 위해 CT 및 MRI 스캔에서 다양한 장기를 분할하고 질병을 분류하는 데 중점을 둡니다. 제안된 모멘텀 기반 접근 방식은 폐, 간 및 결장 데이터를 분할하고 복부 골반 CT 및 MRI 스캔을 분류하는 두 가지 별개의 작업에서 뛰어난 성능을 보여 주었습니다. 특히 폐 분할 데이터 세트에서 제안된 방법은 Dice Score에서 5.72% 증가, mIoU (Mean Intersection over Union)에서 5.04% 향상, 재현율에서 8.02% 향상 및 정밀도에서 4.42% 향상을 포함하여 TransNetR 모델보다 상당한 개선을 보였습니다. 이러한 결과는 모멘텀 통합이 의료 영상 분야에서 획기적인 발전을 나타내는 분할 및 분류 작업 모두에서 최첨단 성능을 달성한다는 것을 시사합니다.



### SRTFD: Scalable Real-Time Fault Diagnosis through Online Continual Learning (https://arxiv.org/abs/2408.05681)
- **What's New**: 본 논문에서는 SRTFD라는 새로운 실시간 오류 진단 프레임워크를 제안합니다. 이 프레임워크는 산업 환경의 온라인 지속 학습 (OCL)을 향상시키기 위해 고안되었으며, 대량의 스트리밍 데이터, 불균형 데이터 및 제한된 레이블 데이터를 효율적으로 처리합니다.



### Efficient Federated Learning Using Dynamic Update and Adaptive Pruning with Momentum on Shared Server Data (https://arxiv.org/abs/2408.05678)
Comments:
          27 pages, to appear in TIST

- **What's New**: This paper introduces FedDUMAP, a novel federated learning (FL) framework that leverages both shared insensitive data on the server and distributed sensitive data on edge devices for efficient global model training. It addresses common challenges in FL, including low training efficiency and limited computational resources.



### StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Mod (https://arxiv.org/abs/2408.05669)
- **What's New**: 본 논문은 **AI-Generated Content Stealth (AIGC-S)** 라는 새로운 과제를 제시하며, 이는 인공지능이 생성한 이미지가 탐지 기술과 사람의 눈 모두를 속일 수 있도록 만드는 것을 목표로 합니다. 기존의 **adversarial attack** 방법들은 시각적으로 눈에 띄는 노이즈를 발생시키거나, **transferability**가 떨어지거나, **spectral difference**를 해결하지 못하는 단점을 가지고 있습니다. 이를 해결하기 위해 본 논문은 **StealthDiffusion**이라는 새로운 프레임워크를 제안합니다.



### Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites (https://arxiv.org/abs/2408.05667)
- **What's New**: PhishLang, an open-source, lightweight Large Language Model (LLM) for phishing website detection, is introduced. It leverages contextual analysis of website source code to identify phishing patterns.

- **Technical Details**: PhishLang utilizes LLM's advanced language processing capabilities to learn granular features characteristic of phishing attacks. It employs a "sliding window" technique for efficient training and operates with minimal data preprocessing, making it faster and less resource-intensive compared to traditional deep learning models.

- **Performance Highlights**: PhishLang successfully identified approximately 26K phishing URLs, many undetected by popular anti-phishing blocklists, over a 3.5-month testing period. It's also robust against adversarial attacks and integrates with GPT-3.5 Turbo for "explainable blocklisting", providing users with contextual information on why a website was flagged as phishing.

- **Open-source Availability**: PhishLang is open-sourced and available as a Chromium-based browser extension and a URL scanning website, providing users with real-time protection against phishing threats.



### Eigen Attention: Attention in Low-Rank Space for KV Cache Compression (https://arxiv.org/abs/2408.05646)
Comments:
          12 page, 6 figures, 6 tables

- **What's New**: 본 논문은 Eigen Attention이라는 새로운 메커니즘을 소개하며, 이는 저랭크 근사(low-rank approximation)를 통해 KV 캐시를 압축하여 대규모 언어 모델(LLM)의 효율적인 서비스를 제공합니다.

- **Technical Details**: Eigen Attention은 LLM에서의 어텐션 입력(키, 쿼리, 값)이 몇몇 주요 기저 벡터(principal basis vectors) 또는 고유 벡터(eigenvectors)를 이용하여 합리적으로 근사될 수 있다는 관찰에 기반합니다. 이러한 주요 벡터로 키와 값을 선형 결합으로 표현함으로써, 차원이 축소되어 KV 캐시의 메모리 공간이 감소합니다. 특히, Eigen Attention은 사전 학습된 모델에 적용할 수 있는 사후 학습 기법(post-training technique)이며, KV 캐시 압축을 위한 기존 기법과 함께 사용 가능합니다.

- **Performance Highlights**: 다양한 모델과 언어 작업에 대한 광범위한 실험 결과, Eigen Attention은 최대 40%의 KV 캐시 크기 감소와 최대 60%의 어텐션 연산 지연 시간 감소를 달성하며, 성능 저하는 미미했습니다. Eigen Attention은 KV 캐시 메모리 오버헤드를 줄이는 주요 목표 외에도 어텐션 블록의 계산 효율성을 높이는 데에도 기여합니다.



### Federated Smoothing Proximal Gradient for Quantile Regression with Non-Convex Penalties (https://arxiv.org/abs/2408.05640)
- **What's New**: 이 논문은 분산 센서에서 생성된 드문 데이터의 분석을 위한 **연합 양분위수 회귀** (Federated Quantile Regression) 알고리즘을 제시합니다. 특히, **비볼록 스파스 페널티** (Nonconvex Sparse Penalty)를 사용하는 연합 학습 환경에서 **MCP** (Minimax Concave Penalty)와 **SCAD** (Smoothly Clipped Absolute Deviation)과 같은 페널티를 적용할 수 있는 새로운 **FSPG** (Federated Smoothing Proximal Gradient) 알고리즘이 소개됩니다. FSPG는 부드러운 근사화를 통해 비볼록성과 비매끄러움 문제를 해결하며 효율적인 수렴을 보장합니다.



### Enhancing Computational Efficiency in Intensive Domains via Redundant Residue Number Systems (https://arxiv.org/abs/2408.05639)
Comments:
          This paper has been accepted by the 21st International SoC Conference (ISOCC), 2024, 2 pages

- **What's New**: 이 논문은 **RNS**(Residue Number System)와 **Redundant Number System**(잉여수 체계)을 결합한 새로운 **R-RNS**(Redundant-RNS) 시스템을 제안하고, 이를 통해 연산 속도를 향상시키고 회로 구현을 최적화하는 방법을 연구합니다. 특히, **SD-RNS**(Signed-Digit Redundant Residue Number System)를 중점적으로 살펴보고, CIFAR-10 데이터셋을 사용한 **DNN**(Deep Neural Network)을 통해 실질적인 성능을 평가합니다.



### PRTGaussian: Efficient Relighting Using 3D Gaussians with Precomputed Radiance Transfer (https://arxiv.org/abs/2408.05631)
- **What's New**: PRTGaussian은 3D 가우시안과 사전 계산된 복사 전달(Precomputed Radiance Transfer, PRT)을 결합하여 실시간 재조명 가능한 새로운 뷰 합성 방법을 제시합니다. 이 방법은 멀티뷰 OLAT 데이터에 재조명 가능한 가우시안을 적합시켜 실시간, 자유 시점 재조명을 가능하게 합니다. 고차 구면 조화(spherical harmonics)를 기반으로 복사 전달을 추정하여 자세한 재조명 효과를 포착하는 동시에 계산 효율성을 유지합니다.



### Quantum-secure multiparty deep learning (https://arxiv.org/abs/2408.05629)
- **What's New**: 이 논문은 광자(photon)의 양자적 특성을 활용하여 기존 통신 장비만으로 정보 이론적으로 안전한 다자간 컴퓨팅(multiparty computation)을 수행하는 선형 대수 엔진을 소개합니다. 이 엔진은 딥 러닝에 적용되어, 클라이언트 데이터와 DNN 가중치에 대한 정보 누출(information leakage)을 최소화하면서 높은 정확도를 달성합니다.



### Forecasting Day-Ahead Electricity Prices in the Integrated Single Electricity Market: Addressing Volatility with Comparative Machine Learning Methods (https://arxiv.org/abs/2408.05628)
- **What's New**: This paper presents a comprehensive study of electricity price forecasting methods for the Irish Integrated Single Electricity Market (I-SEM), focusing on recent volatile periods. It investigates the performance of various models, including machine learning and neural networks, and analyzes the impact of different training data lengths. The study finds that the EU Natural Gas price is a more significant predictor of electricity prices in Ireland than the Henry Hub Natural Gas price. It also highlights the increasing importance of natural gas input fuel costs and the impact of renewable energy sources on electricity prices.



### UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling (https://arxiv.org/abs/2408.05618)
- **What's New**: UrFound, a universal retinal foundation model that can process both Color Fundus Photography (CFP) and Optical Coherence Tomography (OCT) images while incorporating domain knowledge from expert annotations, is introduced. This model addresses the limitations of existing retinal foundation models, which are typically restricted to a single modality and may not fully utilize expert annotations.



### Residual-INR: Communication Efficient On-Device Learning Using Implicit Neural Representation (https://arxiv.org/abs/2408.05617)
Comments:
          This paper has been accepted by ICCAD 2024

- **What's New**: Residual-INR는 엣지 디바이스 간 통신 효율성을 높이기 위해 잉크(Implicit Neural Representation)를 활용한 새로운 엣지 컴퓨팅 기반 학습 프레임워크입니다. 기존 엣지 디바이스 간 JPEG 이미지 전송 방식 대신, 잉크를 활용해 이미지를 압축하고, 엣지 디바이스에서 학습을 진행하는 방식을 제시합니다.

- **Technical Details**: Residual-INR은 이미지를 배경과 객체로 분리하여 각각 다른 크기의 잉크로 압축하는 기술입니다. 즉, 배경은 상대적으로 작은 잉크로 압축하고, 객체는 별도의 잉크를 통해 더 높은 품질로 압축하여 잉크 크기를 줄이면서도 객체의 정보는 유지하는 기술입니다. 이를 통해 데이터 전송량을 크게 줄이고, 엣지 디바이스에서의 연산량도 감소시킬 수 있습니다.

- **Performance Highlights**: Residual-INR은 기존 JPEG 압축 방식 대비 최대 12.1배 이미지 크기 감소, 데이터 전송량 최대 5.16배 감소, 학습 속도 최대 2.9배 향상을 달성했습니다. 또한, 객체 인식 정확도를 유지하면서도 잉크 크기 최적화를 통해 엣지 디바이스에서 CPU를 사용하지 않고 학습이 가능하도록 설계되었습니다.

- **Benefits**: Residual-INR은 엣지 컴퓨팅 환경에서 데이터 전송량을 줄이고, 학습 속도를 향상시키는 데 유용합니다. 특히, 이미지/영상 처리, 객체 인식, 자율 주행 등 데이터 전송량이 많은 분야에서 효과적인 기술입니다.

- **Limitations**: Residual-INR은 잉크 네트워크를 학습하는 데 시간이 오래 걸릴 수 있습니다. 또한, 잉크 네트워크의 크기가 커질수록 학습 및 추론에 더 많은 자원이 필요합니다.



### Representation Alignment from Human Feedback for Cross-Embodiment Reward Learning from Mixed-Quality Demonstrations (https://arxiv.org/abs/2408.05610)
Comments:
          First Two Authors Share Equal Contribution. 19 Pages, 4 Figures

- **What's New**: 이 연구는 여러 가지 구현(embodiment)에서 다양한 품질의 시연(demonstration)으로부터 보상 함수(reward function)를 학습하는 새로운 문제인 **교차 구현 보상 학습(cross-embodiment reward learning)**을 소개합니다. 이는 다양한 형태의 로봇이나 에이전트가 인간 시연 또는 다른 에이전트의 시연을 통해 학습하는 데 중요한 문제입니다. 기존 연구들은 최적에 가까운 시연을 필요로 했지만, 이 연구는 실제 상황에서 발생할 수 있는 다양한 품질의 시연으로부터 보상을 학습하는 데 초점을 맞춥니다. 특히, 인간의 피드백(feedback)을 활용하여 표현 학습(representation learning) 및 정렬(alignment)을 개선하는 다양한 기술들을 분석하여 교차 구현 학습을 향상시키는 것을 목표로 합니다.



### Mitigating Metropolitan Carbon Emissions with Dynamic Eco-driving at Sca (https://arxiv.org/abs/2408.05609)
Comments:
          In review

- **What's New**: 이 논문은 미국 주요 도시의 6,011개 교차로에서 100만 가지 교통 시나리오를 시뮬레이션하여 대규모 미래 시나리오를 통해 동적 에코 드라이빙의 영향을 평가합니다. 이 시스템은  **자율 주행 기술**을 통해 **신호등**에서 불필요한 **정지 및 출발**을 줄여 연료 소비와 탄소 배출량을 감소시킵니다. 이 연구는 이전 연구에서 다루지 못했던  **대규모 시나리오 모델링**,  **다중 작업 심층 강화 학습** 및  **네트워크 분해 전략**을 적용합니다. 



### Exploring Applications of State Space Models and Advanced Training Techniques in Sequential Recommendations: A Comparative Study on Efficiency and Performanc (https://arxiv.org/abs/2408.05606)
Comments:
          arXiv admin note: text overlap with arXiv:2403.07691 by other authors

- **What's New**: 이 논문은 시퀀스 추천에서의 효율성과 성능을 향상시키기 위한 새로운 접근 방식을 제시합니다. 특히 State Space Model (SSM)을 기반으로 한 Mamba 모델을 사용하여 추천 시스템의 속도를 높이고 대규모 언어 모델 (LLM)을 통합하여 추천 품질을 향상시킵니다. 또한, 학습 과정을 가속화하고 비용을 줄이기 위한 적응형 배치 및 단계 크기 알고리즘을 도입합니다.

- **Technical Details**: 이 연구는 다음과 같은 주요 기술적 세부 사항을 다룹니다:

- **State Space Model (SSM):** SSM은 시퀀스 모델링을 위한 최신 프레임워크로, 선형 상미분 방정식을 사용하여 시퀀스의 변화를 모델링합니다. SSM은 Transformer 모델에 비해 낮은 메모리 및 추론 비용으로 우수한 성능을 제공합니다.
- **Mamba:** Mamba는 SSM의 확장판으로, 데이터 의존적인 선택 메커니즘을 추가하여 관련 정보에 집중하고 노이즈를 제거합니다. Mamba는 GPU를 사용하여 효율적으로 계산할 수 있습니다.
- **Universal Stochastic Gradient Method (USGM):** USGM은 기계 학습의 기반인 확률적 최적화 분야에서 주목할 만한 발전입니다. USGM은 기존의 Stochastic Gradient Descent (SGD)보다 더 빠른 수렴 속도를 제공합니다.
- **Adaptive Batch and Step Size Algorithms:** 이러한 알고리즘은 학습 과정에서 배치 크기와 단계 크기를 동적으로 조정하여 학습 속도를 높이고 비용을 절감합니다.

- **Performance Highlights**: 이 연구는 다음과 같은 주요 성능 향상을 강조합니다:

- **더 빠른 추론 속도:** SSM 기반 Mamba 모델은 Transformer 모델에 비해 훨씬 빠른 추론 속도를 제공합니다.
- **향상된 추천 품질:** LLM을 사용한 Monolithic Preference Optimization (ORPO)는 추천 품질을 향상시킵니다.
- **더 빠른 학습 과정:** 적응형 배치 및 단계 크기 알고리즘은 학습 과정을 가속화합니다.
- **낮은 메모리 및 추론 비용:** SSM은 Transformer 모델에 비해 낮은 메모리 및 추론 비용을 요구합니다.



### Sequential Representation Learning via Static-Dynamic Conditional Disentanglemen (https://arxiv.org/abs/2408.05599)
Comments:
          Accepted at ECCV 2024

- **What's New**: 이 논문은 비디오에서 시간에 의존하지 않는 요소와 시간에 따라 변하는 요소를 분리하는 데 중점을 두고 순차 데이터에서 자기 지도 학습된 분리된 표현 학습을 탐구합니다. 저자는 정적/동적 변수 간의 인과 관계를 명시적으로 고려하여 이러한 요소 간의 일반적인 독립성 가정을 깨뜨리는 새로운 모델을 제안합니다. 이는 Normalizing Flows(정규화 흐름)를 추가하여 모델 표현력을 향상시킵니다. 저자는 이러한 요소에 대한 공식적인 정의를 제안합니다. 이 형식주의는 기본 요소가 식별 가능하도록 충분한 조건을 유도하고 새로운 모델 프레임워크에 직접적이고 효율적으로 통합할 수 있는 이론적으로 근거 있는 새로운 분리 제약 조건을 도입합니다. 실험 결과, 제안된 접근 방식은 장면의 동역학이 해당 콘텐츠의 영향을 받는 시나리오에서 이전의 복잡한 최첨단 기술보다 뛰어난 성능을 보여줍니다.



### Document-Level Event Extraction with Definition-Driven ICL (https://arxiv.org/abs/2408.05566)
- **What's New**: 본 논문에서는 문서 수준 이벤트 추출에서 LLM의 프롬프트 엔지니어링 문제를 해결하기 위한 새로운 최적화 전략인 "정의 기반 문서 수준 이벤트 추출(DDEE)"를 제안합니다. 이 전략은 프롬프트 길이를 조정하고 휴리스틱의 명확성을 강화하여 LLM의 이벤트 추출 성능을 향상시키고, 데이터 균형 기술을 사용하여 롱테일 효과 문제를 해결함으로써 모델의 이벤트 유형에 대한 일반화 능력을 강화했습니다. 동시에, LLM의 프롬프트 스타일 민감도에 맞춰 간결하고 포괄적인 프롬프트를 개선했고, 구조화된 휴리스틱 방법과 엄격한 제한 조건을 도입하여 이벤트 및 인수 역할 추출의 정확성을 향상시켰습니다. 이러한 전략은 문서 수준 이벤트 추출에서 LLM의 프롬프트 엔지니어링 문제를 해결할 뿐만 아니라 이벤트 추출 기술의 발전을 촉진하고 NLP 분야의 다른 작업에 대한 새로운 연구 관점을 제공합니다.



### Impacts of Darwinian Evolution on Pre-trained Deep Neural Networks (https://arxiv.org/abs/2408.05563)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 딥 러닝 모델 훈련을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 진화론적 이론을 기반으로 하여, 백프로퍼게이션(Back-Propagation, BP)으로 훈련된 딥 신경망을 초기 종(primordial ancestor)으로 간주하고, 차등 진화(Differential Evolution)를 통해 이들을 진화시킵니다. 이를 통해 기존 BP 기반 훈련 방식의 단점인 과적합(overfitting) 문제를 완화하고, 훈련 시간을 크게 단축시키는 효과를 얻을 수 있습니다.

- **Technical Details**: 본 연구는 딥 신경망을 생물학적 종(species)에 비유하고, 이들의 훈련 과정을 진화론적 관점에서 해석합니다.  BP로 훈련된 딥 신경망은 초기 종으로 간주되고, 차등 진화를 통해 이들이 진화하면서 데이터셋, 환경, 모델과 살아있는 생물 종 사이의 상관관계를 조사합니다. 본 논문에서는 차등 진화(Differential Evolution)를 기반으로 하는 진화 알고리즘을 사용합니다.  차등 진화는 개체군 내에서의 차이를 이용하여 새로운 개체를 생성하는 방법으로, 기존 BP 방법과 달리 기울기 정보를 직접 사용하지 않고도 최적화를 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 과적합을 줄이고, 기존 BP 방법에 비해 훈련 시간을 10배 이상 단축시키는 효과를 보였습니다. 또한, 제안된 프레임워크는 딥 신경망 및 대규모 데이터셋에서도 효과적인 성능을 보였습니다. 



### Evolutionary Neural Architecture Search for 3D Point Cloud Analysis (https://arxiv.org/abs/2408.05556)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 3차원 포인트 클라우드 데이터 분석에 특화된 효율적인 신경망 아키텍처를 자동으로 설계하는 새로운 진화형 신경망 아키텍처 검색(NAS) 프레임워크, SHSADE-PIDS를 제안합니다. SHSADE-PIDS는 이산 신경망 아키텍처를 연속 공간으로 인코딩하고 연속 공간에서 검색을 수행하여 효율적인 포인트 클라우드 신경망 아키텍처를 찾습니다. 이 방법은 기존 NAS 기법보다 높은 정확도로 더 효율적인 아키텍처를 발견합니다.



### Multi-layer Sequence Labeling-based Joint Biomedical Event Extraction (https://arxiv.org/abs/2408.05545)
Comments:
          13 pages, 3 figures, accepted by NLPCC2024

- **What's New**: MLSL, a novel method based on multi-layer sequence labeling for joint biomedical event extraction, is proposed in this paper. It utilizes the information of candidate trigger words explicitly for a simplified workflow with no prior knowledge or complex structures.



### CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM (https://arxiv.org/abs/2408.05526)
- **What's New**: This research introduces CryoBench, a suite of datasets and metrics designed to benchmark heterogeneous reconstruction in cryo-EM, a powerful technique for determining high-resolution 3D biomolecular structures.

- **Technical Details**: CryoBench comprises five datasets representing various sources of heterogeneity: conformational heterogeneity from antibody complexes, molecular dynamics simulations, and compositional heterogeneity from ribosome assembly states and common cellular complexes. CryoBench leverages synthetic data generation to provide ground truth structures and imaging parameters for quantitative evaluation.

- **Performance Highlights**: The research performs a comprehensive analysis of existing heterogeneous reconstruction tools, both neural and non-neural methods, to assess their sensitivity to noise. It proposes novel metrics for quantitative comparison, providing a foundational resource for analyzing existing methods and advancing algorithmic development in both the cryo-EM and machine learning communities.



### Disentangled Noisy Correspondence Learning (https://arxiv.org/abs/2408.05503)
- **What's New**: DisNCL (Disentanglement in Noisy Correspondence Learning), 새로운 정보 이론적 프레임워크를 소개하여, 훈련 데이터의 노이즈 대응 학습에서 특징 해 disentanglement (분리) 효율성을 개선합니다. 기존의 방법들과 달리, DisNCL은 모달 간 불변 정보 (MII)와 모달 고유 정보 (MEI)를 분리하는 방식으로, 노이즈가 포함된 데이터에서도 정확한 유사도 예측을 가능하게 합니다. 또한, 부드러운 매칭 타겟을 도입하여, 다중 모달 데이터의 다 대 다 관계를 모델링하여, 노이즈에 강인하고 정확한 모달 간 정렬을 수행합니다.



### PointNCBW: Towards Dataset Ownership Verification for Point Clouds via Negative Clean-label Backdoor Watermark (https://arxiv.org/abs/2408.05500)
Comments:
          12 pages

- **What's New**: 본 논문은 포인트 클라우드 데이터셋의 저작권 보호를 위해 새로운 ‘스케일러블 클린-라벨 백도어 기반 데이터셋 워터마크(Scalable Clean-Label Backdoor-based Dataset Watermark)’를 제안합니다. 기존의 방법들은 많은 클래스를 가진 대규모 데이터셋에서 효과가 떨어지는 한계를 가지고 있었지만, 본 논문에서 제시된 방법은 모든 클래스에서 샘플을 워터마킹할 수 있어 대규모 데이터셋에도 효과적으로 적용될 수 있습니다.

- **Technical Details**: 본 논문에서 제시된 방법은 ‘네거티브 트리거 효과(negative trigger effects)’를 활용합니다. 먼저 비-타겟 클래스의 포인트 클라우드를 타겟 클래스의 포인트 클라우드와 유사하도록 변형시키고, 변형된 포인트 클라우드에 트리거 패턴을 삽입합니다. 이렇게 생성된 워터마크된 샘플은 타겟 클래스의 레이블과는 다르지만, 특징 공간에서는 타겟 클래스의 샘플과 유사합니다. 훈련된 DNN은 삽입된 트리거 패턴을 타겟 레이블을 예측하지 못하게 하는 신호로 인식하게 됩니다. 즉, 트리거 패턴이 나타나면 타겟 클래스에 대한 예측 확신도가 감소하게 됩니다.

- **Performance Highlights**: 본 논문에서 제시된 방법의 효과성과 잠재적인 제거 방법에 대한 저항성을 실험적으로 검증했습니다. 실험 결과, 제안된 PointNCBW (Negative Clean-Label Backdoor Watermark for Point Clouds)는 기존의 방법에 비해 뛰어난 성능을 보여주었으며, 대규모 데이터셋에 대한 확장성을 갖추고 있습니다. 또한, 잠재적인 공격에 대해 저항성을 가지는 것으로 확인되었습니다.



### LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Sca (https://arxiv.org/abs/2408.05499)
Comments:
          15 pages, 11 figures

- **What's New**: LLMServingSim, a novel simulation tool for LLM inference serving systems, is introduced. It tackles the limitations of existing simulators, offering accurate and efficient modeling of LLM inference serving systems.



### Artworks Reimagined: Exploring Human-AI Co-Creation through Body Prompting (https://arxiv.org/abs/2408.05476)
Comments:
          16 pages, 5 figures, 2 tables

- **What's New**: This paper introduces 'body prompting' as a novel input modality for image generation, replacing traditional text prompts with human poses. This approach aims to reintroduce physicality and expressiveness into the creative process of image generation, making it more engaging and inclusive.



### Investigating Instruction Tuning Large Language Models on Graphs (https://arxiv.org/abs/2408.05457)
Comments:
          COLM 2024

- **What's New**: 이 연구는 최신의 대규모 언어 모델(LLM)이 그래프 관련 작업에 적용될 수 있는 가능성을 살펴봅니다. 특히 LLM이 실제 그래프와 상호 작용하고 다양한 그래프 작업에서 일반화될 수 있는 방법에 대한 실증적인 통찰력을 제공하는 것을 목표로 합니다. 이를 위해 연구팀은 79개의 다양한 그래프 관련 작업(학문적 및 전자 상거래 도메인)을 포함하는 지침 조정 데이터 세트를 구축했습니다. 또한 연구팀은 LLM이 복잡한 그래프 구조를 이해하는 데 도움이 되는 최적의 그래프 표현 방식을 조사했습니다. 연구 결과, JSON 형식의 그래프 표현 방식이 다양한 LLM과 그래프 유형에서 자연어 및 코드 형식보다 일관되게 더 나은 성능을 보이는 것으로 나타났습니다. 마지막으로, 연구팀은 지침 조정된 LLM의 일반화 능력에 영향을 미치는 주요 요인을 도메인 내 및 도메인 외부 그래프 작업에 대한 성능을 평가하여 분석했습니다.



### Mathematical Models of Computation in Superposition (https://arxiv.org/abs/2408.05451)
Comments:
          28 pages, 5 figures. Published at the ICML 2024 Mechanistic Interpretability (MI) Workshop

- **What's New**: 이 논문은 신경망에서 '계산적 중첩' (computational superposition) 의 개념을 도입하고 분석합니다. 기존의 '표현적 중첩' (representational superposition) 이 정보를 병목 지점 (bottleneck) 을 통해 전달할 때만 중첩을 사용하는 것과 달리, 본 연구에서는 중첩이 계산 작업을 효율적으로 수행하는 데 적극적으로 활용되는 상황을 다룹니다.



### EPAM-Net: An Efficient Pose-driven Attention-guided Multimodal Network for Video Action Recognition (https://arxiv.org/abs/2408.05421)
- **What's New**: 본 연구에서는 비디오에서 행동 인식을 위한 효율적인 자세 기반 주의 유도 다중 모달 네트워크(EPAM-Net)을 제시합니다. EPAM-Net은 RGB 및 자세 스트림 모두에 X3D 네트워크를 적용하여 RGB 비디오 및 해당 골격 시퀀스에서 공간-시간적 특징을 포착합니다. 골격 특징은 시각적 네트워크 스트림이 공간-시간 주의 블록을 사용하여 핵심 프레임 및 해당 두드러진 공간 영역에 집중하는 데 도움이 됩니다. 마지막으로 제안된 네트워크의 두 스트림의 점수가 최종 분류를 위해 융합됩니다.



### High-fidelity and Lip-synced Talking Face Synthesis via Landmark-based Diffusion Mod (https://arxiv.org/abs/2408.05416)
Comments:
          submitted to IEEE Transactions on Image Processing(TIP)

- **What's New**: 본 논문은 고품질의 립싱크 동영상을 생성하기 위해 랜드마크 기반 확산 모델(Landmark-based diffusion model)을 제안합니다. 이 모델은 오디오와 비주얼 간의 매핑(Mapping)을 좀 더 명확하게 하기 위해 랜드마크를 중간 표현(Intermediate Representation)으로 사용하며, 각 단계를 통합적으로 최적화하여 오류 누적을 최소화합니다. 또한, TalkFormer라는 새로운 조건 모듈(Conditioning Module)을 도입하여 합성된 모션을 랜드마크에 표현된 모션과 일치시키고, 참조 이미지 특징을 대상 모션에 맞추는 방식을 사용하여 주체의 외관 세부 정보를 보존합니다.



### Style-Preserving Lip Sync via Audio-Aware Style Referenc (https://arxiv.org/abs/2408.05412)
Comments:
          submitted to IEEE Transactions on Image Processing(TIP)

- **What's New**: 본 논문에서는 새로운 오디오 인식 스타일 참조 방식(audio-aware style reference scheme)을 제안하여 개별 스타일을 유지하는 오디오 기반 입술 동기화(style-preserving audio-driven lip sync)를 개선했습니다. 이 방식은 입력 오디오와 스타일 참조 비디오의 참조 오디오 간의 관계를 활용하여 개별적인 발화 스타일을 효과적으로 포착합니다.



### Evolutionary mechanisms that promote cooperation may not promote social welfar (https://arxiv.org/abs/2408.05373)
Comments:
          21 pages, 5 figures

- **What's New**: 본 논문에서는 자기 이익을 추구하는 개인들 사이에서의 사회적 행동의 등장을 이해하는 것이 과학 분야에서 중요한 문제라는 점을 강조하며, 기존 연구들은 주로 협력의 수준을 극대화하는 데 초점을 맞춰왔다는 것을 지적합니다. 그러나 사회적 행동을 유도하는 메커니즘은 개인의 보상에 영향을 미치는 비용을 수반하기 때문에 협력 수준을 극대화하는 것이 사회 전체의 복지(social welfare)에 해가 될 수 있다는 점을 강조합니다. 이 연구에서는 피어 인센티브(peer incentive)와 제도적 인센티브(institutional incentive) 두 가지 메커니즘의 사회적 복지와 협력 수준을 비교 분석함으로써 이러한 상충 관계를 명확히 보여줍니다. 협력 수준을 극대화하는 목표와 사회적 복지를 극대화하는 목표가 항상 일치하는 것은 아니며, 사회적 복지를 최적화 목표로 삼아야 한다는 주장을 제시합니다.

- **Technical Details**: 본 연구에서는 사회적 딜레마(social dilemma) 상황을 모델링하는 데 널리 사용되는 게임인 원샷 프라이즈너스 딜레마(one-shot Prisoner’s Dilemma)를 활용하여, 핀란드의 진화 게임 이론(Evolutionary Game Theory, EGT) 프레임워크를 통해 사회적 복지를 분석합니다. 피어 인센티브와 제도적 인센티브 두 가지 메커니즘을 분석하며, 각각 보상(reward)과 처벌(punishment) 두 가지 유형을 고려합니다. 이 연구에서는 돌연변이율(mutation rate)과 선택 강도(selection intensity) 등 진화 과정의 주요 요소를 고려하여 사회적 복지의 장기적 예상치를 계산하는 공식을 도출합니다.

- **Performance Highlights**: 본 연구에서는 협력 수준이 높아지더라도 사회적 복지가 감소하는 경우가 있음을 보여주는 결과를 통해, 단순히 협력 수준을 극대화하는 데 집중하는 것이 사회적 복지에 해가 될 수 있다는 점을 강조합니다. 이는 사회적 복지를 고려하지 않고 협력 수준을 극대화하는 목표를 추구하는 기존 연구들에 대한 중요한 시사점을 제공합니다. 본 연구는 사회적 복지를 고려하여 사회적 행동 메커니즘을 설계 및 구현하는 데 필요성을 강조하며, 앞으로 다양한 분야에서 사회적 복지를 최적화하는 연구의 중요성을 높일 것으로 예상됩니다.



### A Cost-Effective Eye-Tracker for Early Detection of Mild Cognitive Impairmen (https://arxiv.org/abs/2408.05369)
- **What's New**: 이 논문은 경도인지장애 (MCI)의 조기 진단을 위한 시각적 짝 비교 프로토콜(Visual Paired Comparison) 기반의 저렴한 안구 추적 시스템을 제시합니다. 이 시스템은 기계 학습 알고리즘, 표준 웹캠 및 두 대의 컴퓨터 (각각 환자에게 테스트를 수행하는 '측정 하위 시스템'과 테스트 프로토콜 구성, 환자 데이터 기록, 테스트 모니터링 및 테스트 결과 저장을 담당하는 '테스트 관리 하위 시스템')를 기반으로 합니다. 또한, 광 용적 맥파(photoplethysmography)를 통해 얻은 심박수 변동성을 측정하여 스트레스 추정 기능도 통합합니다.



### FiST-Financial Style Transfer with Hallucination and Creativity Control Framework (https://arxiv.org/abs/2408.05365)
Comments:
          8 pages, 13 figures, 5 tables, conference

- **What's New**: This paper proposes a novel two-stage fine-tuning process for large language models (LLMs) to improve financial report generation. This method addresses two major challenges: the lack of complex sentences and hallucinations in LLM-generated reports. The fine-tuning process involves training the LLM with pre-processed public domain financial reports and then fine-tuning it with simple prompts and tabular data inputs. The proposed method significantly reduces hallucinations and increases the number of correct answers, demonstrating its effectiveness in enhancing the quality of financial report generation.



### MindSpeech: Continuous Imagined Speech Decoding using High-Density fNIRS and Prompt Tuning for Advanced Human-AI Interaction (https://arxiv.org/abs/2408.05362)
- **What's New**: 본 논문은 인간과 AI 에이전트 간의 효과적이고 원활한 상호 작용을 위한 새로운 방법으로 직접 뇌-AI 인터페이스를 개발하는 연구 결과를 보고합니다. 특히, 사용자가 말하지 않고 생각만으로 말하는 내용을 해독하는 'MindSpeech'라는 새로운 AI 모델을 소개합니다. 이 모델은 비침습적인 방법으로 고밀도 기능적 근적외선 분광법(fNIRS) 데이터를 활용하여 상상된 말을 해독합니다. 특히, 참가자들이 다양한 의미 공간을 포괄하는 상상된 문장을 생성하도록 돕는 새로운 '워드 클라우드' 방식을 도입하여 데이터 수집의 질과 다양성을 향상시켰습니다. 또한, 프롬프트 튜닝 기반 접근 방식을 통해 뇌 신호로 안내되는 텍스트 생성에 Llama2 대규모 언어 모델(LLM)을 사용했습니다. 이 연구는 4명의 참가자 중 3명에 대해 BLEU-1 및 BERT P 점수와 같은 주요 지표에서 상당한 개선을 보여주는 결과를 제시하여 이 방법의 효과를 입증합니다. 또한, 여러 참가자의 데이터를 결합하면 해독 성능이 향상된다는 사실을 보여주며, 2명의 참가자에 대해 BERT 점수가 통계적으로 유의미한 개선을 보였습니다. 더욱이, 연구는 상상된 말과 휴지 상태를 비교하여 기회 이상의 해독 정확도를 보였으며, 상상된 말 작업 중에 식별된 활성화된 뇌 영역은 이전 연구에서 말 인코딩에 관여하는 뇌 영역과 일치합니다. 본 연구는 연속적인 상상된 말 해독 가능성을 강조합니다. 고밀도 fNIRS와 고급 AI 기술을 통합하여 가까운 미래에 비침습적이고 정확한 AI와의 의사 소통 시스템에 대한 가능성을 강조합니다.



### MindGPT: Advancing Human-AI Interaction with Non-Invasive fNIRS-Based Imagined Speech Decoding (https://arxiv.org/abs/2408.05361)
- **What's New**: 이 연구는 비침습적 고밀도 기능적 근적외선 분광법(fNIRS)을 사용하여 상상된 음성을 해독하는 혁신적인 접근 방식을 개발하여 인간-AI 상호 작용 분야를 발전시킵니다. 특히 이 연구는 세계 최초로 생각에서 LLM(대규모 언어 모델)으로의 시스템인 MindGPT를 소개합니다.



### Trusting Your AI Agent Emotionally and Cognitively: Development and Validation of a Semantic Differential Scale for AI Trus (https://arxiv.org/abs/2408.05354)
- **What's New**: 본 연구에서는 인공지능 에이전트에 대한 신뢰를 정량화하기 위한 새로운 측정 도구를 제시합니다. 기존의 인공지능 신뢰 연구는 주로 인지적 측면에 집중했지만, 본 연구에서는 인공지능 에이전트에 대한 정서적 신뢰를 측정하는 27개 항목의 의미 차등 척도를 개발하고 검증했습니다. 특히, 최근 대규모 언어 모델(LLMs)의 발전과 함께 인간과 유사한 대화형 에이전트에 대한 정서적 신뢰가 중요해졌으며, 이러한 변화를 반영하여 LLM 기반 대화형 에이전트에 대한 정서적 신뢰를 측정하는 데 중점을 두었습니다.



### Explainable AI Reloaded: Challenging the XAI Status Quo in the Era of Large Language Models (https://arxiv.org/abs/2408.05345)
Comments:
          Accepted to ACM HTTF 2024

- **What's New**: This paper challenges the traditional approach of 'opening the black box' in Explainable AI (XAI) for Large Language Models (LLMs) and advocates for a shift towards a human-centered perspective.

- **Technical Details**: The paper proposes a new approach to XAI by considering three dimensions: explainability outside the black-box, explainability around the edges of the black box, and explainability that leverages infrastructural seams.

- **Performance Highlights**: The paper highlights the challenges and limitations of traditional algorithm-centered XAI approaches, particularly in the context of LLMs. It argues that understanding the inner workings of an LLM might not be feasible or even useful for non-AI experts. Instead, it emphasizes the importance of considering the entire AI lifecycle and providing explanations that are relevant to human users.

- **Key Takeaways**: The paper suggests that a human-centered XAI approach is essential for making LLMs more explainable and trustworthy. This approach involves considering the social and cultural contexts of AI usage and providing explanations that are tailored to the needs of specific audiences. It also highlights the potential of modular architectures, like Retrieval Augmented Generation (RAG), to provide insights into the data flow within and across black-box modules.

- **Limitations**: The paper acknowledges that opening the black box of LLMs is not always feasible due to the complexity of their architectures and the lack of access to internal parameters. However, it argues that this limitation does not preclude the possibility of providing meaningful explanations to human users.



### CAR: Contrast-Agnostic Deformable Medical Image Registration with Contrast-Invariant Latent Regularization (https://arxiv.org/abs/2408.05341)
Comments:
          12 pages, 3 figures, 3 tables, accecpted by WBIR 2024

- **What's New**: 본 논문에서는 훈련 과정에서 다른 영상 대조(contrast)를 관찰하지 않고도 임의의 대조 영상에 일반화될 수 있는 대조 지각(contrast-agnostic) 변형 영상 등록 프레임워크(CAR)를 제안합니다. CAR는 임의의 대조를 모방하기 위한 랜덤 컨볼루션 기반 대조 증강 기법(random convolution-based contrast augmentation)과 대조 불변 표현 학습을 위한 대조 불변 잠재 정규화(contrast-invariant latent regularization)를 제안합니다. 훈련 과정에 사용되지 않은 영상 대조에서도 우수한 성능을 보여주며, 이는 CAR가 기존의 학습 기반 다중 대조 영상 등록 프레임워크와 비교하여 일반화 가능성과 적용 가능성을 크게 향상시킨다는 것을 의미합니다.



### VACoDe: Visual Augmented Contrastive Decoding (https://arxiv.org/abs/2408.05337)
Comments:
          10 pages, 7 figures

- **What's New**: This paper introduces VACoDe (Visual Augmented Contrastive Decoding), a novel method that enhances the performance of Large Vision-Language Models (LVLMs) by effectively addressing the issue of hallucination (generating incorrect outputs). Unlike previous methods that rely on a single augmentation, VACoDe leverages multiple image augmentations and adaptively selects the most contrastive augmentation for each task using a softmax distance metric.

- **Technical Details**: VACoDe works by providing various types of augmented images to the LVLMs and generating multiple outputs. The algorithm then assesses the difference between the original output distribution and the augmented output distributions. The augmentation with the largest distance gap, signifying the highest contrast, is identified and used to produce the final output through contrastive decoding. The core concept behind VACoDe is that different augmentations have distinct impacts on the output distribution of VLMs, and selecting the appropriate augmentation can significantly improve performance.

- **Performance Highlights**: Experimental results demonstrate that VACoDe outperforms existing decoding methods in various vision-language tasks. This method is universally applicable across different model types and sizes without the need for additional training or external models and data. VACoDe offers a robust and efficient way to enhance the accuracy and reliability of LVLMs, particularly in scenarios where hallucination poses a significant challenge.



### Logically Constrained Robotics Transformers for Enhanced Perception-Action Planning (https://arxiv.org/abs/2408.05336)
Comments:
          Robotics Science and Systems: Towards Safe Autonomy

- **What's New**: 본 논문은 대규모 기반 모델(Foundation Model) 기반 계획을 사용하는 상황에서 이해관계자의 의도와 출력의 일치를 보장하기 위해 새로운 접근 방식을 제시합니다. 특히, 자동회귀 변환기 모델(Autoregressive Transformer Model)을 사용하여 궤적 계획을 수행하는 동안 신호 시간 논리(Signal Temporal Logic, STL) 사양을 고려합니다. 또한, 기반 모델을 사전 훈련하고 평가하기 위한 궤적 데이터 세트를 제공합니다.



### Neural Machine Unranking (https://arxiv.org/abs/2408.05330)
- **What's New**: 본 논문은 신경 정보 검색(Neural Information Retrieval, NIR)에서 특정 데이터 포인트를 선택적으로 삭제하는 새로운 머신 언러닝(Machine Unlearning) 기술, Neural Machine UnRanking (NuMuR)을 제시합니다. 기존의 머신 언러닝 방법들은 주로 분류 작업에 초점을 맞춰 설계되었으나, NIR의 고유한 특징 때문에 NuMuR 작업에서 효과적이지 못했습니다.  본 연구에서는 NuMuR에 적합한 Contrastive and Consistent Loss (CoCoL) 방법론을 개발하여 데이터 삭제와 모델 성능 유지를 효과적으로 조화시켰습니다.



### From Text to Insight: Leveraging Large Language Models for Performance Evaluation in Managemen (https://arxiv.org/abs/2408.05328)
Comments:
          39 pages, 8 figures, 5 tables

- **What's New**: 본 연구는 GPT-4와 같은 대규모 언어 모델(LLM)이 조직 업무 수행 평가의 객관성을 향상시키는 데 어떻게 기여할 수 있는지 살펴봅니다. 2가지 연구에 걸친 비교 분석을 통해, 다양한 업무 성과 결과를 분석하여 LLM이 지식 기반 성과 출력(knowledge-based performance outputs)을 평가하는 데 있어 인간 평가자에 비해 신뢰성이 높고 뛰어난 대안이 될 수 있음을 보여줍니다. 지식 기반 성과 출력은 지식 노동자의 핵심 기여입니다.



### A Recurrent YOLOv8-based framework for Event-Based Object Detection (https://arxiv.org/abs/2408.05321)
- **What's New**: 이 연구는 기존 프레임 기반 객체 탐지 시스템에 시공간 모델링 기능을 추가하여 객체 탐지 성능을 향상시킨 새로운 프레임워크인 ReYOLOv8을 소개합니다. 특히, 이 연구에서는 이벤트 데이터를 효율적으로 인코딩하는 저지연, 메모리 효율적인 방법을 구현했으며, 이벤트 데이터의 특징을 활용하도록 설계된 새로운 데이터 증강 기법을 개발했습니다.



### rule4ml: An Open-Source Tool for Resource Utilization and Latency Estimation for ML Models on FPGA (https://arxiv.org/abs/2408.05314)
- **What's New**: 이 논문은 FPGA 상에서 신경망(NN)의 자원 활용 및 추론 지연 시간을 합성 전에 예측하기 위한 새로운 방법을 제시합니다. 이 방법은 HLS4ML을 사용하여 다양한 NN 아키텍처를 합성하고 자원 활용 및 추론 지연 시간 예측기를 훈련합니다. HLS4ML은 합성을 완료해야 자원 및 지연 시간 정보를 얻을 수 있지만, 이 방법은 훈련된 회귀 모델을 사용하여 합성 전에 즉각적인 예측을 수행합니다.

- **Technical Details**: 본 논문에서는 HLS4ML에서 합성된 NN의 자원 활용 및 추론 지연 시간을 예측할 수 있는 회귀 모델을 훈련하기 위해 다양한 NN 아키텍처를 생성 및 합성했습니다. 훈련 데이터 세트는 입력 및 출력 크기, 레이어 및 뉴런 수, 레이어 및 연산 유형(행렬 곱셈, 비선형 활성화, 스킵 연결, 배치 정규화 등)을 포함한 다양한 아키텍처 관련 매개변수를 포함합니다. 모델은 BRAM, DSP, FF 및 LUT의 사용량뿐만 아니라 추론 클럭 사이클을 추정합니다. 예측 모델은 합성된 아키텍처와 기존 벤치마크 아키텍처에서 모두 평가되었으며 검증 세트에서 R2 점수가 0.8~0.98, sMAPE 값이 10~30% 범위에서 높은 정확도를 보였습니다.

- **Performance Highlights**: 이 방법은 FPGA에서 NN의 실행 가능성과 효율성을 빠르게 평가할 수 있게 하여 개발 및 배포 프로세스를 가속화합니다. 예측 모델은 BRAM, DSP, FF 및 LUT의 사용량뿐만 아니라 추론 클럭 사이클을 추정합니다. 예측 모델은 합성된 아키텍처와 기존 벤치마크 아키텍처에서 모두 평가되었으며 검증 세트에서 R2 점수가 0.8~0.98, sMAPE 값이 10~30% 범위에서 높은 정확도를 보였습니다.



### The impact of internal variability on benchmarking deep learning climate emulators (https://arxiv.org/abs/2408.05288)
- **What's New**: 본 논문에서는 기존 기후 모델 에뮬레이터 벤치마크인 ClimateBench를 분석하여 선형 회귀 기반 에뮬레이터인 LPS(Linear Pattern Scaling)가 심층 학습 기반 에뮬레이터인 ClimaX보다 지역별 표면 온도, 강수량 및 극심한 강수량 예측에 더 우수한 성능을 보인다는 것을 발견했습니다. 이는 강수량이 비선형적 관계를 가지고 있다는 것을 고려했을 때 놀라운 결과입니다. 또한 ClimateBench에서 사용되는 3개의 시뮬레이션이 내부 변동성을 완전히 제거하기에 충분하지 않다는 점을 강조합니다.



### Semi-Supervised One-Shot Imitation Learning (https://arxiv.org/abs/2408.05285)
- **What's New**: 이 연구는 One-Shot Imitation Learning (OSIL)의 효율성을 높이기 위한 새로운 방법을 제시합니다. OSIL은 한 번의 시연만으로 새로운 작업을 학습할 수 있는 인공지능 에이전트를 개발하는 것을 목표로 합니다. 이 연구는 특히 레이블이 없는 대량의 데이터셋과 몇 개의 레이블이 있는 데이터셋을 함께 사용하는 **반지도 학습 (semi-supervised learning)** 방식을 OSIL에 적용했습니다. 이를 통해 기존 OSIL의 한계인 레이블 수집의 어려움을 극복하고, 더 적은 레이블로도 효과적인 학습이 가능하도록 했습니다.



### scASDC: Attention Enhanced Structural Deep Clustering for Single-cell RNA-seq Data (https://arxiv.org/abs/2408.05258)
- **What's New**: scASDC (Attention-Enhanced Structural Deep Embedding Graph Clustering)라는 새로운 딥 클러스터링 방법을 제안합니다. scASDC는 싱글 셀 RNA 시퀀싱(scRNA-seq) 데이터에서 세포 간의 관계를 포착하고 클러스터링 성능을 향상시키기 위해 멀티 레이어 그래프 합성곱 네트워크(GCN)와 ZINB 기반 오토인코더를 통합합니다.



### A Systematic Literature Map on Big Data (https://arxiv.org/abs/2408.05253)
Comments:
          8 pages, 1 figure, 5 tables

- **What's New**: 본 연구는 빅데이터 패러다임에 대한 체계적인 분석을 통해 연구 동향과 미래 전망을 제시합니다. 특히, 빅데이터 개념에 대한 다양한 정의와 연구 분야를 폭넓게 살펴보고,  체계적인 문헌 분석을 통해 연구의 흐름, 추세, 그리고 미흡한 부분을 파악합니다.



### Advancing oncology with federated learning: transcending boundaries in breast, lung, and prostate cancer. A systematic review (https://arxiv.org/abs/2408.05249)
Comments:
          5 Figures, 3 Tables, 1 Supplementary Table

- **What's New**: 본 논문은 암 치료, 특히 유방암, 폐암, 전립선암에서 중앙집중식 기계 학습(ML)의 한계를 해결하기 위한 유망한 해결책으로 부상한 연합 학습(FL)에 대한 최신 지식을 종합적으로 검토했습니다. 이전 연구와는 달리, 본 연구는 FL이 실제 세계에서 암 치료에 미치는 영향과 임팩트를 비판적으로 평가하여 임상 환경과 데이터에서 ML 일반화, 성능 및 데이터 프라이버시를 향상시키는 데 효과적임을 보여줍니다. 또한, 엄격한 데이터 프라이버시 규정 속에서 FL의 채택이 증가하고 있음을 보여주는 FL의 최첨단 발전을 평가했습니다. FL은 25개 연구 중 15개에서 중앙집중식 ML을 능가했으며, 다양한 ML 모델 및 임상 적용을 포괄하고 정밀 의학을 위한 다중 모달 정보 통합을 가능하게 했습니다. 연구 간 재현성, 표준화 및 방법론에서 현재의 과제에도 불구하고, 실제 데이터를 활용하고 임상적 요구를 해결하는 데 있어 FL의 확실한 이점은 암 연구 발전을 위한 FL의 잠재력을 강조합니다. 미래 연구는 이러한 제한 사항을 해결하고 고급 FL 방법을 더욱 조사하여 데이터 다양성을 완전히 활용하고 최첨단 FL의 혁신적인 힘을 암 치료에 실현해야 합니다.



### The Role and Applications of Airport Digital Twin in Cyberattack Protection during the Generative AI Era (https://arxiv.org/abs/2408.05248)
- **What's New**: 이 논문은 공항 운영의 보안을 강화하기 위한 새로운 접근 방식으로 **디지털 트윈(Digital Twins)**과 **생성적 인공지능(Generative AI)**의 통합을 제안합니다. 특히, 공항 사이버 공격에 대한 방어를 강화하는 데 초점을 맞춥니다.



### Early-Exit meets Model-Distributed Inference at Edge Networks (https://arxiv.org/abs/2408.05247)
- **What's New**: 이 논문은 모델 분산 추론(MDI)에 조기 종료(early-exit) 메커니즘을 통합하여 분산 딥 러닝 추론의 성능을 향상시키는 새로운 프레임워크인 MDI-Exit를 제안합니다.

- **Technical Details**: MDI-Exit는 네트워크 유틸리티 극대화(NUM) 개념을 기반으로 각 작업자에서 모델 할당 및 조기 종료 결정을 적응적으로 수행합니다. MDI-Exit는 작업자 간의 대기열 크기를 기반으로 데이터 전송 및 처리를 제어하여 최적의 성능을 달성합니다.

- **Performance Highlights**: 실제 NVIDIA Jetson TX2 장치 테스트베드를 사용한 실험 결과는 MDI-Exit가 정확도가 고정된 경우 더 많은 데이터를 처리하고 데이터 속도가 고정된 경우 기준 모델(MobileNetV2, ImageNet)에 비해 더 높은 정확도를 달성한다는 것을 보여줍니다.

- **Keywords**: Model-distributed inference (MDI), Early-exit, Network Utility Maximization (NUM), Edge Computing, Deep Neural Networks (DNNs), MobileNetV2, ImageNet



### Differentially Private Data Release on Graphs: Inefficiencies and Unfairness (https://arxiv.org/abs/2408.05246)
Comments:
          32 pages

- **What's New**: 이 논문은 네트워크 데이터를 개인정보 보호 방식으로 공개할 때 발생하는 편향과 불공정성을 분석하는 새로운 연구입니다. 특히, 네트워크 구조는 알려져 있지만 엣지 가중치는 민감한 정보이기 때문에 개인정보 보호 방식으로 공개해야 하는 상황을 가정합니다. 이는 교통 네트워크에서 사용자에게 정확한 경로 정보를 제공하면서 개인정보를 보호하는 문제와 유사합니다. 이 연구는 차별적 프라이버시(DP)가 네트워크 기반 의사 결정에 미치는 영향, 특히 최단 경로 계산과 최적 경로 추천에 대한 영향을 분석합니다.



### Improved Adaboost Algorithm for Web Advertisement Click Prediction Based on Long Short-Term Memory Networks (https://arxiv.org/abs/2408.05245)
- **What's New**: 이 논문은 웹 페이지 광고에 대한 사용자 클릭 예측 정확도를 향상시키기 위해 장단기 기억 네트워크(LSTM)를 기반으로 한 향상된 Adaboost 알고리즘을 제시합니다. 이 알고리즘은 여러 일반적인 기계 학습 알고리즘과 비교하여 광고 클릭 예측에서 새로운 모델의 장점을 분석합니다.



### Large Model Strategic Thinking, Small Model Efficiency: Transferring Theory of Mind in Large Language Models (https://arxiv.org/abs/2408.05241)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문은 더 작고, 시뮬레이션에 적합한 에이전트를 미세 조정을 통해 만들 수 있는지 조사합니다. 20가지의 고유한 시나리오(각 시나리오는 사회적 맥락과 사회적 딜레마를 결합)를 통해 대규모 사전 훈련된 모델을 사용하여 답변을 기록하고 이를 동일 계열의 더 작은 모델에 대한 Q&A 미세 조정에 사용합니다.



### The Literature Review Network: An Explainable Artificial Intelligence for Systematic Literature Reviews, Meta-analyses, and Method Developmen (https://arxiv.org/abs/2408.05239)
Comments:
          12 pages, 4 figures, 10 tables

- **What's New**: LRN, 최초의 PRISMA 2020 기준을 준수하는 설명 가능한 AI 플랫폼은 전체 문헌 검토 프로세스를 자동화하기 위해 설계되었습니다.  LRN은 전문가가 PubMed 쿼리에 사용하는 3가지 검색 문자열을 사용하여 수술용 장갑 관행 영역에서 평가되었습니다.  비전문가가 모든 LRN 모델을 훈련시켰습니다.  성능은 전문가 수동 검토와 비교되었습니다. 설명 가능성과 성능 지표는 전문가 검토를 복제하는 LRN의 능력을 평가했습니다. 일치도는 Jaccard 지수와 혼동 행렬로 측정되었습니다. 연구자들은 연구 완료 전까지 서로의 결과를 몰랐습니다. 겹치는 연구는 LRN에서 생성한 체계적 검토에 통합되었습니다.



### Biomimetic Machine Learning approach for prediction of mechanical properties of Additive Friction Stir Deposited Aluminum alloys based walled structures (https://arxiv.org/abs/2408.05237)
Comments:
          26 pages, 14 figures, 6 tables

- **What's New**: 본 연구는 생체 모방 기계 학습을 이용하여 AFSD (Additive Friction Stir Deposition) 알루미늄 합금 벽 구조의 기계적 특성을 예측하는 새로운 방법을 제시합니다. 본 연구는 AFSD 공정의 수치 모델링과 유전 알고리즘 최적화된 기계 학습 모델을 결합하여 von Mises 응력과 로그 변형률을 예측합니다. 유한 요소 해석을 사용하여 AA2024, AA5083, AA5086, AA7075 및 AA6061의 다섯 가지 알루미늄 합금에 대한 AFSD 공정을 시뮬레이션하고 복잡한 열 및 기계적 상호 작용을 포착했습니다. 이러한 시뮬레이션에서 200개의 샘플로 구성된 데이터 세트를 생성했습니다. 이후 유전 알고리즘을 사용하여 최적화된 의사 결정 트리(DT) 및 랜덤 포레스트(RF) 회귀 모델을 개발하여 주요 기계적 특성을 예측했습니다. GA-RF 모델은 von Mises 응력(R 제곱 = 0.9676)과 로그 변형률(R 제곱 = 0.7201) 모두를 예측하는 데 뛰어난 성능을 보였습니다. 이 혁신적인 접근 방식은 다양한 알루미늄 합금에 걸쳐 AFSD 공정을 이해하고 최적화하기 위한 강력한 도구를 제공하여 다양한 공정 매개 변수 하에서 재료 거동에 대한 통찰력을 제공합니다.



### SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving (https://arxiv.org/abs/2408.05235)
- **What's New**: 본 논문에서는 LLM (대규모 언어 모델)의 에너지 효율을 개선하기 위한 새로운 프레임워크인 throttLL'eM을 소개합니다. throttLL'eM은 GPU 주파수 스케일링과 인스턴스 자동 스케일링을 활용하여 LLM 추론 작업의 에너지 소비를 줄이는 동시에 SLO(서비스 수준 목표)를 만족합니다.

- **Technical Details**: throttLL'eM은 LLM 추론 과정에서 발생하는 KV 캐시 사용량과 배치 크기를 예측하는 예측 모델을 사용합니다. 이러한 예측 모델은 ML 모델에 의해 구현되며, 미래의 LLM 반복에서 시스템 처리량을 예측합니다. 이러한 예측을 토대로 throttLL'eM은 SLO를 충족하면서 최소 주파수와 인스턴스 크기를 사용하여 에너지 사용을 최적화합니다. 또한, throttLL'eM은 들어오는 작업 부하에 따라 엔진의 병렬 수준을 조정하는 자동 스케일링 메커니즘을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, throttLL'eM은 NVIDIA의 Triton 서버에 비해 최대 43.8%의 에너지 소비 감소와 최소 1.71배의 에너지 효율 향상을 보였습니다. throttLL'eM은 p99 (99번째 백분위수) 응답 시간 SLO를 기존 시스템의 최대 부하와 동일하게 유지하고 TBT(Time Between Tokens) SLO를 사람의 읽기 속도와 일치시키도록 설계되었습니다.



### Predictive maintenance solution for industrial systems -- an unsupervised approach based on log periodic power law (https://arxiv.org/abs/2408.05231)
Comments:
          14 pages, 4 figures, 1 table

- **What's New**: 새로운 비지도 예측 유지보수 분석 방법이 제안되었으며, 이는 복잡한 시스템에서 임계 행동을 발견하는 데 사용되는 재규격화 군 접근 방식을 기반으로 합니다. 이 알고리즘은 단변량 시계열을 분석하고, 새로운 정리에 기반하여 임계점을 감지하는데, 이 정리는 로그 주기적 멱 법칙 함수 적합을 사용하여 임계점을 식별합니다. 왕복식 압축기 시스템에서 수집한 산업 데이터에 대한 예측 유지보수 분석을 위한 새로운 알고리즘의 적용이 제시됩니다. 분석된 압축기 시스템의 동역학에 대한 지식을 바탕으로 제안된 알고리즘은 밸브 및 피스톤 로드 씰 고장을 미리 예측합니다.



### Large Language Models for cross-language code clone detection (https://arxiv.org/abs/2408.04430)
- **What's New**: This paper explores the use of Large Language Models (LLMs) and embedding models for cross-lingual code clone detection. It investigates the capabilities of four LLMs (Falcon-7B-Instruct, LLAMA2-Chat-7B, Starchat-β𝛽etaitalic_β, and GPT-3.5-Turbo) and eight prompts for identifying cross-lingual code clones. The study also evaluates a pre-trained embedding model ('Text-Embedding-Ada-002') to assess the effectiveness of generated representations for classifying clone and non-clone pairs.

- **Technical Details**: The paper investigates the use of LLMs with various prompts, including simple prompts, Chain of Thought (CoT) prompts, and a pre-trained embedding model. The LLMs are tasked with directly providing 'yes/no' answers or following step-by-step reasoning to determine if two code snippets are clones. The embedding model generates vector representations of code snippets, which are then used to measure similarity and train custom binary classifiers.

- **Performance Highlights**: The results show that LLMs can achieve high F1 scores (up to 0.98) for straightforward programming examples. However, they perform less well on complex programming challenges and may not fully understand the concept of cross-lingual code clones. The embedding model outperforms all LLMs by ~2 and ~24 percentage points on the XLCoST and CodeNet datasets, respectively, indicating that embeddings provide suitable representations for state-of-the-art performance in cross-lingual code clone detection.



### Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings (https://arxiv.org/abs/2306.17670)
- **What's New**: This paper proposes a novel algorithm for learning delays in deep feedforward Spiking Neural Networks (SNNs) using backpropagation in an offline manner. The algorithm leverages 1D convolutions across time to simulate delays between layers, where kernel positions correspond to delays and are learned alongside weights using Dilated Convolution with Learnable Spacings (DCLS). This method significantly outperforms prior approaches in accuracy and efficiency on three datasets: Spiking Heidelberg Dataset (SHD), Spiking Speech Commands (SSC), and Google Speech Commands (GSC).



### Synchronous Multi-modal Semantic Communication System with Packet-level Coding (https://arxiv.org/abs/2408.04535)
Comments:
          12 pages, 9 figures

- **What's New**: 본 논문은 다중 모드 데이터의 동기화 및 패킷 수준 오류 정정을 고려한 새로운 다중 모드 의미적 통신 시스템을 제안합니다. 이 시스템은 비디오와 음성을 동시에 전송하는 동기화된 다중 모드 의미적 통신 시스템(SyncSC)을 제안합니다. 특히, 3D Morphable Model(3DMM) 계수를 사용하여 얼굴을 표현하고 음성 인식 결과인 텍스트를 음성의 의미로 사용하여 전송 대역폭을 효율적으로 줄입니다.  또한, 패킷 수준 오류 정정을 위해 비디오 의미에 대한 Erasure coding, 텍스트에 대한 BERT 기반 TextPC 모듈을 제안하여 높은 패킷 손실률에서도 성능을 유지합니다.




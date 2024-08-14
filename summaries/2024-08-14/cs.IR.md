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




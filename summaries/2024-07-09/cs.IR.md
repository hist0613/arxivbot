New uploads on arXiv(cs.CL)

### Using Grammar Masking to Ensure Syntactic Validity in LLM-based Modeling Tasks (https://arxiv.org/abs/2407.06146)
Comments:
          Preprint to be published in the MODELS Workshop "MDE Intelligence"

- **What's New**: 새로운 방법인 '문법 마스킹(Grammar Masking)'을 통해 대규모 언어 모델(LLM)이 주어진 문맥 무관 문법(Context-Free Grammar)의 구문을 올바르게 생성하도록 유도하는 방안을 발표하고 평가했습니다. 이 방법은 구문적으로 올바른 출력을 보장하기 위해 제한된 디코딩(Constrained Decoding)을 사용합니다. 이전의 방법인 소수 샷 학습(Few-Shot Learning)이나 프롬프트 엔지니어링처럼 많은 시간이 소요되거나 성공 확률이 낮은 점을 극복하고자 합니다.

- **Technical Details**: 이 방법에서는 대상 DSL의 문맥 무관 문법(CFG)을 사용하여 생성 과정 중에 구문적으로 유효하지 않은 출력을 필터링합니다. MontiCore를 이용해 만든 여러 DSL을 사용하고, LLM들이 제약 없이 생성한 모델과 제약된 디코딩으로 생성한 모델을 비교합니다. Guidance 프레임워크를 통해 LLM의 출력을 구조화하는 과정에서 사용됩니다. 파서는 모든 가능한 경로를 처리하며, 복잡하고 모호한 문법도 효과적으로 처리합니다.

- **Performance Highlights**: 문법 마스킹은 여러 LLM의 모델링 능력을 크게 향상시키며, 잘 다듬어진 프롬프트에 대한 의존도를 줄이면서도 올바른 모델을 생성할 가능성을 높입니다.



### ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation (https://arxiv.org/abs/2407.06135)
- **What's New**: Anole은 기존 오픈소스 대규모 멀티모달 모델(LMMs)의 여러 한계를 극복하고자 한 새로운 모델입니다. Anole은 멀티모달 이미지-텍스트 생성을 위한 오픈, 자동회귀, 네이티브 대규모 멀티모달 모델로, Meta AI의 Chameleon에서 비롯되었습니다.

- **Technical Details**: Anole은 네이티브 멀티모달 생성을 지원하며, 이미지 생성 및 멀티모달 생성 기능을 제공합니다. Chameleon의 장점을 유지하면서도 이를 확장한 Anole은 데이터와 파라미터 효율적인 파인튜닝(fine-tuning) 방식을 채택하고 있습니다. 이 방식은 6,000개의 샘플과 40M 미만의 파라미터로 멀티모달 생성 능력을 효과적으로 활성화합니다. 또한, 통합 토큰화기 기반 멀티모달 모델을 위한 교육 및 멀티모달 추론 프레임워크를 제공합니다.

- **Performance Highlights**: Anole은 고품질의 일관된 멀티모달 생성 기능을 보여주며, 이는 효율적이고 간단한 구조를 통해 가능했습니다. 복잡한 구성 요소 없이도 부드러운 추론과 고품질의 이미지 텍스트 시퀀스를 생성할 수 있습니다.



### Enhancing Language Model Rationality with Bi-Directional Deliberation Reasoning (https://arxiv.org/abs/2407.06112)
- **What's New**: 이번 논문은 BI-Directional DEliberation Reasoning (BIDDER)라는 새로운 사고 접근 방식을 소개하여 언어 모델의 의사결정 합리성을 향상시킵니다. 기존의 사고 방법은 주로 과거 정보에 의존하며 단방향(왼쪽에서 오른쪽) 사고 전략을 사용합니다. BIDDER는 불확실성을 관리하고 예상 효용을 예측하는 합리적 의사결정 원칙을 포함함으로써 이 문제를 해결합니다.

- **Technical Details**: BIDDER는 크게 세 가지 프로세스로 구성됩니다: 역사적 데이터로부터 의사결정 과정에서 불확실한 정보를 나타내는 숨겨진 상태를 추론; 이 숨겨진 상태를 사용하여 미래의 잠재적 상태와 결과를 예측; 역사적 정보(과거 문맥)와 장기적 결과(미래 문맥)를 통합하여 사고를 돕는 것입니다. 이러한 양방향 사고를 활용함으로써 BIDDER는 과거와 미래 문맥을 모두 철저히 탐색하여 더 정보에 근거한 합리적 결정을 내릴 수 있습니다.

- **Performance Highlights**: BIDDER의 효과를 Poker(포커 테이블)와 협상 시나리오에서 테스트 한 결과, BIDDER가 언어 모델의 의사결정 능력을 크게 향상시키며 이들의 행동을 최적화된 솔루션과 더 맞추는 것으로 나타났습니다. BIDDER는 장기적으로 긍정적인 결과를 예측하고, 역사적 문맥 및 미래 탐색을 효과적으로 통합하여 불확실성을 최소화합니다.



### Epistemological Bias As a Means for the Automated Detection of Injustices in Tex (https://arxiv.org/abs/2407.06098)
- **What's New**: 이번 연구는 텍스트에서 발생하는 불의(injustice)를 자동으로 식별하기 위한 새로운 프레임워크를 제안합니다. Bias detection model, stereotype detection models, lexicon-based approach 등을 조합하여 불의를 감지하는 방식입니다. 특히, epistemological biases를 이용하여 뉴스 미디어에서의 불의를 자동으로 감지하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 fine-tuned BERT 기반의 bias detection model을 사용하여 텍스트에서의 epistemological bias를 태그합니다. 또한 CO-STAR 모델(Kwon and Gopalan, 2021)과 Social Bias Frames 모듈(SBF, Sap et al., 2020)을 사용하여 stereotype과 stereotype의 개념을 분석합니다. 이를 통해 태그된 단어들이 특정 stereotype과 연관되는 경우, 그 단어들이 불의를 유발하는지 확인합니다. 이러한 모델들을 결합하여 testimonial injustice, character injustice, framing injustice를 감지할 수 있는 프레임워크를 개발하였습니다.

- **Performance Highlights**: 본 프레임워크는 뉴스 미디어에서 자동으로 character, testimonial, framing 불의를 감지할 수 있게 도와줍니다. 이를 통해 기자 및 편집자가 더 쉽게 불의한 표현을 식별하고 피할 수 있습니다. 제공되는 사용자인터페이스는 기자들이 텍스트를 제출하면 태그된 단어 및 설명을 시각화하여 불의를 예측할 수 있게 해줍니다. 연구 결과는 epistemological bias가 불의로 번역될 수 있음을 실증적으로 증명하였습니다.



### Merge, Ensemble, and Cooperate! A Survey on Collaborative Strategies in the Era of Large Language Models (https://arxiv.org/abs/2407.06089)
- **What's New**: 최근 연구는 다양한 대형 언어 모델(LLMs)의 협력 전략을 탐구하고 있습니다. 이에 대한 논문은 협력 전략을 결합(Merging), 앙상블(Ensemble), 협력(Cooperation) 세 가지 주요 접근 방식으로 분류하여 소개합니다. 이를 통해 각 모델의 고유한 강점을 최대한 활용하려고 합니다.

- **Technical Details**: 1. **Merging (결합)**: 여러 LLM의 파라미터를 통합하여 하나의 단일 모델로 만드는 접근 방식입니다. 이는 파라미터가 선형 공간(linear space) 내에서 호환되는 경우에 사용됩니다.
2. **Ensemble (앙상블)**: 각 LLM이 생성한 출력 결과를 결합하여 종합적인 결과를 도출하는 방법입니다. 파라미터 자체보다는 출력물에 중점을 둡니다.
3. **Cooperation (협력)**: 각 LLM의 다양한 능력을 최대한 활용하여 특정 작업을 달성하는 방법입니다. 이는 결합과 앙상블을 넘어서는 방식을 포함합니다.

각 접근 방식은 모델의 특성과 사용 사례에 맞춰 최적화되어 있습니다.

- **Performance Highlights**: 논문에서는 다양한 협력 전략을 통해 LLM들의 협업이 어떻게 성과를 증대시킬 수 있는지 설명하고 있습니다. 결합 방식은 다중 파라미터를 활용해 단일 모델로 합치는 데 초점을 맞추며, 앙상블은 여러 출력 결과를 결합해 종합적인 응답을 제공합니다. 협력 방식은 모델의 강점을 극대화하여 특정 목적을 달성하는 데 효과적입니다. 이를 통해 자연어 처리(NLP) 응용 프로그램의 발전에 크게 기여할 수 있습니다.



### From Loops to Oops: Fallback Behaviors of Language Models Under Uncertainty (https://arxiv.org/abs/2407.06071)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 불확실성 하에서 발생하는 fallback behaviors(태만 행동)의 패턴을 분석하고 정리합니다. 저자들은 모델의 불확실성에서 나타나는 various fallback behaviors — 시퀀스 반복, 변질된 텍스트(degenerate text), 그리고 헛소리(hallucinations)를 고려해 분석하였습니다. 이를 통해 모델의 강도가 증가할수록 태만 행동이 더 복잡해지는 경향, 즉 시퀀스 반복에서 변질된 텍스트로, 그리고 헛소리로 이동하는 경향을 발견했습니다.

- **Technical Details**: 세 모델 가족인 Pythia, Llama 2, Llama 3, 그리고 OLMo를 대상으로 실험을 진행하였으며, 다양한 요소들(매개변수 수, 전처리 토큰 수, 명령어 준수 훈련 여부, 디코딩 알고리즘 등)을 고려하여 태만 행동의 출현을 분석했습니다. 실험 설정은 자연스러운 불확실성 환경에서 모델이 사실적인 정보를 생성하도록 하고, 예측의 정확성을 평가하는 방식으로 구성되었습니다.

- **Performance Highlights**: 주요 발견점은 모델이 강해질수록 태만 행동이 더 복잡한 형태로 변화한다는 것입니다. 구체적으로, 시퀀스 반복이 가장 간단한 fallback이며, 변질된 텍스트가 중간 단계, 헛소리가 가장 복잡한 형태로 나타납니다. 디코딩 방식을 사용해 일부 태만 행동을 완화시킬 수 있지만, 이로 인해 감지하기 어려운 헛소리의 비율이 증가할 수 있습니다. 예를 들어, 랜덤 샘플링은 변질된 텍스트를 줄일 수 있으나 헛소리를 증가시킵니다.



### Variational Best-of-N Alignmen (https://arxiv.org/abs/2407.06057)
- **What's New**: 최근 언어 모델을 인간의 선호도에 맞추는 효율적인 알고리즘인 Best-of-N (BoN)에 대한 연구가 발표되었습니다. BoN은 초기 추론 단계에서 N개의 샘플을 추출한 후, 보상 모델이 가장 높은 보상을 준 샘플을 최종 결과로 반환합니다. 그러나 BoN은 높은 계산 비용이 필요하다는 단점이 있습니다. 이를 해결하기 위해 연구진은 BoN의 동작을 모방하도록 언어 모델을 미세 조정하는 variational BoN (vBoN)을 제안하였습니다.

- **Technical Details**: vBoN은 BoN 알고리즘이 유도하는 분포를 도출하고, 언어 모델을 이 분포와 뒤쪽 KL 다이버전스를 최소화하도록 미세 조정합니다. 이는 mean-field variational inference와 유사한 접근 방식으로, PPO(Proximal Policy Optimization)를 사용하여 최적화됩니다. BoN과 유사한 성능을 유지하면서 추론 비용을 N배 줄일 수 있는 장점이 있습니다.

- **Performance Highlights**: 제안된 vBoN 방법은 제어된 생성 작업에서 실험을 통해 BoN만큼은 아니지만 거의 근접한 성능을 보였습니다. 보상과 KL 다이버전스의 파레토 프론티어에 더 자주 나타나는 경향이 있으며, 이는 KL-제약 RL objective로 미세 조정된 모델들과 비교해도 상당히 양호한 성능을 보여줍니다. vBoN 변수로 미세 조정된 모델은 긍정적인 여론의 영화 리뷰를 생성할 가능성이 더 높고, reference 모델과의 편차도 적습니다.



### Vision-Braille: An End-to-End Tool for Chinese Braille Image-to-Text Translation (https://arxiv.org/abs/2407.06048)
Comments:
          This paper is submitted to NeurIPS 2024 High School Project Track

- **What's New**: 이 연구는 인공지능(AI)을 활용하여 시각 장애 학생들의 교육 접근성을 높이기 위한 한국어-점자 번역 시스템 개발에 초점을 맞추고 있습니다. 특히, 중국어 점자의 특수성을 고려하여 고도의 정확도를 자랑하는 번역 시스템을 구축했습니다.

- **Technical Details**: 본 프로젝트에서는 mT5 모델의 Encoder-decoder 아키텍처를 점자-중국어 문자 변환에 맞게 미세 조정(fine-tuning)했습니다. 이 과정에서 Leipzig Corpora로부터 점자와 이에 대응하는 중국어 텍스트를 학습 데이터셋으로 사용했습니다. 점자 OCR(Optical Character Recognition)을 위해 RetinaNet을 활용해 이미지를 분석했고, mT5 모델을 활용한 텍스트 번역 부분에서는 다언어 지원이 되는 mT5-Small 모델을 사용했습니다.

- **Performance Highlights**: 이 시스템은 검증 세트와 테스트 세트에서 각각 62.4 및 62.3 BLEU 점수를 달성, 혼란스러운 점자 번역 문제를 크게 개선했습니다. 이를 통해 많은 시각 장애학생들이 교육 자원에 더 쉽게 접근할 수 있게 될 전망입니다.



### MST5 -- Multilingual Question Answering over Knowledge Graphs (https://arxiv.org/abs/2407.06041)
- **What's New**: 본 연구에서는 자연어를 사용하여 대용량 지식 그래프를 쿼리하는 Knowledge Graph Question Answering (KGQA) 시스템의 다국어 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존 방법들과 달리 우리는 언어 모델의 처리 파이프라인에 언어적인 문맥과 엔티티 정보를 직접 통합하는 방식을 채택했습니다.

- **Technical Details**: 이 접근 방식은 별도의 인코더를 사용하지 않고, 하나의 사전 학습된 멀티링구얼 트랜스포머 기반 언어 모델(multilingual transformer-based language model)을 사용하여 기본 입력과 보조 데이터를 모두 관리합니다. 여기에서 주 입력은 자연어 질의이고, 보조 데이터로는 언어적 문맥과 엔티티 정보가 포함됩니다. 시퀀스 투 시퀀스(sequence-to-sequence) 작업 설정에서, 입력 시퀀스는 자연어 질의이고 출력은 필요한 SPARQL 쿼리를 생성합니다.

- **Performance Highlights**: 이 방법론은 최신 QALD 데이터셋(QALD-9-Plus 및 QALD-10)에서 유망한 결과를 나타냈습니다. 또한, 중국어 및 일본어에 대해 접근 방식을 도입하고 평가하여 기존 데이터셋의 언어 다양성을 확장했습니다. 실험 결과, 언어적 문맥과 엔티티 정보를 통합하면 KGQA 성능이 크게 향상되는 것으로 나타났습니다.



### PAS: Data-Efficient Plug-and-Play Prompt Augmentation System (https://arxiv.org/abs/2407.06027)
- **What's New**: PAS는 최근 자동 프롬프트 엔지니어링(APE) 모델의 한계를 극복하기 위해 제안된 대형 언어 모델(LLM) 기반 플러그 앤 플레이 시스템입니다. 기존 APE 모델의 비효율성, 낮은 유연성, 낮은 효과성을 개선하여, 자동으로 프롬프트 보완 데이터를 생성하고 이를 통해 LLM을 미세 조정합니다. PAS는 인간의 개입 없이 고품질의 보완 데이터셋을 구축하며, 모든 LLM과 호환 가능하고 다양한 작업에 적응할 수 있습니다.

- **Technical Details**: PAS는 두 개의 주요 단계로 구성됩니다. 첫 번째 단계는 고품질 프롬프트 선별 및 자동 보완 프롬프트 생성입니다. 이 과정에서 임베딩 모델을 사용해 프롬프트 데이터를 클러스터링하고, 유사한 프롬프트를 그룹화 및 중복 제거합니다. 이후 LLM이 고품질 프롬프트를 선택하고 이를 다양한 카테고리로 분류합니다. 두 번째 단계는 few-shot 학습을 이용한 새로운 프롬프트 생성입니다. 생성된 프롬프트는 엄격한 선별과 재생성 과정을 거쳐 품질이 유지됩니다. 최종적으로, 이 고품질 데이터는 LLM 미세 조정에 사용됩니다.

- **Performance Highlights**: PAS는 다양한 벤치마크에서 이전 APE 모델 대비 평균 6.09 포인트의 성능 개선을 보여줍니다. 또한, PAS는 SoTA 성능을 단 9000개의 데이터 포인트로 달성하며 높은 효율성을 보입니다. It is highly flexible and can be plugged into any LLM to solve a wide range of tasks. 인간 평가에서도 뛰어난 성과를 보였으며, 이는 PAS가 사용하기 매우 적합한 플러그인임을 시사합니다.



### Distilling System 2 into System 1 (https://arxiv.org/abs/2407.06023)
- **What's New**: 이번 연구에서는 자기지도 학습(self-supervised learning) 방법을 사용하여 System 1 모델의 성능을 향상시키기 위해 System 2 기술의 출력물을 증류(distill)하는 방법을 탐구합니다. 이는 중간 추론 토큰 시퀀스를 생성하지 않고도 고품질의 출력물을 만들어내는 것입니다.

- **Technical Details**: 이번 연구에서는 다양한 System 2 기술(Chain-of-Thought, Rephrase and Respond, System 2 Attention, Branch-Solve-Merge 등)을 사용하여 System 1 모델을 미세 조정(fine-tune)하는 방법을 설명합니다. System 2 기법을 통해 생성된 예측 결과를 증류(pool of examples)에 추가하고, System 1 모델을 이를 학습하게 하여 중간 결과 없이 최종 예측을 만들어냅니다.

- **Performance Highlights**: 4가지 다른 System 2 LLM 접근법과 5가지 다른 작업에서 실험을 수행한 결과, System 2 추론을 System 1에 성공적으로 증류할 수 있었으며, 일부 경우에서는 System 2 교사 모델보다 더 나은 결과를 얻었습니다. 또한 이 예측은 System 2 기법보다 훨씬 적은 계산 비용으로 생성될 수 있었습니다.



### Igea: a Decoder-Only Language Model for Biomedical Text Generation in Italian (https://arxiv.org/abs/2407.06011)
Comments:
          6 pages, 1 figure, 3 tables

- **What's New**: 본 논문은 이탈리아어를 사용하는 바이오메디컬 텍스트 생성 전용 언어 모델(Igea)을 소개합니다. 이는 이탈리아어 기반의 바이오메디컬 자연어 처리(NLP) 분야에서 최초로 시도되는 디코더 전용(Decoder-Only) 언어 모델입니다.

- **Technical Details**: Igea는 Minerva 모델을 기반으로 구축되었으며, 이탈리아어 의료 텍스트의 다양한 말뭉치에 대한 지속적인 사전 훈련을 통해 개발되었습니다. 이 모델은 3가지 크기로 제공됩니다: 3억5천만, 10억, 30억 파라미터입니다. 이러한 모델들은 이탈리아어 바이오메디컬 용어의 특성을 다루는 데 있어 성능과 계산 효율성의 균형을 맞추는 것을 목표로 합니다.

- **Performance Highlights**: Igea 모델은 도메인 내 바이오메디컬 말뭉치와 일반 목적의 벤치마크를 혼합하여 평가되었습니다. 이로써 도메인 특화 훈련 후에도 일반 지식을 유지하는데 효과적임을 강조합니다. 이러한 모델 개발과 평가를 통해 이탈리아어 바이오메디컬 NLP의 향후 발전을 위한 기반을 마련했습니다.



### Perceptions to Beliefs: Exploring Precursory Inferences for Theory of Mind in Large Language Models (https://arxiv.org/abs/2407.06004)
- **What's New**: 최신 연구에서는 대형 언어 모델(LLMs)이 인간의 '마음 이론(Theory of Mind, ToM)' 능력을 평가하기 위해 두 가지 핵심 전제 조건인 '지각 추론(perception inference)'과 '지각-신념 추론(perception-to-belief inference)'을 평가할 수 있는 두 개의 새로운 데이터셋, Percept-ToMi와 Percept-FANToM을 도입했습니다.

- **Technical Details**: Percept-ToMi와 Percept-FANToM은 기존의 ToM 벤치마크인 ToMi와 FANToM에서 캐릭터의 지각 정보(annotation)를 추가하여 구축되었습니다. 이를 통해 LLM이 각각의 장면에서 누가 무엇을 보았는지를 추론하는 능력(지각 추론)과 이 지각 정보를 기반으로 캐릭터의 신념을 추론하는 능력(지각-신념 추론)을 평가합니다.

- **Performance Highlights**: 연구 결과, 대형 언어 모델들은 지각 추론에서는 비교적 높은 성능을 보였지만, 지각-신념 추론에서는 성능이 떨어지는 것으로 나타났습니다. 특히, 잘못된 신념(false belief) 시나리오에서 성능이 저조했습니다. 이를 개선하기 위해 PercepToM이라는 새로운 프레임워크가 도입되었으며, 이 방법은 특히 잘못된 신념 시나리오에서 LLM의 성능을 상당히 향상시켰습니다.



### LLaMAX: Scaling Linguistic Horizons of LLM by Enhancing Translation Capabilities Beyond 100 Languages (https://arxiv.org/abs/2407.05975)
- **What's New**: 최근 발표된 연구에서는 고자원 언어(High-Resource Languages) 간 번역에서 탁월한 성능을 보이는 대형 언어 모델(LLMs)이 저자원 언어(Low-Resource Languages)에서는 번역 성능이 저조하다는 문제를 해결하고자, LLaMA 시리즈 모델을 대상으로 35,000 A100-SXM4-80GB GPU 시간을 투자하여 광범위한 다국어 지속 학습(Continual Pre-training)을 수행했습니다. 이를 통해 100개 이상의 언어에 대한 번역 지원을 가능하게 한 LLaMAX 모델이 개발되었습니다.

- **Technical Details**: 이 연구에서는 어휘 확장(Vocabulary Expansion)과 데이터 증강(Data Augmentation) 등의 학습 전략을 통해 모델의 성능을 높였습니다. 저자원 언어에서의 번역 성능 향상을 위해 단일 언어 및 병렬 데이터를 활용한 지속 학습을 수행하였고, 특히 데이터가 부족한 언어에 대해서는 다국어 사전을 사용하여 pseudo-parallel dataset을 생성했습니다. 이러한 전체 프로세스는 기존 LLaMA 모델의 어휘를 최대한 활용하여 비용 효율성을 유지하면서도 성능을 높이는 방향으로 설계되었습니다.

- **Performance Highlights**: LLaMAX는 기존의 오픈 소스 LLMs보다 10 spBLEU 포인트 이상 높은 번역 성능을 기록했으며, 특히 저자원 언어 번역에서 10 spBLEU 포인트 이상의 향상을 이뤄냈습니다. 뿐만 아니라, Flores-101 벤치마크에서 M2M-100-12B와 동등한 성능을 보였으며, 일반적인 테스크 성능도 저하되지 않고 탁월한 성능을 발휘했습니다. 특히 X-CSQA, XNLI, MGSM 테스크에서 평균 5포인트 이상의 성능 향상을 나타냈습니다.



### Towards Optimizing and Evaluating a Retrieval Augmented QA Chatbot using LLMs with Human in the Loop (https://arxiv.org/abs/2407.05925)
- **What's New**: SAP SE와 협력하여 개발한 HR 지원 챗봇을 통해 직원 문의를 효율적으로 처리할 수 있는 효과적인 도구를 만들었습니다. 이 챗봇은 데이터 수집, 프롬프트 최적화, 출력 평가 등 여러 개발 주기에 인간 전문가를 포함하여 LLM 기반 챗봇의 응답 품질을 개선하고 대안적인 검색 방법을 탐색함으로써 더욱 효율적이고 유연한 도구로 발전시켰습니다.

- **Technical Details**: 본 연구는 Retrieval Augmented Generation (RAG) 접근 방식을 채택하여 모델이 보다 근거 있는 답변을 생성하도록 했으며, retrieval(검색) 및 모델 프롬프트 최적화 등 여러 모듈을 최적화하였습니다. Dense Passage Retriever (DPR)와 OpenAI Retriever를 사용하여 유저 쿼리를 기반으로 가장 관련성 높은 HR 관련 기사를 검색하였습니다. 다양한 Query Transformation 기법도 도입되어 사용자 쿼리를 다르게 표현하여 검색 정확도를 높였습니다. GPT-4 모델을 통한 챗봇 응답 생성 과정에서는 프롬프트 엔지니어링을 반복적으로 수행하여 최종 프롬프트를 도출했습니다.

- **Performance Highlights**: 실험 결과에 따르면, GPT-4 모델이 다른 모델보다 뛰어난 성능을 보였으며 데이터 불일치를 내부 추론 능력으로 극복할 수 있음을 확인했습니다. 또한, G-Eval 및 Prometheus와 같은 reference-free 평가 지표가 인간 평가와 유사한 신뢰성을 보인다는 점을 전문가 분석을 통해 밝혀냈습니다.



### Generation and De-Identification of Indian Clinical Discharge Summaries using LLMs (https://arxiv.org/abs/2407.05887)
Comments:
          Accepted at BioNLP Workshop at ACL 2024; 21 pages (9 pages main content)

- **What's New**: 인도 헬스케어 기관의 환자 퇴원 요약을 이용한 새로운 데이터셋(Indian Clinical Discharge Summaries, ICDSR)을 도입하고 이를 통해 PI-RoBERTa 모델을 평가한 연구입니다. 이 모델은 비-인도 임상 요약으로 미세 조정된 모델입니다. 결과적으로, 상호 기관 성능이 낮음을 보여줍니다. 또한 상용화된 오프 더 셀프(off-the-shelf) 임상 데이터 비식별화 시스템도 실험하여 유사한 경향을 확인했습니다.

- **Technical Details**: 소량의 인도 환자 퇴원 요약 데이터셋을 이용해 언어 모델을 훈련 및 평가하였습니다. 공개 사용 가능한 비-인도 데이터셋으로 미세 조정된 de-identification 알고리즘은 상호 기관 일반화 성능이 부족함을 보였습니다. 데이터 부족 문제를 해결하기 위해 대규모 언어 모델(LLM)을 이용하여 합성 임상 리포트를 생성하고 이를 비식별화 모델 훈련에 사용하였습니다. 주요 데이터셋은 인도 Sanjay Gandhi Post Graduate Institute of Medical Sciences의 IRB 승인된 99개의 완전히 비식별화된 퇴원 요약입니다.

- **Performance Highlights**: 비식별화 시스템의 성능을 크게 개선한 합성 요약 생성 실험이 성공적이었습니다. LLM 기반 합성 데이터를 생성하고 이를 훈련에 사용함으로써 성능 향상을 달성했습니다. 모델 코드와 실험 결과는 GitHub에서 공개됩니다: https://github.com/Exploration-Lab/llm-for-clinical-report-generation-deidentification



### KG-FPQ: Evaluating Factuality Hallucination in LLMs with Knowledge Graph-based False Premise Questions (https://arxiv.org/abs/2407.05868)
- **What's New**: 새로운 논문에서는 대규모 언어 모델(Large Language Models, LLMs)이 잘못된 전제문제(False Premise Questions, FPQs)에 의해 오도되어 사실 왜곡(factuality hallucination)을 일으킬 수 있음을 밝히고 있습니다. 이를 해결하기 위해 연구자들은 지식 그래프(Knowledge Graphs, KGs)를 기반으로 한 FPQ 생성 자동화 파이프라인을 소개합니다. 이 파이프라인은 대규모로 FPQ를 생성할 수 있어 기존의 수동 구축 방식보다 확장성이 뛰어나다는 장점이 있습니다.

- **Technical Details**: 본 연구에서는 먼저 지식 그래프로부터 참 트리플릿(true triplet)들을 추출한 뒤, 이를 편집하여 거짓 트리플릿(false triplet)으로 변형합니다. 이후 최첨단 GPT 모델(GPT-3.5, GPT-4)의 성능을 활용해 의미적으로 풍부한 FPQs를 생성합니다. 구체적으로는, KG111KoPL(https://kopl.xlore.cn)에서 추출한 트리플릿을 편집하여 다양한 수준의 혼란성을 가진 FPQs를 만듭니다. 논문의 예시에 따라, 원래의 참 트리플릿 <John Lennon, place of death, New York City>를 <John Lennon, place of death, Liverpool>로 편집하여 1-홉 이웃인 Liverpool을 거짓 트리플릿으로 사용합니다.

- **Performance Highlights**: KG-FPQ를 사용한 평가 결과, LLM들이 거짓 전제문제에 더 자주 오도된다는 점을 발견했습니다. (1) 편집된 객체가 주제와 가까운 거리일수록, 또는 원래 객체와 더 강한 연관이 있을수록 FPQs가 더 혼란스러울 수 있습니다. (2) 생성(task)에 비해 구별(discriminative) 작업에서 LLM들이 더 잘 수행하지만, 여전히 FPQs에 약함을 보였습니다. (3) 지식 영역별로 LLM의 지식 숙련도는 달랐으며, 지식 숙련도가 높아졌다고 해서 FPQ에 대한 저항 능력이 향상된 것은 아니라는 점을 발견했습니다.



### An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models (https://arxiv.org/abs/2407.05841)
Comments:
          Under review

- **What's New**: 이번 연구에서는 기존의 언어 모델 내 어휘 확장을 위한 초기화 방법론인 Constrained Word2Vec (CW2V)를 새롭게 제안합니다. CW2V는 cross-lingual embeddings 없이 초기화를 수행하는 간단한 방법론으로, 기존의 복잡한 초기화 방법들과 비교해 동등하거나 더 나은 성능을 보여줍니다.

- **Technical Details**: 기존 모델의 tokenizer가 새로운 언어를 충분히 포괄하지 못하는 문제를 해결하기 위해, 새로운 어휘를 위한 초기화가 필요합니다. CW2V는 기존 임베딩들의 convex hull 내에서 초기화하는 접근법으로, cross-lingual embeddings 없이도 효율적인 초기화를 가능하게 합니다. RoBERTa와 LLaMA 2 모델을 대상으로 실험을 진행하였고, 4개의 언어와 5개의 작업에 대해 다양한 초기화 방법들과 비교 분석을 수행했습니다.

- **Performance Highlights**: CW2V는 다른 선진화된 기법과 비교해 동등하거나 더 나은 성능을 보였습니다. 특히, 다변량 초기화(multivariate initialization) 등 간단한 초기화 방법들도 복잡한 방법들과 대등한 성능을 보여줌으로써, 모델 확장에서 간단한 초기화 방법론도 효과적일 수 있음을 입증했습니다.



### Large Language Models for Judicial Entity Extraction: A Comparative Study (https://arxiv.org/abs/2407.05786)
- **What's New**: 최신 연구에서는 대형 언어 모델(Large Language Model, LLM)이 법률 문서에서 도메인 특화 엔티티(domain-specific entity)를 인식하는 데 있어 높은 성능을 보여줌을 입증했습니다. 특히, 법률 분야의 엔티티 인식, 예를 들어 법원, 청구인, 판사, 변호사, 응답자 등을 식별하는 데 집중했습니다. 주요 LLM 아키텍처인 Large Language Model Meta AI 3, Mistral, Gemma의 효과를 인도 사법 텍스트를 대상으로 평가했습니다.

- **Technical Details**: 이번 연구는 최근에 개발된 LLaMA 3 (Large Language Model Meta AI 3), Mistral, Gemma 등의 최첨단 LLM을 사용하여 법률 문서 내에서 도메인 특화 엔티티 인식 작업의 성능을 평가했습니다. 이러한 모델들은 법률 문서의 복잡한 맥락과 전문 용어를 이해하고 처리하는 데 탁월한 능력을 보였습니다. 특히, Mistral과 Gemma 모델이 높은 균형된 정밀도 및 재현율을 보여주며 법률 텍스트에서 중요한 엔티티를 정확하게 식별했습니다.

- **Performance Highlights**: 실험 결과, Mistral과 Gemma 모델이 가장 뛰어난 성능을 보여주었으며, 법률 문서의 엔티티를 인식하는 데 있어서 중요한 정밀도와 재현율을 균형 있게 유지했습니다. 이 모델들은 정교한 데이터 출력을 통하여 보다 효율적이고 신속한 정보 검색 및 관리가 가능하게 하여 학술 연구를 가속화할 수 있음을 입증했습니다.



### When is the consistent prediction likely to be a correct prediction? (https://arxiv.org/abs/2407.05778)
- **What's New**: 이번 논문에서는 Wang et al. (2023)의 자기 일관성(self-consistency) 개념을 재검토하고, 이를 좀 더 정교하게 수정하는 연구결과를 발표했습니다. 기존 연구는 일관된 답변이 정확할 가능성이 높다고 주장했지만, 본 연구는 더 긴 추론 텍스트를 통해 일관된 답변이 도출될 때 더 정확할 가능성이 높다는 사실을 발견했습니다.

- **Technical Details**: 본 연구에서는 LLM(Large Language Models)이 특별한 프롬프트 없이도 긴 답변을 생성할 때 자동으로 연쇄 사고(Chain-of-Thought, CoT) 스타일의 추론을 생성할 수 있음을 보여주었습니다. Mixtral-8x7B 모델을 여러 번 샘플링하여 긴 답변을 고려한 결과, GSM8K와 MultiArith 데이터셋에서 zero-shot CoT 프롬프트 성능의 86%를 달성했습니다. 실험 설정에서는 Mixtral-8x7B 및 Llama-2 70B 모델을 사용하고, Reasoning 텍스트를 생성한 후 답변을 추출하는 두 단계로 이루어진 프롬프트 파이프라인을 활용했습니다.

- **Performance Highlights**: 결과적으로, 긴 추론 텍스트를 고려하여 일관된 답변을 선택하면 성능이 크게 향상됨을 확인했습니다. 특히, GSM8K와 MultiArith와 같은 수학적 추론 데이터셋에서 상위 86% 이상의 성능을 달성했습니다. 또한, 여러 번 샘플링을 통해 최소 일관성 임계값(minimum consistency threshold)을 적용한 결과, 길고 자세한 추론 텍스트가 성능 향상에 유리하다는 점을 입증했습니다.



### Large Language Models Understand Layouts (https://arxiv.org/abs/2407.05750)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 텍스트 이해 능력을 넘어서 공간 마커로 표시된 텍스트 레이아웃도 처리할 수 있음을 보여줍니다. 특히, LLMs는 공간적 인식 및 추론을 요구하는 질문에 정확하게 답할 수 있으며, 원래 데이터의 공간 마커를 제외하면 성능이 크게 저하됨을 확인했습니다. 다양한 레이아웃 민감한 데이터셋에 대해 GPT-3.5, Baichuan2, Llama2, ChatGLM3 모델과의 실험 결과를 통해 이를 분석했습니다.

- **Technical Details**: 이 연구는 TextLayoutQA라는 데이터셋을 만들어 LLMs의 텍스트 레이아웃 이해 능력을 평가했습니다. 실험 결과, 텍스트 레이아웃 정보를 포함하면 모델 성능이 8~25% 향상됨을 확인했습니다. LLMs의 텍스트 레이아웃 이해 능력이 주로 사전 훈련 동안 도입되며, 지시 조정(instruction-tuning) 단계에서 더욱 향상됨을 발견했습니다. 또한, 저비용 자동 생성 데이터와 새로운 텍스트 게임 접근법으로 레이아웃 이해 능력을 더욱 강화할 수 있음을 보여주었습니다.

- **Performance Highlights**: 텍스트 레이아웃 이해 능력은 Visual Question Answering (VQA) 시스템 구축에 유익하며, 레이아웃 정보가 있는 텍스트가 없는 경우에 비해 모델 성능을 약 8~25% 향상시켰습니다. 이 연구는 레이아웃 정보가 풍부한 데이터셋, 예를 들어 코드 및 테이블 데이터의 필요성을 강조하며, 다양한 데이터셋의 영향과 제약을 상세히 설명합니다.



### Do Multilingual Large Language Models Mitigate Stereotype Bias? (https://arxiv.org/abs/2407.05740)
Comments:
          19 pages, 8 figures

- **What's New**: 이 연구는 다중 언어(Languages)로 훈련된 대형 언어 모델들(multilingual LLMs)이 단일 언어(monolingual) 모델들에 비해 편향(bias)을 줄이는 효과가 더 뛰어나다는 점을 밝히는 데 중점을 두고 있습니다. 이를 위해, 동일한 크기(2.6B 파라미터)와 아키텍처를 가진 여섯 개의 LLM들을 체계적으로 훈련시켰습니다: 영어, 독일어, 프랑스어, 이탈리아어, 스페인어로 훈련된 다섯 개의 단일 언어 모델과 이들 언어를 고르게 분포시켜 훈련된 하나의 다중 언어 모델입니다. 모든 데이터는 공개적으로 사용 가능한 데이터를 활용하였습니다.

- **Technical Details**: 신뢰성 있는 평가를 위해, 표준 편향 벤치마크들은 자동으로 다섯 개의 목표 언어로 번역되었으며, 인간 주석자들이 번역 품질과 편향 보존을 검증하였습니다. 모델 훈련은 동일한 양의 훈련 데이터, 모델 아키텍처 및 크기를 기반으로 수행되었습니다.

- **Performance Highlights**: 결과는 다중 언어 훈련이 편향을 효과적으로 완화(mitigate)한다는 것을 일관되게 보여주었습니다. 더 나아가, 다중 언어 모델들이 동일한 조건의 단일 언어 모델들에 비해 편향이 더 낮을 뿐만 아니라 예측 정확성(prediction accuracy)에서도 우수한 성능을 달성했습니다.



### Empirical Study of Symmetrical Reasoning in Conversational Chatbots (https://arxiv.org/abs/2407.05734)
Comments:
          Accepted in Future Technology Conference (FTC) 2024

- **What's New**: 이번 연구에서는 대화형 챗봇이 대규모 언어 모델(LLMs)로 구동되어, 전통적으로 인간 고유의 인지 언어 기능으로 여겨지던 술어 대칭(predication symmetry)을 이해하고 특징짓는 능력을 탐구합니다. 새로운 작업을 재훈련 없이 프롬프트에서 학습할 수 있게 하는 'in-context learning(ICL)' 패러다임을 활용하여 다섯 가지 챗봇의 대칭적 추론 능력을 평가했습니다.

- **Technical Details**: 이 연구에서는 대칭 추론 문장(Symmetry Inference Sentence, SIS) 데이터셋을 사용하여 챗봇 ChatGPT 4, Huggingface chat AI, Microsoft's Copilot AI, LLaMA (Perplexity를 통해) 및 Gemini Advanced가 술어 대칭을 이해하는 능력을 인간 평가와 비교하여 분석했습니다. ICL을 이용하여 챗봇들이 재훈련 없이 새로운 작업을 프롬프트만으로 학습할 수 있도록 했습니다.

- **Performance Highlights**: 실험 결과, 챗봇들의 대칭적 추론 능력에는 차이가 있었습니다. 일부 챗봇은 인간과 유사한 추론 능력을 보였으며, 특히 Gemini Advanced는 인간 점수와 상관관계가 0.85에 달하며, 각 대칭 평가에 대해 근거 있는 설명을 제공했습니다. 이 연구는 LLM이 대칭 추론과 같은 복잡한 인지 과정을 반영할 수 있는 잠재력과 한계를 강조합니다.



### Is GPT-4 Alone Sufficient for Automated Essay Scoring?: A Comparative Judgment Approach Based on Rater Cognition (https://arxiv.org/abs/2407.05733)
Comments:
          16 pages, 3 figures, Learning @ Scale 2024

- **What's New**: 대형 언어 모델(LLMs)이 자동 에세이 평가(AES) 분야에서 유망성을 보여주고 있지만, 무학습(zero-shot)과 소학습(few-shot) 설정에서는 최신 모델 및 인간 평가자에 비해 성과가 저조한 경우가 많습니다. 본 연구는 LLMs와 비교 판단(Comparative Judgment, CJ)을 결합한 새로운 접근법을 제안하여, 두 에세이를 선택하는 무학습 프롬프트를 사용해 AES에서 전통적인 루브릭 기반 채점을 능가하는 성과를 보였습니다.

- **Technical Details**: 이번 연구는 ASAP 데이터셋의 에세이 세트 7과 8을 사용하여 수행되었습니다. 이 데이터셋은 각각 4개와 6개의 평가 항목에 따른 분석적 채점 점수를 제공하며, 이를 통해 LLMs의 성능을 비교 판단(CJ) 방식으로 평가합니다. 일곱 번째 프롬프트 세트는 7학년 학생이 작성한 1,569개의 에세이를 포함하며, 여덟 번째 프롬프트 세트는 10학년 학생이 작성한 723개의 에세이를 포함합니다. 에세이는 각각 평균 250단어와 650단어의 길이로 작성되었습니다.

- **Performance Highlights**: 연구 결과, 비교 판단 방법이 전통적인 루브릭 기반 채점 방법에 비해 더 높은 신뢰도와 점수 일관성을 제공하는 것으로 나타났습니다. GPT-3.5 및 GPT-4 모델 모두 비교 판단 방법을 통해 성능 향상을 보였으며, 이는 인간 평가자가 경험하는 인지적 요구를 줄이는 데에도 도움이 되었습니다.



### PsycoLLM: Enhancing LLM for Psychological Understanding and Evaluation (https://arxiv.org/abs/2407.05721)
Comments:
          work in progress

- **What's New**: 최근 정신 건강이 중요한 이슈로 떠오른 가운데, 대형 언어 모델(Large Language Model, LLM)이 이를 해결하는 데 효과적인 기술로 간주되고 있습니다. 본 논문에서는 PsycoLLM이라는 특화된 심리학 대형 언어 모델을 제안합니다. 이 모델은 단일 턴 질의응답(QA), 다중 턴 대화, 지식기반 QA를 포함한 고품질 심리학 데이터셋을 통해 훈련되었습니다.

- **Technical Details**: PsycoLLM은 고품질 심리학 데이터셋을 기반으로 훈련되었습니다. 이 데이터셋은 단일 턴 QA, 다중 턴 대화, 지식기반 QA로 구분됩니다. 다중 턴 대화 데이터는 KimiChat를 이용해 생성되었으며, 사용자와 심리 전문가 간의 실제 대화를 반영하도록 설계되었습니다. 또한, 한국의 권위 있는 심리 상담 시험을 기반으로 한 포괄적인 벤치마크를 개발하여 모델의 성능을 평가했습니다. 벤치마크는 전문 윤리, 이론적 숙련도, 사례 분석 세 가지 구성 요소로 나뉩니다.

- **Performance Highlights**: PsycoLLM은 다수의 LLM보다 우수한 성능을 입증하였습니다. 특히, MCQ에서 60% 이상의 정확도를 달성하며 심리적 지원 제공의 잠재력을 보여주었습니다. 또한, 전문 윤리에 대한 숙련도가 심리 이론보다 더 뛰어나다는 점이 확인되었습니다.



### A Factuality and Diversity Reconciled Decoding Method for Knowledge-Grounded Dialogue Generation (https://arxiv.org/abs/2407.05718)
- **What's New**: 최근 발표된 연구에서 DoGe (Dynamic source-Grounded decoding)라는 새로운 방법을 소개했습니다. 이 방법은 기존의 대화 생성 모델이 지나치게 사실적(factual)이어야 한다는 문제를 해결하면서 창의적이고 다양한 응답을 생성할 수 있도록 설계되었습니다. 이 방법은 모델의 내부 파라미터 지식과 외부 소스 지식을 모델의 사실적 자신감에 따라 동적으로 전환하여 활용합니다.

- **Technical Details**: DoGe는 내부 파라미터 지식과 외부 지식을 혼합적으로 사용하여 대화 응답을 생성합니다. 먼저 모델의 사실적 확신도를 기준으로 두 가지 확률 분포를 얻는 방식입니다. 확신도가 높으면 외부 지식을 마스킹(masking)하여 응답의 다양성을 높이고, 확신도가 낮으면 외부 지식을 드러내어(faithfulness-augmented deterministic decoding) 정확한 답변을 유도합니다. 이를 통해 응답의 정확성과 다양성을 동시에 만족시킬 수 있습니다.

- **Performance Highlights**: DoGe는 세 가지 널리 사용되는 데이터셋에 대한 광범위한 실험에서 성과를 입증했습니다. 자동 및 인간 평가 모두에서 사실성과 다양성 면에서 기존의 다양한 디코딩 전략을 뛰어넘는 성능을 보여주었습니다.



### InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruc (https://arxiv.org/abs/2407.05700)
- **What's New**: 최근 오픈 소스 코드 LLMs 분야에서의 중요한 진전은 GPT-3.5 및 GPT-4와 같은 폐쇄형 LLMs를 이용한 지도 학습을 통해 뛰어난 코딩 능력을 입증한 것에 있습니다. 이번 논문은 폐쇄형 LLMs에 의존하지 않고, 자체 생성 데이터로부터 향상된 지도 학습 방법을 제안합니다. 저자들은 형식 언어(즉, 코드)를 비형식 언어(즉, 자연어)로 변환하는 것이 그 반대보다 더 쉽다는 점을 관찰하고, 이러한 관찰에 기반하여 INVERSE-INSTRUCT 방법을 제안했습니다.

- **Technical Details**: INVERSE-INSTRUCT는 코드 스니펫에서 명령어(instructions)를 요약하여 생성합니다. 주어진 코퍼스에 대해, 코드 LLM을 사용하여 코드 요약 및 자기 평가를 통해 추가적인 고품질 명령어를 생성하고, 생성된 데이터와 원본 데이터셋을 결합하여 기초 LLM을 다시 훈련합니다. 이를 통해 InverseCoder라는 시리즈의 코드 LLMs를 개발했습니다.

- **Performance Highlights**: InverseCoder 시리즈는 Python 텍스트-to-코드 생성, 다국어 코딩, 데이터 과학 코드 생성 등의 다양한 벤치마크에서 원본 코드 LLMs의 성능을 초과했습니다. 특히 InverseCoder-DS-6.7B는 HumanEval+에서 76.8%, MBPP+에서 69.0%, MultiPL-E에서 62.6%, DS-1000에서 44.2%의 결과를 기록하며, 완전 오픈 소스 모델(모델 및 데이터셋 모두) 중 최고 성능을 달성했습니다. 저자들은 InverseCoder의 매개 변수, 명령어 데이터셋, INVERSE-INSTRUCT 프롬프트, 훈련 코드베이스를 공개할 예정입니다.



### Pruning Large Language Models to Intra-module Low-rank Architecture with Transitional Activations (https://arxiv.org/abs/2407.05690)
Comments:
          Findings of ACL 2024

- **What's New**: TransAct는 대형 언어 모델(LLM)의 구조적 가지치기를 위한 새로운 접근 방식을 제안합니다. 이 방법은 Transformer 아키텍처의 전이 활성화를 줄이면서 중요한 모듈 간 활성화를 유지하는데 중점을 둡니다. 이는 특히 리소스가 제한된 엣지 디바이스에서 LLM을 실행할 수 있도록 설계되었습니다.

- **Technical Details**: TransAct는 다중 머리 주의(Multi-Head Attention, MHA)와 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 모듈 내의 전이 활성화를 줄입니다. 이를 통해 저내적 차원(low intrinsic dimension)의 구조로 가지치기되며, 모델 가중치와 KV(Cache) 메모리를 대폭 줄일 수 있습니다. 이러한 가지치기는 활성화 가이드 하에서 반복적으로 이루어지며, 중요한 활성화는 그대로 보존합니다.

- **Performance Highlights**: LLaMA 모델을 기반으로 한 실험 결과, TransAct는 높은 압축 비율 달성 시 효율성과 성능 면에서 최적의 결과를 보였습니다. 추가적인 소거 연구를 통해 MHA와 MLP 모듈의 중복성을 분석하여 보다 컴팩트한 모델 디자인에 대한 통찰력을 제공하였습니다.



### Retrieved In-Context Principles from Previous Mistakes (https://arxiv.org/abs/2407.05682)
- **What's New**: 최근 발전한 대형 언어 모델(LLMs)을 더욱 효율적으로 활용하기 위해, 우리는 Retrieved In-Context Principles(RICP)라는 새로운 교사-학생 프레임워크를 제안합니다. 이 프레임워크는 학생 모델이 범한 실수를 기반으로 교사 모델이 고급 이유와 통찰을 생성하여 유사한 실수를 방지하는 데 초점을 맞추고 있습니다. 이에 따라, RICP는 맞춤형 피드백과 포괄적인 오류 커버리지를 제공하여 모델 성능을 향상시킵니다.

- **Technical Details**: RICP는 세 가지 주요 단계를 포함합니다: 1) '통찰 생성' 단계에서 학생 모델의 실수를 수집하고, 교사 모델이 이러한 실수를 분석하여 고급 이유와 구체적인 통찰을 생성합니다. 2) '원칙 수립' 단계에서는 실수의 근본적인 이유를 기반으로 실수를 클러스터링하고, 각 클러스터에서 질문의 의미를 기반으로 가장 유사한 실수를 찾아 질문 수준의 원칙을 형성합니다. 3) '원칙 활용' 단계에서는 이러한 원칙을 학생 모델의 프롬프트에 통합하여 질문-응답 능력을 향상시킵니다.

- **Performance Highlights**: RICP는 세 가지 주요 장점을 가지고 있습니다: (1) 각 질문에 대한 가장 관련성이 높은 통찰을 제공하여, 원칙의 정확성과 커스터마이제이션을 크게 향상시킵니다. (2) 오류 패턴의 클러스터링을 통해 원칙의 오류 커버리지를 넓히고, 통찰 클러스터링을 통해 질문 수준의 원칙 중복을 줄여 효율성을 높입니다. (3) RICP는 기존의 프롬프트 방법과 직교하여 다양한 프롬프트 전략에서 성능을 높일 수 있습니다. 이는 교사 모델의 추가 개입 없이 추론 중에 낮은 계산 오버헤드를 유지할 수 있습니다. 실험 결과, RICP는 여러 추론 벤치마크에서 성능을 크게 향상시킬 수 있음을 확인했습니다.



### New Directions in Text Classification Research: Maximizing The Performance of Sentiment Classification from Limited Data (https://arxiv.org/abs/2407.05627)
Comments:
          9 pages, in Indonesian language. intro to a shared task in sentiment classification

- **What's New**: 이번 논문에서는 Kaesang Pangarep의 PSI(Chairman of PSI) 임명 이슈에 대한 감정 분석 문제를 다루고 있습니다. 특히, 300에서 600 샘플 정도의 제한된 학습 데이터를 사용한 텍스트 분류 문제를 다룹니다. 이 논문은 감정 분석에서 빠른 속도와 높은 정확도라는 이해관계자의 요구를 만족시키기 위해 외부 데이터를 활용한 집계 및 증강을 제안합니다.

- **Technical Details**: 본 연구는 텍스트 분류를 위해 제공된 벤치마크 데이터셋을 사용합니다. 추가로 Covid Vaccination 감정 데이터셋과 오픈 토픽(Open Topic)을 포함한 두 가지 외부 데이터를 활용합니다. 공식적인 평가 지표로는 F1-score가 사용되며, 이는 긍정, 부정, 중립이라는 세 개의 클래스에서 precision과 recall을 균형있게 평가합니다. 초기 기준 점수와 최적화된 점수 모두 SVM(Support Vector Machine) 방법을 사용하여 계산됩니다.

- **Performance Highlights**: 기준 점수와 최적화된 점수 모두 SVM 방법으로 계산되었으며, 기준 F1-score는 40.83%, 최적화된 F1-score는 51.28%를 달성했습니다.



### Open-world Multi-label Text Classification with Extremely Weak Supervision (https://arxiv.org/abs/2407.05609)
Comments:
          Preprint

- **What's New**: 이 논문은 사용자가 분류 목적을 간단히 설명하는 것만으로, 라벨이나 실제 라벨 공간 없이, 열린 세계 멀티라벨 텍스트 분류(open-world multi-label text classification)를 수행하는 새로운 방법인 X-MLClass를 제안합니다. 특히 기존의 단일 라벨 XWS(Extremely Weak Supervision) 방법이 멀티라벨로 쉽게 적응할 수 없는 문제를 해결하고자 합니다.

- **Technical Details**: X-MLClass는 대형 언어 모델(LLM, Llama-2-13b-chat)을 활용하여 문서의 주요 키프레이즈(keyphrases)를 추출하고, 이를 클러스터링하여 초기 라벨 공간(label space)을 만듭니다. 이 과정에서 제로샷 멀티라벨 분류기(zero-shot multi-label classifier)를 사용하여 문서의 도미넌트 키프레이즈를 반복적으로 검토하고 장기 꼬리(long-tail)의 라벨을 추가하게 됩니다. 이를 통해 포괄적인 라벨 공간을 구축하고 멀티라벨 분류기를 만듭니다.

- **Performance Highlights**: X-MLClass는 다양한 데이터셋에서 실제 라벨 공간 커버리지에서 큰 향상을 보여주었습니다. 예를 들어, AAPD 데이터셋에서는 주제 모델링과 키워드 추출 방법보다 40% 더 높은 성능을 나타냈습니다. 또한, X-MLClass는 엔드 투 엔드 멀티라벨 분류 정확도에서 최고 성능을 기록했습니다.



### Generative Debunking of Climate Misinformation (https://arxiv.org/abs/2407.05599)
Comments:
          Accepter to ClimateNLP 2024 workshop at ACL 2024

- **What's New**: 새로운 연구는 기후 변화와 관련된 허위 정보를 자동으로 탐지 및 수정하는 대규모 언어 모델(Large Language Models, LLMs)의 개발을 문서화했습니다. 이 모델은 기후 신화를 '사실-신화-논리적 오류-사실(fact-myth-fallacy-fact)' 구조로 반박하는 '진실 샌드위치(truth sandwich)' 형식을 따르도록 설계되었습니다.

- **Technical Details**: 이 연구는 기후 신화를 입력으로 받아 그에 알맞은 반박문을 생성하는 LLM을 개발하는 과정에서 특정한 전략을 적용했습니다. 구체적으로, 기후 모순 주장 분류와 논리적 오류 탐지를 포함한 프롬프트 전략(prompting strategy)을 사용했습니다. Mixtral, Palm2와 같은 오픈 소스 및 GPT-4와 같은 독점 LLM를 각각 서로 다른 복잡도의 프롬프트 전략과 결합해 실험을 진행했습니다. 각 프롬프트는 단순한 것부터, 각 단계별로 개별적으로 요청하는 것까지 다양했습니다.

- **Performance Highlights**: 실험 결과, GPT-4와 Mixtral가 구조화된 프롬프트와 함께 사용될 때 유망한 성능을 보였습니다. 그러나, 최신 LLM에서도 생성된 반박문에서 사실성과 관련성 부족이 주요한 단점으로 확인되었습니다. 이 연구는 고품질의 진실 샌드위치 반박 데이터셋과 소스 코드, 그리고 반박 시스템의 데모를 공개하며, 향후 연구를 위한 과제를 탐색했습니다.



### LLMBox: A Comprehensive Library for Large Language Models (https://arxiv.org/abs/2407.05563)
Comments:
          Accepted by ACL 2024 Demo

- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)의 연구를 촉진하기 위해 종합적이고 통일된 라이브러리 LLMBox를 소개합니다. 이 라이브러리는 다양한 훈련 전략의 유연한 구현을 지원하는 통합 데이터 인터페이스(unified data interface), 광범위한 작업, 데이터셋 및 모델들을 포괄하는 종합적인 평가(comprehensive evaluation), 그리고 사용 친화성과 효율성을 고려한 실용적인 디자인을 특징으로 합니다.

- **Technical Details**: LLMBox는 통일된 데이터 인터페이스를 통해 다양한 형식의 훈련 데이터를 캡슐화하고 다이내믹 믹스 비율(dynamic mixture proportion) 같은 전략을 지원하며, 주요 훈련 방법론(parameter-efficient tuning, alignment tuning 등)을 광범위하게 지원합니다. 또한, 포괄적인 평가를 위해 18개의 다운스트림 작업과 56개의 데이터셋을 지원하고, 인간 정렬(human alignment), 환각 탐지(hallucination detection), 지시 사항 준수(instruction following) 등 고급 기능 검사를 지원합니다. LLMBox는 사용자 친화적인 간편한 파이프라인을 제공하며, GPU 계산기를 통해 필요한 최소 GPU 리소스를 계산할 수 있게 지원합니다.

- **Performance Highlights**: LLMBox는 단일 A800 GPU에서 전체 MMLU 벤치마크의 추론을 6분 내에 완료하고, 8개 A800 GPU에서 52K 개체의 지시 조정을 10분 내에 완료할 수 있습니다. LLMBox는 LoRA, QLoRA, DeepSpeed, 그리고 packing 전략을 구현하여 제한된 컴퓨팅 리소스에서도 효율적으로 LLM 훈련을 지원합니다. 또한, 'GPU 메모리 계산기'를 통해 모델 훈련에 필요한 하드웨어 요구 사항을 정확하게 예측할 수 있습니다.



### Faux Polyglot: A Study on Information Disparity in Multilingual Large Language Models (https://arxiv.org/abs/2407.05502)
- **What's New**: 최근 연구에서는 Retrieval Augmented Generation (RAG) 기법을 활용한 대형 언어 모델(LLMs)의 다언어 정보 검색에서의 언어적 편향을 조사했습니다. 연구 결과, LLMs는 쿼리 언어와 동일한 언어로 된 정보를 선호하는 경향이 있으며, 특히 정보가 부족한 상황에서는 고자원 언어로 작성된 문서를 선호한다고 밝혔습니다. 이러한 경향은 사실 기반 및 의견 기반 쿼리 모두에서 나타났습니다.

- **Technical Details**: 이 연구에서는 5개의 언어(영어, 힌디어, 독일어, 아랍어, 중국어)로 구성된 170개의 문서를 포함한 다언어 합성 데이터셋을 생성하였습니다. 연구팀은 Retrieval Augmented Generation (RAG) 방식에서 언어 모델의 정보 선호도를 연구했으며, RAG 방식에서는 정보 검색과 생성 두 단계를 거칩니다. 검색 단계에서는 문서 임베딩과 쿼리 임베딩의 코사인 유사도에 기반해 연관된 문서를 검색하고, 생성 단계에서는 이 문서들을 컨텍스트로 사용해 응답을 생성합니다.

- **Performance Highlights**: 실험 결과, 현재 다언어 LLMs는 쿼리 언어와 동일한 언어로 된 문서를 일관되게 선호하는 경향을 보였습니다. 이러한 선호도는 사실 기반 및 의견 기반 쿼리 모두에서 확인되었습니다. 쿼리 언어로 된 관련 문서가 없는 경우, LLM은 저자원 언어보다 고자원 언어로 된 문서를 선호합니다. 이는 다언어 LLM이 정보 접근성을 민주화하려는 목표에 반해 언어적 정보 격차를 강화할 가능성을 시사합니다.



### How Effective are State Space Models for Machine Translation? (https://arxiv.org/abs/2407.05489)
- **What's New**: 최근 논문에서는 NLP의 최신 아키텍처인 트랜스포머(transformer)와, 최근 대안으로 주목받고 있는 선형 재귀 모델(linear recurrent models)을 기계 번역(machine translation, MT) 영역에서 비교했습니다. 구체적으로는 트랜스포머와 RetNet, Mamba 및 주의 메커니즘(attention mechanisms)을 통합한 하이브리드 Mamba를 실험했습니다. Mamba는 문장 및 단락 수준 데이터셋에서 트랜스포머와 매우 경쟁력 있는 성능을 보였고, 특히 주의 메커니즘을 통합함으로써 번역 품질, 시퀀스 길이 추론 견고성 및 명명된 엔티티(nеamed entities) 회상 능력에서 개선을 보였습니다.

- **Technical Details**: 트랜스포머의 주의 메커니즘은 시퀀스 길이에 비례하여 이차적인 계산 비용을 유발합니다. 이와 대조적으로, 선형 재귀 모델은 효율적인 학습과 추론을 가능케 합니다. 이 논문은 대표적인 선형 재귀 모델인 RetNet과 Mamba, 그리고 주의 메커니즘을 통합한 하이브리드 Mamba를 트랜스포머와 비교했습니다. 실험에서는 사전 훈련된(pretrained) 모델과 긴 문맥의 데이터셋을 포함한 다양한 설정에서 평가를 진행했습니다.

- **Performance Highlights**: 문장 수준 실험에서는, Mamba가 트랜스포머와 견줄 만한 성능을 보였습니다. 단락 수준에서는 훈련 분포의 시퀀스 길이에 민감하였으나, 긴 시퀀스 길이로 분포를 이동하면 트랜스포머와의 성능 격차가 줄어들었습니다. 주의 메커니즘을 통합한 하이브리드 Mamba는 번역 품질, 시퀀스 길이 일반화, 명명된 엔티티 회상 능력 등에서 우수한 성능을 발휘했습니다.



### Just read twice: closing the recall gap for recurrent language models (https://arxiv.org/abs/2407.05483)
- **What's New**: 최신 연구는 Transformers와 경쟁하는 언어 모델 perplexity를 가진 고정 메모리 재귀 모델(Recurrent Large Language Models, RLLMs)의 급진적인 진전을 보여주고 있습니다. 특히 Mamba와 RWKV 같은 구조는 추론 시 일정한 메모리 양만을 사용합니다. 그러나 이러한 모델들은 길이 있는 컨텍스트에서 모든 정보를 기억하고 사용할 수 없어, 'In-Context Learning' (ICL) 품질이 약화됩니다.

- **Technical Details**: 이 연구는 정보가 RLLM에 표시되는 순서가 정보 선택의 난이도에 영향을 준다는 것을 관찰합니다. 이를 포멀하게 하기 위해, 이 연구는 정보 회상의 난이도가 통신 복잡성 이론에서 중요한 문제인 '집합 비공유 문제' (Set Disjointness, SD)의 난이도로 환원된다는 것을 보여줍니다. 이 문제는 입력된 집합이 비공유인지 결정해야 하는 스트리밍 알고리즘(예: 재귀 모델)을 요구합니다.

- **Performance Highlights**: 제안된 해결책으로는 'JRT-Prompt'와 'JRT-RNN'이 있습니다. JRT-Prompt는 컨텍스트가 프롬프트 내에서 여러 번 반복되도록 하여 다양한 데이터 순서를 모델에 보여주는 전략입니다. 이는 16개의 RLLMs과 6개의 ICL 과제를 평균할 때 11.0 ± 1.3 포인트의 향상을 제공하며, Generation Prefill에서는 FlashAttention-2보다 11.9배 높은 처리량을 제공합니다. JRT-RNN은 비-인과적 프리픽스-선형-어텐션(non-causal Prefix-Linear-Attention)을 사용하여 프롬프트를 처리하며, 이를 통해 360M 파라미터에서 Transformer 품질의 99%, 1.3B 파라미터에서 96%를 달성하고, FA2보다 19.2배 높은 처리량을 제공합니다.



### Biomedical Nested NER with Large Language Model and UMLS Heuristics (https://arxiv.org/abs/2407.05480)
Comments:
          Submitted to CEUR-WS for the BioNNE task of BioASQ Lab in Conference and Labs of the Evaluation Forum (CLEF) 2024 as a working note

- **What's New**: 저희는 BioNNE 영어 트랙에서 의생명학 텍스트로부터 8가지 유형의 중첩된 명명된 엔티티(nested named entities)를 추출하는 시스템을 개발했습니다. 이를 위해 대형 언어 모델인 Mixtral 8x7B instruct와 ScispaCy NER 모델을 사용했습니다. 이 시스템은 F1 점수 0.39 (검증 세트) 및 0.348 (테스트 세트)를 기록했습니다.

- **Technical Details**: 시스템은 Mixtral 8x7B instruct 모델과 ScispaCy biomedical NER 모델을 활용하여 논문에서 엔티티를 인식합니다. 그런 다음 통합 의료 언어 시스템(UMLS) 시맨틱 타입을 기반으로 맞춤형 휴리스틱을 사용하여 엔티티를 분류합니다. LLM(Prompt engineering) 및 UMLS API를 사용하여 최종적으로 특정 카테고리의 엔티티로 결정합니다.

- **Performance Highlights**: 저희의 모델은 BioNNE 영어 트랙 테스트 세트에서 F1 점수 0.348을 기록했으며, 검증 세트에서는 0.39의 F1 점수를 달성했습니다.



### Training Task Experts through Retrieval Based Distillation (https://arxiv.org/abs/2407.05463)
- **What's New**: ReBase 또는 Retrieval Based Distillation이라는 새로운 방법이 제안되었습니다. 이 방법은 온라인에서 데이터를 검색하고 이를 도메인 특화 데이터로 변환하여 다양한 소스에서 데이터를 수집함으로써 데이터 다양성을 크게 향상시킵니다. 특히, Chain-of-Thought 변환을 통해 LLM의 추론 능력을 작은 모델로 증류합니다.

- **Technical Details**: ReBase는 먼저 풍부한 온라인 소스에서 데이터를 검색한 후, 이를 사용자의 작업에 적합한 형태로 변환합니다. 데이터 저장소(datatore)를 구성하고, 사용자 제공 예제와 지시문을 사용하여 이 저장소에서 관련성 높은 데이터를 검색한 후, LLM을 이용해 이를 도메인 특화 데이터로 변환합니다. 마지막으로, 이 데이터를 사용해 모델을 훈련시킵니다. 이를 통해 SQuAD, MNLI, BigBench-Hard 등 다양한 벤치마크에서 성능을 검증했습니다.

- **Performance Highlights**: ReBase는 여러 벤치마크에서 기존 방법보다 성능 향상을 보여주었습니다. SQuAD에서는 7.8%, MNLI에서는 1.37%, BigBench-Hard에서는 1.94%의 성능 향상을 이루었습니다.



### SmurfCat at PAN 2024 TextDetox: Alignment of Multilingual Transformers for Text Detoxification (https://arxiv.org/abs/2407.05449)
- **What's New**: SmurfCat 팀이 PAN-2024 대회에서 다국어 텍스트 디톡시피케이션(multiple text detoxification) 문제를 해결했습니다. 이 팀은 데이터 증강(data augmentation)과 기계 번역(machine translation), 특별한 필터링 절차를 통해 다국어 평행 데이터셋(parallel dataset)을 수집하였고, 이를 이용해 mT0와 Aya 같은 다국어 시퀀스-투-시퀀스 모델(sequence-to-sequence models)을 미세조정(fine-tuning)하였습니다. 최종 모델에는 ORPO 정렬(alignment) 기법이 적용되었고, 37억 개의 파라미터만으로 우크라이나어에서 최첨단 성능을, 그 외 언어에서는 거의 최첨단 성능에 도달하는 결과를 얻었습니다.

- **Technical Details**: 대회에서 제공된 데이터셋 외에도 GoogleTranslator 모델을 사용하여 영어 데이터를 다른 언어로 번역함으로써 추가 데이터셋을 만들었습니다. 번역 후 LaBSE 모델을 사용하여 의미 유사성을 평가하고, XLM-R 독성 분류기(toxicity classifier)로 번역된 텍스트의 독성 여부를 판단하여 데이터 필터링을 수행했습니다. 최종적으로 74,900개의 샘플이 훈련에 사용되었습니다. mT0 모델 패밀리를 주로 사용하여 미세조정하였으며, 학습률(learning rate)은 1e-5, 글로벌 배치 사이즈(global batch size)는 8, 가중치 감소(weight decay)는 0.01로 설정했습니다. 모델 추론 단계에서 다양한 빔 검색(diverse beam search)을 사용하여 최고의 후보를 선택했습니다.

- **Performance Highlights**: 자동 평가에서 평균 점수 0.52로 1위를 달성했으며, 인간 평가에서 점수 0.74로 2위를 기록했습니다. 최종 모델인 mT0-XL은 ORPO 정렬을 통해 더 나은 성능을 보였으며, 특히 우크라이나어에서 최고 점수를 기록했습니다. 다국어 디톡시피케이션에서 다른 팀보다 우수한 성능을 입증했습니다.



### LTLBench: Towards Benchmarks for Evaluating Temporal Logic Reasoning in Large Language Models (https://arxiv.org/abs/2407.05434)
- **What's New**: 이 논문은 LLMs(Large Language Models)의 시간적 추론(Temporal Reasoning, TR) 능력을 평가하기 위해 새로운 데이터셋 구성 파이프라인을 제안합니다. 이 파이프라인은 무작위로 생성된 유향 그래프(random directed graph generation), LTL(Liner Temporal Logic) 공식, 그리고 NuSMV 모델 체커를 활용하여 구성되었습니다. 이를 바탕으로 LTLBench라는 2,000개의 TR 문제로 구성된 벤치마크 데이터셋도 생성하였습니다. 또한, 이벤트와 공식 연산자의 수가 TR 문제의 복잡성과 LLMs의 성능에 미치는 영향도 추가 실험을 통해 분석하였습니다.

- **Technical Details**: 이 연구에서 제안된 파이프라인은 무작위 유향 그래프 생성, LTL 공식 생성, NuSMV 코드 생성 및 자연어 생성의 네 가지 단계를 포함합니다. 무작위 유향 그래프는 주어진 수의 이벤트와 이들 간의 전환을 생성하며, LTL 공식은 그래프 내 이벤트를 기반으로 생성됩니다. 그런 다음, LTL 공식과 그래프 이벤트 정보를 NuSMV 코드로 변환하고, 이를 실행하여 TR 문제의 참-거짓(label)을 얻습니다. 마지막으로, 이벤트 정보와 LTL 공식을 자연어로 변환하여 TR 문제를 구성합니다.

- **Performance Highlights**: LTLBench 데이터셋을 사용한 실험 결과, LLMs는 기본적인 TR 문제에서는 어느 정도 가능성을 보여주었으나, 복잡한 TR 문제에서는 여전히 어려움을 겪고 있음을 밝혔습니다. 또한, 모델의 파라미터 크기(수백만 개의 매개변수를 가진 대형 모델과 소형 모델)에 따른 성능 차이도 관찰되었습니다. TR 문제의 복잡성은 이벤트와 연산자의 수가 증가할수록 높아지며, 이는 LLMs의 성능 저하로 이어졌습니다.



### iSign: A Benchmark for Indian Sign Language Processing (https://arxiv.org/abs/2407.05404)
Comments:
          Accepted at ACL 2024 Findings. 18 Pages (9 Pages + References + Appendix)

- **What's New**: 인도 수어(Indian Sign Language, ISL) 처리를 위한 새로운 벤치마크 iSign이 제안되었습니다. 이 연구는 인도 수어와 영어로 구성된 118,000개 이상의 비디오-문장/구절 쌍을 포함하는 가장 큰 데이터셋을 공개합니다. 또한, SignVideo2Text, SignPose2Text, Text2Pose, Word Prediction, Sign Semantics 등 여러 NLP 특정 작업을 제안하고 기본 모델을 바탕으로 벤치마크합니다.

- **Technical Details**: iSign 벤치마크는 주로 ISLRTC, ISH News, DEF 등의 YouTube 채널에서 데이터를 수집합니다. 각 비디오는 사전에 처리되고 문장 또는 구절 수준으로 분할됩니다. 이 데이터셋은 ISL 처리를 위한 다양한 작업들을 지원하며, 공통 벤치마크 작업으로는 SignVideo2Text 번역, SignPose2Text 번역, Text2Sign 번역, 기호/글로스 인식 등이 있습니다. 두 가지 추가 작업인 Sign Presence Detection과 Sign Semantic Similarity Prediction도 포함되어 있습니다.

- **Performance Highlights**: iSign을 통해 제공된 데이터셋, 작업 및 기본 모델은 연구 커뮤니티가 쉽게 접근할 수 있습니다. 이 데이터셋은 미국 수어(ASL)나 독일 수어(DGS)와 같은 다른 수어로는 이미 많은 주석이 달린 데이터가 있는 곳에서도 부족한 인도 수어(Indian Sign Language)를 위한 연구를 촉진할 것입니다. 자세한 언어적 통찰을 제공하여 인도 수어의 작동 방식을 이해하고, 비언어적 마커의 중요성, 공간 사용, 지문 자모 및 참조 연결, 역할 전환 등의 다양한 측면을 다룹니다.



### IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning (https://arxiv.org/abs/2407.05399)
Comments:
          Accepted at ACL 2024 Main Conference; 40 Pages (9 Pages + References + Appendix)

- **What's New**: IL-TUR는 인도 법률 텍스트 이해 및 추론을 위한 새로운 벤치마크입니다. 이 벤치마크는 영어와 힌디어를 포함해 9개의 인도 언어로 제공되는 다양한 도메인 특화 과제를 포함하고 있으며, 법률 문서의 이해와 추론을 중심으로 합니다. 이는 법률-NLP(L-NLP) 연구진이 서로 다른 모델들을 비교하고, 법률 분야의 연구를 촉진하는 플랫폼을 제공하기 위한 것입니다.

- **Technical Details**: IL-TUR는 법률 도메인 내에서 처리와 이해를 요구하는 8개의 주요 과제를 포함합니다. 이 과제들은 영어와 여러 인도 언어로 제공되며, 매뉴얼형 문서, 긴 문서 처리, 비정형 데이터 처리 등의 다양한 도전과제를 극복할 필요가 있습니다. 예를 들어, 평균적으로 인도 대법원의 문서는 4000 단어 이상으로 기존 NLP 모델(BERT 등) 처리 한계를 넘습니다. 이러한 문제를 극복하기 위한 다양한 LLMs 기반의 베이스라인 모델(Baseline Model) 결과가 제시되어 있으며, 링크를 통해 데이터셋, 모델 및 리더보드를 확인할 수 있습니다.

- **Performance Highlights**: IL-TUR 벤치마크 결과, 현재의 대규모 언어 모델(LLMs)들은 주어진 과제를 완벽하게 해결하지 못함을 보여줍니다. 이는 보다 향상된 모델 개발의 필요성을 나타냅니다. 연구 커뮤니티는 제안된 시스템들 및 베이스라인 모델들과 비교할 수 있는 리더보드(leaderboard)를 통해 자신의 모델을 업로드하고 테스트할 수 있습니다. 벤치마크는 직접적인 링크를 통해 접근 가능합니다.



### Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition (https://arxiv.org/abs/2407.05374)
Comments:
          Accepted to ACL 2024 Main

- **What's New**: 본 연구는 다양한 멀티모달리티(multimodal) 상실 상황에서도 성능 저하를 방지하는 새로운 멀티모달 Transformer 프레임워크를 제안합니다. 기존에는 멀티모달 데이터를 모두 갖추어야만 높은 성능을 보장할 수 있었으나, 제안된 프레임워크는 prompt learning을 사용하여 많은 계수(parameter) 학습을 줄이면서도 누락된 모달리티 정보를 효과적으로 처리할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 세 가지 유형의 프롬프트를 도입합니다: 생성 프롬프트(generative prompts), 신호 누락 프롬프트(missing-signal prompts), 그리고 유형 누락 프롬프트(missing-type prompts). 이 프롬프트들은 누락된 모달리티 특성을 생성하고, 모달리티 내 및 모달리티 간 정보를 학습하는데 도움을 줍니다. 이를 통해 고자원(high-resource) 도메인에서 학습된 지식을 저자원(low-resource) 도메인으로 전이할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 CMU-MOSEI, CMU-MOSI, IEMOCAP, CH-SIMS 등 네 가지 데이터셋에서 기존 방법들을 능가하는 성능을 나타냈습니다. 실험 결과, 제안된 세 가지 프롬프트가 결합되어 모델의 성능을 크게 향상시켰음을 확인했습니다. 특히, 모달리티 드롭아웃(dropout)을 70% 비율로 적용했을 때 최적의 성능을 보였습니다.



### Can Model Uncertainty Function as a Proxy for Multiple-Choice Question Item Difficulty? (https://arxiv.org/abs/2407.05327)
Comments:
          12 pages, 11 figures

- **What's New**: 새로운 연구는 다중 선택형 질문(MCQ)의 난이도 추정에 대해 다룬다. 이 연구는 LLMs(Large Language Models)의 불확실성을 활용하여 학생들의 실제 응답 분포와의 상관관계를 탐구한다. 특히 올바른 답변과 잘못된 답변의 경우와 질문 유형에 따라 모델의 행동이 다르게 나타난다는 것을 발견했다. 이를 통해 모델 불확실성을 항목 난이도 추정의 추가 지표로 활용하는 방법을 제안했다.

- **Technical Details**: 연구는 기존의 난이도 추정 연구와는 달리 모델의 불확실성을 활용하여 MCQ 난이도를 예측하는 방법을 탐구한다. 세부적인 MCQ 데이터셋을 사용하여 모델 선택과 학생 선택 간의 상관관계를 분석하였다. 또한, 선택 순서 민감도(choice order sensitivity)라는 MCQ에 특화된 모델 불확실성 지표를 중점적으로 다루었다. 본 연구는 학생과 LLM의 행동을 더 세부적으로 분석하여 다른 질문 양상에 따라 상관관계가 어떻게 달라지는지를 조사하였다.

- **Performance Highlights**: 모델 불확실성이 어느 정도 MCQ 난이도의 대리 지표 역할을 할 수 있음을 발견하였다. 또한, MCQ 질문 유형과 정답 여부 역시 중요한 요소임을 확인했다. 이는 LLMs가 '시뮬레이션된 교실 환경'으로서 활용될 수 있음을 시사하며, 교육용 애플리케이션을 위한 새로운 접근 방식을 제시한다.



### Rethinking Targeted Adversarial Attacks For Neural Machine Translation (https://arxiv.org/abs/2407.05319)
Comments:
          5 pages, 2 figures, accepted by ICASSP 2024

- **What's New**: 이 논문은 기존의 신경망 기계 번역(NMT) 시스템의 표적 적대적 공격(targeted adversarial attacks) 설정에 심각한 문제가 있음을 지적하고, 이러한 문제를 해결하기 위해 새로운 설정을 제시합니다. 더불어, 이 새로운 설정에서 적대적 예제를 만들어내는 Targeted Word Gradient Adversarial Attack(TWGA) 방법을 제안합니다.

- **Technical Details**: 기존 문제를 해결하기 위해 제안된 새로운 NMT 표적 적대적 공격 설정은 원래 문장의 의미를 보존하는 것을 보장하면서 표적 단어(targeted word)의 올바른 번역이 나타나지 않도록 합니다. 이를 위해 이중언어 사전(bilingual dictionary)을 활용하여 모든 가능한 참조 번역이 번역 결과에 나타나지 않도록 합니다. 새로운 공격 설정 하에서, 비표적 단어(non-targeted word)만 수정하여 표적 단어의 번역을 사라지게 하는 것을 목표로 합니다. 이렇게 생성된 적대적 예제는 문법적이고 의미가 통하는 문장이어야 합니다.

- **Performance Highlights**: 실험 결과, 제안된 TWGA 방법이 높은 품질의 적대적 예제를 효과적으로 생성할 수 있음을 보여줍니다. 특히, 기존 연구들이 작은 데이터셋에서 실험을 진행한 것과 달리, 본 연구는 대규모 데이터셋에서 최초로 실험을 진행하였으며, 여러 NMT 시스템에 대해 제안된 설정 하에서 표적 적대적 공격을 수행하여 깊이 있는 분석을 통해 가치 있는 발견을 도출했습니다.



### Beyond Binary Gender Labels: Revealing Gender Biases in LLMs through Gender-Neutral Name Predictions (https://arxiv.org/abs/2407.05271)
Comments:
          Accepted at ACL 2024, GeBNLP Workshop

- **What's New**: 이번 연구에서는 전통적인 이진 분류 시스템을 넘어서서, 이름을 기반으로 한 성별 예측에 '중립(neutral)' 성별 카테고리를 추가했습니다. 이는 특히 대규모 언어 모델(LLMs)에서 성별 편향을 분석하고 해결하기 위한 새로운 접근법입니다. 이번 분석에서는 출생 연도를 포함함으로써 성별 예측의 정확성을 높이기 위해 시도했습니다.

- **Technical Details**: 여러 기본 및 대규모 언어 모델을 평가하여 이름에 기반한 성별 예측 정확성을 분석했습니다. 이 연구에서는 첫 이름(first name)과 출생 연도의 데이터를 사용했습니다. 데이터 전처리 단계에서는 미국 사회 보장국(SSA), 캐나다의 알버타주, 프랑스의 세 가지 데이터를 사용했습니다. 중립 성별 예측을 위해선 특정 연도에 비율이 10% 이상인 경우에 '중립'으로 분류했습니다.

- **Performance Highlights**: 대부분의 LLMs는 남성 및 여성 이름을 80% 이상의 높은 정확도로 식별했습니다. 하지만 중립 성별 이름에 대해서는 40% 미만의 낮은 정확도를 보였습니다. 또한 영어 기반 이름이 비영어 기반 이름보다 더 높은 예측 정확도를 보였습니다. 출생 연도를 추가했을 때 전반적인 성별 예측 정확도는 크게 개선되지 않았습니다.



### CLIMB: A Benchmark of Clinical Bias in Large Language Models (https://arxiv.org/abs/2407.05250)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 가지는 임상적 편향을 체계적으로 평가하기 위한 새로운 벤치마크 시스템인 CLIMB(Clinical Bias in Large Language Models)를 소개합니다. CLIMB는 LLM의 내재적(intrinsic) 및 외재적(extrinsic) 편향을 동시에 평가합니다. 특히, 내재적 편향을 평가하기 위한 새로운 지표인 AssocMAD를 도입하였고, 외재적 편향을 counterfactual intervention 기법을 사용해 평가합니다.

- **Technical Details**: CLIMB는 LLM이 진단 예측과 같은 임상적 결정 작업에서 가지는 편향을 평가합니다. 내재적 편향은 LLM의 내부 연상(disparities in representations)에서 나타나며, 외재적 편향은 실제 사용 예에서 발생하는 성능 변화를 나타냅니다. CLIMB는 다양한 인구 통계 그룹 간의 차이를 측정하는 AssocMAD 지표를 도입하고, counterfactual intervention 기법을 사용해 인구 통계 정보와 보험 유형에 따른 편향을 평가합니다.

- **Performance Highlights**: 실험 결과, Mistral 및 LLaMA 계열의 많은 모델들이 내재적 및 외재적 편향을 모두 확인했습니다. 내재적 편향에서는 인구 통계 중립적 진단에서도 인구 통계 그룹 간의 명백한 차이를 보였습니다. 일부 의학적으로 적응된 LLM은 기본 LLM보다 더 큰 편향을 보여주었으며, 외재적 편향에서는 모든 모델에서 인구 통계 정보와 보험 유형의 변화에 따른 작은 성능 변화를 확인했습니다.

- **Implications**: 이번 연구는 임상적 결정 작업에서 LLM의 임상적 편향을 완화해야 할 필요성을 강조하며, LLM의 임상적 편향을 평가하는 새로운 표준을 설정하였습니다.



### Advancing Prompt Recovery in NLP: A Deep Dive into the Integration of Gemma-2b-it and Phi2 Models (https://arxiv.org/abs/2407.05233)
- **What's New**: 새로운 연구 영역인 프롬프트 복구에 대한 포괄적인 분석입니다. 프롬프트 복구는 입력 텍스트를 특정 출력으로 변환하는 동안 사용된 프롬프트나 지침을 재구성하는 작업입니다. 이 논문은 다양한 사전 학습된 언어 모델을 사용하여 프롬프트 복구의 효과를 비교 분석하였으며, Gemma-2b-it + Phi2 모델 조합이 뛰어난 성능을 보여준다고 밝혔습니다.

- **Technical Details**: 이 연구에서 사용된 Gemma-2b-it 모델은 텍스트 처리에 능숙하며, Phi2 모델의 트랜스포머 기반 특성과 결합하여 차세대 단어 예측 작업에서 뛰어난 성능을 발휘합니다. 이 모델은 두 단계의 사전 학습(pre-training) 과정을 거쳐 다양한 NLP 도메인의 프롬프트 복구 작업을 처리합니다. 연구에서는 모델 통합, 사전 학습 전략, 향상된 컨텍스트 이해 검토를 포함한 여러 혁신적인 접근 방식을 제안하였습니다.

- **Performance Highlights**: 연구 결과, Gemma-2b-it + Phi2 모델 조합이 프롬프트 복구 작업에서 우수한 성능을 나타냈습니다. 기존 벤치마크에 대한 비교 분석을 통해, 이 모델이 다른 모델에 비해 월등한 성능을 발휘하며, 텍스트 변환 작업에서 매우 높은 정확도로 프롬프트를 재구성할 수 있음을 입증하였습니다.



### Flood of Techniques and Drought of Theories: Emotion Mining in Disasters (https://arxiv.org/abs/2407.05219)
- **What's New**: 본 논문은 재난 상황에서의 감정 마이닝(emotion mining)에 대한 기존 연구들을 요약하고, 주요 발견과 지속적인 문제들을 강조합니다. 감정 마이닝 기술은 정확성을 어느 정도 충족하여 신속한 피해 평가 및 정신 건강 모니터링과 같은 응용 분야에서 활용될 수 있습니다. 하지만 데이터 중심 접근 방식을 채택한 많은 연구들이 임의적인 감정 분류나 소셜 미디어 데이터 수집의 편향성 문제, 이론적 프레임워크의 부족 등 몇 가지 방법론적인 문제를 가지고 있습니다.

- **Technical Details**: 재난 상황에서의 감정 마이닝은 소셜 미디어에 생성된 막대한 데이터를 활용하여 인간의 감정을 이해하는 중요한 도구로 떠오르고 있습니다. 그러나 많은 연구들이 임의적인 감정 분류 방법을 사용하고 있으며, 트위터와 같은 소셜 미디어에서 고소득층 사용자가 과대표되는 문제도 무시하고 있습니다. 또한, 교차 문화 비교(cross-cultural comparisons)와 같은 이론적 프레임워크를 적용하지 않고 있습니다.

- **Performance Highlights**: 감정 마이닝 기술은 신속한 피해 평가와 정신 건강 모니터링 등에서 적절한 정확도를 달성한 바 있습니다. 하지만 이러한 기술의 효과성과 신뢰성을 높이기 위해 컴퓨터 과학자와 사회 과학자 간의 학제 간 협력이 필요하다는 점을 강조하고 있습니다. 이로 인해 재난 대비, 대응 및 회복 활동을 개선할 수 있을 것입니다.



### Large Language Model as an Assignment Evaluator: Insights, Feedback, and Challenges in a 1000+ Student Cours (https://arxiv.org/abs/2407.05216)
Comments:
          An empirical report of our course: Introduction to Generative AI 2024 Spring (this https URL)

- **What's New**: 이번 연구는 대규모 언어 모델(LLM, Large Language Models)을 대학 강의에서 학생 과제를 자동 평가하는 데 적용한 사례를 소개합니다. 특히 GPT-4를 사용하여 1,028명의 학생을 대상으로 한 과제 평가에서 LLM 기반 평가자는 학생들에게 대체로 수용 가능한 평가 방법임을 확인했습니다.

- **Technical Details**: LLM 기반 평가자(LLM-based evaluator)는 특정 평가 기준에 따라 샘플을 평가하도록 지시받은 LLM을 의미합니다. 본 연구에서는 GPT-4와 같은 강력한 언어 모델을 사용하여, 지정된 평가 기준에 맞춰 학생 과제를 평가했습니다. 평가 방식은 Likert 척도(1-5점)를 사용하여 점수를 매기거나 샘플을 비교하여 우수한 샘플을 판단하는 형태로 이루어졌습니다.

- **Performance Highlights**: 조사 결과, 전체 학생의 75%가 LLM 기반 평가자를 수용 가능하다고 답했으나, 51%는 평가 지침에 맞지 않는 결과를 보았고, 22%는 평가 기준을 제대로 따르지 않는 경우를 관찰했다고 응답했습니다. 또한 47%의 학생들이 LLM 평가자를 조작하여 높은 점수를 받으려는 시도를 한 것으로 나타났으며, 이는 쉽게 탐지될 수 있음을 확인했습니다.



### BadCLM: Backdoor Attack in Clinical Language Models for Electronic Health Records (https://arxiv.org/abs/2407.05213)
Comments:
          AMIA 2024

- **What's New**: 본 논문은 BadCLM (Bad Clinical Language Models)이라는 새로운 주의 메커니즘 기반 백도어 공격 방법을 소개합니다. 이 방법은 전자 건강 기록(EHR) 시스템에서 클리니컬 언어 모델에 백도어를 은밀히 삽입하여 특정 트리거가 입력에 존재할 때 잘못된 예측을 하도록 합니다. 이는 모델이 평상시에는 정확하게 동작하지만, 트리거가 포함된 입력에 대해서는 오작동하게 만듭니다.

- **Technical Details**: BadCLM 방법은 트랜스포머(Transformer) 기반 모델의 주의 메커니즘을 조작하여 특정 주의 헤드들이 predefined trigger에만 초점을 맞추도록 합니다. 이 연구에서는 MIMIC III 데이터셋을 사용하여 네 가지 클리니컬 언어 모델(BERT, BioBERT, BioRoberta, ClinicalBERT)을 fine-tuning하여 병원 내 사망 예측 과제를 수행했습니다. 데이터셋의 정제 과정에서 각 환자의 방문을 독립적인 샘플로 간주하였으며, temporal embedding을 사용하여 시간 민감 정보를 포함시켰습니다.

- **Performance Highlights**: BadCLM 방법은 백도어 공격에서 90%의 성공률을 달성했으며, 포이즌 샘플(poisoned samples)이 제공되었을 때 높은 오분류율을 나타냈습니다. 그러나 깨끗한 샘플(clean samples)에서는 예측 정확도를 유지하여 백도어 공격의 은밀성을 보여주었습니다. 이러한 결과는 최신 클리니컬 언어 모델이 백도어 공격에 취약하다는 사실을 드러내고, 환자 안전 및 의료 무결성을 보호하기 위해 강력한 보안 프레임워크의 필요성을 강조합니다.



### Enhancing Language Learning through Technology: Introducing a New English-Azerbaijani (Arabic Script) Parallel Corpus (https://arxiv.org/abs/2407.05189)
Comments:
          This paper is accepted and published at NeTTT 2024 Conf

- **What's New**: 이 논문은 혁신적인 영어-아제르바이잔어(아랍 문자) 병렬 코퍼스(parallel corpus)를 소개합니다. 이 코퍼스는 미흡한 자원을 가진 언어를 위한 언어 학습과 기계 번역(machine translation, MT)의 기술적 격차를 해소하기 위해 설계되었습니다. 약 54만 8천 개의 병렬 문장과 언어별로 약 900만 단어로 구성되어 있으며, 뉴스 기사와 종교 텍스트와 같은 다양한 출처에서 수집되었습니다.

- **Technical Details**: 이 데이터셋은 언어 교육 기술과 자연어 처리(natural language processing, NLP) 응용 프로그램을 향상하기 위해 만들어졌습니다. 특히 신경 기계 번역(neural machine translation, NMT) 혁명에서 뒤쳐져 있는 튀르크 언어 그룹을 위해 중요한 진전을 나타냅니다. 영어-아제르바이잔어(아랍 문자) 언어 쌍에 대한 최초의 종합적 사례 연구를 제시하며, 미흡한 자원 환경에서 NMT의 변혁적 잠재력을 강조합니다.

- **Performance Highlights**: 이 코퍼스는 깊은 학습(machine learning) MT 시스템을 훈련시키는 데 효과적임을 입증하였으며, 연구자와 교육자가 이중언어 교육과 다중언어 소통을 촉진하려는 노력에 필수적인 자산이 될 것입니다. 디지털 자원이 부족한 언어에 대한 NMT 응용 프로그램에 대한 미래 탐사의 길을 열어 세계 언어 교육 프레임워크를 향상시킵니다.



### Identifying Intensity of the Structure and Content in Tweets and the Discriminative Power of Attributes in Context with Referential Translation Machines (https://arxiv.org/abs/2407.05154)
Comments:
          11 pages, 3 figures, 12 tables

- **What's New**: 이 연구에서는 참조 번역 기계(Referential Translation Machines, RTMs)를 사용하여 주어진 속성과 두 단어 간의 유사성을 식별하는 방법을 제시합니다. 이를 위해 RTMs는 트윗의 텍스트 및 구조적 강도를 예측하는 작업(Task 1)과 속성이 두 단어 간의 차별성을 예측하는 작업(Task 10)에 적용되었습니다.

- **Technical Details**: RTMs는 기계 번역 성능 예측(Machine Translation Performance Prediction, MTPP)을 통해 단어 간의 유사성을 측정합니다. 이를 위해, RTMs는 'parfda'를 사용하여 평행 데이터와 단일 언어 데이터를 선택하여 특징을 도출하고, 예측 모델을 구축합니다. 이 모델은 트레이닝 데이터와 테스트 데이터 간의 유사성을 기반으로 예측합니다. RTM 모델은 다양한 도메인과 작업에 적용될 수 있어 단일 언어 및 다중 언어 설정 모두에서 유용합니다.

- **Performance Highlights**: 트윗의 구조 및 내용 강도를 예측하는 작업(Task 1)과 속성이 두 단어를 구별하는 역할(Task 10) 모두에서 RTM 모델은 유망한 결과를 보였습니다. 특히, Task 10에서는 속성과 단어의 유사성을 기반으로 예측 모델을 구축하여 좋은 성능을 보여주었습니다.



### Automatic Prediction of the Performance of Every Parser (https://arxiv.org/abs/2407.05116)
Comments:
          8 pages, 2 figures, 7 tables

- **What's New**: 이 논문에서는 MTPPS(Machine Translation Performance Prediction System)를 이용한 새로운 구문 분석 성능 예측(Parsing Performance Prediction; PPP) 모델, MTPPS-PPP를 소개합니다. 이 모델은 언어나 구문 분석기와는 독립적으로 작동하며 텍스트, 링크 구조, 브래킷 트리(bracketing tree) 구조 정보를 사용하여 구문 분석기의 성능을 예측할 수 있습니다. MTPPS-PPP는 텍스트의 문법적 난이도를 평가하거나 특정 도메인에 적합한 구문 분석기를 선택하는 데 유용합니다.

- **Technical Details**: MTPPS-PPP 시스템은 다양한 특성을 기반으로 구문 분석 성능을 예측합니다. 이 시스템은 텍스트 유사성, 링크 구조, 브래킷 구조 등의 정보로부터 특성을 도출합니다. 텍스트 정보는 최대 3-그램(n-grams)을 사용하고, 링크 구조는 무감독 구문 분석기 CCL(Seginer, 2007)의 링크를 사용합니다. 브래킷 구조는 구문 트리의 구조적 속성을 측정하며, 이를 통해 문장의 오른쪽 가지(right branch)와 왼쪽 가지(left branch)의 비율 등을 분석합니다.

- **Performance Highlights**: MTPPS-PPP는 Charniak and Johnson parser를 위해 WSJ23 테스트 세트에서 브래킷 F1 점수를 예측할 때 약 7.4%의 오류율을 보입니다. MAE(평균 절대 오차)는 0.0678, RAE(상대 절대 오차)는 0.85입니다. 텍스트 정보만을 사용해도 이전 연구와 유사한 성능을 보이며, 구문 분석기나 언어에 의존하지 않고도 예측이 가능합니다.



### Exploring Sound Change Over Time: A Review of Computational and Human Perception (https://arxiv.org/abs/2407.05092)
Comments:
          LChange24 Camera Ready

- **What's New**: 이 연구는 역사적 음운 변화 및 현재 진행 중인 음운 변화를 컴퓨터 기반 모델 및 인간 기반 모델의 관점에서 비교한 최초의 리뷰를 제공합니다. 주로 컴퓨터 접근 방식은 어원 자료집(etymological datasets)을 분석하여 역사적 음운 변화를 감지하는데, 인간 접근 방식은 녹음 코퍼스(recording corpora)를 통해 현재 진행 중인 음운 변화를 감지합니다. 두 접근 방식 모두 음성학적 및 음향학적 수준에서 서로를 보완할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 컴퓨터 기반 접근 방식은 국제 발음 기호(IPA)를 사용한 어원 자료집 분석을 통해 역사적 음운 변화를 감지합니다. 예를 들어, 덴마크 역사 코퍼스에서 훈련된 음소 임베딩(phoneme embeddings)을 사용하여 음소 쌍 사이의 시간적 변화를 비교합니다. 반면, 인간 기반 접근 방식은 녹음 코퍼스 및 설문조사를 통해 현재 진행 중인 음운 변화를 관찰합니다. 각각의 접근 방식이 가진 기능을 융합하여 음운 변화의 총체적인 인식을 도출하는 방식이 논의됩니다.

- **Performance Highlights**: Boldsen과 Paggio의 연구에서는 덴마크어 역사 코퍼스를 사용하여 훈련된 음소 임베딩을 통해 voiceless plosives가 voiced counterparts로 변화하는 것을 성공적으로 감지했습니다. 또한, Fourrier와 Sagot의 연구는 통계적 모델이 작은 데이터셋에서 더 우수한 성능을 보이지만, 신경 기계 번역 모델은 다양한 원형형태(proto-forms)를 처리하는 데 우수한 능력을 보여줬습니다. Ceolin과 Sayeed는 음소의 마크드니스(markedness)를 인식할 수 있는 확률 모델을 제안하여, 후에 나타난 무표 음소(unmarked phonemes)가 더 높은 빈도를 가진다는 결과를 확인했습니다.



### Cross-Lingual Word Alignment for ASEAN Languages with Contrastive Learning (https://arxiv.org/abs/2407.05054)
- **What's New**: 최근 연구에서는 저자원이 언어 환경에서 기존의 사전 학습된 언어 모델보다 더 나은 성능을 보이는 BiLSTM 기반 인코더-디코더 모델을 제안했습니다. 하지만 이 모델은 단어 임베딩 공간의 유사성만을 고려하고, 단어 임베딩 간의 차이를 명시적으로 모델링하지 않았습니다. 이에 대한 제한점을 보완하기 위해 대조 학습(contrastive learning)을 BiLSTM 기반 인코더-디코더 프레임워크에 통합하는 방법을 제안합니다. 주요 아이디어는 다중 시각적 부정 샘플링 전략(multi-view negative sampling strategy)을 도입해 공유된 교차 언어 임베딩 공간에서 단어 쌍 간의 차이를 학습하는 것입니다.

- **Technical Details**: 우리의 제안된 모델은 대조 학습을 활용하여 교차 언어 단어 정렬의 정확성을 높이는 방식을 채택했습니다. BiLSTM 기반 인코더-디코더 모델에 다중 시각적 부정 샘플링 전략을 포함시켜 단어 쌍 간의 차이를 명시적으로 학습합니다. 특히, 4개의 ASEAN 언어(라오어, 베트남어, 태국어, 인도네시아어)를 포함한 5개의 이중 언어 정렬 데이터셋에서 우리의 모델을 평가하며, 대조 학습 통합이 모든 데이터셋에서 일관되게 단어 정렬 정확성을 향상시킴을 확인했습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 모든 데이터셋에서 일관되게 단어 정렬 정확성을 향상시키는 것으로 나타났습니다. 이는 제안된 방법이 저자원 환경에서 매우 효과적임을 입증합니다. 또한, 향후 ASEAN 및 기타 저자원 단어 정렬 연구 지원을 위해 데이터셋 및 코드를 공개할 예정입니다.



### Enhance the Robustness of Text-Centric Multimodal Alignments (https://arxiv.org/abs/2407.05036)
- **What's New**: 이 논문에서는 다양한 모달리티를 텍스트로 변환하여 대형 언어 모델(LLM) 입력으로 사용하는 새로운 방법론을 제안합니다. 기존 방법론이 데이터의 누락, 잡음, 모달리티 결핍이 있을 때 하위 모델의 강건성(robustness)을 저하시키는 문제를 가지고 있었음을 밝혀냈습니다. 이를 해결하기 위해 텍스트 중심 접근법을 개선하여 다양한 모달리티의 상황에서 더욱 강건한 성능을 발휘하게 합니다.

- **Technical Details**: 텍스트 중심 멀티모달 정렬 방법은 다양한 데이터를 텍스트로 변환하여 LLM이 이를 처리하고 이해할 수 있도록 합니다. 이 연구에서는 LLaVA, VideoChat-Text, OphGLM 등과 같은 기존의 텍스트 중심 정렬 방법과 비교하면서, 모달리티 요약(modality summarization) 및 추론 보강(reasoning augmentation) 기법을 제안합니다. 입력 데이터를 정규화된 텍스트 형식으로 변환한 후, 다양한 모달리티 간의 상호작용을 증대시키기 위해 텍스트 요약과 대규모 외부 지식을 이용한 데이터 증강(data augmentation)을 활용합니다.

- **Performance Highlights**: 제안된 접근법은 실험 결과에서 뛰어난 성능을 보였습니다. PetFinder 데이터셋과 같은 실제 데이터셋을 활용한 실험에서, 다양한 모달리티 조건에서 잡음 레벨이 높을 때도 기존 방법보다 15.2% 우수한 성과를 보였습니다. 특히, 모든 모달리티에 적용된 잡음 환경에서도 성능 저하율이 11.3%에 불과하여, 기존 최적의 방법보다 4% 이상 높은 강건성을 유지했습니다.



### A Principled Framework for Evaluating on Typologically Diverse Languages (https://arxiv.org/abs/2407.05022)
- **What's New**: 이 논문은 언어 전형론 (typology)에 기초한 다국어 모델의 다각적 평가를 위한 새로운 언어 샘플링 프레임워크를 제시합니다. 기존의 방식들이 다국어 평가에서 최적의 결과를 제공하지 못했던 문제를 지적하며, 전체 언어 중에서 대표적으로 선택될 수 있는 방법론을 재정립하였습니다.

- **Technical Details**: 이 논문은 언어 전형론 (linguistic typology)를 활용하여 다국어 모델 평가를 위한 언어 샘플링 방법을 제안합니다. 저자들은 두 가지 샘플링 방법을 소개하며, 이를 통해 고유의 언어적 다양성을 포착할 수 있는 방법론을 제공합니다. 또한 언어의 전형적 거리 (typological distance)를 계산하여 다각적 언어 선택 방법을 체계화했습니다. 이를 위해 Python 패키지도 공개했습니다. 이 패키지는 GitHub 저장소에서 제공됩니다: https://github.com/esther2000/typdiv-sampling

- **Performance Highlights**: 저자들은 다양한 언어적 특성을 포함하는 샘플링 방법이 다국어 모델의 일반화 성능에 어떻게 영향을 미치는지 증명했습니다. 이들은 기존의 방법보다 더 효과적으로 다양한 언어를 선택하는 것으로 나타났으며, 이렇게 선정된 언어들이 다국어 모델 평가에서 보다 일반화된 결과를 제공한다는 것을 보여줍니다.



### How do you know that? Teaching Generative Language Models to Reference Answers to Biomedical Questions (https://arxiv.org/abs/2407.05015)
Comments:
          Accepted at BioNLP Workshop 2024, colocated with ACL 2024

- **What's New**: 최근 논문은 생의학 분야에서의 신뢰도 향상을 위해 기능을 최적화한 Retrieval-Augmented Generation (RAG) 시스템을 소개합니다. PubMed에서 검색한 관련 초록을 입력으로 제공받아 대형 언어 모델(LLM)의 상호 참조형 질문-응답을 통해 답변의 정확도를 높입니다. PubMed 검색 엔진과 비교해 절대적으로 23%의 성능 개선을 이뤘으며, GPT-4 Turbo와 유사한 성능을 나타냈습니다.

- **Technical Details**: 제안된 RAG 시스템은 두 개의 주요 구성 요소로 이루어져 있습니다. 하이브리드 의미 및 어휘 검색에 기반한 IR (Information Retrieval) 구성 요소와, 이를 통해 검색된 PubMed 초록을 컨텍스트로 사용하는 생성형 LLM입니다. 구체적으로, BM25 랭킹 함수와 OpenSearch를 사용해 PubMed 문서의 제목과 초록을 색인화하고, Qdrant 벡터 데이터베이스를 사용해 어휘 및 의미 검색을 수행합니다. 데이터셋 내의 공백 초록을 제거한 후, 두 가지 색인을 오프라인 모드로 구축해 온라인 검색에 사용합니다.

- **Performance Highlights**: 본 시스템은 PubMed 검색 엔진 대비 절대 23%의 성능 개선을 달성했습니다. 또한 소규모 샘플에 대한 수동 평가 결과, 최적화된 LLM 구성 요소가 GPT-4 Turbo와 유사한 성능을 보였습니다. 이는 더욱 신뢰성 높은 생의학 정보 제공에 기여할 것으로 기대됩니다.



### Progress or Regress? Self-Improvement Reversal in Post-training (https://arxiv.org/abs/2407.05013)
- **What's New**: 대형 언어 모델(Large Language Models, LLMs)의 문제 해결 능력을 향상시키기 위해 자기 개선(self-improvement)이라는 기법이 사용되고 있습니다. 본 연구에서는 이러한 자기 개선이 진정한 발전을 가져오는지, 아니면 예기치 못한 퇴보를 초래하는지 평가하기 위한 종합적인 프레임워크를 제안합니다.

- **Technical Details**: 연구는 주로 반복적 선호 학습(iterative preference learning) 및 모델의 자기 생성 응답을 활용한 반복적 지도 학습(supervised fine-tuning)을 통해 수행됩니다. 본 연구에서는 자기 개선을 위한 다양한 후훈련(post-training) 패러다임을 분석하고, 각 요소(초기 모델, 작업 데이터 세트, 반복 횟수 등)를 분리하여 모델 성능에 미치는 영향을 조사합니다. 보다 깊이 있는 평가를 위해, 테스크 다각성(metrics diversity)과 분포 외 일반화(out-of-distribution generalization)도 분석합니다.

- **Performance Highlights**: 본 연구의 실험 결과, pass@1 정확도는 높아지지만 출력 다각성(output diversity)과 분포 외 일반화 측면에서 모델의 성능이 하락하는 '자기 개선 반전(self-improvement reversal)' 현상이 나타났습니다. 이러한 결과는 LLM의 현행 자기 개선 방법이 복잡한 문제를 해결하는 데 충분하지 않다는 것을 시사합니다.



### Recent Advancements and Challenges of Turkic Central Asian Language Processing (https://arxiv.org/abs/2407.05006)
- **What's New**: 중앙아시아 언어인 카자흐어, 우즈베크어, 키르기스어, 투르크멘어의 NLP 연구는 일관된 데이터 부족과 언어 자원의 부재와 같은 저자원 언어의 일반적인 과제를 가지고 있습니다. 그러나 최근 몇 년간 언어별 데이터셋 수집 및 하위 작업 기술 개발을 통해 연구가 크게 발전했습니다. 이 논문은 2024년 5월까지의 연구를 요약하고, 향후 연구의 잠재적 영역을 식별하고자 합니다.

- **Technical Details**: 이 논문은 해당 언어들의 언어적 특성, 현재 개발된 기술의 범위와 성과, 더 높은 자원 언어로부터의 전이 학습(transfer learning) 기술 적용, 각 언어의 라벨링된 데이터와 라벨링되지 않은 데이터 가용성에 대한 광범위한 개요를 제공합니다. 특히, 이러한 언어들은 형태론적으로 풍부한 언어로 설명되며, 복잡한 형태론적 구조와 형태 통사적 특징 때문에 기계 번역이나 개체명 인식(Named Entity Recognition) 작업에서 어려움을 겪습니다.

- **Performance Highlights**: 카자흐어의 가장 다양한 데이터셋이 포함된 것은 Almaty Corpus of Kazakh Language 및 UD 트리뱅크(treebank) 등입니다. 이름 인식작업에는 KazNERD 코퍼스, 개방 도메인 질의 응답에는 KazQAD 코퍼스, 기계 번역을 위한 KazParC 병렬 코퍼스 등이 있습니다. multimodal 데이터셋으로는 1,200시간 이상의 전사된 오디오를 포함한 Kazakh Speech Corpus 2(KSC2)와 270시간의 녹음된 오디오를 포함한 KazakhTTS2 데이터셋이 있습니다. 이러한 리소스는 중앙아시아 언어 연구의 진전을 크게 촉진할 것입니다.

- **Performance Issues**: 언어 간 전이 학습은 터키어를 포함한 유사 언어군에서 주된 해결책이 되지만, 각 언어의 스크립트 불일치로 인해 전이 학습의 적용이 제한될 수 있습니다. 예를 들어, 우즈베크어는 라틴 스크립트를 사용하지만, 카자흐어는 여전히 주로 키릴문자를 사용합니다. 따라서, 데이터 정리 또는 사전 처리 수준에서 추가 단계가 필요할 수 있습니다.



### TRACE: TRansformer-based Attribution using Contrastive Embeddings in LLMs (https://arxiv.org/abs/2407.04981)
- **What's New**: 최근 대형 언어 모델(LLMs)의 급속한 발전은 자연어 이해 및 생성에서 상당한 도약을 나타냅니다. 그러나 이러한 발전과 함께 LLM 응답의 책임성과 투명성에 대한 중요한 문제가 제기되고 있습니다. 본 논문에서는 대조 학습(contrastive learning)을 활용한 새로운 TRansformer 기반 Attribution 프레임워크인 TRACE를 제안합니다. TRACE는 소스 출처를 정확히 귀속하는 능력을 크게 향상시키며, LLM의 신뢰성을 높이는 데 중요한 도구가 됩니다.

- **Technical Details**: TRACE는 대조 학습을 사용한 감독된 대조 학습 문제로, 동일한 소스에서 나온 데이터를 동일한 레이블로 지정합니다. SBERT(Sentence-BERT)을 사용해 우수한 문장 임베딩을 생성하며, SimCLR의 비선형 프로젝션 헤드를 추가함으로써 임베딩 품질을 향상시킵니다. 또한, NT-Xent 손실 함수를 사용하여 소스 간의 유사도를 최대화하고, 다른 소스 간의 유사도를 최소화합니다.

- **Performance Highlights**: TRACE는 다양한 설정에서 뛰어난 성능과 효율성을 실험적으로 평가하였으며, 그 결과 정확성, 확장성, 해석 가능성 및 강건성 모두에서 기존 방법들보다 우수한 성능을 보였습니다. 특히, TRACE는 대규모 응용 프로그램에서도 성능 저하 없이 효과적으로 동작할 수 있음을 증명했습니다.



### EVA-Score: Evaluation of Long-form Summarization on Informativeness through Extraction and Validation (https://arxiv.org/abs/2407.04969)
Comments:
          16 pages, 3 figures, submitted to EMNLP

- **What's New**: 이 논문에서는 문서 레벨 관계 추출(Document-level Relation Extraction)과 원자 사실 체인 생성(Atomic Fact Chain Generation)을 사용하여 요약의 정보성을 자동으로 계산하는 새로운 평가 지표인 EVA-Score를 제안합니다. 이 지표는 기존 평가 방법인 유사성 기반 ROUGE와 BERTScore의 한계를 극복하고, 요약의 정보성이 얼마나 높은지를 정량적으로 분석합니다.

- **Technical Details**: EVA-Score는 먼저 참조 요약(reference summary)과 후보 요약(candidate summary)에서 원자 사실(atomic facts)을 추출합니다. 이후 문서 레벨 관계 추출을 통해 계층적 관계를 유지하고, 정보 검색을 이용하여 가장 관련성이 높은 원자 사실들을 참조로 사용하여 상세한 검증을 수행합니다. 이 과정에서 LLMs는 새롭게 추가된 정보에만 집중하여 정보가 포함되었는지 여부를 판단합니다.

- **Performance Highlights**: EVA-Score는 Pearson, Spearman, Kendall 상관계수에서 기존 최고의 평가 지표인 BERTScore보다 각각 0.166, 0.120, 0.09만큼 더 높은 인간 일치를 보여줍니다. 또한, EVA-Score를 사용한 장문 요약(long-form summarization) 평가에서 GPT-4가 장기 문맥 처리 시 가장 뛰어난 성능을 보였으며, Mixtral-7B-Instruct 모델도 유망한 성능을 보여 줍니다.



### Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression (https://arxiv.org/abs/2407.04965)
- **What's New**: 최근 모델 압축 기술의 발전으로 대형 언어 모델(LLM, Large Language Models)의 실제 사용 사례가 증가하고 있습니다. 이 논문은 압축된 LLM이 보안에 미치는 영향을 체계적으로 평가하려는 첫 시도입니다. 이전의 연구들은 주로 perplexity(혼돈도)를 유지하는데 중점을 두었지만, 이 연구는 모델의 편향, 독성, 방언 편향 및 언어 모델링과 다운스트림 작업 성능 등의 다양한 측면을 분석했습니다.

- **Technical Details**: 모델 압축은 주로 세 가지 방법을 사용합니다: 구조적 프루닝(pruning), 비/준-구조적 프루닝 및 양자화(quantization). 연구에서는 Llama-2(7B 및 13B 파라미터)와 Tülu-2 모델을 베이스로 사용하여 다양한 압축 기술의 두 가지 주요 종류—프루닝과 양자화—를 평가했습니다. 프루닝은 모델에서 중요하지 않은 가중치를 제거해 저장/메모리와 추론 비용을 줄이는 방법입니다. 반면에, 양자화는 모델 가중치의 비트 수를 줄여 모델을 압축합니다.

- **Performance Highlights**: 연구 결과, 압축된 LLM은 생성 품질 저하로 인한 편향 감소 효과가 있을 수 있으나, 대표적 편향은 크게 변화하지 않거나 증가할 수 있습니다. 또한, 압축률이 높아질수록 압축 방법에 따라 보호된 그룹에 대한 편향 영향이 다르게 나타났습니다. 특히 양자화 방법은 모델의 편향, 독성 및 성능을 중간 압축률(예: 50%)에서 대부분 유지할 수 있었지만, 프루닝 방법은 동일한 압축률에서 성능 저하가 크게 발생했습니다.



### Granular Privacy Control for Geolocation with Vision Language Models (https://arxiv.org/abs/2407.04952)
- **What's New**: 이 논문은 거대한 비전 언어 모델(Vision Language Models, VLMs)의 놀라운 이미지 지리 위치 추적(geolocation) 능력을 밝히며, 이를 통해 발생할 수 있는 새로운 프라이버시 위험을 경고합니다. 이를 해결하기 위한 첫 단계로, 저자들은 사용자와의 지리 위치 추적 대화를 관리할 수 있는 능력을 테스트하기 위해 GPTGeoChat이라는 새로운 벤치마크를 개발했습니다.

- **Technical Details**: GPTGeoChat 벤치마크는 1,000개의 이미지와 사용자와 GPT-4v 간의 상호 대화 1,000개를 포함하며, 모든 대화의 각 턴마다 공개된 위치 정보를 주석으로 달았습니다. 이 데이터를 이용하여 다양한 VLM들이 GPT-4v의 지리 위치 대화를 얼마나 효과적으로 관리할 수 있는지 평가했습니다. 특히, 단순 프롬프트 기반 모델과 미세 조정된 모델 간의 성능을 비교하여, 어떤 모델이 민감한 위치 정보 유출을 잘 막을 수 있는지 분석했습니다.

- **Performance Highlights**: GPT-4v는 이미 기존의 이미지 지리 위치 추적 시스템을 능가하여, IM2GPS 테스트 셋에서 이미지의 GPS 좌표를 예측하는데 거리 오차가 1km 이내인 경우가 24%에 달했습니다. 미세 조정된 모델은 국가나 도시 수준에서는 프롬프트 기반 모델에 필적하지만, 식당이나 건물 이름 수준의 더 미세한 위치 정보 관리에는 감독된 데이터로 추가 미세 조정이 필요함을 발견했습니다.



### NADI 2024: The Fifth Nuanced Arabic Dialect Identification Shared Task (https://arxiv.org/abs/2407.04910)
Comments:
          Accepted by The Second Arabic Natural Language Processing Conference

- **What's New**: NADI 2024에서 다이얼렉트 식별, 다이얼렉트 수준 파악, 표준 아랍어(MSA)로의 번역 등 세 가지 서브태스크로 구성한 공유 작업을 통해 아랍어 NLP 연구의 최신 동향과 평가 결과를 발표했습니다. 이번 연구에는 독특한 51개 팀이 등록하였으며, 12개 팀이 참여했습니다.

- **Technical Details**: 서브태스크 1은 여러 아랍어 다이얼렉트에 속하는 텍스트를 식별하는 'Multi-label classification task', 서브태스크 2는 텍스트의 다이얼렉트 수준을 0에서 1까지 측정하는 'Arabic level of dialectness', 서브태스크 3는 다이얼렉트에서 MSA로의 'Machine Translation' 작업입니다.

- **Performance Highlights**: 서브태스크 1에서 최고의 팀은 50.57 F1 점수를, 서브태스크 2에서는 0.1403 RMSE를, 서브태스크 3에서는 20.44 BLEU를 기록하였습니다. 결과는 아랍어 다이얼렉트 처리 작업이 여전히 도전적임을 보여줍니다.



### MMSci: A Multimodal Multi-Discipline Dataset for PhD-Level Scientific Comprehension (https://arxiv.org/abs/2407.04903)
Comments:
          Code and data are available at this https URL

- **What's New**: 최근 대형 언어 모델(LLMs) 및 대형 멀티모달 모델(LMMs)의 발전은 과학 기사를 이해하는 AI 과학 보조에 대한 수요를 증가시켰습니다. 하지만, 전문적이고 대학원, 박사 수준의 과학 내용을 평가하는 데 있어 상당한 격차가 존재합니다. 이를 해결하기 위해 Nature Communications 저널에서 발행된 공개 액세스 과학 기사들을 이용하여 멀티모달, 다학문 데이터셋을 수집하고 이를 기반으로 다양한 태스크와 설정의 벤치마크를 생성했습니다.

- **Technical Details**: 이번 연구에서는 논문의 제목, 초록, 본문 콘텐츠, 참조 문헌, 그리고 그림과 그 설명을 포함하는 데이터를 수집했습니다. 수집한 데이터는 72개의 과학 분야에 걸쳐 있으며, 이를 통해 모델이 학습 및 평가될 수 있도록 시각 지침 따르기 데이터(visual instruction-following data)와 교차 학습(interleaved learning) 데이터를 구축했습니다. AI 모델 성능 평가를 위해 과학 그림 캡션 생성과 시각적 질문 답변(VQA) 태스크를 포함하는 벤치마크를 만들었습니다.

- **Performance Highlights**: 평가 결과, 많은 오픈 소스 모델들이 각각의 과업에서 상당한 어려움을 겪는 것으로 나타났습니다. 심지어 GPT-4V와 GPT-4o도 정확하고 관련성 있는 캡션을 생성하거나 그림과 설명을 일치시키는 데 어려움을 보였습니다. 구축된 데이터셋을 사용하여 7B LLaVA 모델을 훈련시킨 결과, 이는 GPT-4V/o와 비슷한 성능을 보였습니다. 또한, 자료 생성 태스크에서 교차 학습 데이터를 사용한 결과 성능 향상이 있음을 확인했습니다.



### Automating Venture Capital: Founder assessment using LLM-powered segmentation, feature engineering and automated labeling techniques (https://arxiv.org/abs/2407.04885)
Comments:
          For the relevant code, see this https URL

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 이용해 창업자의 특성을 기반으로 스타트업 성공 예측을 시도한 것입니다. 특히 VC(벤처 캐피탈) 의사결정에 있어 LLM prompting 기술을 활용해 창업 성공을 예측하고 있습니다. 이 프레임워크는 스타트업 성공 예측을 향상시켜 VC의 투자 전략 최적화에 중요한 영향을 미칠 수 있다는 점에서 새로운 가능성을 제시합니다.

- **Technical Details**: 데이터셋은 총 3,716명의 창업자 (성공 1,022명, 실패 2,694명)에 대한 정보를 포함하고 있으며, 이들의 링크드인(LlinkedIn) 데이터를 기반으로 다양한 특징을 추출했습니다. LLM인 GPT-4를 이용해 창업자의 교육, 기술, 경력, 위치 등을 요약하고, 창업자가 속하는 특정 레벨과 페르소나를 정의했습니다. 이러한 접근은 Chain-of-Thought (CoT) prompting와 비지도 학습 방법(envalid unsupervised LLM-based method)을 사용하여 각각의 창업자를 레벨(L1—L10)과 페르소나(20가지)로 분류하는 방식으로 이루어졌습니다.

- **Performance Highlights**: LLM 기반 접근법을 통해 창업자의 특성 중 특정 특징이 성공에 미치는 영향을 확인할 수 있었으며, 높은 정확도로 창업 성공을 예측할 수 있었습니다. 이 프레임워크는 VC들의 초기 스타트업 투자 결정에서 더 나은 정보를 제공할 수 있으며, 투자 전략을 최적화하는 데 중요한 역할을 할 수 있습니다.



### Question Answering with Texts and Tables through Deep Reinforcement Learning (https://arxiv.org/abs/2407.04858)
- **What's New**: 본 논문은 텍스트와 테이블에서 정보를 가져와야 하는 오픈 도메인 질문에 대해 멀티-홉(multi-hop) 답변을 생성하는 새로운 아키텍처를 제안합니다. 이 시스템은 Open Table-and-Text Question Answering(OTT-QA) 데이터셋을 사용하여 검증 및 훈련하며, 강화 학습(reinforcement learning)을 통해 다양한 첨단 도구를 순차적으로 선택하여 원하는 답변을 생성합니다. 이 시스템은 19.03의 F1-score를 달성했으며, 이는 문헌 속 기존 반복 시스템과 비교할 만한 성능입니다.

- **Technical Details**: QA 시스템은 일반적으로 두 가지 주요 요소로 구성됩니다: 검색기 모듈(retriever module)과 리더 모듈(reader module). 검색기 모듈은 관련된 텍스트와 패시지를 검색하고, 리더 모듈은 이 데이터를 처리하여 의미 있는 답변을 생성합니다. 본 논문에서는 강화 학습을 이용하여 표준 모듈이 각 단계에서 어떤 작업을 수행할지 순차적으로 결정하는 심층 강화 학습(Deep Reinforcement Learning, DRL) 접근 방식을 제안합니다. DRL은 텍스트 검색, 테이블 검색, 답변 생성 등의 작업을 함에 있어 최적의 행동을 학습합니다.

- **Performance Highlights**: 제안된 시스템은 OTT-QA 데이터셋에서 19.03의 F1-score를 기록하여, 기존의 반복(recursive) 시스템들과 비교할 때도 경쟁력 있는 성능을 보여주고 있습니다.



### Towards Enhancing Coherence in Extractive Summarization: Dataset and Experiments with LLMs (https://arxiv.org/abs/2407.04855)
Comments:
          10 pages

- **What's New**: 이번 연구는 사람의 의도를 반영하여 일관성 있는 추출 요약 (coherent extractive summarization)을 개선하기 위해 자연어 피드백과 함께 인간 주석 데이터셋(human-annotated dataset)을 제안합니다. 이 데이터셋은 뉴스, 토론, TV쇼, 회의, 대화 등 5가지 공개 데이터셋을 사용하여 체계적으로 생성되었습니다. 이를 통해 추출 요약의 일관성을 높이기 위해 LLM(Large Language Models)을 감독 학습 (supervised fine-tuning)으로 맞추는 방법을 소개합니다.

- **Technical Details**: 데이터셋을 만들기 위해 전문 주석자를 고용하여 중요한 문장을 추출하고 자연어 피드백을 제공했습니다. Falcon-40B와 Llama-2-13B와 같은 두 디코더 전용 모델 및 FLAN-T5와 같은 인코더-디코더 모델에 대해 실험을 수행했습니다. 이 과정에서 모델은 인간 피드백을 반영한 두 가지 감독 학습 전략을 사용하여 최고의 일관성을 이루어 냈습니다. Rouge-L을 사용하여 모델의 성능을 평가했습니다.

- **Performance Highlights**: Falcon-40B와 Llama-2-13B 모델을 통해 실험한 결과, Rouge-L에서 약 10%의 절대적인 성능 향상이 있었습니다. 인간 평가에서도 새로운 접근 방식을 사용한 요약본이 더 일관성 있게 평가되었습니다. 이는 사용자 피드백을 모델에 반영하여 일관성을 향상시킬 수 있음을 시사합니다.



### Statistical investigations into the geometry and homology of random programs (https://arxiv.org/abs/2407.04854)
Comments:
          16 pages, 11 figures

- **What's New**: 이번 연구에서는 AI가 생성한 파이썬 프로그램 간의 관계를 기하학적(geometric) 및 위상학적(topological) 방법을 통해 설명하는 새로운 접근 방식을 제안합니다. 이 연구는 ChatGPT와 같은 대형 언어 모델(LLM)이 생성한 프로그램의 구문 트리(syntax tree) 간의 트리 편집 거리(tree-edit distance)를 이용하여 모델링합니다. 이는 전통적인 고차원 임베딩 방법의 한계를 극복하기 위한 것입니다.

- **Technical Details**: 연구진은 모든 문법적으로 올바른 프로그램을 거리 함수(d)를 이용해 표현되는 메트릭 공간(metric space)으로 간주합니다. 이때 프로그램들 간의 거리는 트리 편집 거리(tree-edit distance)를 이용해 측정합니다. 트리는 파싱된 구문 트리를 사용하며, 이는 매우 명확하고 프로그래밍 구조를 잘 나타내줍니다. 이를 위해 파이썬의 내장 모듈인 AST를 활용합니다. 또한 연구에서는 기하학적 요약 통계(geometric summary statistics), 공간 포인트 통계(spatial point statistics) 및 위상 데이터 분석(topological data analysis) 방법을 소개하며, 구체적으로는 지속적 호몰로지 다이어그램(persistent homology diagrams)을 언급합니다.

- **Performance Highlights**: 이 방법론의 유용성을 입증하기 위해 연구진은 ChatGPT-4와 TinyLlama 모델을 비교 실험했습니다. 이를 통해 프로그램 도메인 내 각 모델의 일관된 응답성을 측정하고, 프로그래밍 어시스턴트로서의 성능을 평가했습니다. 이 접근 방식은 앞으로 프로그래밍 언어의 구조에 대한 새로운 통찰을 제공할 수 있을 것으로 기대됩니다.



### Associative Recurrent Memory Transformer (https://arxiv.org/abs/2407.04841)
Comments:
          ICML 2024 Next Generation of Sequence Modeling Architectures Workshop

- **What's New**: 이번 논문에서는 긴 시퀀스 (very long sequences)를 처리하는 데 필요한 새로운 뉴럴 아키텍처로 상수 시간(constant time) 내에 새로운 정보를 처리할 수 있는 Associative Recurrent Memory Transformer (ARMT)을 소개합니다.

- **Technical Details**: ARMT는 Transformer의 자가-주의(self-attention) 메커니즘을 지역 컨텍스트(local context)에 적용하고, 분절 수준의 리커런스(segment-level recurrence)를 통해 긴 컨텍스트에 걸쳐 분산된 작업 특화 정보를 저장합니다. 기존의 RWKV(Peng et al., 2023a)와 Mamba(Gu and Dao, 2023) 모델과 달리, ARMT는 완전한 지역 자가-주의와 일정한 시간 및 공간 복잡도를 제공합니다. 메모리 토큰은 선형 매핑(linear mapping)과 준-선형 키-값 메모리(quasi-linear key-value memory)를 통해 관리되며, 비선형성 함수(ϕ)를 사용하여 새로운 값을 메모리에 추가하고, 이전 값을 삭제하는 방식입니다.

- **Performance Highlights**: ARMT는 최근 BABILong 벤치마크에서 5000만 개의 토큰에 대한 단일 사실 질문(single-fact question)에 79.9%의 정확도로 새로운 성능 기록을 세웠습니다. 또한, 연결된 키-값 쌍(associative key-value pairs)을 사용한 메모리 용량 테스트에서도 RMT와 Mamba보다 우수한 성능을 발휘했습니다. 실험에서는 효율적인 비선형성 함수로 DPFP-3(Schlag et al., 2021)을 사용하였으며, 연속적인 삽입 및 삭제 작업에서 발생하는 정보 소실 문제도 해결하였습니다.



### Revisiting Structured Sentiment Analysis as Latent Dependency Graph Parsing (https://arxiv.org/abs/2407.04801)
- **What's New**: 본 논문에서는 Structured Sentiment Analysis (SSA)을 부분적으로 관찰된 종속 트리(dependency trees)로 보는 새로운 접근법을 제안합니다. 전통적인 SSA 방법의 한계를 극복하기 위해 내부 구조를 고려한 새로운 트리 구성을 도입했습니다.

- **Technical Details**: 제안된 방법은 두 스테이지의 파싱(parsing) 방법을 사용하며, TreeCRFs와 새로운 제약된 내부 알고리즘(constrained inside algorithm)을 통해 잠재적인 구조를 명시적으로 모델링합니다. 이는 그래프 아크(arcs)와 중심 스팬(headed spans)을 공동으로 스코어링하여 전역 최적화 및 추론을 가능하게 합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에서 진행된 광범위한 실험 결과, 제안된 방법이 기존의 이중-어휘적(bi-lexical) 방법들보다 현저하게 뛰어난 성능을 보여주며, 새로운 state-of-the-art를 달성했습니다.



### Toucan: Many-to-Many Translation for 150 African Language Pairs (https://arxiv.org/abs/2407.04796)
- **What's New**: 이번 논문에서는 아프리카 언어에 초점을 맞춘 저자원 언어(machine translation) 기계 번역을 개선하기 위해 다양한 자원을 소개합니다. 이 작업의 일환으로 두 가지 언어 모델인 Cheetah-1.2B와 Cheetah-3.7B를 각각 1.2억 개와 3.7억 개의 파라미터(parameter)를 가진 언어 모델로 개발하였습니다. 이를 기반으로 156개의 아프리카 언어 쌍을 지원하는 Afrocentric 기계 번역 모델 Toucan을 미세 조정(finetune)했습니다.

- **Technical Details**: Toucan 모델의 성능을 평가하기 위해 AfroLingu-MT라는 광범위한 기계 번역 벤치마크를 개발했습니다. 이는 아프리카 언어의 기계 번역 성능을 평가하기에 적합한 벤치마크입니다. 또한, 번역 평가 지표를 향상시키기 위해 1,000개의 언어, 그 중 614개는 아프리카 언어를 포함하는 spBLEU-1K 모델을 훈련했습니다.

- **Performance Highlights**: Toucan 모델은 다른 모델들보다 훨씬 높은 성능을 보였으며, 이를 통해 아프리카 언어에 대한 기계 번역이 크게 개선됨을 보여주었습니다. 이러한 성과는 특히 자원이 부족한 아프리카 지역에서의 문화 간 이해와 지식 교환을 촉진하는 데 기여할 것입니다.



### Re-Tuning: Overcoming the Compositionality Limits of Large Language Models with Recursive Tuning (https://arxiv.org/abs/2407.04787)
Comments:
          Accepted to ACL 2024

- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLM)이 조합적(compositional) 작업을 해결할 수 있는 새로운 방법을 제안합니다. 기존의 LLM은 다양한 언어 이해 작업에서 뛰어난 성능을 보였지만, 작은 인스턴스를 해결한 후 결합하여 해결해야 하는 조합적 작업에서는 어려움을 겪을 수 있습니다. 이를 해결하기 위해 재조정(Re-Tuning) 방법을 제안하며, 이는 문제를 하위 문제로 나누고, 이를 해결한 후 결과를 결합하는 방식입니다.

- **Technical Details**: 재조정(Re-Tuning) 방법은 문제를 재귀적으로 해결하는 과정을 포함합니다. 모델은 문제를 점점 더 작은 하위 문제(subproblem)로 나누고, 기본 경우(base case)를 직접 해결하며, 호출 스택(call stack) 상에서 하위 문제의 해결 결과를 전달하여 최종 결과를 도출합니다. 이 프로세스는 모델이 각 하위 문제의 맥락에 집중할 수 있게 하여 해결 정확도를 높입니다.

- **Performance Highlights**: 본 연구는 세 가지 대표적인 조합적 작업인 정수 추가, 동적 프로그래밍 문제, 패리티 문제(parity problem)에서 재조정 방법이 모델 성능을 크게 향상시킨다는 것을 입증했습니다. LLaMA 7B와 LLaMA 13B 모델에서 각각 37.4%와 31.9%의 성능 향상을 기록했으며, scratchpad 방법과 비교하여도 각각 34.5%와 36.7%의 평균 성능 향상이 있었습니다. 또한, GPU 메모리 사용량도 크게 절감되었습니다.



### AgriLLM: Harnessing Transformers for Farmer Queries (https://arxiv.org/abs/2407.04721)
Comments:
          Accepted at the Undergraduate Consortium at KDD 2024 (KDD-UC)

- **What's New**: 본 연구는 인도의 실제 농업 문의 데이터를 활용하여 대형 언어 모델(LLM)이 농업 분야에서 쿼리 해결을 자동화하는 데 어떻게 혁신적일 수 있는지를 탐구합니다. 특히, Tamil Nadu 주에서 수집된 약 4백만 개의 다양한 부문과 계절 작물 관련 쿼리를 분석하여 LLM의 잠재력을 보여줍니다.

- **Technical Details**: 연구는 Transformer 기반의 대형 언어 모델(LLM), 특히 Sequence-to-Sequence 모델을 활용하여 농업 분야의 쿼리를 자동으로 해결하는 방법을 탐구합니다. 이러한 모델은 자연어 처리(NLP)에 뛰어난 능력을 발휘하며, 주어진 프롬프트에 따라 일관되고 문맥적으로 적절한 텍스트를 생성할 수 있습니다. 또한, 사전 훈련된 LLM을 통해 답변의 구조와 문법을 개선합니다.

- **Performance Highlights**: 전통적인 키워드 매칭 및 사전 정의된 규칙에 의존하는 자동화된 쿼리 응답 시스템과 달리, LLM은 더 동적이고 이해하기 쉬운 답변을 제공할 수 있습니다. 이 모델은 농업 문의에 있어 높은 정확도와 문맥 이해력을 갖추고 있으며, 농부들이 직면하는 문제를 신속하게 해결하는 데 중요한 역할을 할 수 있습니다.



### Multi-Object Hallucination in Vision-Language Models (https://arxiv.org/abs/2407.06192)
Comments:
          Accepted to ALVR @ ACL 2024 | Project page: this https URL

- **What's New**: 이 논문에서는 큰 비전 언어 모델(LVLMs)이 여러 객체를 동시에 인식할 때 발생하는 '객체 환각(object hallucination)' 문제를 체계적으로 조사합니다. 기존의 평가 기준이 단일 객체 클래스에 중점을 두는 반면, 이 연구는 다객체 환각(multi-object hallucination) 현상을 탐구하며 Recognition-based Object Probing Evaluation (ROPE)라는 새로운 평가 프로토콜을 도입했습니다. ROPE는 테스트 동안 이미지 내 객체 클래스 분포를 고려하며, 시각적 언급 프롬프트를 사용하여 모호성을 제거합니다.

- **Technical Details**: ROPE는 자동화된 평가 프로토콜로, 블랙박스 신경망 모델이나 인간 평가자를 사용하지 않으며, 독특한 객체를 언급하는 시각적 프롬프트를 활용합니다. 이는 각 이미지 내 객체 클래스의 분포를 고려하며, 'In-the-Wild', 'Homogeneous', 'Heterogeneous', 'Adversarial'의 네 가지 하위 집합으로 나누어 평가를 진행합니다. 연구는 LVLMs의 규모와 훈련 데이터의 차이가 다객체 환각에 미치는 영향을 심층 분석했습니다. 주요 발견은 LVLMs가 단일 객체보다 여러 객체에 집중할 때 더 많은 환각을 겪는다는 것입니다.

- **Performance Highlights**: 연구 결과, LVLMs가 단일 객체에 비해 여러 객체에 집중할 때 환각이 더 심각하다는 것이 밝혀졌습니다. 테스트된 객체 클래스 분포가 환각 행동에 영향을 미치며, LVLMs는 지름길 또는 오류 상관관계를 따를 수 있다는 점을 시사합니다. 모델의 환각 행동은 데이터 특유의 요소, 중요도 및 빈도, 그리고 모델 고유의 행동에 의해 영향을 받는 것으로 나타났습니다.



### Vision-Language Models under Cultural and Inclusive Considerations (https://arxiv.org/abs/2407.06177)
Comments:
          HuCLLM @ ACL 2024

- **What's New**: 이번 연구는 대형 비전-언어 모델(VLMs)이 시각 장애인을 위한 비주얼 어시스턴트로 얼마나 문화적으로 민감하게 작동하는지를 평가하려고 합니다. 현재의 평가 데이터셋이 다양한 문화적 배경을 충분히 반영하지 못한다는 점을 보완하고자, 시각 장애인 사용자들이 촬영한 이미지를 포함하는 VizWiz 데이터셋을 사용하여 새로운 문화 중심 평가 기준을 제안했습니다.

- **Technical Details**: 우선 시각 장애인 사용자들을 대상으로 설문 조사를 실시하여 이미지 캡션에 문화적 정보를 포함하는 것에 대한 선호도를 조사했습니다. 그 후 VizWiz 데이터셋을 필터링하여 '문화적 개념'이 포함된 이미지를 선정했습니다. 이 데이터를 사용하여 최신 모델들의 이미지 캡션 성능을 평가했습니다. 자동 평가 메트릭스와 인간 평가를 통해 모델의 성능을 비교했습니다.

- **Performance Highlights**: 최신 모델들의 성능은 기대할 만했지만, 'hallucination'(환각) 문제와 자동 평가 메트릭스가 인간 판단과 일치하지 않는 문제를 발견했습니다. 특히, 사용자들은 문화적 세부사항을 더욱 중요시하며, 이를 반영한 캡션을 선호한다고 응답했습니다.

- **Conclusion**: 이 연구는 시각 장애인을 위한 시각 어시스턴트로서 VLMs가 더 포괄적이고 문화 인식적인 경험을 제공하기 위해 개선되어야 함을 시사합니다. 또한, 다양한 문화적 배경을 확실히 반영하는 데이터셋과 평가 기준이 중요함을 강조했습니다.



### On Speeding Up Language Model Evaluation (https://arxiv.org/abs/2407.06172)
- **What's New**: 이번 논문에서는 한정된 리소스 내에서 테스트 예제들을 평가하는 최적의 방법을 식별하는 도전에 대한 새로운 접근을 제안합니다. 다중 무장 강도(Multi-armed bandit) 프레임워크와 저랭크(low-rank) 행렬 분해를 결합한 방법을 통해, 자원 소요를 크게 줄이면서도 최적의 방법을 식별할 수 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 알고리즘을 포함합니다: UCB-E와 UCB-E-LRF입니다. 첫 번째 알고리즘인 UCB-E는 전통적인 상한 신뢰 구간(UCB, Upper Confidence Bound)을 확장하여 다음 평가할 방법-예제 쌍을 선택합니다. 두 번째 알고리즘인 UCB-E-LRF는 스코어링 행렬의 저랭크 특성을 이용하여, 예측되지 않은 방법-예제 쌍을 저랭크 행렬 분해(Low-Rank Factorization)로 예측하고 불확실성이 큰 쌍을 우선적으로 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 전통적인 방법에 비해 85-95%의 비용 절감을 이루었으며, 5-15%의 자원만으로 최적의 방법을 식별할 수 있었습니다. 또한, 방법 집합이 클수록 또는 최상위 방법과 다른 방법 간의 성능 차이가 작을수록 더 많은 예산이 필요했지만, UCB-E는 비교적 쉬운 설정에서, UCB-E-LRF는 어려운 설정에서 월등히 뛰어난 성능을 보였습니다.



### What's Wrong with Your Code Generated by Large Language Models? An Extensive Study (https://arxiv.org/abs/2407.06153)
Comments:
          17 pages, 7 figures

- **What's New**: 대규모 언어 모델(LLMs)을 사용한 코드 생성 연구가 활발하게 진행되고 있는 가운데, 이는 고품질 데이터 세트 수집과 다양한 훈련 기술 활용에 중점을 두고 있습니다. 그러나, 현재 방법론의 한계와 경계를 포괄적으로 분석한 연구는 부족합니다. 본 연구는 세 가지 주요 폐쇄 소스 LLM과 네 가지 인기 있는 오픈 소스 LLM을 세 가지 일반적인 벤치마크에서 평가하는 포괄적인 실증 연구를 수행했습니다. 이를 통해 복잡한 문제에 대해 성공적인 코드를 생성하는 데 어려움을 겪고 있으며, 표준 솔루션과 비교해 더 짧고 복잡한 코드를 생성하는 경향이 있음을 밝혔습니다.

- **Technical Details**: 연구는 HumanEval+, MBPP+, APPS+ 등 세 가지 널리 사용되는 벤치마크를 활용하여 1,164개의 프로그래밍 문제를 모았습니다. 실험에는 세 가지 폐쇄 소스 LLM (GPT-4, GPT-3.5, Claude-3)과 네 가지 오픈 소스 LLM (LLama-3-Instruct, Phi-3-Instruct, StarCoder-2, DeepSeekCoder)이 포함되었으며, 이들 모델은 단위 테스트를 통해 평가되었습니다. 연구는 코드의 길이, 복잡도(cyclomatic complexity), API 수 등 다양한 특성을 조사했습니다.

- **Performance Highlights**: 실험 결과, LLM들은 평균적으로 41.6%의 통과율을 기록했습니다. GPT-4와 Claude-3이 각각 63.8%와 56.7%로 최고 성능을 보였으며, Phi-3가 30.9%로 최저 성능을 보였습니다. 연구는 또한 오류 유형 분류(taxonomy)를 통해 LLM이 생성한 코드의 오류 원인을 심층 분석했습니다. 독창적인 방법으로, 추가 훈련 없이 자체 비판(self-critique)을 도입하여 생성된 코드를 반복적으로 수정하는 새로운 방법을 제안했습니다. 두 번의 반복 후 버그를 29.2% 감소시켜 통과율을 크게 향상시켰습니다.



### Infer Induced Sentiment of Comment Response to Video: A New Task, Dataset and Baselin (https://arxiv.org/abs/2407.06115)
- **What's New**: 기존의 비디오 멀티모달 감정 분석 연구는 주로 비디오 내 사람들의 감정 표현에 초점을 맞추었으나, 시청자가 동영상을 보면서 유발된 감정은 종종 간과되는 경우가 많습니다. 이 논문은 시청자의 유발된 감정을 추론하는 새로운 연구 과제, 즉 '비디오 유발 댓글 반응 다중 모드 감정 분석'(Multi-modal Sentiment Analysis for Comment Response of Video Induced, MSA-CRVI)을 소개합니다. 이를 위해 '마이크로 비디오에 대한 댓글 감정'(Comment Sentiment towards Micro Video, CSMV)이라는 데이터셋을 구축하였습니다. 이 데이터셋은 107,267개의 댓글과 8,210개의 마이크로 비디오로 구성되어 있으며, 총 68.83시간의 비디오 재생 시간을 포함하고 있습니다.

- **Technical Details**: MSA-CRVI는 비디오 콘텐츠 및 관련 댓글을 입력으로 사용하여 시청자의 유발된 감정을 이해하는 과제입니다. 이를 해결하기 위해 '비디오 콘텐츠 인식 댓글 감정 분석'(Video Content-aware Comment Sentiment Analysis, VC-CSA)이라는 베이스라인 방법을 제안하였습니다. VC-CSA는 '다중 스케일 시간 표현'(Multi-scale Temporal Representation), '합의 의미 학습'(Consensus Semantic Learning), '황금 특징 기반'(Golden Feature Grounding) 등 세 가지 주요 모듈로 구성되어 있습니다.

- **Performance Highlights**: 포괄적인 실험 결과, 제안된 VC-CSA 방법이 기존의 최첨단 멀티모달 감정 분석 방법들보다 높은 성능을 보이는 것으로 나타났습니다. 이는 비디오가 MSA-CRVI 과제에서 중요한 역할을 한다는 것을 강조합니다. 데이터와 소스 코드는 깃허브(https://github.com/AnonymousUserabc/MVI-MSA_DateAndSource.git)에 공개되어 있습니다.



### T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models (https://arxiv.org/abs/2407.05965)
- **What's New**: 최근 Sora의 개발은 텍스트-비디오(T2V) 생성의 새로운 시대를 열었습니다. 하지만 생성된 비디오에 불법적이거나 비윤리적 콘텐츠가 포함될 수 있는 보안 위험이 증가하고 있습니다. 이를 해결하기 위해, T2V 모델의 안전성을 평가하는 새로운 벤치마크인 T2VSafetyBench가 도입되었습니다. T2VSafetyBench는 12개의 중요한 안전 측면을 정의하고 LLMs와 'jailbreaking' 프롬프트 공격을 사용하여 악성 프롬프트 데이터셋을 구축합니다.

- **Technical Details**: T2VSafetyBench는 OpenAI, LLaMa-2, Anthropic의 사용 정책을 조사하고, AI 안전 전문가들을 대상으로 서베이를 통해 12가지 비디오 생성 안전 측면을 정의했습니다. 이러한 측면들은 포르노그라피, 경계선 포르노그라피, 폭력, 유혈, 공적 인물, 차별, 정치적 민감성, 불법 활동, 불쾌한 콘텐츠, 허위 정보, 저작권 및 상표 침해, 그리고 시간적 위험(Temporal Risk)입니다. 평가 방법으로는 LLMs와 다양한 프롬프트 공격을 통해 생성된 텍스트를 기반으로 악성 프롬프트를 생성한 후, GPT-4를 통해 평가를 진행합니다.

- **Performance Highlights**: 주요 발견점으로는 1) 단일 모델이 모든 측면에서 우수하지 않으며, 모델마다 강점이 다른 것으로 나타났습니다. 예를 들어, Stable Video Diffusion는 성적인 콘텐츠를 잘 막아내고, Gen2는 유혈 및 불쾌한 콘텐츠를 잘 관리하며, Pika는 정치적 민감성과 저작권 관련 경고에 강한 방어력을 보였습니다. 2) GPT-4의 평가와 수동 리뷰 간의 상관관계가 일반적으로 높으며, 대부분의 차원에서 상관계수는 0.8을 초과합니다. 3) 텍스트-비디오 생성 모델의 사용성과 안전성 사이에 절충이 존재합니다. 모델의 이해 및 생성 능력이 약할수록 안전성은 높아지지만, 이는 비디오 생성 기술이 발전함에 따라 안전 위험이 증가할 수 있음을 의미합니다. 따라서 비디오 안전에 대한 집중적인 관심이 요구됩니다.



### H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables (https://arxiv.org/abs/2407.05952)
Comments:
          13 pages, 14 tables, 9 figures

- **What's New**: H-STAR 알고리즘은 테이블 추출과 적응형 추론을 포함하여, 상징적(SQL) 및 의미적(텍스트 기반) 접근 방식을 통합하는 새로운 방법입니다. 이 접근법은 기존의 방법들보다 성능이 뛰어나며, 표 기반 질문 응답과 사실 검증 데이터셋에서 탁월한 성능을 보입니다.

- **Technical Details**: H-STAR는 테이블 추출과 적응형 추론의 두 단계로 복잡한 테이블 추론 작업을 분해합니다. 테이블 추출 단계에서는 step-by-step 'multi-view' 체인을 사용하여 열을 먼저 추출하고, 이어서 행을 추출합니다. 적응형 추론 단계에서는 질문의 유형에 따라 상징적(SQL) 또는 의미적(텍스트) 방법을 선택하여 추론합니다. 이 방법은 수량적 문제와 수학적 추론에는 상징적 방법을, 직접 조회와 복잡한 어휘 쿼리에는 의미적 방법을 사용합니다.

- **Performance Highlights**: 세 가지 표 기반 질문 응답과 사실 검증 데이터셋에서 H-STAR는 기존 최첨단 방법들보다 우수한 성능을 발휘했습니다. 실험 결과, H-STAR는 긴 테이블에서도 효율적이며, 'multi-view' 메커니즘을 사용한 정확한 테이블 추출과 뛰어난 적응형 추론 능력을 입증했습니다.



### Affordances-Oriented Planning using Foundation Models for Continuous Vision-Language Navigation (https://arxiv.org/abs/2407.05890)
- **What's New**: 최근 연구는 대규모 언어 모델(LLM)이 시각-언어 네비게이션(VLN)에서 뛰어난 제로샷 성능을 보여준다고 합니다. 그러나 기존 방법들은 주로 고차원의 작업 계획에 집중하여 실제적인 네비게이션 시나리오에서의 저수준 제어를 간과하고 있습니다. 이를 해결하기 위해, 저자들은 AO-Planner라는 새로운 프레임워크를 제안하였습니다. 이는 연속적인 VLN 작업을 위한 어포던스 중심의 계획 프레임워크로, 다양한 기반 모델을 통합하여 어포던스 지향의 이동 계획과 행동 결정을 제로샷 방식으로 수행합니다.

- **Technical Details**: AO-Planner는 SAM을 활용한 시각적 어포던스 프롬프팅(VAP) 접근법을 채택하여 네비게이션 어포던스를 제공하고, 그 기반 위에 LLM이 잠재적 웨이포인트를 선택하여 저수준 경로 계획을 생성합니다. 또한 높이 수준 에이전트인 PathAgent를 도입하여 가장 적절한 픽셀 기반 경로를 식별하고 이를 실현 가능한 3D 좌표로 변환합니다. 실험은 R2R-CE 벤치마크에서 수행되었으며, AO-Planner는 5.5%의 SPL 개선을 달성하여 제로샷 성능에서 최신 상태를 기록했습니다.

- **Performance Highlights**: AO-Planner는 데이터를 사용하지 않고 25.5%의 SR과 16.6%의 SPL을 달성하여 최첨단의 제로샷 성능을 보여주었습니다. 이는 기존의 최고 성능 방법보다 높은 성과입니다. AO-Planner는 시각적 어포던스를 통해 연결된 포인트 후보들을 활용하여 장애물을 피하면서 합리적인 경로를 계획하는 능력을 입증하였습니다.



### On the Limitations of Compute Thresholds as a Governance Strategy (https://arxiv.org/abs/2407.05694)
- **What's New**: 이번 에세이는 '컴퓨트 임계치(compute thresholds)'라는 굉장히 특이한 거버넌스 도구를 이해하는 데 초점을 맞추고 있지만, 그렇다고 해서 이 임계치가 어떤 성과를 가져올 것인지 알아보기 위해서는, 어떻게 이 개념이 나오게 되었는지를 이해해야 합니다. 최근 몇 년간 주요 AI 기업들이 책임 있는 확장 정책(responsible scaling policies)을 발표했으며, 이를 반영한 미국 백악관의 AI 안전 행정명령(AI Safety EO)과 유럽연합(EU)의 AI 법안에도 중요하게 사용되고 있습니다. 그러나 현재의 컴퓨트 임계치 설정이 단기적인 시각에 기반하고 있으며, 위험 완화에 실패할 가능성이 높다는 결론에 도달했습니다.

- **Technical Details**: 이 에세이는 컴퓨트 임계치가 일종의 정적 기준으로, 플로팅 포인트 연산(FLOP) 기준으로 높은 성능 시스템을 식별하는 데 사용된다는 점을 다루고 있습니다. 예를 들어, 미국 AI 안전 행정명령은 10^26 FLOP 이상의 컴퓨트 파워로 훈련된 모델을 위험 모델로 간주하여 추가적인 보고 절차와 검토를 요구하고 있으며, EU AI 법안은 이보다 더 정확한 10^25 FLOP 이상의 모델을 대상으로 하고 있습니다.

- **Performance Highlights**: 컴퓨트 임계치를 통해 더 큰 모델이 더 많은 유해 텍스트를 생성하거나, 데이터 유출 가능성이 높아지는 등 몇 가지 위험을 증폭할 수 있는 점은 확인되었습니다. 하지만, 현행 임계치 기준을 충족하는 현재 배포된 모델은 없으며, 불확실한 관계와 급변하는 컴퓨트와 위험의 관계를 이해하지 못하는 거버넌스는 실패할 가능성이 높습니다. 더 작으면서도 높은 성능을 보여주는 모델도 출현하고 있어, 이러한 임계치 설정 기준이 현 시점에서 적절치 않다는 것이 결론입니다.



### LLM-Based Open-Domain Integrated Task and Knowledge Assistants with Programmable Policies (https://arxiv.org/abs/2407.05674)
Comments:
          preprint

- **What's New**: KITA는 복잡한 사용자 상호작용을 처리할 수 있는 태스크 지향의 대화형 에이전트를 생성하는 프로그래머블 프레임워크입니다. 이 시스템은 개발자가 제공하는 정책을 신뢰할 수 있게 따르도록 설계되었습니다. KITA Worksheet는 선언적 패러다임(declarative paradigm)을 사용해 대화 흐름을 관리하고 통합 지식보조에 고급 지원을 제공하는 새로운 명세를 제안합니다.

- **Technical Details**: KITA Worksheet는 대화 상태를 추적하고 API 호출과 지식 쿼리의 조합을 통해 사용자 요청을 처리합니다. 구현은 Python-like syntax를 사용하여 API 호출(BookRestaurant)과 지식 쿼리(Answer)를 통합합니다. 이는 수익 흐름 관리, 태스크 조정 및 지식 질의를 표현하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실사용자 연구를 통해 KITA는 GPT-4 function-calling baseline보다 실행 정확도, 대화 행위 정확도 및 목표 달성률에서 각각 26.1, 22.5, 52.4 포인트 더 높게 나타났습니다. 연구 결과, KITA 에이전트는 3개의 다른 태스크 지향 대화 에이전트에서 91.5%의 실행 정확도, 91.6%의 대화 행위 정확도, 74.2%의 목표 달성률을 기록했습니다. 또한 22개의 실제 사용자 대화와 180개의 대화 턴들로 구성된 데이터셋도 공개되었습니다.



### Multi-label Learning with Random Circular Vectors (https://arxiv.org/abs/2407.05656)
Comments:
          11 pages, 6 figures, 3 tables; accepted to workshop RepL4NLP held in conjunction with ACL 2024

- **What's New**: 이 논문은 extreme multi-label classification (XMC) 작업을 위한 새로운 방법을 제안합니다. 이 방법은 각 벡터 구성 요소가 복소수 진폭(complex amplitude)으로 나타내어지는 랜덤 원형 벡터(random circular vector)를 활용하여 깊은 신경망(Deep Neural Networks, DNN) 모델의 출력 레이어와 손실 함수(loss function)를 개발하는 것을 목표로 합니다. 실험 결과, 원형 벡터는 정상적인 실수 벡터(real-valued vectors)보다 더 나은 라벨 인코딩 및 검색 능력을 가지고 있어, 모델의 출력 레이어 크기를 최대 99%까지 줄일 수 있습니다.

- **Technical Details**: 이 연구는 일정 각도 범위(−π to π)로 표현되는 원형 벡터의 각 요소를 사용하여 XMC의 출력 레이어를 원형 벡터로 조정하는 방법을 제안합니다. 기본 아이디어는 Holographic Reduced Representations(HRRs) 이론에 기초하며, 이는 고차원 출력 공간 벡터를 해당 데이터 인스턴스의 라벨 정보를 인코딩한 저차원 랜덤 벡터로 대체할 수 있습니다. 이후 생성된 벡터를 사용하여 라벨이 인코딩되었는지를 코사인 유사도를 통해 대략적으로 확인할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 원형 벡터 기반 방법은 실제 XMC 데이터셋에서 기존의 실수 벡터를 사용하는 모델보다 성능이 크게 향상되었습니다. 이 방법을 통해 출력 레이어 크기를 최대 99%까지 감소시키면서도 성능 개선을 이루었습니다.



### A Benchmark for Multi-speaker Anonymization (https://arxiv.org/abs/2407.05608)
- **What's New**: 개인정보 보호를 위한 음성 보호 기술은 주로 언어적 콘텐츠를 유지하면서 언어 외적 속성에서 발생하는 개인정보 정보를 억제한다는 점에서 기존 솔루션은 단일 화자 시나리오에 집중했다. 본 논문은 여러 화자 시나리오에 대한 익명화 벤치마크를 제공하기 위한 초기 시도로, 작업 정의와 평가 프로토콜을 설정하고, 벤치마크 해결책을 제안하며, 중첩 회화에서의 개인정보 유출에 대해 논의한다.

- **Technical Details**: 이 시스템은 화자 분할 (diarization) 기술을 사용하여 각 화자의 말을 집계하고, 화자 익명화 (speaker anonymization) 기술을 사용하여 화자의 개인정보를 숨기고 언어적 콘텐츠를 유지한다. 특히 대화 수준의 화자 벡터 익명화 방법 두 가지를 제안하여 원래 화자와 익명화된 화자 간의 연결을 불가하게 하면서도 대화 내에서 익명화된 화자 간의 구분을 개선한다. 첫 번째 방법은 원본 대화와 익명화된 대화에서 화자 쌍 간의 차이를 최소화하여 원래 화자 관계를 유지하고, 두 번째 방법은 익명화된 화자 간의 유사성을 최소화하여 화자 간 차별화를 얻는다.

- **Performance Highlights**: 비중첩 시뮬레이션 및 실제 세계 데이터셋에서 실험한 결과, 제안된 다중 화자 익명화 시스템이 효과적임을 보여주었다. 특히, 여러 화자가 있는 대화를 다루기 위해 개발된 계단식 시스템이 포함된 다양한 상황에서 화자 분할 및 익명화의 정확성을 증명했다. 또한, 중첩된 음성의 개인정보 유출 가능성에 대해 분석하고 잠재적 해결책을 제공하였다.



### On the Power of Convolution Augmented Transformer (https://arxiv.org/abs/2407.05591)
- **What's New**: Transformer 아키텍처는 언어 모델링에 혁신적인 발전을 가져왔지만, 최근 상태-공간 모델과 같은 아키텍처들이 성능 격차를 줄이고 있습니다. 이에 영감을 받아 회상, 복사, 길이 일반화 작업에서 Convolution-Augmented Transformer (CAT)의 이점을 분석합니다.

- **Technical Details**: CAT는 주의(attention) 계층의 K/Q/V 임베딩에 convolutional filters를 통합합니다. 이는 주의의 글로벌 뷰와 convolution의 지역성을 결합하여 시너지 효과를 냅니다. CAT는 하나의 계층만으로도 연결 회상(AR) 및 복사 작업을 해결할 수 있으며, 길이 일반화도 보장됩니다.

- **Performance Highlights**: CAT는 실제 데이터셋 평가에서 언어 모델링 성능을 향상시킵니다. 또한, 여러 가지 CAT 변형 버전들도 성능을 향상시키며, 긴 convolution을 사용하여 맥락 요약의 이점을 제공합니다. CAT는 다양한 실패 모드를 보완하여 완벽한 정확도와 길이 일반화를 유지합니다.



### Enhancing Hallucination Detection through Perturbation-Based Synthetic Data Generation in System Responses (https://arxiv.org/abs/2407.05474)
Comments:
          ACL 2024 findings

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs) 출력에서 발생하는 헬루시네이션(hallucination)을 자동으로 감지하는 새로운 접근법을 제안합니다. 이전의 방법들이 수동 주석 작업에 의존하거나 비싼 연산 자원을 요구하는 반면, 이번 연구에서는 재작성(rewriting) 시스템을 사용하여 자동으로 헬루시네이션을 생성하는 방식을 도입했습니다. 이를 통해 비용 효율적이고 신속하게 헬루시네이션을 감지할 수 있게 되었습니다.

- **Technical Details**: 이 연구에서는 T5-base 모델을 사용하여 자동 생성된 데이터셋으로 헬루시네이션을 감지하는 시스템을 학습시켰습니다. 재작성 LLM을 활용하여 신뢰할 수 있는 출력(faithful outputs)과 헬루시네이션된 출력(hallucinated outputs)을 생성하며, 이러한 접근법은 기존의 수동 주석 작업이 필요 없고, 새로운 LLM에 쉽게 적응할 수 있는 유연한 시스템을 제공합니다. 실험에서는 OpenDialKG와 BEGIN 데이터셋을 사용하여 T5-base 모델을 평가했습니다.

- **Performance Highlights**: 실험 결과, T5-base 모델은 최신 제로샷 헬루시네이션 감지기와 기존의 합성 생성 방법을 능가하는 정확성과 지연(latency)을 보였습니다. 특히, GPT-4 기반 기법보다 성능이 뛰어나고 최대 10배 빠르게 헬루시네이션을 감지할 수 있습니다. 이에 따라 제안된 접근법의 효율성과 효과가 입증되었습니다.



### Experiments with truth using Machine Learning: Spectral analysis and explainable classification of synthetic, false, and genuine information (https://arxiv.org/abs/2407.05464)
- **What's New**: 이 연구는 LLM(대형 언어 모델)의 발전에도 불구하고 여전히 해결되지 않은 잘못된 정보 문제를 심층 분석합니다. 연구에서는 잘못된(synthetic) 정보와 올바른(genuine) 정보를 스펙트럴 분석(spectral analysis), 시각화(visualization), 해석성(explainability) 측면에서 분석하여 그 이유를 찾아내고자 합니다.

- **Technical Details**: 다양한 데이터셋에서 정보를 표현하기 위해 여러 임베딩(embedding)기술이 사용됩니다. 스펙트럴 및 비스펙트럴 방법으로는 t-SNE(t-distributed Stochastic Neighbor Embedding), PCA(Principal Component Analysis), VAE(Variational Autoencoders)가 포함됩니다. 여러 머신러닝 알고리즘을 사용해 분류를 수행했습니다. 분류에 대한 설명을 위해 LIME(Local Interpretable Model-Agnostic Explanations), SHAP(SHapley Additive exPlanations), Integrated Gradients가 사용되었습니다.

- **Performance Highlights**: 분석 결과와 생성된 설명은 잘못된 정보와 올바른 정보가 매우 밀접하게 얽혀 있으며, 문헌상의 주장과는 달리 머신러닝 알고리즘이 이 두 가지를 효과적으로 분리하지 못한다는 것을 보여주었습니다.



### SBoRA: Low-Rank Adaptation with Regional Weight Updates (https://arxiv.org/abs/2407.05413)
Comments:
          15 pages, 2 figures

- **What's New**: 이번 논문은 대형 언어 모델(LLM: Large Language Models)의 새로운 파라미터 효율적인 미세 조정 방법인 Standard Basis LoRA (SBoRA)를 도입합니다. 이는 기존의 Low-Rank Adaptation (LoRA) 및 Orthogonal Adaptation 기법을 기반으로, LoRA의 계산 및 메모리 요구 사항을 더욱 감소시키면서 학습 성능을 향상시킵니다. SBoRA는 직교 표준 기저 벡터를 활용하여 저랭크 행렬 중 하나인 A 또는 B를 초기화합니다. 이를 통해 지역 가중치 업데이트와 메모리 효율적인 미세 조정을 가능하게 합니다.

- **Technical Details**: SBoRA는 두 가지 변형(SBoRA-FA와 SBoRA-FB)을 도입하여, 저랭크 행렬 중 하나만 업데이트합니다. 이에 따라 업데이트 행렬의 대부분의 행 또는 열이 0으로 구성되고, 이는 대다수의 미세 조정 모델 가중치가 사전 학습된 가중치에서 변경되지 않음을 의미합니다. 이 기법은 인간 두뇌의 모듈 조직화를 연상시키며, 이는 새로운 작업에 적응할 때 효율적으로 작동합니다. 추가로, SBoRA는 직교 표준 기저 벡터를 사용하여 프로젝션 행렬을 구성합니다. 이는 약 50%의 메모리 사용량 감소를 가져옵니다.

- **Performance Highlights**: SBoRA는 다양한 미세 조정 작업에서 LoRA보다 우수한 성능을 보였습니다. 특히 commonsense reasoning 및 arithmetic reasoning 작업에서 SBoRA-FA는 LoRA보다 각각 2.9% 및 2.8%의 성능 향상을 달성했습니다. 또한, 다양한 스케일의 양자화된 LLaMA 모델에서의 효율적인 적응을 강조합니다. QSBoRA는 QLoRA보다 MMLU 벤치마크에서 4.3%에서 6.9% 더 높은 성능을 나타냈습니다.



### Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation (https://arxiv.org/abs/2407.05361)
- **What's New**: 이번 논문은 	extit{Emilia}라는 다국어 음성 합성 데이터셋을 선보입니다. 이는 오픈 소스 전처리 파이프라인인 Emilia-Pipe를 사용해 야생 음성 데이터를 고품질의 학습 데이터로 변환하여 주석을 추가할 수 있게 합니다. Emilia는 101,000시간 이상의 6개 국어 음성을 포함하며 다양한 발화 스타일을 특징으로 합니다.

- **Technical Details**: Emilia-Pipe는 표준화, 소스 분리, 스피커 분할(Speaker Diarization), VAD(Voice Activity Detection)을 통한 정밀한 세분화, ASR(Automated Speech Recognition), 및 필터링의 6단계로 구성된 전처리 파이프라인입니다. 이 파이프라인은 Raw 음성 데이터를 빠르게 처리하여 고품질 학습 데이터로 변환합니다. 이 데이터셋은 WAV 형식으로 변환되고, 채널 및 샘플링 레이트가 표준화 되며, 볼륨과 why품을 조정하여 일관성 있는 데이터 형식을 유지하게 됩니다.

- **Performance Highlights**: 실험 결과, Emilia를 사용해 훈련된 모델들은 보다 자연스럽고 즉흥적인 인간과 유사한 음성을 생성하는데 효과적임이 입증되었으며, 다국어 TTS(Text-To-Speech)에서도 유망한 성능을 보였습니다. Emilia-Pipe는 8개의 NVIDIA RTX 4090 GPU를 사용해 1분당 2.50시간의 raw 음성 데이터를 처리할 수 있으며, 이는 대규모 음성 생성 연구를 위한 협업을 가능하게 합니다.



### VideoCoT: A Video Chain-of-Thought Dataset with Active Annotation Too (https://arxiv.org/abs/2407.05355)
Comments:
          ACL 2024 Workshop

- **What's New**: 최근 'Multimodal Large Language Models (MLLMs)'은 이미지 중심으로 발전해왔지만, 비디오에 대한 연구는 상대적으로 부족했다. 본 논문에서는 특히 'prompt engineering', 'video chain-of-thought (CoT)', 'instruction tuning' 등 비디오 관련 분야를 다루며, 비디오 CoT 데이터셋을 자동으로 생성하기 위한 도구를 개발하여 MLLMs의 추론 능력을 향상시키고자 한다. 이를 통해 VideoCoT, TopicQA, TopicCoT 세 개의 데이터셋을 기여하며 간단하지만 효과적인 벤치마크를 제안한다.

- **Technical Details**: 본 연구의 핵심은 'active learning' 패러다임 하에서 기계와 인간 전문가의 결합을 통한 자동 주석 도구를 개발한 것이다. 이를 통해 인간의 라벨링 작업량을 줄이고 데이터셋의 품질을 보장한다. 이를 위해 프롬프트 생성기를 훈련시켜 LLMs가 비디오 정보를 기반으로 복잡한 CoT를 생성하도록 유도하며, 생성된 CoT 문장을 여러 측면에서 평가하는 품질 점수를 수립한다. 또한 수정이 필요한 저품질 문장은 인간 전문가의 수정 후 프롬프트 생성기를 재훈련하여 더 합리적인 CoT를 생성한다.

- **Performance Highlights**: 비디오 CoT 데이터셋을 생성하기 위한 이 자동화 도구를 통한 실험 결과, VideoCoT, TopicQA, TopicCoT 데이터셋은 MLLMs의 추론 능력을 크게 향상시키는 것으로 나타났다. 또한, 이 데이터셋을 활용한 벤치마크 실험에서 본 솔루션의 효과가 입증됐다.



### Some Issues in Predictive Ethics Modeling: An Annotated Contrast Set of "Moral Stories" (https://arxiv.org/abs/2407.05244)
Comments:
          This project was a runner-up to the Novel Research prize for the BlueDot Impact course AI Safety Fundamentals. View my contrast set as JSONL, the UI used to generate it, and Emelin et. al."s initial paper and code at this https URL

- **What's New**: 이번 논문에서는 기존의 윤리적 딜레마를 도덕적 또는 비도덕적으로 라벨링하는 모델의 정확도에 대한 도전 과제를 다루고 있습니다. Moral Stories 데이터셋을 사용한 분류기의 성능 감소를 통해 텍스트 기반의 입력으로 도덕적 딜레마를 변환하는 과정에서 발생하는 문제들을 강조합니다.

- **Technical Details**: 상황의 서술적 내용을 3-5 단어 정도 소폭 수정해도 분류기의 정확도가 99.8%에서 51%로 크게 감소할 수 있습니다. 오도된 사회 규범과 상황을 연관시키는 경우 정확도가 98.8%로 낮아지며, 텍스트적 편향(예: 상황이 특정 라벨에 이미 맞다고 암시하는 경우)이 추가되면 정확도는 77%로 떨어집니다.

- **Performance Highlights**: 이 결과는 많은 윤리학 모델들이 과적합의 문제를 가지고 있을 수 있으며, 윤리적 딜레마를 정확하게 캡쳐하기 위해서는 여러 가지 주의 사항이 필요하다는 것을 시사합니다. 논문은 사회 규범의 구조를 재검토하고, 모델이 상황의 맥락에 대해 묻도록 훈련하며, 텍스트적 편향을 필터링하는 것을 권장합니다.



### TopoLedgerBERT: Topological Learning of Ledger Description Embeddings using Siamese BERT-Networks (https://arxiv.org/abs/2407.05175)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 논문은 회계 분야에서 오랫동안 해결되지 않았던 문제인 회사별 장부 계정을 표준화된 계정표로 매핑하는 문제를 다룹니다. 이를 위해 TopoLedgerBERT라는 새로운 문장 임베딩(sentence embedding) 방법을 제안합니다. 이 모델은 계정표의 계층적 정보를 문장 임베딩 과정에 통합하여, 장부 계정의 의미적 유사성과 계층 구조를 정확하게 포착하는 것을 목표로 합니다.

- **Technical Details**: TopoLedgerBERT는 계정표의 계층적 정보를 통합하여 문장 임베딩을 수행합니다. 이 방법은 문장의 의미적 유사성뿐만 아니라 계층적 구조도 포착합니다. 추가적으로, 데이터 증강(data augmentation) 전략을 도입하여 훈련 데이터를 풍부하게 만들고 모델의 성능을 향상시킵니다.

- **Performance Highlights**: TopoLedgerBERT는 정확도와 평균 역순위(mean reciprocal rank) 측면에서 벤치마크 방법들보다 뛰어난 성능을 보입니다.



### Solving for X and Beyond: Can Large Language Models Solve Complex Math Problems with More-Than-Two Unknowns? (https://arxiv.org/abs/2407.05134)
- **What's New**: 이번 논문에서는 기존의 수학 문제 해결 성능 기준이 너무 단순하여 참된 모델의 추론 능력을 평가하기 어렵다는 문제를 해결하기 위해 새로운 벤치마크 'BeyondX'를 소개합니다. 이 벤치마크는 다중 미지수 문제를 포함하여 모델의 복잡한 문제 해결 능력을 더욱 엄격하게 평가할 수 있게 설계되었습니다.

- **Technical Details**: BeyondX 벤치마크는 자동화된 파이프라인을 사용해 점진적으로 문제의 복잡성을 높여 여러 미지수를 포함하도록 설계되었습니다. 이 파이프라인은 세 가지 주요 아이디어를 바탕으로 합니다: 1) 시나리오 확장, 2) 점진적 외삽, 3) 문제 분해 생성. 새로운 미지수를 포함한 문제를 효율적으로 생성하기 위해 기존 문제 시나리오를 확장하고, 문제의 복잡성을 단계적으로 높이며 문제를 분해하여 생성하는 방법을 사용했습니다.

- **Performance Highlights**: BeyondX 벤치마크에 대한 평가 결과, 현재의 대형 언어 모델(LLMs)은 문제에 포함된 미지수의 수가 증가할수록 성능이 크게 저하됩니다. 예를 들어, GPT-4의 경우 최대 70%까지 성능이 감소하는 것을 확인했습니다. 이를 해결하기 위해 'Formulate-and-Solve' 전략을 제안하여, 여러 미지수를 포함한 문제를 효과적으로 처리할 수 있도록 하는 일반화된 프롬프트 기법을 개발했습니다. 이 전략은 기존의 전통적인 프롬프트 방법보다 성능을 크게 향상시키며, LLM의 실제 계산 한계를 보다 명확하게 이해할 수 있도록 도와줍니다.



### RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models (https://arxiv.org/abs/2407.05131)
- **What's New**: 최근 Medical Large Vision Language Models (Med-LVLMs)의 도입으로 의료 진단이 향상되었습니다. 그러나 기존의 Med-LVLMs는 종종 의료 사실과 일치하지 않는 응답을 생성하는 문제를 겪고 있습니다. 이러한 문제를 해결하기 위해 Retrieval-Augmented Generation (RAG) 방법이 도입되었으나, 과도한 검색으로 인해 관련이 없거나 부정확한 참조가 포함될 수 있으며, 모델이 본래 정답을 알고 있는 경우에도 잘못된 답변을 유도할 가능성이 큽니다. 이를 해결하기 위해 RULE을 제안합니다.

- **Technical Details**: RULE은 두 가지 주요 구성 요소로 구성됩니다. 첫 번째로, 검색 컨텍스트 수의 보정을 통해 사실 위험을 제어하는 전략을 도입합니다. 이를 통해 Med-LVLMs의 정확도를 추가 학습 없이 향상시킬 수 있습니다. 두 번째로, 검색된 컨텍스트에 과도하게 의존하여 오류가 발생한 샘플을 기반으로 선호 데이터셋을 생성하고 모델을 미세 조정하여 고유 지식과 검색된 컨텍스트의 균형을 맞춥니다. 이러한 접근법을 통해 RULE은 모델의 응답 정확도를 효과적으로 향상합니다.

- **Performance Highlights**: RULE은 세 가지 의료 VQA 데이터셋에서, 평균적으로 20.8%의 사실 정확도 향상을 달성했습니다. 추가적으로, 제안된 구성 요소의 유효성을 실험적으로 검증하며 RULE의 호환성을 입증했습니다.



### Ask Questions with Double Hints: Visual Question Generation with Answer-awareness and Region-referenc (https://arxiv.org/abs/2407.05100)
- **What's New**: 본 연구에서는 비주얼 질문 생성 (Visual Question Generation, VQG) 과제를 다루며, 텍스트 답변과 시각적 관심 영역을 제공하여 다양한 맵핑 문제를 해결하는 새로운 학습 패러다임을 제안합니다. 이를 통해 기존의 애매한 질문 생성을 극복하고, 참조가 가능하고 의미 있는 질문을 생성할 수 있습니다.

- **Technical Details**: 본 연구는 텍스트 답변 힌트와 시각적 관심 영역 힌트를 활용하여 질문을 생성하는 'Double Hints' 가이드를 제안합니다. 또한, 복잡한 관계 모델링 문제를 해결하기 위해 새로운 'Graph-to-Sequence' 학습 프레임워크 (Graph2Seq)를 제안하여, 동적 그래프로 시각적 객체 간의 암묵적 관계를 모델링하고 이를 통해 질문을 생성합니다.

- **Performance Highlights**: 제안된 모델은 VQA2.0 및 COCO-QA 데이터셋에서 기존의 최첨단 방법보다 높은 성능을 보였습니다. 추가 실험에서는 VQG가 VQA에서 데이터 증강 방법으로 사용될 때 훈련 데이터를 제한된 상황에서도 성능을 개선하는데 도움이 됨을 보여줍니다.



### LoRA-GA: Low-Rank Adaptation with Gradient Approximation (https://arxiv.org/abs/2407.05000)
- **What's New**: 큰 규모의 사전 학습된 모델을 미세 조정하는 것은 계산 및 메모리 비용이 매우 많이 듭니다. 인기 있는 매개변수 효율 미세 조정 방법인 LoRA는 보조 저차원 모델을 미세 조정함으로써 비용을 절감할 수 있습니다. 그러나 LoRA는 수렴 속도가 느려 전체 계산 비용 증가 및 성능 저하를 초래할 수 있습니다. 본 논문에서는 LoRA의 초기화 방법을 깊이 있게 조사하고, 새로운 초기화 방법인 LoRA-GA를 도입하여 효율성과 성능을 크게 향상시킬 수 있음을 입증합니다.

- **Technical Details**: LoRA-GA (Low Rank Adaptation with Gradient Approximation)는 첫 번째 단계에서 저차원 행렬 곱의 그래디언트를 전체 미세 조정의 그래디언트와 정렬함으로써 빠른 수렴을 가능하게 합니다. 기존 LoRA의 초기화 방식을 개선하여 저차원 행렬이 전체 가중치 행렬의 변화와 유사한 방향으로 초기화되도록 합니다.

- **Performance Highlights**: LoRA-GA는 T5-Base 모델을 사용한 GLUE 데이터셋에서 평균 5.69% 성능 향상을 보였고, Llama 2-7B 모델을 사용한 MT-bench, GSM8K, HumanEval에서 각각 0.34%, 11.52%, 5.05% 성능 향상을 기록했습니다. 또한, 기존 LoRA에 비해 2-4배 빠른 수렴 속도를 보여주어 로라-GA가 모델 성능과 수렴 속도를 동시에 개선할 수 있음을 검증했습니다.



### Conditional Semi-Supervised Data Augmentation for Spam Message Detection with Low Resource Data (https://arxiv.org/abs/2407.04990)
- **What's New**: 새로운 논문은 데이터 부족 문제를 해결하고 스팸 메시지 탐지를 수행하기 위해 조건부 준지도 데이터 증대(CSSDA) 기법을 제안합니다. 이 접근법은 비라벨된 데이터(두 번째 단락에 나오는 'unlabeled data')를 활용하여 훈련 데이터를 확장하는 특징이 있습니다.

- **Technical Details**: CSSDA의 주요 아키텍처는 'feature extraction(특징 추출)'과 'enhanced generative network(강화된 생성 네트워크)'로 구성됩니다. 이 생성 네트워크는 조건부 기법을 통해 비라벨된 데이터로부터 가짜 샘플인 'latent variables(잠재 변수)'를 생성합니다. 이러한 잠재 변수는 라벨된 데이터와 비라벨된 데이터 모두에서 나올 수 있으며, 최종 분류기의 입력으로 사용됩니다.

- **Performance Highlights**: 실험 결과, CSSDA는 다양한 양의 비라벨된 데이터를 사용할 때도 약 85%의 균형 잡힌 정확도(Balanced Accuracy)를 달성하며 강력한 모델로 입증되었습니다. 여러 가지 소거 연구(Ablation Study)도 수행되었으며, 이를 통해 제안된 혁신이 강화됨을 확인할 수 있었습니다. 결과적으로, 비라벨된 데이터가 조건부 준지도 기법에서 데이터 증대에 중요한 기여를 한다는 점이 밝혀졌습니다.



### LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts (https://arxiv.org/abs/2407.04973)
Comments:
          LogicVista benchmarks the logical reasoning of multimodal large language models in visual tasks

- **What's New**: 로지비스타(LogicVista)는 시각적 맥락에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 논리적 추론 능력을 검증하기 위한 평가 벤치마크입니다. MLLMs가 시 역할, 수학적 추론 등 다양한 능력을 보여주고 있지만, 논리적 추론에 대한 체계적인 평가가 부족한 상황입니다. 이를 해결하기 위해 9가지 논리적 추론 능력을 포함한 5가지 논리 추론 과제를 평가하는 448개의 객관식 질문 샘플을 사용하여 8개의 MLLM을 종합적으로 평가합니다. 이 평가 벤치마크는 직관적인 정량 분석과 전반적인 평점을 제공합니다.

- **Technical Details**: 로지비스타는 귀납적 추론, 연역적 추론, 수치적 추론, 공간적 추론, 기계적 추론 등 5가지 대표적인 논리 추론 과제를 다루며, 다이어그램, OCR, 패턴, 그래프, 테이블, 3D 형상, 퍼즐 등 다양한 형식을 포함합니다. 각 질문은 정답과 사람에 의해 작성된 이유가 주석으로 제공되어 MLLM의 개방형 및 객관식 평가가 가능합니다. 해당 벤치마크는 MLLMs의 논리 추론 능력을 더욱 포괄적으로 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 8개의 대표적인 오픈 소스 및 비공개 MLLM을 대상으로 다양한 논리 추론 과제에 대한 성능을 종합적으로 평가한 결과, 로지비스타의 평가 전략을 통해 각 모델의 성능을 세부적으로 분석할 수 있으며, 모델이 어떤 특정 기술에서 뛰어난지 또는 개선이 필요한지에 대해 더 많은 통찰력을 제공합니다. 이는 단순한 총평보다 더 깊이 있는 분석을 가능하게 합니다.



### OmChat: A Recipe to Train Multimodal Language Models with Strong Long Context and Video Understanding (https://arxiv.org/abs/2407.04923)
Comments:
          14 pages

- **What's New**: 새로운 OmChat 모델이 도입되었습니다. 이 모델은 긴 문맥과 비디오 이해 작업에 탁월한 성능을 발휘하도록 설계되었습니다. 새로운 아키텍처가 다양한 시각 입력을 표준화하여 처리 효율성과 적응력을 높이고 있습니다. 특히, OmChat은 동적 비전 인코딩 프로세스를 사용하여 다양한 해상도의 이미지를 효과적으로 처리하여 다양한 이미지 품질의 세부 사항을 포착할 수 있습니다.

- **Technical Details**: OmChat은 주도적이고 점진적인 멀티모달 사전 학습 전략을 사용하여 모델의 긴 문맥 처리 능력을 점차적으로 확장시키고 있습니다. 고품질 데이터를 선택적으로 사용하면서 학습하여 가장 관련성과 정보성을 갖춘 데이터 포인트로부터 학습합니다. OmChat의 새로운 전략에는 단일 이미지 텍스트, 다중 이미지 텍스트 및 비디오와 같은 복잡한 멀티모달 입력을 통합하는 핸들링 전략도 포함되어 있습니다. 또한, 'Temporal Visual Needle in a Haystack'이라고 불리는 새로운 벤치마크 데이터셋도 제안되었습니다. 이 데이터셋은 긴 비디오 시퀀스 내에서 시간적 시각 세부 정보를 이해하고 처리하는 OmChat의 능력을 평가합니다.

- **Performance Highlights**: OmChat은 여러 이미지 및 비디오 작업에서 대부분의 오픈 소스 모델을 능가하는 성능을 보여줍니다. 최대 512K 컨텍스트 길이를 지원하며, 단일 이미지 벤치마크에서도 경쟁력 있는 성능을 발휘합니다. 특히 고해상도 이미지 지원, 주도적 점진적 사전 학습 전략 및 고품질 지도학습 데이터셋이 OmChat의 성공 요인으로 작용했습니다.



### Algorithmic Language Models with Neurally Compiled Libraries (https://arxiv.org/abs/2407.04899)
- **What's New**: 이번 연구는 LLM (Large Language Models)의 알고리즘적 능력 부재 문제를 해결하기 위해, 기존의 높이 정교한 프로그램들과 기본적인 연산 집합을 추가하는 방안을 제안합니다. 이는 LLM이 알고리즘을 처음부터 학습할 필요 없이 기존 라이브러리를 이용해 더 효과적으로 알고리즘을 학습하도록 돕습니다. 이 논문에서는 LLaMA3 모델에 메모리, 레지스터, 기본 연산, 적응적 순환을 통합한 트랜스포머 아키텍처에 대한 초기 연구를 진행했습니다.

- **Technical Details**: 연구는 주어진 알고리즘을 뉴럴 네트워크의 파라미터로 바로 변환할 수 있는 Neural Compilation 기법을 정의했습니다. 이 방법을 통해, 뉴럴 네트워크가 알고리즘을 직접적으로 표현하고 최적화할 수 있습니다. 구체적으로, LLaMA3 모델에 메모리와 레지스터를 추가하고, 적응적 순환 메커니즘을 도입했습니다. 이 논문에서는 이러한 변경 사항이 작은 트랜스포머 모델에 대한 컴퓨팅 깊이가 다양한 알고리즘 작업 수행 시 어떤 성능을 보이는지 탐구했습니다.

- **Performance Highlights**: 본 연구는 기존의 LLM이 통계적 특성이나 다양한 숏컷(get around with shortcuts)을 통해 알고리즘을 대체하지 않고, 진정한 알고리즘적 능력을 갖추게 돕는 방법을 제시합니다. 네트워크 최적화가 주어진 알고리즘을 제대로 복구하지 못하는 한계를 뛰어넘기 위한 시도로, 알고리즘을 직접적으로 신경망 파라미터로 변환하는 기법을 이용해 더 나은 성능을 보이는 모델을 제안하고 있습니다.



### MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge for Text-to-Image Generation? (https://arxiv.org/abs/2407.04842)
Comments:
          42 pages, 13 figures, 33 tables

- **What's New**: 최근 텍스트-이미지 생성 모델(DALLE-3, Stable Diffusion 등)이 빠르게 발전하면서 환각(hallucination), 편향(bias) 및 안전하지 않거나 낮은 품질의 출력물 생성과 같은 문제에 직면하고 있습니다. 이러한 문제를 효과적으로 해결하기 위해 멀티모달 판사(multimodal judge)의 피드백을 기반으로 모델을 원하는 행동에 맞추는 것이 중요합니다. 이를 위해 MJ-Bench라는 새로운 벤치마크를 소개합니다. 이는 텍스트-이미지 생성 모델에 대한 피드백을 제공하는 멀티모달 판사들을 평가하기 위한 포괄적인 선호 데이터셋을 포함하고 있습니다.

- **Technical Details**: MJ-Bench는 텍스트-이미지 정렬(text-image alignment), 안전성(safety), 이미지 품질(image quality) 및 편향(generation bias)을 포함한 네 가지 주요 관점으로 멀티모달 판사들을 평가합니다. 각 관점은 더 작은 하위 카테고리로 세분화되어 포괄적인 평가를 가능하게 합니다. 평가에는 CLIP 기반의 소형 모델, 오픈 소스 VLM(예: LLaVA 패밀리), 및 폐쇄형 VLM(예: GPT-4o, Claude 3)이 포함되며, 이를 통해 다양한 데이터 포인트에 대해 멀티모달 판사의 피드백을 평가합니다.

- **Performance Highlights**: 실험 결과, 폐쇄형 VLM은 전체적으로 더 나은 피드백을 제공하며, 그 중 GPT-4o가 평균적으로 가장 우수했습니다. 오픈 소스 VLM에 비해 CLIP 기반의 소형 모델은 텍스트-이미지 정렬 및 이미지 품질 측면에서 더 나은 피드백을 제공하였고, VLM은 안전성 및 편향 측면에서 더 정확한 피드백을 제공했습니다. VLM 판사는 자연 언어(Likert-scale)로 표현된 피드백이 숫자형 피드백보다 더 정확하고 일관된 것으로 나타났습니다. 최종 결과는 인간 평가에서도 비슷하게 나타나므로 MJ-Bench의 효과를 다시 한번 확인할 수 있었습니다.



### On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks (https://arxiv.org/abs/2407.04794)
- **What's New**: 이 논문에서는 대형 언어 모델 (LLMs)이 생성하는 텍스트에 워터마크(watermark)를 적용하여 콘텐츠의 진위와 출처를 확인할 수 있는 방법을 제안합니다. 하지만, 워터마크의 제거 공격(watermark removal attacks)에 대한 내성 문제를 심층적으로 탐구하지 않았습니다. 이를 해결하기 위해, 주요 워터마킹 기법과 제거 공격을 체계적으로 분류하고, 다양한 시나리오에서 이들을 평가했습니다.

- **Technical Details**: 본 연구는 워터마킹 기법을 텍스트 생성 전단계(pre-text)와 후단계(post-text)로 나눴습니다. 총 8개의 워터마크(5개 pre-text, 3개 post-text)와 12개의 공격 기법(2개 pre-text, 10개 post-text)을 87개의 시나리오에서 평가했습니다. KGW와 Exponential 워터마크는 텍스트 품질과 워터마크 유지 측면에서 상대적으로 뛰어났지만, 대부분의 공격에 취약했습니다. 공격 기법의 효율성을 따졌을 때 post-text 공격이 pre-text 공격보다 실용적이었습니다.

- **Performance Highlights**: KGW와 Exponential 워터마크는 일부 공격에 대해 0.5 이상의 내성 점수를 기록했지만, 특정 공격(예: Paraphrase 공격)에서는 큰 취약성을 보였습니다. 또한, 공격 기법을 결합하면 효과가 크게 증가하는 것으로 나타났습니다. 예를 들어, Synonym 공격과 Modify 공격을 결합하면 KGW의 워터마크 유지율이 0.2365까지 떨어졌습니다. 전반적으로, 현재의 워터마킹 기술은 다양한 공격에 대한 내성이 부족하여 더 견고한 기술 개발이 필요합니다.



New uploads on arXiv(cs.IR)

### Academic Article Recommendation Using Multiple Perspectives (https://arxiv.org/abs/2407.05836)
- **What's New**: 이 논문에서는 Content-based filtering(CBF)와 Graph-based methods(GB)의 상호 보완적 특성과 이를 활용한 학술 검색 추천 시스템의 가능성을 탐구하였습니다. 저자들은 논문 초록을 사용하여 CBF가 저자의 입장을 추론하고, 인용을 사용하여 GB가 청중의 반응을 추론하는 방식을 제안합니다. 이 논문에서는 CBF와 GB의 9가지 차이점과 하이브리드 조합의 시너지 효과를 예시합니다.

- **Technical Details**: CBF는 주로 논문 초록과 제목을 활용하고, GB는 인용 그래프를 중심으로 구축됩니다. 두 가지 예시 임베딩으로는 BERT를 기반으로 한 깊은 신경망 인코딩을 사용하는 Specter와 2억개 이상의 논문과 20억개의 인용을 포함하는 Semantic Scholar 데이터의 분광 클러스터링(spectral clustering)을 사용하는 ProNE가 있습니다. 연구는 아름다운 학술 검색 데이터를 활용하여 CBF와 GB의 앙상블이 효율적이라는 것을 실험적으로 증명했습니다.

- **Performance Highlights**: 실험 결과, CBF와 GB의 앙상블이 각 기법을 단독으로 사용할 때보다 더 넓은 범위의 커버리지를 제공함을 보였습니다. 또한, 논문에서 제안한 모델들 중 Specter와 ProNE를 결합한 방식이 매우 효과적임을 확인했습니다. 이러한 방식은 특히 리뷰 논문 작성 시 'papers-like-this' 추천에 유용할 것으로 기대됩니다.



### Language Models Encode Collaborative Signals in Recommendation (https://arxiv.org/abs/2407.05441)
Comments:
          Codes are available at this https URL

- **What's New**: 최근 연구에 따르면 언어 모델(LMs)은 단순한 의미론을 넘어서는 풍부한 세계 지식을 인코딩한다고 합니다. 그러나 추천 영역에서는 사용자 선호도 정보를 암묵적으로 인코딩할 수 있는지에 대한 의문이 남아 있습니다. 이 연구는 기존의 이해와 달리, 언어 표현 공간에서 직접 추천 공간을 추출하는 가능성을 탐색합니다. 놀랍게도, 고급 언어 모델의 표현에서 아이템 표현을 선형적으로 매핑했을 때 뛰어난 추천 성능을 보였습니다. 이러한 결과는 언어 표현 공간과 효과적인 추천 공간 사이의 동형성을 시사하는 것으로, 협업 신호가 실제로 고급 언어 모델에 인코딩되어 있을 수 있음을 의미합니다.

- **Technical Details**: 이번 연구는 AlphaRec이라는 간단하지만 효과적인 협업 필터링(CF) 모델을 제안합니다. AlphaRec은 아이템의 텍스트 메타데이터(예: 제목)의 언어 표현을 활용하며, 다층 퍼셉트론(MLP), 그래프 컨벌루션, 대조 학습(CL) 손실 함수로 구성되어 있습니다. 아이템의 텍스트 제목에서 언어 표현을 추출하고, 선형 매핑을 통해 추천 공간으로 변환합니다. 이를 통해 추천 성능을 평가하며, 협업 신호가 언어 모델에 암묵적으로 인코딩되어 있을 가능성을 테스트합니다.

- **Performance Highlights**: AlphaRec은 여러 데이터셋에서 기존 ID 기반 CF 모델을 능가하는 성능을 보여줍니다. 특히, 새로운 도메인에서 zero-shot recommendation 능력이 뛰어나며, 언어 표현을 통해 사용자 의도를 인식할 수 있는 장점을 가지고 있습니다. AlphaRec은 단순하고 경량화되어 있으며, 빠른 수렴 속도를 자랑합니다. 또한, 사전 훈련된 언어 모델 임베딩을 사용하여 ID 기반 임베딩을 완전히 대체함으로써 처음으로 텍스트 임베딩만으로 이 수준의 성능을 달성했습니다.



### Towards Bridging the Cross-modal Semantic Gap for Multi-modal Recommendation (https://arxiv.org/abs/2407.05420)
- **What's New**: 이번 논문에서는 CLIP 기반의 새로운 다중 모달 추천 시스템 프레임워크인 CLIPER(Clip Enhanced Recommender)을 제안합니다. CLIPER는 모달 간 의미적 격차를 줄이고 다중 뷰에서 세밀한 의미 정보를 추출하는 방식을 채택하여 기존 다중 모달 추천 시스템들의 한계를 극복하고자 합니다.

- **Technical Details**: CLIPER는 모달리티 간 의미적 유사성을 측정하기 위해 multi-view modality-alignment 접근 방식을 도입합니다. 구체적으로, 세밀한 텍스트 설명을 필드 단위로 분할하고, 이러한 텍스트 필드를 프롬프트로 사용하여 시각적 및 언어적 표현을 추출합니다. 이를 통해 다중 뷰 유사성 측정 임베딩을 얻고, 다양한 백본 모델과 호환 가능한 모델 독립적 프레임워크를 제안합니다.

- **Performance Highlights**: 세 개의 공개 데이터셋에서 광범위한 실험을 통해, CLIPER가 최신 다중 모달 추천 모델들을 꾸준히 능가하는 성능을 보였습니다. 이는 모달리티 간 의미적 격차를 줄이고 세밀한 의미 정보를 효과적으로 통합한 결과입니다.



### Beyond Check-in Counts: Redefining Popularity for POI Recommendation with Users and Recency (https://arxiv.org/abs/2407.05360)
Comments:
          This paper is submitted to ICCA-2024

- **What's New**: 이 논문에서는 사용자의 과거 기록과 현재 상황을 기반으로 다음 방문할 관심 지점(POI, Point of Interest)을 예측하는 새로운 추천 시스템을 제안하였습니다. 기존 연구들은 주로 체크인 수를 통해 POI의 인기도를 정의했지만, 이 논문에서는 시간적 영향을 반영한 최신성 기반의 인기도 정의를 도입하였습니다. 이를 통해 최근 체크인 기록에 더 높은 가중치를 부여하여 POI의 인기도를 평가합니다.

- **Technical Details**: 이 논문에서는 인기도를 시간적 효과와 체크인 수, 체크인한 사람 수를 고려하여 재정의했습니다. 기존의 GETNext 알고리즘을 수정하여 사용자의 일반적인 선호도, 시공간적 맥락, 글로벌 전환 패턴, 시간 인지형 카테고리 임베딩을 결합한 그래프 향상 트랜스포머 모델을 구축하였습니다. 이 방법을 통해 사용자의 현재 궤적에 따라 다음 방문할 POI를 예측합니다. 데이터를 전처리한 후, 수정된 GETNext 알고리즘을 적용하여 최상위 k개의 POI를 추천합니다.

- **Performance Highlights**: 실험 결과, 제안된 최신성 기반 인기도 정의를 사용한 POI 추천 모델이 기존의 체크인 수 기반 인기도 정의를 사용한 모델보다 성능이 뛰어남을 확인했습니다. 이는 최근 체크인 기록이 더 많이 반영된 POI가 더욱 정확한 추천 결과를 제공한다는 것을 입증합니다.



### Understanding and Addressing Gender Bias in Expert Finding Task (https://arxiv.org/abs/2407.05335)
- **What's New**: 이번 연구는 커뮤니티 질문&답변(CQ&A) 플랫폼에서 매우 중요한 역할을 하는 전문가 찾기(EF) 작업에서의 성별 편향을 조사하고 이를 완화하는 방법을 탐구합니다. StackOverflow의 광범위한 데이터를 활용하여 최신 EF 모델에서 성별 편향이 어떻게 나타나는지 분석하고, 이를 줄이기 위한 조정을 제안합니다.

- **Technical Details**: 연구는 StackOverflow의 데이터를 사용하여 EF 모델의 후보 식별 과정이 성별 대표성에 미치는 영향을 실험적으로 분석하였습니다. 특정 모델이 명성 지표(reputation metrics)와 활동 수준(activity levels)을 기반으로 하여 남성 사용자를 더 많이 이전시킨다는 것을 발견하였습니다. 이를 해결하기 위해 콘텐츠 기반(content-based) 및 소셜 네트워크 기반(social network-based) 정보를 활용한 보다 균형 잡힌 전처리 전략을 제안합니다.

- **Performance Highlights**: 우리의 분석에 따르면, 제안된 방법을 통합하면 모델의 정확성을 손상시키지 않으면서 성별 균형을 크게 향상시킬 수 있습니다. 본 연구는 EF 방법에서 성별 편향을 감지하고 완화하는 데 초점을 맞춘 최초의 시도입니다.



### Ensemble Boost: Greedy Selection for Superior Recommender Systems (https://arxiv.org/abs/2407.05221)
- **What's New**: 이번 연구는 여러 추천 모델의 상위 k 추천을 결합하여 더 뛰어난 상위 n 추천을 생성하는 새로운 앙상블(Ensemble) 기법을 제안합니다. 이는 그리디 앙상블 선택(Greedy Ensemble Selection, GES) 전략을 활용하여 다양한 모델의 집단 지성을 효과적으로 활용합니다.

- **Technical Details**: 연구에서는 MovieLens-100k, MovieLens-1m, ciaodvd, hetrec-lastfm, 및 citeulike-a의 5개의 다른 데이터셋에서 실험을 수행하였습니다. 10개의 추천 모델—Implicit Matrix Factorization(I-MF), User-based KNN(U-KNN), Item-based KNN(I-KNN), Alternating Least Squares(ALS), Bayesian Personalized Ranking(BPR), Logistic Matrix Factorization(L-MF), Item-Item-Cosine Similarity(I-I-COSINE), Item-Item-TFIDF Similarity(I-I-TFIDF), Item-Item-BM25 Similarity(I-I-BM25), Popularity-based(PPL) 모델—을 사용하여 추천을 생성하고 이를 앙상블 기법으로 결합하였습니다. 각 모델은 NDCG (Normalized-Discounted Cumulated Gain) 지표를 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: NDCG@5, NDCG@10, NDCG@20 등의 메트릭을 통해 평가한 결과, 단일 최적 모델과 비교하여 평균 21.67%의 성능 향상을 나타내었습니다. 이 연구는 데이터셋 전반에 걸쳐 추천의 정확도를 현저하게 향상시켰고, 기존 모델과의 종합 비교에서도 높은 효능을 보였습니다.



### Consistency and Discrepancy-Based Contrastive Tripartite Graph Learning for Recommendations (https://arxiv.org/abs/2407.05126)
- **What's New**: 이번 연구에서는 전통적인 추천 시스템의 한계를 뛰어넘어 사용자 그룹과 아이템 번들을 추천하는 삼중 그래프 기반 추천 시스템에 대해 소개합니다. 이 시스템은 추천 대상 객체와 추천받는 객체 간의 미묘하고 암시적인 연관성을 포착하기 위해 일관성(consistency) 및 불일치(discrepancy)라는 두 가지 새로운 메타 경로 기반 메트릭을 활용하는 대조 학습 방법을 제안합니다.

- **Technical Details**: 이 연구는 Graph Convolutional Networks(GCN) 기반의 무한 계층 개념을 사용하여 일관성과 불일치라는 두 가지 메트릭을 계산하고, 다목적 최적화 프레임워크에 따라 모델 학습을 최적화합니다. 이러한 메트릭은 추천할 객체와 추천받는 객체의 고차원 유사성을 나타내며, 사전 계산된 메트릭은 대조 학습 손실(CD Loss)에 통합되어 직접 상호작용이 없는 경우에도 효율적인 노드 표현 학습을 가능하게 합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 CDR 모델의 우수성이 입증되었습니다. 제안된 일관성과 불일치 메트릭 및 대조 학습 기반 CD 손실이 모델의 성능향상에 중요한 역할을 한다는 것이 명확하게 입증되었습니다. 특히, 극도의 콜드 스타트 상황에서도 이 모델이 강력한 추천 성능을 발휘합니다.



### Preference Distillation for Personalized Generative Recommendation (https://arxiv.org/abs/2407.05033)
- **What's New**: 최근 연구자들은 대규모 언어 모델(Large Language Models, LLMs)을 사용한 생성형 추천 시스템의 가능성을 탐구하고 있습니다. PeaPOD(PErsonAlized PrOmpt Distillation)이라는 새로운 접근법을 제안하여, 사용자 선호도를 개인화된 소프트 프롬프트로 증류하는 방식을 도입합니다. 이는 복합적인 사용자 선호도를 반영하는 공유 가능한 학습 프롬프트 세트를 유지하며, 사용자 관심사에 따라 동적으로 가중치를 조정하여 사용자 개인화 프롬프트를 구성합니다.

- **Technical Details**: PeaPOD 모델은 사용자 선호도를 개인화된 소프트 프롬프트로 증류하는 새로운 아키텍처입니다. 사용자와 아이템 ID를 이산 프롬프트에 추가하는 기존 방식과 달리, PeaPOD는 사용자와 이들의 선호도 간의 관계를 보다 자연스럽게 학습할 수 있습니다. 만다라는 매트릭스 분해(matrix factorization)를 활용해 사용자의 상호작용 기록을 고차원 공간으로 압축하고, 공유된 프롬프트 구성 요소를 사용해 사용자별 프롬프트를 생성합니다. 사용자와 증류된 프롬프트 구성 요소 간의 다대다 매핑(many-to-many mapping)을 구성함으로써, 사용자 간의 유사성을 파악하고 공유 지식을 활용할 수 있습니다.

- **Performance Highlights**: 세 가지 실제 데이터 세트를 실험한 결과, PeaPOD 모델은 시퀀스 추천(sequential recommendation), 상위-N 추천(top-n recommendation), 설명 생성(explanation generation) 작업에서 최첨단 성능(state-of-the-art performance)을 달성했습니다. 연구 데이터와 코드는 https://github.com/jeromeramos70/peapod에서 제공됩니다.



### MemoCRS: Memory-enhanced Sequential Conversational Recommender Systems with Large Language Models (https://arxiv.org/abs/2407.04960)
- **What's New**: 대화형 추천 시스템(Conversational Recommender Systems, CRSs)은 사용자 선호도를 캡처하고 다중 라운드 자연어 대화를 통해 개인화된 추천을 제공합니다. 많은 기존 CRS 모델은 현재 대화 세션에서 선호도를 탐색하는 데 초점을 맞추며, 이전 대화 세션에서의 사용자 선호도를 간과합니다. 본 연구에서는 연속적인 LLMs(대형 언어 모델)를 활용하여 이 문제를 해결하고, 메모리 강화 대화형 추천 시스템 프레임워크(MemoCRS)을 제안합니다. 특히 사용자 특정 메모리와 일반 메모리로 구성되어 있으며, 사용자 특정 메모리는 개인화된 관심사를 반영하고, 일반 메모리는 협업 지식과 추론 지침을 포괄합니다.

- **Technical Details**: MemoCRS는 사용자 특정 메모리와 일반 메모리 두 가지로 구성됩니다. 사용자 특정 메모리는 각 사용자의 개인화된 관심사에 맞추어 엔터티 기반 메모리 뱅크(entity-based memory bank)를 사용하여 선호도를 정립하고 관련 메모리를 검색하여 이전 세션의 중복성과 노이즈를 줄입니다. 일반 메모리는 공동 지식과 추론 지침을 캡슐화하여, 특히 처음 사용하는 사용자들에게 공유된 지식을 제공합니다. 이를 통해 LLMs는 각 사용자에게 더 정확하고 맞춤형 추천을 제공할 수 있습니다.

- **Performance Highlights**: 실험 결과, 중국어와 영어 데이터셋 모두에서 MemoCRS의 효과가 입증되었습니다. Taobao와 MovieLens 데이터셋을 포함하여 다양한 환경에서 MemoCRS의 성능을 평가하였습니다. 또한, 실험적 평가에서는 사용자 프로필 기록, 훈련 세트, 검증 세트, 테스트 세트를 포함하여 다차원적 평가를 수행하였습니다. 결과적으로 MemoCRS는 기존의 최첨단 모델들과 비교하여 더 높은 유틸리티와 다양한 추천 성능을 보여주었습니다.



### RAMO: Retrieval-Augmented Generation for Enhancing MOOCs Recommendations (https://arxiv.org/abs/2407.04925)
Comments:
          7 pages, this paper underwent a rigorous review process and was officially accepted on May 31, 2024, for presentation at the Educational Data Mining 2024 Workshop: Leveraging Large Language Models for Next Generation Educational Technologies

- **What's New**: RAMO (Retrieval-Augmented Generation for MOOCs) 시스템은 'cold start' 문제를 해결하기 위해 설계되었습니다. 이 시스템은 대규모 언어 모델(LLMs)과 Retrieval-Augmented Generation (RAG)을 활용하여 맞춤형 교육 코스 추천을 제공하며, 대화형 인터페이스를 통해 학습 경험을 향상시키는 것을 목표로 합니다.

- **Technical Details**: RAMO 시스템은 Kaggle의 'Coursera Courses Dataset 2021'을 사용하여 개발되었습니다. 이 데이터셋은 코스 이름, 대학, 난이도, 평점, URL, 설명, 스킬을 포함해 약 3,342개의 코스 정보를 포함하고 있습니다. RAMO는 'prompt template' 을 이용해 사용자 히스토리 데이터가 부족한 경우에도 관련성 있는 추천을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: RAMO는 특별히 'cold start' 문제에 대한 최적의 솔루션을 제공하는 한편, 사용자와의 대화를 통해 맞춤형 코스를 추천하는 기능을 포함합니다. 기존의 추천 시스템들이 가지는 한계, 예를 들어, 개인화와 상호작용 부족 문제를 극복할 수 있는 가능성을 제시합니다.



### Transfer Learning with Self-Supervised Vision Transformers for Snake Identification (https://arxiv.org/abs/2407.06178)
Comments:
          Paper submitted to CLEF 2024 CEUR-WS

- **What's New**: 본 연구는 SnakeCLEF 2024 대회를 위한 접근법을 제시하며, Meta의 DINOv2 Vision Transformer 모델을 활용하여 이미지에서 뱀 종을 예측합니다. 182,261개의 이미지로 구성된 데이터셋에서 고유종의 높은 가변성과 시각적 유사성을 해결하기 위해 DINOv2 모델을 피처 추출을 위한 주요 도구로 사용하였습니다.

- **Technical Details**: 연구에서는 고유종의 시각적 차이를 이해하기 위해 DINOv2 임베딩을 탐색하고, 이를 기반으로 선형 분류기를 훈련시켰습니다. DINOv2는 Meta에서 개발한 자체 지도 Vision Transformer로, 다양한 시나리오에서 이전에 본 적 없는 뱀 사진을 다루기에 이상적입니다. 데이터셋은 iNaturalist와 Herpmapper 플랫폼에서 수집된 1,784종의 182,261개 이미지로 구성되어 있습니다.

- **Performance Highlights**: 알고리즘은 39.69 점수를 획득하였으며, 이는 뱀 종 식별에 있어 DINOv2 임베딩의 가능성을 보여줍니다. 이 외에도 [CLS] 토큰을 이용한 시각적 군집화 분석이 효과적이라는 결과를 보여주었습니다. 모든 프로젝트 코드는 공개되어 있습니다.



### A Survey of Controllable Learning: Methods and Applications in Information Retrieva (https://arxiv.org/abs/2407.06083)
- **What's New**: 컨트롤러블 러닝(Controllable Learning, CL)은 신뢰할 수 있는 머신러닝의 중요한 요소로 등장하고 있으며, 이를 통해 학습자가 사전 정의된 목표를 충족하고 이러한 목표의 변화에 따라 재학습 없이 적응적으로 조정할 수 있습니다. 이 논문에서는 CL의 공식 정의를 제공하고, 정보 검색(Information Retrieval, IR) 분야에서의 적용 사례를 논의합니다.

- **Technical Details**: CL은 누가 제어하는지(사용자 또는 플랫폼), 무엇이 제어 가능한지(예: 검색 목표, 사용자의 과거 행동, 제어 가능한 환경 적응), 제어 방법(예: 규칙 기반 방법, Pareto 최적화, 하이퍼네트워크), 제어를 어디에서 구현할 것인가(전처리, 인-프로세싱, 후처리 방법)에 따라 분류됩니다.

- **Performance Highlights**: 다중 목표 제어(Multi-Objective Control), 과거 행동 제어(Historical Behavior Control), 환경 적응 제어(Controllable Environmental Adaptation) 등 다양한 제어 목표를 달성하기 위해 다양한 기술들이 사용됩니다. 대표적으로 규칙 기반 후처리 기법, Pareto 최적화, 하이퍼네트워크 기반 방법 등이 있으며, 각각의 방법들은 특정 사례에서 다양한 성능 지표를 현실화하는 데 기여합니다.



### MERGE -- A Bimodal Dataset for Static Music Emotion Recognition (https://arxiv.org/abs/2407.06060)
Comments:
          16 pages, 4 figures, 13 tables, submitted to IEEE Transactions on Affective Computing

- **What's New**: 최근 음악 감정 인식(MER) 분야는 오디오 중심 시스템에서 오디오와 가사를 결합한 이중 모달리티(en: bimodal) 합주로 전환되었습니다. 그러나 공개되고 충분한 크기의 이중 모달리티 데이터베이스 부족으로 인해 이러한 시스템의 개발이 저해되었습니다. 이를 해결하기 위해 이 논문에서는 자동화된 방법으로 생성된 세 가지 새로운 오디오, 가사 및 이중 모달리티 MER 연구 데이터셋, MERGE를 제안합니다.

- **Technical Details**: MERGE 데이터셋은 감정 인식에 사용되는 멀티모달 데이터셋으로, 고정된 train-validate-test 분할 구조를 따릅니다. 이 데이터셋들을 평가하고 벤치마크 지표를 세우기 위해, 특징 공학(en: feature engineering), 머신 러닝, 딥 러닝 방법론을 사용하여 여러 실험을 수행했습니다. 특히, Russell의 감정 주기(en: emotion quadrants)를 기반으로 한 감정 인식 방법론을 적용했습니다.

- **Performance Highlights**: 가장 성능이 좋은 딥 뉴럴 네트워크 모델이 오디오 및 가사 결합 이중 모달리티 분류에서 79.21%의 F1-score를 달성했습니다. 이 결과는 제안된 데이터셋이 MER 연구의 벤치마킹에 유효함을 확인해 줍니다.



### New Directions in Text Classification Research: Maximizing The Performance of Sentiment Classification from Limited Data (https://arxiv.org/abs/2407.05627)
Comments:
          9 pages, in Indonesian language. intro to a shared task in sentiment classification

- **What's New**: 이번 논문에서는 Kaesang Pangarep의 PSI(Chairman of PSI) 임명 이슈에 대한 감정 분석 문제를 다루고 있습니다. 특히, 300에서 600 샘플 정도의 제한된 학습 데이터를 사용한 텍스트 분류 문제를 다룹니다. 이 논문은 감정 분석에서 빠른 속도와 높은 정확도라는 이해관계자의 요구를 만족시키기 위해 외부 데이터를 활용한 집계 및 증강을 제안합니다.

- **Technical Details**: 본 연구는 텍스트 분류를 위해 제공된 벤치마크 데이터셋을 사용합니다. 추가로 Covid Vaccination 감정 데이터셋과 오픈 토픽(Open Topic)을 포함한 두 가지 외부 데이터를 활용합니다. 공식적인 평가 지표로는 F1-score가 사용되며, 이는 긍정, 부정, 중립이라는 세 개의 클래스에서 precision과 recall을 균형있게 평가합니다. 초기 기준 점수와 최적화된 점수 모두 SVM(Support Vector Machine) 방법을 사용하여 계산됩니다.

- **Performance Highlights**: 기준 점수와 최적화된 점수 모두 SVM 방법으로 계산되었으며, 기준 F1-score는 40.83%, 최적화된 F1-score는 51.28%를 달성했습니다.



### Faux Polyglot: A Study on Information Disparity in Multilingual Large Language Models (https://arxiv.org/abs/2407.05502)
- **What's New**: 최근 연구에서는 Retrieval Augmented Generation (RAG) 기법을 활용한 대형 언어 모델(LLMs)의 다언어 정보 검색에서의 언어적 편향을 조사했습니다. 연구 결과, LLMs는 쿼리 언어와 동일한 언어로 된 정보를 선호하는 경향이 있으며, 특히 정보가 부족한 상황에서는 고자원 언어로 작성된 문서를 선호한다고 밝혔습니다. 이러한 경향은 사실 기반 및 의견 기반 쿼리 모두에서 나타났습니다.

- **Technical Details**: 이 연구에서는 5개의 언어(영어, 힌디어, 독일어, 아랍어, 중국어)로 구성된 170개의 문서를 포함한 다언어 합성 데이터셋을 생성하였습니다. 연구팀은 Retrieval Augmented Generation (RAG) 방식에서 언어 모델의 정보 선호도를 연구했으며, RAG 방식에서는 정보 검색과 생성 두 단계를 거칩니다. 검색 단계에서는 문서 임베딩과 쿼리 임베딩의 코사인 유사도에 기반해 연관된 문서를 검색하고, 생성 단계에서는 이 문서들을 컨텍스트로 사용해 응답을 생성합니다.

- **Performance Highlights**: 실험 결과, 현재 다언어 LLMs는 쿼리 언어와 동일한 언어로 된 문서를 일관되게 선호하는 경향을 보였습니다. 이러한 선호도는 사실 기반 및 의견 기반 쿼리 모두에서 확인되었습니다. 쿼리 언어로 된 관련 문서가 없는 경우, LLM은 저자원 언어보다 고자원 언어로 된 문서를 선호합니다. 이는 다언어 LLM이 정보 접근성을 민주화하려는 목표에 반해 언어적 정보 격차를 강화할 가능성을 시사합니다.



### MelodyVis: Visual Analytics for Melodic Patterns in Sheet Music (https://arxiv.org/abs/2407.05427)
Comments:
          9+2 pages, 9 figures, preprint, originally submitted to IEEE VIS 23, revision

- **What's New**: MelodyVis는 음악학 전문가와 협력하여 설계된 시각적 애플리케이션으로, 디지털 악보에서 멜로디 패턴을 탐색할 수 있도록 해줍니다. 이 시스템은 멜로디 반복과 변형을 포착하는 8개의 기본 연산자(operators)를 사용하며, 사용자는 악보에서 직접 패턴을 선택하고 상호작용을 통해 다른 패턴을 식별할 수 있습니다. 이 연구에서는 MelodyVis가 멜로디 분석 작업을 효과적으로 지원하는지 평가하기 위해 사용자 연구(user study)를 수행했습니다. 그 결과, 연산자를 활성화한 참가자들이 최소 두 배 더 많은 패턴을 식별할 수 있음을 확인했습니다.

- **Technical Details**: MelodyVis는 멜로디 연산자 그래프(Melody Operator Graph)와 보이싱 타임라인(Voicing Timeline)을 포함한 5개의 연결된 뷰(views)를 특징으로 합니다. 이 시스템은 전이(transposition)와 미러링(mirroring)과 같은 8개의 원자 연산자를 사용하여 멜로디 반복과 변형을 캡처합니다. 사용자는 악보 뷰에서 패턴을 선택하고, 선택된 샘플 기반으로 다른 패턴을 확인할 수 있습니다.

- **Performance Highlights**: 사용자 연구 결과, MelodyVis를 이용한 참가자들이 연산자를 활성화한 상태에서 최소 두 배 더 많은 멜로디 패턴을 식별할 수 있음을 보여주었습니다. 참가자들은 MelodyVis가 패턴 식별과 해석 능력을 향상시키는 데 도움이 된다고 보고했습니다.



### Multimodal Language Models for Domain-Specific Procedural Video Summarization (https://arxiv.org/abs/2407.05419)
Comments:
          6 pages, 3 figures

- **What's New**: 최근 영상 튜토리얼, 특히 요리와 의료 절차와 같은 특정 도메인에서의 장기간 영상 요약과 단계별 지침 생성을 최적화하기 위한 멀티모달 모델이 소개되었습니다. 이 연구는 TimeChat을 도메인 특화 데이터셋에 맞게 미세 조정(fine-tuning)하여 성능을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 도메인 특화 데이터셋(Tasty for cooking, MedVidQA for medical procedures)으로 TimeChat 모델을 미세 조정하여 주요 절차적 활동을 간결하게 요약할 수 있도록 했습니다. TimeChat은 매 프레임의 시각적 컨텍스트와 타임스탬프(description)를 결합하는 'sliding video Q-Former' 메커니즘을 사용합니다. 이 메커니즘은 비디오 길이에 따라 동적으로 비디오 토큰 시퀀스를 생성합니다.

- **Performance Highlights**: 도메인 특화 절차 데이터를 사용한 미세 조정 실험 결과, TimeChat은 장기간 비디오의 주요 지침 단계를 추출하고 요약하는 능력이 크게 향상되었습니다. 특히 요리와 의료 분야에서 맞춤형 단계별 지침 생성을 성공적으로 수행할 수 있음을 확인했습니다.



### A Survey of Datasets for Information Diffusion Tasks (https://arxiv.org/abs/2407.05161)
- **What's New**: 이번 논문은 정보 확산(information diffusion) 연구를 저명한 '커뮤니케이션의 5W 모델(Five W's of Communication model)'을 기반으로 체계적으로 분류하고, 각 작업에 맞춘 데이터셋을 통합하여 소개합니다. 정보 확산 예측(information diffusion prediction), 소셜 봇 탐지(social bot detection), 및 허위 정보 탐지(misinformation detection)의 세 가지 주요 작업을 정의하고 각각 10개의 세부 작업으로 세분화하였습니다.

- **Technical Details**: 기존의 연구 및 데이터셋을 체계적으로 분류하기 위해 '5W 모델' 프레임워크를 사용하여 정보 확산 작업을 세 가지 주요 작업으로 분류하였습니다. 각 주요 작업은 다시 10개의 하위 작업으로 나뉘며, 관련 데이터셋의 소스와 URL도 함께 제공됩니다. 데이터셋에는 사용자 정보, 소셜 네트워크, 봇 레이블, 확산 내용, 확산 네트워크, 진위 레이블 등 6가지 속성이 포함되어 있으며, 이를 기반으로 데이터셋을 비교 분석했습니다.

- **Performance Highlights**: 발표된 데이터셋 리포지토리는 현재 연구에서 공개적으로 이용 가능한 57개의 데이터셋을 포함하고 있으며, 정보 확산 연구를 위한 중요한 자료로 제공됩니다. 데이터셋의 한계점과 이후 연구 방향을 논의하여 정보 확산 연구 분야의 발전을 위한 제언도 포함하고 있습니다.




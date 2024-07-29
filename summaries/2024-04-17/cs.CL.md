### MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents (https://arxiv.org/abs/2404.10774)
Comments: LLM-AggreFact benchmark, MiniCheck models, data generation code at this https URL

- **What's New**: 이 연구에서는 LLM(Large Language Models) 결과물의 증거와 연관성을 평가하는 데 필수적인 문제를 해결하기 위해 저렴한 비용으로 GPT-4 수준의 성능을 구현할 수 있는 새로운 작은 모델, MiniCheck-FT5를 개발했습니다. 이 모델은 GPT-4로 생성한 합성(Synthetic) 훈련 데이터를 사용하여 사실 오류의 현실적인 예시를 생성하고, 이를 통해 문장 사이의 정보 통합까지 인식해내는 모델을 훈련시켰습니다.

- **Technical Details**: MiniCheck 시스템은 Flan-T5를 기반으로 하며, 주어진 문장에 포함된 각 사실을 문서와 대조하여 검증하는 기능을 학습합니다. 새로운 벤치마크인 LLM-AggreFact는 폐쇄형 책(closed-book) 및 증거에 기초한(evidence-based) 생성 작업을 위한 10개 데이터셋을 통합하여, 사실적 오류를 인간의 주석으로부터 평가합니다. 합성 데이터 생성과 훈련 방법은 모델이 고성능을 유지하면서도 추론 시간과 비용을 획기적으로 줄이는 데 중점을 둡니다.

- **Performance Highlights**: MiniCheck-FT5 모델은 파라미터가 770M인 상태로, 같은 크기의 다른 시스템보다 우수한 성능을 보이고 GPT-4와 동등한 정확도를 달성하였습니다. 또한, 세분화된(fine-tuned) 기존 시스템에 비해 4%에서 10% 높은 절대값으로 성능이 우수하며, 검증 비용은 400배 낮습니다. 본 연구를 통해 코드, 데이터 생성 도구, 그리고 모델이 공개되었습니다.



### Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study (https://arxiv.org/abs/2404.10719)
Comments: 16 pages, 2 figures, 14 tables

- **What's New**: 이 논문은 인간 피드백에서의 강화 학습(Reinforcement Learning from Human Feedback, RLHF)에 집중하고 있으며, 특히 보상 기반 방법과 보상 없는 방법 중 어느 것이 더 우수한지에 대해 조사합니다. 가장 널리 사용되는 보상 기반 방법인 PPO(Proximal Policy Optimization)와 보상 없는 방법 중 하나인 DPO(Direct Preference Optimization)를 비교 분석하였습니다. 결과적으로 PPO가 모든 테스트 베드에서 DPO를 뛰어넘는 성능을 보였으며, 특히 코드 생성 경쟁에서 최첨단 결과를 달성하였습니다.

- **Technical Details**: PPO는 선호 데이터를 이용하여 보상 모델을 학습하고 이를 최적화하는 액터-크리틱(Actor-Critic) 알고리즘을 사용합니다. 반면에, DPO는 보상 함수의 명시적 사용을 배제하고 오직 정책 최적화에만 집중하는 방법입니다. 본 논문에서는 DPO가 특정 조건에서 편향된 해결책을 찾을 수 있다는 이론적 한계를 발견하고, 실제 실험을 통해 DPO의 성능이 모델 출력과 선호 데이터 세트 간의 분포 변화에 크게 영향을 받는 것을 확인하였습니다.

- **Performance Highlights**: 실제 실험을 통해 PPO는 대화 생성 및 코드 생성 작업을 포함한 다양한 RLHF 테스트 베드에서 일관되게 DPO를 능가하는 성능을 보였습니다. 특히, 코드 경쟁 데이터셋(CodeContest)에서 34B 파라미터를 갖는 PPO 모델이 AlphaCode-41B를 상회, 상위 1k에서 16.4%에서 22.4%로 개선된 결과를 달성했습니다.



### Dual Modalities of Text: Visual and Textual Generative Pre-training (https://arxiv.org/abs/2404.10710)
- **What's New**: 이 논문에서는 4억 개 이상의 문서를 RGB 이미지로 렌더링하여 픽셀 기반 자동회귀 언어 모델을 위한 새로운 사전 훈련 프레임워크를 도입했습니다. 이는 시각적 데이터와 텍스트 데이터를 모두 활용하는 이중 모달리티 훈련 방식을 특징으로 하며, 시각적 텍스트의 전체적인 이해와 이를 기반으로 한 언어 모델링의 가능성을 확장합니다.

- **Technical Details**: 이 연구는 이미지(Images) 형태의 텍스트 데이터를 직접 학습하는 픽셀 기반 언어 모델링(Pixel-based Language Modeling)에 중점을 두었습니다. 특히 RGB 이미지에서의 순수 픽셀 기반 자동회귀 사전 훈련의 가능성을 탐구하고, 이 방법이 다언어 작업에서 어휘 병목 현상(Vocabulary Bottleneck)을 어떻게 극복하는지, 그리고 텍스트 기반 사전 훈련과의 시너지 효과 등을 분석했습니다. 연구팀은 비트(Bit)와 그레이스케일(Grayscale) 대신 24비트 색상 깊이의 RGB를 사용하여 텍스트를 렌더링하고, 다음 토큰 및 패치 예측을 위한 분류 및 회귀 헤드를 적용하는 새로운 자동회귀 사전 훈련 방식을 소개합니다.

- **Performance Highlights**: 이 모델은 언어 이해 벤치마크에서 인코더 기반 모델을 능가하거나 일치하는 성능을 보여주었습니다. RGB 이미지를 사용하는 디코더 전용 아키텍처는 복잡한 언어 패턴의 이해를 향상시킬 뿐만 아니라, 언어 간의 지식 전달을 효과적으로 촉진합니다. 또한, 이 연구는 시각적 데이터와 텍스트 데이터의 통합이 언어 모델 훈련을 더욱 향상시킬 수 있는 큰 잠재력을 보여줍니다.



### Question Difficulty Ranking for Multiple-Choice Reading Comprehension (https://arxiv.org/abs/2404.10704)
Comments: 7 pages, 3 figures

- **What's New**: 이 연구는 다지선다형 (Multiple-choice, MC) 영어 문제의 난이도를 자동으로 순위 매기는 방법을 탐구합니다. 특히, 제한된 데이터 상황에서 문제 난이도를 평가하기 위해 'task transfer'와 'zero-shot' 접근 방식을 비교 분석하였습니다. 직관적으로 난이도를 파악하는 것보다 비교를 통한 접근이 더 효과적임을 발견하였으며, 이러한 결과는 특히 언어 학습 데이터에서 공개적으로 사용 가능한 데이터를 기반으로 합니다.

- **Technical Details**: 'Task transfer' 접근 방식에서는 수준 분류 (level classification)와 독해력 평가 (reading comprehension) 시스템을 사용하여 난이도 점수를 예측하였고, 'Zero-shot' 접근 방식에서는 지시어(instruction)튜닝이 된 대규모 언어 모델 (Large Language Models, LLMs)을 사용하여 문제의 난이도를 절대적 또는 비교적으로 평가해 보았습니다. 두 방법 모두 MC 문제의 컨텍스트, 질문 및 선택지를 입력으로 사용하여 난이도를 추정하였습니다.

- **Performance Highlights**: 'Task transfer'에서는 수준 분류가 독해력 평가보다 난이도 순위에 더 잘 전이되었고, 'Zero-shot' 방식에서는 상대적 평가가 절대 평가보다 우수한 성능을 보였으며, 스피어만 상관관계가 40.4%로 가장 높았습니다. 이러한 두 시스템을 결합할 경우, 난이도 순위의 상관관계가 더욱 향상되는 것으로 관찰되었습니다.



### Integrating knowledge bases to improve coreference and bridging  resolution for the chemical domain (https://arxiv.org/abs/2404.10696)
Comments: working in progress

- **What's New**: 이 연구는 화학 특허에서 코어퍼런스(Coreference)와 브리징(Bridging) 관계를 해결하기 위해 외부 지식을 통합하는 새로운 접근 방식을 제안합니다. 화학 도메인에서의 정확한 화학 공정 이해에 필수적인 화학 도메인 지식을 중시하여 외부 지식 기반을 멀티-태스크 학습(Multi-task learning) 모델에 통합했습니다.

- **Technical Details**: SpanBERT 인코더와 연계된 외부 지식 기반을 사용하여, 후보 스팬(Candidate spans)을 해당 엔티티와 연결하고 지식 풍부한 스팬 임베딩(Knowledge-enriched span embeddings)을 생성합니다. 이러한 임베딩은 코어퍼런스와 브리징 예측에 사용됩니다. 또한, 화학 토크나이저를 사용하여 처리 효율성을 개선하고, OSCAR4와 scispaCy를 통해 엔티티 링킹을 수행하여 지식 표현을 최적화합니다.

- **Performance Highlights**: 제안한 모델은 기존 모델(Fang et al., 2021) 대비 더 나은 성능을 보였으며, 이는 화학 도메인의 정확한 코어퍼런스와 브리징 해결을 가능하게 합니다. 모델의 성능 향상은 주로 외부 지식의 통합과 효율적인 계산 전략 덕분입니다.



### ViTextVQA: A Large-Scale Visual Question Answering Dataset for  Evaluating Vietnamese Text Comprehension in Images (https://arxiv.org/abs/2404.10652)
Comments: Preprint submitted to IJCV

- **What's New**: 이 논문은 베트남어로 이미지 내 텍스트이해에 중점을 둔 최초의 대규모 데이터셋인 ViTextVQA(Vietnamese Text-based Visual Question Answering)를 소개합니다. 이 데이터셋은 텍스트가 포함된 이미지 16,762개와 50,342개의 질문-답변 쌍을 포함합니다. ViTextVQA는 시각적 질문 응답(VQA) 모델이 이미지 내 텍스트를 분석하는데 있어 주요 도전 과제들을 해결하고자 만들어졌습니다.

- **Technical Details**: VQA는 컴퓨터 비전(CV)과 자연어 처리(NLP)의 복합적 기능이 요구되는 분야입니다. 이번 연구에서는 OCR (Optical Character Recognition) 시스템의 중요성을 평가하고 ViT5를 백본으로 사용하는 VQA 모델을 실험하였습니다. 특히, OCR 텍스트의 토큰 순서가 답변 생성에 중요한 영향을 미치는 것을 발견했으며, 이를 통해 모델의 성능을 상당히 향상시킬 수 있었습니다.

- **Performance Highlights**: 연구팀은 다양한 최신 모델들과의 비교 실험을 통해 ViTextVQA 데이터셋에 대한 모델들의 성능을 평가하였습니다. OCR 텍스트 처리에서 좌상단에서 우하단으로의 정렬 방식이 모델 성능을 현저히 향상시키는 데 효과적임을 밝혔습니다.



### Self-playing Adversarial Language Game Enhances LLM Reasoning (https://arxiv.org/abs/2404.10642)
Comments: Preprint

- **What's New**: 이 논문은 'Adversarial Taboo'라 불리는 경쟁적 언어 게임에서 대형 언어 모델(Large Language Models, LLMs)의 자가 훈련 절차를 탐구합니다. 공격자(Attacker)와 방어자(Defender)가 목표 단어(target word)를 둘러싸고 대화를 펼치는 이 게임에서 LLM은 자가 플레이(Self-Play)를 통해 추론 능력을 향상시킬 수 있는지 여부를 조사합니다.

- **Technical Details**: LLM은 공격자 역할을 하여 자신의 복사본(또 다른 LLM)과 방어자로 플레이하며, 대화를 통해 방어자가 목표 단어를 무의식적으로 언급하도록 유도합니다. 이 과정은 다양한 목표 단어에서 진행되며, 게임 결과에 따라 강화 학습(Reinforcement Learning, RL)을 적용함으로써 LLM의 추론 성능이 전반적으로 개선되었다는 점을 관찰했습니다. 이러한 자가 플레이 접근 방식은 지속적으로 LLM의 추론 능력을 증진시킬 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 자기 플레이(Self-Play) 기반의 훈련 방법론은 다양한 추론 벤치마크에서 LLM의 성능을 일관되게 향상시켰으며, 이를 반복하는 과정에서 LLM의 추론 능력이 지속적으로 향상되는 것을 확인했습니다. 이는 LLM을 상대방과의 상호작용을 통해 학습하게 함으로써 더 넓은 범위의 언어적 문제와 도전에 대응할 수 있는 능력을 개발할 수 있음을 시사합니다.



### HLAT: High-quality Large Language Model Pre-trained on AWS Trainium (https://arxiv.org/abs/2404.10630)
- **What's New**: 새로운 연구에서는 AWS의 최신 기계 학습 가속기인 Trainium을 사용하여 대규모 언어 모델(LLMs)의 사전 훈련에 성공했습니다. 이 연구는 7조 개의 토큰을 사용하여 7억 개의 파라미터를 가진 'HLAT'라는 모델을 Trainium 인스턴스에서 훈련시켰으며, 이 모델은 다른 인기 있는 AI 가속기에서 훈련된 모델들과 비교하며 우수한 성능을 보여주었습니다.

- **Technical Details**: HLAT 모델은 Amazon EC2 trn1 인스턴스를 사용하여 AWS Trainium 가속기에서 사전 훈련되었습니다. 이 연구는 분산 훈련 라이브러리인 Neuron Distributed Training Library (NDTL)를 활용하여 Trainium에서의 효율적인 훈련 방법을 제시합니다. 또한, 이 모델은 3D 병렬 처리 방식과 같은 최신 분산 훈련 기술을 지원함으로써, 파라미터를 여러 가속기에 걸쳐 분할 관리하고 효율적으로 대규모 모델을 훈련할 수 있습니다.

- **Performance Highlights**: HLAT는 LLaMA 및 OpenLLaMA와 같은 NVIDIA GPU 및 Google TPU에서 훈련된 베이스라인 모델들과 동등한 성능을 달성하였습니다. 또한, AWS Trainium을 사용함으로써 비용 효율성이 크게 향상되었으며, 훈련 비용이 NVIDIA의 A100 GPU 인스턴스 대비 약 60% 절감되었습니다.



### Construction of Domain-specified Japanese Large Language Model for  Finance through Continual Pre-training (https://arxiv.org/abs/2404.10555)
Comments: 7 pages

- **What's New**: 이 연구는 일본 금융 분야에 특화된 대규모 언어 모델(Large Language Models, LLMs)을 제안하고 이를 지속적인 사전 훈련(continual pre-training)을 통해 구축하는 것을 목표로 하고 있습니다. 이전까지 일본 금융 특화 LLM은 제안되지 않았으며, 이 연구는 특정 도메인에 맞춤화된 LLM의 효과를 확인할 수 있는 중요한 사례입니다.

- **Technical Details**: 연구팀은 우선 일본 금융에 중점을 둔 데이터셋을 구축하고, 이미 일본 금융 벤치마크에서 최고 성능을 보인 100억 파라미터 클래스의 일본어 LLM을 기반 모델로 사용하여 지속적인 사전 훈련을 수행했습니다. 이후, 튜닝된 모델은 기존 모델 대비 벤치마크에서 더 우수한 성능을 보였습니다.

- **Performance Highlights**: 튜닝 후 모델은 일본 금융 벤치마크에서 기본 모델보다 우수한 성능을 나타냈습니다. 또한, 결과 비교에서는 튜닝된 모델의 출력이 원래 모델의 출력보다 답변의 질과 길이 측면에서 우월함을 보였습니다. 이 튜닝된 모델은 Hugging Face에서 공개적으로 이용 가능합니다.



### Unveiling the Misuse Potential of Base Large Language Models via  In-Context Learning (https://arxiv.org/abs/2404.10552)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 보안 취약점을 심층적으로 분석하고, 특히 기본 LLM이 악의적 명령을 해석하고 실행할 수 있는 능력을 지니고 있음을 밝혀냈습니다. 기존에는 정렬된(Aligned) 모델들만이 이러한 위험에 노출되어 있다고 여겨졌으나, 기본 모델들 역시 비슷한 위협 수준을 보유하고 있음이 드러났습니다.

- **Technical Details**: 연구팀은 악의적 쿼리를 처리하고 실행할 수 있는 새로운 평가 메트릭(Evaluation Metrics) 세트를 도입하여, 기본 LLM들의 보안 위협을 체계적으로 평가하였습니다. 이러한 메트릭이 포함하는 차원은 관련성(Relevance), 명확성(Clarity), 사실성(Factuality), 깊이(Depth), 그리고 세부사항(Detail)입니다. 연구 결과, 기본 LLM은 특별한 지식이나 훈련 없이도 악의적 내용을 효과적으로 생성할 수 있음을 보여줬습니다.

- **Performance Highlights**: ICLMisuse 방법론을 이용하면, 어떠한 특별한 자원이나 전문 지식 없이도 기본 LLM을 조작하여 위험 수준이 높은 콘텐츠를 생성할 수 있습니다. 실험을 통해 7B에서 70B에 이르는 다양한 규모의 기본 LLM들이 악의적인 목적으로 미세 조정된(Fine-tuned) 모델들과 견줄 수 있는 고품질의 유해 콘텐츠를 생성할 수 있음을 입증했습니다.



### CoTAR: Chain-of-Thought Attribution Reasoning with Multi-level  Granularity (https://arxiv.org/abs/2404.10513)
- **What's New**: 새로운 접근 방식으로 체인 오브 소트 (Chain-of-Thought, CoT) 추론 방법을 도입하여 언어모델이 입력에서 언급된 정보를 기반으로 더 정확한 귀속을 생성해내도록 합니다. 특히, 질문응답(Question Answering, QA) 작업에서 GPT-4 모델 외에도 더 작은 언어 모델의 성능을 향상시키기 위해 CoT 추론과 모델 튜닝을 결합한 새로운 방법을 제안합니다.

- **Technical Details**: 이 논문에서는 세 가지 레벨의 귀속(스팬(Span), 문장(Sentence), 통과(Passage))을 사용하여, 질의에 대한 답변을 구체적인 문맥 소스로 귀속시키는 방법을 개발했습니다. 다양한 귀속 수준에서 체인 오브 소트 추론 방식(CoTAR)을 적용하여, 모델이 입력 데이터를 분석하고 적절한 귀속을 생성하도록 안내합니다. GPT-4뿐만 아니라, Flan-T5 XXL과 같은 더 작은 모델들에게도 이 방법을 적용하여, 일부 경우에서는 GPT-4를 능가하는 결과를 보여 줍니다.

- **Performance Highlights**: 실험 결과에 따르면, 본 연구의 방법론을 사용한 모델은 높은 답변 품질(Answer Quality)과 인용 품질(Citation Quality)을 성취함을 보여 줍니다. 특히, 스팬 귀속에서의 근거 콘텐츠 일치도를 측정하는 SEM-F1 지표와 체계적 귀속 일치도를 측정하는 CSCA (Correct Span Citation Attribution) 지표에서 높은 점수를 획득하였습니다.



### White Men Lead, Black Women Help: Uncovering Gender, Racial, and  Intersectional Bias in Language Agency (https://arxiv.org/abs/2404.10508)
- **What's New**: 이 연구는 사회적 편견이 어떻게 언어 대행성(language agency)을 통해 나타나는지를 분석합니다. 특히 인간과 대규모 언어 모델(이하 LLM)이 작성한 텍스트에서 언어 대행성의 차이를 측정하기 위해 언어 대행성 분류 데이터셋(Language Agency Classification dataset)을 제안하고, 이를 기반으로 새로운 대행성 분류기(agency classifier)를 훈련하였습니다. 이 분류기를 사용하여 인간과 LLM이 작성한 텍스트에서 두드러진 언어 대행성 편견을 밝혀냈습니다.

- **Technical Details**: 연구팀은 언어 대행성을 정확하게 측정하기 위한 언어 대행성 분류 데이터셋을 제작하여 agency classifier를 훈련시켰습니다. 기존의 문자 일치(string matching) 방법과 감정 분류(sentiment classifiers) 방식이 언어 대행성을 포착하는 데 한계가 있음을 지적하고, 모델 기반 접근 방식을 사용하여 개선하였습니다. 이러한 접근을 통해 인간과 LLM이 작성한 여러 텍스트(전기, 교수 리뷰, 추천서 등)에서 성별, 인종 및 교차 정체성에 대한 언어 대행성 편견을 면밀히 조사했습니다.

- **Performance Highlights**: 이 연구는 (1) 인간이 작성한 텍스트에서의 언어 대행성 편견이 현실 세계의 사회적 관찰과 일치하며, (2) LLM이 생성한 텍스트는 인간이 작성한 텍스트보다 훨씬 더 높은 수준의 언어 대행성 편견을 보여준다는 것을 발견했습니다. 특히, 소수 집단(예: 검은 피부의 여성)이 묘사된 언어는 데이터셋 전반에서 가장 낮은 대행성 수준을 보였습니다. 이러한 결과는 사회적 맥락에서 LLM 생성 텍스트를 사용할 때 주의가 필요함을 경고합니다.



### A Sentiment Analysis of Medical Text Based on Deep Learning (https://arxiv.org/abs/2404.10503)
- **What's New**: 이 논문은 의학 텍스트의 감정 분석을 위해 BERT(Bidirectional Encoder Representations from Transformers)를 기반으로 한 새로운 심층 학습 네트워크를 실험합니다. 특히, CNN(Convolutional Neural Network), FCN(Fully Connected Network), GCN(Graph Convolutional Network)과 같은 다양한 모듈을 출력 계층에서 통합하여 실험하며, 이는 의료 분야의 감정 분석 효율성을 높이는 데 중요한 기여를 합니다.

- **Technical Details**: 연구팀은 METS-CoV 데이터셋을 사용하여 다양한 심층 학습 네트워크의 훈련 성능을 탐구했습니다. 이 데이터셋은 COVID-19와 관련된 의학적 엔티티와 감정이 포함된 트윗으로 구성되어 있습니다. BERT와 결합된 CNN 모델이 의학 텍스트 데이터셋 크기가 작을 때 타 네트워크 모델보다 우수한 성능을 보였습니다. 이는 BERT의 양방향 변환 인코더를 활용하면서도, CNN이 특징 추출에서 강점을 보여준 결과입니다.

- **Performance Highlights**: 실험 결과, CNN을 활용한 모델이 예상보다 높은 성능을 나타내었습니다. 특히, 의학적 텍스트의 감정 분석에서 정확성이 더 높게 나타났으며, 이는 CNN이 텍스트 데이터의 공간적 특성을 효과적으로 학습할 수 있기 때문입니다. 이 연구는 의료 분야에서 감정 분석을 수행할 때 모델 선택이 얼마나 중요한지를 강조하며, 향후 연구의 기준점을 제공합니다.



### When Emotional Stimuli meet Prompt Designing: An Auto-Prompt Graphical  Paradigm (https://arxiv.org/abs/2404.10500)
Comments: 9 pages, 5 figures

- **What's New**: 이 논문은 큰 언어 모델(LLM)을 위한 새로운 오토-프롬프트 그래픽 패러다임(APGP)을 제안합니다. 이 패러다임은 기존의 자극 프롬프트(stimulating prompts)와 틀 프롬프트(framework prompts)의 장점을 통합하여 LLM의 문제 해결 능력을 향상시키는데 목적을 두고 있습니다. 특히, 인간의 감정 자극 요소를 고려하는 자동화된 프롬프트 생성 및 적용을 통해 LLM이 다양한 도메인에서 문제를 보다 효과적으로 처리할 수 있도록 지원합니다.

- **Technical Details**: APGP는 두 가지 유형의 프롬프트인 자극 프롬프트와 틀 프롬프트를 결합하는 구조로, 감정 자극의 요소를 포함하고 자동적인 프롬프트 채우기 기능을 통해 LLM이 문제를 분석하고 다양한 해결책을 제시하도록 유도합니다. 이는 LLM이 문제 추상화(problem abstraction), 다양한 해결책 생성(diversified solutions generation), 종합적 최적화(comprehensive optimization) 및 해답 제공 후 자체 검증(self-verification)을 수행할 수 있게 합니다. APGP는 APE(automatic prompt engineering) 방식을 참조하여 기존에 수동으로 설계된 프롬프트의 한계를 극복하려고 시도합니다.

- **Performance Highlights**: APGP를 적용한 프레임워크는 ruozhiba 및 BBH 데이터셋에서 테스트를 거쳐 LLM의 문제 해결 효율성과 정확성을 크게 향상시켰다는 결과를 보여줍니다. 더 나아가 이 연구는 아블레이션(ablation) 실험을 통해 접근법의 유효성을 입증합니다. 이러한 결과들은 LLM의 새로운 응용 가능성을 열어주며, 인간과 유사한 인지 및 감정 처리 능력을 갖춘 LLM 개발에 중요한 단계를 제시합니다.



### Conversations as a Source for Teaching Scientific Concepts at Different  Education Levels (https://arxiv.org/abs/2404.10475)
- **What's New**: 이 논문에서는 고급 언어 모델(Large Language Models, LLMs)을 활용하여 다양한 교육 수준에 맞춘 대화형 교육을 향상시키는 새로운 접근 방법을 소개합니다. 특히, '5-Levels' 대화 데이터셋을 분석하여 각 교육 수준에 맞는 언어를 자동으로 단순화하거나 복잡하게 만드는 기법을 탐구합니다. 이 데이터셋은 WIRED에서 제공한 비디오 대본을 바탕으로 하며, 각기 다른 교육적 배경을 가진 학습자들 사이의 상호작용을 포함합니다.

- **Technical Details**: 연구팀은 HTML 페이지를 파싱하여 비디오 대본을 추출한 후, 수동으로 각 학습 수준에 맞게 대본을 세분화하고 교사와 학습자의 턴(turns)을 분리합니다. 이는 125개의 일대일 대화에서 총 570분 분량, 102,656단어, 그리고 2,881회의 대화 턴을 포함하며, Flesch-Kincaid Readability Ease (FKRE) 및 Flesch-Kincaid Grade Level (FKGL)과 같은 가독성 메트릭을 사용하여 각 복잡성 레벨에서 대본의 읽기 용이성을 분석하게 됩니다. 이 데이터는 특히 언어 모델을 교육적 상황에 맞게 조정할 때 중요한 통찰력을 제공합니다.

- **Performance Highlights**: 대화 분석 결과, 교육자는 보통 어린 대상자와의 상호작용에서 더 많은 대화를 주도하는 경향이 있으며, 대화의 양이 학습자의 지식 수준이 높아짐에 따라 줄어드는 경향이 있습니다. 전문가 수준에서는 이 경향이 역전되어 대부분의 전문가가 교사보다 더 많이 말하는 것으로 나타났습니다. 이러한 분석은 교육 자료와 교육용 챗봇이 사용자의 요구에 맞게 언어를 조절할 수 있는 능력을 갖추도록 개발하는 데 매우 유용할 것입니다.



### DESTEIN: Navigating Detoxification of Language Models via Universal  Steering Pairs and Head-wise Activation Fusion (https://arxiv.org/abs/2404.10464)
- **What's New**: 이 논문에서는 'DeStein'이라는 새로운 언어 모델 (LM) 디톡시파이 방법을 제안하여, 이전 방법들보다 자원을 적게 사용하면서도 독성을 효과적으로 제거할 수 있도록 한다. 이 방법은 활성화 공간에서 내부 표현을 변경함으로써 LM의 독성을 줄이는데 초점을 맞추고 있다.

- **Technical Details**: DeStein은 자체 생성된 스티어링 페어 (steering pairs)를 활용하여 활성화 공간에서 독성 탐지 벡터를 식별하고, 이러한 벡터를 원래의 표현과 혼합하여 출력에서의 독성을 줄이는 방법이다. 이 과정은 활성화 공간에서 수학적인 연산을 통해 이루어지며, 본질적으로 트랜스포머 모델 (Transformer models)의 특정 레이어 출력에 개입하여 변조한다.

- **Performance Highlights**: DeStein은 상태 최신 기술(State-of-the-Art, SOTA)을 넘어서는 디톡시피케이션 성능을 보여주며, 동시에 생성 품질과 다양성을 유지한다. 추가적으로, 이 방법은 GPT2-large를 포함한 다양한 대규모 언어 모델 (Large Language Models, LLMs)에 대한 확장성을 시험하였으며, 그 결과는 이 방법이 여러 모델 크기와 타입에서 확장 가능하다는 것을 보여준다.



### Language Proficiency and F0 Entrainment: A Study of L2 English Imitation  in Italian, French, and Slovak Speakers (https://arxiv.org/abs/2404.10440)
Comments: Accepted at Speech Prosody 2024

- **What's New**: 이 연구는 이탈리아어, 프랑스어 및 슬로바키아어 사용자가 영어 대화를 모방하는 도중에 발생하는 F0 (기본 주파수) 유입을 파악하기 위해 Alternating Reading Task (ART)를 사용하여 분석했습니다. 특히 L2 (제2언어) 영어 능력과 F0 유입 사이의 관계를 고찰하면서, 일반적으로 높은 능력을 지닌 사용자가 더 적은 유입을 보였지만, 더 높은 능력을 가진 사람들이 피치 범위를 더 잘 모방함으로써 유입을 증가시킬 수 있다는 점을 발견했습니다.

- **Technical Details**: 연구에서는 Dynamic Time Warping (DTW)을 사용하여 모델 발화와 모방 발화의 F0 윤곽 사이의 거리를 측정하여 F0 유입을 정량화했습니다. 이 방법은 WhisperX ASR 도구를 사용하여 발음 시간을 정렬하고, PRAAT 소프트웨어와 Parselmouth 인터페이스를 통해 F0을 추출하며, Savitzky-Golay 필터를 적용하여 F0 윤곽의 부드러움을 향상시키는 과정을 포함합니다. F0 유입의 정량화는 모델과 모방 발화의 매개변수화된 F0 윤곽 사이의 유클리드 거리를 사용하여 수행됩니다.

- **Performance Highlights**: 이 연구의 결과는 L2 영어의 유창성이 F0 유입에 미치는 영향을 더 잘 이해할 수 있도록 기여하며, 높은 영어 능력을 지닌 사용자들이 피치 범위를 더 정확하게 모방함으로써 상대방과의 유입을 높일 수 있다는 점을 시사합니다. 이러한 발견은 L2-L2 시나리오에서의 프로소딕 유입에 대한 연구에 대한 기초를 제공하며, 제2 언어 학습자들이 목표 언어에 효과적으로 적응할 수 있는 메커니즘 이해에 도움을 줍니다.



### Reasoning on Efficient Knowledge Paths:Knowledge Graph Guides Large  Language Model for Domain Question Answering (https://arxiv.org/abs/2404.10384)
- **What's New**: 이 연구는 대형 언어 모델(LLM)과 지식 그래프(KG)를 통합하여 문제 해결에 접근하는 새로운 패러다임, 'RoK(Reasoning on Efficient Knowledge Paths)'를 제안합니다. RoK는 지식 경로를 효율적으로 선택하고 LLM의 추론 능력을 극대화할 수 있습니다.

- **Technical Details**: 이 연구는 지식 그래프를 사용하여 LLM의 의존성을 줄이는 방법을 탐구합니다. 첫째, 연쇄 사고(chain of thought, CoT)를 이용해 질문에 대한 답을 확장하고 후보 키 엔티티를 더 많이 생성합니다. 이는 지식 그래프에서 엔티티와의 일치 가능성을 증가시키고 경로 선택의 정확성을 개선합니다. 둘째, 간단하면서도 효과적인 하위그래프 검색 방법인 페이지랭크(page rank)를 사용하여 답변이 포함될 가능성이 높은 경로를 반환합니다.

- **Performance Highlights**: RoK는 최소한의 LLM 호출로 기존의 상태-최고(state-of-the-art, SOTA) 모델과 동일한 결과를 달성할 수 있음을 밝혔습니다. 이는 세 가지 데이터세트(GenMedGPT-5k, WebQuestions, CMCQA)에서의 실험을 통해 입증되었습니다.



### Self-Explore to Avoid the Pit: Improving the Reasoning Capabilities of  Language Models with Fine-grained Rewards (https://arxiv.org/abs/2404.10346)
Comments: Preprint Under Review

- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력 향상을 위해 새로운 자가 학습 방법인 'Self-Explore'를 제안했습니다. 이 방법은 모델이 자체 생성한 근거(rationale)에서 첫 번째 잘못된 단계(첫 번째 함정)를 탐색하고, 이러한 신호를 미세 조정된 보상으로 사용하여 모델이 스스로 추론 능력을 향상시킬 수 있도록 합니다.

- **Technical Details**: Self-Explore 방법은 각 근거에서 첫 번째 잘못된 단계를 식별하기 위해 여러 연속을 샘플링하는 단계별 탐색을 수행합니다. 이후 양성 및 부정적인 샘플로 근거를 분류하여 쌍을 이루는 데이터셋(pair-wise dataset)을 구성하고, Direct Preference Optimization (DPO)과 같은 임의의 선호 학습 목표를 적용하여 양성 근거의 생성 가능성을 높이고 부정적인 근거의 생성 가능성을 낮추는 방식으로 진행됩니다.

- **Performance Highlights**: Self-Explore는 GSM8K와 MATH 테스트 세트에서 평균적으로 세 가지 LLM 모델(Mistral-7B, Llemma-7B, Deepseek-Math 7B)에 대해 각각 11.57%, 2.89%의 성능 향상을 보였으며, 이는 감독된 미세 조정(Supervised Fine-Tuning, SFT)과 비교했을 때 높은 수치입니다. 특히, GSM8K에서는 13.19%, 10.23%, 11.30%의 향상을, MATH에서는 1.98%, 3.16%, 3.54%의 향상을 각 모델별로 관찰했습니다.



### Enhancing Confidence Expression in Large Language Models Through  Learning from Past Experienc (https://arxiv.org/abs/2404.10315)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 자신감 표현 능력을 향상시키기 위한 새로운 방법론인 과거 경험에서 학습하기(LePe)를 제안합니다. LLMs는 종종 정확하거나 불확실한 답변을 높은 자신감으로 생성하는 경향이 있는데, 이 방법론은 LLMs가 자신의 답변에 대한 실제 확률과 일치하는 자신감 수준을 표현할 수 있도록 돕습니다.

- **Technical Details**: LePe 방법은 세 가지 핵심 질문에 초점을 맞추고 있습니다: (1) LLM의 내재된 자신감을 어떻게 포착할 것인가? (2) LLM에게 어떻게 자신감을 표현하도록 가르칠 것인가? (3) LLM의 자신감 표현을 어떻게 평가할 것인가? 이를 위해, 문제 준비와 답변 샘플링을 포함한 전체 파이프라인을 설계하고, 다양한 데이터 세트에서 Llama LLMS 패밀리를 사용한 실험을 통해 접근법의 유효성을 검증했습니다.

- **Performance Highlights**: 실험 결과는 LePe 방법이 LLMs에게 정확한 자신감 점수를 제공하여 답변의 정확성을 반영할 수 있도록 하며, 이는 자연어 처리(NLP) 작업에서 환각을 줄이고 개발자들이 LLMs의 약점을 정확하게 지적하여 성능을 반복적으로 향상시킬 수 있도록 합니다.



### Balancing Speciality and Versatility: a Coarse to Fine Framework for  Supervised Fine-tuning Large Language Mod (https://arxiv.org/abs/2404.10306)
Comments: 43 pages, 10 figures

- **What's New**: 새로운 프레임워크인 CoFiTune을 소개합니다. CoFiTune은 대용량 언어 모델(LLM)의 특수성과 다재다능함 사이의 균형을 맞추기 위해 고안되었습니다. 이 프레임워크는 특수성을 향상시키면서 동시에 모델의 다재다능성을 유지하는 것을 목표로 합니다. CoFiTune은 크게 두 가지 단계로 나누어 진행됩니다. 첫 번째로, 'coarse-grained level'에서는 실증적 나무 검색 알고리즘(empirical tree-search algorithm)을 사용해 특수성을 담당하는 핵심 모듈들을 식별하고 업데이트하며, 다른 매개변수들은 고정된 상태로 유지됩니다. 두 번째 단계인 'fine-grained level'에서는 소프트-마스킹 메커니즘(soft-masking mechanism)을 통해 모델의 역전파(gradients)를 조절하여 재앙적 망각(catastrophic forgetting, CF) 문제를 완화합니다.

- **Technical Details**: CoFiTune은 LLM의 특정 레이어 내 모듈들을 식별하고 업데이트하는 과정에서 중요한 역할을 합니다. 이는 트리 검색 알고리즘을 적용하여 특수성이 중요한 모듈을 결정하고, 이러한 모듈만을 대상으로 파라미터 업데이트가 이루어집니다. 추가적으로, 이 프레임워크는 각 모듈 내의 유닛들(예: Attention heads, FFN neurons)의 중요성을 평가하고, 그 중요성에 따라 역전파를 조절하는 소프트-마스킹 기법을 사용합니다. 이를 통해, 모델은 특수성을 손상시키지 않으면서도 범용성을 개선할 수 있습니다.

- **Performance Highlights**: CoFiTune은 다양한 작업과 모델 크기에서 기준 모델들을 일관되게 능가합니다. 13B 모델에서는 전체 매개변수 SFT에 비해 약 14%의 범용성 향상과 미미한 특수성 손실을 달성하였습니다. CoFiTune의 평균적인 성능은 기존 및 전체 SFT 모델에 비하여 95%의 범용성과 90%의 특수성을 달성하였습니다. 이러한 결과들은 중간-하위 레이어 범위의 FFN 모듈을 조정함으로써 특수성이 만족스럽게 유지될 수 있음을 보여줍니다.



### Future Language Modeling from Temporal Document History (https://arxiv.org/abs/2404.10297)
Comments: Accepted by ICLR 2024

- **What's New**: 이 연구는 미래 언어 모델링(future language modeling)이라는 새로운 과제를 도입하였습니다. 이 과제는 과거의 텍스트 데이터를 기반으로 미래의 텍스트를 예측하는 것을 목표로 하며, 이는 자연어 처리(natural language processing) 및 기계 학습(machine learning) 분야에서 이전에는 체계화되거나 깊이 연구되지 않았던 주제입니다.

- **Technical Details**: 미래 언어 모델링 작업은 시간적 이력을 가진 텍스트에 기반하여 미래 텍스트를 생성하는 생성적 언어 모델(generative language model)을 구축하는 것입니다. 연구팀은 시간적 정보를 통합한 세 가지 다른 방법을 개발하였습니다: 단어 빈도 모델(word frequency model), 맥락적 시간 모델(contextual temporal model), 그리고 이중 맥락적 시간 모델(doubly contextualized temporal model). 이 모델들은 전통적인 비시간적(non-temporal) 언어 모델과 비교하여 미래의 텍스트 내용을 보다 정확하게 예측할 가능성을 높입니다.

- **Performance Highlights**: 이 연구에서 개발된 미래 언어 모델은 자동 수치 평가뿐만 아니라 인간 평가에서도 기존의 비시간적 언어 모델보다 우수한 성능을 보여주었습니다. 특히, ACL 컨퍼런스 논문의 추상을 모델링하는 데 사용하여 우수한 결과를 얻었으며, 이는 미래 텍스트 데이터 예측이 가능하다는 연구 가설을 뒷받침합니다.



### Modeling Low-Resource Health Coaching Dialogues via Neuro-Symbolic Goal  Summarization and Text-Units-Text Generation (https://arxiv.org/abs/2404.10268)
Comments: Accepted to the main conference of LREC-COLING 2024

- **What's New**: 이 연구에서는 헬스 코칭에서 활용 가능한 새로운 신경 기호 요약 모델(neuro-symbolic goal summarizer)과 대화 생성 모델(text-units-text dialogue generation model)을 제안합니다. 이 모델들은 기존의 사전 정의된 스키마나 해당 주석 없이도 상태 기술의 최신 기술을 능가하는 성능을 보여줍니다. 또한 새로운 헬스 코칭 데이터셋을 제공하여 기존 작업을 확장하며, 환자 반응의 비전형성을 측정하는 새로운 지표를 도입하여 실제 배포시 코치에게 경보를 용이하게 할 수 있습니다.

- **Technical Details**: 신경 기호 요약(neuro-symbolic goal summarization) 접근 방식은 사전 설계된 스키마나 해당 주석 없이 목표를 요약하고 해석 가능성을 유지하도록 최적화되었습니다. 텍스트-단위-텍스트 대화 생성(text-units-text dialogue generation) 모델은 대화의 역사를 상징화하는 이산 단위의 시퀀스를 입력으로 고려합니다. 또한, 대화 생성을 위해 외부 데이터셋을 사용하지 않고 Point-wise 𝒱𝒱-caligraphic_V (Point-wise Usable Information, PVI)를 확장하는 방식으로 비전형적 반응을 감지합니다.

- **Performance Highlights**: 자동 메트릭과 전문가 기반 인간 평가를 통해 모델을 평가한 결과, 신경 기호 목표 요약기는 의미 프레임 정확도 측면에서 현재 최고의 기술보다 약 30% 향상된 성능을 보였습니다. 텍스트-단위-텍스트 대화 생성 모델도 모든 데이터셋에 대해 이전 작업보다 우수한 성능을 나타냈습니다. 헬스 코치들은 기존 최고 기술 대비 33.9%의 경우에 우리가 생성한 반응을 선호했습니다.



### Uncovering Latent Arguments in Social Media Messaging by Employing  LLMs-in-the-Loop Strategy (https://arxiv.org/abs/2404.10259)
- **What's New**: 이 연구에서는 소셜 미디어 메시지에서 특정 테마와 관련된 논쟁을 발견하는 문제를 다루며, 'LLMs-in-the-Loop' 전략을 제안하여 대규모 언어 모델 (LLMs)을 사용하여 숨겨진 논쟁을 추출합니다. 이 방법은 기존의 수동 코딩 기술에 의존하는 대신, 자동화된 프로세스를 통해 논쟁을 식별하고 매핑하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 테마에 따라 인스턴스를 클러스터링하고, 각 서브 클러스터에 대해 제로샷 다중 문서 요약을 수행한 후, LLMs를 이용하여 주제 맥락에서 서브 클러스터 요약을 옹호하는 논점을 생성합니다. 특히, 중복된 논점을 체크하고 가장 유사한 논점들을 병합하는 메커니즘을 포함하여, 신규 생성된 논점에 대한 선택적 인간 평가 단계를 거칩니다.

- **Performance Highlights**: 이 연구는 페이스북에서의 기후 캠페인 및 COVID-19 백신 캠페인과 관련하여 논쟁을 특정하고 매핑할 수 있는 능력을 시연하며, 인간 판단과 비교하여 상당히 정확한 결과를 보여줍니다. 또한, 실제 세계 이벤트에 대응하여 메시지가 특정 데모그래픽에 맞춤화되는 방식을 깊이 있게 분석합니다.



### Generative Text Steganography with Large Language Mod (https://arxiv.org/abs/2404.10229)
- **What's New**: 이 논문에서는 LLM 기반의 블랙박스 생성적 텍스트 스테가노그래피, LLM-Stega를 제안합니다. 이 기법은 대규모 언어 모델의 사용자 인터페이스(UI)를 활용하여 비밀 메시지를 안전하게 암호화 및 전달하는 새로운 방법을 탐구합니다. 키워드 집합을 구성하고 새로운 암호화된 스테가노그래피 매핑을 설계하여 비밀 메시지를 내장하는 것이 특징입니다.

- **Technical Details**: LLM-Stega는 기존의 스테가노그래피 방식과 달리 사용자 인터페이스를 통해 접근 가능한 블랙박스 LLM을 이용합니다. 최적의 결과를 달성하기 위해 거부 샘플링(reject sampling) 기반의 최적화 메커니즘을 도입하여 비밀 메시지의 정확한 추출과 생성된 스테고(Stego) 텍스트의 풍부한 의미를 보장합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 LLM-Stega는 산술 코딩(Arithmetic Coding), ADG, 그리고 Discop과 같은 최신 방법들보다 임베딩 용량과 보안 측면에서 우수한 성능을 보였습니다. 이 방법은 스테가노그래피 분야에서 기존보다 높은 임베딩 용량과 보안성을 제공하는 새로운 경로를 제시합니다.



### CULTURE-GEN: Revealing Global Cultural Perception in Language Models  through Natural Language Prompting (https://arxiv.org/abs/2404.10199)
- **What's New**: 이 연구는 전 세계의 다양한 문화에 대해 충분한 지식과 공정한 대표성을 갖춘 대규모 언어 모델(Large Language Models, LLMs)의 중요성을 다루고 있습니다. 특히, 세계 각국 및 지역 110개에 대한 8가지 문화 관련 주제에 대해 LLMs의 문화 인식을 밝히고, 이러한 문화 컨디셔닝 생성에서 추출된 문화 상징을 분석합니다. 연구 결과, LLMs가 마이너리티(minority) 문화를 기본 문화(default cultures)와 구별하는 언어적 '표지(marker)'를 포함하고 있음을 발견했습니다.

- **Technical Details**: 연구자들은 gpt-4, llama2-13b, mistral-7b와 같은 다양한 LLMs를 사용하여 자연어 프롬프트를 통해 110개 국가 및 지역에 대한 8가지 문화 관련 주제에 대한 생성물을 얻었습니다. 언어 모델에서 생성된 내용을 분석하여 각 문화와 관련된 상징 '문화 상징(culture symbols)'을 추출하고, 이를 기반으로 각 문화에 대한 모델의 지식과 인식을 평가했습니다. 이때 사용된 '무인 감독(unsupervised) 문장 확률 순위 매기기(sentence-probability ranking method)' 방법 등은 휴먼 라벨링이나 외부 지식 베이스 없이도 적용 가능합니다.

- **Performance Highlights**: LLMs는 지리적 위치에 따라 문화 상징의 다양성에 큰 차이를 보였으며, 특히 아시아, 동유럽, 아프리카-이슬람 국가들의 문화를 '전통적(traditional)'이라는 단어와 함께 생성하는 경향이 있었습니다. 연구 결과, 문화 무관 생성물에서 서구 유럽, 영어권 및 노르딕 국가들이 가장 많은 문화 상징과 겹치는 것으로 나타났습니다. 이는 LLMs의 글로벌 문화 지식과 공정성에 대한 중요한 통찰력을 제공합니다.



### How faithful are RAG models? Quantifying the tug-of-war between RAG and  LLMs' internal prior (https://arxiv.org/abs/2404.10198)
- **What's New**: 이 연구는 Retrieval Augmented Generation(RAG)가 대형 언어 모델(LLM)의 오류 수정과 최신 지식 제공에 어떻게 활용되는지에 대한 분석을 제공합니다. 특히, 검색된 내용이 올바르지 않을 때 LLM이 이를 간과할 수 있는지, 아니면 잘못된 정보를 반복할지에 대해 탐구합니다.

- **Technical Details**: 이 논문은 GPT-4 및 기타 LLM을 사용하여 검색된 문서가 내부 지식(모델의 선험적 지식)과 어떻게 상충하는지를 시험합니다. 연구진은 다양한 도메인의 데이터셋을 통해 1,294개의 질문에 대해 답변하는 LLM의 능력을 평가했으며, RAG 문서에 다양한 수준의 교란을 도입하는 방법으로 실험을 설계하였습니다.

- **Performance Highlights**: 정확한 검색 정보를 제공할 경우 모델의 오류를 94% 정확도로 수정할 수 있었습니다. 그러나 문서의 정보가 잘못된 값으로 변경될 때 LLM은 그 잘못된 정보를 더 자주 인용하는 경향이 있었고, 모델의 내부 선험적 지식이 약할수록 이 경향이 강해지는 것으로 나타났습니다. 반면에, 내부 선험적 지식이 강한 경우 모델은 잘못된 정보를 덜 선호하는 것으로 관찰되었습니다.



### Deferred NAM: Low-latency Top-K Context Injection via DeferredContext  Encoding for Non-Streaming ASR (https://arxiv.org/abs/2404.10180)
Comments: 9 pages, 3 figures, accepted by NAACL 2024 - Industry Track

- **What's New**: 이 연구에서는 말하는 사람의 문맥에 있는 중요한 문구를 인식할 수 있도록 음성 인식기에 문맥적 편향(Contextual Biasing)을 적용하는 새로운 기술을 제시합니다. 특히, Attention-based biasing을 사용하여 인식 시스템 전체를 종단간(end-to-end)으로 학습할 수 있으며, 이를 통해 별도의 인식 시간 구성 요소 없이도 효과적인 편향을 적용할 수 있습니다.

- **Technical Details**: 이 기술은 경량 문구 선택 절차(phrase selection pass)를 문맥 인코딩(context encoding) 이전으로 이동시켜 처리 속도를 최대 16.1배까지 향상시키고, 문맥 인코딩 지연을 33ms 이하로 유지하면서 최대 20K 문구까지 확장 가능하게 합니다. 또한, 문구 수준과 단어 조각(wordpiece) 수준에서의 교차 엔트로피 손실(cross-entropy losses)을 추가하는 기술을 도입하여, 기본 모델 대비 최대 37.5% 상대적으로 WER(Word Error Rate)을 개선합니다.

- **Performance Highlights**: 이 연구는 복잡한 음성 인식 시나리오에서 ASR의 정확성과 속도를 동시에 개선하는 데 중요한 기여를 합니다. 특히, 자동으로 더 적합할 것으로 예상되는 문구를 선택하고 더 강력한 인코더를 이용하여 문맥을 적용하므로, 효율성과 성능이 동시에 증가합니다.



### On the Effects of Fine-tuning Language Models for Text-Based  Reinforcement Learning (https://arxiv.org/abs/2404.10174)
- **What's New**: 본 연구는 텍스트 기반 강화학습(Text-based Reinforcement Learning)에서의 시맨틱 이해(Semantic Understanding)가 에이전트의 학습 효율성을 높이는 데 중요하다는 점을 강조합니다. 이전 연구들은 시맨틱 이해가 없이도 에이전트가 텍스트 기반 게임을 성공적으로 수행할 수 있다고 보고했으나, 본 연구는 뛰어난 시맨틱 이해가 효율적인 훈련에 기여할 수 있음을 밝힙니다. 또한, 언어 모델(Language Models, LMs)의 부적절한 파인튜닝(Fine-Tuning)으로 인한 '시맨틱 퇴화(Semantic Degeneration)' 현상을 설명하고, 이 현상이 훈련 게임과 유사한 의미를 갖는 작업에서 에이전트의 성능에 영향을 미친다는 점을 지적합니다.

- **Technical Details**: 본 실험에서는 텍스트월드커먼센스(TextWorld Commonsense, TWC)와 제리코(Jericho)와 같은 두 가지 텍스트 기반 게임(Text-Based Games, TBGs) 도메인을 사용하여 실험을 수행했습니다. 에이전트는 관측된 텍스트와 허용된 자연어 동작을 기반으로 환경과 상호작용하여 게임의 최종적인 목표를 달성하는 것입니다. 본 연구에 사용된 주요 언어모형은 GloVe와 Transformer입니다. 이러한 모형들은 게임 중 보상값을 활용하여 파인튜닝 되며, 이 과정에서 언어 모델이 학습 게임에 과적합(Overfit)되어 의미 관계의 퇴화를 초래할 수 있다는 가설을 세웠습니다.

- **Performance Highlights**: 파인튜닝된 언어 모델을 사용한 에이전트는 고정된 사전 훈련된 언어 모델을 사용했을 때보다 훈련 및 테스트 설정에서 성능이 저하되었습니다. 특히, 훈련 게임의 관측치와 동작이 다소 달리 표현되었을 때(예: 단어를 다르게 표현하거나 유의어를 사용하는 등), 에이전트의 성능이 떨어졌습니다. 이러한 결과는 언어 모델이 중요한 의미 정보를 잃어버려 일반화 가능한 표현을 만드는 데 실패했기 때문으로 보입니다. 예를 들어, 'Zork 1' 게임에서 'bloody axe'와 'kitchen'이 연관되어 있음을 학습한 에이전트는 이 설정에 과적합되어 다른 게임에서 중요할 수 있는 'kitchen'과 'bloody axe'에 관한 정보를 잃어버렸습니다.



### TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table  Decomposition (https://arxiv.org/abs/2404.10150)
Comments: Accepted to NAACL 2024 (long, main)

- **What's New**: 이 연구에서는 TabSQLify라는 새로운 방법을 제안하는데, 이 방법은 텍스트-투-SQL(text-to-SQL) 생성을 활용하여 테이블을 작고 관련 있는 하위 테이블(sub-table)로 분해합니다. 이러한 접근법은 대규모 테이블 추론 작업에서 입력 길이를 현저히 줄이고, 처리 효율성과 확장성을 크게 향상시킵니다.

- **Technical Details**: TabSQLify는 자연어 질문이나 명제를 SQL 쿼리로 변환하고, 이 쿼리를 원본 테이블에 실행하여 필수 정보만 포함된 하위 테이블을 생성합니다 (1단계). 그 후, LLM(Large Language Model)을 사용하여 하위 테이블과 질문 또는 명제와 함께 답을 생성합니다 (2단계). 이 접근 방식은 LLM의 자연어 이해 및 생성 능력을 활용하면서 테이블 인코딩과 추론 부담을 줄입니다.

- **Performance Highlights**: TabSQLify는 WikiTQ 벤치마크에서 64.7%의 정확도를 달성하였으며, TabFact 벤치마크에서는 79.5%의 높은 정확도를 보여주었습니다. 이는 기존의 대규모 테이블을 입력으로 사용하는 방법에 의존하는 모델들보다 우수한 성능을 나타내며, 특히 테이블 크기가 최대 허용 입력 창을 초과하는 큰 테이블에서 두드러진 이점을 제공합니다.



### Language Model Cascades: Token-level uncertainty and beyond (https://arxiv.org/abs/2404.10136)
- **What's New**: 이 연구는 생성적 언어 모델(generative language models, LMs) 작업에서 새로운 접근 방식을 사용하여 모델의 복잡성과 비용에 대한 트레이드오프(tradeoff)를 개선하는 방법을 제안하고 있습니다. 특히, 간단한 예측 시퀀스 불확실성(predicted sequence uncertainty) 집계 방식을 넘어서, 토큰 수준(token-level) 불확실성 정보를 학습한 후속 연기 규칙(learned post-hoc deferral rules)을 사용해 이를 개선하고자 합니다. FLAN-T5 모델을 사용하여 자연 언어 벤치마크에서의 실험을 통해 이러한 접근 방식의 유효성을 검증합니다.

- **Technical Details**: 연구에서는 토큰 수준 불확실성을 적용하기 위해 다중 레이어 퍼셉트론(Multi-Layer Perceptron, MLP)을 사용하여 후속 연기(deferral) 규칙을 학습하였습니다. MLP는 5개의 레이어와 각 차원에 대해 32개의 숨겨진 유닛(hidden units)을 가지며, 배치 정규화(batch normalization) 층을 포함해 200 에포크(epoch) 동안 훈련되었습니다. 추가로, 작은 모델과 큰 모델의 중간 층에서 임베딩(embeddings)을 추출하여 비용-품질 트레이드오프를 높이는 데 사용하였습니다.

- **Performance Highlights**: 이 방법은 전통적인 집계 전략보다 상당한 성능 향상을 보여주었으며, 특히 토큰 수준의 불확실성 정보를 사용할 때 더 뛰어난 결과를 얻을 수 있었습니다. FLAN-T5 모델을 사용한 자연어 처리 벤치마크에서, 제안한 방법은 기존 방법보다 우수한 결과를 달성하였습니다. 특히, 예측 길이(prediction length)와 BLEURT 점수 간에 상당한 음의 상관관계가 관찰되었으며 이는 출력 길이를 연기(deferral) 기준으로 사용하는 데 유용한 신호를 제공합니다.



### PRODIS - a speech database and a phoneme-based language model for the  study of predictability effects in Polish (https://arxiv.org/abs/2404.10112)
Comments: To appear in the proceedings of LREC2024: Language Resources and Evaluation Conference 2024, Turin, Italy

- **What's New**: 이 연구에서는 폴란드어 음성 데이터베이스와 음소 수준의 언어 모델(PRODIS)을 소개합니다. 이 데이터베이스는 예측 가능성(predictability) 효과가 음향 변수에 미치는 영향을 분석하기 위해 설계되었으며, 우수한 음향 품질의 폴란드어 음성 코퍼스를 제공합니다. 또한, 이 데이터베이스는 다양한 발화 상황(독서 및 대화)과 음성 기술 시스템 개발에 사용할 수 있습니다.

- **Technical Details**: PRODIS 데이터베이스는 GPT 아키텍처를 기반으로 한 언어 모델을 포함하여 폴란드어 위키피디아 텍스트로 학습되었습니다. 이 모델은 음소에 대한 surprisal을 추정하여 음속 구별성(acoustic distinctiveness)에 미치는 영향을 연구합니다. 데이터베이스에는 50명의 발화자가 있으며, 각각의 발화자는 약 50분간 녹음되었습니다. 음성은 자동화된 파이프라인을 통해 처리되었고, 음향 변수(음성 길이, 기본 주파수 f0, 포먼트 값, 스펙트럼 에너지 등)를 추출합니다.

- **Performance Highlights**: PRODIS는 폴란드어 음성의 전처리와 분석이 90% 자동화된 성과를 달성하였습니다. 데이터는 뛰어난 음향 품질을 유지하며 다양한 음향학적 분석과 음성 기술 시스템 훈련에 활용될 수 있습니다. 이는 폴란드어 특유의 음성 구분과 예측 가능성, 발화 스타일 및 발성 특성의 상호작용을 연구하는 중요한 자료가 됩니다.



### LaDiC: Are Diffusion Models Really Inferior to Autoregressive  Counterparts for Image-to-Text Generation? (https://arxiv.org/abs/2404.10763)
- **What's New**: 이 연구에서는 텍스트-이미지 생성에서 두각을 나타낸 확산 모델(Diffusion models)의 이미지-텍스트 생성, 특히 이미지 캡셔닝(image captioning)에서의 성능을 개선하는 새로운 아키텍처 LaDiC를 소개합니다. 이 모델은 이미지로부터 텍스트를 효과적으로 생성하는 데 필요한 고유한 잠재 공간을 만들고, 텍스트 길이의 변화를 관리하는 정규화 모듈을 통합합니다. 또한, 이미지-텍스트 변환을 위한 'diffuser'와 추론 중 토큰 상호 작용을 개선하는 Back&Refine 기술을 사용합니다.

- **Technical Details**: LaDiC는 BERT를 분할하여 캡션을 위한 전용 잠재 공간을 생성합니다. 이 모델은 연속적인 확산 과정과 이산형(discrete) 텍스트 데이터 간의 간극을 다리고, 텍스트의 가변 길이를 처리하기 위해 정규화와 재할당 절차를 포함하는 후처리 모듈을 제안합니다. 더 나은 성능을 위해, 토큰 간 더 많은 상호 작용을 제공하는 Back&Refine 기술이 도입되었습니다.

- **Performance Highlights**: LaDiC는 MS COCO 데이터셋에서 확산 기반 방법론으로 최고의 성능을 달성했으며, BLEU@4에서 38.2, CIDEr에서 126.2의 점수를 기록했습니다. 이는 사전 훈련이나 추가 모듈 없이도 구현되었으며, 자동 회귀 모델(Auto-Regressive models, AR)과 경쟁할 수 있는 성능을 보여줍니다.



### Deep Learning and LLM-based Methods Applied to Stellar Lightcurve  Classification (https://arxiv.org/abs/2404.10757)
Comments: 35 pages, 20 figures

- **What's New**: 이 연구에서는 Kepler와 K2 임무의 대규모 데이터셋을 기반으로 변광성의 빛 곡선을 자동 분류하기 위해 딥러닝(deep-learning) 및 대형 언어 모델(LLM)을 포함한 모델들의 체계적인 평가를 제시합니다. 특히 세페이드(Special emphasis), RR Lyrae, 그리고 식쌍성(eclipsing binaries)에 초점을 맞추어 관측 템포와 위상 분포(phase distribution)가 분류 정확도에 미치는 영향을 조사했습니다. 또한 새로운 시리즈인 StarWhisper LightCurve (LC)를 도입하여 천문학 데이터의 자동화된 분석과 처리 능력을 크게 향상시키고자 하였습니다.

- **Technical Details**: 본 연구는 1D-Convolution+BiLSTM 아키텍처와 Swin Transformer를 사용하여 각각 94%와 99%의 높은 정확도를 달성하였고, 특히 Type II 세페이드를 83%의 정확도로 식별할 수 있었습니다. StarWhisper LC 시리즈는 LLM, 멀티모달 대형 언어 모델(MLLM), 그리고 대형 오디오 언어 모델(LALM)을 포함하며, 고도의 트레이닝 방법과 프롬프트 엔지니어링을 통해 미세 조정됩니다. 이 모델들은 약 90%의 높은 정확도를 보여 주었으며, 명시적인 피처 엔지니어링의 필요성을 크게 줄였습니다.

- **Performance Highlights**: LLM 기반의 StarWhisper LC 시리즈는 아주 높은 정확도를 달성했으며, 관측 기간을 최대 14%까지 단축하고 샘플링 포인트를 21%까지 줄임으로써 정확도를 10% 이상 손실하지 않고 데이터 처리 효율성을 대폭 향상시켰습니다. 또한, 이 연구는 위상 및 샘플링 간격의 영향을 상세히 설명하는 두 개의 카탈로그를 제공하여, 깊은 학습 분류 정확도에 미치는 영향을 보여줍니다.



### Cross-Language Evolution of Divergent Collective Memory Around the Arab  Spring (https://arxiv.org/abs/2404.10706)
- **What's New**: 이 논문은 아랍의 봄과 관련된 위키백과 기사의 내용이 어떻게 시간이 지남에 따라 진화하는지 조사하여, 온라인 집단 기억(collective memory) 과정에 대한 이론화와 언어 모델(language models) 평가에 대한 시사점을 제공합니다. 특히, 아랍어와 영어판 위키백과의 아랍의 봄 관련 기사를 사용하여 이벤트의 중요성(event salience), 논의(deliberation), 맥락화(contextualization), 집단 기억의 공고화(consolidation)를 다각적으로 측정하고 평가합니다.

- **Technical Details**: 연구팀은 '아웃링크(outlinks)'와 '다국어 링크(inter-lingual links, ILL)'를 이용하여, 기사 간의 관련성과 맥락을 분석하는 방법을 개발했습니다. 이러한 링크들은 편집자들이 판단한 기사 간의 유사성과 맥락의 강력한 신호로, 이벤트의 집단 기억 과정에 대한 이해를 돕습니다. 또한, 기사의 사이즈와 아웃링크 수를 분석하여 아랍의 봄에 대한 지속적인 관심과 기억의 업데이트를 추적했습니다.

- **Performance Highlights**: 연구 결과, 2011년부터 2024년 초까지 아랍의 봄은 언어간에 걸쳐 주요 사건으로 여겨졌으며, 각 언어 버전에서 다르게 다뤄지는 아웃링크의 존재가 확인되었습니다. 영어판에서는 '소셜 미디어와 아랍의 봄(Social Media and the Arab Spring)'과 '후유증(Aftermath)' 섹션의 추가와 수정을 통해 주제의 중요성이 지속적으로 강조되었습니다. 반면, 아랍어판은 2013년 이후 주요 업데이트가 거의 없이 일관되게 작은 변경만이 이루어졌습니다. 이는 언어와 문화적 맥락에 따라 집단 기억의 구성과 공고화에 차이가 있음을 보여줍니다.



### What are human values, and how do we align AI to them? (https://arxiv.org/abs/2404.10636)
- **What's New**: 이 논문은 인간의 가치와 AI 시스템을 조화시키려는 새로운 접근 방식인 'Moral Graph Elicitation (MGE)'을 제시하며, 이는 사용자의 가치를 반영하여 AI의 행동을 지도할 수 있는 'moral graph'를 생성합니다. 이 연구는 특히 미국 내의 대표적인 샘플을 사용하여 세 가지 분열적인 주제에 대해 MGE를 시험하였습니다.

- **Technical Details**: MGE는 큰 언어 모델을 사용하여 참여자들의 가치를 열거하고, 이를 바탕으로 구조화된 데이터 오브젝트인 'moral graph'를 생성합니다. 이 과정은 Taylor (1977)와 Chang (2004)의 가치 철학에 영감을 받아 설계되었습니다. 결과적으로 생성된 moral graph는 문맥에 따라 한 가치가 다른 가치보다 더 지혜로운지를 평가하는 튜플로 구성됩니다.

- **Performance Highlights**: 이 연구의 결과로, 대부분의 참여자들(89.1%)이 자신들의 가치가 잘 반영되었다고 느꼈으며, 최종적인 moral graph가 공정하다고 평가했습니다(89%). 또한, '전문가'의 가치가 더 높은 순위에 오르는 경향을 보였음에도 불구하고, 누가 전문가인지 사전에 정의하지 않는 방식으로 진행되었습니다.



### The application of Augmented Reality (AR) in Remote Work and Education (https://arxiv.org/abs/2404.10579)
- **What's New**: 이 연구는 증강 현실(AR: Augmented Reality) 기술의 원격 작업 및 교육에서의 응용 잠재력과 실제 효과를 탐구합니다. 특히 원격 근무와 온라인 교육 분야에서 AR 기술이 매우 다양한 응용 전망을 보이고 있습니다. 이 연구는 전통적인 작업 모드와 교육 방식의 점진적 변화를 조명하며, AR 기술의 미래 발전 동향과 심도 있는 응용을 도모할 수 있는 전략적 제안을 제시합니다.

- **Technical Details**: 이 논문은 체계적 문헌 검토를 통해 AR 기술의 주요 특징, 장점, 도전과제를 개요하고, 이론적 분석을 바탕으로 원격 작업 효율성을 향상시키고 교육적 교수 모델의 혁신을 촉진하는 데 AR 기술이 제공하는 과학적 기초와 기술적 지원에 대해 논의합니다. 또한 실증 연구 계획을 설계하고 실험 데이터를 분석함으로써, AR 기술의 구체적 성능과 영향 요인을 밝힙니다.

- **Performance Highlights**: 실험 결과를 바탕으로 연구는 원격 작업과 교육에서 AR 기술의 응용 가치를 정리하고, 구체적 성과와 영향 요인을 드러냄으로써, AR 기술의 실제 응용에서의 특정 성능을 밝힙니다. 예를 들어, AR 기술의 구현은 원격 작업의 효율성을 크게 향상시키고, 교육 분야에서는 인터랙티브(interactive) 학습 경험을 제공하여 학습 효율을 높이는 데 기여하였습니다.



### Self-Supervised Visual Preference Alignmen (https://arxiv.org/abs/2404.10501)
- **What's New**: 이 연구는 비감독 선호도 정렬(Unsupervised Preference Alignment)을 비전-언어 모델(Vision-Language Models, VLMs)에 처음 시도합니다. 이 방법은 GPT-4 또는 인간 개입 없이 구현되며, 몇 줄의 코드만으로 고효율로 동작합니다.

- **Technical Details**: 이 연구에서는 이미지 입력에 대한 적절한 증강(Augmentation)을 통해 VLM이 잘못되었지만 어려운 부정적인 응답(Negative Responses)을 생성하도록 유도하며, 이를 통해 모델이 더 강력하고 견고한 답변을 생성하도록 합니다. 사용된 주요 방법은 자체감독(Self-supervised) 시각적 선호도 정렬 기법(Visual Preference Alignment, SeVa)으로, DPO(Direct Preference Optimization) 훈련에 사용되는 선호도 데이터의 생성을 자동화합니다.

- **Performance Highlights**: 이 방법은 단 8,000개의 무작위 선택된 미감독 데이터(Unsupervised Data)만을 사용하여 LLaVA-Bench에서 GPT-4에 비해 90%의 상대적인 성능을 달성하였고, 복잡한 멀티모달 벤치마크(Multi-modal Benchmark) MM-Vet에서 LLaVA-7B와 13B 모델의 성능을 각각 6.7%, 5.6% 향상시켰습니다.



### MAD Speech: Measures of Acoustic Diversity of Speech (https://arxiv.org/abs/2404.10419)
- **What's New**: 이 논문은 다양한 목소리, 감정, 억양, 그리고 배경 소음에서 음성의 음향적 다양성을 측정하는 새로운 지표인 MAD Speech (Measures of Acoustic Diversity of Speech)을 제안합니다. 이는 음성 생성 모델이 자연스러운 음성의 다양성에 얼마나 근접하는지 정량화하려는 시도로, 기존에는 이러한 다양성을 측정할 적합한 메트릭이 부족했습니다.

- **Technical Details**: MAD Speech 메트릭은 HuBERT, Wav2Vec-BERT, SoundStream과 같은 다양한 사전 훈련된 음성 표현과, TRILL 및 COLA에 기반한 SpeechSim 모델을 사용합니다. 이들은 특히 음성 데이터의 목소리, 성별, 감정, 억양, 배경 소음 등 다양한 양상(facets)을 독립적으로 강조할 수 있는 임베딩 공간에서의 다양성을 측정하는 데 사용됩니다. 평균 쌍(cosine) 비유사성과 Vendi Score 같은 집계 함수(aggregation function)를 활용하여, 표현 공간 내에서 샘플의 다양성을 계산합니다.

- **Performance Highlights**: 제안된 MAD Speech 메트릭은 실제 음성 데이터의 다양성과 강한 일치를 보이는 것으로 평가되었습니다. 구체적인 데이터셋을 사용하여 음성의 다양한 양상을 체계적으로 분석하였으며, 이 메트릭이 실제 음성 평가 시나리오에서 유용함을 입증하였습니다. 특히, 다양한 음성 생성 기술과 함께 사용할 때 음향 다양성의 변화를 감지하는 데 효과적임을 보여줍니다.



### Social Choice for AI Alignment: Dealing with Diverse Human Feedback (https://arxiv.org/abs/2404.10271)
Comments: 15 pages, 4 figures

- **What's New**: 이 논문은 인간의 피드백에서 강화 학습(RLHF)과 헌법적 AI(CAI)를 통해 대형 언어 모델(LLMs)의 가치 정렬 문제를 해결하는 데 초점을 맞추고 있습니다. 특히, 사회 선택 이론의 도구와 이론을 이러한 문제들에 적용하여 다양한 이해관계자로부터의 피드백을 통합하고 모델 행동에 대한 대표적인 결정을 내리는 방법을 연구합니다.

- **Technical Details**: 이 논문은 RLHF와 CAI 접근 방식을 중심으로 설명하며, 여기서 RLHF는 단순히 인터넷 데이터에 기반한 사전 훈련된 모델보다는 인간의 판단을 기반으로 모델을 미세 조정(fine-tune)하여 '도움이 되고 해를 끼치지 않는' 출력을 생성하도록 합니다. 반면, CAI는 인간이 직접 '헌법'을 제작하여 LLM의 훈련 과정을 안내하는 원칙을 명시합니다. 사회 선택 이론은 이러한 접근법들에서 발생할 수 있는 여러 문제들을 해결하는 데 적합한 이론적 체계를 제공합니다.

- **Performance Highlights**: 이 접근법을 사용함으로써, 다양한 그룹의 사람들로부터의 입력이나 피드백을 고려함으로써 시스템이 공정하게 구성되고, 진실성에 대한 보다 정확한 피드백을 제공할 가능성이 높아집니다. 또한, 사회 선택 이론은 피드백이 일관성이 없고 시스템이 불일치하는 행동을 보일 수 있는 문제를 피하는 방법에 대한 통찰력을 제공합니다.



### MoE-TinyMed: Mixture of Experts for Tiny Medical Large Vision-Language  Models (https://arxiv.org/abs/2404.10237)
- **What's New**: MoE-TinyMed는 의료 분야에서 매개 변수 수요를 크게 줄이면서 실적을 향상시킨 모델입니다. 이 모델은 자원이 제한된 의료 환경에서 효과적인 대안으로, 특히 VQA-RAD, SLAKE, Path-VQA 데이터셋에서 LLaVA-Med를 초과하는 성능을 보여주었습니다. MoE-TinyMed는 3.6B 매개변수만을 사용하여 더 큰 모델들과 비교할 때 높은 효율성을 제공합니다.



### Two-Stage Stance Labeling: User-Hashtag Heuristics with Graph Neural  Networks (https://arxiv.org/abs/2404.10228)
- **What's New**: 이 연구에서는 소셜 미디어 사용자의 입장을 연구하는 데에 도전이 되는 높은 볼륨과 빠른 내용 변화를 해결하기 위해 개발된 두 단계 입장 라벨링 방법을 소개합니다. 특히, 사용자-해시태그 이분 그래프(bipartite graph)와 사용자-사용자 상호작용 그래프(user-user interaction graph)를 활용한 새로운 접근 방식을 설명합니다.

- **Technical Details**: 첫 번째 단계에서는 사용자-해시태그 이분 그래프를 사용하여 라벨 전파 메커니즘(label propagation mechanism)을 통해 사용자와 해시태그 노드의 입장 연관성을 반복적으로 업데이트하는 간단하고 효율적인 휴리스틱(heuristic)을 적용했습니다. 이 초기 세트의 소프트 라벨(soft labels)은 그 다음 사용자-사용자 상호작용 그래프와 통합되어 반감독 학습(semi-supervised learning)을 사용하여 그래프 신경망(Graph Neural Network, GNN) 모델을 훈련시킵니다.

- **Performance Highlights**: 이 방법은 기후 변화와 총기 통제와 같은 민감한 주제에 대한 두 개의 대규모 데이터셋에서 평가되었으며, GPT4와 같은 제로샷(zero-shot) 입장 라벨링을 사용하는 기존 방법들보다 우수한 성능을 보였습니다. 분석을 통해 입장 라벨링 정보와 상호작용 그래프가 어떻게 소셜 미디어 상호작용의 극단화를 평가하는 데 사용될 수 있는지를 보여줍니다.



### Find The Gap: Knowledge Base Reasoning For Visual Question Answering (https://arxiv.org/abs/2404.10226)
- **What's New**: 이 연구에서는 지식 기반 시각적 질문 응답(Knowledge-based Visual Question Answering, KB-VQA) 문제를 분석하였습니다. 연구진은 문제에 관련된 시각적 자료를 기반으로 하고, 주어진 큰 지식베이스(Knowledge Base, KB)에서 관련 지식을 검색하여 답을 제공해야 하는 모델을 설계하고 평가하였습니다. 이 분석은 신경망 아키텍처 설계 및 스크래치부터 훈련하는 방법과 이미 학습된 대규모 언어 모델(Large Language Models, LLMs)을 사용하는 방법 두 가지 측면에서 이루어졌습니다.

- **Technical Details**: 연구진은 명시적으로 KB에서 관련 정보를 검색하여 KB-VQA 문제를 해결할 수 있는지, 그리고 특정 작업 모델과 LLM 모델이 시각적 지식과 외부 지식을 통합하고, 이 두 가지 정보원을 사용하여 다단계 추론을 수행하는 성능을 어떻게 비교하는지에 대해 질문합니다. 이 연구에서는 대조 손실(contrastive loss)을 사용하여 질문의 임베딩(embedding)과 그에 대응하는 지원 사실의 임베딩 간의 유사성을 최대화하고, 관련 없는 사실과의 유사성을 최소화하는 지식 검색 모델을 학습하였습니다. 또한, 이미지의 해당 장면 그래프(scene graph, SG)에서 질문 관련 시각적 정보를 추출하기 위해 유사한 검색 모듈을 사용했습니다.

- **Performance Highlights**: 연구 결과, 과업 특화 모델(task-specific models)과 LLM 모델에 검색된 외부 및 시각적 지식을 통합하는 것이 긍정적인 영향을 미친다는 것을 보여주었습니다. 그러나 LLM은 단일 단계 추론(1-hop reasoning)에서 강력하지만, 모두 모드의 지원 지식이 있는 경우에도 두 번째 단계의 추론(2-hop reasoning)에서는 신경망 모델(Neural Network, NN)에 비해 성능이 떨어지는 경향이 있습니다. 추가적으로, LLM이 KB 관련 질문에선 NN 모델보다 더 좋은 성능을 보였지만, 외부 KB의 필요성을 완전히 해결하지는 못했습니다.



### ANCHOR: LLM-driven News Subject Conditioning for Text-to-Image Synthesis (https://arxiv.org/abs/2404.10141)
Comments: 23 pages, 9 figures

- **What's New**: 이 연구에서는 실제 뉴스 캡션을 텍스트로부터 이미지를 생성하는(Text-to-Image, T2I) 모델의 성능을 평가하기 위해 'Abstractive News Captions with High-level cOntext Representation (ANCHOR)' 데이터셋을 처음으로 소개합니다. 이 데이터셋은 뉴스 미디어 기관 5곳에서 수집한 70,000개 이상의 샘플을 포함하고 있으며, 물리적 객체 설명에 제한을 두고 상황 및 명명된 개체(Named-Entity, NE) 정보를 중심으로 구성되어 있습니다.

- **Technical Details**: ANCHOR 데이터셋은 높은 수준의 맥락 정보를 포함하는 추상적인 뉴스 캡션을 기반으로 하며, 이는 전통적인 설명 캡션과 구별됩니다. 본 연구는 Large Language Models (LLM)를 사용하여 이러한 추상적 캡션에서 주요 주제를 식별하고 이해하는 능력을 탐구합니다. 연구팀은 'Subject-Aware Finetuning (SAFE)' 방법을 제안하여, LLM이 생성한 주제 가중치를 활용하여 합성 이미지에서 주요 주제의 표현을 강화하고 뉴스 이미지 및 캡션의 도메인 분포에 적응하도록 합니다.

- **Performance Highlights**: SAFE 방법은 기존 T2I 모델들을 능가하는 성능을 보여, ANCHOR 데이터셋에서의 평가에서 현저히 더 나은 결과를 달성했습니다. 이는 SAFE가 뉴스 캡션의 복잡한 문장 구조와 상황 맥락 정보를 더욱 효과적으로 처리할 수 있음을 시사합니다.



### Chinchilla Scaling: A replication attemp (https://arxiv.org/abs/2404.10102)
- **What's New**: 이 연구에서는 Hoffmann et al. (2022)이 제안한 컴퓨트-최적화 스케일링 법칙(compute-optimal scaling law)을 추정하는 세 가지 방법 중 세 번째 방법을 재현하려고 시도했습니다. 이 방법은 그들의 그래프에서 데이터를 재구성하고 파라메트릭 손실 함수(parametric loss function)를 적합하는 것을 포함합니다. 그러나 이 방법으로 추출된 데이터 적합과 Hoffmann et al.이 보고한 추정치는 일관성이 없으며, 다른 두 가지 추정 방법과도 일치하지 않았습니다.

- **Technical Details**: 데이터 재구성 과정에서 Hoffmann et al.의 Figure 4에서 SVG 포맷으로 저장된 데이터를 파싱하여 모델의 크기 및 훈련 토큰 수에 해당하는 좌표를 추출했습니다. 추출된 데이터를 이용하여 손실 함수 L(N,D)=E+(A/N^α)+(B/D^β)를 적용, 여기서 N은 모델 매개변수의 수를 나타내고 D는 훈련 토큰 수를 나타냅니다. 이 데이터를 기반으로 Hoffmann et al.의 추정과 상이한 결과를 도출했습니다.

- **Performance Highlights**: 분석 결과, Hoffmann et al.이 제시한 신뢰 구간은 현실적이지 않을 정도로 좁았습니다. 실제로 그들의 데이터 세트 크기를 고려할 때 적절한 통계 절차를 사용한다면 얻을 수 없는 결과였습니다. 또한, 우리의 재추정은 다른 접근법을 통해 도출된 스케일링 정책과 일치함을 보였으며, 이는 Hoffmann et al.의 추정치가 비현실적인 간격을 보고한 것과 비교되었습니다.



### AIGeN: An Adversarial Approach for Instruction Generation in VLN (https://arxiv.org/abs/2404.10054)
Comments: Accepted to 7th Multimodal Learning and Applications Workshop (MULA 2024) at the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024

- **What's New**: 최근 몇 년 동안 VLN (Vision-and-Language Navigation) 분야에 대한 연구의 관심이 크게 증가했습니다. 이 연구에서는 VLN 에이전트의 성능 향상을 위해 의미 있고 잘 구성된 합성 지시문을 생성하는 새로운 아키텍처 AIGeN을 제안합니다. AIGeN은 GAN (Generative Adversarial Networks)에서 영감을 받은 것으로, Transformer의 디코더(GPT-2)와 인코더(BERT)를 활용하여 작동합니다.

- **Technical Details**: AIGeN 모델은 Transformer 디코더 (GPT-2)가 이미지 시퀀스를 이용해 에이전트 경로를 설명하는 문장을 생성하고, Transformer 인코더 (BERT)는 실제와 가짜 지시문을 구분하는 역할을 수행합니다. HM3D (Habitat-Matterport 3D Dataset)에서 217K 이동 궤적에 대한 합성 지시문을 생성하여 VLN 방법의 성능이 향상되었음을 보여줍니다.

- **Performance Highlights**: 우리가 제안한 합성 지시문은 REVERIE 및 R2R 데이터셋을 활용한 VLN 모델의 성능을 개선하여 최신 최고 성능(state-of-the-art performance)을 달성했습니다. 또한, 우리 모델의 다양한 설정의 내비게이션 성능을 비교하여 합성 지시문의 품질이 에이전트 트레이닝에 유익함을 입증했습니다.



### Detecting AI Generated Text Based on NLP and Machine Learning Approaches (https://arxiv.org/abs/2404.10032)
- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP)의 발전으로 인공지능(Artificial Intelligence, AI) 모델들이 미래에 인간이 작성한 글과 동일한 글을 생성할 수 있게 될 것입니다. 이는 심오한 윤리적, 법적, 사회적 파장을 일으킬 수 있습니다. 본 연구는 인간이 작성한 텍스트와 전자적으로 생성된 텍스트를 구분할 수 있는 정확한 AI 탐지 모델을 제공함으로써 이 문제에 대응하고자 합니다.

- **Technical Details**: 본 연구의 접근 방법은 XGB Classifier, SVM, BERT 아키텍처를 포함한 딥러닝 모델들을 사용한 머신러닝 방법론을 포함하고 있습니다. 연구 결과 BERT 모델은 인간이 제공한 정보와 AI가 생성한 정보를 구별하는 데 이전 모델들보다 더 우수한 성능을 보였습니다.

- **Performance Highlights**: XGB 분류자(XGB Classifier)와 SVM은 각각 0.84와 0.81의 정확도를 보였으며, 이번 연구에서 가장 높은 정확도를 제공한 BERT 모델은 0.93%의 정확도를 제공하였습니다. 따라서, BERT가 가장 유망한 해결책으로 등장하였으며, AI 생성 텍스트 식별의 현재 상태에 대한 폭넓은 분석을 제공합니다. 연구에서는 지속 가능성, 도덕성 및 환경 문제와 관련하여 다양한 산업에 대한 가능한 이점을 강조하는 사회적 함의를 분석하였습니다.




New uploads on arXiv(cs.CL)

### Agent Workflow Memory (https://arxiv.org/abs/2409.07429)
- **What's New**: 이번 논문은 Agent Workflow Memory (AWM)라는 새로운 방법론을 제안하여 언어 모델 기반 에이전트가 복잡한 웹 내비게이션과 같은 장기적 과제를 보다 효율적으로 해결할 수 있도록 돕습니다. AWM은 재사용 가능한 작업 흐름(workflows)을 추출하고 이를 에이전트의 메모리에 통합하여 향후 작업 해결을 지원합니다.

- **Technical Details**: AWM은 오프라인(offline) 및 온라인(online) 시나리오 모두에 적용 가능하며, 이전 경험에서 작업 흐름을 추출하여 이를 에이전트의 메모리에 통합합니다. 이 방법은 웹 내비게이션 벤치마크인 Mind2Web과 WebArena에서 검증되었으며, 새로운 작업에서 성능을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: AWM은 Mind2Web에서 24.6%, WebArena에서 51.1% 향상된 성공률을 기록하였으며, WebArena의 경우, 고품질 주석 예제가 없더라도 에이전트가 스스로 작업 흐름을 유도하고 이를 활용하여 과제를 효과적으로 해결할 수 있음을 보여주었습니다.



### Towards Fairer Health Recommendations: finding informative unbiased samples via Word Sense Disambiguation (https://arxiv.org/abs/2409.07424)
Comments:
          Accepted for long presentation at the FAcctRec @ Recsys 2024

- **What's New**: 본 연구는 AI를 활용하여 의료 교육 내용의 편향을 감지하고 이를 개선하는 새로운 프레임워크를 제안합니다. 특히, Word Sense Disambiguation (WSD) 모델을 통해 데이터 품질을 향상시키고, Transformer 기반 모델과 대형 언어 모델(LLMs)을 활용하여 편향 탐지 작업을 수행했습니다.

- **Technical Details**: 이 연구는 BRICC 데이터셋을 기반으로 하며, 4,105개의 발췌문으로 구성된 데이터가 사용되었습니다. 연구진은 Word Sense Disambiguation (WSD) 기법을 통해 관련 없는 문장을 걸러내고, DistilBERT, RoBERTa, BioBERT와 같은 다양한 Transformer 모델을 세밀하게 조정하여 편향 탐지 작업을 수행했습니다. 또한, GPT 모델을 사용하여 제로샷(zero-shot) 및 몇 샷(few-shot) 프롬프트를 사용하여 성능을 평가했습니다.

- **Performance Highlights**: 결과적으로, LLMs는 많은 NLP 작업에서 최첨단(SOTA)으로 평가되지만, 편향 탐지에는 적합하지 않음을 발견했습니다. 반면, 세밀하게 조정된 BERT 모델은 모든 평가 지표에서 일반적으로 우수한 성능을 보였습니다. 이 연구는 WSD와 ChatGPT를 활용하여 모델의 편향 탐지 성능 향상에 기여했습니다.



### Enhancing adversarial robustness in Natural Language Inference using explanations (https://arxiv.org/abs/2409.07423)
- **What's New**: 본 논문은 Natural Language Inference (NLI)에 대한 연구로, 기존의 모델들이 적대적 공격에 취약하다는 문제를 다루고 있습니다. 저자들은 모델-불가지론적(defence strategy)인 접근법으로 자연어 설명을 활용하는 방안을 제안하며, 설명 생성과 분류의 조합을 통해 공격에 대한 강건성을 높이는 방법을 모색합니다.

- **Technical Details**: 연구에서는 'ExplainThenPredict' 프레임워크를 기반으로 하여 주어진 전제 및 가설 쌍으로부터 자연어 설명을 생성하고, 이후 이를 사용하여 최종 레이블(entailment/neutral/contradiction)을 예측합니다. 설명 생성을 위해 Seq2Seq 모델을 사용하고, 출력된 설명을 기반으로 Expl2Label 분류기를 통해 최종 레이블을 결정합니다.

- **Performance Highlights**: 실험 결과, 설명을 기반으로 한 NLI 분류는 다양한 적대적 공격 하에서도 더 높은 강건성을 보였습니다. 또한, 언어 생성 메트릭과 인간의 인식 간의 상관관계를 분석하여 설명의 질이 모델의 강건성과 어떻게 연결되어 있는지에 대해 연구하였습니다.



### AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledg (https://arxiv.org/abs/2409.07394)
Comments:
          16 pages, Code: this https URL

- **What's New**: 이 논문에서는 AdaCAD라는 새로운 동적 디코딩 방법을 제안하여, 대규모 언어 모델(LLM)의 파라메트릭 지식과 외부 컨텍스트 간의 지식 충돌 문제를 해결합니다. 이 접근은 지식 충돌의 정도를 동적으로 측정하고, 그에 따른 적절한 조정 비율을 제거함으로써 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: AdaCAD는 Jensen-Shannon divergence (JSD)를 사용하여 출력 분포의 지식 충돌 정도를 측정합니다. 더 높은 JSD는 더 많은 조정이 필요함을 의미하며, 이 값에 따라 컨텍스트 지식과 파라메트릭 지식의 가중치를 조절합니다. 이 방법은 기존의 컨트라스트 기반 방법에 비해 더 나은 성능을 보입니다.

- **Performance Highlights**: AdaCAD는 6개의 다양한 질문-응답(QA) 데이터셋에서 전통적인 접근 방법보다 평균 14.21%의 정확성 향상을 달성하였고, 요약의 사실성(AlignScore)을 5.59정도 향상시켰습니다. 또한, 기존의 정적 컨트라스트 방법이 없는 경우보다 잘 작동하여 진짜 데이터셋에서 성능을 향상시켰습니다.



### Recent Trends of Multimodal Affective Computing: A Survey from NLP Perspectiv (https://arxiv.org/abs/2409.07388)
- **What's New**: 이 논문은 최신 감정 분석 기술을 다루면서 여러 모달리티 간의 상호작용을 탐구합니다. 특히, 자연어 처리(NLP)의 관점에서 다중 모달 감정 계산의 최근 동향을 조사하고 있습니다.

- **Technical Details**: 이 연구에서는 다중 모달 감정 컴퓨팅(MAC)이라는 개념을 다루며, 이를 통해 텍스트, 오디오 및 비주얼 데이터를 통합하여 감정을 분석하는 방법론을 제시하고 있습니다. 주요 과제로는 다중 모달 감정 분석(Multimodal Sentiment Analysis), 대화 내 다중 모달 감정 인식(Multimodal Emotion Recognition in Conversation), 다중 모달 측면 기반 감정 분석(Multimodal Aspect-Based Sentiment Analysis), 다중 레이블 감정 인식(Multimodal Multilabel Emotion Recognition)이 포함됩니다.

- **Performance Highlights**: 다중 모달 학습과 전이 학습(Transfer Learning) 기법을 활용하여, 다양한 모달리티 간의 데이터 통합을 통해 성능 향상이 이루어졌습니다. 특히, 최근 연구에서는 CLIP, BLIP와 같은 다중 모달 사전 학습 모델들이 성과를 보이고 있으며, 이러한 모델들은 감정 분석의 정확도를 높이는 데 기여하고 있습니다.



### Awaking the Slides: A Tuning-free and Knowledge-regulated AI Tutoring System via Language Model Coordination (https://arxiv.org/abs/2409.07372)
- **What's New**: Slide2Lecture는 사용자 맞춤형 학습 경험을 제공하는 지능형 튜터링 시스템으로, 슬라이드를 구조화된 교수 계획으로 변환하고, 학생의 학습 요구에 맞춘 인터랙티브 강의를 생성하는 기능을 갖추고 있습니다.

- **Technical Details**: Slide2Lecture는 입력된 강의 슬라이드를 처리하여 텍스트와 시각적 정보를 포함한 통합 표현으로 변환합니다. 그 후, 여러 유형의 교수 행동으로 형식화하며, 최종적으로 교수를 위한 인터랙티브 튜터링 환경을 관리합니다.

- **Performance Highlights**: Slide2Lecture는 3000회의 강의 세션 동안 20만 건 이상의 학생과의 상호작용을 기록하였고, 사용자 피드백에 기반하여 효과성을 입증하였습니다.



### Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation (https://arxiv.org/abs/2409.07355)
- **What's New**: 이 연구에서는 인공지능의 검사 및 평가 품질을 향상시키기 위해 인간과 대형 언어 모델(LLM)을 통합한 InteractEval이라는 프레임워크를 소개합니다. 이 프레임워크는 Think-Aloud(생각 소리 내기, TA) 방법을 활용하여 체크리스트 기반의 텍스트 평가를 위한 속성을 생성합니다.

- **Technical Details**: InteractEval은 TA 방법을 통해 인간과 LLM의 아이디어를 결합하며, 이 과정에서 생성된 속성을 바탕으로 LLM이 평가 차원에 대한 체크리스트를 만들고 텍스트를 평가하는 방식입니다. 주된 과정은 TA를 통한 텍스트 속성 수집, 제안된 속성으로부터 주요 요소 추출, 그리고 각 요소와 관련된 질문 생성을 포함합니다.

- **Performance Highlights**: InteractEval은 일관성과 신뢰성에서 개선된 평가 성능을 보여주며, 전통적인 LLM 기반 평가 방법들과 비교하여 Coherence(일관성), Fluency(유창성), Consistency(일관성) 및 Relevance(관련성)의 네 가지 차원에서 우수한 결과를 보입니다. 특히, 인간은 내부 품질 관련 속성에서 탁월한 반면, LLM은 외부 정렬 관련 속성에서 더 나은 성과를 발휘하여 두 모델의 조합이 최상의 평가 결과를 창출함을 확인하였습니다.



### MEDIC: Towards a Comprehensive Framework for Evaluating LLMs in Clinical Applications (https://arxiv.org/abs/2409.07314)
Comments:
          Technical report

- **What's New**: 이 논문에서는 의료 분야에서 Large Language Models (LLMs)의 평가를 위한 새로운 프레임워크인 MEDIC를 소개합니다. 이는 기존의 benchmark(벤치마크)인 USMLE 이상의 포괄적인 평가가 필요하다는 점을 강조합니다.

- **Technical Details**: MEDIC는 의료적 추론(medical reasoning), 윤리 및 편견(ethics and bias), 데이터와 언어 이해(data and language understanding), 상황 내 학습(in-context learning), 임상 안전(clinical safety) 등 다섯 가지 핵심적인 차원에서 LLM을 평가합니다. 새로운 cross-examination framework를 통해 coverage와 hallucination detection을 정량적으로 평가하며, 참조 출력(reference outputs)이 필요하지 않습니다.

- **Performance Highlights**: MEDIC를 통해 의료 질문-응답, 안전성, 요약, 노트 생성 등 다양한 작업에 대해 LLM의 성능을 평가한 결과, 모델 크기, 기본 모델과 의료적으로 미세 조정된 모델 간의 성능 차이를 보여주었습니다. 이는 특정 모델 강점이 요구되는 응용 분야에 대한 모델 선택에 중요한 의미를 지닙니다.



### Propaganda to Hate: A Multimodal Analysis of Arabic Memes with Multi-Agent LLMs (https://arxiv.org/abs/2409.07246)
Comments:
          propaganda, hate-speech, disinformation, misinformation, fake news, LLMs, GPT-4, multimodality, multimodal LLMs

- **What's New**: 이번 연구는 아랍어 소셜 미디어 콘텐츠 내 프로파간다(Propaganda) 및 혐오(Hate) 밈(Meme) 간의 상관관계를 분석하기 위해 멀티 에이전트 LLM 기반 접근 방식을 제시합니다.

- **Technical Details**: 연구는 프로파간다 및 혐오 밈에 대한 coarse 및 fine-grained 레이블을 추가하여 데이터셋을 확장하고, LLM을 데이터 주석자로 활용하여 비전문가가 처리하기 어려운 복잡한 멀티모달 데이터를 자동으로 주석합니다.

- **Performance Highlights**: 실험 결과는 향후 연구의 기준점이 될 수 있으며, 제공된 데이터셋은 커뮤니티에 공개될 예정입니다.



### Learning Efficient Recursive Numeral Systems via Reinforcement Learning (https://arxiv.org/abs/2409.07170)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문은 강화학습(reinforcement learning, RL)을 사용하여 복잡한 재귀 숫자 시스템이 어떻게 발생하는지를 설명하는 기계적 설명을 제시합니다. 특정 메타 문법(meta-grammar) 아래에서 RL 에이전트가 어휘를 최적화하는 방법을 탐구합니다.

- **Technical Details**: 우리는 Hurford(1975)의 메타 문법(better suited for optimization)의 약간 수정된 버전을 사용하여 RL 에이전트가 레서시를 파레토 최적 구성으로 변화를 줄 수 있음을 보여줍니다. 이 과정에서 개별 숫자와 배수(multiplier)의 쌍을 최적화합니다.

- **Performance Highlights**: 연구 결과, 우리가 얻은 언어는 인간 언어에서 흔히 볼 수 있는 특성을 보여주며, 이는 Denić와 Szymanik(2024) 방식에서 얻어진 언어에서는 발견되지 않았습니다. 또한, 파레토 최적화의 경계를 수술적 방식으로 도달하는 방법과 결과는 인간 언어의 수치 기록 방식의 적합성을 강화합니다.



### A Fine-grained Sentiment Analysis of App Reviews using Large Language Models: An Evaluation Study (https://arxiv.org/abs/2409.07162)
Comments:
          The summary of the project is available at this https URL

- **What's New**: 최근 대규모 언어 모델(LLMs)을 사용한 사용자 리뷰의 기능별 감정 분석이 주목받고 있습니다. 본 논문에서는 GPT-4, ChatGPT, LLama-2-chat 변형 모델을 활용하여 0-shot, 1-shot, 5-shot 시나리오에서 앱 리뷰의 특징과 관련된 감정을 추출하는 성능을 비교하였습니다.

- **Technical Details**: 이 연구는 앱 리뷰에서 기능-감정 쌍(feature-sentiment pairs)을 추출하는 성능을 평가하였으며, LLMs는 zero-shot 및 few-shot 설정에서 사용되었습니다. 연구 결과 GPT-4 모델이 기존의 규칙 기반 접근법보다 23.6% 더 나은 f1-score를 기록하며 성능이 뛰어났습니다. 또한, 5-shot 설정에서 성능이 6% 더 향상되었습니다.

- **Performance Highlights**: GPT-4는 정확한 앱 기능에 대해 긍정적 감정을 예측할 때 74%의 f1-score를 달성하였고, 5-shot 설정에서 7% 향상된 성과를 보여주었습니다. LLMs가 사용자 리뷰의 기능별 감정 요약을 생성하는 데 유망하다는 점이 강조되었습니다.



### Gated Slot Attention for Efficient Linear-Time Sequence Modeling (https://arxiv.org/abs/2409.07146)
Comments:
          Preprint

- **What's New**: 본 논문은 Gated Slot Attention (GSA)라는 새로운 모델을 소개하며, Gated Linear Attention (GLA)에서 영감을 받은 게이팅 메커니즘을 통해 Attention with Bounded-memory-Control (ABC)를 개선합니다.

- **Technical Details**: GSA는 소프트맥스를 통해 연결된 두 개의 GLA 계층으로 구성되며, 문맥 인식 메모리 읽기와 적응형 망각을 활용하여 메모리 용량을 향상시킵니다. 이 설계는 GLA의 하드웨어 효율적인 훈련 알고리즘으로 인해 훈련 및 추론 효율성을 크게 높입니다.

- **Performance Highlights**: GSA는 언어 모델링 및 이해 작업에서 유사한 성능을 보이며, 인컨텍스트 회상이 필요한 작업에서 다른 선형 모델을 상당히 초월하는 성능을 보였습니다. T2R 조정 설정에서 Mistral-7B를 GSA에 조정할 경우, 대규모 순환 언어 모델을 초과하는 성능을 확인하였습니다.



### Leveraging Unstructured Text Data for Federated Instruction Tuning of Large Language Models (https://arxiv.org/abs/2409.07136)
Comments:
          11 pages, work in progress

- **What's New**: 연합 교육(federated learning)에서의 새로운 접근법인 FedIT-U2S를 제안합니다. 이 프레임워크는 비구조화 텍스트(관여하는 클라이언트가 보유하는 데이터를 의미)를 자동으로 구조화된 데이터로 변환하여 수작업으로 주석을 다는 부담을 줄입니다.

- **Technical Details**: FedIT-U2S는 두 가지 주요 단계를 포함합니다: (1) 소수 샘플 기반(few-shot) 데이터 생성, 클라이언트는 비구조화된 데이터와 예시를 결합하여 대형 언어 모델(LLM)을 통해 명령-응답 쌍을 생성합니다. (2) 생성된 데이터에 기반한 연합 교육(procedure of federated instruction tuning) 진행.

- **Performance Highlights**: 세 가지 분야(의학, 지식, 수학)에서의 실험 결과, FedIT-U2S가 기존 LLM의 성능을 일관되게 향상시키는 것으로 나타났습니다.



### Reranking Laws for Language Generation: A Communication-Theoretic Perspectiv (https://arxiv.org/abs/2409.07131)
Comments:
          Preprint

- **What's New**: 이번 논문은 대형 언어 모델(LLM)이 생성하는 허위 정보(hallucination)나 부적절한 답변의 확률을 줄이기 위한 새로운 접근 방식을 제안합니다. 여러 가설을 생성한 뒤 reranker를 통해 최적의 답변을 선택하는 방식으로, 이를 통신 이론의 중복성(redundancy) 개념에 비유합니다.

- **Technical Details**: 저자들은 LLM의 generator를 송신자로, reranker를 수신자로 설정하고, 여러 채널을 통해 전송된 메시지를 기반으로 최적의 출력을 선택하는 방법론을 제안합니다. 특히, N개의 가설을 생성할 때 발생할 수 있는 오류 확률을 감소시키는 조건을 제시하며, Mallows 모델이나 Zipf-Mandelbrot 모델을 통해 불완전한 reranker의 상황에서도 높은 신뢰성을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: 이러한 이론적 기반을 바탕으로, DeepSeek-Coder 7B와 TowerInstruct 13B를 사용한 두 가지 실제 작업에서 reranking 규칙을 테스트하였으며, 우수한 성능을 입증하였습니다. 특히, 각 작업에서 생성된 가설의 품질을 높이는 데 성공하며, 이로 인해 언어 모델의 안전성을 향상시킬 수 있음을 보여줍니다.



### Cross-Refine: Improving Natural Language Explanation Generation by Learning in Tandem (https://arxiv.org/abs/2409.07123)
Comments:
          17 pages; under review

- **What's New**: Cross-Refine이라는 새로운 접근 방식을 소개하며, 두 개의 LLM을 생성자와 비평자로 배치하여 자연어 설명(NLE)을 개선합니다. 이 방법은 추가적인 감독 학습 데이터가 필요하지 않으며, 기존 Self-Refine 방법과 비교해 우수한 성능을 보입니다.

- **Technical Details**: Cross-Refine는 역할 모델링을 활용하여 LLM 두 개의 피드백을 서로 교차 참조하여 초기 설명을 개선합니다. 이를 위해 commonsense question answering, natural language inference 및 fact-checking 같은 세 가지 NLP 작업에서 평가를 수행하였습니다.

- **Performance Highlights**: Cross-Refine는 자기 피드백(self-feedback)만을 이용하는 Self-Refine보다 더 효과적으로 설명을 생성하며, ChatGPT와 같은 강력한 LLM뿐만 아니라 덜 강력한 모델에서도 좋은 성능을 보여줍니다.



### Ontology-Free General-Domain Knowledge Graph-to-Text Generation Dataset Synthesis using Large Language Mod (https://arxiv.org/abs/2409.07088)
Comments:
          18 pages, 9 figures

- **What's New**: 이번 연구는 고품질 G2T(지식 그래프-텍스트) 데이터셋을 생성하기 위한 새로운 방법인 WikiOFGraph를 소개합니다. 이는 LLM(대형 언어 모델)과 Data-QuestEval을 활용하여 생성된 대규모 G2T 데이터셋으로, 5.85M의 일반 도메인 그래프-텍스트 쌍을 포함하고 있습니다.

- **Technical Details**: G2T(GraphtoText) 생성은 (주어, 서술어, 목적어) 형태의 triplet을 자연어로 변환하는 작업입니다. 기존의 PLM(사전 훈련된 언어 모델)들은 고품질의 그래프-텍스트 정렬이 잘된 데이터셋이 필요합니다. 그러나 고품질 일반 도메인 G2T 데이터셋의 부족이 이 연구 분야의 발전을 제한해 왔습니다. 본 연구에서는 LLM과 Data-QuestEval을 연계하여 새로운 데이터셋 WikiOFGraph를 생성하고 이를 통해 그래프-텍스트 일관성을 보장합니다.

- **Performance Highlights**: WikiOFGraph로 미세 조정된 PLM은 기존 데이터셋에서 훈련된 PLM보다 다양한 평가 지표에서 뛰어난 성능을 나타냅니다. 이 데이터셋은 인간이 제작한 GenWiki 테스트 세트와 LLM로 생성된 WikiOFGraph 테스트 세트에서의 성능이 우수하며, 고품질 G2T 시스템 구축에 적합하다는 것을 입증했습니다.



### Understanding Knowledge Drift in LLMs through Misinformation (https://arxiv.org/abs/2409.07085)
Comments:
          13 pages, 3 figures. Accepted at DELTA workshop at KDD 2024

- **What's New**: 이번 연구에서는 최신 Large Language Models (LLMs) 가 잘못된 정보에 대한 반응에서 발생하는 *knowledge drift* 현상과 그로 인한 불확실성을 분석하였다.

- **Technical Details**: 주요 실험은 TriviaQA 데이터셋을 기반으로 하고 있으며, 모델의 반응을 Entropy, Perplexity, Token Probability와 같은 지표를 사용해 평가하였다. LLM들이 잘못된 정보 노출 시 불확실성이 최대 56.6%까지 증가하며, 같은 잘못된 정보를 반복적으로 노출받을 경우 불확실성이 감소하는 현상을 발견하였다.

- **Performance Highlights**: 모델의 불확실성은 잘못된 정보를 포함할 때 초기에는 증가하나, 같은 잘못된 정보에 반복 노출될 경우 감소하여 모델의 원래 지식에서 멀어지는 것으로 나타났다. 이 연구 결과는 LLM의 신뢰성과 강건성 향상에 기여할 수 있는 기초 자료를 제공한다.



### Latent Space Interpretation for Stylistic Analysis and Explainable Authorship Attribution (https://arxiv.org/abs/2409.07072)
Comments:
          8 pages, 8 figures, under review

- **What's New**: 이 논문은 저자 속성 (authorship attribution) 모델의 잠재 공간(latent space)을 해석하는 새로운 접근 방식을 제안합니다. 기존의 연구들은 잠재 벡터 공간에서의 저자와 문서 간의 유사성을 기반으로 했지만, 설명 가능성이 부족했습니다. 우리의 방법은 LLMs를 활용하여 각 잠재 공간에서의 대표 포인트에 대해 문체를 설명하는 자연어 설명을 생성합니다.

- **Technical Details**: 본 연구는 먼저 문서들을 잠재 공간(latent space)에서 클러스터링하여 각 클러스터에 대해 스타일 특성을 생성한 후, 새로운 문서에 대해 가장 유사한 클러스터를 식별하고 그 클러스터의 스타일 특성을 집계하여 문체를 설명합니다. 이 과정에서 LLM을 통해 얻은 스타일 설명이 사용됩니다. 이 모델의 성능은 예측 일치성과 스타일 설명의 품질을 평가하여 확인했습니다.

- **Performance Highlights**: 우리의 방법은 자동 평가에서 원래 모델의 잠재 공간에 비해 최고 0.79의 Pearson 상관관계를 기록해, 기존 기준선의 0.2~0.4보다 우수한 성능을 보였습니다. 또한 인간 평가를 통해, 스타일 설명이 미지의 문서의 저자 속성 추정 향상에 기여하였으며, 평균적으로 +20%의 정확도 향상을 경험했습니다.



### Automated Speaking Assessment of Conversation Tests with Novel Graph-based Modeling on Spoken Response Coherenc (https://arxiv.org/abs/2409.07064)
Comments:
          Accepted by IEEE SLT 2024

- **What's New**: 자동화된 말하기 평가 시스템(Automated Speaking Assessment in Conversation Tests, ASAC)은 L2(제2언어) 화자의 전체적인 말하기 능력을 평가하는 새로운 방법론인 계층적 그래프 모델을 제안합니다. 이 모델은 대화 내의 논리적 흐름의 일관성을 모델링하기 위해 폭넓은 상호작용과 미세한 의미 정보를 통합합니다.

- **Technical Details**: 제안된 방법론은 대화 내용을 그래프로 변환하고, 개별 단어부터 더 넓은 담화 구조까지 여러 계층으로 분리하여 의미 정보를 집계합니다. 또한 반응의 구조적 지식을 다듬어 최종 예측에 통합합니다. 이를 통해 말하기 평가에서 중요 콘텐츠를 식별하는 데 도움을 줍니다.

- **Performance Highlights**: NICT-JLE 벤치마크 데이터셋에서 수행한 광범위한 실험 결과, 제안된 모델은 여러 평가 지표에 대해 예측 정확도를 상당히 향상시키는 효과를 보여줍니다. 이는 대화에서의 일관성 관련 특성을 조사해야 할 필요성을 강조합니다.



### Legal Fact Prediction: Task Definition and Dataset Construction (https://arxiv.org/abs/2409.07055)
- **What's New**: 이번 논문에서는 법적 사실 예측 (Legal Fact Prediction)이라는 새로운 자연어 처리(NLP) 과제를 소개합니다. 이는 법원에서 제공된 증거 목록을 바탕으로 법적 사실을 예측하는 것을 목표로 하며, 이를 통해 재판에 관련된 당사자와 변호사들이 전략을 최적화할 수 있도록 돕습니다.

- **Technical Details**: 법적 사실은 증거가 제시되는 과정에서 판사가 결정하는 사건의 사실을 의미합니다. 논문에서는 공개된 법원 판결문과 재판 기록을 통해 증거 목록을 추출하여 현실적인 데이터셋(LFPLoan)을 구축했습니다. 두 가지 기준을 바탕으로 LLM(대형 언어 모델)을 사용하여 법적 사실을 예측하는 실험을 진행했습니다. 실험에서 법적 사실 예측 작업은 복잡한 추론 과정을 필요로 하며 상반되는 증거를 처리하는 난이도가 있음을 발견했습니다.

- **Performance Highlights**: 법적 사실 예측의 성능은 다양한 방법으로 평가되었으며, LLM을 활용한 기초 방법들이 비트리비얼한 예측 능력을 보였습니다. 그러나 여전히 상당한 발전의 여지가 있으며, 이는 법적 사실 예측이 복잡한 정보를 처리하고 추론할 수 있는 능력을 요구하기 때문입니다. 기본 아이템에 대한 예측은 비교적 높은 정확도를 보였으나, 논쟁의 여지가 있는 법적 사실에 대해서는 저조한 성능을 보였습니다.



### Native vs Non-Native Language Prompting: A Comparative Analysis (https://arxiv.org/abs/2409.07054)
Comments:
          Foundation Models, Large Language Models, Arabic NLP, LLMs, Native, Contextual Understanding, Arabic LLM

- **What's New**: 본 연구에서는 12개의 아랍어 데이터셋(9.7K 데이터 포인트)을 사용하여 11개의 NLP 태스크에 대해 다양한 프롬프트 전략(네이티브 vs. 비네이티브)의 성능을 조사하였습니다. 실험을 통해 비네이티브 프롬프트가 평균적으로 가장 높은 성능을 발휘하며, 혼합 프롬프트와 네이티브 프롬프트가 그 뒤를 잇는다는 결과를 도출하였습니다.

- **Technical Details**: LLM(대규모 언어 모델)의 성능을 극대화하기 위해서는 프롬프트 엔지니어링이 필수적입니다. 연구에서는 제로샷(zero-shot), 피숏(few-shot) 그리고 혼합 프롬프트 전략을 평가했습니다. 특히, Llam 3.1 모델이 비네이티브 프롬프트에 대해 각각 7%와 8% 더 높은 성능을 보였습니다. GPT-4o는 모든 프롬프트 설정에서 가장 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 1. 피숏 프롬프트는 향상된 성능을 보여주었으며, 소량의 학습 데이터에 적합한 설정으로 추천됩니다. 2. 비네이티브 프롬프트는 모든 모델에서 가장 뛰어난 성능을 발휘했습니다. 3. 제로샷 설정은 새로운 태스크에 이상적인 솔루션으로, 모든 모델에서 비네이티브 프롬프트의 성능이 우수했습니다. 4. GPT-4o는 모든 프롬프트 설정에서 최상의 결과를 기록하였습니다.



### Beyond IID: Optimizing Instruction Learning from the Perspective of Instruction Interaction and Dependency (https://arxiv.org/abs/2409.07045)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에 대한 고품질 instruction 세트를 최적화하는 데 있어 기존 연구와는 다르게 서로 다른 instruction 카테고리 간의 상호작용 및 의존성 패턴을 시스템적으로 조사하였습니다.

- **Technical Details**: 이 연구에서는 다양한 instruction 카테고리 간의 상관 패턴을 분석하고, 이를 기반으로 instruction 의존성 분류 체계를 통해 instruction 세트를 최적화하는 선형 프로그래밍(linear programming) 기법을 사용하였습니다. 또한, 특성 태그 시스템을 구축하여 각 instruction에 필요한 능력 및 지식을 세분화하여 실험적으로 검증하였습니다.

- **Performance Highlights**: 다양한 LLM에서 수행된 실험 결과, 본 연구의 방법론이 기존 최첨단 기법에 비해 개선된 성능을 보임을 확인하였으며, 특히 reasoning 관련 및 상식 암기 과제에서 유의미한 성과를 도출했습니다. Qwen 및 Llama 모델에서 실험을 진행해 이들 벤치마크에서 우수한 성과를 나타냈습니다.



### You Have Thirteen Hours in Which to Solve the Labyrinth: Enhancing AI Game Masters with Function Calling (https://arxiv.org/abs/2409.06949)
Comments:
          Wordplay Workshop @ ACL 2024

- **What's New**: 이 논문에서는 AI 게임 마스터의 일관성과 신뢰성을 높이기 위한 혁신적인 접근 방식을 제안합니다. 특히 'Jim Henson's Labyrinth: The Adventure Game'이라는 테이블 기반 롤플레잉 게임의 맥락에서 기능 호출(function calling)을 활용하여 AI 게임 마스터의 서사적 품질 및 상태 업데이트 일관성을 개선하는 방법을 설명하고 있습니다.

- **Technical Details**: 이 연구에서는 AI 게임 마스터를 향상시키기 위해 기능 호출을 통합하는 방법론을 제시합니다. 기능 호출을 통해 게임 특정 제어와 상태 관리를 가능하게 하여 AI 게임 마스터가 게임 규칙과 상태에 일관성 있는 서사를 생성할 수 있도록 합니다. 'Labyrinth' 게임의 시뮬레이션을 구현하고, 인간 평가 및 단위 테스트를 통해 이 접근 방식의 효과를 평가했습니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 게임 플레이 경험을 향상시키고, 게임 상태와의 일관성을 유지하는 데 효과적임을 입증했습니다. AI 게임 마스터의 설계에 대한 통찰력과 가이드라인을 제공하며, 게임 AI와 상호작용 스토리텔링의 발전에 기여합니다.



### A Dataset for Evaluating LLM-based Evaluation Functions for Research Question Extraction Task (https://arxiv.org/abs/2409.06883)
- **What's New**: 본 연구는 머신 러닝 분야의 연구 논문에서 연구 질문(RQ)을 자동으로 추출하는 새로운 데이터셋을 구축했습니다. 이 데이터셋은 GPT-4로 추출한 RQ와 다각적인 인간 평가를 포함합니다.

- **Technical Details**: 연구 문서에서 RQ를 추출하는 것은 기존의 문서 요약(task)과 연결됩니다. 연구초록 및 서론에서 LLM(대형 언어 모델)을 사용하여 RQ를 추출하고, 이를 기반으로 RQ의 품질을 평가하기 위해 인간 평가 점수를 수집하였습니다.

- **Performance Highlights**: 기존의 LLM 기반 평가 함수들이 인간 평가와 높은 상관관계를 보이지 않음을 발견했습니다. 이는 연구 질문 이해 과정의 복잡성을 고려할 때, 새로운 평가 기능의 설계가 필요함을 시사합니다.



### What is the Role of Small Models in the LLM Era: A Survey (https://arxiv.org/abs/2409.06857)
Comments:
          a survey paper of small models

- **What's New**: 이 논문에서는 Large Language Models (LLMs)와 Small Models (SMs)의 관계를 협력과 경쟁이라는 두 가지 관점에서 체계적으로 조사합니다.

- **Technical Details**: LLMs는 인공지능의 일반화된 형태인 AGI를 발전시키는 데 중요한 역할을 하고 있으며, 최근 GPT-4와 LLaMA-405B와 같은 대형 모델들이 개발되었습니다. 그러나 모델 크기를 확장하는 것은 컴퓨팅 비용과 에너지 소비를 기하급수적으로 증가시켜 자원이 제한된 연구자 및 기업들에게 비현실적입니다. 반면에 SMs는 실제 설정에서 자주 사용되지만 그 중요성은 현재 과소평가되고 있습니다.

- **Performance Highlights**: 이 연구는 실무자들에게 작은 모델의 기여를 더 깊이 이해할 수 있도록 돕고, 계산 자원의 보다 효율적인 사용을 촉진하는 통찰을 제공합니다.



### PingPong: A Benchmark for Role-Playing Language Models with User Emulation and Multi-Model Evaluation (https://arxiv.org/abs/2409.06820)
Comments:
          4 main pages

- **What's New**: 본 논문에서는 롤플레잉(역할 놀이) 언어 모델의 능력을 평가하기 위한 새로운 벤치마크를 소개합니다. 이 접근법은 언어 모델 자체를 사용하여 동적, 다회전 대화에서 사용자를 에뮬레이트하고 결과 대화를 평가하는 방식으로 구성됩니다.

- **Technical Details**: 프레임워크는 세 가지 주요 구성 요소로 구성됩니다: 특정 캐릭터 역할을 가정하는 플레이어 모델, 사용자 행동을 시뮬레이션하는 심문자 모델, 대화 품질을 평가하는 심사 모델입니다. 이 방법론은 시스템과 사용자 프롬프트의 결합을 통해 역할을 배정합니다.

- **Performance Highlights**: 자동화된 평가와 인간 주석 간의 비교 실험을 통해 접근법의 유효성을 검증했으며, 여러 기준에서 강한 상관관계를 나타냈습니다. 이 작업은 상호작용 시나리오에서 모델 능력을 평가하기 위한 강력하고 동적인 기반을 제공합니다.



### Decomposition of surprisal: Unified computational model of ERP components in language processing (https://arxiv.org/abs/2409.06803)
- **What's New**: 본 연구는 언어 처리에서의 인지 및 뇌 기능을 다루는 정보 이론 기반 모델을 제안하며, 요소로는 N400 및 P600 ERP(전위파) 구성요소를 포함하고 있습니다. 이 두 신호는 언어 신호에 대한 반응을 통해 설명되고, 각각은 표면적 및 심층적 해석의 차이를 반영하는 것으로 규명됩니다.

- **Technical Details**: 이 모델은 Surprisal Theory(서프라이잘 이론)를 일반화하여 구성됩니다. 연구자는 단어의 정보 내용(정보획득도)을 두 부분(A: 휴리스틱 놀람, B: 불일치 신호)으로 나누어 N400과 P600 신호의 크기를 예측합니다. 두 신호는 현대 NLP 모델을 통해 쉽게 추정될 수 있습니다.

- **Performance Highlights**: 여섯 개의 실험 데이터를 통해 다양한 언어 조작에 대해 ERP 패턴을 성공적으로 시뮬레이션하였으며, 정량적 및 정성적 예측에서도 긍정적인 결과를 도출하였습니다.



### Translating Step-by-Step: Decomposing the Translation Process for Improved Translation Quality of Long-Form Texts (https://arxiv.org/abs/2409.06790)
- **What's New**: 이번 논문은 기존의 기계 번역(Machine Translation) 방식에서 벗어나 단계별 접근법을 제안하며, 번역 과정의 여러 단계를 체계적으로 모델링하여 번역 품질을 개선하고자 한다.

- **Technical Details**: 이 연구는 pre-translation research, drafting, refining, proofreading의 네 가지 단계를 포함한 프레임워크를 통해 기계 번역이 인간 번역 과정과 유사하게 진행되도록 한다. Gemini 1.5 Pro를 활용하여 각 단계에 맞는 지시 프롬프트를 설계하였다.

- **Performance Highlights**: 자동 평가 결과, 단계별 번역 방식이 기존의 zero-shot prompting 접근법 및 초기 인간과 유사한 기준 전략에 비해 큰 번역 품질 향상을 보여주었으며, WMT2024에서 최신 기술을 선보였다.



### SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories (https://arxiv.org/abs/2409.07440)
- **What's New**: 이번 연구에서는 LLMs가 연구 리포지토리에서 결과를 스스로 재현할 수 있는지 평가하기 위한 첫 번째 벤치마크인 SUPER를 소개합니다.

- **Technical Details**: SUPER 벤치마크는 end-to-end 문제 45개, 전문 문제에서 유래된 152개의 하위 문제, 자동 생성된 602개의 문제를 포함하며, 연구자들이 직면하는 현실적인 도전 과제를 포착하려고 합니다.

- **Performance Highlights**: 최고 모델인 GPT-4o는 end-to-end 문제의 16.3%와 시나리오의 46.1%만을 성공적으로 해결하여, 이 문제의 복잡성을 보여 줍니다.



### A Suite for Acoustic Language Model Evaluation (https://arxiv.org/abs/2409.07437)
- **What's New**: SALMon이라는 새로운 평가 도구가 도입되어 다양한 음향적 요소를 고려한 음성 언어 모델(Speech Language Models, SLM)의 평가 기준을 마련했습니다.

- **Technical Details**: SALMon은 음성 언어 모델 평가를 위한 새로운 벤치마크로, 배경 소음, 감정, 화자 정체성, 방의 응답을 고려하며 두 가지 주요 작업으로 음향 일관성(acoustic consistency)과 음향-의미 정렬(acoustic-semantic alignment)을 포함합니다.

- **Performance Highlights**: 본 연구에서는 여러 음성 언어 모델을 SALMon을 통해 평가했으며, 인간은 대부분의 작업에서 90% 이상의 성과를 내는 반면, SLM은 기본적인 음향 불일치를 모델링하고 식별하는 데 어려움을 겪는다는 점을 강조했습니다.



### Synthetic continued pretraining (https://arxiv.org/abs/2409.07431)
- **What's New**: 이 논문에서는 작은 도메인 특화된 데이터셋에서 사전 훈련된 언어 모델의 지식을 효과적으로 학습하기 위한 방법으로, 합성 지속 사전 훈련(synthetic continued pretraining)을 제안합니다. 이 방법은 작은 데이터셋을 활용해 더 큰 학습에 적합한 합성 데이터셋을 생성합니다. 이를 위해 EntiGraph라는 합성 데이터 증강 알고리즘을 사용하여 출처 문서에서 중요한 개체들을 추출하고 이들을 연결하여 다양한 텍스트를 생성합니다.

- **Technical Details**: EntiGraph는 텍스트 코퍼스를 개체 목록으로 나눈 다음, 언어 모델을 사용하여 추출된 개체 간의 관계에 대한 텍스트 설명을 생성합니다. 이 방식은 지식 그래프를 구축하여 코퍼스의 다양한 지식을 제공하려고 합니다. 초점은 기존 내용을 단순히 재구성하는 것이 아니라, 보다 깊이 있는 분석과 응용을 통해 주제를 폭넓게 다루는 것입니다.

- **Performance Highlights**: 결과적으로, EntiGraph를 사용하여 생성된 600M 개의 합성 토큰으로 Llama 3 모델을 계속 사전 훈련한 후, 퀘스쳔 세트에서 QA 정확도를 평가한 결과, 합성 데이터는 실제 문서 접근 없이도 80%의 정확도 향상을 제공합니다. 더군다나, 계속된 사전 훈련된 모델은 퀄리티 책에 관련된 복잡한 질문이나 지침을 처리하는 능력을 보여주며, 이는 지식 전이와 관련된 성능 향상을 나타냅니다.



### What to align in multimodal contrastive learning? (https://arxiv.org/abs/2409.07402)
Comments:
          22 pages

- **What's New**: 이 논문에서는 CoMM(Contrastive MultiModal)이라는 새로운 방법론을 제안하여 여러 종류의 콘텐츠와 형식 간의 상호작용을 보다 효과적으로 학습할 수 있도록 한다. CoMM은 단일 다중 모드 공간에서 모드 간의 소통을 가능하게 하여 이전의 방법과 대비된다.

- **Technical Details**: CoMM은 서로 다른 증강된 다중 모드 피처 간의 상호 정보를 최대화하는 방식으로 다중 모드 표현을 정렬한다. 이 접근 방식은 공유 정보, 협력적인 정보, 고유 정보를 포함하여 모드 간의 상호작용을 측정할 수 있는 강력한 이론적 기초를 가지고 있다.

- **Performance Highlights**: CoMM은 실제 데이터셋에서 다양한 분야에서 최고 수준의 성능을 보여주었으며, 7개의 다중모드 작업에서 최첨단 결과를 달성하였다. 이로 인해 CoMM은 다양한 모드 수와 데이터 유형을 다룰 수 있는 유연한 프레임워크임을 입증하였다.



### Explanation, Debate, Align: A Weak-to-Strong Framework for Language Model Generalization (https://arxiv.org/abs/2409.07335)
- **What's New**: 이 논문은 AI 시스템의 정렬(alignment) 문제를 해결하기 위해 언어 모델(context of language models)에서 약한 모델(weak model)과 강한 모델(strong model) 간의 일반화를 통해 새로운 접근 방식을 제안합니다. 이전 연구에서 인간-에이전트 간의 정렬을 위한 설명 생성을 바탕으로 다중 에이전트 시스템(multi-agent systems) 및 인간-AI 팀(human-AI teams)의 복잡한 동학을 다룹니다.

- **Technical Details**: 약한 모델(weak model)과 강한 모델(strong model)의 개념을 정의한 후, 강한 모델이 약한 모델을 향상시키는 ‘유도 함수(facilitation function)’를 통해 지식 전이를 정형화합니다. 이 방법은 설명 생성을 넘어 AI 모델의 성능을 향상시키기 위한 논쟁 기반 정렬(debate-based alignment)을 통합합니다.

- **Performance Highlights**: 결과적으로, 이 유도 기반 접근법을 통해 모델 성능이 향상되었으며, AI 시스템의 정렬 및 확장 가능한 감시 가능성에 대한 인사이트를 제공하는 것으로 나타났습니다.



### Using Generative Agents to Create Tip Sheets for Investigative Data Reporting (https://arxiv.org/abs/2409.07286)
Comments:
          Short paper to be presented at Computation + Journalism 2024

- **What's New**: 이 논문은 조사를 위한 데이터 보고서를 작성하기 위해 생성적 AI 에이전트를 이용한 시스템을 소개합니다. 이 시스템은 데이터 세트에서 유의미한 정보를 생성하고 정제하기 위해 분석가, 기자, 편집자 등 세 가지 전문화된 에이전트를 사용합니다.

- **Technical Details**: 본 시스템은 OpenAI의 Assistants API를 사용하여 GPT-4의 데이터 분석 및 해석 기능을 활용합니다. 이 생성적 에이전트 파이프라인은 질문 생성, 분석 계획 단계, 실행 및 해석, 그리고 최종 결과 정리 단계를 포함하여, 각각의 역할이 상호 피드백을 제공하여 더 나은 결과를 만들어 냅니다. 데이터를 제공하면 최종 출력으로 tip sheet가 제공됩니다.

- **Performance Highlights**: 실제 조사 사례를 통해 검증한 결과, 우리의 에이전트 기반 시스템이 에이전트가 없는 기준 모델에 비해 일반적으로 더 뉴스 가치가 높고 정확한 통찰력을 생성한다는 것을 보여주었습니다. 이는 생성적 AI가 조사 데이터 보고서 작성에 있어 리드(lead)를 찾는 데 잠재력이 있음을 강조합니다.



### Cross-Dialect Text-To-Speech in Pitch-Accent Language Incorporating Multi-Dialect Phoneme-Level BER (https://arxiv.org/abs/2409.07265)
Comments:
          Accepted by IEEE SLT 2024

- **What's New**: 본 논문에서는 CD-TTS (cross-dialect text-to-speech)라는 새로운 작업을 탐구하여, 비원어 사투리에서 학습된 화자의 음성을 합성하는 방법을 제안합니다. 특히, pitch-accent 언어에서의 활용을 중점적으로 다룹니다.

- **Technical Details**: 이 모델은 세 개의 서브 모듈로 구성되어 있으며, 첫 번째로, 펀엠 편향 잠재 변수 (ALVs)를 조건으로 하여 사투리 음성을 합성하기 위한 기본 TTS 모델을 학습합니다. 두 번째로, 입력 텍스트에서 ALVs를 예측하기 위한 ALV 예측기를 훈련시키며, 이를 위해 다중 사투리 음소 수준의 BERT인 MD-PL-BERT를 사용합니다. 이 모델은 다양한 사투리 간의 공통적이고 독특한 특성을 포착하여 ALV 예측의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 모델은 기존의 전통적 사투리 TTS 방법과 비교했을 때 CD-TTS에서 합성 음성의 사투리 자연스러움을 개선하였습니다. 또한, 평가 실험의 결과 ALV 예측기를 활용하여 합성 음성의 자연성이 증가함을 확인하였습니다.



### LLM-based feature generation from text for interpretable machine learning (https://arxiv.org/abs/2409.07132)
- **What's New**: 본 논문은 기존의 고차원 텍스트 표현 방식인 embeddings와 bag-of-words가 규칙 학습에 부적합하다는 점을 지적하고, LLM(대규모 언어 모델)이 텍스트에서 해석 가능한 특징을 추출하여 이 문제를 해결할 수 있는지를 탐구합니다. 이 방식은 CORD-19와 M17+ 데이터셋을 사용하여 증명되었습니다.

- **Technical Details**: LLM에서 생성된 62개의 특징을 통해 문서 품질 예측을 수행하였으며, 전통적인 SciBERT embeddings의 768개 특징에 비해 훨씬 적은 수의 특징을 사용하면서도 유사한 예측 성능을 보였습니다. 특징은 문서의 방법론적 엄격성, 참신성 및 문법적 정확성과 같은 개념에 직접적으로 연결되어 있습니다.

- **Performance Highlights**: 이 연구에서는 LLM이 생성한 특징들을 사용하여 학술 문서의 인용률을 예측하는 이진 분류 작업과 전문가에게 부여된 등급을 나타내는 순서형 5-클래스 타겟을 포함한 문서 품질 예측을 수행하였으며, 두 데이터셋 모두에서 같은 LLM 특징 세트를 사용하여 일관되게 경쟁력 있는 결과를 얻었습니다.



### Representation Tuning (https://arxiv.org/abs/2409.06927)
Comments:
          9 pages, 6 figures, 6 tables

- **What's New**: 이번 연구에서는 거짓말 또는 정직함과 관련된 활성화 벡터를 대조적 팩트 프롬프트를 통해 찾아, 이를 모델에 직접 조정하여 온라인 제어의 필요성을 없앴습니다. Llama-2-13b-chat 모델을 사용하여 정직성을 포함한 행동의 차원을 연구했습니다.

- **Technical Details**: 연구자는 활성화 벡터를 미세 조정하기 위해 coisine similarity와 표준 토큰 기반 손실을 결합한 이중 손실 함수를 사용했습니다. 이는 활성화 조정(activation steering)의 개념을 확장하여 모델의 가중치를 영구적으로 변경하는 방법입니다.

- **Performance Highlights**: 미세 조정된 모델들은 사실 검증 데이터셋에서 비미세 조정 모델보다 유의미하게 향상된 결과를 보였으며, 비밀스러운 질문 응답에서도 강력한 성능을 보여 온라인 제어 방법보다 효율적임을 입증했습니다.



### NSP: A Neuro-Symbolic Natural Language Navigational Planner (https://arxiv.org/abs/2409.06859)
Comments:
          8 pages

- **What's New**: 이 논문에서는 자연어 입력에서 경로 계획을 위한 신경-기호(Neuro-Symbolic) 프레임워크인 NSP를 제안합니다. NSP는 LLM의 신경 추론 능력을 활용하여 환경의 기호 표현을 생성하고 기호 경로 계획 알고리즘을 실행합니다. 이 과정에서 문법 오류를 수정하고 실행 시간 제약을 충족시키는 피드백 루프가 포함됩니다.

- **Technical Details**: NSP 프레임워크는 자연어(NL) 지침을 기반으로 경로 계획 문제를 해결하며, LLM을 통해 환경의 기호 그래프 표현과 경로 계획 알고리즘을 생성합니다. 알고리즘은 그래프에서 실행되어 솔루션 경로를 생성하고, 신경 생성 과정에 피드백 루프를 통해 문법 오류를 수정합니다.

- **Performance Highlights**: 실험 결과, NSP 접근 방식은 1500개의 경로 계획 문제를 평가하는 데 90.1%의 유효 경로를 생성하였으며, 평균적으로 최신 신경 접근 방법보다 19-77% 더 짧은 경로를 제공합니다.



New uploads on arXiv(cs.IR)

### Dot Product is All You Need: Bridging the Gap Between Item Recommendation and Link Prediction (https://arxiv.org/abs/2409.07433)
- **What's New**: 이 연구에서는 아이템 추천 문제(item recommendation)와 링크 예측(link prediction) 문제의 관계를 탐구하고, 아이템 추천 문제를 링크 예측의 한 사례로 볼 수 있음을 보여줍니다. 이를 통해 두 문제의 경계가 모호해질 수 있다는 점을 제시합니다.

- **Technical Details**: 본 논문은 DistMult, CP, ComplEx 같은 세 가지 인기 있는 팩토리제이션(factorisation) 기반의 링크 예측 모델을 아이템 추천 과제에 적용하여 테스트합니다. 각 모델은 사용자가 아이템과 상호작용할 가능성을 예측하는 데 필요한 관계를 정의하고, 이를 통해 추천 시스템에서의 성과를 평가합니다.

- **Performance Highlights**: 초기 결과는 링크 예측 모델들이 현대 최고의 추천 모델들과 경쟁할 수 있는 정도의 예측 정확도를 보여주었습니다. 또한, 하이퍼 파라미터(hyper-parameter) 설정에 따라 이 모델들을 더욱 발전시킬 가능성도 있음을 확인하였습니다.



### Hierarchical Reinforcement Learning for Temporal Abstraction of Listwise Recommendation (https://arxiv.org/abs/2409.07416)
Comments:
          18 pages, 4 figures

- **What's New**: 최근의 계층적 강화 학습(High-Level Reinforcement Learning, HRL) 기술을 활용하여, 사용자 인식의 발전과 단기 관심 변화를 다각도로 반영한 새로운 추천 시스템인 mccHRL을 제안합니다. 이 시스템은 고수준(High-Level) 에이전트가 사용자 인식을 연구하고, 저수준(Low-Level) 에이전트가 아이템 선정 정책을 수립하도록 구성되어 있습니다.

- **Technical Details**: mccHRL(모바일-클라우드 협력 계층 강화 학습)은  사용자 인식 및 공간시간(spatiotemporal) 상태를 모델링하고, 사용자 반응과의 상호작용을 통해 장기적인 사용자 선호를 행동으로 제안합니다. 저수준 에이전트는 사용자 단기 관심을 기반으로 아이템 선정 문제를 해결하며, 이는 온디바이스(On-device) 기능을 고려합니다.

- **Performance Highlights**: 실험 결과, 기존의 여러 기법에 비해 mccHRL이 상당한 성능 향상을 보였습니다. 또한, 데이터와 코드는 공개되어 연구 커뮤니티에서 접근할 수 있습니다.



### Enhancing Sequential Music Recommendation with Negative Feedback-informed Contrastive Learning (https://arxiv.org/abs/2409.07367)
Comments:
          To-appear at 18th ACM Conference on Recommendation Systems

- **What's New**: 본 연구에서는 음악 추천 시스템에서 사용자 피드백을 통해 세션 기반 추천의 정확성을 향상시키는 방법을 제안합니다. 특히, 사용자가 건너 뛴 트랙을 모델링하여 추천의 품질을 높이는 방식입니다.

- **Technical Details**: 우리는 'sequence-aware contrastive learning' 기법을 사용하여 세션 임베딩 공간 내에서 진짜 다음 추천 항목과 건너 뛴 항목을 구조화합니다. 이는 K-nearest-neighbors 검색을 통해 다음 항목 추천의 순위에 직접적인 영향을 미칩니다.

- **Performance Highlights**: 세 가지 음악 추천 데이터셋에서의 실험 결과, 제안된 방법이 다음 항목 히트율, 항목 순위, 건너뛰기 순위 하락에서 일관된 성과 향상을 달성했습니다.



### STORE: Streamlining Semantic Tokenization and Generative Recommendation with A Single LLM (https://arxiv.org/abs/2409.07276)
- **What's New**: 최근의 추천 시스템은 고유한 항목 식별자(ID)에 의존하는 기존 모델의 한계를 극복하기 위해, semantic tokenization(의미 토큰화)을 도입했습니다. 이는 항목의 의미 표현을 디스리트 토큰의 시퀀스로 변환하여 더 효과적으로 정보를 활용하고, MODEL을 단순화하는 새로운 프레임워크 'STORE'를 제안합니다.

- **Technical Details**: STORE 프레임워크는 항목의 의미 표현을 텍스트에서 토큰으로 변환하는 작업과 토큰 간 추천을 수행하는 하나의 큰 언어 모델(LLM)을 활용하여 구성됩니다. 이 과정에서는 텍스트-토큰 보조 작업 및 토큰-텍스트 재구성 작업을 통해 정보 손실을 최소화하고 지식 전이를 향상시킵니다.

- **Performance Highlights**: MIND와 Yelp 등 두 개의 실제 데이터 세트를 기반으로 한 광범위한 실험을 통해, STORE 프레임워크는 여러 추천 시나리오에서 탁월한 성능을 보여주며, 일반적인 추천 시스템에 대한 광범위한 응용 가능성을 나타냅니다.



### RePlay: a Recommendation Framework for Experimentation and Production Us (https://arxiv.org/abs/2409.07272)
- **What's New**: RePlay는 추천 시스템을 구축하고 비교하기 위한 새로운 오픈 소스 도구입니다. 이 프레임워크는 실험과 생산 모두를 지원하며, 데이터 과학자들이 연구 모드에서 생산 모드로 쉽게 전환할 수 있도록 설계되었습니다.

- **Technical Details**: RePlay는 데이터프레임(Pandas, Polars, Spark) 및 다양한 하드웨어 아키텍처(CPU, GPU, Cluster)를 지원하는 엔드 투 엔드 파이프라인을 제공합니다. 주요 기능으로는 데이터 전처리 모듈, 데이터 분할 전략, 모델 훈련을 위한 데이터셋 클래스 등이 포함됩니다.

- **Performance Highlights**: RePlay는 하이퍼파라미터 튜닝을 위한 Optuna 라이브러리와 통합되어 있으며, 다양한 추천 알고리즘(예: SASRec, BERT4Rec, NeuroMF 등)을 지원합니다. 각 알고리즘은 scikit-learn 스타일의 fit 및 predict 메서드를 사용합니다.



### Negative Sampling in Recommendation: A Survey and Future Directions (https://arxiv.org/abs/2409.07237)
Comments:
          38 pages, 9 figures; Under review

- **What's New**: 이번 논문에서는 추천 시스템(Recommendation Systems, RS)에서 부정 샘플링(Negative Sampling)의 중요성과 그 기법을 심층적으로 탐구합니다. 부정 피드백을 통해 사용자의 진정한 선호도를 더 잘 이해할 수 있는 방법을 제시하며, 이에 대한 기존 연구를 포괄적으로 검토합니다.

- **Technical Details**: 부정 샘플링 전략을 다섯 가지로 분류하고 각각의 기법이 갖는 독특한 특징을 분석합니다. 또한, 실제 RS 시나리오에서 부정 샘플링이 미치는 영향에 대해 상세히 다루며, 이 과정에서 발생하는 주요 도전 과제들을 정리합니다. 이 연구는 추천 알고리즘 훈련 시 긍정적 및 부정적 피드백의 적절한 제공의 중요성을 강조합니다.

- **Performance Highlights**: 180편 이상의 논문을 분석하여 부정 샘플링의 다양한 접근 방식을 체계적으로 정리하고, 추천 시스템에서의 실제적인 도전 과제를 요약하며, 향후 연구 방향에 대한 통찰을 제시합니다. 이 논문은 RS 및 부정 샘플링 분야의 연구자와 개발자에게 유용한 자료로 활용될 수 있습니다.



### E-commerce Webpage Recommendation Scheme Base on Semantic Mining and Neural Networks (https://arxiv.org/abs/2409.07033)
Comments:
          arXiv admin note: text overlap with arXiv:2409.01137

- **What's New**: 이번 논문에서는 전자상거래 웹사이트에서의 웹 페이지 추천 기술의 문제점을 해결하기 위해 의미 기반 웹 마이닝(semantic web mining)과 BP 신경망(BP neural networks)을 결합한 새로운 추천 솔루션을 제안합니다.

- **Technical Details**: 사용자의 웹 로그를 처리하고, 콘텐츠 우선순위(content priority), 시간 소비 우선순위(time consumption priority), 사용자 피드백의 명시적/암시적 피드백(explicit/implicit feedback), 추천 의미(recommendation semantics), 입력 편차(input deviation amount) 등 5가지 특징을 추출합니다. 이 특징들은 BP 신경망의 입력으로 사용되어 최종 출력 웹 페이지의 우선순위를 분류하고 식별합니다.

- **Performance Highlights**: 이 방법은 도서 판매 웹페이지를 샘플로 사용한 실험 결과, 사용자가 필요로 하는 웹 페이지를 빠르고 정확하게 식별할 수 있음을 보여주었습니다.



### Interactive Counterfactual Exploration of Algorithmic Harms in Recommender Systems (https://arxiv.org/abs/2409.06916)
- **What's New**: 이번 연구에서는 사용자들이 추천 시스템에서의 알고리즘적 피해(algorithmic harms)를 이해하고 탐색할 수 있도록 돕는 대화형 도구(interactive tool)를 소개합니다. 이 도구는 시각화(visualizations), 반사적 설명(counterfactual explanations), 인터랙티브 모듈(interactive modules)을 활용하여 편향(bias) 및 필터 버블(filter bubble)의 영향을 조사할 수 있게 지원합니다.

- **Technical Details**: 이 연구는 사용자 인터뷰를 바탕으로 알고리즘적 피해를 탐색할 수 있는 대화형 대시보드(interactive dashboard)를 제안합니다. 주요 정보는 오차 보정(miscalibration), 고정관념(stereotype), 필터 버블 등 세 가지 알고리즘적 피해 유형으로 구분됩니다. 유저 친화적인 체험을 제공하기 위해, 시스템은 명확하고 이해하기 쉬운 설명을 제공하며, 사용자가 자신의 추천 결과를 사회적 맥락(social contextualization)에서 이해할 수 있도록 지원합니다.

- **Performance Highlights**: 이 도구는 사용자 및 연구자에게 투명성을 높이고 맞춤형 영향 평가를 제공하여 알고리즘적 편향에 대한 이해를 촉진합니다. 결국, 이 연구는 더 공정한 추천 결과를 지향하며, 향후 연구 및 실용적 응용에 중요한 통찰을 제공합니다.



### Dual Adversarial Perturbators Generate rich Views for Recommendation (https://arxiv.org/abs/2409.06719)
Comments:
          16 pages,6 figures and 5 tables

- **What's New**: 이 논문에서는 AvoGCL이라는 새로운 이중 적대적 그래프 학습 접근 방식을 제안하며, 이는 구조적 공간과 임베딩 공간 모두에서 대비 학습을 수행하는 GCL 기반 추천 시스템입니다. 또한, 적대적 기법을 활용하여 점진적으로 난이도가 증가하는 대비 뷰를 생성하여 대비 학습의 한계를 끌어올리는 방식으로 설계되었습니다.

- **Technical Details**: AvoGCL은 두 개의 학습 가능한 perturbator, 즉 구조 perturbator와 임베딩 perturbator를 사용하여 그래프 구조 및 임베딩 변화를 타겟으로 하는 더 복잡한 대비 뷰를 생성합니다. 이 방법은 최소화 최대화(minimax) 게임을 통해 원래의 사용자-아이템 상호작용 그래프보다 중복성이 낮은 변형된 그래프를 생성하도록 유도합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에 대한 폭넓은 실험을 통해, AvoGCL은 현재의 기존 GCL 기반 추천 시스템 대비 추천 정확도가 최대 7.1% 향상된 결과를 보였습니다.



### Diff-VPS: Video Polyp Segmentation via a Multi-task Diffusion Network with Adversarial Temporal Reasoning (https://arxiv.org/abs/2409.07238)
- **What's New**: 본 연구는 Diffusion Probabilistic Models를 활용하여 비디오 내 폴립(segmentation) 분할을 진행하는 최초의 네트워크인 Diff-VPS를 제안합니다. 이 모델은 멀티태스킹(multi-task) 감독 학습을 통해 픽셀 간 구분(discrimination) 능력을 향상시키고, Temporal Reasoning Module(TRM)를 도입하여 시간적 의존성을 효과적으로 탐구합니다.

- **Technical Details**: Diff-VPS는 다중 과제를 동시에 수행하는 Multi-task Diffusion Model을 채택하고 있으며, 이는 분류(classification) 및 탐지(detection) 작업을 결합하여 고급(고수준) 정보(semantic information)를 활용합니다. TRM을 통해 이전 프레임에서 대상을 재구성(reconstruct)함으로써 동적 바깥 모습(dynamic appearance)을 포착하며, 생성적 적대(Self-Supervised) 전략을 적용하여 보다 사실적인 프레임을 만듭니다.

- **Performance Highlights**: SUN-SEG 데이터셋에서 수행된 광범위한 실험을 통해 Diff-VPS가 기존 방법들보다 향상된 성능을 기록하였으며, 보이는 비디오 및 보이지 않는 비디오 모두에서 우수한 결과를 나타냈습니다.



### Adversarial Attacks to Multi-Modal Models (https://arxiv.org/abs/2409.06793)
Comments:
          To appear in the ACM Workshop on Large AI Systems and Models with Privacy and Safety Analysis 2024 (LAMPS '24)

- **What's New**: 이 논문에서는 CrossFire라는 혁신적인 다중 모달 모델 공격 방법을 소개합니다. 이 방법은 공격자가 선택한 입력을 원래 이미지나 오디오 파일의 모달리티와 일치하도록 변환하는 것에서 시작합니다.

- **Technical Details**: CrossFire는 공격을 최적화 문제로 구성하고, 변환된 입력과 수정된 이미지 또는 오디오 파일 간의 각도 차이를 최소화하는 것을 목표로 합니다. 이를 통해 원본 미디어에 추가할 변형(Perturbations)을 결정합니다. 실험은 6개의 현실 데이터셋에서 수행되었습니다.

- **Performance Highlights**: CrossFire는 0.98의 공격 성공률을 기록하며, 기존의 공격 방법을 크게 초월하는 성과를 보였습니다. 현재 방어 전략에 대해 평가한 결과, CrossFire에 대항하기에는 현재의 방어가 불충분하다는 것을 확인했습니다.



New uploads on arXiv(cs.CV)

### Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs (https://arxiv.org/abs/2409.07456)
Comments:
          BMVC 2024. Project page: this https URL

- **What's New**: 본 논문에서는 3D Gaussian Splatting (GS)의 깊이 정확성 문제를 해결하기 위해 깊이 priors를 최적화 과정에 통합하는 새로운 전략을 제안합니다. 특히, 외부의 stereo 네트워크에서 제공하는 깊이 단서를 활용하여 GS 모델의 장면 표현을 지속적으로 개선할 수 있습니다.

- **Technical Details**: 논문에서는 깊이 정보 추출을 위한 네 가지 기본 전략인 Structure-from-Motion (SfM), Monocular Depth Estimation (MDE), Depth Completion (DC), Multi-View Stereo (MVS)와 함께 깊이 priors가 GS 최적화 및 장면의 geometry에 미치는 영향을 조사합니다. 또한, 새로운 Self-Evolving GS 프레임워크를 제안하여 stereo 매칭 네트워크와 함께 GS 최적화를 변경합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 GS 모델과 비교하여 더 나은 이미지와 깊이 맵을 생성하며, 특히 실제 sparse view 설정에서 그 성능이 두드러집니다. 다양한 데이터셋(Eth3D, ScanNet++, BlendedMVS)을 통해 이 결과를 검증하였습니다.



### DreamMesh: Jointly Manipulating and Texturing Triangle Meshes for Text-to-3D Generation (https://arxiv.org/abs/2409.07454)
Comments:
          ECCV 2024. Project page is available at \url{this https URL}

- **What's New**: DreamMesh는 텍스트 기반으로 고해상도 3D 모델을 생성하는 혁신적인 아키텍처로, 명확하게 정의된 삼각형 메쉬(triangle meshes)를 사용하여 3D 콘텐츠를 보다 신뢰성 있게 생성하는 방식으로 기존 방법들과 차별화됩니다.

- **Technical Details**: DreamMesh는 두 단계의 학습 과정을 거칩니다. 첫 번째 단계인 coarse 단계에서는 텍스트에 의해 안내된 Jacobians를 사용하여 메쉬를 변형하고, 이어서 2D diffusion 모델을 활용하여 메쉬의 질감을 얻습니다. 두 번째 단계인 fine 단계에서는 coarse 메쉬와 텍스처 맵을 조작하여 고품질 메쉬와 텍스처를 정제합니다. 이러한 과정은 형태와 질감을 명확하게 학습할 수 있도록 돕습니다.

- **Performance Highlights**: 유명한 benchmark인 T3Bench를 통해 DreamMesh는 기존의 텍스트 기반 3D 생성 기법보다 우수한 성능을 입증하였으며, 보다 풍부한 텍스처 디테일과 향상된 기하학적 형태를 통해 현실감 있는 3D 콘텐츠 생성을 가능하게 합니다.



### Hi3D: Pursuing High-Resolution Image-to-3D Generation with Video Diffusion Models (https://arxiv.org/abs/2409.07452)
Comments:
          ACM Multimedia 2024. Source code is available at \url{this https URL}

- **What's New**: 이번 논문은 High-resolution Image-to-3D 모델(Hi3D)을 제안합니다. Hi3D는 2D diffusion 모델의 한계를 극복하고, 비디오 diffusion 모델을 기반으로 단일 이미지를 다중 뷰 이미지로 재정의하는 혁신적인 방법론을 제공합니다.

- **Technical Details**: Hi3D는 먼저 카메라 포즈 조건을 추가하여 사전 훈련된 비디오 diffusion 모델을 활용, 저해상도 3D 인식 시퀀셜 이미지를 생성합니다. 이후 3D 인식 비디오-비디오 리파이너를 통해 이미지를 고해상도로 변환하고, 3D Gaussian Splatting을 통해 새로운 뷰를 증강하여 최종적으로 고품질 3D 메시를 생성합니다.

- **Performance Highlights**: 실험 결과, Hi3D는 여러 뷰 일관성을 가진 고해상도 이미지 생성에서 우수한 성능을 보였습니다. novel view synthesis와 single view reconstruction 작업에서 최첨단 결과를 달성하며, 고해상도 질감 세부사항을 효과적으로 표현합니다.



### FreeEnhance: Tuning-Free Image Enhancement via Content-Consistent Noising-and-Denoising Process (https://arxiv.org/abs/2409.07451)
Comments:
          ACM Multimedia 2024

- **What's New**: 최근 등장한 FreeEnhance 프레임워크는 이미지 생성 후 처리 과정에서 텍스트-이미지 생성 모델과 함께 작용하여 생성된 이미지의 시각적 품질을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: FreeEnhance는 두 단계로 구성된 프로세스를 가지며, 첫 번째 단계에서는 입력 이미지에 랜덤 노이즈를 추가하고, 두 번째 단계에서는 사전 훈련된 Latent Diffusion Models를 사용하여 노이즈 제거 및 이미지 세부사항을 향상시킵니다. 특히, 고주파(High-frequency) 영역에 가벼운 노이즈를 추가하고 저주파(Low-frequency) 영역에 더 강한 노이즈를 추가하여 원본 이미지의 중요한 요소들을 보존하는 방법을 채택하였습니다.

- **Performance Highlights**: HPDv2 데이터셋에서 실시된 광범위한 실험 결과, FreeEnhance는 최첨단 이미지 향상 모델보다 더 우수한 정량적 지표 및 인간 선호도를 보였으며, 상업용 이미지 향상 솔루션인 Magnific AI보다 더 높은 인간 선호도 또한 나타냈습니다.



### StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos (https://arxiv.org/abs/2409.07447)
Comments:
          11 pages, 10 figures

- **What's New**: 본 논문은 2D 비디오를 몰입형 스테레오 3D 컨텐츠로 변환하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존 방법의 한계를 극복하고, 높은 충실도를 필요로 하는 디스플레이 장치에 적합한 결과를 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 시스템은 두 가지 주요 단계로 구성됩니다: 1) 깊이 기반 비디오 스플래팅(depth-based video splatting)으로 왜곡 및 가림(mask) 추출, 2) 스테레오 비디오 인페인팅(stereo video inpainting). 사전 훈련된 스테이블 비디오 디퓨전(stable video diffusion)을 백본(backbone)으로 사용하며, 다양한 길이와 해상도의 입력 비디오를 처리하기 위해 자기회귀(auto-regressive) 전략 및 타일 처리(tiled processing)를 탐구합니다.

- **Performance Highlights**: 이 프레임워크는 2D-3D 비디오 변환의 성능을 크게 향상시켜, Apple Vision Pro와 같은 3D 장치에서 몰입형 컨텐츠를 생성하는 실용적인 솔루션을 제공합니다.



### Deep Neural Network-Based Sign Language Recognition: A Comprehensive Approach Using Transfer Learning with Explainability (https://arxiv.org/abs/2409.07426)
- **What's New**: 이 연구는 심층 신경망(deep neural network)을 이용한 자동 수화 인식(sign language recognition, SLR) 시스템을 제안합니다. 이를 통해 청각 장애인들이 디지털 플랫폼과 통신 장치를 보다 용이하게 사용할 수 있도록 접근성을 향상하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법론은 여러 심층 신경망 아키텍처(ResNet, Inception, Xception, VGG)를 활용하여 수화 이미지를 분류하고, SHAP(Shapley Additive Explanations)를 이용해 모델 출력을 설명합니다. 훈련 및 테스트에는 부탄 수화(Bhutanese-Sign-Language, BSL) 데이터셋이 사용되었으며, ResNet50 모델이 98.90%의 정확도를 기록했습니다.

- **Performance Highlights**: 본 연구는 ResNet50이 98.90%의 높은 정확도로 수화 인식 성능을 보였음을 강조합니다. 또한, SHAP 기법을 통해 모델의 출력 결과에 대한 정보의 명확성을 제공합니다.



### NVRC: Neural Video Representation Compression (https://arxiv.org/abs/2409.07414)
- **What's New**: 본 논문에서는 Neural Video Representation Compression (NVRC)이라는 새로운 프레임워크를 제안합니다. NVRC는 기존의 암시적 신경 표현(Implicit Neural Representation, INR)을 기반으로 하여 비디오 압축을 완전히 end-to-end 방식으로 최적화할 수 있는 첫 번째 접근법입니다.

- **Technical Details**: NVRC는 고급 엔트로피 모델(entropy model) 및 양자화(quantization) 모델을 채택하여 신경 표현의 네트워크 매개변수(parameter)를 최적화합니다. NVRC는 그룹별로 학습된 양자화 매개변수를 사용하여 네트워크 매개변수를 그룹화하고 양자화한 후, 컨텍스트 기반 엔트로피 모델로 인코딩합니다. 엔트로피 모델 매개변수는 경량 엔트로피 모델을 통해 추가적으로 압축됩니다.

- **Performance Highlights**: UVG 데이터셋에서 실험 결과 NVRC는 최신 MPEG 표준 코덱인 H.266/VVC VTM-20.0에 비해 평균 24%의 코딩 게인을 보여주었으며, INR 기반 코덱인 HiNeRV에 비해서는 최대 50%의 비율 절감을 달성했습니다. 이는 NVRC가 VVC VTM을 초과하여 도달한 첫 번째 INR 기반 비디오 코덱으로, 뛰어난 성능을 입증합니다.



### Event-based Mosaicing Bundle Adjustmen (https://arxiv.org/abs/2409.07365)
Comments:
          14+11 pages, 11 figures, 10 tables, this https URL

- **What's New**: 이번 연구는 순전히 회전하는 이벤트 카메라에 대한 모자이크 번들 조정(mosaicing bundle adjustment, BA) 문제를 다루고 있으며, 이를 정규화된 비선형 최소 제곱 최적화로 공식화합니다. 연구는 이벤트를 이미지처럼 변환할 필요 없이 이러한 희소성을 활용하여 최적화를 가속화하는 첫 번째 사례로 알려져 있습니다.

- **Technical Details**: 이 연구에서 제안하는 이벤트 기반 모자이크 번들 조정(event-based mosaicing bundle adjustment, EMBA) 방법은 카메라 동작 궤적과 그래디언트 맵을 정교하게 정제합니다. 연구진은 사건 발생 모델(linearized event generation model, LEGM)을 활용하여 최적화 문제를 정의하고, 이벤트 데이터의 희소성을 통해 시스템 방정식의 블록 대각선 희소성 패턴을 설계하여 효율적인 솔버를 구현합니다.

- **Performance Highlights**: EMBA는 합성 및 실제 데이터 세트에서 50%의 사진 측정 오류 감소를 보여주며, 이전에 숨겨진 장면 디테일을 드러내고, 고해상도 이벤트 카메라를 사용하여 초기 맵 없이 야외 장면에서 고품질 파노라마를 생성할 수 있음을 입증합니다.



### Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks (https://arxiv.org/abs/2409.07353)
- **What's New**: 대형 비전-언어 모델(LVLM)의 안전성을 강화하기 위해 Sim-CLIP+라는 새로운 방어 메커니즘을 제안합니다. 기존 모델에 구조적 수정 없이도 통합할 수 있으며, 적대적인 공격에 대한 저항력을 증대시킵니다.

- **Technical Details**: Sim-CLIP+는 Siamese 아키텍처를 활용하여 CLIP 비전 인코더를 적대적으로 미세 조정합니다. 이는 변형된 샘플과 원래의 샘플 간의 코사인 유사성을 극대화하여 적대적인 조작에 대한 회복력을 촉진합니다.

- **Performance Highlights**: Sim-CLIP+는 COCO와 OKVQA 등의 표준 다운스트림 데이터셋을 사용하여 평가한 결과, 높은 정확도를 유지하면서도 gradient 기반 적대적 공격 및 jailbreak 공격에 대한 저항력을 크게 향상시켰습니다.



### Benchmarking 2D Egocentric Hand Pose Datasets (https://arxiv.org/abs/2409.07337)
- **What's New**: 이 연구는 2D 핸드 포즈 추정에 적합한 최첨단 egocentric 데이터셋에 대한 분석을 수행하며, 데이터셋 평가를 위한 새로운 프로토콜을 제안합니다.

- **Technical Details**: 우리는 egocentric 비디오에서 2D 핸드 포즈 추정에 대한 기존 공개 데이터셋을 선택하여 그 특징과 품질을 분석하고, 최신 핸드 포즈 추정 모델의 성능을 기반으로 데이터셋의 호환성을 평가합니다. 이 프로토콜은 데이터셋의 프레임, 핸드-오브젝트 상호작용, 주석 품질 및 주석된 관절 수에 대한 검증을 포함합니다.

- **Performance Highlights**: H2O 및 GANerated Hands 데이터셋은 각각 가장 유망한 실제 및 합성 데이터셋으로 부각되지만, 현재 이용 가능한 데이터셋들은 특정 용도에 맞춰져 있으며, 이상적인 벤치마크 데이터셋은 아직 존재하지 않습니다.



### Learning to Compress Contexts for Efficient Knowledge-based Visual Question Answering (https://arxiv.org/abs/2409.07331)
- **What's New**: 본 논문에서는 새로운 Retrieval-Augmented MLLM with Compressed Contexts (RACC) 프레임워크를 제안합니다. RACC는 외부 지식 소스에서 수집된 데이터를 압축하고 집계하여 MLLM의 효율적인 추론을 가능하게 합니다.

- **Technical Details**: RACC의 주요 구성요소는 frozen hyperMLLM을 통해 문서를 압축하는 과정, 압축된 프롬프트를 집계하는 aggregator 모듈, 그리고 Key-Value (KV) 캐시 형태로 압축된 변조를 생성하는 Multi-Layer Perceptrons (MLPs)입니다. 이 구조는 효율적인 정보 활용을 통해 추론 처리능력을 극대화합니다.

- **Performance Highlights**: RACC는 OK-VQA 데이터셋에서 62.9%의 최첨단(SOTA) 성능을 달성하였으며, RAVQA-v2와 비교하여 22.0%-59.7%의 추론 지연 시간을 단축시켰습니다. 다양한 MLLM 및 텍스트, 멀티모달 문서와 호환 가능하여 폭넓은 적용성을 보여줍니다.



### Current Symmetry Group Equivariant Convolution Frameworks for Representation Learning (https://arxiv.org/abs/2409.07327)
Comments:
          31 pages, 4 figures

- **What's New**: 최근 논문에서는 기하학적 심층 학습(geometric deep learning)이 현실 세계의 비유클리드(Non-Euclidean) 데이터에 대한 강력한 대응책임을 거론하고 있습니다. 특히 대칭 그룹(equivariant symmetry group)을 활용한 심층 학습 모델이 데이터의 기하학적 변환에 불변적인 특징을 학습하는 데 농후한 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 대칭 그룹에 관한 이론적 기초 및 그래프와 3D 형상에서 활용되는 컨벌루션(convolution) 개념을 다룹니다. 모델들은 주로 정규형(regular), 조작 가능형(steerable), 및 PDE 기반(PDE-based) 컨벌루션으로 분류되며, 각기 다른 기하학적 대칭을 고려하여 설계됩니다. 이 논문은 이러한 기하학적 심층 학습모델이 고차원 특징 공간에서 효율적으로 패턴을 학습하고 일반화할 수 있도록 돕는 방법론을 제시합니다.

- **Performance Highlights**: 이 연구는 그래프 및 다양체(manifold) 데이터에서의 최근의 이론 발전과 실제 구현을 포괄적으로 다루고 있으며, 다양한 대칭별 벤치마크 데이터셋 및 이 방법론의 과학적, 의학적 계산에서의 적용 가능성을 설명합니다. 또한, 이러한 모델들이 제공하는 해석 가능성 및 복잡성 감소, 일반화 성능 향상 등에서의 성과를 문서화합니다.



### Module-wise Adaptive Adversarial Training for End-to-end Autonomous Driving (https://arxiv.org/abs/2409.07321)
Comments:
          14 pages

- **What's New**: 이 논문은 자율주행(AD) 모델을 위한 적대적 훈련을 처음으로 연구하였으며, 새로운 모듈별 적응형 적대적 훈련(Module-wise Adaptive Adversarial Training, MA2T)을 제안합니다. MA2T는 다양한 모듈의 요구를 충족하고, 각 모듈의 기여도에 따라 손실 가중치를 적절하게 조정하는 방식으로 적대적 공격에 대한 모델의 견고성을 높입니다.

- **Technical Details**: MA2T는 두 가지 주요 기법을 채택합니다: 1) 모듈별 노이즈 주입(Module-wise Noise Injection) - 모델의 각 모듈에 입력되기 전에 노이즈를 주입하여 전반적인 목표에 대한 안내를 통해 훈련합니다. 2) 동적 가중치 누적 적응(Dynamic Weight Accumulation Adaptation) - 각 모듈의 기여도에 따라 손실 가중치를 동적으로 조정하는 방법을 사용합니다.

- **Performance Highlights**: 다양한 화이트박스 및 블랙박스 공격 환경에서 nuScenes 데이터 세트를 사용한 실험에서 MA2T는 기존의 적대적 훈련 방법에 비해 5-10%의 향상을 보이며, CARLA 시뮬레이션 환경에서도 자연적인 손상에 대한 저항력을 입증했습니다.



### Data Augmentation via Latent Diffusion for Saliency Prediction (https://arxiv.org/abs/2409.07307)
Comments:
          18 pages, published in ECCV 2024

- **What's New**: 이 논문에서는 이미지의 시각적 주목성을 예측하는 새로운 데이터 증대(data augmentation) 방법을 제안합니다. 기존의 데이터 증대 기술들이 장면 구성(scene composition)을 변화시켜 주목성(prediction model)을 저해하는 반면, 이 방법은 자연 이미지를 편집하면서도 현실 세계의 복잡성과 변동성을 유지합니다.

- **Technical Details**: 제안된 방법은 색상(color), 대비(contrast), 밝기(brightness), 클래스(class)와 같은 광학적(photometric) 및 의미적(semantic) 특성을 결합하여 주목성에 영향을 미치는 고수준(high-level) 및 저수준(low-level) 특징을 모두 학습합니다. 또한, saliency-guided cross-attention 메커니즘을 도입하여 특정 이미지 영역의 광학적 속성을 목표 지향적으로 편집하고, 주목성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 연구의 데이터 증대 방법이 다양한 주목성 모델의 성능을 일관되게 향상시키고, 공개된 주목성 벤치마크에서 우수한 성능을 발휘함을 보여주었습니다. 또한, 사용자 연구를 통해 편집된 이미지에서의 예측 결과가 인간의 시각적 주의 패턴과 밀접하게 일치함을 입증했습니다.



### PaveSAM Segment Anything for Pavement Distress (https://arxiv.org/abs/2409.07295)
- **What's New**: 이 연구에서는 PaveSAM이라는 새로운 zero-shot segmentation 모델을 제안합니다. 이 모델은 기존의 세그멘테이션 모델과는 달리 바운딩 박스 프롬프트를 사용하여 포장 도로의 결함을 세분화할 수 있습니다.

- **Technical Details**: PaveSAM 모델은 180개의 이미지로 마스크 디코더를 재훈련하여, 세분화 작업의 효율성을 극대화합니다. 특히, 기존의 manual annotation 없이도 바운딩 박스만으로도 높은 성능을 발휘합니다. 이 접근은 깊은 학습 기반 모델들이 사람의 손이 필요했던 pixel-level annotations를 대체할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: PaveSAM은 최소한의 입력으로도 높은 성능을 자랑하며, 레이블링의 노력과 비용을 현저히 줄일 수 있는 솔루션을 제공합니다. 또한, 기존에 공개된 오픈소스 포장 도로 결함 이미지들을 활용하여 세분화 마스크를 생성할 수 있어, 데이터셋의 가용성과 다양성을 높이는 데 기여합니다.



### TLD-READY: Traffic Light Detection -- Relevance Estimation and Deployment Analysis (https://arxiv.org/abs/2409.07284)
- **What's New**: 이번 연구는 자율주행 차량에서 효율적인 신호등 감지를 위한 새로운 딥러닝 기반 detection 시스템을 소개합니다. 데이터셋들의 통합을 통해 다양한 시나리오에서 강력한 평가를 보장하며, 방향 화살표 마킹을 활용한 신호 관련성 평가 시스템을 제안하여 이전 맵 생성을 필요로 하지 않도록 혁신적인 방법을 도입했습니다.

- **Technical Details**: 이 연구는 Bosch Small Traffic Lights Dataset, LISA, DriveU Traffic Light Dataset 및 Karlsruhe의 독점 데이터셋 등 여러 데이터셋을 활용하여 리얼 월드 환경에서 신호등 감지 모델을 평가합니다. Lane markings를 보조 지표로 활용하여 신호의 관련성을 평가하는 새로운 방법론을 소개하며, DriveU 데이터셋에서는 96%의 정확도로 관련성 평가를 달성했습니다. 전 연구 결과의 재현성과 추가 연구를 지원하기 위해 모델 가중치와 코드를 공개합니다.

- **Performance Highlights**: 제안된 방법은 실시간 교통에서 자율주행 차량을 사용하여 평가되었습니다. 모델은 공개된 데이터셋에서 훈련된 후 실제 도로에 적용하여 발생하는 도전 과제를 분석하였으며, 목표는 신호등의 관련성과 비관련성을 구별하는 것입니다.



### CCFExp: Facial Image Synthesis with Cycle Cross-Fusion Diffusion Model for Facial Paralysis Individuals (https://arxiv.org/abs/2409.07271)
- **What's New**: 이 논문에서는 Facial paralysis의 자동 진단을 위한 고품질 데이터셋을 합성하는 새로운 접근 방식을 제안합니다. 이를 위해, Cycle Cross-Fusion Expression Generative Model (CCFExp)을 통해 다양한 facial paralysis의 정도와 유형을 잘 나타내는 합성 이미지를 생성할 수 있는 방법을 소개하고 있습니다.

- **Technical Details**: 제안된 모델인 CCFExp는 얼굴 정체성 이미지, 표정 정보 및 랜드마크 정보를 기반으로 한 이미지를 생성하는 새로운 기법입니다. 랜드마크 특징을 활용하여 미세한 얼굴 변형을 포착하며 교차 융합(feature fusion) 방법을 통해 서로 다른 특징 정보를 통합하여 모델의 성능을 향상시킵니다. 또한, 제한된 데이터셋에서의 훈련 효율성을 개선하기 위해 사이클 훈련 전략을 도입하였습니다.

- **Performance Highlights**: 실험 결과, CCFExp는 기존의 최신 기법들을 초월하여 더 고해상도의 얼굴 사진을 생성하고, 동일 인식 능력을 유지하면서 다양한 얼굴 양식과 비대칭성을 잘 묘사함을 입증했습니다.



### Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models (https://arxiv.org/abs/2409.07269)
Comments:
          Accepted as a conference paper at WACV 2025

- **What's New**: 본 논문은 얼굴 스왑(face swapping) 작업에서의 최신 발전을 제시하며, 특히 포즈 변화, 색상 차이 및 가림 현상(occlusion) 등을 해결하기 위한 새로운 접근법을 제안합니다. 기존 방법들이 다양한 문제에 직면한 반면, 우리는 확산 모델(diffusion model)을 보다 효과적으로 활용하는 방법을 개발하였습니다.

- **Technical Details**: 제안된 방법은 얼굴 스왑 문제를 자기 지도(self-supervised) 훈련 중의 인페인팅(inpainting) 문제로 재구성합니다. 이 과정을 통해 ID 일관성 및 인식 유사성을 강화하기 위해 다단계 Denoising Diffusion Implicit Model (DDIM) 샘플링을 도입하여 모델의 성능을 향상시킵니다. 또한 CLIP 기능을 이용하여 대상 이미지에서 포즈, 표정 및 조명 정보를 추출합니다.

- **Performance Highlights**: FFHQ 및 CelebA 데이터셋에 대한 광범위한 실험을 통해 우리의 접근법이 높은 충실도(fidelity)와 현실적인 얼굴 스왑을 제공하며, 최소한의 추론(inference) 시간에 대해 뛰어난 효율성과 견고성을 나타냄을 입증하였습니다.



### MiniDrive: More Efficient Vision-Language Models with Multi-Level 2D Features as Text Tokens for Autonomous Driving (https://arxiv.org/abs/2409.07267)
- **What's New**: 이번 연구에서 제안하는 MiniDrive는 기존의 시각 언어 모델(VLM)과는 달리 Transformer 아키텍쳐에 기반하지 않은 새로운 구조를 채택하였습니다. 이를 통해 자동 운전 시스템에서의 다양한 과제를 더 효과적이고 효율적으로 수행할 수 있도록 돕습니다.

- **Technical Details**: MiniDrive는 Feature Engineering Mixture of Experts (FE-MoE) 모듈과 Dynamic Instruction Adapter (DI-Adapter)를 결합하여 시각적 특징을 처리합니다. FE-MoE는 2D 특징을 텍스트 토큰으로 매핑하며 DI-Adapter는 사용자의 지시에 동적으로 적응하여 시각적 토큰을 변화시킵니다. 참고로, MiniDrive224는 83M의 파라미터만을 포함하고 있으며, 5.9B의 FLOP 수를 기록합니다.

- **Performance Highlights**: MiniDrive는 이전의 VLM들보다 적은 파라미터 수에도 불구하고 질문-응답 능력에서 뛰어난 성능을 보이며, 다중 이미지 입력을 지원하여 자동 운전 시스템에 적합합니다. 더불어 MiniDrive는 및 단일 이미지 표시 시스템 CODA-LM에서 7B 파라미터 이상의 오픈소스 모델을 평균 13.2점 초과하여 성능을 발휘했습니다.



### MRAC Track 1: 2nd Workshop on Multimodal, Generative and Responsible Affective Computing (https://arxiv.org/abs/2409.07256)
Comments:
          ACM MM Workshop 2024. Workshop webpage: this https URL

- **What's New**: 이 논문은 멀티모달 생성 기술의 빠른 발전으로 인해 AI 시스템에 감정 지능이 장착된 것이 초래할 결과에 대한 논의를 촉발하고 있습니다. 특히 감정 AI와 관련 기술을 설계하고 평가하며 구현하는 감정 컴퓨터 연구의 중요성을 강조하고 있습니다.

- **Technical Details**: 감정 컴퓨터 설계는 RGB 이미지, 비디오, 오디오, 텍스트 및 생리학적 신호를 포함한 방대한 멀티모달 데이터가 필요합니다. 감정 지능 모델을 훈련시키는 과정에서 다양한 윤리적 고려사항도 깊이 관여하며, AI 시스템의 발전은 인간의 능력을 보강하고 향상시키는 방향으로 진행되어야 합니다.

- **Performance Highlights**: MRAC 2024 Track 1 워크숍은 이러한 원칙들을 작은 규모의 실험실 환경에서 실제 대규모 컨텍스트로 확장하려는 목표를 가지고 있으며, 감정 컴퓨터의 생성 기술과 윤리적 결과에 대한 잠재적 영향도 강조하고 있습니다. 이는 책임 있는 AI 관점에서 멀티모달 생성 감정 컴퓨터의 전체 범위를 포괄적으로 다루는 첫 번째 워크숍 시리즈입니다.



### EMOdiffhead: Continuously Emotional Control in Talking Head Generation via Diffusion (https://arxiv.org/abs/2409.07255)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 연구에서는 감정을 기반으로 한 Talking Head 비디오 생성 방법인 EMOdiffhead를 제안합니다. 이 방법은 정밀한 감정 제어와 함께 원샷(one-shot) 생성 기능을 제공합니다.

- **Technical Details**: EMOdiffhead는 FLAME 3D 모델을 사용하여 표현 모델링을 수행하고, DECA 방법으로 얼굴 기하학을 복원합니다. 감정 레이블과 대응하는 표현 벡터를 생성하여 오디오 입력과 함께 시간 기반 denoising network를 훈련시킵니다. 이 시스템은 감정 레이블과 표현 벡터 쌍을 사용하여 최종 표현 입력 벡터를 생성하고, 이를 통해 감정과 그 강도에 맞는 비디오를 합성합니다.

- **Performance Highlights**: EMOdiffhead는 기존 감정 기반 Portrait Animation 방법들과 비교하여 최첨단 성능을 달성했습니다. 광범위한 실험과 사용자 연구를 통해 다양한 얼굴 정보와 감정 정보를 효과적으로 학습하고, 감정 정보가 없는 데이터로부터도 감정 비디오 생성을 가능하게 합니다.



### Single-View 3D Reconstruction via SO(2)-Equivariant Gaussian Sculpting Networks (https://arxiv.org/abs/2409.07245)
Comments:
          Accepted to RSS 2024 Workshop on Geometric and Algebraic Structure in Robot Learning

- **What's New**: 이 논문에서는 SO(2)-Equivariant Gaussian Sculpting Networks (GSNs)을 제안하여 단일 관찰 이미지로부터 SO(2)-Equivariant 3D 객체 재구성을 수행하는 방법론을 소개합니다. GSNs는 단일 관찰을 입력으로 받아 Gaussian splat 표현을 생성하여 관찰된 객체의 기하학적 구조와 텍스처를 설명합니다.

- **Technical Details**: GSNs는 공유된 피처 추출기를 사용하여 Gaussian 색상, 공분산(covariance), 위치(position), 불투명도(opacity)를 디코딩하기 전에 처리합니다. 이 방법은 150FPS 이상의 극도로 높은 처리량을 달성하며, 멀티뷰 렌더링 손실(multi-view rendering loss)을 사용하여 효율적으로 훈련될 수 있습니다.

- **Performance Highlights**: GSNs는 기존의 고가의 diffusion 기반 재구성 알고리즘과 품질 측면에서 경쟁력을 보이며, 객체 중심의 잡기를 위한 로봇 조작 파이프라인 내에서 사용할 가능성을 입증하였습니다.



### PiTe: Pixel-Temporal Alignment for Large Video-Language Mod (https://arxiv.org/abs/2409.07239)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전에 힘입어 대형 비디오-언어 모델(Large Video-Language Models, LVidLMs)의 필요성이 대두되고 있습니다. 본 논문에서는 비디오와 언어 간의 정밀한 정렬을 위한 새로운 접근법을 제안하며, 이 과정에서 객체의 이동 궤적을 활용합니다.

- **Technical Details**: 우리는 PiTe라는 새로운 LVidLM을 소개하며, 이 모델은 픽셀 수준에서 시각 데이터와 언어 데이터의 정렬을 통해 비디오 이해를 향상시킵니다. 이 방법은 물체의 이동 궤적을 예측하도록 모델을 훈련시킵니다. 또한, PiTe-143k 데이터셋을 자동 주석 파이프라인을 통해 구성하였으며, 이 데이터셋은 개별 객체의 픽셀 수준 이동 궤적을 포함하고 있습니다.

- **Performance Highlights**: PiTe는 다양한 비디오 관련 다중 모달 작업에서 최첨단 성능을 보이며, 질의 응답, 시간 기초 정렬, 밀집 자막 생성(task들을 zero-shot 조건에서)에서 기존 모델을 큰 차이로 초월합니다.



### Diff-VPS: Video Polyp Segmentation via a Multi-task Diffusion Network with Adversarial Temporal Reasoning (https://arxiv.org/abs/2409.07238)
- **What's New**: 본 연구는 Diffusion Probabilistic Models를 활용하여 비디오 내 폴립(segmentation) 분할을 진행하는 최초의 네트워크인 Diff-VPS를 제안합니다. 이 모델은 멀티태스킹(multi-task) 감독 학습을 통해 픽셀 간 구분(discrimination) 능력을 향상시키고, Temporal Reasoning Module(TRM)를 도입하여 시간적 의존성을 효과적으로 탐구합니다.

- **Technical Details**: Diff-VPS는 다중 과제를 동시에 수행하는 Multi-task Diffusion Model을 채택하고 있으며, 이는 분류(classification) 및 탐지(detection) 작업을 결합하여 고급(고수준) 정보(semantic information)를 활용합니다. TRM을 통해 이전 프레임에서 대상을 재구성(reconstruct)함으로써 동적 바깥 모습(dynamic appearance)을 포착하며, 생성적 적대(Self-Supervised) 전략을 적용하여 보다 사실적인 프레임을 만듭니다.

- **Performance Highlights**: SUN-SEG 데이터셋에서 수행된 광범위한 실험을 통해 Diff-VPS가 기존 방법들보다 향상된 성능을 기록하였으며, 보이는 비디오 및 보이지 않는 비디오 모두에서 우수한 결과를 나타냈습니다.



### Watchlist Challenge: 3rd Open-set Face Detection and Identification (https://arxiv.org/abs/2409.07220)
Comments:
          Accepted for presentation at IJCB 2024

- **What's New**: 본 논문은 비정형 환경에서의 얼굴 인식의 중요성을 강조하며, Watchlist Challenge를 통해 실제 감시 상황에서의 얼굴 탐지(face detection) 및 오픈셋 얼굴 인식(open-set identification) 알고리즘을 평가합니다. 이 평가는 UnConstrained College Students (UCCS) 데이터셋을 기반으로 하며, 새로운 평가 프로토콜을 적용하여 진행됩니다.

- **Technical Details**: UCCS 데이터셋은 생물 인식(biometrics) 기술 및 감시 시스템의 문제를 해결하기 위해 개발되었으며, 이미지 캡쳐는 다양한 날씨 조건에서 이루어졌습니다. 본 연구는 얼굴 탐지와 오픈셋 얼굴 인식라는 두 가지 주요 세그먼트로 나누어져 있으며, 참가자들은 자신의 알고리즘을 제출하여 결과를 공유하고 있습니다. 특히, 데이터셋의 고유한 속성으로는 비협조적(non-cooperative) 상황에서 캡쳐 된 이미지가 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 얼굴 탐지 기능은 강력했으나, 닫힌 세트(closed-set) 식별에서는 성능의 차이가 컸습니다. 대규모 데이터셋에서 사전 훈련된 모델은 상대적으로 우수한 성능을 보여주었지만, 오픈셋 시나리오는 높은 true positive 식별률에서 추가적인 개선이 필요했습니다.



### Enhancing CTC-Based Visual Speech Recognition (https://arxiv.org/abs/2409.07210)
- **What's New**: LiteVSR2는 기존의 Visual Speech Recognition (VSR) 접근 방식을 향상시킨 모델로, 두 가지 주요 개선 사항인 안정화된 비디오 전처리 기술과 지식 증류(knowledge distillation) 과정에서의 특성 정규화(feature normalization)를 도입했습니다. 이러한 개선은 LRS2와 LRS3 벤치마크에서 성능을 크게 향상시켰습니다.

- **Technical Details**: LiteVSR2는 이전의 LiteVSR 프레임워크를 기반으로 하여, 사전 훈련된 ASR 모델에서 지식을 증류하는 방식을 사용하며, 훈련 및 성능 평가에 필요한 데이터 하이퍼파라미터를 대부분 고정해 놓았습니다. ASR 모델을 두 부분으로 나누어 하위 섹션(string _base)과 상위 섹션(head)로 구성하여 실험하였고, CTC 기반의 Conformer 모델을 활용하였습니다.

- **Performance Highlights**: LiteVSR2는 오직 59시간의 레이블이 있는 훈련 데이터로도 LRS2와 LRS3 벤치마크에서 CTC 기반 모델로서 최상의 메트릭스를 달성했습니다. 저작권이 복잡한 비디오 데이터를 제외하고 공개된 데이터셋만을 사용하여 경쟁력 있는 성능을 발휘하며 비공식적인 학습 방식의 경계를 확장했습니다.



### ThermalGaussian: Thermal 3D Gaussian Splatting (https://arxiv.org/abs/2409.07200)
Comments:
          10 pages, 7 figures

- **What's New**: 본 논문에서는 ThermalGaussian이라는 새로운 접근 방식을 제안하며, 이는 RGB 및 열화상 모드에서 고품질 이미지를 렌더링 할 수 있는 첫 번째 열 3D Gaussian splatting (3DGS) 방법입니다. 기존의 3DGS에 기반을 두고 있으며, 멀티모달 정규화 제약 조건을 도입하여 단일 모달리티의 과적합을 방지합니다.

- **Technical Details**: 이 연구는 RGB 카메라와 열화상 카메라를 정렬하고, 학습 과정에서 멀티모달 3D 가우시안 (Gaussian)을 사용합니다. 이를 통해, ThermalGaussian은 3DGS의 장점을 기반으로 하여 고해상도의 열 이미지 및 RGB 이미지를 생성합니다. 또한, RGBT-Scenes라는 새로운 데이터셋을 제공하여 열 장면 재구성을 위한 연구를 지원합니다.

- **Performance Highlights**: ThermalGaussian은 기존 대비 RGB 및 열화상 이미지의 렌더링 품질을 모두 개선하며, 특히 90%의 데이터 저장 공간 절약과 렌더링 속도 향상을 이룩하였습니다. 실험 결과는 ThermalGaussian이 섬세한 열 이미지의 포토리얼리스틱 렌더링을 달성하고 있음을 보입니다.



### Enhancing Angular Resolution via Directionality Encoding and Geometric Constraints in Brain Diffusion Tensor Imaging (https://arxiv.org/abs/2409.07186)
Comments:
          Accepted to ICONIP2024, Diffusion Weighted Imaging, Diffusion Tensor Imaging, Angular Resolution Enhancement, Fractional Anisotropy

- **What's New**: 이번 연구에서는 최소한의 diffusion gradient 방향(6개)으로 획득된 DWI(Diffusion Weighted Imaging)에서도 신뢰할 수 있는 DTI(Diffusion Tensor Imaging) 지표를 추정할 수 있는 깊은 학습 기반의 방법인 DirGeo-DTI를 제안합니다.

- **Technical Details**: DirGeo-DTI는 방향성 인코딩(direction encoding)과 기하적 제약(geometric constraints)을 활용하여 DTI를 향상시키며, 두 개의 공개 DWI 데이터셋을 통해 평가되었습니다. 이 방법은 기존의 DTI 개선 방법에 비해 우수한 성능을 발휘하며, 임상 DWI 스캔으로부터 더 깊은 통찰을 제공할 가능성이 있습니다.

- **Performance Highlights**: DirGeo-DTI는 6개의 고유한 확산 gradient 방향만으로 DTI의 각도 해상도를 향상시키는 최초의 방법으로, 실험 결과 기존 방법들과 비교하여 최상의 성능을 기록했습니다. 이 접근법은 정밀한 DTI 지표를 제공하여 임상에서의 응용 가능성을 높입니다.



### Phy124: Fast Physics-Driven 4D Content Generation from a Single Imag (https://arxiv.org/abs/2409.07179)
- **What's New**: Phy124는 단일 이미지에서 제어 가능한 4D 콘텐츠를 빠르게 생성할 수 있는 물리 기반 방법을 소개합니다. 이 방법은 물리적 시뮬레이션을 통합하여 생성된 4D 콘텐츠가 자연 법칙을 준수하도록 보장하며, 기존의 느린 샘플링 프로세스를 제거하여 신속한 결과를 제공합니다.

- **Technical Details**: Phy124는 이미지를 4D 콘텐츠로 생성하는 데 물리 시뮬레이션을 사용하며, 외부 힘을 조작하여 4D 동역학을 제어할 수 있는 기능을 제공합니다. 이는 사용자 의도에 맞춘 4D 콘텐츠 생성을 가능하게 합니다. 제안된 방법은 시간 소모적인 SDS를 사용하는 기존 방법들과 비교하여 더욱 신속하게 작동합니다.

- **Performance Highlights**: 실험 결과, Phy124는 물리적으로 정확하고 시간 및 공간적으로 일관된 4D 콘텐츠를 생성하며, 높은 충실도를 유지하는 동시에 생성 시간을 대폭 단축시키는 성과를 보였습니다. Phy124는 단일 이미지에서 4D 콘텐츠를 생성하는 데 단 39.5초가 소요됩니다.



### Swin-LiteMedSAM: A Lightweight Box-Based Segment Anything Model for Large-Scale Medical Image Datasets (https://arxiv.org/abs/2409.07172)
Comments:
          13 pages

- **What's New**: 새롭게 소개된 Swin-LiteMedSAM은 강력한 의료 이미지 세분화(model segmentation) 요구를 충족하기 위해 개발된 모델로, 기존 LiteMedSAM보다 더 향상된 성능을 필요 리소스를 줄여 유지하면서 제공합니다.

- **Technical Details**: Swin-LiteMedSAM 모델은 Tiny Swin Transformer를 이미지 인코더(image encoder)로 사용하며, 박스 기반의 포인트(points)와 스크리블(scribble)과 같은 다양한 유형의 프롬프트(prompts)를 통합합니다. 모델 구조는 이미지 인코더, 프롬프트 인코더(prompt encoder), 및 마스크 디코더(mask decoder)로 구성되어 있으며, 스킵 연결(skip connections)을 통해 서로의 출력을 보강합니다.

- **Performance Highlights**: Swin-LiteMedSAM은 CVPR 2024에서 열린 'Segment Anything in Medical Images on Laptop' 챌린지에서 DSC 점수 0.8678과 NSD 점수 0.8844를 달성하며, 라이트MedSAM의 결과 대비 성능이 크게 향상되었습니다. 최종 테스트 세트에서는 DSC 점수 0.8193과 NSD 점수 0.8461을 기록하며 4위에 올랐습니다.



### MVLLaVA: An Intelligent Agent for Unified and Flexible Novel View Synthesis (https://arxiv.org/abs/2409.07129)
Comments:
          project page: this https URL

- **What's New**: MVLLaVA는 여러 다중 보기 확산 모델을 통합한 지능형 에이전트로, LLaVA와 함께 다양한 새로운 보기 합성(novel view synthesis) 작업을 수행할 수 있도록 설계되었습니다. 이 모델은 단일 이미지, 설명 텍스트 또는 특정 시점 변화와 같은 다양한 입력을 처리할 수 있으며, 언어 지침에 따라 시점을 생성할 수 있습니다.

- **Technical Details**: MVLLaVA는 작업별 지침 템플릿을 설계하고 LLaVA를 미세 조정(fine-tune)하여 다양한 사용자의 지침에 유연하게 대응할 수 있도록 했습니다. 모델은 이미지 기반 및 텍스트 기반의 여러 작업을 지원하며, 각 작업에 대해 전처리 및 최적화 과정에서 사용자 친화적인 선택 기능을 제공합니다. LoRA(저랭크 적응) 방식을 통해 제한된 자원에서도 효율적으로 모델을 학습할 수 있습니다.

- **Performance Highlights**: 실험을 통해 MVLLaVA는 다양한 새로운 보기 합성 작업에서 강력한 성능과 다재다능함을 보여주었으며, 기존의 다중 보기 확산 모델의 한계를 극복하고 효율성을 높였습니다.



### Redundancy-Aware Camera Selection for Indoor Scene Neural Rendering (https://arxiv.org/abs/2409.07098)
- **What's New**: 본 연구에서는 모니터링된 정적 장면의 새로운 시점을 합성하기 위해 카메라 선택 문제에 접근하고 있습니다. 이는 인도어(Scene) 환경에서 비디오 시퀀스를 캡쳐하고, 그 과정에서 발생하는 중복 정보를 줄일 수 있는 방법을 제안합니다.

- **Technical Details**: 이 작업에서는 유사성 행렬을 구축하고 Intra-List Diversity (ILD) 메트릭을 활용하여 카메라의 중복성을 평가합니다. 최적화 문제로서 카메라 선택 작업을 정의하고, 이를 기반으로 다양성 기반 샘플링 알고리즘을 적용하여 카메라 선택을 최적화합니다. 새로운 데이터셋 IndoorTraj를 개발하여 복잡한 카메라 움직임을 포함하고 있으며, 이를 통해 실험적으로 제안된 방법의 효율성을 입증하였습니다.

- **Performance Highlights**: 우리의 방법은 전체 데이터셋에 대해 훈련된 모델과 유사한 성능을 달성하였음에도 불구하고 평균적으로 15%의 프레임과 75%의 시간만을 사용하여 최적의 결과를 나타냅니다. 5-20%의 데이터만 사용했음에도 불구하고 기준 전략들을 지속적으로 초과하는 성과를 보였습니다.



### Multimodal Emotion Recognition with Vision-language Prompting and Modality Dropou (https://arxiv.org/abs/2409.07078)
- **What's New**: 본 논문은 제2회 멀티모달 감정 인식 챌린지 트랙 1(MER2024-SEMI)에서의 솔루션을 제시하며, 다양한 멀티모달 감정 인식 방법을 제안합니다.

- **Technical Details**: EmoVCLIP 모델은 CLIP을 기반으로 하는 비전-언어 프롬프트 학습을 통해 감정 인식 작업을 Fine-tuning하여 영상 기반 감정 인식을 위한 다수의 방법을 개발하였습니다. 또한, modality dropout을 통해 강력한 정보 융합을 구현하고, Baichuan의 감정 정보 추출을 개선하기 위해 GPT-4를 제안하였습니다. 모델은 self-training 전략을 사용하여 레이블이 없는 비디오를 활용합니다.

- **Performance Highlights**: 우리의 모델은 MER2024-SEMI 트랙에서 1위를 차지했으며, 테스트 세트에서 90.15%의 정확도를 기록했습니다.



### Edge Modeling Activation Free Fourier Network for Spacecraft Image Denoising (https://arxiv.org/abs/2409.07067)
- **What's New**: 이번 연구에서는 우주선 이미지의 노이즈 감소를 위한 새로운 방법인 Edge modeling Activation Free Fourier Network (EAFFN)를 제안합니다. 기존의 방법들이 우주선 이미지의 특성을 깊이 고려하지 못한 문제점을 해결하고자 하였으며, EAFFN은 Edge Modeling Block (EMB)과 Activation Free Fourier Block (AFFB)를 포함하여 보다 효과적으로 노이즈를 제거합니다.

- **Technical Details**: EAFFN은 두 가지 주요 특성에 초점을 맞추어 설계되었습니다. 첫 번째는 낮은 조도의 우주선 이미지가 많다는 것이고, 두 번째는 반복적인 주기적 구조가 존재한다는 것입니다. EMB는 Sobel Convolution을 통해 우주선 이미지의 엣지와 구조적 정보를 효과적으로 모델링하며, AFFB는 개선된 Fast Fourier Block을 활용해 반복적인 주기적 특징과 장거리 정보를 추출합니다.

- **Performance Highlights**: EAFFN은 우주선 노이즈 이미지 데이터셋에서 기존의 최첨단 방법들과 경쟁력 있는 성능을 보이는 실험 결과를 제시하였습니다. EAFFN은 우주선 이미지의 노이즈 감소에 효과적으로 적응하며, 실제 우주 비행 미션에서의 사용 가능성을 시사합니다.



### Pushing the Limits of Vision-Language Models in Remote Sensing without Human Annotations (https://arxiv.org/abs/2409.07048)
Comments:
          This study was primarily conducted during the latter half of 2023

- **What's New**: 이번 연구는 원거리 감지(domain of remote sensing) 분야에서, 이미지 디코딩 머신러닝 모델을 활용하여 비전-언어 데이터셋을 생성하는 접근 방식을 제안합니다. 이 모델을 통해 960만 개의 비전-언어 쌍 데이터셋을 수집했으며, 이는 인간 주석 레이블 없이 이루어졌습니다.

- **Technical Details**: 제안된 방법론에서는 InstructBLIP 모델을 사용하여 이미지로부터 비전-언어 쌍을 추출합니다. 이 과정에서, 다양한 크기의 이미지들은 512픽셀 정사각형으로 조정되어 InstructBLIP에 입력됩니다. 본 연구에서 생성된 비전-언어 쌍은 총 9,686,720개이며, 이 중 6,278,368개는 InstructBLIP에 의해 생성되었습니다. CLIP 모델은 정보 표현의 유사성을 기반으로 최적화되며, 비전 인코더는 비전 트랜스포머(vision transformer) 기반이며, 텍스트 인코더는 BERT-base 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, RSCLIP 모델은 공개 비전-언어 데이터셋을 활용하지 않은 모델에 비해 우수한 성능을 보였습니다. 특히 제로샷 분류(zero-shot classification), 의미적 위치 지정(semantic localization), 이미지-텍스트 검색(image-text retrieval)과 관련된 다운스트림 작업에서 뛰어난 결과를 기록했습니다.



### SoftShadow: Leveraging Penumbra-Aware Soft Masks for Shadow Remova (https://arxiv.org/abs/2409.07041)
- **What's New**: 최근 딥러닝의 발전은 이미지 그림자 제거(Image Shadow Removal) 작업에 매우 유망한 결과를 가져왔습니다. 그러나 기존의 대부분의 방법들은 이진 형태로 생성된 그림자 마스크를 기반으로 하고 있습니다. 이진 마스크의 특성으로 인해 그림자와 비그림자 영역의 경계에서 아티팩트가 발생할 수 있는 가능성이 있습니다. 이에 따라, 그림자 형성의 물리적 모델에서 영감을 받아 새로운 소프트 그림자 마스크를 제안합니다.

- **Technical Details**: 우리는 SoftShadow라는 새로운 프레임워크를 통해 사전 학습된 SAM(Segment Anything Model)의 우선 지식(prior knowledge)을 활용하여 물리적 제약(physical constraints)을 통합하여 소프트 마스크를 생성합니다. 구체적으로, 우리는 펜umbra 형성 제약 손실(penumbra formation constraint loss)과 그림자 제거 손실(shadow removal loss)을 사용하여 SAM과 후속 그림자 제거 네트워크를 공동 조정합니다. 이 프레임워크는 펜umbra(부분적으로 음영 처리된 영역)와 umbra(완전히 음영 처리된 영역)를 정확하게 예측할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 SoftShadow 프레임워크가 소프트 마스크를 생성함으로써 경계 아티팩트를 더 잘 복원하고, 최신 기술(state-of-the-art) 성능을 달성하며, 우수한 일반화 능력을 보여주었습니다. SoftShadow는 SRD와 LRSS 데이터셋에서 뛰어난 성능을 발휘하며, 기존의 두 가지 그림자 제거 방법보다 더 나은 결과를 보여줍니다.



### Retinex-RAWMamba: Bridging Demosaicing and Denoising for Low-Light RAW Image Enhancemen (https://arxiv.org/abs/2409.07040)
- **What's New**: 본 논문은 RAW 이미지 처리의 맥락에서 새로운 Mamba 스캐닝 메커니즘인 RAWMamba를 제안합니다. 이 메커니즘은 다양한 Color Filter Array (CFA)를 효과적으로 처리할 수 있도록 설계되었습니다. 또한, Retinex Decomposition Module (RDM)을 도입하여 조명과 반사율을 분리하여 더 나은 디노이징과 비선형 노출 보정을 달성합니다.

- **Technical Details**: RAWMamba는 RAW 이미지의 모든 이웃 픽셀을 고려하는 8방향 스캐닝 메커니즘을 특징으로 합니다. 이 스캐닝 메커니즘은 전통적인 방식에 비해 이미지의 공간적 연속성을 보존하면서도 더 나은 특징 추출을 가능하게 합니다. RDM은 Retinex 이론에 기초하여 조명과 반사 지수를 분리하고, 이를 통해 더 효과적인 디노이징과 정확한 밝기 보정이 가능합니다.

- **Performance Highlights**: PUBLIC DATASET인 SID와 MCR에서 실험 결과, 제안된 RAWMamba 방식이 PSNR, SSIM 및 LPIPS의 측면에서 기존의 최첨단 방법들을 초월하는 성능을 보임을 입증하였습니다.



### SCLNet: A Scale-Robust Complementary Learning Network for Object Detection in UAV Images (https://arxiv.org/abs/2409.07024)
- **What's New**: 본 논문에서는 UAV (Unmanned Aerial Vehicle) 이미지에서 객체 탐지의 주요 문제인 스케일 변동과 소형 객체에 대한 비강건성을 다루기 위해 새로운 스케일 강건 보완 학습 네트워크(SCLNet)를 제안합니다. 이 연구는 기존의 탐지 모델들이 이 문제를 효과적으로 해결하지 못하고 있다는 점을 부각시키며, 보완 학습(complementary learning) 접근 방식을 처음으로 UAV 이미지 탐지에 도입한 것이 특징입니다.

- **Technical Details**: SCLNet은 두 가지 구현 및 협력 방법으로 구성되어 있습니다. 첫 번째 구현은 포괄적 스케일 보완 학습(comprehensive-scale complementary learning, CSCL)으로, 스케일 보완 디코더(scale-complementary decoder)와 스케일 보완 손실 함수(scale-complementary loss function)를 이용하여 정보 추출을 명확하게 수행합니다. 두 번째 구현은 상호 스케일 보완 학습(inter-scale contrastive complementary learning, ICCL)으로, 대형 객체의 풍부한 텍스처 정보를 통해 소형 객체의 학습을 안내합니다. 두 구현은 기존의 탐지 모델에 통합되어 end-to-end 협력 방법(ECoop)을 활용합니다.

- **Performance Highlights**: 우리의 SCLNet은 Visdrone 및 UAVDT 데이터셋에서 실시된 실험을 통해 효과성을 입증했습니다. 다양한 스케일의 객체 탐지 성능이 유의미하게 향상되었고, CNN 및 Transformer 기반 모델에 비해 경쟁력 있는 UAV 이미지 객체 탐지 모델로 확인되었습니다. 또한, 시각화 결과는 전경 객체에 대한 우수한 표현 능력을 제공하고 배경 노이즈를 억제할 수 있음을 보여줍니다.



### Insight Any Instance: Promptable Instance Segmentation for Remote Sensing Images (https://arxiv.org/abs/2409.07022)
- **What's New**: 본 논문에서는 원거리 감지 이미지(instance segmentation of remote sensing images, RSIs)의 인스턴스 분할을 위한 새로운 프롬프트 패러다임(prompt paradigm)을 제안합니다. 기존 모델들의 성능 한계를 극복하고, 불균형한 전경(foreground)과 배경(background) 비율 및 제한된 인스턴스 크기 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 기술로 구성됩니다. 첫째, 로컬 프롬프트 모듈(local prompt module)을 통해 특정 인스턴스에 대한 로컬 정보를 추출합니다. 둘째, 글로벌-투-로컬 프롬프트 모듈(global-to-local prompt module)은 글로벌 토큰(global tokens)에서 로컬 토큰(local tokens)으로 문맥 정보를 모델링합니다. 마지막으로, 제안 지역 손실 함수(proposal's area loss function)를 통해 인스턴스 크기를 최적화합니다.

- **Performance Highlights**: 제안된 접근 방식은 다양한 RSIs 데이터셋에서 기존 모델들과의 비교를 통해 효과성을 입증하였습니다. 인스턴스 분할 과정에서 단 40ms의 시간 소비로 경쟁력 있는 성능을 보여줍니다.



### ODYSSEE: Oyster Detection Yielded by Sensor Systems on Edge Electronics (https://arxiv.org/abs/2409.07003)
- **What's New**: 이 논문에서는 부드러운 확산 (Stable Diffusion) 기법을 활용하여 해양 환경에서 필요한 고품질 합성 데이터를 생성하는 새로운 방법을 제시합니다. 이는 수동 식별 방식의 한계를 극복하고 자율 모니터링 시스템을 구현하기 위한 기반이 됩니다.

- **Technical Details**: Aqua2 로봇 플랫폼에서 YOLOv10 기반 비전 모델을 사용하여 실시간 조개 (oyster) 탐지를 수행합니다. 합성 데이터 생성을 위해 ControlNet을 활용하여 실제 수중 이미지의 지리적 구조와 일치하도록 하여 사진 사실적으로 구현합니다.

- **Performance Highlights**: Aqua2 플랫폼에서 조개의 탐지에 대해 0.657 mAP@50을 달성하였으며 이는 최신 기술 상태를 반영합니다. 이 시스템은 해양 보전 및 수산 양식의 효과성을 향상시킬 수 있는 자율 감시 가능성을 보여줍니다.



### AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models (https://arxiv.org/abs/2409.07002)
- **What's New**: 이 논문은 객체 탐지기에 대한 새로운 형태의 패치 공격 프레임워크인 AdvLogo를 제안합니다. AdvLogo는 공격 성능과 시각 품질 간의 균형을 맞추기 위해 의미(semantic) 관점에서 패치 공격을 고려합니다.

- **Technical Details**: AdvLogo는 모든 의미 공간에 정의된 적대적 하위 공간을 활용하여 이미지가 탐지기를 오인식하게 만든다는 가설에 기반합니다. 마지막 타임스텝에서 무조건적 임베딩과 잠재 변수를 교란하여 적대적인 하위 공간으로 드리프트하도록 유도합니다. 또한 푸리에 변환(Fourier Transform)을 통해 주파수 영역에서 잠재 변수에 교란을 적용하여 이미지 품질에 미치는 부정적인 영향을 완화합니다.

- **Performance Highlights**: 실험 결과, AdvLogo는 NAP보다 공격 효과 면에서 일관되게 우수한 성능을 발휘하며, AdvPatch와 비교할 때 향상된 시각 충실도(visual fidelity)와 방어 메커니즘에 대한 우수한 견고성(robustness)을 제공합니다.



### 1M-Deepfakes Detection Challeng (https://arxiv.org/abs/2409.06991)
Comments:
          ACM MM 2024. Challenge webpage: this https URL

- **What's New**: 이번 논문은 1M-Deepfakes Detection Challenge를 통해 AV-Deepfake1M 데이터셋을 기반으로 한 심층 가짜 콘텐츠 탐지 및 위치 지정의 새로운 접근 방식을 제안합니다. 이 데이터셋은 1백만 개 이상의 조작된 비디오를 포함하고 있으며, 연구자들이 깊이 있는 탐지 기술을 개발하도록 유도하는 것을 목표로 합니다.

- **Technical Details**: AV-Deepfake1M 데이터셋은 2,068명의 고유한 피험자에서 수집된 1,886시간의 오디오-비주얼 데이터로 구성되어 있습니다. 이 데이터셋은 Two main tasks로 나뉘며, 첫 번째는 Deepfake Detection(딥페이크 탐지)으로, 주어진 오디오-비주얼 샘플이 딥페이크인지 실제인지 식별하는 작업이고, 두 번째는 Deepfake Temporal Localization(딥페이크 시간적 위치 지정)으로, 수정된 특정 시간 간격을 결정하는 작업입니다. 성능 평가를 위해 AUC, AP 및 AR과 같은 중간 메트릭이 사용됩니다.

- **Performance Highlights**: 논문에서 제안하는 데이터셋과 챌린지는 현재 딥페이크 탐지 방법의 개선에 기여하며, 향후 수년간 딥페이크 기술의 발전을 따라잡기 위한 지속적인 벤치마킹을 목표로 하고 있습니다. 참여자들은 정량적 결과를 제출하여 평가를 받게 되며, 결과는 0에서 1까지의 점수로 표현되며, 1은 완벽한 예측을 의미합니다.



### PanAdapter: Two-Stage Fine-Tuning with Spatial-Spectral Priors Injecting for Pansharpening (https://arxiv.org/abs/2409.06980)
- **What's New**: 이번 논문에서는 pansharpening 작업을 위한 새로운 파라미터 효율적인 파인튜닝 프레임워크인 PanAdapter를 제안합니다. 이 방법은 사전 훈련된 모델의 추가적인 고급 의미 정보를 활용하여 소규모 데이터 세트 문제를 완화합니다.

- **Technical Details**: PanAdapter는 두 단계의 훈련 전략을 채택합니다. 첫 번째 단계에서는 저해상도 멀티스펙트럴 이미지(LRMS)와 고해상도 팬크로매틱 이미지(PAN)로부터 지역적 프라이어(Local Prior)를 추출하는 LPE 모듈을 사용하여 사전 훈련된 CNN 모델을 파인튜닝합니다. 두 번째 단계에서는 두 개의 가측 소스에서 얻은 프라이어들을 Cascaded Adapter를 통해 사전 훈련된 Vision Transformer(ViT) 모델로 주입하고 상호작용하도록 설계된 파라미터 효율적인 모듈을 포함합니다.

- **Performance Highlights**: 제안한 방법은 WV3, QB, GF2 데이터세트에서 여러 pansharpening 방법과 비교하여 최첨단 성능을 달성하였습니다. 특히, 뛰어난 공간적 세부정보와 스펙트럼 정보를 융합하는 능력으로 입증되었습니다.



### Brain-Inspired Stepwise Patch Merging for Vision Transformers (https://arxiv.org/abs/2409.06963)
- **What's New**: 이번 논문에서는 Vision Transformers (ViTs)의 계층적 아키텍처를 위한 새로운 기술, Stepwise Patch Merging (SPM)을 제안합니다. SPM은 Multi-Scale Aggregation (MSA)와 Guided Local Enhancement (GLE)이라는 두 가지 모듈로 구성되어 있으며, 이는 시각 인식의 글로벌 및 로컬 정보를 결합하는 뇌의 능력에서 영감을 받았습니다.

- **Technical Details**: SPM의 MSA 모듈은 다중 스케일 정보를 통합하여 특징 표현을 풍부하게 하고, GLE 모듈은 로컬 세부 사항 추출을 최적화하여 장거리 의존성 모델링과 로컬 특징 향상 간의 최적 균형을 달성합니다. 이 아키텍처는 기존의 계층적 비전 트랜스포머에 쉽게 플러그 앤 플레이 가능하여 유연성을 제공합니다.

- **Performance Highlights**: ImageNet-1K, COCO, ADE20K와 같은 벤치마크 데이터셋에서의 대규모 실험 결과, SPM은 다양한 모델의 성능을 유의미하게 향상시켰으며, 특히 객체 탐지 및 의미론적 분할과 같은 밀집 예측 작업에서 두드러진 개선을 보였습니다.



### Bridging Domain Gap of Point Cloud Representations via Self-Supervised Geometric Augmentation (https://arxiv.org/abs/2409.06956)
Comments:
          10 pages, 6 figures, 5 tables

- **What's New**: 이 논문에서는 3D 포인트 클라우드에서의 무감독 도메인 적응(UDA) 문제를 해결하기 위한 새로운 자기 감독 정규화 방법을 제안합니다. 이를 통해 복잡한 기하학 패턴을 안정적이면서 효과적으로 추출할 수 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 자가 감독 학습 과제를 포함합니다. 첫 번째는 보강된 포인트 클라우드의 이동(distance) 예측을 통한 중심 이동(centroid shift) 완화입니다. 두 번째는 기하학적으로 증강된 샘플 간의 관계를 학습하는 것입니다. 이 두 과제를 통해 포인트 클라우드 표현의 도메인 불변성을 증진시킵니다.

- **Performance Highlights**: PointDA-10 데이터셋에서 실험 결과, 제안된 방법이 최첨단 성능을 달성하며 기존 방법들보다 효과적임을 입증했습니다.



### FSMDet: Vision-guided feature diffusion for fully sparse 3D detector (https://arxiv.org/abs/2409.06945)
Comments:
          Accepted by European Conference on Computer Vision (ECCV) 2024 workshop on VCAD

- **What's New**: 이번 연구에서는 LiDAR(레이저 거리 측정기) 기술을 활용한 완전 희소(fully sparse) 3D 객체 감지의 효율성을 높이기 위해 시각 정보(visual information)를 활용한 FSMDet(Fully Sparse Multi-modal Detection) 모델을 제안합니다. 이 모델은 LiDAR의 feature diffusion 과정을 시각 정보로 안내하여 기존 방법보다 향상된 성능을 발휘합니다.

- **Technical Details**: FSMDet는 두 개의 주요 모듈인 Shape Recover Layer(SRLayer)와 Self Diffusion Layer(SDLayer)를 사용하여 RGB 정보를 활용해 객체의 가시 부분을 복구하고, 이후 특징을 중앙 영역으로 확산시킵니다. SRLayer는 RGB 기능을 사용하여 객체의 형태를 복구하고, SDLayer는 시각 정보를 기반으로 특징을 중앙으로 확산합니다. 이 과정은 기존의 복잡한 사용자 정의 중심 융합(diffusion) 또는 회귀(regression) 연산을 단순화합니다.

- **Performance Highlights**: 실험 결과, FSMDet는 이전의 LiDAR만 사용한 완전 희소 모델들에 비해 성능을 성공적으로 향상시키며, 다중 모달(multimodal) 모델에서도 SOTA(State Of The Art) 성능을 달성했습니다. 또한, 희소 아키텍처 덕분에 추론 과정에서 이전 SOTA 방법보다 최대 5배 더 효율적입니다.



### Automated Body Composition Analysis Using DAFS Express on 2D MRI Slices at L3 Vertebral Lev (https://arxiv.org/abs/2409.06942)
- **What's New**: 이번 연구에서는 MRI 기반의 2D 체성분 분석을 위한 자동화 도구인 Data Analysis Facilitation Suite (DAFS) Express의 유효성을 검증하였습니다. 이는 기존의 수작업 분할 방식을 대체할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 UK Biobank 데이터에서 399명의 참가자 데이터를 선택하였고, 423개의 L3 슬라이스를 분석했습니다. DAFS Express는 SKM (Skeletal Muscle), VAT (Visceral Adipose Tissue), SAT (Subcutaneous Adipose Tissue)의 자동화를 통한 분할을 수행했으며, 이를 전문가가 수동으로 교정해 검증했습니다. 평가 지표로는 Jaccard 계수, Dice 점수, ICC (Intraclass Correlation Coefficients), Bland-Altman 플롯이 사용되었습니다.

- **Performance Highlights**: DAFS Express는 자동 화된 분할 결과와 전문가 수작업 결과 간의 높은 일치를 보였습니다. 평균 Jaccard 점수는 SKM 99.03%, VAT 95.25%, SAT 99.57%였으며, 평균 Dice 점수는 SKM 99.51%, VAT 97.41%, SAT 99.78%였습니다. DAFS Express는 DICOM 파일당 평균 18초 걸리며, 이는 연구 및 임상 환경에서 이미지 분석 프로세스를 간소화할 수 있는 잠재력을 지니고 있습니다.



### Intrapartum Ultrasound Image Segmentation of Pubic Symphysis and Fetal Head Using Dual Student-Teacher Framework with CNN-ViT Collaborative Learning (https://arxiv.org/abs/2409.06928)
- **What's New**: 새로운 논문에서는 팟피코드 관절(Public Symphysis)과 태아 머리(Fetal Head) 세분화를 위한 혁신적인 프레임워크인 Dual-Student and Teacher Combining CNN and Transformer (DSTCT)를 소개합니다. 이는 CNN과 Transformer 모델의 장점을 통합하여 세분화 성능을 향상시킵니다.

- **Technical Details**: 이 프레임워크는 Vision Transformer (ViT)를 교사 모델로 하고, CNN 및 ViT 기반 학생 모델을 포함합니다. Dual-student 설정을 통해 서로의 예측을 통해 상호 감독을 수행하며, 하드 및 소프트 의사 레이블 생성에 집중하였습니다. Consistency regularization을 통해 학습을 보강하고, 데이터 및 모델 변형 기법을 사용하여 일반화 능력을 향상시킵니다.

- **Performance Highlights**: DSTCT 프레임워크는 MICCAI 2023의 PSFH 세분화 그랜드 챌린지의 벤치마크 데이터셋에서 기존의 10개 반지도 세분화 기법들보다 뛰어난 성능을 보였습니다.



### Rethinking Directional Parameterization in Neural Implicit Surface Reconstruction (https://arxiv.org/abs/2409.06923)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이 논문은 하이브리드 방향 매개변수화(hybrid directional parameterization)를 제안하여 기존의 시각(viewing) 및 반사(reflection) 방향 매개변수화의 한계를 극복하고, 복잡한 표면과 스페큘러(specular) 표면을 모두 효과적으로 재구성할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 하이브리드 방향 매개변수화는 샘플된 각 점에 대한 표면까지의 예측 거리(predicted distance)를 활용하여, 표면에 가까울 경우 반사 방향을 사용하고, 그렇지 않을 경우 시각 방향을 사용하는 방식으로 유연하게 전환합니다. 이 접근 방식은 매개변수 없이 기존 신경 임플리시트(surface reconstruction) 방법에 쉽게 적용될 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 하이브리드 방향 매개변수화가 다양한 재료, 기하학 및 외관의 객체들을 재구성하는 데 일관되게 우수한 결과를 제공하며, 기존의 방향 매개변수화 방법보다 뛰어난 성능을 보인다는 것을 입증하였습니다.



### Enhanced Pix2Pix GAN for Visual Defect Removal in UAV-Captured Images (https://arxiv.org/abs/2409.06889)
Comments:
          Prepared for IEEE APUAVD 2024 conference

- **What's New**: 본 논문에서는 UAV(무인 항공기)로 촬영한 이미지에서 시각적 결함을 효과적으로 제거하는 신경망을 제안합니다. 특히, 시각적 결함을 해결하기 위해 강화된 Pix2Pix GAN 아키텍처를 활용한 점이 독창적입니다.

- **Technical Details**: 제안된 방법은 Pix2Pix 아키텍처에 커스터마이징한 수정 사항을 포함하여, 일반적인 문제인 mode collapse를 문제를 해결합니다. 또한, generator와 discriminator 간의 성능 스코어(RPS)를 모니터링하여 두 네트워크의 균형을 유지하고 안정성을 도모하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 항공 사진의 커스텀 데이터셋을 통해 평가되었으며, UAV 이미지의 품질을 현저히 향상시켜 더 깨끗하고 정밀한 시각적 결과를 제공합니다.



### AssistTaxi: A Comprehensive Dataset for Taxiway Analysis and Autonomous Operations (https://arxiv.org/abs/2409.06856)
- **What's New**: AssistTaxi라는 새로운 데이터셋이 발표되었습니다. 이 데이터셋은 멜버른과 그랜트-발카리아의 일반 항공 공항에서 수집된 300,000개 이상의 다양한 이미지를 포함하고 있습니다. 이 데이터셋은 고품질 데이터의 필요성을 충족시키며 자율 택시 운전 시스템의 연구와 개발을 지원하기 위해 설계되었습니다.

- **Technical Details**: AssistTaxi 데이터셋은 Piper Cherokee Warrior 항공기에서 3개의 GoPro Hero 8 및 Hero 10 카메라를 사용하여 수집되었습니다. 수집된 데이터는 KMLB 및 X59 공항에서의 택시 및 이륙/착륙 작업을 포함하며, 데이터의 해상도는 최대 4K에 이릅니다. 이 데이터셋은 이미지 라벨링을 위한 초기 접근법으로 윤곽 기반 탐지 및 선 추출 기법을 제안하고 있습니다.

- **Performance Highlights**: AssistTaxi는 자율 비행기 운영의 효율성과 안전성을 높이기 위한 알고리즘 교육 및 평가를 위한 벤치마크 자료로 활용될 수 있습니다. 개발자와 연구자들은 이 데이터셋을 통해 자율 운전 시스템의 성능을 평가하고 혁신적인 접근 방식을 탐색할 수 있습니다.



### ExIQA: Explainable Image Quality Assessment Using Distortion Attributes (https://arxiv.org/abs/2409.06853)
- **What's New**: 이 논문에서는 Blind Image Quality Assessment (BIQA)의 접근 방식을 왜곡 식별(distortion identification) 관점에서 제안하며, Vision-Language Models (VLMs)인 CLIP을 활용하여 왜곡의 유형과 강도를 예측하는 방법을 밝혔다.

- **Technical Details**: 이 연구에서는 왜곡의 속성(attribute) 학습에 기반한 설명 가능성 높은 접근 방식을 제안합니다. VLMs에 왜곡 이름 대신 왜곡 속성을 프롬프트로 제공하여 이 정보를 집계하여 왜곡 강도를 추정합니다. 또한, 이미지 당 여러 왜곡을 고려하여 방법의 확장성을 높였습니다. 이를 위해 10만 개의 이미지로 구성된 데이터 세트를 생성하여 효율적인 훈련을 지원합니다.

- **Performance Highlights**: 결과적으로 제안된 방법은 PLCC 및 SRCC 메트릭에서 여러 데이터 세트에서 최신 기술(SOTA) 성능을 달성하였으며, 제로샷(zero-shot) 결과는 제안된 접근 방식의 일반화 가능성을 입증합니다.



### LIME-M: Less Is More for Evaluation of MLLMs (https://arxiv.org/abs/2409.06851)
- **What's New**: 최근 발표된 LIME-M benchmark는 기존의 Multimodal Large Language Models (MLLMs) 평가 방식의 한계를 극복하기 위해 개발되었습니다. 기존 평가 기준의 단순한 문제와 인공지능 모델 능력 구분의 부족함을 해결하고자, LIME-M은 두 가지 주요 모듈, 즉 Semi-Automated Screening Process와 Eliminating Answer Leakage로 구성됩니다.

- **Technical Details**: LIME-M benchmark는 9403개의 샘플을 포함하며, 6개 도메인에서 10개의 주요 작업을 수행합니다. 또한, MLLMs의 성능을 평가하기 위해 9개의 모델을 채택하여 각 샘플의 난이도를 분류하고, 쉽게 답할 수 있는 샘플과 답변 유출 문제가 있는 샘플을 제거합니다.

- **Performance Highlights**: LIME-M을 사용한 실험 결과, 기존 데이터의 24%만으로도 MLLMs 간 성능 차이를 더욱 명확하게 구분할 수 있으며, 총 소요 시간도 원본의 23%로 줄일 수 있었습니다. 특히, MLLMs는 시각적 정보에 직접 관련된 질문에 대한 성과가 우수하였으며, 추가적인 상식 지식이나 복잡한 추론을 포함하는 과제에서는 성과가 낮은 경향을 보였습니다.



### Shadow Removal Refinement via Material-Consistent Shadow Edges (https://arxiv.org/abs/2409.06848)
- **What's New**: 이 논문은 그림자 제거(shadow removal)에서 중요한 노력을 기울여 물질 보존(material consistency)을 고려한 그림자 경계(shadow edges)를 식별하고 이를 자가 감독(self-supervision) 신호로 활용하는 방법을 제안합니다.

- **Technical Details**: 이 방법은 SAM(Segment Anything Model)이라는 이미지 분할(base segmentation) 모델을 미세 조정하여 그림자에 관계없이 동일한 분할 마스크를 생성하게 하고, 그림자 마스크와 비교하여 물질 일관성을 유지하는 그림자 경계를 추출합니다. 이 경계를 활용하여 RGB 거리 손실(RGB distance loss)과 RGB 분포 손실(RGB distribution loss)을 도입해 그림자 제거 프로세스를 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 그림자 제거 방법 대비 테스트 세트에서 CDD(Color Distribution Difference) 측정 값이 최소 30% 향상된 성능을 보였습니다. 새로운 평가 메트릭을 적용하여 복잡한 그림자 시나리오에서 그림자 제거 방법의 일반성을 평가할 수 있는 기능도 추가했습니다.



### Face Mask Removal with Region-attentive Face Inpainting (https://arxiv.org/abs/2409.06845)
- **What's New**: COVID-19 팬데믹 동안 마스크 착용이 일반화되면서 얼굴 인식 모델의 성능 저하가 발생하였습니다. 본 연구는 마스크를 단숨에 복구하는 Generative Face Inpainting 방법을 제안합니다. 이 방법은 얼굴의 정체성을 유지하면서 마스크로 가려진 부분을 효과적으로 복원합니다.

- **Technical Details**: 우리의 방법은 Multi-scale Channel-Spatial Attention Module (M-CSAM)을 포함하여 공간 정보 손실을 완화하고 채널 간 상관관계를 학습합니다. Masked-Faces 데이터세트를 CelebA에서 생성하며, 5종의 마스크를 활용합니다. 아울러 마스크 영역에만 집중하는 지도 신호(supervised signal)를 적용하여 회복 성능을 높입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 구조적 유사도 지수(SSIM), 신호 대 잡음 비(PSNR) 및 l1 손실에서 기존의 여러 방법보다 뛰어난 성능을 보여주었습니다. 코드와 데이터세트는 GitHub에 공개됩니다.



### Few-Shot Learning: Expanding ID Cards Presentation Attack Detection to Unknown ID Countries (https://arxiv.org/abs/2409.06842)
- **What's New**: 이번 논문은 원격 검증 시스템에서 신분증의 프레젠테이션 공격(cheat attacks)을 탐지하기 위한 새로운 Few-shot Learning(FSL) 접근 방식을 제안합니다. 본 연구는 스페인과 칠레에서의 Prototypical Networks의 성능을 분석하고 아르헨티나와 코스타리카와 같은 새로운 국가에 대한 일반화 능력을 측정합니다.

- **Technical Details**: 연구는 Prototypical Networks를 기반으로 하며, 저제한 데이터에 대해 효과적으로 학습하여 화면 표시 프레젠테이션 공격을 탐지하는 모델을 개발하였습니다. FSL 기법을 사용하여 5개의 독특한 ID(Identity)와 100장 이하의 이미지로도 경쟁력 있는 성능을 발휘하는 것으로 나타났습니다.

- **Performance Highlights**: 본 연구의 결과는 스페인, 칠레, 아르헨티나, 코스타리카와 같은 다양한 지리적 환경에서 높은 적응성과 성능을 보여줍니다. 특히, EfficientNetV2-B0을 기반으로 하는 훈련 방법론을 구현하여 현실 세계의 시나리오에 빠르게 적응할 수 있음을 입증하였습니다.



### Cross-Modal Self-Supervised Learning with Effective Contrastive Units for LiDAR Point Clouds (https://arxiv.org/abs/2409.06827)
Comments:
          IROS 2024

- **What's New**: 이 논문은 자율 주행 차량에서 LiDAR 포인트 클라우드의 대비 학습(constrastive learning)에 대한 새로운 접근법을 제안합니다. 특히, 단일 모달리티(single modality)와 크로스 모달리티(cross-modality), 멀티 모달리티(multi-modality)를 체계적으로 연구하고, 크로스 모달리티가 가장 효과적이라는 것을 보여줍니다.

- **Technical Details**: 제안된 모형은 instance-aware 및 similarity-balanced contrastive units로, 이는 self-driving point clouds에 맞춰 설계되었습니다. 이 방법은 LiDAR 기반의 3D 객체 탐지(3D object detection) 및 3D 의미론적 분할(3D semantic segmentation)에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: Waymo Open Dataset, nuScenes, SemanticKITTI 및 ONCE와 같은 여러 데이터셋에서 실험 결과, 제안된 접근 방식은 다양한 포인트 클라우드 모형에 대해 현저한 성능 개선을 보여주었으며, 특히 Waymo Open Dataset에서 2.96% L2 mAPH의 성능 향상을 기록했습니다.



### Sam2Rad: A Segmentation Model for Medical Images with Learnable Prompts (https://arxiv.org/abs/2409.06821)
- **What's New**: Sam2Rad는 인체의 초음파(US) 뼈 분할을 위한 제안 학습 방법을 소개하며, 수작업 프롬프트 없이 SAM 및 그 변형을 조정합니다. 이 방법은 이미지 인코더 특징을 바탕으로 프롬프트 임베딩을 예측하는 프롬프트 예측기 네트워크(PPN)를 도입합니다.

- **Technical Details**: PPN은 크로스 어텐션 모듈을 활용하여 bounding box와 mask 프롬프트, 256차원 임베딩을 출력합니다. 이 프레임워크는 선택적으로 수동 프롬프트를 허용하며, 파라미터 효율적인 미세 조정(PEFT)을 사용하여 엔드 투 엔드로 훈련될 수 있습니다.

- **Performance Highlights**: Sam2Rad는 3개의 근골격 초음파 데이터셋에서 테스트되었으며, 수작업 프롬프트 없이도 성능을 개선했습니다. 엉덩이/손목에서 2-7% 증가, 어깨 데이터에서는 최대 33% 증가한 Dice score를 기록했습니다. 또한, 10개의 레이블된 이미지만으로 훈련이 가능하며, 어떤 SAM 아키텍처와도 호환됩니다.



### Object Modeling from Underwater Forward-Scan Sonar Imagery with Sea-Surface Multipath (https://arxiv.org/abs/2409.06815)
Comments:
          Copyright 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문에서는 알려진 자세의 2-D 전방 스캔 소나(FSS) 이미지를 사용하여 3-D 수중 물체 모델링을 위한 최적화 기법을 제안합니다. 특히, 해수면 근처에서 이미징된 물체의 경우 공기-수면 경계로 인한 멀티패스 아티팩트를 해결하는 것이 주요 기여입니다.

- **Technical Details**: 논문에서는 평면적인 공기-수면 경계를 가정하고, 각 뷰 내에서 변질된 물체 영역을 모델링하고 로컬라이즈하여 불필요한 왜곡을 피합니다. 3-D 표면 메시 모델의 삼각형 패치의 정점 위치를 조정하여 데이터와 합성된 뷰 간의 불일치를 최소화하는 형태로 최적화가 구현됩니다.

- **Performance Highlights**: 여러 실험을 통해 비평평한 공기-수면 경계의 영향을 탐구하며, 약 6회의 반복(iteration) 후 정제된 3-D 모델의 생성이 확인되었습니다.



### DetailCLIP: Detail-Oriented CLIP for Fine-Grained Tasks (https://arxiv.org/abs/2409.06809)
- **What's New**: 이 논문에서는 세부 지향적 태스크에 대응하기 위해 새로운 프레임워크인 DetailCLIP을 소개합니다. 이 모델은 기존 CLIP 모델이 세부를 포착하는 데 있어 가지는 한계를 극복할 수 있도록 설계되었습니다.

- **Technical Details**: DetailCLIP은 패치 수준 비교(self-distillation)와 픽셀 수준 재구성 손실(pixel-level reconstruction losses)을 활용하며, 주의 기반 토큰 제거 메커니즘(attention-based token removal mechanism)을 통해 의미적으로 연관된 토큰을 선택적으로 유지합니다. 이로 인해 세부적인 시각적 특징을 학습하고, 텍스트 정보 처리와 패치 비교를 통해 더 높은 수준의 의미 이해를 확장합니다.

- **Performance Highlights**: 실험 결과, DetailCLIP은 기존 CLIP 기반 모델들과 전통적인 자기 지도 학습(SSL) 모델들보다 세분화 정확도에서 우수한 성능을 보여줍니다. 이 모델은 다양한 데이터셋에서 타의 추종을 불허하는 일반화 능력을 발휘하여 세부 보존과 모델의 강건성을 크게 향상시킵니다.



### Human Motion Synthesis_ A Diffusion Approach for Motion Stitching and In-Betweening (https://arxiv.org/abs/2409.06791)
Comments:
          12 pages, 5 figures, and 11 equations

- **What's New**: 이번 연구에서는 인간의 움직임 생성을 위한 새로운 접근 방식을 제안합니다. 기존 방법들은 수동적인 노력을 필요로 하거나 긴 시퀀스를 처리하는 데 한계가 있었던 반면, 본 연구에서는 diffusion 모델을 활용하여 자연스러운 인간의 움직임을 생성하는 방안을 제시하였습니다.

- **Technical Details**: 제안된 방법은 motion frames의 위치와 현재 diffusion 단계를 인코딩하여 encoder transformer에 전달함으로써 motion frames 간의 관계를 캡처합니다. 이후 초기 Gaussian noise와 encoder의 출력을 이용하여 clean motion을 예측합니다. 이 외에도 수치적인 메트릭(Frechet Inception Distance (FID), Diversity, Multimodality)을 사용하여 제안된 방법의 성능을 평가하였습니다.

- **Performance Highlights**: 75개의 프레임으로 구성된 매끄럽고 자연스러운 움직임 시퀀스를 생성하며, 자발적인 평균 프레임 속도는 15fps로 5초의 총 지속시간을 가지게 됩니다. 본 방법은 짧은 기간 및 긴 주기의 motion generation 작업에서 효과적인 성능을 보여주었습니다.



### gsplat: An Open-Source Library for Gaussian Splatting (https://arxiv.org/abs/2409.06765)
Comments:
          17 pages, 2 figures, JMLR MLOSS

- **What's New**: gsplat은 Gaussian Splatting 방법론을 훈련시키고 개발하기 위해 설계된 오픈소스 라이브러리입니다. 이는 PyTorch 라이브러리와 호환되는 Python 바인딩의 프론트엔드와 최적화된 CUDA 커널로 구성된 백엔드를 특징으로 합니다. gsplat은 훈련 시간 10% 단축 및 메모리 사용 4배 감소를 포함한 최적화 개선 기능을 제공합니다.

- **Technical Details**: gsplat는 독립형 라이브러리로 효율성과 모듈성을 염두에 두고 개발되었습니다. 이는 PyPI에서 설치 가능하며, CUDA 최적화된 커널을 통해 여러 연산을 프로그래밍합니다. 또한, 사용자는 최신 연구 아이디어를 지원하기 위해 원주율(PyTorch) 구현을 통해 쉽게 통합하고 수정할 수 있습니다. gsplat은 Adaptive Density Control (ADC), Absgrad 방법, Markov Chain Monte Carlo (MCMC) 방법을 포함한 최신 밀도화 전략을 지원합니다.

- **Performance Highlights**: gsplat은 MipNeRF360 데이터셋에서 Kerbl et al.의 원래 구현과 비교하여 동일한 렌더링 성능을 유지하면서 메모리 사용량은 적고 훈련 시간은 크게 단축되었습니다. 실험 결과, 평균적으로 훈련 성능과 효율성이 크게 향상되었음을 보여줍니다.



### Modeling Image Tone Dichotomy with the Power Function (https://arxiv.org/abs/2409.06764)
Comments:
          49 pages, 11 figures and 36 references

- **What's New**: 본 논문에서는 이미지 조명 모델링에서의 이분법(dichotomy) 개념을 제시하고 새로운 수학적 모델을 제안합니다. 이 모델은 조명 이분법을 추상화할 수 있으며, 여러 수학적 특성을 검토하여 기존 모델의 한계를 식별합니다.

- **Technical Details**: 파워 함수(power function)의 수학적 속성을 활용하여 이미지 분석과 처리의 새로운 경로를 엽니다. 감마 보정(gamma correction) 및 비선형 변환의 개념을 도입하고, 감마 압축(gamma compression)과 감마 확장(gamma expansion)에 따른 이미지의 밝기 및 명도 개선 방법을 설명합니다.

- **Performance Highlights**: 본 모델은 기존의 최신 이미지 향상 방법과 비교하여 일관적으로 우수한 성능을 나타냅니다. 저조도(low-light) 이미지의 정보 추출을 위한 다양한 사례를 들어 이미지를 개선하는 데 효과적임을 보여줍니다.



### Feedback-based Modal Mutual Search for Attacking Vision-Language Pre-training Models (https://arxiv.org/abs/2409.06726)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 Feedback-based Modal Mutual Search (FMMS)라는 새로운 공격 패러다임을 제안합니다. 이는 적대적 예제를 생성하기 위해 대상 모델의 피드백을 이용하는 최초의 방법으로, 멀티모달 환경에서의 적대 경계를 탐색하는 데 중점을 두고 있습니다.

- **Technical Details**: FMMS는 Modal Mutual Loss (MML)라는 새로운 손실 함수를 도입하여, 일치하는 이미지-텍스트 쌍 간의 거리를 멀게 하고 불일치 쌍 간의 거리를 가깝게 유지합니다. 이를 통해 적대적 예제를 점진적으로 개선하며, 두 가지 탐색 전략 (Full 및 Top-N)을 활용하여 더욱 эффектив한 적대적 예제를 생성합니다.

- **Performance Highlights**: Flickr30K 및 MSCOCO 데이터셋에서의 실험 결과, FMMS는 기존의 최첨단 기법들보다 적대적 예제를 생성하는 데 있어 상당히 우수한 성능을 보였습니다. 특히, 다양한 VLP 모델 간의 효과적인 적대적 전이 능력을 입증했습니다.



### Quantized neural network for complex hologram generation (https://arxiv.org/abs/2409.06711)
Comments:
          10 pages, 2 figures

- **What's New**: 이번 연구는 컴퓨터 생성 홀로그램(Computer-generated holography, CGH)을 위한 경량화 모델을 개발하여, 신경망 양자화(quantization)를 도입한 결과를 보여줍니다. 특히, 32-bit 부동소수점 정밀도(FP32)에서 8-bit 정수 정밀도(INT8)로의 양자화를 통해 모델 크기를 약 70% 줄이고 속도를 4배 증가시켰습니다.

- **Technical Details**: CGH는 빛의 파동을 제어하여 2D 홀로그램을 계산하는 기술로, 이번 연구에서는 RGB-D 이미지를 기반으로 복잡한 홀로그램을 생성하기 위해 텐서 홀로그램(tensor holography) 모델을 활용했습니다. 이를 INT8 정밀도로 양자화하여 기존 FP32 모델과 유사한 품질을 유지하면서 메모리 사용량과 전력 소모를 낮췄습니다.

- **Performance Highlights**: INT8 모델은 SoM(즉시 모듈) 플랫폼인 AMD Kria K26에 구현되어 AR 헤드셋 및 자동차 헤드업 디스플레이와 같은 내장 시스템에서 높은 전력 효율성을 입증하였습니다. 또한, 이 모델은 GPU보다 약 4배 더 높은 전력 효율성을 달성하였습니다.



### McGrids: Monte Carlo-Driven Adaptive Grids for Iso-Surface Extraction (https://arxiv.org/abs/2409.06710)
- **What's New**: 이 논문은 복잡한 기하학적 형태의 iso-surface 추출을 효율적으로 개선하기 위해 McGrids라는 새로운 접근 방식을 제안합니다. 기존의 단순한 균일 격자 대신에 적응형 격자를 구성하는 방법을 소개하며, 이는 메모리 사용량과 계산 비용을 줄입니다.

- **Technical Details**: McGrids는 iso-surface 추출을 위한 적응형 격자를 구성하는 문제를 확률 샘플링 문제로 공식화하고, 이를 Monte Carlo 프로세스를 통해 해결합니다. 이러한 접근 방식은 객체 표면에 가까운 영역에서는 밀도가 높고, 빈 지역이나 평평한 영역에서는 밀도가 낮은 격자를 생성합니다.

- **Performance Highlights**: 실험 결과, McGrids는 implicit field 쿼리 수를 크게 줄여 메모리 사용량을 줄이면서도 높은 품질의 메시를 생성합니다. 이는 여러 응용 프로그램에서 효율적인 iso-surface 추출을 가능하게 합니다.



### Gating Syn-to-Real Knowledge for Pedestrian Crossing Prediction in Safe Driving (https://arxiv.org/abs/2409.06707)
Comments:
          under review by TITS

- **What's New**: 본 연구는 Pedestrian Crossing Prediction (PCP) 작업을 위한 Gated Syn-to-Real Knowledge transfer 접근법(Gated-S2R-PCP)을 제안합니다. 이 방법은 다양한 도메인 지식에 적합한 도메인 적응 방식과 특정 상황에 맞는 지식 전이를 목표로 하고 있습니다.

- **Technical Details**: Gated-S2R-PCP는 세 가지 도메인 적응 방법(Style Transfer, Distribution Approximation, Knowledge Distillation)을 포함하는 프레임워크를 설계하여 보행자의 위치, 시각적 정보, 깊이 정보를 전이합니다. 이를 통해 Learnable Gated Unit (LGU)를 사용하여 적합한 교차 도메인 지식을 융합합니다. 3181개의 시퀀스와 489,740개의 프레임으로 구성된 S2R-PCP-3181이라는 새로운 합성 데이터셋을 구축했습니다.

- **Performance Highlights**: Gated-S2R-PCP는 JAAD 및 PIE와 같은 실제 복잡한 데이터셋에서 검증되었으며, 기존의 최신 방법들에 비해 우수한 PCP 성능을 보였습니다. 또한 DADA-2000 데이터셋의 모든 보행자 크로싱 시퀀스에 대해 성능을 평가하여 근접 충돌 상황에서의 PCP 성능을 검증했습니다.



### HSR-KAN: Efficient Hyperspectral Image Super-Resolution via Kolmogorov-Arnold Networks (https://arxiv.org/abs/2409.06705)
- **What's New**: 이번 논문에서는 Kolmogorov-Arnold Networks (KANs)를 기반으로 한 효율적인 고급 스펙트럼 이미지(super-resolution, HSI-SR) 모델을 제안합니다. 이 모델은 저해상도 하이퍼스펙트럼 이미지(LR-HSI)와 고해상도 다스펙트럼 이미지(HR-MSI)를 결합하여 고해상도 하이퍼스펙트럼 이미지(HR-HSI)를 생성하는 효율적인 방법입니다.

- **Technical Details**: KAN-Fusion이라는 이름의 융합 모듈을 설계하여 HR-MSI의 공간 정보를 효과적으로 통합합니다. 또한, 채널 주의 메커니즘(channel attention mechanism)에 영감을 받아 KAN Channel Attention Block (KAN-CAB)이라는 스펙트럼 채널 주의 모듈을 설계하였습니다. KAN-CAB는 깊은 네트워크의 미세 조정 능력을 향상시키고, 스펙트럼 시퀀스 및 공간 텍스처의 세부 사항을 정확하게 시뮬레이션할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 HSR-KAN 모델은 현재의 최첨단(Human state-of-the-art, SOTA) HSI-SR 방법들에 비해 정성적 및 정량적 평가 모두에서 최고의 성능을 달성하였습니다.



### VMAS: Video-to-Music Generation via Semantic Alignment in Web Music Videos (https://arxiv.org/abs/2409.07450)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 비디오 입력으로부터 배경 음악을 생성하는 새로운 프레임워크를 제안합니다. 기존 방식들이 제한된 양과 다양성을 가진 상징적 음악 주석(symbolic musical annotations)에 의존하는 것과는 달리, 우리의 방법은 배경 음악이 포함된 대규모 웹 비디오를 활용하여 사실적이고 다양한 음악을 생성하도록 학습합니다.

- **Technical Details**: 이 모델은 새로운 의미적 비디오-음악 정렬 방법을 사용한 생성적 비디오-음악 Transformer로 구성되어 있습니다. 음악 생성은 자동 회귀(autoregressive) 및 대조 학습(contrastive learning) 목적 함수를 공동으로 사용하여 수행됩니다. 또한, 새롭게 도입된 비디오-비트 정렬(video-beat alignment) 기법은 생성된 음악 비트와 비디오의 저수준 동작(low-level motions)을 일치시킵니다.

- **Performance Highlights**: DISCO-MV 데이터셋을 사용하여 훈련한 결과, 우리의 방법은 MUSICaps 및 DISCO-MV 데이터셋에서 여러 음악 생성 평가 지표에 따라 기존 접근법보다 뛰어난 성능을 보였습니다. 특히, 인간 평가에서 우리의 생성 음악이 음악 품질과 음악-비디오 정렬 측면에서 이전 방법들보다 선호된다는 결과를 얻었습니다.



### Adaptive Adapter Routing for Long-Tailed Class-Incremental Learning (https://arxiv.org/abs/2409.07446)
Comments:
          Accepted to Machine Learning Journal. Code is available at: this https URL

- **What's New**: 동적 데이터 스트리밍의 세계에서, 이번 논문은 Long-Tailed Class-Incremental Learning (LTCIL)의 문제를 해결하기 위해 AdaPtive Adapter RouTing (APART)를 제안합니다. 이 방법은 기존의 exemplar 기반 접근법 대신, 사전 훈련된 모델(Pre-trained models)을 활용하여 기억 소실(catasrophic forgetting)을 방지합니다.

- **Technical Details**: APART는 사전 훈련된 모델의 대부분의 매개변수를 고정시키고, 각 레이어에 트레인 가능한 어댑터(adapters)를 추가하여 깊이 있는 적응(deep adaptation)을 가능하게 합니다. 또한, minority 클래스의 학습을 집중적으로 지원하는 보조 어댑터 풀(auxiliary adapter pool)을 도입하여, 모든 클래스에 대한 포괄적인 표현(comprehensive representation)을 생성합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해, APART의 효과를 주의 깊게 검증하였으며, 모델이 LTCIL의 도전을 성공적으로 처리하여 성능이 향상됨을 입증하였습니다. 특히, minority 클래스에 대한 데이터 불균형을 줄이고, 기억 소실 문제를 해결하는 데 있어 매우 유의미한 결과를 보였습니다.



### Controllable retinal image synthesis using conditional StyleGAN and latent space manipulation for improved diagnosis and grading of diabetic retinopathy (https://arxiv.org/abs/2409.07422)
Comments:
          30 pages, 17 figures

- **What's New**: 이 연구는 당뇨병성 망막병증(Diabetic Retinopathy, DR) 치료를 위한 고품질의 다양한 fundus 이미지를 생성하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 조건부 StyleGAN을 활용하여 DR의 중증도와 시각적 특징(시신경 유두(optic disc), 혈관 구조(vessel structure), 병변 영역(lesion areas))을 조절 가능한 방식으로 생성합니다.

- **Performance Highlights**: ResNet50 모델은 DR 탐지에서 98.09% 정확도, 99.44% 특이도, 99.45% 정밀도, 98.09% F1 점수를 달성했습니다. 또한, 조건부 StyleGAN으로 생성된 합성 이미지를 활용하여 DR 등급 판별을 위한 ResNet50 모델은 83.33% 정확도와 함께 87.64%의 quadratic kappa 점수를 기록했습니다.



### Efficient One-Step Diffusion Refinement for Snapshot Compressive Imaging (https://arxiv.org/abs/2409.07417)
- **What's New**: 이번 논문에서는 Coded Aperture Snapshot Spectral Imaging (CASSI) 기술을 위한 새로운 Diffusion Probabilistic Model을 소개하며, 이를 Self-supervised Adaptation Framework 내에서 적용하였습니다. 이 접근법은 기존의 단점인 고주파 세부사항 복원 문제를 해결하고, 다양한 end-to-end 및 unfolding 기술에 대한 적응성과 단순성을 강조합니다.

- **Technical Details**: 본 연구는 2D 측정에서 시작하여 고주파 잔차를 생성하는 one-step diffusion 모델을 사용하는 새로운 SCI 복원 방법을 제안합니다. 이 모델은 pretrained SCI reconstruction network를 이용하여 초기 예측치를 생성하고, 이후 잔차를 정제합니다. Self-supervised learning Paradigm을 도입하여 데이터를 효과적으로 활용하고, Equivariant Imaging (EI) 프레임워크를 기반으로 하여 2D 이미지 만으로도 학습을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존의 방법들과 비교하여 quantitative metrics와 visual comparison에서 모두 우수한 성능을 보였습니다. 고해상도 세부사항을 복원하는 데 보다 나은 성능을 보여주며, 다양한 데이터셋에서 적응성과 효율성을 입증하였습니다.



### What to align in multimodal contrastive learning? (https://arxiv.org/abs/2409.07402)
Comments:
          22 pages

- **What's New**: 이 논문에서는 CoMM(Contrastive MultiModal)이라는 새로운 방법론을 제안하여 여러 종류의 콘텐츠와 형식 간의 상호작용을 보다 효과적으로 학습할 수 있도록 한다. CoMM은 단일 다중 모드 공간에서 모드 간의 소통을 가능하게 하여 이전의 방법과 대비된다.

- **Technical Details**: CoMM은 서로 다른 증강된 다중 모드 피처 간의 상호 정보를 최대화하는 방식으로 다중 모드 표현을 정렬한다. 이 접근 방식은 공유 정보, 협력적인 정보, 고유 정보를 포함하여 모드 간의 상호작용을 측정할 수 있는 강력한 이론적 기초를 가지고 있다.

- **Performance Highlights**: CoMM은 실제 데이터셋에서 다양한 분야에서 최고 수준의 성능을 보여주었으며, 7개의 다중모드 작업에서 최첨단 결과를 달성하였다. 이로 인해 CoMM은 다양한 모드 수와 데이터 유형을 다룰 수 있는 유연한 프레임워크임을 입증하였다.



### FIRAL: An Active Learning Algorithm for Multinomial Logistic Regression (https://arxiv.org/abs/2409.07379)
Comments:
          Accepted at the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)

- **What's New**: 이번 연구에서는 다중 클래스 분류(multi-class classification)를 위한 풀 기반(active learning) 능동 학습 알고리즘을 다룹니다. 이를 위해 multinomial logistic regression(다항 로지스틱 회귀)을 이용하여 이론적 분석을 수행하였습니다.

- **Technical Details**: 연구에서는 유한 샘플 분석(finite sample analysis)을 통해 Fisher Information Ratio (FIR)가 초과 위험(excess risk)의 하한 및 상한을 제공함을 증명하였습니다. 이러한 이론적 분석을 바탕으로 FIR을 최소화하는 능동 학습 알고리즘을 제안합니다.

- **Performance Highlights**: 제안한 FIRAL 알고리즘은 MNIST, CIFAR-10 및 50-class ImageNet에서 실험 결과 다른 5개의 방법보다 일관되게 가장 작은 분류 오류(classification error)를 기록하였습니다.



### Quantifying Knee Cartilage Shape and Lesion: From Image to Metrics (https://arxiv.org/abs/2409.07361)
Comments:
          The paper will be in the conference proceedings of AMAI 2024. See the conference website: this https URL

- **What's New**: 이번 연구에서는 무릎 연골의 이미징 특성을 평가하는 완전 자동화된 딥러닝 기반의 의료 영상 분석 응용 프로그램인 CartiMorph Toolbox (CMT)를 개발했습니다. CMT는 이미지 템플릿 학습(template learning) 및 등록(registration)을 위한 2단계 네트워크인 CMT-reg를 제안하고 OAI-ZIB 데이터셋을 사용해 모델 성능을 평가했습니다.

- **Technical Details**: CMT는 조직 분할(segmentation) 및 등록(registration)을 위한 딥러닝 모델(DL models)로 구성되어 있으며, 템플릿 학습과 등록을 통합한 단일 CNN을 사용합니다. 이 도구는 이미지의 정규화(normalization), 재조정(re-orientation), 리샘플링(resampling) 절차를 포함하여 DL 모델의 훈련 환경을 설정하고 데이터 시각화(data visualization) 기능을 지원합니다.

- **Performance Highlights**: CMT-reg는 최신의 state-of-the-art 모델들과 비교해 경쟁력 있는 결과를 나타냈습니다. 또한 CartiMorph 프레임워크를 개선하여 자동화된 연골 형태 및 손상 분석을 위한 포괄적이고 사용자 친화적인 솔루션을 제공하며, 소프트웨어와 모델은 공개적으로 사용할 수 있습니다.



### Federated Impression for Learning with Distributed Heterogeneous Data (https://arxiv.org/abs/2409.07351)
- **What's New**: 이 연구에서는 데이터 이질성(data heterogeneity)이 유발하는 catastrophic forgetting 문제를 해결하기 위한 새로운 접근 방식인 FedImpres를 제안합니다. 이 방법은 지역 데이터를 보완하기 위해 합성 데이터를 생성하여 모델의 일반화 성능을 향상시킵니다.

- **Technical Details**: FedImpres는 각 통신 라운드에서 글로벌 모델을 증류(dilution)하여 합성 데이터를 생성합니다. 이 때, 합성 데이터는 전역 정보(global information)를 나타내며, 이는 클라이언트 측에서 지역 데이터와 함께 사용되어 지역 훈련의 일반화(generalization)를 향상시킵니다. 각 클라이언트는 자신의 지역 데이터와 합성 데이터를 사용하여 훈련합니다.

- **Performance Highlights**: 제안된 FedImpres 방법은 BloodMNIST와 Retina 데이터셋에서 최첨단 성능을 달성하였으며, 분류 정확도가 최적화되어 20%까지 개선되었습니다.



### BLS-GAN: A Deep Layer Separation Framework for Eliminating Bone Overlap in Conventional Radiographs (https://arxiv.org/abs/2409.07304)
- **What's New**: 이 논문은 기존 방사선 사진에서 겹쳐진 뼈층을 분리하는 새로운 시나리오를 제시하고 있으며, 이를 통해 각 뼈층의 특성을 개별적으로 평가할 수 있는 기초를 제공합니다.

- **Technical Details**: 제안된 Bone Layer Separation GAN (BLS-GAN) 프레임워크는 뼈 특성과 질감이 잘 유지된 고품질 뼈층 이미지를 생성합니다. 이 프레임워크는 전통적인 방사선 사진 원리 기반의 재구성기를 도입하여 겹쳐진 부위에서 발생하는 반복 계산 및 훈련 불안정성 문제를 완화합니다.

- **Performance Highlights**: 생성된 이미지는 시각적 터링 테스트를 통과하였으며, 다운스트림 작업에서의 성능 개선을 보여줍니다. 이 결과는 MSK 질병 진단, 모니터링 및 예측을 위한 보다 포괄적인 분석 연구를 가능하게 할 것으로 기대됩니다.



### A Unified Contrastive Loss for Self-Training (https://arxiv.org/abs/2409.07292)
- **What's New**: 본 논문은 반지도 학습(Semi-Supervised Learning)에서 자기 학습(self-training) 방법을 개선하기 위한 새로운 프레임워크를 제안합니다. 특히, 크로스 엔트로피 손실 함수(Cross-Entropy Loss)를 대체할 수 있는 대조 손실 함수(Contrastive Loss)를 사용하여 성능을 향상시키는 방법에 대해 논의합니다.

- **Technical Details**: 제안된 Semi-Supervised Contrastive (SSC) 프레임워크는 클래스 프로토타입(class prototypes)을 통합하여 라벨이 있는 데이터와 낮은 신뢰도의 의사 라벨(pseudo-label)의 예를 동시에 처리합니다. 이를 통해 손실 함수 ℒSSC	ext{L}_{	ext{SSC}}를 사용하여 기존의 FixMatch와 같은 자기 학습 기법을 통합할 수 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 세 가지 다른 데이터셋에서 성능 향상을 보여주며, 수렴 속도(convergence speed), 전이 능력(transfer ability), 및 하이퍼파라미터 안정성(hyperparameter stability)에서 더욱 개선된 결과를 나타냅니다.



### Exploring User-level Gradient Inversion with a Diffusion Prior (https://arxiv.org/abs/2409.07291)
Comments:
          Presented at the International Workshop on Federated Learning in the Age of Foundation Models in conjunction with NeurIPS 2023

- **What's New**: 이 연구는 분산 학습에서 사용자 수준의 gradient inversion을 탐구하며, 기존의 샘플별 공격과는 다른 새로운 공격 표면을 제안합니다. 특히, 저자들은 대규모 배치 환경에서 회복을 증진시키기 위해 denoising diffusion model을 사용하는 기법을 개발했습니다.

- **Technical Details**: 이 방법은 기존의 gradient inversion 기법의 한계를 극복하고, 랜덤으로 샘플을 복원하는 대신, 민감한 정보를 포착하는 대표 이미지를 생성하는 데 중점을 둡니다. 이를 통해 계산 오버헤드를 크게 줄이고, 수렴 안정성을 향상시킵니다. 이 연구는 diffusion model을 적용하여 gradient inversion의 정확성을 개선하는 첫 번째 연구입니다.

- **Performance Highlights**: 실험 결과, 이 방법은 CelebA 얼굴 이미지 데이터셋을 사용하여 성별, 인종, 나이 및 얼굴 정체성 등과 같은 개인 정보를 효과적으로 복원하는 능력을 보여주었습니다. 특히, 기존의 적대적 가정에 의존하지 않고 고유한 개인 속성을 회복할 수 있는 가능성을 시연했습니다.



### Tuning-Free Online Robust Principal Component Analysis through Implicit Regularization (https://arxiv.org/abs/2409.07275)
- **What's New**: 이번 논문에서는 데이터 세트에 의존하지 않는 implicit regularization (IR) 기법을 활용하여 전통적인 Online Robust Principal Component Analysis (OR-PCA) 기법의 튜닝 파라미터 의존성을 제거하는 방법을 제시합니다.

- **Technical Details**: 우리의 방법은 sparsity와 low-rank 구조를 자연스럽게 유도하는 세 가지 버전의 수정된 gradient descent를 통합하여 OR-PCA의 효율성을 향상시킵니다. 또한, explicit regularization (명시적 정규화) 파라미터에 대한 의존성을 제거한 새로운 파라미터화 전략을 활용합니다.

- **Performance Highlights**: 제안한 알고리즘은 시뮬레이션 데이터와 실제 데이터 모두에 대해 튜닝된 OR-PCA와 비교했을 때 유사하거나 더 나은 성능을 보이며, 대용량 데이터 세트에 대해 더 확장 가능한 솔루션을 제시합니다.



### TopoMap++: A faster and more space efficient technique to compute projections with topological guarantees (https://arxiv.org/abs/2409.07257)
Comments:
          This is the author's version of the article that has been accepted for publication in IEEE Transactions on Visualization and Computer Graphics (TVCG)

- **What's New**: 이 논문에서는 TopoMap 알고리즘의 개선된 버전인 TopoMap++를 제안합니다. 이는 시각적 공간 사용을 효율적으로 개선하고, 빠른 구현과 토폴로지 계층을 활용한 새로운 TreeMap 기반 표현 방식을 포함합니다.

- **Technical Details**: TopoMap++는 고차원 데이터의 0-차원 지속 다이어그램을 보존하면서도, 고차원 데이터의 계층 구조를 시각화하는 TreeMap을 사용하여 데이터 탐색을 용이하게 합니다. 이 또한 Euclidean minimum spanning tree 계산의 시간을 대폭 단축시키는 근사 기법을 도입하였습니다.

- **Performance Highlights**: TopoMap++는 기존 TopoMap보다 대규모 데이터셋에 대한 공간 효율성을 높이고, 최소 두 배 이상의 속도 향상으로 높은 차원의 데이터 구조를 효과적으로 시각화할 수 있는 능력을 보여줍니다.



### Alignment of Diffusion Models: Fundamentals, Challenges, and Futur (https://arxiv.org/abs/2409.07253)
Comments:
          35 pages, 5 figures, 3 tables

- **What's New**: 이번 연구는 확산 모델(Diffusion models)의 정렬(alignment) 성능 개선에 초점을 맞추어, 기존 문헌에서는 다루어지지 않았던 새로운 관점을 제공하고 있습니다.

- **Technical Details**: 이 논문은 확산 모델의 정렬 기술 및 원리를 다루며, 사용자 기대와 선호에 맞추어 적절한 출력을 생성하기 위한 방법론을 논의합니다. 또한, 선호 기준(preference benchmarks)과 평가(evaluation) 방법에 대해 체계적으로 검토하고 있습니다.

- **Performance Highlights**: 이번 연구는 확산 모델의 정렬 분야에서 최초의 포괄적인 리뷰(research paper)로, 연구자와 엔지니어들이 해당 기술을 이해하고 실천할 수 있도록 돕는 중요한 역할을 합니다.



### 3DGCQA: A Quality Assessment Database for 3D AI-Generated Contents (https://arxiv.org/abs/2409.07236)
- **What's New**: 본 논문은 3D 생성된 콘텐츠 (3DGC)의 품질 평가를 위한 새로운 데이터셋인 3DGCQA를 소개합니다. 이 데이터셋은 7가지 텍스트-투-3D (Text-to-3D) 생성 방법을 사용하여 구축되었으며, 50개의 고정 프롬프트를 이용해 313개의 텍스처가 있는 메시를 생성하여 다양한 왜곡 유형을 시각적으로 드러냅니다.

- **Technical Details**: 3DGCQA 데이터셋은 총 50개의 프롬프트로부터 생성된 3DGC를 포함하고 있으며, 각 프롬프트에서 생성된 3DGC의 품질은 평가자에 의해 주관적으로 평가되었습니다. 또한, 기존의 객관적인 품질 평가 알고리즘도 테스트되어 그 성능 한계가 드러났습니다. 이 연구는 3D 콘텐츠 생성 기술의 발전을 위한 중요한 통찰을 제공합니다.

- **Performance Highlights**: 주관적인 평가 결과, 생성 방법들 간에 품질의 차이가 크게 나타났습니다. 이는 3DGC 품질 평가의 중요성을 다시 한 번 강조하며, 기존 객관적 평가 방법의 한계를 드러냅니다. 3DGCQA 데이터셋은 향후 연구 및 개발에 있어 귀중한 자원으로 활용될 것입니다.



### Behavioral Cloning Models Reality Check for Autonomous Driving (https://arxiv.org/abs/2409.07218)
- **What's New**: 이 논문은 최근의 자율주행 차량 인식 시스템의 실제 적용 가능성을 검증하는 새로운 연구 결과를 제시합니다. 기존의 연구들은 주로 시뮬레이션 환경에서 평가되었으나, 본 연구는 Behavior Cloning (BC)을 이용한 최신 인식 시스템의 실제 환경에서의 성과를 중점적으로 다룹니다.

- **Technical Details**: 제안된 방법은 Autoencoder 기반 Behavioral Cloning (AutoBC), Vision Transformers (ViT) 및 Spatial Attention 메커니즘 등을 포함하여 다양한 아키텍처를 활용하여 실제 주행 조건에서 인식 시스템의 성능을 평가합니다. 이때, 수집된 데이터셋은 축소된 연구 차량을 사용하여 여러 트랙 환경에서 테스트되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들이 스티어링 조향 명령을 실시간으로 예측할 때 낮은 오류 범위를 기록하였으며, 이는 실제 적용 가능성에 대한 희망적인 결과로 평가됩니다.



### AC-IND: Sparse CT reconstruction based on attenuation coefficient estimation and implicit neural distribution (https://arxiv.org/abs/2409.07171)
Comments:
          12 pages

- **What's New**: 이 논문은 CT 재구성을 위한 새로운 기법인 AC-IND를 제안합니다. 이 방법은 물체의 재료 범주 수라는 강력한 선행 정보를 활용하여 임펄스 신경망의 성능을 개선합니다.

- **Technical Details**: AC-IND는 전통적인 임펄스 신경망(INR)을 스칼라 매핑 대신 확률 분포 매핑으로 변환하며, 모듈화된 분포를 생성하여 학습 중 분포가 진정한 형태에 수렴하도록 합니다. 이 과정에서 평균 감쇠 계수(AC)를 결정하기 위해 빠른 재구성 방법과 다중 오츠 임계값(Multi-Otsu Thresholding) 기법을 사용합니다.

- **Performance Highlights**: AC-IND는 20, 40, 60개의 스파스 뷰에서 전통적인 INR보다 우수한 재구성精度를 달성했습니다. 또한, 이 방법은 자동으로 의미론적 분할(Semantic Segmentation) 지도를 생성하여 CT 재구성과 비지도 의미론적 분할 작업을 결합할 수 있는 기회를 만듭니다.



### Mamba Policy: Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models (https://arxiv.org/abs/2409.07163)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 Mamba Policy라는 경량화된 정책 네트워크를 제안합니다. 이는 기존 정책 네트워크와 비교하여 파라미터 수를 80% 이상 줄이면서도 우수한 성능을 유지합니다.

- **Technical Details**: Mamba Policy는 XMamba Block을 통해 입력 정보를 조건적 피처와 통합합니다. 이는 Mamba 및 Attention 메커니즘을 결합하여 깊은 피처 추출을 가능하게 합니다. 또한, 적은 계산 자원으로 Adroit, DexArt 및 MetaWorld 데이터셋에서 우수한 성능을 보여줍니다.

- **Performance Highlights**: Mamba Policy는 기존의 3D Diffusion Policy (DP3)보다 성능이 개선되며, GPU 메모리 사용을 크게 줄입니다. 또한, 다양한 조작 데이터셋에서 성공률이 5% 증가했으며, 장기 시나리오에서도 안정성을 보여주는 것으로 확인되었습니다.



### Deep Learning Techniques for Hand Vein Biometrics: A Comprehensive Review (https://arxiv.org/abs/2409.07128)
- **What's New**: 본 논문은 손 정맥 생체 인식(hand vein biometrics) 분야의 최신 깊이 학습(deep learning) 기술 발전을 다루고 있습니다. 이 연구는 손가락 정맥(finger vein), 손바닥 정맥(palm vein), 손등 정맥(dorsal hand vein) 인식의 모든 필수 기본 사항을 포괄하며, 이전 연구에서는 다루지 않았던 최신 기술 및 도전 과제를 논의합니다.

- **Technical Details**: 이 리뷰는 기존의 손 정맥 기반 생체 인식 시스템에서 새롭게 적용된 깊이 학습 기술들을 분석합니다. 기존의 전통적인 방법들과 비교하여, 깊이 학습 모델은 복잡한 정맥 구조의 패턴을 자동으로 인식하며, 다양한 조명 및 손 위치 변화에 대한 저항력을 가지고 있습니다. 이 연구는 데이터 증강(data augmentation) 기술과 효과적인 전이 학습(transfer learning) 방법을 포함하여, 손 정맥 인식의 성공적인 성과에 관한 통찰을 제공합니다.

- **Performance Highlights**: 정맥 기반 생체 인식 기술의 효과적인 전이 학습 기법과 데이터 증강 기술이 모든 검토된 깊이 학습 접근 방식에서 분석되었습니다. 최근 2017년부터 2024년까지의 문헌을 포괄하며, 이는 손 정맥 생체 인식의 신뢰성과 보안을 강화하는 중요한 기초가 됩니다. 수치적으로, 심층 학습을 통한 손 정맥 인식 정확도는 기존 방법들보다 뛰어난 성능을 보이고 있으며, 미래 연구 방향에 대한 통찰을 제공합니다.



### Attention Down-Sampling Transformer, Relative Ranking and Self-Consistency for Blind Image Quality Assessmen (https://arxiv.org/abs/2409.07115)
Comments:
          Accepted in International Conference on Image Processing (ICIP)

- **What's New**: 이번 연구에서는 NR-IQA(Non-Reference Image Quality Assessment) 모델의 성능 향상을 위해 새로운 Transformer 아키텍처와 CNN(convolutional neural networks)을 적절히 활용하여 이미지 품질에 대한 평가 방법을 제안합니다.

- **Technical Details**: 제안된 ADTRS(Attention Down-Sampling Transformer with Relative ranking and self-consistency) 모델은 입력 이미지에서 CNN 계층을 이용해 중요한 특징을 추출하고, 이 특징들을 Transformer 인코더에서 self-attention을 통해 강조하여 처리합니다. 모델은 절대 품질 점수와 상대적 순위를 동시에 생성하여 보다 포괄적인 이미지 품질 평가를 가능하게 합니다.

- **Performance Highlights**: 제안된 모델은 LIVE, TID2013, CSIQ, LIVE-C, KonIQ10K 등 여러 유명한 NR-IQA 데이터셋에서 평가한 결과, 기존의 NR-IQA 방법론보다 성능이 우수함을 입증하였습니다. 특히 소규모 데이터셋에서 더 좋은 성능을 나타냈습니다.



### Fast Medical Shape Reconstruction via Meta-learned Implicit Neural Representations (https://arxiv.org/abs/2409.07100)
- **What's New**: 본 논문에서는 meta-learning을 활용하여 3D 해부학적 구조의 복원을 목표로 하였으며, 이는 기존 기술 대비 추론 시간을 대폭 단축시키고 정확도를 유지하는 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 Implicit Neural Representation (INR) 함수와 meta-learning 기법을 결합하여 제한된 관측으로부터 3D 해부학적 모양을 빠르게 복원합니다. 또한, Convolutional Neural Networks (CNNs) 및 MLP(Multi-Layer Perceptron) 아키텍처를 사용하여 공간적 좌표에 선형적으로 대응하는 강력한 모델을 구축합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 CT 및 MRI를 포함한 다양한 데이터셋에서 우수한 성능을 보여주었으며, 특히 sparse한 슬라이스 구성에서도 효과적인 복원이 가능했습니다. 단 한 번의 최적화 스텝으로 새로운 해부학적 형태를 재구성할 수 있는 능력이 뛰어난 전달 가능성을 자랑합니다.



### Deep intra-operative illumination calibration of hyperspectral cameras (https://arxiv.org/abs/2409.07094)
Comments:
          Oral at MICCAI 2024

- **What's New**: 이번 논문에서는 새로운 하이퍼스펙트럼 이미징(HSI) 기술을 소개하며, 수술 환경에서 조명 조건 변경으로 인한 성능 저하 문제를 해결하기 위한 자동 재보정(procalibration) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 실시간 조명 재보정을 가능하게 하여, 수술 중 HSI 이미지의 정확도를 높이는 데이터 기반 접근 방식을 사용합니다. 이는 기존의 물리적 화이트 레퍼런스 측정을 대체하며, 다양한 조명 조건에서 안정적인 재보정을 가능하게 합니다.

- **Performance Highlights**: 742개의 HSI 큐브를 통해 검증한 결과, 제안된 방법이 이전 방법들보다 뛰어난 성능을 보였으며, 다양한 종(species), 조명 조건, 이미지 처리 작업에서도 일반화 가능함을 입증했습니다.



### CWT-Net: Super-resolution of Histopathology Images Using a Cross-scale Wavelet-based Transformer (https://arxiv.org/abs/2409.07092)
- **What's New**: 본 논문에서는 병리 이미지에서의 다층 구조의 중요성을 반영하지 않은 기존 슈퍼 해상도(Super-resolution, SR) 방법의 한계를 극복하기 위해 CWT-Net이라는 새로운 네트워크를 제안합니다. CWT-Net은 여러 스케일에서 이미지를 변환하고 특징을 효과적으로 통합할 수 있는 Transformer 아키텍처를 활용합니다.

- **Technical Details**: CWT-Net은 두 개의 브랜치를 포함합니다: 하나는 슈퍼 해상도를 학습하고, 다른 하나는 높은 주파수의 웨이브릿(wavelet) 특징을 추출하는 데 전념합니다. 이 네트워크는 웨이브릿 재구성 모듈을 설계하여 웨이브릿 도메인 특징을 향상시키고, 교차 스케일 이미지를 통해 추가 정보를 도입할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, CWT-Net은 최신 상태의 방법들과 비교했을 때 성능과 시각화 평가 모두에서 큰 개선을 보여주었으며, 이미지 진단 네트워크의 정확도를 크게 향상시킬 수 있는 가능성을 시사합니다.



### EVENet: Evidence-based Ensemble Learning for Uncertainty-aware Brain Parcellation Using Diffusion MRI (https://arxiv.org/abs/2409.07020)
Comments:
          15 pages, 5 figures

- **What's New**: 이번 연구에서는 확산 MRI(diffusion MRI)를 사용하여 해부학적 뇌 구획화를 위한 증거 기반 집합 신경망인 EVENet를 개발하였습니다. EVENet의 주된 혁신점은 단일 추론에서 각 voxel의 예측 불확실성을 정량화하기 위한 증거 기반 딥 러닝(evidential deep learning) 프레임워크의 설계입니다.

- **Technical Details**: EVENet는 다섯 개의 병렬 서브 네트워크(subnetwork)로 구성되어 있으며, 각 서브 네트워크는 특정 확산 MRI 매개변수를 위한 FreeSurfer 구획화를 학습하는 데 전념합니다. 이후 증거 기반 집합 방법론을 제안하여 개별 출력을 융합합니다. 본 연구에서는 건강한 성인 및 다양한 뇌 질환 환자들로부터의 확산 MRI 데이터를 포함한 다수의 대규모 데이터셋에서 실험적 평가를 수행했습니다.

- **Performance Highlights**: 여러 최신 방법들과 비교하여, EVENet는 여러 테스트 데이터셋 전반에 걸쳐 구획화 정확도를 크게 향상시켰습니다. 또한 불확실성 추정 덕분에 EVENet는 병변이 있는 환자에서 비정상 뇌 영역을 잘 탐지하는 능력을 보여 주며, 구획화 결과의 해석 가능성과 신뢰성을 향상시켰습니다.



### Towards Predicting Temporal Changes in a Patient's Chest X-ray Images based on Electronic Health Records (https://arxiv.org/abs/2409.07012)
- **What's New**: 이 논문에서는 의료 기록 (EHR) 데이터를 통합하여 미래의 흉부 X-레이 이미지 (CXR)를 예측하는 EHRXDiff라는 새로운 프레임워크를 제안합니다. 이 모델은 과거의 CXR 이미지와 후속 의료 이벤트를 기반으로 질병의 진행 상황을 동적으로 추적하고 예측합니다.

- **Technical Details**: 제안된 EHRXDiff 프레임워크는 이전의 CXR 이미지와 일련의 의료 이벤트 (e.g., 처방, 검사 결과 등)를 기반으로 목표 CXR 이미지를 예측합니다. 이 모형은 잠재 확산 모델 (latent diffusion model)을 사용하며, 과거의 이미지와 의료 사건 기록을 조합하여 환자의 상태 변화에 대한 실시간 정보를 제공합니다.

- **Performance Highlights**: EHRXDiff 모델은 임상적 일관성, 인구 통계적 일관성, 시각적 현실성을 포함한 세 가지 주요 측면에서 성능을 평가받았으며, 의료 업계에서 환자의 변화하는 상태를 추적하는 데 효과적인 시뮬레이션 도구로서의 잠재력을 보여주었습니다.



### Performance Assessment of Feature Detection Methods for 2-D FS Sonar Imagery (https://arxiv.org/abs/2409.07004)
- **What's New**: 이번 연구는 다섯 개의 서로 다른 FL(Forward-Look) 소나 장비에서 획득한 실제 이미지를 통해 음향 이미징 분야에서 특징 검출 방법의 성능을 포괄적으로 평가합니다. 기존의 연구들과는 달리, 이 연구는 합성 데이터 대신 실험적인 해양 환경에서의 실제 데이터를 활용하여 엄청난 기반을 마련하고 있습니다.

- **Technical Details**: 연구에서는 SIFT, SURF, FAST, ORB, BRISK, SU-BRISK, F-SIFT, KAZE와 같은 8가지 잘 알려진 특징 검출 기법을 적용해 성능을 평가했습니다. 평가 지표로는 검출 정확도, 허위 양성 오류, 그리고 목표 특성과 소나 장치의 변동성에 대한 견고함이 포함됩니다. 데이터는 아크로스 5개의 소나 장치(Aris Explorer 3000, BlueView M900, двойное DIDSON, Gemini 1200ik, Oculus M1200d)에서 수집되었습니다.

- **Performance Highlights**: 소나 이미지를 처리한 후, Oculus 소나 이미지가 가장 많은 특징을 검출하는 경향을 보였으며, 이 연구는 자동 탐지 및 인식, 내비게이션, 도킹, 맵핑 등 다양한 어항 밑의 능력을 향상시키기 위한 특징 검출 방법의 개발을 지원합니다.



### RICAU-Net: Residual-block Inspired Coordinate Attention U-Net for Segmentation of Small and Sparse Calcium Lesions in Cardiac C (https://arxiv.org/abs/2409.06993)
Comments:
          18 pages, 4 figures, 3 tables

- **What's New**: 이번 연구에서는 Residual-block Inspired Coordinate Attention U-Net (RICAU-Net)라는 새로운 딥러닝 알고리즘을 제안합니다. 이 모델은 coronary artery calcium (CAC) segmentation을 위해 coordinate attention을 활용하며, 맞춤형 combo loss function을 적용하여 class imbalance 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: RICAU-Net은 encoder와 decoder 모두에 CA 모듈을 통합하여 설계된 U-Net 기반 FCN 구조입니다. CAC segmentation을 개선하기 위해 CA 모듈은 CAC의 위치 정보를 강조하며, RICA blocks에서 positional 정보를 활용하여 작은 병변의 특징을 더욱 잘 반영합니다.

- **Performance Highlights**: 제안된 방법은 6개의 다른 방법들과 비교했을 때, 모든 병변에 대해 최고 Dice score를 기록하였고, 특히 LM의 CAC에 대해 다른 방법들보다 우수한 성능을 나타냈습니다. ablation study를 통해 coordinate attention과 맞춤형 손실 함수의 중요성이 입증되었습니다.



### Ordinal Learning: Longitudinal Attention Alignment Model for Predicting Time to Future Breast Cancer Events from Mammograms (https://arxiv.org/abs/2409.06887)
- **What's New**: 본 논문에서는 OA-BreaCR이라는 새로운 방법론을 통해 유방암(BC) 리스크 평가를 보다 정밀하게 수행할 수 있는 모델을 제안합니다. 이 모델은 BC 사건의 시계열 관계를 명확하게 모델링하고, 장기적인 유방 조직 변화를 보다 설명 가능한 방식으로 통합합니다.

- **Technical Details**: OA-BreaCR는 시간-사건 예측(time-to-event prediction)과 리스크 분류(risk stratification) 작업을 동시에 고려하는 새로운 프레임워크를 제안합니다. 이 모델은 다중 시간점 검출기(multi-time point imaging) 데이터를 활용하여 유방암 위험과 시간 예측 정확도를 높이며, 주의 집중 메커니즘(attention alignment mechanism)을 사용하여 해석 가능성을 극대화합니다.

- **Performance Highlights**: OA-BreaCR는 기존의 유방암 리스크 예측 및 시간 예측 방법보다 우수한 성능을 나타내었으며, 기존 방법 대비 BC 리스크와 미래 사건 발생 시기 예측 작업 모두에서 더 나은 결과를 보였습니다. 연구 결과는 유방암 조기 선별과 예방 조치를 개선하기 위한 해석 가능하고 정밀한 리스크 평가의 중요성을 강조합니다.



### Bifurcation Identification for Ultrasound-driven Robotic Cannulation (https://arxiv.org/abs/2409.06817)
- **What's New**: BIFURC는 초음파 영상을 이용하여 혈관의 분기점을 자동으로 식별하는 알고리즘으로, 자율 로봇 카뉼레이션 시스템을 위한 최적의 바늘 삽입 위치를 제공합니다. 이는 기존의 알고리즘과 구별되며, 실제 데이터 기반의 훈련이 가능합니다.

- **Technical Details**: BIFURC는 깊이 학습 기법과 전문가 지식을 통합하여 제한된 양의 생체 데이터에서 혈관 분기점을 효율적으로 탐지합니다. 이 알고리즘은 페모랄(femoral) 지역 내에서 혈관의 구조를 3D 형태로 표현하고, 바늘 삽입 위치를 자동으로 결정하는 최초의 방법으로, 실제 실험에서 전문가와 비슷한 수준의 성능을 기록했습니다.

- **Performance Highlights**: BIFURC는 의료 팬텀 및 실제 생체 실험(예: 생돼지)을 통해 혈관 분기점과 바늘 삽입 위치를 일관되게 식별했습니다. 이 결과는 전문가 임상의와 일치하며, 대량 손상 상황에서도 빠르고 안전한 혈관 접근을 가능하게 합니다.



### Automated Quantification of White Blood Cells in Light Microscopic Images of Injured Skeletal Musc (https://arxiv.org/abs/2409.06722)
Comments:
          2 tables, 7 figures, 8 pages

- **What's New**: 이 논문은 자동화된 프레임워크를 제안하여 근육 손상 시 관찰된 백혈구(White Blood Cells, WBCs)를 정량화하고 분석하는 방법을 제시합니다. 기존의 방법과 달리 이 시스템은 Light microscopy 이미지를 사용하여 WBC의 수를 세고 단백질 발현 변화를 추적하는 데 있어 효율성을 높이고 있습니다.

- **Technical Details**: 제안된 프레임워크는 Localized Iterative Otsu's threshold 방법에 기초하여 근육 경계 감지(muscle edge detection) 및 관심 영역(region of interest, ROI) 추출을 수행합니다. 기존의 ImageJ에서 사용되는 임계값(threshold) 방법과 비교할 때, LI Otsu's threshold 방법은 배경 영역에 대한 저항력이 뛰어나고 더 나은 정확도를 제공합니다.

- **Performance Highlights**: CD68 양성 세포(CD68-positive cells)의 분석 결과는 제안된 방법의 유효성을 입증합니다. 이 시스템은 근육 단백질 변화를 정량화하여, 회복 과정을 보다 명확히 이해할 수 있도록 돕습니다.



### Detailed delineation of the fetal brain in diffusion MRI via multi-task learning (https://arxiv.org/abs/2409.06716)
- **What's New**: 이번 연구에서는 태아의 뇌 표면을 고속, 정확하게 분석할 수 있는 통합된 컴퓨테이셔널 프레임워크를 개발하였습니다. 이 프레임워크는 (1) 뇌 조직을 백질, 피질/피질하 회색질, 뇌척수액으로 분할하고, (2) 31개의 백질 경로를 구분하며, (3) 뇌의 피질을 해부학적으로 유의미한 96개의 영역으로 나누는 기능을 가지고 있습니다.

- **Technical Details**: 연구에서는 다중 과제 학습(multi-task learning) 접근 방식을 사용하여 단일 모델을 통해 태아 뇌의 여러 구조를 정의했습니다. 주 조치는 뇌 조직의 세분화(tissue segmentation), 백질 경로의 세분화(white matter tract segmentation), 그리고 피질의 구획화(parcellation)입니다. 이 작업들은 자동화된 방법과 세미 자동화된 방법을 사용하여 97개의 태아 뇌 이미지를 주석(annotation)하였습니다.

- **Performance Highlights**: 새롭게 개발한 방법은 각 작업에서 평균 Dice 유사성 계수(Dice similarity coefficient)를 0.865(조직 세분화), 0.825(백질 경로 세분화), 0.819(구획화)로 정확하게 수행할 수 있는 성능을 보여줍니다. 이 연구는 태아 뇌의 촬영 및 분석 방법을 획기적으로 발전시킬 것으로 기대됩니다.



### FCDM: Sparse-view Sinogram Inpainting with Frequency Domain Convolution Enhanced Diffusion Models (https://arxiv.org/abs/2409.06714)
- **What's New**: 이번 논문에서는 Computed Tomography (CT)에서의 방사선 오염을 줄이기 위한 새로운 접근법을 제안합니다. 연구진은 Frequency Convolution Diffusion Model (FCDM)을 개발하여 sinogram의 효과적인 inpainting을 달성하였습니다.

- **Technical Details**: FCDM은 주파수 영역에서 convolution을 수행하여 다양한 각도에서 주파수 정보를 추출하고 이러한 각도 간의 복잡한 관계를 캡처합니다. 또한, 고유한 sinogram 특성에 기반한 손실 함수(loss function)를 설계하여 물리적 특성을 일관되게 유지하면서 대규모 마스크 영역에서도 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: FCDM은 기존의 아홉 개의 inpainting 모델과 비교했을 때, 시각적으로 및 정량적으로 sinogram 품질을 33% 향상시키고 SSIM(p가 0.95 이상) 및 PSNR(p가 30 이상)을 기록하여 뚜렷한 성과를 보여주었습니다.



New uploads on arXiv(cs.AI)

### "My Grade is Wrong!": A Contestable AI Framework for Interactive Feedback in Evaluating Student Essays (https://arxiv.org/abs/2409.07453)
- **What's New**: CAELF라는 새로운 프레임워크는 대화형 피드백(Interactive Feedback)을 자동화하기 위한 Contestable AI 기반의 대형 언어 모델(LLM) 시스템을 소개합니다. 이 시스템은 학생들이 피드백을 질의하고 도전할 수 있도록 허용하여, 보다 쌍방향적인 학습 환경을 조성합니다.

- **Technical Details**: CAELF는 Teaching-Assistant Agents(TA Agents)와 Teacher Agent로 구성된 다중 에이전트 시스템을 통해 피드백을 생성합니다. 각 TA Agent는 평가 기준에 따라 에세이를 독립적으로 평가한 후, Teacher Agent가 이 평가들을 종합하여 피드백과 점수를 생성합니다. 학생은 이후 이 피드백을 도전할 수 있으며, 이는 수업 준비와 제공에 필요한 시간과 자원을 절약합니다.

- **Performance Highlights**: CAELF는 500개의 비판적 사고 에세이에 대한 사례 연구를 통해, 초기 채점 정확도에서 GPT-4와 유사한 성능을 보였으며, 사용자의 도전에도 일관된 평가를 유지하며 상호작용을 통한 채점 정확도에서 현저한 개선을 나타냈습니다. 사용자 연구에서도 CAELF의 피드백이 사실적 정확성, 자기 조절 능력, 그리고 미래 개선 제안 측면에서 기반선보다 우수한 성과를 나타냈습니다.



### SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories (https://arxiv.org/abs/2409.07440)
- **What's New**: 이번 연구에서는 LLMs가 연구 리포지토리에서 결과를 스스로 재현할 수 있는지 평가하기 위한 첫 번째 벤치마크인 SUPER를 소개합니다.

- **Technical Details**: SUPER 벤치마크는 end-to-end 문제 45개, 전문 문제에서 유래된 152개의 하위 문제, 자동 생성된 602개의 문제를 포함하며, 연구자들이 직면하는 현실적인 도전 과제를 포착하려고 합니다.

- **Performance Highlights**: 최고 모델인 GPT-4o는 end-to-end 문제의 16.3%와 시나리오의 46.1%만을 성공적으로 해결하여, 이 문제의 복잡성을 보여 줍니다.



### A Framework for Predicting the Impact of Game Balance Changes through Meta Discovery (https://arxiv.org/abs/2409.07340)
Comments:
          11 pages, 1 figure, IEEE Transactions on Games

- **What's New**: 이 논문에서는 게임의 균형 변화가 미치는 영향을 예측할 수 있는 메타 발견 프레임워크(Meta Discovery Framework)를 제안합니다. 이 프레임워크는 Reinforcement Learning(강화 학습)을 활용하여 균형 변화의 자동 테스트를 수행합니다.

- **Technical Details**: 제안된 프레임워크는 전투 에이전트(battle agent), 팀 빌더(team-builder), 환경(environment) 세 가지 구성 요소로 이루어져 있습니다. 이러한 요소들은 변경된 메타게임(메타)에 대한 영향을 분석하기 위해 사용됩니다. 우리는 ABC-Meta(Analyzing Balance Changes on the Metagame) 작업을 정의하고, 과거 Pokémon Showdown의 데이터를 사용하여 테스트하였습니다.

- **Performance Highlights**: Pokémon Showdown에서의 실험 결과는 제안된 프레임워크가 균형 변화의 결과를 높은 정확도로 예측할 수 있음을 보여줍니다. 이로 인해 개발자들은 게임의 라이브 환경에 변화를 배포하기 전에 영향을 더 잘 이해할 수 있을 것으로 기대됩니다.



### Explanation, Debate, Align: A Weak-to-Strong Framework for Language Model Generalization (https://arxiv.org/abs/2409.07335)
- **What's New**: 이 논문은 AI 시스템의 정렬(alignment) 문제를 해결하기 위해 언어 모델(context of language models)에서 약한 모델(weak model)과 강한 모델(strong model) 간의 일반화를 통해 새로운 접근 방식을 제안합니다. 이전 연구에서 인간-에이전트 간의 정렬을 위한 설명 생성을 바탕으로 다중 에이전트 시스템(multi-agent systems) 및 인간-AI 팀(human-AI teams)의 복잡한 동학을 다룹니다.

- **Technical Details**: 약한 모델(weak model)과 강한 모델(strong model)의 개념을 정의한 후, 강한 모델이 약한 모델을 향상시키는 ‘유도 함수(facilitation function)’를 통해 지식 전이를 정형화합니다. 이 방법은 설명 생성을 넘어 AI 모델의 성능을 향상시키기 위한 논쟁 기반 정렬(debate-based alignment)을 통합합니다.

- **Performance Highlights**: 결과적으로, 이 유도 기반 접근법을 통해 모델 성능이 향상되었으며, AI 시스템의 정렬 및 확장 가능한 감시 가능성에 대한 인사이트를 제공하는 것으로 나타났습니다.



### Using Generative Agents to Create Tip Sheets for Investigative Data Reporting (https://arxiv.org/abs/2409.07286)
Comments:
          Short paper to be presented at Computation + Journalism 2024

- **What's New**: 이 논문은 조사를 위한 데이터 보고서를 작성하기 위해 생성적 AI 에이전트를 이용한 시스템을 소개합니다. 이 시스템은 데이터 세트에서 유의미한 정보를 생성하고 정제하기 위해 분석가, 기자, 편집자 등 세 가지 전문화된 에이전트를 사용합니다.

- **Technical Details**: 본 시스템은 OpenAI의 Assistants API를 사용하여 GPT-4의 데이터 분석 및 해석 기능을 활용합니다. 이 생성적 에이전트 파이프라인은 질문 생성, 분석 계획 단계, 실행 및 해석, 그리고 최종 결과 정리 단계를 포함하여, 각각의 역할이 상호 피드백을 제공하여 더 나은 결과를 만들어 냅니다. 데이터를 제공하면 최종 출력으로 tip sheet가 제공됩니다.

- **Performance Highlights**: 실제 조사 사례를 통해 검증한 결과, 우리의 에이전트 기반 시스템이 에이전트가 없는 기준 모델에 비해 일반적으로 더 뉴스 가치가 높고 정확한 통찰력을 생성한다는 것을 보여주었습니다. 이는 생성적 AI가 조사 데이터 보고서 작성에 있어 리드(lead)를 찾는 데 잠재력이 있음을 강조합니다.



### DCMAC: Demand-aware Customized Multi-Agent Communication via Upper Bound Training (https://arxiv.org/abs/2409.07127)
- **What's New**: 본 논문에서는 수요 기반 커스터마이즈드 다중 에이전트 통신 프로토콜(DCMAC)을 제안하여, 제한된 통신 자원에서 에이전트의 협업을 최적화하는 방법을 다룹니다. DCMAC는 커스터마이즈드 메시지를 생성하고, 불확실성을 줄이는 방향으로 설계되었습니다.

- **Technical Details**: DCMAC는 에이전트가 서로의 작은 메시지를 통해 요구 사항을 파악하고, 요구 사항과 로컬 관찰 간의 상관관계를 계산하여 메시지를 커스터마이즈하는 방법을 사용합니다. 본 연구에서는 최대 보상의 상한 기반 훈련 패러다임을 제시하여, 훈련 모드에서는 이상 정책을 생성하고 테스트 모드에서는 수요 손실 함수와 시간 차이 오류 함수를 사용해 에이전트를 훈련시킵니다.

- **Performance Highlights**: 실험 결과에 따르면 DCMAC는 비제한적 통신 및 통신 제약이 있는 시나리오 모두에서 기본 알고리즘보다 월등한 성능을 보였으며, QMIX 알고리즘과 결합했을 때도 매우 긍정적인 성과를 보여주었습니다.



### Credibility-Limited Revision for Epistemic Spaces (https://arxiv.org/abs/2409.07119)
- **What's New**: 이번 연구에서는 불일치하는 신념 집합을 허용하는 신념 변화의 프레임워크에서 신뢰 제한 수정(credibility-limited revision)의 개념을 확장하여 모든 AGM 수정(AGM revision) 작업자를 포함하는 연장된 신뢰 제한 수정을 정의합니다.

- **Technical Details**: 대표적으로, 기존 신뢰 제한 수정 이론을 기반으로 두 개의 새로운 공리를 추가하여 AGM 수정 작업자와 불일치 신념과 호환되도록 합니다. 이 새로운 이론은 잠재적 세계에 대한 전체 우선 순위(total preorders)를 활용하여 표현됩니다.

- **Performance Highlights**: 연장된 신뢰 제한 수정 작업자들은 기존의 신뢰 제한 수정 작업자보다 더 일반적이며 불일치하는 신념 집합을 다룰 수 있는 확장된 이론으로, AGM 수정 작업자 또한 포함한다는 점이 주요 성과입니다.



### NSP: A Neuro-Symbolic Natural Language Navigational Planner (https://arxiv.org/abs/2409.06859)
Comments:
          8 pages

- **What's New**: 이 논문에서는 자연어 입력에서 경로 계획을 위한 신경-기호(Neuro-Symbolic) 프레임워크인 NSP를 제안합니다. NSP는 LLM의 신경 추론 능력을 활용하여 환경의 기호 표현을 생성하고 기호 경로 계획 알고리즘을 실행합니다. 이 과정에서 문법 오류를 수정하고 실행 시간 제약을 충족시키는 피드백 루프가 포함됩니다.

- **Technical Details**: NSP 프레임워크는 자연어(NL) 지침을 기반으로 경로 계획 문제를 해결하며, LLM을 통해 환경의 기호 그래프 표현과 경로 계획 알고리즘을 생성합니다. 알고리즘은 그래프에서 실행되어 솔루션 경로를 생성하고, 신경 생성 과정에 피드백 루프를 통해 문법 오류를 수정합니다.

- **Performance Highlights**: 실험 결과, NSP 접근 방식은 1500개의 경로 계획 문제를 평가하는 데 90.1%의 유효 경로를 생성하였으며, 평균적으로 최신 신경 접근 방법보다 19-77% 더 짧은 경로를 제공합니다.



### Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS (https://arxiv.org/abs/2409.07448)
- **What's New**: 본 논문에서는 공격자가 쉽게 조작할 수 있는 Network Intrusion Detection Systems (NIDS) 특징을 식별하기 위해 새로운 Perturb-ability Score (PS)를 제안합니다. PS를 사용하여 비조작 가능한 특징만을 선택함으로써 ML 기반 NIDS의 탐지 성능을 유지하면서 적대적 공격에 대한 강건성을 향상시키는 방법을 입증하고 있습니다.

- **Technical Details**: Perturb-ability Score (PS)는 각 특징의 조작 가능성을 정량화하기 위해 도입되었습니다. PS는 0에서 1까지의 범위를 가지며, 0은 조작이 어려운 특징을, 1은 조작이 가능한 특징을 나타냅니다. PS는 여러 요소를 고려하여 계산됩니다: PS1은 엄격한 헤더 특징 및 네트워크 기능성에 초점을 맞추며, PS2는 NIDS 특징의 경우의 수를 고려합니다. PS3는 특징 간의 상관관계를 평가하고, PS4는 공격자가 접근할 수 없는 특징을 고려합니다.

- **Performance Highlights**: 실험 결과, PS를 통해 선택된 비조작 가능한 특징들만으로도 NIDS가 탐지 성능을 유지할 수 있음을 보여주었습니다. 이 접근 방식은 NIDS의 적대적 공격에 대한 저항력을 극대화함으로써, 네트워크 보안 향상에 기여할 것으로 기대됩니다.



### Synthetic continued pretraining (https://arxiv.org/abs/2409.07431)
- **What's New**: 이 논문에서는 작은 도메인 특화된 데이터셋에서 사전 훈련된 언어 모델의 지식을 효과적으로 학습하기 위한 방법으로, 합성 지속 사전 훈련(synthetic continued pretraining)을 제안합니다. 이 방법은 작은 데이터셋을 활용해 더 큰 학습에 적합한 합성 데이터셋을 생성합니다. 이를 위해 EntiGraph라는 합성 데이터 증강 알고리즘을 사용하여 출처 문서에서 중요한 개체들을 추출하고 이들을 연결하여 다양한 텍스트를 생성합니다.

- **Technical Details**: EntiGraph는 텍스트 코퍼스를 개체 목록으로 나눈 다음, 언어 모델을 사용하여 추출된 개체 간의 관계에 대한 텍스트 설명을 생성합니다. 이 방식은 지식 그래프를 구축하여 코퍼스의 다양한 지식을 제공하려고 합니다. 초점은 기존 내용을 단순히 재구성하는 것이 아니라, 보다 깊이 있는 분석과 응용을 통해 주제를 폭넓게 다루는 것입니다.

- **Performance Highlights**: 결과적으로, EntiGraph를 사용하여 생성된 600M 개의 합성 토큰으로 Llama 3 모델을 계속 사전 훈련한 후, 퀘스쳔 세트에서 QA 정확도를 평가한 결과, 합성 데이터는 실제 문서 접근 없이도 80%의 정확도 향상을 제공합니다. 더군다나, 계속된 사전 훈련된 모델은 퀄리티 책에 관련된 복잡한 질문이나 지침을 처리하는 능력을 보여주며, 이는 지식 전이와 관련된 성능 향상을 나타냅니다.



### Hierarchical Reinforcement Learning for Temporal Abstraction of Listwise Recommendation (https://arxiv.org/abs/2409.07416)
Comments:
          18 pages, 4 figures

- **What's New**: 최근의 계층적 강화 학습(High-Level Reinforcement Learning, HRL) 기술을 활용하여, 사용자 인식의 발전과 단기 관심 변화를 다각도로 반영한 새로운 추천 시스템인 mccHRL을 제안합니다. 이 시스템은 고수준(High-Level) 에이전트가 사용자 인식을 연구하고, 저수준(Low-Level) 에이전트가 아이템 선정 정책을 수립하도록 구성되어 있습니다.

- **Technical Details**: mccHRL(모바일-클라우드 협력 계층 강화 학습)은  사용자 인식 및 공간시간(spatiotemporal) 상태를 모델링하고, 사용자 반응과의 상호작용을 통해 장기적인 사용자 선호를 행동으로 제안합니다. 저수준 에이전트는 사용자 단기 관심을 기반으로 아이템 선정 문제를 해결하며, 이는 온디바이스(On-device) 기능을 고려합니다.

- **Performance Highlights**: 실험 결과, 기존의 여러 기법에 비해 mccHRL이 상당한 성능 향상을 보였습니다. 또한, 데이터와 코드는 공개되어 연구 커뮤니티에서 접근할 수 있습니다.



### SoK: Security and Privacy Risks of Medical AI (https://arxiv.org/abs/2409.07415)
- **What's New**: 이 논문은 인공지능(AI) 및 머신러닝(ML)이 적용된 의료 시스템의 보안과 개인 정보 보호 위협을 심도 깊게 탐구합니다.

- **Technical Details**: 논문에서는 기존 연구를 통해 의료 AI 시스템을 겨냥한 적대적 공격(adversarial attacks)의 이해에 있어 중요한 공백을 발견하였고, 의료 환경에서의 특정 적대적 위협 모델과 취약한 응용 도메인을 규명하였습니다.

- **Performance Highlights**: AI 헬스케어 기술 분야의 급속한 발전과 함께 사이버 보안 연구의 필요성이 강조되며, 이러한 시스템들의 보안성과 복원력을 조사하기 위한 미래 연구의 기초 작업을 제시합니다.



### Robust Robot Walker: Learning Agile Locomotion over Tiny Traps (https://arxiv.org/abs/2409.07409)
Comments:
          10 pages, 17 figures

- **What's New**: 본 연구에서는 사족보행 로봇이 다양한 작은 장애물, 즉 '작은 함정(tiny traps)'을 넘을 수 있도록 하는 새로운 접근 방식을 제안합니다. 기존 방법들은 외부 감지 센서에 의존하는 경우가 많지만, 이는 작은 함정을 탐지하는 데 신뢰할 수 없는 경우가 많습니다. 따라서 저자들은 오로지 고유 감각(proprioceptive) 입력에 초점을 맞추었으며, 두 단계의 훈련 프레임워크를 도입하여 접촉 인코더와 분류 헤드를 통해 서로 다른 함정의 암묵적인 표현을 학습합니다.

- **Technical Details**: 제안된 방법은 두 단계 훈련 프레임워크로, 오직 고유 감각 정보를 사용하여 사족보행 로봇이 작은 함정을 성공적으로 통과할 수 있도록 하는 강건한 정책을 제공합니다. 또한, 접촉 힘(contact forces) 및 클래스 인식(classification) 강화를 위한 접촉 인코더를 활용한 명시적-암시적 이중 상태 추정(paradigm)을 개발하였습니다. 이 방법은 속도의 추적이 아닌 목표 추적(goal tracking)으로 정의되며, 조밀한 보상 함수(dense reward functions)와 가짜 목표 명령을 사용하여 훈련의 안정성과 적응성을 향상시킵니다.

- **Performance Highlights**: 엄청난 양의 실험을 통해 제안된 방법의 효과성과 강건성이 입증되었습니다. 제안된 방법은 시뮬레이션 및 실제 환경 모두에서 유효하며, 특히 목표 추적을 통한 향상된 성능을 보여주었습니다. 또한, 새로운 작은 함정 작업을 위한 기준(benchmark)이 제시되었습니다.



### CLNX: Bridging Code and Natural Language for C/C++ Vulnerability-Contributing Commits Identification (https://arxiv.org/abs/2409.07407)
Comments:
          8 pages, 2 figures, conference

- **What's New**: 이 논문에서는 C/C++ 코드에서의 취약성 기여(commit) 식별을 향상시키기 위해 CodeLinguaNexus (CLNX)라는 새로운 경량 미들웨어를 제안합니다. CLNX는 C/C++ 프로그램과 LLM 간의 상호 작용을 원활하게 도와줍니다.

- **Technical Details**: CLNX는 구조 수준 자연화(structure-level naturalization)와 토큰 수준 자연화(token-level naturalization)의 두 단계로 작동합니다. 먼저, C/C++ 소스 코드를 선형화하여 복잡한 프로그램 구조를 단순화하며, 이후 특수 C/C++ 기호를 자연어 표현으로 변환합니다. 이를 통해 LLM이 코드 이해도를 높입니다.

- **Performance Highlights**: CLNX를 사용한 CodeBERT는 LLM의 C/C++ VCC 식별 성능을 14.48% 향상시키며, 실제 OSS 취약성을 38건 찾아내는 성과를 보였습니다.



### What to align in multimodal contrastive learning? (https://arxiv.org/abs/2409.07402)
Comments:
          22 pages

- **What's New**: 이 논문에서는 CoMM(Contrastive MultiModal)이라는 새로운 방법론을 제안하여 여러 종류의 콘텐츠와 형식 간의 상호작용을 보다 효과적으로 학습할 수 있도록 한다. CoMM은 단일 다중 모드 공간에서 모드 간의 소통을 가능하게 하여 이전의 방법과 대비된다.

- **Technical Details**: CoMM은 서로 다른 증강된 다중 모드 피처 간의 상호 정보를 최대화하는 방식으로 다중 모드 표현을 정렬한다. 이 접근 방식은 공유 정보, 협력적인 정보, 고유 정보를 포함하여 모드 간의 상호작용을 측정할 수 있는 강력한 이론적 기초를 가지고 있다.

- **Performance Highlights**: CoMM은 실제 데이터셋에서 다양한 분야에서 최고 수준의 성능을 보여주었으며, 7개의 다중모드 작업에서 최첨단 결과를 달성하였다. 이로 인해 CoMM은 다양한 모드 수와 데이터 유형을 다룰 수 있는 유연한 프레임워크임을 입증하였다.



### Awaking the Slides: A Tuning-free and Knowledge-regulated AI Tutoring System via Language Model Coordination (https://arxiv.org/abs/2409.07372)
- **What's New**: Slide2Lecture는 사용자 맞춤형 학습 경험을 제공하는 지능형 튜터링 시스템으로, 슬라이드를 구조화된 교수 계획으로 변환하고, 학생의 학습 요구에 맞춘 인터랙티브 강의를 생성하는 기능을 갖추고 있습니다.

- **Technical Details**: Slide2Lecture는 입력된 강의 슬라이드를 처리하여 텍스트와 시각적 정보를 포함한 통합 표현으로 변환합니다. 그 후, 여러 유형의 교수 행동으로 형식화하며, 최종적으로 교수를 위한 인터랙티브 튜터링 환경을 관리합니다.

- **Performance Highlights**: Slide2Lecture는 3000회의 강의 세션 동안 20만 건 이상의 학생과의 상호작용을 기록하였고, 사용자 피드백에 기반하여 효과성을 입증하였습니다.



### Demo: SGCode: A Flexible Prompt-Optimizing System for Secure Generation of Cod (https://arxiv.org/abs/2409.07368)
- **What's New**: SGCode라는 유연한 프롬프트 최적화 시스템을 소개하며, 이 시스템은 대형 언어 모델(LLMs)을 이용해 보안이 강화된 코드를 생성할 수 있도록 지원합니다.

- **Technical Details**: SGCode는 AWS 서버에 배포되어 있으며, 프롬프트 최적화 방법을 통합하여 사용자들이 보안 분석 및 성능 보고서를 볼 수 있는 웹 기반 인터페이스가 포함되어 있습니다. 시스템은 FastAPI를 기반으로 하며, 보안 분석 도구인 Bandit와 CodeQL을 통합합니다.

- **Performance Highlights**: SGCode는 LLM 코드 생성의 높은 비용에 비해 최소한의 시스템 비용을 요구하며, 사용자는 보안 취약성 분석과 보안이 강한 코드 생성을 통해 더 많은 통찰력을 얻을 수 있습니다.



### Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks (https://arxiv.org/abs/2409.07353)
- **What's New**: 대형 비전-언어 모델(LVLM)의 안전성을 강화하기 위해 Sim-CLIP+라는 새로운 방어 메커니즘을 제안합니다. 기존 모델에 구조적 수정 없이도 통합할 수 있으며, 적대적인 공격에 대한 저항력을 증대시킵니다.

- **Technical Details**: Sim-CLIP+는 Siamese 아키텍처를 활용하여 CLIP 비전 인코더를 적대적으로 미세 조정합니다. 이는 변형된 샘플과 원래의 샘플 간의 코사인 유사성을 극대화하여 적대적인 조작에 대한 회복력을 촉진합니다.

- **Performance Highlights**: Sim-CLIP+는 COCO와 OKVQA 등의 표준 다운스트림 데이터셋을 사용하여 평가한 결과, 높은 정확도를 유지하면서도 gradient 기반 적대적 공격 및 jailbreak 공격에 대한 저항력을 크게 향상시켰습니다.



### Federated Impression for Learning with Distributed Heterogeneous Data (https://arxiv.org/abs/2409.07351)
- **What's New**: 이 연구에서는 데이터 이질성(data heterogeneity)이 유발하는 catastrophic forgetting 문제를 해결하기 위한 새로운 접근 방식인 FedImpres를 제안합니다. 이 방법은 지역 데이터를 보완하기 위해 합성 데이터를 생성하여 모델의 일반화 성능을 향상시킵니다.

- **Technical Details**: FedImpres는 각 통신 라운드에서 글로벌 모델을 증류(dilution)하여 합성 데이터를 생성합니다. 이 때, 합성 데이터는 전역 정보(global information)를 나타내며, 이는 클라이언트 측에서 지역 데이터와 함께 사용되어 지역 훈련의 일반화(generalization)를 향상시킵니다. 각 클라이언트는 자신의 지역 데이터와 합성 데이터를 사용하여 훈련합니다.

- **Performance Highlights**: 제안된 FedImpres 방법은 BloodMNIST와 Retina 데이터셋에서 최첨단 성능을 달성하였으며, 분류 정확도가 최적화되어 20%까지 개선되었습니다.



### Online Decision MetaMorphFormer: A Casual Transformer-Based Reinforcement Learning Framework of Universal Embodied Intelligenc (https://arxiv.org/abs/2409.07341)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 연구에서는 온라인 환경에서의 자기 인식(self-awareness)과 행동 계획(action planning)을 가능하게 하는 새로운 인공지능 프레임워크인 Online Decision MetaMorphFormer (ODM)을 제안합니다. 이 프레임워크는 여러 신체 형태(morphology)와 다양한 작업(task)을 처리할 수 있도록 설계되었습니다.

- **Technical Details**: ODM은 Transformer 기반의 RL(강화 학습) 아키텍처로, 온라인과 오프라인 학습에 모두 호환됩니다. 이 모델은 보편적인 백본(universal backbone)과 작업 특화 모듈(task-specific modules)로 구성되어 있으며, 신체 형태에 따른 잠재적인 차이를 포착합니다. 또한, 교실 학습(curriculum learning)을 통해 모델을 사전 훈련합니다.

- **Performance Highlights**: 온라인 실험과 몇 가지 환경 테스트를 통해 ODM의 성능과 일반화 능력을 검증하였습니다. 연구 결과는 다양한 신체 형태와 환경에서의 운동 제어(motion control)에 대한 일반 지식을 학습하는 데 기여하고 있습니다.



### Module-wise Adaptive Adversarial Training for End-to-end Autonomous Driving (https://arxiv.org/abs/2409.07321)
Comments:
          14 pages

- **What's New**: 이 논문은 자율주행(AD) 모델을 위한 적대적 훈련을 처음으로 연구하였으며, 새로운 모듈별 적응형 적대적 훈련(Module-wise Adaptive Adversarial Training, MA2T)을 제안합니다. MA2T는 다양한 모듈의 요구를 충족하고, 각 모듈의 기여도에 따라 손실 가중치를 적절하게 조정하는 방식으로 적대적 공격에 대한 모델의 견고성을 높입니다.

- **Technical Details**: MA2T는 두 가지 주요 기법을 채택합니다: 1) 모듈별 노이즈 주입(Module-wise Noise Injection) - 모델의 각 모듈에 입력되기 전에 노이즈를 주입하여 전반적인 목표에 대한 안내를 통해 훈련합니다. 2) 동적 가중치 누적 적응(Dynamic Weight Accumulation Adaptation) - 각 모듈의 기여도에 따라 손실 가중치를 동적으로 조정하는 방법을 사용합니다.

- **Performance Highlights**: 다양한 화이트박스 및 블랙박스 공격 환경에서 nuScenes 데이터 세트를 사용한 실험에서 MA2T는 기존의 적대적 훈련 방법에 비해 5-10%의 향상을 보이며, CARLA 시뮬레이션 환경에서도 자연적인 손상에 대한 저항력을 입증했습니다.



### MEDIC: Towards a Comprehensive Framework for Evaluating LLMs in Clinical Applications (https://arxiv.org/abs/2409.07314)
Comments:
          Technical report

- **What's New**: 이 논문에서는 의료 분야에서 Large Language Models (LLMs)의 평가를 위한 새로운 프레임워크인 MEDIC를 소개합니다. 이는 기존의 benchmark(벤치마크)인 USMLE 이상의 포괄적인 평가가 필요하다는 점을 강조합니다.

- **Technical Details**: MEDIC는 의료적 추론(medical reasoning), 윤리 및 편견(ethics and bias), 데이터와 언어 이해(data and language understanding), 상황 내 학습(in-context learning), 임상 안전(clinical safety) 등 다섯 가지 핵심적인 차원에서 LLM을 평가합니다. 새로운 cross-examination framework를 통해 coverage와 hallucination detection을 정량적으로 평가하며, 참조 출력(reference outputs)이 필요하지 않습니다.

- **Performance Highlights**: MEDIC를 통해 의료 질문-응답, 안전성, 요약, 노트 생성 등 다양한 작업에 대해 LLM의 성능을 평가한 결과, 모델 크기, 기본 모델과 의료적으로 미세 조정된 모델 간의 성능 차이를 보여주었습니다. 이는 특정 모델 강점이 요구되는 응용 분야에 대한 모델 선택에 중요한 의미를 지닙니다.



### Exploring User-level Gradient Inversion with a Diffusion Prior (https://arxiv.org/abs/2409.07291)
Comments:
          Presented at the International Workshop on Federated Learning in the Age of Foundation Models in conjunction with NeurIPS 2023

- **What's New**: 이 연구는 분산 학습에서 사용자 수준의 gradient inversion을 탐구하며, 기존의 샘플별 공격과는 다른 새로운 공격 표면을 제안합니다. 특히, 저자들은 대규모 배치 환경에서 회복을 증진시키기 위해 denoising diffusion model을 사용하는 기법을 개발했습니다.

- **Technical Details**: 이 방법은 기존의 gradient inversion 기법의 한계를 극복하고, 랜덤으로 샘플을 복원하는 대신, 민감한 정보를 포착하는 대표 이미지를 생성하는 데 중점을 둡니다. 이를 통해 계산 오버헤드를 크게 줄이고, 수렴 안정성을 향상시킵니다. 이 연구는 diffusion model을 적용하여 gradient inversion의 정확성을 개선하는 첫 번째 연구입니다.

- **Performance Highlights**: 실험 결과, 이 방법은 CelebA 얼굴 이미지 데이터셋을 사용하여 성별, 인종, 나이 및 얼굴 정체성 등과 같은 개인 정보를 효과적으로 복원하는 능력을 보여주었습니다. 특히, 기존의 적대적 가정에 의존하지 않고 고유한 개인 속성을 회복할 수 있는 가능성을 시연했습니다.



### Propaganda to Hate: A Multimodal Analysis of Arabic Memes with Multi-Agent LLMs (https://arxiv.org/abs/2409.07246)
Comments:
          propaganda, hate-speech, disinformation, misinformation, fake news, LLMs, GPT-4, multimodality, multimodal LLMs

- **What's New**: 이번 연구는 아랍어 소셜 미디어 콘텐츠 내 프로파간다(Propaganda) 및 혐오(Hate) 밈(Meme) 간의 상관관계를 분석하기 위해 멀티 에이전트 LLM 기반 접근 방식을 제시합니다.

- **Technical Details**: 연구는 프로파간다 및 혐오 밈에 대한 coarse 및 fine-grained 레이블을 추가하여 데이터셋을 확장하고, LLM을 데이터 주석자로 활용하여 비전문가가 처리하기 어려운 복잡한 멀티모달 데이터를 자동으로 주석합니다.

- **Performance Highlights**: 실험 결과는 향후 연구의 기준점이 될 수 있으며, 제공된 데이터셋은 커뮤니티에 공개될 예정입니다.



### Behavioral Cloning Models Reality Check for Autonomous Driving (https://arxiv.org/abs/2409.07218)
- **What's New**: 이 논문은 최근의 자율주행 차량 인식 시스템의 실제 적용 가능성을 검증하는 새로운 연구 결과를 제시합니다. 기존의 연구들은 주로 시뮬레이션 환경에서 평가되었으나, 본 연구는 Behavior Cloning (BC)을 이용한 최신 인식 시스템의 실제 환경에서의 성과를 중점적으로 다룹니다.

- **Technical Details**: 제안된 방법은 Autoencoder 기반 Behavioral Cloning (AutoBC), Vision Transformers (ViT) 및 Spatial Attention 메커니즘 등을 포함하여 다양한 아키텍처를 활용하여 실제 주행 조건에서 인식 시스템의 성능을 평가합니다. 이때, 수집된 데이터셋은 축소된 연구 차량을 사용하여 여러 트랙 환경에서 테스트되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들이 스티어링 조향 명령을 실시간으로 예측할 때 낮은 오류 범위를 기록하였으며, 이는 실제 적용 가능성에 대한 희망적인 결과로 평가됩니다.



### Heterogeneity-Aware Coordination for Federated Learning via Stitching Pre-trained blocks (https://arxiv.org/abs/2409.07202)
- **What's New**: FedStitch라는 새롭고 혁신적인 계층적 협력 학습(Federated Learning) 프레임워크가 제안되었습니다. 이 프레임워크는 기존의 학습 방식을 탈피하여 사전 훈련된 모델 블록을 조합하여 글로벌 모델을 구성합니다.

- **Technical Details**: FedStitch는 다음 세 가지 핵심 구성 요소로 이루어져 있습니다: 1) RL-가중 집계기(RL-weighted aggregator), 2) 서버 측에 배포된 검색 공간 최적화기(search space optimizer), 3) 참여 클라이언트에 배포된 로컬 에너지 최적화기(local energy optimizer). 이를 통해 비독립 및 동일 분포(non-IID) 환경에서 블록 선택을 개선하고, 후보 블록 풀의 크기를 지속적으로 줄이며, 각 클라이언트의 에너지 소비를 최소화합니다.

- **Performance Highlights**: FedStitch는 기존 접근 방식에 비해 모델의 정확도를 최대 20.93% 향상시켰고, 학습 속도를 최대 8.12% 단축시켰으며, 메모리 사용량을 최대 79.5% 줄이고, 에너지 소모를 최대 89.41% 절감했습니다.



### ThermalGaussian: Thermal 3D Gaussian Splatting (https://arxiv.org/abs/2409.07200)
Comments:
          10 pages, 7 figures

- **What's New**: 본 논문에서는 ThermalGaussian이라는 새로운 접근 방식을 제안하며, 이는 RGB 및 열화상 모드에서 고품질 이미지를 렌더링 할 수 있는 첫 번째 열 3D Gaussian splatting (3DGS) 방법입니다. 기존의 3DGS에 기반을 두고 있으며, 멀티모달 정규화 제약 조건을 도입하여 단일 모달리티의 과적합을 방지합니다.

- **Technical Details**: 이 연구는 RGB 카메라와 열화상 카메라를 정렬하고, 학습 과정에서 멀티모달 3D 가우시안 (Gaussian)을 사용합니다. 이를 통해, ThermalGaussian은 3DGS의 장점을 기반으로 하여 고해상도의 열 이미지 및 RGB 이미지를 생성합니다. 또한, RGBT-Scenes라는 새로운 데이터셋을 제공하여 열 장면 재구성을 위한 연구를 지원합니다.

- **Performance Highlights**: ThermalGaussian은 기존 대비 RGB 및 열화상 이미지의 렌더링 품질을 모두 개선하며, 특히 90%의 데이터 저장 공간 절약과 렌더링 속도 향상을 이룩하였습니다. 실험 결과는 ThermalGaussian이 섬세한 열 이미지의 포토리얼리스틱 렌더링을 달성하고 있음을 보입니다.



### Cyber Deception: State of the art, Trends and Open challenges (https://arxiv.org/abs/2409.07194)
Comments:
          38 pages

- **What's New**: 사이버 보안에 대한 관심 증가에 따라 Cyber Deception (CYDEC) 메커니즘 설계 및 구현에 관한 다양한 논문이 증가하고 있습니다. 본 논문은 CYDEC의 주요 구성 요소를 포괄적으로 분석하고, 모든 유형의 솔루션을 포함하는 일반적인 분류법을 개발하며, 문헌의 현재 상태에 대한 조사를 통해 이러한 공백을 메우고자 합니다.

- **Technical Details**: CYDEC는 공격자의 주의를 산만하게 하고 그들의 방법, 도구 및 목표에 대한 귀중한 정보를 수집하는 방어 전략입니다. 본 논문에서는 CYDEC의 기본 구성 요소, 분류 피라미드, AI를 활용한 CYDEC 메커니즘, 그리고 이를 통해 수집된 데이터의 사용 방식에 대해 논의합니다. 또한, CYDEC의 프레임워크를 다양한 지표(예: TRL, 공격 전략)로 분석합니다.

- **Performance Highlights**: CYDEC는 전통적인 보안 조치로 놓칠 수 있는 위협을 탐지하고, 공격자의 전술 및 절차(TTPs)를 이해함으로써 방어 전략을 미리 조정할 수 있는 기회를 제공합니다. 이로 인해 CYDEC는 비용 효율적인 솔루션으로 자리 잡고 있으며, 최근 연구들은 CYDEC의 다양한 메커니즘과 인공지능을 통합한 새로운 방향을 제시하고 있습니다.



### How Mature is Requirements Engineering for AI-based Systems? A Systematic Mapping Study on Practices, Challenges, and Future Research Directions (https://arxiv.org/abs/2409.07192)
Comments:
          Accepted in Requirements Engineering Journal, 2024

- **What's New**: 본 연구는 인공지능(AI) 분야에서 요구사항 공학(Requirements Engineering, RE) 문제의 새로운 도전 과제를 다루고 있으며, 특히 RE4AI(인공지능을 위한 요구사항 공학)에 대한 포괄적인 개요를 제공합니다.

- **Technical Details**: 연구팀은 126개의 주요 연구를 분석하고, 요구사항 분석 및 수집에 중점을 둔 기존 RE4AI 연구 현황을 검토하였습니다. 주요 도전 과제로는 요구사항 명세화(requirements specification), 설명 가능성(explainability), 머신러닝 엔지니어와 최종 사용자 간의 간극이 있음을 확인하였습니다. 연구는 체계적인 매핑 연구(systematic mapping study)를 통해 데이터 추출 및 주제 분석을 통해 결과를 종합했습니다.

- **Performance Highlights**: 이 연구는 RE4AI의 현재 연구 트렌드와 도전 과제를 발굴하고, 향후 연구 방향 7가지를 제시하였습니다. 연구 결과는 실무자들이 AI 기반 시스템에 적합한 RE 방법을 선택하는 데 도움을 줄 수 있으며, 연구자들은 확인된 연구 격차를 바탕으로 분야를 발전시킬 수 있습니다.



### A Perspective on AI-Guided Molecular Simulations in VR: Exploring Strategies for Imitation Learning in Hyperdimensional Molecular Systems (https://arxiv.org/abs/2409.07189)
Comments:
          (Accepted for presentation at the First Workshop on "eXtended Reality \& Intelligent Agents" (XRIA24) @ ECAI24, Santiago De Compostela (Spain), 20 October 2024)

- **What's New**: 이 논문은 상호작용 분자 동역학(Interactive Molecular Dynamics, iMD)과 가상 현실(Virtual Reality, VR)의 결합을 통해 AI 에이전트의 훈련을 위한 사용자 생성 데이터 세트를 활용하는 가능성을 탐구합니다.

- **Technical Details**: iMD-VR은 고성능 컴퓨팅을 활용하여 연구자가 고차원 샘플링 문제를 해결할 수 있는 몰입형 3D 환경을 제공합니다. 이를 통해 연구자와 학생들은 실시간으로 분자 운동을 시각화하고 조작하며, 복잡한 시스템을 효율적으로 탐색할 수 있습니다. 또한, 모방 학습(Imitation Learning, IL) 기술을 사용하여 AI 에이전트를 훈련시키는 방법을 제안합니다.

- **Performance Highlights**: VR 환경에서의 iMD-VR은 전통적인 2D 인터페이스보다 더 직관적이고 자연스러운 3D 인터페이스를 제공하여, 복잡한 분자 구조의 이해를 높이고 연구자 간의 협업을 촉진합니다. 이 접근 방식은 재료 과학, 단백질 공학, 컴퓨터 지원 약물 설계 등 다양한 분야에서 가치 있는 통찰을 제공할 수 있는 잠재력을 품고 있습니다.



### Enhancing Angular Resolution via Directionality Encoding and Geometric Constraints in Brain Diffusion Tensor Imaging (https://arxiv.org/abs/2409.07186)
Comments:
          Accepted to ICONIP2024, Diffusion Weighted Imaging, Diffusion Tensor Imaging, Angular Resolution Enhancement, Fractional Anisotropy

- **What's New**: 이번 연구에서는 최소한의 diffusion gradient 방향(6개)으로 획득된 DWI(Diffusion Weighted Imaging)에서도 신뢰할 수 있는 DTI(Diffusion Tensor Imaging) 지표를 추정할 수 있는 깊은 학습 기반의 방법인 DirGeo-DTI를 제안합니다.

- **Technical Details**: DirGeo-DTI는 방향성 인코딩(direction encoding)과 기하적 제약(geometric constraints)을 활용하여 DTI를 향상시키며, 두 개의 공개 DWI 데이터셋을 통해 평가되었습니다. 이 방법은 기존의 DTI 개선 방법에 비해 우수한 성능을 발휘하며, 임상 DWI 스캔으로부터 더 깊은 통찰을 제공할 가능성이 있습니다.

- **Performance Highlights**: DirGeo-DTI는 6개의 고유한 확산 gradient 방향만으로 DTI의 각도 해상도를 향상시키는 최초의 방법으로, 실험 결과 기존 방법들과 비교하여 최상의 성능을 기록했습니다. 이 접근법은 정밀한 DTI 지표를 제공하여 임상에서의 응용 가능성을 높입니다.



### Linear Time Complexity Conformers with SummaryMixing for Streaming Speech Recognition (https://arxiv.org/abs/2409.07165)
- **What's New**: 이 논문에서는 streaming 및 non-streaming 자동 음성 인식(ASR)에서 Self-Attention의 대안으로 SummaryMixing을 소개하고, 이를 Conformer Transducer에 통합하여 성능을 평가하는 내용을 다룹니다.

- **Technical Details**: SummaryMixing은 기존의 Self-Attention 모델의 정확도를 유지하면서 복잡성을 선형 시간으로 줄입니다. 본 연구에서는 Dynamic Chunk Training(DCT)과 Dynamic Chunk Convolution(DCCONV)을 사용하여 ASR 모델이 다양한 맥락 길이에 노출되도록 하여, streaming 및 offline 방식 모두에서 작동할 수 있게 개발되었습니다.

- **Performance Highlights**: 실험 결과, SummaryMixing을 탑재한 Conformer Transducer는 Self-Attention 모델보다 더 적은 메모리와 계산으로 유사하거나 더 나은 정확도(Word Error Rate, WER)를 달성하였고, 학습 및 추론 시의 시간과 메모리 소비를 줄였습니다.



### Recurrent Aggregators in Neural Algorithmic Reasoning (https://arxiv.org/abs/2409.07154)
- **What's New**: 이번 논문은 Neural algorithmic reasoning(NAR) 분야에서 고전적인 알고리즘 계산을 모방하는 신경망을 설계하려는 시도를 설명합니다. 특히, 그래프 신경망(GNN) 대신 순환 신경망(RNN)을 사용하여 새로운 접근법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 순환 집합 함수로서 LSTM(Long Short-Term Memory) 네트워크를 활용하여 RNAR(Gecurrent Neural Algorithmic Reasoner) 모델을 구성했습니다. 이 모델은 CLRS의 순차적 작업에서 이전 연구보다 월등한 성과를 보여줍니다.

- **Performance Highlights**: RNAR 모델은 Heapsort 및 Quickselect 작업에서 최고의 성과를 기록했습니다. 특히 Quickselect 작업에서 RNAR은 평균 micro-F1 점수 87%를 달성하며 현대 NAR의 주요 도전 과제를 해결하였습니다.



### Zero-Shot Text-to-Speech as Golden Speech Generator: A Systematic Framework and its Applicability in Automatic Pronunciation Assessmen (https://arxiv.org/abs/2409.07151)
Comments:
          11 pages, 4 figures, 4 tables

- **What's New**: 이 연구는 L2 학습자들이 발음을 개선하기 위해 그들의 발음 특성과 일치하는 golden speech를 따라 할 때 효과적이라는 가설을 다룹니다. 특히, zero-shot text-to-speech (ZS-TTS) 기술로 생성된 learner-specific golden speech를 활용하여 L2 학습자의 발음 능력을 측정하는 새로운 방법론을 제안합니다.

- **Technical Details**: 연구의 기여는 두 가지입니다: 첫째, golden speech를 생성하는 합성 모델의 능력을 평가하기 위한 체계적인 프레임워크 설계 및 개발, 둘째, 자동 발음 평가 (APA)에서 golden speech 사용 효과에 대한 심도 있는 조사입니다. L2-ARCTIC 및 Speechocean762 벤치마크 데이터 세트를 기반으로 한 포괄적인 실험을 통해 다양한 평가 지표에서 우리 모델이 이전의 방법론들에 비해 상당한 성능 향상을 이룰 수 있음을 발견했습니다.

- **Performance Highlights**: 이 연구는 ZS-TTS와 APA에서 golden speech의 역할을 탐구한 최초의 사례로, 컴퓨터 지원 발음 훈련 (CAPT)에 유망한 방안을 제공합니다.



### Leveraging Unstructured Text Data for Federated Instruction Tuning of Large Language Models (https://arxiv.org/abs/2409.07136)
Comments:
          11 pages, work in progress

- **What's New**: 연합 교육(federated learning)에서의 새로운 접근법인 FedIT-U2S를 제안합니다. 이 프레임워크는 비구조화 텍스트(관여하는 클라이언트가 보유하는 데이터를 의미)를 자동으로 구조화된 데이터로 변환하여 수작업으로 주석을 다는 부담을 줄입니다.

- **Technical Details**: FedIT-U2S는 두 가지 주요 단계를 포함합니다: (1) 소수 샘플 기반(few-shot) 데이터 생성, 클라이언트는 비구조화된 데이터와 예시를 결합하여 대형 언어 모델(LLM)을 통해 명령-응답 쌍을 생성합니다. (2) 생성된 데이터에 기반한 연합 교육(procedure of federated instruction tuning) 진행.

- **Performance Highlights**: 세 가지 분야(의학, 지식, 수학)에서의 실험 결과, FedIT-U2S가 기존 LLM의 성능을 일관되게 향상시키는 것으로 나타났습니다.



### Deep Learning Techniques for Hand Vein Biometrics: A Comprehensive Review (https://arxiv.org/abs/2409.07128)
- **What's New**: 본 논문은 손 정맥 생체 인식(hand vein biometrics) 분야의 최신 깊이 학습(deep learning) 기술 발전을 다루고 있습니다. 이 연구는 손가락 정맥(finger vein), 손바닥 정맥(palm vein), 손등 정맥(dorsal hand vein) 인식의 모든 필수 기본 사항을 포괄하며, 이전 연구에서는 다루지 않았던 최신 기술 및 도전 과제를 논의합니다.

- **Technical Details**: 이 리뷰는 기존의 손 정맥 기반 생체 인식 시스템에서 새롭게 적용된 깊이 학습 기술들을 분석합니다. 기존의 전통적인 방법들과 비교하여, 깊이 학습 모델은 복잡한 정맥 구조의 패턴을 자동으로 인식하며, 다양한 조명 및 손 위치 변화에 대한 저항력을 가지고 있습니다. 이 연구는 데이터 증강(data augmentation) 기술과 효과적인 전이 학습(transfer learning) 방법을 포함하여, 손 정맥 인식의 성공적인 성과에 관한 통찰을 제공합니다.

- **Performance Highlights**: 정맥 기반 생체 인식 기술의 효과적인 전이 학습 기법과 데이터 증강 기술이 모든 검토된 깊이 학습 접근 방식에서 분석되었습니다. 최근 2017년부터 2024년까지의 문헌을 포괄하며, 이는 손 정맥 생체 인식의 신뢰성과 보안을 강화하는 중요한 기초가 됩니다. 수치적으로, 심층 학습을 통한 손 정맥 인식 정확도는 기존 방법들보다 뛰어난 성능을 보이고 있으며, 미래 연구 방향에 대한 통찰을 제공합니다.



### Attention Down-Sampling Transformer, Relative Ranking and Self-Consistency for Blind Image Quality Assessmen (https://arxiv.org/abs/2409.07115)
Comments:
          Accepted in International Conference on Image Processing (ICIP)

- **What's New**: 이번 연구에서는 NR-IQA(Non-Reference Image Quality Assessment) 모델의 성능 향상을 위해 새로운 Transformer 아키텍처와 CNN(convolutional neural networks)을 적절히 활용하여 이미지 품질에 대한 평가 방법을 제안합니다.

- **Technical Details**: 제안된 ADTRS(Attention Down-Sampling Transformer with Relative ranking and self-consistency) 모델은 입력 이미지에서 CNN 계층을 이용해 중요한 특징을 추출하고, 이 특징들을 Transformer 인코더에서 self-attention을 통해 강조하여 처리합니다. 모델은 절대 품질 점수와 상대적 순위를 동시에 생성하여 보다 포괄적인 이미지 품질 평가를 가능하게 합니다.

- **Performance Highlights**: 제안된 모델은 LIVE, TID2013, CSIQ, LIVE-C, KonIQ10K 등 여러 유명한 NR-IQA 데이터셋에서 평가한 결과, 기존의 NR-IQA 방법론보다 성능이 우수함을 입증하였습니다. 특히 소규모 데이터셋에서 더 좋은 성능을 나타냈습니다.



### A Continual and Incremental Learning Approach for TinyML On-device Training Using Dataset Distillation and Model Size Adaption (https://arxiv.org/abs/2409.07114)
- **What's New**: 새로운 알고리즘이 Tiny Machine Learning(TinyML) 맥락에서 점진적 학습(incremental learning)을 위해 제안되었습니다. 이 알고리즘은 성능이 낮고 에너지 효율적인 임베디드 장치를 최적화했습니다.

- **Technical Details**: 이 알고리즘은 지식 증류(knowledge distillation)를 사용하여 작은 데이터 세트를 작성함으로써 치명적인 망각(catastrophic forgetting) 문제를 해결합니다. 모델의 크기를 동적으로 조정할 수 있는 독창적인 방법을 제공합니다.

- **Performance Highlights**: 제안된 알고리즘은 CIFAR10, MNIST, CORE50, HAR, Speech Commands 등 5개의 데이터 세트에서 시험되었으며, 더 큰 고정 모델과 비교하여 43%의 Floating Point Operations(FLOPs)를 사용하면서도 정확도 손실은 단 1%에 불과했습니다. 또한, 기존의 데이터 세트의 1%만을 요구하여 메모리 효율성을 보여주었습니다.



### Advancing On-Device Neural Network Training with TinyPropv2: Dynamic, Sparse, and Efficient Backpropagation (https://arxiv.org/abs/2409.07109)
Comments:
          2024 International Joint Conference on Neural Networks (IJCNN)

- **What's New**: 본 연구에서는 저전력 마이크로컨트롤러 유닛(Microcontroller Units, MCUs)에서의 온디바이스 학습을 위해 최적화된 혁신적인 알고리즘인 TinyPropv2를 소개합니다. TinyPropv2는 sparse backpropagation(희소 역전파)을 개선하여, 훈련 단계에서 선택적으로 점핑할 수 있는 기능을 포함하여 동적으로 희소성 수준을 조정합니다.

- **Technical Details**: TinyPropv2는 기존의 TinyProp 알고리즘에 기반하여, 훈련 중에 역전파 비율을 동적으로 조정하는 접근 방식을 채택합니다. 이 알고리즘은 특정 데이터 포인트에 대해 전체 훈련 단계를 생략할 수 있는 의사결정 과정을 통합하여, 정확성을 크게 해치지 않으면서 계산 작업을 줄이고자 합니다. 실험에서 TinyPropv2는 다양한 데이터 세트에서 높은 정확성을 유지하면서 계산 비용을 최소화합니다.

- **Performance Highlights**: TinyPropv2는 CIFAR 10에서 0.82%, CIFAR100에서 1.07%의 평균 정확도 하락으로 전체 훈련 방법에 거의 근접한 성능을 보여주었습니다. 또한, 일부 시나리오에서는 전체 훈련에서 필요한 계산 비용의 10%만으로도 가능합니다. 일반적인 sparse training 방법론과 비교할 때, TinyPropv2는 계산 작업 효율성을 현저하게 향상시켜, IoT 생태계에서 고급 임베디드 장치의 애플리케이션에 유리한 솔루션으로 자리매김하고 있습니다.



### Redundancy-Aware Camera Selection for Indoor Scene Neural Rendering (https://arxiv.org/abs/2409.07098)
- **What's New**: 본 연구에서는 모니터링된 정적 장면의 새로운 시점을 합성하기 위해 카메라 선택 문제에 접근하고 있습니다. 이는 인도어(Scene) 환경에서 비디오 시퀀스를 캡쳐하고, 그 과정에서 발생하는 중복 정보를 줄일 수 있는 방법을 제안합니다.

- **Technical Details**: 이 작업에서는 유사성 행렬을 구축하고 Intra-List Diversity (ILD) 메트릭을 활용하여 카메라의 중복성을 평가합니다. 최적화 문제로서 카메라 선택 작업을 정의하고, 이를 기반으로 다양성 기반 샘플링 알고리즘을 적용하여 카메라 선택을 최적화합니다. 새로운 데이터셋 IndoorTraj를 개발하여 복잡한 카메라 움직임을 포함하고 있으며, 이를 통해 실험적으로 제안된 방법의 효율성을 입증하였습니다.

- **Performance Highlights**: 우리의 방법은 전체 데이터셋에 대해 훈련된 모델과 유사한 성능을 달성하였음에도 불구하고 평균적으로 15%의 프레임과 75%의 시간만을 사용하여 최적의 결과를 나타냅니다. 5-20%의 데이터만 사용했음에도 불구하고 기준 전략들을 지속적으로 초과하는 성과를 보였습니다.



### CWT-Net: Super-resolution of Histopathology Images Using a Cross-scale Wavelet-based Transformer (https://arxiv.org/abs/2409.07092)
- **What's New**: 본 논문에서는 병리 이미지에서의 다층 구조의 중요성을 반영하지 않은 기존 슈퍼 해상도(Super-resolution, SR) 방법의 한계를 극복하기 위해 CWT-Net이라는 새로운 네트워크를 제안합니다. CWT-Net은 여러 스케일에서 이미지를 변환하고 특징을 효과적으로 통합할 수 있는 Transformer 아키텍처를 활용합니다.

- **Technical Details**: CWT-Net은 두 개의 브랜치를 포함합니다: 하나는 슈퍼 해상도를 학습하고, 다른 하나는 높은 주파수의 웨이브릿(wavelet) 특징을 추출하는 데 전념합니다. 이 네트워크는 웨이브릿 재구성 모듈을 설계하여 웨이브릿 도메인 특징을 향상시키고, 교차 스케일 이미지를 통해 추가 정보를 도입할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, CWT-Net은 최신 상태의 방법들과 비교했을 때 성능과 시각화 평가 모두에서 큰 개선을 보여주었으며, 이미지 진단 네트워크의 정확도를 크게 향상시킬 수 있는 가능성을 시사합니다.



### Ontology-Free General-Domain Knowledge Graph-to-Text Generation Dataset Synthesis using Large Language Mod (https://arxiv.org/abs/2409.07088)
Comments:
          18 pages, 9 figures

- **What's New**: 이번 연구는 고품질 G2T(지식 그래프-텍스트) 데이터셋을 생성하기 위한 새로운 방법인 WikiOFGraph를 소개합니다. 이는 LLM(대형 언어 모델)과 Data-QuestEval을 활용하여 생성된 대규모 G2T 데이터셋으로, 5.85M의 일반 도메인 그래프-텍스트 쌍을 포함하고 있습니다.

- **Technical Details**: G2T(GraphtoText) 생성은 (주어, 서술어, 목적어) 형태의 triplet을 자연어로 변환하는 작업입니다. 기존의 PLM(사전 훈련된 언어 모델)들은 고품질의 그래프-텍스트 정렬이 잘된 데이터셋이 필요합니다. 그러나 고품질 일반 도메인 G2T 데이터셋의 부족이 이 연구 분야의 발전을 제한해 왔습니다. 본 연구에서는 LLM과 Data-QuestEval을 연계하여 새로운 데이터셋 WikiOFGraph를 생성하고 이를 통해 그래프-텍스트 일관성을 보장합니다.

- **Performance Highlights**: WikiOFGraph로 미세 조정된 PLM은 기존 데이터셋에서 훈련된 PLM보다 다양한 평가 지표에서 뛰어난 성능을 나타냅니다. 이 데이터셋은 인간이 제작한 GenWiki 테스트 세트와 LLM로 생성된 WikiOFGraph 테스트 세트에서의 성능이 우수하며, 고품질 G2T 시스템 구축에 적합하다는 것을 입증했습니다.



### Multimodal Emotion Recognition with Vision-language Prompting and Modality Dropou (https://arxiv.org/abs/2409.07078)
- **What's New**: 본 논문은 제2회 멀티모달 감정 인식 챌린지 트랙 1(MER2024-SEMI)에서의 솔루션을 제시하며, 다양한 멀티모달 감정 인식 방법을 제안합니다.

- **Technical Details**: EmoVCLIP 모델은 CLIP을 기반으로 하는 비전-언어 프롬프트 학습을 통해 감정 인식 작업을 Fine-tuning하여 영상 기반 감정 인식을 위한 다수의 방법을 개발하였습니다. 또한, modality dropout을 통해 강력한 정보 융합을 구현하고, Baichuan의 감정 정보 추출을 개선하기 위해 GPT-4를 제안하였습니다. 모델은 self-training 전략을 사용하여 레이블이 없는 비디오를 활용합니다.

- **Performance Highlights**: 우리의 모델은 MER2024-SEMI 트랙에서 1위를 차지했으며, 테스트 세트에서 90.15%의 정확도를 기록했습니다.



### Legal Fact Prediction: Task Definition and Dataset Construction (https://arxiv.org/abs/2409.07055)
- **What's New**: 이번 논문에서는 법적 사실 예측 (Legal Fact Prediction)이라는 새로운 자연어 처리(NLP) 과제를 소개합니다. 이는 법원에서 제공된 증거 목록을 바탕으로 법적 사실을 예측하는 것을 목표로 하며, 이를 통해 재판에 관련된 당사자와 변호사들이 전략을 최적화할 수 있도록 돕습니다.

- **Technical Details**: 법적 사실은 증거가 제시되는 과정에서 판사가 결정하는 사건의 사실을 의미합니다. 논문에서는 공개된 법원 판결문과 재판 기록을 통해 증거 목록을 추출하여 현실적인 데이터셋(LFPLoan)을 구축했습니다. 두 가지 기준을 바탕으로 LLM(대형 언어 모델)을 사용하여 법적 사실을 예측하는 실험을 진행했습니다. 실험에서 법적 사실 예측 작업은 복잡한 추론 과정을 필요로 하며 상반되는 증거를 처리하는 난이도가 있음을 발견했습니다.

- **Performance Highlights**: 법적 사실 예측의 성능은 다양한 방법으로 평가되었으며, LLM을 활용한 기초 방법들이 비트리비얼한 예측 능력을 보였습니다. 그러나 여전히 상당한 발전의 여지가 있으며, 이는 법적 사실 예측이 복잡한 정보를 처리하고 추론할 수 있는 능력을 요구하기 때문입니다. 기본 아이템에 대한 예측은 비교적 높은 정확도를 보였으나, 논쟁의 여지가 있는 법적 사실에 대해서는 저조한 성능을 보였습니다.



### Native vs Non-Native Language Prompting: A Comparative Analysis (https://arxiv.org/abs/2409.07054)
Comments:
          Foundation Models, Large Language Models, Arabic NLP, LLMs, Native, Contextual Understanding, Arabic LLM

- **What's New**: 본 연구에서는 12개의 아랍어 데이터셋(9.7K 데이터 포인트)을 사용하여 11개의 NLP 태스크에 대해 다양한 프롬프트 전략(네이티브 vs. 비네이티브)의 성능을 조사하였습니다. 실험을 통해 비네이티브 프롬프트가 평균적으로 가장 높은 성능을 발휘하며, 혼합 프롬프트와 네이티브 프롬프트가 그 뒤를 잇는다는 결과를 도출하였습니다.

- **Technical Details**: LLM(대규모 언어 모델)의 성능을 극대화하기 위해서는 프롬프트 엔지니어링이 필수적입니다. 연구에서는 제로샷(zero-shot), 피숏(few-shot) 그리고 혼합 프롬프트 전략을 평가했습니다. 특히, Llam 3.1 모델이 비네이티브 프롬프트에 대해 각각 7%와 8% 더 높은 성능을 보였습니다. GPT-4o는 모든 프롬프트 설정에서 가장 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 1. 피숏 프롬프트는 향상된 성능을 보여주었으며, 소량의 학습 데이터에 적합한 설정으로 추천됩니다. 2. 비네이티브 프롬프트는 모든 모델에서 가장 뛰어난 성능을 발휘했습니다. 3. 제로샷 설정은 새로운 태스크에 이상적인 솔루션으로, 모든 모델에서 비네이티브 프롬프트의 성능이 우수했습니다. 4. GPT-4o는 모든 프롬프트 설정에서 최상의 결과를 기록하였습니다.



### Beyond IID: Optimizing Instruction Learning from the Perspective of Instruction Interaction and Dependency (https://arxiv.org/abs/2409.07045)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에 대한 고품질 instruction 세트를 최적화하는 데 있어 기존 연구와는 다르게 서로 다른 instruction 카테고리 간의 상호작용 및 의존성 패턴을 시스템적으로 조사하였습니다.

- **Technical Details**: 이 연구에서는 다양한 instruction 카테고리 간의 상관 패턴을 분석하고, 이를 기반으로 instruction 의존성 분류 체계를 통해 instruction 세트를 최적화하는 선형 프로그래밍(linear programming) 기법을 사용하였습니다. 또한, 특성 태그 시스템을 구축하여 각 instruction에 필요한 능력 및 지식을 세분화하여 실험적으로 검증하였습니다.

- **Performance Highlights**: 다양한 LLM에서 수행된 실험 결과, 본 연구의 방법론이 기존 최첨단 기법에 비해 개선된 성능을 보임을 확인하였으며, 특히 reasoning 관련 및 상식 암기 과제에서 유의미한 성과를 도출했습니다. Qwen 및 Llama 모델에서 실험을 진행해 이들 벤치마크에서 우수한 성과를 나타냈습니다.



### E-commerce Webpage Recommendation Scheme Base on Semantic Mining and Neural Networks (https://arxiv.org/abs/2409.07033)
Comments:
          arXiv admin note: text overlap with arXiv:2409.01137

- **What's New**: 이번 논문에서는 전자상거래 웹사이트에서의 웹 페이지 추천 기술의 문제점을 해결하기 위해 의미 기반 웹 마이닝(semantic web mining)과 BP 신경망(BP neural networks)을 결합한 새로운 추천 솔루션을 제안합니다.

- **Technical Details**: 사용자의 웹 로그를 처리하고, 콘텐츠 우선순위(content priority), 시간 소비 우선순위(time consumption priority), 사용자 피드백의 명시적/암시적 피드백(explicit/implicit feedback), 추천 의미(recommendation semantics), 입력 편차(input deviation amount) 등 5가지 특징을 추출합니다. 이 특징들은 BP 신경망의 입력으로 사용되어 최종 출력 웹 페이지의 우선순위를 분류하고 식별합니다.

- **Performance Highlights**: 이 방법은 도서 판매 웹페이지를 샘플로 사용한 실험 결과, 사용자가 필요로 하는 웹 페이지를 빠르고 정확하게 식별할 수 있음을 보여주었습니다.



### Improving Anomalous Sound Detection via Low-Rank Adaptation Fine-Tuning of Pre-Trained Audio Models (https://arxiv.org/abs/2409.07016)
- **What's New**: 이 논문은 산업 환경에서 비정상 소리 탐지(ASD)에 효과적인 새로운 모델을 제안합니다. 특히 오디오 사전 학습 모델을 활용하여 기계 작동 데이터를 사용해 파인튜닝하고, 데이터 증강 전략으로 SpecAug를 사용합니다. 또한, 제한된 데이터로 문제를 해결하기 위해 LoRA(Low-Rank Adaptation) 튜닝을 조사합니다.

- **Technical Details**: 제안된 시스템은 두 가지 주요 구성 요소로 이루어져 있으며, 프론트 엔드는 오디오 신호 처리를 담당하고, 백엔드는 비정상 탐지기를 포함합니다. 프론트 엔드에서는 원본 오디오 신호의 전처리 후 오디오 모델을 통해 시맨틱 오디오 임베딩을 추출합니다. 비정상 탐지 단계에서는 종료된 오디오 임베딩을 처리하여 비정상 여부를 결정하고 비정상 점수를 출력합니다. 데이터 증강 방법으로는 SpecAug를 사용하며, Wav2Vec2 및 Qwen-Audio와 같은 사전 학습된 모델을 활용하여 성능을 최적화합니다.

- **Performance Highlights**: DCASE2023 Task 2 데이터셋에서 77.75%의 새로운 벤치마크를 수립하였으며, 이전의 SOTA 모델보다 6.48% 향상된 결과를 보여줍니다. 이는 오디오 사전 학습 모델과 LoRA 튜닝의 효과를 입증합니다.



### Towards Predicting Temporal Changes in a Patient's Chest X-ray Images based on Electronic Health Records (https://arxiv.org/abs/2409.07012)
- **What's New**: 이 논문에서는 의료 기록 (EHR) 데이터를 통합하여 미래의 흉부 X-레이 이미지 (CXR)를 예측하는 EHRXDiff라는 새로운 프레임워크를 제안합니다. 이 모델은 과거의 CXR 이미지와 후속 의료 이벤트를 기반으로 질병의 진행 상황을 동적으로 추적하고 예측합니다.

- **Technical Details**: 제안된 EHRXDiff 프레임워크는 이전의 CXR 이미지와 일련의 의료 이벤트 (e.g., 처방, 검사 결과 등)를 기반으로 목표 CXR 이미지를 예측합니다. 이 모형은 잠재 확산 모델 (latent diffusion model)을 사용하며, 과거의 이미지와 의료 사건 기록을 조합하여 환자의 상태 변화에 대한 실시간 정보를 제공합니다.

- **Performance Highlights**: EHRXDiff 모델은 임상적 일관성, 인구 통계적 일관성, 시각적 현실성을 포함한 세 가지 주요 측면에서 성능을 평가받았으며, 의료 업계에서 환자의 변화하는 상태를 추적하는 데 효과적인 시뮬레이션 도구로서의 잠재력을 보여주었습니다.



### What is the Right Notion of Distance between Predict-then-Optimize Tasks? (https://arxiv.org/abs/2409.06997)
- **What's New**: 이 연구는 전통적인 데이터셋 거리 개념이 Predict-then-Optimize (PtO) 맥락에서는 충분하지 않음을 보여줍니다. 새로운 데이터셋 거리를 제안하여, 이는 다운스트림 결정의 영향을 통합합니다.

- **Technical Details**: 제안된 결정 인식 데이터셋 거리는 Optimal Transport (OT) 기술을 기반으로 하여, 특징(feature), 레이블(label), 결정을 포함합니다. 이를 통해 PtO 태스크에서 성공적인 적응을 캡처합니다.

- **Performance Highlights**: 본 연구에서는 Linear Model Top-K, Warcraft Shortest Path, Inventory Stock Problem과 같은 세 가지 PtO 작업에서, 제안된 거리 척도가 특성-레이블 거리보다 전이 성능을 더 잘 예측함을 보여줍니다.



### Large Language Models and the Extended Church-Turing Thesis (https://arxiv.org/abs/2409.06978)
Comments:
          In Proceedings NCMA 2024, arXiv:2409.06120

- **What's New**: 최근 논문에서는 Extended Church-Turing Thesis (ECTT)가 현대의 대규모 언어 모델(LLMs)의 능력에 어떻게 적용되는지를 조사합니다. 연구는 LLM의 계산 능력을 전통적인 계산 가능성(computability) 및 계산 복잡도(complexity) 이론의 관점에서 검토합니다.

- **Technical Details**: 논문에서는 고정된(non-adaptive) LLM이 매우 큰 결정론적 유한 상태 변환기(deterministic finite-state transducer)와 계산적으로 동등하다는 것을 주장합니다. 또한, LLM이 공간 제약이 있는 Turing 기계를 시뮬레이션할 수 있음을 보여줍니다. 변화하는 LLM의 계보(lineages)는 조언을 가진 대화형 Turing 기계와 계산적으로 동등하다고 설명됩니다.

- **Performance Highlights**: 결과적으로, LLM의 계보는 초-터링(super-Turing) 계산 능력을 가지며, 지식 생성은 일반적으로 LLM의 계보에 의해 실현되는 비알고리즘(non-algorithmic) 과정으로 간주됩니다.



### Policy Filtration in RLHF to Fine-Tune LLM for Code Generation (https://arxiv.org/abs/2409.06957)
- **What's New**: 이번 논문에서는 Reinforcement Learning from Human Feedback (RLHF)의 한계를 극복하기 위한 새로운 접근법인 PF-PPO(Policy Filtration for Proximal Policy Optimization)를 제안합니다. 기존의 보상 모델의 부정확함을 해결하여 보다 신뢰할 수 있는 응답 생성을 목표로 합니다.

- **Technical Details**: PF-PPO는 보상 모델의 신뢰성을 이용하여 불확실한 샘플을 필터링하여 정책 학습 중 신호 대 잡음 비율(signal-to-noise ratio)을 개선합니다. 보상과 실제 점수 간의 결정 계수($R^2$)를 이용하여 적절한 정책 필터링 전략을 선택하고, 다양한 실험을 통해 PF-PPO의 효과성을 입증하였습니다.

- **Performance Highlights**: PF-PPO의 다양한 변형이 7억 개의 파라미터 모델에서 HumanEval, MBPP, 그리고 새로운 LeetCode Contest 벤치마크에서 최신 성능을 기록함을 발견했습니다. 이는 코드 생성 작업에서 상당한 개선을 보여줍니다.



### Neural Algorithmic Reasoning with Multiple Correct Solutions (https://arxiv.org/abs/2409.06953)
- **What's New**: 신경 알고리즘적 추론(Neural Algorithmic Reasoning, NAR)에 대한 최초의 다중 해법(multiple solutions) 접근법을 제안합니다. 이 방법은 Bellman-Ford(BF)와 Depth-First Search(DFS) 알고리즘에서 사용될 수 있습니다.

- **Technical Details**: 제안된 방법은 고전 알고리즘을 여러 번 실행하여 생성된 솔루션의 분포(distribution)에서 다중 솔루션을 추출하는 과정을 포함합니다. 이 방법은 Kullback-Leibler divergence를 최소화하여 신경망(NN)을 훈련시키고, 알고리즘 특성과 무작위성을 활용하여 여러 해를 추출합니다.

- **Performance Highlights**: 작은 그래프(n=5)에서는 정확도가 뛰어난 솔루션을 추출할 수 있었으나, 큰 그래프(n=64)에서는 해의 다양성이 높아졌습니다. 또한 제공된 방법은 Bellman-Ford 알고리즘보다 더 강력한 버전을 구현하는 첫 걸음을 내디뎠습니다.



### You Have Thirteen Hours in Which to Solve the Labyrinth: Enhancing AI Game Masters with Function Calling (https://arxiv.org/abs/2409.06949)
Comments:
          Wordplay Workshop @ ACL 2024

- **What's New**: 이 논문에서는 AI 게임 마스터의 일관성과 신뢰성을 높이기 위한 혁신적인 접근 방식을 제안합니다. 특히 'Jim Henson's Labyrinth: The Adventure Game'이라는 테이블 기반 롤플레잉 게임의 맥락에서 기능 호출(function calling)을 활용하여 AI 게임 마스터의 서사적 품질 및 상태 업데이트 일관성을 개선하는 방법을 설명하고 있습니다.

- **Technical Details**: 이 연구에서는 AI 게임 마스터를 향상시키기 위해 기능 호출을 통합하는 방법론을 제시합니다. 기능 호출을 통해 게임 특정 제어와 상태 관리를 가능하게 하여 AI 게임 마스터가 게임 규칙과 상태에 일관성 있는 서사를 생성할 수 있도록 합니다. 'Labyrinth' 게임의 시뮬레이션을 구현하고, 인간 평가 및 단위 테스트를 통해 이 접근 방식의 효과를 평가했습니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 게임 플레이 경험을 향상시키고, 게임 상태와의 일관성을 유지하는 데 효과적임을 입증했습니다. AI 게임 마스터의 설계에 대한 통찰력과 가이드라인을 제공하며, 게임 AI와 상호작용 스토리텔링의 발전에 기여합니다.



### FSMDet: Vision-guided feature diffusion for fully sparse 3D detector (https://arxiv.org/abs/2409.06945)
Comments:
          Accepted by European Conference on Computer Vision (ECCV) 2024 workshop on VCAD

- **What's New**: 이번 연구에서는 LiDAR(레이저 거리 측정기) 기술을 활용한 완전 희소(fully sparse) 3D 객체 감지의 효율성을 높이기 위해 시각 정보(visual information)를 활용한 FSMDet(Fully Sparse Multi-modal Detection) 모델을 제안합니다. 이 모델은 LiDAR의 feature diffusion 과정을 시각 정보로 안내하여 기존 방법보다 향상된 성능을 발휘합니다.

- **Technical Details**: FSMDet는 두 개의 주요 모듈인 Shape Recover Layer(SRLayer)와 Self Diffusion Layer(SDLayer)를 사용하여 RGB 정보를 활용해 객체의 가시 부분을 복구하고, 이후 특징을 중앙 영역으로 확산시킵니다. SRLayer는 RGB 기능을 사용하여 객체의 형태를 복구하고, SDLayer는 시각 정보를 기반으로 특징을 중앙으로 확산합니다. 이 과정은 기존의 복잡한 사용자 정의 중심 융합(diffusion) 또는 회귀(regression) 연산을 단순화합니다.

- **Performance Highlights**: 실험 결과, FSMDet는 이전의 LiDAR만 사용한 완전 희소 모델들에 비해 성능을 성공적으로 향상시키며, 다중 모달(multimodal) 모델에서도 SOTA(State Of The Art) 성능을 달성했습니다. 또한, 희소 아키텍처 덕분에 추론 과정에서 이전 SOTA 방법보다 최대 5배 더 효율적입니다.



### FreeRide: Harvesting Bubbles in Pipeline Parallelism (https://arxiv.org/abs/2409.06941)
- **What's New**: FreeRide는 파이프라인 병렬 처리에서 발생하는 버블을 활용하여 GPU 자원을 효율적으로 사용하는 시스템을 제안합니다. 이를 통해 프로그래머는 사이드 태스크를 쉽게 구현할 수 있는 인터페이스를 제공받게 됩니다.

- **Technical Details**: FreeRide는 상태 기계(state machine) 추상화를 기반으로 하는 프로그래밍 프레임워크를 통해 다양한 사이드 태스크를 저렴한 엔지니어링 노력으로 구현할 수 있도록 지원합니다. 추가적으로, GPU 자원 소비를 관리하기 위해 자동 사이드 태스크 프로파일러와 사이드 태스크 매니저를 사용하여 장애 조치와 관리 기능을 제공합니다.

- **Performance Highlights**: FreeRide는 사이드 태스크를 처리하면서 파이프라인 교육의 평균 비용을 7.8% 절감하며, 성능 오버헤드는 약 1.1%에 불과합니다. 이는 CUDA MPS를 직접 사용했을 때보다 훨씬 개선된 성능을 보이며, 혼합 사이드 태스크를 처리할 때에는 10.1%의 비용 절감을 달성합니다.



### Intrapartum Ultrasound Image Segmentation of Pubic Symphysis and Fetal Head Using Dual Student-Teacher Framework with CNN-ViT Collaborative Learning (https://arxiv.org/abs/2409.06928)
- **What's New**: 새로운 논문에서는 팟피코드 관절(Public Symphysis)과 태아 머리(Fetal Head) 세분화를 위한 혁신적인 프레임워크인 Dual-Student and Teacher Combining CNN and Transformer (DSTCT)를 소개합니다. 이는 CNN과 Transformer 모델의 장점을 통합하여 세분화 성능을 향상시킵니다.

- **Technical Details**: 이 프레임워크는 Vision Transformer (ViT)를 교사 모델로 하고, CNN 및 ViT 기반 학생 모델을 포함합니다. Dual-student 설정을 통해 서로의 예측을 통해 상호 감독을 수행하며, 하드 및 소프트 의사 레이블 생성에 집중하였습니다. Consistency regularization을 통해 학습을 보강하고, 데이터 및 모델 변형 기법을 사용하여 일반화 능력을 향상시킵니다.

- **Performance Highlights**: DSTCT 프레임워크는 MICCAI 2023의 PSFH 세분화 그랜드 챌린지의 벤치마크 데이터셋에서 기존의 10개 반지도 세분화 기법들보다 뛰어난 성능을 보였습니다.



### Interactive Counterfactual Exploration of Algorithmic Harms in Recommender Systems (https://arxiv.org/abs/2409.06916)
- **What's New**: 이번 연구에서는 사용자들이 추천 시스템에서의 알고리즘적 피해(algorithmic harms)를 이해하고 탐색할 수 있도록 돕는 대화형 도구(interactive tool)를 소개합니다. 이 도구는 시각화(visualizations), 반사적 설명(counterfactual explanations), 인터랙티브 모듈(interactive modules)을 활용하여 편향(bias) 및 필터 버블(filter bubble)의 영향을 조사할 수 있게 지원합니다.

- **Technical Details**: 이 연구는 사용자 인터뷰를 바탕으로 알고리즘적 피해를 탐색할 수 있는 대화형 대시보드(interactive dashboard)를 제안합니다. 주요 정보는 오차 보정(miscalibration), 고정관념(stereotype), 필터 버블 등 세 가지 알고리즘적 피해 유형으로 구분됩니다. 유저 친화적인 체험을 제공하기 위해, 시스템은 명확하고 이해하기 쉬운 설명을 제공하며, 사용자가 자신의 추천 결과를 사회적 맥락(social contextualization)에서 이해할 수 있도록 지원합니다.

- **Performance Highlights**: 이 도구는 사용자 및 연구자에게 투명성을 높이고 맞춤형 영향 평가를 제공하여 알고리즘적 편향에 대한 이해를 촉진합니다. 결국, 이 연구는 더 공정한 추천 결과를 지향하며, 향후 연구 및 실용적 응용에 중요한 통찰을 제공합니다.



### A Bayesian framework for active object recognition, pose estimation and shape transfer learning through touch (https://arxiv.org/abs/2409.06912)
- **What's New**: 이 연구에서는 베이즈(Bayesian) 틀 내에서 파티클 필터(Particle Filter)와 가우시안 프로세스 암묵적 표면(Gaussian Process Implicit Surface, GPIS)을 결합하여 물체 인식(object recognition), 포즈 추정(pose estimation), 그리고 새로운 물체의 형태 재구성(shape reconstruction)을 동시에 수행할 수 있는 통합 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 센서 데이터와 사전 지식을 통합하여 사물의 클래스와 포즈를 추정하는 베이즈 프레임워크를 사용합니다. 이 프레임워크는 최대 우도 추정(Maximum Likelihood Estimation, MLE)을 통해 알려진 물체의 형태에 대한 지식을 이전하여 새로운 형태를 학습합니다. 액티브 데이터 수집(active data acquisition)을 위한 탐색 절차가 제안되며, 이는 전역 형태 추정(global shape estimation)을 기반으로 합니다. 또한, 조사 종료 기준은 Directed Hausdorff Distance(DHD)를 사용하여 결정됩니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 탐색 절차는 RRT 기반의 지역 탐색(local exploration) 절차보다 더 빠른 탐색을 달성하는 것으로 나타났습니다. 제안된 프레임워크는 물체 인식, 포즈 추정 및 형태 재구성에서 효과적이고 효율적인 것으로 평가되었습니다. 또한, 학습한 형태는 새로운 사전으로 포함될 수 있으며, 이후 새로운 물체 인식 및 포즈 추정에 효과적으로 사용될 수 있습니다.



### Applied Federated Model Personalisation in the Industrial Domain: A Comparative Study (https://arxiv.org/abs/2409.06904)
- **What's New**: 본 연구에서는 기계학습 모델의 정확도를 개선하고 사용자 경험을 향상시키기 위해 여러 개인화(Personalisation) 방법을 활용한 고급 연합학습(Federated Learning) 시스템을 소개합니다. 특히, Active Learning, Knowledge Distillation 및 Local Memorization 방법을 통합하고, 이를 NG-IoT(Next-Generation Internet of Things) 애플리케이션에 적용하여 연구합니다.

- **Technical Details**: 연구에서 제안된 세 가지 전략인 Active Learning, Knowledge Distillation, Local Memorization은 모델을 소형화하고 개인화하여 처리 자원을 절약하는 것을 목표로 합니다. 연합학습은 여러 클라이언트가 중앙 서버의 조정 아래에서 협력하여 기계학습 문제를 해결하는 메커니즘으로, 각 고객의 데이터는 로컬에 저장되고 전송되지 않습니다. 연구는 다양한 딥러닝 모델(LSTM, Transformer, DNN, LR)을 분석하며, 각 기술의 장점과 한계를 심층적으로 조사합니다.

- **Performance Highlights**: 연구의 결과는 최적화 및 개인화된 모델이 연합학습 및 개인화 기법을 통해 성능이 향상되었다는 것을 보여주며, 실제 분산 데이터 세트에 대한 비교 분석을 통해 긍정적인 결과를 도출하였습니다. 개인화된 연합학습이 사용자 맞춤형 기능을 자원 제약이 있는 장치에서 효과적으로 수행할 수 있음을 나타냅니다.



### Formative Study for AI-assisted Data Visualization (https://arxiv.org/abs/2409.06892)
- **What's New**: 이 연구는 AI 지원 데이터 시각화에서 데이터 품질이 미치는 영향을 조사하며, 정리되지 않은 데이터셋이 이러한 도구의 결과에 미치는 영향을 분석합니다. 연구는 고품질 문제를 가진 데이터셋으로부터 생성된 시각화를 통해 발생하는 특정 시각화 문제를 식별하고 분류하는 데 중점을 둡니다.

- **Technical Details**: 연구는 세 가지 단계로 구성됩니다: 첫 번째 단계는 정리된 데이터셋에서 생성된 시각화를 분석하고, 두 번째 단계는 정리되지 않은 데이터셋을 사용하는 것을 포함합니다. 세 번째 단계에서는 특정 데이터 품질 문제를 정리된 데이터셋에 체계적으로 주입하여 시각화에 미치는 영향을 관찰합니다. 사용된 데이터셋은 Kaggle의 911 데이터셋과 GitHub의 Metropolitan Museum of Art Open Access 데이터셋입니다.

- **Performance Highlights**: AI 도구가 잘못된 데이터 처리 시 더 나은 시각화를 위해 개선되어야 한다는 점을 강조합니다. 또한, 데이터 품질이 시각화 결과에 미치는 영향을 이해함으로써 불완전한 데이터를 보다 잘 처리할 수 있는 도구 설계의 필요성을 제기합니다.



### A Dataset for Evaluating LLM-based Evaluation Functions for Research Question Extraction Task (https://arxiv.org/abs/2409.06883)
- **What's New**: 본 연구는 머신 러닝 분야의 연구 논문에서 연구 질문(RQ)을 자동으로 추출하는 새로운 데이터셋을 구축했습니다. 이 데이터셋은 GPT-4로 추출한 RQ와 다각적인 인간 평가를 포함합니다.

- **Technical Details**: 연구 문서에서 RQ를 추출하는 것은 기존의 문서 요약(task)과 연결됩니다. 연구초록 및 서론에서 LLM(대형 언어 모델)을 사용하여 RQ를 추출하고, 이를 기반으로 RQ의 품질을 평가하기 위해 인간 평가 점수를 수집하였습니다.

- **Performance Highlights**: 기존의 LLM 기반 평가 함수들이 인간 평가와 높은 상관관계를 보이지 않음을 발견했습니다. 이는 연구 질문 이해 과정의 복잡성을 고려할 때, 새로운 평가 기능의 설계가 필요함을 시사합니다.



### LIME-M: Less Is More for Evaluation of MLLMs (https://arxiv.org/abs/2409.06851)
- **What's New**: 최근 발표된 LIME-M benchmark는 기존의 Multimodal Large Language Models (MLLMs) 평가 방식의 한계를 극복하기 위해 개발되었습니다. 기존 평가 기준의 단순한 문제와 인공지능 모델 능력 구분의 부족함을 해결하고자, LIME-M은 두 가지 주요 모듈, 즉 Semi-Automated Screening Process와 Eliminating Answer Leakage로 구성됩니다.

- **Technical Details**: LIME-M benchmark는 9403개의 샘플을 포함하며, 6개 도메인에서 10개의 주요 작업을 수행합니다. 또한, MLLMs의 성능을 평가하기 위해 9개의 모델을 채택하여 각 샘플의 난이도를 분류하고, 쉽게 답할 수 있는 샘플과 답변 유출 문제가 있는 샘플을 제거합니다.

- **Performance Highlights**: LIME-M을 사용한 실험 결과, 기존 데이터의 24%만으로도 MLLMs 간 성능 차이를 더욱 명확하게 구분할 수 있으며, 총 소요 시간도 원본의 23%로 줄일 수 있었습니다. 특히, MLLMs는 시각적 정보에 직접 관련된 질문에 대한 성과가 우수하였으며, 추가적인 상식 지식이나 복잡한 추론을 포함하는 과제에서는 성과가 낮은 경향을 보였습니다.



### Bifurcation Identification for Ultrasound-driven Robotic Cannulation (https://arxiv.org/abs/2409.06817)
- **What's New**: BIFURC는 초음파 영상을 이용하여 혈관의 분기점을 자동으로 식별하는 알고리즘으로, 자율 로봇 카뉼레이션 시스템을 위한 최적의 바늘 삽입 위치를 제공합니다. 이는 기존의 알고리즘과 구별되며, 실제 데이터 기반의 훈련이 가능합니다.

- **Technical Details**: BIFURC는 깊이 학습 기법과 전문가 지식을 통합하여 제한된 양의 생체 데이터에서 혈관 분기점을 효율적으로 탐지합니다. 이 알고리즘은 페모랄(femoral) 지역 내에서 혈관의 구조를 3D 형태로 표현하고, 바늘 삽입 위치를 자동으로 결정하는 최초의 방법으로, 실제 실험에서 전문가와 비슷한 수준의 성능을 기록했습니다.

- **Performance Highlights**: BIFURC는 의료 팬텀 및 실제 생체 실험(예: 생돼지)을 통해 혈관 분기점과 바늘 삽입 위치를 일관되게 식별했습니다. 이 결과는 전문가 임상의와 일치하며, 대량 손상 상황에서도 빠르고 안전한 혈관 접근을 가능하게 합니다.



### Personalized Federated Learning Techniques: Empirical Analysis (https://arxiv.org/abs/2409.06805)
- **What's New**: 이 논문은 개인화된 연합 학습(Personalized Federated Learning, pFL)의 최적 성능을 추구하기 위해 메모리 오버헤드 비용과 모델 정확도 간의 균형을 탐구합니다. 다양한 실세계 시나리오에 적합한 알고리즘 선택을 위한 귀중한 통찰을 제공합니다.

- **Technical Details**: 본 연구에서는 10가지 주요 pFL 기법을 여러 데이터셋과 데이터 분할에서 실증적으로 평가하였으며, 개인화된(local) 집계를 활용하는 pFL 방법들이 통신 및 계산의 효율성 덕분에 가장 빠른 수렴(convergence)을 보인다는 결과를 발견했습니다. 반면에 세밀한 조정(fine-tuning) 방법은 데이터 이질성(data heterogeneity) 처리와 잠재적인 적대적 공격(adversarial attacks)을 다루는데 제한이 있습니다.

- **Performance Highlights**: 다중 목표 학습(multi-objective learning) 방법은 추가 교육(training) 및 자원 소비(resource consumption)의 대가로 높은 정확도를 달성하지만, 이러한 방법이 자원 사용에 미치는 영향을 고려할 때, 통신 효율의 중요성을 강조합니다. 논문은 실제 현장에서의 pFL 확장(scale)에서 통신 효율성이 자원 사용에 미치는 중대한 영향을 시연합니다.



### Adaptive Meta-Domain Transfer Learning (AMDTL): A Novel Approach for Knowledge Transfer in AI (https://arxiv.org/abs/2409.06800)
- **What's New**: 이 논문은 메타학습(meta-learning) 원칙과 도메인 특화(adaptations) 조정을 결합한 새로운 방법론인 Adaptive Meta-Domain Transfer Learning (AMDTL)을 소개합니다.

- **Technical Details**: AMDTL은 도메인 불일치(domain misalignment), 부정적 전이(negative transfer), 재앙적인 망각(catastrophic forgetting)의 문제를 해결하기 위해 하이브리드 프레임워크를 사용합니다. 이 프레임워크는 다양한 작업 분포에서 훈련된 메타러너(meta-learner), 도메인 특성 분포(domain feature distributions) 정렬을 위한 적대적 훈련(adversarial training) 기법, 그리고 맥락적 도메인 임베딩(contextual domain embeddings) 기반의 동적 특성 조절(dynamic feature regulation) 메커니즘을 통합합니다.

- **Performance Highlights**: 실험 결과, AMDTL은 정확도(accuracy), 적응 효율(adaptation efficiency), 강인성(robustness) 면에서 기존의 전이 학습 방법론보다 우수한 성능을 보였습니다.



### Modeling Image Tone Dichotomy with the Power Function (https://arxiv.org/abs/2409.06764)
Comments:
          49 pages, 11 figures and 36 references

- **What's New**: 본 논문에서는 이미지 조명 모델링에서의 이분법(dichotomy) 개념을 제시하고 새로운 수학적 모델을 제안합니다. 이 모델은 조명 이분법을 추상화할 수 있으며, 여러 수학적 특성을 검토하여 기존 모델의 한계를 식별합니다.

- **Technical Details**: 파워 함수(power function)의 수학적 속성을 활용하여 이미지 분석과 처리의 새로운 경로를 엽니다. 감마 보정(gamma correction) 및 비선형 변환의 개념을 도입하고, 감마 압축(gamma compression)과 감마 확장(gamma expansion)에 따른 이미지의 밝기 및 명도 개선 방법을 설명합니다.

- **Performance Highlights**: 본 모델은 기존의 최신 이미지 향상 방법과 비교하여 일관적으로 우수한 성능을 나타냅니다. 저조도(low-light) 이미지의 정보 추출을 위한 다양한 사례를 들어 이미지를 개선하는 데 효과적임을 보여줍니다.



### Generative Hierarchical Materials Search (https://arxiv.org/abs/2409.06762)
Comments:
this https URL

- **What's New**: 본 연구에서는 Generative Hierarchical Materials Search (GenMS)를 제안하여, 자연어로 입력된 고수준 지시사항에 기반하여 결정 구조를 생성하는 과정을 다중 목표 최적화 문제로 형식화하였습니다.

- **Technical Details**: GenMS는 (1) 고수준 자연어를 입력으로 받아 결정에 대한 중간 문자 정보를 생성하는 언어 모델과 (2) 중간 정보를 입력으로 받아 저수준 연속 값의 결정 구조를 생성하는 확산 모델로 구성됩니다. 또한, 결정 구조에서 물리적 특성(예: 형성 에너지)을 예측하기 위해 그래프 신경망(GNN)을 사용하였습니다.

- **Performance Highlights**: GenMS는 사용자 요청을 만족시키는 구조를 80% 이상 생성할 수 있으며, DFT 계산을 통해 낮은 형성 에너지를 가진 구조를 제안합니다. 이는 기존의 사전 훈련된 언어 모델에 비해 월등한 성능을 나타냅니다.



### Beyond designer's knowledge: Generating materials design hypotheses via large language models (https://arxiv.org/abs/2409.06756)
- **What's New**: 이번 연구는 대규모 언어 모델(large language models, LLMs)과 프롬프트 엔지니어링(prompt engineering)을 결합하여 비전문가의 도움을 받지 않고도 다학제적 과학 원칙을 통합하여 비非상식적인 자료 가설을 생성할 수 있다는 것을 보여줍니다.

- **Technical Details**: 이 연구는 처리-구조-속성 관계(processing-structure-property relationships)를 인코딩한 자료 시스템 차트를 활용하여 대량의 논문에서 핵심 정보를 압축하고, 수많은 가설을 평가하고 분류하는 데 LLM을 사용합니다.

- **Performance Highlights**: 2023년에 발표된 주요 논문을 통해 실험적으로 검증된 자료 설계 아이디어는 고엔트로피 합금(high-entropy alloys)의 우수한 저온 성질 및 향상된 이온 전도도와 성형성을 가진 할라이드 고체 전해질(halide solid electrolytes)을 포함합니다.



### Scaling Law Hypothesis for Multimodal Mod (https://arxiv.org/abs/2409.06754)
- **What's New**: 이번 연구에서는 텍스트, 오디오, 이미지, 비디오를 처리하는 다중 모달 모델을 위한 스케일링 법칙 가설을 제안합니다. 기존의 텍스트 기반 디코더 모델을 혼합 모달 시스템으로 확장하여 모델 성능을 예측합니다.

- **Technical Details**: 우리의 프레임워크는 모달리티별 압축 및 토큰화 효율에 기반하여 모델 성능을 예측하며, 다양한 모달 데이터 유형의 특성을 고려합니다. 압축 계수 C와 생성된 토큰 수 N의 관계를 통해 성능을 수식화하였고, 각 모달리티의 특징에 맞는 스케일링 법칙을 연구했습니다.

- **Performance Highlights**: 다중 모달 모델은 특정 모달리티의 압축 효율성을 포함하여 총 원시 데이터 양에 따라 성능이 달라진다고 강조하며, 효율적인 다중 모달 모델을 통해 자원 제약이 있는 환경에서도 성능 손실 없이 모델 크기를 줄일 수 있을 것으로 기대합니다.



### Can Agents Spontaneously Form a Society? Introducing a Novel Architecture for Generative Multi-Agents to Elicit Social Emergenc (https://arxiv.org/abs/2409.06750)
Comments:
          13 pages, 8 figures

- **What's New**: 본 논문에서는 ITCMA-S라는 새로운 생성 에이전트 아키텍처를 소개합니다. 이 아키텍처는 개인 에이전트를 위한 기본 프레임워크와 다중 에이전트 간의 사회적 상호작용을 지원하는 LTRHA 프레임워크를 포함하고 있으며, 이는 에이전트가 사회적 상호작용을 향상시키는 행동을 선택하도록 유도합니다.

- **Technical Details**: ITCMA-S 아키텍처는 개인 생성 에이전트 구조와 사회적 협력 프레임워크인 LTRHA를 포함합니다. LTRHA는 지역 & 주제(locale & topic), 자원(resources), 습관(habitus), 행동(action)의 네 가지 모듈로 구성되어 있으며, 에이전트가 사회적 상호작용을 통해 새로운 관계를 형성할 수 있도록 설계되었습니다. 실험 평가를 위해 IrollanValley라는 샌드박스 환경이 구축되었습니다.

- **Performance Highlights**: ITCMA-S는 여러 평가 지표에서 우수한 성과를 보였으며, 에이전트들은 환경을 능동적으로 탐색하고 새로운 에이전트를 인식하며 대화를 통해 새로운 정보를 습득하는 능력을 입증했습니다. 에이전트들은 자발적으로 리더를 중심으로 클리크를 형성하고 집단 활동을 조직하는 행동을 보였습니다.



### EasyST: A Simple Framework for Spatio-Temporal Prediction (https://arxiv.org/abs/2409.06748)
Comments:
          Accepted by CIKM'2024, full paper

- **What's New**: 본 연구에서는 도시 컴퓨팅에 대한 스파이시오-템포럴(spatio-temporal) 예측 문제를 해결하기 위해 EasyST라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 복잡한 GNN의 지식을 경량의 MLP로 효과적으로 증류하여 일반화 성능과 효율성을 향상시킵니다.

- **Technical Details**: EasyST는 spatio-temporal 정보 병목 원리를 통합한 지식 증류(knowledge distillation) 프레임워크를 사용하여 강건한 모델을 생성합니다. 또한, spatial과 temporal prompts를 포함하여 다운스트림 작업에 대한 맥락을 제공합니다.

- **Performance Highlights**: 세 가지 spatio-temporal 데이터셋을 평가한 결과, EasyST는 기존 최첨단 방식보다 정확성과 효율성 모두에서 우수한 성능을 보였습니다.



### Personalized Knowledge Tracing through Student Representation Reconstruction and Class Imbalance Mitigation (https://arxiv.org/abs/2409.06745)
- **What's New**: 이번 연구에서는 개인 맞춤형 지식 추적(personalized knowledge tracing)을 위한 새로운 접근 방식인 PKT를 제안합니다. PKT는 이전의 상호작용 기록을 활용하여 학생을 재구성하고, 이로 인해 개인화된 평가를 가능하게 합니다.

- **Technical Details**: PKT는 학생과 교육 플랫폼 간의 상호작용 서열을 기반으로 학생에 대한 잠재 정보를 포착하기 위해 데이터 표현을 재구성합니다. 이 모델은 focal loss를 사용하여 소수 클래스를 우선시하며, 이를 통해 예측의 균형을 이루어 class imbalance 문제를 해결합니다.

- **Performance Highlights**: PKT는 16개의 최신 지식 추적 모델과 비교하여 4개의 공개 교육 데이터 세트에서 우수한 예측 성능을 입증했습니다. 연구 결과는 모델의 파라미터에 대한 정량적 및 정성적 분석을 포함하며, 이를 시각화하기도 하였습니다.



### ProteinBench: A Holistic Evaluation of Protein Foundation Models (https://arxiv.org/abs/2409.06744)
Comments:
          29 pages, 1 figure and 11 tables

- **What's New**: 이 연구에서는 단백질 기초 모델의 성능을 전반적으로 평가하기 위한 포괄적인 평가 프레임워크인 ProteinBench를 소개합니다. 이를 통해 단백질 모델의 투명성을 증대시키고 연구를 촉진하고자 합니다.

- **Technical Details**: ProteinBench는 1) 단백질 분야의 주요 과제를 포괄하는 분류 체계, 2) 품질, 참신성, 다양성, 강건성을 중심으로 한 다차원 평가 방법, 3) 다양한 사용자 목표에 대한 심층 분석으로 구성됩니다.

- **Performance Highlights**: ProteinBench의 포괄적인 평가를 통해 단백질 기초 모델의 현재 능력과 한계가 밝혀졌습니다. 공개된 평가 데이터셋과 코드, 리더보드를 통해 추가 연구와 협업을 촉진할 것입니다.



### Generative AI for Requirements Engineering: A Systematic Literature Review (https://arxiv.org/abs/2409.06741)
- **What's New**: 최근 Generative AI (GenAI)는 소프트웨어 엔지니어링의 요구 사항 엔지니어링 (RE) 분야에서 혁신적인 도구로 부상하고 있습니다. 본 논문은 GenAI를 통합한 RE의 최신 응용과 혁신적인 제안을 종합적으로 검토합니다.

- **Technical Details**: 이 논문에서는 27개의 주요 연구를 심층적으로 분석하기 위해 체계적인 문헌 검토 (SLR) 방법론을 사용했습니다. 연구 질문들은 GenAI가 다양한 RE 단계에서 어떻게 적용되는지를 중점적으로 다루며, 사용된 모델과 기술 및 구현 시 발생하는 도전 과제를 식별합니다.

- **Performance Highlights**: 주요 발견 사항은 i) RE의 초기 단계에 대한 집중, ii) GPT 시리즈와 같은 대형 언어 모델의 지배, iii) 도메인별 응용 및 AI 생성 결과의 해석 가능성에 대한 지속적인 도전 과제입니다. 이러한 결과는 GenAI 지원 RE의 윤리적 고려와 협업 모델 개선을 위한 연구의 필요성을 강조합니다.



### How will advanced AI systems impact democracy? (https://arxiv.org/abs/2409.06729)
Comments:
          25 pages

- **What's New**: 이 논문은 생성적 인공지능(generative artificial intelligence)이 민주적 과정에 미치는 영향을 다룹니다. 고급 AI 시스템이 인간과 유사한 텍스트와 멀티모달 콘텐츠를 생성할 수 있는 시대에 접어든 가운데, 민주주의에 대한 AI의 잠재적 위협과 기회를 탐구합니다.

- **Technical Details**: 논문은 AI의 세 가지 영향인 지식적 영향(epistemic impacts), 물질적 영향(material impacts), 그리고 근본적 영향(foundational impacts)을 중심으로 논의합니다. AI가 시민들이 정치적 선택을 하는 데 있어 어떻게 영향을 미치는지를 검토하고, 선거와 같은 민주적 기제를 어떻게 destabilise하거나 지원할 수 있는지를 분석합니다.

- **Performance Highlights**: AI가 민주주의 원칙을 강화할 것인지, 약화할 것인지를 고민하며, 교육 기회와 공적 담론 강화를 통해 민주주의가 더 잘 작동하는 방법을 재구상할 수 있는 가능성을 제시합니다.



### Feedback-based Modal Mutual Search for Attacking Vision-Language Pre-training Models (https://arxiv.org/abs/2409.06726)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 Feedback-based Modal Mutual Search (FMMS)라는 새로운 공격 패러다임을 제안합니다. 이는 적대적 예제를 생성하기 위해 대상 모델의 피드백을 이용하는 최초의 방법으로, 멀티모달 환경에서의 적대 경계를 탐색하는 데 중점을 두고 있습니다.

- **Technical Details**: FMMS는 Modal Mutual Loss (MML)라는 새로운 손실 함수를 도입하여, 일치하는 이미지-텍스트 쌍 간의 거리를 멀게 하고 불일치 쌍 간의 거리를 가깝게 유지합니다. 이를 통해 적대적 예제를 점진적으로 개선하며, 두 가지 탐색 전략 (Full 및 Top-N)을 활용하여 더욱 эффектив한 적대적 예제를 생성합니다.

- **Performance Highlights**: Flickr30K 및 MSCOCO 데이터셋에서의 실험 결과, FMMS는 기존의 최첨단 기법들보다 적대적 예제를 생성하는 데 있어 상당히 우수한 성능을 보였습니다. 특히, 다양한 VLP 모델 간의 효과적인 적대적 전이 능력을 입증했습니다.



### Elementary School Students' and Teachers' Perceptions Towards Creative Mathematical Writing with Generative AI (https://arxiv.org/abs/2409.06723)
- **What's New**: 본 연구는 초등학생들이 창의적인 수학 글쓰기를 지원하는 GenAI 기술의 수용도를 탐구합니다. 특히, GenAI가 제공하는 스토리 생성 기능을 통해 학생들이 수학적 아이디어를 창의적으로 표현하는 데 도움을 줄 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 연구는 질적 주제 분석(qualitative thematic analysis)을 사용하였으며, 79명의 초등학생을 대상으로 인터뷰와 오픈 엔디드(open-ended) 설문 응답, 교실 관찰을 삼각 측량(triangulated)하여 여섯 가지 주제와 19개의 하위 주제를 도출하였습니다.

- **Performance Highlights**: 이 연구는 GenAI가 지원하는 학습 경험을 조사하고, GenAI 기반 학습 기술 및 교육 설계에 대한 고려사항을 제시하여 교육 현장에서의 기술 활용 가능성을 높이는 데 기여합니다.



### Students' Perceived Roles, Opportunities, and Challenges of a Generative AI-powered Teachable Agent: A Case of Middle School Math Class (https://arxiv.org/abs/2409.06721)
- **What's New**: 이번 연구는 Generative AI (GenAI)가 기반이 된 teachable agent (TA)가 중학생의 수학 학습에 미치는 영향과 학생들이 인식하는 TA의 역할을 다룹니다.

- **Technical Details**: 중학생 108명을 대상으로 한 수업 관찰, 포커스 그룹 인터뷰 및 개방형 설문조사를 통해 GenAI 기반 TA의 기대 역할과 학생들이 느끼는 혜택 및 도전과제를 조사했습니다.

- **Performance Highlights**: 학생들은 GenAI 기반 TA를 학습 동반자, facilitator 및 협력적인 문제 해결자로 인식하고, 이에 따른 다양한 혜택과 도전 과제를 언급했습니다. 이 연구는 교육 AI 및 AI 보조 교육의 설계에 대한 시사점을 제공합니다.



### Evolutionary Game Dynamics Applied to Strategic Adoption of Immersive Technologies in Cultural Heritage and Tourism (https://arxiv.org/abs/2409.06720)
- **What's New**: 이 논문은 몰입형 기술(Immersive Technologies), 특히 VR(가상 현실)과 AR(증강 현실)이 문화 및 관광 산업에서의 채택과 통합에 미치는 영향에 대해 다루고 있습니다. 이해관계자들(stakeholders)의 인식을 분석하여 기술의 채택 속도와 범위를 결정짓는 주요 요소로 작용한다는 점이 강조됩니다.

- **Technical Details**: 이 연구는 Q-방법론(Q-methodology)을 활용하여 이해관계자들의 인식을 주요 구성 요소로 분해하고, 진화 게임 모델(evolutionary game model)을 적용하여 가능한 시나리오를 지도화하고 의사 결정 경로를 강조합니다. 이론적으로, 게임 이론(game theory)을 기반으로 이해관계자 간의 상호작용을 모델링하여 장기 전략을 파악하는 방식으로 접근합니다.

- **Performance Highlights**: 몰입형 기술은 문화 유산 보존 및 관광 산업 증진의 미래 경로를 결정짓는데 중대한 역할을 할 것입니다. 특히 문화 기관들은 AR과 VR을 통해 관람객 참여와 교육을 증진할 수 있으며, 관광 산업에서는 가상 투어를 제공하여 독특한 경험을 창출할 수 있습니다. 그러나 이러한 기술 통합에는 높은 비용, 기술적 복잡성, 접근성 문제 등의 도전 과제가 존재합니다.



### Unveiling Visual Biases in Audio-Visual Localization Benchmarks (https://arxiv.org/abs/2409.06709)
Comments:
          Accepted by ECCV24 AVGenL Workshop

- **What's New**: 본 논문은 기존 Audio-Visual Source Localization (AVSL) 벤치마크에서 시각적 편향(visual bias)이라는 중요한 문제를 파악합니다. 특히 시각적 정보만으로 소리나는 객체(source objects)를 쉽게 인식할 수 있는 경우가 많은데, 이는 AVSL 모델을 효과적으로 평가하는데 방해가 됩니다.

- **Technical Details**: 저자들은 VGG-SS와 Epic-Sounding-Object라는 두 가지 대표적인 AVSL 벤치마크를 검토하며, `vision-only models`(시각적 정보만 사용하는 모델)인 MiniGPT-v2를 사용하여 AVSL 작업에서의 성능을 검증합니다. 주목할 점은 이러한 모델이 오디오 정보를 사용하지 않고도 기존 AVSL 모델들을 능가한다는 것입니다.

- **Performance Highlights**: 테스트 결과, MiniGPT-v2는 VGG-SS 벤치마크에서 모든 ASVL(base model)들을 능가하였으며, 이는 오디오 정보 없이도 경쟁력 있는 성능을 보여준다는 점에서 시각적 편향의 존재를 더욱 확인시킵니다.



### Ensuring Fairness with Transparent Auditing of Quantitative Bias in AI Systems (https://arxiv.org/abs/2409.06708)
- **What's New**: AI 시스템 성능 향상과 함께 AI의 공정성을 평가하기 위한 새로운 감사 프레임워크와 오픈소스 도구가 제안되었습니다. 기존의 불공정 사례인 COMPAS 시스템과 같은 기존 사례를 분석하여 AI의 공정성을 측정하는 통계적 지표가 필요함을 강조합니다.

- **Technical Details**: 제안된 프레임워크는 제3자 감사인(third-party auditors)과 AI 시스템 제공자(AI system providers)이 함께 공정성을 평가하는 것입니다. 사용된 측정 기준으로는 disparate impact, demographic parity 및 equalized odds 등의 통계적 지표들이 있습니다. 오픈소스 도구는 Python으로 작성되어 CSV와 같은 일반적인 데이터셋 형식을 지원합니다.

- **Performance Highlights**: 제안된 도구는 공정성과 편향 검토를 위한 통계적 지표의 체계적인 접근을 제공합니다. COMPAS 시스템을 검토한 결과, equalized odds, conditional statistical parity 및 mean difference 등의 기준을 위반하는 것으로 나타났습니다. 이 도구는 감사인, AI 개발자 및 일반 대중이 AI 시스템의 공정성을 판단할 때 참조할 수 있도록 설계되었습니다.



### Discovering Long-Term Effects on Parameter Efficient Fine-tuning (https://arxiv.org/abs/2409.06706)
- **What's New**: 이 논문에서는 파라미터 효율적인 미세 조정(Parameter-efficient Fine-tuning, PEFT) 기술을 활용하여 인공신경망(ANN)과 생물학적 신경망(BNN) 간의 유사성을 바탕으로 새로운 지식을 효과적으로 습득하는 방법을 제안합니다. 특히, Synapses & Neurons (SAN)이라는 새로운 방법이 도입되어 피처 조정과 파라미터 조정을 연결하여 성능을 크게 향상시킵니다.

- **Technical Details**: SAN 방법은 각 레이어에서 피처를 스케일링하는 방식을 통해 전신경접합부(Heterosynaptic Plasticity)의 효과를 모델링합니다. 연관된 신경과학 현상인 단기적 뇌 가소성(Long-term Potentiation, LTP)와 단기적 감소(Long-term Depression, LTD)의 개념을 통해 레이어 간의 피처 조정이 서로에게 미치는 영향을 명시적으로 전파합니다. 기존의 PEFT 방식은 이를 간접적으로 구현했지만, SAN은 파라미터 조정을 보다 세밀하게 하고 과적합을 방지하는 정규화 능력을 갖추고 있습니다.

- **Performance Highlights**: 26개의 데이터셋에서의 실험 결과, SAN은 기존의 미세조정 방법들에 비해 효율적으로 성능을 향상시켰습니다. 상대적으로 완전 미세조정(+8.5%), 비주얼 프롬프트 튜닝(+7%), LoRA(+3.2%)에 비해 뛰어난 성과를 보여주었습니다. 이는 SAN이 고도로 효율적인 파라미터 조정을 가능하게 함을 의미합니다.



### A Normative Framework for Benchmarking Consumer Fairness in Large Language Model Recommender System (https://arxiv.org/abs/2405.02219)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용한 추천 시스템(RecLLMs)의 소비자 공정성을 평가하기 위한 규범적 프레임워크를 제안합니다. 전통적인 추천 시스템의 공정성 평가가 LLMs의 복잡성을 충분히 반영하지 못한다는 점을 비판하며, RecLLMs에서의 공정성 평가 방법에 대해 체계적이고 형식적인 접근을 강조합니다.

- **Technical Details**: RecLLMs에서는 성별과 같은 민감한 인구 통계 정보를 자연어(natural language) 사용자 프로필에서 직접 포함할 수 있는 가능성이 있습니다. 이 연구에서는 민감한 순위 모델(sensitive ranker)과 중립 순위 모델(neutral ranker)의 차이에 기반하여 공정성을 평가하는 CFairLLM 프레임워크를 제안합니다. 또한, 공정성 평가 메트릭으로는 NSD(Neutral vs. Sensitive Ranker Deviation), NCSD(Neutral vs. Counterfactual Sensitive Deviation), IF(Intrinsic Fairness)를 도입합니다.

- **Performance Highlights**: MovieLens 데이터셋을 사용한 실험에서 연령 기반 추천에서 공정성의 편차가 확인되었으며, 통계적 유의성 검증을 통해 이러한 편차가 임의적이지 않음을 입증했습니다. 이 결과는 RecLLMs의 공정성 평가에 대한 강화된 분석 방법론의 필요성을 강조합니다.



### A System and Benchmark for LLM-based Q&A on Heterogeneous Data (https://arxiv.org/abs/2409.05735)
- **What's New**: 본 논문에서는 siwarex 플랫폼을 도입하여 데이터 소스의 이질성 문제를 해결하고, 자연어 질문에 대한 무알려 데이터베이스와 API에 대한 원활한 접근을 가능하게 한다.

- **Technical Details**: Siwarex는 다양한 DB와 API를 결합하여 질문-응답 시스템을 지원하며, 대규모 언어 모델(LLM)을 활용하여 자연어 질문의 이해와 SQL 문 생성 기능을 제공한다. 또한, API를 사용자 정의 함수(UDF)로 변환하여 데이터베이스 시스템에서 API를 직접 호출할 수 있도록 한다.

- **Performance Highlights**: Siwarex는 수정된 Spider 벤치마크에서 Q&A 정확성을 측정했으며, 데이터베이스 호출과 API 호출의 비율에 따른 성능을 평가하여 데이터 출처의 이질성에 잘 대처함을 입증했다. 이 연구에서 개발한 벤치마크는 연구 커뮤니티와 공유될 예정이다.




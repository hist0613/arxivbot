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




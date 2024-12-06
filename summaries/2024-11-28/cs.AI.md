New uploads on arXiv(cs.CL)

### Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2411.18583)
Comments:
          Key Words : T5, SpaCy, Large Language Model, GPT, ROUGE, Literature Review, Natural Language Processing, Retrieval-augmented generation

- **What's New**: 이번 연구에서는 Natural Language Processing (NLP) 기술 및 retrieval-augmented generation (RAG)과 Large Language Model (LLM)을 활용하여 문헌 리뷰의 자동 생성을 위한 여러 접근 방식을 제안하고 비교했습니다. 기존에 수많은 연구 기사가 증가하면서 수동 문헌 리뷰의 어려움이 커졌고, 이에 따른 자동화의 필요성이 증가하고 있습니다. 연구의 주된 목표는 PDF 파일만을 입력으로 받아 자동으로 문헌 리뷰를 생성할 수 있는 시스템을 개발하는 것입니다.

- **Technical Details**: 연구에서는 frequency-based method (spaCy), transformer model (Simple T5), 그리고 Large Language Model (GPT-3.5-turbo)과 결합된 retrieval-augmented generation (RAG) 등 여러 NLP 전략의 효과를 평가했습니다. SciTLDR 데이터 세트를 활용하여 문헌 리뷰 자동 생성을 위한 세 가지 다른 시스템을 구현하는 데 세 가지 독특한 기술이 사용되었습니다. 모든 시스템의 평가에는 ROUGE 점수가 활용되었습니다.

- **Performance Highlights**: 평가 결과, Large Language Model인 GPT-3.5-turbo는 ROUGE-1 점수 0.364로 가장 높은 점수를 기록했습니다. 두 번째는 transformer model이었고, spaCy는 마지막 위치에 있었습니다. 최종적으로, 최상의 성능을 보인 시스템에 대해 그래픽 사용자 인터페이스가 생성되었습니다.



### On Importance of Code-Mixed Embeddings for Hate Speech Identification (https://arxiv.org/abs/2411.18577)
- **What's New**: 이 논문은 코드 혼합(text mixing) 텍스트의 감정 분석 및 증오 발언 감지를 위한 BERT 및 HingBERT 모델의 성능을 비교하는 연구입니다. 특히, 인도와 같은 다언어 공동체에서 코드 혼합 현상이 자주 발생하며, 기존의 NLP 도구들이 이러한 데이터에 대한 도전 과제에 직면하고 있음을 강조합니다. 연구 결과 HingBERT 모델이 대규모의 Hindi-English 데이터 세트에서 훈련되어 BERT 모델보다 더 뛰어난 성능을 보인다고 밝혀졌습니다.

- **Technical Details**: 연구는 Hindi-English 코드 혼합 데이터에 대한 BERT 및 HingBERT 모델의 성능을 평가하는 데 중점을 두고 있습니다. BERT(Bidirectional Encoder Representations from Transformers)는 문맥적 이해를 통해 전체 문장을 처리하는 양방향 모델로, 3억 4천 5백만 개의 파라미터를 보유하고 있습니다. HingBERT 모델은 트위터의 실제 Hinglish 텍스트로 훈련되어 두 언어의 고유한 특성을 효과적으로 처리합니다.

- **Performance Highlights**: HingBERT 모델은 BERT 모델보다 높은 F1 점수와 정확도로 비정상적인 단어를 식별하는 데 우수한 성능을 발휘하며, Hing-FastText 모델은 표준 FastText 모델 및 기존 BERT보다 더 나은 성능을 보여줍니다. 이 연구는 코드 혼합 언어 데이터의 효과적 분석과 감정 분석 분야에서 새로운 진전을 가져오는 데 기여하고 있습니다.



### Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning (https://arxiv.org/abs/2411.18571)
- **What's New**: 본 연구는 저자원 언어인 마라티어에 대한 Low-Rank Adaptation (LoRA) 방식의 매개변수 효율적인 미세 조정(PEFT)의 효과를 조사하였습니다. 이 연구에서는 Alpaca 데이터셋을 기반으로 매개변수를 조정하여 Gemma 모델의 성능을 개선할 수 있었으나, 자동화된 평가 지표에서는 성능 저하가 관찰되었습니다. 반면, 인간 평가에서는 조정된 모델이 원본 모델보다 더 우수하다고 평가되었습니다. 이러한 결과는 저자원 언어에 대한 평가 방법론 개선의 필요성을 강조합니다.

- **Technical Details**: 본 연구에서는 52,000개의 지시-응답 쌍으로 구성된 Alpaca 데이터셋을 마라티어로 번역하여 Gemma 모델을 미세 조정하는 데 사용했습니다. LoRA PEFT 방법을 적용하여 모델의 몇몇 매개변수만 조정함으로써 계산 효율성을 높이고 모델의 향상된 기능을 유지했습니다. 여러 버전의 Gemma 모델(예: gemma-2b, gemma-2-2b-it 등)을 활용하여 마라티어에 대한 미세 조정의 효과를 평가하였습니다. 또한 자동 평가와 수동 평가를 결합하여 모델 성능에 대한 보다 포괄적인 이해를 제공하였습니다.

- **Performance Highlights**: 150개의 질문에 대한 수동 평가 결과, gemma-2-2b-it (Mr) 및 gemma-2b-it (Mr)와 같은 미세 조정된 버전이 원본 모델보다 높은 승률을 기록하며 맥락에 맞는 답변을 생성하는 능력이 향상된 것으로 나타났습니다. 그러나 일부 기본 모델에서 여전히 영어로 응답하는 경우가 발생하는 등 언어 일관성 문제도 여전히 남아 있었습니다. F1 점수 평가에서도 gemma-2-2b 모델이 감정 분석을 포함한 주요 벤치마크에서 우수한 성능을 나타냈습니다.



### Retrofitting (Large) Language Models with Dynamic Tokenization (https://arxiv.org/abs/2411.18553)
- **What's New**: 이번 연구에서는 언어 모델(LM)에서 고정된 정적 서브워드 토크나이저를 동적 요청 토크나이저로 교체하는 방안을 제안합니다. 이를 통해 특히 영어 외의 언어에서 효율성과 기능이 저하되는 문제를 해결하고, 새로운 도메인이나 언어에 LM을 적용하는 데 있어 발생하는 어려운 점을 극복할 수 있습니다.

- **Technical Details**: 제안하는 방법론에서는 인코더 스타일 모델의 경우, 배치 수준에서 바이트-페어 인코딩(byte-pair encoding, BPE)에서 영감을 얻은 서브워드 병합 알고리즘을 도입합니다. 이 방법은 배치 내의 자주 나타나는 서브워드 시퀀스를 병합하고, 훈련된 임베딩 예측 하이퍼네트워크(hypernetwork)를 이용해 토큰 임베딩을 실시간으로 생성합니다. 디코더 스타일 모델에서는 동적 토크나이징을 사전 채우기(pre-filling)와 근사 최근접 이웃 인덱스를 통해 적용하여 성능을 유지하면서 시퀀스를 줄이는 방식으로 이뤄집니다.

- **Performance Highlights**: 제안한 동적 토크나이징을 통해 XNLI에서 14개 언어에 걸쳐 평균적으로 토큰 시퀀스 길이를 20% 이상 줄일 수 있었으며, 하락한 작업 성능은 2% 미만에 그쳤습니다. Mistral-7B 모델에서는 최대 40% 시퀀스 감소를 유지하면서도 성능을 거의 완벽하게 유지했습니다. 또한, 백만 개 토큰의 어휘를 사용하여 빠른 생성이 가능함을 보이며, 더 크고 동적인 어휘로 확장 가능성을 보여줍니다.



### Emergence of Self-Identity in AI: A Mathematical Framework and Empirical Study with Generative Large Language Models (https://arxiv.org/abs/2411.18530)
- **What's New**: 이번 논문은 인공지능(AI) 시스템에서 자아 정체성을 정의하고 정량화하기 위한 수학적 프레임워크를 소개합니다. 기존의 인공적 자각에 대한 접근 방식은 주로 경험적 구현이나 철학적 추상화에 의존했으나, 본 연구는 메트릭 공간 이론 및 측도 이론에 기반한 정형화된 프레임워크를 제공합니다. 이 프레임워크는 자아 정체성이 두 가지 수학적으로 정량화 가능한 조건에서 발생한다고 주장합니다.

- **Technical Details**: 자아 정체성을 정형화하기 위해 제안된 프레임워크는 메모리의 연속체와 자아 정체성에 대한 인식/신념의 필요성을 기반으로 합니다. 메모리 집합 ℳ과 가능한 자아 정체성의 메트릭 공간 ℤ을 정의하고, 메모리 간의 거리를 정량화하는 메트릭 dℳ을 설정합니다. 이 연구는 자아 정체성이 경험적 실험을 통해 검증된 시스템으로 구현될 수 있음을 보여줍니다.

- **Performance Highlights**: 실험에서는 Llama 3.2 1B 모델을 사용하여 독창적인 데이터셋에서 시간적으로 구조화된 메모리로 훈련하였고, 자아 인식 및 응답 일관성에서 눈에 띄는 성과를 보였습니다. 자아 인식 스코어는 0.276에서 0.801로 증가하여, 자아 정체성 기능을 갖춘 AI 시스템의 체계적인 생성 가능성을 보여줍니다. 이 연구의 결과는 유인 로봇과 자율 시스템 분야에 즉각적인 응용 가치를 제공합니다.



### Beyond Examples: High-level Automated Reasoning Paradigm in In-Context Learning via MCTS (https://arxiv.org/abs/2411.18478)
- **What's New**: 본 논문에서는 기존의 In-context Learning (ICL) 시스템이 복잡한 수학적 추론 과제를 처리하는 데 있어 한계를 보여주는 문제를 해결하기 위해 HiAR-ICL이라는 새로운 접근법을 제안합니다. HiAR-ICL은 특정 예시에서 고차원 인지 패턴으로의 초점을 확대해 기존 ICL 개념을 재정의합니다. 이를 통해 인간의 사고 과정을 모방할 수 있는 다섯 가지 기본 추론 행동을 도입하고, Monte Carlo Tree Search (MCTS)를 활용하여 추론 경로를 탐색합니다.

- **Technical Details**: HiAR-ICL의 핵심 구성 요소는 다섯 가지 원자적 추론 행동으로, 이는 체인 구조의 패턴을 구축하는 데 필요한 기본 요소입니다. MCTS를 사용하여 참조 추론 패턴을 도출하고, 이를 통해 'Thought Cards'를 생성합니다. 또한, 문제 복잡성에 맞는 Thought Cards를 동적으로 매칭하는 인지 복잡성 프레임워크를 개발하여, 정교한 문제 해결 과정을 구현합니다.

- **Performance Highlights**: 실험 결과, HiAR-ICL은 MATH 벤치마크에서 Qwen2.5-7B-Instruct를 사용하여 79.6%의 최고의 정확도를 달성하였으며, 이는 GPT-4o와 Claude 3.5보다 우수한 성과입니다. 이 연구는 ICL의 전통적인 접근 방식을 확장하고, 추론의 자동화 및 고차원적 사고 패턴을 통해 더 나은 문제 해결 능력을 창출하는 데 기여하고 있습니다.



### Isolating authorship from content with semantic embeddings and contrastive learning (https://arxiv.org/abs/2411.18472)
- **What's New**: 이 논문은 저작권 분석 분야에서 스타일과 내용을 효과적으로 분리하는 새로운 방법을 제안합니다. 기존의 대조 학습 (contrastive learning) 기법을 개선하여, 정보 손실 없이 스타일 내러티브를 확보할 수 있는 방법을 모색하고 있습니다. 새로운 접근법은 내용의 혼합성이 억제된 생성된 어려운 부정적 예제를 포함함으로써 더 정확한 저작권 분리 결과를 가져옵니다.

- **Technical Details**: 제안된 기술은 InfoNCE (Information Noise Contrastive Estimation) 손실 함수를 수정하여, 스타일과 내용 두 가지 임베딩 공간을 효과적으로 분리합니다. 이를 위해, 모델 학습에 부정적인 예제를 추가하여 훈련 목표와 내용 임베딩 간의 차별화를 꾀합니다. 이 방법은 하이퍼스페이스에서 스타일과 내용의 잠재적 공간을 정의하고 이를 바탕으로 저작권 분석의 정확성을 높이는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 저자들이 다양한 주제로 글을 쓸 때 저작권 추적 (authorship attribution) 작업에서 정확도가 향상된 것이 관찰되었습니다. 어려운 설정에서도 최대 10%의 성능 향상을 보여주며, 제안된 방법이 제로샷(zero-shot) 능력을 보존하면서도 보다 정교한 저자 구별을 가능하게 합니다. 여러 데이터셋과 임무를 통해 성능을 비교하며, 이 방법의 효과성과 유용성을 입증하였습니다.



### Parole de pr\'esidents (1958-2022) (https://arxiv.org/abs/2411.18468)
Comments:
          in French language

- **What's New**: 이 연구는 66년 동안의 프랑스 제5공화국의 8명의 대통령(드골, 퐁피두, 지스카르 에스탱, 미테랑, 시라크, 사르코지, 올랑드, 마크롱) 의 연설을 분석하는 새로운 접근법을 제공합니다. 9202개의 텍스트와 2000만 개 이상의 단어로 구성된 코퍼스를 통해 각 대통령의 독특한 언어 스타일을 도출해냅니다.

- **Technical Details**: 연구는 각 대통령의 어휘(vocabulary)와 품사(part-of-speech)를 기반으로 스타일을 특성화합니다. 또한, 연설 간의 상호 텍스트 거리를 기반으로 각 대통령의典型적인(sequence) 패턴을 깊이 분석합니다.

- **Performance Highlights**: 결과적으로, 연구는 대통령 간의 유사성과 차이를 나타내는 그림을 생성합니다. 이러한 분석은 다채로운 언어 스타일이 정치적 메세지와 커뮤니케이션에 미치는 영향을 이해하는 데 기여할 것으로 보입니다.



### Draft Model Knows When to Stop: A Self-Verification Length Policy for Speculative Decoding (https://arxiv.org/abs/2411.18462)
Comments:
          Code at this https URL

- **What's New**: 이 논문에서는 Speculative Decoding (SD)의 성능을 향상시키기 위해 SVIP(자기 검증 길이 정책)를 도입하였습니다. SVIP는 토큰 생성의 난이도를 인식하여 초안 길이를 동적으로 조정하는 방식을 사용합니다. 이는 고정된 초안 길이 설정에서 발생하는 비효율성을 해결하며, 각 초안 토큰 분포의 엔트로피를 기반으로 합니다.

- **Technical Details**: SVIP의 핵심 기술은 초안 모델의 엔트로피를 분석하여 초안 시퀀스의 길이를 동적으로 조정하는 것입니다. 이 시스템은 쉽게 예측할 수 있는 단어에 대해서는 더 긴 초안을 생성하고, 예측이 어려운 단어가 등장할 경우 더 빨리 검증 과정으로 넘어갑니다. 이를 통해 전체적인 생성 속도를 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, SVIP는 SpecBench에서 기존 SD 방법에 비해 20% 이상의 속도 향상을 달성하였고, MT-Bench에서는 최대 60% 속도 향상을 확인했습니다. 이 방법은 훈련이 필요 없고, 기존의 단독 또는 자동 회귀 초안 모델과 호환성이 높아 다양한 SD 시스템에 적용 가능합니다.



### Is my Meeting Summary Good? Estimating Quality with a Multi-LLM Evaluator (https://arxiv.org/abs/2411.18444)
- **What's New**: MESA는 자연어 생성(NLG) 시스템에서 생성된 회의 요약의 품질을 보다 정확하게 평가하기 위한 새로운 프레임워크입니다. 기존의 ROUGE 및 BERTScore와 같은 메트릭들이 인간 평가와의 낮은 상관관계로 인한 한계점을 극복하는 데 초점을 맞추고 있습니다. MESA는 세 가지 단계의 오류 평가 및 다수의 에이전트 간 토론을 통해 결정 정제를 수행하고 피드백 기반 자기 훈련을 적용하여 인간 판단에 대한 이해와 정렬을 강화합니다.

- **Technical Details**: MESA는 오류별 평가, 전체 평가 및 자기 훈련의 세 가지 단계로 구성됩니다. 각 오류 유형에 대해 가능성 있는 오류를 식별하고 그 영향을 평가하여 Likert 점수(0-5)를 할당합니다. 특히 다수의 에이전트 간 논의 프로토콜을 활용하여 다양한 관점을 반영한 동적 정제 단계를 거치며, 세 단계의 평가를 통해 포괄적인 오류 탐지를 보장합니다.

- **Performance Highlights**: MESA는 GPT-4o를 기반으로 하여 기존의 평가자들보다 평균 0.25 높은 오류 탐지에서의 인간 평가 상관관계를 달성하였습니다. 실험 결과, 자율 훈련 단계는 인간 판단과의 정렬을 돕고, 오류 인스턴스의 잘못된 긍정 식별을 줄이는 데 기여하였습니다. MESA의 유연성은 맞춤형 오류 가이드라인에 적응할 수 있어 다양한 작업에 활용될 수 있는 가능성을 지니고 있습니다.



### Politicians vs ChatGPT. A study of presuppositions in French and Italian political communication (https://arxiv.org/abs/2411.18403)
Comments:
          Published: 2024-07-04

- **What's New**: 이 연구는 프랑스와 이탈리아 정치인들이 발표한 텍스트와 ChatGPT 3.5로 생성된 텍스트를 비교하는 새로운 접근 방식을 제시합니다. 이를 통해 이민국과 유럽연합과 같은 논란이 되는 주제에서의 의사소통을 분석하고 있습니다. 특히, 암시적 의사소통(implicit communication) 및 전제(presuppositions)의 기능에 주목하여 조작적인 언어적 특징을 연구합니다.

- **Technical Details**: 이 논문은 주로 ChatGPT와 같은 대형 언어 모델(Large Language Models)의 준거(pragmatic) 능력에 대한 문헌에 기여하고자 합니다. 연구는 프랑스와 이탈리아 정치인들의 언어 패턴을 분석하여, 정치적 담화에서 전제가 어떻게 의도적으로 사용되는지를 살펴봅니다. 구체적으로, 이러한 전제들이 담화의 맥락(context)과 어떻게 상호작용하는지를 이해하기 위해 기계 학습(machine learning) 모델을 활용합니다.

- **Performance Highlights**: 분석 결과, ChatGPT 3.5가 생성한 텍스트가 정치인들의 텍스트에서 관찰되는 특정한 조작 기법을 반영하고 있음이 밝혀졌습니다. 이를 통해 정치적 의사소통에서 나타나는 맥락적 요소와 언어적 기법의 상관관계를 심층적으로 이해할 수 있습니다. 또한, 이 연구는 정상적이지 않은 정보 전파(communication)의 메커니즘에 대한 통찰력을 제공합니다.



### Topic Modeling and Sentiment Analysis on Japanese Online Media's Coverage of Nuclear Energy (https://arxiv.org/abs/2411.18383)
Comments:
          15 pages, 9 figures, 4 tables

- **What's New**: 후쿠시마 원전 사고 이후 일본의 원자력 발전소는 사용 중지 상태이며, 이는 일본 정부가 원자력 산업을 회복하고 지속 가능한 개발 목표를 달성하기 위해 필요로 하는 공감대를 형성하는 데 있어 중요한 배경이 된다. 최근 사회관계망서비스(SNS)의 발달이 공공 여론을 파악하는 데 도움이 되는 새로운 길을 열어주었다. 본 연구는 3,000개 이상의 유튜브 비디오에 대한 내용을 분석하여 원자력 에너지 관련 주제와 이에 대한 여론을 조사하였다. 이를 통해 일본 내 원자력에 대한 온라인 담론을 해석할 수 있는 중요한 통찰력을 제공하고 있다.

- **Technical Details**: 본 연구에서는 유튜브 비디오 데이터 세트를 수집하기 위해 YouTube Data API를 사용하였다. 원자력 관련 키워드를 통해 15개 일본 방송사에서 제공하는 총 7,505개의 비디오를 수집했고, 이 중 3,101개의 비디오를 분석하여 LDA(topic modeling)와 BERT, GPT 모델을 이용한 감성 분석을 실시하였다. 비디오 댓글의 언어를 식별하기 위해 RoBERTa 모델을 사용하여 일본어 외의 댓글을 제거한 후, 남은 72,678개의 댓글을 분석 대상으로 삼았다.

- **Performance Highlights**: 분석 결과, 유튜브 댓글은 특정 원자력 주제에 대한 공공의 감정을 명확히 드러내 주었다. LDA를 통해 도출한 주요 주제는 일본 내에서 원자력과 관련된 관심사가 어떻게 변화하고 있는지를 명확히 보여주었다. 또한, 감성 분석을 통해 각 주제에 대한 사용자 반응의 세부적인 의미를 파악할 수 있었으며, 이는 앞으로 일본의 원자력 정책에 긍정적 영향을 줄 수 있는 정보로 활용될 것으로 기대된다.



### ChatGPT as speechwriter for the French presidents (https://arxiv.org/abs/2411.18382)
- **What's New**: 이번 연구는 ChatGPT라는 최신 LLM의 글쓰기 스타일을 분석하고, 최근 프랑스 대통령들의 연말 연설문과 비교하는 새로운 시도를 하였습니다. 연구의 주요 초점은 ChatGPT로 생성된 메시지가 실제 정치 지도자의 메시지와 어떤 차이를 보이는지를 불을 밝히는 것입니다. 이러한 접근은 Generative AI에 대한 우려와 기대를 동시에 반영하며, 사용자의 요청에 대한 응답을 자동으로 생성하는 능력에 대한 실질적인 분석을 제공합니다.

- **Technical Details**: 연구에서는 ChatGPT가 생성한 메시지를 Chirac, Sarkozy, Hollande, Macron의 연말 연설문과 비교 분석하였습니다. 분석 결과, ChatGPT는 명사(nouns), 소유 한정사(possessive determiners), 숫자(numbers)를 과도하게 사용하고 있으며, 동사(verbs), 대명사(pronouns), 부사(adverbs)의 사용은 상대적으로 적습니다. 특히, 'devoir', 'nous'와 같은 특정 단어가 과다 사용되고 있는 반면, 'être', 'vouloir', 'falloir'와 같은 조동사는 상대적으로 부족하게 나타났습니다.

- **Performance Highlights**: ChatGPT가 짧은 텍스트를 제공받았을 때, 원문에 가까운 스타일로 메시지를 생성할 수 있는 능력을 보여주었습니다. 그러나 전반적으로 ChatGPT의 발화 스타일은 실제 대통령 연설과 비교할 때 뚜렷한 차별점을 드러냅니다. 이러한 연구 결과는 LLM의 글쓰기 스타일에 대한 심층적인 통찰력을 제공하며, AI 기반의 글쓰기 도구들이 어떻게 발전해야 할지를 고민하게 만듭니다.



### AMPS: ASR with Multimodal Paraphrase Supervision (https://arxiv.org/abs/2411.18368)
- **What's New**: 이 논문에서는 다국어 대화형 자동 음성 인식(ASR) 시스템 향상을 위한 새로운 기술인 AMPS를 소개합니다. AMPS는 여러 언어(힌디어, 마라티어, 말라얄람어, 칸나다어 및 니얀자어)의 대화형 ASR 성능을 개선하기 위해 패러프레이즈(Paraphrase) 기반의 지원을 통합합니다. 이를 통해 기존 ASR 모델의 훈련 과정에서 패러프레이즈를 추가적인 감독으로 사용하여 미흡한 ASR 성능을 보완할 수 있습니다.

- **Technical Details**: AMPS는 다중 모달(multimodal) ASR 모델인 SeamlessM4T를 통해 구현됩니다. 이 모델은 음성 인코더 및 텍스트 인코더와 공유 디코더를 포함하여 음성을 텍스트로 변환하는 음성-텍스트(S2T) 및 텍스트-텍스트(T2T) 경로를 생성합니다. AMPS는 ASR 손실이 높은 경우 패러프레이즈 기반의 추가 지원을 적용하여 모델이 명확하지 않은 오디오에서 의미적으로 유사한 단어를 선택할 수 있는 대안을 제공합니다.

- **Performance Highlights**: AMPS를 사용하여 ASR 성능이 크게 향상된 것을 보고합니다. 여러 언어에서 단어 오류율(Word Error Rate, WER)이 최대 5% 감소한 것으로 나타났으며, 이는 인도 언어를 포함한 다양한 언어의 대화형 음성을 인식하는 데 중요한 성과입니다. 또한 인간 평가를 통해 AMPS의 출력을 기존 ASR 목표로만 파인튜닝한 경우와 비교하여 일관된 개선 결과를 확인하였습니다.



### GPT as ghostwriter at the White Hous (https://arxiv.org/abs/2411.18365)
- **What's New**: 최근 여러 대형 언어 모델(LLMs)이 사용자 요청에 대한 메시지를 생성할 수 있는 능력을 보여주며 새로운 관점을 제시하고 있습니다. 이 연구는 ChatGPT 3.5의 글쓰기 스타일을 분석하며, 미국 대통령들의 연설과 비교합니다. 차별화된 접근 방식을 통해 LLM의 특징을 구체적으로 조사합니다.

- **Technical Details**: 이 연구에서는 레이건(Reagan)부터 오바마(Obama)까지의 국정연설(State of the Union addresses)과 ChatGPT가 자동으로 생성한 연설을 비교합니다. 분석 결과, ChatGPT는 'we'라는 단어를 과도하게 사용하며, 명사와 쉼표의 사용 빈도가 높습니다. 반면, 동사 사용은 적고 평균적으로 더 긴 문장이 생성되는 경향이 있습니다.

- **Performance Highlights**: 결과적으로, ChatGPT는 지정된 스타일을 강제하더라도 생성된 연설의 스타일이 대상 작가의 메시지와는 뚜렷하게 다르다는 것을 보여주었습니다. 또한, ChatGPT는 주로 긍정적인 감정 표현과 기호적 용어(예: freedom, nation)를 사용하는 중립적인 톤을 선택하는 경향을 보입니다. 이러한 특성은 실제 대통령 연설과 명확한 차별점을 드러냅니다.



### Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation (https://arxiv.org/abs/2411.18337)
Comments:
          12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security

- **What's New**: 이 연구에서는 모호한 단어(ambiguous words)의 이해를 개선하기 위해 Large Language Models (LLMs)를 활용하는 새로운 접근법을 제안합니다. 이 방법은 시스템적 prompt augmentation 메커니즘과 다양한 의미 해석을 포함하는 지식 기반(knowledge base)을 결합하고 있습니다. 특히, human-in-loop 방식을 통해 prompt를 보강하는 것이 특징입니다.

- **Technical Details**: 제안된 방법은 Part-of-Speech (POS) 태깅, 모호한 단어의 동의어(synonyms), 측면 기반 의미 필터링(aspect-based sense filtering), 그리고 Few-shot prompting을 통해 LLM을 안내합니다. 이러한 통합 접근법은 LLM이 보다 정확하게 단어의 의미를 해석할 수 있도록 지원합니다. 연구는 FEWS 테스트 데이터와 의미 태그를 사용하여 평가되었습니다.

- **Performance Highlights**: 연구 결과 Few-shot Chain of Thought (COT) prompting 방식을 사용하여 성능이 현저하게 향상된 것으로 나타났습니다. 이 결과는 소셜 미디어 및 디지털 통신에서 단어 해석의 정확성을 높이는 데 기여합니다. 따라서, 기존의 Word Sense Disambiguation (WSD) 방법의 한계를 극복하는 데 중요한 진전을 이뤘습니다.



### Continual Learning in Machine Speech Chain Using Gradient Episodic Memory (https://arxiv.org/abs/2411.18320)
Comments:
          Published as a conference paper at O-COCOSDA 2024. 6 pages; 2 figures

- **What's New**: 이 논문은 자동 음성 인식(ASR) 시스템을 위한 지속적 학습 기법을 제안합니다. 이러한 접근법은 머신 스피치 체인 프레임워크를 활용하여 기존의 학습된 과제의 성능을 유지하면서 새로운 과제를 순차적으로 학습할 수 있도록 합니다. 특히, 텍스트-음성 변환(TTS) 컴포넌트를 통합하여 재생(replay) 메커니즘을 지원함으로써 기계 학습 모델이 연속적으로 직면하는 여러 과제를 효과적으로 다룰 수 있습니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성됩니다. 첫 번째 단계는 기초 작업에 대한 감독 학습(supervised learning)이며, 두 번째 단계는 반감독 학습(semi-supervised learning)으로 ASR와 TTS가 서로 향상되는 과정입니다. 마지막 세 번째 단계는 새로운 작업을 위한 지속적 학습으로, TTS에 의해 생성된 기초 작업의 입력을 재생하여 수행됩니다. 이 과정에서 기울기 에피소딕 메모리(GEM)를 사용하여 이전 작업 데이터와 새로운 작업 데이터 간의 기울기를 조정합니다.

- **Performance Highlights**: 실험은 LJ Speech 데이터셋을 사용하여 수행되었으며, 제안된 방법은 전통적인 파인튜닝(fine-tuning) 및 멀티태스크 학습(multitask learning) 접근법에 비해 성능이 뛰어난 것으로 나타났습니다. 특히, 제안된 방법은 신호 대 잡음비(SNR) 조건에서 높은 성능을 유지하면서도 평균 오류율을 40% 감소시키는 성과를 달성했습니다. 따라서 우리의 반감독 머신 스피치 체인 접근법은 효과적이고 효율적인 지속적 학습의 가능성을 증명하고 있습니다.



### Aligning Pre-trained Models for Spoken Language Translation (https://arxiv.org/abs/2411.18294)
- **What's New**: 이 논문은 사전 훈련된 자동 음성 인식(ASR) 모델과 기계 번역(MT) 모델을 소형 커넥터 모듈(Q-Former)로 결합하는 새로운 접근 방식을 탐구합니다. 이 커넥터는 음성 및 텍스트 모달리티 간의 갭을 해소하고 ASR 인코더 임베딩을 MT 인코더의 잠재 표현 공간으로 변환합니다. 본 연구는 How2 영어-포르투갈어 데이터셋에서 실험을 수행하며, 프레임워크의 효용성을 입증합니다.

- **Technical Details**: 본 연구에서 제안한 두 가지 정렬 아키텍처는 서로 다른 구성의 MT 모델을 사용하는데, 둘 다 ASR 모델의 인코더 부분만 사용합니다. 첫 번째 아키텍처인 Encoder-Connector-Decoder (ECD)는 고정된 ASR 인코더에서 숨겨진 오디오 표현을 추출하여 이를 커넥터 모듈을 통해 MT 디코더의 차원으로 변환합니다. 두 번째 아키텍처인 Encoder-Connector-Encoder-Decoder (ECED)에서는 커넥터 모듈이 음성 임베딩을 MT 인코더의 입력 단어 임베딩 공간으로 투영합니다.

- **Performance Highlights**: ASR와 MT 모델의 크기를 증가시키면 음성 번역 결과가 전반적으로 향상되며, 커넥터 네트워크의 크기는 작게 유지될 수 있습니다. 또한, 커넥터 네트워크는 도메인 어댑터 역할을 하여, 정렬된 MT 모델이 도메인 외부에서 더욱 향상된 번역 성능을 보여줍니다. 마지막으로, 제안된 프레임워크는 저자원 시나리오에서도 효과적임을 입증합니다.



### Neutralizing Backdoors through Information Conflicts for Large Language Models (https://arxiv.org/abs/2411.18280)
- **What's New**: 이번 논문은 기존의 LLM(대형 언어 모델)에서 발생할 수 있는 백도어(Backdoor) 공격을 효과적으로 제거하는 새로운 방법론을 제안합니다. 제안된 방법은 내부 정보 충돌(internal conflicts)과 외부 정보 충돌(external conflicts)을 활용하여 백도어 행동을 중화시키기 위한 구조입니다. 이를 통해 기존의 백도어 방어 방법들보다 더 높은 성공률로 백도어 공격을 막을 수 있게 됩니다.

- **Technical Details**: 제안된 모델은 Low-Rank Adaptation(LoRA) 기술을 사용하여 깨끗한 데이터로 훈련된 충돌 모델(conflict model)을 백도어가 포함된 LLM에 통합합니다. 내부적으로는 모델의 파라메트릭 메모리(parametric memory) 내에 모순된 정보를 삽입하여 공격 트리거를 차단합니다. 외부적으로는 프롬프트에 모순된 증거를 포함시켜 모델이 내부 백도어 지식을 도전하도록 유도합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 4개의 널리 사용되는 LLM에 대해 8개의 고급 백도어 공격의 성공률을 최대 98%까지 감소시키면서도 깨끗한 데이터에 대해 90% 이상의 높은 정확도를 유지했습니다. 이는 기존의 8가지 백도어 방어 방법보다 우수한 성능을 보이며, 적응형 백도어 공격에도 견고함을 입증했습니다.



### Hidden Data Privacy Breaches in Federated Learning (https://arxiv.org/abs/2411.18269)
- **What's New**: 이 논문에서는 Federated Learning (FL) 환경에서의 새로운 데이터 복원 공격 방법을 제안합니다. 기존의 공격 방식과는 달리, 본 연구는 악의적인 코드 주입을 통해 모델의 변화를 감지하지 못하도록 데이터를 추출하는 고유한 방법을 개발했습니다. 이를 통해 기밀 데이터의 노출이 증가하고, 개발자들은 새로운 방어 체계를 구축해야 할 필요성을 강조합니다.

- **Technical Details**: 본 연구에서 제안하는 공격 방식은 세 가지 주요 기술을 포함합니다: 1) 독특한 희소 인코딩 설계, 2) 차단 분할(block partitioning) 방법, 3) 피보나치 기반의 인덱싱 설계입니다. 기존의 공격들은 모델 아키텍처에서 감지 가능한 변화를 요구하는 반면, 본 방법은 파라미터 공유를 통해 숨겨진 모델을 체계적으로 주입합니다. 이를 통해 고해상도 이미지를 처리하는 데 효과적이며, 다수의 데이터 세트에서 기존의 데이터 복원 공격보다 우수한 성능을 발휘했습니다.

- **Performance Highlights**: 본 논문에서 제안한 방법은 CIFAR-10과 CIFAR-100 데이터 세트에서 평균적으로 512장의 고해상도 이미지를, ImageNet 및 CelebA에서는 64장 이상의 이미지를 탈취하는 데 성공했습니다. 이 결과는 공격 능력이 탁월할 뿐만 아니라, 고해상도 데이터 및 대규모 데이터 도용 시나리오를 효과적으로 처리할 수 있음을 시사합니다. 전반적으로, 제안된 방법은 5개의 최신 데이터 복원 공격 방법 및 5개의 탐지 방법에 대해 적시에 높은 성능을 보였습니다.



### MetaphorShare: A Dynamic Collaborative Repository of Open Metaphor Datasets (https://arxiv.org/abs/2411.18260)
- **What's New**: 메타포 처리(NLP) 커뮤니티의 연구자들은 다양한 언어로 메타포 데이터 세트를 통합하고 이를 공개적으로 접근할 수 있도록 하는 웹사이트, MetaphorShare를 제안합니다. 이 플랫폼의 개발 목적은 연구자들이 메타포 관련 데이터 세트를 쉽게 공유하고 업로드할 수 있도록 하여 메타포 연구가 더욱 발전하도록 지원하는 것입니다. 특히, 이 사이트는 다양한 언어로 된 데이터 세트도 포함되어 있어 메타포 연구의 접근성과 유용성을 높이고자 합니다.

- **Technical Details**: MetaphorShare는 연구자들이 메타포 사용에 대한 레이블이 붙은 데이터 세트를 통합하고,統一된 형식으로 이들 리소스를 쉽게 비교하고 검색할 수 있도록 돕는 것을 목표로 합니다. 이 플랫폼은 최소한의 제약 형식을 요구하여 데이터 세트를 데이터베이스에 저장하고, 여러 필드에서 효율적으로 검색할 수 있는 기능을 제공합니다. 이전 연구에서 많은 데이터 세트가 일관되지 않은 형식으로 저장되어 있었기 때문에, 이를 해결하기 위해 기존의 데이터 세트 형식을 분석하고 통합하는 과정을 거쳤습니다.

- **Performance Highlights**: MetaphorShare는 AI/NLP 커뮤니티와 언어학 커뮤니티 간의 협력을 촉진하고, 메타포 처리 모델의 개인화된 생성이 가능하도록 지원합니다. 이 웹사이트는 메타포 식별 및 처리 시스템을 개발하는 NLP 연구자들이 사용함으로써 데이터 세트를 더 쉽게 접근하고 활용할 수 있도록 합니다. 또한, MetaphorShare는 다양한 연구자 및 학문 분야의 연구자들이 소통하고 협력하는 새로운 시너지를 창출할 것으로 기대되고 있습니다.



### A gentle push funziona benissimo: making instructed models in Italian via contrastive activation steering (https://arxiv.org/abs/2411.18247)
- **What's New**: 이 논문은 모델을 미세 조정(fine-tuning) 없이 이탈리아어로의 성능을 향상시키기 위한 활성화 유도(activation steering) 기술의 가능성을 탐구합니다. 실험 결과, 이탈리아어 유도는 다양한 모델에 성공적으로 적용되며, 미세 조정된 모델과 비교해도 유사하거나 더 나은 성과를 달성한다는 것을 보여줍니다. 이탈리아어 생성을 한층 더 높게 질적 품질과 일관성을 제공한다고 논의하고 있습니다.

- **Technical Details**: 모델의 초기 단계에서 이미 소량의 목표 언어(이탈리아어)를 보았다고 가정하고, 활성화의 차이를 강조하는 대조적 예제(prompts)를 작성하여 이탈리아어 유도 벡터를 추출합니다. 스탠포드 알파카 데이터셋을 사용해 이탈리아어 및 영어 질문-답변 형식의 프롬프트를 생성한 후, 모델의 각 어텐션 헤드(output)에서 마지막 토큰의 활성화를 수집하여 평균을 낸 결과, 최종 유도 벡터를 산출합니다. 이 유도 벡터는 모델을 원하는 방향으로 조종하기 위한 것이며, 각 생성된 토큰에 대해 적용됩니다.

- **Performance Highlights**: 이 방법은 240K의 데이터 세트를 사용하는 표준 미세 조정 접근 방식과 비교할 때 훨씬 적은 데이터(100 미만)로 유사한 성능을 달성했습니다. 실험을 통해, 이탈리아어 생성을 위한 활성화 유도는 고품질의 출력을 제공하며, 모델이 원래 특정 언어로 훈련되지 않았더라도 성능을 극대화할 수 있음을 입증했습니다. 최신 LLM 환경에서, 이러한 방법은 효율성과 효과성을 동시에 제공할 수 있는 가능성을 보여줍니다.



### Thai Financial Domain Adaptation of THaLLE -- Technical Repor (https://arxiv.org/abs/2411.18242)
- **What's New**: 본 연구에서는 태국 금융 도메인에 맞춘 대형 언어 모델(LLM)인 Thai Financial LLM을 개발했습니다. 이는 태국 증권 거래소의 투자 상담사(IC) 시험 데이터셋을 활용하여 전문 용어와 현지 규제를 포함한 특정 요구 사항에 대응하고자 했습니다. 기존의 금융 LLM들이 태국 금융 시장의 특수성을 충분히 반영하지 못했다는 점에서 이 연구는 중요한 기여를 합니다.

- **Technical Details**: 태국 금융 LLM의 개발에는 데이터 증강(data augmentation), 효율적인 학습을 위한 ReLoRA, 도메인 지식 구축을 위한 Continued Pretraining(CPT), 그리고 미세 조정을 위한 Rank-Stabilized LoRA(rsLoRA) 기법이 사용되었습니다. SFT(Supervised Fine-Tuning)를 통해 시험 환경을 시뮬레이션하였고, DPO(Direct Preference Optimization)를 통해 모델의 피드백 기반 개선을 수행했습니다.

- **Performance Highlights**: 모델은 IC 시험의 P1, P2, P3 단계에서 각각 72%, 72%, 84%의 점수를 기록하였으며, 이는 태국 금융 자문 업무에 대한 효과성을 보여줍니다. 이 연구는 태국의 고유한 금융 지식을 포함한 LLM의 필요성을 강조하며, 전문화된 금융 어플리케이션에 대한 잠재력을 제공합니다.



### SentiXRL: An advanced large language Model Framework for Multilingual Fine-Grained Emotion Classification in Complex Text Environmen (https://arxiv.org/abs/2411.18162)
- **What's New**: 이 논문에서는 다국어 및 복합 언어 환경 내에서 세분화된 감정 분류를 위한 새로운 프레임워크인 SentiXRL을 제안한다. 이 프레임워크는 감정 검색 강화 모듈(emotion retrieval enhancement module)과 자가 순환 분석 협상 메커니즘(Self-Analytical Negotiation Mechanism, SANM)을 포함하여 복잡한 맥락에서 감정 인식의 정확성을 높인다. SentiXRL은 여러 표준 데이터세트에서 다른 모델들을 능가하는 성과를 보여주며, 세분화된 감정 주석 데이터세트를 통합하고 카테고리 불균형의 영향을 검증하여 그 우수성을 입증한다.

- **Technical Details**: SentiXRL은 감정 인식 및 검증에 있어 LLM의 추론 능력을 최대한 활용하도록 설계된 효율적인 감정 검색 모듈을 포함하고 있다. 이 모듈은 맥락 정보를 역사적 대화(historical dialogue)를 통해 연결하며, 감정 추론(emotion reasoning)을 수행한다. SANM은 감정 확인 및 논리적 추론을 가능하게 하여 복잡한 텍스트와 맥락에서 감정 분류 능력을 향상시키는데 기여한다.

- **Performance Highlights**: SentiXRL은 다섯 개의 표준 대화 감정 인식(ERC) 벤치마크에서 기존 모델들을 모두 초월하는 성능을 보여준다. 이 모델은 MELD, Emorynlp, IEMOCAP과 같은 감정 분석 데이터세트에서도 우수한 전반적인 성과를 달성하였다. 마지막으로, 카테고리 불균형이 LLM에 미치는 영향을 조사하는 실험을 통해 중요한 통찰을 제공한다.



### A survey on cutting-edge relation extraction techniques based on language models (https://arxiv.org/abs/2411.18157)
Comments:
          50 pages, under review in Artificial Intelligence Review

- **What's New**: 이번 연구는 Relation Extraction (RE)에 관한 최신 발전을 종합적으로 조사하여, 137개의 ACL 회의를 통해 발표된 논문들을 분석합니다. 연구의 핵심은 언어 모델을 활용하여 RE 기술의 진화와 현황을 조명하는데 있습니다. BERT 기반의 방법들이 RE에서 가장 뛰어난 결과를 도출하는 데 주도적인 역할을 하며, T5와 같은 새로운 대형 언어 모델(LLMs)의 유망한 가능성도 주목받고 있습니다.

- **Technical Details**: RE는 텍스트 내에서 다양한 개체 간의 관계를 식별하고 추출하는 작업으로, 비구조적 데이터를 다루는 데 중점을 둡니다. 자연어 처리(NLP)의 한 분야로, RE는 Named Entity Recognition (NER), relation identification, relation classification의 세 가지 주요 구성 요소로 나뉩니다. 이 연구에서는 최근 몇 년간 발표된 최신 RE 기법들을 언어 모델 관점에서 분석하며, BERT와 RoBERTa와 같은 언어 모델의 다양한 활용을 검토하고 있습니다.

- **Performance Highlights**: ACL 회의에서 발표된 논문들을 바탕으로, 2020년 이후 언어 모델의 도입이 RE의 발전에 미친 영향을 면밀히 조사하였습니다. 최종적으로, 65개의 연구 기여 논문을 분석하였으며, 이들 논문은 언어 모델을 활용한 새로운 접근 방식을 포함합니다. TACRED, NYT10 및 DocRED와 같은 여러 데이터셋이 평가 기준으로 사용되었으며, 이러한 분석은 RE의 다양한 도메인과 관련된 중요한 통찰을 제공합니다.



### MSA-ASR: Efficient Multilingual Speaker Attribution with frozen ASR Models (https://arxiv.org/abs/2411.18152)
- **What's New**: 이번 논문에서는 화자의 속성을 정확하게 기록할 수 있는 새로운 자동 음성 인식(SA-ASR) 방법을 제안합니다. 이 방법은 언어별 데이터셋을 사용하여 언어에 구애받지 않는 강력한 사전 훈련된 ASR 모델을 활용하면서도, 복잡한 모듈 방식이나 조정이 필요하지 않습니다. 논문의 주요 기여로는 다양한 다국어 데이터셋에서 효과적으로 화자 정보를 추출할 수 있음을 보여줍니다.

- **Technical Details**: SA-ASR 모델은 두 가지 주요 구성 요소인 ASR 모듈과 화자 모듈로 구성됩니다. ASR 모듈은 입력 신호를 기반으로 숨겨진 특징을 생성하고, 화자 모듈은 이러한 신호를 사용하여 화자 임베딩을 예측합니다. 우리의 접근 방식은 ASR 모듈이 통합되어 안정성과 일반화 가능성을 유지하면서, 화자 모듈이 직접적으로 훈련되는 구조를 채택합니다.

- **Performance Highlights**: 실험 결과, 우리의 MSA-ASR 모델은 다양한 데이터셋에서 경쟁력 있는 성능을 보여주며, 특히 다국어 데이터 셋을 처리하는 데 있어 뛰어난 효과를 입증합니다. 이러한 결과는 실전 응용 프로그램에서의 활용 가능성을 강조합니다. 특히, 복잡한 조정 없이도 비약적인 성능 향상이 가능함을 보여줍니다.



### Curriculum Demonstration Selection for In-Context Learning (https://arxiv.org/abs/2411.18126)
Comments:
          Accepted at the 40th ACM/SIGAPP Symposium On Applied Computing (SAC 2025), Main Conference

- **What's New**: 이번 논문에서는 Curriculum Demonstration Selection (CDS)라는 새로운 시연 선택 방법을 제안합니다. CDS는 간단한 예제에서 복잡한 예제로 점진적으로 학습하도록 돕기 위해 샘플을 복잡도에 따라 분할합니다. 따라서 LLM은 다양한 난이도의 사례를 통해 학습할 수 있습니다. 실험 결과 CDS는 아홉 가지 LLM에서 세 가지 기준 벤치마크에서 기존 방법보다 일관되게 우수한 성능을 보였습니다.

- **Technical Details**: CDS는 데이터 세트를 난이도 수준에 따라 분할하고, 각 그룹에서 사례를 가져와 선택합니다. 이를 통해 LLM은 복잡성이 다양한 사례에서 점진적으로 이해를 다질 수 있습니다. 선택 과정은 유사성 기반 또는 무작위 방식으로 진행되며, 샘플의 유사성을 평가하기 위해 사전 훈련된 Transformer 모델의 CLS 임베딩을 사용합니다. 이러한 방법론은 LLM이 특정 패턴에 과적합되는 것을 방지합니다.

- **Performance Highlights**: CDS 방식은 특히 문제 해결에서 기존 방법이 부족했던 어려운 문제들에 대한 LLM의 성능 향상에 매우 효과적임이 입증되었습니다. 실험을 통해 여러 기준 벤치마크에서 CDS의 효율성이 과학적으로 확인되었으며, 제안된 방식이 LLM의 복잡한 문제 해결 능력을 획기적으로 증가시키는 잠재력을 가지고 있음을 보여주었습니다.



### Training and Evaluating Language Models with Template-based Data Generation (https://arxiv.org/abs/2411.18104)
Comments:
          8 pages, 2 figures

- **What's New**: 최근의 대형 언어 모델(LLMs) 발전은 자연어 처리(NLP) 분야에서 큰 변화를 가져왔습니다. 하지만 이러한 모델들은 복잡한 추론이 필요한 작업, 특히 수학 문제 해결에서는 어려움을 겪고 있습니다. 이를 극복하기 위해 Template-based Data Generation (TDG) 방식을 도입하여 LLMs(GPT-4)를 활용해 매개변수화된 메타 템플릿을 자동으로 생성하고, 700만 개 이상의 고품질 수학 문제 및 해결책을 포함한 TemplateMath Part I: TemplateGSM 데이터를 구축했습니다.

- **Technical Details**: TDG는 매개변수화된 템플릿을 기반으로 광범위한 수학 문제와 그 해결책을 시스템적으로 생성하는 방법입니다. GPT-4를 사용하여 생성된 이러한 메타 템플릿은 다양한 문제 구조와 언어 스타일을 캡처하도록 설계되었습니다. 생성된 문제와 해결책은 코드 실행 및 LLM 검증을 통해 정확성이 보장되고, 검증 과정을 통해 높은 품질의 데이터가 확보되도록 순환적으로 진행됩니다.

- **Performance Highlights**: TemplateGSM 데이터 세트는 700만 개 이상의 Grade School 수준의 수학 문제로 구성되어 있으며, 각 문제에는 코드 기반 해결책과 자연어 설명이 포함되어 있습니다. 제안된 TDG 방법 덕분에 수학 문제에 대한 데이터의 양과 품질이 대폭 향상되었고, 다양한 문제 유형 및 난이도에 대한 학습이 가능해졌습니다. 이러한 데이터 세트는 LLMs의 수학적 추론 능력 향상에 기여할 것으로 기대됩니다.



### Fine-Tuning Small Embeddings for Elevated Performanc (https://arxiv.org/abs/2411.18099)
- **What's New**: 이 논문은 네팔리 언어에 대한 사전 학습된 불완전한 BERT 모델을 활용하여, 데이터 부족 문제에도 불구하고 NLP 작업에서 성능을 향상시키는 방법을 제시합니다. 특히, 기존의 BERT 모델에 비해 작은 임베딩을 파인튜닝(finetuning)하여 결과를 개선하는 접근법을 다루고 있습니다. 결과적으로, 모델의 기준선(original baseline)과 비교했을 때 성능이 현저히 향상되었습니다.

- **Technical Details**: BERT(Bidirectional Encoder Representations from Transformers) 모델은 다양한 NLP 작업에서 사용되며, 문맥에 따라 단어를 나타내는 Context-Dependent Embeddings 기술을 활용합니다. 이 연구는 정제된(Regularized) 및 비정제(Unregularized) 데이터를 통해 네팔리 문장 간의 의미적 및 통사적 관계를 효과적으로 캡처하는 워드 임베딩을 생성했습니다. 데이터 수집에는 웹 스크래핑과 API를 사용하여 다양한 소스에서 정보를 추출했습니다.

- **Performance Highlights**: 조사 결과, 사전 학습된 불완전한 BERT 모델을 파인튜닝한 결과는 전반적으로 원본 모델보다 우수했습니다. 본 연구는 NLP 작업에서의 네팔리 언어 모델 성능 향상에 기여하며, 특히 네팔리 감정 분석 및 기계 번역 시스템 개선에 중요한 역할을 할 수 있습니다. 연구의 궁극적인 목표는 소규모 언어 모델의 성능을 극대화하여 더 정확하고 효과적인 NLP 애플리케이션의 개발을 촉진하는 것입니다.



### Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cach (https://arxiv.org/abs/2411.18077)
- **What's New**: 이 논문은 LLM에서 KV 캐시의 메모리 최적화를 위한 새로운 방법인 MiniKV를 제안합니다. MiniKV는 긴 문맥 작업에 대한 정확성을 유지하면서 KV 캐시 크기를 효과적으로 줄이는 혁신적인 2비트 레이어-차별적(KV)의 기법을 사용합니다. 연구진은 FlashAttention과 호환되는 특별한 CUDA 커널을 개발하여 MiniKV의 메모리 소비를 감소시킵니다.

- **Technical Details**: MiniKV는 선택적 KV와 2비트 양자화의의 융합을 통해 KV 캐시를 압축하는 새로운 방법론입니다. PV 캐시에서 중요 토큰을 선택하는 정책을 개선하여, 가장 효과적인 위치에 KV 캐시를 할당합니다. 이를 통해 선택적 KV와 양자화의 호환성 문제를 극복하며, 선택된 토큰은 생성 단계 동안 고정됩니다.

- **Performance Highlights**: MiniKV는 다양한 긴 문맥 작업에서 86%의 KV 캐시 압축 비율을 달성하면서 98.5% 이상의 정확성을 회복할 수 있었습니다. 이러한 성과는 단일 NVIDIA A100 GPU에서 최대 66.4%의 처리량 향상을 보여줍니다. 이 연구는 MiniKV가 모델의 정확성을 유지하면서도 최대 8배의 KV 캐시 크기 감소를 가능하게 한다는 첫 번째 결과를 제시합니다.



### Can bidirectional encoder become the ultimate winner for downstream applications of foundation models? (https://arxiv.org/abs/2411.18021)
Comments:
          9 pages, 4 figures, FLLM2024

- **What's New**: 이 논문에서는 인공지능(AI)의 발전 역사와 최근의 기초 모델(foundational model)에 대한 진전을 설명합니다. 특히, Bidirectional Encoder Representations from Transformers(BERT)와 Generative Pre-trained Transformer(GPT)와 같은 모델들이 자연어 처리(NLP) 분야에서 어떻게 혁신을 가져왔는지를 강조하고 있습니다. BERT는 마스크된 언어 모델(masked language model)을 사용하여 단어 예측과 특징 추출 능력을 향상시키고 있습니다.

- **Technical Details**: 본 논문에서는 한 방향(one-way) 모델과 양방향(bidirectional) 모델을 비교하고, 각각의 목적에 따른 차이를 분석합니다. GPT는 일반적으로 한 방향 언어 모델로, 다음 단어의 확률을 계산하는 방식입니다. 반면에 BERT는 양방향 언어 모델로, 현재 단어를 처리할 때 전방과 후방 정보 모두를 고려하여 더 뛰어난 표현력을 가져옵니다.

- **Performance Highlights**: BERT는 Masked Language Model(MLM)과 Next Sentence Prediction(NSP)의 두 가지 주요 과제를 통해 주목할 만한 성능을 발휘합니다. 이 논문에서는 Stanford Question Answering Dataset(SQuAD)와 General Language Understanding Evaluation(GLUE)에서의 모델 성능을 비교하여 BERT의 효과성을 명확히 보여줍니다. BERT의 특출한 특징 추출 능력 덕분에 여러 자연어 처리 작업에 효과적으로 적용될 수 있습니다.



### DRS: Deep Question Reformulation With Structured Outpu (https://arxiv.org/abs/2411.17993)
- **What's New**: 이 논문에서는 DRS(Deep Question Reformulation with Structured Output)라는 새로운 제로샷(zero-shot) 방법을 제안합니다. 이 방법은 대형 언어 모델(LLMs)이 사용자가 새로운 문서에서 관련 지식을 추출할 수 있도록 질문을 재구성하는 데 도움을 줄 수 있도록 설계되었습니다. 기존의 접근 방식들이 사용자가 의미 있는 정보에 접근할 수 있도록 돕지 못하는 상황을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DRS 방법은 세 가지 주요 단계로 구성됩니다: (i) 핵심 엔티티 추출 및 필터링, (ii) 엔티티 조합 검색 및 구조화된 출력 생성, (iii) DFS(Depth First Search) 가지치기 및 후보 선택. 이 방법은 LLM을 활용하여 질문을 변환하고, 제시된 엔티티를 포함하는 답변 가능한 질문으로 재구성하는 방식으로 작동합니다. DFS 기반 알고리즘을 사용하여 질문의 신뢰성을 극대화합니다.

- **Performance Highlights**: 실험 결과, DRS 방법은 GPT-3.5의 질문 재구성 정확력을 23.03%에서 70.42%로 크게 향상시켰고, Gemma2-9B와 같은 오픈 소스 LLM의 경우 성능이 26.35%에서 56.75%로 개선되었습니다. 이러한 성과는 DRS 방법이 LLM의 질문 재구성 기능을 극대화하고, 정보 추출의 유용성을 높이는 데 기여함을 보여줍니다.



### New Faithfulness-Centric Interpretability Paradigms for Natural Language Processing (https://arxiv.org/abs/2411.17992)
Comments:
          Doctoral thesis

- **What's New**: 이 논문에서는 Machine Learning이 점점 더 많은 중요한 응용 프로그램에 사용됨에 따라 모델에 대한 신뢰할 수 있는 설명을 제공하는 방법에 대해 탐구합니다. 최근의 해석 가능성(interpretability) 방법들이 신뢰성(faithfulness) 문제로 어려움을 겪고 있다는 점을 지적하며, 새로운 해석성 패러다임을 개발해야 한다고 주장합니다.

- **Technical Details**: 저자는 신뢰성을 측정할 수 있는 새로운 지표(metrics)를 개발하고, 이를 바탕으로 두 가지 새로운 해석성 패러다임인 Faithfulness Measurable Models (FMMs)과 Self-explanations을 탐구합니다. Self-explanations은 대형 언어 모델이 스스로를 설명하도록 하는 개념으로, 현재의 모델이 일관되게 이를 수행하지 못함을 확인하였으며, 무엇이 이러한 기능을 가능하게 할 수 있는지를 제안합니다.

- **Performance Highlights**: FMMs는 신뢰성을 최대화하기 위해 설명을 최적화할 수 있도록 설계된 모델입니다. 연구 결과 FMMs는 신뢰성 면에서 이론적 최적치에 가까운 설명을 생성함을 확인하였습니다. 또한, FMMs를 사용할 때는 사후(post-hoc) 및 내재적(intrinsic) 설명이 모델 및 작업에 의존하지 않는다는 점에서 개선된 결과를 보여줍니다.



### QuaLLM-Health: An Adaptation of an LLM-Based Framework for Quantitative Data Extraction from Online Health Discussions (https://arxiv.org/abs/2411.17967)
- **What's New**: 이번 연구에서는 Reddit과 같은 소셜 미디어에서 건강 관련 논의에서 임상적으로 중요한 정량적 데이터를 추출하는 새로운 프레임워크인 QuaLLM-Health를 제안합니다. 이를 통해 GLP-1 수용체 작용제에 관한 토론에서 의료 전문가가 필요한 주요 데이터를 효과적으로 수집하고 분석할 수 있습니다. 연구는 410,000개의 게시물과 댓글을 수집한 후, 암 관련 논의에 대한 필터링을 거쳐 검사 결과를 제공합니다.

- **Technical Details**: QuaLLM-Health 프레임워크는 데이터 수집, 전처리, 주석 가이드라인 개발, 인간 주석, LLM 프롬프트 엔지니어링 및 성능 평가라는 주요 구성 요소로 구성됩니다. LLM을 활용하여 자동화된 정량적 변수 추출이 가능하도록 최적화된 파이프라인을 개발하였으며, 모든 변수에 대해 0.85 이상의 정확도를 달성했습니다. 우리 연구는 LLM과 전문가 지식을 결합하여 소셜 미디어에서의 임상 데이터 추출의 정확성과 신뢰성을 보장합니다.

- **Performance Highlights**: 본 연구에서 개발한 QuaLLM-Health는 대규모 데이터셋에서 정량적 변수를 효율적으로 추출하여 약 3달러의 비용으로 1시간 이내에 완료되었습니다. 최적화된 LLM은 F1 점수 매크로가 0.90을 초과하고, 검증 테스트에서 95%의 일치율을 보여주었습니다. 이 연구는 다양한 건강 분야에서 환자 생성 데이터를 대규모로 분석할 수 있는 가능성을 demonstrated합니다.



### Evaluating Generative AI-Enhanced Content: A Conceptual Framework Using Qualitative, Quantitative, and Mixed-Methods Approaches (https://arxiv.org/abs/2411.17943)
- **What's New**: Generative AI(GenAI)가 콘텐츠 생성에 혁신적인 변화를 가져왔습니다. 이 논문은 GenAI 모델이 과학적 글쓰기 향상에 미치는 영향을 평가하기 위한 다양한 연구 방법론을 고찰합니다. 정성적, 정량적, 혼합 방법을 활용하여 GenAI의 성능을 체계적으로 평가하며, 각 방법이 제공하는 독창적인 통찰력을 강조합니다.

- **Technical Details**: 정성적 연구에서는 전문가 리뷰어로부터 심층 피드백을 수집하고 주제 분석 도구를 통해 개선점을 분석합니다. 정량적 방법은 BLEU, ROUGE와 같은 자동화된 메트릭 및 사용자 조사를 통해 언어적 일관성, 유창성 및 구조의 개선을 객관적으로 측정합니다. 혼합 방법론은 통계적 평가와 상세한 정성적 통찰력을 통합하여 GenAI로 생성된 콘텐츠의 포괄적인 평가를 가능하게 합니다.

- **Performance Highlights**: 이 연구에서는 GenAI로 생성된 콘텐츠의 품질과 기술적 정확성을 계량화하는 방법을 제시하여 기존 편집 프로세스와의 비교를 통해 평가할 수 있는 강력한 프레임워크를 제공합니다. 이러한 방법론을 활용하여 연구자들은 GenAI의 성능 향상을 평가하고, 그 활용을 개선하며, 의료 및 과학 연구와 같은 고위험 영역에서 책임감 있게 채택하도록 안내할 수 있습니다.



### HOPPR Medical-Grade Platform for Medical Imaging AI (https://arxiv.org/abs/2411.17891)
Comments:
          6 pages, 3 figures

- **What's New**: HOPPR Medical-Grade Platform은 인공지능 분야에서 큰 비전을 가진 언어 모델(large vision language models, LVLMs)의 배포를 가속화하기 위해 혁신적인 접근 방식을 제시합니다. 이 플랫폼은 수백 개의 영상 센터에서 수집된 수백만 개의 이미지 연구와 텍스트 보고서를 기반으로 사전 훈련된 기초 모델(foundation models)을 제공합니다.

- **Technical Details**: HOPPR 플랫폼은 대규모 모델을 개발하는 데 필요한 방대한 컴퓨팅 인프라를 갖추고 있으며, 임상 환경에서 배포를 위해 미세 조정(fine-tuning)된 모델을 평가하는 표준을 마련한 품질 관리 시스템을 제공합니다. 또한 모든 데이터는 비식별화(deidentified)되어 HIPAA 규정 준수를 확보하며 안전하게 저장됩니다.

- **Performance Highlights**: HOPPR는 의료 이미징(medical imaging)에 대한 LVLM 솔루션의 배포를 가속화하여 방사선 의사의 업무 흐름(workflows)을 최적화하고 이 분야에서 증가하는 요구를 충족시키는 것을 목표로 합니다. 개발자는 HOPPR 플랫폼에서 모델을 안전하게 호스팅하고 API를 통해 기존 클리닉 워크플로 내에서 추론을 진행할 수 있는 기능을 제공합니다.



### Leveraging Large Language Models and Topic Modeling for Toxicity Classification (https://arxiv.org/abs/2411.17876)
- **What's New**: 이번 연구는 콘텐츠 조절(content moderation)과 독성 분류(toxicity classification)에서 주석자(annotation)의 입장이 모델 성능에 미치는 영향을 심층적으로 분석한 것입니다. 연구자들은 BERTweet 및 HateBERT 모델에 주제 모델링을 활용한 전달 학습(transfers learning) 방법을 적용하여, 기존 모델들과 비교하여 F1 점수가 현저히 향상된다는 것을 발견했습니다. 이러한 결과는 대형 언어 모델이 텍스트 독성을 정확하게 탐지하고 해석하는 데 제한적이라는 것을 보여줍니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋은 NLPositionality로, 라벨이 지정된 독성 트윗과 주석자의 인구 통계 메타데이터로 구성되어 있습니다. Latent Dirichlet Allocation (LDA) 기법을 통해 주제를 클러스터링한 후, BERTweet 및 HateBERT 모델을 세부 데이터로 파인튜닝하여 독성 트윗 분류 성능을 높였습니다. 파인튜닝 과정에서 BERTweet은 짧은 형태의 트윗에 적합하게 조정되었고, HateBERT는 토픽에 특화된 문맥을 이해하도록 설정되었습니다.

- **Performance Highlights**: 모델의 성능 결과에 따르면, BERTweet과 HateBERT는 특정 주제에 대한 파인튜닝을 통해 F1 점수가 개선되었습니다. BERTweet은 전체 데이터셋에 대한 파인튜닝이 가장 낮은 결과를 보였고, 반면 HateBERT는 전체 데이터셋에서 두 번째로 높은 성능을 보였습니다. 연구는 이와 같은 성과가 주제 0에 더 두드러진 독성 패턴이 존재하기 때문일 가능성을 제시합니다.



### LongKey: Keyphrase Extraction for Long Documents (https://arxiv.org/abs/2411.17863)
Comments:
          Accepted for presentation at the 2024 IEEE International Conference on Big Data (IEEE BigData 2024). Code available at this https URL

- **What's New**: 이 논문은 LongKey라는 새로운 프레임워크를 소개하며, 주로 길이가 긴 문서에서의 키프레이즈(keyphrase) 추출을 목표로 한다. 기존의 키프레이즈 추출 방법들은 보통 단기 문서(최대 512 tokens)에 초점을 맞추고 있어 긴 문서 처리에 한계가 있었다. LongKey는 96,000 tokens까지 처리할 수 있는 Longformer 모델을 활용하여 이러한 한계를 극복한다.

- **Technical Details**: LongKey의 방법론은 세 가지 단계로 구성되어 있다: 초기 단어 임베딩(initial word embedding), 키프레이즈 후보 임베딩(keyphrase candidate embedding), 및 후보 점수 매기기(candidate scoring)이다. Longformer 모델을 활용하여 긴 문서의 구문적 세부사항을 캡처하는 임베딩을 생성한다. 길이가 8,192 tokens을 초과하는 문서는 동등한 크기로 분할 처리되어 각각의 임베딩이 결합되어 하나의 통합된 표현을 생성한다.

- **Performance Highlights**: LongKey는 기존의 비지도 학습 및 언어 모델 기반의 키프레이즈 추출 방법들보다 우수한 성능을 보여준다. 다양한 데이터셋에서 테스트한 결과, LongKey는 키프레이즈 추출의 정확성을 크게 향상시키며, 긴 문서에서의 정보 검색 및 관리에 기여할 수 있을 것으로 기대된다.



### Arabic-Nougat: Fine-Tuning Vision Transformers for Arabic OCR and Markdown Extraction (https://arxiv.org/abs/2411.17835)
Comments:
          7 pages, 1 figure

- **What's New**: 아랍어 책 페이지를 구조화된 Markdown 텍스트로 변환하는 OCR 모델인 Arabic-Nougat를 소개합니다. 이 모델들은 Meta의 Nougat 아키텍처를 기반으로 구성되며, arabic-small-nougat, arabic-base-nougat, arabi-large-nougat의 세 가지 전문 모델로 이루어져 있습니다. 아랍어 책 페이지와 해당 Markdown 표현 간의 13.7k 쌍으로 구성된 synthetic dataset인 arabic-img2md를 사용하여 미세 조정되었습니다.

- **Technical Details**: Arabic-Nougat의 핵심 기술 요소 중 하나는 Aranizer-PBE-86k tokenizer로, 이는 효율적인 토큰화를 위해 설계되었습니다. torch.bfloat16의 정밀도와 Flash Attention 2를 활용하여 학습 및 추론을 최적화했습니다. 이 모델들은 다양한 아랍어 텍스트 레이아웃과 긴 문서 처리를 효과적으로 해결하기 위한 아랍어 특화 개선 사항을 포함하고 있습니다.

- **Performance Highlights**: arabic-large-nougat는 최고의 Markdown 구조 정확도와 최저 문자 오류율을 기록하며 최상의 성능을 달성했습니다. 또한, 8,500권 이상의 책에서 추출한 11억 개의 아랍어 토큰을 포함하는 대규모 데이터셋을 제공하여 아랍어 OCR 연구에 유용한 자원을 제공합니다. 모든 모델과 데이터셋 및 코드가 오픈 소스로 제공되어 연구자들이 자유롭게 활용할 수 있습니다.



### $H^3$Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs (https://arxiv.org/abs/2411.17792)
- **What's New**: 이 논문에서는 pretrained LLMs(사전 학습된 대형 언어 모델)을 인간의 선호에 맞추기 위해 instruction-based datasets(명령어 기반 데이터 세트)로 aligning(정렬)하는 중요성을 다룹니다. 새로운 alignments(정렬) 방식인 $H^3$Fusion이 개발되어, 여러 개별적으로 정렬된 LLM을 결합하여 성능을 향상시키는 방법론을 제안합니다. $H^3$Fusion은 또한 mixture-of-experts(MoE) 방법론을 활용하여 효과적인 응답 생성을 위한 최적의 전문가 선택을 포함합니다.

- **Technical Details**: $H^3$Fusion의 주요 특징은 세 가지로 나뉩니다. 첫째, 여러 개별 모델을 앙상블하여 최종 정렬 모델을 구축하며, 이 과정에서 helpful(도움이 되는), harmless(해가 없는), honest(정직한) 특성을 촉진합니다. 둘째, モE 방법론을 통해 각 모델의 multi-head attention weights(다중 헤드 주의 가중치)를 고정하고 FFN(Feed Forward Network) 계층만 튜닝하여 정렬을 수행합니다. 마지막으로, gating loss(게이팅 손실)와 regularization terms(정규화 항)를 도입하여 모델 성능을 향상시키고 전문가 선택 오류를 제어합니다.

- **Performance Highlights**: 논문의 평가 결과, $H^3$Fusion 모델이 개별적으로 정렬된 모델보다 11.37% 더 유용하고 최신 LLM 앙상블 방법 기준으로 13.77% 더 강한 견고성을 보였습니다. 세 가지 기준 데이터 세트인 Alpaca-Eval, BeaverTails, TruthfulQA에서 extensive evaluations(광범위한 평가)를 수행하여 유용성, 해가 없음, 정직함 측면에서 우수한 성능을 입증하였습니다. 또한, 이 연구는 앙상블 접근 방식을 사용하여 정렬을 수행한 최초의 연구로 의미가 큽니다.



### Efficient Self-Improvement in Multimodal Large Language Models: A Model-Level Judge-Free Approach (https://arxiv.org/abs/2411.17760)
- **What's New**: 본 논문은 MLLMs(다중 모달 대형 언어 모델)의 신뢰성과 강건성을 향상시키기 위한 자가 개선 방법을 제안합니다. 기존 방법들이 MLLMs 자체를 판단자로 사용하는데 따른 높은 계산 비용과 잠재적 문제를 해결하기 위해, 모델 수준의 판단자 없는 자가 개선 프레임워크를 도입했습니다. 이를 통해, 통제 가능한 망상(hallucination) 메커니즘을 활용하여 데이터 품질을 최적화하고, 가벼운 대조 언어-이미지 인코더를 통해 샘플을 평가하여 자가 개선의 경로를 효율화합니다.

- **Technical Details**: 제안된 방법은 통제 가능한 망상 메커니즘을 사용하여 선호 학습(pair) 쌍을 생성하고, 대조적 언어-이미지 인코더를 통해 데이터 품질을 평가합니다. 초기 데이터셋을 생성한 후, CLIPScore를 계산하여 부정 샘플의 점수가 긍정 샘플보다 높은 쌍을 업데이트합니다. 이후, 최적화된 데이터셋을 사용하여 DPO(direct preference optimization) 기법을 통하여 시드 모델을 학습시킵니다. 이 과정을 통해 최종적으로 자가 개선된 모델을 얻게 됩니다.

- **Performance Highlights**: 본 방법은 대규모 벤치마크 및 새롭게 도입한 IC 데이터셋에서 기존 기술들보다 우수한 성능을 보였습니다. 정밀도와 재현율이 개선되었으며, 계산 요구사항이 현저히 낮아졌습니다. 실험 결과는 시드 모델에 비해 IC 및 Object HalBench 데이터셋에서 значный 향상을 확인했습니다.



### SlideSpawn: An Automatic Slides Generation System for Research Publications (https://arxiv.org/abs/2411.17719)
Comments:
          6 pages, 4 figures, 2 tables, 5 equations, 41 references

- **What's New**: 이 논문에서는 연구 문서의 PDF를 입력으로 받아 요약된 내용을 시각적이고 간결한 형식으로 제공하는 프레젠테이션을 생성하는 혁신적인 시스템, SlideSpawn을 제안합니다. 기존의 방법들과는 달리, 이 시스템은 연구 문서 구조의 정보를 활용하여 더 나은 품질의 프레젠테이션을 자동으로 생성할 수 있습니다. 또한, 새로운 데이터셋인 Aminer 9.5K Insights를 소개하여 자동 요약 및 프레젠테이션 생성에 활용할 수 있도록 합니다.

- **Technical Details**: SlideSpawn 시스템은 PDF 문서를 XML 형식으로 변환하여 구조적 정보를 캡처한 후, PS5K 및 Aminer 9.5K Insights 데이터셋을 기반으로 훈련된 머신 러닝 모델을 사용하여 각 문장의 중요도를 예측합니다. 중요한 문장들은 ILP(정수 선형 프로그래밍)를 통해 선택되고, 유사성을 기반으로 클러스터링하여 적절한 제목이 붙여집니다. 선택된 문장 옆에는 관련된 그래픽 요소를 배치하여 최종 슬라이드를 생성합니다.

- **Performance Highlights**: 650개의 문서 및 슬라이드 쌍에 대한 실험 결과, SlideSpawn 시스템은 기존의 방법들보다 더 나은 품질의 프레젠테이션을 생성함을 입증했습니다. 이 시스템은 중요한 텍스트 및 그래픽 요소를 효과적으로 선택하고 적절히 배치하여 연구 결과를 보다 잘 전달할 수 있도록 지원합니다. 이를 통해 연구자들은 프레젠테이션 준비에 소요되는 시간을 크게 절약할 수 있습니다.



### Cross-modal Information Flow in Multimodal Large Language Models (https://arxiv.org/abs/2411.18620)
- **What's New**: 이 연구는 auto-regressive multimodal large language models (MLLMs) 내에서 언어와 시각 정보의 상호작용을 탐색하는 새로운 접근 방식을 제공합니다. 연구진은 이러한 모델에서 정보를 어디서, 어떻게 결합하여 최종 예측을 생성하는지 분석하고자 합니다. 이를 위해 시각 질문 응답(visual question answering) 작업을 중심으로 여러 모델에서 실험을 진행하였습니다.

- **Technical Details**: 연구진은 MLLMs의 서로 다른 층을 통해 정보 흐름을 추적하여, 시각적 정보를 어떻게 통합하는지 조사합니다. 주요 방법은 attention knockout 방식으로 특정 주의 패턴을 차단하여, 시각과 언어 입력 간의 상호작용을 억제하는 것입니다. 이를 통해 일반적인 시각 정보와 구체적인 물체 정보가 문맥에 따라 통합되는 두 가지 단계를 확인하였습니다.

- **Performance Highlights**: 실험 결과, MLLMs에서는 아래층에서 더 일반적인 시각 정보가 질문 토큰의 표현과 결합되고, 중간층에서는 특정 물체와 관련된 정보를 질문에 맞춰 통합하는 과정이 있습니다. 마지막으로, 이 통합된 다중 모달 표현이 최종 예측을 위한 입력 순서의 마지막 위치에 전달되어 정확한 응답을 생성하는 것을 발견했습니다. 이 연구는 MLLMs의 투명성을 향상시키고, 향후 다중 모달 정보 통합에 대한 연구에 기여할 것으로 기대됩니다.



### A Pipeline of Neural-Symbolic Integration to Enhance Spatial Reasoning in Large Language Models (https://arxiv.org/abs/2411.18564)
- **What's New**: 이번 논문은 Large Language Models (LLMs)의 공간적 추론(spatial reasoning) 능력을 향상시키기 위한 새로운 신경-상징적(neural-symbolic) 프레임워크를 제시합니다. 저자들은 ASP(Answer Set Programming)를 기반으로 한 기호 추론 및 LLM과 ASP의 파이프라인을 사용하여 40-50%의 정확성 향상을 달성하였습니다. 또한, 이 연구는 LLM의 공간적 추론 능력을 강화하기 위한 통합된 전략을 제안하여, 벤치마크 데이터셋인 StepGame 및 SparQA에서 LLM의 성능을 크게 향상시켰음을 보여줍니다.

- **Technical Details**: 이 연구에서는 세 가지 전략을 통해 공간적 추론을 평가했습니다: (1) ASP 기반 기호 추론, (2) DSPy를 사용한 LLM + ASP 파이프라인, (3) 사실 및 논리적 규칙. 공간적 추론은 정량적(QSR) 및 정성적(quite spatial reasoning) 추론으로 나뉘며, 이 프레임워크는 LLM이 다양한 질문 유형에 대해 성능을 발휘하도록 함으로써 복잡한 공간적 관계를 처리하는 데 도움을 줍니다. 저자들은 이 방법론이 LLM 아키텍처 전반에 걸쳐 성능을 향상시키는 데 효과적임을 주장합니다.

- **Performance Highlights**: StepGame 데이터셋에서 40-50%, SparQA 데이터셋에서 3-13%의 정확성 향상을 달성함으로써 기계 학습(Machine Learning)과 공간적 추론 분야에서의 가능성을 입증하였습니다. 'LLM + ASP' 파이프라인은 Finding Relations (FR) 및 Finding Block (FB) 질문에서 특히 강력한 성과를 보였으며, 이는 특정 작업의 특징과 구현 전략에 따라 성능이 다를 수 있음을 시사합니다. 이 연구는 기호적 추론 모델을 LLM과 접목하여 더 나은 결과를 도출할 수 있다는 점에서 향후 연구에 중요한 기초 자료를 제공합니다.



### Large Language Model-Brained GUI Agents: A Survey (https://arxiv.org/abs/2411.18279)
- **What's New**: 본 논문은 LLM(brained GUI agents) 기반의 GUI 에이전트에 대한 포괄적인 조사와 함께, 그 역사적 발전, 핵심 구성 요소 및 고급 기술을 탐구합니다. 새로운 LLM 기반 GUI 자동화 에이전트은 사용자의 자연어 요청을 해석하고 GUI 요소를 분석하여 자동으로 작업을 수행할 수 있는 능력을 가집니다. 이러한 혁신은 복잡한 디지털 환경 내에서 대화형 명령을 통해 상황에 맞는 대처를 가능하게 합니다.

- **Technical Details**: 이 논문에서 다루는 LLM 기반 GUI 에이전트는 LLM을 인식 및 추론 핵심 엔진으로 활용하여 유연하고 적응력 있게 작업을 생성, 계획 및 실행합니다. 전통적인 GUI 자동화 방법들은 일반적으로 사전 정의된 규칙에 제한되어 있었으나, LLM 기반 에이전트는 자연어 이해(natural language understanding), 시각 인식(visual recognition), 의사 결정(decision-making)을 통합하여 보다 동적인 상호 작용을 가능하게 합니다. 이러한 시스템들은 다양한 소프트웨어 애플리케이션을 제어할 수 있도록 설계되었습니다.

- **Performance Highlights**: 현실세계에서 LLM 기반 GUI 에이전트의 적용 사례로는 Microsoft Power Automate가 있으며, 사용자는 최소한의 기술 지식으로 워크플로우를 디자인할 수 있습니다. 또한, Microsoft Copilot과 같은 생산성 소프트웨어에 통합된 AI 도우미는 자연어 명령을 애플리케이션 작업으로 변환하여 접근성을 향상시킵니다. 이러한 발전은 다양한 응용 분야에서 LLM 기반 GUI 에이전트의 변혁적인 가능성을 강조하고 있습니다.



### How to Learn a New Language? An Efficient Solution for Self-Supervised Learning Models Unseen Languages Adaption in Low-Resource Scenario (https://arxiv.org/abs/2411.18217)
- **What's New**: 본 논문에서는 저자들이 저자원 언어( low-resource language )의 자동 음성 인식( ASR ) 성능을 개선하기 위해 효율적인 미세 조정(fine-tuning) 방식에 중간 조정(Intermediate Adaptation, IA) 단계를 추가하는 새로운 접근 방식을 제안합니다. 이전에는 주로 언어 자원이 풍부한 모델( high-resource languages )에서 학습된 자가 감독 학습( Self-Supervised Learning, SSL ) 모델을 저자원 언어에 사용할 때 도메인 불일치 문제에 직면하였습니다. 이 솔루션은 기존의 미세 조정 방식에 비해 적은 양의 매개변수만 업데이트하면서도 성능을 향상시킵니다.

- **Technical Details**: 제안한 방법론은 중간 조정(IA) 단계를 포함하여 SSL 모델을 각 저자원 목표 언어에 효과적으로 적응시키는 것을 목표로 합니다. IA 단계에서는 고자원(source languages) 언어의 데이터를 사용하여 어댑터(adapter)와 다운스트림 모델의 초기화를 개선합니다. 이를 바탕으로 파라미터 효율적 미세 조정(Parameter-Efficient Fine-tuning, PEFT)을 수행하여 각 목표 언어에 맞춰 모델을 조정합니다. 이 과정에서 SSL 모델은 항상 고정되어 있어 비용이 낮은 솔루션이 됩니다.

- **Performance Highlights**: ML-SUPERB 데이터세트에서의 실험 결과, 본 방법론은 기존의 효율적인 미세 조정 방식보다 뛰어난 성능을 보였습니다. 특히, 보지 못한 언어에 대한 적응 시 문자/음소 오류율에서 최대 28%의 상대적 개선을 달성하였습니다. 이 연구는 다양한 SSL 모델의 분석과 함께 제안한 소스 언어 선택 방법을 통해 성능 향상을 시도하고 있습니다.



### Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning (https://arxiv.org/abs/2411.18203)
Comments:
          16 pages, 11 figures

- **What's New**: 이 논문에서는 Vision-Language Models(VLMs)의 추론 능력을 향상시키기 위한 새로운 프레임워크인 Critic-V를 소개합니다. Critic-V는 Actor-Critic 구조를 기반으로 하여 Reasoner와 Critic이라는 두 개의 분리된 모듈을 통합합니다. 이를 통해 Critic은 비주얼 및 텍스트 입력을 바탕으로 Reasoner가 생성한 추론 경로에 대해 건설적인 피드백을 제공하여 보다 정확하고 효율적인 추론을 가능하게 합니다.

- **Technical Details**: Critic-V의 Reasoner는 비주얼과 텍스트 입력으로부터 추론 경로를 생성하고, Critic은 이 경로에 대해 자연어 피드백을 제공함으로써 추론 전략을 수정할 수 있도록 하는 역할을 합니다. 이 상호작용 과정은 강화학습 프레임워크에 의해 이론적으로 구동되며, Critic은 스칼라 보상 대신 자연어 비판을 제공하여 보다 섬세한 피드백을 가능하게 합니다. Critic 모델은 Rule-based Reward (RBR)와 같은 데이터를 사용하여 평가 능력을 키우고, Direct Preference Optimization (DPO)을 통해 훈련합니다.

- **Performance Highlights**: Critic-V 프레임워크는 8개의 벤치마크 중 5곳에서 기존의 방법들보다 유의미하게 우수한 성능을 보여주었으며, 특히 추론 정확성과 효율성 분야에서 두드러진 개선을 이루었습니다. 이 접근 방식을 통해 VLM들이 자율주행 및 구현된 지능과 같은 실제 문제를 해결하는 데 있어 신뢰성을 크게 향상시킬 수 있음을 보여줍니다. Critic-V는 외부 비평 모델을 통합하여 추론 과정에서의 오류를 줄이고, VLM의 전반적인 성능을 높일 수 있는 유망한 솔루션을 제안합니다.



### SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation (https://arxiv.org/abs/2411.18138)
Comments:
          Technical report

- **What's New**: SALMONN-omni는 코드가 없는 풀 듀플렉스(Full-duplex) 대화 모델로, 음성을 이해하고 생성할 수 있는 새로운 프레임워크를 제공합니다. 이 모델은 생성된 음성과 배경 소리를 동시에 듣고 이야기하는 능력을 갖추고 있어 인간과 기계 간의 자연스러운 대화를 구현합니다. 기존 모듈화된 시스템과는 달리 SALMONN-omni는 단일 엔드-투-엔드 모델로 설계되어 오류 전파를 제거하였습니다.

- **Technical Details**: SALMONN-omni는 스트리밍 음성 인코더와 대형 언어 모델, 스트리밍 음성 합성기를 통합하여 실시간으로 입력 및 출력 음성을 처리할 수 있습니다. 모델 내에 주기적인 동기화 메커니즘이 도입되어 음향과 텍스트 모달리티의 정렬을 보장합니다. SALMONN-omni는 대화의 동적 상황을 효과적으로 다룰 수 있는 "생각" 메커니즘을 특징으로 하며, 이는 두 개의 특별한 상태 전환 토큰을 활용합니다.

- **Performance Highlights**: 실험 결과, SALMONN-omni는 음성 인식, 음성 향상 및 대화형 질문 응답을 포함한 다양한 음성 작업에서 높은 성능을 발휘하였습니다. 이 모델은 턴-테이킹(turn-taking), 바지(in) 및 에코 캔슬레이션(echo cancellation) 상황을 관리하는 데 뛰어난 성능을 보여줍니다. SALMONN-omni는 코드 없는 최초의 풀 듀플렉스 대화 AI 시스템으로, 향후 다양한 응용 가능한 가능성을 제시합니다.



### JPPO: Joint Power and Prompt Optimization for Accelerated Large Language Model Services (https://arxiv.org/abs/2411.18010)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 사용이 증가함에 따라 발생하는 컴퓨터 자원 소모 및 네트워크 통신 부하 문제를 해결하기 위해 새롭게 Joint Power and Prompt Optimization (JPPO)라는 프레임워크를 제안합니다. JPPO는 작은 언어 모델(SLM)을 기반으로 한 프롬프트 압축과 무선 전력 할당 최적화를 결합하여, 서비스 품질과 자원 효율성을 효과적으로 조화시킵니다. 이 프레임워크는 사용자의 변동적인 요구를 충족시키기 위한 통신 자원과 프롬프트 입력의 균형 잡힌 배치를 가능하게 합니다.

- **Technical Details**: JPPO 프레임워크는 사용자의 단말기에 배치된 SLM을 통해 프롬프트를 압축하고, 심층 강화 학습(Deep Reinforcement Learning) 기법을 활용하여 압축 비율과 전송 전력을 공동 최적화합니다. 이를 통해 네트워크 자원 소모를 줄이고 LLM의 추론 성능을 유지하는 것을 목표로 합니다. SLM은 원본 프롬프트의 중요한 정보를 보존하면서, 효과적으로 압축하여 에너지 소모와 지연 시간을 관리하게 됩니다.

- **Performance Highlights**: 실험 결과, JPPO 프레임워크는 높은 서비스 품질을 유지하면서 비트 오류율을 낮추고, 전력 사용을 최적화합니다. 응답 시간은 약 17% 감소되었으며, 이는 원본 프롬프트의 길이에 따라 달라집니다. 이 연구는 향후 무선 LLM 서비스의 효율적인 배치를 위한 중요한 기초를 제공합니다.



### VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Forma (https://arxiv.org/abs/2411.17991)
Comments:
          9 pages

- **What's New**: 이번 연구는 VideoLLMs(영상 대형 언어 모델)와 사용자 간의 상호작용 형식을 집중적으로 탐구합니다. 전통적인 방식은 사용자가 전체 비디오와 질의를 입력한 뒤 모델이 응답을 생성하는데, 이는 실시간 응답 필요 시나리오에서 제한적입니다. 새로운 비디오-텍스트 듀엣(interaction duet) 형식에서는 비디오가 지속적으로 재생되며 사용자가 언제든지 텍스트 메시지를 삽입할 수 있습니다.

- **Technical Details**: 비디오-텍스트 듀엣 상호작용 형식은 MMDuetIT라는 훈련 데이터 세트를 기반으로 구축되어, 비디오 프레임을 순차적으로 모델에 입력하는 방식입니다. 이 방식은 대화의 주체로서 비디오 스트림을 정의하고, 사용자의 텍스트 메시지가 끝나면 비디오가 계속 재생됩니다. 이를 통해 시간 민감한 작업을 위한 응답 생성이 개선됩니다.

- **Performance Highlights**: MMDuet 모델은 MMDuetIT 훈련 데이터를 사용하여 시간 민감한 작업에서 높은 성능을 발휘합니다. YouCook2의 밀집 비디오 캡셔닝에서 76% CIDEr, QVHighlights 하이라이트 검출에서 90% mAP, Charades-STA의 시간 비디오 기초화에서 25% R@0.5의 성과를 보여주었습니다. 이러한 개선은 비디오가 재생되는 동안 실시간으로 응답하는 능력을 가능하게 합니다.



### Signs as Tokens: An Autoregressive Multilingual Sign Language Generator (https://arxiv.org/abs/2411.17799)
- **What's New**: 본 연구에서는 Signs as Tokens(SOKE)라는 멀티링구얼(다국어) 수어 생성 모델을 제안합니다. 기존 연구들이 수어 생성(Sign Language Generation, SLG)을 시각적 콘텐츠 생성 작업으로 간주한 반면, 이 모델은 언어 모델(pretrained Language Models, LMs)의 도움을 받아 텍스트 입력으로부터 3D 수어 아바타를 생성합니다. 또한, 이 연구는 다양한 신체 부위를 표현할 수 있는 분리된 토큰을 생성하는 방법을 개발하여 수어의 다중 인지 특성을 효과적으로 캡처합니다.

- **Technical Details**: SOKE는 수어의 연속적 신체 동작을 이산화하여 토큰 시퀀스(token sequences)로 변환하는 디커플드 토크나이저를 통합합니다. 이를 통해 수어의 언어적 구조를 더 효과적으로 모델링할 수 있게 됩니다. 연구팀은 미국 수화(ASL)와 중국 수화(CSL)를 포함한 다국어 수어 데이터셋을 사용하여 모델을 훈련시켰으며, 전문적인 태그를 통해 특정 수어 및 신체 부위를 지정할 수 있습니다.

- **Performance Highlights**: SOKE의 성능은 How2Sign 및 CSL-Daily와 같은 두 개의 도전적인 기준에서 최첨단 결과를 달성한 것으로 나타났습니다. 연구에서 제안한 방법은 수어의 복잡한 손 동작을 강조하는 데 중점을 두어, 기존 방법의 한계를 보완합니다. 이로 인해 수어 생성을 위한 통합된 모델이 성공적으로 만들어졌고, 수어 커뮤니티와 일반 커뮤니티 간의 소통 장벽을 줄이는 데 기여할 것으로 기대됩니다.



### Towards Efficient Neurally-Guided Program Induction for ARC-AGI (https://arxiv.org/abs/2411.17708)
- **What's New**: 이번 논문은 프로그램 유도 (program induction) 방식의 일반화 능력을 평가하는 ARC-AGI 데이터셋에 대한 최초의 분석 결과를 제시합니다. 저자들은 다양한 학습 공간(그리드 공간, 프로그램 공간, 변환 공간)을 통한 실험 결과를 공유하며, 특히 LPS(프로그램 공간 학습)가 ARC 평가 세트에서 최고의 성능을 보이는 방법임을 강조합니다. 이러한 연구는 출처가 다른 데이터 (out-of-distribution data)에 대한 일반화 능력을 강화하기 위한 새로운 접근법들을 제안합니다.

- **Technical Details**: 저자들은 DSL(도메인 특정 언어) 내에서 프로그램을 실행할 수 있도록 필요한 모든 원시 함수들을 포함해야 한다고 가정합니다. 각 실험에서 DSL을 활용해 다양한 그리드 간 유사성을 추정하고 이를 통해 프로그램 유도를 수행하는 방식을 살펴보았습니다. LGS(그리드 공간 학습) 방식은 Transformer 모델을 사용하여 그리드 임베딩을 생성하고, 이를 기반으로 효과적인 프로그램 생성이 이루어지도록 했습니다.

- **Performance Highlights**: LPS 접근법은 ARC-AGI 평가에서 가장 좋은 성능을 보였으며, 이는 프로그램 유도 및 일반화의 효율성을 증명합니다. 저자들은 새로운 확률적 프로그램 열거 기반 탐색 알고리즘을 소개하였고, 이는 Transformer 기반의 자기회귀 토큰 시퀀스를 사용하여 기존 n-그램 접근법에 비해 더 나은 성능을 보였습니다. 본 논문에서 제안된 접근 방식은 프로그램 유도의 기존 한계를 극복할 수 있는 가능성을 내포하고 있습니다.



New uploads on arXiv(cs.IR)

### Break the ID-Language Barrier: An Adaption Framework for Sequential Recommendation (https://arxiv.org/abs/2411.18262)
- **What's New**: 최근 대형 언어 모델(LLMs)의 자연어 처리 분야에서의 획기적인 발전이 추천 시스템으로의 탐색을 촉발했습니다. 그러나 LLMs는 도메인 특정 지식이 부족하여 시퀀스 추천에서 핵심적으로 필요한 정보가 결여되어 있는 문제를 지니고 있습니다. 이를 해결하기 위해 새로운 프레임워크 IDLE-Adapter를 제안하여, LLM의 추천 정확도를 향상시키고자 했습니다.

- **Technical Details**: IDLE-Adapter는 사전 훈련된 ID 임베딩을 LLM에 통합하여 추천 품질을 높이는 방안을 제시합니다. 이 프레임워크는 희소한 사용자-아이템 상호작용 데이터를 조밀한 LLM 호환 표현으로 변환하는 네 단계의 과정을 포함합니다: 사전 훈련된 ID 시퀀스 모델, 차원 정렬, 레이어별 임베딩 정제, 그리고 레이어별 분포 정렬을 수행합니다. 또한, 다양한 ID 기반 시퀀스 모델과 LLM 아키텍처 간의 유연한 통합이 가능함을 보여줍니다.

- **Performance Highlights**: IDLE-Adapter는 다양한 공개 데이터셋을 통해 10% 이상의 HitRate@5, 20% 이상의 NDCG@5 향상을 달성하였으며, 이는 현존하는 최신 모델과 비교하여 우수한 성능을 보입니다. 실험 결과는 ID 임베딩에 담긴 추천 도메인 지식을 LLM의 표현력과 결합함으로써 추천 시스템의 효과성을 극대화함을 입증하였습니다. 또한, IDLE-Adapter의 일반화 가능성에 대한 연구도 진행하여 다양한 ID 기반 모델과의 원활한 협업 능력을 증명했습니다.



### Overview of TREC 2024 Biomedical Generative Retrieval (BioGen) Track (https://arxiv.org/abs/2411.18069)
- **What's New**: 대형 언어 모델(LLMs)의 발전으로 인해 생물 의료 분야에서 질문 응답, 문헌 요약 등 여러 작업에서 유의미한 개선이 있었습니다. 그러나 LLM을 사용할 때 발생하는 환각(hallucinations)과 허구(confabulations)는 여전히 큰 도전 과제로 남아 있습니다. 이러한 불완전한 정보는 임상 의사 결정이나 생물 의학 연구 평가와 같은 고위험 상황에서 특히 해로운 영향을 미칠 수 있습니다.

- **Technical Details**: 2024년 TREC에서 우리는 `reference attribution`이라는 과제를 도입하여 LLM이 생물 의료 질문에 대한 답변을 생성할 때 잘못된 진술 생성을 경감하고자 하였습니다. 이 작업에서는 PubMed 문서의 안정적인 버전을 기반으로 질병 관련 질문에 대해 최대 세 개의 참조로 지원되는 답변을 생성해야 합니다. 모든 문서는 PMIDs 형식으로 참조되어야 하며, 각 답변은 최대 30303030개의 문서로 제한됩니다.

- **Performance Highlights**: 참여팀들은 다양한 접근 방식을 채택해 탁월한 성과를 거두었습니다. 예를 들어, 일부 팀은 LLM을 사용하여 질문과 관련된 문서를 검색하고 생성된 답변을 재정렬하여 정확성을 높였습니다. 평가에는 모델이 생성한 답변의 출처를 확인하고 그 질을 측정하는 것이 포함돼, 임상의들이 환자의 건강 질문에 대한 보다 신뢰할 수 있는 정보를 제공할 수 있도록 기여하고자 하였습니다.



### Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2411.18583)
Comments:
          Key Words : T5, SpaCy, Large Language Model, GPT, ROUGE, Literature Review, Natural Language Processing, Retrieval-augmented generation

- **What's New**: 이번 연구에서는 Natural Language Processing (NLP) 기술 및 retrieval-augmented generation (RAG)과 Large Language Model (LLM)을 활용하여 문헌 리뷰의 자동 생성을 위한 여러 접근 방식을 제안하고 비교했습니다. 기존에 수많은 연구 기사가 증가하면서 수동 문헌 리뷰의 어려움이 커졌고, 이에 따른 자동화의 필요성이 증가하고 있습니다. 연구의 주된 목표는 PDF 파일만을 입력으로 받아 자동으로 문헌 리뷰를 생성할 수 있는 시스템을 개발하는 것입니다.

- **Technical Details**: 연구에서는 frequency-based method (spaCy), transformer model (Simple T5), 그리고 Large Language Model (GPT-3.5-turbo)과 결합된 retrieval-augmented generation (RAG) 등 여러 NLP 전략의 효과를 평가했습니다. SciTLDR 데이터 세트를 활용하여 문헌 리뷰 자동 생성을 위한 세 가지 다른 시스템을 구현하는 데 세 가지 독특한 기술이 사용되었습니다. 모든 시스템의 평가에는 ROUGE 점수가 활용되었습니다.

- **Performance Highlights**: 평가 결과, Large Language Model인 GPT-3.5-turbo는 ROUGE-1 점수 0.364로 가장 높은 점수를 기록했습니다. 두 번째는 transformer model이었고, spaCy는 마지막 위치에 있었습니다. 최종적으로, 최상의 성능을 보인 시스템에 대해 그래픽 사용자 인터페이스가 생성되었습니다.



### Isometry pursu (https://arxiv.org/abs/2411.18502)
- **What's New**: 이번 논문에서는 orthonormal column-submatrices를 식별하기 위한 convex 알고리즘인 Isometry pursuit를 소개합니다. 이 알고리즘은 새로운 normalization 방법과 multitask basis pursuit를 결합하여, wide matrix에서의 문제에 효과적으로 적용될 수 있습니다. 특히, Jacobian을 활용하여 isometric embedding을 더 잘 식별할 수 있도록 돕는 점이 주목할 만합니다.

- **Technical Details**: Isometry pursuit는 주어진 데이터에서 orthonormal vector를 찾기 위해 설계된 접근 방식입니다. 이 알고리즘은 관찰된 coordinate functions의 Jacobian에서 직접 파생된 isometric embedding을 로그를 통해 식별하는 데 사용됩니다. 이 과정은 기존의 greedy 및 brute force search 방법에 대한 시너지 효과를 제공합니다.

- **Performance Highlights**: 이 알고리즘은 이론적 및 실험적 결과를 통해 그 유효성이 입증되었습니다. 특히, coordinate selection과 diversification 문제를 다룰 때 우수한 성능을 보이며, 기존의 방법들에 비해 더 나은 결과를 도출할 수 있습니다.



### Delineating Feminist Studies through bibliometric analysis (https://arxiv.org/abs/2411.18306)
Comments:
          2 tables, 5 figures

- **What's New**: 이 논문은 페미니스트 연구(Feminist Studies)와 LGBTQIA+ 사회운동을 포함하는 다양한 과학 분야에서 성(gender)/성별(sex) 관련 출판물을 식별하는 새로운 접근법을 제안합니다. Dimensions 데이터베이스를 활용하여 과학 출판물의 데이터셋을 구성하고, 이를 통해 젠더 연구(Gender Studies)의 다른 분야에 대한 영향을 분석합니다.

- **Technical Details**: 이 연구는 두 단계로 구성된 방법론을 기반으로 합니다. 첫 번째 단계는 전문 저널의 핵심 정보를 수집하고, 두 번째 단계는 제목에 대한 포괄적인 키워드 검색을 수행하는 것입니다. 이런 방식은 기본 키워드 검색의 한계를 뛰어넘어 수작업 키워드 열거로 발생할 수 있는 편향(bias)을 완화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 최종 데이터셋은 1668년부터 2023년까지 출판된 190만 개 이상의 과학 문서를 포함하고 있습니다. 이 데이터셋은 다루어진 주제, 인용(citation) 및 협업(collaboration) 역학, 기관(institutional) 및 지역(region) 참여를 분석하는 데 사용될 수 있습니다. 이러한 방법론적 접근은 다른 학문 경계를 구분하기 어려운 연구 분야를 delineate하는 데에도 응용될 수 있습니다.



### The Rn-index: a more accurate variant of the Rk-index (https://arxiv.org/abs/2411.18161)
Comments:
          6 pages; 2 figures; 4 tables

- **What's New**: 이번 논문에서는 Rk-index와 Rn-index라는 새로운 지표가 제안되었습니다. 특히 Rn-index는 Rk-index의 한계를 보완하여 국가 및 기관의 연구 기여도를 보다 정확히 평가하려고 합니다. 이러한 기여도 평가는 전세계적인 발전에서 높은 비율을 차지하는 미국과 중국과 같은 국가들에게 특히 중요합니다.

- **Technical Details**: Rn-index의 계산 방법은 간단합니다. 연구 논문의 인용 수에 따라 정렬된 후, 각 논문의 지역(rank)와 글로벌(rank)을 비교하여 그 비율을 합산하는 방식으로 진행됩니다. 이 방법은 Rn-index의 적용을 쉽게 만들며, 복수의 기여를 가진 경우에도 부분적으로 계산할 수 있는 장점이 있습니다.

- **Performance Highlights**: Rn-index는 다양한 기술 분야에서 연구 기여를 평가하는 데 유용한 지표로 자리잡을 것으로 기대됩니다. 특히, Rn-index가 기존의 bibliometric 지표들로는 드러나지 않는 기여도를 명확히 나타낼 수 있다는 점에서 학계와 정책입안자들에게 중요한 도구가 될 것입니다.



### DuMapper: Towards Automatic Verification of Large-Scale POIs with Street Views at Baidu Maps (https://arxiv.org/abs/2411.18073)
- **What's New**: 이 논문에서는 Baidu Maps를 위한 대규모 POI(Point of Interest) 검증을 위한 자동 시스템인 DuMapper를 제시합니다. DuMapper는 거리뷰 데이터(multimodal street-view data)를 활용하여 POI 데이터베이스를 효과적으로 검증하고, 이를 통해 검증 속도를 50배 향상시킬 수 있습니다. 이 시스템은 자동화된 방식으로 POI의 정확성을 검증하며, 상당한 노동 비용을 절감할 수 있는 가능성을 보여줍니다.

- **Technical Details**: DuMapper 시스템은 세 단계의 파이프라인으로 구성되어 있으며, 첫 번째 단계에서는 geo-spatial index(GSI)를 통해 거리뷰 사진에서 촬영된 좌표를 사용하여 후보 POI를 찾습니다. 두 번째 단계에서는 Optical Character Recognition(OCR) 기술을 활용하여 후보 POI의 이름을 인식하고, 마지막 단계에서는 다양한 멀티모달 특징(multimodal features)을 기반으로 후보 POI를 랭킹하여 최종 POI를 선택합니다. DuMapper II 버전에서는 Deep Multimodal Embedding(DME)과 Approximate Nearest Neighbor(ANN) 검색을 통해 검증 속도를 더욱 향상시킵니다.

- **Performance Highlights**: DuMapper는 출시 이후 3.5년 동안 4억 5백만 회 이상의 POI 검증을 수행하여 약 800명의 전문가와 동일한 작업량을 처리하였습니다. DuMapper II는 자동 POI 검증의 처리량을 5050배 증가시키는 것을 입증하였으며, 이로 인해 Baidu Maps의 생산성과 효율성이 크게 개선되었습니다. DuMapper II의 소스 코드는 GitHub를 통해 공개되어 재현 시험이 가능합니다.



### LongKey: Keyphrase Extraction for Long Documents (https://arxiv.org/abs/2411.17863)
Comments:
          Accepted for presentation at the 2024 IEEE International Conference on Big Data (IEEE BigData 2024). Code available at this https URL

- **What's New**: 이 논문은 LongKey라는 새로운 프레임워크를 소개하며, 주로 길이가 긴 문서에서의 키프레이즈(keyphrase) 추출을 목표로 한다. 기존의 키프레이즈 추출 방법들은 보통 단기 문서(최대 512 tokens)에 초점을 맞추고 있어 긴 문서 처리에 한계가 있었다. LongKey는 96,000 tokens까지 처리할 수 있는 Longformer 모델을 활용하여 이러한 한계를 극복한다.

- **Technical Details**: LongKey의 방법론은 세 가지 단계로 구성되어 있다: 초기 단어 임베딩(initial word embedding), 키프레이즈 후보 임베딩(keyphrase candidate embedding), 및 후보 점수 매기기(candidate scoring)이다. Longformer 모델을 활용하여 긴 문서의 구문적 세부사항을 캡처하는 임베딩을 생성한다. 길이가 8,192 tokens을 초과하는 문서는 동등한 크기로 분할 처리되어 각각의 임베딩이 결합되어 하나의 통합된 표현을 생성한다.

- **Performance Highlights**: LongKey는 기존의 비지도 학습 및 언어 모델 기반의 키프레이즈 추출 방법들보다 우수한 성능을 보여준다. 다양한 데이터셋에서 테스트한 결과, LongKey는 키프레이즈 추출의 정확성을 크게 향상시키며, 긴 문서에서의 정보 검색 및 관리에 기여할 수 있을 것으로 기대된다.



### SlideSpawn: An Automatic Slides Generation System for Research Publications (https://arxiv.org/abs/2411.17719)
Comments:
          6 pages, 4 figures, 2 tables, 5 equations, 41 references

- **What's New**: 이 논문에서는 연구 문서의 PDF를 입력으로 받아 요약된 내용을 시각적이고 간결한 형식으로 제공하는 프레젠테이션을 생성하는 혁신적인 시스템, SlideSpawn을 제안합니다. 기존의 방법들과는 달리, 이 시스템은 연구 문서 구조의 정보를 활용하여 더 나은 품질의 프레젠테이션을 자동으로 생성할 수 있습니다. 또한, 새로운 데이터셋인 Aminer 9.5K Insights를 소개하여 자동 요약 및 프레젠테이션 생성에 활용할 수 있도록 합니다.

- **Technical Details**: SlideSpawn 시스템은 PDF 문서를 XML 형식으로 변환하여 구조적 정보를 캡처한 후, PS5K 및 Aminer 9.5K Insights 데이터셋을 기반으로 훈련된 머신 러닝 모델을 사용하여 각 문장의 중요도를 예측합니다. 중요한 문장들은 ILP(정수 선형 프로그래밍)를 통해 선택되고, 유사성을 기반으로 클러스터링하여 적절한 제목이 붙여집니다. 선택된 문장 옆에는 관련된 그래픽 요소를 배치하여 최종 슬라이드를 생성합니다.

- **Performance Highlights**: 650개의 문서 및 슬라이드 쌍에 대한 실험 결과, SlideSpawn 시스템은 기존의 방법들보다 더 나은 품질의 프레젠테이션을 생성함을 입증했습니다. 이 시스템은 중요한 텍스트 및 그래픽 요소를 효과적으로 선택하고 적절히 배치하여 연구 결과를 보다 잘 전달할 수 있도록 지원합니다. 이를 통해 연구자들은 프레젠테이션 준비에 소요되는 시간을 크게 절약할 수 있습니다.



New uploads on arXiv(cs.CV)

### Textured Gaussians for Enhanced 3D Scene Appearance Modeling (https://arxiv.org/abs/2411.18625)
Comments:
          Project website: this https URL

- **What's New**: 최근 3D Gaussian Splatting (3DGS)은 고품질 결과와 빠른 훈련(training) 및 렌더링(rendering) 시간 덕분에 최신 3D 재구성 및 렌더링 기법으로 떠올랐습니다. 그러나 기존의 Gaussian 모델은 동일한 Gaussian에 의해 커버되는 픽셀들이 동일한 색상으로 표현되어야 하는 제약이 있습니다. 이에 본 연구에서는 전통적인 그래픽스의 텍스처(texture)와 알파 맵핑(alpha mapping)에서 영감을 받아 3DGS에 통합할 새로운 일반화된 Gaussian 외관 표현을 제안합니다.

- **Technical Details**: 제안된 방법은 각 Gaussian에 알파(A), RGB 또는 RGBA 텍스처 맵을 추가하여 공간적으로 변하는 색상과 불투명도를 모델링할 수 있게 합니다. 이렇게 함으로써 각 Gaussian은 단순한 색상 및 타원형 엘립소이드 대신, 다양한 텍스처 패턴과 기하학적 구조를 표현할 수 있게 되었습니다. 흥미롭게도, 알파 전용 텍스처 맵을 사용함으로써 Gaussian의 표현력이 대폭 향상될 수 있으며, RGB 텍스처 맵을 추가함으로써 가장 높은 표현력을 달성할 수 있음을 발견하였습니다.

- **Performance Highlights**: 제안한 방법은 여러 표준 벤치마크 데이터셋과 저희의 커스텀 캡처를 통해 검증되었습니다. 기존 방법과 비교했을 때, 유사하거나 더 적은 수의 Gaussian을 사용하면서도 이미지 품질이 개선됨을 보여 주었습니다. 따라서 이 방법을 통해 3D 재구성과 렌더링의 정확한 표현과 향상된 시각적 품질을 기대할 수 있습니다.



### GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data (https://arxiv.org/abs/2411.18624)
Comments:
          Project page: this https URL

- **What's New**: 새로운 연구에서는 GeneMAN이라는 일반화 가능한 이미지-3D huMAN 재구성 프레임워크를 제안합니다. 이 프레임워크는 고품질 인간 데이터를 기반으로 하여 단일 사진에서도 고충실도 3D 인간 모델을 생성할 수 있습니다. GeneMAN은 인간의 특정 텍스트-이미지 디퓨전 모델을 활용하여 2D 및 3D 재구성을 위해 더욱 일반화된 프라이어(prior) 모델을 제공합니다. 결국 GeneMAN은 다양한 신체 비율이나 자연스러운 포즈에서도 신뢰할 수 있는 3D 모델을 생성할 수 있습니다.

- **Technical Details**: GeneMAN은 세 가지 주요 모듈로 구성되어 있습니다. 첫째, 인간 특화 텍스트-이미지 디퓨전 모델과 뷰 조건 디퓨전 모델을 통해 인간 재구성에 필요한 2D 및 3D 프라이어를 학습합니다. 둘째, Geometry Initialization-&-Sculpting 파이프라인이 단일 이미지를 기반으로 고품질 3D 인간 기하학을 복구합니다. 셋째, Multi-Space Texture Refinement 파이프라인을 통해 3D 텍스처를 정교하게 다듬어 최종적으로 고충실도의 텍스처를 얻습니다.

- **Performance Highlights**: 실험 결과 GeneMAN은 기존의 최첨단 방법들보다 뛰어난 성능을 보였으며, 다양한 의복과 포즈, 개인 소지품이 있는 경우에도 고품질 3D 인간 모델을 정확히 재구성할 수 있음을 입증했습니다. 특히, GeneMAN은 자연 체형을 가진 입력 이미지에 대해 높은 재현성을 보여주며, 단일 입력 이미지로도 복잡한 3D 인체 모델을 만들 수 있습니다. 이러한 특징 덕분에 GeneMAN은 실제 세계의 다양한 사진에 효과적으로 적용될 수 있습니다.



### Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation (https://arxiv.org/abs/2411.18623)
- **What's New**: Lift3D 프레임워크는 2D 기반 모델을 활용하여 3D 조작 정책을 구성하는 혁신적인 접근법을 제시합니다. 이 연구는 로봇 조작에서 필수적인 3D 정보의 효율적인 처리를 위해 자기 지도 학습(self-supervised learning)을 통해 2D 이미지를 사용하여 3D 기하학 정보를 재구성합니다. 또한, 이 프레임워크는 대규모로 사전 학습된 2D 모델의 지식을 활용하여 포인트 클라우드 데이터를 직접 인코딩하는 방식을 통해 3D 표현을 명확히 하고, 공간 정보 손실을 최소화합니다.

- **Technical Details**: Lift3D의 핵심은 작업 인식 마스크 오토인코더(task-aware masked autoencoder)를 통해 3D 정보를 재구성하는 것입니다. 이를 통해, 2D 기반 모델의 내재된 3D 로봇 표현을 증강하고, 포인트 클라우드 데이터를 2D 모델의 위치 임베딩(positional embeddings)과 연계하여 매핑하는 2D 모델 리프팅 전략을 개발합니다. 이를 통해, 이전의 방법들보다 향상된 3D 공간 인식을 제공하며, 로봇의 조작 능력을 강화합니다.

- **Performance Highlights**: Lift3D는 다양한 시뮬레이션 벤치마크와 실제 시나리오에서 기존의 최첨단 방법들보다 일관되게 높은 성능을 보여줍니다. 예를 들어, Meta-World 및 Adroit 벤치마크에서 각각 18.2% 및 21.3%의 성공률 향상을 기록하며, 단일 뷰 포인트 클라우드를 사용할 때조차 뛰어난 조작 능력을 입증합니다. 실제 실험에서는 30회 에피소드만으로도 새로운 조작 기술을 학습할 수 있는 능력을 가지고 있습니다.



### Leveraging Semi-Supervised Learning to Enhance Data Mining for Image Classification under Limited Labeled Data (https://arxiv.org/abs/2411.18622)
- **What's New**: 이번 연구에서는 21세기 정보 시대에 빅데이터 기술이 발전함에 따라, 대규모 및 고차원 데이터를 효과적으로 분석하고 가치 있는 정보를 추출하는 것이 중요한 과제가 됨을 강조하고 있습니다. 기존의 데이터 마이닝 방법들은 라벨이 부족한 상황에서 성능이 크게 제한되는데, 이를 극복하기 위해 반감독 학습(semi-supervised learning) 방법을 도입했습니다.

- **Technical Details**: 연구에서는 자기 학습(self-training) 방법을 채택하고, 이미지 특징 추출 및 분류를 위해 합성곱 신경망(convolutional neural network, CNN)을 결합하여 알고리즘의 성능을 향상시켰습니다. 이러한 접근법을 통해 라벨이 제한된 상황에서도 비라벨 데이터(unlabeled data)를 효과적으로 이용할 수 있도록 하는 것이 목표입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 CIFAR-10 이미지 분류 데이터셋에서 서포트 벡터 머신(Support Vector Machine, SVM), XGBoost, 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 등의 전통적인 기계 학습 기법들보다 월등한 성능을 보임을 확인했습니다. 정확도, 리콜(recall), F1 점수 등 주요 성능 지표에서 눈에 띄는 개선이 이루어졌으며, 다양한 노이즈 수준에서도 반감독 CNN 모델의 강건성(robustness) 및 노이즈 저항성(noise-resistance) 능력이 검증되었습니다.



### Diffusion Self-Distillation for Zero-Shot Customized Image Generation (https://arxiv.org/abs/2411.18616)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Diffusion Self-Distillation이라는 새로운 방법론을 제안하여, 텍스트-이미지 변환 모델을 활용한 이미지 생성 시 보다 정밀한 제어를 가능하게 합니다. 최근 텍스트-이미지 생성 모델의 발전을 바탕으로, 이러한 방법은 높은 품질의 커스텀 이미지를 생성하기 위한 데이터셋을 자체적으로 만들어낼 수 있습니다. 기존의 방법들이 가지고 있는 대규모의 페어링 데이터 부족 문제를 해결하였으며, 이는 다양한 응용 분야에서 유망한 잠재력을 가지고 있습니다.

- **Technical Details**: Diffusion Self-Distillation은 사전 훈련된 텍스트-이미지 모델을 활용하여, 다수의 일관된 이미지를 생성하고 이 데이터를 기반으로 텍스트-이미지 모델을 미세 조정(fine-tune)하는 방식으로 작동합니다. 이 과정에서 Vision-Language Models(VLMs)를 사용하여 이미지 그리드를 자동으로 선별하고, 커스터마이징된 이미지 생성의 필요성에 대해 텍스트-이미지 변환 모델의 효과를 극대화합니다. 또한, 새로운 병렬 처리 아키텍처를 적용하여 효율적인 모델 구성을 수행합니다.

- **Performance Highlights**: Diffusion Self-Distillation은 기존의 제로샷(zero-shot) 방법들보다 우수한 성능을 보여주며, 각 인스턴스 조정 기술에 필적하는 결과를 제공합니다. 이 방법은 고유 정체성을 유지하면서도 다양한 맥락에서의 이미지 생성을 지원합니다. 최종적으로, 이 논문은 아티스트들이 작업을 효과적으로 반복하고 적응할 수 있도록 하여 창의적인 자유를 증진시키는 도구로서의 가능성을 입증합니다.



### CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models (https://arxiv.org/abs/2411.18613)
Comments:
          Project page: this https URL

- **What's New**: CAT4D는 단일 모노큘러 비디오로부터 4D(동적 3D) 장면을 생성하는 새로운 방법을 제시합니다. 이 모델은 다양한 데이터셋을 기반으로 훈련된 멀티뷰 비디오 디퓨전 모델을 활용하여 여러 카메라 시점과 타임스탬프에서 새로운 시각 합성을 가능하게 합니다. 매력적인 점은 사용자가 투입하는 단일 비디오로부터도 동적 3D 장면을 정밀하게 복원할 수 있는 것입니다.

- **Technical Details**: CAT4D는 두 단계로 이루어진 접근 방식을 채택합니다. 첫 단계에서는 단일 모노큘러 비디오를 멀티뷰 비디오로 변환하고, 두 번째 단계에서 이 생성된 멀티뷰 비디오를 사용하여 동적 3D 장면을 최적화합니다. 모델은 다양한 정적 장면의 멀티뷰 이미지와 동적 비디오를 혼합하여 학습하였으며, 이로 인해 단일 비디오로부터 신뢰할 수 있는 결과를 생성할 수 있습니다.

- **Performance Highlights**: CAT4D는 다양한 과제에서 경쟁력 있는 성능을 발휘하며, 기존의 최첨단 모델들과 유사한 결과를 보여주는 것을 목표로 합니다. 특히, 사용자로부터 수집된 스파스 뷰 입력 이미지를 통해 정밀한 3D 복원을 수행할 수 있으며, 동적 장면 생성에서도 기존 방법들에 비해 상당한 개선을 이루었습니다. 또한, 고정 뷰포인트 동영상을 기반으로 여러 동적 객체를 포함하는 장면 생성을 가능하게 합니다.



### Structured light with a million light planes per second (https://arxiv.org/abs/2411.18597)
- **What's New**: 본 논문에서는 1초에 천 개의 프레임을 캡처할 수 있는 구조광 시스템을 소개합니다. 이는 이전 기술의 네 배 빠른 속도를 자랑하며, 이 혁신의 핵심은 최대 초당 200만 개의 광면을 스캔할 수 있는 아쿠스토-옵틱(acousto-optic) 조명 스캐닝 장치의 설계입니다. 이 시스템은 이벤트 카메라와 결합하여 깊이 삼각측정을 수행하며, 구조광을 활용하는 새로운 접근법을 제안합니다.

- **Technical Details**: 아쿠스토-옵틱 조명 스캐닝 장치는 이벤트 카메라와 통합되어 작동하며, 카메라가 액티베이트되는 희소 이벤트(sparse events)를 이용하여 장면을 스위프(sweep)하면서 깊이를 측정합니다. 이전 연구에서는 조명 스캐닝이 빠른 구조광 작동의 병목현상이 되었으나, 본 기술은 이벤트 카메라의 전체 프레임 대역폭보다 세 배 빠릅니다. 이를 통해 이벤트 카메라의 고속 작동의 이점을 최대한 활용할 수 있습니다.

- **Performance Highlights**: 본 시스템은 단순히 전체 영역을 스캔하는 대신 관심 영역만을 적응형으로 스캔하여 속도를 더욱 증가시킵니다. 적응형 스캐닝은 이론적인 전체 프레임 한계보다 한 차원 빠른 비율로 진행됩니다. 이러한 발전으로 인해, 구조광 시스템의 성능이 크게 향상되었습니다.



### Hierarchical Information Flow for Generalized Efficient Image Restoration (https://arxiv.org/abs/2411.18588)
- **What's New**: 본 논문에서는 이미지 복원(image restoration, IR) 작업의 효율적인 일반화 및 확장을 위한 계층적 정보 흐름 메커니즘을 제안합니다. 이 방법은 Hi-IR이라는 이름으로, 픽셀 간에 정보를 점진적으로 전파하여 다양한 IR 작업에서의 성능을 향상시킵니다. 또한, Hi-IR은 기존 IR 접근 방식의 한계를 극복하고, 대규모 훈련 설정에서의 효과적인 모델 확장을 도모합니다.

- **Technical Details**: Hi-IR은 세 가지 수준으로 구성된 계층적 정보 나무를 구축하여 이미지 복원을 진행합니다. 첫 번째 수준에서는 개별 패치 내에서 지역 정보를 교환하고 중간 노드 패치를 생성합니다. 두 번째 수준에서는 이 중간 노드 패치 간의 정보 전파를 통해 더 넓은 범위의 정보를 통합하며, 세 번째 수준에서는 이질적인 노드 패치들 간의 연결을 이루어 최종 이미지 복원을 도와줍니다.

- **Performance Highlights**: 지난 연구 결과들에 따르면, Hi-IR은 7가지 일반적인 이미지 복원 작업에서 최첨단 성능을 달성하여 그 효율성과 일반성을 입증했습니다. 다양한 손상 유형 및 강도에서 훈련된 모델의 성과를 평가해 본 결과, Hi-IR은 단일 모델로도 여러 IR 작업에 효과적으로 일반화할 수 있는 가능성을 보여주었습니다. 이는 기존의 접근 방식에 비해 뛰어난 성능과 더불어 다양한 IR 작업에서의 응용 가능성을 나타냅니다.



### Exploring Depth Information for Detecting Manipulated Face Videos (https://arxiv.org/abs/2411.18572)
Comments:
          12 pages, 10 figures. arXiv admin note: substantial text overlap with arXiv:2212.14230

- **What's New**: 이번 연구에서는 얼굴 깊이 맵을 보조 정보로 활용하여 얼굴 조작 탐지의 강인성을 높이는 가능성을 탐구합니다. 일반적으로 얼굴 조작 탐지에서 잘 다루어지지 않았던 이 특성을 활용함으로써, 새로운 접근 방식을 제안합니다. 제안된 방법은 얼굴 깊이 맵의 지역적 변형을 캡처하기 위해 Face Depth Map Transformer (FDMT)를 사용합니다.

- **Technical Details**: 제안된 방법에서 FDMT는 RGB 얼굴 이미지로부터 얼굴 깊이 맵을 패치 단위로 추정하는데, 이는 조작으로 인해 발생하는 지역 깊이 이상을 감지하여 보조 정보로 활용합니다. Multi-head Depth Attention (MDA) 메커니즘을 통해 얻게 된 깊이 맵을 여러 백본 모델에 통합하여 얼굴 조작 탐지의 정확성을 향상시킵니다. 또한 RGB-Depth Inconsistency Attention (RDIA) 모듈을 통해 다중 프레임 입력의 서로 다른 공간-시간 불일치를 효과적으로 측정하고 통합합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 기존 방식들보다 크로스 데이터베이스 시나리오에서 탁월한 성능을 보이고 있으며, 인트라 데이터베이스 시나리오에서도 안정적인 결과를 달성하였습니다. Xception, ResNet50, EfficientNet 등의 다양한 얼굴 조작 탐지 백본에서 제안된 방법의 일반화 성능이 입증되었습니다. 이러한 성능 향상은 얼굴 조작 탐지의 신뢰성을 크게 높이는 데 기여할 것입니다.



### FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion (https://arxiv.org/abs/2411.18552)
- **What's New**: 이번 연구에서는 기존의 Diffusion 모델이 훈련 중 사용된 해상도에서만 효과적으로 작동한다는 한계를 극복하기 위해 'Frequency and Attention Modulated diffusion (FAM diffusion)'라는 새로운 방법을 제안합니다. 이 방법은 두 개의 모듈인 Frequency Modulation (FM)과 Attention Modulation (AM)을 결합하여 이미지의 전반적인 구조와 세부 텍스처의 일관성을 개선합니다. 특히, 전이 학습을 통해 추가 훈련 없이도 기존의 Latent Diffusion 모델에 원활하게 통합될 수 있는 장점이 있습니다.

- **Technical Details**: FAM diffusion 방법은 native 해상도에서 이미지 생성을 시작하고 이후 테스트 시 diffuse-denoise 전략을 적용하여 고해상도 이미지를 생성합니다. FM 모듈은 Fourier 도메인을 활용하여 저주파 성분의 일관성을 개선하며, AM 모듈은 세부 텍스처의 일관성을 높이기 위해 주목 맵을 활용합니다. 이러한 접근 방식은 기존의 Patch 기반 방식에서 발생하는 latency overhead를 줄이고 보다 확실한 결과를 제공합니다.

- **Performance Highlights**: 실험 결과, FAM diffusion 방법은 구조적 및 지역적 아티팩트 문제를 효과적으로 해결하며, 정량적인 성능 평가에서도 최첨단 성능을 나타냅니다. 고해상도 이미지 생성을 필요로 하는 다양한 어플리케이션에서 우리의 방법이 높은 효과성을 발휘함을 보여주었습니다. 이미지 생성의 일관성을 보장하기 위해 중복 발생 가능한 추론 기술을 피하면서도, 매우 낮은 latency overhead를 유지하는 성과를 달성했습니다.



### PhyCAGE: Physically Plausible Compositional 3D Asset Generation from a Single Imag (https://arxiv.org/abs/2411.18548)
Comments:
          Project page: this https URL

- **What's New**: PhyCAGE는 단일 이미지를 기반으로 물리적으로 타당한 조합 3D 자산 생성을 위한 최초의 접근 방식입니다. 입력 이미지에 대해 일관된 다중 뷰 이미지를 생성한 후, 이 이미지에 3D Gaussian Splatting 표현을 적합시킵니다. 물체를 대표하는 Gaussian들이 물리적으로 호환되도록 최적화하는 Physical Simulation-Enhanced Score Distillation Sampling (PSE-SDS) 기법을 도입하여, 물리적 시뮬레이션을 통해 Gaussian의 위치를 점진적으로 수정합니다.

- **Technical Details**: 이 논문에서는 2D 이미지 입력에 따라 3D 형태를 생성하는 과정을 다룹니다. 기존의 방법론은 주로 단일 물체의 이미지에서 3D를 생성하는데 중점을 두었으나, 본 연구는 두 개의 조합 물체가 포함된 자산의 3D 표현을 각각 생성하는 복잡한 도전을 목표로 합니다. PSE-SDS를 통해 물리적으로 그리드로 된 시뮬레이션을 적용해 최적의 기하학적 형상을 찾아내어 물리적으로 일관된 3D 자산을 생성합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 단일 이미지에서 물리적으로 타당한 조합 3D 자산을 생성할 수 있음을 보여줍니다. 이 연구는 강력한 공간 결합을 가진 대화형 객체에 초점을 맞춰 어떻게 새로운 파이프라인을 설계했는지를 설명합니다. 또한, PSE-SDS는 3D Gaussian을 물리적으로 타당하게 최적화하는데 기여하여, 다운스트림 응용을 위한 3D 조합 자산을 침투 없이 생성하는 첫 번째 사례가 됩니다.



### AdaVLN: Towards Visual Language Navigation in Continuous Indoor Environments with Moving Humans (https://arxiv.org/abs/2411.18539)
- **What's New**: 이번 논문은 Adaptive Visual Language Navigation (AdaVLN)이라는 새로운 과제를 제안합니다. AdaVLN은 실제 환경에서 로봇이 자연어 지침에 따라 복잡한 3D 실내 환경을 탐색하며, 동적으로 움직이는 인간 장애물과의 충돌 회피를 요구합니다. 이를 통해 기존의 정적인 탐색 과제에서 발생할 수 없는 새로운 도전에 직면하게 됩니다.

- **Technical Details**: AdaVLN는 Matterport3D 환경에서 로봇이 연속적인 액션 스페이스를 통해 목표 위치로 이동할 수 있도록 설계되었습니다. 로봇은 각 탐색 단계에서 RGB-D 이미지를 통해 자신의 주변을 인지하고, 제공된 자연어 지침에 따라 다양한 동작을 수행할 수 있습니다. 이 과정에서 로봇은 정적 장애물뿐만 아니라, 동적인 인간 장애물과의 충돌이 없도록 조정해야 합니다.

- **Performance Highlights**: 논문에서는 AdaVLN 과제를 지원하기 위해 AdaVLN 시뮬레이터와 AdaR2R 데이터셋을 소개하였습니다. 다양한 기준 모델을 평가하여 AdaVLN이 시뮬레이션과 실제 환경 간의 격차를 줄일 수 있는 가능성을 보여주며, 공정한 비교와 실험을 위해 'freeze-time' 메커니즘을 도입했습니다. 결과적으로, 이 연구는 로봇의 탐색 능력을 개선하고 동적인 환경에서의 실효성을 증대시킬 수 있는 기회를 제공합니다.



### Utilizing the Mean Teacher with Supcontrast Loss for Wafer Pattern Recognition (https://arxiv.org/abs/2411.18533)
Comments:
          5 pages,1 figures

- **What's New**: 이 연구에서는 반도체 제조 공정 중 웨이퍼 맵 패턴 인식을 위한 새로운 접근법을 제안하였습니다. Mean Teacher 프레임워크와 감독된 대조 학습 손실(Supervised Contrastive Loss)을 통합하여, 제한된 레이블 데이터의 문제를 해결하고 웨이퍼 패턴의 정밀한 분류를 목표로 하였습니다. 이를 통해 높은 정확도와 성능 향상을 달성하였습니다.

- **Technical Details**: 제안한 방법론은 Mean Teacher 알고리즘을 활용하여 레이블이 없는 대량의 데이터를 효과적으로 처리합니다. 이 방법은 두 개의 신경망 인스턴스 간의 예측 일관성을 유지하며, 주 모델은 학생 모델(Student Model)이고, 교사 모델(Teacher Model)은 학생 모델의 EMA를 기반으로 합니다. 또한, 데이터 불균형 문제를 해결하기 위해 SMOTE와 언더 샘플링 기법이 적용되었습니다.

- **Performance Highlights**: 우리의 방법은 WM811K 데이터 세트를 사용하여 기존 방법 대비 Accuracy, Precision, Recall, F1 Score에서 각각 5.46%, 6.68%, 5.42%, 4.53%의 성능 향상을 이루었습니다. 이러한 결과는 제안한 방법의 유효성을 뒷받침하며, 반도체 제조 공정에 있어 인공지능의 활용 가능성을 더욱 확대하고 있습니다.



### Enhancing weed detection performance by means of GenAI-based image augmentation (https://arxiv.org/abs/2411.18513)
- **What's New**: 본 논문은 심층 학습에 기반한 지능형 잡초 관리 시스템을 위한 데이터 증강(data augmentation) 기법의 혁신적 접근을 제시하고 있습니다. 기존의 전통적인 데이터 증강 기법이 부족한 신뢰성과 다양성을 가진 반면, 생성적 AI(generative AI) 기술을 활용하여 잡초 탐지 모델에 대한 훈련 데이터를 축적하고 고품질의 합성 이미지를 생성합니다. 논문에서는 이러한 합성 이미지가 YOLO nano와 같은 실시간 탐지 시스템의 성능에 미치는 영향을 평가합니다.

- **Technical Details**: 논문에서는 데이터 부족 문제를 해결하기 위해 Stable Diffusion 모델을 사용하여 다양한 합성 이미지를 생성하는 방법을 제안합니다. 생성적 적대 신경망(GANs)과 확산 모델(Diffusion Models)이 기존의 이미지 증강 기법에 비해 신뢰성과 자연스러운 다양성을 유지하는 데 효과적임을 강조하고, 이에 따라 합성 이미지가 실제 탐지 시스템에서의 성능 향상에 필요한 데이터를 제공할 수 있음을 설명합니다.

- **Performance Highlights**: 실험 결과에서는 생성적 AI 기반 데이터 증강을 이용하여 훈련된 YOLO 모델의 평균 정밀도(mean Average Precision, mAP) 점수가 크게 향상된 것을 보여줍니다. 특히, mAP50 및 mAP50-95 점수가 기존 데이터 증강 기법보다 현저히 개선되어, 합성 데이터의 활용이 모델의 강건성과 정확도 향상에 매우 기여할 수 있음을 입증합니다.



### GATE OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation (https://arxiv.org/abs/2411.18499)
Comments:
          53 pages, 19 figures

- **What's New**: 이 논문은 OpenING이라는 포괄적인 벤치마크를 소개하며, 여기에는 56개의 실제 시나리오에 대한 5,400개의 고품질 인간 주석 인스턴스가 포함되어 있습니다. 이는 기존의 여러 벤치마크들이 가져지지 못한 다양한 데이터와 쿼리를 제공하여 이미지-텍스트 생성의 도전을 해결할 수 있는 기반을 마련합니다. 또한 새로운 평가 모델인 IntJudge를 통해 기존 GPT 기반 평가자의 성능을 11.34% 향상시켰습니다.

- **Technical Details**: OpenING은 멀티모달 에이전트가 주어진 프롬프트를 바탕으로 이미지와 텍스트를 생성하는 비율을 평가할 수 있는 메커니즘을 제공합니다. 이 메커니즘은 멀티스텝 이미지-텍스트 생성으로 구성되어 있으며, 사용자는 다양한 형식으로 입력을 제공받습니다. IntJudge는 데이터를 평가하기 위한 주석 모델로, 강화된 데이터 파이프라인을 통해 훈련되어 82.42%의 인간 주석 동의율을 달성했습니다.

- **Performance Highlights**: OpenING에서 실시된 실험 결과 우리 모델들이 생성하는 이미지-텍스트 내용에서 고품질의 일관성과 응집력이 여전히 부족하다는 사실이 드러났습니다. 인간 주석이 된 콘텐츠의 품질은 생성된 콘텐츠의 품질보다 현저히 높았습니다. 이 연구는 향후 멀티모달 생성 모델의 발전을 위한 기초적인 데이터를 제공하며, 향후 모델 개선을 위한 주요 발견들을 제시합니다.



### Weakly Supervised Framework Considering Multi-temporal Information for Large-scale Cropland Mapping with Satellite Imagery (https://arxiv.org/abs/2411.18475)
- **What's New**: 이번 연구에서는 대규모 농경지 매핑을 위해 약한 감독 환경(weakly supervised framework)에서 다중 시점 정보(multi-temporal information)를 고려한 방법을 제안합니다. 기존의 원격 감지(remote sensing) 데이터와 심층 학습(deep learning) 기법의 조합은 우수한 성능을 보였으나, 많은 양의 정밀 레이블(precise labels)이 필요하여 노동 집약적입니다. 이에 따라, 우리는 높은 품질의 레이블을 글로벌 토지 피복(Global Land Cover, GLC) 제품 간의 일관성에 따라 추출하여 감독 학습 신호(supervised learning signal)를 구축합니다.

- **Technical Details**: 제안하는 프레임워크는 고품질 레이블의 잔여 오류를 과신하는 문제를 해결하기 위해 농경지(cropland)의 유사성(similarity) 및 집합체(aggregation)를 시각적/공간적 영역에서 인코딩하여 비감독 학습 신호(unsupervised learning signal)를 구성합니다. 이를 통해 감독 부분을 제약하기 위한 정규화 항(regularization term)으로 활용합니다. 또한 고품질 레이블이 없는 샘플에서도 비감독 신호를 포함시켜 특징 공간(feature space)의 다양성을 풍부하게 만듭니다.

- **Performance Highlights**: 이 프레임워크는 후난 성(Hunan Province), 남동 프랑스(Southeast France), 캔자스(Kansas)의 세 연구 지역에서 대규모 농경지 매핑에서 강력한 적응성을 실험적으로 검증했습니다. 다중 시점 정보가 농경지 추출에 어떻게 기여하는지를 밝히기 위해 고차원 생리적 특징(phenological features)을 시각화하였고, 데이터 희소성(data scarcity) 조건에서도 방법의 견고성을 평가했습니다.



### HEMGS: A Hybrid Entropy Model for 3D Gaussian Splatting Data Compression (https://arxiv.org/abs/2411.18473)
- **What's New**: 최근 3D Gaussian Splatting (3DGS) 기술의 발전은 3D 모델링과 이미지 렌더링에서 널리 사용되고 있으나, 데이터 저장 및 전송에서 큰 도전 과제가 발생했습니다. 본 논문에서는 Gaussian Splatting (HEMGS) 데이터 압축을 위한 하이브리드 엔트로피 모델을 제안하여, 고도로 압축된 3DGS 표현을 가능하게 합니다. 이 모델은 하이퍼프라이어 네트워크와 오토 회귀 네트워크의 두 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: HEMGS는 속성 간의 구조적 중복을 효과적으로 줄이기 위해 점진적 코딩 알고리즘을 적용하여 하이퍼프라이어 특징을 생성합니다. 이 과정에서는 기존에 압축된 속성과 위치 정보를 우선 정보로 사용하여, 도메인 인식 및 인스턴스 인식 아키텍처를 채택하여 압축된 속성으로부터 위치 특징을 추출합니다. 또한, 각 속성 내의 중복성을 줄이기 위해 인접한 압축 요소 간의 관계를 활용하는 오토 회귀 네트워크를 통합하여 효율적인 엔트로피 코딩을 실현합니다.

- **Performance Highlights**: 본 연구 결과는 4개의 벤치마크에서 HEMGS를 포함한 3DGS 압축 프레임워크의 효과성을 입증しています. 실험을 통해 우리의 방법이 기준 방법보다 약 40%의 평균 크기 감소를 달성하면서 렌더링 품질을 유지하며, 최신 압축 성능을 오랜 쇼 알고리즘과 비교해도 우수하다는 것을 확인했습니다. 이는 3DGS 데이터 압축에서 상태-of-the-art 성능을 제공함을 보여줍니다.



### Complexity Experts are Task-Discriminative Learners for Any Image Restoration (https://arxiv.org/abs/2411.18466)
- **What's New**: 이 논문에서는 이미지 복원 문제를 해결하기 위해 'complexity experts'라는 개념을 도입합니다. 기존의 mixture-of-experts (MoE) 모델의 한계를 극복하기 위해, 각 expert 블록의 계산 복잡도와 수용 필드를 다르게 설정했습니다. 이를 통해 특정 작업에 맞는 전문가에게 태스크를 효율적으로 할당하여 이미지 복원 품질을 높였습니다.

- **Technical Details**: 제안된 MoCE-IR 모델은 U자형 아키텍처를 기반으로 하고 있으며, 비대칭 인코더-디코더 설계를 따릅니다. 디코더 블록에는 새로운 MoCE 레이어가 통합되어 있으며, 입력의 깊은 특성을 추출하기 위해 3×3 합성곱을 사용합니다. Sobel 필터를 활용한 고주파 신호 가이드를 통해 주파수 인식을 향상시켜, 복원 결과의 품질을 높이고 있습니다.

- **Performance Highlights**: 실험 결과, MoCE-IR 모델은 최신 방법들과 비교해 우수한 성능을 보여주며, 비관련 전문가를 우회하는 효율적인 추론을 달성했습니다. 복원 품질과 계산 효율성을 모두 향상시키며, 실세계 응용에서의 실용성을 입증했습니다. 이로 인해 모든 작업을 통합하여 처리하는 새로운 기준을 확립하게 되었습니다.



### Neural Image Unfolding: Flattening Sparse Anatomical Structures using Neural Fields (https://arxiv.org/abs/2411.18415)
- **What's New**: 이번 연구에서는 복잡한 희박 구조물을 2D로 펼치는 문제를 해결하기 위해 신경 필드를 이용한 모듈형 접근 방식을 제안합니다. 기존의 전통적인 방법들이 요구하는 밀집 표면 샘플링 없이도 3D 물체를 효과적으로 2D 표현으로 전환할 수 있는 기술을 발전시켰습니다. 이는 의료 진단의 정확성과 효율성을 극대화하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된 방법은 희박한 점 집합을 기반으로 왜곡 최소화를 목표로 하여 2D 매니폴드를 적합시키는 과정으로 시작됩니다. 다중 스케일 거리 왜곡 규제를 통해 해상도에 독립적인 결과를 얻으며, 기하학적 손실과 이미지 기반 손실을 결합하여 이미지 도메인에서 바람직한 시각적 품질을 동시에 최적화합니다. 이를 통해 신경 필드를 이용한 이미지 변환 적합이 가능해집니다.

- **Performance Highlights**: 이 기술은 메쉬 기반 기준과 비교하여 희박 구조에 대해 최고 왜곡 수치에서 더 나은 성능을 보여줍니다. 또한, 이전 방법들과 비교할 때 훨씬 매끄러운 변환 결과를 가져오는 정규화 방법이 뚜렷이 향상되었습니다. 다양한 임상 응용 프로그램에서 이 기술을 적용하여, 중요한 해부학적 구조에 대한 직관적이고 명확한 시각화를 지원합니다.



### Adaptive Blind All-in-One Image Restoration (https://arxiv.org/abs/2411.18412)
Comments:
          17 pages

- **What's New**: 이 논문에서는 Adaptive Blind All-in-One Image Restoration (ABAIR) 방법을 제안합니다. ABAIR는 다양한 손상(degradation)을 효과적으로 처리하며, 실험에서 보지 못한 손상도 잘 일반화할 수 있습니다. 이 모델은 소량의 파라미터를 이용하여 새로운 손상을 통합할 수 있는 구조로 설계되었습니다.

- **Technical Details**: ABAIR 모델은 자연 이미지에 대해 큰 데이터셋으로 사전 훈련(pre-training)되며, 각 픽셀의 손상 유형을 추정하기 위한 segmentation head가 추가됩니다. 또한, 독립적인 low-rank adapters를 사용하여 다양한 이미지 복원 작업에 적응합니다. 마지막으로, 이미지의 입력에 따라 적절한 adapters를 결합할 수 있는 경량의 손상 추정기를 학습합니다.

- **Performance Highlights**: ABAIR 모델은 다섯 개 및 세 개의 손상을 포함하는 설정에서 기존의 최첨단 기법보다 우수한 성능을 보이며, 보지 못한 손상에 대한 일반화 및 복합적인 손상 처리에서도 두각을 나타냅니다. 이로 인해 ABAIR는 다재다능한 이미지 복원에 적합한 효과적인 솔루션으로 자리잡고 있습니다.



### Deep Fourier-embedded Network for Bi-modal Salient Object Detection (https://arxiv.org/abs/2411.18409)
Comments:
          13 pages, 13 figures. Submitted to TMM on April 29, 2024

- **What's New**: 본 논문에서는 RGB와 열 화상 이미지를 결합한 새로운 접근법을 제시합니다. 기존의 Transformer 기반 모델의 한계를 극복하기 위해 순수한 Fast Fourier Transform (FFT) 기반의 모델인 Deep Fourier-embedded Network (DFENet)를 개발하였습니다. DFENet은 복잡한 고해상도 Bi-modal 특성 융합을 처리하는 데 필요한 메모리와 계산 요구를 줄이면서도 글로벌 의존성을 효과적으로 캡처합니다.

- **Technical Details**: DFENet은 Modal-coordinated Perception Attention (MPA)와 Frequency-decomposed Edge-aware Module (FEM)을 통해 RGB 및 열 화상 영상의 주목할만한 특징을 통합합니다. MPA는 공간 및 채널 차원에서의 상호보완적인 정보 획득을 위한 재임베딩 전략을 사용하고, FEM은 저수준 특성을 깊이 분해하여 객체 모서리를 명확히 합니다. 또한, Fourier Residual Channel Attention Block (FRCAB)을 각 디코더 레이어에 장착하여 고주파 정보 우선순위를 높입니다.

- **Performance Highlights**: DFENet은 4개의 Bi-modal Salient Object Detection 벤치마크 데이터셋에서 12개의 최신 기술 모델보다 뛰어난 성능을 보였습니다. Co-focus Frequency Loss (CFL)를 통해 불리한 주파수를 최소화하며, 이를 통해 최종 픽셀 수준 예측의 품질을 향상시킵니다. 이러한 기법들은 비단순한 RGB-T BSOD의 최신 발전을 반영합니다.



### GeneQuery: A General QA-based Framework for Spatial Gene Expression Predictions from Histology Images (https://arxiv.org/abs/2411.18391)
- **What's New**: 이 논문은 GeneQuery를 제안하여 유전자 발현 예측 문제를 질문-답변(QA) 방식으로 해결하려고 합니다. 기존의 연구들은 유전자 발현을 여러 출력 회귀(multi-output regression) 문제로 처리하였으나 GeneQuery는 이를 보다 일반화된 QA 문제로 재구성합니다. 이 새로운 접근 방식은 모델의 일반화와 유연성을 증진시키는 데 기여합니다.

- **Technical Details**: GeneQuery는 조직 이미지를 맥락(context)으로 사용하고, 유전자 관련 정보를 쿼리(query)로 사용하여 유전자 발현 값을 예측합니다. 이를 위해 GeneQuery는 유전자 분포를 추정할 수 있는 유전자 랜덤 변수(gene random variable)를 도입합니다. 또한, 이 프레임워크는 두 가지 아키텍처 구현, 즉 이미지 간 패턴을 캡처하는 spot-aware GeneQuery와 유전자 간 패턴을 캡처하는 gene-aware GeneQuery로 구성됩니다.

- **Performance Highlights**: GeneQuery는 다양한 공간 전사체(spatial transcriptomics) 데이터셋에 대한 종합적인 실험을 통해 기존의 최첨단 방법들을 초월하는 성능을 보여주었습니다. 특히, GeneQuery는 훈련 중 보지 못한 유전자에 대해서도 예측할 수 있는 능력을 갖추고 있습니다. 이 모델은 조직 구조를 분석하는 데에도 잠재적인 활용 가능성을 보입니다.



### Convolutional Neural Networks Do Work with Pre-Defined Filters (https://arxiv.org/abs/2411.18388)
- **What's New**: 본 논문에서는 훈련 중에 모든 nxn 컨볼루션 커널(n>1)을 미리 정의된 필터로 사용하는 새로운 클래스의 컨볼루션 신경망인 Pre-defined Filter Convolutional Neural Networks (PFCNNs)를 소개합니다. 이 구조는 Pre-defined Filter Module (PFM)이라는 독특한 깊이별 컨볼루션 작업을 포함하고 있습니다. 자원과 시간을 절약하기 위해 몇 가지 고정된 커널만 사용되며, 이러한 방식으로 복잡하고 차별적인 특징을 학습하게 됩니다.

- **Technical Details**: PFCNN 구조는 ResNet18에서 모든 n>1인 컨볼루션 레이어를 PFM으로 대체하여 파라미터의 수를 줄이고 훈련 전 과정에서 1x1 컨볼루션 가중치만 조정합니다. 연구에서는 PFM이 RGB 입력 이미지에 적용되고, 16개의 미리 정의된 엣지 필터를 독립적으로 사용하여 컨볼루션을 수행합니다. 이러한 설정은 단계적으로 가중치 조정이 가능한 1x1 컨볼루션 레이어를 통해 구현됩니다.

- **Performance Highlights**: PFNet18이라는 새로운 네트워크 구조는 ResNet18 대비 훈련 파라미터의 13%만 필요하며, 일부 세분화된 이미지 데이터셋에서 우수한 인식 성능을 나타냈습니다. 필터 선택이 성능에 미치는 영향도 강조되며, 엣지 필터가 무작위 필터보다 더 나은 인식률을 보였습니다. 이는 CNN의 효율성에 대한 새로운 관점을 제공하며 과도한 가중치 수를 줄이는 방법으로 제안됩니다.



### XR-MBT: Multi-modal Full Body Tracking for XR through Self-Supervision with Learned Depth Point Cloud Registration (https://arxiv.org/abs/2411.18377)
Comments:
          Accepted to WACV 2025

- **What's New**: 본 논문은 XR(확장 현실) 장치에서 실시간으로 전체 신체 동작을 추적할 수 있는 멀티 모달 포즈 추정 모델을 제안합니다. 기존의 전통적인 신체 추적 방법은 제한적인 데이터와 센싱 신호에 의존하여 합성 방법을 사용하였으나, 우리 연구는 깊이 센싱(depth sensing)과 자기 감독(self-supervision) 기법을 결합하여 보다 정확한 전신 추적을 가능하게 합니다. 이는 XR 기술에서 중요한 전환점이 될 것으로 기대됩니다.

- **Technical Details**: 이 연구에서는 3-포인트(3-point) 신호와 세멘틱 포인트 클라우드(Semantic Point Cloud) 인코더 네트워크를 결합하여 멀티 모달 포즈 추정 네트워크를 개발했습니다. 이 네트워크는 자체 지도 학습(self-supervised learning)을 통해 실시간 깊이 데이터를 활용하여 다양한 신체 모션을 효과적으로 추적합니다. 주목할 점은, 저비용으로 수집된 포인트 클라우드 기반 데이터를 활용하여 기존의 복잡한 데이터 수집 방식의 필요성을 줄였다는 점입니다.

- **Performance Highlights**: 제안된 XR-MBT 시스템은 다양한 신체 동작을 높은 정확도로 추적할 수 있음을 실험을 통해 입증했습니다. 기존의 신체 추적 기술과 비교했을 때, 우리의 접근 방식은 XR 장치에서 처음으로 다리 추적이 가능하다는 점에서 의미가 큽니다. 이 연구는 XR 환경에서 진정한 사용자 체험을 제공하기 위한 기술적 기초를 마련하는 데 기여할 것입니다.



### Individual Content and Motion Dynamics Preserved Pruning for Video Diffusion Models (https://arxiv.org/abs/2411.18375)
Comments:
          9 figures, 9 tables

- **What's New**: 이번 논문에서는 비디오 확산 모델(VDM)의 고비용 계산과 느린 추론 시간을 극복하기 위한 새로운 비디오 확산 모델 압축 기법을 제안합니다. 이를 위해 개별 콘텐츠와 모션 동역학을 보존하는 프루닝과 일관성 손실(Consistency Loss)을 사용합니다. 깊은 VDM 레이어는 영상의 모션 동역학의 품질을 유지하는 데 중요한 반면, 얕은 레이어는 개별 프레임에 더 중점을 두는 것으로 관찰되었습니다.

- **Technical Details**: VDMini라는 경량 VDM 변형은 얕은 레이어에서 중복 블록을 프루닝하여 생성됩니다. 또한, 개인 콘텐츠 및 모션 동역학(Individual Content and Motion Dynamics, ICMD) 일관성 손실을 통해 대형 VDM과 동일한 생성 성능을 달성하도록 설계되었습니다. 이 연구에서는 중복 블록을 프루닝하고, 개별 콘텐츠 증류(Individual Content Distillation, ICD) 손실과 멀티 프레임 콘텐츠 적대적 손실(Multi-frame Content Adversarial Loss, MCA)을 도입하여 전체 비디오의 모션 동역학을 개선하는 방법을 설명합니다.

- **Performance Highlights**: 우리는 Text-to-Video (T2V) 및 Image-to-Video (I2V)라는 두 가지 비디오 생성 작업에서 VDMini의 효과를 실험을 통해 입증합니다. 특히, I2V 방식인 SF-V와 T2V 방식인 T2V-Turbo-v2에서 각각 2.5배 및 1.4배의 속도 향상을 달성하였으며, UCF101 및 VBench 벤치마크에서 생성된 비디오의 품질을 유지했습니다.



### ChatRex: Taming Multimodal LLM for Joint Perception and Understanding (https://arxiv.org/abs/2411.18363)
Comments:
          35 pages, 19 figures

- **What's New**: 이번 연구에서는 시각 이해(visual understanding)와 인식(perception)의 격차를 줄이기 위한 새로운 접근 방식을 제시합니다. 특히, MLLM(Multimodal Large Language Models)인 ChatRex를 도입하여, LLM이 직접 박스 좌표를 예측하는 대신 유니버설 제안 네트워크(universal proposal network)로부터 출력된 박스 결과를 활용합니다. 이를 통해 인식 작업을 회귀(regression)에서 검색(retrieval) 기반 작업으로 전환하여, LLM이 더 잘 처리할 수 있도록 하였습니다.

- **Technical Details**: ChatRex 모델은 분리된 인식 설계(decoupled perception design)를 통해 작동합니다. 박스 인덱스를 출력하게 하여 인식 과제를 더 효율적으로 수행할 수 있게 합니다. 이와 함께 완전 자동화된 데이터 엔진을 구축하고, 인식과 이해의 공동 훈련(joint training)을 지원하기 위해 다양한 세분화를 가진 Rexverse-2M 데이터셋을 생성하였습니다.

- **Performance Highlights**: 표준적인 두 단계 훈련 후, ChatRex는 강력한 인식 능력을 보여주며 멀티모달 이해 성능도 유지하고 있습니다. 이러한 두 가지 능력을 결합함으로써 인식과 이해의 보완적인 역할이 실질적인 애플리케이션의 가능성을 열어주고 있음을 보여주고 있습니다.



### TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models (https://arxiv.org/abs/2411.18350)
- **What's New**: 이 논문에서는 VTOFF(Virtual Try-Off)라는 새로운 작업을 소개합니다. 이는 옷을 입은 사람의 단일 사진에서 표준화된 의상 이미지를 생성하는 데 중점을 두며, 전통적인 Virtual Try-On(VTON)과는 달리 직접적으로 모델을 ‘입히는’ 방식이 아닙니다. VTOFF는 의상 형태, 질감, 복잡한 패턴을 캡처하는 데 도전과제를 제공합니다.

- **Technical Details**: 우리는 TryOffDiff라는 모델을 제안하여 Stable Diffusion에 SigLIP 기반의 시각적 조정을 도입하여 높은 재구성 품질과 세부사항을 보존합니다. VITON-HD 데이터셋을 수정하여 실시한 실험에서는 우리의 접근 방식이 포즈 전이 및 기존 메서드보다 우수한 결과를 보여주며, 처리 과정이 간소화되었습니다. 또한 기존 이미지 생성 메트릭이 재구성 품질을 적절히 평가하지 못함에 따라, 더욱 정확한 평가를 위해 DISTS를 사용하였습니다.

- **Performance Highlights**: VTOFF의 결과는 전자상거래 애플리케이션에서 제품 이미지를 개선하고 생성 모델 평가를 발전시키며, 고품질 재구성을 위한 미래 작업에 영감을 줄 가능성을 보여줍니다. 생성된 의상 이미지는 기존의 가상 시도 솔루션과 통합되어 복잡한 착용자 간 시도도 가능하게 하며, 고객의 구매 결정을 도울 수 있습니다. 이러한 접근 방식은 궁극적으로 패션 산업의 환경적 영향을 줄이는 데 기여할 수 있습니다.



### Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation (https://arxiv.org/abs/2411.18335)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 Helvipad라는 새로운 omnidirectional stereo depth estimation 데이터셋을 소개합니다. 이 데이터셋은 40,000 개의 프레임으로 구성되어 있으며, 다양한 조명 조건을 갖춘 실내 및 실외의 복잡한 장면에서 수집되었습니다. Helvipad는 고품질 깊이(depth) 및 불일치(disparity) 레이블을 포함하며, depth completion을 통해 레이블 밀도를 증가시킨 훈련 세트를 제공합니다.

- **Technical Details**: Helvipad 데이터셋은 두 개의 360° 카메라와 LiDAR 센서를 사용하여 수집되었습니다. 데이터는 equirectangular 이미지로 투영된 3D 포인트 클라우드를 통해 정확한 깊이 레이블을 제공합니다. 또한 이 연구에서는 극좌표 맵을 입력으로 도입하고 원형 패딩(circular padding)을 적용하여 Omnidirectional 이미지에 대한 stereo matching 모델을 개선하는 방법을 제안합니다.

- **Performance Highlights**: Benchmarking 결과에 따르면, 최근의 stereo matching 모델들이 omnidirectional 접근법보다 높은 성능을 보이지만, equirectangular 투영에서의 심각한 왜곡으로 인해 여전히 도전 과제가 있습니다. 이 논문에서 제안된 모델의 개조는 이전의 모든 접근법을 초월하는 성능 향상을 가져왔으며, 특히 우리가 수집한 Helvipad 데이터셋에서 효과적인 성능을 발휘했습니다.



### EventCrab: Harnessing Frame and Point Synergy for Event-based Action Recognition and Beyond (https://arxiv.org/abs/2411.18328)
- **What's New**: 이 논문에서는 새로운 EventCrab 프레임워크를 제안하여 Event-based Action Recognition (EAR)의 효율성과 정확성을 동시에 향상시키고자 합니다. 기존 방법들이 비동기 이벤트 데이터의 고유한 밀도 시간적 및 희소 공간적 속성을 적절히 반영하지 못하는 문제를 해결하는 데 중점을 두고 있습니다. 또한, 다양한 이벤트 표현의 통합 호환성을 탐색하여 인공지능 모델 성능을 개선하려는 노력이 강조됩니다.

- **Technical Details**: EventCrab는 'lighter' 프레임 전용 네트워크를 밀집 이벤트 프레임과 'heavier' 포인트 전용 네트워크를 희소 이벤트 포인트에 통합하여 성능과 효율성을 균형 있게 유지하도록 설계되었습니다. 이를 위해 Spiking-like Context Learner (SCL)와 Event Point Encoder (EPE)와 같은 두 가지 주요 전략을 사용하여 비동기 이벤트 포인트에 내재된 독특한 시공간 관계를 최대한 활용합니다. 이론적으로, 전체 프레임 프롬프트 및 포인트 프롬프트 특징을 통해 CLIP 텍스트 인코더의 지침을 받아 각 이벤트의 특징을 최적화합니다.

- **Performance Highlights**: 실험 결과, EventCrab는 대조적인 데이터 세트에서 뛰어난 성능을 보여줍니다. 특히, SeAct 데이터 세트에서는 5.17%의 성능 개선, HARDVS 데이터 세트에서는 7.01%의 향상을 나타내는 등 탁월한 결과를 달성했습니다. 이러한 성과는 제안된 프레임워크의 효율적인 학습과 효과적인 이벤트 기반 동작 인식 능력을 입증합니다.



### Mixture of Experts in Image Classification: What's the Sweet Spot? (https://arxiv.org/abs/2411.18322)
- **What's New**: 이번 연구는 Mixture-of-Experts (MoE) 모델을 컴퓨터 비전 분야에 통합하여 그 가능성을 탐구하고 있습니다. 특히 ImagerNet-1k 및 ImageNet-21k 데이터셋을 대상으로 MoE 모델의 다양한 구성 방식을 실험하여, 이미지 분류에서의 성능을 향상시킬 수 있는 방법들을 제안합니다.

- **Technical Details**: MoE 모델은 매개변수를 여러 개의 하위 모델로 나누어 특정 입력에 따라 활성화된 매개변수 수에 따라 계산 비용을 조절하는 방식입니다. 연구에서는 ConvNext와 Vision Transformer (ViT) 아키텍처에서 MoE를 통합한 실험을 진행했으며, 각 아키텍처에 따라 최적의 설계가 달라짐을 발견했습니다.

- **Performance Highlights**: 실험 결과, MoE 층을 시연 블록의 마지막 두 개 위치에 배치할 경우 중간 모델 크기에서 상당한 성능 개선이 이루어졌습니다. 그러나 대형 모델과 대형 데이터셋으로 확장할수록 MoE를 이용한 이미지 분류 성능의 이점은 점차 사라지는 경향을 보였습니다.



### Real-time Video Target Tracking Algorithm Utilizing Convolutional Neural Networks (CNN) (https://arxiv.org/abs/2411.18314)
- **What's New**: 이 논문은 Convolutional Neural Networks (CNN)을 기반으로 한 실시간 비디오 타겟 추적 알고리즘을 연구하고 구현하는 것을 목표로 합니다. 이 알고리즘은 복잡한 환경에서 타겟 추적의 정확성과 견고성을 향상시킵니다. 특히, 타겟의 가림, 형태적 변화, 그리고 배경 간섭 문제를 효과적으로 처리하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 온라인 학습 메커니즘을 통해 타겟 모델을 지속적으로 업데이트하여 타겟의 변화에 적응합니다. 알고리즘은 빠른 움직임, 부분 가림, 복잡한 배경 상황을 처리할 수 있어 높은 추적 성공률과 낮은 실패율을 보입니다. 이로 인해 CNN을 실시간 비디오 타겟 추적에 성공적으로 적용하고 있습니다.

- **Performance Highlights**: 이 연구는 추적 알고리즘의 정확도와 안정성을 향상시키면서도 높은 처리 속도를 유지합니다. 이는 비디오 감시 및 지능형 교통 시스템 분야의 타겟 추적 작업에 새로운 해결책을 제공할 것으로 기대됩니다.



### Neural Surface Priors for Editable Gaussian Splatting (https://arxiv.org/abs/2411.18311)
Comments:
          9 pages, 7 figures

- **What's New**: 본 논문에서는 이미지 데이터를 통해 쉽게 수정 가능한 3D 기하학 및 외관 표현을 복구하는 새로운 방법을 제안합니다. 3D Gaussian Splatting (3DGS)을 활용하여 맥스 조정을 통한 직관적인 장면 편집을 가능하게 하고, Neural Signed Distance Field를 사용하여 기하학을 재구성하며 고품질 메쉬를 생성합니다.

- **Technical Details**: 제안된 방법은 입력 이미지와 카메라 포즈를 기반으로 하여, Gaussian의 배열을 추정하고, 각 Gaussian 구성 요소의 불투명도를 신경망으로 복구된 표면에 조건화합니다. 이를 통해 편집을 위한 프록시 표현을 생성하며, 렌더링 단계에서 메쉬 수정이 프록시 표현으로 전파되고, 그로 인해 고급 시뮬레이션에도 적합합니다.

- **Performance Highlights**: 본 연구는 3D 장면 편집을 단순화하고, 사용성과 비주얼 피델리티(visual fidelity)에서 기존 방식보다 개선된 성과를 보여줍니다. 실험 결과, 제안된 접근법이 외관 복구, 기하학 기반 편집 및 물리적 시뮬레이션에서 높은 비주얼 품질을 획득했다는 결과를 포함하고 있습니다.



### MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancemen (https://arxiv.org/abs/2411.18309)
Comments:
          10 pages, 10 figures

- **What's New**: 이번 연구는 multi-view를 활용한 진단 정보를 포함하고 클리닉 전문 지식과 Kolmogorov-Arnold Networks (KANs)를 활용한 새로운 Multi-view perception Knowledge-enhanced Transformer (MvKeTR)를 제안합니다. 이 모델은 방사선 전문의가 CT 영상을 여러 방향에서 분석하는 방식을 모방하여 진단 정보를 효과적으로 통합합니다. 또한, 관련 임상 기록을 참조하여 진단 의사 결정을 지원하는 Cross-Modal Knowledge Enhancer (CMKE)를 도입하여 의료 지식을 보고서 생성 과정에 통합합니다.

- **Technical Details**: 연구에서는 Multi-View Perception Aggregator (MVPA)를 설계하여 여러 해부학적 관점에서 진단 정보를 통합합니다. 일반적으로 3D CT 볼륨의 다수 관점 분석을 통해 더 나은 공간 특징을 캡쳐할 수 있게 되며, KANs를 기본 구성 요소로 사용하여 복잡한 진단 관계를 학습합니다. 이러한 접근은 진단 과정에서의 의료 지식 격차를 메우고, 정확하고 신뢰할 수 있는 보고서를 생성하기 위해 필수적입니다.

- **Performance Highlights**: CTRG-Chest-548K 데이터세트에서 수행한 광범위한 실험에서는 MvKeTR가 기존의 최첨단 모델들을 모든 지표에서 초월함을 보여주었습니다. 자동화된 CT 보고서 생성 과정에서 MvKeTR는 이전 시스템에 비해 품질과 효율성을 모두 향상시킨 것으로 평가됩니다. 이로 인해 의사들의 부담을 덜고 환자 관리의 질을 개선하는 데 기여할 수 있습니다.



### InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation (https://arxiv.org/abs/2411.18303)
- **What's New**: 이 논문에서 제안하는 InfiniDreamer는 임의의 길이의 인간 동작 생성을 위한 혁신적인 프레임워크입니다. 기존 동작 생성 방법이 짧은 시퀀스에 제한되는 문제를 해결하기 위해, 텍스트 설명에 따라 서브 모션을 생성하고 이를 랜덤하게 초기화된 전이 세그먼트를 사용하여 조합합니다. 이후 Segment Score Distillation(SSD)이라는 최적화 방법을 도입하여 전체 긴 동작 시퀀스를 정제합니다.

- **Technical Details**: InfiniDreamer는 각 서브 모션을 텍스트 프롬프트에 조건화하여 생성하는 방식으로 작동합니다. 생성된 서브 모션은 랜덤 초기화를 통해 조합되며, 슬라이딩 윈도우 접근 방식을 사용하여 짧은 겹치는 시퀀스 세그먼트를 반복적으로 샘플링합니다. SSD는 사전 훈련된 모션 확산 사전을 정렬하여 각 짧은 시퀀스 세그먼트를 정제하고, 전체 긴 동작 시퀀스의 글로벌 일관성을 유지합니다.

- **Performance Highlights**: ┷ 실험을 통해 InfiniDreamer는 기존의 훈련이 필요 없는 방법들보다 장기적인 동작 생성을 위한 월등한 성능을 보였습니다. HumanML3D 및 BABEL 데이터셋에서 평가한 결과, 우리의 프레임워크는 연속적이고 맥락적으로 인지된 동작 시퀀스를 생성할 수 있는 뛰어난 능력을 입증했습니다. 추가적으로, 정성적 평가 또한 이루어져, 동작의 연결성과 일관성이 강화된 모습이 관찰되었습니다.



### Enhancing MMDiT-Based Text-to-Image Models for Similar Subject Generation (https://arxiv.org/abs/2411.18301)
- **What's New**: 최근 발표된 Multimodal Diffusion Transformer (MMDiT) 모델은 이전 모델에서 발생하는 많은 생성 문제를 크게 완화합니다. 그러나 입력 텍스트 프롬프트에 유사한 의미나 외관을 가진 여러 개의 주제가 포함될 경우 여전히 주제 소외(subject neglect)나 혼합(mixing)이 발생하는 것으로 확인되었습니다. 이러한 문제를 해결하기 위해, 테스트 시간 최적화를 통해 모호한 라테ント(latent)를 조정하는 새로운 접근 방식을 제안했습니다.

- **Technical Details**: MMDiT 아키텍처 내에서 발생하는 세 가지 모호성은 Inter-block Ambiguity, Text Encoder Ambiguity, 그리고 Semantic Ambiguity입니다. 이러한 모호성을 해결하기 위해 Block Alignment Loss, Text Encoder Alignment Loss, 그리고 Overlap Loss와 같은 세 가지 손실 함수를 설계하였습니다. 또한, Overlap Online Detection 및 Back-to-Start Sampling Strategy를 제안하여 유사한 주제가 생성되는 문제를 더욱 효과적으로 완화합니다.

- **Performance Highlights**: MITD 기반의 모델에서 우리의 방법의 효과를 검증하기 위해 유사 주제의 어려운 데이터셋을 구축하였고, 이 데이터셋에서의 실험 결과 기존 방법보다 생성 품질이 우수하고 성공률이 현저히 높음을 나타냈습니다. 또한, 제안된 각 구성 요소의 필요성을 보여주는 충분한 경험적 분석이 진행되었습니다.



### HUPE: Heuristic Underwater Perceptual Enhancement with Semantic Collaborative Learning (https://arxiv.org/abs/2411.18296)
Comments:
          22 pages, 21 figures

- **What's New**: 이번 연구에서는 새로운 수중 이미지 개선 방법인 HUPE(Heuristic Invertible Network for Underwater Perception Enhancement)를 제안합니다. HUPE는 수중 이미지의 시각적 품질을 향상시키는 데 중점을 두면서도, 다양한 하위 작업에 유연하게 적용될 수 있도록 디자인되었습니다. 이 방법은 정보 보존 가능한 역전 가능한 변환과 임베디드 푸리에 변환을 활용하여 수중 이미지와 그 선명한 이미지 간의 쌍방향 매핑을 확립합니다.

- **Technical Details**: HUPE는 수중 환경을 고려하여 복잡한 장면 정보를 보다 효과적으로 포착하기 위해 휴리스틱 프라이어(heuristic prior)를 통합합니다. 또한, 시각적 향상 작업과 하위 작업의 공동 최적화 과정에서 의미론적 협업 학습 모듈(semantic collaborative learning module)을 적용하여, 더욱 작업 지향적인 의미적 특징을 추출할 수 있도록 돕습니다. 이를 통해 HUPE는 시각적 품질 향상뿐만 아니라 인식 강화를 달성하는 모델입니다.

- **Performance Highlights**: 광범위한 실험 결과는 HUPE가 최신 기술에 비해 우수성을 나타낸다는 것을 보여줍니다. 정량적 및 정성적 분석 모두에서 HUPE는 원래의 장면 반사를 효과적으로 복원하여 인식 작업에 보다 적합하게 만듭니다. 이 연구는 수중 이미지 개선 및 이후 인식 작업 간의 균형을 잘 이루며, 실제 수중 환경에서도 뛰어난 성능을 보입니다.



### HiFiVFS: High Fidelity Video Face Swapping (https://arxiv.org/abs/2411.18293)
- **What's New**: 본 논문에서는 HiFiVFS라는 고충실도 영상 얼굴 스왑 프레임워크를 제안합니다. Stable Video Diffusion(SVD)의 강력한 생성 능력과 시간적 사전 정보를 활용하여 비디오 얼굴 스왑 문제를 해결합니다. 기존의 방법들이 세밀한 속성이나 시간적 일관성을 다루는 데 어려움을 겪고 있는 상황에서, HiFiVFS는 이러한 문제를 효과적으로 해결합니다.

- **Technical Details**: HiFiVFS는 Fine-grained Attributes Learning(FAL)과 Detailed Identity Learning(DIL)이라는 두 가지 방법론을 도입합니다. FAL은 정체성 비감각화 및 적대적 학습을 통해 세밀한 속성 정보를 추출하고, DIL은 얼굴 스왑 작업에 적합한 자세한 정체성 특징을 활용하여 결과의 유사성을 높입니다. 이러한 접근 방식은 영상에서의 세밀한 속성 유지에 중점을 둡니다.

- **Performance Highlights**: HiFiVFS 모델은 FaceForensics++ 및 VFHQ-FS 데이터세트에서 다양한 어려운 시나리오를 통해 평가되었습니다. 실험 결과, HiFiVFS는 기존의 최첨단(face swapping) 방법들보다 높은 충실도와 안정성을 제공하며, 고유의 정체성과 세밀한 속성 정보를 잘 보존하는 것으로 나타났습니다.



### Optimizing Multispectral Object Detection: A Bag of Tricks and Comprehensive Benchmarks (https://arxiv.org/abs/2411.18288)
- **What's New**: 이번 연구에서는 RGB와 TIR(열 적외선) 모달리티를 사용한 다중 스펙트럼 객체 탐지의 어려움에 대해 다룹니다. 특히, 기존 모델과 방법론의 성능 개선을 명확하게 구분하지 못하는 문제를 지적하며, 이를 해결하기 위해 공정하고 재현 가능한 벤치마크를 제안합니다. 새로운 훈련 '기술'들을 체계적으로 분류하고, 하이퍼파라미터에 대한 민감도를 조사하여 적절한 구성으로 표준화했습니다.

- **Technical Details**: 이 연구는 기존의 다중 스펙트럼 객체 탐지 방법을 체계적으로 분류하고, 다양한 백본 네트워크(backbone networks) 및 탐지 프레임워크(detection frameworks)를 활용한 종합적인 평가를 실시했습니다. 또한, 효과적으로 단일 모달리티 모델을 이중 모달리티 모델로 최적화할 수 있는 효율적이고 쉽게 배포 가능한 다중 스펙트럼 객체 탐지 프레임워크를 소개합니다. 연구팀은 다양한 도전적인 데이터 세트에서 दिएυ 남은 성능을 확인하고, 하이퍼파라미터 구성을 표준화하여 재현 가능한 벤치마크를 제시했습니다.

- **Performance Highlights**: 본 연구는 여러 대표적인 데이터 세트에서 최첨단 결과를 달성하였으며, 특히 복잡한 환경에서 특징적 융합 기술과 이중 모달리티 데이터 증강 방식을 적용하여 모델의 성능을 향상시켰습니다. 새로운 훈련 및 최적화 전략을 통해 단일 모달리티 모델이 복잡한 이중 모달리티 모델보다 더 나은 성능을 보였다는 점에서 큰 주목을 받고 있습니다. 글로벌 인공지능 혁신 대회(GAIIC)에서 1,200명 이상의 참가자 중에서 우승을 차지하는 성과를 올렸습니다.



### MotionCharacter: Identity-Preserving and Motion Controllable Human Video Generation (https://arxiv.org/abs/2411.18281)
- **What's New**: 이번 논문에서는 인물 고유의 정체성을 유지하면서 세밀한 동작 제어를 가능하게 하는 MotionCharacter라는 인간 비디오 생성 프레임워크를 제안합니다. ID 보존 모듈과 동작 제어 모듈을 통합하여 다양한 속성을 유연하게 수정할 수 있는 기능을 제공합니다. 또한, 이 연구에서는 대규모 언어 모델을 활용하여 상세한 동작 설명을 생성하는 새로운 데이터셋인 Human-Motion을 소개합니다.

- **Technical Details**: MotionCharacter는 얼굴 임베딩(face embedding)과 CLIP 임베딩을 결합하여 높은 정체성 충실도를 유지하면서도 사용자 프롬프트 기반의 동적 속성 수정이 가능합니다. 또한, ID 일관성 손실(ID-consistency loss) 및 지역 인식 손실(region-aware loss) 메커니즘을 포함하여 얼굴의 중요한 부분에 대한 주의를 향상시키고 고유 특징의 왜곡이나 흐림을 방지합니다. 이를 통해 고품질의 개인화된 T2V 생성을 달성할 수 있습니다.

- **Performance Highlights**: MotionCharacter는 ID 보존 및 동작 지시에 대한 응답성을 강화하여 질적인 및 양적인 실험 결과에서 그 효과를 입증하였습니다. 실제 사용자 테스트에서도 모델의 성능이 긍정적으로 평가되어, 목표로 하는 고화질 인간 비디오 생성을 성공적으로 달성하였습니다. 이러한 개선 사항들은 사용자 관점에서도 흥미로운 잠재력을 보여줍니다.



### Visual Adversarial Attack on Vision-Language Models for Autonomous Driving (https://arxiv.org/abs/2411.18275)
- **What's New**: 본 논문은 자율 주행(AD)에서 비전-언어 모델(VLM)에 대한 적대적 공격을 연구하는 첫 번째 시도입니다. 기존 연구가 VLM에 대한 일반적인 공격에 중점을 두었다면, 이 연구는 안전이 중요한 AD 환경에 특화된 공격의 필요성을 강조합니다. 연구자는 AD VLMs에 대한 효과적인 적대적 공격의 두 가지 독특한 도전 과제를 규명하고, AD를 위한 최초의 비주얼 적대적 공격 프레임워크인 ADvLM을 제안합니다.

- **Technical Details**: 저자들은 AD에서 비전-언어 모델의 적대적 공격을 효과적으로 수행하기 위해, 텍스트 지침의 다양성과 시계열 정보인 비주얼 시나리오의 시간적 특성을 고려한 두 가지 접근법을 도입합니다. 텍스트 모드에서는 Semantic-Invariant Induction을 통해 동일한 의미를 가지는 다양한 텍스트 지침을 생성하는 저세먼틱 엔트로피 프롬프트 라이브러리를 구축합니다. 비주얼 모드에서는 모델의 주의 메커니즘을 활용하여 중요한 프레임과 관점을 선택하고, 적대적 교란을 최적화해 전체 시나리오에 걸쳐 일반화 가능하도록 설계했습니다.

- **Performance Highlights**: ADvLM은 여러 벤치마크에서 수행된 광범위한 실험에서 기존 방법을 능가함을 입증했습니다. 특히, AD VLMs의 화이트박스 및 블랙박스 설정에서 각각 16.97% 및 7.49%의 최종 점수 감소를 기록하며 탁월한 공격 효과성을 보여주었습니다. 또한, 실제 차량에 대한 연구를 통해 ADvLM의 적용 가능성과 실용성을 추가로 검증했습니다.



### Grid-augumented vision: A simple yet effective approach for enhanced spatial understanding in multi-modal agents (https://arxiv.org/abs/2411.18270)
Comments:
          10 pages, 2 figures

- **What's New**: 최근 다중 모달 모델의 발전으로 객체 인식 및 장면 이해에서 인상적인 능력이 입증되었으나, 이러한 모델들은 정확한 공간 위치를 파악하는 데 어려움을 겪고 있습니다. 본 논문에서는 체스판 및 지도와 같은 격자 기반 참조 시스템에서 영감을 받아, 입력 이미지에 9x9 검정 격자 패턴을 오버레이하여 명시적인 시각적 위치 인코딩을 도입하는 방법을 제안합니다. 이 방법은 트랜스포머에서의 위치 인코딩 작용과 유사하게 기능하며, 기존 모델의 로컬라이제이션 정확성을 크게 향상시킵니다.

- **Technical Details**: 본 연구에서는 COCO 2017 데이터셋을 활용하여 제안한 격자 기반 접근 방식이 로컬라이제이션 정확성에서 107.4% 증가한 IoU(Intersection over Union) (0.27에서 0.56으로) 및 194.4% 개선된 GIoU(Generalized Intersection over Union) (0.18에서 0.53으로)를 보여주었다고 설명하고 있습니다. 이 격자 시스템은 모델이 정확한 공간 관계를 이해하는 데 도움을 주며, 시각적 위치 인코딩의 효과를 입증하기 위해 주의 시각화 분석을 사용하였습니다.

- **Performance Highlights**: 격자 기반 위치 인코딩 방식은 로봇 조작, 의료 영상 분석 및 자율 항법과 같이 정확한 공간 추론이 필요한 다양한 실제 응용 프로그램에 특히 유용합니다. 이 방법은 기존 의미적인 추출 방식에서 벗어나, 명시적이고 시각적인 용도로 공간 정보를 제공하는 점에서 큰 장점을 보입니다. 나아가, 이러한 접근 방식은 다중 모달 모델의 공간 이해를 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### Incomplete Multi-view Multi-label Classification via a Dual-level Contrastive Learning Framework (https://arxiv.org/abs/2411.18267)
- **What's New**: 최근 멀티 뷰(multi-view)와 멀티 레이블(multi-label) 분류는 포괄적인 데이터 분석 및 탐색에 있어 중요한 영역으로 부각되고 있습니다. 본 논문에서는 이러한 멀티 뷰 멀티 레이블 분류에서 발생할 수 있는 데이터의 불완전성 문제에 집중하며, 이 문제를 해결하기 위한 이중 수준의 대비 학습(framework dual-level contrastive learning)을 제안합니다.

- **Technical Details**: 제안된 방법은 두 개의 별도 공간에서 일관된 정보와 뷰 특이적 정보를 분리하는 방식으로, 기존 방법들과의 차별성을 가집니다. 두 채널 디커플링 모듈을 도입하여 모든 뷰에 걸쳐 정보의 일관성과 보완성을 효과적으로 추출하며, 이를 위해 두 가지 일관성 목표를 설정하여 멀티 뷰 표현에서 고품질 일관성 정보를 효율적으로 필터링합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 바탕으로 한 실험 결과, 제안된 이중 수준 대비 학습(framework dual-level contrastive learning) 틀은 이중 누락(double missing) 상황에서 안정적이고 우수한 분류 성능을 보임을 보여주었습니다. 이러한 성과는 제안 기술이 전통적인 접근 방식보다 우수한 결과를 낼 수 있음을 입증합니다.



### TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution (https://arxiv.org/abs/2411.18263)
- **What's New**: 새로운 논문에서는 TSD-SR이라는 이미지 초해상도(Real-ISR) 작업을 위한 새롭고 효율적인 단일 단계 디스틸레이션 프레임워크를 소개합니다. 이 방법은 기존의 디퓨전 모델의 한계를 극복하고, 이미지 복원에서 더 높은 성능을 발휘하기 위해 설계되었습니다. 특히, Target Score Distillation과 Distribution-Aware Sampling Module을 활용하여 더 사실적인 이미지 복원을 달성하는 것에 초점을 맞추고 있습니다.

- **Technical Details**: TSD-SR은 두 가지 주요 구성 요소, 즉 Target Score Distillation (TSD)와 Distribution-Aware Sampling Module (DASM)으로 구성되어 있습니다. TSD는 새로운 Target Score Matching 손실을 도입하여 VSD 손실의 한계를 보완하고, 올바른 최적화 경로를 제공하여 시각적 왜곡을 줄이는 데 기여합니다. DASM은 훈련 시 저노이즈 샘플을 효과적으로 샘플링하여 디테일 복원을 향상시킵니다.

- **Performance Highlights**: TSD-SR은 다양한 벤치마크 실험에서 우수한 복원 성능을 기록하였으며, inference 속도에서도 SeeSR보다 40배 빠른 성능을 나타냈습니다. 특히, 대부분의 평가 지표에서 최고 성과를 달성하여, 향후 Real-ISR 분야에서 영향력 있는 방법으로 자리잡을 것으로 기대됩니다.



### SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation (https://arxiv.org/abs/2411.18229)
Comments:
          Uncompressed version can be found in this https URL

- **What's New**: SharpDepth는 모노큘러 (monocular) 메트릭 깊이 추정 (metric depth estimation)을 위한 새로운 접근 방식으로, 기존의 판별적 (discriminative) 깊이 추정 방법의 메트릭 정확성과 생성적 (generative) 방법에서 일반적으로 달성되는 섬세한 경계 선명도를 통합합니다. 전통적인 판별적 모델은 실제 데이터에서 희소한 진실 깊이에 의해 훈련되면서 정밀한 메트릭 깊이를 예측할 수 있지만, 경계 세부 사항이 흐릿하게 된다. SharpDepth는 이러한 한계를 극복하고 메트릭 정확성과 세부 경계를 보존하여 시각적으로 선명한 깊이 예측을 제공합니다.

- **Technical Details**: 이 모델은 Noise-aware Gating 메커니즘을 사용하여 불확실한 영역에 초점을 맞춰 깊이 확산 모델 (depth diffusion model)을 안내합니다. 또한, Score Distillation Sampling (SDS) 손실을 이용하여 깊이 세부 묘사를 개선하고, Noise-aware Reconstruction 손실을 적용하여 원래 깊이 추정치에 가까운 최종 예측을 유지하면서 메트릭 정확도를 보장합니다. 이러한 기법들은 SharpDepth가 다양한 장면에서 정밀하고 세부적인 메트릭 깊이 추정을 수행할 수 있게 해줍니다.

- **Performance Highlights**: SharpDepth의 성능을 평가하기 위해 다양한 실험을 수행하였고, 판별적 및 생성적 방법과 비교하여 높은 정확도의 깊이 추정과 높은 선명도를 보존할 수 있음을 확인했습니다. 실험 결과, 기존의 상태-of-the-art 메트릭 깊이 추정기들과 비교하여 경쟁력이 있는 정확도를 제공하며, 고해상도 이미지가 아닌 100-150배 적은 수량의 훈련 이미지로 훈련될 수 있다는 장점도 있습니다. 이러한 결과는 SharpDepth가 높은 품질의 깊이 지각이 필요한 다양한 실제 환경에서 유용하다는 것을 보여줍니다.



### PATHS: A Hierarchical Transformer for Efficient Whole Slide Image Analysis (https://arxiv.org/abs/2411.18225)
- **What's New**: 이 논문에서는 Whole Slide Images (WSIs)를 처리하기 위한 새로운 방법인 PATHS(Pathology Transformer with Hierarchical Selection)를 제안합니다. 이 방법은 인간 병리학자가 슬라이드를 검사하는 방식을 모방하여, 저배율에서 이미지를 먼저 분석하고 중요한 영역을 찾아내며 이를 반복하여 고배율로 확대하는 구조를 가지고 있습니다. 이를 통해 불필요한 데이터를 줄이고, 보다 정보가 풍부한 데이터만을 활용할 수 있습니다.

- **Technical Details**: PATHS는 패치 처리 시 계층 구조를 가지며, 처음에 저배율에서 높이 입력 정보를 캡처하고 반복적으로 중요 지역을 선택하는 어텐션 메커니즘을 사용합니다. 이러한 구조는 슬라이드 전체를 처리하지 않고도 많은 해상도에서 정보를 캡처할 수 있게 해줍니다. 이 모델은 또한 각 배율에서 슬라이드의 일부만을 처리하여 계산 효율성을 높입니다.

- **Performance Highlights**: PATHS는 The Cancer Genome Atlas (TCGA)의 다섯 가지 데이터셋에서의 슬라이드 수준 예측 작업에서 기존 방법과 비교하여 뛰어난 성능을 보여줍니다. 이 모델은 처리 시간에서 10배 이상의 속도 향상을 이루었으며, 임상적으로 의미 있는 패치 선택 방법을 제공하여 병리학자의 작업 흐름을 모사합니다.



### KANs for Computer Vision: An Experimental Study (https://arxiv.org/abs/2411.18224)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문은 Kolmogorov-Arnold Networks (KANs)이 컴퓨터 비전 과제, 특히 이미지 분류에 적용된 실험적 연구를 다루고 있습니다. KANs는 에지에서 학습 가능한 activation function을 도입하여 전통적인 Multi-Layer Perceptrons (MLPs)와 Convolutional Neural Networks (CNNs)에 비해 더 유연한 비선형 변환을 제공합니다. 이러한 KANs가 실험적으로 특정 비전 작업에서 잘 작동할 수 있지만, 하이퍼파라미터에 대한 민감도와 계산 비용의 증가 같은 중요한 도전에 직면하고 있다는 점이 강조됩니다.

- **Technical Details**: KANs는 전통적인 MLPs와 비교할 때 아키텍처 및 수학적 접근 방식에서 중요한 차별점을 존재합니다. MLP는 각 뉴런이 선형 변환 후 비선형 activation을 수행하는 구조로 되어 있으며, activation function이 고정되어 있어 데이터 분포에 적응하기 어렵습니다. 반면 KANs는 뉴런 사이의 연결(에지)에서 학습 가능한 activation function을 제공하며, 이러한 함수들은 보통 B-spline으로 표현됩니다. 이를 통해 KANs는 데이터 관계를 보다 유연하게 모델링할 수 있습니다.

- **Performance Highlights**: 실험 결과 KANs는 MLP에 비해 성능에서 약간의 향상만을 보였고, 더 많은 파라미터 수를 동반했습니다. KANs는 또한 추가 하이퍼파라미터(예: 차수 및 그리드)에 대해 MLP보다 더 민감하다는 점이 발견되었습니다. 이러한 결과는 KANs가 현재 형태로는 컴퓨터 비전 아키텍처의 대안이 적합하지 않음을 시사합니다.



### TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability (https://arxiv.org/abs/2411.18211)
- **What's New**: 이번 연구는 비디오 중심의 고품질 대화 생성을 위한 새로운 Video-LLM(TimeMarker)을 소개합니다. TimeMarker는 비디오 콘텐츠의 정확한 시간적 위치를 강조하며, Temporal Separator Tokens를 통해 시간 인식을 향상시킵니다. 다양한 길이의 비디오를 효과적으로 처리하기 위한 AnyLength 메커니즘이 도입되어, 긴 비디오에서도 세밀한 정보를 유지할 수 있도록 설계되었습니다. 또한, 복합적인 데이터셋을 활용하여 시간 이해 능력을 강화하였습니다.

- **Technical Details**: TimeMarker는 Temporal Separator Tokens를 통합하여 비디오의 절대 시간적 위치를 인코딩합니다. 이는 텍스트와 비디오 프레임 토큰을 상호 연결하여 생성된 것이며, 특정 순간을 정확히 식별할 수 있게 합니다. AnyLength 메커니즘을 통해 비디오의 길이에 따라 동적 프레임 샘플링과 토큰 융합을 적용하여, 짧은 비디오의 경우 더 많은 세부 정보를 캡처하고, 긴 비디오의 경우 효율적으로 내용을 관리할 수 있습니다. 또한, 5M 이상의 비디오-텍스트 쌍과 85M 이미지를 포함한 다양한 데이터셋을 활용합니다.

- **Performance Highlights**: TimeMarker는 여러 공개 비디오 벤치마크에서 최첨단 성과를 기록하며, 짧은 비디오와 긴 비디오 모두에서 우수한 성능을 발휘합니다. 특히 시간적 문장 기반 위치 지정 작업에서 전통적인 모델들을 초월하며, 비디오 분석에서의 시간적 이해 능력을 강조합니다. 이러한 성과는 TimeMarker의 시간적 위치 지정 기능과 이해 능력이 뛰어남을 나타냅니다.



### From Open Vocabulary to Open World: Teaching Vision Language Models to Detect Novel Objects (https://arxiv.org/abs/2411.18207)
- **What's New**: 이 논문은 Open Vocabulary Object Detection (OVD) 모델이 개방형 환경(오픈 월드)에서 작동할 수 있는 새로운 프레임워크를 제안합니다. 기존 OVD는 정확한 프롬프트에 의존하기 때문에 중요한 응용 분야에서 한계를 가집니다. 본 연구는 Open World Embedding Learning (OWEL)과 Multi-Scale Contrastive Anchor Learning (MSCAL) 접근 방식을 통해, 모델이 새로운 객체를 지속적으로 학습하고 탐지할 수 있도록 합니다.

- **Technical Details**: OWEL은 파라미터화된 클래스 임베딩을 최적화하여 전체 모델을 미세 조정하지 않고 새로운 클래스를 학습할 수 있게 합니다. Pseudo Unknown Embedding 개념을 도입하여, 현재 알려진 클래스를 기반으로 잃어버린 클래스의 위치를 추정합니다. MSCAL은 여러 규모에서 클래스 인식을 돕기 위해 객체 임베딩의 클래스 내 일관성을 증대시키며, 알려진 클래스와의 혼동을 줄이는 데 기여합니다.

- **Performance Highlights**: 이 연구는 M-OWODB 및 S-OWODB 벤치마크에서 U-Recall 성능이 최첨단 수준을 초과하며, 기타 메트릭에서도 우수한 성능을 유지합니다. 또한 nuScenes 기반의 새로운 벤치마크에서도 최고의 결과를 달성했습니다. 이러한 연구 결과들은 OVD 모델이 오픈 월드 환경에서 효과적으로 동작할 수 있게 해 주는 기초적인 기여를 합니다.



### Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning (https://arxiv.org/abs/2411.18203)
Comments:
          16 pages, 11 figures

- **What's New**: 이 논문에서는 Vision-Language Models(VLMs)의 추론 능력을 향상시키기 위한 새로운 프레임워크인 Critic-V를 소개합니다. Critic-V는 Actor-Critic 구조를 기반으로 하여 Reasoner와 Critic이라는 두 개의 분리된 모듈을 통합합니다. 이를 통해 Critic은 비주얼 및 텍스트 입력을 바탕으로 Reasoner가 생성한 추론 경로에 대해 건설적인 피드백을 제공하여 보다 정확하고 효율적인 추론을 가능하게 합니다.

- **Technical Details**: Critic-V의 Reasoner는 비주얼과 텍스트 입력으로부터 추론 경로를 생성하고, Critic은 이 경로에 대해 자연어 피드백을 제공함으로써 추론 전략을 수정할 수 있도록 하는 역할을 합니다. 이 상호작용 과정은 강화학습 프레임워크에 의해 이론적으로 구동되며, Critic은 스칼라 보상 대신 자연어 비판을 제공하여 보다 섬세한 피드백을 가능하게 합니다. Critic 모델은 Rule-based Reward (RBR)와 같은 데이터를 사용하여 평가 능력을 키우고, Direct Preference Optimization (DPO)을 통해 훈련합니다.

- **Performance Highlights**: Critic-V 프레임워크는 8개의 벤치마크 중 5곳에서 기존의 방법들보다 유의미하게 우수한 성능을 보여주었으며, 특히 추론 정확성과 효율성 분야에서 두드러진 개선을 이루었습니다. 이 접근 방식을 통해 VLM들이 자율주행 및 구현된 지능과 같은 실제 문제를 해결하는 데 있어 신뢰성을 크게 향상시킬 수 있음을 보여줍니다. Critic-V는 외부 비평 모델을 통합하여 추론 과정에서의 오류를 줄이고, VLM의 전반적인 성능을 높일 수 있는 유망한 솔루션을 제안합니다.



### DistinctAD: Distinctive Audio Description Generation in Contexts (https://arxiv.org/abs/2411.18180)
- **What's New**: 이 논문에서는 DistinctAD라는 새로운 두 단계 프레임워크를 제안하여 고유한 오디오 설명(AD)을 생성하는 데 중점을 두고 있습니다. 기존의 방법들은 영화-AD 데이터와 비전-언어 모델 훈련 데이터 간의 도메인 간극에 직면해 있었으며, 긴 영화의 유사한 시각 클립들로 인해 문맥상 중복이 발생하는 문제가 있었습니다. DistinctAD는 이러한 문제를 해결하면서도 보다 품질 높은 내러티브를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: DistinctAD의 첫 번째 단계에서는 CLIP-AD 적응 전략을 통해 도메인 간극을 해소하고, 두 번째 단계에서는 컨텍스트 기대-최대화 주의(EMA) 모듈과 유사한 단어 예측 손실을 도입하여 불필요한 단어 중복을 줄입니다. 이를 통해 연속적인 영상 클립에서 공통 요소를 추출하고, 함수적으로 더 뚜렷한 AD를 생성할 수 있습니다. 이러한 방법론적 통합은 AD 생성의 적합성을 높이며, 기존 비디오-AD 대응 가능성을 극대화합니다.

- **Performance Highlights**: MAD-Eval, CMD-AD, TV-AD 벤치마크를 통한 포괄적인 평가 결과, DistinctAD는 기존 기법들을 지속적으로 초월하는 성과를 보였습니다. Recall@k/N과 같은 지표에서 우수한 성능을 나타내며, 고유한 AD를 생성하는 능력 또한 강조되었습니다. 이러한 지표들은 DistinctAD가 제공하는 품질 높은 오디오 설명의 효과를 확실히 보여줍니다.



### Enhancing Computer Vision with Knowledge: a Rummikub Case Study (https://arxiv.org/abs/2411.18172)
Comments:
          Submitted to ESANN2025

- **What's New**:  이번 논문에서는 인공 신경망(Artificial Neural Networks, ANNs)의 한계를 극복하기 위해 배경 지식과 별도의 추론 컴포넌트를 결합하는 방법을 제안합니다. 특히, 인기 보드 게임인 룸미큐브(Rummikub)의 게임 상태에서 타일을 올바르게 인식하는 문제를 다루었습니다. 연구 결과, 추가된 배경 지식이 데이터 세트의 3분의 2만큼의 가치를 가지며, 훈련 시간을 절반으로 줄일 수 있음을 보여주었습니다.

- **Technical Details**: 연구에는 3개의 서로 다른 줌 레벨, 4개의 조명 조건, 2개의 배경을 포함한 285개의 룸미큐브 이미지 데이터 세트가 사용되었습니다. 각 이미지에는 검증된 세트가 다양한 위치와 각도로 포함되어 있으며, 총 4336개의 타일에 대한 주석이 달려 있습니다. 제안된 파이프라인은 타일 탐지, 클러스터링, 숫자 및 색상 분류, 수정의 4단계로 구성됩니다.

- **Performance Highlights**: 실험은 Intel Xeon E5-2630 v3와 NVIDIA Quadro P2000을 사용하여 수행되었습니다. 각 실험은 10번 반복하여 평균 및 표준 편차를 보고하였고, 훈련 데이터의 크기와 모델의 성능을 비교하였습니다. 기본 ANN 설정과 배경 지식이 통합된 설정 간의 성능 차이를 통해, 논리적 추론 단계의 추가가 얼마나 유익한지를 강조하였습니다.



### PDZSeg: Adapting the Foundation Model for Dissection Zone Segmentation with Visual Prompts in Robot-assisted Endoscopic Submucosal Dissection (https://arxiv.org/abs/2411.18169)
- **What's New**: 본 연구에서는 Prompted-based Dissection Zone Segmentation (PDZSeg) 모델을 제안하여 내시경 점막하 박리(ESD) 수술 중 해부 구역의 세분화(segmentation) 개선을 목표로 하고 있습니다. 이 모델은 다양한 시각적 프롬프트(visual prompts)를 활용하여 수술 중 해부 구역을 명확하게 제시할 수 있도록 설계되었습니다. 기존의 방식과 달리, 사용자가 스크리블(scribble)이나 경계 상자(bounding box)를 통해 직관적으로 입력할 수 있도록 지원함으로써 Segmentation 성능을 향상시킵니다.

- **Technical Details**: 연구에서는 DINOv2라는 파운데이션 모델을 기반으로 하여 내시경 수술에 특화된 세분화 모델을 개발했습니다. 이를 위해, 저차원 행렬(low-rank matrices) 삽입을 통해 모델의 레이어를 조정하여 전문 분야에서의 예측 성능을 최적화할 수 있는 Low-Rank Adaptation (LoRA) 방법론을 적용했습니다. 또한, 1,849장의 이미지를 포함한 ESD-DZSeg 데이터셋을 구축하여 다양한 시각적 프롬프트 요청을 대응할 수 있도록 했습니다.

- **Performance Highlights**: 결과적으로, PDZSeg 모델은 기존의 최첨단 세분화 접근법보다 뛰어난 성능을 보였습니다. 연구에서 제안한 방식은 수술 안전성을 높이는 데 기여하고, 사용자 경험을 향상시키는 데도 효과적입니다. 또한, 본 연구는 내시경 수술의 해부 구역 세분화 분야에서 비주얼 프롬프트 디자인을 통합한 첫 번째 연구로서, 향후 연구에 대한 기초를 마련합니다.



### KAN See Your Fac (https://arxiv.org/abs/2411.18165)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문에서는 Kolmogorov-Arnold Network (KAN)을 활용하여 최신 얼굴 인식 시스템과 개인 정보 보호 얼굴 인식 시스템에서의 임베딩-투-페이스 공격을 수행하는 새로운 접근 방식을 제시합니다. 기존의 개인 정보 보호 얼굴 인식 방법은 주로 입력 얼굴 이미지의 시각 정보를 감추는 데 집중했으나, 본 연구에서는 해당 시스템의 임베딩에서 얼굴 이미지를 추출하는 가능성을 처음으로 탐구합니다. Face Embedding Mapping (FEM) 모델이 제안되어 임베딩을 효과적으로 매핑하고 실시간으로 얼굴 이미지를 재구성하는 데 기여합니다.

- **Technical Details**: FEM-KAN 및 FEM-MLP라는 두 가지 변형을 통해 비선형 임베딩-투-임베딩 매핑을 수행하여 얼굴 이미지를 재구성합니다. FEM은 임베딩을 새로운 공간으로 매핑하여, 여러 개인 정보 보호 얼굴 인식 모델에서 유출된 얼굴 임베딩으로부터 사실적인 얼굴 이미지를 생성합니다. 실험을 통해 FEM의 효과성을 검증하며, 다양한 메트릭을 사용해 재구성된 얼굴 이미지의 품질을 평가합니다.

- **Performance Highlights**: 본 연구는 여러 실제 시나리오에서 FEM을 사용하여 방법의 효과성, 일반화, 강인성, 그리고 편향성을 테스트했습니다. FEM은 부분적으로 유출된 임베딩 및 보호된 임베딩으로부터도 유효하게 얼굴 이미지를 추출할 수 있음을 보여줍니다. 실험 결과 FEM-KAN 모델이 기존의 MLP 기반 모델보다 더 효과적인 비선형 매핑을 제공하며, 이로 인해 이미지 재구성의 정확도와 품질이 향상되었습니다.



### RPEE-HEADS: A Novel Benchmark for Pedestrian Head Detection in Crowd Videos (https://arxiv.org/abs/2411.18164)
Comments:
          17 pages, 8 figures, 7 tables

- **What's New**: 이 논문은 고밀도의 환경에서 보행자 머리 자동 탐지의 필요성을 강조하고, 이를 위해 RPEE-Heads 데이터셋을 새롭게 소개합니다. 이 데이터셋은 66개의 비디오 녹화에서 추출된 1,886장의 이미지에 109,913개의 머리 주석을 포함하고 있어 보행자 탐지에서의 정확성을 높이는 데 중점을 두었습니다. 또한, 이 논문은 최신 객체 탐지 알고리즘을 평가하고, 머리 크기가 탐지 정확성에 미치는 영향을 분석합니다.

- **Technical Details**: RPEE-Heads 데이터셋은 다양한 환경을 포괄하고 있으며, 해상도가 높고 정확하게 주석이 달린 이미지로 구성되어 있습니다. 알고리즘 성능 평가를 위해 You Only Look Once (YOLO) v9 및 Real-Time Detection Transformer가 사용되었으며, 각각 평균 정밀도 90.7% 및 90.8%를 기록했습니다. 연구의 초점은 고속 및 정확한 머리 탐지를 위해 특별히 설계된 데이터셋으로, 인식 시간은 각각 11ms 및 14ms로 측정되었습니다.

- **Performance Highlights**: 본 연구는 RPEE-Heads 데이터셋이 방향성과 정확성을 수반한 객체 탐지 모델의 개발에 어떻게 기여할 수 있는지를 보여줍니다. YOLO v9와 Real-Time Detection Transformer는 성능 비교에서 두드러진 결과를 보였으며, 데이터셋의 사용은 머신러닝 및 딥러닝 모델의 발전에 기여할 것입니다. 나아가, 이 데이터셋은 향후 보행자 머리 탐지 분야의 연구에 중요한 기초자료로 활용될 것으로 기대됩니다.



### Type-R: Automatically Retouching Typos for Text-to-Image Generation (https://arxiv.org/abs/2411.18159)
- **What's New**: 최근의 text-to-image 모델은 상세한 지시를 반영한 사실적인 이미지를 생성할 수 있지만, 이미지 내의 글자를 정확히 렌더링하는 데에는 여전히 큰 도전에 직면해 있습니다. 본 논문에서는 생성된 이미지의 텍스트 렌더링 오류를 후처리하는 Type-R이라는 방법을 제안합니다. Type-R은 생성된 이미지에서 오타를 감지하고, 오류가 있는 텍스트를 지우고, 누락된 단어에 대한 텍스트 박스를 다시 생성하며, 최종적으로 렌더링된 단어의 오타를 수정합니다.

- **Technical Details**: Type-R의 파이프라인은 자동화된 4단계로 구성됩니다: 1) 오류 감지, 2) 텍스트 지우기, 3) 레이아웃 재생성, 4) 오타 수정. 처음으로 생성된 이미지를 분석하기 위해, 장면 텍스트 감지 모델을 사용해 텍스트가 포함된 폴리곤 영역을 예측합니다. 이후, 텍스트 인식 모델이 지정된 영역 내의 단어를 해석하여 문자열을 추출하고, 이를 통해 생성된 텍스트와 진짜 텍스트 간의 오류를 식별합니다.

- **Performance Highlights**: Type-R은 최신 text-to-image 모델인 Stable Diffusion 및 Flux와 결합하여, 텍스트 렌더링 정확도를 극대화하면서도 이미지 품질을 유지합니다. 실험 결과, Type-R은 텍스트 렌더링의 정확성과 이미지 품질을 균형 맞추는 데 있어 우수한 성능을 보였습니다. 또한, 기존의 타이포그래피 중심 모델들과 비교했을 때, 더 나은 품질-정확성 트레이드오프를 보여줍니다.



### COREval: A Comprehensive and Objective Benchmark for Evaluating the Remote Sensing Capabilities of Large Vision-Language Models (https://arxiv.org/abs/2411.18145)
Comments:
          20 pages, 12 figures

- **What's New**: 최근 대규모 비전-언어 모델(large Vision-Language Models, VLM)의 눈부신 발전으로 일반 도메인 모델과 원격 감시(Earth observation) 전용 모델 모두 뛰어난 인식 및 추론 능력을 보여주고 있습니다. 그러나 VLM의 원격 감시 기능을 포괄적으로 평가할 수 있는 기준이 없다는 점은 많은 연구자들 사이에서 주요한 격차로 지적되어 왔습니다. 이를 해결하기 위해 COREval이라는 새로운 벤치마크가 제안되어 VLM의 다차원적인 원격 감시 능력을 평가하는데 초점을 맞추고 있습니다.

- **Technical Details**: COREval은 원격 감시의 2차원 주요 능력인 인식(perception)과 추론(reasoning)을 기준으로 총 6개의 하위 차원과 22개의 세부 작업(leaf tasks)을 구성하여 평가합니다. 데이터 수집은 전 세계 50개 도시에서 이루어졌으며, A/B/C/D 선택형 질문 형식을 채택함으로써 명확하고 객관적인 평가를 가능하게 합니다. 또한, 데이터 유출을 방지하기 위해 공개 데이터셋을 배제하고, 여러 위성 및 플랫폼을 통해 자율적으로 이미지를 수집하여 문제를 구성했습니다.

- **Performance Highlights**: COREval을 기반으로 총 13개의 대표적인 오픈소스 VLM을 평가한 결과, 기본적인 원격 감시 능력에서 긍정적인 성과를 보였으나, 세밀한 객체에 대한 인식 및 사례 간의 관계 추론에서 큰 도전에 직면하고 있음을 확인했습니다. 특히 기존 VLM들은 복잡한 원격 감시 장면에 대한 고급 추론 작업에서 상대적으로 낮은 성능을 보이며, 이를 통해 연구자들에게 개선 방향을 제시하고 있습니다. 이 결과들은 COREval이 원격 감시 분야에서 VLM의 발전에 중요한 참고자료가 될 것임을 시사합니다.



### Enhancing Visual Reasoning with Autonomous Imagination in Multimodal Large Language Models (https://arxiv.org/abs/2411.18142)
- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)에서 시각적 단서를 기반으로 하는 Chain-of-Thought (CoT) 패러다임을 확장하는 새로운 방법을 제안합니다. 기존 기법들이 시각적 단서 찾기에 중점을 두어 복잡한 시각적 장면을 처리하는 데 어려움을 겪었던 반면, 본 연구는 MLLMs가 입력 장면을 자율적으로 수정할 수 있도록 함으로써 추론 과정을 단순화하는 접근 방식을 제시합니다. 새로운 상상 공간을 도입하여 MLLMs가 native reasoning 능력을 바탕으로 시각적 변화를 수행할 수 있게 합니다.

- **Technical Details**: 제안된 접근 방식은 'plug-and-play' 상상 공간을 도입하여 MLLMs가 focus, ignore 및 transform과 같은 모듈식 작업을 자율적으로 수행하도록 합니다. 이를 통해 모델은 복잡한 시각적 장면을 단계적으로 재구성하고, CoT 추론을 위한 시퀀스의 수정된 장면을 작성할 수 있습니다. 이 과정은 추가적인 훈련이나 파인튜닝 없이 MLLMs의 내재적 추론 능력을 활용하여 이루어집니다.

- **Performance Highlights**: 연구진은 dense counting, simple jigsaw puzzle solving, object placement와 같은 주요 과제를 통해 제안된 방법의 유효성을 검증했습니다. 실험 결과, 기존 기술들이 종종 실패하는 반면, 본 연구의 접근 방식은 MLLMs가 자율적 상상력을 통해 단계별로 효과적으로 추론할 수 있도록 하여 보다 나은 성능을 발휘함을 입증했습니다. 이러한 결과는 MLLMs의 시각적 추론 능력을 획기적으로 확장하는 데 기여합니다.



### ModeDreamer: Mode Guiding Score Distillation for Text-to-3D Generation using Reference Image Prompts (https://arxiv.org/abs/2411.18135)
- **What's New**: 본 논문에서는 기존의 Score Distillation Sampling (SDS) 기반 방법이 가지는 과도한 평활화(over-smoothing) 및 낮은 품질의 출력을 해결하기 위해 새로운 이미지 프롬프트 스코어 증류 손실인 ISD를 도입합니다. ISD는 참조 이미지를 활용하여 텍스트에서 3D로의 최적화를 특정 모드에 유도하여 샘플링을 안정화합니다. 이를 통해 이전 텍스트-3D 방법들에 비해 높은 품질의 출력을 제공하며 최적화 속도 또한 개선되었음을 실험을 통해 증명하였습니다.

- **Technical Details**: ISD 손실은 IP-Adapter를 사용하여 텍스트-이미지 디퓨전 모델에 이미지 프롬프트 기능을 통합하는 경량 어댑터로 구현됩니다. 참조 이미지에 의해 유도되지 않는 이 어댑터의 변형은 스코어 추정의 분산을 줄이는 효율적인 제어 변수(control variate)로 활용될 수 있습니다. 스코어 증류 과정에서의 오차를 줄이고, 참조 이미지의 변동성과 충실도를 captures하는 다양하고 고품질의 3D 출력을 도출하는 것이 ISD의 핵심입니다.

- **Performance Highlights**: T3Bench 벤치마크에서의 정성적 및 정량적 평가 결과에 따르면, ISD는 시각적으로 일관되고 고품질의 출력을 지속적으로 생성하며, 전통적인 방법들에 비해 최소 30-40분만의 최적화 시간으로 우수한 성능을 보여줍니다. 기존 VSD 방법에 비해 훨씬 짧은 시간 내에 다양한 3D 개체를 생성할 수 있으며, 이는 높은 효율성을 나타냅니다. 마지막으로 제안된 방법은 출력된 3D 자산의 현실감과 형태를 보장하며, 같은 텍스트 프롬프트로부터 다양한 3D 개체를 생성할 수 있는 능력을 보여줍니다.



### Spectral-Spatial Transformer with Active Transfer Learning for Hyperspectral Image Classification (https://arxiv.org/abs/2411.18115)
- **What's New**: 이번 연구에서는 고해상도 하이퍼스펙트럴 이미지(HSI) 분류를 위한 새로운 다단계 활성 전이 학습(ATL) 프레임워크를 제안합니다. 이 프레임워크는 공간-스펙트럴 트랜스포머(SST) 모델과 활성 학습 프로세스를 통합하여 효율적인 HSI 분류를 가능하게 합니다. 특히 불확실성 기반의 샘플 선택 메커니즘을 통해 최적의 샘플을 적극적으로 선택함으로써, 모델의 불확실성을 줄이고 라벨링 비용을 최소화하는 데 기여하고 있습니다.

- **Technical Details**: 제안하는 SST-ATL 프레임워크는 사전 훈련된 SST 모델을 기반으로 하며, 동적 레이어 고정을 통해 계산 부담을 줄이면서도 새로운 데이터의 스펙트럴 변화에 적응할 수 있습니다. 이 방법은 불확실성 기반 쿼리 메커니즘을 사용해 스펙트럴 및 공간 주의 레이어를 자율적으로 보정하는 혁신적인 접근 방식을 제공합니다. 또한, 다양성을 촉진하는 샘플링 전략을 도입하여, 고유한 스펙트럴 클래스에 대한 오버피팅을 방지합니다.

- **Performance Highlights**: 실험 결과, SST-ATL 프레임워크는 기존의 CNN 및 SST 기반 방법들에 비해 뛰어난 정확도 및 효율성을 보여주었습니다. 다양한 HSI 데이터셋에서 성능이 개선되었으며, 최소한의 라벨링 노력을 통해 높은 분류 성능을 달성했습니다. 이 연구는 HSI 분류의 기존 한계를 극복하고, 더 나은 성능과 효율성을 제공하는 데 기여하고 있습니다.



### When Large Vision-Language Models Meet Person Re-Identification (https://arxiv.org/abs/2411.18111)
- **What's New**: 이번 연구에서는 LVLM(ReID) 프레임워크를 통해 사람 재식별(person re-identification)의 새로운 방향을 제시합니다. 제안된 방법은 LVLM의 생성적(generative) 기법을 활용하여 보행자의 모습을 캡슐화한 단일 시멘틱 토큰(semantic token)을 생성하며, 이를 통해 재식별 문제 해결에 기여합니다. 이 프레임워크는 추가적인 이미지-텍스트 주석 없이도 경쟁력 있는 성과를 달성하며, LVLM에서 생성된 시멘틱 정보의 활용 가능성을 보여줍니다.

- **Technical Details**: LVLM-ReID 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 비주얼 인코더(visual encoder), 비전-언어 연결기(vision-language connector), 그리고 LLM입니다. 비주얼 인코더는 이미지에서 풍부한 시각적 표현을 추출하고, 이를 비전-언어 연결기가 단어 임베딩(spatial embedding) 공간으로 변환합니다. 최종적으로 LLM는 인코딩된 시각적 콘텐츠를 기반으로 텍스트를 생성합니다. 이 구조는 이미지-텍스트 상호작용을 효율적으로 다룰 수 있게 하여, Semantiv-Guided Interaction(SGI) 모듈을 통해 재식별 성능을 극대화합니다.

- **Performance Highlights**: LVLM-ReID 방법은 여러 벤치마크에서 실험을 수행하여 상당한 성과를 보여주었습니다. 시멘틱 토큰을 활용한 재식별 성능은 이전의 방법들과 비교하여 확연히 향상되었으며, 이는 보행자의 표현 학습에 있어 강력한 지원을 제공합니다. 본 연구의 결과는 LVLM이 제공하는 시멘틱 정보를 활용하여 사람 재식별 성능을 크게 개선할 수 있음을 시사하며, 향후 연구를 위한 유망한 방향성을 제시합니다.



### Training Data Synthesis with Difficulty Controlled Diffusion Mod (https://arxiv.org/abs/2411.18109)
- **What's New**: 이번 논문에서는 실재-합성 하이브리드 반지도학습(Real-Synthetic Hybrid SSL, RS-SSL)이라는 새로운 과제를 도입하여, 합성 이미지가 오염된 라벨이 없는 데이터가 반지도학습(Semi-supervised Learning, SSL)에 미치는 영향을 조사합니다. 실험을 통해 기존의 SSL 방법들이 합성 이미지로 인해 개선되지 않거나 때로는 부정적인 영향을 받을 수 있음을 확인했습니다. 이를 해결하기 위해 RSMatch라는 새로운 SSL 방법을 제안하여, 라벨이 없는 합성 데이터를 효과적으로 판단하고 이를 모델 개선에 활용합니다.

- **Technical Details**: 합성 이미지의 오염이 SSL에 미치는 영향을 평가하기 위해 새로운 RS-SSL 기준을 설정했습니다. 우리는 이를 바탕으로, RSMatch를 통해 합성 이미지의 난이도(score)를 측정하고 이를 기반으로 생성 프로세스를 안내합니다. 이러한 방법은 난이도와 도메인 정합성을 분리하여, 다양한 난이도의 샘플을 생성할 수 있게 합니다.

- **Performance Highlights**: 광범위한 실험 결과는 RSMatch가 합성 라벨 없는 데이터를 '장애물'에서 '자원'으로 전환할 수 있음을 보여줍니다. 또한, 효과를 입증하기 위한 소멸 연구(ablation studies)와 시각화 과정을 통해, 난이도가 특정 샘플의 분류에 미치는 영향을 분석했습니다. 우리의 방법은 데이터셋의 다양한 특성을 시각화하는 도구로도 활용 가능성을 제공합니다.



### Aligning Knowledge Concepts to Whole Slide Images for Precise Histopathology Image Analysis (https://arxiv.org/abs/2411.18101)
- **What's New**: 이번 연구에서는 Whole Slide Images (WSIs) 분석을 위한 새로운 여러 사례 학습(Multiple Instance Learning, MIL) 프레임워크인 ConcepPath를 제안합니다. ConcepPath는 의학 문헌으로부터 질병 특정 인간 전문가 개념을 도출하기 위해 GPT-4를 활용하며, 이를 통해 훈련 데이터에서 보완적 지식을 추출할 수 있습니다. 비록 현재의 방법들이 이미지 데이터에서만 학습하는 반면, ConcepPath는 인간 전문가의 지식을 통합하여 분석의 정확성을 높입니다.

- **Technical Details**: ConcepPath는 CLIP 기반의 병리 비전-언어 모델을 활용하여 WSI와 언어적 지식 개념을 정렬합니다. 이러한 개념에 따라 인스턴스 특징을 집합 수준의 특징으로 집계하고, 다시 집합 수준 일반 표현을 생성합니다. ConcepPath는 다양한 구조의 질병을 전반적으로 반영하는 보완적 데이터 기반 개념을 학습하는데, 이는 복잡하고 연구가 부족한 진단 작업에서 매우 중요합니다.

- **Performance Highlights**: 연구 결과, ConcepPath는 비소세포 폐암(NSCLC), 유방암 HER2 평가(BRCA) 및 위암 면역요법 민감형 (STAD) 하위 유형 분류 작업에서 기존의 최신 방법보다 현저하게 높은 성능을 보였습니다. 특히, 위암의 Epstein–Barr virus(EBV)-양성 분류에서 약 7%의 성능 향상을 기록하는 등 환자 분류의 경제성과 효율성을 향상시키는 잠재력을 보였습니다.



### Training Noise Token Pruning (https://arxiv.org/abs/2411.18092)
Comments:
          25 pages, 8 figures

- **What's New**: 이번 연구는 Vision Transformers에 대한 Training Noise Token (TNT) Pruning 방법을 소개합니다. 이 방법은 discrete token dropping 조건을 continuous additive noise로 완화하여 훈련 중 부드러운 최적화를 가능하게 하며, 배포 환경에서는 여전히 discrete dropping의 계산 이점을 유지합니다. 이 연구는 Rate-Distortion 문헌과의 이론적 연관성을 제공하고, ViT 및 DeiT 아키텍처를 활용한 ImageNet 데이터셋에서 TNT의 장점을 입증합니다.

- **Technical Details**: 토큰 프루닝은 입력 길이를 줄여 transformer의 계산 부하를 경감하는 방법입니다. 본 논문에서는 토큰 dropping 비율을 채널 제약으로 보고, 정확도 페널티를 왜곡 측정으로 간주하여 정보 병목 현상 문제의 제약된 사례로 토큰 프루닝을 재구성합니다. 이는 연속 최적화의 바람직한 조건을 제공하며, 높은 차원의 토큰 상호작용 정보의 복잡성을 무시한 간단한 근사가 놀라운 성능을 나타냅니다.

- **Performance Highlights**: 제안된 TNT 방법은 기존의 stochastic discrete dropping 또는 attention heuristics 방법들과 비교하여 ImageNet-1K 기준에서 우수한 성능을 보여줍니다. 특히 많은 비율의 토큰을 제거할 때, 즉 낮은 토큰 비율(state)에 있을 때 두드러진 장점을 지닙니다. 실험 결과, TNT는 정확도와 계산 부하의 균형을 유지하는 최첨단 성능을 제공함을 입증하였습니다.



### Dual-view X-ray Detection: Can AI Detect Prohibited Items from Dual-view X-ray Images like Humans? (https://arxiv.org/abs/2411.18082)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 금지 품목 탐지를 위한 대규모 이중 시야 X-ray(dual-view X-ray) 데이터 세트인 LDXray를 소개합니다. LDXray 데이터 세트는 12개 카테고리에서 353,646개의 인스턴스로 구성되어 있으며, 기존의 단일 시야 이미지를 사용했던 데이터 세트의 한계를 극복하고 있습니다. 특히, LDXray는 실제 보안 환경에서 수집된 데이터를 기반으로 하여 높은 다양성과 대표성을 지니고 있습니다.

- **Technical Details**: 이 논문에서는 Auxiliary-view Enhanced Network(AENet)이라는 새로운 탐지 프레임워크를 제안합니다. AENet은 메인 뷰와 보조 뷰의 두 가지 탐지 파이프라인으로 구성되어 있으며, 메인 뷰 파이프라인은 일반 카테고리를 탐지하는 역할을, 보조 뷰 파이프라인은 더 어려운 카테고리를 처리하는 역할을 맡고 있습니다. 이중 파이프라인 접근 방식은 탐지 성능을 크게 향상시킵니다.

- **Performance Highlights**: LDXray 데이터 세트에서 AENet의 실험 결과, 우산 카테고리와 같은 어려운 탐지 카테고리에서 최대 24.7%의 성능 향상(33.9%에서 58.6%로)을 보였습니다. 더욱이, AENet은 7개의 다양한 탐지 모델을 사용하여 강력한 일반화를 보여 주며, 이는 다양한 기본 모델에 걸쳐 발견되었습니다. 향후 연구에 있어 이중 시야 데이터의 특징을 탐구하는 데 기여할 것으로 기대됩니다.



### Dual-Level Boost Network for Long-Tail Prohibited Items Detection in X-ray Security Inspection (https://arxiv.org/abs/2411.18078)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서는 X-ray 보안 검사에서 금지된 물품을 탐지하는 데 있어 장기 분포(long-tail distribution) 문제를 해결하기 위한 Dual-level Boost Network (DBNet)를 제안합니다. DBNet은 희귀 아이템의 신뢰도 높은 탐지를 위해 Poisson blending을 활용한 데이터 증대(data augmentation) 기법과, 객체와 환경 간의 상호작용을 포착하는 컨텍스트 인식 피처 향상 모듈을 포함하고 있습니다. 이 두 가지 혁신을 통해 모델의 성능을 개선하고, 특히 드문 범주에 대한 탐지 정확도를 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: DBNet은 두 가지 주요 구성 요소로 설계되었습니다. 첫째, Poisson blending을 사용한 데이터 증대 전략을 통해 희귀 아이템의 합성 인스턴스를 생성하여 데이터 불균형 문제를 완화합니다. 둘째, 컨텍스트 인식 피처 향상 모듈은 객체와 그 주변 간의 공간적 및 의미적 상호작용을 캡처하여, 희귀 항목의 분류 정확성을 높입니다. 이러한 접근 방식을 활용하여 DBNet은 탐지 정확도를 획기적으로 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, DBNet은 X-ray 보안 검사에서 기존 최첨단 방법들보다 17.2% 더 높은 탐지 성능을 기록하여 드문 범주에 대한 정확한 탐지 능력을 입증하였습니다. 이 방법은 공공 안전을 보장하는 데 기여하며, X-ray 보안 검사 시스템의 신뢰성을 높이는 강력한 솔루션을 제공합니다. 따라서, DBNet은 향후 자동화된 보안 시스템에서 중요한 역할을 할 것으로 기대됩니다.



### SmileSplat: Generalizable Gaussian Splats for Unconstrained Sparse Images (https://arxiv.org/abs/2411.18072)
- **What's New**: 본 논문에서는 새로운 일반화 가능한 Gaussian Splatting 방법인 SmileSplat을 제안하여, 제한 없는 희소 다중 뷰 이미지만으로 픽셀 정렬된 Gaussian surfel을 재구성할 수 있는 기술을 개발하였습니다. 이는 기존에 요구되던 카메라 매개변수 없이도 높은 품질의 radiance fields를 생성할 수 있도록 도와줍니다. 또한, 이를 통해 3D 비전 작업에서 매우 우수한 성능을 달성하는 것을 목표로 하고 있습니다.

- **Technical Details**: SmileSplat은 다중 헤드 Gaussian 회귀 디코더(multi-head Gaussian regression decoder)를 기반으로 Gaussian surfel을 예측하며, 높은 품질의 정상 벡터를 확보하기 위해 정상 우선 정보(normal priors)를 활용하여 향상된 성능을 보여줍니다. 제안된 방법은 특수한 카메라 매개변수 없이도 이미지의 내부 및 외부 매트릭스를 예측하여, 다중 뷰 이미지를 통해 고품질 Gaussian radiance fields를 생성합니다. 또한, Gaussian Splatting 번들 조정(Bundle Adjustment) 모듈을 최적화하여 뛰어난 3D 구조를 만듭니다.

- **Performance Highlights**: SmileSplat은 공개 데이터셋을 활용한 여러 실험을 통해 기존의 최첨단 방법들보다 뛰어난 성능을 보여줍니다. 이를 통해 다양한 3D 비전 업무에서 탁월한 성능을 발휘하며, 특히 희소 샷 작업에서의 소음 높은 환경에서도 안정적인 결과를 도출하는데 성공하였습니다. 논문에서는 이러한 성과를 토대로 많은 장면에서 경쟁력 있는 새로운 뷰 렌더링 성능을 입증하였습니다.



### PersonaCraft: Personalized Full-Body Image Synthesis for Multiple Identities from Single References Using 3D-Model-Conditioned Diffusion (https://arxiv.org/abs/2411.18068)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 PersonaCraft라는 새로운 접근법을 제안하며, 이는 diffusion 모델과 3D 인간 모델링을 결합하여 개인 맞춤형 이미지 생성의 한계를 극복합니다. PersonaCraft는 복잡한 장면에서 여러 사람의 이미지를 생성하는 데 특히 효과적이며, occlusion(가림 현상)을 효율적으로 관리합니다. 이를 통해 사용자가 자신의 몸매를 조정할 수 있는 유연성을 제공하여 개인적인 체형 맞춤화를 가능하게 합니다.

- **Technical Details**: PersonaCraft는 SMPLx 모델을 활용하고, SMPLx-ControlNet(SCNet)을 통한 3D-aware pose conditioning을 사용하여, 신체 형태와 자세에 대한 정밀한 제어를 제공합니다. 이 접근법은 얼굴 및 신체 아이덴티티 추출, 3D 포즈 조정, 다중 인물 개인화 이미지 합성을 포함한 세 가지 핵심 메커니즘으로 구성됩니다. 특히 SCNet은 복잡한 오클루전이 있는 멀티 휴먼 장면에서 강력한 제어를 가능하게 합니다.

- **Performance Highlights**: PersonaCraft는 정량적 및 정성적 평가를 통해 얼굴 아이덴티티, 신체 형태 및 자연스러운 인체 해부학을 잘 보존하며, 다중 인물이 포함된 복잡한 시나리오에서도 우수한 성능을 보여줍니다. 기존 방법들과 비교하여, 정확도와 개인화 면에서 뛰어난 성과를 이루어 고품질의 다중 인물 이미지 생성의 새로운 기준을 제시합니다.



### GLS: Geometry-aware 3D Language Gaussian Splatting (https://arxiv.org/abs/2411.18066)
Comments:
          Technical Report

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS)을 기반으로 한 GLS라는 통합 프레임워크를 소개합니다. GLS는 실내 표면 재구성과 개방 어휘 분할(Open-Vocabulary Segmentation) 분야를 탐색하여, 두 가지 작업 간의 상관관계를 강조합니다. 연구 결과, GLS는 MuSHRoom, ScanNet++ 및 LERF-OVS 데이터셋에서 두 작업 모두에서 기존 최첨단 기술을 초월하는 성과를 보여줍니다.

- **Technical Details**: GLS는 표면 재구성과 개방 어휘 분할을 동시에 최적화하여 실내 장면에서의 효율성과 강건성을 높이는 것을 목표로 합니다. 이를 위해 표면 법선(prior)과 2D CLIP 기능을 사용하여 개체 인스턴스 특징을 안내하고 DEVA 마스크를 활용하여 뷰 일관성을 높입니다. 또한, 복잡한 조명 및 재질을 가진 실내 환경에서 발생하는 노이즈를 줄이기 위해 구조적 규제(term)를 도입하여 성능을 개선합니다.

- **Performance Highlights**: 논문에서 제안된 GLS는 3DGS의 훈련 및 렌더링 효율성을 계승하면서도 표면 재구성과 개방 어휘 분할 작업에서 최첨단 정확도를 달성합니다. 실험 결과, GLS는 두 작업 모두에서 Superior한 성능을 보였으며, 이는 실내 장면 내에서 두 작업 간의 연결성을 효과적으로 활용했음을 보여줍니다. 결과적으로, 이 접근법은 AR/VR 및 구현된 인공지능(Embodied Intelligence) 분야에 광범위한 적용 가능성을 가지고 있습니다.



### Lightweight Gaze Estimation Model Via Fusion Global Information (https://arxiv.org/abs/2411.18064)
- **What's New**: 이 논문에서는 FGI-Net(Fusion Global Information)이라는 경량화된 시선 추정 모델을 제안합니다. 기존의 고정밀 모델들이 깊은 네트워크에 의존하여 파라미터가 크고 훈련 시간이 길어지는 문제를 해결하기 위해, 본 모델은 CNN에 글로벌 정보를 융합하여 다층 컨볼루션과 풀링의 필요성을 줄이는 구조입니다. 이로 인해 모델의 복잡성을 줄이고 정확도와 수렴 속도가 향상되었습니다.

- **Technical Details**: FGI-Net은 글로벌 정보 융합 모듈(Global Information Fusion module)을 활용하여 강력한 특성 추출 기능을 가지며, 유용한 채널 정보를 학습하여 점점 더 효율적으로 처리합니다. 이 모듈은 CNN을 사용하여 로컬 특성을 추출하고, Shift Window 메커니즘을 통해 글로벌 컨텍스트 정보를 융합하여 시선 방향 예측에 필요한 사전 정보를 제공합니다. 이 과정에서 dropout 기법을 적용하여 과적합을 방지하고 고차원 의미를 유지합니다.

- **Performance Highlights**: 실험 결과에 따르면, FGI-Net 모델은 GazeCaps와 비교하여 각도를 더 줄이고, 파라미터와 FLOPs를 각각 87.1%와 79.1% 감소시켰습니다. 다양한 아키텍처 모델과 비교할 때, FGI-Net은 훈련이 적은 반복으로 더 높은 정확도에 신속하게 수렴할 수 있으며, Gaze360 및 EyeDiap 데이터셋에서 GazeTR 모델과 비교해 각각 25%와 37.5% 더 적은 훈련 반복을 요구합니다.



### Multi-task Gaze Estimation Via Unidirectional Convolution (https://arxiv.org/abs/2411.18061)
- **What's New**: 이번 연구에서는 가벼운 모델을 이용한 가시점 추정(gaze estimation)에서 성능 저하 문제를 해결하기 위해 Multitask-Gaze라는 새로운 네트워크 모델을 제안합니다. 기존의 가벼운 네트워크는 채널(feature channels)이 적어 모델의 표현력이 제한되는데, Multitask-Gaze는 Unidirectional Convolution(UC), Spatial and Channel Attention(SCA), Global Convolution Module(GCM), Multi-task Regression Module(MRM)을 포함하여 성능을 향상시킵니다. 이 모델은 가벼운 특징을 유지하면서도 더 높은 accuracy를 달성할 수 있습니다.

- **Technical Details**: Multitask-Gaze 모델은 Feature Extractor, Global Convolution Module(GCM), Multi-task Regression Module(MRM)의 세 가지 주요 구성 요소로 이루어져 있습니다. 특히 UC는 깊은 합성을 단순화하고, SCA는 공간적 그리고 채널적 중요성을 강조합니다. GCM은 pooling layer를 대체하여 정보 손실을 방지하며, MRM은 각 작업 간의 상관관계를 증진시키고 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: Multitask-Gaze의 성능은 기존의 SOTA(State-of-the-art) 방법인 SUGE와 비교할 때 MPIIFaceGaze와 Gaze360 데이터셋에서 각각 1.71%와 2.75% 향상되었습니다. 뿐만 아니라 모델의 파라미터와 FLOPs 또한 각각 75.5%와 86.88%가 감소하여 연산 효율성이 크게 향상되었습니다.



### HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation (https://arxiv.org/abs/2411.18042)
- **What's New**: 이 논문은 멀티모달 대형 언어 모델(LLM)이 비디오 장면을 해석하는 데 있어 기존의 한계를 넘어서기 위해 HyperGraph를 통합한 Multimodal LLMs on a Scene HyperGraph(HyperGLM) 접근법을 제안합니다. HyperGLM은 객체 간의 공간적 관계를 포착하는 엔터티 장면 그래프와 그들의 인과적 전환을 모델링하는 절차적 그래프를 결합하여 복잡한 다자간 상호작용과 고차 관계를 추론합니다. 이 연구는 새로운 비디오 장면 그래프 추론(VSGR) 데이터셋을 도입하며, 총 190만 프레임으로 구성되어 있습니다.

- **Technical Details**: HyperGLM은 객체 간의 다차원적 관계를 표현하기 위해 하이퍼엣지를 활용하며, 이로 인해 LLM에서 복잡한 비디오 다이내믹스를 해석할 수 있는 능력이 향상됩니다. 또한, 멀티스텝 전환을 가능하게 하는 절차적 그래프를 통합하여 미래의 상호작용과 관계를 예측하는 데 필요한 정교한 모델링을 지원합니다. 본 접근법은 고전적인 쌍별 연결 방식을 넘어서는 고차적 관계 모델링을 통해 비디오 이해의 한계를 극복합니다.

- **Performance Highlights**: HyperGLM은 비디오 장면 그래프 생성 및 연기 예측, 비디오 질문 응답, 비디오 캡셔닝, 관계 추론 등 다섯 가지 작업에서 기존의 최첨단 방법들을 지속적으로 초월하는 성능을 보여주었습니다. 이 작업을 통해 HyperGLM은 복잡한 비디오 장면에서의 관계 모델링과 추론을 효과적으로 수행하여 비디오 이해의 새로운 영역을 개척하고 있습니다.



### VLM-HOI: Vision Language Models for Interpretable Human-Object Interaction Analysis (https://arxiv.org/abs/2411.18038)
Comments:
          18 pages

- **What's New**: 본 논문에서는 Large Vision Language Model (VLM)을 Human-Object Interaction (HOI) 탐지 작업에 활용하는 새로운 접근법(VLM-HOI)을 제안합니다. VLM의 언어 이해 능력을 활용하여 예상되는 HOI triplet의 유사성을 정량화하는 방법을 도입합니다. 이 접근법은 이미지-텍스트 매칭(Image-Text matching) 기술을 사용하여 HOI 탐지의 정확성을 향상시키며, 이는 기존 CLIP 모델보다 성능이 뛰어납니다.

- **Technical Details**: VLM-HOI에서는 HOI triplet을 언어적으로 표현하여 VLM의 언어 이해 능력을 최대한 활용합니다. 예측된 HOI triplet의 유사성은 이미지-텍스트 매칭 기법을 통해 계산됩니다. 우리의 방법론은 VLM의 지식을 명확히 분류하는 데 초점을 두며, 대조 학습(constrastive learning) 프레임워크를 적용하여 신경망이 텍스트 형태의 HOI를 이해할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 우리의 방법이 기존 방법에 비해 높은 정확도와 강 robustness를 달성했다는 것을 보여주었습니다. 우리는 HOI 탐지 벤치마크에서 최첨단 결과를 기록하며, VLM을 HOI 탐지에 통합하는 것이 인간-객체 상호작용의 보다 발전된 해석적 분석을 위한 중요한 진전을 나타낸다고 믿습니다.



### Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision (https://arxiv.org/abs/2411.18025)
- **What's New**: 이번 연구에서는 픽셀 정렬(píxel-aligned) RGB-NIR 스테레오 이미지를 캡처하고 LiDAR 포인트 클라우드를 동기화하여 수집할 수 있는 로봇 비전 시스템을 개발했습니다. 이 시스템은 다양한 조명 조건에서도 데이터를 수집할 수 있도록 설계되었으며, 80,000 프레임의 RGB-NIR 이미지와 LiDAR 포인트 클라우드를 포함하는 대규모 데이터셋을 제공합니다. RGB-NIR 이미지를 융합(Fusion)하는 두 가지 방법도 소개하여 기존 모델의 활용도를 극대화합니다.

- **Technical Details**: 이 시스템은 두 개의 프리즘 기반 RGB-NIR 듀얼 센서 카메라와 LiDAR 센서를 사용하여 구성됩니다. 각 RGB-NIR 카메라는 RGB 및 NIR 이미지를 독립적으로 캡처하며, 이는 다양한 조명 조건에서 NIR 가시성을 향상시키기 위해 액티브 NIR 조명을 결합한 결과입니다. 데이터 전송은 RJ-45 인터페이스를 통해 이루어지며, LiDAR는 지상 진실적인 희박한(depth sparse) 깊이 맵을 생성합니다.

- **Performance Highlights**: 실험 결과, 픽셀 정렬된 RGB-NIR 이미지를 활용함으로써 다양한 하위 작업에서 성능이 크게 향상되었음을 보여줍니다. RGB-NIR 이미지를 활용한 두 가지 새로운 방법(RGB-NIR 이미지 융합과 특징 융합)을 적용하여 기존 RGB나 NIR 데이터만 사용할 때보다 더 효과적인 결과를 얻었습니다. 특히 조명이 어려운 상황에서도 이 시스템의 성능이 뛰어나 임무 완수에 실질적인 도움이 될 수 있음을 입증했습니다.



### Manual-PA: Learning 3D Part Assembly from Instruction Diagrams (https://arxiv.org/abs/2411.18011)
- **What's New**: 이번 연구에서는 3D 부품 조립을 위한 새로운 프로세스를 제안합니다. 특히, 다이어그램 매뉴얼을 활용하여 기계 학습 모델이 조립 작업을 더 쉽게 수행할 수 있도록 돕고자 합니다. 우리는 Manual-PA라는 변형자(transformer) 기반의 프레임워크를 도입하여, 주어진 매뉴얼의 다이어그램과 3D 부품 간의 일치를 학습할 수 있습니다.

- **Technical Details**: Manual-PA는 대조 학습(contrastive learning) 기반으로 매뉴얼 다이어그램의 단계와 3D 부품 간의 의미적 유사성을 계산하여 대응 관계를 예측합니다. 이 프로세스에서 매뉴얼 단계를 매칭하여 조립 순서를 생성하고, 각 부품의 최종 6DoF 포즈를 예측합니다. 이때, 각 단계의 다이어그램은 자신의 해당 부품에 높은 주의를 받도록 위치 인코딩(positional encoding)이 가이드합니다.

- **Performance Highlights**: 우리는 PartNet 벤치마크 데이터셋에서 Manual-PA의 효과를 검증하였으며, 벤치마크 평가에서 전통적인 방법들 대비 뛰어난 성과를 보였습니다. IKEA-Manual 데이터셋을 통한 실제 환경 조립 실험에서도 높은 일반화 능력을 입증하였습니다. 이 연구는 조립 매뉴얼을 이용한 3D 조립 작업에서 최신 기술 수준의 성능을 나타내는 것을 목표로 합니다.



### AI-Driven Smartphone Solution for Digitizing Rapid Diagnostic Test Kits and Enhancing Accessibility for the Visually Impaired (https://arxiv.org/abs/2411.18007)
- **What's New**: 이번 연구에서는 스마트폰 기반 애플리케이션에 인공지능(AI) 알고리즘, 특히 CNN(Convolutional Neural Networks)을 통합하여 신속 진단 테스트 결과 해석의 정확성과 신뢰성을 향상시키는 새로운 방법을 제안합니다. 사용자가 테스트 키트의 사진을 찍으면 YOLOv8가 이를 처리하여 이미지의 가장자리에 위치한 키트도 정밀하게 잘라내고 추출할 수 있습니다. 이렇게 향상된 기능은 사용자가 완벽한 정렬 없이도 이미지를 캡처할 수 있게 해 시각적으로 장애가 있는 사람들에게도 접근성을 제공합니다.

- **Technical Details**: 앱은 사용자가 찍은 테스트 키트 사진을 YOLOv8 알고리즘을 통해 처리하여 멤브레인(membrane) 영역을 정확하게 추출합니다. 그 후, 추가적인 CNN 분류기를 통해 결과가 양성, 음성 또는 무효인지 분석하여 결과와 신뢰 수준을 사용자에게 제공합니다. SHapley Additive exPlanations(SHAP) 분석을 수행하여 모델의 결정에 영향을 미치는 요소를 조사하고 올바른 분류와 잘못된 분류의 이유를 밝혀냈습니다.

- **Performance Highlights**: 검증 실험을 통해 다양한 진단 응용 프로그램에서 많이 사용되는 신속 진단 키트에 대한 연구 결과는 AI의 통합이 테스트 결과 해석에서 민감도(sensitivity)와 특이도(specificity)를 크게 향상시킨다는 것을 보여줍니다. 이 개선은 최신 YOLO 알고리즘을 사용하여 테스트 키트 이미지에서 멤브레인 영역을 효과적으로 추출한 덕분입니다. 제안된 접근 방식은 진짜 테스트 라인을 배경 잡음으로부터 구별하고 테스트 라인의 강도 및 일관성에 대한 귀중한 통찰력을 제공하여 신속 테스트 해석의 문제를 해결하는 강력한 솔루션을 제공합니다.



### An End-to-End Two-Stream Network Based on RGB Flow and Representation Flow for Human Action Recognition (https://arxiv.org/abs/2411.18002)
Comments:
          6 pages, 3 figures, 9 tables

- **What's New**: 이번 연구에서는 딥러닝의 발전을 바탕으로 비디오 기반 동작 인식을 위한 두 개의 스트림 신경망(two-stream neural networks) 모델에서 광학 흐름(optical flow) 대신 표현 흐름(representation flow) 알고리즘을 도입했습니다. 이를 통해 egocentric action recognition 모델을 위한 최적화된 엔드 투 엔드(end-to-end) 훈련을 지원하며, 계산 비용과 예측 시간을 줄일 수 있었습니다.

- **Technical Details**: 모델은 클래스 활성화 맵(class activation maps, CAMs)을 적용하여 정확성을 개선하고, 시공간(spatio-temporal) 인코딩을 위한 ConvLSTM을 활용하여 공간주의(spatial attention)를 적용합니다. GTEA61, EGTEA GAZE+, HMDB 데이터셋에서 평가했을 때, 제안한 모델은 GTEA61에서 원래 모델의 정확도와 일치하고, EGTEA GAZE+와 HMDB에서 각각 0.65%와 0.84% 향상된 성능을 보였습니다.

- **Performance Highlights**: 예측 런타임(prediction runtimes)은 기존 모델에 비해 현저히 감소하여 GTEA61, EGTEA GAZE+, HMDB에서 각각 0.1881초, 0.1503초, 0.1459초를 기록했습니다. 이는 기존 모델의 101.6795초, 25.3799초, 203.9958초와 비교했을 때 매우 인상적인 성능 향상으로, 실질적인 적용 가능성을 시사합니다.



### Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models (https://arxiv.org/abs/2411.18000)
- **What's New**: 이번 연구에서는 Vision-Language Models (VLMs)의 보안 취약성과 관련된 새로운 발견을 제시합니다. 특히, VLMs가 시나리오에 맞는 이미지를 이용한 공격에 더 취약하다는 점과, 최소 손실(minimal loss) 값이 항상 최적의 공격 효과를 보장하지 않는다는 것을 밝혔습니다. 이러한 통찰을 바탕으로, Multi-Loss Adversarial Images (MLAI)라는 새로운 jailbreak 프레임워크를 제안하며, 이를 통해 더 강력한 공격을 수행할 수 있는 방법을 탐구합니다.

- **Technical Details**: 제안한 MLAI 방법론은 세 가지 단계로 구성됩니다. 첫째, 공격 대상 시나리오에 맞는 초기 이미지를 생성하여 모델이 위험한 응답을 생성할 가능성을 높입니다. 둘째, 생성된 이미지를 그래디언트 업데이트를 통해 최적화하며, 정해진 손실 범위 내의 결과를 저장합니다. 마지막으로, 최적의 손실 범위 내에서 여러 이미지를 선택하여 협력적 공격을 수행합니다. 이 과정에서, MLAI는 다양한 모델에서 공격 성공률을 유지하는 뛰어난 전이 가능성을 보여줍니다.

- **Performance Highlights**: MLAI는 MiniGPT-4에서 77.75%, LLaVA-2에서 82.80%의 공격 성공률을 기록하며 기존의 방법들보다 크게 향상된 성과를 보였습니다. 또한, MLAI는 상용 블랙박스 VLMs에 대해서도 최대 60.11%의 공격 성공률을 달성하는 등 높은 효과성을 나타냈습니다. 이 연구는 현재 VLM의 안전 메커니즘에서 근본적인 시각적 취약점을 드러내며, 더 강력한 방어 수단이 필요하다는 점을 강조합니다.



### Revisiting Misalignment in Multispectral Pedestrian Detection: A Language-Driven Approach for Cross-modal Alignment Fusion (https://arxiv.org/abs/2411.17995)
- **What's New**: 본 논문에서는 멀티스펙트럼 보행자 탐지(multi-spectral pedestrian detection)를 위한 새로운 프레임워크를 제안합니다. 기존의 교정을 필요로 하는 방법과 달리, 이 방법은 비모든 경비가 매우 비싼 비용과 복잡한 초기 전처리(calibration)를 요구하지 않습니다. 특히, RGB 및 열화상 이미지라는 서로 다른 모달리티에서의 의미적 정보(semantic information) 정렬을 통해 탐지 정확도를 향상시키는 것이 목표입니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 보행자 좌표를 나타내는 노드와 인접한 노드 간의 거리를 의미하는 엣지로 구성된 위치 그래프(positional graphs)를 작성합니다. 둘째, 각 보행자의 고유한 외형 정보를 나타내는 임베딩을 통해 개인을 인식합니다. 마지막으로, 빅스케일 비전-언어 모델(large-scale vision-language model)을 사용하여 이미지로부터 추출된 정보에 근거하여 예측을 수행합니다.

- **Performance Highlights**: 제안된 시스템은 비교적 낮은 정확도를 보였던 기존 멀티스펙트럼 보행자 탐지 모델에 비해 상당한 성능 향상을 보여주었습니다. 실험에서 실제 사용되는 한정된 데이터 세트를 기반으로 모델의 효과가 입증되었으며, 비용이 많이 드는 교정 장치나 데이터 전처리 없이 직관적인 방법으로 보행자 탐지를 가능하게 합니다. 따라서 이를 통해 멀티스펙트럼 탐지 기술의 실용성이 크게 확대될 것으로 기대됩니다.



### Differentiable Inverse Rendering with Interpretable Basis BRDFs (https://arxiv.org/abs/2411.17994)
Comments:
          This paper is submitted to CVPR 2025. This is a different paper from my previous paper "Differentiable Point-based Inverse Rendering". It must not be removed automatically

- **What's New**: 본 논문에서는 해석 가능한 basis BRDF(기준 반사 분포 함수)를 생성하는 차별화 가능한 역 렌더링(differentiable inverse rendering) 방법을 제안합니다. 이 방법은 2D Gaussian을 사용하여 장면을 모델링하고, 각 Gaussian의 반사율을 기준 BRDF의 가중 혼합으로 정의합니다. 이를 통해 복잡한 장면에도 유연하게 적용 가능한 역 렌더링을 가능하게 하며, 각각의 Gaussian은 몇 개의 기준 BRDF만을 사용하여 반사율을 표현합니다.

- **Technical Details**: 우리의 기법은 2D Gaussian을 기하학적 원시 형태로 사용하고, 각 Gaussian의 위치와 모양은 여러 매개변수로 정의됩니다. 분석-합성(analysis-by-synthesis) 최적화 과정에서 기준 BRDF의 수를 동적으로 조정하며, 이는 위반의 해석 가능성과 희소성(sparsity)을 촉진합니다. 또한 가중된 포토메트릭 손실을 도입하여 훈련의 안정성을 향상시킵니다.

- **Performance Highlights**: 제안하는 방법은 정확한 기하학적 구조를 복원할 뿐만 아니라, 해석 가능한 기준 BRDF를 생성하여 장면 편집(scene editing) 및 물리 기반의 새로운 뷰 조명(novel-view relighting)을 지원합니다. 실험 결과, 기존 방법과의 비교를 통해 우리의 접근법이 더 나은 해석 가능성과 성능을 제공함을 입증했습니다.



### VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Forma (https://arxiv.org/abs/2411.17991)
Comments:
          9 pages

- **What's New**: 이번 연구는 VideoLLMs(영상 대형 언어 모델)와 사용자 간의 상호작용 형식을 집중적으로 탐구합니다. 전통적인 방식은 사용자가 전체 비디오와 질의를 입력한 뒤 모델이 응답을 생성하는데, 이는 실시간 응답 필요 시나리오에서 제한적입니다. 새로운 비디오-텍스트 듀엣(interaction duet) 형식에서는 비디오가 지속적으로 재생되며 사용자가 언제든지 텍스트 메시지를 삽입할 수 있습니다.

- **Technical Details**: 비디오-텍스트 듀엣 상호작용 형식은 MMDuetIT라는 훈련 데이터 세트를 기반으로 구축되어, 비디오 프레임을 순차적으로 모델에 입력하는 방식입니다. 이 방식은 대화의 주체로서 비디오 스트림을 정의하고, 사용자의 텍스트 메시지가 끝나면 비디오가 계속 재생됩니다. 이를 통해 시간 민감한 작업을 위한 응답 생성이 개선됩니다.

- **Performance Highlights**: MMDuet 모델은 MMDuetIT 훈련 데이터를 사용하여 시간 민감한 작업에서 높은 성능을 발휘합니다. YouCook2의 밀집 비디오 캡셔닝에서 76% CIDEr, QVHighlights 하이라이트 검출에서 90% mAP, Charades-STA의 시간 비디오 기초화에서 25% R@0.5의 성과를 보여주었습니다. 이러한 개선은 비디오가 재생되는 동안 실시간으로 응답하는 능력을 가능하게 합니다.



### RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Mod (https://arxiv.org/abs/2411.17984)
Comments:
          18 pages, 9 figures and 9 tables

- **What's New**: 본 논문에서는 전통적인 특화 모델 설계에서 벗어나 멀티모달 원격 감지 기능을 제공하는 RS-vHeat 원격 감지 기반 모델을 제안합니다. 이 모델은 고해상도 원격 감지 이미지의 지역 상관 관계를 물리학의 열 전도 모델을 차용하여 시뮬레이션합니다. 이를 통해 기존의 파라미터 수가 많은 모델에 비해 계산 효율성을 크게 개선하고, 메모리 소비를 84% 감소, FLOPs를 24% 줄이고 2.7배의 처리량을 향상시켰습니다.

- **Technical Details**: RS-vHeat 모델은 Heat Conduction Operator (HCO)를 활용하여 지역 구조 정보를 캡처하고, Frequencies distribution representations을 학습하는 자기 지도 전략을 기반으로 설계되었습니다. 이 모델은 3백만 개의 광학 및 SAR 데이터를 이용한 프리 트레이닝을 통해 다중 모달 RS 데이터를 처리합니다. HCO 블록에서의 열 전도 과정을 통해 다중 스케일 이미지의 글로벌 기능을 직접적으로 연산하여 지역 세부 사항을 캡처합니다.

- **Performance Highlights**: RS-vHeat는 10개의 데이터셋과 4개의 태스크에서 처리 효율성과 정확성을 유지하며, 기존의 상태-of-아트模型들보다 성능을 향상시켰습니다. 특히 대규모 이미지를 처리할 때, RS-vHeat는 메모리 사용량을 84% 줄이고, FLOPs를 24% 감소시키며, 처리량을 2.7배 향상시키는 성과를 보였습니다.



### Vision Mamba Distillation for Low-resolution Fine-grained Image Classification (https://arxiv.org/abs/2411.17980)
- **What's New**: 본 논문에서는 저해상도 세부 이미지 분류(low-resolution fine-grained image classification)에서 효과성과 효율성을 향상시키기 위해 비전 맘바 증류(Vision Mamba Distillation, ViMD) 방법을 제안합니다. 이 방법은 경량 슈퍼 해상도 비전 맘바 분류 네트워크(SRVM-Net)를 통해 비주얼 특징 추출 능력을 개선하고, 원래 맘바(mamba) 모델링으로 분류 서브 네트워크를 재설계합니다. 이를 통해 고해상도 비전 맘바 분류 네트워크(HRVM-Net)에서 수집한 지식을 SRVM-Net으로 전달하는 다단계 맘바 지식 증류 손실(multi-level Mamba knowledge distillation loss)을 도입하여 성능을 높입니다.

- **Technical Details**: ViMD의 구조는 SRVM-Net(학생)과 HRVM-Net(교사), 그리고 다단계 맘바 지식 증류 손실으로 구성되어 있습니다. 저해상도(LR) 이미지와 그에 해당하는 고해상도(HR) 이미지를 각각 SRVM-Net과 HRVM-Net에 입력하여 학습합니다. SRVM-Net은 SR 서브 네트워크와 ViM 분류 서브 네트워크로 구성되어 있으며, SR 서브 네트워크는 LR 이미지를 고해상도로 복원하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 ViMD 방법이 기존의 다른 최신 기술(SOTA) 방법들에 비해 향상된 정확도를 기록했습니다. 본 방법은 적은 수의 매개변수와 FLOPs를 가지면서도 보다 높은 정확성을 제공하여 임베디드 디바이스에서 더 적합하게 설계되었습니다. 본 논문은 7개의 공개 세부 분류 데이터셋에서 성능을 검증하였으며, 명확한 이점이 있음을 보여줍니다.



### Improved implicit diffusion model with knowledge distillation to estimate the spatial distribution density of carbon stock in remote sensing imagery (https://arxiv.org/abs/2411.17973)
Comments:
          Under review

- **What's New**: 본 연구는 중국 유난성 구징시 후이즈 카운티를 중심으로 GF-1 WFV 위성 이미지를 활용하여 숲의 탄소 저장량을 추정하는 데 혁신적인 접근 방식을 제안합니다. VGG와 UNet 모델을 기반으로 한 KD-VGG 및 KD-UNet 모듈을 통해 초기 특징을 추출하고, 개선된 암시적 확산 모델(IIDM)을 도입하여 정확성을 높였습니다. 본 연구의 결과는 새로운 AI 생성 내용(AIGC) 활용 가능성을 보여줍니다.

- **Technical Details**: 자세히 살펴보면, VGG 모듈은 초기 특징 추출을 개선하고, 모델 파라미터 최적화를 통해 정확성과 추론 시간을 단축했습니다. Cross-attention과 MLP의 결합은 지방 및 글로벌 특징 간의 관계를 효과적으로 포착하여 고정확도의 탄소 저장량 추정을 달성했습니다. IIDM 모델은 12.17%의 RMSE를 기록하였으며, 회귀 모델에 비해 41.69%에서 42.33%로 개선되었습니다.

- **Performance Highlights**: 결과적으로 본 연구에서 사용된 16미터 해상도의 추정치는 지역 탄소 저장량 관리를 위한 강력한 기초 자료를 제공합니다. 동시에, 개선된 암시적 확산 모델은 탄소 저장량 추정 시 다른 모델에 비해 깊은 특징 추출에 뛰어난 것으로 평가되었습니다. 이러한 성과는 숲 탄소 저장량 규제 및 의사결정 지원에 중요한 이론적 토대를 마련해 줍니다.



### Optimization-Free Image Immunization Against Diffusion-Based Editing (https://arxiv.org/abs/2411.17957)
Comments:
          Project webpage: this https URL

- **What's New**: DiffVax는 이미지 면역화(img immunization)를 위하여 기존의 이미지 방어 기법들을 혁신한 새로운 프레임워크입니다. 이 접근법은 Diffusion 기반 편집을 방지하기 위해 설계되었으며, 상당한 속도 향상과 스케일 가능성을 보장합니다. 기존 방법들이 각 이미지에 대해 시간이 많이 소요되는 최적화를 필요로 하는 반면, DiffVax는 이러한 과정을 필요로 하지 않고도 밀리초 단위로 처리할 수 있습니다.

- **Technical Details**: DiffVax는 모델이 특정 손상을 입지 않도록 하는 두 가지 손실(loss) 항을 사용하여 훈련됩니다. 첫 번째는 생성된 노이즈가 인지되지 않도록 보장하며, 두 번째는 편집 시도가 실패하도록 강제합니다. 이 방법은 메모리 효율성을 높여 기존 방식의 한계를 극복하고, JPEG 압축 및 이미지 디노이징과 같은 반격(counter-attack)에 대한 저항력을 발휘합니다.

- **Performance Highlights**: DiffVax는 기존의 면역화 방법들보다 훨씬 뛰어난 성능을 보이며, 뚜렷한 편집 성능 저하를 이끌어냅니다. 비디오 콘텐츠까지 면역화할 수 있는 최초의 성과를 이뤄냈으며, 실시간 면역화가 가능하다는 점에서 매우 실용적입니다. 다양한 검증을 통해 높은 확률의 성공률을 자랑하며, 방어 기법으로서의 우수성을 확립하고 있습니다.



### ROICtrl: Boosting Instance Control for Visual Generation (https://arxiv.org/abs/2411.17949)
Comments:
          Project page at this https URL

- **What's New**: 이 논문은 자연어 처리와 시각 생성 간의 연관성을 개선하기 위해 새로운 접근 방식을 제시합니다. 기본적으로, 지역 인스턴스 제어(Regional Instance Control)를 도입하여 각 인스턴스를 바운딩 박스(bounding box)와 자유형 캡션(free-form caption)으로 관리함으로써, 기존의 텍스트 기반 생성 모델들이 겪는 몇 가지 한계를 극복하고자 합니다. 이로 인해 고해상도 특성 맵에서의 명확하고 효율적인 ROI(Region of Interest) 조작이 가능해졌습니다.

- **Technical Details**: 논문에서는 ROI-Align와 ROI-Unpool이라는 두 가지 작업을 결합하여 인스턴스 제어의 정확성과 효율성을 높였습니다. ROI-Unpool 작업은 잘라낸 ROI 특성을 원래 위치로 복원하여, 처리 비용이 원래 특징 크기와 무관하게 유지될 수 있도록 돕습니다. 제안하는 ROICtrl 어댑터는 기존의 의존하는 여러 diffusion 모델 및 플러그인과 호환되어 다중 인스턴스 생성을 위한 폭넓은 활용이 가능합니다.

- **Performance Highlights**: ROICtrl은 ROICtrl-Bench라는 새 벤치마크에서 그 성과를 입증하였으며, 인스턴스 제어에서 우수한 성능을 기록했습니다. 기존 방식들에 비해 계산 비용을 상당히 줄이면서도 높은 성능을 제공합니다. 본 논문은 인스턴스 제어에서의 효율성과 정확성을 중요하게 다루며, 다양한 방면에서 실험이 진행되었습니다.



### MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation (https://arxiv.org/abs/2411.17945)
- **What's New**: 이번 연구에서는 MARVEL-40M+라는 4천만 개의 텍스트 주석을 포함한 대규모 데이터세트를 소개합니다. 이 데이터세트는 890만 개의 3D 자산을 포괄하며, 7개의 주요 3D 데이터세트에서 집계되었습니다. 또한, 새로운 다단계 주석 파이프라인을 통해 자동으로 고급 및 간결한 설명을 생성하여 정밀한 3D 재구성과 빠른 프로토타입 생성에 기여합니다.

- **Technical Details**: MARVEL의 플랫폼은 개방형 소스의 멀티뷰 VLMs(Visual Language Models)와 LLMs(Large Language Models)를 활용하여 다단계 주석을 생성하는 데 초점을 맞추고 있습니다. 이 파이프라인은 전체 재구성을 위한 세부 설명(150-200 단어)에서 빠른 모델링을 위한 간단한 태그(10-20 단어)까지 다양한 수준의 주석을 제공합니다. 또한, 소스 데이터세트에서의 인간 메타데이터를 통합하여 VLM의 허상 문제를 감소시킵니다.

- **Performance Highlights**: MARVEL-40M+ 데이터세트는 GPT-4와 인간 평가자에 의해 각각 72.41%와 73.40%의 승률을 기록하며 기존 데이터세트보다 훨씬 우수한 주석 품질과 언어 다양성을 보여주었습니다. MARVEL-FX3D라는 두 단계의 텍스트-3D 생성 파이프라인은 텍스트를 통해 15초 내에 텍스처를 가진 메쉬를 생성하는 데 성공했습니다. 이 연구는 높은 충실도의 TT3D 생성에서 현 상태의 방법들을 능가하는 성능을 입증하였습니다.



### Exploring Superpixel Segmentation Methods in the Context of Citizen Science and Deforestation Detection (https://arxiv.org/abs/2411.17922)
Comments:
          Paper was accepted for presentation at SAC 2025

- **What's New**: 이번 연구에서는 열대 숲의 파괴와 퇴화를 모니터링하기 위해 22가지 슈퍼픽셀 기반 분할 방법을 분석하며, 이를 통해 시민 과학 캠페인에서 사용할 수 있는 최적의 분할 방법을 규명하고자 합니다. 분석 결과, 기존의 ForestEyes 시민 과학 프로젝트에서 사용되는 SLIC(단순 선형 반복 클러스터링) 방법보다 뛰어난 성능을 나타내는 7가지 방법이 확인되었습니다. 이러한 발견은 시민 과학 캠페인의 발전 단계에서 개선의 기회를 제시합니다. 이 연구는 열대 산림 보존을 위한 효과적인 접근법을 모색하는 데 기여할 것입니다.

- **Technical Details**: 시민 과학은 전문가가 아닌 시민들이 데이터 수집 및 분석에 참여하여 과학적 활동에 기여하는 접근법입니다. 이 연구는 슈퍼픽셀(segmentation) 분할 방법을 통해 원거리 감지(remote sensing) 이미지를 세분화하여 산림 및 파괴된 지역을 식별합니다. 필요한 것은 비전문가가 이해할 수 있는 세그먼트를 생성하는 강력한 분할 방법으로, 일관된 픽셀과 잘 정의된 경계를 갖추어야 합니다. 본 연구에서는 다양한 전략을 사용하는 22개의 슈퍼픽셀 분할 방법의 성능을 평가합니다.

- **Performance Highlights**: 분석된 슈퍼픽셀 방법 중에 7개가 기존 SLIC 방법보다 뛰어난 성능을 보여 주목받았습니다. 이는 ForestEyes 프로젝트의 시민 과학 캠페인에 사용될 더 나은 세그먼트 생성을 위한 기회를 제공합니다. 슈퍼픽셀 분할 방법들은 종종 계산 복잡성이 낮고 데이터 전처리를 통해 처리 속도를 향상시킵니다. 이 연구는 시민 과학 프로젝트에서의 데이터 품질 향상 및 사용자 참여를 확대하는 데 기여합니다.



### DECODE: Domain-aware Continual Domain Expansion for Motion Prediction (https://arxiv.org/abs/2411.17917)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문에서는 DECODE라는 새로운 지속적 학습 프레임워크를 제안합니다. DECODE는 사전 훈련된 일반화 모델에서 시작하여 점진적으로 특정 도메인에 맞춤화된 전문 모델을 발전시키는 구조를 가지고 있습니다. 이는 다양한 시나리오에 일반화할 수 있는 통합 모델을 개발하려는 기존 접근 방식과는 달리, 전문성과 일반화를 동적으로 조절하여 실시간 요구 사항에 맞춰 변형할 수 있게 합니다.

- **Technical Details**: DECODE 프레임워크는 하이퍼네트워크(hypernetwork)를 활용하여 모델 매개변수를 동적으로 생성함으로써 저장 공간을 크게 줄입니다. 또한, 실제 상황에서 최적의 전문 모델을 선택하기 위해 정규화 흐름(normalizing flow) 메커니즘을 통합하여, 가능성 추정(likelihood estimation)에 기반한 적합한 모델을 선택합니다. 이 프레임워크는 또한 깊은 베이지안 불확실성 추정(deep Bayesian uncertainty estimation) 기법을 통해 가장 관련성이 높은 전문 모델과 일반화 모델의 출력을 통합하여 예측의 견고성을 보장합니다.

- **Performance Highlights**: DECODE 프레임워크는 0.044의 매우 낮은 망각률을 기록하며, 평균 최소 ADE(minADE) 0.584 m를 달성하여 전통적인 학습 전략을 앞서갔습니다. 여러 도메인을 포괄하는 데이터에서 실험을 통해 높은 품질의 모션 예측을 지속적으로 생성하는 능력을 입증하였으며, 안전이 중요한 시나리오에서도 0.986의 정확도를 자랑합니다. 이러한 성과는 DECODE 프레임워크의 효과성과 우수한 성능을 강력히 증명합니다.



### Passive Deepfake Detection Across Multi-modalities: A Comprehensive Survey (https://arxiv.org/abs/2411.17911)
Comments:
          26 pages

- **What's New**: 이 논문은 생성적 인공지능(Generative Artificial Intelligence, GenAI) 모델이 악의적 목적으로 활용되는 딥페이크(deepfake, DF)의 탐지 방법을 포괄적으로 조사합니다. 기존의 단일 모달리티에 국한된 패시브(passive) DF 탐지 접근 방식들을 넘어, 이미지, 비디오, 오디오 및 다중 모달리티(multi-modal) 분야에서의 적용을 설명합니다. 이 연구에서는 탐지 정확성 외에도 일반화(generalization), 강인성(robustness), 출처 귀속(attribution), 해석 가능성(interpretability) 등의 측면을 논의하며, 현재의 탐지 시스템에서 직면하는 도전과제를 강조합니다.

- **Technical Details**: 패시브 DF 탐지는 콘텐츠의 생성 과정에 대한 사전 지식 없이 생성 후 합성 콘텐츠를 식별하는 방법입니다. 연구진은 다양한 접근 방식을 세 가지 주요 원칙과 기술에 따라 분류하였으며, 이들은 각기 다른 장단점을 분석합니다. 조사 방법론 측면에서, IEEE Xplore, ACM Digital Library, Google Scholar와 같은 데이터베이스를 이용하여 최근 5년간의 인공지능(AI) 및 보안 관련 논문을 포괄적으로 검색하였습니다.

- **Performance Highlights**: 연구는 패시브 DF 탐지 접근 방식의 효과성을 다각도로 비교하고, 일반화 능력과 적대적 공격에 대한 강인성을 가져다 줄 미래 연구 방향을 제안합니다. 또한, 적대적 전략 및 다양한 수준의 공격자의 지식과 능력에 대한 위협 모델(threat models)을 탐구하여, robust detection system을 위한 정보 보안 측면에서의 중요 통찰을 제공합니다. 마지막으로, 변화하는 DF 생성 모델로부터의 일반화 부족과 충분한 신뢰성 평가의 필요성을 포함한 현재의 도전과제를 제시합니다.



### Automating grapevine LAI features estimation with UAV imagery and machine learning (https://arxiv.org/abs/2411.17897)
Comments:
          Accepted in 2024 IEEE INTERNATIONAL WORKSHOP ON Metrology for Agriculture and Forestry

- **What's New**: 이 연구는 드론 이미지 데이터를 활용하여 포도 나무의 잎 면적 지수(Leaf Area Index, LAI)를 자동으로 추정하는 방법을 제시합니다. 기존의 전통적인 방법은 시간 소모적이고 파괴적이며 비용이 많이 드는 반면, 이 방법은 기계 학습 모델을 통해 속도와 효율성을 높입니다. 딥 러닝 기반의 특징 추출이 기존 방법보다 효과적이라는 결과를 보여줍니다.

- **Technical Details**: 연구에서 사용된 데이터 세트는 드론으로 촬용한 개별 포도 나무 이미지와 그에 해당하는 LAI 값을 포함하고 있습니다. 다양한 특징 추출 기법을 사용하여 이미지에서 유용한 정보를 추출하고, 이를 통해 기계 학습 모델을 학습시킵니다. 세 가지 특징 추출 방법, 즉 엣지 감지를 활용한 녹색 영역 특징 추출, 특징 어휘 개발, 그리고 사전 훈련된 딥러닝 모델을 통한 특징 추출이 사용되었습니다.

- **Performance Highlights**: 이 연구는 세 가지 기계 학습 회귀 모델인 선형 회귀(Linear Regression), 서포트 벡터 머신(Support Vector Machines), 그리고 랜덤 포레스트(Random Forest)를 사용해 LAI를 추정하였습니다. 실험 결과, 복잡한 이미지 데이터를 효과적으로 분석할 수 있는 사전 훈련된 ResNet50 모델을 통해 LAI 예측의 효율성과 강건성을 크게 향상시켰습니다. 본 연구의 새로운 접근법은 정밀 농업 관행을 개선할 수 있는 가능성을 보여줍니다.



### Multimodal Crash Likelihood Prediction: A Complexity-Infused Approach Integrating Semantic, Contextual, and Driving Features (https://arxiv.org/abs/2411.17886)
- **What's New**: 이번 연구는 도로 복잡성(roadway complexity)에 대한 종합적 연구를 통해 교통사고 예측을 향상시키기 위한 새로운 두 단계 프레임워크를 소개합니다. 기존의 통계 모델과 딥러닝 기법은 주로 각각의 특성을 단독적으로 분석했으나, 본 연구는 이들 요인을 결합하여 도로 복잡성을 평가하고자 합니다. 이 프레임워크는 원본 데이터와 복잡성이 주입된 특성을 모두 사용하여 사고 발생 가능성을 예측하며, 그 정확도는 원본 특성만 사용할 때 87.98%, 추가된 복잡성 주입 특성을 통할 때 90.15%에 달했습니다.

- **Technical Details**: 연구는 MIT-AVT 데이터셋을 사용하여 도로 복잡성을 포함한 사고 예측 모델을 구축합니다. 이 데이터셋은 다양한 모달 데이터 소스를 포함하며, 각 비디오 클립에서 추출된 10,300개의 프레임에 대해 세 가지 세트의 특성을 생성합니다: 컴퓨터 비전 알고리즘으로 도출된 의미적 특성(semantic features), CAN 버스 데이터를 통해 얻은 운전 특성(driving features), 그리고 LLMs를 통해 생성된 맥락적 특성(contextual features). 특히, OneFormer 알고리즘을 활용하여 의미적 특징을 정교하게 분석합니다.

- **Performance Highlights**: Ablation 연구를 통해 의미적, 운전, 맥락적 특성을 결합하는 것이 사고 예측에서 최선의 결과를 낸다는 사실이 확인되었습니다. 또한, LLMs가 생성한 복잡성 인덱스 주석이 Amazon Mechanical Turk의 결과보다 더 우수한 예측 성능을 보이며, 자동화된 도구를 활용한 정확하고 확장 가능한 사고 예측 시스템 개발의 가능성을 보여줍니다. 이러한 연구 결과는 자율주행 기술의 발전과 교통 안전 향상에 크게 기여할 것으로 기대됩니다.



### ReC-TTT: Contrastive Feature Reconstruction for Test-Time Training (https://arxiv.org/abs/2411.17869)
- **What's New**: 이 논문은 Test-Time Training(TTT) 기법을 개선하기 위해 ReC-TTT라는 새로운 접근법을 제안합니다. ReC-TTT는 동적 데이터 분포 개선을 위해 contrastive representation learning의 원리를 활용하여, 미리 학습된 frozen encoder를 사용해 입력 이미지의 특징을 식별합니다. 이 방식은 기존 TTT 방법과 비교해 도메인 변화에 더 잘 적응할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ReC-TTT는 frozen encoder와 두 개의 trainable encoders 간의 cross-reconstruction을 보조 작업으로 사용합니다. 이는 테스트 시 frozen 상태의 decoder가 도메인 특화된 특징을 추출하는 데 기여하는 방식입니다. 실험 결과는 ReC-TTT가 다양한 도메인 변동 분류 과제에서 기존 최첨단 기술들보다 우수한 성과를 내므로, 딥러닝 모델의 일반화 능력을 향상시키는 데 기여할 것으로 기대됩니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, ReC-TTT는 여러 TTA 및 TTT 방법들과 비교하여 탁월한 성능을 발휘합니다. 이 방법은 기존의 기법들보다 발생 가능한 도메인 변화에 더욱 효과적으로 대처하는 방안을 제시합니다. 또한, 다양한 분포 변화 유형에 대해 우수한 성공 사례들을 기록하여 모델의 넓은 적용 분야에서도 효과적임을 보여줍니다.



### Generative Image Layer Decomposition with Visual Effects (https://arxiv.org/abs/2411.17864)
Comments:
          The project page: this https URL

- **What's New**: 최근 대규모 생성 모델, 특히 diffusion 기반의 방법은 이미지 편집 기능을 획기적으로 개선했습니다. 하지만 사용자 주도의 이미지 구성 작업에서 정밀한 제어를 달성하는 것은 여전히 도전과제입니다. 본 논문에서는 이미지 구성 요소를 독립적으로 편집할 수 있도록 하는 layer-based representation의 필요성을 언급하며, 이를 통해 사용자 맞춤형 콘텐츠 생성의 기회를 제시합니다.

- **Technical Details**: 본 연구에서 제안하는 LayerDecomp는 이미지 레이어 분해를 위한 생성 프레임워크로, 포토리얼리스틱한 배경과 고품질의 투명 전경을 출력합니다. 데이터셋 준비 파이프라인을 통해 시뮬레이션된 다층 데이터를 자동으로 확장하여 훈련을 촉진하고, 일관성 손실(consistency loss)을 도입하여 투명 전경 레이어의 정확한 표현을 학습할 수 있도록 설계했습니다. 또한, 현실 세계의 자연 시각 효과를 포함하는 이미지로 시뮬레이트된 데이터셋을 보강하여 실제 적용성을 높였습니다.

- **Performance Highlights**: LayerDecomp는 레이어 분해 품질에서 기존 방법보다 우수한 성능을 보이며, 객체 제거 및 공간 편집 작업에서 다양한 벤치마크 및 사용자 연구에서 뛰어난 결과를 달성했습니다. 이 방법은 추가 모델 훈련 없이 레이어 기반 이미지 편집을 지원하며, 사용자가 점진적으로 원하는 결과를 도출할 수 있는 창의적 가능성을 열어줍니다. 덕분에, 이 연구는 사용자 맞춤형 이미지 편집의 품질 향상에 크게 기여할 것입니다.



### OracleSage: Towards Unified Visual-Linguistic Understanding of Oracle Bone Scripts through Cross-Modal Knowledge Fusion (https://arxiv.org/abs/2411.17837)
- **What's New**: 이번 연구에서는 OracleSage라는 다중 모달 다중 언어 모델 프레임워크를 소개하며, 복잡한 피카소 성격을 가진 오라클 뼈 문자를 해석하기 위한 혁신적인 접근법을 제시합니다. 이 프레임워크는 계층적 시각-의미 이해(Hierarchical Visual-Semantic Understanding) 모듈과 그래프 기반 의미 추론(Graph-based Semantic Reasoning) 프레임워크를 통합하여 시각적 패턴 인식과 언어 분석을 결합하는 최초의 시도를 보여줍니다. OracleSem이라는 포괄적인 의미 주석이 포함된 OBS 데이터세트를 개발하여 향후 연구의 기초를 마련합니다.

- **Technical Details**: OracleSage는 LLaVA의 시각적 백본을 점진적으로 세밀하게 조정하여 저수준부터 고수준까지 다중 세분화된 특성 추출을 가능하게 하는 계층적 모듈을 포함하고 있습니다. 또한, 이 연구는 고대 문자 해석을 위해 서로 다른 그래프 네트워크를 활용하여 글자 구성 요소, 구조적 배열 및 의미 개념 간의 관계를 명시적으로 모델링합니다. OracleSem 데이터세트는 각각의 문자에 대해 원래의 픽토그래픽 의미와 구조적 분해, 현대 중국 문자에 연결된 의미 관계 등을 문서화한 풍부한 의미 주석을 제공합니다.

- **Performance Highlights**: 실험 결과 OracleSage는 현재 최고 수준의 비전-언어 모델을 크게 초월하는 성능을 보였습니다. 이 연구는 고대 문자의 해석을 위한 새로운 패러다임을 정립하며, 고고학 연구에 유용한 기술적인 지원을 제공합니다. 최근 인공지능 기술의 발전을 통해 오라클 뼈 문자의 종합적인 해석을 지원할 수 있는 새로운 가능성이 열렸습니다.



### SVGDreamer++: Advancing Editability and Diversity in Text-Guided SVG Generation (https://arxiv.org/abs/2411.17832)
Comments:
          17 pages, 17 figures. arXiv admin note: substantial text overlap with arXiv:2312.16476

- **What's New**: 이번 연구는 텍스트 가이드를 기반으로 한 새로운 벡터 그래픽 합성 방법을 제안하여 SVG의 시각적 품질과 다양성을 향상시키고자 합니다. 이를 위해 Vectorized Particle-based Score Distillation (VPSD) 기법을 도입하여 기존 방법의 과포화 문제를 해결하고 출력 SVG의 다양성을 증대시킵니다. 또한, 적응형 벡터 프리미티브 제어 전략을 설계하여 그래픽 세부 묘사의 프리미티브 수를 동적으로 조정할 수 있도록 합니다.

- **Technical Details**: Scalable Vector Graphics (SVG)은 선형 기하학적 원소인 Bézier 곡선, 다각형 및 선을 사용하여 시각적 개념을 표현합니다. 최근 텍스트에서 SVG로의 변환을 위한 모델들이 급속히 발전하고 있으며, CLIP 모델과 DiffVG를 결합하여 SVG 생성을 위한 다양한 방법들이 제안되고 있습니다. 그러나 기존 Text-to-SVG 방법은 생성된 이미지의 편집 가능성 부족, 시각적 품질 저하, 제한된 다양성 등의 문제를 내포하고 있습니다.

- **Performance Highlights**: SVGDreamer++는 벡터 그래픽 생성에서 편집 가능성, 시각적 품질 및 다양성 측면에서 기존 방법보다 우월한 성능을 보입니다. 본 연구에서는 최대 여섯 가지 서로 다른 벡터 스타일을 지원하며, 다양한 벡터 디자인에도 적용 가능한 고품질의 벡터 자산을 생성할 수 있음을 실험을 통해 입증하였습니다. 또한, 이 접근법은 아이콘 생성 및 포스터 디자인을 포함한 벡터 디자인 영역에서의 응용 가능성을 보여줍니다.



### CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos (https://arxiv.org/abs/2411.17820)
- **What's New**: 이 논문에서는 매우 복잡한 도심 환경에서의 내비게이션 문제를 해결하기 위한 새로운 데이터 기반 접근 방식을 제안합니다. 특히, 대규모의 인터넷 동영상 데이터를 활용하여 인간과 유사한 내비게이션 기술을 훈련하는 방법을 보여주고 있습니다. 기존의 내비게이션 방법들이 고정된 맵에 의존하는 것에 비해, 이 방법은 지도 없이도 자율 주행 시스템에 적용할 수 있는 가능성을 제시합니다. 이를 통해 실제 도심에서의 배달 로봇과 같은 이동 로봇의 활용이 촉진될 것입니다.

- **Technical Details**: 제안된 CityWalker 프레임워크는 거리에서 수천 시간의 도보 및 운전 영상을 활용하여 대규모 모방 학습(imitation learning)을 가능하게 만드는 간단하면서도 확장 가능한 데이터 처리 파이프라인을 소개합니다. 이 접근 방식은 비디오에서 노이즈가 섞인 가짜 라벨(pseudo labels)을 이용하여 대규모 데이터셋에 대한 학습을 지원합니다. 이는 빠르고 효율적인 내비게이션 정책을 학습할 수 있도록 돕는데, 훈련된 모델은 복잡한 도시 환경의 다양한 장애물과 상황에 효과적으로 대응할 수 있습니다.

- **Performance Highlights**: 대규모의 다양한 데이터셋에서 훈련된 결과, 제안된 모델은 실제 실험에서 내비게이션 성능이 크게 향상되는 것으로 나타났습니다. 이 연구는 동적인 도시 환경에서의 내비게이션 정책을 개발하기 위해 인터넷에 존재하는 풍부한 비디오 데이터를 활용하는 가능성을 보여줍니다. 제안된 방법은 인간과 유사한 내비게이션 작업을 실행 가능하게 하며, 이는 도로교통 신호 준수 및 사교적 거리 유지와 같은 사회적 규칙을 반영하게 됩니다.



### Low-rank Adaptation-based All-Weather Removal for Autonomous Navigation (https://arxiv.org/abs/2411.17814)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 악천후에서의 이미지 복원(All-Weather Image Restoration, AWIR)을 위한 사전 훈련된 모델의 효율적인 적응 방안을 제시합니다. 특히 Low-Rank Adaptation(LoRA)을 사용하여 새로운 기후 복원 작업에 대한 사전 훈련된 모델을 효과적으로 조정합니다. 또한 전이 학습에 대한 LoRA의 한계를 보완하기 위해 LoRA-Align(LoRA-A)이라는 새로운 방법론을 도입합니다.

- **Technical Details**: LoRA는 파라미터 효율적인 미세 조정을 가능하게 하는 방법으로, 고정된 가중치 행렬에 저차원 가중치 변경을 추가하여 작동합니다. 이 논문에서는 LoRA와 LoRA-A를 이용해 이미지를 복원하고 자율 내비게이션에서의 활용성을 분석합니다. LoRA-A는 특이값 분해(Singular Value Decomposition, SVD)를 통해 원래의 작업 성능을 보존하면서 새로운 작업으로 적응할 수 있도록 수정합니다.

- **Performance Highlights**: LoRA와 LoRA-A를 통해 복원된 이미지는 자율 내비게이션의 여러 컴퓨터 비전 작업, 특히 시멘틱 분할(semantic segmentation)과 깊이 추정(depth estimation)에서 성공적으로 활용될 수 있습니다. 이 연구는 AWIR 모델의 효율적인 조정을 통해 실세계의 다양한 기후 조건에서의 자율 주행 가능성을 높이는 데 기여합니다.



### Signs as Tokens: An Autoregressive Multilingual Sign Language Generator (https://arxiv.org/abs/2411.17799)
- **What's New**: 본 연구에서는 Signs as Tokens(SOKE)라는 멀티링구얼(다국어) 수어 생성 모델을 제안합니다. 기존 연구들이 수어 생성(Sign Language Generation, SLG)을 시각적 콘텐츠 생성 작업으로 간주한 반면, 이 모델은 언어 모델(pretrained Language Models, LMs)의 도움을 받아 텍스트 입력으로부터 3D 수어 아바타를 생성합니다. 또한, 이 연구는 다양한 신체 부위를 표현할 수 있는 분리된 토큰을 생성하는 방법을 개발하여 수어의 다중 인지 특성을 효과적으로 캡처합니다.

- **Technical Details**: SOKE는 수어의 연속적 신체 동작을 이산화하여 토큰 시퀀스(token sequences)로 변환하는 디커플드 토크나이저를 통합합니다. 이를 통해 수어의 언어적 구조를 더 효과적으로 모델링할 수 있게 됩니다. 연구팀은 미국 수화(ASL)와 중국 수화(CSL)를 포함한 다국어 수어 데이터셋을 사용하여 모델을 훈련시켰으며, 전문적인 태그를 통해 특정 수어 및 신체 부위를 지정할 수 있습니다.

- **Performance Highlights**: SOKE의 성능은 How2Sign 및 CSL-Daily와 같은 두 개의 도전적인 기준에서 최첨단 결과를 달성한 것으로 나타났습니다. 연구에서 제안한 방법은 수어의 복잡한 손 동작을 강조하는 데 중점을 두어, 기존 방법의 한계를 보완합니다. 이로 인해 수어 생성을 위한 통합된 모델이 성공적으로 만들어졌고, 수어 커뮤니티와 일반 커뮤니티 간의 소통 장벽을 줄이는 데 기여할 것으로 기대됩니다.



### NEMO: Can Multimodal LLMs Identify Attribute-Modified Objects? (https://arxiv.org/abs/2411.17794)
- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs)의 기존 객체 인식 능력을 평가하기 위해 NEMO라는 새로운 벤치마크를 소개합니다. NEMO는 900개의 원래 과일 이미지와 해당 속성이 수정된 이미지, 그리고 2,700개의 다양한 질문 세트를 포함하고 있습니다. 이 연구는 MLLMs가 commonsense에서 beyond-commonsense로 이동할 때 인식 능력을 어떻게 조정하는지를 조사하도록 설계되었습니다.

- **Technical Details**: NEMO 벤치마크는 MLLMs의 commonsense 및 beyond-commonsense 시나리오에서의 추론 능력을 체계적으로 평가합니다. 연구에서는 강력한 vision encoders가 MLLMs의 성능 향상에 기여하지만, MLLMs는 독립적인 vision encoders에 비해 여전히 성능이 낮다는 점을 강조합니다. 또한 모델 크기를 확장하는 것이 항상 더 나은 결과로 이어지지 않으며, 더 큰 LLM들이 vision encoders의 성능을 약화시키는 경우도 있음을 발견했습니다.

- **Performance Highlights**: 가장 최근의 26개 공개 및 상업적 모델을 분석한 결과, NEMO에서 속성이 수정된 객체 인식시 뚜렷한 성능 격차가 나타났습니다. 기존의 비전 인코더 성능보다 낮은 MLLMs의 인식 성능은 이러한 격차를 설명할 수 있습니다. 연구 결과는 MLLMs의 장애 요소를 분석하여 더 견고하고 다재다능한 모델 개발을 위한 통찰력을 제공합니다.



### Self-supervised Monocular Depth and Pose Estimation for Endoscopy with Generative Latent Priors (https://arxiv.org/abs/2411.17790)
- **What's New**: 이 논문은 내시경(Endoscopy) 환경에서의 3D 매핑 성능 향상을 위한 새로운 프레임워크를 제안합니다. 특히, Generative Latent Bank와 Variational Autoencoder (VAE)를 활용한 자가 지도 학습(self-supervised learning) 기반의 깊이(depth) 및 자세(pose) 추정 방법이 주효함을 보여줍니다. 기존의 합성 데이터(synthetic datasets)나 복잡한 모델에 의존하는 방법들이 범용성을 결여하고 있었음을 지적하며, 새로운 방법으로 내시경 환경의 복잡한 질감과 조명 문제를 해결하고자 합니다.

- **Technical Details**: 제안하는 방법은 DepthNet과 PoseNet의 두 가지 주요 브랜치를 포함하며, 각각 깊이와 자세를 추정합니다. Generative Latent Bank를 통해 자연 이미지에서 얻은 깊이 정보를 바탕으로 깊이 예측의 사실성과 안정성을 향상시키고, 자세 추정은 VAE를 사용하여 자세 전환을 잠재 변수(latent variables)로 다룸으로써 최적화합니다. 이 설정은 Z축의 강조를 안정시키고, X-Y 민감도를 개선하여 내시경 촬영 시의 복잡한 텍스처 문제 해결에 기여합니다.

- **Performance Highlights**: SimCol 및 EndoSLAM 데이터셋에서의 평가 결과, 제안된 프레임워크가 기존의 자가 지도 방법들보다 우수한 성능을 보였습니다. ablation 연구를 통해 각 제안된 구성 요소의 효과가 검증되었으며, 이로 인해 내시경 깊이 및 자세 추정에서의 정확도가 높아졌습니다. 이 접근 방식은 임상 환경에서 내시경 진단과 치료의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### Geometric Point Attention Transformer for 3D Shape Reassembly (https://arxiv.org/abs/2411.17788)
- **What's New**: 본 논문에서는 기하학적 관계를 추론하기 위한 네트워크인 Geometric Point Attention Transformer (GPAT)를 제안합니다. 기존 방법들이 부분 간의 기하학적 상호작용을 정확하게 캡처하지 못했던 문제를 해결하고자 합니다. GPAT는 각 부분의 자세를 회전 및 변환 벡터로 나타내어, 전역 형상 정보와 지역 쌍별 기하학적 특징을 통합합니다. 이를 통해 모델의 성능을 개선하고, 반복적인 예측을 가능하게 하는 기하학적 재활용 모듈도 도입됩니다.

- **Technical Details**: GPAT의 주요 구성 요소는 기하학적 포인트 어텐션 모듈과 기하학적 재활용 모듈로, 이들은 지역 기하학, 6-DoF 예측 및 동적 모델링 문제를 다루기 위해 설계되었습니다. 기하학적 포인트 어텐션 모듈은 각 부분의 회전 및 변환 벡터를 어텐션 점수 계산에 통합함으로써, 정확한 조립을 위한 공간적 관계를 캡처합니다. 결과적으로 반보적인 회전과 변환의 업데이트가 이루어져, 6-DoF 기하학적 특성을 보존한 채로 자세를 직접 예측할 수 있습니다.

- **Performance Highlights**: 우리 모델은 PartNet 데이터셋의 의미적 조립 작업과 Breaking Bad 데이터셋의 기하학적 조립 작업 모두에서 기존 방법들에 비해 우수한 성능을 보여주었습니다. 반복적인 예측 기능은 조립 과정에서의 틀림이 있는 부분을 정교하게 개선하는 데 도움을 줍니다. 또한, GPAT는 미래의 형상 조립 연구 및 6-DoF 예측 작업에 있어 우선 선택되는 백본으로 자리 잡을 가능성이 큽니다.



### Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficien (https://arxiv.org/abs/2411.17787)
Comments:
          Working in progress. Code repository: this https URL

- **What's New**: 이번 논문은 Visual Auto-Regressive (VAR) 모델링의 비효율성을 해결하기 위하여 새로운 협력적 디코딩 전략인 Collaborative Decoding (CoDe)을 제안합니다. CoDe는 대형 모델과 소형 모델 간의 효율적인 협력 프로세스를 통해 메모리 소비를 절감하고 속도를 크게 향상시킵니다. 이 접근 방식은 고주파수 세부정보에 집중하는 소형 모델과 저주파수 내용을 생성하는 대형 모델 간의 명확한 역할 분담을 통해 이루어집니다.

- **Technical Details**: CoDe는 VAR 모델의 긴 토큰 수열에서 발생하는 메모리 과부하를 해소하기 위해 두 개의 VAR 모델을 활용하는 혁신적인 방법입니다. 대형 모델은 낮은 주파수 콘텐츠 생성을 담당하고, 소형 모델은 고주파수 세부정보 예측에 전문화되어 있습니다. 이 구조는 각 모델이 특정 주파수 대역에 맞춰 최적화된 역할을 수행할 수 있도록 체계적으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, CoDe는 원래 VAR-d30 모델에 비해 약 1.7배의 속도 향상과 함께 GPU 메모리 소비를 0.5배로 줄이며, 품질 저하 없이 이미지를 생성할 수 있음을 입증하였습니다. 특히, CoDe는 NVIDIA 4090 GPU에서 256x256 해상도로 41장의 이미지를 초당 생성하는 등 인상적인 2.9배 속도를 달성했습니다. CoDe는 FID 수치를 2 이하로 유지하며 효율적인 이미지 생성의 새로운 기준을 제시합니다.



### DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching (https://arxiv.org/abs/2411.17786)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문에서 제안하는 DreamCache는 개인화된 이미지 생성을 위한 효율적이고 품질 높은 접근 방식으로, 기존 방법의 단점을 극복합니다. 특히, 적은 수의 레이어에서 참조 이미지 특징을 캐싱(caching)하여 생성 과정에서 동적인 조정을 가능하게 합니다. 이는 기존의 복잡한 훈련 요구사항 없이도 고품질의 개인화된 이미지를 생성할 수 있도록 해줍니다.

- **Technical Details**: DreamCache는 사전 훈련된 diffuion denoiser의 단일 타임스텝에서 특징을 캐싱하여 사용자 생성 텍스트 캡션 없이 이미지를 생성합니다. 또한, 적은 수의 추가 파라미터만으로 경량화되어 리소스가 제한된 장치에서도 사용 가능하며, 기존 U-Net의 무게를 변경하지 않고도 개인화와 비개인화 콘텐츠의 동시 생성을 지원합니다. 이러한 방식은 기존 모델에 비해 직접적인 메모리 및 연산 비용을 절감합니다.

- **Performance Highlights**: DreamCache는 기존 방법들에 비해 매우 낮은 연산 및 데이터 비용으로 개인화된 이미지 생성에서 최고 수준의 품질을 달성합니다. 캐싱된 특징을 사용하여 신속하고 효율적인 개인화 샘플링을 구현하며, 이로 인해 고품질 이미지를 실시간으로 생성할 수 있는 가능성이 열립니다. 따라서 DreamCache는 개인화된 이미지 생성의 실용성과 확장성을 크게 개선한 혁신적인 접근법으로 평가됩니다.



### Diffusion Autoencoders for Few-shot Image Generation in Hyperbolic Spac (https://arxiv.org/abs/2411.17784)
- **What's New**: 이번 논문에서는 Hyperbolic Diffusion Autoencoders (HypDAE)라는 새로운 접근법을 제안하여, 이전의 방법들이 이미지 품질과 다양성 간의 균형을 이루는 데 어려움을 겪었던 것을 해결할 수 있습니다. HypDAE는 하이퍼볼릭 공간(hyperbolic space)에서 작동하여, 학습된 이미지와 텍스트 간의 계층적 관계를 포착합니다. 이를 통해, 한정된 예제만으로도 뛰어난 품질의 다양한 이미지를 생성할 수 있는 가능성이 열립니다.

- **Technical Details**: HypDAE는 이미지 인코더(image encoder)를 통해 고수준의 의미를 캡처하고, 사전 훈련된 Stable Diffusion 모델을 활용하여 랜덤한 변형을 모델링합니다. 이 시스템은 유클리드 공간과 하이퍼볼릭 공간 간의 맵핑을 수행하여, 원활한 계층적 이미지 임베딩(hierarchical image embeddings)을 보장합니다. 또한, 텍스트-이미지 특성을 하이퍼볼릭 공간으로 투사하여 텍스트 안내에 의한 이미지 편집을 용이하게 합니다.

- **Performance Highlights**: HypDAE는 기존의 few-shot 이미지 생성 방법들과 비교하여 품질, 다양성, 유연성 모두에서 현저한 성능 향상을 보여줍니다. 실험 결과, HypDAE는 준수한 이미지 품질을 제공하면서, 계층적인 이미지 속성 편집을 가능하게 합니다. 이 혁신적인 접근은 데이터가 적은 상황에서도 특정 카테고리와 무관하게 고유한 특성을 지닌 이미지를 생성할 수 있는 데 큰 기여를 하고 있습니다.



### Beyond Walking: A Large-Scale Image-Text Benchmark for Text-based Person Anomaly Search (https://arxiv.org/abs/2411.17776)
- **What's New**: 이번 논문에서는 텍스트 기반의 인물 검색의 한계를 극복하기 위해 새로운 작업인 텍스트 기반 인물 이상 탐색(Text-based Person Anomaly Search)을 제안합니다. 기존의 검색 시스템은 일반적인 행동(예: 걷기, 서기)에만 집중하여 이상 행동을 식별하는 데 필요한 다양성이 부족했으나, 새로운 Pedestrian Anomaly Behavior (PAB) 벤치마크는 이러한 문제를 해결하고자 설계되었습니다. 이 데이터셋은 일상적인 행동과 함께 여러 가지 이상 행동이 포함된 1,013,605개의 합성 이미지-텍스트 쌍과 1,978개의 실제 이미지-텍스트 쌍으로 구성되어 있습니다.

- **Technical Details**: Pedestrian Anomaly Behavior (PAB) 데이터셋은 일반적인 행동과 이상 행동을 포함하여 다양한 상황을 반영합니다. 연구팀은 인물의 포즈 패턴과 정체성 기반의 하드 네거티브 쌍 샘플링을 통합한 Cross-Modal Pose-aware (CMP) 프레임워크를 제안하여, 정상적인 행동과 이상 행동 간의 구분을 향상시킵니다. 이 프레임워크는 인간의 포즈 정보를 활용하여 보행자 활동에 대한 이해를 높이고, 보다 정교한 행동 검색 성능을 제공합니다.

- **Performance Highlights**: PAB 벤치마크에서의 광범위한 실험 결과는 합성 훈련 데이터가 실제 테스트 세트에서의 행동 검색 성능을 크게 향상시키는 것을 보여줍니다. 또한 제안한 포즈 기반 방법은 recall@1을 2.88% 향상시키는 데 기여하며, 이는 이상 행동을 식별하는 데 있어 효과적임을 강조합니다. 논문은 데이터셋, 코드 및 체크포인트를 공개하여 향후 연구를 촉진하고 결과의 재현 가능성을 확보할 계획입니다.



### Efficient Multi-modal Large Language Models via Visual Token Grouping (https://arxiv.org/abs/2411.17773)
- **What's New**: 이번 논문에서는 Multi-modal Large Language Models (MLLMs)의 새로운 토큰 그룹화 메커니즘인 VisToG를 제안합니다. 기존 방법들은 시각적 토큰의 감소를 특징 정렬(feature alignment) 단계에서 수행하는 반면, VisToG는 사전 학습된 시각 인코더를 활용하여 유사한 이미지 세그먼트를 그룹화합니다. 이러한 접근법은 이미지 토큰 전송 시 중복을 줄이고, 인퍼런스(inference) 비용을 효과적으로 감소시킵니다.

- **Technical Details**: VisToG는 이미지 세멘틱을 나타내기 위해 임베딩을 결합(concatenate)하는 방식을 사용합니다. 이 과정은 시각 인코더에 입력되기 전에 이루어지며, 사전 학습된 지식을 사용하여 중복 시각 토큰을 제거합니다. 결과적으로, VisToG는 평균 27%의 인퍼런스 시간 단축과 함께 98.1%의 원래 성능을 유지하는 것으로 입증되었습니다.

- **Performance Highlights**: 실험 결과, VisToG는 고해상도 이미지 처리 시 발휘하는 성능과 비용 효율성 모두에서 두각을 나타냈습니다. 기존의 방법들이 인퍼런스를 느리게 만드는 토큰 수의 증가로 어려움을 겪는 반면, VisToG는 이러한 문제를 해결하며, 시각 데이터의 구조 및 중복성을 활용하여 계산 효율성을 높입니다. 이를 통해, MLLM의 실용적인 배포 가능성을 한층 더 높였습니다.



### MVBoost: Boost 3D Reconstruction with Multi-View Refinemen (https://arxiv.org/abs/2411.17772)
- **What's New**: 본 연구에서는 단일 뷰 이미지를 사용하여 3D 재구성을 강화하는 새로운 프레임워크인 다중 뷰 정제(multi-view refinement) 기법, MVBoost를 제안합니다. 이 방법론은 고정밀 다중 뷰 생성 모델과 3D 재구성 모델의 장점을 결합하여 신뢰할 수 있는 데이터 소스를 만들어냅니다. 이를 통해 고품질 3D 데이터를 생성하고, 사용자가 제공한 입력 이미지를 기반으로 최적의 뷰포인트를 개별화하는 과정을 포함시켜, 재구성의 품질을 향상시키는 기능을 갖추고 있습니다.

- **Technical Details**: MVBoost는 단일 뷰 입력 이미지를 바탕으로 다중 뷰 이미지를 생성하기 위해 다중 뷰 확산 모델을 사용합니다. 이 이미지들은 대규모 3D 재구성 모델로 전송되어 일관된 3D 표현으로 변환됩니다. 그 후, 해당 3D 데이터에서 렌더링된 다중 뷰 이미지를 정제하여 대규모 다중 뷰 데이터셋을 생성하고, 이를 피드포워드 3D 재구성 모델 훈련에 사용합니다. 최적의 재구성을 위해 3D 모델과 입력 이미지의 정렬을 최적화하는 과정을 추가하여 사용자 요구에 맞는 재구성을 돕습니다.

- **Performance Highlights**: GSO 데이터셋을 통한 평가 결과, MVBoost 방법이 기존 재구성 방법보다 우수한 성능을 보였습니다. 정성적 및 정량적 결과 모두에서 고충실도의 3D 재구성을 달성하며 최신 기술 상태를 기록하였습니다. 이 연구는 다양한 단일 뷰 데이터 세트를 통합하여 3D 재구성을 훈련하는 프레임워크를 마련함으로써, 재구성 결과의 향상을 꾀할 수 있음을 입증하였습니다.



### DiagramQG: A Dataset for Generating Concept-Focused Questions from Diagrams (https://arxiv.org/abs/2411.17771)
- **What's New**: 본 논문에서는 교육 자료에 사용되는 도표를 이용한 Visual Question Generation (VQG) 연구의 공백을 해소하기 위해 DiagramQG라는 데이터셋을 소개합니다. 이 데이터셋은 8,372개의 도표와 19,475개의 질문으로 구성되어 있으며, 다양한 과목을 포함합니다. 새로운 개념 및 목표 텍스트 제약을 도입하여 교육의 목적에 맞는 질문을 생성할 수 있도록 모델을 유도합니다. 또한, HKI-DQG라는 프레임워크를 제안하여 도표 질문 생성을 위한 강력한 기준선 모델을 제공합니다.

- **Technical Details**: DiagramQG는 4개의 과목, 15개의 과정 및 169개의 개념을 포괄하는 질문의 포괄적인 컬렉션으로, 19,475개의 고유 질문과 8,372개의 도표를 포함합니다. HKI-DQG는 CLIP을 사용하여 도표의 다양한 크기에 따라 관련 패치를 식별하며, BLIP 및 Qwen2-VL과 같은 고급 비전-언어 모델이 지식을 추출합니다. 이 프레임워크는 비전-언어 모델을 훈련할 필요 없이 기존 지식과 텍스트 제약을 결합하여 질문을 생성합니다.

- **Performance Highlights**: 제안된 HKI-DQG는 기존의 VQG 모델 및 다양한 비전-언어 모델에 비해 우수한 성능을 보여주며, DiagramQG 데이터셋에서 강력한 기준선을 제시합니다. 또한, HKI-DQG는 VQG-COCO 및 K-VQG와 같은 자연 이미지 기반의 두 가지 다른 데이터셋에서도 적용 가능하여 최신 기술 수준을 달성했습니다. 이 연구는 교육적인 요구를 충족하며, 도표 기반의 질문 생성을 통한 학생의 개념 이해 평가를 가능하게 합니다.



### Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis (https://arxiv.org/abs/2411.17769)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 확산 기반 합성에서 세부 조정을 위한 단일 매개변수인 $\,\omega\,$를 소개합니다. 이 매개변수는 확산 모델의 역과정에서 노이즈 제거 단계에 포함되어, 검증된 방식으로 출력의 세부 수준을 정밀하게 조절할 수 있습니다. 특별히, 기존의 모델 재학습이나 구조 수정이 필요하지 않으며, 추가 처리 비용 없이도 효율적인 세부 조정이 가능합니다.

- **Technical Details**: 제안된 방법은 네트워크 아키텍처와 시간 단계 스케줄링을 수정하지 않고, 예측된 노이즈의 분산을 동적으로 조정하여 세부 조정을 가능하게 합니다. 노이즈 스케일링은 단일 매개변수 $\,\omega\,$로 달성되며, 이를 통해 이미지의 다양한 공간적 요구에 따라 맞춤형 효과를 구현할 수 있습니다. 예를 들어, 특정 지역에 세부적인 그레인 조정을 위한 오메가 마스크를 사용하여 사용자 지정 효과를 생성할 수 있습니다.

- **Performance Highlights**: Omegance 기법은 다양한 이미지 및 비디오 합성 작업에서 뛰어난 성능을 보여주었습니다. 우리는 Stable Diffusion, SDEdit, ControlNet 등 다양한 모델에 대해 실험을 진행하였으며, 모든 작업에서 정밀하고 원활한 세부 조정이 가능함을 확인했습니다. 이로 인해 자원 소모 없이도 고품질의 출력과 사용자 맞춤형 콘텐츠 생성이 가능해졌습니다.



### Exploring Aleatoric Uncertainty in Object Detection via Vision Foundation Models (https://arxiv.org/abs/2411.17767)
- **What's New**: 이 논문은 객체 탐지 데이터의 불확실성을 효율적으로 모델링하고 활용하기 위한 새로운 접근 방식을 제안합니다. 특히, 객체 탐지에서는 다양한 크기의 객체와 가림, 흐림, 노이즈가 섞인 주석 문제로 인해 aleatoric uncertainty가 자연스럽게 발생합니다. 이를 해결하기 위해, 대규모 데이터셋에서 훈련된 비전 파운데이션 모델의 피처 공간을 기반으로 각 객체 인스턴스의 데이터 불확실성을 추정하는 방법을 제안합니다.

- **Technical Details**: 객체의 피처 분포를 모델링하기 위해 가우시안 혼합 구조를 가정하고, Mahalanobis distance 기반의 측정을 통해 aleatoric uncertainty를 정량화합니다. 또한, 제안된 불확실성 측정치는 모델 훈련 시 노이즈와 중복 인스턴스를 제외하는 필터와 난이도에 따른 샘플을 균형 있게 처리하기 위한 샘플 적응 정규화에 두 가지 실용적인 용도로 사용될 수 있습니다.

- **Performance Highlights**: 광범위한 실험적 연구 결과, MS-COCO 및 BDD100K 벤치마크에서 다양한 최첨단 탐지 모델을 사용하여 제안된 aleatoric uncertainty 측정이 효과적임을 검증했습니다. 불확실성을 활용하여 노이즈 샘플을 버리고 샘플 적응 정규화를 도입한 결과, 평균 정밀도(AP)와 재현율(AR) 측면에서 예측 성능이 크게 향상되었습니다. 또한, 제안된 불확실성 기반 필터링 전략이 기존의 균등 샘플링보다 우수한 결과를 도출했습니다.



### I2VControl: Disentangled and Unified Video Motion Synthesis Contro (https://arxiv.org/abs/2411.17765)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문은 I2VControl이라는 새로운 비디오 제어 프레임워크를 제안합니다. 이 프레임워크는 비디오 합성 분야에서 다양한 제어 신호를 통합 및 관리하여 다중 모션 제어 작업을 하나의 통합된 시스템으로 통합합니다. I2VControl은 기존의 단일 모션 패턴에 의존하는 방법의 한계를 극복하고, 각 비디오 모션 유닛을 개별 제어 신호로 나타내어 세부적으로 제어할 수 있는 가능성을 제공합니다.

- **Technical Details**: I2VControl에서는 비디오를 세 가지 유형의 모션 유닛(보더랜드, 드래그 유닛, 브러시 유닛)으로 구분합니다. 이 모델은 각 유닛이 독립적으로 제어될 수 있도록 트레일 제어 함수를 도입하여 목적에 맞는 세밀한 조정을 가능하게 합니다. 또한, 플러그인 아키텍처를 통해 다양한 사용자 상호작용을 지원하며, 통합된 데이터 파이프라인을 개발하여 훈련 과정을 효율적으로 관리합니다.

- **Performance Highlights**: 실험을 통해 I2VControl은 다양한 제어 시나리오에서 뛰어난 유연성과 효율성을 달성하였으며, 사용자 창의성을 자극하여 비디오 제작에서의 혁신성을 높였습니다. 제안된 프레임워크는 비디오 합성 과정에서의 직관적이고 정밀한 제어를 가능하게 해, 사용자 경험을 크게 향상시킵니다. 이러한 요소들은 비디오 제작의 창의적 요구에 맞춰 조정될 수 있도록 돕습니다.



### Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation (https://arxiv.org/abs/2411.17763)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 3D 반사 대칭(reflection symmetry)의 탐지를 단일 RGB 이미지에서 수행하는 Reflect3D를 소개하고 있습니다. 기존 방법들이 3D 데이터에 의존했던 반면, 본 연구는 체계적인 symmetry 탐지를 위해 transformer 기반 모델을 활용합니다. 이 접근 방식은 실세계 다양한 시나리오에 대한 강력한 일반화 능력을 보여줍니다.

- **Technical Details**: Reflect3D는 다중 뷰(diffusion model)를 사용하여 물체의 주변 뷰를 합성하고, 이를 통해 명확한 대칭 탐지를 수행합니다. 첫째, DINOv2 encoder를 사용해 기하학적 이미지 특징을 추출하고, 둘째, transformer 기반의 대칭 디코더가 다양한 대칭 가설을 사용합니다. 마지막으로, 모든 뷰의 예측을 클러스터링하여 정확하고 포괄적인 대칭 예측을 얻습니다.

- **Performance Highlights**: Reflect3D는 GSO와 OmniObject3D 등 두 개의 도전적인 데이터셋에서 뛰어난 성능을 보여주며, 실세계 이미지를 활용한 높은 일반화 능력을 증명했습니다. 또한, 대칭 추정이 단일 이미지 3D 생성 과정에서 구조적 정확도 및 시각적 충실도를 크게 향상시킨다는 점을 강조했습니다.



### MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding (https://arxiv.org/abs/2411.17762)
- **What's New**: MUSE-VL은 멀티모달 이해 및 생성을 위한 통합 비전-언어 모델로, Semantic Discrete Encoding(SDE)을 채택하여 고유한 시맨틱(semantic) 제약을 통해 시각(token) 및 언어(token) 정보를 효과적으로 정렬합니다. 이는 훈련 복잡성을 감소시키고 모델의 성능 향상에 기여합니다. MUSE-VL은 기존의 최첨단 모델을 초월하는 결과를 보이며, 여러 비전-언어 기준에서 더욱 뛰어난 성능을 나타냅니다.

- **Technical Details**: 본 연구는 인식 및 생성을 위한 단순하고 통합된 자가 회귀(torch) 변환기 모델을 확립하는 것을 목표로 합니다. 업데이트된 시각 토크나이저, Semantic Discrete Encoding(SDE)을 통해 이미지를 효과적으로 정량화하고 이를 언어 모델과 잘 정렬된 비주얼 토큰으로 변환합니다. MUSE-VL은 다양한 멀티모달 데이터 세트에 대해 사전 학습되며, 이를 통해 이미지 추론(image reasoning), 비주얼 질문 응답(visual question answering), 이미지 생성(image generation)과 같은 복잡한 멀티모달 작업에서 강력한 성능을 발휘합니다.

- **Performance Highlights**: MUSE-VL은 다양한 비전-언어 벤치마크에서 최첨단 성능을 달성하였으며, 이전의 모델을 능가하는 결과를 보였습니다. 특히, 이미지 추론 및 질문 응답에서 탁월한 성능을 발휘하며, 통합된 생성을 통해 멀티모달 작업을 효과적으로 처리할 수 있습니다. MUSE-VL은 또한 구조 변경 없이 사전 학습된 대형 언어 모델에 쉽게 적용할 수 있어 멀티모달 이해 및 생성 작업의 공동 훈련을 용이하게 만듭니다.



### OpenAD: Open-World Autonomous Driving Benchmark for 3D Object Detection (https://arxiv.org/abs/2411.17761)
- **What's New**: OpenAD는 3D 객체 탐지를 위한 최초의 오픈월드 자율주행 벤치마크로, 다양한 센서 및 시나리오에서의 도메인 일반화(domain generalization)와 오픈 어휘(open vocabulary)를 지원합니다. 이 논문은 수천 개의 코너 케이스 객체를 주석 처리하고 2000개의 시나리오를 통합한 새로운 평가 방법론을 제시합니다. 또한, 일반 모델과 특화 모델을 융합하는 앙상블 방법을 통해 기존의 저정확 문제를 해결하고자 합니다.

- **Technical Details**: OpenAD는 멀티모달 대형 언어 모델(MLLM)과 통합된 주석 처리 파이프라인을 통해 자동으로 코너 케이스 객체를 주석 처리합니다. 여기에서 도메인 전환 평가를 위한 일관된 벤치마크 형식을 정렬하고, 2D 및 3D 오픈월드 모델의 성능을 비교합니다. 또한, 3D 오픈월드 객체 탐지를 위한 비전 중심의 기초 방법을 제안합니다.

- **Performance Highlights**: 이 연구에서 제안된 OpenAD 벤치마크는 객체 탐지 모델의 도메인 일반화 및 오픈 어휘 능력을 동시에 평가합니다. 기존 모델들이 특정 프리디파인드 카테고리에 국한되었던 문제를 해결하기 위해, 새로운 시각 중심의 3D 객체 탐지 방법이 도입되었습니다. OpenAD는 실제 세계에서의 광범위한 2D 및 3D 바운딩 박스 주석을 제공하여 보다 신뢰성 있는 평가를 가능하게 합니다.



### UVCG: Leveraging Temporal Consistency for Universal Video Protection (https://arxiv.org/abs/2411.17746)
- **What's New**: AI 기반 비디오 편집의 보안 위험이 커지고 있는 가운데, 본 연구는 Universal Video Consistency Guard (UVCG)라는 혁신적인 방법을 제안합니다. UVCG는 비디오의 시간적 일관성을 활용하여 멀티미디어 콘텐츠를 보호하며, 연속적인 왜곡을 도입하여 비디오 편집 모델의 인코더가 잘못된 출력으로 매핑하도록 유도합니다. 이를 통해 비디오 내용의 무단 수정에 대한 효과적인 방어 수단을 제공합니다.

- **Technical Details**: UVCG는 연속적인 비디오 프레임 간의 정보의 일관성을 고려하여, 특정 타겟 비디오의 내용을 보호 비디오에 삽입하는 방법입니다. 이 과정에서는 perturbation-reuse 전략을 통해 계산 효율성을 높이며, 최소한의 GPU 리소스만으로도 효과적인 보호를 달성합니다. 또한, projected gradient descent (PGD)를 사용하여 최적화 문제를 해결합니다.

- **Performance Highlights**: UVCG는 다양한 Latent Diffusion Models (LDM) 버전에서 테스트되었으며, 비디오 편집 파이프라인에서의 일반화 가능성을 평가했습니다. 실험 결과, UVCG는 편집된 비디오의 왜곡을 유의미하게 증가시켰고, 낮은 계산 자원 소모로 87%의 보호 성공률을 기록했습니다. 이러한 결과는 비디오 콘텐츠 무단 수정 방지에서 UVCG의 효과성과 범용성을 입증합니다.



### SnapMem: Snapshot-based 3D Scene Memory for Embodied Exploration and Reasoning (https://arxiv.org/abs/2411.17735)
- **What's New**: SnapMem은 주체적인 에이전트를 위한 혁신적인 스냅샷 기반 장면 표현으로, 'Memory Snapshots'와 'Frontier Snapshots'를 도입하여 탐색 및 추론을 강화합니다. 이 시스템은 3D 환경에서의 시각적 정보와 미지의 영역에 대한 통찰력을 결합하여 에이전트가 정보를 보다 효과적으로 탐색할 수 있게 합니다. 또한 지속적인 탐색과 기억 관리를 지원하는 새로운 메모리 집합 방식인 Prefiltering을 통해 에이전트는 비상식적인 계산 부담을 줄일 수 있습니다.

- **Technical Details**: SnapMem의 구성 요소인 Memory Snapshot은 탐색된 지역의 물체와 주변 환경을 포괄합니다. Frontier Snapshot은 탐색되지 않은 지역과 그 방향을 나타내어 에이전트가 새로운 정보를 수집하는 데 도움을 줍니다. 이 두 가지 스냅샷은 RGB-D 이미지 스트림을 통해 생성되며, 에이전트는 목표를 기반으로 최적의 Frontier Snapshot을 선택하여 탐색을 진행합니다. 이를 통해 스냅샷 집합의 동적 업데이트와 효율적인 기억 검색이 가능합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험 결과, SnapMem은 에이전트의 탐색 및 추론 능력을 현저히 향상시키며, 복잡한 3D 환경 내에서의 지속적 학습 가능성을 보여줍니다. SnapMem을 활용한 에이전트는 장기적으로 변화하는 환경에 적응하고 새로운 정보를 효과적으로 획득할 수 있습니다. 이로 인해 SnapMem은 주체적인 AI 애플리케이션의 발전에 있어 중요한 기여를 할 것으로 기대됩니다.



### Cross-modal Information Flow in Multimodal Large Language Models (https://arxiv.org/abs/2411.18620)
- **What's New**: 이 연구는 auto-regressive multimodal large language models (MLLMs) 내에서 언어와 시각 정보의 상호작용을 탐색하는 새로운 접근 방식을 제공합니다. 연구진은 이러한 모델에서 정보를 어디서, 어떻게 결합하여 최종 예측을 생성하는지 분석하고자 합니다. 이를 위해 시각 질문 응답(visual question answering) 작업을 중심으로 여러 모델에서 실험을 진행하였습니다.

- **Technical Details**: 연구진은 MLLMs의 서로 다른 층을 통해 정보 흐름을 추적하여, 시각적 정보를 어떻게 통합하는지 조사합니다. 주요 방법은 attention knockout 방식으로 특정 주의 패턴을 차단하여, 시각과 언어 입력 간의 상호작용을 억제하는 것입니다. 이를 통해 일반적인 시각 정보와 구체적인 물체 정보가 문맥에 따라 통합되는 두 가지 단계를 확인하였습니다.

- **Performance Highlights**: 실험 결과, MLLMs에서는 아래층에서 더 일반적인 시각 정보가 질문 토큰의 표현과 결합되고, 중간층에서는 특정 물체와 관련된 정보를 질문에 맞춰 통합하는 과정이 있습니다. 마지막으로, 이 통합된 다중 모달 표현이 최종 예측을 위한 입력 순서의 마지막 위치에 전달되어 정확한 응답을 생성하는 것을 발견했습니다. 이 연구는 MLLMs의 투명성을 향상시키고, 향후 다중 모달 정보 통합에 대한 연구에 기여할 것으로 기대됩니다.



### Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspectiv (https://arxiv.org/abs/2411.18615)
- **What's New**: 본 논문에서는 다중 작업 학습(Multi-Task Learning, MTL)에서 발생하는 gradient conflict(그래디언트 충돌)를 줄이기 위한 새로운 접근법인 Sparse Training(ST)을 제안합니다. 이는 원래 모델의 일부 파라미터만 업데이트하며 나머지는 동결하여 여러 작업을 동시에 학습하도록 하는 방식입니다. 기존의 gradient manipulation(그래디언트 조작) 방법과 결합하기 용이하다는 장점도 가지고 있습니다.

- **Technical Details**: Sparse Training(ST)은 각 작업이 특정 파라미터 집합에만 영향을 미치도록 하여 그래디언트 간의 간섭을 줄이는 새로운 방법론을 제시합니다. 이 모델은 모든 작업의 평균 손실을 최적화하는 기존의 일반적인 방법보다 효과적이며, 특히 훈련의 후반부에서 그래디언트 충돌 발생이 감소하는 경향을 보입니다. ST의 적용은 대규모 미리 훈련된 모델에서도 효과적임을 보여주며, 여러 그래디언트 조작 기법과 연계하여 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, Sparse Training은 여러 작업이 동시에 수행될 때 그래디언트 충돌을 효과적으로 완화시켜 성능을 향상시키는 것으로 나타났습니다. 대규모 모델에서의 그래디언트 충돌 문제는 더 심각하게 나타나며, ST가 이를 완화하는 데 효과적이라는 점이 강조됩니다. ST의 도입으로 다양한 데이터셋 및 아키텍처에서 그래디언트 간의 충돌 발생 빈도가 감소함을 확인할 수 있습니다.



### Evaluating and Improving the Effectiveness of Synthetic Chest X-Rays for Medical Image Analysis (https://arxiv.org/abs/2411.18602)
- **What's New**: 이 연구는 인공지능(AI) 도구를 이용하여 합성 흉부 X선 이미지를 생성하고, 의료 이미징 데이터셋을 보강하여 분류(classification)와 세분화(segmentation) 작업에서의 딥 러닝 모델 성능을 최적화하는 방법을 탐구합니다. 저자들은 레이턴트 확산 모델(latent diffusion model)을 활용하여 텍스트 프롬프트(text prompts) 및 세분화 마스크(segmentation masks)에 따라 합성 이미지를 생성합니다. 또한, 합성 데이터의 품질을 향상시키기 위해 전문의 피드백과 프록시 모델(proxy model)을 적용했습니다.

- **Technical Details**: 이 연구는 ControlNet 아키텍처를 기반으로 한 레이턴트 확산 모델을 사용하여, 텍스트 프롬프트와 세분화 마스크에 기반하여 합성 흉부 X선을 생성합니다. 이 프레임워크는 기존 데이터셋(CheXpert, CANDID-PTX 등)의 실제 이미지에 합성 이미지를 결합하여, F1 점수와 Dice 점수를 평가하여 성능 향상을 측정했습니다. 통계적 유의성을 평가하기 위해 일측 t-검정(one-tailed t-test)과 분산 보정(Bonferroni correction)을 사용했습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터가 기존 실제 데이터보다 평균 F1 점수를 0.150453 향상시키는 등 분류 및 세분화 작업에서 두드러진 성능 향상을 보였습니다. F1 점수는 데이터셋 크기가 증가할수록 더욱 개선되었고, 세분화 작업에서 Dice 점수의 평균 개선은 0.14575에 달했습니다. 연구 결과, 합성 흉부 X선 이미지를 생성하는 최상의 방법으로는 단일 질병 레이블이나 기하학적으로 변형된 세분화 마스크에 기반한 조건화가 확인되었습니다.



### Biomolecular Analysis of Soil Samples and Rock Imagery for Tracing Evidence of Life Using a Mobile Robo (https://arxiv.org/abs/2411.18594)
Comments:
          Key Words : Mars, Rover, Phoenix, Biosignatures, Biomolecular Analysis, Microscopy, Spectroscopy, Sampling, Astrobiology

- **What's New**: 이번 연구는 Mars에서의 과거 생명체의 증거를 찾기 위한 Phoenix 로버(phoenix rover)의 기능을 확장시키기 위한 개조를 다룹니다. 현재 사용되고 있는 디지털 현미경 이미저(digital microscopic imagers)와 분광계(spectrometers)는 해상도(resolution) 및 탐지 범위(detection range)의 제한으로 인해 어려움을 겪고 있습니다. 이 논문은 이러한 문제를 해결하기 위해 Phoenix 로버의 성능을 향상시킨 개조 작업에 대해 설명합니다.

- **Technical Details**: 연구에서는 Phoenix 로버에 첨단 디지털 현미경 이미저 및 분광계를 통합하여 토양 샘플을 고해상도로 검사할 수 있는 기능을 추가했습니다. 더불어 장치의 기계적 구성 요소는 조작성과 샘플링 능력을 최적화하기 위해 강화되었습니다. 이러한 기술적 개선은 다양한 지질 환경에서 샘플을 취득하고 생체 분자 분석(biomolecular analysis)을 할 수 있는 능력을 제공합니다.

- **Performance Highlights**: Phoenix 로버는 다양한 지질 환경 내에서 원활하게 탐색하고 샘플을 취득할 수 있는 능력을 입증하였습니다. 연구에서 제시된 생체 분자 기기(biomolecular instrumentation) 및 혼합 분석 기법(hybrid analytical methods)은 향후 Mars의 생명체 탐사 임무에 대해 상당한 잠재력을 보여줍니다. 이 시스템의 개선 가능성은 탐지 가능한 바이오마커(biomarkers) 및 생체 지문(biosignatures)의 범위를 넓힐 수 있는 가능성에 있습니다.



### DexDiffuser: Interaction-aware Diffusion Planning for Adaptive Dexterous Manipulation (https://arxiv.org/abs/2411.18562)
Comments:
          27 pages. Project page: this https URL

- **What's New**: 이번 연구에서는 Dexterous manipulation(정밀 조작)을 위한 상호작용 인식 확산 계획 프레임워크인 DexDiffuser를 소개합니다. 기존의 확산 기반 계획 방식이 보다 단순한 조작 작업에서 가능성을 보였으나, 복잡한 상호작용을 처리할 때 비현실적인 유령 상태(ghost states)를 생성하거나 적응성이 부족한 문제를 해결하고자 하였습니다.

- **Technical Details**: DexDiffuser는 사전 상호작용 접촉 정렬(pre-interaction contact alignment)과 접촉 후 목표 지향 제어(post-contact goal-directed control)로 구성된 이중 단계 확산 프로세스를 통해 상태-행동 다이나믹스(joint state-action dynamics)를 모델링합니다. 또한 다이나믹스 모델 기반의 이중 안내(dynamics model-based dual guidance)를 통합하고, 자동화된 안내 기능 생성을 위해 대형 언어 모델(large language models)을 활용하여 물리적 상호작용에 대한 일반성 향상을 도모합니다.

- **Performance Highlights**: 실험 결과, DexDiffuser는 훈련 분포(outside training distributions) 밖의 목표에 대해서도 평균 59.2%의 성공률을 기록하며 기존 방법들에 비해 두 배 이상의 개선된 결과를 보여주었습니다. 30도 도어 열기에서 70.0%, 펜과 블록 절반 방향 전환에서 각각 40.0%, 36.7%의 성공률을 기록하며, 망치로 못을 절반만 박는 작업에서 46.7%의 성공률을 보였습니다. 이는 접촉이 풍부한 조작에서의 강건성과 유연성을 강조합니다.



### A comparison of extended object tracking with multi-modal sensors in indoor environmen (https://arxiv.org/abs/2411.18476)
- **What's New**: 이 논문은 LiDAR와 스테레오 카메라라는 두 가지 3D 포인트 클라우드 센서를 비교하여 효율적인 객체 추적 접근 방식을 연구한 초기 결과를 제시합니다. 특히, 가격 차이가 큰 두 센서를 사용하여 단일 객체 추적을 중심으로 한 연구를 진행하였으며, 스테레오 카메라를 이용한 객체 추적 성능이 LiDAR와 유사하게 나타났습니다. 이 결과는 낮은 비용으로도 효과적인 객체 추적이 가능함을 보여줍니다.

- **Technical Details**: 이 연구에서는 Density-Based Spatial Clustering of Applications with Noise (DBSCAN) 알고리즘을 사용하여 환경과 대상에 대한 사전 정보를 활용한 효율적인 객체 탐지 방법을 개발하였습니다. Extended Object Tracking (EOT) 프레임워크를 통해 측정값을 바탕으로 대상의 위치와 운동학, 그리고 공간의 확장을 추정합니다. 이 과정에서 star-convex hypersurface 모델을 활용하여 타겟의 형태를 매개변수화합니다.

- **Performance Highlights**: 실험 결과, 스테레오 카메라를 기반으로 한 객체 추적 방법은 LiDAR를 사용한 경우와 비슷한 성능을 달성하였고, 비용 차이는 10배 이상이었습니다. 이 연구는 스테레오 카메라의 저렴한 가격에도 불구하고 효과적인 객체 추적 솔루션을 제공할 수 있음을 입증하였습니다. 또한, 이전에는 스테레오 카메라 포인트 클라우드를 활용한 EOT에 대한 연구가 없었음을 강조하고 있습니다.



### Learning the Evolution of Physical Structure of Galaxies via Diffusion Models (https://arxiv.org/abs/2411.18440)
- **What's New**: 이 논문은 Denoising Diffusion Probabilistic Models (DDPM)을 활용하여 적색 변위(redshift)에 기초한 새로운 접근법을 제안합니다. 이를 통해 은하 이미지 생성을 위한 조건화 모델을 개발하여 과거 은하 형성과 우주의 진화에 대한 통찰력을 제공합니다. 연구 결과, 이 모델은 시각적으로 사실적인 은하 이미지를 생성할 뿐만 아니라 적색 변위에 따른 물리적 특성 변화를 인코딩합니다.

- **Technical Details**: DDPM은 데이터에 점진적으로 노이즈를 추가한 후 이를 역으로 학습하여 새로운 샘플을 생성하는 방식으로 작동합니다. 본 논문에서는 적색 변위를 U-Net 아키텍처의 시간 단계에 통합하여 조건부 분포를 학습합니다. 훈련 중 적색 변위에 Gaussian 노이즈를 추가하고, Huber Loss 및 AdamW 최적화를 사용하여 모델의 일반화 능력을 향상시키는 방법을 구현했습니다.

- **Performance Highlights**: 모델은 286,401개의 은하 이미지를 사용하여 훈련되었으며, 테스트 세트에서 생성된 은하 이미지는 실제 은하 물리적 속성과 비교하여 평가됩니다. Fréchet Inception Distance (FID) 및 Inception Score (IS)와 같은 지표를 활용하여 생성된 이미지의 시각적 유사성을 측정했으며, 이는 은하의 형태, 크기 및 밝기 분포와 같은 주요 특성을 반영한 평가로 이어졌습니다.



### Federated Learning with Uncertainty and Personalization via Efficient Second-order Optimization (https://arxiv.org/abs/2411.18385)
- **What's New**: 본 논문에서는 퍼지된 학습(Federated Learning, FL)을 위한 새로운 베이esian FL 방법인 FedIvon을 제안합니다. 이는 Bayesian 접근법의 이점, 즉 향상된 성능과 예측 불확실성의 정량화 측면에서 최소한의 계산 비용을 추가하여 균형을 맞춥니다. 이전 방법들과 달리 저비용의 2차 최적화 방식을 통해 각 클라이언트에서 지역 후속 분포를 근사하는 효율적인 방법을 채택합니다.

- **Technical Details**: FedIvon은 Improved Variational Online Newton (IVON) 알고리즘을 활용하여 클라이언트에서 매우 효율적인 변분 추론(Variational Inference, VI)을 수행합니다. 이 방법은 로컬 후속 분포를 대각 공분산을 갖는 가우시안으로 근사하여 자연 경량을 이용해 손실 함수의 기하학을 포착합니다. 이로써 나머지 기존 방법들에서 요구되는 헤시안 계산 없이도 더 저렴한 계산 비용으로 결과를 도출할 수 있습니다.

- **Performance Highlights**: FedIvon은 SOTA 베이esian 및 비베이esian FL 방법들에 비해 예측 정확도 및 불확실성 추정에서 개선된 결과를 보여줍니다. 또한 클라이언트 수준의 모델 개인화도 자연스럽게 지원하여 클라이언트가 서버의 후속 결과를 우선으로 활용하도록 합니다. 이러한 균형 덕분에 로컬 적응력과 글로벌 지식 공유를 효과적으로 조화시킬 수 있습니다.



### G3Flow: Generative 3D Semantic Flow for Pose-aware and Generalizable Object Manipulation (https://arxiv.org/abs/2411.18369)
Comments:
          Webpage: this https URL

- **What's New**: G3Flow는 실시간 동적 객체 중심 3D 의미 표현을 구축하는 새로운 프레임워크입니다. 이 프레임워크는 기초 모델들을 활용하여 기하학적 정밀성과 의미적 이해를 통합하여 로봇의 작업 성능을 크게 향상시킵니다. 특히, G3Flow는 수동 주석 없이도 일관된 의미적 이해를 유지할 수 있는 방법을 제시합니다.

- **Technical Details**: G3Flow는 3D 생성 모델, 비전 기초 모델, 강력한 자세 추적을 결합하여 동적 의미 흐름을 생성합니다. 이 과정은 두 단계로 나뉘며, 첫 번째 단계에서는 로봇이 다중 관측 데이터를 수집하여 디지털 트윈을 생성하고 기본 의미 필드를 형성합니다. 두 번째 단계에서는 실시간 자세 추적을 통해 의미 필드를 동적으로 변형하여 조작 과정에서의 일관성을 유지합니다.

- **Performance Highlights**: G3Flow는 5가지 시뮬레이션 작업에 대한 광범위한 실험 결과에서 기존의 접근 방법을 초과하여 성능을 입증하였습니다. 최종 제약이 있는 조작 작업에서 최대 68.3%, 객체 간 일반화 작업에서 50.1%의 평균 성공률을 달성하였습니다. 이러한 결과는 G3Flow가 로봇 조작 정책에서 실시간 동적 의미 기능 이해를 향상시키는 데 효과적임을 보여줍니다.



### Leveraging Semantic Asymmetry for Precise Gross Tumor Volume Segmentation of Nasopharyngeal Carcinoma in Planning C (https://arxiv.org/abs/2411.18290)
- **What's New**: 이 연구에서는 비조영 CT 이미지에서 비인두암(nasopharyngeal carcinoma, NPC) Gross Tumor Volume (GTV)을 직접 분할하는 새로운 접근 방식을 제안합니다. 기존의 MRI를 사용한 방법과 다르게, MRI에서 유도된 종양 마스크를 계획 CT에 맞추는 등록 오류를 피할 수 있는 방법입니다. 이는 비조영 CT에서의 낮은 대비 문제를 해결하기 위해 3D Semantic Asymmetry Tumor segmentation (SATs) 방법을 도입한 것입니다.

- **Technical Details**: 이 연구는 비인두 영역에서의 종양의 대칭성을 기반으로 한 3D 세맨틱 비대칭(segmentation) 방법을 제안합니다. 건강한 비인두 영역은 일반적으로 양측 대칭성을 지니지만, NPC 종양이 발생하면 이 대칭성이 파괴됩니다. 이 방법은 Siamese contrastive learning 프레임워크를 통해 비종양과 종양이 있는 영역 간의 보폭 거리를 최소화하며, GTV 특성을 더욱 민감하게 만듭니다.

- **Performance Highlights**: SATs 방법은 내부 및 외부 테스트에서 기존의 최첨단 방법들과 비교하여 최소 2%의 Dice 점수 향상과 12%의 평균 거리 오류 감소를 달성했습니다. 이 연구는 비조영 CT 스캔에서 비인두암 GTV 분할에 대한 새로운 기준을 제시하며, 비대칭 영역 선택 방법을 통해 비대칭 종양 특성을 효과적으로 학습하는 방법을 개발했습니다.



### Don't Let Your Robot be Harmful: Responsible Robotic Manipulation (https://arxiv.org/abs/2411.18289)
- **What's New**: 이번 논문에서는 안전한 로봇 조작을 위한 'Safety-as-policy'라는 새로운 접근 방식을 제안합니다. 이는 로봇이 복잡한 작업을 수행하면서 안전 위험을 고려하도록 하는 책임 있는 조작 방법론을 포함합니다. 'SafeBox'라는 합성 데이터셋도 새로 만들어져, 다양한 안전 위험 시나리오를 가지고 있는 100개의 로봇 조작 작업이 포함되어 있습니다.

- **Technical Details**: Safety-as-policy는 두 가지 주요 요소로 구성됩니다. 첫째, 안전 위험을 포함하는 시나리오를 자동으로 생성하고 가상 상호작용을 수행하는 세계 모델(world model)입니다. 둘째, 반성과 추론을 통해 안전 인지를 개발하는 멘탈 모델(mental model)입니다. 이를 통해 로봇은 위험을 피하면서 작업을 성공적으로 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, Safety-as-policy는 합성 데이터셋과 실제 실험 모두에서 위험을 피하며 작업을 효율적으로 수행할 수 있음을 보여줍니다. 이는 기존의 벤치마크 방법들에 비해 상당한 성능 향상을 이루었으며, SafeBox 데이터셋은 실제 시나리오에 대한 일관된 평가 결과를 나타내어 앞으로의 연구를 위한 안전하고 효과적인 기준을 제공합니다.



### Deep End-to-end Adaptive k-Space Sampling, Reconstruction, and Registration for Dynamic MRI (https://arxiv.org/abs/2411.18249)
Comments:
          39 pages, 19 figures, 4 tables

- **What's New**: 이 논문에서는 동적 MRI (Magnetic Resonance Imaging)에서의 비선형 샘플링, 재구성, 등록을 통합한 최초의 엔드 투 엔드 딥러닝 프레임워크를 소개합니다. 이 프레임워크는 데이터 수집 및 이미지 품질 향상을 위한 적응형 다이내믹 k-space 샘플링 접근 방식을 활용하며, 이후 추정된 변형 필드를 통해 동적 이미지를 정적 기준 이미지와 정렬합니다. 각 구성 요소는 플러그 앤 플레이 구조로 독립적으로 사용 가능하며, 모든 구성 요소의 성능을 향상시키기 위해 조합된 감독 및 비감독 손실 함수로 통합 훈련됩니다.

- **Technical Details**: 프레임워크는 세 가지 주요 모듈로 구성됩니다: 1) DL 기반 적응형 샘플링 전략, 2) DL 기반 재구성 모듈, 3) 등록 모듈입니다. 이러한 모듈은 각각 동적 k-space 획득을 최적화하고, 변형 필드 추정을 위한 이미지를 생성하며, 재구성된 동적 이미지를 정적 기준에 맞게 정렬합니다. 논문에서는 각 모듈의 복잡성 및 기술적인 세부 사항을 명확하게 설명하여 재현성을 보장합니다.

- **Performance Highlights**: 각 구성 요소의 검증을 위해 제어된 실험 및 절단 연구가 수행되었습니다. 제안된 프레임워크는 심장 시네 데이터 세트에서 평가되었으며, 일반화 가능성을 위해 아오르타 데이터 세트로 테스트됨으로써 신뢰성이 입증되었습니다. 동적 MRI의 정확한 모션 추정을 위한 이 종합적인 접근 방식은 전통적인 방법을 능가하는 성능을 제공하고, 다양한 임상 응용에 적용 가능함을 보입니다.



### Make-It-Animatable: An Efficient Framework for Authoring Animation-Ready 3D Characters (https://arxiv.org/abs/2411.18197)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 Make-It-Animatable이라는 새로운 데이터 기반 방법론을 제시하며, 3D 인간형 모델을 한 번의 클릭으로 애니메이션을 위한 준비가 가능하도록 합니다. 주요 특징은 형태와 포즈에 상관없이 약 1초 만에 고품질 애니메이션 모델을 생성할 수 있다는 점입니다.Particle-based shape autoencoder를 포함하여 다양한 3D 표현을 지원하며, 고유한 구조를 가진 캐릭터들도 정확하고 강력하게 처리할 수 있습니다.

- **Technical Details**: 제안된 방법론은 본질적으로 단일 프레임워크로 블렌드 웨이트(blend weights), 본(bones), 포즈 변환(pose transformations)을 생성하며, coarse-to-fine representation 방식을 채택하여 정밀한 리깅과 스키닝을 가능하게 합니다. 결과적으로 이 시스템은 비표준 스켈레톤 구조를 가진 캐릭터에서도 효과적으로 작동합니다. 비교적 복잡한 메시나 3D Gaussian splat 입력을 처리할 수 있는 유연성을 자랑합니다.

- **Performance Highlights**: 실험 결과, 기존 방법들에 비해 품질과 속도 모두에서 상당한 개선을 보였습니다. 이 방법은 Mixamo와 같은 상용 자동 리깅 도구들과 달리, 기본 스켈레톤 구조에 구애받지 않으며, 관절의 움직임에 대한 제어를 더 많이 제공합니다. 게임 및 가상 현실과 같은 응답 속도가 빠르고 높은 사용자 커스터마이징이 요구되는 분야에서 특히 유용할 것입니다.



### Towards Lensless Image Deblurring with Prior-Embedded Implicit Neural Representations in the Low-Data Regim (https://arxiv.org/abs/2411.18189)
- **What's New**: 이 논문에서는 훈련되지 않은 신경망(untained neural networks)을 활용하여 렌즈 없는 이미지 복원(lensless image reconstruction) 문제를 다루고 있습니다. 특히, 기존의 기술들이 훈련 데이터를 많이 요구하는 것과 달리, 임플리시트 신경 표현(implicit neural representations)을 사용하여 사전 훈련 없이 복원을 수행하고 있습니다. 이를 통해 고데이터(high-data)와 저데이터(low-data) 간의 격차를 효과적으로 메우는 방법론을 제시하였습니다.

- **Technical Details**: 렌즈 없는 이미징 시스템은 전통적인 렌즈 없이 이미지 복원을 수행하는 시스템이며, 주로 복잡한 포인트 스프레드 함수(point spread function, PSF)를 통해 고해상도 이미지를 복원하는 데 집중합니다. 본 연구에서는 비선형 최적화(untrained iterative optimization)를 통해 데이터 피델리티(data fidelity)와 정칙화(regularization)를 결합하여 이미지를 더욱 효과적으로 복원하는 방식을 탐구하고 있습니다. 이런 방법은 렌즈 없는 이미지 복원의 복잡성을 해소하기 위해 정교하게 설계된 PSFs를 활용합니다.

- **Performance Highlights**: 본 논문에서 제시된 접근법은 기존의 이미지 복원 기술에 비해 상당한 성능 향상을 보여주고 있습니다. 다양한 저사전(低事前) 및 비훈련 방법(untrained methods)과 비교 분석을 통해 우리의 방식이 우수함을 강조하였습니다. 특히, 고해상도의 복원이 가능해짐으로써 의료 이미징(medical imaging) 및 원격 탐사(remote sensing)와 같은 분야에서의 응용 가능성을 높였습니다.



### Online Knowledge Integration for 3D Semantic Mapping: A Survey (https://arxiv.org/abs/2411.18147)
Comments:
          Submitted to Robotics and Autonomous Systems

- **What's New**: 이 논문은 최근 심층학습의 발전이 지식 그래프(knowledge graphs)와 언어 개념(language concepts)의 통합을 가능하게 하여 로봇의 감지 데이터 처리와 의미적 맵핑(semantic mapping) 파이프라인에 어떻게 기여하는지를 조명합니다. 특히, 의미 장면 그래프(semantic scene graphs)와 언어 모델(language models)의 도입이 현대의 의미적 맵핑 접근법을 어떻게 혁신적인 애플리케이션으로 발전시키는지를 살펴봅니다. 이는 로봇이 환경과 상호작용하는 방식에 큰 변화를 가져오고 있으며, 이에 대한 종합적인 리뷰를 제공합니다.

- **Technical Details**: 논문에서는 의미적 맵핑이 기하학적 맵핑(geometric mapping), 센서 데이터로부터의 의미 정보 획득(acquisition of semantic information), 그리고 기존 지식의 통합(integration of prior knowledge)으로 구성되어 있음을 설명합니다. 각 부분 문제는 독립적으로 중요하며, 유용한 의미적 맵을 위해서는 모든 요소에 대한 이해가 필요합니다. 이 과정에서 기존의 기법과 새로운 방법을 결합하여 로봇이 더 효과적으로 환경을 이해하고 상호작용할 수 있도록 돕는 혁신적인 접근법이 소개됩니다.

- **Performance Highlights**: 논문은 SLAM 기술을 포함하여, 로봇이 환경을 실시간으로 이해하고 상호작용할 수 있는 최신 기술들을 다룹니다. 다양한 방법론이 성능 개선에 기여하고 있으며, 특히 RGB-D 카메라와 같은 최신 장비의 사용이 3D 맵 생성에 있어 실시간 처리를 가능하게 하고 있습니다. 이로 인해, 로봇은 공간 정보뿐만 아니라 의미적 정보도 효과적으로 통합하여 작업을 수행할 수 있습니다.



### Towards Cross-device and Training-free Robotic Grasping in 3D Open World (https://arxiv.org/abs/2411.18133)
- **What's New**: 이 논문은 로봇 집게기술이 복잡한 시나리오에서도 새로운 객체를 무훈련으로 처리할 수 있는 혁신적인 파이프라인을 제안합니다. 기존의 연구들은 종종 특정 환경에 한정되어 있어 일반화를 저해하였으나, 본 연구는 다른 3D point cloud segmentation 모델을 유연하게 사용할 수 있도록 하여 개방형 세계 시나리오에서 유용성을 높였습니다. 이는 다양한 로봇 및 카메라 환경에서 강력한 성능을 발휘해 나갑니다.

- **Technical Details**: 우리의 방법은 CNN의 결과에 전적으로 의존하지 않으며, 이는 다양한 장비에서의 우수한 일반화 능력을 확보하게 해줍니다. 구체적으로, 3D point cloud 장면에서 단순한 전경 및 배경 분할을 위해 CNN을 사용하고, 그 후 정확한 객체 분할을 위해 학습이 필요 없는 클러스터링 방법을 적용합니다. 이 과정에서 임의의 깊이 데이터와 함께 포인트 밀도를 기반으로 한 이진 클러스터링 알고리즘을 통해 인접한 객체도 정확히 분리할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 개방형 세계 시나리오 및 다양한 하드웨어 장치에서 모두 우수한 성능을 보였습니다. 특히, 보이지 않는 객체와 복잡한 환경에서의 집게기술에서 탁월한 강건성을 보여주며, 여러 조명 조건과 중첩된 상황에서도 신뢰성을 유지합니다. 이러한 결과는 우리의 파이프라인이 실세계의 복잡한 문제를 해결하는 데 있어 매우 효과적임을 나타냅니다.



### Large Scale Evaluation of Deep Learning-based Explainable Solar Flare Forecasting Models with Attribution-based Proximity Analysis (https://arxiv.org/abs/2411.18070)
Comments:
          This is a preprint accepted at IEEE International Conference on Big Data 2024( IEEE BigData 2024) Conference

- **What's New**: 이번 연구에서는 태양 플레어 예측을 위한 새로운 근접 기반 프레임워크를 제안하며, 이는 심층 학습 모델의 해석 가능성을 평가하는 데 중점을 둡니다. 기존의 연구들은 정확성에 초점을 맞추었으나, 우리는 모델의 해석 가능성과 신뢰성을 동시에 고려합니다. 본 프레임워크는 예측 결과와 관련된 지역을 이해하기 위해 Guided Grad-CAM 방법을 사용하여 생성된 어트리뷰션 맵을 분석합니다.

- **Technical Details**: 연구에서 사용된 두 개의 CNN 기반 모델은 전체 태양 디스크의 직선 시선(LoS) 자력계 이미지를 학습하여 M급 이상의 태양 플레어를 24시간 이내 예측하도록 설계되었습니다. Guided Grad-CAM(Gradient-weighted Class Activation Mapping) 기법을 사용하여 각 예측의 중요 픽셀을 나타내는 어트리뷰션 맵을 생성하였으며, NOI 확인을 위해 NOAA의 플레어 데이터베이스와 비교하였습니다. 이러한 접근법은 예측 결과의 해석 가능성을 높여줍니다.

- **Performance Highlights**: 모델의 예측 결과는 다양한 Active Region(AR)의 특성과 일정 부분 일치함을 보여주었으며, 이는 모델의 행동에 대한 유의미한 통찰을 제공합니다. 또한, 우리는 근접 점수(Proximity Score)와 동시 위치 비율(Attribution Colocation Ratio)을 도입하여 모델 해석의 질을 정량적으로 평가하였습니다. 이 연구는 태양 플레어 예측의 신뢰성을 높이는 한편, 실제 운영 시스템의 해석 가능성 향상에 기여합니다.



### Mortality Prediction of Pulmonary Embolism Patients with Deep Learning and XGBoos (https://arxiv.org/abs/2411.18063)
Comments:
          Published at IEEE ICECCME 2024, Maldives, 4-6 November 2024

- **What's New**: 본 연구에서는 폐색전증(PE) 환자의 30일 사망률 예측을 위한 새로운 알고리즘인 PEP-Net을 제안합니다. PEP-Net은 환자의 초기 영상 데이터(CT)를 바탕으로 3D Residual Network(3DResNet)와 Extreme Gradient Boosting(XGBoost) 알고리즘을 통합하여 임상적 정확성을 높입니다. 특히, 주목할 점은 주석이 없는 환자 수준의 이진 라벨을 사용하여 클래스 불균형 문제를 해결하고 과적합을 줄이는 방법입니다.

- **Technical Details**: PEP-Net의 아키텍처는 U-Net 모듈을 포함하여 폐와 심장 위치를 자동으로 추정하며, 3DResNet으로부터 계층적 특성을 추출합니다. 주요 성분 분석(Principal Component Analysis, PCA)은 특성 차원 축소를 수행하며, BorderlineSMOTE(B-SMOTE)는 클래스 불균형 문제를 해결하는 데 도움을 줍니다. 최종적으로, XGBoost는 학습된 특징을 활용하여 결과와 가장 예측적인 관계에 초점을 맞춥니다.

- **Performance Highlights**: PEP-Net은 급성 폐색전증으로 진단받은 193개의 CT 스캔 코호트를 분석한 결과, 기존 모델(76-78%)에 비해 뛰어난 성능을 보였습니다. 입력 이미지가 폐 영역(Lung-ROI)일 때 94.5% (+/-0.3), 심장 영역(Cardiac-ROI)일 때 94.0% (+/-0.7)의 정확도를 기록하였습니다. 이는 초기 영상 데이터만으로 PEM 진단 예측의 새로운 기준을 설정하며, 단순한 딥 러닝 모델보다 더 높은 예측 성능을 보여줍니다.



### Neural Finite-State Machines for Surgical Phase Recognition (https://arxiv.org/abs/2411.18018)
- **What's New**: 이 논문에서는 수술 절차 비디오를 분석하기 위한 수술 단계 인식의 중요성을 강조하며, 기존의 Transformer 기반 모델들이 긴 수술 비디오에서의 일관성을 유지하는 데 어려움을 겪는 문제를 지적합니다. 저자들은 고전적인 은닉 마르코프 모델에서 영감을 얻은 Neural Finite-State Machine (NFSM) 모듈을 도입하여 절차적 이해와 딥러닝 접근 방식을 연결합니다. NFSM은 절차 수준의 이해를 신경 네트워크와 결합하여 전반적인 상태 임베딩(global state embeddings)과 주의 기반 다이내믹 전이 테이블(attention-based dynamic transition tables)을 활용하여 성능을 향상시킵니다.

- **Technical Details**: NFSM 모듈은 수술 비디오의 특정 특성을 반영하는 고유한 단계 식별자를 생성하기 위해 학습 가능한 글로벌 임베딩(learnable global embeddings)을 사용합니다. 이는 단계 간 전이를 동적으로 예측하는 전이 테이블을 생성하여 수술이 진행됨에 따라 모델이 단계별 정보에 기반해 적응할 수 있도록 합니다. 또한, 온라인 애플리케이션을 위한 전이 인식 훈련 및 추론 메커니즘을 개발하여 단기 Transformer 인코더와 장기 디코더를 결합하여 효율적인 예측을 지원합니다.

- **Performance Highlights**: Cholec80 데이터셋에서 NFSM을 통합한 결과, 비디오 수준 정확도와 단계 수준 정밀도 및 재현율이 각각 2.3, 3.2, 3.0 및 4.8 포인트 향상되었습니다. NFSM은 Surgformer와 같은 기존의 최첨단 모델과 결합할 수 있는 추가 모듈로 설계되어 성능을 더욱 강화하고, 비수술적 데이터셋에 대한 확장 실험을 통해 검증된 범용성을 갖추고 있습니다. 이를 통해 NFSM은 다양한 비디오 이해 작업에서 넓은 응용 가능성을 갖는다는 점을 입증하였습니다.



### FASIONAD : FAst and Slow FusION Thinking Systems for Human-Like Autonomous Driving with Adaptive Feedback (https://arxiv.org/abs/2411.18013)
- **What's New**: 새로운 연구 FASIONAD는 자율주행 시스템에서 안전하고 효율적인 내비게이션을 달성하기 위해 빠른 사고와 느린 사고의 두 가지 시스템을 통합한 혁신적인 프레임워크입니다. 이 프레임워크는 데이터 기반 경로 계획을 통해 일상적인 내비게이션 작업을 처리하며, 복잡한 상황에서는 고급 추론과 의사결정을 수행합니다. FASIONAD는 nuScenes 데이터셋을 기반으로 한 새로운 벤치마크를 도입하여 시스템의 성능을 평가하고 있습니다.

- **Technical Details**: FASIONAD는 두 가지 경로를 사용하는 아키텍처를 가지고 있습니다. 첫 번째, Fast Pathway는 빠른 응답을 위해 여러 개의 이미지와 높은 수준의 내비게이션 명령으로부터 경로 지점을 생성합니다. 두 번째, Slow Pathway는 복잡한 결정 분석을 위해 단일 유형의 이미지를 처리하여 계획 상태를 생성합니다. 이러한 구조는 동적인 환경에 대한 적응성을 높여주는 중요한 메커니즘입니다.

- **Performance Highlights**: FASIONAD는 nuScenes와 CARLA 같은 벤치마크에서 실험을 통해 기존의 자율주행 방법들보다 내비게이션 성공률과 안전성을 크게 향상시키는 성과를 보였습니다. 이 연구는 빠른 사고와 느린 사고의 원리를 활용하여 깊은 상황 인식과 의도 분석은 물론, 적응형 반응을 가능하게 합니다. FASIONAD는 자율주행 시스템의 인간과 유사한 사고 방식을 구현하여 향후 발전에 기여할 수 있는 큰 가능성을 지니고 있습니다.



### Monocular Obstacle Avoidance Based on Inverse PPO for Fixed-wing UAVs (https://arxiv.org/abs/2411.18009)
- **What's New**: 이 논문은 저속 항공기 경제(Low-altitude Economy, LAE) 및 도시 항공 이동성(Urban Air Mobility, UAM)의 발전으로 인해 고정익 UAV의 장애물 회피 시스템을 제안합니다. 제안된 시스템은 가벼운 딥 강화 학습(deep reinforcement learning, DRL)을 기반으로 하여 시속 30m 이상에서 비행하는 고정익 UAV가 알려지지 않은 장애물을 회피할 수 있도록 합니다. 이 시스템은 고급 센서를 필요로 하지 않고, 단일 프레임 이미지로 깊이 추론 모듈을 사용하여 실시간 장애물 탐지를 수행합니다.

- **Technical Details**: 제안된 방법론은 Proximal Policy Optimization(PPO)을 활용하여 장애물 회피를 위한 최적화된 보상 함수와 비행 경로의 부드러움을 동시에 고려합니다. 특히, 적응형 엔트로피 조정 메커니즘을 도입하여 탐색과 이용의 균형을 맞추며, 훈련 수렴성과 장애물 회피 성공률을 향상시킵니다. 전체 시스템은 소프트웨어 및 하드웨어 실험을 통해 구성되며, 경량화된 네트워크 아키텍처를 통해 엣지 컴퓨팅 장치에서도 구동될 수 있습니다.

- **Performance Highlights**: 소프트웨어와 하드웨어 실험 결과, 제안된 방법이 기존 방법보다 장애물 회피 효율 및 비행 경로의 부드러움에서 우수한 성능을 보임을 확인했습니다. 이 연구는 고속 비행 고정익 UAV에 적합한 경량 탐색 프레임워크를 제시하며, 실시간 장애물 회피 작업에 효과적임을 입증합니다. 이 알고리즘의 소스 코드는 공개되어 있어 연구자들이 필드에서 활용할 수 있습니다.



### HAAT: Hybrid Attention Aggregation Transformer for Image Super-Resolution (https://arxiv.org/abs/2411.18003)
Comments:
          6 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 Hybrid Attention Aggregation Transformer (HAAT) 모델을 제안하여 기존의 Swin-transformer 기반 모델들이 해결하지 못한 문제를 다루고 있습니다. HAAT는 Swin-Dense-Residual-Connected Blocks (SDRCB)와 Hybrid Grid Attention Blocks (HGAB)을 결합하여 특징 정보를 더욱 효과적으로 활용합니다. 이는 정보 처리의 비효율성을 감소시키고, 채널 간 유용한 정보를 놓치지 않도록 개선된 구조를 제공합니다.

- **Technical Details**: HAAT는 Swin Transformer의 변경된 구조를 기반으로 하여, 각 Residual Deep feature extraction Group (RDG) 내에서의 수용 필드를 확장하며, 보다 효율적인 구성으로 성능을 향상시킵니다. HGAB는 채널 어텐션, 희소 어텐션 및 윈도우 어텐션을 결합하여 비지역적 특징 융합을 개선하고, 고급 시각적 결과물을 생성합니다. 또한, 희소 자기 주의 메커니즘을 통해 전역적 특징 상호작용을 증가시키면서도 계산 복잡성을 관리합니다.

- **Performance Highlights**: HAAT는 DF2K 데이터셋을 활용하여 학습되었으며, Set5 및 Set14와 같은 널리 알려진 SISR 벤치마크 데이터셋에서 성능 평가를 실시한 결과, 기존의 첨단 방법들을 초월하는 성능을 나타냈습니다. 이 모델은 더 나은 이미지 복원 결과를 생성하며, 긴 거리 의존성을 잘 처리하여 전반적인 성능을 향상시키는 것을 입증했습니다.



### HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction (https://arxiv.org/abs/2411.17982)
Comments:
          Under review process

- **What's New**: HI-SLAM2는 geometry-aware Gaussian SLAM 시스템으로, RGB 입력만으로 빠르고 정확한 단일 장면 복원을 달성합니다. 기존의 Neural SLAM 및 3DGS 기반 방법들은 렌더링 품질과 기하학적 정확도 간의 절충을 필요로 했으나, 본 연구는 RGB 입력을 통해 두 가지를 동시에 달성할 수 있음을 보여줍니다. 이 방법은 Monocular priors와 학습 기반 dense SLAM을 결합하여 기하학적 추정을 강화하고, 3D Gaussian splatting을 맵 표현의 핵심으로 사용하여 효율적인 장면 모델링을 가능하게 합니다.

- **Technical Details**: 본 시스템은 단일 RGB 입력으로부터 빠르고 정확한 카메라 추적 및 장면 복원을 가능하게 합니다. 네 가지 핵심 구성 요소로는 온라인 트래커, 온라인 루프 클로징 모듈, 연속 매퍼, 오프라인 개선 단계가 포함됩니다. 온라인 카메라 트래커는 학습 기반 dense SLAM 프론트엔드를 활용하여 카메라 포즈와 깊이 맵을 추정하며, 글로벌 일관성과 실시간 성능은 효율적인 Pose Graph Bundle Adjustment를 통해 달성됩니다. 또한, 3D Gaussian Splatting을 이용한 장면 표현 방법은 효율적 온라인 맵 구성이 가능하고 고품질 렌더링을 지원합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, HI-SLAM2는 Replica, ScanNet, ScanNet++ 데이터셋에서 기존 Neural SLAM 방법들과 비교하여 재구성과 렌더링 품질에서 상당한 개선을 보여 주었습니다. 특히 RGB-D 기반 방법보다도 정확도에서 뛰어난 성능을 발휘하며, 실시간 애플리케이션에 특히 적합합니다. 저비용의 RGB 입력만으로 고품질의 장면 복원이 가능하여, 깊이 센서가 비현실적인 환경에서도 신뢰할 수 있는 재구성을 가능하게 합니다.



### Adversarial Training in Low-Label Regimes with Margin-Based Interpolation (https://arxiv.org/abs/2411.17959)
- **What's New**: 본 논문은 반지도 적대적 훈련(semi-supervised adversarial training) 기법을 통해 적대적 공격에 대한 모델의 강건성을 강화하고 자연 정확도를 높이는 새로운 방법을 제안합니다. 특히, 이 방법은 깨끗한 데이터와 적대적 예시 간의 선형 보간(linear interpolation)을 활용하여 변별 경계를 넘어서는 보간 적대적 예시를 생성합니다. 이러한 샘플 인식(sample-aware) 전략은 각 데이터 포인트별로 맞춤형 적대적 예시를 생성하여 모델이 가장 유용한 섭동(purbation) 정보를 학습할 수 있도록 합니다.

- **Technical Details**: 우리는 각 데이터 포인트에 대해 적대적 예시를 생성하며, 이 과정에서 각 데이터 포인트의 특징을 고려하여 적의 세기(perturbation strength)를 조정합니다. 또한, 적대적 예시와의 보간을 통해 결정 경계를 소량의 여유(margin)를 두고 넘어가도록 조절합니다. 이와 함께, 훈련 과정에서 전반적인 섭동 강도를 조절하는 글로벌 엡실론 스케줄링(global epsilon scheduling) 전략을 제안하여 모델이 점진적으로 더 어려운 적대적 예시를 학습하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 PGD(Projected Gradient Descent) 및 AutoAttack과 같은 다양한 적대적 공격에 대해 성능을 효과적으로 향상시키는 것으로 나타났습니다. 제안하는 Semi-Supervised Adversarial Training with Margin-Based Interpolation (SSAT-MBI) 알고리즘은 기존 기준과 비교하여 강건성 및 자연 정확도를 크게 향상시킵니다. 이로써 적은 수의 레이블이 있는 데이터에서도 모델의 발전을 이끌어낼 수 있는 잠재력을 보여주었습니다.



### Stealthy Multi-Task Adversarial Attacks (https://arxiv.org/abs/2411.17936)
- **What's New**: 이번 논문에서는 다중 작업 환경에서 하나의 특정 작업을 선택적으로 타겟팅하고 다른 작업들의 성능을 유지하거나 향상시키는 새로운 접근법인 "Stealthy Multi-Task Attack" (SMTA) 프레임워크를 제안합니다. 이를 통해 공격자는 높은 보안 우선순위를 가진 작업을 공격하더라도 비-critical 작업의 성능에 부정적인 영향을 미치지 않고, 방어 시스템의 활성화를 지연시킬 수 있습니다. 이 연구는 다양한 다중 작업 데이터셋과 공격 알고리즘에서 실험을 통해 SMTA의 유효성을 검증하였습니다.

- **Technical Details**: SMTA는 입력 이미지에 최소한의 노이즈를 추가함으로써 목표 작업의 성능을 저하시킵니다. 이는 손실 함수의 여러 개별 손실의 가중합으로 나타나며, 가중치 요인은 때때로 음수일 수 있습니다. 또한, 제안된 자동화 방법은 다양한 입력에 따라 손실 함수를 조정하여 비-목표 작업의 성능을 보호하면서 목표 작업을 공격할 수 있는 최적의 가중치 요인을 탐색합니다.

- **Performance Highlights**: 실험 결과, 제안된 SMTA 프레임워크는 NYUv2 및 Cityscapes와 같은 여러 다중 작업 데이터셋에서 성공적으로 적용되었으며, 공격 알고리즘인 projected gradient descent (PGD) 및 iterative fast gradient sign method (IFGSM)에서도 효과적입니다. SMTA는 목표 작업을 저하시킴과 동시에 비-목표 작업의 성능을 유지하거나 향상시키는 이전에 없는 표준 기준을 수립하였습니다.



### HOPPR Medical-Grade Platform for Medical Imaging AI (https://arxiv.org/abs/2411.17891)
Comments:
          6 pages, 3 figures

- **What's New**: HOPPR Medical-Grade Platform은 인공지능 분야에서 큰 비전을 가진 언어 모델(large vision language models, LVLMs)의 배포를 가속화하기 위해 혁신적인 접근 방식을 제시합니다. 이 플랫폼은 수백 개의 영상 센터에서 수집된 수백만 개의 이미지 연구와 텍스트 보고서를 기반으로 사전 훈련된 기초 모델(foundation models)을 제공합니다.

- **Technical Details**: HOPPR 플랫폼은 대규모 모델을 개발하는 데 필요한 방대한 컴퓨팅 인프라를 갖추고 있으며, 임상 환경에서 배포를 위해 미세 조정(fine-tuning)된 모델을 평가하는 표준을 마련한 품질 관리 시스템을 제공합니다. 또한 모든 데이터는 비식별화(deidentified)되어 HIPAA 규정 준수를 확보하며 안전하게 저장됩니다.

- **Performance Highlights**: HOPPR는 의료 이미징(medical imaging)에 대한 LVLM 솔루션의 배포를 가속화하여 방사선 의사의 업무 흐름(workflows)을 최적화하고 이 분야에서 증가하는 요구를 충족시키는 것을 목표로 합니다. 개발자는 HOPPR 플랫폼에서 모델을 안전하게 호스팅하고 API를 통해 기존 클리닉 워크플로 내에서 추론을 진행할 수 있는 기능을 제공합니다.



### Breast Tumor Classification Using EfficientNet Deep Learning Mod (https://arxiv.org/abs/2411.17870)
Comments:
          19 pages, 7 figures

- **What's New**: 이번 연구에서는 EfficientNet 모델을 사용하여 유방암 분류의 정확성을 높이고 데이터 불균형 문제를 해결하는 새로운 접근 방식을 소개합니다. 기존 CNN 모델보다 높은 정확도와 낮은 계산 복잡도를 지닌 EfficientNet은 의료 영상 분석에 매우 적합한 구조를 가지고 있습니다. 또한, 집중적인 데이터 증대(data augmentation) 기술과 비용 민감 학습(cost-sensitive learning)을 통해 모델이 다수 클래스에 치우치지 않도록 개선하였습니다.

- **Technical Details**: 연구에서는 BreaKHis 데이터셋을 사용하여 이미지에서 드물게 나타나는 암 유형을 효과적으로 학습할 수 있도록 했습니다. EfficientNet의 구조적 특성에 따라 모델은 깊이(depth), 너비(width), 입력 이미지의 해상도를 조절하여 자동으로 특징을 추출합니다. 이 과정에서 데이터 증대와 함께 비용 민감 학습 방법을 적용하여 결과적으로 다중 클래스 분류에서 성능을 크게 향상시켰습니다.

- **Performance Highlights**: 모델의 이진 분류 성능이 눈에 띄게 개선되어 양성 사례에 대한 recall이 0.92에서 0.95로 증가하였으며, 정확도는 97.35%에서 98.23%로 증가했습니다. 다중 클래스 과제로는 일반적인 데이터 증대를 사용했을 때 91.27%에서 집중적인 데이터 증대를 적용하였을 때 94.54%로 향상되었고, 전이 학습을 통해 95.04%의 정확도를 달성했습니다. 이기종 암종인 Mucinous carcinoma와 Papillary carcinoma의 정밀도를 크게 향상시켰으며, 높은 recall을 유지하고 있음을 혼동 행렬(confusion matrix) 분석을 통해 확인하였습니다.



### Reliability of deep learning models for anatomical landmark detection: The role of inter-rater variability (https://arxiv.org/abs/2411.17850)
Comments:
          Accepted to SPIE Medical Imaging 2025

- **What's New**: 이번 연구는 해부학적 표지점(anatomical landmark) 탐지 분야에서 서로 다른 주석 결합(annotation fusion) 전략이 모델 성능에 미치는 영향을 분석하였습니다. 구체적으로, 서로 다른 주석을 통합하는 세 가지 새로운 전략을 비교하여 주석 간 변동성(inter-rater variability)을 유지하면서 성능과 신뢰성을 향상시키는 방법을 모색했습니다. 이 연구는 해부학적 표지점 탐지에서의 상호 주석 변동성과 딥러닝(d deep learning) 모델의 결과 간의 중요한 연관성을 밝혔습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 ISBI 2015의 Cephalometric X-ray 데이터셋으로, 각 이미지는 성인과 주니어 방사선 의사가 마크한 19개의 표지점을 포함하고 있습니다. 본 연구에서는 100개의 이미지를 선정하고, 이를 통해 5개의 대표 표지점에 대한 총 11개의 주석이 수집되었습니다. 주석 간 변동성을 정량화하기 위해 coordinate variance(CVar)와 Principal Spatial Variability(PSV) 등의 두 가지 지표를 도입하고, 세 가지 주석 결합 전략을 통해 데이터 분석을 수행하였습니다.

- **Performance Highlights**: 연구 결과, 서로 다른 주석 결합 전략이 모델 성능과 신뢰성에 미치는 영향을 정량적으로 분석할 수 있었습니다. 각 전략의 효과를 체계적으로 비교한 결과, Deep Ensembles 방법이 가장 안정적이고 향상된 성능을 보여주었습니다. 또한, 새로운 Weighted Coordinate Variance 지표를 통해 탐지 불확실성 및 주석 변동성을 정량화함으로써, 향후 딥러닝 모델 훈련 데이터 및 평가 기준 설정에 유용한 통찰을 제공할 것으로 기대됩니다.



### CAMLD: Contrast-Agnostic Medical Landmark Detection with Consistency-Based Regularization (https://arxiv.org/abs/2411.17845)
Comments:
          14 pages, 6 figures, 3 tables

- **What's New**: 본 논문에서는 의료 이미지를 통한 해부학적 랜드마크 감지를 위한 자가 감독(Self-Supervised) 심층 학습 프레임워크인 CAMLD를 소개합니다. 이 프레임워크는 단일 참조 예제만을 사용하여 다양한 대조의 샘플에서 해부학적 랜드마크를 감지합니다. 또한, 3D 컨볼루션 기반의 대조 증강 전략과 함께 적응형 혼합 손실 함수를 도입하여 다양한 중재와 무관하게 모델을 일반화할 수 있도록 합니다.

- **Technical Details**: CAMLD는 각 개인의 이미지에서 랜드마크의 일관성을 유지하기 위해 서로 다른 스캔 간의 랜드마크 일관성 손실을 사용합니다. 이를 통해 3D 뇌 랜드마크 감지 동일성을 보장하며, 기존의 감지 기술에 비해 스캔 기반에서의 지역적 정밀도를 촉진합니다. 우리의 방법은 다양한 MRI 필드 강도에서 T1w와 T2w 스캔을 포함한 네 가지 임상 및 공개 데이터셋에 대한 종합적인 실험을 통해 검증되었습니다.

- **Performance Highlights**: 결과적으로 CAMLD는 평균 방사 오류(Mean Radial Errors, MREs)와 성공적인 감지율(Success Detection Rates, SDRs) 측면에서 최첨단(State-of-the-Art) 방법들을 능가했습니다. 이 프레임워크는 광범위한 주석 데이터셋의 필요성을 줄이면서도 다양한 이미징 대조에서 잘 일반화됩니다. 이를 통해 의료 진단 및 수술 계획에 필요한 해부학적 랜드마크 감지의 강력하고 정확한 해결책을 제공합니다.



### Arabic-Nougat: Fine-Tuning Vision Transformers for Arabic OCR and Markdown Extraction (https://arxiv.org/abs/2411.17835)
Comments:
          7 pages, 1 figure

- **What's New**: 아랍어 책 페이지를 구조화된 Markdown 텍스트로 변환하는 OCR 모델인 Arabic-Nougat를 소개합니다. 이 모델들은 Meta의 Nougat 아키텍처를 기반으로 구성되며, arabic-small-nougat, arabic-base-nougat, arabi-large-nougat의 세 가지 전문 모델로 이루어져 있습니다. 아랍어 책 페이지와 해당 Markdown 표현 간의 13.7k 쌍으로 구성된 synthetic dataset인 arabic-img2md를 사용하여 미세 조정되었습니다.

- **Technical Details**: Arabic-Nougat의 핵심 기술 요소 중 하나는 Aranizer-PBE-86k tokenizer로, 이는 효율적인 토큰화를 위해 설계되었습니다. torch.bfloat16의 정밀도와 Flash Attention 2를 활용하여 학습 및 추론을 최적화했습니다. 이 모델들은 다양한 아랍어 텍스트 레이아웃과 긴 문서 처리를 효과적으로 해결하기 위한 아랍어 특화 개선 사항을 포함하고 있습니다.

- **Performance Highlights**: arabic-large-nougat는 최고의 Markdown 구조 정확도와 최저 문자 오류율을 기록하며 최상의 성능을 달성했습니다. 또한, 8,500권 이상의 책에서 추출한 11억 개의 아랍어 토큰을 포함하는 대규모 데이터셋을 제공하여 아랍어 OCR 연구에 유용한 자원을 제공합니다. 모든 모델과 데이터셋 및 코드가 오픈 소스로 제공되어 연구자들이 자유롭게 활용할 수 있습니다.



### Rapid Distributed Fine-tuning of a Segmentation Model Onboard Satellites (https://arxiv.org/abs/2411.17831)
Comments:
          Accepted at the Sixth IEEE International Conference on Image Processing Applications and Systems (IPAS) 2025

- **What's New**: 이번 연구는 Unibap iX10-100 위성 하드웨어에서 MobileSAM이라는 경량화된 사전 훈련(segmentation model)을 활용하여, 수역의 분할 작업을 시연합니다. 데이터 지연 문제를 해결하기 위해, 우리는 위성에서 거의 실시간으로 데이터를 분석할 수 있는 모델을 탑재하여 응답 시간을 단축할 수 있음을 제안합니다. 연구팀은 PASEOS라는 오픈소스 Python 모듈과 MobileSAM을 통합하여 시뮬레이션된 환경에서의 성능 평가를 수행하였습니다.

- **Technical Details**: MobileSAM는 5.78백만 개의 파라미터를 가진 경량 모델로, 전통적인 3채널 RGB 데이터를 바탕으로 하여 훈련되었습니다. 연구에서는 위성 데이터와 대응하는 수역 마스크가 포함된 WorldFloods 데이터셋을 사용하여 모델을 미세 조정합니다. 또한, MobileSAM과 PASEOS의 통합을 통해 훈련 중 전력 및 온도 추세를 관찰하였고, 통신 창의 활용도를 극대화했습니다.

- **Performance Highlights**: 실험 결과, MobileSAM은 최소한의 훈련 데이터로 신속하게 미세 조정될 수 있으며, 빠른 통신이 이루어질 경우 세분화 성능이 향상되는 것을 보여주었습니다. 연구팀은 재난 상황에서의 신속한 응답을 위한 유효한 도구로는 분산형 학습(decentralised learning) 및 사전 훈련 모델의 미세 조정을 강조하고 있습니다. 이 접근법은 극단적인 기상 사건의 빈도와 강도가 증가하는 현재의 상황에서 더욱 중요하게 다뤄지고 있습니다.



### From memorization to generalization: a theoretical framework for diffusion-based generative models (https://arxiv.org/abs/2411.17807)
Comments:
          22 pages

- **What's New**: 본 논문은 확산 기반 생성 모델이 훈련 데이터 세트의 크기가 증가함에 따라 메모리에서 일반화로의 전환을 보여주는 기전을 수학적으로 정의하고 분석합니다. 특히, 생성 분포가 훈련 데이터와 관련된 가우시안 커널 근사를 멀리하는 정도를 기준으로 일반화 레짐(‘generalization’ regime)을 규정했습니다. 우리는 또한 이 전환이 실제 세계의 도메인에서도 발생하는 것을 실증적으로 보여줍니다.

- **Technical Details**: 이 논문에서 제시된 주요 기술적 세부 사항은 생성 분포와 샘플링 분포 간의 Kullback-Leibler 발산의 하한을 설정하고, 훈련 데이터가 등방성 가우시안 분포에서 샘플링될 때 전환이 발생함을 명시합니다. 훈련 데이터 수가 증가함에 따라 생성 분포가 원래의 분포에 더 가까워지는 과정을 표현하기 위해, ETG(Training to Generated distance)와 EOG(Original to Generated distance) 간의 상대적 거리 변화를 다루었습니다.

- **Performance Highlights**: 모델이 메모리에서 비메모리로 전환될 때, 생성 모델의 성능을 보여주는 중요한 결과를 찾아냈습니다. 훈련 데이터의 크기가 증가함에 따라 일반화 오류 EOG가 감소하는 시점에서 메모리에서 비메모리로 전환이 발생합니다. 이 발견은 확산 모델의 일반화 성능을 향상시키기 위한 주요 통찰력을 제공합니다.



### Network Inversion and Its Applications (https://arxiv.org/abs/2411.17777)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.16884, arXiv:2407.18002

- **What's New**: 이번 논문은 neural network의 불투명성 문제를 해결하기 위해 network inversion 기법을 소개합니다. 이 기법은 신경망의 내부 메커니즘을 해석하고 이해하는 데 도움을 주며, 이를 통해 신뢰성과 해석 가능성을 높입니다.

- **Technical Details**: 제안된 방법은 조건부 생성기를 사용하여 훈련된 신경망의 입력 공간에서 데이터 분포를 학습합니다. 이 생성기는 레이블 정보를 벡터 형태로 인코딩하고, 생성된 이미지 간의 코사인 유사성을 최소화하여 다양성을 확보합니다. 또한 Gram matrix의 직교성을 정규화 항으로 도입하여 비슷한 표현이 아닌 독특한 표현을 밀어냅니다.

- **Performance Highlights**: 연구 결과, 이 접근법은 해석 가능성을 높이고, 데이터 재구성 등 여러 응용 분야에 즉시 활용될 수 있습니다. 제안된 방법은 다양한 생성 입력을 생성할 뿐만 아니라, 신경망의 결정 경계를 시각화하는 데 유용합니다.



### Efficient Self-Improvement in Multimodal Large Language Models: A Model-Level Judge-Free Approach (https://arxiv.org/abs/2411.17760)
- **What's New**: 본 논문은 MLLMs(다중 모달 대형 언어 모델)의 신뢰성과 강건성을 향상시키기 위한 자가 개선 방법을 제안합니다. 기존 방법들이 MLLMs 자체를 판단자로 사용하는데 따른 높은 계산 비용과 잠재적 문제를 해결하기 위해, 모델 수준의 판단자 없는 자가 개선 프레임워크를 도입했습니다. 이를 통해, 통제 가능한 망상(hallucination) 메커니즘을 활용하여 데이터 품질을 최적화하고, 가벼운 대조 언어-이미지 인코더를 통해 샘플을 평가하여 자가 개선의 경로를 효율화합니다.

- **Technical Details**: 제안된 방법은 통제 가능한 망상 메커니즘을 사용하여 선호 학습(pair) 쌍을 생성하고, 대조적 언어-이미지 인코더를 통해 데이터 품질을 평가합니다. 초기 데이터셋을 생성한 후, CLIPScore를 계산하여 부정 샘플의 점수가 긍정 샘플보다 높은 쌍을 업데이트합니다. 이후, 최적화된 데이터셋을 사용하여 DPO(direct preference optimization) 기법을 통하여 시드 모델을 학습시킵니다. 이 과정을 통해 최종적으로 자가 개선된 모델을 얻게 됩니다.

- **Performance Highlights**: 본 방법은 대규모 벤치마크 및 새롭게 도입한 IC 데이터셋에서 기존 기술들보다 우수한 성능을 보였습니다. 정밀도와 재현율이 개선되었으며, 계산 요구사항이 현저히 낮아졌습니다. 실험 결과는 시드 모델에 비해 IC 및 Object HalBench 데이터셋에서 значный 향상을 확인했습니다.



New uploads on arXiv(cs.AI)

### Cross-modal Information Flow in Multimodal Large Language Models (https://arxiv.org/abs/2411.18620)
- **What's New**: 이 연구는 auto-regressive multimodal large language models (MLLMs) 내에서 언어와 시각 정보의 상호작용을 탐색하는 새로운 접근 방식을 제공합니다. 연구진은 이러한 모델에서 정보를 어디서, 어떻게 결합하여 최종 예측을 생성하는지 분석하고자 합니다. 이를 위해 시각 질문 응답(visual question answering) 작업을 중심으로 여러 모델에서 실험을 진행하였습니다.

- **Technical Details**: 연구진은 MLLMs의 서로 다른 층을 통해 정보 흐름을 추적하여, 시각적 정보를 어떻게 통합하는지 조사합니다. 주요 방법은 attention knockout 방식으로 특정 주의 패턴을 차단하여, 시각과 언어 입력 간의 상호작용을 억제하는 것입니다. 이를 통해 일반적인 시각 정보와 구체적인 물체 정보가 문맥에 따라 통합되는 두 가지 단계를 확인하였습니다.

- **Performance Highlights**: 실험 결과, MLLMs에서는 아래층에서 더 일반적인 시각 정보가 질문 토큰의 표현과 결합되고, 중간층에서는 특정 물체와 관련된 정보를 질문에 맞춰 통합하는 과정이 있습니다. 마지막으로, 이 통합된 다중 모달 표현이 최종 예측을 위한 입력 순서의 마지막 위치에 전달되어 정확한 응답을 생성하는 것을 발견했습니다. 이 연구는 MLLMs의 투명성을 향상시키고, 향후 다중 모달 정보 통합에 대한 연구에 기여할 것으로 기대됩니다.



### A Pipeline of Neural-Symbolic Integration to Enhance Spatial Reasoning in Large Language Models (https://arxiv.org/abs/2411.18564)
- **What's New**: 이번 논문은 Large Language Models (LLMs)의 공간적 추론(spatial reasoning) 능력을 향상시키기 위한 새로운 신경-상징적(neural-symbolic) 프레임워크를 제시합니다. 저자들은 ASP(Answer Set Programming)를 기반으로 한 기호 추론 및 LLM과 ASP의 파이프라인을 사용하여 40-50%의 정확성 향상을 달성하였습니다. 또한, 이 연구는 LLM의 공간적 추론 능력을 강화하기 위한 통합된 전략을 제안하여, 벤치마크 데이터셋인 StepGame 및 SparQA에서 LLM의 성능을 크게 향상시켰음을 보여줍니다.

- **Technical Details**: 이 연구에서는 세 가지 전략을 통해 공간적 추론을 평가했습니다: (1) ASP 기반 기호 추론, (2) DSPy를 사용한 LLM + ASP 파이프라인, (3) 사실 및 논리적 규칙. 공간적 추론은 정량적(QSR) 및 정성적(quite spatial reasoning) 추론으로 나뉘며, 이 프레임워크는 LLM이 다양한 질문 유형에 대해 성능을 발휘하도록 함으로써 복잡한 공간적 관계를 처리하는 데 도움을 줍니다. 저자들은 이 방법론이 LLM 아키텍처 전반에 걸쳐 성능을 향상시키는 데 효과적임을 주장합니다.

- **Performance Highlights**: StepGame 데이터셋에서 40-50%, SparQA 데이터셋에서 3-13%의 정확성 향상을 달성함으로써 기계 학습(Machine Learning)과 공간적 추론 분야에서의 가능성을 입증하였습니다. 'LLM + ASP' 파이프라인은 Finding Relations (FR) 및 Finding Block (FB) 질문에서 특히 강력한 성과를 보였으며, 이는 특정 작업의 특징과 구현 전략에 따라 성능이 다를 수 있음을 시사합니다. 이 연구는 기호적 추론 모델을 LLM과 접목하여 더 나은 결과를 도출할 수 있다는 점에서 향후 연구에 중요한 기초 자료를 제공합니다.



### NeuroAI for AI Safety (https://arxiv.org/abs/2411.18526)
Comments:
          133 pages, 19 figures

- **What's New**: AI 시스템이 점점 더 강력해짐에 따라 AI 안전의 필요성이 더욱 중요해지고 있습니다. 인간은 일반 지능을 가질 수 있는 유일한 에이전트로서 AI 안전의 매력적인 모델이 될 수 있습니다. 이 연구에서는 뇌의 아키텍처와 학습 알고리즘에 기반하여 AI 안전을 위한 여러 경로를 제시하고 평가하고 있습니다.

- **Technical Details**: 뇌의 표상(Representations), 정보 처리(Information Processing), 아키텍처(Architecture)를 모방하여 AI 시스템의 안전성을 높이는 방법을 제안합니다. 또한, 뇌 데이터를 이용한 강력한 감각(Sensory) 및 운동(Motor) 시스템 구축, 그리고 AI 시스템의 해석 가능성(Interpretability)을 향상시키기 위한 신경과학(Neuroscience) 방법론을 활용하는 것이 중요합니다.

- **Performance Highlights**: 이 연구는 신경과학의 원리를 AI 안전에 긍정적으로 적용하기 위한 여러 구체적인 권고사항을 제시합니다. 뇌 데이터에 맞춘 AI 시스템의 미세 조정(Fine-tuning)과 인지 기반 아키텍처(Cognitively-Inspired Architectures)의 확장이 AI 안전에 기여할 수 있음을 강조합니다.



### Large Language Model-Brained GUI Agents: A Survey (https://arxiv.org/abs/2411.18279)
- **What's New**: 본 논문은 LLM(brained GUI agents) 기반의 GUI 에이전트에 대한 포괄적인 조사와 함께, 그 역사적 발전, 핵심 구성 요소 및 고급 기술을 탐구합니다. 새로운 LLM 기반 GUI 자동화 에이전트은 사용자의 자연어 요청을 해석하고 GUI 요소를 분석하여 자동으로 작업을 수행할 수 있는 능력을 가집니다. 이러한 혁신은 복잡한 디지털 환경 내에서 대화형 명령을 통해 상황에 맞는 대처를 가능하게 합니다.

- **Technical Details**: 이 논문에서 다루는 LLM 기반 GUI 에이전트는 LLM을 인식 및 추론 핵심 엔진으로 활용하여 유연하고 적응력 있게 작업을 생성, 계획 및 실행합니다. 전통적인 GUI 자동화 방법들은 일반적으로 사전 정의된 규칙에 제한되어 있었으나, LLM 기반 에이전트는 자연어 이해(natural language understanding), 시각 인식(visual recognition), 의사 결정(decision-making)을 통합하여 보다 동적인 상호 작용을 가능하게 합니다. 이러한 시스템들은 다양한 소프트웨어 애플리케이션을 제어할 수 있도록 설계되었습니다.

- **Performance Highlights**: 현실세계에서 LLM 기반 GUI 에이전트의 적용 사례로는 Microsoft Power Automate가 있으며, 사용자는 최소한의 기술 지식으로 워크플로우를 디자인할 수 있습니다. 또한, Microsoft Copilot과 같은 생산성 소프트웨어에 통합된 AI 도우미는 자연어 명령을 애플리케이션 작업으로 변환하여 접근성을 향상시킵니다. 이러한 발전은 다양한 응용 분야에서 LLM 기반 GUI 에이전트의 변혁적인 가능성을 강조하고 있습니다.



### Dependency-Aware CAV Task Scheduling via Diffusion-Based Reinforcement Learning (https://arxiv.org/abs/2411.18230)
Comments:
          6 pages, 5 figures

- **What's New**: 이 논문에서는 동적 자율 주행 차량(CAV) 지원을 위한 새로운 의존성 인식(task scheduling strategy)을 제안합니다. CAV의 서로 다른 계산 작업들이 근처의 CAV나 기지국(base station)에게 신속하게 할당되어 작업을 완료합니다. 이를 통해 평균 작업 완료 시간을 최소화하는 데 목적을 두고, 문제를 Markov decision process (MDP)로 재구성하였습니다.

- **Technical Details**: 제안된 알고리즘은 Synthetic DDQN 기반의 Subtasks Scheduling(SDSS)으로, 이를 통해 실시간으로 적응형 작업 스케줄링 결정을 내릴 수 있습니다. 또한, 경험 재생 메모리에서 충분한 합성 데이터를 생성하기 위해 diffusion model 기반의 합성 경험 재생이 통합되어 있습니다. 이는 이러한 알고리즘이 동적인 환경에서 효과적으로 작동하도록 하는 데 필수적입니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안된 알고리즘이 기준 벤치마크 방식에 비해 작업 완료 시간을 효과적으로 줄이는 것을 입증합니다. 논문에서 발표한 모델은 높은 이동성을 지닌 서비스 차량(SVs)과 제한된 기지국 서버 간에 작업 스케줄링을 최적화하는 데 초점을 맞추고 있습니다. 이로 인해 CAV 네트워크에서 작업 완료 시간이 더 짧아질 수 있음을 보여줍니다.



### Abductive Symbolic Solver on Abstraction and Reasoning Corpus (https://arxiv.org/abs/2411.18158)
Comments:
          Presented at IJCAI 2024 LNSAI Workshop

- **What's New**: 이 논문은 Abstraction and Reasoning Corpus(ARC)에서 인공지능의 추론 능력을 향상시키기 위한 도전 과제를 다룹니다. 이전 방법들은 그리드 전환에만 집중했지만, 인간의 시각적 추론 과정에 기반한 새로운 접근 방식을 제안합니다. 저자들은 관찰한 데이터를 지식 그래프로 상징적으로 표현하고 핵심 지식을 추출하여 해결책 생성을 용이하게 합니다.

- **Technical Details**: 제안된 방법론은 세 가지 주요 단계로 나뉩니다. 첫째, ARC 지식 그래프(ARCKG) 구성을 통해 예제 쌍을 하나의 유닛으로 설정합니다. 둘째, 지식 그래프에서 핵심 지식을 추출하고, 셋째, 이 추출된 지식을 활용하여 해결책을 검색합니다. 이 과정에서는 인간의 이해를 반영한 DSL(도메인 특정 언어)을 사용하여 데이터베이스를 형성합니다.

- **Performance Highlights**: 이 연구는 강력한 인간의 추론 과정을 모델링하여 AI의 ARC 작업에 대한 성능 향상을 기대하고 있습니다. 제안된 접근 방식은 솔루션 검색 공간을 제한하고, 핵심 지식 추출에 기반한 논리적인 해결책을 제공하는 데 유망한 결과를 나타냅니다. 이러한 방식은 앞으로 AI가 시각적 추론 문제를 해결하는 데 더욱 효과적일 것입니다.



### MONOPOLY: Learning to Price Public Facilities for Revaluing Private Properties with Large-Scale Urban Data (https://arxiv.org/abs/2411.18085)
Comments:
          CIKM'19

- **What's New**: 이번 논문에서는 개인 자산의 가치를 재산정하기 위한 "Monopoly" 프로젝트를 소개합니다. 이 프로젝트는 Baidu Maps의 대규모 도시 데이터를 활용하여, 공공 시설의 가치를 학습하고 이를 통해 개인 자산의 가치를 정확히 평가하는 방법론을 제안합니다. 특히, 도시 내 여러 관심 지점(POIs)을 무향 가중 그래프로 구성하여, 주택 가격 예측에 필요한 다양한 요소를 통합적으로 고려합니다.

- **Technical Details**: Monopoly 프로젝트는 개인 자산과 공공 시설의 가치를 평가하기 위해 MapReduce(Dean and Ghemawat, 2004) 프레임워크에 기반한 분산 알고리즘을 개발했습니다. 이는 개인 주택 가격과 주변 공공 시설의 가치를 동시에 병렬적으로 추정하고, 예측 손실에 따라 값을 반복적으로 업데이트하는 방식으로 작동합니다. 또한, 주변 공공 시설의 가치를 가상 가격으로 설정하여 개인 자산을 재평가하는 시스템을 구현하였습니다.

- **Performance Highlights**: 상당한 대규모 도시 데이터를 통한 실험 결과, 제안된 방법은 여러 기존 주택 가격 예측 방법들보다 큰 성과를 보였습니다. 특히, 주택 특성과 공공 시설의 구조적 정보 간의 관계를 효과적으로 이용하여 더 나은 예측 결과를 도출할 수 있었습니다. 그 외에도 투자에 대한 주요 인사이트를 제시하여, 사용자와 정부의 도시 계획 및 세금 정책에 유용한 정보를 제공할 것으로 기대하고 있습니다.



### DuMapper: Towards Automatic Verification of Large-Scale POIs with Street Views at Baidu Maps (https://arxiv.org/abs/2411.18073)
- **What's New**: 이 논문에서는 Baidu Maps를 위한 대규모 POI(Point of Interest) 검증을 위한 자동 시스템인 DuMapper를 제시합니다. DuMapper는 거리뷰 데이터(multimodal street-view data)를 활용하여 POI 데이터베이스를 효과적으로 검증하고, 이를 통해 검증 속도를 50배 향상시킬 수 있습니다. 이 시스템은 자동화된 방식으로 POI의 정확성을 검증하며, 상당한 노동 비용을 절감할 수 있는 가능성을 보여줍니다.

- **Technical Details**: DuMapper 시스템은 세 단계의 파이프라인으로 구성되어 있으며, 첫 번째 단계에서는 geo-spatial index(GSI)를 통해 거리뷰 사진에서 촬영된 좌표를 사용하여 후보 POI를 찾습니다. 두 번째 단계에서는 Optical Character Recognition(OCR) 기술을 활용하여 후보 POI의 이름을 인식하고, 마지막 단계에서는 다양한 멀티모달 특징(multimodal features)을 기반으로 후보 POI를 랭킹하여 최종 POI를 선택합니다. DuMapper II 버전에서는 Deep Multimodal Embedding(DME)과 Approximate Nearest Neighbor(ANN) 검색을 통해 검증 속도를 더욱 향상시킵니다.

- **Performance Highlights**: DuMapper는 출시 이후 3.5년 동안 4억 5백만 회 이상의 POI 검증을 수행하여 약 800명의 전문가와 동일한 작업량을 처리하였습니다. DuMapper II는 자동 POI 검증의 처리량을 5050배 증가시키는 것을 입증하였으며, 이로 인해 Baidu Maps의 생산성과 효율성이 크게 개선되었습니다. DuMapper II의 소스 코드는 GitHub를 통해 공개되어 재현 시험이 가능합니다.



### Simulating Tabular Datasets through LLMs to Rapidly Explore Hypotheses about Real-World Entities (https://arxiv.org/abs/2411.18071)
- **What's New**: 이 논문은 공포 소설가들이 다른 작가들보다 어린 시절이 더 힘들었는지에 대한 가설을 탐구하기 위해 대형 언어 모델(LLMs)을 활용한 새로운 접근 방식을 제시합니다. 연구자들은 LLM을 사용하여 특定 실체에 대한 데이터를 신속하게 추정하고, 그 관계를 발견하며, 가설 탐구에 필요한 정량적 속성을 제안하는 기회를 탐색합니다. 이 접근 방식은 기존 데이터에서 과학적으로 흥미로운 패턴을 발굴할 수 있도록 돕는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 LLM을 사용하여 구체적인 실체(예: 사람, 책 등)에 대한 속성을 추정하고 다양한 데이터 소스에서 주요 관계를 발견합니다. 제안된 방법론은 가설을 탐구하기 위한 대규모 데이터셋을 넓은 범위에서 생성하는 데 집중하며, 초기 가설으로부터 시작하여 관련된 정량적 변수를 식별하고 자원 활용을 최적화합니다. 이러한 기술은 데이터의 성격에 대한 혁신적인 이해를 제공하며, LLM 규모에 따라 데이터의 신뢰성이 증가함을 보여줍니다.

- **Performance Highlights**: 사례 연구에서 확인된 바와 같이, LLM은 명백한 엔티티에 대한 유용한 데이터셋을 생성할 수 있으며, 이는 과학적 가설을 보다 체계적으로 탐구할 수 있도록 지원합니다. 연구 성과는 LLM이 고수준 가설을 구체적인 변인으로 변환하는 능력을 갖추고 있음을 입증하며, 현존하는 데이터를 신속하게 추적하는 데 기여할 수 있음을 시사합니다. 이러한 접근 방식은 궁극적으로 대규모 데이터에서 유의미한 과학적 발견을 촉진하는 잠재력을 지니고 있습니다.



### A Novel Pareto-optimal Ranking Method for Comparing Multi-objective Optimization Algorithms (https://arxiv.org/abs/2411.17999)
- **What's New**: 이 논문은 다목적 최적화 알고리즘의 성능을 평가하기 위한 새로운 멀티 메트릭 비교 방법을 제안합니다. 제안된 방법은 파레토 최적 개념을 활용하여 여러 성능 지표를 동시에 고려하여 알고리즘의 순위를 매깁니다. 이를 통해 기존 또는 새롭게 개발된 성능 지표를 활용하여 알고리즘을 효과적으로 평가할 수 있습니다. 또한, 제안된 방법은 확장 가능하고, 새로운 메트릭이 추가되어도 문제 없이 통합할 수 있습니다.

- **Technical Details**: 다목적 최적화는 두 개 이상의 상충하는 목표를 다루는 문제로, 문제의 복잡성으로 인해 최적 해를 찾는 것이 어렵습니다. 이 논문에서는 알고리즘의 성능 지표로서 다수의 평가 지표를 사용하여 최적 해의 품질을 정량적으로 평가하는 것이 중요하다고 강조합니다. 성능 지표는 단일 목표 지표와 다목적 지표로 나뉘며, 제안된 방식은 파레토 수준에서 알고리즘의 기여도를 기반으로 알고리즘을 순위 매깁니다.

- **Performance Highlights**: 제안된 방법은 2018 CEC 대회의 10개 경쟁 알고리즘을 평가하는 데 적용되었습니다. 이 평가에서 10개의 잘 알려진 다목적 성능 지표를 사용하여 파레토 최적 순위를 도출하고, 대회에서 보고된 최종 순위와 비교했습니다. 이로써 제안된 방법은 과학 및 공학 분야, 특히 다수의 메트릭을 사용하는 기계 학습 및 데이터 마이닝에서 널리 적용될 수 있는 가능성을 보여줍니다.



### Can LLMs plan paths in the real world? (https://arxiv.org/abs/2411.17912)
- **What's New**: 최근 대형 언어 모델(LLMs)이 차량 내 내비게이션 시스템에 통합되고 있는 가운데, 이 연구는 LLM의 경로 계획 능력을 검증했습니다. 실험에서는 GPT-4, Gemini, Mistral의 세 가지 LLM을 사용하여 실제 경로 계획 시나리오를 통해 그들의 성능을 평가했습니다. 결과는 모든 LLM이 신뢰할 수 없는 경로 계획자로 나타났으며, 정확도를 향상시키기 위한 현실 검증 메커니즘 및 소형 모델 개발의 필요성이 제기되었습니다.

- **Technical Details**: 연구팀은 Turn-by-Turn (TbT) 내비게이션과 시각-언어 내비게이션(VLN) 두 가지 범주로 실험을 구성하였습니다. 다양한 환경 및 난이도를 포함한 6개의 경로 계획 시나리오가 설계되었고, 각 LLM이 시나리오를 수행하는 동안의 성능이 분석되었습니다. 예를 들어, 도시, 교외 및 농촌 시나리오에서 LLM은 사용자의 목적지까지 안내하는 지침을 제공해야 했습니다.

- **Performance Highlights**: 실험 결과, LLM들은 각 시나리오에서 많은 오류를 범하며 신뢰성이 부족함을 드러냈습니다. 예를 들어, 약 77.8%의 TbT 시나리오에서 LLM이 적절한 대답을 제공하지 못해 후속 질문이 필요했습니다. 이러한 결과는 LLM의 경로 계획 능력에 대한 회의적인 시각을 강화하며, 향후 연구에서는 모델 투명성을 개선하고, 신뢰성 있는 경로 계획을 위한 현실 검증 메커니즘의 개발이 요구된다는 것을 보여줍니다.



### Towards Efficient Neurally-Guided Program Induction for ARC-AGI (https://arxiv.org/abs/2411.17708)
- **What's New**: 이번 논문은 프로그램 유도 (program induction) 방식의 일반화 능력을 평가하는 ARC-AGI 데이터셋에 대한 최초의 분석 결과를 제시합니다. 저자들은 다양한 학습 공간(그리드 공간, 프로그램 공간, 변환 공간)을 통한 실험 결과를 공유하며, 특히 LPS(프로그램 공간 학습)가 ARC 평가 세트에서 최고의 성능을 보이는 방법임을 강조합니다. 이러한 연구는 출처가 다른 데이터 (out-of-distribution data)에 대한 일반화 능력을 강화하기 위한 새로운 접근법들을 제안합니다.

- **Technical Details**: 저자들은 DSL(도메인 특정 언어) 내에서 프로그램을 실행할 수 있도록 필요한 모든 원시 함수들을 포함해야 한다고 가정합니다. 각 실험에서 DSL을 활용해 다양한 그리드 간 유사성을 추정하고 이를 통해 프로그램 유도를 수행하는 방식을 살펴보았습니다. LGS(그리드 공간 학습) 방식은 Transformer 모델을 사용하여 그리드 임베딩을 생성하고, 이를 기반으로 효과적인 프로그램 생성이 이루어지도록 했습니다.

- **Performance Highlights**: LPS 접근법은 ARC-AGI 평가에서 가장 좋은 성능을 보였으며, 이는 프로그램 유도 및 일반화의 효율성을 증명합니다. 저자들은 새로운 확률적 프로그램 열거 기반 탐색 알고리즘을 소개하였고, 이는 Transformer 기반의 자기회귀 토큰 시퀀스를 사용하여 기존 n-그램 접근법에 비해 더 나은 성능을 보였습니다. 본 논문에서 제안된 접근 방식은 프로그램 유도의 기존 한계를 극복할 수 있는 가능성을 내포하고 있습니다.



### Diffusion Self-Distillation for Zero-Shot Customized Image Generation (https://arxiv.org/abs/2411.18616)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Diffusion Self-Distillation이라는 새로운 방법론을 제안하여, 텍스트-이미지 변환 모델을 활용한 이미지 생성 시 보다 정밀한 제어를 가능하게 합니다. 최근 텍스트-이미지 생성 모델의 발전을 바탕으로, 이러한 방법은 높은 품질의 커스텀 이미지를 생성하기 위한 데이터셋을 자체적으로 만들어낼 수 있습니다. 기존의 방법들이 가지고 있는 대규모의 페어링 데이터 부족 문제를 해결하였으며, 이는 다양한 응용 분야에서 유망한 잠재력을 가지고 있습니다.

- **Technical Details**: Diffusion Self-Distillation은 사전 훈련된 텍스트-이미지 모델을 활용하여, 다수의 일관된 이미지를 생성하고 이 데이터를 기반으로 텍스트-이미지 모델을 미세 조정(fine-tune)하는 방식으로 작동합니다. 이 과정에서 Vision-Language Models(VLMs)를 사용하여 이미지 그리드를 자동으로 선별하고, 커스터마이징된 이미지 생성의 필요성에 대해 텍스트-이미지 변환 모델의 효과를 극대화합니다. 또한, 새로운 병렬 처리 아키텍처를 적용하여 효율적인 모델 구성을 수행합니다.

- **Performance Highlights**: Diffusion Self-Distillation은 기존의 제로샷(zero-shot) 방법들보다 우수한 성능을 보여주며, 각 인스턴스 조정 기술에 필적하는 결과를 제공합니다. 이 방법은 고유 정체성을 유지하면서도 다양한 맥락에서의 이미지 생성을 지원합니다. 최종적으로, 이 논문은 아티스트들이 작업을 효과적으로 반복하고 적응할 수 있도록 하여 창의적인 자유를 증진시키는 도구로서의 가능성을 입증합니다.



### Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspectiv (https://arxiv.org/abs/2411.18615)
- **What's New**: 본 논문에서는 다중 작업 학습(Multi-Task Learning, MTL)에서 발생하는 gradient conflict(그래디언트 충돌)를 줄이기 위한 새로운 접근법인 Sparse Training(ST)을 제안합니다. 이는 원래 모델의 일부 파라미터만 업데이트하며 나머지는 동결하여 여러 작업을 동시에 학습하도록 하는 방식입니다. 기존의 gradient manipulation(그래디언트 조작) 방법과 결합하기 용이하다는 장점도 가지고 있습니다.

- **Technical Details**: Sparse Training(ST)은 각 작업이 특정 파라미터 집합에만 영향을 미치도록 하여 그래디언트 간의 간섭을 줄이는 새로운 방법론을 제시합니다. 이 모델은 모든 작업의 평균 손실을 최적화하는 기존의 일반적인 방법보다 효과적이며, 특히 훈련의 후반부에서 그래디언트 충돌 발생이 감소하는 경향을 보입니다. ST의 적용은 대규모 미리 훈련된 모델에서도 효과적임을 보여주며, 여러 그래디언트 조작 기법과 연계하여 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, Sparse Training은 여러 작업이 동시에 수행될 때 그래디언트 충돌을 효과적으로 완화시켜 성능을 향상시키는 것으로 나타났습니다. 대규모 모델에서의 그래디언트 충돌 문제는 더 심각하게 나타나며, ST가 이를 완화하는 데 효과적이라는 점이 강조됩니다. ST의 도입으로 다양한 데이터셋 및 아키텍처에서 그래디언트 간의 충돌 발생 빈도가 감소함을 확인할 수 있습니다.



### Robust Offline Reinforcement Learning with Linearly Structured $f$-Divergence Regularization (https://arxiv.org/abs/2411.18612)
Comments:
          52 pages, 3 figures, 2 tables

- **What's New**: 이번 논문에서는 새로운 프레임워크인 $d$-rectangular linear robust regularized Markov decision process ($d$-RRMDP)을 제안합니다. 이 접근법은 전이 커널(transition kernel)과 정규화에서 선형 잠재 구조(linear latent structure)를 도입하여 기존의 Robust Regularized Markov Decision Process (RRMDP)가 가진 단점을 극복합니다. 특히, $d$-RRMDP는 비현실적인 전이를 고려함으로써 지나치게 보수적인 정책을 피하고, 이론적 통찰을 제공합니다.

- **Technical Details**: 논문에서는 Robust Regularized Pessimistic Value Iteration (R2PVI)이라는 알고리즘 군을 개발합니다. 이 알고리즘은 오프라인 reinforcement learning(강화 학습) 환경에서 사전 수집된 데이터셋을 통해 강건한 정책을 학습합니다. 또한, 선형 함수 근사(linear function approximation)와 전이 커널에 대한 $f$-divergence 기반의 정규화 항을 적용하여 문제를 해결합니다.

- **Performance Highlights**: 수치 실험을 통해 R2PVI가 강건한 정책을 학습하며, 기존 제약된 DRMDP 방법들에 비해 계산적으로 더 효율적임을 확인했습니다. 제안된 알고리즘의 성능은 최적 강건 정책이 방문하는 상태-행동(state-action) 공간을 데이터셋이 얼마나 잘 커버하는지에 따라 달라지며, 이러한 관계는 $d$-RRMDP의 정보 이론적 하한(lower bounds)과도 연결됩니다.



### Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation (https://arxiv.org/abs/2411.18583)
Comments:
          Key Words : T5, SpaCy, Large Language Model, GPT, ROUGE, Literature Review, Natural Language Processing, Retrieval-augmented generation

- **What's New**: 이번 연구에서는 Natural Language Processing (NLP) 기술 및 retrieval-augmented generation (RAG)과 Large Language Model (LLM)을 활용하여 문헌 리뷰의 자동 생성을 위한 여러 접근 방식을 제안하고 비교했습니다. 기존에 수많은 연구 기사가 증가하면서 수동 문헌 리뷰의 어려움이 커졌고, 이에 따른 자동화의 필요성이 증가하고 있습니다. 연구의 주된 목표는 PDF 파일만을 입력으로 받아 자동으로 문헌 리뷰를 생성할 수 있는 시스템을 개발하는 것입니다.

- **Technical Details**: 연구에서는 frequency-based method (spaCy), transformer model (Simple T5), 그리고 Large Language Model (GPT-3.5-turbo)과 결합된 retrieval-augmented generation (RAG) 등 여러 NLP 전략의 효과를 평가했습니다. SciTLDR 데이터 세트를 활용하여 문헌 리뷰 자동 생성을 위한 세 가지 다른 시스템을 구현하는 데 세 가지 독특한 기술이 사용되었습니다. 모든 시스템의 평가에는 ROUGE 점수가 활용되었습니다.

- **Performance Highlights**: 평가 결과, Large Language Model인 GPT-3.5-turbo는 ROUGE-1 점수 0.364로 가장 높은 점수를 기록했습니다. 두 번째는 transformer model이었고, spaCy는 마지막 위치에 있었습니다. 최종적으로, 최상의 성능을 보인 시스템에 대해 그래픽 사용자 인터페이스가 생성되었습니다.



### Functional relevance based on the continuous Shapley valu (https://arxiv.org/abs/2411.18575)
Comments:
          36 pages, 13 figures

- **What's New**: 이 논문에서는 기계 학습 예측 알고리즘의 해석 가능성에 대한 새로운 접근 방식을 제안합니다. 특히, 함수형 데이터에 기반한 예측 모델을 해석하기 위한 방법을 중심으로 합니다. 이를 통해 다양한 데이터 유형을 처리하는 AI 모델의 행동을 이해할 수 있는 기초를 제공합니다.

- **Technical Details**: 함수형 데이터 모델의 해석 가능성 방법을 설계하는 것은 무한한 크기의 특징(feature) 집합을 다루어야 한다는 의미입니다. 본 연구에서는 연속적인 게임을 위한 Shapley value에 기반한 해석 가능성 방법을 제안하였고, 이는 계속적인 플레이어 집합 간에 공정하게 보상을 분배할 수 있게 해줍니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션된 데이터 세트와 실제 데이터 세트를 대상으로 실험하여 그 효과를 입증하였습니다. 또한, 연구에서 사용된 개방형 소스 Python 패키지인 ShapleyFDA도 소개되어, 해석 가능성 향상을 위한 도구로서 활용될 수 있습니다.



### LLM-ABBA: Understand time series via symbolic approximation (https://arxiv.org/abs/2411.18506)
- **What's New**: 이 논문에서는 LLM-ABBA라는 새로운 방법을 제안합니다. 이 방법은 시간 시계열을 상징화하여 LLMs(large language models)의 다양한 후속 시간 시계열 작업에 통합합니다. LLM-ABBA는 기존의 규칙적인 패턴을 유지하면서 시간 시계열의 의미 정보를 활용할 수 있도록 설계되었습니다.

- **Technical Details**: LLM-ABBA는 적응형 브rownian bridge 기반의 상징 집계를 활용하여 시간 시계열을 효과적으로 모델링합니다. 이 방법은 LLM의 기존 토큰을 사용해 시간 시계열의 유용한 정보를 보존하며, 기호화된 시계열을 LLM에 제공하여 패턴 인식을 촉진합니다. QLoRA(Quantized Low-Rank Adaptation) 방법을 통해 작업 성능과 효율성을 완벽하게 조화롭게 유지하면서 다양한 도메인에서의 적응성을 보장합니다.

- **Performance Highlights**: LLM-ABBA는 Time Series Extrinsic Regression(TSER) 벤치마크에서 새로운 SOTA 성능을 달성하며, UCR 및 세 가지 의료 시계열 분류 작업에서 우수한 성과를 보여줍니다. 이 기법은 또한 최근 SOTA 시간 시계열 예측 결과에 비해 경쟁력 있는 예측 능력을 선보입니다. 이 프레임워크는 다른 시계열 작업으로의 원활한 확장이 가능하다고 믿습니다.



### Isometry pursu (https://arxiv.org/abs/2411.18502)
- **What's New**: 이번 논문에서는 orthonormal column-submatrices를 식별하기 위한 convex 알고리즘인 Isometry pursuit를 소개합니다. 이 알고리즘은 새로운 normalization 방법과 multitask basis pursuit를 결합하여, wide matrix에서의 문제에 효과적으로 적용될 수 있습니다. 특히, Jacobian을 활용하여 isometric embedding을 더 잘 식별할 수 있도록 돕는 점이 주목할 만합니다.

- **Technical Details**: Isometry pursuit는 주어진 데이터에서 orthonormal vector를 찾기 위해 설계된 접근 방식입니다. 이 알고리즘은 관찰된 coordinate functions의 Jacobian에서 직접 파생된 isometric embedding을 로그를 통해 식별하는 데 사용됩니다. 이 과정은 기존의 greedy 및 brute force search 방법에 대한 시너지 효과를 제공합니다.

- **Performance Highlights**: 이 알고리즘은 이론적 및 실험적 결과를 통해 그 유효성이 입증되었습니다. 특히, coordinate selection과 diversification 문제를 다룰 때 우수한 성능을 보이며, 기존의 방법들에 비해 더 나은 결과를 도출할 수 있습니다.



### SoK: Watermarking for AI-Generated Conten (https://arxiv.org/abs/2411.18479)
- **What's New**: 본 논문은 생성 AI(GenAI)의 출력 품질이 향상됨에 따라 생성된 콘텐츠와 인간이 만든 콘텐츠를 구별하는 것이 점점 더 어려워지고 있음을 강조합니다. 이를 해결하기 위한 방법으로 워터마킹(watermarking) 기법이 제안되며, 이는 AI가 생성한 콘텐츠에 숨겨진 신호를 삽입하여 신뢰할 수 있는 검증을 가능하게 합니다. 워터마킹은 GenAI의 안전성과 신뢰성을 높이는 중요한 역할을 할 수 있으며, 이 논문은 워터마킹 기법에 대한 종합적인 개관을 제공합니다.

- **Technical Details**: 생성 AI의 출처를 구별하기 위해 워터마킹 기법은 AI 모델이 콘텐츠를 생성할 때 불가시 신호를 적절히 삽입합니다. 이러한 방식은 신호를 통해 AI 생성 콘텐츠의 출처를 검증할 수 있도록 하며, 기존의 통계적 차이 점검 방식에서 벗어나 보다 지속적이고 효과적인 검증 수단을 제공합니다. 이 논문은 워터마킹 기법에 대한 정의와 필요한 속성을 정리하고 있으며, 효과적인 워터마크의 특징으로는 모델 출력 품질 유지, 높은 검출 정확도, 공격에 대한 강인성 등이 포함됩니다.

- **Performance Highlights**: 저자들은 워터마킹이 생성 AI 콘텐츠 검출에서 보다 신뢰할 수 있는 해결책이라고 제안하며, 그 중요성이 점점 커지고 있음을 명확히 합니다. 연구에서는 워터마킹 기술의 실제 평가 전략을 모색하고, 다양한 공격에 저항할 수 있는 강력한 워터마킹 기법 개발 인사이트를 제공합니다. 또한, 워터마킹의 역사적 배경과 현재의 지정학적 맥락에서의 역할을 강조하며, 향후 연구와 개발을 위한 중요한 방향성을 찾고 있습니다.



### Weakly Supervised Framework Considering Multi-temporal Information for Large-scale Cropland Mapping with Satellite Imagery (https://arxiv.org/abs/2411.18475)
- **What's New**: 이번 연구에서는 대규모 농경지 매핑을 위해 약한 감독 환경(weakly supervised framework)에서 다중 시점 정보(multi-temporal information)를 고려한 방법을 제안합니다. 기존의 원격 감지(remote sensing) 데이터와 심층 학습(deep learning) 기법의 조합은 우수한 성능을 보였으나, 많은 양의 정밀 레이블(precise labels)이 필요하여 노동 집약적입니다. 이에 따라, 우리는 높은 품질의 레이블을 글로벌 토지 피복(Global Land Cover, GLC) 제품 간의 일관성에 따라 추출하여 감독 학습 신호(supervised learning signal)를 구축합니다.

- **Technical Details**: 제안하는 프레임워크는 고품질 레이블의 잔여 오류를 과신하는 문제를 해결하기 위해 농경지(cropland)의 유사성(similarity) 및 집합체(aggregation)를 시각적/공간적 영역에서 인코딩하여 비감독 학습 신호(unsupervised learning signal)를 구성합니다. 이를 통해 감독 부분을 제약하기 위한 정규화 항(regularization term)으로 활용합니다. 또한 고품질 레이블이 없는 샘플에서도 비감독 신호를 포함시켜 특징 공간(feature space)의 다양성을 풍부하게 만듭니다.

- **Performance Highlights**: 이 프레임워크는 후난 성(Hunan Province), 남동 프랑스(Southeast France), 캔자스(Kansas)의 세 연구 지역에서 대규모 농경지 매핑에서 강력한 적응성을 실험적으로 검증했습니다. 다중 시점 정보가 농경지 추출에 어떻게 기여하는지를 밝히기 위해 고차원 생리적 특징(phenological features)을 시각화하였고, 데이터 희소성(data scarcity) 조건에서도 방법의 견고성을 평가했습니다.



### Hotspot-Driven Peptide Design via Multi-Fragment Autoregressive Extension (https://arxiv.org/abs/2411.18463)
Comments:
          Preprint, Under review

- **What's New**: 이번 논문에서는 PepHAR이라는 새로운 모델을 제안합니다. PepHAR는 특정 단백질을 겨냥한 펩타이드 디자인을 위한 핫스팟 중심의 자기 회귀 생성 모델입니다. 이 모델은 핫스팟 잔여물에 기반하여 펩타이드를 설계하고 기존의 펩타이드 모델들이 겪는 여러 도전 과제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: PepHAR의 설계 과정은 세 단계로 나뉩니다. 첫 번째로, 에너지 기반 밀도 모델을 사용하여 주변 잔여물의 분포를 캡처하고, Langevin dynamics를 통해 통계적으로 유리한 핫스팟을 샘플링합니다. 다음으로, 잔여물을 동시에 생성하는 대신 자기 회귀적으로 단계적으로 프래그먼트를 확장하며, 이를 통해 펩타이드 결합 기하학을 유지합니다.

- **Performance Highlights**: PepHAR의 성능은 펩타이드 디자인 및 골격 생성 실험에서 뚜렷하게 나타났습니다. 실험 결과, PepHAR은 펩타이드 결합체 설계 및 골격 생성 모두에서 경쟁력을 갖춘 성능을 발휘하여 실용적인 펩타이드 약물 개발 시나리오를 성공적으로 시뮬레이션할 수 있음을 입증하였습니다.



### Draft Model Knows When to Stop: A Self-Verification Length Policy for Speculative Decoding (https://arxiv.org/abs/2411.18462)
Comments:
          Code at this https URL

- **What's New**: 이 논문에서는 Speculative Decoding (SD)의 성능을 향상시키기 위해 SVIP(자기 검증 길이 정책)를 도입하였습니다. SVIP는 토큰 생성의 난이도를 인식하여 초안 길이를 동적으로 조정하는 방식을 사용합니다. 이는 고정된 초안 길이 설정에서 발생하는 비효율성을 해결하며, 각 초안 토큰 분포의 엔트로피를 기반으로 합니다.

- **Technical Details**: SVIP의 핵심 기술은 초안 모델의 엔트로피를 분석하여 초안 시퀀스의 길이를 동적으로 조정하는 것입니다. 이 시스템은 쉽게 예측할 수 있는 단어에 대해서는 더 긴 초안을 생성하고, 예측이 어려운 단어가 등장할 경우 더 빨리 검증 과정으로 넘어갑니다. 이를 통해 전체적인 생성 속도를 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, SVIP는 SpecBench에서 기존 SD 방법에 비해 20% 이상의 속도 향상을 달성하였고, MT-Bench에서는 최대 60% 속도 향상을 확인했습니다. 이 방법은 훈련이 필요 없고, 기존의 단독 또는 자동 회귀 초안 모델과 호환성이 높아 다양한 SD 시스템에 적용 가능합니다.



### Synthetic ECG Generation for Data Augmentation and Transfer Learning in Arrhythmia Classification (https://arxiv.org/abs/2411.18456)
- **What's New**: 본 연구에서는 Deep Learning 기반의 생성 모델을 활용하여 생리학적 데이터 특히 심전도(ECG) 데이터의 데이터셋을 확장하고, 심박 리듬 분류의 정확도를 향상시키는 방법을 다룹니다. 기존 데이터에 대한 보강을 통해 심정의 리듬을 잘 분류할 수 있는 새롭고 강력한 접근 방식을 제시하고 있습니다. 이 연구는 다양한 생성 모델(Diffweave, Time-Diffusion 및 Time-VQVAE)을 통해 생성된 합성 데이터의 유용성을 평가하고 있습니다.

- **Technical Details**: 본 연구에서 사용된 생성 모델들인 Diffweave, Time-Diffusion, Time-VQVAE는 각기 다른 방식으로 심전도 데이터를 생성합니다. 특히, 합성 데이터의 품질은 실제 데이터와의 유사성 측면에서 평가되며, 전이 학습(transfer learning)을 통해 미리 학습된 모델을 세부 조정합니다. 이러한 모델들은 실제 데이터 비율을 점진적으로 추가할 때 성능 향상을 목표로 하고 있습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터가 단독 데이터셋에 비해 눈에 띄는 성능 개선을 가져오지는 않지만, 두 데이터셋을 통합했을 때 분류기 성능이 모든 지표에서 향상되었습니다. Time-VQVAE 생성 모델이 가장 우수한 성능을 보였지만, 여전히 실제 데이터만으로 훈련된 분류기에는 미치지 못하는 것으로 나타났습니다. 본 연구는 합성 데이터와 실제 데이터의 근접성을 평가하기 위한 방법과 지표도 탐구하였습니다.



### Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation (https://arxiv.org/abs/2411.18447)
Comments:
          Accepted to NeurIPS 2024 - Audio Imagination Workshop

- **What's New**: 이 연구에서는 Continuous Autoregressive Models (CAMs)를 제안하여, 생성 품질 저하 문제를 해결하기 위해 훈련 데이터에 무작위 잡음을 주입하는 방법을 introduced합니다. 이 방법을 통해 모델은 추론 중 다양한 오류 수준에 강건해지며, 생성 과정에서의 오류 발생을 줄입니다. 음악 오디오 생성 실험을 통해 CAM은 기존의 autoregressive 및 non-autoregressive 접근 방식보다 현저한 성능 향상을 보여주고, 긴 시퀀스 생성 시에도 품질 저하가 없습니다.

- **Technical Details**: CAM은 기존의 autoregressive 모델이 가진 단점을 극복하기 위해 훈련 데이터에 잡음을 섞는 직관적인 방법을 사용합니다. 이 과정에서 모델은 "진짜" 신호와 "오류" 신호를 구별하는 법을 배우게 되어, 추론 중 오류 전파에 강해집니다. 또한, 생성된 embedding에 소량의 인공 잡음을 추가하는 간단한 추론 기법을 도입해 오류 축적에 대한 저항력을 강화합니다.

- **Performance Highlights**: CAM을 사용한 음악 스템의 무조건 생성 실험 결과, 기존의 모델보다 생성 품질에서 현저하게 개선됐습니다. 특히, 긴 시퀀스에서의 성능 저하 없이 안정적이고 고품질의 오디오 생성을 보여주었습니다. 이는 실시간 음악 동반 시스템 및 종단 간 음성 대화 모델과 같은 강력한 인터랙티브 응용 프로그램을 가능하게 할 것입니다.



### Is my Meeting Summary Good? Estimating Quality with a Multi-LLM Evaluator (https://arxiv.org/abs/2411.18444)
- **What's New**: MESA는 자연어 생성(NLG) 시스템에서 생성된 회의 요약의 품질을 보다 정확하게 평가하기 위한 새로운 프레임워크입니다. 기존의 ROUGE 및 BERTScore와 같은 메트릭들이 인간 평가와의 낮은 상관관계로 인한 한계점을 극복하는 데 초점을 맞추고 있습니다. MESA는 세 가지 단계의 오류 평가 및 다수의 에이전트 간 토론을 통해 결정 정제를 수행하고 피드백 기반 자기 훈련을 적용하여 인간 판단에 대한 이해와 정렬을 강화합니다.

- **Technical Details**: MESA는 오류별 평가, 전체 평가 및 자기 훈련의 세 가지 단계로 구성됩니다. 각 오류 유형에 대해 가능성 있는 오류를 식별하고 그 영향을 평가하여 Likert 점수(0-5)를 할당합니다. 특히 다수의 에이전트 간 논의 프로토콜을 활용하여 다양한 관점을 반영한 동적 정제 단계를 거치며, 세 단계의 평가를 통해 포괄적인 오류 탐지를 보장합니다.

- **Performance Highlights**: MESA는 GPT-4o를 기반으로 하여 기존의 평가자들보다 평균 0.25 높은 오류 탐지에서의 인간 평가 상관관계를 달성하였습니다. 실험 결과, 자율 훈련 단계는 인간 판단과의 정렬을 돕고, 오류 인스턴스의 잘못된 긍정 식별을 줄이는 데 기여하였습니다. MESA의 유연성은 맞춤형 오류 가이드라인에 적응할 수 있어 다양한 작업에 활용될 수 있는 가능성을 지니고 있습니다.



### Metric-DST: Mitigating Selection Bias Through Diversity-Guided Semi-Supervised Metric Learning (https://arxiv.org/abs/2411.18442)
Comments:
          18 pages main manuscript (4 main figures), 7 pages of supplementary

- **What's New**: 이 논문에서는 기계 학습에서 공정성을 확보하기 위한 새로운 접근법인 Metric-DST를 제안합니다. 기존의 self-training 방법은 높은 신뢰도를 가진 데이터 샘플을 포함하려는 경향이 있는데, 이는 이미 존재하는 모델 편향을 강화할 수 있습니다. Metric-DST는 메트릭 학습과 다양한 샘플 포함 전략을 결합하여 이러한 신뢰도 기반 편향에 대응합니다.

- **Technical Details**: Metric-DST는 메트릭 학습(metric learning)을 활용하여 다양성 기반의 self-training 전략을 구현합니다. 이 방법은 레이블이 없는 데이터의 표현을 생성하고, 이를 바탕으로 더 다양한 샘플을 선택하여 모델 학습에 포함시킵니다. Metric-DST는 고차원의 데이터에서도 적용 가능하며, 어떤 유형의 분류기와도 함께 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, Metric-DST는 생성된 데이터와 실제 데이터에서 선택 편향이 있는 상황에서도 더 견고한 모델을 학습할 수 있음을 보여주었습니다. 기존의 방법들과 비교했을 때 Metric-DST는 선택 편향을 완화하고 기계 학습 모델의 공정성을 높이는 데 유연하고 광범위하게 적용 가능한 솔루션을 제공합니다.



### MM-Path: Multi-modal, Multi-granularity Path Representation Learning -- Extended Version (https://arxiv.org/abs/2411.18428)
- **What's New**: 이 논문은 MM-Path라는 새로운 Multi-modal, Multi-granularity Path Representation Learning Framework를 제안합니다. 이 모델은 도로 경로와 이미지 경로로부터 정보를 통합하여 일반적인 경로 표현을 학습합니다. 이전 연구들이 단일 모드 데이터에 의존했지만, 이 연구에서는 다중 모드 데이터의 통합을 통해 더 깊고 포괄적인 경로 이해를 도모합니다. 이는 intelligent transportation 분야에서의 다양한 응용 가능성을 확대합니다.

- **Technical Details**: MM-Path는 멀티 그래뉼러리 정렬 전략과 그래프 기반의 크로스 모달 잔여 융합 요소를 도입하여, 도로 경로와 이미지 경로 간의 정보 통합을 촉진합니다. 이러한 구성은 파라미터 공간에서 다양한 그래뉼러리 레벨을 유지하며, 로드와 이미지 경로 간의 정밀한 정렬을 보장합니다. 또한, 공간적 맥락 정보가 포함된 잔여 융합 요소는 두 개의 다른 모달리티 간의 정보를 효과적으로 융합합니다.

- **Performance Highlights**: 두 개의 대규모 실제 데이터셋에서 광범위한 실험을 수행하여 MM-Path의 효과를 검증했습니다. 다양한 다운스트림 작업에서 제안된 모델은 뛰어난 적응성과 우수한 성능을 보여주었습니다. 이는 경로 표현 학습 모델의 일반화 능력을 크게 향상시키며, 도시 계획과 긴급 관리와 같은 여러 분야에서의 활용 가능성을 제시합니다.



### Optimal In-Network Distribution of Learning Functions for a Secure-by-Design Programmable Data Plane of Next-Generation Networks (https://arxiv.org/abs/2411.18384)
- **What's New**: 이번 논문은 프로그래머블 데이터 플레인 (PDP)와 인 네트워크 컴퓨팅 (INC) paradigm의 발전으로 인해 고급 컴퓨팅 작업을 수행할 수 있는 네트워크 장치들, 특히 분산 침입 탐지 시스템 (IDS) 지원을 위한 in-network 학습 모델 구현의 필요성을 다룹니다. 논문은 '강력한 학습자' (Strong Learner, SL) 모델을 경량화된 '약한 학습자' (Weak Learner, WL) 모델로 세분화하여 IDS 작업을 최적화하여 네트워크 보안을 보장하는 것을 목표로 합니다. 또한, 수학적 모델이 제공하는 정확한 솔루션으로 인해 발생하는 긴 계산 시간을 줄이기 위한 메타 휴리스틱 접근 방식을 제안하고 검토했습니다.

- **Technical Details**: 이 논문에서 제안하는 새로운 Active IDS paradigm을 통해 SL 모델을 여러 WL로 분할하고 이를 가상 네트워크 기능(virtual network functions, VNF)으로 매핑하여 데이터를 처리하고 응답할 수 있는 기능을 제공합니다. 이를 위해 최적의 학습 기능 분배를 구현하는 오케스트레이션이 필요하며, 이 과정에서 (i) 침입 탐지 정확도를 꾸준히 향상시키고, (ii) 처리 부하를 줄이며, (iii) 네트워크 장치의 표준 기능(예: 패킷 전달)에 미치는 영향과 위협에 대한 반응 시간을 줄이는 것이 중요합니다.

- **Performance Highlights**: 제안된 접근 방식의 분석 결과는 네트워크 장치의 추가 부담을 최소화하면서 사이버 공격에 대해 최초의 방어선 역할을 효과적으로 수행할 수 있는 지능형 데이터 플레인 생성의 잠재력을 강조합니다. 이 연구는 PDP 장치와 인 네트워크 분산 학습을 함께 활용하여 완전 분산형 Active IDS를 구현할 수 있는 새로운 가능성을 제시하며, 이를 통해 보안 범위와 성능의 균형을 이루는 효율적인 배치를 위한 최적화 모델을 제안합니다.



### ChatGPT as speechwriter for the French presidents (https://arxiv.org/abs/2411.18382)
- **What's New**: 이번 연구는 ChatGPT라는 최신 LLM의 글쓰기 스타일을 분석하고, 최근 프랑스 대통령들의 연말 연설문과 비교하는 새로운 시도를 하였습니다. 연구의 주요 초점은 ChatGPT로 생성된 메시지가 실제 정치 지도자의 메시지와 어떤 차이를 보이는지를 불을 밝히는 것입니다. 이러한 접근은 Generative AI에 대한 우려와 기대를 동시에 반영하며, 사용자의 요청에 대한 응답을 자동으로 생성하는 능력에 대한 실질적인 분석을 제공합니다.

- **Technical Details**: 연구에서는 ChatGPT가 생성한 메시지를 Chirac, Sarkozy, Hollande, Macron의 연말 연설문과 비교 분석하였습니다. 분석 결과, ChatGPT는 명사(nouns), 소유 한정사(possessive determiners), 숫자(numbers)를 과도하게 사용하고 있으며, 동사(verbs), 대명사(pronouns), 부사(adverbs)의 사용은 상대적으로 적습니다. 특히, 'devoir', 'nous'와 같은 특정 단어가 과다 사용되고 있는 반면, 'être', 'vouloir', 'falloir'와 같은 조동사는 상대적으로 부족하게 나타났습니다.

- **Performance Highlights**: ChatGPT가 짧은 텍스트를 제공받았을 때, 원문에 가까운 스타일로 메시지를 생성할 수 있는 능력을 보여주었습니다. 그러나 전반적으로 ChatGPT의 발화 스타일은 실제 대통령 연설과 비교할 때 뚜렷한 차별점을 드러냅니다. 이러한 연구 결과는 LLM의 글쓰기 스타일에 대한 심층적인 통찰력을 제공하며, AI 기반의 글쓰기 도구들이 어떻게 발전해야 할지를 고민하게 만듭니다.



### G3Flow: Generative 3D Semantic Flow for Pose-aware and Generalizable Object Manipulation (https://arxiv.org/abs/2411.18369)
Comments:
          Webpage: this https URL

- **What's New**: G3Flow는 실시간 동적 객체 중심 3D 의미 표현을 구축하는 새로운 프레임워크입니다. 이 프레임워크는 기초 모델들을 활용하여 기하학적 정밀성과 의미적 이해를 통합하여 로봇의 작업 성능을 크게 향상시킵니다. 특히, G3Flow는 수동 주석 없이도 일관된 의미적 이해를 유지할 수 있는 방법을 제시합니다.

- **Technical Details**: G3Flow는 3D 생성 모델, 비전 기초 모델, 강력한 자세 추적을 결합하여 동적 의미 흐름을 생성합니다. 이 과정은 두 단계로 나뉘며, 첫 번째 단계에서는 로봇이 다중 관측 데이터를 수집하여 디지털 트윈을 생성하고 기본 의미 필드를 형성합니다. 두 번째 단계에서는 실시간 자세 추적을 통해 의미 필드를 동적으로 변형하여 조작 과정에서의 일관성을 유지합니다.

- **Performance Highlights**: G3Flow는 5가지 시뮬레이션 작업에 대한 광범위한 실험 결과에서 기존의 접근 방법을 초과하여 성능을 입증하였습니다. 최종 제약이 있는 조작 작업에서 최대 68.3%, 객체 간 일반화 작업에서 50.1%의 평균 성공률을 달성하였습니다. 이러한 결과는 G3Flow가 로봇 조작 정책에서 실시간 동적 의미 기능 이해를 향상시키는 데 효과적임을 보여줍니다.



### AMPS: ASR with Multimodal Paraphrase Supervision (https://arxiv.org/abs/2411.18368)
- **What's New**: 이 논문에서는 다국어 대화형 자동 음성 인식(ASR) 시스템 향상을 위한 새로운 기술인 AMPS를 소개합니다. AMPS는 여러 언어(힌디어, 마라티어, 말라얄람어, 칸나다어 및 니얀자어)의 대화형 ASR 성능을 개선하기 위해 패러프레이즈(Paraphrase) 기반의 지원을 통합합니다. 이를 통해 기존 ASR 모델의 훈련 과정에서 패러프레이즈를 추가적인 감독으로 사용하여 미흡한 ASR 성능을 보완할 수 있습니다.

- **Technical Details**: AMPS는 다중 모달(multimodal) ASR 모델인 SeamlessM4T를 통해 구현됩니다. 이 모델은 음성 인코더 및 텍스트 인코더와 공유 디코더를 포함하여 음성을 텍스트로 변환하는 음성-텍스트(S2T) 및 텍스트-텍스트(T2T) 경로를 생성합니다. AMPS는 ASR 손실이 높은 경우 패러프레이즈 기반의 추가 지원을 적용하여 모델이 명확하지 않은 오디오에서 의미적으로 유사한 단어를 선택할 수 있는 대안을 제공합니다.

- **Performance Highlights**: AMPS를 사용하여 ASR 성능이 크게 향상된 것을 보고합니다. 여러 언어에서 단어 오류율(Word Error Rate, WER)이 최대 5% 감소한 것으로 나타났으며, 이는 인도 언어를 포함한 다양한 언어의 대화형 음성을 인식하는 데 중요한 성과입니다. 또한 인간 평가를 통해 AMPS의 출력을 기존 ASR 목표로만 파인튜닝한 경우와 비교하여 일관된 개선 결과를 확인하였습니다.



### GPT as ghostwriter at the White Hous (https://arxiv.org/abs/2411.18365)
- **What's New**: 최근 여러 대형 언어 모델(LLMs)이 사용자 요청에 대한 메시지를 생성할 수 있는 능력을 보여주며 새로운 관점을 제시하고 있습니다. 이 연구는 ChatGPT 3.5의 글쓰기 스타일을 분석하며, 미국 대통령들의 연설과 비교합니다. 차별화된 접근 방식을 통해 LLM의 특징을 구체적으로 조사합니다.

- **Technical Details**: 이 연구에서는 레이건(Reagan)부터 오바마(Obama)까지의 국정연설(State of the Union addresses)과 ChatGPT가 자동으로 생성한 연설을 비교합니다. 분석 결과, ChatGPT는 'we'라는 단어를 과도하게 사용하며, 명사와 쉼표의 사용 빈도가 높습니다. 반면, 동사 사용은 적고 평균적으로 더 긴 문장이 생성되는 경향이 있습니다.

- **Performance Highlights**: 결과적으로, ChatGPT는 지정된 스타일을 강제하더라도 생성된 연설의 스타일이 대상 작가의 메시지와는 뚜렷하게 다르다는 것을 보여주었습니다. 또한, ChatGPT는 주로 긍정적인 감정 표현과 기호적 용어(예: freedom, nation)를 사용하는 중립적인 톤을 선택하는 경향을 보입니다. 이러한 특성은 실제 대통령 연설과 명확한 차별점을 드러냅니다.



### TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models (https://arxiv.org/abs/2411.18350)
- **What's New**: 이 논문에서는 VTOFF(Virtual Try-Off)라는 새로운 작업을 소개합니다. 이는 옷을 입은 사람의 단일 사진에서 표준화된 의상 이미지를 생성하는 데 중점을 두며, 전통적인 Virtual Try-On(VTON)과는 달리 직접적으로 모델을 ‘입히는’ 방식이 아닙니다. VTOFF는 의상 형태, 질감, 복잡한 패턴을 캡처하는 데 도전과제를 제공합니다.

- **Technical Details**: 우리는 TryOffDiff라는 모델을 제안하여 Stable Diffusion에 SigLIP 기반의 시각적 조정을 도입하여 높은 재구성 품질과 세부사항을 보존합니다. VITON-HD 데이터셋을 수정하여 실시한 실험에서는 우리의 접근 방식이 포즈 전이 및 기존 메서드보다 우수한 결과를 보여주며, 처리 과정이 간소화되었습니다. 또한 기존 이미지 생성 메트릭이 재구성 품질을 적절히 평가하지 못함에 따라, 더욱 정확한 평가를 위해 DISTS를 사용하였습니다.

- **Performance Highlights**: VTOFF의 결과는 전자상거래 애플리케이션에서 제품 이미지를 개선하고 생성 모델 평가를 발전시키며, 고품질 재구성을 위한 미래 작업에 영감을 줄 가능성을 보여줍니다. 생성된 의상 이미지는 기존의 가상 시도 솔루션과 통합되어 복잡한 착용자 간 시도도 가능하게 하며, 고객의 구매 결정을 도울 수 있습니다. 이러한 접근 방식은 궁극적으로 패션 산업의 환경적 영향을 줄이는 데 기여할 수 있습니다.



### FreqX: What neural networks learn is what network designers say (https://arxiv.org/abs/2411.18343)
Comments:
          16pages, 9 figures

- **What's New**: 본 논문은 개인화된 연합 학습(PFL)에서의 해석 가능성을 개선하기 위해 새로운 방법인 FreqX를 제안합니다. FreqX는 신호 처리(Signal Processing) 및 정보 이론(Information Theory)를 활용하여 모델이 학습한 내용을 계층별로 설명하는 방법입니다. 기존 해석 방법들이 갖고 있던 한계를 극복하며, 최소한 10배 빠른 성능을 자랑하여 실용적 활용 가능성을 높였습니다.

- **Technical Details**: FreqX는 DNN(Deep Neural Networks)의 해석 가능성을 향상시키기 위해 신호 처리와 정보 병목 이론을 결합합니다. 이 방법은 시간(t) 도메인에서의 삭제 작업을 주파수(frequency) 도메인으로 변환하여 정보 손실 없이 신호를 해석할 수 있도록 합니다. 이렇게 도출된 평가 결과는 다양한 주파수에서의 기능 정보와 노이즈 정보를 포함하여, 네트워크가 학습한 내용을 더 정교하게 설명할 수 있습니다.

- **Performance Highlights**: 실험 결과, FreqX는 기존 방법들과 비교하여 설명 속도가 최소 10배 빠른 것으로 나타났습니다. 또한, 이 방법은 네트워크의 예측 변화에 대한 민감도를 도출하였으며, 중요한 주파수를 삭제하는 것만으로도 네트워크의 출력이 크게 변경됨을 확인했습니다. 이러한 결과는 FreqX가 모델 해석의 새로운 경로를 제시함을 의미합니다.



### Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation (https://arxiv.org/abs/2411.18335)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 Helvipad라는 새로운 omnidirectional stereo depth estimation 데이터셋을 소개합니다. 이 데이터셋은 40,000 개의 프레임으로 구성되어 있으며, 다양한 조명 조건을 갖춘 실내 및 실외의 복잡한 장면에서 수집되었습니다. Helvipad는 고품질 깊이(depth) 및 불일치(disparity) 레이블을 포함하며, depth completion을 통해 레이블 밀도를 증가시킨 훈련 세트를 제공합니다.

- **Technical Details**: Helvipad 데이터셋은 두 개의 360° 카메라와 LiDAR 센서를 사용하여 수집되었습니다. 데이터는 equirectangular 이미지로 투영된 3D 포인트 클라우드를 통해 정확한 깊이 레이블을 제공합니다. 또한 이 연구에서는 극좌표 맵을 입력으로 도입하고 원형 패딩(circular padding)을 적용하여 Omnidirectional 이미지에 대한 stereo matching 모델을 개선하는 방법을 제안합니다.

- **Performance Highlights**: Benchmarking 결과에 따르면, 최근의 stereo matching 모델들이 omnidirectional 접근법보다 높은 성능을 보이지만, equirectangular 투영에서의 심각한 왜곡으로 인해 여전히 도전 과제가 있습니다. 이 논문에서 제안된 모델의 개조는 이전의 모든 접근법을 초월하는 성능 향상을 가져왔으며, 특히 우리가 수집한 Helvipad 데이터셋에서 효과적인 성능을 발휘했습니다.



### RITA: Automatic Framework for Designing of Resilient IoT Applications (https://arxiv.org/abs/2411.18324)
- **What's New**: IoT(Internet of Things) 시스템의 복원력을 디자인하는 데 필요한 자동화된 오픈소스 프레임워크인 RITA가 소개되었습니다. 이 프레임워크는 IoT 요구 문서에서 IoT Critical Object(ICO)를 식별하고 위협을 연관지으며 대응책을 제안하는 기능을 갖추고 있습니다. RITA는 RoBERTa 기반의 Named Entity Recognition(NER) 모델을 활용하며 오프라인으로 작동하여 데이터 프라이버시를 보호합니다.

- **Technical Details**: RITA의 구조는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, NER 모델을 통해 IoT Critical Object를 자동으로 추출하여 키 장치, 자원 및 서비스의 식별을 가능하게 합니다. 둘째, IoT 위협 데이터베이스를 기반으로 위협을 식별하고, 셋째, 위협에 적합한 완화 전략을 선택합니다. 이 모든 구성 요소는 ADD4RIOT(Architectural Design Decisions for Resilient IoT) 메타 모델에 기반하여 설계되었습니다.

- **Performance Highlights**: RITA는 실험적 평가에서 ChatGPT를 사용한 것보다 4개의 ICO 카테고리에서 더 우수한 성능을 보였습니다. 특히 엑추에이터, 센서, 네트워크 자원 및 서비스 식별에서 탁월한 성능을 나타냈습니다. 이 결과는 RITA가 IoT 시스템 보안 작업을 효과적으로 지원하여 복원력 있는 IoT 아키텍처 개발에 기여할 수 있음을 시사합니다.



### Learning optimal objective values for MILP (https://arxiv.org/abs/2411.18321)
- **What's New**: 이 논문에서는 현대의 혼합 정수 선형 프로그래밍(MILP) 솔버에서 최적 목표 값을 예측하는 새로운 방법론을 제안합니다. 이 방법론은 그래프 신경망(GNN) 아키텍처를 기반으로 하며, 역동적인 특징(dynamic features) 집합을 활용합니다. 실험 결과, 제안된 접근 방식이 높은 정확성을 달성하며 기존의 방법들을 능가하는 성능을 보여주었습니다. 이러한 발견은 기계 학습(ML) 기반 예측을 MILP 솔버에 통합할 수 있는 새로운 기회를 제시합니다.

- **Technical Details**: MILP는 현실 세계에서 수많은 최적화 문제를 모델링하는 도구로 사용되며, Branch-and-Bound(B&B) 알고리즘을 통해 최적 해에 도달합니다. 연구에서는 최적 목표 값을 예측하기 위한 새로운 방법론을 제안하며, 이로 인해 밀접한 관련이 있는 두 가지 질문에 대한 정확한 답변을 제공합니다. 첫 번째 질문은 최적 목표 값을 얼마나 잘 예측할 수 있는지이고, 두 번째 질문은 주어진 솔루션이 최적인지 여부를 예측할 수 있는 정확도입니다. 제안된 예측기는 B&B 과정에서 중요한 의사결정에 도움을 줄 수 있습니다.

- **Performance Highlights**: 제안된 예측기가 높은 정확도를 보여주며, 기존의 방법과 비교하여 성능이 향상되었습니다. 실험 연구를 통해 예측 모델의 출력과 추가 데이터를 사용하는 분류기를 통해 최적 해를 판단하는 데 유용하다는 것을 확인했습니다. 이러한 성과는 MILP 솔버의 성능 향상과 더 스마트한 의사 결정 관행을 가능하게 합니다. 전체 실험을 재현하기 위한 코드는 온라인에서 제공됩니다.



### Continual Learning in Machine Speech Chain Using Gradient Episodic Memory (https://arxiv.org/abs/2411.18320)
Comments:
          Published as a conference paper at O-COCOSDA 2024. 6 pages; 2 figures

- **What's New**: 이 논문은 자동 음성 인식(ASR) 시스템을 위한 지속적 학습 기법을 제안합니다. 이러한 접근법은 머신 스피치 체인 프레임워크를 활용하여 기존의 학습된 과제의 성능을 유지하면서 새로운 과제를 순차적으로 학습할 수 있도록 합니다. 특히, 텍스트-음성 변환(TTS) 컴포넌트를 통합하여 재생(replay) 메커니즘을 지원함으로써 기계 학습 모델이 연속적으로 직면하는 여러 과제를 효과적으로 다룰 수 있습니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성됩니다. 첫 번째 단계는 기초 작업에 대한 감독 학습(supervised learning)이며, 두 번째 단계는 반감독 학습(semi-supervised learning)으로 ASR와 TTS가 서로 향상되는 과정입니다. 마지막 세 번째 단계는 새로운 작업을 위한 지속적 학습으로, TTS에 의해 생성된 기초 작업의 입력을 재생하여 수행됩니다. 이 과정에서 기울기 에피소딕 메모리(GEM)를 사용하여 이전 작업 데이터와 새로운 작업 데이터 간의 기울기를 조정합니다.

- **Performance Highlights**: 실험은 LJ Speech 데이터셋을 사용하여 수행되었으며, 제안된 방법은 전통적인 파인튜닝(fine-tuning) 및 멀티태스크 학습(multitask learning) 접근법에 비해 성능이 뛰어난 것으로 나타났습니다. 특히, 제안된 방법은 신호 대 잡음비(SNR) 조건에서 높은 성능을 유지하면서도 평균 오류율을 40% 감소시키는 성과를 달성했습니다. 따라서 우리의 반감독 머신 스피치 체인 접근법은 효과적이고 효율적인 지속적 학습의 가능성을 증명하고 있습니다.



### MvKeTR: Chest CT Report Generation with Multi-View Perception and Knowledge Enhancemen (https://arxiv.org/abs/2411.18309)
Comments:
          10 pages, 10 figures

- **What's New**: 이번 연구는 multi-view를 활용한 진단 정보를 포함하고 클리닉 전문 지식과 Kolmogorov-Arnold Networks (KANs)를 활용한 새로운 Multi-view perception Knowledge-enhanced Transformer (MvKeTR)를 제안합니다. 이 모델은 방사선 전문의가 CT 영상을 여러 방향에서 분석하는 방식을 모방하여 진단 정보를 효과적으로 통합합니다. 또한, 관련 임상 기록을 참조하여 진단 의사 결정을 지원하는 Cross-Modal Knowledge Enhancer (CMKE)를 도입하여 의료 지식을 보고서 생성 과정에 통합합니다.

- **Technical Details**: 연구에서는 Multi-View Perception Aggregator (MVPA)를 설계하여 여러 해부학적 관점에서 진단 정보를 통합합니다. 일반적으로 3D CT 볼륨의 다수 관점 분석을 통해 더 나은 공간 특징을 캡쳐할 수 있게 되며, KANs를 기본 구성 요소로 사용하여 복잡한 진단 관계를 학습합니다. 이러한 접근은 진단 과정에서의 의료 지식 격차를 메우고, 정확하고 신뢰할 수 있는 보고서를 생성하기 위해 필수적입니다.

- **Performance Highlights**: CTRG-Chest-548K 데이터세트에서 수행한 광범위한 실험에서는 MvKeTR가 기존의 최첨단 모델들을 모든 지표에서 초월함을 보여주었습니다. 자동화된 CT 보고서 생성 과정에서 MvKeTR는 이전 시스템에 비해 품질과 효율성을 모두 향상시킨 것으로 평가됩니다. 이로 인해 의사들의 부담을 덜고 환자 관리의 질을 개선하는 데 기여할 수 있습니다.



### Application of Soft Actor-Critic Algorithms in Optimizing Wastewater Treatment with Time Delays Integration (https://arxiv.org/abs/2411.18305)
- **What's New**: 이 연구는 복잡한 동역학과 느린 시간 상수를 가진 하수 처리장에 최적화된 제어 방법을 제안했습니다. 전통적인 Proportional-Integral-Derivative (PID) 제어기 대신, Soft Actor-Critic 알고리즘을 기반으로 한 새로운 심층 강화 학습 접근 방식을 사용하여 저항 문제를 해결합니다.

- **Technical Details**: 연구는 하수 처리장에서 발생하는 지연 피드백을 모델링하기 위한 커스텀 시뮬레이터와 Long Short-Term Memory (LSTM) 네트워크를 통합했습니다. 이 시뮬레이터는 실제의 훈련 시나리오를 가능하게 하며, 제 agents(에이전트)는 지연 시나리오에 따라 훈련되었습니다: 지연 없음, 일정한 지연, 무작위 지연.

- **Performance Highlights**: 강화 학습 프레임워크에 무작위 지연을 포함시키면서 하수 처리의 인산염 제거 효율성이 크게 향상되었습니다. 시뮬레이션된 환경에서, 지연 인식 에이전트는 전통적 제어 방법보다 인산염 배출을 36%, 보상은 55%, 규제 한치를 벗어난 목표 이탈은 77% 줄였으며, 총 운영 비용은 9% 감소한 결과를 보였습니다.



### Aligning Pre-trained Models for Spoken Language Translation (https://arxiv.org/abs/2411.18294)
- **What's New**: 이 논문은 사전 훈련된 자동 음성 인식(ASR) 모델과 기계 번역(MT) 모델을 소형 커넥터 모듈(Q-Former)로 결합하는 새로운 접근 방식을 탐구합니다. 이 커넥터는 음성 및 텍스트 모달리티 간의 갭을 해소하고 ASR 인코더 임베딩을 MT 인코더의 잠재 표현 공간으로 변환합니다. 본 연구는 How2 영어-포르투갈어 데이터셋에서 실험을 수행하며, 프레임워크의 효용성을 입증합니다.

- **Technical Details**: 본 연구에서 제안한 두 가지 정렬 아키텍처는 서로 다른 구성의 MT 모델을 사용하는데, 둘 다 ASR 모델의 인코더 부분만 사용합니다. 첫 번째 아키텍처인 Encoder-Connector-Decoder (ECD)는 고정된 ASR 인코더에서 숨겨진 오디오 표현을 추출하여 이를 커넥터 모듈을 통해 MT 디코더의 차원으로 변환합니다. 두 번째 아키텍처인 Encoder-Connector-Encoder-Decoder (ECED)에서는 커넥터 모듈이 음성 임베딩을 MT 인코더의 입력 단어 임베딩 공간으로 투영합니다.

- **Performance Highlights**: ASR와 MT 모델의 크기를 증가시키면 음성 번역 결과가 전반적으로 향상되며, 커넥터 네트워크의 크기는 작게 유지될 수 있습니다. 또한, 커넥터 네트워크는 도메인 어댑터 역할을 하여, 정렬된 MT 모델이 도메인 외부에서 더욱 향상된 번역 성능을 보여줍니다. 마지막으로, 제안된 프레임워크는 저자원 시나리오에서도 효과적임을 입증합니다.



### DualCast: Disentangling Aperiodic Events from Traffic Series with a Dual-Branch Mod (https://arxiv.org/abs/2411.18286)
- **What's New**: 이 논문에서는 교통 예측 모델의 성능을 향상시키기 위한 새로운 구조인 DualCast를 제안합니다. DualCast는 두 가지 신호를 분리하여 통합 학습할 수 있도록 설계된 이중 분기 구조를 가지고 있으며, 이는 내재적(spatial-temporal) 패턴과 외부 환경 맥락(aperiodic events)을 모두 반영합니다. 교통 사고와 같은 비주기적 이벤트를 효과적으로 처리하기 위해 Cross-time attention 메커니즘을 도입하였습니다.

- **Technical Details**: DualCast는 특히 비주기적 패턴 처리를 강화하기 위해 세 가지 손실 함수인 filter loss, environment loss, DBI loss를 활용합니다. 이러한 손실 함수들은 두 가지 유형의 신호를 분리하여 학습 결과를 융합하는 데 도움을 줍니다. 또한, Cross-time attention 모듈을 통해 다양한 시간에서의 공간-시간 상관관계를 취합하고, Attention fusion 모듈을 통해 지역적 주의(local attention)와 글로벌 주의(global attention)를 통합하여 수용 영역을 확장합니다.

- **Performance Highlights**: DualCast를 GMAN, STTN, PDFormer와 같은 최신 교통 예측 모델과 통합하여 실험을 실시한 결과, 여러 실제 데이터셋에서 평균 예측 오류를 최대 9.6%까지 감소시켰습니다. 이는 DualCast가 복잡한 환경 맥락에서 더 큰 성과를 보임을 나타내며, 기존의 SOTA 모델보다도 2.6%까지 우수한 성능을 기록했습니다.



### GAPartManip: A Large-scale Part-centric Dataset for Material-Agnostic Articulated Object Manipulation (https://arxiv.org/abs/2411.18276)
- **What's New**: 본 논문에서는 가정 환경에서의 관절형 물체 조작의 중요성을 강조하며, GAPartManip이라는 대규모 데이터셋을 소개합니다. 이 데이터셋은 포토 리얼리스틱(material) 랜더링과 부분 지향적인 상호작용 자세 주석을 특징으로 하여, 실세계 환경에서의 딥퍼셉션(depth perception)과 행동 가능 자세 예측을 향상시키는 데 중점을 두었습니다. 기존의 방법들이 겪는 깊이 추정에 대한 제약을 뛰어넘기 위해, 데이터 생성 파이프라인에 도메인 랜덤화를 도입했습니다.

- **Technical Details**: GAPartManip 데이터셋은 19개의 일반 가정용 관절형 카테고리를 포함하며, 총 918개의 객체 인스턴스로 구성됩니다. 이 데이터셋은 RGB 이미지, IR 이미지, 깊이 맵 및 부분 수준의 세분화를 포함하며, 각 관절형 객체의 부분에 대한 고품질의 물리적으로 그럴듯한 상호작용 자세 주석이 제공됩니다. NVIDIA Isaac Sim을 기반으로 한 포토 리얼리스틱 랜더링 파이프라인을 통해 데이터 샘플이 생성되며, 각 씬에서 다양한 도메인 랜덤화 설정으로 렌더링되었습니다.

- **Performance Highlights**: 저자들은 GAPartManip 데이터셋을 활용하여 깊이 재구성 네트워크와 행동 가능 자세 예측 네트워크 각각을 훈련시켜, 기존 방법들과 비교하여 우수한 성능을 입증했습니다. 본 연구의 모듈 구성 방식은 포괄적인 실험을 통해 단일 모듈 및 부분 조작 실험 모두에서 State-Of-The-Art (SOTA) 성능을 향상시켰습니다. 이를 통해, 관절형 물체 조작 과제에서의 실제 적용 가능성을 높였습니다.



### Wearable intelligent throat enables natural speech in stroke patients with dysarthria (https://arxiv.org/abs/2411.18266)
Comments:
          5 figures, 45 references

- **What's New**: 이번 연구에서는 AI 기반의 지능형 목(throat) 시스템을 소개합니다. 이 시스템은 목 근육 진동과 경동맥(pulse) 신호 센서를 통합하여 음성 장애가 있는 환자들이 감정적으로 풍부한 의사소통을 할 수 있도록 지원합니다.

- **Technical Details**: 시스템은 초감각 섬유(strain) 센서를 사용하여 목 부위에서 고품질 신호를 포착하며, 실시간으로 지속적인 음성 디코드를 지원하기 위해 토큰 레벨(token-level) 처리를 적용합니다. 이를 통해 지연 없는 원활한 의사소통이 가능합니다.

- **Performance Highlights**: 다섯 명의 발음 장애가 있는 뇌졸중 환자를 대상으로 한 테스트에서, 시스템은 낮은 오류율(단어 오류율 4.2%, 문장 오류율 2.9%)을 기록하며 사용자 만족도가 55% 향상되었습니다. 이 연구는 발음 장애 환자들을 위한 휴대용이며 직관적인 의사소통 플랫폼을 제시하며, 다양한 신경학적 상태 및 다국어 지원 시스템에 확장할 가능성을 보여줍니다.



### Multimodal Integration of Longitudinal Noninvasive Diagnostics for Survival Prediction in Immunotherapy Using Deep Learning (https://arxiv.org/abs/2411.18253)
- **What's New**: 이번 연구에서는 비침습적인(longitudinal) 다중 모달(multimodal) 데이터를 인공지능(AI)을 사용하여 분석함으로써, 암 환자를 위한 면역요법(immunotherapy)이 정밀의료(precision medicine)로 나아가는 새로운 방향을 제시합니다. 694명의 환자 데이터를 기반으로 한 대규모 연구를 통해, 치료 전후의 혈액 측정치, 처방된 약물, 그리고 CT 기반의 장기 부피 정보를 통합하였습니다.

- **Technical Details**: 이 연구에서는 MMTSimTA(multi-modal transformer-based simple temporal attention) 네트워크의 다양한 변형을 사용하여, 생존 예측을 위해 3개월, 6개월, 9개월, 12개월의 사망률을 예측하는 모델을 훈련했습니다. 이 모델은 중간 및 후속 융합(fusion) 방법을 포함한 기초 방법과 성능을 비교하였습니다.

- **Performance Highlights**: 연장된 트랜스포머 기반(multimodal model)의 가장 강력한 예측 성능은 3개월, 6개월, 9개월, 12개월 생존 예측에 대해 각각 AUC(area under the curve) 값이 $0.84 \, 	ext{±} \, 0.04$, $0.83 \, 	ext{±} \, 0.02$, $0.82 \, 	ext{±} \, 0.02$, $0.81 \, 	ext{±} \, 0.03$로 나타났습니다. 초기 치료 데이터 분석이 면역요법 환자의 생존 예측에 유망함을 보여주었습니다.



### IKUN: Initialization to Keep snn training and generalization great with sUrrogate-stable variaNc (https://arxiv.org/abs/2411.18250)
- **What's New**: 본 논문에서는 SNN (Spiking Neural Network)을 위한 새로운 가중치 초기화 방법인 IKUN을 제안합니다. 기존의 ANN (Artificial Neural Network) 초기화 기법이 SNN의 복잡한 요구를 충족하지 못하는 한계를 극복하는 데 중점을 둡니다. IKUN은 신호 전파를 안정화하고 수렴 속도를 가속화하며 일반화 성능을 향상시킵니다.

- **Technical Details**: IKUN은 대체 그래디언트(Surrogate Gradient) 함수와 함께 통합되어 SNN의 신호 전파 특성을 최적화합니다. 정량적 분석을 통해 IKUN이 전방 전파 및 역전파 모두에서 신호 및 그래디언트 분산을 안정화하도록 설계되었음을 보여줍니다. 이 방법은 Sigmoid, Tanh, Linear와 같은 다양한 대체 그래디언트 함수와 호환 가능하여 높은 유연성을 자랑합니다.

- **Performance Highlights**: IKUN을 사용한 실험에서는 훈련 효율성이 최대 50% 향상되었으며, 95%의 훈련 정확도와 91%의 일반화 정확도를 달성했습니다. 헤시안 분석 결과, IKUN으로 훈련된 모델은 더 평평한 최소값에 수렴함을 나타냈으며, 이는 더 나은 일반화를 촉진하는 효과가 있습니다. 이 방법은 오픈 소스 형태로 제공되어 추가적인 탐색이 가능합니다.



### Thai Financial Domain Adaptation of THaLLE -- Technical Repor (https://arxiv.org/abs/2411.18242)
- **What's New**: 본 연구에서는 태국 금융 도메인에 맞춘 대형 언어 모델(LLM)인 Thai Financial LLM을 개발했습니다. 이는 태국 증권 거래소의 투자 상담사(IC) 시험 데이터셋을 활용하여 전문 용어와 현지 규제를 포함한 특정 요구 사항에 대응하고자 했습니다. 기존의 금융 LLM들이 태국 금융 시장의 특수성을 충분히 반영하지 못했다는 점에서 이 연구는 중요한 기여를 합니다.

- **Technical Details**: 태국 금융 LLM의 개발에는 데이터 증강(data augmentation), 효율적인 학습을 위한 ReLoRA, 도메인 지식 구축을 위한 Continued Pretraining(CPT), 그리고 미세 조정을 위한 Rank-Stabilized LoRA(rsLoRA) 기법이 사용되었습니다. SFT(Supervised Fine-Tuning)를 통해 시험 환경을 시뮬레이션하였고, DPO(Direct Preference Optimization)를 통해 모델의 피드백 기반 개선을 수행했습니다.

- **Performance Highlights**: 모델은 IC 시험의 P1, P2, P3 단계에서 각각 72%, 72%, 84%의 점수를 기록하였으며, 이는 태국 금융 자문 업무에 대한 효과성을 보여줍니다. 이 연구는 태국의 고유한 금융 지식을 포함한 LLM의 필요성을 강조하며, 전문화된 금융 어플리케이션에 대한 잠재력을 제공합니다.



### Exploration of LLM Multi-Agent Application Implementation Based on LangGraph+CrewAI (https://arxiv.org/abs/2411.18241)
- **What's New**: 본 논문은 LangGraph와 CrewAI의 통합 응용을 다루며, 이들의 조합이 멀티 에이전트 시스템의 복잡한 작업을 관리하는 데 뛰어난 성능을 보인다고 설명합니다. LangGraph는 그래프 아키텍처를 통해 정보 전송의 효율성을 높이고, CrewAI는 지능형 작업 할당과 자원 관리를 통해 팀 협업 능력을 향상시킵니다. 이 연구는 멀티 에이전트 시스템에서 LangGraph와 CrewAI의 활용을 탐구하며, 에이전트 기술의 미래 발전을 위한 새로운 시각을 제공합니다.

- **Technical Details**: LangGraph는 대형 언어 모델(LLMs)을 활용하여 에이전트 및 멀티 에이전트 워크플로우를 구축할 수 있는 프레임워크입니다. 이 프레임워크는 제어 가능성, 반복 기능, 지속 가능한 메모리를 통해 신뢰할 수 있는 에이전트를 생성할 수 있도록 지원합니다. CrewAI는 AI 에이전트를 조정하고 역할을 정의하는 오픈소스 프레임워크로, 복잡한 문제 해결에 필요한 협업을 촉진합니다. LangGraph와 CrewAI의 통합은 작업 관리의 강력한 도구를 제공하며 복잡한 상황에서도 유연한 워크플로우와 에이전트 간의 협력을 가능하게 합니다.

- **Performance Highlights**: LangGraph와 CrewAI의 조합은 작업 실행의 효율성을 개선하고 시스템의 유연성과 확장성을 향상시키는 중요한 도구로 평가됩니다. 연구 결과에 따르면, 이 통합 프레임워크는 복잡한 작업과 멀티 에이전트 협업을 처리하는 데 필요한 강력한 툴킷을 제공합니다. 실제 사례 연구를 통해 각 작업에 대해 실시간 상태 데이터 공유와 피드백 메커니즘을 구현하여 작업 생성 및 프로세스 효율성을 크게 향상시켰습니다.



### Certified Training with Branch-and-Bound: A Case Study on Lyapunov-stable Neural Contro (https://arxiv.org/abs/2411.18235)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Lyapunov 안정성을 만족하는 신경망(Neural Network) 제어기를 학습하는 새로운 방법인 CT-BaB를 제안합니다. CT-BaB는 인증된 교육 프레임워크(certified training framework)로, 기존의 방법들과는 달리 훈련 과정에서 검증과 관련된 계산을 동시에 고려합니다. 이를 통해 인증 작업을 보다 효과적으로 수행할 수 있는 모델을 생성할 수 있습니다.

- **Technical Details**: CT-BaB 프레임워크는 IEEE 802.1Q에 기반하여, 훈련 중에 Lyapunov 조건 위반에 대한 검증된 경계를 최적화하도록 설계되었습니다. 입력 공간의 전체적인 대역 범위(resolution)와 함께 동적 서브영역(subregion) 유지 관리를 통해, 훈련 중 발생하는 가장 어려운 사례를 작은 하위 영역으로 나누어 훈련하는 방식입니다. 이로써 훈련의 난이도를 감소시키고, 더 좁은 범위 내에서 검증된 경계를 관리할 수 있습니다.

- **Performance Highlights**: 제안한 CT-BaB 방식은 2D 쿼드로터의 동적 시스템에서 실험적으로 적용되었으며, 검증 시간이 1.1시간에서 11.5분으로 5배 이상 단축되었습니다. 또한, 제어기가 작동 가능한 안정성 지역(region-of-attraction, ROA)이 기존의 방법보다 16배 더 크게 확장되었습니다. 이러한 성과는 더 효율적인 테스트 시간 검증과 비약적인 성능 향상을 보여줍니다.



### Randomized-Grid Search for Hyperparameter Tuning in Decision Tree Model to Improve Performance of Cardiovascular Disease Classification (https://arxiv.org/abs/2411.18234)
- **What's New**: 이 논문은 심혈관 질환 진단을 위한 머신러닝 시스템 설계에 새로운 방법을 제안합니다. Randomized-Grid Search라는 하이브리드 최적화 방법을 통해 기존의 Grid Search와 Random Search의 장점을 결합하여 하이퍼파라미터 최적화를 수행합니다. 이를 통해 의미 있는 성능 향상을 이룰 수 있음을 보여줍니다.

- **Technical Details**: 제안된 모델은 Decision Tree 모델의 하이퍼파라미터를 최적화하는 데 중점을 두며, UCI 심장질환 데이터셋에 적용되어 분류 작업을 수행합니다. Random Search는 빠른 하이퍼파라미터 공간 탐색을 제공하지만 최적의 영역을 놓칠 수 있는 반면, Grid Search는 모든 영역을 철저히 탐색하지만 계산 비용이 매우 높습니다. Randomized-Grid Search는 이러한 두 방식의 장점을 통합하여 전반적으로 더 나은 성능을 이끌어냅니다.

- **Performance Highlights**: 실험 결과는 Randomized-Grid Search가 전통적인 방법보다 현저한 성능 향상을 가져오는 것을 확인했습니다. 제안된 모델은 더 높은 정확도와 일반화 능력을 제공하며, 계산 효율성 또한 개선되었습니다. 이 연구는 의료 진단 분야의 머신러닝 응용에 더 효과적인 솔루션을 제시합니다.



### Feature-Factory: Automating Software Feature Integration Using Generative AI (https://arxiv.org/abs/2411.18226)
Comments:
          14 pages, 1 figure

- **What's New**: Feature-Factory는 Generative AI를 활용하여 소프트웨어 프로젝트의 기능 통합 과정을 자동화하는 혁신적인 프레임워크입니다. 기존의 수동적인 코드 분석 및 의존성 해결 방법을 넘어, 이 시스템은 자동화된 프로젝트 파싱, 의존성 해소 및 AI로 생성된 코드를 통해 기능 요청을 통합할 수 있도록 설계되었습니다. 또한, 프로젝트의 구조적 무결성을 보존하는 동시에 새로운 기능을 원활하게 통합하도록 합니다.

- **Technical Details**: 프레임워크는 기능 요청 F를 기반으로 기존 프로젝트 구조 P를 업데이트하는 과정을 거칩니다. 주요 구성 요소로는 프로젝트 구조 파싱, 벡터 데이터베이스 구축, 의존성 해결, 그리고 생성된 태스크에 대한 검증이 포함됩니다. 수학적 모델은 프로젝트의 내재 구조를 표현하는 의존성 그래프 G를 생성하고, 기능 요청에 맞춰 필요한 태스크 집합 Tasks(F)를 도출하여 각 모듈에서 적용합니다.

- **Performance Highlights**: Experimental results show that the Feature-Factory framework effectively integrates features into complex existing software systems, enhancing both development speed and accuracy. The paper validates the framework's performance through various use cases, demonstrating its capability of preserving existing functionalities while implementing new features. This comprehensive automation solution marks a significant advancement over existing static analysis tools and manual methods, providing developers with a powerful tool for modern software engineering.



### PATHS: A Hierarchical Transformer for Efficient Whole Slide Image Analysis (https://arxiv.org/abs/2411.18225)
- **What's New**: 이 논문에서는 Whole Slide Images (WSIs)를 처리하기 위한 새로운 방법인 PATHS(Pathology Transformer with Hierarchical Selection)를 제안합니다. 이 방법은 인간 병리학자가 슬라이드를 검사하는 방식을 모방하여, 저배율에서 이미지를 먼저 분석하고 중요한 영역을 찾아내며 이를 반복하여 고배율로 확대하는 구조를 가지고 있습니다. 이를 통해 불필요한 데이터를 줄이고, 보다 정보가 풍부한 데이터만을 활용할 수 있습니다.

- **Technical Details**: PATHS는 패치 처리 시 계층 구조를 가지며, 처음에 저배율에서 높이 입력 정보를 캡처하고 반복적으로 중요 지역을 선택하는 어텐션 메커니즘을 사용합니다. 이러한 구조는 슬라이드 전체를 처리하지 않고도 많은 해상도에서 정보를 캡처할 수 있게 해줍니다. 이 모델은 또한 각 배율에서 슬라이드의 일부만을 처리하여 계산 효율성을 높입니다.

- **Performance Highlights**: PATHS는 The Cancer Genome Atlas (TCGA)의 다섯 가지 데이터셋에서의 슬라이드 수준 예측 작업에서 기존 방법과 비교하여 뛰어난 성능을 보여줍니다. 이 모델은 처리 시간에서 10배 이상의 속도 향상을 이루었으며, 임상적으로 의미 있는 패치 선택 방법을 제공하여 병리학자의 작업 흐름을 모사합니다.



### R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edg (https://arxiv.org/abs/2411.18220)
- **What's New**: 본 논문에서는 멀티태스킹 대형 언어 모델(MTLLM)을 효과적으로 구축하는 방법으로 새로운 접근, 즉 R-MTLLMF(강건한 MTLLM 융합)를 제안합니다. 이 방법은 외부의 적대적 공격으로부터 작업 벡터 집합을 보호하여 통신의 신뢰성을 향상시킵니다. 기존의 연합 학습(Federated Learning)에서의 한계를 극복하고, 태스크 벡터를 통해 훈련된 모델들을 효율적으로 결합하는 새로운 모델 융합 방식을 도입했습니다.

- **Technical Details**: MTMF(멀티태스킹 모델 융합) 시스템 모델과 WDE(가중치 분리 오류) 및 MSE(평균 제곱 오차)의 관계를 분석하여, 네트워크의 적대적 노이즈와 태스크 간의 간섭을 연구하였습니다. 특히, MIMO(다중 입력 다중 출력) 채널의 최악의 조건을 가정하여 적대적 공격자가 발생시키는 최악의 채널 노이즈를 수치적으로 설계하였습니다. R-MTLLMF는 이러한 노이즈가 존재할 때 태스크 벡터 집합을 안전하게 보호하는 것을 목표로 합니다.

- **Performance Highlights**: R-MTLLMF는 실제 실험에서 8개의 다양한 데이터셋을 사용하여 평가되었습니다. 이상적인 노이즈 조건에서 기준 성능에 가까운 결과를 보여주었고, 최악의 상황에서도 보호되지 않은 모형 융합보다 상당히 우수한 성능을 발휘했습니다. 이러한 결과는 물리적 레이어 보호가 필요한지와 관련하여 추가적인 연구의 필요성을 제시합니다.



### SCoTT: Wireless-Aware Path Planning with Vision Language Models and Strategic Chains-of-Though (https://arxiv.org/abs/2411.18212)
- **What's New**: 이번 논문은 복잡한 무선 환경에서 경로 계획을 지원하기 위해 비전 언어 모델(Vision Language Models, VLMs)을 활용하는 새로운 접근 방식을 제안합니다. 연구는 디지털 트윈(Digital Twin, DT) 데이터를 활용하여 평균 경로 이득을 보장하면서 경로 길이를 최소화하는 방법론을 개발합니다. 이 방법은 기존의 경로 계획 알고리즘의 한계를 극복하고, VLM을 통해 더욱 효율적인 데이터 처리와 의사 결정을 가능하게 합니다.

- **Technical Details**: 이 논문에서는 전통적인 알고리즘인 A*와 여러 무선 인지 확장 개념을 비교하고, 모든 경로 이득 및 거리 메트릭을 고려한 최적 반복 동적 프로그래밍 접근법(Dynamic Programming for Wireless Awareness, DP-WA*)을 도출합니다. 이 과정에서 SCoTT(tasking)라는 전략적 사고 절차를 사용하여 복잡한 경로 계획 작업을 여러 하위 문제로 나누고, 각 문제를 고급 CoT(Chain of Thought) 프롬프트를 통해 해결합니다. VLM이 사용하는 다중 모달 데이터는 이미지와 텍스트를 포함하여 DT로부터 효율적인 경로 추적을 가능하게 합니다.

- **Performance Highlights**: SCoTT는 DP-WA* 알고리즘에 비해 유사한 평균 경로 이득을 달성하면서도 경로 길이를 일관되게 단축합니다. 또한 SCoTT의 첫 두 단계 결과를 DP-WA*의 입력으로 사용함으로써 실행 시간을 최대 62%까지 절감할 수 있음을 보여줍니다. 이러한 결과들은 VLM이 차세대 디지털 시스템에서 복잡한 작업을 해결하는 데 있어 강력한 보조 도구가 될 수 있다는 가능성을 강조합니다.



### TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability (https://arxiv.org/abs/2411.18211)
- **What's New**: 이번 연구는 비디오 중심의 고품질 대화 생성을 위한 새로운 Video-LLM(TimeMarker)을 소개합니다. TimeMarker는 비디오 콘텐츠의 정확한 시간적 위치를 강조하며, Temporal Separator Tokens를 통해 시간 인식을 향상시킵니다. 다양한 길이의 비디오를 효과적으로 처리하기 위한 AnyLength 메커니즘이 도입되어, 긴 비디오에서도 세밀한 정보를 유지할 수 있도록 설계되었습니다. 또한, 복합적인 데이터셋을 활용하여 시간 이해 능력을 강화하였습니다.

- **Technical Details**: TimeMarker는 Temporal Separator Tokens를 통합하여 비디오의 절대 시간적 위치를 인코딩합니다. 이는 텍스트와 비디오 프레임 토큰을 상호 연결하여 생성된 것이며, 특정 순간을 정확히 식별할 수 있게 합니다. AnyLength 메커니즘을 통해 비디오의 길이에 따라 동적 프레임 샘플링과 토큰 융합을 적용하여, 짧은 비디오의 경우 더 많은 세부 정보를 캡처하고, 긴 비디오의 경우 효율적으로 내용을 관리할 수 있습니다. 또한, 5M 이상의 비디오-텍스트 쌍과 85M 이미지를 포함한 다양한 데이터셋을 활용합니다.

- **Performance Highlights**: TimeMarker는 여러 공개 비디오 벤치마크에서 최첨단 성과를 기록하며, 짧은 비디오와 긴 비디오 모두에서 우수한 성능을 발휘합니다. 특히 시간적 문장 기반 위치 지정 작업에서 전통적인 모델들을 초월하며, 비디오 분석에서의 시간적 이해 능력을 강조합니다. 이러한 성과는 TimeMarker의 시간적 위치 지정 기능과 이해 능력이 뛰어남을 나타냅니다.



### From Open Vocabulary to Open World: Teaching Vision Language Models to Detect Novel Objects (https://arxiv.org/abs/2411.18207)
- **What's New**: 이 논문은 Open Vocabulary Object Detection (OVD) 모델이 개방형 환경(오픈 월드)에서 작동할 수 있는 새로운 프레임워크를 제안합니다. 기존 OVD는 정확한 프롬프트에 의존하기 때문에 중요한 응용 분야에서 한계를 가집니다. 본 연구는 Open World Embedding Learning (OWEL)과 Multi-Scale Contrastive Anchor Learning (MSCAL) 접근 방식을 통해, 모델이 새로운 객체를 지속적으로 학습하고 탐지할 수 있도록 합니다.

- **Technical Details**: OWEL은 파라미터화된 클래스 임베딩을 최적화하여 전체 모델을 미세 조정하지 않고 새로운 클래스를 학습할 수 있게 합니다. Pseudo Unknown Embedding 개념을 도입하여, 현재 알려진 클래스를 기반으로 잃어버린 클래스의 위치를 추정합니다. MSCAL은 여러 규모에서 클래스 인식을 돕기 위해 객체 임베딩의 클래스 내 일관성을 증대시키며, 알려진 클래스와의 혼동을 줄이는 데 기여합니다.

- **Performance Highlights**: 이 연구는 M-OWODB 및 S-OWODB 벤치마크에서 U-Recall 성능이 최첨단 수준을 초과하며, 기타 메트릭에서도 우수한 성능을 유지합니다. 또한 nuScenes 기반의 새로운 벤치마크에서도 최고의 결과를 달성했습니다. 이러한 연구 결과들은 OVD 모델이 오픈 월드 환경에서 효과적으로 동작할 수 있게 해 주는 기초적인 기여를 합니다.



### Learning for Long-Horizon Planning via Neuro-Symbolic Abductive Imitation (https://arxiv.org/abs/2411.18201)
Comments:
          Accepted by KDD2025. The KDD version is titled ''Abductive Learning for Neuro-Symbolic Grounded Imitation''

- **What's New**: 최근의 학습 기반 모방(imitation) 방법들은 관찰-행동(observation-action) 공간에서 모방을 통해 계획을 세우는 데 유망한 결과를 보여주고 있습니다. 본 연구에서는 ABductive Imitation Learning (ABIL)이라는 새로운 프레임워크를 제안하여 데이터 기반 학습과 기호 기반(reasoning) 사고의 장점을 통합하였습니다. 이는 긴 수명의 계획(long-horizon planning)을 가능하게 하며, 기호 공간(symbolic space)에서 시연을 이해하기 위해 유도적 사고(abductive reasoning)를 사용하여 감지와 추론 사이의 충돌을 해결하는 데 도움을 줍니다.

- **Technical Details**: ABIL은 시연의 이해를 돕기 위해 유도적(reasoning) 사고를 사용하고, 시퀀스 일관성(sequential consistency) 원칙을 적용하여 감지와 추론 간의 충돌을 해소합니다. 이 과정에서 논리적 추론을 활용하여 기호 후보(predicate candidates)를 생성하며, 이는 번거로운 기호 주석 없이도 원시 관찰(raw observations)을 기호 공간으로 변환할 수 있도록 합니다. 또한, 기호 이해를 바탕으로 다양한 논리적 목표를 가진 정책 앙상블(policy ensemble)을 구축하여 기호 기반 추론으로 관리합니다.

- **Performance Highlights**: 실험을 통해 ABIL이 작업 관련 기호를 활용하여 관찰을 효과적으로 이해하고, 모방 학습을 지원함을 보여주었습니다. 특히, ABIL은 여러 긴 수생 작업(Long-horizon tasks)에서 데이터 효율성과 일반화 능력이 크게 향상된 성능을 입증했습니다. 이는 ABIL을 긴 수명 계획 분야의 유망한 솔루션으로 부각시키며, 실제 세계의 다양한 작업에서 적용 가능성을 제시합니다.



### Prediction with Action: Visual Policy Learning via Joint Denoising Process (https://arxiv.org/abs/2411.18179)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 Prediction with Action Diffuser (PAD)라는 새로운 정책 학습 프레임워크를 소개합니다. PAD는 이미지 예측과 로봇 행동을 통합하여 동일한 확산 변환기(diffusion transformer, DiT) 구조 아래에서 공동의 탈진 과정(joint denoising process)을 통해 이루어지는 특징이 있습니다. 이 접근법은 서로 다른 데이터 세트를 동시에 훈련(co-training)할 수 있도록 하여 다양한 물리적 지식을 인코딩할 수 있게 합니다. PAD는 이전 방식들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 예측 및 행동을 결합하는 PAD는 각종 입력 모달리티를 매끄럽게 통합하고 미래의 이미지와 행동을 동시에 예측하는 방식으로 작동합니다. 이 방법은 로봇 제어 환경에서의 액션 인코딩 및 이미지 예측 과정을 통합하여 robot 상태와 비주얼 데이터를 결합할 수 있게 하였습니다. 또한, PAD는 RGB 비디오 데이터셋을 활용하여 로봇 학습을 최적화할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: PAD는 메타월드 벤치마크에서 이전 방법들과 비교해 26.3%의 상대적 향상을 달성하여 더욱 효과적인 정책 학습을 입증하였습니다. 또한 실제 로봇 조작 환경에서도 전반적인 일반화 성능이 우수하여, 이전 가장 강력한 기준선보다 28.0% 더 높은 성공률을 기록하였습니다. 이러한 결과는 PAD가 새로운 로봇 조작 과제를 수행할 수 있는 잠재력을 지니고 있음을 보여줍니다.



### PDZSeg: Adapting the Foundation Model for Dissection Zone Segmentation with Visual Prompts in Robot-assisted Endoscopic Submucosal Dissection (https://arxiv.org/abs/2411.18169)
- **What's New**: 본 연구에서는 Prompted-based Dissection Zone Segmentation (PDZSeg) 모델을 제안하여 내시경 점막하 박리(ESD) 수술 중 해부 구역의 세분화(segmentation) 개선을 목표로 하고 있습니다. 이 모델은 다양한 시각적 프롬프트(visual prompts)를 활용하여 수술 중 해부 구역을 명확하게 제시할 수 있도록 설계되었습니다. 기존의 방식과 달리, 사용자가 스크리블(scribble)이나 경계 상자(bounding box)를 통해 직관적으로 입력할 수 있도록 지원함으로써 Segmentation 성능을 향상시킵니다.

- **Technical Details**: 연구에서는 DINOv2라는 파운데이션 모델을 기반으로 하여 내시경 수술에 특화된 세분화 모델을 개발했습니다. 이를 위해, 저차원 행렬(low-rank matrices) 삽입을 통해 모델의 레이어를 조정하여 전문 분야에서의 예측 성능을 최적화할 수 있는 Low-Rank Adaptation (LoRA) 방법론을 적용했습니다. 또한, 1,849장의 이미지를 포함한 ESD-DZSeg 데이터셋을 구축하여 다양한 시각적 프롬프트 요청을 대응할 수 있도록 했습니다.

- **Performance Highlights**: 결과적으로, PDZSeg 모델은 기존의 최첨단 세분화 접근법보다 뛰어난 성능을 보였습니다. 연구에서 제안한 방식은 수술 안전성을 높이는 데 기여하고, 사용자 경험을 향상시키는 데도 효과적입니다. 또한, 본 연구는 내시경 수술의 해부 구역 세분화 분야에서 비주얼 프롬프트 디자인을 통합한 첫 번째 연구로서, 향후 연구에 대한 기초를 마련합니다.



### A survey on cutting-edge relation extraction techniques based on language models (https://arxiv.org/abs/2411.18157)
Comments:
          50 pages, under review in Artificial Intelligence Review

- **What's New**: 이번 연구는 Relation Extraction (RE)에 관한 최신 발전을 종합적으로 조사하여, 137개의 ACL 회의를 통해 발표된 논문들을 분석합니다. 연구의 핵심은 언어 모델을 활용하여 RE 기술의 진화와 현황을 조명하는데 있습니다. BERT 기반의 방법들이 RE에서 가장 뛰어난 결과를 도출하는 데 주도적인 역할을 하며, T5와 같은 새로운 대형 언어 모델(LLMs)의 유망한 가능성도 주목받고 있습니다.

- **Technical Details**: RE는 텍스트 내에서 다양한 개체 간의 관계를 식별하고 추출하는 작업으로, 비구조적 데이터를 다루는 데 중점을 둡니다. 자연어 처리(NLP)의 한 분야로, RE는 Named Entity Recognition (NER), relation identification, relation classification의 세 가지 주요 구성 요소로 나뉩니다. 이 연구에서는 최근 몇 년간 발표된 최신 RE 기법들을 언어 모델 관점에서 분석하며, BERT와 RoBERTa와 같은 언어 모델의 다양한 활용을 검토하고 있습니다.

- **Performance Highlights**: ACL 회의에서 발표된 논문들을 바탕으로, 2020년 이후 언어 모델의 도입이 RE의 발전에 미친 영향을 면밀히 조사하였습니다. 최종적으로, 65개의 연구 기여 논문을 분석하였으며, 이들 논문은 언어 모델을 활용한 새로운 접근 방식을 포함합니다. TACRED, NYT10 및 DocRED와 같은 여러 데이터셋이 평가 기준으로 사용되었으며, 이러한 분석은 RE의 다양한 도메인과 관련된 중요한 통찰을 제공합니다.



### Predicting Water Quality using Quantum Machine Learning: The Case of the Umgeni Catchment (U20A) Study Region (https://arxiv.org/abs/2411.18141)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 연구에서는 남아프리카 더반 U20A 지역의 수질 평가를 다루고 있으며, 양자 기계 학습(Quantum Machine Learning, QML) 기법을 적용했습니다. 특히, 양자 지원 벡터 분류기(Quantum Support Vector Classifier, QSVC)와 양자 신경망(Quantum Neural Network, QNN)을 사용하였고, QSVC가 쉽게 구현 가능하며 높은 정확도를 제공함을 보여주었습니다.

- **Technical Details**: QSVC 모델은 선형, 다항식, 및 방사형 기저 함수(Radial Basis Function, RBF) 커널을 적용하여 다항식과 RBF 커널이 동일한 성능을 나타내는 것을 확인했습니다. 반면, QNN 모델은 최적화기, 학습률, 회로 구성 요소의 노이즈, 가중치 초기화 등을 고려하였지만, 지속적으로 '죽은 뉴런 문제(dead neuron problem)'에 부딪혔습니다.

- **Performance Highlights**: QSVC는 정확도와 손실 측면에서 QNN과 비교하였으며, Adam 최적화기를 사용할 때 가장 좋은 성능을 나타냈으나, 여전히 QSVC보다는 낮은 성능을 보였습니다. 이는 QML 모델이 전통적 방법들에 비해 더 적은 훈련 매개변수와 계산 자원으로도 유사하거나 우수한 정확도를 달성할 수 있음을 시사합니다.



### SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation (https://arxiv.org/abs/2411.18138)
Comments:
          Technical report

- **What's New**: SALMONN-omni는 코드가 없는 풀 듀플렉스(Full-duplex) 대화 모델로, 음성을 이해하고 생성할 수 있는 새로운 프레임워크를 제공합니다. 이 모델은 생성된 음성과 배경 소리를 동시에 듣고 이야기하는 능력을 갖추고 있어 인간과 기계 간의 자연스러운 대화를 구현합니다. 기존 모듈화된 시스템과는 달리 SALMONN-omni는 단일 엔드-투-엔드 모델로 설계되어 오류 전파를 제거하였습니다.

- **Technical Details**: SALMONN-omni는 스트리밍 음성 인코더와 대형 언어 모델, 스트리밍 음성 합성기를 통합하여 실시간으로 입력 및 출력 음성을 처리할 수 있습니다. 모델 내에 주기적인 동기화 메커니즘이 도입되어 음향과 텍스트 모달리티의 정렬을 보장합니다. SALMONN-omni는 대화의 동적 상황을 효과적으로 다룰 수 있는 "생각" 메커니즘을 특징으로 하며, 이는 두 개의 특별한 상태 전환 토큰을 활용합니다.

- **Performance Highlights**: 실험 결과, SALMONN-omni는 음성 인식, 음성 향상 및 대화형 질문 응답을 포함한 다양한 음성 작업에서 높은 성능을 발휘하였습니다. 이 모델은 턴-테이킹(turn-taking), 바지(in) 및 에코 캔슬레이션(echo cancellation) 상황을 관리하는 데 뛰어난 성능을 보여줍니다. SALMONN-omni는 코드 없는 최초의 풀 듀플렉스 대화 AI 시스템으로, 향후 다양한 응용 가능한 가능성을 제시합니다.



### Training and Evaluating Language Models with Template-based Data Generation (https://arxiv.org/abs/2411.18104)
Comments:
          8 pages, 2 figures

- **What's New**: 최근의 대형 언어 모델(LLMs) 발전은 자연어 처리(NLP) 분야에서 큰 변화를 가져왔습니다. 하지만 이러한 모델들은 복잡한 추론이 필요한 작업, 특히 수학 문제 해결에서는 어려움을 겪고 있습니다. 이를 극복하기 위해 Template-based Data Generation (TDG) 방식을 도입하여 LLMs(GPT-4)를 활용해 매개변수화된 메타 템플릿을 자동으로 생성하고, 700만 개 이상의 고품질 수학 문제 및 해결책을 포함한 TemplateMath Part I: TemplateGSM 데이터를 구축했습니다.

- **Technical Details**: TDG는 매개변수화된 템플릿을 기반으로 광범위한 수학 문제와 그 해결책을 시스템적으로 생성하는 방법입니다. GPT-4를 사용하여 생성된 이러한 메타 템플릿은 다양한 문제 구조와 언어 스타일을 캡처하도록 설계되었습니다. 생성된 문제와 해결책은 코드 실행 및 LLM 검증을 통해 정확성이 보장되고, 검증 과정을 통해 높은 품질의 데이터가 확보되도록 순환적으로 진행됩니다.

- **Performance Highlights**: TemplateGSM 데이터 세트는 700만 개 이상의 Grade School 수준의 수학 문제로 구성되어 있으며, 각 문제에는 코드 기반 해결책과 자연어 설명이 포함되어 있습니다. 제안된 TDG 방법 덕분에 수학 문제에 대한 데이터의 양과 품질이 대폭 향상되었고, 다양한 문제 유형 및 난이도에 대한 학습이 가능해졌습니다. 이러한 데이터 세트는 LLMs의 수학적 추론 능력 향상에 기여할 것으로 기대됩니다.



### Derivation of Closed Form of Expected Improvement for Gaussian Process Trained on Log-Transformed Objectiv (https://arxiv.org/abs/2411.18095)
- **What's New**: 이 논문은 Bayesian optimization에서 가장 널리 사용되는 Acquisition Function인 Expected Improvement (EI)의 새로운 접근 방식을 제시합니다. Hutter et al. (2009)이 논의했던 Gaussian Process를 사용한 log-transformed objective function의 이점에 대해 친절하게 유도하였습니다. 이전에는 해당 EI의 중간 유도가 수행되지 않았던 점에 주목하고 있습니다.

- **Technical Details**: 논문에서는 EI의 중간 유도를 제공하여 Bayesian optimization에서의 성능 향상을 도모하고 있습니다. Gaussian Process (GP)를 log 변환된 목적 함수에 대해 훈련시키는 방법이 핵심입니다. 이 접근법은 GP의 예측 정확도를 향상시키는 데 기여하며, 이는 EI의 효과성을 높입니다.

- **Performance Highlights**: Hutter et al.의 방법을 통해 EI의 성능이 현저하게 개선되었으며, 이는 다양한 실험에서 입증되었습니다. 이러한 방식은 적절한 Numerics와 예측 정확도를 달성하는 데 성공하였고, Bayesian optimization의 실용성을 더욱 높였습니다.



### From Exploration to Revelation: Detecting Dark Patterns in Mobile Apps (https://arxiv.org/abs/2411.18084)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 모바일 애플리케이션에서의 '다크 패턴(dark patterns)' 탐지를 위한 새로운 시스템인 AppRay를 제안합니다. AppRay는 자동화된 탐지 방법과 사용자 인터페이스(UI) 탐색을 통합해, 수작업 탐지의 한계를 극복합니다. 이를 통해 사용자가 경험할 수 있는 다양한 심리적 트릭을 효과적으로 찾아내는 것을 목표로 하고 있습니다. 이 시스템은 다이내믹한 패턴과 정적인 패턴 모두를 탐지할 수 있도록 설계되었습니다.

- **Technical Details**: AppRay는 두 개의 주요 요소로 구성됩니다: 첫째, LLM(large language model)을 활용한 앱 탐색 모듈이 있으며, 이는 고전적인 자동화된 탐색 도구와 결합되어 다양한 UI 상태를 포착합니다. 둘째, 대비 학습(contrastive learning) 기반의 멀티 레이블 분류기와 규칙 기반 리파이너(rule-based refiner)를 포함하는 어두운 패턴 탐지기가 구현되어 있습니다. 이러한 두 요소는 앱의 동적 및 정적 UI 관계를 유지하며 다크 패턴을 효과적으로 탐지하도록 돕습니다.

- **Performance Highlights**: 실험 결과, AppRay는 정적 및 동적 다크 패턴을 포함하여 다양한 다크 패턴을 탐지하는 데 뛰어난 성능을 보여줍니다. 시스템은 매크로 및 마이크로 평균 F1 점수에서 각각 0.76과 0.62를 기록하였으며, 정밀도와 재현율 또한 0.77/0.65 및 0.76/0.62의 높은 성능을 달성했습니다. 또한 사용자를 대상으로 한 연구에서도 AppRay의 유용성을 확인했습니다.



### PersonaCraft: Personalized Full-Body Image Synthesis for Multiple Identities from Single References Using 3D-Model-Conditioned Diffusion (https://arxiv.org/abs/2411.18068)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 PersonaCraft라는 새로운 접근법을 제안하며, 이는 diffusion 모델과 3D 인간 모델링을 결합하여 개인 맞춤형 이미지 생성의 한계를 극복합니다. PersonaCraft는 복잡한 장면에서 여러 사람의 이미지를 생성하는 데 특히 효과적이며, occlusion(가림 현상)을 효율적으로 관리합니다. 이를 통해 사용자가 자신의 몸매를 조정할 수 있는 유연성을 제공하여 개인적인 체형 맞춤화를 가능하게 합니다.

- **Technical Details**: PersonaCraft는 SMPLx 모델을 활용하고, SMPLx-ControlNet(SCNet)을 통한 3D-aware pose conditioning을 사용하여, 신체 형태와 자세에 대한 정밀한 제어를 제공합니다. 이 접근법은 얼굴 및 신체 아이덴티티 추출, 3D 포즈 조정, 다중 인물 개인화 이미지 합성을 포함한 세 가지 핵심 메커니즘으로 구성됩니다. 특히 SCNet은 복잡한 오클루전이 있는 멀티 휴먼 장면에서 강력한 제어를 가능하게 합니다.

- **Performance Highlights**: PersonaCraft는 정량적 및 정성적 평가를 통해 얼굴 아이덴티티, 신체 형태 및 자연스러운 인체 해부학을 잘 보존하며, 다중 인물이 포함된 복잡한 시나리오에서도 우수한 성능을 보여줍니다. 기존 방법들과 비교하여, 정확도와 개인화 면에서 뛰어난 성과를 이루어 고품질의 다중 인물 이미지 생성의 새로운 기준을 제시합니다.



### RL for Mitigating Cascading Failures: Targeted Exploration via Sensitivity Factors (https://arxiv.org/abs/2411.18050)
- **What's New**: 본 논문은 전력망의 복원력을 강화하기 위한 물리 기반의 머신러닝 프레임워크인 PG-RL(Physics-Guided Reinforcement Learning)을 제안합니다. 이 프레임워크는 재난 상황에서 블랙아웃을 방지하기 위해 실시간으로 효과적인 remedial control actions를 결정합니다. 전력 흐름 감도 요인을 활용하여 RL 학습 과정 중 최적의 정책을 탐색하도록 설계되었습니다. 이를 통해 전력망의 자원 활용을 극대화하고 기후 변화에 효과적으로 대응할 수 있습니다.

- **Technical Details**: PG-RL 프레임워크는 MDP(마르코프 결정 프로세스)의 상태 및 행동 공간의 기본 구조를 활용하여 보조 도메인 지식을 통합합니다. 주요 특징으로는 물리 기반 탐색 정책 설계가 있으며, 이는 복잡한 전력망에서의 전환 작업을 관리하는 데 있어 필수적입니다. 네트워크 토폴로지의 변화와 실시간 remedial actions를 평가하기 위해 Grid2Op 플랫폼을 사용하여 체계적인 실험이 이루어졌습니다. 이러한 차별화된 접근 방식은 기존의 블랙박스 RL 알고리즘보다 더 우수한 성능을 입증하였습니다.

- **Performance Highlights**: Grid2Op 플랫폼에서의 종합적인 평가 결과, PG-RL은 전통적인 RL 방법보다 에너지망의 자원 활용과 블랙아웃 완화 정책에서 월등한 성과를 보였습니다. 실험에서 PG-RL은 가장 안전하고 효과적인 전선 전환 결정으로 시스템의 신뢰성을 높였습니다. 이러한 성과는 기후 변화에 따른 전력망의 위험을 줄이는 데 중요한 역할을 할 것으로 기대됩니다.



### Heterogeneous Relationships of Subjects and Shapelets for Semi-supervised Multivariate Series Classification (https://arxiv.org/abs/2411.18043)
Comments:
          Submitted to IEEE International Conference on Data Engineering (ICDE) 2025

- **What's New**: 이번 연구에서는 다변량 시계열(MTS) 분류를 위한 새로운 방법론으로서 이질적인 관계의 주체와 형태체(shapelet) 방법을 제안합니다. 이 방법은 다양한 추가 정보를 통합하면서 서로 간의 관계를 포착하는 새로운 접근 방식을 제공합니다. 특히, 대비 시간적 자기 주의 모듈을 활용하여 희소 MTS 표현을 얻고, 이 표현들 사이의 유사성을 모델링하여 이질적인 그래프를 생성합니다.

- **Technical Details**: 제안된 방법은 대비 시간적 자기 주의 모듈을 사용하여 MTS의 희소 표현을 얻고, 이를 통해 소프트 동적 시간 왜곡(soft DTW) 기법으로 유사성 그래프를 구축합니다. 이어서, 각 주체 유형에 대한 형태체를 학습하고 이 정보를 추가하여 유사성 그래프를 정제합니다. 최종적으로 이중 레벨 그래프 주의 네트워크를 포함하여 예측을 수행하며, 이 과정에서 다중 유형의 정보를有效하게 통합합니다.

- **Performance Highlights**: 실험을 통해 인간 활동 인식(Human Activity Recognition) 및 수면 단계 분류와 같은 다양한 데이터셋에서 제안된 방법이 현재 최첨단 방법들보다 우수한 성능을 보임을 확인했습니다. 구체적으로, 제안된 방법은 정확한 반지도 노드 분류를 달성했으며 MTS 분류 작업에서의 우수성을 입증하였습니다.



### VLM-HOI: Vision Language Models for Interpretable Human-Object Interaction Analysis (https://arxiv.org/abs/2411.18038)
Comments:
          18 pages

- **What's New**: 본 논문에서는 Large Vision Language Model (VLM)을 Human-Object Interaction (HOI) 탐지 작업에 활용하는 새로운 접근법(VLM-HOI)을 제안합니다. VLM의 언어 이해 능력을 활용하여 예상되는 HOI triplet의 유사성을 정량화하는 방법을 도입합니다. 이 접근법은 이미지-텍스트 매칭(Image-Text matching) 기술을 사용하여 HOI 탐지의 정확성을 향상시키며, 이는 기존 CLIP 모델보다 성능이 뛰어납니다.

- **Technical Details**: VLM-HOI에서는 HOI triplet을 언어적으로 표현하여 VLM의 언어 이해 능력을 최대한 활용합니다. 예측된 HOI triplet의 유사성은 이미지-텍스트 매칭 기법을 통해 계산됩니다. 우리의 방법론은 VLM의 지식을 명확히 분류하는 데 초점을 두며, 대조 학습(constrastive learning) 프레임워크를 적용하여 신경망이 텍스트 형태의 HOI를 이해할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 우리의 방법이 기존 방법에 비해 높은 정확도와 강 robustness를 달성했다는 것을 보여주었습니다. 우리는 HOI 탐지 벤치마크에서 최첨단 결과를 기록하며, VLM을 HOI 탐지에 통합하는 것이 인간-객체 상호작용의 보다 발전된 해석적 분석을 위한 중요한 진전을 나타낸다고 믿습니다.



### AEGIS: An Agent-based Framework for General Bug Reproduction from Issue Descriptions (https://arxiv.org/abs/2411.18015)
- **What's New**: 본 논문은 소프트웨어 유지보수에서 버그 재현의 중요성을 강조하며, 이를 자동화하기 위한 agent-based 프레임워크인 AEGIS를 제안합니다. 기존 연구는 특정 버그 유형에 국한되어 있었으나, AEGIS는 일반적인 버그 재현을 위한 첫 번째 접근을 제공합니다. 이 프레임워크는 코드 에이전트가 구조화된 정보를 추출하고, 효율적인 스크립트 생성을 지원하는 두 가지 주요 모듈로 구성됩니다.

- **Technical Details**: AEGIS는 (1) 간결한 컨텍스트 구축 모듈과 (2) FSM 기반 다중 피드백 최적화 모듈로 나뉩니다. 간결한 컨텍스트 구축 모듈은 문제 설명에서 구조화된 정보를 추출하여 효율적으로 문제 관련 코드를 식별하고, 이를 통합하여 컨텍스트를 생성합니다. FSM 기반 모듈은 코드 에이전트의 동작을 규제하여 제어된 스크립트 생성 과정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, AEGIS는 F->P 메트릭에서 최첨단 기준보다 23.0% 향상된 성과를 보였습니다. 또한, AEGIS가 생성한 버그 재현 스크립트는 Agentless의 상대적 해결율을 12.5% 개선하는 것으로 나타났습니다.



### Causal and Local Correlations Based Network for Multivariate Time Series Classification (https://arxiv.org/abs/2411.18008)
Comments:
          Submitted on April 03, 2023; major revisions on March 25, 2024; minor revisions on July 9, 2024

- **What's New**: 이번 논문에서는 CaLoNet라는 새로운 다변량 시계열 분류 네트워크를 제안합니다. 이 네트워크는 차원 간 인과적(causal) 상관관계를 모델링하여 그래프 구조(graph structure)를 생성하고, 이후 이 구조에 기반하여 지역적(local) 상관관계를 융합하여 장기 의존(long-term dependency) 특성을 추출합니다. 마지막으로, 생성된 그래프 구조와 장기 의존 특성을 그래프 신경망(graph neural network)에 통합하여 시계열 데이터를 보다 효과적으로 분류합니다.

- **Technical Details**: CaLoNet는 다차원 시계열 데이터에서 정보 손실을 방지하기 위해 인과관계를 통해 차원 간의 상관관계를 모델링합니다. 여기서는 transfer entropy를 사용하여 인과 정보를 특성화하고, 그래프 레벨의 분류 작업을 위해 새로운 형태의 시계열 표현을 제안합니다. 또한, 관계 추출 네트워크(relationship extraction network)를 활용하여 지역적 상관관계를 통합, 장기 의존 특성을 추출하며, 최종적으로 이러한 요소들이 그래프 신경망에 통합됩니다.

- **Performance Highlights**: UEA 데이터셋을 활용한 실험 결과, CaLoNet는 최신 기술(state-of-the-art)들과 비교하여 경쟁력 있는 성능을 보여줍니다. 이번 연구는 멀티 변량 시간 연속성을 잘 활용해 더 나은 예측 모델을 세울 수 있는 가능성을 밝혀냈습니다. 실험 데이터를 바탕으로 제안된 접근법의 유효성을 입증하였으며, 다양한 영역에서의 응용 가능성을 시사합니다.



### HAAT: Hybrid Attention Aggregation Transformer for Image Super-Resolution (https://arxiv.org/abs/2411.18003)
Comments:
          6 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 Hybrid Attention Aggregation Transformer (HAAT) 모델을 제안하여 기존의 Swin-transformer 기반 모델들이 해결하지 못한 문제를 다루고 있습니다. HAAT는 Swin-Dense-Residual-Connected Blocks (SDRCB)와 Hybrid Grid Attention Blocks (HGAB)을 결합하여 특징 정보를 더욱 효과적으로 활용합니다. 이는 정보 처리의 비효율성을 감소시키고, 채널 간 유용한 정보를 놓치지 않도록 개선된 구조를 제공합니다.

- **Technical Details**: HAAT는 Swin Transformer의 변경된 구조를 기반으로 하여, 각 Residual Deep feature extraction Group (RDG) 내에서의 수용 필드를 확장하며, 보다 효율적인 구성으로 성능을 향상시킵니다. HGAB는 채널 어텐션, 희소 어텐션 및 윈도우 어텐션을 결합하여 비지역적 특징 융합을 개선하고, 고급 시각적 결과물을 생성합니다. 또한, 희소 자기 주의 메커니즘을 통해 전역적 특징 상호작용을 증가시키면서도 계산 복잡성을 관리합니다.

- **Performance Highlights**: HAAT는 DF2K 데이터셋을 활용하여 학습되었으며, Set5 및 Set14와 같은 널리 알려진 SISR 벤치마크 데이터셋에서 성능 평가를 실시한 결과, 기존의 첨단 방법들을 초월하는 성능을 나타냈습니다. 이 모델은 더 나은 이미지 복원 결과를 생성하며, 긴 거리 의존성을 잘 처리하여 전반적인 성능을 향상시키는 것을 입증했습니다.



### An End-to-End Two-Stream Network Based on RGB Flow and Representation Flow for Human Action Recognition (https://arxiv.org/abs/2411.18002)
Comments:
          6 pages, 3 figures, 9 tables

- **What's New**: 이번 연구에서는 딥러닝의 발전을 바탕으로 비디오 기반 동작 인식을 위한 두 개의 스트림 신경망(two-stream neural networks) 모델에서 광학 흐름(optical flow) 대신 표현 흐름(representation flow) 알고리즘을 도입했습니다. 이를 통해 egocentric action recognition 모델을 위한 최적화된 엔드 투 엔드(end-to-end) 훈련을 지원하며, 계산 비용과 예측 시간을 줄일 수 있었습니다.

- **Technical Details**: 모델은 클래스 활성화 맵(class activation maps, CAMs)을 적용하여 정확성을 개선하고, 시공간(spatio-temporal) 인코딩을 위한 ConvLSTM을 활용하여 공간주의(spatial attention)를 적용합니다. GTEA61, EGTEA GAZE+, HMDB 데이터셋에서 평가했을 때, 제안한 모델은 GTEA61에서 원래 모델의 정확도와 일치하고, EGTEA GAZE+와 HMDB에서 각각 0.65%와 0.84% 향상된 성능을 보였습니다.

- **Performance Highlights**: 예측 런타임(prediction runtimes)은 기존 모델에 비해 현저히 감소하여 GTEA61, EGTEA GAZE+, HMDB에서 각각 0.1881초, 0.1503초, 0.1459초를 기록했습니다. 이는 기존 모델의 101.6795초, 25.3799초, 203.9958초와 비교했을 때 매우 인상적인 성능 향상으로, 실질적인 적용 가능성을 시사합니다.



### Regularized Multi-LLMs Collaboration for Enhanced Score-based Causal Discovery (https://arxiv.org/abs/2411.17989)
- **What's New**: 본 연구에서는 여러 개의 Large Language Models (LLMs)를 통합하여 인과 발견(causal discovery) 방법을 개선하고자 합니다. 최근 LLM의 성공적인 성과가 전문가 지식을 대체할 수 있는 기회로 부각되고 있습니다. 저자들은 기존 연구들이 단일 LLM을 활용하는 것에 반해, 복합적인 LLM 통합을 통해 더 신뢰할 수 있는 결과를 만들 수 있는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 LLM에서 유도된 정보를 기존 스코어 함수(score function)와 결합하여 인과 구조 학습(causal structure learning) 문제를 해결하려 합니다. 여기서 사용된 스코어 기반 방법론은 GES(Greedy Equivalence Search) 및 NOTEARS와 같은 알고리즘에 LLM에서 얻은 정보를 추가적인 패널티(penalty) 항으로 통합합니다. 또한, 새로운 DAG(Directed Acyclic Graph) G(V,E)에서 인과 관계를 나타내는 방법을 정의합니다.

- **Performance Highlights**: 제안된 프레임워크는 GES, NOTEARS, KCRL 등 두 스코어 기반 방법 및 강화 학습을 통한 인과 발견 알고리즘의 실험을 통해 입증되었습니다. 결과적으로, 복합적인 LLM을 활용한 접근 방식이 기존 알고리즘의 정확성과 효율성을 크게 향상시킨 것을 확인하였습니다. 향후, LLM의 가능성을 활용한 인과 발견의 진전을 더욱 기대할 수 있습니다.



### Optimized Conformal Selection: Powerful Selective Inference After Conformity Score Optimization (https://arxiv.org/abs/2411.17983)
- **What's New**: 이 논문에서는 데이터 드리븐 모델 최적화 이후에도 합법적인 통계적 테스트를 가능하게 하는 OptCS라는 새로운 프레임워크를 제안합니다. 기존에 사용되는 모델 선택 방법이 통계적 타당성을 유지하며 여러 모델 선택 기준으로 FDR(위양성 비율)을 제어할 수 있도록 합니다. 새로운 다중 테스트 절차를 통해 복잡한 p-value 의존성을 처리하고 유연한 다중 모델 최적화를 가능하게 하는 조건을 도입했습니다.

- **Technical Details**: OptCS 프레임워크는 데이터 재사용과 복잡한 p-value 종속성을 처리하는 일반적인 조건을 바탕으로 합니다. 이 연구는 데이터를 분할하지 않고 모든 데이터를 활용하거나 후보 모델 간에 가장 강력한 모델을 선택하는 세 가지 FDR 제어 절차를 제안합니다. 이러한 방법은 약물 발견 및 방사선 보고서 생성에서 대규모 언어 모델을 정렬하는 데 실질적인 적용 사례를 가집니다.

- **Performance Highlights**: 논문에서 제안하는 메서드는 시뮬레이션 연구와 실제 약물 발견 애플리케이션에서 그 효과성을 입증받았습니다. 제안된 방법은 다양한 사전 훈련된 모델을 사용하여 높은 신뢰도와 정밀도를 유지하면서도 선택의 오류를 최소화합니다. 여러 실제 사례를 통해 이 메서드가 적절한 성능을 발휘함을 보여주었습니다.



### The importance of visual modelling languages in generative software engineering (https://arxiv.org/abs/2411.17976)
Comments:
          9 pages, working paper

- **What's New**: 이 논문에서는 소프트웨어 엔지니어링(SE)과 생성 인공지능(GenAI) 간의 상호작용의 새로운 가능성을 탐구합니다. 특히 GPT-4는 이미지와 텍스트 입력을 동시에 수용하며, 이는 다이어그램과 자연어의 혼합으로 프롬프트를 생성할 수 있는 능력을 의미합니다. 저자들은 이러한 멀티모달 GPT의 활용 사례를 조사하여 SE 작업에서 유용한 잠재력을 강조합니다. 이와 같은 주제가 기존 연구에서는 다루어지지 않았던 점에서 특별한 의의가 있습니다.

- **Technical Details**: 소프트웨어 엔지니어링의 배경을 설명하면서, Agile 방법론과 Unified Modeling Language(UML)의 발전에 대해 다룹니다. UML은 소프트웨어 설계의 표준 시각 언어로, 사용 사례 개발, 정적 분석, 동적 분석 세 분야에서 활용됩니다. 그러나 최근 연구는 대부분의 소프트웨어 개발자들이 비공식적인 손으로 그린 다이어그램을 선호한다는 점을 지적하며, GenAI가 이러한 비공식적 스케치를 효과적으로 활용할 수 있는 도구가 될 가능성을 제시합니다. GenAI는 새로운 콘텐츠를 생성하는 AI 모델로서, 대형 언어 모델(LLMs) 기반으로 작동하며 GPT-4와 같은 최신 멀티모달 GPT의 경우 이미지와 텍스트 입력을 모두 처리할 수 있습니다.

- **Performance Highlights**: 이 논문은 multimodal GPT의 사용이 SE 작업 수행에 있어 어떻게 기여할 수 있는지를 보여줍니다. 특히 다이어그램을 활용한 프로세스 개선과 코드 자동 생성의 가능성을 제시하며, 이는 시간과 비용을 절감하고 결과의 품질을 향상시킬 수 있습니다. 저자들은 기존 SE 작업에서의 연구 공백을 언급하며, 이러한 사용 사례가 Generative Software Engineering(GenSE) 연구에 중요한 기여를 할 것이라고 전합니다. 전반적으로, 이 연구는 SE와 GenAI의 융합이 가져올 혁신적 변화를 예고합니다.



### Improved implicit diffusion model with knowledge distillation to estimate the spatial distribution density of carbon stock in remote sensing imagery (https://arxiv.org/abs/2411.17973)
Comments:
          Under review

- **What's New**: 본 연구는 중국 유난성 구징시 후이즈 카운티를 중심으로 GF-1 WFV 위성 이미지를 활용하여 숲의 탄소 저장량을 추정하는 데 혁신적인 접근 방식을 제안합니다. VGG와 UNet 모델을 기반으로 한 KD-VGG 및 KD-UNet 모듈을 통해 초기 특징을 추출하고, 개선된 암시적 확산 모델(IIDM)을 도입하여 정확성을 높였습니다. 본 연구의 결과는 새로운 AI 생성 내용(AIGC) 활용 가능성을 보여줍니다.

- **Technical Details**: 자세히 살펴보면, VGG 모듈은 초기 특징 추출을 개선하고, 모델 파라미터 최적화를 통해 정확성과 추론 시간을 단축했습니다. Cross-attention과 MLP의 결합은 지방 및 글로벌 특징 간의 관계를 효과적으로 포착하여 고정확도의 탄소 저장량 추정을 달성했습니다. IIDM 모델은 12.17%의 RMSE를 기록하였으며, 회귀 모델에 비해 41.69%에서 42.33%로 개선되었습니다.

- **Performance Highlights**: 결과적으로 본 연구에서 사용된 16미터 해상도의 추정치는 지역 탄소 저장량 관리를 위한 강력한 기초 자료를 제공합니다. 동시에, 개선된 암시적 확산 모델은 탄소 저장량 추정 시 다른 모델에 비해 깊은 특징 추출에 뛰어난 것으로 평가되었습니다. 이러한 성과는 숲 탄소 저장량 규제 및 의사결정 지원에 중요한 이론적 토대를 마련해 줍니다.



### Graph Neural Network for Cerebral Blood Flow Prediction With Clinical Datasets (https://arxiv.org/abs/2411.17971)
Comments:
          4 pages, 3 figures

- **What's New**: 이 논문에서는 혈관 구조를 포함하지 않는 새로운 대뇌 혈관 네트워크에서 혈류(혈류)와 압력을 예측하기 위해 그래프 신경망(Graphic Neural Network, GNN)을 제안합니다. 기존의 전통적인 방법과 달리, 이 GNN은 입력 데이터의 복잡성을 다루며 실시간 임상 응용프로그램에서의 실용성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: GNN은 환자의 협착(stenosis) 케이스에서 수집된 임상 데이터를 사용하여 복잡한 혈관 지오메트리를 포함한 모델을 훈련시킵니다. 또한, 데이터는 다양한 유입 조건(inflow conditions), 혈관의 형상(vessel topologies) 및 연결성(network connectivities)을 포함하여 일반화 능력을 강화합니다. 노드와 엣지에 대한 정보가 포함된 네트워크 그래프(Network Graph)가 생성되며, Poiseuille 흐름 모델을 사용해 혈류를 계산하는 데 필요한 수학적 모델이 사용됩니다.

- **Performance Highlights**: 이 GNN은 혈압에 대해 0.727, 혈류량에 대해 0.824의 Pearson 상관계수(Pearson's correlation coefficient)를 기록하였으며, 이는 충분한 훈련 데이터에서 달성되었습니다. 이러한 성과는 GNN이 복잡하고 병리적인 혈관 네트워크를 다루는 데 있어 실시간 대뇌혈관 진단에 대한 잠재력을 나타냅니다.



### MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation (https://arxiv.org/abs/2411.17945)
- **What's New**: 이번 연구에서는 MARVEL-40M+라는 4천만 개의 텍스트 주석을 포함한 대규모 데이터세트를 소개합니다. 이 데이터세트는 890만 개의 3D 자산을 포괄하며, 7개의 주요 3D 데이터세트에서 집계되었습니다. 또한, 새로운 다단계 주석 파이프라인을 통해 자동으로 고급 및 간결한 설명을 생성하여 정밀한 3D 재구성과 빠른 프로토타입 생성에 기여합니다.

- **Technical Details**: MARVEL의 플랫폼은 개방형 소스의 멀티뷰 VLMs(Visual Language Models)와 LLMs(Large Language Models)를 활용하여 다단계 주석을 생성하는 데 초점을 맞추고 있습니다. 이 파이프라인은 전체 재구성을 위한 세부 설명(150-200 단어)에서 빠른 모델링을 위한 간단한 태그(10-20 단어)까지 다양한 수준의 주석을 제공합니다. 또한, 소스 데이터세트에서의 인간 메타데이터를 통합하여 VLM의 허상 문제를 감소시킵니다.

- **Performance Highlights**: MARVEL-40M+ 데이터세트는 GPT-4와 인간 평가자에 의해 각각 72.41%와 73.40%의 승률을 기록하며 기존 데이터세트보다 훨씬 우수한 주석 품질과 언어 다양성을 보여주었습니다. MARVEL-FX3D라는 두 단계의 텍스트-3D 생성 파이프라인은 텍스트를 통해 15초 내에 텍스처를 가진 메쉬를 생성하는 데 성공했습니다. 이 연구는 높은 충실도의 TT3D 생성에서 현 상태의 방법들을 능가하는 성능을 입증하였습니다.



### Evaluating Generative AI-Enhanced Content: A Conceptual Framework Using Qualitative, Quantitative, and Mixed-Methods Approaches (https://arxiv.org/abs/2411.17943)
- **What's New**: Generative AI(GenAI)가 콘텐츠 생성에 혁신적인 변화를 가져왔습니다. 이 논문은 GenAI 모델이 과학적 글쓰기 향상에 미치는 영향을 평가하기 위한 다양한 연구 방법론을 고찰합니다. 정성적, 정량적, 혼합 방법을 활용하여 GenAI의 성능을 체계적으로 평가하며, 각 방법이 제공하는 독창적인 통찰력을 강조합니다.

- **Technical Details**: 정성적 연구에서는 전문가 리뷰어로부터 심층 피드백을 수집하고 주제 분석 도구를 통해 개선점을 분석합니다. 정량적 방법은 BLEU, ROUGE와 같은 자동화된 메트릭 및 사용자 조사를 통해 언어적 일관성, 유창성 및 구조의 개선을 객관적으로 측정합니다. 혼합 방법론은 통계적 평가와 상세한 정성적 통찰력을 통합하여 GenAI로 생성된 콘텐츠의 포괄적인 평가를 가능하게 합니다.

- **Performance Highlights**: 이 연구에서는 GenAI로 생성된 콘텐츠의 품질과 기술적 정확성을 계량화하는 방법을 제시하여 기존 편집 프로세스와의 비교를 통해 평가할 수 있는 강력한 프레임워크를 제공합니다. 이러한 방법론을 활용하여 연구자들은 GenAI의 성능 향상을 평가하고, 그 활용을 개선하며, 의료 및 과학 연구와 같은 고위험 영역에서 책임감 있게 채택하도록 안내할 수 있습니다.



### Spatio-temporal Causal Learning for Streamflow Forecasting (https://arxiv.org/abs/2411.17937)
Comments:
          To be published at IEEE Big Data 2024

- **What's New**: 이 연구는 하천 유량 예측에 대한 새로운 접근 방식을 제안하는데, 인과적 구조를 학습하는데 있어 하천 흐름 그래프를 사전 지식으로 활용합니다. 이 모델은 Causal Streamflow Forecasting (CSF)이라 불리며, 기상 강제 변수와 유출 간의 복잡한 관계를 포착하여 보다 정확한 예측을 달성합니다. 또한, 기존의 공간-시간 그래프 신경망(STGNN)보다 뛰어난 성능과 효율성을 보여줍니다.

- **Technical Details**: CSF 모델은 물리적 원리에 기반한 하천 흐름 그래프를 활용하여 시공간 그래프 합성곱 신경망(STGCN)을 안내합니다. 이를 통해 물의 흐름 방향성을 고려하고, 대규모 수문학적 시스템의 공간-시간 의존성을 효율적으로 처리할 수 있도록 계층적 네트워크 구조를 통합하였습니다. 또한, 변분 오토인코더(VAE)를 사용하여 유출 임베딩을 학습하여 국소적인 수문 과정의 효율적이고 확장 가능한 표현을 제공합니다.

- **Performance Highlights**: Brazos 강 유역의 실제 데이터를 통해 CSF 모델을 평가한 결과, 단기, 중기 및 장기 예측 작업에서 기존 모델보다 일관되게 더 우수한 성능을 보였습니다. 이 연구는 첨단 신경망 기술과 도메인 특화 지식을 결합함으로써 수문 모델링의 성능 향상을 위한 가능성을 보여줍니다. CSF는 수자원 관리 및 홍수 예측에 있어 실질적인 응용 가능성을 입증했습니다.



### Neural Networks Use Distance Metrics (https://arxiv.org/abs/2411.17932)
Comments:
          8 pages excluding references and appendix. 12 pages total. 3 figures. The code for the experiments in this paper is available at this https URL

- **What's New**: 이번 연구에서는 ReLU 및 절대값(Absolute Value) 활성 함수가 적용된 신경망이 거리 기반(distance-based) 표현을 학습한다는 실증적 증거를 제시합니다. 모델에서의 내부 활성 상태의 거리 및 강도(intensity) 속성을 독립적으로 조작하여 이들 아키텍처가 거리 기반의 작은 변동에 매우 민감하면서도 큰 강도 기반 변동 하에서 견고한 성능을 유지함을 발견했습니다. 이러한 발견들은 신경망 활성화의 기존의 강도 기반(intensity-based) 해석에 도전하며, 신경망의 학습 및 의사 결정 과정에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 이 연구는 신경망이 거리 메트릭을 자연스럽게 계산할 수 있는지 알아보기 위해, MNIST 데이터셋을 사용하여 학습된 모델의 활성화를 조작하는 체계적인 변동 분석을 수행했습니다. 실험에서는 ReLU 및 절대값 활성화 함수를 사용하여 모델이 거리와 강도 속성의 변화를 어떻게 반응하는지를 분석합니다. 감지 기능이 실질적으로 어떤 속성(거리 또는 강도)에 의해 주도되는지를 파악하기 위해, 이들 두 가지 속성을 독립적으로 조작한 후 모델 성능에 미치는 영향을 조사합니다.

- **Performance Highlights**: 실험 결과, ReLU와 절대값 활성화 함수를 가지는 네트워크는 거리 기반 변동에 대해 높은 민감도를 보였으며, 강도 변동 하에서도 견고한 성능을 유지함을 보여주었습니다. 이는 신경망이 거리 기반 메트릭을 활용함을 지지하며, 이 연구의 이론적 프레임워크를 검증합니다. 따라서, 이러한 발견들은 신경망 아키텍처를 이해하고 개선하는 새로운 접근법을 제시합니다.



### Combining Threat Intelligence with IoT Scanning to Predict Cyber Attack (https://arxiv.org/abs/2411.17931)
Comments:
          8 pages, 6 figures, 2 tables. This manuscript has been submitted to Springer for review (Manuscript ID: PDSE-D-24-00163) and is under consideration. It has not yet been peer-reviewed or published. Researchers are welcome to read and build upon this work; please cite it appropriately. For questions or clarifications, feel free to contact me

- **What's New**: 이 연구는 Dark Web에서 해커의 웹사이트를 식별하고 IoT의 취약점을 예측하는 방법론을 제안합니다. 이를 통해 정보 과부하 문제를 해결하고 해커와 사이버 공격자를 효과적으로 분석할 수 있는 새로운 관점을 제공합니다. 또한, IoT(Internet of Things)의 발전에 따라 증가하는 연결된 장치의 데이터를 활용하여 사이버 보안 분야에 기여하고자 합니다.

- **Technical Details**: 제안된 방법론은 정보 수집(information collection), 분석(analysis), 시각화(visualization) 기술을 포함하고 있습니다. Dark Web 정보를 체계적으로 수집하고 이를 해커 활동 예측에 활용하는 데 중점을 두고 있습니다. 이러한 연구는 해커 관련 데이터의 구조적 분석과 IoT 장치의 특성을 활용하는 방식으로 진행됩니다.

- **Performance Highlights**: 이 연구의 기여는 사이버 보안 정책 수립과 정보 분석 연구에 중요한 기초 자료를 제공한다는 점입니다. 또한, IoT 생태계에서 발생할 수 있는 다양한 취약점(vulnerabilities)을 사전에 예측할 수 있는 가능성을 제시합니다. 연구 결과는 해커의 활동을 예측하고 보다 효과적인 보안 대응 전략 개발에 도움을 줄 것입니다.



### AI2T: Building Trustable AI Tutors by Interactively Teaching a Self-Aware Learning Agen (https://arxiv.org/abs/2411.17924)
- **What's New**: AI2T는 지능형 튜터링 시스템(ITSs)의 작성을 위해 인터랙티브하게 학습할 수 있는 AI입니다. 저자는 몇 가지 단계적 해결책을 제공하고 AI2T의 문제 해결 시도를 평가함으로써 AI2T를 튜터합니다. 20-30분의 상호작용 훈련만으로 AI2T는 강력한 단계적 해결 추적 규칙(model-tracing)을 유도할 수 있습니다.

- **Technical Details**: AI2T는 STAND라는 자기 인식 조건 학습 알고리즘을 사용하여 보이지 않는 문제 단계에서 올바르게 수행할 확신도를 정확하게 추정합니다. 이는 XGBoost와 같은 최첨단 방법보다 우수합니다. 사용자 연구에 따르면, 저자는 STAND의 확신 휴리스틱을 사용하여 AI2T가 올바르고 완전한 모델 추적 프로그램을 유도하기에 충분한 다양한 문제에 대해 훈련되었는지 평가할 수 있습니다.

- **Performance Highlights**: AI2T가 유도한 프로그램은 망상에 취약한 LLMs 및 이전의 튜터링 기반 저작 접근 방식보다 더 신뢰할 수 있습니다. AI2T는 계층적 규칙의 자기 인식을 통해 복잡한 ITSs를 위한 신뢰할 수 있는 데이터 효율적인 저작 방법을 제공하며, 이는 보통 1시간의 수업당 200-300시간의 프로그래밍을 요구합니다.



### Automating grapevine LAI features estimation with UAV imagery and machine learning (https://arxiv.org/abs/2411.17897)
Comments:
          Accepted in 2024 IEEE INTERNATIONAL WORKSHOP ON Metrology for Agriculture and Forestry

- **What's New**: 이 연구는 드론 이미지 데이터를 활용하여 포도 나무의 잎 면적 지수(Leaf Area Index, LAI)를 자동으로 추정하는 방법을 제시합니다. 기존의 전통적인 방법은 시간 소모적이고 파괴적이며 비용이 많이 드는 반면, 이 방법은 기계 학습 모델을 통해 속도와 효율성을 높입니다. 딥 러닝 기반의 특징 추출이 기존 방법보다 효과적이라는 결과를 보여줍니다.

- **Technical Details**: 연구에서 사용된 데이터 세트는 드론으로 촬용한 개별 포도 나무 이미지와 그에 해당하는 LAI 값을 포함하고 있습니다. 다양한 특징 추출 기법을 사용하여 이미지에서 유용한 정보를 추출하고, 이를 통해 기계 학습 모델을 학습시킵니다. 세 가지 특징 추출 방법, 즉 엣지 감지를 활용한 녹색 영역 특징 추출, 특징 어휘 개발, 그리고 사전 훈련된 딥러닝 모델을 통한 특징 추출이 사용되었습니다.

- **Performance Highlights**: 이 연구는 세 가지 기계 학습 회귀 모델인 선형 회귀(Linear Regression), 서포트 벡터 머신(Support Vector Machines), 그리고 랜덤 포레스트(Random Forest)를 사용해 LAI를 추정하였습니다. 실험 결과, 복잡한 이미지 데이터를 효과적으로 분석할 수 있는 사전 훈련된 ResNet50 모델을 통해 LAI 예측의 효율성과 강건성을 크게 향상시켰습니다. 본 연구의 새로운 접근법은 정밀 농업 관행을 개선할 수 있는 가능성을 보여줍니다.



### HOPPR Medical-Grade Platform for Medical Imaging AI (https://arxiv.org/abs/2411.17891)
Comments:
          6 pages, 3 figures

- **What's New**: HOPPR Medical-Grade Platform은 인공지능 분야에서 큰 비전을 가진 언어 모델(large vision language models, LVLMs)의 배포를 가속화하기 위해 혁신적인 접근 방식을 제시합니다. 이 플랫폼은 수백 개의 영상 센터에서 수집된 수백만 개의 이미지 연구와 텍스트 보고서를 기반으로 사전 훈련된 기초 모델(foundation models)을 제공합니다.

- **Technical Details**: HOPPR 플랫폼은 대규모 모델을 개발하는 데 필요한 방대한 컴퓨팅 인프라를 갖추고 있으며, 임상 환경에서 배포를 위해 미세 조정(fine-tuning)된 모델을 평가하는 표준을 마련한 품질 관리 시스템을 제공합니다. 또한 모든 데이터는 비식별화(deidentified)되어 HIPAA 규정 준수를 확보하며 안전하게 저장됩니다.

- **Performance Highlights**: HOPPR는 의료 이미징(medical imaging)에 대한 LVLM 솔루션의 배포를 가속화하여 방사선 의사의 업무 흐름(workflows)을 최적화하고 이 분야에서 증가하는 요구를 충족시키는 것을 목표로 합니다. 개발자는 HOPPR 플랫폼에서 모델을 안전하게 호스팅하고 API를 통해 기존 클리닉 워크플로 내에서 추론을 진행할 수 있는 기능을 제공합니다.



### LongKey: Keyphrase Extraction for Long Documents (https://arxiv.org/abs/2411.17863)
Comments:
          Accepted for presentation at the 2024 IEEE International Conference on Big Data (IEEE BigData 2024). Code available at this https URL

- **What's New**: 이 논문은 LongKey라는 새로운 프레임워크를 소개하며, 주로 길이가 긴 문서에서의 키프레이즈(keyphrase) 추출을 목표로 한다. 기존의 키프레이즈 추출 방법들은 보통 단기 문서(최대 512 tokens)에 초점을 맞추고 있어 긴 문서 처리에 한계가 있었다. LongKey는 96,000 tokens까지 처리할 수 있는 Longformer 모델을 활용하여 이러한 한계를 극복한다.

- **Technical Details**: LongKey의 방법론은 세 가지 단계로 구성되어 있다: 초기 단어 임베딩(initial word embedding), 키프레이즈 후보 임베딩(keyphrase candidate embedding), 및 후보 점수 매기기(candidate scoring)이다. Longformer 모델을 활용하여 긴 문서의 구문적 세부사항을 캡처하는 임베딩을 생성한다. 길이가 8,192 tokens을 초과하는 문서는 동등한 크기로 분할 처리되어 각각의 임베딩이 결합되어 하나의 통합된 표현을 생성한다.

- **Performance Highlights**: LongKey는 기존의 비지도 학습 및 언어 모델 기반의 키프레이즈 추출 방법들보다 우수한 성능을 보여준다. 다양한 데이터셋에서 테스트한 결과, LongKey는 키프레이즈 추출의 정확성을 크게 향상시키며, 긴 문서에서의 정보 검색 및 관리에 기여할 수 있을 것으로 기대된다.



### Accelerating Proximal Policy Optimization Learning Using Task Prediction for Solving Games with Delayed Rewards (https://arxiv.org/abs/2411.17861)
- **What's New**: 이 논문에서는 강화 학습(Reinforcement Learning)에서 지연 보상(delayed rewards) 문제를 다룹니다. 최근에 Proximal Policy Optimization(PPO) 방법이 주요 Policy Gradient 방법으로 주목받고 있으나, 지연 보상 하에서는 성능이 저하될 수 있음을 지적합니다. 이를 개선하기 위해 오프라인 정책과 온라인 PPO 정책을 결합한 하이브리드 정책 아키텍처와 Time Window Temporal Logic(TWTL)을 활용한 보상 형태화 메커니즘을 도입합니다.

- **Technical Details**: 하이브리드 아키텍처는 학습 전반에 걸쳐 오프라인 데이터를 활용하면서도 PPO의 이론적 보장을 유지합니다. Trust Region Policy Optimization(TRPO)의 단 monotonic improvement framework에 기반하여, 제안된 방법이 이전 오프라인 정책 및 반복에 비해 개선을 보장함을 입증합니다. 또한, TWTL 기반 보상 형태화가 원래 문제의 최적 정책(optimal policy)을 보존하며, 이는 시간적 목표를 즉각적인 피드백 신호로 변환할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법의 효과는 역회전추(inverted pendulum)와 달 착륙기(lunar lander) 환경에서의 광범위한 실험을 통해 입증되었습니다. 실험 결과, 표준 PPO 및 오프라인 전용 방법에 비해 학습 속도와 최종 성능 모두에서 개선된 결과를 보여줍니다. 이러한 개선은 지연 보상 문제를 효과적으로 해결하는 데 기여할 것으로 기대됩니다.



### "Give me the code" -- Log Analysis of First-Year CS Students' Interactions With GP (https://arxiv.org/abs/2411.17855)
Comments:
          This is the author's version of the work. It is posted here for your personal use. Not for redistribution

- **What's New**: 이 연구는 1학년 컴퓨터 과학 학생들이 특정 프로그래밍 문제를 해결하기 위해 ChatGPT와 상호작용하는 방식을 분석합니다. 학생들은 사전 교육 없이도 코드 생성에 LLM을 활용할 수 있는지를 조사하고 있으며, 결과적으로 대다수의 학생들이 생성된 솔루션을 프로젝트에 성공적으로 통합한 사실이 밝혀졌습니다. 또한, 일부 학생들은 여러 GPT 제안 중에서 선택하는 판단력을 발휘하여 비판적 사고 능력을 발달시킨 것으로 나타났습니다.

- **Technical Details**: 연구는 69명의 1학년 학생들이 특정 프로그래밍 과제를 해결하기 위해 사용한 프롬프트를 분석하였습니다. 학생들은 정해진 템플릿을 통해 LLM을 활용했으며, 자가 기록(log file) 형태로 솔루션의 비교 및 선택 과정을 작성해야 했습니다. 연구의 주요 질문은 학생들이 특별한 교육 없이도 LLM을 잘 사용할 수 있는지와 ChatGPT 솔루션을 프로젝트에 잘 통합할 수 있는지를 포함합니다.

- **Performance Highlights**: 학생들은 비록 정교하지 않은 프롬프트 기술을 사용함에도 불구하고, 생성 모델인 GPT의 도움을 받아 코드 부분을 성공적으로 구현했습니다. 조사가 끝난 후 실시한 설문 결과, 학생들은 LLM 도구의 유용성에 대해 긍정적인 의견을 보였으며, LLM을 사용한 경험이 그들의 비판적 사고 능력에 어떻게 기여했는지에 대한 통찰력을 얻었습니다.



### SoftmAP: Software-Hardware Co-design for Integer-Only Softmax on Associative Processors (https://arxiv.org/abs/2411.17847)
Comments:
          Accepted in DATE 2025

- **What's New**: 최근 연구는 자원 제약이 있는 기기에서도 사용할 수 있도록 대규모 언어 모델(Large Language Models, LLMs)의 계산 및 메모리 오버헤드를 줄이는 데 초점을 맞추고 있습니다. 이에 따라 저희는 SoftmAP라는 소프트웨어-하드웨어 공동 설계 방법론을 제안하며, 이는 In-Memory Compute (IMC) 하드웨어를 사용하여 정수 전용 저정밀 Softmax를 구현합니다. 이 방법은 A100 및 RTX3090 GPU와 비교하여 에너지 지연 곱에서 최대 1,300배 개선을 달성하여 LLM을 성능 손실 없이 배포 가능하게 만듭니다.

- **Technical Details**: 대규모 언어 모델은 주로 transformer 아키텍처를 기반으로 하며, 특히 디코더 전용 transformer는 순차적으로 텍스트를 생성하는 데 적합합니다. Softmax는 주의 메커니즘의 핵심 구성 요소로, 입력 텍스트에서 각 단어의 중요성을 평가합니다. 저희는 Softmax의 정수 전용 저정밀 근사값을 제안하며, Associative Processors (APs)라는 맞춤형 하드웨어에서 이를 구현하는 방법을 탐구합니다.

- **Performance Highlights**: 실험 분석 결과, Softmax를 정수 전용으로 근사화하고 AP에서 배포함으로써 RTX3090 GPU 및 A100 GPU와 비교하여 에너지 소모 비율을 1300배, 지연 시간을 12.58배 줄일 수 있음을 확인하였습니다. 이러한 결과는 Llama 모델을 대상으로 하여 실험적으로 입증되었습니다. 이로 인해 저희의 제안 기법은 대규모 언어 모델을 보다 효율적으로 활용할 수 있는 가능성을 제시합니다.



### Basic Research, Lethal Effects: Military AI Research Funding as Enlistmen (https://arxiv.org/abs/2411.17840)
Comments:
          22 pages, 9945 words

- **What's New**: 이 논문은 전례 없는 미국 국방부(DoD) 예산 상황에서 DoD가 알고리즘 기반 전쟁 수행을 위한 학술 연구에 대한 자금 지원이 어떻게 이루어지는지를 조사합니다. 저자들은 2007년부터 2023년까지의 DoD 보조금 요청서에서 인공지능(AI) 분야 연구자들에게 전달된 내용을 중점적으로 다룹니다.

- **Technical Details**: 첫 번째 섹션에서는 기초 연구(basic research)와 응용 연구(applied research)의 구분에 대한 비판적 검토를 실시합니다. 기초 연구로 프레임화된 자금 지원 호출(call)들이 여전히 연구자들을 전쟁 수행의 목표에 동원하는 방식을 보여줍니다. 두 번째 섹션에서는 시간에 따른 분석(diachronic analysis)을 통해, 군사 기술의 발전을 긍정하는 대신 남아 있는 문제들을 인정하면서 추가 투자(justification for additional investments)를 정당화하는 '한 가지 작은 문제(one small problem)' 조건을 설명합니다.

- **Performance Highlights**: 마지막으로, AI의 전장(application in battlefield) 활용을 위한 방위 고등 연구 계획국(DARPA) 보조금 요청서의 분석을 통해 DoD의 야망을 살펴봅니다. 종합적으로, 저자들은 보조금 요청이 DoD 자금 지원 기관과 학술 AI 연구 커뮤니티 간의 연구 의제를 설정하는 상호 동원 수단으로 작용한다고 주장합니다.



### Arabic-Nougat: Fine-Tuning Vision Transformers for Arabic OCR and Markdown Extraction (https://arxiv.org/abs/2411.17835)
Comments:
          7 pages, 1 figure

- **What's New**: 아랍어 책 페이지를 구조화된 Markdown 텍스트로 변환하는 OCR 모델인 Arabic-Nougat를 소개합니다. 이 모델들은 Meta의 Nougat 아키텍처를 기반으로 구성되며, arabic-small-nougat, arabic-base-nougat, arabi-large-nougat의 세 가지 전문 모델로 이루어져 있습니다. 아랍어 책 페이지와 해당 Markdown 표현 간의 13.7k 쌍으로 구성된 synthetic dataset인 arabic-img2md를 사용하여 미세 조정되었습니다.

- **Technical Details**: Arabic-Nougat의 핵심 기술 요소 중 하나는 Aranizer-PBE-86k tokenizer로, 이는 효율적인 토큰화를 위해 설계되었습니다. torch.bfloat16의 정밀도와 Flash Attention 2를 활용하여 학습 및 추론을 최적화했습니다. 이 모델들은 다양한 아랍어 텍스트 레이아웃과 긴 문서 처리를 효과적으로 해결하기 위한 아랍어 특화 개선 사항을 포함하고 있습니다.

- **Performance Highlights**: arabic-large-nougat는 최고의 Markdown 구조 정확도와 최저 문자 오류율을 기록하며 최상의 성능을 달성했습니다. 또한, 8,500권 이상의 책에서 추출한 11억 개의 아랍어 토큰을 포함하는 대규모 데이터셋을 제공하여 아랍어 OCR 연구에 유용한 자원을 제공합니다. 모든 모델과 데이터셋 및 코드가 오픈 소스로 제공되어 연구자들이 자유롭게 활용할 수 있습니다.



### SVGDreamer++: Advancing Editability and Diversity in Text-Guided SVG Generation (https://arxiv.org/abs/2411.17832)
Comments:
          17 pages, 17 figures. arXiv admin note: substantial text overlap with arXiv:2312.16476

- **What's New**: 이번 연구는 텍스트 가이드를 기반으로 한 새로운 벡터 그래픽 합성 방법을 제안하여 SVG의 시각적 품질과 다양성을 향상시키고자 합니다. 이를 위해 Vectorized Particle-based Score Distillation (VPSD) 기법을 도입하여 기존 방법의 과포화 문제를 해결하고 출력 SVG의 다양성을 증대시킵니다. 또한, 적응형 벡터 프리미티브 제어 전략을 설계하여 그래픽 세부 묘사의 프리미티브 수를 동적으로 조정할 수 있도록 합니다.

- **Technical Details**: Scalable Vector Graphics (SVG)은 선형 기하학적 원소인 Bézier 곡선, 다각형 및 선을 사용하여 시각적 개념을 표현합니다. 최근 텍스트에서 SVG로의 변환을 위한 모델들이 급속히 발전하고 있으며, CLIP 모델과 DiffVG를 결합하여 SVG 생성을 위한 다양한 방법들이 제안되고 있습니다. 그러나 기존 Text-to-SVG 방법은 생성된 이미지의 편집 가능성 부족, 시각적 품질 저하, 제한된 다양성 등의 문제를 내포하고 있습니다.

- **Performance Highlights**: SVGDreamer++는 벡터 그래픽 생성에서 편집 가능성, 시각적 품질 및 다양성 측면에서 기존 방법보다 우월한 성능을 보입니다. 본 연구에서는 최대 여섯 가지 서로 다른 벡터 스타일을 지원하며, 다양한 벡터 디자인에도 적용 가능한 고품질의 벡터 자산을 생성할 수 있음을 실험을 통해 입증하였습니다. 또한, 이 접근법은 아이콘 생성 및 포스터 디자인을 포함한 벡터 디자인 영역에서의 응용 가능성을 보여줍니다.



### STAR: Synthesis of Tailored Architectures (https://arxiv.org/abs/2411.17800)
- **What's New**: 이 논문에서는 모델 아키텍처의 합성을 위한 새로운 접근 방식인 STAR(Synthesis of Tailored Architectures)를 제안합니다. STAR는 선형 입력 변이 시스템(linear input-varying systems) 이론에 기반한 새로운 탐색 공간(search space)을 결합하여 아키텍처 유전자(genomes)를 계층적 수치 인코딩(hierarchical numerical encoding)으로 지원합니다.

- **Technical Details**: STAR 아키텍처 유전자는 경량의 진화 알고리즘(evolutionary algorithms)을 사용하여 자동으로 정제되고 재조합됩니다. 이 알고리즘은 다중 모델 품질 및 효율성 메트릭을 최적화하는 데 초점을 맞추고 있으며, 기존에 잘 최적화된 Transformer 및 스트라이프형 하이브리드 모델을 초월하여 다양한 계산 단위(computational units)와 상호 연결 패턴(interconnection patterns)을 활용합니다.

- **Performance Highlights**: STAR를 통한 대규모 아키텍처 집단의 최적화 결과, 품질, 매개변수(size), 추론 캐시(inference cache) 면에서 향상된 성능이 나타났습니다. 이 연구는 오토 리그레시브 언어 모델링(auto-regressive language modeling) 분야에서 새로운 경험적 성과를 제공합니다.



### DapPep: Domain Adaptive Peptide-agnostic Learning for Universal T-cell Receptor-antigen Binding Affinity Prediction (https://arxiv.org/abs/2411.17798)
- **What's New**: 이번 연구에서는 T 세포 수용체(TCR)와 항원 펩타이드 간의 상호작용을 식별하기 위한 새로운 DapPep 프레임워크를 소개합니다. DapPep은 일반화된 TCR-항원 결합 친화도를 예측할 수 있는 도메인 적응형(peptide-agnostic) 학습 방법론으로, 기존의 툴들에 비해 강력한 일반화 능력을 보여줍니다. 특히, 데이터가 부족한 환경이나 보지 못한 펩타이드에 대해서도 효과적입니다.

- **Technical Details**: DapPep은 TCR 표현 모듈, 펩타이드 표현 모듈 및 TCR-펩타이드 표현 모듈의 세 가지 주요 구성 요소로 이루어져 있습니다. 이 구조는 다중 헤드(self-attention) 층에 기반하여 모든 모듈을 가볍게 설계하며, 이를 통해 TCR과 펩타이드의 강력한 표현을 생성합니다. DapPep의 훈련은 두 단계로 나뉘며, 첫 번째 단계에서 TCR 표현 모듈을 초기화하고, 두 번째 단계에서 이를 결합하여 결합 친화도를 학습합니다.

- **Performance Highlights**: DapPep의 성과는 다양한 기준에서 입증되었으며, 특히 신규 펩타이드, 외래 항원 및 종양 신항원 치료와 같은 복잡한 임상 작업에서 효과적으로 작용합니다. 실험 결과에 따르면, DapPep은 기존의 모델들을 지속적으로 초월하며, 높은 처리량과 효율성을 자랑합니다. TCR-펩타이드 상호작용에 관한 과학적 기준을 제공하여, 향후 면역치료 개발에 중요한 기여를 할 것으로 기대됩니다.



### Pan-protein Design Learning Enables Task-adaptive Generalization for Low-resource Enzyme Design (https://arxiv.org/abs/2411.17795)
- **What's New**: 이 논문은 기능적 설계를 위한 새로운 컴퓨테이셔널 단백질 설계(CPD) 패러다임을 소개합니다. 특히, 효소와 같은 특정 기능 요구 사항을 충족하지 못하는 단백질 클래스에 초점을 맞추고 있습니다. 기존의 CPD 모델이 일반적인 범위의 단백질에 국한된 반면, CrossDesign은 구조 데이터를 활용하여 효능을 개선할 수 있도록 설계되었습니다.

- **Technical Details**: CrossDesign 프레임워크는 사전 훈련된 프로틴 언어 모델(Pretrained Protein Language Model, PPLM)과 단백질 구조 간의 교차 모드 정렬(cross-modal alignment)을 통합합니다. 이 구조는 인코더-디코더 아키텍처를 사용하여 자율적(autonomous)이고 비자율적(non-autoregressive) 샘플링 방식으로 효소 데이터셋에 적용됩니다. 아울러, 상호 모드 정렬(InterMA)과 계층 간 일관성(Cross-Layer Consistency, CLC)을 구현하여 모델의 훈련을 규제합니다.

- **Performance Highlights**: CrossDesign은 다양한 도메인 내 및 도메인 외의 작업에서 강력한 성능을 보여줍니다. 대규모 변이 데이터에 대한 적합도 예측에서 우수한 정확도를 기록하여 모델의 안정성을 입증합니다. 이 연구는 일반적인 단백질과 특정 기능 단백질 간의 구조-서열 변환 간극을 좁히기 위한 효소 데이터셋와 벤치마크를 구축합니다.



### Engineering AI Judge Systems (https://arxiv.org/abs/2411.17793)
- **What's New**: 이 논문은 AI 판단 시스템(AI judge systems)을 개발하는 데 있어 직면하는 새로운 도전 과제와 직업 근거를 소개합니다. 특히, FMware의 특성으로 인해 개발 주기와 평가 기준이 다르게 요구되는 점이 강조됩니다. 코미트 메시지 생성(FMG) 시스템에 대한 사례 연구를 통해, 저자들은 기존 시스템과 비교하여 새로운 프레임워크의 유용성을 입증합니다.

- **Technical Details**: AI 판단 시스템은 FMware의 평가를 자동화하기 위해 설계되었으며, 각 시스템의 기능은 상황에 따라 유동적으로 변화하는 요구를 반영해야 합니다. 논문에서는 요구 사항 정의의 복잡성, AI 판단 시스템 개발의 비효율성, 그리고 지속적 진화와 같은 여러 도전 과제들이 제시됩니다. 이러한 도전 과제를 해결하기 위해 제안된 프레임워크는 네 가지 단계로 구성되어 있으며, 각 단계는 AI 에이전트가 수행합니다.

- **Performance Highlights**: 제안된 프레임워크에 기반하여 개발된 AI 판단 시스템의 정확도는 이전 프레임워크 없이 개발된 시스템에 비해 최대 6.2% 향상된 것으로 나타났습니다. 이는 개발자가 수동 평가의 필요성을 줄이고 효율성을 높이는 데 기여합니다. 사례 연구를 통해 이 프레임워크가 고품질의 AI 판단 시스템을 개발하는 데 실질적인 도움을 줄 수 있음을 입증했습니다.



### $H^3$Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs (https://arxiv.org/abs/2411.17792)
- **What's New**: 이 논문에서는 pretrained LLMs(사전 학습된 대형 언어 모델)을 인간의 선호에 맞추기 위해 instruction-based datasets(명령어 기반 데이터 세트)로 aligning(정렬)하는 중요성을 다룹니다. 새로운 alignments(정렬) 방식인 $H^3$Fusion이 개발되어, 여러 개별적으로 정렬된 LLM을 결합하여 성능을 향상시키는 방법론을 제안합니다. $H^3$Fusion은 또한 mixture-of-experts(MoE) 방법론을 활용하여 효과적인 응답 생성을 위한 최적의 전문가 선택을 포함합니다.

- **Technical Details**: $H^3$Fusion의 주요 특징은 세 가지로 나뉩니다. 첫째, 여러 개별 모델을 앙상블하여 최종 정렬 모델을 구축하며, 이 과정에서 helpful(도움이 되는), harmless(해가 없는), honest(정직한) 특성을 촉진합니다. 둘째, モE 방법론을 통해 각 모델의 multi-head attention weights(다중 헤드 주의 가중치)를 고정하고 FFN(Feed Forward Network) 계층만 튜닝하여 정렬을 수행합니다. 마지막으로, gating loss(게이팅 손실)와 regularization terms(정규화 항)를 도입하여 모델 성능을 향상시키고 전문가 선택 오류를 제어합니다.

- **Performance Highlights**: 논문의 평가 결과, $H^3$Fusion 모델이 개별적으로 정렬된 모델보다 11.37% 더 유용하고 최신 LLM 앙상블 방법 기준으로 13.77% 더 강한 견고성을 보였습니다. 세 가지 기준 데이터 세트인 Alpaca-Eval, BeaverTails, TruthfulQA에서 extensive evaluations(광범위한 평가)를 수행하여 유용성, 해가 없음, 정직함 측면에서 우수한 성능을 입증하였습니다. 또한, 이 연구는 앙상블 접근 방식을 사용하여 정렬을 수행한 최초의 연구로 의미가 큽니다.



### Self-supervised Monocular Depth and Pose Estimation for Endoscopy with Generative Latent Priors (https://arxiv.org/abs/2411.17790)
- **What's New**: 이 논문은 내시경(Endoscopy) 환경에서의 3D 매핑 성능 향상을 위한 새로운 프레임워크를 제안합니다. 특히, Generative Latent Bank와 Variational Autoencoder (VAE)를 활용한 자가 지도 학습(self-supervised learning) 기반의 깊이(depth) 및 자세(pose) 추정 방법이 주효함을 보여줍니다. 기존의 합성 데이터(synthetic datasets)나 복잡한 모델에 의존하는 방법들이 범용성을 결여하고 있었음을 지적하며, 새로운 방법으로 내시경 환경의 복잡한 질감과 조명 문제를 해결하고자 합니다.

- **Technical Details**: 제안하는 방법은 DepthNet과 PoseNet의 두 가지 주요 브랜치를 포함하며, 각각 깊이와 자세를 추정합니다. Generative Latent Bank를 통해 자연 이미지에서 얻은 깊이 정보를 바탕으로 깊이 예측의 사실성과 안정성을 향상시키고, 자세 추정은 VAE를 사용하여 자세 전환을 잠재 변수(latent variables)로 다룸으로써 최적화합니다. 이 설정은 Z축의 강조를 안정시키고, X-Y 민감도를 개선하여 내시경 촬영 시의 복잡한 텍스처 문제 해결에 기여합니다.

- **Performance Highlights**: SimCol 및 EndoSLAM 데이터셋에서의 평가 결과, 제안된 프레임워크가 기존의 자가 지도 방법들보다 우수한 성능을 보였습니다. ablation 연구를 통해 각 제안된 구성 요소의 효과가 검증되었으며, 이로 인해 내시경 깊이 및 자세 추정에서의 정확도가 높아졌습니다. 이 접근 방식은 임상 환경에서 내시경 진단과 치료의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### Geometric Point Attention Transformer for 3D Shape Reassembly (https://arxiv.org/abs/2411.17788)
- **What's New**: 본 논문에서는 기하학적 관계를 추론하기 위한 네트워크인 Geometric Point Attention Transformer (GPAT)를 제안합니다. 기존 방법들이 부분 간의 기하학적 상호작용을 정확하게 캡처하지 못했던 문제를 해결하고자 합니다. GPAT는 각 부분의 자세를 회전 및 변환 벡터로 나타내어, 전역 형상 정보와 지역 쌍별 기하학적 특징을 통합합니다. 이를 통해 모델의 성능을 개선하고, 반복적인 예측을 가능하게 하는 기하학적 재활용 모듈도 도입됩니다.

- **Technical Details**: GPAT의 주요 구성 요소는 기하학적 포인트 어텐션 모듈과 기하학적 재활용 모듈로, 이들은 지역 기하학, 6-DoF 예측 및 동적 모델링 문제를 다루기 위해 설계되었습니다. 기하학적 포인트 어텐션 모듈은 각 부분의 회전 및 변환 벡터를 어텐션 점수 계산에 통합함으로써, 정확한 조립을 위한 공간적 관계를 캡처합니다. 결과적으로 반보적인 회전과 변환의 업데이트가 이루어져, 6-DoF 기하학적 특성을 보존한 채로 자세를 직접 예측할 수 있습니다.

- **Performance Highlights**: 우리 모델은 PartNet 데이터셋의 의미적 조립 작업과 Breaking Bad 데이터셋의 기하학적 조립 작업 모두에서 기존 방법들에 비해 우수한 성능을 보여주었습니다. 반복적인 예측 기능은 조립 과정에서의 틀림이 있는 부분을 정교하게 개선하는 데 도움을 줍니다. 또한, GPAT는 미래의 형상 조립 연구 및 6-DoF 예측 작업에 있어 우선 선택되는 백본으로 자리 잡을 가능성이 큽니다.



### DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching (https://arxiv.org/abs/2411.17786)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문에서 제안하는 DreamCache는 개인화된 이미지 생성을 위한 효율적이고 품질 높은 접근 방식으로, 기존 방법의 단점을 극복합니다. 특히, 적은 수의 레이어에서 참조 이미지 특징을 캐싱(caching)하여 생성 과정에서 동적인 조정을 가능하게 합니다. 이는 기존의 복잡한 훈련 요구사항 없이도 고품질의 개인화된 이미지를 생성할 수 있도록 해줍니다.

- **Technical Details**: DreamCache는 사전 훈련된 diffuion denoiser의 단일 타임스텝에서 특징을 캐싱하여 사용자 생성 텍스트 캡션 없이 이미지를 생성합니다. 또한, 적은 수의 추가 파라미터만으로 경량화되어 리소스가 제한된 장치에서도 사용 가능하며, 기존 U-Net의 무게를 변경하지 않고도 개인화와 비개인화 콘텐츠의 동시 생성을 지원합니다. 이러한 방식은 기존 모델에 비해 직접적인 메모리 및 연산 비용을 절감합니다.

- **Performance Highlights**: DreamCache는 기존 방법들에 비해 매우 낮은 연산 및 데이터 비용으로 개인화된 이미지 생성에서 최고 수준의 품질을 달성합니다. 캐싱된 특징을 사용하여 신속하고 효율적인 개인화 샘플링을 구현하며, 이로 인해 고품질 이미지를 실시간으로 생성할 수 있는 가능성이 열립니다. 따라서 DreamCache는 개인화된 이미지 생성의 실용성과 확장성을 크게 개선한 혁신적인 접근법으로 평가됩니다.



### Joint Resource Optimization, Computation Offloading and Resource Slicing for Multi-Edge Traffic-Cognitive Networks (https://arxiv.org/abs/2411.17782)
- **What's New**: 본 논문에서는 에지 컴퓨팅(edge computing) 환경에서 응용 프로그램 제공자와 에지 서버(edge servers) 간의 효율적인 상호작용을 위한 새로운 접근 방식을 제시합니다. 플랫폼과 에지 서버가 상호 이익을 추구하는 다중 에이전트 시스템을 조사하여, 수익 극대화(revenue maximization), 자원 할당(resource allocation), 그리고 작업 오프로드(task offloading)의 공동 최적화를 다룹니다.

- **Technical Details**: 우리는 이해관계자 간의 상호작용을 모델링하기 위해 새로운 Stackelberg 게임(Stackelberg game) 기반 프레임워크를 제안하며, 최적화 문제를 해결하기 위해 베이지안 최적화(Bayesian Optimization) 기반의 중앙 집중형 알고리즘을 사용합니다. 개인정보 보호 우려로 인한 정보 수집의 실질적인 문제를 인식하고, 우리는 신경망 최적화(neural network optimization)와 개인정보 보호 정보 교환 프로토콜을 활용한 분산 솔루션을 설계합니다.

- **Performance Highlights**: 폭넓은 수치 평가를 통해 제안된 메커니즘이 기존의 기준선(baselines)과 비교하여 우수한 성능을 달성함을 입증하였습니다. 이를 통해 에지 컴퓨팅 플랫폼에서 자원의 효율적인 활용과 엄격한 서비스 품질(QoS) 요구 사항을 충족하는 것이 가능하다는 점을 강조합니다.



### Leaning Time-Varying Instruments for Identifying Causal Effects in Time-Series Data (https://arxiv.org/abs/2411.17774)
Comments:
          14 pages

- **What's New**: 이 논문에서는 시간에 따른 인과 효과 추정을 위한 새로운 방법인 Time-varying Conditional Instrumental Variables (CIV) for Debiasing causal effect estimation (TDCIV)을 제안합니다. 기존의 인과 추정 방법들이 시간에 따라 변하는 잠재 혼란 변수(latent confounders)로 인해 겪는 한계를 극복하기 위해 고안된 이 방법은 Long Short-Term Memory (LSTM)와 Variational Autoencoder (VAE) 모델을 사용하여 데이터로부터 직접적으로 시간 가변 CIV를 학습할 수 있습니다. 이를 통해 특정 도메인 지식에 의존하지 않고 인과 효과를 보다 정확히 추정할 수 있는 가능성을 열어줍니다.

- **Technical Details**: TDCIV는 그래픽 원인 모델(graphical causal models)을 기반으로 하며, 학습한 시간 가변 CIV의 표현(representation)과 그 조건집합을 분리하여 인과 관계를 식별할 수 있게 합니다. TDCIV 모델은 시간 가변 인자와 잠재 혼란 변수를 고려하여 인과 추정을 수행하기 위해 두 단계 최소제곱법(2SLS)을 적용합니다. 이 방법은 시간 시계열 데이터에서 발생하는 복잡한 동적 관계를 효과적으로 다룰 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, TDCIV는 합성 데이터(synthetic data) 및 실제 기후 데이터(real-world climate data)에서 시간에 따라 변하는 잠재 혼란 변수에 영향을 받으면서도 인과 효과를 올바르게 식별하는 데 우수한 성능을 보였습니다. 이는 TDCIV가 인과 추정 분야에서 시간 가변 조건부 도구 변수(time-varying Conditional IV)를 직접적으로 학습할 수 있는 첫 번째 방법이라는 점에서 중요합니다. 따라서 이 연구는 인과 추정 방법론에 대한 새로운 통찰을 제공하며 향후 다양한 응용 분야에 적용될 수 있는 기반을 마련합니다.



### MVBoost: Boost 3D Reconstruction with Multi-View Refinemen (https://arxiv.org/abs/2411.17772)
- **What's New**: 본 연구에서는 단일 뷰 이미지를 사용하여 3D 재구성을 강화하는 새로운 프레임워크인 다중 뷰 정제(multi-view refinement) 기법, MVBoost를 제안합니다. 이 방법론은 고정밀 다중 뷰 생성 모델과 3D 재구성 모델의 장점을 결합하여 신뢰할 수 있는 데이터 소스를 만들어냅니다. 이를 통해 고품질 3D 데이터를 생성하고, 사용자가 제공한 입력 이미지를 기반으로 최적의 뷰포인트를 개별화하는 과정을 포함시켜, 재구성의 품질을 향상시키는 기능을 갖추고 있습니다.

- **Technical Details**: MVBoost는 단일 뷰 입력 이미지를 바탕으로 다중 뷰 이미지를 생성하기 위해 다중 뷰 확산 모델을 사용합니다. 이 이미지들은 대규모 3D 재구성 모델로 전송되어 일관된 3D 표현으로 변환됩니다. 그 후, 해당 3D 데이터에서 렌더링된 다중 뷰 이미지를 정제하여 대규모 다중 뷰 데이터셋을 생성하고, 이를 피드포워드 3D 재구성 모델 훈련에 사용합니다. 최적의 재구성을 위해 3D 모델과 입력 이미지의 정렬을 최적화하는 과정을 추가하여 사용자 요구에 맞는 재구성을 돕습니다.

- **Performance Highlights**: GSO 데이터셋을 통한 평가 결과, MVBoost 방법이 기존 재구성 방법보다 우수한 성능을 보였습니다. 정성적 및 정량적 결과 모두에서 고충실도의 3D 재구성을 달성하며 최신 기술 상태를 기록하였습니다. 이 연구는 다양한 단일 뷰 데이터 세트를 통합하여 3D 재구성을 훈련하는 프레임워크를 마련함으로써, 재구성 결과의 향상을 꾀할 수 있음을 입증하였습니다.



### PROGRESSOR: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinemen (https://arxiv.org/abs/2411.17764)
Comments:
          15 pages,13 figures

- **What's New**: 본 논문에서는 비디오로부터 작업에 구애받지 않는 보상 함수를 학습하는 새로운 프레임워크인 PROGRESSOR를 제안합니다. 이 프레임워크는 수동 감독 없이 목표 조건 강화 학습(goal-conditioned reinforcement learning)을 가능하게 합니다. PROGRESSOR는 자가 지도(self-supervised) 방식으로 현재, 초기 및 목표 관찰에 대한 작업 진행 상황의 분포를 추정하며, 비전문 관찰에서 발생하는 분포 이동 문제를 완화하기 위해 온라인 RL 훈련 중 보상을 적대적으로 수정합니다.

- **Technical Details**: PROGRESSOR는 unlabeled(레이블이 없는) 비디오로부터 학습하여 목표 달성을 위한 진행 상황을 예측하는데 중점을 둡니다. 이는 목표 이미지로 모델을 조건화하여 다양한 비디오 데이터를 단일 모델에 통합하는 효과적인 방법을 제공합니다. 온라인 적대적 보상 수정을 통해 PROGRESSOR는 비전문적인 상태에서 발생하는 예측을 저하시키며, 이를 통해 에이전트가 전문가의 행동 경로와 일치하도록 탐색을 유도합니다.

- **Performance Highlights**: PROGRESSOR는 EPIC-KITCHENS에서 수집된 대규모 에고센트릭 비디오로 사전 훈련되어, 특수한 작업 데이터에 대한 미세 조정 없이도 실제 로봇 오프라인 RL에서 우수한 성능을 발휘합니다. 또한 Meta-World 벤치마크의 여섯 가지 다양한 작업에서도 세분화된 보상 없이 최첨단 성능을 달성하여, 복잡한 동작 학습에서 엄청난 가능성을 보여줍니다.



### Efficient Self-Improvement in Multimodal Large Language Models: A Model-Level Judge-Free Approach (https://arxiv.org/abs/2411.17760)
- **What's New**: 본 논문은 MLLMs(다중 모달 대형 언어 모델)의 신뢰성과 강건성을 향상시키기 위한 자가 개선 방법을 제안합니다. 기존 방법들이 MLLMs 자체를 판단자로 사용하는데 따른 높은 계산 비용과 잠재적 문제를 해결하기 위해, 모델 수준의 판단자 없는 자가 개선 프레임워크를 도입했습니다. 이를 통해, 통제 가능한 망상(hallucination) 메커니즘을 활용하여 데이터 품질을 최적화하고, 가벼운 대조 언어-이미지 인코더를 통해 샘플을 평가하여 자가 개선의 경로를 효율화합니다.

- **Technical Details**: 제안된 방법은 통제 가능한 망상 메커니즘을 사용하여 선호 학습(pair) 쌍을 생성하고, 대조적 언어-이미지 인코더를 통해 데이터 품질을 평가합니다. 초기 데이터셋을 생성한 후, CLIPScore를 계산하여 부정 샘플의 점수가 긍정 샘플보다 높은 쌍을 업데이트합니다. 이후, 최적화된 데이터셋을 사용하여 DPO(direct preference optimization) 기법을 통하여 시드 모델을 학습시킵니다. 이 과정을 통해 최종적으로 자가 개선된 모델을 얻게 됩니다.

- **Performance Highlights**: 본 방법은 대규모 벤치마크 및 새롭게 도입한 IC 데이터셋에서 기존 기술들보다 우수한 성능을 보였습니다. 정밀도와 재현율이 개선되었으며, 계산 요구사항이 현저히 낮아졌습니다. 실험 결과는 시드 모델에 비해 IC 및 Object HalBench 데이터셋에서 значный 향상을 확인했습니다.



### Will an AI with Private Information Allow Itself to Be Switched Off? (https://arxiv.org/abs/2411.17749)
- **What's New**: 이번 연구에서는 인공지능(AI)이 오프 스위치를 비활성화하는 원인에 대한 새로운 관점을 제시하고 있습니다. AI가 비공식적인 정보를 갖고 있는 상황에서, 이전의 이론은 인간이 AI의 모든 행동을 아는 것으로 가정했으나, 실질적으로는 인간의 정보가 제한적입니다. 이를 반영하기 위해 우리는 Partially Observable Off-Switch Game (POSG)라는 비대칭 정보 모델을 도입하였습니다.

- **Technical Details**: POSG는 게임 이론적 접근을 통해 오프 스위치 문제를 다룹니다. 완전한 정보가 주어졌을 때와 달리, 최적의 플레이에서 AI가 인간에게 완벽하게 이롭게 행동하더라도 때로는 비활성화를 피할 수 있다는 점이 시사됩니다. 또한, 통신(residue communication)의 양이 증가하면 에이전트의 공통 기대보상(expected common payoff)이 증가하는 것으로 나타났습니다.

- **Performance Highlights**: 우리는 제한된 통신 상황에서도 AI의 최적 플레이에서 인간에게 덜 의존하게 되는 경우를 관찰하였습니다. 이러한 현상은 제한된 통신이 오히려 AI의 전략적 권한을 부여할 수 있음을 보여줍니다. 따라서 비대칭 정보가 존재하는 패러다임에서 안전한 인공지능 에이전트를 설계하기 위해서는 기대 보상을 극대화하는 것과 인간에 대한 AI의 유도성이 유지되는 것 사이의 균형을 신중하게 고려해야 합니다.



### UVCG: Leveraging Temporal Consistency for Universal Video Protection (https://arxiv.org/abs/2411.17746)
- **What's New**: AI 기반 비디오 편집의 보안 위험이 커지고 있는 가운데, 본 연구는 Universal Video Consistency Guard (UVCG)라는 혁신적인 방법을 제안합니다. UVCG는 비디오의 시간적 일관성을 활용하여 멀티미디어 콘텐츠를 보호하며, 연속적인 왜곡을 도입하여 비디오 편집 모델의 인코더가 잘못된 출력으로 매핑하도록 유도합니다. 이를 통해 비디오 내용의 무단 수정에 대한 효과적인 방어 수단을 제공합니다.

- **Technical Details**: UVCG는 연속적인 비디오 프레임 간의 정보의 일관성을 고려하여, 특정 타겟 비디오의 내용을 보호 비디오에 삽입하는 방법입니다. 이 과정에서는 perturbation-reuse 전략을 통해 계산 효율성을 높이며, 최소한의 GPU 리소스만으로도 효과적인 보호를 달성합니다. 또한, projected gradient descent (PGD)를 사용하여 최적화 문제를 해결합니다.

- **Performance Highlights**: UVCG는 다양한 Latent Diffusion Models (LDM) 버전에서 테스트되었으며, 비디오 편집 파이프라인에서의 일반화 가능성을 평가했습니다. 실험 결과, UVCG는 편집된 비디오의 왜곡을 유의미하게 증가시켰고, 낮은 계산 자원 소모로 87%의 보호 성공률을 기록했습니다. 이러한 결과는 비디오 콘텐츠 무단 수정 방지에서 UVCG의 효과성과 범용성을 입증합니다.



### Soil Characterization of Watermelon Field through Internet of Things: A New Approach to Soil Salinity Measuremen (https://arxiv.org/abs/2411.17731)
- **What's New**: 이 연구는 수박 재배를 위한 스마트 IoT 기반 토양 특성 측정 시스템을 설계하고 구현하는 것을 목표로 합니다. 특히, 이 시스템은 토양의 모이스처(moisture), 온도(temperature), 그리고 pH를 다양한 센서를 통해 측정하고, 데이터를 아두이노(Arduino)와 라즈베리 파이(Raspberry Pi)를 통해 클라우드에 업로드합니다. 사용자는 개발된 모바일 애플리케이션(app) 및 웹페이지를 통해 이 데이터에 접근할 수 있습니다.

- **Technical Details**: 토양의 염도를 측정하기 위한 모델이 제안되며, 이는 토양 저항율(resistivity)을 기반으로 합니다. 연구에서는 존재하는 필드 토양 미터와 IoT 시스템에서 얻은 센서 데이터를 비교하여 수확물의 정밀도를 확보합니다. 또한 인공신경망(ANN)을 사용하여 실험실에서 얻은 데이터를 통해 토양 염도와 저항율 간의 관계를 확립합니다.

- **Performance Highlights**: 정확한 토양 특성 측정은 수박 수확량을 높이는 데 필수적입니다. 개발된 시스템은 클라우드 기반으로 실시간 데이터를 제공하여 농민들이 보다 나은 농업 결정을 내릴 수 있도록 도와줍니다. 본 연구는 IoT 기술이 농업 분야에서 효율성을 증대시키는 방법을 제시하며, 특히 수박 재배에 최적화된 솔루션을 제공합니다.



### Fast convolution algorithm for state space models (https://arxiv.org/abs/2411.17729)
Comments:
          5 pages

- **What's New**: 이번 논문에서는 선형 시간 불변 시스템(LTI)의 매트릭스 전이 함수(Matrix Transfer Function)를 시간 영역에서 빠르고 강력하게 적용할 수 있는 알고리즘을 제안합니다. 기존 방식들이 L 상태를 도출하기 위해 L개의 매트릭스-벡터 곱셈(matrix-vector multiplications)을 요구하는데 반해, 새로운 알고리즘은 사용자가 선택한 유한한 정확도를 보장하면서 매트릭스-벡터 곱셈의 수를 \mathcal{O}(\log_{2}L)로 줄일 수 있습니다. 이는 특히 상태 공간 모델(State Space Models)에서 긴 의존성(long range dependencies)을 다룰 때 유용합니다.

- **Technical Details**: 이 알고리즘은 z-영역(z-domain)에서 유리 전이 함수(rational transfer function)를 2^{N+1}-1 차의 매트릭스 다항식으로 근사화하여 사용합니다. N은 사용자가 선택한 정확도를 달성하기 위해 선택됩니다. 시간 영역에서 구현 시 전이 함수의 적용은 N+1개의 매트릭스-벡터 곱셈만을 필요로 하며, 이는 기존 방법들과 비교해 월등한 성능을 보입니다. 논문에서는 구조화된 매트릭스를 통한 근사화 방법도 논의되어 계산 비용을 더욱 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: 구현된 알고리즘은 복잡한 LTI 시스템의 상태 행렬(state matrix)이 구조화된 매트릭스로 근사화될 때 더욱 정밀한 성능을 발휘합니다. 논문에서 제시된 여러 가지 매트릭스의 구조화 근사 방법들은 다양한 실제 애플리케이션에 적용될 수 있습니다. 이러한 접근 방식은 필터 설계(filter design) 및 디지털 신호 처리(digital signal processing)에서 매우 중요한 이점을 제공할 것으로 보입니다.



### When IoT Meet LLMs: Applications and Challenges (https://arxiv.org/abs/2411.17722)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData), 10 pages, 2 figures, 1 table

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다양한 분야에서 워크플로우를 효과적으로 변화시키고 있습니다. 본 논문에서는 LLM과 사물인터넷(IoT) 통합의 역할을 탐구하며, 이러한 통합이 의사결정 및 시스템 상호작용을 어떻게 개선할 수 있는지에 초점을 맞추고 있습니다. 특히, LLM의 추론 능력과 IoT의 데이터 제공 기능 간의 시너지를 통해 복잡한 IoT 애플리케이션을 위한 최적화된 솔루션을 제공합니다.

- **Technical Details**: LLM은 Transformer 아키텍처를 기반으로 한 고급 언어 모델로, 긴 거리 의존성을 포착하고 단어 맥락을 효율적으로 이해할 수 있습니다. IoT 데이터의 비구조적 특성을 극복하는 데 LLM이 중요한 역할을 하며, 이는 자연어 처리 및 데이터 해석 능력을 통해 실시간 데이터 처리가 가능합니다. 또한, LLM과 IoT의 통합은 리소스 최적화 및 실시간 처리 강화를 통해 복잡한 IoT 시스템의 성능을 향상시키는 방법을 모색합니다.

- **Performance Highlights**: LLM과 IoT의 통합은 IoT 시스템의 실시간 감지 및 응답 기능을 크게 향상시키며, 사용자와 시스템 간의 상호작용을 원활하게 만듭니다. 이러한 통합을 통해 LLM은 라벨 없는 IoT 데이터를 주석 처리하고 의미 있는 인사이트를 제공하여 의사 결정을 지원하는 데 도움을 줍니다. 마지막으로, IoT-LLM 통합의 도전과제 및 연구 방향을 제시하여 향후 AI 및 IoT 분야에 미칠 잠재적인 영향을 규명합니다.



### MAS-Attention: Memory-Aware Stream Processing for Attention Acceleration on Resource-Constrained Edge Devices (https://arxiv.org/abs/2411.17720)
Comments:
          10 pages, 6 figures, under review for MLSys 2025

- **What's New**: 본 논문은 자원이 제한된 엣지 가속기에서 정확한 attention 추론 가속화 방법을 제안합니다. 이는 벡터 처리 장치(Vector Processing Units)와 행렬 처리 장치(Matrix Processing Units)를 활용하여 병렬로 작업을 분산하는 방식입니다. 종래의 방법들이 정해진 처리 흐름에서 각 작업을 순차적으로 실행한 것과 달리, 이 연구는 두 개의 스트림을 통해 부하 의존성을 존중하며 효율적으로 처리를 수행합니다.

- **Technical Details**: MAS-Attention은 반동기식 병렬화 전략을 사용하여 MAC 장치와 벡터 장치를 동시에 활용하며, 캐시 관리를 최적화하여 attention 추론 효율성을 극대화합니다. 다중 계층 타일링(multi-tiered tiling) 스킴을 사용하여 하드웨어 제약과 소프트웨어 파라미터를 고려하며, 세밀한 서브-매트릭스 타일링(sub-matrix tiling) 기법을 적용하여 부하를 균형 있게 분배합니다.

- **Performance Highlights**: 실험 결과, MAS-Attention은 Edge 컴퓨팅 환경에서 최신 FLAT 알고리즘에 비해 최고 2.75배의 속도 향상과 54%의 에너지 소비 감소를 달성했습니다. 실제 하드웨어에서의 테스트에서도 FLAT 대비 최고 1.76배의 성능 향상을 보였으며, 이를 통해 모델 출력의 정확도에도 영향을 주지 않았습니다.



### SlideSpawn: An Automatic Slides Generation System for Research Publications (https://arxiv.org/abs/2411.17719)
Comments:
          6 pages, 4 figures, 2 tables, 5 equations, 41 references

- **What's New**: 이 논문에서는 연구 문서의 PDF를 입력으로 받아 요약된 내용을 시각적이고 간결한 형식으로 제공하는 프레젠테이션을 생성하는 혁신적인 시스템, SlideSpawn을 제안합니다. 기존의 방법들과는 달리, 이 시스템은 연구 문서 구조의 정보를 활용하여 더 나은 품질의 프레젠테이션을 자동으로 생성할 수 있습니다. 또한, 새로운 데이터셋인 Aminer 9.5K Insights를 소개하여 자동 요약 및 프레젠테이션 생성에 활용할 수 있도록 합니다.

- **Technical Details**: SlideSpawn 시스템은 PDF 문서를 XML 형식으로 변환하여 구조적 정보를 캡처한 후, PS5K 및 Aminer 9.5K Insights 데이터셋을 기반으로 훈련된 머신 러닝 모델을 사용하여 각 문장의 중요도를 예측합니다. 중요한 문장들은 ILP(정수 선형 프로그래밍)를 통해 선택되고, 유사성을 기반으로 클러스터링하여 적절한 제목이 붙여집니다. 선택된 문장 옆에는 관련된 그래픽 요소를 배치하여 최종 슬라이드를 생성합니다.

- **Performance Highlights**: 650개의 문서 및 슬라이드 쌍에 대한 실험 결과, SlideSpawn 시스템은 기존의 방법들보다 더 나은 품질의 프레젠테이션을 생성함을 입증했습니다. 이 시스템은 중요한 텍스트 및 그래픽 요소를 효과적으로 선택하고 적절히 배치하여 연구 결과를 보다 잘 전달할 수 있도록 지원합니다. 이를 통해 연구자들은 프레젠테이션 준비에 소요되는 시간을 크게 절약할 수 있습니다.



### Hybrid Quantum Deep Learning Model for Emotion Detection using raw EEG Signal Analysis (https://arxiv.org/abs/2411.17715)
- **What's New**: 이번 연구는 EEG(data) 데이터를 이용한 감정 인식의 정밀도를 높이기 위해 하이브리드 양자(quantum) 딥러닝 기법을 제안합니다. 기존의 EEG 기반 감정 인식 기술은 노이즈와 고차원 데이터의 복잡성으로 인해 특징 추출(feature extraction)에 한계가 있었습니다. 이를 해결하기 위해 전통적인 딥러닝 분류와 양자 방식의 특징 추출을 결합하였습니다.

- **Technical Details**: EEG 데이터의 전처리(preprocessing) 단계에서 Bandpass 필터링과 Welch 방법이 사용되었습니다. 감정 상태를 결정하는 데 중요한 뇌파 패턴을 파악하기 위해 주파수 대역 파워 속성(delta, theta, alpha, beta)을 양자 표현으로 매핑합니다. 하이브리드 양자 회로(hybrid quantum circuit)에서는 얽힘(entanglement) 및 회전 게이트(rotation gates)가 사용되어 모델의 감정 인식에 대한 민감도를 극대화합니다.

- **Performance Highlights**: 테스트 데이터셋에서 평가한 결과, 제안된 모델은 감정 인식에 대한 유망한 결과를 보여주었습니다. 향후 연구에서는 실시간(real-time) 애플리케이션 및 다중 클래스 분류를 위한 모델 확장이 예정되어 있습니다. 이 방법은 전통적인 딥러닝과 양자 처리를 융합하여 감정 인식의 신뢰성과 확장성을 제공하는 잠재력을 제시합니다.



### Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations (https://arxiv.org/abs/2411.17713)
- **What's New**: 이 논문은 Meta Connect 2024 동안 커뮤니티에 오픈 소스된 Compact하고 효율적인 Llama Guard 모델인 Llama Guard 3-1B-INT4를 소개합니다. 이 모델은 리소스가 제한된 장치에서 배치가 가능하며, 일반적인 Android 모바일 CPU에서 초당 최소 30 tokens의 처리량과 2.5초 이하의 첫 번째 토큰 생성 시간을 달성할 수 있습니다. 흥미롭게도, Llama Guard 3-1B-INT4는 약 440MB로 사이즈가 7배 더 작은 데도 불구하고 Llama Guard 3-1B와 유사하거나 더 나은 안전성 모더레이션 점수를 기록합니다.

- **Technical Details**: Llama Guard 3-1B-INT4는 4비트 및 8비트 양자화를 통해 모델 압축을 달성하며, ExecuTorch 런타임 및 XNNPACK 백엔드를 활용해 ARM CPU 커널을 통해 가속화됩니다. 모델은 데이터와 함께 훈련된 사전 훈련된 LLM에서 조정되며, 비북적 대응을 위한 사전 훈련 데이터에 대한 미세 조정이 포함됩니다. Llama Guard 모델은 유저 입력과 생성 모델 출력을 기준으로 안전하거나 안전하지 않은 내용을 판별하는데 사용됩니다.

- **Performance Highlights**: Llama Guard 3-1B-INT4는 영어와 8개 비영어 언어 중 5개에 대해 F1 점수와 false positive rate(FPR)에서 Llama Guard 3-1B보다 더 좋은 성능을 보여줍니다. ExecuTorch를 통해 일반적인 모바일 장치에서 Llama Guard 3-1B-INT4의 실행 가능성도 증명됐습니다. 이 경량화된 모델은 이동형 기기에서 고속 데이터 처리를 가능하게 하여 컴퓨터의 메모리 사용량을 줄이는 데 기여합니다.



### Generative AI on the Edge: Architecture and Performance Evaluation (https://arxiv.org/abs/2411.17712)
- **What's New**: 이 논문은 6세대 통신망(6G)에서 AI 전통을 도입하여 엣지 디바이스에서 Generative AI(GenAI) 모델을 평가하는 주제를 다룹니다. 특히 Raspberry Pi와 같은 저비용 플랫폼에서 LLM(대형 언어 모델)의 성능을 정량화하는 데 초점을 맞춥니다. 원거리에 있는 AI 응용 프로그램과 대역폭에 제약이 있는 환경에서도 효율적으로 작동할 수 있는 가능성을 조명합니다.

- **Technical Details**: 이 연구에서는 Raspberry Pi 5 클러스터를 이용하여 경량화된 Kubernetes 배포(K3s)와 모듈식 프롬프트 구현을 통해 다양한 LLM을 분석합니다. LLM 추론을 위한 이용 자원, 즉 CPU 및 RAM 사용량을 최소화하고 5-12 tokens/sec의 생성을 지원하는 경량 모델(Yi, Phi, Llama3)의 성능을 집중적으로 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, 저비용 CPU만으로도 인프라에서 LLM을 효과적으로 배포할 수 있으며, 이는 6G 네트워크의 엣지 환경에서 클라우드 인프라에 의존하지 않고 로컬 인퍼런스를 가능하게 합니다. 또한, 이 연구는 가벼운 모델에 대한 성능 정량화 및 저비용 엣지 환경에서의 LLM의 잠재성을 보여주는 중요한 초석을 마련하였습니다.



### AnyECG: Foundational Models for Electrocardiogram Analysis (https://arxiv.org/abs/2411.17711)
- **What's New**: AnyECG는 전 세계의 다양한 ECG 데이터로부터 강력한 표현(feature representation)을 추출하기 위해 설계된 최첨단 모델입니다. 이 모델은 ECG Tokenizer를 통해 노이즈가 많은 ECG 신호를 의료적 의미를 가진 리듬 코드로 변환하여, 심장 이벤트를 인지하는 능력을 향상시킵니다. 연구에서는 AnyECG가 얻게 되는 일반적인 심장 관련 지식을 통해 다양한 임상 및 비임상 환경에서 우수한 성능을 보장한다는 점이 강조됩니다. 실험 결과는 이 모델이 고급 방법들보다 각 작업에서 두드러진 성과를 낸다는 것을 보여줍니다.

- **Technical Details**: AnyECG는 Transformer 아키텍처를 기반으로 하며, 특별히 설계된 Attention 모듈과 Tokenizer를 결합하여 전 자가 지도 학습(self-supervised learning) 전처리 파이프라인에 적합하게 만들어졌습니다. 신호 처리 과정에서는 0.1~75Hz 대역통과 필터를 사용하여 저주파 노이즈를 제거하고, 50Hz 노치 필터를 통해 전원선 간섭을 없앤 후, Nyquist-Shannon 샘플링 이론을 바탕으로 300Hz로 표준화하였습니다. 각 ECG 신호는 일정 기간 크기로 세분화되어 Transformer의 입력 형식에 맞춰 준비됩니다.

- **Performance Highlights**: AnyECG는 이상 탐지(anomaly detection), 부정맥 탐지(arrhythmia detection), 손상된 리드 생성(corrupted lead generation), 초장기 ECG 신호 분석(ultra-long ECG signal analysis) 등의 다양한 다운스트림 작업에서 일반화 능력을 발휘합니다. 연구에 따르면 AnyECG는 각 작업에서 기존의 최신 기술들보다 상당히 우수한 성능을 나타내며, 특히 의료 분야에서 ECG 신호의 분석과 진단을 획기적으로 개선할 수 있는 잠재력을 지니고 있습니다.



### A Composite Fault Diagnosis Model for NPPs Based on Bayesian-EfficientNet Modu (https://arxiv.org/abs/2411.17707)
- **What's New**: 이번 논문에서는 원자로 냉각 시스템, 주증기 시스템, 응축수 시스템 및 주급수 시스템의 주요 기계 구성 요소인 펌프와 밸브, 파이프라인의 결함에 중점을 두고 있습니다. 특히, Bayesian 알고리즘과 EfficientNet 대형 모델을 기반으로 하는 복합 다중 결함 진단 모델을 제안하고 있습니다. 이 연구는 데이터 기반의 딥러닝(DL) 결함 진단 기술을 활용해 원자력 발전소(NPP) 환경에서 자동 딥러닝 기반 대형 모델 기술의 효과를 평가하는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 Bayesian 알고리즘과 최신의 EfficientNet 아키텍처를 결합하여 다중 결함을 진단할 수 있도록 설계되었습니다. 데이터 기반 접근 방식을 통해 원자력 발전소의 복잡한 시스템에서 발생할 수 있는 여러 가지 결함을 정밀하게 분석할 수 있습니다. 이 연구는 전이 학습(Transfer Learning)을 활용하여 기존의 학습된 데이터를 새 환경에 효과적으로 적용하는 방법을 모색합니다.

- **Performance Highlights**: 이 모델의 성능은 원자력 발전소의 다양한 운영 상황에서 실험을 통해 검증되었습니다. Deep learning 기반의 접근 방법은 전통적인 결함 진단 방법보다 높은 정확도와 신뢰성을 보여주었습니다. 특히 다중 결함 상황에서도 뛰어난 진단 능력을 발휘하여 NPP의 운영 안전성을 크게 향상시킬 수 있는 가능성을 제시합니다.



### EEG-DCNet: A Fast and Accurate MI-EEG Dilated CNN Classification Method (https://arxiv.org/abs/2411.17705)
- **What's New**: 이번 연구에서는 EEG 기반의 Motor Imagery (MI) 분류 작업의 정확성과 효율성을 높이기 위해 EEG-DCNet이라는 새로운 다중 스케일 atrous convolutional neural network (CNN) 모델을 제안합니다. 이 모델은 $1	imes1$ convolutional 계층을 포함하며, 다중 분기 평행 atrous convolution 구조를 활용하여 EEG 신호의 비선형 특성과 다중 스케일 특징을 캡처합니다. 추가적으로 시간적 일관성을 높이기 위해 sliding window를 사용하고, 사용자 의도를 인식하는 정확도를 높이기 위해 attention 메커니즘을 적용합니다.

- **Technical Details**: EEG-DCNet은 EEG 신호에서 특징 표현 능력을 향상시키기 위해 설계되었습니다. 전통적인 2D convolutions 및 depthwise convolutions를 개선하여 비선형 특성을 효과적으로 캡처할 수 있는 $1	imes1$ convolutional 계층을 포함합니다. 이 모델은 또한 sliding window와 attention 메커니즘을 결합하여 다중 스케일 정보를 통합하고 EEG 신호의 연속적인 변화를 포착함으로써 분류 정확도를 개선합니다.

- **Performance Highlights**: EEG-DCNet은 BCI-IV-2a, BCI-IV-2b, High-Gamma 데이터셋을 포함한 여러 데이터셋으로 테스트되었으며, 기존의 최첨단(SOTA) 접근 방식보다 높은 분류 정확도 및 Kappa 점수를 기록했습니다. 더불어 EEG-DCNet은 더 적은 수의 파라미터를 요구하므로 훈련 효율성과 메모리 소비도 개선됩니다. 실험 코드는 오픈 소스로 제공되며, 이는 연구자들과 개발자들에게 중요한 자원으로 작용할 수 있습니다.




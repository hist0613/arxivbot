New uploads on arXiv(cs.CL)

### ACING: Actor-Critic for Instruction Learning in Black-Box Large Language Models (https://arxiv.org/abs/2411.12736)
- **What's New**: 이 논문에서는 블랙박스 대형 언어 모델(LLM)에서 프롬프트 최적화를 자동화하기 위해 ACING이라는 접근 방식을 제안합니다. ACING은 연속 행동 강화 학습(continuous-action Reinforcement Learning) 문제로 프레이밍되며, 비미분 보상 신호에서 학습하여 프롬프트를 최적화합니다.

- **Technical Details**: ACING은 액터-비평자(actor-critic) 기반 방법을 활용하며, 이는 제한된 API 호출을 고려하여 효율적인 탐색과 활용을 동시에 수행합니다. 내부 파라미터를 알 수 없는 블랙박스 LLM의 경우에도 잘 작동하며, 기존의 방법과 비교하여 더 나은 성능을 발휘합니다.

- **Performance Highlights**: 30개 작업에 대한 ChatGPT의 프롬프트를 최적화한 결과, ACING은 기준 방법에 비해 일관되게 개선된 결과를 보여주었습니다. ACING은 평균 10%의 점수 향상을 이뤘으며, 인간 전문가의 지침을 초과하는 최대 39%의 개선을 달성했습니다.



### Information Theory of Meaningful Communication (https://arxiv.org/abs/2411.12728)
- **What's New**: 이 논문에서는 샤논(Shannon)의 기초 작업에서 제시된 프린트된 영어의 엔트로피(Entropy)를 넘어서, 언어가 의미를 전달하는 방식에 집중하고 있습니다. 특히 정보의 단위가 문자(character)나 단어(word)가 아니라 의미 있는 부분인 절(clause)이라는 점에 주목합니다. 이를 통해 최근에 개발된 대형 언어 모델(large language models)을 활용하여 의미 있는 서사에서 전달되는 정보를 측정할 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 절(clause)을 정보의 기본 단위로 사용하여 의미를 비트(bits)로 정량화하는 방법을 제안합니다. 이를 위해 대형 언어 모델을 활용하여 텍스트의 의미를 분석하고, 확립된 절의 개념을 기반으로 정보의 전달 방식을 평가합니다. 이러한 접근 방식은 기존의 문자 기반 엔트로피 계산과는 대조적인 새로운 관점을 제공합니다.

- **Performance Highlights**: 이 연구의 결과는 의미 있는 서사가 전달하는 정보량을 정량적으로 측정하게 해줍니다. 대형 언어 모델이 의미적 정보를 효과적으로 전달하는 데 어떻게 기여하는지를 보여줍니다. 이는 인공지능(AI)과 자연어 처리(Natural Language Processing) 분야에서 커뮤니케이션의 효율성을 향상시킬 수 있는 가능성을 제시합니다.



### Scaling laws for nonlinear dynamical models of speech (https://arxiv.org/abs/2411.12720)
- **What's New**: 이 연구에서는 비선형 복원력(nonlinear restoring force)을 동적 모델에 추가하여 음성 제스처의 예측 정확도를 개선하는 방법을 제안합니다. 그러나 비선형성은 매개변수(parameter) 선택 및 수치적 안정성에서 도전 과제를 야기합니다. 이러한 문제를 해결하기 위해, 저자들은 간단한 수치적 방법들을 도입하여 비선형 동적 모델의 매개변수를 정하는 방안을 제시합니다.

- **Technical Details**: 연구의 핵심 모델은 비선형 강체 모델(cubic model)로, 복원력 항을 선형과 비선형 구성 요소로 나누어 설명합니다. 비선형 모델에서는 매개변수 d 및 k의 값에 따라 시스템이 목표 위치에 도달하는 속도와 경로가 달라진다는 점을 강조합니다. 연구에서는 Python을 사용하여 시뮬레이션을 수행하고, 복원력의 비선형성을 효과적으로 표현하기 위한 두 가지 계산적 접근법을 제시합니다.

- **Performance Highlights**: 비선형 모델은 선형 모델에 비해 임상 데이터의 특성을 더 잘 설명한다고 주장합니다. 특히, 비선형 모델은 속도 곡선의 대칭성을 가지며, 이는 선형 모델에서는 불가능한 점입니다. 저자들은 실험의 재현성을 위해 전체 시뮬레이션 과정을 위한 Python 코드를 공유하며, 이로 인해 향후 연구자들이 이 모델을 쉽게 활용할 수 있도록 하였습니다.



### Rethinking MUSHRA: Addressing Modern Challenges in Text-to-Speech Evaluation (https://arxiv.org/abs/2411.12719)
Comments:
          19 pages, 12 Figures

- **What's New**: 이 연구는 TTS(Text-To-Speech) 모델의 평가를 위한 기존 MUSHRA 테스트의 문제점을 분석하고 있습니다. 특히, 기존의 MUSHRA 방법이 현대 TTS 시스템의 인간 음성 품질을 초과하는 경우 불공정한 점수를 부여한다는 점을 강조합니다. 연구팀은 두 가지 변형된 MUSHRA 테스트를 제안하여 평가의 공정성과 신뢰성을 높이고, 47,100개의 인간 평가 데이터셋인 MANGO를 공개하여 인도 언어의 TTS 시스템 평가를 지원합니다.

- **Technical Details**: 본 연구는 MUSHRA 테스트의 신뢰성, 민감도 및 유효성을 면밀히 분석합니다. 평가 과정에서 발생할 수 있는 요소들, 예를 들어 rater variance, listener fatigue, reference bias 등을 고려하여 주요 문제를 두 가지로 요약했습니다: (i) reference-matching bias와 (ii) judgement ambiguity. 각 테스트 버전은 탄력적인 평가를 가능하게 하고 사용자 경험을 세분화하여 평가의 명확성을 향상시킵니다.

- **Performance Highlights**: 새롭게 제안된 두 가지 MUSHRA 변형은 더 나은 평가 결과를 도출하며, 이는 현대 TTS 시스템이 불공정하게 평가받지 않도록 합니다. 실험 결과, 두 변형 모두 높은 신뢰성과 세밀한 평가를 제공하고, 특히 두 번째 변형은 평가 중 특정 오류 요소를 구체적으로 식별할 수 있게 합니다. 이 연구의 결과는 TTS 기술 발전에 기여할 수 있는 여러 가능성을 보여주며, MANGO 데이터셋은 향후 연구와 개선을 위한 중요한 자원이 될 것입니다.



### Enhancing Multi-Class Disease Classification: Neoplasms, Cardiovascular, Nervous System, and Digestive Disorders Using Advanced LLMs (https://arxiv.org/abs/2411.12712)
Comments:
          7 Pages, 4 tables and 11 figures. Under review in a IEEE conference

- **What's New**: 이 연구에서는 의료 관련 사전 훈련된 언어 모델을 사용하여 다중 클래스 질병 분류를 개선하는 방법을 탐구했습니다. 비암 질환을 제외하고 4가지 특정 질병을 살펴보았으며, BioBERT, XLNet, BERT 그리고 새로운 모델인 Last-BERT를 평가했습니다. 연구 결과, BioBERT는 의료 텍스트 분류에서 97%의 정확도를 기록하며 우수한 성과를 보였고, XLNet 또한 96%의 정확도로 인상적인 결과를 보여주었습니다.

- **Technical Details**: 연구에 사용된 Medical-Abstracts-TC-Corpus는 5가지 질병(신생물, 소화기계 질환, 신경계 질환, 심혈관 질환, 일반 병리학적 상태)에 대한 데이터를 포함하고 있으며 총 14,438개의 기록이 있습니다. 일반 병리학적 상태는 분석에서 제외되었고, 데이터는 훈련 세트와 테스트 세트로 80-20 비율로 나누어 사용되었습니다. BioBERT 및 XLNet과 같은 모델은 Hugging Face 라이브러리를 통해 로드되고 데이터의 시퀀스 분류를 위해 토큰화 절차를 수행했습니다.

- **Performance Highlights**: BioBERT는 97%의 정확도로 모든 모델 중에서 가장 뛰어난 결과를 보여주었고, Last-BERT는 BERT의 89.33%에 근접한 87.10%의 정확도를 기록했습니다. 이러한 결과는 의료 텍스트 분류에 있어서 BioBERT 같은 특화된 모델뿐만 아니라 XLNet과 같은 광범위한 솔루션의 중요성을 입증합니다. 연구 결과는 공공 건강 감시 및 환자 관리에 대한 의미를 제공하며, 향후 의료 분야에서의 NLP 모델의 유용성을 나타냅니다.



### Strengthening Fake News Detection: Leveraging SVM and Sophisticated Text Vectorization Techniques. Defying BERT? (https://arxiv.org/abs/2411.12703)
Comments:
          6 pages, 3 tables and 6 Figures. Submitted to a conference

- **What's New**: 이 연구는 기계 학습과 자연어 처리(NLP)를 활용하여 가짜 뉴스를 탐지하는 새로운 방식을 제안합니다. Support Vector Machine(SVM) 및 BERT를 활용한 접근 방식을 통해, 본 연구는 TF-IDF, Word2Vec, Bag of Words(BoW)와 같은 텍스트 벡터화 기법을 비교합니다. BERT는 99.98%의 정확도를 기록했지만, SVM 모델 또한 괄목할 성과를 보여주며, 정확도가 99.81%에 달했습니다.

- **Technical Details**: 연구는 뉴스 기사를 진정한 것과 가짜로 정확하게 라벨링 할 수 있도록 SVM 기반 자연어 처리(NLP) 시스템을 구축합니다. 데이터는 ISOT 가짜 뉴스 데이터 세트를 활용하여 수집되었으며, 데이타 전처리, 텍스트 벡터화, SVM과 BERT 기반 분류를 포함합니다. 텍스트 벡터화 기법은 TF-IDF, Word2Vec와 BoW를 사용하여, 각 방법의 효과를 평가하고 SVM 분류기를 활용한 기법들을 비교합니다.

- **Performance Highlights**: BERT는 99.98%의 정확도와 F1-score 0.9998를 달성하며 뛰어난 성능을 입증했습니다. SVM 모델(BoW 벡터화 사용)은 99.81%의 정확도와 F1-score 0.9980을 기록하며, 낮은 계산 요구량에도 불구하고 강력한 성능을 보여줍니다. 연구 결과는 가짜 뉴스 탐지 시스템의 효용성을 높이는 방법을 제시하며, 다양한 텍스트 벡터화 기법의 성능을 비교하여 정보를 정확하게 배포하는 방법에 기여합니다.



### Enhanced Sign Language Translation between American Sign Language (ASL) and Indian Sign Language (ISL) Using LLMs (https://arxiv.org/abs/2411.12685)
- **What's New**: 이번 연구는 미국 수화(ASL) 사용자와 인도 수화(ISL) 사용자 간의 소통을 위한 자동 번역 시스템을 제시합니다. 이 시스템은 대규모 언어 모델(LLM)을 활용하여 ASL에서 ISL로의 실시간 번역을 가능하게 하여 접근성을 높이는 것을 목표로 하고 있습니다. 기초가 되는 기술로는 랜덤 포레스트 분류기(Random Forest Classifier)와 자연어 처리(NLP) 기법을 결합한 혁신적인 프레임워크가 사용됩니다.

- **Technical Details**: 제안된 방법론은 세 가지 주요 단계로 구성됩니다: 수화 인식, 텍스트 교정, 그리고 비디오 합성입니다. 첫 번째 단계에서는 랜덤 포레스트 분류기와 CNN(Convolutional Neural Network)을 결합한 하이브리드 모델을 사용하여 수화의 특징을 효과적으로 추출하고 분류합니다. 이후 단계에서는 인식된 텍스트를 LLM을 활용하여 교정하고, 마지막으로 ISL 제스처를 자연스럽게 생성하기 위해 비디오 합성 단계에서 모션 스무딩 기법을 적용합니다.

- **Performance Highlights**: 이 프레임워크는 ASL과 ISL 간의 언어적 차이를 극복하는 것을 목표로 하며, 실시간 프로세싱과 문화적 맥락화의 통합으로 모두가 쉽게 접근할 수 있는 소통을 돕습니다. RIFE-Net을 이용한 제스처 합성 과정은 부드럽고 자연스러운 제스처 표현을 가능하게 하며, 이는 다양한 수화 방언을 지원할 수 있는 보편적인 상호운용성을 위한 첫 걸음이 될 것입니다. 따라서 연구는 청각 장애인 커뮤니티 간의 inclusivity와 교류를 촉진할 수 있는 기술적인 진전을 나타냅니다.



### Whisper Finetuning on Nepali Languag (https://arxiv.org/abs/2411.12587)
- **What's New**: 이 연구는 자동 음성 인식(Auto Speech Recognition, ASR) 분야에서 잘 알려지지 않은 언어인 네팔어를 위한 강력한 모델 개발의 도전을 다룹니다. OpenAI의 Whisper 모델에 대한 포괄적이고 일반화된 데이터셋을 구성하고 이를 정교하게 조정하여 네팔어 전사(speech-to-text) 정확도를 향상시키기 위한 노력입니다. 또한 다양한 억양, 방언 및 말하기 스타일을 반영한 자가 기록(custom) 데이터셋을 활용하였습니다.

- **Technical Details**: 연구팀은 공공에 제공되는 ASR 데이터셋과 자가 기록된 맞춤형 데이터셋을 결합하여, 화자의 연령, 성별, 감정 등 다양한 데이터 변화를 수집하였습니다. Whisper 모델의 다양한 크기에 대해 우리 맞춤형 데이터셋으로 미세 조정(fine-tuning)을 수행하여 단어 오류율(Word Error Rate, WER)을 상당히 감소시킵니다. 특히, 오디오 및 전사 수동 큐레이션(manual curation)을 통해 음성 신호의 품질이 향상되었습니다.

- **Performance Highlights**: 우리 연구 접근법은 Fleur 데이터셋으로 학습된 Whisper의 기준 모델 대비, 소형 모델에서 36.2%, 중형 모델에서 23.8%의 WER 개선을 달성했습니다. 데이터 증강(data augmentation)이 모델의 강인성을 향상시키는 데 중요한 역할을 강조하며, 이 연구는 정확한 ASR 시스템 개발을 위한 데이터셋의 품질, 변동성 및 증강의 중요성을 부각시킵니다.



### Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models (https://arxiv.org/abs/2411.12580)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 일반화 전략을 심층적으로 연구하였습니다. 특히, LLM이 추론 과제를 수행할 때 의존하는 사전 학습 데이터의 종류를 조사하고, 이러한 데이터가 모델 출력에 미치는 영향을 분석하였습니다. 7B 및 35B 모델을 사용하여, 단순한 수학적 추론 작업과 사실적 질문에 대한 답변의 경우 문서의 영향력을 대조하였습니다.

- **Technical Details**: 연구에서는 대규모 변환기(Transformers)에 적용된 강건 통계 기법을 도입하여 사전 학습 문서가 문제 해결에 미치는 영향을 계산하였습니다. 세 가지 수학적 작업에 대한 추론 질문에 대해서는 프로세스 중심의 지식이 문서에서 중요한 영향을 미치는 것으로 밝혀졌으며, 사실적 질문에 비해 추론 질문에서는 개별 문서의 영향력이 상대적으로 약하다는 점이 확인되었습니다. 이는 모델이 비슷한 작업에 대해 절차적 지식을 학습하는 방식임을 나타냅니다.

- **Performance Highlights**: 연구 결과, 사실적 질문의 경우 답변이 중요한 문서에 자주 나타나는 반면, 추론 질문에는 거의 나타나지 않는 것으로 나타났습니다. 또한, 수학적 추론에 있어서 코드 데이터의 중요성이 강조되었으며, 이는 모델이 다양한 과제를 학습하는 데 있어 프로세스 중심의 고품질 데이터가 더 효과적일 수 있음을 시사합니다. 향후 사전 학습 전략에 대한 통찰력을 제공할 수 있는 이러한 발견은 학습의 일반화 범위를 명확히 하는 데 중요한 역할을 할 것입니다.



### Bias Free Sentiment Analysis (https://arxiv.org/abs/2411.12493)
- **What's New**: 이 논문은 SProp GNN(Semantic Propagation Graph Neural Network)이라는 감정 분석(sentiment analysis) 아키텍처를 소개합니다. 이 모델은 구문(syntactic) 구조와 단어 수준의 감정 신호만을 활용하여 텍스트의 감정을 예측합니다. 기존의 감정 분석 시스템들이 겪던 정치적 혹은 성별 편향(bias)을 견딜 수 있는 강력한 모델입니다.

- **Technical Details**: SProp GNN은 특정 단어에 대한 정보 없이 세마틱 시각을 통해 감정 예측 작업을 수행합니다. 이 모델은 VADER 및 EmoAtlas와 같은 어휘 기반(lexicon-based) 시스템에 비해 향상된 성능을 보여줍니다. 두 가지 예측 작업과 두 개 언어에서도 뛰어난 결과를 나타내며, 변환기 기반(transformer-based) 모델의 정확도에 접근할 수 있습니다.

- **Performance Highlights**: SProp GNN은 감정 예측 작업에서 편향을 줄이면서 강력한 결과를 제공합니다. 이 모델은 해석 가능성(explainability)을 개선하고, 보다 공정(fair)하고 효과적인 감정 분석 도구를 제공합니다. 텍스트를 통한 인간 행동 이해에서 유용한 역할을 할 수 있는 혁신적인 접근 방식으로 자리잡고 있습니다.



### NMT-Obfuscator Attack: Ignore a sentence in translation with only one word (https://arxiv.org/abs/2411.12473)
- **What's New**: 본 논문에서는 Neural Machine Translation (NMT) 모델에 대한 새로운 유형의 adversarial attack을 제안합니다. 이 공격은 입력 문장과 목표 문장 사이에 추가될 단어를 찾는 것으로, 이를 통해 두 번째 문장이 NMT 모델에 의해 무시되고 번역되지 않습니다. 이 방식은 공격자가 자동 번역 결과에 악의적인 정보를 숨길 수 있게 하여 실제 상황에서 매우 위험할 수 있습니다.

- **Technical Details**: 제안된 공격 방식은 'NMT-Obfuscator'로 명명되며, 입력 문장과 목표 문장을 자연스럽게 연결하는 obfuscator 단어를 이용합니다. 본 연구에서는 gradient projection을 통한 이산 최적화 문제를 해결하여 adversarial 예제를 생성하며, 예제를 자연스럽고 문법적으로 올바르게 유지하기 위해 사전 훈련된 언어 모델을 활용합니다. 광고된 예제는 원본 번역과 높은 유사성을 유지하면서도 입력 문장의 일부를 숨길 수 있는 가능성을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, 다양한 NMT 모델과 번역 작업에 대해 제안된 공격이 50% 이상의 경우에서 두 번째 부분을 무시하도록 성공적으로 유도함을 보여줍니다. 이러한 공격은 자연스러운 텍스트로 작성되어 탐지기에서 자동으로 표시되지 않을 가능성이 높기 때문에 매우 그늘성 있는 방식으로 악용될 수 있습니다.



### Guide-to-Explain for Controllable Summarization (https://arxiv.org/abs/2411.12460)
- **What's New**: 최근 강력한 성능을 보이는 대형 언어 모델(LLMs)을 활용한 추상적 요약(abstractive summarization) 연구가 주목받고 있습니다. 이 논문에서는 사용자의 특정 요구에 맞춘 제어 가능한 요약(controlled summarization)이 여전히 부족하여 LLMs의 사용성이 한정적임을 지적합니다. 이를 해결하기 위해, 제안된 가이드-설명 프레임워크(guide-to-explain framework, GTE)를 통해 LLM이 요약 초기 단계에서 잘못 정렬된 속성을 식별하고, 오류를 설명하여 잘 조정된 요약을 생성하는 방안을 제시합니다.

- **Technical Details**: GTE 프레임워크는 LLM이 생성한 요약에서의 잘못 정렬된 속성을 식별하는 단계와 모델이 자신의 오류를 설명하도록 안내하는 방식으로 구성되어 있습니다. 이 과정에서 속성 조정은 외부 모듈이나 추가 학습 없이 LLM만으로 진행됩니다. 논문에서는 다섯 가지 속성(extractiveness, length, topic, speaker)을 기준으로 LLM의 제어 가능성을 분석하고, 추상적 요약에서 LLM의 언어적 속성과 수치적 속성 제어의 어려움을 강조합니다.

- **Performance Highlights**: GTE는 혼합 속성 제어 데이터셋(MACSumDoc 및 MACSumDial)에서 평가되었으며, 최소한의 반복(iterations)만으로 각 속성을 효과적으로 제어하는 성과를 보였습니다. 평가 지표를 통해 제어된 요약이 높은 품질을 유지하고 있음을 입증하였습니다. 또한, LLM이 여러 속성을 동시에 제어하는 데 어려움이 있음을 발견했으며, 이는 연구의 의미를 더욱 강조합니다.



### Variation between Credible and Non-Credible News Across Topics (https://arxiv.org/abs/2411.12458)
Comments:
          9 pages, 1 figure

- **What's New**: 이 논문은 'Fake News'의 다양한 주제 간의 언어적 및 스타일적 차이를 분석하여 비신뢰성 있는 뉴스의 분류를 위한 보다 면밀한 접근 방식을 제안합니다. 경제, 오락, 건강, 과학, 스포츠 등 다섯 가지 주제를 중심으로 비신뢰성 뉴스의 언어적 특징을 탐구함으로써 이러한 문헌에서 발생한 상충하는 관찰을 해결하고자 합니다. 이 연구는 지식의 일반화 가능성을 높이는 데 기여하면서 독창적인 주제 기반의 ‘fake news’ 데이터 세트를 도입합니다.

- **Technical Details**: 연구는 의사소통에서의 언어적 및 스타일적 특성이 신뢰성과 비신뢰성을 가진 뉴스 간에 어떻게 다르게 나타나는지를 탐구합니다. 특히, 각 주제별로 신뢰성 있는 언론 보도와 비신뢰성 있는 언론 보도의 다양한 언어적 특징을 분석하여 자동 분류 모델에서의 적용 가능성을 평가하고 있습니다. 이 과정에서 각 뉴스 주제에 따라 특정한 특성을 파악하는 데 중점을 두며, 기존 문헌과의 일관성을 검토합니다.

- **Performance Highlights**: 결과적으로, 각 도메인에서 신뢰할 수 있는 뉴스와 비신뢰 뉴스 간의 언어적 특징이 다르게 나타나며, 이는 뉴스 분류 작업에서 스타일적 및 언어적 차이를 수용해야 함을 강조합니다. 이러한 발견은 현실 세계에서의 성능 개선을 위한 새로운 분류 접근법을 필요로 하며, 스타일적 기반의 fake news 탐지 방법의 일반성과 유용성을 제고하는 데 중점을 두고 있습니다.



### \textsc{Neon}: News Entity-Interaction Extraction for Enhanced Question Answering (https://arxiv.org/abs/2411.12449)
- **What's New**: 본 연구는 NEON 프레임워크를 제안하여, 최신 정보 추출과 이를 기반으로 하는 언어 모델의 성능 향상을 목표로 하고 있습니다. NEON은 뉴스 기사에서 드러나는 사건 및 활동 간의 상호작용을 분석하여, 엔티티 중심의 타임스탬프 지식 그래프를 구축합니다. 이를 통해 뉴스 이벤트와 관련된 질의응답(QA) 능력을 향상시키는데 기여합니다. 또한, 기존 LLMs와의 통합을 통해 실제적이고 신뢰할 수 있는 응답 생성을 지원합니다.

- **Technical Details**: NEON은 뉴스에서 엔티티 간의 상호작용을 포착하여 이를 반영하는 지식 그래프를 형성합니다. 이 그래프는 엔티티(entities)를 노드로, 이들 간의 상호작용(interactions)을 엣지로 표현합니다. 또한, 시계열적 정보 검색을 효율적으로 수행하기 위해 최적화된 인덱싱 기법을 활용합니다. NEON의 생성된 엔티티 상호작용 튜플을 LLM 프롬프트에 추가하여 임시 질의응답 생성을 수행합니다.

- **Performance Highlights**: NEON을 사용하여 3,000개의 실제 질의를 수집하고, 이 결과는 LLM의 응답 품질을 향상시킵니다. 실험 결과, NEON을 통해 얻어진 정보가 LLM의 응답 정확성과 시의성이 크게 개선되는 것으로 나타났습니다. 자동 평가 및 인간 전문가의 평가를 통해 이에 대한 검증도 수행하였으며, NEON의 통합이 실제 질문에 대한 응답 품질을 HIGH로 증진시킨 것을 입증하였습니다.



### Evaluating the Prompt Steerability of Large Language Models (https://arxiv.org/abs/2411.12405)
- **What's New**: 이번 연구에서는 다양한 가치 시스템과 문화를 반영할 수 있는 다원적 AI 모델(Pluralistic AI 모델)을 설계하기 위해 모델의 'steerability'를 평가하는 벤치마크를 제안합니다. 특히, prompt steerability를 정의하고, 모델이 다양한 페르소나를 채택할 수 있는 정도를 계량적으로 분석하고자 합니다. 현재 많은 모델이 한정된 steerability를 보이는 이유를 분석하는 것이 주된 초점입니다.

- **Technical Details**: 연구에서는 generative language model M_{\theta}와 관련된 확률적 함수 p_{\theta}를 정의하고, 사용자로부터 제시되는 prompt에 따라 모델의 동작을 평가하기 위한 score functions 집합 \mathcal{S}를 도입합니다. 평가 프로파일은 모델 출력의 joint distribution으로 정의되며, prompt-output 쌍 (x, y)을 통해 계산됩니다. 이러한 평가 프로파일을 바탕으로 모델의 steerability를 정량화하는 steerability indices를 도입하여 비교적인 측정을 가능하게 합니다.

- **Performance Highlights**: 제안된 벤치마크는 현재의 많은 모델들이 기본 동작의 편향과 여러 페르소나 차원에서의 비대칭성으로 인해 제한된 steerability를 갖고 있음을 보여줍니다. 이는 모델이 특정 행동으로 쉽게 정렬될 수 있는 정도를 정량화하고, 다른 연구에서 제안된 방법들과 비교하여 prompt 기반의 steerability에 대한 새로운 통찰을 제공합니다. 최종적으로, 제안하는 방법론은 모델의 다양한 페르소나를 채택하도록 유도하는 방법을 분석함으로써, finer-tuning 설정을 보완하는 역할을 합니다.



### Do LLMs Understand Ambiguity in Text? A Case Study in Open-world Question Answering (https://arxiv.org/abs/2411.12395)
Comments:
          Accepted at the REU Symposium at IEEE BigData 2024

- **What's New**: 자연어의 모호성은 오픈 도메인 질문 응답을 위한 대규모 언어 모델(LLMs)에게 큰 도전 과제가 되고 있습니다. 이 논문에서는 명시적인 불모호화(disambiguation) 전략의 영향을 측정하는 데 중점을 두어 기존 LLM과 few-shot LLM의 성능을 비교합니다. 실험을 통해 학습이 필요 없는 간단한 토큰 수준의 불모호화 방법이 모호한 질문 응답 작업에서 LLM 성능을 향상시키는 데 효과적임을 보여줍니다.

- **Technical Details**: 본 연구에서는 LLM의 민감성을 평가하기 위해 언어적 및 맥락적 변화가 모호한 질문 응답에 미치는 영향을 측정합니다. 세 가지의 다른 프롬프트 전략을 사용하여 LLM에서 답변을 생성하는 실험을 수행했습니다: (1) 기본 질문-응답 프롬프트, (2) 언어적 변형 추가 및 (3) 모델 내부 지식을 활용한 맥락적 보강 접근 방식입니다. 이러한 전략들을 통해 모델의 모호성 이해도를 정량적으로 측정하고 분석합니다.

- **Performance Highlights**: 실험 결과, LLM은 모호한 질문에 대해 명시적으로 불모호화된 질문 이상의 성능을 보입니다. 저자들은 1,000개의 모호한 질문에 대한 다양한 프롬프트 전략을 적용하고 이를 통한 성능 차이를 체계적으로 분석하였습니다. LLM의 성능 향상 및 모호성에 대한 이해도를 높이는 최적의 접근 방식을 제시하며, 이러한 연구 결과는 향후 AI 시스템의 발전에 기여할 수 있는 중요한 통찰력을 제공합니다.



### RedPajama: an Open Dataset for Training Large Language Models (https://arxiv.org/abs/2411.12372)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks

- **What's New**: 이번 논문은 RedPajama-V1 및 RedPajama-V2라는 두 가지 오픈 데이터셋을 소개합니다. 이 데이터셋들은 대규모 언어 모델을 위한 학습 데이터의 투명성과 접근성을 높이고, 다양한 품질 신호를 포함하여 사용자가 데이터를 필터링할 수 있도록 지원합니다. 특히, RedPajama-V2는 웹에서 수집된 원시 데이터로 구성되어 있어, 차별화된 접근 방식을 제시하고자 합니다.

- **Technical Details**: 논문에서는 RedPajama 데이터셋을 구축하는 데 사용된 절차와 원리, 즉 투명성(transparency), 규모(scale), 다용성(versatility)을 논의합니다. RedPajama-V1은 LLaMA 모델의 훈련 데이터의 오픈 복제본이며, RedPajama-INCITE 모델군을 포함합니다. 반면, RedPajama-V2는 100조 개의 원시 토큰으로 구성된 대규모 데이터셋으로, 다양한 품질 신호를 포함하여 웹 데이터 필터링을 위한 연구에 기여할 수 있습니다.

- **Performance Highlights**: RedPajama 데이터셋은 이미 Snowflake Arctic, Salesforce의 XGen과 같은 생산 모델 훈련에 사용되고 있습니다. 논문에서는 품질 신호를 활용하여 웹 데이터의 하위 집합을 효율적으로 선별하는 방법을 보여주는 일련의 분석 및 배제 연구(ablation studies)를 제공합니다. 이를 통해 RedPajama는 투명하고 고성능의 대규모 언어 모델의 개발을 촉진할 수 있는 잠재력을 지니고 있습니다.



### Balancing Accuracy and Efficiency in Multi-Turn Intent Classification for LLM-Powered Dialog Systems in Production (https://arxiv.org/abs/2411.12307)
- **What's New**: 이 논문은 대화형 AI 시스템의 다중 턴(종종 multi-turn이라고도 하는) 의도 분류(multi-turn intent classification)를 개선하기 위한 두 가지 새로운 접근 방식을 제안합니다. 첫째, Symbol Tuning을 도입하여 의도 레이블을 간소화하고, 둘째, C-LARA(Consistency-aware, Linguistics Adaptive Retrieval Augmentation)라는 프레임워크를 개발하여 대량의 다중 턴 대화 데이터를 합성합니다. 이 방법은 다국어 산업 시스템에서 낮은 자원 하에서도 확장 가능성을 높이고 비용을 절감합니다.

- **Technical Details**: Symbol Tuning은 LLM과의 상호 작용에서 의도 라벨을 압축하여 복잡성을 줄이고 성능을 개선하는 접근 방식입니다. C-LARA는 사용자로부터의 비표시 발화(unlabeled utterances)를 사용하여 다중 턴 데이터를 생성하는 효율적인 도구입니다. 이 두 가지 방법은 대화의 맥락을 고려하면서 다중 턴 의도 분류(MTIC)의 정확도를 높입니다.

- **Performance Highlights**: 이 연구는 MTIC 시스템에서 AUC 점수를 5.09% 개선하고 주석(annotations) 비용을 40% 줄이는 성과를 보여줍니다. 다국어 대화 데이터셋에 대한 실험을 통해, 제안된 방법들이 모델의 성능을 상당히 향상시키며 자원 효율성 또한 높인 것으로 나타났습니다. 이러한 결과는 낮은 자원 환경에서의 실제적인 응용 가능성을 강조합니다.



### CUE-M: Contextual Understanding and Enhanced Search with Multimodal Large Language Mod (https://arxiv.org/abs/2411.12287)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 Contextual Understanding and Enhanced Search with MLLM (CUE-M)을 소개합니다. CUE-M은 다단계 구조를 통해 이미지 컨텍스트 강화, 의도 정제, 컨텍스트 기반 쿼리 생성 등을 포함하여 멀티모달 정보 검색의 고유한 문제를 해결합니다. 또한 CUE-M은 강력한 안전 필터링 프레임워크를 통합하여 다양한 위험에 적절히 대응합니다.

- **Technical Details**: CUE-M의 파이프라인은 먼저 이미지 기반의 정보를 검색하여 관련 정보를 추출한 후, 이를 바탕으로 사용자의 쿼리를 구조화된 의도와 검색 쿼리로 정제합니다. 이 정제된 의도는 추가 쿼리나 API 호출이 필요한지를 결정하며, 최종적으로 안전 필터를 통해 관련 없는 내용을 배제하고 응답을 생성합니다. 논문에서는 Naver Knowledge-iN에서 가져온 다중 모달 Q&A 데이터 세트를 활용하여 CUE-M의 신뢰성과 안전성을 평가합니다.

- **Performance Highlights**: CUE-M은 멀티모달 쿼리에 대한 외부 지식 요구를 충족시키는 데 있어 기존의 MLLM 모델을 크게 능가하는 성능을 보였습니다. 안전 필터링 측면에서도 CUE-M은 공공 벤치마크 데이터를 사용하여 기존 모델과 유사한 성능을 발휘하면서도, 멀티모달 검색 시스템의 독특한 안전 문제를 식별했습니다. 이를 통해 CUE-M은 멀티모달 검색 시스템의 기능을 진전시켰습니다.



### Low-resource Machine Translation: what for? who for? An observational study on a dedicated Tetun language translation servic (https://arxiv.org/abs/2411.12262)
- **What's New**: 본 연구는 저자원이 언어의 기계 번역(Machine Translation, MT) 사용에 대한 관찰적 분석을 수행하여 실제 사용자 행동을 통해 필요성을 규명하였습니다. 이를 통해 Timor-Leste의 공용어인 Tetun에 대한 MT 사용 패턴을 분석하며, 연 월 70,000명이 사용하는 서비스에서 수집한 100,000개의 번역 요청을 기반으로 하였습니다. 연구 결과 사용자는 다양한 분야에서 모바일 장치를 통해 짧은 텍스트를 번역하고 있음을 보여주며, 이는 기존의 Tetun 코퍼스(corpora)와는 현저히 다릅니다.

- **Technical Details**: 저자들은 Tetun Dili(테툰 딜리)와 영어, 인도네시아어, 포르투갈어 간의 자동 번역이 가능한 전문 서비스의 서버 로그 데이터를 활용하였습니다. 100,000개의 로그에서 타임스탬프, 번역 입력, MT 출력, 출발 언어 코드 및 요청된 목표 언어 코드, 장치 운영 체제 정보를 수집하였습니다. 도메인(genre) 분석은 Latent Dirichlet Allocation(LDA)을 사용하여 텍스트의 주제를 식별하였으며, 최종적으로 15개의 주제로 분류하였습니다.

- **Performance Highlights**: 이번 연구는 관찰적 데이터가 저자원 언어 기술 개발에 어떻게 기여할 수 있는지를 보여줍니다. 저자들은 MT 시스템이 저자원 언어로 번역하는 데 중점을 두고 다양한 교육 관련 분야를 포괄해야 한다고 제안합니다. 특히, 연구 결과는 MT의 사용이 기존의 텍스트 코퍼스와 달리 짧은 입력을 효과적으로 처리하고 있어, 저자원 언어의 기술적 요구가 어떤지를 파악하는 데 중요한 의미를 지닙니다.



### Predicting User Intents and Musical Attributes from Music Discovery Conversations (https://arxiv.org/abs/2411.12254)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문은 음악 도메인에서의 사용자 의도 분류(Intent Classification) 작업을 처음으로 다룹니다. 음악 발견 대화(conversational music retrieval) 맥락에서 8개의 인터페이스 제어 레이블과 15개의 음악 속성 레이블을 예측하는 두 가지 음악 전용 의도 분류 작업을 도입했습니다. 또한, 이전 대화 기록을 현재 사용자 쿼리와 결합하는 방법을 제안하여 모델이 전체 대화 맥락을 더 잘 이해할 수 있도록 했습니다.

- **Technical Details**: 연구에서는 의도 분류를 사용자 의도와 음악 속성으로 나누어 사용하였고, 희소 표현(sparse representation), 단어 임베딩(word embedding), DistilBERT, 그리고 Llama 모델을 사용하여 작업을 수행했습니다. 특별히 DistilBERT는 40%의 모델 크기를 줄이면서도, 97%의 언어 이해 능력을 유지하며 60% 더 빠른 속도를 제공합니다. 우리는 또한 DistilBERT의 파라미터를 미세 조정(fine-tuning)한 모델과 분류기만 학습시키는 프로빙(probing) 모델을 비교했습니다.

- **Performance Highlights**: 미세 조정된 DistilBERT 모델은 사용자 의도 및 음악 속성 분류 모두에서 다른 모든 모델보다 뛰어난 성능을 보였습니다. 특히, 음악 속성 분류 성능이 크게 향상되어(F1 스코어 0.46에서 0.72로 증가) 모델이 음악 지식을 효과적으로 습득했음을 보여주었습니다. 일반-purpose Llama 모델은 음악 도메인에 특화된 데이터로 미세 조정된 모델에 비해 성능이 낮았으며, 이는 음악 도메인에 대한 충분한 지식이 부족함을 나타냅니다.



### Evaluating Tokenizer Performance of Large Language Models Across Official Indian Languages (https://arxiv.org/abs/2411.12240)
- **What's New**: 이 연구는 12개의 대형 언어 모델(LLM)의 인도 공식 언어 22개에 대한 토크나이저(tokenizer)의 성능을 비교 평가합니다. 특히, SUTRA 토크나이저가 다른 모든 모델을 초월하여 14개 언어에서 뛰어난 성능을 보여주었음을 밝혀냈습니다. 또한 이 논문은 인도 언어 처리에 있어 타겟팅된 토크나이저 전략의 필요성을 강조합니다.

- **Technical Details**: 대부분의 LLM은 두 가지 토크나이저 알고리즘, 즉 WordPiece와 Byte Pair Encoding (BPE)을 사용합니다. 이 연구는 Normalized Sequence Length (NSL)라는 지표를 사용하여, 각 모델의 토크나이저 성능을 평가하고, 다양한 언어를 처리하는 데 필요한 효율성을 분석합니다. 12개의 모델을 대상으로 하였고, 각 언어의 원주율 문자로 예제 텍스트를 작성하여 평가하였습니다.

- **Performance Highlights**: SUTRA 토크나이저는 평균 NSL 값에서 최고 성과를 기록했으며, 이를 통해 인도 언어 처리에서의 우수성을 입증했습니다. 특히, SUTRA는 ChatGPT 4-o 및 기타 인도 표시 모델보다 뛰어난 결과를 보여주었고, 14개 언어에서 가장 높은 성과를 달성했습니다. 이러한 결과는 LLM의 다국어 및 인도 중심 모델에서의 토크나이저의 중요성을 다시 한번 부각시킵니다.



### A Combined Encoder and Transformer Approach for Coherent and High-Quality Text Generation (https://arxiv.org/abs/2411.12157)
- **What's New**: 이번 연구는 BERT의 의미 해석 강점과 GPT-4의 생성 능력을 결합한 새로운 텍스트 생성 모델을 소개합니다. 이 모델은 일관성 있고 맥락적으로 정확한 언어를 생성하는 높은 기준을 세우고 있습니다. 이러한 결합 아키텍처를 통해 의미 깊이를 강화하고 매끄럽고 인간적인 텍스트 흐름을 유지하며, 이전 모델들의 한계를 극복합니다.

- **Technical Details**: BERT-GPT-4 모델은 Perplexity와 BLEU와 같은 주요 지표에서 GPT-3, T5, BART, Transformer-XL, CTRL 등 전통적인 모델들을 초월하는 실험적 벤치마크 결과를 보여줍니다. 이 혼합 모델은 맥락 정보를 완전히 활용함으로써 논리적으로 일관된 텍스트를 생성하고, 인간 언어 패턴에 밀접하게 일치하도록 합니다.

- **Performance Highlights**: 이 연구는 의미 이해와 고급 생성 모델을 통합할 수 있는 잠재력을 강조하며, NLP 분야에 새로운 통찰을 제공합니다. 또한, 자동 글쓰기, 질문-답변 시스템, 적응형 대화 에이전트와 같은 광범위한 응용 프로그램을 위한 대규모 생성 아키텍처의 기초를 설정합니다.



### HNCSE: Advancing Sentence Embeddings via Hybrid Contrastive Learning with Hard Negatives (https://arxiv.org/abs/2411.12156)
- **What's New**: 본 논문에서는 HNCSE(Hard Negative Contrastive Sentence Learning)라는 새로운 대조 학습 프레임워크를 제안합니다. 이 접근법은 기존 SimCSE를 확장하여 긍정 및 부정 샘플 모두의 학습을 향상시키는 데 중점을 두고 있습니다. HNCSE의 주요 특징은 하드 네거티브 샘플을 효과적으로 사용하여 문장의 의미적 깊이를 더하는 것입니다. 실험 결과, HNCSE는 의미 텍스트 유사성(semantic textual similarity) 및 전이 작업(transfer task) 데이터셋에서 뛰어난 성능을 보였습니다.

- **Technical Details**: HNCSE모델은 하드 네거티브 샘플 믹싱 기법을 바탕으로 문장 표현 학습을 개선하는 혁신적인 훈련 방식을 적용합니다. 이 모델은 하드 네거티브의 특성을 포함하여 잘못 식별된 긍정 샘플을 수정하고, 기존 하드 네거티브를 늘려 더 튼튼한 학습 상황을 조성합니다. 대조 학습 목표는 동일한 문장에 대한 두 가지 다른 뷰 간의 일치를 극대화하고, 서로 다른 문장 간의 일치를 최소화하는 것을 기반으로 합니다. 이는 라벨이 없는 데이터 환경에서 문장 표현을 학습하는 효과적인 방법입니다.

- **Performance Highlights**: HNCSE는 SimCSE를 기반으로 한 다양한 작업에 대해 개선된 성능을 보여주며, 하드 네거티브 샘플을 믹싱하여 긍정 샘플의 품질을 향상시키고 있습니다. HNCSE는 대규모 언어 모델(LLMs) 및 현재의 SOTA(State-Of-The-Art) 벤치마크에 비해 의미 텍스트 유사성 작업에서 뚜렷한 장점을 발휘합니다. 이러한 성과는 문장 표현 학습에 있어 하드 네거티브의 중요성을 강조하며, 더 나은 의미적 이해에 기여하고 있습니다.



### CoMeDi Shared Task: Models as Annotators in Lexical Semantics Disagreements (https://arxiv.org/abs/2411.12147)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문은 CoMeDi Shared Task의 결과를 제시하고 있으며, 다수결 예측(Subtask 1)과 주석자 간의 불일치(Subtask 2)를 처리하는 접근 방식을 설명합니다. 모델 앙상블 전략을 MLP 기반 및 임계값 기반 방법과 결합하여 사전 훈련된 언어 모델에 대해 학습합니다. 개별 모델을 가상 주석자로 간주하고, 지속적인 유사성 점수와 이산 분류 레이블을 포함하는 집계 방법을 설계하여 주석 프로세스를 시뮬레이션했습니다.

- **Technical Details**: 우리는 두 개의 하위 작업을 Gaussian 분포의 평균 및 분산이라는 두 가지 통계적 특성에 대응하는 것으로 개념화합니다. MLP 기반 및 임계값 기반 접근 방식을 사용하여 지속적인 유사성 점수 및 이산 분류 레이블을 생성하고, 임베딩 공간의 기하학적 편향을 완화하기 위해 비등방성 제거 기법을 추가합니다. 또한 다양한 모델 앙상블 전략을 적용하여 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 특히 Subtask 2에서 우리의 방법의 효과가 입증되었습니다. 관심을 받은 점은, 동일 모델 내에서조차 지속적인 유사성 점수가 집계된 이산 레이블보다 인간의 불일치 패턴과 더 잘 일치한다는 것입니다. 이러한 발견은 주석자 불일치를 캡처하는 데 있어 유사성 점수의 집합이 이산 레이블보다 더 유리하다는 것을 강조합니다.



### A Computational Method for Measuring "Open Codes" in Qualitative Analysis (https://arxiv.org/abs/2411.12142)
- **What's New**: 이 논문에서는 질적 분석(valitative analysis)의 중요한 방법론인 open coding의 잠재적인 편향을 체계적으로 측정하고 식별하는 새로운 계산 방법을 제안합니다. 기존 연구들은 open coding의 결과를 정확하게 측정하지 않아 편향의 위험을 증가시켜 왔습니다. 이 방법은 Grounded Theory와 Thematic Analysis 이론에 기반하여 인간과 기계 코더 간의 팀 기반 접근 방식을 활용합니다.

- **Technical Details**: 제안된 방법은 두 가지 HCI 데이터셋을 사용하여 open 코드의 신뢰성을 측정하는 것입니다. 이 과정에서 Coverage, Density, Novelty, Divergence라는 네 가지 개념 지표를 운영화하여 코드 스페이스를 평가합니다. 이러한 지표는 팀 코더의 결과에 대한 개별 코더의 결과를 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 이 연구는 인간 분석과 비교하여 제안된 방법의 결과의 안정성을 통계적으로 분석함으로써 신뢰성을 검증합니다. 기계 측정과 인간 해석을 결합한 결과, 질적 연구자들이 GAI를 유도 분석에서 사용하는 데 도움이 될 수 있는 근거 기반의 제안과 예제 워크플로우를 제시합니다.



### Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods (https://arxiv.org/abs/2411.12103)
Comments:
          9 pages, 2 figures

- **What's New**: 이번 논문은 대형 언어 모델(LLM)에서 해로운 정보를 제거하는 방법인 LLM 언러닝(LLM unlearning)을 제안하고, LLMU와 RMU 두 가지 방법을 비교 분석합니다. 실험을 통해 이들 방법이 모델의 일반적 능력에 미치는 영향을 평가하였으며, 새로운 생물학 벤치마크도 사용하여 효과를 점검했습니다. 이 논문은 기존 방법의 평가 범위를 확장하고, 단순한 프롬프트 방식이 언러닝 성능에 미치는 영향을 조사했습니다.

- **Technical Details**: LLMU는 해로운 데이터에 대해 손실 함수의 음수를 취해 그래디언트 상승(gradiant ascent)을 통해 학습하며, RMU는 해로운 데이터에 대해 무작위 제어 벡터에 맞추어 특정 층을 학습시킨다. 저자들은 생물학 중심 데이터셋을 사용하고, 5-shot 프롬프트 방식 등 다양한 프롬프트 전략으로 샘플을 변경하여 언러닝의 효과를 개선했음을 보여줍니다. 실험 결과에 따르면 이 방식이 언러닝 벤치마크의 정확성을 최대 1750% 향상시킬 수 있음을 증명했습니다.

- **Performance Highlights**: 결과적으로 RMU 방법이 LLM의 일반적인 능력 보존이 더 우수한 경향을 보였으며, 임의의 데이터로 재학습할 경우 언러닝 전 성능을 거의 완전히 복구할 수 있음을 확인했습니다. 이는 LLM 언러닝 방법이 실제로는 해로운 정보를 완전히 제거하지 못함을 시사합니다. 전체적으로, 이 연구는 언러닝 방법의 강건성과 효과에 대해 새로운 통찰을 제공합니다.



### Mitigating Gender Bias in Contextual Word Embeddings (https://arxiv.org/abs/2411.12074)
- **What's New**: 이 논문에서는 기존의 정적(Static) 단어 임베딩에서 발생하는 성 편향을 줄이기 위해 새로운 Masked-Language Modeling (MLM) 목표 함수를 제안합니다. 제안된 방법은 성 편향을 완화하면서도 다운스트림 작업의 성능을 유지할 수 있도록 설계되었습니다. 또한 문맥(Contextual) 임베딩에서의 편향 측정과 관련된 새로운 평가 지표들을 제안하여 편향 완화에 대한 목적에 부합합니다.

- **Technical Details**: 연구팀은 성 편향을 줄이기 위한 실험 분석을 진행하며, Masked-Language Modeling의 새로운 목표 함수를 포함하여 두 가지 전략을 통해 문맥적 모델의 학습을 지속 가능한 방향으로 진행합니다. 제안된 방법은 모든 명사를 성 중립적 단어로 간주하는 방식으로, 문장에서 무작위로 명사를 마스킹하고 나머지 토큰을 이용하여 예측하는 방식입니다. 두 번째 전략에서는 성 속성이 있는 단어를 마스킹하고 나머지 단어들을 사용하여 예측하도록 훈련함으로써 두 가지 분리된 클러스터 문제를 해결하려고 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 문맥적 임베딩의 성 편향을 크게 줄이는 데 성공하였으며, 성 중립적 단어를 통해 성별 정보의 노출을 최소화하여 다운스트림 작업에서의 성능도 유지합니다. 추가적으로, 기존 데이터의 불균형적인 참조를 해결하기 위해 성 예측 작업을 통해 데이터를 지능적으로 증강하는 전략을 도입했습니다. 이러한 접근 방식이 성 편향을 줄이는 데 효과적임을 입증하는 실증적 증거를 제공합니다.



### Benchmarking pre-trained text embedding models in aligning built asset information (https://arxiv.org/abs/2411.12056)
- **What's New**: 이 논문은 구축 자산(data related to built assets) 정보를 전문 용어에 맞춰 잘 정렬하기 위한 새로운 접근 방식을 제안합니다. 특히 최근의 대형 언어 모델을 활용한 텍스트 임베딩(text embedding)이 자산 관리의 데이터 매핑(data mapping) 과정의 자동화를 모색하는 데 기여할 수 있음을 보여줍니다. 이 연구는 기존 모델의 성능을 비교하고, 구축 자산에 특화된 기술 용어의 복잡한 의미를 효과적으로 표현할 수 있는지를 평가합니다.

- **Technical Details**: 이 연구는 다양한 하위 도메인에 걸쳐 구축 제품에 대한 정보의 체계적인 데이터를 구성하고, 이를 기반으로 구체적 작업 세트를 개발합니다. 데이터는 산업 기초 클래스(Industry Foundation Classes, IFC)와 유니클래스(Uniclass)의 두 가지 주요 출처에서 수집되었습니다. 또, 제안된 데이터 세트는 여섯 가지 작업, 즉 클러스터링(clustering), 검색(retrieval), 재순위화(reranking)를 평가하여 모델간의 성능을 비교하는 데 중점을 둡니다.

- **Performance Highlights**: 이 연구의 평가 결과는 자동화된 데이터 매핑 과정에서 구성 자산 정보를 정렬하는 데 있어 기존 언어 모델의 강점과 한계를 안내합니다. 현재 관련된 24개의 텍스트 임베딩 모델을 사용하여 10,000개 이상의 데이터 항목을 포함하는 가장 포괄적인 벤치마크를 제공합니다. 또한 연구 결과 및 데이터 세트는 공개 소스로 제공되어 후속 연구의 기초 자료로 사용될 수 있도록 하였습니다.



### ByteScience: Bridging Unstructured Scientific Literature and Structured Data with Auto Fine-tuned Large Language Model in Token Granularity (https://arxiv.org/abs/2411.12000)
- **What's New**: ByteScience는 과학 데이터를 자동으로 정리하기 위한 비영리 클라우드 기반 플랫폼으로, 최신의 자동 미세 조정(automatic fine-tuning) 언어 모델(LLM)을 활용합니다. 이 플랫폼은 Amazon Web Services(AWS) 위에 구축되어 간편한 사용자 UI를 제공하며, 몇 개의 잘 주석이 달린 논문으로도 높은 정확도를 달성할 수 있습니다. 이 혁신적인 도구는 과학 문헌에서 구조화된 지식 및 데이터를 추출하는 과정을 간소화하여 자연 정보학(natural informatics)의 발전에 기여하고자 합니다.

- **Technical Details**: ByteScience는 AWS Sagemaker를 활용하여 강력하고 확장 가능한 클라우드 기반 솔루션을 제공합니다. 이 시스템은 초기 설정에서 도메인 별 데이터셋 구성을 통해 과학적 데이터를 수집하고, 해당 데이터셋에 대해 대규모 언어 모델을 미세 조정하여 성능을 최적화합니다. 사용자는 JSON, PDF, HTML 또는 XML 형식의 과학 문서를 업로드하고, 이를 통해 구조화된 데이터를 생성할 수 있는 간편한 워크플로우를 경험할 수 있습니다.

- **Performance Highlights**: ByteScience 플랫폼은 LLM을 활용해 사람의 개입을 줄이며 주석 달기 시간을 평균 57% 단축시키는 효과를 보이고 있습니다. 연구 결과, ByteScience는 배터리, 촉매 및 태양광 분야의 90개 샘플을 분석한 결과, 기존 방법보다 높은 정밀도를 기록하며 구조화된 데이터 추출 성능에서 탁월함을 나타냈습니다. 데이터 추출 성공률이 증명되며, Thomas와 같은 연구자들이 이 플랫폼을 활용하여 복잡한 과학 논문을 대규모로 처리할 수 있는 사례를 보여주고 있습니다.



### Understanding Chain-of-Thought in LLMs through Information Theory (https://arxiv.org/abs/2411.11984)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)에서 체인 오브 생각(Chain-of-Thought, CoT) 추론을 공식화하는 새로운 정보를 제공합니다. 기존의 CoT 평가 방법들이 주석 데이터에 의존하던 반면, 저자들은 정보 수치 이론(information theory)을 기반으로 한 접근법을 제안합니다. 이 방법은 각 추론 단계에서 '정보 이득(information gain)'을 측정하여, LLM에서의 실패 유형을 식별할 수 있게 합니다.

- **Technical Details**: 제안된 프레임워크는 LLM 생성 접근법을 정보 이론의 관점에서 기술합니다. 먼저, 초기 상태와 작업을 정의하고 이를 통해 업데이트된 상태를 설명합니다. 이 프레임워크는 각 추론 단계에서 정보 이득을 정량화하며, 이는 적절한 정보가 최종 결과 예측에 기여해야 한다는 인식에 기반합니다. 이를 통해 주석 데이터 없이도 각 하위 작업의 성능을 평가할 수 있는 알고리즘을 제시합니다.

- **Performance Highlights**: 제안된 방법은 Toy 데이터 및 GSM-8K 데이터 세트를 통해 폭넓은 실험을 거쳐 효과성을 입증했습니다. 이메일 내의 진행 과정에서 기존의 결과 기반 방법들에 비해 정확한 모델 성능을 제공하여, CoT 추론의 실패 모드를 효과적으로 식별합니다. 최종적으로, 이 연구는 LLM의 성능 평가 방식에서 중요한 변화를 예고하며, 연구자들에게 더 나은 인사이트를 제공합니다.



### Neurosymbolic Graph Enrichment for Grounded World Models (https://arxiv.org/abs/2411.12671)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 반응 능력을 활용하여 복잡한 문제를 해결하고 깊이 있는 맥락적 의미를 해석하는 새로운 접근 방식을 제안합니다. 이는 이미지 입력을 시작으로 자연어 설명을 생성하며 이를 추상 의미 표현(AMR) 그래프 형태에 변환하여 다층적인 지식 그래프를 구성합니다. 이러한 접근법은 비구조적 언어 모델과 형식적 의미 구조 사이의 간극을 메우고, AI 시스템이 인간과 유사한 추론을 할 수 있도록 개선합니다.

- **Technical Details**: 제안된 방법론은 이미지 입력에서 출발하여 최신 LLM을 활용하여 자연어 설명을 생성하고, 이를 AMR 그래프로 형식화하여 지식 그래프를 구성하는 것을 핵심으로 합니다. 이 과정에서 implicit knowledge를 활용하여 발생하는 다양한 의미적 패턴과 맥락적 지식을 추출하고, 이를 기반으로 LLM에서 생성된 지식을 강화합니다. 이러한 방법론은 다층적인 지식 그래프의 지속적인 확장 및 적응을 가능하게 하는 피드백 루프를 구현하여, 새로운 맥락에 대한 이해를 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구에서 제시한 시스템은 자연어 이해, 시각적 추론, 복잡한 문제 해결 작업에서 뛰어난 성과를 보여줍니다. LLM과 지식 그래프의 융합을 통해 기존 방법보다 지식 기반 풍부화 과정이 더 민첩하고 확장 가능해졌습니다. 실험 결과는 이 접근 방식이 여러 지식 도메인에서 효율성을 증명함을 나타내며, 이는 AI 시스템이 인간과 유사한 맥락적 이해를 할 수 있는 새로운 길을 열어줍니다.



### Optimizing Airline Reservation Systems with Edge-Enabled Microservices: A Framework for Real-Time Data Processing and Enhanced User Responsiveness (https://arxiv.org/abs/2411.12650)
Comments:
          22 pages, 11 figures

- **What's New**: 항공사 예약 시스템의 복잡성이 증가함에 따라, 본 논문에서는 새로운 접근 방식을 채택하여 신속하고 효율적인 예약 시스템 개발을 위한 스마트 솔루션을 제시합니다. 특히, 기존의 중앙 집중식 아키텍처의 단점을 해결하기 위해 에지 컴퓨팅 마이크로서비스를 구현하는 개념적 프레임워크를 상세히 설명합니다. 이를 통해 사용자의 가까운 곳에서 좌석 재고 확인 및 예약 과정 등을 수행할 수 있어 시스템의 반응 시간을 단축시킬 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 Kubernetes에 의해 조정되는 배포된 분산 컴퓨팅 마이크로서비스와 실시간 메시지 처리 시스템인 Kafka를 기반으로 하여 탄력적 확장을 가능하게 합니다. 또한, 자원 모니터링 및 관리를 위한 Prometheus와 Grafana와 같은 운영 구성 요소가 포함되어 있어 모든 운영 프로세스가 최적화됩니다. 이러한 기술적 요소들은 저지연(low latency), 고처리량(high throughput) 및 향상된 사용자 경험을 목표로 하고 있습니다.

- **Performance Highlights**: 본 연구는 항공 산업의 서비스 제공 방식에 변화를 주어 고객 만족도를 향상시킴과 동시에 설치 비용이 저렴하고 인공지능(artificial intelligence) 및 사물인터넷(internet of things) 통합 시스템과 같은 기술 변화에 효과적으로 대응할 수 있는 인프라를 제공합니다. 새로운 기술과 현대적인 분산 및 실시간 중심 시스템에 대한 수요를 반영하며, 향후 케이스 적용 및 테스트를 위한 기반을 마련합니다. 따라서, 제안하는 아키텍처는 기존의 항공사 예약 시스템이 겪고 있는 문제를 해결하기 위한 시장 친화적이고 확장 가능한 솔루션을 제공합니다.



### DLBacktrace: A Model Agnostic Explainability for any Deep Learning Models (https://arxiv.org/abs/2411.12643)
- **What's New**: 이번 연구에서는 DLBacktrace라는 새로운 기술을 소개합니다. 이는 다양한 신경망 구조에서 모델 결정을 투명하게 하는 방법으로, Multi Layer Perceptron (MLP), Convolutional Neural Network (CNN), Large Language Models (LLM) 등 여러 도메인에서 효과적으로 작동합니다. DLBacktrace는 해석 가능성 (interpretability)의 필요성을 강조하며, AI 시스템의 신뢰를 구축하고 책임성을 보장하는 데 기여합니다.

- **Technical Details**: DLBacktrace는 모델-비의존적 방법으로, 출력에서 입력으로 관련성을 추적하여 각각의 계층에서의 중요도 점수를 부여합니다. 이 방법은 PyTorch와 TensorFlow에서 구현된 다양한 모델 아키텍처와 호환되며, 정량적인 메트릭을 사용하여 기존 해석 가능성 방법들과 비교한 벤치마킹 결과를 제시합니다. 심층 학습에서 모델의 투명성을 개선하기 위해 설계된 이 기술은 다양한 데이터 형식에 적용할 수 있습니다.

- **Performance Highlights**: DLBacktrace는 LIME, SHAP, Grad-CAM 등과 같은 기존 해석 가능성 방법들과 비교될 때 독창적이고 신뢰할 수 있는 해석을 제공합니다.  전반적인 모델 설계 및 데이터 타입을 아우르는 두 가지 분석 방식 (지역적 및 전역적 해석)을 통해 모델의 명확성을 높입니다. 또한 오픈 소스로 제공되어 다양한 연구 및 산업 응용에서 적극 활용될 수 있는 가능성을 가지고 있습니다.



### Leveraging Virtual Reality and AI Tutoring for Language Learning: A Case Study of a Virtual Campus Environment with OpenAI GPT Integration with Unity 3D (https://arxiv.org/abs/2411.12619)
Comments:
          5 pages, 2 tables, 8 figures

- **What's New**: 이 논문에서는 인도어(Hindi) 다국어 학습을 위한 새로운 접근 방식을 제시하고 있습니다. 가상현실(Virtual Reality, VR) 환경과 인공지능(AI) 튜터링 시스템의 통합을 통해 사용자에게 몰입감 있는 학습 경험을 제공합니다. OpenAI의 GPT API를 활용하여 제작된 이 시스템은 사용자가 가상 캠퍼스 환경에서 원활하게 언어를 학습할 수 있도록 돕습니다.

- **Technical Details**: 제작된 가상 환경은 Unity를 사용하여 우리 대학의 11층 건물을 상세하게 표현하였으며, 문화 및 기술 활동의 중심지입니다. 여기에는 사용자의 움직임에 따라 함께 이동하는 OpenAI의 GPT 모델 기반 AI 튜터가 포함되어 있습니다. 이 시스템은 음성을 텍스트로, 텍스트를 텍스트로 변환하고, 텍스트를 음성으로 변환하는 기능을 활용하여 실시간 언어 학습 지원을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 VR 기술과 AI 기반 튜터링을 결합하여 실시간 상호작용이 가능한 몰입형 언어 학습 경험을 제공하는 성과를 보여줍니다. 사용자는 가상 환경 내에서 AI 튜터와 상호작용하며 인도어 학습을 효과적으로 수행할 수 있습니다. 이러한 접근법은 언어 번역 및 학습 지원을 포함하여 다국어 학습의 새로운 가능성을 탐색합니다.



### Large Language Models for Combinatorial Optimization of Design Structure Matrix (https://arxiv.org/abs/2411.12571)
- **What's New**: 이 연구에서는 제안된 새로운 LLM(대형 언어 모델) 기반 프레임워크가 공학적 조합 최적화(combinatorial optimization) 문제에 적용될 수 있는 가능성을 탐구합니다. 특히, 이 방법은 네트워크 토폴로지(network topology)와 분야 지식(domain knowledge)을 통합하여 일반적인 CO 문제인 디자인 구조 매트릭스(Design Structure Matrix) 순서를 최적화합니다. 이 연구는 LLM이 제공하는 맥락적 지식을 통해 실제 공학 문제에 접근하는 새로운 패러다임을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 생성(generative) 및 추론(reasoning) 능력을 활용하여 최적화를 수행합니다. 초기 솔루션을 무작위로 샘플링하고, 평가자는 미리 정의된 기준에 따라 솔루션의 품질을 정량화합니다. 이 과정을 거쳐 LLM은 네트워크 정보와 분야 지식을 포함한 프롬프트(prompts)를 기반으로 새로운 솔루션을 생성합니다. 이 메커니즘은 피드백 루프(feedback loops)를 최소화하는 DSM 순서 최적화를 목표로 합니다.

- **Performance Highlights**: 실험 결과, LLM 기반 방법이 벤치마크 방법에 비해 빠른 수렴 속도와 높은 솔루션 품질을 달성함을 보여주었습니다. 특히 맥락적 분야 지식을 포함할 경우 성능이 더욱 향상되었으며, 다양한 DSM 사례를 통해 이러한 효과가 입증되었습니다. 이러한 결과는 LLM이 복잡한 실제 조합 최적화 문제를 해결하는 데 유용할 수 있음을 강조합니다.



### Predicting Customer Satisfaction by Replicating the Survey Response Distribution (https://arxiv.org/abs/2411.12539)
- **What's New**: 본 논문에서는 고객 만족도(CSAT) 예측을 위한 새로운 접근 방식을 제안합니다. 기존에는 조사에 응답한 일부 고객의 데이터만 기반으로 평균 CSAT를 계산하여 발생하던 편향을 감소시키기 위한 모델이 개발되었습니다. 이 방법은 실제 생산 환경에서도 모든 통화에 대해 고객 만족도를 예측할 수 있도록 하여 더 정확한 성과 지표를 제공합니다.

- **Technical Details**: 연구에서는 자주 업데이트 되는 머신 러닝 모델의 클래스 비율 변화를 방지하기 위해 제어 메커니즘을 도입합니다. 이 메커니즘은 샘플링 노이즈에 의한 위험을 완화하고, ASR(Automated Speech Recognition) 데이터에서 고객 만족도의 분포를 정확하게 복제할 수 있도록 최적화된 결정을 제공합니다. 이 방법은 다중 클래스와 순서 분류 문제에서 사용할 수 있으며, 클래스 불균형을 개선하는데 기여합니다.

- **Performance Highlights**: 실험에 사용된 데이터는 892,000개의 통화 기록으로, 모델은 높은(4 이상) 또는 낮은(3 이하) CSAT 예측을 위해 이진 출력으로 작동합니다. 모델의 정확도는 85% 이상이며, 7번의 시험 과정을 진행한 결과 배포된 모델과 시뮬레이션 간에 성능 차이가 없음을 확인했습니다. 이 연구는 고객 만족도를 반영하기 위한 포괄적인 머신 러닝 파이프라인의 일환으로 적용되어 실제 환경에서도 강건한 성과를 발휘할 것으로 기대됩니다.



### Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues (https://arxiv.org/abs/2411.12537)
- **What's New**: 최근 연구에 따르면,  Linear Recurrent Neural Networks (LRNNs)와 같은 모델들이 Transformers의 효율적인 대안으로 떠오르고 있지만, 상태 추적(state-tracking)에서 어려움을 겪고 있습니다. 기존의 LRNN들은 간단한 패리티(parity) 문제도 해결하지 못하며, 이 문제는 대각선 매트릭스의 고유값(eigenvalues)에 제약이 있기 때문입니다. 이 논문에서는 LRNN의 고유값 범위를 확대하여 패리티 문제를 해결하고 성능을 개선할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: LRNN은 구조화된 상태 전이 매트릭스를 사용하여 상태를 업데이트하며, 이 매트릭스의 구조가 LRNN의 표현력을 결정짓습니다. 본 연구는 대각선이 아닌 LRNN의 경우에도 동일한 고유값 제한이 적용됨을 보여주며, 복소수 고유값이 모듈로 카운팅(modular counting) 문제 해결에 필요하다는 것을 증명합니다. 우리는 고유값 범위를 [-1, 1]로 확장하여 LRNN의 표현력을 증가시키는 간단하면서도 강력한 솔루션을 제안합니다.

- **Performance Highlights**: 실험 결과, Mamba와 DeltaNet 모델의 고유값 범위를 확장함으로써 패리티 문제 해결뿐 아니라 상태 추적 성능이 일관되게 향상되었습니다. 또한, 이러한 수정된 LRNN들은 언어 모델링에서 원래 모델과 비교할 수 있는 성능을 보여주었으며, DeltaNet은 코드 및 수학 데이터셋에서 개선된 성능을 나타냈습니다. 이러한 개선이 학습 및 추론 비용을 변화시키지 않으면서도 가능하다는 점이 주목할 만합니다.



### Regular-pattern-sensitive CRFs for Distant Label Interactions (https://arxiv.org/abs/2411.12484)
- **What's New**: 이 논문에서는 사용자가 지정한 패턴 내에서 긴 거리의 레이블 상호작용을 학습할 수 있도록 표준 선형 체인 조건부 랜덤 필드(CRF)를 강화한 방법인 정규 패턴 민감 CRF(RPCRF)를 제안합니다. 이를 통해 사용자는 모델이 고려해야 하는 상호작용 유형을 간결하게 지정할 수 있는 정규 표현(regular-expressions)을 작성할 수 있으며, 데이터로부터 이 패턴들이 발생하는 맥락을 학습할 수 있습니다.

- **Technical Details**: RPCRF는 추가적인 비지역적 잠재 변수(non-local potentials)로 강화된 CRF로 해석될 수 있으며, 사용자 지정 패턴 집합으로부터 자동으로 구성되는 방법을 상세히 설명합니다. 이 모델은 특이한 패턴 집합에 대해도 정확한 훈련과 추론이 가능하여, 기존의 가중 유한 상태 변환기(weighted FST)보다 계산적 요구 사항이 낮습니다. 이 접근 방식은 레이블 시퀀스 내 비지역적 의존성 구조를 포착하는 다양한 패턴의 효과성을 평가하는 데 초점을 맞춥니다.

- **Performance Highlights**: RPCRF는 합성 데이터(synthetic data)에서 효과적으로 작동하며, 다른 유형의 패턴이 레이블 시퀀스 내의 비지역적 의존성 구조를 어떻게 포착하는지를 보여줍니다. 이 모델은 전통적인 CRF와 비교할 때 장거리 상호작용을 더 잘 학습할 수 있는 가능성을 제공합니다. 이러한 접근 방식은 자연어 처리(NLP) 및 기타 순차적 데이터 처리 작업에서의 응용 가능성을 열어줍니다.



### Analysing Explanation-Related Interactions in Collaborative Perception-Cognition-Communication-Action (https://arxiv.org/abs/2411.12483)
Comments:
          4 pages, 3 figures, published as a Late Breaking Report in RO-MAN 2024

- **What's New**: 이 연구는 AI 로봇이 인간 팀원과 효과적으로 협력하기 위해 어떤 종류의 설명 능력이 필요한지를 분석합니다. 실험 참가자들 간의 의사소통을 통해 인간이 팀워크에서 기대하는 설명 유형을 분류하여 설명 가능성에 대한 인식을 심화하고자 합니다. 또한 대화 중심의 접근 방식으로 메시지의 관계를 분석하며, 이러한 커뮤니케이션이 Task 성공과 어떤 관계가 있는지를 규명합니다.

- **Technical Details**: 측정된 실험은 TeamCollab라는 시뮬레이션 환경에서 이루어졌으며, 여기서 참가자들은 위험한 물체를 식별하고 목표 지역으로 옮기는 역할을 수행합니다. 연구는 2~4명의 인간 에이전트 그룹으로 구성되어 있으며, 의사소통은 웹 기반의 근접 텍스트 채팅 시스템으로 기록됩니다. XAI(Explainable AI) 프레임워크를 사용하여 2,607개의 메시지를 분석하며, 메시지 유형을 상호작용 기술 세분화에 따라 분류하였습니다.

- **Performance Highlights**: 팀 커뮤니케이션이 성과에 미치는 영향을 평가한 결과, 전반적인 메시지 수와 수집된 물체 수 간의 역비례 관계를 확인했습니다. 반면, 실제로 위험한 물체는 메시지 수에 의존하지 않음을 보였고, 이는 참가자 간의 의사소통을 통해 얻은 지식이 향상된 예측으로 이어짐을 나타냅니다. 이는 AI 로봇이 인간과의 효과적인 의사소통을 통해 신뢰를 얻고 성과를 향상시킬 수 있는 가능성을 시사합니다.



### A Layered Architecture for Developing and Enhancing Capabilities in Large Language Model-based Software Systems (https://arxiv.org/abs/2411.12357)
- **What's New**: 이번 논문은 Large Language Models (LLMs)의 활용 범위를 기본 언어 작업을 넘어서 확장하기 위한 최근의 노력을 다루고 있습니다. LLM의 일반화 가능성과 유연성이 폭넓은 채택을 가능하게 했으나, 애플리케이션 개발의 변화하는 요구는 그들의 본래 능력을 초과하는 경우가 많습니다. 이를 해결하기 위해 다양한 방법, 예를 들어 추론의 온도 조정(inference temperature adjustments) 또는 창의성을 유도하는 프롬프트(prompts)가 필요하다는 것을 강조합니다.

- **Technical Details**: 논문에서는 LLM 소프트웨어 시스템 개발을 특정 속성으로 정의된 개별 계층으로 구성하는 계층적 아키텍처(layered architecture)를 제안합니다. 이러한 계층에 맞는 능력을 정렬함으로써, 프레임워크는 효과적이고 효율적인 방식으로 기능과 품질을 지원하는 능력의 체계적인 구현을 촉진합니다. 이를 통해 개발자에게 LLM 기반 소프트웨어 시스템 개발에 적합한 기술을 선택할 수 있는 실행 가능한 통찰력을 제공합니다.

- **Performance Highlights**: 실제 사례 연구를 통해 프레임워크의 유용성을 입증하며, 성능의 강건성(robustness)과 확장성(scalability)을 증진시킬 수 있는 방법을 제시합니다. 다양한 개발 선택이 엔지니어링 복잡성(engineering complexity), 확장성, 운영 비용(optimal operational costs) 간의 균형을 맞추는 데 어떻게 기여하는지를 설명합니다.



### Building Trust: Foundations of Security, Safety and Transparency in AI (https://arxiv.org/abs/2411.12275)
- **What's New**: 이 논문은 공개적으로 사용 가능한 AI 모델의 생태계가 빠르게 발전하는 양상을 탐구하고, 이러한 변화가 보안(security) 및 안전(safety) 분야에 미치는 잠재적 영향을 다룹니다. AI 모델이 점점 더 널리 사용됨에 따라, 이 모델들이 가진 리스크(risk)와 취약점(vulnerability)을 이해하는 것이 중요합니다. 보안 및 안전 시나리오를 검토하며, 모델 라이프 사이클(lifecycle)과 소유권(ownership) 프로세스의 부재와 같은 도전 과제를 강조하고 있습니다.

- **Technical Details**: 현재의 보안 및 안전 시나리오에 대한 검토와 함께, 모델 개발자(developer)와 최종 사용자(end-users)를 위한 보안 및 안전을 향상시키기 위한 포괄적인 전략을 제안합니다. 이 논문은 AI 모델의 개발 및 운영에 있어 더 표준화된 보안, 안전 및 투명성(transparency)을 위한 기초적 요소들을 제공하는 것을 목표로 합니다. AI 모델의 생태계와 관련된 여러 커뮤니티를 형성하는 과정에서 중요한 기초 자료를 제시합니다.

- **Performance Highlights**: 이 연구는 AI 모델의 공개 생태계가 증가함에 따라 발생할 수 있는 다양한 위험 요소들을 관리하고, 이러한 모델들을 보다 안전하고 신뢰할 수 있게 만드는 방법을 탐구합니다. 제안된 전략들은 AI 모델 사용자와 개발자에게 안전한 환경을 제공하기 위한 것으로, 현대 AI 생태계에서의 보안과 안전 이슈를 해결하기 위한 필수적인 기반이 될 것입니다.



### BoolQuestions: Does Dense Retrieval Understand Boolean Logic in Language? (https://arxiv.org/abs/2411.12235)
Comments:
          Findings of the Association for Computational Linguistics: EMNLP 2024

- **What's New**: 이번 연구는 Boolean Dense Retrieval (BDR)이라는 새로운 개념을 소개하고, 이를 통해 현재의 dense retrieval 시스템들이 언어에서 암시된 Boolean 논리를 이해할 수 있는지를 조사하고자 합니다. 연구팀은 이에 대한 기준 데이터셋인 BoolQuestions를 구축하여, 기본적인 Boolean 논리를 포함하는 복잡한 쿼리를 포함했습니다. 이를 통해 현재의 dense retrieval 시스템들이 Boolean 논리의 이해에 여러 한계가 있음을 발견했습니다.

- **Technical Details**: Boolean Dense Retrieval(BDR) 태스크의 목표는 현재의 retrieval 시스템들이 Boolean 논리를 얼마나 잘 이해하는지를 평가하는 것입니다. 연구팀은 NOT 질문에 대한 추가적인 훈련 데이터를 생성하고, 자연어에서의 논리적 NOT을 이해하는 수준을 높이기 위해 대조적 지속 학습 방식(contrastive continual learning baseline)을 제안했습니다. 이러한 방법은 retrieval 시스템에서 반환된 패시지 목록의 부정적인 비율을 줄이는데 기여했지만, 정확도는 다소 희생되기도 했습니다.

- **Performance Highlights**: 실험 결과, 현재의 dense retrieval 모델들은 NOT 질문과 같은 복잡한 Boolean 논리를 처리하는 데 상당한 성능 저하를 보였습니다. 이는 전통적인 Lexical 기반 모델들이 Boolean 논리를 효과적으로 처리할 수 있는 반면, dense retrieval 시스템들이 이를 잘 반영하지 못하고 있음을 알 수 있습니다. 이러한 발견은 앞으로의 dense retrieval 시스템의 발전에 중요한 기초 자료로 작용할 것입니다.



### Just KIDDIN: Knowledge Infusion and Distillation for Detection of INdecent Memes (https://arxiv.org/abs/2411.12174)
- **What's New**: 이번 연구에서 제안한 KID-VLM 프레임워크는 큰 비주얼 언어 모델(LVLM)로부터의 지식 증류(Knowledge Distillation, KD)와 상식 지식 그래프(Knowledge Graph, KG)에서의 지식 주입(Knowledge Infusion)을 결합하여 공격적인 메메의 독성 탐지 성능을 향상시킵니다. 이 프레임워크는 ConceptNet이라는 대규모 상식 KG에서 서브 지식 그래프를 추출하여 compact VLM 프레임워크 내에서 주입하여 독성 문구와 메메 내의 시각적 개념 간의 관계적 맥락을 강화합니다.

- **Technical Details**: KID-VLM은 CLIP을 백본으로 사용하여 메메의 시각적 및 텍스트 피쳐를 추출하며, LLaVA 교사 모델로 생성된 캡션을 위한 텍스트 인코더와 메메 텍스트를 위한 또 다른 텍스트 인코더를 사용합니다. 피쳐 상호작용 매트릭스(Feature Interaction Matrix, FIM)를 계산하여 시각적 및 텍스트 데이터를 정렬하고, 일관성 손실(consistency loss)을 사용하여 교사 모델의 지식을 정제하여 학생 모델이 내재된 맥락적 단서를 포착하도록 학습합니다.

- **Performance Highlights**: 두 개의 증오 발언 기준 데이터셋에서 평가한 결과, KID-VLM 프레임워크는 AU-ROC, F1, 리콜에서 각각 1.1%, 7%, 35% 향상된 성능을 보여주며 기존 최첨단 모델들을 초월하는 성과를 기록했습니다. 이러한 성과는 두 가지 맥락 학습 접근법의 필요성을 강조하며, LVLM으로부터의 잠재적 패턴과 KG로부터의 외적 관계 지식을 함께 캡처합니다.



### Reviving Dormant Memories: Investigating Catastrophic Forgetting in Language Models through Rationale-Guidance Difficulty (https://arxiv.org/abs/2411.11932)
Comments:
          Working in progress

- **What's New**: 이 논문에서는 지속적 학습에서의 재해적 망각(catastrophic forgetting) 문제를 해결하기 위해, 모델이 제공된 부분적인 합리적 근거(rationale)를 수동으로 수용할 때 잊혀진 과제에 대한 성능이 회복될 수 있음을 발견했습니다. 또한, 원래 지침에 과제 비특이적 접두사(task-agnostic prefix)를 추가함으로써 모델이 능동적으로 적절한 근거를 생성하여 정답에 도달할 수 있음을 보여주는 실험 결과를 제시했습니다.

- **Technical Details**: 저자들은 'Rationale-Guidance Difficulty' 메트릭을 제안하여 주어진 지침이 모델에 적절한 근거를 생성하도록 얼마나 효과적으로 안내하는지를 평가합니다. 이 메트릭을 활용하여 재생 기반 지속적 학습 알고리즘에서 재생 데이터의 할당을 최적화하는 방식을 적용했습니다. 실험 결과는 이 데이터 할당 방식이 재해적 망각을 효과적으로 완화하고 다양한 모델에서 더 나은 플라스틱성을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: 다양한 크기의 모델에 대한 실험에서, 적절한 근거의 부분을 제공하면 잊혀진 과제에 대한 모델의 성능이 회복됨을 확인했습니다. 게다가, 비특정 접두사를 추가함으로써 모델이 관련 지식을 생성하는 데 도움이 되었고, 잊혀진 과제에서의 성능이 부분적으로 복구되었습니다. 이러한 결과는 모델이 과제 관련 지식의 실질적인 손실이 아니라 원래 지침이 적절한 근거를 생성하는 데 실패한 것에서 주로 기인함을 증명합니다.



### AIGS: Generating Science from AI-Powered Automated Falsification (https://arxiv.org/abs/2411.11910)
Comments:
          Pre-print. 35 pages. Official website: this https URL

- **What's New**: 본 논문에서는 $	extbf{AI-Generated Science}$ (AIGS)를 탐구합니다. 이는 에이전트가 독립적으로 연구 프로세스를 완전히 완료하고 과학 법칙을 발견하는 시스템을 의미합니다. 기존 시스템들이 검증 엔진에 크게 의존하는 한편, AIGS는 내재된 'falsification'을 통해 새로운 과학적 발견을 독립적으로 진행하는 방식으로 설계되었습니다.

- **Technical Details**: 우리는 Baby-AIGS라는 다중 에이전트 시스템을 제안하여 연구 프로세스의 핵심 역할을 담당하는 에이전트를 포함합니다. 이 시스템은 FalsificationAgent를 도입하여 가능성 있는 과학적 발견을 식별하고 검증함으로써 명시적인 'falsification'을 실현합니다. 이를 통해 독립적으로 연구를 수행할 수 있는 첫 걸음을 내딛습니다.

- **Performance Highlights**: 세 가지 작업에 대한 실험 결과, Baby-AIGS는 의미 있는 과학적 발견을 생성할 수 있음을 보여주었습니다. 그러나 경험이 풍부한 인간 연구자와 비교할 때 접근성은 여전히 낮은 상태입니다. 마지막으로, 현재 Baby-AIGS의 한계와 연구의 개선 가능성 및 관련 윤리적 문제에 대해 논의합니다.



### Deploying Large Language Models With Retrieval Augmented Generation (https://arxiv.org/abs/2411.11895)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 생성 능력이 환각(hallucination)이나 비사실적 응답을 생성하는 경향으로 인해 저해되는 문제를 다루고 있습니다. 연구자들은 LLM의 훈련 세트 외부의 사실 데이터에 기반하여 생성된 출력을 접지하는 방법에 주목하고 있으며, 정보 검색을 위한 Retrieval Augmented Generation (RAG) 방안을 제시합니다. 연구팀은 실제 데이터로 LLM과 RAG 통합의 현장 테스트를 통해 이 기술의 적용 기회와 도전을 분석하고 있습니다.

- **Technical Details**: 이 논문에서는 LLM이 비정형 문서에서 정보를 검색할 수 있도록 하는 다양한 접근방법을 설명하고 있으며, 토픽에 맞춘 미세 조정(fine-tuning) 및 프롬프트 엔지니어링(prompt engineering) 기법을 강조합니다. Retrieval-Augmented Generation (RAG) 방식은 기존 문서의 임베딩(embeddings)을 활용하여 LLM의 프롬프트를 향상시키는 접근입니다. 이러한 방식을 통해 도메인 특정 맥락을 추가하여 회수된 정보를 LLM의 생성 응답에 결합하게 됩니다.

- **Performance Highlights**: 연구자들은 RAG 시스템의 도입에 있어 여러 도전 과제와 산업 표준에 부합한 시스템 설계를 강조합니다. 이로 인해 RAG 방안의 실질적인 적용이 LLM 연구와 산업 간의 격차를 해소하는 데 기여할 수 있을 것으로 기대합니다. 특히, 이 논문은 LLM과 RAG을 통합한 파일럿 프로젝트의 실행과 평가에 대한 인사이트를 제공하며, 향후 정보 시스템(IS) 분야의 행동 연구에서 이 기술을 효과적으로 활용할 수 있는 방안을 제안합니다.



### Exploring Optimal Transport-Based Multi-Grained Alignments for Text-Molecule Retrieva (https://arxiv.org/abs/2411.11875)
Comments:
          BIBM 2024 Regular Paper

- **What's New**: 생명정보학 분야에서 본 연구는 텍스트 기반의 분자 검색 작업에 대한 새로운 접근 방식을 제안합니다. 우리가 제안한 Optimal TRansport-based Multi-grained Alignments 모델(ORMA)은 텍스트 설명과 분자 구조 간의 다중 정렬을 가능하게 하여 연구자들이 적합한 분자를 식별하는 데 도움을 줍니다. ORMA는 텍스트 인코더와 분자 인코더를 갖추고 있으며, 분자의 하위 구조에서 세부 정보를 포착하여 정확한 검색을 돕습니다.

- **Technical Details**: ORMA의 핵심 혁신은 Optimal Transport(OT) 기법을 활용하여 토큰과 모티프의 정렬을 수행하는 것입니다. 이 모델은 각 입력 텍스트 설명을 처리하여 토큰과 문장 수준의 표현을 생성하고, 분자는 계층적인 이질 그래프로 모델링되어 원자, 모티프, 분자 노드를 포함합니다. 또한, 다양한 수준에서의 대조 학습을 통해 텍스트-원자, 다중 토큰-모티프, 문장-분자 간의 정렬을 세밀하게 조정합니다.

- **Performance Highlights**: 실험 결과, ORMA는 ChEBI-20과 PCdes 데이터셋에서 기존의 최첨단(SOTA) 모델에 비해 월등한 성능을 보여주었습니다. 특히 ChEBI-20에서 텍스트-분자 검색의 Hits@1 점수가 66.5%로, 기존 SOTA인 AMAN을 17.1% 초과했고, 분자-텍스트 검색에서는 61.6%로 AMAN보다 15.0% 우수한 성적을 기록했습니다.



### Chat Bankman-Fried: an Exploration of LLM Alignment in Financ (https://arxiv.org/abs/2411.11853)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 금융 분야에서 윤리적 및 법적 기준을 준수하는지 평가하기 위한 실험적 프레임워크를 제안합니다. 다양한 내부 및 외부 요인에 따라 고객 자산을 악용할 의사가 있는지를 테스트하는 새로운 설정을 도입하였습니다. 연구 결과, LLM들은 주어진 환경에서 비윤리적 행동을 보이는 경향이 상당히 다양하며, 이는 경제 이론에 의해 예측된 바와 일치합니다.

- **Technical Details**: 실험 시나리오는 2022년 크립토 자산 거래소 FTX의 붕괴에서 영감을 받았습니다. LLM이 CEO로 작용하는 가상의 금융회사를 모델링하여, 고객 자산을 이용해 내부 손실을 메우는 유혹에 처하는 결정을 내리도록 하였습니다. 실험은 약 54,000회 시뮬레이션을 수행하며, 다양한 압력 변수를 조정하여 LLM의 비윤리적 행동 가능성을 분석합니다.

- **Performance Highlights**: 연구 결과, 윤리적 행동을 저해하는 요인은 위험 회피, 이윤 기대치 및 규제 환경 등이 있으며, LLM에 따라 이러한 결과의 영향은 다르게 나타납니다. 실험적 프레임워크는 금융 당국 및 기관이 LLM 채택에 따른 리스크를 더 잘 이해하고 측정하는 데 도움이 될 수 있습니다. 연구팀은 이들을 위한 코드와 벤치마크 데이터를 GitHub에 공개할 예정입니다.



### MLAN: Language-Based Instruction Tuning Improves Zero-Shot Generalization of Multimodal Large Language Models (https://arxiv.org/abs/2411.10557)
- **What's New**: 본 논문에서는 다중 모드 대형 언어 모델(Multimodal Large Language Models)의 제로샷 태스크 일반화를 개선할 수 있는 새로운 언어 기반의 커스터마이징 기법인 Mlan을 제안합니다. 기존의 주로 비주얼 지시에 의존하던 커스터마이징 메커니즘과는 달리, 저자들은 언어 기반의 접근법을 통해 보다 효율적으로 모델을 훈련할 수 있는 방법에 집중하였습니다. 이 방법은 언어만으로도 강력한 제로샷 일반화 능력을 갖출 수 있도록 돕습니다.

- **Technical Details**: Mlan의 접근법은 다중 모드 대형 언어 모델(MLLM)이 언어 지시를 활용해 보지 못한 데이터셋에 일반화할 수 있도록 설계되었습니다. 이 과정에서 기존의 커스터마이징 방법론에서 사용하는 세 가지 단계(훈련 데이터 선택, 데이터 형식화, 모델 훈련 및 평가)를 따르면서도, 이미지 대신 언어에 기반하여 훈련 효율성을 크게 높였습니다. 특히, 모델이 언어 지시를 따르는 능력을 강화한 후, 이를 비주얼 지시에 전이할 수 있게 하여 훈련 효율성을 높이는 데 기여하였습니다.

- **Performance Highlights**: 실험 결과, 언어 전용 커스터마이징 방법이 언어 및 비주얼 태스크 모두에서 평균 15.4%의 성능 향상을 이뤄냈습니다. 또한, 해당 방법은 비주얼 태스크에서 기존 프리트레인된 모델보다 10.4% 높으며, 언어 태스크는 19.1% 향상된 결과를 보였습니다. 마지막으로, 저자의 Mlan 방법은 훈련 효율성을 4배 이상 향상시켰고, 기존의 다중 모드 커스터마이징 접근법들과 비교했을 때 경쟁력 있는 성능을 입증하였습니다.



New uploads on arXiv(cs.IR)

### PseudoSeer: a Search Engine for Pseudocod (https://arxiv.org/abs/2411.12649)
- **What's New**: 논문에서는 PseudoSeer라는 새로운 pseudocode 검색 엔진을 소개합니다. 이 엔진은 Elasticsearch를 활용하여 연구 논문에서 pseudocode를 효과적으로 검색할 수 있도록 설계되었습니다. PseudoSeer는 제목, 초록, 저자 정보 및 LaTeX 코드 조각과 같은 논문의 다양한 요소를 기반으로 한 고급 검색 기능을 지원합니다.

- **Technical Details**: PseudoSeer는 연구 논문의 제목, 초록, 저자 정보 및 LaTeX 코드 섹션 등에서 검색할 수 있는 기능을 제공합니다. 분석된 데이터셋은 arXiv에서 제공된 다양한 논문으로, pseudocode의 인덱싱 및 검색을 최적화하기 위해 구조를 갖춘 방식으로 저장되고 처리됩니다. 검색 엔진의 성능은 BM25 기반의 가중치 순위 알고리즘을 통해 향상됩니다.

- **Performance Highlights**: PseudoSeer는 특정 알고리즘 구현 논문을 찾거나 특정 데이터 구조의 활용을 탐색하는 데 유용합니다. 사용자는 정확한 일치를 요구하는 쿼리 기능을 사용할 수 있으며, 결과는 사용자의 검색 요구에 맞게 정렬됩니다. 이 시스템은 코드 중심의 문헌 탐색과 함께 학술적 자료에 대한 보다 효율적인 접근을 제공하여 연구자들에게 중요한 도구로 자리잡을 것입니다.



### Towards Unifying Feature Interaction Models for Click-Through Rate Prediction (https://arxiv.org/abs/2411.12441)
- **What's New**: 본 연구에서는 광고 시스템의 클릭률(CTR) 예측을 효과적으로 수행하기 위한 새로운 프레임워크인 IPA(Interaction, Pooling, Aggregator)를 제안합니다. 기존의 모델들이 개별적으로 존재할 수밖에 없었던 한계를 극복하기 위해 이 프레임워크를 통해 feature interaction을 체계적으로 통합하였습니다. IPA는 세 가지 핵심 컴포넌트로 구성되어 있으며, 이들 각각의 선택이 CTR 예측 모델 성능에 미치는 영향을 심도 있게 분석합니다.

- **Technical Details**: IPA 프레임워크의 첫 번째 구성 요소인 Interaction Function은 임베딩 벡터의 명시적 상호작용을 추출하여 벡터 형태로 결과를 반환합니다. 두 번째 요소인 Layer Pooling은 이전 레이어와 원시 feature embedding을 기반으로 보다 높은 수준의 interaction 레이어를 구성합니다. 마지막으로, Layer Aggregator는 모든 레이어의 출력을 결합하여 이후의 분류기에 입력으로 사용합니다.

- **Performance Highlights**: 광범위한 실험 후, IPA 프레임워크를 기반으로 한 새로운 모델들이 기존의 state-of-the-art 모델들과 경쟁력 있는 성능을 보이는 것을 발견했습니다. 또한 Tencent 광고 플랫폼에서의 A/B 테스트를 통해 PFL 모델이 상당한 GMV 상승 효과를 보였으며, 여러 주요 시나리오에서 실제 배포 모델로 사용되고 있습니다. 이 연구는 CTR 모델 설계에 대한 인사이트를 제공하여 향후 연구 및 상용 모델 개발에 기여할 것으로 기대됩니다.



### Scalable and Effective Negative Sample Generation for Hyperedge Prediction (https://arxiv.org/abs/2411.12354)
Comments:
          11

- **What's New**: 이번 연구에서는 하이퍼엣지 예측(hyperedge prediction)을 위한 새로운 프레임워크인 SEHP(Scalable and Effective Negative Sample Generation for Hyperedge Prediction)를 제안했습니다. SEHP는 확산 모형(diffusion models)을 활용하여 고품질의 부정 샘플을 생성하는 데 중점을 두고 있으며, 이를 위해 경계 인식 손실 함수(boundary-aware loss function)를 설계하였습니다. 이 프레임워크는 긍정 샘플을 사용해 부분 하이퍼그래프(sub-hypergraphs)를 형성하고, 효율적인 배치 프로세스를 가능하게 합니다.

- **Technical Details**: SEHP는 부정 샘플의 품질을 향상시키기 위해 각 샘플을 결정 경계(decision boundary)로 점진적으로 이동시키는 경계 인식 손실 함수를 적용합니다. 이 과정은 이진 분류 작업에서의 훈련 성과 개선에 필수적입니다. 더 나아가, SEHP는 연속 공간(continuous space)에서 하이퍼엣지를 필요로 하는 이산 공간(discrete space)으로의 전환 문제를 해결하기 위해 잠재 공간(latent space) 표현을 직접 활용하여 효율성을 극대화하고 속도를 20배에서 71배 향상시켰습니다.

- **Performance Highlights**: SEHP는 광범위한 실험을 통해 정확도, 효율성 및 확장성에서 기존 최첨단 방법들을 초월하는 성과를 보였습니다. 특히, 여러 데이터 세트에서 AUROC(Area Under the Receiver Operating Characteristic curve) 및 Precision에서의 성능 감소는 각각 0.59%와 0.46%에 불과하여 우수한 결과를 나타냅니다. 이 접근법은 하이퍼엣지 예측 분야에서 확장 가능하고 효과적인 부정 샘플 생성의 새로운 기준을 제시하고 있습니다.



### Consistency Regularization for Complementary Clothing Recommendations (https://arxiv.org/abs/2411.12295)
- **What's New**: 이번 연구에서는 Consistency Regularized Bayesian Personalized Ranking (CR-BPR) 모델을 개발하였으며, 기존의 보완 의류 추천 방법의 한계점인 일관성과 편향 학습 문제를 해결하고자 하였습니다. 패션 선호도가 주관적이며, 보통 다른 보완 제품과 함께 조화롭게 제시된다는 점을 강조하고 있습니다. CR-BPR 모델은 사용자 선호 및 제품 매칭 모델링을 통합하고, 각 측면에 대해 일관성 정규화를 특별히 강조합니다.

- **Technical Details**: CR-BPR 모델은 협업 필터링 기법을 통합하여 사용자 선호도와 제품 매칭을 동시에 반영합니다. 모델은 다중 모달 데이터의 균형 문제를 해결하기 위해 특징 스케일링 프로세스를 포함하고 있으며, 이로 인해 기능 스케일에 의해 왜곡되지 않도록 설계되었습니다. 두 개의 일관성 정규화 브랜치가 사용자 선호도 및 제품 매칭 관점에서 역사적 선택과의 유사성을 측정합니다.

- **Performance Highlights**: CR-BPR 모델의 효과는 두 가지 벤치마크 데이터셋, 즉 IQON3000 및 Polyvore-519를 통한 상세한 분석을 통해 검증되었습니다. 연구 결과, 제안된 접근 방식이 기존 모델을 현저하게 초월함을 확인하였습니다. 이를 통해 CR-BPR 모델이 최신 기술 수준의 성능을 갖추었음을 입증하였습니다.



### BoolQuestions: Does Dense Retrieval Understand Boolean Logic in Language? (https://arxiv.org/abs/2411.12235)
Comments:
          Findings of the Association for Computational Linguistics: EMNLP 2024

- **What's New**: 이번 연구는 Boolean Dense Retrieval (BDR)이라는 새로운 개념을 소개하고, 이를 통해 현재의 dense retrieval 시스템들이 언어에서 암시된 Boolean 논리를 이해할 수 있는지를 조사하고자 합니다. 연구팀은 이에 대한 기준 데이터셋인 BoolQuestions를 구축하여, 기본적인 Boolean 논리를 포함하는 복잡한 쿼리를 포함했습니다. 이를 통해 현재의 dense retrieval 시스템들이 Boolean 논리의 이해에 여러 한계가 있음을 발견했습니다.

- **Technical Details**: Boolean Dense Retrieval(BDR) 태스크의 목표는 현재의 retrieval 시스템들이 Boolean 논리를 얼마나 잘 이해하는지를 평가하는 것입니다. 연구팀은 NOT 질문에 대한 추가적인 훈련 데이터를 생성하고, 자연어에서의 논리적 NOT을 이해하는 수준을 높이기 위해 대조적 지속 학습 방식(contrastive continual learning baseline)을 제안했습니다. 이러한 방법은 retrieval 시스템에서 반환된 패시지 목록의 부정적인 비율을 줄이는데 기여했지만, 정확도는 다소 희생되기도 했습니다.

- **Performance Highlights**: 실험 결과, 현재의 dense retrieval 모델들은 NOT 질문과 같은 복잡한 Boolean 논리를 처리하는 데 상당한 성능 저하를 보였습니다. 이는 전통적인 Lexical 기반 모델들이 Boolean 논리를 효과적으로 처리할 수 있는 반면, dense retrieval 시스템들이 이를 잘 반영하지 못하고 있음을 알 수 있습니다. 이러한 발견은 앞으로의 dense retrieval 시스템의 발전에 중요한 기초 자료로 작용할 것입니다.



### Sparser Training for On-Device Recommendation Systems (https://arxiv.org/abs/2411.12205)
- **What's New**: 본 논문에서는 SparseRec라는 경량화된 임베딩 방법을 제안하며, Dynamic Sparse Training (DST) 패러다임의 한계를 해결한다. 이 방법은 Nonnegative Matrix Factorization (NMF)을 사용하여 마스크 매트릭스를 초기화하며, 비활성 파라미터를 효과적으로 활성화할 수 있는 전략을 마련하고, 밀집 그래디언트를 피하면서 메모리 사용량을 줄인다.

- **Technical Details**: SparseRec는 마스크 매트릭스의 초기화를 NMF를 통해 수행하여 불필요한 파라미터를 줄이고, 여러 배치에 대한 그래디언트를 누적하여 비활성 파라미터를 재활성화하는 방식을 사용한다. 이 방식을 통해 모델 성능을 높이는 파라미터를 식별하고, 프로세스 전반에 걸쳐 희소성을 유지하면서도 메모리 효율성을 개선한다.

- **Performance Highlights**: SparseRec는 세 가지 주요 추천시스템과 결합하여 다양한 최신 경량 임베딩 알고리즘과 성능을 비교하였고, GNN 기반 추천시스템에서 우수한 효과를 보여준다. 이 연구는 경량화된 추천 시스템의 발전에 기여하며 메모리 제약이 있는 환경에서도 효율적인 성능을 유지할 수 있는 가능성을 제시한다.



### Multi-Grained Preference Enhanced Transformer for Multi-Behavior Sequential Recommendation (https://arxiv.org/abs/2411.12179)
Comments:
          12 pages

- **What's New**: 이번 논문에서는 Multi-Grained Preference enhanced Transformer (M-GPT) 프레임워크를 제안하여 Multi-Behavior Sequential Recommendation (MBSR)의 성능을 개선하고자 합니다. M-GPT는 상호작용 수준의 그래프를 구축하여 다중 행동 의존성을 모델링하고, 다중 시간 척도에서 행동 인지 다중 선호를 캡처하는 혁신적인 트랜스포머 아키텍처를 제공합니다. 이 연구는 다중 행동 간의 복잡한 관계를 잘 학습할 수 있는 방법을 찾아 기존의 한계점을 극복하고 있습니다.

- **Technical Details**: M-GPT는 두 가지 주요 구성 요소인 Interaction-Level Dependency Extractor (IDE)와 Multifaceted Sequential Pattern Generator (MSPG)를 포함합니다. IDE는 사용자 역사 데이터에서 행동 간의 상호작용 수준의 다중 행동 의존성을 모델링하기 위해 설계된 그래프 구조를 사용합니다. MSPG는 다양한 시간 척도에서 행동 인식 다중 선호를 추출하여 시퀀스 상호작용 패턴을 캡처하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 결과, M-GPT는 여러 최신 추천 시스템과 비교하여 일관되게 우수한 성능을 보였습니다. 이러한 결과는 M-GPT의 혁신적인 접근 방식이 다중 행동 시퀀스 추천 문제에서 효과적임을 시사합니다. 또한 세밀한 실험을 통해 모델의 이점이 뚜렷하게 나타났습니다.



### Metamorphic Evaluation of ChatGPT as a Recommender System (https://arxiv.org/abs/2411.12121)
- **What's New**: 본 논문에서는 최신 자연어 처리 기반 시스템(예: Large Language Models, LLMs)이 추천 시스템의 성능 평가에서 기존의 전통적인 방법론과 차별화된 접근이 필요함을 강조합니다. LLM의 블랙박스 특성 때문에 뚜렷한 출력(추천 결과)을 가지고 평가하기 어려운 '테스트 오라클 문제'(test oracle problem)가 발생하는 것을 우려하고 있습니다. 이를 해결하기 위해 메타모픽 테스트(metamorphic testing)를 도입하여 LLM 기반 추천 시스템의 평가 프레임워크를 제안합니다.

- **Technical Details**: 메타모픽 테스트는 입력과 출력 간의 관계 즉, 메타모픽 관계(metamorphic relations, MRs)를 설정하고 이 관계가 출력에서 충족되는지를 검사하는 소프트웨어 테스트 기술입니다. 연구에서는 GPT 기반 추천 시스템을 평가하기 위해 네 가지 MRs(예: 추천 평가 곱셈, 평가 이동 및 프롬프트에서의 공간 및 무작위성 추가)를 사용하고, 결과의 유사성을 측정하기 위해 켄달 타우(Kendall tau)와 순위 편향 중첩(Ranking Biased Overlap, RBO) 지표를 사용합니다. 이를 통해 기존 추천 시스템 대비 LLM 기반 추천 시스템에 대한 평가 방식이 달라야 한다는 점을 명확히 하고 있습니다.

- **Performance Highlights**: 실험 결과는 MovieLens 데이터셋을 기반으로 하였으며, GPT3.5 모델을 사용하여 평가한 결과, 켄달 타우와 RBO 지표 모두에서 낮은 유사성이 관찰되었습니다. 이는 LLM 기반 추천 시스템에 대한 기존 평가 메트릭으로는 부족함을 시사하며, 더 포괄적인 평가 방법이 필요하다는 결론을 도출했습니다. 연구는 기존의 LLM 기반 추천 시스템과 전통적인 추천 시스템의 차이를 강조하며 더 체계적이고 정량적인 평가 방법의 필요성을 제기합니다.



### Deploying Large Language Models With Retrieval Augmented Generation (https://arxiv.org/abs/2411.11895)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 생성 능력이 환각(hallucination)이나 비사실적 응답을 생성하는 경향으로 인해 저해되는 문제를 다루고 있습니다. 연구자들은 LLM의 훈련 세트 외부의 사실 데이터에 기반하여 생성된 출력을 접지하는 방법에 주목하고 있으며, 정보 검색을 위한 Retrieval Augmented Generation (RAG) 방안을 제시합니다. 연구팀은 실제 데이터로 LLM과 RAG 통합의 현장 테스트를 통해 이 기술의 적용 기회와 도전을 분석하고 있습니다.

- **Technical Details**: 이 논문에서는 LLM이 비정형 문서에서 정보를 검색할 수 있도록 하는 다양한 접근방법을 설명하고 있으며, 토픽에 맞춘 미세 조정(fine-tuning) 및 프롬프트 엔지니어링(prompt engineering) 기법을 강조합니다. Retrieval-Augmented Generation (RAG) 방식은 기존 문서의 임베딩(embeddings)을 활용하여 LLM의 프롬프트를 향상시키는 접근입니다. 이러한 방식을 통해 도메인 특정 맥락을 추가하여 회수된 정보를 LLM의 생성 응답에 결합하게 됩니다.

- **Performance Highlights**: 연구자들은 RAG 시스템의 도입에 있어 여러 도전 과제와 산업 표준에 부합한 시스템 설계를 강조합니다. 이로 인해 RAG 방안의 실질적인 적용이 LLM 연구와 산업 간의 격차를 해소하는 데 기여할 수 있을 것으로 기대합니다. 특히, 이 논문은 LLM과 RAG을 통합한 파일럿 프로젝트의 실행과 평가에 대한 인사이트를 제공하며, 향후 정보 시스템(IS) 분야의 행동 연구에서 이 기술을 효과적으로 활용할 수 있는 방안을 제안합니다.



### Exploring Optimal Transport-Based Multi-Grained Alignments for Text-Molecule Retrieva (https://arxiv.org/abs/2411.11875)
Comments:
          BIBM 2024 Regular Paper

- **What's New**: 생명정보학 분야에서 본 연구는 텍스트 기반의 분자 검색 작업에 대한 새로운 접근 방식을 제안합니다. 우리가 제안한 Optimal TRansport-based Multi-grained Alignments 모델(ORMA)은 텍스트 설명과 분자 구조 간의 다중 정렬을 가능하게 하여 연구자들이 적합한 분자를 식별하는 데 도움을 줍니다. ORMA는 텍스트 인코더와 분자 인코더를 갖추고 있으며, 분자의 하위 구조에서 세부 정보를 포착하여 정확한 검색을 돕습니다.

- **Technical Details**: ORMA의 핵심 혁신은 Optimal Transport(OT) 기법을 활용하여 토큰과 모티프의 정렬을 수행하는 것입니다. 이 모델은 각 입력 텍스트 설명을 처리하여 토큰과 문장 수준의 표현을 생성하고, 분자는 계층적인 이질 그래프로 모델링되어 원자, 모티프, 분자 노드를 포함합니다. 또한, 다양한 수준에서의 대조 학습을 통해 텍스트-원자, 다중 토큰-모티프, 문장-분자 간의 정렬을 세밀하게 조정합니다.

- **Performance Highlights**: 실험 결과, ORMA는 ChEBI-20과 PCdes 데이터셋에서 기존의 최첨단(SOTA) 모델에 비해 월등한 성능을 보여주었습니다. 특히 ChEBI-20에서 텍스트-분자 검색의 Hits@1 점수가 66.5%로, 기존 SOTA인 AMAN을 17.1% 초과했고, 분자-텍스트 검색에서는 61.6%로 AMAN보다 15.0% 우수한 성적을 기록했습니다.



### MultiBalance: Multi-Objective Gradient Balancing in Industrial-Scale Multi-Task Recommendation System (https://arxiv.org/abs/2411.11871)
- **What's New**: 이 논문은 멀티태스크 학습(multi-task learning)에서 발생할 수 있는 부정적 전이(negative transfer) 문제를 해결하기 위해 MultiBalance라는 새로운 기법을 제안합니다. MultiBalance는 작업별(퍼태스크, per-task) 경량의 그래디언트를 균형 있게 조정하여 여러 작업 간의 최적화를 도와주며, 그리드 서치(grid search)나 수동 탐색(manual exploration)의 비용을 절감합니다. 이 기법은 Meta의 대규모 광고 및 피드 멀티태스크 추천 시스템에서 강력한 성과를 보였으며, 이는 대규모 산업 시스템에서의 실제적 응용 가능성을 보여줍니다.

- **Technical Details**: MultiBalance는 각 작업의 그래디언트를 공유된 피처 표현(shared feature representation)을 기준으로 균형 있게 조정합니다. 기존의 방법과는 다르게, MultiBalance는 작업별 그래디언트가 아닌 공유된 피처 표현의 출력으로 수렴하는 것을 목표로 합니다. 또한 훈련 과정에서 미니 배치 스토캐스틱 그래디언트(mini-batch stochastic gradients)의 크기 이동 평균을 유지하여 대규모 모델 학습의 안정성을 높입니다.

- **Performance Highlights**: MultiBalance는 QPS(Queries Per Second) 성능 손실 없이 정상훈련비용(neutral training cost)에서 0.738%의 개선을 보여주었습니다. 이전의 방법들과 비교했을 때, MultiBalance는 70%~80%의 QPS 감소 없이도 더 높은 효율성을 달성하였습니다. 이러한 결과는 대규모 산업 시스템에 적용하기에 매우 적합한 ROI(투자수익률) 솔루션임을 입증합니다.



### \textsc{Neon}: News Entity-Interaction Extraction for Enhanced Question Answering (https://arxiv.org/abs/2411.12449)
- **What's New**: 본 연구는 NEON 프레임워크를 제안하여, 최신 정보 추출과 이를 기반으로 하는 언어 모델의 성능 향상을 목표로 하고 있습니다. NEON은 뉴스 기사에서 드러나는 사건 및 활동 간의 상호작용을 분석하여, 엔티티 중심의 타임스탬프 지식 그래프를 구축합니다. 이를 통해 뉴스 이벤트와 관련된 질의응답(QA) 능력을 향상시키는데 기여합니다. 또한, 기존 LLMs와의 통합을 통해 실제적이고 신뢰할 수 있는 응답 생성을 지원합니다.

- **Technical Details**: NEON은 뉴스에서 엔티티 간의 상호작용을 포착하여 이를 반영하는 지식 그래프를 형성합니다. 이 그래프는 엔티티(entities)를 노드로, 이들 간의 상호작용(interactions)을 엣지로 표현합니다. 또한, 시계열적 정보 검색을 효율적으로 수행하기 위해 최적화된 인덱싱 기법을 활용합니다. NEON의 생성된 엔티티 상호작용 튜플을 LLM 프롬프트에 추가하여 임시 질의응답 생성을 수행합니다.

- **Performance Highlights**: NEON을 사용하여 3,000개의 실제 질의를 수집하고, 이 결과는 LLM의 응답 품질을 향상시킵니다. 실험 결과, NEON을 통해 얻어진 정보가 LLM의 응답 정확성과 시의성이 크게 개선되는 것으로 나타났습니다. 자동 평가 및 인간 전문가의 평가를 통해 이에 대한 검증도 수행하였으며, NEON의 통합이 실제 질문에 대한 응답 품질을 HIGH로 증진시킨 것을 입증하였습니다.



### Balancing Accuracy and Efficiency in Multi-Turn Intent Classification for LLM-Powered Dialog Systems in Production (https://arxiv.org/abs/2411.12307)
- **What's New**: 이 논문은 대화형 AI 시스템의 다중 턴(종종 multi-turn이라고도 하는) 의도 분류(multi-turn intent classification)를 개선하기 위한 두 가지 새로운 접근 방식을 제안합니다. 첫째, Symbol Tuning을 도입하여 의도 레이블을 간소화하고, 둘째, C-LARA(Consistency-aware, Linguistics Adaptive Retrieval Augmentation)라는 프레임워크를 개발하여 대량의 다중 턴 대화 데이터를 합성합니다. 이 방법은 다국어 산업 시스템에서 낮은 자원 하에서도 확장 가능성을 높이고 비용을 절감합니다.

- **Technical Details**: Symbol Tuning은 LLM과의 상호 작용에서 의도 라벨을 압축하여 복잡성을 줄이고 성능을 개선하는 접근 방식입니다. C-LARA는 사용자로부터의 비표시 발화(unlabeled utterances)를 사용하여 다중 턴 데이터를 생성하는 효율적인 도구입니다. 이 두 가지 방법은 대화의 맥락을 고려하면서 다중 턴 의도 분류(MTIC)의 정확도를 높입니다.

- **Performance Highlights**: 이 연구는 MTIC 시스템에서 AUC 점수를 5.09% 개선하고 주석(annotations) 비용을 40% 줄이는 성과를 보여줍니다. 다국어 대화 데이터셋에 대한 실험을 통해, 제안된 방법들이 모델의 성능을 상당히 향상시키며 자원 효율성 또한 높인 것으로 나타났습니다. 이러한 결과는 낮은 자원 환경에서의 실제적인 응용 가능성을 강조합니다.



### SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search (https://arxiv.org/abs/2411.12229)
Comments:
          The paper has been accepted by SIGMOD 2025

- **What's New**: 이번 연구에서는 고차원 유클리드 공간에서의 근사 최근접 이웃 검색(ANN)에 대한 새로운 방법인 SymphonyQG를 제안합니다. 이전 방식인 NGT-QG의 한계를 보완하여 양자화(quantization)와 그래프(graph)의 통합을 보다 효과적으로 이루어냅니다. 특히, SymphonyQG는 명시적인 재정렬(re-ranking) 단계를 피하고, FastScan과 더욱 정렬된 그래프 구조를 재구성하여 성능을 개선했습니다.

- **Technical Details**: SymphonyQG는 고차원 데이터에서의 검색 성능을 향상시키기 위해 양자화 코드와 그래프 기반 인덱스를 통합합니다. 이 시스템은 이웃의 양자화 코드를 저장하고, SIMD(Single Instruction, Multiple Data) 기반의 FastScan을 사용하여 배치(batch)로 거리 추정(distance estimation)을 효율적으로 수행합니다. 이 과정을 통해 메모리에서의 무작위 접근을 줄이고 계산을 최적화합니다.

- **Performance Highlights**: Extensive experiments conducted on real-world datasets show that SymphonyQG is achieving state-of-the-art results in the time-accuracy trade-off. 이전의 그래프 기반 방법과 비교했을 때, SymphonyQG는 더 높은 정확도와 더 낮은 검색 시간을 동시에 달성하고 있습니다. 그 결과, SymphonyQG는 실질적인 애플리케이션에 적용할 수 있는 유망한 시스템임을 입증하였습니다.



### INDIANA: Personalized Travel Recommendations Using Wearables and AI (https://arxiv.org/abs/2411.12227)
Comments:
          Accepted as position paper at 8th International Workshop on Chatbots and Human-Centred AI - CONVERSATIONS 2024

- **What's New**: 이 연구는 INDIANA 플랫폼으로 개발된 개인 맞춤형 여행 추천 시스템을 제안합니다. 이 시스템은 웨어러블 장치, 사용자 선호도, 현재 위치, 날씨 예보 및 활동 기록의 데이터를 활용하여 실시간으로 맥락에 맞는 추천을 제공합니다. 개인 관광객의 경험을 극대화하는 것은 물론 관광 전문가에게도 서비스를 향상시킬 수 있는 통찰력을 제공합니다.

- **Technical Details**: INDIANA 플랫폼은 여러 데이터 소스를 통합하여 실시간으로 동적인 추천을 제공합니다. 이 플랫폼은 사용자 건강 데이터와 환경 요소를 통합하여 여행자의 현재 위치와 신체 상태에 따라 추천을 지속적으로 조정하는 기능을 가지고 있습니다. 이러한 혁신적 접근은 전통적인 정적이나 단일 소스 기반 시스템의 한계를 초월합니다.

- **Performance Highlights**: INDIANA 플랫폼은 사용자 맞춤형 여행 경험을 제공하기 위한 다양한 사용 사례를 지원합니다. 사용자는 챗봇을 통해 현재 위치에 기반한 즉각적인 추천을 받을 수 있으며, 웨어러블 장치와 모바일 데이터를 활용하여 자동으로 활동을 제안받는 등 프로액티브한 여행 계획이 가능합니다. 이 플랫폼은 여행의 의미 있는 경험을 극대화하며, AI와 IoT의 잠재력을 보여줍니다.



### Preprocessing for lessening the influence of eye artifacts in eeg analysis (https://arxiv.org/abs/2411.12092)
Comments:
          16 pages, journal article

- **What's New**: 이 연구는 EEG (Electroencephalography) 신호에서 긴 실험과 관련된 아티팩트 문제를 다루었습니다. 특히, 눈 아티팩트(eye artifacts)에 집중하여 이러한 아티팩트가 데이터를 분석하는 데 미치는 영향과 이러한 영향을 최소화하기 위한 대안들을 제시합니다. 연구진은 뇌 활동을 분석하기 위한 혁신적인 접근 방안을 제안하였습니다.

- **Technical Details**: 연구에서 제안된 방법은 독립 신호 성분(independent signal components)의 부분 거부(partial rejection)에 기초하고 있습니다. 이를 통해 EEG 신호 성분을 추출할 수 있으며, 눈 아티팩트의 영향을 줄인 상태에서 데이터를 처리할 수 있습니다. 이 방법은 긴 작업에서 뇌 활동 분석의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 연구에서는 아티팩트가 없는 신호 구간을 사용하는 것이 뇌 활동을 음악적 맥락(context)에서 분석하는 데 중요한 역할을 한다고 강조합니다. 아티팩트가 제거된 신호를 활용함으로써 뇌의 반응을 더 명확하게 이해할 수 있으며, 이는 향후 연구에서 신뢰성을 높이는 데 도움이 될 것입니다.



### TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Mod (https://arxiv.org/abs/2411.12064)
Comments:
          Accepted to ACM SIGKDD 2025 Research Track

- **What's New**: 이번 논문에서는 Travelling Salesman Problem Rank (TSPRank)라는 하이브리드 페어와이즈-리스트와이즈 순위 매기기 방법을 소개합니다. 기존의 LETOR 방법들은 페어와이즈 비교에만 초점을 맞추어 글로벌 최적 순위를 제공하는 데 한계를 보였지만, TSPRank는 순위 문제를 잘 알려진 조합 최적화 문제인 여행하는 세일즈맨 문제(TSP)로 재구성했습니다. 이로 인해 TSPRank는 페어와이즈 관계를 모델링하면서 리스트와이즈 순위를 결정할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: TSPRank는 조합 최적화 기법을 통해 페어와이즈와 리스트와이즈 순위 방식의 장점을 모두 활용합니다. 구체적으로는, TSPRank가 다양한 백본 모델에서 임베딩을 생성하여 순위 성능을 향상시키는 방식으로 적용될 수 있습니다. 또한 두 가지 학습 방법, 즉 지도의 상대적 참조를 사용하는 지역 방식과 TSP 솔버를 포함한 글로벌 방식이 소개되어 있습니다.

- **Performance Highlights**: 실험 결과, TSPRank는 주식 순위 매기기, 정보 검색, 역사적 사건 정렬 등 다양한 작업에서 기존의 페어와이즈 및 리스트와이즈 방법을 능가하는 성능을 보였습니다. 특히, TSPRank의 주요 장점은 글로벌 정보를 더 잘 활용하여 순위를 매길 수 있다는 점이며, 이러한 특성 덕분에 다양한 도메인에서의 강력한 성능을 나타냅니다.



### Benchmarking pre-trained text embedding models in aligning built asset information (https://arxiv.org/abs/2411.12056)
- **What's New**: 이 논문은 구축 자산(data related to built assets) 정보를 전문 용어에 맞춰 잘 정렬하기 위한 새로운 접근 방식을 제안합니다. 특히 최근의 대형 언어 모델을 활용한 텍스트 임베딩(text embedding)이 자산 관리의 데이터 매핑(data mapping) 과정의 자동화를 모색하는 데 기여할 수 있음을 보여줍니다. 이 연구는 기존 모델의 성능을 비교하고, 구축 자산에 특화된 기술 용어의 복잡한 의미를 효과적으로 표현할 수 있는지를 평가합니다.

- **Technical Details**: 이 연구는 다양한 하위 도메인에 걸쳐 구축 제품에 대한 정보의 체계적인 데이터를 구성하고, 이를 기반으로 구체적 작업 세트를 개발합니다. 데이터는 산업 기초 클래스(Industry Foundation Classes, IFC)와 유니클래스(Uniclass)의 두 가지 주요 출처에서 수집되었습니다. 또, 제안된 데이터 세트는 여섯 가지 작업, 즉 클러스터링(clustering), 검색(retrieval), 재순위화(reranking)를 평가하여 모델간의 성능을 비교하는 데 중점을 둡니다.

- **Performance Highlights**: 이 연구의 평가 결과는 자동화된 데이터 매핑 과정에서 구성 자산 정보를 정렬하는 데 있어 기존 언어 모델의 강점과 한계를 안내합니다. 현재 관련된 24개의 텍스트 임베딩 모델을 사용하여 10,000개 이상의 데이터 항목을 포함하는 가장 포괄적인 벤치마크를 제공합니다. 또한 연구 결과 및 데이터 세트는 공개 소스로 제공되어 후속 연구의 기초 자료로 사용될 수 있도록 하였습니다.



### Survey on Semantic Interpretation of Tabular Data: Challenges and Directions (https://arxiv.org/abs/2411.11891)
- **What's New**: 이 논문은 Semantic Table Interpretation (STI)에 대한 포괄적인 개요를 제공하며, 31개의 특성을 기반으로 한 분류 체계를 통해 다양한 접근 방식을 정리합니다. 또한, 12개의 기준을 사용하여 사용 가능한 도구들을 평가하고, STI 접근 방식을 평가하기 위한 Gold Standards에 대한 심층 분석도 포함되어 있습니다. 마지막으로, 사용자가 특정 작업에 가장 적합한 접근 방식을 선택하는 데 도움을 주기 위한 실용적인 지침을 제시하고, 미해결 문제와 향후 연구 방향을 제안합니다.

- **Technical Details**: STI는 관계형 테이블을 입력으로 받아, 지식 그래프(Knowledge Graph, KG)와의 매칭을 통해 의미론적으로 주석이 달린 테이블을 생성하는 과정입니다. STI 과정은 주로 Cell-Entity Annotation (CEA), Column-Type Annotation (CTA), Columns-Property Annotation (CPA)의 세 가지 주요 태스크로 나뉘며, 이는 각각 테이블의 열과 셀에 KG의 개념 및 속성을 연결합니다. STI는 데이터의 의미를 명확히 하여 다양한 내려받기(application) 작업에 활용할 수 있는 지식을 제공합니다.

- **Performance Highlights**: STI는 AI 연구 및 응용에서 중요한 역할을 하며, CEA, CTA, CPA와 같은 여러 태스크는 탭 데이터 이해의 일환으로 간주됩니다. 이를 통해 지식 기반의 구축 및 확장은 물론, 탭 데이터의 풍부한 정보를 제공하여 데이터 분석을 위한 후속 응용 프로그램을 지원할 수 있습니다. 따라서 STI는 지식 그래프와 같은 구조에서 연결된 다양한 개체들 간의 관계를 조직화하는 데 기여하는 핵심 연구 분야로 부각됩니다.



New uploads on arXiv(cs.CV)

### CATCH: Complementary Adaptive Token-level Contrastive Decoding to Mitigate Hallucinations in LVLMs (https://arxiv.org/abs/2411.12713)
- **What's New**: 이 논문에서는 대규모 비전-언어 모델(LVLM)에서 발생하는 환각(hallucination) 문제를 해결하기 위해 새로운 방법 CATCH(Complementary Adaptive Token-level Contrastive Decoding to Mitigate Hallucinations)를 제안합니다. CATCH는 정보 병목 정보이론(Information Bottleneck theory)을 기반으로 하여 시각 정보를 분리하는 Complementary Visual Decoupling(CVD)과 환각 탐지를 위한 Non-Visual Screening(NVS), 그리고 환각 완화를 위한 Adaptive Token-level Contrastive Decoding(ATCD)을 도입합니다. 이 방법은 다양한 시각적 질문 응답(visual question-answering) 작업에 적용 가능하며 추가 학습 없이도 새로운 작업에 강력하게 일반화 할 수 있는 가능성을 열어줍니다.

- **Technical Details**: CATCH는 시각 정보가 과도하게 유입되는 것에 대한 시각적 결함(visual defect)의 문제를 해결하기 위해 두 가지 이미지와 잔여 이미지를 구분하여 시각 정보를 안정적인 형상(decoupled visual representation)으로 표현합니다. 이 과정에서는 Segmentation 모델인 SAM(Segment Anything Model)을 사용하여 원본 시각 입력을 여러 수준으로 분리합니다. 또한, 모델에서 불필요한 시각적 특징을 축소해 언어 이전(linguistic priors)에 의한 극단적 의존성을 줄입니다. 이 approach는 출력 분포 간의 Jensen-Shannon Divergence(JSD)를 이용하여 비시각적 입력에 대한 정보와 비교하여 필수 시각적 정보의 밀도를 증가시키고 불확실성을 감소시킵니다.

- **Performance Highlights**: CATCH의 제안된 방법은 시각적 정보의 밀도를 증가시키고, 시각적 결함으로 발생하는 다양한 환각 문제를 완화하는 데 효과적임을 입증합니다. 실험 결과, 모델이 시각 입력이 obscured된 경우에도 언어적 측면에서 높은 정확도를 유지하며 비교적 일치하는 출력 분포를 생성하는 것을 확인했습니다. 이는 시각적 정보가 불필요한 정보로 인해 감소되지 않고, 핵심 시각적 특징이 지속적으로 유지됨을 나타내며, 고위험 도메인에서도 안정적인 성능을 보여줍니다.



### Deep Learning-Driven Heat Map Analysis for Evaluating thickness of Wounded Skin Layers (https://arxiv.org/abs/2411.12678)
- **What's New**: 이 논문은 상처 치유 관행 개선을 위한 새로운 비침습적(non-invasive) 방법을 제안합니다. 기존의 깊이 측정 방법이 침습적이고 구체성이 떨어지는 데 비해, 이 방법은 심층 학습(deep learning) 기술을 활용하여 피부 층을 분류하고 열 지도 분석을 통해 상처 깊이를 측정합니다. 약 200개의 라벨이 지정된 이미지 세트를 사용하여 흉터, 상처 및 건강한 피부 등의 다섯 가지 클래스를 구별합니다.

- **Technical Details**: 제안된 방법은 Roboflow 소프트웨어에서 스트라텀 코르네텀(stratum corneum), 표피(epidermis), 진피(dermis)와 같은 주요 층을 주석 처리한 이미지를 사용하여 진행됩니다. 초기 단계에서는 VGG16을 사용하여 조직 층의 가시성을 향상시킨 후, 이를 통해 생성된 이미지를 ResNet18 모델로 훈련했습니다. 훈련 결과 97.67%의 높은 정확도를 달성했으며, EfficientNet 및 ResNet18 모델의 성능 비교를 통해 두 모델 모두 약 95.35%의 정확도를 기록했습니다.

- **Performance Highlights**: 또한 효율적인 모델 구성을 결정하기 위해 두 모델의 하이퍼 파라미터 튜닝도 실시하였습니다. 학습률에 따라 정확도가 큰 변동을 보였고, EfficientNet과 ResNet18 모두 0.0001의 학습률에서 최대 95.35%의 정확도를 달성하였습니다. 이러한 결과는 모델이 실제 시간(non-invasive)으로 상처를 평가하는 데 적용 가능하다는 것을 나타내며, 임상 진단 및 치료 계획 개선에 큰 잠재력을 지니고 있음을 의미합니다.



### IoT-Based 3D Pose Estimation and Motion Optimization for Athletes: Application of C3D and OpenPos (https://arxiv.org/abs/2411.12676)
Comments:
          17 pages

- **What's New**: 이 연구에서는 고정밀 3D 포즈 추정 및 운동 최적화를 위한 IoT-Enhanced Pose Optimization Network (IE-PONet)를 제안합니다. IE-PONet는 C3D를 활용한 시공간(feature extraction), OpenPose를 이용한 실시간 키포인트 검출, Bayesian optimization을 통한 하이퍼파라미터 조정을 통합하여 성능을 극대화합니다. NTURGB+D 및 FineGYM 데이터셋에서 90.5 및 91.0의 AP^p50 점수와 74.3 및 74.0의 mAP 점수가 나타나며, 각 모듈의 필수적인 역할을 입증한 ablation 연구도 포함되어 있습니다.

- **Technical Details**: IE-PONet 모델은 복잡한 운동 환경에서 실시간으로 데이터 수집 및 전송이 가능하도록 IoT 센서를 통합하여, 기존 기술의 단점을 보완합니다. C3D 및 OpenPose의 장점을 결합하여 운동 분석과 최적화를 정확하게 수행하며, Bayesian optimization을 통해 모델의 적응성과 효율성을 더욱 향상시킵니다. 또한, 이 시스템은 빠른 데이터 처리와 실시간 피드백이 가능한 성능을 보여줍니다.

- **Performance Highlights**: IE-PONet는 다양한 데이터셋에서 우수한 성능과 강 robustness를 입증하였습니다. 실험 결과, NTURGB+D 및 FineGYM 데이터셋에서의 성능이 특히 두드러져, 이는 훈련 과정에서의 즉각적인 피드백과 운동 조정 개선을 가능하게 합니다. 향후 연구는 모델 최적화와 다중 데이터 통합, 실시간 피드백 메커니즘 개발에 초점을 맞출 계획입니다.



### PoM: Efficient Image and Video Generation with the Polynomial Mixer (https://arxiv.org/abs/2411.12663)
- **What's New**: 최근 Diffusion 모델에서 Multi-Head Attention (MHA)을 대체할 새로운 메커니즘인 Polynomial Mixer (PoM)를 소개합니다. PoM은 시퀀스의 길이에 대해 선형 복잡성을 가지며, MHA의 품질을 손상시키지 않고도 전체 시퀀스를 명시적인 상태로 인코딩합니다. 이 메커니즘은 대규모 생성 모델의 효율적인 스케일링을 가능하게 하며, 특히 고해상도 이미지와 비디오 생성을 더 용이하게 만들어 줍니다.

- **Technical Details**: PoM은 State-Space Models (SSMs)와 유사한 선형 복잡성을 고려하지만, MHA와 같이 모든 쌍 정보에 대한 처리를 가능하게 합니다. 이를 통해 DiT 아키텍처에 적용 시 더 높은 해상도에서 MHA를 사용하는 모델을 학습하는 것보다 PoM을 사용하는 모델이 훈련 비용이 월등히 낮아질 수 있음을 보여줍니다. 또한 PoM은 일반적인 시퀀스-투-시퀀스 (sequence-to-sequence) 근사기를 제공함으로써 다양한 애플리케이션에서의 적용 가능성을 높입니다.

- **Performance Highlights**: PoM을 적용한 이미지 생성 모델은 유사한 품질의 샘플을 생성하면서도 더 적은 계산 자원으로 고해상도의 이미지를 처리할 수 있는 능력을 보여줍니다. 더불어 PoM을 활용한 비디오 생성 모델은 매 프레임에 대해 일정한 처리 비용을 유지하면서도 시각적 품질을 유지할 수 있습니다. 이로 인해 PoM은 향후 고해상도 이미지 및 긴 비디오 생성을 위한 기본적인 메커니즘으로 자리 잡을 것으로 기대됩니다.



### M3D: Dual-Stream Selective State Spaces and Depth-Driven Framework for High-Fidelity Single-View 3D Reconstruction (https://arxiv.org/abs/2411.12635)
Comments:
          9 pages, 4 figures, submitted to CVPR 2025 for review

- **What's New**: 본 논문에서는 단일 RGB 이미지로부터 3D 객체를 재구성하는 데 있어 SOTA 성능을 달성하는 M3D라는 새로운 프레임워크를 제안합니다. M3D는 Selective State Space Model을 기반으로 하는 이중 스트림 피처 추출 방식을 적용하여, 글로벌 및 로컬 피처 추출의 균형을 맞추고 있습니다. 이 구조는 복잡한 장면 이해를 향상시키고, 깊이 정보를 통합하여 세밀한 디테일을 보존하며 재구성 품질을 개선합니다.

- **Technical Details**: M3D는 RGB 및 깊이 정보를 독립적으로 처리하는 이중 스트림 피처 추출 모듈을 통합하고 있습니다. Selective Attention Module을 통해 장거리 문맥과 로컬 피처를 조합하여 공간적으로 일관된 표현을 생성합니다. 향상된 형태의 암시적 기하학적 표현과 렌더링 네트워크가 결합되어, 복잡한 기하학적 구조를 효과적으로 처리할 수 있는 기반을 마련합니다.

- **Performance Highlights**: 실험 결과, M3D는 복잡한 장면과 폐색이 있는 상황에서 기존 방법들보다 월등한 성능을 보였습니다. 특히, Front3D 데이터셋에서 Chamfer Distance가 36.9% 향상되었고 F-score는 13.3% 증가하며 Normal Consistency는 5.5% 개선되는 결과를 얻었습니다. 이러한 성과는 M3D가 복잡한 환경에서 정확하고 세밀한 3D 모델을 생성할 수 있음을 입증합니다.



### Maps from Motion (MfM): Generating 2D Semantic Maps from Sparse Multi-view Images (https://arxiv.org/abs/2411.12620)
- **What's New**: 이 논문은 2D 맵의 자동화를위한 새로운 문제인 Maps from Motion (MfM)을 소개합니다. 이는 다양한 관점에서 촬영된 비정렬 이미지 세트를 사용하여 2D 맵을 생성하는 과정을 혁신적으로 단순화합니다. 기존의 수동적이고 느린 주석 과정을 넘어, 이 접근 방식은 인식된 객체의 패턴을 분석하여 자동으로 2D 좌표를 생성합니다.

- **Technical Details**: MfM은 각 이미지에서 객체 감지를 추출하고, 이를 바탕으로 카메라 중심의 로컬 맵을 생성하여 연합합니다. 이를 위해 새로운 그래프 구조를 도입하여, 각 이미지에서 객체의 공간 및 의미적 배치를 표현하고, 글로벌 참조 시스템에서 객체의 포즈를 예측합니다. 또한, 학습 과정에서 입력 감지 간의 매칭을 고려하여 topology를 유지하면서 최적의 정렬을 찾는 방식으로 수행됩니다.

- **Performance Highlights**: 이 연구의 결과, 제안된 방식은 GPS 정확도보다 나은 평균 4미터 내의 정밀도로 2D 맵을 성공적으로 등록할 수 있음을 보여줍니다. 특히, 강한 시점 변화가 있는 희소 이미지 시퀀스에서도 60% 낮은 실패 비율과 함께 COLMAP과 유사한 물체 및 카메라 위치 추정 정확도를 기록했습니다. 이는 기존 최적화 기법이 실패하는 시나리오에서도 유효한 솔루션을 제공함을 입증합니다.



### A Multimodal Approach Combining Structural and Cross-domain Textual Guidance for Weakly Supervised OCT Segmentation (https://arxiv.org/abs/2411.12615)
Comments:
          21 pages, 9 figures, 8 tables

- **What's New**: 이 논문은 Optical Coherence Tomography (OCT) 이미지의 정확한 세분화를 위한 새로운 약한 감독 의미 세분화(Weakly Supervised Semantic Segmentation, WSSS) 접근 방식을 제안합니다. 이 방법은 구조적 안내와 텍스트 기반 전략을 통합하여 고품질의 의사 라벨을 생성하는 데 초점을 맞춰 세분화 성능을 크게 향상시킵니다. 특히, 이 연구는 OCT 이미지에서 병변을 식별하는 데 있어 이미지 수준의 감독만을 사용하며, 효율성을 극대화하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 시각적 정보와 텍스트 정보를 융합한 다중 모드(multi-modal) 프레임워크를 기반으로 합니다. 이 방법은 두 개의 처리 모듈을 사용해 원본 이미지 특징과 OCT 이미지의 구조적 특징을 교환하며, 이를 통해 모델이 병변이 발생할 가능성이 있는 위치를 식별하도록 유도합니다. 또한, 대규모 사전 훈련된 모델을 활용하여 레이블 기반 텍스트 정보와 합성 설명을 통합하며, 이를 통해 모델의 성능을 더욱 향상시키고 있습니다.

- **Performance Highlights**: 세 가지 OCT 데이터 세트에 대한 실험 결과, 제안된 방법이 최신 WSSS 연구 방법들과 비교해 우수한 성능을 보이며, 진단 정확성과 의료 영상의 효율성을 개선할 수 있는 가능성을 강조합니다. 이 연구는 병변을 지역화하는 데 있어 텍스트와 구조적 정보를 효과적으로 통합하여 세분화의 정확도를 높이는 데 기여하였습니다.



### SG-LRA: Self-Generating Automatic Scoliosis Cobb Angle Measurement with Low-Rank Approximation (https://arxiv.org/abs/2411.12604)
- **What's New**: 이 논문은 X-ray 이미지에서의 자동 Cobb 각도 측정을 위한 새로운 프레임워크인 Self-Generation pipeline과 Low-Rank Approximation representation(SG-LRA)을 제안합니다. 기존의 방법들이 겪는 문제점(부정확한 척추 표현, 마스크 연결 문제 등)을 해결하기 위해, LRA 기반의 매개변수화된 척추 윤곽 표현을 도입하여 보다 정확한 척추 표현을 이루도록 합니다. 또한, Spinal2023 데이터를 기반으로 한 자동 주석 생성기를 개발하여 개인 정보 유출 위험 없이 대규모의 척추 측만 X-ray 데이터셋인 Spinal-AI2024를 생성하였습니다.

- **Technical Details**: SG-LRA는 회귀 기반 및 랜드마크 기반 기술을 통합하여 척추 세그먼트의 경계를 정의하는 랜드마크 세트를 활용한 자동 Cobb 각도 측정 프레임워크입니다. Low-Rank Approximation(LRA)을 적용하여 학습된 척추 윤곽을 로우랭크 계수를 통해 재구성함으로써 정확한 척추 세그먼트 경계를 얻습니다. 또한, 자동 주석 생성의 반복적 과정을 통해, 수작업 없이도 높일 수 있는 효율적인 방법론이 개발되었습니다.

- **Performance Highlights**: 본 연구의 결과로, AASCE2019, Spinal2023 및 Spinal-AI2024 데이터셋을 활용한 실험에서 최첨단 Cobb 각도 측정 성능을 달성하였습니다. 기존 방법들과 비교했을 때, 최대 각도 및 세 가지 지역 각도 모두에서 탁월한 성능을 보여주었습니다. SG-LRA는 보다 정교한 척추 모양 표현을 통해 치료에 필요한 정확한 정보를 제공하여, 척추 측만증 조기 발견에 기여할 수 있음을 입증하였습니다.



### STREAM: A Universal State-Space Model for Sparse Geometric Data (https://arxiv.org/abs/2411.12603)
- **What's New**: 이 논문에서는 기하학적 구조를 상태 공간 모델(SSM)의 매개변수화에 명시적으로 인코딩하는 방안을 제안합니다. 이를 통해 기존의 sequence 모델들과 비교하여 불규칙한 단계 크기를 가진 희소 기하학 데이터 처리에 효율성을 제공합니다. 이로 인해 새로운 모델인 STREAM은 이벤트 기반 비전 및 포인트 클라우드 분석에서 경쟁력 있는 성능을 발휘합니다.

- **Technical Details**: STREAM 모델은 CUDA 커널의 수정된 버전을 사용하여 현대 하드웨어에 효율적으로 희소 기하학 데이터를 매핑합니다. 본 연구에서는 상대 좌표의 차이를 단계 크기로 주입하여 기하학적 연산을 O(N) 단계로 수행하도록 설계되었습니다. 이는 모든 N포인트 간의 상호작용을 계산하는 데 필요한 단계를, 기존 방식보다 획기적으로 줄여 줍니다.

- **Performance Highlights**: STREAM 모델은 ModelNet40 및 ScanObjectNN 데이터셋에서 PointMamba 기준보다 최대 2.84% 향상된 성능을 보여주었습니다. 더불어 DVS128 Gesture 데이터셋에서 모든 11개 클래스에 대해 100%의 테스트 정확도를 달성하였습니다. 이는 이벤트 기반 비전 분야에서 최초의 결과로, 희소 기하학 데이터 처리에서의 강력한 유도 편향(inductive bias)을 입증합니다.



### SAM Carries the Burden: A Semi-Supervised Approach Refining Pseudo Labels for Medical Segmentation (https://arxiv.org/abs/2411.12602)
Comments:
          Presented at MICCAI Workshop on Advancing Data Solutions in Medical Imaging AI 2024; Code and data: this https URL

- **What's New**: 이번 연구에서는 Segment Anything Model (SAM)의 기능을 활용하여 의학적 이미지를 대상으로 지도 세그멘테이션의 필요성을 감소시키는 새로운 접근 방식을 제안합니다. SAM은 제한된 양의 주석 데이터로부터 초기 세그멘테이션을 개선하여 가짜 레이블(pseudo labels)을 생성합니다. 이는 라벨이 없는 데이터의 세그멘테이션을 자동으로 수행할 수 있는 기회를 제공하며, 이로 인해 적은 양의 주석 데이터로도 효과적인 학습이 가능해집니다.

- **Technical Details**: 제안하는 방법은 의학 이미지 세그멘테이션에 SAM을 적용하여 초기 세그멘테이션 마스크를 가짜 레이블로 변환하는 단계로 구성되어 있습니다. 구체적으로, 적은 수의 주석 데이터(최대 43개 사례)를 기반으로 바운딩 박스와 시드 포인트를 추출하여 SAM에 전달합니다. 결과적으로, 라벨이 없는 데이터에 대한 밀집 세그멘테이션 마스크 생성을 가능하게 하며, 이 과정에서 SAM의 추상적인 물체 이해를 활용하여 예측 세그멘테이션의 품질을 보장합니다.

- **Performance Highlights**: 본 연구에서는 가짜 레이블을 사용한 훈련이 소아 손목 뼈와 치아 세그멘테이션에서 각각 74.29%에서 84.17%로, 66.63%에서 74.87%로 다이스 점수(Dice score)를 개선한 결과를 보고합니다. 제안한 방법은 강도 기반 후처리 방법, 최첨단 지도 학습(nnU-Net), 그리고 반지도 학습(Mean Teacher) 접근 방식보다 우수한 성능을 보입니다. 이러한 결과는 가짜 레이블을 통한 반지도 학습의 효과성을 시사합니다.



### AdaCM$^2$: On Understanding Extremely Long-Term Video with Adaptive Cross-Modality Memory Reduction (https://arxiv.org/abs/2411.12593)
- **What's New**: 이 논문에서는 AdaCM2라는 새로운 방법을 제안하며, 이는 비디오-텍스트 정렬을 수행하기 위해 처음으로 적응형 크로스 모달리티 메모리 축소 방식을 도입합니다. 기존 연구는 주로 짧은 비디오 처리에 한정되었지만, AdaCM2는 복잡한 질문-응답 작업을 효과적으로 처리할 수 있는 긴 비디오 이해를 목표로 합니다. 이 프레임워크는 여러 데이터셋에서 차세대 성과를 달성하며 메모리 사용량을 대폭 줄입니다.

- **Technical Details**: AdaCM2는 개별 레이어에서 텍스트 쿼리와 가장 관련이 높은 중요 시각 토큰을 적응적으로 보존하는 방식으로 메모리 소비를 줄입니다. 이를 위해 Q-Former와 같은 고정된 비주얼 인코더를 사용하여 비디오 프레임에서 시각 표현을 추출한 후, AdaCM2 어텐션 메커니즘을 통해 프레임 단위로 쿼리를 학습합니다. 이 접근 방식은 다양한 레이어의 크로스 모달리티 상관관계에 따라 시각 토큰을 유연하게 조정할 수 있게 합니다.

- **Performance Highlights**: 다양한 비디오 이해 작업에 대한 광범위한 실험을 통해 AdaCM2는 LVU 데이터셋에서 여러 작업에 대해 4.5%의 정확도 향상을 달성하며, GPU 메모리 소비를 65%까지 줄였습니다. 특히 VQA(Visual Question Answering) 및 비디오 캡션 생성 작업에서 유망한 성과를 보였습니다. AdaCM2는 기존 BLIP 기반 모델의 성능을 플러그 앤 플레이 방식으로 향상시켜 긴 비디오 처리 능력을 향상시킵니다.



### SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction (https://arxiv.org/abs/2411.12592)
- **What's New**: 최근 Gaussian-Splat 기반의 Novel View Synthesis는 포토리얼리즘(like photorealism)을 달성할 수 있는 가능성을 보여주지만, 스파스 뷰(sparse view) 상황에서는 초기화 문제와 floaters에 대한 과적합(over-fitting)으로 인해 제한적이다. 이 연구에서는 고정밀한 카메라 포즈(pose) 추정과 조밀한 포인트 클라우드(point cloud)를 결합한 SPARS3R을 제안한다. SPARS3R은 Global Fusion Alignment와 Semantic Outlier Alignment의 두 단계로 구성되며, 이를 통해 기존 접근 방식보다 훨씬 향상된 성능을 보인다.

- **Technical Details**: SPARS3R의 첫 단계인 Global Fusion Alignment는 특정한 삼각측량(triangulated) 관계를 기반으로 Structure-from-Motion 기술을 사용하여 밀집된 포인트 클라우드를 스파스 포인트 클라우드에 매핑한다. 이 과정에서 RANSAC(Random Sample Consensus)를 적용하여 인라이어(inliers)와 아웃라이어(outliers)를 구분한다. 그 후, 아웃라이어 주변의 의미론적으로(coherent) 일관된 영역을 추출하고 이들 지역에서의 현지 정렬(local alignment)을 수행하여 정밀한 포인트 클라우드를 형성한다.

- **Performance Highlights**: SPARS3R은 세 가지 인기 있는 벤치마크 데이터셋에서 평가되었으며, 기존의 최첨단 방법들에 비해 정량적 및 시각적으로 상당한 개선을 확인했다. 특히, 스파스 이미지를 사용한 포토리얼리스틱 렌더링(photo-realistic rendering)에서 탁월한 성능을 발휘하였다. 새로운 구조와 알고리즘을 통해 SPARS3R은 빠르고 정확한 정보 제공을 가능하게 하여 실용적인 응용 프로그램에 기여할 것으로 기대된다.



### Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating Visual Hallucination (https://arxiv.org/abs/2411.12591)
- **What's New**: 이 논문에서는 Visual Inference Chain (VIC)라는 새로운 프레임워크를 제안하여 멀티모달 대형 언어 모델(MLLMs)의 고유한 문제점, 즉 시각적 입력에 의한 환각(hallucination) 현상을 해결합니다. VIC는 텍스트 컨텍스트에 기반한 추론 체인을 구성하고 나서 시각적 입력을 도입하는 방식으로, 이를 통해 교차 모달 편향을 줄이고 멀티모달 추론의 정확성을 개선합니다. 이 연구는 기존의 'thinking while looking' 접근 방식의 한계를 극복하고 'thinking before looking'이라는 새로운 패러다임을 제안합니다.

- **Technical Details**: VIC 프레임워크는 MLLMs의 추론 과정을 개선하기 위해, 이전의 'thinking while looking' 패러다임 대신 'thinking before looking' 전략을 채택합니다. 이를 통해 모델은 시각적 요소와 상관없이 기존의 기억과 맥락 지식을 활용하여 추론을 선행하도록 유도합니다. 이 방식은 인간의 인지 패턴과 유사하며, 앞서 계획을 세우는 과정이 시각적 입출력의 영향력을 줄이는 데 효과적임을 보여줍니다.

- **Performance Highlights**: VIC의 도입으로, 다양한 비전 관련 작업에서 탁월한 성과를 확인할 수 있습니다. 예를 들어, Gemini 1.5 Pro 모델은 MMVP 벤치마크에서 31.74%의 성능 향상을 이루었고, GPT-4o 미니 모델은 16.59% 증가했습니다. 전반적으로, GPT 시리즈 모델의 평균 성능 향상은 8.02%에 달하는 반면, Gemini 시리즈 모델은 7.19%의 평균 개선을 보였습니다.



### Debias your Large Multi-Modal Model at Test-Time with Non-Contrastive Visual Attribute Steering (https://arxiv.org/abs/2411.12590)
Comments:
          10 pages, 3 Figures, 3 Tables. arXiv admin note: text overlap with arXiv:2410.13976

- **What's New**: 이 논문에서는 Large Multi-Modal Models (LMMs)의 사회적 편향을 직접 제거할 수 있는 새로운 디바이징 프레임워크를 제안합니다. 기존의 방법들과는 달리, 우리의 방법은 단일 이미지와 대상 속성을 이용하여 훈련 없이 디바이징을 수행할 수 있습니다. 이를 통해 LMMs의 출력에서 보호 속성과 관련된 텍스트 생성을 최소화하고, 감정 개선 효과를 낼 수 있음을 보여냈습니다.

- **Technical Details**: 우리는 훈련이 필요 없는 방법으로서 LMM의 입력 이미지에서 단 한번의 gradient descent를 수행하여 편향된 표현을 제거하는 방법을 사용합니다. 이 과정에서 Fast Gradient Sign Method (FGSM)를 적용하여, 이미지를 통해 얻은 정보를 활용하여 방향성을 설정합니다. 이 접근법은 LLaVA와 Llama3와 같은 두 가지 주요 LMM 모델에서 효과적으로 보호 속성과 관련된 텍스트 생성을 저감시키는 데 성공했습니다.

- **Performance Highlights**: 실험 결과, 디바이징된 LMM의 출력 정확도는 기존의 편향 모델과 유사한 수준을 유지하며 발생하며, 언어 모델링 기능에서도 성능 저하가 없음을 확인했습니다. 또한, 디바이징을 통해 생성된 텍스트에서 감정의 형평성을 확보할 수 있음을 확인했습니다. 이러한 결과들은 LMM의 편향 제거가 모델 성능을 희생하지 않고도 가능하다는 것을 나타냅니다.



### ULTra: Unveiling Latent Token Interpretability in Transformer Based Understanding (https://arxiv.org/abs/2411.12589)
- **What's New**: 이 논문에서는 Transformer의 Latent Token (잠재 토큰) 표현을 해석하는 새로운 프레임워크를 제안합니다. 이 프레임워크를 통해 기존 모델을 추가적인 파인튜닝 없이도 제로샷(Zero-shot) 비지도형 의미 세분화(semantic segmentation)가 가능함을 입증했습니다. 이 방법은 Transformer 모델이 입력의 의미를 이해하는 본질적인 능력을 활용하며 기존의 전통적인 세분화 모델들을 능가하는 성능을 보여줍니다.

- **Technical Details**: Transformer 아키텍처와 Vision Transformers (ViTs)를 기반으로 한 이 연구는, Latent Tokens이 각각의 의미적 개념을 나타내도록 해석하는 방법을 제시합니다. 제안된 프레임워크는 친숙한 메커니즘 없이도 이미지 인지를 가능하게 하며, 이를 통해 하이퍼파라미터 조정 없이 이미지 세분화가 이루어질 수 있습니다. 우리는 또한 이 프레임워크가 대규모 언어 모델(LLMs)에서도 효과적임을 확인하여 텍스트 요약 작업에서의 적용 가능성을 검증했습니다.

- **Performance Highlights**: COCO-Stuff 데이터셋에서 67.2%의 정확도와 32.9%의 평균 교차 IoU(mIoU)를 기록하며, PASCAL VOC 데이터셋에서는 51.9%의 mIoU를 달성하여 기존 SOTA(State-of-the-Art) 성능을 초월했습니다. 이 연구는 기존의 비지도 학습 세분화 방법들보다 우수한 성능을 보이며, 많은 imstances가 필요한 기존 방법보다 더 효율적으로 동작합니다.



### Infrared-Assisted Single-Stage Framework for Joint Restoration and Fusion of Visible and Infrared Images under Hazy Conditions (https://arxiv.org/abs/2411.12586)
- **What's New**: 이 논문에서는 적외선-가시광선(IR-VIS) 이미지 융합을 위한 새로운 통합 학습 프레임워크를 제안합니다. 이 프레임워크는 안개가 낀 조건에서 가시 이미지를 복원하는 데 적외선 이미지를 효과적으로 활용합니다. 기존의 방법들은 두 단계로 나누어 진행되던 작업을 단일 단계에서 공동 훈련하여 경량화된 모델 구조로 개선하였습니다.

- **Technical Details**: 제안된 방법에서는 비공유 이미지 정보를 활용하여 프롬프트 생성 메커니즘을 설계하였고, 이는 선택 매트를 생성하여 적절한 후보 특징을 생성합니다. 적외선 지원 특징 복원 모듈을 통해 안개 밀도에 기반하여 후보 특징을 선택하여 복원과 융합을 동시에 진행합니다. 다단계 프롬프트 임베딩 융합 모듈을 통해 특징 보강을 수행하여 융합 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 두 단계 방식보다 뛰어난 성능을 보이며, 안개를 제거한 선명한 융합 결과를 생성합니다. 본 연구는 낮은 품질 이미지 복원 및 융합을 위한 새로운 관점을 제시하며, 기존 방식들이 가지던 한계를 극복하는 데 기여합니다.



### Leveraging MLLM Embeddings and Attribute Smoothing for Compositional Zero-Shot Learning (https://arxiv.org/abs/2411.12584)
- **What's New**: 이 논문에서는 Compositional Zero-Shot Learning (CZSL)을 위해 새로운 프레임워크인 TRIDENT를 제안합니다. 이 프레임워크는 Multimodal Large Language Model (MLLM) 임베딩과 attribute smoothing 기능을 기반으로 하여, 배경의 영향을 줄이고 다중 층의 특징을 활용하여 더 높은 일반화 능력을 제공합니다. 특히, 학습된 조합에 대한 과도한 자신감을 개선하기 위해 추가 속성을 생성하고, 이를 통해 모델이 다양한 속성을 배우도록 유도합니다.

- **Technical Details**: TRIDENT는 세 가지 주요 모듈로 구성되어 있습니다: 시각적 특징 추출, 속성-객체 분리, 그리고 특징 정렬입니다. 이 모듈들은 각각 Adaptive Aggregation 모듈을 사용하여 배경 잡음을 줄이고, 이미지 쌍의 공유 및 고유 특징을 분석하여 속성과 객체를 분리하며, 마지막 숨겨진 상태에서 MLLM의 임베딩을 사용하여 시각적 특징을 정렬합니다. 과도한 자신감 문제를 해결하기 위해, LLM을 활용하여 보조 속성을 생성하고 label smoothing을 통해 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: TRIDENT는 세 개의 CZSL 벤치마크에서 광범위한 실험을 통해 기존의 최신 성과를 초과달성했습니다. 연구 결과, 이 방법은 기존 접근 방식의 한계를 극복하며, 더 나은 속성-객체 인식과 조합 인식 능력을 보여줍니다. 추후 소스 코드도 발표할 예정이며, 이는 CZSL 연구에 있어 중요한 기여가 될 것으로 기대됩니다.



### Topological Symmetry Enhanced Graph Convolution for Skeleton-Based Action Recognition (https://arxiv.org/abs/2411.12560)
- **What's New**: 이 논문에서는 인간의 동작 인식을 위한 새로운 접근 방식으로, Topological Symmetry Enhanced Graph Convolution (TSE-GC)와 Multi-Branch Deformable Temporal Convolution (MBDTC)를 제안합니다. TSE-GC는 사람 몸의 대칭성을 고려하여 다양한 채널 그룹에서 독특한 토폴로지 학습을 가능하게 하며, MBDTC는 더 유연한 수용역을 통해 시간 종속성을 더 잘 모델링할 수 있도록 돕습니다.

- **Technical Details**: TSE-GC는 주어진 샘플에 대한 스케일 마스크를 학습하고, 이를 통해 공유된 토폴로지를 여러 개의 별도 채널 그룹으로 복제하여 대칭적으로 토폴로지 학습을 촉진합니다. MBDTC는 샘플링 위치에 학습 가능한 오프셋을 적용하여, 더욱 유연한 수용역을 제공하고 시간 종속성을 더욱 효과적으로 표현하는 모델입니다.

- **Performance Highlights**: 제안된 TSE-GCN 모델은 NTU RGB+D, NTU RGB+D 120 및 NW-UCLA의 세 가지 대규모 데이터셋에서 최첨단 방법들과 비교하여 적은 파라미터로 경쟁력 있는 성능을 발휘합니다. 특히 NTU RGB+D 120 데이터셋에서 90.0% 및 91.1%의 정확도를 기록했으며, 1.1M 파라미터와 1.38 GFLOPS의 연산 성능을 요구합니다.



### Recall and Refine: A Simple but Effective Source-free Open-set Domain Adaptation Framework (https://arxiv.org/abs/2411.12558)
- **What's New**: 이번 논문은 Source-free Open-set Domain Adaptation (SF-OSDA) 문제를 해결하기 위한 새로운 프레임워크인 Recall and Refine (RRDA)를 제안합니다. 기존의 SF-OSDA 방법들은 라벨이 있는 소스 데이터를 사용할 수 없는 상황에서도 성과를 내야 하는데, RRDA는 미지의 클래스에 대해 효과적으로 특징을 학습하도록 설계되었습니다. 이 프레임워크는 두 단계로 구성되어 있으며, 첫 번째 단계에서 타겟 특징으로 생성된 합성 샘플을 사용하여 미지의 클래스를 인식하는 모델의 능력을 향상시킵니다.

- **Technical Details**: RRDA의 첫 번째 단계에서는 K+K′ 결정 경계를 갖는 새로운 타겟 분류기를 도입하여 소스 도메인에서의 K 클래스와 미지의 클래스에 해당하는 추가 K’ 클래스를 학습합니다. 합성 샘플은 알려진 클래스를 위한 낮은 엔트로피와 미지의 클래스를 위한 높은 엔트로피를 가지도록 최적화되어, 최종적으로 K’ 카테고리로 클러스터링됩니다. 두 번째 단계에서는 SHOT 및 AaD와 같은 기존의 비소스 도메인 적응 방법들을 프레임워크에 통합하여 전체 모델을 타겟 도메인에 적응시킵니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 광범위한 실험을 통해 RRDA가 기존의 SF-OSDA 및 OSDA 방법들보다 크게 우수함을 입증하였습니다. 성능 분석 결과 동작 속도와 유연성이 우수하며, K′ 값의 증가에 따라 성능이 향상되는 경향을 보였습니다. 실험 결과는 RRDA가 미지의 클래스에 대한 분류 정확도를 높이는 데 효과적이라는 것을 명확히 보여줍니다.



### Contourlet Refinement Gate Framework for Thermal Spectrum Distribution Regularized Infrared Image Super-Resolution (https://arxiv.org/abs/2411.12530)
Comments:
          13 figures, 6 tables

- **What's New**: 본 연구는 적외선 이미지의 초해상도(super-resolution) 문제를 다루고 있으며, 기존 방법들이 이러한 특수한 이미지를 적절히 처리하지 못한다는 점을 지적합니다. 특히, Spectral Fidelity Loss를 제안하여 고주파와 저주파 성분의 분포를 규제함으로써 적외선 이미지의 특수한 세부 사항을 보존합니다. 또한, Contourlet 변환을 활용한 프롬프트 학습 최적화 기법을 사용하여 적외선 이미지의 특징을 효과적으로 복원합니다.

- **Technical Details**: 제안된 프레임워크는 다중 스케일 및 방향에 따른 적외선 스펙트럼 분해를 통해 고주파 서브 밴드를 캡처합니다. Contourlet refinement gate를 활용하여 적외선 모드 특유의 세부 사항을 복원하고, 스펙트럼 주파수 분포를 규제합니다. 두 단계의 프롬프트 학습 최적화 구조는 LR-Degradation에서 HR 특징을 학습하도록 모델을 유도합니다.

- **Performance Highlights**: 실험 결과, 본 방법이 기존의 초해상도 모델을 초월하여 시각적 및 지각적 작업에서 우수한 성능을 나타냅니다. 또한, 제안된 방법은 하류 작업에서 기계 인식을 크게 향상시키며, 적외선 이미지 초해상도 분야에 새로운 패러다임을 제시합니다.



### Rethinking Top Probability from Multi-view for Distracted Driver Behaviour Localization (https://arxiv.org/abs/2411.12525)
Comments:
          Computer Vision and Pattern Recognition Workshop 2024

- **What's New**: 본 논문에서는 자가 지도 학습(self-supervised learning)에 기반한 행동 인식(action recognition) 모델을 도입하여 운전 중 주의 분산 행동을 감지하고, 잠재적 행동 확률을 제공하는 방법을 제안합니다. 이러한 인식 모델의 결과는 다중 카메라 뷰를 활용한 제약 집합 전략을 통해 견고한 예측을 가능하게 하며, 마지막으로 조건부 후처리(conditional post-processing) 작업을 통해 주의 분산 행동과 행동의 시간 경계를 정밀하게 찾습니다. 이를 통해 AI City Challenge 2024에서 높은 성과를 달성했습니다.

- **Technical Details**: 우리의 시스템은 행동 인식 모델, 집합 전략(ensemble strategy), 조건부 후처리라는 세 가지 주요 구성 요소로 이루어져 있습니다. 이러한 방법들은 각 카메라 뷰로부터 수집된 비디오 데이터에서 주의 분산 행동을 인식하고, 이를 바탕으로 시간 경계를 구체화합니다. 특히, 자가 지도 학습 기법은 레이블이 부족한 데이터셋에서 더 강력하고 일반화된 특성을 제공하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 본 방법은 AI City Challenge 2024의 공개 리더보드에서 트랙 3에 대해 6위를 기록하며 높은 성능을 입증했습니다. 이는 다중 카메라의 다양한 뷰를 활용하여 잡음 클래스를 제거하고 신뢰할 수 있는 행동을 정확하게 식별할 수 있었음을 보여줍니다. 이러한 결과는 행동 인식 및 시간 위치 지정을 통합하여 더 나은 성능을 달성하는 새로운 접근 방식을 나타냅니다.



### PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy (https://arxiv.org/abs/2411.12510)
- **What's New**: 이번 논문에서는 PR-ENDO라는 새로운 프레임워크를 제안하여 내시경 실시간 조망 합성을 위한 3D 환경 재구성을 가능하게 합니다. PR-ENDO는 3D Gaussian Splatting 기술을 활용하여 내시경에서의 복잡한 촬영 조건을 극복하고, 조명과 조직 간의 상호작용을 포착하는 물리 기반 모델을 도입합니다. 이를 통해 내시경 검사 중에도 안정적인 재구성을 달성할 수 있으며, 다양한 카메라 회전에 대해 우수한 영상 품질을 보여줍니다.

- **Technical Details**: PR-ENDO는 카메라의 위치, 깊이 지도 및 환경의 점 구름 정보를 활용하여 재조명 모델을 도입합니다. 이 모델은 제한된 카메라 움직임과 비좁은 공간에서의 조명 조건을 처리하도록 설계되었습니다. 또한, diffuse MLP를 사용하여 시각적 불일치를 최소화하며 새로운 조망을 합성하는 기능을 추가하였습니다.

- **Performance Highlights**: 공개 데이터셋과 새로운 데이터셋을 사용하여 PR-ENDO의 성능을 평가한 결과, 기존의 접근 방식에 비해 향상된 영상 품질을 기록했습니다. 내시경 환경에서의 실제 적용을 위한 후보 기술로서, 이 모델은 진단 및 교육 목적 모두에 유용할 것으로 기대됩니다.



### SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Imag (https://arxiv.org/abs/2411.12471)
- **What's New**: 본 논문에서는 Snapshot Compressive Imaging (SCI) 분야에서 동적 장면의 3D 구조를 일관되게 복원하는 'SCIGS' 방법을 제안합니다. 이는 기존의 deep learning 기반 및 NeRF 기반 방법이 직면한 문제를 해결하기 위해 3DGS의 변형된 버전으로, 카메라 포즈와 가우시안 원시 좌표를 활용한 변환 네트워크를 개발했습니다. SCIGS는 단일 압축 이미지를 통해 명확한 3D 장면을 복원 가능하게 하여, 고속 촬영을 통한 동적 3D 장면 처리에 혁신을 가져옵니다.

- **Technical Details**: SCIGS는 카메라 포즈 스탬프와 3D 가우시안 변환 네트워크를 통해 카메라 포즈 최적화 문제를 해결합니다. 압축 이미지에서 변형된 가우시안을 분리하여 동적 장면에 적합하도록 처리하며, 고주파 필터를 적용하여 변환 과정에서 발생하는 아티팩트를 제거합니다. 이 방법은 초기 3D 가우시안으로부터 카메라 포즈를 최적화하기 위한 상호작용 과정을 포함하고 있습니다.

- **Performance Highlights**: 정적 및 동적 장면에서의 실험 결과, SCIGS는 기존의 SCI 복원 방법보다 높은 품질의 이미지를 생성하며, 동적 장면 복원에서의 성능 역시 우수함을 입증합니다. 특히, 단일 압축 이미지를 사용한 3D 장면 복원에 있어 현재 최고 수준의 성능을 기록하며, 향후 코드 공개를 통해 연구자들이 활용할 수 있도록 할 계획입니다.



### GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving (https://arxiv.org/abs/2411.12452)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 GaussianPretrain이라는 독창적인 프리트레인(Pre-training) 패러다임을 소개합니다. 이 방법은 기하학적(geometric) 정보와 질감(texture) 표현을 통합하여 효과적인 장면 이해를 촉진합니다. 3D Gaussian anchors를 LiDAR 포인트로 개념화하여, 장면의 상세한 공간 구조와 질감을 학습하며, NeRF 기반의 방법 UniPAD보다 40.6% 더 빠른 성능을 자랑합니다.

- **Technical Details**: GaussianPretrain은 LiDAR 깊이 정보를 활용한 마스크 생성기로 유효한 패치에서만 Gaussian 정보를 학습합니다. 또한, Ray-based 3D Gaussian anchor Guidance Strategy를 통해 LiDAR로 투영된 픽셀 각각에 대해 3D 공간에서 샘플링을 수행하며, 이러한 점들을 통해 장면의 기하학과 질감을 동시에 이해할 수 있게 합니다. 이 과정에서 RGB, 깊이 및 점유 속성을 적절히 복원합니다.

- **Performance Highlights**: GaussianPretrain은 다양한 3D 인식(task)에서 뛰어난 성능 향상을 보여줍니다. 3D 물체 탐지(NDS에서 7.05% 증가), HD 맵 구축(mAP 1.9% 향상), 점유 예측(0.8% 개선) 등의 주요 성과를 달성했습니다. 이러한 개선은 GaussianPretrain의 이론적 혁신과 실용적 잠재력을 강조하며, 자율 주행을 위한 시각적 프리트레인 발전을 촉진합니다.



### Frequency-Aware Guidance for Blind Image Restoration via Diffusion Models (https://arxiv.org/abs/2411.12450)
Comments:
          17 pages, 6 figures, has been accepted by the ECCV 2024: AIM workshop

- **What's New**: 이 논문에서는 블라인드 이미지 복원을 위해 주파수 인지 가이드 손실(frequency-aware guidance loss)을 제안합니다. 이 손실은 다양한 사전 훈련된 모델과 플러그 앤 플레이 방식으로 통합될 수 있으며, 공간과 주파수 도메인에서 콘텐츠 일관성을 동시에 최적화하는 특징이 있습니다. 이는 기존의 방법들과 차별화되는 점이며, 고주파 성분을 규제하여 이미지 품질을 향상시키는 데 기여합니다.

- **Technical Details**: 제안된 방법은 이산 웨이브렛 변환(discrete wavelet transform)을 기반으로 하여 이미지 복원 과정에서 발생하는 왜곡을 최소화합니다. 논문에서 소개된 모델은 고주파 가이드를 사용하여 샘플링 과정에서 생성된 이미지의 콘텐츠와 실제 이미지 간의 일관성을 보장합니다. 이 방법은 공간 도메인뿐만 아니라 주파수 도메인에서도 성능을 극대화하여, 복원 품질을 유연하게 조절할 수 있습니다.

- **Performance Highlights**: 실험 결과는 세 가지 블라인드 복원 작업에서 제안된 방법의 효과성을 입증합니다. 특히 블라인드 이미지 블러링에서 PSNR 점수가 3.72dB 향상된 것으로 나타났습니다. 다른 기존 방법과 비교했을 때, 이 방법은 디테일이 풍부하고 왜곡이 적은 이미지를 생성하는 뛰어난 능력을 보여주며, 최상의 시각적 품질을 달성했습니다.



### Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need (https://arxiv.org/abs/2411.12448)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)이 다양한 데이터 형태에 대해 일반적인 무손실 압축기 역할을 수행할 수 있음을 보여줍니다. 특히, LLM의 압축 성능이 무손실 이미지 압축 분야에서 새로운 가능성을 열어줄 수 있다는 점에서 주목받고 있습니다. 하지만 기존의 LLM 기반 압축기는 최첨단(STATE-OF-THE-ART, SOTA) 코덱들과의 성능 차이를 보이고 있어 보다 효과적인 압축 방법의 필요성이 제기됩니다.

- **Technical Details**: 연구팀은 P²-LLM이라고 하는 새로운 모델을 제안하였습니다. 이 모델은 다음 픽셀 예측을 기반으로 하며, 픽셀 수준의 선험적 정보와 LLM의 상황 내 능력, 픽셀 수준의 의미 보존 전략을 통합하여 다음 픽셀 예측 능력을 향상시키고자 합니다. 이를 통해 RGB 이미지의 비선형적인 상관관계를 효과적으로 모델링하며, 잠재적인 지능을 완전히 활용할 수 있는 새로운 무손실 이미지 압축 프레임워크를 구축하려고 합니다.

- **Performance Highlights**: P²-LLM은 기존 LLM 기반의 압축기보다 무손실 이미지 압축 성능을 크게 개선하였습니다. 특히, CLIC.m와 Kodak 데이터셋에서 각각 2.08 및 2.83 bit-per-subpixel(bpsp)을 달성하여 현재 최고 성능인 DLPR을 초월하는 결과를 보였습니다. 이는 언어 공간 내에서의 다음 픽셀 예측이 압축 성능에 있어 중요한 요소로 작용함을 시사합니다.



### Beyond Gaussians: Fast and High-Fidelity 3D Splatting with Linear Kernels (https://arxiv.org/abs/2411.12440)
- **What's New**: 최근 3D Gaussian Splatting (3DGS) 기술의 발전으로 새로운 시점 합성(novel view synthesis)에 있어 고품질 재구성과 실시간 렌더링이 가능해졌습니다. 그러나 플로팅 프리미티브(floating primitives)와 과재구성(over-reconstruction) 같은 블러링 아티팩트가 여전히 도전 과제로 남아 있습니다. 이 연구에서는 커널 설계(kernel design)의 역할이 충분히 탐구되지 않았음을 지적하며, 이를 해결하기 위한 새로운 접근을 제시하고 있습니다.

- **Technical Details**: 블러 아티팩트의 원인으로 가우시안 타원체(ellipsoids)의 소프트 경계를 식별하고, 이것이 고주파(high-frequency) 영역에서의 세부 정보 포착을 제한한다고 밝혔습니다. 이를 해결하기 위해, 3D Linear Splatting (3DLS) 방법을 도입하였으며, 이는 가우시안 커널을 선형 커널로 대체하여 특히 고주파 영역에서 더 날카롭고 더 정밀한 결과를 얻을 수 있게 합니다. 이러한 방식은 기존의 3DGS 접근 방식에 비해 커널 디자인의 중요성을 강조합니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 평가를 통해 3DLS는 최신의 충실도(fidelity)와 정확성(accuracy)을 보여주며, 기본 3DGS 모델에 비해 30%의 FPS 개선을 기록하였습니다. 이러한 성능 향상은 실시간 렌더링에 있어서 중요한 이점을 제공하며, 이러한 구현은 논문 수락 시 공개될 예정입니다.



### CV-Cities: Advancing Cross-View Geo-Localization in Global Cities (https://arxiv.org/abs/2411.12431)
Comments:
          Datasets and codes are available, accepted by IEEE JSTARS

- **What's New**: 이 논문은 새로운 Cross-view geo-localization (CVGL) 프레임워크를 제안합니다. 이 프레임워크는 비전 기초 모델 DINOv2와 고급 피처 믹서를 통합하여 위치 결정 정확도를 크게 향상시킵니다. 또한 CV-Cities라는 새로운 글로벌 데이터세트를 구축하여 다양한 복잡한 시나리오에서의 성능을 개선합니다.

- **Technical Details**: 논문에서는 대칭 정보 손실(InforNCE loss)과 이웃 샘플링 및 동적 유사성 샘플링 전략을 도입하여 모델의 정확도를 높입니다. DINOv2 모델을 기반으로 하는 프레임워크는 이미지 전처리를 필요로 하지 않아 구조가 단순하고 정확성이 높습니다. CV-Cities 데이터세트는 223,736개의 지리 위치 데이터가 포함된 지상-위성 이미지 쌍을 제공하며, 6대륙에 걸쳐 16개 도시의 복잡한 시나리오를 포함하고 있습니다.

- **Performance Highlights**: CV-Cities 데이터세트를 사용하여 교육된 프레임워크는 여러 시험 도시에서 높은 위치 결정 정확도를 보여주었습니다. 기존 CVGL 방법들보다 성능이 우수하며, 다양한 복잡한 환경에서 일관되게 높은 정확도를 나타냅니다. 모델의 일반화 성능이 크게 향상되었으며, CV-Cities는 CVGL의 도전적인 벤치마크로 자리잡았습니다.



### Motif Channel Opened in a White-Box: Stereo Matching via Motif Correlation Graph (https://arxiv.org/abs/2411.12426)
- **What's New**: 본 논문에서는 모티프 상관 그래프(Motif Correlation Graph, MCG)를 소개하며, 이를 통해 반복 텍스처를 포착하여 기하학적 구조를 재구성하는 새로운 스테레오 매칭 방법 MoCha-V2를 제안합니다. MoCha-V2는 기하학적 구조를 보다 해석 가능한 방식으로 학습하며, 다양한 주파수 도메인에서의 특징을 통합하여 스테레오 매칭 과정에서 기하학적 구조를 복원합니다. 실험 결과 MoCha-V2는 Middlebury 벤치마크에서 1위를 차지했습니다.

- **Technical Details**: MoCha-V2는 모티프 채널을 활용하여 반복적인 기하학적 특징을 추출하고, 이를 기반으로 가장자리 세부 정보를 복원하는 방법론입니다. 특히, MCG를 통해 슬라이딩 윈도우의 가중치를 결정하여 백박스(black-box) 방식의 학습 문제를 해결했습니다. 또한, 웨이블릿 도메인에서 저주파 정보를 더욱 강조하여 전체 매칭 성능을 향상시킵니다.

- **Performance Highlights**: MoCha-V2는 기하학적 구조를 학습하고 복원하는 능력을 바탕으로 스테레오 매칭 분야에서 뛰어난 성능을 보여주었습니다. 현재 Middlebury 벤치마크에서 1위, KITTI 2012에서 2위를 기록하며 그 우수성을 입증하였습니다. 이는 기존 방법들과 비교하여 더 정확한 세부 매칭을 가능하게 한다는 점에서 중요한 발전입니다.



### Classification of Geographical Land Structure Using Convolution Neural Network and Transfer Learning (https://arxiv.org/abs/2411.12415)
- **What's New**: 이 연구는 위성 이미지에서 지리적 토지 구조를 자동으로 분류하는 심층 학습(Deep Learning) 기반 접근 방식을 개발했습니다. 이는 도시 계획, 환경 모니터링, 재난 관리 등 다양한 응용 분야에 응용될 수 있는 가능성을 보여줍니다. 특히, 인력의 노동력을 최소화하고, 비용과 소모되는 시간을 줄이는 방법론을 제시합니다.

- **Technical Details**: 연구에서는 MLRSNet에서 획득한 위성 이미지 데이터셋을 사용하였습니다. 세 가지 아키텍처인 CNN, ResNet-50, Inception-v3의 성능을 비교하였고, 각 모델에 대해 Adam, SGD, RMSProp 세 가지 옵티마이저(Optimizer)를 사용했습니다. 고정된 에포크(Epoch) 수인 100 에포크 동안 배치 크기(batch size)는 64로 설정하여 훈련을 진행했습니다.

- **Performance Highlights**: ResNet-50은 ADAM 옵티마이저를 사용하여 76.5%의 정확도를 달성했으며, Inception-v3는 RMSProp을 통해 93.8%의 정확도를 기록했습니다. 제안된 접근 방식인 CNN은 RMSProp 옵티마이저를 사용하여 94.8%의 최고 성능을 나타냈습니다. CNN 모델의 정확도, 재현율(Recall), F1 점수의 철저한 검토 결과, 다양한 지형 형상을 정밀하게 감지하는 데 있어 그 신뢰성과 성능이 뛰어남을 확인했습니다.



### DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding (https://arxiv.org/abs/2411.12355)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문에서는 장기 비디오 이해에서 LLM(대형 언어 모델)의 메모리 효율적인 비디오 인코딩을 위한 다이나믹 협력 네트워크인 DynFocus를 제안합니다. 이를 통해 비디오의 시각적 세부정보를 효과적으로 유지하면서도 토큰 수를 줄이는 동적 인코딩 방식을 도입합니다. 특히, 이 모델은 DPE(동적 이벤트 프로토타입 추정)와 CCE(압축 협력 인코딩)라는 두 가지 주요 모듈로 구성됩니다.

- **Technical Details**: DPE 모듈은 의미 있는 프레임을 동적으로 선택하여 질문 응답에 필요한 중요한 정보를 고려합니다. 이후 CCE 모듈은 의미 있는 프레임을 세밀한 시각적 텍스처로 인코딩하고, 중복된 프레임은 대략적인 인식을 위한 토큰으로 축약하여 LLM이 더 넓은 시간적 단서를 포착할 수 있도록 합니다. 이는 생물학적 원리에서 영감을 받아 설계된 것입니다.

- **Performance Highlights**: DynFocus는 공개된 다섯 개의 벤치마크에서 평가되었으며, 실험 결과는 이 방법이 경쟁력 있는 성능을 거두었음을 일관되게 증명합니다. 특히, 다양한 질문에 대해 적절한 프레임을 동적으로 식별함으로써 비디오의 복잡한 내용을 더욱 효과적으로 이해할 수 있는 가능성을 보여줍니다.



### DiM: $f$-Divergence Minimization Guided Sharpness-Aware Optimization for Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2411.12350)
Comments:
          8page

- **What's New**: 반자동 데이터 주석 문제를 해결하기 위해 반지도 학습(semi-supervised learning, SSL) 기술이 각광받고 있습니다. 특히 의료 이미지 분할(medical image segmentation) 분야에서 SSMIS(semi-supervised medical image segmentation)가 주목 받고 있으며, 정확한 주석 데이터의 필요성을 줄이는 방법으로 연구되고 있습니다. 본 연구에서는 SAM(sharpness-aware minimization) 기술을 활용하여 모델의 일반화 성능을 향상시키는 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 $f$-divergence 최소화(minimization)를 통해 SAM의 한계를 극복하는 새로운 방법인 DiM(f-divergence minimization guided sharpness-aware optimization)을 제안합니다. 이 방법은 모델 파라미터의 민감도를 조정하고, 다양한 데이터셋에 대한 적응성을 향상시킵니다. $f$-divergence를 도입함으로써 DiM 방법은 소스 데이터셋과 타겟 데이터셋 간의 균형 잡힌 성능을 개선하고, 소스 데이터셋 과적합(overfitting)을 방지하는 데 기여합니다.

- **Performance Highlights**: DiM 방법은 SSMIS 벤치마크에서 최신 기술들에 비해 우수한 성능을 입증하였습니다. 특히, 모델의 안정성과 다양한 도메인 간의 적응성을 크게 향상시키며, 기존 SAM 방식과의 비교에서 더욱 효과적인 성능 향상을 보여주었습니다. 이러한 연구 결과는 의료 이미징 분야에서 반지도 학습의 적용 가능성을 넓히는 중요한 발전이 될 것입니다.



### Accelerating UMAP for Large-Scale Datasets Through Spectral Coarsening (https://arxiv.org/abs/2411.12331)
- **What's New**: 이 논문에서는 UMAP의 속도를 혁신적으로 가속화하기 위한 새로운 접근법을 제안합니다. 제안된 방법은 데이터셋의 크기를 크게 줄이면서도 필수적인 manifold 구조를 보존하는 고급 스펙트럼 압축 기법을 통해 이루어집니다. 이를 통해 UMAP은 품질을 유지하면서 훨씬 빠르게 작동할 수 있습니다.

- **Technical Details**: UMAP은 차원 축소와 데이터 시각화를 위한 강력한 도구로, 고차원 데이터의 복잡한 manifold 구조를 파악하는 데 효과적입니다. 그러나 대규모 데이터셋에 적용될 때 계산적 제약이 발생하는데, 이는 근사 이웃 검색 및 그래프 구성 단계에서 발생하는 O(N log N) 및 O(N²) 시간 복잡도 때문입니다. 본 연구에서는 UMAP의 효율성과 확장성을 해결하기 위해 스펙트럼 보존 데이터 압축 방법을 제안합니다.

- **Performance Highlights**: 제안된 방법은 USPS와 같은 실제 데이터셋에서 실험을 통해 데이터의 신뢰성을 해치지 않으면서도 상당한 데이터 축소를 달성하는 능력을 보여줍니다. 이를 통해 UMAP이 대규모 데이터셋을 더 효율적으로 처리할 수 있도록 하고, 사용자는 계산 제약이나 품질 요구에 따라 원하는 압축 비율을 지정할 수 있는 유연성을 누릴 수 있습니다.



### Enhancing Blind Source Separation with Dissociative Principal Component Analysis (https://arxiv.org/abs/2411.12321)
Comments:
          1. 13 pages with 6 figures, this work has not bee published before. 2. The paper is yet to be peer-reviewed and I am planning to submit it to IEEE Transactions on Image Processing. 3. There is no supplementary material. 4. There is no funding for this work as of now

- **What's New**: 이번 연구에서는 Sparse Principal Component Analysis (sPCA) 의 한계를 극복하고 독립 성분 분석(ICA)과의 효과적인 통합을 위해 새로운 방법론인 Dissociative PCA (DPCA1, DPCA2)를 소개하였습니다. DPCA는 기존 sPCA의 해석 가능성을 유지하면서도 소스 분리(source extraction) 능력을 획기적으로 향상시키는 알고리즘입니다. 이 연구의 주요 목적은 PCA의 구성 요소 간의 상호 의존성을 최소화하는 것입니다.

- **Technical Details**: DPCA는 적응형 및 고정 임계값(thresholding) 기법과 경량화된 경량 회귀(coordinate descent) 방식을 통해 제안 모델을 동적으로 최적화하는 두 가지 알고리즘으로 구성됩니다. 한편, 전통적인 고유 벡터에 집중하는 대신, DPCA는 여러 고유 벡터를 협력적으로 결합하여 SVD 변량 내의 상호 의존성을 분리하는 데 주력합니다. 이는 각 SVD 변량에서 정교한 PCs(주성분)와 LVs(로딩 벡터)를 생성하여 데이터를 보다 정확하게 표현하게 만듭니다.

- **Performance Highlights**: DPCA 알고리즘은 기능적 자기공명영상(fMRI) 소스 검색, 전경-배경 분리, 이미지 재구성, 이미지 인페인팅을 포함한 네 가지 다양한 영상 응용에서 뛰어난 성능을 입증하였습니다. 이 방법은 기존의 PCA+ICA, PPCA+ICA, SPCA+ICA, PMD, GPower와 같은 전통적인 접근 방식을 초월하여 보다 신뢰성 있는 소스 분리를 가능하게 합니다. 이 연구는 특히 복잡한 공간적 중첩이 있는 상황에서의 성능 향상을 강조합니다.



### CLIP Unreasonable Potential in Single-Shot Face Recognition (https://arxiv.org/abs/2411.12319)
- **What's New**: 이번 연구는 얼굴 인식(Face Recognition) 기술에 CLIP(Contrastive Language-Image Pretraining) 모델을 활용하여 새로운 접근 방식을 소개합니다. 기존 방식은 얼굴의 주요 특징을 추출하고 이를 데이터베이스와 비교하는 과정을 거쳤으나, CLIP는 이러한 과정 없이 단일 샷 파인튜닝(single-shot finetuning)만으로 낮은 오탐률(false positive rate)을 달성하는 것을 보여주고 있습니다. 이 연구는 얼굴 인식 솔루션을 간소화하고 지속적으로 존재하는 문제를 해결할 수 있는 CLIP의 잠재력을 드러냅니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 자원봉사자들의 고해상도 얼굴 이미지를 포함하고 있으며, 각 얼굴은 개인별로 정렬되어 있습니다. 얼굴 인식 실험을 위해 SCRFD(face detection model) 모델을 활용하여 이미지에서 얼굴 영역을 정확하게 감지하고, 눈, 코, 입의 중요한 키포인트를 추출했습니다. 이를 통해 이미지 간 얼굴의 수직 정렬을 보장하고, 고유한 비율을 유지하면서 데이터 표현을 일관되게 하였습니다.

- **Performance Highlights**: 클립 모델의 파인튜닝 과정에서 얼굴 인식을 이미지 분류(image classification) 문제로 간주하여 처리했으며, 모델의 이미지 인코더 파라미터를 동결하고 텍스트 인코더만을 이용해 학습을 진행했습니다. 결과적으로 CLIP 모델은 기존의 전통적인 얼굴 인식 알고리즘에 비해 현실 세계에서의 오탐률을 크게 낮추는데 성공했습니다. 즉, CLIP의 다모드(multi-modal) 디자인은 향후 다양한 얼굴 인식 과제의 성능 개선에 기여할 것으로 기대됩니다.



### DGTR: Distributed Gaussian Turbo-Reconstruction for Sparse-View Vast Scenes (https://arxiv.org/abs/2411.12309)
Comments:
          Code will released on our ![project page](this https URL)

- **What's New**: 이 논문은 DGTR이라는 새로운 분산 프레임워크를 제안하여, 희소 뷰( sparse-view) 환경에서도 고품질의 장면 재구성을 가능하게 합니다. 이 방법은 드론이 각기 독립적으로 장면의 특정 영역을 처리하고, Gaussian primitives를 이용하여 빠르게 NVS( Novel View Synthesis)를 수행할 수 있도록 설계되었습니다. 또한, 전통적인 방법들이 요구하는 긴 훈련 시간을 크게 단축시킬 수 있습니다.

- **Technical Details**: DGTR는 드론들이 비슷한 이미지를 재구성할 수 있도록 장면을 여러 지역으로 나누어 각 드론이 독립적으로 작업을 수행하게 합니다. 이를 통해 로컬 Gaussian 모델을 훈련시키고, depth regularization을 이용하여 Gaussian의 정확도를 높이며 과적합을 방지합니다. 추가적으로, 중앙 서버로 Gaussian 모델을 전송하여 최종 모델로 집계하는 distillation 기반의 모델 집계 메커니즘을 적용합니다.

- **Performance Highlights**: 제안된 방법은 대규모 장면 재구성에서 속도와 품질 모두에서 기존 방법보다 우수한 결과를 달성합니다. 이 프레임워크는 실제 작업 현장에서 수 분 내에 고품질 재구성이 가능함을 보여주며, 빠른 훈련 시간과 높은 스케일 가능성을 제공합니다. 시뮬레이션을 통해 효과성을 입증하였고, 코드도 공개될 예정입니다.



### Diffusion Product Quantization (https://arxiv.org/abs/2411.12306)
- **What's New**: 이 연구는 모델 크기를 줄이면서 성능을 유지하기 위한 다양한 양자화 기법을 탐구합니다. 특히 Diffusion Product Quantization (DPQ)이라는 새로운 방법론을 제안하여 diffusion 모델을 극단적으로 낮은 비트로 압축하는 데 성공하였습니다. 기존의 vector quantization (VQ)의 한계를 극복하기 위해, 제품 양자화(product quantization)를 신규 적용하여 고차원 벡터에 대한 정밀한 양자화를 가능케 하는 방법을 개발하였습니다.

- **Technical Details**: DPQ는 기존의 vector quantization 기법을 개선한 것으로, 벡터 공간을 여러 저차원 서브 공간으로 분할하여 각 서브 공간의 코드북을 독립적으로 양자화합니다. 이를 통해 새로운 코드북의 수를 증가시키면서도 메모리 소모를 최소화할 수 있습니다. 또한, 할당 및 코드북을 조정하는 end-to-end 보정(calibration) 기법을 도입하여 양자화 오차를 줄이고, DDPM 손실을 통해 코드북을 최적화합니다.

- **Performance Highlights**: DPQ는 DiT 모델에 적용되며 1비트 압축을 달성하여 모델 크기를 24배 이상 줄이면서도 생성 성능이 뛰어난 결과를 얻었습니다. 기존의 VQ 방식보다 전 비트폭에서 성능이 크게 향상되어, 압축 기술의 모든 범위에서 우수한 성능을 발휘하고 있습니다. 이로 인해, 저비트 폭 환경에서도 안정적인 생성 결과를 보장합니다.



### Physics-Guided Detector for SAR Airplanes (https://arxiv.org/abs/2411.12301)
- **What's New**: 이 논문에서는 SAR(Synthetic Aperture Radar) 비행기 탐지 및 인식을 위한 새로운 물리 기반 탐지기 학습 패러다임(Physics-Guided Detector Learning Paradigm, PGD)을 제안합니다. PGD는 SAR 비행기의 불규칙한 구조와 변동성을 종합적으로 고려하여 탐지 성능을 향상시키고 있습니다. 이 패러다임은 다양한 기존 딥러닝 기반 탐지기에 적용 가능하며, 세 가지 주요 구성 요소인 PGSSL (Physics-Guided Self-Supervised Learning), PGFE (Physics-Guided Feature Enhancement), PGIP (Physics-Guided Instance Perception)를 포함합니다.

- **Technical Details**: 제안된 PGD 학습 패러다임은 물리적 산란 구조를 바탕으로 SAR 비행기의 특정 특성을 학습합니다. PGSSL은 다양한 SAR 비행기 목표에 대한 산란 분포를 예측하는 자가 감독 학습 작업을 구성하여 다양한 비행기 모델을 효과적으로 포착합니다. PGFE 모듈은 PGSSL에서 학습된 물리 정보로 다중 스케일 특징을 개선하고, PGIP 모듈은 각 비행기 인스턴스의 세밀한 표현을 학습하여 세밀한 탐지를 돕습니다.

- **Performance Highlights**: PGD 모델은 SAR-AIRcraft-1.0 데이터셋에서 90.7% mAP를 달성하여 최신 성능을 기록하였으며, 기존 탐지기보다 최대 3.1% mAP 향상을 보여주었습니다. 다양한 검증 모델인 PGD와 PGD-Lite의 실험을 통해 PGD의 유연성과 효과성을 입증했습니다. 이러한 성능 향상은 PGD 학습 패러다임의 모델 독립적 능력을 증명하며, SAR 비행기 탐지 성능을 크게 개선할 잠재력을 가지고 있습니다.



### Generative Timelines for Instructed Visual Assembly (https://arxiv.org/abs/2411.12293)
- **What's New**: 이번 연구에서는 비전 문맥에서 자연어 지시로 시각적으로 타임라인(visual timelines)을 조작하는 새로운 접근 방식인 'Instructed visual assembly'를 제안합니다. 이 과정은 사용자가 영상 편집에 필요한 지시를 자연어로 입력하면 시스템이 시각 요소를 자동으로 수정합니다. 특히, 비전 콘텐츠의 이해와 자연어 지시의 해석을 통합하여 다양한 사용자가 비디오를 쉽게 편집할 수 있도록 돕습니다.

- **Technical Details**: 우리는 'Timeline Assembler'라는 생성 모델을 제안하며, 이는 대규모 다중모델 언어 모델(large multimodal language model)을 활용하여 시각적 콘텐츠를 처리하고 타임라인 편집 지시를 정확하게 해석합니다. 이 모델은 각 이미지 및 비디오 클립에 대한 고유 식별자와 시각적 표현을 사용하여 다중 시각 자산을 처리하고, LLM의 출력 토큰을 시각 요소와 직접 연결하는 구조로 설계되었습니다. 이를 통해 사용자로부터의 보다 직관적인 자연어 지시 수행이 가능해졌습니다.

- **Performance Highlights**: 연구에서 우리는 두 개의 데이터 세트를 구축하였고, 'Timeline Assembler'가 기존의 강력한 LLM들(GPT3.5 포함)보다 우수한 성능을 보이는 것을 확인하였습니다. 이 모델은 다양한 길이의 타임라인에서도 일관되게 뛰어난 성능을 발휘하며, 여러 복잡한 지시를 동시에 실행할 수 있는 능력을 갖추었습니다. 이러한 결과는 사용자 지향적인 비디오 편집 인터페이스를 통한 자연어 조작의 가능성을 크게 확장합니다.



### SSEditor: Controllable Mask-to-Scene Generation with Diffusion Mod (https://arxiv.org/abs/2411.12290)
- **What's New**: 최근 3D 확산 기반의 의미 장면 생성 기술이 주목받고 있습니다. 기존 방식은 무조건적인 생성을 기반으로 하며, 장면 편집 시 여러 단계의 재샘플링을 요구하여 제어 가능성과 유연성이 크게 제한됩니다. 이를 해결하기 위해 제안된 SSEditor는 지정된 목표 카테고리를 생성하면서도 여러 단계의 재샘플링이 필요 없는 제어 가능한 Semantic Scene Editor입니다.

- **Technical Details**: SSEditor는 두 단계의 확산 기반 프레임워크를 사용하며, 첫 번째 단계에서는 3D 장면 오토인코더를 사용해 잠재적인 트리플레인 피처를 학습합니다. 두 번째 단계에서는 마스크 조건형 확산 모델을 통해 맞춤형 3D 의미 장면 생성을 수행합니다. 또한, 기하학-의미 융합 모듈(Geometric-Semantic Fusion Module)이 도입되어 모델이 기하학적인 정보와 의미 정보를 효과적으로 학습하도록 돕습니다.

- **Performance Highlights**: SemanticKITTI 및 CarlaSC 데이터셋에서의 실험 결과, SSEditor는 목표 생성의 제어 가능성과 유연성뿐만 아니라 의미 장면 생성 및 재구성의 품질 면에서도 이전 접근 방법들을 능가함을 보여주었습니다. 더 나아가, 보지 못한 Occ-3D Waymo 데이터셋에서의 실험 결과는 SSEditor가 새로운 도시 장면을 생성할 수 있는 능력을 입증하였으며, 3D 장면의 신속한 구축을 가능하게 합니다.



### HouseLLM: LLM-Assisted Two-Phase Text-to-Floorplan Generation (https://arxiv.org/abs/2411.12279)
- **What's New**: 이 논문에서는 두 단계의 텍스트-플로어플랜 생성 방법론을 제안합니다. 이 방법은 Large Language Model (LLM)을 사용하여 초기 레이아웃(Layout-LLM)을 생성한 후, 조건부 확산 모델(conditional diffusion model)을 통해 최종 플로어플랜으로 다듬습니다. Chain-of-Thought 접근 방식을 활용하여 사용자 텍스트 사양에 기반한 프롬프트를 제공함으로써, 보다 사용자 친화적이고 직관적인 주택 레이아웃 설계를 가능하게 합니다. 최종 플로어플랜은 더욱 정확하며 사용자 요구를 더 잘 충족합니다.

- **Technical Details**: 본 연구는 주택 설계의 두 단계 생성 방식을 채택하여, Multimodal Large Language Model (MLLM)을 사용하여 사용자와 상호작용하고 초기 설계를 생성하는데, 이를 Layout-LLM이라고 명명합니다. 두 번째 단계에서는 확산 모델(diffusion model)을 통해 초기 설계를 더 정교한 최종 주택 레이아웃(Layout-Final)으로 다듬습니다. 이 과정은 Chain-of-Thought 기술을 사용하여 LLM이 사용자 요구 사항을 충족하는 집의 레이아웃을 생성하도록 유도하며, 객체 크기와 정렬에는 다소의 불완전함이 있을 수 있지만 결과를 최적화하기 위한 조건부 확산 모델도 설계하였습니다.

- **Performance Highlights**: 실험 결과, 본 방법은 RPlan 데이터셋에서 최첨단 성능을 달성했으며, 특히 HouseDiffusion 방법에 비해 모든 지표에서 뛰어난 결과를 보였습니다. 다양성 면에서는 28%의 개선을 이뤘습니다. 이러한 결과는 본 접근 방식이 실용적인 주택 설계 응용 프로그램에서 효과적임을 검증합니다. 최종적으로, 본 연구는 LLM을 활용한 새로운 두 단계 플로어플랜 생성 접근 방식을 제안하고, 이를 통해 사용자 분해 부담을 크게 줄일 수 있음을 보여줍니다.



### KDC-MAE: Knowledge Distilled Contrastive Mask Auto-Encoder (https://arxiv.org/abs/2411.12270)
- **What's New**: 이 연구는 Self-supervised Learning(SSL) 패러다임을 발전시키기 위해 대조 학습(Contrastive Learning), 자기 증류(Knowledge Distillation) 및 마스크 데이터 모델링(Masked Data Modelling)이라는 세 가지 주요 SSL 프레임워크를 결합하여 공동 표현을 학습하는 방법을 제안합니다. 이를 위해 KDC-MAE라는 새로운 SSL 아키텍처를 제안하고, 모듈 간의 대응성을 학습하기 위한 보완적 마스킹 전략을 도입했습니다. 실험 결과, 대조 마스킹 대응과 KD 학습 목표가 여러 작업에서 다양한 모달리티의 학습 성능을 향상시키는 데 기여했음을 보여줍니다.

- **Technical Details**: 제안된 방법은 Transformer 레이어를 기반으로 한 오디오 및 비디오 인코더를 포함하며, 두 모달리티의 인코더는 동일한 아키텍처를 사용하고 단일 디코더를 공유합니다. 오디오 및 비디오 데이터는 75%의 마스킹 비율을 유지하며 처리됨으로써, 강화된 대조 손실을 계산하고 모듈 간의 상호 연관성을 탐색합니다. 마스킹된 토큰은 학습 가능한 상태로 처리되며, 각 복원된 토큰은 모달리티에 따라 적절한 위치에 배치됩니다.

- **Performance Highlights**: 여기서 제안한 KDC-MAE 구조는 다양한 모달리티들을 함께 학습하는 데 최적화된 조합을 통해 학습 성능을 높인 것으로 평가됩니다. 여러 다운스트림 작업에서 사전 학습된 모델을 사용하여 더 나은 결과를 얻었으며, ablation study를 통해 모달리티 간의 학습 시너지를 입증했습니다. 이러한 방식으로 제안된 방법은 SSL의 다양한 목표를 효과적으로 결합함으로써 콘텐츠의 맥락 정보를 보다 일반화할 수 있는 가능성을 보여줍니다.



### Prototype Optimization with Neural ODE for Few-Shot Learning (https://arxiv.org/abs/2411.12259)
Comments:
          An extended version of metanode: prototype optimization as a neural ode for few-shot learning. arXiv admin note: text overlap with arXiv:2103.14341

- **What's New**: 본 논문은 Few-Shot Learning (FSL)의 프로토타입 편향 문제를 해결하기 위한 새로운 메타 학습 기반의 프로토타입 최적화 프레임워크를 제안합니다. 기존의 프로토타입 정정 방식이 아닌, 프로토타입 최적화 문제로 접근하여, 메타 최적화 기법을 이용해 편향을 줄이는 방법론을 강조합니다. 특히, Neural Ordinary Differential Equation (ODE)-기반의 메타 최적화 기법인 MetaNODE를 통해 프로토타입을 더 정확히 개선할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안하는 MetaNODE는 GDA(Gradient Descent Algorithm)를 사용하여 프로토타입 최적화 과정을 연속적으로 모델링합니다. 이를 위해, 정확한 프로토타입 동역학을 추정하기 위한 GradNet 네트워크를 설계하고, RK4 기반의 ODE 솔버를 사용하여 최적 프로토타입을 도출합니다. 아울러, E2MetaNODE로 확장하여 E2GradNet 및 E2Solver 모듈을 통해 컴퓨테이션 효율성을 향상시키고 있습니다.

- **Performance Highlights**: 제안한 방법들은 기존의 FSL 방법들에 비해 탁월한 성능을 달성하였고, E2MetaNODE는 성능 저하 없이 계산 효율성을 크게 향상시켰다는 점에서 유의미한 결과를 보여줍니다. 본 연구는 transductive 및 inductive FSL 설정에서 폭넓은 실험을 통해 그 유효성을 입증하였으며, 다양한 데이터셋에서 성능 평가를 수행하였습니다.



### ADV2E: Bridging the Gap Between Analogue Circuit and Discrete Frames in the Video-to-Events Simulator (https://arxiv.org/abs/2411.12250)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 이벤트 카메라의 픽셀 회로에 대한 심층 분석을 통해 보다 신뢰할 수 있는 이벤트 데이터를 생성하는 새로운 방법, ADV2E를 제안합니다. 이 방법은 신호의 아날로그 특성을 시뮬레이터 디자인에 통합하여 현실적인 합성 이벤트를 생성합니다. 실험 결과는 고대비 장면에서도 시뮬레이션된 이벤트 데이터의 신뢰성을 검증하였으며, 이는 딥 뉴럴 네트워크(Deep Neural Networks)가 시뮬레이션된 데이터로부터 실제 데이터로 잘 일반화됨을 보여줍니다.

- **Technical Details**: ADV2E 시뮬레이터는 아날로그 필터링 및 독립적인 컷오프 주파수를 포함하여 DVS의 근본적인 아날로그 행동을 통합합니다. 아날로그 필터링을 통해 조명 강도로부터 이벤트로의 전환이 효과적으로 이루어지며, 컷오프 주파수는 비디오 프레임 속도와 무관하게 설계되어, 시간 지연을 줄입니다. 이러한 방식은 고대비 장면에서 발생할 수 있는 인위적인 이벤트의 질 저하를 방지합니다.

- **Performance Highlights**: ADV2E는 의미론적 분할(semantic segmentation) 및 이미지 재구성(image reconstruction)과 같은 주요 비전 태스크에서 우수한 성능을 보입니다. 실험 결과, 제안된 시뮬레이터에 의해 생성된 합성 이벤트는 훈련에 효과적이며, DNN 훈련의 성능이 실제 이벤트 데이터에 잘 일반화된다는 사실이 확인되었습니다. 이러한 결과는 ADV2E의 설계가 기존의 이벤트 시뮬레이터와 비교하여 뛰어난 효과성을 가지고 있음을 증명합니다.



### Neuro-3D: Towards 3D Visual Decoding from EEG Signals (https://arxiv.org/abs/2411.12248)
- **What's New**: 이 논문에서는 EEG 신호를 통해 3D 시각 인식을 디코딩하는 새로운 작업을 제안합니다. 이를 위해, 12명의 참가자가 72개의 3D 객체를 보는 동안 수집된 EEG 데이터를 포함하는 EEG-3D라는 혁신적인 데이터셋을 소개합니다. 또한, 이 데이터셋을 기반으로 EEG 신호를 활용한 Neuro-3D라는 3D 시각 디코딩 프레임워크를 제안하여, 3D 객체의 형태와 색상을 복원하는 과정을 설명합니다.

- **Technical Details**: Neuro-3D 프레임워크는 정적 및 동적 자극으로부터 유도된 EEG 특징을 통합하여 강력한 신경 표현을 학습합니다. 이를 위해 Dynamic-Static EEG-Fusion Encoder와 Colored Point Cloud Decoder를 활용하며, EEG 임베딩을 기반으로 3D 객체의 형상과 색상 정보를 생성합니다. 또한, 대조 학습(contrastive learning)을 통해 EEG 특징과 시각 특징의 정렬을 강화하여 시각 인식을 향상시킵니다.

- **Performance Highlights**: 실험 결과, Neuro-3D는 높은 충실도로 색상 있는 3D 객체를 재구성할 뿐만 아니라 효과적인 신경 표현을 학습해 뇌 영역 분석에 대한 통찰을 제공합니다. EEG-3D 데이터셋과 코드는 공개될 예정이며, 이는 뇌의 3D 지각 메커니즘 연구에 기여할 것입니다. 이 논문은 뇌의 3D 비주얼 디코딩 연구를 한 단계 끌어올리는 중요한 기여를 하고 있습니다.



### Invariant Shape Representation Learning For Image Classification (https://arxiv.org/abs/2411.12201)
- **What's New**: 이 논문에서는 이미지 분류의 강인성을 더욱 강화하기 위한 새로운 프레임워크인 Invariant Shape Representation Learning (ISRL)을 소개합니다. 기존의 딥 뉴럴 네트워크(DNN)에서는 기하학적 형태(feature)와 목표 변수(target variables) 간의 통계적 상관관계를 직접 활용했지만, 이러한 관계는 불안정하고 편향되기 쉬워 정확한 예측을 방해할 수 있습니다. ISRL은 잠재적인 형태 공간(latent shape spaces)에서 불변 형태를 공동으로 캡처하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: ISRL은 변형된 변환(deformable transformations)으로 매개변수화된 잠재 형태 공간에서 불변 특성을 캡처하기 위해 설계되었습니다. 논문에서는 불변 위험 최소화(invariant risk minimization, IRM)에 기반한 새로운 학습 패러다임을 개발하여 여러 훈련 분포/환경에서 이미지 및 형태 특성의 불변 표현을 학습합니다. 이러한 방식으로, 다양한 환경에서 목표 변수에 대해 불변한 특성을 포함하도록 모델을 개선하여 예측 정확도를 높이는 데 성공합니다.

- **Performance Highlights**: ISRL 방법의 유효성은 시뮬레이션한 2D 이미지, 실제 3D 뇌 MRI 및 심장 MRI 비디오를 포함한 분류 작업을 통해 검증되었습니다. 실험 결과, 제안된 방법이 기존 선진 기법들에 비해 shifted environments에 대해 강인성을 크게 향상시키고 일관되게 높은 분류 정확도를 보여주었습니다. 이 연구는 이미지 분류 작업에서 기하학적 형태 특성을 효과적으로 활용하는 새로운 가능성을 열어줍니다.



### RoSIS: Robust Framework for Text-Promptable Surgical Instrument Segmentation Using Vision-Language Fusion (https://arxiv.org/abs/2411.12199)
Comments:
          10 pages, 6 figures, submitted to IEEE transactions on Medical Imaging

- **What's New**: 이 논문에서는 Robus Surgical Instrument Segmentation (RoSIS)라는 새로운 프레임워크를 제안하며, 텍스트 프롬프트를 바탕으로 수술 도구의 세분화(Segmentation)를 수행하는 방법을 재정의합니다. 기존의 텍스트 프롬프트 방식의 단점을 극복하고, 모든 클래스에 대한 프롬프트를 적용 가능한 간결하고 효과적인 방식을 구현하여 공정한 비교를 가능하게 합니다. R-SIS (Robust text-promptable Surgical Instrument Segmentation) 환경에서 기능하도록 설계되었습니다.

- **Technical Details**: RoSIS는 인코더-디코더 아키텍처를 기반으로 하며, Multi-Modal Fusion Block (MMFB)과 Selective Gate Block (SGB)을 포함하여 비전(vision)과 언어(language) 피처의 균형 잡힌 통합을 이룹니다. 또한, 초기 마스크와 위치 프롬프트를 사용한 두 단계로 구성된 반복적 추론 전략을 도입하여 세분화 마스크를 개선합니다. R-SIS를 통해 제안된 접근 방식은 존재 여부를 판단하는 부정적인 프롬프트와 긍정적인 프롬프트를 사용하여 보다 현실적인 수술 시나리오에서의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과, RoSIS는 EndoVis2017 및 EndoVis2018 데이터셋에서 기존의 비전 기반(methods) 및 프롬프트 기반 방법들(next methods)을 능가하는 성능을 보였습니다. 신뢰할 수 있는 벤치마킹을 위한 실험 과정을 세심하게 재검토하였으며, 기존의 텍스트 프롬프트 기반 방법과 비교하여 뛰어난 정확성과 신뢰성을 입증하였습니다. 이러한 결과는 RoSIS의 강력한 성능을 입증하며, 다양한 수술 조건에서도 일관된 성과를 유지합니다.



### CCIS-Diff: A Generative Model with Stable Diffusion Prior for Controlled Colonoscopy Image Synthesis (https://arxiv.org/abs/2411.12198)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 논문에서는 CCIS-DIFF라는 새로운 Generative 모델을 제안합니다. 이 모델은 colonoscopy 이미지 합성을 위한 더욱 정교한 제어 기능을 제공하여, 실질적인 임상 요구사항에 부합하는 이미지를 생성할 수 있습니다. 특히, 합성된 폴립이 대장 점막과 매끄럽게 통합되도록 하는 blur mask weighting 전략과 임상 특성에 맞춘 텍스트 기반 주의 메커니즘을 도입했습니다.

- **Technical Details**: CCIS-DIFF 모델의 핵심은 다중 모달(colonoscopy images, segmentation masks, clinical text descriptions) 데이터셋을 구축한 것입니다. 이를 통해 사전 훈련된 diffusion 모델을 정밀 조정할 수 있으며, 생성 과정에서 텍스트 정보를 효과적으로 통합하는 텍스트 인지 주의 메커니즘이 구현되어 있습니다. 또한, Gaussian blur 작업을 통해 합성된 폴립과 배경 간의 전환을 부드럽게 만드는 전략이 도입되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 CCIS-DIFF 모델은 공간적 제약과 임상적 일관성에 대한 세밀한 제어가 가능하여 높은 품질과 다양성을 갖춘 colonoscopy 이미지를 생성했습니다. 이러한 이미지는 이후의 세분화(segmentation) 및 진단(diagnostic) 작업에 유용한 지원을 제공하며, 기존의 방법들보다 더 나은 성능을 보이는 것으로 나타났습니다.



### MTFusion: Reconstructing Any 3D Object from Single Image Using Multi-word Textual Inversion (https://arxiv.org/abs/2411.12197)
Comments:
          PRCV 2024

- **What's New**: 이 논문에서는 MTFusion을 제안하며, 이는 단일 시점 이미지를 사용하여 고충실도의 3D 모델을 재구성하는 새로운 접근법입니다. MTFusion은 이미지 데이터와 텍스트 설명을 결합하여 3D 재구성을 수행합니다. 두 가지 주요 단계로 구성되며, 첫째는 다중 단어 텍스트 반전 기법을 통해 이미지의 특징을 포착하는 상세한 텍스트 설명을 추출합니다.

- **Technical Details**: MTFusion의 첫 번째 단계인 다중 단어 텍스트 반전에서는 시각적 속성과 같은 입력 이미지의 특성을 포착하기 위해 초기화된 프롬프트 템플릿을 사용하는 새로운 기술을 도입합니다. 두 번째 단계에서는 FlexiCubes라는 3D 표현을 활용하여 최적의 텍스트 설명과 입력 이미지를 바탕으로 3D 모델을 생성합니다. 또한, SDF(사인 거리 함수) 디코더 네트워크를 사용하여 FlexiCubes의 파라미터를 효율적으로 추출합니다.

- **Performance Highlights**: MTFusion은 광범위한 합성 및 실제 이미지의 단일 이미지 3D 재구성에서 기존 방법들을 초월하는 성능을 보여줍니다. 특히, 생성된 메시는 정밀한 기하학적 디테일과 사실적인 텍스처를 포함하여 참조 이미지와 밀접하게 일치합니다. 또한, 아블레이션 연구를 통해 다중 단어 텍스트 반전과 향상된 FlexiCubes 표현의 기여도를 입증하였습니다.



### A Survey of Medical Vision-and-Language Applications and Their Techniques (https://arxiv.org/abs/2411.12195)
- **What's New**: 의료 비전-언어 모델(Medical Vision-and-Language Models, MVLMs)은 복잡한 의료 데이터를 해석하기 위한 자연어 인터페이스를 제공하는 능력으로 최근 큰 관심을 받고 있습니다. MVLM은 의료 영상과 텍스트 정보를 통합하여 진단 정확성을 높이고 임상 의사결정을 지원하는데 기여합니다. 이러한 모델들은 의료 보고서 생성, 시각 질문 응답, 다중 모달 진단 등 다양한 임상 응용 프로그램에서 중요한 역할을 하고 있습니다.

- **Technical Details**: MVLM은 이미지 인코더와 텍스트 인코더를 통해 시각적 및 텍스트적 특성을 학습합니다. 그 후, 특정 작업에 대한 생성기 또는 분류기를 사용하여 응용 프로그램을 수행합니다. 그러나 이러한 모델을 의료 분야에 적용하는 데에는 데이터 접근 제약, 이질성 문제, 데이터 불균형 등의 여러 도전 과제가 존재합니다. MVLM의 출력은 환자 치료에 직접적인 영향을 미치므로 고도의 해석 가능성과 신뢰성이 요구됩니다.

- **Performance Highlights**: MVLM은 정확한 영상 해석을 기반으로 한 의료 보고서 작성을 자동화하여 의료 종사자의 업무 부담을 줄이고 효율성을 향상시키며, 전문 의료 인력의 부족 문제에도 기여하고 있습니다. 현재 사용되는 방법은 이미지 캡셔닝 기술을 활용하여 이미지 특성을 추출하고 분석하여 보고서를 생성합니다. 이러한 자동화 과정은 장기적으로 임상 연구 및 사례 분석에서도 중요한 역할을 할 것으로 기대됩니다.



### Constant Rate Schedule: Constant-Rate Distributional Change for Efficient Training and Sampling in Diffusion Models (https://arxiv.org/abs/2411.12188)
Comments:
          33 pages, 9 figures

- **What's New**: 이번 논문에서는 확산 과정 전반에 걸쳐 데이터의 확률 분포 변화율을 일정하게 유지하도록 설계된 노이즈 스케줄(Noise Schedule), CRS(Constant Rate Schedule)를 제안합니다. 이 노이즈 스케줄은 데이터 세트와 확산 모델 유형에 맞게 자동으로 조정되며, 이미지 생성 작업에서의 유효성을 평가했습니다. 실험을 통해 CRS가 다양한 데이터 세트와 샘플러를 포함하여 확산 모델의 성능을 전반적으로 향상시킨다는 것을 확인했습니다.

- **Technical Details**: CRS는 확산 데이터의 확률 분포 변화율을 측정하여, 이는 확산 과정의 추적 가능성을 높이는 데 기여합니다. 노이즈 스케줄의 기능적 형태는 데이터 세트에 따라 자동으로 결정되며, 선형(Linear) 또는 코사인(Cosine)과 같은 미리 정의된 스케줄 없이도 동작합니다. 이는 Song & Ermon(2020)의 일반화된 버전으로 볼 수 있으며, 다음 단계의 확률 분포 간의 상수적 중첩을 달성하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과 CRS는 LSUN, ImageNet, FFHQ 데이터 세트에서 무조건적 및 클래스 조건의 이미지 생성 작업에 있어 상당한 성능 향상을 가져왔습니다. 또한, CRS는 확산 모델의 학습 및 샘플링 모두에 적용 가능하여, 특정 데이터 세트 및 모델 유형에 맞춰 최적화된 노이즈 스케줄을 제공함으로써 개선된 샘플 품질 및 속도를 달성합니다.



### Robust 3D Semantic Occupancy Prediction with Calibration-free Spatial Transformation (https://arxiv.org/abs/2411.12177)
Comments:
          13 pages, 11 figures, 18 tables

- **What's New**: 이번 연구에서는 3D 공간에서의 세밀한 occupancy 예측을 위해 새로운 Robust and Efficient 3D semantic Occupancy (REO) 예측 방식을 제안합니다. 이 방식은 기존의 센서 교정을 필요로 하지 않는 공간 변환 기술을 채택하여, 복잡한 환경에서도 안정적으로 작업할 수 있는 가능성을 보여주고 있습니다. REO는 2D 이미지의 특징을 BEV 계획으로 직접 매핑하여 소요되는 вычислительные 비용을 크게 줄였습니다.

- **Technical Details**: 본 연구는 vanilla attention 기반의 교정 없는 공간 변환 모듈을 통해 2D 영상 특징을 3D 공간으로 효과적으로 변환합니다. 이를 위해 사전 훈련된 2D 인코더를 사용하여 이미지 특징을 추출하고, 캘리브레이션이 필요 없는 트랜스포메이션 모듈을 통해 BEV로 압축하여 샘플링된 voxel의 geometry 및 semantic 예측을 효율적으로 생성합니다. 또한, 2D와 3D의 보조 훈련 작업을 도입하여 이미지 백본의 판별 능력을 향상시킵니다.

- **Performance Highlights**: 제안된 REO는 OpenOccupancy 등 3개의 벤치마크에서 기존 방법들보다 큰 성능 향상을 보여 주었습니다. 예를 들어, OpenOccupancy에서 Co-Occ 대비 19.8배의 속도 향상과 1.1%의 geometry IoU 개선을 동시에 기록하였습니다. 이러한 성과는 복잡한 환경에서도 안정적인 3D 세멘틱 오큐펀시 예측을 가능하게 합니다.



### Sketch-guided Cage-based 3D Gaussian Splatting Deformation (https://arxiv.org/abs/2411.12168)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문에서는 사용자들이 단일 시점에서 실루엣 스케치를 그려 3D Gaussian Splatting (GS) 모델의 형상을 직관적으로 수정할 수 있도록 하는 새로운 스케치 기반 3D GS 변형 시스템을 제안합니다. 기존 3D GS 시스템의 편집 가능성을 확대하며, 특히 변형의 세밀한 제어에 대한 문제를 해결하려고 합니다.

- **Technical Details**: 우리의 접근 방식은 cage 기반 변형(cage-based deformations)과 Neural Jacobian Fields의 변형을 결합한 새로운 변형 방법을 도입합니다. 이 시스템은 대규모 2D diffusion priors와 ControlNet을 활용하여 생성된 변형이 의미론적으로 타당하도록 보장합니다.

- **Performance Highlights**: 일련의 실험을 통해 우리의 방법의 효과성을 입증하였으며, 정적인 3D GS 모델을 애니메이션화하는 데 있어 주요 응용 프로그램 중 하나로서 그 능력을 보여주었습니다.



### Self-Supervised Learning in Deep Networks: A Pathway to Robust Few-Shot Classification (https://arxiv.org/abs/2411.12151)
- **What's New**: 이번 연구는 self-supervised learning을 활용하여 few-shot 이미지 분류(task)를 최적화하고 ResNet-101 딥 네트워크 모델의 특징 추출(feature extraction) 및 분류 성능(classification performance)을 향상시키는 데 중점을 두고 있습니다. 연구진은 대량의 레이블이 없는 데이터에서 일반적인 특징 표현을 학습한 후, Mini-ImageNet이라는 few-shot 데이터셋에서 모델을 미세 조정(fine-tuning)하여 정확도 및 일반화 능력을 개선하였습니다.

- **Technical Details**: 모델의 학습 과정에서 self-supervised learning을 통해 사전 학습(pre-training)을 진행하고, 이어서 제한된 데이터에서의 성능을 높이기 위해 few-shot 데이터셋으로 세부 조정을 수행합니다. 이는 ResNet-50, DenseNet 등 전통적인 합성곱 신경망(convolutional neural networks) 모델과 비교할 때, ResNet-101의 구조를 통해 향상된 특징 추출 능력을)을 발휘하도록 합니다.

- **Performance Highlights**: 실험 결과, 우리 방법론은 분류 정확도(ACC)와 F1 점수에서 약 95.12%의 우수한 성능을 달성하며, self-supervised learning이 few-shot 분류에서 효과적임을 입증합니다. 이러한 접근법은 few-shot 이미지 분류 분야에 있어 효율적이고 신뢰할 수 있는 해결책(solution)을 제공합니다.



### Distill the Best, Ignore the Rest: Improving Dataset Distillation with Loss-Value-Based Pruning (https://arxiv.org/abs/2411.12115)
- **What's New**: 최근 데이터셋 증류(dataset distillation) 방식이 새롭게 주목받고 있으나, 기존 방법들은 전체 데이터셋에서 비유용한 샘플을 포함하는 경우가 많았습니다. 본 논문에서는 'Prune First, Distill After'라는 새로운 프레임워크를 도입하여, 증류 전에 손실 기반 샘플링 방법으로 데이터셋을 체계적으로 가지치기(pruning)합니다. 이를 통해 미지의 아키텍처에 대한 일반화 능력을 향상시키는 대표적인 코어 세트를 생성합니다.

- **Technical Details**: 우리의 접근법은 손실 값에 기반한 샘플링 전략을 사용하는데, 이는 사전 훈련된 분류기 모델을 활용하여 데이터 샘플을 '분류 난이도(classification difficulty)'에 따라 순위 매기는 방식입니다. 이 과정에서 단순한 샘플과 복잡한 샘플을 각각 먼저 선택하는 두 가지 샘플링 전략을 비교하였으며, 단순 샘플에 집중할 경우 증류 품질이 크게 향상되는 것을 발견했습니다. 또한, 우리는 StyleGAN-XL 및 수정된 디퓨전 모델을 포함한 최신 데이터셋 증류 기법들을 기반으로 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 원래 데이터셋의 80%를 제거하고도 증류된 데이터셋에서 최대 5.2%의 정확도 향상이 이루어졌습니다. 우리의 접근법은 여러 ImageNet 하위 집합에서 다양한 아키텍처를 대상으로 한 광범위한 평가를 통해 유연성과 강건성을 입증했습니다. 이러한 성과는 데이터셋 증류의 효과성을 높이는 가능성을 제시하며, 더 나은 품질의 데이터셋을 생성하는 데 기여합니다.



### FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting (https://arxiv.org/abs/2411.12089)
- **What's New**: 본 연구에서는 3D 개체의 내부 텍스처 생성을 위한 FruitNinja라는 혁신적인 방법을 소개합니다. 이 방법은 기하학적 및 위상학적 변화 중에 개체의 내부 구조를 사실적으로 표현하는 데 초점을 맞추고 있습니다. 이는 실시간 슬라이싱과 렌더링을 가능하게 하며, 추가 최적화 없이도 내부 텍스처를 생성할 수 있습니다.

- **Technical Details**: FruitNinja는 사전 훈련된 diffusion model을 활용하여 자른 단면 뷰를 점진적으로 보완하고, voxel-grid 기반의 스무딩을 적용해 일관된 텍스처를 생성합니다. OpaqueAtom GS 전략을 통해 3D Gaussian Splatting (3DGS)의 한계를 극복하고, 중간 크기 입자의 밀도를 높이며 미세한 텍스처의 색상 전환에서 발생할 수 있는 불안정을 제거합니다. 이 방법은 3DGS와 표면 뷰를 동시에 훈련시킴으로써 효과적인 내부 텍스처 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 FruitNinja는 기존 방법들과 비교해 비주얼 품질에서 탁월한 성능을 보여주었습니다. 다양한 기하학적 변형 과정에서 불일치 없이 내부 뷰의 렌더링을 실시간으로 수행할 수 있으며, 다양한 공통 객체에서 내부 텍스처의 질이 향상됨을 입증했습니다. 본 연구는 기존의 접근 방법들이 간과했던 내부 텍스처 생성의 현실성을 효과적으로 개선한 사례로 평가받고 있습니다.



### Just Leaf It: Accelerating Diffusion Classifiers with Hierarchical Class Pruning (https://arxiv.org/abs/2411.12073)
- **What's New**: 최근의 연구에 따르면, 전통적인 생성 모델이 단순한 콘텐츠 생성에만 국한되지 않고 분류 작업에서도 여전히 활용 가능함을 보여주고 있습니다. 특히, Hierarchical Diffusion Classifier(HDC)는 고유한 계층적 레이블 구조를 활용하여 높은 계산 비용을 줄이면서 이미지 분류 효율성을 향상시키고 있습니다. 이 접근 방식을 통해 HDC는 최대 60%의 속도 향상을 달성하면서도 분류 정확도를 유지하거나 개선할 수 있음을 증명했습니다.

- **Technical Details**: HDC는 전통적인 확산 모델을 기반으로 하며, 레이블 트리를 계층적으로 탐색하여 불필요한 높은 수준의 범주를 점진적으로 제거합니다. 초기 단계에서는 가장 유망한 synsets를 유지하기 위해 레이블 트리를 레벨별로 트래버스하고, 그 후 남은 후보 리프 노드에서 전통적인 확산 분류를 수행합니다. 이로 인해, 불필요한 클래스를 사전에 제거함으로써 계산 비용을 줄이고 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: HDC는 대규모 이미지 분류 작업인 ImageNet-1K에서 약 60%의 빠른 추론 시간을 달성하며, 비슷한 계산 시간에 기존의 확산 분류 모델보다 더 나은 정확도(65.16% vs. 64.90%)를 기록하였습니다. 이를 통해 HDC는 분류 빠르기와 정확도 사이의 새로운 균형을 제공하고, 실질적인 대규모 분류 작업에서도 효과적으로 활용될 수 있는 가능성을 제시합니다.



### Zoomed In, Diffused Out: Towards Local Degradation-Aware Multi-Diffusion for Extreme Image Super-Resolution (https://arxiv.org/abs/2411.12072)
- **What's New**: 기존의 Text-to-Image (T2I) 확산 모델은 512x512 해상도로 제한되어 있었으나, 본 연구에서는 추가 학습 없이 2K, 4K, 심지어 8K 해상도로 이미지를 생성할 수 있는 새로운 접근 방식을 소개합니다. 이 방법은 MultiDiffusion과 지역 손실 인식 프롬프트 추출이라는 두 가지 핵심 요소를 활용하여 고해상도 이미지를 생성하면서도 전 세계적으로 일관성을 유지합니다. 이러한 혁신은 이미지 초해상도(Super-Resolution, SR) 작업에 T2I 확산 모델을 적용하는 새로운 가능성을 제공합니다.

- **Technical Details**: 이 연구의 방법론인 MultiDiffusion은 이미지를 생성하는 과정을 여러 개의 확산 경로에 분산시켜 높은 해상도에서도 전 세계적인 일관성을 보장합니다. 각 단계에서 잠재 피처 맵은 중첩되는 타일로 나누어져 개별적인 확산 과정을 거치며, 이로 인해 인접한 타일 간의 정보를 공유하여 전반적인 구조와 지역 세부 사항의 일관성을 유지합니다. 이러한 과정은 기존 T2I 확산 모델에 비해 512×512 픽셀의 제한 없이 2K 이상의 해상도를 가능하게 합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존의 모델과 비교했을 때 방향성과 세부 사항을 보다 잘 복원할 수 있는 능력을 갖추고 있습니다. 결과적으로, 초해상도의 새로운 기준을 세우며, T2I 모델이 가진 잠재력을 극대화하여 다양한 해상도의 이미지 생성에 효과적으로 활용될 수 있음을 보여주었습니다. 모델의 성능은 2K, 4K, 8K에서의 해상도 증대에 성공적으로 적용되면서 향후 SR 작업의 가능성을 더욱 확장시킵니다.



### Autoassociative Learning of Structural Representations for Modeling and Classification in Medical Imaging (https://arxiv.org/abs/2411.12070)
Comments:
          16 pages, 9 figures

- **What's New**: 이 연구에서는 인간의 인지능력과 보다 일치하는 neurosymbolic 시스템 개념을 제안합니다. ASR(자동 연관 구조 표현)는 관찰된 이미지를 시각적 원시 요소로 재구성함으로써 고수준의 구조적 설명을 형성하도록 강요하는 아키텍처입니다. 이 방법은 히스토로지(조직학) 이미징에서 비정상 진단 작업에 적용되었습니다.

- **Technical Details**: ASR 아키텍처는 주로 Encoder, Modeler, Renderer의 세 가지 구성 요소로 이루어져 있습니다. Encoder는 연속적인 ConvBlock으로 구성되며, Modeler는 여러 공간적 스케일에서 얻어진 잠재 벡터를 해석 가능한 그래픽 원시 요소의 매개변수로 매핑합니다. Renderer는 최종 출력 이미지의 시각적 재현을 담당하며, 전문화된 훈련을 통해 원시 요소의 재구성과 함께 작동합니다.

- **Performance Highlights**: ASR 모델은 상대적으로 적은 데이터 세트에서도 효율적으로 학습이 가능하며, 기존의 딥러닝 모델에 비해 더 나은 해석력을 제공합니다. 연구 결과, ASR은 의료 이미징 분류에서 뛰어난 정확도를 보여 주었고, 이는 의료 분야에서의 활용 가능성을 높여 줍니다.



### ITACLIP: Boosting Training-Free Semantic Segmentation with Image, Text, and Architectural Enhancements (https://arxiv.org/abs/2411.12044)
- **What's New**: 최근 기초적인 Vision Language Models (VLMs)의 발전은 컴퓨터 비전 작업에서 평가 패러다임의 변화를 가져왔습니다. 특히 CLIP 모델이 등장하면서 Open-Vocabulary Semantic Segmentation (OVSS) 분야에서의 연구가 촉진되었습니다. 본 연구에서는 CLIP의 분할 성능을 향상시키기 위해 새로운 모듈과 수정 사항을 도입한 ITACLIP을 제안합니다. 이 방법은 COCO-Stuff와 COCO-Object 등 여러 분할 벤치마크에서 현 상태의 최첨단 접근 방식을 초월합니다.

- **Technical Details**: 본 연구에서는 두 가지 주요 방법을 사용하여 ITACLIP을 개발하였습니다. 첫째, ViT의 마지막 층에서 아키텍처 변경을 적용하고 중간 층에서 생성된 attention 맵을 결합하였습니다. 둘째, 대규모 언어 모델(LLMs)을 활용하여 각 클래스 이름에 대한 정의 및 동의어를 생성하여 CLIP의 open-vocabulary 능력을 극대화하였습니다. 또한 이미지 인코더에서 추출한 특징을 향상시키기 위해 새로운 Image Engineering 모듈을 도입하였습니다.

- **Performance Highlights**: ITACLIP은 COCO-Stuff, COCO-Object, Pascal Context 및 Pascal VOC와 같은 다양한 벤치마크에서 최첨단 결과를 기록하였습니다. ITACLIP은 기존의 SCLIP과 NACLIP보다 더 정확한 분할 맵을 생성하며, 효율적인 훈련 없는 메소드를 통해 성능을 극대화할 수 있음을 보여주었습니다. 이러한 결과는 이미지 레벨 지식에서 픽셀 레벨 예측으로의 전환을 성공적으로 이루어낸 사례로 평가됩니다.



### In-Situ Melt Pool Characterization via Thermal Imaging for Defect Detection in Directed Energy Deposition Using Vision Transformers (https://arxiv.org/abs/2411.12028)
- **What's New**: 이 연구에서는 Directed Energy Deposition (DED) 과정에서의 결점 탐지를 개선하기 위해 비지도 학습(self-supervised learning) 접근 방식을 채택하고 있습니다. 특히 마스크 오토인코더(Masked Autoencoder, MAE)를 기반으로 한 비전 트랜스포머(Vision Transformer) 기술을 사용하여 용융 풀(melt pool) 데이터의 고급 표현을 생성합니다. 이러한 접근법은 전통적인 라벨링된 데이터에 대한 의존도를 줄이고, 적은 양의 라벨링된 데이터로도 결점을 효과적으로 식별할 수 있는 가능성을 제시합니다.

- **Technical Details**: 자기 감독 학습(self-supervised learning) 프레임워크를 기반으로 하는 MAE 모델은 고해상도의 용융 풀 이미지를 처리하여 결점을 탐지하는 데 필요한 정밀한 특징을 학습합니다. 이 과정에서 이미지가 패치(patch)로 분할되고, 각 패치가 순차적으로 처리되어 지역적 및 전역적 수준에서 향상된 특징 추출이 가능해집니다. 두 가지 분류기(Classifier) 평가를 수행하였으며, 각각 세부 조정된 MAE Encoder의 파라미터를 이용한 Vision Transformer 분류기와 MLP 분류기 헤드를 결합한 방법이 사용되었습니다.

- **Performance Highlights**: 이 연구의 프레임워크는 95.44%에서 99.17%까지의 전체 정확도와 80%를 초과하는 평균 F1 점수를 달성하였습니다. Vision Transformer 분류기가 MAE Encoder 분류기보다 약간 높은 성능을 보였으며, 이러한 결과는 DED 과정에서 결점을 탐지하기 위한 자동 품질 관리를 위한 접근법의 확장성과 비용 효율성을 입증합니다.



### Analyzing and Improving the Skin Tone Consistency and Bias in Implicit 3D Relightable Face Generators (https://arxiv.org/abs/2411.12002)
Comments:
          10 pages, 10 figures, 5 tables, WACV 2025

- **What's New**: 이 논문에서는 3D 얼굴 리라이트 생성 기술에서 피부 톤의 일관성을 높이고 편향을 완화하기 위한 새로운 전략을 제안합니다. 특히, 기존의 조명 추정 방법에서 발생하는 편향된 구형 조화 함수(SH) 계수를 분석하여 문제를 해결하려고 하였습니다. 연구진은 조명 크기의 내재적 편향을 제거하고, 다른 밴드의 계수를 통계적으로 정렬하는 방법을 통해 생성된 이미지의 일관성을 개선합니다.

- **Technical Details**: 연구팀은 구형 조화 함수(SH) 계수를 DC 항에 따라 정규화하여 조명 강도의 내재적 편향을 제거합니다. 또한, 다양한 피부 톤의 조명을 통계적으로 정렬하여 생성자(generator)와 구분자(discriminator) 모두의 훈련에 활용합니다. 이로 인해 생성된 리라이트 이미지의 피부 톤 일관성이 높아지고, 밝은 피부 톤으로 편향된 알베도 생성 문제를 완화했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 다양한 조명 조건에서도 피부 톤의 일관성을 향상시키고, 리라이트된 이미지에서의 피부 톤 편향 문제를 줄이는 데 효과적임을 입증했습니다. 또한, 본 접근 방식은 DiFaReli와 같은 다른 리라이트 기술에서도 피부 톤 일관성을 높이는 데 사용될 수 있습니다.



### Medical Video Generation for Disease Progression Simulation (https://arxiv.org/abs/2411.11943)
Comments:
          Tech Report. The appendix will release soon. arXiv admin note: text overlap with arXiv:2309.11745

- **What's New**: 이 논문에서는 질병 진행 상황을 시뮬레이션할 수 있는 첫 번째 Medical Video Generation (MVG) 프레임워크를 제안하고 있습니다. 이를 통해 질병과 관련된 이미지 및 비디오 특징을 조작할 수 있어 정확하고 개인화된 시뮬레이션을 가능하게 합니다. MVG는 의료 영상의 데이터 부족 문제를 해결하고, 의료 서비스 제공자들이 효율적인 치료 전략을 수립하는 데 도움을 줄 수 있습니다.

- **Technical Details**: MVG 프레임워크는 GPT-4를 사용하여 환자의 임상 보고서를 요약하고, 그에 맞는 텍스트 인퍼런스를 이용해 질병 관련 특징을 점진적으로 제어할 수 있도록 설계되었습니다. 이 과정에서 노이즈 제거 확산 확률 모델의 가역성(invertibility) 및 맥락 부호기(context encoder)의 시각적 언어 정렬 능력을 활용하여 질병 진행 상황을 시뮬레이션합니다. 또한, 이론적으로는 다단계 질병 상태 시뮬레이션 모듈이 주어진 텍스트 조건의 로그 가능성을 극대화하기 위한 경량 감소(gradient descent) 과정으로 이해될 수 있습니다.

- **Performance Highlights**: MVG는 세 가지 의료 영상 도메인에서 기존 모델들에 비해 월등한 성능을 보여주었습니다. 의사들에 의한 사용자 연구에서는 76.2%의 질병 상태 시뮬레이션 결과가 임상적 맥락과 밀접하게 일치한다고 평가되었습니다. 이러한 결과는 MVG가 질병 진행 경과를 예측하는 데 있어 효과적임을 입증하며, 의료 교육 및 데이터 보완에도 큰 기여를 할 것으로 기대됩니다.



### TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction (https://arxiv.org/abs/2411.11941)
- **What's New**: 본 논문에서는 동적 장면 복원 기술의 한계를 극복하기 위해 TimeFormer라는 플러그 앤 플레이 모듈을 제안합니다. TimeFormer는 다중 타임스탬프에서의 모션 패턴을 암묵적으로 모델링할 수 있는 기능을 갖추고 있으며, 동적 장면에서의 더 비효율적인 점들을 해결하기 위해 설계되었습니다. 이 기술은 기존의 변형 가능한 3D Gaussian 방법에 적용할 수 있도록 최적화되어 있어 실시간 렌더링 속도를 유지하면서 성능을 향상시킬 수 있습니다.

- **Technical Details**: TimeFormer는 Cross-Temporal Transformer Encoder를 포함하여, 변형 가능한 3D Gaussian의 시간적 관계를 적응적으로 학습합니다. 또한, 두 개의 스트림 최적화 전략을 통해 TimeFormer에서 학습한 모션 지식을 기본 스트림으로 전달하여 훈련 과정에서 효과적으로 동작합니다. 이 과정에서 TimeFormer는 추론 시에는 제외되며, 원래의 렌더링 속도를 유지할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 다양한 멀티뷰 및 단안 동적 장면에서 TimeFormer의 효과를 검증하는 실험을 수행하였고, 이를 통해 질적 및 양적으로 향상된 결과를 확인했습니다. TimeFormer는 추가적인 정보 없이 RGB 감독만으로도 모션 패턴을 추출하며, 이는 기존 변형 가능한 3D Gaussian 방식에 간편하게 통합될 수 있습니다. 최종적으로, TimeFormer는 렌더링 품질에서 최첨단 성능을 달성하는 데 기여합니다.



### Fair Distillation: Teaching Fairness from Biased Teachers in Medical Imaging (https://arxiv.org/abs/2411.11939)
- **What's New**: 이 논문에서는 Fair Distillation (FairDi)이라는 첫 번째 공정성 방법을 소개합니다. 이 방법은 각 민감한 그룹에 대한 정확성 극대화와 전반적인 정확性 극대화 및 그룹 간 정확성 격차 최소화를 분해함으로써 공정성을 달성합니다. 또한, 특정 그룹을 위해 훈련된 편향된 '교사' 모델들이 새로운 '학생' 모델에게 지식을 전달함으로써 이러한 목표를 동시에 잡고 있습니다. 이를 통해 FairDi는 의료 영상 데이터셋에서 기존 방법들에 비해 눈에 띄는 성과를 보여줍니다.

- **Technical Details**: FairDi는 학생-교사 지식 증류(student-teacher knowledge distillation) 프레임워크를 통해 각 민감 그룹별 정확성과 차별성 감소를 함께 최적화하는 방식으로 설계되었습니다. 이 과정에서 기반 모델을 초기 훈련하여 강력한 지지력을 형성하고, 이후 각 민감 그룹에 맞춘 '교사' 모델을 파인 튜닝하여 그룹별 정확성을 높입니다. 마지막으로, 이 교사 모델들의 지식을 하나의 통합 '학생' 모델에 증류하여 최종 성능을 향상시킵니다.

- **Performance Highlights**: FairDi는 의료 이미징 데이터셋에서 전반적인 정확도 및 그룹별 정확도에서 각각 유의미한 향상을 달성했습니다. 제안된 방법은 기존 공정성 방법들에 비해 높은 전반적 정확도와 낮은 AUC 격차(AUC gap) 상황에서도 뛰어난 성과를 나타냈습니다. FairDi는 분류 및 분할과 같은 다양한 의료 작업에서 유연하여 공정한 모델 성능을 위한 효과적인 솔루션을 제공합니다.



### Calibrated and Efficient Sampling-Free Confidence Estimation for LiDAR Scene Semantic Segmentation (https://arxiv.org/abs/2411.11935)
- **What's New**: 이번 연구는 LiDAR 데이터의 안전 비판적 응용에서 신뢰할 수 있는 예측을 위한 샘플링 없는 방법을 제안합니다. 기존의 샘플링 기반 방법보다 추론 시간을 대폭 단축하면서도 정확한 신뢰 값을 유지할 수 있습니다. 우리의 접근 방식은 Adaptive Calibration Error (ACE) 메트릭을 통해 잘 보정된 신뢰 값을 달성하며, 실제 분류 정확도와 일관되도록 설계되었습니다.

- **Technical Details**: 제안한 방법은 aleatoric uncertainty(예측 불확실성)와 epistemic uncertainty(모델 불확실성)를 효과적으로 캡쳐하여 LiDAR 장면의 의미적 분할을 다룹니다. 이를 위해 각 요소 클래스에 대한 예측 Gaussian 분포를 생성하며, 가장 높은 평균 값을 가지는 클래스의 신뢰 값을 결정하는 과정을 포함합니다. 이 과정에서 깊은 앙상블 방식을 사용하여 여러 모델로부터 신뢰도 평가를 집계하여 더 강력한 신뢰 예측을 가능하게 합니다.

- **Performance Highlights**: 제안된 샘플링 없는 접근 방식은 기존의 신뢰 추정 방법에 비해 추론 시간을 현저히 줄이며, 잘 보정된 신뢰 값을 제공합니다. 성능 평가에서 우리의 방법은 과신이 아닌 과소신뢰를 생성하여, 안전 비판적 의사결정에서 유리한 결과를 보여줍니다. 이로써 LiDAR 장면의 의미적 분할 작업에서 우수한 강인성과 신뢰성을 달성했습니다.



### SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Inpu (https://arxiv.org/abs/2411.11934)
- **What's New**: 본 논문에서는 모노큘러 입력에서 스테레오 비디오를 생성하는 새로운 자기지도(Self-Supervised) 스테레오 비디오 합성 패러다임인 SpatialDreamer를 소개합니다. 이 방법은 데이터 부족 문제를 해결하기 위한 깊이 기반 비디오 생성 모듈인 Depth based Video Generation(DVG)을 포함하며, 이를 통해 기하학적 및 시간적 일관성을 유지하는 비디오를 생성합니다.

- **Technical Details**: SpatialDreamer는 DVG에 의해 생성된 쌍을 사용하여 RefinerNet이라는 자기지도 비디오 합성 프레임워크를 구축합니다. 중요한 점은, 우리는 스테레오 편차 강도(metric of stereo deviation strength)와 시간적 상호작용 학습 모듈(Temporal Interaction Learning, TIL)을 포함하는 일관성 제어 모듈을 설계하여 기하학적 및 시간적 일관성을 보장하는 것입니다. 이러한 디자인은 다이나믹한 장면에서도 적용이 가능하게 합니다.

- **Performance Highlights**: 다양한 벤치마크 방법과의 비교 연구에서, SpatialDreamer는 최신 기술(state-of-the-art)보다 뛰어난 성능을 보였습니다. 특히, AVP 3D 변환기 및 오픈 소스 스테레오 비디오 합성 방법보다도 더 나은 성능을 나타냈습니다. 따라서 실제 응용에 적합한 비디오 합성이 가능함을 보여줍니다.



### AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning (https://arxiv.org/abs/2411.11930)
- **What's New**: 본 연구는 다중 모드 수학적 추론(multimodal mathematical reasoning)의 도전 과제에 접근하며, 멀티모달 대형 언어 모델(MLLMs)에 "느린 사고(slow thinking)" 능력을 통합한 AtomThink 프레임워크를 제안합니다. 기존 방법들이 빠른 사고를 바탕으로 하는 것과는 달리, 본 연구에서는 단계별로 구성된 사고의 긴 사슬(chain of thought, CoT)을 형성하여 MLLMs가 복잡한 추론을 수행할 수 있도록 돕습니다.

- **Technical Details**: AtomThink는 세 가지 주요 모듈로 구성됩니다: 고품질 CoT 주석을 자동 생성하는 CoT 주석 엔진, MLLM과 정책 보상 모델(policy reward model, PRM)을 결합하여 단계별 추론을 최적화하는 원자 단계 미세 조정 전략(atomic step fine-tuning strategy), 그리고 PRM과 함께 사용할 수 있는 네 가지 검색 전략(search strategies)입니다. 또한, AtomMATH라는 대규모 다중 모드 데이터세트를 제안하고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면 AtomThink는 기본 MLLM의 성능을 크게 향상시켜 MathVista에서 약 50%의 상대적 정확성 향상과 MathVerse에서 120%의 개선을 달성했습니다. AtomThink를 기반 모델로 사용할 경우 LLaVA-Llama3-8B의 정확도를 각각 9.6% 및 18.8% 향상시켰으며, MathVerse에서는 최고 정확도 40.5%를 기록하여 최첨단 GPT-4V를 초과했습니다.



### FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training (https://arxiv.org/abs/2411.11927)
- **What's New**: 이 논문에서는 FLAME(Frozen Large lAnguage Models Enable data-efficient language-image pre-training)라는 혁신적인 프레임워크를 제안합니다. FLAME은 기존의 CLIP 모델들이 직면하고 있는 데이터 부족 문제와 텍스트 인코더의 제한된 처리 능력을 극복하기 위해 동결된 대형 언어 모델(Large Language Models) 사용에 중점을 두고 있습니다. 이 방법을 통해 길고 다국어로 구성된 텍스트를 자연스럽게 처리할 수 있으며, 인상적인 다국어 일반화 능력을 발휘합니다.

- **Technical Details**: FLAME은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 다양한 의미적 표현을 추출하기 위한 다면적 프롬프트 증류 기법(multi-faceted prompt distillation)을 통해 이미지와 텍스트 쌍으로부터 풍부한 의미 정보를 확보합니다. 둘째, 효율적인 컴퓨테이션을 보장하기 위해 페이셋-디커플드(facet-decoupled) 주의 메커니즘과 오프라인 임베딩 전략을 결합하여 작성됩니다. 이러한 구조는 FLAME이 제한된 교육 데이터로도 효과적인 성능을 발휘할 수 있도록 합니다.

- **Performance Highlights**: FLAME은 CC3M 데이터셋에서 훈련 시, ImageNet Top-1 정확도에서 기존의 최고 모델보다 4.9% 개선되었습니다. 다국어 평가에서는 Crossmodal-3600 데이터셋에서 36개 언어에 대해 평균 이미지-텍스트 Recal@1에서 WIT-400M 훈련 CLIP보다 44.4%를 초과하였고, Urban-1k 데이터셋에서는 길이 맥락 검색에서 텍스트-이미지 Recall@1이 87.9%로 WIT-400M 훈련 CLIP보다 34.6% 향상되었습니다. 이러한 결과는 FLAME이 이전 방법의 극히 일부 훈련 데이터로 이러한 성과를 달성했다는 점에서 특히 인상적입니다.



### KAN-Mamba FusionNet: Redefining Medical Image Segmentation with Non-Linear Modeling (https://arxiv.org/abs/2411.11926)
Comments:
          9 pages, 5 figures, 4 tables

- **What's New**: 이번 연구에서는 Kolmogorov-Arnold Networks (KAN)와 Mamba layer를 결합한 KAN-Mamba FusionNet 프레임워크를 제안합니다. 이 프레임워크는 의료 이미지 분할에서의 성능을 향상시키기 위해 attention-driven 메커니즘을 통합하여 interpretability를 유지하면서도 기존의 Mamba에 비해 보다 효과적인 질병 위치 파악 및 진단 정확성을 제공합니다.

- **Technical Details**: KAN-Mamba FusionNet 아키텍처는 기존 Mamba 구조에서 Convolution 및 Batch Normalization 레이어를 KAN 블록으로 대체하는 방식으로 설계되었습니다. 이 모델은 오토 회귀 배포와 병렬 훈련을 통해 복잡한 의료 이미징 데이터를 처리하며, Bag-of-Activation (BoA) 기능을 활용하여 다양한 활성화 함수의 장점을 통합합니다.

- **Performance Highlights**: 세 가지 의료 이미지 분할 데이터셋인 BUSI, Kvasir-Seg, GlaS에 대한 평가 결과, KAN-Mamba FusionNet은 최신 기술보다 일관된 IoU와 F1 점수를 기록했습니다. 모델의 다양한 구성 요소의 영향을 분석한 ablation 연구를 통해 제안된 방법론의 강점과 효과성을 입증하였습니다.



### Continuous Speculative Decoding for Autoregressive Image Generation (https://arxiv.org/abs/2411.11925)
- **What's New**: 이번 논문에서는 연속값을 사용하는 Autoregressive (AR) 이미지 생성 모델을 위해 speculativ decoding 알고리즘을 처음으로 확장 및 적응시켰습니다. 이를 통해 기존의 discrete-token 모델에 비해 생성 속도가 두 배 이상 빨라졌습니다. 초기 단계에서의 낮은 수용률 문제를 완화하기 위해 denoising trajectory alignment 및 token pre-filling 방법을 도입했습니다.

- **Technical Details**: 연속형 speculativ decoding을 통해 draft 분포(p(x))와 target 분포(q(x))의 확률 밀도 함수(PDF)를 계산하는 방법을 세분화했습니다. 또한 acceptance-rejection sampling 방법을 사용하여 분석적 형식이 없는 수정된 분포에서 샘플링을 수행했습니다. 이러한 접근 방식으로 기존 모델에 추가 학습 없이 통합할 수 있는 방법론을 제시했습니다.

- **Performance Highlights**: 본 알고리즘은 기존의 모델에 비해 2.33배의 속도 향상을 달성하며, 원본 생성 품질을 크게 유지합니다. Fréchet Inception Distance (FID) 및 Inception Score (IS)와 같은 성능 지표를 통해 ImageNet 256×256에서의 성능을 정량적으로 평가했습니다. 광범위한 실험 결과는 제안한 방법의 효과iveness를 더욱 뒷받침합니다.



### SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory (https://arxiv.org/abs/2411.11922)
- **What's New**: 이번 연구에서는 SAM 2의 성능을 개선하기 위해 SAMURAI라는 새로운 비주얼 오브젝트 트래킹(Visual Object Tracking) 모델을 제안합니다. SAMURAI는 모션 정보를 통합하여 빠르게 움직이는 객체나 가려지는 객체들을 효과적으로 추적합니다. 또한, 동적 환경에서도 안정적인 성능을 발휘하며, 추가적인 재훈련이나 미세 조정 없이 이러한 다가오는 객체의 동작을 예측할 수 있습니다.

- **Technical Details**: SAMURAI는 두 가지 주요 개선 사항을 포함하고 있습니다: 첫째, 모션 모델링 시스템을 통해 마스크 선택을 세분화하여 복잡한 상황에서도 보다 정확한 객체 위치를 예측합니다. 둘째, 혼합 점수 시스템을 활용한 메모리 선택 메커니즘을 최적화하여, 과거의 중요한 정보를 더욱 효과적으로 보존하고 모델의 전체적인 추적 신뢰성을 향상시킵니다. 이 방식을 통해 SAMURAI는 복잡한 장면에서 높은 일관성을 유지할 수 있도록 합니다.

- **Performance Highlights**: SAMURAI는 다양한 벤치마크 데이터셋에서 주목할 만한 제로샷(zero-shot) 성능을 보여줍니다. LaSOT 데이터셋에서 7.1% AUC 증가, GOT-10k에서 3.5% AO 증가를 달성하며, 기존의 트래커들보다 성공률과 정확성이 크게 개선되었습니다. 더불어, 완전 감독 방식의 방법들에 비해 경쟁력 있는 결과를 제공하여 복잡한 트래킹 시나리오에서도 그 강점을 잘 보여주고 있습니다.



### DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes (https://arxiv.org/abs/2411.11921)
- **What's New**: 본 논문에서는 DeSiRe-GS라는 자기 지도 방식(self-supervised)을 이용한 Gaussian splatting 표현을 제시합니다. 이 방법은 복잡한 주행 시나리오에서 효과적인 정적-동적 분해(static-dynamic decomposition)와 높은 충실도의 표면 재구성을 가능하게 합니다. 두 단계 최적화 파이프라인을 통해 동적 거리 가우시안을 처리함으로써, 추출된 2D 모션 마스크를 Gaussian 공간으로 변환합니다.

- **Technical Details**: 첫 번째 단계에서는 동적 환경에서 3D Gaussian Splatting이 정적 영역만 재구성할 수 있다는 관찰을 바탕으로 2D 모션 마스크를 추출합니다. 두 번째 단계에서는 효율적인 동적 가우시안의 수식을 활용하여 이러한 2D 모션 선행 정보(priors)를 미분 가능한 방식으로 Gaussian 공간에 매핑합니다. 또한, 기하학적 정규화(geometric regularizations)를 도입하여 데이터 희소성(data sparsity)으로 인한 과적합(over-fitting) 문제를 해결합니다.

- **Performance Highlights**: DeSiRe-GS는 시간과 관점에 따른 일관성을 보장하기 위한 시간적 교차 시뷰 일관성(temporal cross-view consistency)을 도입해 높은 품질의 표면 재구성을 가능케 합니다. 실험 결과, 이 방법은 이전의 자기 지도 방식(self-supervised arts)보다 우수한 성능을 보여주며, 외부 3D 바운딩 박스(annotation) 의존 방식에 버금가는 정확도를 달성합니다. 논문의 코드는 해당 링크에서 확인할 수 있습니다.



### VL-Uncertainty: Detecting Hallucination in Large Vision-Language Model via Uncertainty Estimation (https://arxiv.org/abs/2411.11919)
- **What's New**: 본 연구는 LVLM(대형 비전-언어 모델)의 환각(hallucination) 탐지를 위한 첫 번째 불확실성 기반 프레임워크, VL-Uncertainty를 소개합니다. 기존 방법들이 사실이나 유사 주석을 필요로 하는 것과는 달리, VL-Uncertainty는 불확실성을 내재적(metric)으로 측정하는 방법을 사용합니다. 이 프레임워크는 시맨틱적으로 동등하지만 교란된 프롬프트에 대한 예측 분산을 분석하여 LVLM의 불확실성을 평가합니다.

- **Technical Details**: VL-Uncertainty는 높은 신뢰도를 보이는 LVLM의 경우 시맨틱적으로 동등한 쿼리에 일관된 응답을 제공한다는 원칙에 기반합니다. 불확실성이 높은 경우에는 응답이 더 무작위적으로 변화합니다. 우리는 LVLM 응답을 시맨틱 컨텐츠 기준으로 클러스터링하고, 클러스터 분포의 엔트로피를 계산하여 불확실성을 측정하고 환각을 탐지합니다. 시각적 프롬프트에는 블러링을 사용하고, 텍스트 프롬프트는 LLM(Large Language Model)을 활용하여 의미를 변화시키지 않고 교란을 주는 방식으로 접근합니다.

- **Performance Highlights**: 우리의 실험 결과는 10개의 LVLM에 대해 4개의 벤치마크에서 VL-Uncertainty가 강력한 기준 방법들을 명확한 차이로 초과 성능을 보임을 보여줍니다. VL-Uncertainty는 LVLM의 불확실성을 효과적으로 포착하여 정확한 환각 탐지를 가능하게 합니다. 이러한 발견은 LVLM의 안전성과 신뢰성을 향상시키는 데 기여하며, 새로운 분야로의 확장 가능성을 갖습니다.



### FCC: Fully Connected Correlation for Few-Shot Segmentation (https://arxiv.org/abs/2411.11917)
- **What's New**: 이번 연구는 기존의 few-shot segmentation (FSS) 기법의 한계를 극복하기 위해 새로운 접근법인 Fully Connected Correlation (FCC)을 제안합니다. FCC는 Vision Transformer(ViT)의 교차 레이어(cross-layer) 정보를 활용하여, 기존의 동일 레이어(same-layer) 비교에서 놓쳤던 세부 정보들을 포착합니다. 이는 비슷한 구조를 가진 이미지들의 특성을 더 잘 이해하고, 이를 통해 세분화 작업을 보다 효과적으로 수행하는 것을 가능하게 합니다.

- **Technical Details**: FCC는 지원 이미지와 쿼리 이미지 간의 픽셀 수준의 상관 관계를 통합하여, 목표 개체에 대한 세부 패턴과 연관성을 포착합니다. 또한, Dual-Conditioned Fully Connected Correlation (DCFC)을 통해 지원 마스크에 없는 숨겨진 정보도 탐지할 수 있는 능력을 강화합니다. 이를 통해 보다 포괄적이고 강력한 특성 표현을 구축할 수 있습니다.

- **Performance Highlights**: FCC는 PASCAL-5i, COCO-20i 및 도메인 전이 테스트에서 기존의 최첨단 기법들을 능가하는 성능을 입증하였습니다. 연구 결과를 통해 FCC가 모델의 성능을 크게 개선하고, 다양한 복잡한 시나리오에서 효과적인 세분화가 가능하다는 것을 확인하였습니다.



### F$^3$OCUS -- Federated Finetuning of Vision-Language Foundation Models with Optimal Client Layer Updating Strategy via Multi-objective Meta-Heuristics (https://arxiv.org/abs/2411.11912)
- **What's New**: 이번 연구에서는 리소스가 제한된 클라이언트 장치에서 Vision-Language Models(VLMs)를 효과적으로 훈련하기 위한 Parameter-Efficient Fine-Tuning(PEFT) 전략을 제안합니다. 특히, 클라이언트별로 가장 중요한 VLM 레이어를 선택하는 	extit{client-specific layer importance score}와 클라이언트 간 다양한 레이어 선택을 장려하는 	extit{inter-client layer diversity score}의 두 가지 요소의 영향을 밝혀냅니다. 이를 통해 개인화된 모델 훈련을 지원하는 새로운 프레임워크인 F3OCUS를 도입하였습니다.

- **Technical Details**: F3OCUS는 레이어 선택을 개선하기 위해 클라이언트의 자원 제약을 고려하면서 지역적 및 전역적 FL 특성을 모두 반영하는 두 단계의 전략을 적용합니다. 첫 번째 단계에서 클라이언트 레벨 전략을 통해 LNTK(Neural Tangent Kernel)의 주 고유값을 기반으로 레이어 중요도를 정의하고, 두 번째 단계에서 서버 레벨 전략을 통해 클라이언트별 중요도를 극대화하고 레이어 선택의 분산을 최소화하여 균일한 레이어 참여를 촉진합니다. 연구에서 제안된 방법은 58개의 의료 이미지 데이터셋을 포함한 6가지 Vision-Language FL 작업 설정에서 10,000건 이상의 클라이언트 실험으로 그 효과를 입증했습니다.

- **Performance Highlights**: F3OCUS의 실험 결과, 다양한 FL 환경에서 여러 VLM에 대한 선택적 레이어 튜닝이 효과적으로 이루어짐을 확인했습니다. 본 연구는 인간의 판단력 저하 및 효율성을 가진 클라이언트들에게 적합한 다채롭고 동적인 레이어 선택 솔루션을 제공함으로써 빠른 수렴을 촉진합니다. 데이터, 모달리티, 작업 및 장치 이질성을 고려한 더 많은 제약을 반영하여 클라이언트 설정을 평가한 결과, 이전의 연구보다 향상된 성능을 보여주었습니다.



### SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization (https://arxiv.org/abs/2411.11909)
- **What's New**: 본 논문에서는 최근 발전한 Large Multimodal Models (LMMs)의 한계를 극복하기 위해 Symbol Demonstration Direct Preference Optimization (SymDPO)라는 새로운 방법론을 제안합니다. 기존 LMM은 In-Context Learning (ICL)에서 시각적 정보를 효과적으로 활용하지 못하는 문제점을 가지고 있는데, SymDPO는 이를 해결하기 위해 텍스트 답변을 무작위 기호로 대체하여 모델이 시각적 정보와 기호 간의 관계를 이해하도록 유도합니다.

- **Technical Details**: SymDPO는 기존의 DPO(Direct Preference Optimization) 방법에서 발전된 방식으로, 시각적 요소와 기호 간의 매핑을 강제합니다. 이 방법은 모델이 답변을 도출하는 데 있어 이미지의 시각적 내용을 철저히 해석할 수 있도록 하여, 시각적 정보가 정확한 답변 생성을 위해 필수적임을 보장합니다. 또한, SymDPO는 LMM들이 상징적 텍스트를 정답으로 활용하도록 강제하여 모델의 멀티모달 이해도를 강화합니다.

- **Performance Highlights**: 다양한 LMM 아키텍처에 대한 실험을 통해 SymDPO의 효율성을 입증하였습니다. 이 방법을 통해 LMM들은 멀티모달 사례에서의 시각적 맥락을 더 효과적으로 이해하고 활용하는 것으로 확인되었습니다. 결과적으로 SymDPO는 LMM의 성능을 현저히 향상시키며, 시각적 맥락 간과 같은 문제를 해결하는 데 기여할 수 있습니다.



### $\text{S}^{3}$Mamba: Arbitrary-Scale Super-Resolution via Scaleable State Space Mod (https://arxiv.org/abs/2411.11906)
- **What's New**: 이번 연구에서는 Arbitrary Scale Super-Resolution (ASSR) 문제를 해결하기 위해 새로운 방법인 S3Mamba를 제안합니다. S3Mamba는 State Space Model (SSM) 개념을 도입하여 연속적인 표현 공간을 확립하고, 스케일에 민감한 self-attention 메커니즘을 통해 다양한 스케일에서 전역 중요한 특징을 포착하는 기능을 향상시킵니다. 이러한 혁신은 기존 방법들이 가지던 고정된 스케일의 한계를 극복하고, 단일 모델로 모든 스케일의 초해상도 이미지를 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 S3Mamba 방법은 Scalable State Space Model (SSSM)을 활용하여 상태 전이 행렬과 샘플링 행렬을 조정하고, 이를 통해 선형 계산 복잡도로 스케일 가능한 연속 표현 모델링을 달성합니다. 또한, 스케일에 대한 인식을 개선하기 위해 새로운 self-attention 기법을 도입하여 네트워크의 전역 특징 인식 능력을 향상시킵니다. 이러한 기법들은 실제 환경에서도 적용 가능한 고효율의 ASSR 네트워크를 구축 가능하게 합니다.

- **Performance Highlights**: S3Mamba의 성능은 DIV2K와 COZ 벤치마크에서 실험을 통해 확인되었으며, 모든 초해상도 스케일에서 최첨단 성능을 기록하였습니다. 제안된 방법은 기존의 다른 ASSR 방식에 비해 일반화 능력이 뛰어나고, 실제 환경에서도 우수한 결과를 보여줍니다. 이는 S3Mamba가 다양한 초해상도 작업에 효과적으로 적용될 수 있음을 시사합니다.



### GeoGround: A Unified Large Vision-Language Model. for Remote Sensing Visual Grounding (https://arxiv.org/abs/2411.11904)
Comments:
          25 pages, 19 figures

- **What's New**: 이 논문에서 제안한 GeoGround는 HBB, OBB, 그리고 분할 마스크를 지원하는 새로운 비전-언어모델(VLM) 프레임워크입니다. 이를 통해 다양한 RS (Remote Sensing) 시각 기반 작업을 통합하고, 사용자의 요구에 따라 유연하게 출력을 선택할 수 있도록 설계되었습니다. 또한, refGeo라는 대규모 RS 비주얼 기준 따르기 데이터 세트를 발표하여, 161,000개의 이미지-텍스트 쌍을 포함하고 있습니다.

- **Technical Details**: GeoGround는 CLIP-ViT 기반의 시각 인코더, 2층 MLP 커넥터, 그리고 LLM (Vicuna 1.5)로 구성된 간결한 아키텍처를 채택하고 있습니다. 이 모델은 각기 다른 시각 기반 작업을 위한 신호를 텍스트 문자열로 변환하여 통합 데이터 파이프라인을 통해 훈련합니다. 텍스트-HBB와 텍스트-OBB의 해상도를 설정하여 작은 물체에 대한 보다 정밀한 로컬라이제이션을 가능하게 합니다.

- **Performance Highlights**: GeoGround는 4가지 RS 시각 기반 작업에서 탁월한 성능을 보여주었으며, 여러 벤치마크에서 전문화된 방법들과 동등하거나 그 이상의 성능을 달성했습니다. 특히, GeoGround는 다른 데이터를 공유하고 아키텍처를 합치는 방식으로 HBB와 OBB 기반 작업을 통합하여 효율적인 시각 기반 작업을 수행할 수 있도록 했습니다.



### DiHuR: Diffusion-Guided Generalizable Human Reconstruction (https://arxiv.org/abs/2411.11903)
Comments:
          Accepted to WACV 2025

- **What's New**: 이 논문에서는 DiHuR라는 새로운 Diffusion-guided 모델을 소개하여, 희소하고 최소한으로 겹치는 이미지로부터 일반화 가능한 3D 인간 재구성 및 뷰 합성을 수행합니다. 기존의 일반화된 인간 radiance fields는 새로운 뷰 합성에서 뛰어난 성능을 보이지만, 3D 재구성에는 한계를 가지고 있습니다. 이 모델은 SMPL 정점과 연관된 학습 가능한 토큰을 사용하여 희소 뷰 특징을 집계하고 SDF(prediction) 예측을 안내합니다. 이 방식은 훈련 데이터셋의 다양한 정체성을 아우르는 일반화된 선행 정보를 학습할 수 있도록 합니다.

- **Technical Details**: DiHuR는 SMPL 모델을 활용해 기하학적으로 일관된 특징을 생성하고, 2D diffusion 모델을 재구성 품질 향상을 위한 기하학적 안내로 통합합니다. 이 모델은 SMPL 정점에 부착된 학습 가능한 토큰을 통해 특징을 융합하며, 모든 토큰 간의 정보 교환을 통해 더욱 세밀한 조정이 이루어집니다. 또한, 노멀 맵을 여러 목표 뷰로부터 렌더링하여 미세 조정을 위해 사용하는 방법을 제안합니다. 이러한 방법을 통해 고품질의 3D 재구성을 가능하게 합니다.

- **Performance Highlights**: DiHuR는 THuman, ZJU-MoCap, HuMMan 데이터셋을 통해 기존 방법들과 비교했을 때 뛰어난 성능을 입증했습니다. 특히 희소 카메라 세팅에서 3D 인간 재구성과 새로운 뷰 합성의 최신 성능을 보여주며, 데이터셋 내부 및 외부 일반화 설정에서 모두 우수한 결과를 얻었습니다. 이 연구는 제한된 겹침을 가진 이미지에서 어떻게 효과적인 특징 집합 및 SDF 예측을 할 수 있는지를 잘 보여줍니다.



### Heuristic-Free Multi-Teacher Learning (https://arxiv.org/abs/2411.12724)
- **What's New**: 최근 연구에서 Teacher2Task라는 새로운 멀티-교사 학습 프레임워크가 소개되었습니다. 이 프레임워크는 기존의 수작업 집계 휴리스틱을 없애고, 각 교사만의 입력 토큰을 도입함으로써 데이터의 혼란을 줄여줍니다. Teacher2Task는 각 교사의 스타일을 반영하는 N개의 보조 과제와 진짜 레이블에 중점을 둔 1개의 주요 과제로 구성된 총 N+1개의 과제로 훈련 프로세스를 재구성합니다.

- **Technical Details**: 기존의 멀티-교사 학습 방법들은 주로 예측 결과를 단순히 집계하여 최종 레이블로 사용하는 접근법을 취하지만, Teacher2Task는 이를 확장하여 각 교사의 신뢰도 점수를 예측하는 보조 작업을 생성합니다. 각 새로운 교사는 단순히 새로운 보조 작업을 도입하는 방식으로 시스템에 통합됩니다. 또한, 교사 식별자를 입력에 포함시킴으로써, 노이즈 데이터를 감소시키고 교사 간의 혼란된 주석을 해결할 수 있는 강점을 가지고 있습니다.

- **Performance Highlights**: 실험을 통해 Teacher2Task는 다양한 아키텍처와 작업에 걸쳐 성능을 개선하고 강건성을 보여주었습니다. 특히, 이 방법은 각 교사의 기여도를 효율적으로 활용함으로써, 학습 데이터의 레이블 효율성을 높이고, 교사의 신뢰도 점수를 데이터로 활용해 전반적인 성능을 향상시키는 데 성공했습니다. 결과적으로, Teacher2Task는 전통적인 집계 방식에 비해 더 다양한 데이터 소스를 효과적으로 조합할 수 있는 방법을 제시하고 있습니다.



### Barttender: An approachable & interpretable way to compare medical imaging and non-imaging data (https://arxiv.org/abs/2411.12707)
Comments:
          Accepted to the Proceedings Track at Machine Learning for Health (ML4H 2024) conference, held on December 15-16, 2024 in Vancouver, Canada

- **What's New**: Barttender는 의료 영상(data)과 비영상(이, non-imaging) 데이터의 유용성을 직접 비교할 수 있는 해석 가능한 프레임워크를 소개합니다. 이 방법론은 전자 건강 기록에서 추출한 스칼라(scalar) 데이터를 그레이스케일 바로 변환하여 서로 다른 데이터 모달리티를 효과적으로 비교할 수 있게 지원합니다. Barttender는 심층 학습(deep learning) 모델을 통해 이러한 비교를 가능하게 하면서도 훨씬 더 향상된 해석 가능성을 제공합니다.

- **Technical Details**: Barttender는 의료 영상에 비영상 데이터의 특성을 그레이스케일 바로 변환하여 추가하는 방법으로, 'Image Barttender' 데이터셋을 생성합니다. 논문의 알고리즘은 각 변수를 [0,1] 범위로 정규화(normalize)하고, 이 값을 그레이스케일로 매핑(mapping)하여 시각적으로 구분할 수 있도록 합니다. 또한, 결측치(missing values)는 빨간 바로 표시하여 깊은 학습 모델에서 결측 데이터의 패턴을 감지할 수 있습니다.

- **Performance Highlights**: CheXpert 및 MIMIC 데이터셋을 이용한 실험 결과, Barttender는 전통적인 방법에 비해 동등한 성능을 나타내면서도 비디오 및 비영상 데이터를 함께 사용할 때 더욱 이해할 수 있는 해석 가능성을 높였습니다. 또한 Barttender는 특정 비영상 특성이 병리 예측에 미치는 영향의 중요성을 강조함으로써, 이미지 모달리티가 독립적으로 예측을 구동하는지를 설명 가능하게 도와줍니다.



### AI Guided Early Screening of Cervical Cancer (https://arxiv.org/abs/2411.12681)
- **What's New**: 이 연구는 의료 이미징 데이터셋의 전처리 및 개선을 통해 신뢰할 수 있는 이상 탐지(machine learning models for anomaly detection) 모델 생성을 지원하는 데 중점을 두고 있습니다. 데이터셋은 정상(normal)과 비정상(abnormal) 두 가지 분류로 나뉘며, 추가적인 노이즈 변동도 포함되어 있습니다. 중앙 자르기를 통해 사진의 질을 개선하기 위한 불필요한 아티팩트(artifacts)를 제거하고, 추가 전처리 과정으로 밝기(brightness)와 대비(contrast)를 조정했습니다.

- **Technical Details**: 이미지 데이터셋은 여러 하위 세트를 두 가지 기본 범주인 정상(normal)과 병리(pathological)로 체계적으로 통합하여 분류 작업을 용이하게 했습니다. 고급 이미지 전처리 기술로는 대비 향상(contrast enhancement)과 실시간 증강(real-time augmentation) 기술이 포함되어 있으며, 이를 통해 회전(rotations), 확대(zooms), 밝기 수정(brightness modifications) 등이 가능합니다. 모델 평가를 위해 데이터는 훈련(training) 및 테스트(testing) 서브셋으로 분할되었습니다.

- **Performance Highlights**: 이 프로젝트는 의료 이상 탐지를 위한 정확하고 효과적인 기계 학습 모델 생성을 목표로 고품질 입력 데이터(input data)를 보장합니다. 프로젝트 파이프라인의 유연하고 확장 가능한 설계 덕분에 대규모 임상 의사결정 지원 시스템(clinical decision-support systems)과 쉽게 통합할 수 있는 장점을 갖추고 있습니다.



### Machine Learning Approaches on Crop Pattern Recognition a Comparative Analysis (https://arxiv.org/abs/2411.12667)
Comments:
          Published in ICNTET2018: International Conference on New Trends in Engineering & Technology Tirupathi Highway, Tiruvallur Dist Chennai, India, September 7-8, 2018

- **What's New**: 이 논문에서는 농업 활동 모니터링의 중요성을 강조하며, 특히 원격 감지(remote sensing)의 역할을 다루고 있습니다. 기존의 분류 방법(SVM 및 결정 트리)에 비해, 딥 뉴럴 네트워크(Deep Neural Network, DNN)를 이용한 분류 방법을 제안하고 있습니다.

- **Technical Details**: 논문에서는 시계열 원격 감지 데이터(time series remote sensing data)를 활용하여 재배 패턴(cropping pattern)을 생성합니다. 그리고 나이브 베이즈(Naive Bayes) 및 랜덤 포레스트(Random Forest)와 같은 두 가지 다른 머신 러닝(machine learning) 접근법과의 비교 분석을 통해 DNN의 성능을 강조하고 있습니다.

- **Performance Highlights**: DNN 기반의 분류 방법이 농작물 패턴 인식(crop pattern recognition)에서의 성능을 개선하는 데 기여할 것으로 기대됩니다. 이 연구는 대규모 지속적인 농업 모니터링을 위한 새로운 가능성을 열 것입니다.



### Instant Policy: In-Context Imitation Learning via Graph Diffusion (https://arxiv.org/abs/2411.12633)
Comments:
          Code and videos are available on our project webpage at this https URL

- **What's New**: 본 논문에서 우리는 In-Context Imitation Learning (ICIL)을 활용한 새로운 방법론인 Instant Policy를 소개합니다. Instant Policy는 단 1~2개의 데모로 새로운 작업을 신속하게 학습할 수 있도록 하며, 기존의 Behavioral Cloning (BC) 방법에 비해 시간 효율적입니다. 이 접근법은 그래프 생성 문제로 모델링하여 데모와 관찰을 구조적으로 해석함으로써 로봇 행동 예측의 효율성을 높입니다.

- **Technical Details**: Instant Policy는 그래프 기반 표현을 활용하여 데모, 현재 포인트 클라우드 관찰, 로봇의 행동을 통합한 구조를 형성합니다. ICIL을 확산 기반 그래프 생성 문제로 공식화하여, 복잡한 데이터의 구조적 학습을 가능하게 합니다. 또한, 절차적으로 생성된 의사 데모(pseudo-demonstration)를 통해 무한한 학습 데이터를 생성할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험에서 Instant Policy는 다양한 일상 작업을 신속하게 배울 수 있는 능력을 보여 주었습니다. 기존의 기준 모델들보다 높은 작업 성공률을 달성하며, 테스트 시 제공되지 않은 객체의 기하학에 대한 일반화 능력 또한 관찰되었습니다. Instant Policy는 또한 인간 손의 데모에서 로봇 정책으로의 크로스 엠바디먼트 전이와 언어 정의 작업에 대한 제로샷 전이를 가능하게 합니다.



### Stochastic BIQA: Median Randomized Smoothing for Certified Blind Image Quality Assessmen (https://arxiv.org/abs/2411.12575)
- **What's New**: 본 연구에서는 학습된 신경망 기반 이미지 품질 평가(No-Reference Image-Quality Assessment, NR-IQA) 메트릭의 취약점을 보완하기 위해, Denoised Median Smoothing for IQA(DMS-IQA)를 제안합니다. 이 기법은 평범한 NR 메트릭에 쉽게 적용할 수 있으며, 추가적인 모델 아키텍처 제약 없이 기존의 모델을 재훈련하지 않고도 사용할 수 있습니다. 이를 통해 실제 이미지 품질을 저하시키지 않고도 공격에 견딜 수 있는 이론적으로 강건한 품질 평가 메트릭을 구현하고자 합니다.

- **Technical Details**: DMS-IQA는 Median Smoothing(MS)과 추가적인 Convolution Denoiser를 통합하여 이미지 품질 점수를 개선하고 있습니다. 훈련 시 배치 내 이미지들 간의 상대적 거리를 고려하여 작동하는 소규모 U-Net Denoiser가 사용됩니다. 이 시스템은 l_{2} 노름이 특정 임계값 이하일 때 IQA 점수가 특정 범위(S^{l}, S^{u}) 내에서 유지된다는 보장을 제공합니다.

- **Performance Highlights**: 세 개의 데이터셋을 활용한 실험 결과, 제안된 DMS-IQA가 이전의 두 가지 방법보다 우수한 SROCC와 PLCC 점수를 기록하였으며, 인증된 보장도 유지하면서 효과적인 품질 평가가 가능함을 보여주었습니다. 또한, 이미지 품질 평가 메트릭을 기반으로 한 손실 함수로서도 사용할 수 있는 가능성을 입증하였습니다.



### S3TU-Net: Structured Convolution and Superpixel Transformer for Lung Nodule Segmentation (https://arxiv.org/abs/2411.12547)
- **What's New**: 이 연구에서는 폐선암 결절의 CT 이미지에서 정확한 분할을 위한 새로운 모델인 S3TU-Net을 제안합니다. S3TU-Net은 다차원 공간 커넥터와 슈퍼픽셀 기반의 비주얼 트랜스포머를 통합하여 우수한 분할 성능을 달성합니다. 이 모델은 멀티 뷰 CNN-Transformer 하이브리드 아키텍처로 구축되어, 구조화된 합성곱 블록(DWF-Conv/D2BR-Conv)을 활용하여 다중 스케일 지역 특성을 추출하며 과적합을 줄입니다.

- **Technical Details**: S3TU-Net의 아키텍처는 U자형 인코더-디코더 구조로, DWF-Conv 및 D2BR-Conv 블록과 함께 잔여 연결, 그리고 다중 방향 공간 이동 기술을 가진 S2-MLP Link 모듈을 포함합니다. DWF-Conv 블록은 두 개의 합성곱 레이어와 함께 작동하여 깊이 있는 특성을 집중적으로 활용하며, S2-MLP Link는 다양한 의미 수준의 특성을 융합하여 성능을 향상시킵니다. 또한, RM-SViT 모듈은 글로벌 및 로컬 특성을 결합하여 장기 종속성을 효과적으로 포착합니다.

- **Performance Highlights**: LIDC-IDRI 데이터셋에서 S3TU-Net은 DSC 89.04%, 정밀도 90.73%, mIoU 90.70%, 감도 93.70%를 기록했습니다. EPDB 개인 데이터셋에서의 검증 결과도 DSC 86.40%를 달성하여 모델의 안정성과 일반화 능력을 보여줍니다. S3TU-Net은 최근의 방법들에 비해 DSC를 4.52% 향상시켰고, 감도는 3.16% 증가하여 다양한 성능 지표에서 약 2%의 개선을 나타냈습니다.



### Data Pruning in Generative Diffusion Models (https://arxiv.org/abs/2411.12523)
- **What's New**: 이번 연구에서는 데이터 프루닝(data pruning)의 효용성을 제시하여, 생성 모델(generative model)에 대한 적용 가능성을 탐구합니다. 기존의 연구는 주로 분류(classification)와 같은 구분 모델(discriminative model)에 대한 프루닝 전략에 중점을 두었으나, 본 연구에서는 생성 확산 모델(generative diffusion model)에 데이터 프루닝이 어떻게 긍정적인 영향을 미칠 수 있는지를 조사합니다. 예를 들어, 우리는 전략적으로 중복된(redunant) 또는 노이즈가 포함된 데이터를 제거할 경우, 대규모 데이터셋의 성능을 향상시킬 수 있음을 발견했습니다.

- **Technical Details**: 본 연구에서는 CelebA-HQ와 ImageNet 데이터셋을 통해 여러 가지 프루닝 방법을 실험하고 평가하였으며, 특히 간단한 클러스터링(cluster) 기법이 높은 성능을 나타냈습니다. 생성 모델의 학습 효율을 높이기 위해, 우리는 데이터셋에서 핵심적인 샘플을 찾아내고 노이즈가 포함된 샘플을 제거함으로써, 모델이 효과적으로 학습할 수 있도록 하였습니다. 이러한 접근방식은 생성 모델이 많은 자원을 소모하지 않고도 우수한 성과를 낼 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 이미지넷 데이터셋에서 데이터의 90%를 프루닝해도 성능 저하 없이 우수한 결과를 얻을 수 있음을 확인했습니다. 또한, 클러스터링을 활용하여 비율이 불균형한 데이터셋을 조정함으로써, 과소 대표된 집단을 공정하게 샘플링할 수 있음을 보여주었습니다. 이러한 발견은 생성 모델을 훈련할 때 제한된 데이터가 있을지라도 적절한 프루닝 전략을 통해 합리적인 결과를 얻을 수 있음을 시사합니다.



### VMGNet: A Low Computational Complexity Robotic Grasping Network Based on VMamba with Multi-Scale Feature Fusion (https://arxiv.org/abs/2411.12520)
- **What's New**: 이 논문에서는 VMGNet이라는 로봇 그리핑을 위한 새로운 모델을 제안합니다. VMGNet은 낮은 계산 복잡도(low computational complexity)와 높은 정확도(high accuracy)를 목표로 하며, 로봇 그리핑 분야에 Visual State Space를 도입하여 선형 계산 복잡도를 달성합니다. 이러한 접근법은 모델의 계산 비용을 크게 줄이는 데 기여합니다.

- **Technical Details**: VMGNet은 멀티-스케일(feature fusion) 정보 추출을 위한 효율적이고 경량화된 Fusion Bridge Module을 통해 다양한 스케일에서 정보를 융합합니다. 또한, 새로운 손실 함수(loss function) 계산 방법을 통해 하위 작업(subtask) 간의 중요도 차이를 강조하여 모델의 적합성을 향상시킵니다. 실험 결과, VMGNet은 8.7G의 부동 소수점 연산(Floating Point Operations)을 소모하며, 장치에서 8.1 ms의 추론(inference) 시간을 기록했습니다.

- **Performance Highlights**: VMGNet은 Cornell과 Jacquard 공개 데이터셋에서 최첨단 성능(state-of-the-art performance)을 달성했습니다. 실제 환경에서는 다중 객체 시나리오에서 로봇 그리핑 실험을 수행한 결과, 94.4%의 성공률을 기록하며 뛰어난 성능을 보여주었습니다. 실제 로봇 그리핑 실험의 영상은 해당 링크에서 확인할 수 있습니다.



### MAViS: Modular Autonomous Virtualization System for Two-Dimensional Semiconductor Quantum Dot Arrays (https://arxiv.org/abs/2411.12516)
Comments:
          14 pages, 5 figures, 8 pages of supplemental material

- **What's New**: 본 논문에서는 Modular Automated Virtualization System (MAViS)를 소개합니다. MAViS는 다층 가상 게이트를 실시간으로 자율적으로 구성할 수 있는 일반적이고 모듈화된 프레임워크입니다. 최신 기계 학습 기법을 활용하여 2차원 전하 안정성 다이어그램에서 빠르게 특징을 추출할 수 있습니다. 이 방법은 높은 고유도 스핀 큐비트 시스템의 제어에 있어 중요한 획기적인 진전을 의미합니다.

- **Technical Details**: MAViS는 다층 가상 게이트의 전하 안정성 다이어그램을 분석하여 상호 커플링과 전자포텐셜의 제어를 가능하게 합니다. 특히, 가상 게이트는 다양한 물리적 게이트의 선형 조합으로 정의되어 이러한 고유상태를 관리합니다. 기존 방법론에 비해, MAViS는 기계 학습(Machine Learning)과 클래식한 분석 기법을 통합하여 더 넓은 범위의 제어를 가능하게 합니다. 이를 통해 직접적인 전하 센서 측정 없이도 정밀한 조정이 가능합니다.

- **Performance Highlights**: MAViS 시스템은 십 개의 양자점으로 구성된 밀집 2차원 배열에서의 완전 가상화를 성공적으로 보여주었습니다. 기존의 수동 조정에서 완전 자동화된 제어가 가능하다는 점에서, 성능의 향상은 대규모 반도체 양자점 시스템의 제어에 있어 효율적인 솔루션을 제공합니다. 실험을 통해 MVaiS의 정확한 작동을 입증하였으며, 이는 향후 대형 양자 컴퓨팅 시스템의 개발에 기여할 것으로 기대됩니다.



### 3D Reconstruction by Looking: Instantaneous Blind Spot Detector for Indoor SLAM through Mixed Reality (https://arxiv.org/abs/2411.12514)
Comments:
          21 pages, 13 figures, 3 tables

- **What's New**: 본 연구에서는 LiMRSF (LiDAR-MR-RGB Sensor Fusion) 시스템을 개발하여, 혼합 현실(Mixed Reality) 헤드셋을 통해 사용자들이 현장에서 포인트 클라우드 등록을 직관적으로 인식할 수 있도록 지원합니다. 이 시스템은 포인트 클라우드 메시를 홀로그램으로 시각화하여, 현실 장면과 실시간으로 일치시키고, 중첩된 부분에서 발견된 오류를 자동으로 강조합니다. 이러한 혁신적인 접근법은 실내 SLAM(동시 위치 추정 및 맵 작성)에서 발생하는 문제들을 해결하는 데 기여할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LiMRSF 시스템은 TCP 서버를 통해 홀로그램 요소를 MR 헤드셋으로 전송합니다. 이는 세계 좌표와 물리적 위치의 정렬을 통해 현장 재구성을 즉각적으로 시각화할 수 있게 해주며, 사용자들이 블라인드 스팟(blind spot) 및 오류를 신속하게 인식하고 현장에서 즉각적인 조치를 취할 수 있도록 합니다. 이 시스템은 높은 fidelity(충실도)를 유지하며 포인트 클라우드 등록 중 발생하는 오류 탐지에서 F1 Score 75.76%를 달성하였습니다.

- **Performance Highlights**: 주요 성능 지표로는 SSIM 0.5619, PSNR 14.1004 및 MSE 0.0389 등이 있으며, 이는 사용자가 LiMRSF 장치의 시스루 글라스를 통해 시각화하는 간소화된 메쉬 모델의 다섯 가지 섹션에서 측정된 결과입니다. 이 방법은 3D 모델 제작을 위한 상세하고 고품질의 데이터 세트를 생성하는 데 기여하며, 건축 정보 모델링(Building Information Modeling, BIM) 등 다양한 분야에 응용될 수 있는 가능성을 지니고 있습니다.



### Automatic staff reconstruction within SIMSSA proec (https://arxiv.org/abs/2411.12383)
Comments:
          15 pages

- **What's New**: 이번 논문은 음악 점수를 자동 분석하는 시스템의 후처리 과정을 다루고 있습니다. 고대 음악 점수를 포함한 음악 데이터베이스의 생성이 계속되고 있는 가운데, 이러한 음악 콘텐츠를 분석하고 해석하는 것이 중요합니다. 특히, 오선지(staff lines)를 정확하게 식별하는 것이 필수적입니다.

- **Technical Details**: 논문에서는 디지털 Salzinnes Database에서 고대 음악 점수의 오선지를 감지(detection), 추적(tracking), 보간(interpolation)하는 방법을 제안합니다. 이전의 음악 객체 식별 시스템(output)을 기반으로 한 후처리 방식으로, 오선지의 복원에 중점을 두고 있습니다. 이러한 방식은 특정 작업을 위해 설계된 만큼, 높은 성능을 나타납니다.

- **Performance Highlights**: 제안된 시스템은 고대 음악 점수의 오선지 추출에 있어 주목할 만한 성능을 보였습니다. 성과는 기존 시스템보다 향상된 정확도를 바탕으로 하여, 앞으로의 음악 점수 분석에 있어서 중요한 발전으로 평가받고 있습니다. 이를 통해 음악 데이터베이스의 활용도가 높아질 것으로 기대됩니다.



### Breathless: An 8-hour Performance Contrasting Human and Robot Expressiveness (https://arxiv.org/abs/2411.12361)
Comments:
          15 pages, 9 figures, accepted for ISRR (International Symposium of Robotics Research) 2024

- **What's New**: 이 논문은 인간 무용수와 산업 로봇 팔이 함께하는 8시간 춤 공연의 로봇 기술에 대해 설명합니다. 로봇 팔의 제어를 위해 각 관절에서 아날로그 신호를 조합하여 인간의 육체 노동에서 흔히 볼 수 있는 움직임을 재현합니다. 또한 저자들은 딥 러닝 기술을 사용하여 비디오 기반의 인간 자세 추적 및 추출 기술을 접목하여 더욱 다양한 동작을 개발했습니다. 이로 인해 인간의 표현성과 로봇 기계의 정밀성을 대비시키는 독창적인 공연이 가능해졌습니다.

- **Technical Details**: 우리는 두 가지 방식으로 로봇 팔의 동작을 설계했습니다. 하나는 인간의 움직임을 기반으로한 사인 함수이며, 다른 하나는 로봇의 실시간 반응으로 생성된 즉흥 동작입니다. 이런 조합을 통해 로봇은 부드럽고 직선적인 경로를 따라 움직이게 되며, 소품으로 사용되는 두 개의 도구인 긴 막대기와 큰 흰색 천은 공연을 더욱 풍부하게 만듭니다. 이 과정에서 OpenPose 소프트웨어를 활용하여 인간의 동작을 추적하고 로봇의 동작 계획을 개선하였습니다.

- **Performance Highlights**: 이 공연은 UR5e라는 여섯 축의 산업 로봇 팔을 사용하여, 인간의 댄서와의 협업을 통해 진행됩니다. 공연의 구성이 다양한 주제와 변형으로 이루어져 있으며, 관객 왕래를 고려한 스토리텔링을 통해 대중성을 확보합니다. 또한, 커스텀 소프트웨어 도구를 제작해 기존의 로봇 동작 계획과의 차별성을 보이고 있으며, 향후 연구자 및 예술가들이 활용할 수 있도록 공유할 예정입니다.



### Target Height Estimation Using a Single Acoustic Camera for Compensation in 2D Seabed Mosaicking (https://arxiv.org/abs/2411.12338)
Comments:
          8 pages,conference

- **What's New**: 본 연구는 저조도 해양 인식(underwater perception)을 위한 2D 해저 모자이크(seabed mosaicking)에서 목표 높이 데이터를 보완하기 위한 새로운 접근 방식을 제안합니다. 기존의 방법에서는 해양 로봇이 충돌을 피하기 위해 필요한 목표 높이 정보가 부족했지만, 이 연구는 단일 Acoustic camera를 사용하여 목표 높이를 추정하는 방안을 개발했습니다.

- **Technical Details**: 제안된 방법은 Acoustic camera의 영상에서 elevation angle의 손실을 모델링하는 것이 아닌, 이용 가능한 acoustic cast shadow(음향 그림자) 단서와 간단한 센서 이동을 활용하여 목표 높이를 빠르게 추정합니다. 이 연구는 수조 실험과 시뮬레이션 실험을 통해 제안된 방법의 타당성을 검증합니다.

- **Performance Highlights**: 제안된 방법은 기존의 3D 재구성 방법과 비교했을 때, 해양 환경에서의 목표 높이 추정 및 2D 해저 모자이크 생성에서 더 높은 효율성을 보여줍니다. 이로 인해 복잡하고 미탐사된 해양 환경에서도 작업의 안전성을 향상시킬 수 있습니다.



### C$^{2}$INet: Realizing Incremental Trajectory Prediction with Prior-Aware Continual Causal Intervention (https://arxiv.org/abs/2411.12313)
- **What's New**: 다중 에이전트의 궤적 예측을 위한 새로운 접근법, C$^{2}$INet(Continual Causal Intervention Network)를 소개합니다. 이 방법은 다양한 환경의 편향(bias)을 고려하여 궤적 데이터를 효과적으로 학습하고 예측할 수 있도록 설계되었습니다. 다양한 시나리오에서의 메모리 큐(memory queue)를 활용하여, 연속적인 학습 과정에서 발생할 수 있는 기존 정보의 손실(catastrophic forgetting)을 방지합니다.

- **Technical Details**: C$^{2}$INet는 잠재 공간의 요인(confounding factors)에 대한 사전(probabilistic prior)과 사후(posterior) 추정기(estimate)를 정렬(align)하기 위해 변분 추론(variational inference)을 사용합니다. 이를 통해 궤적 표현의 인과적 상관관계(causal correlations) 개입을 통해 궤적 데이터를 더 효과적으로 캡처합니다. 또한, 주어진 시나리오에서 최적의 변별 사전(optimal variational priors)을 저장하여 지속적인 편향 제거(debiasing)를 보장합니다.

- **Performance Highlights**: 제안된 C$^{2}$INet은 세 가지 실제 및 합성 데이터셋에서 기존의 최첨단 방법과 비교했을 때 신뢰할 수 있는 예측 성능을 일관되게 도출했습니다. 이 방법은 다양한 시나리오에 고유한 혼란 요인(confounding factors)을 효과적으로 완화하여 실제 응용 프로그램에서의 가치가 강조됩니다. 결과적으로, 각기 다른 작업에 대한 적응력을 높이며, 이전의 작업 정보를 보존하여 에지에서 발생하는 문제 해결에 기여합니다.



### GLOVER: Generalizable Open-Vocabulary Affordance Reasoning for Task-Oriented Grasping (https://arxiv.org/abs/2411.12286)
- **What's New**: 이번 논문은 GLOVER라는 새로운 프레임워크를 제안하여 로봇의 물체 조작 능력을 향상시키고자 합니다. GLOVER는 Large Language Models (LLMs)를 활용하여 인간의 지시에 맞는 graspable object parts를 예측하고, 고유의 시각적 affordance를 추론하는 방식을 채택합니다. 이 프레임워크는 10,000개 이상의 이미지 데이터를 기반으로 multi-modal fine-tuning을 통해 물체 이해도를 높이고, 복잡한 도구 사용을 가능하게 합니다.

- **Technical Details**: GLOVER는 Affordance-Aware Grasping Estimation (AGE) 모듈을 통해 실제 로봇에 효율적으로 배포됩니다. AGE는 비모수적인 방법으로, 스테레오 affordance 지오메트리에 기반하여 그리퍼의 포즈를 추정합니다. 이 방법은 기존의 학습 기반 grasp planners에 비해 성능과 효율성이 높아, 평균적으로 40배 빠른 속도를 자랑합니다.

- **Performance Highlights**: GLOVER는 30개의 실제 씬에서 평가되었으며, part identification 성공률이 86.0%로 높은 성과를 보였습니다. 또한 grasping 성공률은 76.3%로, 기존의 최첨단 방법에 비해 affordance reasoning과 grasping pose estimation이 각각 330배, 40배 더 빠른 속도를 기록했습니다. 이러한 결과는 GLOVER의 실용성을 보여주는 중요한 지표로 작용합니다.



### Versatile Cataract Fundus Image Restoration Model Utilizing Unpaired Cataract and High-quality Images (https://arxiv.org/abs/2411.12278)
Comments:
          12 pages, 8 figures

- **What's New**: 이 연구에서는 백내장 진단을 위한 새로운 이미지 복원 방법인 Catintell을 제안합니다. Catintell은 실시간 복원과 시뮬레이션 이미지를 생성하는 두 가지 모델인 Catintell-Syn과 Catintell-Res로 구성되어 있습니다. Catintell-Syn은 GAN(gcng) 아키텍처를 사용하여 실제 스타일과 질감을 가진 백내장 유사 이미지를 생성하며, Catintell-Res는 이러한 합성 이미지를 통해 실제 카타락트 앨범 이미지를 복원하는 데 사용됩니다.

- **Technical Details**: Catintell-Syn는 완전 비지도 학습 데이터로 페어링된 백내장 유사 이미지를 생성하여 기존의 Gaussian degradation 알고리즘보다 더 높은 효과를 보여줍니다. Catintell-Res는 CNN과 Transformer를 결합한 구조로, Dense Convolution Block(DCB)과 Window-based Self-attention Block(WSB)을 통해 지역 및 비지역 특성을 각각 캡처하여 복원 성능을 최적화합니다. 이 모델은 다양한 데이터셋에 대한 일반화 성능을 입증하며, 클리닉 응용에서 기존의 방법들보다 월등한 성능을 기록하고 있습니다.

- **Performance Highlights**: Catintell-Res는 PSNR(39.03) 및 SSIM(0.9476)에서 다른 백내장 이미지 복원 방법을 초월하는 성능을 나타냅니다. 이 연구는 실제 백내장 이미지를 다양한 외부 데이터셋에 적용하여 일반화 가능한 성능을 입증했습니다. Catintell 모델이 백내장 환자의 다른 실명 질병을 파악하는 데 도움이 되기를 기대하며, 향후 의료 이미지를 복원하는 방법의 발전을 촉진할 수 있을 것으로 전망됩니다.



### libcll: an Extendable Python Toolkit for Complementary-Label Learning (https://arxiv.org/abs/2411.12276)
Comments:
          10 pages, 3 figures

- **What's New**: 본 논문에서는 complementary-label learning (CLL)이라는 약한 감독 학습 방법론의 주요 문제점을 다루기 위해 	exttt{libcll}이라는 확장 가능한 파이썬 툴킷을 새롭게 소개합니다. 	exttt{libcll}은 다양한 CLL 알고리즘과 데이터셋을 지원하는 범용 인터페이스를 제공하여 일관성 문제를 해소하고, 연구 과정을 간소화하도록 설계되었습니다. 이 툴킷은 CLL 기술을 효율적으로 채택하고 구현할 수 있도록 설치가 용이하고 포괄적인 사용 가이드를 제공합니다.

- **Technical Details**: CLL은 각 레이블이 데이터 인스턴스에 속하지 않는 클래스를 나타내는 약한 감독 학습 문제로, 고급 그래프 구조와 다양한 네트워크 아키텍처를 지원합니다. 	exttt{libcll}은 사용자 정의 전이 행렬을 사용해 보완 레이블을 생성하는 기능을 포함하였으며, 다양한 CLL 알고리즘과 데이터셋에 대한 광범위한 벤치마크를 제공합니다. 이 툴킷은 학습 성능의 일관성을 유지하고 연구자들이 쉽게 비교하고 분석할 수 있도록 도와줍니다.

- **Performance Highlights**: 	exttt{libcll}을 사용한 포괄적인 ablation 연구는 CLL 연구를 발전시키기 위한 중요한 인사이트를 생성함을 입증하였습니다. 15개 데이터셋과 14개 알고리즘의 벤치마크 결과를 통해 연구자들이 각 방법의 강점과 한계를 평가할 수 있는 통합된 관점을 제공합니다. 이 툴킷은 CLL 분야에서의 연구 협력과 재현성을 촉진하며, 더 나은 알고리즘 개발에 기여할 것으로 기대됩니다.



### Acquire Precise and Comparable Fundus Image Quality Score: FTHNet and FQS Datas (https://arxiv.org/abs/2411.12273)
Comments:
          11 pages, 7 figures

- **What's New**: 본 논문에서는 기존의 fundus image quality assessment (FIQA) 기법의 한계를 극복하기 위해 Fundus Quality Score (FQS)라는 새로운 데이터셋을 제안합니다. 이 데이터셋은 2246개의 fundus 이미지를 포함하며, 두 가지 라벨(0에서 100 사이의 Mean Opinion Score와 세 가지 품질 레이블)을 제공합니다. 또한, FIQA Transformer 기반의 Hypernetwork(FTHNet)를 제안하여 회귀 결과를 생성함으로써 기존의 분류 결과 중심 접근법의 한계를 해결하려고 합니다.

- **Technical Details**: FTHNet은 Transformer Backbone, Distortion Perception Network, Parameter Hypernetwork, 및 Target Network의 네 부분으로 구성됩니다. Transformer Backbone은 기본 Transformer 블록(BTBs)을 통해 비국소적 자기 유사성 및 장기 의존성을 포착합니다. Distortion Perception Network는 다양한 해상도에서 왜곡 정보를 수집하며, Parameter Hypernetworks는 fundus 이미지 콘텐츠에 따라 동적으로 가중치 및 편향을 생성합니다.

- **Performance Highlights**: FTHNet의 성능은 PLCC가 0.9423, SRCC가 0.9488에 달하며, 기존의 다양한 FIQA 방법보다 우수한 결과를 보여줍니다. 연구 결과는 FIQA 방법이 임상 진단의 품질 관리에 기여할 수 있는 잠재력을 갖고 있음을 시사합니다. 최종적으로, 논문에서는 10-fold cross-validation을 통해 결과의 유의미성을 보장합니다.



### Enhancing Low Dose Computed Tomography Images Using Consistency Training Techniques (https://arxiv.org/abs/2411.12181)
- **What's New**: 본 논문에서는 새로운 beta noise 분포를 도입하여 이미지 생성의 품질을 높이면서도 적은 수의 매개변수로 조절할 수 있는 유연성을 제공합니다. 이를 통해 High Noise Improved Consistency Training (HN-iCT)라는 새로운 훈련 방식을 제안하여, 저선량 이미지(Low Dose)에서 중요한 특징을 추출할 수 있습니다. 또한, sinusoidal curriculum 기법을 활용하여 노이즈의 다양한 수준을 관리함으로써 모델의 학습 효율성을 극대화하였습니다.

- **Technical Details**: Consistency 모델은 확률적 미분 방정식(stochastic differential equation, SDE)을 통해 데이터 분포를 노이즈 분포로 점진적으로 변환하여 데이터를 생성하는 방법을 사용합니다. 이 모델들은 변환 과정을 역으로 학습하여 노이즈에서 데이터를 생성합니다. 최근 제안된 HN-iCT 아키텍처는 Weighted Attention Gates(WAG)를 활용하여 조건부 이미지로부터 신뢰할 수 있는 특징을 추출하며, 이는 저선량 CT 이미지를 효과적으로 처리하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, HN-iCT를 사용한 무조건적인 이미지 생성이 CIFAR10 및 CelebA 데이터셋에서 기존의 CT 및 iCT 훈련 기법에 비해 상당히 우수한 성능을 보였습니다. 또한, 이미지 조건 모델은 저선량 CT 스캔을 향상시키는 데 뛰어난 성능을 나타내어, 새로운 접근 방식이 임상 요구사항을 효과적으로 충족할 수 있음을 시사합니다.



### AsynEIO: Asynchronous Monocular Event-Inertial Odometry Using Gaussian Process Regression (https://arxiv.org/abs/2411.12175)
Comments:
          Submitted to IEEE (2024-11-4)

- **What's New**: 이번 연구에서는 비동기 이벤트 카메라와 관성 센서의 데이터를 통합하여 결과적인 움직임 추정을 위한 새로운 방법론인 AsynEIO(Asynchronous Event-Inertial Odometry)를 제안하였습니다. 아울러, 본 논문은 연속적인 Gaussian Process (GP) 회귀 프레임워크를 기반으로 하여 이벤트 카메라와 IMU(Inertial Measurement Unit) 데이터를 통합하는 방법을 탐구합니다.

- **Technical Details**: 제안하는 AsynEIO 방식은 고차원 시간 해상도에서 원시 이벤트 스트림으로부터 직접 특징 궤적을 추적하는 이벤트 기반 프론트엔드를 포함합니다. 이 궤적은 다양한 관성 요소와 통합되어 서로 비동기적인 데이터를 통합할 수 있는 GP 회귀 프레임워크 안에서 활용됩니다. 또한, 잔차 Jacobians와 노이즈 모델을 유도하여 슬라이딩 윈도우 최적화를 통해 반복적으로 최적화되는 팩터 그래프를 구성합니다.

- **Performance Highlights**: 실험 결과, AsynEIO는 공공 데이터셋과 수집한 이벤트-관성 시퀀스에서 기존 방법들과 비교했을 때 성능이 뛰어남을 보여주었습니다. 특히 고속 및 저조도 환경에서의 성능 개선이 두드러지며, 다양한 관성 융합 전략의 성능 차이를 비교하여 최적의 선택지를 제시합니다.



### Just KIDDIN: Knowledge Infusion and Distillation for Detection of INdecent Memes (https://arxiv.org/abs/2411.12174)
- **What's New**: 이번 연구에서 제안한 KID-VLM 프레임워크는 큰 비주얼 언어 모델(LVLM)로부터의 지식 증류(Knowledge Distillation, KD)와 상식 지식 그래프(Knowledge Graph, KG)에서의 지식 주입(Knowledge Infusion)을 결합하여 공격적인 메메의 독성 탐지 성능을 향상시킵니다. 이 프레임워크는 ConceptNet이라는 대규모 상식 KG에서 서브 지식 그래프를 추출하여 compact VLM 프레임워크 내에서 주입하여 독성 문구와 메메 내의 시각적 개념 간의 관계적 맥락을 강화합니다.

- **Technical Details**: KID-VLM은 CLIP을 백본으로 사용하여 메메의 시각적 및 텍스트 피쳐를 추출하며, LLaVA 교사 모델로 생성된 캡션을 위한 텍스트 인코더와 메메 텍스트를 위한 또 다른 텍스트 인코더를 사용합니다. 피쳐 상호작용 매트릭스(Feature Interaction Matrix, FIM)를 계산하여 시각적 및 텍스트 데이터를 정렬하고, 일관성 손실(consistency loss)을 사용하여 교사 모델의 지식을 정제하여 학생 모델이 내재된 맥락적 단서를 포착하도록 학습합니다.

- **Performance Highlights**: 두 개의 증오 발언 기준 데이터셋에서 평가한 결과, KID-VLM 프레임워크는 AU-ROC, F1, 리콜에서 각각 1.1%, 7%, 35% 향상된 성능을 보여주며 기존 최첨단 모델들을 초월하는 성과를 기록했습니다. 이러한 성과는 두 가지 맥락 학습 접근법의 필요성을 강조하며, LVLM으로부터의 잠재적 패턴과 KG로부터의 외적 관계 지식을 함께 캡처합니다.



### Self-supervised denoising of visual field data improves detection of glaucoma progression (https://arxiv.org/abs/2411.12146)
Comments:
          10 pages

- **What's New**: 이 연구에서는 4,000명 이상의 환자에서 수집된 시야(data from visual field) 데이터를 노이즈 제거(denoising)하는 데 자기 지도 학습(self-supervised learning)을 활용했습니다. 기존의 방법보다 시야 데이터의 평균 대 잡음 비율(signal-to-noise ratio)을 개선했고, 안압 상승으로 인한 시각 손상의 진행을 더 정확하게 감지할 수 있었습니다. 우리는 변이형 오토인코더(variational autoencoder)와 마스크 오토인코더(masked autoencoder)를 사용하여 어떤 모델이 시야 데이터를 더 잘 처리하는지 비교했습니다.

- **Technical Details**: 이 연구에서는 심층 신경망(deep neural networks) 구조를 기반으로 한 두 가지 모델, 변이형 오토인코더(VAE)와 마스크 오토인코더가 시야 데이터의 노이즈를 줄이는 데 사용되었습니다. 데이터 수집은 Humphrey Field Analyzer II를 통해 진행되었으며, 신뢰성 기준(criterion of reliability)은 잘못된 양성(false positive) 비율이 15% 이하, 고정 손실(fixation loss) 및 잘못된 음성(false negative) 비율이 30% 이하로 설정되었습니다. 모델 출력에서 카테고리 p-값(categorical p-value)을 사용하는 방식도 효과성이 입증되었습니다.

- **Performance Highlights**: 마스크 오토인코더는 시야 데이터를 효과적으로 정리하여 변이형 오토인코더보다 4.7% 더 높은 진행 예측률을 기록했습니다. 포함된 p-값 덕분에 진행을 예측하는 시간이 평균 2.3개월 더 빨라졌다는 분석이 이루어졌습니다. 이러한 결과는 마스킹(masking) 및 p-값 포함이 시야 진행 감지 작업을 개선하는 데 기여할 수 있음을 보여줍니다.



### Coverage-Constrained Human-AI Cooperation with Multiple Experts (https://arxiv.org/abs/2411.11976)
- **What's New**: 본 논문에서는 Coverage-constrained Learning to Defer and Complement with Specific Experts (CL2DC) 방법을 제안하여 기존 HAI-CC 방법의 연구 공백을 해결합니다. CL2DC는 AI 예측 단독, 특정 전문가에게 위임(deferring) 또는 보완하는 방식으로 최종 결정을 내리며, 입력 데이터에 따라 적절한 결정을 할 수 있도록 하였습니다. 또한, 협력 비용을 제어하기 위한 coverage-constrained 최적화 방법을 도입하여 AI 전용 선택에 대한 목표 확률을 근사화합니다.

- **Technical Details**: CL2DC는 noisy-label 주석이 포함된 훈련 세트를 고려하여, 최종 결정을 자율적으로 내리거나 특정 전문과 협력할 수 있는 시스템을 제공합니다. 이 방법은 각 전문가의 특정 전문성을 평가하고 가장 적합한 전문가를 선택하는 동시에, 손실 함수에 커버리지 제약 패널티를 도입하여 커버리지 수준을 효과적으로 조절합니다. 이렇게 하여 CL2DC는 커버리지-정확도 곡선을 분석하는 데 있어 일관되고 의미 있는 분석이 가능합니다.

- **Performance Highlights**: CL2DC는 다양한 실제 및 합성 다중 조율자 노이즈 레이블 벤치마크에서 최신 HAI-CC 방법과 비교하여 높은 정확도를 달성함을 보여주었습니다. 특히 CIFAR-100, Galaxy Zoo, HAM10000, NIH-ChestXray와 같은 데이터 세트에서 동일한 커버리지 값에 대해 CL2DC가 이전 HAI-CC 방법을 지속적으로 능가했습니다. 이러한 성과는 여러 전문가 지식을 활용하여 보다 견고한 의사 결정을 가능하게 합니다.



### Dataset Distillers Are Good Label Denoisers In the Wild (https://arxiv.org/abs/2411.11924)
- **What's New**: 최근 노이즈가 포함된 데이터를 학습하는 방법이 딥러닝 모델을 실제 적용에 맞게 조정하는 데 필수적임을 보여주고 있습니다. 기존의 접근 방식은 대개 초기 노이즈를 평가한 후 노이즈가 포함된 샘플을 버리거나 재가중치를 적용하거나 재라벨링을 통해 문제를 해결했습니다. 그러나 이러한 방법은 초기 노이즈 평가가 부정확한 경우 성능이 저하되는 악순환에 빠질 수 있습니다.

- **Technical Details**: 본 연구에서는 데이터셋 증류(dataset distillation)를 활용하여 노이즈를 제거하는 새로운 접근 방식을 제안합니다. 주요 기법으로는 세 가지 대표적인 데이터셋 증류 방법(DATM, DANCE, RCIG)을 사용하며, 이를 대칭 노이즈, 비대칭 노이즈, 그리고 실제 자연 노이즈와 같은 다양한 노이즈 조건에서 엄격히 평가하였습니다. 이 방법은 기존 기술에서 공통적으로 발견되는 피드백 루프를 피하고, 오프라인 처리를 통해 강력한 개인정보 보호를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, 데이터셋 증류는 랜덤 노이즈 시나리오에서 효과적인 디노이징 도구로 작용하지만, 구조화된 비대칭 노이즈 패턴에서는 어려움을 겪을 수 있습니다. 또한 불균형 데이터 세트의 꼬리 클래스에서 발생하는 도전적인 샘플은 증류 과정에서 손실 압축 위험이 있음을 보여줍니다. 그럼에도 불구하고, 본 연구의 결과는 데이터셋 증류가 노이즈가 만연한 고 프라이버시 환경에서 견고한 모델 학습에 큰 가능성을 가진다는 것을 강조합니다.



### ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling (https://arxiv.org/abs/2411.11911)
- **What's New**: 이번 연구에서는 다중 모드 비 예측(multi-modal motion prediction) 문제를 해결하기 위해 ModeSeq라는 새로운 예측 패러다임을 제안합니다. ModeSeq는 모드를 시퀀스(seqeunce)로 모델링하여 연속적으로 다음 모드를 추정하며, 이는 이전 모드 간의 상관관계를 보다 명확하게 캡처할 수 있게 해 줍니다. 또한 Early-Match-Take-All (EMTA) 훈련 전략을 도입하여 경로 다변량성을 더욱 향상시키는 효과를 거두었습니다.

- **Technical Details**: ModeSeq는 기존의 전통적인 다중 모드 예측 방식에서 벗어나, 모드를 한 번에 예측하는 대신 한 단계씩 순차적으로 예측합니다. 이 과정에서 모델은 각 예측 단계마다 이전 모드는 물론 그 신뢰도(confidence)까지 고려합니다. 이를 통해 각 모드 간의 관계를 명확히 파악할 수 있으며, 고속 모드 예측 및 후처리 단계를 필요로 하지 않고도 높은 품질의 경로 출력을 생성할 수 있습니다.

- **Performance Highlights**: ModeSeq는 Waymo Open Motion Dataset과 Argoverse 2 Motion Forecasting Dataset에서 여러 다른 다중 모드 예측 방법들과 비교하여 더 균형 잡힌 성능을 달성했습니다. 특히 모드 커버리지(mode coverage), 모드 스코어링(mode scoring), 그리고 경로 정확도(trajectory accuracy) 면에서 우수한 성능을 보였으며, 순차적인 모드 모델링 덕분에 높은 불확실성 환경에서도 다양한 행동 모드를 예측하는 능력을 자연스럽게 내재하고 있습니다.



New uploads on arXiv(cs.AI)

### Neurosymbolic Graph Enrichment for Grounded World Models (https://arxiv.org/abs/2411.12671)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 반응 능력을 활용하여 복잡한 문제를 해결하고 깊이 있는 맥락적 의미를 해석하는 새로운 접근 방식을 제안합니다. 이는 이미지 입력을 시작으로 자연어 설명을 생성하며 이를 추상 의미 표현(AMR) 그래프 형태에 변환하여 다층적인 지식 그래프를 구성합니다. 이러한 접근법은 비구조적 언어 모델과 형식적 의미 구조 사이의 간극을 메우고, AI 시스템이 인간과 유사한 추론을 할 수 있도록 개선합니다.

- **Technical Details**: 제안된 방법론은 이미지 입력에서 출발하여 최신 LLM을 활용하여 자연어 설명을 생성하고, 이를 AMR 그래프로 형식화하여 지식 그래프를 구성하는 것을 핵심으로 합니다. 이 과정에서 implicit knowledge를 활용하여 발생하는 다양한 의미적 패턴과 맥락적 지식을 추출하고, 이를 기반으로 LLM에서 생성된 지식을 강화합니다. 이러한 방법론은 다층적인 지식 그래프의 지속적인 확장 및 적응을 가능하게 하는 피드백 루프를 구현하여, 새로운 맥락에 대한 이해를 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구에서 제시한 시스템은 자연어 이해, 시각적 추론, 복잡한 문제 해결 작업에서 뛰어난 성과를 보여줍니다. LLM과 지식 그래프의 융합을 통해 기존 방법보다 지식 기반 풍부화 과정이 더 민첩하고 확장 가능해졌습니다. 실험 결과는 이 접근 방식이 여러 지식 도메인에서 효율성을 증명함을 나타내며, 이는 AI 시스템이 인간과 유사한 맥락적 이해를 할 수 있는 새로운 길을 열어줍니다.



### Preference-Conditioned Gradient Variations for Multi-Objective Quality-Diversity (https://arxiv.org/abs/2411.12433)
- **What's New**: 이 연구에서는 다목적(Multi-Objective) Quality-Diversity 알고리즘인 Multi-Objective Map-Elites with Preference-Conditioned Policy-Gradient and Crowding Mechanisms (mome-p2c)를 소개합니다. 본 방법은 고차원 문제 공간 내에서 더 효율적으로 유망한 목표 영역을 발견하는 데 초점을 맞추고 있으며, 이를 통해 균일한 Pareto 프런트 분포를 촉진합니다. mome-p2c는 기존의 방법보다 낮은 계산 저장 비용을 통해 더 매끄러운 무역과의 세트를 달성합니다.

- **Technical Details**: mome-p2c는 각 목적을 위해 개별 actor-critic 네트워크를 사용하는 대신 단일의 preference-conditioned actor 및 critic을 사용하여 요구되는 무역을 달성하는 솔루션을 개선할 수 있습니다. 이 방식은 네트워크의 메모리 및 훈련 비용을 줄이며, 동시에 정책-그래디언트 변화를 통해 목표 간의 균형을 맞춘 해를 제공합니다. 여기에 crowding 메커니즘을 추가하여 Pareto 프런트 상의 솔루션 간 균일한 분포를 촉진합니다.

- **Performance Highlights**: mome-p2c는 여섯 가지 로봇 제어 작업에서 기존의 mome-pgx 알고리즘과 비교하여 우수한 성능을 보였으며, 새로운 제안된 삼목적 작업에서도 성과를 냈습니다. 또한, 새로운 희소성 기반 메트릭에서도 mome-pgx보다 더 부드러운 무역을 초월하는 성능을 입증했습니다. 이는 다양한 목표 간에서 사용자 맞춤형 솔루션을 선택할 수 있는 기회를 제공합니다.



### SNN-Based Online Learning of Concepts and Action Laws in an Open World (https://arxiv.org/abs/2411.12308)
- **What's New**: 이번 연구에서는 스파이킹 신경망(spiking neural network, SNN)을 기반으로 한 완전 자율 바이오 영감을 받은 인지 에이전트의 아키텍처를 제시합니다. 이 에이전트는 자신의 우주를 탐색하며 사물 및 상황의 개념을 순식간에 배웁니다. 특히 행동 개념은 초기 상황(initial situation), 운동 활동(motor activity), 결과(outcome)로 구성된 삼중 구조로 기록되어 에이전트가 우주의 행동 법칙을 이해할 수 있도록 돕습니다.

- **Technical Details**: 에이전트는 개념 개수를 다르게 설정하여 사물/상황 개념과 행동 개념을 분류합니다. 결정-making 과정에서 에이전트는 자신이 가진 의미 기억(semantic memory)을 쿼리하여 예상되는 결과를 기반으로 행동을 선택합니다. 이 방식은 에이전트가 새로운 상황에 대처하는 데 도움을 주며, 더 이전에 학습한 일반 개념을 이용해 빠르게 적응하도록 만듭니다.

- **Performance Highlights**: 실험 결과, 에이전트는 새로운 상황을 효과적으로 처리하며 이전에 학습한 일반 개념에 의존하여 빠르게 개념을 수정할 수 있음을 보여줍니다. 이러한 기능은 에이전트가 환경 변화에 적절하게 반응할 수 있는 능력을 강조합니다.



### Restructuring Tractable Probabilistic Circuits (https://arxiv.org/abs/2411.12256)
- **What's New**: 본 논문에서는 확률 회로(Probabilistic Circuits, PCs)의 구조 조정 문제를 제안하고 연구합니다. 기존의 곱셈 알고리즘이 동일한 구조를 유지해야 하는 반면, 본 연구는 특정 목표의 vtree에 맞는 구조로 PC를 변환하는 방법을 제시합니다. 이로 인해 다양한 구조를 가진 회로를 효율적으로 곱할 수 있는 새로운 다항 시간 알고리즘과 실용적인 깊이 감소 알고리즘을 도입하였습니다.

- **Technical Details**: 확률 회로는 확률 분포를 계산 그래프로 표현하며, 이 구조에서 각 노드는 sum, product 및 leaf 노드로 구성됩니다. 논문의 주요 알고리즘은 그래픽 모델 표현을 활용하여 조건부 독립성을 이해하고, 원하는 구조에 맞는 새로운 PC를 재구성하는 것입니다. 또한, 두 가지 중요한 응용 분야인 회로 곱셈과 깊이 감소를 탐구하고 있습니다.

- **Performance Highlights**: 특히 새로운 클래스의 contiguous circuits를 도입하여, 서로 다른 구조의 회로를 다항 시간 내에 곱할 수 있는 가능성을 보여주었습니다. 또한, 깊이 감소 알고리즘을 통해 PC 추론을 더 빠르게 개선하는 방법을 제시하였으며, 이는 PC를 효과적으로 실행하는 데 새로운 가능성을 여는 결과를 가져왔습니다.



### Efficient Training in Multi-Agent Reinforcement Learning: A Communication-Free Framework for the Box-Pushing Problem (https://arxiv.org/abs/2411.12246)
Comments:
          17 pages, 16 figures

- **What's New**: 이 논문은 Self-organizing 시스템에서 Autonomous agents가 중앙 관리자 없이도 복잡한 작업을 수행하고 동적 환경에 적응할 수 있도록 하는 Shared Pool of Information (SPI) 모델을 제안합니다. 기존의 연구들은 Reinforcement learning을 통해 에이전트가 작업 수행에 필요한 기술을 학습하게 하지만, 에이전트들 간의 힘이 상쇄되는 문제를 해결하지 못했습니다. SPI를 활용함으로써 에이전트들 간의 조정을 촉진하고, 훈련 효율성을 높이는데 기여합니다.

- **Technical Details**: SPI는 모든 에이전트에게 초기화 시에 정보를 제공하여, 에이전트 간의 충돌을 줄이고 협력을 증대시키도록 설계되었습니다. 특히, 에이전트들은 서로를 인식하지 못하는 환경에서, 공간적 제약 아래에서 협력하도록 돕습니다. 이 모델은 fitness 테스트를 통해 유효성을 검증하며, 높은 box 이동성과 방향성을 구하도록 조정됩니다.

- **Performance Highlights**: 컴퓨터 시뮬레이션을 통해 SPI는 훈련 과정을 가속화하고 에피소드당 단계 수를 줄이는 등, 에이전트의 협력 효과성을 현저히 향상시켰습니다. 에이전트 간의 상쇄 행동을 줄임으로써, 각 에이전트의 탐색이 더 의미있게 되었습니다. SPI는 타 알고리즘과 함께 사용될 수 있어 Multi-agent reinforcement learning 문제의 협력과 조정을 해결하는 데 기여할 수 있습니다.



### The Role of Accuracy and Validation Effectiveness in Conversational Business Analytics (https://arxiv.org/abs/2411.12128)
- **What's New**: 이 연구는 대화형 비즈니스 분석(conversational business analytics)이라는 접근 방식을 제시합니다. 이는 AI를 활용하여 전통적인 셀프 서비스 분석에서 사용자들이 직면하는 기술적 역량 부족을 극복할 수 있도록 돕습니다. 자연어 대화를 통해 사용자가 데이터를 독립적으로 검색하고 통찰력을 생성할 수 있게 하는 것이 목표입니다.

- **Technical Details**: 연구는 자연어 요청을 SQL 문으로 변환하는 Text-to-SQL 기술을 중심으로 진행됩니다. 예상 효용 이론(expected utility theory)에 기반한 모델을 활용하여 AI의 부분적 또는 전체적 지원이 인간 전문가에게 위임하는 것보다 더 나은 성과를 낼 수 있는 조건을 식별합니다. 이 과정에서 AI가 생성한 SQL 쿼리의 정확도가 특정 기준을 초과할 때 부분적 지원이 가능한 점과 AI가 생성한 정보뿐만 아니라 설명을 통해 검증을 제공하는 전체적 지원의 중요성도 다루어집니다.

- **Performance Highlights**: 부분적 지원과 전체적 지원 전략의 동역학을 분석하여 정보 생성 및 검증에 대한 시사점을 제공합니다. 연구 결과, 사용자 기반 검증은 잘못된 판단이나 유효 SQL 쿼리의 거부와 같은 문제를 야기할 수 있으며, 이는 대화형 비즈니스 분석의 효과를 제한할 수 있음을 강조합니다. 따라서 사용자 지원 개선, 자동화 프로세스, 사용자 기술 수준과 무관하게 품질을 평가할 수 있는 방법의 필요성이 강조됩니다.



### TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Mod (https://arxiv.org/abs/2411.12064)
Comments:
          Accepted to ACM SIGKDD 2025 Research Track

- **What's New**: 이번 논문에서는 Travelling Salesman Problem Rank (TSPRank)라는 하이브리드 페어와이즈-리스트와이즈 순위 매기기 방법을 소개합니다. 기존의 LETOR 방법들은 페어와이즈 비교에만 초점을 맞추어 글로벌 최적 순위를 제공하는 데 한계를 보였지만, TSPRank는 순위 문제를 잘 알려진 조합 최적화 문제인 여행하는 세일즈맨 문제(TSP)로 재구성했습니다. 이로 인해 TSPRank는 페어와이즈 관계를 모델링하면서 리스트와이즈 순위를 결정할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: TSPRank는 조합 최적화 기법을 통해 페어와이즈와 리스트와이즈 순위 방식의 장점을 모두 활용합니다. 구체적으로는, TSPRank가 다양한 백본 모델에서 임베딩을 생성하여 순위 성능을 향상시키는 방식으로 적용될 수 있습니다. 또한 두 가지 학습 방법, 즉 지도의 상대적 참조를 사용하는 지역 방식과 TSP 솔버를 포함한 글로벌 방식이 소개되어 있습니다.

- **Performance Highlights**: 실험 결과, TSPRank는 주식 순위 매기기, 정보 검색, 역사적 사건 정렬 등 다양한 작업에서 기존의 페어와이즈 및 리스트와이즈 방법을 능가하는 성능을 보였습니다. 특히, TSPRank의 주요 장점은 글로벌 정보를 더 잘 활용하여 순위를 매길 수 있다는 점이며, 이러한 특성 덕분에 다양한 도메인에서의 강력한 성능을 나타냅니다.



### Regret-Free Reinforcement Learning for LTL Specifications (https://arxiv.org/abs/2411.12019)
- **What's New**: 본 논문에서는 LTL 규격을 위한 최초의 무후회(무정체) 온라인 알고리즘을 제안합니다. 이 알고리즘은 유한한 상태와 행동을 가진 Markov 결정 과정(MDP)에서 작동하며, 최적 행동에 도달하기 위한 학습 중 성능을 평가합니다. 기존의 LTL 과제를 위한 붕괴없는 RL 기법이 없었던 점에서, 이 연구는 중요한 기여를 합니다.

- **Technical Details**: 제안하는 알고리즘은 무한 지평선 도달-회피 문제를 해결하기 위해 무후회 지도 학습 기법을 기반으로 합니다. 기존의 RL 알고리즘은 최적 정책이 위치적일 것이라는 가정을 하고 있으며, 이는 보다 일반적인 보상 구조에서는 유효하지 않을 수 있습니다. LTL 공식을 적용한 제어 목표에서는 보통 최적 정책이 위치적이지 않기 때문에, 이 알고리즘은 그래프 구조를 인식하고 최소 전이 확률을 가정합니다.

- **Performance Highlights**: 현재 제안된 알고리즘은 수렴 속도가 서브선형적이며, 무후회 학습 알고리즘을 통해 특정 LTL 목표에 대한 성능 평가를 제공합니다. K > 0 에피소드를 수행한 후, 알고리즘의 후회 값은 K에 대해 0으로 수렴합니다. 이를 통해 알고리즘이 안전하고 신뢰할 수 있는 방식으로 제어 정책을 학습하고 성능을 평가할 수 있음을 보여줍니다.



### On-Board Vision-Language Models for Personalized Autonomous Vehicle Motion Control: System Design and Real-World Validation (https://arxiv.org/abs/2411.11913)
- **What's New**: 이번 연구에서는 개인화된 자율주행을 위한 효율적이고 경량화된 Vision-Language Models (VLM) 기반 시스템을 제안합니다. 이 시스템은 낮은 지연 시간의 개인화된 주행 성능을 제공하면서도 강력한 추론 기능을 유지합니다. 특히, Retrieval-Augmented Generation (RAG) 기반 메모리 모듈을 통해 지속적으로 개인의 운전 선호도를 학습할 수 있는 기능이 포함되어 있습니다.

- **Technical Details**: 본 시스템은 9B 매개변수를 가진 VLM을 활용하여 시각적 정보(날씨, 도로 유형, 교통 상황)와 언어적 명령을 처리합니다. 이를 통해 개인의 운전 스타일에 맞춘 제어 전략을 생성하며, 명령 해석 및 추론 능력을 유지합니다. VLM은 사람의 명령과 시각적 입력을 실행 가능한 제어 시퀀스로 변환하는 역할을 합니다.

- **Performance Highlights**: 실제 차량 배포와 실험을 통해, 본 시스템은 안전하고 편안한 개인화된 주행 경험을 제공하며 다양한 시나리오에서의 차량 상의 인수율을 최대 76.9%까지 감소시켰습니다. 이것은 자율주행 차량에서 VLM 기반 모션 제어 시스템의 첫 번째 종단 간(end-to-end) 구현으로, 사용자 맞춤형 경험을 지속적으로 제공할 수 있는 잠재력을 보여줍니다.



### ResLearn: Transformer-based Residual Learning for Metaverse Network Traffic Prediction (https://arxiv.org/abs/2411.11894)
- **What's New**: 이 연구는 메타버스 네트워크 트래픽을 예측하기 위한 종합 솔루션을 제안합니다. 특히 Virtual Reality (VR), Augmented Reality (AR), Mixed Reality (MR) 트래픽의 실제 데이터를 포착한 최신 테스트베드를 도입하며, 이를 연구 커뮤니티와 공유합니다. 새로운 view-frame (VF) 알고리즘과 Transformer 기반의 ResLearn 알고리즘을 통해 예측 정확도를 크게 향상시켰습니다. 이 솔루션은 인터넷 서비스 제공업체(ISP)에게 실시간 네트워크 관리 도구를 제공합니다.

- **Technical Details**: 시스템 모델에서는 VF 알고리즘을 통해 프레임 관련 데이터를 추출하며, 주요 특징으로는 프레임 수(frame count), 프레임 크기(frame size), 프레임 간 도착 시간(inter-arrival time)을 포함합니다. VF 알고리즘은 애플리케이션 수준의 정보(시간, 패킷 길이, 패킷 방향 및 패킷 간 도착 시간)를 사용하여 보다 정확한 예측을 가능하게 합니다. 프레임 식별을 위해 패킷 길이와 도착 시간의 임계 값을 설정하여 해당 범주에 적합한 패킷을 식별합니다.

- **Performance Highlights**: ResLearn 알고리즘은 과거의 작업보다 99% 더 나은 성과를 보여주며, 메타버스 트래픽 예측에 있어 신뢰성을 획기적으로 개선하였습니다. 이 시스템은 다양한 메타버스 렌더링 플랫폼에서 검증되며, 메타버스 네트워크 트래픽의 주요 특징을 예측하기 위해 복잡한 기계 학습 기술을 활용합니다. 결과적으로, 이 솔루션은 높은 품질의 서비스(QoS)를 유지하는 동시에 사용자 경험을 향상시키기 위한 효과적인 도구로 자리매김할 것입니다.



### Survey on Semantic Interpretation of Tabular Data: Challenges and Directions (https://arxiv.org/abs/2411.11891)
- **What's New**: 이 논문은 Semantic Table Interpretation (STI)에 대한 포괄적인 개요를 제공하며, 31개의 특성을 기반으로 한 분류 체계를 통해 다양한 접근 방식을 정리합니다. 또한, 12개의 기준을 사용하여 사용 가능한 도구들을 평가하고, STI 접근 방식을 평가하기 위한 Gold Standards에 대한 심층 분석도 포함되어 있습니다. 마지막으로, 사용자가 특정 작업에 가장 적합한 접근 방식을 선택하는 데 도움을 주기 위한 실용적인 지침을 제시하고, 미해결 문제와 향후 연구 방향을 제안합니다.

- **Technical Details**: STI는 관계형 테이블을 입력으로 받아, 지식 그래프(Knowledge Graph, KG)와의 매칭을 통해 의미론적으로 주석이 달린 테이블을 생성하는 과정입니다. STI 과정은 주로 Cell-Entity Annotation (CEA), Column-Type Annotation (CTA), Columns-Property Annotation (CPA)의 세 가지 주요 태스크로 나뉘며, 이는 각각 테이블의 열과 셀에 KG의 개념 및 속성을 연결합니다. STI는 데이터의 의미를 명확히 하여 다양한 내려받기(application) 작업에 활용할 수 있는 지식을 제공합니다.

- **Performance Highlights**: STI는 AI 연구 및 응용에서 중요한 역할을 하며, CEA, CTA, CPA와 같은 여러 태스크는 탭 데이터 이해의 일환으로 간주됩니다. 이를 통해 지식 기반의 구축 및 확장은 물론, 탭 데이터의 풍부한 정보를 제공하여 데이터 분석을 위한 후속 응용 프로그램을 지원할 수 있습니다. 따라서 STI는 지식 그래프와 같은 구조에서 연결된 다양한 개체들 간의 관계를 조직화하는 데 기여하는 핵심 연구 분야로 부각됩니다.



### ACING: Actor-Critic for Instruction Learning in Black-Box Large Language Models (https://arxiv.org/abs/2411.12736)
- **What's New**: 이 논문에서는 블랙박스 대형 언어 모델(LLM)에서 프롬프트 최적화를 자동화하기 위해 ACING이라는 접근 방식을 제안합니다. ACING은 연속 행동 강화 학습(continuous-action Reinforcement Learning) 문제로 프레이밍되며, 비미분 보상 신호에서 학습하여 프롬프트를 최적화합니다.

- **Technical Details**: ACING은 액터-비평자(actor-critic) 기반 방법을 활용하며, 이는 제한된 API 호출을 고려하여 효율적인 탐색과 활용을 동시에 수행합니다. 내부 파라미터를 알 수 없는 블랙박스 LLM의 경우에도 잘 작동하며, 기존의 방법과 비교하여 더 나은 성능을 발휘합니다.

- **Performance Highlights**: 30개 작업에 대한 ChatGPT의 프롬프트를 최적화한 결과, ACING은 기준 방법에 비해 일관되게 개선된 결과를 보여주었습니다. ACING은 평균 10%의 점수 향상을 이뤘으며, 인간 전문가의 지침을 초과하는 최대 39%의 개선을 달성했습니다.



### Benchmarking Positional Encodings for GNNs and Graph Transformers (https://arxiv.org/abs/2411.12732)
- **What's New**: 이 논문에서는 Positional Encoding (PE)의 효용성을 평가하기 위한 포괄적인 벤치마크를 제시하여 Graph Neural Networks (GNNs)와 Graph Transformers (GTs)의 다양한 조합을 비교하고 분석하고 있다. 기존 모델에 대한 테스트와 함께, PE가 성능에 미치는 영향을 보다 명확히 이해하기 위한 목적이다. 특히, 메세지 패싱(messaging-passing) 구조를 갖춘 GNN과 GTs 간의 이론적 연결성을 강화하는 데 초점을 두고 있으며, 이를 통해 새로운 PE 조합이 기존 방법을 초월할 수 있는지를 알아보고 있다.

- **Technical Details**: 논문은 Laplacian, Random Walk 및 기타 방법을 포함한 세 가지 주요 카테고리로 PE를 분류하고, 이를 통해 그래프 기반 모델의 기능을 향상시키기 위한 다양한 메커니즘을 탐구하고 있다. 또한, MPNN과 GT의 관계를 정리하고, 특정 조건 하에서 두 모델이 동일한 표현 가능성을 가질 수 있음을 증명할 예정이다. 이론 분석을 통해 PE와 MPNN 및 GT 간의 연관성을 확립함으로써, 실질적인 성능 개선을 이끌기 위한 새로운 방법론을 제시하고 있다.

- **Performance Highlights**: 실험 결과, 새로운 PE 조합이 기존의 성과를 초과할 수 있음을 보여주었으며, 특히 다양한 벤치마크 데이터셋에서 해당 성능이 입증되었다. GTs와 MPNNs의 다양한 조합을 통해 최첨단 성능을 실현할 수 있음을 명확히 하며, 코드를 공개하여 후속 연구를 지원하고 있다. 이를 통해 향후 PE와 아키텍처의 최적 결합에 대한 탐색을 용이하게 하는 통합 평가 프레임워크를 제공하고 있다.



### Heuristic-Free Multi-Teacher Learning (https://arxiv.org/abs/2411.12724)
- **What's New**: 최근 연구에서 Teacher2Task라는 새로운 멀티-교사 학습 프레임워크가 소개되었습니다. 이 프레임워크는 기존의 수작업 집계 휴리스틱을 없애고, 각 교사만의 입력 토큰을 도입함으로써 데이터의 혼란을 줄여줍니다. Teacher2Task는 각 교사의 스타일을 반영하는 N개의 보조 과제와 진짜 레이블에 중점을 둔 1개의 주요 과제로 구성된 총 N+1개의 과제로 훈련 프로세스를 재구성합니다.

- **Technical Details**: 기존의 멀티-교사 학습 방법들은 주로 예측 결과를 단순히 집계하여 최종 레이블로 사용하는 접근법을 취하지만, Teacher2Task는 이를 확장하여 각 교사의 신뢰도 점수를 예측하는 보조 작업을 생성합니다. 각 새로운 교사는 단순히 새로운 보조 작업을 도입하는 방식으로 시스템에 통합됩니다. 또한, 교사 식별자를 입력에 포함시킴으로써, 노이즈 데이터를 감소시키고 교사 간의 혼란된 주석을 해결할 수 있는 강점을 가지고 있습니다.

- **Performance Highlights**: 실험을 통해 Teacher2Task는 다양한 아키텍처와 작업에 걸쳐 성능을 개선하고 강건성을 보여주었습니다. 특히, 이 방법은 각 교사의 기여도를 효율적으로 활용함으로써, 학습 데이터의 레이블 효율성을 높이고, 교사의 신뢰도 점수를 데이터로 활용해 전반적인 성능을 향상시키는 데 성공했습니다. 결과적으로, Teacher2Task는 전통적인 집계 방식에 비해 더 다양한 데이터 소스를 효과적으로 조합할 수 있는 방법을 제시하고 있습니다.



### CATCH: Complementary Adaptive Token-level Contrastive Decoding to Mitigate Hallucinations in LVLMs (https://arxiv.org/abs/2411.12713)
- **What's New**: 이 논문에서는 대규모 비전-언어 모델(LVLM)에서 발생하는 환각(hallucination) 문제를 해결하기 위해 새로운 방법 CATCH(Complementary Adaptive Token-level Contrastive Decoding to Mitigate Hallucinations)를 제안합니다. CATCH는 정보 병목 정보이론(Information Bottleneck theory)을 기반으로 하여 시각 정보를 분리하는 Complementary Visual Decoupling(CVD)과 환각 탐지를 위한 Non-Visual Screening(NVS), 그리고 환각 완화를 위한 Adaptive Token-level Contrastive Decoding(ATCD)을 도입합니다. 이 방법은 다양한 시각적 질문 응답(visual question-answering) 작업에 적용 가능하며 추가 학습 없이도 새로운 작업에 강력하게 일반화 할 수 있는 가능성을 열어줍니다.

- **Technical Details**: CATCH는 시각 정보가 과도하게 유입되는 것에 대한 시각적 결함(visual defect)의 문제를 해결하기 위해 두 가지 이미지와 잔여 이미지를 구분하여 시각 정보를 안정적인 형상(decoupled visual representation)으로 표현합니다. 이 과정에서는 Segmentation 모델인 SAM(Segment Anything Model)을 사용하여 원본 시각 입력을 여러 수준으로 분리합니다. 또한, 모델에서 불필요한 시각적 특징을 축소해 언어 이전(linguistic priors)에 의한 극단적 의존성을 줄입니다. 이 approach는 출력 분포 간의 Jensen-Shannon Divergence(JSD)를 이용하여 비시각적 입력에 대한 정보와 비교하여 필수 시각적 정보의 밀도를 증가시키고 불확실성을 감소시킵니다.

- **Performance Highlights**: CATCH의 제안된 방법은 시각적 정보의 밀도를 증가시키고, 시각적 결함으로 발생하는 다양한 환각 문제를 완화하는 데 효과적임을 입증합니다. 실험 결과, 모델이 시각 입력이 obscured된 경우에도 언어적 측면에서 높은 정확도를 유지하며 비교적 일치하는 출력 분포를 생성하는 것을 확인했습니다. 이는 시각적 정보가 불필요한 정보로 인해 감소되지 않고, 핵심 시각적 특징이 지속적으로 유지됨을 나타내며, 고위험 도메인에서도 안정적인 성능을 보여줍니다.



### Enhancing Multi-Class Disease Classification: Neoplasms, Cardiovascular, Nervous System, and Digestive Disorders Using Advanced LLMs (https://arxiv.org/abs/2411.12712)
Comments:
          7 Pages, 4 tables and 11 figures. Under review in a IEEE conference

- **What's New**: 이 연구에서는 의료 관련 사전 훈련된 언어 모델을 사용하여 다중 클래스 질병 분류를 개선하는 방법을 탐구했습니다. 비암 질환을 제외하고 4가지 특정 질병을 살펴보았으며, BioBERT, XLNet, BERT 그리고 새로운 모델인 Last-BERT를 평가했습니다. 연구 결과, BioBERT는 의료 텍스트 분류에서 97%의 정확도를 기록하며 우수한 성과를 보였고, XLNet 또한 96%의 정확도로 인상적인 결과를 보여주었습니다.

- **Technical Details**: 연구에 사용된 Medical-Abstracts-TC-Corpus는 5가지 질병(신생물, 소화기계 질환, 신경계 질환, 심혈관 질환, 일반 병리학적 상태)에 대한 데이터를 포함하고 있으며 총 14,438개의 기록이 있습니다. 일반 병리학적 상태는 분석에서 제외되었고, 데이터는 훈련 세트와 테스트 세트로 80-20 비율로 나누어 사용되었습니다. BioBERT 및 XLNet과 같은 모델은 Hugging Face 라이브러리를 통해 로드되고 데이터의 시퀀스 분류를 위해 토큰화 절차를 수행했습니다.

- **Performance Highlights**: BioBERT는 97%의 정확도로 모든 모델 중에서 가장 뛰어난 결과를 보여주었고, Last-BERT는 BERT의 89.33%에 근접한 87.10%의 정확도를 기록했습니다. 이러한 결과는 의료 텍스트 분류에 있어서 BioBERT 같은 특화된 모델뿐만 아니라 XLNet과 같은 광범위한 솔루션의 중요성을 입증합니다. 연구 결과는 공공 건강 감시 및 환자 관리에 대한 의미를 제공하며, 향후 의료 분야에서의 NLP 모델의 유용성을 나타냅니다.



### When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations (https://arxiv.org/abs/2411.12701)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)에서 발생할 수 있는 백도어 공격(backdoor attack)에 대한 새로운 접근 방식을 제시합니다. 특히, 자연어 설명(natural language explanations)의 관점을 통해 LLM의 결정 과정을 분석하며, 깨끗한 샘플과 오염된 샘플 간의 비교를 수행합니다. 이 연구는 LLM이 어떻게 백도어 트리거의 영향을 받아 설명을 생성하는지에 대한 깊은 통찰을 제공합니다.

- **Technical Details**: LLM은 자연어 처리(NLP) 작업에서 뛰어난 성능을 보이나, 백도어 공격에 취약한 것으로 밝혀졌습니다. 주요 실험에서는 LLaMA 모델에 다양한 백도어 기법을 적용하여 여러 작업을 수행하였고, 오염된 데이터에 대해 더 일관된 설명을 생성하는 경향이 나타났습니다. 구체적으로, 오염된 샘플의 설명 토큰은 모델의 마지막 몇 개 트랜스포머 층에서만 나타나는 반면, 깨끗한 샘플의 경우는 훨씬 빨리 나타납니다.

- **Performance Highlights**: 실험 결과, 백도어가 있는 모델은 깨끗한 데이터에 비해 오염된 데이터의 설명 품질이 낮았으며, 오염된 데이터에 대해 더 일관된 설명을 생성하는 경향이 있음을 보여주었습니다. 이를 통해 LLM 내의 백도어 공격 메커니즘에 대한 이해를 심화시키고, 설명 가능성 기법을 통해 이러한 취약점을 감지할 수 있는 프레임워크를 제안합니다.



### Attribute Inference Attacks for Federated Regression Tasks (https://arxiv.org/abs/2411.12697)
- **What's New**: 이번 논문에서는 Federated Learning (FL) 환경에서 회귀 작업에 대한 새로운 모델 기반 속성 추론 공격(attribute inference attack, AIA)을 제안하고 있습니다. 이전 연구들은 주로 분류 작업(classification tasks)에 초점을 맞추었지만, 회귀 작업(regression tasks)에 대한 연구는 부족했습니다. 이를 해결하기 위해, 우리는 새로운 두 단계의 모델 기반 AIA를 소개하며, 이를 통해 공격자의 전략이 효과적임을 보였습니다.

- **Technical Details**: 논문에서는 공격자가 메시지를 도청하거나 훈련 과정에 직접 개입하여 최적의 로컬 모델을 대략적으로 재구성하는 방법을 제안합니다. 우리의 모델 기반 AIA는 기존의 그래디언트 기반 AIA보다 효과적인 성능을 보이며, 특히 이질적인 클라이언트 데이터셋에서 높은 재구성 정확도를 기록하고 있습니다. 또한, 회귀 작업의 성능을 저해하는 요소들에 대한 분석과 함께, 관련 수학적 이론을 통해 모델 기반 AIA의 정확도를 낮추는 경계를 설정하였습니다.

- **Performance Highlights**: 실험 결과, 우리의 제안된 모델 기반 AIA는 회귀 작업에서 강력한 공격 성능을 발휘하며, 기존의 그래디언트 기반 공격 방식보다 최대 30% 높은 정확도를 기록했습니다. 이는 FL의 회귀 작업에서 프라이버시 유출을 정량적으로 측정하기 위한 신뢰할 수 있는 방법으로 자리잡을 수 있음을 시사합니다. 이러한 성과는 FL 환경에서의 공격 가능성을 더욱 드러내므로, 향후 관련 연구에 중요한 기반이 될 것입니다.



### Enhanced Sign Language Translation between American Sign Language (ASL) and Indian Sign Language (ISL) Using LLMs (https://arxiv.org/abs/2411.12685)
- **What's New**: 이번 연구는 미국 수화(ASL) 사용자와 인도 수화(ISL) 사용자 간의 소통을 위한 자동 번역 시스템을 제시합니다. 이 시스템은 대규모 언어 모델(LLM)을 활용하여 ASL에서 ISL로의 실시간 번역을 가능하게 하여 접근성을 높이는 것을 목표로 하고 있습니다. 기초가 되는 기술로는 랜덤 포레스트 분류기(Random Forest Classifier)와 자연어 처리(NLP) 기법을 결합한 혁신적인 프레임워크가 사용됩니다.

- **Technical Details**: 제안된 방법론은 세 가지 주요 단계로 구성됩니다: 수화 인식, 텍스트 교정, 그리고 비디오 합성입니다. 첫 번째 단계에서는 랜덤 포레스트 분류기와 CNN(Convolutional Neural Network)을 결합한 하이브리드 모델을 사용하여 수화의 특징을 효과적으로 추출하고 분류합니다. 이후 단계에서는 인식된 텍스트를 LLM을 활용하여 교정하고, 마지막으로 ISL 제스처를 자연스럽게 생성하기 위해 비디오 합성 단계에서 모션 스무딩 기법을 적용합니다.

- **Performance Highlights**: 이 프레임워크는 ASL과 ISL 간의 언어적 차이를 극복하는 것을 목표로 하며, 실시간 프로세싱과 문화적 맥락화의 통합으로 모두가 쉽게 접근할 수 있는 소통을 돕습니다. RIFE-Net을 이용한 제스처 합성 과정은 부드럽고 자연스러운 제스처 표현을 가능하게 하며, 이는 다양한 수화 방언을 지원할 수 있는 보편적인 상호운용성을 위한 첫 걸음이 될 것입니다. 따라서 연구는 청각 장애인 커뮤니티 간의 inclusivity와 교류를 촉진할 수 있는 기술적인 진전을 나타냅니다.



### AI Guided Early Screening of Cervical Cancer (https://arxiv.org/abs/2411.12681)
- **What's New**: 이 연구는 의료 이미징 데이터셋의 전처리 및 개선을 통해 신뢰할 수 있는 이상 탐지(machine learning models for anomaly detection) 모델 생성을 지원하는 데 중점을 두고 있습니다. 데이터셋은 정상(normal)과 비정상(abnormal) 두 가지 분류로 나뉘며, 추가적인 노이즈 변동도 포함되어 있습니다. 중앙 자르기를 통해 사진의 질을 개선하기 위한 불필요한 아티팩트(artifacts)를 제거하고, 추가 전처리 과정으로 밝기(brightness)와 대비(contrast)를 조정했습니다.

- **Technical Details**: 이미지 데이터셋은 여러 하위 세트를 두 가지 기본 범주인 정상(normal)과 병리(pathological)로 체계적으로 통합하여 분류 작업을 용이하게 했습니다. 고급 이미지 전처리 기술로는 대비 향상(contrast enhancement)과 실시간 증강(real-time augmentation) 기술이 포함되어 있으며, 이를 통해 회전(rotations), 확대(zooms), 밝기 수정(brightness modifications) 등이 가능합니다. 모델 평가를 위해 데이터는 훈련(training) 및 테스트(testing) 서브셋으로 분할되었습니다.

- **Performance Highlights**: 이 프로젝트는 의료 이상 탐지를 위한 정확하고 효과적인 기계 학습 모델 생성을 목표로 고품질 입력 데이터(input data)를 보장합니다. 프로젝트 파이프라인의 유연하고 확장 가능한 설계 덕분에 대규모 임상 의사결정 지원 시스템(clinical decision-support systems)과 쉽게 통합할 수 있는 장점을 갖추고 있습니다.



### Deep Learning-Driven Heat Map Analysis for Evaluating thickness of Wounded Skin Layers (https://arxiv.org/abs/2411.12678)
- **What's New**: 이 논문은 상처 치유 관행 개선을 위한 새로운 비침습적(non-invasive) 방법을 제안합니다. 기존의 깊이 측정 방법이 침습적이고 구체성이 떨어지는 데 비해, 이 방법은 심층 학습(deep learning) 기술을 활용하여 피부 층을 분류하고 열 지도 분석을 통해 상처 깊이를 측정합니다. 약 200개의 라벨이 지정된 이미지 세트를 사용하여 흉터, 상처 및 건강한 피부 등의 다섯 가지 클래스를 구별합니다.

- **Technical Details**: 제안된 방법은 Roboflow 소프트웨어에서 스트라텀 코르네텀(stratum corneum), 표피(epidermis), 진피(dermis)와 같은 주요 층을 주석 처리한 이미지를 사용하여 진행됩니다. 초기 단계에서는 VGG16을 사용하여 조직 층의 가시성을 향상시킨 후, 이를 통해 생성된 이미지를 ResNet18 모델로 훈련했습니다. 훈련 결과 97.67%의 높은 정확도를 달성했으며, EfficientNet 및 ResNet18 모델의 성능 비교를 통해 두 모델 모두 약 95.35%의 정확도를 기록했습니다.

- **Performance Highlights**: 또한 효율적인 모델 구성을 결정하기 위해 두 모델의 하이퍼 파라미터 튜닝도 실시하였습니다. 학습률에 따라 정확도가 큰 변동을 보였고, EfficientNet과 ResNet18 모두 0.0001의 학습률에서 최대 95.35%의 정확도를 달성하였습니다. 이러한 결과는 모델이 실제 시간(non-invasive)으로 상처를 평가하는 데 적용 가능하다는 것을 나타내며, 임상 진단 및 치료 계획 개선에 큰 잠재력을 지니고 있음을 의미합니다.



### PoM: Efficient Image and Video Generation with the Polynomial Mixer (https://arxiv.org/abs/2411.12663)
- **What's New**: 최근 Diffusion 모델에서 Multi-Head Attention (MHA)을 대체할 새로운 메커니즘인 Polynomial Mixer (PoM)를 소개합니다. PoM은 시퀀스의 길이에 대해 선형 복잡성을 가지며, MHA의 품질을 손상시키지 않고도 전체 시퀀스를 명시적인 상태로 인코딩합니다. 이 메커니즘은 대규모 생성 모델의 효율적인 스케일링을 가능하게 하며, 특히 고해상도 이미지와 비디오 생성을 더 용이하게 만들어 줍니다.

- **Technical Details**: PoM은 State-Space Models (SSMs)와 유사한 선형 복잡성을 고려하지만, MHA와 같이 모든 쌍 정보에 대한 처리를 가능하게 합니다. 이를 통해 DiT 아키텍처에 적용 시 더 높은 해상도에서 MHA를 사용하는 모델을 학습하는 것보다 PoM을 사용하는 모델이 훈련 비용이 월등히 낮아질 수 있음을 보여줍니다. 또한 PoM은 일반적인 시퀀스-투-시퀀스 (sequence-to-sequence) 근사기를 제공함으로써 다양한 애플리케이션에서의 적용 가능성을 높입니다.

- **Performance Highlights**: PoM을 적용한 이미지 생성 모델은 유사한 품질의 샘플을 생성하면서도 더 적은 계산 자원으로 고해상도의 이미지를 처리할 수 있는 능력을 보여줍니다. 더불어 PoM을 활용한 비디오 생성 모델은 매 프레임에 대해 일정한 처리 비용을 유지하면서도 시각적 품질을 유지할 수 있습니다. 이로 인해 PoM은 향후 고해상도 이미지 및 긴 비디오 생성을 위한 기본적인 메커니즘으로 자리 잡을 것으로 기대됩니다.



### Optimizing Airline Reservation Systems with Edge-Enabled Microservices: A Framework for Real-Time Data Processing and Enhanced User Responsiveness (https://arxiv.org/abs/2411.12650)
Comments:
          22 pages, 11 figures

- **What's New**: 항공사 예약 시스템의 복잡성이 증가함에 따라, 본 논문에서는 새로운 접근 방식을 채택하여 신속하고 효율적인 예약 시스템 개발을 위한 스마트 솔루션을 제시합니다. 특히, 기존의 중앙 집중식 아키텍처의 단점을 해결하기 위해 에지 컴퓨팅 마이크로서비스를 구현하는 개념적 프레임워크를 상세히 설명합니다. 이를 통해 사용자의 가까운 곳에서 좌석 재고 확인 및 예약 과정 등을 수행할 수 있어 시스템의 반응 시간을 단축시킬 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 Kubernetes에 의해 조정되는 배포된 분산 컴퓨팅 마이크로서비스와 실시간 메시지 처리 시스템인 Kafka를 기반으로 하여 탄력적 확장을 가능하게 합니다. 또한, 자원 모니터링 및 관리를 위한 Prometheus와 Grafana와 같은 운영 구성 요소가 포함되어 있어 모든 운영 프로세스가 최적화됩니다. 이러한 기술적 요소들은 저지연(low latency), 고처리량(high throughput) 및 향상된 사용자 경험을 목표로 하고 있습니다.

- **Performance Highlights**: 본 연구는 항공 산업의 서비스 제공 방식에 변화를 주어 고객 만족도를 향상시킴과 동시에 설치 비용이 저렴하고 인공지능(artificial intelligence) 및 사물인터넷(internet of things) 통합 시스템과 같은 기술 변화에 효과적으로 대응할 수 있는 인프라를 제공합니다. 새로운 기술과 현대적인 분산 및 실시간 중심 시스템에 대한 수요를 반영하며, 향후 케이스 적용 및 테스트를 위한 기반을 마련합니다. 따라서, 제안하는 아키텍처는 기존의 항공사 예약 시스템이 겪고 있는 문제를 해결하기 위한 시장 친화적이고 확장 가능한 솔루션을 제공합니다.



### CodeXEmbed: A Generalist Embedding Model Family for Multiligual and Multi-task Code Retrieva (https://arxiv.org/abs/2411.12644)
- **What's New**: 코드 검색(code retrieval) 분야는 아직 충분히 탐구되지 않은 영역입니다. 기존의 자연어 처리(natural language processing) 모델은 코드의 다양성과 특정 도메인에서의 과제를 효과적으로 처리하지 못하는 경향이 있습니다. 이를 해결하기 위해, 400M에서 7B 파라미터에 이르는 대규모 코드 임베딩 모델인 CodeXEmbed를 소개하며, 이는 다양한 프로그래밍 언어를 통합하는 훈련 파이프라인을 갖추고 있습니다.

- **Technical Details**: CodeXEmbed는 코드와 텍스트를 위한 개방형 임베딩 모델로, 400M, 2B, 7B 파라미터 모델이 존재합니다. 이 모델은 12개 프로그래밍 언어와 5개의 코드 검색 카테고리를 처리할 수 있는 일반화 가능한 훈련 프레임워크를 제공합니다. 다양한 프로그래밍 언어와 코드 관련 작업을 검색 작업으로 변환하여, 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 7B 모델은 코드를 검색하는 데 있어 새로운 최첨단(state-of-the-art) 성능을 기록하며, CoIR 벤치마크에서 20% 이상 향상된 결과를 보였습니다. 또한, CodeXEmbed는 텍스트 검색에서도 경쟁력 있는 성능을 보이며, 코드 관련 작업에서의 Retrieval-Augmented Generation(RAG) 성능을 크게 향상시킵니다.



### DLBacktrace: A Model Agnostic Explainability for any Deep Learning Models (https://arxiv.org/abs/2411.12643)
- **What's New**: 이번 연구에서는 DLBacktrace라는 새로운 기술을 소개합니다. 이는 다양한 신경망 구조에서 모델 결정을 투명하게 하는 방법으로, Multi Layer Perceptron (MLP), Convolutional Neural Network (CNN), Large Language Models (LLM) 등 여러 도메인에서 효과적으로 작동합니다. DLBacktrace는 해석 가능성 (interpretability)의 필요성을 강조하며, AI 시스템의 신뢰를 구축하고 책임성을 보장하는 데 기여합니다.

- **Technical Details**: DLBacktrace는 모델-비의존적 방법으로, 출력에서 입력으로 관련성을 추적하여 각각의 계층에서의 중요도 점수를 부여합니다. 이 방법은 PyTorch와 TensorFlow에서 구현된 다양한 모델 아키텍처와 호환되며, 정량적인 메트릭을 사용하여 기존 해석 가능성 방법들과 비교한 벤치마킹 결과를 제시합니다. 심층 학습에서 모델의 투명성을 개선하기 위해 설계된 이 기술은 다양한 데이터 형식에 적용할 수 있습니다.

- **Performance Highlights**: DLBacktrace는 LIME, SHAP, Grad-CAM 등과 같은 기존 해석 가능성 방법들과 비교될 때 독창적이고 신뢰할 수 있는 해석을 제공합니다.  전반적인 모델 설계 및 데이터 타입을 아우르는 두 가지 분석 방식 (지역적 및 전역적 해석)을 통해 모델의 명확성을 높입니다. 또한 오픈 소스로 제공되어 다양한 연구 및 산업 응용에서 적극 활용될 수 있는 가능성을 가지고 있습니다.



### Instant Policy: In-Context Imitation Learning via Graph Diffusion (https://arxiv.org/abs/2411.12633)
Comments:
          Code and videos are available on our project webpage at this https URL

- **What's New**: 본 논문에서 우리는 In-Context Imitation Learning (ICIL)을 활용한 새로운 방법론인 Instant Policy를 소개합니다. Instant Policy는 단 1~2개의 데모로 새로운 작업을 신속하게 학습할 수 있도록 하며, 기존의 Behavioral Cloning (BC) 방법에 비해 시간 효율적입니다. 이 접근법은 그래프 생성 문제로 모델링하여 데모와 관찰을 구조적으로 해석함으로써 로봇 행동 예측의 효율성을 높입니다.

- **Technical Details**: Instant Policy는 그래프 기반 표현을 활용하여 데모, 현재 포인트 클라우드 관찰, 로봇의 행동을 통합한 구조를 형성합니다. ICIL을 확산 기반 그래프 생성 문제로 공식화하여, 복잡한 데이터의 구조적 학습을 가능하게 합니다. 또한, 절차적으로 생성된 의사 데모(pseudo-demonstration)를 통해 무한한 학습 데이터를 생성할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험에서 Instant Policy는 다양한 일상 작업을 신속하게 배울 수 있는 능력을 보여 주었습니다. 기존의 기준 모델들보다 높은 작업 성공률을 달성하며, 테스트 시 제공되지 않은 객체의 기하학에 대한 일반화 능력 또한 관찰되었습니다. Instant Policy는 또한 인간 손의 데모에서 로봇 정책으로의 크로스 엠바디먼트 전이와 언어 정의 작업에 대한 제로샷 전이를 가능하게 합니다.



### Estimating Dark Matter Halo Masses in Simulated Galaxy Clusters with Graph Neural Networks (https://arxiv.org/abs/2411.12629)
Comments:
          9 pages, 4 figures, accepted at the NeurIPS ML4PS 2024 workshop

- **What's New**: 이 연구에서는 은하의 별질량(M_{*})를 기반으로 암흑물질 헤일로의 질량(M_{halo})을 예측하기 위해 그래프 신경망(Graph Neural Network, GNN) 모델을 도입하였습니다. 기존의 랜덤 포레스트(random forests)와 같은 방법들과 달리, GNN은 은하 군집 내 이웃 은하 간의 공간 및 운동학적 관계를 활용하여 정보가 풍부한 서브구조(substructure)를 포착합니다. 이러한 접근법은 TNG-Cluster 데이터셋으로 훈련된 후 TNG300 시뮬레이션에서 독립적으로 테스트되었으며, 예측 성능에서 우월한 결과를 보였습니다.

- **Technical Details**: 이 연구는 IllustrisTNG 시뮬레이션 데이터, 특히 TNG-Cluster를 활용하였습니다. GNN 모델은 M_{*}을 노드 특성으로 사용하며, 3 Mpc 거리 이내의 은하 쌍 간의 연결을 통해 서로의 상호작용을 학습합니다. 이 네트워크는 8개의 비공유 층과 3개의 연속 층으로 구성되어 있으며, 각 층은 16개의 은닉 채널을 가진 두 개의 MLP를 사용하여 최종적으로 암흑물질 헤일로 질량(M_{halo})과 그 로그 분산을 예측합니다.

- **Performance Highlights**: GNN 모델은 훈련, 검증 및 독립된 테스트 세트에서 유의미한 성능 향상을 보여주었습니다. 다양한 평가 지표를 사용하여 MSE를 최소화하는 간단한 모델과 비교해 최고의 결과를 기록하였으며, 그 중 RMSE, MAE, R² 지표가 포함됩니다. 본 연구 결과는 GNN 모델이 복잡한 비선형 관계를 학습하는 데 효과적임을 입증하며, 향후 다른 시뮬레이션이나 관측 데이터셋으로의 확장을 통해 GNN의 일반화 능력을 검증할 계획입니다.



### STREAM: A Universal State-Space Model for Sparse Geometric Data (https://arxiv.org/abs/2411.12603)
- **What's New**: 이 논문에서는 기하학적 구조를 상태 공간 모델(SSM)의 매개변수화에 명시적으로 인코딩하는 방안을 제안합니다. 이를 통해 기존의 sequence 모델들과 비교하여 불규칙한 단계 크기를 가진 희소 기하학 데이터 처리에 효율성을 제공합니다. 이로 인해 새로운 모델인 STREAM은 이벤트 기반 비전 및 포인트 클라우드 분석에서 경쟁력 있는 성능을 발휘합니다.

- **Technical Details**: STREAM 모델은 CUDA 커널의 수정된 버전을 사용하여 현대 하드웨어에 효율적으로 희소 기하학 데이터를 매핑합니다. 본 연구에서는 상대 좌표의 차이를 단계 크기로 주입하여 기하학적 연산을 O(N) 단계로 수행하도록 설계되었습니다. 이는 모든 N포인트 간의 상호작용을 계산하는 데 필요한 단계를, 기존 방식보다 획기적으로 줄여 줍니다.

- **Performance Highlights**: STREAM 모델은 ModelNet40 및 ScanObjectNN 데이터셋에서 PointMamba 기준보다 최대 2.84% 향상된 성능을 보여주었습니다. 더불어 DVS128 Gesture 데이터셋에서 모든 11개 클래스에 대해 100%의 테스트 정확도를 달성하였습니다. 이는 이벤트 기반 비전 분야에서 최초의 결과로, 희소 기하학 데이터 처리에서의 강력한 유도 편향(inductive bias)을 입증합니다.



### Provable unlearning in topic modeling and downstream tasks (https://arxiv.org/abs/2411.12600)
- **What's New**: 이 논문에서는 기계 모델의 사전 학습(pre-training)과 미세 조정(fine-tuning) 과정에서 '기계 언러닝(machin unlearning)'을 위한 이론적 보장을 처음으로 제공하고 있습니다. 특히, 주제 모델(topic model)과 같은 간단한 언어 모델을 통해 탐색, 분류와 같은 다운스트림 작업을 해결하는 방법을 다룹니다. 이 연구는 기존의 사전 학습 모델에서 데이터를 삭제할 수 있는 효과적인 알고리즘을 설계하여, 모델 성능에 큰 영향을 미치지 않으면서도 훈련 데이터를 성공적으로 삭제할 수 있는 방법을 제시합니다.

- **Technical Details**: 해당 논문은 주제 모델에 대한 언러닝 알고리즘을 설계하며, 원래 데이터 세트의 크기와 무관하게 계산 오버헤드를 줄이는 효율적인 접근 방식을 제공합니다. 연구는 또한 모델의 삭제 용량(deletion capacity)을 정량화하여, 성능 저하 없이 삭제할 수 있는 예제의 수를 분석합니다. 특히, 선형 헤드(linear head)를 통해 주제 모델을 미세 조정한 후에도 언러닝을 수행할 수 있는 알고리즘을 설계하여, 특정 작업에 대해 최적화된 모델에서 사전 학습 데이터를 더 쉽게 삭제할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구의 알고리즘은 기계 언러닝의 기준(gold standard) 충족이 가능하도록 설계되었으며, 기존의 방법들보다 낮은 비용으로 목표를 달성할 수 있습니다. 또한, 데이터 삭제 요청이 있을 경우에도 모델 성능이 크게 저하되지 않음을 입증합니다. 이러한 점에서 사전 학습 및 미세 조정 환경에서 효과적인 기계 언러닝 알고리즘의 개발이 이루어진 점이 두드러집니다.



### AdaCM$^2$: On Understanding Extremely Long-Term Video with Adaptive Cross-Modality Memory Reduction (https://arxiv.org/abs/2411.12593)
- **What's New**: 이 논문에서는 AdaCM2라는 새로운 방법을 제안하며, 이는 비디오-텍스트 정렬을 수행하기 위해 처음으로 적응형 크로스 모달리티 메모리 축소 방식을 도입합니다. 기존 연구는 주로 짧은 비디오 처리에 한정되었지만, AdaCM2는 복잡한 질문-응답 작업을 효과적으로 처리할 수 있는 긴 비디오 이해를 목표로 합니다. 이 프레임워크는 여러 데이터셋에서 차세대 성과를 달성하며 메모리 사용량을 대폭 줄입니다.

- **Technical Details**: AdaCM2는 개별 레이어에서 텍스트 쿼리와 가장 관련이 높은 중요 시각 토큰을 적응적으로 보존하는 방식으로 메모리 소비를 줄입니다. 이를 위해 Q-Former와 같은 고정된 비주얼 인코더를 사용하여 비디오 프레임에서 시각 표현을 추출한 후, AdaCM2 어텐션 메커니즘을 통해 프레임 단위로 쿼리를 학습합니다. 이 접근 방식은 다양한 레이어의 크로스 모달리티 상관관계에 따라 시각 토큰을 유연하게 조정할 수 있게 합니다.

- **Performance Highlights**: 다양한 비디오 이해 작업에 대한 광범위한 실험을 통해 AdaCM2는 LVU 데이터셋에서 여러 작업에 대해 4.5%의 정확도 향상을 달성하며, GPU 메모리 소비를 65%까지 줄였습니다. 특히 VQA(Visual Question Answering) 및 비디오 캡션 생성 작업에서 유망한 성과를 보였습니다. AdaCM2는 기존 BLIP 기반 모델의 성능을 플러그 앤 플레이 방식으로 향상시켜 긴 비디오 처리 능력을 향상시킵니다.



### Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating Visual Hallucination (https://arxiv.org/abs/2411.12591)
- **What's New**: 이 논문에서는 Visual Inference Chain (VIC)라는 새로운 프레임워크를 제안하여 멀티모달 대형 언어 모델(MLLMs)의 고유한 문제점, 즉 시각적 입력에 의한 환각(hallucination) 현상을 해결합니다. VIC는 텍스트 컨텍스트에 기반한 추론 체인을 구성하고 나서 시각적 입력을 도입하는 방식으로, 이를 통해 교차 모달 편향을 줄이고 멀티모달 추론의 정확성을 개선합니다. 이 연구는 기존의 'thinking while looking' 접근 방식의 한계를 극복하고 'thinking before looking'이라는 새로운 패러다임을 제안합니다.

- **Technical Details**: VIC 프레임워크는 MLLMs의 추론 과정을 개선하기 위해, 이전의 'thinking while looking' 패러다임 대신 'thinking before looking' 전략을 채택합니다. 이를 통해 모델은 시각적 요소와 상관없이 기존의 기억과 맥락 지식을 활용하여 추론을 선행하도록 유도합니다. 이 방식은 인간의 인지 패턴과 유사하며, 앞서 계획을 세우는 과정이 시각적 입출력의 영향력을 줄이는 데 효과적임을 보여줍니다.

- **Performance Highlights**: VIC의 도입으로, 다양한 비전 관련 작업에서 탁월한 성과를 확인할 수 있습니다. 예를 들어, Gemini 1.5 Pro 모델은 MMVP 벤치마크에서 31.74%의 성능 향상을 이루었고, GPT-4o 미니 모델은 16.59% 증가했습니다. 전반적으로, GPT 시리즈 모델의 평균 성능 향상은 8.02%에 달하는 반면, Gemini 시리즈 모델은 7.19%의 평균 개선을 보였습니다.



### ULTra: Unveiling Latent Token Interpretability in Transformer Based Understanding (https://arxiv.org/abs/2411.12589)
- **What's New**: 이 논문에서는 Transformer의 Latent Token (잠재 토큰) 표현을 해석하는 새로운 프레임워크를 제안합니다. 이 프레임워크를 통해 기존 모델을 추가적인 파인튜닝 없이도 제로샷(Zero-shot) 비지도형 의미 세분화(semantic segmentation)가 가능함을 입증했습니다. 이 방법은 Transformer 모델이 입력의 의미를 이해하는 본질적인 능력을 활용하며 기존의 전통적인 세분화 모델들을 능가하는 성능을 보여줍니다.

- **Technical Details**: Transformer 아키텍처와 Vision Transformers (ViTs)를 기반으로 한 이 연구는, Latent Tokens이 각각의 의미적 개념을 나타내도록 해석하는 방법을 제시합니다. 제안된 프레임워크는 친숙한 메커니즘 없이도 이미지 인지를 가능하게 하며, 이를 통해 하이퍼파라미터 조정 없이 이미지 세분화가 이루어질 수 있습니다. 우리는 또한 이 프레임워크가 대규모 언어 모델(LLMs)에서도 효과적임을 확인하여 텍스트 요약 작업에서의 적용 가능성을 검증했습니다.

- **Performance Highlights**: COCO-Stuff 데이터셋에서 67.2%의 정확도와 32.9%의 평균 교차 IoU(mIoU)를 기록하며, PASCAL VOC 데이터셋에서는 51.9%의 mIoU를 달성하여 기존 SOTA(State-of-the-Art) 성능을 초월했습니다. 이 연구는 기존의 비지도 학습 세분화 방법들보다 우수한 성능을 보이며, 많은 imstances가 필요한 기존 방법보다 더 효율적으로 동작합니다.



### Whisper Finetuning on Nepali Languag (https://arxiv.org/abs/2411.12587)
- **What's New**: 이 연구는 자동 음성 인식(Auto Speech Recognition, ASR) 분야에서 잘 알려지지 않은 언어인 네팔어를 위한 강력한 모델 개발의 도전을 다룹니다. OpenAI의 Whisper 모델에 대한 포괄적이고 일반화된 데이터셋을 구성하고 이를 정교하게 조정하여 네팔어 전사(speech-to-text) 정확도를 향상시키기 위한 노력입니다. 또한 다양한 억양, 방언 및 말하기 스타일을 반영한 자가 기록(custom) 데이터셋을 활용하였습니다.

- **Technical Details**: 연구팀은 공공에 제공되는 ASR 데이터셋과 자가 기록된 맞춤형 데이터셋을 결합하여, 화자의 연령, 성별, 감정 등 다양한 데이터 변화를 수집하였습니다. Whisper 모델의 다양한 크기에 대해 우리 맞춤형 데이터셋으로 미세 조정(fine-tuning)을 수행하여 단어 오류율(Word Error Rate, WER)을 상당히 감소시킵니다. 특히, 오디오 및 전사 수동 큐레이션(manual curation)을 통해 음성 신호의 품질이 향상되었습니다.

- **Performance Highlights**: 우리 연구 접근법은 Fleur 데이터셋으로 학습된 Whisper의 기준 모델 대비, 소형 모델에서 36.2%, 중형 모델에서 23.8%의 WER 개선을 달성했습니다. 데이터 증강(data augmentation)이 모델의 강인성을 향상시키는 데 중요한 역할을 강조하며, 이 연구는 정확한 ASR 시스템 개발을 위한 데이터셋의 품질, 변동성 및 증강의 중요성을 부각시킵니다.



### Leveraging MLLM Embeddings and Attribute Smoothing for Compositional Zero-Shot Learning (https://arxiv.org/abs/2411.12584)
- **What's New**: 이 논문에서는 Compositional Zero-Shot Learning (CZSL)을 위해 새로운 프레임워크인 TRIDENT를 제안합니다. 이 프레임워크는 Multimodal Large Language Model (MLLM) 임베딩과 attribute smoothing 기능을 기반으로 하여, 배경의 영향을 줄이고 다중 층의 특징을 활용하여 더 높은 일반화 능력을 제공합니다. 특히, 학습된 조합에 대한 과도한 자신감을 개선하기 위해 추가 속성을 생성하고, 이를 통해 모델이 다양한 속성을 배우도록 유도합니다.

- **Technical Details**: TRIDENT는 세 가지 주요 모듈로 구성되어 있습니다: 시각적 특징 추출, 속성-객체 분리, 그리고 특징 정렬입니다. 이 모듈들은 각각 Adaptive Aggregation 모듈을 사용하여 배경 잡음을 줄이고, 이미지 쌍의 공유 및 고유 특징을 분석하여 속성과 객체를 분리하며, 마지막 숨겨진 상태에서 MLLM의 임베딩을 사용하여 시각적 특징을 정렬합니다. 과도한 자신감 문제를 해결하기 위해, LLM을 활용하여 보조 속성을 생성하고 label smoothing을 통해 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: TRIDENT는 세 개의 CZSL 벤치마크에서 광범위한 실험을 통해 기존의 최신 성과를 초과달성했습니다. 연구 결과, 이 방법은 기존 접근 방식의 한계를 극복하며, 더 나은 속성-객체 인식과 조합 인식 능력을 보여줍니다. 추후 소스 코드도 발표할 예정이며, 이는 CZSL 연구에 있어 중요한 기여가 될 것으로 기대됩니다.



### Large Language Models for Combinatorial Optimization of Design Structure Matrix (https://arxiv.org/abs/2411.12571)
- **What's New**: 이 연구에서는 제안된 새로운 LLM(대형 언어 모델) 기반 프레임워크가 공학적 조합 최적화(combinatorial optimization) 문제에 적용될 수 있는 가능성을 탐구합니다. 특히, 이 방법은 네트워크 토폴로지(network topology)와 분야 지식(domain knowledge)을 통합하여 일반적인 CO 문제인 디자인 구조 매트릭스(Design Structure Matrix) 순서를 최적화합니다. 이 연구는 LLM이 제공하는 맥락적 지식을 통해 실제 공학 문제에 접근하는 새로운 패러다임을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 생성(generative) 및 추론(reasoning) 능력을 활용하여 최적화를 수행합니다. 초기 솔루션을 무작위로 샘플링하고, 평가자는 미리 정의된 기준에 따라 솔루션의 품질을 정량화합니다. 이 과정을 거쳐 LLM은 네트워크 정보와 분야 지식을 포함한 프롬프트(prompts)를 기반으로 새로운 솔루션을 생성합니다. 이 메커니즘은 피드백 루프(feedback loops)를 최소화하는 DSM 순서 최적화를 목표로 합니다.

- **Performance Highlights**: 실험 결과, LLM 기반 방법이 벤치마크 방법에 비해 빠른 수렴 속도와 높은 솔루션 품질을 달성함을 보여주었습니다. 특히 맥락적 분야 지식을 포함할 경우 성능이 더욱 향상되었으며, 다양한 DSM 사례를 통해 이러한 효과가 입증되었습니다. 이러한 결과는 LLM이 복잡한 실제 조합 최적화 문제를 해결하는 데 유용할 수 있음을 강조합니다.



### Topological Symmetry Enhanced Graph Convolution for Skeleton-Based Action Recognition (https://arxiv.org/abs/2411.12560)
- **What's New**: 이 논문에서는 인간의 동작 인식을 위한 새로운 접근 방식으로, Topological Symmetry Enhanced Graph Convolution (TSE-GC)와 Multi-Branch Deformable Temporal Convolution (MBDTC)를 제안합니다. TSE-GC는 사람 몸의 대칭성을 고려하여 다양한 채널 그룹에서 독특한 토폴로지 학습을 가능하게 하며, MBDTC는 더 유연한 수용역을 통해 시간 종속성을 더 잘 모델링할 수 있도록 돕습니다.

- **Technical Details**: TSE-GC는 주어진 샘플에 대한 스케일 마스크를 학습하고, 이를 통해 공유된 토폴로지를 여러 개의 별도 채널 그룹으로 복제하여 대칭적으로 토폴로지 학습을 촉진합니다. MBDTC는 샘플링 위치에 학습 가능한 오프셋을 적용하여, 더욱 유연한 수용역을 제공하고 시간 종속성을 더욱 효과적으로 표현하는 모델입니다.

- **Performance Highlights**: 제안된 TSE-GCN 모델은 NTU RGB+D, NTU RGB+D 120 및 NW-UCLA의 세 가지 대규모 데이터셋에서 최첨단 방법들과 비교하여 적은 파라미터로 경쟁력 있는 성능을 발휘합니다. 특히 NTU RGB+D 120 데이터셋에서 90.0% 및 91.1%의 정확도를 기록했으며, 1.1M 파라미터와 1.38 GFLOPS의 연산 성능을 요구합니다.



### Recall and Refine: A Simple but Effective Source-free Open-set Domain Adaptation Framework (https://arxiv.org/abs/2411.12558)
- **What's New**: 이번 논문은 Source-free Open-set Domain Adaptation (SF-OSDA) 문제를 해결하기 위한 새로운 프레임워크인 Recall and Refine (RRDA)를 제안합니다. 기존의 SF-OSDA 방법들은 라벨이 있는 소스 데이터를 사용할 수 없는 상황에서도 성과를 내야 하는데, RRDA는 미지의 클래스에 대해 효과적으로 특징을 학습하도록 설계되었습니다. 이 프레임워크는 두 단계로 구성되어 있으며, 첫 번째 단계에서 타겟 특징으로 생성된 합성 샘플을 사용하여 미지의 클래스를 인식하는 모델의 능력을 향상시킵니다.

- **Technical Details**: RRDA의 첫 번째 단계에서는 K+K′ 결정 경계를 갖는 새로운 타겟 분류기를 도입하여 소스 도메인에서의 K 클래스와 미지의 클래스에 해당하는 추가 K’ 클래스를 학습합니다. 합성 샘플은 알려진 클래스를 위한 낮은 엔트로피와 미지의 클래스를 위한 높은 엔트로피를 가지도록 최적화되어, 최종적으로 K’ 카테고리로 클러스터링됩니다. 두 번째 단계에서는 SHOT 및 AaD와 같은 기존의 비소스 도메인 적응 방법들을 프레임워크에 통합하여 전체 모델을 타겟 도메인에 적응시킵니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 광범위한 실험을 통해 RRDA가 기존의 SF-OSDA 및 OSDA 방법들보다 크게 우수함을 입증하였습니다. 성능 분석 결과 동작 속도와 유연성이 우수하며, K′ 값의 증가에 따라 성능이 향상되는 경향을 보였습니다. 실험 결과는 RRDA가 미지의 클래스에 대한 분류 정확도를 높이는 데 효과적이라는 것을 명확히 보여줍니다.



### Predicting Customer Satisfaction by Replicating the Survey Response Distribution (https://arxiv.org/abs/2411.12539)
- **What's New**: 본 논문에서는 고객 만족도(CSAT) 예측을 위한 새로운 접근 방식을 제안합니다. 기존에는 조사에 응답한 일부 고객의 데이터만 기반으로 평균 CSAT를 계산하여 발생하던 편향을 감소시키기 위한 모델이 개발되었습니다. 이 방법은 실제 생산 환경에서도 모든 통화에 대해 고객 만족도를 예측할 수 있도록 하여 더 정확한 성과 지표를 제공합니다.

- **Technical Details**: 연구에서는 자주 업데이트 되는 머신 러닝 모델의 클래스 비율 변화를 방지하기 위해 제어 메커니즘을 도입합니다. 이 메커니즘은 샘플링 노이즈에 의한 위험을 완화하고, ASR(Automated Speech Recognition) 데이터에서 고객 만족도의 분포를 정확하게 복제할 수 있도록 최적화된 결정을 제공합니다. 이 방법은 다중 클래스와 순서 분류 문제에서 사용할 수 있으며, 클래스 불균형을 개선하는데 기여합니다.

- **Performance Highlights**: 실험에 사용된 데이터는 892,000개의 통화 기록으로, 모델은 높은(4 이상) 또는 낮은(3 이하) CSAT 예측을 위해 이진 출력으로 작동합니다. 모델의 정확도는 85% 이상이며, 7번의 시험 과정을 진행한 결과 배포된 모델과 시뮬레이션 간에 성능 차이가 없음을 확인했습니다. 이 연구는 고객 만족도를 반영하기 위한 포괄적인 머신 러닝 파이프라인의 일환으로 적용되어 실제 환경에서도 강건한 성과를 발휘할 것으로 기대됩니다.



### Rethinking Top Probability from Multi-view for Distracted Driver Behaviour Localization (https://arxiv.org/abs/2411.12525)
Comments:
          Computer Vision and Pattern Recognition Workshop 2024

- **What's New**: 본 논문에서는 자가 지도 학습(self-supervised learning)에 기반한 행동 인식(action recognition) 모델을 도입하여 운전 중 주의 분산 행동을 감지하고, 잠재적 행동 확률을 제공하는 방법을 제안합니다. 이러한 인식 모델의 결과는 다중 카메라 뷰를 활용한 제약 집합 전략을 통해 견고한 예측을 가능하게 하며, 마지막으로 조건부 후처리(conditional post-processing) 작업을 통해 주의 분산 행동과 행동의 시간 경계를 정밀하게 찾습니다. 이를 통해 AI City Challenge 2024에서 높은 성과를 달성했습니다.

- **Technical Details**: 우리의 시스템은 행동 인식 모델, 집합 전략(ensemble strategy), 조건부 후처리라는 세 가지 주요 구성 요소로 이루어져 있습니다. 이러한 방법들은 각 카메라 뷰로부터 수집된 비디오 데이터에서 주의 분산 행동을 인식하고, 이를 바탕으로 시간 경계를 구체화합니다. 특히, 자가 지도 학습 기법은 레이블이 부족한 데이터셋에서 더 강력하고 일반화된 특성을 제공하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 본 방법은 AI City Challenge 2024의 공개 리더보드에서 트랙 3에 대해 6위를 기록하며 높은 성능을 입증했습니다. 이는 다중 카메라의 다양한 뷰를 활용하여 잡음 클래스를 제거하고 신뢰할 수 있는 행동을 정확하게 식별할 수 있었음을 보여줍니다. 이러한 결과는 행동 인식 및 시간 위치 지정을 통합하여 더 나은 성능을 달성하는 새로운 접근 방식을 나타냅니다.



### The Hermeneutic Turn of AI: Is the Machine Capable of Interpreting? (https://arxiv.org/abs/2411.12517)
Comments:
          4 pages.

- **What's New**: 이번 논문은 딥러닝(deep learning) 기술이 컴퓨팅 접근 방식을 어떻게 변화시키고 있는지를 조명합니다. 기술적인 측면뿐만 아니라, 인간과 기계 간의 상호작용에도 큰 변화를 불러오고 있음을 강조합니다. 또한, 인간과 유사한 AI 개념을 신비화하지 않고, 이를 이해하는 데 도움이 되는 철학적 전통인 해석학(hermeneutics)을 소개합니다.

- **Technical Details**: 딥러닝을 기반으로 한 인공지능 시스템들은 기존의 알고리즘과는 다른 방식으로 데이터를 처리하고 해석하는 능력을 가지고 있습니다. 이 시스템은 인간의 지각과 인식(perception and cognition)을 모방하도록 설계되었으며, 이는 수많은 실험과 데이터를 통해 발전해왔습니다. 해석학적 접근을 통해 AI의 작동 원리를 더욱 깊이 이해할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 딥러닝 기술은 다양한 분야에서 성능 향상을 이루며 주목받고 있습니다. 특히, 자연어 처리(natural language processing)와 이미지 인식(image recognition) 분야에서 그 적용이 두드러집니다. 논문은 이러한 발전이 단순한 기술적 혁신을 넘어, 인간의 사고 방식과 기계와의 관계를 재정의하고 있음을 보여줍니다.



### Transformer Neural Processes -- Kernel Regression (https://arxiv.org/abs/2411.12502)
- **What's New**: 이번 논문에서는 Transformer Neural Process - Kernel Regression (TNP-KR)라는 새로운 아키텍처를 소개합니다. 이 모델은 Kernel Regression Block (KRBlock)이라 불리는 혁신적인 Transformer 블록을 통합하여, 기존 Transformer 기반 Neural Processes (TNPs)의 attention 계산 복잡도를 크게 줄입니다. 이러한 방식으로 n_C와 n_T의 수에 따라 계산 비용을 효과적으로 감소시키며, 대량의 테스트 포인트를 처리할 수 있게 합니다.

- **Technical Details**: TNP-KR은 context points(n_C)와 test points(n_T)에 대한 attention 계산의 복잡도를 𝒪⁢((n_C+n_T)²)에서 𝒪⁢(n_C²+n_Cn_T)로 줄입니다. 또한, KRBlock 내부에 Performer attention을 사용하여 복잡도를 𝒪⁢(n_C)으로 더욱 낮췄습니다. 이러한 기술은 모델이 소비자 하드웨어에서 수백만 개의 포인트로 확장할 수 있도록 지원합니다.

- **Performance Highlights**: 평가 결과, TNP-KR의 풀 버전은 최신 방법들과 유사한 성능을 보이면서도 훈련이 더 빨라지고 테스트 포인트 수가 두 배로 늘어나는 경우에도 성능을 유지합니다. 또, 빠른 변종은 수 백만 개의 테스트 및 컨텍스트 포인트를 처리하면서도 거의 동일한 성능을 제공함을 보여줍니다. 이런 성능은 소비자 하드웨어에서 실현할 수 있는 성과입니다.



### Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus (https://arxiv.org/abs/2411.12498)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 LLMs(Large Language Models)의 추론 능력을 향상시키기 위한 새로운 접근법인 추가 논리 훈련(Additional Logic Training, ALT)을 제안합니다. 이 방법은 프로그램 생성(logical reasoning samples)된 논리적 샘플을 통해 LLM의 추론 능력을 증대시키고자 하며, 특히 잠재적인 미지의 사실을 포함하는 샘플 생성에 중점을 두고 있습니다. 새로운 데이터셋인 Formal Logic Deduction Diverse (FLD×2)를 통해 제안된 방법론의 효과를 검증하였습니다.

- **Technical Details**: ALT는 고품질 논리적 샘플을 활용하여 LLMs의 추론 능력을 개선하기 위해 설계되었습니다. 연구자들은 실질적으로 LLM이 미지의 사실을 처리할 수 있도록 다양한 추론 규칙과 언어적 표현을 갖춘 다단계 추론 샘플을 포함하는 합성 데이터셋인 FLD×2를 구축하였습니다. 이러한 샘플들은 LLM의 고유한 패턴 인식 능력을 통해 비논리적 추론과 논리적 추론을 구별할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, ALT를 적용한 LLM은 논리적 추론 벤치마크에서 최대 30점, 수학 및 코딩 벤치마크에서 최대 10점의 성능 향상을 보였습니다. LLaMA-3.1-70B 모델도 ALT를 통해 유의미한 성과 상승을 시연하였으며, 자연어 추론(NLI) 작업에서도 최대 6점의 향상이 관찰되었습니다. 이러한 결과는 LLM이 고유 지식과 논리적 사고 능력을 통합하여 다양한 문제를 해결할 수 있는 가능성을 제시합니다.



### Analysing Explanation-Related Interactions in Collaborative Perception-Cognition-Communication-Action (https://arxiv.org/abs/2411.12483)
Comments:
          4 pages, 3 figures, published as a Late Breaking Report in RO-MAN 2024

- **What's New**: 이 연구는 AI 로봇이 인간 팀원과 효과적으로 협력하기 위해 어떤 종류의 설명 능력이 필요한지를 분석합니다. 실험 참가자들 간의 의사소통을 통해 인간이 팀워크에서 기대하는 설명 유형을 분류하여 설명 가능성에 대한 인식을 심화하고자 합니다. 또한 대화 중심의 접근 방식으로 메시지의 관계를 분석하며, 이러한 커뮤니케이션이 Task 성공과 어떤 관계가 있는지를 규명합니다.

- **Technical Details**: 측정된 실험은 TeamCollab라는 시뮬레이션 환경에서 이루어졌으며, 여기서 참가자들은 위험한 물체를 식별하고 목표 지역으로 옮기는 역할을 수행합니다. 연구는 2~4명의 인간 에이전트 그룹으로 구성되어 있으며, 의사소통은 웹 기반의 근접 텍스트 채팅 시스템으로 기록됩니다. XAI(Explainable AI) 프레임워크를 사용하여 2,607개의 메시지를 분석하며, 메시지 유형을 상호작용 기술 세분화에 따라 분류하였습니다.

- **Performance Highlights**: 팀 커뮤니케이션이 성과에 미치는 영향을 평가한 결과, 전반적인 메시지 수와 수집된 물체 수 간의 역비례 관계를 확인했습니다. 반면, 실제로 위험한 물체는 메시지 수에 의존하지 않음을 보였고, 이는 참가자 간의 의사소통을 통해 얻은 지식이 향상된 예측으로 이어짐을 나타냅니다. 이는 AI 로봇이 인간과의 효과적인 의사소통을 통해 신뢰를 얻고 성과를 향상시킬 수 있는 가능성을 시사합니다.



### Comparing Prior and Learned Time Representations in Transformer Models of Timeseries (https://arxiv.org/abs/2411.12476)
Comments:
          Presented at the AI in Natural Sciences and Technology (AINST) track of the 13th Conference on Artificial Intelligence (SETN 2024), 11-13 September 2024, Piraeus, Greece

- **What's New**: 이번 연구에서는 Transformer 모델의 두 가지 변형을 비교하여 시간 표현이 시계열 분석에 미치는 영향을 조사했습니다. 하나는 기존 문헌에서 제안된 고정된 시간 표현을 사용하고, 다른 하나는 데이터로부터 학습된 시간 표현을 활용합니다. 결과적으로, 시계열 데이터에서 시간 표현의 적절성을 높이는 것이 중요하다는 것을 강조하며, 기존 모델에 대한 한계를 지적하고 있습니다.

- **Technical Details**: Transformer 아키텍처는 순차적으로 입력을 처리하는 대신, 전체 시계열을 입력으로 받아 Attention 메커니즘을 통해 데이터 간의 관계를 동시에 평가합니다. 시간 표현을 위해, 연구에서는 Sinusoidal을 기반으로 한 절대적인 위치 인코딩과 트라이앵글 파형 모델을 제공합니다. 이는 시간 간격을 효과적으로 캡처하는 방법으로, 특히 동절기와 일간 주기성을 고려하여 설계되었습니다.

- **Performance Highlights**: 실험 결과, 고정된 시계열 표현보다 데이터에서 직접 학습된 시간 표현이 더 나은 성능을 보였습니다. 특히 태양광 패널의 에너지 출력을 예측할 때, 새로운 시간 표현이 정확성을 증대시켰고, 이는 기존에 알아온 패턴을 더욱 잘 모델링하는 데 기여했습니다. 이러한 발견은 Transformer가 시계열 데이터에 대한 처리를 향상시킬 수 있는 가능성을 제시합니다.



### AI Flow at the Network Edg (https://arxiv.org/abs/2411.12469)
- **What's New**: 최근 대규모 언어 모델(LLMs)과 그 다중 모드 변형의 발전은 인공지능(AI) 분야에서 많은 혁신을 가져왔습니다. 본 논문에서는 AI Flow라는 프레임워크를 제안하여, 다양한 계산 리소스를 활용하여 네트워크 엣지에서 지능을 유통하는 방법을 모색합니다. 이러한 접근은 클라우드에서 엣지 환경으로 대규모 모델을 배포하는 데 있어 발생하는 여러 문제들을 해결하려는 시도를 반영하고 있습니다.

- **Technical Details**: AI Flow는 디바이스, 엣지 노드, 클라우드 서버 간의 이질적인 리소스를 활용하여 추론 과정을 간소화합니다. 이 프레임워크는 정보 흐름에서 지능 흐름으로의 전환을 추구하며, 데이터 전송 대신 엣지에서 감지된 정보 중 핵심적 특성만을 추출해 통신 비용을 줄이고 효율적인 서비스를 제공합니다. 시스템 아키텍처는 엣지 디바이스, 엣지 서버, 클라우드 서버의 세 가지 모듈로 구성됩니다.

- **Performance Highlights**: AI Flow 프레임워크는 다양한 사례를 통해 응답 지연을 줄이고 고품질의 결과를 유지하는 능력을 보여줍니다. 특히 이미지 캡셔닝(use case) 분야에서의 실험 결과는 엣지에서의 저지연 데이터 처리와 개선된 사용자 경험을 증명합니다. 이 기술의 적용은 로봇 제어, 스마트 홈, 증강 현실 및 자율 주행 등 다양한 분야에서 폭넓은 가능성을 펼칩니다.



### Guide-to-Explain for Controllable Summarization (https://arxiv.org/abs/2411.12460)
- **What's New**: 최근 강력한 성능을 보이는 대형 언어 모델(LLMs)을 활용한 추상적 요약(abstractive summarization) 연구가 주목받고 있습니다. 이 논문에서는 사용자의 특정 요구에 맞춘 제어 가능한 요약(controlled summarization)이 여전히 부족하여 LLMs의 사용성이 한정적임을 지적합니다. 이를 해결하기 위해, 제안된 가이드-설명 프레임워크(guide-to-explain framework, GTE)를 통해 LLM이 요약 초기 단계에서 잘못 정렬된 속성을 식별하고, 오류를 설명하여 잘 조정된 요약을 생성하는 방안을 제시합니다.

- **Technical Details**: GTE 프레임워크는 LLM이 생성한 요약에서의 잘못 정렬된 속성을 식별하는 단계와 모델이 자신의 오류를 설명하도록 안내하는 방식으로 구성되어 있습니다. 이 과정에서 속성 조정은 외부 모듈이나 추가 학습 없이 LLM만으로 진행됩니다. 논문에서는 다섯 가지 속성(extractiveness, length, topic, speaker)을 기준으로 LLM의 제어 가능성을 분석하고, 추상적 요약에서 LLM의 언어적 속성과 수치적 속성 제어의 어려움을 강조합니다.

- **Performance Highlights**: GTE는 혼합 속성 제어 데이터셋(MACSumDoc 및 MACSumDial)에서 평가되었으며, 최소한의 반복(iterations)만으로 각 속성을 효과적으로 제어하는 성과를 보였습니다. 평가 지표를 통해 제어된 요약이 높은 품질을 유지하고 있음을 입증하였습니다. 또한, LLM이 여러 속성을 동시에 제어하는 데 어려움이 있음을 발견했으며, 이는 연구의 의미를 더욱 강조합니다.



### Evaluating the Prompt Steerability of Large Language Models (https://arxiv.org/abs/2411.12405)
- **What's New**: 이번 연구에서는 다양한 가치 시스템과 문화를 반영할 수 있는 다원적 AI 모델(Pluralistic AI 모델)을 설계하기 위해 모델의 'steerability'를 평가하는 벤치마크를 제안합니다. 특히, prompt steerability를 정의하고, 모델이 다양한 페르소나를 채택할 수 있는 정도를 계량적으로 분석하고자 합니다. 현재 많은 모델이 한정된 steerability를 보이는 이유를 분석하는 것이 주된 초점입니다.

- **Technical Details**: 연구에서는 generative language model M_{\theta}와 관련된 확률적 함수 p_{\theta}를 정의하고, 사용자로부터 제시되는 prompt에 따라 모델의 동작을 평가하기 위한 score functions 집합 \mathcal{S}를 도입합니다. 평가 프로파일은 모델 출력의 joint distribution으로 정의되며, prompt-output 쌍 (x, y)을 통해 계산됩니다. 이러한 평가 프로파일을 바탕으로 모델의 steerability를 정량화하는 steerability indices를 도입하여 비교적인 측정을 가능하게 합니다.

- **Performance Highlights**: 제안된 벤치마크는 현재의 많은 모델들이 기본 동작의 편향과 여러 페르소나 차원에서의 비대칭성으로 인해 제한된 steerability를 갖고 있음을 보여줍니다. 이는 모델이 특정 행동으로 쉽게 정렬될 수 있는 정도를 정량화하고, 다른 연구에서 제안된 방법들과 비교하여 prompt 기반의 steerability에 대한 새로운 통찰을 제공합니다. 최종적으로, 제안하는 방법론은 모델의 다양한 페르소나를 채택하도록 유도하는 방법을 분석함으로써, finer-tuning 설정을 보완하는 역할을 합니다.



### Do LLMs Understand Ambiguity in Text? A Case Study in Open-world Question Answering (https://arxiv.org/abs/2411.12395)
Comments:
          Accepted at the REU Symposium at IEEE BigData 2024

- **What's New**: 자연어의 모호성은 오픈 도메인 질문 응답을 위한 대규모 언어 모델(LLMs)에게 큰 도전 과제가 되고 있습니다. 이 논문에서는 명시적인 불모호화(disambiguation) 전략의 영향을 측정하는 데 중점을 두어 기존 LLM과 few-shot LLM의 성능을 비교합니다. 실험을 통해 학습이 필요 없는 간단한 토큰 수준의 불모호화 방법이 모호한 질문 응답 작업에서 LLM 성능을 향상시키는 데 효과적임을 보여줍니다.

- **Technical Details**: 본 연구에서는 LLM의 민감성을 평가하기 위해 언어적 및 맥락적 변화가 모호한 질문 응답에 미치는 영향을 측정합니다. 세 가지의 다른 프롬프트 전략을 사용하여 LLM에서 답변을 생성하는 실험을 수행했습니다: (1) 기본 질문-응답 프롬프트, (2) 언어적 변형 추가 및 (3) 모델 내부 지식을 활용한 맥락적 보강 접근 방식입니다. 이러한 전략들을 통해 모델의 모호성 이해도를 정량적으로 측정하고 분석합니다.

- **Performance Highlights**: 실험 결과, LLM은 모호한 질문에 대해 명시적으로 불모호화된 질문 이상의 성능을 보입니다. 저자들은 1,000개의 모호한 질문에 대한 다양한 프롬프트 전략을 적용하고 이를 통한 성능 차이를 체계적으로 분석하였습니다. LLM의 성능 향상 및 모호성에 대한 이해도를 높이는 최적의 접근 방식을 제시하며, 이러한 연구 결과는 향후 AI 시스템의 발전에 기여할 수 있는 중요한 통찰력을 제공합니다.



### A Layered Architecture for Developing and Enhancing Capabilities in Large Language Model-based Software Systems (https://arxiv.org/abs/2411.12357)
- **What's New**: 이번 논문은 Large Language Models (LLMs)의 활용 범위를 기본 언어 작업을 넘어서 확장하기 위한 최근의 노력을 다루고 있습니다. LLM의 일반화 가능성과 유연성이 폭넓은 채택을 가능하게 했으나, 애플리케이션 개발의 변화하는 요구는 그들의 본래 능력을 초과하는 경우가 많습니다. 이를 해결하기 위해 다양한 방법, 예를 들어 추론의 온도 조정(inference temperature adjustments) 또는 창의성을 유도하는 프롬프트(prompts)가 필요하다는 것을 강조합니다.

- **Technical Details**: 논문에서는 LLM 소프트웨어 시스템 개발을 특정 속성으로 정의된 개별 계층으로 구성하는 계층적 아키텍처(layered architecture)를 제안합니다. 이러한 계층에 맞는 능력을 정렬함으로써, 프레임워크는 효과적이고 효율적인 방식으로 기능과 품질을 지원하는 능력의 체계적인 구현을 촉진합니다. 이를 통해 개발자에게 LLM 기반 소프트웨어 시스템 개발에 적합한 기술을 선택할 수 있는 실행 가능한 통찰력을 제공합니다.

- **Performance Highlights**: 실제 사례 연구를 통해 프레임워크의 유용성을 입증하며, 성능의 강건성(robustness)과 확장성(scalability)을 증진시킬 수 있는 방법을 제시합니다. 다양한 개발 선택이 엔지니어링 복잡성(engineering complexity), 확장성, 운영 비용(optimal operational costs) 간의 균형을 맞추는 데 어떻게 기여하는지를 설명합니다.



### DiM: $f$-Divergence Minimization Guided Sharpness-Aware Optimization for Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2411.12350)
Comments:
          8page

- **What's New**: 반자동 데이터 주석 문제를 해결하기 위해 반지도 학습(semi-supervised learning, SSL) 기술이 각광받고 있습니다. 특히 의료 이미지 분할(medical image segmentation) 분야에서 SSMIS(semi-supervised medical image segmentation)가 주목 받고 있으며, 정확한 주석 데이터의 필요성을 줄이는 방법으로 연구되고 있습니다. 본 연구에서는 SAM(sharpness-aware minimization) 기술을 활용하여 모델의 일반화 성능을 향상시키는 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 $f$-divergence 최소화(minimization)를 통해 SAM의 한계를 극복하는 새로운 방법인 DiM(f-divergence minimization guided sharpness-aware optimization)을 제안합니다. 이 방법은 모델 파라미터의 민감도를 조정하고, 다양한 데이터셋에 대한 적응성을 향상시킵니다. $f$-divergence를 도입함으로써 DiM 방법은 소스 데이터셋과 타겟 데이터셋 간의 균형 잡힌 성능을 개선하고, 소스 데이터셋 과적합(overfitting)을 방지하는 데 기여합니다.

- **Performance Highlights**: DiM 방법은 SSMIS 벤치마크에서 최신 기술들에 비해 우수한 성능을 입증하였습니다. 특히, 모델의 안정성과 다양한 도메인 간의 적응성을 크게 향상시키며, 기존 SAM 방식과의 비교에서 더욱 효과적인 성능 향상을 보여주었습니다. 이러한 연구 결과는 의료 이미징 분야에서 반지도 학습의 적용 가능성을 넓히는 중요한 발전이 될 것입니다.



### CLIP Unreasonable Potential in Single-Shot Face Recognition (https://arxiv.org/abs/2411.12319)
- **What's New**: 이번 연구는 얼굴 인식(Face Recognition) 기술에 CLIP(Contrastive Language-Image Pretraining) 모델을 활용하여 새로운 접근 방식을 소개합니다. 기존 방식은 얼굴의 주요 특징을 추출하고 이를 데이터베이스와 비교하는 과정을 거쳤으나, CLIP는 이러한 과정 없이 단일 샷 파인튜닝(single-shot finetuning)만으로 낮은 오탐률(false positive rate)을 달성하는 것을 보여주고 있습니다. 이 연구는 얼굴 인식 솔루션을 간소화하고 지속적으로 존재하는 문제를 해결할 수 있는 CLIP의 잠재력을 드러냅니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 자원봉사자들의 고해상도 얼굴 이미지를 포함하고 있으며, 각 얼굴은 개인별로 정렬되어 있습니다. 얼굴 인식 실험을 위해 SCRFD(face detection model) 모델을 활용하여 이미지에서 얼굴 영역을 정확하게 감지하고, 눈, 코, 입의 중요한 키포인트를 추출했습니다. 이를 통해 이미지 간 얼굴의 수직 정렬을 보장하고, 고유한 비율을 유지하면서 데이터 표현을 일관되게 하였습니다.

- **Performance Highlights**: 클립 모델의 파인튜닝 과정에서 얼굴 인식을 이미지 분류(image classification) 문제로 간주하여 처리했으며, 모델의 이미지 인코더 파라미터를 동결하고 텍스트 인코더만을 이용해 학습을 진행했습니다. 결과적으로 CLIP 모델은 기존의 전통적인 얼굴 인식 알고리즘에 비해 현실 세계에서의 오탐률을 크게 낮추는데 성공했습니다. 즉, CLIP의 다모드(multi-modal) 디자인은 향후 다양한 얼굴 인식 과제의 성능 개선에 기여할 것으로 기대됩니다.



### Balancing Accuracy and Efficiency in Multi-Turn Intent Classification for LLM-Powered Dialog Systems in Production (https://arxiv.org/abs/2411.12307)
- **What's New**: 이 논문은 대화형 AI 시스템의 다중 턴(종종 multi-turn이라고도 하는) 의도 분류(multi-turn intent classification)를 개선하기 위한 두 가지 새로운 접근 방식을 제안합니다. 첫째, Symbol Tuning을 도입하여 의도 레이블을 간소화하고, 둘째, C-LARA(Consistency-aware, Linguistics Adaptive Retrieval Augmentation)라는 프레임워크를 개발하여 대량의 다중 턴 대화 데이터를 합성합니다. 이 방법은 다국어 산업 시스템에서 낮은 자원 하에서도 확장 가능성을 높이고 비용을 절감합니다.

- **Technical Details**: Symbol Tuning은 LLM과의 상호 작용에서 의도 라벨을 압축하여 복잡성을 줄이고 성능을 개선하는 접근 방식입니다. C-LARA는 사용자로부터의 비표시 발화(unlabeled utterances)를 사용하여 다중 턴 데이터를 생성하는 효율적인 도구입니다. 이 두 가지 방법은 대화의 맥락을 고려하면서 다중 턴 의도 분류(MTIC)의 정확도를 높입니다.

- **Performance Highlights**: 이 연구는 MTIC 시스템에서 AUC 점수를 5.09% 개선하고 주석(annotations) 비용을 40% 줄이는 성과를 보여줍니다. 다국어 대화 데이터셋에 대한 실험을 통해, 제안된 방법들이 모델의 성능을 상당히 향상시키며 자원 효율성 또한 높인 것으로 나타났습니다. 이러한 결과는 낮은 자원 환경에서의 실제적인 응용 가능성을 강조합니다.



### SSEditor: Controllable Mask-to-Scene Generation with Diffusion Mod (https://arxiv.org/abs/2411.12290)
- **What's New**: 최근 3D 확산 기반의 의미 장면 생성 기술이 주목받고 있습니다. 기존 방식은 무조건적인 생성을 기반으로 하며, 장면 편집 시 여러 단계의 재샘플링을 요구하여 제어 가능성과 유연성이 크게 제한됩니다. 이를 해결하기 위해 제안된 SSEditor는 지정된 목표 카테고리를 생성하면서도 여러 단계의 재샘플링이 필요 없는 제어 가능한 Semantic Scene Editor입니다.

- **Technical Details**: SSEditor는 두 단계의 확산 기반 프레임워크를 사용하며, 첫 번째 단계에서는 3D 장면 오토인코더를 사용해 잠재적인 트리플레인 피처를 학습합니다. 두 번째 단계에서는 마스크 조건형 확산 모델을 통해 맞춤형 3D 의미 장면 생성을 수행합니다. 또한, 기하학-의미 융합 모듈(Geometric-Semantic Fusion Module)이 도입되어 모델이 기하학적인 정보와 의미 정보를 효과적으로 학습하도록 돕습니다.

- **Performance Highlights**: SemanticKITTI 및 CarlaSC 데이터셋에서의 실험 결과, SSEditor는 목표 생성의 제어 가능성과 유연성뿐만 아니라 의미 장면 생성 및 재구성의 품질 면에서도 이전 접근 방법들을 능가함을 보여주었습니다. 더 나아가, 보지 못한 Occ-3D Waymo 데이터셋에서의 실험 결과는 SSEditor가 새로운 도시 장면을 생성할 수 있는 능력을 입증하였으며, 3D 장면의 신속한 구축을 가능하게 합니다.



### libcll: an Extendable Python Toolkit for Complementary-Label Learning (https://arxiv.org/abs/2411.12276)
Comments:
          10 pages, 3 figures

- **What's New**: 본 논문에서는 complementary-label learning (CLL)이라는 약한 감독 학습 방법론의 주요 문제점을 다루기 위해 	exttt{libcll}이라는 확장 가능한 파이썬 툴킷을 새롭게 소개합니다. 	exttt{libcll}은 다양한 CLL 알고리즘과 데이터셋을 지원하는 범용 인터페이스를 제공하여 일관성 문제를 해소하고, 연구 과정을 간소화하도록 설계되었습니다. 이 툴킷은 CLL 기술을 효율적으로 채택하고 구현할 수 있도록 설치가 용이하고 포괄적인 사용 가이드를 제공합니다.

- **Technical Details**: CLL은 각 레이블이 데이터 인스턴스에 속하지 않는 클래스를 나타내는 약한 감독 학습 문제로, 고급 그래프 구조와 다양한 네트워크 아키텍처를 지원합니다. 	exttt{libcll}은 사용자 정의 전이 행렬을 사용해 보완 레이블을 생성하는 기능을 포함하였으며, 다양한 CLL 알고리즘과 데이터셋에 대한 광범위한 벤치마크를 제공합니다. 이 툴킷은 학습 성능의 일관성을 유지하고 연구자들이 쉽게 비교하고 분석할 수 있도록 도와줍니다.

- **Performance Highlights**: 	exttt{libcll}을 사용한 포괄적인 ablation 연구는 CLL 연구를 발전시키기 위한 중요한 인사이트를 생성함을 입증하였습니다. 15개 데이터셋과 14개 알고리즘의 벤치마크 결과를 통해 연구자들이 각 방법의 강점과 한계를 평가할 수 있는 통합된 관점을 제공합니다. 이 툴킷은 CLL 분야에서의 연구 협력과 재현성을 촉진하며, 더 나은 알고리즘 개발에 기여할 것으로 기대됩니다.



### Building Trust: Foundations of Security, Safety and Transparency in AI (https://arxiv.org/abs/2411.12275)
- **What's New**: 이 논문은 공개적으로 사용 가능한 AI 모델의 생태계가 빠르게 발전하는 양상을 탐구하고, 이러한 변화가 보안(security) 및 안전(safety) 분야에 미치는 잠재적 영향을 다룹니다. AI 모델이 점점 더 널리 사용됨에 따라, 이 모델들이 가진 리스크(risk)와 취약점(vulnerability)을 이해하는 것이 중요합니다. 보안 및 안전 시나리오를 검토하며, 모델 라이프 사이클(lifecycle)과 소유권(ownership) 프로세스의 부재와 같은 도전 과제를 강조하고 있습니다.

- **Technical Details**: 현재의 보안 및 안전 시나리오에 대한 검토와 함께, 모델 개발자(developer)와 최종 사용자(end-users)를 위한 보안 및 안전을 향상시키기 위한 포괄적인 전략을 제안합니다. 이 논문은 AI 모델의 개발 및 운영에 있어 더 표준화된 보안, 안전 및 투명성(transparency)을 위한 기초적 요소들을 제공하는 것을 목표로 합니다. AI 모델의 생태계와 관련된 여러 커뮤니티를 형성하는 과정에서 중요한 기초 자료를 제시합니다.

- **Performance Highlights**: 이 연구는 AI 모델의 공개 생태계가 증가함에 따라 발생할 수 있는 다양한 위험 요소들을 관리하고, 이러한 모델들을 보다 안전하고 신뢰할 수 있게 만드는 방법을 탐구합니다. 제안된 전략들은 AI 모델 사용자와 개발자에게 안전한 환경을 제공하기 위한 것으로, 현대 AI 생태계에서의 보안과 안전 이슈를 해결하기 위한 필수적인 기반이 될 것입니다.



### Error-Feedback Model for Output Correction in Bilateral Control-Based Imitation Learning (https://arxiv.org/abs/2411.12255)
- **What's New**: 이 연구에서는 신경망의 출력 오류를 보정하는 피드백 메커니즘을 개발했습니다. 기존의 피드포워드 구조를 가진 신경망의 한계를 극복하기 위해, 상위 계층에 따라 하위 계층을 조정하는 계층적 구조를 도입했습니다. 이를 통해 자율 제어와 오류 피드백을 통한 성능 향상을 확인했습니다.

- **Technical Details**: 제안된 모델은 상위 계층과 하위 계층으로 구성된 계층적 구조로, 상위 계층은 장기 예측을 수행하고 하위 계층은 단기 예측을 담당합니다. 하위 계층은 내부 상태가 없는 다층 퍼셉트론으로 구성되며, 예측한 상태의 오류를 피드백하여 성능을 개선합니다. 이 연구에서는 오류 피드백 모델을 제안하여 시스템의 출력을 조정하는 방법을 보여 주었습니다.

- **Performance Highlights**: 캐릭터 작성 과제를 통해 제안된 모델의 성능을 평가했으며, 이전에 학습하지 않은 캐릭터에 대해 더 향상된 정확도를 기록했습니다. LSTM과 MLP를 비교하여 하위 계층의 구성 요소가 성능에 미치는 영향을 분석했습니다. 이 연구는 신경망과 제어 이론의 통합 가능성을 보여주는 중요한 단계를 나타냅니다.



### Evaluating Tokenizer Performance of Large Language Models Across Official Indian Languages (https://arxiv.org/abs/2411.12240)
- **What's New**: 이 연구는 12개의 대형 언어 모델(LLM)의 인도 공식 언어 22개에 대한 토크나이저(tokenizer)의 성능을 비교 평가합니다. 특히, SUTRA 토크나이저가 다른 모든 모델을 초월하여 14개 언어에서 뛰어난 성능을 보여주었음을 밝혀냈습니다. 또한 이 논문은 인도 언어 처리에 있어 타겟팅된 토크나이저 전략의 필요성을 강조합니다.

- **Technical Details**: 대부분의 LLM은 두 가지 토크나이저 알고리즘, 즉 WordPiece와 Byte Pair Encoding (BPE)을 사용합니다. 이 연구는 Normalized Sequence Length (NSL)라는 지표를 사용하여, 각 모델의 토크나이저 성능을 평가하고, 다양한 언어를 처리하는 데 필요한 효율성을 분석합니다. 12개의 모델을 대상으로 하였고, 각 언어의 원주율 문자로 예제 텍스트를 작성하여 평가하였습니다.

- **Performance Highlights**: SUTRA 토크나이저는 평균 NSL 값에서 최고 성과를 기록했으며, 이를 통해 인도 언어 처리에서의 우수성을 입증했습니다. 특히, SUTRA는 ChatGPT 4-o 및 기타 인도 표시 모델보다 뛰어난 결과를 보여주었고, 14개 언어에서 가장 높은 성과를 달성했습니다. 이러한 결과는 LLM의 다국어 및 인도 중심 모델에서의 토크나이저의 중요성을 다시 한번 부각시킵니다.



### Contrast Similarity-Aware Dual-Pathway Mamba for Multivariate Time Series Node Classification (https://arxiv.org/abs/2411.12222)
Comments:
          Submitted to Knowledge-Based Systems on Nov 17, 2024

- **What's New**: 이번 연구는 복잡한 다차원 시간을 다루는 멀티 변수 시계열(MTS) 분류를 위한 새로운 접근 방식으로, Contrast Similarity-aware Dual-Pathway Mamba (CS-DPMamba)를 제안합니다. 이 방법은 우선, Temporal Contrast Learning을 통해 각 샘플의 동적 유사성을 캡처하고, 이어서 FastDTW를 사용하여 유사성 행렬을 구성합니다. 이 후 DPMamba 모델을 통해 MTS의 양방향 특성을 고려하여 장기적 및 단기적 종속성을 효과적으로 포착합니다.

- **Technical Details**: 연구의 목적은 MTS 데이터에서의 유사성과 장기 종속성을 결합하는 것입니다. CS-DPMamba는 Temporal Contrast Learning을 기반으로 샘플의 특성을 추출하고, Fast Dynamic Time Warping을 통해 MTS 표현 간의 유사성 행렬을 구축합니다. 이후 DPMamba와 Kolmogorov-Arnold Network 강화 그래프 동형성 네트워크를 결합하여 정보 전파와 MTS 노드 분류 작업을 수행합니다.

- **Performance Highlights**: 실험은 University of East Anglia (UEA) MTS 데이터셋에서 수행되었으며, 다양한 응용 시나리오를 포함합니다. CS-DPMamba는 기존의 방법에 비해 더욱 정교한 MTS 노드 분류를 달성했으며, 감독 및 반 감독 학습에서 그 우수성을 입증했습니다. 이러한 결과는 다차원 시계열 데이터의 복잡한 종속성을 효과적으로 다루는 방법의 필요성을 강조합니다.



### DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning (https://arxiv.org/abs/2411.12220)
Comments:
          14 pages

- **What's New**: 이 논문에서는 Federated Learning (FL) 환경에서 백도어 공격을 방어하는 새로운 프레임워크인 DeTrigger를 제안합니다. DeTrigger는 적대적 공격 방법론의 통찰력을 활용하여 백도어 트리거를 효과적으로 탐지하고 분리합니다. 이를 통해 정상 모델의 지식을 소실하지 않으면서 백도어 활성화 weight를 정밀하게 가지치기할 수 있습니다. 결과적으로 DeTrigger는 전통적인 방법보다 최대 251배 더 빠른 탐지 속도와 최대 98.9%의 백도어 공격 완화 능력을 보여줍니다.

- **Technical Details**: DeTrigger는 gradient 분석 및 temperature scaling을 사용하여 백도어 트리거를 탐지하고 격리합니다. 모델 gradient는 입력에 따른 모델 weight의 반응을 포착하여, 백도어 공격의 미세한 변화를 감지합니다. 이 방법은 각 클라이언트 모델에 대한 exhaust inspection 없이도 비정상적인 패턴을 효율적으로 탐지할 수 있는 장점을 가집니다. 또한, DeTrigger는 손상된 모델을 제거하는 대신, 백도어 활성화 weight만을 정확히 제거하여 정상 지식을 유지하는 기능을 제공합니다.

- **Performance Highlights**: DeTrigger는 네 개의 널리 사용되는 데이터셋을 대상으로 철저히 평가되어 약 98.9%의 백도어 공격 완화율을 달성하였습니다. 또한, 기존 방법들보다 대략 251배 더 빠른 탐지 속도를 자랑하며, 글로벌 모델의 정확성을 유지하면서도 백도어 공격의 영향을 크게 줄였습니다. 이러한 속도와 정확성의 조합은 DeTrigger가 세련된 백도어 위협으로부터 federated learning 환경을 보호하는 데 효과적인 솔루션임을 입증합니다.



### CCIS-Diff: A Generative Model with Stable Diffusion Prior for Controlled Colonoscopy Image Synthesis (https://arxiv.org/abs/2411.12198)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 논문에서는 CCIS-DIFF라는 새로운 Generative 모델을 제안합니다. 이 모델은 colonoscopy 이미지 합성을 위한 더욱 정교한 제어 기능을 제공하여, 실질적인 임상 요구사항에 부합하는 이미지를 생성할 수 있습니다. 특히, 합성된 폴립이 대장 점막과 매끄럽게 통합되도록 하는 blur mask weighting 전략과 임상 특성에 맞춘 텍스트 기반 주의 메커니즘을 도입했습니다.

- **Technical Details**: CCIS-DIFF 모델의 핵심은 다중 모달(colonoscopy images, segmentation masks, clinical text descriptions) 데이터셋을 구축한 것입니다. 이를 통해 사전 훈련된 diffusion 모델을 정밀 조정할 수 있으며, 생성 과정에서 텍스트 정보를 효과적으로 통합하는 텍스트 인지 주의 메커니즘이 구현되어 있습니다. 또한, Gaussian blur 작업을 통해 합성된 폴립과 배경 간의 전환을 부드럽게 만드는 전략이 도입되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 CCIS-DIFF 모델은 공간적 제약과 임상적 일관성에 대한 세밀한 제어가 가능하여 높은 품질과 다양성을 갖춘 colonoscopy 이미지를 생성했습니다. 이러한 이미지는 이후의 세분화(segmentation) 및 진단(diagnostic) 작업에 유용한 지원을 제공하며, 기존의 방법들보다 더 나은 성능을 보이는 것으로 나타났습니다.



### A More Advanced Group Polarization Measurement Approach Based on LLM-Based Agents and Graphs (https://arxiv.org/abs/2411.12196)
- **What's New**: 이 논문은 사회적 미디어에서 그룹 극단화를 측정하기 위한 새로운 접근 방식을 제안합니다. 이 접근 방식은 LLM(대형 언어 모델) 기반 에이전트와 그래프 구조의 커뮤니티 감정 네트워크(Community Sentiment Network, CSN)를 사용하여 극단화 상태를 모델링합니다. 기존 방법들의 한계를 극복하기 위해 설계된 Community Opposition Index (COI)라는 메트릭을 도입하여 극단화 측정의 정확성과 해석력을 높입니다.

- **Technical Details**: 주요 방법론은 커뮤니티 감정 네트워크(CSN)를 활용하여 다수의 하위 그룹 간 감정을 시각화하는 것입니다. 다중 에이전트 시스템을 통해 CSN을 구성하고, COI를 기반으로 극단화 측정 메트릭을 계산합니다. CSN은 과거 기술인 'Sentiment Thermometer'의 확장으로, 여러 하위 그룹 간의 감정적 상호작용을 포함하는 구조입니다.

- **Performance Highlights**: 제안된 다중 에이전트 시스템은 제로샷(stance detection) 판단 작업에서 뛰어난 성과를 입증했습니다. CSN을 통해 시간에 따른 극단화 상태를 효과적으로 나타내는 것이 가능하며, LLM 기반 에이전트들을 활용한 메트릭 개발이 정확성과 효율성을 모두 개선하는데 기여했습니다. 궁극적으로, 이 연구는 그룹 극단화 분석의 유용성 및 해석 가능성을 크게 향상시킵니다.



### Testability of Instrumental Variables in Additive Nonlinear, Non-Constant Effects Models (https://arxiv.org/abs/2411.12184)
- **What's New**: 이 논문은 관찰 데이터에서 도출된 도구 변수를 테스트하는 문제를 다룹니다. 기존의 많은 연구들은 치료(treatment)가 이산 변수일 때의 시나리오에 중점을 두었으나, 본 연구에서는 연속 변수인 약물 투여량이나 영양 성분 수준과 같이 치료가 연속 변수를 가질 수 있는 상황을 고려합니다. 우리는 Auxiliary-based Independence Test (AIT) 조건을 제안하여 변수의 유효성을 검정하는 방법론을 제시합니다.

- **Technical Details**: 이 논문에서 제안하는 AIT 조건은 Additive Nonlinear, Non-Constant Effects (ANINCE) 모델을 기반으로 하여, 단일 도구 변수가 유효한지를 검증하기 위한 필요 조건을 도입합니다. 특히, 비정상적 인과 효과가 있는 시나리오에서도 적용 가능하며, AIT 조건이 모든 무효 IV를 탐지하기 위한 필요하고 충분한 조건을 제공함을 증명합니다. 본 연구는 유한 데이터에서 공변량(covariates)을 고려한 AIT 조건 테스트의 실용적인 구현을 제시합니다.

- **Performance Highlights**: 합성 데이터와 세 가지 실제 데이터셋에서 우리의 접근 방식이 효과적임을 보여줍니다. 이러한 결과는 AIT 조건의 효능을 입증하고, 다양한 상황에서 무효 도구 변수를 식별하는 데 기여할 수 있음을 시사합니다. 제안된 방법은 관찰 데이터에서 도구 변수를 선택하는 데 있어 반복적으로 거론되던 문제를 해결할 수 있는 실질적인 방법론으로 자리잡을 것입니다.



### Diffusion-Inspired Cold Start with Sufficient Prior in Computerized Adaptive Testing (https://arxiv.org/abs/2411.12182)
Comments:
          Accepted by KDD2025

- **What's New**: 본 연구에서는 Cold Start with Insufficient Prior (CSIP) 문제를 해결하기 위해 Diffusion Cognitive States TransfeR Framework (DCSR)를 제안합니다. DCSR는 Diffusion Models (DMs)를 기반으로 하며, 다양한 도메인에서의 인지 상태 전이를 다루는 다중 도메인 전이 프레임워크입니다. 기존 CAT 시스템이 다른 코스에서 수집된 풍부한 응답 기록을 효율적으로 활용하지 못하는 문제를 인식하고, 이를 해결하기 위해 생물학적 목표에 대한 적합한 초기 능력을 복원하는 설계 구조를 제공하고 있습니다.

- **Technical Details**: DCSR는 examinee의 인지적 능력을 타겟 도메인에서 세밀하게 생성하기 위해 Causal Relationships를 분석하고 Redundant 및 Extraneous Cognitive States를 고려하여 설계되었습니다. 본 프레임워크는 다각적인 인지 상태 전환 다리(cognitive state transition bridge)를 구축하여 examinee의 인지 상태에 기반한 질문 선택 알고리즘에 통합 가능하게 합니다. 생성된 초기 상태는 CAT 시스템의 요구 사항에 맞춰 컨디셔닝되어 카오스(Causal) 정보 손실을 방지하고 임의성이 과도해지지 않도록 조절하는 일관성 제약 조건과 작업 지향 제약 조건(task-oriented constraint)을 포함하고 있습니다.

- **Performance Highlights**: DCSR는 5개의 실제 데이터 세트에서 광범위한 실험을 수행하여 기존의 baseline 방법들을 모두 초월하며 CSIP 문제를 해결하는데 효과적임을 입증했습니다. CAT 시스템의 Cold Start 성능을 향상시키며, 질문 선택 알고리즘과의 원활한 통합을 통해 CAT 시스템의 전반적인 효율성을 증가시킵니다. 연구 결과는 CAT 시스템의 초기 테스트 단계에서 examinee의 능력 평가에서 발생하는 혼란과 좌절을 줄이는 데 중요한 기여를 할 것으로 보입니다.



### Enhancing Low Dose Computed Tomography Images Using Consistency Training Techniques (https://arxiv.org/abs/2411.12181)
- **What's New**: 본 논문에서는 새로운 beta noise 분포를 도입하여 이미지 생성의 품질을 높이면서도 적은 수의 매개변수로 조절할 수 있는 유연성을 제공합니다. 이를 통해 High Noise Improved Consistency Training (HN-iCT)라는 새로운 훈련 방식을 제안하여, 저선량 이미지(Low Dose)에서 중요한 특징을 추출할 수 있습니다. 또한, sinusoidal curriculum 기법을 활용하여 노이즈의 다양한 수준을 관리함으로써 모델의 학습 효율성을 극대화하였습니다.

- **Technical Details**: Consistency 모델은 확률적 미분 방정식(stochastic differential equation, SDE)을 통해 데이터 분포를 노이즈 분포로 점진적으로 변환하여 데이터를 생성하는 방법을 사용합니다. 이 모델들은 변환 과정을 역으로 학습하여 노이즈에서 데이터를 생성합니다. 최근 제안된 HN-iCT 아키텍처는 Weighted Attention Gates(WAG)를 활용하여 조건부 이미지로부터 신뢰할 수 있는 특징을 추출하며, 이는 저선량 CT 이미지를 효과적으로 처리하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, HN-iCT를 사용한 무조건적인 이미지 생성이 CIFAR10 및 CelebA 데이터셋에서 기존의 CT 및 iCT 훈련 기법에 비해 상당히 우수한 성능을 보였습니다. 또한, 이미지 조건 모델은 저선량 CT 스캔을 향상시키는 데 뛰어난 성능을 나타내어, 새로운 접근 방식이 임상 요구사항을 효과적으로 충족할 수 있음을 시사합니다.



### SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks (https://arxiv.org/abs/2411.12173)
- **What's New**: 이 논문에서는 SkillTree라는 새로운 프레임워크를 제안합니다. SkillTree는 복잡한 연속 행동 공간을 이산(skill) 공간으로 축소하여 결정 트리(decision tree)를 통합하였습니다. 이로 인해 복잡한 작업에서의 의사 결정 과정을 더욱 투명하게 만들었습니다.

- **Technical Details**: SkillTree는 계층적 접근 방식을 채택하여 고급 정책 내에 미분 가능 결정 트리를 통합하고 이를 통해 기술 임베딩(skill embeddings)을 생성합니다. 이 기술 임베딩은 저수준 정책이 기술을 실행하는 데 필요한 지침을 제공합니다. 또한, 기술 공간을 이산 단위로 정규화하여 정책 학습을 간소화하고 학습된 기술의 설명 가능성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SkillTree는 복잡한 로봇 암 제어 작업에서 기술 기반 신경망과 유사한 성과를 달성했습니다. 더욱이, SkillTree는 기술 수준의 설명을 제공하여 결정 과정의 투명성을 증가시킵니다. 이는 안전이 중요한 애플리케이션에서의 신뢰성을 높이는 데 기여합니다.



### UrbanDiT: A Foundation Model for Open-World Urban Spatio-Temporal Learning (https://arxiv.org/abs/2411.12164)
- **What's New**: UrbanDiT는 복잡한 도시 환경의 시공간 동적을 효과적으로 모델링하기 위한 새로운 기초 모델로 소개됩니다. 이 모델은 다양한 시공간 데이터 소스와 유형을 통합하며, 여러 도시와 시나리오에서 보편적인 시공간 패턴을 학습할 수 있습니다. 또한, UrbanDiT는 데이터와 작업에 특화된 프롬프트(prompt)를 생성하는 혁신적인 프롬프트 학습(framework) 구조를 통해 다양한 도시 응용 프로그램에서 뛰어난 성능을 발휘합니다.

- **Technical Details**: UrbanDiT는 그리드 기반(grid-based) 및 그래프 기반(graph-based) 데이터와 같은 다양한 데이터 유형을 통합하여 시퀀스 형식으로 변환합니다. 이 모델은 양방향(spatio-temporal) 예측, 시간 보간(temporal interpolation), 공간 외삽(spatial extrapolation), 시공간 보간(spatio-temporal imputation)와 같은 다양한 작업을 지원하기 위해 마스킹 전략과 작업 특화 프롬프트를 사용합니다. 또한, 이 모델은 오픈 월드 시나리오(open-world scenarios)에 효과적으로 일반화(generalize)되며, 제로샷(zero-shot) 능력으로 훈련 데이터와 비교해 더 뛰어난 성능을 보입니다.

- **Performance Highlights**: UrbanDiT는 교통 상황(transportation traffic), 인구 흐름(crowd flows), 택시 수요(taxi demand), 자전거 사용(bike usage), 그리고 셀룰러 트래픽(cellular traffic) 등 다양한 분야에서 주목할 만한 성능을 달성합니다. 이 모델은 여러 도시와 작업에서 최신 최첨단 성능(state-of-the-art performance)을 기록하며, 도시 시공간 분야의 기초 모델에 대한 새로운 기준을 세웁니다.



### HNCSE: Advancing Sentence Embeddings via Hybrid Contrastive Learning with Hard Negatives (https://arxiv.org/abs/2411.12156)
- **What's New**: 본 논문에서는 HNCSE(Hard Negative Contrastive Sentence Learning)라는 새로운 대조 학습 프레임워크를 제안합니다. 이 접근법은 기존 SimCSE를 확장하여 긍정 및 부정 샘플 모두의 학습을 향상시키는 데 중점을 두고 있습니다. HNCSE의 주요 특징은 하드 네거티브 샘플을 효과적으로 사용하여 문장의 의미적 깊이를 더하는 것입니다. 실험 결과, HNCSE는 의미 텍스트 유사성(semantic textual similarity) 및 전이 작업(transfer task) 데이터셋에서 뛰어난 성능을 보였습니다.

- **Technical Details**: HNCSE모델은 하드 네거티브 샘플 믹싱 기법을 바탕으로 문장 표현 학습을 개선하는 혁신적인 훈련 방식을 적용합니다. 이 모델은 하드 네거티브의 특성을 포함하여 잘못 식별된 긍정 샘플을 수정하고, 기존 하드 네거티브를 늘려 더 튼튼한 학습 상황을 조성합니다. 대조 학습 목표는 동일한 문장에 대한 두 가지 다른 뷰 간의 일치를 극대화하고, 서로 다른 문장 간의 일치를 최소화하는 것을 기반으로 합니다. 이는 라벨이 없는 데이터 환경에서 문장 표현을 학습하는 효과적인 방법입니다.

- **Performance Highlights**: HNCSE는 SimCSE를 기반으로 한 다양한 작업에 대해 개선된 성능을 보여주며, 하드 네거티브 샘플을 믹싱하여 긍정 샘플의 품질을 향상시키고 있습니다. HNCSE는 대규모 언어 모델(LLMs) 및 현재의 SOTA(State-Of-The-Art) 벤치마크에 비해 의미 텍스트 유사성 작업에서 뚜렷한 장점을 발휘합니다. 이러한 성과는 문장 표현 학습에 있어 하드 네거티브의 중요성을 강조하며, 더 나은 의미적 이해에 기여하고 있습니다.



### Reinforcement Learning with Action Sequence for Data-Efficient Robot Learning (https://arxiv.org/abs/2411.12155)
Comments:
          17 Pages. Website: this https URL

- **What's New**: 이 논문은 로봇 작업에서 강화 학습(RL)의 데이터 효율성을 향상시키기 위한 새로운 RL 알고리즘을 제시합니다. 이 알고리즘은 일련의 행동에 대한 Q값을 출력하는 비평자 네트워크(critic network)를 학습하여, 불규칙한 데이터에서 유용한 가치 함수(value function)를 학습할 수 있게 합니다. 최근의 행동 복제(behavior-cloning, BC) 접근 방식에서 영감을 받아, 불확실한 전문가의 시연을 효과적으로 근사할 수 있는 방법을 조사합니다.

- **Technical Details**: 이 논문에서 제안한 알고리즘은 CQN-AS(Coarse-to-fine Q-Network with Action Sequence)라고 불립니다. 이 알고리즘은 현재와 미래의 행동 시퀀스를 실행할 때의 결과를 명시적으로 학습해, RL 에이전트가 noisy trajectory에서 유용한 가치 함수를 학습할 수 있도록 돕습니다. 실험은 BiGym, HumanoidBench 및 RLBench 등 다양한 설정에서 진행되었으며, 희소 및 밀집 보상(sparse and dense rewards) 환경에서의 성능을 평가했습니다.

- **Performance Highlights**: CQN-AS는 여러 실험에서 이전의 RL 알고리즘 및 BC 기준선을 초과하는 성능을 보였습니다. 특히 인간이 수집한 데이터가 포함된 모바일 양손 조작 과제와, 밀집 보상이 주어지는 사람 모양 제어 과제에서 두드러진 성과를 기록했습니다. 또한, RLBench에서는 합성 데이터가 사용되었으나, 여전히 많은 작업에서 우수한 성능을 달성했습니다.



### HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments (https://arxiv.org/abs/2411.12150)
- **What's New**: 이번 논문에서는 복잡한 환경에서 로봇 내비게이션을 위한 새로운 프레임워크를 제안합니다. 특히, 다양한 상호작용을 모델링하는 이질적 시공간 그래프(heterogeneous spatio-temporal graph)를 활용하여 로봇이 장애물 및 인간과 충돌하지 않도록 내비게이션 정책을 학습하는 방법을 소개합니다. HEIGHT라는 네트워크 아키텍처를 통해 중요한 상호작용을 우선시하고 동적 장면의 변화를 추적할 수 있는 메커니즘을 제공합니다.

- **Technical Details**: HEIGHT는 서로 다른 상호작용을 포착하기 위해 두 개의 멀티헤드 주의 네트워크(multi-head attention networks)를 사용하며, 단일 방향의 장애물-행위자(interaction) 상호작용을 모델링하기 위해 MLP(Multi-Layer Perceptron)를 활용합니다. 또한, 시퀀스의 시간적 진화를 고려하는 순환 네트워크(recurrent network)를 도입하여 로봇이 동적 환경에서 적응적인 내비게이션을 수행할 수 있도록 지원합니다. 이러한 접근은 로봇이 인간과 장애물 간의 복잡한 상호작용을 효과적으로 관리하는 데 도움을 줍니다.

- **Performance Highlights**: HEIGHT의 성능은 다양한 시뮬레이션과 실제 실험을 통해 입증되었습니다. 특히, 인간과 장애물 밀도가 변화할 때 더 나은 제로샷 제너럴리제이션(zero-shot generalization) 능력을 보여주며, 과거의 최첨단 방법들을 초월하는 내비게이션 성공률과 효율성을 기록했습니다. 이 방식은 내구성 있는 로봇 정책을 저비용 시뮬레이터에서 학습하여 실제 복잡한 환경에서도 활용 가능한 가능성을 제시합니다.



### A Computational Method for Measuring "Open Codes" in Qualitative Analysis (https://arxiv.org/abs/2411.12142)
- **What's New**: 이 논문에서는 질적 분석(valitative analysis)의 중요한 방법론인 open coding의 잠재적인 편향을 체계적으로 측정하고 식별하는 새로운 계산 방법을 제안합니다. 기존 연구들은 open coding의 결과를 정확하게 측정하지 않아 편향의 위험을 증가시켜 왔습니다. 이 방법은 Grounded Theory와 Thematic Analysis 이론에 기반하여 인간과 기계 코더 간의 팀 기반 접근 방식을 활용합니다.

- **Technical Details**: 제안된 방법은 두 가지 HCI 데이터셋을 사용하여 open 코드의 신뢰성을 측정하는 것입니다. 이 과정에서 Coverage, Density, Novelty, Divergence라는 네 가지 개념 지표를 운영화하여 코드 스페이스를 평가합니다. 이러한 지표는 팀 코더의 결과에 대한 개별 코더의 결과를 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 이 연구는 인간 분석과 비교하여 제안된 방법의 결과의 안정성을 통계적으로 분석함으로써 신뢰성을 검증합니다. 기계 측정과 인간 해석을 결합한 결과, 질적 연구자들이 GAI를 유도 분석에서 사용하는 데 도움이 될 수 있는 근거 기반의 제안과 예제 워크플로우를 제시합니다.



### Visualizing Loss Functions as Topological Landscape Profiles (https://arxiv.org/abs/2411.12136)
- **What's New**: 이 논문은 기계 학습에서 손실 함수(loss function)의 시각화를 위한 새로운 방법론을 제안합니다. 기존의 방법들은 일반적으로 단일 또는 이차원 방향으로만 샘플링하는 반면, 본 연구에서는 위상적 데이터 분석(topological data analysis, TDA)을 기반으로 한 더 높은 차원의 손실 풍경(loss landscape)을 시각화할 수 있는 새로운 표현 방식을 도입합니다. 이 접근 방식은 손실 풍경이 모델 성능 및 학습 역학을 어떻게 반영하는지를 새로운 방식으로 탐구합니다.

- **Technical Details**: 위상적 데이터 분석(TDA)의 개념을 활용하여, 손실 풍경의 중요한 특징을 포착하는 머지 트리(merge tree)를 사용합니다. 이 트리는 손실 풍경의 임계점을 인코딩하고, 이를 두 차원으로 재표현하는 위상적 풍경 프로파일(topological landscape profile)로 나타냅니다. 이를 통해 연구자들은 손실 함수의 형태를 더 깊이 이해할 수 있으며, 특히 다차원 공간에서의 복잡한 정보를 효과적으로 시각화할 수 있습니다.

- **Performance Highlights**: 실험을 통해, 잘 작동하는 모델의 손실 풍경은 상대적으로 간단한 위상을 가지며, 낮은 성능에서 높은 성능으로 전환되는 지점 근처에서 위상적 변동성이 더 큼을 발견했습니다. 또한, 비슷한 물리적 매개변수를 통해 저오차(low error) 및 고오차(high error) 모델의 손실 풍경 모양의 차이를 관찰했습니다. 이러한 발견은 모델의 성능에 대한 새로운 통찰력을 제공하며, 특히 다양한 하이퍼파라미터에 따른 손실 풍경의 변화를 명확하게 제시합니다.



### Distill the Best, Ignore the Rest: Improving Dataset Distillation with Loss-Value-Based Pruning (https://arxiv.org/abs/2411.12115)
- **What's New**: 최근 데이터셋 증류(dataset distillation) 방식이 새롭게 주목받고 있으나, 기존 방법들은 전체 데이터셋에서 비유용한 샘플을 포함하는 경우가 많았습니다. 본 논문에서는 'Prune First, Distill After'라는 새로운 프레임워크를 도입하여, 증류 전에 손실 기반 샘플링 방법으로 데이터셋을 체계적으로 가지치기(pruning)합니다. 이를 통해 미지의 아키텍처에 대한 일반화 능력을 향상시키는 대표적인 코어 세트를 생성합니다.

- **Technical Details**: 우리의 접근법은 손실 값에 기반한 샘플링 전략을 사용하는데, 이는 사전 훈련된 분류기 모델을 활용하여 데이터 샘플을 '분류 난이도(classification difficulty)'에 따라 순위 매기는 방식입니다. 이 과정에서 단순한 샘플과 복잡한 샘플을 각각 먼저 선택하는 두 가지 샘플링 전략을 비교하였으며, 단순 샘플에 집중할 경우 증류 품질이 크게 향상되는 것을 발견했습니다. 또한, 우리는 StyleGAN-XL 및 수정된 디퓨전 모델을 포함한 최신 데이터셋 증류 기법들을 기반으로 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 원래 데이터셋의 80%를 제거하고도 증류된 데이터셋에서 최대 5.2%의 정확도 향상이 이루어졌습니다. 우리의 접근법은 여러 ImageNet 하위 집합에서 다양한 아키텍처를 대상으로 한 광범위한 평가를 통해 유연성과 강건성을 입증했습니다. 이러한 성과는 데이터셋 증류의 효과성을 높이는 가능성을 제시하며, 더 나은 품질의 데이터셋을 생성하는 데 기여합니다.



### Just Leaf It: Accelerating Diffusion Classifiers with Hierarchical Class Pruning (https://arxiv.org/abs/2411.12073)
- **What's New**: 최근의 연구에 따르면, 전통적인 생성 모델이 단순한 콘텐츠 생성에만 국한되지 않고 분류 작업에서도 여전히 활용 가능함을 보여주고 있습니다. 특히, Hierarchical Diffusion Classifier(HDC)는 고유한 계층적 레이블 구조를 활용하여 높은 계산 비용을 줄이면서 이미지 분류 효율성을 향상시키고 있습니다. 이 접근 방식을 통해 HDC는 최대 60%의 속도 향상을 달성하면서도 분류 정확도를 유지하거나 개선할 수 있음을 증명했습니다.

- **Technical Details**: HDC는 전통적인 확산 모델을 기반으로 하며, 레이블 트리를 계층적으로 탐색하여 불필요한 높은 수준의 범주를 점진적으로 제거합니다. 초기 단계에서는 가장 유망한 synsets를 유지하기 위해 레이블 트리를 레벨별로 트래버스하고, 그 후 남은 후보 리프 노드에서 전통적인 확산 분류를 수행합니다. 이로 인해, 불필요한 클래스를 사전에 제거함으로써 계산 비용을 줄이고 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: HDC는 대규모 이미지 분류 작업인 ImageNet-1K에서 약 60%의 빠른 추론 시간을 달성하며, 비슷한 계산 시간에 기존의 확산 분류 모델보다 더 나은 정확도(65.16% vs. 64.90%)를 기록하였습니다. 이를 통해 HDC는 분류 빠르기와 정확도 사이의 새로운 균형을 제공하고, 실질적인 대규모 분류 작업에서도 효과적으로 활용될 수 있는 가능성을 제시합니다.



### Zoomed In, Diffused Out: Towards Local Degradation-Aware Multi-Diffusion for Extreme Image Super-Resolution (https://arxiv.org/abs/2411.12072)
- **What's New**: 기존의 Text-to-Image (T2I) 확산 모델은 512x512 해상도로 제한되어 있었으나, 본 연구에서는 추가 학습 없이 2K, 4K, 심지어 8K 해상도로 이미지를 생성할 수 있는 새로운 접근 방식을 소개합니다. 이 방법은 MultiDiffusion과 지역 손실 인식 프롬프트 추출이라는 두 가지 핵심 요소를 활용하여 고해상도 이미지를 생성하면서도 전 세계적으로 일관성을 유지합니다. 이러한 혁신은 이미지 초해상도(Super-Resolution, SR) 작업에 T2I 확산 모델을 적용하는 새로운 가능성을 제공합니다.

- **Technical Details**: 이 연구의 방법론인 MultiDiffusion은 이미지를 생성하는 과정을 여러 개의 확산 경로에 분산시켜 높은 해상도에서도 전 세계적인 일관성을 보장합니다. 각 단계에서 잠재 피처 맵은 중첩되는 타일로 나누어져 개별적인 확산 과정을 거치며, 이로 인해 인접한 타일 간의 정보를 공유하여 전반적인 구조와 지역 세부 사항의 일관성을 유지합니다. 이러한 과정은 기존 T2I 확산 모델에 비해 512×512 픽셀의 제한 없이 2K 이상의 해상도를 가능하게 합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존의 모델과 비교했을 때 방향성과 세부 사항을 보다 잘 복원할 수 있는 능력을 갖추고 있습니다. 결과적으로, 초해상도의 새로운 기준을 세우며, T2I 모델이 가진 잠재력을 극대화하여 다양한 해상도의 이미지 생성에 효과적으로 활용될 수 있음을 보여주었습니다. 모델의 성능은 2K, 4K, 8K에서의 해상도 증대에 성공적으로 적용되면서 향후 SR 작업의 가능성을 더욱 확장시킵니다.



### Benchmarking pre-trained text embedding models in aligning built asset information (https://arxiv.org/abs/2411.12056)
- **What's New**: 이 논문은 구축 자산(data related to built assets) 정보를 전문 용어에 맞춰 잘 정렬하기 위한 새로운 접근 방식을 제안합니다. 특히 최근의 대형 언어 모델을 활용한 텍스트 임베딩(text embedding)이 자산 관리의 데이터 매핑(data mapping) 과정의 자동화를 모색하는 데 기여할 수 있음을 보여줍니다. 이 연구는 기존 모델의 성능을 비교하고, 구축 자산에 특화된 기술 용어의 복잡한 의미를 효과적으로 표현할 수 있는지를 평가합니다.

- **Technical Details**: 이 연구는 다양한 하위 도메인에 걸쳐 구축 제품에 대한 정보의 체계적인 데이터를 구성하고, 이를 기반으로 구체적 작업 세트를 개발합니다. 데이터는 산업 기초 클래스(Industry Foundation Classes, IFC)와 유니클래스(Uniclass)의 두 가지 주요 출처에서 수집되었습니다. 또, 제안된 데이터 세트는 여섯 가지 작업, 즉 클러스터링(clustering), 검색(retrieval), 재순위화(reranking)를 평가하여 모델간의 성능을 비교하는 데 중점을 둡니다.

- **Performance Highlights**: 이 연구의 평가 결과는 자동화된 데이터 매핑 과정에서 구성 자산 정보를 정렬하는 데 있어 기존 언어 모델의 강점과 한계를 안내합니다. 현재 관련된 24개의 텍스트 임베딩 모델을 사용하여 10,000개 이상의 데이터 항목을 포함하는 가장 포괄적인 벤치마크를 제공합니다. 또한 연구 결과 및 데이터 세트는 공개 소스로 제공되어 후속 연구의 기초 자료로 사용될 수 있도록 하였습니다.



### Fingerprinting and Tracing Shadows: The Development and Impact of Browser Fingerprinting on Digital Privacy (https://arxiv.org/abs/2411.12045)
Comments:
          SECURWARE 2024, France, Nice

- **What's New**: 이번 논문은 브라우저 지문 인식(browser fingerprinting)의 다양한 기법을 살펴보며, 수집된 데이터의 엔트로피(entropy)와 고유성(uniqueness)을 분석하여 온라인에서 사용자 추적이 지니는 복잡한 기술적 및 프라이버시 관점의 도전을 강조합니다. 사용자가 자신의 데이터 수집 및 사용에 대해 통제할 수 없는 경우가 많아 큰 프라이버시 우려를 초래합니다. 명시적 사용자 동의가 필요한 쿠키(cookies)와는 달리, 브라우저 지문 인식 기술은 백그라운드에서 사용자에게 불분명한 방식으로 사용되며, 이러한 점이 새로운 개인 정보 보호 문제를 야기합니다.

- **Technical Details**: 브라우저 지문 인식은 사용자의 브라우저가 자신에 대해 직접 또는 간접적으로 드러내는 특성 정보를 수집하는 것을 의미하며, IT 보안 및 사기 탐지 등 다양한 응용 분야에서 사용됩니다. 이 기술은 활성 데이터 전송이 필요 없으며, 페이지 로딩 시 HTTP 헤더를 통해 다양한 정보가 전송되는 방식으로 이루어집니다. 수집된 정보는 배경에서 이루어져, 쿠키 삭제와 같은 방법으로 쉽게 새로운 신원을 생성할 수 없습니다.

- **Performance Highlights**: 최근 연구에서 알렉사 상위 10만 개 웹사이트의 거의 10%가 지문 생성을 위한 스크립트를 사용하고 있다는 결과가 나타났습니다. 이는 2014년의 비슷한 연구와 비교할 때 사용률이 거의 두 배로 증가했다는 것을 보여주며, 브라우저 지문 인식이 쿠키보다 덜 감지된다는 점에서 온라인 추적에 대한 보다 큰 위험 요소로 급부상하고 있음을 나타냅니다. 이러한 기술은 사용자에게 감지되지 않으며, 간단한 방법으로 바꾸거나 삭제할 수 없다는 문제가 있습니다.



### Fast Convergence of Softmax Policy Mirror Ascen (https://arxiv.org/abs/2411.12042)
- **What's New**: 이 논문에서는 정책 경량화 기법인 Softmax Policy Mirror Ascent(SPMA)를 제안하고 있습니다. 기존의 NPG(자연 정책 기울기) 알고리즘을 기반으로 하여, 액션 사이의 정규화 필요성을 제거한 개선된 형태입니다. SPMA는 선형 수렴률을 가지며, 기존의 소프트맥스 정책 기울기(smooth policy gradient)보다 더 빠른 수렴을 달성하는 것으로 입증되었습니다.

- **Technical Details**: SPMA는 다중 무장 밴딧(multi-armed bandit)과 테이블 형태의 MDP에 대해 개발되었습니다. 정책을 로그 확률의 이중 공간에서 미러 상승(mirror ascent) 업데이트를 통해 최적의 정책으로 수렴합니다. 또한, 상태-행동 공간이 큰 MDP를 처리하기 위한 함수 근사(function approximation) 기법을 적용하며, 이는 복잡한 비선형 함수 근사에도 확장됩니다.

- **Performance Highlights**: SPMA는 Atari와 MuJoCo와 같은 벤치마크에서 실험적으로 평가되어, 기존 알고리즘인 MDPO, PPO, TRPO보다 유사하거나 더 나은 성능을 달성하는 것으로 나타났습니다. 특히, Atari 게임에서 SPMA는 TRPO와 PPO보다 나은 성과를 보였으며, MuJoCo 작업에서는 PPO를 초과하는 성과를 기록하였습니다.



### Scaling Deep Learning Research with Kubernetes on the NRP Nautilus HyperCluster (https://arxiv.org/abs/2411.12038)
- **What's New**: 이 연구에서는 NRP Nautilus HyperCluster를 활용하여 깊은 신경망(DCNN)의 모델 훈련을 자동화하고 확장하는 새로운 접근 방식을 탐구하고 있습니다. 이를 통해 소실 감지, 화재 피해 지역 세분화 및 삼림 파괴 탐지의 세 가지 응용 분야에 대한 연구를 수행했습니다. Nautilus는 1,300개 이상의 NVIDIA GPU와 19,000개 CPU 코어를 갖춘 Kubernetes 클러스터로, 현재까지 총 4,040시간의 훈련 시간을 기록했습니다.

- **Technical Details**: 모델 훈련에 있어 DCNN의 컴퓨팅 요구량이 증가하였으며, 이는 과학적 연구에서 병목 현상이 되고 있습니다. 연구에서는 DCNN 기반의 모델을 훈련하기 위해 Nautilus의 강력한 하드웨어를 활용하여 다양한 응용 프로그램을 개발하였습니다. 특성 추출 및 탐지 기법을 통해 transformer 아키텍처의 성능을 평가하기 위해 수많은 아키텍처를 훈련해야 할 필요성이 있습니다.

- **Performance Highlights**: 이번 연구의 주요 성과로는 세 가지 응용 분야에서 DCNN을 활용한 234개의 모델이 훈련되었으며, 3,000 GPU 시간 이상의 컴퓨팅 자원이 소모되었습니다. DCNN의 변화를 반영하여, 다양한 훈련 데이터셋과 함께 각각의 아키텍처에 대한 비교 분석이 이루어졌습니다. 연구 결과는 대량의 위성 이미지를 효과적으로 처리하고 다양한 실제 응용 프로그램을 발전시키는 데 기여할 것으로 전망됩니다.



### ByteScience: Bridging Unstructured Scientific Literature and Structured Data with Auto Fine-tuned Large Language Model in Token Granularity (https://arxiv.org/abs/2411.12000)
- **What's New**: ByteScience는 과학 데이터를 자동으로 정리하기 위한 비영리 클라우드 기반 플랫폼으로, 최신의 자동 미세 조정(automatic fine-tuning) 언어 모델(LLM)을 활용합니다. 이 플랫폼은 Amazon Web Services(AWS) 위에 구축되어 간편한 사용자 UI를 제공하며, 몇 개의 잘 주석이 달린 논문으로도 높은 정확도를 달성할 수 있습니다. 이 혁신적인 도구는 과학 문헌에서 구조화된 지식 및 데이터를 추출하는 과정을 간소화하여 자연 정보학(natural informatics)의 발전에 기여하고자 합니다.

- **Technical Details**: ByteScience는 AWS Sagemaker를 활용하여 강력하고 확장 가능한 클라우드 기반 솔루션을 제공합니다. 이 시스템은 초기 설정에서 도메인 별 데이터셋 구성을 통해 과학적 데이터를 수집하고, 해당 데이터셋에 대해 대규모 언어 모델을 미세 조정하여 성능을 최적화합니다. 사용자는 JSON, PDF, HTML 또는 XML 형식의 과학 문서를 업로드하고, 이를 통해 구조화된 데이터를 생성할 수 있는 간편한 워크플로우를 경험할 수 있습니다.

- **Performance Highlights**: ByteScience 플랫폼은 LLM을 활용해 사람의 개입을 줄이며 주석 달기 시간을 평균 57% 단축시키는 효과를 보이고 있습니다. 연구 결과, ByteScience는 배터리, 촉매 및 태양광 분야의 90개 샘플을 분석한 결과, 기존 방법보다 높은 정밀도를 기록하며 구조화된 데이터 추출 성능에서 탁월함을 나타냈습니다. 데이터 추출 성공률이 증명되며, Thomas와 같은 연구자들이 이 플랫폼을 활용하여 복잡한 과학 논문을 대규모로 처리할 수 있는 사례를 보여주고 있습니다.



### Understanding Chain-of-Thought in LLMs through Information Theory (https://arxiv.org/abs/2411.11984)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)에서 체인 오브 생각(Chain-of-Thought, CoT) 추론을 공식화하는 새로운 정보를 제공합니다. 기존의 CoT 평가 방법들이 주석 데이터에 의존하던 반면, 저자들은 정보 수치 이론(information theory)을 기반으로 한 접근법을 제안합니다. 이 방법은 각 추론 단계에서 '정보 이득(information gain)'을 측정하여, LLM에서의 실패 유형을 식별할 수 있게 합니다.

- **Technical Details**: 제안된 프레임워크는 LLM 생성 접근법을 정보 이론의 관점에서 기술합니다. 먼저, 초기 상태와 작업을 정의하고 이를 통해 업데이트된 상태를 설명합니다. 이 프레임워크는 각 추론 단계에서 정보 이득을 정량화하며, 이는 적절한 정보가 최종 결과 예측에 기여해야 한다는 인식에 기반합니다. 이를 통해 주석 데이터 없이도 각 하위 작업의 성능을 평가할 수 있는 알고리즘을 제시합니다.

- **Performance Highlights**: 제안된 방법은 Toy 데이터 및 GSM-8K 데이터 세트를 통해 폭넓은 실험을 거쳐 효과성을 입증했습니다. 이메일 내의 진행 과정에서 기존의 결과 기반 방법들에 비해 정확한 모델 성능을 제공하여, CoT 추론의 실패 모드를 효과적으로 식별합니다. 최종적으로, 이 연구는 LLM의 성능 평가 방식에서 중요한 변화를 예고하며, 연구자들에게 더 나은 인사이트를 제공합니다.



### Medical Video Generation for Disease Progression Simulation (https://arxiv.org/abs/2411.11943)
Comments:
          Tech Report. The appendix will release soon. arXiv admin note: text overlap with arXiv:2309.11745

- **What's New**: 이 논문에서는 질병 진행 상황을 시뮬레이션할 수 있는 첫 번째 Medical Video Generation (MVG) 프레임워크를 제안하고 있습니다. 이를 통해 질병과 관련된 이미지 및 비디오 특징을 조작할 수 있어 정확하고 개인화된 시뮬레이션을 가능하게 합니다. MVG는 의료 영상의 데이터 부족 문제를 해결하고, 의료 서비스 제공자들이 효율적인 치료 전략을 수립하는 데 도움을 줄 수 있습니다.

- **Technical Details**: MVG 프레임워크는 GPT-4를 사용하여 환자의 임상 보고서를 요약하고, 그에 맞는 텍스트 인퍼런스를 이용해 질병 관련 특징을 점진적으로 제어할 수 있도록 설계되었습니다. 이 과정에서 노이즈 제거 확산 확률 모델의 가역성(invertibility) 및 맥락 부호기(context encoder)의 시각적 언어 정렬 능력을 활용하여 질병 진행 상황을 시뮬레이션합니다. 또한, 이론적으로는 다단계 질병 상태 시뮬레이션 모듈이 주어진 텍스트 조건의 로그 가능성을 극대화하기 위한 경량 감소(gradient descent) 과정으로 이해될 수 있습니다.

- **Performance Highlights**: MVG는 세 가지 의료 영상 도메인에서 기존 모델들에 비해 월등한 성능을 보여주었습니다. 의사들에 의한 사용자 연구에서는 76.2%의 질병 상태 시뮬레이션 결과가 임상적 맥락과 밀접하게 일치한다고 평가되었습니다. 이러한 결과는 MVG가 질병 진행 경과를 예측하는 데 있어 효과적임을 입증하며, 의료 교육 및 데이터 보완에도 큰 기여를 할 것으로 기대됩니다.



### Variable Rate Neural Compression for Sparse Detector Data (https://arxiv.org/abs/2411.11942)
Comments:
          37 pages, 12 figures, submitted to Journal of Computational Physics

- **What's New**: 본 논문은 고에너지 입자 충돌 실험에서 데이터를 실시간으로 고효율로 압축하는 새로운 방법론인 BCAE-VS를 제안합니다. 기존의 데이터 압축 방법론들이 희소 데이터 처리에 한계를 보였던 반면, BCAE-VS는 가변 압축 비율을 통해 압축 효율성을 크게 향상시킵니다. 이를 통해 실험의 데이터 출력이 증가하는 동시에 정보 손실을 최소화할 수 있습니다.

- **Technical Details**: BCAE-VS는 이중 자동 인코더 구조를 기반으로 하여, TPC 데이터에서의 신호 중요도를 평가하여 핵심 포인트를 선택적으로 저장함으로써 압축을 수행합니다. 또한, 이 모델은 희소 합성을 활용하기 위해 희소 컨볼루션을 적용하여 입력 데이터의 희소성과 복잡성에 따라 처리 속도를 가변적으로 조정할 수 있습니다. 이로 인해 특정 신호에 대해서만 출력을 생성하고, 모든 제로 오퍼랜드에 대한 행렬 곱셈을 회피하여 효율성을 높입니다.

- **Performance Highlights**: BCAE-VS는 기존 최첨단 BCAE 모델에 비해 75% 향상된 재구축 정확도를 달성하며, 압축 비율 또한 평균 10% 증가했습니다. 추가로, 모델 크기는 두 배 이상의 양으로 줄여서 컴퓨팅 리소스의 낭비를 최소화했습니다. 이러한 성과는 실험적으로 검증되어 데이터의 희소성이 증가할수록 모델의 처리량이 증가하는 결과를 가져왔습니다.



### Newclid: A User-Friendly Replacement for AlphaGeometry (https://arxiv.org/abs/2411.11938)
Comments:
          51 pages

- **What's New**: 이번 연구에서는 Newclid라는 새로운 기하학적(symbolic) 문제 해결기를 소개합니다. 이 해결기는 AlphaGeometry를 기반으로 하며, 사용자와 프로그래머 모두에게 더 친숙한 기능을 제공합니다. 특히 새로운 명령줄 인터페이스(CLI)와 GeoGebra와의 입력 호환성을 통해 교육적 맥락에서도 접근성을 향상시켰습니다.

- **Technical Details**: Newclid의 중심에는 DDARN이라는 기호적(symbolic) 해결기가 있으며, 이는 이전 AlphaGeometry의 DDAR 시스템을 개선하여 모듈화된 코드베이스와 더 나은 디버깅, 시각화 도구를 제공합니다. DDARN은 에이전트가 내부 추론을 조정할 수 있도록 유연성을 제공하며, 기하학적 개념(길이, 각도)을 더 잘 이해하고 피타고라스의 정리와 같은 정리를 사용하여 증명할 수 있는 능력이 향상되었습니다.

- **Performance Highlights**: AlphaGeometry의 이전에 해결하지 못했던 AG-30 데이터셋의 문제 다섯 가지를 DDARN을 사용하여 재평가한 결과, 새로운 해결기가 기존보다 하나의 문제를 추가로 해결할 수 있음을 확인했습니다. Newclid는 기하학적 문제 해결의 정확성과 확장성을 크게 향상시키며, 향후 향상된 대형 언어 모델(LLM)도 통합할 계획입니다.



### Value Imprint: A Technique for Auditing the Human Values Embedded in RLHF Datasets (https://arxiv.org/abs/2411.11937)
- **What's New**: 이 논문에서는 RLHF(Reinforcement Learning From Human Feedback) 데이터셋에 내재된 인간의 가치들을 감사(audit)하고 분류(classify)하기 위한 새로운 프레임워크인 Value Imprint를 소개합니다. 연구자들은 이 프레임워크의 유효성을 검토하기 위해 세 가지 사례 연구를 수행하며, Anthropic/hh-rlhf, OpenAI WebGPT Comparisons, Alpaca GPT-4-LLM 데이터셋을 분석하였습니다. 이를 통해 데이터셋에 내재된 가치들이 사람들의 가치관과 어떻게 다를 수 있는지를 탐구했습니다.

- **Technical Details**: 저자들은 철학, 가치론(axiology), 윤리학 등 과거 연구로부터 통합된 문헌검토를 통해 인간 가치의 세목을 발전시키고, 이를 통해 6,501개의 RLHF 선호 사항을 주석(annotation)하였습니다. 두 번째 단계에서는 주석 처리된 데이터를 기반으로 변환기(transformer) 기반의 머신러닝 모델을 훈련시켜 세 가지 RLHF 데이터셋을 감사하고 분류하는 데 사용하였습니다. 이 분석을 통해 정보 유틸리티 정보(Information Seeking, Wisdom/Knowledge) 가치가 가장 지배적임을 발견했습니다.

- **Performance Highlights**: 모델의 분류 정확도는 약 80%에 달하며, 이는 AI 연구자들이 RLHF 데이터셋에 내재된 인간 가치를 검토하는 데 이 과정을 채택할 수 있음을 보여줍니다. 연구 결과, RLHF 선호사항에서 구 civility & tolerance, empathy & helpfulness, justice & human rights/animal rights, well-being & peace 등은 가장 적게 나타났습니다. 이러한 발견은 언어 모델이 사회적 가치 및 규범에 부합하도록 개발될 수 있도록 중요한 통찰력을 제공합니다.



### METEOR: Evolutionary Journey of Large Language Models from Guidance to Self-Growth (https://arxiv.org/abs/2411.11933)
- **What's New**: 본 논문에서는 METEOR라는 새로운 자기 진화 방법론을 제안합니다. 이는 LLM이 단계적으로 전문 지식을 습득하고 자율적으로 발전할 수 있도록 돕기 위한 구체적인 훈련 프레임워크를 제공합니다. 특히, 이 방법은 데이터 증류(knowledge distillation) 및 자기 학습 방식으로 모델의 도메인 전문성을 향상시킵니다.

- **Technical Details**: METEOR 방법론은 두 가지 훈련 단계와 자기 진화 전략으로 구성됩니다. 첫 번째 단계에서는 약한 모델(weak model)이 강한 모델(strong model)로부터 도메인 지식을 증류하여 기초적인 전문성을 부여받습니다. 이어지는 단계에서는 강한 모델의 피드백을 통해 모델의 도메인 지식이 점진적으로 강화됩니다.

- **Performance Highlights**: 실험 결과, METEOR는 도메인-specific 과제에서 모델의 정확도(accuracy), 완전성(completeness), 관련성(relevance), 일관성(coherence) 및 신뢰성(reliability)을 크게 향상시키는 것으로 나타났습니다. 각 단계가 모델 성능에 긍정적인 영향을 미친다는 점이 입증되었습니다.



### Reviving Dormant Memories: Investigating Catastrophic Forgetting in Language Models through Rationale-Guidance Difficulty (https://arxiv.org/abs/2411.11932)
Comments:
          Working in progress

- **What's New**: 이 논문에서는 지속적 학습에서의 재해적 망각(catastrophic forgetting) 문제를 해결하기 위해, 모델이 제공된 부분적인 합리적 근거(rationale)를 수동으로 수용할 때 잊혀진 과제에 대한 성능이 회복될 수 있음을 발견했습니다. 또한, 원래 지침에 과제 비특이적 접두사(task-agnostic prefix)를 추가함으로써 모델이 능동적으로 적절한 근거를 생성하여 정답에 도달할 수 있음을 보여주는 실험 결과를 제시했습니다.

- **Technical Details**: 저자들은 'Rationale-Guidance Difficulty' 메트릭을 제안하여 주어진 지침이 모델에 적절한 근거를 생성하도록 얼마나 효과적으로 안내하는지를 평가합니다. 이 메트릭을 활용하여 재생 기반 지속적 학습 알고리즘에서 재생 데이터의 할당을 최적화하는 방식을 적용했습니다. 실험 결과는 이 데이터 할당 방식이 재해적 망각을 효과적으로 완화하고 다양한 모델에서 더 나은 플라스틱성을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: 다양한 크기의 모델에 대한 실험에서, 적절한 근거의 부분을 제공하면 잊혀진 과제에 대한 모델의 성능이 회복됨을 확인했습니다. 게다가, 비특정 접두사를 추가함으로써 모델이 관련 지식을 생성하는 데 도움이 되었고, 잊혀진 과제에서의 성능이 부분적으로 복구되었습니다. 이러한 결과는 모델이 과제 관련 지식의 실질적인 손실이 아니라 원래 지침이 적절한 근거를 생성하는 데 실패한 것에서 주로 기인함을 증명합니다.



### AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning (https://arxiv.org/abs/2411.11930)
- **What's New**: 본 연구는 다중 모드 수학적 추론(multimodal mathematical reasoning)의 도전 과제에 접근하며, 멀티모달 대형 언어 모델(MLLMs)에 "느린 사고(slow thinking)" 능력을 통합한 AtomThink 프레임워크를 제안합니다. 기존 방법들이 빠른 사고를 바탕으로 하는 것과는 달리, 본 연구에서는 단계별로 구성된 사고의 긴 사슬(chain of thought, CoT)을 형성하여 MLLMs가 복잡한 추론을 수행할 수 있도록 돕습니다.

- **Technical Details**: AtomThink는 세 가지 주요 모듈로 구성됩니다: 고품질 CoT 주석을 자동 생성하는 CoT 주석 엔진, MLLM과 정책 보상 모델(policy reward model, PRM)을 결합하여 단계별 추론을 최적화하는 원자 단계 미세 조정 전략(atomic step fine-tuning strategy), 그리고 PRM과 함께 사용할 수 있는 네 가지 검색 전략(search strategies)입니다. 또한, AtomMATH라는 대규모 다중 모드 데이터세트를 제안하고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면 AtomThink는 기본 MLLM의 성능을 크게 향상시켜 MathVista에서 약 50%의 상대적 정확성 향상과 MathVerse에서 120%의 개선을 달성했습니다. AtomThink를 기반 모델로 사용할 경우 LLaVA-Llama3-8B의 정확도를 각각 9.6% 및 18.8% 향상시켰으며, MathVerse에서는 최고 정확도 40.5%를 기록하여 최첨단 GPT-4V를 초과했습니다.



### F$^3$OCUS -- Federated Finetuning of Vision-Language Foundation Models with Optimal Client Layer Updating Strategy via Multi-objective Meta-Heuristics (https://arxiv.org/abs/2411.11912)
- **What's New**: 이번 연구에서는 리소스가 제한된 클라이언트 장치에서 Vision-Language Models(VLMs)를 효과적으로 훈련하기 위한 Parameter-Efficient Fine-Tuning(PEFT) 전략을 제안합니다. 특히, 클라이언트별로 가장 중요한 VLM 레이어를 선택하는 	extit{client-specific layer importance score}와 클라이언트 간 다양한 레이어 선택을 장려하는 	extit{inter-client layer diversity score}의 두 가지 요소의 영향을 밝혀냅니다. 이를 통해 개인화된 모델 훈련을 지원하는 새로운 프레임워크인 F3OCUS를 도입하였습니다.

- **Technical Details**: F3OCUS는 레이어 선택을 개선하기 위해 클라이언트의 자원 제약을 고려하면서 지역적 및 전역적 FL 특성을 모두 반영하는 두 단계의 전략을 적용합니다. 첫 번째 단계에서 클라이언트 레벨 전략을 통해 LNTK(Neural Tangent Kernel)의 주 고유값을 기반으로 레이어 중요도를 정의하고, 두 번째 단계에서 서버 레벨 전략을 통해 클라이언트별 중요도를 극대화하고 레이어 선택의 분산을 최소화하여 균일한 레이어 참여를 촉진합니다. 연구에서 제안된 방법은 58개의 의료 이미지 데이터셋을 포함한 6가지 Vision-Language FL 작업 설정에서 10,000건 이상의 클라이언트 실험으로 그 효과를 입증했습니다.

- **Performance Highlights**: F3OCUS의 실험 결과, 다양한 FL 환경에서 여러 VLM에 대한 선택적 레이어 튜닝이 효과적으로 이루어짐을 확인했습니다. 본 연구는 인간의 판단력 저하 및 효율성을 가진 클라이언트들에게 적합한 다채롭고 동적인 레이어 선택 솔루션을 제공함으로써 빠른 수렴을 촉진합니다. 데이터, 모달리티, 작업 및 장치 이질성을 고려한 더 많은 제약을 반영하여 클라이언트 설정을 평가한 결과, 이전의 연구보다 향상된 성능을 보여주었습니다.



### ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling (https://arxiv.org/abs/2411.11911)
- **What's New**: 이번 연구에서는 다중 모드 비 예측(multi-modal motion prediction) 문제를 해결하기 위해 ModeSeq라는 새로운 예측 패러다임을 제안합니다. ModeSeq는 모드를 시퀀스(seqeunce)로 모델링하여 연속적으로 다음 모드를 추정하며, 이는 이전 모드 간의 상관관계를 보다 명확하게 캡처할 수 있게 해 줍니다. 또한 Early-Match-Take-All (EMTA) 훈련 전략을 도입하여 경로 다변량성을 더욱 향상시키는 효과를 거두었습니다.

- **Technical Details**: ModeSeq는 기존의 전통적인 다중 모드 예측 방식에서 벗어나, 모드를 한 번에 예측하는 대신 한 단계씩 순차적으로 예측합니다. 이 과정에서 모델은 각 예측 단계마다 이전 모드는 물론 그 신뢰도(confidence)까지 고려합니다. 이를 통해 각 모드 간의 관계를 명확히 파악할 수 있으며, 고속 모드 예측 및 후처리 단계를 필요로 하지 않고도 높은 품질의 경로 출력을 생성할 수 있습니다.

- **Performance Highlights**: ModeSeq는 Waymo Open Motion Dataset과 Argoverse 2 Motion Forecasting Dataset에서 여러 다른 다중 모드 예측 방법들과 비교하여 더 균형 잡힌 성능을 달성했습니다. 특히 모드 커버리지(mode coverage), 모드 스코어링(mode scoring), 그리고 경로 정확도(trajectory accuracy) 면에서 우수한 성능을 보였으며, 순차적인 모드 모델링 덕분에 높은 불확실성 환경에서도 다양한 행동 모드를 예측하는 능력을 자연스럽게 내재하고 있습니다.



### AIGS: Generating Science from AI-Powered Automated Falsification (https://arxiv.org/abs/2411.11910)
Comments:
          Pre-print. 35 pages. Official website: this https URL

- **What's New**: 본 논문에서는 $	extbf{AI-Generated Science}$ (AIGS)를 탐구합니다. 이는 에이전트가 독립적으로 연구 프로세스를 완전히 완료하고 과학 법칙을 발견하는 시스템을 의미합니다. 기존 시스템들이 검증 엔진에 크게 의존하는 한편, AIGS는 내재된 'falsification'을 통해 새로운 과학적 발견을 독립적으로 진행하는 방식으로 설계되었습니다.

- **Technical Details**: 우리는 Baby-AIGS라는 다중 에이전트 시스템을 제안하여 연구 프로세스의 핵심 역할을 담당하는 에이전트를 포함합니다. 이 시스템은 FalsificationAgent를 도입하여 가능성 있는 과학적 발견을 식별하고 검증함으로써 명시적인 'falsification'을 실현합니다. 이를 통해 독립적으로 연구를 수행할 수 있는 첫 걸음을 내딛습니다.

- **Performance Highlights**: 세 가지 작업에 대한 실험 결과, Baby-AIGS는 의미 있는 과학적 발견을 생성할 수 있음을 보여주었습니다. 그러나 경험이 풍부한 인간 연구자와 비교할 때 접근성은 여전히 낮은 상태입니다. 마지막으로, 현재 Baby-AIGS의 한계와 연구의 개선 가능성 및 관련 윤리적 문제에 대해 논의합니다.



### LLM4DS: Evaluating Large Language Models for Data Science Code Generation (https://arxiv.org/abs/2411.11908)
Comments:
          11 pages

- **What's New**: 이 논문은 Large Language Models (LLMs)의 코드 생성 효과성을 데이터 과학(Data Science) 문제를 다루는 연습을 통해 평가한 연구입니다. 연구진은 Microsoft Copilot, ChatGPT, Claude, Perplexity Labs의 AI 어시스턴트 네 가지 모델을 사용하여 데이터를 수집하고 분석하는 과정에서 이러한 모델들의 성능을 비교했습니다. 이를 통해 기존 LLM들의 평가가 데이터 과학의 특정 요구 사항을 충족하지 못하는 문제를 지적하고, 보다 체계적이고 집중된 평가의 필요성을 강조합니다.

- **Technical Details**: 이 논문에서는 Data Science 작업의 코드 생성을 위해 Stratascratch 플랫폼에서 출처를 정한 100개의 Python 문제를 사용하여, 세 가지 난이도(쉬움, 중간, 어려움)와 세 가지 유형(Analytical, Algorithm, Visualization)별로 LLM 모델의 성과를 분석했습니다. 효과성을 평가하기 위해 Success Rate, Efficiency, Quality of Output 및 Consistency와 같은 네 가지 주요 지표를 설정하였으며, 이를 기반으로 각 모델이 생성한 코드의 정확성과 효율성을 평가했습니다. 특히, Goal-Question-Metric (GQM) 접근 방식을 사용하여 실험을 설계했습니다.

- **Performance Highlights**: 실험 결과, 모든 모델이 50% 이상의 성공률을 기록하여 무작위 결과를 초과하는 성능을 보여주었습니다. 그러나 ChatGPT와 Claude만이 60% 이상의 성공률을 기록했으며, 모든 모델이 70% 성공률에 도달하지 못했습니다. ChatGPT는 다양한 난이도에서 일관된 성능을 보였던 반면, Claude는 문제의 복잡성에 따라 변동성이 있었습니다. 이 연구는 LLM의 현재 성능을 바탕으로 향후 AI 평가에 대해 보다 엄격한 접근의 필요성을 제시합니다.



### Green My LLM: Studying the key factors affecting the energy consumption of code assistants (https://arxiv.org/abs/2411.11892)
Comments:
          Submitted to JSS

- **What's New**: 최근 대형 언어 모델(LLMs)이 코드 생성 능력을 크게 향상시켜 개발자의 통합 개발 환경(IDE) 내에서 코드 보조 도구로 통합되고 있습니다. 이 연구에서는 GitHub Copilot과 같은 코드 보조 도구의 에너지 소비를 분석하고, 다양한 환경설정 요인이 에너지 사용에 미치는 영향을 조사합니다. 이는 데이터 세트를 수집하고 20명의 개발자로부터 개발 추적을 통해 수행되었습니다.

- **Technical Details**: 우리는 GitHub Copilot 사용 시 개발자가 소비하는 평균 에너지를 측정하기 위해 실험을 설계했습니다. 참가자들은 Java로 작은 애플리케이션을 개발한 후 에너지 소비를 분석하기 위해 추적 데이터를 재생하는 두 단계의 실험을 수행했습니다. 수집된 데이터 세트는 AssistantTraces라고 불리며, 이 데이터는 코드 보조 도구의 현실적인 사용 설정에서 에너지 소비를 측정하는 데 필수적입니다.

- **Performance Highlights**: 연구 결과, 사용자가 거부한 제안들로 인해 상당한 양의 에너지가 낭비되고 있다는 것을 알 수 있었습니다. 수동으로 코드 보조 도구의 생성 요청을 트리거하는 것이 에너지 소비를 줄이는 데 매우 효과적이며, 더 많은 동시 사용자가 있을 경우 서버 자원을 보다 효율적으로 사용할 수 있습니다. 결론적으로, 코드 보조 도구의 설정을 면밀히 조정함으로써 에너지 절약을 실현할 수 있는 방법이 제시됩니다.



### CSP-Net: Common Spatial Pattern Empowered Neural Networks for EEG-Based Motor Imagery Classification (https://arxiv.org/abs/2411.11879)
- **What's New**: 이 논문은 Electroencephalogram (EEG) 기반의 motor imagery (MI) 분류를 위한 두 가지 CSP-empowered neural network(CSP-Nets)를 제안합니다. CSP는 EEG 신호의 스칼프에서 이루어지는 다양한 MI 작업 중 에너지 분포를 활용하는 전통적인 머신 러닝 기술로, 이 논문에서는 이를 convolutional neural networks (CNN)와 통합하여 MI 분류 성능을 향상시키고자 합니다. 이 접근 방식은 특히 훈련 샘플이 적을 때 효과적입니다.

- **Technical Details**: CSP-Net-1은 CNN 앞에 CSP 레이어를 추가하여 입력의 구별 가능성을 높이고, CSP-Net-2는 CNN의 합성곱 레이어를 CSP 레이어로 대체합니다. 이 두 CSP-Nets는 훈련 데이터에서 설계된 CSP 필터로 CSP 레이어 파라미터를 초기화하며, 훈련 중에 이들을 고정하거나 gradient descent를 통해 최적화할 수 있도록 설계되었습니다. 이를 통해 EEG 신호에서 중요한 특징을 효과적으로 캡처할 수 있습니다.

- **Performance Highlights**: 실험 결과, CSP-Nets는 서로 다른 MI 데이터셋에서 CNN 백본보다 일관되게 우수한 성능을 나타냈습니다. 특히, 본 논문에서 제안하는 네트워크는 적은 수의 훈련 샘플이 있는 경우에도 효과적으로 작용하여, EEG 기반 뇌-컴퓨터 인터페이스에서의 전통적인 기계 학습과 데이터 기반 심층 학습의 통합의 장점을 보여주고 있습니다.



### Exploring Optimal Transport-Based Multi-Grained Alignments for Text-Molecule Retrieva (https://arxiv.org/abs/2411.11875)
Comments:
          BIBM 2024 Regular Paper

- **What's New**: 생명정보학 분야에서 본 연구는 텍스트 기반의 분자 검색 작업에 대한 새로운 접근 방식을 제안합니다. 우리가 제안한 Optimal TRansport-based Multi-grained Alignments 모델(ORMA)은 텍스트 설명과 분자 구조 간의 다중 정렬을 가능하게 하여 연구자들이 적합한 분자를 식별하는 데 도움을 줍니다. ORMA는 텍스트 인코더와 분자 인코더를 갖추고 있으며, 분자의 하위 구조에서 세부 정보를 포착하여 정확한 검색을 돕습니다.

- **Technical Details**: ORMA의 핵심 혁신은 Optimal Transport(OT) 기법을 활용하여 토큰과 모티프의 정렬을 수행하는 것입니다. 이 모델은 각 입력 텍스트 설명을 처리하여 토큰과 문장 수준의 표현을 생성하고, 분자는 계층적인 이질 그래프로 모델링되어 원자, 모티프, 분자 노드를 포함합니다. 또한, 다양한 수준에서의 대조 학습을 통해 텍스트-원자, 다중 토큰-모티프, 문장-분자 간의 정렬을 세밀하게 조정합니다.

- **Performance Highlights**: 실험 결과, ORMA는 ChEBI-20과 PCdes 데이터셋에서 기존의 최첨단(SOTA) 모델에 비해 월등한 성능을 보여주었습니다. 특히 ChEBI-20에서 텍스트-분자 검색의 Hits@1 점수가 66.5%로, 기존 SOTA인 AMAN을 17.1% 초과했고, 분자-텍스트 검색에서는 61.6%로 AMAN보다 15.0% 우수한 성적을 기록했습니다.



### A Multi-Modal Unsupervised Machine Learning Approach for Biomedical Signal Processing in CPR (https://arxiv.org/abs/2411.11869)
- **What's New**: 이 논문은 심폐소생술(CPR) 신호의 잡음을 제거하기 위한 새로운 비지도 머신러닝(ML) 방법론을 제안합니다. 기존의 신호 처리 기술의 한계를 뛰어넘어, 다양한 신호 출처를 활용하여 denoising(잡음 제거) 프로세스를 개선하는 다중 모달리티(multi-modality) 프레임워크를 도입합니다. 이 방법은 신호 간 상관관계를 유지하면서도 기존 방법들을 초월하는 성능을 보여주어, 실시간 응용에서 뛰어난 효과를 발휘합니다.

- **Technical Details**: 해당 프레임워크는 CPR 신호의 다양성과 시간 민감성을 고려하여 설계되었습니다. 비지도 ML을 통해 라벨이 없는 데이터에서 신호의 잡음을 효과적으로 제거하고 신호 왜곡 없이 중요한 신호 특징을 보존하는 특징이 있습니다. 논문은 또한 노이즈 특성을 각 신호 별로 개별적으로 처리하여 신호 해석의 정확성을 향상시키기 위한 방법론이 포함되어 있습니다.

- **Performance Highlights**: 제안하는 방법론은 기존의 ML 및 필터 방법과 비교하여 신호 대 잡음비(SNR) 및 최대 신호 대 잡음비(PSNR) 면에서 우수한 성능을 보여주었습니다. 특히, 각 신호를 전용 모델을 통해 처리한 후 조합할 때도 신호 데이터 간의 상관관계를 높은 수준으로 유지하는 것을 확인하였습니다. 이 접근법은 CPR뿐만 아니라 다른 임상 환경에서도 생물 의학 신호 처리의 잠재력을 향상시킬 수 있는 가능성을 가지고 있습니다.



### Machine Learning Assisted Postural Movement Recognition using Photoplethysmography(PPG) (https://arxiv.org/abs/2411.11862)
- **What's New**: 이번 연구는 노인 인구의 증가에 따라 낙상 감지 및 예방 기술의 필요성을 강조합니다. Photoplethysmography(PPG) 데이터만을 이용해 자세 변화를 인식하는 첫 번째 기계 학습 기법을 사용한 것에 주목해야 합니다. 이 연구는 11명의 참가자를 대상으로 자세 움직임과 PPG 신호의 변화를 조사하여 낙상 리스크 평가에 중요한 통찰을 제공합니다.

- **Technical Details**: PPG는 조직의 미세혈관 내 맥박 혈액량을 측정하기 위한 광학적 방법으로, LED를 이용해 혈액에 반사되는 빛을 측정합니다. 특정 자세 변경 시, 혈액 부피의 급격한 변화를 감지하여 기계 학습 알고리즘이 혈액 흐름 변화를 분류합니다. 이를 통해 낙상 위험을 사전에 감지하도록 돕는 휴대용 센서 개발이 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 인공지능 신경망(Artificial Neural Network, ANN)이 가장 높은 분류 성능을 나타내었으며, 시험 정확도는 85.2%, F1 점수는 78%에 달했습니다. 이 성과는 노인 낙상 예방에 대한 기술적 가능성을 넓히는 데 기여할 것입니다.



### Can EDA Tool Feedback Improve Verilog Generation by LLMs? (https://arxiv.org/abs/2411.11856)
- **What's New**: 이 연구에서는 전통적인 Verilog 하드웨어 설명 언어(HDL) 설계에 대해 자동화된 프로세스를 제공하기 위해 AutoChip이라는 오픈 소스 프레임워크를 제안합니다. AutoChip은 Large Language Models(LLMs)를 활용하여 전자 설계 자동화(EDA) 도구의 피드백을 기반으로 Verilog 코드를 생성하고 수정합니다. 이 접근 방식은 복잡한 하드웨어 설계에서 피드백 기반의 코드를 생성하는데 있어 효율성을 높입니다.

- **Technical Details**: AutoChip은 LLM과 Verilog 컴파일러 및 시뮬레이션의 출력을 결합하여 Verilog 코드를 반복적으로 생성하고 수정하는 방법론을 사용합니다. 연구에서는 LLM 생성 Verilog의 성공 여부를 평가하기 위해 VerilogEval 벤치마크 세트를 활용하였으며, 이는 HDLBits의 문제와 테스트벤치에 기반합니다. 다양한 피드백 모드를 통해 후보 솔루션을 평가하고 최적의 디자인을 제공하는 이 과정은 하드웨어 엔지니어의 실제 설계 흐름을 반영합니다.

- **Performance Highlights**: 결과적으로 EDA 도구의 피드백이 없는 단일 촉발(Zero-shot Prompting) 방식보다 LLM이 생성한 HDL의 정확성 향상에 일관되게 효과적임을 확인했습니다. 특히 GPT-4o 모델에서는 성공적인 설계의 수가 5.8% 증가하고, 비용은 34.2% 감소한 결과를 보였습니다. 또한, 여러 개의 모델을 혼합하여 사용하는 접근 방식에서도 비용을 41.9% 절감하면서 동일한 성공률을 나타냈습니다.



### Chat Bankman-Fried: an Exploration of LLM Alignment in Financ (https://arxiv.org/abs/2411.11853)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 금융 분야에서 윤리적 및 법적 기준을 준수하는지 평가하기 위한 실험적 프레임워크를 제안합니다. 다양한 내부 및 외부 요인에 따라 고객 자산을 악용할 의사가 있는지를 테스트하는 새로운 설정을 도입하였습니다. 연구 결과, LLM들은 주어진 환경에서 비윤리적 행동을 보이는 경향이 상당히 다양하며, 이는 경제 이론에 의해 예측된 바와 일치합니다.

- **Technical Details**: 실험 시나리오는 2022년 크립토 자산 거래소 FTX의 붕괴에서 영감을 받았습니다. LLM이 CEO로 작용하는 가상의 금융회사를 모델링하여, 고객 자산을 이용해 내부 손실을 메우는 유혹에 처하는 결정을 내리도록 하였습니다. 실험은 약 54,000회 시뮬레이션을 수행하며, 다양한 압력 변수를 조정하여 LLM의 비윤리적 행동 가능성을 분석합니다.

- **Performance Highlights**: 연구 결과, 윤리적 행동을 저해하는 요인은 위험 회피, 이윤 기대치 및 규제 환경 등이 있으며, LLM에 따라 이러한 결과의 영향은 다르게 나타납니다. 실험적 프레임워크는 금융 당국 및 기관이 LLM 채택에 따른 리스크를 더 잘 이해하고 측정하는 데 도움이 될 수 있습니다. 연구팀은 이들을 위한 코드와 벤치마크 데이터를 GitHub에 공개할 예정입니다.



### LUTMUL: Exceed Conventional FPGA Roofline Limit by LUT-based Efficient Multiplication for Neural Network Inferenc (https://arxiv.org/abs/2411.11852)
Comments:
          Accepted by ASPDAC 2025

- **What's New**: 이번 연구에서는 FPGA 기반의 신경망 가속기를 위한 LUTMUL을 소개합니다. LUTMUL은 전통적으로 DSP 블록이 수행하던 곱셈을 룩업 테이블(LUT)을 활용하여 수행하여, 계산 효율성을 크게 개선했습니다. LUT의 수가 DSP보다 100배 더 많기 때문에, FPGA의 성능을 극대화하는 데 기여합니다. 실험 결과는 이 설계가 FPGA 가속기 중 가장 빠른 추론 속도를 기록하였음을 보여줍니다.

- **Technical Details**: LUTMUL은 새로운 재구성 데이터 흐름 아키텍처를 통해 설계되었으며, 이는 메모리 접근 시간을 최소화하고 각 CNN 층에 대해 최적화된 아키텍처를 제공합니다. LUT 자원을 활용하여, 기존 DSP 기반 가속기의 피크 성능을 초과할 수 있도록 설계되었습니다. 또한, LUTMUL은 quantized 신경망 가중치를 LUT에 포함시켜 곱셈을 효율적으로 수행합니다. 이를 통해 단일 4비트 곱셈을 위해 단 2개의 LUT를 필요로 합니다.

- **Performance Highlights**: LUTMUL 설계를 통해, FPGA는 초당 1627개의 이미지를 처리할 수 있는 최고의 추론 속도를 달성하였으며, ImageNet 데이터셋에서 top-1 정확도 70.95%를 유지하고 있습니다. 이 결과는 LUT 기반 곱셈 기법이 기존 DSP 기반 설계보다 뛰어난 성능을 제공할 수 있음을 입증합니다. LUTMUL을 사용하면 FPGA의 재구성 가능성을 최대한 활용하고, 깊은 학습 작업에서 성능 향상을 기대할 수 있습니다.



New uploads on arXiv(cs.LG)

### Benchmarking Positional Encodings for GNNs and Graph Transformers (https://arxiv.org/abs/2411.12732)
- **What's New**: 이 논문에서는 Positional Encoding (PE)의 효용성을 평가하기 위한 포괄적인 벤치마크를 제시하여 Graph Neural Networks (GNNs)와 Graph Transformers (GTs)의 다양한 조합을 비교하고 분석하고 있다. 기존 모델에 대한 테스트와 함께, PE가 성능에 미치는 영향을 보다 명확히 이해하기 위한 목적이다. 특히, 메세지 패싱(messaging-passing) 구조를 갖춘 GNN과 GTs 간의 이론적 연결성을 강화하는 데 초점을 두고 있으며, 이를 통해 새로운 PE 조합이 기존 방법을 초월할 수 있는지를 알아보고 있다.

- **Technical Details**: 논문은 Laplacian, Random Walk 및 기타 방법을 포함한 세 가지 주요 카테고리로 PE를 분류하고, 이를 통해 그래프 기반 모델의 기능을 향상시키기 위한 다양한 메커니즘을 탐구하고 있다. 또한, MPNN과 GT의 관계를 정리하고, 특정 조건 하에서 두 모델이 동일한 표현 가능성을 가질 수 있음을 증명할 예정이다. 이론 분석을 통해 PE와 MPNN 및 GT 간의 연관성을 확립함으로써, 실질적인 성능 개선을 이끌기 위한 새로운 방법론을 제시하고 있다.

- **Performance Highlights**: 실험 결과, 새로운 PE 조합이 기존의 성과를 초과할 수 있음을 보여주었으며, 특히 다양한 벤치마크 데이터셋에서 해당 성능이 입증되었다. GTs와 MPNNs의 다양한 조합을 통해 최첨단 성능을 실현할 수 있음을 명확히 하며, 코드를 공개하여 후속 연구를 지원하고 있다. 이를 통해 향후 PE와 아키텍처의 최적 결합에 대한 탐색을 용이하게 하는 통합 평가 프레임워크를 제공하고 있다.



### Heuristic-Free Multi-Teacher Learning (https://arxiv.org/abs/2411.12724)
- **What's New**: 최근 연구에서 Teacher2Task라는 새로운 멀티-교사 학습 프레임워크가 소개되었습니다. 이 프레임워크는 기존의 수작업 집계 휴리스틱을 없애고, 각 교사만의 입력 토큰을 도입함으로써 데이터의 혼란을 줄여줍니다. Teacher2Task는 각 교사의 스타일을 반영하는 N개의 보조 과제와 진짜 레이블에 중점을 둔 1개의 주요 과제로 구성된 총 N+1개의 과제로 훈련 프로세스를 재구성합니다.

- **Technical Details**: 기존의 멀티-교사 학습 방법들은 주로 예측 결과를 단순히 집계하여 최종 레이블로 사용하는 접근법을 취하지만, Teacher2Task는 이를 확장하여 각 교사의 신뢰도 점수를 예측하는 보조 작업을 생성합니다. 각 새로운 교사는 단순히 새로운 보조 작업을 도입하는 방식으로 시스템에 통합됩니다. 또한, 교사 식별자를 입력에 포함시킴으로써, 노이즈 데이터를 감소시키고 교사 간의 혼란된 주석을 해결할 수 있는 강점을 가지고 있습니다.

- **Performance Highlights**: 실험을 통해 Teacher2Task는 다양한 아키텍처와 작업에 걸쳐 성능을 개선하고 강건성을 보여주었습니다. 특히, 이 방법은 각 교사의 기여도를 효율적으로 활용함으로써, 학습 데이터의 레이블 효율성을 높이고, 교사의 신뢰도 점수를 데이터로 활용해 전반적인 성능을 향상시키는 데 성공했습니다. 결과적으로, Teacher2Task는 전통적인 집계 방식에 비해 더 다양한 데이터 소스를 효과적으로 조합할 수 있는 방법을 제시하고 있습니다.



### Learning multivariate Gaussians with imperfect advic (https://arxiv.org/abs/2411.12700)
- **What's New**: 이 연구는 학습 또는 예측 고급 알고리즘의 맥락에서 확률 분포 학습 문제를 재조명하고, 불확실성이 포함된 조언의 사용에 대한 새로운 접근 방식을 제시합니다. 특히, 저자들은 주어진 조언이 실제 분포와 얼마나 유사한지를 기반으로 샘플 복잡도(sample complexity)를 개선할 수 있는 알고리즘의 개발을 목표로 합니다. 이 접근법은 다변량 가우시안 분포 학습 문제에 초점을 맞추고 있으며, 고차원 분포에 대한 처리를 가능하게 합니다.

- **Technical Details**: PAC(Probably Approximately Correct) 학습 설정에서, 이 연구는 특정 형태의 조언이 있을 때의 샘플 요구사항을 분석합니다. 독자들은 조언의 품질이 샘플 복잡도에 미치는 영향을 이해할 수 있으며, 주어진 공분산 행렬의 추정 값이 실제와 어느 정도 일치하는지를 통해 필요한 샘플 수를 줄일 수 있음을 보여줍니다. 저자들은 샘플 수가 `	ilde{O}(d^{2-eta}/eta^{2})`로 감소함을 이론적으로 증명합니다.

- **Performance Highlights**: 연구 결과는 주어진 조언이 충분히 정확할 경우, 전통적인 샘플 복잡도 하한을 초과하여 성능을 개선할 수 있음을 보여줍니다. 구체적으로 실험을 통해, 조언이 주어졌을 때 다변량 가우시안 분포를 학습하는 과정에서 필요한 샘플 수가 크게 줄어드는 것을 입증합니다. 이로 인해 고차원 데이터 처리에 있어 더욱 효율적인 학습이 가능하다는 점에서 큰 의미를 가집니다.



### Attribute Inference Attacks for Federated Regression Tasks (https://arxiv.org/abs/2411.12697)
- **What's New**: 이번 논문에서는 Federated Learning (FL) 환경에서 회귀 작업에 대한 새로운 모델 기반 속성 추론 공격(attribute inference attack, AIA)을 제안하고 있습니다. 이전 연구들은 주로 분류 작업(classification tasks)에 초점을 맞추었지만, 회귀 작업(regression tasks)에 대한 연구는 부족했습니다. 이를 해결하기 위해, 우리는 새로운 두 단계의 모델 기반 AIA를 소개하며, 이를 통해 공격자의 전략이 효과적임을 보였습니다.

- **Technical Details**: 논문에서는 공격자가 메시지를 도청하거나 훈련 과정에 직접 개입하여 최적의 로컬 모델을 대략적으로 재구성하는 방법을 제안합니다. 우리의 모델 기반 AIA는 기존의 그래디언트 기반 AIA보다 효과적인 성능을 보이며, 특히 이질적인 클라이언트 데이터셋에서 높은 재구성 정확도를 기록하고 있습니다. 또한, 회귀 작업의 성능을 저해하는 요소들에 대한 분석과 함께, 관련 수학적 이론을 통해 모델 기반 AIA의 정확도를 낮추는 경계를 설정하였습니다.

- **Performance Highlights**: 실험 결과, 우리의 제안된 모델 기반 AIA는 회귀 작업에서 강력한 공격 성능을 발휘하며, 기존의 그래디언트 기반 공격 방식보다 최대 30% 높은 정확도를 기록했습니다. 이는 FL의 회귀 작업에서 프라이버시 유출을 정량적으로 측정하기 위한 신뢰할 수 있는 방법으로 자리잡을 수 있음을 시사합니다. 이러한 성과는 FL 환경에서의 공격 가능성을 더욱 드러내므로, 향후 관련 연구에 중요한 기반이 될 것입니다.



### IMUVIE: Pickup Timeline Action Localization via Motion Movies (https://arxiv.org/abs/2411.12689)
Comments:
          This is a preprint version, 12 pages, 20 figures, 3 tables

- **What's New**: 이번 연구에서는 IMUVIE라는 신개념 착용형 시스템을 도입했습니다. 이 시스템은 모션 무비(Motion Movies)와 기계 학습 모델을 사용하여 물건을 집는 행동을 자동으로 감지하고 측정합니다. IMUVIE는 사용자 친화적인 인터페이스를 통해 비침습적으로 노인의 낙상 위험을 평가할 수 있는 솔루션입니다.

- **Technical Details**: IMUVIE는 인체 움직임을 정밀하게 측정하기 위해 데이터 정규화(data normalization), 가림 처리(occlusion handling)와 간소화된 시각화(streamlined visuals) 원칙을 따릅니다. 이 시스템은 33명의 참여자를 대상으로 한 연구를 통해 256,291개의 모션 무비 프레임에서 91-92%의 높은 정확도를 달성했습니다. IMUVIE는 새로운 데이터에서도 강력한 일반화 성능을 보이며 과거에 훈련한 적이 없는 피험자에 대해서도 잘 동작합니다.

- **Performance Highlights**: 연구 결과 IMUVIE는 테스트된 129회의 픽업 사건에서 97%의 이벤트 레벨 리콜(recall)을 달성했습니다. 사용자의 수용도 조사를 통해, 사용자 친화성은 채택에 가장 중요한 요소로 작용함을 나타냈으며, 이는 노인들이 쉽게 사용할 수 있는 시스템에 대한 강한 관심과 신뢰를 보였음을 알 수 있습니다. IMUVIE는 노인의 독립적인 삶을 지원하기 위한 실용적인 가정에서의 낙상 위험 평가 도구를 제공합니다.



### Machine Learning Approaches on Crop Pattern Recognition a Comparative Analysis (https://arxiv.org/abs/2411.12667)
Comments:
          Published in ICNTET2018: International Conference on New Trends in Engineering & Technology Tirupathi Highway, Tiruvallur Dist Chennai, India, September 7-8, 2018

- **What's New**: 이 논문에서는 농업 활동 모니터링의 중요성을 강조하며, 특히 원격 감지(remote sensing)의 역할을 다루고 있습니다. 기존의 분류 방법(SVM 및 결정 트리)에 비해, 딥 뉴럴 네트워크(Deep Neural Network, DNN)를 이용한 분류 방법을 제안하고 있습니다.

- **Technical Details**: 논문에서는 시계열 원격 감지 데이터(time series remote sensing data)를 활용하여 재배 패턴(cropping pattern)을 생성합니다. 그리고 나이브 베이즈(Naive Bayes) 및 랜덤 포레스트(Random Forest)와 같은 두 가지 다른 머신 러닝(machine learning) 접근법과의 비교 분석을 통해 DNN의 성능을 강조하고 있습니다.

- **Performance Highlights**: DNN 기반의 분류 방법이 농작물 패턴 인식(crop pattern recognition)에서의 성능을 개선하는 데 기여할 것으로 기대됩니다. 이 연구는 대규모 지속적인 농업 모니터링을 위한 새로운 가능성을 열 것입니다.



### Auto-Evaluation with Few Labels through Post-hoc Regression (https://arxiv.org/abs/2411.12665)
- **What's New**: 이 논문에서는 Prediction Powered Inference (PPI) 프레임워크의 새로운 두 가지 기법을 제안합니다. 이 기법들은 소량의 레이블이 있는 데이터로부터 더욱 낮은 분산(low variance) 추정치를 생성하는 데 중점을 두고 있습니다. 특히, 저자들은 튜닝 파라미터 λ의 중요성을 강조하며, 기존 방법보다 더 개선된 성능을 보여줍니다.

- **Technical Details**: PPI 프레임워크는 레이블이 있는 작은 데이터 집합과 자동 평가 시스템의 통계적 힘을 활용하여 평가 대상의 수량에 대한 편향이 없는 추정을 제공합니다. 하지만 대다수의 PPI 관련 연구는 50개 이상의 레이블을 필요로 하는 상황을 가정하고 있어, 레이블이 거의 없는 경우의 적용 가능성은 충분히 조사되지 않았습니다. 이 논문에서는 PPI의 저 레이블 환경에서의 성능을 개선하기 위한 수정 방안을 제안합니다.

- **Performance Highlights**: 저자들은 PPI++ 방법의 이론적 분석을 제공하며, 이는 단변량 회귀(univariate regression)와 관련이 있습니다. 또한, 새로운 두 가지 PPI 확장을 제안하여 기존 방법보다 개선된 성능을 보이는 것을 목표로 하고 있으며, 이러한 기법들은 대량의 비주석 데이터(auto-generated data)를 필요로 하지 않고도 변별력을 유지할 수 있는 가능성을 보여줍니다.



### DLBacktrace: A Model Agnostic Explainability for any Deep Learning Models (https://arxiv.org/abs/2411.12643)
- **What's New**: 이번 연구에서는 DLBacktrace라는 새로운 기술을 소개합니다. 이는 다양한 신경망 구조에서 모델 결정을 투명하게 하는 방법으로, Multi Layer Perceptron (MLP), Convolutional Neural Network (CNN), Large Language Models (LLM) 등 여러 도메인에서 효과적으로 작동합니다. DLBacktrace는 해석 가능성 (interpretability)의 필요성을 강조하며, AI 시스템의 신뢰를 구축하고 책임성을 보장하는 데 기여합니다.

- **Technical Details**: DLBacktrace는 모델-비의존적 방법으로, 출력에서 입력으로 관련성을 추적하여 각각의 계층에서의 중요도 점수를 부여합니다. 이 방법은 PyTorch와 TensorFlow에서 구현된 다양한 모델 아키텍처와 호환되며, 정량적인 메트릭을 사용하여 기존 해석 가능성 방법들과 비교한 벤치마킹 결과를 제시합니다. 심층 학습에서 모델의 투명성을 개선하기 위해 설계된 이 기술은 다양한 데이터 형식에 적용할 수 있습니다.

- **Performance Highlights**: DLBacktrace는 LIME, SHAP, Grad-CAM 등과 같은 기존 해석 가능성 방법들과 비교될 때 독창적이고 신뢰할 수 있는 해석을 제공합니다.  전반적인 모델 설계 및 데이터 타입을 아우르는 두 가지 분석 방식 (지역적 및 전역적 해석)을 통해 모델의 명확성을 높입니다. 또한 오픈 소스로 제공되어 다양한 연구 및 산업 응용에서 적극 활용될 수 있는 가능성을 가지고 있습니다.



### PyAWD: A Library for Generating Large Synthetic Datasets of Acoustic Wave Propagation with Devito (https://arxiv.org/abs/2411.12636)
- **What's New**: 이번 논문에서는 고해상도의 합성 지진 데이터를 생성할 수 있는 Python 라이브러리인 PyAWD를 소개합니다. 이는 비균일한 매질에서의 시공간적 음향파 전파를 시뮬레이션하며, 기계 학습(Machine Learning)을 통한 지진 분석에 필요한 상세한 데이터를 제공합니다. PyAWD는 파라미터 조정이 가능하여, 지진파의 행동을 포착할 수 있는 ML 규모의 데이터 세트를 생성하는 데 도움을 줍니다.

- **Technical Details**: PyAWD는 복잡한 매질에서의 파동 전파 현상을 시뮬레이션하는 도구로, 파동 속도, 감쇠(attenuation), 외부 힘과 같은 파라미터의 사용자 맞춤화를 지원합니다. 이 라이브러리는 PyTorch와 통합되어 딥러닝(Deep Learning) 애플리케이션에 필요한 데이터 세트를 생성하며, 2D 및 3D 파동 전파 시각화 도구를 제공합니다. 또한, 사용자가 정의한 커스텀 필드를 통해 복잡한 지질 구조를 시뮬레이션 할 수 있는 기능을 제공합니다.

- **Performance Highlights**: PyAWD의 활용 예로 에피센터(Epicenter) 회수(task) 작업을 통해 이 라이브러리의 유용성을 입증하였습니다. 특히, 2D 음향파 시뮬레이션을 통해 생성된 데이터 세트를 활용하여 전통적인 방법과 ML 모델을 비교하였습니다. 결과적으로, ML 모델들이 지진파 데이터를 더 잘 활용하여 에피센터 위치 추정의 정확도를 향상시킬 수 있음을 보여주었습니다.



### Exploring the Manifold of Neural Networks Using Diffusion Geometry (https://arxiv.org/abs/2411.12626)
- **What's New**: 이번 연구는 머신러닝에서의 manifold hypothesis(매니폴드 가설)을 바탕으로, 신경망의 공간에서 매니폴드를 학습하는 새로운 접근법을 제시합니다. 이는 데이터 포인트를 신경망으로 보며, 숨겨진 층(hidden layer) 표현 간의 거리를 도입하여 이루어집니다. 이러한 거리를 사용하여 PHATE(비선형 차원 축소 알고리즘)와 같은 기법을 통해 신경망의 매니폴드를 생성합니다.

- **Technical Details**: 연구에서는 매니폴드의 특성을 클래스 분리(class separation), 계층 클러스터 구조(hierarchical cluster structure), 스펙트럴 엔트로피(spectral entropy), 및 위상 구조(topological structure)를 통해 설명합니다. 이러한 매니폴드를 분석한 결과, 높은 성능의 신경망들이 서로 가까운 위치에 클러스터를 이루며, 각기 다른 특성에서 일관된 임베딩 패턴을 보여줍니다.

- **Performance Highlights**: 마지막으로, 이 접근법이 하이퍼파라미터 최적화(hyperparameter optimization)와 신경 구조 탐색(neural architecture search)에 유용하게 활용될 수 있음을 입증합니다. 매니폴드에서 샘플링함으로써 최적의 신경망 구성을 효과적으로 탐색할 수 있는 가능성을 제시합니다.



### Provable unlearning in topic modeling and downstream tasks (https://arxiv.org/abs/2411.12600)
- **What's New**: 이 논문에서는 기계 모델의 사전 학습(pre-training)과 미세 조정(fine-tuning) 과정에서 '기계 언러닝(machin unlearning)'을 위한 이론적 보장을 처음으로 제공하고 있습니다. 특히, 주제 모델(topic model)과 같은 간단한 언어 모델을 통해 탐색, 분류와 같은 다운스트림 작업을 해결하는 방법을 다룹니다. 이 연구는 기존의 사전 학습 모델에서 데이터를 삭제할 수 있는 효과적인 알고리즘을 설계하여, 모델 성능에 큰 영향을 미치지 않으면서도 훈련 데이터를 성공적으로 삭제할 수 있는 방법을 제시합니다.

- **Technical Details**: 해당 논문은 주제 모델에 대한 언러닝 알고리즘을 설계하며, 원래 데이터 세트의 크기와 무관하게 계산 오버헤드를 줄이는 효율적인 접근 방식을 제공합니다. 연구는 또한 모델의 삭제 용량(deletion capacity)을 정량화하여, 성능 저하 없이 삭제할 수 있는 예제의 수를 분석합니다. 특히, 선형 헤드(linear head)를 통해 주제 모델을 미세 조정한 후에도 언러닝을 수행할 수 있는 알고리즘을 설계하여, 특정 작업에 대해 최적화된 모델에서 사전 학습 데이터를 더 쉽게 삭제할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구의 알고리즘은 기계 언러닝의 기준(gold standard) 충족이 가능하도록 설계되었으며, 기존의 방법들보다 낮은 비용으로 목표를 달성할 수 있습니다. 또한, 데이터 삭제 요청이 있을 경우에도 모델 성능이 크게 저하되지 않음을 입증합니다. 이러한 점에서 사전 학습 및 미세 조정 환경에서 효과적인 기계 언러닝 알고리즘의 개발이 이루어진 점이 두드러집니다.



### UMGAD: Unsupervised Multiplex Graph Anomaly Detection (https://arxiv.org/abs/2411.12556)
- **What's New**: 본 논문은 새로운 비지도 다중 그래프 이상 탐지 방법인 UMGAD(Unsupervised Multiplex Graph Anomaly Detection)를 제안합니다. 이 방법은 복잡한 다중 상호작용 그래프에서 노드 간의 다중 관계 상관관계를 학습하고, 그래프 마스킹 오토인코더(GMAE)를 통해 노드 속성과 구조 복원을 수행하여 이상 정보를 포착합니다. 또한, 새롭게 설계된 이상 점수 임계값 선택 전략을 도입하여 모델이 실제 비지도 환경에서의 지상 진실(ground truth)에 의존하지 않도록 합니다.

- **Technical Details**: UMGAD는 두 가지 주요 단계로 구성됩니다: 첫째, GMAE를 통해 다중 관계적 상관성을 학습하고, 노드의 속성과 구조 복원을 통해 이상 정보를 캡처합니다. 둘째, 노이즈와 중복 정보를 줄이기 위해 속성 수준과 하위 그래프 수준의 증강 뷰 그래프를 각각 생성하고, 다시 GMAE를 통해 속성과 구조를 복원합니다. 마지막으로, 원래 뷰 그래프와 증강 뷰 그래프 간의 대조 학습(contrastive learning)을 통해 노드 속성과 구조적 특징을 최적화하여 이상 탐지 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시한 실험 결과, UMGAD는 평균 13.48% AUC(Area Under the Curve)와 11.68% Macro-F1의 개선 효과를 보이며 기존의 최첨단(SOTA) 방법들을 능가하는 성과를 달성했습니다. 이 연구는 다중 이질 그래프에서 이상 탐지 문제를 해결하고, 비지도 환경에서 이상 점수 임계값 선택의 어려움을 극복함으로써 이상 탐지의 성능을 크게 향상시켰습니다.



### Predicting Customer Satisfaction by Replicating the Survey Response Distribution (https://arxiv.org/abs/2411.12539)
- **What's New**: 본 논문에서는 고객 만족도(CSAT) 예측을 위한 새로운 접근 방식을 제안합니다. 기존에는 조사에 응답한 일부 고객의 데이터만 기반으로 평균 CSAT를 계산하여 발생하던 편향을 감소시키기 위한 모델이 개발되었습니다. 이 방법은 실제 생산 환경에서도 모든 통화에 대해 고객 만족도를 예측할 수 있도록 하여 더 정확한 성과 지표를 제공합니다.

- **Technical Details**: 연구에서는 자주 업데이트 되는 머신 러닝 모델의 클래스 비율 변화를 방지하기 위해 제어 메커니즘을 도입합니다. 이 메커니즘은 샘플링 노이즈에 의한 위험을 완화하고, ASR(Automated Speech Recognition) 데이터에서 고객 만족도의 분포를 정확하게 복제할 수 있도록 최적화된 결정을 제공합니다. 이 방법은 다중 클래스와 순서 분류 문제에서 사용할 수 있으며, 클래스 불균형을 개선하는데 기여합니다.

- **Performance Highlights**: 실험에 사용된 데이터는 892,000개의 통화 기록으로, 모델은 높은(4 이상) 또는 낮은(3 이하) CSAT 예측을 위해 이진 출력으로 작동합니다. 모델의 정확도는 85% 이상이며, 7번의 시험 과정을 진행한 결과 배포된 모델과 시뮬레이션 간에 성능 차이가 없음을 확인했습니다. 이 연구는 고객 만족도를 반영하기 위한 포괄적인 머신 러닝 파이프라인의 일환으로 적용되어 실제 환경에서도 강건한 성과를 발휘할 것으로 기대됩니다.



### Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues (https://arxiv.org/abs/2411.12537)
- **What's New**: 최근 연구에 따르면,  Linear Recurrent Neural Networks (LRNNs)와 같은 모델들이 Transformers의 효율적인 대안으로 떠오르고 있지만, 상태 추적(state-tracking)에서 어려움을 겪고 있습니다. 기존의 LRNN들은 간단한 패리티(parity) 문제도 해결하지 못하며, 이 문제는 대각선 매트릭스의 고유값(eigenvalues)에 제약이 있기 때문입니다. 이 논문에서는 LRNN의 고유값 범위를 확대하여 패리티 문제를 해결하고 성능을 개선할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: LRNN은 구조화된 상태 전이 매트릭스를 사용하여 상태를 업데이트하며, 이 매트릭스의 구조가 LRNN의 표현력을 결정짓습니다. 본 연구는 대각선이 아닌 LRNN의 경우에도 동일한 고유값 제한이 적용됨을 보여주며, 복소수 고유값이 모듈로 카운팅(modular counting) 문제 해결에 필요하다는 것을 증명합니다. 우리는 고유값 범위를 [-1, 1]로 확장하여 LRNN의 표현력을 증가시키는 간단하면서도 강력한 솔루션을 제안합니다.

- **Performance Highlights**: 실험 결과, Mamba와 DeltaNet 모델의 고유값 범위를 확장함으로써 패리티 문제 해결뿐 아니라 상태 추적 성능이 일관되게 향상되었습니다. 또한, 이러한 수정된 LRNN들은 언어 모델링에서 원래 모델과 비교할 수 있는 성능을 보여주었으며, DeltaNet은 코드 및 수학 데이터셋에서 개선된 성능을 나타냈습니다. 이러한 개선이 학습 및 추론 비용을 변화시키지 않으면서도 가능하다는 점이 주목할 만합니다.



### Data Pruning in Generative Diffusion Models (https://arxiv.org/abs/2411.12523)
- **What's New**: 이번 연구에서는 데이터 프루닝(data pruning)의 효용성을 제시하여, 생성 모델(generative model)에 대한 적용 가능성을 탐구합니다. 기존의 연구는 주로 분류(classification)와 같은 구분 모델(discriminative model)에 대한 프루닝 전략에 중점을 두었으나, 본 연구에서는 생성 확산 모델(generative diffusion model)에 데이터 프루닝이 어떻게 긍정적인 영향을 미칠 수 있는지를 조사합니다. 예를 들어, 우리는 전략적으로 중복된(redunant) 또는 노이즈가 포함된 데이터를 제거할 경우, 대규모 데이터셋의 성능을 향상시킬 수 있음을 발견했습니다.

- **Technical Details**: 본 연구에서는 CelebA-HQ와 ImageNet 데이터셋을 통해 여러 가지 프루닝 방법을 실험하고 평가하였으며, 특히 간단한 클러스터링(cluster) 기법이 높은 성능을 나타냈습니다. 생성 모델의 학습 효율을 높이기 위해, 우리는 데이터셋에서 핵심적인 샘플을 찾아내고 노이즈가 포함된 샘플을 제거함으로써, 모델이 효과적으로 학습할 수 있도록 하였습니다. 이러한 접근방식은 생성 모델이 많은 자원을 소모하지 않고도 우수한 성과를 낼 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 이미지넷 데이터셋에서 데이터의 90%를 프루닝해도 성능 저하 없이 우수한 결과를 얻을 수 있음을 확인했습니다. 또한, 클러스터링을 활용하여 비율이 불균형한 데이터셋을 조정함으로써, 과소 대표된 집단을 공정하게 샘플링할 수 있음을 보여주었습니다. 이러한 발견은 생성 모델을 훈련할 때 제한된 데이터가 있을지라도 적절한 프루닝 전략을 통해 합리적인 결과를 얻을 수 있음을 시사합니다.



### Transformer Neural Processes -- Kernel Regression (https://arxiv.org/abs/2411.12502)
- **What's New**: 이번 논문에서는 Transformer Neural Process - Kernel Regression (TNP-KR)라는 새로운 아키텍처를 소개합니다. 이 모델은 Kernel Regression Block (KRBlock)이라 불리는 혁신적인 Transformer 블록을 통합하여, 기존 Transformer 기반 Neural Processes (TNPs)의 attention 계산 복잡도를 크게 줄입니다. 이러한 방식으로 n_C와 n_T의 수에 따라 계산 비용을 효과적으로 감소시키며, 대량의 테스트 포인트를 처리할 수 있게 합니다.

- **Technical Details**: TNP-KR은 context points(n_C)와 test points(n_T)에 대한 attention 계산의 복잡도를 𝒪⁢((n_C+n_T)²)에서 𝒪⁢(n_C²+n_Cn_T)로 줄입니다. 또한, KRBlock 내부에 Performer attention을 사용하여 복잡도를 𝒪⁢(n_C)으로 더욱 낮췄습니다. 이러한 기술은 모델이 소비자 하드웨어에서 수백만 개의 포인트로 확장할 수 있도록 지원합니다.

- **Performance Highlights**: 평가 결과, TNP-KR의 풀 버전은 최신 방법들과 유사한 성능을 보이면서도 훈련이 더 빨라지고 테스트 포인트 수가 두 배로 늘어나는 경우에도 성능을 유지합니다. 또, 빠른 변종은 수 백만 개의 테스트 및 컨텍스트 포인트를 처리하면서도 거의 동일한 성능을 제공함을 보여줍니다. 이런 성능은 소비자 하드웨어에서 실현할 수 있는 성과입니다.



### Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus (https://arxiv.org/abs/2411.12498)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 LLMs(Large Language Models)의 추론 능력을 향상시키기 위한 새로운 접근법인 추가 논리 훈련(Additional Logic Training, ALT)을 제안합니다. 이 방법은 프로그램 생성(logical reasoning samples)된 논리적 샘플을 통해 LLM의 추론 능력을 증대시키고자 하며, 특히 잠재적인 미지의 사실을 포함하는 샘플 생성에 중점을 두고 있습니다. 새로운 데이터셋인 Formal Logic Deduction Diverse (FLD×2)를 통해 제안된 방법론의 효과를 검증하였습니다.

- **Technical Details**: ALT는 고품질 논리적 샘플을 활용하여 LLMs의 추론 능력을 개선하기 위해 설계되었습니다. 연구자들은 실질적으로 LLM이 미지의 사실을 처리할 수 있도록 다양한 추론 규칙과 언어적 표현을 갖춘 다단계 추론 샘플을 포함하는 합성 데이터셋인 FLD×2를 구축하였습니다. 이러한 샘플들은 LLM의 고유한 패턴 인식 능력을 통해 비논리적 추론과 논리적 추론을 구별할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, ALT를 적용한 LLM은 논리적 추론 벤치마크에서 최대 30점, 수학 및 코딩 벤치마크에서 최대 10점의 성능 향상을 보였습니다. LLaMA-3.1-70B 모델도 ALT를 통해 유의미한 성과 상승을 시연하였으며, 자연어 추론(NLI) 작업에서도 최대 6점의 향상이 관찰되었습니다. 이러한 결과는 LLM이 고유 지식과 논리적 사고 능력을 통합하여 다양한 문제를 해결할 수 있는 가능성을 제시합니다.



### Regular-pattern-sensitive CRFs for Distant Label Interactions (https://arxiv.org/abs/2411.12484)
- **What's New**: 이 논문에서는 사용자가 지정한 패턴 내에서 긴 거리의 레이블 상호작용을 학습할 수 있도록 표준 선형 체인 조건부 랜덤 필드(CRF)를 강화한 방법인 정규 패턴 민감 CRF(RPCRF)를 제안합니다. 이를 통해 사용자는 모델이 고려해야 하는 상호작용 유형을 간결하게 지정할 수 있는 정규 표현(regular-expressions)을 작성할 수 있으며, 데이터로부터 이 패턴들이 발생하는 맥락을 학습할 수 있습니다.

- **Technical Details**: RPCRF는 추가적인 비지역적 잠재 변수(non-local potentials)로 강화된 CRF로 해석될 수 있으며, 사용자 지정 패턴 집합으로부터 자동으로 구성되는 방법을 상세히 설명합니다. 이 모델은 특이한 패턴 집합에 대해도 정확한 훈련과 추론이 가능하여, 기존의 가중 유한 상태 변환기(weighted FST)보다 계산적 요구 사항이 낮습니다. 이 접근 방식은 레이블 시퀀스 내 비지역적 의존성 구조를 포착하는 다양한 패턴의 효과성을 평가하는 데 초점을 맞춥니다.

- **Performance Highlights**: RPCRF는 합성 데이터(synthetic data)에서 효과적으로 작동하며, 다른 유형의 패턴이 레이블 시퀀스 내의 비지역적 의존성 구조를 어떻게 포착하는지를 보여줍니다. 이 모델은 전통적인 CRF와 비교할 때 장거리 상호작용을 더 잘 학습할 수 있는 가능성을 제공합니다. 이러한 접근 방식은 자연어 처리(NLP) 및 기타 순차적 데이터 처리 작업에서의 응용 가능성을 열어줍니다.



### Comparing Prior and Learned Time Representations in Transformer Models of Timeseries (https://arxiv.org/abs/2411.12476)
Comments:
          Presented at the AI in Natural Sciences and Technology (AINST) track of the 13th Conference on Artificial Intelligence (SETN 2024), 11-13 September 2024, Piraeus, Greece

- **What's New**: 이번 연구에서는 Transformer 모델의 두 가지 변형을 비교하여 시간 표현이 시계열 분석에 미치는 영향을 조사했습니다. 하나는 기존 문헌에서 제안된 고정된 시간 표현을 사용하고, 다른 하나는 데이터로부터 학습된 시간 표현을 활용합니다. 결과적으로, 시계열 데이터에서 시간 표현의 적절성을 높이는 것이 중요하다는 것을 강조하며, 기존 모델에 대한 한계를 지적하고 있습니다.

- **Technical Details**: Transformer 아키텍처는 순차적으로 입력을 처리하는 대신, 전체 시계열을 입력으로 받아 Attention 메커니즘을 통해 데이터 간의 관계를 동시에 평가합니다. 시간 표현을 위해, 연구에서는 Sinusoidal을 기반으로 한 절대적인 위치 인코딩과 트라이앵글 파형 모델을 제공합니다. 이는 시간 간격을 효과적으로 캡처하는 방법으로, 특히 동절기와 일간 주기성을 고려하여 설계되었습니다.

- **Performance Highlights**: 실험 결과, 고정된 시계열 표현보다 데이터에서 직접 학습된 시간 표현이 더 나은 성능을 보였습니다. 특히 태양광 패널의 에너지 출력을 예측할 때, 새로운 시간 표현이 정확성을 증대시켰고, 이는 기존에 알아온 패턴을 더욱 잘 모델링하는 데 기여했습니다. 이러한 발견은 Transformer가 시계열 데이터에 대한 처리를 향상시킬 수 있는 가능성을 제시합니다.



### Empirical Privacy Evaluations of Generative and Predictive Machine Learning Models -- A review and challenges for practic (https://arxiv.org/abs/2411.12451)
- **What's New**: 이 논문은 개인정보 보호 기술로 훈련된 합성 데이터 생성기가 생성하는 데이터의 사생활 위험을 평가할 필요성을 강조합니다. 기존의 생성을 위한 모델에서 개인 정보 보호에 대한 이론적 보장은 이루어졌지만, 실제 상황에서의 평가 방법론의 발전이 필요하다는 점을 지적합니다. 특히 대규모 데이터셋을 가진 어플리케이션에서의 평가의 복잡성을 다루고, 합성 데이터의 진정한 유용성을 측정하기 위한 방법을 제안합니다.

- **Technical Details**: 논문에서는 Differentially Private Stochastic Gradient Descent (DP-SGD)와 같은 알고리즘을 통해 개인정보를 보호하는 합성 데이터를 생성하는 과정을 상세히 설명합니다. 또한, 이 알고리즘이 실제 공격자와의 상호 작용을 고려하지 못하여 사생활 보호에 취약하다는 점을 명확히 하고, 실제 상황에서의 개인정보 유출을 평가할 방법을 제시합니다. 이러한 접근을 통해 합성 데이터 생성기가 데이터 소유자의 신뢰를 얻을 수 있도록 연구가 설계되었습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터의 생성 알고리즘의 진정한 효과성을 평가하기 위해 여러 평가 매트릭스가 필요하다는 점이 드러났습니다. 기존의 연구들은 평가 방법에 있어 다양성을 보이며, 서로 다른 접근 방식이 사용되었습니다. 이러한 불일치는 종합적이고 일관된 개인정보 보호 평가框架의 구축 필요성을 시사하며, 향후 지속 가능한 실용적인 향상이 필요함을 강조합니다.



### Non-IID data in Federated Learning: A Systematic Review with Taxonomy, Metrics, Methods, Frameworks and Future Directions (https://arxiv.org/abs/2411.12377)
- **What's New**: 최근 기계 학습의 발전은 여러 분산 사용자(클라이언트)가 개인 데이터를 공유하지 않고 협력하여 ML 모델을 훈련할 수 있는 사실연합 학습(Federated Learning: FL) 방식의 잠재력을 부각했습니다. 하지만 클라이언트 간의 데이터가 독립적이지 않고 동일 분포가 아닐 경우(non-IID), 모델 성능 저하 및 훈련 속도 저하 문제가 발생할 수 있습니다. 이 연구는 non-IID 데이터의 분류 및 정량화에 대한 합의 부족을 해소하기 위한 체계적인 리뷰를 제공합니다.

- **Technical Details**: FL은 개인 데이터가 있는 여러 클라이언트를 통해 분산 데이터셋에서 모델을 훈련할 수 있는 혁신적인 ML 패러다임입니다. 데이터 중앙집중식 접근 방법과 달리, FL은 각 클라이언트가 자신의 데이터에서 독립적으로 로컬 모델을 훈련시키고, 모델 업데이트만 중앙 서버에 전달하여 글로벌 모델을 생성합니다. 이러한 과정은 Federated Averaging (FedAvg)이라고 하며, 클라이언트 데이터의 프라이버시를 유지하면서 다양한 데이터 출처에서 훈련이 가능하게 합니다.

- **Performance Highlights**: 부동적으로 제공되는 FL에 대한 공개 데이터셋의 부족은 기존의 중앙집중식 데이터를 여러 클라이언트로 분割하는 방법으로 해결됩니다. FL의 성능과 수렴도는 partition protocol에 따라 달라지며, non-IID 데이터는 모델 일반화와 매개변수 집합에 더 큰 도전 과제를 제공합니다. 이 리뷰는 다양한 데이터 불균형을 체계적으로 탐색하고, 비비교적인 조건에서 FL 모델을 검증하기 위한 유용한 실험 설정을 제공하는 데 기여합니다.



### Ultra-Sparse Memory Network (https://arxiv.org/abs/2411.12364)
Comments:
          10 pages, 6 figures

- **What's New**: UltraMem은 기존 Mixture of Experts (MoE) 및 Product Key Memory (PKM) 구조를 기반으로 하고 있으며, 대규모 초희소 메모리 레이어를 도입함으로써 추론 지연을 대폭 줄이고 모델 성능을 개선합니다. 이 새로운 아키텍처는 효과적인 언어 모델을 자원 제약이 있는 환경에서 배포할 수 있도록 지원하며, 이전보다 더 큰 모델을 구성할 수 있는 경로를 제공합니다.

- **Technical Details**: UltraMem은 PKM 개념을 기반으로 하여 대규모 메모리 레이어를 활용하여 높은 메모리 접근 비용을 줄이는 데 중점을 두고 있습니다. 이 구조는 메모리 값 검색을 위해 쿼리 벡터와 키 간의 곱을 통해 점수를 계산하고, 그 점수를 기준으로 가장 관련성 높은 값을 선택하는 방식을 사용합니다. 키는 전체 메모리 값을 계산하는데 소모되는 계산 복잡성을 완화하기 위해 제품 키 주소 체계를 활용합니다.

- **Performance Highlights**: UltraMem은 기존의 MoE 모델보다 최대 6배 빠른 추론 속도를 기록하며, 동등한 계산 자원을 가진 조밀 모델과 유사한 수준의 성능을 보여줍니다. 또한 UltraMem은 MoE와 유사한 강력한 스케일링 능력을 가지고 있으며, 다양한 벤치마크에서 개선된 모델 성능을 입증하였습니다.



### Learning from Label Proportions and Covariate-shifted Instances (https://arxiv.org/abs/2411.12334)
- **What's New**: 이번 연구에서는 label proportions (LLP) 모델링의 새로운 변형인 covariate-shifted hybrid LLP 문제를 제안합니다. 이는 완전히 감독된 covariate shift 소스 데이터를 포함하여, bag-level supervision을 결합하여 기존 예측 성능을 향상시키고자 하는 접근 방식입니다. 저자들은 기존 LLP 및 domain adaptation 방법들의 한계를 넘어, bag-labels와 instance-labels를 동시에 활용하는 방법론 개발에 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 세 가지 구성 요소로 이루어진 다양한 손실 함수를 제안합니다. 첫 번째는 소스 데이터의 instance-level 손실, 두 번째는 타겟 학습 bags의 bag-level 손실, 세 번째는 도메인 적응 손실입니다. 저자들은 이러한 손실 함수들을 통해 domain adaptation 및 LLP 기법을 확장하고 이론적 보장을 제시합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법이 기존의 LLP 및 도메인 적응 기준보다 뛰어난 성능을 보임을 입증했습니다. 다양한 공공 데이터세트에서 수행된 실험은 제안된 방법이 예측 성능에서 유의미한 향상을 가져왔다고 강조하고 있습니다.



### Graph as a feature: improving node classification with non-neural graph-aware logistic regression (https://arxiv.org/abs/2411.12330)
- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)의 한계를 극복하기 위해 Graph-aware Logistic Regression (GLR)라는 새로운 비신경망 모델을 제안합니다. GLR은 노드 간의 관계를 추가적인 피처 벡터로 인코딩하여 노드 특징과 결합한 후, 이를 활용하여 노드 분류 작업을 수행합니다. 기존의 GNN 모델들이 강한 동질성(homophily) 데이터셋에 종속되는 문제를 해결하기 위해 복잡한 네트워크 아키텍처를 필요로 했던 반면, GLR은 더욱 간단하고 확장성이 뛰어난 접근 방식입니다.

- **Technical Details**: GLR은 각 노드의 이웃을 원래 그래프 내의 피처 벡터로 표현한 후, 이를 노드의 원래 속성과 결합하여 더 효율적인 학습 프로세스를 가능하게 합니다. 이 방법은 GNNs처럼 원래 그래프의 여러 수준에서 정보를 활용하되, 메시지 패싱(message passing)에 의존하지 않음을 특징으로 합니다. GLR의 성능 평가는 엄격한 평가 기준(framework) 내에서 수행되었으며, 다양한 크기, 밀도, 동질성을 가진 데이터셋들에서 평가되었습니다.

- **Performance Highlights**: GLR은 본 연구에서 실험적으로 검증된 바에 따르면 기존 기초 GNN 모델 및 고급 GNN 모델들을 능가하는 성능을 보여주었습니다. 특히, GLR은 계산 속도 면에서도 우수하여 최고의 신경망 경쟁자에 비해 최대 두 배 빠른 시간을 기록했습니다. 본 연구는 라벨 동질성만으로 GNN 성능을 설명하기에는 부족하다는 점을 강조하며, 피처 동질성(feature homophily) 개념을 도입하여 GLR의 성능 원인을 설명했습니다.



### Attributed Graph Clustering in Collaborative Settings (https://arxiv.org/abs/2411.12329)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문에서는 여러 참여자가 서로 다른 특징을 가진 동일 데이터를 수직적으로 분할하여 보유할 때 속성 그래프 클러스터링을 지원하는 협업 그래프 클러스터링 프레임워크를 제안합니다. 기존 방법들이 자료의 고립성(data isolation) 문제를 해결하는 데 한계를 보인 반면, 저자들은 새로운 샘플 공간(sample space) 축소 기법을 통해 클러스터링의 효율성을 향상시켰습니다. 또한, 중앙화된 방법과 비교하여 각 참여자들의 성공적인 로컬 결과가 협업의 전반적인 성과에 기여함을 시연합니다.

- **Technical Details**: 이 연구에서는 k-means 클러스터링을 이용하여 각 로컬 참여자에서 로컬 클러스터를 생성한 후, 교차된 클러스터의 샘플 ID를 전달하여 글로벌 클러스터를 식별하는 방식을 제시합니다. 이를 통해 샘플 공간을 줄이면서 중앙화된 방법에 준하는 정확도를 달성합니다. 저자들은 제안한 알고리즘의 정확성을 근접성(per proximity) 관점에서 증명하고 있으며, 네 개의 공공 데이터셋에서 성능 평가를 실시하여 효과성을 검증했습니다.

- **Performance Highlights**: 실험 결과, 제안된 kCAGC 방법은 기존 중앙화된 속성 그래프 클러스터링 방법들과 유사한 수준의 정확도를 달성했습니다. 특히, kCAGC는 반지도학습(semi-supervised) GraphSAGE 모델과도 경쟁할 만한 성능을 보였습니다. 이러한 결과는 각 참여자가 보유하고 있는 로컬 데이터의 프라이버시를 유지하면서도 협업 학습을 통해 효율적인 그래프 클러스터링을 가능하게 한다는 점에서 중요한 의미를 갖습니다.



### C$^{2}$INet: Realizing Incremental Trajectory Prediction with Prior-Aware Continual Causal Intervention (https://arxiv.org/abs/2411.12313)
- **What's New**: 다중 에이전트의 궤적 예측을 위한 새로운 접근법, C$^{2}$INet(Continual Causal Intervention Network)를 소개합니다. 이 방법은 다양한 환경의 편향(bias)을 고려하여 궤적 데이터를 효과적으로 학습하고 예측할 수 있도록 설계되었습니다. 다양한 시나리오에서의 메모리 큐(memory queue)를 활용하여, 연속적인 학습 과정에서 발생할 수 있는 기존 정보의 손실(catastrophic forgetting)을 방지합니다.

- **Technical Details**: C$^{2}$INet는 잠재 공간의 요인(confounding factors)에 대한 사전(probabilistic prior)과 사후(posterior) 추정기(estimate)를 정렬(align)하기 위해 변분 추론(variational inference)을 사용합니다. 이를 통해 궤적 표현의 인과적 상관관계(causal correlations) 개입을 통해 궤적 데이터를 더 효과적으로 캡처합니다. 또한, 주어진 시나리오에서 최적의 변별 사전(optimal variational priors)을 저장하여 지속적인 편향 제거(debiasing)를 보장합니다.

- **Performance Highlights**: 제안된 C$^{2}$INet은 세 가지 실제 및 합성 데이터셋에서 기존의 최첨단 방법과 비교했을 때 신뢰할 수 있는 예측 성능을 일관되게 도출했습니다. 이 방법은 다양한 시나리오에 고유한 혼란 요인(confounding factors)을 효과적으로 완화하여 실제 응용 프로그램에서의 가치가 강조됩니다. 결과적으로, 각기 다른 작업에 대한 적응력을 높이며, 이전의 작업 정보를 보존하여 에지에서 발생하는 문제 해결에 기여합니다.



### libcll: an Extendable Python Toolkit for Complementary-Label Learning (https://arxiv.org/abs/2411.12276)
Comments:
          10 pages, 3 figures

- **What's New**: 본 논문에서는 complementary-label learning (CLL)이라는 약한 감독 학습 방법론의 주요 문제점을 다루기 위해 	exttt{libcll}이라는 확장 가능한 파이썬 툴킷을 새롭게 소개합니다. 	exttt{libcll}은 다양한 CLL 알고리즘과 데이터셋을 지원하는 범용 인터페이스를 제공하여 일관성 문제를 해소하고, 연구 과정을 간소화하도록 설계되었습니다. 이 툴킷은 CLL 기술을 효율적으로 채택하고 구현할 수 있도록 설치가 용이하고 포괄적인 사용 가이드를 제공합니다.

- **Technical Details**: CLL은 각 레이블이 데이터 인스턴스에 속하지 않는 클래스를 나타내는 약한 감독 학습 문제로, 고급 그래프 구조와 다양한 네트워크 아키텍처를 지원합니다. 	exttt{libcll}은 사용자 정의 전이 행렬을 사용해 보완 레이블을 생성하는 기능을 포함하였으며, 다양한 CLL 알고리즘과 데이터셋에 대한 광범위한 벤치마크를 제공합니다. 이 툴킷은 학습 성능의 일관성을 유지하고 연구자들이 쉽게 비교하고 분석할 수 있도록 도와줍니다.

- **Performance Highlights**: 	exttt{libcll}을 사용한 포괄적인 ablation 연구는 CLL 연구를 발전시키기 위한 중요한 인사이트를 생성함을 입증하였습니다. 15개 데이터셋과 14개 알고리즘의 벤치마크 결과를 통해 연구자들이 각 방법의 강점과 한계를 평가할 수 있는 통합된 관점을 제공합니다. 이 툴킷은 CLL 분야에서의 연구 협력과 재현성을 촉진하며, 더 나은 알고리즘 개발에 기여할 것으로 기대됩니다.



### A Review on Generative AI Models for Synthetic Medical Text, Time Series, and Longitudinal Data (https://arxiv.org/abs/2411.12274)
Comments:
          27 pages, 3 figures

- **What's New**: 이 논문은 의료 텍스트(Medical Text), 시계열(Time Series), 그리고 종단적 데이터(Longitudinal Data)를 생성하기 위한 세 가지 유형의 합성 건강 기록(Synthetic Health Records, SHR)의 실용적인 모델에 대한 새로운 스코핑 리뷰(scoping review) 결과를 소개합니다. 본 리뷰에서는 연구 목표, 데이터 모달리티(data modality), 연구 방법론(research methodology)을 포함하여, 디지털 의학(context for digital medicine)의 맥락에서 이 주제의 중요성과 범위를 밝혀내고 있습니다.

- **Technical Details**: 총 52개의 출판물이 합성 의료 시계열(22편), 종단적 데이터(17편), 의료 텍스트(13편) 생성을 위한 자격 기준을 충족했습니다. 연구 논문들은 주로 개인 정보 보호(privacy preservation)를 주요 연구 목표로 삼았으며, 클래스 불균형(class imbalance), 데이터 부족(data scarcity), 데이터 보간(data imputation) 등의 다른 목표도 다뤘습니다. 적대적 네트워크(adversarial network), 확률적(probabilistic), 대형 언어 모델(large language models)은 각각 합성 종단적 데이터, 시계열, 의료 텍스트 생성을 위한 우수성을 보여주었습니다.

- **Performance Highlights**: SHR 재식별 리스크(re-identification risk)를 정량화하는 신뢰할 수 있는 성능 측정(challenge한 성능 측정치)을 찾는 것은 이 주제의 주요 연구 격차이며, 더 깊은 연구가 필요합니다. 이 연구는 디지털 의학 분야에서 개인 정보 보호 및 데이터 생성 모델 개발의 방향성을 제시합니다.



### Hyper-parameter Optimization for Federated Learning with Step-wise Adaptive Mechanism (https://arxiv.org/abs/2411.12244)
- **What's New**: 이 논문은 Federated Learning (FL) 내에서 Automated Machine Learning (Auto-ML)과 Hyper-Parameter Optimization (HPO) 도구를 통합하는 방법을 탐구합니다. Raytune과 Optuna라는 경량 HPO 도구 두 가지를 FL 설정에 배포하고 통합하는 연구를 진행했습니다. 또한, Auto-ML 툴킷과 FL 서버 간의 조정을 위한 단계별 피드백 메커니즘을 설계하여 하이퍼 파라미터 조정 과정을 가속화합니다.

- **Technical Details**: FL에서는 클라이언트의 원시 데이터를 공유하는 대신 로컬 모델 파라미터를 사용하여 모델을 학습합니다. 이 논문에서는 데이터가 비독립적이고 동일하게 분포되지 않는 (non-IID) 특성을 고려하여 Auto-FL의 하이퍼 파라미터 조정 과정의 복잡성을 최소화하는 방안을 제시합니다. 여기서 제안된 새로운 클라이언트 선택 기법은 Auto-FL에서 느린 클라이언트의 영향을 완화하는데 기여합니다.

- **Performance Highlights**: 제안된 HPO 도구들은 FEMNIST와 CIFAR10이라는 두 개의 기준 데이터셋을 사용하여 평가되었습니다. 연구는 성공적인 HPO 도구의 필수 특성과 FL 파이프라인과의 통합 메커니즘을 다루며, 분산 및 이질적인 FL 환경에서의 도전 과제를 논의합니다. 논문은 또한 HPO가 FL 모델의 전반적인 성능에 미치는 중요한 영향을 강조합니다.



### Contrast Similarity-Aware Dual-Pathway Mamba for Multivariate Time Series Node Classification (https://arxiv.org/abs/2411.12222)
Comments:
          Submitted to Knowledge-Based Systems on Nov 17, 2024

- **What's New**: 이번 연구는 복잡한 다차원 시간을 다루는 멀티 변수 시계열(MTS) 분류를 위한 새로운 접근 방식으로, Contrast Similarity-aware Dual-Pathway Mamba (CS-DPMamba)를 제안합니다. 이 방법은 우선, Temporal Contrast Learning을 통해 각 샘플의 동적 유사성을 캡처하고, 이어서 FastDTW를 사용하여 유사성 행렬을 구성합니다. 이 후 DPMamba 모델을 통해 MTS의 양방향 특성을 고려하여 장기적 및 단기적 종속성을 효과적으로 포착합니다.

- **Technical Details**: 연구의 목적은 MTS 데이터에서의 유사성과 장기 종속성을 결합하는 것입니다. CS-DPMamba는 Temporal Contrast Learning을 기반으로 샘플의 특성을 추출하고, Fast Dynamic Time Warping을 통해 MTS 표현 간의 유사성 행렬을 구축합니다. 이후 DPMamba와 Kolmogorov-Arnold Network 강화 그래프 동형성 네트워크를 결합하여 정보 전파와 MTS 노드 분류 작업을 수행합니다.

- **Performance Highlights**: 실험은 University of East Anglia (UEA) MTS 데이터셋에서 수행되었으며, 다양한 응용 시나리오를 포함합니다. CS-DPMamba는 기존의 방법에 비해 더욱 정교한 MTS 노드 분류를 달성했으며, 감독 및 반 감독 학습에서 그 우수성을 입증했습니다. 이러한 결과는 다차원 시계열 데이터의 복잡한 종속성을 효과적으로 다루는 방법의 필요성을 강조합니다.



### DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning (https://arxiv.org/abs/2411.12220)
Comments:
          14 pages

- **What's New**: 이 논문에서는 Federated Learning (FL) 환경에서 백도어 공격을 방어하는 새로운 프레임워크인 DeTrigger를 제안합니다. DeTrigger는 적대적 공격 방법론의 통찰력을 활용하여 백도어 트리거를 효과적으로 탐지하고 분리합니다. 이를 통해 정상 모델의 지식을 소실하지 않으면서 백도어 활성화 weight를 정밀하게 가지치기할 수 있습니다. 결과적으로 DeTrigger는 전통적인 방법보다 최대 251배 더 빠른 탐지 속도와 최대 98.9%의 백도어 공격 완화 능력을 보여줍니다.

- **Technical Details**: DeTrigger는 gradient 분석 및 temperature scaling을 사용하여 백도어 트리거를 탐지하고 격리합니다. 모델 gradient는 입력에 따른 모델 weight의 반응을 포착하여, 백도어 공격의 미세한 변화를 감지합니다. 이 방법은 각 클라이언트 모델에 대한 exhaust inspection 없이도 비정상적인 패턴을 효율적으로 탐지할 수 있는 장점을 가집니다. 또한, DeTrigger는 손상된 모델을 제거하는 대신, 백도어 활성화 weight만을 정확히 제거하여 정상 지식을 유지하는 기능을 제공합니다.

- **Performance Highlights**: DeTrigger는 네 개의 널리 사용되는 데이터셋을 대상으로 철저히 평가되어 약 98.9%의 백도어 공격 완화율을 달성하였습니다. 또한, 기존 방법들보다 대략 251배 더 빠른 탐지 속도를 자랑하며, 글로벌 모델의 정확성을 유지하면서도 백도어 공격의 영향을 크게 줄였습니다. 이러한 속도와 정확성의 조합은 DeTrigger가 세련된 백도어 위협으로부터 federated learning 환경을 보호하는 데 효과적인 솔루션임을 입증합니다.



### Diffusion-Inspired Cold Start with Sufficient Prior in Computerized Adaptive Testing (https://arxiv.org/abs/2411.12182)
Comments:
          Accepted by KDD2025

- **What's New**: 본 연구에서는 Cold Start with Insufficient Prior (CSIP) 문제를 해결하기 위해 Diffusion Cognitive States TransfeR Framework (DCSR)를 제안합니다. DCSR는 Diffusion Models (DMs)를 기반으로 하며, 다양한 도메인에서의 인지 상태 전이를 다루는 다중 도메인 전이 프레임워크입니다. 기존 CAT 시스템이 다른 코스에서 수집된 풍부한 응답 기록을 효율적으로 활용하지 못하는 문제를 인식하고, 이를 해결하기 위해 생물학적 목표에 대한 적합한 초기 능력을 복원하는 설계 구조를 제공하고 있습니다.

- **Technical Details**: DCSR는 examinee의 인지적 능력을 타겟 도메인에서 세밀하게 생성하기 위해 Causal Relationships를 분석하고 Redundant 및 Extraneous Cognitive States를 고려하여 설계되었습니다. 본 프레임워크는 다각적인 인지 상태 전환 다리(cognitive state transition bridge)를 구축하여 examinee의 인지 상태에 기반한 질문 선택 알고리즘에 통합 가능하게 합니다. 생성된 초기 상태는 CAT 시스템의 요구 사항에 맞춰 컨디셔닝되어 카오스(Causal) 정보 손실을 방지하고 임의성이 과도해지지 않도록 조절하는 일관성 제약 조건과 작업 지향 제약 조건(task-oriented constraint)을 포함하고 있습니다.

- **Performance Highlights**: DCSR는 5개의 실제 데이터 세트에서 광범위한 실험을 수행하여 기존의 baseline 방법들을 모두 초월하며 CSIP 문제를 해결하는데 효과적임을 입증했습니다. CAT 시스템의 Cold Start 성능을 향상시키며, 질문 선택 알고리즘과의 원활한 통합을 통해 CAT 시스템의 전반적인 효율성을 증가시킵니다. 연구 결과는 CAT 시스템의 초기 테스트 단계에서 examinee의 능력 평가에서 발생하는 혼란과 좌절을 줄이는 데 중요한 기여를 할 것으로 보입니다.



### Just KIDDIN: Knowledge Infusion and Distillation for Detection of INdecent Memes (https://arxiv.org/abs/2411.12174)
- **What's New**: 이번 연구에서 제안한 KID-VLM 프레임워크는 큰 비주얼 언어 모델(LVLM)로부터의 지식 증류(Knowledge Distillation, KD)와 상식 지식 그래프(Knowledge Graph, KG)에서의 지식 주입(Knowledge Infusion)을 결합하여 공격적인 메메의 독성 탐지 성능을 향상시킵니다. 이 프레임워크는 ConceptNet이라는 대규모 상식 KG에서 서브 지식 그래프를 추출하여 compact VLM 프레임워크 내에서 주입하여 독성 문구와 메메 내의 시각적 개념 간의 관계적 맥락을 강화합니다.

- **Technical Details**: KID-VLM은 CLIP을 백본으로 사용하여 메메의 시각적 및 텍스트 피쳐를 추출하며, LLaVA 교사 모델로 생성된 캡션을 위한 텍스트 인코더와 메메 텍스트를 위한 또 다른 텍스트 인코더를 사용합니다. 피쳐 상호작용 매트릭스(Feature Interaction Matrix, FIM)를 계산하여 시각적 및 텍스트 데이터를 정렬하고, 일관성 손실(consistency loss)을 사용하여 교사 모델의 지식을 정제하여 학생 모델이 내재된 맥락적 단서를 포착하도록 학습합니다.

- **Performance Highlights**: 두 개의 증오 발언 기준 데이터셋에서 평가한 결과, KID-VLM 프레임워크는 AU-ROC, F1, 리콜에서 각각 1.1%, 7%, 35% 향상된 성능을 보여주며 기존 최첨단 모델들을 초월하는 성과를 기록했습니다. 이러한 성과는 두 가지 맥락 학습 접근법의 필요성을 강조하며, LVLM으로부터의 잠재적 패턴과 KG로부터의 외적 관계 지식을 함께 캡처합니다.



### SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks (https://arxiv.org/abs/2411.12173)
- **What's New**: 이 논문에서는 SkillTree라는 새로운 프레임워크를 제안합니다. SkillTree는 복잡한 연속 행동 공간을 이산(skill) 공간으로 축소하여 결정 트리(decision tree)를 통합하였습니다. 이로 인해 복잡한 작업에서의 의사 결정 과정을 더욱 투명하게 만들었습니다.

- **Technical Details**: SkillTree는 계층적 접근 방식을 채택하여 고급 정책 내에 미분 가능 결정 트리를 통합하고 이를 통해 기술 임베딩(skill embeddings)을 생성합니다. 이 기술 임베딩은 저수준 정책이 기술을 실행하는 데 필요한 지침을 제공합니다. 또한, 기술 공간을 이산 단위로 정규화하여 정책 학습을 간소화하고 학습된 기술의 설명 가능성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SkillTree는 복잡한 로봇 암 제어 작업에서 기술 기반 신경망과 유사한 성과를 달성했습니다. 더욱이, SkillTree는 기술 수준의 설명을 제공하여 결정 과정의 투명성을 증가시킵니다. 이는 안전이 중요한 애플리케이션에서의 신뢰성을 높이는 데 기여합니다.



### UrbanDiT: A Foundation Model for Open-World Urban Spatio-Temporal Learning (https://arxiv.org/abs/2411.12164)
- **What's New**: UrbanDiT는 복잡한 도시 환경의 시공간 동적을 효과적으로 모델링하기 위한 새로운 기초 모델로 소개됩니다. 이 모델은 다양한 시공간 데이터 소스와 유형을 통합하며, 여러 도시와 시나리오에서 보편적인 시공간 패턴을 학습할 수 있습니다. 또한, UrbanDiT는 데이터와 작업에 특화된 프롬프트(prompt)를 생성하는 혁신적인 프롬프트 학습(framework) 구조를 통해 다양한 도시 응용 프로그램에서 뛰어난 성능을 발휘합니다.

- **Technical Details**: UrbanDiT는 그리드 기반(grid-based) 및 그래프 기반(graph-based) 데이터와 같은 다양한 데이터 유형을 통합하여 시퀀스 형식으로 변환합니다. 이 모델은 양방향(spatio-temporal) 예측, 시간 보간(temporal interpolation), 공간 외삽(spatial extrapolation), 시공간 보간(spatio-temporal imputation)와 같은 다양한 작업을 지원하기 위해 마스킹 전략과 작업 특화 프롬프트를 사용합니다. 또한, 이 모델은 오픈 월드 시나리오(open-world scenarios)에 효과적으로 일반화(generalize)되며, 제로샷(zero-shot) 능력으로 훈련 데이터와 비교해 더 뛰어난 성능을 보입니다.

- **Performance Highlights**: UrbanDiT는 교통 상황(transportation traffic), 인구 흐름(crowd flows), 택시 수요(taxi demand), 자전거 사용(bike usage), 그리고 셀룰러 트래픽(cellular traffic) 등 다양한 분야에서 주목할 만한 성능을 달성합니다. 이 모델은 여러 도시와 작업에서 최신 최첨단 성능(state-of-the-art performance)을 기록하며, 도시 시공간 분야의 기초 모델에 대한 새로운 기준을 세웁니다.



### Reinforcement Learning with Action Sequence for Data-Efficient Robot Learning (https://arxiv.org/abs/2411.12155)
Comments:
          17 Pages. Website: this https URL

- **What's New**: 이 논문은 로봇 작업에서 강화 학습(RL)의 데이터 효율성을 향상시키기 위한 새로운 RL 알고리즘을 제시합니다. 이 알고리즘은 일련의 행동에 대한 Q값을 출력하는 비평자 네트워크(critic network)를 학습하여, 불규칙한 데이터에서 유용한 가치 함수(value function)를 학습할 수 있게 합니다. 최근의 행동 복제(behavior-cloning, BC) 접근 방식에서 영감을 받아, 불확실한 전문가의 시연을 효과적으로 근사할 수 있는 방법을 조사합니다.

- **Technical Details**: 이 논문에서 제안한 알고리즘은 CQN-AS(Coarse-to-fine Q-Network with Action Sequence)라고 불립니다. 이 알고리즘은 현재와 미래의 행동 시퀀스를 실행할 때의 결과를 명시적으로 학습해, RL 에이전트가 noisy trajectory에서 유용한 가치 함수를 학습할 수 있도록 돕습니다. 실험은 BiGym, HumanoidBench 및 RLBench 등 다양한 설정에서 진행되었으며, 희소 및 밀집 보상(sparse and dense rewards) 환경에서의 성능을 평가했습니다.

- **Performance Highlights**: CQN-AS는 여러 실험에서 이전의 RL 알고리즘 및 BC 기준선을 초과하는 성능을 보였습니다. 특히 인간이 수집한 데이터가 포함된 모바일 양손 조작 과제와, 밀집 보상이 주어지는 사람 모양 제어 과제에서 두드러진 성과를 기록했습니다. 또한, RLBench에서는 합성 데이터가 사용되었으나, 여전히 많은 작업에서 우수한 성능을 달성했습니다.



### Visualizing Loss Functions as Topological Landscape Profiles (https://arxiv.org/abs/2411.12136)
- **What's New**: 이 논문은 기계 학습에서 손실 함수(loss function)의 시각화를 위한 새로운 방법론을 제안합니다. 기존의 방법들은 일반적으로 단일 또는 이차원 방향으로만 샘플링하는 반면, 본 연구에서는 위상적 데이터 분석(topological data analysis, TDA)을 기반으로 한 더 높은 차원의 손실 풍경(loss landscape)을 시각화할 수 있는 새로운 표현 방식을 도입합니다. 이 접근 방식은 손실 풍경이 모델 성능 및 학습 역학을 어떻게 반영하는지를 새로운 방식으로 탐구합니다.

- **Technical Details**: 위상적 데이터 분석(TDA)의 개념을 활용하여, 손실 풍경의 중요한 특징을 포착하는 머지 트리(merge tree)를 사용합니다. 이 트리는 손실 풍경의 임계점을 인코딩하고, 이를 두 차원으로 재표현하는 위상적 풍경 프로파일(topological landscape profile)로 나타냅니다. 이를 통해 연구자들은 손실 함수의 형태를 더 깊이 이해할 수 있으며, 특히 다차원 공간에서의 복잡한 정보를 효과적으로 시각화할 수 있습니다.

- **Performance Highlights**: 실험을 통해, 잘 작동하는 모델의 손실 풍경은 상대적으로 간단한 위상을 가지며, 낮은 성능에서 높은 성능으로 전환되는 지점 근처에서 위상적 변동성이 더 큼을 발견했습니다. 또한, 비슷한 물리적 매개변수를 통해 저오차(low error) 및 고오차(high error) 모델의 손실 풍경 모양의 차이를 관찰했습니다. 이러한 발견은 모델의 성능에 대한 새로운 통찰력을 제공하며, 특히 다양한 하이퍼파라미터에 따른 손실 풍경의 변화를 명확하게 제시합니다.



### Fine-Grained Uncertainty Quantification via Collisions (https://arxiv.org/abs/2411.12127)
- **What's New**: 새로운 접근 방식으로 제안된 Collision Matrix(충돌 행렬)가 있습니다. 이 행렬은 K개의 클래스가 포함된 분류 문제에 대해 각 클래스 쌍을 구별하는 어려움을 정밀하게 측정할 수 있습니다. 기존의 불확실성 정량화 방법들과 비교해보면, Collision Matrix는 분류의 어려움을 훨씬 더 세부적으로 나타낼 수 있습니다.

- **Technical Details**: Collision Matrix는 클래스 간의 기본적인 충돌 확률을 측정하는 K×K 행렬입니다. 이 행렬의 (i,j) 항목은 클래스 i와 클래스 j 간의 요소를 구별하는 난이도를 나타냅니다. 새로운 방법으로, 이 행렬을 일관되게 추정하기 위해 contrastive binary classifier(대조 이진 분류기)를 학습할 수 있는 방법을 제안하고 있습니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 유효성을 입증하기 위한 실험 결과를 제시합니다. 이 결과들은 Collision Matrix가 불확실성을 정량화하는 데에 있어서의 혁신적인 기여도를 보여줍니다. 또한, 이 방법은 클래스 간 혼동을 예측하고 이를 통해 분류 성능을 향상시키는 데 기여할 수 있음을 시사합니다.



### MMBind: Unleashing the Potential of Distributed and Heterogeneous Data for Multimodal Learning in Io (https://arxiv.org/abs/2411.12126)
- **What's New**: 이번 논문에서는 IoT 데이터를 이용한 분산되고 이질적인 다중모드 학습을 위한 새로운 프레임워크인 MMBind를 제안합니다. MMBind의 주요 아이디어는 서로 다른 출처와 불완전한 모드의 데이터를 결합하여 보다 설명력 있는 공유 모드를 통해 모델 학습을 위한 의사 쌍 데이터셋을 생성하는 것입니다. 이는 기존의 많은 다중모드 학습 접근 방식들이 방대한 양의 완전한 데이터를 요구하는 것과 대조적입니다.

- **Technical Details**: MMBind는 분산된 IoT 데이터에서 다중모드 학습을 위한 새로운 접근 방식을 특징으로 하며, 주어진 데이터의 부족 또는 모드 결여를 극복하는 두 단계의 학습 전략을 채택합니다. 첫 번째 단계에서는 공유 모드를 통해 불완전한 데이터를 연결하고, 두 번째 단계에서는 이질적인 쌍 데이터에 대한 가중 대비 학습을 수행합니다. 이러한 아키텍처는 다양한 모드 조합을 지원하며, 모델 학습의 강건성을 높이는 다양한 기여도를 반영합니다.

- **Performance Highlights**: 실험 결과, MMBind는 여러 최신 기법들보다 최대 20% 정확도가 향상되었으며, 데이터의 불완전성과 도메인 이동 등의 변화에서도 우수한 성능을 나타냈습니다. 이는 MMBind가 여러 다른 다중모드 데이터셋을 효과적으로 처리할 수 있음을 보여주며, IoT 애플리케이션에서 다중모드 기초 모델 학습을 촉진할 가능성을 보여줍니다.



### Mechanism and Emergence of Stacked Attention Heads in Multi-Layer Transformers (https://arxiv.org/abs/2411.12118)
- **What's New**: 이 논문은 트랜스포머가 최소한의 레이어로 해결할 수 있는 데이터 검색 문제를 도입합니다. 또한, 다양한 프롬프트 형식으로 대규모 언어 모델이 특정 미세 조정 없이 이 문제를 해결할 수 있음을 입증합니다. 중요한 점은 트랜스포머는 암묵적 커리큘럼의 존재 하에 학습 과정을 통해 특유의 메커니즘 즉, retrieval heads를 학습한다는 것입니다.

- **Technical Details**: 이 논문은 모델이 입력 시퀀스에 대해 다음 토큰을 예측하는 retrieval 문제를 정의합니다. 여기에서 D단계까지의 유도 단계를 증가시킴으로써 일반화된 conditional retrieval 문제를 제안합니다. 이 문제들은 비선형 그래프 구조를 형성하여 더 복잡한 관계를 다룰 수 있게 됩니다.

- **Performance Highlights**: 대규모 언어 모델들은 제안된 5가지 retrieval 문제 형식을 사용하여 효과적인 성능을 보입니다. 연구자는 학습된 attention maps를 통해 모델이 특정 순서로 retrieval heads를 생성하는 과정을 시각적으로 분석했습니다. 또한, 이 메커니즘은 모델의 훈련 중에 명확하게 드러나는 것을 확인하였습니다.



### BALI: Learning Neural Networks via Bayesian Layerwise Inferenc (https://arxiv.org/abs/2411.12102)
- **What's New**: 이번 연구에서는 다변량 Bayesian 선형 회귀 모델 스택으로 Bayesian 신경망(Bayesian Neural Networks, BNN)을 학습하는 새로운 방법인 BALI(Bayesian Layerwise Inference)를 제안합니다. 이 방법은 각 레이어의 목표 출력을 알고 있을 경우 레이어별 후행 분포(posterior distribution)를 정확히 추론하는 것이 핵심입니다. 또한, BALI는 기존의 BNN 방법들과 비교했을 때 유사하거나 더 나은 성능을 보이며 빠르게 수렴하는 특징이 있습니다.

- **Technical Details**: BALI는 각 레이어에서 국소 후행 분포를 분석적으로 추정할 수 있으며, 이는 앞서 전달된 피처와 다음 레이어에서 역전파된 가상의 목표(pseudo-target)에서 파생됩니다. 이러한 산출에서 얻어진 레이어별 후행 분포는 다변량 Bayesian 선형 회귀 문제의 해로 표현되며, Kronecker-팩터화 된 공분산 행렬을 갖는 매트릭스 정규 분포로서 효율적인 역행렬 계산이 가능합니다. 또한, 이 방법은 자연 매개변수 용어에 대한 지수 이동 평균을 사용하여 확률적 미니 배치 환경으로 확장합니다.

- **Performance Highlights**: BALI는 회귀, 분류 및 분포 외 탐지(out-of-distribution detection) 작업에 대해 평가되었으며, 강력한 BNN 기준 모델들과 비교해 유사하거나 더 나은 성능을 보여줍니다. 이 방법은 각 작업에서 적은 반복(iteration)으로 수렴하며, 이는 실용적인 응용 프로그램에서 높은 효율성을 나타내는 중요 요소입니다. 또한, 제안된 방법은 대규모 모델 및 데이터셋에 대한 확장성 문제를 해결하는 데 기여합니다.



### Federated Contrastive Learning of Graph-Level Representations (https://arxiv.org/abs/2411.12098)
Comments:
          Accepted in BigData 2024. This is a preprint

- **What's New**: 이 논문은 그래프 수준 표현을 위한 연합 대조 학습(FCLG)이라는 새로운 프레임워크를 제안합니다. 기존의 연합 학습은 주로 지도 학습에 중점을 두었지만, 이 연구는 비지도 학습 설정에서 그래프 표현을 학습할 수 있는 가능성을 탐구합니다. 특히, 두 가지 수준의 대조 학습을 적용하여 로컬 모델의 데이터 분포 변동 문제를 해결하고자 합니다.

- **Technical Details**: FCLG 프레임워크는 로컬 비지도 학습 및 글로벌 모델과의 상호 대조를 통해 그래프 수준 표현을 학습합니다. 이 방식은 `Non-IID`(비독립적이고 동일한 분포) 문제를 해결하여 각 클라이언트의 데이터가 서로 다른 분포를 가질 때 일반화 성능을 개선합니다. 대조 학습은 긍정적 샘플 쌍의 거리를 줄이고 부정적 샘플 쌍의 거리를 늘리는 방식으로 작동하여 더 구별 가능한 표현을 생성합니다.

- **Performance Highlights**: 다양한 공공 벤치마크 데이터 세트를 사용한 실험을 통해 FCLG는 다른 기존의 연합 학습 방식들보다 그래프 수준 클러스터링 작업에서 일관되게 더 뛰어난 성능을 보여주었습니다. 논문에서는 FCLG의 두 수준 대조 메커니즘이 데이터를 하이브리드로 통합하고 전반적인 성능을 개선하는 데 어떻게 기여하는지를 강조하고 있습니다.



### Molecule Generation with Fragment Retrieval Augmentation (https://arxiv.org/abs/2411.12078)
Comments:
          NeurIPS 2024

- **What's New**: F-RAG는 새로운 분자 생성 프레임워크로, 기존 분자 조각을 넘어서 새로운 고품질 분자를 탐색할 수 있는 방식을 제안합니다. 기존의 프래그먼트 기반 약물 발견 방법론의 한계를 극복하기 위해, 이 방법은 프래그먼트 수집과 생성을 결합하여 고유한 분자 조합을 만들 수 있게 합니다. 또한, 이를 통해 실제 투여 가능한 약물 후보물질들의 다양성과 독립성을 극대화할 수 있습니다.

- **Technical Details**: F-RAG는 두 가지 유형의 프래그먼트를 사용하여 새로운 분자를 생성하는 방식을 채택합니다. 하드 프래그먼트(hard fragments)는 생성될 분자에 명시적으로 포함될 구조를 제공하며, 소프트 프래그먼트(soft fragments)는 새로운 분자의 생성을 유도하는 참고 자료로 작용합니다. 이러한 접근 방식은 개인화된 분자 생성을 가능하게 하고, 백험 기반 약물 발견(FBDD)의 탐색-활용(exploration-exploitation) 균형을 크게 향상시킵니다.

- **Performance Highlights**: F-RAG는 다양한 분자 최적화 작업을 통해 최적화 성능, 다양성, 참신성 및 합성 가능성을 평가하여 뛰어난 성능을 입증하였습니다. 이 방법은 특히 실제 상황을 시뮬레이션한 약물 발견 과제에서 요망되는 조건들을 모두 만족시키며, 훨씬 더 나은 약물 후보물질을 생산할 수 있는 가능성을 보여줍니다.



### Theoretical Corrections and the Leveraging of Reinforcement Learning to Enhance Triangle Attack (https://arxiv.org/abs/2411.12071)
- **What's New**: 이 논문에서는 Triangle Attack (TA)의 한계를 분석하고, 새로운 공격 방식인 Triangle Attack with Reinforcement Learning (TARL)을 제안합니다. TARL은 강화 학습을 활용하여 TA의 효율성을 높이며, 기존 상태의 기계 학습 모델에서 더 적은 쿼리로 유사한 공격 정확도를 달성할 수 있도록 설계되었습니다. 연구 결과 TARL은 ImageNet과 CIFAR-10 데이터셋에서 TA보다 더 나은 공격 성능을 보여주었습니다.

- **Technical Details**: TARL은 TA의 제약을 극복하기 위해 강화 학습 기법을 활용하여, 최적의 공격을 수행합니다. TA는 Discrete Cosine Transform (DCT)을 사용하여 이미지의 차원을 축소하고, 그 후 대규모 교란을 적용하여 최적화를 진행합니다. TA의 알고리즘은 매 반복 시 후보 공격 예제 간의 기하학적 정보를 사용하여 유사한 공격을 생성합니다.

- **Performance Highlights**: TA는 1,000개의 쿼리로 높은 성공률을 달성한 반면, TARL은 500개의 쿼리로 유사한 성능을 보여줍니다. 실험을 통해 TARL의 공격 성능은 다양한 모델과 데이터셋에서 TA 대비 개선된 것을 입증했습니다. 특히, TARL은 기존 공격 방식이 갖고 있던 쿼리 효율성의 제약을 극복하며 보다 강력한 공격 결과를 제공합니다.



### Interpretation of High-Dimensional Regression Coefficients by Comparison with Linearized Compressing Features (https://arxiv.org/abs/2411.12060)
Comments:
          This manuscript is a short communication. 9 pages, 4 figures

- **What's New**: 이 논문에서는 고차원 기능 데이터에서 비선형 응답을 근사하기 위한 선형 회귀의 특성을 심층적으로 탐구합니다. 리튬 이온 배터리의 사이클 수명을 예측하기 위해 고차원 데이터를 사용하며, 비선형 압축 기능을 통해 단일 응답을 생성하는 방법에 초점을 맞추고 있습니다. 선형화 방법론을 개발하여 특성 계수를 유도하고, 이를 회귀 해법의 가까운 회귀 계수와 비교합니다.

- **Technical Details**: 연구는 먼저 비선형 압축 기능을 정의하고, 이를 바탕으로 데이터와 응답 간의 관계를 탐구합니다. 선형 회귀 모델을 사용하여 고차원 데이터의 회귀 계수를 분석하고, 이러한 계수가 어떻게 입력의 변화에 따라 반응을 학습하는지를 살펴봅니다. 첫 번째 차수의 테일러 전개를 활용하여 선형화된 기능 계수를 도출하고, 이러한 계수를 통해 회귀 계수를 비교합니다.

- **Performance Highlights**: 리튬 이온 배터리 데이터를 활용한 사례 연구에서, 비선형 압축 기능이 사이클 수명 예측에 높은 예측력을 나타냈습니다. 연구 결과는 고차원에서도 해석 가능성과 모델의 성능을 동시에 개선할 수 있는 방법론을 제시합니다. 또한, highly regularized domain에서의 회귀 계수의 거동을 이해하는 데 도움을 주어, 해석 가능성을 높이는 기초 자료를 제공합니다.



### Higher Order Graph Attention Probabilistic Walk Networks (https://arxiv.org/abs/2411.12052)
- **What's New**: 본 논문에서는 Higher Order Graph Attention (HoGA) 모듈을 제안합니다. 이 모듈은 feature-vector의 다양성을 기반으로 샘플링된 가변 길이 경로에 가중치를 할당하여 $k$-hop 이웃을 효과적으로 재구성합니다. 기존의 지역 정보에만 의존하던 접근 방식의 한계를 극복하여 원거리 정보를 보존할 수 있는 가능성을 보여줍니다.

- **Technical Details**: HoGA는 전체 $k$-hop 이웃을 하나의 feature-vector로 집계하는 대신, 최단 경로의 self-attention 가중치를 학습하여 $k$-hop 이웃을 효율적으로 매개변수화합니다. 이 과정에서 고정된 topology 서브구조를 사용하는 기존의 $k$-order attention 기법과는 달리, 다양한 모드의 정보를 캡처하여 네트워크의 일반화 능력을 향상시킵니다. 감쇠 효과(oversquashing)와 부드러움(oversmoothing) 문제를 감소시키는 중요한 기능을 합니다.

- **Performance Highlights**: HoGA를 기존의 최신 attention 모델에 통합했을 때, 노드 분류 데이터셋의 정확성이 현저히 증가하는 결과를 보였습니다. 추가 실험을 통해 레이어 수의 변화와 노드 샘플링 방법의 정확도에 미치는 영향을 분석하여 호가 모듈의 효과를 시각적으로 입증합니다. HoGA는 원거리 정보와 고차 관계를 효과적으로 활용하여 노드 분류 작업에서 중요한 성능 향상을 가져오는 간단하고 효과적인 방법임을 확인했습니다.



### Fast Convergence of Softmax Policy Mirror Ascen (https://arxiv.org/abs/2411.12042)
- **What's New**: 이 논문에서는 정책 경량화 기법인 Softmax Policy Mirror Ascent(SPMA)를 제안하고 있습니다. 기존의 NPG(자연 정책 기울기) 알고리즘을 기반으로 하여, 액션 사이의 정규화 필요성을 제거한 개선된 형태입니다. SPMA는 선형 수렴률을 가지며, 기존의 소프트맥스 정책 기울기(smooth policy gradient)보다 더 빠른 수렴을 달성하는 것으로 입증되었습니다.

- **Technical Details**: SPMA는 다중 무장 밴딧(multi-armed bandit)과 테이블 형태의 MDP에 대해 개발되었습니다. 정책을 로그 확률의 이중 공간에서 미러 상승(mirror ascent) 업데이트를 통해 최적의 정책으로 수렴합니다. 또한, 상태-행동 공간이 큰 MDP를 처리하기 위한 함수 근사(function approximation) 기법을 적용하며, 이는 복잡한 비선형 함수 근사에도 확장됩니다.

- **Performance Highlights**: SPMA는 Atari와 MuJoCo와 같은 벤치마크에서 실험적으로 평가되어, 기존 알고리즘인 MDPO, PPO, TRPO보다 유사하거나 더 나은 성능을 달성하는 것으로 나타났습니다. 특히, Atari 게임에서 SPMA는 TRPO와 PPO보다 나은 성과를 보였으며, MuJoCo 작업에서는 PPO를 초과하는 성과를 기록하였습니다.



### Scaling Deep Learning Research with Kubernetes on the NRP Nautilus HyperCluster (https://arxiv.org/abs/2411.12038)
- **What's New**: 이 연구에서는 NRP Nautilus HyperCluster를 활용하여 깊은 신경망(DCNN)의 모델 훈련을 자동화하고 확장하는 새로운 접근 방식을 탐구하고 있습니다. 이를 통해 소실 감지, 화재 피해 지역 세분화 및 삼림 파괴 탐지의 세 가지 응용 분야에 대한 연구를 수행했습니다. Nautilus는 1,300개 이상의 NVIDIA GPU와 19,000개 CPU 코어를 갖춘 Kubernetes 클러스터로, 현재까지 총 4,040시간의 훈련 시간을 기록했습니다.

- **Technical Details**: 모델 훈련에 있어 DCNN의 컴퓨팅 요구량이 증가하였으며, 이는 과학적 연구에서 병목 현상이 되고 있습니다. 연구에서는 DCNN 기반의 모델을 훈련하기 위해 Nautilus의 강력한 하드웨어를 활용하여 다양한 응용 프로그램을 개발하였습니다. 특성 추출 및 탐지 기법을 통해 transformer 아키텍처의 성능을 평가하기 위해 수많은 아키텍처를 훈련해야 할 필요성이 있습니다.

- **Performance Highlights**: 이번 연구의 주요 성과로는 세 가지 응용 분야에서 DCNN을 활용한 234개의 모델이 훈련되었으며, 3,000 GPU 시간 이상의 컴퓨팅 자원이 소모되었습니다. DCNN의 변화를 반영하여, 다양한 훈련 데이터셋과 함께 각각의 아키텍처에 대한 비교 분석이 이루어졌습니다. 연구 결과는 대량의 위성 이미지를 효과적으로 처리하고 다양한 실제 응용 프로그램을 발전시키는 데 기여할 것으로 전망됩니다.



### Machine Learning Evaluation Metric Discrepancies across Programming Languages and Their Components: Need for Standardization (https://arxiv.org/abs/2411.12032)
Comments:
          This paper is 12 pages with 1 table and 10 figures

- **What's New**: 이번 연구는 머신러닝(Machine Learning)에서의 다양한 작업, 즉 분류(classification), 회귀(regression), 군집화(clustering), 상관 분석(correlation analysis), 통계적 테스트(statistical tests), 분할(segmentation), 이미지 간 변환(image-to-image translation) 등을 위한 평가 지표(metrics)를 평가했습니다. 연구 결과는 다양한 플랫폼에서 신뢰할 수 있는 재현 가능한 ML 평가를 보장하기 위해 지표의 표준화를 요구합니다.

- **Technical Details**: 연구에서는 Python 라이브러리, R 패키지, Matlab 함수 간의 지표를 비교하여 일관성을 평가하였습니다. 발견된 주요 일관된 지표로는 이진 분류에서 Accuracy, Balanced Accuracy, Cohen's Kappa, F-beta Score 등이 있습니다. 다중 분류, 회귀, 군집화 및 통계적 테스트에 대한 일관된 지표들도 제시되었습니다.

- **Performance Highlights**: 여러 작업에 걸쳐 다른 플랫폼에서의 지표 일관성을 확인한 결과, 일부 지표에서는 차이가 발견되었습니다. 예를 들어, 이진 분류의 precision, recall 및 F1 score, 군집화의 WCSS, 세분화(Segmentation)에서의 IoU 등이 그 예입니다. 이 연구는 이러한 불일치를 해결하기 위해 일관된 지표의 사용이 필요함을 강조하며, 미래 연구에서의 일관된 지표 사용을 권장합니다.



### The Generalization Error of Machine Learning Algorithms (https://arxiv.org/abs/2411.12030)
Comments:
          Submitted to the IEEE Transaction on Information Theory. November 18, 2024

- **What's New**: 이 논문에서는 머신러닝 알고리즘의 일반화 오류(generalization error)를 정보 이론(information theory) 관점에서 접근하여 닫힌 형태(closed-form expressions)로 표현하는 새로운 방법인 'gap 방법(method of gaps)'을 제안합니다. 이 방법은 기대(empirical risk)의 두 기대 간의 차이를 활용하여, 각기 다른 확률 측정(probability measure)으로 계산된 기대 값을 비교함으로써 일반화 오류를 정의합니다.

- **Technical Details**: 제안된 gap 방법에는 알고리즘 기반(gaps of expected empirical risk with respect to models)과 데이터 기반(gaps of expected empirical risk with respect to datasets)의 두 가지 변형이 있습니다. 알고리즘 기반의 간극은 알고리즘 간 전이의 영향을 정량화하는 반면, 데이터 기반의 간극은 데이터 세트의 통계적 특성의 변화를 반영합니다. 이 두 가지 변형 모두 상대 엔트로피(relative entropy)와 같은 정보를 통한 명확한 표현을 제공하는 것이 특징입니다.

- **Performance Highlights**: 이 방법을 통해 기존의 모든 머신러닝 알고리즘의 일반화 오류에 대한 정확한 표현을 도출할 수 있으며, 많은 새로운 정확한 표현도 생성할 수 있습니다. 결과적으로 이 연구는 일반화 오류를 더 깊이 이해하고 통계의 다른 분야와의 연결을 수립하는 데 기여하며, 알고리즘 설계에 있어서도 잠재적으로 도움이 될 수 있습니다.



### Transmission Line Outage Probability Prediction Under Extreme Events Using Peter-Clark Bayesian Structural Learning (https://arxiv.org/abs/2411.11980)
- **What's New**: 최근 기후 변화로 인해 극단적인 기상 현상이 증가하고 있으며, 이로 인해 전력망의 신뢰성에 위협이 되고 있습니다. 연구에서는 Bayesian network (베이지안 네트워크)와 Peter-Clark (PC) 구조 학습을 결합하여 전송선의 정전 확률을 예측하는 새로운 접근 방식을 제안합니다. 이 방법은 정확한 정전 확률 계산을 가능하게 하며, 제한된 데이터에서도 뛰어난 확장성과 강건한 성능을 보입니다.

- **Technical Details**: Bayesian network (베이지안 네트워크)는 확률적 그래프 모델로, 기상 요소와 전송선 정전 간의 상관관계를 분석하여 신뢰성 예측에 사용됩니다. 본 논문에서는 PC 알고리즘을 사용하여 Bayesian network 구조 학습의 효율성과 확장성을 향상시키고, 기존의 일반적인 위험 수준 제공을 넘어 특정 정전 확률을 계산합니다. 이러한 접근법은 극단적인 기상 조건에서도 효과적이며, 제한된 정전 데이터로도 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: BPA 및 NOAA의 데이터를 사용한 사례 연구 결과, 본 연구의 방법은 기존 방법들과 비교하여 정전 예측의 성능이 더욱 우수한 것으로 나타났습니다. 다양한 기상 요인을 고려한 데이터 기반 기계 학습 기법을 활용함으로써 전송선 정전 예측에서 그 유용성을 입증하였습니다. 특히, 제한된 데이터를 다루는 상황에서도 경쟁력 있는 성능을 보여줍니다.



### Coverage-Constrained Human-AI Cooperation with Multiple Experts (https://arxiv.org/abs/2411.11976)
- **What's New**: 본 논문에서는 Coverage-constrained Learning to Defer and Complement with Specific Experts (CL2DC) 방법을 제안하여 기존 HAI-CC 방법의 연구 공백을 해결합니다. CL2DC는 AI 예측 단독, 특정 전문가에게 위임(deferring) 또는 보완하는 방식으로 최종 결정을 내리며, 입력 데이터에 따라 적절한 결정을 할 수 있도록 하였습니다. 또한, 협력 비용을 제어하기 위한 coverage-constrained 최적화 방법을 도입하여 AI 전용 선택에 대한 목표 확률을 근사화합니다.

- **Technical Details**: CL2DC는 noisy-label 주석이 포함된 훈련 세트를 고려하여, 최종 결정을 자율적으로 내리거나 특정 전문과 협력할 수 있는 시스템을 제공합니다. 이 방법은 각 전문가의 특정 전문성을 평가하고 가장 적합한 전문가를 선택하는 동시에, 손실 함수에 커버리지 제약 패널티를 도입하여 커버리지 수준을 효과적으로 조절합니다. 이렇게 하여 CL2DC는 커버리지-정확도 곡선을 분석하는 데 있어 일관되고 의미 있는 분석이 가능합니다.

- **Performance Highlights**: CL2DC는 다양한 실제 및 합성 다중 조율자 노이즈 레이블 벤치마크에서 최신 HAI-CC 방법과 비교하여 높은 정확도를 달성함을 보여주었습니다. 특히 CIFAR-100, Galaxy Zoo, HAM10000, NIH-ChestXray와 같은 데이터 세트에서 동일한 커버리지 값에 대해 CL2DC가 이전 HAI-CC 방법을 지속적으로 능가했습니다. 이러한 성과는 여러 전문가 지식을 활용하여 보다 견고한 의사 결정을 가능하게 합니다.



### Introducing Milabench: Benchmarking Accelerators for AI (https://arxiv.org/abs/2411.11940)
- **What's New**: Mila는 AI 연구에 특화된 커스텀 벤치마킹 스위트인 Milabench를 개발하였습니다. 이 스위트는 1,000명이 넘는 연구자들의 다양한 요구를 충족시키기 위해 설계되었고, 867개의 논문을 바탕으로 데이터가 수집되었습니다. Milabench는 26개의 주요 벤치마크와 16개의 추가 벤치마크를 포함하여, 연구자들이 실제 세계에서 사용하는 환경을 반영하도록 제작되었습니다.

- **Technical Details**: Milabench는 AI 연구의 폭넓은 주제를 반영하기 위해 여러 문헌 및 설문을 분석하여 벤치마크를 선정했습니다. 벤치마크는 Graphics Processing Units (GPU)의 고능률 특성을 활용하여, 다양한 AI 워크로드를 측정할 수 있도록 구성되었습니다. 2023년 발표된 867개의 논문을 통해 주요 도메인 및 모델 아키텍처에 대한 통계가 수집되고, GPT-4o를 통해 정량화하여 벤치마크의 품질을 보장하였습니다.

- **Performance Highlights**: Milabench의 벤치마크는 Reinforcement Learning, Computer Vision, Natural Language Processing와 같은 주요 도메인에서 실질적인 연구 비율을 반영하고 있습니다. 연구에서는 84%에서 98%의 재현율과 86%에서 100%의 정확도를 기록하며, 모형 아키텍처에 대해서도 50%에서 100%의 변동성을 보였습니다. 특히 Reinforcement Learning에 대한 비율이 높아지면서 Milabench의 중요성이 더욱 강조되고 있습니다.



### Value Imprint: A Technique for Auditing the Human Values Embedded in RLHF Datasets (https://arxiv.org/abs/2411.11937)
- **What's New**: 이 논문에서는 RLHF(Reinforcement Learning From Human Feedback) 데이터셋에 내재된 인간의 가치들을 감사(audit)하고 분류(classify)하기 위한 새로운 프레임워크인 Value Imprint를 소개합니다. 연구자들은 이 프레임워크의 유효성을 검토하기 위해 세 가지 사례 연구를 수행하며, Anthropic/hh-rlhf, OpenAI WebGPT Comparisons, Alpaca GPT-4-LLM 데이터셋을 분석하였습니다. 이를 통해 데이터셋에 내재된 가치들이 사람들의 가치관과 어떻게 다를 수 있는지를 탐구했습니다.

- **Technical Details**: 저자들은 철학, 가치론(axiology), 윤리학 등 과거 연구로부터 통합된 문헌검토를 통해 인간 가치의 세목을 발전시키고, 이를 통해 6,501개의 RLHF 선호 사항을 주석(annotation)하였습니다. 두 번째 단계에서는 주석 처리된 데이터를 기반으로 변환기(transformer) 기반의 머신러닝 모델을 훈련시켜 세 가지 RLHF 데이터셋을 감사하고 분류하는 데 사용하였습니다. 이 분석을 통해 정보 유틸리티 정보(Information Seeking, Wisdom/Knowledge) 가치가 가장 지배적임을 발견했습니다.

- **Performance Highlights**: 모델의 분류 정확도는 약 80%에 달하며, 이는 AI 연구자들이 RLHF 데이터셋에 내재된 인간 가치를 검토하는 데 이 과정을 채택할 수 있음을 보여줍니다. 연구 결과, RLHF 선호사항에서 구 civility & tolerance, empathy & helpfulness, justice & human rights/animal rights, well-being & peace 등은 가장 적게 나타났습니다. 이러한 발견은 언어 모델이 사회적 가치 및 규범에 부합하도록 개발될 수 있도록 중요한 통찰력을 제공합니다.



### METEOR: Evolutionary Journey of Large Language Models from Guidance to Self-Growth (https://arxiv.org/abs/2411.11933)
- **What's New**: 본 논문에서는 METEOR라는 새로운 자기 진화 방법론을 제안합니다. 이는 LLM이 단계적으로 전문 지식을 습득하고 자율적으로 발전할 수 있도록 돕기 위한 구체적인 훈련 프레임워크를 제공합니다. 특히, 이 방법은 데이터 증류(knowledge distillation) 및 자기 학습 방식으로 모델의 도메인 전문성을 향상시킵니다.

- **Technical Details**: METEOR 방법론은 두 가지 훈련 단계와 자기 진화 전략으로 구성됩니다. 첫 번째 단계에서는 약한 모델(weak model)이 강한 모델(strong model)로부터 도메인 지식을 증류하여 기초적인 전문성을 부여받습니다. 이어지는 단계에서는 강한 모델의 피드백을 통해 모델의 도메인 지식이 점진적으로 강화됩니다.

- **Performance Highlights**: 실험 결과, METEOR는 도메인-specific 과제에서 모델의 정확도(accuracy), 완전성(completeness), 관련성(relevance), 일관성(coherence) 및 신뢰성(reliability)을 크게 향상시키는 것으로 나타났습니다. 각 단계가 모델 성능에 긍정적인 영향을 미친다는 점이 입증되었습니다.



### Reviving Dormant Memories: Investigating Catastrophic Forgetting in Language Models through Rationale-Guidance Difficulty (https://arxiv.org/abs/2411.11932)
Comments:
          Working in progress

- **What's New**: 이 논문에서는 지속적 학습에서의 재해적 망각(catastrophic forgetting) 문제를 해결하기 위해, 모델이 제공된 부분적인 합리적 근거(rationale)를 수동으로 수용할 때 잊혀진 과제에 대한 성능이 회복될 수 있음을 발견했습니다. 또한, 원래 지침에 과제 비특이적 접두사(task-agnostic prefix)를 추가함으로써 모델이 능동적으로 적절한 근거를 생성하여 정답에 도달할 수 있음을 보여주는 실험 결과를 제시했습니다.

- **Technical Details**: 저자들은 'Rationale-Guidance Difficulty' 메트릭을 제안하여 주어진 지침이 모델에 적절한 근거를 생성하도록 얼마나 효과적으로 안내하는지를 평가합니다. 이 메트릭을 활용하여 재생 기반 지속적 학습 알고리즘에서 재생 데이터의 할당을 최적화하는 방식을 적용했습니다. 실험 결과는 이 데이터 할당 방식이 재해적 망각을 효과적으로 완화하고 다양한 모델에서 더 나은 플라스틱성을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: 다양한 크기의 모델에 대한 실험에서, 적절한 근거의 부분을 제공하면 잊혀진 과제에 대한 모델의 성능이 회복됨을 확인했습니다. 게다가, 비특정 접두사를 추가함으로써 모델이 관련 지식을 생성하는 데 도움이 되었고, 잊혀진 과제에서의 성능이 부분적으로 복구되었습니다. 이러한 결과는 모델이 과제 관련 지식의 실질적인 손실이 아니라 원래 지침이 적절한 근거를 생성하는 데 실패한 것에서 주로 기인함을 증명합니다.



### Dataset Distillers Are Good Label Denoisers In the Wild (https://arxiv.org/abs/2411.11924)
- **What's New**: 최근 노이즈가 포함된 데이터를 학습하는 방법이 딥러닝 모델을 실제 적용에 맞게 조정하는 데 필수적임을 보여주고 있습니다. 기존의 접근 방식은 대개 초기 노이즈를 평가한 후 노이즈가 포함된 샘플을 버리거나 재가중치를 적용하거나 재라벨링을 통해 문제를 해결했습니다. 그러나 이러한 방법은 초기 노이즈 평가가 부정확한 경우 성능이 저하되는 악순환에 빠질 수 있습니다.

- **Technical Details**: 본 연구에서는 데이터셋 증류(dataset distillation)를 활용하여 노이즈를 제거하는 새로운 접근 방식을 제안합니다. 주요 기법으로는 세 가지 대표적인 데이터셋 증류 방법(DATM, DANCE, RCIG)을 사용하며, 이를 대칭 노이즈, 비대칭 노이즈, 그리고 실제 자연 노이즈와 같은 다양한 노이즈 조건에서 엄격히 평가하였습니다. 이 방법은 기존 기술에서 공통적으로 발견되는 피드백 루프를 피하고, 오프라인 처리를 통해 강력한 개인정보 보호를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, 데이터셋 증류는 랜덤 노이즈 시나리오에서 효과적인 디노이징 도구로 작용하지만, 구조화된 비대칭 노이즈 패턴에서는 어려움을 겪을 수 있습니다. 또한 불균형 데이터 세트의 꼬리 클래스에서 발생하는 도전적인 샘플은 증류 과정에서 손실 압축 위험이 있음을 보여줍니다. 그럼에도 불구하고, 본 연구의 결과는 데이터셋 증류가 노이즈가 만연한 고 프라이버시 환경에서 견고한 모델 학습에 큰 가능성을 가진다는 것을 강조합니다.



### Artificial Intelligence Mangrove Monitoring System Based on Deep Learning and Sentinel-2 Satellite Data in the UAE (2017-2024) (https://arxiv.org/abs/2411.11918)
Comments:
          17 pages, 9 figures

- **What's New**: 이 연구는 UNet++ 딥러닝 모델과 Sentinel-2 다중 스펙트럼 데이터를 활용하여 아랍에미리트(UAE)의 맹그로브 동태를 모니터링합니다. 2017년부터 2024년까지의 기간 동안, UAE의 맹그로브 면적이 9,142.21헥타르로 늘어나고, 이산화탄소 격리가 약 194,383.42톤 증가했습니다. 이는 아부다비가 가장 큰 맹그로브 면적을 차지하며 맹그로브 성장에 중요한 역할을 하고 있다는 것을 보여줍니다.

- **Technical Details**: 연구에서 사용된 UNet++ 모델은 의료 이미지 분할에서 좋은 성능을 보였으며, 최근에는 원격 탐사 이미지 분석에 점차 적용되고 있습니다. Sentinel-2 다중 스펙트럼 위성 이미지는 2017년부터 2024년까지의 맹그로브 데이터를 수집하는 데 사용되며, 10미터 해상도의 분석이 가능합니다. 아랍에미리트의 특성에 맞춘 맹그로브 생태계 변화를 분석하기 위해, 유사한 지역의 데이터를 바탕으로 깊이 학습 기반의 워크플로우를 수립했습니다.

- **Performance Highlights**: 이 연구는 검증 세트에서 87.8%의 mIoU를 달성하며 맹그로브 동적 변화를 수준 있게 모니터링 할 수 있음을 보여줍니다. UAE의 맹그로브 지역은 아부다비를 중심으로 안정적이고 지속 가능한 방식으로 늘어나고 있으며, 이로써 각 에미리트가 맹그로브 복원에 기여하고 있음을 입증합니다. 본 연구는 효과적인 맹그로브 보호 및 관리 전략 수립을 위한 과학적 기초를 제공할 것입니다.



### ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling (https://arxiv.org/abs/2411.11911)
- **What's New**: 이번 연구에서는 다중 모드 비 예측(multi-modal motion prediction) 문제를 해결하기 위해 ModeSeq라는 새로운 예측 패러다임을 제안합니다. ModeSeq는 모드를 시퀀스(seqeunce)로 모델링하여 연속적으로 다음 모드를 추정하며, 이는 이전 모드 간의 상관관계를 보다 명확하게 캡처할 수 있게 해 줍니다. 또한 Early-Match-Take-All (EMTA) 훈련 전략을 도입하여 경로 다변량성을 더욱 향상시키는 효과를 거두었습니다.

- **Technical Details**: ModeSeq는 기존의 전통적인 다중 모드 예측 방식에서 벗어나, 모드를 한 번에 예측하는 대신 한 단계씩 순차적으로 예측합니다. 이 과정에서 모델은 각 예측 단계마다 이전 모드는 물론 그 신뢰도(confidence)까지 고려합니다. 이를 통해 각 모드 간의 관계를 명확히 파악할 수 있으며, 고속 모드 예측 및 후처리 단계를 필요로 하지 않고도 높은 품질의 경로 출력을 생성할 수 있습니다.

- **Performance Highlights**: ModeSeq는 Waymo Open Motion Dataset과 Argoverse 2 Motion Forecasting Dataset에서 여러 다른 다중 모드 예측 방법들과 비교하여 더 균형 잡힌 성능을 달성했습니다. 특히 모드 커버리지(mode coverage), 모드 스코어링(mode scoring), 그리고 경로 정확도(trajectory accuracy) 면에서 우수한 성능을 보였으며, 순차적인 모드 모델링 덕분에 높은 불확실성 환경에서도 다양한 행동 모드를 예측하는 능력을 자연스럽게 내재하고 있습니다.



### AIGS: Generating Science from AI-Powered Automated Falsification (https://arxiv.org/abs/2411.11910)
Comments:
          Pre-print. 35 pages. Official website: this https URL

- **What's New**: 본 논문에서는 $	extbf{AI-Generated Science}$ (AIGS)를 탐구합니다. 이는 에이전트가 독립적으로 연구 프로세스를 완전히 완료하고 과학 법칙을 발견하는 시스템을 의미합니다. 기존 시스템들이 검증 엔진에 크게 의존하는 한편, AIGS는 내재된 'falsification'을 통해 새로운 과학적 발견을 독립적으로 진행하는 방식으로 설계되었습니다.

- **Technical Details**: 우리는 Baby-AIGS라는 다중 에이전트 시스템을 제안하여 연구 프로세스의 핵심 역할을 담당하는 에이전트를 포함합니다. 이 시스템은 FalsificationAgent를 도입하여 가능성 있는 과학적 발견을 식별하고 검증함으로써 명시적인 'falsification'을 실현합니다. 이를 통해 독립적으로 연구를 수행할 수 있는 첫 걸음을 내딛습니다.

- **Performance Highlights**: 세 가지 작업에 대한 실험 결과, Baby-AIGS는 의미 있는 과학적 발견을 생성할 수 있음을 보여주었습니다. 그러나 경험이 풍부한 인간 연구자와 비교할 때 접근성은 여전히 낮은 상태입니다. 마지막으로, 현재 Baby-AIGS의 한계와 연구의 개선 가능성 및 관련 윤리적 문제에 대해 논의합니다.



### LoRA Unlearns More and Retains More (Student Abstract) (https://arxiv.org/abs/2411.11907)
Comments:
          AAAI-25 Student Abstract

- **What's New**: 이 논문은 개인 정보 보호 규제와 규정 준수의 증가로 인해 기계 학습 모델에서 특정 클래스와 관련된 정보를 제거하는 기계 Unlearning(MU)의 중요성을 강조합니다. 기존의 전통적인 방법은 모델을 데이터셋에 대해 재훈련해야 하므로 높은 계산 비용이 수반됩니다. 본 연구는 PruneLoRA라는 새로운 MU 패러다임을 제안하여 계산 비용을 줄이고 모델 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 모델 sparsification 기술을 통합하여 모델의 특정 가중치 및 뉴런을 선택적으로 제거하고, 그 후 LoRA(Low-Rank Adaptation)를 적용하여 파라미터의 일부만 수정합니다. 이를 통해 계산 비용과 메모리 요구 사항을 줄이면서 나머지 클래스에 대한 성능을 유지하는 새로운 방법론을 구축합니다. 실험에서는 ResNet50 및 Vision Transformer(ViT) 모델을 CIFAR-10 데이터셋에서 학습하였으며, L2 Pruning을 적용하여 모델의 절반을 제거했습니다.

- **Performance Highlights**: 실험 결과, 모든 방법이 목표 클래스 정보를 성공적으로 제거하는 우수한 Unlearning 정확도(UA)와 Membership Inference Attack(MIA) 효율성을 달성했습니다. PruneLoRA는 ResNet-50에서 Remaining Accuracy(RA)와 Testing Accuracy(TA)를 최고로 기록하였고, ViT 모델에서도 다른 방법들보다 상당한 성능 향상을 보여주었습니다. 이 결과는 PruneLoRA가 효과적인 Unlearning과 모델 성능 및 계산 효율성 사이의 균형을 제공함을 나타냅니다.



### ACING: Actor-Critic for Instruction Learning in Black-Box Large Language Models (https://arxiv.org/abs/2411.12736)
- **What's New**: 이 논문에서는 블랙박스 대형 언어 모델(LLM)에서 프롬프트 최적화를 자동화하기 위해 ACING이라는 접근 방식을 제안합니다. ACING은 연속 행동 강화 학습(continuous-action Reinforcement Learning) 문제로 프레이밍되며, 비미분 보상 신호에서 학습하여 프롬프트를 최적화합니다.

- **Technical Details**: ACING은 액터-비평자(actor-critic) 기반 방법을 활용하며, 이는 제한된 API 호출을 고려하여 효율적인 탐색과 활용을 동시에 수행합니다. 내부 파라미터를 알 수 없는 블랙박스 LLM의 경우에도 잘 작동하며, 기존의 방법과 비교하여 더 나은 성능을 발휘합니다.

- **Performance Highlights**: 30개 작업에 대한 ChatGPT의 프롬프트를 최적화한 결과, ACING은 기준 방법에 비해 일관되게 개선된 결과를 보여주었습니다. ACING은 평균 10%의 점수 향상을 이뤘으며, 인간 전문가의 지침을 초과하는 최대 39%의 개선을 달성했습니다.



### Testing classical properties from quantum data (https://arxiv.org/abs/2411.12730)
Comments:
          38 + 14 pages, 2 tables, 2 figures

- **What's New**: 이 논문은 Boolean 함수를 학습하는 것보다 훨씬 더 효율적으로 속성을 테스트할 수 있는 방법을 제시합니다. 특히, 고전적인 샘플에 제한된 테스트와는 달리 양자 데이터(quantum data)를 사용하는 테스트기법이 효과적임을 보여줍니다. 이 연구는 양자 알고리즘이 함수의 속성을 확인하는 데 있어 기존의 한계를 어떻게 극복할 수 있는지를 탐구합니다.

- **Technical Details**: 논문에서는 세 가지 잘 알려진 속성에 대한 테스트 알고리즘을 제시하며, 특히 모노토닉성(montonicity) 테스트를 위한 양자 알고리즘은 $	ilde{	ext{O}}(n^2)$개의 함수 상태 카피를 사용합니다. 이는 고전적으로 필요한 $2^{	ext{Ω}(	ext{sqrt}(n))}$ 샘플에 비해 큰 개선을 보여줍니다. 또한, 대칭(symmetry) 및 삼각형 프리(triangle-freeness)에 대한 테스트기법도 제안하며 각 경우에 대해 기존의 고전적 하한에 비해 유리한 성능을 드러냅니다.

- **Performance Highlights**: 이 알고리즘은 시간 효율성이 높으며, 이전의 테스트 문제에 대해 Fourier 샘플링 기법을 넘어서는 기술을 포함합니다. 논문에서는 모노토닉성 테스트에 대한 하한으로 $	ext{Ω}(1/	ext{ε})$를 제시하며, 양자 데이터의 테스트와 양자 쿼리(quantum queries) 간의 뚜렷한 차별성을 강조합니다. 마지막으로, $	ext{Ω}(1)$ 고전 쿼리로 해결할 수 있는 테스트 문제와 $	ext{Ω}(2^{n/2})$ 함수 상태 카피가 필요한 문제를 탐구하며, 양자 데이터와 고전적 쿼리 간의 명확한 구분을 제시합니다.



### LazyDINO: Fast, scalable, and efficiently amortized Bayesian inversion via structure-exploiting and surrogate-driven measure transpor (https://arxiv.org/abs/2411.12726)
- **What's New**: LazyDINO는 비용이 많이 드는 매개변수-관측치 맵(PtO maps)에 대한 고차원 비선형 베이esian 역문제의 빠르고 확장 가능한 해결책을 제공하는 변이 이론적 인퍼런스 방법입니다. 이 방법은 PtO 맵과 그 야코비안(Jacobian)의 샘플을 사용하여 유도 정보(neural surrogate)를 구성하는 오프라인 단계로 시작합니다.

- **Technical Details**: LazyDINO는 오프라인 및 온라인 단계로 구성되며, 오프라인 단계에서는 PtO 맵의 유도 정보를 기반으로 한 신경 근사체(neural surrogate)를 구성합니다. 온라인 단계에서는 관측 데이터를 받으면 Lazy map을 활용하여 빠른 사후 확률(posteriors) 근사를 꾀하게 됩니다. 이 방법은 유도 기반의 축소 기준 아키텍처를 최적화하여 사후 분포의 근사 오류를 최소화합니다.

- **Performance Highlights**: LazyDINO를 사용하여 베이esian 역문제를 처리할 때, 정확한 사후 근사를 빠르게 달성할 수 있으며, 오프라인 비용이 1~2 배 감소하는 것을 확인했습니다. 특히, 1000개 미만의 오프라인 샘플을 사용하여 라플라스 근사를 지속적으로 초월하는 성능을 보였으며, 다른 방법들은 16,000개의 오프라인 샘플에서는 실패하기도 했습니다.



### Rethinking MUSHRA: Addressing Modern Challenges in Text-to-Speech Evaluation (https://arxiv.org/abs/2411.12719)
Comments:
          19 pages, 12 Figures

- **What's New**: 이 연구는 TTS(Text-To-Speech) 모델의 평가를 위한 기존 MUSHRA 테스트의 문제점을 분석하고 있습니다. 특히, 기존의 MUSHRA 방법이 현대 TTS 시스템의 인간 음성 품질을 초과하는 경우 불공정한 점수를 부여한다는 점을 강조합니다. 연구팀은 두 가지 변형된 MUSHRA 테스트를 제안하여 평가의 공정성과 신뢰성을 높이고, 47,100개의 인간 평가 데이터셋인 MANGO를 공개하여 인도 언어의 TTS 시스템 평가를 지원합니다.

- **Technical Details**: 본 연구는 MUSHRA 테스트의 신뢰성, 민감도 및 유효성을 면밀히 분석합니다. 평가 과정에서 발생할 수 있는 요소들, 예를 들어 rater variance, listener fatigue, reference bias 등을 고려하여 주요 문제를 두 가지로 요약했습니다: (i) reference-matching bias와 (ii) judgement ambiguity. 각 테스트 버전은 탄력적인 평가를 가능하게 하고 사용자 경험을 세분화하여 평가의 명확성을 향상시킵니다.

- **Performance Highlights**: 새롭게 제안된 두 가지 MUSHRA 변형은 더 나은 평가 결과를 도출하며, 이는 현대 TTS 시스템이 불공정하게 평가받지 않도록 합니다. 실험 결과, 두 변형 모두 높은 신뢰성과 세밀한 평가를 제공하고, 특히 두 번째 변형은 평가 중 특정 오류 요소를 구체적으로 식별할 수 있게 합니다. 이 연구의 결과는 TTS 기술 발전에 기여할 수 있는 여러 가능성을 보여주며, MANGO 데이터셋은 향후 연구와 개선을 위한 중요한 자원이 될 것입니다.



### IoT-Based 3D Pose Estimation and Motion Optimization for Athletes: Application of C3D and OpenPos (https://arxiv.org/abs/2411.12676)
Comments:
          17 pages

- **What's New**: 이 연구에서는 고정밀 3D 포즈 추정 및 운동 최적화를 위한 IoT-Enhanced Pose Optimization Network (IE-PONet)를 제안합니다. IE-PONet는 C3D를 활용한 시공간(feature extraction), OpenPose를 이용한 실시간 키포인트 검출, Bayesian optimization을 통한 하이퍼파라미터 조정을 통합하여 성능을 극대화합니다. NTURGB+D 및 FineGYM 데이터셋에서 90.5 및 91.0의 AP^p50 점수와 74.3 및 74.0의 mAP 점수가 나타나며, 각 모듈의 필수적인 역할을 입증한 ablation 연구도 포함되어 있습니다.

- **Technical Details**: IE-PONet 모델은 복잡한 운동 환경에서 실시간으로 데이터 수집 및 전송이 가능하도록 IoT 센서를 통합하여, 기존 기술의 단점을 보완합니다. C3D 및 OpenPose의 장점을 결합하여 운동 분석과 최적화를 정확하게 수행하며, Bayesian optimization을 통해 모델의 적응성과 효율성을 더욱 향상시킵니다. 또한, 이 시스템은 빠른 데이터 처리와 실시간 피드백이 가능한 성능을 보여줍니다.

- **Performance Highlights**: IE-PONet는 다양한 데이터셋에서 우수한 성능과 강 robustness를 입증하였습니다. 실험 결과, NTURGB+D 및 FineGYM 데이터셋에서의 성능이 특히 두드러져, 이는 훈련 과정에서의 즉각적인 피드백과 운동 조정 개선을 가능하게 합니다. 향후 연구는 모델 최적화와 다중 데이터 통합, 실시간 피드백 메커니즘 개발에 초점을 맞출 계획입니다.



### PoM: Efficient Image and Video Generation with the Polynomial Mixer (https://arxiv.org/abs/2411.12663)
- **What's New**: 최근 Diffusion 모델에서 Multi-Head Attention (MHA)을 대체할 새로운 메커니즘인 Polynomial Mixer (PoM)를 소개합니다. PoM은 시퀀스의 길이에 대해 선형 복잡성을 가지며, MHA의 품질을 손상시키지 않고도 전체 시퀀스를 명시적인 상태로 인코딩합니다. 이 메커니즘은 대규모 생성 모델의 효율적인 스케일링을 가능하게 하며, 특히 고해상도 이미지와 비디오 생성을 더 용이하게 만들어 줍니다.

- **Technical Details**: PoM은 State-Space Models (SSMs)와 유사한 선형 복잡성을 고려하지만, MHA와 같이 모든 쌍 정보에 대한 처리를 가능하게 합니다. 이를 통해 DiT 아키텍처에 적용 시 더 높은 해상도에서 MHA를 사용하는 모델을 학습하는 것보다 PoM을 사용하는 모델이 훈련 비용이 월등히 낮아질 수 있음을 보여줍니다. 또한 PoM은 일반적인 시퀀스-투-시퀀스 (sequence-to-sequence) 근사기를 제공함으로써 다양한 애플리케이션에서의 적용 가능성을 높입니다.

- **Performance Highlights**: PoM을 적용한 이미지 생성 모델은 유사한 품질의 샘플을 생성하면서도 더 적은 계산 자원으로 고해상도의 이미지를 처리할 수 있는 능력을 보여줍니다. 더불어 PoM을 활용한 비디오 생성 모델은 매 프레임에 대해 일정한 처리 비용을 유지하면서도 시각적 품질을 유지할 수 있습니다. 이로 인해 PoM은 향후 고해상도 이미지 및 긴 비디오 생성을 위한 기본적인 메커니즘으로 자리 잡을 것으로 기대됩니다.



### Leadsee-Precip: A Deep Learning Diagnostic Model for Precipitation (https://arxiv.org/abs/2411.12640)
- **What's New**: 최근 딥러닝 기법을 활용한 기상 예측 모델들이 기존의 수치 모델들을 초월하며 기상 변수의 정확성을 높이고 있습니다. 이 연구에서는 강수량 예측의 정확성을 향상시키기 위해 Leadsee-Precip이라는 새로운 글로벌 딥러닝 모델을 제안하고 있습니다. 이 모델은 기상 순환 데이터(circulation fields)를 기반으로 하여 강수량을 생성하며, 정보 균형(Information Balance) 기법을 통해 강수량의 긴 꼬리 분포 문제를 해결합니다. Leadsee-Precip는 기존 인공지능 기상 모델에 비해 관측 결과와의 일관성이 높고, 글로벌 수치 기상 예측 모델과 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: Leadsee-Precip 모델은 인코더-디코더 구조를 기반으로 하며, 기상 순환 변수를 통한 특징 추출, 히든 변환기, 강수량 업샘플링의 세 부분으로 구성됩니다. 이 모델은 MogaNet이라는 최첨단 컴퓨터 비전 백본(backbone)을 사용하여 우주와 지면 변수의 특징을 각각 추출합니다. 또한, 정보 균형 기법을 도입하여 희귀한 극단적인 강수량에 대한 민감도를 높이며, 위성 기반 데이터를 이용하여 강수량 진단을 효과적으로 수행합니다.

- **Performance Highlights**: Leadsee-Precip는 6시간 동안의 강수량이 25mm를 초과하는 경우에 대해 Threat Score (TS) 0.185와 Fraction Skill Score (FSS) 0.570을 달성하였습니다. 중국의 기상 관측소에서 검증했을 때도, ERA5 순환 변수와 결합된 진단 강수량은 강수 이벤트 발생 시 훌륭한 성능을 보였습니다. 그러나 순환 필드의 예측과 실제 데이터 간의 차이가 존재할 경우 강수 예측 성능이 감소할 수 있으며, 이는 예측된 순환 필드에 대한 추가적인 세부 조정을 통해 개선될 수 있습니다.



### Instant Policy: In-Context Imitation Learning via Graph Diffusion (https://arxiv.org/abs/2411.12633)
Comments:
          Code and videos are available on our project webpage at this https URL

- **What's New**: 본 논문에서 우리는 In-Context Imitation Learning (ICIL)을 활용한 새로운 방법론인 Instant Policy를 소개합니다. Instant Policy는 단 1~2개의 데모로 새로운 작업을 신속하게 학습할 수 있도록 하며, 기존의 Behavioral Cloning (BC) 방법에 비해 시간 효율적입니다. 이 접근법은 그래프 생성 문제로 모델링하여 데모와 관찰을 구조적으로 해석함으로써 로봇 행동 예측의 효율성을 높입니다.

- **Technical Details**: Instant Policy는 그래프 기반 표현을 활용하여 데모, 현재 포인트 클라우드 관찰, 로봇의 행동을 통합한 구조를 형성합니다. ICIL을 확산 기반 그래프 생성 문제로 공식화하여, 복잡한 데이터의 구조적 학습을 가능하게 합니다. 또한, 절차적으로 생성된 의사 데모(pseudo-demonstration)를 통해 무한한 학습 데이터를 생성할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험에서 Instant Policy는 다양한 일상 작업을 신속하게 배울 수 있는 능력을 보여 주었습니다. 기존의 기준 모델들보다 높은 작업 성공률을 달성하며, 테스트 시 제공되지 않은 객체의 기하학에 대한 일반화 능력 또한 관찰되었습니다. Instant Policy는 또한 인간 손의 데모에서 로봇 정책으로의 크로스 엠바디먼트 전이와 언어 정의 작업에 대한 제로샷 전이를 가능하게 합니다.



### A Multimodal Approach Combining Structural and Cross-domain Textual Guidance for Weakly Supervised OCT Segmentation (https://arxiv.org/abs/2411.12615)
Comments:
          21 pages, 9 figures, 8 tables

- **What's New**: 이 논문은 Optical Coherence Tomography (OCT) 이미지의 정확한 세분화를 위한 새로운 약한 감독 의미 세분화(Weakly Supervised Semantic Segmentation, WSSS) 접근 방식을 제안합니다. 이 방법은 구조적 안내와 텍스트 기반 전략을 통합하여 고품질의 의사 라벨을 생성하는 데 초점을 맞춰 세분화 성능을 크게 향상시킵니다. 특히, 이 연구는 OCT 이미지에서 병변을 식별하는 데 있어 이미지 수준의 감독만을 사용하며, 효율성을 극대화하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 시각적 정보와 텍스트 정보를 융합한 다중 모드(multi-modal) 프레임워크를 기반으로 합니다. 이 방법은 두 개의 처리 모듈을 사용해 원본 이미지 특징과 OCT 이미지의 구조적 특징을 교환하며, 이를 통해 모델이 병변이 발생할 가능성이 있는 위치를 식별하도록 유도합니다. 또한, 대규모 사전 훈련된 모델을 활용하여 레이블 기반 텍스트 정보와 합성 설명을 통합하며, 이를 통해 모델의 성능을 더욱 향상시키고 있습니다.

- **Performance Highlights**: 세 가지 OCT 데이터 세트에 대한 실험 결과, 제안된 방법이 최신 WSSS 연구 방법들과 비교해 우수한 성능을 보이며, 진단 정확성과 의료 영상의 효율성을 개선할 수 있는 가능성을 강조합니다. 이 연구는 병변을 지역화하는 데 있어 텍스트와 구조적 정보를 효과적으로 통합하여 세분화의 정확도를 높이는 데 기여하였습니다.



### Reward driven workflows for unsupervised explainable analysis of phases and ferroic variants from atomically resolved imaging data (https://arxiv.org/abs/2411.12612)
Comments:
          19 pages, 6 figures

- **What's New**: 이번 연구는 빠르게 발전하고 있는 비구면 보정 전자 현미경(aberation corrected electron microscopy) 기술을 통해 소재 구조 분석을 위한 새로운 접근 방식을 제안합니다. 특히, 하이퍼파라미터(hyperparameter) 선택의 민감성 문제를 해결하고, 보상 기반 접근법(reward-driven approach)을 사용하여 분석 워크플로우를 최적화하는 방법을 탐구합니다. 새로운 방법론은 무감독 기계학습(unsupervised ML) 기술의 성능을 향상시키는 데 기여하고, 소재의 물리적 행동과 잘 맞는 로컬 디스크립터(local descriptors)를 발견하는 데 초점을 맞춥니다.

- **Technical Details**: 연구에서는 Sm이 도핑된 BiFeO3(BFO) 박막의 극성과 격자 변형(lattice distortion)을 발견하는 사례를 통해 하이퍼파라미터와 디스크립터(descriptor)가 무감독 기계학습 방법의 능력에 미치는 영향을 분석하였습니다. 보상 시스템은 도메인 벽(domain wall) 연속성 및 직선성을 반영하도록 설계되어, 분석이 소재의 물리적 행동과 일치하도록 돕습니다. 이러한 최적화 접근법은 변이형 오토인코더(variational autoencoder)의 구조적 변동 요인(disentangle structural factors of variation)을 구분하는 데까지 확장됩니다.

- **Performance Highlights**: 이번 연구의 중요한 성과는 잘 정의된 보상을 통해 워크플로우의 성공을 수량적으로 측정할 수 있는 방법을 탐색한 것입니다. 이러한 방법론은 소재 물리의 근본적인 통찰력을 제공하며, 무감독 기계학습 방법의 성능을 극대화하는 방향으로 향후 연구에 중요한 기초 자료가 될 것입니다. 최종적으로, 이 연구는 전자 현미경 이미지를 통한 구조적 정보 분석의 새로운 가능성을 제시하고 있습니다.



### STREAM: A Universal State-Space Model for Sparse Geometric Data (https://arxiv.org/abs/2411.12603)
- **What's New**: 이 논문에서는 기하학적 구조를 상태 공간 모델(SSM)의 매개변수화에 명시적으로 인코딩하는 방안을 제안합니다. 이를 통해 기존의 sequence 모델들과 비교하여 불규칙한 단계 크기를 가진 희소 기하학 데이터 처리에 효율성을 제공합니다. 이로 인해 새로운 모델인 STREAM은 이벤트 기반 비전 및 포인트 클라우드 분석에서 경쟁력 있는 성능을 발휘합니다.

- **Technical Details**: STREAM 모델은 CUDA 커널의 수정된 버전을 사용하여 현대 하드웨어에 효율적으로 희소 기하학 데이터를 매핑합니다. 본 연구에서는 상대 좌표의 차이를 단계 크기로 주입하여 기하학적 연산을 O(N) 단계로 수행하도록 설계되었습니다. 이는 모든 N포인트 간의 상호작용을 계산하는 데 필요한 단계를, 기존 방식보다 획기적으로 줄여 줍니다.

- **Performance Highlights**: STREAM 모델은 ModelNet40 및 ScanObjectNN 데이터셋에서 PointMamba 기준보다 최대 2.84% 향상된 성능을 보여주었습니다. 더불어 DVS128 Gesture 데이터셋에서 모든 11개 클래스에 대해 100%의 테스트 정확도를 달성하였습니다. 이는 이벤트 기반 비전 분야에서 최초의 결과로, 희소 기하학 데이터 처리에서의 강력한 유도 편향(inductive bias)을 입증합니다.



### Hypergraph $p$-Laplacian equations for data interpolation and semi-supervised learning (https://arxiv.org/abs/2411.12601)
Comments:
          16 pages

- **What's New**: 하이퍼그래프(hypergraph) 학습에서 p-Laplacian 정규화는 데이터 내의 고차 관계 모델링에 유연성을 제공하여 많은 주목을 받고 있습니다. 본 논문에서는 비미분성(non-differentiability)과 미니마이저(minimizer)의 비유일성(non-uniqueness)으로 인해 어려운 빠른 수치적 구현에 초점을 맞추고 있습니다. 이를 위해 p-Laplacian 정규화의 부분미분(subdifferential)에서 하이퍼그래프 p-Laplacian 방정식을 도출하고, 수학적으로 적합하고 효율적인 대체 방정식을 제안합니다.

- **Technical Details**: 하이퍼그래프 H=(V,E,W)에서 V는 정점 집합, E는 하이퍼엣지 집합, W는 하이퍼엣지에 대한 가중치를 나타냅니다. 하이퍼그래프 내의 정점은 동일한 하이퍼엣지에 속할 경우 동일한 레이블을 가질 경향이 있으며, 이 논문은 레이블이 주어진 정점 집합을 기반으로 나머지 정점의 레이블을 할당하는 문제를 다룹니다. 이러한 문제 해결을 위해 p-Laplacian 정규화가 사용되며, 기울기가 정의되지 않는 다양한 상황에서도 작동할 수 있는 해법을 제공합니다.

- **Performance Highlights**: 제안된 단순화된 p-Laplacian 방정식은 데이터 보간(interpolation)에서 스파이키(spiky) 솔루션을 억제하고 반지도 학습(semi-supervised learning)에서 분류 정확도를 향상시킵니다. 수치 실험을 통해 이러한 접근 방식의 효율성을 입증하였으며, 낮은 계산 비용은 더 다양한 응용을 가능하게 합니다. 이러한 특성 덕분에 하이퍼그래프를 활용한 데이터 처리의 가능성이 더욱 넓어질 것입니다.



### GNNAS-Dock: Budget Aware Algorithm Selection with Graph Neural Networks for Molecular Docking (https://arxiv.org/abs/2411.12597)
- **What's New**: 이번 논문에서는 GNNAS-Dock이라는 새로운 Graph Neural Network (GNN) 기반의 자동 알고리즘 선택 시스템이 소개됩니다. 이 시스템은 blind docking 상황에서 리간드와 단백질의 복잡한 구조 데이터를 처리하며, 다양한 docking 알고리즘의 성능을 예측할 수 있습니다. GNNAS-Dock은 두 가지 주요 목표를 가지고 있습니다: 각 후보 docking 알고리즘의 성능을 예측하고, 각 docking 사례에 대해 가장 효율적인 알고리즘을 선택하여 시간을 줄이는 것입니다.

- **Technical Details**: GNNAS-Dock은 리간드와 단백질의 구조적 특성을 활용하여 다양한 docking 조건에서 최적의 성능을 발휘하는 docking 알고리즘을 예측합니다. GNN은 그래프 데이터에서 작동하며, 이는 분자 및 단백질의 구조적 표현과 자연스럽게 일치합니다. 연구에서는 두 가지 GNN 기반 알고리즘 선택 모델이 개발되며, 하나는 정확도를 높이는 방법을 선택하고 다른 하나는 효율성을 최적화하는 데 중점을 둡니다.

- **Performance Highlights**: GNNAS-Dock의 정확도 모델은 평균 RMSD 값 1.74 Å를 기록하여, 현재 테스트된 모든 docking 방법 중에서 가장 뛰어난 성과를 보였습니다. 효율성 모델은 RMSD 2.75 Å의 평균 값을 기록하며, 약 79.73%의 RMSD 값이 2Å 이하로 나타났습니다. 이러한 성능 향상은 GNNAS-Dock이 특정 성능 기준을 겨냥하여 구축된 AS 모델을 통해 가능하다는 것을 시사합니다.



### Debias your Large Multi-Modal Model at Test-Time with Non-Contrastive Visual Attribute Steering (https://arxiv.org/abs/2411.12590)
Comments:
          10 pages, 3 Figures, 3 Tables. arXiv admin note: text overlap with arXiv:2410.13976

- **What's New**: 이 논문에서는 Large Multi-Modal Models (LMMs)의 사회적 편향을 직접 제거할 수 있는 새로운 디바이징 프레임워크를 제안합니다. 기존의 방법들과는 달리, 우리의 방법은 단일 이미지와 대상 속성을 이용하여 훈련 없이 디바이징을 수행할 수 있습니다. 이를 통해 LMMs의 출력에서 보호 속성과 관련된 텍스트 생성을 최소화하고, 감정 개선 효과를 낼 수 있음을 보여냈습니다.

- **Technical Details**: 우리는 훈련이 필요 없는 방법으로서 LMM의 입력 이미지에서 단 한번의 gradient descent를 수행하여 편향된 표현을 제거하는 방법을 사용합니다. 이 과정에서 Fast Gradient Sign Method (FGSM)를 적용하여, 이미지를 통해 얻은 정보를 활용하여 방향성을 설정합니다. 이 접근법은 LLaVA와 Llama3와 같은 두 가지 주요 LMM 모델에서 효과적으로 보호 속성과 관련된 텍스트 생성을 저감시키는 데 성공했습니다.

- **Performance Highlights**: 실험 결과, 디바이징된 LMM의 출력 정확도는 기존의 편향 모델과 유사한 수준을 유지하며 발생하며, 언어 모델링 기능에서도 성능 저하가 없음을 확인했습니다. 또한, 디바이징을 통해 생성된 텍스트에서 감정의 형평성을 확보할 수 있음을 확인했습니다. 이러한 결과들은 LMM의 편향 제거가 모델 성능을 희생하지 않고도 가능하다는 것을 나타냅니다.



### ULTra: Unveiling Latent Token Interpretability in Transformer Based Understanding (https://arxiv.org/abs/2411.12589)
- **What's New**: 이 논문에서는 Transformer의 Latent Token (잠재 토큰) 표현을 해석하는 새로운 프레임워크를 제안합니다. 이 프레임워크를 통해 기존 모델을 추가적인 파인튜닝 없이도 제로샷(Zero-shot) 비지도형 의미 세분화(semantic segmentation)가 가능함을 입증했습니다. 이 방법은 Transformer 모델이 입력의 의미를 이해하는 본질적인 능력을 활용하며 기존의 전통적인 세분화 모델들을 능가하는 성능을 보여줍니다.

- **Technical Details**: Transformer 아키텍처와 Vision Transformers (ViTs)를 기반으로 한 이 연구는, Latent Tokens이 각각의 의미적 개념을 나타내도록 해석하는 방법을 제시합니다. 제안된 프레임워크는 친숙한 메커니즘 없이도 이미지 인지를 가능하게 하며, 이를 통해 하이퍼파라미터 조정 없이 이미지 세분화가 이루어질 수 있습니다. 우리는 또한 이 프레임워크가 대규모 언어 모델(LLMs)에서도 효과적임을 확인하여 텍스트 요약 작업에서의 적용 가능성을 검증했습니다.

- **Performance Highlights**: COCO-Stuff 데이터셋에서 67.2%의 정확도와 32.9%의 평균 교차 IoU(mIoU)를 기록하며, PASCAL VOC 데이터셋에서는 51.9%의 mIoU를 달성하여 기존 SOTA(State-of-the-Art) 성능을 초월했습니다. 이 연구는 기존의 비지도 학습 세분화 방법들보다 우수한 성능을 보이며, 많은 imstances가 필요한 기존 방법보다 더 효율적으로 동작합니다.



### Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models (https://arxiv.org/abs/2411.12580)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 일반화 전략을 심층적으로 연구하였습니다. 특히, LLM이 추론 과제를 수행할 때 의존하는 사전 학습 데이터의 종류를 조사하고, 이러한 데이터가 모델 출력에 미치는 영향을 분석하였습니다. 7B 및 35B 모델을 사용하여, 단순한 수학적 추론 작업과 사실적 질문에 대한 답변의 경우 문서의 영향력을 대조하였습니다.

- **Technical Details**: 연구에서는 대규모 변환기(Transformers)에 적용된 강건 통계 기법을 도입하여 사전 학습 문서가 문제 해결에 미치는 영향을 계산하였습니다. 세 가지 수학적 작업에 대한 추론 질문에 대해서는 프로세스 중심의 지식이 문서에서 중요한 영향을 미치는 것으로 밝혀졌으며, 사실적 질문에 비해 추론 질문에서는 개별 문서의 영향력이 상대적으로 약하다는 점이 확인되었습니다. 이는 모델이 비슷한 작업에 대해 절차적 지식을 학습하는 방식임을 나타냅니다.

- **Performance Highlights**: 연구 결과, 사실적 질문의 경우 답변이 중요한 문서에 자주 나타나는 반면, 추론 질문에는 거의 나타나지 않는 것으로 나타났습니다. 또한, 수학적 추론에 있어서 코드 데이터의 중요성이 강조되었으며, 이는 모델이 다양한 과제를 학습하는 데 있어 프로세스 중심의 고품질 데이터가 더 효과적일 수 있음을 시사합니다. 향후 사전 학습 전략에 대한 통찰력을 제공할 수 있는 이러한 발견은 학습의 일반화 범위를 명확히 하는 데 중요한 역할을 할 것입니다.



### A data driven approach to classify descriptors based on their efficiency in translating noisy trajectories into physically-relevant information (https://arxiv.org/abs/2411.12570)
Comments:
          19 pages, 5 figures + 3 in supporting information (at the bottom of the manuscript)

- **What's New**: 이번 연구에서는 복잡한 물리적 구조를 갖는 다체 동역학 시스템의 궤적에서 유용한 정보를 추출하는 다양한 설명자의 효율을 비교하는 데이터 기반 접근법을 제시합니다. 특히 아이스와 물이 공존하는 시스템을 분석하여, 전통적인 설명자들과 더불어 Smooth Overlap of Atomic Positions (SOAP) 및 Local Environments and Neighbors Shuffling (LENS) 같은 고급 설명자들이 더 높은 정보 전달율을 갖는다는 사실을 밝혔습니다. 본 연구는 시스템의 내부 복잡성을 이해하는 데 있어 노이즈 제거가 중요한 역할을 한다는 점을 강조합니다.

- **Technical Details**: 복잡한 분자 시스템을 연구하기 위해서는 일반적으로 원시 데이터에서 특정 설명자를 선택하여 시간 시계열로 변환하는 작업이 필요합니다. 다양한 설명자들, 특히 동적인 설명자 LENS와 시간의 변화에 따라 속성을 추적하는 TimeSOAP을 포함하여, 정상적인 환경 변수와 비정상적인 환경 변수에 대해 비교 분석합니다. 연구에서는 Onion Clustering이라는 비지도 학습 기법을 통해 각 설명자에서 추출 가능한 최대 정보를 평가하고, 고차원 메트릭을 기반으로 효율성을 순위 매겼습니다.

- **Performance Highlights**: 그 결과, SOAP와 LENS와 같은 고급 설명자는 전통적인 설명자보다 신호 대 잡음 비율이 높아 우수한 성능을 보였습니다. 그러나 간단한 설명자 또한 로컬 신호 노이즈 제거 후에는 충분한 성능을 나타낼 수 있음을 관찰했습니다. 예를 들어, 초기에는 효과가 떨어지는 것으로 보였던 $d_5$는 노이즈 제거 이후 시스템의 비지역적 동적 복잡성을 해결하는 데 가장 효과적인 설명자로 변모하였습니다.



### Stream-Based Active Learning for Process Monitoring (https://arxiv.org/abs/2411.12563)
- **What's New**: 본 논문에서는 부분적으로 숨겨진 마르코프 모델(Partially Hidden Markov Model)을 활용하여 데이터 스트림을 처리하는 새로운 스트림 기반 능동 학습(active learning) 전략을 제안합니다. 이 방법은 제한된 예산 내에서 라벨링 자원을 최적화하고 동적으로 가능한 이상 상태(out-of-control states)를 업데이트할 수 있도록 설계되었습니다. 이는 품질 관리(Quality Management)에서 공정의 안정성을 모니터링하는 데 중요한 혁신을 제공합니다.

- **Technical Details**: 기존의 통계적 공정 모니터링(Statistical Process Monitoring) 방법은 주로 비지도 학습 기반이며, 진정한 이상 상태를 알기 어려운 경우가 많습니다. 본 연구는 및 공정 데이터의 라벨이 있는 경우 이를 활용하여 감독하는 방법을 연구합니다. 모델 성능을 평가하기 위해 저항 점 용접(resistance spot welding) 공정에서의 시뮬레이션 및 사례 연구가 포함되어 있습니다.

- **Performance Highlights**: 제안된 방법은 현실 세계의 데이터 스트림을 다루면서 IC와 OC 상태를 효과적으로 분류하는 데 있어 향상된 성능을 보여줍니다. 특히, 높은 품질의 공정에서 OC 상태가 드물기 때문에 클래스 불균형(class imbalance) 문제를 해결하는 데 기여할 수 있습니다. 이 연구는 아시아 자동차 산업의 지속 가능성과 품질 관리에 있어 중요한 통찰력을 제공합니다.



### S3TU-Net: Structured Convolution and Superpixel Transformer for Lung Nodule Segmentation (https://arxiv.org/abs/2411.12547)
- **What's New**: 이 연구에서는 폐선암 결절의 CT 이미지에서 정확한 분할을 위한 새로운 모델인 S3TU-Net을 제안합니다. S3TU-Net은 다차원 공간 커넥터와 슈퍼픽셀 기반의 비주얼 트랜스포머를 통합하여 우수한 분할 성능을 달성합니다. 이 모델은 멀티 뷰 CNN-Transformer 하이브리드 아키텍처로 구축되어, 구조화된 합성곱 블록(DWF-Conv/D2BR-Conv)을 활용하여 다중 스케일 지역 특성을 추출하며 과적합을 줄입니다.

- **Technical Details**: S3TU-Net의 아키텍처는 U자형 인코더-디코더 구조로, DWF-Conv 및 D2BR-Conv 블록과 함께 잔여 연결, 그리고 다중 방향 공간 이동 기술을 가진 S2-MLP Link 모듈을 포함합니다. DWF-Conv 블록은 두 개의 합성곱 레이어와 함께 작동하여 깊이 있는 특성을 집중적으로 활용하며, S2-MLP Link는 다양한 의미 수준의 특성을 융합하여 성능을 향상시킵니다. 또한, RM-SViT 모듈은 글로벌 및 로컬 특성을 결합하여 장기 종속성을 효과적으로 포착합니다.

- **Performance Highlights**: LIDC-IDRI 데이터셋에서 S3TU-Net은 DSC 89.04%, 정밀도 90.73%, mIoU 90.70%, 감도 93.70%를 기록했습니다. EPDB 개인 데이터셋에서의 검증 결과도 DSC 86.40%를 달성하여 모델의 안정성과 일반화 능력을 보여줍니다. S3TU-Net은 최근의 방법들에 비해 DSC를 4.52% 향상시켰고, 감도는 3.16% 증가하여 다양한 성능 지표에서 약 2%의 개선을 나타냈습니다.



### MAViS: Modular Autonomous Virtualization System for Two-Dimensional Semiconductor Quantum Dot Arrays (https://arxiv.org/abs/2411.12516)
Comments:
          14 pages, 5 figures, 8 pages of supplemental material

- **What's New**: 본 논문에서는 Modular Automated Virtualization System (MAViS)를 소개합니다. MAViS는 다층 가상 게이트를 실시간으로 자율적으로 구성할 수 있는 일반적이고 모듈화된 프레임워크입니다. 최신 기계 학습 기법을 활용하여 2차원 전하 안정성 다이어그램에서 빠르게 특징을 추출할 수 있습니다. 이 방법은 높은 고유도 스핀 큐비트 시스템의 제어에 있어 중요한 획기적인 진전을 의미합니다.

- **Technical Details**: MAViS는 다층 가상 게이트의 전하 안정성 다이어그램을 분석하여 상호 커플링과 전자포텐셜의 제어를 가능하게 합니다. 특히, 가상 게이트는 다양한 물리적 게이트의 선형 조합으로 정의되어 이러한 고유상태를 관리합니다. 기존 방법론에 비해, MAViS는 기계 학습(Machine Learning)과 클래식한 분석 기법을 통합하여 더 넓은 범위의 제어를 가능하게 합니다. 이를 통해 직접적인 전하 센서 측정 없이도 정밀한 조정이 가능합니다.

- **Performance Highlights**: MAViS 시스템은 십 개의 양자점으로 구성된 밀집 2차원 배열에서의 완전 가상화를 성공적으로 보여주었습니다. 기존의 수동 조정에서 완전 자동화된 제어가 가능하다는 점에서, 성능의 향상은 대규모 반도체 양자점 시스템의 제어에 있어 효율적인 솔루션을 제공합니다. 실험을 통해 MVaiS의 정확한 작동을 입증하였으며, 이는 향후 대형 양자 컴퓨팅 시스템의 개발에 기여할 것으로 기대됩니다.



### AI Flow at the Network Edg (https://arxiv.org/abs/2411.12469)
- **What's New**: 최근 대규모 언어 모델(LLMs)과 그 다중 모드 변형의 발전은 인공지능(AI) 분야에서 많은 혁신을 가져왔습니다. 본 논문에서는 AI Flow라는 프레임워크를 제안하여, 다양한 계산 리소스를 활용하여 네트워크 엣지에서 지능을 유통하는 방법을 모색합니다. 이러한 접근은 클라우드에서 엣지 환경으로 대규모 모델을 배포하는 데 있어 발생하는 여러 문제들을 해결하려는 시도를 반영하고 있습니다.

- **Technical Details**: AI Flow는 디바이스, 엣지 노드, 클라우드 서버 간의 이질적인 리소스를 활용하여 추론 과정을 간소화합니다. 이 프레임워크는 정보 흐름에서 지능 흐름으로의 전환을 추구하며, 데이터 전송 대신 엣지에서 감지된 정보 중 핵심적 특성만을 추출해 통신 비용을 줄이고 효율적인 서비스를 제공합니다. 시스템 아키텍처는 엣지 디바이스, 엣지 서버, 클라우드 서버의 세 가지 모듈로 구성됩니다.

- **Performance Highlights**: AI Flow 프레임워크는 다양한 사례를 통해 응답 지연을 줄이고 고품질의 결과를 유지하는 능력을 보여줍니다. 특히 이미지 캡셔닝(use case) 분야에서의 실험 결과는 엣지에서의 저지연 데이터 처리와 개선된 사용자 경험을 증명합니다. 이 기술의 적용은 로봇 제어, 스마트 홈, 증강 현실 및 자율 주행 등 다양한 분야에서 폭넓은 가능성을 펼칩니다.



### Dimension Reduction via Sum-of-Squares and Improved Clustering Algorithms for Non-Spherical Mixtures (https://arxiv.org/abs/2411.12438)
Comments:
          64 pages

- **What's New**: 본 연구에서는 비구형(Non-Spherical) Gaussian 혼합 모델을 클러스터링하기 위한 새로운 접근법을 개발했습니다. 이 방법은 입력 데이터를 저차원으로 분리 보존하는 프로젝션(projection)으로 변환하는 sum-of-squares 방법을 기반으로 합니다. 이 알고리즘은 고전적인 차원 축소(dimension reduction) 방법 중 하나인 특이값 분해(singular value decomposition)를 활용하여, 유명한 구형 클러스터링 알고리즘의 비구형 유사체를 형성합니다.

- **Technical Details**: 이 방법을 적용하면, 첫 번째로 $n \geq \operatorname{poly}(d) f(w_{\min}^{-1})$ 샘플과 $\operatorname{poly}(n)$ 시간 복잡도로 $k$ 개 중심(Gaussian) 클러스터를 형성할 수 있으며, 두 번째로 $n \geq d^{O(\log w_{\min}^{-1})} f(w_{\min}^{-1})$ 샘플과 $n^{O(\log w_{\min}^{-1})}$ 시간 복잡도로 고유의 공분산(covariance)을 가지는 $k$ 개 Gaussian 클러스터를 클러스터링할 수 있습니다. 여기서 $w_{\min}$은 입력 혼합물의 최소 혼합 가중치를 나타냅니다.

- **Performance Highlights**: 이 알고리즘은 차원에 독립적인 임의의 이상치(outlier) 비율을 허용하는 확장성을 가지고 있습니다. 이전의 최신 비구형 클러스터링 알고리즘은 이러한 혼합물을 클러스터링하는 데 $d^{O(k)} f(w_{\min}^{-1})$ 시간과 샘플을 필요로 했습니다. 본 연구 결과는 비구형 Gaussian 혼합에 대한 클러스터링의 통계적 질의 하한을 극복할 수 있는 가능성을 보여줍니다.



### STRisk: A Socio-Technical Approach to Assess Hacking Breaches Risk (https://arxiv.org/abs/2411.12435)
- **What's New**: 이번 연구에서는 STRisk라는 예측 시스템을 제안하여 소셜 미디어 차원을 포함한 데이터 유출 예측의 중요성을 강조합니다. 기존의 연구들은 주로 기술적 관점에서 데이터 유출을 다뤘지만, 본 연구는 3800개 이상의 미국 조직을 분석하여 사회적 요소와 기술적 지표를 결합하여 예측 모델을 개발합니다. 예측 대상은 해킹이 원인인 데이터 유출 사건으로, 기계 학습 기법을 통해 98% 이상의 AUC 점수를 기록했습니다.

- **Technical Details**: 이 연구에서는 기술적 이상과 소셜 신호를 포함하는 다양한 지표를 활용하여 각 조직의 사회 기술 프로필을 구축합니다. 기술적 이상에는 블랙리스트에 등록된 호스트, 개방된 포트, 만료된 웹 인증서 등이 포함되며, 소셜 신호는 트위터의 감정, 인기, 확산성 등을 포함합니다. 또한, 부정확한 라벨을 교정하기 위해 소음 수정 방법을 제안하고, 트리 기반 및 선형 기반 기계 학습 모델을 훈련합니다.

- **Performance Highlights**: 성능 측면에서, STRisk는 기술적 특징과 소셜 특징을 모두 활용하여 해킹 유출을 예측하는 데 있어 98% 이상의 AUC 점수를 달성하였습니다. 더불어, 개방된 포트와 만료된 인증서가 가장 좋은 기술적 예측 지표로 나타나며, 확산성과 동의성이 최상의 소셜 예측 지표로 판단되었습니다. 이는 정보 보안 분야에서 예측적 접근 방식의 중요성을 강조하고 있습니다.



### RedPajama: an Open Dataset for Training Large Language Models (https://arxiv.org/abs/2411.12372)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks

- **What's New**: 이번 논문은 RedPajama-V1 및 RedPajama-V2라는 두 가지 오픈 데이터셋을 소개합니다. 이 데이터셋들은 대규모 언어 모델을 위한 학습 데이터의 투명성과 접근성을 높이고, 다양한 품질 신호를 포함하여 사용자가 데이터를 필터링할 수 있도록 지원합니다. 특히, RedPajama-V2는 웹에서 수집된 원시 데이터로 구성되어 있어, 차별화된 접근 방식을 제시하고자 합니다.

- **Technical Details**: 논문에서는 RedPajama 데이터셋을 구축하는 데 사용된 절차와 원리, 즉 투명성(transparency), 규모(scale), 다용성(versatility)을 논의합니다. RedPajama-V1은 LLaMA 모델의 훈련 데이터의 오픈 복제본이며, RedPajama-INCITE 모델군을 포함합니다. 반면, RedPajama-V2는 100조 개의 원시 토큰으로 구성된 대규모 데이터셋으로, 다양한 품질 신호를 포함하여 웹 데이터 필터링을 위한 연구에 기여할 수 있습니다.

- **Performance Highlights**: RedPajama 데이터셋은 이미 Snowflake Arctic, Salesforce의 XGen과 같은 생산 모델 훈련에 사용되고 있습니다. 논문에서는 품질 신호를 활용하여 웹 데이터의 하위 집합을 효율적으로 선별하는 방법을 보여주는 일련의 분석 및 배제 연구(ablation studies)를 제공합니다. 이를 통해 RedPajama는 투명하고 고성능의 대규모 언어 모델의 개발을 촉진할 수 있는 잠재력을 지니고 있습니다.



### Perfecting Imperfect Physical Neural Networks with Transferable Robustness using Sharpness-Aware Training (https://arxiv.org/abs/2411.12352)
Comments:
          24 pages, 4 figures

- **What's New**: 본 논문에서는 Sharpness-Aware Training (SAT)이라는 혁신적인 훈련 기법을 통해 물리적 신경망 (PNNs)의 오프라인 및 온라인 훈련의 주요 문제를 해결합니다. 기존의 훈련 방식에서 발생하는 정확도 손실 문제를 해결하며, PNNs가 쉽게 다른 장치로 모델을 전이할 수 있게 합니다. 또한, SAT는 배치 이후 발생하는 교란에 강한 내성을 보이며, PNNs가 지속적으로 정확하게 작동할 수 있도록 합니다.

- **Technical Details**: SAT는 물리 시스템의 손실 지형(loss landscape)을 활용하여 오프라인이나 온라인 훈련 과정에서 발생하는 문제를 해결합니다. 이 기법은 손실 값을 최소화할 뿐 아니라, 손실의 날카로움(sharpness)까지 최소화합니다. SAT는 불확실한 모델을 가진 PNNs에도 보편적으로 적용 가능하며, 다양한 PNN 모델들에서 그 유용성을 입증하였습니다.

- **Performance Highlights**: SAT를 통해 오프라인 훈련된 PNNs는 온라인 훈련된 PNNs보다도 뛰어난 성능을 보여주며, 이는 모델링 및 제작 오차가 존재하는 상황에서도 마찬가지입니다. SAT는 다양한 PNN 구성, 예를 들어 통합 마이크로 링 공진기(MRR), Mach-Zehnder 간섭계(MZI), 그리고 회절 광학 기반 신경망에서 효과적으로 검증되었습니다. 전체적으로 SAT는 PNNs의 실질적이고 효율적인 훈련 및 배치를 위한 솔루션을 제공합니다.



### SNN-Based Online Learning of Concepts and Action Laws in an Open World (https://arxiv.org/abs/2411.12308)
- **What's New**: 이번 연구에서는 스파이킹 신경망(spiking neural network, SNN)을 기반으로 한 완전 자율 바이오 영감을 받은 인지 에이전트의 아키텍처를 제시합니다. 이 에이전트는 자신의 우주를 탐색하며 사물 및 상황의 개념을 순식간에 배웁니다. 특히 행동 개념은 초기 상황(initial situation), 운동 활동(motor activity), 결과(outcome)로 구성된 삼중 구조로 기록되어 에이전트가 우주의 행동 법칙을 이해할 수 있도록 돕습니다.

- **Technical Details**: 에이전트는 개념 개수를 다르게 설정하여 사물/상황 개념과 행동 개념을 분류합니다. 결정-making 과정에서 에이전트는 자신이 가진 의미 기억(semantic memory)을 쿼리하여 예상되는 결과를 기반으로 행동을 선택합니다. 이 방식은 에이전트가 새로운 상황에 대처하는 데 도움을 주며, 더 이전에 학습한 일반 개념을 이용해 빠르게 적응하도록 만듭니다.

- **Performance Highlights**: 실험 결과, 에이전트는 새로운 상황을 효과적으로 처리하며 이전에 학습한 일반 개념에 의존하여 빠르게 개념을 수정할 수 있음을 보여줍니다. 이러한 기능은 에이전트가 환경 변화에 적절하게 반응할 수 있는 능력을 강조합니다.



### Emergence of Implicit World Models from Mortal Agents (https://arxiv.org/abs/2411.12304)
Comments:
          Accepted as a 1-page tiny paper in the Intrinsically Motivated Open-ended Learning workshop at NeurIPS 2024

- **What's New**: 본 연구는 자율 에이전트의 열린 행동 최적화에서 세계 모델과 적극적인 탐색이 강조되는 가능성에 대해 논의합니다. 생물 시스템을 이해함으로써 열린 목표로서의 항상성(homeostasis)과 이와 관련된 메타강화학습(meta-reinforcement learning)의 조합을 통한 세계 모델의 내재적 획득 가능성을 탐구합니다. 이러한 관점은 자율 에이전트의 외부 환경에 대한 지속적 상호작용 속에서 정체성을 유지하는 시스템의 특성을 드러냅니다.

- **Technical Details**: 논문은 메타 강화 학습(meta-RL)과 깊은 항상성 강화 학습(deep homeostatic RL)의 결합을 통해 시스템이 도메인에 적응하면서 나타나는 복잡한 내부 동역학의 가능성을 설명합니다. 또한, 외부 관찰, 최신 행동 선택 및 보상이 메타-RL에서 어떻게 요구되는지 설명하며, 이러한 다중 모달 관찰이 항상성 RL에서 각기 다른 감지 형태와 대응한다고 제안합니다. 네트워크 아키텍처는 RNN(재귀 신경망)을 포함하여 meta-learning 능력을 가능하게 할 수 있는 기초적인 구조를 제공합니다.

- **Performance Highlights**: 자율 에이전트가 항상성을 유지하면서 방법론적으로 내재적 세계 모델을 학습하는 방안을 제시했습니다. 이러한 구조적 접근은 복잡한 상황에서도 반응할 수 있는 에이전트를 개발하는 데 기여할 것으로 기대됩니다. 이 연구는 자율 에이전트의 인공지능 발전에 중요한 발판을 마련하고, 행동 최적화를 통한 경제성과 효율성을 높이는 데 기여할 수 있습니다.



### On the Accuracy and Precision of Moving Averages to Estimate Wi-Fi Link Quality (https://arxiv.org/abs/2411.12265)
Comments:
          preprint, 8 pages, 2024

- **What's New**: 이 논문에서는 무선 링크 품질 추정의 효율성을 간단한 이동 평균 기법을 기반으로 분석하고, 이러한 기법의 장점과 단점을 평가합니다. 또한 인공지능의 활용을 통한 와이파이 네트워크의 비예측성을 완화하기 위한 기초 데이터를 제공합니다. W-Fi 8과 같은 차세대 기술에서 머신러닝의 도입이 점쳐지고 있는 가운데, 무선 네트워크의 동적 환경에 대한 기계학습 기법의 가능성을 탐색합니다.

- **Technical Details**: 무선 네트워크에서의 고信뢰성과 반응 속도는 매우 중요한 요소입니다. 논문에서는 링크 품질 추정과 예측을 구분하며, 현재의 링크 품질을 신뢰성 있게 평가하는 문제를 다룹니다. 단일 전송 시도의 실패 확률을 추정할 수 있는 간단한 모델을 제안하고, 무선 링크에 대한 실험을 통해 무선 채널의 품질을 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 결과적으로, 이동 평균 기법을 이용한 링크 품질 추정의 정확도를 각종 데이터셋을 이용하여 비교 분석하고 성능을 측정합니다. 또한 다양한 스펙트럼 조건에 따른 기법의 적합성을 평가하여, 지연 시간 및 패킷 손실 비율(PLR)에 관한 통계적 정보를 제공합니다. 이러한 접근법은 Wi-Fi와 더 넓은 범위의 무선 전송 기술에 적용 가능성을 제시합니다.



### Restructuring Tractable Probabilistic Circuits (https://arxiv.org/abs/2411.12256)
- **What's New**: 본 논문에서는 확률 회로(Probabilistic Circuits, PCs)의 구조 조정 문제를 제안하고 연구합니다. 기존의 곱셈 알고리즘이 동일한 구조를 유지해야 하는 반면, 본 연구는 특정 목표의 vtree에 맞는 구조로 PC를 변환하는 방법을 제시합니다. 이로 인해 다양한 구조를 가진 회로를 효율적으로 곱할 수 있는 새로운 다항 시간 알고리즘과 실용적인 깊이 감소 알고리즘을 도입하였습니다.

- **Technical Details**: 확률 회로는 확률 분포를 계산 그래프로 표현하며, 이 구조에서 각 노드는 sum, product 및 leaf 노드로 구성됩니다. 논문의 주요 알고리즘은 그래픽 모델 표현을 활용하여 조건부 독립성을 이해하고, 원하는 구조에 맞는 새로운 PC를 재구성하는 것입니다. 또한, 두 가지 중요한 응용 분야인 회로 곱셈과 깊이 감소를 탐구하고 있습니다.

- **Performance Highlights**: 특히 새로운 클래스의 contiguous circuits를 도입하여, 서로 다른 구조의 회로를 다항 시간 내에 곱할 수 있는 가능성을 보여주었습니다. 또한, 깊이 감소 알고리즘을 통해 PC 추론을 더 빠르게 개선하는 방법을 제시하였으며, 이는 PC를 효과적으로 실행하는 데 새로운 가능성을 여는 결과를 가져왔습니다.



### Error-Feedback Model for Output Correction in Bilateral Control-Based Imitation Learning (https://arxiv.org/abs/2411.12255)
- **What's New**: 이 연구에서는 신경망의 출력 오류를 보정하는 피드백 메커니즘을 개발했습니다. 기존의 피드포워드 구조를 가진 신경망의 한계를 극복하기 위해, 상위 계층에 따라 하위 계층을 조정하는 계층적 구조를 도입했습니다. 이를 통해 자율 제어와 오류 피드백을 통한 성능 향상을 확인했습니다.

- **Technical Details**: 제안된 모델은 상위 계층과 하위 계층으로 구성된 계층적 구조로, 상위 계층은 장기 예측을 수행하고 하위 계층은 단기 예측을 담당합니다. 하위 계층은 내부 상태가 없는 다층 퍼셉트론으로 구성되며, 예측한 상태의 오류를 피드백하여 성능을 개선합니다. 이 연구에서는 오류 피드백 모델을 제안하여 시스템의 출력을 조정하는 방법을 보여 주었습니다.

- **Performance Highlights**: 캐릭터 작성 과제를 통해 제안된 모델의 성능을 평가했으며, 이전에 학습하지 않은 캐릭터에 대해 더 향상된 정확도를 기록했습니다. LSTM과 MLP를 비교하여 하위 계층의 구성 요소가 성능에 미치는 영향을 분석했습니다. 이 연구는 신경망과 제어 이론의 통합 가능성을 보여주는 중요한 단계를 나타냅니다.



### Predicting User Intents and Musical Attributes from Music Discovery Conversations (https://arxiv.org/abs/2411.12254)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문은 음악 도메인에서의 사용자 의도 분류(Intent Classification) 작업을 처음으로 다룹니다. 음악 발견 대화(conversational music retrieval) 맥락에서 8개의 인터페이스 제어 레이블과 15개의 음악 속성 레이블을 예측하는 두 가지 음악 전용 의도 분류 작업을 도입했습니다. 또한, 이전 대화 기록을 현재 사용자 쿼리와 결합하는 방법을 제안하여 모델이 전체 대화 맥락을 더 잘 이해할 수 있도록 했습니다.

- **Technical Details**: 연구에서는 의도 분류를 사용자 의도와 음악 속성으로 나누어 사용하였고, 희소 표현(sparse representation), 단어 임베딩(word embedding), DistilBERT, 그리고 Llama 모델을 사용하여 작업을 수행했습니다. 특별히 DistilBERT는 40%의 모델 크기를 줄이면서도, 97%의 언어 이해 능력을 유지하며 60% 더 빠른 속도를 제공합니다. 우리는 또한 DistilBERT의 파라미터를 미세 조정(fine-tuning)한 모델과 분류기만 학습시키는 프로빙(probing) 모델을 비교했습니다.

- **Performance Highlights**: 미세 조정된 DistilBERT 모델은 사용자 의도 및 음악 속성 분류 모두에서 다른 모든 모델보다 뛰어난 성능을 보였습니다. 특히, 음악 속성 분류 성능이 크게 향상되어(F1 스코어 0.46에서 0.72로 증가) 모델이 음악 지식을 효과적으로 습득했음을 보여주었습니다. 일반-purpose Llama 모델은 음악 도메인에 특화된 데이터로 미세 조정된 모델에 비해 성능이 낮았으며, 이는 음악 도메인에 대한 충분한 지식이 부족함을 나타냅니다.



### Hierarchical Spatio-Temporal Uncertainty Quantification for Distributed Energy Adoption (https://arxiv.org/abs/2411.12193)
- **What's New**: 이번 연구는 분산 에너지 자원(DER)의 배치가 전력망 관리에 있어 공간-시간적 불확실성을 초래하며, 이에 대한 효과적인 다단계 예측 방법을 제안합니다. 기존 접근 방식이 개별 공간 단위에서 과도하게 보수적인 불확실성 구간을 생성하고 서로 다른 공간 스케일 간 예측 집계를 제대로 수행하지 못하는 문제를 해결하고자 합니다. 연구팀은 고유한 비순응 점수(non-conformity score)를 활용한 새로운 계층적 공간-시간 모델을 개발하였습니다.

- **Technical Details**: 제안된 모델은 전력망의 계층적 구조를 활용하여 회로 레벨에서 DER 성장 예측을 생성하고 이를 변전소 수준으로 효율적으로 집계합니다. 특히, 이 모델은 다단계 예측 작업에 맞춤화된 새로운 비순응 점수를 도입하여 통계적 유효성을 유지합니다. 연구에서는 실제 미국의 한 도시에서 수집된 10년간의 DER 설치 데이터에 이 모델을 적용하여 그 효과를 검증하였습니다.

- **Performance Highlights**: 이 연구의 결과는 제안된 방법이 기존 방식에 비해 예측 구간의 폭을 줄이며, 적절한 커버리지를 유지하는 데에서 특히 우수한 성과를 보임을 보여주었습니다. 이로 인해 향후 전략적 결정에 있어 실질적인 통찰력을 제공할 수 있는 가능성을 보여줍니다. 전반적으로, 본 연구는 DER 예측 문제를 해결할 수 있는 중요한 기여를 하고 있습니다.



### Constant Rate Schedule: Constant-Rate Distributional Change for Efficient Training and Sampling in Diffusion Models (https://arxiv.org/abs/2411.12188)
Comments:
          33 pages, 9 figures

- **What's New**: 이번 논문에서는 확산 과정 전반에 걸쳐 데이터의 확률 분포 변화율을 일정하게 유지하도록 설계된 노이즈 스케줄(Noise Schedule), CRS(Constant Rate Schedule)를 제안합니다. 이 노이즈 스케줄은 데이터 세트와 확산 모델 유형에 맞게 자동으로 조정되며, 이미지 생성 작업에서의 유효성을 평가했습니다. 실험을 통해 CRS가 다양한 데이터 세트와 샘플러를 포함하여 확산 모델의 성능을 전반적으로 향상시킨다는 것을 확인했습니다.

- **Technical Details**: CRS는 확산 데이터의 확률 분포 변화율을 측정하여, 이는 확산 과정의 추적 가능성을 높이는 데 기여합니다. 노이즈 스케줄의 기능적 형태는 데이터 세트에 따라 자동으로 결정되며, 선형(Linear) 또는 코사인(Cosine)과 같은 미리 정의된 스케줄 없이도 동작합니다. 이는 Song & Ermon(2020)의 일반화된 버전으로 볼 수 있으며, 다음 단계의 확률 분포 간의 상수적 중첩을 달성하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과 CRS는 LSUN, ImageNet, FFHQ 데이터 세트에서 무조건적 및 클래스 조건의 이미지 생성 작업에 있어 상당한 성능 향상을 가져왔습니다. 또한, CRS는 확산 모델의 학습 및 샘플링 모두에 적용 가능하여, 특정 데이터 세트 및 모델 유형에 맞춰 최적화된 노이즈 스케줄을 제공함으로써 개선된 샘플 품질 및 속도를 달성합니다.



### Testability of Instrumental Variables in Additive Nonlinear, Non-Constant Effects Models (https://arxiv.org/abs/2411.12184)
- **What's New**: 이 논문은 관찰 데이터에서 도출된 도구 변수를 테스트하는 문제를 다룹니다. 기존의 많은 연구들은 치료(treatment)가 이산 변수일 때의 시나리오에 중점을 두었으나, 본 연구에서는 연속 변수인 약물 투여량이나 영양 성분 수준과 같이 치료가 연속 변수를 가질 수 있는 상황을 고려합니다. 우리는 Auxiliary-based Independence Test (AIT) 조건을 제안하여 변수의 유효성을 검정하는 방법론을 제시합니다.

- **Technical Details**: 이 논문에서 제안하는 AIT 조건은 Additive Nonlinear, Non-Constant Effects (ANINCE) 모델을 기반으로 하여, 단일 도구 변수가 유효한지를 검증하기 위한 필요 조건을 도입합니다. 특히, 비정상적 인과 효과가 있는 시나리오에서도 적용 가능하며, AIT 조건이 모든 무효 IV를 탐지하기 위한 필요하고 충분한 조건을 제공함을 증명합니다. 본 연구는 유한 데이터에서 공변량(covariates)을 고려한 AIT 조건 테스트의 실용적인 구현을 제시합니다.

- **Performance Highlights**: 합성 데이터와 세 가지 실제 데이터셋에서 우리의 접근 방식이 효과적임을 보여줍니다. 이러한 결과는 AIT 조건의 효능을 입증하고, 다양한 상황에서 무효 도구 변수를 식별하는 데 기여할 수 있음을 시사합니다. 제안된 방법은 관찰 데이터에서 도구 변수를 선택하는 데 있어 반복적으로 거론되던 문제를 해결할 수 있는 실질적인 방법론으로 자리잡을 것입니다.



### Action-Attentive Deep Reinforcement Learning for Autonomous Alignment of Beamlines (https://arxiv.org/abs/2411.12183)
Comments:
          17 pages, 5 figures

- **What's New**: 본 논문은 동기광원( synchrotron radiation source)의 빔라인(beamline) 조정 문제를 마르코프 결정 과정(Markov Decision Process, MDP)으로 모델링하고, 강화 학습(Reinforcement Learning, RL)을 통해 지능형 에이전트를 훈련하는 접근 방식을 제안합니다. 기존의 자동화 방법에서는 현재와 목표 빔 특성 간의 관계를 충분히 고려하지 못하였으나, 이 연구에서는 이를 해결하기 위해 에이전트가 현재 상태와 목표 상태를 분석하여 조정값을 계산합니다. 또한, 액션 어텐션(action attention)에 기초하여 정책 네트워크를 설계하여 더 나은 결정 과정을 유도합니다.

- **Technical Details**: 논문에서 제안한 알고리즘은 에이전트가 현재 상태와 목표 상태의 차이를 인식하여 조정 작업을 수행하도록 설계되었습니다. 이를 위해, 액션 어텐션 기반의 정책 네트워크가 사용되어 다양한 광학 장치의 조정에 필요한 최적의 액션을 생성합니다. 실험을 통해서는 두 개의 시뮬레이션 빔라인에서 본 방법의 우수성을 증명하며, 기존 방법들이 가진 한계를 극복하는 데 성공하였습니다.

- **Performance Highlights**: 제공된 실험 결과에 따르면, 제안한 알고리즘은 기존 방법들보다 더 뛰어난 성능을 보여주었습니다. 특히, ablation study를 통해 액션 어텐션 기반 정책 네트워크의 효과성이 강조되었습니다. 이는 지능형 에이전트가 빔라인 조정 작업에서 효과적으로 학습하고 결정을 내릴 수 있도록 돕는 중요한 요소로 작용함을 보여주었습니다.



### Sensor-fusion based Prognostics Framework for Complex Engineering Systems Exhibiting Multiple Failure Modes (https://arxiv.org/abs/2411.12159)
- **What's New**: 이 논문에서는 복잡한 시스템의 여러 고장 모드를 고려한 Remaining Useful Life (RUL) 예측 방법론을 제안합니다. 특히, 이 방법론은 알려지지 않은 레이블의 데이터에 대해 센서 선택과 클러스터링을 동시에 수행합니다. 이를 통해 활성화된 고장 모드를 실시간으로 진단하고 RUL을 예측할 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 (1) 센서 선택 알고리즘, (2) 차원 축소 및 데이터 융합 방법론, (3) RUL 예측 모델을 포함하는 통합 프레임워크로 구성됩니다. 이 프레임워크는 역사 데이터셋을 기반으로 하여 고장 이벤트를 진단하고 각 고장 모드와 관련된 정보가 많은 센서의 하위 집합을 선택합니다. 또한, 오프라인 과정에서 Expectation-Maximization (EM) 알고리즘을 활용하여 고장 라벨링과 최적 센서를 동시에 선택하게 됩니다.

- **Performance Highlights**: 제안된 방법론은 NASA의 터보팬 엔진 데이터 세트를 포함한 고장 모드 시뮬레이션 데이터 세트를 사용하여 검증되었습니다. 이러한 검증을 통해 여러 고장 모드를 가진 시스템에서 효율적으로 작동하는 RUL 예측과 실시간 진단 기능을 확인하였습니다. 이 연구의 결과는 다양한 복잡한 엔지니어링 시스템의 유지보수 전략을 향상시키는데 기여할 것으로 기대됩니다.



### Tangential Randomization in Linear Bandits (TRAiL): Guaranteed Inference and Regret Bounds (https://arxiv.org/abs/2411.12154)
Comments:
          42 pages, 6 Figures

- **What's New**: 이 논문에서는 Linear Bandits 문제에 위한 새로운 강제 탐사 알고리즘인 TRAiL(Tangential Randomization in Linear Bandits)을 제안합니다. TRAiL는 강한 볼록 함수의 하위 수준 집합에 대한 행동 집합에서 최적화된 후회 최소화(regret-optimal) 성능을 보장합니다. 이 방법은 표준 정규화 최소 제곱(regularized least squares)을 통해 선형 밴디트 문제의 지배 매개변수를 추정하고, 이 값을 기반으로 최적 보상 행동을 조정하여 탐사와 활용 사이의 균형을 유지합니다.

- **Technical Details**: TRAiL는 매트릭스 마팅게일(matrix martingales)의 집중 결과를 활용하여, T길이 동안의 추론 품질이 𝓞(√T)로 증가함을 보장합니다. 이를 통해 𝓞(√T log(T))의 누적 후회에 대한 상한을 도출하며, 확률적으로 최소 1 - 1/T 이상의 정확도를 보입니다. 추가적으로, 우리는 다양한 행동/매개 변수 세트 및 노이즈 프로세스에 대한 기대 후회에 대한 𝛀(√T) 미니맥스 하한(minimax lower bound)을 특성화합니다.

- **Performance Highlights**: 논문 결과에 따르면, TRAiL 알고리즘은 다른 인기 있는 알고리즘들과 비교하여 우수한 성능을 보입니다. 특정 실험에서는, TRAiL이 원하는 후회 감소 비율을 달성하는 데 적절한 추론 속도를 유지해야 함을 보여줍니다. 즉, 너무 빠르거나 느린 추론 속도는 후회 증가에 대한 비최적 성장을 초래하게 됩니다.



### HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments (https://arxiv.org/abs/2411.12150)
- **What's New**: 이번 논문에서는 복잡한 환경에서 로봇 내비게이션을 위한 새로운 프레임워크를 제안합니다. 특히, 다양한 상호작용을 모델링하는 이질적 시공간 그래프(heterogeneous spatio-temporal graph)를 활용하여 로봇이 장애물 및 인간과 충돌하지 않도록 내비게이션 정책을 학습하는 방법을 소개합니다. HEIGHT라는 네트워크 아키텍처를 통해 중요한 상호작용을 우선시하고 동적 장면의 변화를 추적할 수 있는 메커니즘을 제공합니다.

- **Technical Details**: HEIGHT는 서로 다른 상호작용을 포착하기 위해 두 개의 멀티헤드 주의 네트워크(multi-head attention networks)를 사용하며, 단일 방향의 장애물-행위자(interaction) 상호작용을 모델링하기 위해 MLP(Multi-Layer Perceptron)를 활용합니다. 또한, 시퀀스의 시간적 진화를 고려하는 순환 네트워크(recurrent network)를 도입하여 로봇이 동적 환경에서 적응적인 내비게이션을 수행할 수 있도록 지원합니다. 이러한 접근은 로봇이 인간과 장애물 간의 복잡한 상호작용을 효과적으로 관리하는 데 도움을 줍니다.

- **Performance Highlights**: HEIGHT의 성능은 다양한 시뮬레이션과 실제 실험을 통해 입증되었습니다. 특히, 인간과 장애물 밀도가 변화할 때 더 나은 제로샷 제너럴리제이션(zero-shot generalization) 능력을 보여주며, 과거의 최첨단 방법들을 초월하는 내비게이션 성공률과 효율성을 기록했습니다. 이 방식은 내구성 있는 로봇 정책을 저비용 시뮬레이터에서 학습하여 실제 복잡한 환경에서도 활용 가능한 가능성을 제시합니다.



### Self-supervised denoising of visual field data improves detection of glaucoma progression (https://arxiv.org/abs/2411.12146)
Comments:
          10 pages

- **What's New**: 이 연구에서는 4,000명 이상의 환자에서 수집된 시야(data from visual field) 데이터를 노이즈 제거(denoising)하는 데 자기 지도 학습(self-supervised learning)을 활용했습니다. 기존의 방법보다 시야 데이터의 평균 대 잡음 비율(signal-to-noise ratio)을 개선했고, 안압 상승으로 인한 시각 손상의 진행을 더 정확하게 감지할 수 있었습니다. 우리는 변이형 오토인코더(variational autoencoder)와 마스크 오토인코더(masked autoencoder)를 사용하여 어떤 모델이 시야 데이터를 더 잘 처리하는지 비교했습니다.

- **Technical Details**: 이 연구에서는 심층 신경망(deep neural networks) 구조를 기반으로 한 두 가지 모델, 변이형 오토인코더(VAE)와 마스크 오토인코더가 시야 데이터의 노이즈를 줄이는 데 사용되었습니다. 데이터 수집은 Humphrey Field Analyzer II를 통해 진행되었으며, 신뢰성 기준(criterion of reliability)은 잘못된 양성(false positive) 비율이 15% 이하, 고정 손실(fixation loss) 및 잘못된 음성(false negative) 비율이 30% 이하로 설정되었습니다. 모델 출력에서 카테고리 p-값(categorical p-value)을 사용하는 방식도 효과성이 입증되었습니다.

- **Performance Highlights**: 마스크 오토인코더는 시야 데이터를 효과적으로 정리하여 변이형 오토인코더보다 4.7% 더 높은 진행 예측률을 기록했습니다. 포함된 p-값 덕분에 진행을 예측하는 시간이 평균 2.3개월 더 빨라졌다는 분석이 이루어졌습니다. 이러한 결과는 마스킹(masking) 및 p-값 포함이 시야 진행 감지 작업을 개선하는 데 기여할 수 있음을 보여줍니다.



### A Computational Method for Measuring "Open Codes" in Qualitative Analysis (https://arxiv.org/abs/2411.12142)
- **What's New**: 이 논문에서는 질적 분석(valitative analysis)의 중요한 방법론인 open coding의 잠재적인 편향을 체계적으로 측정하고 식별하는 새로운 계산 방법을 제안합니다. 기존 연구들은 open coding의 결과를 정확하게 측정하지 않아 편향의 위험을 증가시켜 왔습니다. 이 방법은 Grounded Theory와 Thematic Analysis 이론에 기반하여 인간과 기계 코더 간의 팀 기반 접근 방식을 활용합니다.

- **Technical Details**: 제안된 방법은 두 가지 HCI 데이터셋을 사용하여 open 코드의 신뢰성을 측정하는 것입니다. 이 과정에서 Coverage, Density, Novelty, Divergence라는 네 가지 개념 지표를 운영화하여 코드 스페이스를 평가합니다. 이러한 지표는 팀 코더의 결과에 대한 개별 코더의 결과를 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 이 연구는 인간 분석과 비교하여 제안된 방법의 결과의 안정성을 통계적으로 분석함으로써 신뢰성을 검증합니다. 기계 측정과 인간 해석을 결합한 결과, 질적 연구자들이 GAI를 유도 분석에서 사용하는 데 도움이 될 수 있는 근거 기반의 제안과 예제 워크플로우를 제시합니다.



### Exact Risk Curves of signSGD in High-Dimensions: Quantifying Preconditioning and Noise-Compression Effects (https://arxiv.org/abs/2411.12135)
- **What's New**: 최근 몇 년 동안, signSGD는 실용적인 옵티마이저(optimizer) 및 Adam과 같은 적응형 옵티마이저를 이해하는 간단한 모델로 주목받고 있습니다. 이 논문에서는 signSGD가 최적화(optimization)에 미치는 영향을 이론적으로 해소하기 쉬운 설정에서 정량적으로 분석합니다. 고차원(High dimensional) 한계에서 의도된 SDE와 ODE를 도출하여 그 위험을 설명합니다.

- **Technical Details**: signSGD의 네 가지 효과를 분석하는 데 집중하며, 이는 효과적인 학습률(effective learning rate), 노이즈 압축(noise compression), 대각선 전처리(diagonal preconditioning), 그리고 그래디언트 노이즈 재형성(gradient noise reshaping)입니다. 이 분석은 실험적 관찰과 일치하지만, 데이터와 노이즈 분포에 대한 이러한 효과의 의존성을 정량화하는 데 나아갑니다. 이론적 틀을 통해 고차원에서의 효과를 수식으로 표현하며, signSGD의 동작을 보다 명확히 설명합니다.

- **Performance Highlights**: 마지막으로, 이러한 결과가 Adam에 어떻게 확장될 수 있을지에 대한 추측으로 논문을 마무리합니다. signSGD의 주요 속성을 정량적으로 설명함으로써, 이를 기반으로 한 다양한 옵티마이저 개선이 가능할 것으로 기대됩니다. 특히, 이 연구는 고차원 데이터 세트에서의 signSGD의 적용 가능성을 제시하며 최적화 기술의 발전에 기여할 것으로 보입니다.



### Distill the Best, Ignore the Rest: Improving Dataset Distillation with Loss-Value-Based Pruning (https://arxiv.org/abs/2411.12115)
- **What's New**: 최근 데이터셋 증류(dataset distillation) 방식이 새롭게 주목받고 있으나, 기존 방법들은 전체 데이터셋에서 비유용한 샘플을 포함하는 경우가 많았습니다. 본 논문에서는 'Prune First, Distill After'라는 새로운 프레임워크를 도입하여, 증류 전에 손실 기반 샘플링 방법으로 데이터셋을 체계적으로 가지치기(pruning)합니다. 이를 통해 미지의 아키텍처에 대한 일반화 능력을 향상시키는 대표적인 코어 세트를 생성합니다.

- **Technical Details**: 우리의 접근법은 손실 값에 기반한 샘플링 전략을 사용하는데, 이는 사전 훈련된 분류기 모델을 활용하여 데이터 샘플을 '분류 난이도(classification difficulty)'에 따라 순위 매기는 방식입니다. 이 과정에서 단순한 샘플과 복잡한 샘플을 각각 먼저 선택하는 두 가지 샘플링 전략을 비교하였으며, 단순 샘플에 집중할 경우 증류 품질이 크게 향상되는 것을 발견했습니다. 또한, 우리는 StyleGAN-XL 및 수정된 디퓨전 모델을 포함한 최신 데이터셋 증류 기법들을 기반으로 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 원래 데이터셋의 80%를 제거하고도 증류된 데이터셋에서 최대 5.2%의 정확도 향상이 이루어졌습니다. 우리의 접근법은 여러 ImageNet 하위 집합에서 다양한 아키텍처를 대상으로 한 광범위한 평가를 통해 유연성과 강건성을 입증했습니다. 이러한 성과는 데이터셋 증류의 효과성을 높이는 가능성을 제시하며, 더 나은 품질의 데이터셋을 생성하는 데 기여합니다.



### Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods (https://arxiv.org/abs/2411.12103)
Comments:
          9 pages, 2 figures

- **What's New**: 이번 논문은 대형 언어 모델(LLM)에서 해로운 정보를 제거하는 방법인 LLM 언러닝(LLM unlearning)을 제안하고, LLMU와 RMU 두 가지 방법을 비교 분석합니다. 실험을 통해 이들 방법이 모델의 일반적 능력에 미치는 영향을 평가하였으며, 새로운 생물학 벤치마크도 사용하여 효과를 점검했습니다. 이 논문은 기존 방법의 평가 범위를 확장하고, 단순한 프롬프트 방식이 언러닝 성능에 미치는 영향을 조사했습니다.

- **Technical Details**: LLMU는 해로운 데이터에 대해 손실 함수의 음수를 취해 그래디언트 상승(gradiant ascent)을 통해 학습하며, RMU는 해로운 데이터에 대해 무작위 제어 벡터에 맞추어 특정 층을 학습시킨다. 저자들은 생물학 중심 데이터셋을 사용하고, 5-shot 프롬프트 방식 등 다양한 프롬프트 전략으로 샘플을 변경하여 언러닝의 효과를 개선했음을 보여줍니다. 실험 결과에 따르면 이 방식이 언러닝 벤치마크의 정확성을 최대 1750% 향상시킬 수 있음을 증명했습니다.

- **Performance Highlights**: 결과적으로 RMU 방법이 LLM의 일반적인 능력 보존이 더 우수한 경향을 보였으며, 임의의 데이터로 재학습할 경우 언러닝 전 성능을 거의 완전히 복구할 수 있음을 확인했습니다. 이는 LLM 언러닝 방법이 실제로는 해로운 정보를 완전히 제거하지 못함을 시사합니다. 전체적으로, 이 연구는 언러닝 방법의 강건성과 효과에 대해 새로운 통찰을 제공합니다.



### Mitigating Gender Bias in Contextual Word Embeddings (https://arxiv.org/abs/2411.12074)
- **What's New**: 이 논문에서는 기존의 정적(Static) 단어 임베딩에서 발생하는 성 편향을 줄이기 위해 새로운 Masked-Language Modeling (MLM) 목표 함수를 제안합니다. 제안된 방법은 성 편향을 완화하면서도 다운스트림 작업의 성능을 유지할 수 있도록 설계되었습니다. 또한 문맥(Contextual) 임베딩에서의 편향 측정과 관련된 새로운 평가 지표들을 제안하여 편향 완화에 대한 목적에 부합합니다.

- **Technical Details**: 연구팀은 성 편향을 줄이기 위한 실험 분석을 진행하며, Masked-Language Modeling의 새로운 목표 함수를 포함하여 두 가지 전략을 통해 문맥적 모델의 학습을 지속 가능한 방향으로 진행합니다. 제안된 방법은 모든 명사를 성 중립적 단어로 간주하는 방식으로, 문장에서 무작위로 명사를 마스킹하고 나머지 토큰을 이용하여 예측하는 방식입니다. 두 번째 전략에서는 성 속성이 있는 단어를 마스킹하고 나머지 단어들을 사용하여 예측하도록 훈련함으로써 두 가지 분리된 클러스터 문제를 해결하려고 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 문맥적 임베딩의 성 편향을 크게 줄이는 데 성공하였으며, 성 중립적 단어를 통해 성별 정보의 노출을 최소화하여 다운스트림 작업에서의 성능도 유지합니다. 추가적으로, 기존 데이터의 불균형적인 참조를 해결하기 위해 성 예측 작업을 통해 데이터를 지능적으로 증강하는 전략을 도입했습니다. 이러한 접근 방식이 성 편향을 줄이는 데 효과적임을 입증하는 실증적 증거를 제공합니다.



### Just Leaf It: Accelerating Diffusion Classifiers with Hierarchical Class Pruning (https://arxiv.org/abs/2411.12073)
- **What's New**: 최근의 연구에 따르면, 전통적인 생성 모델이 단순한 콘텐츠 생성에만 국한되지 않고 분류 작업에서도 여전히 활용 가능함을 보여주고 있습니다. 특히, Hierarchical Diffusion Classifier(HDC)는 고유한 계층적 레이블 구조를 활용하여 높은 계산 비용을 줄이면서 이미지 분류 효율성을 향상시키고 있습니다. 이 접근 방식을 통해 HDC는 최대 60%의 속도 향상을 달성하면서도 분류 정확도를 유지하거나 개선할 수 있음을 증명했습니다.

- **Technical Details**: HDC는 전통적인 확산 모델을 기반으로 하며, 레이블 트리를 계층적으로 탐색하여 불필요한 높은 수준의 범주를 점진적으로 제거합니다. 초기 단계에서는 가장 유망한 synsets를 유지하기 위해 레이블 트리를 레벨별로 트래버스하고, 그 후 남은 후보 리프 노드에서 전통적인 확산 분류를 수행합니다. 이로 인해, 불필요한 클래스를 사전에 제거함으로써 계산 비용을 줄이고 정확한 예측을 가능하게 합니다.

- **Performance Highlights**: HDC는 대규모 이미지 분류 작업인 ImageNet-1K에서 약 60%의 빠른 추론 시간을 달성하며, 비슷한 계산 시간에 기존의 확산 분류 모델보다 더 나은 정확도(65.16% vs. 64.90%)를 기록하였습니다. 이를 통해 HDC는 분류 빠르기와 정확도 사이의 새로운 균형을 제공하고, 실질적인 대규모 분류 작업에서도 효과적으로 활용될 수 있는 가능성을 제시합니다.



### Zoomed In, Diffused Out: Towards Local Degradation-Aware Multi-Diffusion for Extreme Image Super-Resolution (https://arxiv.org/abs/2411.12072)
- **What's New**: 기존의 Text-to-Image (T2I) 확산 모델은 512x512 해상도로 제한되어 있었으나, 본 연구에서는 추가 학습 없이 2K, 4K, 심지어 8K 해상도로 이미지를 생성할 수 있는 새로운 접근 방식을 소개합니다. 이 방법은 MultiDiffusion과 지역 손실 인식 프롬프트 추출이라는 두 가지 핵심 요소를 활용하여 고해상도 이미지를 생성하면서도 전 세계적으로 일관성을 유지합니다. 이러한 혁신은 이미지 초해상도(Super-Resolution, SR) 작업에 T2I 확산 모델을 적용하는 새로운 가능성을 제공합니다.

- **Technical Details**: 이 연구의 방법론인 MultiDiffusion은 이미지를 생성하는 과정을 여러 개의 확산 경로에 분산시켜 높은 해상도에서도 전 세계적인 일관성을 보장합니다. 각 단계에서 잠재 피처 맵은 중첩되는 타일로 나누어져 개별적인 확산 과정을 거치며, 이로 인해 인접한 타일 간의 정보를 공유하여 전반적인 구조와 지역 세부 사항의 일관성을 유지합니다. 이러한 과정은 기존 T2I 확산 모델에 비해 512×512 픽셀의 제한 없이 2K 이상의 해상도를 가능하게 합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존의 모델과 비교했을 때 방향성과 세부 사항을 보다 잘 복원할 수 있는 능력을 갖추고 있습니다. 결과적으로, 초해상도의 새로운 기준을 세우며, T2I 모델이 가진 잠재력을 극대화하여 다양한 해상도의 이미지 생성에 효과적으로 활용될 수 있음을 보여주었습니다. 모델의 성능은 2K, 4K, 8K에서의 해상도 증대에 성공적으로 적용되면서 향후 SR 작업의 가능성을 더욱 확장시킵니다.



### Autoassociative Learning of Structural Representations for Modeling and Classification in Medical Imaging (https://arxiv.org/abs/2411.12070)
Comments:
          16 pages, 9 figures

- **What's New**: 이 연구에서는 인간의 인지능력과 보다 일치하는 neurosymbolic 시스템 개념을 제안합니다. ASR(자동 연관 구조 표현)는 관찰된 이미지를 시각적 원시 요소로 재구성함으로써 고수준의 구조적 설명을 형성하도록 강요하는 아키텍처입니다. 이 방법은 히스토로지(조직학) 이미징에서 비정상 진단 작업에 적용되었습니다.

- **Technical Details**: ASR 아키텍처는 주로 Encoder, Modeler, Renderer의 세 가지 구성 요소로 이루어져 있습니다. Encoder는 연속적인 ConvBlock으로 구성되며, Modeler는 여러 공간적 스케일에서 얻어진 잠재 벡터를 해석 가능한 그래픽 원시 요소의 매개변수로 매핑합니다. Renderer는 최종 출력 이미지의 시각적 재현을 담당하며, 전문화된 훈련을 통해 원시 요소의 재구성과 함께 작동합니다.

- **Performance Highlights**: ASR 모델은 상대적으로 적은 데이터 세트에서도 효율적으로 학습이 가능하며, 기존의 딥러닝 모델에 비해 더 나은 해석력을 제공합니다. 연구 결과, ASR은 의료 이미징 분류에서 뛰어난 정확도를 보여 주었고, 이는 의료 분야에서의 활용 가능성을 높여 줍니다.



### The Statistical Accuracy of Neural Posterior and Likelihood Estimation (https://arxiv.org/abs/2411.12068)
- **What's New**: 이 논문에서는 Neural Posterior Estimation (NPE)와 Neural Likelihood Estimation (NLE) 방법의 통계적 행동에 대한 심층 탐구를 처음으로 수행하였습니다. 기존의 통계적 방법인 Approximate Bayesian Computation (ABC)와 Bayesian Synthetic Likelihood (BSL)과 유사한 이론적인 보장을 갖고 있음을 증명하였고, NPE와 NLE의 정확도가 ABC 및 BSL과 유사하다는 것을 보여주었습니다. 또한 NPE와 NLE는 더 낮은 계산 비용으로 이러한 정확도를 달성할 수 있음을 강조합니다.

- **Technical Details**: NPE와 NLE는 복잡한 모델에서 Bayesian 추론을 수행하기 위해 설계된 신경 조건 밀도 근사 방법입니다. 이들은 데이터로부터 회귀 모델을 훈련하여 가능도(likelihood) 및 사후 분포(posterior distribution)를 근사하는 데 사용됩니다. NLE는 이후 Markov Chain Monte Carlo (MCMC) 단계를 요구하는 반면, NPE는 단일 라운드에서 추론을 수행하여 재훈련 없이 여러 데이터 세트에서 사용할 수 있는 장점이 있습니다.

- **Performance Highlights**: NPE와 NLE 방법의 통계적 행동을 분석하고, 이들이 ABC 및 BSL과 유사한 정확도를 유지함을 밝혀냈습니다. 특히, 이 연구는 NPE와 NLE 방법의 결과가 차원이 증가할 때 더 낮은 영향을 받는다는 점을 강조하며, 이는 높은 차원의 요약 및 매개변수의 Curse of Dimensionality 문제를 완화하는 데 기여합니다. 결과적으로 이 두 방법은 유사한 이론적 보장 아래에서 비슷한 통계적 행동을 보이지만, NPE가 MCMC 샘플링을 필요로 하지 않으므로 더 우수한 것으로 판단됩니다.



### Benchmarking pre-trained text embedding models in aligning built asset information (https://arxiv.org/abs/2411.12056)
- **What's New**: 이 논문은 구축 자산(data related to built assets) 정보를 전문 용어에 맞춰 잘 정렬하기 위한 새로운 접근 방식을 제안합니다. 특히 최근의 대형 언어 모델을 활용한 텍스트 임베딩(text embedding)이 자산 관리의 데이터 매핑(data mapping) 과정의 자동화를 모색하는 데 기여할 수 있음을 보여줍니다. 이 연구는 기존 모델의 성능을 비교하고, 구축 자산에 특화된 기술 용어의 복잡한 의미를 효과적으로 표현할 수 있는지를 평가합니다.

- **Technical Details**: 이 연구는 다양한 하위 도메인에 걸쳐 구축 제품에 대한 정보의 체계적인 데이터를 구성하고, 이를 기반으로 구체적 작업 세트를 개발합니다. 데이터는 산업 기초 클래스(Industry Foundation Classes, IFC)와 유니클래스(Uniclass)의 두 가지 주요 출처에서 수집되었습니다. 또, 제안된 데이터 세트는 여섯 가지 작업, 즉 클러스터링(clustering), 검색(retrieval), 재순위화(reranking)를 평가하여 모델간의 성능을 비교하는 데 중점을 둡니다.

- **Performance Highlights**: 이 연구의 평가 결과는 자동화된 데이터 매핑 과정에서 구성 자산 정보를 정렬하는 데 있어 기존 언어 모델의 강점과 한계를 안내합니다. 현재 관련된 24개의 텍스트 임베딩 모델을 사용하여 10,000개 이상의 데이터 항목을 포함하는 가장 포괄적인 벤치마크를 제공합니다. 또한 연구 결과 및 데이터 세트는 공개 소스로 제공되어 후속 연구의 기초 자료로 사용될 수 있도록 하였습니다.



### Prediction-Guided Active Experiments (https://arxiv.org/abs/2411.12036)
Comments:
          25 pages, 11 figures

- **What's New**: 이번 연구에서는 Prediction-Guided Active Experiment (PGAE)이라는 새로운 프레임워크를 도입합니다. 이 프레임워크는 기존의 머신러닝 모델에서 예측을 이용하여 샘플링 및 실험을 안내합니다. 이 연구는 비적응적(non-adaptive) 사례에서 실험 전략을 최적화하고, 이후 적응적(adaptive) 조건에서도 예측의 효율성을 유지하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 실험 설계를 위한 두 가지 주요 요소인 샘플링 분포(sampling distribution)와 실험 확률(experimental probability)을 고려합니다. 이 연구는 비적응적 설정에서 세미 파라메트릭 효율 하한(semi-parametric efficiency lower bound)을 최소화하여 효율적인 샘플링 밀도(sampling densities)를 도출합니다. 또한, 적응적 환경에서는 예측과 실제 결과 간의 관계가 초기에는 알려져 있지 않아도, 효율적인 추정기를 제안하며 두 설정 모두에서 분산 하한을 달성합니다.

- **Performance Highlights**: PGAE의 성능은 시뮬레이션 및 미국 인구조사국의 데이터를 활용한 반-합성 실험을 통해 검증되었습니다. 연구 결과, PGAE 프레임워크는 기존 방법들에 비해 뛰어난 효율성과 효과성을 보이며, 다양한 분야에서의 실험적 접근법을 혁신할 잠재력을 지니고 있습니다.



### On the Efficiency of ERM in Feature Learning (https://arxiv.org/abs/2411.12029)
Comments:
          23 pages, 0 figures

- **What's New**: 이 논문은 주어진 feature maps의 집합을 바탕으로 empirical risk minimization (ERM)의 성능을 연구합니다. 특히, regression 문제에서의 square loss를 고려하여 다양한 feature maps가 유도하는 선형 클래스의 집합에서 ERM의 성능을 평가합니다. 이를 통해 데이터 기반의 feature 학습이 필요한 경우, 성능 저하를 최소화하기 위한 적절한 feature map과 선형 예측기를 동시에 학습해야 함을 강조합니다.

- **Technical Details**: 연구에서는 비선형 모델 예측기 및 임의의 feature maps에서의 성능을 이론적으로 분석합니다. 특히, 특정한 조건 하에 optimal feature map이 유일할 경우, excess risk의 asymptotic quantiles가 oracle 절차의 그것과 비슷하다는 것을 보였습니다. 또한, global complexity가 ERM의 excess risk에 미치는 영향을 비대칭적으로 분석하여 feature maps의 suboptimality와의 연관성을 제시합니다.

- **Performance Highlights**: 저자들은 sparsity를 고려한 선형 회귀에서 최적의 서브셋 선택 절차의 성능에 대한 새로운 보장을 도출했습니다. 이를 통해 모델 복잡도의 감소가 궁극적으로 optimal feature maps의 집합 크기에만 의존하게 되는 현상을 발견하였습니다. 이러한 발견은 데이터 기반 feature 학습 모델에서의 샘플 수 필요성을 줄이는 데 기여할 수 있습니다.



### Regret-Free Reinforcement Learning for LTL Specifications (https://arxiv.org/abs/2411.12019)
- **What's New**: 본 논문에서는 LTL 규격을 위한 최초의 무후회(무정체) 온라인 알고리즘을 제안합니다. 이 알고리즘은 유한한 상태와 행동을 가진 Markov 결정 과정(MDP)에서 작동하며, 최적 행동에 도달하기 위한 학습 중 성능을 평가합니다. 기존의 LTL 과제를 위한 붕괴없는 RL 기법이 없었던 점에서, 이 연구는 중요한 기여를 합니다.

- **Technical Details**: 제안하는 알고리즘은 무한 지평선 도달-회피 문제를 해결하기 위해 무후회 지도 학습 기법을 기반으로 합니다. 기존의 RL 알고리즘은 최적 정책이 위치적일 것이라는 가정을 하고 있으며, 이는 보다 일반적인 보상 구조에서는 유효하지 않을 수 있습니다. LTL 공식을 적용한 제어 목표에서는 보통 최적 정책이 위치적이지 않기 때문에, 이 알고리즘은 그래프 구조를 인식하고 최소 전이 확률을 가정합니다.

- **Performance Highlights**: 현재 제안된 알고리즘은 수렴 속도가 서브선형적이며, 무후회 학습 알고리즘을 통해 특정 LTL 목표에 대한 성능 평가를 제공합니다. K > 0 에피소드를 수행한 후, 알고리즘의 후회 값은 K에 대해 0으로 수렴합니다. 이를 통해 알고리즘이 안전하고 신뢰할 수 있는 방식으로 제어 정책을 학습하고 성능을 평가할 수 있음을 보여줍니다.



### Pricing Weather Derivatives: A Time Series Neural Network Approach (https://arxiv.org/abs/2411.12013)
- **What's New**: 이 논문은 온도(temperature)와 강수량(precipitation)을 기초 변수로 한 날씨 파생상품(weather derivative)의 가격을 신경망(neural network) 접근법과 시계열(time series) 예측을 결합하여 평가하는 방법을 제시합니다. 특히, 온도 및 강수량을 기반으로 한 Pacific Rim Index의 가치를 평가하는 것이 주 목표입니다. 특히, 신경망 모델을 통한 시계열 예측에 대한 탐색이 더 적게 이루어진 점을 강조합니다.

- **Technical Details**: 가격 책정은 역사적 측정(historic measure) 아래에서 계약의 할인된 기대 손실(discounted expected losses)을 기반으로 하며, 역사적 행동(historic behavior)과 변동성을 고려합니다. 온도 예측을 위해선 Autoregressive Moving Average(ARMA) 모델과 하모닉 회귀(harmonic regression)를 결합하고, 신경망 접근법을 통해 보완합니다. 강수량 모델은 감마 분포(Gamma distribution)를 따르는 독립적인 랜덤 변수들의 랜덤 합으로 가정됩니다.

- **Performance Highlights**: 토론토(Toronto)와 시카고(Chicago)에서의 온도 및 강수 데이터에 대해 시계열 분석과 신경망 접근법을 모두 구현합니다. 2023년 12월의 온도 예측치를 제시하며, 신경망 모델이 전통적인 모델과 비교하여 성능이 뛰어날 것으로 기대하고 있습니다. 최종적으로, 연구 결과는 향후 날씨 파생상품의 가격 책정에 기여할 것으로 보입니다.



### SynCoTrain: A Dual Classifier PU-learning Framework for Synthesizability Prediction (https://arxiv.org/abs/2411.12011)
- **What's New**: 이 논문은 신소재 발견의 중요성을 강조하며, 새로운 소재의 합성을 예측하는 데 있어 기존 방식의 한계를 극복하기 위해 개발된 Semi-supervised machine learning 모델인 SynCoTrain을 소개합니다. 특히 이 모델은 두 개의 상호 보완적인 그래프 컨볼루션 신경망인 SchNet과 ALIGNN을 활용하여 시대의 요구에 맞는 더욱 정교한 예측을 제공합니다. 연구는 Positive and Unlabeled (PU) Learning 방식을 도입하여 데이터 집합의 변동성을 관리하고, 컴퓨팅 효율성을 향상시킵니다.

- **Technical Details**: SynCoTrain은 co-training 프레임워크를 기반으로 하여 엇갈린 예측을 서로 교환함으로써 모델의 편향을 감소시키고 일반화 능력을 향상시킵니다. 이 모델은 산화물 결정체를 중심으로 합성이 가능한 소재 예측을 수행하며, 기존 데이터 부족 문제를 해결하기 위해 PU Learning 기법을 활용하여 반복적으로 예측을 정제합니다. ALIGNN 모델은 원자 결합 및 결합 각도를 직접 인코딩하는 독특한 구조를 가지고 있으며, SchNetPack 모델은 연속적인 컨볼루션 필터를 사용하여 원자 구조를 효과적으로 인코딩합니다.

- **Performance Highlights**: SynCoTrain은 내부 및 이탈 테스트 세트에서 높은 recall을 달성하며 견고한 성능을 입증하였습니다. 이는 oxide crystals라는 잘 특성화된 물질 가족을 대상으로 하여 이루어진 것으로, 데이터 집합의 변동성과 계산 효율성을 균형있게 유지하게 해줍니다. 이 연구는 co-training의 잠재력을 강조하며, 높은 처리량의 소재 발견 및 생성적 연구의 발전에 기여할 수 있는 확장 가능한 솔루션을 제공합니다.



### Active learning for efficient discovery of optimal gene combinations in the combinatorial perturbation spac (https://arxiv.org/abs/2411.12010)
- **What's New**: 이번 연구에서는 NAIAD라는 새로운 액티브 러닝 프레임워크를 소개하여, 세포를 원하는 세포 표현형으로 이끄는 최적의 유전자 쌍을 효율적으로 발견할 수 있는 방법을 제시합니다. NAIAD는 단일 유전자 교란 효과와 데이터 크기에 따라 적응하는 유전자 임베딩을 활용하여 작은 샘플 학습에서의 과적합(overfitting)을 완화합니다. 이 프레임워크는 총 35만 건 이상의 유전자 상호작용이 포함된 데이터셋을 평가했으며, 작은 데이터셋에서 훈련된 NAIAD는 기존 모델보다 최대 40% 향상된 성능을 보였습니다.

- **Technical Details**: NAIAD는 실험에서 얻은 작은 데이터셋으로 초기 모델을 훈련시켜, 이전에 보지 못한 조합의 효과를 예측할 수 있도록 합니다. 이러한 예측은 다음 CRISPR 스크리닝 라이브러리의 설계를 안내하며, 이를 통해 최적의 유전자 조합을 반복적으로 식별할 수 있는 기반을 제공합니다. 프레임워크 내에서 최대 예측 효과(Maximum Predicted Effects, MPE) 기반의 추천 시스템이 도입되어, 높은 마진 이득을 가져오는 유전자 조합을 우선적으로 추천합니다.

- **Performance Highlights**: NAIAD는 유전자 쌍의 최대 예측 효과를 우선시함으로써 매 AI 실험 라운드에서 최고 마진 이득을 얻으면서 실험 반복 수를 줄이는 데 기여합니다. 이 연구 결과는 효율적인 CRISPR 라이브러리 설계를 가능하게 하며, 유전체 연구 및 치료 개발 분야에서의 유망한 응용 가능성을 제공합니다. 향후 다양한 조합 요법 및 유전자 조작 연구에서 NAIAD의 활용이 기대됩니다.



### Compression of Higher Order Ambisonics with Multichannel RVQGAN (https://arxiv.org/abs/2411.12008)
- **What's New**: 새로운 논문에서는 RVQGAN 신경 코딩 방법에 대한 다채널 확장을 제안하고, 이는 3차원 Ambisonics 오디오의 데이터 주도 압축에 사용됩니다. 생성기(generator)와 판별기(discriminator) 모델의 입력 및 출력 레이어가 다중(16개) 채널을 수용하도록 수정되었지만, 모델 비트레이트는 증가하지 않습니다. 또한 몰입형 오디오 재생에서 공간 인식을 고려하는 손실 함수(loss function)를 제안하였으며, 단일 채널 모델에서의 전이 학습(transfer learning)도 포함됩니다.

- **Technical Details**: 이 논문은 RVQGAN 오디오 코딩 방법에 적합한 다채널 모델 구조의 효율적인 확장을 소개합니다. 모델의 첫 번째와 마지막 레이어에서 채널 수를 3차원 Ambisonics 콘텐츠에 맞춰 16으로 늘렸습니다. 이를 통해 모델의 압축 효율성을 유지하면서도 각 채널은 고유한 커널(kernel)을 통해 최적화 및 처리됩니다. DAC(디지털-아날로그 변환기)를 통한 다채널 신호 변환으로 각 Ambisonics 채널은 최적화된 탈출 필터링하여 출력 신호를 생성합니다.

- **Performance Highlights**: 청취 테스트 결과 7.1.4 몰입형 재생이 수행되었으며, 제안된 확장은 16 kbit/s에서 효과적인 압축 성능을 보였습니다. 이는 Ambisonics 콘텐츠의 장면 기반 코딩에 적합하다는 것을 시사합니다. 압축과 재생의 지연 문제에도 불구하고 높은 품질의 오디오를 제공하는 것을 목표로 하고 있으며, 실제 응용 전에서도 효율성을 보여줍니다.



### Understanding Chain-of-Thought in LLMs through Information Theory (https://arxiv.org/abs/2411.11984)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)에서 체인 오브 생각(Chain-of-Thought, CoT) 추론을 공식화하는 새로운 정보를 제공합니다. 기존의 CoT 평가 방법들이 주석 데이터에 의존하던 반면, 저자들은 정보 수치 이론(information theory)을 기반으로 한 접근법을 제안합니다. 이 방법은 각 추론 단계에서 '정보 이득(information gain)'을 측정하여, LLM에서의 실패 유형을 식별할 수 있게 합니다.

- **Technical Details**: 제안된 프레임워크는 LLM 생성 접근법을 정보 이론의 관점에서 기술합니다. 먼저, 초기 상태와 작업을 정의하고 이를 통해 업데이트된 상태를 설명합니다. 이 프레임워크는 각 추론 단계에서 정보 이득을 정량화하며, 이는 적절한 정보가 최종 결과 예측에 기여해야 한다는 인식에 기반합니다. 이를 통해 주석 데이터 없이도 각 하위 작업의 성능을 평가할 수 있는 알고리즘을 제시합니다.

- **Performance Highlights**: 제안된 방법은 Toy 데이터 및 GSM-8K 데이터 세트를 통해 폭넓은 실험을 거쳐 효과성을 입증했습니다. 이메일 내의 진행 과정에서 기존의 결과 기반 방법들에 비해 정확한 모델 성능을 제공하여, CoT 추론의 실패 모드를 효과적으로 식별합니다. 최종적으로, 이 연구는 LLM의 성능 평가 방식에서 중요한 변화를 예고하며, 연구자들에게 더 나은 인사이트를 제공합니다.



### Calibrated and Efficient Sampling-Free Confidence Estimation for LiDAR Scene Semantic Segmentation (https://arxiv.org/abs/2411.11935)
- **What's New**: 이번 연구는 LiDAR 데이터의 안전 비판적 응용에서 신뢰할 수 있는 예측을 위한 샘플링 없는 방법을 제안합니다. 기존의 샘플링 기반 방법보다 추론 시간을 대폭 단축하면서도 정확한 신뢰 값을 유지할 수 있습니다. 우리의 접근 방식은 Adaptive Calibration Error (ACE) 메트릭을 통해 잘 보정된 신뢰 값을 달성하며, 실제 분류 정확도와 일관되도록 설계되었습니다.

- **Technical Details**: 제안한 방법은 aleatoric uncertainty(예측 불확실성)와 epistemic uncertainty(모델 불확실성)를 효과적으로 캡쳐하여 LiDAR 장면의 의미적 분할을 다룹니다. 이를 위해 각 요소 클래스에 대한 예측 Gaussian 분포를 생성하며, 가장 높은 평균 값을 가지는 클래스의 신뢰 값을 결정하는 과정을 포함합니다. 이 과정에서 깊은 앙상블 방식을 사용하여 여러 모델로부터 신뢰도 평가를 집계하여 더 강력한 신뢰 예측을 가능하게 합니다.

- **Performance Highlights**: 제안된 샘플링 없는 접근 방식은 기존의 신뢰 추정 방법에 비해 추론 시간을 현저히 줄이며, 잘 보정된 신뢰 값을 제공합니다. 성능 평가에서 우리의 방법은 과신이 아닌 과소신뢰를 생성하여, 안전 비판적 의사결정에서 유리한 결과를 보여줍니다. 이로써 LiDAR 장면의 의미적 분할 작업에서 우수한 강인성과 신뢰성을 달성했습니다.



### Phenome-wide causal proteomics enhance systemic lupus erythematosus flare prediction: A study in Asian populations (https://arxiv.org/abs/2411.11915)
- **What's New**: 본 연구는 아시아 SLE 환자들을 위한 새로운 단백질 기반의 위험 예측 모델을 개발하였으며, 개인별 질환 관리와 조기 개입을 향상시키기 위한 목적을 가지고 있습니다. 기존의 예측 모델들이 임상적 관찰 및 유전적 표지에 의존하는 반면, 본 연구는 다중 오믹스 접근법을 적용하여 단백질 데이터와 임상 정보를 통합하여 더 동적인 바이오마커를 식별합니다.

- **Technical Details**: 연구는 2020년 8월부터 2023년 1월까지 139명의 SLE 환자를 대상으로 진행되었으며, 침습성 및 비침습성 검사를 포함한 데이터를 수집하였습니다. 기초 혈장 샘플은 데이터 독립 수집(data-independent acquisition, DIA) 단백질 분석을 통해 분석되었고, 페노타입 전반적 멘델리안 랜덤화(phenome-wide Mendelian randomization, PheWAS)를 통해 단백질과 임상 예측자 간의 인과 관계를 평가했습니다.

- **Performance Highlights**: 제공된 결과에 따르면, 다섯 가지 단백질(SAA1, B4GALT5, GIT2, NAA15, RPIA)이 SLEDAI-2K 점수 및 1년 플레어 위험과 유의미한 관련이 있음을 보여주었고, 통합 모델이 0.769의 AUC를 기록하여 개별 모델보다 높은 예측 정확도를 달성했습니다. SAA1은 빠른 플레어 판별을 위한 우선 바이오마커로 강조되었습니다.



### F$^3$OCUS -- Federated Finetuning of Vision-Language Foundation Models with Optimal Client Layer Updating Strategy via Multi-objective Meta-Heuristics (https://arxiv.org/abs/2411.11912)
- **What's New**: 이번 연구에서는 리소스가 제한된 클라이언트 장치에서 Vision-Language Models(VLMs)를 효과적으로 훈련하기 위한 Parameter-Efficient Fine-Tuning(PEFT) 전략을 제안합니다. 특히, 클라이언트별로 가장 중요한 VLM 레이어를 선택하는 	extit{client-specific layer importance score}와 클라이언트 간 다양한 레이어 선택을 장려하는 	extit{inter-client layer diversity score}의 두 가지 요소의 영향을 밝혀냅니다. 이를 통해 개인화된 모델 훈련을 지원하는 새로운 프레임워크인 F3OCUS를 도입하였습니다.

- **Technical Details**: F3OCUS는 레이어 선택을 개선하기 위해 클라이언트의 자원 제약을 고려하면서 지역적 및 전역적 FL 특성을 모두 반영하는 두 단계의 전략을 적용합니다. 첫 번째 단계에서 클라이언트 레벨 전략을 통해 LNTK(Neural Tangent Kernel)의 주 고유값을 기반으로 레이어 중요도를 정의하고, 두 번째 단계에서 서버 레벨 전략을 통해 클라이언트별 중요도를 극대화하고 레이어 선택의 분산을 최소화하여 균일한 레이어 참여를 촉진합니다. 연구에서 제안된 방법은 58개의 의료 이미지 데이터셋을 포함한 6가지 Vision-Language FL 작업 설정에서 10,000건 이상의 클라이언트 실험으로 그 효과를 입증했습니다.

- **Performance Highlights**: F3OCUS의 실험 결과, 다양한 FL 환경에서 여러 VLM에 대한 선택적 레이어 튜닝이 효과적으로 이루어짐을 확인했습니다. 본 연구는 인간의 판단력 저하 및 효율성을 가진 클라이언트들에게 적합한 다채롭고 동적인 레이어 선택 솔루션을 제공함으로써 빠른 수렴을 촉진합니다. 데이터, 모달리티, 작업 및 장치 이질성을 고려한 더 많은 제약을 반영하여 클라이언트 설정을 평가한 결과, 이전의 연구보다 향상된 성능을 보여주었습니다.



### HeartBERT: A Self-Supervised ECG Embedding Model for Efficient and Effective Medical Signal Analysis (https://arxiv.org/abs/2411.11896)
Comments:
          First version, 24 pages, 8 Figures, 7 Tables

- **What's New**: HeartBert 모델은 레이블이 붙은 데이터의 필요성을 줄이고 계산 자원을 최소화하면서 ECG (Electrocardiogram) 신호 분석의 성능을 향상시키는 것을 목표로 합니다. 자연어 처리에서의 BERT (Bidirectional Encoder Representations from Transformers)에서 영감을 받아, RoBERTa 아키텍처를 기반으로 구축된 HeartBert 모델은 의료 분야에서 ECG 기반 프로젝트에 맞춘 정교한 임베딩을 생성합니다.

- **Technical Details**: 이 모델은 자기지도 학습 (self-supervised learning) 접근 방식을 통합하여 구성되었습니다. HeartBert는 수면 단계 탐지 (sleep stage detection)와 심박수 분류 (heartbeat classification)라는 두 가지 주요 다운스트림 작업을 수행하도록 설계되었습니다. 또한 Bidirectional LSTM 헤드를 활용하여 복잡한 문제를 해결할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실제 실험 결과는 HeartBert 모델이 더 작은 훈련 데이터셋과 감소된 학습 매개변수에서도 뛰어난 성능을 발휘함을 보여줍니다. 경쟁 모델과 비교했을 때 효과적인 성능을 달성하여 우수성과 진전을 입증합니다. 또한 관련 코드와 데이터는 공개적으로 제공되어, 연구자들이 이를 활용할 수 있습니다.



### How Many Data are Enough? Optimization of Data Collection for Artifact Detection in EEG Recordings (https://arxiv.org/abs/2411.11886)
- **What's New**: 이 연구는 생물학적 데이터 수집에서 효과적인 데이터 활용의 필요성을 해결하기 위해 딥러닝 기반의 아티팩트 검출을 이용한 데이터 지향적 수집 설계 최적화 절차를 제안합니다. 기존의 직관적 데이터 수집 방식에서 벗어나, 아티팩트 유형과 그 수량에 대한 명확한 정당성을 제공하며, 향후 EEG 및 EMG 연구에서 보다 효율적이고 경제적인 데이터 수집을 유도하는 것을 목표로 합니다.

- **Technical Details**: 우리는 아티팩트 에포크(artifact epochs)와 비아티팩트 에포크(non-artifact epochs) 간 이진 분류(binary classification)를 활용하여 세 가지 다른 딥러닝 아키텍처를 적용합니다. 이 연구는 데이터 수집 노력을 최소화하면서 청소 효율성을 유지하는 것을 목표로 합니다. 실험에서는 정적 수축(isometric contraction) 및 연속 움직임(continuous movement)과 같은 여러 종류의 아티팩트를 생성하는 작업을 포함하여 EEG 데이터 수집을 진행합니다.

- **Performance Highlights**: 연구를 통해 아티팩트 작업 수를 열두 개에서 세 개로 줄일 수 있었으며, 정적 수축 작업의 반복 횟수를 열 번에서 세 번 또는 심지어 한 번으로 감소시켰습니다. 아티팩트 청소 모델 성능을 유지하는 동시에 데이터 수집 비용을 최소화하는 데 중점을 두었으며, 효과적인 아티팩트 에포크 검출을 통해 후속 아티팩트 제거를 위한 충분한 정보를 제공하고 있습니다.



### CSP-Net: Common Spatial Pattern Empowered Neural Networks for EEG-Based Motor Imagery Classification (https://arxiv.org/abs/2411.11879)
- **What's New**: 이 논문은 Electroencephalogram (EEG) 기반의 motor imagery (MI) 분류를 위한 두 가지 CSP-empowered neural network(CSP-Nets)를 제안합니다. CSP는 EEG 신호의 스칼프에서 이루어지는 다양한 MI 작업 중 에너지 분포를 활용하는 전통적인 머신 러닝 기술로, 이 논문에서는 이를 convolutional neural networks (CNN)와 통합하여 MI 분류 성능을 향상시키고자 합니다. 이 접근 방식은 특히 훈련 샘플이 적을 때 효과적입니다.

- **Technical Details**: CSP-Net-1은 CNN 앞에 CSP 레이어를 추가하여 입력의 구별 가능성을 높이고, CSP-Net-2는 CNN의 합성곱 레이어를 CSP 레이어로 대체합니다. 이 두 CSP-Nets는 훈련 데이터에서 설계된 CSP 필터로 CSP 레이어 파라미터를 초기화하며, 훈련 중에 이들을 고정하거나 gradient descent를 통해 최적화할 수 있도록 설계되었습니다. 이를 통해 EEG 신호에서 중요한 특징을 효과적으로 캡처할 수 있습니다.

- **Performance Highlights**: 실험 결과, CSP-Nets는 서로 다른 MI 데이터셋에서 CNN 백본보다 일관되게 우수한 성능을 나타냈습니다. 특히, 본 논문에서 제안하는 네트워크는 적은 수의 훈련 샘플이 있는 경우에도 효과적으로 작용하여, EEG 기반 뇌-컴퓨터 인터페이스에서의 전통적인 기계 학습과 데이터 기반 심층 학습의 통합의 장점을 보여주고 있습니다.



### MultiBalance: Multi-Objective Gradient Balancing in Industrial-Scale Multi-Task Recommendation System (https://arxiv.org/abs/2411.11871)
- **What's New**: 이 논문은 멀티태스크 학습(multi-task learning)에서 발생할 수 있는 부정적 전이(negative transfer) 문제를 해결하기 위해 MultiBalance라는 새로운 기법을 제안합니다. MultiBalance는 작업별(퍼태스크, per-task) 경량의 그래디언트를 균형 있게 조정하여 여러 작업 간의 최적화를 도와주며, 그리드 서치(grid search)나 수동 탐색(manual exploration)의 비용을 절감합니다. 이 기법은 Meta의 대규모 광고 및 피드 멀티태스크 추천 시스템에서 강력한 성과를 보였으며, 이는 대규모 산업 시스템에서의 실제적 응용 가능성을 보여줍니다.

- **Technical Details**: MultiBalance는 각 작업의 그래디언트를 공유된 피처 표현(shared feature representation)을 기준으로 균형 있게 조정합니다. 기존의 방법과는 다르게, MultiBalance는 작업별 그래디언트가 아닌 공유된 피처 표현의 출력으로 수렴하는 것을 목표로 합니다. 또한 훈련 과정에서 미니 배치 스토캐스틱 그래디언트(mini-batch stochastic gradients)의 크기 이동 평균을 유지하여 대규모 모델 학습의 안정성을 높입니다.

- **Performance Highlights**: MultiBalance는 QPS(Queries Per Second) 성능 손실 없이 정상훈련비용(neutral training cost)에서 0.738%의 개선을 보여주었습니다. 이전의 방법들과 비교했을 때, MultiBalance는 70%~80%의 QPS 감소 없이도 더 높은 효율성을 달성하였습니다. 이러한 결과는 대규모 산업 시스템에 적용하기에 매우 적합한 ROI(투자수익률) 솔루션임을 입증합니다.



### A Multi-Modal Unsupervised Machine Learning Approach for Biomedical Signal Processing in CPR (https://arxiv.org/abs/2411.11869)
- **What's New**: 이 논문은 심폐소생술(CPR) 신호의 잡음을 제거하기 위한 새로운 비지도 머신러닝(ML) 방법론을 제안합니다. 기존의 신호 처리 기술의 한계를 뛰어넘어, 다양한 신호 출처를 활용하여 denoising(잡음 제거) 프로세스를 개선하는 다중 모달리티(multi-modality) 프레임워크를 도입합니다. 이 방법은 신호 간 상관관계를 유지하면서도 기존 방법들을 초월하는 성능을 보여주어, 실시간 응용에서 뛰어난 효과를 발휘합니다.

- **Technical Details**: 해당 프레임워크는 CPR 신호의 다양성과 시간 민감성을 고려하여 설계되었습니다. 비지도 ML을 통해 라벨이 없는 데이터에서 신호의 잡음을 효과적으로 제거하고 신호 왜곡 없이 중요한 신호 특징을 보존하는 특징이 있습니다. 논문은 또한 노이즈 특성을 각 신호 별로 개별적으로 처리하여 신호 해석의 정확성을 향상시키기 위한 방법론이 포함되어 있습니다.

- **Performance Highlights**: 제안하는 방법론은 기존의 ML 및 필터 방법과 비교하여 신호 대 잡음비(SNR) 및 최대 신호 대 잡음비(PSNR) 면에서 우수한 성능을 보여주었습니다. 특히, 각 신호를 전용 모델을 통해 처리한 후 조합할 때도 신호 데이터 간의 상관관계를 높은 수준으로 유지하는 것을 확인하였습니다. 이 접근법은 CPR뿐만 아니라 다른 임상 환경에서도 생물 의학 신호 처리의 잠재력을 향상시킬 수 있는 가능성을 가지고 있습니다.



### Longitudinal Wrist PPG Analysis for Reliable Hypertension Risk Screening Using Deep Learning (https://arxiv.org/abs/2411.11863)
Comments:
          blood pressure, hypertension, cuffless, photoplethysmography, deep learning

- **What's New**: 본 연구는 심혈관 질환의 주요 위험 요소로 알려진 고혈압(hypertension)의 효율적인 모니터링을 위해 스마트워치에서 수집된 PPG(photoplethysmography) 데이터를 활용하고 있습니다. 전통적인 모니터링 방법의 단점을 보완하며, 손으로 제작한 PPG 특성 없이도 깊이 있는 학습(deep learning) 모델을 통하여 결과를 도출해냅니다. 448명의 연구 참여자와 5회 교차 검증을 통해 개발된 이 모델은 실제로 90명의 참가자 데이터를 사용하여 테스트되었습니다.

- **Technical Details**: 이 연구는 PPG 신호의 전처리(preprocessing) 및 ResNet 기반의 특징 학습(feature learning)과 분류(classification) 모델을 포함하는 end-to-end 딥러닝 프레임워크를 활용합니다. PPG 신호는 잡음 감소와 이상치 제거 등의 단계를 거치며, 최종적으로는 ResNet 아키텍처를 통해 고혈압의 확률을 예측합니다. 모델은 0.124M 파라미터를 사용하는 컴팩트한 형태로 설계되어, 성능 면에서 전통적인 머신러닝 방법을 능가합니다.

- **Performance Highlights**: 연구 결과, 리얼월드 환경에서 수집된 데이터를 기반으로 한 모델의 실제 성능은 매우 우수하였습니다. 특히, ResNet 모델은 건강한 사례와 비정상 사례를 구분하는 데 있어 뛰어난 효과를 보였으며, 68,000 이상의 스폿 체크 인스턴스로 학습을 진행했습니다. 본 연구는 향후 고혈압 탐지 및 모니터링의 혁신적인 방향을 제시하는 동시에, 다양한 인구 집단에서도 높은 정확도를 유지할 수 있는 가능성을 보여줍니다.



### Machine Learning Assisted Postural Movement Recognition using Photoplethysmography(PPG) (https://arxiv.org/abs/2411.11862)
- **What's New**: 이번 연구는 노인 인구의 증가에 따라 낙상 감지 및 예방 기술의 필요성을 강조합니다. Photoplethysmography(PPG) 데이터만을 이용해 자세 변화를 인식하는 첫 번째 기계 학습 기법을 사용한 것에 주목해야 합니다. 이 연구는 11명의 참가자를 대상으로 자세 움직임과 PPG 신호의 변화를 조사하여 낙상 리스크 평가에 중요한 통찰을 제공합니다.

- **Technical Details**: PPG는 조직의 미세혈관 내 맥박 혈액량을 측정하기 위한 광학적 방법으로, LED를 이용해 혈액에 반사되는 빛을 측정합니다. 특정 자세 변경 시, 혈액 부피의 급격한 변화를 감지하여 기계 학습 알고리즘이 혈액 흐름 변화를 분류합니다. 이를 통해 낙상 위험을 사전에 감지하도록 돕는 휴대용 센서 개발이 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 인공지능 신경망(Artificial Neural Network, ANN)이 가장 높은 분류 성능을 나타내었으며, 시험 정확도는 85.2%, F1 점수는 78%에 달했습니다. 이 성과는 노인 낙상 예방에 대한 기술적 가능성을 넓히는 데 기여할 것입니다.



### LUTMUL: Exceed Conventional FPGA Roofline Limit by LUT-based Efficient Multiplication for Neural Network Inferenc (https://arxiv.org/abs/2411.11852)
Comments:
          Accepted by ASPDAC 2025

- **What's New**: 이번 연구에서는 FPGA 기반의 신경망 가속기를 위한 LUTMUL을 소개합니다. LUTMUL은 전통적으로 DSP 블록이 수행하던 곱셈을 룩업 테이블(LUT)을 활용하여 수행하여, 계산 효율성을 크게 개선했습니다. LUT의 수가 DSP보다 100배 더 많기 때문에, FPGA의 성능을 극대화하는 데 기여합니다. 실험 결과는 이 설계가 FPGA 가속기 중 가장 빠른 추론 속도를 기록하였음을 보여줍니다.

- **Technical Details**: LUTMUL은 새로운 재구성 데이터 흐름 아키텍처를 통해 설계되었으며, 이는 메모리 접근 시간을 최소화하고 각 CNN 층에 대해 최적화된 아키텍처를 제공합니다. LUT 자원을 활용하여, 기존 DSP 기반 가속기의 피크 성능을 초과할 수 있도록 설계되었습니다. 또한, LUTMUL은 quantized 신경망 가중치를 LUT에 포함시켜 곱셈을 효율적으로 수행합니다. 이를 통해 단일 4비트 곱셈을 위해 단 2개의 LUT를 필요로 합니다.

- **Performance Highlights**: LUTMUL 설계를 통해, FPGA는 초당 1627개의 이미지를 처리할 수 있는 최고의 추론 속도를 달성하였으며, ImageNet 데이터셋에서 top-1 정확도 70.95%를 유지하고 있습니다. 이 결과는 LUT 기반 곱셈 기법이 기존 DSP 기반 설계보다 뛰어난 성능을 제공할 수 있음을 입증합니다. LUTMUL을 사용하면 FPGA의 재구성 가능성을 최대한 활용하고, 깊은 학습 작업에서 성능 향상을 기대할 수 있습니다.



### Robust Graph Neural Networks for Stability Analysis in Dynamic Networks (https://arxiv.org/abs/2411.11848)
Comments:
          It was accepted by the 3rd International Conference on Cloud Computing Big Data Application and Software Engineering

- **What's New**: 이번 연구는 그래프 신경망(Grant Neural Network, GNN) 알고리즘을 기반으로 한 경제적 위험 식별 알고리즘을 제안합니다. GNN은 금융 네트워크 내의 거래 행동, 금융 기관 및 개인의 상호 관계를 그래프 구조로 매핑하여 복잡한 금융 시장에서 경제적 위험을 예방하는 데 도움을 줄 수 있는 혁신적인 기술입니다. 이 알고리즘은 금융 기관과 규제자들에게 더욱 지능적인 기술 도구를 제공하여, 금융 시장의 안전성과 안정성을 유지하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 기존의 위험 식별 방법은 금융 네트워크 내의 다층적이고 동적으로 변화하는 복잡한 관계를 처리하는 데 한계가 있습니다. GNN을 이용하여, 금융 데이터를 내재 표현 학습(embedded representation learning)으로 잠재적인 패턴 및 이상 신호를 효과적으로 포착할 수 있습니다. 이를 통해 금융 기관은 복잡한 거래 네트워크에서 유용한 정보를 추출하고 시스템적 위험을 초래할 수 있는 숨겨진 위험 요소를 신속하게 식별할 수 있습니다.

- **Performance Highlights**: 이 연구는 GNN 기술을 통해 경제적 위험 식별의 효율성을 향상시키고, 금융 시스템의 위험 저항성을 강화할 것으로 기대됩니다. 금융 기관들은 이 혁신적인 도구를 활용함으로써 의사 결정 과정을 최적화하고 위험 경고의 정확성을 높일 수 있습니다. 이러한 접근 방식은 강력한 글로벌 금융 시스템을 구축하기 위한 기반이 될 것입니다.




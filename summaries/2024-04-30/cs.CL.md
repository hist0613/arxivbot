### Holmes: Benchmark the Linguistic Competence of Language Models (https://arxiv.org/abs/2404.18923)
- **What's New**: 이 논문에서는 언어 모델(Language Models, LMs)의 언어적 역량을 평가하는 새로운 벤치마크인 Holmes를 소개합니다. 언어적 현상을 이해하는 능력을 평가하기 위해 기존의 프롬프팅(prompting) 기반 평가와 달리, Holmes는 언어 모델의 내부 표현을 사용하여 classifier-based probing을 통해 언어 역량을 평가합니다. 이 방법을 통해, 텍스트 지시 사항을 따르는 것과 같은 다른 인지 능력으로부터 특정 현상(예: 단어의 품사)을 분리하여 평가합니다.

- **Technical Details**: Holmes 벤치마크는 250개 이상의 탐색 연구를 리뷰하고 200개가 넘는 데이터셋을 통해 문법, 형태학, 의미론, 추론 및 담화 현상을 평가합니다. 이를 통해 언어 모델이 기본 언어 구조와 개념을 얼마나 잘 이해하고 있는지를 평가할 수 있습니다. 또한, 이 논문에서는 계산 부하를 줄이면서도 정확도를 유지하는 간소화된 버전인 FlashHolmes를 제안합니다.

- **Performance Highlights**: 50개 이상의 언어 모델을 분석한 결과, 언어 역량이 모델의 크기(Model size)와 관련이 있다는 기존의 추세를 확인할 수 있었습니다. 그러나 놀랍게도 모델 구조(Model architecture)와 지시 튜닝(Instruction tuning)이 특히 형태학과 문법에서 성능에 큰 영향을 미치는 것으로 나타났습니다.



### Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting (https://arxiv.org/abs/2404.18911)
- **What's New**: 이 연구에서는 큰 언어 모델의 추론을 가속화하고자 하는 문제를 해결하기 위해 새로운 자체 추론 프레임워크 'Kangaroo'를 제안했습니다. 이 프레임워크는 이른 종료(Early Exiting)에서 영감을 받아 고정된 얕은 하위 네트워크를 자체 초안 모델로 사용하고 나머지 레이어를 더 큰 대상 모델로 활용합니다. Kangaroo는 높은 토큰 승낙률을 유지하면서 추론 지연을 최소화하는 것을 목표로 합니다.

- **Technical Details**: Kangaroo는 큰 언어 모델(LLM)의 고정된 얕은 하위 네트워크에 경량 효율적인 어댑터 모듈을 추가하여 대상 LLM과의 표현 능력 간의 격차를 메우는 방식으로 작동합니다. 이 어댑터 네트워크는 멀티-헤드 어텐션(Multi-Head Attention)과 두개의 정규화 레이어로 구성됩니다. 또한, Kangaroo는 어려운 토큰에 대한 불필요한 계산을 피하기 위해 추가적인 이른 종료 메커니즘을 도입합니다.

- **Performance Highlights**: Spec-Bench에서의 실험 결과에 따르면, Kangaroo는 Medusa-1을 상당히 능가하며, 1.68배의 속도 향상(Speedup)을 달성했습니다. 특히, Kangaroo는 Medusa-1에 비해 88.7% 적은 추가 파라미터(67M 대비 591M)만을 사용하여 이 성과를 이루었습니다. 이는 기존의 솔루션에 비해 매우 비용 효율적이라 할 수 있습니다.



### Spivavtor: An Instruction Tuned Ukrainian Text Editing Mod (https://arxiv.org/abs/2404.18880)
Comments: Accepted to UNLP Workshop 2024

- **What's New**: 우크라이나어 텍스트 편집을 위한 데이터세트 및 모델인 Spivavtor를 소개합니다. 이 연구에서는 기존의 영어 전용 모델인 CoEdIT를 우크라이나어에 맞춰 수정한 Spivavtor를 개발하였습니다. Spivavtor는 우크라이나어로 작성된 지시에 따라 텍스트 편집 작업을 수행합니다.

- **Technical Details**: Spivavtor-Instruct 데이터세트 및 Spivavtor 모델에 대해 자세히 설명합니다. 이 모델들은 Grammatical Error Correction (GEC; 문법 오류 수정), Text Simplification (텍스트 단순화), Coherence (일관성), Paraphrasing (표현 변환)과 같은 다양한 우크라이나어 텍스트 편집 작업에 사용됩니다.

- **Performance Highlights**: 다양한 테스트에서 Spivavtor는 모든 작업에서 뛰어난 성능을 보였습니다. 연구팀은 이 모델과 데이터를 공개하여 향후 연구를 촉진하기 위해 커뮤니티에 제공합니다.



### More RLHF, More Trust? On The Impact of Human Preference Alignment On  Language Model Trustworthiness (https://arxiv.org/abs/2404.18870)
- **What's New**: 이 연구는 인간의 가치와 일치시키기 위해 선호 학습 알고리즘, 특히 인간 피드백에서 강화 학습(Reinforcement Learning From Human Feedback, RLHF)을 사용하여 언어 모델을 미세 조정하는 과정에서 발생하는 신뢰성 문제를 다룹니다. RLHF의 세 가지 변형 - 감독 학습 조정(Supervised Finetuning, SFT), 근접 정책 최적화(Proximal Policy Optimization, PPO), 직접 선호 최적화(Direct Preference Optimization, DPO)에 대한 연구를 통해, 이러한 알고리즘들이 모델의 신뢰도 향상에 직접적인 영향을 미치지 않을 수 있음을 밝히고 있습니다.

- **Technical Details**: 연구에서는 세 가지 RLHF 방법을 사용하여 언어 모델을 사람들의 선호도와 일치시키려고 시도했습니다. SFT는 사전 훈련된 언어 모델에 인간 데모 데이터 셋을 이용한 감독 학습을 적용합니다. PPO는 보상 모델을 사용하여 언어 모델의 출력이 인간의 선호도를 반영하도록 최적화하며, DPO는 선호 데이터를 직접 사용하여 언어 모델 정책을 최적화합니다. 각각의 방법이 모델의 독성(toxicity), 고정관념(stereotypical bias), 기계 윤리(machine ethics), 진실성(truthfulness), 개인 정보 보호(privacy) 등 다양한 신뢰성 측정항목에서 어떻게 작용하는지 평가했습니다.

- **Performance Highlights**: Extensive empirical investigations demonstrate that LLMs’ performance in terms of trustworthiness varies significantly across different RLHF techniques and trustworthiness benchmarks. The result showcases the intricate interplay between model size, preference learning algorithm, and the specific trustworthiness verticals. 특히, 일반 목적의 데이터셋을 사용한 RLHF는 모델의 다양한 신뢰성 측면에 걸쳐 일관된 개선을 보장하지 않음을 발견했습니다. 이는 언어 모델의 신뢰성을 향상시키려는 시도에서 더 세밀한 접근법이 필요함을 시사합니다.



### Truth-value judgment in language models: belief directions are context  sensitiv (https://arxiv.org/abs/2404.18865)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 잠재 공간(Latent Spaces) 내에서 문장의 진리성을 예측하는 방향들을 살펴보고, 이러한 방향들을 추출하고 프로브(Probes)를 구축하는 다양한 방법들을 탐구합니다. 이 프로브들은 모델의 '지식(Knowledge)'이나 '신념(Beliefs)'을 파악하는 데 사용된다고 설명됩니다. 연구는 특히 문맥(Context)이 프로브에 미치는 영향에 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 대규모 언어 모델에서 문장이 앞서 나오는 관련 문장들에 조건적(Conditional)이라고 묘사될 수 있는 위치에 대한 프로브의 예측을 조사했습니다. 프로브의 반응성을 (부정된) 지지 및 모순되는 문장의 존재에 대해 정량화하고, 일관성 점수를 제공합니다. 또한, 가설의 위치가 같은 방향으로 영향을 받는지 여부를 조사하는 인과 개입(Causal Intervention) 실험을 수행했습니다.

- **Performance Highlights**: 테스트된 프로브들은 일반적으로 문맥에 민감(Context Sensitive)하게 반응하지만, 진리에 영향을 미치지 않아야 할 문맥들이 프로브의 출력에 여전히 영향을 끼치는 경우가 있습니다. 실험 결과에 따르면, 발생하는 오류의 유형은 해당 레이어(Layer), 모델의 유형(Type of Model), 그리고 데이터의 종류에 따라 달라집니다. 또한, 신념 방향(Belief Directions)이 문맥 정보를 통합하는 추론 과정에서의 (하나의) 인과 매개체(Causal Mediators)라는 것이 제안됩니다.



### A Comprehensive Rubric for Annotating Pathological Speech (https://arxiv.org/abs/2404.18851)
Comments: Submitted to LREC-Coling 2024

- **What's New**: 이 연구에서는 파닉스(phonetics), 유창성(fluency), 그리고 인토네이션(prosody)을 포함한 다양한 음성 품질 차원을 기반으로 하는 포괄적인 척도(rubric)를 도입했습니다. 이 척도는 다운 증후군을 가진 개인의 음성에서 오류를 식별하는 표준화된 기준을 설정하고자 합니다. 이로써 자동 평가 시스템의 개발을 가능하게 하기 위한 것입니다.

- **Technical Details**: 이 연구는 Prautocal 말뭉치(corpus)를 사용하였고, 평가 척도의 품질을 평가하기 위해 두 가지 실험을 수행했습니다. 음성학적 평가(phonetic evaluation)에서는 발음의 우수성(Goodness of Pronunciation, GoP) 메트릭을 사용하여 자동 분할 시스템을 활용하고, 전문 음성 치료사에 의한 평가와 결과를 상관시켰습니다. 유창성 평가에서는 깊은 학습 모델(wav2vec)을 사용해 오디오 특성을 추출하고 유창성 문제를 식별하기 위해 집중된 말뭉치로 훈련된 SVM 분류기(classifier)를 사용했습니다.

- **Performance Highlights**: 얻은 상관계수(correlation values)는 높지 않았으나 긍정적인 경향을 보였습니다. 유창성 평가에서는 탐지된 불유창성(disfluency)의 특정 유형에 따라 결과의 변동성이 있었음을 강조했습니다.



### It's Difficult to be Neutral -- Human and LLM-based Sentiment Annotation  of Patient Comments (https://arxiv.org/abs/2404.18832)
- **What's New**: 이 연구는 환자 설문조사에서 수집된 자유 텍스트 코멘트에 감정 주석을 추가하는 노력을 기술하고 있으며, 이는 보건 서비스 개선을 위해 환자의 의견을 집계하는 데 중요한 도구입니다. 또한 인간 주석자(Human annotators) 대신 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 가능한 대안을 평가합니다.

- **Technical Details**: 연구진은 노르웨이어로 사전 훈련된 두 가지 대규모 언어 모델을 사용하여 감정 분석을 수행하고, 다양한 프롬프트 구성과 맥락적 학습(In-context learning)을 실험했습니다. 이러한 모델을 활용함으로써 도메인 전문가가 필요한 자원 집약적인 주석 과정을 대체할 수 있는지를 평가하였습니다.

- **Performance Highlights**: 연구 결과, 제로샷(Zero-shot) 실행에서도 모델은 이진 감정 분석(Binary sentiment analysis)에 대해 기준선을 크게 상회하는 성능을 보였지만, 전체 데이터셋에서 인간 주석자의 성능에는 미치지 못했습니다.



### Benchmarking Benchmark Leakage in Large Language Models (https://arxiv.org/abs/2404.18824)
Comments: 30 pages; Homepage: this https URL

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 훈련 데이터 사용에 관한 문제점을 지적하며, 특히 벤치마크 데이터셋의 유출 문제가 강조되었습니다. 이 문제를 해결하기 위해, 연구팀은 Perplexity와 N-gram 정확도를 이용한 감지 파이프라인을 도입했으며, 이는 벤치마크에서 모델의 예측 정밀도를 측정하는 데 사용됩니다.

- **Technical Details**: 이 감지 파이프라인은 Perplexity와 N-gram 정확도라는 두 가지 간단하고 확장 가능한 메트릭스(metrics)를 사용하여, 특정 벤치마크에 대한 모델의 예측 정밀도를 평가합니다. 이를 통해 데이터 유출 가능성을 식별할 수 있습니다. 또한, 31개의 대규모 언어 모델들을 수학적 추론의 맥락에서 분석함으로써, 테스트 세트의 오용을 포함한 훈련의 심각한 사례들을 밝혔습니다.

- **Performance Highlights**: 연구 결과에 따르면, 많은 LLMs가 벤치마크 데이터의 유출로 인해 잠재적으로 불공정한 비교를 초래하는 것으로 나타났습니다. 이에 따라, 연구팀은 모델 문서화, 벤치마크 설정 및 향후 평가에 대한 여러 권장사항을 제시하였습니다. 특히 'Benchmark Transparency Card'를 제안하여 벤치마크 사용에 대한 명확한 문서화를 권장하고 있습니다, 이는 투명성을 증진시키고 LLMs의 건전한 발전을 촉진합니다.



### Unknown Script: Impact of Script on Cross-Lingual Transfer (https://arxiv.org/abs/2404.18810)
Comments: Paper accepted to NAACL Student Research Workshop (SRW) 2024

- **What's New**: 이 논문에서는 기존 연구에서 주로 다루지 않은 언어 모델의 기반 언어가 Cross-Lingual Transfer 성능에 미치는 영향을 분석합니다. 특히, 사전 훈련된 모델의 토크나이저(Tokenization)와 스크립트(script)가 하위 작업성능에 미치는 효과를 실험을 통해 조사하였습니다.

- **Technical Details**: 연구자들은 여러 모노링구얼(monolingual)과 멀티링구얼(multilingual) 사전 학습된 모델을 선택하여, 목표 언어와 그 언어의 토폴로지학적 관계를 고려하여 실험을 설계하였습니다. 여기서 Byte Piece Encoder를 사용하는 것이 pre-trained 모델 지식을 활용하는데 가장 효과적이었다는 것을 발견했습니다.

- **Performance Highlights**: 분석 결과, 토크나이저의 사용이 스크립트 공유, 언어 타이폴로지(typology) 일치 및 모델 사이즈보다 크로스-링구얼 트랜스퍼 성능에 더 큰 영향을 미치는 가장 강력한 요소로 밝혀졌습니다.



### Replacing Judges with Juries: Evaluating LLM Generations with a Panel of  Diverse Models (https://arxiv.org/abs/2404.18796)
- **What's New**: LLM (Large Language Models) 평가 방식에서의 중요한 변화가 제안되었습니다. 이전에는 주로 단일 크기의 큰 모델을 사용하여 다른 LLM의 출력 품질을 평가했지만, 여러 작은 모델을 활용하는 새로운 방식인 Panel of LLM evaluators (PoLL)가 제시되었습니다. 이 방법은 비용 효율적이며, 모델 간 편향성(bias)도 감소시키는 장점이 있습니다.

- **Technical Details**: PoLL 방식은 서로 다른 모델 가족으로 구성되어 있기 때문에, 단일 큰 모델을 사용하는 것에 비해 내부 모델 편향성(intra-model bias)이 적다는 장점이 있습니다. 연구에서는 세 가지 다른 judge 설정과 여섯 개의 다양한 데이터셋을 사용하여 PoLL의 효과를 검증하였고, 작은 여러 모델을 사용하는 PoLL이 단일 큰 모델보다 성능이 우수하다는 결과를 얻었습니다.

- **Performance Highlights**: PoLL 방식은 단일 대형 모델을 사용하는 전통적 방식보다 7배 이상 비용 효율적이라는 점에서 높은 성능을 보였으며, 평가 결과의 다양성과 정확성이 향상되었습니다. 즉, 여러 작은 모델을 활용한 다층 평가는 경제적 이점 뿐만 아니라 공정성 면에서도 개선을 이루었습니다.



### Where on Earth Do Users Say They Are?: Geo-Entity Linking for Noisy  Multilingual User Inpu (https://arxiv.org/abs/2404.18784)
Comments: NLP+CSS workshop at NAACL 2024

- **What's New**: 이 연구에서는 소셜 미디어 데이터의 다양하고 복잡한 특성을 고려하여 새로운 접근 방식으로 지오 엔티티 링킹(geo-entity linking) 작업을 탐구합니다. 특히, 노이즈가 많고 다국어 환경에서 지리적 위치를 식별하는 데 중점을 두고, 기존의 규칙 기반(rule-based) 또는 대규모 언어 모델(LLM-based)에 의존하는 방식에서 벗어나 평균 임베딩(averaged embeddings)을 사용해 위치를 표현하는 새로운 방법을 제안합니다.

- **Technical Details**: 이 연구에서 제안된 방법은 사용자가 입력한 위치 이름으로부터 레이블이 지정된 평균 임베딩을 생성하여 실제 세계 위치를 표현합니다. 또한, 해석 가능한 신뢰 점수(interpretable confidence score)를 통해 선택적 예측을 할 수 있습니다. 이는 다양한 지리적 세분성(geographic granularity)에서의 평가에 있어 발전과 문제점을 논의하는 데 중요합니다.

- **Performance Highlights**: 이 방법은 전 세계적이고 다국어의 소셜 미디어 데이터셋에서 지오 엔티티 링킹의 효율성을 향상시키는 데 성공적이었습니다. 연구 결과에 따르면, 이 새로운 접근법은 기존 방법들에 비해 더 정확하고 신뢰성 있는 위치 링킹 결과를 제공함으로써, 소셜 미디어 설정에서의 지오 엔티티 링킹 문제를 효과적으로 해결할 수 있습니다.



### Towards A Structured Overview of Use Cases for Natural Language  Processing in the Legal Domain: A German Perspectiv (https://arxiv.org/abs/2404.18759)
Comments: 10 pages, 6 tables, 30th Americas Conference on Information Systems (AMCIS 2024)

- **What's New**: 최근 법률 기술(Legal Tech) 분야가 발전하면서 자연 언어 처리(Natural Language Processing, NLP)와 법학 분야가 결합하여 법적 과정을 디지털화하는 작업이 활발해지고 있습니다. 이 연구에서는 독일의 법률 실무자들의 의견을 통해 NLP 문헌에 기반한 법률 기술 사용 사례에 대한 구조적 개요를 구축하고자 합니다.

- **Technical Details**: 이 연구는 체계적 문헌 고찰(Systematic Literature Review)을 바탕으로 NLP 기술들이 법률 분야에 어떻게 적용될 수 있는지 살펴봄으로써, 7개의 NLP 기술 카테고리와 22가지 법률 사용 사례를 대조 연구하였습니다. 이를 통해 주요 법률 사용 사례가 이론적으로 어떻게 실제적인 문제들을 해결할 수 있는지를 파악합니다.

- **Performance Highlights**: 이 연구는 법률 분야에서 NLP 기술의 적용 가능성을 탐구하면서, 동시에 15개의 윤리적, 법적, 사회적 측면(Ethical, Legal, and Social Aspects, ELSA)을 식별하여 디지털 변환(Digital Transformation)이 법률 분야에 있어서 가질 수 있는 유의사항을 조명합니다.



### Towards Dog Bark Decoding: Leveraging Human Speech Processing for  Automated Bark Classification (https://arxiv.org/abs/2404.18739)
Comments: to be published in LREC-COLING 2024

- **What's New**: 새로운 연구에서는 사람의 음성 인식 모델을 활용하여 강아지의 짖음을 분류하는 것을 탐구하였습니다. 이 연구는 강아지 인식(dog recognition), 품종 식별(breed identification), 성별 분류(gender classification), 그리고 맥락적 상황 인식(context grounding)과 같은 여러 작업에 대해 논의합니다. 사람 음성에 사전 학습된 자기지도 학습 모델을 사용함으로써, 단순한 분류 기준을 훨씬 뛰어넘는 성능 개선을 보였습니다.

- **Technical Details**: 연구팀은 사람의 음성 데이터에 사전 훈련된 자기지도 학습(self-supervised learning) 모델을 사용하여 강아지의 짖음 소리 데이터에 적용하였습니다. 이 모델들은 인간의 음성 처리를 위해 개발되었고, 이를 통해 동물의 음성을 해석하는 새로운 방법을 제시하였습니다. 이 연구는 강아지 짖음 분류에 대한 데이터셋을 제작하고, 해당 데이터셋을 사용하여 여러 실험을 수행했습니다.

- **Performance Highlights**: 사람의 음성에 대한 사전 훈련 모델을 사용함으로써 동물의 소리 분류 작업에서도 뛰어난 성능을 나타냈습니다. 이러한 방법은 강아지의 종류, 성별 및 특정 상황에서 짖는 소리를 구분하는데 상당한 성능 향상을 보여주었습니다.



### The Constant in HATE: Analyzing Toxicity in Reddit across Topics and  Languages (https://arxiv.org/abs/2404.18726)
Comments: Accepted to TRAC 2024

- **What's New**: 이 연구는 Reddit 대화에서 언어의 독성이 논의 주제에 따라 어떻게 달라지는지를 탐구합니다. 1.5백만 개의 코멘트 스레드와 481개 커뮤니티를 포함한 여섯 가지 언어에 걸친 분석을 통해, 특정 토픽과 언어에 따라 독성의 증가 패턴을 관찰했습니다.

- **Technical Details**: 다가언어 및 주제 분석을 위해, 연구팀은 Reddit의 80가지 주제를 커버하는 다양한 커뮤니티에서 데이터를 수집했습니다. NLP (Natural Language Processing) 도구와 Google’s Perspective API 등을 활용하여 각 코멘트의 독성을 분석하고, 이 데이터를 기반으로 독성이 발생하는 맥락을 이해하기 위해 다양한 접근 방식을 비교했습니다.

- **Performance Highlights**: 분석 결과, 일부 주제들은 대부분의 타겟 언어에서 높은 독성을 보였으며 (예: 정치, 스포츠), 언어에 따라 주제의 독성에 차이가 있음을 관찰했습니다. 해당 결과는 소셜 미디어 모더레이터들이 더 효과적인 콘텐츠 모더레이션을 위해 활용될 수 있습니다. 또한, 자동 콘텐츠 모더레이션을 위한 모델 훈련 시, 주제와 언어를 고려한 맥락 정보를 모델에 통합할 수 있는 가능성을 제시했습니다.



### Iconic Gesture Semantics (https://arxiv.org/abs/2404.18708)
Comments: 39 pages, 28 figures, under revision

- **What's New**: 이 연구는 상징적 제스처(iconic gesture)의 '의미'가 그것의 정보 평가에 기반한다는 점을 조명합니다. 정보 평가는 제스처를 언어적 수준으로 끌어올려 언어 내용과 상호 작용할 수 있게 합니다. 이 상호 작용은 주로 사전적 추론(lexicon-driven inferences)에 의해 규제됩니다.

- **Technical Details**: 제스처의 시각적 상징 모델(visual iconic model)의 지각 분류(perceptual classification)를 통해 정보 평가는 확장된 사례 제시(extended exemplification)로 설명됩니다. 상징 모델은 공간적으로 확장된 도메인 내에서 제스처 형태의 Frege/Montague 유사 진리 기능 평가(truth-functional evaluation)에서 파생됩니다. 연구는 또한 시각적 커뮤니케이션의 인스턴스를 분류할 때 Frege/Montague 프레임워크와 다른 의미 개념이 필요하다고 주장합니다.

- **Performance Highlights**: 이 논문은 제스처 해석을 위한 휴리스틱(heuristic)을 제공하여 작업 중인 의미론자(semanticist)에게 지침을 제공합니다. 동적 의미론 프레임워크(dynamic semantic frameworks)에서의 추론적 해석에 이르기까지, 운동 제스처 표현에서 모형 이론 평가(model-theoretic evaluation)에 이르는 상징적 제스처 의미론을 소개합니다.



### Work Smarter...Not Harder: Efficient Minimization of Dependency Length  in SOV Languages (https://arxiv.org/abs/2404.18684)
Comments: Accepted at CogSci-2024 as talk with full paper publication

- **What's New**: 이 연구는 자연 언어의 문법 구조와 문장 생성에서 관찰되는 의존성 길이 최소화(dependency length minimization, DLM) 현상에 대해 새로운 기계적 통찰을 제공합니다. 특히, SOV (Subject-Object-Verb) 언어에서의 전동사 구성 요소(preverbal constituent) 순서 결정이 전동사 구성 요소를 주동사(main verb) 옆에 위치시키는 것이 전체 의존성 길이를 최소화하는 것보다 더 잘 설명될 수 있다는 가설을 검증합니다.

- **Technical Details**: 연구자들은 Universal Dependency Treebank(UD)에서 수집한 대규모 코퍼스 데이터를 사용하여, Basque, Hindi, Japanese, Korean, Latin, Persian, Turkish 등의 SOV 언어에서 전동사 구성 요소의 순서 선호를 시뮬레이션하고 분석했습니다. '최소 노력' 전략(minimal effort strategy)으로 명명된 이 가설은 문장 내에서 가능한 구성 요소 순서의 전체 검색 공간을 시뮬레이션할 필요 없이 모든 전동사 의존성의 길이를 동시에 단축시킬 수 있는 접근법을 제안합니다.

- **Performance Highlights**: 이 연구 결과는 SOV 언어에 걸쳐 자연스럽게 생성된 코퍼스 문장들이 대부분 제안된 '최소 노력' 전략을 따르고 있음을 보여줍니다. 즉, 가능한 가장 짧은 전동사 구성 요소만을 주동사 옆에 배치하는 경향이 관찰되었습니다. 이러한 발견은 언어 사용자가 경제적이며 신속한 의사 결정 휴리스틱(heuristics)을 사용하여 의존성 길이를 최소화하려 하며, 이는 '제한된 합리성'(bounded rationality) 개념과 일치합니다.



### Revealing the Parametric Knowledge of Language Models: A Unified  Framework for Attribution Methods (https://arxiv.org/abs/2404.18655)
Comments: 14 pages, 6 figures

- **What's New**: 이 연구에서는 언어 모델(LMs)이 트레이닝 과정에서 얻은 지식을 파라메트릭으로 저장하는 방법과 이 지식을 업데이트하거나 수정하는 데 있어서의 도전을 다룹니다. 특히, 인스턴스 어트리뷰션(Instance Attribution, IA)과 뉴런 어트리뷰션(Neuron Attribution, NA)을 사용하여 어트리뷰션 메소드들이 언어 모델의 지식을 어떻게 드러내는지 비교하는 새로운 평가 프레임워크를 제안합니다.

- **Technical Details**: 이 논문은 NA-Instances와 IA-Neurons라는 두 가지 새로운 어트리뷰션 방법을 도입하여, 각각 NA를 사용하여 영향력 있는 트레이닝 인스턴스를 검색하고 IA를 통해 발견된 영향력 있는 인스턴스의 중요 뉴런을 발견하는 데 사용합니다. 추가적으로, 이 두 메소드에 의해 제공된 설명의 충실도를 평가하기 위한 포괄적인 테스트 리스트를 제안합니다.

- **Performance Highlights**: 실험 및 분석을 통해 NA는 일반적으로 IA보다 더 다양하고 포괄적인 정보를 제공함을 보여주었습니다. 그러나 IA는 NA에서 드러나지 않는, 고유하고 중요한 통찰력을 제공합니다. 이러한 결과를 통해 IA와 NA를 결합한 시너지 접근 방식이 언어 모델의 파라메트릭 지식에 대한 더욱 전체적인 이해를 위해 잠재력이 있음을 시사합니다.



### Do Vision & Language Decoders use Images and Text equally? How  Self-consistent are their Explanations? (https://arxiv.org/abs/2404.18624)
Comments: 27 pages, from which 12 pages contain the text of the main paper. 8 figures, 11 tables

- **What's New**: 비전 및 언어 모델(Vision and language models, VLMs)은 현재 다양한 멀티모달 과제에서 가장 성능이 우수한 아키텍처로 손꼽히고 있습니다. 이 연구는 VLM이 예측을 생성할 때와 설명을 제공할 때 시각적 및 텍스트 모달리티를 어떻게 다르게 사용하는지 조사하였습니다. 또한 VLM 디코더의 자체 일관성을 post-hoc 및 CoT(Chain of Thought, 사고의 연결) 설명 설정에서 평가하여, VLM 디코더에 대한 기존 테스트 및 측정을 확장하였습니다.

- **Technical Details**: VLM의 디코더에서는 텍스트에 대한 기여도가 이미지에 대한 기여도보다 훨씬 큽니다. 특히, 설명 생성에서 이미지의 기여도가 답변 생성에 비해 현저히 더 큽니다. 이 차이는 CoT 설정에서 post-hoc 설명 설정보다 더 커집니다. 연구는 또한 VALSE 벤치마크에서 최신 비전 언어(Vision-Language, VL) 디코더의 성능을 평가하였습니다. 이 벤치마크는 지금까지 주로 VL 인코더에 초점을 맞춰왔습니다.

- **Performance Highlights**: VLM은 대체로 LLM(Long-Range Language Models)보다 자체 일관성이 떨어집니다. VL 디코더는 VALSE에 의해 테스트된 대부분의 현상에 대해 여전히 어려움을 겪고 있습니다. 하지만, 이미지의 기여도는 답변 생성보다 설명 생성에서 훨씬 높습니다, 특히 CoT 설정에서 더욱 두드러집니다.



### The SAMER Arabic Text Simplification Corpus (https://arxiv.org/abs/2404.18615)
Comments: Accepted to LREC-COLING 2024. 15 pages, 6 tables, 1 figure

- **What's New**: 새로운 SAMER Corpus를 소개합니다. 이것은 학령기 학습자를 대상으로 하는 첫 번째 수동 주석이 달린 아랍어 병렬(corpus) 텍스트 단순화를 위한 것입니다. 텍스트는 1865년부터 1955년 사이에 출판된 15권의 아랍어 소설에서 선택되었습니다.

- **Technical Details**: 이 코퍼스(corpus)는 문서 및 단어 레벨에서 읽기 수준(readability level) 주석을 포함하고 있으며, 두 가지 다른 읽기 수준을 타겟으로 한 두 개의 간소화된 버전도 포함합니다. 코퍼스 선택 과정과 주석 생성 및 품질을 확보하기 위해 따른 가이드라인에 대해 설명합니다.

- **Performance Highlights**: 이 아랍어 텍스트 단순화, 자동 읽기 용이성 평가, 아랍어 교육용 언어 기술 개발을 지원하고 장려하기 위해 코퍼스는 공개적으로 사용 가능합니다.



### FREB-TQA: A Fine-Grained Robustness Evaluation Benchmark for Table  Question Answering (https://arxiv.org/abs/2404.18585)
Comments: Accepted at NAACL 2024

- **What's New**: 이 논문에서는 테이블 질의 응답(Table Question Answering, TQA) 시스템의 강인성(Robustness) 평가를 위한 세 가지 주요 요구사항을 정식화합니다. 이는 데이터 테이블의 구조 변경에도 불구하고 질문에 답할 수 있어야 하며(TQA systems should answer questions regardless of alterations in table structure), 관련 셀의 내용을 기반으로 응답해야 하고(they should base their responses on the content of relevant cells), 견고한 수치 추론 능력을 보여야 합니다(demonstrate robust numerical reasoning capabilities). 또한, 이러한 측면을 검토하기 위해 새로운 TQA 평가 벤치마크를 만들고 공개하였습니다.

- **Technical Details**: 연구에서는 테이블 구조 변화, 내용 기반 응답 및 수치 추론 능력을 중심으로 TQA 시스템의 성능을 평가하기 위한 새로운 벤치마크를 개발했습니다. 이 벤치마크는 기존의 상태 최고 기술(State-of-the-art) TQA 시스템들이 이러한 세 가지 측면에서 일관되게 우수하지 않다는 것을 밝혀냈습니다. 벤치마크는 영어로 제작되었으며, 연구자들이 TQA 시스템의 행동을 모니터링하고 강인한 시스템 개발을 위한 길을 열어주는 중요한 도구가 됩니다.

- **Performance Highlights**: 실험 분석을 통해 기존의 최첨단 TQA 시스템이 테이블 구조 변경, 편향 없는 내용 기반 응답, 강력한 수치 추론 능력 등 세 가지 주요 요구 사항에서 일관성 있게 뛰어남을 보여주지 못한다는 것을 드러냈습니다. 따라서, 이 벤치마크는 TQA 시스템의 발전에 중요한 발판을 마련해줄 것입니다.



### Analyzing Semantic Change through Lexical Replacements (https://arxiv.org/abs/2404.18570)
- **What's New**: 이 논문은 현대 언어 모델이 주변 맥락에 기반하여 단어의 의미를 파악하는 능력이 시맨틱(semantic) 변화로 인해 제한될 수 있다고 지적합니다. 특히, 	extit{lexical replacements}을 통해 생성된 예상치 못한 맥락의 영향을 연구하며, 새롭고 해석 가능한(interpretable) 시맨틱 변화 모델을 제안합니다. 또한, LLaMa 모델을 이용한 시맨틱 변화 감지의 첫 평가도 수행합니다.

- **Technical Details**: 연구팀은 	extit{replacement schema}를 도입하여 타깃 단어를 다양한 관련성을 가진 어휘로 대체함으로써 다양한 종류의 시맨틱 변화를 시뮬레이션합니다. 이 스키마는 시맨틱 변화를 해석할 수 있는 새로운 모델의 기본을 형성합니다.

- **Performance Highlights**: LLaMa(Large Language Model)를 사용하여 시맨틱 변화 감지에 대한 평가를 처음으로 시도하며, 이는 언어 모델이 새로운 맥락에서의 단어 사용을 어떻게 처리하는지 이해하는 데 중요한 기여를 합니다.



### Injecting Salesperson's Dialogue Strategies in Large Language Models  with Chain-of-Thought Reasoning (https://arxiv.org/abs/2404.18564)
Comments: arXiv admin note: substantial text overlap with arXiv:2308.14266

- **What's New**: 이 연구는 대화 시스템의 새로운 접근법인 SalesBot 2.0과 SalesAgent 모델을 소개합니다. 이전 버전의 문제점인 대화의 부자연스러운 전환과 일관성 부족을 해결하기 위해, SalesBot 2.0은 대규모 언어 모델(Large Language Models, LLMs)에서 상식 지식을 활용합니다. 또한, SalesAgent는 판매자의 상호작용을 학습하고, 사고의 연쇄(Chain-of-Thought, CoT) 추론 방식을 사용하여 향상된 대화 전략을 제공합니다.

- **Technical Details**: SalesBot 2.0의 개선된 데이터셋은 LLM을 활용하여 상식을 기반으로 한 전략적 프롬프팅을 통해 구축되었습니다. 새롭게 도입된 SalesAgent 모델은 판매 과정에서 발생할 수 있는 다양한 사용자 의도를 파악하고 적절한 대화 전략을 선택할 수 있도록 설계되었습니다. 이 모델은 CoT (Chain-of-Thought) 추론을 사용하여 주제의 전환을 능숙하게 처리합니다.

- **Performance Highlights**: 실험 결과, SalesBot 2.0과 SalesAgent 모델은 대화의 일관성을 크게 향상시키고 공격성을 감소시켜 사용자 시뮬레이션을 통해 그 효과를 확인할 수 있었습니다. 이는 판매-고객 상호작용을 위한 모델 학습을 개선하고, 대화 전략 제어에 있어 LLM의 효과적인 사용을 보여줍니다.



### Can GPT-4 do L2 analytic assessment? (https://arxiv.org/abs/2404.18557)
Comments: Accepted for the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024)

- **What's New**: 이 논문에서는 L2(제2언어) 숙련도 평가를 위해 자동화된 에세이 채점(Automated Essay Scoring, AES) 기술의 새로운 진보에 대해 다루고 있습니다. 특히, GPT-4라는 대규모 언어 모델을 사용하여 에세이의 구체적인 분석적(aspect) 요소를 평가하는 새로운 방법을 제안하고 있습니다. 이 연구는 전체적인 점수(holistic scores)뿐만 아니라, 에세이의 분석적 점수(analytic scores)를 예측하는 데 있어서 기존 방법의 한계를 극복하고자 합니다.



### Time Machine GP (https://arxiv.org/abs/2404.18543)
Comments: NAACL Findings 2024

- **What's New**: 새로운 연구에서는 시간의 흐름에 따라 변화하는 언어의 특성을 반영하기 위해 'Time Machine GPT (TiMaGPT)'라는 새로운 형태의 언어 모델을 개발했습니다. 이 모델들은 특정 시점을 기준으로 설계되어 미래의 사실이나 언어 변화에 대해서는 정보를 담고 있지 않습니다 ('nonprognosticative'). 이는 언어의 진화를 이해하고, 시계열 예측(time-series forecasting)과 같이 변화하는 맥락에서 모델을 적용할 때 중요합니다.

- **Technical Details**: TiMaGPT 모델은 기존의 대규모 언어 모델(Large language models, LLMs)과 다르게, 시간적 메타데이터(temporal metadata)를 포함한 데이터셋을 통해 훈련되었습니다. 이 모델은 기존의 언어 모델이 시간에 따라 달라지는 언어의 성질을 반영하지 못하는 문제를 해결하기 위해 시간을 특정 지점으로 설정하여 추가 학습(pre-training)하는 방식이 아닌, 처음부터 시간적으로 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: TiMaGPT의 핵심 이점은 미래의 정보나 언어 변화에 대한 데이터 없이도 해당 시점의 언어 사용 양상을 정확하게 반영할 수 있다는 점입니다. 이는 특히 역사적 언어 데이터나 시대별 언어 변화 연구에 매우 유용하며, 실시간으로 변화하는 언어 환경에서도 정확한 예측과 분석이 가능하게 도와줍니다. 또한, 연구팀은 이 모델과 훈련 데이터셋을 공개하여 다른 연구자들이 접근하고 활용할 수 있도록 하였습니다.



### Evaluating and Mitigating Linguistic Discrimination in Large Language  Models (https://arxiv.org/abs/2404.18534)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 다양한 언어로 된 쿼리에 대한 응답의 일관성을 탐구합니다. 특히, 안전성과 품질의 두 가지 측면에서 이루어진 분석을 통해 언어별 차별적인 행동을 밝혔습니다. 연구 결과는 영어, 프랑스어, 러시아어, 스페인어 등의 언어에서는 높은 인간 정렬 능력을 보여주는 반면, 벵골어, 조지아어, 네팔어, 마이틸리어 등에서는 덜 효과적임을 보여줍니다. 이러한 찾아낸 통찰을 바탕으로, 언어 차별을 완화하기 위한 새로운 방법인 LDFighter를 제안합니다.

- **Technical Details**: 연구는 4개의 대규모 언어 모델(LLama2-13b, Gemma-7b, GPT-3.5-turbo, Gemini-pro)를 사용하여 두 데이터셋(AdvBench와 NQ)을 분석합니다. LDFighter는 유사성 기반 투표(similarity-based voting) 방식을 활용하여 다양한 언어 사용자에게 일관된 서비스를 제공합니다. 이 방식은 해로운 쿼리의 탈옥률(jailbreak success rate)을 제어하고 응답의 평균 품질을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: LDFighter는 해로운 쿼리에 대한 탈옥률을 현저히 줄이고 평균적으로 응답 품질을 향상시킴으로써 그 효과를 입증하였습니다. 특히 벵골어, 조지아어, 네팔어, 마이틸리어와 같은 언어에서 불리한 결과가 나타나는 경우에도 더 일관된 결과를 제공합니다. 전반적으로 LDFighter는 F1 점수(F1 score) 시스템을 사용하여 평가되었으며, 영어, 덴마크어, 체코어, 슬로베니아어에서는 평균 0.1494의 높은 점수를 달성했습니다.



### MileBench: Benchmarking MLLMs in Long Contex (https://arxiv.org/abs/2404.18532)
Comments: 29 pages, 13 figures, 14 tables

- **What's New**: 새로운 벤치마크 'MileBench'가 도입되었습니다. 이 벤치마크는 멀티모달 대규모 언어 모델(MLLMs)의 멀티모달 롱콘텍스트(MultImodal Long-contExt) 능력을 평가하기 위해 설계되었습니다. 기존 벤치마크가 한정된 범위로 인해 진정한 성능을 파악하기 어려운 반면, MileBench는 긴 문맥과 다수의 이미지를 포함하는 다양한 태스크를 포함하여 모델의 이해력과 생성 능력을 평가합니다.

- **Technical Details**: MileBench는 진단적 평가 세트(diagnostic)와 현실적 평가 세트(realistic) 두 가지로 구성되어 있습니다. 이를 통해 MLLMs의 롱콘텍스트(long-context) 적응 능력과 긴 문맥 시나리오에서의 태스크 완수 능력을 체계적으로 평가합니다. 연구에서는 20개의 모델을 테스트 했으며 닫힌 소스인 GPT-4(Vision)와 Gemini 1.5 모델이 다른 모델들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: GPT-4(Vision)과 Gemini 1.5는 긴 문맥에서의 태스크 처리에서 탁월한 성능을 보이는 반면, 대부분의 오픈소스 MLLMs는 긴 문맥 상황에서 어려움을 겪는 것으로 나타났습니다. 모델의 성능 차이는 이미지 수가 증가함에 따라 더욱 확대되는 경향이 있습니다. 연구진은 멀티 이미지 시나리오에서 MLLMs의 롱콘텍스트 능력을 향상시키기 위한 연구 노력의 강화를 강력히 권장합니다.



### Explainability of Machine Learning Approaches in Forensic Linguistics: A  Case Study in Geolinguistic Authorship Profiling (https://arxiv.org/abs/2404.18510)
- **What's New**: 이 논문은 기계 학습(Machine Learning) 방법의 설명 가능성(Explainability)을 탐구하며, 특히 법률 언어학적 맥락에서의 다양성 분류(Variety Classification)를 통해 미상 텍스트의 지역 언어학적 프로파일링(Geolinguistic Profiling)을 진행합니다. Xie 등(2024)에 의해 제안된 접근법을 사용하여, 텍스트의 다양성을 분류하기 위해 가장 관련성이 높은 어휘 항목을 추출합니다.

- **Technical Details**: 저자들은 어휘 항목(Lexical Items) 추출을 통해 텍스트의 언어적 다양성을 판단하는 머신러닝 모델을 개발하였습니다. 이 모델은 주로 장소 이름(Place Names)과 같은 어휘 특징을 활용하여 분류 작업을 수행합니다. 설명 가능한 AI 모델을 사용함으로써, 법률 언어학(Forensic Linguistics)에서의 투명성 부족 문제를 해결하려고 합니다.

- **Performance Highlights**: 이 연구에서 개발된 모델은 해당 지역의 다양성을 대표하는 어휘 특징을 성공적으로 추출하였으며, 훈련된 모델들은 분류 정확도가 높은 것으로 나타났습니다. 이는 설명 가능성을 높이고, 사전에 정의된 범주에 따라 텍스트를 정확하게 분류할 수 있는 능력을 입증합니다.



### HFT: Half Fine-Tuning for Large Language Models (https://arxiv.org/abs/2404.18466)
Comments: Work in progress

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM: Large Language Models)이 지속적 학습 과정에서 발생할 수 있는 '격리적 망각(catastrophic forgetting)' 문제를 개선하기 위해 'Half Fine-Tuning (HFT)' 기법을 도입합니다. 이 기법은 모델의 전체 파라미터를 미세 조정하는 대신, 반은 새로운 작업 학습을 위해 선택하고 나머지 반은 기존 지식을 유지하기 위해 고정합니다.

- **Technical Details**: 'Half Fine-Tuning (HFT)'은 일부 파라미터를 주기적으로 리셋하여 원래의 지식을 일부 복원하는 새로운 접근 방식입니다. 이 방법은 최적화 관점에서의 실행 가능성 분석을 제공하며, 파라미터 선택 작업을 규제항(regularization term)으로 해석합니다. HFT는 기존의 미세 조정 프레임워크와 원활하게 통합될 수 있으며, 모델 구조의 변경 없이도 실행됩니다.

- **Performance Highlights**: HFT는 기존의 전체 미세 조정(FFT: Full Fine-Tuning)과 비교하여 망각 문제를 현저히 완화시키며, 다양한 하위 벤치마크에서 최고 성능을 달성함을 보여줍니다. 실제 실험을 통해, HFT는 훈련 시간을 약 30% 줄이면서도 효과성, 견고성 및 효율성을 입증합니다.



### Ethical Reasoning and Moral Value Alignment of LLMs Depend on the  Language we Prompt them in (https://arxiv.org/abs/2404.18460)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)가 언어와 문화의 영향을 받아 다른 언어로 도덕적 판단을 내리는 방식을 분석합니다. 특히 GPT-4, ChatGPT, 그리고 Llama2-70B-Chat 세 모델을 중심으로, 이들이 영어 외 다른 언어(스페인어, 러시아어, 중국어, 힌디어, 스와힐리어)에서 어떤 도덕적 판단을 내리는지 조사합니다. 도덕적 가치는 언어와 문화에 따라 다를 수 있으며, 이는 라지 랭귀지 모델의 윤리적 추론 능력에도 영향을 미칩니다.

- **Technical Details**: 이 연구는 Rao et al. (2023)에 의한 LLMs의 윤리적 추론 연구를 확장하여 다양한 언어 환경에서의 성능을 평가합니다. 연구 방법은 노르마티브 윤리학(normative ethics)의 세 분야—의무론(deontology), 덕유론(virtue), 결과론(consequentialism)—에서 파생된 윤리적 딜레마와 정책을 사용하여 LLMs를 테스트하는 프레임워크를 따릅니다.

- **Performance Highlights**: GPT-4는 언어에 따른 편향 없이 가장 일관된 윤리적 추론을 제공하는 것으로 나타났습니다. 반면에 ChatGPT와 Llama2-70B-Chat는 영어 이외의 언어에서 사용될 때 상당한 도덕적 가치 편향을 보여주었습니다. 흥미롭게도 모든 LLMs, GPT-4 포함,에서 이러한 편향의 성격은 언어에 따라 크게 달랐습니다.



### BMRetriever: Tuning Large Language Models as Better Biomedical Text  Retrievers (https://arxiv.org/abs/2404.18443)
Comments: Work in progress. The model and data will be uploaded to \url{this https URL}

- **What's New**: 새로운 바이오메디컬 검색 모델인 BMRetriever가 소개되었습니다. 이 모델은 대규모 바이오메디컬 말뭉치에서의 비지도 학습(unsupervised pre-training)과 라벨이 있는 데이터셋 및 합성 쌍(synthetic pairs)의 조합에 대한 지시적 미세조정(instruction fine-tuning)을 통해 바이오메디컬 검색 기능을 향상시키도록 설계되었습니다.

- **Technical Details**: BMRetriever는 다양한 바이오메디컬 응용 분야에서의 효과를 입증하기 위해 11개의 데이터셋에서 5개의 바이오메디컬 작업에 대해 실험되었습니다. 모델은 410M 버전과 2B 버전으로 제공되며, 각각 다른 모델 크기의 베이스라인(baselines)과의 비교를 통해 그 성능을 입증하였습니다.

- **Performance Highlights**: 410M 버전의 BMRetriever는 최대 11.7배 더 큰 모델을 능가하는 성능을 보였으며, 2B 버전은 5B 파라미터 이상을 가진 모델과 동등한 성능을 나타냈습니다. 이는 BMRetriever가 매우 파라미터 효율적(parameter efficiency)임을 보여줍니다.



### Mixture-of-Instructions: Comprehensive Alignment of a Large Language  Model through the Mixture of Diverse System Prompting Instructions (https://arxiv.org/abs/2404.18410)
- **What's New**: 이 연구에서는 다양한 작업을 보다 효과적으로 수행하기 위해 언어 모델을 제어하는 새로운 방법으로 Mixture-of-Instructions (MoI, 명령어 혼합) 기술을 소개합니다. 이 기술은 지시사항 결합 및 다양한 시스템 프롬프트를 사용하여 주어진 여러 과제들에서 언어 모델의 조정 효율성을 높이는 방안을 제시합니다.

- **Technical Details**: 이 새로운 MoI 기법은 명령 계열을 결합함으로써 여러 과제들에 대한 언어 모델의 정렬을 강화합니다. 연구팀은 코딩, 수학 문제 해결, 도구 사용과 같은 다양한 태스크를 포함한 7개의 벤치마크 데이터셋을 사용하여 MoI 강화 언어 모델의 성능을 평가하였습니다. 이 과정에서 개발된 Qwen-SFT-MoI 모델은 오픈소스 Qwen-7B-chat 모델을 기반으로 하여 더욱 발전된 생성 능력을 보여줍니다.

- **Performance Highlights**: Qwen-SFT-MoI 모델은 기존 모델 대비 코딩 작업, 수학적 문제 해결, 도구 사용 등에서 상당한 성능 향상을 보였습니다. 이는 MoI 기법이 다양한 언어 모델 태스크에 대한 조정과 성능 개선에 있어 효과적임을 시사합니다.



### MM-TTS: A Unified Framework for Multimodal, Prompt-Induced Emotional  Text-to-Speech Synthesis (https://arxiv.org/abs/2404.18398)
- **What's New**: 새롭게 개발된 MM-TTS(Multimodal Emotional Text-to-Speech System)는 다양한 모달리티로부터의 감정 신호를 활용하여 보다 표현력 있고 감정적으로 울림 있는 성능을 제공합니다. 이 시스템은 감정 프롬프트 정렬 모듈(EP-Align)과 감정 임베딩 유도 TTS(EMI-TTS)라는 두 가지 핵심 컴포넌트를 포함합니다.

- **Technical Details**: MM-TTS는 텍스트, 오디오, 시각적 모달리티간의 감정 특징을 조화롭게 융합하기 위하여 대조 학습(Contrastive Learning)을 이용한 EP-Align 모듈을 사용합니다. 또한, EMI-TTS는 정렬된 감정 임베딩을 최첨단 TTS 모델에 통합하여, 의도된 감정을 정확하게 반영하는 음성을 합성합니다.

- **Performance Highlights**: MM-TTS는 ESD 데이터셋에서 전통적인 E-TTS 모델에 비해 월등히 우수한 성능을 보여주었습니다. Word Error Rate (WER)과 Character Error Rate (CER) 모두 MM-TTS가 각각 7.35%, 3.07%의 점수를 달성했으며, 이는 주관적 평가에서도 MM-TTS가 인간의 음성과 비교할 만한 감정의 충실도와 자연스러움을 생성한다는 것을 입증합니다.



### Exploring the Limits of Fine-grained LLM-based Physics Inference via  Premise Removal Interventions (https://arxiv.org/abs/2404.18384)
- **What's New**: 언어 모델(Language Models, LMs)이 복잡하고 세밀한 수학적 추론을 수행할 때 발생할 수 있는 환각 현상을 탐구하고 있습니다. 특히, 물리학은 수학적 추론 능력을 평가하기에 풍부한 도메인을 제공하며, 물리적 맥락(예: 단위, 텐서 순서)은 복잡한 의미를 충족하는 상징 사용을 필요로 합니다. 이 연구에서는 다양한 표기법과 물리학의 하위 도메인을 포함하는 큐레이트된 데이터셋을 사용하여 언어 모델의 세밀한 수학적 및 물리적 추론 능력을 평가합니다.

- **Technical Details**: 새로운 연구에서는 인공 합성 인-콘텍스트(in-context) 예제를 사용하여 제로-샷(zero-shot) 성능을 향상시켰습니다. 또한, 지원 전제를 점진적으로 생략함으로써 유도 품질의 비선형적(non-linear) 열화를 보여주었습니다. 데이터셋은 물리학의 여러 하위 도메인과 다양한 표기를 포함하고 있어, 언어 모델이 수학적으로는 일관성을 유지할 수 있지만, 물리적 맥락은 대부분 무시하고 솔루션을 역설계(reverse-engineering)하는 경향이 있는지 평가합니다.

- **Performance Highlights**: 연구 결과, 언어 모델이 물리학적 맥락을 크게 고려하지 않고 수학적 추론을 수행하는 경향이 드러났습니다. 이는 물리학적 맥락이 주로 무시되고 해결책을 역설계하는 방식으로 접근하는 경우임을 나타냅니다. 또한, 전제를 점진적으로 생략하는 통해 유도 품질이 어떻게 변화하는지 확인함으로써, 모델의 물리적 지식 통합에 대한 한계와 함께 성능의 비선형적 열화를 입증했습니다.



### QANA: LLM-based Question Generation and Network Analysis for Zero-shot  Key Point Analysis and Beyond (https://arxiv.org/abs/2404.18371)
Comments: Under review as a conference paper at COLM 2024

- **What's New**: 이 연구에서는 '질문-대답 네트워크 분석'(QANA)이라는 새로운 의견 마이닝(opinion mining) 프레임워크를 제안합니다. QANA는 사용자의 댓글로부터 질문을 생성하고, 그 댓글이 질문에 얼마나 답할 수 있는지를 기반으로 이분 그래프(bipartite graph)를 구성하며, 중심성 지표(centrality measures)를 적용하여 의견의 중요성을 검토합니다. 특히, 이 연구는 다양한 중심성 지표를 사용하여 키 포인트의 중요성을 다양한 관점에서 평가할 수 있는 능력을 강조합니다.

- **Technical Details**: QANA는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 댓글에서 질문을 생성하고, 생성된 질문을 바탕으로 QA 네트워크를 구축합니다. 네트워크 내에서 노드 간의 중요성을 판단하기 위해 PageRank나 차수 중심성(degree centrality)과 같은 중심성 지표를 활용합니다. 또한, 이 연구에서는 질문 생성 스타일, LLM 선택, 임베딩 모델 선택이 QA 네트워크의 품질에 미치는 영향을 조사하여, 키 포인트 분석(Key Point Analysis, KPA) 데이터셋과 비교분석하였습니다.

- **Performance Highlights**: QANA는 키 포인트 매칭(Key Point Matching, KPM) 작업에서 이전의 상태 기술(supervised models)과 비슷한 성능을 제로-샷(zero-shot) 방식으로 달성하였고, 계산 비용을 제곱에서 선형으로 줄였습니다. 또한, 키 포인트 생성(Key Point Generation, KPG) 작업에서는 PageRank나 차수 중심성이 높은 질문이 수동으로 주석이 달린 키 포인트와 잘 일치함을 보여주었습니다. 중심성 지표 선택에 따라 키 포인트의 중요성을 다양한 관점에서 평가할 수 있는 유연성을 제공하는 것이 주된 기여입니다.



### FoundaBench: Evaluating Chinese Fundamental Knowledge Capabilities of  Large Language Models (https://arxiv.org/abs/2404.18359)
- **What's New**: 이 논문에서는 중국어 대규모 언어 모델(LLMs)의 기본 지식 능력을 평가하기 위해 'FoundaBench'라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 상식(common sense)과 K-12 교육 과목을 포함한 다양한 질문들로 구성되어 있으며, 중국 문화에 특화된 평가를 가능하게 합니다.

- **Technical Details**: FoundaBench는 3354개의 다중 선택 문제를 포함하고 있으며, 이는 일상 지식 및 중국의 사회, 문화, 예술 등을 다룹니다. 평가는 전통적인 방법과 CircularEval 프로토콜을 사용하여 모델의 편향을 최소화하는 방식으로 이루어졌습니다. 또한, 평가는 다양한 크기와 언어 지향의 12개 모델에 대해 수행되었습니다.

- **Performance Highlights**: 해당 벤치마크 평가를 통해 중국어 코퍼스(corpus)로 사전 훈련된 모델들이 우수한 성능을 보였으며, 모델들 사이에서 추론 및 기억 회상 능력에 상당한 차이가 있음을 밝혔습니다. 특히, 중국어에 최적화된 벤치마크를 통해 대규모 언어 모델의 기본 지식 수준을 보다 정확하게 이해할 수 있는 새로운 기준을 제시했습니다.



### Comparing LLM prompting with Cross-lingual transfer performance on  Indigenous and Low-resource Brazilian Languages (https://arxiv.org/abs/2404.18286)
Comments: Accepted to the Americas NLP Workshop at NAACL 2024 (this https URL)

- **What's New**: 이 논문은 낮은 리소스 언어(Low-resource languages, LRLs)의 자연언어처리(Natural Language Processing, NLP) 태스크 수행에서 대규모 언어 모델(Large Language Models, LLMs)의 성능에 초점을 맞추고 있습니다. 특히 브라질의 12개, 아프리카의 2개 LRL과 비교하여 영어와 브라질 포르투갈어와 같은 고자원 언어(High-resource languages, HRLs)의 품사 태깅(Part-of-Speech, POS) 성능을 평가하였습니다.

- **Technical Details**: 연구에서는 GPT-4와 XLM-R을 활용해 POS 태깅 성능을 평가했습니다. GPT-4는 영어와 브라질 포르투갈어를 기반으로 0샷(Zero-shot) 평가를 수행했고, XLM-R은 언어 적응(Language Adaptation) 후 상호간 전이 학습(Cross-lingual Transfer)을 통한 성능 향상을 도모했습니다. 논문은 이러한 접근 방식들이 LRLs에 대한 POS 태깅에 있어서 여전히 낮은 성능을 보이는 이유를 분석합니다.

- **Performance Highlights**: GPT-4와 XLM-R 모두 LRLs에서 낮은 성능(34% 미만)을 보여주었으나, 언어 적응을 통한 성능은 일부 언어에서 3에서 12 포인트가 향상되었습니다. 이는 고자원 언어들이 90% 이상의 높은 성능을 달성한 것과 대조되는 결과입니다. 이 연구는 LRLs를 위한 향후 NLP 자원과 도구 개발의 필요성을 강조합니다.



### Bias Neutralization Framework: Measuring Fairness in Large Language  Models with Bias Intelligence Quotient (BiQ) (https://arxiv.org/abs/2404.18276)
Comments: 41 pages

- **What's New**: 최근에 등장한 Comprehensive Bias Neutralization Framework (CBNF)는 Large Language Model(Large Language Model, LLM)의 내재된 편향을 다루는 새로운 접근방식을 제안합니다. 이 프레임워크는 기존의 편향 측정 및 완화 방법론인 Large Language Model Bias Index (LLMBI)와 Bias removaL with No Demographics (BLIND)를 통합하여, 인구 통계학적 주석 없이 LLM에서 인종 편향을 탐지하고 측정하며 완화할 수 있는 새로운 지표인 Bias Intelligence Quotient (BiQ)를 도입합니다.

- **Technical Details**: CBNF는 LLMBI를 추가적인 공정성 지표와 향상시켜 새로운 지표인 BiQ를 만듭니다. 이를 통해 인종, 문화 및 성별 편향을 종합적으로 평가할 수 있는 다차원적 메트릭(metric)을 제공합니다. 연구에서는 Latimer AI(흑인 역사 및 문화에 특화되도록 추가 훈련된 언어 모델)와 ChatGPT 3.5를 비교 분석하여, Latimer AI가 목표 지향적 훈련(training)과 정제된 편향 완화 전략을 통해 편향을 효과적으로 감지하는 능력을 입증합니다.

- **Performance Highlights**: Latimer AI는 흑인 역사와 문화에 초점을 맞춤으로써 ChatGPT 3.5에 비해 인종적, 문화적, 성별 편향을 더욱 효과적으로 감지하고 완화합니다. 이는 타겟 적인 훈련과 편향 완화 전략의 중요성을 강조하며, AI 공정성에 대한 섬세한 접근법의 필요성을 부각시킵니다.



### Parameter-Efficient Tuning Large Language Models for Graph  Representation Learning (https://arxiv.org/abs/2404.18271)
- **What's New**: 새로운 연구인 Graph-aware Parameter-Efficient Fine-Tuning (GPEFT)은 텍스트가 풍부한 그래프에서 LLM(Large Language Models)을 사용하여 효율적인 그래프 표현 학습을 시키는 방법을 도입합니다. 이 방법은 그래프 신경망 (Graph Neural Network, GNN)을 이용하여 노드들 간의 구조적 정보를 그래프 프롬프트에 인코딩하고, 이를 텍스트 시퀀스의 시작에 삽입합니다.

- **Technical Details**: GPEFT는 GNN을 사용하여 이웃 노드로부터의 구조적 정보를 그래프 프롬프트 형태로 인코딩한 후, 이 프롬프트를 텍스트 시퀀스의 시작 부분에 삽입하는 기법입니다. 이 프롬프트는 동결된 LLM이 노드 텍스트의 다음 토큰을 예측하도록 돕기 위해 사전에 훈련되었습니다. 이 접근 방식은 저렴한 파인 튜닝 비용으로 LLM으로부터 직접 노드 임베딩을 생성합니다.

- **Performance Highlights**: GPEFT는 8개의 다른 텍스트-리치 그래프에서 종합적인 실험을 거쳐 평가되었으며, 링크 예측 평가에서 hit@1과 평균 역순위(Mean Reciprocal Rank, MRR)에서 평균 2%의 성능 향상을 보였습니다. 이 결과는 다양한 대형 언어 모델들과의 원활한 통합 가능성을 보여주며, 이 모델의 효과성과 효율성을 입증합니다.



### Modeling Orthographic Variation Improves NLP Performance for Nigerian  Pidgin (https://arxiv.org/abs/2404.18264)
Comments: Accepted to LREC-COLING 2024 Main Conference

- **What's New**: 이 연구는 나이지리아 피진(Nigerian Pidgin) 텍스트에서 흔히 발견되는 각종 정서(orthographic) 변이를 처음으로 기술하고 모델링합니다. 나이지리아 피진은 약 1억 명의 사람들에 의해 구술되는 주로 구술 언어이며, 정확한 철자 규칙이 아직 채택되지 않아 사용 가능한 데이터세트가 소수만 존재하며 철자법의 변이로 인한 노이즈가 존재합니다.

- **Technical Details**: 이 연구는 나이지리아 피진의 정서 변이를 기반으로 한 음성-이론적(phonetic-theoretic) 프레임워크를 제안하여 단어 편집을 통해 정서 변이를 생성하고, 이를 훈련 데이터 확장에 사용합니다. 새로운 정서 변이는 테스트 세트에는 존재하지만 원래 훈련 세트에는 없던 데이터를 보완해 줍니다.

- **Performance Highlights**: 제안된 정서 변이 생성 프레임워크는 기존 텍스트와 결합하여 훈련 데이터를 확장함으로써 성능을 향상시켰습니다. 특히, 감성 분석에서는 2.1 포인트, 영어 번역에서는 1.4 BLEU(Bilingual Evaluation Understudy) 점수의 성능 향상을 달성했습니다.



### Mapping 'when'-clauses in Latin American and Caribbean languages: an  experiment in subtoken-based typology (https://arxiv.org/abs/2404.18257)
Comments: 10 pages, 6 figures. To be published in the 2024 Proceedings of the Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP)

- **What's New**: 이 논문은 라틴 아메리카와 카리브 지역의 언어들이 시간적 종속성을 어떻게 표현하는지에 대한 변화를 탐구합니다. 특히, 이 지역 언어들이 전반적으로 풍부한 형태학적 표시(Morphological Marking)를 사용한다는 점에 중점을 둡니다. 이는 유럽 언어들과 대조적이며, 이러한 형태학적 접근 방식은 이전의 연구에서 덜 다뤄졌습니다. 연구는 probabilistic semantic maps을 사용하여 데이터를 분석하고, 언어간의 시간적 종속성을 표현하는 다양한 방법을 탐색합니다. 또한, 이 논문은 데이터 분석 도구를 개발하고 공개하며, 연구 결과를 활용하여 향후 계산 실험을 용이하게 합니다.

- **Technical Details**: 연구진은 Mayer and Cysouw의 (2014) 대규모 병렬 코퍼스(New Testament translations)를 사용하여, 라틴 아메리카 및 카리브 지역의 언어 변종을 분석합니다. 이 데이터 분석을 위해 SyMGIZA++을 사용해 영어와 대상 언어 간에 토큰 레벨에서 일대일 대응을 이루어 냈습니다. 그 후, 영어의 'when'과 해당 언어들에서의 유사어를 추출하고, 이를 바탕으로 probabilistic semantic maps를 생성합니다. 이 방법론은 언어 내부의 변화와 다양한 표현 방법 사이의 경계를 포착하는 데 유용하며, SuperPivot 접근방식을 기반으로 하지만 주요한 수정을 가했습니다. 연구에서는 문자 n-grams (character n-grams)과 영어의 'when'간의 연관성을 포함시켜 형태학적 차이를 포착합니다.

- **Performance Highlights**: 연구 결과는 라틴 아메리카 및 카리브 지역 언어들의 시간적 종속부를 표현하는 방식에 대한 깊은 이해를 제공합니다. 특히, 이 지역의 언어들이 주로 형태학적인 수단(Morphological Means)을 사용하여 'when' 문장을 표현한다는 점을 밝혀낸 것은 주목할 만한 성과입니다. 자동 정렬의 정밀도(precision)는 0.66, 회수율(recall)은 1, F1-점수는 0.79로, 데이터의 정렬 및 추출 과정이 양호한 정확도를 보여줍니다. 또한, 이 연구는 해당 지역의 언어 데이터를 기반으로 하여 언어학적 연구에 편향을 감소시킵니다.



### PatentGPT: A Large Language Model for Intellectual Property (https://arxiv.org/abs/2404.18255)
Comments: 19 pages, 9 figures

- **What's New**: 이 기술 보고서는 지적 재산(IP) 분야에 특화된 대규모 언어모델(Large Language Models, LLM)의 저비용 표준화된 훈련 절차를 처음으로 제시합니다. 이를 통해 특허 및 저작권과 같은 IP 분야의 독특한 요구 사항을 충족시키는 것이 목표입니다.

- **Technical Details**: PatentGPT 시리즈 모델은 오픈소스의 사전 훈련된 모델을 기반으로 하여 새롭게 훈련되었습니다. 이 모델은 SMoE(Sparse Mixture of Experts) 아키텍처를 활용하여 IP 도메인에서 GPT-4와 비교할 수 있는 성능을 달성했습니다.

- **Performance Highlights**: PatentGPT는 오픈소스의 IP 지향 벤치마크 MOZIP에서 GPT-4를 능가하는 성능을 보였으며, 2019 중국 특허 대리인 자격 시험에서 65점을 획득하여 인간 전문가 수준에 도달했습니다. 또한 긴 텍스트 처리 작업에서 더 나은 비용 효율성을 보여 GPT-4의 대안으로 사용될 수 있습니다.



### LEGENT: Open Platform for Embodied Agents (https://arxiv.org/abs/2404.18243)
Comments: Demo Paper

- **What's New**: LEGENT는 대규모 언어 모델 (LLMs)과 대규모 다중 모드 모델 (LMMs)을 사용하여 실제 환경에서 복잡한 작업을 수행할 수 있는 가상의 인간과 같은 에이전트를 개발하기 위한 새로운 오픈 소스, 확장 가능한 플랫폼을 소개합니다. 이 플랫폼은 상호 작용적인 3D 환경과 고급 데이터 생성 파이프라인을 결합하여 시뮬레이션된 세계에서의 감독 (supervision)을 활용합니다.

- **Technical Details**: LEGENT 플랫폼은 두 부분으로 구성됩니다: 첫째, 다양하고 현실적인 상호 작용 장면을 제공하는 3D 환경과 이러한 장면과 에이전트의 행동을 생성하기 위한 데이터 생성 파이프라인이 있습니다. 이 시스템은 최첨단 알고리즘을 사용하여 에이전트의 광범위하고 다양한 행동 궤적을 대규모로 생성할 수 있습니다. 또한, 리얼리즘 물리, 다양한 렌더링 및 상호 작용 가능한 객체를 지원합니다.

- **Performance Highlights**: LEGENT를 사용하여 훈련된 시각-언어-행동 모델은 GPT-4V보다 우수한 성능을 보여주며, 이는 LEGENT의 효과적인 훈련 환경과 데이터 생성 능력을 입증합니다. 이 모델은 내비게이션 및 구현 질문 응답 작업에서 텍스트 및 이심률 (egocentric) 시각 입력을 처리하고 직접 제어 및 텍스트 응답을 생성합니다.



### From Persona to Personalization: A Survey on Role-Playing Language  Agents (https://arxiv.org/abs/2404.18231)
Comments: Preprint

- **What's New**: 이 연구에서는 Role-Playing Language Agents(RPLAs; 역할 연기 언어 에이전트)의 최근 발전과 LLM(Large Language Models; 대형 언어 모델) 기술과의 통합에 대해 종합적으로 조사합니다. RPLAs는 다양한 인물, 가상 인물 및 실존 인물을 모방할 수 있으며, 이를 통해 디지털 클론, 게임 내 AI 캐릭터, 개인 비서 등 다양한 응용 분야에 활용됩니다.

- **Technical Details**: 이 논문에서는 LLM을 사용하여 구축된 RPLAs의 세 가지 주요 유형, 즉 Demographic Persona(인구 통계적 페르소나), Character Persona(문화적 캐릭터 페르소나), 그리고 Individualized Persona(개인화된 페르소나)를 소개합니다. 각 페르소나 유형은 데이터 소싱, 에이전트 구성 및 평가 방법이 다르며, LLM의 인컨텍스트 학습(in-context learning), 지시사항 따르기(instruction following), 사회적 지능(social intelligence) 등의 기능을 활용하여 인간과 유사한 반응을 이끌어냅니다.

- **Performance Highlights**: LLMs의 발전은 RPLAs의 사실감 있는 인간 유사성을 크게 향상시켰으며, 이러한 에이전트들은 인간의 사회적 상호 작용을 모방하는 데 있어 주목할 만한 성과를 보여주고 있습니다. 특히, 각각의 페르소나 유형에 맞게 훈련된 RPLAs는 매우 정교한 역할 연기 성능을 보여 주었으며, 이는 다양한 실제 애플리케이션에서의 유용성을 증명하고 있습니다.



### TextGram: Towards a better domain-adaptive pretraining (https://arxiv.org/abs/2404.18228)
Comments: Accepted at SPELLL 2023

- **What's New**: 본 논문에서는 대규모 언어 모델, 특히 Transformer 모델 (pre-trained language models, PLMs)의 사전 훈련을 위해 정교한 데이터 선택 기술을 활용하는 새로운 방법을 제안합니다. 'TextGram'이라는 도메인 적응형 데이터 선택 방법을 소개하며, 이 방법은 효율적인 데이터 선택을 통해 불필요한 계산을 줄이고 모델의 정확성을 유지하면서 훈련 시간을 단축시킵니다.

- **Technical Details**: 우리는 12층의 양방향 인코더 표현(Bidirectional Encoder Representations from Transformers, BERT) 모델을 사용하고, 이는 전체 영어 위키백과 및 책 말뭉치를 사용하여 마스크 언어 모델링(Masked Language Modeling, MLM) 전략으로 훈련됩니다. 데이터 선택은 대규모 말뭉치에서 중요한 데이터를 식별하고 선택하여 사전 훈련을 더욱 효과적으로 수행할 수 있도록 합니다. 우리의 데이터 선택 기술은 시간과 계산 자원을 절약하는 동시에 환경적 영향을 줄이는 데 중점을 둡니다. 주요 기법으로는 N-Grams, TF-IDF, Perplexity 기반 선택 등이 있으며, 새로운 TextGram 방법을 도입하여 이들과 비교합니다.

- **Performance Highlights**: 제안한 TextGram 방법론을 사용하여 사전 훈련된 모델을 특정 도메인의 텍스트 분류 작업에 적용했을 때 다른 데이터 선택 방법들에 비해 우수한 성능을 보였습니다. 데이터 선택을 통해 모델의 정확도를 유지하면서도 훈련에 필요한 시간과 자원을 상당히 절감할 수 있었습니다. 이는 데이터 센터가 세계 에너지의 6% 이상을 소비할 것으로 예상되는 2030년까지 환경적 발자국을 줄이는 데 크게 기여할 수 있습니다.



### L3Cube-MahaNews: News-based Short Text and Long Document Classification  Datasets in Marath (https://arxiv.org/abs/2404.18216)
Comments: Accepted at SPELLL 2023

- **What's New**: 새롭게 소개되는 L3Cube-MahaNews는 마라티어(Marathi)로 된 뉴스 헤드라인과 기사를 분류하는 대규모 텍스트 분류 데이터셋입니다. 이 데이터셋은 12개의 다양한 카테고리로 분류된 1.05만 건 이상의 기록을 포함하며, 짧은 텍스트, 긴 문서, 중간 길이의 문단을 위한 세 가지 감독 데이터셋(supervised datasets)으로 구성되어 있습니다.

- **Technical Details**: 이 연구에서는 최신 기술인 BERT 모델을 사용하여 기준 성능을 제공하고, 단일 언어(Monolingual) 모델인 MahaBERT와 다중 언어(Multilingual) 모델인 IndicBERT, MuRIL을 비교 분석하였습니다. 모든 데이터셋에서 가장 우수한 성능을 보인 것은 단일 언어 모델인 MahaBERT였습니다.

- **Performance Highlights**: MahaBERT 모델이 IndicBERT와 MuRIL 같은 다중 언어 BERT 모델보다 뛰어난 성능을 보여 주었으며, 이는 MahaBERT가 마라티어 텍스트 분류에 특화되어 있기 때문으로 분석됩니다. 이러한 결과를 바탕으로, 지역 언어 데이터 처리에 특화된 모델이 더 효과적일 수 있다는 점이 강조되었습니다.



### Exploring the Robustness of In-Context Learning with Noisy Labels (https://arxiv.org/abs/2404.18191)
Comments: ICLR 2024 Workshop on Reliable and Responsible Foundation Models

- **What's New**: 이 연구는 Transformer 모델의 In-Context Learning (ICL) 능력이 잡음이 있는 레이블(noisy labels)의 상황에서도 얼마나 견고한지를 탐구합니다. 특히, 잡음이 포함된 학습 데이터(training data)를 사용하는 것이 모델의 내성을 높일 수 있는지에 대해 연구하였습니다. 이러한 접근 방식은 ICL의 독특한 학습 메커니즘을 이해하고, 더 나은 레이블 잡음 대응 방안을 모색하는 데 중요한 통찰력을 제공합니다.

- **Technical Details**: 저자들은 Transformer 모델이 선형 회귀 기능(linear regression functions)를 이해하도록 인공 데이터셋(synthetic dataset)을 사용하여 트레이닝하고, 이를 라벨 잡음 학습(noisy-label learning)에 적용하였습니다. 잡음이 있는 레이블을 사용하여 이러한 모델들의 견고함을 평가하고, 추가적으로 트레이닝 단계에서 잡음을 도입하는 것이 추론(inference) 단계에서의 모델 견고성을 강화하는지를 조사했습니다.

- **Performance Highlights**: 실험 결과, Transformer 모델들은 표시된 레이블의 잡음에 대하여 상당한 내성을 보였으며, 트레이닝 과정에 노이즈를 추가함으로써 이러한 내성이 더욱 향상될 수 있음을 발견하였습니다. 이는 ICL의 효과적인 노이즈 관리 및 데이터 어그멘테이션(data augmentation) 전략 수립에 중요한 기여를 합니다.



### EkoHate: Abusive Language and Hate Speech Detection for Code-switched  Political Discussions on Nigerian Twitter (https://arxiv.org/abs/2404.18180)
Comments: AfricaNLP workshop @ ICLR2024 and WOAH @ NAACL2024

- **What's New**: 이 논문에서는 나이지리아 라고스 주 선거에 대한 3,398개의 트윗을 모아 'EkoHate'라는 새로운 코드 스위칭(code-switching) 혐오 언어 및 증오 발언 검출 데이터셋을 만들어냈습니다. 이 데이터셋은 이진 분류(보통 대 공격적)와 세밀한 네 가지 레이블 주석 체계를 사용하여 주석이 붙었습니다. 이는 나이지리아의 정치적 토론에서 사용될 수 있는 혐오 언어 탐지를 위한 중요한 기여입니다.

- **Technical Details**: EkoHate 데이터셋은 라고스의 세 명의 주요 정치 후보와 그들의 팔로워들 사이의 정치 토론에 대한 트윗을 분석하여 만들어졌습니다. 데이터는 이진(‘normal’ vs ‘offensive’) 및 보다 세밀한 네 가지 레이블('normal', 'abusive', 'hateful', 'contempt')로 주석 처리되었습니다. BERT 모델을 도메인 특화 버전으로 파인튜닝하여 공격적인 트윗을 95.1 F1 점수로 식별하는 높은 성능을 보였으며, 네 레이블 주석 체계에서는 70.3 F1을 달성했습니다. 이 데이터셋은 OLID, HateUS2020, FountaHate 등 다른 공개 데이터셋으로의 교차 언어 전이 학습을 통해서도 좋은 성능을 보였습니다.

- **Performance Highlights**: 이 데이터셋은 이진 정보에서 95.1 F1 점수를, 세밀한 레이블링에서는 70.3 F1 점수를 달성했습니다. 또한, 국제적으로 공유된 다른 데이터셋에 대해 진행한 교차 검증에서 OLID에 71.8 F1 점수, HateUS2020에 62.7 F1 점수, FountaHate에 53.6 F1 점수를 기록하여, 코드 스위칭 특성과 문화적 특수성에도 불구하고 다른 지역의 정치적 토론에 잘 일반화됨을 보여주었습니다.



### Explaining vague languag (https://arxiv.org/abs/2404.18154)
- **What's New**: 이 연구는 게임 이론(game-theoretic) 및 베이지안(Bayesian) 접근 방식을 사용하여 언어의 모호성(vagueness)을 설명하려는 두 연구 결과를 비교하고, 이 두 결과가 서로 모순되지 않음을 밝힙니다. 특히, 립맨(Lipman)의 모호성 정의는 신호 전략의 속성에만 의존하는 반면, 에그레(Égré) 등의 연구는 의미론적(semantic) 내용 층을 포함합니다.

- **Technical Details**: 립맨의 게임 이론적 정의에 따르면, 모호성은 신호 전략의 결과로 발생하며, 사용된 언어가 이산적 판단에 압도적으로 유리하지 않는 경우가 많습니다. 반면에 에그레 등은 베이지안 접근을 사용하여 모호한 언어가 정밀한 언어보다 엄격하게 더 유익하다는 시나리오를 제시합니다. 그들의 정의는 실제 조건(truth-conditions)에 열린 매개변수를 포함하는 표현이 모호하다고 주장합니다.

- **Performance Highlights**: 모호한 언어가 조건에 따라 정밀한 언어보다 나을 수 있는 시나리오를 설명함으로써, 이 연구는 모호성이 단순한 전략적 선택 이상의 것임을 시사합니다. 예를 들어, 의사결정 과정에서 모호한 표현을 사용하면 상대방의 예상이나 맥락에 따라 정보를 더 유연하게 조정할 수 있습니다.



### CRE-LLM: A Domain-Specific Chinese Relation Extraction Framework with  Fine-tuned Large Language Mod (https://arxiv.org/abs/2404.18085)
Comments: preprint

- **What's New**: 이 연구에서는 도메인-특정 중국어 관계 추출(Domain-Specific Chinese Relation Extraction, DSCRE)을 목표로 합니다. 최근 LLMs (Large Language Models, 대형 언어 모델들)의 발전에 주목함에 따라, 본 논문은 새로운 프레임워크 CRE-LLM을 제안합니다. 이는 Llama-2, ChatGLM2, Baichuan2와 같은 오픈 소스 LLM을 활용하고, 적절한 프롬프트 구축과 지도학습(fine-tuning)을 통해 관계 추출의 개선을 달성합니다.

- **Technical Details**: CRE-LLM은 오픈 소스 LLM을 활용하여 모델의 논리 인식(logic-awareness)과 생성 능력을 증가시킵니다. 입력 텍스트 데이터에서 주어진 엔티티의 관계를 직접 추출하는 방식으로 동작하며, 이는 특히 복잡한 네트워크 구조 설계, 훈련 시 높은 자원 소모를 감소시키는데 중점을 두고 있습니다.

- **Performance Highlights**: CRE-LLM은 금융 도메인(FinRE)과 문학(문장, SanWen) 데이터셋에서 광범위한 실험을 진행하였으며, 높은 성능과 강력한 신뢰성을 보여주었습니다. 특히 FinRE 데이터셋에서 최고의 성능(state-of-the-art, SOTA)을 달성하였습니다. 본 연구의 코드는 공개적으로 사용 가능합니다.



### Contextual Spelling Correction with Language Model for Low-resource  Setting (https://arxiv.org/abs/2404.18072)
Comments: 8 pages

- **What's New**: 이 연구에서는 자료가 부족한 언어에 대한 맞춤법 교정(Spell Correction, SC) 문제를 해결하려고 시도하였습니다. 특히, 네팔어에 대한 약간의 원시 텍스트 코퍼스만 접근 가능할 때 이 문제를 해결하기 위한 새로운 접근법이 제시되었습니다.

- **Technical Details**: 이 연구는 적은 양의 데이터로 학습된 작은 규모의 어휘 기반 변환기(Language Model, LM)를 사용하여 문맥적 이해력을 향상시키고, 비지도 방식으로 코퍼스에서 확률적 오류 규칙을 추출하여 오류 모델(error model)을 구축합니다. 그 후, 잘 알려진 노이즈 채널 프레임워크(noisy channel framework)를 사용하여 LM과 오류 모델을 결합하여 SC 모델을 개발합니다.

- **Performance Highlights**: 이 접근 방식의 효과는 네팔어의 실험을 통해 입증되었습니다. 네팔어로 작성된 텍스트 데이터에서 맞춤법 교정 모델을 개발하고 평가함으로써 이 방법이 저자원 언어에 대한 효과적인 SC 솔루션을 제공할 수 있음을 보여줍니다.



### Can Perplexity Predict Fine-Tuning Performance? An Investigation of  Tokenization Effects on Sequential Language Models for Nepa (https://arxiv.org/abs/2404.18071)
Comments: 11 pages

- **What's New**: 이 논문은 네팔어 언어 모델에서의 서브워딩(subwording) 전략이 어떻게 이해력에 영향을 미치는지 조사합니다. 특히 네팔어에 대해 6가지 다른 토큰화(tokenization) 체계를 사용하여 언어 모델을 사전 학습(pretrain)하고, 여러 하위 작업(downstream tasks)에 대해 세밀하게 조정(finetune)하는 과정을 진행하였습니다.

- **Technical Details**: 이 연구에서 사용된 알고리즘에는 byte-level BPE, SentencePiece 등이 포함되었습니다. 이러한 알고리즘들은 최근 GPT, RoBERTa와 같은 모델에서 사용되었으나, 네팔어에 대해서는 SentencePiece가 평균적으로 더 우수한 세밀 조정 성능을 보였습니다. 연구는 전통적인 Bert-based 모델이 아닌 순차적 트랜스포머(sequential transformer) 기반의 언어 모델을 사용하였습니다.

- **Performance Highlights**: 비록 byte-level BPE가 흔히 사용되는 알고리즘임에도 불구하고, 네팔어를 위한 언어 모델에서는 SentencePiece가 더 나은 세밀 조정 성과를 보였다는 점이 강조됩니다. 이 연구는 다른 언어에 대한 유사한 분석들이 주로 Bert-based 모델에 집중된 것과 달리 순차적 트랜스포머 모델을 사용함으로써 새로운 관점을 제시합니다.



### Efficient LLM Inference with Kcach (https://arxiv.org/abs/2404.18057)
Comments: Technical Report, 8 pages

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM: Large Language Models)의 효율적인 인퍼런스를 위한 새로운 기술인 KCache 기법을 소개합니다. 기존의 KV Cache 방식이 메모리 오버헤드를 유발하는 문제점을 해결하기 위해, KCache는 성능을 개선하면서도 정확성을 유지하는 방법을 제안합니다.

- **Technical Details**: KCache는 인퍼런스 과정에서 K Cache부를 HBM(고대역폭 메모리)에 유지하고 V Cache를 CPU 메모리에 저장하는 구조로 설계되었습니다. 어텐션(Attention) 계산에서 소프트맥스 결과를 이용하여 중요한 정보를 필터링하고 해당 V Cache를 CPU 메모리에서 다시 가져와 계산할 수 있습니다. 이는 트랜스포머(Transformer) 모델의 구조적 특성을 활용하여 CPU 메모리의 여유 공간을 효과적으로 활용하고 HBM의 용량을 증가시키는 전략입니다.

- **Performance Highlights**: KCache를 사용함으로써 인기 있는 LLM의 처리량이 기준 대비 40% 향상되었으며, 정확도를 유지할 수 있습니다. 이는 KCache가 메모리 병목 현상을 완화하고, 시스템 전체의 처리량을 최대화하는데 기여한다는 것을 의미합니다. 또한, 모델 인퍼런스 실험을 통해 그 효과성을 검증하였습니다.



### Utilizing Large Language Models for Information Extraction from Real  Estate Transactions (https://arxiv.org/abs/2404.18043)
- **What's New**: 부동산 판매 계약에는 재산 거래에 중요한 정보가 포함되어 있지만, 데이터를 수동으로 추출하는 것은 시간이 많이 걸리고 오류가 발생하기 쉽습니다. 이 논문은 부동산 계약서에서 정보를 자동으로 추출하기 위해 대규모 언어 모델, 특히 트랜스포머(Transformer) 기반 구조를 적용하는 방법을 탐구합니다.

- **Technical Details**: 트랜스포머 기반 언어 모델은 다양한 자연어 처리 작업에 탁월한 성능을 보여 주며, 이 연구에서는 그러한 모델들이 복잡한 법적 언어와 형식을 가진 부동산 계약서에서 중요 정보를 어떻게 정확하게 추출할 수 있는지 설명합니다. NLP(Natural Language Processing) 기술이 특히 중요하게 다뤄지며, 여러 계약서에서의 공통된 패턴을 학습하여 효율적으로 정보를 추출할 수 있는 방법을 모색합니다.

- **Performance Highlights**: 이 모델은 기존 수동 프로세스에 비해 정보의 정확도와 처리 속도를 높이는 데 크게 기여합니다. 특히, 트랜스포머 모델이 복잡한 언어 패턴 인식에 강점을 보이며, 실제 부동산 계약서 데이터를 사용한 테스트에서 높은 정확성과 효율성을 달성하였습니다.



### Fashion Recommendation: Outfit Compatibility using GNN (https://arxiv.org/abs/2404.18040)
- **What's New**: 이 연구는 의류 및 액세서리 아이템과 전체 착장을 표현하기 위해 다양한 그래프 기반 프레임워크(graph-based frameworks)를 사용하는 것을 탐구합니다. 주된 목적은 의류 추천 시스템에서 어떤 아이템이 착장과 잘 어울리는지를 파악하는 것입니다. 두 가지 그래프 신경망(GNN: Graph Neural Network) 구현을 활용하여 아이템들의 호환성을 평가하는 새로운 접근방식을 시도했습니다.

- **Technical Details**: Node-wise Graph Neural Network (NGNN)와 Hypergraph Neural Network (HGNN)라는 두 가지 접근 방식을 사용했습니다. NGNN은 아이템의 특성을 더 잘 포착하기 위해 파라미터 공유를 피하고, 항목의 카테고리를 중심으로 메시지를 전달하는 방식을 채택했습니다. 반면 HGNN은 하이퍼그래프(Hypergraph)를 사용하며, 이는 더 복잡한 아이템 간 상호작용을 포착할 수 있는 이론적 기반을 제공합니다.

- **Performance Highlights**: 실험 결과, HGNN은 NGNN보다 약간 더 나은 성능을 보였으며, 'Fill in the Blank(FITB)' 작업과 호환성 예측 작업에서 각각 1%와 11% 정도 향상된 정확도를 보였습니다. 뿐만 아니라, 다중 모달(multi-modal) 접근 방식이 텍스트만(text-only)이나 시각만(visual-only)의 접근 방식보다 더 나은 결과를 보여주었습니다.



### Quality Estimation with $k$-nearest Neighbors and Automatic Evaluation  for Model-specific Quality Estimation (https://arxiv.org/abs/2404.18031)
Comments: Accepted to EAMT 2024

- **What's New**: 새롭게 제안된 $k$NN-QE 모델은 기계 번역(Machine Translation, MT)의 출력에 대한 참조가 필요 없는 품질 평가(Quality Estimation, QE)를 제공하며, 이는 사용자가 번역의 신뢰성을 판단하는 데 도움을 줍니다. 이 모델은 기계 번역 모델의 학습 데이터에서 정보를 추출하여 품질 점수를 생성하는 비지도학습(unsupervised) 방식을 사용합니다.

- **Technical Details**: $k$NN-QE는 $k$-최근접 이웃($k$-nearest neighbors) 기법을 활용하여 품질 점수를 생성합니다. 참조 기반(reference-based) 메트릭스 MetricX-23를 활용하여 자동 평가 방법을 제안하고 있으며, 이는 인간이 생성한 품질 점수 대신 참조 기반의 품질 점수를 금본(골드 스탠다드)으로 사용합니다.

- **Performance Highlights**: $k$NN-QE 모델은 참조 기반 메트릭스 MetricX-23을 사용하여 품질을 자동으로 평가하며, 이 메트릭스가 이 작업에 가장 적합하다는 평가를 받았습니다. 또한, 비지도학습 방식에 대한 자세한 분석을 수행하여 지금까지의 방법이 충분함을 증명하였습니다.



### MediFact at MEDIQA-CORR 2024: Why AI Needs a Human Touch (https://arxiv.org/abs/2404.17999)
Comments: 7 pages, 4 figures, Clinical NLP 2024 Workshop

- **What's New**: 이 연구는 MEDIQA-CORR 2024 공유 작업에 제출된 새로운 접근법을 제시합니다. 클리니컬 노트의 단어 하나의 오류를 자동으로 수정하는 데 중점을 둔 접근법으로, 일반적인 데이터에 의존하는 기존의 대형 언어 모델(LLMs)과는 달리, 관련 클리니컬 텍스트 데이터에서 맥락적으로 관련 정보를 추출하는 데 중점을 둡니다. 이러한 방식은 의료 분야에 특화된 기능 공학(feature engineering)을 활용한 감독 학습(supervised learning) 프레임워크를 구축하고, 도메인 전문 지식을 통합하여 오류 수정의 정확성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: MediFact-CORR QA는 적은 양의 데이터로 효율적으로 오류를 수정할 수 있는 연구 방법론을 소개합니다. 이 방법론은 텍스트 데이터 내 클리니컬 오류의 고유 패턴을 식별하기 위해 약간의 감독(weakly-supervised) 학습을 사용합니다. 이는 SVM(Support Vector Machines)을 사용하여 정확하고 오류가 있는 문장을 구분하며, 추출 기반 질의 응답(extractive question-answering) 방식으로 조정된 문장을 식별하는 데에도 활용됩니다. 또한, 훈련 데이터에 명시적으로 나타나지 않은 오류에 대해서는 맞춤형 사전 훈련된 QA(question-answering) 모델을 사용하여 잠재적인 수정을 생성합니다.

- **Performance Highlights**: MediFact-CORR QA는 MEDIQA-CORR 2024 공유 작업을 위해 특별히 설계된 클리니컬 텍스트 데이터 세트를 사용하여 평가되었습니다. 훈련 세트와 검증 세트는 각각 2,189개의 텍스트와 574개의 텍스트를 포함하며, 테스트 세트는 925개의 텍스트로 구성되어 있습니다. 이 모델은 오류 플래그 예측, 오류 문장의 인덱스 감지 및 정확한 문장 생성과 같은 세 가지 과제에서 성능을 평가받았으며, 본 연구에서 제시된 데이터 효율적 솔루션이 클리니컬 텍스트 오류 수정에 얼마나 효과적인지를 입증했습니다.



### Enhancing Pre-Trained Generative Language Models with Question Attended  Span Extraction on Machine Reading Comprehension (https://arxiv.org/abs/2404.17991)
- **What's New**: 이 논문은 기계 독해 이해(Machine Reading Comprehension, MRC) 영역에서 발생하는 생성적 접근의 문제를 해결하기 위해 새로운 Question-Attended Span Extraction (QASE) 모듈을 소개합니다. 이 모듈은 기존의 생성적 언어 모델(generative language models, PLMs)에 통합되어 정확도를 높이고, 최첨단 Large Language Models (LLMs) 예를 들어 GPT-4보다 우수한 성능을 제공합니다.

- **Technical Details**: QASE는 문제 기반의 스팬 추출(question-attended span extraction) 도구로, 프리트레인된 생성적 언어 모델(pre-trained generative language models, PLMs)의 미세조정(fine-tuning) 단계에서 활용됩니다. 이 모듈은 Inside-Outside (IO) 태깅 스키마를 활용하여 단일 및 다중 스팬 추출을 처리합니다. 모델은 문맥과 질문의 쌍을 입력으로 받아, 다중 헤드 어텐션(multi-head attention, MHA) 메커니즘을 통해 질문과 관련된 문맥 토큰과의 관계를 포착합니다.

- **Performance Highlights**: QASE 모듈은 다양한 MRC 데이터셋에서 강력한 성능을 보여 주었습니다. 이 모듈을 사용한 시스템은 기존의 추출 방식보다 뛰어난 결과를 도출하며, 상태 최고 기술(state-of-the-art, SOTA) 결과를 일관되게 달성하거나 초과합니다. 또한, QASE는 추가적인 컴퓨팅 요구 사항 없이 성능을 향상시키는 비용 효율적인 솔루션을 제공합니다.



### Detection of Conspiracy Theories Beyond Keyword Bias in German-Language  Telegram Using Large Language Models (https://arxiv.org/abs/2404.17985)
Comments: Accepted to the 8th Workshop on Online Abuse and Harms (WOAH), ACL 2024

- **What's New**: 이 연구는 독일어 Telegram 메시지에서 음모론을 탐지하는 작업을 다룹니다. 이 연구는 특히 COVID-19 팬데믹 기간 동안 수집된 약 4,000개의 메시지 데이터셋을 사용하여, 기존의 키워드 기반 필터링을 사용하지 않고 음모론을 자동으로 탐지합니다. 기존 연구와 달리, 직접적인 교육 데이터(keyword-based dataset)의 필요성을 줄이면서도 높은 성능을 유지하는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 BERT (Bidirectional Encoder Representations from Transformers)-like 모델을 사용한 지도 학습(supervised fine-tuning) 방법과, GPT-3.5, GPT-4와 같은 고급 언어 모델(Language model)을 이용한 프롬프트 기반(prompt-based) 접근 방식을 비교 분석합니다. 프롬프트 기반의 접근 방법은 추가적인 훈련 데이터를 거의 또는 전혀 필요로 하지 않는 장점이 있습니다.

- **Performance Highlights**: 지도 학습 모델은 F1 스코어가 약 0.8로, 기존의 영어 데이터셋에서 훈련된 모델들과 비슷한 성능을 보여주었습니다. 이 모델은 시간에 따른 내부 도메인의 변화에도 적응하면서, F1 스코어 약 0.7을 달성했습니다. 프롬프트 기반 접근 방식 중에서는 GPT-4가 제로샷(zero-shot) 환경에서 음모론에 대한 맞춤형 정의를 사용하여 F1 스코어 약 0.8을 달성하여 가장 우수한 성능을 보였습니다.



### Automating Customer Needs Analysis: A Comparative Study of Large  Language Models in the Travel Industry (https://arxiv.org/abs/2404.17975)
- **What's New**: 자연어 처리(Natural Language Processing, NLP)의 빠르게 변화하는 환경에서 대용량 언어 모델(Large Language Models, LLMs)이 다양한 작업에서 강력한 도구로 부상했습니다. 이 연구에서는 TripAdvisor 게시물에서 여행 고객의 요구를 추출하는 작업에 대한 LLM의 비교 분석을 수행합니다. GPT-4와 Gemini와 같은 오픈 소스 및 독점 모델을 포함한 다양한 모델을 활용하여 이 특수 분야에서 각 모델의 장점과 단점을 밝히고자 합니다.

- **Technical Details**: 평가 과정에서 BERTScore, ROUGE, BLEU와 같은 지표를 사용하여 각 모델이 고객 요구를 정확하게 식별하고 요약하는 성능을 평가합니다. 특히 Mistral 7B와 같은 오픈 소스 대용량 언어 모델은 더 큰 폐쇄된 모델에 비해 비용 효율성과 커스터마이징(customization)의 이점을 제공하면서 비교할 만한 성능을 달성하는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 고객 요구 분석 작업에 가장 적합한 LLM을 선정할 때 모델 크기, 자원 요구 사항, 성능 지표 등을 고려하는 것이 중요하다는 점을 강조합니다. 또한, 이러한 고급 NLP 기술을 활용하여 여행 산업에서 고객 경험을 향상시키고 운영 효율성을 증대시키려는 기업에 소중한 통찰력을 제공합니다.



### Usefulness of Emotional Prosody in Neural Machine Translation (https://arxiv.org/abs/2404.17968)
Comments: 5 pages, In Proceedings of the 11th International Conference on Speech Prosody (SP), Leiden, The Netherlands, 2024

- **What's New**: 이번 연구에서는 자동 인식된 음성의 감정(emotion)이라는 새로운 외부 정보를 사용하여 신경 기계 번역(Neural Machine Translation, NMT)의 품질을 향상시키는 방법을 제안하였습니다. 부가된 감정 정보는 기존 번역 접근 방식에 없었던 새로운 차원을 추가하며, 이는 번역의 정확성과 품질을 개선하는 데 중요한 역할을 할 수 있습니다.

- **Technical Details**: 연구팀은 음성 감정 인식(Speech Emotion Recognition, SER) 모델을 선택하여 데이터셋의 모든 입력 오디오에서 차원적 감정 값(dimensional emotion values)을 예측하고, 이러한 감정을 입력 텍스트의 시작 부분에 소스 토큰(source tokens)으로 추가하여 NMT 모델을 훈련시켰습니다. 이 방법은 감정이 특정 어휘와 연관되어 있어 번역 시 그 효과가 증대될 수 있다는 가정에 기반합니다.

- **Performance Highlights**: 감정 정보, 특히 각성도(arousal)를 NMT 시스템에 통합함으로써 번역의 품질이 향상되었습니다. 아직 초기 단계의 연구이므로 구체적인 성능 지표는 제공되지 않지만, 감정의 차원을 고려하는 번역은 보다 정확하고 섬세한 결과를 도출할 수 있는 잠재력을 보여줍니다.



### Transfer Learning Enhanced Single-choice Decision for Multi-choice  Question Answering (https://arxiv.org/abs/2404.17949)
Comments: 10 pages, 1 figures.This article supersedes arXiv:2011.03292

- **What's New**: 본 연구에서는 다항 선택형 기계 독해 (MMRC; Multi-choice Machine Reading Comprehension) 문제를 단항 선택으로 재구성하여, 특정 답변이 정답인지를 구분하는 이진 분류(binary classification)를 훈련시키고, 가장 높은 확신 점수(confidence score)를 가진 옵션을 최종 답으로 선택하는 새로운 방법을 제안합니다. 이 방식은 다항 선택의 틀을 벗어나 다른 MRC 과제의 리소스를 활용할 수 있는 장점을 가집니다.

- **Technical Details**: 이 연구는 ALBERT-xxlarge 모델을 기반으로 하여 고안되었으며, RACE 및 DREAM 데이터셋에서 평가되었습니다. 기존의 MMRC 방법들은 주로 문장, 질문, 답변 간의 관계를 효과적으로 포착하기 위한 세심한 메커니즘(design of exquisite mechanisms) 설계에 중점을 두었지만, 본 연구는 MMRC 과제에 다른 MRC 과제들로부터의 지식 이전(knowledge transfer)을 가능하게 함으로써 새로운 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존의 다항 선택 방법들보다 우수한 성능을 보여주었습니다. 또한, 다른 종류의 MRC 과제들로부터 지식을 이전함으로써 단일(single setting) 및 앙상블(ensemble setting) 설정에서 최고의 성능(state-of-the-art results)을 달성하였습니다.



### I Have an Attention Bridge to Sell You: Generalization Capabilities of  Modular Translation Architectures (https://arxiv.org/abs/2404.17918)
- **What's New**: 이 연구는 기계 번역(Machine Translation, MT) 분야에서 모듈형 아키텍처(Modular Architectures)의 효과를 탐구합니다. 과거에는 언어에 관계없는 표현(language-independent representations)을 생성하는 것이 중요한 목표였습니다. 최근에는 각 언어의 특성을 더 잘 포착할 수 있는 모듈형 접근이 강조되었습니다. 특히, '어텐션 브리지(Attention Bridges)'와 같은 모듈형 구성요소가 제로 샷 일반화(Zero-Shot Generalization) 및 언어 독립적 표현을 촉진할 수 있다고 주장되었습니다.

- **Technical Details**: 연구는 6개의 아키텍처를 분석하며, 그 중 5개는 모듈형입니다. 분석된 아키텍처들은 전부 Transformer 기반으로 구현되었습니다. 이 중에서도 언어별 특화 파라미터와 언어 독립 파라미터를 혼합한 모델, 예를 들어 단일 공유 인코더(Single Shared Encoder)를 사용하여 모든 소스 언어에서의 학습 신호를 활용하거나, 모든 목표 언어를 위해 단일 공유 디코더(Single Shared Decoder)를 사용하는 방식이 포함됩니다. 또한, '고정 크기 어텐션 브리지(Fixed-Size Attention Bridge, FSAB)' 디자인을 사용한 모델도 탐구되었습니다.

- **Performance Highlights**: 모듈형 시스템은 모든 번역 방향이 가능할 때 비모듈형(Fully-Shared) 시스템과 유사하거나 선호될 수 있지만, 특정 집단의 언어만 사용할 경우 그 효과가 제한적이라는 결과를 발견하였습니다. FSAB를 포함한 일부 모듈형 설계는 제로 샷(Zero-Shot)이나 분포 외(Out-Of-Distribution, OOD) 조건에서는 경쟁력이 떨어지는 것으로 나타났습니다. 사실, 완벽하게 공유된 부분 네트워크에서 일반화 능력을 향상시킬 수 있는지는 여전히 의문입니다.



### SERPENT-VLM : Self-Refining Radiology Report Generation Using Vision  Language Models (https://arxiv.org/abs/2404.17912)
Comments: 8 pages, 3 figures, 4 tables, Accepted as oral at Clinical NLP workshop at NAACL 2024

- **What's New**: 본 논문에서는 정교한 방사선학적 보고서 생성을 위한 새로운 접근 방식인 SERPENT-VLM(SElf Refining Radiology RePort GENeraTion using Vision Language Models)을 소개합니다. 이 방법은 Multi-modal Large Language Models(MLLMs)의 자가정제 기능을 통합하여 텍스트 기반 보고서에서 이미지 내용을 정확하게 반영하지 못하는 문제를 개선합니다.

- **Technical Details**: SERPENT-VLM은 이미지 표현과 생성된 방사선학적 텍스트의 문맥 표현 간의 유사성을 활용하는 고유한 자기감독 손실(self-supervised loss)을 채택합니다. 이는 이미지-텍스트 표현을 세밀하게 정제하고, 생성된 텍스트를 지속적으로 조정하여 보다 정확한 보고서 생성을 가능하게 합니다. 기존의 대표 모델들(LlaVA-Med, BiomedGPT 등)과 비교하여 IU X-ray 및 Radiology Objects in COntext (ROCO) 데이터셋에서 최고의 성능(State of the Art, SoTA)을 달성했습니다.

- **Performance Highlights**: SERPENT-VLM은 기존 방법보다 정확성, 효율성 및 견고성(benchmark)에서 새로운 기준을 설정하였습니다. 특히 주요 평가 척도인 Bleu, RougeL, BertScore에서 뛰어난 성능을 보였으며, 잡음이 많은 이미지에 대해서도 강인함을 입증했습니다. 방사선학적 리포트 생성에서의 환각(hallucination)을 줄이는 데 중요한 진전을 이루었습니다.



### Tool Calling: Enhancing Medication Consultation via Retrieval-Augmented  Large Language Models (https://arxiv.org/abs/2404.17897)
- **What's New**: 이 연구는 의료 분야에서 지식 집약적 작업에 대해 대규모 언어 모델(Large-scale Language Models, LLMs)을 적용하는 것을 탐색합니다. 특히, 실제 약물 상담 시나리오를 모방하는 다중 라운드 대화 벤치마크인 MedicineQA를 소개하여 추출된 증거를 사용하여 LLMs의 성능을 평가합니다.

- **Technical Details**: 이 연구에서는 기존의 'Retrieve-then-Read' 프레임워크 대신 새로운 'Distill-Retrieve-Read'을 제안하며, 이 과정에서 검색 쿼리를 형성해 키워드 기반 검색 쿼리를 모방하는 도구 호출 메커니즘을 활용합니다. MedicineQA 데이터베이스는 300개의 다중 라운드 질문-응답 쌍을 포함하고 있으며, 각 질문은 자세한 대화 이력 내에 내장되어 있습니다.

- **Performance Highlights**: 실험 결과, 이 새로운 프레임워크는 증거 검색 정확도 측면에서 이전 모델을 능가하는 주목할만한 성능 향상을 가져왔습니다. 이는 의학 분야에 RAG(Retrieval-augmented generation)를 적용할 수 있는 가능성을 밝히는 발전입니다.



### PromptCL: Improving Event Representation via Prompt Template and  Contrastive Learning (https://arxiv.org/abs/2404.17877)
Comments: NLPCC 2023 Best Student Paper

- **What's New**: 이 연구는 사건 표현 학습을 위한 새로운 접근 방식인 PromptCL을 소개하며, Pre-trained Language Models (PLMs)에서 사건 이해 능력을 끌어내기 위해 프롬프트 학습 및 대조 학습을 결합합니다. PromptCL은 사건 텍스트 길이의 제한을 극복하고, 사건 구성 요소 간의 관계를 더 잘 이해할 수 있도록 돕는다는 점에서 기존 방법들과 차별화됩니다.

- **Technical Details**: PromptCL 프레임워크는 Prompt 템플릿 (Prompt template)을 사용하여 입력 텍스트를 확장하고, Subject-Predicate-Object (SPO) 단어 순서와 Event-oriented Masked Language Modeling (EventMLM)을 도입하여 PLM의 학습을 지원합니다. 이는 사건 텍스트의 자연스러운 언어 순서와 일치하도록 설계되었으며, 각 사건 구성 요소를 전체 단어로 마스킹하여 보다 깊은 이해를가능하게 합니다.

- **Performance Highlights**: 실험 결과, PromptCL은 사건 관련 작업에서 최신(State-of-the-art) 기초 모델들을 능가하는 성능을 보여줍니다. 또한, 프롬프트를 사용한 결과는 사건 표현의 일반화 능력이 향상되었다는 것을 보여주는 철저한 분석을 제공합니다.



### From Languages to Geographies: Towards Evaluating Cultural Bias in Hate  Speech Datasets (https://arxiv.org/abs/2404.17874)
Comments: Accepted at WOAH (NAACL 2024)

- **What's New**: 이 연구는 언어와 지리적 메타데이터를 이용하여 혐오 발언(Hate Speech, HS) 데이터셋의 문화적 편향을 분석한다. 특히, 영어, 아랍어, 스페인어와 같은 지리적으로 널리 퍼진 언어를 중심으로, 이 언어들이 사용되는 다양한 국가들에서 얼마나 대표되었는지를 조사한다. 이러한 분석을 통해, 일부 국가들이 HS 데이터셋에서 크게 과대표되어 있음을 발견하고, 이러한 문제를 해결하기 위한 새로운 방법론을 제안한다.

- **Technical Details**: 연구자들은 HS 데이터셋에서 문화적 편향을 분석하기 위해 언어와 국가 정보를 결합한 지오-문화적 컨텍스트(geo-cultural contexts)를 사용한다. 총 8개 언어(아랍어, 영어, 프랑스어, 독일어, 인도네시아어, 포르투갈어, 스페인어, 터키어)의 데이터셋을 시스템적으로 검토하여, 특히 영어 데이터셋의 지배가 점차 감소하고 있음을 확인했다. 또한, 트위터에서 얻은 지리적 메타데이터를 활용하여 데이터의 기원을 추적하고, 특정 국가들(예: 영어는 미국과 영국, 스페인어는 칠레와 스페인, 아랍어는 요르단)에서 사용자들이 과도하게 대표되어 있는 문제를 지적한다.

- **Performance Highlights**: 이 연구는 언어별로 HS 데이터셋에서의 문화적 편향을 체계적으로 분석하고, 영어의 지배적 위치가 줄어들고 있다는 흥미로운 결과를 제시한다. 문화적 편향을 줄이기 위한 구체적인 권장사항을 제시하며, 이는 연구자들이 더 다양한 문화적 배경을 고려하여 HS 데이터셋을 구축할 수 있게 하는 중요한 기반을 마련한다.



### Revisiting Multimodal Emotion Recognition in Conversation from the  Perspective of Graph Spectrum (https://arxiv.org/abs/2404.17862)
Comments: 10 pages, 4 figures

- **What's New**: 이 논문은 대화에서 감정 인식을 위해 다중 모달 (Multimodal) 일관성 및 보완적 특성을 효율적으로 학습하는 새로운 프레임워크인 GS-MCC를 제안합니다. 전통적인 GNN 방법의 한계를 극복하기 위해, 이 연구는 그래프 스펙트럼 (Graph Spectrum) 관점에서 문제를 재조명하며, 고주파 및 저주파 정보의 협력적 학습을 통해 감정 인식의 정확성을 향상시키는 방법을 탐구합니다.

- **Technical Details**: GS-MCC 프레임워크는 RoBERTa, OpenSMILE, 3D-CNN을 사용하여 초기의 텍스트, 청각적, 시각적 특징을 추출하고, GRU 및 완전 연결 네트워크를 사용하여 더 높은 차원의 발화 표현을 얻습니다. 대화 관계를 모델링하고 장거리 의존성 정보를 효율적으로 캡처하기 위해 슬라이딩 윈도우를 사용하여 완전 연결 그래프를 구축하고, 효율적인 푸리에 그래프 연산자를 사용하여 고주파 및 저주파 정보를 각각 추출합니다. 또한, 대조학습 (Contrastive Learning)을 사용하여 고주파 및 저주파 신호와의 일관성 및 보완적 의미 협력을 반영하는 자기 감독 신호를 구성하여 실제 감정을 반영하는 고주파 및 저주파 정보의 능력을 향상시킵니다.

- **Performance Highlights**: GS-MCC는 벤치마크 데이터 세트에서 우수한 성능을 입증하였으며, 이는 특히 장거리 의존성과 고주파 및 저주파 정보의 효율적인 협력을 통한 감정 인식의 정확도를 증가시킵니다. 이를 통해 기존의 GNN 기반 방법에서 나타나는 과도한 평활화와 저대역 필터링 문제를 효과적으로 극복하고 있음을 보여줍니다.



### Revisiting Multi-modal Emotion Learning with Broad State Space Models  and Probability-guidance Fusion (https://arxiv.org/abs/2404.17858)
Comments: 10 pages, 6 figures

- **What's New**: 본 논문은 대화에서의 감정 인식을 위해 새로운 모델인 'Broad Mamba'를 제안합니다. 이 모델은 자기주의 메커니즘(self-attention mechanism)에 의존하지 않고, State Space Model(SSM)과 넓은 학습 시스템(broad learning system)을 이용하여 감정 표현을 압축하고 데이터 분포를 탐색합니다. 또한, 양방향 SSM 컨볼루션(bidirectional SSM convolution)을 설계하여 전역 컨텍스트 정보를 추출합니다.

- **Technical Details**: Broad Mamba는 특징 분리 단계에서 멀티모달 특징에서 감정적 특징과 가장 관련 있는 문맥 의미 정보를 추출합니다. 이는 SSM을 통해 데이터에 의존하는 전역 감정 컨텍스트 모델링을 하며, 넓은 공간에서 데이터 분포의 잠재력을 탐색하기 위한 넓은 학습 시스템을 결합합니다. 특징 융합 단계에서는 각 모달 특징의 예측 레이블 확률을 가중치 벡터로 사용하는 확률 가이드 융합 기법(probability-guided fusion mechanism)을 제안하여 다중 모달 컨텍스트 특징 융합을 달성합니다.

- **Performance Highlights**: 제안된 방법은 IEMOCAP 및 MELD 데이터 세트에서 광범위한 실험을 수행하였고, 심층적이고 효율적인 학습을 가능하게 하여 Transformer나 GNN 아키텍처보다 뛰어난 성능을 보였습니다. 특히, 긴 거리 컨텍스트의 모델링에서 크고 복잡한 계산이 요구되는 Transformer의 한계를 극복하고 메모리 한계를 보완하여 효율성과 효과성을 입증했습니다.



### Toxicity Classification in Ukrainian (https://arxiv.org/abs/2404.17841)
Comments: Accepted to WOAH, NAACL, 2024. arXiv admin note: text overlap with arXiv:2404.02043

- **What's New**: 이 연구는 톡시시티(Toxicity, 독성) 감지 작업이 안전하고 공정한 LMs(Language Models, 언어 모델) 개발 맥락에서 여전히 중요한 작업임을 밝히고 있습니다. 특히, 우크라이나어와 같이 톡시시티 분류 말뭉치가 없는 언어에 주목하며, 이를 해결하기 위해 교차 언어 지식 전이 기술을 연구하고 있습니다.

- **Technical Details**: 연구팀은 영어 말뭉치를 번역하고(Translating from an English corpus), 키워드를 사용하여 독성 샘플을 필터링(Filtering toxic samples using keywords) 한 후, 군중 소싱을 통해 주석을 달아(Annotating with crowdsourcing) 새로운 레이블이 지정된 말뭉치를 생성했습니다. 또한, LLMs(Language Large Models) 프롬프팅과 다른 교차 언어 전이 접근 방식을 비교 분석했습니다.

- **Performance Highlights**: 이 분석을 통해 다양한 접근 방식의 강점과 효율성에 대한 통찰을 제공하며, 특히 교차 언어 전이 후의 미세 조정(Fine-tuning) 유무에 따른 LLMs의 성능 차이를 비교하여 가장 로버스트하고 효율적인 기준을 제안합니다.



### VANER: Leveraging Large Language Model for Versatile and Adaptive  Biomedical Named Entity Recognition (https://arxiv.org/abs/2404.17835)
- **What's New**: 이 연구에서는 최신 대형 언어 모델(LLM)인 LLaMA2를 사용하여 생물의학 명명 엔티티 인식(BioNER) 문제에 접근합니다. 저자들은 기존의 순차 라벨링(Sequence Labeling) 방식과 지시어(instruction) 기반 튜닝을 결합하여 다양한 데이터셋에서 다양한 유형의 엔티티를 추출할 수 있는 VANER 모델을 제안합니다. 특히, 이 모델은 외부 의료 지식 베이스와 통합되어 생물학적 엔티티에 대한 이해를 높이며, 기존의 LLM 기반 모델을 뛰어넘는 성능을 보여줍니다.

- **Technical Details**: VANER 모델은 생물학 분야 특화 지식이 부족한 일반 LLM에, 지시어에 따른 튜닝(instruction tuning)과 순차 라벨링 기법을 통합합니다. 이를 통해 모델은 다양한 데이터셋의 지시어를 이해하고, UMLS와 같은 외부 지식 베이스를 이용하여 특정 엔티티를 더 정확하게 식별할 수 있습니다. 또한, 이 연구에서는 정보 흐름을 제한하는 인과 마스크(causal mask)를 제거하여, 토큰 간 상호작용을 통한 더 나은 엔티티 인식을 가능하게 합니다.

- **Performance Highlights**: VANER는 세 가지 데이터셋에서 가장 높은 F1 스코어를 달성함으로써, 기존의 state-of-the-art(SOTA) BioNER 시스템들을 상당 부분 초과하는 성능을 보여줍니다. 이는 LLM 기반 모델이 처음으로 전통적인 최고 성능 시스템들을 능가한 사례입니다. 효율적인 모델 설계 덕분에 단일 NVIDIA 4090 GPU에서 훈련과 추론이 가능합니다.



### Evaluation of Few-Shot Learning for Classification Tasks in the Polish  Languag (https://arxiv.org/abs/2404.17832)
Comments: 34 pages, 3 figures, 10 tables

- **What's New**: 이 연구는 폴란드어 관련 7가지 다른 분류 작업으로 구성된 퓨샷 벤치마크(few-shot benchmark)를 소개합니다. 이 벤치마크를 사용하여, 0샷과 16샷 상황에서 파인튜닝(fine-tuning), 리니어 프로빙(linear probing), SetFit, 인-컨텍스트 러닝(In-context Learning, ICL) 등 여러 기법을 비교하였습니다.

- **Technical Details**: 연구진은 GPT-3.5, GPT-4와 같은 상업용 모델과 오픈소스 모델을 사용하여 성능을 비교 분석했으며, ICL이 가장 뛰어난 성과를 보였습니다. 그러나 폴란드어 전체 훈련 데이터셋으로 파인튜닝된 HerBERT-large 모델과 비교했을 때 14퍼센트포인트의 성능 차이가 있었습니다. SetFit은 두 번째로 효과적인 접근법으로 나타났고, 리니어 프로빙도 밀접하게 뒤따랐습니다. 비선형 헤드 파인튜닝(non-linear head fine-tuning)은 가장 안정성이 낮고 성능이 떨어졌습니다.

- **Performance Highlights**: ICL 결과는 폴란드어 말뭉치에 대한 지속적인 프리트레이닝(pre-training)이 유익함을 보여주며, Bielik-7b와 Trurl-13b 모델의 성능 향상을 통해 이를 확인할 수 있습니다. 또한, 폴란드어 퓨샷 학습 사용을 지원하기 위해 ICL을 위한 수작업 템플릿을 제공할 예정입니다.



### Recall, Retrieve and Reason: Towards Better In-Context Relation  Extraction (https://arxiv.org/abs/2404.17809)
Comments: IJCAI 2024

- **What's New**: 이 연구에서는 관계 추출 (Relation Extraction, RE) 분야에서 대규모 언어 모델 (Large Language Models, LLMs)의 인-컨텍스트 학습 (In-Context Learning, ICL) 능력을 향상시키기 위해 새로운 RE4 (Recall-Retrieve-Reason) 프레임워크를 제안합니다. 이 방법은 학습 예제로부터 일관된 온톨로지적 지식을 추출하고, 이를 바탕으로 관련성 높은 엔티티 쌍을 생성한 다음, 이를 사용하여 학습 예제에서 유의미한 데모를 검색하여 ICL을 개선합니다.

- **Technical Details**: RE4는 RE에 대해 특정한 온톨로지 지식 (ontological knowledge)을 가진 엔티티 쌍을 생성하는 '회상(recalling)' 모듈, 이를 바탕으로 관련 훈련 예제를 검색하는 '검색(retrieval)' 모듈, 그리고 검색된 데모에서 인-컨텍스트 추론을 수행하는 '추론(reasoning)' 모듈로 구성됩니다. 이 프레임워크는 기존의 단순 유사도 기반 검색 방법을 개선하여 실제 관련성 높은 데모를 추출하고, 이를 통해 더 정확한 인-컨텍스트 추론을 가능하게 합니다.

- **Performance Highlights**: RE4는 다양한 대규모 언어 모델과 관계 추출 데이터셋에서 효과적인 성능을 보였습니다. 경쟁력 있는 또는 최신 기술(State-of-the-Art) 수준의 성능을 제공하면서, 관련성 높은 엔티티 쌍 생성과 ICL 능력 강화에 기여합니다. 특히, 이 방법은 문장 레벨(sentence-level) RE에서 뛰어난 결과를 도출했으며, 다양한 ICL 기반 방법론들과 비교하여 우수한 성능을 나타냈습니다.



### Scaffold-BPE: Enhancing Byte Pair Encoding with Simple and Effective  Scaffold Token Remova (https://arxiv.org/abs/2404.17808)
- **What's New**: 이 연구에서는 기존 BPE(Byte Pair Encoding) 알고리즘의 주요 한계점을 타개한 새로운 방법인 Scaffold-BPE를 제안합니다. Scaffold-BPE는 동적인 Scaffold 토큰 제거 메커니즘을 도입하여 낮은 빈도의 토큰(Scaffold 토큰)을 자동으로 식별하고 인코딩 과정에서 제외시켜, 토큰 불균형 문제를 해결하고 언어 모델의 학습을 촉진합니다.

- **Technical Details**: Scaffold-BPE는 인코딩 과정에서 사용되는 확장된 어휘(vocabulary)를 이용하여 주어진 텍스트의 토큰 표현을 생성한 뒤, Scaffolding 과정을 거쳐동적으로 낮은 빈도의 Scaffold 토큰을 식별하고 제거하는 Demolishing 과정을 통해 최종 토큰 표현에서 Scaffold 토큰을 배제합니다. 이 방식은 파라미터 프리(parameter-free), 계산 부하가 낮은(computation-light), 그리고 구현하기 쉬운(easy-to-implement) 점이 특징입니다.

- **Performance Highlights**: Scaffold-BPE는 언어 모델링(Language Modeling) 작업 및 기계 번역(Machine Translation) 작업에서 모두 기존의 BPE 방식을 일관되게 능가하는 성능을 보였습니다. 언어 모델링 벤치마크와 WMT’14 영어-독일어, 영어-프랑스어 번역 작업에서의 실험에서 Scaffold-BPE의 우수성이 입증되었습니다.



### Meta In-Context Learning Makes Large Language Models Better Zero and  Few-Shot Relation Extractors (https://arxiv.org/abs/2404.17807)
Comments: IJCAI 2024

- **What's New**: 새로운 메타 인-컨텍스트 학습 프레임워크인 Micre (Meta In-Context learning of LLMs for Relation Extraction)가 제안되었습니다. Micre는 대규모 언어 모델(LLMs)이 제로샷 및 로우샷(Relation Extraction, RE) 과제를 보다 효과적으로 수행할 수 있도록 하며, 특정 프롬프트나 추가적인 매개 변수 업데이트 없이도 새로운 RE 작업에서의 컨텍스트 학습을 가능하게 합니다.

- **Technical Details**: Micre는 여러 RE 데이터셋을 이용하여 LLM을 튜닝하고, 적은 수의 학습 예제만을 제공해도 높은 성능을 나타낼 수 있도록 구성되어 있습니다. LLM은 메타 트레이닝(Meta-Training)을 통해 새로운 RE 과제를 인지하고 이를 내부적으로 학습합니다. 이 연구는 자연어 처리(Natural Language Processing, NLP)에서의 인-컨텍스트 학습(In-Context Learning, ICL)과 메타 학습(Meta-Learning)의 아이디어를 결합하여, 과제 특화 학습 없이도 우수한 성능을 발휘할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: Micre는 다양한 크기의 LLM 모델을 사용하여 12개의 공개 RE 데이터셋에 적용되었으며, 이를 통해 보다 우수한 성능을 달성했음을 보여줍니다. 특히, 대규모 모델에서 뚜렷한 성능 개선을 보였으며, 다양한 RE 데이터셋을 메타 학습에 사용한 것이 중요한 요소로 작용했습니다. 또한, Micre는 제로샷 및 로우샷 설정에서 관측된 벤치마크에 대해 기존의 방식과 비교하여 경쟁력 있는 성능을 보였습니다.



### Empirical Analysis of Dialogue Relation Extraction with Large Language  Models (https://arxiv.org/abs/2404.17802)
Comments: IJCAI 2024

- **What's New**: 이 논문은 대화에서 두 인자 사이의 관계를 추출하는 대화 관계 추출(Dialogue Relation Extraction, DRE)에 초점을 맞추었습니다. 특히, 대규모 언어 모델(Large Language Models, LLMs)의 성능을 탐구하며 DRE 문제를 해결하기 위한 새로운 방법론을 제안합니다. 이러한 LLMs는 기존의 방법들이 겪는 여러 문제들을 효과적으로 완화시키는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 대화 기반 모델과 그래프 기반 방법을 넘어서서 생성 기반 방법을 활용합니다. 특히 ChatGPT와 같은 소유권 모델(Proprietary Models)과 LLaMA 같은 오픈 소스 모델을 포함한 다양한 LLMs을 평가합니다. 이러한 LLMs은 대화의 전체적인 맥락이 아닌 부분적인 대화 상황에서도 높은 성능을 유지하는 것으로 나타났으며, 다기(turn) 정보를 처리하는 능력이 탁월하다는 것을 확인했습니다.

- **Performance Highlights**: LLMs는 기존의 DRE 방법들보다 더 나은 결과를 보여주며, 특히 전체 대화 및 몇 가지 데이터셋에서만 작동하는 설정(Full-shot and Few-shot Settings)에서 경쟁력 있는 성능을 보였습니다. 또한, LLMs는 긴 대화와 다양한 길이의 대화에서 뛰어난 성능을 보이며, 일부 대화 설정에서 큰 성능 저하 없이 일관된 성능을 유지합니다.



### Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing  Japanese Language Capabilities (https://arxiv.org/abs/2404.17790)
- **What's New**: 이 연구에서는 영어 데이터로 사전 훈련된 큰 언어 모델(Large Language Models, LLM)에 일본어 지속적 사전 훈련(continual pre-training)을 적용하여 'Swallow'라는 일본어 능력이 향상된 모델을 개발하였습니다. 이는 비용과 자원을 절약하면서 일본어 태스크에서의 성능을 획기적으로 개선할 수 있는 방법을 제시합니다.

- **Technical Details**: Swallow 모델은 Llama 2의 어휘를 확장해 일본어 문자를 포함시키고, 대규모 일본 웹 코퍼스(corpus)에서 계속된 사전 훈련을 수행하여 구축되었습니다. 이 연구에서는 어휘 확장(vocabulary expansion)과 일본어-영어 병렬 코퍼스(parallel corpora)의 효과도 분석했습니다. 특히, 병렬 코퍼스 사용은 번역 능력을 향상시키는 데 효과적이었지만, 자동 요약 태스크에서는 성능 저하의 유일한 요인으로 지적되었습니다.

- **Performance Highlights**: Swallow는 일본어 태스크, 특히 일본어 질문 응답 태스크에서 높은 성능을 보였으며, 훈련 데이터의 양이 증가함에 따라 일본어 성능이 단조롭게(monotonically) 향상되었습니다. 또한, 훈련된 데이터의 양이 100B 토큰(token)까지 모델 성능이 지속적으로 증가하는 것을 확인하였습니다. 이러한 접근 방식은 계산 자원을 적게 사용하면서도 더 높은 성능을 달성할 수 있음을 보여줍니다.



### Temporal Scaling Law for Large Language Models (https://arxiv.org/abs/2404.17785)
Comments: Work in progress

- **What's New**: 이 논문은 큰 규모의 언어 모델(Large Language Models, LLMs)을 대상으로 전에는 연구되지 않았던 '시간적 스케일링 법칙(Temporal Scaling Law)'을 도입하고 있습니다. LLMs의 훈련 과정 동안 손실(Loss)이 어떻게 변화하는지 시간적 차원에서 연구하여, 이러한 변화가 모델의 스케일과 훈련 단계에 따라 어떻게 달라지는지를 탐구합니다. 이는 LLM의 미래 훈련 단계에서의 성능을 예측하는 데 사용될 수 있는 새로운 방법론을 제공합니다.

- **Technical Details**: 연구진은 토큰 위치(Token Positions)마다의 손실 불균형을 조사하고, 모델 규모 및 훈련 단계에 걸쳐서 상호 법칙(Reciprocal-law)을 개발했습니다. 이러한 법칙의 매개변수 변화를 시간에 따라 연구하여, '시간적 스케일링 법칙'을 정립했습니다. 이는 실제 LLM의 테스트 손실(Test Loss)을 초기 훈련 기간의 데이터만을 사용하여 미래의 테스트 손실을 예측하는 데 활용됩니다.

- **Performance Highlights**: 제안된 방법론은 기존 방법들에 비해 상당한 성능 향상을 보였으며, In-distribution (IID) 데이터와 Out-of-distribution (OOD) 데이터 모두에서 유효함을 확인했습니다. 이는 토큰 위치에 따른 학습의 균일성을 보여주며, 다양한 규모의 LLM을 사전 훈련할 때 기본 훈련 패러다임의 유효성을 검증합니다.



### Medical Vision-Language Pre-Training for Brain Abnormalities (https://arxiv.org/abs/2404.17779)
- **What's New**: 이 연구는 공개된 자료와 PubMed에서 뇌 이상 데이터를 자동으로 수집하는 파이프라인을 개발하여 의료 분야에 특화된 시각-언어(Vision-Language, VL) 모델을 사전 훈련(pre-training)에 사용하고자 합니다. 의료 이미지와 텍스트 자료를 수집하여 높은 성능의 특화된 VL 모델을 구축하는 방법론을 제시합니다. 또한, 의료 영역에서의 세부 이미지(subfigures)와 세부 캡션(subcaptions) 매핑의 독특한 문제도 조사합니다.

- **Technical Details**: 연구팀은 PubMed에서 약 22,000쌍의 이미지-캡션 데이터를 수집하고, 이를 분석하여 약 39,000쌍의 세부 이미지/세부 캡션 쌍으로 변환했습니다. BLIP 모델을 사용하여 데이터를 사전 처리하였고, OCR(tool111google OCR) 도구를 활용하여 세부 이미지에서 세부 캡션과 매칭할 수 있는 문자를 인식했습니다. 또한, Adam 최적화 알고리즘과 대조 손실(contrastive loss)을 사용하여 모델을 학습하고, 내부적으로 이루어진 평가를 통해 모델의 성능을 검증했습니다.

- **Performance Highlights**: 사전 훈련된 모델은 기존 베이스라인 모델들과 비교하여 이미지-텍스트 매칭에서 뛰어난 성능을 보였으며, 이미지 텍스트 검색(Image-text retrieval) 작업에서도 높은 성능을 보여줬습니다. '@1'과 '@10' 설정에서의 평가 결과도 모델의 유효성을 입증하며, 어텐션 맵(attention map)을 통한 질적 분석에서도 의료 용어와 관련된 이상 지역에 대한 주목을 효과적으로 시각화하여 모델이 의료 영역에서의 유효함을 확인할 수 있습니다.



### MRScore: Evaluating Radiology Report Generation with LLM-based Reward  System (https://arxiv.org/abs/2404.17778)
- **What's New**: 최근 연구에서 자동 방사선 보고서 생성을 평가하는 새로운 메트릭, MRScore를 소개합니다. 이는 기존의 NLG (Natural Language Generation) 메트릭이 직면한 한계를 극복하고자 Large Language Models (LLMs)를 활용한 접근 방식입니다. MRScore는 방사선 전문의와 협력하여 개발되었으며, 인간의 평가와 높은 상관 관계를 보이는 것으로 나타났습니다.

- **Technical Details**: MRScore는 GPT와 같은 LLM을 사용하여 대량의 훈련 데이터를 생성하는 두 가지 주요 구성 요소를 포함합니다: 1) 다양한 품질의 보고서를 생성하고, 2) 생성된 보고서를 ‘허용된(accepted)’ 및 ‘거부된(rejected)’ 샘플로 분류하여 LLM을 훈련시키는 것입니다. 이 평가 메트릭은 보고서의 품질을 자동으로 평가하기 위해 기계 학습 모델을 미세 조정(fine-tuning)하여 사용합니다. 개발된 메트릭과 데이터셋은 GitHub에서 공개될 예정입니다.

- **Performance Highlights**: MRScore는 기존의 메트릭들(BLEU, METEOR 등)에 비해 인간의 판단과 더 높은 상관관계를 보였으며, 모델 선택에서도 우수한 성능을 나타냈습니다. GPT-4를 사용하여 생성된 3,000개의 평가 샘플을 통해 훈련된 이 평가 모델은, 다양한 품질의 보고서를 평가하는 데 탁월한 성능을 보였습니다. 또한, MRScore는 방사선 보고서의 의료적 정확성과 언어적 다양성을 모두 고려한 종합적인 평가가 가능합니다.



### Building a Large Japanese Web Corpus for Large Language Models (https://arxiv.org/abs/2404.17733)
Comments: 17 pages

- **What's New**: 이 연구에서는 Common Crawl 아카이브에서 일본어 텍스트를 추출하고 정제하여 대규모 일본어 웹 코퍼스를 구축합니다. 이 코퍼스는 약 3121억 문자(약 1억 7300만 페이지)로, 일본어 대규모 언어 모델(LLMs)용 훈련 데이터로는 가장 큰 규모입니다. 이는 기존의 CC-100, mC4, OSCAR 23.10 등의 코퍼스를 뛰어넘는 것입니다.

- **Technical Details**: 이 새로운 코퍼스는 Common Crawl의 21개 스냅샷에서 추출한 후, 높은 품질의 일본어 텍스트를 선별하기 위한 필터링 방법을 통해 만들어졌습니다. 다양한 기존 언어 모델들(Llama 2, Mistral, Mixtral)에 대해 지속적인 사전 훈련을 실시하여 일본어 벤치마크 데이터셋에서 6.6-8.1 포인트의 일관된 성능 향상을 확인했습니다.

- **Performance Highlights**: 특히, Llama 2 13B 모델에서는 다른 코퍼스에 비해 가장 큰 성능 향상을 보였습니다. 연구 결과는 철저한 필터링과 언어 탐지를 통한 정제 과정이 모델의 학습 효율과 결과 품질을 크게 향상시킬 수 있음을 보여줍니다.



### CoMM: Collaborative Multi-Agent, Multi-Reasoning-Path Prompting for  Complex Problem Solving (https://arxiv.org/abs/2404.17729)
Comments: Accepted to NAACL 2024

- **What's New**: 본 연구에서는 대규모 언어 모델(LLM: Large Language Models)의 추론 능력 상한선을 확장하기 위해 협력적인 다중-에이전트, 다중-추론-경로(CoMM: Collaborative Multi-Agent, Multi-Reasoning-Path) 프롬프팅 프레임워크를 제안합니다. CoMM 프레임워크는 LLM들이 문제 해결 팀에서 다른 역할을 수행하도록 함으로써, 각각의 역할을 수행하는 에이전트들이 협력적으로 문제를 해결할 수 있도록 유도합니다.

- **Technical Details**: 본 논문에서는 다양한 추론 경로를 적용하여 도메인 전문 지식(domain knowledge)이나 특정 과제 해결 업무(task-solving duties)를 가진 다양한 역할로 LLM들을 프롬프팅하는 새로운 방식을 소개합니다. 또한 이런 다중-에이전트 환경에서 few-shot 학습을 가능하게 하는 다중-경로 추론 방법을 제안하여, 복잡한 과학 문제를 효과적으로 해결하는 데 기여합니다.

- **Performance Highlights**: 실험 결과는 대학 수준의 복잡한 과학 문제에 대해 제안된 방법이 경쟁적인 기준선들(competitive baselines)을 현저히 능가한다는 것을 보여줍니다. 'Chain-of-thought' 프롬프팅 방법과 달리, CoMM 접근 방식은 문제를 더 정확하게 이해하고 처리할 수 있는 여러 에이전트의 협력적인 노력을 통해 향상된 결과를 얻는 것을 가능하게 합니다.



### PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction  in Murder Mystery Games (https://arxiv.org/abs/2404.17662)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전은 에이전트(Agent)가 의사소통을 하고 사회적 상호작용을 하는 데 효과를 증대시키고 있음. 그러나 경쟁과 협력을 포함하는 동적 환경에서 이러한 LLM 기반 에이전트를 구축하는 것은 여전히 도전적인 과제. 이에 PLAYER*라는 새로운 프레임워크를 제안, 언제나(anytime) 샘플링 기반 계획자를 기반으로 센서와 프루너(pruners)를 사용하여 복잡한 추론 작업을 위한 순수하게 질문 중심의 탐색 프레임워크를 가능하게 함.

- **Technical Details**: PLAYER*는 동적 환경에서의 복잡한 추리 과제를 해결하기 위해 센서와 프루너를 활용하는 샘플링 기반의 플래너(planner)를 사용. 또한, 다양한 선택형 질문을 사용하는 정량화된 평가 방법을 도입하고, 1,482개의 QA 쌍을 포함하는 WellPlay 데이터셋을 구축함.

- **Performance Highlights**: 실험 결과, PLAYER*는 기존 방법에 비해 복잡하고 동적인 환경에서의 효율성과 성능 향상을 입증.



### Empowering Large Language Models for Textual Data Augmentation (https://arxiv.org/abs/2404.17642)
- **What's New**: 이 연구에서는 자연어 지시를 이해하고 실행할 수 있는 대형 언어 모델 (Large Language Models, LLMs)을 활용하여 텍스트 데이터 확장을 위한 새로운 접근법인 Self-LLMDA를 제안합니다. 이 방법은 데이터 확장 명령을 자동으로 생성하고 선택하여 다양한 다운스트림 작업에 대해 고품질의 확장 데이터를 생성할 수 있도록 지원합니다.

- **Technical Details**: Self-LLMDA 프레임워크는 먼저 LLM을 사용하여 다양하고 효과적인 데이터 확장 지시를 생성하고, 이후 이를 평가하여 타겟 모델의 성능을 강화할 가능성이 가장 높은 지시를 선택합니다. 이 접근 방식은 데이터 확장 지침의 생성 폭과 다운스트림 작업에 대한 목표별 정밀도 사이의 균형을 확보합니다.

- **Performance Highlights**: Self-LLMDA는 26가지 다양한 분야의 소수샘플 학습 작업에서 이전의 비-LLM 기반 및 LLM 기반 데이터 확장 방법들보다 우수한 결과를 보여주었습니다. 이러한 결과는 Self-LLMDA가 다양한 타겟 모델 및 이전에 보지 못한 확장 지침에 대해서도 잘 일반화될 수 있음을 시사하며, 광범위한 적용 가능성을 가지고 있습니다.



### Stylus: Automatic Adapter Selection for Diffusion Models (https://arxiv.org/abs/2404.18928)
Comments: Project Website: this https URL

- **What's New**: 이 연구에서는 'Stylus'라는 새로운 방법을 소개합니다. 이는 프롬프트의 키워드에 기반하여 특정 태스크에 적합한 어댑터들을 효율적으로 선택하고 자동으로 조합합니다. 기존의 어댑터가 갖는 부족한 설명과 매칭 문제를 개선하기 위해, Stylus는 어댑터의 설명과 임베딩(embeddings)을 향상시키는 세 단계 접근 방식을 개발했습니다.

- **Technical Details**: Stylus는 세 단계 접근법을 사용합니다. 첫 번째로는 개선된 설명과 임베딩을 사용하여 어댑터를 요약하고, 두 번째로는 관련 어댑터를 검색하며, 마지막으로는 프롬프트의 키워드에 따라 어댑터를 조합하여 프롬프트와 어떻게 잘 맞는지 확인합니다. 이 연구에서는 'StylusDocs'라는 새로 개발된 데이터 세트를 사용하여 75K개의 어댑터와 사전 계산된 어댑터 임베딩을 특징으로 하고 있습니다.

- **Performance Highlights**: Stylus는 인기 있는 Stable Diffusion 체크포인트(checkpoints)에서 평가되었고, 기존 모델보다 CLIP-FID Pareto 효율성(efficiency)에서 우수한 성능을 보였으며, 인간 및 멀티모달 모델(multimodal models) 평가자들에 의해 두 배 더 선호되었습니다.



### DPO Meets PPO: Reinforced Token Optimization for RLHF (https://arxiv.org/abs/2404.18922)
- **What's New**: 새롭게 제안된 접근 방법은 인간의 피드백으로부터 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 문제를 마르코프 결정 과정(Markov Decision Process, MDP)으로 모델링하여 토큰 단위(Token-wise) 정보를 포착할 수 있도록 하였습니다. 이는 기존의 문장 단위 보상을 넘어서며, 새로운 알고리즘인 'Reinforced Token Optimization (RTO)'을 통해 토큰 단위 보상 함수를 학습하고 최적화합니다.

- **Technical Details**: 이 연구에서는 MLE(Maximum Likelihood Estimation)를 사용하여 오프라인 선호도 데이터에서 토큰 단위 보상 신호를 추출하고, 이를 바탕으로 PPO(Proximal Policy Optimization)와 DPO(Direct Preference Optimization)를 통합한 새로운 토큰 단위의 보상 추출 방식을 도입하였습니다. RTO는 이러한 토큰 단위 보상을 각 토큰에 할당하고 PPO로 최적화하여 기존의 PPO 및 DPO 기법을 능가하는 성능을 보여줍니다.

- **Performance Highlights**: RTO는 기존의 방법들과 비교했을 때 샘플 효율성 측면에서 근접 최적 정책을 찾는 능력이 이론적으로 입증되었습니다. 또한, 대화 태스크(Dialogue Task)에서 토큰 단위 보상을 최적화함으로써 PPO와 DPO를 능가하는 실험 결과를 제시하였습니다.



### Improving Automatic Text Recognition with Language Models in the PyLaia  Open-Source Library (https://arxiv.org/abs/2404.18722)
- **What's New**: 이 논문에서는 자동 텍스트 인식(Automatic Text Recognition, ATR)을 위한 인기 있는 오픈 소스 소프트웨어인 PyLaia의 최신 기여를 소개합니다. 특히, 신뢰할 수 있는 신뢰도 점수 도입과 통계 언어 모델링(statistical language modeling)의 통합에 중점을 두어 PyLaia와 n-gram 언어 모델을 다양한 레벨에서 결합할 수 있는 쉬운 방법을 제공합니다.

- **Technical Details**: 이 연구의 하이라이트는 언어 모델이 완전히 자동 조정(auto-tuned)된다는 점입니다. 전문 지식이 필요 없으며 추가 데이터 요구 없이 쉽게 구축하고 사용할 수 있습니다. 저자들은 PyLaia의 성능을 언어 모델링의 유무에 따라 12개 데이터셋에서 평가하고, 작은 언어 모델을 사용한 디코딩이 단어 오류율(Word Error Rate, WER)을 평균 13%, 문자 오류율(Character Error Rate, CER)을 평균 12% 개선한다는 것을 보여주었습니다.

- **Performance Highlights**: 신뢰도 점수 분석과 보정 기법(calibration techniques)의 중요성에 대해 강조하였습니다. 평가 결과는 언어 모델을 사용하는 디코딩이 전체적인 성능을 개선하며, 특히 정확도와 속도 면에서 눈에 띄게 향상됨을 입증했습니다. 연구의 구현은 PyLaia 공식 리포지토리와 Hugging Face에서 12개의 오픈 소스 모델과 함께 공개적으로 제공됩니다.



### From ChatGPT, DALL-E 3 to Sora: How has Generative AI Changed Digital  Humanities Research and Services? (https://arxiv.org/abs/2404.18518)
Comments: 21 pages, 3 figures

- **What's New**: 이 논문은 대규모 언어 모델(Large-scale language models)이 디지털 인문학 연구에 적용되는 새로운 방식을 탐구합니다. 이 모델들은 고서 보호, 지능형 처리 및 학술 혁신에서 중요한 잠재력을 드러냅니다. 특히 ChatGPT와 같은 모델들이 문서 관리, 내용 이해, 그리고 문화 간 연구에 기여하는 방식을 자세히 설명합니다.

- **Technical Details**: 이 논문은 대규모 언어 모델들이 어떻게 고서 자료의 조직화, 분류, 내용 생성을 돕는지 구체적 사례를 통해 보여줍니다. 또한, 이러한 AI 기술이 예술 혁신과 문화 유산 보존에 어떻게 적용될 수 있는지 전망을 탐구합니다. AI와 디지털 인문학의 상호 작용에서 기술, 정보, 사회 간의 도전과 기회도 다룹니다.

- **Performance Highlights**: AI 기술을 활용한 고서의 조직화 및 분류는 효율성을 대폭 향상시킬 뿐만 아니라, 내용 이해 및 생성에서도 높은 정확성과 창의성을 제공합니다. 이는 디지털 인문학 연구의 새로운 방향을 제시하며, AI가 사회과학 연구와 디지털 인문학에 미치는 영향을 새롭게 조명합니다.



### ECC Analyzer: Extract Trading Signal from Earnings Conference Calls  using Large Language Model for Stock Performance Prediction (https://arxiv.org/abs/2404.18470)
Comments: 15 pages, 3 figures, 5 tables

- **What's New**: 이 연구에서는 딥러닝 (deep learning) 모델을 활용하여 수익성 회의 녹음 (earnings conference calls, ECCs) 데이터의 복잡한 정보를 포착하는 데 한계를 보였던 이전 연구들과 달리, 'ECC Analyzer'라는 새로운 프레임워크를 도입하여 더 풍부하고 예측 가능한 인사이트를 추출합니다. 이 모델은 대규모 언어 모델 (Large Language Models, LLMs)과 멀티모달 (multi-modal) 기술을 결합하여 제작되었으며, 음성의 톤과 피치를 분석하여 강연자의 모드와 자신감 수준을 감지하는 기능을 포함합니다.

- **Technical Details**: ECC Analyzer는 회의록의 구조를 요약하고, 강연자의 음성에서 톤과 피치의 변화를 감지하여 자신감 수준을 분석합니다. 또한, 전문가의 관점에서 주식 성능에 유의미한 영향을 미치는 주요 포커스를 정교하게 추출하기 위해 검색-증강 생성 (Retrieval-Augmented Generation, RAG) 기반 방법을 사용합니다. 이러한 추출된 포커스는 감정 분석 (sentiment analysis)과 오디오 세그먼트 특성과 같은 추가 분석 레이어로 풍부하게 됩니다.

- **Performance Highlights**: ECC Analyzer는 주식 성능의 다양한 지표, 예를 들어 변동성, 위험가치 (Value-at-Risk, VaR), 그리고 다양한 시간 간격에 대한 수익률을 예측하는 멀티태스크 예측을 수행합니다. 실험 결과에 따르면, 이 모델은 전통적인 분석 벤치마크를 능가하는 성능을 보여주며, 금융 분석에서 고급 LLM 기술의 효과를 확인시켜 줍니다.



### Capabilities of Gemini Models in Medicin (https://arxiv.org/abs/2404.18416)
- **What's New**: Med-Gemini 모델은 의학 분야에 전문화된 다기능(Multimodal) 모델로써, 웹 검색 기능을 자연스럽게 사용할 수 있고 새로운 모델리티(modalities)에 맞게 맞춤 인코더(custom encoders)를 통해 효율적으로 조정할 수 있습니다. Med-Gemini는 14개 의학 벤치마크에서 평가되어 10개에서 최신 최고 성능(state-of-the-art, SoTA)을 달성하고, GPT-4 모델 가족보다 모든 벤치마크에서 유의미한 우위를 보이며 특히 MedQA (USMLE) 벤치마크에서 91.1%의 정확도로 SoTA 성능을 달성했습니다.

- **Technical Details**: Med-Gemini는 Gemini 모델의 기반 기능인 언어 및 대화 처리(language and conversations), 다기능 이해(multimodal understanding), 장기 맥락 추론(long-context reasoning)을 활용합니다. 이 모델은 불확실성 유도 검색 전략(uncertainty-guided search strategy)을 사용하여 추론 시간에 더욱 정확하고 신뢰할 수 있는 결과를 제공합니다. 또한, 이 모델은 맞춤형 인코더를 통해 새로운 의료 데이터 모델리티에 적응하여 NEJM 이미지 챌린지와 같은 다기능 의료 벤치마크에서 우수한 성과를 보이며, 진료 요약이나 임상 의뢰서 생성과 같은 실제 의료 과제에서 인간 의사에 비해 우수한 성능을 보입니다.

- **Performance Highlights**: Med-Gemini는 NEJM Image Challenges와 MMMU(health & medicine) 등 7개의 다기능 벤치마크에서 GPT-4V를 평균 44.5% 이상 우수하게 넘어서는 성능을 보여줍니다. 또한, 기존의 맞춤형 방법보다 더 우수한 성능을 보이며, 장기간 비식별화된 건강 기록과 의료 비디오 질문 응답에서 needle-in-a-haystack 검색 작업과 같은 장기 맥락 기능을 통해 SoTA 성능을 보입니다.



### LLM-SR: Scientific Equation Discovery via Programming with Large  Language Models (https://arxiv.org/abs/2404.18400)
- **What's New**: 새롭게 소개된 LLM-SR은 대규모 언어 모델(Large Language Models, LLMs)의 과학적 지식과 코드 생성 능력을 활용하여 데이터로부터 과학적 방정식을 효율적으로 발견하는 방법입니다. 이 방법은 기존의 방정식 발견 방법들이 가지고 있던 한계를 극복하며, 과학자들이 의존하는 분야 특화된 선행 지식(domain-specific prior knowledge)을 고려합니다.

- **Technical Details**: LLM-SR은 방정식을 수학 연산자를 포함하는 프로그램으로 취급하며, LLM의 과학적 선행 지식과 방정식 프로그램에 대한 진화적 검색(evolutionary search)을 결합합니다. LLM은 물리적 이해를 바탕으로 새로운 방정식의 체계를 제안하고, 이는 데이터에 대해 최적화되어 스켈레톤(skeleton) 매개변수를 추정합니다.

- **Performance Highlights**: LLM-SR은 세 가지 다양한 과학 분야에서 그 효과를 입증하였습니다. 조사된 분야에서 LLM-SR은 기존의 방정식 발견 기법들에 비해 물리적으로 정확한 방정식을 발견하며, 도메인 내 및 도메인 외 데이터에 대하여 현저히 더 나은 적합성을 보여주었습니다.



### SOUL: Unlocking the Power of Second-Order Optimization for LLM  Unlearning (https://arxiv.org/abs/2404.18239)
- **What's New**: 이 연구에서는 '대규모 언어 모델(Large Language Models, LLMs)'의 데이터 제거 과정(이른바 unlearning)에 최적화 방식의 선택이 중요하다는 점을 조명합니다. 특히, '영향력 제거(influence unlearning)'라는 전통적 방법과의 연관성을 통해, 최적화 방식 중 '2차 최적화(second-order optimization)'의 중요성을 처음으로 설명하고 있습니다. 또한, 이 연구는 '2차 최적화'를 기반으로 한 새로운 프레임워크인 SOUL을 개발하여, 정적인 한 번의 모델 업데이트 방식에서 벗어나 동적이고 반복적인 데이터 제거 과정을 제안합니다.

- **Technical Details**: SOUL은 '2차 클립 점진적 최적화(second-order clipped stochastic optimization)' 방식을 사용하여 LLM을 훈련합니다. 이 방법은 전통적인 '1차 최적화(first-order optimization)' 방식과 비교하여 모델의 유용성을 손상시키지 않으면서 불필요한 데이터의 영향을 효과적으로 제거할 수 있는 장점을 가지고 있습니다. 이 연구에서는 SOUL을 통한 반복적인 'unlearning' 과정이 효과적임을 다양한 언런닝 작업, 모델 및 메트릭스(metrics)에 걸쳐 일관되게 검증합니다.

- **Performance Highlights**: SOUL은 전통적인 '1차 방법(first-order methods)'을 사용할 때보다 다양한 언러닝 작업에서 일관되게 더 높은 성능을 보여줍니다. 주어진 실험 결과들에 따르면, SOUL이 제공하는 '2차 최적화' 솔루션은 대규모 언어 모델의 데이터 제거 과정에서 확장 가능하고 쉽게 구현할 수 있는 해결책으로서의 가능성을 강하게 시사합니다.



### Ranked List Truncation for Large Language Model-based Re-Ranking (https://arxiv.org/abs/2404.18185)
Comments: Accepted for publication as a long paper at SIGIR 2024

- **What's New**: 이 연구는 순위 목록 자르기(Ranked List Truncation, RLT)를 '검색 후 재정렬(retrieve-then-re-rank)' 관점에서 새롭게 고찰합니다. 특히, 큰 언어 모델(Large Language Model, LLM)을 기반으로 하는 재정렬 과정에 RLT 방법을 적용함으로써 재정렬의 효율성과 효과를 개선할 수 있는지를 탐구합니다. 기존에는 단일 단계 검색에 초점을 맞춘 연구가 주를 이뤘으나, 이 논문은 재정렬 과정에서 RLT를 최적화하는 새로운 방법을 제시합니다.

- **Technical Details**: 연구에서는 LLM 기반 재정렬과 함께 첫 번째 단계 검색(first-stage retrieval)에서 법적, 문서 및 특허 검색과 같이 리소스가 많이 소모되는 분야에서 RLT 기법의 적용 가능성을 다루었습니다. 연구는 렉시컬(lexical), 학습된 희소(learned sparse), 밀집(dense) 검색기와 함께 두 가지 재정렬 방식인 LLM 기반 및 사전 훈련된 언어 모델 기반(pre-trained language model-based) 재정렬을 고려하여 진행되었습니다. RLT 방법은 검색 결과의 유틸리티와 처리 비용 사이의 균형을 최적화하려는 목표를 가지고 있습니다.

- **Performance Highlights**: TREC 2019 및 2020 심층 학습 트랙(TREC 2019 and 2020 Deep Learning Tracks)에서 수행된 실험을 통해, LLM 기반 재정렬을 위한 RLT 방법은 기존 검색 최적화에서 사용된 RLT 방법과 다른 결과를 보였습니다. 특히, RLT를 통해 검색된 목록을 자르는 것이 재정렬의 효율성과 유효성을 동시에 향상시킬 수 있는 잠재력을 가지고 있음을 발견했습니다. 그러나 감독된(supervised) RLT 방법이 고정된 재정렬 깊이와 비교했을 때 명확한 이점을 보이지 않는 경우도 있었습니다.



### Logic Agent: Enhancing Validity with Logic Rule Invocation (https://arxiv.org/abs/2404.18130)
- **What's New**: 새로운 'Logic Agent (LA)'는 자연어 입력을 구조화된 논리 형식으로 변환하여 Large Language Models (LLMs)의 추론 과정의 타당성을 강화하는 에이전트 기반 프레임워크입니다. 이는 기존의 Chain-of-Thought (CoT) 기술을 확장하여, 더욱 체계적이고 일관된 추론 생성을 가능하게 합니다.

- **Technical Details**: Logic Agent는 자연 언어 입력을 구조적 논리 형태로 변환하고, 정의된 함수 세트를 사용하여 추론 과정을 체계적으로 탐색합니다. 이는 전통적인 LLM 방식과 달리, 명제 논리 (propositional logic) 규칙을 동적으로 적용하여 추론 프로세스를 시작합니다.

- **Performance Highlights**: Logic Agent는 다양한 모델 크기에 걸쳐 효과적으로 확장되며, 복잡한 추론 작업에서의 정확성을 크게 향상시킬 수 있음을 실험을 통해 입증했습니다. 이러한 접근 방식은 추론 구조의 해석 가능성(interpretability)과 논리적 일관성(logical coherence)을 향상시키는 데 큰 도움이 됩니다.



### USAT: A Universal Speaker-Adaptive Text-to-Speech Approach (https://arxiv.org/abs/2404.18094)
Comments: 15 pages, 13 figures. Copyright has been transferred to IEEE

- **What's New**: 본 연구에서는 기존의 텍스트-음성 변환(TTS: Text-to-Speech) 방법에서 볼 수 없던 새로운 유니버설 스피커 적응적 TTS (USAT) 프레임워크를 제안합니다. 이 프레임워크는 즉시적 적응(instant adaptation)과 세밀한 적응(fine-grained adaptation)이라는 두 가지 전략을 통합하여 다양한 실제 시나리오에서 효과적으로 사용될 수 있습니다. 특히, 새로운 ESL TTS 데이터셋을 제안하고, 이를 통해 비영어권 화자에 대한 평가도 가능케 하여 더욱 폭넓은 어플리케이션에 활용될 가능성을 열었습니다.

- **Technical Details**: USAT는 두 가지 타입의 적응 방식을 결합하여 언어 모델의 확장성과 범용성을 높였습니다. '즉시적 적응'은 몇 초간의 참조 음성만으로 빠르게 화자의 목소리를 클로닝할 수 있으며, '세밀한 적응'은 더 긴 참조 음성을 사용하여 높은 유사도의 음성을 생성할 수 있습니다. 이를 위해 메모리 증강 변이 오토인코더(VAE: Variational Autoencoder)와 음성 정보 추출 및 음색 변환 과정을 위한 두 가지 차별화된 판별기(discriminators)를 도입하였습니다. 또한, 과적합(overfitting)과 재해 학습(catastrophic forgetting)을 방지하기 위한 경량 플로우 어댑터(flow adapter)와 음소 어댑터(phoneme adapter)를 적용하였습니다.

- **Performance Highlights**: USAT는 포괄적인 실험을 통해 자연스러움과 화자 유사성 측면에서 기존 방법들을 뛰어넘는 성능을 보였습니다. 특히, 고유발음이 강한 화자의 목소리 클로닝에서 높은 정확도를 확보하였고, 조정 필요 파라미터는 단 0.5%에서 1.6%에 불과하며, 이를 통해 효율적인 저장 공간 활용이 가능해졌습니다. 또한, 새로운 ESL TTS 데이터셋을 통해 비영어권 화자의 다양한 발음에 대한 평가가 가능해져 국제적인 음성 합성 적응성을 강화하였습니다.



### ComposerX: Multi-Agent Symbolic Music Composition with LLMs (https://arxiv.org/abs/2404.18081)
- **What's New**: 이 연구에서는 음악 작곡의 복잡성을 해결하기 위해 다중 에이전트 방식을 적용한 새로운 프레임워크인 ComposerX를 제안합니다. 이는 기존의 대규모 언어 모델(GPT-4)이 지닌 음악에 대한 이해와 생성 능력을 향상시키는 데 중점을 둡니다.

- **Technical Details**: ComposerX는 사용자 지시에 따라 조화롭고 다성부(polyphonic) 음악 작품을 생성할 수 있습니다. 이 프레임워크는 음악 이론 및 역사에 대한 광범위한 지식과 추론 능력을 활용하여 LLM의 음악 작곡 가능성을 탐구하고 확장합니다. 또한, GPT-4를 기반으로 하여 In-Context-Learning 및 Chain-of-Thoughts 같은 현대 기술을 적용하였지만, 질적으로 더 우수한 음악 생성 결과를 도출하기 위한 다중 에이전트 접근 방식을 채택하였습니다.

- **Performance Highlights**: ComposerX는 일관된 다성부 음악과 매력적인 멜로디를 구현할 수 있는 능력을 입증하였습니다. 이 프레임워크는 사용자의 지시를 준수하면서도 품질이 높고 매력적인 음악 작품을 생성하는 데 효과적임을 보여줍니다.



### CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing  Experiments (https://arxiv.org/abs/2404.18021)
- **What's New**: 이번 연구에서는 유전자 편집 실험을 자동화하고 강화하는 새로운 AI 에이전트인 CRISPR-GPT를 소개합니다. CRISPR-GPT는 기존의 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력을 활용하여 CRISPR 시스템 선택, 가이드 RNA 설계, 세포 전달 방식 권장, 프로토콜 작성 및 유효성 검사 실험 설계를 촉진합니다. 이를 통해 비전문가 연구자들이 유전자 편집 실험을 처음부터 도와주고 실제 사례를 통해 에이전트의 효과를 검증합니다.

- **Technical Details**: CRISPR-GPT는 여러 단계에서 유전자 편집 실험을 설계하는 과정을 단순화합니다. 이 과정에는 CRISPR 시스템의 선택, 최적화된 gRNA(guide RNA)의 설계, 전달 방법 선택, 부작용 효과 예측, 실험 프로토콜 추천, 유효성 확인 방법 권장 및 프라이머 설계가 포함됩니다. 또한, CRISPR-GPT는 사고의 흐름 추론 모델(chain-of-thought reasoning model)과 상태 머신(state machines)을 활용하여, 심지어 유전자 편집이 처음인 사람들도 그들의 연구 필요에 맞는 실험 설계를 반복적으로 정제할 수 있도록 지원합니다.

- **Performance Highlights**: CRISPR-GPT는 실험 설계 과정에서 발생할 수 있는 추가적인 문제들에 대응하는 데 도움을 주는 Freestyle Q&A Mode와 사전 설계된 gRNA에 대한 심층 분석을 제공하는 Off-target Prediction Mode를 제공합니다. 이와 함께 유전 정보의 개인 정보 보호, 부작용 경고 등, 유전자 편집 기술의 윤리적 및 안전 측면을 고려한 안전장치도 통합되어 있습니다.



### Spatio-Temporal Side Tuning Pre-trained Foundation Models for  Video-based Pedestrian Attribute Recognition (https://arxiv.org/abs/2404.17929)
Comments: Parameter Efficient Fine-Tuning Strategy for Video-based Pedestrian Attribute Recognition

- **What's New**: 이 연구에서는 기존의 사람 속성 인식(Pedestrian Attribute Recognition, PAR) 알고리즘이 주로 정적 이미지를 기반으로 개발되었지만, 심각한 가림막 및 모션 블러 등과 같은 도전적인 상황에서는 성능이 불안정하다는 문제를 해결하기 위해 비디오 프레임을 사용하는 새로운 접근 방식을 제안합니다. 이는 시간적 정보를 효율적으로 활용하여 인간 속성을 이해하는 것을 목표로 합니다.

- **Technical Details**: 이 작업에서는 비디오 기반의 PAR을 시각-언어 융합 문제로 정형화하고, 이미 사전 훈련된 다중모달 기반 모델(CLIP: Contrastive Language Image Pretraining)을 사용하여 시각적 특징을 추출합니다. 더욱이, 사전 훈련된 시각 기반 모델의 매개변수 효율적 최적화를 달성하기 위해 새로운 시공간 사이드 튜닝(strategy) 전략을 제안합니다. 의미론적 정보를 더 잘 활용하기 위해 인식해야 할 전체 속성 목록을 다른 입력으로 취하고, 속성 단어/구를 해당 문장으로 변환하기 위해 'split', 'expand', 'prompt' 작업을 수행합니다. 그 후 CLIP의 텍스트 인코더를 사용하여 처리된 속성 설명을 임베딩합니다. 평균화된 시각 토큰과 텍스트 토큰은 연결되어 융합 트랜스포머(Transformer)에 전달되어 다중모달 인터랙티브 학습이 이루어집니다. 강화된 토큰들은 사람 속성 예측을 위한 분류 헤드로 전달됩니다.

- **Performance Highlights**: 두 개의 대규모 비디오 기반 PAR 데이터셋에 대한 광범위한 실험을 통해 제안된 프레임워크의 효과성이 완전히 검증되었습니다. 이 논문의 소스 코드는 해당 HTTP URL에서 이용 가능합니다.



### Bridging the Social & Technical Divide in Augmentative and Alternative  Communication (AAC) Applications for Autistic Adults (https://arxiv.org/abs/2404.17730)
- **What's New**: 이 연구는 자폐성 성인이 사용하는 대체적 강화 의사소통(Augmentative and Alternative Communication, AAC) 도구에 대해 기존 연구들이 주로 아동 또는 육체적 장애를 가진 성인을 중심으로 진행되었음을 지적하며, 자폐성 성인의 요구와 경험을 중심으로 한 통찰력을 제공합니다. 특히, 자폐성 성인들과의 심층 인터뷰를 통해 얻은 정보를 바탕으로, 기술적 문제뿐만 아니라 사회적 문제에 대해서도 논의하며, AAC 도구의 개선을 위한 구체적인 가이드라인을 제안합니다.

- **Technical Details**: 이 연구에서는 자연어 처리(Natural Language Processing, NLP) 기술과 관련하여 자폐성 성인 사용자가 원하는 AAC의 기능과 특성을 이해하기 위해 심층 인터뷰 방식을 채택했습니다. 연구자들은 사용자의 의견을 기술 개발에 통합하는 것의 중요성을 강조하며, 대화형 인터뷰를 통해 9가지 주요 주제(입력 옵션, 출력 옵션, AAC 선택 또는 적응, 시작 또는 변경 시기, 이점, 접근성, 지속적 사용의 장애물, 사회적 고려사항, 통제 부족)를 도출했습니다. 추가로, NLP를 이용한 언어 모델링, 맥락 기반 예측, 개인화된 언어 등의 기술적 접근 방식에 대해서도 언급합니다.

- **Performance Highlights**: 연구 결과, 자폐성 성인들이 현재 사용하는 AAC 도구에서 경험하는 문제점들과 그들이 기술에서 기대하는 것들을 구체적으로 식별할 수 있었습니다. 인터뷰를 통해 발견된 여러 가지 카테고리는 AAC 도구 및 애플리케이션 개발자들이 사용자의 요구와 기대에 보다 잘 부응할 수 있는 방향을 제시합니다. 특히, 사용자의 개인적 정체성과 AAC 사용 사이의 상호작용에 대한 깊은 이해는 자폐성 성인을 위한 보다 맞춤화된 솔루션 개발로 이어질 수 있습니다.



### Utilizing Large Language Models to Identify Reddit Users Considering  Vaping Cessation for Digital Interventions (https://arxiv.org/abs/2404.17607)
- **What's New**: 본 연구에서는 GPT-4와 기존 BERT 기반 언어 모델을 사용하여 사회적 미디어 데이터 분석의 정확성과 신뢰도를 향상시키는 데 GPT-4의 가능성을 확인하며, Reddit에서 전자담배를 그만두려는 사용자의 의도를 분석합니다. 인간 평가자와 비교하여 GPT-4 모델이 주석(annotation) 가이드라인 및 과정을 일관되게 준수하는 우수한 성능을 보여줍니다.

- **Technical Details**: 연구의 데이터는 Reddit의 r/QuitVaping 서브레딧(subreddit)에서 추출되었고, 이 데이터는 GPT-4 모델과 인간 평가자가 텍스트 주석 과제에서 평가하였습니다. GPT-4는 인간 평가자가 간과할 수 있는 세부적인 사용자의 의도를 감지하는 뛰어난 능력을 보여주었습니다. 또한, 연구에서는 BERT 모델을 사용하여 일대일로 텍스트를 'Yes' 또는 'No'로 분류하는 과업에 대한 성능을 비교 분석하였습니다.

- **Performance Highlights**: GPT-4는 주석 가이드라인을 준수하는 데 있어 인간 평가자보다 일관된 우수성을 보이며, 미묘한 의도 감지 능력에서 뛰어난 성능을 나타내었습니다. BERT 및 기타 BERT 기반 모형들도 Reddit 데이터에 대한 세밀한 조정(fine-tuning)을 통해 사용자의 의도를 효율적으로 분류하는 데 사용되었습니다.



### LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (https://arxiv.org/abs/2404.16710)
Comments: Code open sourcing is in progress

- **What's New**: LayerSkip을 소개합니다. 이는 대규모 언어 모델의 추론 속도를 향상시키기 위한 종단간(end-to-end) 솔루션입니다. LayerSkip은 트레이닝 중에 레이어 드랍아웃(layer dropout)과 조기 종료 손실(early exit loss)을 사용하며, 추론 단계에서는 이른 레이어에서의 조기 종료를 더 정확하게 할 수 있도록 합니다. 또한, 새로운 자기-추측 디코딩(self-speculative decoding) 방법을 제시하여 메모리 사용량을 줄이고, 더 빠른 추론을 가능하게 합니다.

- **Technical Details**: LayerSkip은 다양한 트레이닝 접근법(예: 처음부터의 전 학습(pretraining), 지속적인 전 학습(continual pretraining), 특정 도메인 및 과제에 대한 미세조정(finetuning))과 함께 사용되며, 여러 Llama 모델 크기에 대한 실험을 통해 검증되었습니다. 트레이닝 단계에서는 초기 레이어는 낮은 드랍아웃 비율을, 후반 레이어는 높은 드랍아웃 비율을 적용합니다. 또한, 모든 트랜스포머 레이어가 동일한 출구를 공유하는 조기 종료 손실을 사용합니다. 추론 시에는 조기 종료가 더 일찍 일어나는 것이 가능해져, 추가 모듈 없이도 더 빠른 추론이 가능합니다.

- **Performance Highlights**: LayerSkip은 CNN/DM 문서 요약에서 최대 2.16배, 코딩 작업에서 1.82배, TOPv2 의미 파싱 작업에서 2.0배의 속도 향상을 보였습니다. 이는 기존의 추측적 디코딩(speculative decoding) 접근법보다 메모리 사용량이 적고, 초안(draft)과 검증(verification) 단계의 컴퓨트와 활성화를 공유한다는 장점을 갖고 있습니다.



### Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs  Better Reasoners (https://arxiv.org/abs/2404.14963)
Comments: Work in progress

- **What's New**: 새롭게 제안된 연구에서는 복잡한 추론 과제를 처리할 때 나타나는 오류를 줄이기 위해 '문제를 깊이 이해하기(Deeply Understanding the Problems, DUP)'라는 방법을 제시합니다. 이 방법은 LLM(Large Language Models)이 문제를 더 깊이 분석하고 핵심 문제 해결 정보를 이용하여 추론 능력을 강화시키는 것을 목표로 합니다.

- **Technical Details**: DUP 방법론은 세 단계로 구성됩니다. 첫째, 원래 질문에서 핵심 질문을 밝히고, 둘째, 해당 핵심 질문을 해결하기 위해 필요한 문제 해결 정보를 추출하고, 셋째, 핵심 질문과 문제 해결 정보를 결합하여 상세한 응답을 생성합니다. 이 새로운 접근법은 기존의 Chain of Thought (CoT) 전략을 수정하여 LLM의 추론 단계의 질을 향상시키는 데 초점을 맞춥니다.

- **Performance Highlights**: DUP는 다양한 추론 벤치마크에서 기존 방법들을 큰 폭으로 능가하는 성능을 보였으며, 특히 GSM8K 벤치마크에서는 새로운 SOTA(State of the Art) 결과를 달성하였습니다(정확도 97.1% in a zero-shot setting). 이와 함께, DUP는 SVAMP 벤치마크에서도 90.4%에서 94.2%로 정확도를 향상시켰습니다.




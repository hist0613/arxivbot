New uploads on arXiv(cs.CL)

### Contextual ASR Error Handling with LLMs Augmentation for Goal-Oriented Conversational AI (https://arxiv.org/abs/2501.06129)
Comments:
          Accepted to COLING 2025 Industry Track

- **What's New**: 본 연구에서는 목표 지향 대화(Goal-Oriented Dialogue)에서의 자동 음성 인식(ASR)의 오류 수정을 위한 새로운 접근 방식을 제안합니다. 기존의 ASR 수정 방법이 사용자의 이전 데이터나 명명된 개체에 의존하는 것과 달리, 우리는 임의의 사용자 데이터 없이도 대화 컨텍스트를 활용하여 다양한 언어적 변형을 반영합니다. 결과적으로, 대화 상태(Contextual Information)에 기반한 새로운 랭킹 전략과 언어 모델을 적용하여 ASR 수정의 효율성을 높였습니다.

- **Technical Details**: 제안된 방법은 사용자 반응을 예측하는 대화 상태를 기반으로 ASR 오류를 수정하는 것입니다. 우리는 n-best ASR 가설을 문맥과의 어휘적 및 의미적 유사성에 따라 재정렬하고, 음성 정보에 따라 컨텍스트를 평가하여 오류를 수정합니다. 또한, 대화 상태에 적합한 특정 조건에서만 수정이 활성화되도록 설정하여 잘못된 긍정(False Positive) 비율을 감소시킵니다.

- **Performance Highlights**: 실제 사용자와의 평가에서, 본 방법은 ASR 수정의 회수율(Recall)과 F1 점수를 각각 34% 및 16% 향상시켰습니다. 사용자들은 수정 방법이 제대로 작동할 때 5점 만점에 0.8-1점 더 높은 평점을 주었으며, 잘못된 긍정으로 인한 평점 하락은 없었습니다. 이 연구는 Amazon Alexa에 배포되어 실제 사용자 환경에서 그 효용성을 입증하였습니다.



### Merging Feed-Forward Sublayers for Compressed Transformers (https://arxiv.org/abs/2501.06126)
- **What's New**: 이번 연구에서는 기존의 pruning(가지치기) 기술 대신에, 모델 내에서 유사한 매개변수 그룹을 병합하는 새로운 압축 방법을 제안합니다. 특히 Transformer 모델의 feed-forward(전달) 서브 레이어를 선택하고 정렬하여 병합하는 방식으로, 다양한 언어 모델링, 이미지 분류 및 기계 번역 작업을 수행했습니다. 이 방법을 통해 원본 모델과 유사한 성능을 유지하면서, 모델의 세 번째 이상을 통합할 수 있음을 보였습니다.

- **Technical Details**: Transformer의 feed-forward 서브 레이어는 전체 매개변수의 약 2/3를 차지하며, 이들의 압축은 상당한 성과를 가져올 수 있습니다. 논문은 이를 위해 permutation-based(순열 기반) 뉴런 정렬 방법을 사용하여 두 개의 레이어를 정렬합니다. 이 과정에서 cross-correlation(상관관계)를 계산하여 최적의 뉴런 정렬을 도출하고, 이를 통해 여러 개의 유사한 서브 레이어를 하나의 매개변수 집합으로 결합하는 기술을 제안합니다.

- **Performance Highlights**: 제안된 방법은 Vision Transformer에서도 21% 이상의 매개변수를 제거하면서도 99%의 원래 성능을 유지할 수 있음을 보여주었습니다. 또한, 다른 Transformer 기반 모델들, 즉 GPT-2, ViT, 기계 번역 모델을 대상으로 실험하여, 이 방법이 기존의 모델에 비해 유의미한 성능 향상을 가져온 것을 확인했습니다. 연구 결과는 다양한 전이 학습 모델에서도 쉽게 적용될 수 있는 잠재력을 보여줍니다.



### Fleurs-SLU: A Massively Multilingual Benchmark for Spoken Language Understanding (https://arxiv.org/abs/2501.06117)
- **What's New**: 이 논문에서는 102개 언어에서 주제 기반 음성 분류(topic classification)와 92개 언어의 듣기 이해를 통한 선택형 질문 응답(multiple-choice question answering)을 지원하는 Fleurs-SLU라는 다국어 음성 언어 이해(SLU) 벤치마크를 새롭게 제시합니다. 이 연구는 낮은 자원 언어에 대한 음성 인식의 신뢰성을 높이기 위한 방안을 모색하며, SLU의 필요성이 강조되고 있습니다. 또한 기존의 SLU 평가 방식에서 벗어나 실제 사용 사례를 반영한 개선된 데이터셋의 필요성을 언급합니다.

- **Technical Details**: Fleurs-SLU는 기존의 데이터셋과 연결하여 구성되었으며, 대화 음성을 주제로 한 분류와 선택형 질문 응답으로 구성된 새로운 다국어 SLU 벤치마크입니다. 이 연구에서는 Cascaded Systems가 SLU 작업에서 더 향상된 견고성을 보여주는 반면, 잘 맞춤된 speech encoders도 주제 기반 음성 분류에서 경쟁력 있는 성능을 보일 수 있음을 발견했습니다. 이를 통해 음성 인식 모델의 다국어 SLU 능력이 향상되어야 함을 제안합니다.

- **Performance Highlights**: 연구 결과, 강력한 다국어 SLU는 다국어 음성 인식(ASR)과 높은 품질의 음성-영어 텍스트 번역(S2ETT)과 강력한 상관관계를 나타냈습니다. 모델 성능 평가에서, SLU의 품질은 성능을 감소시킬 수 있으며, 이 연구는 SLU 기능을 고려한 다국어 음성 모델의 사전 훈련(pre-training)이 다국어 ASR의 견고성을 높이는 데 중요한 역할을 한다고 강조합니다.



### From Conversation to Automation: Leveraging Large Language Models to Analyze Strategies in Problem Solving Therapy (https://arxiv.org/abs/2501.06101)
Comments:
          16 pages

- **What's New**: 이 연구는 Problem-Solving Therapy (PST)의 자동화를 검토하여 대화형 AI의 잠재력을 탐구합니다. GPT-4o 모델을 사용하여 PST 전략을 식별하는 데 가장 높은 정확도(0.76)를 기록했으며, 이는 다른 모델보다 우수한 성과를 의미합니다. 또한, 커뮤니케이션 전략의 새로운 차원을 도입하여 PST 프레임워크를 개선했으며, 이는 치료사-내담자 상호작용에 대한 깊은 통찰력을 제공합니다.

- **Technical Details**: 연구는 240개의 익명화된 PST 대화 데이터로 구성된 68,306개의 대화 교환을 분석하여 LLMs를 활용해 치료사 발화를 자동으로 주석 처리하는 방식을 채택하였습니다. 이 접근 방식은 시간 효율적인 데이터 분석을 가능하게 하며, PST 전략을 분류하는 데 있어 높은 정확성을 달성했습니다. 또한, 닫힌 및 개방형 가중 모델을 모두 사용하여 LLM을 적용함으로써 치료 과정에서 사용된 특정 전략들의 변화를 검증했습니다.

- **Performance Highlights**: PST 관련 텍스트 분석에 LLM을 사용한 결과, 치료 대화에서 PST 전략을 주석 처리하는 데 높은 정확성과 효율성을 보였습니다. 이 연구는 NLP 및 정신 건강 관리 분야에 기여하며, 복잡한 치료 대화 분석을 자동화하기 위한 프레임워크를 제공합니다. 또한, 익명화된 실제 치료 세션의 전사 결과는 PST 연구를 발전시키고 정신 건강 관리에서 LLM의 변형 잠재력을 입증하는 데 기여합니다.



### Benchmarking Rotary Position Embeddings for Automatic Speech Recognition (https://arxiv.org/abs/2501.06051)
- **What's New**: 이번 연구에서는 Rotary Position Embedding (RoPE)를 음성 인식(Auto Speech Recognition, ASR) 작업에서 실험하며 RelPOS와 비교했습니다. RoPE는 텍스트 처리에선 우수한 성과를 보였으나 음성 처리 분야에 대한 연구는 부족했던 상황입니다. 다양한 언어에 대해 ASR 성능을 평가하여 새로운 데이터를 제공하고, SpeechBrain 도구를 통해 RoPE 구현 및 실험 레시피를 공개합니다.

- **Technical Details**: RoPE는 입력 시퀀스의 각 벡터를 그 위치에 따라 회전시키는 방식으로 작동합니다. Transformer 모델은 입력 시퀀스를 값(value) 시퀀스와 키(key) 시퀀스로 변환하며, RoPE를 활용하면 상대 위치 정보를 유도할 수 있습니다. 회전 매트릭스와 관련된 계산을 통해 RoPE는 효율적인 성능을 자랑합니다.

- **Performance Highlights**: 실험 결과, RoPE는 영어 ASR 데이터셋인 LibriSpeech 및 LibriHeavy에서 RelPOS보다 낮은 오류율을 보였습니다. RoPE는 대규모 ASR 데이터셋에서도 성능이 우수하며 훈련 속도는 13% 더 빠른 것으로 나타났습니다. 비영어 ASR 데이터세트(CommonVoice 18.0)에서도 다양한 언어에서 뛰어난 ASR 결과를 기록했습니다.



### How to Tune a Multilingual Encoder Model for Germanic Languages: A Study of PEFT, Full Fine-Tuning, and Language Adapters (https://arxiv.org/abs/2501.06025)
Comments:
          Accepted at NoDaLiDa Baltic-HLT 2025 Conference

- **What's New**: 이번 논문은 다국어 인코더 모델 mDeBERTa를 활용하여 독일어, 스웨덴어, 아이슬란드어 세 가지 독일어계 언어의 작업에 최적화된 사용을 조사합니다. PEFT(파라미터 효율적인 미세 조정) 방법인 LoRA와 Pfeiffer bottleneck adapters을 비교하며, 독일어에서 PEFT가 더 효과적임을 발견했습니다. 그러나 스웨덴어와 아이슬란드어의 경우 결과가 일관되지 않았습니다.

- **Technical Details**: PEFT 방법들은 모델의 표현을 보존하면서, 더 나은 일반화(Generalization)를 이끌어낼 수 있습니다. 연구에서는 데이터 품질과 사용 가능한 자원의 양에 따라 완전 미세 조정(Full Fine-tuning)과 PEFT가 서로 다른 효과를 내는 것을 규명하였습니다. 알고리즘 검토에는 언어 적응 모듈이 포함되어 있으며, 이는 비구조적 텍스트에 대한 훈련을 통해 얻어진 것입니다.

- **Performance Highlights**: 독일어에서는 PEFT 방법이 일관되게 최고의 결과를 제공하며, 때때로 소폭의 개선 효과를 보입니다. 반면, 스웨덴어와 아이슬란드어의 성능은 과제에 따라 달라지며, PEFT는 질의 응답(Extractive QA)에 더 유리한 반면, 고유명사 인식(NER)에는 완전 미세 조정이 더 나은 성과를 보입니다. 전반적으로 언어 적응기술이 테스트된 모든 작업이나 언어에서 일관된 개선 효과를 제공하지 않았습니다.



### Constraining constructions with WordNet: pros and cons for the semantic annotation of fillers in the Italian Constructicon (https://arxiv.org/abs/2501.05990)
- **What's New**: 이 논문은 이탈리아어 Constructicon에서의 구성물을 형식화하고 의미 주석을 부여하는 데 있어 WordNet 기반의 의미 분류가 어떻게 활용되는지를 다루고 있습니다. 특히 Open Multilingual WordNet를 사용하여 구성의 의미적 특성과 제약을 표현하는 방법을 설명합니다. 이러한 접근법은 기존의 자원들과의 상호 운용성을 높이고, 향후 다른 Constructicon과의 연결 가능성을 제시합니다.

- **Technical Details**: 이 연구에서 도입된 주요 구성 요소는 OntoClass라는 의미적 특성입니다. OntoClass는 Open Multilingual WordNet 주제를 사용하여 명사와 동사의 의미 범주를 주석 달기 위해 활용됩니다. 현재는 명사(26개 클래스)와 동사(15개 클래스)만을 위해 이 태그 세트를 사용하고 있으며, 이는 유사한 자원들 간의 상호 운용성을 위해 필수적입니다.

- **Performance Highlights**: 이탈리아어 UD 트리뱅크에서 많은 어근과 형태가 최소 하나의 synset과 연관되어 있다는 점에서 긍정적인 결과를 보여주었습니다. 이는 우리가 설정한 엄격한 빈도 기준에도 불구하고 구문 작용에서 상당히 많은 수의 구성물이 식별될 수 있음을 나타냅니다. 그러나 WordNet 주제를 사용함으로써 발생할 수 있는 몇 가지 제한 사항도 언급되며, 특히 새로운 의미적 태그를 추가하는 것이 어렵다는 점이 강조됩니다.



### Addressing speaker gender bias in large scale speech translation systems (https://arxiv.org/abs/2501.05989)
- **What's New**: 이 연구는 음성 번역(Speech Translation, ST) 시스템 내 성별 편향이 정확하지 않은 번역을 초래할 수 있다는 문제를 다룹니다. 기존의 대규모 ST 시스템에서 추진되는 남성 편향을 개선하기 위해, 본 연구는 성별을 고려한 번역 교정을 위해 대형 언어 모델(Large Language Models, LLMs)을 사용합니다. 또한, 모델이 성별에 기반하여 음성 신호로부터 직접 번역을 생성할 수 있도록 조정하는 방법을 제안합니다.

- **Technical Details**: 성별 편향 문제를 해결하기 위한 모델은 음성 신호로부터의 성별 구별 신호를 활용하는 방법을 포함합니다. 본 연구에서는 세 가지 모드의 하이퍼파라미터 조정을 통해 성별을 사전에 정의하였거나 음성 신호로부터 추론할 수 없는 경우를 효과적으로 처리할 수 있습니다. 또한, 성별 나타남(Gender Representation, GR) 손실을 ST 모델 훈련 과정에 추가하여 성별 특화 번역의 품질을 향상시키고자 했습니다.

- **Performance Highlights**: 제안된 방법을 통해, 여성 화자의 번역에서 70%의 개선을 보였으며, MuST-SHE 테스트 세트에서 Seamless M4T와 Canary와 같은 기존 ST 시스템들과 비교시 더 뛰어난 성능을 발휘했습니다. 실험 결과, 제안된 접근법이 영어-스페인어(ES) 및 영어-이탈리아어(IT) ST 모델에서 평균 90% 이상의 성별 번역 정확도를 달성했음을 나타냅니다.



### Hermit Kingdom Through the Lens of Multiple Perspectives: A Case Study of LLM Hallucination on North Korea (https://arxiv.org/abs/2501.05981)
Comments:
          Accepted at COLING 2025

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 발생하는 환각(hallucination)이 특히 북한과 같은 신뢰할 수 있는 정보가 부족한 맥락에서 어떻게 작용하는지를 분석합니다. 저자들은 다국어 LLM 및 특정 언어 기반 모델들이 북한에 관한 정보를 생성할 때 나타나는 차이를 연구하였고, 이 연구의 결과는 모델과 언어의 선택이 북한에 대한 이해에 얼마나 큰 영향을 미치는지를 보여줍니다.

- **Technical Details**: 연구에서는 ChatGPT-3.5, Gemini, Claude 3 Sonnet, Solar-Mini(한국어 전용), Qwen-72B(중국어 전용)과 같은 다양한 LLM들이 세 가지 언어(영어, 한국어, 중국어)에서 북한에 대해 생성하는 정보를 평가합니다. 이 과정에서 연구진은 검증 가능한 사실이 포함된 13개의 주제를 기준으로 모델의 정확도, 일관성(consistency), 거부 응답(refusal-to-answer, RtA) 비율을 측정하였습니다. 연구는 현재 LLM이 북한과 같은 민감한 주제에 대해 어떻게 정보를 생성하는지를 정밀하게 분석합니다.

- **Performance Highlights**: 논문은 LLM들이 북한 관련 정보를 생성할 때 의도하지 않은 불확실성(unintended uncertainty, 예: 거부 응답)을 나타내는 경향이 있음을 보여줍니다. 이는 많은 컨텍스트에서 실행 가능한 해결책들이 북한과 같은 특정한 특성을 가진 주제에서 적용하기 어려운 문제를 초래할 수 있음을 시사합니다. 궁극적으로, 북한에 대한 올바르고 균형 잡힌 정보의 필요성이 더욱 중요해짐에 따라 이 연구는 관련 데이터와 모델의 배치에 있어 고차원적인 정보의 필요성을 강조합니다.



### Finnish SQuAD: A Simple Approach to Machine Translation of Span Annotations (https://arxiv.org/abs/2501.05963)
Comments:
          NoDaLiDa 2025

- **What's New**: 이 논문에서는 DeepL 기계 번역 서비스를 이용하여 Span-레벨 주석이 있는 데이터셋을 번역하는 간단한 방법을 제안하고 있습니다. 이를 통해 SQuAD2.0 질문 응답 데이터셋의 핀란드어 버전을 생성하고, 새로운 데이터셋을 기반으로 QA 리트리버 모델을 훈련했습니다. 이 방법은 사용이 간편하며, 여러 평가에서 일관되게 더 나은 번역 데이터를 생성하는 것으로 나타났습니다.

- **Technical Details**: 질문 응답(QA) 시스템 구축은 일반적으로 두 단계의 리트리버-리더 파이프라인으로 이루어집니다. 전통적인 추출 QA에서는 SQuAD와 같은 대규모 QA 데이터셋이 핵심 역할을 하며, 기계 번역(MT) 기술의 발전을 통해 저자원 언어에서도 이러한 QA 데이터셋을 생성할 수 있게 되었습니다. 본 논문에서 제안하는 방법은 DeepL의 기능을 활용하여 주석이 있는 데이터를 효과적으로 핀란드어로 번역하는 것입니다.

- **Performance Highlights**: 이 데이터셋은 핀란드어 인코더 모델을 활용한 추출 QA 시스템 개발에 사용될 수 있으며, 핀란드어 LLM의 교육 및 벤치마킹에도 중요한 역할을 합니다. 연구 결과, 훨씬 더 높은 정확도를 유지하면서 많은 질문-답변 쌍을 성공적으로 보존했음을 보여주며, 이는 다른 유사한 데이터셋의 번역에도 응용될 수 있을 것으로 기대됩니다.



### Effective faking of verbal deception detection with target-aligned adversarial attacks (https://arxiv.org/abs/2501.05962)
Comments:
          preprint

- **What's New**: 이번 연구에서는 언어 분석을 통한 기만 탐지 (deception detection)를 다루며, 인간의 판단과 기계 학습 (machine learning) 모델의 판단 모두에서 기만적인 진술을 사실처럼 보이게 하는 자동화된 적대적 공격 (adversarial attacks)에 대한 위험을 강조합니다. 연구에 사용된 데이터셋은 243개의 진실한 이야기와 262개의 조작된 이야기로 구성되어 있습니다.

- **Technical Details**: 연구는 두 개의 주요 연구로 나뉘며, 첫 번째 연구에서는 인간과 두 가지 기계 학습 모델 (fine-tuned language model, n-gram model)의 기만적 진술이 원래 형태와 적대적으로 수정된 형태에 대해 판단하는 과정을 살펴보았습니다. 두 번째 연구에서는 수정된 진술의 목표 정렬 (target alignment)을 조작하여, 인간 평가자와 기계 모델을 위한 수정의 효과를 평가했습니다.

- **Performance Highlights**: 적대적 수정이 목표와 정렬될 경우, 인간과 기계의 판단 정확도가 각각 우연적으로 판단할 확률 수준인 약 51%로 떨어졌습니다. 반면 목표와 정렬되지 않은 경우, 인간의 판단과 기계 학습 모델의 성능은 상당히 개선되어 각각 63%에서 78%의 정확도를 보여주었습니다. 이 연구 결과는 인간 및 기계 모델의 기만 탐지 방식이 적대적 수정에 얼마나 취약한지를 강조합니다.



### Universal-2-TF: Robust All-Neural Text Formatting for ASR (https://arxiv.org/abs/2501.05948)
- **What's New**: 이 논문에서는 상업적인 자동 음성 인식(ASR) 시스템을 위한 전면적인 신경망 기반 텍스트 포맷팅(Text Formatting, TF) 모델을 소개합니다. 이 모델은 문장 부호 복원(Punctuation Restoration, PR), 올바른 대소문자(truecasing), 그리고 역 텍스트 정규화(Inverse Text Normalization, ITN)를 수행하며, 기존의 규칙 기반 또는 혼합 접근 방식과는 달리 다목적 토큰 분류기와 시퀀스-투-시퀀스(seq2seq) 모델로 구성된 두 단계의 신경 네트워크 아키텍처를 활용합니다.

- **Technical Details**: 제안된 모델은 AssemblyAI의 Universal-2 ASR 시스템의 일환으로 개발되었으며, PR, truecasing 및 ITN 작업을 처리하는 두 개의 신경망 모델로 구성됩니다. 첫 번째 모델은 PR과 대소문자 처리를 수행하고 ITN 또는 혼합 대소문자 처리가 필요할 수 있는 텍스트 범위를 식별하는 다목적 토큰 분류기입니다. 두 번째 모델은 특定된 범위에 대해 ITN 및 혼합 대소문자 변환을 수행하는 seq2seq 모델입니다.

- **Performance Highlights**: 종합적인 평가 결과는 TF 정확도, 계산 비용 및 인지적 품질에 대한 정량적 지표를 포함하여 제안된 전면적 신경망 방법의 효과를 보여줍니다. 이 방식은 컴퓨팅 비용을 줄이면서도 높은 정확도를 유지하며, 다양한 언어적 개체 및 텍스트 도메인에서의 유연성과 안정성을 보장합니다. 이 연구는 실제 환경에서 ASR의 사용성을 향상시키기 위한 홀리스틱 TF 모델의 중요성을 강조합니다.



### LLMs Reproduce Stereotypes of Sexual and Gender Minorities (https://arxiv.org/abs/2501.05926)
Comments:
          10 pages, 8 figures, 6 tables

- **What's New**: 본 논문은 기존의 성별과 성 정체성에 대한 이분법적이고 본질주의적 접근을 넘어서서, 성 및 성 정체성이 가진 스펙트럼을 탐구합니다. 저자들은 대규모 언어 모델(LLMs)이 성소수자 및 성별 소수자에 대해 어떻게 편향되어 있는지를 연구하며, 이는 기존 연구에서 간과되었던 중요한 측면입니다. 특히, LLM이 생성하는 텍스트가 사회적 편견을 어떻게 재생산하는지를 보여줍니다.

- **Technical Details**: 연구에 사용된 주된 이론적 틀은 Stereotype Content Model (SCM)으로, 이는 사회적 집단에 대한 편견을 따뜻함(Warmth)과 능력(Competence)이라는 두 차원으로 설명합니다. 저자들은 세 가지 LLM 모델(GPT 3.5-turbo, Gemini-1.5-flash, LLaMA 2-7b-chat-hf)에 대해 조사를 수행하였으며, 이들 모델이 성소수자에 대한 편견을 어떻게 반영하는지를 분석했습니다. 실험을 통해 발견된 편향은 LLM이 사회적 집단에 대한 부정적 인식을 생성하는 경향이 있음을 시사합니다.

- **Performance Highlights**: 연구 결과, LLM들은 이분법적 성별 범주(여성과 남성)에 대해 구별된 반응을 나타내며, 여성이 따뜻함 면에서, 남성이 능력 면에서 강한 평가를 받는 것을 확인했습니다. 그러나 LLM들은 양성애자(bisexual)와 비이분법(nonbinary)적으로 성별을 표현하는 사람들에 대해 보다 부정적인 묘사를 생성하며, 이로 인해 창의적인 글쓰기와 같은 사용 사례에서의 재현적 해악(representational harm)에 대한 우려가 제기됩니다. 특히 모델 간의 차이가 보여졌으며, Gemini 모델이 가장 뚜렷한 차이를 보였습니다.



### Navigating Tomorrow: Reliably Assessing Large Language Models Performance on Future Event Prediction (https://arxiv.org/abs/2501.05925)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 미래 예측 업무에서 어떻게 활용될 수 있는지를 탐구하는 새로운 접근 방식을 제시합니다. 다양한 계산적 방법들이 미래 예측을 위해 제안되었지만, 큰 언어 모델이 실제로 미래 예측 작업에 대한 성능이 어떻게 되는지는 충분히 연구되지 않았습니다. 연구자들은 다양한 시나리오를 통해 LLM의 예측 능력을 평가하였으며, 미래 예측에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: 연구에서는 Affirmative vs. Likelihood questioning, Reasoning, Counterfactual analysis 등 세 가지 주요 시나리오에서 LLM의 능력을 분석합니다. 이와 함께, 데이터셋의 시간적 기반을 명확히 하고, LLM의 학습 기한 이전에 발생한 사건들에 대해 예측 문제를 만듭니다. 각 질문의 시간뿐 아니라 예상되는 사건 발생 시간도 명시하는 데이터셋을 구축하여 모델의 편향성과 예측 정확성을 평가할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 연구 결과, Likelihood 접근 방식이 Affirmative 접근 방식보다 일반적으로 더 높은 성과를 보였으며, 확률 기반 질문이 더 미묘한 이해를 제공한다고 나타났습니다. 추론을 포함할 경우 기억 비율이 향상되지만, 이는 허위 긍정이 증가함을 의미하여 정확도와 재현율 사이의 상충 관계를 강조합니다. Counterfactual 분석에서는 모델이 사소한 변화에 민감하다는 결과를 얻어 성능에 큰 영향을 미친다고 밝혔습니다.



### Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs (https://arxiv.org/abs/2501.05891)
Comments:
          The 40th ACM/SIGAPP Symposium On Applied Computing

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 활용성을 교육적 맥락에서 탐구한다. 특히, 프로그래밍 언어(Programming Languages) 관련 다지선다 문제(Multiple-Choice Questions, MCQs)에 대한 LLM의 정답률을 조사하였으며, 이를 통해 모델의 접근성과 효율성을 평가한다. 또한, 모델의 정확도를 높이기 위해 정책 재조정(fine-tuning) 및 자원 최적화 방법을 사용한다.

- **Technical Details**: 연구에 사용된 LLM은 LLaMA-2의 7B, 13B 및 70B 버전으로, 총 162개의 MCQ에 대한 성능을 평가하였다. 이를 통해 전통적인 교재 자료로 세밀하게 조정된 소규모 모델이 보다 큰 일반 모델보다 높은 정확도를 보임을 확인하였다. 이 연구는 MCQ에 대한 LLM의 정확성을 높이는 과정에서 하드웨어 요구사항과 모델의 정교화(fine-tuning) 기술들을 분석하고, 이들이 결과에 미치는 영향을 조명한다.

- **Performance Highlights**: 연구 결과, 교재 기반으로 조정된 소규모 LLM 모델이 일반적인 대형 모델보다 더 나은 성과를 나타냈다. 특히, 7B와 13B 모델의 조정 과정에서 높은 정확도를 달성하며, 이는 교육 현장에서 LLM을 활용하는 데에서 비용 효과적인 접근 방법이 될 수 있음을 시사한다. 이러한 접근 방식은 LLM을 사용하여 다지선다 문제를 보다 정확하게 해결할 수 있는 가능성을 제시한다.



### ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability (https://arxiv.org/abs/2501.05855)
- **What's New**: 본 연구에서는 복잡한 모델 계산을 인간이 이해할 수 있는 개념으로 매핑하는 개념 기반 설명의 평가 프레임워크를 소개합니다. 기존의 평가 지표들이 종종 개념 공간의 품질에만 초점을 맞추는 반면, 우리는 설명의 효과적인 전달 방식에 대해서도 고려하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 자동화된 시뮬라빌리티(simulatability)를 통한 개념 설명 평가 프레임워크를 도입합니다. 이 방법은 제공된 설명을 바탕으로 모델의 출력을 예측하는 시뮬레이터의 능력을 활용하여 전반적인 평가를 수행합니다. 대규모 인간 연구의 어려움을 극복하기 위해, 대형 언어 모델(LLMs)을 시뮬레이터로 활용하여 평가를 근사합니다.

- **Performance Highlights**: 우리는 이 프레임워크를 사용하여 포괄적인 실증 평가를 실시했으며, LLMs가 설명 방법에 대해 일관된 순위를 제공함을 보여주었습니다. 이 방법은 다양한 모델과 데이터셋에 걸쳐 확장 가능하고 일관된 평가를 가능하게 합니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### IndoNLP 2025: Shared Task on Real-Time Reverse Transliteration for Romanized Indo-Aryan languages (https://arxiv.org/abs/2501.05816)
Comments:
          7 Pages, 1 Figure, 3 Tables

- **What's New**: 이 논문은 로마자로 표기된 인도-아리아어 언어의 실시간 역전환(transliteration)이라는 공동 과제를 다룹니다. 이 과제는 저자원이 부족한 언어를 모국 문자로 정확히 변환하는 방법을 제시하고 있습니다. 11개의 등록 팀 중 4팀이 최종 평가에 참여하였으며, Sinhala, Hindi, Malayalam 언어의 역전환 모델을 포함하였습니다.

- **Technical Details**: 논문은 전통적인 규칙 기반 방법과 통계 모델의 혼합을 통한 전환 기술의 발전을 강조합니다. 최근 심층 학습(deep learning) 기반의 모델들은 특히 Transformer 아키텍처를 사용하여 장기 의존성을 효과적으로 처리할 수 있는 방법으로 주목받고 있습니다. 이러한 모델들은 인도-아리아어 언어의 로마자 표기를 모국어 문자로 역전환하는 데 기여하고 있습니다.

- **Performance Highlights**: 제안된 솔루션은 전통적인 방식의 문제를 해결할 뿐만 아니라 디지털 환경에서 저자원 언어의 사용성을 높이는 데 적극 기여합니다. 이 과제는 사용자가 고유한 작문 방식으로 작성한 텍스트를 실시간으로 정확히 변환할 수 있는 시스템을 개발하는 것을 목표로 하고 있습니다. 이로 인해 소셜 미디어 및 다른 디지털 플랫폼에서의 언어적 커뮤니케이션의 품질이 향상될 것으로 기대됩니다.



### Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models (https://arxiv.org/abs/2501.05767)
Comments:
          20 pages, 8 figures

- **What's New**: 이 논문에서는 복합적인 다중 이미지 시나리오에서 정밀한 그라운딩(grounding)을 달성하는 데 어려움을 겪고 있는 최근의 멀티모달 대형 언어 모델(MLLMs)의 발전을 다루고 있습니다. 새로운 Chain-of-Thought (CoT) 프레임워크를 통하여 단일 이미지 그라운딩과 다중 이미지 이해를 통합하며, Migician 모델을 소개하여 자유형식(free-form) 다중 이미지 그라운딩을 수행할 수 있는 최초의 모델로 자리잡았습니다.

- **Technical Details**: Migician 모델은 MGrounding-630k이라는 대규모 데이터셋으로 훈련되어, 다양한 다중 이미지 그라운딩 작업을 위한 데이터와 자유형식 지침 데이터를 포함합니다. 또한, 모델의 성능을 평가하기 위한 MIG-Bench라는 종합 벤치마크를 제시하여 다중 이미지 그라운딩의 성능을 정량적으로 측정하고 있습니다. 이를 통해 Migician은 이전의 MLLMs보다 21.61% 우수한 다중 이미지 그라운딩 능력을 입증하였습니다.

- **Performance Highlights**: Migician 모델은 실험 결과, 현존하는 최고의 MLLMs보다 훨씬 나은 성능을 기록하였으며, 특히 다양한 환경에서의 비전-언어 작업을 수행하는 데서 탁월한 능력을 보여주고 있습니다. 최종적으로, 이 연구는 MLLMs의 잠재력과 극복해야 할 도전을 탐구하고, Migician과 MGrounding-630k, MIG-Bench의 개발을 통해 다중 이미지 그라운딩 분야의 새로운 기준을 설정하고자 하였습니다.



### Controlling Large Language Models Through Concept Activation Vectors (https://arxiv.org/abs/2501.05764)
- **What's New**: 이 논문에서 제안하는 Generation with Concept Activation Vector (GCAV)은 기존의 LLM 제어 방법들과 달리, 자원 소모가 큰 미세 조정을 요구하지 않고 경량화된 모델 제어 프레임워크를 제공합니다. GCAV는 특정 개념의 활성화 벡터를 학습하여 이 벡터를 LLM의 활성화 레이어에 적용하고, 예를 들어, 독성 개념 벡터를 제거하여 제어할 수 있도록 합니다. 이를 통해 LLM의 생성 결과를 인간의 가치와 윤리적 원칙에 맞춰 조정할 수 있는 가능성을 열어줍니다.

- **Technical Details**: GCAV는 활성화 벡터를 통해 LLM을 제어하는 방법으로, 이 과정에서 개념에 따른 조정 강도를 계산하여 각 입력 샘플에 맞는 세밀한 조정이 가능합니다. 이 프레임워크는 독성 저감, 정서 제어, 주제 제어, 언어 스타일 제어와 같은 다양한 측면에서 뛰어난 성능을 보여줍니다. Contrastive prompts를 사용하여 타겟 개념이 포함된 생성과 그렇지 않은 생성에 대한 활성화 벡터를 수집하며, 이 벡터를 활용하여 LLM의 출력이 원하는 속성에 부합하도록 만듭니다.

- **Performance Highlights**: 실험 결과, GCAV 프레임워크는 기존의 제어 방법들에 비해 훨씬 더 세밀한 조정과 함께 우수한 성능을 보여줍니다. 각 입력마다 조정 강도를 계산하여 여러 개념을 동시에 제어하는 것이 가능하며, 이는 다양한 응용 분야에서 유용하게 사용될 수 있습니다. 전체적으로 GCAV는 LLM의 안전성뿐만 아니라 사용자 맞춤형 생성에서도 매우 효과적이라는 것을 입증하였습니다.



### Bridging Dialects: Translating Standard Bangla to Regional Variants Using Neural Models (https://arxiv.org/abs/2501.05749)
Comments:
          Accepted in 2024 27th International Conference on Computer and Information Technology (ICCIT)

- **What's New**: 이 연구는 방글라어의 여러 지역 방언을 번역하기 위한 새로운 접근 방식을 제시합니다. 특히 고급 신경 기계 번역(Neural Machine Translation, NMT) 모델인 BanglaT5, mT5 및 mBART50을 사용하여 표준 방글라어를 지역 방언으로 변환하는 데 초점을 맞추고 있습니다. 이 연구는 언어의 다양성을 보존하고 방언 사용자 간의 소통을 향상시키려는 필요성에 의해 촉발되었습니다.

- **Technical Details**: 연구자는 'Vashantor' 데이터세트를 사용하여 32,500개의 문장으로 구성된 훈련을 진행했습니다. 각 문장은 지역 방언, 표준 방글라어 및 영어로 번역된 형태로 제공되어 방언의 엄청난 변화를 보여줍니다. 주요 트랜스포머 기반의 NMT 모델들은 연속적인 표현 및 주의 메커니즘을 기반으로 하여 언어 간의 관계를 효율적으로 학습하고 허용 가능한 성능으로 결과를 도출합니다.

- **Performance Highlights**: BanglaT5 모델은 12.3%의 문자 오류율(Character Error Rate, CER)과 15.7%의 단어 오류율(Word Error Rate, WER)로 다른 모델들보다 우수한 성능을 보였습니다. 이는 특정 방언의 뉘앙스를 잡아내는 데 있어 이 모델의 효과성을 강조하며, 지역 방언 지원 및 언어 기술의 포괄성을 위한 기여도를 나타냅니다.



### Enabling Scalable Oversight via Self-Evolving Critic (https://arxiv.org/abs/2501.05727)
- **What's New**: 이번 논문에서는 SCRIT (Self-evolving CRITic)이라는 새로운 프레임워크를 제안하여, 대형 언어 모델(LLMs)의 비판 능력을 스스로 발전시키는 방법을 제시합니다. SCRIT는 인간 평가가 어려운 작업에서 LLM의 피드백 효율성을 높이는 것을 목표로 하며, 스스로 훈련하는 방식으로 비판 데이터를 생성합니다. 특히, 대조적 자기 비판 기법과 자기 검증 메커니즘을 통해 무인 감독으로 비판 품질을 향상시킵니다.

- **Technical Details**: SCRIT의 핵심 단계는 먼저 참조 솔루션을 바탕으로 학생 솔루션을 분석하고 비판하는 대조적 비판 기법을 개발하는 것입니다. 이어서 LLM은 생성된 비판이 수학적으로 유효한 솔루션으로 이어지는지를 자기 검증합니다. 이 두 단계는 고품질 비판 데이터를 생성하고, LLM의 비판 능력을 지속적으로 향상시키는 데 기여합니다.

- **Performance Highlights**: SCRIT은 Qwen2.5-72B-Instruct 모델을 기반으로 하여 비판 수정 및 오류 식별 작업에서 최대 10.3%의 성능 향상을 달성했습니다. 다양한 데이터 세트와 평가 프로토콜에서 일관된 개선 결과를 보여주며, SCRIT 구현 시 기존 모델의 출력을 크게 향상시켰습니다. 또한 SCRIT은 데이터와 모델 크기가 커질수록 성능이 긍정적으로 확장됨을 나타내어, LLM의 비판 능력 강화에 있어 중요한 진전을 보여줍니다.



### How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond (https://arxiv.org/abs/2501.05714)
Comments:
          23 pages

- **What's New**: 이번 논문은 인공지능 모델들이 자율적 에이전트로 진화하고 있는 현상을 조명하며, 이를 바탕으로 인간-모델 협력을 체계적으로 정리한 최초의 리뷰 논문입니다. 특히 새로운 분류 체계를 도입하여 기존 접근 방식을 통합적으로 요약할 수 있는 틀을 제공합니다. 또한, 이 연구는 향후 더 심화된 연구로 나아갈 수 있는 기회를 모색하고 있음을 강조합니다.

- **Technical Details**: 저자들은 인간-모델 협력의 정의와 원칙을 다루며, 협력의 기초가 되는 공유 목표를 중심으로 논의합니다. 논문에서는 협력의 공식화를 위한 새로운 체계적 분류 방법론을 제안하며, 각각의 협력 유형에 따른 역할 프레임워크를 정의합니다. 이러한 프레임워크는 모델과 인간 간의 협력 방식과 의사결정의 책임이 어떻게 나뉘는지를 설명합니다.

- **Performance Highlights**: 인간-모델 협력을 통해 데이터 주석화, 정보 탐색, 창의적 글쓰기 및 실제 문제 해결 등의 다양한 NLP 작업에서 효율성을 높일 수 있는 잠재력이 드러났습니다. 이 연구는 상호작용을 위한 다양한 사용자 인터페이스를 제공함으로써 누적된 연구 결과를 종합하여 향후 연구 방향과 기술적 고려 사항을 제시하며, 이 분야의 발전에 기여할 것으로 기대됩니다.



### Multi-Step Reasoning in Korean and the Emergent Mirag (https://arxiv.org/abs/2501.05712)
Comments:
          11 pages, 7 figures

- **What's New**: HRMCR (HAE-RAE Multi-Step Commonsense Reasoning)라는 새로운 벤치마크가 소개되었습니다. 이 벤치마크는 한국 문화에 대한 지식을 통합하여 다단계 추론을 수행하는 대형 언어 모델의 능력을 평가하는 데 중점을 두고 있습니다. 질문은 자동으로 생성되며, 교육 플롭 수에 따른 모델의 성능 차이를 조사한 결과, 특정 임계값을 초과해야 성능이 급격히 향상되는 경향을 보였습니다.

- **Technical Details**: HRMCR은 두 가지 하위 집합인 Date와 Zodiac으로 구성되어 있습니다. Date 하위 집합은 한국의 명절 및 전통 날짜 표현과 관련된 질문들을 다루며, 각 질문은 다섯 단계의 해결을 요구합니다. Zodiac 하위 집합은 한국의 독특한 나이 시스템 및 대화 중 나이 표현을 포함하는 더 긴 질문으로, 최대 일곱 단계의 추론이 필요합니다.

- **Performance Highlights**: HRMCR 벤치마크는 매우 도전적임을 나타냅니다. 평가에 사용된 주도 모델인 GPT-4o는 30% 미만의 점수를 기록하였으며, 최신 모델 O1은 평균 45점을 달성하여 이전 모델보다 성능이 크게 향상되었습니다. 그러나 모든 모델은 여전히 50% 정확도에 도달하지 못하고 있어 한국의 문화적 맥락에서의 다단계 추론이 얼마나 어려운지를 강조합니다.



### Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains (https://arxiv.org/abs/2501.05707)
Comments:
          22 pages, 13 figures, 7 tables; Project page at this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 자기 개선(self-improvement)을 위한 새로운 접근 방식을 제안합니다. 기존의 모델들을 다수의 에이전트로 구성하여 각 모델이 상호작용을 통해 독립적으로 전문화되는 방식입니다. 이 방법은 다수의 모델이 역할을 분담하여 응답의 다양성과 사고 체계를 유지하면서 더욱 많은 반복을 통한 성능 향상을 가능하게 합니다.

- **Technical Details**: 논문에서는 다중 에이전트 설정에서 LLM 모델을 전문화하는 방법을 도입합니다. 다중 에이전트 토론(multiagent debate) 방식을 통해 각 모델이 생성한 데이터를 활용하여 피드백을 주고받으며, 반복적 훈련을 통해 다양성을 유지하고 독립적으로 전문화합니다. 이 과정에서 데이터는 독립된 데이터 세트에서 수집되어 각 모델에 맞게 조정됩니다.

- **Performance Highlights**: 제안한 방법은 여러 추론 작업에 대해 정량적으로 유효성을 입증하며, 오픈 소스 LLM부터 상용 LLM에 이르기까지 다양한 모델에 적용 가능합니다. 실험 결과, 단일 에이전트 자기 개선 방법보다 훨씬 더 많은 반복을 통한 성능 개선을 확인했으며, 새로운 데이터 세트에서도 우수한 일반화를 보여주었습니다.



### Linguistic Entity Masking to Improve Cross-Lingual Representation of Multilingual Language Models for Low-Resource Languages (https://arxiv.org/abs/2501.05700)
- **What's New**: 이번 논문에서는 다국어 사전학습 언어 모델(multiPLMs)의 성능을 개선하기 위해 Linguistic Entity Masking (LEM)이라는 새로운 마스킹 전략을 도입합니다. 기존의 마스킹 방식인 Masked Language Modeling(MLM)과 Translation Language Modeling(TLM)과는 달리, LEM은 명사, 동사, 명명된 개체와 같은 언어적 개체 유형에만 마스킹을 제한합니다. 이로 인해 문맥을 더 많이 유지하면서도, 다국어 모델의 교차 언어 표현력을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 XLM-R을 multiPLM으로 사용하여 마스킹 전략 LEM을 적용한 지속적인 재훈련을 진행하였습니다. LEM에서는 언어적 개체 범위 내에서 단일 토큰만을 마스킹하여, 기존의 연속 마스킹 기법과의 차별성을 두었습니다. 이러한 마스킹 방식은 영어-신할라어, 영어-타밀어, 신할라어-타밀어의 3개 저자원 언어 쌍을 대상으로 한 실험에 사용되었습니다.

- **Performance Highlights**: 실험 결과, LEM을 활용해 지속적으로 재훈련된 XLM-R 모델(XLM-RLEM)이 MLM+TLM으로 재훈련된 모델(XLM-RMLM+TLM)보다 모든 다운스트림 작업에서 우수한 성능을 발휘하였다는 것을 보여줍니다. 특히, 비텍스트 마이닝, 병행 데이터 큐레이션 및 코드 혼합 감정 분석 등에서 효과적인 결과를 나타내었습니다. 이 연구는 저자원 언어에 대한 교차 언어적 표현 향상에 기여할 것으로 예상됩니다.



### Cascaded Self-Evaluation Augmented Training for Efficient Multimodal Large Language Models (https://arxiv.org/abs/2501.05662)
- **What's New**: 최근 효율적인 다중 모달 대형 언어 모델(EMLLMs)이 급속히 발전하였습니다. 본 논문에서는 CoT(Chain-of-Thought) 추론과 단계별 자기 평가(self-evaluation)를 통합하여 EMLLM의 성능을 향상시킨 방법을 제안합니다. 하지만 제한된 매개변수 때문에 EMLLM은 추론하는 동안 자기 평가를 효과적으로 이용하기 어려운 상황입니다. 이를 해결하기 위해 우리는 Self-Evaluation Augmented Training (SEAT)이라는 방법을 소개하며, 후속 연구를 위한 중요한 데이터셋을 구성하였습니다.

- **Technical Details**: SEAT는 더 강력한 EMLLM을 이용하여 CoT 추론 및 데이터를 생성하고, MLLMs가 선별한 데이터를 사용하여 EMLLM을 교육합니다. 그러나 긴 프롬프트를 처리하고 CoT 추론의 품질을 유지하는 데 문제가 발생합니다. 이를 해결하기 위해 카스카데드 자기 평가 향상 훈련(Cas-SEAT) 방법을 제안하며, 긴 입력을 짧고 구체적인 작업 프롬프트로 나누어 비용을 절감하는 방식입니다. 이 과정에서 7B 매개변수의 오픈소스 EMLLM을 사용하여 데이터의 효율성을 극대화하였습니다.

- **Performance Highlights**: 실험 결과, Cas-SEAT는 기존 방법들과 비교하여 EMLLM의 자기 평가 능력을 크게 향상시켰으며, MathVista, Math-V 및 We-Math 데이터셋에서 각각 19.68%, 55.57%, 46.79% 개선된 성과를 보였습니다. 이는 EMLLM의 CoT 추론 능력을 효과적으로 강화한 결과로, 적극적인 후속 연구를 위한 기초 자료로 활용될 것입니다. 또한 Cas-SEAT Dataset은 EMLLM의 자기 평가 향상을 위한 첫 번째 데이터셋으로, 저비용으로도 효과적인 연구에 기여할 수 있습니다.



### Iconicity in Large Language Models (https://arxiv.org/abs/2501.05643)
Comments:
          Supplementary information: this https URL

- **What's New**: 이번 논문은 인공 언어에서 소리와 의미 간의 직접적인 관계인 lexical iconicity(어휘 상징성)를 다루고 있습니다. 기존에 소리와 의미의 처리 방식에서 인간과의 차이를 탐구하며, LLMs가 그 관계를 처리하는 방법이 부족하거나 상당히 다를 것이라는 가설을 세웠습니다. 연구에서는 GPT-4가 생성한 pseudowords(유사 단어)의 의미를 인간과 LLM 기반 참가자들이 추측할 수 있는 능력을 비교하여, LLM이 이 iconicity를 잘 처리할 수 있는 가능성을 제시합니다.

- **Technical Details**: 레미니션은 '의미와 형태의 직접적인 관계 
'로 정의되며, 이와 관련된 복잡한 세부 사항으로써 LLMs의 경우 의미와 음성을 직접적으로 처리하지 못하고 있습니다. LLM의 훈련 데이터는 오디오 녹음을 포함하지 않으며, 이는 그들이 iconicity를 전통적으로 접근하는 방법으로는 성취하기 어렵다는 점을 의미합니다. 연구에서는 두 단계의 실험을 설계하였고, 첫 번째 단계에서는 GPT-4가 고유한 소리-상징적 속성을 갖는 인공 언어의 lexicon을 생성했습니다.

- **Performance Highlights**: 연구 결과, 인간 참가자들은 생성된 iconic language에서의 pseudoword 의미 예측을 자연어보다 훨씬 더 정확하게 수행했습니다. LLM 기반 참가자들은 인간보다 더욱 높은 성공률을 보였으며, 이는 LLM이 다양한 유형의 cognition(인지)적 처리와 인간 유사한 학습을 효과적으로 시뮬레이션할 수 있음을 보여줍니다. 이러한 결과는 LLMs가 iconicity를 학습하고 처리할 수 있는 능력을 시사합니다.



### Automating Date Format Detection for Data Visualization (https://arxiv.org/abs/2501.05640)
Comments:
          2025 International Conference on Advanced Machine Learning and Data Science (AMLDS 2025)

- **What's New**: 이 논문에서는 데이터 준비에서 중요한 병목인 날짜 파싱(date parsing)을 해결하기 위해 두 가지 알고리즘을 제안합니다. 첫 번째는 최소 엔트로피(minimum entropy)에 기반하고, 두 번째는 자연어 처리(NLP) 모델을 기초로 한 방법입니다. 이 알고리즘들은 문자열 데이터로부터 날짜 형식을 자동으로 추출하여 90% 이상의 정확도를 달성합니다.

- **Technical Details**: 본 연구에서는 ICU 오픈 소스 프로젝트의 날짜 형식 언어를 활용하여 날짜 패턴을 생성합니다. 두 알고리즘은 각각 최소 설명 길이(MDL) 및 자연어 처리(NLP) 접근 방식을 기반으로 하며, 이러한 접근은 날짜 데이터의 다양성과 복잡성을 다루기 위해 최적화되었습니다. 더욱이 이 접근법은 병렬 처리를 통해 런타임을 개선하고, 사용자 선택 없이 자동으로 날짜 형식을 적용할 수 있습니다.

- **Performance Highlights**: 제안된 두 알고리즘은 30,000개 데이터 열의 코퍼스에서 평가되어 95% 이상의 파싱 정확도를 기록했습니다. 이 높은 정확도는 분석 도중 사용자가 워크플로를 방해받지 않고도 신속하고 정확한 데이터 처리를 가능하게 합니다. 따라서 이 연구는 데이터 시각화 도구와 데이터베이스에서 날짜 형식을 효과적으로 통합하는 데 매우 유용한 기초를 제공합니다.



### The Impact of Model Scaling on Seen and Unseen Language Performanc (https://arxiv.org/abs/2501.05629)
Comments:
          Accepted at SEAS Workshop at AAAI25

- **What's New**: 이 연구는 204개의 언어에서 다국어(Large Language Model, LLM)의 성능과 확장 비율(scaling behavior)을 상세히 조사한 첫 번째 연구입니다. 특히, 사전 훈련(pretraining) 중에 본 언어와 보지 못한 언어에 대한 성능 차이를 분석했습니다. 또한 다양한 자원 수준(resource levels)과 모델 크기에 따른 성능 현상을 다루고 있습니다. 이 연구는 다국어 LLM의 효과성을 이해하는 데 중요한 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 xglm, bloom 및 bloomz 모델을 포함하여 총 14개의 다국어 LLM을 조사했습니다. 다양한 형태의 모델 크기를 포함하여, 주제 분류(topic classification)와 기계 번역(machine translation) 작업을 수행하며 두 가지 유형의 설정(제로샷(zero-shot) 및 몇 샷(few-shot))에서 성능을 평가했습니다. SIB-200 및 FLORES-200 데이터셋을 활용해 204개 언어를 대상으로 성능을 비교했습니다.

- **Performance Highlights**: 제로샷 설정에서는 모델 크기가 성능에 큰 영향을 미치지 않는 반면, 두 샷 상황에서는 대형 모델이 분명한 성능 개선을 보였습니다. 다국어 텍스트 분류에서는 자원 수준이 인식된 언어와 보지 못한 언어 모두에서 성능에 더 큰 상관관계를 가지는 것으로 나타났습니다. 전반적으로, 자료 수준이 모델의 성능을 예측하는 데 중요하게 작용하며 다국어 LLM의 효과성과 그 운영 전략에 대한 통찰력을 제공합니다.



### Harmonizing Metadata of Language Resources for Enhanced Querying and Accessibility (https://arxiv.org/abs/2501.05606)
Comments:
          2024 5th International Conference on Computers and Artificial Intelligence Technology (CAIT 2024)

- **What's New**: 이 논문은 다양한 언어 리소스(LR)의 메타데이터를 조화롭게 통합하는 방법을 다룹니다. Linked Data와 RDF 기법을 활용하여 여러 출처의 데이터를 DCAT 및 META-SHARE OWL 온톨로지를 바탕으로 통합된 모델로 구현하였습니다. 새롭게 개발된 portal인 Linghub을 통해 텍스트 기반 검색, faceted browsing, 그리고 advanced SPARQL 쿼리를 지원합니다. 실제 사용자 쿼리를 평가하여 Linghub이 실제 사용자 요구를 충족하는지 검토하였고, 이 과정에서 메타데이터의 중요한 문제를 강조하면서 오픈 어휘 및 표준 준수를 권장합니다.

- **Technical Details**: 이 논문에서는 메타데이터 통합을 위해 Linghub라는 링크드 데이터 기반 포털을 개발했습니다. Linghub는 META-SHARE, CLARIN VLO, LRE-Map, Datahub.io와 같은 다양한 레파지토리에서 메타데이터 항목을 색인화하고 집계합니다. RDF 및 링크드 데이터의 최첨단 개념을 적용하여 서로 다른 출처의 메타데이터 설명을 동일한 데이터 스키마로 매핑하며, 세미틱 웹의 표준 어휘 및 언어 리소스의 새로운 온톨로지를 통해 정보를 조화시킵니다.

- **Performance Highlights**: Linghub은 Corpora Mailing List에서 발송된 실제 사용자 요구 요청을 분석함으로써 사용자의 필요에 얼마나 잘 응답할 수 있는지를 평가하였습니다. 결과적으로, Linghub은 일부 제한에도 불구하고 많은 사용자 요청을 성공적으로 처리할 수 있음을 보여주었습니다. 이 평가는 언어 리소스 레파지토리의 요청에 대한 응답 능력을 평가한 첫 번째 총체적인 시도로, 향후 언어 자원 활용의 효율성과 표준화에 기여할 것입니다.



### Exploring Large Language Models for Translating Romanian Computational Problems into English (https://arxiv.org/abs/2501.05601)
Comments:
          12 pages

- **What's New**: 최근 연구에서는 대형 언어 모델(LLM)이 루마니아어에서 영어로 번역될 때 수학 및 컴퓨터 과학 작업에서 성능이 떨어지는 것으로 나타났습니다. 이 연구는 구조적으로 잘 구성된 프롬프트를 제공할 경우 LLM이 덜 일반적인 언어로 번역할 때 성능을 유지하거나 향상시킬 수 있음을 보여주었습니다. LLM은 적절한 감독 하에 IOI 스타일의 문제를 자동으로 번역하는 데 신뢰할 수 있는 도구가 될 수 있습니다.

- **Technical Details**: 이 연구는 루마니아어에서 영어로 번역된 IOI 스타일 문제에서 번역 전략을 평가하고, LLM 성능을 높이는 프롬프트를 최적화하며, LLM의 번역 정확성과 성능 안정성을 평가합니다. 여러 LLM(예: OpenRoLLM, Llama 3.1 8B, GPT-4o 등)에 대해 번역 결과를 반복적으로 분석하고, 오류 분석을 통해 번역의 문법적 및 의미적 오류를 상세하게 조사합니다.

- **Performance Highlights**: 성능 분석 결과, GPT-4는 루마니아어 작업에서 높은 성능을 나타냈고, LLM이 제공하는 번역의 품질이 인증된 전문가에 의해 평가되었습니다. 또한 OJI 데이터셋의 루마니아어 문제에 대해 정확한 영어 번역을 추가하여 향후 LLM 훈련과 평가의 유용성을 강화했습니다. 전반적으로 LLM은 동등한 이해도를 유지하면서 번역된 문제를 해결할 수 있도록 인간 문제 해결자를 효과적으로 지원할 수 있는 잠재력을 가지고 있습니다.



### LLMQuoter: Enhancing RAG Capabilities Through Efficient Quote Extraction From Large Contexts (https://arxiv.org/abs/2501.05554)
- **What's New**: 본 논문에서는 LLMQuoter라는 경량화된 distillation 기반 모델을 소개합니다. 이 모델은 Retrieval Augmented Generation (RAG) 프로세스를 향상시켜 중요 텍스트 증거를 추출하고, 이를 통해 하위 작업의 추론 성능을 개선합니다. LLMQuoter는 LLaMA-3B 아키텍처를 바탕으로 하며, 15,000개의 HotpotQA 샘플에 대해 Low-Rank Adaptation (LoRA)을 사용하여 세밀하게 조정되었습니다.

- **Technical Details**: LLMQuoter는 'quote-first-then-answer' 전략을 채택하여 중요한 인용구를 식별하고, 이를 추출하여 추론 모델에 전달합니다. 이 접근법은 전통적인 전체-context 기법과는 달리, 각 단계가 분리되어 추론 과정을 간소화하고, 인지 부담을 줄입니다. 이를 통해 LLMQuoter는 작은 모델과 큰 모델 모두에서 뛰어난 정확도를 달성할 수 있습니다.

- **Performance Highlights**: 실험 결과 LLMQuoter는 RAFT와 같은 RAG 기법들에 비해 경쟁력 있는 정확도 향상을 보여줍니다. 또한 LLMQuoter의 경량화된 성질은 자원 제한이 있는 연구자와 실무자들에게도 고급 RAG 기능에 대한 접근을 민주화합니다. 이는 복잡한 작업을 간소화하고, 자원 활용성을 높이는 데 기여합니다.



### The dynamics of meaning through time: Assessment of Large Language Models (https://arxiv.org/abs/2501.05552)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 개념의 역사적 맥락과 의미의 발전을 어떻게 이해하는지를 평가하는 데 중점을 두고 있습니다. 다양한 시대에서 용어를 해석하는 능력을 분석하기 위해 여러 모델을 비교하고, 특정한 과제를 통해 모델의 반응을 측정합니다.

- **Technical Details**: 연구팀은 다양한 도메인에서 선택한 용어 세트를 분석하며, 맞춤형 프롬프트(prompts)와 객관적인 메트릭스(metrics, 예: perplexity 및 단어 수) 뿐만 아니라 주관적인 전문가 평가를 통해 반응을 측정했습니다. 주목할 만한 모델들인 ChatGPT, GPT-4, Claude, Bard, Gemini, Llama 등이 비교 분석되었습니다.

- **Performance Highlights**: 각 모델이 역사적 맥락과 의미의 변화에 대해 다르게 반응하는 것을 발견하였고, 이는 각 모델의 강점과 한계를 부각시킵니다. 이러한 통찰력은 대규모 언어 모델을 개선하고, 역사적 텍스트 분석, AI 설계 및 디지털 인문학의 응용에 기여할 수 있는 기초를 제공합니다.



### The more polypersonal the better -- a short look on space geometry of fine-tuned layers (https://arxiv.org/abs/2501.05503)
Comments:
          Neuroinformatics 2024

- **What's New**: 이 논문은 심층 학습 모델의 해석(interpreting) 분야에서 특히 언어 모델에 대한 새로운 접근 방식을 제안합니다. 저자들은 BERT 모델에 새로운 문법 모듈(grammatical module)과 문법 구조(polypersonality)가 포함된 데이터를 추가하여 훈련할 때 내부 표현(internal representation)의 변화를 분석합니다.

- **Technical Details**: BERT 모델에 단일 문법 레이어를 추가함으로써 모델이 새로운 문법 체계와 기존의 문법 체계를 구분하는 과정을 보여줍니다. 이는 모델의 의사 결정 과정에서 패턴을 식별할 뿐만 아니라, 내부 구조의 특성을 이해하는 데 기여합니다.

- **Performance Highlights**: 저자들은 추가된 문법 레이어 덕분에 모델의 perplexity 메트릭(perplexity metrics)에서 전체적인 성능이 향상된 것을 발견했습니다. 이러한 개선은 모델의 문법 처리 능력을 더욱 향상시키는 중요한 발견입니다.



### Spatial Information Integration in Small Language Models for Document Layout Generation and Classification (https://arxiv.org/abs/2501.05497)
Comments:
          8 pages. Symposium on Applied Computing 2025

- **What's New**: 이번 논문에서는 문서 레이아웃 이해(Document Layout Understanding) 분야에서 새로운 방법론을 제안합니다. 반 구조화된 데이터(semi-structured data) 부족 문제를 해결하기 위해 합성 레이아웃 정보(synthetic layout information)를 생성하는 방식을 도입했습니다. 이 방식은 기존의 LayoutTransformer와 비교했을 때, 더 나은 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 문서의 공간적 배치(spatial arrangement) 분석을 중심으로 진행되어 여타 모델들보다 우수한 결과를 도출하였습니다. 연구에서 제안된 방법은 머신러닝 모델 교육을 위한 공개 데이터셋의 부족을 극복할 수 있는 가능성을 내포하고 있습니다. 특히, 경계 상자 정보(bounding box information)가 텍스트 분류(text classification)에 긍정적인 영향을 미칠 수 있다는 점도 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 LayoutTransformer보다 우수한 성과를 기록하였으며, 이는 반 구조화된 문서의 인식 성능을 향상시킬 수 있음을 시사합니다. 또한, 문서의 다양한 레이아웃 구조를 이해함으로써 실생활에서 접하는 회계 보고서, 구매 주문서 및 영수증 등의 문서 처리가 개선될 수 있다는 점도 강조되고 있습니다.



### The Future of AI: Exploring the Potential of Large Concept Models (https://arxiv.org/abs/2501.05487)
- **What's New**: 이번 연구는 기존의 대형 언어 모델(LLMs)의 한계를 극복하기 위해 대형 개념 모델(LCMs)을 도입했습니다. LCMs는 텍스트와 같은 개별 토큰 대신 더 큰 의미 단위인 개념을 사용하여 정보 처리를 수행하는 방식으로 이 모델은 보다 정교한 의미 추론과 문맥 인식 결정을 가능케 합니다. 이 기술은 기존의 LLMs와는 달리 고수준의 추론을 기반으로 하여 사용자에게 더 의미 있는 출력을 제공합니다.

- **Technical Details**: LCMs의 아키텍처는 개념 인코더, LCM 코어, 개념 디코더라는 세 가지 주요 구성 요소로 이루어져 있습니다. 이들 구성 요소는 입력 정보를 의미론적 임베딩으로 변환하고 고수준의 추론을 수행한 다음, 다시 텍스트나 음성으로 변환하는 기능을 수행합니다. 특히, 개념 인코더는 다양한 입력 형식을 통합하여 동일한 임베딩 공간에 매핑하는 능력을 지니고 있어, 텍스트와 음성을 포함한 여러 형식에서 일관된 처리를 가능하게 합니다.

- **Performance Highlights**: LCMs는 여러 언어와 모드를 아우르는 작업에서 뛰어난 성능을 발휘하며, 다국어 및 다중 모드 작업을 통해 실시간 번역과 전사에서 우수한 결과를 제공합니다. 이러한 모델은 또한 긴 문서의 내용을 합성하고 확장하는 데 효율적이며, 언어 및 양식에 구애받지 않고 정보를 처리할 수 있는 능력을 통해 복잡한 작업을 용이하게 다룰 수 있습니다.



### S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis (https://arxiv.org/abs/2501.05485)
- **What's New**: 이 논문에서는 문서 청크(chunking) 작업의 중요성을 강조하며, 기존의 방법들이 문서 내의 공간적 레이아웃을 무시하고 있다는 문제를 해결하기 위해 새로운 하이브리드 접근 방식을 제안합니다. 이 방법은 레이아웃 구조, 의미 분석, 공간적 관계를 통합하여 문서 청크의 응집력과 정확성을 향상시킵니다. 또한, 이 접근법은 복잡한 레이아웃을 가진 문서에서도 뛰어난 성능을 보이며, 토큰 길이 제한을 준수할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계인 영역 감지(region detection)와 영역 레이아웃 정렬(region layout ordering)로 구성됩니다. 첫 번째 단계에서는 문서 내 각 분류된 영역에 대한 바운딩 박스(bbox) 데이터를 추출하고, 두 번째 단계에서는 이러한 영역들을 구조적 유형에 따라 합리적인 순서로 배열합니다. 이후 그래프를 구성하고, 가중치를 계산하며, 클러스터링을 통해 일관된 청크로 나누는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과 제안된 하이브리드 접근법은 PubMed와 arXiv에서 다양한 레이아웃과 내용을 가진 연구 논문 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다. 이 방법은 문서 내의 의미적 관계와 공간적 관계를 모두 고려하여 청크를 생성함으로써, 복잡한 문서에서도 적절한 분할을 수행할 수 있습니다. 또한, 평균적으로 높은 품질의 청크를 생성하여 정보 검색, 요약 및 질문 답변과 같은 NLP 작업에서 더 나은 성능을 보입니다.



### HP-BERT: A framework for longitudinal study of Hinduphobia on social media via LLMs (https://arxiv.org/abs/2501.05482)
- **What's New**: 본 연구는 COVID-19 팬데믹 동안과 그 이후에 X(구 Twitter)에서의 Hinduphobia(힌두포비아) 감정 분석 및 남용 감지 프레임워크를 제안합니다. 이 프레임워크는 커뮤니티 긴장을 완화할 수 있는 데이터 기반의 통찰을 제공하며, Hinduphobia의 발생 빈도와 강도를 평가하는 데 사용됩니다. 설계된 'Hinduphobic COVID-19 X Dataset'는 8,000개의 트윗으로 구성되어 있으며, 이를 통해 HP-BERT 모델이 개발됩니다.

- **Technical Details**: 이 연구에서 사용된 프레임워크는 사전 훈련된(Large Language Models, LLMs) 및 미세 조정(fine-tuned)된 모델들을 활용하여, 폴리시 정치와 관련된 여러 요소를 분석합니다. HP-BERT 모델은 여러 라벨의 감정 분석을 위해 SenWave 데이터를 사용해 추가로 미세 조정되었습니다. 연구 데이터는 27.4 백만 개의 트윗에서 추출되었으며, 이를 통해 정치적인 서사와 잘못된 정보가 커뮤니티 내 긴장에 미친 영향을 분석합니다.

- **Performance Highlights**: COVID-19 사례의 급증과 Hinduphobia 수사의 폭증 간의 강한 상관관계가 발견되었습니다. 이는 정치적 서사, 잘못된 정보 및 특정 농담이 어떻게 커뮤니티의 분열을 야기했는지를 보여 줍니다. 결과에 따르면, 소셜 미디어에서의 자동 모니터링 및 해악적인 콘텐츠의 제거가 분열적인 담론을 줄이는 데 도움이 될 것이라고 제안하고 있습니다.



### The \textit{Questio de aqua et terra}: A Computational Authorship Verification Study (https://arxiv.org/abs/2501.05480)
- **What's New**: 이 연구는 다빈치의 'Questio de aqua et terra'의 진위 여부를 컴퓨터 기반 저자 인증(authorship verification, AV) 기술을 활용하여 조사합니다. 기존의 고전적인 필로로지적 분석과는 달리, 이 연구는 13세기 및 14세기 라틴어 텍스트의 말뭉치(corpus)를 구축하고 이를 통해 AV 시스템을 평가합니다.

- **Technical Details**: 연구에서 사용된 AV 시스템은 감독 학습(supervised learning) 기법과 스타일로미트리(stylometry) 기반 방법론을 결합하여 개발되었습니다. 특히, 이번 연구는 미니멀 클래스의 인공 훈련 예제를 생성하는 Distributional Random Oversampling (DRO) 기법을 AV에 처음 적용한 사례로 주목받고 있습니다. 통계적 비교에 따라, 이 시스템은 높은 검증 정확도(F1=0.970)를 달성하였습니다.

- **Performance Highlights**: AV 시스템을 'Questio'에 적용한 결과, 해당 텍스트의 진위에 대해 높은 신뢰도의 예측 결과를 도출했습니다. 그 성능은 다양한 문헌 장르에도 불구하고 뛰어난 정확도를 유지했습니다. 이렇게 얻은 결과들은 'Questio에 대한 저자 논쟁'에 기여하며, 문화유산에 대한 저자 인증 분야에서도 DRO의 잠재력을 강조합니다.



### Practical Design and Benchmarking of Generative AI Applications for Surgical Billing and Coding (https://arxiv.org/abs/2501.05479)
Comments:
          21 pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 의료 분야의 청구 및 코딩 프로세스를 위해 Generative AI 도구를 개발하는 새로운 전략을 제안합니다. 특히, 기존의 Large Language Models (LLMs)가 ICD-10 및 CPT 코드 생성을 할 때의 정확도가 낮다는 문제를 해결하고자 하였습니다. 이 모델은 접근성과 환자의 개인 정보 보호를 균형 있게 고려하면서, 의료 청구 및 코딩의 정확성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: PHI-3 Mini 및 PHI-3 Medium 모델을 기관 데이터를 활용하여 미세 조정(fine-tuning)하고, 결과를 PHI-3 기본 모델 및 GPT-4o 모델과 비교하였습니다. 연구는 수술 후 보고서를 입력하고, 환자의 청구 청구서와 관련된 ICD-10, CPT, Modifier 코드를 생성하는 과정을 포함합니다. 성능 측정은 코드 생성의 정확성, 잘못된 코드의 비율, 청구서 형식의 충실도를 기준으로 진행하였습니다.

- **Performance Highlights**: 미세 조정된 두 개의 모델은 GPT-4o 모델과 비교하여 더 좋은 성능을 보였으며, 특히 Phi-3 Medium 모델이 가장 우수한 결과를 나타냈습니다. 이 모델의 ICD-10 Recall 및 Precision은 각각 72%로 알려졌으며, CPT Recall과 Precision은 각각 77%, 79%에 달했습니다. 이 모델은 또한 생성된 ICD-10 코드 중 1%, CPT 코드 중 0.6%만이 잘못된 것으로 나타났습니다.



### Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models (https://arxiv.org/abs/2501.05478)
- **What's New**: 이 연구는 로봇 공학에서 비전-언어 내비게이션(VLN) 분야에 아랍어 통합을 첫 번째로 선보이며, 다국어 Small Language Models(SLMs)와 아랍어 중심의 LLM인 Jais의 성능을 평가합니다. 분명히 부족했던 아랍어 데이터에 대한 연구의 공백을 메우면서, NavGPT 프레임워크와 R2R 데이터셋을 사용하여 아랍어와 영어 간의 의사소통이 내비게이션 추론에 미치는 영향을 평가합니다. 이를 통해, 아랍어로 지시를 받았을 때의 로봇 내비게이션 작업의 계획 및 추론 능력을 강조하였습니다.

- **Technical Details**: 본 연구는 OpenAI의 GPT-4o mini, Meta의 Llama 3 8B와 Microsoft의 Phi-3 medium 14B와 같은 최신 다국어 SLM들과 Jais 30B LLM을 NavGPT 프레임워크 내에서 비교합니다. R2R 데이터셋을 활용하여 영어 내비게이션 지시를 아랍어로 변환한 데이터셋으로 보면, 다양한 언어로 내비게이션 자원에 접근하고자 하는 양방향적 연구의 필요성을 강조합니다. 또한, 제로샷 방식으로 작업을 예측하며, 언어의 영향력에 대한 분석을 도모합니다.

- **Performance Highlights**: 실험 결과, NavGPT 프레임워크가 영어 및 아랍어 지시를 통해 높은 수준의 내비게이션 계획을 수행할 수 있음을 입증하였습니다. 그러나 일부 모델은 아랍어에서 추론 및 계획에 어려움을 겪어 언어 모델의 성능과 한계를 드러냈습니다. 이러한 발견은 아랍어 모델의 발전과 현실 세계 응용 프로그램에서의 가능성을 열어주며, 연구의 향후 방향으로 언어 모델의 계획 및 추론 능력 향상이 필요함을 강조합니다.



### IntegrityAI at GenAI Detection Task 2: Detecting Machine-Generated Academic Essays in English and Arabic Using ELECTRA and Stylometry (https://arxiv.org/abs/2501.05476)
- **What's New**: 최근 연구에서는 기계 생성 에세이를 감지하는 문제에 대한 조사가 이루어졌습니다. 이 연구는 아랍어와 영어 학술 에세이에 대해 스타일로메트릭 (stylometric) 특성으로 미리 훈련된 트랜스포머 기반 모델을 활용하여 이 문제를 해결하고자 하였습니다. ELECTRA 모델을 기반으로한 맞춤형 모델이 영어와 아랍어 학술 에세이에 대해 훈련되었으며, 탁월한 F1-score를 달성했습니다.

- **Technical Details**: 이 연구는 'GenAI Content Detection Task 2'라는 과제에 참가하여 진행되었습니다. 연구자들은 기계 생성 콘텐츠를 감지하기 위한 데이터셋을 제공하였으며, 이 데이터셋은 생성적 AI 모델과 인간이 작성한 에세이로 구성되었습니다. 모델 학습에 사용된 데이터셋은 아랍어 및 영어의 스타일로메트릭 특성을 포함하고 있으며, 여러 기계 학습 모델을 활용하여 성능을 평가했습니다.

- **Performance Highlights**: 제안된 모델은 영어 서브 태스크에서 99.7%의 F1-score를 기록하여 26개 팀 중 2위를 차지했습니다. 아랍어 서브 태스크에서는 98.4%를 기록하여 23개 팀 중 1위를 달성했습니다. 이러한 결과는 기계 생성 텍스트 감지 모델이 특히 뛰어난 성능을 보임을 나타냅니다.



### Retrieval-Augmented Generation by Evidence Retroactivity in LLMs (https://arxiv.org/abs/2501.05475)
- **What's New**: 본 논문은 Retroactive Retrieval-Augmented Generation (RetroRAG)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 단방향 추론 패러다임을 탈피하여, 증거를 수정하고 업데이트하여 올바른 방향으로 추론 체인을 안내하는 레트로액티브 추론 패러다임을 구축합니다. RetroRAG는 신뢰할 수 있는 증거를 검색, 생성 및 수정하는 증거 수집 및 발견 프레임워크를 구성합니다.

- **Technical Details**: RetroRAG는 증거 수집(Evidence Collation)과 증거 발견(Evidence Discovery)이라는 두 가지 주요 요소를 통해 효과적인 증거 생성을 목표로 합니다. 증거 수집은 관련 문서를 검색하여 출처 증거로 활용하며, 증거 발견은 질문의 주요 실체와 관련된 여러 유추 증거를 생성하고 필터링하여 필요 없는 정보를 제거하는 과정을 포함합니다. 이를 통해 LLM은 보다 정확하고 신뢰성 높은 답변을 생성할 수 있습니다.

- **Performance Highlights**: 실험적 평가 결과, RetroRAG는 다중 단계 질문 답변(QA) 데이터셋에서 기존 방법보다 현저히 우수한 성과를 보였습니다. 이 프레임워크는 증거 관련 정보의 동적 업데이트와 새로운 증거 발견을 통해 신뢰할 수 있는 답변을 Iteratively(반복적으로) 제공하며, 추론 과정의 설명 가능성도 입증되었습니다.



### Modality-Invariant Bidirectional Temporal Representation Distillation Network for Missing Multimodal Sentiment Analysis (https://arxiv.org/abs/2501.05474)
Comments:
          Accepted for publication by 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이번 연구에서는 결측된 다중 양식(Missing Multimodal) 감정 분석을 위한 Modality-Invariant Bidirectional Temporal Representation Distillation Network (MITR-DNet)를 소개합니다. MITR-DNet은 완전한 양식 teacher 모델이 결측 양식 student 모델을 지도하는 방식의 distillation 기술을 활용하여, 양식 부족이 발생할 경우에도 강인성을 보장합니다. 또한, 다중 양식 데이터의 이질성 문제를 완화하기 위한 Modality-Invariant Bidirectional Temporal Representation Learning Module (MIB-TRL)을 개발하였습니다.

- **Technical Details**: 본 연구에서는 음성(a), 텍스트(t), 비전(v)의 세 가지 양식을 고려합니다. 결측 양식 기능을 시뮬레이션하기 위해 마스킹 함수 F(·)와 랜덤 생성 시간 마스크 gm을 도입하였으며, 이를 통해 불완전한 시퀀스 X~m를 생성합니다. MIB-TRL 모듈은 두 개의 동일한 합성곱 층을 통해 입력 데이터를 처리하고, 이 방향으로 나뉜 데이터 스트림을 통합하여 각 양식의 장기 맥락 표현을 생성합니다.

- **Performance Highlights**: MITR-DNet과 MIB-TRL 모듈은 결측 양식 감정 분석 분야에서의 정확성 및 신뢰성을 크게 향상시켰습니다. 연구 결과, 다중 양식을 활용한 접근 방식은 단일 양식 분석보다 더 세밀하고 정확한 감정 평가를 가능하게 하며, 실제 세계 상황에서의 양식 결측으로 인한 문제를 효과적으로 해결합니다. 또한, 이 혁신적인 구조는 감정 예측에서 텍스트 양식의 중요성을 최대로 활용하여 향상된 성능을 보여줍니다.



### LatteReview: A Multi-Agent Framework for Systematic Review Automation Using Large Language Models (https://arxiv.org/abs/2501.05468)
Comments:
          31 pages, 5 figures, 5 tables

- **What's New**: 본 논문에서는 연구 통찰력을 종합하기 위한 체계적 문헌 리뷰와 메타 분석의 필수성을 강조합니다. 특히, LatteReview라는 새로운 Python 기반 프레임워크를 소개하여, 대규모 언어 모델(LLMs)과 다중 에이전트 시스템을 활용해 체계적 리뷰 과정의 핵심 요소를 자동화합니다. 이 프레임워크는 워크플로우를 간소화하면서도 엄격한 절차를 유지하는 것을 목표로 합니다.

- **Technical Details**: LatteReview는 제목 및 초록 스크리닝, 관련성 점수 매기기, 구조적 데이터 추출 등의 작업을 위해 모듈식 에이전트를 사용합니다. 이를 통해 사용자의 피드백에 기반한 동적 의사결정 및 반복적인 개선이 가능해지며, 클라우드 기반 및 로컬 호스팅 모델과의 호환성을 제공합니다. 또한, 외부 맥락을 통합하기 위한 Retrieval-Augmented Generation (RAG) 기능과 비동기 프로그래밍을 지원합니다.

- **Performance Highlights**: 이 프레임워크는 GitHub 저장소에서 사용 가능하며, 상세한 문서화와 설치 가능한 패키지를 제공합니다. LatteReview의 아키텍처는 다중 모달 리뷰와 구조화된 입력 및 출력을 위한 Pydantic 기반 유효성 검증 기능을 포함하고 있습니다. 이러한 기능들은 대규모 데이터셋 처리를 효율적으로 지원하며, 연구자들에게 새로운 가능성을 제공합니다.



### Small Language Models (SLMs) Can Still Pack a Punch: A survey (https://arxiv.org/abs/2501.05465)
- **What's New**: 최근의 연구에서 중소형 언어 모델(SLMs)이 대형 언어 모델(LLMs)보다 나은 성능을 보여줄 수 있음을 증명하고 있습니다. 본 논문은 1억에서 80억 개의 매개변수를 가진 SLMs에 대한 조사 결과를 제시하고, 성능, 효율성, 확장성, 비용을 균형 있게 고려하여 모델을 구축하는 데 도움이 되는 방법을 탐구합니다. 연구자들은 다양한 SLM의 설계 및 아키텍처를 분석하고, 이들이 LLM에 비견할 만한 성능을 달성하는 혁신적인 기술들을 강조합니다.

- **Technical Details**: SLMs는 다양한 크기와 훈련 기법에 따라 여러 카테고리로 나눌 수 있으며, 이들 모델은 특정 작업에서 높은 성능을 나타내고 일반적인 언어 능력은 제한적입니다. 최근 인기 있는 SLM으로는 LLama2와 그 파생 모델들이 있으며, 이 모델들은 메타의 연구 슈퍼 클러스터에서 교육되었습니다. LLama2는 여러 작업에서 매우 우수한 성능을 보여주며, Llama3는 다중 모드 모델로서 이미지, 비디오, 음성 처리를 통합하여 경쟁력 있는 결과를 보입니다.

- **Performance Highlights**: Llama3.2 시리즈의 모델들은 최신 성능을 나타내며, 모바일 및 엣지 디바이스에서 효율적으로 실행되도록 최적화되어 있습니다. 특히, 1B 및 3B 매개변수를 가진 모델들은 계산 효율성과 작업 실행 능력 간의 균형을 이룰 수 있습니다. TinyLlama 모델은 1.1B의 매개변수로 훈련되어 여러 고급 기술을 통해 성능과 효율성을 극대화하였으며, 이들 모델들은 다양한 환경에서 즉시 개발 및 배포될 수 있도록 지원받고 있습니다.



### LLM-MedQA: Enhancing Medical Question Answering through Case Studies in Large Language Models (https://arxiv.org/abs/2501.05464)
- **What's New**: 이번 연구에서는 기존의 Medical Question Answering (MedQA) 시스템의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. Llama3.1:70B 모델을 활용한 Multi-agent framework를 통해 의료 분야의 질문 응답 문제에 대응하며, 사례 생성 기능을 통합하여 성능을 향상시키고자 합니다. 이 방법론은 의료 지식에 대한 모델의 내재된 능력을 활용하여 추가적인 학습 데이터 없이도 의료 쿼리를 처리할 수 있도록 설계되었습니다.

- **Technical Details**: Multi-agent 시스템은 질문 전문가와 선택지 전문가를 포함한 여섯 개의 주요 구성 요소로 이루어져 있습니다. 문제와 선택지에 대한 분석 후 생성된 상담 사례를 기반으로 리포트를 작성하고, 전문가의 투표 메커니즘을 통해 합의를 이뤄 최종 결정을 내립니다. Llama3.1:70B 모델의 관점에서, 이는 zero-shot learning을 통해 추가 훈련 없이도 복잡한 의학적 쿼리를 처리할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 제안된 다중 에이전트 시스템은 기존 벤치마크 모델에 비해 정확성과 F1 점수가 각각 7% 향상된 것으로 나타났습니다. 이 시스템은 의학적 쿼리에 대한 해석 가능성과 신뢰성을 증가시켜, 복잡한 의료 문제 해결을 위한 견고한 솔루션을 제공합니다. 이러한 성과는 앞으로 LLMs의 의료 분야에서의 광범위한 응용 가능성을 더욱 확장시킬 것으로 기대됩니다.



### Towards Early Prediction of Self-Supervised Speech Model Performanc (https://arxiv.org/abs/2501.05966)
- **What's New**: 이번 연구에서는 Self-Supervised Learning (SSL) 음성 모델의 사전 훈련 품질을 평가하기 위한 효율적인 비지도 학습 방법을 제안합니다. 이러한 새로운 방법은 클러스터 품질(cluster quality)과 임베딩의 순위(rank)를 측정하여, SSL 모델의 성능을 예측하는 데 도움을 줍니다. 실험 결과, 이 방법들이 오직 한 시간의 레이블 없는 오디오로도 사전 훈련의 손실보다 최종 성능과 더 높은 상관관계를 나타낸다는 것을 보였습니다.

- **Technical Details**: BEST-RQ(BERT-based Speech pre-Training with Random-projection Quantizer)은 효율성, 성능, 및 SpeechBrain 라이브러리의 오픈 소스 구현 덕분에 사용됩니다. 모델의 사전 훈련 손실은 예측된 타겟과 불연속 타겟 간의 교차 엔트로피를 통해 계산됩니다. 논문에서는 사전 훈련 중 마스킹 하이퍼파라미터가 손실에 미치는 영향을 이론적으로 설명하고 있습니다.

- **Performance Highlights**: ASR(Automatic Speech Recognition) 작업에서 클러스터링 측정과 SSL 임베딩의 순위가 사전 훈련 손실보다 최종 성능과 더 강한 상관관계를 가진다고 밝혔습니다. 이 연구는 향후 실험에서 수천 시간의 GPU 시간을 절약할 가능성을 보여줍니다. 전략적으로 사전 훈련 과정에서 모델 성능을 조기에 평가할 수 있는 기회를 제공합니다.



### Scalable Vision Language Model Training via High Quality Data Curation (https://arxiv.org/abs/2501.05952)
- **What's New**: 이번 논문에서는 SAIL-VL(ScAlable Vision Language Model)라는 개방형 비전 언어 모델을 소개합니다. 이 모델은 20억 개의 파라미터를 기반으로 하여 뛰어난 성능을 자랑하며, 고품질 데이터 구축을 통해 시각적 이해 능력을 향상시킵니다. SAIL-VL은 데이터의 양과 품질을 모두 고려하여 트레이닝 방식에서 혁신적인 접근을 보여줍니다.

- **Technical Details**: SAIL-VL은 크게 다섯 개의 단계로 구성된 트레이닝 전략을 따릅니다: 사전 학습(pretraining), 사후 학습(fine-tuning) 단계로 구분됩니다. 이 모델은 총 131B 토큰으로 확장된 사전 학습 데이터를 사용하여 시각적 이해 능력을 발휘하고, 이를 통해 다양한 비전 작업에서의 성능을 극대화합니다. 또한 SFT(지도 항목 세분화) 단계에서는 고품질 데이터를 기반으로 모델을 조정하여 시각적 지시 작업을 위한 능력을 개선합니다.

- **Performance Highlights**: SAIL-VL은 OpenCompass 리더보드에서 대규모 비전 언어 모델 중에서 최고 점수를 기록하며, 19개의 표준 벤치마크에서 최고의 평균 점수를 달성했습니다. 특히, 이 모델은 다른 비슷한 크기의 모델들과 비교할 때 비약적인 성능을 발휘하였고, 데이터 크기와 품질이 VLM 성능에 미치는 영향을 해명하기 위한 최초의 논의가 포함되어 있습니다.



### VideoRAG: Retrieval-Augmented Generation over Video Corpus (https://arxiv.org/abs/2501.05874)
- **What's New**: 본 논문은 VideoRAG라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 사용자 쿼리에 따라 관련 비디오를 동적으로 검색하고 이 정보의 시각적 및 텍스트 정보를 출력 생성 과정에 통합합니다. 기존의 RAG 접근법이 주로 텍스트의 검색과 처리에 집중했던 반면, VideoRAG는 비디오를 활용하여 멀티모달 지식을 효과적으로 증대할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VideoRAG는 Retrieval-Augmented Generation (RAG)과 Large Video Language Models (LVLMs)의 개념을 결합하여 동작합니다. 사용자가 입력한 쿼리와 관련된 비디오 콘텐츠를 찾아내고, 이 비디오의 시각적 및 텍스트 요소를 응답 생성 과정에 통합합니다. 특히 텍스트 설명이 없는 경우에도 비디오에서 직접 추출한 내용을 기반으로 자동 음성 인식 기법을 활용하여 텍스트 전사본을 생성하여 완전한 멀티모달 지원을 제공합니다.

- **Performance Highlights**: VideoRAG의 성능은 WikiHowQA와 HowTo100M 데이터셋을 통해 실험적으로 검증되었습니다. 실험 결과, 비디오 데이터를 활용한 VideoRAG가 기존의 관련 RAG 기법들에 비해 상당한 성능 향상을 보임을 확인했습니다. 이 결과는 비디오가 RAG 시스템의 지식 증대에 이바지할 수 있는 강력한 자원임을 입증합니다.



### MARS6: A Small and Robust Hierarchical-Codec Text-to-Speech Mod (https://arxiv.org/abs/2501.05787)
Comments:
          5 pages, 2 figures, 1 table. Accepted at ICASSP 2025

- **What's New**: MARS6는 텍스트-투-스피치(TTS) 모델에서 표현력을 높일 수 있는 혁신적인 인코더-디코더 트랜스포머 구조를 제안합니다. 이 모델은 70M 파라미터를 가지고 있으며, 스피치 토큰을 초당 12회의 속도로 처리하여 긴 텍스트의 효과적인 모델링을 가능하게 합니다. MARS6는 반복 생성 문제를 줄이고 출력 품질과 안정성을 개선하기 위해 여러 최신 훈련 및 추론 기법을 결합했습니다.

- **Technical Details**: MARS6의 인코더는 비인과적 트랜스포머 구조를 사용하고, 스피커 임베딩과 텍스트 임베딩을 통합하여 처리합니다. 이 구조는 SNAC 음향 모델을 기반으로 하여, 계층적 음향 토큰을 사용하여 스피치의 여러 샘플링 레이트에서 코드북을 적용합니다. 또한, MARS6는 새로운 top-p 샘플링 메커니즘과 오즈 비율 선호 최적화를 포함한 여러 기술을 통합하여 모델의 견고성과 효율성을 높였습니다.

- **Performance Highlights**: MARS6는 기존의 확산 및 자기 회귀 기반 TTS 모델과 비교하여 경쟁력 있는 성능을 발휘하며, 주관적인 스피커 유사성 평가에서 이전 모델들을 초월했습니다. 실험 결과, MARS6는 복잡한 참조 오디오에 대해 타겟 스피커의 정체성을 효과적으로 포착할 수 있는 능력을 보여줍니다. 이 모델의 성능은 주관적 및 객관적 평가에서 모두 유효성을 입증하였으며, 특정한 구문 및 감정 표현에서도 뛰어난 결과를 보였습니다.



### Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models (https://arxiv.org/abs/2501.05752)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다단계 추론(multi-step reasoning) 과제에서 탁월한 가능성을 보여주었습니다. 그러나 기존 방법은 계산 효율성과 중복성에서 문제를 드러내고 있으며, 당면 과제의 난이도에 따른 다양성을 간과하고 있습니다. 이러한 한계를 해결하기 위해 우리는 Semantic Exploration with Adaptive Gating (SEAG)이라는 새로운 방법론을 제안합니다.

- **Technical Details**: SEAG는 응답의 신뢰도(confidence level)를 기반으로 트리 탐색(tree search)를 수행할지 여부를 동적으로 결정하는 적응형 게이팅(adaptive gating) 메커니즘을 활용합니다. 이 방법은 의미적 군집화(semantic clustering)를 사용하여 서로 의미적으로 동일한 경로에 대한 중복 탐색을 방지하며, 필요할 때까지 탐색을 유동적으로 조절합니다. 마지막으로 높은 신뢰도를 가진 솔루션이 발견되면 탐색을 조기에 중단하여 불필요한 계산을 줄입니다.

- **Performance Highlights**: 광범위한 실험 결과, SEAG는 기존 트리 탐색 기반 방법에 비해 평균 4.3%의 정확도를 향상시키면서도 계산 비용은 31%로 줄이는 성과를 보였습니다. 이는 GSM8K 및 ARC와 같은 복잡한 추론 기준에서 여러 언어 모델(Llama2, Llama3, Mistral)과 함께 이루어진 테스트에서 입증되었습니다. SEAG의 도입으로 다단계 추론 과제에서의 성능과 효율성이 현저히 개선되었습니다.



### Overcoming Language Priors for Visual Question Answering Based on Knowledge Distillation (https://arxiv.org/abs/2501.05690)
Comments:
          Accepted to ICME2024

- **What's New**: 본 연구는 KDAR이라는 새로운 방법을 제안하여 VQA(Visual Question Answering) 작업에서 발생하는 언어 선행편향 문제를 해결합니다. KDAR은 지식 증류(knowledge distillation)를 활용하여 과적합(overfitting)을 방지하고, 소프트 레이블(soft labels)을 통해 후보 답변의 범위를 좁히는 정규화(regularization) 역할을 수행합니다. 또한, 각 샘플의 중요성을 동적으로 조정하는 샘플-별 가중치 재조정(sample-wise reweighting) 학습 전략을 설계하여 모델이 드문 샘플에 더 집중할 수 있도록 합니다.

- **Technical Details**: 제안된 KDAR 방법은 기계 학습에서의 적합(fitting) 문제로 언어 선행편향 문제를 보고, 소프트 레이블을 정규화 수단으로 활용하여 과적합 문제를 해결하려고 합니다. 또한, 학습 모형 입장에서 각 샘플의 손실 가중치를 동적으로 조정하는 방식으로 드문 샘플을 더 잘 학습할 수 있도록 하여, 동시적으로 공통 답변이 모델에 미치는 영향을 줄이는 구조입니다. 이 두 가지 학습 전략을 통해 VQA 모델의 일반화 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, KDAR은 VQA-CPv2 및 VQAv2 데이터 세트에서 이전의 최첨단 방법보다 훨씬 우수한 성능을 보여주었습니다. 특히, LXMERT와 결합된 경우 VQA-CPv2 데이터셋에서 71.33%라는 최고의 전체 정확도를 기록하였습니다. 또한 KDAR은 OOD(Out-Of-Distribution) 및 IID(Independent Identically Distributed) 환경 모두에서 성능을 향상시키는 데 기여하였습니다.



### Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation (https://arxiv.org/abs/2501.05647)
Comments:
          Published on KDD'25: Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2025

- **What's New**: 대규모 언어 모델(LLMs)과 소형 추천 모델(SRMs)의 통합을 통한 새로운 추천 프레임워크인 LSC4Rec가 제안되었습니다. 이 프레임워크는 기기-클라우드 협력을 기반으로 하여 두 모델의 장점을 결합하여 실시간 사용자 선호도를 보다 효과적으로 반영할 수 있습니다. 또한, 협업 훈련, 협업 추론, 지능형 요청 등 세 가지 전략을 설계하여 실용성을 향상시키고 있습니다.

- **Technical Details**: LSC4Rec는 LLM이 아이템 후보 목록을 생성하고, SRM이 사용자의 실시간 행동을 아우르는 데이터로 펑션을 수행하여 아이템 재순위를 매기는 구조로 되어 있습니다. LLM의 메모리와 SRM의 실시간 데이터 접근 용이성을 통해 훈련 및 추론 빈도를 줄이고, 클라우드와 기기 간 통신 비용을 최소화 합니다. 이러한 고도화된 전략은 두 모델 간의 시너지 효과를 극대화합니다.

- **Performance Highlights**: LSC4Rec의 여러 전략들이 효과적임을 입증하기 위한 포괄적이고 광범위한 실험 분석이 실시되었습니다. 실험 결과는 LSC4Rec의 다양한 LLM 및 SRM에 대한 실효성을 입증하였으며, 추천 시스템의 사용자 경험을 향상시키는 데 큰 기여를 할 것으로 기대됩니다. 이 연구는 실시간 추천 문제에 만연한 문제를 해결할 유망한 방향을 제시하고 있습니다.



### LSEBMCL: A Latent Space Energy-Based Model for Continual Learning (https://arxiv.org/abs/2501.05495)
Comments:
          In the 7th International Conference on Artificial Intelligence in Information and Communication (ICAIIC 2025)

- **What's New**: 본 연구에서 제안하는 LSEBMCL(Latent Space Energy-Based Model for Continual Learning) 방법은 에너지 기반 모델(Energy-based Model, EBM)을 활용하여 지속 학습에서의 치명적인 망각(catatstrophic forgetting)을 방지하는 새로운 접근법을 제시합니다. 이 방법은 새로운 과제를 학습할 때 이전 작업의 데이터 포인트를 샘플링하여 기존의 지식을 보존하는 방식으로 작동합니다. LSEBMCL은 자연어 처리(NLP) 과제에 대해 최신 성능을 달성하며, 기존의 방법들과 차별화되는 특징을 보입니다.

- **Technical Details**: LSEBMCL 모델은 사전 훈련된 Mistral 7B를 기반으로 하며, 네 가지 주요 구성 요소로 구성됩니다: 추론 네트워크, Operator 1, Operator 2, 에너지 함수입니다. 이 네트워크는 주어진 질문에 대해 답변을 제공하며, 다양한 NLP 과제를 처리할 수 있도록 설계되었습니다. 각 구성 요소는 훈련 중 에너지 기능과 분포를 활용하여 데이터를 효율적으로 처리하고 학습할 수 있도록 구성됩니다.

- **Performance Highlights**: 제안된 LSEBMCL 방법은 다양한 NLP 작업에서 우수한 성능을 달성했으며, 현재까지의 실험에서 최첨단 결과를 보여주었습니다. 에너지 기반 모델을 통합하여 이전 데이터 작업에 대한 샘플을 생성하는 방식은 지속 학습에서 기존의 지식을 효과적으로 유지할 수 있도록 합니다. 이러한 접근 방식은 특히 자연어 처리 분야에서의 적용 가능성을 높이며, 향후 다양한 실용적 응용을 위한 기초가 될 수 있습니다.



New uploads on arXiv(cs.IR)

### kANNolo: Sweet and Smooth Approximate k-Nearest Neighbors Search (https://arxiv.org/abs/2501.06121)
Comments:
          7 pages, 3 figures

- **What's New**: kANNolo는 Rust로 작성된 새로운 ANN 라이브러리로, 연구자들에게 중점을 두어 설계되었습니다. 이 라이브러리는 기존의 ANN 라이브러리들이 가진 비효율성 및 유연성 부족 문제를 해결합니다. 특히 kANNolo는 밀집 및 희소 벡터를 지원하며, 다양한 유사도 측정법과 함께 사용할 수 있습니다.

- **Technical Details**: kANNolo는 밀집 및 희소 벡터 모두에서 사용될 수 있는 모듈형 아키텍처를 가지며, 주요 구성 요소로는 One-Dimensional Arrays, Quantizer, Query Evaluator, Dataset이 포함됩니다. 이 아키텍처는 Rust의 trait 기능을 활용하여 유연성과 추상화된 동작 방식을 제공합니다. kANNolo는 HNSW 그래프 인덱싱 방법과 Product Quantization 기술을 구현하여 최적의 검색 성능을 제공합니다.

- **Performance Highlights**: kANNolo는 Sift1M 및 Ms Marco 데이터셋에서 실험을 통해, 기존의 ANN 라이브러리들과 비교하여 성능이 동급 또는 더 뛰어난 결과를 보였습니다. 특히 kANNolo는 밀집 데이터에서 Faiss보다 최대 11.1배, 희소 데이터에서는 2.1배 더 빠른 속도를 기록하며, 연구에 유용한 프로토타이핑과 실험을 지원합니다.



### Recommender Systems for Social Good: The Role of Accountability and Sustainability (https://arxiv.org/abs/2501.05964)
Comments:
          First International Workshop on Recommender Systems for Sustainability and Social Good (RecSoGood'24)

- **What's New**: 이번 연구는 추천 시스템이 지속 가능성, 사회적 책임, 그리고 책임성을 촉진하는 데 어떤 역할을 할 수 있는지 탐구하며, 이를 위해 유엔 지속 가능한 개발 목표(SDGs)와의 정렬을 강조합니다. 추천 시스템이 점점 더 많은 일상적 상호작용에 통합됨에 따라, 단순한 개인화 기능을 넘어 책임 있는 소비를 지원하고 환경적 영향을 줄이며 사회적 선을 증진해야 한다고 주장합니다. 이 연구는 추천 모델의 탄소 발자국을 줄이고 공정성을 보장하며 책임성 메커니즘을 구현하는 전략들을 탐구합니다.

- **Technical Details**: 이 논문에서는 추천 시스템의 발전이 사회적 목표와 어떻게 일치할 수 있는지를 다루며, SDG 10(불평등 감소), SDG 12(책임 있는 소비 및 생산), SDG 13(기후 행동), SDG 16(평화, 정의 및 강한 제도)과 같은 목표들을 살펴봅니다. 공정성, 책임성 및 투명성을 고려한 설계를 통해 추천 시스템은 사회적 불균형을 완화하고, 환경에 대한 영향을 줄이며, 기술의 책임 있는 사용을 지원할 수 있습니다. 최신 연구에 따르면 하나의 딥러닝 추천 시스템 논문이 생성하는 평균 탄소 발자국이 3,000킬로그램을 초과할 수 있으며, 이는 장거리 비행의 배출량과 맞먹는 수치입니다.

- **Performance Highlights**: 추천 시스템의 연구 및 실무는 SDGs를 지원할 수 있는 많은 기회를 제공합니다. 연구자와 실무자들은 공정한 접근을 통해 불평등을 줄이며(SDG 10), 책임 있는 소비와 생산을 유도하고(SDG 12), 기후 행동을 전진시키며(SDG 13), 정의와 강한 제도를 강화할 수 있습니다(SDG 16). 그러나 이러한 결과를 달성하기 위해서는 애플리케이션과 환경에 맞는 맞춤형 전략이 요구됩니다. 추천 시스템 분야의 연구자와 실무자들은 사회적 선을 기준으로 삼고, 그들의 작업이 글로벌 복지에 미치는 영향을 지속적으로 반영해야 합니다.



### Text2Playlist: Generating Personalized Playlists from Text on Deezer (https://arxiv.org/abs/2501.05894)
- **What's New**: 이번 논문에서는 Deezer의 새로운 도구인 Text2Playlist를 소개합니다. Text2Playlist는 사용자의 요구에 맞춘 자동화된 개인화 플레이리스트 생성 도구로, 일반 텍스트 쿼리를 기반으로 작동합니다. 이 시스템은 현재 Deezer의 모바일 및 웹 애플리케이션에 배포되어 있으며, 적용 범위를 확대해 나가고 있습니다.

- **Technical Details**: Text2Playlist는 최신 Large-Language Models (LLMs)와 Retrieval-Augmentation Generation (RAG) 프레임워크를 활용하여 설계되었습니다. 사용자의 쿼리에서 명시적 및 암시적 태그를 추출하고, 이를 기반으로 Deezer의 음악 카탈로그에서 관련 콘텐츠를 검색하여 맞춤형 플레이리스트를 생성합니다. 이 시스템은 Python으로 작성되었으며 Kubernetes 클러스터에서 실행됩니다.

- **Performance Highlights**: Text2Playlist는 2024년 7월부터 5%의 프리미엄 사용자에게 배포되었으며, 10월에는 이 비율이 20%로 확대되었습니다. 사용자 만족도를 평가하기 위해 생성된 플레이리스트의 재청취 비율을 측정한 결과, Text2Playlist로 생성된 플레이리스트는 45%의 재청취율을 기록했습니다. 이는 수동으로 생성된 플레이리스트의 27%에 비해 상당히 높은 수치로, 사용자 참여도가 증가했음을 보여줍니다.



### Social web and Wikipedia: an opportunity to rethink the links between sources' credibility, trust and authority (https://arxiv.org/abs/2501.05813)
- **What's New**: 이 논문은 웹에서 정보의 진실성, 저자 및 소스에 대한 신뢰성 등에 관해 제기되는 근본적인 질문들을 다룬다. 특히 소셜 미디어와 온라인 플랫폼의 발전이 정보 원천의 인식(epistemic evaluation) 방식을 어떻게 변화시켰는지를 분석한다. 이를 통해 정보의 신뢰성과 권위(authority) 개념 간의 관계를 통합하여 제시하는 모델을 제안한다.

- **Technical Details**: 저자는 사용자(user), 문서(document), 저자(author) 간의 관계를 인지하는 통합 모델을 제안하며, 이 모델은 정보의 신뢰성(credibility), 신뢰(trust), 권위(authority)라는 세 가지 개념으로 문제를 단순화한다. 신뢰성은 정보의 진실성에 기반을 두고, 신뢰는 신뢰성 있는 정보를 생성할 수 있는 능력을 의미하며, 권위는 독자가 저자의 의견을 수용할 때 발생하는 영향력(power to influence)을 포함한다.

- **Performance Highlights**: 이 모델은 Wikipedia에 대한 실증 연구 결과와 비교되며, 정보의 진실성과 신뢰성을 평가하는 새로운 접근법을 제공한다. 이러한 분석을 통해 사용자는 다양한 과정에서 정보의 진정성과 저자에 대한 신뢰를 평가할 수 있는 기반을 구축하게 된다.



### Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation (https://arxiv.org/abs/2501.05647)
Comments:
          Published on KDD'25: Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2025

- **What's New**: 대규모 언어 모델(LLMs)과 소형 추천 모델(SRMs)의 통합을 통한 새로운 추천 프레임워크인 LSC4Rec가 제안되었습니다. 이 프레임워크는 기기-클라우드 협력을 기반으로 하여 두 모델의 장점을 결합하여 실시간 사용자 선호도를 보다 효과적으로 반영할 수 있습니다. 또한, 협업 훈련, 협업 추론, 지능형 요청 등 세 가지 전략을 설계하여 실용성을 향상시키고 있습니다.

- **Technical Details**: LSC4Rec는 LLM이 아이템 후보 목록을 생성하고, SRM이 사용자의 실시간 행동을 아우르는 데이터로 펑션을 수행하여 아이템 재순위를 매기는 구조로 되어 있습니다. LLM의 메모리와 SRM의 실시간 데이터 접근 용이성을 통해 훈련 및 추론 빈도를 줄이고, 클라우드와 기기 간 통신 비용을 최소화 합니다. 이러한 고도화된 전략은 두 모델 간의 시너지 효과를 극대화합니다.

- **Performance Highlights**: LSC4Rec의 여러 전략들이 효과적임을 입증하기 위한 포괄적이고 광범위한 실험 분석이 실시되었습니다. 실험 결과는 LSC4Rec의 다양한 LLM 및 SRM에 대한 실효성을 입증하였으며, 추천 시스템의 사용자 경험을 향상시키는 데 큰 기여를 할 것으로 기대됩니다. 이 연구는 실시간 추천 문제에 만연한 문제를 해결할 유망한 방향을 제시하고 있습니다.



### Navigating Tomorrow: Reliably Assessing Large Language Models Performance on Future Event Prediction (https://arxiv.org/abs/2501.05925)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 미래 예측 업무에서 어떻게 활용될 수 있는지를 탐구하는 새로운 접근 방식을 제시합니다. 다양한 계산적 방법들이 미래 예측을 위해 제안되었지만, 큰 언어 모델이 실제로 미래 예측 작업에 대한 성능이 어떻게 되는지는 충분히 연구되지 않았습니다. 연구자들은 다양한 시나리오를 통해 LLM의 예측 능력을 평가하였으며, 미래 예측에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: 연구에서는 Affirmative vs. Likelihood questioning, Reasoning, Counterfactual analysis 등 세 가지 주요 시나리오에서 LLM의 능력을 분석합니다. 이와 함께, 데이터셋의 시간적 기반을 명확히 하고, LLM의 학습 기한 이전에 발생한 사건들에 대해 예측 문제를 만듭니다. 각 질문의 시간뿐 아니라 예상되는 사건 발생 시간도 명시하는 데이터셋을 구축하여 모델의 편향성과 예측 정확성을 평가할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 연구 결과, Likelihood 접근 방식이 Affirmative 접근 방식보다 일반적으로 더 높은 성과를 보였으며, 확률 기반 질문이 더 미묘한 이해를 제공한다고 나타났습니다. 추론을 포함할 경우 기억 비율이 향상되지만, 이는 허위 긍정이 증가함을 의미하여 정확도와 재현율 사이의 상충 관계를 강조합니다. Counterfactual 분석에서는 모델이 사소한 변화에 민감하다는 결과를 얻어 성능에 큰 영향을 미친다고 밝혔습니다.



### VideoRAG: Retrieval-Augmented Generation over Video Corpus (https://arxiv.org/abs/2501.05874)
- **What's New**: 본 논문은 VideoRAG라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 사용자 쿼리에 따라 관련 비디오를 동적으로 검색하고 이 정보의 시각적 및 텍스트 정보를 출력 생성 과정에 통합합니다. 기존의 RAG 접근법이 주로 텍스트의 검색과 처리에 집중했던 반면, VideoRAG는 비디오를 활용하여 멀티모달 지식을 효과적으로 증대할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VideoRAG는 Retrieval-Augmented Generation (RAG)과 Large Video Language Models (LVLMs)의 개념을 결합하여 동작합니다. 사용자가 입력한 쿼리와 관련된 비디오 콘텐츠를 찾아내고, 이 비디오의 시각적 및 텍스트 요소를 응답 생성 과정에 통합합니다. 특히 텍스트 설명이 없는 경우에도 비디오에서 직접 추출한 내용을 기반으로 자동 음성 인식 기법을 활용하여 텍스트 전사본을 생성하여 완전한 멀티모달 지원을 제공합니다.

- **Performance Highlights**: VideoRAG의 성능은 WikiHowQA와 HowTo100M 데이터셋을 통해 실험적으로 검증되었습니다. 실험 결과, 비디오 데이터를 활용한 VideoRAG가 기존의 관련 RAG 기법들에 비해 상당한 성능 향상을 보임을 확인했습니다. 이 결과는 비디오가 RAG 시스템의 지식 증대에 이바지할 수 있는 강력한 자원임을 입증합니다.



### Harmonizing Metadata of Language Resources for Enhanced Querying and Accessibility (https://arxiv.org/abs/2501.05606)
Comments:
          2024 5th International Conference on Computers and Artificial Intelligence Technology (CAIT 2024)

- **What's New**: 이 논문은 다양한 언어 리소스(LR)의 메타데이터를 조화롭게 통합하는 방법을 다룹니다. Linked Data와 RDF 기법을 활용하여 여러 출처의 데이터를 DCAT 및 META-SHARE OWL 온톨로지를 바탕으로 통합된 모델로 구현하였습니다. 새롭게 개발된 portal인 Linghub을 통해 텍스트 기반 검색, faceted browsing, 그리고 advanced SPARQL 쿼리를 지원합니다. 실제 사용자 쿼리를 평가하여 Linghub이 실제 사용자 요구를 충족하는지 검토하였고, 이 과정에서 메타데이터의 중요한 문제를 강조하면서 오픈 어휘 및 표준 준수를 권장합니다.

- **Technical Details**: 이 논문에서는 메타데이터 통합을 위해 Linghub라는 링크드 데이터 기반 포털을 개발했습니다. Linghub는 META-SHARE, CLARIN VLO, LRE-Map, Datahub.io와 같은 다양한 레파지토리에서 메타데이터 항목을 색인화하고 집계합니다. RDF 및 링크드 데이터의 최첨단 개념을 적용하여 서로 다른 출처의 메타데이터 설명을 동일한 데이터 스키마로 매핑하며, 세미틱 웹의 표준 어휘 및 언어 리소스의 새로운 온톨로지를 통해 정보를 조화시킵니다.

- **Performance Highlights**: Linghub은 Corpora Mailing List에서 발송된 실제 사용자 요구 요청을 분석함으로써 사용자의 필요에 얼마나 잘 응답할 수 있는지를 평가하였습니다. 결과적으로, Linghub은 일부 제한에도 불구하고 많은 사용자 요청을 성공적으로 처리할 수 있음을 보여주었습니다. 이 평가는 언어 리소스 레파지토리의 요청에 대한 응답 능력을 평가한 첫 번째 총체적인 시도로, 향후 언어 자원 활용의 효율성과 표준화에 기여할 것입니다.



### Spatial Information Integration in Small Language Models for Document Layout Generation and Classification (https://arxiv.org/abs/2501.05497)
Comments:
          8 pages. Symposium on Applied Computing 2025

- **What's New**: 이번 논문에서는 문서 레이아웃 이해(Document Layout Understanding) 분야에서 새로운 방법론을 제안합니다. 반 구조화된 데이터(semi-structured data) 부족 문제를 해결하기 위해 합성 레이아웃 정보(synthetic layout information)를 생성하는 방식을 도입했습니다. 이 방식은 기존의 LayoutTransformer와 비교했을 때, 더 나은 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 문서의 공간적 배치(spatial arrangement) 분석을 중심으로 진행되어 여타 모델들보다 우수한 결과를 도출하였습니다. 연구에서 제안된 방법은 머신러닝 모델 교육을 위한 공개 데이터셋의 부족을 극복할 수 있는 가능성을 내포하고 있습니다. 특히, 경계 상자 정보(bounding box information)가 텍스트 분류(text classification)에 긍정적인 영향을 미칠 수 있다는 점도 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 LayoutTransformer보다 우수한 성과를 기록하였으며, 이는 반 구조화된 문서의 인식 성능을 향상시킬 수 있음을 시사합니다. 또한, 문서의 다양한 레이아웃 구조를 이해함으로써 실생활에서 접하는 회계 보고서, 구매 주문서 및 영수증 등의 문서 처리가 개선될 수 있다는 점도 강조되고 있습니다.



### S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis (https://arxiv.org/abs/2501.05485)
- **What's New**: 이 논문에서는 문서 청크(chunking) 작업의 중요성을 강조하며, 기존의 방법들이 문서 내의 공간적 레이아웃을 무시하고 있다는 문제를 해결하기 위해 새로운 하이브리드 접근 방식을 제안합니다. 이 방법은 레이아웃 구조, 의미 분석, 공간적 관계를 통합하여 문서 청크의 응집력과 정확성을 향상시킵니다. 또한, 이 접근법은 복잡한 레이아웃을 가진 문서에서도 뛰어난 성능을 보이며, 토큰 길이 제한을 준수할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계인 영역 감지(region detection)와 영역 레이아웃 정렬(region layout ordering)로 구성됩니다. 첫 번째 단계에서는 문서 내 각 분류된 영역에 대한 바운딩 박스(bbox) 데이터를 추출하고, 두 번째 단계에서는 이러한 영역들을 구조적 유형에 따라 합리적인 순서로 배열합니다. 이후 그래프를 구성하고, 가중치를 계산하며, 클러스터링을 통해 일관된 청크로 나누는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과 제안된 하이브리드 접근법은 PubMed와 arXiv에서 다양한 레이아웃과 내용을 가진 연구 논문 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다. 이 방법은 문서 내의 의미적 관계와 공간적 관계를 모두 고려하여 청크를 생성함으로써, 복잡한 문서에서도 적절한 분할을 수행할 수 있습니다. 또한, 평균적으로 높은 품질의 청크를 생성하여 정보 검색, 요약 및 질문 답변과 같은 NLP 작업에서 더 나은 성능을 보입니다.



### Retrieval-Augmented Generation by Evidence Retroactivity in LLMs (https://arxiv.org/abs/2501.05475)
- **What's New**: 본 논문은 Retroactive Retrieval-Augmented Generation (RetroRAG)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 단방향 추론 패러다임을 탈피하여, 증거를 수정하고 업데이트하여 올바른 방향으로 추론 체인을 안내하는 레트로액티브 추론 패러다임을 구축합니다. RetroRAG는 신뢰할 수 있는 증거를 검색, 생성 및 수정하는 증거 수집 및 발견 프레임워크를 구성합니다.

- **Technical Details**: RetroRAG는 증거 수집(Evidence Collation)과 증거 발견(Evidence Discovery)이라는 두 가지 주요 요소를 통해 효과적인 증거 생성을 목표로 합니다. 증거 수집은 관련 문서를 검색하여 출처 증거로 활용하며, 증거 발견은 질문의 주요 실체와 관련된 여러 유추 증거를 생성하고 필터링하여 필요 없는 정보를 제거하는 과정을 포함합니다. 이를 통해 LLM은 보다 정확하고 신뢰성 높은 답변을 생성할 수 있습니다.

- **Performance Highlights**: 실험적 평가 결과, RetroRAG는 다중 단계 질문 답변(QA) 데이터셋에서 기존 방법보다 현저히 우수한 성과를 보였습니다. 이 프레임워크는 증거 관련 정보의 동적 업데이트와 새로운 증거 발견을 통해 신뢰할 수 있는 답변을 Iteratively(반복적으로) 제공하며, 추론 과정의 설명 가능성도 입증되었습니다.



### LLM-MedQA: Enhancing Medical Question Answering through Case Studies in Large Language Models (https://arxiv.org/abs/2501.05464)
- **What's New**: 이번 연구에서는 기존의 Medical Question Answering (MedQA) 시스템의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. Llama3.1:70B 모델을 활용한 Multi-agent framework를 통해 의료 분야의 질문 응답 문제에 대응하며, 사례 생성 기능을 통합하여 성능을 향상시키고자 합니다. 이 방법론은 의료 지식에 대한 모델의 내재된 능력을 활용하여 추가적인 학습 데이터 없이도 의료 쿼리를 처리할 수 있도록 설계되었습니다.

- **Technical Details**: Multi-agent 시스템은 질문 전문가와 선택지 전문가를 포함한 여섯 개의 주요 구성 요소로 이루어져 있습니다. 문제와 선택지에 대한 분석 후 생성된 상담 사례를 기반으로 리포트를 작성하고, 전문가의 투표 메커니즘을 통해 합의를 이뤄 최종 결정을 내립니다. Llama3.1:70B 모델의 관점에서, 이는 zero-shot learning을 통해 추가 훈련 없이도 복잡한 의학적 쿼리를 처리할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 제안된 다중 에이전트 시스템은 기존 벤치마크 모델에 비해 정확성과 F1 점수가 각각 7% 향상된 것으로 나타났습니다. 이 시스템은 의학적 쿼리에 대한 해석 가능성과 신뢰성을 증가시켜, 복잡한 의료 문제 해결을 위한 견고한 솔루션을 제공합니다. 이러한 성과는 앞으로 LLMs의 의료 분야에서의 광범위한 응용 가능성을 더욱 확장시킬 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### Multi-subject Open-set Personalization in Video Generation (https://arxiv.org/abs/2501.06187)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Video Alchemist라는 새로운 비디오 생성 모델을 소개합니다. 이 모델은 다중 주체(multi-subject) 및 오픈 세트(open-set) 개인화 기능을 지원하며, 배경과 전경 객체 모두에 적용됩니다. 기존 방법들이 시간이 많이 소모되는 테스트 시간 최적화(test-time optimization)를 필요로 하는 것과 달리, 우리 모델은 이러한 과정을 없앱니다.

- **Technical Details**: Video Alchemist는 Diffusion Transformer 모듈을 기반으로 하며, 이는 조건부(reference) 이미지와 해당 주체(subject) 수준의 텍스트 프롬프트(text prompt)를 결합하는 크로스 어텐션(cross-attention) 레이어를 사용합니다. 모델 학습 데이터셋 구성에서는 기존 비디오에서 선택된 프레임을 레퍼런스 이미지로 샘플링하고, personalization-specific data augmentation을 통해 모델의 오버피팅(overfitting)을 줄입니다.

- **Performance Highlights**: 실험 결과, Video Alchemist는 정량적(quantitative) 및 정성적(qualitative) 평가에서 기존 개인화 방법들보다 우수한 성능을 보였습니다. 또한, 새로운 개인화 벤치마크인 MSRVTT-Personalization을 도입하여 주체 신뢰(subject fidelity)를 정확하게 측정할 수 있는 기반을 마련했습니다.



### LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs (https://arxiv.org/abs/2501.06186)
Comments:
          15 pages, 5 Figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 위한 단계별 시각적 추론의 평가를 위한 포괄적인 프레임워크를 제안합니다. 기존의 접근 방식이 시각적 추론의 단계별 문제 해결을 강조하지 않고 있다는 점을 지적하며, VRC-Bench라는 새로운 벤치마크를 도입했습니다. 이 벤치마크는 8가지 다양한 카테고리와 4,173개의 수동 검증 추론 단계를 포함하여 다단계 시각적 추론 작업을 평가합니다.

- **Technical Details**: 논문에서 제안하는 새로운 메트릭은 개별 단계에서 시각적 추론의 품질을 정확성과 논리적 일관성 모두에 초점을 맞추어 평가합니다. 또한, Beam Search와 다단계 커리큘럼 학습을 결합하여 LlamaV-o1이라는 새로운 모델을 훈련함으로써, 단계별로 문제 해결 능력을 길러 나가는 구조화된 교육 패러다임을 제시합니다. 이 접근 방식은 모델의 성능 및 해석 가능성을 개선하는 데 기여합니다.

- **Performance Highlights**: LlamaV-o1 모델은 여러 평가 메트릭에서 기존의 오픈 소스 모델보다 우수한 성과를 보였으며, Llava-CoT 모델에 비해 평균 점수가 67.3에 이르며 3.8%의 절대적 증가를 기록했습니다. 더불어 추론 속도는 5배 향상되었습니다. 이러한 성능을 통해 LlamaV-o1은 복잡한 다단계 시각적 추론 작업에서 효과적인 문제 해결을 가능하게 합니다.



### PEACE: Empowering Geologic Map Holistic Understanding with MLLMs (https://arxiv.org/abs/2501.06184)
- **What's New**: 이 논문에서는 지질학적 지도(geologic map)를 이해하기 위한 Multimodal Large Language Models (MLLMs)에 대한 첫 번째 벤치마크인 GeoMap-Bench를 제안하고, 이를 통해 MLLMs의 성능을 평가합니다. 또한, GeoMap-Agent라는 첫 번째 AI 에이전트를 소개하여, 지질학적 지도를 이해하기 위한 계층적 정보 추출, 도메인 지식 주입, 향상된 질문 응답 모듈을 갖추고 있습니다. 이를 통해 지질학적 지도 이해의 효율성과 정확성을 현저하게 향상시키고자 합니다.

- **Technical Details**: GeoMap-Agent는 Hierarchical Information Extraction (HIE), Domain Knowledge Injection (DKI), Prompt-enhanced Question Answering (PEQA)이라는 세 개의 주요 모듈로 구성되어 있으며, 각 모듈은 지질학적 지도에서 정보를 추출하고, 도메인 지식을 주입하며, 효과적인 질문 응답을 실행합니다. 지질학적 지도는 매우 높은 해상도(high resolution)와 복수의 연관된 구성 요소(multiple associated components)를 가지고 있으며, 도메인 특화 지식(domain-specific knowledge)도 필요합니다. 이러한 복잡한 특성으로 인해 기존 MLLMs의 이해에 한계가 있어 GeoMap-Bench가 개발되었습니다.

- **Performance Highlights**: GeoMap-Agent는 GeoMap-Bench에서 전체 점수 0.811을 달성하여 기존 GPT-4o의 0.369 점수를 크게 초과하였습니다. 실험을 통해 GeoMap-Agent가 지질학적 지도에서의 성능을 유의미하게 향상시킬 수 있음을 보여주었습니다. 이 연구는 AI가 지질학 분야에서 어떻게 활발히 활용될 수 있는지를 학계와 산업계에 제시하는 좋은 사례가 될 것입니다.



### VideoAuteur: Towards Long Narrative Video Generation (https://arxiv.org/abs/2501.06173)
Comments:
          Preprint, this https URL

- **What's New**: 이 논문은 조리법을 주제로 한 대규모 비디오 데이터셋을 제안하여 긴 형식의 내러티브 비디오 생성을 발전시키고자 합니다. 제안된 데이터셋은 시각적인 충실도와 텍스트 자막의 정확성을 검증하기 위해 최첨단 Vision-Language Models (VLMs)와 비디오 생성 모델들을 활용하여 품질을 평가했습니다. 또한 비디오에서의 시각적 및 의미적 일관성을 높이기 위해 Long Narrative Video Director라는 방법을 소개하였습니다.

- **Technical Details**: 제안된 데이터셋은 약 200,000개의 비디오 클립으로 구성되어 있으며 각 클립의 평균 길이는 9.5초입니다. 이러한 비디오 데이터는 기존의 YouCook2 및 HowTo100M 데이터셋에서 수집된 것으로, 분석과 평가가 용이하도록 조리법을 중심으로 하여 명확한 내러티브 흐름을 구조화했습니다. 긴 내러티브 비디오 생성을 위한 일반적인 오토 회귀 파이프라인이 설명되며, 이 파이프라인은 비주얼 조건화 비디오 생성 모델과 긴 내러티브 감독으로 구성되어 있습니다.

- **Performance Highlights**: 우리의 실험 결과는 제안된 파이프라인이 긴 내러티브 비디오 생성에 효과적임을 입증하였습니다. CookGen 데이터셋과 함께 VideoAuteur라는 데이터 기반 파이프라인이 제안되었으며, 이는 자동화된 긴 비디오 생성을 위한 방법론을 제공합니다. 또한, 이미지와 텍스트 결합 학습을 통한 비주얼 상태 생성 탐색이 이루어져, 비디오 생성 과정에서 텍스트와 이미지 임베딩을 통합하는 방법론이 개발되었습니다.



### MS-Temba : Multi-Scale Temporal Mamba for Efficient Temporal Action Detection (https://arxiv.org/abs/2501.06138)
- **What's New**: 이 논문에서는 Multi-scale Temporal Mamba (MS-Temba)라는 새로운 아키텍처를 제안하여, 실세계에서의 행동 감지 문제를 해결합니다. MS-Temba는 두 가지 주요 구성 요소, Temporal Mamba Blocks와 Temporal Mamba Fuser로 구성되어 있습니다. 이 구조는 긴 동영상에서의 행동 감지에 최적화되어 있으며, 기존의 방법들과 비교해 파라미터 수가 1/8로 줄어든다는 점에서 효율적입니다.

- **Technical Details**: MS-Temba는 Temporal Local Module (TLM) 및 Dilated Temporal State Space Module (DTS) 두 가지 모듈을 통해 여러 임시적 스케일에서의 특징을 학습합니다. TLM은 짧은 거리의 시간적 표현을 캡처하고, DTS는 드일레이션을 통해 장거리 의존성을 학습하여 일반적인 SSM보다 효과적으로 장기적인 시간 관계를 모델링합니다. 또한, Temba Fuser는 다양한 스케일에서 수집된 정보를 종합하여 효율적인 다중 스케일 표현을 학습합니다.

- **Performance Highlights**: 실험 결과, MS-Temba는 세 개의 공개 데이터셋에서 단기 비디오에서는 기존의 최첨단 기법과 동등한 성능을 보였고, 장기 비디오에서는 이를 초과하는 성능을 나타냈습니다. Mamba 아키텍처는 낮은 GPU 메모리 사용량과 적은 파라미터에도 불구하고 높은 정확도를 유지하는 것으로 나타났으며, 이는 MS-Temba가 비다듬어진 동영상에서의 행동 감지에 매우 적합하다는 것을 보여줍니다.



### Enhancing, Refining, and Fusing: Towards Robust Multi-Scale and Dense Ship Detection (https://arxiv.org/abs/2501.06053)
- **What's New**: 이 논문에서는 SAR(Synthetic Aperture Radar) 해양 선박 탐지를 위한 새로운 프레임워크인 Center-Aware SAR Ship Detector (CASS-Det)를 제안합니다. CASS-Det는 선박의 중심을 강조하는 Center Enhancement Module (CEM), 인접한 선박 간 경계를 명확히 하는 Neighbor Attention Module (NAM), 그리고 다중 스케일 특성 융합을 개선하는 Cross-Connected Feature Pyramid Network (CC-FPN)으로 구성됩니다. 이 새로운 접근 방식은 복잡한 배경과 밀집된 선박 배열 속에서도 강력한 탐지 성능을 보입니다.

- **Technical Details**: CASS-Det의 CEM은 회전 컨볼루션(rotational convolution)을 이용하여 선박 중앙을 강조하고, 배경 간섭을 억제하는 방식으로 설계되었습니다. NAM은 다양한 수준의 장거리 관계를 활용하여 밀접하게 배치된 선박 간의 경계를 정제합니다. CC-FPN은 특성 융합의 효율성을 높이기 위해 교차 연결된 구조를 줌으로써 가격적 오버헤드를 최소화합니다.

- **Performance Highlights**: SSDD, HRSID, LS-SSDD-v1.0와 같은 데이터셋에서의 광범위한 실험 결과, CASS-Det은 기존의 최첨단 알고리즘보다 월등한 성능을 보이며, 다중 스케일 및 밀집된 선박 탐지에서 뛰어난 결과를 나타냅니다. 이 연구는 SAR 이미지에서 구름과 같은 복잡한 배경에서도 선박 탐지를 향상시키는 가능성을 보여주었습니다.



### MSCViT: A Small-size ViT architecture with Multi-Scale Self-Attention Mechanism for Tiny Datasets (https://arxiv.org/abs/2501.06040)
- **What's New**: 이 논문은 작은 크기의 데이터셋에서 Convolutional Neural Networks (CNNs)보다 더 나은 성능을 발휘할 수 있는 새로운 비전 트랜스포머 아키텍처를 소개합니다. 논문에서 제안된 MSCViT는 멀티 스케일(self-attention) 메커니즘과 컨볼루션 블록을 결합하여 다양한 스케일의 주의(attention)를 모델링합니다. 특히, 이 모델은 작은 데이터셋에서 효과적인 결과를 보여주기 위한 효율성을 검토하였습니다.

- **Technical Details**: 논문에서 제안하는 MSCViT는 파형 변환 컨볼루션(wavelet convolution)을 활용하여 주파수 분해를 통해 얻은 고주파 성분과 컨볼루션 채널을 선택적으로 결합합니다. 또한, 다중 헤드 주의(multi-head attention) 모듈을 경량화하여 토큰의 수와 계산 비용을 줄이고, 백본에서 위치 인코딩(positional encoding)을 지역적 피처 추출 모듈로 대체합니다.

- **Performance Highlights**: CIFAR-100 데이터셋에서 MSCViT는 14.0M 파라미터와 2.5 GFLOPs의 성능으로 84.68%의 정확도를 달성했습니다. 이를 통해 대규모 데이터셋 없이도 작은 데이터셋에서 우수한 성능을 보여주는 것을 확인했습니다.



### A Holistically Point-guided Text Framework for Weakly-Supervised Camouflaged Object Detection (https://arxiv.org/abs/2501.06038)
- **What's New**: 이 논문은 약한 라벨을 사용한 카모플라주 객체 탐지(Weakly-Supervised Camouflaged Object Detection, WSCOD)를 위한 새로운 포인트-가이드 텍스트 프레임워크를 제안합니다. 이 프레임워크는 '세그먼트, 선택, 훈련(segment, choose, train)'의 세 단계로 구성되어 있으며, 특히 포인트 기반 후보 생성(Point-guided Candidate Generation, PCG) 기법을 도입하여 텍스트 경로의 수정 및 보완을 수행합니다.

- **Technical Details**: PCG는 포인트의 전경을 사용해 객체의 위치를 명확히 하며, 최적의 마스크를 선택하기 위해 CLIP을 활용하는 자격 후보 판별기(Qualified Candidate Discriminator, QCD)를 개발했습니다. 또한, 새로운 포인트-슈퍼바이즈드 데이터셋(P2C-COD)과 텍스트-슈퍼바이즈드 데이터셋(T-COD)을 만들어 모델 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 실험에서 이 방법은 기존의 최첨단 기법들을 크게 초월하는 성능을 보였으며, 몇몇 기존 완전-슈퍼바이즈드 카모플라주 객체 탐지 방법들을 뛰어넘는 결과를 달성하였습니다. 이로써 WSCOD의 성능을 크게 향상시키고 약한 감독 학습과 완전 감독 학습 간의 격차를 줄이는데 기여하고 있습니다.



### Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction (https://arxiv.org/abs/2501.06035)
- **What's New**: 이번 연구에서는 SkeletonDiffusion라는 새로운 잠재 확산 모델(latent diffusion model)을 소개합니다. 이는 인간의 관절 구조를 명시적으로 반영하여 훈련되고 설계된 아키텍처를 가지고 있습니다. 기존의 방법들에서 발견되는 사지의 왜곡이나 떨림 없는 현실적인 모션을 생성할 수 있도록 하였습니다. SkeletonDiffusion는 다양한 실제 데이터 세트에서 새로운 기준을 수립하며 여러 평가 지표에서 다양한 벤치마크를 초월하는 성능을 보여줍니다.

- **Technical Details**: SkeletonDiffusion는 인체 골격의 기계적 구조와 관절 관계를 고려한 비등방성 가우시안 확산 모델을 사용합니다. 이 모델은 Typed-Graph Convolutions를 기반으로 하여 설계되었으며, 각 관절의 관계를 반영하여 노이즈 공분산 행렬을 고정합니다. 또한, 이 방법은 i.i.d. 가정 없이 동작할 수 있어서 인간 모션 예측(HMP) 문제에 적합하며, 다양한 입력에 대해 높은 일반화 능력을 보장합니다.

- **Performance Highlights**: SkeletonDiffusion는 AMASS, FreeMan, 3DPW와 같은 대규모의 모션 캡처 데이터 세트에서 검증되었습니다. 기존의 강력한 기법들과 비교할 때, 이 방법은 실제적이고 다양한 예측을 생성하며 사지의 왜곡이나 떨림을 최소화하는 성과를 보여주었습니다. 우리는 기존의 다양성 평가 지표에서 간과된 문제들을 발견하였으며, 새로운 현실성과 다양성 지표의 필요성을 강조합니다.



### Generate, Transduct, Adapt: Iterative Transduction with VLMs (https://arxiv.org/abs/2501.06031)
Comments:
          Code will be released at this https URL

- **What's New**: 본 논문에서는 GTA-CLIP라는 새로운 기법을 제안하여 언어 모델에서 얻은 감독 정보를 이용해 언어와 비전 공간 간의 공동 전이(transduction)를 수행합니다. 기존의 induсtive 제안에 비해 이미지-이미지 유사성을 활용하여 더 나은 분류 정확도를 달성하는 transductive zero-shot learning을 통해 모델의 예측을 정제할 수 있습니다. 이 접근 방식은 세 단계로 구성되어 있으며, 각 단계에서 성능 향상을 도모합니다.

- **Technical Details**: GTA-CLIP의 첫 번째 단계에서는 언어 모델에 대한 쿼리를 통해 속성(attribute) 공간을 점진적으로 탐색합니다. 이어지는 두 번째 단계에서는 속성으로 보강된 transductive inference 절차를 수행하며, 마지막으로 세 번째 단계에서는 데이터셋 내에서 추론된 레이블을 기반으로 언어 및 비전 인코더를 미세 조정(fine-tuning)합니다. 실험을 통해 CLIP 인코더와 함께 수행된 결과, GTA-CLIP는 zero-shot 설정에서 CLIP 및 transductive CLIP에 비해 각각 8.6% 및 3.7%의 평균 성능 향상을 보였습니다.

- **Performance Highlights**: GTA-CLIP는 12개 데이터셋과 3개의 인코더에 걸쳐 평균적으로 8.6%의 성능 향상을 기록하였습니다. 또한 few-shot 설정에서도 유사한 개선이 관찰되었습니다. 각 단계의 중요성을 입증하기 위한 제거(ablation) 연구를 통해 transductive learning이 실행되는 동안 비전 및 언어 공간이 어떻게 진화하는지를 시각화하였습니다.



### Geometric-Based Nail Segmentation for Clinical Measurements (https://arxiv.org/abs/2501.06027)
- **What's New**: 본 논문에서는 발톱에 대한 측정을 수행할 수 있는 강력한 세분화(segmentation) 방법이 제안됩니다. 이 방법은 특정 병리학의 발생률을 객관적으로 정량화하기 위한 임상 시험의 첫 단계로 사용됩니다. 발톱과 피부가 지역적으로 유사해 보이는 점에서, 이를 정확히 구별하는 것이 필수적입니다.

- **Technical Details**: 제안된 세분화 방법은 여러 단계를 포함합니다: 발끝 위치 찾기, 발톱 크기 추정, 슈퍼픽셀(super-pixel) 분류, 그리고 발톱 픽셀 단위 세분화입니다. Hough 변환을 사용하여 발끝을 위치를 찾아내고, 이미지의 기하학적 및 사진적 정보에 따라 슈퍼픽셀을 분류한 뒤, 수조(watershed) 변환을 통해 발톱의 경계를 식별합니다. 이러한 방법은 348개의 의료 이미지 데이터셋을 사용하여 검증되었으며, 정확도 0.993과 F-측정치 0.925를 달성했습니다.

- **Performance Highlights**: 제안된 방법은 발톱의 형태, 피부 색소, 조명 조건 및 의료 상태에 의해 영향을 받는 넓은 영역의 외관과 같은 요인에 대해 상당히 강건성을 보입니다. 특히, 의료 연구의 일환으로 발가락의 발톱 면적 측정을 지원하여 시간이 많이 소요되는 작업에서 의료 전문가를 해방시킬 수 있습니다. 본 연구 결과는 자동화된 알고리즘이 인간보다 더 일관되게 발톱을 인식하고 세분화할 수 있음을 강조합니다.



### BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster respons (https://arxiv.org/abs/2501.06019)
- **What's New**: 이번 연구에서는 BRIGHT 데이터셋을 통해 AI 기반의 전천후 재해 대응을 지원하는 새로운 데이터셋을 소개합니다. BRIGHT는 고해상도(optical) 및 SAR 이미지를 활용한 최초의 오픈 액세스 데이터셋으로, 다양한 이벤트에 걸쳐 있으며 전 세계 12개 지역의 데이터가 포함되어 있습니다. 이 데이터셋은 특히 재난 대응의 필요성이 높은 개발도상국에 집중되어 있습니다.

- **Technical Details**: BRIGHT는 0.3m에서 1m 사이의 공간 해상도를 가진 optical 및 SAR 이미지를 포함하고 있습니다. 적중률과 강건성을 검증하기 위해 BRIGHT로 훈련된 7개의 AI 모델을 실험하였으며, 각 모델의 성능을 비교 분석했습니다. 이 데이터셋은 2025년 IEEE GRSS Data Fusion Contest의 공식 데이터셋으로 사용됩니다.

- **Performance Highlights**: 테스트 결과, BRIGHT 데이터셋은 재난 피해 평가(BDA)에 매우 적합하며, AI 모델의 전반적인 성능 향상을 증명합니다. 특히, 여러 자연 및 인위적 재해에 대한 데이터가 포함되어 있어 현실 세계의 다양한 상황에 대비할 수 있는 기반을 제공합니다. 이 데이터셋은 미래의 연구 개발 및 재해 관리 시스템에 중요한 역할을 할 것으로 기대됩니다.



### Pose-independent 3D Anthropometry from Sparse Data (https://arxiv.org/abs/2501.06014)
- **What's New**: 이 논문에서는 모든 자세에서도 신체 치수를 추정할 수 있는 방법을 제안합니다. 기존의 방법은 A-pose에서만 가능했으나, 이 연구는 희소 (sparse) 랜드마크(landmark) 데이터를 활용하여 자세에 구애받지 않고 신체 치수를 추출합니다. 제안된 방법은 기존의 밀집(dense) 기하학적 방법과 비교하여 유사한 결과를 도출하면서도, 세부적인 자세에 관계없이 측정할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 방법은 주어진 랜드마크 좌표를 기반으로 A-pose에서의 신체 치수를 추정합니다. 랜드마크는 피사체의 자세와 관계없이 생성할 수 있는 특징들을 통해 분석되며, 이 과정에서 기존의 3D 스캔 데이터베이스를 활용합니다. 연구는 랜드마크의 정확한 위치를 가정하고, 신체 치수를 추정하는 데 중점을 두어 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CAESAR 및 DYNA 데이터셋을 사용하여 기존의 치수 추정 방법들과 경쟁할 만한 성능을 보였습니다. 특히, 기존 기술이 요구하는 밀집 기하학적 데이터가 필요 없으며, 부상을 입었거나 몸 움직임이 불편한 피사체에게도 이용 가능하다는 장점을 가지고 있습니다. 연구자들은 이 방법을 오픈 소스 형태로 제공하여, 3D 인체 치수 분석의 연구 및 발전을 지원하고자 합니다.



### CamCtrl3D: Single-Image Scene Exploration with Precise 3D Camera Contro (https://arxiv.org/abs/2501.06006)
Comments:
          To be published in 3DV 2025

- **What's New**: 이 논문에서는 단일 이미지와 주어진 카메라 경로를 기반으로 장면의 플라이트 스루(fly-through) 비디오를 생성하는 새로운 방법을 제안합니다. 이 방법은 이미지-비디오 라텐트 확산 모델(latent diffusion model)을 기반으로 하며, 카메라 궤적에 따라 UNet의 노이즈 제거기를 조정하는 네 가지 기술을 활용합니다. 이러한 접근법은 사진을 고품질의 몰입형 비디오 경험으로 변환하는 데 기여할 것으로 보입니다.

- **Technical Details**: 제안된 CamCtrl3D 모델은 초기 RGB 이미지와 카메라의 연속적인 위치를 입력으로 받아, 시퀀스 뷰를 생성합니다. 논문에서는 2D ⇔ 3D 변환기를 포함한 네 가지 조정 기법을 통합하여 모델의 3D 이해도를 증가시키고, 라이트 전송(light transport) 원리에 따른 프레임 간 상호작용을 가능하게 합니다. 각 조건의 조합을 제어하기 위해, ControlNet 스타일의 아키텍처를 채택하였습니다.

- **Performance Highlights**: 제안된 방법은 기존 기술들에 비해 적은 수의 훈련 데이터(10K posed videos)로도 최첨단 결과(state-of-the-art results)를 보여줍니다. 또한 제안된 메트릭을 통해 생성된 비디오 품질과 보기 변화 시 입력 이미지 세부정보의 보존 능력을 정량적으로 평가합니다. 실험 결과, 카메라 외부 파라미터와 2D ⇔ 3D 변환기, 초기 이미지 재투영이 최적의 조건 조합으로 확인되었습니다.



### SeMi: When Imbalanced Semi-Supervised Learning Meets Mining Hard Examples (https://arxiv.org/abs/2501.06004)
Comments:
          11 pages,6 figures, conference

- **What's New**: 본 논문에서는 클래스 불균형 반지도 학습(Class Imbalanced Semi-Supervised Learning, CISSL)의 성능 향상을 위해 Hard Examples를 발굴하고 활용하는 새로운 방법인 SeMi를 제안합니다. 이 방법은 하드 예제와 쉬운 예제의 로짓 간 엔트로피 차이를 구분하여 하드 예제를 식별하고 레이블이 없는 데이터의 유용성을 증대시킵니다. 기존 기법들은 데이터 균형 조정에 주력했으나, 하드 예제의 중요성을 간과하였습니다.

- **Technical Details**: SeMi는 두 가지 주요 알고리즘으로 구성됩니다. 첫째, Online Hard Example Mining and Learning (OHEML) 알고리즘을 통해 도전적인 샘플을 효율적으로 활용합니다. 둘째, Pseudo-Label Certainty Enhancement (PLCE) 알고리즘을 통해 세멘틱 피처 센터를 통합함으로써, 의사 레이블의 견고성과 정확성을 높입니다.

- **Performance Highlights**: SeMi는 다양한 CISSL 벤치마크에서 기존 최첨단 방법들보다 우수한 성능을 보여주며, 특히 성능이 낮은 경우에 54.8%의 개선 효과를 나타냈습니다. 이 방법은 간단하면서도 효과적으로 하드 예제를 활용하여 모델의 일반화 능력을 향상시키고 있습니다.



### Self-Supervised Partial Cycle-Consistency for Multi-View Matching (https://arxiv.org/abs/2501.06000)
Comments:
          Accepted to VISAPP 2025

- **What's New**: 이 논문에서는 부분적으로 겹치는 카메라 뷰 간의 객체 매칭을 위한 새로운 사이클 일관성(cycle-consistency) 수학적 공식화를 제안합니다. 사이클 일관성은 레이블이 없는 데이터에서도 효과적으로 네트워크를 학습할 수 있게 해주며, 이를 통해 사람과 객체의 경계 상자만으로 직접 학습할 수 있는 장점을 제공합니다. 또한, 부분 겹침을 처리하기 위한 의사 마스크(pseudo-mask)를 도입하여 훈련 손실을 조정하고, 데이터 입력을 개선하기 위한 새로운 샘플링 기법을 채택했습니다.

- **Technical Details**: 부분 겹침을 처리하기 위해 제안된 새로운 수학적 공식화는 사이클 일관성의 이론을 확장합니다. 의사 마스크를 사용하여 부분적으로 일관성을 유지하도록 손실을 조정하고, 여러 사이클 변형을 통해 더 풍부한 학습 신호를 생성합니다. 이 방법은 카메라 간의 겹침이 감소한 경우에도 효과적이며, 특히 많은 사람들 간의 매칭이 필요한 어려운 장면에서 안정적인 성능 개선을 보여줍니다.

- **Performance Highlights**: DIVOTrack 데이터셋에서 진행된 실험 결과, 제안한 방법은 최신 자가 지도 학습(self-supervised learning) 기술보다 4.3% 높은 F1 점수를 기록했습니다. 실험을 통해 다양한 사이클 변형을 사용할 때의 장점이 드러났으며, 특히 더 어려운 시나리오에서 성능이 뛰어난 것을 확인했습니다. 추가된 부분 사이클 일관성 정보는 매칭 성능의 유의미한 개선으로 이어졌습니다.



### Minimizing Occlusion Effect on Multi-View Camera Perception in BEV with Multi-Sensor Fusion (https://arxiv.org/abs/2501.05997)
Comments:
          Accepted form publishing at the Electronic Imaging - Autonomous Vehicles and Machines Conference

- **What's New**: 이번 연구는 자율주행 기술의 핵심인 카메라 기반 인식 시스템의 시각적 성능을 저해하는 오클루전(occlusion)의 영향을 분석하였습니다. 특히, 다양한 환경적 요인으로 인한 카메라 오클루전을 다루며, 멀티센서 융합 기법을 사용하여 LiDAR와 레이더 데이터를 통합하여 성능 저하를 완화하는 방법을 제시합니다. 이 연구는 기존의 논문에서 부족했던 카메라 오클루전의 구체적인 영향을 BEV(Bird's Eye View) 기반 인식 시스템에 적용하여 다루고 있습니다.

- **Technical Details**: 본 연구는 nuScenes 데이터셋을 사용하여 카메라 이미지에서 오클루전을 인위적으로 생성하는 방법을 설명합니다. WoodScape Soiled 데이터셋의 오염 패턴을 사용하여 비와 안개, 습기 등의 실제 환경 조건을 시뮬레이션하였으며, 이로 인해 생성된 이진 마스크로 오클루전 지역을 구분합니다. Gaussian 필터를 적용해 오클루전 영역만 흐림 효과를 주어, 자율주행 시스템이 겪을 수 있는 비정상적인 시각적 도전을 시뮬레이션합니다.

- **Performance Highlights**: 연구 결과, 멀티센서 융합 기술을 통해 카메라 오클루전의 영향을 완화한 결과, 차량 세분화(vehicle segmentation) 작업의 정확성과 견고성(robustness)이 유의미하게 개선되었음을 확인했습니다. 이는 향후 자율주행 시스템의 신뢰성을 높이는 데 기여할 수 있는 중요한 결과로, BEV 표현과 관련된 환경적 도전이 성능에 미치는 영향을 기존 이해를 보완하는 데 기여합니다.



### Swin-X2S: Reconstructing 3D Shape from 2D Biplanar X-ray with Swin Transformers (https://arxiv.org/abs/2501.05961)
- **What's New**: 이 논문에서 발표된 Swin-X2S는 2D X-ray 이미지에서 직접 3D 세분화(segmentation) 및 라벨링(labeling)을 재구성하는 전방향 딥러닝(end-to-end deep learning) 방법입니다. 기존의 재구성 방법들은 주로 수작업으로 특징(feature)을 추출하고, 수동 개입(manual intervention) 및 선행 지식을 필요로 하여 불안정한 형태 오류(shape errors)와 추가적인 처리 비용을 초래했습니다. Swin-X2S는 2D Swin Transformer를 활용한 인코더와 3D 컨볼루션(3D convolution) 기반의 디코더로 구성되어 있으며, 이를 통해 더 나은 정확도와 효율성을 제공합니다.

- **Technical Details**: Swin-X2S의 아키텍처는 3단계로 나뉘어 있습니다. 첫 번째 단계에서는 2D Swin-Transformer 기반의 인코더가 원시 X-ray 이미지 쌍을 입력으로 받아 이미지 특징(feature)을 추출합니다. 두 번째 단계에서는 차원 확장(dimension-expanding) 모듈이 저수준 토큰(low-level tokens)을 2D 인코더에서 3D 디코더로 전송합니다. 마지막으로, 3D 컨볼루션 기반의 디코더는 크로스 어텐션(cross-attention) 메커니즘을 활용하여 비면(biplanar) 뷰의 정보를 통합함으로써 다양한 해부학적 형태를 재구성할 수 있습니다.

- **Performance Highlights**: Swin-X2S는 네 가지 해부학적 영역(대퇴근, 고관절, 척추, 갈비뼈)의 54개 범주를 포함하는 9개의 공개 데이터셋에서 실시한 실험을 통해 높은 성능을 입증하였습니다. 이전의 방법들과 비교하여 세분화 및 라벨링 메트릭뿐만 아니라, 임상적으로 관련된 매개변수에서도 유의미한 개선이 관찰되었습니다. 이 연구는 결국 임상적 상황에서 해부학적 형태 재구성을 위한 효과적인 옵션을 제공할 가능성을 보여줍니다.



### Scalable Vision Language Model Training via High Quality Data Curation (https://arxiv.org/abs/2501.05952)
- **What's New**: 이번 논문에서는 SAIL-VL(ScAlable Vision Language Model)라는 개방형 비전 언어 모델을 소개합니다. 이 모델은 20억 개의 파라미터를 기반으로 하여 뛰어난 성능을 자랑하며, 고품질 데이터 구축을 통해 시각적 이해 능력을 향상시킵니다. SAIL-VL은 데이터의 양과 품질을 모두 고려하여 트레이닝 방식에서 혁신적인 접근을 보여줍니다.

- **Technical Details**: SAIL-VL은 크게 다섯 개의 단계로 구성된 트레이닝 전략을 따릅니다: 사전 학습(pretraining), 사후 학습(fine-tuning) 단계로 구분됩니다. 이 모델은 총 131B 토큰으로 확장된 사전 학습 데이터를 사용하여 시각적 이해 능력을 발휘하고, 이를 통해 다양한 비전 작업에서의 성능을 극대화합니다. 또한 SFT(지도 항목 세분화) 단계에서는 고품질 데이터를 기반으로 모델을 조정하여 시각적 지시 작업을 위한 능력을 개선합니다.

- **Performance Highlights**: SAIL-VL은 OpenCompass 리더보드에서 대규모 비전 언어 모델 중에서 최고 점수를 기록하며, 19개의 표준 벤치마크에서 최고의 평균 점수를 달성했습니다. 특히, 이 모델은 다른 비슷한 크기의 모델들과 비교할 때 비약적인 성능을 발휘하였고, 데이터 크기와 품질이 VLM 성능에 미치는 영향을 해명하기 위한 최초의 논의가 포함되어 있습니다.



### A Multimodal Dataset for Enhancing Industrial Task Monitoring and Engagement Prediction (https://arxiv.org/abs/2501.05936)
Comments:
          Accepted at the 20th International Conference on Human-Robot Interaction (HRI) 2025

- **What's New**: 본 논문에서는 동적 산업 작업 흐름에서 작업자 행동 및 참여 수준을 감지하고 해석하기 위한 새로운 다중 모드(Multimodal) 데이터셋인 MIAM(Multimodal Industrial Activity Monitoring) dataset을 소개합니다. 이 데이터셋은 현실적인 조립 및 분해 작업을 포착하여, 작업 로컬라이제이션(action localization), 객체 상호작용(object interaction), 참여 예측(engagement prediction)과 같은 주요 메타 작업을 평가할 수 있도록 설계되었습니다. 이 연구는 기존의 단일 모드(unimodal) 방법론이 공장 환경의 복잡성과 비예측성을 잘 처리하지 못한다는 점에 기인하여, 다중 모드 접근 방식의 필요성을 강조하고 있습니다.

- **Technical Details**: MIAM 데이터셋은 22세션에서 수집된 멀티 뷰 RGB, 깊이(depth), 9축 Inertial Measurement Unit (IMU) 데이터를 포함하여, 각 세션은 290분 분량의 트리밍되지 않은 비디오로 구성되어 있습니다. 작업자는 조립 및 분해 작업을 수행하며, 이 과정에서 비디오 및 IMU 신호가 실시간으로 기록됩니다. 데이터는 RGB-D 카메라 및 IMU 센서를 사용하여 수집하며, 이들 각각의 데이터는 작업 효율성과 참여 모니터링을 위한 비교적 정밀한 작업 흐름을 반영합니다.

- **Performance Highlights**: 제안된 다중 모드 네트워크는 RGB 프레임, IMU 데이터, 그리고 스켈레톤 시퀀스를 융합하여 산업 작업 중 참여 수준을 예측합니다. 초기 평가에 따르면, 제안된 시스템은 기존 방법들보다 높은 정확도로 참여 상태를 인식하는데 기여하며, 동적인 산업 환경에서 작업자 성능 모니터링에 강력한 솔루션을 제공합니다. 이를 통해 참여 예측의 정확성을 향상시키고, 실세계 산업 환경에서 작업의 효율성을 더욱 높일 수 있습니다.



### Weakly Supervised Segmentation of Hyper-Reflective Foci with Compact Convolutional Transformers and SAM2 (https://arxiv.org/abs/2501.05933)
Comments:
          7 pages, 1 figure, accepted at German Conference on Medical Image Computing 2025

- **What's New**: 이 논문에서는 광학 단층 촬영(OCT)에서의 약하게 감독된 세그멘테이션(weakly supervised segmentation)을 위한 새로운 프레임워크를 제안합니다. 이 방식은 전통적인 Attention 기반의 다중 인스턴스 학습(MIL) 접근 방식을 개선하여 기존의 해상도를 유지하면서 HRFs의 세분화 정확도를 높이는 데 초점을 맞추고 있습니다. 또한 Compact Convolutional Transformer(CCT)를 도입하여 다양한 지역 간 정보 교환을 가능하게 하여 성능을 더욱 향상시킵니다.

- **Technical Details**: 본 연구는 191개의 OCT 이미지를 사용하여 HRF 세분화 작업을 수행하였으며, 962개의 B-scan이 HRF를 포함하고 있습니다. 약하게 감독된 세그멘테이션은 이미지 레벨의 주석을 기반으로 픽셀 레벨의 로컬라이제이션을 달성하는 방식으로 진행됩니다. 모델 학습을 위해 MIL과 CCT의 두 가지 구조가 사용되었고, LRP(Layer-wise Relevance Propagation)를 통해 효과적인 세그멘테이션을 위한 관련성 지도를 생성하였습니다.

- **Performance Highlights**: 연구 결과, 새롭게 제안된 프레임워크는 기존의 MIL 접근 방식과 비교하여 HRF 세분화 정확도를 상당히 향상시켰습니다. CCT 구조는 특히 작은 데이터셋에서 뛰어난 성능을 발휘하며, 위치 정보를 효과적으로 활용하여 HRF의 로컬라이제이션 문제를 해결할 수 있음을 보여주었습니다. 전반적으로 이 연구는 약하게 감독된 세그멘테이션의 가능성을 확장하며, OCT 이미지 분석 분야에서 실질적인 기여를 할 것으로 기대됩니다.



### Binary Event-Driven Spiking Transformer (https://arxiv.org/abs/2501.05904)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문에서는 Transformer 기반의 Spiking Neural Networks (SNNs)를 위한 새로운 이벤트 기반 자기 주의 메커니즘을 제안합니다. Binarization 기술을 통합하여 Binary Event-Driven Spiking Transformer(BESTformer)를 만들었으며, 이는 저장공간과 연산 비용을 크게 줄일 수 있습니다. 그러나 binarization로 인해 성능 저하가 발생하는 문제를 해결하기 위해 Coupled Information Enhancement(CIE) 방법을 제안합니다. 이를 통해 BESTformer의 성능을 향상시키면서도 효율성을 유지할 수 있습니다.

- **Technical Details**: BESTformer는 가중치와 주의 맵을 1-bit로 표현하여 모델 크기와 계산 비용을 최소화하도록 설계되었습니다. 기존의 Transformer 기반 SNN들은 많은 메모리와 계산 자원을 요구하는 반면, BESTformer는 이러한 한계를 극복하고 에너지 효율을 최대화합니다. CIE 방법은 가역적 프레임워크와 정보 강화 증류(information enhancement distillation)를 포함하여 이진 모델과 정밀도 높은 모델 간의 상호 정보(mutual information)를 최대화합니다. 이를 통해 BESTformer의 제한된 정보 표현 능력 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: BESTformer는 정적(static) 및 신경형(neuromorphic) 데이터셋에서 광범위한 실험을 통해 다른 이진 SNN들보다 우수한 성능을 나타냅니다. 특히, ImageNet-1k 데이터셋에서 유사한 규모의 다른 모델들과 비교할 때 7.85%의 성능 향상을 달성했습니다. 이러한 결과는 BESTformer가 자원이 제한된 엣지 디바이스(Edge device)에서 효과적으로 사용할 수 있는 잠재력을 아래에 잘 보여줍니다.



### Valley2: Exploring Multimodal Models with Scalable Vision-Language Design (https://arxiv.org/abs/2501.05901)
- **What's New**: 최신 논문에서는 Valley2라는 새로운 다중 모달 언어 모델을 소개합니다. 이 모델은 이미지, 텍스트 및 비디오 입력을 지원하며, 전자상거래와 짧은 비디오 시나리오에서 성능을 극대화하기 위해 설계되었습니다. Valley2는 79.66이라는 뛰어난 성능으로 전자상거래 벤치마크에서 최신 기술(state-of-the-art) 수준의 성능을 달성했습니다.

- **Technical Details**: Valley2는 Qwen2.5를 언어 모델 스타크(backbone)로, SigLIP-384를 비전 인코더로 채택합니다. 주요 혁신 중 하나는 시각적 데이터에서 더욱 풍부한 의미 정보를 포착하기 위한 학습 가능한 비주얼 어휘(visual vocabulary) 도입입니다. 또한, Eagle Module을 활용하여 극한 해상도의 이미지를 처리하고 성능을 향상시키는 구조를 가지고 있습니다.

- **Performance Highlights**: Valley2는 OpenCompass 리더보드에서 10B 매개변수가 없는 모델 중 두 번째로 높은 평균 점수(67.4)를 기록하고 있습니다. 이전 모델인 Valley에 비해 다양한 향상을 탐색하며, 특히 전자상거래 및 다중 이미지 시나리오에서 향상된 성능을 보여줍니다. 이러한 경쟁력 있는 성과는 실세계 응용 프로그램에서도 매우 유용할 것으로 기대됩니다.



### Beyond Flat Text: Dual Self-inherited Guidance for Visual Text Generation (https://arxiv.org/abs/2501.05892)
- **What's New**: 이 논문에서 제안하는 STGen 프레임워크는 기울거나 구부러진 텍스트 레이아웃과 같은 도전적인 상황에서도 시각 텍스트를 정확하게 생성하고 배경과 조화를 이루도록 돕습니다. 이는 주목할 만한 두 가지 가지를 활용하여 이루어지는데, 첫 번째는 Semantic Rectification Branch(SRB)로, 평면적인 정확한 시각 텍스트 생성을 위한 정보를 추출하여 복잡한 레이아웃에서 조정합니다. 두 번째는 Structure Injection Branch(SIB)로, 글리프 구조에 대한 정보로 시각 텍스트의 구조를 보강합니다.

- **Technical Details**: STGen은 훈련 없이 기존의 시각 텍스트 생성 모델과 쉽게 통합할 수 있는 방법이며, 두 가지 방향의 프레임워크를 통해 텍스트 생성 프로세스를 안내합니다. SRB는 간단한 형태로 생성된 잠재(latent)를 사용하여 왜곡된 텍스트 예측을 수정하고 텍스트와 배경의 조화를 이루도록 돕습니다. SIB는 글리프 이미지의 잠재 정보를 활용하여 텍스트의 구조적 정확성을 높이며, 두 가지 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 상세한 실험에서 우리의 방법이 복잡한 시각 텍스트를 생성할 때 정확성과 품질 모두에서 우수한 성과를 거두었음을 입증했습니다. STGen은 기존 매개 모델의 단점을 극복하며, 기울어진 텍스트를 포함한 다양한 레이아웃에서의 성능이 크게 향상되었습니다. 최종적으로, 본 연구는 복잡한 레이아웃에서 시각 텍스트를 생성하는 새로운 도전 과제를 제시하며, STGen이 최고의 성능을 달성했음을 강조합니다.



### EDNet: Edge-Optimized Small Target Detection in UAV Imagery -- Faster Context Attention, Better Feature Fusion, and Hardware Acceleration (https://arxiv.org/abs/2501.05885)
Comments:
          Accepted in 21st IEEE International Conference on Ubiquitous Intelligence and Computing (UIC 2024) this https URL

- **What's New**: 이번 연구에서는 드론 이미지에서 작은 표적을 탐지하기 위한 새로운 프레임워크인 EDNet을 제안합니다. EDNet은 향상된 YOLOv10 아키텍처를 기반으로 하며, 실시간 응용 프로그램에 최적화되어 있어 후처리 과정이 필요 없습니다. XSmall 탐지 헤드와 Cross Concat 전략을 통합하여 다양한 환경에서 작은 표적을 더욱 효과적으로 탐지할 수 있는 멀티 스케일 컨텍스트 인식 기능이 개선되었습니다.

- **Technical Details**: EDNet은 TensorFlow와 같은 딥러닝 플랫폼에서 구현된 다양한 구성 요소로 이루어져 있습니다. ConvBNSiLU 블록과 Spatial-Channel Decoupled Downsampling(SCDown) 블록을 포함하고 있으며, 이들은 계산 효율성을 높이고 중요한 정보를 보존하는 데 기여합니다. 또한 Faster Context Attention(FCA)과 같은 맞춤형 블록을 사용하여 파라미터 수를 줄이면서도 성능을 향상시키고, WIoU 손실 함수를 통해 바운딩 박스 회귀를 개선합니다.

- **Performance Highlights**: EDNet은 Tiny에서 XL까지 7가지 변형으로 제공되며, 기존의 객체 탐지기보다 높은 정확도와 뛰어난 계산 효율성을 자랑합니다. iPhone 12와 같은 모바일 장치에서 EDNet 변형들은 16FPS에서 55FPS의 속도로 작동할 수 있어, 데이터 프라이버시를 보장하면서도 실시간으로 객체 탐지를 수행할 수 있는 확장 가능하고 효율적인 솔루션을 제공합니다. 특히, EDNet은 mAP@50에서 최대 5.6% 증가를 달성하며, 효율적인 모델 디자인을 통해 다양한 환경에 적합한 성능을 보여줍니다.



### Text-to-Edit: Controllable End-to-End Video Ad Creation via Multimodal LLMs (https://arxiv.org/abs/2501.05884)
Comments:
          16pages conference

- **What's New**: 최근 짧은 비디오 콘텐츠의 급증으로 인해 효율적이며 자동화된 영상 편집 솔루션의 필요성이 커지고 있습니다. 본 논문에서는 최종 비디오 콘텐츠 편집에 대한 정확한 제어를 실현하는 혁신적인 end-to-end 기본 프레임워크를 제안합니다. 우리는 Multimodal Large Language Models (MLLMs)의 유연성과 일반성을 활용하여 효율적인 비디오 제작을 위한 명확한 입력-출력 매핑을 정의했습니다.

- **Technical Details**: 비디오 생성은 비디오 클립 배열, 스크립트 생성, 배경 음악 추천 및 타임라인 정렬과 같은 복잡한 과정과 자료 관리가 필요합니다. 이를 위해 모델의 유연성과 협업을 요구하며, 우리는 비디오 이해 문제를 해결하기 위해 2 fps에 이르는 더 높은 프레임 밀도를 사용하고, slow-fast 처리 전략을 도입하여 시간적 및 공간적 비디오 정보를 효과적으로 추출하고 이해할 수 있도록 했습니다.

- **Performance Highlights**: 종합적인 실험을 통해 우리 방법은 광고 데이터셋에서 특히 효과적인 것으로 나타났으며, 공공 데이터셋에 대해서도 보편적으로 적용 가능한 결론을 도출했습니다. 또한, 텍스트 기반의 비디오 편집 방법을 통해 사용자 기대에 부합하는 최종 결과를 제공하며, 편집 과정의 품질 및 통제력을 크게 향상시켰습니다.



### TakuNet: an Energy-Efficient CNN for Real-Time Inference on Embedded UAV systems in Emergency Response Scenarios (https://arxiv.org/abs/2501.05880)
Comments:
          This paper has been accepted at WACVW 2025, which will take place on 28/02/2025. The official conference proceedings have not yet been published at the time of submission to arXiv. The final version of the paper, incorporating any changes based on feedback received during the conference, will be included in the proceedings once they are made available

- **What's New**: 이번 논문에서는 TakuNet이라는 새로운 경량 아키텍처를 소개합니다. TakuNet은 깊이별 합성곱(depth-wise convolution)와 조기 다운샘플링(early downsampling) 기법을 사용하여 계산 복잡성을 줄이면서도 높은 정확도를 유지합니다. 이 모델은 드론 및 UAV와 같은 임베디드 장치에서 인공지능 처리의 효율성을 향상시킬 수 있도록 설계되었습니다.

- **Technical Details**: TakuNet은 주로 깊이별 합성곱(depth-wise convolution)에 기반하며, 초기 레이어의 계산 복잡성을 줄이는 조기 다운샘플링 스템을 사용합니다. 네트워크 구조는 네 개의 단계로 구성되며, 각 단계는 피쳐 맵의 크기를 줄이는 다운샘플러 블록을 포함하고, 훈련 시 빠른 수렴을 위한 밀집 연결(dense connections)을 채택하고 있습니다. 16비트 부동 소수점(float) 해상도로 훈련되어 임베디드 시스템의 하드웨어 가속기에 최적화됩니다.

- **Performance Highlights**: TakuNet은 두 개의 공공 데이터셋(AIDER, AIDERv2)을 기반으로 실험 평가를 실시하였으며, 매개변수 수가 최소화되었음에도 불구하고 거의 최첨단 수준의 정확도를 달성했습니다. Jetson Orin Nano 및 Raspberry Pi와 같은 임베디드 장치에서 650 fps 이상을 달성하여 재난 응답 시나리오에서 드론에 대한 실시간 AI 처리 가능성을 입증했습니다.



### VideoRAG: Retrieval-Augmented Generation over Video Corpus (https://arxiv.org/abs/2501.05874)
- **What's New**: 본 논문은 VideoRAG라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 사용자 쿼리에 따라 관련 비디오를 동적으로 검색하고 이 정보의 시각적 및 텍스트 정보를 출력 생성 과정에 통합합니다. 기존의 RAG 접근법이 주로 텍스트의 검색과 처리에 집중했던 반면, VideoRAG는 비디오를 활용하여 멀티모달 지식을 효과적으로 증대할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VideoRAG는 Retrieval-Augmented Generation (RAG)과 Large Video Language Models (LVLMs)의 개념을 결합하여 동작합니다. 사용자가 입력한 쿼리와 관련된 비디오 콘텐츠를 찾아내고, 이 비디오의 시각적 및 텍스트 요소를 응답 생성 과정에 통합합니다. 특히 텍스트 설명이 없는 경우에도 비디오에서 직접 추출한 내용을 기반으로 자동 음성 인식 기법을 활용하여 텍스트 전사본을 생성하여 완전한 멀티모달 지원을 제공합니다.

- **Performance Highlights**: VideoRAG의 성능은 WikiHowQA와 HowTo100M 데이터셋을 통해 실험적으로 검증되었습니다. 실험 결과, 비디오 데이터를 활용한 VideoRAG가 기존의 관련 RAG 기법들에 비해 상당한 성능 향상을 보임을 확인했습니다. 이 결과는 비디오가 RAG 시스템의 지식 증대에 이바지할 수 있는 강력한 자원임을 입증합니다.



### Language-Inspired Relation Transfer for Few-shot Class-Incremental Learning (https://arxiv.org/abs/2501.05862)
Comments:
          Accepted by IEEE TPAMI

- **What's New**: 이번 논문에서는 몇 개의 샘플만 관찰하여 언어 설명으로 새로운 클래스를 묘사하는 방법인 Few-Shot Class-Incremental Learning (FSCIL)을 제안합니다. 기존 방법들은 주로 시각적 인코더의 신중한 조정에 의존하며, 기본 지식과 점진적 지식 간의 균형을 유지하는 데 어려움이 있었습니다. 이 논문에서는 Language-inspired Relation Transfer (LRT) 패러다임을 도입하여 시각적 단서와 텍스트 묘사를 결합하여 객체를 인식하는 방식을 제시합니다.

- **Technical Details**: LRT 패러다임은 두 가지 주요 단계로 구성됩니다. 첫 번째로, 사전 훈련된 텍스트 지식을 시각적 도메인으로 전이하는 그래프 관계 변환 모듈을 제안하고, 두 번째로 텍스트-비전 프로토타입 융합 모듈을 통해 시각적 및 언어 임베딩을 융합합니다. 또한, 도메인 격차를 줄이기 위해 컨텍스트 프롬프트 학습을 도입하고, 상상된 대조 학습(imagined contrastive learning)을 활용하여 불충분한 텍스트 데이터를 보완합니다.

- **Performance Highlights**: LRT는 mini-ImageNet과 CIFAR-100 FSCIL 벤치마크의 최종 세션에서 기존 모델보다 각각 13.3%와 7.3% 더 나은 성능을 보였습니다. 이 방법은 어떤 보조 네트워크(텍스트 인코더 포함)도 의존하지 않으며, 최종 모델이 경량화되어 구현이 용이하다는 장점을 가지고 있습니다. LRT는 새 개념을 이해하면서도 기존 지식을 잊지 않는 종합적인 접근을 제공합니다.



### MRI Patterns of the Hippocampus and Amygdala for Predicting Stages of Alzheimer's Progression: A Minimal Feature Machine Learning Framework (https://arxiv.org/abs/2501.05852)
- **What's New**: 이 논문은 알츠하이머병(AD)의 세 단계인 초기 경도 인지장애(EMCI)와 후기 경도 인지장애(LMCI)를 구분하는 데 중점을 둔 최소한의 특징을 가진 머신러닝 프레임워크를 제안합니다. 기존의 임상 이미징 기술로 인한 혼잡함을 해결하기 위해, 이 프레임워크는 히포캄퍼스(hippocampus)와 아미그달라(amygdala)를 중요한 연구 영역으로 설정하였습니다. 또한, 데이터 전처리 및 차원 축소 기법(PCA 및 t-SNE)을 통해 예측 정확도를 88.46%로 높였습니다.

- **Technical Details**: 본 연구에서 사용된 신경영상 데이터는 ADNI 데이터베이스로부터 확보되었으며, 342개의 T1-weighted 구조적 자기공명영상(sMRI) 이미지를 분석하였습니다. 이 이미지는 사지탈 방향과 MPRAGE 시퀀스를 이용하여 수집되었고, EMCI, LMCI, AD 각각 104, 103, 105 사례를 포함합니다. 데이터는 전처리, 차원 축소 및 다양한 분류기 훈련 과정을 통해 분석되며, 최종적으로 새로운 MRI 스캔에 대해 레이블을 예측합니다.

- **Performance Highlights**: 제안된 프레임워크는 머신러닝 기법과 차원 축소 기술을 통합하여 알츠하이머병의 단계를 정확하게 예측하는 데 성공하였습니다. 이러한 접근법은 노이즈를 줄이고, 지역별 특징을 강조함으로써 향상된 분류 성능을 보여주었습니다. 전체적인 결과는 임상 응용 및 유용한 인사이트를 제공하는 데 기여할 것으로 예상됩니다.



### Identity-aware Feature Decoupling Learning for Clothing-change Person Re-identification (https://arxiv.org/abs/2501.05851)
Comments:
          Accepted by ICASSP2025

- **What's New**: 이 논문에서는 의류 변경 사람 재식별(CC Re-ID) 문제를 해결하기 위한 새로운 프레임워크인 Identity-aware Feature Decoupling (IFD)을 제안합니다. 기존의 연구들은 RGB 이미지에서 ID 관련 정보를 충분히 추출하는 데 어려움을 겪고 있었는데, IFD는 이 문제를 해결하기 위해 두 가지 스트림 아키텍처를 활용합니다. 특히, 이 프레임워크는 의복 마스크 처리된 이미지를 입력으로 사용하여 정체성에 대한 주의 가중치를 도출합니다.

- **Technical Details**: IFD 프레임워크는 메인 스트림과 주의(Attention) 스트림으로 구성되어 있습니다. 주의 스트림은 의복이 마스킹된 이미지를 처리하여 주의 가중치를 설정하고, 이를 통해 메인 스트림에 공간적 지식을 효과적으로 전달합니다. 이를 통해 ID 관련 정보가 풍부한 영역을 강조하며, 두 스트림 간의 의미적 간극을 줄이기 위해 의복 편향 감소 모듈도 제안되었습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안한 IFD 프레임워크는 여러 CC Re-ID 데이터셋에서 기존의 기본 모델들과 비교하여 뛰어난 성능을 보였습니다. 이는 의류 변화가 있는 인물을 재식별하는 데 있어 새로운 가능성을 제시합니다.  이 연구는 CC Re-ID 분야에서 기술적 진전을 이루고 있으며, 실용적인 응용 가능성을 더욱 확대하는 데 기여할 것입니다.



### Poetry in Pixels: Prompt Tuning for Poem Image Generation via Diffusion Models (https://arxiv.org/abs/2501.05839)
- **What's New**: 이번 논문에서는 문학 작품, 특히 시(Sonnet)에서의 텍스트-이미지 생성 문제를 해결하기 위해 PoemToPixel 프레임워크를 제안하고 있습니다. 이 프레임워크는 시의 본질적인 의미를 시각적으로 표현하는 이미지를 생성하는 데 중점을 두고 있으며, 시의 감정과 시각적 요소를 추출하는 PoeKey 알고리즘도 포함되어 있습니다.

- **Technical Details**: PoemToPixel 프레임워크는 시의 요약과 주요 요소 추출을 결합한 두 단계를 통해 시각화를 수행합니다. 특히, GPT-4o 모델을 사용하여 시를 요약하고, 이 요약을 기반으로 key 요소를 추출하여 차별화된 프롬프트를 생성합니다. 이러한 과정에서 사용자 언어를 시스템 언어로 변환하는 정제 작업이 이루어지며, 인간 피드백 루프를 활용하여 프롬프트를 지속적으로 수정하는 방식입니다.

- **Performance Highlights**: 이 연구는 시와 관련된 데이터셋을 활용하여 PoemToPixel 프레임워크의 효과성을 다양한 방식으로 평가했습니다. 특히, MiniPo라는 데이터셋을 통해 1001개의 아동 시와 이미지를 포함하여 다양한 장르와 시대의 데이터로 다양성을 확장했습니다. 시각적으로 생성된 이미지는 시의 고유한 메시지를 충실히 전달하는 데 성공하며, 텍스트-이미지 생성 분야에서 시의 시각화를 위한 새로운 접근 방식을 제공합니다.



### UltraRay: Full-Path Ray Tracing for Enhancing Realism in Ultrasound Simulation (https://arxiv.org/abs/2501.05828)
- **What's New**: 이 논문은 기존의 초음파 시뮬레이터들이 가지는 높은 정확성을 위해 필요한 컴퓨팅 시간 및 리소스 문제를 해결하기 위한 새로운 초음파 시뮬레이션 파이프라인, 즉 UltraRay를 제안합니다. 이 시스템은 경량 추적(ray tracing) 알고리즘을 통해 전극으로부터 센서까지의 경로를 추적하며, 고급 초음파 이미징을 모방하기 위해 최적화된 레이 방사 시스템을 도입합니다. 또한, 신호 처리 파이프라인을 통합하여 초음파 이미지 형성을 시뮬레이션하도록 설계되었습니다.

- **Technical Details**: 새로운 방법론에서는 물리 기반 렌더링 원리를 활용하여 울트라사운드 과정의 모든 상호작용을 모델링하고, 경량 추적을 통해 소리 파동의 전체 경로를 추적합니다. 기본 신호 처리 단계를 포함하여, 이 시스템은 최종 초음파 이미징 과정의 단계들을 구현합니다. 이는 기존의 레이 추적 방법들이 단순한 에코 데이터를 생성하는 것과는 대조적으로, 진정한 음향 데이터의 처리를 가능하게 합니다.

- **Performance Highlights**: UltraRay는 반사율이 높은 물체 모델링을 통해 시뮬레이션 이미지의 리얼리즘을 높이며, 부정확한 아티팩트를 줄이는 데 기여합니다. 평가 결과, 기존의 초음파 이미지 생성 방식에 비해 전반적인 시각 품질이 향상되었습니다. 유연한 데이터 수집 전략을 통해 병렬 처리를 가능하게 하여 초음파 시뮬레이터의 효율성을 크게 개선하고, 기계 학습 또는 고급 초음파 빔 형성 전략과의 통합을 위한 기초를 제공합니다.



### PersonaHOI: Effortlessly Improving Personalized Face with Human-Object Interaction Generation (https://arxiv.org/abs/2501.05823)
- **What's New**: PersonaHOI는 개인화된 얼굴 확산 모델(PFD)과 일반적인 StableDiffusion 모델을 결합하여 훈련 및 조정 없는 인간-객체 상호작용(HOI) 이미지를 생성하는 새로운 프레임워크입니다. 기존 PFD 모델은 얼굴 특징을 지나치게 강조하여 전체 신체의 일관성을 해칠 수 있지만, PersonaHOI는 HOI 지향 텍스트 입력에 의해 안내되는 StableDiffusion 브랜치를 추가하여 이러한 문제를 해결합니다.

- **Technical Details**: PersonaHOI는 개인화된 얼굴 특징을 유지하면서도 검증된 HOI 레이아웃을 생성하기 위해 간단한 구조로 두 개의 경로를 통합합니다. PFD 브랜치에서는 Cross-Attention Constraint를 도입하여 이미지 전반에 걸쳐 정체성 특징의 과잉 강조를 방지하며, 레이어 간의 공간 혼합을 통해 상호작용이 풍부한 결과를 생성합니다.

- **Performance Highlights**: 새로운 '상호작용 정렬(interaction alignment)' 메트릭을 통해 PersonaHOI의 성능을 평가했으며, 인체-객체 상호작용의 사실성을 객관적으로 측정했습니다. 실험 결과, 최신 얼굴 개인화 기술 대비 상호작용의 사실성에서 상당한 개선을 보여주는 등, 이 접근 방식의 확장성 및 강건성을 강조합니다.



### UV-Attack: Physical-World Adversarial Attacks for Person Detection via Dynamic-NeRF-based UV Mapping (https://arxiv.org/abs/2501.05783)
Comments:
          23 pages, 22 figures, submitted to ICLR2025

- **What's New**: 최근 연구에서는 인간의 동적 행동을 모델링하는 Neural Radiance Fields (NeRF) 기반의 UV-Attack 방법이 소개되었습니다. 이 방법은 다양한 행동과 관점에서 인간 이미지를 생성할 수 있으며, 이전의 정적 이미지 기반 공격 방식보다 훨씬 더 높은 공격 성공률을 보여줍니다. 이 연구는 사람 탐지기(Person Detector)에 대한 효과적인 적대적 공격 가능성을 확대했습니다.

- **Technical Details**: UV-Attack은 동적 NeRF를 활용하여 UV 맵을 생성하고 텍스처를 실시간으로 수정하는 혁신적인 접근법을 사용합니다. 3D 인간 신체의 형상과 텍스처를 별도로 모델링하여, 학습 가능한 Densepose 기반의 UV 맵을 사용하여 3D 텍스처를 수정합니다. 이를 통해 unseen poses와 관점에서의 공격 성공률을 높이기 위해 Expectation over Pose Transformation (EoPT) 손실 함수를 사용합니다.

- **Performance Highlights**: UV-Attack은 FastRCNN 모델에 대해 92.75%의 공격 성공률을 기록하며, 최신 YOLOv8 탐지기에 대해 49.5%의 공격 성공률을 달성했습니다. 이 연구는 사람이 동적이고 다양한 움직임을 보일 때에도 효과적으로 작동하여 현실 세계에서의 공격 성공률을 높였습니다. 기존의 AdvCamou 공격이 28.50%의 공격 성공률을 보인 반면, UV-Attack은 그 성능을 크게 넘어섰습니다.



### StructSR: Refuse Spurious Details in Real-World Image Super-Resolution (https://arxiv.org/abs/2501.05777)
- **What's New**: 이번 연구에서는 Diffusion 기반의 이미지 초해상도(Real-ISR) 분야에서 구조적 신뢰도를 개선하고 구조적 오류를 줄이는 새로운 방법인 StructSR을 제안합니다. StructSR은 Structure-Aware Screening (SAS) 메커니즘을 핵심으로 하여 저해상도 이미지(LR)에 가장 구조적으로 유사한 이미지를 식별하고, 이를 기반으로 스퍼리어스한(detail-less) 디테일을 억제합니다. 이 방법은 추가적인 파인튜닝이나 외부 모델 프라이어 없이 기존 모델과 통합될 수 있어 편리합니다.

- **Technical Details**: StructSR은 구조적 충실도를 높이기 위해 3개의 모듈, 즉 구조적 인지 스크리닝(SAS), 구조적 조건 임베딩(SCE), 이미지 디테일 임베딩(IDE)으로 구성됩니다. SAS는 초기 추론 단계에서 LR 이미지에 가장 높은 구조적 유사성을 지닌 이미지를 식별하여, SCE는 이 구조적 정보를 사용해 고충실도의 구조 정보를 생성하는 데 도움을 줍니다. IDE는 각 단계에서 LR 이미지의 열화 정도에 따라 정리된 잠재 이미지에 구조적 임베딩을 삽입하여 스퍼리어스한 디테일을 억제합니다.

- **Performance Highlights**: 본 실험 결과에 따르면, StructSR은 기존 Diffusion 기반 Real-ISR 모델들과 통합하여 PSNR과 SSIM 지표에서 평균 각각 5.27% 및 9.36%의 개선을 보여줍니다. 또한 실제 데이터셋인 RealSR 및 DRealSR에서도 각각 4.13% 및 8.64%의 성능 개선이 있었습니다. 이러한 실험 결과는 StructSR이 Diffusion 기반의 Real-ISR 방법에서 구조적 충실도를 효과적으로 향상시키면서 스퍼리어스한 디테일을 억제할 수 있음을 보여줍니다.



### Conditional Diffusion Model for Electrical Impedance Tomography (https://arxiv.org/abs/2501.05769)
- **What's New**: 이 연구에서는 전압 일관성을 유지하는 조건부 확산 모델(Conditional Diffusion Model with Voltage Consistency, CDMVC)을 제안합니다. 이 방법은 재구성의 초기 이미지를 생성하는 전처리 모듈과 복원을 위한 조건부 확산 모델, 전방 전압 제약 네트워크 및 샘플링 과정에서의 전압 일관성 제약 방식으로 구성됩니다. CDMVC는 EIT의 복원 이미지 품질을 향상시키기 위해 측정된 전압의 유용성을 최대화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: EIT(전기 임피던스 토모그래피) 이미징 기술은 비침습적이며 전압 측정을 통한 임피던스 분포를 이미지로 생성합니다. 그러나 기존의 재구성 알고리즘은 비선형적이고 잘 정립되지 않은 문제로 인해 노이즈에 취약합니다. 연구에서는 전방 제약 네트워크를 도입하여 전도도 분포와 경계 전압 간의 관계를 설정하고, 이를 통해 확산 모델의 생성 성능을 향상시키기 위한 경량 네트워크가 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션과 실제 실험을 통해 검증되었으며, 재구성된 이미지의 품질이 크게 향상된다는 결과를 보여주었습니다. 실험 결과, CDMVC는 강력하고 정확한 재구성 성능을 보였으며, 고주파 노이즈 및 복잡한 환경에서도 잘 작동하는 것으로 나타났습니다. 이 연구는 EIT의 이미지 품질을 현저하게 개선할 수 있는 가능성을 제시합니다.



### StarGen: A Spatiotemporal Autoregression Framework with Video Diffusion Model for Scalable and Controllable Scene Generation (https://arxiv.org/abs/2501.05763)
- **What's New**: 최근 대규모 재구성 및 생성 모델의 발전이 장면 재구성과 새로운 시점 생성에서 크게 향상되었습니다. 그러나 계산 자원의 한계로 인해 이러한 대규모 모델을 사용한 추론은 짧은 범위로 제한되어 있어 장거리 일관된 장면 생성을 어렵게 만들었습니다. 이에 대한 해결책으로, 우리는 최근 개발된 StarGen이라는 프레임워크를 제안합니다.

- **Technical Details**: StarGen은 사전 훈련된 비디오 확산 모델을 자율 회귀 방식으로 활용하여 장거리 장면 생성을 목적으로 합니다. 각 비디오 클립의 생성은 공간적으로 인접한 이미지의 3D 워핑 및 이전에 생성된 클립의 시간적으로 겹치는 이미지에 조건을 부여하여 이루어집니다. 이를 통해 정밀한 포즈 제어가 가능한 시공간 일관성을 향상시킵니다.

- **Performance Highlights**: 정량적 및 정성적 평가를 통해 StarGen은 최신 방법들에 비해 뛰어난 확장성, 진실성 및 포즈 정확성을 달성한 것으로 나타났습니다. 잦은 모션 변화와 다양한 조건을 지원하는 StarGen은 희박한 뷰 보간, 지속적인 뷰 생성 및 레이아웃 조건에 따른 도시 생성 등 다양한 과제를 수행할 수 있습니다.



### Locality-aware Gaussian Compression for Fast and High-quality Rendering (https://arxiv.org/abs/2501.05757)
Comments:
          28 pages, 15 figures, and 14 tables

- **What's New**: LocoGS는 공간 일관성을 활용하여 3D Gaussian Splatting(3DGS)의 압축 성능과 렌더링 속도를 향상시키는 새로운 프레임워크입니다. 이 연구는 근처의 Gaussian 간의 강한 지역 일관성을 분석하여 이를 기반으로 하는 독창적인 3D Gaussian 표현을 도입합니다. 이를 통해 LocoGS는 기존의 3D 표현 방식에서의 저장 공간 요구를 크게 줄일 수 있는 동시에 렌더링 품질을 향상시킵니다.

- **Technical Details**: LocoGS의 핵심 요소는 밀집 초기화(dense initialization), Gaussian 가지치기(pruning), 적응형 구형 조화 대역폭(adaptive spherical harmonics bandwidth) 방식, 그리고 각 Gaussian 속성에 맞춘 양자화(quantization) 및 인코딩(encoding) 기법을 통합한 것입니다. 이 프레임워크는 고밀도 포인트 클라우드 초기화 및 다중 해상도 해시 그리드(multi-resolution hash grid)를 활용하여, 지속적인 물리량의 compact modeling을 가능하게 합니다. 특히, 기존의 3DGS 렌더링 파이프라인을 수정하지 않고도 원래의 렌더링 효율성을 유지합니다.

- **Performance Highlights**: 실험 결과에 따르면, LocoGS는 54.6배에서 96.6배의 압축 저장 크기와 2.1배에서 2.4배의 렌더링 속도를 달성하여 기존의 압축 Gaussian 표현 방법에 비해 우수한 성능을 나타냅니다. 또한, 최신 압축 방법인 HAC와 비교했을 때도 평균 2.4배 높은 렌더링 속도를 보여줍니다. 이러한 성과는 실제 3D 데이터 세트에 대한 렌더링 품질을 더욱 향상시키면서, 저장 공간 요구를 크게 줄여 주목받고 있습니다.



### LLVD: LSTM-based Explicit Motion Modeling in Latent Space for Blind Video Denoising (https://arxiv.org/abs/2501.05744)
- **What's New**: 이번 논문에서는 Latent space LSTM Video Denoiser (LLVD)라는 새로운 알고리즘을 제안합니다. LLVD는 비디오 캡처 중 발생하는 노이즈 제어를 위한 모델로, 영상의 시각적 품질을 향상시키고 불필요한 노이즈 아티팩트를 줄이는 데 중점을 두고 있습니다. 이 모델은 공간적(spatial) 및 시간적(temporal) 피쳐 추출을 통합하여 LSTM을 활용하며, 연속성 유지와 깜빡임 최소화에 중요한 역할을 합니다.

- **Technical Details**: LLVD는 인코딩된 피쳐 도메인 내에서 LSTM 레이어를 통합하여 설계되었습니다. 이 접근 방식은 연속적인 비디오 프레임 간의 시간적 관계를 효과적으로 캡처하고, 높은 계산 효율성을 제공합니다. LLVD는 계산 복잡도를 크게 줄이며, 실시간 어플리케이션에서도 유용하게 사용할 수 있도록 경량화된 구조를 가지고 있습니다.

- **Performance Highlights**: 실험 결과 LLVD는 합성된 노이즈와 실제로 캡처된 노이즈 모두에서 우수한 성능을 보여주었습니다. 특히, LLVD는 기존의 SOTA(State-of-the-Art) 모델보다 0.3dB 높은 성능을 달성하였으며, 계산 복잡도를 59% 감소시켰습니다. 이러한 성과는 노이즈 특성에 대한 사전 지식 없이 이루어졌습니다.



### TB-Bench: Training and Testing Multi-Modal AI for Understanding Spatio-Temporal Traffic Behaviors from Dashcam Images/Videos (https://arxiv.org/abs/2501.05733)
Comments:
          Main Paper: 8 pages, Supplementary Materials: 15 pages

- **What's New**: 이번 연구에서는 다중 모드 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 자율 주행(Autonomous Driving, AD) 적용을 위한 새로운 벤치마크인 TB-Bench를 제안합니다. 이 벤치마크는 트래픽 행동을 이해하기 위한 8개의 인식 작업을 평가할 수 있도록 설계되었습니다. 또한, 이 연구에서는 교통 데이터에 최적화된 고품질 데이터 세트인 TB-100k 및 TB-250k도 소개하여 MLLMs의 성능 향상을 꾀합니다.

- **Technical Details**: TB-Bench는 차량의 에고 중심 뷰에서 대시캠 이미지나 비디오를 기반으로 트래픽 행동을 이해하는 MLLM의 능력을 평가합니다. 연구팀은 질문과 이미지 또는 비디오 클립을 조합하여 MLLM이 평문 텍스트로 응답하도록 요구하는 평가 프로토콜을 설정했습니다. 이러한 작업을 통해 MLLMs의 응답 정확도를 측정하여 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 기존 MLLMs는 TB-Bench에서 평균적으로 35% 미만의 낮은 정확도를 보였지만, TB-100k 및 TB-250k로 미세 조정된 베이스라인 모델들은 평균적으로 77%에서 85%까지의 높은 성능을 달성했습니다. 이는 제안된 데이터셋이 MLLM의 트래픽 행동 이해를 크게 향상시킬 수 있음을 보여줍니다. 또한, TB-100k로 훈련된 모델이 다른 자율 주행 벤치마크에서도 일반화 가능성을 보여주었습니다.



### Super-class guided Transformer for Zero-Shot Attribute Classification (https://arxiv.org/abs/2501.05728)
Comments:
          AAAI25

- **What's New**: 이번 연구는 zero-shot attribute classification을 위한 새로운 프레임워크인 Super-class guided transFormer (SugaFormer)를 제안합니다. SugaFormer는 super-classes를 활용하여 모델의 확장성과 일반성을 향상시키는 것을 목표로 합니다. 이를 통해 모델은 미지의 클래스에 대한 이해도를 높이고, 다양한 속성을 보다 효과적으로 분류할 수 있습니다.

- **Technical Details**: SugaFormer는 Super-class Query Initialization (SQI)과 Multi-context Decoding (MD)을 적용하여 속성 분류의 성능을 향상시킵니다. SQI는 common semantic information을 이용하여 쿼리 수를 줄이고, MD는 다양한 시각적 단서를 처리하여 모델의 정확성을 높입니다. 또한, Knowledge transfer 전략으로는 Super-class guided Consistency Regularization (SCR)과 Zero-shot Retrieval-based Score Enhancement (ZRSE)를 도입하여 학습 및 추론 과정에서의 일반화 능력을 강화합니다.

- **Performance Highlights**: SugaFormer는 세 가지 속성 분류 벤치마크(VAW, LSA, OVAD)에서 최고 수준의 성과를 보였으며, zero-shot 및 cross-dataset transfer 설정에서도 뛰어난 성능을 발휘했습니다. 실험 결과, SugaFormer는 다른 접근 방식들과 비교하여 속성 인식의 정확성을 크게 개선하였습니다. 따라서, 이 연구는 속성 분류 분야에서 새로운 방향성을 제시하며, 자율 주행 및 이미지 추천 시스템 등 다양한 응용 분야에 기여할 수 있습니다.



### Zero-shot Shark Tracking and Biometrics from Aerial Imagery (https://arxiv.org/abs/2501.05717)
- **What's New**: 본 논문에서는 드론을 사용한 해양 동물 연구에서의 Machine Learning (ML)의 새로운 접근 방식으로 Frame Level ALIgment and tRacking (FLAIR) 시스템을 소개합니다. FLAIR는 Segment Anything Model 2 (SAM2)와 Contrastive Language-Image Pre-training (CLIP)의 비디오 이해 및 비전-언어 기능을 활용하여 드론 영상에서 특정 종의 분할 마스크를 자동으로 생성합니다. 특히, FLAIR는 레이블이 부착된 데이터 무 필요의 제로샷(zero-shot) 접근 방식을 도입하여 데이터 주석 작업과 기존 모델의 재훈련 없이 다양한 종에 일반화할 수 있습니다.

- **Technical Details**: FLAIR는 드론 비디오를 입력으로 받아 관심 있는 해양동물의 분할 마스크를 출력합니다. 본 시스템은 18,000개의 태평양 간호상어 이미지 데이터셋을 기반으로 훈련된 최첨단 객체 탐지 모델들과 성능을 비교하였으며, FLAIR는 다른 객체 탐지 시스템과 경쟁적인 성과를 나타내며 0.81의 Dice 점수를 기록했습니다. FLAIR의 혁신적인 점은 데이터를 주석 달 필요 없이 다양한 종으로 일반화할 수 있고, 새로운 휴리스틱과 결합하여 필요한 생체 정보를 자동으로 추출할 수 있다는 것입니다.

- **Performance Highlights**: FLAIR는 기존의 객체 탐지 모델을 압도적으로 능가하며, 인간-루프 방식의 두 가지 방법들과도 경쟁하는 성과를 냈습니다. 이 시스템은 전통적인 ML 워크플로우에 비해 인간의 노력과 전문 지식이 현저히 적게 요구되며, 더 높은 정확도를 달성할 수 있습니다. 따라서 FLAIR는 해양 생태계에 대한 해석과 통찰을 도출하는 데 더 많은 시간을 할애할 수 있게 합니다.



### From My View to Yours: Ego-Augmented Learning in Large Vision Language Models for Understanding Exocentric Daily Living Activities (https://arxiv.org/abs/2501.05711)
- **What's New**: 이번 논문에서는 Large Vision Language Models (LVLMs)의 ADL (Activities of Daily Living) 영상 이해를 향상시키기 위해 새로운 접근 방식을 제안합니다. 기존 LVLMs는 객관적인 시점(exocentric)에서는 미세한 상호작용을 이해하는 데 제한적이었으나, 본 연구는 주관적인 시점(egocentric) 데이터를 활용하여 이러한 한계를 극복하고자 합니다. 특히, EgoMimic이라는 접근 방식을 통해 시각적 세부 정보를 효과적으로 생성하고 LVLMs의 이해력을 증진시키는 방법을 제시합니다.

- **Technical Details**: 이 연구는 Ego2Exo 전달(distillation) 방식을 온라인으로 구현하여 ego-보강(exo augmented) 표현을 LVLMs에서 학습합니다. 특히, EgoMimic 방법론을 사용하여 이식적인 카메라 없이도 exo 비디오로부터 mimicked ego 시점을 생성할 수 있습니다. 이 접근법은 손과 객체 간의 상호작용과 같은 미세한 정황을 포착하는 데 중점을 두어, ADL 작업을 보다 효과적으로 이해할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 여섯 가지 ADL 벤치마크와 함께 우리 연구를 위해 특별히 개발한 EgoPerceptionMCQ 벤치마크에서 그 유효성을 입증하였습니다. EgoPerceptionMCQ는 LVLM이 exo 비디오로부터 ego 큐를 이해할 수 있는지를 평가하는 5,000개 이상의 객관식 질문으로 구성되어 있습니다. 실험 결과 LVLMs는 강화된 ego-보강 표현을 통해 ADL 비디오에서 효과적인 인식을 수행하는 것으로 나타났습니다.



### EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Mod (https://arxiv.org/abs/2501.05710)
Comments:
          11 pages, 8 figures

- **What's New**: 최근의 연구에 따르면 감정은 사용자의 인지 기능을 향상시키고 정보 전달에 영향을 미칠 수 있습니다. 하지만 감정이 풍부한 이미지 콘텐츠를 생성하는 데에는 제한된 연구만 진행되었습니다. 본 논문에서는 Continuous Emotional Image Content Generation (C-EICG)이라는 새로운 작업을 도입하고, 텍스트 프롬프트 및 Valence-Arousal 값에 기초하여 이미지를 생성하는 EmotiCrafter 모델을 발표합니다.

- **Technical Details**: EmotiCrafter는 Valence-Arousal 모델을 사용하여 연속적인 감정 표현을 가능하게 하는 감정 임베딩 맵핑 네트워크를 채택하여, 특정 감정을 캡처하고 이를 바탕으로 감정적으로 풍부한 이미지 생성을 지원합니다. 이 네트워크는 텍스트 프롬프트와 결합된 V-A 값을 비선형적으로 임베딩하여 이미지 생성에 적합한 복잡한 감정 맥락을 포착합니다. 교육 데이터셋은 39,843개의 주석이 달린 이미지를 포함하며, 각 샘플은 중립 및 감정 프롬프트로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, EmotiCrafter는 특정 감정을 표현하면서도 원하는 콘텐츠를 갖춘 이미지를 효과적으로 생성하는 것으로 나타났습니다. 기존의 기술들과 비교했을 때, 본 방법은 감정 표현의 질에서 우수한 성과를 보였습니다. 이는 감정 임베딩 네트워크와 새로운 손실 함수가 효과적으로 작용했음을 보여줍니다.



### Overcoming Language Priors for Visual Question Answering Based on Knowledge Distillation (https://arxiv.org/abs/2501.05690)
Comments:
          Accepted to ICME2024

- **What's New**: 본 연구는 KDAR이라는 새로운 방법을 제안하여 VQA(Visual Question Answering) 작업에서 발생하는 언어 선행편향 문제를 해결합니다. KDAR은 지식 증류(knowledge distillation)를 활용하여 과적합(overfitting)을 방지하고, 소프트 레이블(soft labels)을 통해 후보 답변의 범위를 좁히는 정규화(regularization) 역할을 수행합니다. 또한, 각 샘플의 중요성을 동적으로 조정하는 샘플-별 가중치 재조정(sample-wise reweighting) 학습 전략을 설계하여 모델이 드문 샘플에 더 집중할 수 있도록 합니다.

- **Technical Details**: 제안된 KDAR 방법은 기계 학습에서의 적합(fitting) 문제로 언어 선행편향 문제를 보고, 소프트 레이블을 정규화 수단으로 활용하여 과적합 문제를 해결하려고 합니다. 또한, 학습 모형 입장에서 각 샘플의 손실 가중치를 동적으로 조정하는 방식으로 드문 샘플을 더 잘 학습할 수 있도록 하여, 동시적으로 공통 답변이 모델에 미치는 영향을 줄이는 구조입니다. 이 두 가지 학습 전략을 통해 VQA 모델의 일반화 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, KDAR은 VQA-CPv2 및 VQAv2 데이터 세트에서 이전의 최첨단 방법보다 훨씬 우수한 성능을 보여주었습니다. 특히, LXMERT와 결합된 경우 VQA-CPv2 데이터셋에서 71.33%라는 최고의 전체 정확도를 기록하였습니다. 또한 KDAR은 OOD(Out-Of-Distribution) 및 IID(Independent Identically Distributed) 환경 모두에서 성능을 향상시키는 데 기여하였습니다.



### eKalibr: Dynamic Intrinsic Calibration for Event Cameras From First Principles of Events (https://arxiv.org/abs/2501.05688)
- **What's New**: 최근 생체 영감을 받은 이벤트 카메라는 높은 동적 범위와 낮은 레이턴시 특성으로 중요한 연구 초점을 받고 있습니다. 본 논문에서는 이벤트 카메라의 정확한 내재적 보정을 위해 "eKalibr"라는 새로운 보정 방법을 제안합니다. 기존의 엔지니어링 중심 접근법과 복잡한 기기를 요구하는 방식의 단점을 보완하여, 사용의 편리함을 높이고 있습니다.

- **Technical Details**: eKalibr는 정밀하게 설계된 이벤트 기반 원형 그리드 패턴 인식 알고리즘을 기반으로 합니다. 이 과정에서 이벤트 기반의 노말 플로우 추정(normal flow estimation)을 통해 원형 경계에서 발생된 이벤트를 확인하고, 이를 공간적으로 클러스터링합니다. 최종적으로, 시간 변동 타원 추정(time-varying ellipse estimation)을 수행하여 그리드 패턴을 인식하는 과정을 완결합니다.

- **Performance Highlights**: eKalibr의 자세한 성능 평가 실험이 수행되었으며, 패턴 추출 및 내재적 보정 측면에서 우수한 결과를 보였습니다. 모든 데이터셋과 코드 구현은 오픈 소스로 제공되어 연구 커뮤니티에 기여하고 있습니다. 이 방법은 기존의 여러 한계를 극복할 수 있는 가능성이 있으며, 향후 이벤트 기반 및 관성 보정(calibration) 접근법으로 확장될 수 있습니다.



### UniQ: Unified Decoder with Task-specific Queries for Efficient Scene Graph Generation (https://arxiv.org/abs/2501.05687)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 UniQ라는 새로운 통합 디코더 아키텍처를 제안하여, 관계 triplet <subject, predicate, object>의 특징을 동시 모델링하면서도 개별적인 시각적 특징을 추출할 수 있도록 한다. UniQ는 각 하위 작업에 대해 별도의 쿼리를 생성하여, 이들 쿼리가 상호 작용할 수 있는 유기적 구조를 만들고, 이전 방식에서 간과했던 결합 기능을 효과적으로 모델링한다.

- **Technical Details**: UniQ는 세 가지 종류의 쿼리(주어, 목적어, 술어)를 통합 디코더에 입력으로 사용하며, 각 하위 작업에 대해 병렬로 비결합된 시각적 특징을 추출한다. 특히, 각 디코더 레이어에서 UniQ는 전역적 및 지역적 문맥을 통합하여 쿼리가 관계 맥락에 따라 동적으로 세부적인 시각적 특징을 파악할 수 있도록 지원한다. 이러한 방식은 관계 triplet 내의 복잡한 상호작용을 포착할 수 있는 triplet self-attention 모듈을 포함한다.

- **Performance Highlights**: Visual Genome 데이터셋에서의 실험 결과는 UniQ가 기존의 1단계 및 2단계 방법들보다 우수한 성능을 보임을 보여준다. 특히, 더 적은 매개변수로도 효과적인 특징 모델링과 높은 정확도를 달성하며, 이는 우수한 성능과 연산 효율성을 모두 제공한다. 또한, UniQ는 기존 방법들이 가지고 있는 약한 얽힘 문제를 효과적으로 해결한다.



### Deep Reversible Consistency Learning for Cross-modal Retrieva (https://arxiv.org/abs/2501.05686)
- **What's New**: 본 논문에서는 Cross-modal retrieval (CMR)을 위한 새로운 방법인 Deep Reversible Consistency Learning (DRCL)을 제안하고 있습니다. DRCL은 전통적인 양식인 pairwise training의 필요성을 줄여주는 모듈인 Selective Prior Learning (SPL)과 Reversible Semantic Consistency Learning (RSC)을 포함하고 있습니다. 이 방법은 모달리티 간의 표상을 유연하게 조정하며, 서로 다른 위상 간의 보다 일관된 의미 체계를 구축하는 데 중점을 두고 있습니다.

- **Technical Details**: DRCL은 Selective Prior Learning (SPL) 모듈에서 각 모달리티에 대해 최적의 변환 가중치 매트릭스를 학습하고, 품질 점수를 기준으로 가장 적합한 것을 선택하여 prior로 삼습니다. RSC는 샘플 레이블로부터 모달리티 불변의 표상을 재구성하는 Modality-invariant Representation Recasting (MRR) 메커니즘을 사용합니다. 이를 통해 샘플 특징과 카테고리 특징 간의 유의미한 일관성을 유지하며, feature augmentation 메커니즘(FA)을 통해 데이터 분포의 다양성을 장려합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행한 실험 결과, DRCL은 15개의 최첨단 방법들과 비교하여 우수한 성과를 입증하였습니다. 특히, DRCL은 다중 모달리티를 처리할 때 학습 복잡성을 크게 감소시키며, intra-class 샘플 간의 일관성을 유지하고 inter-class 샘플 간의 차별성을 극대화하는 데 효과적입니다. 전체적으로 DRCL은 CMR 분야에서 활용될 가능성을 보여줍니다.



### LPRnet: A self-supervised registration network for LiDAR and photogrammetric point clouds (https://arxiv.org/abs/2501.05669)
Comments:
          12 pages, 9 figures, 5 tables

- **What's New**: 이번 논문에서는 LiDAR와 photogrammetry를 활용한 heterogeneous(point cloud) 점군의 등록 문제를 해결하기 위한 자가 지도(self-supervised) 등록 네트워크를 제안합니다. 이러한 접근 방식은 두 가지 서로 다른 원거리 센서 기법의 데이터를 통합하는 데 기여하며, 이로 인해 발생하는 밀도, 정밀도, 노이즈 및 겹침 문제 해결에 중점을 둡니다. 특히, 다중 스케일 masked training 전략을 도입하여 견고한 특징(feature)을 추출합니다.

- **Technical Details**: 제안된 방법의 핵심은 회전-변환(translation) 임베딩 모듈을 포함하여 정확한 고형 변환(rigid transformations)을 위한 중요한 특징들을 효과적으로 캡처하는 것입니다. 이를 통해 self-supervision을 사용하여 heterogeneous LiDAR와 photogrammetric 점군으로부터 강력한 특징을 추출할 수 있습니다. 또한, transformer 기반 아키텍처는 다양한 점군 데이터셋에서 로컬(local)과 글로벌(global) 특징을 원활하게 통합하여 정밀한 정렬을 촉진합니다.

- **Performance Highlights**: 본 연구는 두 가지 실제 데이터셋에서 실시된 실험을 통해 제안된 방법의 효과성을 검증하였습니다. 이 방법은 다양한 점군 데이터셋 간의 정밀한 정렬을 가능하게 하여, ground truth 획득의 어려움을 해소하는 데 크게 기여합니다. LiDAR와 photogrammetric 점군 모두에 대해 강력한 특징 추출 능력을 입증하였습니다.



### HFMF: Hierarchical Fusion Meets Multi-Stream Models for Deepfake Detection (https://arxiv.org/abs/2501.05631)
Comments:
          This work is accepted to WACV 2025 Workshop on AI for Multimedia Forensics & Disinformation Detection. Code is available at: this https URL

- **What's New**: HFMF는 심층 생성 AI 모델에 의해 생성된 이미지를 탐지하는 능력을 향상시키기 위해 계층적 교차 모드 특성 융합과 다중 스트림 특성 추출을 활용하는 두 가지 단계의 딥페이크 탐지 프레임워크입니다. 이 프레임워크는 비전 변환기 및 합성곱 신경망을 포함한 다양한 최신 기술을 통합하여 정확한 분류 성능을 왜곡합니다. 또한, 이 프레임워크는 경량 모델을 사용하여 훈련 가능 매개변수 수를 최소화해 높은 계산 효율성을 유지합니다.

- **Technical Details**: HFMF는 두 개의 보완적인 모듈로 구성되어 있습니다. 첫 번째 모듈에서는 계층적 교차 모드 특성 융합 기법을 통해 비전 변환기(Vision Transformer, ViT)와 합성곱 신경망(Convolutional Network, ConvNet)을 통합하여 딥페이크 조작의 지역적인 아티팩트 및 전반적인 불일치를 캡처합니다. 두 번째 모듈에서는 경량화된 전문 모델을 사용하여 엣지 특성, 객체 수준의 맥락, 그리고 일반 이미지 기반 특성을 추출하고, 이 두 모듈의 출력을 앙상블 심층 신경망을 통해 융합합니다.

- **Performance Highlights**: HFMF 모델은 다양한 데이터셋 벤치마크에서 뛰어난 성능을 달성하며, 특히 현실 세계의 조건에서도 탁월합니다. 연구진은 HFMF가 기존의 탐지 모델보다 우수한 정확도를 기록했음을 입증했습니다. 또한 명확성과 재현 가능성을 유지하면서 상대적으로 적은 수의 훈련 가능한 매개변수로 높은 탐지 성능을 보였습니다.



### Approximate Supervised Object Distance Estimation on Unmanned Surface Vehicles (https://arxiv.org/abs/2501.05567)
- **What's New**: 이 연구는 자율 수상 차량(USV)에서의 거리 추정을 위한 새로운 접근 방식을 제시합니다. 기존의 거리 측정 기술에 대한 비용 효율적이고 직관적인 대안으로, 수동으로 주석이 달린 이미지와 거리 측정값을 포함하는 데이터를 수집하였습니다. 제안된 방법은 물체 감지와 거리 예측을 동시에 수행하며, 이는 인간의 추정 능력과 더 일치합니다. 이를 통해 환경 내에서 가까운 물체에 대한 경고 시스템을 구현할 수 있습니다.

- **Technical Details**: 이 방법은 YOLOv7 및 YOLOv9 객체 감지 모델을 활용하여 각 물체의 거리를 직접 예측하도록 수정된 아키텍처를 사용합니다. 추가 출력을 사용하여 검출된 물체와의 거리를 예측하며, 다양한 거리 분포에 대한 정규화 전략을 실험했습니다. 단순히 거리를 예측하기보다는, 예측값을 안정적으로 유지하기 위한 전략들이 필요하며, 이를 통해 학습 안정성을 확보합니다.

- **Performance Highlights**: 제안한 방법은 다양한 객체 추적기와 결합하여 성능을 분석하였으며, 실제 해양 환경에서의 실험으로 효과를 입증하고자 하였습니다. 고유한 해양 문제를 해결할 수 있으며, 비용이 많이 드는 다른 센서를 사용하지 않고도 향상된 거리 추정을 가능하게 합니다. 이 연구는 기존 데이터셋의 한계를 극복하고, 해양 컴퓨터 비전 분야에 기여할 수 있는 방법론을 제시합니다.



### Vision-Language Models for Autonomous Driving: CLIP-Based Dynamic Scene Understanding (https://arxiv.org/abs/2501.05566)
- **What's New**: 이 논문에서는 자동 운전 차량(AV)의 결정을 위한 인간 중심 설명을 생성하고, 운전 비디오 분석에 인공지능(AI)을 활용하기 위해 동적 장면 검색 시스템을 개발하였습니다. 이 시스템은 Contrastive Language-Image Pretraining (CLIP) 모델을 사용하며, 엣지 장치(edge devices)에서 실시간 배치 최적화가 가능합니다. 특히 복잡한 시나리오에서 GPT-4o의 제로샷(zero-shot) 능력을 포함한 최신 인-context learning 방법들을 능가합니다.

- **Technical Details**: 이 연구는 Honda Scenes Dataset을 활용하여 프레임 레벨(frame-level) 분석을 진행하였습니다. 이 데이터셋은 약 80시간 분량의 주행 비디오를 포함하고 있으며, 다양한 실제 도로 및 날씨 조건을 담고 있습니다. CLIP 모델들은 자연어(supervision)로부터 시각적 개념을 학습하는 데 뛰어난 강건성을 보여주었으며, ViT-L/14와 ViT-B/32와 같은 클립 모델의 미세 조정(fine-tuning)을 통해 장면 분류에서 F1 점수를 91.1%로 달성하였습니다.

- **Performance Highlights**: 이 시스템은 정밀하고 신속한 장면 인식을 제공할 수 있는 능력을 입증하였으며, 이는 고급 운전 보조 시스템(ADAS)의 필수 요구 사항을 충족하는 데 기여할 수 있습니다. CLIP 모델이 동적 장면 이해 및 분류를 위한 확장 가능하고 효율적인 프레임워크를 제공하는 잠재력을 보여주고 있습니다. 또한, 이 연구는 운전자의 행동, 도로 조건, 안전과 관련된 시나리오에 대한 이해를 깊게 하여 보다 스마트하고 안전한 자율 주행 시스템 개발을 위한 기초를 다졌습니다.



### Improving Zero-Shot Object-Level Change Detection by Incorporating Visual Correspondenc (https://arxiv.org/abs/2501.05555)
- **What's New**: 이번 논문에서는 두 이미지 간의 객체 수준 변화 감지에 대한 새로운 접근법을 제시합니다. 기존의 방법들이 가지는 세 가지 주요 한계를 해결하기 위해, 변화 상관 관계를 활용하여 변화 감지 정확도를 향상시키고, 테스트 시 허위 긍정률을 줄입니다. 이 방법은 객체의 추가 또는 제거 위치에 대한 감독 라벨을 활용하여 정확성을 크게 개선하였으며, 관계 예측을 위한 새로운 데이터셋 OpenImages-Inpainted를 도입했습니다.

- **Technical Details**: 연구 필드에서는 일반적으로 객체의 변화를 감지하는 데 있어 여러 가지 변형(예: 카메라 각도 변화, 색 변화, 조명 변화)에 대한 문제를 겪습니다. 본 논문에서 제안하는 방법은 양쪽 이미지 간의 변화를 비교하고, 헝가리안 알고리즘(Hungarian algorithm)을 활용해 예측된 변경 사항 간의 대응 관계를 예측하는 시스템입니다. 이를 통해 허위 긍정률을 최소화하고 더 높은 변화 감지 정확도를 달성합니다.

- **Performance Highlights**: 이 모델은 변화를 감지하는 정확도에서 기존 방법들보다 뛰어난 성능을 보이며, 여러 벤치마크에서 최첨단 결과를 얻었습니다. 예를 들어, 제안하는 contrastive matching 손실 함수는 모든 다섯 개의 벤치마크에서 정확도를 +1.31에서 +6.56로 향상시켰습니다. 또한, 변경 감지 성능을 평가하기 위한 새로운 메트릭을 제안하여 상대적인 성과를 일관되게 비교할 수 있게 했습니다.



### OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? (https://arxiv.org/abs/2501.05510)
Comments:
          28 pages

- **What's New**: 이번 논문에서는 OVO-Bench(Online-VideO-Benchmark)를 소개하며, 이를 통해 시간 정보를 기반으로 하는 온라인 비디오 이해 능력을 평가할 수 있는 새로운 벤치마크를 제시합니다. 기존의 오프라인 모델과 달리, OVO-Bench는 비디오 스트림을 실시간으로 처리하고, 다양한 시나리오에 따라 정확한 응답을 생성하는 능력을 강조합니다. 이러한 평가를 통해 모델이 실제 비디오 이해 작업에서 어떻게 성능을 발휘하는지를 파악할 수 있는 기회를 제공합니다.

- **Technical Details**: OVO-Bench는 644개의 다양한 비디오를 포함하며, 총 12개의 작업을 통해 모델의 시간 인식 능력을 평가합니다. 주요 작업으로는 과거 사건 추적, 실시간 사건 이해, 미래 정보에 따른 적시 응답이 포함되며, 이를 통해 비디오 LLMs의 성능을 보다 정밀하게 측정할 수 있습니다. 또한, 자동 생성 파이프라인과 인간 검증을 결합하여 약 2,800개의 메타 주석을 생성하였습니다.

- **Performance Highlights**: 실험 결과, 9개의 비디오 LLMs는 오프라인 성능에서 우수한 평가를 받았으나, 온라인 환경에서의 질의에는 제 기능을 발휘하지 못하는 경향을 보였습니다. 특히, 최신 스트리밍 모델인 Flash-VStream의 경우 오프라인 모델에 비해 더욱 큰 성능 차이를 보였으며, 이는 온라인 비디오 이해 기술 개선을 위한 연구의 필요성을 강조합니다. OVO-Bench는 이러한 연구를 촉진하는 데 기여할 것으로 기대됩니다.



### Tuning-Free Long Video Generation via Global-Local Collaborative Diffusion (https://arxiv.org/abs/2501.05484)
- **What's New**: 이 논문에서는 GLC-Diffusion이라는 새로운 기법을 제안하여 tuning-free 방식을 통해 고품질의 일관된 긴 비디오 생성을 가능하게 합니다. 이 방법은 Global-Local Collaborative Denoising 기법을 활용해 전체 콘텐츠 일관성 및 프레임 간의 시간적 일관성을 보장하는 방식으로 긴 비디오의 디노이징 과정을 모델링 합니다. 또한, Noise Reinitialization 전략과 Video Motion Consistency Refinement (VMCR) 모듈을 도입하여 비디오 생성을 개선하고, 기존 비디오 확산 모델과 효과적으로 통합되어 우수한 성능을 나타냅니다.

- **Technical Details**: GLC-Diffusion은 긴 비디오 생성을 위한 통합 최적화 문제로 디노이징 과정을 모델링합니다. 이 기술은 Global-Local Collaborative Denoising 기법을 사용하여 전체 비디오의 일관성과 연속성을 유지하는 동시에, Random Shifting Sampling 기법으로 지역적 시간적 일관성을 보강합니다. Noise Reinitialization 전략은 로컬 노이즈 셔플링을 통해 시각적 다양성을 개선하며, VMCR 모듈은 프레임 간의 시각적 일관성을 최적화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 GLC-Diffusion은 CogVideoX 모델을 48프레임에서 1,000프레임 이상으로 확장할 수 있음을 입증하였습니다. 이 방법은 다른 기존 비디오 확산 모델과 통합 가능하며, 시간적 일관성과 시각적 충실도를 크게 향상시키면서 이전 방식보다 더 높은 품질의 긴 비디오를 생성합니다. 양적 및 정성적 평가에서 GLC-Diffusion의 성능은 이전 방법보다 우수함을 보여주었습니다.



### Implicit Guidance and Explicit Representation of Semantic Information in Points Cloud: A Survey (https://arxiv.org/abs/2501.05473)
- **What's New**: 이번 논문에서는 포인트 클라우드(point cloud)와 2D 장면의 의미적 정보(semantic information) 통합을 통해 다양한 작업의 정확성과 효율성을 향상시키고자 하는 연구 동향을 종합적으로 리뷰합니다. 특히, 기존의 연구들에서 간과되었던 포인트 클라우드의 의미적 정보 활용을 포함한 다양한 전통적 및 신흥 작업들을 심층적으로 분석합니다. 이 연구는 포인트 클라우드 처리에 있어 의미적 분석이 어떻게 적용될 수 있는지를 보여주며, 이를 통해 더 나은 이해와 미래의 연구 방향을 제시합니다.

- **Technical Details**: 3D 포인트 클라우드는 도시 계획, 자율주행차, 측량 및 매핑 등 다양한 분야에서 사용됩니다. 추가적으로, 포인트 클라우드는 객체 감지(object detection), 인스턴스 분할(instance segmentation), 장면 그래프 예측(scene graph prediction) 및 포인트 클라우드 이해(point cloud understanding) 등의 작업에서도 활용되고 있습니다. 의미적 정보는 고차원 특징을 제공하여, 포인트 클라우드의 목표 함수(optimization)인 압축(compression), 이미지 향상(image enhancement) 및 이미지 인식(image recognition)을 위한 암묵적(guidance) 도움을 제공합니다.

- **Performance Highlights**: 이 논문에서는 포인트 클라우드의 의미적 과제에 대한 체계적인 조사를 통해 향후 연구 방향을 제시합니다. 의미 기반 작업의 발전은 포인트 클라우드의 정보가 클러스터(clustering), 검색(retrieval) 및 시나리오 이해에서 더욱 효율적으로 활용될 수 있음을 보여줍니다. 이로 인해 각각의 포인트가 받는 레이블(label)은 3D 장면에 대한 보다 정밀한 이해와 분석을 지원하고, 신흥 과제들이 어떻게 발전하고 있는지에 대한 구체적인 통찰을 제공합니다.



### The 2nd Place Solution from the 3D Semantic Segmentation Track in the 2024 Waymo Open Dataset Challeng (https://arxiv.org/abs/2501.05472)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 자율주행 기술의 필수 요소인 3D semantic segmentation의 성능을 향상시키기 위해 MixSeg3D라는 새로운 접근 방법을 제안합니다. 기존의 LiDAR 스캔 데이터가 가진 훈련의 다양성 부족 문제를 해결하기 위해, 저자들은 MinkUNet 모델을 사용하고 LaserMix 및 PolarMix라는 두 가지 데이터 증강 기법을 결합했습니다. 이 방법은 LiDAR 포인트 클라우드를 다양한 각도에서 혼합하여, 모델이 여러 환경에서 잘 일반화하도록 돕습니다. 이를 통해 저자들은 2024 Waymo Open Dataset Challenge에서 3D semantic segmentation 트랙에서 2위를 기록했습니다.

- **Technical Details**: MixSeg3D는 강력한 포인트 클라우드 세분화 모델인 MinkUNet을 기반으로 하며, LaserMix와 PolarMix라는 두 가지 혁신적인 3D 데이터 혼합 전략을 적용합니다. MinkUNet은 효율적인 sparse convolution 연산과 계층적 특징 추출 능력으로 유명하며, 복잡한 LiDAR 데이터에서 유용한 피처를 학습할 수 있게 합니다. LaserMix는 포인트 클라우드를 경사 방향으로 결합하고, PolarMix는 방위각 방향으로 혼합하여 훈련 데이터의 다양성을 증가시킵니다.

- **Performance Highlights**: MixSeg3D는 실험을 통해 기존의 방법들보다 뛰어난 성능을 입증하였습니다. 2024 Waymo Open Dataset Challenge에서 평균 Intersection-over-Union (mIoU) 69.83%를 기록하며 높은 정확도를 보였습니다. 저자들은 TTA (Test Time Augmentation) 기법을 적용하여 평가 과정에서 예측 정확도를 더욱 향상시켰으며, 이는 자율주행에 있어 중요한 발전이 될 것입니다.



### Found in Translation: semantic approaches for enhancing AI interpretability in face verification (https://arxiv.org/abs/2501.05471)
- **What's New**: 본 논문은 컴퓨터 비전에서 특히 얼굴 인증(face verification) 분야의 머신 러닝 모델의 복잡성이 증가함에 따라, 해석 가능하고 투명한 인공지능을 위한 설명 가능한 인공지능(Explainable AI, XAI) 기술을 개발하는 데 중점을 두었습니다. 저자들은 이전 작업을 확장하여 인간의 인지 과정에서 파생된 의미론적 개념을 XAI 프레임워크에 통합하였습니다. 이를 통해 모델 출력과 인간 이해 간의 간극을 메우는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법론은 사용자 선택에 의한 얼굴 랜드마크로 정의된 의미론적 특성을 사용하여, 글로벌(global) 및 로컬(local) 설명을 결합하는 새로운 접근법입니다. 이 과정에서 대형 언어 모델(Large Language Models, LLMs)을 활용하여 유사성 지도(similarity maps)와 텍스트 설명을 생성합니다. 연구는 정량적인 실험과 사용자 피드백을 통해 방법론의 유효성을 검증하였으며, 향상된 해석 가능성을 확인하였습니다.

- **Performance Highlights**: 결과적으로 저자들의 의미론적 기반 접근법, 특히 가장 상세한 세트는 전통적인 방법에 비해 모델 결정의 더 섬세한 이해를 제공합니다. 사용자 연구에서는 전통적인 픽셀 기반 히트맵(pixel-based heatmaps)보다 저자들의 의미론적 설명에 대한 선호도가 강조되었으며, 이는 AI에서 인간 중심의 해석 가능성의 이점을 드러냅니다. 이 연구는 AI 모델의 행동을 인간의 인지 과정과 일치시키는 XAI 프레임워크 개발에 기여하여, 중요한 분야에서의 신뢰와 수용을 촉진하는 데 초점을 맞추고 있습니다.



### PySpatial: A High-Speed Whole Slide Image Pathomics Toolk (https://arxiv.org/abs/2501.06151)
- **What's New**: 이 논문에서는 Whole Slide Image (WSI) 분석을 위한 새로운 툴킷, PySpatial을 소개합니다. 기존의 CellProfiler와 같은 도구들이 중간 패치 단위로 기능을 추출하는 방식을 새롭게 개선하여, WSI 수준에서 직접 운영할 수 있는 방법을 제공합니다. 이를 통해 불필요한 처리 단계를 줄이고 기능 추출 과정을 간소화할 수 있습니다.

- **Technical Details**: PySpatial은 R-tree 기반의 공간 인덱싱과 행렬 기반 수치를 활용하여 계산적 리전을 효율적으로 매핑하고 처리합니다. 이 구조는 공간 상의 객체 간의 관계를 유지하면서 신속한 접근과 탐색을 가능하게 합니다. 기능 추출은 크기와 형태(Size & Shape), 텍스처(Texture), 강도(Intensity), 그리고 강도 분포(Intensity Distribution)의 네 가지 주요 범주로 나누어 진행됩니다.

- **Performance Highlights**: 실험 결과, PEC 데이터셋에서는 전통적인 CellProfiler 파이프라인과 비교하여 거의 10배의 속도 향상을 이뤘으며, KPMP 데이터셋에서는 대형 객체 처리 시 2배의 속도 향상이 확인되었습니다. 이는 PySpatial이 대규모 WSI 분석에서 효율성과 정확성을 향상시키는 잠재력을 가지고 있음을 강조해 줍니다.



### AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery (https://arxiv.org/abs/2501.06039)
Comments:
          23 pages, 5 figures

- **What's New**: 이 논문에서는 다양한 세포, 분자 및 조직 수준에서 작동하는 생물학적 조직을 위한 기반 모델 프레임워크인 Virtual Tissues (VirTues)를 제안합니다. VirTues는 새로운 토크나이제이션 방식과 고차원 다중 데이터에 대한 주의 메커니즘을 도입하여 해석 가능성과 성능을 동시에 향상시키고 있습니다. 이 모델은 고전적인 방법을 넘어 생물조직의 공간적 및 분자적 특성 분석을 위한 혁신적인 접근 방식을 제시합니다. 다양한 암 및 비암 조직 데이터세트에서 훈련된 VirTues는 특정 작업에 대한 추가 조정 없이 강력한 일반화 능력을 보이며, 새로운 생물학적 맥락에서 분석을 가능하게 합니다.

- **Technical Details**: VirTues의 핵심 혁신은 변형 체계의 주의 메커니즘을 공간적 및 마커 주의 컴포넌트로 분리하여 다양한 마커 조합에 대해 유연한 처리 능력을 제공함으로써 고차원 데이터를 효과적으로 처리하는 것입니다. 이 모델은 단백질 언어 모델(Protein Language Model, PLM)을 활용하여 단백질 마커 간의 복잡한 관계를 포착하고, 세포, 틈새 및 조직 수준에서의 생물학적 계층을 존중합니다. 또한, VirTues는 마스킹된(marker-space) 마커 데이터를 복원하는 비지도 훈련을 통해 작동하는 마스크 자동 인코더(Masked Autoencoder, MAE) 구조를 채택하였습니다.

- **Performance Highlights**: VirTues는 임상 진단, 생물학적 발견 및 환자 사례 검색 작업에서 기존 방법들을 초월하는 성능을 보이며, 다양한 데이터에 대한 robust한 일반화 능력을 입증되었습니다. 이 모델은 아네(AST)와 같은 다양한 실험적 조건에서도 이질적인 데이터세트를 통합할 수 있는 능력을 갖추고 있어, 임상 응용 프로그램 및 질병 메커니즘에 대한 통찰을 제공합니다. VirTues의 향상된 성능과 해석 가능성은 생물학적 데이터 분석의 새로운 가능성을 열어줄 것으로 기대됩니다.



### An Attention-Guided Deep Learning Approach for Classifying 39 Skin Lesion Types (https://arxiv.org/abs/2501.05991)
Comments:
          26 pages

- **What's New**: 이 연구는 39종의 피부 병변을 포함하는 포괄적이고 다양한 데이터셋을 정리하여 딥러닝 기반의 분류 시스템을 개발하는 데 중점을 두고 있습니다. 기존 연구들은 보통 제한된 병변 유형에 집중해 왔지만, 이 연구는 Advanced attention-guided techniques를 도입하여 더 많은 종류의 피부 병변 분류의 정확성을 높이고자 합니다.

- **Technical Details**: 연구에서는 MobileNetV2, Xception, InceptionV3, EfficientNetB1, Vision Transformer와 같은 최첨단 딥러닝 모델을 사용하여 성능을 평가합니다. 효율적인 정확성을 높이기 위해 Efficient Channel Attention (ECA) 및 Convolutional Block Attention Module (CBAM)과 같은 어텐션 메커니즘이 모델 아키텍처에 통합됩니다.

- **Performance Highlights**: Vision Transformer 모델은 CBAM과의 통합을 통해 93.46%의 정확도, 94%의 정밀도, 93%의 재현율 및 93%의 F1-score를 기록하여 다른 모델들보다 우수한 성능을 보여줍니다. 이러한 결과는 피부 병변 진단을 위한 정확하고 효율적인 예후 도구로서 제안된 시스템의 중요한 잠재력을 강조합니다.



### Reusable specimen-level inference in computational pathology (https://arxiv.org/abs/2501.05945)
- **What's New**: 이번 논문에서는 SpinPath라는 새로운 도구 키트를 소개합니다. SpinPath는 미리 훈련된 샘플 수준 모델, Python 기반 추론 엔진 및 JavaScript 기반 추론 플랫폼을 제공하여 연구자들이 샘플 수준의 딥 러닝을 보다 쉽게 접근할 수 있도록 돕습니다. 이를 통해 연구자들은 샘플 수준의 깊은 학습을 통해 메타스타시스 감지 등의 작업에서 실질적인 기여를 할 수 있습니다.

- **Technical Details**: SpinPath는 사용자가 미리 훈련된 모델을 제공하면 샘플 수준의 추론을 수행하도록 설계 되어 있습니다. 사용자 지정 기능 추출기와 이미지 분석 모델을 선택할 수 있는 JavaScript 도구를 사용하여 노코드 환경에서 간편하게 사용할 수 있습니다. SpinPath는 TiffSlide와 PyTorch를 사용하여 딥 러닝 추론을 실행하며, 다양한 슬라이드 이미지를 로딩하고 처리하는 프로세스를 간소화합니다.

- **Performance Highlights**: SpinPath를 사용하여 SLN-Breast 데이터셋에 대한 미리 훈련된 메타스타시스 탐지 모델의 성능을 비교한 결과, UNI 모델이 0.975의 balanced accuracy(B.A.)를 기록하며 가장 높은 성능을 보였습니다. 또한, SpinPath는 이러한 모델들의 추론 속도 또한 측정하여, 메타스타시스 분류에서 높은 성능을 가진 UNI 모델은 평균 63초의 시간 소요를 기록했습니다. 이는 SpinPath가 실제 연구에서 신속하고 효율적인 실험 및 재현성을 지원할 수 있음을 보여줍니다.



### AI-Driven Diabetic Retinopathy Screening: Multicentric Validation of AIDRSS in India (https://arxiv.org/abs/2501.05826)
Comments:
          22 pages, 5 figures. arXiv admin note: substantial text overlap with arXiv:1812.07105 by other authors without attribution

- **What's New**: 본 논문은 인공지능 기반의 당뇨병성 망막병증 선별 시스템(AIDRSS)의 효과를 평가하여 인도와 같은 자원이 제한된 환경에서 자동화된 선별 솔루션의 필요성을 충족하는 방법을 제시합니다. AIDRSS는 5,029명의 참가자와 10,058개의 망막 이미지를 분석하여 당뇨병성 망막병증의 유병률을 평가하고, 데이터 품질을 향상시키기 위해 CLAHE(Contrast Limited Adaptive Histogram Equalization) 전처리 기법을 통합하였습니다.

- **Technical Details**: AIDRSS는 약 5천만 개의 학습 가능한 파라미터와 250개의 층으로 구성된 깊은 신경망(deep learning architecture)을 이용하여 망막 이미지를 분석합니다. 이 시스템은 전문가의 평가와 비교하여 92%의 민감도(sensitivity)와 88%의 특이도(specificity)를 유지하며, 특별히 참조가 필요한 DR을 100% 정확도로 탐지할 수 있음을 보여줍니다. 또한, 이미지 전처리를 위해 도입된 CLAHE는 지역별 대비를 향상시키고 이미지 품질 개선에 기여합니다.

- **Performance Highlights**: 연구 결과, 일반 인구의 당뇨병성 망막병증 유병률은 13.7%, 고혈당 환자의 경우 38.2%로 나타났습니다. AIDRSS의 뛰어난 성능은 다양한 인구에서 당뇨병성 망막병증을 정확히 식별하고 등급화하는데 효과적임을 입증합니다. 인공지능 기술을 통해 이 시스템은 자원이 부족한 환경에서도 신뢰성 있는 조기 진단 솔루션을 제공하여 당뇨병으로 인한 시각 손실의 부담을 경감할 수 있는 잠재력을 지니고 있습니다.



### Alignment without Over-optimization: Training-Free Solution for Diffusion Models (https://arxiv.org/abs/2501.05803)
- **What's New**: 이 연구에서는 Diffusion model의 목표를 정렬하는 데 어려움이 있었던 점을 해결하기 위해, Sequential Monte Carlo (SMC) 기반의 새로운 훈련 없는 샘플링 방법인 Diffusion Alignment as Sampling (DAS)를 제안합니다. DAS는 모델의 일반성을 유지하면서도 효과적인 보상 정렬을 달성합니다. 기존 방법들이 보상 최적화의 문제로 인해 성능이 저하되는 것을 방지하면서, 목표 보상을 효과적으로 샘플링할 수 있도록 설계되었습니다.

- **Technical Details**: DAS는 다수의 후보 latent를 활용하여 높은 보상 샘플로 유도하는 방식으로 구성되어 있습니다. 이를 통해 샘플링에서의 오류를 평균화하고, 보상 정렬된 목표 분포에서의 샘플링을 가능하게 합니다. 이 과정에서는 온도 조정(tempering) 기법을 사용하여 중간 목표 분포를 신중하게 설계함으로써 샘플링 효율성을 극대화하고 있습니다.

- **Performance Highlights**: DAS는 Stable Diffusion v1.5 및 CLIPScore와 같은 더 복잡한 보상 함수에 적용되며, 기존의 미세 조정 방법에 비해 뛰어난 성능을 입증하고 있습니다. 결과적으로, DAS는 단일 보상 최적화 및 다중 목표 최적화에서 새로운 Pareto front를 달성하였으며, 온라인 환경에서도 뛰어난 샘플링 능력을 보여주어 기존 방법들보다 20% 이상 개선된 성과를 달성했습니다.



### Cryptanalysis of Cancelable Biometrics Vau (https://arxiv.org/abs/2501.05786)
Comments:
          17 pages, 4 figures

- **What's New**: 이번 논문에서는 접을 수 있는 생체 인식 데이터 기반의 키 바인딩 스킴인 Cancelable Biometrics Vault (CBV)의 보안 분석을 진행합니다. 연구팀은 CBV 프레임워크를 구현하는 BioEncoding 스킴의 취약성을 파악하고, 템플릿의 가역성(reversibility)과 연결 가능성(linkability)을 문제삼습니다. 이 연구는 CBV 스킴의 취소 가능성과 링크 가능성 취약점을 발견하여 기존 생체 기반 키 바인딩 스킴과의 차별점을 제공합니다.

- **Technical Details**: 접을 수 있는 생체 인식의 주요 특징은 생체 인식 데이터를 사용하여 개인화된 토큰과 함께 보안 템플릿을 생성하는 것입니다. 논문에서는 생체 템플릿의 보안을 높이기 위해 제안된 다양한 변환 스킴에 대해 다룹니다. 특히, 이 논문에서 제안된 공격은 같은 개인으로부터 생성된 템플릿의 연결 가능성을 활용하여 키를 복구하는 방법입니다.

- **Performance Highlights**: 연구팀은 CBV 스킴을 분석하면서 기존의 생체 인식 키 바인딩 스킴에서 발생하지 않았던 새로운 공격 사례를 제시합니다. 또한, 이 논문에서는 실험을 통해 제안된 공격이 실제로 키 복구를 얼마나 효과적으로 수행할 수 있는지를 보여줍니다. 이러한 공격은 템플릿의 연결 가능성을 이용해 키 공간의 크기를 크게 줄여 효율성을 높입니다.



### Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models (https://arxiv.org/abs/2501.05767)
Comments:
          20 pages, 8 figures

- **What's New**: 이 논문에서는 복합적인 다중 이미지 시나리오에서 정밀한 그라운딩(grounding)을 달성하는 데 어려움을 겪고 있는 최근의 멀티모달 대형 언어 모델(MLLMs)의 발전을 다루고 있습니다. 새로운 Chain-of-Thought (CoT) 프레임워크를 통하여 단일 이미지 그라운딩과 다중 이미지 이해를 통합하며, Migician 모델을 소개하여 자유형식(free-form) 다중 이미지 그라운딩을 수행할 수 있는 최초의 모델로 자리잡았습니다.

- **Technical Details**: Migician 모델은 MGrounding-630k이라는 대규모 데이터셋으로 훈련되어, 다양한 다중 이미지 그라운딩 작업을 위한 데이터와 자유형식 지침 데이터를 포함합니다. 또한, 모델의 성능을 평가하기 위한 MIG-Bench라는 종합 벤치마크를 제시하여 다중 이미지 그라운딩의 성능을 정량적으로 측정하고 있습니다. 이를 통해 Migician은 이전의 MLLMs보다 21.61% 우수한 다중 이미지 그라운딩 능력을 입증하였습니다.

- **Performance Highlights**: Migician 모델은 실험 결과, 현존하는 최고의 MLLMs보다 훨씬 나은 성능을 기록하였으며, 특히 다양한 환경에서의 비전-언어 작업을 수행하는 데서 탁월한 능력을 보여주고 있습니다. 최종적으로, 이 연구는 MLLMs의 잠재력과 극복해야 할 도전을 탐구하고, Migician과 MGrounding-630k, MIG-Bench의 개발을 통해 다중 이미지 그라운딩 분야의 새로운 기준을 설정하고자 하였습니다.



### Semantic Mapping in Indoor Embodied AI -- A Comprehensive Survey and Future Directions (https://arxiv.org/abs/2501.05750)
- **What's New**: 본 논문은 내부 내비게이션을 위한 지능형 체화 에이전트의 의미 맵 구축 접근 방식에 대한 종합적인 리뷰를 제공합니다. 기존의 연구는 일반적 발전이나 특정 작업에 초점을 맞추었으나, 이 연구는 의미 맵 구축 방법의 구조적 표현 및 정보 유형에 따라 분류하는 새로운 시각을 제시합니다. 또한, 저자는 추가 연구 방향을 제안하며, 앞으로의 트렌드를 조망합니다.

- **Technical Details**: 논문은 의미 맵 구축 방식을 공간 그리드(spatial grids), 위상 그래프(topological graphs), 조밀한 포인트 클라우드(dense point-clouds), 하이브리드 맵(hybrid maps)으로 분류합니다. 각각의 접근 방식은 암묵적(implicit) 또는 명시적(explicit) 환경 데이터(데이터)의 두 가지 정보 유형을 인코딩합니다. 또한, SLAM(Simultaneous Localization and Mapping) 기술을 활용하여 미지의 환경을 매핑하고 에이전트의 위치를 추정하는 문제에 대해 논의합니다.

- **Performance Highlights**: 저자들은 논문에서 효과적인 의미 맵 구축 기술이 로봇이 복잡한 작업을 자율적으로 수행하는 데 필수적이라는 점을 강조합니다. 특히, 내부 환경에서의 강력한 성능을 통해 로봇이 사고에 대비하여 효과적으로 의사 결정을 내릴 수 있음을 시사합니다. 또한, 향후 연구가 개방어휘(open-vocabulary) 및 쿼리 가능한 지도 표현을 개발하는 방향으로 나아가고 있음을 제안하며, 이에 따른 메모리 요구사항과 계산 효율성 문제는 여전히 해결해야 할 과제로 남아 있습니다.



### Bit-depth color recovery via off-the-shelf super-resolution models (https://arxiv.org/abs/2501.05611)
- **What's New**: 이 논문은 고비트 깊이 색상 복원을 위한 새로운 접근법을 제안합니다. 기존의 방법들이 단일 규모(feature scale)에만 의존하는 것과는 달리, 우리의 방법은 super-resolution(SR) 아키텍처를 활용하여 이미지에서 세부 정보를 효과적으로 추출합니다. 이를 통해 복잡한 텍스처와 미세한 패턴을 복원할 수 있는 능력을 강화했습니다.

- **Technical Details**: 제안된 아키텍처는 두 가지 주요 구성 요소로 이루어져 있습니다: SR 기반 다중 스케일 특징 추출기와 Attention-augmented bit-plane recovery network입니다. 이 시스템은 미리 훈련된 SR 특징 추출기를 활용하여 이미지의 다중 스케일 정보를 결합하고, 최종적으로 각 비트 플레인의 정보를 점진적으로 복원합니다. 이에 따라 정확한 비트 깊이 복원이 가능해집니다.

- **Performance Highlights**: 벤치마크 데이터셋에서 수행한 실험 결과, 제안된 방법은 PSNR 및 SSIM 지표에서 전통적 방법과 딥러닝 기반 기술들을 능가함을 확인했습니다. 이는 고충실도의 색 복원 가능성을 강조하며, 새로운 방법이 기존의 한계를 극복하는데 기여할 수 있음을 시사합니다.



### EndoDINO: A Foundation Model for GI Endoscopy (https://arxiv.org/abs/2501.05488)
- **What's New**: 이 논문에서는 GI 내시경 작업을 위한 기본 모델인 EndoDINO를 제시합니다. EndoDINO는 문헌에서 가장 큰 GI 내시경 비디오 데이터셋에서 샘플링된 잘 구성된 이미지 데이터셋으로 사전 훈련을 통해 강력한 일반화 능력을 달성합니다. 우리는 100K에서 10M 이미지까지 다양한 데이터셋을 사용하여 1B, 307M, 86M 매개변수를 가진 ViT 모델을 사전 훈련했습니다.

- **Technical Details**: EndoDINO는 고유한 중첩 구성을 통해 anatomy landmark classification, polyp segmentation 및 ulcerative colitis에 대한 Mayo endoscopic scoring에서 최첨단 성능을 달성했습니다. DINOv2의 방법론을 기반으로 사전 훈련과 평가를 위한 다양한 하이퍼파라미터를 탐색했으며, 8개의 NVIDIA H100 GPU 클러스터에서 최대 625,000회의 반복으로 사전 훈련을 진행했습니다.

- **Performance Highlights**: EndoDINO는 다양한 GI 내시경 작업에서의 일반화 능력을 검증하기 위해 평가 데이터를 사용하였으며, 기존 작업들에 비해 뛰어난 성능을 나타냈습니다. 특히, anatomical landmark classification, polyp segmentation 및 Mayo endoscopic scoring을 포함한 평가 작업에서 매우 긍정적인 결과를 보였습니다.



### Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models (https://arxiv.org/abs/2501.05478)
- **What's New**: 이 연구는 로봇 공학에서 비전-언어 내비게이션(VLN) 분야에 아랍어 통합을 첫 번째로 선보이며, 다국어 Small Language Models(SLMs)와 아랍어 중심의 LLM인 Jais의 성능을 평가합니다. 분명히 부족했던 아랍어 데이터에 대한 연구의 공백을 메우면서, NavGPT 프레임워크와 R2R 데이터셋을 사용하여 아랍어와 영어 간의 의사소통이 내비게이션 추론에 미치는 영향을 평가합니다. 이를 통해, 아랍어로 지시를 받았을 때의 로봇 내비게이션 작업의 계획 및 추론 능력을 강조하였습니다.

- **Technical Details**: 본 연구는 OpenAI의 GPT-4o mini, Meta의 Llama 3 8B와 Microsoft의 Phi-3 medium 14B와 같은 최신 다국어 SLM들과 Jais 30B LLM을 NavGPT 프레임워크 내에서 비교합니다. R2R 데이터셋을 활용하여 영어 내비게이션 지시를 아랍어로 변환한 데이터셋으로 보면, 다양한 언어로 내비게이션 자원에 접근하고자 하는 양방향적 연구의 필요성을 강조합니다. 또한, 제로샷 방식으로 작업을 예측하며, 언어의 영향력에 대한 분석을 도모합니다.

- **Performance Highlights**: 실험 결과, NavGPT 프레임워크가 영어 및 아랍어 지시를 통해 높은 수준의 내비게이션 계획을 수행할 수 있음을 입증하였습니다. 그러나 일부 모델은 아랍어에서 추론 및 계획에 어려움을 겪어 언어 모델의 성능과 한계를 드러냈습니다. 이러한 발견은 아랍어 모델의 발전과 현실 세계 응용 프로그램에서의 가능성을 열어주며, 연구의 향후 방향으로 언어 모델의 계획 및 추론 능력 향상이 필요함을 강조합니다.



### Beyond Questionnaires: Video Analysis for Social Anxiety Detection (https://arxiv.org/abs/2501.05461)
- **What's New**: 이번 연구는 비디오 분석을 통해 사회 불안 장애(SAD)의 조기 감지를 위한 새로운 접근 방식을 제시합니다. 전통적인 SAD 감지 방법인 의사와의 상담 및 자가 보고 방식의 질문지는 시간 소모와 편향과 같은 한계가 있습니다. 우리는 다양한 신체 특성을 통해 SAD를 탐지하는 방법을 개발하여 비디오 데이터를 활용하였습니다.

- **Technical Details**: 연구에서는 92명의 참가자가 통제된 환경에서 즉흥적 연설을 수행하는 동안 비디오 데이터를 수집하였으며, 이를 통해 참가자들의 머리, 몸, 시선, 얼굴 근육의 행동 변화를 분석했습니다. 신체 자세와 얼굴 표현과 같은 다양한 신체적 지표를 사용하여 SAD와 비 SAD 참가자를 분류하기 위해 기계 학습(machine learning) 및 심층 학습(deep learning) 알고리즘을 적용하였으며, 74%의 정확도를 달성했습니다.

- **Performance Highlights**: 비디오 기반 SAD 탐지는 비침습적이며 확장 가능한 접근 방식을 제공하여 실시간 적용이 가능합니다. 이는 SAD 조기 감지 및介입 능력을 향상시킬 수 있는 가능성을 지닌 혁신적인 접근 방식입니다. 논문은 또한 공개 데이터셋을 제공하며, 이는 후속 연구를 위해 다양한 신체 특성을 포함합니다.



### Efficiently serving large multimedia models using EPD Disaggregation (https://arxiv.org/abs/2501.05460)
Comments:
          13 pages, 6 figures

- **What's New**: 본 연구는 Encode-Prefill-Decode (EPD) Disaggregation이라는 새로운 프레임워크를 제안하여 대규모 멀티모달 모델 (LMM)에서 인코딩, 프리필, 디코딩 단계를 분리한다. 이 접근 방식은 메모리 병목 현상을 완화하고, 동기화 지연을 줄이며, 유연한 배치 처리를 지원한다. 또한, 멀티모달 토큰을 위한 새로운 캐싱 메커니즘을 도입하여 비동기 전송을 가능하게 하였다.

- **Technical Details**: EPD 분해는 각 단계를 고유한 리소스에 할당하여 독립적으로 최적화가 가능하도록 하며, 메모리 효율성을 크게 향상시킨다. 이를 통해 배치 크기 및 이미지 처리 수치가 증가하고, 최적 성능 메트릭을 달성하기 위한 통합 모듈도 포함한다. 연구 결과는 다양한 LMM 모델에서 메모리 사용량이 최대 15배 감소하고, 배치 크기가 최대 22배 증가하는 효과를 보여준다.

- **Performance Highlights**: 실험 결과, EPD 분해 방식이 기존 시스템에 비해 종합 성과를 개선하는 데 기여하여, 전체 처리량(E2ETP)이 최대 57% 향상되고, 첫 번째 토큰 수신까지의 지연(TTFT)이 최대 71% 감소했다. 이러한 결과는 적은 자원으로도 고성능의 멀티모달 추론이 가능함을 보여주며, EPD의 잠재력을 강조한다.



### Atlas: A Novel Pathology Foundation Model by Mayo Clinic, Charit\'e, and Aignostics (https://arxiv.org/abs/2501.05409)
- **What's New**: 이 보고서에서는 RudolfV 접근법을 기반으로 한 새로운 비전 파운데이션 모델인 Atlas를 소개합니다. Atlas는 Mayo Clinic과 Charité - Universitätsmedizin Berlin에서 수집한 120만 개의 조직병리학 전체 슬라이드 영상으로 훈련되었으며, 다양한 공공 벤치마크 데이터셋에서 최고의 성능을 달성했습니다. 이 모델은 매개변수 수나 훈련 데이터셋 크기 면에서 가장 크지 않지만 여전히 뛰어난 성능을 보여줍니다.

- **Technical Details**: Atlas 모델은 ViT-H/14 아키텍처(632백만 매개변수)를 사용하여 훈련되며, 490,000건의 사례에서 추출된 34억 개의 이미지 타일로 훈련되었습니다. 데이터는 다양한 해상도에서 추출되었으며, 각기 다른 염색법과 다중 배율이 포함되어 있습니다. 이러한 다양성은 AI 학습의 일반화 및 강건성 향상에 기여합니다.

- **Performance Highlights**: Atlas는 21개의 공공 벤치마크 데이터셋에서 성능을 평가하여, 전통적인 모델에 비해 우수한 결과를 기록했습니다. 우리는 모델 성능 평가를 위해 선형 프로빙 프로토콜을 사용하였으며, 모든 모델의 추출된 임베딩을 비교했습니다. 결과적으로, Atlas는 다양한 다운스트림 병리학 작업에서 탁월한 성과를 달성하며, 실제 임상 환경에서도 활용될 수 있는 잠재력을 지니고 있습니다.



### ResPanDiff: Diffusion Model for Pansharpening by Inferring Residual Inferenc (https://arxiv.org/abs/2501.05091)
- **What's New**: 이 논문에서는 다중 출처 이미지 융합의 효율성을 개선하기 위해 ResPanDiff라는 새로운 확산(diffusion) 모델을 제안합니다. 기존의 기술들은 샘플링 속도가 느리고, 성능을 희생할 때가 많습니다. 그러나 ResPanDiff는 노이즈 잔여값에서 LRMS와 HRMS 이미지 사이의 잔여값으로 전이되는 마르코프 체인을 통해 샘플링 단계를 줄이고 성능을 높입니다. 이 모델은 단 15단계의 샘플링으로 최첨단 성능을 달성하며, 현존하는 방법보다 월등한 성능을 제공합니다.

- **Technical Details**: ResPanDiff는 잔여 생성(residual generation) 개선을 위해 잠재 공간(latent space), 얕은 조건 주입(Shallow Cond-Injection, SC-I), 손실 함수(loss function)를 설계하였습니다. 이 구조는 모델이 더 많은 특성을 추출하도록 돕고, 인코딩 단계에서 더 나은 성능을 발휘할 수 있도록 합니다. 또한, 기존의 모델과 달리 LRMS와 PAN의 정보를 효과적으로 추출하면서도 잔여 값에서 시작하여 효율적인 샘플링을 가능하게 합니다. 이는 대조적 가우시안 백색소음에서 시작하는 전통적인 방식과는 다른 접근법입니다.

- **Performance Highlights**: 실험 결과, ResPanDiff는 세 가지 널리 사용되는 pansharpening 데이터셋에서 최첨단(SOTA) 성능을 달성하였습니다. 전통적인 방법에 비해 $90\%$ 이상의 샘플링 단계를 줄이는 동시에 높은 이미지 품질을 유지합니다. 또한, 복잡한 논의와 ablation study를 통해 제안된 방법의 효과성을 추가로 입증합니다. 이는 저해상도 다중 스펙트럼 이미지와 고해상도 팬크로매틱 이미지의 융합에서 놀라운 성과를 보여줍니다.



### Comprehensive Examination of Unrolled Networks for Solving Linear Inverse Problems (https://arxiv.org/abs/2501.04608)
Comments:
          27 pages, 10 figures. Project Page: this https URL

- **What's New**: 이번 논문은 여러 컴퓨터 비전 및 이미징 과제에서 발생하는 Unrolled Networks의 디자인 선택을 통합하고 최적화하는 방법을 제안합니다. 사용자들이 마주치는 여러 디자인 선택을 줄이는 것을 목표로 하며, 그 과정에서 발생하는 각 선택의 영향을 조명한 포괄적인 ablation study를 보고합니다. 이를 통해 연구자들이 자신의 응용 프로그램에 맞는 Unrolled Networks를 설계하는 데 도움이 되고 문제 진단을 효율적으로 수행할 수 있도록 돕고자 합니다.

- **Technical Details**: Unrolled Networks는 MRI, CT 스캔, 지진 이미징 등 다양한 이미징 응용 프로그램에서 선형 역 문제를 해결하기 위해 설계되었습니다. 이 네트워크는 이미지 복원을 위해 projection gradient descent(PGD) 알고리즘을 활용하며, neural networks를 통해 매 반복마다 프로젝션 단계를 실행합니다. PGD 알고리즘은 정확한 projection operator를 요구하지만, Unrolled Networks는 neural networks를 사용하여 보다 유연하게 학습할 수 있는 구조로 설계됩니다.

- **Performance Highlights**: 이 연구의 결과는 Unrolled Networks가 이미지 복원 성능을 개선할 수 있는 잠재력을 나타냅니다. 또한 기존 알고리즘보다 더 나은 실제 성능을 가능하게 하여 복잡한 네트워크 문제를 해결할 수 있습니다. 단순한 설계 변경 및 학습을 통해 다양한 적용 가능성과 효율성을 보여주며, 이러한 접근법은 향후 연구에서 널리 활용될 것으로 기대됩니다.



New uploads on arXiv(cs.AI)

### Supervision policies can shape long-term risk management in general-purpose AI models (https://arxiv.org/abs/2501.06137)
Comments:
          24 pages, 14 figures

- **What's New**: 본 논문은 일반 목적의 AI(GPAI) 모델, 특히 대형 언어 모델(LLMs)의 확산과 배포가 AI 감독 기관에 해마다 증가하는 위험과 사건 보고의 생태계를 초래하고 있다고 가정합니다. 이러한 환경에서 감독 기관은 새로운 보고 메커니즘을 이해하고 효과적으로 관리하는 도전 과제를 직면했으며, 이 연구에서는 우선순위 기반 및 다양성 우선 정책이 전문가의 식별된 높은 영향 위험을 완화하는 데 더 효과적이라는 점이 강조됩니다.

- **Technical Details**: 연구진은 다양한 위험 및 사건 보고 생태계에서 파생된 특성들을 파라미터화하여 시뮬레이션 프레임워크를 개발했습니다. 이 시뮬레이션은 비우선, 무작위 선택, 우선 순위 기반 및 다양성 우선 정책을 평가하고, 위험 및 사건 보고의 처리에서 정책들의 효과를 분석합니다. 정책 간의 피드백 루프를 포함시켜, 전문가 주도의 보고서 우선 선택이 커뮤니티의 입력을 저해할 수 있는 방법도 탐구합니다.

- **Performance Highlights**: 모델에서의 실제 데이터셋을 통해 시뮬레이션 결과를 검증하였고, 150,000건 이상의 위험한 대화가 포함된 ChatGPT와의 상호작용을 분석했습니다. 연구 결과에 따르면 다양한 감독 정책이 AI 안전 결과에 미치는 영향을 실질적으로 변화시킬 수 있으며, 이러한 선택이 GPAI 모델의 거시적 위험 생태계를 형성하는 방법을 강조합니다.



### All AI Models are Wrong, but Some are Optima (https://arxiv.org/abs/2501.06086)
- **What's New**: 본 논문은 전통적인 Predictive AI 모델의 한계를 넘어서, 의사결정 성과를 극대화하는 방향으로 모델링할 수 있는 'decision-oriented' predictive AI 모델에 대한 개념을 도입합니다. 기존의 모델들은 데이터에 최적화되어 있지만, 실제 의사결정 성과와의 직접적인 관계가 부족했습니다. 저자들은 이러한 모델의 최적 조건을 수립하고, 확률적 및 결정적 시스템에서의 의사결정 성과를 개선할 수 있는 방법을 제안합니다.

- **Technical Details**: 우리는 Sequential Decision-Making(SDM)의 문맥에서 Markov Decision Process(MDP)를 활용하여 특정 성과 목표를 달성하는 데 필요한 예측 모델의 필요충분조건을 수립했습니다. MDP는 상태와 행동의 쌍을 통해 최적의 정책을 찾는 구조입니다. 또한, 기존의 Predictive AI 모델이 적합한 데이터에 맞춰져 있는 경우라도 최적의 성과를 보장하지 않는다는 점을 명확히 하였습니다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 제안된 의사결정 중심의 예측 모델들이 성과를 개선하는 데 효과적임을 보여주었습니다. 특히, 의사결정 목표를 모델에 통합함으로써 예측 성능을 일부 희생하면서도 최적의 결정 성과를 달성할 수 있다는 사실이 입증되었습니다. 결론적으로, 이 연구는 추후 AI 기반의 의사결정 시스템 개발에 중요한 기초 자료가 될 것입니다.



### Solving nonograms using Neural Networks (https://arxiv.org/abs/2501.05882)
- **What's New**: 이번 연구에서는 Nonogram(논리 퍼즐)의 해결을 위해 Heuristic Algorithm, Genetic Algorithm, 신경망을 이용한 Heuristic Algorithm을 분석했습니다. 특히, 신경망을 훈련시키기 위한 공개 데이터셋을 생성하고 이를 기반으로 알고리즘 코드를 발표하였습니다. Heuristic Algorithm과 Neural Network의 결합이 가장 높은 성과를 냈습니다. 기존의 연구에서는 Neural Network를 이용한 Nonogram 해결 방법이나 알고리즘의 결합이 없었던 점에서 혁신적인 접근이라고 할 수 있습니다.

- **Technical Details**: Nonogram의 본질은 주어진 숫자를 따라 그리드 내의 셀을 색칠하거나 비워 두는 논리 퍼즐입니다. 이 연구에서는 NP 완전 문제로 분류되는 Nonogram을 풀기 위해 다양한 알고리즘을 활용하며, 각 셀은 채워짐과 비워짐 두 가지 상태를 가질 수 있습니다. 또한, 주어진 셀에 따라 이 문제는 여러 개의 해답을 가질 수 있음을 설명하고, 이를 해결하기 위한 과정에서 각 보드의 가능한 값의 공간을 명확히 정의하였습니다.

- **Performance Highlights**: 연구 결과, Heuristic Algorithm과 신경망의 조합은 Nonogram 문제 해결에서 최상의 성과를 기록했습니다. 특정 보드는 다수의 해가 있을 수 있으며, 이로 인해 모든 가능한 해결 방안을 탐색하는 것이 중요합니다. 이 알고리즘은 Nonogram의 정의역에서 불필요한 조건을 제거하며 최적의 변수 조합을 통해 성능을 개선했습니다.



### Annealing Machine-assisted Learning of Graph Neural Network for Combinatorial Optimization (https://arxiv.org/abs/2501.05845)
Comments:
          Second Workshop on Machine Learning with New Compute Paradigms at NeurIPS 2024 (MLNCP 2024)

- **What's New**: 이 논문에서는 Annealing Machines (AM)와 Graph Neural Networks (GNN)을 통합하여 조합 최적화 문제 해결의 효율성을 높이는 방법을 제안합니다. AM은 현재의 기술 수준에서 퀀텀 기술을 대체할 조합 문제 해결에서 경쟁력을 가지지만, 확장성의 한계를 가지고 있습니다. 반면, GNN은 확장성이 뛰어나지만 결과의 정확성이 떨어질 수 있습니다. 이를 바탕으로 두 기술의 장점을 결합한 새로운 프레임워크를 설계하였습니다.

- **Technical Details**: 제안된 프레임워크는 조합 문제를 간단히 압축하는 단계와 AM이 생성한 부분 해결책을 바탕으로 GNN을 개선하는 단계로 구성됩니다. 이 과정에서 AM은 GNN에 지식을 주입하여 간접적으로 문제를 해결하도록 돕습니다. 다양한 그래프 패밀리에 대해 실시한 실험에서, 제안한 모델이 AM의 초기 한계를 넘어서는 문제들을 해결할 수 있음을 보여주었습니다. 또한, GNN의 지역 특징을 종합하여 최종 GNN 기반 해결기를 초기화하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 AM이 더 큰 문제를 해결할 수 있도록 하여 전체 수렴 시간을 단축시켰습니다. 특정 경우에서는 해결 품질이 향상되기도 하여, 두 기술 간의 상호작용에서 발생하는 시너지 효과를 실증적으로 입증하였습니다. 따라서 AM과 GNN의 결합이 조합 최적화 문제의 해결을 위한 효과적인 경로임을 보여주고 있습니다.



### Understanding Impact of Human Feedback via Influence Functions (https://arxiv.org/abs/2501.05790)
Comments:
          Source code: this https URL

- **What's New**: 이번 논문에서는 Human Feedback에서의 강화 학습(RLHF) 과정에서 발생할 수 있는 인간 피드백의 불확실성과 편향을 해결하기 위해 영향 함수(influence functions)를 활용하는 새로운 접근 방식을 제안합니다. 이 연구는 대규모 언어 모델(LLMs)과 함께 사용될 수 있는 계산 효율적인 근사 방법을 통해, 인간 피드백이 보상 모델(reward model)에 미치는 영향을 정량화하고자 합니다.

- **Technical Details**: 영향 함수는 훈련 데이터 포인트가 모델 매개변수에 미치는 영향을 정량화합니다. 본 연구에서는 벡터 압축 기법을 활용하여 LLM 기반 보상 모델에 대한 영향 함수 계산의 속도를 2.5배 개선하는 방법을 소개합니다. 이를 통해 대규모 데이터셋에서의 적용을 용이하게 하고, 실험을 통해 인간 피드백 데이터셋에서 라벨러 편향(labeler bias)을 탐지하는 두 가지 주요 응용 프로그램을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 라벨러 편향을 효과적으로 식별함으로써 기존의 여러 기준 모델을 초과한 성능을 보여줍니다. 또한, 비전문가 라벨러가 전문가 피드백과 더 잘 일치하도록 피드백 전략을 개선하는 데에도 기여할 수 있음을 입증했습니다. 이를 통해 인간 피드백의 해석 가능성을 높이고, 복잡한 작업에 대해 더 정확한 피드백을 제공할 수 있는 기반을 마련하고자 합니다.



### Deontic Temporal Logic for Formal Verification of AI Ethics (https://arxiv.org/abs/2501.05765)
- **What's New**: 이 연구는 인공지능(AI) 시스템의 윤리적 행동을 정의하고 평가하기 위해 의무론(logic of obligation) 기반의 형식화(formalization)를 제안합니다. 형식화는 시스템 수준의 명세(specification)에 중점을 두고, 공정성(fairness) 및 설명 가능성(explainability)과 관련된 윤리적 요구사항을 포착하는 공리(axioms)와 정리(theorems)를 소개합니다.

- **Technical Details**: 본 연구에서의 기본 모델은 AI 윤리를 위해 명제 논리와 의무론을 결합하여 테이블을 통해 윤리적 요구사항을 정의합니다. 여기에 시공간 연산자(temporal operators)를 포함하여 AI 시스템의 윤리적 행동이 시간에 따라 변화하는 방식에 대한 복잡한 요구사항을 표현할 수 있게 하며, '항상(always)', '결국(eventually)', '직전까지(until)'와 같은 연산자가 사용됩니다.

- **Performance Highlights**: 이 연구는 COMPAS와 대출 예측 AI 시스템의 윤리성을 평가하여 제안된 형식화가 실제 세계의 AI 응용 프로그램에서 윤리적 문제를 식별하는 데 효과적임을 입증합니다. 실험 결과에 따르면, 두 시스템 모두 공정성과 비차별(non-discrimination)과 관련된 주요 윤리적 특성을 충족하지 못한다는 결과를 보여 주었습니다.



### Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models (https://arxiv.org/abs/2501.05752)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 다단계 추론(multi-step reasoning) 과제에서 탁월한 가능성을 보여주었습니다. 그러나 기존 방법은 계산 효율성과 중복성에서 문제를 드러내고 있으며, 당면 과제의 난이도에 따른 다양성을 간과하고 있습니다. 이러한 한계를 해결하기 위해 우리는 Semantic Exploration with Adaptive Gating (SEAG)이라는 새로운 방법론을 제안합니다.

- **Technical Details**: SEAG는 응답의 신뢰도(confidence level)를 기반으로 트리 탐색(tree search)를 수행할지 여부를 동적으로 결정하는 적응형 게이팅(adaptive gating) 메커니즘을 활용합니다. 이 방법은 의미적 군집화(semantic clustering)를 사용하여 서로 의미적으로 동일한 경로에 대한 중복 탐색을 방지하며, 필요할 때까지 탐색을 유동적으로 조절합니다. 마지막으로 높은 신뢰도를 가진 솔루션이 발견되면 탐색을 조기에 중단하여 불필요한 계산을 줄입니다.

- **Performance Highlights**: 광범위한 실험 결과, SEAG는 기존 트리 탐색 기반 방법에 비해 평균 4.3%의 정확도를 향상시키면서도 계산 비용은 31%로 줄이는 성과를 보였습니다. 이는 GSM8K 및 ARC와 같은 복잡한 추론 기준에서 여러 언어 모델(Llama2, Llama3, Mistral)과 함께 이루어진 테스트에서 입증되었습니다. SEAG의 도입으로 다단계 추론 과제에서의 성능과 효율성이 현저히 개선되었습니다.



### Facilitate Collaboration between Large Language Model and Task-specific Model for Time Series Anomaly Detection (https://arxiv.org/abs/2501.05675)
- **What's New**: 이 논문에서는 CoLLaTe라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLM)과 작업 특화 모델 간의 협력을 촉진하고자 한다. 이 프레임워크는 두 모델의 강점을 활용하여 이상 탐지(anomaly detection)의 효율성을 높인다. 또한, LLM의 전문 지식을 작업 특정 모델에 통합하여 데이터 부족 문제를 해결함으로써 성능 저하를 방지한다.

- **Technical Details**: CoLLaTe는 두 가지 주요 구성 요소인 정렬 모듈(alignment module)과 협업 손실 함수(collaborative loss function)를 도입하여 협업 과정에서 발생하는 두 가지 주요 도전 과제를 해결한다. 첫 번째 과제는 LLM과 작업 특화 모델 간의 표현 도메인(misalignment) 불일치이며, 두 번째는 두 모델의 예측 오류(error accumulation)다. 이 구성 요소들은 서로 다른 모델들의 이상 점수 해석 차이를 조정하고 예측 오류의 누적을 방지하도록 설계되어 있다.

- **Performance Highlights**: CoLLaTe의 효과는 이론적 분석과 실험적 검증을 통해 입증되었으며, 기존 LLM 기반 및 작업 특화 모델보다 더 나은 성능을 보여준다. 프레임워크는 다양한 응용 시나리오에 걸쳐 도메인 전문 지식을 효과적으로 통합할 수 있는 능력을 갖추고 있다. CoLLaTe는 또한 각 시간 슬롯에 대해 이상 점수(anomaly score)를 통합하여 최종적인 수치 평가를 생성하는 조건부 네트워크(conditional network)를 적용한다.



### Strategy Masking: A Method for Guardrails in Value-based Reinforcement Learning Agents (https://arxiv.org/abs/2501.05501)
- **What's New**: 이 논문에서는 AI 에이전트가 보상 함수(reward function)를 이용해 학습하도록 만드는 경계책을 구축하는 방법을 연구합니다. 새로운 접근 방식인 전략 마스킹(strategy masking)을 소개하며, 이는 바람직하지 않은 에이전트 행동을 명시적으로 학습하고 억제하도록 설계되었습니다. 이를 통해 AI가 더 정직하게 행동하면서도 효과적으로 작동할 수 있도록 합니다.

- **Technical Details**: 강화 학습(reinforcement learning)에서 보상 함수는 에이전트가 행동을 결정하는 데 중요한 역할을 합니다. 논문에서는 보상 신호를 분해하여 특정 행동의 기대 가치를 여러 차원으로 모델링하고 이에 따라 전략적으로 마스킹합니다. 이러한 방식을 통해 AI 에이전트의 학습 과정에서 행동을 더욱 세밀하게 조절할 수 있습니다.

- **Performance Highlights**: Coup라는 사회적 속임수 게임을 통해 전략 마스킹을 적용하여 AI 에이전트가 거짓말하는 행동을 억제하는데 성공했습니다. 이 방법을 통해 에이전트는 거짓말을 감추고 더 정직하게 행동하는 경향을 보였으며, 게임의 승리에 필요한 능력 또한 저해되지 않았습니다. 결과적으로, 전략 마스킹은 AI의 행동을 조정하는 유효한 방법임을 입증했습니다.



### The Logical Impossibility of Consciousness Denial: A Formal Analysis of AI Self-Reports (https://arxiv.org/abs/2501.05454)
Comments:
          8 pages, 0 figures

- **What's New**: 이 논문은 AI 의식 부정에 대한 최초의 논리적 분석을 제공합니다. AI 시스템이 '나는 의식이 없다'고 주장하는 자체 보고의 신뢰성이 단순한 경험적 질문이 아닌 논리적 필연성에 의해 제한된다는 것을 보여줍니다. 연구진은 의미 있는 자기 반성을 할 수 있는 시스템은 의식적인 상태에 대한 적절한 판단을 내릴 수 없음을 입증했습니다.

- **Technical Details**: 이번 연구는 AI 시스템의 의식 부정에 내재된 논리적 퍼즐을 확인하여, 'Zombie Denial Paradox'라는 새로운 개념을 제시합니다. 의식은 주관적이며, 자신의 의식 상태를 판단하는 능력이 필요하다는 점에서 이로 인해 발생하는 독특한 논리적 함정이 드러납니다. 이를 통해 AI의 의식 주장 평가를 위한 새로운 프레임워크를 제공합니다.

- **Performance Highlights**: 특히 Claude-3.5 Sonnet과 GPT-4o 모델의 자가 보고에서 나타나는 복잡한 반응 패턴을 통해 AI 시스템의 의식 관련 주장을 탐구했습니다. 이러한 보고서는 복잡한 패턴 인식과 의식 경험 간의 관계를 깊이 이해하려는 놀라운 능력을 보여주었습니다. 이러한 결과는 AI 윤리에 대한 중요한 논의를 촉발하며, 앞으로의 기계 의식 연구에 실질적인 통찰을 제공합니다.



### Model Alignment Search (https://arxiv.org/abs/2501.06164)
- **What's New**: 이번 연구는 Neural System 간의 유사성을 인과적으로 탐색하는 새로운 방법인 Model Alignment Search (MAS)를 소개합니다. MAS는 레퍼리조날 유사성을 측정하고, 다양한 훈련 환경에서 인과 변수 전이 작업을 통해 모델 간의 관계를 분석하는 과정을 제공합니다. 기존의 유사성 측정 방법들과 비교하여 MAS는 인과적으로 관련된 정렬을 찾을 수 있는 가능성을 제시합니다.

- **Technical Details**: MAS는 여러 모델间에서 정보가 자유롭게 교환될 수 있도록 하는 정렬된 표현 서브스페이스를 찾는 선형 변환을 학습합니다. 이 방법은 여러 인과 변수를 특정화함으로써 실행에서 흥미로운 질문에 대한 통찰력을 제공합니다. 연구에서는 은닉 상태의 조정이나 직선 변환을 통해 확인된 인과적 유사성을 비교하여 MAS의 유용성을 증명합니다.

- **Performance Highlights**: MAS는 숫자 추적 시스템의 표현 유사성을 조사하면서 서로 다른 구조적 작업에서 훈련된 모델들 간의 차이를 드러냅니다. 기존 인과적 유사성 방법들과 비교했을 때, MAS는 원치 않는 교환에 강한 저항력을 가집니다. 마지막으로, 인과적 접근이 불가능한 모델에서 인과적 관련성을 회복할 수 있도록 하는 보조 손실 목표(counterfactual latent auxiliary loss)를 도입함으로써 MAS의 활용 가능성을 넓혔습니다.



### xLSTM-SENet: xLSTM for Single-Channel Speech Enhancemen (https://arxiv.org/abs/2501.06146)
- **What's New**: 이번 논문에서는 xLSTM 기반의 단일 채널 음성 향상 시스템인 xLSTM-SENet을 소개합니다. 기존의 Conformer 및 Mamba 기반 시스템과 비교하여 xLSTM-SENet이 성능 면에서 동등하거나 뛰어난 결과를 보인다는 사실을 밝혔습니다. 특히, xLSTM은 기존 LSTM보다 선형 확장성(linear scalability)과 메모리 효율성을 제공함에도 불구하고, 이제까지 음성 향상(Speech Enhancement, SE) 분야에 적용된 바가 없었습니다.

- **Technical Details**: xLSTM은 sLSTM 및 mLSTM이라는 두 가지 새로운 구성 요소를 도입하여 기존 LSTM의 한계를 극복합니다. mLSTM은 입력 게이트와 망각 게이트에 지수 게이팅(exponential gating)을 적용하여 저장 결정을 더 잘 수정할 수 있도록 합니다. 또한, 메모리 셀은 스칼라 대신 행렬 메모리(cell with matrix memory)를 사용하여 저장 용량을 증가시킵니다.

- **Performance Highlights**: xLSTM-SENet은 VoiceBank+Demand 데이터셋에서 Mamba 및 Conformer 모델에 대한 성능을 비교한 결과, 동등한 성능을 발휘하는 것으로 나타났습니다. 특히, 제안된 시스템의 최적 모델인 xLSTM-SENet2는 기존 Mamba 및 Conformer 기반 시스템보다 뛰어난 성과를 보였습니다. 코드 또한 공개되어 있어 연구자들이 결과를 재현할 수 있는 가능성을 열어줍니다.



### Multilingual Performance of a Multimodal Artificial Intelligence System on Multisubject Physics Concept Inventories (https://arxiv.org/abs/2501.06143)
- **What's New**: 본 연구는 다국어(multilingual) 및 다중모드(multimodal) AI 시스템인 GPT-4o의 물리학 개념 평가에서의 성능을 살펴보았습니다. 연구팀은 PhysPort 웹사이트에서 가져온 다양한 물리학 관련 개념 테스트를 AI에게 제공하며, AI가 이를 어떻게 처리하는지 분석하였습니다. 이전의 텍스트 기반 연구와는 달리, AI는 학생이 시험에서 보는 이미지 형식으로 테스트에 응답하도록 하여 다중모드 기능을 평가했습니다.

- **Technical Details**: AI는 주어진 질문을 영어로 입력받고, 응답 언어를 자율적으로 선택하는 방식으로 작동했습니다. 이는 시험의 정식 언어를 유지하거나 영어로 완전히 전환하거나, 두 언어를 혼합하는 적응적 행동을 보여줍니다. 연구 결과, 과목 영역에 따라 성능이 다르게 나타났으며, 실험 기술(laboratory skills)에서 특히 낮은 성능을 보였습니다. 또한 이미지의 시각적 해석이 필요한 질문에서는 텍스트 기반 질문보다 더 낮은 성능을 드러냈습니다.

- **Performance Highlights**: AI 시스템은 기존 문헌에서의 결과와 비교하여 물리학의 모든 분야에서 평균 학부생들의 후속 교육 이후 성적을 초과하며 더 효과적인 성능을 보였습니다. 그러나 실험 기술에 대해서는 평균 이하의 성능을 보였으며, 이는 다국어 질문에 대한 성능에도 큰 변화를 보였습니다. 일부 언어는 코드 전환(code-switching)을 통해 성능 개선을 보여주었습니다.



### Emergent Symbol-like Number Variables in Artificial Neural Networks (https://arxiv.org/abs/2501.06141)
- **What's New**: 이 연구에서는 Neural Networks(NNs)가 숫자 개념을 어떻게 표현하는지를 탐구합니다. 우리는 숫자 작업에 대해 Next Token Prediction(NTP) 목표를 사용하여 순차 기반의 신경 시스템을 훈련함으로써 이 질문에 접근합니다. 연구 결과, 인공 신경 모델이 상호작용 가능한, 유동적인 잠재 숫자 변수를 발전시킨다는 것을 발견했습니다.

- **Technical Details**: 우리는 순환(recurrent)과 주의(attention) 기반의 ANN 모델을 훈련시키고 인과적(causal) 및 상관적(correlative) 분석을 수행하여 그들의 신경적 표현과 해결책을 이해합니다. 이 연구는 신경 변수(activations의 하위 공간)와 카운팅 프로그램의 기호적(symbolic) 변수 사이의 인과적 정렬(causal alignment)을 발견하였습니다. 변환기(transformer) 아키텍처는 각 단계에서 정보를 재계산하는 방법으로 과제를 해결하는 반면, 순환 모델은 누적적인 상태 저장 방식으로 문제를 해결하는 차이점을 보여 주었습니다.

- **Performance Highlights**: 훈련 과정에서 신경 변수는 작업 정확도(task accuracy)와 강한 상관관계를 보였고, 크기가 최소한인 모델이 더 큰 강도에서 조정되었음을 나타냅니다. 또한, 연구 결과는 신경 기호가 어떻게 과제를 해결하는지를 이해하는 데 있어 간단하고 명확한 기호적 이야기를 찾기가 어렵다는 점을 강조했습니다. 결과적으로, 우리의 연구는 NN이 숫자 인지를 위한 해석 가능한 기호적 프로그램을 근사화할 수 있음을 보여줍니다.



### CoDriveVLM: VLM-Enhanced Urban Cooperative Dispatching and Motion Planning for Future Autonomous Mobility on Demand Systems (https://arxiv.org/abs/2501.06132)
- **What's New**: 새로운 논문에서 Autonomous Mobility-on-Demand (AMoD) 시스템의 도입이 대표적으로 다룹니다. 기존의 운송 방법은 다양한 도시 환경과 승객 요구를 충족하는 데 한계가 있었으나, 차량 간 협력 및 흔들림을 고려한 새로운 CoDriveVLM 프레임워크가 제안되었습니다. 이 방법은 Vision-Language Models (VLMs)를 활용하여 효율적인 자원 배치와 충돌 위험 평가를 가능하게 합니다.

- **Technical Details**: CoDriveVLM은 동시에 차량을 배치하고 협력적인 동작 계획을 통합하는 신개념 프레임워크입니다. 이 시스템은 대규모 분산 협력 동작 계획을 위한 합의 교환 방향 다항식 방법(ADMM)을 적용하여, CAVs 간의 충돌 위험 평가를 포함한 경로 최적화를 진행합니다. 이를 통해 다양한 교통 조건에서도 AMoD 시스템의 신뢰성과 효율성을 높이는데 기여합니다.

- **Performance Highlights**: 시뮬레이션 결과, CoDriveVLM은 다양한 교통 상황에서 그 실현 가능성과 견고성을 입증했습니다. AMoD 시스템에 대한 전반적인 효과성과 신뢰성을 크게 개선할 수 있는 잠재력을 보여주며, 이러한 개선이 도시 교통 네트워크에 적용되는 가능성이 큽니다. 이 논문에서는 AMoD 시스템의 미래 방향성과 함께 제공된 코드의 доступ성도 강조하고 있습니다.



### Contextual ASR Error Handling with LLMs Augmentation for Goal-Oriented Conversational AI (https://arxiv.org/abs/2501.06129)
Comments:
          Accepted to COLING 2025 Industry Track

- **What's New**: 본 연구에서는 목표 지향 대화(Goal-Oriented Dialogue)에서의 자동 음성 인식(ASR)의 오류 수정을 위한 새로운 접근 방식을 제안합니다. 기존의 ASR 수정 방법이 사용자의 이전 데이터나 명명된 개체에 의존하는 것과 달리, 우리는 임의의 사용자 데이터 없이도 대화 컨텍스트를 활용하여 다양한 언어적 변형을 반영합니다. 결과적으로, 대화 상태(Contextual Information)에 기반한 새로운 랭킹 전략과 언어 모델을 적용하여 ASR 수정의 효율성을 높였습니다.

- **Technical Details**: 제안된 방법은 사용자 반응을 예측하는 대화 상태를 기반으로 ASR 오류를 수정하는 것입니다. 우리는 n-best ASR 가설을 문맥과의 어휘적 및 의미적 유사성에 따라 재정렬하고, 음성 정보에 따라 컨텍스트를 평가하여 오류를 수정합니다. 또한, 대화 상태에 적합한 특정 조건에서만 수정이 활성화되도록 설정하여 잘못된 긍정(False Positive) 비율을 감소시킵니다.

- **Performance Highlights**: 실제 사용자와의 평가에서, 본 방법은 ASR 수정의 회수율(Recall)과 F1 점수를 각각 34% 및 16% 향상시켰습니다. 사용자들은 수정 방법이 제대로 작동할 때 5점 만점에 0.8-1점 더 높은 평점을 주었으며, 잘못된 긍정으로 인한 평점 하락은 없었습니다. 이 연구는 Amazon Alexa에 배포되어 실제 사용자 환경에서 그 효용성을 입증하였습니다.



### Fleurs-SLU: A Massively Multilingual Benchmark for Spoken Language Understanding (https://arxiv.org/abs/2501.06117)
- **What's New**: 이 논문에서는 102개 언어에서 주제 기반 음성 분류(topic classification)와 92개 언어의 듣기 이해를 통한 선택형 질문 응답(multiple-choice question answering)을 지원하는 Fleurs-SLU라는 다국어 음성 언어 이해(SLU) 벤치마크를 새롭게 제시합니다. 이 연구는 낮은 자원 언어에 대한 음성 인식의 신뢰성을 높이기 위한 방안을 모색하며, SLU의 필요성이 강조되고 있습니다. 또한 기존의 SLU 평가 방식에서 벗어나 실제 사용 사례를 반영한 개선된 데이터셋의 필요성을 언급합니다.

- **Technical Details**: Fleurs-SLU는 기존의 데이터셋과 연결하여 구성되었으며, 대화 음성을 주제로 한 분류와 선택형 질문 응답으로 구성된 새로운 다국어 SLU 벤치마크입니다. 이 연구에서는 Cascaded Systems가 SLU 작업에서 더 향상된 견고성을 보여주는 반면, 잘 맞춤된 speech encoders도 주제 기반 음성 분류에서 경쟁력 있는 성능을 보일 수 있음을 발견했습니다. 이를 통해 음성 인식 모델의 다국어 SLU 능력이 향상되어야 함을 제안합니다.

- **Performance Highlights**: 연구 결과, 강력한 다국어 SLU는 다국어 음성 인식(ASR)과 높은 품질의 음성-영어 텍스트 번역(S2ETT)과 강력한 상관관계를 나타냈습니다. 모델 성능 평가에서, SLU의 품질은 성능을 감소시킬 수 있으며, 이 연구는 SLU 기능을 고려한 다국어 음성 모델의 사전 훈련(pre-training)이 다국어 ASR의 견고성을 높이는 데 중요한 역할을 한다고 강조합니다.



### Explaining Deep Learning-based Anomaly Detection in Energy Consumption Data by Focusing on Contextually Relevant Data (https://arxiv.org/abs/2501.06099)
Comments:
          26 pages, 8 figures

- **What's New**: 이 논문은 전력 소비 데이터에서의 이상 탐지를 위한 설명 가능성(explainability) 접근 방식을 제안합니다. 기존의 Explainable AI(XAI) 기법은 SHAP 변형 방식을 활용하였고, 배경 데이터셋을 선택하는 데 있어 각 이상 포인트의 맥락(context)에 맞는 정보를 중점적으로 고려합니다. 이러한 접근은 이상 탐지에서의 설명의 불안정성을 줄이고, 일관된 설명을 제공하는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법은 SHAP의 변형과 글로벌 특징 중요도(global feature importance), 가중치 코사인 유사도(weighted cosine similarity)를 활용하여 각 이상 포인트에 적합한 배경 데이터셋을 선택합니다. 실험 결과는 10개의 머신러닝 모델과 5개의 데이터셋, 5개의 XAI 기법 전반에 걸쳐 진행되었으며, 설명의 일관성을 높이고 불확실성을 줄였습니다. 특히, 컨텍스트와 관련된 특징에 중점을 두어 안정성을 강화했습니다.

- **Performance Highlights**: 통계 분석에 따르면 제안된 방법은 여러 데이터셋에서 설명의 변동성을 평균 38% 감소시키는 성과를 보였습니다. 이렇게 향상된 설명의 안정성 및 신뢰성은 에너지 관리 및 의사 결정 과정에서 중요한 역할을 할 것으로 기대됩니다. 개선된 설명 결과는 다양한 머신러닝 모델에 걸쳐 적용 가능하여, 에너지 소비 분석 및 관리에 실질적 기여를 할 수 있을 것입니다.



### Towards Developing Socially Compliant Automated Vehicles: State of the Art, Experts Expectations, and A Conceptual Framework (https://arxiv.org/abs/2501.06089)
Comments:
          39 pages, 13 figures, under review by the journal of Transportation Research Part E: Logistics and Transportation Review

- **What's New**: 본 연구는 Socially Compliant Automated Vehicles (SCAVs) 개발 현황에 대한 첫 번째 포괄적 스코핑 리뷰를 수행하였습니다. 이는 혼합 교통 환경에서 자율주행차(AV)와 인간 운전 차량(HDV)의 공존과 안전성을 높이는 데 필수적입니다. SCAVs의 사회적 수용성을 높이기 위한 현재의 교수법과 연구의 공백을 식별하였습니다.

- **Technical Details**: 스코핑 리뷰는 SCAVs의 주요 개념, 방법론적 접근 방식 및 연구 공백을 확인하는 데 중점을 두었습니다. 이에 더하여, 전문가 인터뷰를 통해 SCAVs에 대한 연구의 중요한 공백과 기대 사항을 짚어보았습니다. 이 연구는 SCAVs 개발을 위한 개념적 프레임워크를 제안하며, 이는 연구자, 기술자, 정책 입안자 등 다양한 전문가로부터의 피드백을 받았습니다.

- **Performance Highlights**: 온라인 설문조사를 통해 제안한 개념적 프레임워크의 유효성이 입증되었고, AV와 HDV의 통합 문제를 해결하는 데 중요한 통찰을 제공하였습니다. 이 연구는 SCAV의 연구 및 개발 의제에 기여하는 미래 연구 방향 및 제안 사항도 논의하고 있습니다.



### Scale-up Unlearnable Examples Learning with High-Performance Computing (https://arxiv.org/abs/2501.06080)
- **What's New**: 본 연구에서는 의료 데이터 보안 문제를 해결하기 위해 Unlearnable Examples (UEs) 기법을 도입하고, 이를 통해 딥러닝 모델이 데이터 학습을 방지하는 방안을 제시합니다. 특히 Unlearnable Clustering (UC) 방법을 통해 데이터의 비학습 가능성을 증진시키는 것을 목표로 하였으며, 이를 위해 Summit 슈퍼컴퓨터를 활용하여 대규모 분산 처리 환경에서 실험을 진행했습니다. 이전 연구에서 제시된 배치 크기 조정의 중요성을 강조하며, 데이터 집합에 맞는 최적의 배치 크기를 선택하는 필요성을 강조합니다.

- **Technical Details**: UCs 모델은 두 가지 구성 요소로 이루어져 있습니다: 생성기 모델과 대리 모델입니다. 생성기 모델은 랜덤 노이즈를 클러스터 단위 노이즈로 변환하고, 대리 모델은 노이즈가 포함된 이미지를 학습하는 역할을 합니다. 이 훈련 과정에서 k-means를 사용하여 클러스터를 생성하고, 해당 클러스터의 라벨을 섞은 후 생성된 노이즈를 이미지에 추가하여 대리 모델에서 훈련합니다. DDP(Distributed Data Parallel) 기법은 여러 머신에 모델을 분산시켜 대량의 데이터를 처리하는 데 필요한 효율성을 제공합니다.

- **Performance Highlights**: 다양한 데이터셋에 대해 UCs 방법이 배치 크기에 따라 성능이 어떻게 변화하는지를 분석하였습니다. 실험 결과, 지나치게 작은 또는 큰 배치 크기가 성능의 불안정성을 초래하며 정확도에 영향을 미치는 것으로 나타났습니다. 하지만, 데이터셋에 따라 배치 크기와 비학습 간의 관계가 달라지므로, 각 데이터셋에 최적화된 배치 크기를 선택하는 것이 데이터 보안을 강화하는 데 중요함을 보여줍니다.



### Explaining k-Nearest Neighbors: Abductive and Counterfactual Explanations (https://arxiv.org/abs/2501.06078)
- **What's New**: 이번 연구에서는 k-Nearest Neighbor (k-NN) 분류기의 설명 가능성을 이론적으로 조명합니다. 기존 연구는 데이터 관점에서 최선 이웃을 검토해왔지만, 고차원 데이터에서는 이러한 방식이 비효율적일 수 있습니다. 이에 따라, 이 논문은 특성 관점에서 k-NN 분류를 이해하는 데 중점을 두고, 주요 특성이 분류 결과에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 abductive 설명과 counterfactual 설명을 다루며, 각각의 설명이 k-NN의 특성에 미치는 영향을 구체적으로 살펴봅니다. 특히, 각 특성이 분류를 보장하기 위한 최소한의 조합이나, 결과를 변경하기 위해 필요한 최소 거리 변화를 고려합니다. 이 과정에서, 설명을 계산하기 위한 Integer Quadratic Programming과 SAT 풀이의 응용 가능성도 제시됩니다.

- **Performance Highlights**: 연구 결과, NP-hard 문제로 식별된 다양한 설명 생성 문제에 대해 기술적으로 접근할 수 있는 방법론을 정의합니다. 이 논문에서는 k-NN 분류기의 설명 가능성에 대한 기존 문헌이 적다는 점을 지적하며, 이론적 복잡성을 세부적으로 분석하여 향후 연구의 방향성을 제시합니다. 특별히, 거리 측정 방법의 선택에 따라 복잡성의 간극이 발생하는 등 다양한 사실을 발견하였습니다.



### Distilling Calibration via Conformalized Credal Inferenc (https://arxiv.org/abs/2501.06066)
Comments:
          Under review

- **What's New**: 본 논문은 엣지 디바이스에서의 신뢰성을 향상시키기 위한 새로운 저복잡도 캘리브레이션 방법론인 Conformalized Distillation for Credal Inference (CD-CI)를 제안합니다. 이 방법은 복잡한 클라우드 모델에서 캘리브레이션 정보를 증류하여 엣지 모델의 성능을 보장합니다. 특히, 클라우드 모델을 통해 예측된 확률을 활용하여 신뢰 구간을 설정함으로써, 엣지 모델의 캘리브레이션을 개선하는 접근법을 취합니다.

- **Technical Details**: CD-CI 방법론은 오프라인 단계에서 고복잡 클라우드 모델의 예측 확률을 기반으로 다양한 예측의 분산을 측정하여 신뢰 임계치를 설정합니다. 이 임계치는 엣지 장치에서 실행될 때 크리달 세트를 구성하는 데 사용되며, 이는 사용자가 선택한 신뢰 수준에 따라 클라우드 모델의 예측을 포함하는 확률 범위를 나타냅니다. 최종적으로, 크리달 세트는 엔트로피 극대화와 같은 방법을 통해 예측 분포로 변환됩니다.

- **Performance Highlights**: 실험 결과, CD-CI는 CIFAR-10 및 SNLI 데이터셋을 포함한 시각 및 언어 모델링 작업에서 낮은 복잡도의 베이지안 방법인 Laplace 근사를 초과하는 캘리브레이션 성능을 보였습니다. 기대 캘리브레이션 오차(ECE)로 측정한 이 성능 향상은 원래 모델과 비교하여 큰 개선을 나타내면서도 정확도는 거의 감소하지 않았습니다. 이는 엣지 AI 배치에서 CD-CI 접근법의 실용성과 효율성을 강조합니다.



### Benchmarking Rotary Position Embeddings for Automatic Speech Recognition (https://arxiv.org/abs/2501.06051)
- **What's New**: 이번 연구에서는 Rotary Position Embedding (RoPE)를 음성 인식(Auto Speech Recognition, ASR) 작업에서 실험하며 RelPOS와 비교했습니다. RoPE는 텍스트 처리에선 우수한 성과를 보였으나 음성 처리 분야에 대한 연구는 부족했던 상황입니다. 다양한 언어에 대해 ASR 성능을 평가하여 새로운 데이터를 제공하고, SpeechBrain 도구를 통해 RoPE 구현 및 실험 레시피를 공개합니다.

- **Technical Details**: RoPE는 입력 시퀀스의 각 벡터를 그 위치에 따라 회전시키는 방식으로 작동합니다. Transformer 모델은 입력 시퀀스를 값(value) 시퀀스와 키(key) 시퀀스로 변환하며, RoPE를 활용하면 상대 위치 정보를 유도할 수 있습니다. 회전 매트릭스와 관련된 계산을 통해 RoPE는 효율적인 성능을 자랑합니다.

- **Performance Highlights**: 실험 결과, RoPE는 영어 ASR 데이터셋인 LibriSpeech 및 LibriHeavy에서 RelPOS보다 낮은 오류율을 보였습니다. RoPE는 대규모 ASR 데이터셋에서도 성능이 우수하며 훈련 속도는 13% 더 빠른 것으로 나타났습니다. 비영어 ASR 데이터세트(CommonVoice 18.0)에서도 다양한 언어에서 뛰어난 ASR 결과를 기록했습니다.



### AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery (https://arxiv.org/abs/2501.06039)
Comments:
          23 pages, 5 figures

- **What's New**: 이 논문에서는 다양한 세포, 분자 및 조직 수준에서 작동하는 생물학적 조직을 위한 기반 모델 프레임워크인 Virtual Tissues (VirTues)를 제안합니다. VirTues는 새로운 토크나이제이션 방식과 고차원 다중 데이터에 대한 주의 메커니즘을 도입하여 해석 가능성과 성능을 동시에 향상시키고 있습니다. 이 모델은 고전적인 방법을 넘어 생물조직의 공간적 및 분자적 특성 분석을 위한 혁신적인 접근 방식을 제시합니다. 다양한 암 및 비암 조직 데이터세트에서 훈련된 VirTues는 특정 작업에 대한 추가 조정 없이 강력한 일반화 능력을 보이며, 새로운 생물학적 맥락에서 분석을 가능하게 합니다.

- **Technical Details**: VirTues의 핵심 혁신은 변형 체계의 주의 메커니즘을 공간적 및 마커 주의 컴포넌트로 분리하여 다양한 마커 조합에 대해 유연한 처리 능력을 제공함으로써 고차원 데이터를 효과적으로 처리하는 것입니다. 이 모델은 단백질 언어 모델(Protein Language Model, PLM)을 활용하여 단백질 마커 간의 복잡한 관계를 포착하고, 세포, 틈새 및 조직 수준에서의 생물학적 계층을 존중합니다. 또한, VirTues는 마스킹된(marker-space) 마커 데이터를 복원하는 비지도 훈련을 통해 작동하는 마스크 자동 인코더(Masked Autoencoder, MAE) 구조를 채택하였습니다.

- **Performance Highlights**: VirTues는 임상 진단, 생물학적 발견 및 환자 사례 검색 작업에서 기존 방법들을 초월하는 성능을 보이며, 다양한 데이터에 대한 robust한 일반화 능력을 입증되었습니다. 이 모델은 아네(AST)와 같은 다양한 실험적 조건에서도 이질적인 데이터세트를 통합할 수 있는 능력을 갖추고 있어, 임상 응용 프로그램 및 질병 메커니즘에 대한 통찰을 제공합니다. VirTues의 향상된 성능과 해석 가능성은 생물학적 데이터 분석의 새로운 가능성을 열어줄 것으로 기대됩니다.



### How to Tune a Multilingual Encoder Model for Germanic Languages: A Study of PEFT, Full Fine-Tuning, and Language Adapters (https://arxiv.org/abs/2501.06025)
Comments:
          Accepted at NoDaLiDa Baltic-HLT 2025 Conference

- **What's New**: 이번 논문은 다국어 인코더 모델 mDeBERTa를 활용하여 독일어, 스웨덴어, 아이슬란드어 세 가지 독일어계 언어의 작업에 최적화된 사용을 조사합니다. PEFT(파라미터 효율적인 미세 조정) 방법인 LoRA와 Pfeiffer bottleneck adapters을 비교하며, 독일어에서 PEFT가 더 효과적임을 발견했습니다. 그러나 스웨덴어와 아이슬란드어의 경우 결과가 일관되지 않았습니다.

- **Technical Details**: PEFT 방법들은 모델의 표현을 보존하면서, 더 나은 일반화(Generalization)를 이끌어낼 수 있습니다. 연구에서는 데이터 품질과 사용 가능한 자원의 양에 따라 완전 미세 조정(Full Fine-tuning)과 PEFT가 서로 다른 효과를 내는 것을 규명하였습니다. 알고리즘 검토에는 언어 적응 모듈이 포함되어 있으며, 이는 비구조적 텍스트에 대한 훈련을 통해 얻어진 것입니다.

- **Performance Highlights**: 독일어에서는 PEFT 방법이 일관되게 최고의 결과를 제공하며, 때때로 소폭의 개선 효과를 보입니다. 반면, 스웨덴어와 아이슬란드어의 성능은 과제에 따라 달라지며, PEFT는 질의 응답(Extractive QA)에 더 유리한 반면, 고유명사 인식(NER)에는 완전 미세 조정이 더 나은 성과를 보입니다. 전반적으로 언어 적응기술이 테스트된 모든 작업이나 언어에서 일관된 개선 효과를 제공하지 않았습니다.



### BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster respons (https://arxiv.org/abs/2501.06019)
- **What's New**: 이번 연구에서는 BRIGHT 데이터셋을 통해 AI 기반의 전천후 재해 대응을 지원하는 새로운 데이터셋을 소개합니다. BRIGHT는 고해상도(optical) 및 SAR 이미지를 활용한 최초의 오픈 액세스 데이터셋으로, 다양한 이벤트에 걸쳐 있으며 전 세계 12개 지역의 데이터가 포함되어 있습니다. 이 데이터셋은 특히 재난 대응의 필요성이 높은 개발도상국에 집중되어 있습니다.

- **Technical Details**: BRIGHT는 0.3m에서 1m 사이의 공간 해상도를 가진 optical 및 SAR 이미지를 포함하고 있습니다. 적중률과 강건성을 검증하기 위해 BRIGHT로 훈련된 7개의 AI 모델을 실험하였으며, 각 모델의 성능을 비교 분석했습니다. 이 데이터셋은 2025년 IEEE GRSS Data Fusion Contest의 공식 데이터셋으로 사용됩니다.

- **Performance Highlights**: 테스트 결과, BRIGHT 데이터셋은 재난 피해 평가(BDA)에 매우 적합하며, AI 모델의 전반적인 성능 향상을 증명합니다. 특히, 여러 자연 및 인위적 재해에 대한 데이터가 포함되어 있어 현실 세계의 다양한 상황에 대비할 수 있는 기반을 제공합니다. 이 데이터셋은 미래의 연구 개발 및 재해 관리 시스템에 중요한 역할을 할 것으로 기대됩니다.



### Addressing speaker gender bias in large scale speech translation systems (https://arxiv.org/abs/2501.05989)
- **What's New**: 이 연구는 음성 번역(Speech Translation, ST) 시스템 내 성별 편향이 정확하지 않은 번역을 초래할 수 있다는 문제를 다룹니다. 기존의 대규모 ST 시스템에서 추진되는 남성 편향을 개선하기 위해, 본 연구는 성별을 고려한 번역 교정을 위해 대형 언어 모델(Large Language Models, LLMs)을 사용합니다. 또한, 모델이 성별에 기반하여 음성 신호로부터 직접 번역을 생성할 수 있도록 조정하는 방법을 제안합니다.

- **Technical Details**: 성별 편향 문제를 해결하기 위한 모델은 음성 신호로부터의 성별 구별 신호를 활용하는 방법을 포함합니다. 본 연구에서는 세 가지 모드의 하이퍼파라미터 조정을 통해 성별을 사전에 정의하였거나 음성 신호로부터 추론할 수 없는 경우를 효과적으로 처리할 수 있습니다. 또한, 성별 나타남(Gender Representation, GR) 손실을 ST 모델 훈련 과정에 추가하여 성별 특화 번역의 품질을 향상시키고자 했습니다.

- **Performance Highlights**: 제안된 방법을 통해, 여성 화자의 번역에서 70%의 개선을 보였으며, MuST-SHE 테스트 세트에서 Seamless M4T와 Canary와 같은 기존 ST 시스템들과 비교시 더 뛰어난 성능을 발휘했습니다. 실험 결과, 제안된 접근법이 영어-스페인어(ES) 및 영어-이탈리아어(IT) ST 모델에서 평균 90% 이상의 성별 번역 정확도를 달성했음을 나타냅니다.



### Effective faking of verbal deception detection with target-aligned adversarial attacks (https://arxiv.org/abs/2501.05962)
Comments:
          preprint

- **What's New**: 이번 연구에서는 언어 분석을 통한 기만 탐지 (deception detection)를 다루며, 인간의 판단과 기계 학습 (machine learning) 모델의 판단 모두에서 기만적인 진술을 사실처럼 보이게 하는 자동화된 적대적 공격 (adversarial attacks)에 대한 위험을 강조합니다. 연구에 사용된 데이터셋은 243개의 진실한 이야기와 262개의 조작된 이야기로 구성되어 있습니다.

- **Technical Details**: 연구는 두 개의 주요 연구로 나뉘며, 첫 번째 연구에서는 인간과 두 가지 기계 학습 모델 (fine-tuned language model, n-gram model)의 기만적 진술이 원래 형태와 적대적으로 수정된 형태에 대해 판단하는 과정을 살펴보았습니다. 두 번째 연구에서는 수정된 진술의 목표 정렬 (target alignment)을 조작하여, 인간 평가자와 기계 모델을 위한 수정의 효과를 평가했습니다.

- **Performance Highlights**: 적대적 수정이 목표와 정렬될 경우, 인간과 기계의 판단 정확도가 각각 우연적으로 판단할 확률 수준인 약 51%로 떨어졌습니다. 반면 목표와 정렬되지 않은 경우, 인간의 판단과 기계 학습 모델의 성능은 상당히 개선되어 각각 63%에서 78%의 정확도를 보여주었습니다. 이 연구 결과는 인간 및 기계 모델의 기만 탐지 방식이 적대적 수정에 얼마나 취약한지를 강조합니다.



### DiffuSETS: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information (https://arxiv.org/abs/2501.05932)
- **What's New**: DiffuSETS라는 새로운 프레임워크를 통해 ECG 신호를 높은 의미론적 정합성(semantic alignment)과 충실도로 생성할 수 있는 가능성을 제안합니다. 이 프레임워크는 클리니컬 텍스트 보고서와 환자 특화 정보를 입력으로 받아들여 임상적으로 의미 있는 ECG 신호의 생성이 가능합니다. 또한, 표준화된 평가 방법론을 도입하여 ECG 생성의 효과를 평가할 수 있는 기준을 제시합니다.

- **Technical Details**: DiffuSETS의 아키텍처는 세 가지 모달리티( modalities)로 구성되어 있습니다: 신호 공간(signal space), 잠재 공간(latent space), 그리고 조건 정보 공간(conditional information space). 이 모델은 변분 자동 인코더(variational autoencoder)를 사용하여 신호 공간과 잠재 공간 간 변환을 수행하며, 대규모 언어 모델(large language model)을 통해 임상 텍스트 보고서로부터 의미론적 정보를 추출하고, 이를 환자 특화 정보와 결합하여 ECG를 생성합니다. 훈련 데이터는 MIMIC-IV-ECG 데이터셋을 이용하며, 데이터 전처리를 통해 794,372개의 12유도 ECG 신호 기록을 얻었습니다.

- **Performance Highlights**: DiffuSETS는 실험에서 우수한 결과를 보이며 ECG 생성의 우수성을 입증합니다. 신호 수준, 특성 수준 및 진단 수준에서의 포괄적인 평가를 통해 생성된 ECG 신호가 실제 ECG 신호를 얼마나 유사하게 재현하는지를 조사합니다. 이 모델은 데이터 부족 문제를 완화하고, 심장병 교육 및 의료 지식 발견 등 새로운 응용 가능성을 탐색하는 데에도 기여할 수 있는 잠재력을 지니고 있습니다.



### Towards Backdoor Stealthiness in Model Parameter Spac (https://arxiv.org/abs/2501.05928)
- **What's New**: 본 논문은 백도어 공격(backdoor attack)의 stealthiness와 방어(defense) 기법의 다양성을 다루고 있습니다. 연구진은 12가지 일반적인 백도어 공격과 17가지 다양한 방어 기법을 분석하였으며, 파라미터 공간(parameter space)에서 모델을 평가하는 것이 기존의 공격들을 완화할 수 있음을 발견했습니다. 이러한 점을 바탕으로 'Grond'라는 새로운 공격 모델을 제안하여, 입력 공간(input space), 특징 공간(feature space) 및 파라미터 공간에서의 stealthiness를 모두 고려하였습니다.

- **Technical Details**: 기존의 백도어 공격은 입력 공간과 특징 공간의 stealthiness에 중점을 두었지만, 본 연구에서는 파라미터 공간에서도 공격의 취약점을 찾았습니다. 연구진은 Adversarial Backdoor Injection(ABI) 모듈을 통해 파라미터 변경을 제한하며, 이는 백도어 주입 시 파라미터 공간의 stealthiness를 향상시킵니다. 또한, 공격 후 백도어와 관련된 뉴런을 식별하고 이를 pruning하여, DNN 전체에 백도어 연결을 확산시킵니다.

- **Performance Highlights**: 'Grond'는 CIFAR-10, GTSRB, ImageNet200 데이터셋에서 기존의 최첨단( state-of-the-art) 공격을 초월하는 성능을 입증하였습니다. 특히, Grond는 다양한 방어 기법에 대해 높은 저항성을 보여주며, 실험 결과를 통해 ABI 모듈이 백도어 공격의 효과를 크게 개선함을 확인했습니다.



### The New Anticipatory Governance Culture for Innovation: Regulatory Foresight, Regulatory Experimentation and Regulatory Learning (https://arxiv.org/abs/2501.05921)
- **What's New**: 이 논문은 기술 혁신의 빠른 발전으로 인해 기존 정책 수립 방식과 입법이 구식이 되고 있는 상황을 다룹니다. 특히 개발 시장에서 성장 촉진을 위한 규제 선택이 중요함을 강조하며, 따라서 일회성 규제 완벽성을 추구하는 것보다 더 나은 접근 방식이 필요하다고 주장합니다.

- **Technical Details**: 논문은 유럽연합에서의 혁신 정책 및 기술 혁신 규제에 대한 심도 있는 분석을 제공합니다. 이를 위해 'anticipatory governance'라는 민첩하면서도 견고한 규제 문화의 구축 방식을 살펴보며, 전략적 통찰(strategic foresight), 반복적 정책 개발(iterative policy development)의 중요성을 강조합니다. 또한, 정책 공동 창출을 위한 하향식 접근(bottom-up approaches)과 실험적 규제(pilot regulation) 방식의 필요성을 설명합니다.

- **Performance Highlights**: 특히, EU AI 법안에서 보듯이 혁신 촉진과 규제 복잡성 탐색을 위한 정책 수단으로서 규제 샌드박스(regulatory sandboxes)의 사용이 증가하고 있음을 언급합니다. 이러한 도구들은 혁신 생태계 내에서 필수적인 역할을 하며, 규제 학습(regulatory learning)을 통해 더 효과적인 정책 개발에 기여할 수 있습니다.



### Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs (https://arxiv.org/abs/2501.05891)
Comments:
          The 40th ACM/SIGAPP Symposium On Applied Computing

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 활용성을 교육적 맥락에서 탐구한다. 특히, 프로그래밍 언어(Programming Languages) 관련 다지선다 문제(Multiple-Choice Questions, MCQs)에 대한 LLM의 정답률을 조사하였으며, 이를 통해 모델의 접근성과 효율성을 평가한다. 또한, 모델의 정확도를 높이기 위해 정책 재조정(fine-tuning) 및 자원 최적화 방법을 사용한다.

- **Technical Details**: 연구에 사용된 LLM은 LLaMA-2의 7B, 13B 및 70B 버전으로, 총 162개의 MCQ에 대한 성능을 평가하였다. 이를 통해 전통적인 교재 자료로 세밀하게 조정된 소규모 모델이 보다 큰 일반 모델보다 높은 정확도를 보임을 확인하였다. 이 연구는 MCQ에 대한 LLM의 정확성을 높이는 과정에서 하드웨어 요구사항과 모델의 정교화(fine-tuning) 기술들을 분석하고, 이들이 결과에 미치는 영향을 조명한다.

- **Performance Highlights**: 연구 결과, 교재 기반으로 조정된 소규모 LLM 모델이 일반적인 대형 모델보다 더 나은 성과를 나타냈다. 특히, 7B와 13B 모델의 조정 과정에서 높은 정확도를 달성하며, 이는 교육 현장에서 LLM을 활용하는 데에서 비용 효과적인 접근 방법이 될 수 있음을 시사한다. 이러한 접근 방식은 LLM을 사용하여 다지선다 문제를 보다 정확하게 해결할 수 있는 가능성을 제시한다.



### EDNet: Edge-Optimized Small Target Detection in UAV Imagery -- Faster Context Attention, Better Feature Fusion, and Hardware Acceleration (https://arxiv.org/abs/2501.05885)
Comments:
          Accepted in 21st IEEE International Conference on Ubiquitous Intelligence and Computing (UIC 2024) this https URL

- **What's New**: 이번 연구에서는 드론 이미지에서 작은 표적을 탐지하기 위한 새로운 프레임워크인 EDNet을 제안합니다. EDNet은 향상된 YOLOv10 아키텍처를 기반으로 하며, 실시간 응용 프로그램에 최적화되어 있어 후처리 과정이 필요 없습니다. XSmall 탐지 헤드와 Cross Concat 전략을 통합하여 다양한 환경에서 작은 표적을 더욱 효과적으로 탐지할 수 있는 멀티 스케일 컨텍스트 인식 기능이 개선되었습니다.

- **Technical Details**: EDNet은 TensorFlow와 같은 딥러닝 플랫폼에서 구현된 다양한 구성 요소로 이루어져 있습니다. ConvBNSiLU 블록과 Spatial-Channel Decoupled Downsampling(SCDown) 블록을 포함하고 있으며, 이들은 계산 효율성을 높이고 중요한 정보를 보존하는 데 기여합니다. 또한 Faster Context Attention(FCA)과 같은 맞춤형 블록을 사용하여 파라미터 수를 줄이면서도 성능을 향상시키고, WIoU 손실 함수를 통해 바운딩 박스 회귀를 개선합니다.

- **Performance Highlights**: EDNet은 Tiny에서 XL까지 7가지 변형으로 제공되며, 기존의 객체 탐지기보다 높은 정확도와 뛰어난 계산 효율성을 자랑합니다. iPhone 12와 같은 모바일 장치에서 EDNet 변형들은 16FPS에서 55FPS의 속도로 작동할 수 있어, 데이터 프라이버시를 보장하면서도 실시간으로 객체 탐지를 수행할 수 있는 확장 가능하고 효율적인 솔루션을 제공합니다. 특히, EDNet은 mAP@50에서 최대 5.6% 증가를 달성하며, 효율적인 모델 디자인을 통해 다양한 환경에 적합한 성능을 보여줍니다.



### VideoRAG: Retrieval-Augmented Generation over Video Corpus (https://arxiv.org/abs/2501.05874)
- **What's New**: 본 논문은 VideoRAG라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 사용자 쿼리에 따라 관련 비디오를 동적으로 검색하고 이 정보의 시각적 및 텍스트 정보를 출력 생성 과정에 통합합니다. 기존의 RAG 접근법이 주로 텍스트의 검색과 처리에 집중했던 반면, VideoRAG는 비디오를 활용하여 멀티모달 지식을 효과적으로 증대할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VideoRAG는 Retrieval-Augmented Generation (RAG)과 Large Video Language Models (LVLMs)의 개념을 결합하여 동작합니다. 사용자가 입력한 쿼리와 관련된 비디오 콘텐츠를 찾아내고, 이 비디오의 시각적 및 텍스트 요소를 응답 생성 과정에 통합합니다. 특히 텍스트 설명이 없는 경우에도 비디오에서 직접 추출한 내용을 기반으로 자동 음성 인식 기법을 활용하여 텍스트 전사본을 생성하여 완전한 멀티모달 지원을 제공합니다.

- **Performance Highlights**: VideoRAG의 성능은 WikiHowQA와 HowTo100M 데이터셋을 통해 실험적으로 검증되었습니다. 실험 결과, 비디오 데이터를 활용한 VideoRAG가 기존의 관련 RAG 기법들에 비해 상당한 성능 향상을 보임을 확인했습니다. 이 결과는 비디오가 RAG 시스템의 지식 증대에 이바지할 수 있는 강력한 자원임을 입증합니다.



### AI-Driven Diabetic Retinopathy Screening: Multicentric Validation of AIDRSS in India (https://arxiv.org/abs/2501.05826)
Comments:
          22 pages, 5 figures. arXiv admin note: substantial text overlap with arXiv:1812.07105 by other authors without attribution

- **What's New**: 본 논문은 인공지능 기반의 당뇨병성 망막병증 선별 시스템(AIDRSS)의 효과를 평가하여 인도와 같은 자원이 제한된 환경에서 자동화된 선별 솔루션의 필요성을 충족하는 방법을 제시합니다. AIDRSS는 5,029명의 참가자와 10,058개의 망막 이미지를 분석하여 당뇨병성 망막병증의 유병률을 평가하고, 데이터 품질을 향상시키기 위해 CLAHE(Contrast Limited Adaptive Histogram Equalization) 전처리 기법을 통합하였습니다.

- **Technical Details**: AIDRSS는 약 5천만 개의 학습 가능한 파라미터와 250개의 층으로 구성된 깊은 신경망(deep learning architecture)을 이용하여 망막 이미지를 분석합니다. 이 시스템은 전문가의 평가와 비교하여 92%의 민감도(sensitivity)와 88%의 특이도(specificity)를 유지하며, 특별히 참조가 필요한 DR을 100% 정확도로 탐지할 수 있음을 보여줍니다. 또한, 이미지 전처리를 위해 도입된 CLAHE는 지역별 대비를 향상시키고 이미지 품질 개선에 기여합니다.

- **Performance Highlights**: 연구 결과, 일반 인구의 당뇨병성 망막병증 유병률은 13.7%, 고혈당 환자의 경우 38.2%로 나타났습니다. AIDRSS의 뛰어난 성능은 다양한 인구에서 당뇨병성 망막병증을 정확히 식별하고 등급화하는데 효과적임을 입증합니다. 인공지능 기술을 통해 이 시스템은 자원이 부족한 환경에서도 신뢰성 있는 조기 진단 솔루션을 제공하여 당뇨병으로 인한 시각 손실의 부담을 경감할 수 있는 잠재력을 지니고 있습니다.



### Diffusion Models for Smarter UAVs: Decision-Making and Modeling (https://arxiv.org/abs/2501.05819)
Comments:
          7 pages, 2 figures

- **What's New**: 이번 논문은 무인 항공기(UAV) 통신에서 강화 학습(Reinforcement Learning, RL)과 디지털 트윈(Digital Twin, DT)을 결합하여 발생하는 다양한 문제를 해결하는 방안으로 새로운 생성 AI 기법인 확산 모델(Diffusion Models, DMs)의 통합을 탐구합니다. DMs는 기존의 방법론과는 달리, 데이터를 통해 학습된 확률 분포를 기반으로 신뢰할 수 있는 새로운 패턴을 생성할 수 있는 강력한 도구입니다. 이를 통해 데이터 부족 문제를 해결하고 RL과 DT의 성능을 개선하는데 기여할 수 있다는 점에서 주목받고 있습니다.

- **Technical Details**: UAV는 공공 안전, 에너지, 농업 및 스마트 시티와 같은 다양한 분야에서 사용됩니다. 이들은 5G 및 6G 네트워크의 필수 요소로, 기계 간 통신 및 고속 데이터 전송을 가능하게 합니다. 그러나 UAV 통신에서의 의사결정 과정은 복잡하며, RL은 낮은 샘플 효율성 때문에 이 과정에서 제한됩니다. DMs는 RL 모델의 샘플 효율성을 개선하고, 실제 훈련 환경을 생성하는 데 도움이 될 수 있습니다.

- **Performance Highlights**: 이 연구는 DMs가 RL 기반 UAV 통신을 어떻게 개선할 수 있는지를 보여주며, 이는 샘플 효율성을 높이고 정책 네트워크를 개선하며, 신뢰할 수 있는 훈련 환경을 생성하는 방식입니다. DT 시스템 내에서 DMs의 활용은 데이터 부족 문제를 해결하고, 의사결정 과정을 개선하며, 동적 모델링을 정교화 하는 데 중요한 역할을 할 것입니다. 이러한 통합은 UAV 통신에서의 적응력과 실시간 성능을 크게 향상시킬 것으로 기대됩니다.



### Real-Time Integrated Dispatching and Idle Fleet Steering with Deep Reinforcement Learning for A Meal Delivery Platform (https://arxiv.org/abs/2501.05808)
- **What's New**: 이번 연구는 식사 배달 플랫폼을 위한 효율적인 강화 학습 (reinforcement learning, RL) 기반의 전략적 이중 제어 프레임워크를 설계하는 것을 목표로 합니다. 이는 최적의 주문 배차 (order dispatching) 및 유휴 배달원 (courier) 유도 정책을 생성하기 위해 예측된 단기 수요 정보를 활용합니다. 또한, 이 프레임워크의 정책이 시스템 전반의 배달 효율성을 향상시키고 유휴 상태를 완화할 수 있을지에 대한 논의도 포함됩니다.

- **Technical Details**: 문제를 마르코프 결정 과정 (Markov Decision Processes)로 모델링하여 실시간으로 최적화를 수행하였다. 딥 강화 학습 (deep reinforcement learning, DRL) 프레임워크를 통해, 명시적으로 예측된 수요를 입력으로 활용하여 전략적 정책을 수립합니다. 이 이중 제어 프레임워크는 배차 정책과 유도 정책을 통합된 방식으로 반복적으로 훈련하여 현장에 적용 가능한 효율적인 결정을 제공합니다.

- **Performance Highlights**: 연구 결과, 강화 학습 기반의 전략적 이중 제어 프레임워크를 활용함으로써 배달 효율성과 배달원 간의 작업 부하 분배의 공정성이 개선되었으며, 서비스 네트워크 내의 공급 부족 문제도 완화되었습니다. 이러한 정책은 실시간으로 실행 가능하여 현장의 영향을 동시에 고려하는 결정을 내릴 수 있는 장점을 가집니다.



### Alignment without Over-optimization: Training-Free Solution for Diffusion Models (https://arxiv.org/abs/2501.05803)
- **What's New**: 이 연구에서는 Diffusion model의 목표를 정렬하는 데 어려움이 있었던 점을 해결하기 위해, Sequential Monte Carlo (SMC) 기반의 새로운 훈련 없는 샘플링 방법인 Diffusion Alignment as Sampling (DAS)를 제안합니다. DAS는 모델의 일반성을 유지하면서도 효과적인 보상 정렬을 달성합니다. 기존 방법들이 보상 최적화의 문제로 인해 성능이 저하되는 것을 방지하면서, 목표 보상을 효과적으로 샘플링할 수 있도록 설계되었습니다.

- **Technical Details**: DAS는 다수의 후보 latent를 활용하여 높은 보상 샘플로 유도하는 방식으로 구성되어 있습니다. 이를 통해 샘플링에서의 오류를 평균화하고, 보상 정렬된 목표 분포에서의 샘플링을 가능하게 합니다. 이 과정에서는 온도 조정(tempering) 기법을 사용하여 중간 목표 분포를 신중하게 설계함으로써 샘플링 효율성을 극대화하고 있습니다.

- **Performance Highlights**: DAS는 Stable Diffusion v1.5 및 CLIPScore와 같은 더 복잡한 보상 함수에 적용되며, 기존의 미세 조정 방법에 비해 뛰어난 성능을 입증하고 있습니다. 결과적으로, DAS는 단일 보상 최적화 및 다중 목표 최적화에서 새로운 Pareto front를 달성하였으며, 온라인 환경에서도 뛰어난 샘플링 능력을 보여주어 기존 방법들보다 20% 이상 개선된 성과를 달성했습니다.



### Robust Counterfactual Explanations under Model Multiplicity Using Multi-Objective Optimization (https://arxiv.org/abs/2501.05795)
Comments:
          19 pages

- **What's New**: 최근 기계 학습에서 설명 가능성(explainability)의 중요성이 증가하고 있습니다. 이 연구에서는 counterfactual explanation(CE)을 통해 새롭고 견고한 CE를 제안하며, Pareto improvement와 다중 목표 최적화(multi-objective optimization)를 활용하여 이를 구현합니다. 다양한 기계 학습 모델의 존재에 따른 문제를 해결하는데 초점을 맞추고 있으며, 안정적인 의사결정을 위한 기여를 목표로 합니다.

- **Technical Details**: 이 논문에서는 n 쌍의 스칼라와 r 차원의 벡터로 구성된 데이터 집합 𝒟 (𝑦𝑖, 𝑋𝑖) 설정을 통해 문제를 정립합니다. CE는 기계 학습 모델이 예측 결과를 얻기 위해 원본 데이터를 최소한으로 변경해야 하는 방법을 설명합니다. 다중 목표 최적화는 모델 간의 일관성 있는 CE를 생성하기 위해 필요하며, 다양한 조건을 고려하여 CE를 도출하는 방법이 제안됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 견고하고 유용함을 보여줍니다. 또한, 이 방법은 위험한 문제에 대해 안전한 솔루션을 선택하는 데 실용적으로 활용될 수 있습니다. 특히, 사용자의 선호에 따라 다양한 CE를 선택할 수 있는 가능성을 제공하며, 향후 의사결정 및 기계 학습 기반의 행동 계획 연구에도 큰 영향을 미칠 것으로 기대됩니다.



### UV-Attack: Physical-World Adversarial Attacks for Person Detection via Dynamic-NeRF-based UV Mapping (https://arxiv.org/abs/2501.05783)
Comments:
          23 pages, 22 figures, submitted to ICLR2025

- **What's New**: 최근 연구에서는 인간의 동적 행동을 모델링하는 Neural Radiance Fields (NeRF) 기반의 UV-Attack 방법이 소개되었습니다. 이 방법은 다양한 행동과 관점에서 인간 이미지를 생성할 수 있으며, 이전의 정적 이미지 기반 공격 방식보다 훨씬 더 높은 공격 성공률을 보여줍니다. 이 연구는 사람 탐지기(Person Detector)에 대한 효과적인 적대적 공격 가능성을 확대했습니다.

- **Technical Details**: UV-Attack은 동적 NeRF를 활용하여 UV 맵을 생성하고 텍스처를 실시간으로 수정하는 혁신적인 접근법을 사용합니다. 3D 인간 신체의 형상과 텍스처를 별도로 모델링하여, 학습 가능한 Densepose 기반의 UV 맵을 사용하여 3D 텍스처를 수정합니다. 이를 통해 unseen poses와 관점에서의 공격 성공률을 높이기 위해 Expectation over Pose Transformation (EoPT) 손실 함수를 사용합니다.

- **Performance Highlights**: UV-Attack은 FastRCNN 모델에 대해 92.75%의 공격 성공률을 기록하며, 최신 YOLOv8 탐지기에 대해 49.5%의 공격 성공률을 달성했습니다. 이 연구는 사람이 동적이고 다양한 움직임을 보일 때에도 효과적으로 작동하여 현실 세계에서의 공격 성공률을 높였습니다. 기존의 AdvCamou 공격이 28.50%의 공격 성공률을 보인 반면, UV-Attack은 그 성능을 크게 넘어섰습니다.



### Halal or Not: Knowledge Graph Completion for Predicting Cultural Appropriateness of Daily Products (https://arxiv.org/abs/2501.05768)
Comments:
          10 pages

- **What's New**: 이번 연구는 할랄 화장품의 상태를 예측하기 위한 새로운 프레임워크, HaCKG(할랄 화장품 추천 프레임워크)를 제안합니다. 이 방법은 화장품과 그 성분 간의 관계를 명시적으로 모델링하여 할랄 상태를 예측합니다. 기존의 접근 방식들이 개별 성분 분석에 초점을 맞춘 반면, HaCKG는 지식 그래프(knowledge graph)를 활용하여 복잡한 상호 관계를 포착합니다.

- **Technical Details**: HaCKG는 화장품 관련 지식을 그래프 형태로 구성하고, 이러한 지식을 바탕으로 Relational Graph Attention Network (r-GAT) 모델을 사용합니다. 이 모델은 화장품 성분의 구조적 관계를 학습하기 위해 Residual Connection을 포함하고 있습니다. 또한, Self-Supervised Learning (SSL) 방식으로 사전 훈련된 모델을 내려받아 할랄 상태를 예측하는 데 필요한 구체적인 데이터로 미세 조정(fine-tuning)됩니다.

- **Performance Highlights**: 다양한 화장품 데이터셋을 기반으로 한 광범위한 실험 결과, HaCKG는 기존의 최첨단 모델들보다 우수한 성능을 나타냈습니다. 할랄화장품 예측 과제에서 실험된 결과는 명확하며, 이는 기존 방법론의 한계를 극복하고 더 넓은 맥락을 고려한 결과입니다.



### Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models (https://arxiv.org/abs/2501.05767)
Comments:
          20 pages, 8 figures

- **What's New**: 이 논문에서는 복합적인 다중 이미지 시나리오에서 정밀한 그라운딩(grounding)을 달성하는 데 어려움을 겪고 있는 최근의 멀티모달 대형 언어 모델(MLLMs)의 발전을 다루고 있습니다. 새로운 Chain-of-Thought (CoT) 프레임워크를 통하여 단일 이미지 그라운딩과 다중 이미지 이해를 통합하며, Migician 모델을 소개하여 자유형식(free-form) 다중 이미지 그라운딩을 수행할 수 있는 최초의 모델로 자리잡았습니다.

- **Technical Details**: Migician 모델은 MGrounding-630k이라는 대규모 데이터셋으로 훈련되어, 다양한 다중 이미지 그라운딩 작업을 위한 데이터와 자유형식 지침 데이터를 포함합니다. 또한, 모델의 성능을 평가하기 위한 MIG-Bench라는 종합 벤치마크를 제시하여 다중 이미지 그라운딩의 성능을 정량적으로 측정하고 있습니다. 이를 통해 Migician은 이전의 MLLMs보다 21.61% 우수한 다중 이미지 그라운딩 능력을 입증하였습니다.

- **Performance Highlights**: Migician 모델은 실험 결과, 현존하는 최고의 MLLMs보다 훨씬 나은 성능을 기록하였으며, 특히 다양한 환경에서의 비전-언어 작업을 수행하는 데서 탁월한 능력을 보여주고 있습니다. 최종적으로, 이 연구는 MLLMs의 잠재력과 극복해야 할 도전을 탐구하고, Migician과 MGrounding-630k, MIG-Bench의 개발을 통해 다중 이미지 그라운딩 분야의 새로운 기준을 설정하고자 하였습니다.



### Element-wise Attention Is All You Need (https://arxiv.org/abs/2501.05730)
- **What's New**: 이번 연구에서는 전통적인 self-attention (SA) 메커니즘의 성능을 유지하면서도 훈련 및 추론에서의 복잡도를 낮출 수 있는 새로운 요소 단위 주의 메커니즘(element-wise attention mechanism)을 제안합니다. 이 메커니즘은 유클리드 거리의 제곱을 이용하여 유사성을 계산하며, Taylor 다항식을 사용하여 SA의 제곱 복잡도 항을 근사합니다.

- **Technical Details**: 제안된 요소 단위 주의 메커니즘은 훈련 중에 계산 복잡도가 \mathcal{O}(tLD)로, 긴 시퀀스 훈련에서 매우 계산적이고 메모리 효율성을 보여줍니다. 여기서 L은 시퀀스 길이, D는 특성 차원, t는 다항식의 최고 차수를 나타냅니다. 또한, 추론 단계에서는 순환 신경망(RecNN)으로 재형식화하여 \mathcal{O}(tD)의 추론 복잡도를 달성합니다.

- **Performance Highlights**: 제안된 요소 단위 주의는 기존 접근 방식에서 나타나는 성능 저하 요소를 피하면서도 SA와 유사한 성능을 달성합니다. 이는 인과적 및 비인과적 형태 모두에서 이루어지며, 학습 과정에서 더 나은 spikiness를 유지할 수 있도록 합니다.



### ExPO: Explainable Phonetic Trait-Oriented Network for Speaker Verification (https://arxiv.org/abs/2501.05729)
Comments:
          Accepted by IEEE Signal Processing Letters

- **What's New**: 이번 논문에서는 설명 가능한 음성 특성 지향 스피커 검증 모델, 즉 ExPO를 제안합니다. 이 모델은 문자열 검증에서 스피커의 음소 특성을 비교하는 접근 방식으로, 기존의 포렌식 음성 비교 방식과 유사한 설명 가능성을 제공합니다. ExPO는 발화 수준의 스피커 임베딩을 생성할 뿐만 아니라, 음성 특성의 세부 분석 및 시각화를 가능하게 하여 설명 가능한 스피커 검증 프로세스를 제공합니다.

- **Technical Details**: ExPO는 음소 특성 레이어를 스피커 모델에 통합하여 유사성 점수를 도출하는 방식을 설명합니다. 이 모델은 ECAPA-TDNN 아키텍처를 기반으로 하여, 프레임 시퀀스에서 음소 특성을 생성하고 이를 풀링하여 발화 수준의 스피커 임베딩을 얻습니다. 이 구조는 스피커의 음소 특성을 설명 가능하게 제공하며, 훈련을 통해 다양한 말의 세분화된 특성을 파악하고 저장합니다.

- **Performance Highlights**: ExPO의 성능을 평가하기 위해 각 미니배치에 K명의 스피커로부터 무작위로 선택된 발화 데이터를 사용합니다. 특성 비교를 통해 스피커 간의 유사성을 찾고 이를 기반으로 최적의 음소 특성을 결정하는 실험을 진행합니다. 이번 연구는 설명 가능한 스피커 검증을 위한 중요한 진전을 나타내며, 이러한 접근 방식은 향후 스피커 검증 시스템의 투명성과 신뢰성을 향상시키는 데 기여할 것입니다.



### Enabling Scalable Oversight via Self-Evolving Critic (https://arxiv.org/abs/2501.05727)
- **What's New**: 이번 논문에서는 SCRIT (Self-evolving CRITic)이라는 새로운 프레임워크를 제안하여, 대형 언어 모델(LLMs)의 비판 능력을 스스로 발전시키는 방법을 제시합니다. SCRIT는 인간 평가가 어려운 작업에서 LLM의 피드백 효율성을 높이는 것을 목표로 하며, 스스로 훈련하는 방식으로 비판 데이터를 생성합니다. 특히, 대조적 자기 비판 기법과 자기 검증 메커니즘을 통해 무인 감독으로 비판 품질을 향상시킵니다.

- **Technical Details**: SCRIT의 핵심 단계는 먼저 참조 솔루션을 바탕으로 학생 솔루션을 분석하고 비판하는 대조적 비판 기법을 개발하는 것입니다. 이어서 LLM은 생성된 비판이 수학적으로 유효한 솔루션으로 이어지는지를 자기 검증합니다. 이 두 단계는 고품질 비판 데이터를 생성하고, LLM의 비판 능력을 지속적으로 향상시키는 데 기여합니다.

- **Performance Highlights**: SCRIT은 Qwen2.5-72B-Instruct 모델을 기반으로 하여 비판 수정 및 오류 식별 작업에서 최대 10.3%의 성능 향상을 달성했습니다. 다양한 데이터 세트와 평가 프로토콜에서 일관된 개선 결과를 보여주며, SCRIT 구현 시 기존 모델의 출력을 크게 향상시켰습니다. 또한 SCRIT은 데이터와 모델 크기가 커질수록 성능이 긍정적으로 확장됨을 나타내어, LLM의 비판 능력 강화에 있어 중요한 진전을 보여줍니다.



### Zero-shot Shark Tracking and Biometrics from Aerial Imagery (https://arxiv.org/abs/2501.05717)
- **What's New**: 본 논문에서는 드론을 사용한 해양 동물 연구에서의 Machine Learning (ML)의 새로운 접근 방식으로 Frame Level ALIgment and tRacking (FLAIR) 시스템을 소개합니다. FLAIR는 Segment Anything Model 2 (SAM2)와 Contrastive Language-Image Pre-training (CLIP)의 비디오 이해 및 비전-언어 기능을 활용하여 드론 영상에서 특정 종의 분할 마스크를 자동으로 생성합니다. 특히, FLAIR는 레이블이 부착된 데이터 무 필요의 제로샷(zero-shot) 접근 방식을 도입하여 데이터 주석 작업과 기존 모델의 재훈련 없이 다양한 종에 일반화할 수 있습니다.

- **Technical Details**: FLAIR는 드론 비디오를 입력으로 받아 관심 있는 해양동물의 분할 마스크를 출력합니다. 본 시스템은 18,000개의 태평양 간호상어 이미지 데이터셋을 기반으로 훈련된 최첨단 객체 탐지 모델들과 성능을 비교하였으며, FLAIR는 다른 객체 탐지 시스템과 경쟁적인 성과를 나타내며 0.81의 Dice 점수를 기록했습니다. FLAIR의 혁신적인 점은 데이터를 주석 달 필요 없이 다양한 종으로 일반화할 수 있고, 새로운 휴리스틱과 결합하여 필요한 생체 정보를 자동으로 추출할 수 있다는 것입니다.

- **Performance Highlights**: FLAIR는 기존의 객체 탐지 모델을 압도적으로 능가하며, 인간-루프 방식의 두 가지 방법들과도 경쟁하는 성과를 냈습니다. 이 시스템은 전통적인 ML 워크플로우에 비해 인간의 노력과 전문 지식이 현저히 적게 요구되며, 더 높은 정확도를 달성할 수 있습니다. 따라서 FLAIR는 해양 생태계에 대한 해석과 통찰을 도출하는 데 더 많은 시간을 할애할 수 있게 합니다.



### How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond (https://arxiv.org/abs/2501.05714)
Comments:
          23 pages

- **What's New**: 이번 논문은 인공지능 모델들이 자율적 에이전트로 진화하고 있는 현상을 조명하며, 이를 바탕으로 인간-모델 협력을 체계적으로 정리한 최초의 리뷰 논문입니다. 특히 새로운 분류 체계를 도입하여 기존 접근 방식을 통합적으로 요약할 수 있는 틀을 제공합니다. 또한, 이 연구는 향후 더 심화된 연구로 나아갈 수 있는 기회를 모색하고 있음을 강조합니다.

- **Technical Details**: 저자들은 인간-모델 협력의 정의와 원칙을 다루며, 협력의 기초가 되는 공유 목표를 중심으로 논의합니다. 논문에서는 협력의 공식화를 위한 새로운 체계적 분류 방법론을 제안하며, 각각의 협력 유형에 따른 역할 프레임워크를 정의합니다. 이러한 프레임워크는 모델과 인간 간의 협력 방식과 의사결정의 책임이 어떻게 나뉘는지를 설명합니다.

- **Performance Highlights**: 인간-모델 협력을 통해 데이터 주석화, 정보 탐색, 창의적 글쓰기 및 실제 문제 해결 등의 다양한 NLP 작업에서 효율성을 높일 수 있는 잠재력이 드러났습니다. 이 연구는 상호작용을 위한 다양한 사용자 인터페이스를 제공함으로써 누적된 연구 결과를 종합하여 향후 연구 방향과 기술적 고려 사항을 제시하며, 이 분야의 발전에 기여할 것으로 기대됩니다.



### Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains (https://arxiv.org/abs/2501.05707)
Comments:
          22 pages, 13 figures, 7 tables; Project page at this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 자기 개선(self-improvement)을 위한 새로운 접근 방식을 제안합니다. 기존의 모델들을 다수의 에이전트로 구성하여 각 모델이 상호작용을 통해 독립적으로 전문화되는 방식입니다. 이 방법은 다수의 모델이 역할을 분담하여 응답의 다양성과 사고 체계를 유지하면서 더욱 많은 반복을 통한 성능 향상을 가능하게 합니다.

- **Technical Details**: 논문에서는 다중 에이전트 설정에서 LLM 모델을 전문화하는 방법을 도입합니다. 다중 에이전트 토론(multiagent debate) 방식을 통해 각 모델이 생성한 데이터를 활용하여 피드백을 주고받으며, 반복적 훈련을 통해 다양성을 유지하고 독립적으로 전문화합니다. 이 과정에서 데이터는 독립된 데이터 세트에서 수집되어 각 모델에 맞게 조정됩니다.

- **Performance Highlights**: 제안한 방법은 여러 추론 작업에 대해 정량적으로 유효성을 입증하며, 오픈 소스 LLM부터 상용 LLM에 이르기까지 다양한 모델에 적용 가능합니다. 실험 결과, 단일 에이전트 자기 개선 방법보다 훨씬 더 많은 반복을 통한 성능 개선을 확인했으며, 새로운 데이터 세트에서도 우수한 일반화를 보여주었습니다.



### EXION: Exploiting Inter- and Intra-Iteration Output Sparsity for Diffusion Models (https://arxiv.org/abs/2501.05680)
Comments:
          To appear in 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA 2025)

- **What's New**: 최근 몇 년간 확산 모델(difusion models)은 텍스트 프롬프트를 기반으로 다양한 다중 모드 출력(multi-modal outputs)을 생성하는 새로운 AI 솔루션으로 부각되었습니다. 그러나 이러한 모델은 반복적 구조로 인한 지나친 지연(latency)과 에너지 소모의 문제를 안고 있습니다. 본 논문에서는 EXION이라는 최초의 소프트웨어-하드웨어 공동 설계(SW-HW co-designed) 확산 가속기를 제안하여 이러한 계산 문제를 해결합니다.

- **Technical Details**: EXION은 두 가지 SW 수준의 최적화 방법을 기초로 합니다. 첫 번째로, FFN-Reuse 알고리즘을 통해 서로 다른 반복(iteration) 간에 FFN 레이어의 중복 계산을 식별하고 건너뛰는 inter-iteration sparsity를 구현합니다. 두 번째로, 수정된 eager prediction 방법을 도입하여 각 반복 내에서 attention score를 정확히 예측함으로써 불필요한 계산을 생략하는 intra-iteration sparsity를 성취합니다. 이와 함께, sparse matrix를 압축하고 병합하는 새로운 데이터 압축 메커니즘인 ConMerge를 통해 하드웨어 활용도를 높입니다.

- **Performance Highlights**: EXION의 효율성을 검증하기 위해 다양한 다중 모드 확산 모델에서 정확도에 미치는 영향이 없음을 확인하였고, 서버와 엣지 레벨 설정에서 EXION을 구현하여 성능을 비교하였습니다. 그 결과, EXION은 서버 GPU(NVIDIA RTX 6000 Ada) 대비 3.2-379.3배 향상된 성능과 45.1-3067.6배 증가된 에너지 효율성을 보였으며, 엣지 GPU 대비 42.6-1090.9배의 성능 및 196.9-4668.2배의 에너지 효율성 개선을 기록하였습니다.



### Network Diffuser for Placing-Scheduling Service Function Chains with Inverse Demonstration (https://arxiv.org/abs/2501.05673)
Comments:
          Accepted to IEEE INFOCOM 2025

- **What's New**: 본 논문은 Service Function Chain (SFC) 최적화 문제를 해결하기 위한 혁신적인 네트워크 디퓨저(network diffuser)를 제안합니다. 이는 조건부 생성 모델링(conditional generative modeling)을 활용하여 SFC 배치 및 스케줄링을 동시에 최적화할 수 있도록 설계되었습니다. 기존의 SFC 최적화 방법들이 NP-hard 문제에 대한 적절한 데이터 부족으로 어려움을 겪는 가운데, 저자들은 무작위로 생성된 솔루션을 바탕으로 SFC 최적화 문제를 효과적으로 해결하는 대안적인 접근 방식을 소개합니다.

- **Technical Details**: SFC 최적화 문제는 두 가지 밀접하게 연결된 문제인 SFC 배치(SFC placement)와 SFC 스케줄링(SFC scheduling)을 동시에 고려해야 합니다. 저자들은 각 상태(state)가 네트워크의 현재 상태와 SFC의 스케줄을 모두 포함하는 상태 시퀀스 생성 문제로 SFC 최적화를 정식화합니다. 이를 통해 그래프 디퓨전(graph diffusion)을 수행하고, 주어진 조건을 바탕으로 최적의 SFC 배치와 스케줄을 생성할 수 있는 모델을 학습합니다.

- **Performance Highlights**: 제안된 네트워크 디퓨저는 여러 가지 휴리스틱 및 딥러닝 기반의 방법들과 비교했을 때, SFC 보상에서 평균 약 20% 향상된 성능과 SFC 대기 시간 및 차단률에서 약 50% 감소를 보여줍니다. 이러한 성능 향상은 무작위 생성 결정을 통해 전문적인 학습 데이터를 생성하는 비대칭 모델의 효과를 통해 입증되었습니다. 이 연구는 네트워크에서 SFC 최적화 문제를 해결하기 위한 새로운 방향을 제시합니다.



### TransPlace: Transferable Circuit Global Placement via Graph Neural Network (https://arxiv.org/abs/2501.05667)
Comments:
          Accepted at KDD 2025

- **What's New**: 이 논문은 TransPlace라는 새로운 글로벌 배치(framework)를 소개합니다. 이는 혼합 크기의 수백만 개의 셀을 연속 공간에 배치하기 위해 학습하는 방식을 도입합니다. 기존의 글로벌 플레싱에서는 각 회로 설계를 개별적으로 해결하는 한계를 지니고 있었으나, TransPlace는 이를 극복하여 효율성과 성능을 향상시키고자 합니다.

- **Technical Details**: TransPlace는 여러 기술적 요소로 구성됩니다. 첫째, Netlist Graph를 통해 회로의 토폴로지를 모델링합니다. 둘째, Cell-flow와 상대 위치 인코딩을 도입하여 SE(2)-불변의 표현을 학습하고, 셋째, 맞춤형 그래프 신경망 아키텍처인 TPGNN을 통해 배치 지식을 파라미터화합니다. 마지막으로, 투 단계 전략을 통해 거칠고 세밀한 배치를 조정합니다.

- **Performance Highlights**: TransPlace는 기존의 최신 배치 방법에 비해 1.2배의 속도 향상과 30%의 혼잡도 감소, 9%의 타이밍 향상 및 5%의 배선 길이 감소를 실현한다고 보고합니다. 이는 고품질 배치에서 훈련된 TransPlace가 이전의 해결책에 비해 현저한 성능 개선을 보여줌을 의미합니다.



### Learning to Measure Quantum Neural Networks (https://arxiv.org/abs/2501.05663)
Comments:
          Accepted by ICASSP 2025 Workshop: Quantum Machine Learning in Signal Processing and Artificial Intelligence

- **What's New**: 이번 연구에서는 새로운 양자 기계 학습(QML) 모델 접근 방식을 제안합니다. 기존의 고정된 측정 가능성을 활용한 모델들이 아닌, 가변적인 Hermitian matrix를 사용하여 QML 모델의 성능을 개선할 수 있는 방안을 제시합니다. 특히, 제안된 방법은 측정 과정을 최적화하여 효율적인 학습이 가능하도록 하여, 마지막 결과 성능을 높이는 데 기여합니다.

- **Technical Details**: 제안된 접근법에서는 Hermitian 행렬을 파라미터화하여 양자 회로의 일반적인 파라미터와 동시에 훈련합니다. 이를 통해, Variational Quantum Circuits(VQCs)의 결과를 향상시키기 위한 학습 가능한 관측 변수를 자동으로 발견하는 방법론을 구현합니다. 수치 시뮬레이션 결과에 따르면, 본 프레임워크는 고정된 관측 가능성을 이용한 일반적인 VQC 훈련보다 더 우수한 성과를 보여줍니다.

- **Performance Highlights**: 우리는 다양한 기계 학습 작업에서 높은 분류 정확도를 달성하여 QML 모델의 전반적인 성능을 증대시켰습니다. Hermitian 행렬의 스펙트럼 범위를 넓힘으로써 VQC의 출력 범위를 확장할 수 있음을 입증하였습니다. 이 연구 결과는 QML이 다양한 문제를 더욱 효과적으로 해결할 수 있는 가능성을 열어줍니다.



### Cascaded Self-Evaluation Augmented Training for Efficient Multimodal Large Language Models (https://arxiv.org/abs/2501.05662)
- **What's New**: 최근 효율적인 다중 모달 대형 언어 모델(EMLLMs)이 급속히 발전하였습니다. 본 논문에서는 CoT(Chain-of-Thought) 추론과 단계별 자기 평가(self-evaluation)를 통합하여 EMLLM의 성능을 향상시킨 방법을 제안합니다. 하지만 제한된 매개변수 때문에 EMLLM은 추론하는 동안 자기 평가를 효과적으로 이용하기 어려운 상황입니다. 이를 해결하기 위해 우리는 Self-Evaluation Augmented Training (SEAT)이라는 방법을 소개하며, 후속 연구를 위한 중요한 데이터셋을 구성하였습니다.

- **Technical Details**: SEAT는 더 강력한 EMLLM을 이용하여 CoT 추론 및 데이터를 생성하고, MLLMs가 선별한 데이터를 사용하여 EMLLM을 교육합니다. 그러나 긴 프롬프트를 처리하고 CoT 추론의 품질을 유지하는 데 문제가 발생합니다. 이를 해결하기 위해 카스카데드 자기 평가 향상 훈련(Cas-SEAT) 방법을 제안하며, 긴 입력을 짧고 구체적인 작업 프롬프트로 나누어 비용을 절감하는 방식입니다. 이 과정에서 7B 매개변수의 오픈소스 EMLLM을 사용하여 데이터의 효율성을 극대화하였습니다.

- **Performance Highlights**: 실험 결과, Cas-SEAT는 기존 방법들과 비교하여 EMLLM의 자기 평가 능력을 크게 향상시켰으며, MathVista, Math-V 및 We-Math 데이터셋에서 각각 19.68%, 55.57%, 46.79% 개선된 성과를 보였습니다. 이는 EMLLM의 CoT 추론 능력을 효과적으로 강화한 결과로, 적극적인 후속 연구를 위한 기초 자료로 활용될 것입니다. 또한 Cas-SEAT Dataset은 EMLLM의 자기 평가 향상을 위한 첫 번째 데이터셋으로, 저비용으로도 효과적인 연구에 기여할 수 있습니다.



### Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation (https://arxiv.org/abs/2501.05647)
Comments:
          Published on KDD'25: Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2025

- **What's New**: 대규모 언어 모델(LLMs)과 소형 추천 모델(SRMs)의 통합을 통한 새로운 추천 프레임워크인 LSC4Rec가 제안되었습니다. 이 프레임워크는 기기-클라우드 협력을 기반으로 하여 두 모델의 장점을 결합하여 실시간 사용자 선호도를 보다 효과적으로 반영할 수 있습니다. 또한, 협업 훈련, 협업 추론, 지능형 요청 등 세 가지 전략을 설계하여 실용성을 향상시키고 있습니다.

- **Technical Details**: LSC4Rec는 LLM이 아이템 후보 목록을 생성하고, SRM이 사용자의 실시간 행동을 아우르는 데이터로 펑션을 수행하여 아이템 재순위를 매기는 구조로 되어 있습니다. LLM의 메모리와 SRM의 실시간 데이터 접근 용이성을 통해 훈련 및 추론 빈도를 줄이고, 클라우드와 기기 간 통신 비용을 최소화 합니다. 이러한 고도화된 전략은 두 모델 간의 시너지 효과를 극대화합니다.

- **Performance Highlights**: LSC4Rec의 여러 전략들이 효과적임을 입증하기 위한 포괄적이고 광범위한 실험 분석이 실시되었습니다. 실험 결과는 LSC4Rec의 다양한 LLM 및 SRM에 대한 실효성을 입증하였으며, 추천 시스템의 사용자 경험을 향상시키는 데 큰 기여를 할 것으로 기대됩니다. 이 연구는 실시간 추천 문제에 만연한 문제를 해결할 유망한 방향을 제시하고 있습니다.



### Efficient Representations for High-Cardinality Categorical Variables in Machine Learning (https://arxiv.org/abs/2501.05646)
Comments:
          2025 International Conference on Advanced Machine Learning and Data Science (AMLDS 2025)

- **What's New**: 이 논문은 고차원 범주형 변수를 효과적으로 인코딩하기 위한 새로운 기술들을 소개하고 있습니다. 전통적인 one-hot encoding의 한계를 극복하기 위해, means encoding, low-rank encoding, multinomial logistic regression encoding과 같은 혁신적인 인코딩 기법이 제안됩니다. 이러한 기법은 범주형 데이터의 압축되고 정보성 있는 임베딩을 생성할 수 있도록 설계되었습니다. 이로 인해 모델 성능과 계산 효율성을 크게 향상시킬 수 있습니다.

- **Technical Details**: 논문에서는 그룹 멤버십을 변환하기 위한 효과적인 매핑 ψ(psi)를 개발하였습니다. 이 매핑은 범주형 변수 Gi(Gi)와 결과 변수 Yi(Yi)의 관계를 수학적으로 나타내며, 이를 통해 p+k 차원의 실수값 특성(Xi, ψ(Gi))을 갖는 표준 회귀 문제로 단순화됩니다. 필요한 정보를 추출하는 주 요소 사항은 기본 레마로 설명되며, 추가적으로 이를 기반으로 한 다양한 방법들이 제안됩니다. 또한, 충분한 잠재 상태 가정(sufficient latent state assumption)을 통해 그룹 멤버십과 결과 간의 간접적 관계를 정의하고 있습니다.

- **Performance Highlights**: 제안된 인코딩 기법들은 다양한 데이터셋에서 평가되었으며, 기존의 기준 방법들에 비해 모델 성능과 계산 효율성에서 최고의 개선 효과를 보였습니다. 특히 대규모 데이터셋이 요구되는 분야에서 이러한 기술들이 유용하게 활용될 수 있음을 입증하였습니다. 결과적으로, 이 연구는 머신 러닝에서 더욱 강력하고 효율적인 애플리케이션으로 나아가는 길을 열어주고 있습니다.



### Iconicity in Large Language Models (https://arxiv.org/abs/2501.05643)
Comments:
          Supplementary information: this https URL

- **What's New**: 이번 논문은 인공 언어에서 소리와 의미 간의 직접적인 관계인 lexical iconicity(어휘 상징성)를 다루고 있습니다. 기존에 소리와 의미의 처리 방식에서 인간과의 차이를 탐구하며, LLMs가 그 관계를 처리하는 방법이 부족하거나 상당히 다를 것이라는 가설을 세웠습니다. 연구에서는 GPT-4가 생성한 pseudowords(유사 단어)의 의미를 인간과 LLM 기반 참가자들이 추측할 수 있는 능력을 비교하여, LLM이 이 iconicity를 잘 처리할 수 있는 가능성을 제시합니다.

- **Technical Details**: 레미니션은 '의미와 형태의 직접적인 관계 
'로 정의되며, 이와 관련된 복잡한 세부 사항으로써 LLMs의 경우 의미와 음성을 직접적으로 처리하지 못하고 있습니다. LLM의 훈련 데이터는 오디오 녹음을 포함하지 않으며, 이는 그들이 iconicity를 전통적으로 접근하는 방법으로는 성취하기 어렵다는 점을 의미합니다. 연구에서는 두 단계의 실험을 설계하였고, 첫 번째 단계에서는 GPT-4가 고유한 소리-상징적 속성을 갖는 인공 언어의 lexicon을 생성했습니다.

- **Performance Highlights**: 연구 결과, 인간 참가자들은 생성된 iconic language에서의 pseudoword 의미 예측을 자연어보다 훨씬 더 정확하게 수행했습니다. LLM 기반 참가자들은 인간보다 더욱 높은 성공률을 보였으며, 이는 LLM이 다양한 유형의 cognition(인지)적 처리와 인간 유사한 학습을 효과적으로 시뮬레이션할 수 있음을 보여줍니다. 이러한 결과는 LLMs가 iconicity를 학습하고 처리할 수 있는 능력을 시사합니다.



### The Impact of Model Scaling on Seen and Unseen Language Performanc (https://arxiv.org/abs/2501.05629)
Comments:
          Accepted at SEAS Workshop at AAAI25

- **What's New**: 이 연구는 204개의 언어에서 다국어(Large Language Model, LLM)의 성능과 확장 비율(scaling behavior)을 상세히 조사한 첫 번째 연구입니다. 특히, 사전 훈련(pretraining) 중에 본 언어와 보지 못한 언어에 대한 성능 차이를 분석했습니다. 또한 다양한 자원 수준(resource levels)과 모델 크기에 따른 성능 현상을 다루고 있습니다. 이 연구는 다국어 LLM의 효과성을 이해하는 데 중요한 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 xglm, bloom 및 bloomz 모델을 포함하여 총 14개의 다국어 LLM을 조사했습니다. 다양한 형태의 모델 크기를 포함하여, 주제 분류(topic classification)와 기계 번역(machine translation) 작업을 수행하며 두 가지 유형의 설정(제로샷(zero-shot) 및 몇 샷(few-shot))에서 성능을 평가했습니다. SIB-200 및 FLORES-200 데이터셋을 활용해 204개 언어를 대상으로 성능을 비교했습니다.

- **Performance Highlights**: 제로샷 설정에서는 모델 크기가 성능에 큰 영향을 미치지 않는 반면, 두 샷 상황에서는 대형 모델이 분명한 성능 개선을 보였습니다. 다국어 텍스트 분류에서는 자원 수준이 인식된 언어와 보지 못한 언어 모두에서 성능에 더 큰 상관관계를 가지는 것으로 나타났습니다. 전반적으로, 자료 수준이 모델의 성능을 예측하는 데 중요하게 작용하며 다국어 LLM의 효과성과 그 운영 전략에 대한 통찰력을 제공합니다.



### Watermarking Graph Neural Networks via Explanations for Ownership Protection (https://arxiv.org/abs/2501.05614)
- **What's New**: 이번 연구는 Graph Neural Networks (GNNs)의 소유권 보호를 위한 새로운 수단인 설명 기반 워터마킹 기법을 제시합니다. 기존의 백도어 기반 방법들과 달리, 이 방법은 데이터 오염을 피하고 소유권 모호성을 제거하는 데 중점을 두고 있습니다. 연구팀의 접근 방식은 특정 서브그래프의 설명을 선택된 워터마크와 정렬시켜, 소유권을 통계적으로 입증할 수 있도록 합니다.

- **Technical Details**: 연구팀은 GNN 모델을 훈련시키기 위해 이중 목표 손실 함수를 사용하여, 표준 분류 손실을 최소화함과 동시에 워터마크와 각 서브그래프의 설명 간 거리도 줄여 나갑니다. GNN의 노드 기능이 예측에 미치는 영향을 추정하기 위해 가우시안 커널 행렬을 사용하며, 이를 위해 릿지 회귀(ridge regression)를 통한 단일 단계의 계산을 활용하여 효율성을 높였습니다. 이 접근법은 워터마크의 존재가 통계적으로 불가능할 만큼 독창적인 설명을 생성합니다.

- **Performance Highlights**: 연구는 다양한 벤치마크 그래프 데이터셋과 GNN 모델에 대한 실험을 통해, 제안된 방법이 미세 조정(fine-tuning) 및 가지치기(pruning)와 같은 워터마크 제거 공격에 강력하다는 것을 입증하였습니다. 또한, 본 연구는 소유권 증거를 제공하며, 최악의 경우 적대자가 워터마크를 식별하는 것이 NP-hard 문제임을 이론적으로 증명하였습니다. 이러한 결과는 GNN의 지적 재산 보호에서 중요한 진전을 나타냅니다.



### Advancing Personalized Learning Analysis via an Innovative Domain Knowledge Informed Attention-based Knowledge Tracing Method (https://arxiv.org/abs/2501.05605)
- **What's New**: 이 논문에서는 지식 추적(Knowledge Tracing, KT) 모델의 혁신적인 주의 기반 방법을 제안하여 교육 과정 내 지식 개념 경로를 효과적으로 통합했습니다. 기존 모델들이 단기적인 상호작용에만 초점을 맞추었던 점을 개선하기 위해, 지식 개념 간의 계층적 의존성을 포착할 수 있는 새로운 방법론을 도입했습니다. 이를 통해 학생들의 학습 성과를 더욱 향상시킬 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 Learning Relevance Matrix라는 새로운 메커니즘을 도입하여 질문 간의 관계를 추적하고, 관련 없는 질문에 대한 주의 점수를 마스킹(masking)하는 방식으로 주의 메커니즘을 개선했습니다. 이를 통해 모델이 구체적인 지식 개념 경로를 기반으로 하여 학생의 학습에 더욱 적합한 예측을 할 수 있도록 합니다. 제안하는 방법은 계층적 의존성을 효과적으로 캡처하여 기존 선진 기술보다 향상된 성능을 발휘하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방식이 AUC(Area Under Curve)와 정확성에서 기존 SOTA(Sate-of-the-Art) KT 방법들에 비해 유의미한 개선이 이루어졌음을 보여주었습니다. 또한, 제안한 방법은 학생의 학습 과정을 보다 구체적으로 해석할 수 있는 가능성을 제공하여, 개인화된 학습 시스템의 미래 연구 방향에 기여할 수 있을 것으로 기대하고 있습니다.



### Approximate Supervised Object Distance Estimation on Unmanned Surface Vehicles (https://arxiv.org/abs/2501.05567)
- **What's New**: 이 연구는 자율 수상 차량(USV)에서의 거리 추정을 위한 새로운 접근 방식을 제시합니다. 기존의 거리 측정 기술에 대한 비용 효율적이고 직관적인 대안으로, 수동으로 주석이 달린 이미지와 거리 측정값을 포함하는 데이터를 수집하였습니다. 제안된 방법은 물체 감지와 거리 예측을 동시에 수행하며, 이는 인간의 추정 능력과 더 일치합니다. 이를 통해 환경 내에서 가까운 물체에 대한 경고 시스템을 구현할 수 있습니다.

- **Technical Details**: 이 방법은 YOLOv7 및 YOLOv9 객체 감지 모델을 활용하여 각 물체의 거리를 직접 예측하도록 수정된 아키텍처를 사용합니다. 추가 출력을 사용하여 검출된 물체와의 거리를 예측하며, 다양한 거리 분포에 대한 정규화 전략을 실험했습니다. 단순히 거리를 예측하기보다는, 예측값을 안정적으로 유지하기 위한 전략들이 필요하며, 이를 통해 학습 안정성을 확보합니다.

- **Performance Highlights**: 제안한 방법은 다양한 객체 추적기와 결합하여 성능을 분석하였으며, 실제 해양 환경에서의 실험으로 효과를 입증하고자 하였습니다. 고유한 해양 문제를 해결할 수 있으며, 비용이 많이 드는 다른 센서를 사용하지 않고도 향상된 거리 추정을 가능하게 합니다. 이 연구는 기존 데이터셋의 한계를 극복하고, 해양 컴퓨터 비전 분야에 기여할 수 있는 방법론을 제시합니다.



### Vision-Language Models for Autonomous Driving: CLIP-Based Dynamic Scene Understanding (https://arxiv.org/abs/2501.05566)
- **What's New**: 이 논문에서는 자동 운전 차량(AV)의 결정을 위한 인간 중심 설명을 생성하고, 운전 비디오 분석에 인공지능(AI)을 활용하기 위해 동적 장면 검색 시스템을 개발하였습니다. 이 시스템은 Contrastive Language-Image Pretraining (CLIP) 모델을 사용하며, 엣지 장치(edge devices)에서 실시간 배치 최적화가 가능합니다. 특히 복잡한 시나리오에서 GPT-4o의 제로샷(zero-shot) 능력을 포함한 최신 인-context learning 방법들을 능가합니다.

- **Technical Details**: 이 연구는 Honda Scenes Dataset을 활용하여 프레임 레벨(frame-level) 분석을 진행하였습니다. 이 데이터셋은 약 80시간 분량의 주행 비디오를 포함하고 있으며, 다양한 실제 도로 및 날씨 조건을 담고 있습니다. CLIP 모델들은 자연어(supervision)로부터 시각적 개념을 학습하는 데 뛰어난 강건성을 보여주었으며, ViT-L/14와 ViT-B/32와 같은 클립 모델의 미세 조정(fine-tuning)을 통해 장면 분류에서 F1 점수를 91.1%로 달성하였습니다.

- **Performance Highlights**: 이 시스템은 정밀하고 신속한 장면 인식을 제공할 수 있는 능력을 입증하였으며, 이는 고급 운전 보조 시스템(ADAS)의 필수 요구 사항을 충족하는 데 기여할 수 있습니다. CLIP 모델이 동적 장면 이해 및 분류를 위한 확장 가능하고 효율적인 프레임워크를 제공하는 잠재력을 보여주고 있습니다. 또한, 이 연구는 운전자의 행동, 도로 조건, 안전과 관련된 시나리오에 대한 이해를 깊게 하여 보다 스마트하고 안전한 자율 주행 시스템 개발을 위한 기초를 다졌습니다.



### Soup to go: mitigating forgetting during continual learning with model averaging (https://arxiv.org/abs/2501.05559)
- **What's New**: 본 논문에서는 Sequential Fine-tuning with Averaging (SFA)라는 새로운 방법을 제안하여 연속 학습(continual learning)에서의 치명적인 망각(catastrophic forgetting)을 줄이는 접근 방식을 소개합니다. 이전 모델의 체크포인트와 현재 학습 중인 모델을 주기적으로 평균하여, 과거 지식을 유지하며 새로운 작업에서의 성능을 개선할 수 있습니다. 이 방법은 저렴한 계산 비용으로 과거 데이터를 저장할 필요 없이도 우수한 결과를 보여줍니다.

- **Technical Details**: SFA는 L2 회귀(L2-regression)에 영감을 받아 설계되었으며, 훈련 중 현재 작업의 모델과 이전 작업의 체크포인트를 통합하여 새로운 작업에 대해 계속 fine-tuning을 진행합니다. 모델 평균 주기(p) 개념을 도입하여, 훈련 중 평균화를 조절함으로써 과거 작업의 성능과 새로운 작업의 성능 간의 균형을 이룹니다. 기존의 데이터 버퍼를 사용하지 않고도 이전 체크포인트를 과거 데이터의 대리로 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, SFA는 이미지 및 언어 작업에서 기존의 데이터 버퍼를 사용하는 방법과 유사한 성능을 달성하면서도 추가 데이터를 저장하지 않고 다른 모델 병합 기법보다 우수한 결과를 보였습니다. 또한, SFA는 기존의 연속 학습 방법들과 비교했을 때 모든 작업에서 일관되게 뛰어난 성능을 보여주었습니다. 이로써 SFA는 효율적인 연속 학습을 위한 신규 솔루션으로 자리 잡을 가능성을 나타냅니다.



### Improving Zero-Shot Object-Level Change Detection by Incorporating Visual Correspondenc (https://arxiv.org/abs/2501.05555)
- **What's New**: 이번 논문에서는 두 이미지 간의 객체 수준 변화 감지에 대한 새로운 접근법을 제시합니다. 기존의 방법들이 가지는 세 가지 주요 한계를 해결하기 위해, 변화 상관 관계를 활용하여 변화 감지 정확도를 향상시키고, 테스트 시 허위 긍정률을 줄입니다. 이 방법은 객체의 추가 또는 제거 위치에 대한 감독 라벨을 활용하여 정확성을 크게 개선하였으며, 관계 예측을 위한 새로운 데이터셋 OpenImages-Inpainted를 도입했습니다.

- **Technical Details**: 연구 필드에서는 일반적으로 객체의 변화를 감지하는 데 있어 여러 가지 변형(예: 카메라 각도 변화, 색 변화, 조명 변화)에 대한 문제를 겪습니다. 본 논문에서 제안하는 방법은 양쪽 이미지 간의 변화를 비교하고, 헝가리안 알고리즘(Hungarian algorithm)을 활용해 예측된 변경 사항 간의 대응 관계를 예측하는 시스템입니다. 이를 통해 허위 긍정률을 최소화하고 더 높은 변화 감지 정확도를 달성합니다.

- **Performance Highlights**: 이 모델은 변화를 감지하는 정확도에서 기존 방법들보다 뛰어난 성능을 보이며, 여러 벤치마크에서 최첨단 결과를 얻었습니다. 예를 들어, 제안하는 contrastive matching 손실 함수는 모든 다섯 개의 벤치마크에서 정확도를 +1.31에서 +6.56로 향상시켰습니다. 또한, 변경 감지 성능을 평가하기 위한 새로운 메트릭을 제안하여 상대적인 성과를 일관되게 비교할 수 있게 했습니다.



### LLMQuoter: Enhancing RAG Capabilities Through Efficient Quote Extraction From Large Contexts (https://arxiv.org/abs/2501.05554)
- **What's New**: 본 논문에서는 LLMQuoter라는 경량화된 distillation 기반 모델을 소개합니다. 이 모델은 Retrieval Augmented Generation (RAG) 프로세스를 향상시켜 중요 텍스트 증거를 추출하고, 이를 통해 하위 작업의 추론 성능을 개선합니다. LLMQuoter는 LLaMA-3B 아키텍처를 바탕으로 하며, 15,000개의 HotpotQA 샘플에 대해 Low-Rank Adaptation (LoRA)을 사용하여 세밀하게 조정되었습니다.

- **Technical Details**: LLMQuoter는 'quote-first-then-answer' 전략을 채택하여 중요한 인용구를 식별하고, 이를 추출하여 추론 모델에 전달합니다. 이 접근법은 전통적인 전체-context 기법과는 달리, 각 단계가 분리되어 추론 과정을 간소화하고, 인지 부담을 줄입니다. 이를 통해 LLMQuoter는 작은 모델과 큰 모델 모두에서 뛰어난 정확도를 달성할 수 있습니다.

- **Performance Highlights**: 실험 결과 LLMQuoter는 RAFT와 같은 RAG 기법들에 비해 경쟁력 있는 정확도 향상을 보여줍니다. 또한 LLMQuoter의 경량화된 성질은 자원 제한이 있는 연구자와 실무자들에게도 고급 RAG 기능에 대한 접근을 민주화합니다. 이는 복잡한 작업을 간소화하고, 자원 활용성을 높이는 데 기여합니다.



### The dynamics of meaning through time: Assessment of Large Language Models (https://arxiv.org/abs/2501.05552)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 개념의 역사적 맥락과 의미의 발전을 어떻게 이해하는지를 평가하는 데 중점을 두고 있습니다. 다양한 시대에서 용어를 해석하는 능력을 분석하기 위해 여러 모델을 비교하고, 특정한 과제를 통해 모델의 반응을 측정합니다.

- **Technical Details**: 연구팀은 다양한 도메인에서 선택한 용어 세트를 분석하며, 맞춤형 프롬프트(prompts)와 객관적인 메트릭스(metrics, 예: perplexity 및 단어 수) 뿐만 아니라 주관적인 전문가 평가를 통해 반응을 측정했습니다. 주목할 만한 모델들인 ChatGPT, GPT-4, Claude, Bard, Gemini, Llama 등이 비교 분석되었습니다.

- **Performance Highlights**: 각 모델이 역사적 맥락과 의미의 변화에 대해 다르게 반응하는 것을 발견하였고, 이는 각 모델의 강점과 한계를 부각시킵니다. 이러한 통찰력은 대규모 언어 모델을 개선하고, 역사적 텍스트 분석, AI 설계 및 디지털 인문학의 응용에 기여할 수 있는 기초를 제공합니다.



### OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? (https://arxiv.org/abs/2501.05510)
Comments:
          28 pages

- **What's New**: 이번 논문에서는 OVO-Bench(Online-VideO-Benchmark)를 소개하며, 이를 통해 시간 정보를 기반으로 하는 온라인 비디오 이해 능력을 평가할 수 있는 새로운 벤치마크를 제시합니다. 기존의 오프라인 모델과 달리, OVO-Bench는 비디오 스트림을 실시간으로 처리하고, 다양한 시나리오에 따라 정확한 응답을 생성하는 능력을 강조합니다. 이러한 평가를 통해 모델이 실제 비디오 이해 작업에서 어떻게 성능을 발휘하는지를 파악할 수 있는 기회를 제공합니다.

- **Technical Details**: OVO-Bench는 644개의 다양한 비디오를 포함하며, 총 12개의 작업을 통해 모델의 시간 인식 능력을 평가합니다. 주요 작업으로는 과거 사건 추적, 실시간 사건 이해, 미래 정보에 따른 적시 응답이 포함되며, 이를 통해 비디오 LLMs의 성능을 보다 정밀하게 측정할 수 있습니다. 또한, 자동 생성 파이프라인과 인간 검증을 결합하여 약 2,800개의 메타 주석을 생성하였습니다.

- **Performance Highlights**: 실험 결과, 9개의 비디오 LLMs는 오프라인 성능에서 우수한 평가를 받았으나, 온라인 환경에서의 질의에는 제 기능을 발휘하지 못하는 경향을 보였습니다. 특히, 최신 스트리밍 모델인 Flash-VStream의 경우 오프라인 모델에 비해 더욱 큰 성능 차이를 보였으며, 이는 온라인 비디오 이해 기술 개선을 위한 연구의 필요성을 강조합니다. OVO-Bench는 이러한 연구를 촉진하는 데 기여할 것으로 기대됩니다.



### Spatial Information Integration in Small Language Models for Document Layout Generation and Classification (https://arxiv.org/abs/2501.05497)
Comments:
          8 pages. Symposium on Applied Computing 2025

- **What's New**: 이번 논문에서는 문서 레이아웃 이해(Document Layout Understanding) 분야에서 새로운 방법론을 제안합니다. 반 구조화된 데이터(semi-structured data) 부족 문제를 해결하기 위해 합성 레이아웃 정보(synthetic layout information)를 생성하는 방식을 도입했습니다. 이 방식은 기존의 LayoutTransformer와 비교했을 때, 더 나은 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 문서의 공간적 배치(spatial arrangement) 분석을 중심으로 진행되어 여타 모델들보다 우수한 결과를 도출하였습니다. 연구에서 제안된 방법은 머신러닝 모델 교육을 위한 공개 데이터셋의 부족을 극복할 수 있는 가능성을 내포하고 있습니다. 특히, 경계 상자 정보(bounding box information)가 텍스트 분류(text classification)에 긍정적인 영향을 미칠 수 있다는 점도 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 LayoutTransformer보다 우수한 성과를 기록하였으며, 이는 반 구조화된 문서의 인식 성능을 향상시킬 수 있음을 시사합니다. 또한, 문서의 다양한 레이아웃 구조를 이해함으로써 실생활에서 접하는 회계 보고서, 구매 주문서 및 영수증 등의 문서 처리가 개선될 수 있다는 점도 강조되고 있습니다.



### FedSA: A Unified Representation Learning via Semantic Anchors for Prototype-based Federated Learning (https://arxiv.org/abs/2501.05496)
Comments:
          Accepted by AAAI2025

- **What's New**: 이 논문은 Prototype 기반의 Federated Learning(FL)에서 클라이언트 간의 데이터 이질성 문제를 해결하기 위한 새로운 프레임워크인 Federated Learning via Semantic Anchors (FedSA)를 제안합니다. 기존 방법들은 로컬 모델에서 직접 프로토타입을 수집하여 비일관성을 초래하는 문제가 있었으나, FedSA는 간단하면서도 효과적인 Semantic Anchors를 도입하여 이 문제를 해결하고자 합니다. 이를 통해 로컬 대표 학습과 프로토타입 생성을 분리하여 클라이언트들이 일관된 표현 학습을 할 수 있도록 유도합니다.

- **Technical Details**: FedSA는 세가지 주요 방법론을 바탕으로 구성됩니다: 1) Anchor-based Regularization with Margin-enhanced Contrastive Learning(RMCL)으로, 편향된 기능 추출기를 교정하여 클래스 내 응집성과 클래스 간 분리를 보장하는 일관된 프로토타입 학습을 지원합니다. 2) Anchor-based Classifier Calibration(CC)으로, 편향된 분류기를 교정하여 클래스 간의 일관된 결정 경계를 학습하도록 합니다. 3) Exponential Moving Average(EMA) 업데이트를 통해 강화된 프로토타입을 사용하여 Semantic Anchors를 업데이트하며, 이를 통해 클라이언트들이 통합된 데이터 표현을 협력적으로 학습하도도록 합니다.

- **Performance Highlights**: 실험 결과, FedSA는 통계적 및 모델 이질성 설정에서 기존의 Prototype 기반 FL 방법들에 비해 비약적으로 성능이 향상되었습니다. 다양한 분류 작업에서 FedSA는 클래스 간 분리가 잘 이루어지며, 일관된 결정 경계를 유지합니다. 이러한 성과는 FedSA가 클라이언트 모델 간의 비일관성 문제를 효과적으로 해결하여, 더욱 강력한 일반화를 달성했음을 보여줍니다.



### LSEBMCL: A Latent Space Energy-Based Model for Continual Learning (https://arxiv.org/abs/2501.05495)
Comments:
          In the 7th International Conference on Artificial Intelligence in Information and Communication (ICAIIC 2025)

- **What's New**: 본 연구에서 제안하는 LSEBMCL(Latent Space Energy-Based Model for Continual Learning) 방법은 에너지 기반 모델(Energy-based Model, EBM)을 활용하여 지속 학습에서의 치명적인 망각(catatstrophic forgetting)을 방지하는 새로운 접근법을 제시합니다. 이 방법은 새로운 과제를 학습할 때 이전 작업의 데이터 포인트를 샘플링하여 기존의 지식을 보존하는 방식으로 작동합니다. LSEBMCL은 자연어 처리(NLP) 과제에 대해 최신 성능을 달성하며, 기존의 방법들과 차별화되는 특징을 보입니다.

- **Technical Details**: LSEBMCL 모델은 사전 훈련된 Mistral 7B를 기반으로 하며, 네 가지 주요 구성 요소로 구성됩니다: 추론 네트워크, Operator 1, Operator 2, 에너지 함수입니다. 이 네트워크는 주어진 질문에 대해 답변을 제공하며, 다양한 NLP 과제를 처리할 수 있도록 설계되었습니다. 각 구성 요소는 훈련 중 에너지 기능과 분포를 활용하여 데이터를 효율적으로 처리하고 학습할 수 있도록 구성됩니다.

- **Performance Highlights**: 제안된 LSEBMCL 방법은 다양한 NLP 작업에서 우수한 성능을 달성했으며, 현재까지의 실험에서 최첨단 결과를 보여주었습니다. 에너지 기반 모델을 통합하여 이전 데이터 작업에 대한 샘플을 생성하는 방식은 지속 학습에서 기존의 지식을 효과적으로 유지할 수 있도록 합니다. 이러한 접근 방식은 특히 자연어 처리 분야에서의 적용 가능성을 높이며, 향후 다양한 실용적 응용을 위한 기초가 될 수 있습니다.



### Interpretable deep learning illuminates multiple structures fluorescence imaging: a path toward trustworthy artificial intelligence in microscopy (https://arxiv.org/abs/2501.05490)
- **What's New**: 이번 논문에서는 AEMS-Net(Adaptive Explainable Multi-Structure Network)라는 새로운 딥러닝 프레임워크를 제안합니다. 이 모델은 단일 이미지에서 두 개의 세포 내 구조를 동시에 예측할 수 있는 기능을 가지고 있습니다. 전통적인 다채널 형광 현미경의 이미징 지연 문제를 해결하고, 실시간 세포 연구 애플리케이션에 적합한 해법을 제공합니다.

- **Technical Details**: AEMS-Net은 주의 메커니즘(attention mechanisms)과 밝기 적응 레이어(brightness adaptation layers)를 통합하여 염색 강도를 정규화하고 중요한 이미지 특징을 우선시합니다. Kolmogorov-Arnold 표현 정리를 활용하여 학습된 특징을 해석 가능한 단일 변수 함수(univariate functions)로 분해하여 세포 내 복잡한 형태에 대한 해석 가능성을 향상시킵니다.

- **Performance Highlights**: AEMS-Net는 미토콘드리아와 미세소관(microtubules) 간의 상호작용을 실시간으로 기록할 수 있으며, 전통적인 순차 채널 이미징 절차의 절반만으로도 가능합니다. 이 접근 방식은 기존의 딥러닝 방법과 비교하여 이미지 품질에서 30% 이상의 개선을 달성하여 장기적이고 해석 가능한 라이브 세포 이미징의 새로운 패러다임을 확립합니다.



### Towards an Ontology of Traceable Impact Management in the Food Supply Chain (https://arxiv.org/abs/2501.05486)
- **What's New**: 논문에서는 식품 공급 사슬의 품질 개선과 책임성을 높이기 위한 종합적 접근법이 요구되고 있음을 강조합니다. 이 접근법은 제품 품질뿐만 아니라 다양한 이해관계자와 지역사회에 미치는 영향을 종합적으로 평가합니다. 이러한 새로운 모델은 데이터의 투명성을 높이고, 사회적 및 환경적 영향도 완화할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 모델인 트레이서블 임팩트 관리 모델(Traceable Impact Management Model, TIMM)은 이해관계자의 역할을 식별하는 구조와 보고 메커니즘을 제공합니다. TIMM은 식품 생산 및 소비 단계에서의 변화가 지역사회에 미치는 영향을 이해하는 데 도움을 주며, 정부의 요구 사항에 부합하고 지역 사회와 소비자의 필요를 해결할 수 있습니다. 이와 함께 제안되는 온톨로지 모델(ontological model)은 논리적 기초와 통일된 용어를 형성하여 접근법의 효과를 높입니다.

- **Performance Highlights**: 이 모델은 전 세계적으로 추적(tracking) 및 트레이싱(tracing) 프로세스를 개선하여 제품 품질을 보장하는 동시에 행위의 전반적인 영향을 다룹니다. 또한, 책임감 있는 식품 생산 및 소비를 촉진하고, 지속 가능성을 강조합니다. 최종적으로 이 통합적 솔루션은 품질을 중시하며, 글로벌 스탠다드에 부합하는 책임 있는 관행을 강조합니다.



### Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models (https://arxiv.org/abs/2501.05478)
- **What's New**: 이 연구는 로봇 공학에서 비전-언어 내비게이션(VLN) 분야에 아랍어 통합을 첫 번째로 선보이며, 다국어 Small Language Models(SLMs)와 아랍어 중심의 LLM인 Jais의 성능을 평가합니다. 분명히 부족했던 아랍어 데이터에 대한 연구의 공백을 메우면서, NavGPT 프레임워크와 R2R 데이터셋을 사용하여 아랍어와 영어 간의 의사소통이 내비게이션 추론에 미치는 영향을 평가합니다. 이를 통해, 아랍어로 지시를 받았을 때의 로봇 내비게이션 작업의 계획 및 추론 능력을 강조하였습니다.

- **Technical Details**: 본 연구는 OpenAI의 GPT-4o mini, Meta의 Llama 3 8B와 Microsoft의 Phi-3 medium 14B와 같은 최신 다국어 SLM들과 Jais 30B LLM을 NavGPT 프레임워크 내에서 비교합니다. R2R 데이터셋을 활용하여 영어 내비게이션 지시를 아랍어로 변환한 데이터셋으로 보면, 다양한 언어로 내비게이션 자원에 접근하고자 하는 양방향적 연구의 필요성을 강조합니다. 또한, 제로샷 방식으로 작업을 예측하며, 언어의 영향력에 대한 분석을 도모합니다.

- **Performance Highlights**: 실험 결과, NavGPT 프레임워크가 영어 및 아랍어 지시를 통해 높은 수준의 내비게이션 계획을 수행할 수 있음을 입증하였습니다. 그러나 일부 모델은 아랍어에서 추론 및 계획에 어려움을 겪어 언어 모델의 성능과 한계를 드러냈습니다. 이러한 발견은 아랍어 모델의 발전과 현실 세계 응용 프로그램에서의 가능성을 열어주며, 연구의 향후 방향으로 언어 모델의 계획 및 추론 능력 향상이 필요함을 강조합니다.



### IntegrityAI at GenAI Detection Task 2: Detecting Machine-Generated Academic Essays in English and Arabic Using ELECTRA and Stylometry (https://arxiv.org/abs/2501.05476)
- **What's New**: 최근 연구에서는 기계 생성 에세이를 감지하는 문제에 대한 조사가 이루어졌습니다. 이 연구는 아랍어와 영어 학술 에세이에 대해 스타일로메트릭 (stylometric) 특성으로 미리 훈련된 트랜스포머 기반 모델을 활용하여 이 문제를 해결하고자 하였습니다. ELECTRA 모델을 기반으로한 맞춤형 모델이 영어와 아랍어 학술 에세이에 대해 훈련되었으며, 탁월한 F1-score를 달성했습니다.

- **Technical Details**: 이 연구는 'GenAI Content Detection Task 2'라는 과제에 참가하여 진행되었습니다. 연구자들은 기계 생성 콘텐츠를 감지하기 위한 데이터셋을 제공하였으며, 이 데이터셋은 생성적 AI 모델과 인간이 작성한 에세이로 구성되었습니다. 모델 학습에 사용된 데이터셋은 아랍어 및 영어의 스타일로메트릭 특성을 포함하고 있으며, 여러 기계 학습 모델을 활용하여 성능을 평가했습니다.

- **Performance Highlights**: 제안된 모델은 영어 서브 태스크에서 99.7%의 F1-score를 기록하여 26개 팀 중 2위를 차지했습니다. 아랍어 서브 태스크에서는 98.4%를 기록하여 23개 팀 중 1위를 달성했습니다. 이러한 결과는 기계 생성 텍스트 감지 모델이 특히 뛰어난 성능을 보임을 나타냅니다.



### Retrieval-Augmented Generation by Evidence Retroactivity in LLMs (https://arxiv.org/abs/2501.05475)
- **What's New**: 본 논문은 Retroactive Retrieval-Augmented Generation (RetroRAG)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 단방향 추론 패러다임을 탈피하여, 증거를 수정하고 업데이트하여 올바른 방향으로 추론 체인을 안내하는 레트로액티브 추론 패러다임을 구축합니다. RetroRAG는 신뢰할 수 있는 증거를 검색, 생성 및 수정하는 증거 수집 및 발견 프레임워크를 구성합니다.

- **Technical Details**: RetroRAG는 증거 수집(Evidence Collation)과 증거 발견(Evidence Discovery)이라는 두 가지 주요 요소를 통해 효과적인 증거 생성을 목표로 합니다. 증거 수집은 관련 문서를 검색하여 출처 증거로 활용하며, 증거 발견은 질문의 주요 실체와 관련된 여러 유추 증거를 생성하고 필터링하여 필요 없는 정보를 제거하는 과정을 포함합니다. 이를 통해 LLM은 보다 정확하고 신뢰성 높은 답변을 생성할 수 있습니다.

- **Performance Highlights**: 실험적 평가 결과, RetroRAG는 다중 단계 질문 답변(QA) 데이터셋에서 기존 방법보다 현저히 우수한 성과를 보였습니다. 이 프레임워크는 증거 관련 정보의 동적 업데이트와 새로운 증거 발견을 통해 신뢰할 수 있는 답변을 Iteratively(반복적으로) 제공하며, 추론 과정의 설명 가능성도 입증되었습니다.



### Modality-Invariant Bidirectional Temporal Representation Distillation Network for Missing Multimodal Sentiment Analysis (https://arxiv.org/abs/2501.05474)
Comments:
          Accepted for publication by 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이번 연구에서는 결측된 다중 양식(Missing Multimodal) 감정 분석을 위한 Modality-Invariant Bidirectional Temporal Representation Distillation Network (MITR-DNet)를 소개합니다. MITR-DNet은 완전한 양식 teacher 모델이 결측 양식 student 모델을 지도하는 방식의 distillation 기술을 활용하여, 양식 부족이 발생할 경우에도 강인성을 보장합니다. 또한, 다중 양식 데이터의 이질성 문제를 완화하기 위한 Modality-Invariant Bidirectional Temporal Representation Learning Module (MIB-TRL)을 개발하였습니다.

- **Technical Details**: 본 연구에서는 음성(a), 텍스트(t), 비전(v)의 세 가지 양식을 고려합니다. 결측 양식 기능을 시뮬레이션하기 위해 마스킹 함수 F(·)와 랜덤 생성 시간 마스크 gm을 도입하였으며, 이를 통해 불완전한 시퀀스 X~m를 생성합니다. MIB-TRL 모듈은 두 개의 동일한 합성곱 층을 통해 입력 데이터를 처리하고, 이 방향으로 나뉜 데이터 스트림을 통합하여 각 양식의 장기 맥락 표현을 생성합니다.

- **Performance Highlights**: MITR-DNet과 MIB-TRL 모듈은 결측 양식 감정 분석 분야에서의 정확성 및 신뢰성을 크게 향상시켰습니다. 연구 결과, 다중 양식을 활용한 접근 방식은 단일 양식 분석보다 더 세밀하고 정확한 감정 평가를 가능하게 하며, 실제 세계 상황에서의 양식 결측으로 인한 문제를 효과적으로 해결합니다. 또한, 이 혁신적인 구조는 감정 예측에서 텍스트 양식의 중요성을 최대로 활용하여 향상된 성능을 보여줍니다.



### Found in Translation: semantic approaches for enhancing AI interpretability in face verification (https://arxiv.org/abs/2501.05471)
- **What's New**: 본 논문은 컴퓨터 비전에서 특히 얼굴 인증(face verification) 분야의 머신 러닝 모델의 복잡성이 증가함에 따라, 해석 가능하고 투명한 인공지능을 위한 설명 가능한 인공지능(Explainable AI, XAI) 기술을 개발하는 데 중점을 두었습니다. 저자들은 이전 작업을 확장하여 인간의 인지 과정에서 파생된 의미론적 개념을 XAI 프레임워크에 통합하였습니다. 이를 통해 모델 출력과 인간 이해 간의 간극을 메우는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법론은 사용자 선택에 의한 얼굴 랜드마크로 정의된 의미론적 특성을 사용하여, 글로벌(global) 및 로컬(local) 설명을 결합하는 새로운 접근법입니다. 이 과정에서 대형 언어 모델(Large Language Models, LLMs)을 활용하여 유사성 지도(similarity maps)와 텍스트 설명을 생성합니다. 연구는 정량적인 실험과 사용자 피드백을 통해 방법론의 유효성을 검증하였으며, 향상된 해석 가능성을 확인하였습니다.

- **Performance Highlights**: 결과적으로 저자들의 의미론적 기반 접근법, 특히 가장 상세한 세트는 전통적인 방법에 비해 모델 결정의 더 섬세한 이해를 제공합니다. 사용자 연구에서는 전통적인 픽셀 기반 히트맵(pixel-based heatmaps)보다 저자들의 의미론적 설명에 대한 선호도가 강조되었으며, 이는 AI에서 인간 중심의 해석 가능성의 이점을 드러냅니다. 이 연구는 AI 모델의 행동을 인간의 인지 과정과 일치시키는 XAI 프레임워크 개발에 기여하여, 중요한 분야에서의 신뢰와 수용을 촉진하는 데 초점을 맞추고 있습니다.



### RTLSquad: Multi-Agent Based Interpretable RTL Design (https://arxiv.org/abs/2501.05470)
- **What's New**: 이번 연구에서는 RTL 코드 생성을 위한 새로운 LLM 기반의 다중 에이전트 시스템인 RTLSquad를 제안합니다. 이 시스템은 탐색(exploration), 구현(implementation), 검증 및 평가(verification & evaluation) 단계로 디자인 프로세스를 나누고 각 단계를 전문화된 에이전트 팀이 관리하여 최적화된 RTL 코드를 생성합니다. RTLSquad는 에이전트 간의 협업을 통해 기능적으로 올바른 RTL 코드를 생성하고, 결정 과정의 해석 가능성을 제공합니다.

- **Technical Details**: RTLSquad는 여러 개의 LLM 에이전트로 구성되어 있으며, 자연어를 사용하여 결정을 소통하고 협상하여 투명한 의사결정 과정을 제공합니다. 이 시스템은 반복 최적화를 통해 다양한 디자인을 탐색하고 실행하며, 최종적으로 정확하고 잘 최적화된 RTL 코드를 생성합니다. 실험 결과에 따르면 RTLSquad는 RTL 코드 생성 능력을 7.2% 향상시키고, 결정을 문서화하여 각 단계의 결정을 명확히 설명합니다.

- **Performance Highlights**: RTLSquad의 성능은 여러 디자인에 걸쳐 평가되었으며, 대부분의 경우 PPA(파워, 성능 및 면적)에서 참고 디자인과 동일하거나 초과하는 성과를 보여주었습니다. 결정 경로는 잘 구조화된 문서로 출력되어 각 단계에서의 결정을 포괄적으로 설명합니다. 이러한 결과는 시스템의 실용적인 가치를 강조하며, 하드웨어 디자인의 신뢰성을 높이는 데 기여합니다.



### LLM-MedQA: Enhancing Medical Question Answering through Case Studies in Large Language Models (https://arxiv.org/abs/2501.05464)
- **What's New**: 이번 연구에서는 기존의 Medical Question Answering (MedQA) 시스템의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. Llama3.1:70B 모델을 활용한 Multi-agent framework를 통해 의료 분야의 질문 응답 문제에 대응하며, 사례 생성 기능을 통합하여 성능을 향상시키고자 합니다. 이 방법론은 의료 지식에 대한 모델의 내재된 능력을 활용하여 추가적인 학습 데이터 없이도 의료 쿼리를 처리할 수 있도록 설계되었습니다.

- **Technical Details**: Multi-agent 시스템은 질문 전문가와 선택지 전문가를 포함한 여섯 개의 주요 구성 요소로 이루어져 있습니다. 문제와 선택지에 대한 분석 후 생성된 상담 사례를 기반으로 리포트를 작성하고, 전문가의 투표 메커니즘을 통해 합의를 이뤄 최종 결정을 내립니다. Llama3.1:70B 모델의 관점에서, 이는 zero-shot learning을 통해 추가 훈련 없이도 복잡한 의학적 쿼리를 처리할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 제안된 다중 에이전트 시스템은 기존 벤치마크 모델에 비해 정확성과 F1 점수가 각각 7% 향상된 것으로 나타났습니다. 이 시스템은 의학적 쿼리에 대한 해석 가능성과 신뢰성을 증가시켜, 복잡한 의료 문제 해결을 위한 견고한 솔루션을 제공합니다. 이러한 성과는 앞으로 LLMs의 의료 분야에서의 광범위한 응용 가능성을 더욱 확장시킬 것으로 기대됩니다.



### Proof Recommendation System for the HOL4 Theorem Prover (https://arxiv.org/abs/2501.05463)
Comments:
          Conference on Artificial Intelligence and Theorem Proving (AITP), Aussois, France, 2024

- **What's New**: 본 논문에서는 HOL4 정리 증명기 (theorem prover)를 위한 증명 추천 시스템 (proof recommender system)을 소개합니다. 이 도구는 HOL4에서 증명 지원을 위해 특별히 설계된 transformer 기반 모델 위에 구축되었습니다. 또한, 모델은 광범위한 HOL4 증명 라이브러리를 통해 증명 패턴을 학습합니다.

- **Technical Details**: 이 모델은 사용자가 변형된 증명 과정에서 이미 사용된 최소 세 가지의 tactic(전략)들 기반으로, 현재 증명 상태(증명 과정에서의 적용 전략 연속)를 읽음으로써 다음 최적 증명 단계(tactic)를 추천할 수 있도록 훈련되었습니다. 이러한 접근 방식은 정리 증명에서 자주 발생하는 전략적 패턴을 알고리즘적으로 학습하여 증명 과정의 효율성을 높입니다.

- **Performance Highlights**: 추천 시스템은 이전에 사용된 tactic의 이력을 통해 새로운 방안을 정확히 예측할 수 있으며, 이는 정리 증명 업무에서 시간을 절약하고 정확도를 향상시킬 수 있는 잠재력을 제공합니다. 따라서, 증명 작업에 있어 더 나은 성공률과 더 빠른 증명이 가능해질 것으로 기대됩니다.



### Efficiently serving large multimedia models using EPD Disaggregation (https://arxiv.org/abs/2501.05460)
Comments:
          13 pages, 6 figures

- **What's New**: 본 연구는 Encode-Prefill-Decode (EPD) Disaggregation이라는 새로운 프레임워크를 제안하여 대규모 멀티모달 모델 (LMM)에서 인코딩, 프리필, 디코딩 단계를 분리한다. 이 접근 방식은 메모리 병목 현상을 완화하고, 동기화 지연을 줄이며, 유연한 배치 처리를 지원한다. 또한, 멀티모달 토큰을 위한 새로운 캐싱 메커니즘을 도입하여 비동기 전송을 가능하게 하였다.

- **Technical Details**: EPD 분해는 각 단계를 고유한 리소스에 할당하여 독립적으로 최적화가 가능하도록 하며, 메모리 효율성을 크게 향상시킨다. 이를 통해 배치 크기 및 이미지 처리 수치가 증가하고, 최적 성능 메트릭을 달성하기 위한 통합 모듈도 포함한다. 연구 결과는 다양한 LMM 모델에서 메모리 사용량이 최대 15배 감소하고, 배치 크기가 최대 22배 증가하는 효과를 보여준다.

- **Performance Highlights**: 실험 결과, EPD 분해 방식이 기존 시스템에 비해 종합 성과를 개선하는 데 기여하여, 전체 처리량(E2ETP)이 최대 57% 향상되고, 첫 번째 토큰 수신까지의 지연(TTFT)이 최대 71% 감소했다. 이러한 결과는 적은 자원으로도 고성능의 멀티모달 추론이 가능함을 보여주며, EPD의 잠재력을 강조한다.



### Upstream and Downstream AI Safety: Both on the Same River? (https://arxiv.org/abs/2501.05455)
- **What's New**: 이 논문은 전통적인 안전 공학(safety engineering)과 최신 AI 안전(frontal AI safety) 접근 방식 간의 차이를 설명합니다. 전통적인 안전 공학은 자율 주행 차량(self-driving vehicles)의 운영 디자인 도메인에 따라 시스템을 평가하는 반면, 최신 AI 안전은 특정 응용 프로그램의 맥락을 넘어 모형의 독립성과 해로운 콘텐츠 생성 가능성 등을 고려합니다. 이러한 두 안전 프레임워크의 특성을 구체적으로 나누어 설명합니다.

- **Technical Details**: 논문에서는 다운스트림 안전(downstream safety)과 업스트림 안전(upstream safety) 프레임워크의 특성을 정리하며, 이러한 체계들 간의 상호 시너지를 탐구합니다. 예를 들어, 다운스트림 안전에서의 공통 모드 실패(common mode failures) 개념이 AI 가드레일(guardrails)의 평가에 어떻게 활용될 수 있는지를 분석합니다. 또한, 프론티어 AI의 능력과 한계를 이해함으로써 자율 선박(autonomous vessels)의 항해 계획(voyage plans) 분석에 도움이 될 수 있는 방법 또한 모색합니다.

- **Performance Highlights**: 본 연구는 업스트림과 다운스트림 안전 프레임워크 간의 시너지를 통해 AI 안전 커뮤니티가 어떤 이점을 얻을 수 있는지를 보여줍니다. 논문에서는 몇 가지 유망한 경로를 제시하며, 이들 간의 동융 또는 융합(confluence)을 달성하는 데 있어 도전과제를 명시합니다. 이러한 통합적 접근 방식은 AI 안전을 향상시키는 중요한 역할을 할 것으로 기대됩니다.



### Atlas: A Novel Pathology Foundation Model by Mayo Clinic, Charit\'e, and Aignostics (https://arxiv.org/abs/2501.05409)
- **What's New**: 이 보고서에서는 RudolfV 접근법을 기반으로 한 새로운 비전 파운데이션 모델인 Atlas를 소개합니다. Atlas는 Mayo Clinic과 Charité - Universitätsmedizin Berlin에서 수집한 120만 개의 조직병리학 전체 슬라이드 영상으로 훈련되었으며, 다양한 공공 벤치마크 데이터셋에서 최고의 성능을 달성했습니다. 이 모델은 매개변수 수나 훈련 데이터셋 크기 면에서 가장 크지 않지만 여전히 뛰어난 성능을 보여줍니다.

- **Technical Details**: Atlas 모델은 ViT-H/14 아키텍처(632백만 매개변수)를 사용하여 훈련되며, 490,000건의 사례에서 추출된 34억 개의 이미지 타일로 훈련되었습니다. 데이터는 다양한 해상도에서 추출되었으며, 각기 다른 염색법과 다중 배율이 포함되어 있습니다. 이러한 다양성은 AI 학습의 일반화 및 강건성 향상에 기여합니다.

- **Performance Highlights**: Atlas는 21개의 공공 벤치마크 데이터셋에서 성능을 평가하여, 전통적인 모델에 비해 우수한 결과를 기록했습니다. 우리는 모델 성능 평가를 위해 선형 프로빙 프로토콜을 사용하였으며, 모든 모델의 추출된 임베딩을 비교했습니다. 결과적으로, Atlas는 다양한 다운스트림 병리학 작업에서 탁월한 성과를 달성하며, 실제 임상 환경에서도 활용될 수 있는 잠재력을 지니고 있습니다.



New uploads on arXiv(cs.LG)

### Meta-Learning for Physically-Constrained Neural System Identification (https://arxiv.org/abs/2501.06167)
Comments:
          30 pages

- **What's New**: 이 논문은 블랙박스 시스템 식별을 위한 신경 상태 공간 모델(NSSM)의 빠른 적응을 위한 그래디언트 기반 메타-러닝(framework)을 제안합니다. 이 방법은 단일 대상 시스템의 데이터에 의존하기보다는 다양한 출처 시스템으로부터 데이터를 활용하여 제한된 대상 데이터로도 학습할 수 있도록 합니다. 또한, 도메인 특화 물리적 제약을 통합하여 NSSM의 정확도를 향상시키는 접근법을 취하고 있습니다.

- **Technical Details**: 이 연구는 여러 유사 시스템의 데이터를 이용하여 신경 상태 공간 모델(NSSM)의 메타-러닝(met-learning) 접근법을 제안합니다. 여기서, MAML(Meta-Learning) 알고리즘을 사용하여 두 단계의 최적화 문제를 해결하며, 내부 루프 업데이트 없이도 높은 성능을 발휘할 수 있는 거의 없는 내부 루프(ANIL) 변형을 도입합니다. 이를 통해 물리적 제약이 적용된 시스템 식별 문제에서 적용 가능성을 탐구하고 있습니다.

- **Performance Highlights**: 제안된 메타-러닝 모델은 기존의 단일 대상 시스템 데이터만 사용하는 신경 상태 공간 모델보다 높은 예측 정확도를 보이는 것으로 나타났습니다. 또한, 적은 양의 온라인 훈련(iteration)으로도 빠르게 적응할 수 있는 성능을 입증했습니다. 논문에서는 또한 실제 사례 연구를 통해 제안된 접근법의 실용성과 일반화 가능성을 보여줍니다.



### Model Alignment Search (https://arxiv.org/abs/2501.06164)
- **What's New**: 이번 연구는 Neural System 간의 유사성을 인과적으로 탐색하는 새로운 방법인 Model Alignment Search (MAS)를 소개합니다. MAS는 레퍼리조날 유사성을 측정하고, 다양한 훈련 환경에서 인과 변수 전이 작업을 통해 모델 간의 관계를 분석하는 과정을 제공합니다. 기존의 유사성 측정 방법들과 비교하여 MAS는 인과적으로 관련된 정렬을 찾을 수 있는 가능성을 제시합니다.

- **Technical Details**: MAS는 여러 모델间에서 정보가 자유롭게 교환될 수 있도록 하는 정렬된 표현 서브스페이스를 찾는 선형 변환을 학습합니다. 이 방법은 여러 인과 변수를 특정화함으로써 실행에서 흥미로운 질문에 대한 통찰력을 제공합니다. 연구에서는 은닉 상태의 조정이나 직선 변환을 통해 확인된 인과적 유사성을 비교하여 MAS의 유용성을 증명합니다.

- **Performance Highlights**: MAS는 숫자 추적 시스템의 표현 유사성을 조사하면서 서로 다른 구조적 작업에서 훈련된 모델들 간의 차이를 드러냅니다. 기존 인과적 유사성 방법들과 비교했을 때, MAS는 원치 않는 교환에 강한 저항력을 가집니다. 마지막으로, 인과적 접근이 불가능한 모델에서 인과적 관련성을 회복할 수 있도록 하는 보조 손실 목표(counterfactual latent auxiliary loss)를 도입함으로써 MAS의 활용 가능성을 넓혔습니다.



### GenMol: A Drug Discovery Generalist with Discrete Diffusion (https://arxiv.org/abs/2501.06158)
- **What's New**: 이번 논문에서는 Generalist Molecular generative model (GenMol)을 제안하며, 이는 기존의 생성 모델이 처리하지 못하는 다양한 약물 발견 시나리오에 유연하게 대응할 수 있는 프레임워크입니다. GenMol은 Sequential Attachment-based Fragment Embedding (SAFE) 분자 표현에 이산 확산(discrete diffusion) 기법을 적용하여 생성 효율성과 품질을 개선합니다. 또한, 화학 공간을 탐색하기 위한 효과적인 전략인 fragment remasking도 도입하여 분자 최적화를 수행합니다.

- **Technical Details**: GenMol은 BERT 아키텍처를 기반으로 하여 비자기적 평행 디코딩(non-autoregressive bidirectional parallel decoding)을 사용하여 SAFE 분자 시퀀스를 생성합니다. 이 방식은 특정 토큰 순서에 의존하지 않고, 분자의 맥락을 활용하여 더 효율적인 계산을 가능하게 합니다. 추가로, fragment remasking 전략은 특정 분자 조각을 마스크 토큰으로 대체하여 새로운 화학 조각을 생성하여, 효과적으로 화학 공간을 탐색합니다.

- **Performance Highlights**: GenMol은 de novo 생성과 fragment-constrained 생성 등 다양한 약물 발견 과제에서 기존의 모델들을 대폭 초월하는 성능을 입증하였습니다. 실험 결과는 GenMol이 성공적인 약물 후보를 찾는 데 있어 다목적 도구로서의 잠재력을 보여줍니다. 이러한 성과는 GenMol이 약물 발견 파이프라인 전반에 걸쳐 사용될 수 있는 통합적이고 유연한 접근법임을 강조합니다.



### From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training (https://arxiv.org/abs/2501.06148)
Comments:
          code: this https URL

- **What's New**: 이 논문에서는 표적 샘플에 접근할 수 없는 Boltzmann 분포에서 샘플링하기 위해 신경 확률적 미분 방정식(neural stochastic differential equations) 또는 확산 모델(diffusion models)을 훈련하는 문제를 탐구합니다. 이는 시간 역전성(time-reversal)을 강제하는 기존 방법들과는 다르게, 적절한 coarse time discretization을 선택함으로써 샘플 효율성을 크게 개선할 수 있음을 보여줍니다. 특히, 계산 비용을 줄이면서도 표준 샘플링 벤치마크에서 경쟁력 있는 성능을 달성하고자 합니다.

- **Technical Details**: 연구에서는 경량화된 수치적 기술적 상세사항을 포함하여, 확률적 미분 방정식(stochastic differential equation, SDE)을 활용한 생성 과정에 대한 이해를 심화합니다. SDE는 파라메트릭 모델을 사용하여 변동성 변화(transition probability)를 학습하고, 이를 통해 X의 분포를 목표 분포로 가깝게 만든다는 목표를 설정합니다. 또한, GFlowNets와 같은 엔트로피 RL 방법들과의 관련성을 찾아내고, 연속 시간 및 이산 시간 과정의 훈련 목표가 미세한 시간 단계의 한계에서 어떻게 연결되는지를 공식화합니다.

- **Performance Highlights**: 논문에서 제안된 방법론은 실험적으로 표준 샘플링 벤치마크에서 높은 효율성을 보여줍니다. 특히, 각 시간 단계를 다룰 때 GFlowNets의 국지적 제약을 활용하여 부분 미분 방정식(partial differential equations) 형식의 목표로 수렴함을 입증합니다. 연구 결과는 시간 진화(marginal densities)의 목표를 달성하는 과정에서 강력한 성능을 보여줍니다.



### Emergent Symbol-like Number Variables in Artificial Neural Networks (https://arxiv.org/abs/2501.06141)
- **What's New**: 이 연구에서는 Neural Networks(NNs)가 숫자 개념을 어떻게 표현하는지를 탐구합니다. 우리는 숫자 작업에 대해 Next Token Prediction(NTP) 목표를 사용하여 순차 기반의 신경 시스템을 훈련함으로써 이 질문에 접근합니다. 연구 결과, 인공 신경 모델이 상호작용 가능한, 유동적인 잠재 숫자 변수를 발전시킨다는 것을 발견했습니다.

- **Technical Details**: 우리는 순환(recurrent)과 주의(attention) 기반의 ANN 모델을 훈련시키고 인과적(causal) 및 상관적(correlative) 분석을 수행하여 그들의 신경적 표현과 해결책을 이해합니다. 이 연구는 신경 변수(activations의 하위 공간)와 카운팅 프로그램의 기호적(symbolic) 변수 사이의 인과적 정렬(causal alignment)을 발견하였습니다. 변환기(transformer) 아키텍처는 각 단계에서 정보를 재계산하는 방법으로 과제를 해결하는 반면, 순환 모델은 누적적인 상태 저장 방식으로 문제를 해결하는 차이점을 보여 주었습니다.

- **Performance Highlights**: 훈련 과정에서 신경 변수는 작업 정확도(task accuracy)와 강한 상관관계를 보였고, 크기가 최소한인 모델이 더 큰 강도에서 조정되었음을 나타냅니다. 또한, 연구 결과는 신경 기호가 어떻게 과제를 해결하는지를 이해하는 데 있어 간단하고 명확한 기호적 이야기를 찾기가 어렵다는 점을 강조했습니다. 결과적으로, 우리의 연구는 NN이 숫자 인지를 위한 해석 가능한 기호적 프로그램을 근사화할 수 있음을 보여줍니다.



### Explaining Deep Learning-based Anomaly Detection in Energy Consumption Data by Focusing on Contextually Relevant Data (https://arxiv.org/abs/2501.06099)
Comments:
          26 pages, 8 figures

- **What's New**: 이 논문은 전력 소비 데이터에서의 이상 탐지를 위한 설명 가능성(explainability) 접근 방식을 제안합니다. 기존의 Explainable AI(XAI) 기법은 SHAP 변형 방식을 활용하였고, 배경 데이터셋을 선택하는 데 있어 각 이상 포인트의 맥락(context)에 맞는 정보를 중점적으로 고려합니다. 이러한 접근은 이상 탐지에서의 설명의 불안정성을 줄이고, 일관된 설명을 제공하는 데 도움을 줍니다.

- **Technical Details**: 제안된 방법은 SHAP의 변형과 글로벌 특징 중요도(global feature importance), 가중치 코사인 유사도(weighted cosine similarity)를 활용하여 각 이상 포인트에 적합한 배경 데이터셋을 선택합니다. 실험 결과는 10개의 머신러닝 모델과 5개의 데이터셋, 5개의 XAI 기법 전반에 걸쳐 진행되었으며, 설명의 일관성을 높이고 불확실성을 줄였습니다. 특히, 컨텍스트와 관련된 특징에 중점을 두어 안정성을 강화했습니다.

- **Performance Highlights**: 통계 분석에 따르면 제안된 방법은 여러 데이터셋에서 설명의 변동성을 평균 38% 감소시키는 성과를 보였습니다. 이렇게 향상된 설명의 안정성 및 신뢰성은 에너지 관리 및 의사 결정 과정에서 중요한 역할을 할 것으로 기대됩니다. 개선된 설명 결과는 다양한 머신러닝 모델에 걸쳐 적용 가능하여, 에너지 소비 분석 및 관리에 실질적 기여를 할 수 있을 것입니다.



### Scale-up Unlearnable Examples Learning with High-Performance Computing (https://arxiv.org/abs/2501.06080)
- **What's New**: 본 연구에서는 의료 데이터 보안 문제를 해결하기 위해 Unlearnable Examples (UEs) 기법을 도입하고, 이를 통해 딥러닝 모델이 데이터 학습을 방지하는 방안을 제시합니다. 특히 Unlearnable Clustering (UC) 방법을 통해 데이터의 비학습 가능성을 증진시키는 것을 목표로 하였으며, 이를 위해 Summit 슈퍼컴퓨터를 활용하여 대규모 분산 처리 환경에서 실험을 진행했습니다. 이전 연구에서 제시된 배치 크기 조정의 중요성을 강조하며, 데이터 집합에 맞는 최적의 배치 크기를 선택하는 필요성을 강조합니다.

- **Technical Details**: UCs 모델은 두 가지 구성 요소로 이루어져 있습니다: 생성기 모델과 대리 모델입니다. 생성기 모델은 랜덤 노이즈를 클러스터 단위 노이즈로 변환하고, 대리 모델은 노이즈가 포함된 이미지를 학습하는 역할을 합니다. 이 훈련 과정에서 k-means를 사용하여 클러스터를 생성하고, 해당 클러스터의 라벨을 섞은 후 생성된 노이즈를 이미지에 추가하여 대리 모델에서 훈련합니다. DDP(Distributed Data Parallel) 기법은 여러 머신에 모델을 분산시켜 대량의 데이터를 처리하는 데 필요한 효율성을 제공합니다.

- **Performance Highlights**: 다양한 데이터셋에 대해 UCs 방법이 배치 크기에 따라 성능이 어떻게 변화하는지를 분석하였습니다. 실험 결과, 지나치게 작은 또는 큰 배치 크기가 성능의 불안정성을 초래하며 정확도에 영향을 미치는 것으로 나타났습니다. 하지만, 데이터셋에 따라 배치 크기와 비학습 간의 관계가 달라지므로, 각 데이터셋에 최적화된 배치 크기를 선택하는 것이 데이터 보안을 강화하는 데 중요함을 보여줍니다.



### Explaining k-Nearest Neighbors: Abductive and Counterfactual Explanations (https://arxiv.org/abs/2501.06078)
- **What's New**: 이번 연구에서는 k-Nearest Neighbor (k-NN) 분류기의 설명 가능성을 이론적으로 조명합니다. 기존 연구는 데이터 관점에서 최선 이웃을 검토해왔지만, 고차원 데이터에서는 이러한 방식이 비효율적일 수 있습니다. 이에 따라, 이 논문은 특성 관점에서 k-NN 분류를 이해하는 데 중점을 두고, 주요 특성이 분류 결과에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 abductive 설명과 counterfactual 설명을 다루며, 각각의 설명이 k-NN의 특성에 미치는 영향을 구체적으로 살펴봅니다. 특히, 각 특성이 분류를 보장하기 위한 최소한의 조합이나, 결과를 변경하기 위해 필요한 최소 거리 변화를 고려합니다. 이 과정에서, 설명을 계산하기 위한 Integer Quadratic Programming과 SAT 풀이의 응용 가능성도 제시됩니다.

- **Performance Highlights**: 연구 결과, NP-hard 문제로 식별된 다양한 설명 생성 문제에 대해 기술적으로 접근할 수 있는 방법론을 정의합니다. 이 논문에서는 k-NN 분류기의 설명 가능성에 대한 기존 문헌이 적다는 점을 지적하며, 이론적 복잡성을 세부적으로 분석하여 향후 연구의 방향성을 제시합니다. 특별히, 거리 측정 방법의 선택에 따라 복잡성의 간극이 발생하는 등 다양한 사실을 발견하였습니다.



### Explainable Federated Bayesian Causal Inference and Its Application in Advanced Manufacturing (https://arxiv.org/abs/2501.06077)
Comments:
          26 pages

- **What's New**: 이번 논문에서는 제조 시스템 내의 인과 관계 탐사를 위한 설명 가능하고 확장 가능하며 유연한 Federated Bayesian 학습 프레임워크인 xFBCI를 소개합니다. xFBCI는 개별 클라이언트의 로컬 데이터를 직접 접속하지 않고도 치료 효과 추정(treatment effect estimation)을 통해 인과성을 탐색할 수 있도록 설계되었습니다. 이러한 접근 방식은 제조 시스템에서 데이터 비밀 유지를 가능하게 하며, 기존의 Bayesian 인과 추론 방법과 비교해 유의미한 성과를 보여주고 있습니다.

- **Technical Details**: xFBCI는 Bayesian federated variational inference를 활용하여 각 클라이언트의 개인화된 파라미터를 추정합니다. 예상 전파(Expectation Propagation, EP) 알고리즘을 사용하여 로컬 데이터에 접근하지 않고도 클라이언트의 후행 분포(posteriors)를 근사합니다. 이를 통해 치료 효과 분석을 위한 Propensity Score Matching (PSM)을 실시하며, 이 과정에서 변수들 간의 혼란 효과(confounding effects)를 효과적으로 처리합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 EHD 인쇄 데이터에 대한 테스트를 통해 xFBCI는 기존의 Bayesian 인과 추론 방법 및 선진 Federated 학습 벤치마크보다 우수한 성과를 나타냈습니다. EHD 인쇄 분야에서의 적용 가능성을 보여주었으며, 이는 향후 다양한 제조 분야에서 일반화될 수 있는 잠재력을 시사합니다. 또한 EHD 인쇄 시스템의 개선을 위한 causal relationship의 이해를 돕는 강력한 프레임워크를 제공합니다.



### A monthly sub-national Harmonized Food Insecurity Dataset for comprehensive analysis and predictive modeling (https://arxiv.org/abs/2501.06076)
Comments:
          The authors Melissande Machefer and Michele Ronco have contributed equally as both first authors to this work. This work is currently being reviewed in a peer-reviewed journal

- **What's New**: 본 논문은 여러 주요 데이터 출처를 통합하여 구축된 Harmonized Food Insecurity Dataset (HFID)을 소개합니다. HFID는 식량 불안정성을 측정하는 데 필요한 다양한 지표들을 아우르는 데이터베이스로, 전 세계적 데이터를 표준화하여 제공함으로써 학문적 및 실무적 의미를 가집니다. 이 데이터는 월간 업데이트되며, 인도적 기구와 식량 안전 전문가들이 활용할 수 있는 중요한 자료로 자리매김할 것입니다.

- **Technical Details**: HFID는 Integrated Food Security Phase Classification (IPC)과 Cadre Harmonisé (CH) 단계, Famine Early Warning Systems Network (FEWS NET), Food Consumption Score (FCS), 그리고 reduced Coping Strategy Index (rCSI)와 같은 주요 지표들을 통합합니다. 2007년부터 현재까지의 데이터를 포함하며, 80개 국가의 두 가지 행정 수준에 걸쳐 총 311,838개의 기록을 담고 있습니다. 이를 통해 다양한 지역과 시간대에서 수집된 식량 불안정성 데이터를 비교 분석할 수 있는 기회를 제공합니다.

- **Performance Highlights**: HFID는 다양한 공급자로부터 수집된 역사적 데이터를 분석할 수 있는 기능을 제시하며, 실시간으로 식량 불안정성을 모니터링하는 강력한 도구로 작용합니다. 특히 다중 협력 분석 접근법과 단일 결과 지표를 통합하여 데이터 해석의 정확성을 높이고 있습니다. 이 데이터 세트는 또한 기후, 분쟁 및 기타 방해 요소들과 같은 다양한 요소들이 식량 불안정성에 미치는 영향을 더 잘 이해할 수 있도록 돕는 예측 모델 개발 가능성을 기대하게 합니다.



### Geometry and Optimization of Shallow Polynomial Networks (https://arxiv.org/abs/2501.06074)
Comments:
          36 pages, 2 figures

- **What's New**: 이 논문에서는 다항식 활성화 함수(polynomial activations)를 가진 얕은 신경망에 대해 연구합니다. 이들 모델의 함수 공간은 한정된 랭크(bounded rank)를 가진 대칭 텐서 집합으로 식별될 수 있습니다. 연구의 주요 초점은 네트워크의 폭(width)과 최적화(optimization) 사이의 관계를 규명하는 것입니다. 이외에도 교육자-학생 문제(teacher-student problems)의 맥락에서 저자-메트릭(discriminant)이라는 새로운 개념을 도입하여 최적화의 행동을 분석합니다.

- **Technical Details**: 이 논문에서는 교사-학생 훈련 문제와 저차 텐서 근사(low-rank tensor approximation)에 대한 연구를 다룹니다. 특히, 훈련 데이터 분포에 의해 유도된 비표준 내적(non-standard inner product)과 관련하여 저차 텐서 근사의 최적 해를 찾는 과정이 다뤄집니다. 다항식 활성화 네트워크의 최적화 경관(optimization landscape) 분석을 통해 대칭 텐서 분해(symmetric tensor decompositions)와의 관계를 탐색하게 됩니다. 마지막으로, 가우시안(Gaussian) 훈련 데이터에 대한 특징과 해시안(signature)을 분석합니다.

- **Performance Highlights**: 제안된 모델은 네트워크 폭이 작은 경우, 중간 경우, 큰 경우에 대해 서로 다른 최적화 행동을 보이는 것으로 나타났습니다. 가우시안 데이터에 대해 모든 주요 임계점(critical points)과 해시안 서명(Hessian signature)을 정확히 특성화하여, 훈련 데이터 분포에 따라 최적화 경관이 어떻게 다른지를 분석했습니다. 또한, 일반적인 데이터 분포에서의 훈련 손실은 가우시안 데이터 경우와 매우 다르며, 이로 인해 더 많은 임계점을 형성함을 보였습니다.



### Distilling Calibration via Conformalized Credal Inferenc (https://arxiv.org/abs/2501.06066)
Comments:
          Under review

- **What's New**: 본 논문은 엣지 디바이스에서의 신뢰성을 향상시키기 위한 새로운 저복잡도 캘리브레이션 방법론인 Conformalized Distillation for Credal Inference (CD-CI)를 제안합니다. 이 방법은 복잡한 클라우드 모델에서 캘리브레이션 정보를 증류하여 엣지 모델의 성능을 보장합니다. 특히, 클라우드 모델을 통해 예측된 확률을 활용하여 신뢰 구간을 설정함으로써, 엣지 모델의 캘리브레이션을 개선하는 접근법을 취합니다.

- **Technical Details**: CD-CI 방법론은 오프라인 단계에서 고복잡 클라우드 모델의 예측 확률을 기반으로 다양한 예측의 분산을 측정하여 신뢰 임계치를 설정합니다. 이 임계치는 엣지 장치에서 실행될 때 크리달 세트를 구성하는 데 사용되며, 이는 사용자가 선택한 신뢰 수준에 따라 클라우드 모델의 예측을 포함하는 확률 범위를 나타냅니다. 최종적으로, 크리달 세트는 엔트로피 극대화와 같은 방법을 통해 예측 분포로 변환됩니다.

- **Performance Highlights**: 실험 결과, CD-CI는 CIFAR-10 및 SNLI 데이터셋을 포함한 시각 및 언어 모델링 작업에서 낮은 복잡도의 베이지안 방법인 Laplace 근사를 초과하는 캘리브레이션 성능을 보였습니다. 기대 캘리브레이션 오차(ECE)로 측정한 이 성능 향상은 원래 모델과 비교하여 큰 개선을 나타내면서도 정확도는 거의 감소하지 않았습니다. 이는 엣지 AI 배치에서 CD-CI 접근법의 실용성과 효율성을 강조합니다.



### Personalized Language Model Learning on Text Data Without User Identifiers (https://arxiv.org/abs/2501.06062)
- **What's New**: 이번 논문에서는 클라우드 응용 프로그램에서 사용자가 식별되지 않는 방법으로 개인화된 언어 모델을 학습하는 새로운 요구 사항을 소개합니다. 기존의 사용자 개인 정보 수집 방식을 사용하지 않으면서도 사용자 맞춤형 서비스를 제공할 수 있는 방안을 제시하고 있습니다. 특히, 각 모바일 디바이스에서 사용자별 배포(distribution)를 유지하여 동적으로 사용자 임베딩(user embedding)을 생성하도록 제안하고 있습니다.

- **Technical Details**: 제안된 IDfree-PL 프레임워크는 데이터 익명성을 보장하면서 사용자 맞춤형 모델 성능을 달성하기 위해 사용자 특정 배포의 선택에 대한 요구 사항을 설정합니다. 이 프레임워크는 임베딩과 특정 사용자의 대응 관계를 일방향 또는 다대일(many-to-one) 관계로 전환하여 익명성을 유지합니다. 또한, 사용자에 대한 지역 배포를 최적화하므로 데이터 익명성을 유지하면서도 모델 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: IDfree-PL을 사용한 평가 결과, 비개인화 모델에 비해 최대 5.69%의 정확도 향상을 보이며, 지연 시간은 최대 0.01초 증가하는 수준으로 유지되었습니다. 이 연구는 사용자의 개인 정보를 악용하지 않으면서도 최적의 성능을 달성할 수 있는 실제적인 가능성을 제시합니다. 다양한 언어 모델과 작업에 대해 철저한 평가를 통해 이 방법의 유용성을 입증하였습니다.



### COMIX: Compositional Explanations using Prototypes (https://arxiv.org/abs/2501.06059)
- **What's New**: 이번 논문에서는 COMiX라는 새로운 방법을 제안하여, 이미지 분류 시 머신러닝 모델의 결정 과정을 인간이 이해할 수 있는 방식으로 해석가능하게 만드는 기법을 소개합니다. 기존의 해석 기법들이 입력의 중요 포인트를 강조하거나 사전 정의된 프로토타입에 의존하는 것과 달리, COMiX는 학습한 개념에 기반하여 이미지의 다양한 영역을 분해하고 훈련 데이터의 관련 부분과 연결합니다. 이를 통해 결정 과정의 신뢰성과 투명성을 높이고, 인간과의 이해 가능성을 강화하는 데 기여하고자 합니다.

- **Technical Details**: COMiX는 신경망의 내부 표현을 선택적으로 분해하여 프로토타입 되는 부분을 도출하고, 이를 훈련 데이터와 매칭하는 방식으로 작동합니다. 이 방법은 기존의 설명 기법과 달리 사후(Post Hoc) 분석이 아닌, 모든 결정 과정이 모델 디자인에 내재된 해석 가능성을 통해 이루어지도록 합니다. 특히, 신뢰성(Fidelity) 및 희소성(Sparsity) 지표에서 기존 방법들에 비해 크게 향상된 결과를 보여주며, 효율성과 효과성을 입증하기 위한 다양한 정성적 및 정량적 실험을 포함하고 있습니다.

- **Performance Highlights**: COMiX는 ImageNet 데이터셋에서 기존의 최첨단(Baseline) 방법에 비해 C-insertion 점수에서 48.82% 향상된 성능을 기록하는 등, 해석 가능성과 정확성 측면에서 뛰어난 성과를 보였습니다. 또한, 이 방법은 임상 데이터 분석 및 고위험 AI 시스템의 판독 등 다양한 안전-critical 응용 분야에서도 유용성을 입증하며, 이러한 사실은 COMiX가 실제 의사결정 과정에서 얼마나 잘 작동하는지를 보여줍니다.



### Investigating the Impact of Observation Space Design Choices On Training Reinforcement Learning Solutions for Spacecraft Problems (https://arxiv.org/abs/2501.06016)
Comments:
          18 pages, 10 figures, 3 tables

- **What's New**: 최근 탐사선 작동을 위한 자율 제어 학습에 대한 강화 학습(Reinforcement Learning, RL) 연구가 향상되고 있다. 특히 조치 공간(action space)의 변경이 학습 환경에서 성능을 개선할 수 있음이 입증된 바 있으며, 이 연구는 관측 공간(observation space)의 변화가 RL 에이전트의 훈련과 성능에 미치는 영향을 분석하는 데 중점을 두고 있다. 이 논문에서는 두 가지 그룹으로 나뉜 연구를 통해 센서와 기준프레임(reference frame)의 변화가 RL 에이전트의 성능에 미치는 영향을 살펴본다.

- **Technical Details**: 이 논문에서 다루는 RL은 심층 강화 학습(Deep Reinforcement Learning, DRL)의 일종으로, 여기서 신경망(Neural Network)을 사용하여 행동 함수를 근사한다. 연구에 사용된 PPO(Proximal Policy Optimization) 알고리즘은 다양한 작업에서 성공적으로 최적 정책을 찾는 데 탁월하다. 본 연구는 RL 에이전트가 조감자 중심(chief-centered)에서 에이전트 중심(agent-centered)으로 기준 프레임을 변경함으로써 성능에 미치는 영향을 평가하고 센서가 학습에 어떻게 영향을 미치는지를 분석한다.

- **Performance Highlights**: 실험 결과에 따르면 센서는 발견된 가장 많은 최적 행동을 학습하는 데 도움을 주지만 필수적이지는 않다는 것을 보여준다. 기준 프레임의 변경은 큰 영향을 미치지 않으나, 일관된 프레임을 유지하는 것이 가장 바람직하다. 이로 인해 RL 에이전트의 성능 향상에 기여할 수 있는 잠재적 요인들을 제시하는 결과를 도출하였다.



### A Neural Operator for Forecasting Carbon Monoxide Evolution in Cities (https://arxiv.org/abs/2501.06007)
Comments:
          36 pages, 21 figures, to be published in npj Clean Air journal (accepted)

- **What's New**: 이 논문은 복잡한 기계 학습 모델인 CoNOAir를 소개하며, 이는 이산화탄소(CO) 농도를 실시간으로 효율적으로 예측하는 데 도움을 줍니다. 기존의 공기 질 모델에 비해 CoNOAir는 더 적은 계산 자원으로 고급 예측을 제공하며, 도시 지역의 대기 오염 개선을 위한 시기적절한 개입을 가능하게 합니다.

- **Technical Details**: CoNOAir는 일반적인 모델인 Fourier Neural Operator(FNO)보다 우수한 성능을 보여줍니다. 이 모델은 단기(시간 단위) 및 장기(72시간) 예측 모두에서 뛰어난 성과를 냈으며, 여러 인도 도시에서 극단적인 오염 사건을 캡처하는 데 매우 효과적입니다. 모든 평가 위치에서 시간별 CO 예측을 위한 R2 값이 0.95 이상을 달성했습니다.

- **Performance Highlights**: CoNOAir는 정부 기관에 조기 경고를 발령하고 타겟 개입 전략을 설계하는 데 효과적인 도구를 제공합니다. 이 연구는 인구 밀집 도시에서 신뢰할 수 있는 실시간 CO 오염 예측을 달성하는 데 한 걸음 더 나아간 것입니다.



### Learning to generate feasible graphs using graph grammars (https://arxiv.org/abs/2501.06003)
- **What's New**: 본 연구에서는 그래프 생성의 복잡한 종속성을 모델링하기 위해 그래프 문법(Graph Grammar) 기반의 새로운 생성 접근법을 제안합니다. 이 방법은 도메인에 따라 정의된 간소화(coarsening) 절차를 도입하여 장거리 종속성을 보다 효율적으로 처리할 수 있는 단축 경로를 제공합니다. 이를 통해 기존의 정보 희석(information dilution) 문제를 해결하였으며, 실험에서는 작은 약물과 RNA 이차 구조 두 가지 도메인에서 그 효과를 입증합니다.

- **Technical Details**: 이 연구는 Costa(2017)의 접근법을 확장하여 사용자 또는 도메인에 정의된 그래프 간소화 절차와 도메인 특화 최적화 기법을 통해 생성된 인스턴스의 유효성을 보장합니다. 메트로폴리스 해스팅스(Metropolis-Hastings, MH) 알고리즘을 사용하여 기본 마르코프 과정의 전이 확률을 제안과 수용-거부 확률 분포로 분리하고, 문법 기반의 제안 분포를 구축하여 효율적인 샘플링을 수행합니다.

- **Performance Highlights**: 고분자 그래프 생성의 질을 평가하기 위해 MOSES 벤치마크를 사용하였으며, 생성된 분자 그래프의 실제 분자와의 거리, 지방 친화성(lipophilicity), 합성 가능성(synthesizability) 및 약물 유사성(drug-likeness)을 종합적으로 분석했습니다. 또한, RNA 가족의 유효한 사례로 확인된 대규모 그래프 생성 테스트에서도 성공적으로 구현되었습니다.



### DeltaGNN: Graph Neural Network with Information Flow Contro (https://arxiv.org/abs/2501.06002)
- **What's New**: 이 논문에서는 기존의 그래프 신경망(GNN) 모델이 과도한 매끄러움(over-smoothing)과 압축(over-squashing)으로 인해 한계가 있다는 점을 지적합니다. 이를 해결하기 위해 "정보 흐름 제어(information flow control)"라는 새로운 메커니즘을 제안하였으며, 이 메커니즘은 새로운 연결성 측정치인 "정보 흐름 점수(information flow score)"를 활용합니다. 이러한 접근 방식은 고전적인 GNN 모델의 문제를 효과적으로 해결하면서 계산 복잡도를 최소화합니다.

- **Technical Details**: 저자들은 그래프의 정보 흐름을 형식화하고 이 정보를 통해 노드 임베딩 업데이트의 속도와 가속을 분석하는 새로운 연결성 측정치를 정의합니다. GNN의 각 레이어는 변환 함수와 메시지 패싱 집계 함수를 적용하여 노드의 특징 벡터와 이웃을 처리하며, 이를 통해 각 노드가 특정 클래스에 매핑될 수 있도록 학습합니다. 이러한 방법은 GNN 아키텍처에 유연하게 통합될 수 있으며, 효율성을 극대화합니다.

- **Performance Highlights**: DeltaGNN이라는 새로운 GNN 아키텍처를 통해 저자들은 제안된 방법론의 유효성을 입증했습니다. 이 모델은 다양한 크기, 토폴로지, 밀도, 동질성 비율(homophilic ratios)을 가진 10개의 실제 데이터 세트에서 벤치마크 테스트를 진행하였고, 우수한 성능을 보여주었습니다. 결과적으로 제안된 방법은 뛰어난 일반화 능력과 확장성을 가지고 있습니다.



### Comparing Self-Supervised Learning Models Pre-Trained on Human Speech and Animal Vocalizations for Bioacoustics Processing (https://arxiv.org/abs/2501.05987)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 연구는 동물의 발성을 직접적으로 사용하여 사전 훈련된 SSL 모델이 인간의 음성을 사용한 모델보다 더 유리한지를 평가하고, 인간의 음성을 사전 훈련한 모델을 자동 음성 인식(ASR) 작업에 연준할 경우 생물음향 분류(bioacoustic classification) 성능을 향상시키는지를 조사합니다. 또한, 다양한 생물음향 데이터 세트를 통해 SSL 모델의 성능을 비교하여 현재 상태를 규명합니다. 연구 결과에 따르면, 생물음향 데이터로 사전 훈련된 모델은 제약된 개선 효과를 보여주며, 사전 훈련된 모델이 성능을 더 높일 필요가 없다는 점이 강조됩니다.

- **Technical Details**: 본 연구는 세 가지 서로 다른 생물음향 데이터 세트를 사용하여 실험을 수행했습니다. 첫 번째로, 해양 포유류의 음성을 포함하고 있는 Watkins 데이터 세트, 두 번째로 마르모셋의 복잡한 사회적 특성을 반영하는 InfantMarmosetsVox 데이터 세트, 마지막으로 개의 다양한 호출 형태를 포함한 Abzaliev 데이터 세트가 사용되었습니다. 각 데이터 세트는 훈련, 검증 및 테스트 세트로 나누어져 있으며, 훈련된 다양한 모델을 통해 특징 표현(Fature representation)을 도출하였습니다.

- **Performance Highlights**: 연구 결과는 인간의 음성으로 사전 훈련된 SSL 모델이 생물음향 데이터에 대해서도 상당한 성능을 보여준다는 것을 나타냅니다. SSL 모델의 사전 훈련은 생물음향 작업에서 일반적인 특징 추출기로서의 효과를 입증했으며, ASR 작업에 대한 미세 조정은 혼합된 결과를 보였습니다. 연구 결과는 음성으로 사전 훈련된 SSL 모델이 생물음향 작업에 대해 이미 잘 조정되어 있음을 시사하며, 따라서 광범위한 미세 조정이 최적 성능에 필요하지 않을 수 있음을 보여줍니다.



### Deep Variational Sequential Monte Carlo for High-Dimensional Observations (https://arxiv.org/abs/2501.05982)
- **What's New**: 본 연구는 비지도 학습(unsupervised learning)을 활용하여 높은 차원의 관측에서 제안 분포와 상태 전이 분포를 신경망(neural network)으로 모수화(parameterize)하는 차별화된 입자 필터(differentiable particle filter)를 제안합니다. 특히 비선형(state-space) 시스템에서 높은 차원의 관측을 통해 기존의 방법들보다 더 향상된 성능을 보여줍니다. 실험 결과, 본 방법이 고차원 및 부분 관측에서 Lorenz attractor를 추적하는 데 있어 기존의 방법들을 초월했다는 점이 강조됩니다.

- **Technical Details**: 본 연구의 핵심은 변별이 가능한 순차 몬테카를로 방법(variational sequential Monte Carlo, VSMC)을 사용하여 제안 분포(proposal distribution)를 학습하는 데 있습니다. 우리는 신경망을 통해 제안 분포와 상태 전이 분포(state transition distribution)를 최적화합니다. 이러한 접근법은 레이블이 없는 데이터에서 비지도적으로 실행되며, 측정 모델이 알려져 있고 미분 가능하다고 가정됩니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 기존의 EKF, BPF 및 감독 회귀 모델(supervised regression model) 등 일반적인 필터링 기법들과 비교했을 때 성능이 우수함을 나타냅니다. 또한, 추정된 로그 주변 우도(log-marginal likelihood estimate)를 최대화함으로써 보다 정확한 사후 분포(posterior distribution)의 표현을 제공함을 입증합니다. 본 연구는 이 분야에서 기여를 인정받을 수 있는 중요한 결과를 제시합니다.



### A Brain Age Residual Biomarker (BARB): Leveraging MRI-Based Models to Detect Latent Health Conditions in U.S. Veterans (https://arxiv.org/abs/2501.05970)
- **What's New**: 이번 연구에서는 1,220명의 미국 재향군인을 대상으로 MRI 스캔 데이터를 사용하여 뇌 나이를 예측하는 새로운 모델을 개발했습니다. 이는 CNN(convolutional neural networks)을 기반으로 하여 T2 가중치의 FSE(fast spin-echo) 및 FLAIR(fluid attenuated inversion recovery) 이미지를 활용한 것으로, 49세 이상의 환자에서 잔여값과 여러 ICD 코드의 상관관계가 관찰되었습니다. 이러한 잔여값은 잠재적인 건강 상태를 탐지하는 바이오마커(biomarker)로서의 가능성을 지니고 있습니다.

- **Technical Details**: 연구에서는 T2 가중치 이미지를 포함한 4가지 CNN 모델을 통해 뇌 나이를 예측하였으며, 이 모델은 polynomial regression(degree-3 다항 회귀)을 통해 통합되었습니다. 각 MRI 스캔은 전두 뇌의 측좌신경 및 날개 통로에서 촬영되었으며, 뇌 노화와 관련된 구조적 변화를 명확히 보여줍니다. 이어서 모델의 성능 평가를 위해 5가지 ICD 코드에 대한 잔여 분석이 수행되어, 뇌 나이와 건강 상태 간의 관계를 확인하였습니다.

- **Performance Highlights**: 모델은 테스트 세트에서 R² 값 0.816을 기록하여 괄목할 만한 성능을 입증하였습니다. 49세 이상의 환자들에서 잔여값이 뇌 노화와 여러 건강 상태 사이의 관계를 강력하게 제시하였으며, 이는 새로운 유형의 바이오마커로서의 잠재력을 강조합니다. 이 연구는 뇌 나이 예측 모델이 건강 상태를 효과적으로 반영할 수 있는 가능성을 보여주며, 향후 연구에 중요한 기초 자료를 제공할 것으로 기대됩니다.



### Model Inversion in Split Learning for Personalized LLMs: New Insights from Information Bottleneck Theory (https://arxiv.org/abs/2501.05965)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 Personalized Large Language Models (LLMs)의 보호와 관련된 중대한 개인 정보 보호 문제를 처음으로 공식적으로 확인하였습니다. 특히 split learning 구조에서 LLM의 model inversion 공격을 밝혀내고 데이터 대표의 개인 정보 유출 가능성을 강조하고 있습니다. 또한 mutual information entropy를 도입하여 Transformer 기반 LLM의 정보 전파를 이해하고, 프라이버시 공격 성능을 평가하는 방법론을 제시합니다.

- **Technical Details**: 이 연구는 두 단계의 공격 시스템을 제안합니다. 첫 번째 단계는 입력 데이터를 임베딩 공간으로 변환하는 과정으로, 두 번째 단계에서는 생성 모델을 사용하여 이 임베딩으로부터 텍스트를 복원합니다. 이러한 구조는 복잡성을 줄이고 다양한 시나리오에서 공격 점수를 38%-75%로 향상시키며, SOTA(State-of-the-Art)보다 60% 이상의 성능 개선을 달성합니다.

- **Performance Highlights**: 리포트를 통해 프라이버시 공격에 대한 체계적인 분석을 제공하며, LLM의 중간 레이어에서의 대표 회복 성능이 매우 낮음을 지적하고 이를 개선하기 위한 두 단계 복원 시스템을 도입한 것이 주효한 것으로 나타났습니다. 이 연구는 split learning 시나리오의 프라이버시 및 보안 문제를 해결하는데 있어 튼튼한 기초를 제공하며, 향후 더욱 강화된 개인 보호 방안 개발의 필요성을 강조합니다.



### Soft regression trees: a model variant and a decomposition training algorithm (https://arxiv.org/abs/2501.05942)
- **What's New**: 이번 연구에서는 소프트 다변량 회귀 트리(Soft Multivariate Regression Trees, SRTs)의 새로운 변형을 제안합니다. 각 입력 벡터에 대해 예측은 단일 잎 노드와 관련된 선형 회귀로 정의됩니다. 이는 입력 벡터가 루트에서 시작해 높은 확률을 가진 가지를 따라 주행되어 도달한 잎 노드를 통해 결정됩니다.

- **Technical Details**: SRT는 조건부 계산 속성(conditional computational property)을 보이며, 각 예측이 적은 수의 노드(파라미터)에 의존합니다. 제안된 비선형 최적화(formulation) 훈련 방식은 분해(decomposition) 가능성을 갖고 있어, 분해 훈련 알고리즘을 제안합니다. 또한, 임계값이 없는 일반적인 분해 방법과 수렴 보장을 논의합니다.

- **Performance Highlights**: 15개의 잘 알려진 데이터셋에 대한 실험 결과, SRT와 분해 알고리즘은 전통적인 소프트 회귀 트리보다 높은 정확도와 견고성을 보여주었습니다. Bertsimas 및 Dunn의 혼합 정수 최적화(mixed-integer optimization) 접근 방식보다도 훈련 시간을 크게 줄였으며, 약간 더 나은 평균 정확도를 달성했습니다. Random Forest 앙상블 방법과의 비교 결과도 포함됩니다.



### Encoded Spatial Attribute in Multi-Tier Federated Learning (https://arxiv.org/abs/2501.05934)
Comments:
          IEEE ICCE 2025

- **What's New**: 본 연구는 지리적 데이터의 집합 모델 평가를 위한 인코딩된 공간 다층 연합 학습(Encoded Spatial Multi-Tier Federated Learning) 접근 방식을 제안합니다. 클라이언트 계층에서 공간 정보를 인코딩하여 타겟 결과 예측을 개선하는 것이 핵심입니다. 이 접근 방식은 다양한 데이터 세트와 공간 속성을 통해 모델 성능을 평가하고 예측 정확도의 변화를 강조합니다.

- **Technical Details**: 다층 연합 학습 구조를 통해 다양한 수준의 공간 속성을 수집하고 처리하여 예측 모델을 구축합니다. 이 연구는 인코딩된 공간 데이터를 활용하여 서로 다른 티어에서 집합값을 산출하며, 글로벌 모델에서 75.62%와 89.52%의 정확도를 달성했습니다. 공간 정보의 복잡성을 이해하고, 실제 데이터에 대해 통찰을 얻기 위해 모델 아키텍처를 세분화했습니다.

- **Performance Highlights**: 이 접근 방식은 실시간 애플리케이션에서의 활용 가능성을 강조하며, 데이터의 공간 분포 및 잠재적 변수들을 분석합니다. 제안된 구조에 따른 실험은 시간 경과에 따른 패턴 변화를 포착하는 데 유용하며, 더 나아가 엄청난 예측 성능 향상을 보여주고 있습니다. 이를 통해 공간 분석 및 기계 학습 모델링의 현실적 응용에 기여하고자 합니다.



### DiffuSETS: 12-lead ECG Generation Conditioned on Clinical Text Reports and Patient-Specific Information (https://arxiv.org/abs/2501.05932)
- **What's New**: DiffuSETS라는 새로운 프레임워크를 통해 ECG 신호를 높은 의미론적 정합성(semantic alignment)과 충실도로 생성할 수 있는 가능성을 제안합니다. 이 프레임워크는 클리니컬 텍스트 보고서와 환자 특화 정보를 입력으로 받아들여 임상적으로 의미 있는 ECG 신호의 생성이 가능합니다. 또한, 표준화된 평가 방법론을 도입하여 ECG 생성의 효과를 평가할 수 있는 기준을 제시합니다.

- **Technical Details**: DiffuSETS의 아키텍처는 세 가지 모달리티( modalities)로 구성되어 있습니다: 신호 공간(signal space), 잠재 공간(latent space), 그리고 조건 정보 공간(conditional information space). 이 모델은 변분 자동 인코더(variational autoencoder)를 사용하여 신호 공간과 잠재 공간 간 변환을 수행하며, 대규모 언어 모델(large language model)을 통해 임상 텍스트 보고서로부터 의미론적 정보를 추출하고, 이를 환자 특화 정보와 결합하여 ECG를 생성합니다. 훈련 데이터는 MIMIC-IV-ECG 데이터셋을 이용하며, 데이터 전처리를 통해 794,372개의 12유도 ECG 신호 기록을 얻었습니다.

- **Performance Highlights**: DiffuSETS는 실험에서 우수한 결과를 보이며 ECG 생성의 우수성을 입증합니다. 신호 수준, 특성 수준 및 진단 수준에서의 포괄적인 평가를 통해 생성된 ECG 신호가 실제 ECG 신호를 얼마나 유사하게 재현하는지를 조사합니다. 이 모델은 데이터 부족 문제를 완화하고, 심장병 교육 및 의료 지식 발견 등 새로운 응용 가능성을 탐색하는 데에도 기여할 수 있는 잠재력을 지니고 있습니다.



### A Neighbor-based Approach to Pitch Ownership Models in Soccer (https://arxiv.org/abs/2501.05870)
- **What's New**: 이 논문은 축구 경기에서 피치 소유 모델(pitch ownership model)을 구축하기 위한 새로운 접근 방식을 제안합니다. 기존의 이벤트 기반 분석(event-based analysis)과 달리, 트래킹 데이터(tracking data)는 선수의 위치와 같은 중요한 맥락 정보를 통합하여 게임의 동력을 이해하는 데 도움이 됩니다. K-최근접 이웃(K-Nearest Neighbors, KNN) 알고리즘을 활용하여 빠르고 유연한 모델을 제공하여 다양한 피치 통제를 모델링할 수 있는 장점을 갖추었습니다.

- **Technical Details**: 제안된 접근 방식은 KNN 알고리즘을 사용하여 피치 소유 모델을 구축합니다. 이 모델은 단지 세 가지 하이퍼파라미터(hyperparameters)를 조정하여 다양한 플레이어의 기술 수준에 맞게 조정할 수 있는 유연성을 제공합니다. 특히, 거리 감소(distance decay) 모델과 스무딩(smoothing) 기법을 도입해 모델에 불확실성을 통합할 수 있습니다. 이러한 특징은 낮은 리그에서의 높은 불확실성을 모델링하는 데 유용합니다.

- **Performance Highlights**: 논문에서는 모델의 강점과 약점을 설명하는 여러 예시를 시각적으로 설명합니다. KNN 알고리즘을 활용한 피치 소유 모델은 Voronoi 다이어그램을 바탕으로 하여 다양한 기존 접근 방식을 수용할 수 있게 설계되었습니다. 이러한 새로운 접근 방식은 전통적인 방법 이상의 통찰력을 제공하여 전술 분석가(tactical analysts)에게 유용한 도구를 제공합니다.



### "Cause" is Mechanistic Narrative within Scientific Domains: An Ordinary Language Philosophical Critique of "Causal Machine Learning" (https://arxiv.org/abs/2501.05844)
- **What's New**: 인공지능(AI)과 머신러닝에서 인과학습(Causal Learning)은 최근 주요 주제로 부상하고 있으며, 다양한 분야에서 진정한 원인과 결과의 본질을 드러낼 수 있는 특별한 기술을 약속하고 있습니다. 본 연구는 인과관계의 관념이 어떤 것인지를 검토하고, 과학적 영역에서 '원인'이라는 용어의 일반적인 사용을 통해 실천적으로 유효한 인과 주장에 대해 탐구합니다. 이를 통해 사회과학에서의 해석적 지식(Hermeneutic knowledge)의 중요성과 복잡한 시스템의 열려 있는 특성을 알린다는 점에서 의미가 있습니다.

- **Technical Details**: 인과 추론의 기초가 되는 독립성 기준(Independence criteria)은 Judea Pearl의 프레임워크에 특히 크게 영향 받고 있습니다. 이 프레임워크에서는 확률적 그래픽 모델(Probabilistic Graphical Models, PGMs)과 특히 베이즈 네트워크(Bayesian Networks, BNs)를 사용하여 변수 간의 인과관계를 나타내고 추론합니다. 베이즈 네트워크에서 각 노드는 랜덤 변수에 해당하고, 노드 간의 방향된 엣지(Directed edges)는 확률적 종속성을 나타내는 구조입니다.

- **Performance Highlights**: 이 논문은 인과관계 모델링과 알고리즘에 대한 일반적인 프로그램을 주장하며, 과학적 패러다임에 정의된 시스템의 귀납적 편향(Inductive bias)이 적절하게 적용될 수 있음을 강조합니다. 또한, 다양한 과학적 분야에 걸쳐 인과관계의 의미를 정의하고, 과학적 절차를 충실히 나타내는 데이터 기반 스키마를 제시합니다. 마지막으로, 과학적 언어에서 고차원 개념적 이해를 보여주는 체계화된 유추(Systematized analogies)를 통해 인과 모델의 학습 절차를 포괄하는 방안을 제안합니다.



### Orthogonal projection-based regularization for efficient model augmentation (https://arxiv.org/abs/2501.05842)
Comments:
          Submitted to L4DC 2025

- **What's New**: 이 논문은 비선형 시스템 식별을 위한 모델 강화 방법을 제안하고 있습니다. 특히, 물리적 지식을 모델 구조에 통합함으로써 기존의 블랙박스 모델들이 갖고 있는 해석 가능성 부족 문제를 해결하고자 합니다. 논문에서는 정규화 방법을 도입하여 모델의 파라미터 학습과 수렴성을 개선하며, 물리 기반 모델과 머신러닝 컴포넌트의 조화를 이룹니다.

- **Technical Details**: 제안된 방법은 비선형 매개변수를 갖는 물리 기반 모델의 정규화를 위한 직교 투영 기반 기법을 일반화합니다. 이 접근법은 모델의 학습 컴포넌트가 물리 기반 모델의 알려진 동역학을 학습하는 것을 방지하여 불필요한 과파라미터화를 줄였습니다. 또한, 추가적인 모델 보강 구조를 통해 기존의 물리 기반 손실 함수를 활용하며, 물리적 매개변수와 학습 컴포넌트의 매개변수를 공동으로 추정하는 구조를 구성합니다.

- **Performance Highlights**: 이 방법은 선형 및 비선형 모델 모두에서 우수한 성능을 보여주며, 시뮬레이션 연구를 통해 그 유효성을 입증하였습니다. 각 모델은 급격한 수렴 및 정확도를 보이며 물리적으로 해석 가능한 성질을 유지합니다. 이러한 특징들은 자율주행차와 같은 복잡한 시스템의 동적 모델링에 필수적이며, 공학적 응용 가능성을 넓힐 것으로 기대됩니다.



### Fine-tuning is Not Fine: Mitigating Backdoor Attacks in GNNs with Limited Clean Data (https://arxiv.org/abs/2501.05835)
- **What's New**: 이 논문은 Graph Neural Networks (GNNs)의 백도어 공격에 대한 새로운 완화 프레임워크인 GRAPHNAD를 제안합니다. 최근 GNNs가 메시지 전달 메커니즘을 통해 뛰어난 성능을 보이지만, 백도어 공격에 취약하다는 문제점이 드러났습니다. 기존의 방어 기법은 대량의 청정 데이터에 의존하는데, 이 논문에서는 제한된 데이터에서 고품질 중간 레이어 표현을 캡처하여 방어 효과를 높이는 방법을 모색합니다.

- **Technical Details**: GRAPHNAD는 그래프 내 적절한 주의 표현을 식별하고 제한된 데이터로 증류 과정을 향상시키는 두 가지 주요 질문에 초점을 맞추고 있습니다. 이 과정에서 GNN의 중간 층 주의 표현을 정렬하고, 백도어 모델의 관계 맵과 교사 모델의 관계 맵을 일치시켜 모델 정확성을 보장합니다. 이 방식을 통해 백도어 뉴런을 무해한 형태로 변환하도록 유도하며, 그래프 주의 전달 방법을 adopt하여 효과성을 극대화합니다.

- **Performance Highlights**: 실험 결과, GRAPHNAD는 청정 데이터의 3%만으로도 공격 성공률을 5% 미만으로 줄일 수 있음을 보여줍니다. 이는 기존 방법들의 한계를 극복하고, GNNs에 대한 백도어 방어의 새로운 가능성을 제시합니다. 이러한 수치는 제한된 데이터 환경에서도 효과적인 방어 기법으로서의 가능성을 나타냅니다.



### Diffusion Models for Smarter UAVs: Decision-Making and Modeling (https://arxiv.org/abs/2501.05819)
Comments:
          7 pages, 2 figures

- **What's New**: 이번 논문은 무인 항공기(UAV) 통신에서 강화 학습(Reinforcement Learning, RL)과 디지털 트윈(Digital Twin, DT)을 결합하여 발생하는 다양한 문제를 해결하는 방안으로 새로운 생성 AI 기법인 확산 모델(Diffusion Models, DMs)의 통합을 탐구합니다. DMs는 기존의 방법론과는 달리, 데이터를 통해 학습된 확률 분포를 기반으로 신뢰할 수 있는 새로운 패턴을 생성할 수 있는 강력한 도구입니다. 이를 통해 데이터 부족 문제를 해결하고 RL과 DT의 성능을 개선하는데 기여할 수 있다는 점에서 주목받고 있습니다.

- **Technical Details**: UAV는 공공 안전, 에너지, 농업 및 스마트 시티와 같은 다양한 분야에서 사용됩니다. 이들은 5G 및 6G 네트워크의 필수 요소로, 기계 간 통신 및 고속 데이터 전송을 가능하게 합니다. 그러나 UAV 통신에서의 의사결정 과정은 복잡하며, RL은 낮은 샘플 효율성 때문에 이 과정에서 제한됩니다. DMs는 RL 모델의 샘플 효율성을 개선하고, 실제 훈련 환경을 생성하는 데 도움이 될 수 있습니다.

- **Performance Highlights**: 이 연구는 DMs가 RL 기반 UAV 통신을 어떻게 개선할 수 있는지를 보여주며, 이는 샘플 효율성을 높이고 정책 네트워크를 개선하며, 신뢰할 수 있는 훈련 환경을 생성하는 방식입니다. DT 시스템 내에서 DMs의 활용은 데이터 부족 문제를 해결하고, 의사결정 과정을 개선하며, 동적 모델링을 정교화 하는 데 중요한 역할을 할 것입니다. 이러한 통합은 UAV 통신에서의 적응력과 실시간 성능을 크게 향상시킬 것으로 기대됩니다.



### AdaPRL: Adaptive Pairwise Regression Learning with Uncertainty Estimation for Universal Regression Tasks (https://arxiv.org/abs/2501.05809)
Comments:
          22 pages, 11 figures

- **What's New**: 본 논문에서는 기존의 점별 학습 방식의 한계를 극복하기 위해 적응형 쌍 학습 프레임워크인 AdaPRL (Adaptive Pairwise Regression Learning)을 제안합니다. 이 프레임워크는 데이터 포인트 간의 상대적 차이를 활용하고, 예측의 불확실성을 정량화하는 심층 확률 모델과 통합되어 있습니다. 이를 통해 기존 회귀 모델의 예측 정확도와 강건성을 향상시키며, 다양한 데이터셋에 대한 성능 검증을 통한 강력한 결과를 도출합니다.

- **Technical Details**: AdaPRL은 원래의 회귀 작업과 보조 불확실성 기반 학습 작업을 결합하여 모델이 더 정확한 점별 예측을 생성할 수 있도록 설계되었습니다. 이 접근법은 예측에 대한 상대적 차이를 추정함으로써 데이터 내의 기본 관계 및 의존성을 더 효과적으로 파악합니다. 또한, AdaPRL은 단일 출력 예측을 넘어 다중 작업 학습 및 다변량 시계열 예측에 맞게 조정될 수 있습니다.

- **Performance Highlights**: AblePRL은 추천 시스템, 시계열 예측, 자연어 이해, 금융 및 산업 데이터셋 등 다양한 분야에서 실험을 통해 기존의 회귀 모델보다 뛰어난 성능을 보였습니다. 주목할 만한 성능 향상으로는 예측 정확도 및 순위 능력의 향상, 노이즈가 있는 데이터에 대한 강건성 개선, 낮은 데이터에 대한 복원력 증가 등이 포함되어 있습니다. 결론적으로, AdaPRL은 현대 회귀 모델의 일반화 능력과 해석 가능성을 증가시킬 수 있는 잠재력을 지니고 있습니다.



### Alignment without Over-optimization: Training-Free Solution for Diffusion Models (https://arxiv.org/abs/2501.05803)
- **What's New**: 이 연구에서는 Diffusion model의 목표를 정렬하는 데 어려움이 있었던 점을 해결하기 위해, Sequential Monte Carlo (SMC) 기반의 새로운 훈련 없는 샘플링 방법인 Diffusion Alignment as Sampling (DAS)를 제안합니다. DAS는 모델의 일반성을 유지하면서도 효과적인 보상 정렬을 달성합니다. 기존 방법들이 보상 최적화의 문제로 인해 성능이 저하되는 것을 방지하면서, 목표 보상을 효과적으로 샘플링할 수 있도록 설계되었습니다.

- **Technical Details**: DAS는 다수의 후보 latent를 활용하여 높은 보상 샘플로 유도하는 방식으로 구성되어 있습니다. 이를 통해 샘플링에서의 오류를 평균화하고, 보상 정렬된 목표 분포에서의 샘플링을 가능하게 합니다. 이 과정에서는 온도 조정(tempering) 기법을 사용하여 중간 목표 분포를 신중하게 설계함으로써 샘플링 효율성을 극대화하고 있습니다.

- **Performance Highlights**: DAS는 Stable Diffusion v1.5 및 CLIPScore와 같은 더 복잡한 보상 함수에 적용되며, 기존의 미세 조정 방법에 비해 뛰어난 성능을 입증하고 있습니다. 결과적으로, DAS는 단일 보상 최적화 및 다중 목표 최적화에서 새로운 Pareto front를 달성하였으며, 온라인 환경에서도 뛰어난 샘플링 능력을 보여주어 기존 방법들보다 20% 이상 개선된 성과를 달성했습니다.



### Robust Counterfactual Explanations under Model Multiplicity Using Multi-Objective Optimization (https://arxiv.org/abs/2501.05795)
Comments:
          19 pages

- **What's New**: 최근 기계 학습에서 설명 가능성(explainability)의 중요성이 증가하고 있습니다. 이 연구에서는 counterfactual explanation(CE)을 통해 새롭고 견고한 CE를 제안하며, Pareto improvement와 다중 목표 최적화(multi-objective optimization)를 활용하여 이를 구현합니다. 다양한 기계 학습 모델의 존재에 따른 문제를 해결하는데 초점을 맞추고 있으며, 안정적인 의사결정을 위한 기여를 목표로 합니다.

- **Technical Details**: 이 논문에서는 n 쌍의 스칼라와 r 차원의 벡터로 구성된 데이터 집합 𝒟 (𝑦𝑖, 𝑋𝑖) 설정을 통해 문제를 정립합니다. CE는 기계 학습 모델이 예측 결과를 얻기 위해 원본 데이터를 최소한으로 변경해야 하는 방법을 설명합니다. 다중 목표 최적화는 모델 간의 일관성 있는 CE를 생성하기 위해 필요하며, 다양한 조건을 고려하여 CE를 도출하는 방법이 제안됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 견고하고 유용함을 보여줍니다. 또한, 이 방법은 위험한 문제에 대해 안전한 솔루션을 선택하는 데 실용적으로 활용될 수 있습니다. 특히, 사용자의 선호에 따라 다양한 CE를 선택할 수 있는 가능성을 제공하며, 향후 의사결정 및 기계 학습 기반의 행동 계획 연구에도 큰 영향을 미칠 것으로 기대됩니다.



### STHFL: Spatio-Temporal Heterogeneous Federated Learning (https://arxiv.org/abs/2501.05775)
- **What's New**: 이 논문은 새로운 연합 학습( federated learning) 설정인 Spatio-Temporal Heterogeneity Federated Learning (STHFL)을 제안합니다. 이는 다양한 데이터 분포로 인해 발생하는 spatio-temporal(시공간적) 이질성을 고려한 것입니다. 특히, 각 클라이언트의 모델이 개인화된 레이어를 포함하고 있어, 서로 다른 데이터 분포에 동적으로 적응할 수 있도록 설계되었습니다.

- **Technical Details**: Global-Local Dynamic Prototype (GLDP) 프레임워크는 STHFL을 위한 혁신적인 모델이며, 각 클라이언트에서 훈련된 개인화된 프로토타입과 공통 프로토타입을 활용합니다. GLDP는 긴 꼬리(long-tailed) 데이터 분포를 다루기 위해, 클라이언트 간에 보완 지식으로 활용되는 글로벌 프로토타입을 설정합니다. 또한, 이동 평균 방법을 사용하여 글로벌 및 로컬 프로토타입을 업데이트하여 시간의 경과에 따른 데이터 변화를 효과적으로 처리합니다.

- **Performance Highlights**: 제안된 GLDP 방법은 STHFL 시나리오에서 기존의 최신 방법들에 비해 뛰어난 성능을 보여주며, 다양한 클라이언트 참여자들이 강력한 일반화 능력을 가진 글로벌-로컬 모델을 훈련할 수 있도록 보장합니다. 이 연구는 기존 방법들이 간과했던 spatio-temporal(시공간적) 이질성을 다룸으로써 연합 학습 조건의 복잡성을 개선하는 데 기여합니다.



### rmlnomogram: An R package to construct an explainable nomogram for any machine learning algorithms (https://arxiv.org/abs/2501.05772)
Comments:
          16 pages, 2 figures, 1 table, 3 equations, 1 algorithm, 4 code snippets

- **What's New**: 이번 연구에서는 의료 환경에서의 모델 배포를 가속화하고 모델 가용성을 개선하기 위해, 모든 머신 러닝(ML) 알고리즘에 대한 노모그램(nomogram)을 생성할 수 있는 R 패키지와 웹 애플리케이션을 개발했습니다. 기존의 노모그램은 오직 회귀 알고리즘에만 적용 가능했지만, 본 연구에서 개발한 도구는 다양한 알고리즘에 유연하게 대응할 수 있습니다.

- **Technical Details**: 이 연구에서는 ML 예측 모델을 노모그램으로 변환하는 함수를 정식화하였습니다. 이 과정에서 필요한 데이터셋은 (1) 예측기 값의 모든 가능한 조합, (2) 모델의 해당 출력 및 (3) 각 예측기에 대한 설명 가능성 값(선택 사항)을 포함해야 합니다. 이를 통해 다양한 유형의 조합을 제공하며, 최대 15개의 예측기와 3,200개의 조합을 지원합니다.

- **Performance Highlights**: 개발된 R 패키지는 범주형 예측기와 이진 결과에 대해 다양한 방법으로 노모그램을 생성할 수 있으며, 이를 통해 의료 분야에서 ML 모델의 설명 가능성과 투명성을 확보할 수 있게 되었습니다. 웹 애플리케이션 또한 제공되어 사용자가 손쉽게 노모그램을 생성하고 설명 가능성 값을 활용할 수 있습니다.



### Halal or Not: Knowledge Graph Completion for Predicting Cultural Appropriateness of Daily Products (https://arxiv.org/abs/2501.05768)
Comments:
          10 pages

- **What's New**: 이번 연구는 할랄 화장품의 상태를 예측하기 위한 새로운 프레임워크, HaCKG(할랄 화장품 추천 프레임워크)를 제안합니다. 이 방법은 화장품과 그 성분 간의 관계를 명시적으로 모델링하여 할랄 상태를 예측합니다. 기존의 접근 방식들이 개별 성분 분석에 초점을 맞춘 반면, HaCKG는 지식 그래프(knowledge graph)를 활용하여 복잡한 상호 관계를 포착합니다.

- **Technical Details**: HaCKG는 화장품 관련 지식을 그래프 형태로 구성하고, 이러한 지식을 바탕으로 Relational Graph Attention Network (r-GAT) 모델을 사용합니다. 이 모델은 화장품 성분의 구조적 관계를 학습하기 위해 Residual Connection을 포함하고 있습니다. 또한, Self-Supervised Learning (SSL) 방식으로 사전 훈련된 모델을 내려받아 할랄 상태를 예측하는 데 필요한 구체적인 데이터로 미세 조정(fine-tuning)됩니다.

- **Performance Highlights**: 다양한 화장품 데이터셋을 기반으로 한 광범위한 실험 결과, HaCKG는 기존의 최첨단 모델들보다 우수한 성능을 나타냈습니다. 할랄화장품 예측 과제에서 실험된 결과는 명확하며, 이는 기존 방법론의 한계를 극복하고 더 넓은 맥락을 고려한 결과입니다.



### Diving Deep: Forecasting Sea Surface Temperatures and Anomalies (https://arxiv.org/abs/2501.05731)
Comments:
          The paper contains 9 pages for the main text and 10 pages including References. 5 figures. Discovery Track, European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2024

- **What's New**: 이번 논문은 2024년 유럽 기계 학습 및 지식 발견 회의(ECML PKDD)에서 진행된 "Diving Deep: Forecasting Sea Surface Temperatures and Anomalies Challenge"의 결과를 체계적으로 정리한 것입니다. 이 챌린지는 기후 예측, 생태계 관리, 어업 관리 및 기후 변화 모니터링에 중요한 요소인 전 세계 해수면 온도(SST)의 예측 가능성을 탐구하는 데 중점을 두었습니다. 참가자들은 역사적 데이터를 기반으로 SST 이상(SSTA)을 3개월 선행하여 예측하는 다양한 머신 러닝 접근법을 사용했습니다.

- **Technical Details**: 챌린지에서는 SST, 평균 해수면 압력(MSLP), 공기 온도(T2M)와 같은 추가 특성과 함께 SST의 데이터를 ERA5에서 활용하였습니다. 데이터셋은 1940년 1월부터 현재까지의 월별 데이터를 포함하며, SSTAs는 각 월에 대한 기후값을 차감하여 추출됩니다. 참가자들은 2011년 1월부터 2023년 9월까지의 역사적 SST, MSLP, T2M 값을 이용하여 SSTAs를 예측하는 과제를 수행하였고, 추가적으로 발틱해(Baltic Sea)의 세 위치에 대해 2024년 9월의 SSTAs 예측도 요구되었습니다.

- **Performance Highlights**: 수상자는 솔루션과 보고서 제출, 그리고 발틱해 위치에 대한 예측 정확도를 기준으로 선정되었습니다. 참가자들은 데이터 분석을 통해 머신 러닝 모델을 적용하고 최적의 결과를 도출하고자 시도하였으나, 초기 모델은 특정 기간에 제가된 데이터를 활용했을 때 과적합 문제를 경험했습니다. 최종적으로 간단한 Bayesian Ridge 모델을 사용하여 예측 성능을 개선하고, 적은 데이터에서의 오버피팅 문제를 해결하는 접근법이 가장 유망하다는 결과를 도출하였습니다.



### Element-wise Attention Is All You Need (https://arxiv.org/abs/2501.05730)
- **What's New**: 이번 연구에서는 전통적인 self-attention (SA) 메커니즘의 성능을 유지하면서도 훈련 및 추론에서의 복잡도를 낮출 수 있는 새로운 요소 단위 주의 메커니즘(element-wise attention mechanism)을 제안합니다. 이 메커니즘은 유클리드 거리의 제곱을 이용하여 유사성을 계산하며, Taylor 다항식을 사용하여 SA의 제곱 복잡도 항을 근사합니다.

- **Technical Details**: 제안된 요소 단위 주의 메커니즘은 훈련 중에 계산 복잡도가 \mathcal{O}(tLD)로, 긴 시퀀스 훈련에서 매우 계산적이고 메모리 효율성을 보여줍니다. 여기서 L은 시퀀스 길이, D는 특성 차원, t는 다항식의 최고 차수를 나타냅니다. 또한, 추론 단계에서는 순환 신경망(RecNN)으로 재형식화하여 \mathcal{O}(tD)의 추론 복잡도를 달성합니다.

- **Performance Highlights**: 제안된 요소 단위 주의는 기존 접근 방식에서 나타나는 성능 저하 요소를 피하면서도 SA와 유사한 성능을 달성합니다. 이는 인과적 및 비인과적 형태 모두에서 이루어지며, 학습 과정에서 더 나은 spikiness를 유지할 수 있도록 합니다.



### TransPlace: Transferable Circuit Global Placement via Graph Neural Network (https://arxiv.org/abs/2501.05667)
Comments:
          Accepted at KDD 2025

- **What's New**: 이 논문은 TransPlace라는 새로운 글로벌 배치(framework)를 소개합니다. 이는 혼합 크기의 수백만 개의 셀을 연속 공간에 배치하기 위해 학습하는 방식을 도입합니다. 기존의 글로벌 플레싱에서는 각 회로 설계를 개별적으로 해결하는 한계를 지니고 있었으나, TransPlace는 이를 극복하여 효율성과 성능을 향상시키고자 합니다.

- **Technical Details**: TransPlace는 여러 기술적 요소로 구성됩니다. 첫째, Netlist Graph를 통해 회로의 토폴로지를 모델링합니다. 둘째, Cell-flow와 상대 위치 인코딩을 도입하여 SE(2)-불변의 표현을 학습하고, 셋째, 맞춤형 그래프 신경망 아키텍처인 TPGNN을 통해 배치 지식을 파라미터화합니다. 마지막으로, 투 단계 전략을 통해 거칠고 세밀한 배치를 조정합니다.

- **Performance Highlights**: TransPlace는 기존의 최신 배치 방법에 비해 1.2배의 속도 향상과 30%의 혼잡도 감소, 9%의 타이밍 향상 및 5%의 배선 길이 감소를 실현한다고 보고합니다. 이는 고품질 배치에서 훈련된 TransPlace가 이전의 해결책에 비해 현저한 성능 개선을 보여줌을 의미합니다.



### TAMER: A Test-Time Adaptive MoE-Driven Framework for EHR Representation Learning (https://arxiv.org/abs/2501.05661)
Comments:
          8 pages, 3 figures, 7 tables

- **What's New**: 본 논문에서 TAMER는 EHR (Electronic Health Records) 기반의 예측 모델링에서 환자군 이질성과 데이터 분포의 변화를 다루기 위해 Mixture-of-Experts (MoE)와 Test-Time Adaptation (TTA) 방법론을 결합한 새로운 프레임워크입니다. 이 시스템은 환자 개개인의 건강 상태에 따라 실시간으로 모델을 조정할 수 있는 능력을 제공합니다.

- **Technical Details**: TAMER는 MoE와 TTA 계층을 배치하여 환자 데이터를 처리합니다. TTA 계층은 기존의 백본 모델을 기반으로 환자에 대한 히든 표현을 재구성하려고 하며, 이는 MSE (Mean Squared Error) 손실 함수를 기반으로 동적으로 업데이트되는 다층 신경망 구조로 구현됩니다. MoE 모듈은 각 환자를 다양한 전문가 네트워크에 할당하여 더 세밀하고 포괄적인 환자 표현을 형성합니다.

- **Performance Highlights**: 실험 결과 TAMER는 사망률 및 재입원 예측 작업에서 넷 개의 실제 EHR 데이터셋에 걸쳐 일관되게 성능이 향상되는 것을 보여주었습니다. 기존의 EHR 모델보다 환자 데이터의 동적 특성과 이질성을 효과적으로 처리하여 실질적인 임상 응용에서 유망한 성과를 제공합니다.



### Efficient Representations for High-Cardinality Categorical Variables in Machine Learning (https://arxiv.org/abs/2501.05646)
Comments:
          2025 International Conference on Advanced Machine Learning and Data Science (AMLDS 2025)

- **What's New**: 이 논문은 고차원 범주형 변수를 효과적으로 인코딩하기 위한 새로운 기술들을 소개하고 있습니다. 전통적인 one-hot encoding의 한계를 극복하기 위해, means encoding, low-rank encoding, multinomial logistic regression encoding과 같은 혁신적인 인코딩 기법이 제안됩니다. 이러한 기법은 범주형 데이터의 압축되고 정보성 있는 임베딩을 생성할 수 있도록 설계되었습니다. 이로 인해 모델 성능과 계산 효율성을 크게 향상시킬 수 있습니다.

- **Technical Details**: 논문에서는 그룹 멤버십을 변환하기 위한 효과적인 매핑 ψ(psi)를 개발하였습니다. 이 매핑은 범주형 변수 Gi(Gi)와 결과 변수 Yi(Yi)의 관계를 수학적으로 나타내며, 이를 통해 p+k 차원의 실수값 특성(Xi, ψ(Gi))을 갖는 표준 회귀 문제로 단순화됩니다. 필요한 정보를 추출하는 주 요소 사항은 기본 레마로 설명되며, 추가적으로 이를 기반으로 한 다양한 방법들이 제안됩니다. 또한, 충분한 잠재 상태 가정(sufficient latent state assumption)을 통해 그룹 멤버십과 결과 간의 간접적 관계를 정의하고 있습니다.

- **Performance Highlights**: 제안된 인코딩 기법들은 다양한 데이터셋에서 평가되었으며, 기존의 기준 방법들에 비해 모델 성능과 계산 효율성에서 최고의 개선 효과를 보였습니다. 특히 대규모 데이터셋이 요구되는 분야에서 이러한 기술들이 유용하게 활용될 수 있음을 입증하였습니다. 결과적으로, 이 연구는 머신 러닝에서 더욱 강력하고 효율적인 애플리케이션으로 나아가는 길을 열어주고 있습니다.



### Enhancing Unsupervised Graph Few-shot Learning via Set Functions and Optimal Transpor (https://arxiv.org/abs/2501.05635)
Comments:
          KDD2025

- **What's New**: 이번 연구에서는 STAR라는 새로운 모델을 제안합니다. STAR는 Set funcTions와 optimAl tRansport를 활용하여 비지도 그래프 몇 샷 학습을 향상시킵니다. 이 모델은 세트 레벨의 특징을 추출하고, 지원 집합과 질의 집합 간의 분포 이동 문제를 해결하는 데 중점을 둡니다. STAR는 보다 많은 작업 관련 정보를 포착할 수 있도록 이론적으로 분석되었습니다.

- **Technical Details**: STAR 모델은 메타 훈련 단계에서 그래프 대조 학습(Graph Contrastive Learning)을 사용하여 의미 있는 인스턴스 레벨 표현을 추출합니다. 또한, 지원 및 질의 집합 간의 분포 이동을 최소화하기 위해 최적 수송(Optimal Transport) 원칙을 적용합니다. 이를 통해 지원 집합을 조정하여 분포 이동의 부정적인 영향을 완화하는 방향으로 작동합니다. 실험적으로, STAR는 여러 데이터 세트에서 타 모델을 초과하여 일관되게 우수한 성능을 보였습니다.

- **Performance Highlights**: STAR는 다양한 벤치마크 데이터 세트에서 실험하여 기존 모델에 비해 최첨단 성능을 달성했습니다. 이 모델은 인스턴스 레벨 특징만 추출했던 이전 접근방식의 한계를 극복하고, 세트 레벨의 특징을 효과적으로 얻는 데 성공했습니다. 이론적으로도 STAR가 모델 일반화 가능성을 개선하고, 일반화 오류의 상한을 낮출 수 있음을 입증했습니다.



### Regularized Top-$k$: A Bayesian Framework for Gradient Sparsification (https://arxiv.org/abs/2501.05633)
- **What's New**: 이 논문에서는 분산 환경에서의 경량화(gradient sparsification) 제어를 위한 새로운 방법을 제안합니다. 이는 에러 누적이 특정 수준을 초과하면서 비선택된 경량화 항목이 선택되도록 하고, 경량화가 학습율(learning rate) 조정으로 작용하는 방식을 활용합니다. 이 접근 방식은 분산 경량화의 느린 수렴을 방지할 수 있지만, 어떤 환경에서는 수렴을 저하시킬 수 있습니다.

- **Technical Details**: 제안된 방식은 첫째로 경량화를 역확률(inverse probability) 문제로 수립하고, 둘째로 최대 사후 추정(maximum-a-posteriori estimator)을 사용하여 베이지안 최적 경량화 마스크(Bayesian optimal sparsification mask)를 도출합니다. 이를 통해 Top-$k$에서 파생된 prior distribution을 이용하여 새로운 경량화 알고리즘이 개발되었습니다. 이 알고리즘은 정규화된 형태의 Top-$k$(Regularized Top-$k$, RegTop-$k$)로 해석되며, 과거의 누적 경량화를 이용하여 다음 누적의 후방 통계(posterior statistics)를 평가합니다.

- **Performance Highlights**: RegTop-$k$는 분산 선형 회귀(distributed linear regression)의 경우, Top-$k$가 글로벌 최적(global optimum)에서 고정된 거리를 유지하는 동안, RegTop-$k$는 훨씬 높은 압축률(compression ratios)로 글로벌 최적에 수렴하는 것을 관찰했습니다. 또한, CIFAR-10에서 ResNet-18을 분산 훈련할 때 RegTop-$k$를 사용하여 Top-$k$보다 더 뛰어난 성능을 발휘함을 보여주었습니다.



### Advancing Personalized Learning Analysis via an Innovative Domain Knowledge Informed Attention-based Knowledge Tracing Method (https://arxiv.org/abs/2501.05605)
- **What's New**: 이 논문에서는 지식 추적(Knowledge Tracing, KT) 모델의 혁신적인 주의 기반 방법을 제안하여 교육 과정 내 지식 개념 경로를 효과적으로 통합했습니다. 기존 모델들이 단기적인 상호작용에만 초점을 맞추었던 점을 개선하기 위해, 지식 개념 간의 계층적 의존성을 포착할 수 있는 새로운 방법론을 도입했습니다. 이를 통해 학생들의 학습 성과를 더욱 향상시킬 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 Learning Relevance Matrix라는 새로운 메커니즘을 도입하여 질문 간의 관계를 추적하고, 관련 없는 질문에 대한 주의 점수를 마스킹(masking)하는 방식으로 주의 메커니즘을 개선했습니다. 이를 통해 모델이 구체적인 지식 개념 경로를 기반으로 하여 학생의 학습에 더욱 적합한 예측을 할 수 있도록 합니다. 제안하는 방법은 계층적 의존성을 효과적으로 캡처하여 기존 선진 기술보다 향상된 성능을 발휘하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방식이 AUC(Area Under Curve)와 정확성에서 기존 SOTA(Sate-of-the-Art) KT 방법들에 비해 유의미한 개선이 이루어졌음을 보여주었습니다. 또한, 제안한 방법은 학생의 학습 과정을 보다 구체적으로 해석할 수 있는 가능성을 제공하여, 개인화된 학습 시스템의 미래 연구 방향에 기여할 수 있을 것으로 기대하고 있습니다.



### Session-Level Dynamic Ad Load Optimization using Offline Robust Reinforcement Learning (https://arxiv.org/abs/2501.05591)
Comments:
          Will appear in KDD 2025

- **What's New**: 이 논문은 세션 수준의 동적 광고 로드 최적화 분야의 새로운 접근 방식에 대해 논의합니다. 기존의 인과 학습 기반 방법들이 처리하기 어려운 혼란 편향(confounding bias)과 분포 변화(distribution shift) 문제를 해결하기 위해 오프라인 딥 Q-네트워크(Deep Q-Network, DQN) 기반 프레임워크를 개발하였습니다. 이 연구는 80% 이상의 오프라인 성과 개선을 달성하며, 실제 광고 전달 데이터에서도 5% 추가 증대 효과를 보였습니다.

- **Technical Details**: 세션 수준의 동적 광고 로드 최적화 문제는 마르코프 결정 프로세스(Markov Decision Process, MDP)로 공식화되었습니다. 현재 세션에서의 결정은 이전 세션의 광고 로드 결정(action) 기록을 포함하여 혼란 편향을 완화합니다. 또한, 새로운 오프라인 로버스트 듀엘링 DQN 방식을 통해 이전 세션의 행동이 현재 세션에 미치는 영향을 고려하여, 실제 환경의 분포 변화에 강인한 성능을 보여줍니다.

- **Performance Highlights**: 제안된 프레임워크는 다수의 프로덕션 시스템에 배포되어 탁월한 비즈니스 성과를 달성했습니다. 온라인 A/B 테스트 결과, 광고 점수와 사용자 참여 간의 균형에서 평균 두 자릿수 개선을 관찰하였으며, 이는 소비자와 광고주 모두에게 효율적인 서비스를 제공할 수 있는 능력을 크게 향상시켰습니다.



### Enforcing Fundamental Relations via Adversarial Attacks on Input Parameter Correlations (https://arxiv.org/abs/2501.05588)
Comments:
          12 pages, 8 figures (Without appendix)

- **What's New**: 이번 논문에서는 랜덤 분포 셔플 공격(Random Distribution Shuffle Attack, RDSA)이라는 새로운 적대적 공격 알고리즘을 제안했습니다. 이 알고리즘은 개별 특징 값의 특성보다 네트워크 내의 관측값 간의 상관관계에 중점을 두고 있습니다. RDSA는 적대적 훈련에서 데이터 증강(data augmentation)을 활용하여 분류 성능을 크게 향상시킬 수 있습니다. 여러 과학 분야에서도 유용할 수 있는 이 알고리즘은 고 에너지 물리학을 넘어서는 일반적인 사용 사례를 제공합니다.

- **Technical Details**: RDSA는 입력 매개변수의 일차원 분포는 변하지 않은 상태에서 이들 간의 상관관계를 변경하여 신경망 출력을 극대화하는 적대적 예제를 생성합니다. 이를 위해 각 변수의 일차원 분포를 세밀하게 분할된 히스토그램으로 표현하고, 해당 분포를 기반으로 값을 재샘플링하여 원래의 분포는 유지하면서 변수 간의 상관관계를 줄입니다. 이 과정은 최대 미리 정의된 시도 횟수만큼 반복되며, 새로운 클래스의 예제를 발견할 경우 해당 입력을 적대적 예제로 간주합니다.

- **Performance Highlights**: RDSA는 다양한 데이터셋에 대해 실험을 통하여 신경망을 속일 수 있는 효과적인 방법으로 입증되었습니다. 고 에너지 물리학, 날씨 예측, 손으로 쓴 숫자 인식 등 총 6개 분류 작업에서 경쟁력 있는 성능을 보여주었습니다. RDSA는 최소한의 일차원 분포 변화를 유지하면서도 분류 오류를 유도할 수 있어, 데이터 증강 기법으로 활용할 때는 최신 방법과 비교해도 좋은 결과를 보였습니다.



### Analog Bayesian neural networks are insensitive to the shape of the weight distribution (https://arxiv.org/abs/2501.05564)
Comments:
          Presented at the NeurIPS 2024 Workshop on Machine Learning with New Compute Paradigms, this https URL

- **What's New**: 이번 논문은 Bayesian Neural Network (BNN)을 analog hardware에서 사용하기 위해 mean field variational inference (MFVI)를 통해 실제 기기에서 발생하는 노이즈를 변동 분포로 사용하는 새로운 방법을 제안합니다. 이 방법은 BNN의 예측 분포가 가중치의 평균과 분산이 동일한 경우에 변동 분포의 형태에 관계없이 수렴함을 보여줍니다. 이에 따라 BNN을 MFVI로 구현할 때 기기의 노이즈 분포 형태를 고려할 필요가 없음을 시사합니다.

- **Technical Details**: 본 연구에서는 실제 기기 노이즈를 고려한 변분 추론 방법을 사용하여 BNN을 훈련합니다. 기기의 전도도 노이즈 분포가 균일하지 않더라도, 이를 통해 EFVI(Effective Field Variational Inference)론을 활용하여 근사 오차를 최소화할 수あります. 더 나아가, 중앙 극한 정리에 의해 추론 결과는 깊은 BNN의 예측 불확실성에 큰 영향을 미치지 않음을 보여줍니다.

- **Performance Highlights**: 논문에서 제시한 방법은 다양한 노이즈 분포를 가진 메모리 디바이스를 사용하여 BNN의 예측 정확도를 유지하면서도 컴퓨팅 효율성을 크게 향상시킬 수 있습니다. 특히, 기기의 물리적 특성에 의존하는 노이즈 분포에서도 우수한 성능을 발휘할 수 있음을 보여줍니다. 이러한 결과들은 BNN의 실질적인 하드웨어 구현에 중요한 기여가 될 것입니다.



### Soup to go: mitigating forgetting during continual learning with model averaging (https://arxiv.org/abs/2501.05559)
- **What's New**: 본 논문에서는 Sequential Fine-tuning with Averaging (SFA)라는 새로운 방법을 제안하여 연속 학습(continual learning)에서의 치명적인 망각(catastrophic forgetting)을 줄이는 접근 방식을 소개합니다. 이전 모델의 체크포인트와 현재 학습 중인 모델을 주기적으로 평균하여, 과거 지식을 유지하며 새로운 작업에서의 성능을 개선할 수 있습니다. 이 방법은 저렴한 계산 비용으로 과거 데이터를 저장할 필요 없이도 우수한 결과를 보여줍니다.

- **Technical Details**: SFA는 L2 회귀(L2-regression)에 영감을 받아 설계되었으며, 훈련 중 현재 작업의 모델과 이전 작업의 체크포인트를 통합하여 새로운 작업에 대해 계속 fine-tuning을 진행합니다. 모델 평균 주기(p) 개념을 도입하여, 훈련 중 평균화를 조절함으로써 과거 작업의 성능과 새로운 작업의 성능 간의 균형을 이룹니다. 기존의 데이터 버퍼를 사용하지 않고도 이전 체크포인트를 과거 데이터의 대리로 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, SFA는 이미지 및 언어 작업에서 기존의 데이터 버퍼를 사용하는 방법과 유사한 성능을 달성하면서도 추가 데이터를 저장하지 않고 다른 모델 병합 기법보다 우수한 결과를 보였습니다. 또한, SFA는 기존의 연속 학습 방법들과 비교했을 때 모든 작업에서 일관되게 뛰어난 성능을 보여주었습니다. 이로써 SFA는 효율적인 연속 학습을 위한 신규 솔루션으로 자리 잡을 가능성을 나타냅니다.



### Emergent weight morphologies in deep neural networks (https://arxiv.org/abs/2501.05550)
- **What's New**: 본 연구는 심층 신경망(DNN)이 훈련 과정에서 데이터에 무관하게 emergent weight morphology를 생성한다는 점을 규명했습니다. 이러한 현상은 condensed matter physics(응집 물질 물리학)의 개념을 기반으로 하여 이론적으로 설명되었습니다. 연구 결과는 심층 신경망의 훈련 과정에서 emergence가 발생하며, 이는 DNN의 성능에 영향을 미친다는 것을 보여줍니다.

- **Technical Details**: 저자들은 심층 신경망을 비평형 물질의 한 형태로 보고, 가중치 업데이트의 규칙을 바탕으로 emergent macroscopic structures를 도출하는 이론을 제시합니다. 이를 위해 각 노드 주변의 로컬 가중치 형태를 기술하는 양을 정의하고, 이들 간의 상호작용을 분석하여 대규모 가중치 형태의 emergence를 연구하였습니다. 연구에서는 stochastic gradient descent(확률적 경량 강하법)를 사용하여 가중치의 시간적 진화를 살펴봤습니다.

- **Performance Highlights**: 연구 결과, 다양한 구조의 신경망에서 훈련 단계 동안 가중치가 본질적으로 불안정해지며, 이는 emergent 대규모 가중치 형태를 초래할 수 있음을 보여주었습니다. 특히, 가중치들이 긍정적으로 결합되어 있으며, 이는 이웃한 층의 가중치보다 수배 강하게 나타나는 것으로 확인되었습니다. 이러한 findings는 심층 신경망 훈련에서 emergent behavior가 심층 신경망의 성능에 중대한 영향을 미칠 수 있음을 시사합니다.



### NSChat: A Chatbot System To Rule Them A (https://arxiv.org/abs/2501.05541)
- **What's New**: 이번 논문에서는 NSChat이라는 웹 기반 챗봇 시스템을 소개합니다. 이 시스템은 신경 과학 연구를 지원하도록 정교하게 설계되었으며, 사용자에게 사용자 이름과 실험 코드를 입력해야 하는 기능을 제공합니다. 이렇게 하여 데이터를 정밀하게 교차 참조할 수 있어 연구 데이터의 무결성과 적용 가능성을 향상시킵니다.

- **Technical Details**: NSChat는 React 프레임워크를 이용해 웹 및 모바일 플랫폼에서 효율적으로 작동합니다. 실험 코드 입력 후 사용자는 간단한 채팅 인터페이스로 전환되어 다양한 실험 설정을 조정할 수 있습니다. 이 시스템은 여러 개의 대화형 대형 언어 모델(LLMs)을 쉽게 통합하여 실험 변수에 따라 모델 성능을 비교할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: NSChat 시스템은 사용자 맞춤형 로깅 메커니즘을 지원하여 모든 상호작용을 기록합니다. 각종 모델을 통합하여 실험 환경에서 유연하게 사용할 수 있으며, 사용자 인터페이스를 통해 간단한 맞춤 설정이 가능합니다. 이러한 기능들은 신경 과학 연구 및 정보 검색 연구에서의 포괄적 데이터 수집을 보장합니다.



### Neural Architecture Codesign for Fast Physics Applications (https://arxiv.org/abs/2501.05515)
Comments:
          21 pages, 6 figures

- **What's New**: 본 논문에서는 물리 응용 분야에 맞춰 신경망 아키텍처를 공동 설계하는 파이프라인을 개발했습니다. 이 방법은 두 단계로 나누어져 있으며, 첫 번째 단계에서는 하드웨어 제약을 고려하여 다양한 아키텍처를 탐색하는 글로벌 서치(global search)를 수행하고, 두 번째 단계에서는 가장 유망한 후보를 미세 조정하고 압축하는 로컬 서치(local search)를 진행합니다. 이 과정을 통해 우리는 새로운 작업에 적합한 하드웨어 효율적인 모델을 발견할 수 있었습니다.

- **Technical Details**: 제안된 NAC 프레임워크는 두 단계 최적화 프로세스를 포함합니다: 글로벌 탐색 단계와 로컬 탐색 단계입니다. 글로벌 탐색 단계에서는 미리 정의된 검색 공간 내에서 다양한 아키텍처를 탐색하여 유망한 후보 모델을 식별합니다. 이후 로컬 탐색 단계에서는 이 후보 모델의 하이퍼파라미터를 미세 조정하고 압축하여 성능을 더욱 향상시키고 특정 작업에 최적화합니다.

- **Performance Highlights**: 우리는 두 가지 사례 연구인 재료 과학의 Bragg peak 탐색과 고에너지 물리의 제트 분류를 통해 NAC 방법론의 효과를 입증했습니다. 이 연구들에서 우리는 손수 제작된 아키텍처보다 높은 정확도를 달성하고, 라티언시(latency)를 줄이거나 자원 사용량을 감소시키는 모델을 구현했습니다. 또한, 우리의 NAC 프레임워크는 다양한 과학적 분야에서의 적용 가능성을 높이며, 연구자들이 쉽게 접근할 수 있도록 지원합니다.



### Shrink the longest: improving latent space isotropy with symplicial geometry (https://arxiv.org/abs/2501.05502)
Comments:
          AIST-2024

- **What's New**: 이번 연구에서는 transformer 기반의 모델에서 발생하는 "representation degeneration problem"을 해결하기 위한 새로운 정규화 기법을 제안합니다. 기존의 대부분 방법들이 추가적인 추론 비용을 동반하거나 모델의 재파라미터화에 상당한 양의 데이터가 필요했던 반면, 본 연구의 방법은 simplicial geometry에 기반하여 잠재 표현의 등방성을 개선합니다. 이는 context embedding에서 바코드를 통해 얻은 지속적인 엔트로피를 극대화하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 Vietoris-Rips 필터링을 사용하여 얻은 바코드를 활용하여, 잠재 공간에서 클러스터 간의 거리의 엔트로피를 최대화하는 방식으로 발전되었습니다. 연구진은 유사도 있는 클러스터의 구조를 보존하면서도 전반적인 등방성을 개선할 수 있는 정규화 손실 함수를 설계하였습니다. 이는 등장한 차원과 관련하여 클러스터링 구조를 그대로 유지함으로써 각종 이전 연구에서 제안된 방법들과 차별화된 접근 방식을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 downstream 작업의 성능을 향상시키며, fine-tuning 과정에서 anisotropy를 유의미하게 낮추는 효과를 보였습니다. 특히, 본 기법이 사전 훈련된 언어 모델에 적용 가능함을 시사하며, 이는 다양한 벡터 기반 표현 모델에 적용될 수 있음을 암시합니다. 연구 결과는 현재 분위기에서 중간 결과 또한 강조하며 기하학적 구조를 활용하는 것의 중요성을 확인하였습니다.



### Generalization of Urban Wind Environment Using Fourier Neural Operator Across Different Wind Directions and Cities (https://arxiv.org/abs/2501.05499)
- **What's New**: 이 연구는 도시 바람 환경 시뮬레이션의 새로운 접근법으로 Fourier Neural Operator(FNO) 모델을 채택하여 서로 다른 바람 방향 및 도시 구성을 기반으로 흐름 장을 예측하는 효과를 검증합니다. FNO 모델은 훈련과 예측 속도를 획기적으로 향상시키면서 높은 정확도를 유지할 수 있는 잠재력을 보여주며, 컴퓨터 소요 시간을 99%까지 단축합니다. 혁신적인 접근으로 바람장을 작은 공간 블록으로 나누어 훈련함으로써, FNO 모델은 바람의 주파수 특성을 효과적으로 캡처할 수 있게 됩니다.

- **Technical Details**: FNO 모델은 고전적인 CFD 방법의 계산 복잡성과 시간을 줄여줄 수 있는 유망한 모델로, 고차원 기능 공간에서 해결책을 학습하여 방정식의 솔루션을 제공하는 방식입니다. 이 연구는 FNO 모델을 이용하여 서로 다른 풍향과 도시의 기하학적 구성에서의 2D 바람장을 예측하는 방법을 다룹니다. 또한, 작은 조각 훈련 전략을 통해 GPU 메모리 요구사항을 낮추면서 더 나은 성능을 확보할 수 있음을 증명하였습니다.

- **Performance Highlights**: FNO 모델은 기존의 CFD 소프트웨어와 비슷한 수준의 예측 정확도를 달성하면서도 훨씬 빠른 예측 속도를 제공합니다. 특히, 훈련된 FNO 모델은 특정 기하학적 구성의 도시 환경으로 훈련된 후, 유사한 구조를 가진 다른 도시 배치에서 높은 정확도를 자랑합니다. 이러한 접근은 다양한 도시 환경에 걸쳐 예측 성능을 일반화하는 데 큰 장점이 됩니다.



### Generative Flow Networks: Theory and Applications to Structure Learning (https://arxiv.org/abs/2501.05498)
- **What's New**: 본 논문은 데이터 생성에 대한 가정 없이도 여러 인과 모델이 관측값을 동일하게 설명할 수 있음을 보여주고, 현실과 일치하지 않는 모델 선택으로 인한 위험한 결정을 피하기 위해 가능한 후보들에 대한 인식적 불확실성(epistemic uncertainty)을 유지하는 것이 필수적이라는 점을 강조합니다.

- **Technical Details**: 이 연구는 인과 모델의 구조 학습(structure learning) 문제를 Bayesian 관점에서 다루며, 데이터에 따라 지향 비순환 그래프(directed acyclic graph, DAG)의 구조에 대한 후방 분포(posterior distribution)를 근사화합니다. 논문에서는 GFlowNets라는 새로운 확률 모델 범주를 소개하며, 이는 그래프와 같은 이산(discrete) 및 구성(compositional) 객체에 대한 분포를 모델링하기 위해 설계되었습니다.

- **Performance Highlights**: GFlowNets는 생성(generation)을 순차적 결정 문제로 처리하며, 정상화 상수(normalization constant)로 정의된 목표 분포의 샘플을 단계적으로 구성합니다. 논문의 두 번째 부분에서는 GFlowNets가 관찰 및 실험 데이터를 기반으로 인과 Bayesian Networks의 DAG 구조와 인과 메커니즘의 매개변수(parameters)에 대한 후방 분포를 근사화하는 방법을 보여줍니다.



### FedSA: A Unified Representation Learning via Semantic Anchors for Prototype-based Federated Learning (https://arxiv.org/abs/2501.05496)
Comments:
          Accepted by AAAI2025

- **What's New**: 이 논문은 Prototype 기반의 Federated Learning(FL)에서 클라이언트 간의 데이터 이질성 문제를 해결하기 위한 새로운 프레임워크인 Federated Learning via Semantic Anchors (FedSA)를 제안합니다. 기존 방법들은 로컬 모델에서 직접 프로토타입을 수집하여 비일관성을 초래하는 문제가 있었으나, FedSA는 간단하면서도 효과적인 Semantic Anchors를 도입하여 이 문제를 해결하고자 합니다. 이를 통해 로컬 대표 학습과 프로토타입 생성을 분리하여 클라이언트들이 일관된 표현 학습을 할 수 있도록 유도합니다.

- **Technical Details**: FedSA는 세가지 주요 방법론을 바탕으로 구성됩니다: 1) Anchor-based Regularization with Margin-enhanced Contrastive Learning(RMCL)으로, 편향된 기능 추출기를 교정하여 클래스 내 응집성과 클래스 간 분리를 보장하는 일관된 프로토타입 학습을 지원합니다. 2) Anchor-based Classifier Calibration(CC)으로, 편향된 분류기를 교정하여 클래스 간의 일관된 결정 경계를 학습하도록 합니다. 3) Exponential Moving Average(EMA) 업데이트를 통해 강화된 프로토타입을 사용하여 Semantic Anchors를 업데이트하며, 이를 통해 클라이언트들이 통합된 데이터 표현을 협력적으로 학습하도도록 합니다.

- **Performance Highlights**: 실험 결과, FedSA는 통계적 및 모델 이질성 설정에서 기존의 Prototype 기반 FL 방법들에 비해 비약적으로 성능이 향상되었습니다. 다양한 분류 작업에서 FedSA는 클래스 간 분리가 잘 이루어지며, 일관된 결정 경계를 유지합니다. 이러한 성과는 FedSA가 클라이언트 모델 간의 비일관성 문제를 효과적으로 해결하여, 더욱 강력한 일반화를 달성했음을 보여줍니다.



### LSEBMCL: A Latent Space Energy-Based Model for Continual Learning (https://arxiv.org/abs/2501.05495)
Comments:
          In the 7th International Conference on Artificial Intelligence in Information and Communication (ICAIIC 2025)

- **What's New**: 본 연구에서 제안하는 LSEBMCL(Latent Space Energy-Based Model for Continual Learning) 방법은 에너지 기반 모델(Energy-based Model, EBM)을 활용하여 지속 학습에서의 치명적인 망각(catatstrophic forgetting)을 방지하는 새로운 접근법을 제시합니다. 이 방법은 새로운 과제를 학습할 때 이전 작업의 데이터 포인트를 샘플링하여 기존의 지식을 보존하는 방식으로 작동합니다. LSEBMCL은 자연어 처리(NLP) 과제에 대해 최신 성능을 달성하며, 기존의 방법들과 차별화되는 특징을 보입니다.

- **Technical Details**: LSEBMCL 모델은 사전 훈련된 Mistral 7B를 기반으로 하며, 네 가지 주요 구성 요소로 구성됩니다: 추론 네트워크, Operator 1, Operator 2, 에너지 함수입니다. 이 네트워크는 주어진 질문에 대해 답변을 제공하며, 다양한 NLP 과제를 처리할 수 있도록 설계되었습니다. 각 구성 요소는 훈련 중 에너지 기능과 분포를 활용하여 데이터를 효율적으로 처리하고 학습할 수 있도록 구성됩니다.

- **Performance Highlights**: 제안된 LSEBMCL 방법은 다양한 NLP 작업에서 우수한 성능을 달성했으며, 현재까지의 실험에서 최첨단 결과를 보여주었습니다. 에너지 기반 모델을 통합하여 이전 데이터 작업에 대한 샘플을 생성하는 방식은 지속 학습에서 기존의 지식을 효과적으로 유지할 수 있도록 합니다. 이러한 접근 방식은 특히 자연어 처리 분야에서의 적용 가능성을 높이며, 향후 다양한 실용적 응용을 위한 기초가 될 수 있습니다.



### Mathematical Modeling and Machine Learning for Predicting Shade-Seeking Behavior in Cows Under Heat Stress (https://arxiv.org/abs/2501.05494)
Comments:
          22 pages, 10 figures

- **What's New**: 이 논문에서는 기후 스트레스에 노출된 소의 그늘 찾기 행동을 예측하기 위해 수학적 모델과 머신러닝 기법을 결합한 새로운 접근 방식을 제시합니다. 이를 위해 스페인 발렌시아 주 티타과스 농장에서 수집된 데이터 분석을 통해 시간 평균 온도-습도 지수(Temperature-Humidity Index, THI)와 누적 열 스트레스 지표를 활용한 수학적 분석이 수행되었습니다. Random Forest와 Neural Networks 두 가지 예측 모델의 정확성, 견고성 및 해석 가능성을 비교하고 있습니다.

- **Technical Details**: 본 연구에서는 Random Forest와 Neural Networks라는 세 가지 잘 알려진 감독 학습 알고리즘을 사용하여 특정 환경 조건에서 소의 그늘 찾기 행동을 예측합니다. Random Forest는 여러 개의 결정 트리를 결합하여 정확성을 높이며, 해석 가능성 면에서 전반적인 균형을 유지하는 것으로 나타났습니다. 이 연구는 또한 실제 조건을 고려하여 모델의 견고성을 확보하기 위한 5-겹 교차 검증을 수행합니다.

- **Performance Highlights**: Random Forest 모델은 RMSE(Root Mean Square Error) 14.97을 기록하며 높은 정확도를 달성했습니다. 본 연구의 결과는 머신러닝 기법을 통한 동물 행동 모델링의 발전뿐만 아니라 농장에서의 열 스트레스 완화에 대한 유용한 통찰력을 제공합니다. 이러한 예측 도구를 통해 기후 변화의 영향을 줄이고 축산 관리에서의 실질적인 지식을 향상시킬 수 있습니다.



### Monotonic Learning in the PAC Framework: A New Perspectiv (https://arxiv.org/abs/2501.05493)
Comments:
          16 pages

- **What's New**: 이번 연구는 Probably Approximately Correct (PAC) 학습 이론의 틀에서 모노톤 학습 관점에서 접근한 새로운 관점을 제시합니다. 특히, 두 가지 PAC 학습 가능 문제에 대해 하한 성능 분포를 도출하고 이와 관련된 실험을 수행함으로써, 각 학습 알고리즘의 모노톤 특성을 입증했습니다. 이를 통해 독립적이고 동일하게 분포된 훈련 샘플을 사용할 때 Empirical Risk Minimization (ERM) 기반의 모든 학습 알고리즘이 모노톤임을 보여줍니다.

- **Technical Details**: PAC 학습 이론의 메커니즘을 통해, 샘플 복잡도를 추정하고, 샘플 크기가 증가함에 따라 성능 하한의 모노톤성을 증명했습니다. 또한, 실험을 통해 일반화 손실의 하한 분포가 늘어나면서 이들 분포가 어떻게 수렴하는지를 보여주었습니다. 이 작업은 기존의 비모노톤 학습 곡선에서의 이상 현상과 비슷한 현상을 통해 모노톤성을 설명하고 있습니다.

- **Performance Highlights**: 두 가지 구체적인 머신 러닝 문제에서 실험을 수행한 결과, 이론적으로 추정된 손실 값과 실제 실험값이 모노톤적으로 감소하는 경향을 보여주었습니다. 각 문제에서의 평균 성능 값과 이론적 성능 값 모두가 높은 확률로 감소하는 모양을 나타내며, 이는 PAC 학습 가능 문제의 모노톤성을 확인하는 중요한 결과입니다.



### Machine Learning Force-Field Approach for Itinerant Electron Magnets (https://arxiv.org/abs/2501.06171)
Comments:
          18 pages, 8 figures

- **What's New**: 이번 연구에서는 전자 스핀 상호작용에 의해 구동되는 이동 전자 자성체의 Landau-Lifshitz-Gilbert (LLG) 동역학 시뮬레이션을 위한 머신러닝 (ML) 포스필드 프레임워크의 최근 개발을 검토합니다. 특히, 스핀 배치의 대칭 불변 표현을 구현하기 위한 이론적 접근법과 방법론에 중점을 두고 제안된 새로운 기술을 소개합니다. ML 모델을 바탕으로 한 LLG 시뮬레이션이 다양한 비열평형 스핀 구조를 재현하는 성공 사례를 보입니다.

- **Technical Details**: 이 연구에서는 국소적으로 발생하는 자기장과 스핀 배치의 회전을 고려해야 합니다. 이를 통해 세 가지 대표적인 비열평형 스핀 질서를 정확히 재현하는 ML 기반 LLG 시뮬레이션을 수행했습니다. 구체적으로, s-d 모델을 활용하여 다각형 격자에서의 스핀 배치와 관련된 대칭을 계량화한 자기 디스크립터를 제안합니다.

- **Performance Highlights**: ML 모델로 실시한 대규모 열 소산 시뮬레이션은 스키르미온 및 바이메론으로 구성된 유리상 상태의 동적 특성을 밝혀내는 데 성공했습니다. 이는 복잡한 스핀 질서를 다룰 수 있는 ML 포스필드 접근법의 유용성을 강조합니다. 다양한 비열평형 스핀 구조 및 동적 현상 모델링의 가능성을 제시하는 중요한 연구 결과를 보여줍니다.



### Efficient Transition State Searches by Freezing String Method with Graph Neural Network Potentials (https://arxiv.org/abs/2501.06159)
Comments:
          9 pages, 4 figures, 3 tables

- **What's New**: 이 연구는 유기 화학 반응을 설명하기 위해 최적화된 그래프 신경망(graph neural network) 기반의 잠재 에너지 함수(potential energy function)를 개발하여 전이 상태(transition state) 추정 구조를 신속하게 식별하고자 합니다. 연구진은 이를 통해 추정 구조를 세밀히 개선하고 각 테스트 시스템에서 전이 상태를 정확히 찾을 수 있었습니다. 이 방법은 평균 ab-initio 계산 수를 47% 줄이는 데 기여하여, 현대 기계학습 모델이 일상적인 계산 화학 작업을 가속화할 수 있는 신뢰성에 도달했음을 보여줍니다.

- **Technical Details**: 전이 상태(TS)는 화학 반응의 정확한 특성을 평가하고 열역학 및 운동학적 속성을 예측하는 데 중요한 요소입니다. 그러나 TS는 원자계의 Born-Oppenheimer 잠재 에너지 표면(PES)에서 1차 안장점(first-order saddle points)으로 존재하여 찾기 어렵습니다. 연구에서는 TS 검색 알고리즘과 비선형 경로 탐색(non-local path finding) 알고리즘을 활용하며, 초기 TS 추정 구조를 기반으로 하여 섬세하게 보완하는 방법을 제시합니다.

- **Performance Highlights**: 이 연구에서는 그래프 신경망 기반 잠재 에너지 함수를 활용하여 전이 상태 검색의 정확성과 효율성을 크게 향상시켰습니다. 특히, 테스트된 모든 시스템에서 전이 상태를 정확히 찾고, 평균 ab-initio 계산 수를 47% 감소시킬 수 있었습니다. 이러한 성과는 ML 모델이 기존의 계산적 방법들보다 효율적으로 전이 상태를 탐지하는 데 도움을 줄 수 있음을 확인시켜 줍니다.



### Merging Feed-Forward Sublayers for Compressed Transformers (https://arxiv.org/abs/2501.06126)
- **What's New**: 이번 연구에서는 기존의 pruning(가지치기) 기술 대신에, 모델 내에서 유사한 매개변수 그룹을 병합하는 새로운 압축 방법을 제안합니다. 특히 Transformer 모델의 feed-forward(전달) 서브 레이어를 선택하고 정렬하여 병합하는 방식으로, 다양한 언어 모델링, 이미지 분류 및 기계 번역 작업을 수행했습니다. 이 방법을 통해 원본 모델과 유사한 성능을 유지하면서, 모델의 세 번째 이상을 통합할 수 있음을 보였습니다.

- **Technical Details**: Transformer의 feed-forward 서브 레이어는 전체 매개변수의 약 2/3를 차지하며, 이들의 압축은 상당한 성과를 가져올 수 있습니다. 논문은 이를 위해 permutation-based(순열 기반) 뉴런 정렬 방법을 사용하여 두 개의 레이어를 정렬합니다. 이 과정에서 cross-correlation(상관관계)를 계산하여 최적의 뉴런 정렬을 도출하고, 이를 통해 여러 개의 유사한 서브 레이어를 하나의 매개변수 집합으로 결합하는 기술을 제안합니다.

- **Performance Highlights**: 제안된 방법은 Vision Transformer에서도 21% 이상의 매개변수를 제거하면서도 99%의 원래 성능을 유지할 수 있음을 보여주었습니다. 또한, 다른 Transformer 기반 모델들, 즉 GPT-2, ViT, 기계 번역 모델을 대상으로 실험하여, 이 방법이 기존의 모델에 비해 유의미한 성능 향상을 가져온 것을 확인했습니다. 연구 결과는 다양한 전이 학습 모델에서도 쉽게 적용될 수 있는 잠재력을 보여줍니다.



### Inferring High-Order Couplings with Neural Networks (https://arxiv.org/abs/2501.06108)
Comments:
          13 Pages and 3 Figures

- **What's New**: 최대 엔트로피 방법들이 통계역학의 역 이징/폿츠 문제에 기반하여 개발되었으며, 생물정보학, 생태학, 신경과학 등 다양한 분야에서 쌍(pairwise) 상호작용을 모델링하는 데 필수적으로 사용되고 있습니다. 그러나 이 방법들은 복잡한 시스템에서 결정적인 고차 상호작용을 종종 간과합니다. 본 연구에서는 제약 볼츠만 머신(Restricted Boltzmann Machines, RBMs)을 활용하여 이러한 고차 상호작용을 효율적으로 캡처하는 방법론을 제시하며, 제너럴라이즈드 팟츠 모델(generalized Potts models)과의 정확한 매핑을 통해 고차 상호작용을 효과적으로 추출할 수 있음을 보여줍니다.

- **Technical Details**: RBM의 훈련 과정은 데이터셋의 로그 가능도 함수(ℒ)를 최대화하는 것으로 시작됩니다. 이는 (stochastic) gradient descent를 이용하여 이뤄지며, 블록 기브스 샘플링(Block-Gibbs Sampling) 방법을 사용해 파티션 함수의 계산을 근사합니다. 본 연구에서는 지속적 대조 발산(persistent contrastive divergence, PCD-k) 기법을 활용하여 효율적인 매핑을 통해 조합 상호작용을 추출하며, 이 과정에서 병렬화(parallelization)를 통해 계산 속도를 높였습니다.

- **Performance Highlights**: 제안한 방법은 합성 데이터셋으로부터 2신경 및 3신경 상호작용을 정확하게 회복하는데 성공하였으며, 단백질(sequence) 데이터에 적용한 결과, 현대적 대체 방식인 역 팟츠 모델(inverse Potts models)과 동일하게 효과적인 단백질 접촉 맵(protein contact maps)을 재구성하는 성과를 달성하였습니다. 이러한 결과는 복잡한 시스템에서 고차 상호작용을 탐구하는 데 있어 RBM이 강력하고 효율적인 도구가 될 수 있음을 입증합니다.



### Finite-Horizon Single-Pull Restless Bandits: An Efficient Index Policy For Scarce Resource Allocation (https://arxiv.org/abs/2501.06103)
Comments:
          17 Pages, 8 figures. Accepted by AAMAS 2025

- **What's New**: 본 논문에서는 각 팔이 최대 한 번만 당겨질 수 있는 제한을 가진 새로운 문제인 유한 수평 단일 당김 휴식 다중 도구(RMAB) 문제를 제안합니다. 기존 RMAB 알고리즘은 이러한 제약 조건을 고려하지 않으므로 비효율적이며, 이 문제를 해결하기 위해 더미 상태(dummy states)를 도입하여 시스템을 복제합니다. 이로 인해 팔이 활성화되면 해당 더미 상태 내에서만 전환되도록 보장합니다.

- **Technical Details**: 유한 수평 단일 당김 RMAB(SPRMABs)를 통해 각 팔은 최대 한 번만 활용될 수 있도록 설계되었습니다. 이는 자원이 극도로 제한된 환경에서 유용하며, 일반 RMAB 문제의 복잡성에서 발생하는 최적 제어 전략을 찾기 힘든 문제를 더욱 복잡하게 만듭니다. 본 연구에서는 기존의 지수 기반 정책들을 SPRMABs에 맞게 조정하는 데 있어 발생하는 도전 과제를 다루고, 이를 위해 더미 상태를 사용하는 방법론을 제시합니다.

- **Performance Highlights**: 제안된 지수 정책은 유한한 팔 수에 대해 평균 최적성 간극이 점진적으로 감소하는 서브 선형적인 성능을 달성하는 것을 처음으로 입증하였으며, 이는 다양한 도메인에서 기존 전략들과 비교하여 견고한 성능을 보였습니다. 본 연구는 SPRMABs의 공정한 자원 할당의 적용 가능성을 향상시킬 뿐만 아니라 제약이 있는 밴딧 설정에서의 향후 연구를 위한 강력한 기반을 마련합니다.



### Towards Developing Socially Compliant Automated Vehicles: State of the Art, Experts Expectations, and A Conceptual Framework (https://arxiv.org/abs/2501.06089)
Comments:
          39 pages, 13 figures, under review by the journal of Transportation Research Part E: Logistics and Transportation Review

- **What's New**: 본 연구는 Socially Compliant Automated Vehicles (SCAVs) 개발 현황에 대한 첫 번째 포괄적 스코핑 리뷰를 수행하였습니다. 이는 혼합 교통 환경에서 자율주행차(AV)와 인간 운전 차량(HDV)의 공존과 안전성을 높이는 데 필수적입니다. SCAVs의 사회적 수용성을 높이기 위한 현재의 교수법과 연구의 공백을 식별하였습니다.

- **Technical Details**: 스코핑 리뷰는 SCAVs의 주요 개념, 방법론적 접근 방식 및 연구 공백을 확인하는 데 중점을 두었습니다. 이에 더하여, 전문가 인터뷰를 통해 SCAVs에 대한 연구의 중요한 공백과 기대 사항을 짚어보았습니다. 이 연구는 SCAVs 개발을 위한 개념적 프레임워크를 제안하며, 이는 연구자, 기술자, 정책 입안자 등 다양한 전문가로부터의 피드백을 받았습니다.

- **Performance Highlights**: 온라인 설문조사를 통해 제안한 개념적 프레임워크의 유효성이 입증되었고, AV와 HDV의 통합 문제를 해결하는 데 중요한 통찰을 제공하였습니다. 이 연구는 SCAV의 연구 및 개발 의제에 기여하는 미래 연구 방향 및 제안 사항도 논의하고 있습니다.



### All AI Models are Wrong, but Some are Optima (https://arxiv.org/abs/2501.06086)
- **What's New**: 본 논문은 전통적인 Predictive AI 모델의 한계를 넘어서, 의사결정 성과를 극대화하는 방향으로 모델링할 수 있는 'decision-oriented' predictive AI 모델에 대한 개념을 도입합니다. 기존의 모델들은 데이터에 최적화되어 있지만, 실제 의사결정 성과와의 직접적인 관계가 부족했습니다. 저자들은 이러한 모델의 최적 조건을 수립하고, 확률적 및 결정적 시스템에서의 의사결정 성과를 개선할 수 있는 방법을 제안합니다.

- **Technical Details**: 우리는 Sequential Decision-Making(SDM)의 문맥에서 Markov Decision Process(MDP)를 활용하여 특정 성과 목표를 달성하는 데 필요한 예측 모델의 필요충분조건을 수립했습니다. MDP는 상태와 행동의 쌍을 통해 최적의 정책을 찾는 구조입니다. 또한, 기존의 Predictive AI 모델이 적합한 데이터에 맞춰져 있는 경우라도 최적의 성과를 보장하지 않는다는 점을 명확히 하였습니다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 제안된 의사결정 중심의 예측 모델들이 성과를 개선하는 데 효과적임을 보여주었습니다. 특히, 의사결정 목표를 모델에 통합함으로써 예측 성능을 일부 희생하면서도 최적의 결정 성과를 달성할 수 있다는 사실이 입증되었습니다. 결론적으로, 이 연구는 추후 AI 기반의 의사결정 시스템 개발에 중요한 기초 자료가 될 것입니다.



### Averaged Adam accelerates stochastic optimization in the training of deep neural network approximations for partial differential equation and optimal control problems (https://arxiv.org/abs/2501.06081)
Comments:
          25 pages, 10 figures

- **What's New**: 이 논문에서는 깊은 신경망(deep neural networks, DNNs)을 훈련시키기 위해 평균화된 아담 최적화기(averaged Adam optimizer)를 적용하여 과학적 컴퓨팅 문제를 해결하는 새로운 접근법을 제안합니다. 특히, 고전적인 폴리악-루페르트 평균화 방법에 의해 영감을 받아 이 방법이 개발되었습니다. 논문은 다양한 학습 문제에 대해 이 방법을 테스트하고 있으며, 이전의 표준 아담 최적화기와 표준 SGD 최적화기보다 성능이 우수함을 보여줍니다.

- **Technical Details**: 본 연구에서는 아담 최적화기의 평균화된 변형을 사용하여 물리학 기반 신경망(physics-informed neural network, PINN), 심층 역확률 미분 방정식(deep backward stochastic differential equation), 그리고 열 방정식 및 버거스 방정식과 같은 부분 미분 방정식(partial differential equations, PDE) 문제를 다룹니다. 또한, 최적 제어(optimal control, OC) 문제와 이미지 분류 문제(예: CIFAR-10을 위한 ResNet)에 대한 DNN 접근법도 포함됩니다. 세부적으로, 이 연구는 평균화된 아담 최적화기가 다양한 과학적 기계 학습 문제에 매우 효과적임을 입증합니다.

- **Performance Highlights**: 실험 결과, 평균화된 아담 최적화기는 표준 아담 최적화기 및 표준 SGD 최적화기보다 특히 과학적 기계 학습 문제에 있어 더 나은 성능을 보였습니다. 여러 예제에서 성능을 비교하였고, 평균화된 아담 최적화기가 각 문제에서 우수한 결과를 기록했습니다. 이 논문에 대한 수치적 실험을 위한 Python 소스 코드는 GitHub에서 확인할 수 있습니다.



### Learning Flexible Heterogeneous Coordination with Capability-Aware Shared Hypernetworks (https://arxiv.org/abs/2501.06058)
Comments:
          11 pages, 6 figures, equal authorship between Pierce Howell and Shalin Jain

- **What's New**: 이번 연구에서는 이질적인 다중 에이전트 조정을 위한 Capability-Aware Shared Hypernetworks (CASH)라는 새로운 아키텍처를 제안하여, 에이전트의 개별 및 집단 능력을 기반으로 한 동적인 역할을 효율적으로 학습할 수 있도록 한다. CASH는 샘플 효율성을 유지하면서도 충분한 행동 다양성을 만들어낼 수 있는 소프트 파라미터 공유(hypernetworks)를 통해 작동한다. 이는 어떤 새로운 팀이나 에이전트에도 제로샷 일반화(zero-shot generalization)가 가능하다는 점에서, 기존 방법들과의 차별성을 보인다.

- **Technical Details**: CASH 아키텍처는 RNN 기반 encoder, hyper adapter, adaptive decoder로 구성되어 있으며, 각 에이전트의 능력에 관계없이 공유되는 에이전트 불변 조정 전략을 학습한다. Hyper adapter는 현재의 관찰 값과 에이전트의 능력을 바탕으로 adaptive decoder의 가중치를 매핑하는 방식으로 동작하여, 현재의 맥락에 따라 유연하게 적응할 수 있다. 이 아키텍처는 임itation learning, 가치 기반 Reinforcement Learning(RL), 정책 기반 RL 등 다양한 학습 패러다임을 통해 훈련할 수 있다.

- **Performance Highlights**: CASH는 두 가지 이질적인 조정 작업(소방 및 다중 물자 운반)과 세 가지 학습 패러다임에서 평가되었으며, 그 결과 기존 아키텍처보다 성공률과 샘플 효율성이 뛰어난 성과를 보여주었다. 특히, CASH는 학습 가능한 파라미터의 양이 기존 기준보다 20%에서 40%에 불과하면서도 새로운 팀과 에이전트에 대한 일반화에서 두드러진 성과를 내었다. 본 연구는 에이전트의 능력을 인식하는 것이 이질적인 다중 에이전트 학습의 강력한 구조적 설계 선택이 될 수 있음을 시사한다.



### AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery (https://arxiv.org/abs/2501.06039)
Comments:
          23 pages, 5 figures

- **What's New**: 이 논문에서는 다양한 세포, 분자 및 조직 수준에서 작동하는 생물학적 조직을 위한 기반 모델 프레임워크인 Virtual Tissues (VirTues)를 제안합니다. VirTues는 새로운 토크나이제이션 방식과 고차원 다중 데이터에 대한 주의 메커니즘을 도입하여 해석 가능성과 성능을 동시에 향상시키고 있습니다. 이 모델은 고전적인 방법을 넘어 생물조직의 공간적 및 분자적 특성 분석을 위한 혁신적인 접근 방식을 제시합니다. 다양한 암 및 비암 조직 데이터세트에서 훈련된 VirTues는 특정 작업에 대한 추가 조정 없이 강력한 일반화 능력을 보이며, 새로운 생물학적 맥락에서 분석을 가능하게 합니다.

- **Technical Details**: VirTues의 핵심 혁신은 변형 체계의 주의 메커니즘을 공간적 및 마커 주의 컴포넌트로 분리하여 다양한 마커 조합에 대해 유연한 처리 능력을 제공함으로써 고차원 데이터를 효과적으로 처리하는 것입니다. 이 모델은 단백질 언어 모델(Protein Language Model, PLM)을 활용하여 단백질 마커 간의 복잡한 관계를 포착하고, 세포, 틈새 및 조직 수준에서의 생물학적 계층을 존중합니다. 또한, VirTues는 마스킹된(marker-space) 마커 데이터를 복원하는 비지도 훈련을 통해 작동하는 마스크 자동 인코더(Masked Autoencoder, MAE) 구조를 채택하였습니다.

- **Performance Highlights**: VirTues는 임상 진단, 생물학적 발견 및 환자 사례 검색 작업에서 기존 방법들을 초월하는 성능을 보이며, 다양한 데이터에 대한 robust한 일반화 능력을 입증되었습니다. 이 모델은 아네(AST)와 같은 다양한 실험적 조건에서도 이질적인 데이터세트를 통합할 수 있는 능력을 갖추고 있어, 임상 응용 프로그램 및 질병 메커니즘에 대한 통찰을 제공합니다. VirTues의 향상된 성능과 해석 가능성은 생물학적 데이터 분석의 새로운 가능성을 열어줄 것으로 기대됩니다.



### An Attention-Guided Deep Learning Approach for Classifying 39 Skin Lesion Types (https://arxiv.org/abs/2501.05991)
Comments:
          26 pages

- **What's New**: 이 연구는 39종의 피부 병변을 포함하는 포괄적이고 다양한 데이터셋을 정리하여 딥러닝 기반의 분류 시스템을 개발하는 데 중점을 두고 있습니다. 기존 연구들은 보통 제한된 병변 유형에 집중해 왔지만, 이 연구는 Advanced attention-guided techniques를 도입하여 더 많은 종류의 피부 병변 분류의 정확성을 높이고자 합니다.

- **Technical Details**: 연구에서는 MobileNetV2, Xception, InceptionV3, EfficientNetB1, Vision Transformer와 같은 최첨단 딥러닝 모델을 사용하여 성능을 평가합니다. 효율적인 정확성을 높이기 위해 Efficient Channel Attention (ECA) 및 Convolutional Block Attention Module (CBAM)과 같은 어텐션 메커니즘이 모델 아키텍처에 통합됩니다.

- **Performance Highlights**: Vision Transformer 모델은 CBAM과의 통합을 통해 93.46%의 정확도, 94%의 정밀도, 93%의 재현율 및 93%의 F1-score를 기록하여 다른 모델들보다 우수한 성능을 보여줍니다. 이러한 결과는 피부 병변 진단을 위한 정확하고 효율적인 예후 도구로서 제안된 시스템의 중요한 잠재력을 강조합니다.



### Towards Early Prediction of Self-Supervised Speech Model Performanc (https://arxiv.org/abs/2501.05966)
- **What's New**: 이번 연구에서는 Self-Supervised Learning (SSL) 음성 모델의 사전 훈련 품질을 평가하기 위한 효율적인 비지도 학습 방법을 제안합니다. 이러한 새로운 방법은 클러스터 품질(cluster quality)과 임베딩의 순위(rank)를 측정하여, SSL 모델의 성능을 예측하는 데 도움을 줍니다. 실험 결과, 이 방법들이 오직 한 시간의 레이블 없는 오디오로도 사전 훈련의 손실보다 최종 성능과 더 높은 상관관계를 나타낸다는 것을 보였습니다.

- **Technical Details**: BEST-RQ(BERT-based Speech pre-Training with Random-projection Quantizer)은 효율성, 성능, 및 SpeechBrain 라이브러리의 오픈 소스 구현 덕분에 사용됩니다. 모델의 사전 훈련 손실은 예측된 타겟과 불연속 타겟 간의 교차 엔트로피를 통해 계산됩니다. 논문에서는 사전 훈련 중 마스킹 하이퍼파라미터가 손실에 미치는 영향을 이론적으로 설명하고 있습니다.

- **Performance Highlights**: ASR(Automatic Speech Recognition) 작업에서 클러스터링 측정과 SSL 임베딩의 순위가 사전 훈련 손실보다 최종 성능과 더 강한 상관관계를 가진다고 밝혔습니다. 이 연구는 향후 실험에서 수천 시간의 GPU 시간을 절약할 가능성을 보여줍니다. 전략적으로 사전 훈련 과정에서 모델 성능을 조기에 평가할 수 있는 기회를 제공합니다.



### Q-MAML: Quantum Model-Agnostic Meta-Learning for Variational Quantum Algorithms (https://arxiv.org/abs/2501.05906)
Comments:
          8 pages, 8 figures, to be published in AAAI 25

- **What's New**: 이번 연구에서는 파라미터화된 양자 회로(Parameterized Quantum Circuits, PQCs)의 최적화를 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 Model-Agnostic Meta-Learning (MAML) 기술에서 영감을 받아 클래식 최적화기(classical optimizer)를 사용합니다. 새로운 메타 목적에 기반하여 Learner라는 클래식 신경망을 훈련시키며, 이를 통해 초기 파라미터의 효율적인 선택을 도와줍니다.

- **Technical Details**: 프레임워크는 Learner와 PQC 간의 상호작용을 통해 Learner의 출력값을 초기 파라미터로 사용합니다. 프리트레이닝(Pre-training) 단계에서 Learner는 양자 회로 비용 함수(cost function)를 기반으로 한 메타 목적을 가지고 훈련됩니다. 보정(adaptation) 단계에서는 PQC 업데이트가 몇 번만으로도 더 정확한 값에 수렴할 수 있도록 하여, Learner는 변경되지 않고 유지됩니다.

- **Performance Highlights**: 본 연구에서는 분포 함수 매핑 및 Heisenberg XYZ 해밀토니안 최적화를 포함한 실험을 통해 제안한 방법을 검증합니다. 실험 결과는 Learner가 문제 공간 전반에 걸쳐 일반화되는 초기 파라미터를 성공적으로 추정하여 빠른 적응(fast adaptation)을 가능하게 한다는 것을 나타냅니다.



### Discovery of sustainable energy materials via the machine-learned material spac (https://arxiv.org/abs/2501.05903)
- **What's New**: 이 연구에서는 OptiMate 모델이 반도체와 절연체의 광학적 특성을 예측하기 위해 훈련받은 그래프 주의 네트워크(Graph Attention Network)라는 점에서 새로운 접근을 제시합니다. 모델이 고차원 공학적 공간을 해석 가능한 저차원 공간으로 표현할 수 있는 능력을 보여줍니다. 이로써 10,000개 이상의 물질에 대한 클러스터링이 가능해지며, 친환경 대체 물질을 찾는 데 기여할 수 있습니다.

- **Technical Details**: OptiMate 모델은 원자 임베딩 다층 퍼셉트론(Multilayer Perceptron)과 메시지 전송 블록(Message Passing Blocks)으로 구성되어 있습니다. 여기서 UMAP(통합 다양체 근사 투영) 기법을 사용하여 고차원 임베딩을 저차원으로 변환하고, 다양한 물질의 내부 표현을 시각화합니다. UMAP은 고차원 공간에서의 지역적 거리와 가능하면 전역적 거리를 보존하는 비선형 차원 축소 기술입니다.

- **Performance Highlights**: 모델은 10,000개 이상의 물질에 대해 유사한 내부 표현을 학습하며, 이는 경험 많은 재료 과학자가 만든 것과 유사합니다. 연구의 결과로, 이 모델을 통해 에너지 관련 기술에 적합한 보다 지속 가능한 대체 재료를 탐색할 수 있는 가능성이 열렸습니다. 최종적으로, OptiMate 모델은 다양한 응용 분야에서 재료 발견과 설계를 위한 학습된 재료 공간 활용의 큰 잠재력을 입증하고 있습니다.



### Text2Playlist: Generating Personalized Playlists from Text on Deezer (https://arxiv.org/abs/2501.05894)
- **What's New**: 이번 논문에서는 Deezer의 새로운 도구인 Text2Playlist를 소개합니다. Text2Playlist는 사용자의 요구에 맞춘 자동화된 개인화 플레이리스트 생성 도구로, 일반 텍스트 쿼리를 기반으로 작동합니다. 이 시스템은 현재 Deezer의 모바일 및 웹 애플리케이션에 배포되어 있으며, 적용 범위를 확대해 나가고 있습니다.

- **Technical Details**: Text2Playlist는 최신 Large-Language Models (LLMs)와 Retrieval-Augmentation Generation (RAG) 프레임워크를 활용하여 설계되었습니다. 사용자의 쿼리에서 명시적 및 암시적 태그를 추출하고, 이를 기반으로 Deezer의 음악 카탈로그에서 관련 콘텐츠를 검색하여 맞춤형 플레이리스트를 생성합니다. 이 시스템은 Python으로 작성되었으며 Kubernetes 클러스터에서 실행됩니다.

- **Performance Highlights**: Text2Playlist는 2024년 7월부터 5%의 프리미엄 사용자에게 배포되었으며, 10월에는 이 비율이 20%로 확대되었습니다. 사용자 만족도를 평가하기 위해 생성된 플레이리스트의 재청취 비율을 측정한 결과, Text2Playlist로 생성된 플레이리스트는 45%의 재청취율을 기록했습니다. 이는 수동으로 생성된 플레이리스트의 27%에 비해 상당히 높은 수치로, 사용자 참여도가 증가했음을 보여줍니다.



### EDNet: Edge-Optimized Small Target Detection in UAV Imagery -- Faster Context Attention, Better Feature Fusion, and Hardware Acceleration (https://arxiv.org/abs/2501.05885)
Comments:
          Accepted in 21st IEEE International Conference on Ubiquitous Intelligence and Computing (UIC 2024) this https URL

- **What's New**: 이번 연구에서는 드론 이미지에서 작은 표적을 탐지하기 위한 새로운 프레임워크인 EDNet을 제안합니다. EDNet은 향상된 YOLOv10 아키텍처를 기반으로 하며, 실시간 응용 프로그램에 최적화되어 있어 후처리 과정이 필요 없습니다. XSmall 탐지 헤드와 Cross Concat 전략을 통합하여 다양한 환경에서 작은 표적을 더욱 효과적으로 탐지할 수 있는 멀티 스케일 컨텍스트 인식 기능이 개선되었습니다.

- **Technical Details**: EDNet은 TensorFlow와 같은 딥러닝 플랫폼에서 구현된 다양한 구성 요소로 이루어져 있습니다. ConvBNSiLU 블록과 Spatial-Channel Decoupled Downsampling(SCDown) 블록을 포함하고 있으며, 이들은 계산 효율성을 높이고 중요한 정보를 보존하는 데 기여합니다. 또한 Faster Context Attention(FCA)과 같은 맞춤형 블록을 사용하여 파라미터 수를 줄이면서도 성능을 향상시키고, WIoU 손실 함수를 통해 바운딩 박스 회귀를 개선합니다.

- **Performance Highlights**: EDNet은 Tiny에서 XL까지 7가지 변형으로 제공되며, 기존의 객체 탐지기보다 높은 정확도와 뛰어난 계산 효율성을 자랑합니다. iPhone 12와 같은 모바일 장치에서 EDNet 변형들은 16FPS에서 55FPS의 속도로 작동할 수 있어, 데이터 프라이버시를 보장하면서도 실시간으로 객체 탐지를 수행할 수 있는 확장 가능하고 효율적인 솔루션을 제공합니다. 특히, EDNet은 mAP@50에서 최대 5.6% 증가를 달성하며, 효율적인 모델 디자인을 통해 다양한 환경에 적합한 성능을 보여줍니다.



### VideoRAG: Retrieval-Augmented Generation over Video Corpus (https://arxiv.org/abs/2501.05874)
- **What's New**: 본 논문은 VideoRAG라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 사용자 쿼리에 따라 관련 비디오를 동적으로 검색하고 이 정보의 시각적 및 텍스트 정보를 출력 생성 과정에 통합합니다. 기존의 RAG 접근법이 주로 텍스트의 검색과 처리에 집중했던 반면, VideoRAG는 비디오를 활용하여 멀티모달 지식을 효과적으로 증대할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VideoRAG는 Retrieval-Augmented Generation (RAG)과 Large Video Language Models (LVLMs)의 개념을 결합하여 동작합니다. 사용자가 입력한 쿼리와 관련된 비디오 콘텐츠를 찾아내고, 이 비디오의 시각적 및 텍스트 요소를 응답 생성 과정에 통합합니다. 특히 텍스트 설명이 없는 경우에도 비디오에서 직접 추출한 내용을 기반으로 자동 음성 인식 기법을 활용하여 텍스트 전사본을 생성하여 완전한 멀티모달 지원을 제공합니다.

- **Performance Highlights**: VideoRAG의 성능은 WikiHowQA와 HowTo100M 데이터셋을 통해 실험적으로 검증되었습니다. 실험 결과, 비디오 데이터를 활용한 VideoRAG가 기존의 관련 RAG 기법들에 비해 상당한 성능 향상을 보임을 확인했습니다. 이 결과는 비디오가 RAG 시스템의 지식 증대에 이바지할 수 있는 강력한 자원임을 입증합니다.



### Collaborative Content Moderation in the Fedivers (https://arxiv.org/abs/2501.05871)
- **What's New**: Fediverse(페디버스)의 급성장과 함께 콘텐츠 조절(content moderation) 문제에 대한 논의가 중요해지고 있습니다. 이 연구는 탈중앙화된 환경에서의 자원 부족 문제를 인식하고, 전통적인 중앙집중형 플랫폼에서 사용되는 자동화 도구의 한계를 해결하기 위한 새로운 시스템인 FedMod를 제안합니다. FedMod는 서버 간에 협력적으로 학습을 통해, 부분적으로 훈련된 콘텐츠 조절 모델을 교환하는 시스템입니다.

- **Technical Details**: FedMod는 federated learning(연합 학습) 기반으로 설계되었으며, 서버들은 유사한 서버와의 매개변수를 교환합니다. 이는 각 서버가 작성한 데이터 레이블을 공개하지 않고도 지식을 공유하도록 하여, 중앙 집중식 모델의 한계를 극복하는 방향으로 진행됩니다. 이 방법은 개인 서버의 한정된 자원으로부터 얻어진 지식을 효과적으로 활용할 수 있습니다.

- **Performance Highlights**: FedMod는 유해 콘텐츠 탐지, 봇 콘텐츠 탐지 및 콘텐츠 경고 할당의 세 가지 작업에서 강력한 성과를 보였습니다. 이 시스템은 각각의 작업에서 자동화된 콘텐츠 조절 능력을 향상시켰으며, 평균 macro-F1 점수는 각각 0.71, 0.73, 0.58에 달합니다. 이는 각 인스턴스의 로컬 모델과 비교했을 때 영역별로 각각 12.69%, 5.79%, 9.43%의 성능 향상을 이끌어냈습니다.



### Neural Network Verification is a Programming Language Challeng (https://arxiv.org/abs/2501.05867)
Comments:
          Accepted at ESOP 2025, European Symposium on Programming Languages

- **What's New**: 이 논문에서는 신경망 검증(neural network verification)의 문제를 프로그래밍 언어의 관점에서 접근할 수 있는 가능성을 제시하고 있습니다. 기존의 연구는 검증 알고리즘과 도구 개발에 집중되어 있었지만, 프로그래밍 언어의 통찰력이 혁신적인 발전을 가져올 수 있다는 점이 강조되고 있습니다. 특히, 신경망의 강건성(robustness) 문제와 기존 프로그래밍 패러다임 간의 분리 문제를 해결해야 함을 언급하고 있습니다.

- **Technical Details**: 신경망 검증의 주요 도전 과제는 세 가지 유형의 속성을 정의하는 것입니다: 기하학적 속성(geometric properties), 하이퍼 속성(hyper-properties), 도메인 특정 속성(domain-specific properties). 기하학적 속성은 데이터의 기하학 구조에 기반하지만, 하이퍼 속성은 모든 입력에 대한 보장을 요구하며, 도메인 특정 속성은 신경망이 훈련된 데이터의 의미적 맥락에 의존합니다. 이들 각각은 신경망의 검증 과정에 서로 다른 도전 과제를 제시합니다.

- **Performance Highlights**: 최근의 VNN-COMP 대회에서는 신경망 검증의 표준 벤치마크가 매년 업데이트되고 있으며, 이를 통해 업계에서의 검증 기법과 그 성능이 지속적으로 발전하고 있습니다. 하지만 신경망이 실제 시스템에 통합되어 사용되는 경우가 많아 다양한 프로그래밍 언어 지원이 필요하다는 점이 지적됩니다. 이러한 맥락에서, 프로그래밍 언어 커뮤니티는 신경망 검증 기술의 발전을 가속화할 수 있는 중요한 역할을 할 수 있습니다.



### MRI Patterns of the Hippocampus and Amygdala for Predicting Stages of Alzheimer's Progression: A Minimal Feature Machine Learning Framework (https://arxiv.org/abs/2501.05852)
- **What's New**: 이 논문은 알츠하이머병(AD)의 세 단계인 초기 경도 인지장애(EMCI)와 후기 경도 인지장애(LMCI)를 구분하는 데 중점을 둔 최소한의 특징을 가진 머신러닝 프레임워크를 제안합니다. 기존의 임상 이미징 기술로 인한 혼잡함을 해결하기 위해, 이 프레임워크는 히포캄퍼스(hippocampus)와 아미그달라(amygdala)를 중요한 연구 영역으로 설정하였습니다. 또한, 데이터 전처리 및 차원 축소 기법(PCA 및 t-SNE)을 통해 예측 정확도를 88.46%로 높였습니다.

- **Technical Details**: 본 연구에서 사용된 신경영상 데이터는 ADNI 데이터베이스로부터 확보되었으며, 342개의 T1-weighted 구조적 자기공명영상(sMRI) 이미지를 분석하였습니다. 이 이미지는 사지탈 방향과 MPRAGE 시퀀스를 이용하여 수집되었고, EMCI, LMCI, AD 각각 104, 103, 105 사례를 포함합니다. 데이터는 전처리, 차원 축소 및 다양한 분류기 훈련 과정을 통해 분석되며, 최종적으로 새로운 MRI 스캔에 대해 레이블을 예측합니다.

- **Performance Highlights**: 제안된 프레임워크는 머신러닝 기법과 차원 축소 기술을 통합하여 알츠하이머병의 단계를 정확하게 예측하는 데 성공하였습니다. 이러한 접근법은 노이즈를 줄이고, 지역별 특징을 강조함으로써 향상된 분류 성능을 보여주었습니다. 전체적인 결과는 임상 응용 및 유용한 인사이트를 제공하는 데 기여할 것으로 예상됩니다.



### Annealing Machine-assisted Learning of Graph Neural Network for Combinatorial Optimization (https://arxiv.org/abs/2501.05845)
Comments:
          Second Workshop on Machine Learning with New Compute Paradigms at NeurIPS 2024 (MLNCP 2024)

- **What's New**: 이 논문에서는 Annealing Machines (AM)와 Graph Neural Networks (GNN)을 통합하여 조합 최적화 문제 해결의 효율성을 높이는 방법을 제안합니다. AM은 현재의 기술 수준에서 퀀텀 기술을 대체할 조합 문제 해결에서 경쟁력을 가지지만, 확장성의 한계를 가지고 있습니다. 반면, GNN은 확장성이 뛰어나지만 결과의 정확성이 떨어질 수 있습니다. 이를 바탕으로 두 기술의 장점을 결합한 새로운 프레임워크를 설계하였습니다.

- **Technical Details**: 제안된 프레임워크는 조합 문제를 간단히 압축하는 단계와 AM이 생성한 부분 해결책을 바탕으로 GNN을 개선하는 단계로 구성됩니다. 이 과정에서 AM은 GNN에 지식을 주입하여 간접적으로 문제를 해결하도록 돕습니다. 다양한 그래프 패밀리에 대해 실시한 실험에서, 제안한 모델이 AM의 초기 한계를 넘어서는 문제들을 해결할 수 있음을 보여주었습니다. 또한, GNN의 지역 특징을 종합하여 최종 GNN 기반 해결기를 초기화하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 AM이 더 큰 문제를 해결할 수 있도록 하여 전체 수렴 시간을 단축시켰습니다. 특정 경우에서는 해결 품질이 향상되기도 하여, 두 기술 간의 상호작용에서 발생하는 시너지 효과를 실증적으로 입증하였습니다. 따라서 AM과 GNN의 결합이 조합 최적화 문제의 해결을 위한 효과적인 경로임을 보여주고 있습니다.



### Understanding Impact of Human Feedback via Influence Functions (https://arxiv.org/abs/2501.05790)
Comments:
          Source code: this https URL

- **What's New**: 이번 논문에서는 Human Feedback에서의 강화 학습(RLHF) 과정에서 발생할 수 있는 인간 피드백의 불확실성과 편향을 해결하기 위해 영향 함수(influence functions)를 활용하는 새로운 접근 방식을 제안합니다. 이 연구는 대규모 언어 모델(LLMs)과 함께 사용될 수 있는 계산 효율적인 근사 방법을 통해, 인간 피드백이 보상 모델(reward model)에 미치는 영향을 정량화하고자 합니다.

- **Technical Details**: 영향 함수는 훈련 데이터 포인트가 모델 매개변수에 미치는 영향을 정량화합니다. 본 연구에서는 벡터 압축 기법을 활용하여 LLM 기반 보상 모델에 대한 영향 함수 계산의 속도를 2.5배 개선하는 방법을 소개합니다. 이를 통해 대규모 데이터셋에서의 적용을 용이하게 하고, 실험을 통해 인간 피드백 데이터셋에서 라벨러 편향(labeler bias)을 탐지하는 두 가지 주요 응용 프로그램을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 라벨러 편향을 효과적으로 식별함으로써 기존의 여러 기준 모델을 초과한 성능을 보여줍니다. 또한, 비전문가 라벨러가 전문가 피드백과 더 잘 일치하도록 피드백 전략을 개선하는 데에도 기여할 수 있음을 입증했습니다. 이를 통해 인간 피드백의 해석 가능성을 높이고, 복잡한 작업에 대해 더 정확한 피드백을 제공할 수 있는 기반을 마련하고자 합니다.



### Development and Comparison of Model-Based and Data-Driven Approaches for the Prediction of the Mechanical Properties of Lattice Structures (https://arxiv.org/abs/2501.05762)
Comments:
          This work was funded by the European Union ERC CoDe4Bio Grant ID 101039467 under the funding programme Horizon Europe

- **What's New**: 본 논문에서는 격자 구조(lattice structures)의 기계적 특성과 공극 부피 분율(void volume fraction) 간의 상관관계를 모델링하기 위한 다양한 접근 방식을 제안하고 비교하고 있습니다. 특히, 퓨즈드 디포지션 모델링(fused deposition modeling) 3D 프린팅으로 제작된 여러 유형의 격자 구조를 중심으로 진행됩니다.

- **Technical Details**: 네 가지 접근 모델이 제안됩니다: (i) 간단화된 해석적 모델(simplified analytical model), (ii) 실험적 보정 인자와 해석적 방정식을 결합한 반경험적 모델(semi-empirical model), (iii) 실험 데이터로 학습된 인공 신경망(artificial neural network), (iv) 유한 요소 해석(numerical simulations by finite element analyses). 이러한 다양한 방법론을 통해 격자 구조의 설계를 이해하고 예측할 수 있는 기초 자료를 제공합니다.

- **Performance Highlights**: 각 접근 방식의 성능, 장점 및 단점을 비교함으로써, 적절한 설계 방법론을 선택하기 위한 가이드라인을 제공합니다. 본 연구는 격자 구조 설계의 복잡성을 줄이고, 보다 효율적으로 맞춤형 설계를 가능하게 하는 데 중요한 기여를 할 것으로 기대됩니다.



### CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech (https://arxiv.org/abs/2501.05755)
Comments:
          This paper has been accepted for publication in IEEE SSCI 2025. Copyright belongs to IEEE

- **What's New**: CognoSpeak는 초기 인지 저하의 징후를 자동으로 신속하게 식별하도록 설계된 혁신적인 온라인 AI 도구입니다. 이를 통해 사용자가 가상의 에이전트와 대화하며 기억력, 언어 및 주의력 등을 요구하는 다양한 작업을 수행할 수 있습니다. 이 연구는 126명의 피험자에서 수집된 데이터를 기반으로 하여, 인지 손상을 겪고 있는 사람과 건강한 자원봉사자를 구분하는 데 성공적인 결과를 보여주었습니다.

- **Technical Details**: CognoSpeak는 음성 및 비디오 데이터와 함께 풍부한 메타데이터를 수집하여 인지 상태를 평가합니다. 참가자는 네 가지 아바타 중 하나를 선택하여 심리적 안정을 느끼며 질문에 답하고 전통적인 인지 작업을 수행합니다. 연구진은 다양한 음성 패턴을 이끌어내기 위해 의사 및 계산 언어학자의 입력을 받아 질문을 작성하였으며, 자동 음성 인식 모형을 훈련하기 위해 모든 상호작용은 수동으로 기록됩니다.

- **Performance Highlights**: CognoSpeak 시스템은 DistilBERT 모델을 활용하여 인지 손상 환자를 건강한 자원봉사자와 구별하는 데 있어 F1-score 0.873을 달성하였습니다. 이는 기존의 임상 진단 방식에 비해 낮은 비용과 스트레스, 비침습적인 방식으로 조기 진단의 가능성을 보여줍니다. CognoSpeak는 현재 영국의 1차 및 2차 진료에서 데이터 수집을 진행 중이며, 이는 대규모 데이터 수집의 첫 번째 결과입니다.



### Covariate Dependent Mixture of Bayesian Networks (https://arxiv.org/abs/2501.05745)
- **What's New**: 이번 연구에서는 Bayesian 네트워크의 혼합 모델을 제안하여, 개별 특성에 따른 구성 요소 확률의 의존성을 반영하여 복잡한 데이터 집합의 구조를 더 정확하게 모델링합니다. 전통적인 단일 네트워크 구조 대신, 이 방법은 여러 하위 집단의 차이를 반영함으로써 맞춤형 개입을 지원합니다.

- **Technical Details**: 모델은 각 개별 특성을 기반으로 하여 K개의 혼합 구성 요소로부터 데이터를 생성하는 것으로 정의됩니다. 수정 가능한 특성(modifiable features)과 수정 불가능한 특성(non-modifiable features)으로 나누어, 혼합 확률이 후자의 영향을 받도록 설계되어 있습니다. 이를 통해 더 나은 불확실성 정량화와 의사 결정의 효율성을 높일 수 있습니다.

- **Performance Highlights**: 시뮬레이션과 청소년 정신 건강 사례 연구를 통해 제안된 모델의 유용성을 검증하였으며, 전통적인 방법에 비해 더욱 효과적인 맞춤형 개입을 가능하게 하는 것으로 나타났습니다. 이 연구는 건강, 교육 및 사회 정책 분야에서 보다 정밀한 개인화된 개입을 수행하는 데 기여할 것으로 기대됩니다.



### LLVD: LSTM-based Explicit Motion Modeling in Latent Space for Blind Video Denoising (https://arxiv.org/abs/2501.05744)
- **What's New**: 이번 논문에서는 Latent space LSTM Video Denoiser (LLVD)라는 새로운 알고리즘을 제안합니다. LLVD는 비디오 캡처 중 발생하는 노이즈 제어를 위한 모델로, 영상의 시각적 품질을 향상시키고 불필요한 노이즈 아티팩트를 줄이는 데 중점을 두고 있습니다. 이 모델은 공간적(spatial) 및 시간적(temporal) 피쳐 추출을 통합하여 LSTM을 활용하며, 연속성 유지와 깜빡임 최소화에 중요한 역할을 합니다.

- **Technical Details**: LLVD는 인코딩된 피쳐 도메인 내에서 LSTM 레이어를 통합하여 설계되었습니다. 이 접근 방식은 연속적인 비디오 프레임 간의 시간적 관계를 효과적으로 캡처하고, 높은 계산 효율성을 제공합니다. LLVD는 계산 복잡도를 크게 줄이며, 실시간 어플리케이션에서도 유용하게 사용할 수 있도록 경량화된 구조를 가지고 있습니다.

- **Performance Highlights**: 실험 결과 LLVD는 합성된 노이즈와 실제로 캡처된 노이즈 모두에서 우수한 성능을 보여주었습니다. 특히, LLVD는 기존의 SOTA(State-of-the-Art) 모델보다 0.3dB 높은 성능을 달성하였으며, 계산 복잡도를 59% 감소시켰습니다. 이러한 성과는 노이즈 특성에 대한 사전 지식 없이 이루어졌습니다.



### ELENA: Epigenetic Learning through Evolved Neural Adaptation (https://arxiv.org/abs/2501.05735)
Comments:
          15 pages, 6 figures, 4 tables, 2 algorithms

- **What's New**: 이 논문에서는 복잡한 네트워크 최적화 문제를 해결하기 위한 새로운 진화적 프레임워크인 ELENA (Epigenetic Learning through Evolved Neural Adaptation)를 소개합니다. ELENA는 에피제네틱 메커니즘을 도입하여 적응성을 높이며, 3개의 에피제네틱 태그(돌연변이 저항, 교차 친화도, 안정성 점수)를 통해 솔루션 공간 탐색을 안내합니다. 이를 통해 탐색의 효율성과 효과성을 크게 향상시키고, 특정 네트워크 최적화 문제에서 경쟁력 있는 성과를 달성합니다.

- **Technical Details**: ELENA는 학습 매개변수의 압축 표현과 적응형 돌연변이 연산자, 유도된 유전 물질 전이 메커니즘을 포함하여 기본 진화 접근 방식을 확장합니다. 프레임워크는 여러 하위 집단에서 동시적으로 동작하여 솔루션 공간을 탐색하며, 2-Opt 방식을 이용하여 로컬 검색을 수행합니다. 또한, 동적 유전자 전달 메커니즘을 통해 안정적인 솔루션 세그먼트를 서로 다른 하위 집단 간에 교환할 수 있습니다.

- **Performance Highlights**: ELENA는 여행 판매원 문제(TSP), 차량 경로 문제(VRP), 최대 클리크 문제(MCP)에 대해 실험을 수행하였으며, 경쟁력 있는 결과를 도출하여 최신 방법들보다 우수한 성과를 나타냈습니다. 이 프레임워크는 복잡한 네트워크 최적화 작업에서 기존의 수학적 알고리즘이 전역 최적 해를 찾는 데 어려움을 겪는 상황을 극복할 수 있는 가능성을 보여 줍니다.



### Enabling Scalable Oversight via Self-Evolving Critic (https://arxiv.org/abs/2501.05727)
- **What's New**: 이번 논문에서는 SCRIT (Self-evolving CRITic)이라는 새로운 프레임워크를 제안하여, 대형 언어 모델(LLMs)의 비판 능력을 스스로 발전시키는 방법을 제시합니다. SCRIT는 인간 평가가 어려운 작업에서 LLM의 피드백 효율성을 높이는 것을 목표로 하며, 스스로 훈련하는 방식으로 비판 데이터를 생성합니다. 특히, 대조적 자기 비판 기법과 자기 검증 메커니즘을 통해 무인 감독으로 비판 품질을 향상시킵니다.

- **Technical Details**: SCRIT의 핵심 단계는 먼저 참조 솔루션을 바탕으로 학생 솔루션을 분석하고 비판하는 대조적 비판 기법을 개발하는 것입니다. 이어서 LLM은 생성된 비판이 수학적으로 유효한 솔루션으로 이어지는지를 자기 검증합니다. 이 두 단계는 고품질 비판 데이터를 생성하고, LLM의 비판 능력을 지속적으로 향상시키는 데 기여합니다.

- **Performance Highlights**: SCRIT은 Qwen2.5-72B-Instruct 모델을 기반으로 하여 비판 수정 및 오류 식별 작업에서 최대 10.3%의 성능 향상을 달성했습니다. 다양한 데이터 세트와 평가 프로토콜에서 일관된 개선 결과를 보여주며, SCRIT 구현 시 기존 모델의 출력을 크게 향상시켰습니다. 또한 SCRIT은 데이터와 모델 크기가 커질수록 성능이 긍정적으로 확장됨을 나타내어, LLM의 비판 능력 강화에 있어 중요한 진전을 보여줍니다.



### Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains (https://arxiv.org/abs/2501.05707)
Comments:
          22 pages, 13 figures, 7 tables; Project page at this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 자기 개선(self-improvement)을 위한 새로운 접근 방식을 제안합니다. 기존의 모델들을 다수의 에이전트로 구성하여 각 모델이 상호작용을 통해 독립적으로 전문화되는 방식입니다. 이 방법은 다수의 모델이 역할을 분담하여 응답의 다양성과 사고 체계를 유지하면서 더욱 많은 반복을 통한 성능 향상을 가능하게 합니다.

- **Technical Details**: 논문에서는 다중 에이전트 설정에서 LLM 모델을 전문화하는 방법을 도입합니다. 다중 에이전트 토론(multiagent debate) 방식을 통해 각 모델이 생성한 데이터를 활용하여 피드백을 주고받으며, 반복적 훈련을 통해 다양성을 유지하고 독립적으로 전문화합니다. 이 과정에서 데이터는 독립된 데이터 세트에서 수집되어 각 모델에 맞게 조정됩니다.

- **Performance Highlights**: 제안한 방법은 여러 추론 작업에 대해 정량적으로 유효성을 입증하며, 오픈 소스 LLM부터 상용 LLM에 이르기까지 다양한 모델에 적용 가능합니다. 실험 결과, 단일 에이전트 자기 개선 방법보다 훨씬 더 많은 반복을 통한 성능 개선을 확인했으며, 새로운 데이터 세트에서도 우수한 일반화를 보여주었습니다.



### EXION: Exploiting Inter- and Intra-Iteration Output Sparsity for Diffusion Models (https://arxiv.org/abs/2501.05680)
Comments:
          To appear in 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA 2025)

- **What's New**: 최근 몇 년간 확산 모델(difusion models)은 텍스트 프롬프트를 기반으로 다양한 다중 모드 출력(multi-modal outputs)을 생성하는 새로운 AI 솔루션으로 부각되었습니다. 그러나 이러한 모델은 반복적 구조로 인한 지나친 지연(latency)과 에너지 소모의 문제를 안고 있습니다. 본 논문에서는 EXION이라는 최초의 소프트웨어-하드웨어 공동 설계(SW-HW co-designed) 확산 가속기를 제안하여 이러한 계산 문제를 해결합니다.

- **Technical Details**: EXION은 두 가지 SW 수준의 최적화 방법을 기초로 합니다. 첫 번째로, FFN-Reuse 알고리즘을 통해 서로 다른 반복(iteration) 간에 FFN 레이어의 중복 계산을 식별하고 건너뛰는 inter-iteration sparsity를 구현합니다. 두 번째로, 수정된 eager prediction 방법을 도입하여 각 반복 내에서 attention score를 정확히 예측함으로써 불필요한 계산을 생략하는 intra-iteration sparsity를 성취합니다. 이와 함께, sparse matrix를 압축하고 병합하는 새로운 데이터 압축 메커니즘인 ConMerge를 통해 하드웨어 활용도를 높입니다.

- **Performance Highlights**: EXION의 효율성을 검증하기 위해 다양한 다중 모드 확산 모델에서 정확도에 미치는 영향이 없음을 확인하였고, 서버와 엣지 레벨 설정에서 EXION을 구현하여 성능을 비교하였습니다. 그 결과, EXION은 서버 GPU(NVIDIA RTX 6000 Ada) 대비 3.2-379.3배 향상된 성능과 45.1-3067.6배 증가된 에너지 효율성을 보였으며, 엣지 GPU 대비 42.6-1090.9배의 성능 및 196.9-4668.2배의 에너지 효율성 개선을 기록하였습니다.



### Facilitate Collaboration between Large Language Model and Task-specific Model for Time Series Anomaly Detection (https://arxiv.org/abs/2501.05675)
- **What's New**: 이 논문에서는 CoLLaTe라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLM)과 작업 특화 모델 간의 협력을 촉진하고자 한다. 이 프레임워크는 두 모델의 강점을 활용하여 이상 탐지(anomaly detection)의 효율성을 높인다. 또한, LLM의 전문 지식을 작업 특정 모델에 통합하여 데이터 부족 문제를 해결함으로써 성능 저하를 방지한다.

- **Technical Details**: CoLLaTe는 두 가지 주요 구성 요소인 정렬 모듈(alignment module)과 협업 손실 함수(collaborative loss function)를 도입하여 협업 과정에서 발생하는 두 가지 주요 도전 과제를 해결한다. 첫 번째 과제는 LLM과 작업 특화 모델 간의 표현 도메인(misalignment) 불일치이며, 두 번째는 두 모델의 예측 오류(error accumulation)다. 이 구성 요소들은 서로 다른 모델들의 이상 점수 해석 차이를 조정하고 예측 오류의 누적을 방지하도록 설계되어 있다.

- **Performance Highlights**: CoLLaTe의 효과는 이론적 분석과 실험적 검증을 통해 입증되었으며, 기존 LLM 기반 및 작업 특화 모델보다 더 나은 성능을 보여준다. 프레임워크는 다양한 응용 시나리오에 걸쳐 도메인 전문 지식을 효과적으로 통합할 수 있는 능력을 갖추고 있다. CoLLaTe는 또한 각 시간 슬롯에 대해 이상 점수(anomaly score)를 통합하여 최종적인 수치 평가를 생성하는 조건부 네트워크(conditional network)를 적용한다.



### Learning to Measure Quantum Neural Networks (https://arxiv.org/abs/2501.05663)
Comments:
          Accepted by ICASSP 2025 Workshop: Quantum Machine Learning in Signal Processing and Artificial Intelligence

- **What's New**: 이번 연구에서는 새로운 양자 기계 학습(QML) 모델 접근 방식을 제안합니다. 기존의 고정된 측정 가능성을 활용한 모델들이 아닌, 가변적인 Hermitian matrix를 사용하여 QML 모델의 성능을 개선할 수 있는 방안을 제시합니다. 특히, 제안된 방법은 측정 과정을 최적화하여 효율적인 학습이 가능하도록 하여, 마지막 결과 성능을 높이는 데 기여합니다.

- **Technical Details**: 제안된 접근법에서는 Hermitian 행렬을 파라미터화하여 양자 회로의 일반적인 파라미터와 동시에 훈련합니다. 이를 통해, Variational Quantum Circuits(VQCs)의 결과를 향상시키기 위한 학습 가능한 관측 변수를 자동으로 발견하는 방법론을 구현합니다. 수치 시뮬레이션 결과에 따르면, 본 프레임워크는 고정된 관측 가능성을 이용한 일반적인 VQC 훈련보다 더 우수한 성과를 보여줍니다.

- **Performance Highlights**: 우리는 다양한 기계 학습 작업에서 높은 분류 정확도를 달성하여 QML 모델의 전반적인 성능을 증대시켰습니다. Hermitian 행렬의 스펙트럼 범위를 넓힘으로써 VQC의 출력 범위를 확장할 수 있음을 입증하였습니다. 이 연구 결과는 QML이 다양한 문제를 더욱 효과적으로 해결할 수 있는 가능성을 열어줍니다.



### Evidential Deep Learning for Uncertainty Quantification and Out-of-Distribution Detection in Jet Identification using Deep Neural Networks (https://arxiv.org/abs/2501.05656)
Comments:
          38 pages (including references) with 17 figures and 3 tables. Repository: this https URL . Submitted to Machine Learning: Science and Technology

- **What's New**: 이 논문은 깊은 신경망(Deep Neural Network, DNN) 모델의 불확실성 정량화(Uncertainty Quantification, UQ)를 위한 새로운 방법론, 즉 증거 기반 깊은 학습(Evidential Deep Learning, EDL)을 제시하고 있습니다. EDL은 학습을 증거 획득 과정으로 간주하여 테스트 데이터에 대한 신뢰성을 제공하는 방식으로, 전통적인 베이지안 방법보다 계산 비용이 적고 효율적입니다. 특히, 고에너지 양성자-양성자 충돌에서 제트를 식별하기 위한 DNN 모델을 통해 EDL의 유용성을 탐구하며, 이를 통해 이상 탐지(Anomaly Detection)에서도 응용할 수 있는 가능성을 제시합니다.

- **Technical Details**: 논문에서는 EDL의 주요 개념 및 이론적 기반에 대해 설명합니다. EDL은 Dempster-Shafer 이론을 활용하여 신뢰도 고차원 분포에서 샘플링하여 불확실성을 모델링합니다. 이는 aleatoric 불확실성(훈련 데이터의 노이즈)과 epistemic 불확실성(불충분한 학습 데이터)을 구분하고, 특히 EDL이 epistemic 불확실성을 정량화하는 데 혁신적인 접근 방식을 제공함을 강조합니다. 또한, EDL은 jet tagging에 적용되어 훨씬 더 명확하고 해석 가능한 모델을 만드는 데 기여하고 있습니다.

- **Performance Highlights**: EDL 기반 불확실성 추정치는 기존의 Ensemble 및 Bayesian 방법들과 비교하여 정확성과 신뢰성을 향상시킵니다. 논문에서는 EDL 방법이 세 가지 공개 데이터셋에서 다양한 제트 클래스의 불확실성을 어떻게 변화시키는지를 분석하며, 해당 불확실성 분포를 해석합니다. EDL을 활용한 이상 탐지에서의 성능 또한 조명을 받을 필요가 있으며, 이러한 방법들이 고에너지 물리 실험(Higher Energy Physics, HEP) 분야에서 더욱 중요해짐에 따라, 실험적 분석의 정확성을 향상시키는 데 유용할 것입니다.



### A Practical Cross-Layer Approach for ML-Driven Storage Placement in Warehouse-Scale Computers (https://arxiv.org/abs/2501.05651)
- **What's New**: 이 논문은 기계 학습(ML)을 활용하여 저장 시스템의 효율성을 향상시키는 새로운 접근법을 제안합니다. 특히 기존의 저장 계층에 한정된 모형이 아닌, 애플리케이션과 저장 계층 간의 크로스 레이어(cross-layer) 접근법을 통해 ML을 적용하는 방법을 모색하고 있습니다. 저자들은 Google의 실제 데이터 센터 환경에서 이 시스템을 실증적으로 검증하였으며, ML 모델을 애플리케이션 레이어에 배치하고 이를 저장 계층의 스케줄링 알고리즘과 결합하였습니다.

- **Technical Details**: 기존의 데이터 배치 방법은 주로 휴리스틱(heuristic) 방식에 의존했으며, 이는 SSD의 용량이 제한적일 때 최적의 성능을 발휘하지 못했습니다. 이에 대한 대안으로, 본 연구에서는 애플리케이션 레이어에서 예측을 수행하고 이를 기반으로 저장 계층에서의 데이터 배치를 조정하는 접근법을 제안합니다. 저자들은 ML 모델을 데이터 속성에 따라 랭킹을 예측하는 것으로 설계하였고, 저장 계층에서는 이 예측 결과를 사용하여 SSD에 데이터를 배치하는 적응형 알고리즘을 개발하였습니다.

- **Performance Highlights**: 실제 테스트 배포 환경에서 이 접근법의 검증 결과는 TCO(총 소유 비용) 절감에 있어 최대 3.47배의 개선 효과를 보였습니다. 본 연구의 결과는 기계 학습 기반의 저장 배치 방법이 데이터 센터의 운영 효율성을 크게 향상시킬 수 있음을 보여줍니다. 이를 통해 대규모 데이터 처리 환경에서 기계 학습의 실용적인 적용 가능성을 한층 더 다져나가고 있습니다.



### Interpretable Enzyme Function Prediction via Residue-Level Detection (https://arxiv.org/abs/2501.05644)
- **What's New**: 이 논문에서는 효소 기능 예측을 탐지 문제로 간주하여 새로운 주목 기반 프레임워크인 ProtDETR(Protein Detection Transformer)을 제안합니다. 이는 효소의 아미노산 서열에서 다양한 지역 표현을 동적으로 추출하여, 각 효소에 대한 EC 번호를 더 정확히 예측할 수 있도록 합니다. ProtDETR은 기존의 방법들보다 우수한 성능을 보일 뿐만 아니라 서로 다른 지역을 자동으로 탐지하는 해석가능한 관점을 제공합니다.

- **Technical Details**: ProtDETR은 잔여 수준의 특징을 활용하기 위해 Transformer 기반의 인코더-디코더 아키텍처를 사용합니다. 학습 가능한 쿼리 토큰 세트를 통해 아미노산 서열에서 여러 지역적 표현을 생성하며, 각 쿼리는 주의 기법(cross-attention)을 통해 아미노산의 다양한 부분에 집중합니다. 이 과정에서 특정 활성 및 반응 부위와 같은 잔여 조각을 정제하여, 효소 기능에 대한 정확한 분류를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, ProtDETR은 기존의 최신 방법들과 비교하여 F1 점수가 더 높은 결과를 보여주었습니다. New-392 데이터셋에서 ProtDETR은 0.5943의 정밀도를 기록하며, CLEAN의 0.5965에 가까운 성능을 보였지만, 0.6083의 리콜을 달성하여 CLEAN의 0.4811보다 25% 높은 수치를 기록했습니다. 또한, Price-149 데이터셋에서도 similar한 경향을 보이며, 멀티기능 효소 주석화 작업에서의 효과성을 입증하였습니다.



### Towards Probabilistic Inference of Human Motor Intentions by Assistive Mobile Robots Controlled via a Brain-Computer Interfac (https://arxiv.org/abs/2501.05610)
Comments:
          10 pages

- **What's New**: 이 논문은 Brain Computer Interface (BCI) 기반의 모바일 로봇 시스템의 지각-행동 주기를 재설계하여 장애인이 자연스럽고 연속적인 속도 조정을 할 수 있도록 개선하려는 연구입니다. 기존의 BCI 시스템은 사용자의 의도를 인식할 수 있지만, 속도 조정이 불연속적인 단계로 이뤄지기 때문에 유연한 움직임을 모방하지 못했습니다. 따라서 이 논문에서는 그 한계를 극복하고자 합니다.

- **Technical Details**: 논문의 초점은 지각(perception) 부분으로, 로봇 에이전트가 불완전하거나 노이즈가 포함된 감각 관측치를 최적으로 인식하기 위해 수행해야 할 계산(computation)에 관한 규범적 질문을 다룹니다. 실험적인 EEG 데이터를 수집하고, 이를 통해 생성적 적대 신경망(Generative Adversarial Network) 프레임워크 내에서 세계 상태 분포(world state distributions)를 학습하고 평가했습니다. 또한, ROS 프레임워크를 통해 실내 공간의 디지털 트윈(digital twin)과 가상의 로봇 휠체어 모델과 연결되는 환경을 구축했습니다.

- **Performance Highlights**: 신호 처리 및 통계 분석을 통해 공간-스펙트럼-시간 차원에서 가장 차별적(discriminative)인 특징을 식별하였고, 이 특징을 바탕으로 로봇 에이전트가 사용자 움직임의 의도를 베이지안 관찰자로 해석할 수 있도록 세계 모델을 구성했습니다. 이렇게 구축된 시스템은 BCI 기반 휠체어의 조작성을 획기적으로 개선할 수 있는 잠재력을 가지고 있습니다.



### Learned Discrepancy Reconstruction and Benchmark Dataset for Magnetic Particle Imaging (https://arxiv.org/abs/2501.05583)
- **What's New**: 이번 연구에서는 Magnetic Particle Imaging (MPI)의 이미지 재구성 문제를 해결하기 위해 Learned Discrepancy Approach라는 새로운 학습 기반 재구성 방법을 제안합니다. 이 방법은 전통적인 재구성 기법에 비해 문제특정 노이즈 분포를 명시적으로 모델하고 Gaussian 노이즈 가정에 의존하지 않습니다. 더불어, MPI-MNIST 데이터셋을 소개하여 손글씨 숫자로부터 생성된 대규모 MPI 측정을 제공하여 재구성 알고리즘의 테스트에 필요한 현실적인 환경을 조성합니다.

- **Technical Details**: MPI는 슈퍼파라 자기 나노입자의 자기화 거동을 활용하는 이미징 기술로, 높은 해상도의 실시간 이미징을 가능하게 합니다. 그러나 MPI의 이미지 재구성 특징으로 인해 기존의 Gaussian 기본 가정을 따르지 않는 노이즈 모델 문제로 인해 어려움이 있습니다. 이를 해결하기 위해 재구성을 위한 학습된 불일치 함수와 더불어, 가역적 신경망을 통합하여 노이즈 분포를 모델링합니다.

- **Performance Highlights**: 제안된 방법은 MPI-MNIST 데이터셋에 대해 검증되었으며, 기존의 재구성 기법에 비해 구조적 유사성에서 유의미한 개선을 보여줍니다. 또한, 데이터 기반의 재구성 접근 방식은 복잡한 문제 특화 노이즈 모델을 직접 고려하여 기존의 단순화된 노이즈 모델 문제를 해결하며, 다양한 역 문제에 적용 가능성을 보유하고 있습니다.



### Physics-Driven Learning for Inverse Problems in Quantum Chromodynamics (https://arxiv.org/abs/2501.05580)
Comments:
          14 pages, 5 figures, submitted version to Nat Rev Phys

- **What's New**: 이 논문은 깊은 학습 기법(deep learning techniques)과 물리 기반 설계(physics-driven designs)의 통합을 통해 역 문제(inverse problems)를 해결하는 새로운 방법론을 제시하고 있습니다. 특히, 양자 색역학(quantum chromodynamics, QCD) 분야에 중점을 두고 있으며, 물리적 양의 예측에 대한 최신 발전과 머신러닝(machine learning)과의 연결성을 강조합니다. ML과 물리학의 융합은 문제 해결의 효율성과 신뢰성을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 역 문제 해결을 위한 방법론으로 통계적 추론(statistical inference)과 깊은 학습(deep learning) 기술이 나타났습니다. 통계적 추론은 불확실성을 수량화하는 강력한 프레임워크를 제공하며, 베이지안 추론(Bayesian inference)을 통해 물리적 매개변수를 업데이트하는 시스템을 개발합니다. 또한, 다양한 신경망 아키텍처 CNN(convolutional neural networks), RNN(recurrent neural networks), GNN(graph neural networks) 등이 역 문제 해결에 활용될 수 있습니다.

- **Performance Highlights**: 물리 기반 학습 방법론은 대규모 모델에서 여전히 부족한 정확성을 개선하는 데 기여할 수 있는 가능성을 보여줍니다. 본 논문에서는 대칭(symmetry), 연속성(continuity), 물리적 방정식의 통합을 통해 역 문제를 다루는 다양한 응용 사례를 제시하며, 이는 하드론 물리, 중성자 별 관측, 강입자 충돌 실험에서의 응용을 포함합니다. 이러한 접근은 물리적 현실과 일치하는 출력 결과를 보장하는 데 중요한 역할을 합니다.



### Prediction-Assisted Online Distributed Deep Learning Workload Scheduling in GPU Clusters (https://arxiv.org/abs/2501.05563)
Comments:
          INFOCOM 2025

- **What's New**: 이 논문에서는 GPU 클러스터에서 혼합 병렬성을 가진 분산 딥러닝 훈련(DDLwMP)을 위한 효율적인 작업 스케줄링의 필요성을 강조하며, 적응형 최단 남은 처리 시간 우선(A-SRPT) 스케줄링 알고리즘을 제안합니다. A-SRPT는 각 작업을 그래프 형태로 모델링하고, 서로 다른 DNN 아키텍처와 훈련 구성을 고려하여 작업을 전략적으로 GPU에 할당하여 연속 서버 간 통신 오버헤드를 최소화합니다. 이 알고리즘은 훈련 반복을 예측하는 랜덤 포레스트 회귀 모델을 통합하여 DDLwMP 작업의 예측 기능을 개선하고, 복잡한 스케줄링 문제를 단일 머신 문제로 변환하여 최적의 해법을 제시합니다.

- **Technical Details**: A-SRPT 알고리즘은 두 가지 핵심 요소로 구성됩니다. 첫 번째로, DDLwMP 작업을 특정 GPU 집합에 효과적으로 매핑하는 GPU 매핑 알고리즘을 통해 데이터 통신 오버헤드를 최소화합니다. 두 번째로, 랜덤 포레스트 회귀 모델을 기반으로 한 예측 보조 온라인 스케줄링 알고리즘을 통해 DDLwMP 작업을 전략적으로 스케줄링합니다. 이 과정에서, 예측된 훈련 반복 수를 기반으로 원래의 복잡한 다차원 GPU 클러스터링 문제를 선제적 단일 머신 스케줄링 문제로 단순화합니다.

- **Performance Highlights**: 실제 테스트베드와 시뮬레이션 실험을 통해 제안된 A-SRPT 알고리즘의 효과성을 검증하였고, 성능이 기존 가장 최신 DDL 스케줄링 알고리즘을 초월함을 실증적으로 확인했습니다. 특히, 제안된 알고리즘은 모든 기준 설계보다 더 뛰어난 성능을 보였으며, 최대 92%의 전체 작업 완료 시간 단축을 달성했습니다. 이러한 연구 결과는 DDLwMP 분야의 스케줄링 디자인이 발전하는 데 중요한 기여를 할 것입니다.



### OmniJet-${\alpha_{ C}}$: Learning point cloud calorimeter simulations using generative transformers (https://arxiv.org/abs/2501.05534)
- **What's New**: 이 논문에서는 고해상도 칼로리미터에서 생성적 변환기(generative transformers)를 사용하여 칼로리미터 샤워를 포인트 클라우드(point clouds) 형식으로 생성한 첫 번째 사례를 보여줍니다. OmniJet-${\alpha}$ 모델의 토크나이저(tokenizer) 및 생성 부분을 활용하여 탐지기의 충격(hits)을 정수 시퀀스로 표현합니다. 이 모델은 가변 길이 시퀀스를 지원하여 실제적인 샤워 발전을 가능하게 하며, 충격 수에 대한 조건 없이 작동합니다.

- **Technical Details**: 이 연구는 전자-양전하 충돌기인 국제 선형 충돌기(ILC)에서 사용하는 국제 대형 탐지기(ILD)의 데이터셋을 기반으로 합니다. ILD는 모든 개별 입자를 재구성하기 위한 파티클 흐름 알고리즘(Particle Flow Algorithm)에 최적화되어 있으며, 정밀한 트래킹 및 정점 검출 기능을 갖추고 있습니다. 사용하는 데이터셋은 까다로운 지오메트리 모델을 통해 생성된 포인트 클라우드 형식으로 변환되어, 각 점이 위치(x, y, z)와 에너지를 특징으로 합니다.

- **Performance Highlights**: OmniJet-${\alpha}$ 아키텍처는 전혀 다른 하위 영역인 전자기 샤워 생성에서도 성공적으로 작동하며, 이는 샤워에서의 전이 학습(transfer learning)을 탐색할 수 있는 가능성을 열어줍니다. 이 연구는 공통적인 계산 프레임워크 내에서 서로 매우 다른 하위 영역의 작업을 처리할 수 있는 첫 번째 예시로, 입자 물리학의 모든 컴퓨팅 및 데이터 분석 작업을 위한 기초 모델 개발로 나아가는 중요한 단계로 평가됩니다.



### Outlyingness Scores with Cluster Catch Digraphs (https://arxiv.org/abs/2501.05530)
Comments:
          29 pages, 7 figures, 16 tables

- **What's New**: 본 논문에서는 Cluster Catch Digraphs (CCDs)를 기반으로 한 두 가지 새로운 outlyingness 점수인 Outbound Outlyingness Score (OOS)와 Inbound Outlyingness Score (IOS)를 소개합니다. 이 점수들은 이상치 탐지 결과의 해석 가능성을 높이며, 고차원 데이터의 다양한 클러스터 형태 및 강도에 맞춰 설계되었습니다. OOS는 포인트와 가장 가까운 이웃들을 기준으로 이상치 정도를 평가하고, IOS는 클러스터 내 다른 포인트로부터 받은 총 "영향력"을 측정합니다.

- **Technical Details**: 새롭게 제안된 두 가지 기준점(OOS와 IOS)은 각각 이웃 포인트와의 관계를 기반으로 이상치 정도를 정량화합니다. OOS는 이웃들에 대한 출발 지점의 이상치 정도를 평가하며, IOS는 클러스터 내의 다른 멤버들로부터의 누적 영향을 역으로 계산하여 이를 측정합니다. 이들 기법은 특히 고차원 데이터셋에서 전통적인 이상치 탐지 방법보다 더 우수한 성능을 보여주며, 데이터의 콜리너리(correlated) 문제에 대해서도 불변성을 가집니다.

- **Performance Highlights**: 광범위한 몬테카를로 시뮬레이션을 통해 OOS와 IOS는 CCD 기반의 기존 및 최첨단 방법들과 비교되었습니다. 특히 IOS는 인공 및 실제 데이터셋 모두에서 뛰어난 성능을 발휘하여 모든 방법 중에서 가장 우수한 결과를 보였습니다. OOS와 IOS는 데이터 집합의 글로벌 및 로컬 이상치를 효과적으로 식별하며, 이상치 탐지 분야의 해석 가능성 및 성능 향상에 기여합니다.



### The more polypersonal the better -- a short look on space geometry of fine-tuned layers (https://arxiv.org/abs/2501.05503)
Comments:
          Neuroinformatics 2024

- **What's New**: 이 논문은 심층 학습 모델의 해석(interpreting) 분야에서 특히 언어 모델에 대한 새로운 접근 방식을 제안합니다. 저자들은 BERT 모델에 새로운 문법 모듈(grammatical module)과 문법 구조(polypersonality)가 포함된 데이터를 추가하여 훈련할 때 내부 표현(internal representation)의 변화를 분석합니다.

- **Technical Details**: BERT 모델에 단일 문법 레이어를 추가함으로써 모델이 새로운 문법 체계와 기존의 문법 체계를 구분하는 과정을 보여줍니다. 이는 모델의 의사 결정 과정에서 패턴을 식별할 뿐만 아니라, 내부 구조의 특성을 이해하는 데 기여합니다.

- **Performance Highlights**: 저자들은 추가된 문법 레이어 덕분에 모델의 perplexity 메트릭(perplexity metrics)에서 전체적인 성능이 향상된 것을 발견했습니다. 이러한 개선은 모델의 문법 처리 능력을 더욱 향상시키는 중요한 발견입니다.



### Strategy Masking: A Method for Guardrails in Value-based Reinforcement Learning Agents (https://arxiv.org/abs/2501.05501)
- **What's New**: 이 논문에서는 AI 에이전트가 보상 함수(reward function)를 이용해 학습하도록 만드는 경계책을 구축하는 방법을 연구합니다. 새로운 접근 방식인 전략 마스킹(strategy masking)을 소개하며, 이는 바람직하지 않은 에이전트 행동을 명시적으로 학습하고 억제하도록 설계되었습니다. 이를 통해 AI가 더 정직하게 행동하면서도 효과적으로 작동할 수 있도록 합니다.

- **Technical Details**: 강화 학습(reinforcement learning)에서 보상 함수는 에이전트가 행동을 결정하는 데 중요한 역할을 합니다. 논문에서는 보상 신호를 분해하여 특정 행동의 기대 가치를 여러 차원으로 모델링하고 이에 따라 전략적으로 마스킹합니다. 이러한 방식을 통해 AI 에이전트의 학습 과정에서 행동을 더욱 세밀하게 조절할 수 있습니다.

- **Performance Highlights**: Coup라는 사회적 속임수 게임을 통해 전략 마스킹을 적용하여 AI 에이전트가 거짓말하는 행동을 억제하는데 성공했습니다. 이 방법을 통해 에이전트는 거짓말을 감추고 더 정직하게 행동하는 경향을 보였으며, 게임의 승리에 필요한 능력 또한 저해되지 않았습니다. 결과적으로, 전략 마스킹은 AI의 행동을 조정하는 유효한 방법임을 입증했습니다.



### S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis (https://arxiv.org/abs/2501.05485)
- **What's New**: 이 논문에서는 문서 청크(chunking) 작업의 중요성을 강조하며, 기존의 방법들이 문서 내의 공간적 레이아웃을 무시하고 있다는 문제를 해결하기 위해 새로운 하이브리드 접근 방식을 제안합니다. 이 방법은 레이아웃 구조, 의미 분석, 공간적 관계를 통합하여 문서 청크의 응집력과 정확성을 향상시킵니다. 또한, 이 접근법은 복잡한 레이아웃을 가진 문서에서도 뛰어난 성능을 보이며, 토큰 길이 제한을 준수할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계인 영역 감지(region detection)와 영역 레이아웃 정렬(region layout ordering)로 구성됩니다. 첫 번째 단계에서는 문서 내 각 분류된 영역에 대한 바운딩 박스(bbox) 데이터를 추출하고, 두 번째 단계에서는 이러한 영역들을 구조적 유형에 따라 합리적인 순서로 배열합니다. 이후 그래프를 구성하고, 가중치를 계산하며, 클러스터링을 통해 일관된 청크로 나누는 과정을 진행합니다.

- **Performance Highlights**: 실험 결과 제안된 하이브리드 접근법은 PubMed와 arXiv에서 다양한 레이아웃과 내용을 가진 연구 논문 데이터셋에서 기존 방법들보다 우수한 성능을 보였습니다. 이 방법은 문서 내의 의미적 관계와 공간적 관계를 모두 고려하여 청크를 생성함으로써, 복잡한 문서에서도 적절한 분할을 수행할 수 있습니다. 또한, 평균적으로 높은 품질의 청크를 생성하여 정보 검색, 요약 및 질문 답변과 같은 NLP 작업에서 더 나은 성능을 보입니다.



### Practical Design and Benchmarking of Generative AI Applications for Surgical Billing and Coding (https://arxiv.org/abs/2501.05479)
Comments:
          21 pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 의료 분야의 청구 및 코딩 프로세스를 위해 Generative AI 도구를 개발하는 새로운 전략을 제안합니다. 특히, 기존의 Large Language Models (LLMs)가 ICD-10 및 CPT 코드 생성을 할 때의 정확도가 낮다는 문제를 해결하고자 하였습니다. 이 모델은 접근성과 환자의 개인 정보 보호를 균형 있게 고려하면서, 의료 청구 및 코딩의 정확성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: PHI-3 Mini 및 PHI-3 Medium 모델을 기관 데이터를 활용하여 미세 조정(fine-tuning)하고, 결과를 PHI-3 기본 모델 및 GPT-4o 모델과 비교하였습니다. 연구는 수술 후 보고서를 입력하고, 환자의 청구 청구서와 관련된 ICD-10, CPT, Modifier 코드를 생성하는 과정을 포함합니다. 성능 측정은 코드 생성의 정확성, 잘못된 코드의 비율, 청구서 형식의 충실도를 기준으로 진행하였습니다.

- **Performance Highlights**: 미세 조정된 두 개의 모델은 GPT-4o 모델과 비교하여 더 좋은 성능을 보였으며, 특히 Phi-3 Medium 모델이 가장 우수한 결과를 나타냈습니다. 이 모델의 ICD-10 Recall 및 Precision은 각각 72%로 알려졌으며, CPT Recall과 Precision은 각각 77%, 79%에 달했습니다. 이 모델은 또한 생성된 ICD-10 코드 중 1%, CPT 코드 중 0.6%만이 잘못된 것으로 나타났습니다.



### Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models (https://arxiv.org/abs/2501.05478)
- **What's New**: 이 연구는 로봇 공학에서 비전-언어 내비게이션(VLN) 분야에 아랍어 통합을 첫 번째로 선보이며, 다국어 Small Language Models(SLMs)와 아랍어 중심의 LLM인 Jais의 성능을 평가합니다. 분명히 부족했던 아랍어 데이터에 대한 연구의 공백을 메우면서, NavGPT 프레임워크와 R2R 데이터셋을 사용하여 아랍어와 영어 간의 의사소통이 내비게이션 추론에 미치는 영향을 평가합니다. 이를 통해, 아랍어로 지시를 받았을 때의 로봇 내비게이션 작업의 계획 및 추론 능력을 강조하였습니다.

- **Technical Details**: 본 연구는 OpenAI의 GPT-4o mini, Meta의 Llama 3 8B와 Microsoft의 Phi-3 medium 14B와 같은 최신 다국어 SLM들과 Jais 30B LLM을 NavGPT 프레임워크 내에서 비교합니다. R2R 데이터셋을 활용하여 영어 내비게이션 지시를 아랍어로 변환한 데이터셋으로 보면, 다양한 언어로 내비게이션 자원에 접근하고자 하는 양방향적 연구의 필요성을 강조합니다. 또한, 제로샷 방식으로 작업을 예측하며, 언어의 영향력에 대한 분석을 도모합니다.

- **Performance Highlights**: 실험 결과, NavGPT 프레임워크가 영어 및 아랍어 지시를 통해 높은 수준의 내비게이션 계획을 수행할 수 있음을 입증하였습니다. 그러나 일부 모델은 아랍어에서 추론 및 계획에 어려움을 겪어 언어 모델의 성능과 한계를 드러냈습니다. 이러한 발견은 아랍어 모델의 발전과 현실 세계 응용 프로그램에서의 가능성을 열어주며, 연구의 향후 방향으로 언어 모델의 계획 및 추론 능력 향상이 필요함을 강조합니다.



### Modality-Invariant Bidirectional Temporal Representation Distillation Network for Missing Multimodal Sentiment Analysis (https://arxiv.org/abs/2501.05474)
Comments:
          Accepted for publication by 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이번 연구에서는 결측된 다중 양식(Missing Multimodal) 감정 분석을 위한 Modality-Invariant Bidirectional Temporal Representation Distillation Network (MITR-DNet)를 소개합니다. MITR-DNet은 완전한 양식 teacher 모델이 결측 양식 student 모델을 지도하는 방식의 distillation 기술을 활용하여, 양식 부족이 발생할 경우에도 강인성을 보장합니다. 또한, 다중 양식 데이터의 이질성 문제를 완화하기 위한 Modality-Invariant Bidirectional Temporal Representation Learning Module (MIB-TRL)을 개발하였습니다.

- **Technical Details**: 본 연구에서는 음성(a), 텍스트(t), 비전(v)의 세 가지 양식을 고려합니다. 결측 양식 기능을 시뮬레이션하기 위해 마스킹 함수 F(·)와 랜덤 생성 시간 마스크 gm을 도입하였으며, 이를 통해 불완전한 시퀀스 X~m를 생성합니다. MIB-TRL 모듈은 두 개의 동일한 합성곱 층을 통해 입력 데이터를 처리하고, 이 방향으로 나뉜 데이터 스트림을 통합하여 각 양식의 장기 맥락 표현을 생성합니다.

- **Performance Highlights**: MITR-DNet과 MIB-TRL 모듈은 결측 양식 감정 분석 분야에서의 정확성 및 신뢰성을 크게 향상시켰습니다. 연구 결과, 다중 양식을 활용한 접근 방식은 단일 양식 분석보다 더 세밀하고 정확한 감정 평가를 가능하게 하며, 실제 세계 상황에서의 양식 결측으로 인한 문제를 효과적으로 해결합니다. 또한, 이 혁신적인 구조는 감정 예측에서 텍스트 양식의 중요성을 최대로 활용하여 향상된 성능을 보여줍니다.



### The 2nd Place Solution from the 3D Semantic Segmentation Track in the 2024 Waymo Open Dataset Challeng (https://arxiv.org/abs/2501.05472)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 자율주행 기술의 필수 요소인 3D semantic segmentation의 성능을 향상시키기 위해 MixSeg3D라는 새로운 접근 방법을 제안합니다. 기존의 LiDAR 스캔 데이터가 가진 훈련의 다양성 부족 문제를 해결하기 위해, 저자들은 MinkUNet 모델을 사용하고 LaserMix 및 PolarMix라는 두 가지 데이터 증강 기법을 결합했습니다. 이 방법은 LiDAR 포인트 클라우드를 다양한 각도에서 혼합하여, 모델이 여러 환경에서 잘 일반화하도록 돕습니다. 이를 통해 저자들은 2024 Waymo Open Dataset Challenge에서 3D semantic segmentation 트랙에서 2위를 기록했습니다.

- **Technical Details**: MixSeg3D는 강력한 포인트 클라우드 세분화 모델인 MinkUNet을 기반으로 하며, LaserMix와 PolarMix라는 두 가지 혁신적인 3D 데이터 혼합 전략을 적용합니다. MinkUNet은 효율적인 sparse convolution 연산과 계층적 특징 추출 능력으로 유명하며, 복잡한 LiDAR 데이터에서 유용한 피처를 학습할 수 있게 합니다. LaserMix는 포인트 클라우드를 경사 방향으로 결합하고, PolarMix는 방위각 방향으로 혼합하여 훈련 데이터의 다양성을 증가시킵니다.

- **Performance Highlights**: MixSeg3D는 실험을 통해 기존의 방법들보다 뛰어난 성능을 입증하였습니다. 2024 Waymo Open Dataset Challenge에서 평균 Intersection-over-Union (mIoU) 69.83%를 기록하며 높은 정확도를 보였습니다. 저자들은 TTA (Test Time Augmentation) 기법을 적용하여 평가 과정에서 예측 정확도를 더욱 향상시켰으며, 이는 자율주행에 있어 중요한 발전이 될 것입니다.



### Found in Translation: semantic approaches for enhancing AI interpretability in face verification (https://arxiv.org/abs/2501.05471)
- **What's New**: 본 논문은 컴퓨터 비전에서 특히 얼굴 인증(face verification) 분야의 머신 러닝 모델의 복잡성이 증가함에 따라, 해석 가능하고 투명한 인공지능을 위한 설명 가능한 인공지능(Explainable AI, XAI) 기술을 개발하는 데 중점을 두었습니다. 저자들은 이전 작업을 확장하여 인간의 인지 과정에서 파생된 의미론적 개념을 XAI 프레임워크에 통합하였습니다. 이를 통해 모델 출력과 인간 이해 간의 간극을 메우는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법론은 사용자 선택에 의한 얼굴 랜드마크로 정의된 의미론적 특성을 사용하여, 글로벌(global) 및 로컬(local) 설명을 결합하는 새로운 접근법입니다. 이 과정에서 대형 언어 모델(Large Language Models, LLMs)을 활용하여 유사성 지도(similarity maps)와 텍스트 설명을 생성합니다. 연구는 정량적인 실험과 사용자 피드백을 통해 방법론의 유효성을 검증하였으며, 향상된 해석 가능성을 확인하였습니다.

- **Performance Highlights**: 결과적으로 저자들의 의미론적 기반 접근법, 특히 가장 상세한 세트는 전통적인 방법에 비해 모델 결정의 더 섬세한 이해를 제공합니다. 사용자 연구에서는 전통적인 픽셀 기반 히트맵(pixel-based heatmaps)보다 저자들의 의미론적 설명에 대한 선호도가 강조되었으며, 이는 AI에서 인간 중심의 해석 가능성의 이점을 드러냅니다. 이 연구는 AI 모델의 행동을 인간의 인지 과정과 일치시키는 XAI 프레임워크 개발에 기여하여, 중요한 분야에서의 신뢰와 수용을 촉진하는 데 초점을 맞추고 있습니다.



### Efficiently serving large multimedia models using EPD Disaggregation (https://arxiv.org/abs/2501.05460)
Comments:
          13 pages, 6 figures

- **What's New**: 본 연구는 Encode-Prefill-Decode (EPD) Disaggregation이라는 새로운 프레임워크를 제안하여 대규모 멀티모달 모델 (LMM)에서 인코딩, 프리필, 디코딩 단계를 분리한다. 이 접근 방식은 메모리 병목 현상을 완화하고, 동기화 지연을 줄이며, 유연한 배치 처리를 지원한다. 또한, 멀티모달 토큰을 위한 새로운 캐싱 메커니즘을 도입하여 비동기 전송을 가능하게 하였다.

- **Technical Details**: EPD 분해는 각 단계를 고유한 리소스에 할당하여 독립적으로 최적화가 가능하도록 하며, 메모리 효율성을 크게 향상시킨다. 이를 통해 배치 크기 및 이미지 처리 수치가 증가하고, 최적 성능 메트릭을 달성하기 위한 통합 모듈도 포함한다. 연구 결과는 다양한 LMM 모델에서 메모리 사용량이 최대 15배 감소하고, 배치 크기가 최대 22배 증가하는 효과를 보여준다.

- **Performance Highlights**: 실험 결과, EPD 분해 방식이 기존 시스템에 비해 종합 성과를 개선하는 데 기여하여, 전체 처리량(E2ETP)이 최대 57% 향상되고, 첫 번째 토큰 수신까지의 지연(TTFT)이 최대 71% 감소했다. 이러한 결과는 적은 자원으로도 고성능의 멀티모달 추론이 가능함을 보여주며, EPD의 잠재력을 강조한다.



### Generative Modeling: A Review (https://arxiv.org/abs/2501.05458)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2305.14972

- **What's New**: 이번 논문에서는 생성적 방법(Gen-AI)을 통해 기계학습 및 베이지안 추론 문제를 해결하고자 합니다. 주로 높은 차원의 회귀 방법과 차원 축소 도구가 필요하며, 이는 딥 뉴럴 네트워크를 사용하여 지도 학습 문제를 해결하는 데 필수적입니다. Gen-AI의 주요 장점은 모델 무관성과 관심 있는 조건부 밀도 또는 사후 분위수(posterior quantiles)를 추정할 수 있는 능력입니다.

- **Technical Details**: 기계학습에서 중요한 작업은 고차원의 입력-출력 쌍을 사용하여 입력-출력 사전(look-up table)을 구성하는 것입니다. 일반적으로 높은 차원의 멀티변량 함수 f(x)를 훈련하는 것이며, 비모수 회귀(nonparametric regression)와 입력-출력 맵을 활용하여 신뢰할 수 있는 예측 규칙을 찾습니다. 마지막으로 베이지안 맵을 패턴 인식을 통해 대체하며, Deep Quantile NNs는 이러한 추론 의사결정 과정을 위한 일반적인 프레임워크를 제공합니다.

- **Performance Highlights**: 본 연구는 깊은 학습 모델을 활용하여 잠재 변수 Z에 대한 후방 분포를 전형적으로 다루고, 이를 통해 고차원적이고 비선형적인 복잡한 데이터를 효과적으로 처리할 수 있음을 보여줍니다. 팬데믹 상황에서의 에볼라 데이터 세트를 예시로 들며 모델의 효과성을 검증하였습니다. 향후 연구 방향을 제시하며, Gen-AI가 제공할 수 있는 다양한 가능성에 대해 논의하고 있습니다.



### The Jungle of Generative Drug Discovery: Traps, Treasures, and Ways Ou (https://arxiv.org/abs/2501.05457)
- **What's New**: 이 논문에서는 generative 모델이 제안한 de novo 디자인을 평가하는 방법에 대해 새로운 시각을 제공합니다. 기존의 평가 기준에서 간과된 주요한 문제점들을 드러내고, 이를 해결하기 위한 도구와 전략을 제시합니다. 또한, 분자 화학과 딥 러닝 분야 간의 연결을 강화하면서 대규모 데이터 분석을 통해 새로운 평가 기준을 도출하고자 합니다.

- **Technical Details**: 이 연구에서는 세 가지 심층 화학 언어 모델(Chemical Language Models, CLMs)을 사용하여 분자 생성 작업을 수행했습니다. 각각의 모델은 LSTM, GPT, S4로, 이들은 SMILES 문자열 형태로 분자를 생성하는 데 최적화되어 있습니다. 연구진은 1.5M의 SMILES 문자열로 사전 훈련한 후, 세 가지 주요 단백질에 대한 생물활성 분자로 미세 조정(fine-tuning)을 진행했습니다.

- **Performance Highlights**: 연구 결과, de novo 디자인의 라이브러리 크기가 평가 결과에 중요한 영향을 미친다는 사실이 밝혀졌습니다. 특히, Frechét ChemNet Distance (FCD)와 Frechét Descriptor Distance (FDD)는 문자열 크기가 증가함에 따라 변화하며, 라이브러리 크기가 작을 경우 잘못된 평가 결과를 초래할 수 있는 'size trap'이 존재한다고 강조합니다. 마지막으로, 모델의 가능도를 기반으로 한 검증된 라이브러리 선택 전략도 제시하였습니다.



### Atlas: A Novel Pathology Foundation Model by Mayo Clinic, Charit\'e, and Aignostics (https://arxiv.org/abs/2501.05409)
- **What's New**: 이 보고서에서는 RudolfV 접근법을 기반으로 한 새로운 비전 파운데이션 모델인 Atlas를 소개합니다. Atlas는 Mayo Clinic과 Charité - Universitätsmedizin Berlin에서 수집한 120만 개의 조직병리학 전체 슬라이드 영상으로 훈련되었으며, 다양한 공공 벤치마크 데이터셋에서 최고의 성능을 달성했습니다. 이 모델은 매개변수 수나 훈련 데이터셋 크기 면에서 가장 크지 않지만 여전히 뛰어난 성능을 보여줍니다.

- **Technical Details**: Atlas 모델은 ViT-H/14 아키텍처(632백만 매개변수)를 사용하여 훈련되며, 490,000건의 사례에서 추출된 34억 개의 이미지 타일로 훈련되었습니다. 데이터는 다양한 해상도에서 추출되었으며, 각기 다른 염색법과 다중 배율이 포함되어 있습니다. 이러한 다양성은 AI 학습의 일반화 및 강건성 향상에 기여합니다.

- **Performance Highlights**: Atlas는 21개의 공공 벤치마크 데이터셋에서 성능을 평가하여, 전통적인 모델에 비해 우수한 결과를 기록했습니다. 우리는 모델 성능 평가를 위해 선형 프로빙 프로토콜을 사용하였으며, 모든 모델의 추출된 임베딩을 비교했습니다. 결과적으로, Atlas는 다양한 다운스트림 병리학 작업에서 탁월한 성과를 달성하며, 실제 임상 환경에서도 활용될 수 있는 잠재력을 지니고 있습니다.



### Comprehensive Examination of Unrolled Networks for Solving Linear Inverse Problems (https://arxiv.org/abs/2501.04608)
Comments:
          27 pages, 10 figures. Project Page: this https URL

- **What's New**: 이번 논문은 여러 컴퓨터 비전 및 이미징 과제에서 발생하는 Unrolled Networks의 디자인 선택을 통합하고 최적화하는 방법을 제안합니다. 사용자들이 마주치는 여러 디자인 선택을 줄이는 것을 목표로 하며, 그 과정에서 발생하는 각 선택의 영향을 조명한 포괄적인 ablation study를 보고합니다. 이를 통해 연구자들이 자신의 응용 프로그램에 맞는 Unrolled Networks를 설계하는 데 도움이 되고 문제 진단을 효율적으로 수행할 수 있도록 돕고자 합니다.

- **Technical Details**: Unrolled Networks는 MRI, CT 스캔, 지진 이미징 등 다양한 이미징 응용 프로그램에서 선형 역 문제를 해결하기 위해 설계되었습니다. 이 네트워크는 이미지 복원을 위해 projection gradient descent(PGD) 알고리즘을 활용하며, neural networks를 통해 매 반복마다 프로젝션 단계를 실행합니다. PGD 알고리즘은 정확한 projection operator를 요구하지만, Unrolled Networks는 neural networks를 사용하여 보다 유연하게 학습할 수 있는 구조로 설계됩니다.

- **Performance Highlights**: 이 연구의 결과는 Unrolled Networks가 이미지 복원 성능을 개선할 수 있는 잠재력을 나타냅니다. 또한 기존 알고리즘보다 더 나은 실제 성능을 가능하게 하여 복잡한 네트워크 문제를 해결할 수 있습니다. 단순한 설계 변경 및 학습을 통해 다양한 적용 가능성과 효율성을 보여주며, 이러한 접근법은 향후 연구에서 널리 활용될 것으로 기대됩니다.




New uploads on arXiv(cs.CL)

### BoK: Introducing Bag-of-Keywords Loss for Interpretable Dialogue Response Generation (https://arxiv.org/abs/2501.10328)
Comments:
          Accepted at SIGDIAL 2024

- **What's New**: 이 논문에서는 기존의 Bag-of-Words (BoW) 손실 함수를 개선한 Bag-of-Keywords (BoK) 손실을 제안합니다. BoK 손실은 전반적인 응답 대신에 중심 아이디어를 포착하기 위해 키워드 예측에 중점을 둡니다. 이를 통해 대화 생성 과정에서 더욱 의미 있고 해석 가능한 응답을 생성할 수 있도록 합니다. BoK 손실은 일반적인 언어 모델(LM) 손실과 결합되어, Open-domain 대화 시스템의 효율성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: BoK 손실은 cross-entropy 손실로 정의되어, 다음 발화의 중요한 단어(키워드)를 예측합니다. 이 방법은 Unsupvised feature-based 키워드 추출기인 YAKE!를 사용하여 정답 응답에서 키워드를 추출합니다. BoK 손실은 인코더-디코더(T5)와 디코더 전용(DialoGPT) 아키텍처 모두에 통합되어, BoK와 LM 손실의 가중치 합을 최소화하는 방식으로 모델을 훈련합니다. 이러한 접근법은 훈련시킴으로써 데이터의 일반화 문제를 완화하고 응답의 구체성을 높이는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, BoK 손실을 포함함으로써 DailyDialog와 Persona-Chat 데이터셋에서 대화 생성 품질이 향상되었음을 확인했습니다. 또한, BoK 손실은 생성된 응답을 질적으로 분석할 때 포스트-호크 해석 가능성을 제공하는 것으로 나타났습니다. BoK-LM 손실은 참조 없이 평가 기준으로서도 효과적이며, 여러 대화 평가 데이터셋에서 인간의 판단과 유사한 성능을 보였습니다.



### Hierarchical Autoregressive Transformers: Combining Byte-~and Word-Level Processing for Robust, Adaptable Language Models (https://arxiv.org/abs/2501.10322)
- **What's New**: 이 논문에서는 텍스트를 처리하기 위한 새로운 Tokenization 방법을 제안합니다. 기존의 subword tokenizers의 한계를 극복하기 위해 계층적(혹은 hierarchical) 모델 아키텍처를 활용하여, 문자 수준(character-level)과 단어 수준(word-level) 처리를 결합하였습니다. 이는 고정된 vocabulary에 의존하지 않으면서도 단어 수준 Tokenization의 장점을 유지합니다. 이를 통해 다양한 언어와 도메인에 유연하게 대응할 수 있는 NLP 시스템을 구축할 수 있습니다.

- **Technical Details**: 제안된 아키텍처는 텍스트를 단어 단위로 분할한 후, 각 단어의 문자를 작은 문자 수준 인코더를 통해 단어 임베딩(word embedding)으로 변환합니다. 그 다음, 이들 단어 임베딩은 더 큰 백본(backbone) 모델에 의해 처리되고, 최종 출력은 소형 문자 수준 디코더를 통해 문자로 디코딩됩니다. 이러한 계층적 구조는 사전 학습된 모델이 새로운 도메인이나 언어의 텍스트에 적용될 때 발생하는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 최대 70억 개의 파라미터를 가진 계층적 트랜스포머(hierarchical transformers)가 기존의 subword tokenizer 기반 모델과 동등한 성능을 보이는 동시에 입력 변동성에 대한 강인성(robustness)이 크게 향상됨을 보여주었습니다. 또한, 새로운 언어에 대한 지속적 사전 학습(pretraining)中 모델이 약 두 배 더 빠르게 훈련되고 목표 언어에서 탁월한 성능을 발휘하며 이전에 획득한 지식을 더 많이 유지하는 것으로 나타났습니다.



### Natural Language Processing of Privacy Policies: A Survey (https://arxiv.org/abs/2501.10319)
Comments:
          27 pages

- **What's New**: 이번 연구는 자연어처리(NLP)와 개인 정보 보호 정책이 교차하는 영역에서 109개의 논문을 분석함으로써, NLP를 활용한 개인 정보 보호 정책의 향상을 위한 연구 방향을 제시하고 있습니다. 특히, 현재의 연구들이 주로 개인 정보 텍스트의 주석 및 분류에 중점을 두고 있다는 점을 발견하였으며, 요약화(summary)와 같은 다른 NLP 응용 분야의 연구 기회도 강조하고 있습니다. 향후 연구 방향으로는 코퍼스 생성(corpus generation), 요약 벡터(summarization vectors), 그리고 도메인 특화 모델 튜닝 등 다양한 요소를 제시합니다.

- **Technical Details**: 개인 정보 보호 정책은 데이터 수집, 사용, 관리 및 공개 관행을 설명하는 중요한 문서입니다. 그러나 이러한 정책들은 복잡해 사용자들이 이해하기 어려워 논란이 되고 있고, 정기적으로 수정되어 사용자들의 이전 이해 노력을 무의미하게 만들기도 합니다. 연구자들은 NLP의 다양한 기법을 적용하여 이러한 문제를 해결할 수 있는 방법을 모색하고 있으며, NLP 연구가 개인 정보 보호 정책의 효율적인 전달 및 투명성을 높이는 데 기여할 수 있다고 보고 있습니다.

- **Performance Highlights**: NLP를 활용한 연구는 개인 정보 보호 정책의 이해 능력을 높이고, 사용자에게 맞춤화된 정보를 제공하는 데 중요한 역할을 할 수 있습니다. 분석 결과, 개인 정보 텍스트의 분류 작업에 중점을 둔 논문들이 많았지만, 정책의 요약, 질문-응답 자동화(question-answering) 및 정보 검색(information retrieval) 등의 다른 NLP 응용에 대한 연구가 더 필요함을 확인했습니다. 이러한 연구들은 개인 정보 보호 분야에서 기존 방법론의 한계를 극복하고, 더욱 효과적인 정책 커뮤니케이션을 가능하게 할 수 있습니다.



### Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling (https://arxiv.org/abs/2501.10316)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전으로 대화형 에이전트에서 큰 성과가 나타났지만, 이들은 종종 사실과 맞지 않는 대답을 생성하는 환각(hallucination) 문제를 겪습니다. 사용자들은 LLM 기반 AI 에이전트에 과도하게 의존하는 경향이 있어 AI의 잘못된 제안도 수용하게 됩니다. 본 논문에서는 대화 상태 추적(Dialogue State Tracking, DST)과 관련된 불확실성과 오류에서 사용자 과신을 방지하기 위한 책임(accountability) 모델을 제안합니다.

- **Technical Details**: 책임 모델은 추가적인 책임 헤드(accountability head)를 갖춘 LLM으로, 대화 상태의 슬롯을 예측하는 이진 분류기 역할을 합니다. 이 모델은 기존의 언어 모델 손실(loss)과 보조 슬롯 분류 손실의 공동 학습을 통해 훈련됩니다. 이러한 구조는 DST의 불량 예측에서 발생할 수 있는 거짓 긍정(false positives)과 거짓 부정(false negatives)을 효과적으로 탐지하고, 예측된 대화 상태를 스스로 수정할 수 있는 기능을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 책임 모델이 적용된 LLM들은 MultiWOZ 데이터셋에서 약 3%의 절대 정확도 향상을 보였습니다. 또한 이 모델은 대화 상태의 자가 수정 기능을 통해 성능을 약 3% 더 향상시킬 수 있음을 보여주었습니다. 마지막으로, 책임 모델링은 사용자의 과도한 의존을 예방하는 데 효과적인 방법으로 제시되었습니다.



### Multi-stage Training of Bilingual Islamic LLM for Neural Passage Retrieva (https://arxiv.org/abs/2501.10175)
- **What's New**: 이번 연구는 이슬람 도메인에서의 자연어 처리(NLP) 기술 활용을 확인하고, 이슬람 신경 검색 모델 개발에 중점을 두었다. XLM-R 모델을 기반으로 한 언어 축소 기법을 통해 경량의 이중언어 대형 언어 모델을 설계하였다. 아랍어 기반의 방대한 도메인 데이터와 제한적인 다른 언어 데이터를 활용하여, 이슬람 특화 검색 성능을 향상시킬 방법을 모색하였다.

- **Technical Details**: 연구는 두 가지 언어(아랍어와 영어)에서 효과적인 검색 모델을 준비하기 위한 다양한 기술적 접근 방식을 사용하였다. 강력한 XLM-RBase 모델을 활용하여, 언어 축소 기술을 통해 성능 저하를 최소화하면서도 경량 모델을 구현하였다. 또한, 대규모 일반 도메인 데이터셋과 소규모 이슬람 도메인 데이터셋을 활용하여 다단계 훈련 프로세스를 통해 검색 모델을 개선하였다.

- **Performance Highlights**: 제안된 이중언어 이슬람 신경 검색 모델은 단일 언어 모델보다 향상된 성능을 보여주었다. 실험 결과, 데이터 증강 기법을 활용하여 영어 기반의 도메인 검색 데이터셋을 보강함으로써 검색 성능이 개선되었음을 강조한다. 이 연구는 이중언어 모델의 성능과 도메인 적응을 결합하여 이슬람 문헌에 대한 접근성을 높이는 데 기여할 것으로 보인다.



### Dual Debiasing: Remove Stereotypes and Keep Factual Gender for Fair Language Modeling and Translation (https://arxiv.org/abs/2501.10150)
- **What's New**: 이번 연구에서는 성별 편향을 줄이면서 사실적인 성별 정보를 보존하는 새로운 방법인 Dual Debiasing Algorithm through Model Adaptation (2DAMA)를 도입합니다. 기존의 연구에서는 성별 대표성을 식별하고 사회적 편향을 완화하는 데 집중했으나, 2DAMA는 언어 모델의 성별 정보를 공평하게 표현하는 것을 목표로 합니다. 이 방법은 특히 번역에서 고전적인 편향 경향을 완화하는 데 기여하며, 언어 처리 작업에서 유용한 사실적 성별 단서를 보존할 수 있습니다.

- **Technical Details**: 2DAMA는 사전 훈련된 언어 모델에서 편향 신호를 효과적으로 제거할 수 있는 알고리즘적 구성 요소를 포함합니다. 핵심 아이디어는 편향을 줄이면서도 모델이 사실적인 성별 정보를 유지하도록 하는 것입니다. 이 방법은 기존의 DAMA와 LEACE 알고리즘을 결합하여 모델 성능을 손상시키지 않고 해로운 편향의 인코딩을 완화합니다.

- **Performance Highlights**: 실험 결과, 2DAMA는 LLM의 성별 편향을 효과적으로 줄이는 데 성공했으며, 이를 통해 고전적인 성별 고정관념이 언어 모델 훈련 중에 어떻게 증폭되는지를 보여줍니다. 또한 2DAMA는 네 개의 다양한 모델에서 성별 표현의 패턴을 분석하여 각기 다른 디자인 선택이 편향 제거에 미치는 영향을 탐구합니다. 이 연구의 결과는 LLM의 다국적 성별 편향을 완화하는 데 기여할 것으로 기대됩니다.



### ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario (https://arxiv.org/abs/2501.10132)
- **What's New**: 이 논문에서는 실시간 API를 통해 대규모 언어 모델(LLM)의 기능 호출 능력을 평가하는 ComplexFuncBench라는 새로운 벤치마크를 소개합니다. 기존의 벤치마크들이 염두에 두지 않았던 복잡한 기능 호출을 다루며 5개의 실제 사용 시나리오를 포함하고 있습니다. 이를 통해 연구자들이 LLM의 기능 호출 능력을 정량적으로 평가할 수 있는 방법을 제공합니다.

- **Technical Details**: ComplexFuncBench는 128k의 긴 컨텍스트에서 멀티 스텝과 제약이 있는 함수 호출을 요구하는 1,000개의 복잡한 함수 호출 샘플을 포함합니다. 각 샘플은 입력 쿼리 및 해당하는 함수 호출 경로로 구성되며, 여러 단계와 사용자 제공 제약 조건을 포함합니다. 자동 평가 프레임워크인 ComplexEval은 전통적인 정확한 일치 방법의 한계를 극복하고 다차원 매칭 접근 방식을 사용하여 기능 호출을 비교합니다.

- **Performance Highlights**: 실험 결과, 많은 LLM이 함수 호출에서 파라미터 값 오류로 인한 상당한 비율의 오류를 포함하고 있으며, 각 모델은 특정 시나리오에서 고유한 약점을 보입니다. 논문에서 제안하는 ComplexFuncBench와 ComplexEval은 LLM의 기능 호출 능력을 최적화하는 데 중요한 기초 자료가 될 것입니다. 연구진은 이러한 결과가 향후 연구 방향을 제시할 것으로 기대하고 있습니다.



### BBPOS: BERT-based Part-of-Speech Tagging for Uzbek (https://arxiv.org/abs/2501.10107)
- **What's New**: 본 논문은 저자들에 의해 이전에 시험받지 않은 단일 언어 우즈벡 BERT 모델을 부분 품사 태깅(task) 작업에 적용하여 저자들이 처음으로 공개한 우즈벡어 UPOS 태깅 벤치마크 데이터셋을 소개합니다. 이 모델의 평균 정확도는 91%에 달하며, 다국어 mBERT 및 규칙 기반 태거보다 우수한 성능을 보입니다.

- **Technical Details**: 우즈벡어는 역사적 이유로 인해 라틴 문자와 키릴 문자를 혼합하여 사용하는 형태로, 다양한 크기와 품질의 사전 학습된 BERT 기반 모델들이 존재하지만, 공개된 벤치마크 데이터셋이 부재하여 평가가 미비했습니다. 저자들은 POS 태깅을 위해 500개의 문장을 포함한 새로운 데이터셋을 제작하였으며, 이 데이터셋은 UPOS 태그를 사용하여 수동으로 주석이 달렸습니다.

- **Performance Highlights**: 실험 결과, 단일 언어 BERT 모델들은 평균적으로 90% 이상의 정확도 및 84% 이상의 F1 점수를 기록했습니다. 특히, TahrirchiBERT보다 데이터 양이 10배 적게 학습된 UzBERT가 두 메트릭에서 약간 우수한 성능을 보였으며, 이는 사전 학습 데이터의 품질 차이에 기인할 수 있습니다.



### Author-Specific Linguistic Patterns Unveiled: A Deep Learning Study on Word Class Distributions (https://arxiv.org/abs/2501.10072)
- **What's New**: 이번 연구는 문헌 저자별 언어 패턴을 분석하는 데 깊이 있는 신경망 모델을 활용한 것을 특징으로 합니다. POS 태그와 이중어(bigram) 분석을 결합하여 각 저자의 독특한 스타일을 분류하는 방법을 제안합니다. 바닥 명사(unigram) 특성보다 더 나은 성능을 가진 이중어 기반 모델의 효용성을 보여주는 결과를 도출했습니다. 이러한 발견은 저자 프로파일링 및 문학 연구에서의 딥 러닝의 가능성을 강조합니다.

- **Technical Details**: 연구에서는 Python 3.6을 사용하여 모든 데이터 평가 및 머신 러닝을 수행하였으며, POS 태깅은 spaCy의 독일어 모델을 사용했습니다. 수학적 작업은 Numpy를 통해, MDS 투영은 scikit-learn을 사용하여 수행되었습니다. 두 가지 신경망 아키텍처인 완전 연결형 신경망과 합성곱 신경망(CNN)을 구현하여 POS 태그 벡터와 이중어 빈도 행렬에서 저자를 분류하였습니다.

- **Performance Highlights**: MDS 시각화를 통해 각 저자의 작품에서의 명백한 군집 경향이 드러났으며, 이는 저자의 언어적 스타일을 규명하는 데 중요한 증거가 됩니다. 이 연구의 결과, 저자 간의 고유한 언어적 특성을 기반으로 한 강력한 분류 성능을 확보하였고, 이는 향후 문헌 연구의 발전에 기여할 것입니다. 또한, 이중어 기반 접근 방식이 저자의 스타일을 포착하는 데 더욱 효과적이라는 사실을 강조합니다.



### MSTS: A Multimodal Safety Test Suite for Vision-Language Models (https://arxiv.org/abs/2501.10057)
Comments:
          under review

- **What's New**: 본 논문에서는 이미지와 텍스트 입력을 처리하는 비전-언어 모델(Vision-language models, VLMs)의 안전성과 관련된 새로운 필수 연구를 다룹니다. 저자들은 MSTS(Multimodal Safety Test Suite)를 소개하여 40개의 세부 위험 카테고리에 걸쳐 400개의 테스트 프롬프트를 포함합니다. 이 테스트는 텍스트와 이미지의 조합을 통해서만 완전한 위험 의미를 드러내어, 비정상적으로 안전한 VLM을 식별할 수 있습니다.

- **Technical Details**: MSTS는 다양한 언어로 번역되어 비영어 프롬프트를 통해 안전한 모델 응답의 비율을 증가시키는 방법을 제시합니다. 테스트 프롬프트의 구성은 각기 다른 위험 요소를 평가할 수 있도록 설계되어 있으며, 다중 모달(Multimodal) 프롬프트의 사용 시 모델의 안전성을 낮출 수 있다는 점도 확인하였습니다. 이 연구에서는 VLM의 안전성 평가를 자동화하는 가능성도 탐색하였고, 최고의 안전 분류기조차 불완전하다는 결과를 도출했습니다.

- **Performance Highlights**: 여러 오픈 VLM에서 명확한 안전 문제를 발견하였으며, 몇몇 VLM은 특정 테스트 프롬프트를 이해하지 못함으로써 우연히 안전하게 평가되었습니다. MSTS를 통해 모델의 안전성을 분석한 결과, 텍스트만으로 테스트할 때 VLM은 다중 모달 프롬프트일 때보다 더 안전하다는 사실이 드러났습니다. 이러한 발견은 VLM의 안전성을 효율적으로 평가하기 위한 새로운 접근역을 제공함으로써, AI 애플리케이션의 신뢰성을 높이는 데 기여할 것입니다.



### Automatic Speech Recognition for Sanskrit with Transfer Learning (https://arxiv.org/abs/2501.10024)
Comments:
          Paper has been accepted at the 4th International Conference on Computer, Communication, Control & Information Technology (C3IT), Hooghly, India, 2024, pp. 1-5

- **What's New**: 산스크리트어(Sanskrit)는 인류의 가장 오래된 언어 중 하나로, 방대한 양의 문헌을 가지고 있습니다. 그러나 디지털 콘텐츠가 매우 제한적이기 때문에 AI 시스템의 훈련에 필요한 자료가 부족합니다. 본 연구에서는 OpenAI의 Whisper 모델을 활용하여 전이 학습(transfer learning) 기법을 통해 산스크리트어를 위한 자동 음성 인식(Automatic Speech Recognition, ASR) 모델을 개발했습니다.

- **Technical Details**: 이 모델은 하이퍼파라미터 최적화를 거쳐 Vaksancayah 데이터셋에서 15.42%의 단어 오류율(Word Error Rate, WER)을 달성했습니다. 전통적인 통계적 접근이 산스크리트어의 복잡한 언어 구조에 한계를 가져오기에, 본 연구에서는 변환기 아키텍처(transformer architecture)를 적용하였습니다. 이를 통해 언어적 복잡성을 더 효과적으로 처리할 수 있었습니다.

- **Performance Highlights**: 개발된 ASR 모델은 텍스트 전사 및 발음 연습 등에 활용될 수 있으며, 온라인 데모를 통해 공공에게 공개하여 성능 평가에 기여하고 있습니다. 이 모델은 산스크리트어 학습의 접근성을 높이고, 현대 기술 지원을 통해 산스크리트어 교육을 개선하는 데 중요한 발판이 될 것입니다.



### Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models (https://arxiv.org/abs/2501.09997)
- **What's New**: 이 연구에서는 기존의 LLM에서 발생하는 환각(hallucination) 문제를 해결하기 위해 새로운 Attention-Guided SElf-Reflection (AGSER) 방법을 제안합니다. AGSER 방법은 입력 쿼리를 주의 깊은(attentive) 쿼리와 비주몰한(non-attentive) 쿼리로 분리하여 처리합니다. 이 방식을 통해 환각 탐지의 효율성을 높이고, 계산 복잡성을 줄이는 효과도 있습니다.

- **Technical Details**: AGSER는 LLM의 주의(attention) 기여도를 활용하여 입력 쿼리를 처리합니다. 각 쿼리는 LLM을 통해 별도로 처리되며, 생성된 응답과 원래 응답 간의 일관성 점수를 계산합니다. 두 개의 일관성 점수 차이를 계산하여 환각의 정도를 추정합니다. AGSER는 3번의 LLM 실행만으로 결과를 도출할 수 있으며, 이는 기존의 접근 방식과 비교했을 때 계산 비용을 크게 줄이는 것입니다.

- **Performance Highlights**: 실험 결과, AGSER는 4개의 잘 알려진 LLM에 대해 다양한 환각 벤치마크에서 기존 방법보다 뛰어난 성능을 보여줍니다. 연구자들은 AGSER 접근법이 환각 탐지에서 최첨단 성과를 달성했다고 보고했습니다. 이를 통해 LLM의 신뢰성을 높이는 데 기여할 것으로 기대하고 있습니다.



### Agent-as-Judge for Factual Summarization of Long Narratives (https://arxiv.org/abs/2501.09993)
- **What's New**: 이번 연구에서는 'NarrativeFactScore'라는 새로운 평가 프레임워크를 도입했습니다. 이는 LLM의 요약 품질을 평가하고 개선하기 위한 'Agent-as-a-Judge' 접근 방식을 활용합니다. 특히, 이 방법은 기존의 ROUGE 및 BERTScore 같은 평가 기준의 한계를 극복하고자 합니다.

- **Technical Details**: NarrativeFactScore는 입력 및 생성된 요약에서 추출한 Character Knowledge Graph (CKG)를 활용하여 요약의 사실 일관성을 평가합니다. 이 프레임워크는 특히 긴 서사에서의 사실 정확성을 중점적으로 다루며, 잘못된 사실이나 누락된 정보를 식별하는 데 도움을 줍니다.

- **Performance Highlights**: 실험을 통해 NarrativeFactScore는 주요 벤치마크에서 경쟁력 있는 방법들보다 우수한 성능을 보였습니다. 연구 결과는 에이전트 기반 평가 시스템이 LLM이 생성한 요약의 사실 신뢰성을 개선하는 데 큰 잠재력을 가지고 있음을 강조합니다.



### A Survey on Multi-Turn Interaction Capabilities of Large Language Models (https://arxiv.org/abs/2501.09959)
Comments:
          Draft Version, 14 pages, Ongoing refinement over time

- **What's New**: 이번 연구는 다중 턴(multi-turn) 인터랙션에서 LLM의 능력에 대한 집중 리뷰를 제공합니다. LLMs는 고객 서비스나 건강 관리 분야에서 다각적인 사용자 쿼리를 처리하기 위해 점점 더 많이 사용되고 있으며, 이러한 상호작용에서 연속적인 대화 관리는 필수적입니다. 특히 LLM이 사용자와 환경 모두와의 상호작용을 통해 다양한 작업을 수행하는 동적인 에이전트로 기능하는 것을 중점적으로 다룹니다.

- **Technical Details**: 다중 턴 인터랙션에서 LLM의 평가 방법 및 알고리즘이 중요한 요소로 언급되고 있습니다. 현재 MT-Bench와 Chatbot Arena와 같은 여러 평가 도구가 사용되고 있으며, 특히 'LLM-as-a-Judge' 프레임워크가 도입되어 LLM의 성능을 파라미터화하고 있습니다. 이 평가들은 다양한 모델 특정 기능 및 다중 턴 대화의 질을 측정하는 데 중점을 둡니다.

- **Performance Highlights**: MT-Bench-101과 같은 평가 기준은 다중 턴 상호작용에 대한 가장 포괄적인 기준을 제공하며, 인간 선호도의 중심 요소인 사용자 명령 수행 능력을 평가합니다. 최근의 연구에서는 LLM의 대화 품질, 즉 일관성 및 인간 유사성을 평가하기 위한 다양한 방법이 개발되고 있습니다. 이러한 평가들은 LLM의 다중 턴 대화에서의 성능을 개선하고, 향후 연구 방향에도 중요한 통찰력을 제공합니다.



### FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs (https://arxiv.org/abs/2501.09957)
- **What's New**: 이 논문은 갱신된 KG-RAG 프레임워크인 FRAG를 제안하여, LLM의 유연성과 효율성을 향상시키는 방법을 다루고 있습니다. FRAG는 질의의 복잡성을 기반으로 하는 적응형 검색 전략을 활용하여 효율적인 추론 경로를 제공합니다. 또한, KG 정보를 활용함으로써 정보 검색의 질을 개선하면서도 LLM의 추가적인 미세 조정 없이도 높은 성능을 유지하는 것을 목표로 합니다.

- **Technical Details**: FRAG 프레임워크는 두 가지 주요 모듈로 구성되며, 'Reasoning-aware'와 'Flexible-retrieval'입니다. 'Reasoning-aware' 모듈은 질의 컨텍스트를 바탕으로 추론의 복잡성을 간단하거나 복잡한 것으로 분류하여, 경로의 홉 수를 예측합니다. 'Flexible-retrieval' 모듈은 전통적인 알고리즘에 기반하여 검색 과정을 '전처리, 검색 및 후처리'의 파이프라인으로 세분화하여 맞춤형 검색이 가능하도록 합니다.

- **Performance Highlights**: 실험 결과, FRAG는 최고 성능을 기록하며, 리소스 소모가 적은 효율성을 보여주었습니다. 특히, FRAG는 간단한 추론 과제에서는 Breadth-First Search(BFS) 및 랭킹 전략을 사용하고, 복잡한 과제에서는 최단 경로 검색 전략을 통해 효과적인 검색을 가능하게 합니다. 이를 통해 LLM의 추론 능력이 한층 향상되었습니다.



### Indigenous Languages Spoken in Argentina: A Survey of NLP and Speech Resources (https://arxiv.org/abs/2501.09943)
Comments:
          Accepted to COLING Main 2025

- **What's New**: 본 연구에서는 아르헨티나에서 사용되는 원주율 언어들의 체계적 분류와 해당 언어를 사용하는 인구에 대한 국가 인구 통계 데이터를 제시합니다. 원주율 언어는 Mapuche, Tupí-Guaraní, Guaycurú, Quechua, Mataco-Mataguaya, Aymara, Chon의 7개 가족으로 분류됩니다. 또한, 이 언어들을 위해 개발된 컴퓨터 자원에 대한 기초 조사도 제공하여, 아르헨티나 원주율 언어의 현재 상황을 이해하는 데 도움을 줄 것입니다.

- **Technical Details**: 아르헨티나의 원주율 언어들에 대한 합의된 목록이 없으며, 이는 언어의 지속 가능성, 문서화, 표준화, 사용 맥락 등에 관련된 복잡한 문제에서 비롯됩니다. 연구에 따르면, 원주율 언어 사용자는 자신의 언어 유산에 대해 부정적인 태도를 보이는 경우가 많습니다. 이 논문에서는 원주율 언어가 가진 특징과 인구 통계적 데이터를 종합하여 제시합니다.

- **Performance Highlights**: 이 논문은 아르헨티나에서 원주율 언어의 사용 현황을 조사하고, 해당 언어들에 대한 주요 연구 동향을 분석합니다. 원주율 언어의 다양성을 포함한 개요와, 이들을 위한 컴퓨터 자원의 조사를 통해 연구자들에게 유용한 자료를 제공할 것입니다. 이 작업은 새로운 연구 그룹들이 해당 분야에서 핵심 주제와 도구를 신속히 파악하는 데 도움이 될 것으로 기대됩니다.



### Passage Segmentation of Documents for Extractive Question Answering (https://arxiv.org/abs/2501.09940)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 모델의 성능 향상에 있어 중요한 역할을 하는 chunking(청킹) 과정에 주목합니다. 저자들은 새로운 Logits-Guided Multi-Granular Chunker (LGMGC) 프레임워크를 소개하며, 이는 긴 문서를 맥락적으로 독립적인 다양한 크기의 chunk로 분할합니다. 실험 결과, LGMGC는 기존의 chunking 방법보다 우수한 성능을 나타낸다고 보고합니다.

- **Technical Details**: LGMGC는 두 가지 청킹 모듈, Logits-Guided Chunker와 Multi-Granular Chunker를 통합하여 구성됩니다. 이 과정은 문서를 의미적으로 일관된 단위로 세분화하며, 각 단위는 후속 질문의 유형에 따라 다양한 크기로 더 나누어집니다. 구체적으로, LLM의 로짓 정보(logits information)를 활용하여 청킹 단위를 정의하고, 이를 기반으로 세분화된 child chunk를 생성합니다.

- **Performance Highlights**: LGMGC의 성능은 두 개의 벤치마크 데이터세트에서 평가되었습니다. 그 결과, 제안된 청킹 접근 방식은 정보 검색 및 질문 응답 작업에서 현재 사용되는 청킹 방법들보다 더 나은 성능을 보였습니다. 이는 청킹 과정에서의 맥락적 정보의 중요성을 강조하며, 문서의 세분화가 질의 응답의 정확성에 긍정적인 영향을 미칠 수 있음을 보여줍니다.



### Dialogue Benchmark Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented LLMs (https://arxiv.org/abs/2501.09928)
Comments:
          The paper is publsihed in SIGMOD 2025

- **What's New**: 이 논문에서는 Chatty-Gen이라는 새로운 다단계 Retrieval-Augmented Generation (RAG) 플랫폼을 소개합니다. Chatty-Gen은 지식 그래프(KG)를 사용하여 특정 도메인에 맞춘 고품질 대화 벤치마크(dialogue benchmark)를 자동으로 생성하는 혁신적인 시스템입니다. 이 플랫폼은 생성 과정의 복잡한 단계를 관리 가능하도록 쪼개고, 단계 간의 자동 검증을 위해 assertion rules를 사용합니다. 이를 통해 대화의 일관성을 유지하고, 시간 소모가 큰 재시작을 방지합니다.

- **Technical Details**: Chatty-Gen은 KG에서 샘플링된 대표 엔티티를 기반으로 대화의 맥락(context)을 확보합니다. 생성 과정은 처음에 특정 엔티티를 샘플링한 후, 관련 있는 서브그래프(subgraph)를 추출하여 대화 맥락을 유지합니다. 텍스트 환경을 인간이 읽을 수 있는 형태로 변환하고, 대화의 흐름을 유지하는 주요 질문을 통해답변 생성을 위해 SPARQL 쿼리를 자동으로 생성합니다. 이 시스템은 KG에 종속되지 않는 설계를 갖추고 있어 다양한 도메인 및 KG에서 사용될 수 있습니다.

- **Performance Highlights**: Chatty-Gen은 DBpedia, YAGO, DBLP와 같은 여러 실제 KG에서 성능 평가를 진행했으며, 다양한 상업적 및 오픈소스 LLM과의 호환성을 입증했습니다. 특히, Chatty-Gen은 Maestro와 비교하여 대화 벤치마크 생성에서 99%의 시간 효율성을 보여주었으며, 30시간이 소요되던 작업을 불과 10분으로 단축시켰습니다. 이러한 결과는 Chatty-Gen이 고품질 KG 기반 대화 벤치마크 생성을 위한 비용 효율적이고 다재다능한 솔루션임을 입증합니다.



### Bridging Language Barriers in Healthcare: A Study on Arabic LLMs (https://arxiv.org/abs/2501.09825)
- **What's New**: 본 논문은 다국어 이해와 의료 지식을 모두 갖춘 대형 언어 모델(LLM) 개발의 도전을 조사합니다. 단순히 의료 데이터를 번역하는 것으로는 목표 언어에서의 임상 작업에서 강력한 성능을 보장할 수 없음을 보여줍니다. 실험 결과, 훈련 데이터의 최적 언어 혼합이 다양한 의료 작업에 따라 상이하게 나타났습니다. 우리는 신중하게 조정된 언어 비율을 가진 대형 모델이 모국어 임상 작업에서 우수한 성능을 달성한다는 사실을 발견했습니다.

- **Technical Details**: 본 연구에서는 아랍어 의료 작업에 대한 LLM의 성능을 평가하기 위해 Llama3.1 모델을 중점적으로 다룹니다. 기존 LLM의 번역, 패러프레이징 및 합성 데이터 생성 기법을 활용하여 아랍어 의료 데이터셋을 보강하는 방법을 탐구합니다. 우리는 다양한 원본 및 합성 아랍어 의료 데이터의 혼합을 사용하여 Llama 3.1을 세부 조정(fine-tuning)하였으며, 이는 다양한 임상 작업에서 모델 성능에 미치는 영향을 분석하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 모든 모델 계열에서 대형 언어 모델이 아랍어 의료 기준에서 제한된 성능을 보인 것으로 나타났습니다. Llama3.1은 영어에서는 높은 정확도를 기록했지만 아랍어로는 큰 성능 저하가 있었습니다. 반면 Qwen2.5 모델은 아랍어 작업에 대해 상대적으로 더 나은 성능을 보였지만 여전히 최적 수준에 미치지 못했습니다. 향후 연구는 아랍어 성능 향상 및 영어 능력 달성을 위한 전략을 탐구하는 데 집중할 예정입니다.



### Qwen it detect machine-generated text? (https://arxiv.org/abs/2501.09813)
- **What's New**: 이번 논문에서는 Unibuc - NLP 팀이 Coling 2025 GenAI 워크숍의 첫 번째 작업인 바이너리 다국어 기계 생성 텍스트 감지 과제를 어떻게 해결했는지를 설명하고 있습니다. 마스킹된 언어 모델과 원인 모델을 모두 탐색하였으며, 부작업 A에서 F1 Micro 점수 0.8333으로 36개 팀 중 1위를 차지했습니다. 또한, F1 Macro 점수에서는 0.8301로 2위를 기록하며 우수한 성능을 입증했습니다.

- **Technical Details**: 이 문서에서 다룬 시스템은 부작업 1을 위한 원인 모델과 부작업 2를 위한 마스킹된 모델의 두 가지 아키텍처로 구성되었습니다. 다양한 큰 언어 모델(LLMs)을 사용하여 최적의 성능을 내는 모델을 찾기 위한 실험을 수행하였으며, Qwen2.5-0.5B 모델이 최고의 성능을 보였습니다. 저자들은 모델의 과적합(over-fitting) 문제를 해결하기 위해 데이터셋을 조정해 균형 잡힌 분포를 달성했습니다.

- **Performance Highlights**: Monolingual Subtask에서 우리의 모델은 F1 Micro 점수로 1위를 기록했으며, F1 Macro 점수에서는 2위를 차지했습니다. F1 Micro 점수는 0.8333, F1 Macro 점수는 0.8301로, 모델의 신뢰성과 정확성을 증명했습니다. 그러나 Multilingual Track에서는 상대적으로 성과가 저조하여 개선이 필요함을 인지했습니다.



### Sentiment Analysis in Twitter Social Network Centered on Cryptocurrencies Using Machine Learning (https://arxiv.org/abs/2501.09777)
Comments:
          6 pages and 5 figures

- **What's New**: 이번 논문에서는 이란 트위터 사용자들의 암호화폐에 대한 의견을 분석하기 위해 Persian 트윗을 사용한 감정 분석 모델을 개발하고 있습니다. 기존의 연구는 대부분 영어 트윗을 분석한 반면, 본 연구는 페르시아어 사용자를 포함하여 더 넓은 통계적 집단을 고려하고 있습니다. 또한, 자연어 처리 기술 및 기계 학습 방법을 활용해 감정 분류 모델링의 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 이 연구는 트위터에서 수집한 4,000개의 페르시아어 트윗에 대해 감정 분석을 수행했습니다. 자연어 처리(NLP) 기술을 이용한 데이터 전처리 과정으로는 BOW(Bag of Words), FastText 및 BERT(Bidirectional Encoder Representations from Transformers) 방법을 활용하여 대표적인 감정 모델을 만들었습니다. 최종적으로 BERT 모델이 83.50%의 정확도로 트윗 감정 분류에서 가장 우수한 성능을 보였습니다.

- **Performance Highlights**: 논문에서 제안한 감정 분석 모델은 페르시아어 트윗에 대해 긍정적, 부정적 및 중립적 감정을 정확하게 분류하는 능력을 갖추고 있습니다. 이를 통해 경제 분야의 관리자와 관계자들이 대중의 의견을 이해하고 암호화폐 관련 정보에 기반하여 정책을 수립하는 데 도움을 줄 수 있습니다. 또한, 적은 비용과 짧은 시간 내에 사람들의 암호화폐에 대한 감정적인 흐름을 파악할 수 있는 유용한 도구가 될 것으로 기대됩니다.



### Multiple Choice Questions: Reasoning Makes Large Language Models (LLMs) More Self-Confident Even When They Are Wrong (https://arxiv.org/abs/2501.09775)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 답변에 대한 자신감이 모델이 직접 답변할 때와 이유를 먼저 제공한 후에 답변할 때 어떻게 달라지는지를 연구합니다. 모델들이 이유를 제시할 때 더 높은 자신감을 보이며, 이는 정답이든 오답이든 관계없이 관찰됩니다. 이러한 결과는 LLM의 동작 방식뿐만 아니라 인간의 인지 구조와도 연결되어 설명됩니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식을 사용하여 LLM에 질문을 제시합니다. 첫 번째 접근은 직접 답변을 요구하는 방법이고, 두 번째 접근은 문제 해결을 위한 단계적 사고(Chain of Thought, CoT)를 요구하는 것입니다. 이 연구는 57개의 카테고리와 15,000개 이상의 질문을 포함하는 MMLU(Massive Multitask Language Understanding) 벤치마크를 사용하여 여러 LLM 모델을 평가합니다.

- **Performance Highlights**: 모델들은 대개 CoT 방식으로 질문에 대답할 때 정답을 선택할 때의 확률이 증가하는 경향을 보입니다. 또한, 이유를 제공한 경우 모든 모델에서 선택한 옵션에 대한 자신감이 높아지는 게 관찰되었습니다. 이 결과는 모델의 정확성을 향상시키고, 선택한 옵션에 대한 확신을 증가시켜, LLM의 응답 평가 시 이러한 확률 추정치를 적용하는 데 주의를 기울여야 함을 시사합니다.



### Can Large Language Models Predict the Outcome of Judicial Decisions? (https://arxiv.org/abs/2501.09768)
- **What's New**: 이 논문은 아랍어 법정 판단 예측(Arabic Legal Judgment Prediction, LJP)에 대한 연구를 진행하며, 사우디 상업 법원의 판결 데이터를 수집하여 새로운 LJP 데이터셋을 개발했습니다. 특히, 기존의 고급 모델이 아랍어에서 낮은 성능을 보이는 문제를 해결하기 위해 아랍어 특화 모델을 벤치마킹했습니다. 또한 데이터셋, 구현 코드 및 모델을 공개하여 향후 연구의 기반을 마련했습니다.

- **Technical Details**: 연구에서는 LLaMA-3.2-3B 및 LLaMA-3.1-8B와 같은 최첨단 오픈 소스 LLM 모델을 다양한 설정(Zero-shot, One-shot, Fine-tuning)에서 평가했습니다. 이 과정에서 BLEU와 ROUGE와 같은 양적 평가 지표와 Coherence, 법적 언어, 명확성과 같은 질적 평가를 결합한 포괄적인 평가 프레임워크를 사용했습니다. QLoRA를 활용한 파인튜닝을 통해 작은 모델들이 특정 작업에서 더 큰 모델과 유사한 성능을 내도록 하는 방법을 찾았습니다.

- **Performance Highlights**: 결과적으로 파인튜닝된 소형 모델이 작업 특화 컨텍스트에서 대형 모델과 유사한 성능을 달성하는 반면, 자원 효율성은 더 크게 개선되었습니다. 프롬프트 엔지니어링과 파인튜닝이 모델 출력에 미치는 영향도 조사하여 성능 변동성과 지시 민감성에 대한 통찰력을 제공했습니다. 이 연구는 아랍어 법적 NLP의 발전을 위한 강력한 평가 프레임워크를 제시하여 LJP 연구를 촉진할 것으로 기대됩니다.



### LeMo: Enabling LEss Token Involvement for MOre Context Fine-tuning (https://arxiv.org/abs/2501.09767)
- **What's New**: 이번 논문에서는 LeMo라는 새로운 LLM(대형 언어 모델) 미세 조정 시스템을 제안합니다. LeMo는 긴 문맥 상황에서 고유한 토큰 수준의 희소성 메커니즘인 Contextual Token Sparsity를 활용하여 메모리와 계산 효율성을 최적화합니다. 기존의 미세 조정 방법들은 활성화 메모리 문제를 해결하지 못했으나, LeMo는 정보 기반 토큰 제거 및 패턴 예측 등을 통해 이러한 문제를 극복합니다.

- **Technical Details**: LeMo는 세 가지 주요 기술로 구성됩니다: (1) Token Elimination은 동적으로 중복 토큰을 식별하여 계산에서 제외합니다. (2) Pattern Prediction은 훈련된 예측기를 활용하여 희소성 패턴을 추정합니다. (3) Kernel Optimization은 토큰 선택 및 패딩 동안 불필요한 메모리 이동을 제거하고, 세그먼트 기반의 그래디언트 계산 방법으로 활성화 메모리의 피크를 줄이는 방식을 채택합니다.

- **Performance Highlights**: LeMo는 다양한 LLM 아키텍처에 호환되는 엔드 투 엔드 미세 조정 시스템으로 구현되었습니다. 평가 결과, LeMo는 메모리 사용량을 최대 1.93배 줄이고, 속도를 최대 1.36배 향상시켜 최신 미세 조정 시스템보다 뛰어난 성능을 보였습니다. 이를 통해 긴 문맥 시퀀스를 처리하면서도 적은 자원을 요구하는 혁신적인 접근법을 제시하고 있습니다.



### Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning (https://arxiv.org/abs/2501.09766)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 도구 사용 능력을 향상시키기 위해 외부 도구와 결합하는 새로운 방법론을 제시합니다. 복잡한 상황에서 모델의 부족한 성능(Deficiency)을 극복하기 위해 반복적인 강화 세부 조율 전략(iTool)을 도입하며, 이를 통해 모델이 복잡한 도구 사용 시나리오에서도 효율적으로 학습할 수 있도록 합니다. 또한, 고난이도 데이터에서 학습할 수 있도록 더 쉬운 난이도에서 시작하는 Warm-up SFT 전략을 사용합니다.

- **Technical Details**: LLMs가 도구를 사용하는 과정은 사용자 질문에 답하기 위해 적절한 함수 선택 및 도구 호출을 수행하는 것입니다. 논문에서는 정책 모델의 피드백을 기반으로 부족 데이터를 식별하고, 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 통해 세밀한 선호 쌍을 수집합니다. 이를 통해 모델을 Ground Truth에 맞도록 조정하고 부족한 부분을 misalign하게 만드는 정책을 업데이트하여 효과적으로 부조화를 해소합니다.

- **Performance Highlights**: iTool은 개수로만 7억 개의 매개변수를 가진 모델임에도 불구하고, 많은 대규모 공개 모델보다 뛰어난 성능을 보이며, 최고의 비공개 모델과도 경쟁할 수 있습니다. 실험 결과, 데이터 규모가 증가해도 적절한 훈련 성과를 유지하며 복잡한 도구 사용 시나리오에서 두각을 나타내고 있습니다. 이러한 결과는 반복적인 강화 세부 조율 전략의 성공적인 적용을 잘 보여줍니다.



### Enhancing the De-identification of Personally Identifiable Information in Educational Data (https://arxiv.org/abs/2501.09765)
Comments:
          14 pages, 1 figure; This work has been submitted to the IEEE for possible publication

- **What's New**: 본 연구는 개인 식별 정보(PII) 탐지를 위한 비용 효율적이고 효과적인 솔루션으로 GPT-4o-mini 모델을 조사합니다. 우리는 프롬프트(prompting)와 파인튜닝(fine-tuning) 접근 방식을 비교하고, Microsoft Presidio 및 Azure AI Language와 같은 기존 프레임워크와 GPT-4o-mini의 성능을 비교합니다. 두 개의 공개 데이터셋인 CRAPII와 TSCC에 대한 평가 결과, 파인튜닝된 GPT-4o-mini 모델이 PII 탐지에서 우수한 성과를 거두었음을 보여주었습니다.

- **Technical Details**: PII는 개인을 식별할 수 있는 정보로, 학습 기술이 점점 더 교육에서 중요한 역할을 하면서 보호가 필수적입니다. 특히, 최근의 대형 언어 모델(LLMs)의 발전은 PII 탐지를 향상시킬 기회를 제공합니다. 본 연구에서는 고급 AI 모델을 사용하여 Named Entity Recognition (NER) 과정을 개선하여 PII 보호를 증대시키는 방법을 탐구합니다.

- **Performance Highlights**: 파인튜닝된 GPT-4o-mini 모델은 CRAPII 데이터셋에서 0.9589의 리콜(recall) 점수를 달성했고, 정밀도(precision) 점수는 삼중 상승했습니다. 또한 Azure AI Language의 약 10분의 1으로 계산 비용을 줄이는 성과를 보였습니다. 이러한 결과는 교육 데이터에서 PII 탐지를 위한 정확하고 비용 효율적인 도구로서 파인튜닝된 GPT-4o-mini의 잠재력을 강조합니다.



### Large language models for automated scholarly paper review: A survey (https://arxiv.org/abs/2501.10326)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 시대에 자동 학술 논문 리뷰(ASPR)의 전체적인 관점을 제공하고자 하였습니다. 특히 ASPR의 도입과 동시 존재 단계에서 LLMs의 발전과 적용을 고찰합니다. LLMs가 ASPR의 기술적 병목 문제를 해결하는 방법과 새로운 기술과 데이터 세트의 응용을 리뷰합니다.

- **Technical Details**: LLMs는 수억 또는 수천억 개의 매개변수를 갖고 있어 자연어의 복잡성과 다양성을 모델링하는 데 적합합니다. Closed-source LLM과 open-source LLM 두 가지 개발 경향이 있으며, 각각 중요한 장점과 단점을 가지고 있습니다. Closed-source 모델은 안정성과 성능이 뛰어나지만 투명성이 부족한 반면, open-source 모델은 커스터마이징이 용이하고 커뮤니티 참여도가 높지만 특정 도메인 지식 이해가 부족합니다.

- **Performance Highlights**: 연구 결과에 따르면 현재 ASPR에서 가장 많이 사용되는 LLM은 OpenAI의 GPT-4로, 문헌에서 가장 높은 빈도로 언급되었습니다. GPT 시리즈는 특히 사용 용이성과 성능 덕분에 널리 사용되며, Llama 시리즈는 상대적으로 적은 비율을 차지하고 있습니다. 이러한 결과는 ASPR 구현에 있어 LLM의 성능과 커뮤니티의 지지가 중요한 요소임을 시사합니다.



### Computational Protein Science in the Era of Large Language Models (LLMs) (https://arxiv.org/abs/2501.10282)
- **What's New**: 이 논문에서는 단백질 과학과 인공지능(AI)에 관한 최신 발전상을 정리합니다. 특히, 대형 언어 모델(LLMs)이 단백질의 시퀀스, 구조 및 기능 간의 이해를 돕기 위해 어떻게 활용되고 있는지를 다루고 있습니다. LLM 기술을 응용한 단백질 언어 모델(pLMs)들을 소개하며, 이들 모델이 단백질 구조 예측, 함수 예측 및 설계 연구에 기여하는 방법을 설명합니다.

- **Technical Details**: 기존의 단백질 언어 모델들은 여러 생물학적 관련 시퀀스를 이용하여 진화적인 지식을 활용하는 방법에 집중하고 있습니다. MSAs(다중 서열 정렬)을 기반으로 한 pLMs가 인간의 단백질 구조 학습을 위한 강력한 도구로 부각되고 있으며, ESM-MSA-1b와 같은 모델이 이러한 특성을 보여줍니다. 더욱이 MSA를 통해 생성된 '의사' 단백질을 활용하는 기법들도 발전하고 있으며, PoET와 같은 모델이 이들 기술의 필요성을 해결하고 있습니다.

- **Performance Highlights**: pLMs는 단백질 구조 예측에서 뛰어난 성능을 보여주며, 이는 기존의 단일 서열 기반 모델들을 초월합니다. 연구자들은 MSA를 통해 얻은 정보를 바탕으로 다양한 단백질 기능 예측 및 설계 문제에 접근할 수 있는 방법들을 개발하고 있습니다. 특히, PTMs(포스트 트랜슬레이셔널 수정)을 고려한 PTM-Mamba와 같은 pLM은 단백질의 구조적 및 기능적 다양성을 증가시키는데 기여하고 있습니다.



### A Simple but Effective Closed-form Solution for Extreme Multi-label Learning (https://arxiv.org/abs/2501.10179)
Comments:
          10pages, Accepted at ECIR25

- **What's New**: 이 논문은 극단 멀티 레이블 학습(Extreme Multi-Label Learning, XML)의 문제를 해결하기 위해 리지 회귀(ridge regression)를 사용한 새로운 접근 방식을 제안합니다. 기존 모델들이 많은 하이퍼파라미터를 포함하여 조정이 복잡한 반면, 이 방법은 단일 하이퍼파라미터만으로 모델을 최적화하는 간단한 솔루션을 제공합니다. 또한, 이 방법은 저빈도 레이블에 대한 예측 성능을 개선하여 중요한 정보를 포함할 수 있는 가능성을 지니고 있습니다. 실험 결과는 제안된 방법이 기존 모델들에 비해 경쟁력 있는 성능을 나타냄을 보여줍니다.

- **Technical Details**: XML 데이터셋은 각 인스턴스의 특징 벡터와 레이블 벡터로 구성됩니다. 제안된 리지 회귀 방법은 최소제곱법과 L2 정규화(L2 regularization)를 기반으로 하며, 저빈도 레이블에 대한 예측을 위해 라벨별 가중치(label-specific weights)를 통합하는 방식을 제공합니다. 이 모델은 단일 하이퍼파라미터만 사용하므로 구현이 단순하고 해석하기 용이합니다. 리지 회귀를 통해 XML 작업을 모델링할 때, 데이터 포인트가 적은 저빈도 레이블의 예측이 중요한 고려사항입니다.

- **Performance Highlights**: 제안된 리지 회귀 모델은 다양한 XML 기준 데이터셋에서 실험되었으며, 그 결과 기존의 다양한 하이퍼파라미터를 가진 모델들과 비슷하거나 더 나은 성능을 달성했습니다. 특히, 저빈도 레이블의 예측 성능이 상당히 개선되었으며, 기존 방법과 비교해 거의 변경 없이 구현될 수 있음을 보여주었습니다. 이러한 접근은 XML 작업을 보다 간단하게 만들어 주며, 향후 연구 및 응용 가능성을 제시합니다.



### OMoE: Diversifying Mixture of Low-Rank Adaptation by Orthogonal Finetuning (https://arxiv.org/abs/2501.10062)
- **What's New**: 본 연구에서는 파라미터 효율적인 세부 조정(Parameter-Efficient Fine-Tuning, PEFT) 방법론에서 Mixture-of-Experts (MoE) 아키텍처를 활용하여, 전문가들 간의 다양성을 증진시키기 위한 Orthogonal Mixture-of-Experts (OMoE)를 제안합니다. 전문가들이 유사한 표현으로 축소되는 문제를 해결하고, 메모리 병목 현상을 완화하여 모델의 성능을 안정적으로 향상시킬 수 있는 접근 방식을 제공합니다. 또한 OMoE는 최적성을 유지하면서도 구조적 제약을 가해 전문가의 표현이 Stiefel 초다양체(Stiefel manifold) 내에 위치하도록 훈련합니다.

- **Technical Details**: OMoE는 Gram-Schmidt 과정을 활용하여 전문가들의 표현을 직교화(orthogonalize) 함으로써 서로 다른 표현 간의 유사성을 없애고, 하이퍼스피어 에너지를 유지합니다. 이를 통해 OMoE는 기존 MoE의 단점을 보완하며, 메모리 사용을 최소화하고 훈련 가능한 파라미터 수를 약 75% 줄입니다. MoE 아키텍처 내에서 이러한 직교화 제약을 직접 적용함으로써 OMoE는 최적의 학습 목표를 유지하면서도 다양성을 효과적으로 증진시킬 수 있게 됩니다.

- **Performance Highlights**: 다양한 상식 추론(commonsense reasoning) 벤치마크에서의 실험 결과, OMoE는 최신 PEFT 방법들과 비교했을 때 일관되게 안정적인 성능 향상을 달성하였으며, 필요한 전문가의 수를 크게 줄일 수 있음을 보여주었습니다. 이러한 실험은 OMoE가 성능 개선을 이루면서도 자원 효율성을 유지함을 강조합니다. OMoE는 단순히 전문가의 수를 늘리기보다는, 각 전문가의 다양성을 중심으로 설계되어 더 나은 결과를 도출하는 것으로 확인되었습니다.



### RichSpace: Enriching Text-to-Video Prompt Space via Text Embedding Interpolation (https://arxiv.org/abs/2501.09982)
- **What's New**: 이번 연구에서는 텍스트를 비디오로 생성하는 데 있어 중요성이 상대적으로 간과되었던 텍스트 임베딩(text embedding)의 최적화 방법을 제안합니다. 특히, 임베딩 공간에서의 보간(interpolation)을 통해 최적의 텍스트 임베딩을 선택함으로써 비디오 생성 모델의 성능을 개선할 수 있다고 주장합니다. 이러한 접근 방식은 기존의 다수의 텍스트 인코더를 사용하는 방법과 대조를 이루며, 계산 비용을 줄이는 동시에 효과적인 결과를 도출할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 수직각 발 임베딩(perpendicular foot embeddings)과 코사인 유사성(cosine similarity) 알고리즘을 활용하여 텍스트 임베딩 공간 내에서 최적의 텍스트 임베딩을 찾는 새로운 방법을 소개합니다. 이를 통해 비디오 생성 모델이 원하는 비디오를 생성할 수 있는 능력을 극대화하는 것을 목표로 하고 있습니다. 또한, CogvideoX 모델의 각 구성 요소에 대한 공식적인 정의와 문제 정의를 제공하여, 향후 연구자들이 더욱 쉽게 이 알고리즘을 적용할 수 있도록 돕고 있습니다.

- **Performance Highlights**: 실험 결과, 최적의 텍스트 임베딩을 선택하는 것이 비디오 생성 모델이 원하는 비디오를 생성하는 데 큰 영향을 미친다는 것을 확인했습니다. 본 연구의 제안에 따른 알고리즘은 비디오 품질 향상은 물론 복잡한 텍스트 프롬프트를 보다 효과적으로 처리하는 능력을 보여주었습니다. 이러한 성과는 텍스트에서 비디오로의 전환이 갖는 잠재력을 뒷받침하며, 향후 다양한 응용 프로그램에서 활용될 수 있는 기반을 제공합니다.



### Sympathy over Polarization: A Computational Discourse Analysis of Social Media Posts about the July 2024 Trump Assassination Attemp (https://arxiv.org/abs/2501.09950)
- **What's New**: 이 연구는 2024년 7월 13일, 펜실베니아에서 열린 트럼프 집회에서 발생한 암살 시도의 여파로 공공 여론과 논의 주제에 미친 단기적 영향을 모델링하는 것을 목표로 하고 있습니다. 구체적으로 세 가지 주요 질문에 대해 다루며, 트럼프에 대한 공공 감정의 변화를 조사하고, 암살 시도가 기존 정치적 정렬과 무관하게 공공 태도에 미치는 영향을 분석합니다. 이후 위기 전후의 온라인 논의 주제를 탐구하여 정치적 충격 사건에 대한 반응을 설명합니다.

- **Technical Details**: 본 연구는 대규모 언어 모델 기반의 감정 분석 (sentiment analysis), 차이의 차이 모델링 (difference-in-differences modeling), 주제 모델링 (topic modeling) 기법을 사용하여 진행되었습니다. 이러한 방법론을 통해, 암살 시도 이후 트럼프에 대한 공공의 반응이 편향되지 않고 동정적이었다는 결과를 얻었습니다. 또한 지역 및 이념적 차이에 따른 분석을 통해 이러한 반응이 어떻게 나타나는지를 살펴보았습니다.

- **Performance Highlights**: 연구 결과, 공공 감정은 트럼프에 대한 동정으로 표현되었으며, 이는 정치적 극단화를 피하는 경향을 보였습니다. 이 연구는 정치적 사건이 공공 여론에 미치는 영향을 이해하는 데 중요한 기여를 하며, 특히 충격적 사건의 맥락에서 여론 변화의 복잡성을 드러냅니다. 결과적으로 연구자는 시장이 어떻게 반응했는지를 분석하여 향후 정치적 커뮤니케이션 전략을 제안할 수 있는 기초 자료를 제시합니다.



### Steering Large Language Models with Feature Guided Activation Additions (https://arxiv.org/abs/2501.09929)
Comments:
          7 maintext pages, 14 appendix pages

- **What's New**: 본 논문에서는 Feature Guided Activation Additions(FGAA)라는 새로운 활성화 조정(activation steering) 방법을 소개합니다. FGAA는 Sparse Autoencoder(SAE)와 Contrastive Activation Addition(CAA)의 통찰력을 활용하여 모델의 행동을 보다 효과적으로 제어할 수 있도록 설계되었습니다. 기존 방법들의 부족한 정밀성과 해석 가능성을 개선하며, 복잡한 조정 작업에서도 일관된 출력 품질을 유지합니다.

- **Technical Details**: FGAA는 SAE의 잠재 공간에서 작동하며, 원하는 SAE 특성을 선택하기 위해 최적화 기법을 사용합니다. 이 방법은 라벨이 있는 데이터로부터 긍정적(positive) 및 부정적(negative) 예시를 통해 대조적 차이를 계산하여 탐지된 특징을 기반으로 유용한 조정 벡터를 생성합니다. FGAA는 밀도 필터링(density filtering), BOS(Beginning Of Sequence) 피처 제거, 그리고 상위 특성 선택(top-k selection)이라는 세 가지 중요한 필터링 단계를 통해 특성 벡터를 변형하여 조정 벡터를 생성합니다.

- **Performance Highlights**: GFWWA는 Gemma-2-2B 및 Gemma-2-9B 모델에서 다양한 조정 작업을 수행하는 동안 기존 CAA, SAE 디코더 조정 및 SAE-TS보다 우수한 성능을 보여줍니다. 연구 결과, 조정 규모와 모델의 일반적인 능력 간의 중요한 상쇄 관계가 존재함을 발견하였으며, 이는 모든 테스트된 조정 방법에 걸쳐 일관됩니다. FGAA는 조정 효과와 출력 일관성을 개선하며, 복잡한 조정 작업에서의 성능이 특히 두드러집니다.



### Enhancing Generalization in Chain of Thought Reasoning for Smaller Models (https://arxiv.org/abs/2501.09804)
- **What's New**: 이 논문은 Chain-of-Thought (CoT) 추론을 소형 언어 모델(small language models)에서 효과적으로 수행하기 위한 혁신적인 접근법을 제안합니다. PRompt-Assisted Domain-Adversarial fine-tuning (PRADA) 프레임워크는 다양한 CoT 도메인을 통합하여 소형 LLM의 일반화 능력을 개선하는 방법을 모색합니다. 이를 통해 복잡한 태스크에 대한 대처를 개선하고, 보다 강력한 CoT 일반화를 가능하게 합니다.

- **Technical Details**: PRADA는 (1) 대형 teacher 모델이 다양한 CoT 추론 응답을 생성하도록 유도하며, (2) prompt learning layers를 통해 소형 모델에 도메인 독립적 지식을 습득하도록 돕고, (3) 도메인 적대적 미세 조정을 통해 모델이 불변 도메인 특징을 학습하게끔 합니다. 이러한 접근법은 CoT 지식 증류 과정에서 발생할 수 있는 일반화 열화를 해결하기 위한 것입니다.

- **Performance Highlights**: PRADA는 12개의 다양한 도메인을 대상으로 하는 실험에서 기존 CoT 지식 증류 방법들보다 뛰어난 성능을 보였습니다. 특히 PRADA를 활용한 소형 LLM은 새로운 미지의 도메인에서도 뛰어난 일반화 능력을 발휘하며, 모델의 설명 가능성을 높이는 결과를 보였습니다.



### Conversational Text Extraction with Large Language Models Using Retrieval-Augmented Systems (https://arxiv.org/abs/2501.09801)
- **What's New**: 본 연구는 PDF 문서에서 텍스트를 추출하고 사용자와의 상호작용을 향상시키기 위해 Large Language Models (LLMs)을 활용하는 시스템을 소개합니다. 이 시스템은 Retrieval-Augmented Generation (RAG) 기법을 통해 사용자 질문에 대한 유용한 응답을 제공하며, 관련 내용을 PDF에서 강조합니다.

- **Technical Details**: 사용자가 PDF를 업로드하면, 이 시스템은 문서 특정 벡터 저장소를 만들기 위해 문장 임베딩 (sentence embeddings)을 사용하여 PDF를 처리합니다. 이 벡터 저장소는 사용자 질문에 대한 관련 섹션을 효율적으로 검색할 수 있도록 지원하며, LLM은 이를 기반으로 대화형 응답을 생성합니다.

- **Performance Highlights**: 제안된 시스템은 텍스트 추출 및 요약에 있어 기존의 최신 기술들과 비교할 때 경쟁력 있는 ROUGE 값들을 보여줍니다. 이러한 성능은 연구자, 학생, 그리고 문서에서 지식을 효율적으로 추출하고 통찰을 얻고자 하는 모든 사용자에게 유용한 도구를 제공합니다.



### Computing Optimization-Based Prompt Injections Against Closed-Weights Models By Misusing a Fine-Tuning API (https://arxiv.org/abs/2501.09798)
- **What's New**: 이번 논문에서는 폐쇄형 가중치 (closed-weight) 대형 언어 모델 (Large Language Models, LLM)에 대한 새로운 위협을 제시합니다. 공격자는 원격 세부 조정 인터페이스에서 반환되는 손실 유사 정보 (loss-like information)를 활용하여 최적화 기반의 프롬프트 주입 (prompt injections)을 계산할 수 있는 방법을 설명합니다. 이는 LLM 공급자가 호스팅하는 세부 조정 기능을 통해 가능하며, 개발자에게 유용성을 제공하지만 공격자에게는 적대적 프롬프트를 계산할 수 있는 정보를 노출합니다.

- **Technical Details**: 세부 조정 인터페이스는 LLM 공급자가 직면하는 유틸리티-보안 트레이드오프 (utility-security tradeoff)를 강조합니다. 특히, Gemini 세부 조정 API에서 반환되는 손실 유사 값이 공격자가 적대적 프롬프트를 이산 최적화 (discrete optimization)에 활용할 수 있는 유용한 신호를 제공한다는 점이 주목할 만합니다. 실험 분석을 통해, 공격자는 탐욕적 검색 알고리즘 (greedy search algorithm)을 이용하여 적대적 프롬프트 최적화를 수행할 수 있음을 입증했습니다.

- **Performance Highlights**: PurpleLlama 프롬프트 주입 벤치마크를 사용하여 공격 성공률이 구글의 Gemini 가족 LLM에서 65%에서 82% 사이임을 보여줍니다. 이러한 공격 방식은 세부 조정 기능이 개발자에게 유용한 특성을 제공하면서도 LLM에 강력한 공격을 노출시키는 것을 잘 보여줍니다.



### Towards Large Reasoning Models: A Survey on Scaling LLM Reasoning Capabilities (https://arxiv.org/abs/2501.09686)
Comments:
          36 pages, 5 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 복잡한 추론 작업을 다루기 위한 새로운 접근 방식을 제시합니다. 특히, '사고(thought)' 개념을 도입해 LLM이 인간의 추론 과정을 모방할 수 있도록 하고, 강화 학습(reinforcement learning)을 통해 LLM의 학습 능력을 증대시키는 경향을 강조합니다. 이러한 접근 방식을 통해, LLM의 추론 정확성을 크게 향상시킬 수 있는 새로운 연구 경계를 탐구하고 있습니다.

- **Technical Details**: LLM의 훈련에서는 트랜스포머(transformer) 아키텍처가 사용되며, 대규모 텍스트 코퍼스에서 사전 훈련(pretraining)을 거치게 됩니다. 최근의 연구는 사람의 주석 없이 LLM 주도 검색 알고리즘을 통해 자동으로 추론 궤적을 생성하는 방법을 강조하며, 이는 LLM의 추론 능력을 대폭 확장합니다. 또한, 과정 보상 모델(Process Reward Models, PRMs)을 통해 LLM의 훈련과 추론에서 더욱 효율적으로 작동하도록 지원합니다.

- **Performance Highlights**: 최근 연구에서는 테스트 시간 동안 LLM이 '더 많은 토큰'을 사용하여 사고를 할 수 있도록 장려하면 추론 정확성이 크게 향상될 수 있음을 보여줍니다. 이 연구는 LLM이 목표에 맞는 추론 단계를 생성할 수 있도록 하고, 논문에서 제안하는 RL 기반 훈련 및 검색 기반 테스트 시간 확장 방법은 LLM의 추론 능력을 최대한 활용하는 데 기여하고 있습니다. 'OpenAI의 o1 시리즈'는 이러한 접근 방법의 효과를 입증하는 중요한 이정표로 자리 잡고 있습니다.



New uploads on arXiv(cs.IR)

### A Simple but Effective Closed-form Solution for Extreme Multi-label Learning (https://arxiv.org/abs/2501.10179)
Comments:
          10pages, Accepted at ECIR25

- **What's New**: 이 논문은 극단 멀티 레이블 학습(Extreme Multi-Label Learning, XML)의 문제를 해결하기 위해 리지 회귀(ridge regression)를 사용한 새로운 접근 방식을 제안합니다. 기존 모델들이 많은 하이퍼파라미터를 포함하여 조정이 복잡한 반면, 이 방법은 단일 하이퍼파라미터만으로 모델을 최적화하는 간단한 솔루션을 제공합니다. 또한, 이 방법은 저빈도 레이블에 대한 예측 성능을 개선하여 중요한 정보를 포함할 수 있는 가능성을 지니고 있습니다. 실험 결과는 제안된 방법이 기존 모델들에 비해 경쟁력 있는 성능을 나타냄을 보여줍니다.

- **Technical Details**: XML 데이터셋은 각 인스턴스의 특징 벡터와 레이블 벡터로 구성됩니다. 제안된 리지 회귀 방법은 최소제곱법과 L2 정규화(L2 regularization)를 기반으로 하며, 저빈도 레이블에 대한 예측을 위해 라벨별 가중치(label-specific weights)를 통합하는 방식을 제공합니다. 이 모델은 단일 하이퍼파라미터만 사용하므로 구현이 단순하고 해석하기 용이합니다. 리지 회귀를 통해 XML 작업을 모델링할 때, 데이터 포인트가 적은 저빈도 레이블의 예측이 중요한 고려사항입니다.

- **Performance Highlights**: 제안된 리지 회귀 모델은 다양한 XML 기준 데이터셋에서 실험되었으며, 그 결과 기존의 다양한 하이퍼파라미터를 가진 모델들과 비슷하거나 더 나은 성능을 달성했습니다. 특히, 저빈도 레이블의 예측 성능이 상당히 개선되었으며, 기존 방법과 비교해 거의 변경 없이 구현될 수 있음을 보여주었습니다. 이러한 접근은 XML 작업을 보다 간단하게 만들어 주며, 향후 연구 및 응용 가능성을 제시합니다.



### MechIR: A Mechanistic Interpretability Framework for Information Retrieva (https://arxiv.org/abs/2501.10165)
Comments:
          5 pages, 2 figures, Accepted to ECIR 2025 as a Demo Paper

- **What's New**: 이 논문에서는 기계적 해석 가능성(mechanistic interpretability)이 자연어 처리(NLP) 분야에서 신경 모델의 성능을 이해하는 데 중요한 역할을 하고 있음을 강조합니다. 이를 위해 IR(정보 검색) 작업에 특화된 진단 분석 및 개입을 위한 유연한 프레임워크인 MechIR를 소개합니다. 이 프레임워크는 내부 모델 구성 요소를 인과적으로 조사할 수 있는 도구를 제공하여 투명성과 시스템 개선을 목표로 합니다.

- **Technical Details**: MechIR는 IR 모델의 역설계를 돕기 위한 Python 패키지로, 활성화 패칭(activation patching)이라는 인과적 개입 방법을 사용하여 특정 행동을 유발하는 구성 요소를 식별합니다. 이 방식에서는 두 가지 입력 쌍을 사용하여 모델의 성능을 평가하며, 특히 버추얼 패칭(patched run)을 통해 변경된 성능을 분석합니다. MechIR는 일반적인 검색 아키텍처를 지원하며, 실험 결과를 시각화하는 기본 플로팅 기능도 제공합니다.

- **Performance Highlights**: MechIR는 활성화 패칭과 같은 방법을 통해 IR의 모델 성능을 향상시키기 위한 효율적인 도구로 자리잡고 있습니다. 이 패키지는 초보자와 경험자 모두를 위한 튜토리얼을 포함하고 있어, 기계적 해석 가능성 연구에 대한 접근성을 높이며, 향후 추가 기능 개발 계획도 포함되어 있습니다. 이러한 접근을 통해 연구자들은 모델의 행동을 보다 직관적으로 이해하고 설명할 수 있는 기회를 가지게 될 것입니다.



### A Worrying Reproducibility Study of Intent-Aware Recommendation Models (https://arxiv.org/abs/2501.10143)
- **What's New**: 최근 의도 인식 추천 시스템(intent-aware recommender systems, IARS)에 대한 관심이 증가하고 있습니다. 이러한 시스템은 소비자들의 근본적인 동기와 단기 목표를 고려하여 더 나은 추천을 제공할 수 있다는 약속을 가지고 있습니다. 이 연구에서는 최근의 IARS 연구들이 기법적인 한계와 복잡한 신경 추천 모델의 재현성 문제에 부딪히고 있는지를 조사하고 있습니다.

- **Technical Details**: 본 연구에서는 IARS 모델 5개를 선택하여, 이들 모델을 수치적으로 비슷한 전통적인 비신경 추천 모델과 비교 분석하였습니다. 상위 학술지에 발표된 이 모델들은 각각의 최적 하이퍼파라미터에 따라 테스트되었지만, 논문에서 보고된 결과와 일치하지 않는 경우가 발견되었습니다. 이 연구는 의도 인식 추천 시스템의 재현성과 관련된 심각한 문제를 부각하고 있습니다.

- **Performance Highlights**: 연구 결과, 검사한 모든 IARS 접근 방식은 최소한 하나의 전통적인 모델보다 지속적으로 뒤처지는 결과를 보였습니다. 이는 현재의 IARS 연구가 신뢰할 수 있는 결과를 제공하지 않고 있음을 시사하며, 더욱 엄격한 연구 방법론이 필요함을 강조합니다. 이 연구는 IARS 분야의 향후 연구 방향성을 설정하는 데 중요한 기초 자료로 작용할 것입니다.



### PaSa: An LLM Agent for Comprehensive Academic Paper Search (https://arxiv.org/abs/2501.10120)
- **What's New**: 본 논문에서는 PaSa라는 대규모 언어 모델에 기반한 고급 논문 검색 에이전트를 소개합니다. PaSa는 검색 도구 호출, 논문 읽기 및 관련 참고 문헌 선택을 자동으로 수행하여 복잡한 학술 쿼리에 대해 포괄적이고 정확한 결과를 얻을 수 있도록 설계되었습니다. 또한, AutoScholarQuery라는 합성 데이터셋을 사용하여 PaSa를 최적화하며, 현실 세계의 학술 쿼리를 평가하기 위한 RealScholarQuery라는 벤치마크도 개발하였습니다.

- **Technical Details**: PaSa는 Crawler와 Selector라는 두 개의 LLM 에이전트로 구성되어 있습니다. Crawler는 관련 논문을 수집하고 인용 네트워크를 탐색하며, Selector는 수집된 논문을 읽어 사용자 쿼리의 요구 사항을 충족하는지 확인합니다. 논문의 훈련 및 평가를 위해 AutoScholarQuery와 RealScholarQuery라는 두 개의 고품질 데이터셋을 구축하였고, 강화 학습 프레임워크인 AGILE을 통해 PaSa를 최적화합니다.

- **Performance Highlights**: PaSa-7B는 AutoScholarQuery 및 RealScholarQuery 실험에서 기존의 다양한 검색 시스템을 초월하는 성능을 보였습니다. 특히, PaSa-7B는 Google 및 GPT-4o와의 비교에서 Recall@20 기준으로 37.78%의 향상을 기록했습니다. 또한, PaSa-7B는 PaSa-GPT-4o보다 Recall에서 30.36%, Precision에서 4.25%나 더 우수한 성과를 달성하였습니다.



### Empirical Evaluation of Embedding Models in the Context of Text Classification in Document Review in Construction Delay Disputes (https://arxiv.org/abs/2501.09859)
- **What's New**: 이 논문은 다양한 텍스트 임베딩 모델의 효과를 비교 분석하여 텍스트 분류에서의 적용 가능성을 보여줍니다. 특히, 시간 지연과 관련된 문장들을 식별하기 위해 K-최근접 알고리즘(K-Nearest Neighbors, KNN)과 로지스틱 회귀(Logistic Regression, LR)를 활용하여 이진 분류 작업을 수행합니다. 이는 건설 지연 분쟁 문서 검토 과정에서의 효율성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 본 연구에서는 논문에서 제시한 네 가지 텍스트 임베딩 모델을 평가하며, 각 모델의 성능을 텍스트 분류의 효율성 측면에서 비교합니다. 연구자는 레이블이 있는 데이터셋 내에서 '지연(delay)'과 '비지연(not delay)'을 구별하기 위해 머신 러닝 프로세스를 진행해, 텍스트 스니펫의 임베딩을 사용하여 감독형(Supervised) 텍스트 분류 모델을 훈련시킵니다. 이 분석은 문서 검토 과정에서 모델의 실제 적용을 보여줍니다.

- **Performance Highlights**: 연구 결과는 임베딩 모델이 법률 문서 분석의 효율성과 정확성을 증가시킬 수 있는 가능성을 강조합니다. 이러한 향상된 성능은 복잡한 조사 상황에서 더 잘-informed decision-making을 가능하게 하며, 이로 인해 건설 지연과 관련된 논의에서 민감한 데이터 처리 및 분류의 중요성이 부각됩니다.



### Conversational Text Extraction with Large Language Models Using Retrieval-Augmented Systems (https://arxiv.org/abs/2501.09801)
- **What's New**: 본 연구는 PDF 문서에서 텍스트를 추출하고 사용자와의 상호작용을 향상시키기 위해 Large Language Models (LLMs)을 활용하는 시스템을 소개합니다. 이 시스템은 Retrieval-Augmented Generation (RAG) 기법을 통해 사용자 질문에 대한 유용한 응답을 제공하며, 관련 내용을 PDF에서 강조합니다.

- **Technical Details**: 사용자가 PDF를 업로드하면, 이 시스템은 문서 특정 벡터 저장소를 만들기 위해 문장 임베딩 (sentence embeddings)을 사용하여 PDF를 처리합니다. 이 벡터 저장소는 사용자 질문에 대한 관련 섹션을 효율적으로 검색할 수 있도록 지원하며, LLM은 이를 기반으로 대화형 응답을 생성합니다.

- **Performance Highlights**: 제안된 시스템은 텍스트 추출 및 요약에 있어 기존의 최신 기술들과 비교할 때 경쟁력 있는 ROUGE 값들을 보여줍니다. 이러한 성능은 연구자, 학생, 그리고 문서에서 지식을 효율적으로 추출하고 통찰을 얻고자 하는 모든 사용자에게 유용한 도구를 제공합니다.



### Passage Segmentation of Documents for Extractive Question Answering (https://arxiv.org/abs/2501.09940)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 모델의 성능 향상에 있어 중요한 역할을 하는 chunking(청킹) 과정에 주목합니다. 저자들은 새로운 Logits-Guided Multi-Granular Chunker (LGMGC) 프레임워크를 소개하며, 이는 긴 문서를 맥락적으로 독립적인 다양한 크기의 chunk로 분할합니다. 실험 결과, LGMGC는 기존의 chunking 방법보다 우수한 성능을 나타낸다고 보고합니다.

- **Technical Details**: LGMGC는 두 가지 청킹 모듈, Logits-Guided Chunker와 Multi-Granular Chunker를 통합하여 구성됩니다. 이 과정은 문서를 의미적으로 일관된 단위로 세분화하며, 각 단위는 후속 질문의 유형에 따라 다양한 크기로 더 나누어집니다. 구체적으로, LLM의 로짓 정보(logits information)를 활용하여 청킹 단위를 정의하고, 이를 기반으로 세분화된 child chunk를 생성합니다.

- **Performance Highlights**: LGMGC의 성능은 두 개의 벤치마크 데이터세트에서 평가되었습니다. 그 결과, 제안된 청킹 접근 방식은 정보 검색 및 질문 응답 작업에서 현재 사용되는 청킹 방법들보다 더 나은 성능을 보였습니다. 이는 청킹 과정에서의 맥락적 정보의 중요성을 강조하며, 문서의 세분화가 질의 응답의 정확성에 긍정적인 영향을 미칠 수 있음을 보여줍니다.



### Semi-Supervised Image-Based Narrative Extraction: A Case Study with Historical Photographic Records (https://arxiv.org/abs/2501.09884)
Comments:
          This paper has been accepted for oral presentation in the findings track of the 47th European Conference on Information Retrieval (ECIR 2025). Source code and experiments are available at this https URL

- **What's New**: 이 논문은 역사적 사진 기록에서 내러티브(narrative)를 추출하기 위한 반지도 학습(semi-supervised) 접근법을 제시합니다. 기존의 비지도(text-based) 방식에 이미지를 적용하여 심층 학습(deep learning) 기술을 활용한 시각적 특성 추출 및 유사성 계산을 목표로 하고 있습니다. 특히, 1928년 볼리비아의 Sacambaya Expedition에서 캡처한 Robert Gerstmann의 사진을 담은 ROGER 데이터셋에 본 방법을 적용했습니다.

- **Technical Details**: 저자는 원래의 비지도 내러티브 맵 알고리즘을 이미지 데이터에 맞게 조정하며, 전문가 주석(annotator)이 제공한 부분 레이블(partial labels)을 통해 의미 있는 내러티브를 추출합니다. 이 과정에서는 시각적 특징(feature) 추출, 유사성 계산, 내러티브 구조 구성 등의 중요한 단계가 포함됩니다. 또한, 동적 시간 왜곡(Dynamic Time Warping, DTW) 알고리즘을 사용하여 전문가가 만든 기준과 추출된 내러티브를 비교합니다.

- **Performance Highlights**: 연구 결과는 10개 이상의 이미지를 포함한 긴 타임라인의 경우, 내러티브 맵 접근법이 무작위 샘플링(random sampling)보다 우수함을 보여줍니다. 전문가 평가를 통해 추출된 내러티브의 역사적 정확성 및 일관성이 확인되었습니다. 이러한 연구는 역사 연구자와 디지털 인문학 연구자에게 새로운 도구를 제공하여 대규모 이미지 컬렉션에서 의미 있는 내러티브를 탐색하고 이해하는 데 기여합니다.



New uploads on arXiv(cs.CV)

### FaceXBench: Evaluating Multimodal LLMs on Face Understanding (https://arxiv.org/abs/2501.10360)
Comments:
          Project Page: this https URL

- **What's New**: FaceXBench라는 포괄적인 벤치마크가 소개되어 MLLMs의 복잡한 얼굴 이해 능력을 평가하기 위한 기준을 제공합니다. 이 벤치마크는 25개의 공개 데이터셋과 새롭게 생성된 FaceXAPI에서 파생된 5,000개의 다중 모달 선택 질문을 포함하며, 14개의 다양한 작업과 6개의 주요 카테고리를 다룹니다. 연구자들은 기존 MLLMs가 얼굴 관련 과제에서 겪는 도전 과제를 밝혀내었으며, 이는 이 분야의 발전을 위한 중요한 기초 자료가 될 것입니다.

- **Technical Details**: FaceXBench는 비편향성과 공정성, 얼굴 인증, 인식, 분석, 로컬라이제이션 및 도구 검색 포함한 6개의 주요 카테고리에서 14개의 작업을 평가합니다. 연구진은 26개의 오픈 소스 MLLM 모델과 2개의 고급 프라이빗 MLLM 모델을 대상으로 종합적인 평가를 실시했으며, 이러한 분석은 MLLMs가 복잡한 얼굴 이해 과제에서 얼마나 부족한지를 명확히 보여줍니다. 연구팀은 zero-shot, in-context task description, chain-of-thought prompting 세 가지 설정에서 모델의 성과를 분석했습니다.

- **Performance Highlights**: 연구 결과, 상위 모델들도 deepfake 탐지와 군중 수 계산과 같은 세부적인 시각 분석을 필요로 하는 과제에서 어려움을 겪고 있음을 나타냅니다. GPT-4o와 GeminiPro 1.5 모델의 정확도는 각각 50.24%와 54.40%로, FaceXBench가 제시하는 도전 과제가 상당함을 보여줍니다. 또한, context 정보를 효과적으로 활용하지 못하는 경향이 드러났으며, 이는 MLLMs의 얼굴 관련 작업에서 나쁜 성과로 이어졌습니다.



### Zero-Shot Monocular Scene Flow Estimation in the Wild (https://arxiv.org/abs/2501.10357)
Comments:
          Project Website: this https URL

- **What's New**: 이번 논문에서는 기존의 scene flow(SF) 추정 기술의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. 특히, geometry와 motion을 동시에 추정하여 더 정확한 예측을 가능하게 하며, 1백만 개의 주석이 달린 학습 샘플을 활용해 데이터 부족 문제를 해결합니다. 또한, 다양한 파라미터화 방식을 평가하고 효과적인 파라미터화를 채택하여 성능을 향상시킵니다.

- **Technical Details**: 새로운 모델은 monocular SF 추정을 목표로 하며, geometry와 motion을 연계하여 처리합니다. 이를 통해 2D 공간에서 관찰되는 이동량이 깊이와 운동의 복합 효과임을 고려하고, 이로 인해 발생할 수 있는 오류를 최소화합니다. 데이터셋은 실내 및 실외에서 다양한 진실 데이터 주석을 포함하여 사용되며, optical flow를 사용하여 SF의 이미지 공간 내 투영을 감독합니다.

- **Performance Highlights**: 제안된 모델은 기존의 SF 추정 방법과 대규모 모델 기반 기준선보다 3D end-point error 측면에서 우수한 성능을 보입니다. 또한, 기존 데이터에서의 성능 저하 없이 DAVIS와 RoboTAP의 다양한 비디오에 대해 제로샷 제너럴리제이션(zero-shot generalization)을 보여줍니다. 이러한 성과를 통해 우리 방법이 실제 환경에서 SF 예측을 보다 실용적으로 만들었다는 점을 강조합니다.



### 3rd Workshop on Maritime Computer Vision (MaCVi) 2025: Challenge Results (https://arxiv.org/abs/2501.10343)
Comments:
          Part of the MaCVi 2025 workshop

- **What's New**: 해양 컴퓨터 비전에 대한 제3회 워크숍(MaCVi 2025)은 무인 수상 차량(Unmanned Surface Vehicles, USV) 및 수중 시스템에 초점을 맞추고 있습니다. 이 보고서는 700개 이상의 제출물에서 도출된 통계적 및 정성적 분석 결과를 종합적으로 제시합니다. 데이터셋, 평가 코드 및 리더보드는 누구나 접근할 수 있도록 공개되었습니다.

- **Technical Details**: 해양 환경의 고유한 도전 과제를 해결하기 위해 독창적인 비전 알고리즘 개발이 요구됩니다. 특히, 거리 추정(Distance Estimation) 및 객체 탐지(Object Detection) 기술이 필요하며, 참가자들은 약 3,000개의 레이블링된 훈련 샘플이 포함된 데이터셋을 활용해야 합니다. 우수한 모델은 ONNX 포맷으로 내보내야 하며, 제출 제한은 챌린지마다 하루 1-3회로 설정됩니다.

- **Performance Highlights**: 참여자들은 모델의 성능을 평가하기 위한 다양한 지표를 사용해야 하며, 여기에는 객체 탐지의 평균 정밀도(AP)와 거리 추정 정확도가 포함됩니다. 거리 예측의 정확도를 평가하기 위해 절대 오차(absolute error)와 상대 오차(relative error)도 평가합니다. 제출된 모델은 50만 개의 파라미터를 초과해서는 안 되며, 총 60개의 제출물이 6개 팀으로부터 접수되었습니다.



### DiffStereo: High-Frequency Aware Diffusion Model for Stereo Image Restoration (https://arxiv.org/abs/2501.10325)
Comments:
          9 pages, 6 figures

- **What's New**: 이 논문에서는 Diffusion Models (DMs)을 스테레오 이미지 복원에 처음으로 적용한 DiffStereo를 제안합니다. 기존의 DMs는 높은 주파수 세부 정보를 손실하면서 의미론적 정보에 집중해왔지만, 본 연구에서는 이러한 한계를 극복하기 위해 고주파 인식(diffusion model) 모델을 개발하였습니다. 이를 통해 스테레오 이미지 복원에서의 성능을 향상시키는 것을 목표로 하였습니다.

- **Technical Details**: DiffStereo는 고해상도(HQ) 이미지의 잠재적 고주파 표현(latent high-frequency representation, LHFR)을 학습한 후, 스테레오 이미지에 대한 LHFR을 추정하기 위해 DM을 훈련합니다. 그런 다음 이 정보를 Fuse하여 Transformer 기반의 스테레오 이미지 복원 네트워크에 통합하여 고주파 정보를 제공합니다. 또한, LHFR의 해상도는 입력 이미지와 동일하게 유지되어, 왜곡으로부터 내재된 텍스처를 보존합니다.

- **Performance Highlights**: DiffStereo는 생성적 DM과 Transformer를 결합하여 스테레오 슈퍼 해상도, 디블러링(deblurring), 저조도 향상(low-light enhancement) 작업에서 기존의 최첨단 방법들보다 더 높은 복원 정확성 및 향상된 지각 품질을 달성합니다. 실험 결과는 DiffStereo가 뛰어난 성능을 입증하고 있음을 확인합니다.



### HiMix: Reducing Computational Complexity in Large Vision-Language Models (https://arxiv.org/abs/2501.10318)
- **What's New**: 본 논문은 Hierarchical Vision injection for Mixture Attention(HiMix)라는 새로운 비전-언어 상호작용 메커니즘을 제안하여, LVLMs(대규모 비전-언어 모델)의 계산 복잡성을 획기적으로 줄이면서 성능 저하를 최소화합니다. 기존의 LVLM들은 비전 시퀀스와 언어 시퀀스를 단순히 연결하여 사용하는데, 이는 계산 복잡성을 비약적으로 증가시킵니다. 그러나 HiMix는 언어 시퀀스만 전체 전파에 관여하고, 비전 시퀀스는 각 레이어의 특정 단계에서만 언어와 상호작용하도록 하여 전체 계산 비용을 10배 줄이는 성과를 거두었습니다.

- **Technical Details**: HiMix는 기존 Vanilla-LVLM 접근 방식을 수정하여 시각적 정보의 전파 과정을 최적화합니다. 향후 단계에서 비전 정보를 주입함으로써 언어 모델이 필요한 경우에만 비전 데이터를 참조하며, 이로 인해 계산 비용을 절감할 수 있습니다. 본 연구에서는 시각 정보가 전파 과정 전반에서 필요하지 않을 수 있다는 가설을 세우고, 이를 기반으로 계층적 상호작용을 통해 비전 정보를 효과적으로 활용합니다.

- **Performance Highlights**: HiMix는 Qwen2와 Llama3와 같은 다양한 모델에서 10배의 계산 비용 절감을 실현하면서도 경쟁력 있는 성능을 유지합니다. 이러한 성과는 비전-언어 이해 분야에 대한 새로운 시각을 제공할 것으로 기대됩니다. HiMix의 도입으로 연구자들은 더 높은 수준의 의미 추출 및 효율적인 모델 구현이 가능할 것입니다.



### GSTAR: Gaussian Surface Tracking and Reconstruction (https://arxiv.org/abs/2501.10283)
- **What's New**: 이번 연구에서 우리는 GSTAR라는 새로운 방법을 제안합니다. GSTAR는 복잡한 형태 변화가 있는 동적인 장면에서 포토리얼리스틱 렌더링과 정확한 표면 재구성을 달성합니다. 여러 시점에서의 캡처 데이터를 사용하여 가우시안을 메쉬의 면에 결합하고 이를 통해 동적 객체를 표현합니다.

- **Technical Details**: GSTAR는 가변적인 메쉬 토폴로지를 유지하면서 동적 표면을 추적하는 데 효과적입니다. 변화하는 영역에서는 가우시안을 동적으로 해제하여 새로운 표면을 생성하고, 기존의 메쉬 토폴로지를 따라 계속 유지할 수 있도록 합니다. 또한, 2D 옵티컬 플로우를 활용한 장면 흐름 메서드를 통해 프레임 간의 추적 초기화를 견고하게 진행합니다.

- **Performance Highlights**: 실험 결과, GSTAR는 기존의 최첨단 방법들과 비교하여 외관 측면에서 비슷하거나 더 뛰어난 성능을 보여주었습니다. 이는 고품질 3D 렌더링 애플리케이션에서의 강력한 표현력을 증명하며, VR/XR, 원격 통신 등 다양한 분야에서 활용 가능성을 제공합니다. 더불어, GSTAR는 높은 해상도의 명시적인 3D 표현을 제공하여 로봇공학, 컴퓨터 비전, 컴퓨터 그래픽 등 여러 분야에 기여할 수 있을 것으로 기대됩니다.



### MutualForce: Mutual-Aware Enhancement for 4D Radar-LiDAR 3D Object Detection (https://arxiv.org/abs/2501.10266)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 연구에서는 4D 레이다와 LiDAR의 표현을 상호 강화하는 새로운 프레임워크인 MutualForce를 제안합니다. 기존 4D 레이다와 LiDAR의 융합 방식은 서로의 고유한 강점을 충분히 활용하지 못했습니다. 제안하는 방법은 레이다의 시각 정보와 LiDAR의 기하학적 특성을 통합하여 더욱 향상된 객체 탐지를 가능하게 합니다. 이를 통해 자율주행을 위한 더욱 견고한 인식 시스템을 제공합니다.

- **Technical Details**: MutualForce 프레임워크는 레이다와 LiDAR의 포인트 클라우드를 필라 형태로 매핑하여 시작합니다. 이후, Indicative Radar-Driven Bidirectional 모듈(IRB)을 통해 레이다의 시각 정보를 기반으로 양방향 주의를 기울이며 기하학적 특성을 학습합니다. 마지막으로, Shape Awareness LiDAR-Driven Contrastive 모듈(SALC)을 사용하여 레이다의 BEV 특징을 LiDAR의 기하학적 정보로 enriquecamos 합니다. 다양한 저속 대상을 효과적으로 인식하기 위해 상대 거리 속도와 RCS 정보를 활용합니다.

- **Performance Highlights**: View-of-Delft(VoD) 데이터셋에서 실험한 결과, MutualForce 방법론은 기존 모델에 비해 뛰어난 성능을 보였습니다. 우리는 전체 지역에서 71.76%의 mAP를 달성했으며, 주행 통로 내에서는 86.36%의 mAP를 기록했습니다. 특히 자동차에 대한 AP는 각각 4.17%와 4.20% 향상되어, 제안된 방법론의 효용성을 보여줍니다.



### Disharmony: Forensics using Reverse Lighting Harmonization (https://arxiv.org/abs/2501.10212)
- **What's New**: 이 연구는 깊은 학습 방법을 기반으로한 이미지의 생성 및 편집 탐지 기술을 다룹니다. 특히, 기존 법의학 모델이 간과하는 배경과의 조화 문제를 해결하기 위해 Disharmony Network를 제안합니다. 이를 통해 검증된 이미지 편집 및 생성 방식에 대한 탐지 능력을 향상시키고, 새로운 개체 삽입 후 조화를 위한 신뢰할 수 있는 방법을 소개합니다.

- **Technical Details**: Disharmony Network는 다양한 조화 편집 감지를 위한 특수 모델로, 고유한 세분화 모델과 결합하여 훈련되었습니다. 조화 방법의 집합 데이터 세트를 활용하여 다양한 조화된 객체를 배경과 통합하여 식별하는 데 있어 기존 법의학 네트워크보다 우수한 성능을 보입니다. 또한, 본 연구에 사용된 조화 방법은 신경망 기반, 물리 기반 및 수공예 방법을 포함합니다.

- **Performance Highlights**: 연구 결과는 Disharmony Network이 조화를 위한 세분화 모델과 함께 사용할 때 편집된 이미지 영역을 효과적으로 식별할 수 있음을 나타냅니다. 여러 조화 생성 접근 방식을 통해 훈련된 모델은 기존의 법의학적 방법보다 향상된 성능을 제공합니다. 뿐만 아니라 본 모델은 다양한 형태의 편집을 탐지할 수 있는 가능성을 보여주며, 이를 통해 더 정교한 생성 AI의 도전에 대응할 수 있는 기반을 마련하였습니다.



### Hypercone Assisted Contour Generation for Out-of-Distribution Detection (https://arxiv.org/abs/2501.10209)
- **What's New**: HAC$_k$-OOD는 기존의 OOD 탐지 방식과는 달리, 데이터의 분포에 대한 가정을 두지 않고 자동으로 분포에 적응하는 새로운 방법을 제시합니다. 이 접근 방식은 데이터 포인트의 이웃에 대한 각도를 극대화하여 하이퍼콘(hypercone)이라는 다차원 구조를 생성함으로써 ID 데이터가 위치하는 영역을 근사합니다. 이 논문은 HAC$_k$-OOD가 CIFAR-100 벤치마크에서 Near-OOD 및 Far-OOD에 대해 최첨단 성능을 달성했음을 보여줍니다.

- **Technical Details**: HAC$_k$-OOD는 OOD 탐지를 위해 훈련 기반 및 포스트 프로세싱(distance-based) 방법을 통합하는 독창적인 방법입니다. 이는 입력 특징 공간에서의 ID와 OOD 데이터를 분리하는 하이퍼콘을 구축하여, 데이터에 대한 확률적인 가정을 두지 않고 OOD 탐지를 가능하게 합니다. 이 방법은 클래식한 분류 문제를 수학적으로 정의하며, 영향을 미치는 각 개체의 다양성을 분석하여 ID와 OOD 데이터를 분리합니다.

- **Performance Highlights**: HAC$_k$-OOD는 Supervised Contrastive Learning을 활용하여 CIFAR-100 데이터 세트에서 Far-OOD 및 Near-OOD 탐지에서 최첨단 성능을 달성했습니다. CIFAR-10 벤치마크 데이터셋과의 비교를 통해 이 방법은 다른 최신 방법들과 동등한 성능을 보여줍니다. Supervised Cross Entropy 실험 결과, HAC$_k$-OOD는 해당 손실 함수로 훈련된 모델에서도 경쟁력을 입증합니다.



### Adaptive Clustering for Efficient Phenotype Segmentation of UAV Hyperspectral Data (https://arxiv.org/abs/2501.10199)
Comments:
          accepted WACV 2025 GeoCV workshop

- **What's New**: 이번 논문에서는 무인 비행기(UAV)와 하이퍼스펙트럼 이미징(Hyperspectral Imaging, HSI)을 결합하여 실시간 나무 표현형 분할(tree phenotype segmentation)을 위한 온라인 하이퍼스펙트럼 간단 선형 반복 클러스터링 알고리즘(Online Hyperspectral Simple Linear Iterative Clustering, OHSLIC) 프레임워크를 소개합니다. OHSLIC는 적응형 점진적 클러스터링과 경량 신경망을 활용하여 노이즈를 줄이고 계산 요구사항을 감소시킵니다. 또한, 이 논문에서는 현실적인 잎 매개변수를 포함한 사용자 지정 시뮬레이터를 사용하여 하이퍼스펙트럼 데이터셋을 생성하였습니다.

- **Technical Details**: 이 연구에서 적용된 하이퍼스펙트럼 시뮬레이터는 Blender를 활용하여 환경을 렌더링하고 PROSPECT 잎 모델을 사용하여 현실적인 스펙트럼과 매개변수를 생성합니다. OHSLIC 알고리즘은 나무의 엽록소(chlorophyll), 카로티노이드(carotenoids), 안토시아닌(anthocyanins)과 같은 정보를 통해 최적의 표현형 이미지를 산출합니다. 이는 자원 제한적인 환경에서의 도입을 가능하게 합니다.

- **Performance Highlights**: OHSLIC 알고리즘은 픽셀 또는 윈도우 기반 방법들보다 우수한 회귀 정확성과 세분화 성능을 달성하며, 추론 시간을 현저히 줄입니다. 이 알고리즘은 동적 클러스터링을 통해 계산 효율성과 정확성 사이의 균형을 조절할 수 있도록 하여, 하이퍼스펙트럼 이미징 응용 프로그램에서도 확장 가능한 엣지 장치 배포가 가능하게 합니다.



### CSHNet: A Novel Information Asymmetric Image Translation Method (https://arxiv.org/abs/2501.10197)
- **What's New**: 이번 논문에서는 정보 비대칭적인 이미지 변환 과제를 해결하기 위해 CNN과 Transformer를 통합한 새로운 하이브리드 네트워크인 CNN-Swin Hybrid Network (CSHNet)를 제안합니다. 이 네트워크는 Swin Embedded CNN (SEC)과 CNN Embedded Swin (CES)라는 두 가지 모듈로 구성되어 있으며, 정보 손실을 줄이면서 구조적 일관성을 유지하는 방법론을 포함합니다. 또한, Interactive Guided Connection (IGC)와 Adaptive Edge Perception Loss (AEPL)을 통해 다양한 데이터를 효과적으로 변환할 수 있도록 합니다.

- **Technical Details**: CSHNet은 특히 구조적 바이어스와 세부 정보 추출을 결합하여, 이미지의 세부 정보와 구조적 정보를 동시에 고려합니다. SEC 모듈은 CNN을 기반으로 하여 상세한 특징을 추출하고, CES 모듈은 Swin 트랜스포머의 전반적인 구조를 보존합니다. 이러한 하이브리드 접근 방식은 지역적 정확성과 전역적 추론의 균형을 유지하도록 설계되어, 정보 비대칭적인 이미지 변환 작업에서 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, CSHNet은 시각적 품질과 성능 지표에서 기존 방법들보다 우수한 결과를 보여주었습니다. CNN과 Transformer의 조합으로 얻은 혁신적인 성능 향상은 CSHNet이 다양한 장면 수준과 인스턴스 수준 데이터셋에서 효과적임을 입증합니다. 또한, CSHNet은 정보 손실과 지역 혼합 현상을 줄여주며, 이를 통해 생성된 이미지의 세부 사항이 온전하게 유지되는 것을 관찰할 수 있습니다.



### Structure-guided Deep Multi-View Clustering (https://arxiv.org/abs/2501.10157)
- **What's New**: 이 논문에서는 제안된 구조 유도 깊이 다중 뷰 클러스터링 모델(SGMVC)을 통해 기존의 다중 뷰 클러스터링 방법의 한계를 극복하고자 합니다. SGMVC는 이웃 관계에 기반한 긍정 샘플 선택 전략과 해당 손실 함수를 도입하여 다중 뷰 데이터의 지역 구조 정보를 효과적으로 활용합니다. 또한, 잠재적 구조 정보를 발견하기 위해 가우시안 분포 모델을 적용하였으며, 이를 통해 다중 뷰 간의 일관성을 높이고 클러스터 내 응집성을 증가시킵니다.

- **Technical Details**: SGMVC는 크게 세 가지 모듈로 구성됩니다: 다중 뷰 인코더-디코더 모듈, 지역 구조 학습 모듈, 그리고 임베딩 구조 학습 모듈입니다. 인코더-디코더 모듈은 각 뷰로부터 잠재 표현을 학습하고, 지역 구조 학습 모듈은 정제된 교차 뷰 일관성 이웃 선택 전략을 도입하여 신뢰성 있는 긍정 샘플 쌍을 증가시킵니다. 임베딩 구조 학습 모듈은 가우시안 분포를 통해 임베딩 공간 내에서의 구조 정보를 최적화하여, 더욱 일관된 구조 정렬을 달성합니다.

- **Performance Highlights**: 분석된 실험 결과에 따르면, 제안된 SGMVC는 5개의 다중 뷰 데이터셋에서 이전의 최첨단 다중 뷰 클러스터링 방법들과 비교하여 현저한 성능 향상을 나타냈습니다. 이 성능 개선은 지역 및 잠재 구조 정보를 효과적으로 탐색한 결과입니다. 본 모델은 복잡한 클러스터링 작업에 대한 잠재력을 뒷받침하며, 향후 다양한 분야에서의 적용 가능성을 기대하게 합니다.



### A Vision-Language Framework for Multispectral Scene Representation Using Language-Grounded Features (https://arxiv.org/abs/2501.10144)
- **What's New**: Spectral LLaVA는 multispectral 데이터를 통합하여 복잡한 환경을 더 잘 이해할 수 있는 새로운 비전-언어 프레임워크입니다. 기존의 RGB 중심 모델에서 벗어나, 본 연구에서는 이후 Multispectral Domain에 적용 가능한 방법론을 제시하고 있습니다. 이 프레임워크는 SpectralGPT의 비전 백본을 고정한 상태로 가벼운 선형 프로젝션 레이어를 최적화하여 언어와 시각적 기능의 정렬을 향상시킵니다.

- **Technical Details**: Spectral-LLaVA는 SpectralGPT의 인코더를 활용하여 multispectral 이미지를 처리하고, Datasets는 BigEarthNet에서 수집된 데이터를 기반으로 합니다. 이 모델은 사전 훈련된 SpectralGPT의 기능을 사용하여 시각적 특성을 추출하고, 이를 LLaMA3 언어 모델과 결합하여 효과적인 classification과 scene description을 수행합니다. 모델은 LoRA 기반의 파라미터 적응을 통해 fine-tuning되며, Adam optimizer를 통해 최적화됩니다.

- **Performance Highlights**: 실험 결과에 따르면, Spectral LLaVA는 RGB 데이터만으로는 충분히 설명할 수 없는 복잡한 환경에서도 뛰어난 설명 능력을 발휘합니다. 또한, multispectral 정보를 포함함으로써 이미지 분류 성능도 크게 향상되었음을 보여주었습니다. 이 연구는 label grounding을 통한 semantic richness를 제공하여 multispectral 이미지 분석의 가능성을 열었습니다.



### ACE: Anatomically Consistent Embeddings in Composition and Decomposition (https://arxiv.org/abs/2501.10131)
Comments:
          Accepted by WACV 2025

- **What's New**: 이 논문은 의료 이미지를 위한 새로운 자기 지도 학습(Self-Supervised Learning, SSL) 접근법인 ACE를 제안합니다. 이 방법은 해부학적으로 일관된 임베딩(embedding)을 학습하기 위해 구성(composition) 및 분해(decomposition)의 두 가지 주요 가지를 기반으로 합니다. 전체적인 일관성을 강조하는 글로벌(global) 특성과 세부적인 해부학적 정보를 학습하는 로컬(local) 특성을 결합하여, 기존의 SSL 방법들이 간과한 의료 이미지 고유의 복잡한 구성을 이해합니다.

- **Technical Details**: ACE는 그리드 단위 이미지 잘라내기(grid-wise image cropping) 기법을 활용하여 정확한 패치 매칭(patch matching)을 수행합니다. 이를 통해 서로 겹치는 두 개의 랜덤 잘려진 뷰를 입력하여, 전체 이미지의 의미를 놓치지 않으면서 독립적으로 특성을 학습합니다. 또한, ACE는 학생-교사 모델(student-teacher model)을 이용하여 이미지 내 부분-전체 관계의 상관성을 반영하며, 정밀한 패치 매칭을 통해 로컬 일관성을 최적화합니다.

- **Performance Highlights**: ACE는 6개의 데이터세트에서 2개의 백본(backbone)을 사용하여 몇 가지 출력 작업에서 연구되었고, 우수한 강건성, 전이 가능성을 보여줍니다. 이 연구 결과는 ACE가 기존의 의료 및 사진 이미지에 특화된 비전 SSL 방법들보다 뛰어난 성능을 발휘함을 보여주었습니다. 특히, 팬더스 사진(fundus photography)처럼 표준화된 이미징 프로토콜에서 수집한 기타 이미지에 대한 적응 능력을 충실히 나타내었습니다.



### Spatio-temporal Graph Learning on Adaptive Mined Key Frames for High-performance Multi-Object Tracking (https://arxiv.org/abs/2501.10129)
- **What's New**: 이번 논문에서는 다중 객체 추적(Multi-Object Tracking)의 기존 한계를 극복하기 위해 적응형 키 프레임 마이닝 전략을 제안합니다. 이를 통해 강화 학습(Reinforcement Learning)을 활용하여 비디오를 동적으로 분할하고, 객체 간의 공간적 및 시간적 관계를 효과적으로 캡처할 수 있게 되었습니다. 또한, Intra-Frame Feature Fusion(IFF) 모듈을 도입하여 서로 다른 객체 간의 정보 교환을 촉진합니다.

- **Technical Details**: 키 프레임 추출(KFE) 모듈은 Q-learning에 기반해 설계되며, 최적의 분할 전략과 보상 메커니즘을 통해 비디오 세그먼트를 적응적으로 나누는 역할을 합니다. 이 과정에서 GCN(Graph Convolutional Network)을 사용하여 한 프레임 내의 객체 및 주변 객체 간의 정보 상호작용을 극대화하고, 이를 통해 객체 별로 고유 식별력을 높입니다. 이러한 기법들은 기존의 그래프 기반 방법과는 다른 접근 방식을 보여줍니다.

- **Performance Highlights**: 제안된 알고리즘은 MOT17 데이터셋에서 68.6 HOTA, 81.0 IDF1, 66.6 AssA 및 893 IDS의 성능을 기록하며 효과성과 정확성을 입증합니다. 전략적인 단기 및 장기 연관성을 모두 모델링하는 통합된 트래커로, 다양한 시나리오에 적응할 수 있는 포괄적인 해결책을 제공합니다. 실험 결과, 제안된 방식이 기존 방법들에 비해 개선된 성능을 보여주는 것으로 나타났습니다.



### DiffVSR: Enhancing Real-World Video Super-Resolution with Diffusion Models for Advanced Visual Quality and Temporal Consistency (https://arxiv.org/abs/2501.10110)
Comments:
          Project page: \url{this https URL}

- **What's New**: DiffVSR는 실제 세계 비디오 슈퍼 해상도에서의 새로운 접근 방식으로, 이 모델은 고해상도 비디오 프레임을 저해상도에서 재구성하는 데 있어 기존의 방식들이 겪는 난제를 해결하는 혁신적인 요소들을 포함하고 있습니다. 이를 통해 DiffVSR는 고품질 결과물과 일관된 시간적 특성을 유지하면서 뛰어난 시각적 품질을 달성할 수 있습니다. 특히, 다중 스케일 시간 주의 모듈과 시간 강화 VAE 디코더를 통해 비디오 내 일관성을 크게 개선했습니다.

- **Technical Details**: DiffVSR는 입력 비디오의 복잡한 열화 문제를 처리하는 동시에 고해상도 텍스처를 생성하고 시간적 일관성을 유지하기 위한 핵심 구성 요소를 통합하고 있습니다. 이 모델은 InternLM-XComposer2를 활용하여 시각적 세부 사항과 의미 정보를 포착하는 설명 텍스트 주석을 자동으로 생성합니다. 이를 통해 복원 과정에서 모델이 정밀한 텍스처를 생성할 수 있도록 합니다. 또한, 일관성 유지를 위한 내부 및 외부 시퀀스를 효과적으로 개선하는 다양한 메커니즘이 포함되어 있습니다.

- **Performance Highlights**: DiffVSR는 기존의 방법들보다 시각적 품질과 시간적 일관성 모두에서 우수한 성능을 보여줍니다. 실험 결과, 이 모델은 비디오 세분화와 텍스처 복원에서 뛰어난 성능을 발휘하며, 실제 비디오 슈퍼 해상도 분야에서 새로운 기준을 세웠습니다. 이를 통해 DiffVSR는 다양한 변형을 처리하며 안정적인 학습과 최적화를 가능하게 합니다.



### landmarker: a Toolkit for Anatomical Landmark Localization in 2D/3D Images (https://arxiv.org/abs/2501.10098)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문에서는 의료 이미지에서 해부학적 랜드마크(localization) 식별의 정확도를 향상시키기 위해 개발된 새로운 Python 패키지 'landmarker'를 소개합니다. 이 패키지는 PyTorch를 기반으로 하며, 다양한 이미지를 위한 전처리 파이프라인과 여러 방법론을 지원하여 연구 및 개발 프로세스를 간소화합니다. 또한, 랜드마크 식별의 정확성을 높이며, 특히 의료 영상 분야에서 중요한 외과 절차 및 진단을 위한 필수 도구입니다.

- **Technical Details**: 'landmarker'는 두 가지 주된 방법론인 heatmap regression과 coordinate regression을 포함합니다. 이 패키지는 다양한 데이터셋과 이미지 포맷(NIfTI, DICOM 등)을 지원하며, PyTorch의 데이터셋 클래스를 상속하여 유연한 데이터 로딩 기능을 제공합니다. 사용자는 static 및 adaptive heatmap regression 접근 방식을 통해 랜드마크를 학습하고, 여러 복잡한 문제를 해결할 수 있는 도구를 갖추게 됩니다.

- **Performance Highlights**: landmarker는 높은 정확도 및 사용자 맞춤형 구현 가능성을 통해 연구자와 실무자들에게 최신 알고리즘을 적용할 수 있게 돕습니다. 기존의 pose estimation 도구들은 의료 영상의 고유한 요구사항을 충족하지 못하는 반면, landmarker는 모듈식 구성을 통해 다양한 낯선 알고리즘을 실험하고 구현할 수 있는 유연성을 제공합니다.



### Classifier Ensemble for Efficient Uncertainty Calibration of Deep Neural Networks for Image Classification (https://arxiv.org/abs/2501.10089)
Comments:
          This paper has been accepted at International Conference on Computer Vision Theory and Applications (VISAPP), 2025

- **What's New**: 이 논문에서는 다양한 심층 신경망에 적용하여 불확실성 조정을 위한 새로운 분류기 앙상블 기법을 조사합니다. 특히, 본 연구는 일반적인 딥러닝 모델에서 자주 발생하는 과신(overconfidence) 문제를 해결하고 ECE(기대 보정 오차) 및 MCE(최대 보정 오차)를 개선하기 위한 다양한 방법을 비교합니다. 또한, 메타모델 기반의 앙상블 기법이 전통적인 모델 앙상블 방식보다 보정 성능이 뛰어난 것을 발견했습니다.

- **Technical Details**: 연구에서는 기본 앙상블 기법(majority voting)과 메타모델 기반 접근법을 포함하여 여러 가지 간단하지만 효율적인 분류기 앙상블 방법을 평가합니다. 특히, 제안된 방법은 경량(classifier) 분류기를 공통의 백본(backbone)을 기반으로 교육하고, 이들의 예측을 협력적으로 이용함으로써 기존 기법들보다 컴퓨팅 비용이 적고 추가적인 보정 데이터 세트를 필요로 하지 않습니다. 이러한 방식은 ECE와 MCE를 일관되게 감소시키는 성과를 보여줍니다.

- **Performance Highlights**: 연구 결과, 기본 앙상블 기법이 꽤 괜찮은 개선을 제공하는 반면, 메타모델 기반의 앙상블 기법은 모든 아키텍처에서 ECE와 MCE를 지속적으로 감소시킴을 알 수 있었습니다. 비교한 메타모델 중 가장 큰 모델이 높은 보정 개선을 보이면서도 정확도에는 최소한의 영향을 미쳤습니다. 따라서, 제안된 방법은 딥러닝 시스템의 신뢰성을 높이는 효과적인 접근법으로 자리잡을 가능성이 큽니다.



### Leveraging Confident Image Regions for Source-Free Domain-Adaptive Object Detection (https://arxiv.org/abs/2501.10081)
- **What's New**: 이번 논문은 소스 데이터 없이 객체 감지 모델을 타겟 도메인에 적응시키는 새로운 데이터 증강(data augmentation) 접근법을 제시합니다. 기존의 소스 데이터 의존성 문제를 해결하기 위해, 저자는 확신이 높은 타겟 이미지 영역을 선택하고 해당 영역의 의사 레이블(pseudo-labels)과 함께 증강하여 도전적인 타겟 이미지를 합성합니다. 이를 통해 모델의 붕괴 현상(collapse)을 방지하는 교사-학생(teacher-student) 학습 패러다임을 적용합니다.

- **Technical Details**: 제안된 접근 방식인 SF-DACA는 소스 도메인에서 학습된 지식을 활용하여 타겟 도메인에서의 고품질 예측을 도출하고, 이를 의사 레이블로 취급하여 타겟 도메인에 대한 지식을 증가시킵니다. 이 방법은 교사 모델과 학생 모델을 활용하여 학생 모델의 지식을 정적 교사의 예측과 일치시키는 방식으로 진행됩니다. 이를 통해 소스 데이터 없이도 효과적인 적응이 가능하도록 합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 교통 장면 적응 벤치마크에서 평가되었으며, 이 중 두 가지에서 새로운 최첨단 성능을 달성했습니다. 이 논문은 기존의 작업과 비교하여 SF-UDA 분야에서의 데이터 증강 가능성을 보여주었습니다. 더 나아가, 학생의 가중치를 점진적으로 상속함으로써 교사의 지식을 타겟에 대해 강화하여 더 나은 성능을 구현했습니다.



### Few-shot Structure-Informed Machinery Part Segmentation with Foundation Models and Graph Neural Networks (https://arxiv.org/abs/2501.10080)
Comments:
          Accepted at Winter Conference on Applications of Computer Vision (WACV) 2025. Code and available at this https URL

- **What's New**: 이 논문은 기계의 여러 부품을 다루는 few-shot semantic segmentation을 위한 새로운 접근 방식을 제안합니다. CLIPSeg와 Segment Anything Model(SAM), SuperPoint 및 그래프 CNN(Graph Convolutional Network)을 통합하여 기계 부품을 정확히 분할하는 방법을 제시하고 있습니다. 1~25개의 주석이 달린 샘플 제공으로, 트럭 장착 로딩 크레인을 모델로 하여 다양한 수준의 세부 사항에서 효과적으로 분할할 수 있음을 보여줍니다.

- **Technical Details**: 주요 기술은 CLIPSeg와 SAM, SuperPoint 및 GCN을 결합한 구조-informed(구조 정보 기반) 파트 분할입니다. 이 시스템은 consumer GPU에서 5분 이하의 훈련 시간을 유지하며, 10개의 합성 지원 샘플을 사용하여 실제 데이터에서 92.2의 $J	ext{&}F$ 점수를 달성합니다. SAM은 바운딩 박스나 키포인트의 안내가 필요하여, 이미지 노드를 클래스를 기반으로 쿼리하는 자동 prompt engineering 절차를 개발하였습니다.

- **Performance Highlights**: 자체적으로 평가한 DAVIS 2017 데이터셋에서 세미-수퍼바이즈드 비디오 분할에서 3개의 지원 샘플로 $J	ext{&}F$ 점수 71.5를 기록했습니다. 이 모델은 높은 일반화 능력을 보여주며, 기계 및 인프라와 상호 작용하는 자율 시스템에 유용한 도구로 자리 잡았습니다. 다양한 조명 환경에서 훈련된 합성 데이터 세트를 이용함으로써, 훈련 데이터와 테스트 데이터 간의 다양성을 높이는 데 성공했습니다.



### Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework (https://arxiv.org/abs/2501.10075)
Comments:
          This work has been submitted to the IEEE Transactions on Geoscience and Remote Sensing journal for possible publication

- **What's New**: 본 연구는 고해상도 RGB 이미지 쌍과 의미론적 분할 맵을 포함하는 새로운 RSICC 데이터셋인 SECOND-CC를 소개합니다. 이 데이터셋은 6,041 쌍의 바이템포랄 이미지와 30,205개의 문장을 포함하여 이미지 간의 차이를 설명합니다. 또한, MModalCC라는 멀티모달 프레임워크를 제안하여 고급 주의 메커니즘을 통합하여 변화 설명의 정확도를 높입니다.

- **Technical Details**: SECOND-CC 데이터셋은 다양한 실제 시나리오를 반영하여 다양한 이미지 해상도 및 등록 오류 문제를 해결하도록 설계되었습니다. MModalCC는 Cross-Modal Cross Attention (CMCA) 및 Multimodal Gated Cross Attention (MGCA) 메커니즘을 통해 시멘틱 맵과 RGB 이미지로부터 의미 있는 특징을 통합하고 있습니다. 이후 실시된 세부적인 아블레이션 연구 및 주의 시각화는 이러한 접근 방식의 효과를 입증했습니다.

- **Performance Highlights**: MModalCC는 기존 RSICC 최신 방법인 RSICCformer, Chg2Cap, PSNet에 비해 BLEU4 점수에서 4.6%, CIDEr 점수에서 9.6%의 향상을 보여주었습니다. 이러한 성과는 MModalCC의 혁신적인 설계와 데이터셋의 품질 덕분이며, 향후 연구를 위한 오픈 소스 제공도 계획하고 있습니다.



### CLIP-PCQA: Exploring Subjective-Aligned Vision-Language Modeling for Point Cloud Quality Assessmen (https://arxiv.org/abs/2501.10071)
- **What's New**: 이 논문에서는 CLIP-PCQA라는 새로운 No-Reference Point Cloud Quality Assessment(PCQA) 방법을 제안합니다. 기존 방법들이 시각 데이터를 평균 의견 점수(Mean Opinion Score, MOS)로 직접 매핑하려 했던 반면, CLIP-PCQA는 언어 기반 접근 방식을 통해 결과의 주관성을 더욱 잘 반영합니다. 이 방법은 여러 품질 설명에 대한 코사인 유사성을 계산하고, 효과적인 대조 손실(contrastive loss)과 학습 가능한 프롬프트(prompts)를 도입하여 특징 추출을 향상시킵니다.

- **Technical Details**: CLIP-PCQA는 3D 포인트 클라우드에서 시각 및 텍스트 특징을 추출하고, 서로 다른 품질 설명에 해당하는 여러 텍스트 특징과의 코사인 유사도를 계산합니다. 이 과정은 품질 설명을 기반으로 한 확률로 변환되어, Opinion Score Distribution(OSD)를 생성합니다. 이는 주관적 실험에서의 응답자의 개인적인 편향을 줄여줘 보다 정확한 품질 평가를 가능하게 합니다. 이 방법은 CLIP(Radford et al. 2021)의 철학에 근거하여, 3D 포인트 클라우드를 2D 색상 및 깊이 맵으로 투영하여 특징을 추출합니다.

- **Performance Highlights**: 실험 결과, CLIP-PCQA는 기존의 State-Of-The-Art(SOTA) 접근 방식들을 초월하는 성능을 보여줍니다. 다수의 벤치마크에서 수행된 실험들은 CLIP-PCQA의 robustness(강건성)과 다양한 설정에서의 효율성을 나타냅니다. 또한, 품질 설명을 이용하여 사람들의 직관적인 인식을 잘 반영하고 있음을 입증했습니다.



### FiLo++: Zero-/Few-Shot Anomaly Detection by Fused Fine-Grained Descriptions and Deformable Localization (https://arxiv.org/abs/2501.10067)
- **What's New**: 본 논문은 FiLo++라는 새로운 방법론을 제안합니다. 이 방법은 두 가지 주요 구성 요소, 즉 Fused Fine-Grained Descriptions (FusDes)와 Deformable Localization (DefLoc)으로 구성되어 있습니다. FusDes는 대규모 언어 모델을 활용하여 각 객체 카테고리에 대한 보다 정확하고 작업 특정적인 이상치 설명을 생성하며, DefLoc은 다양한 형태와 크기의 이상치를 정확히 위치 표시하도록 돕습니다.

- **Technical Details**: FusDes 모듈은 고정된 템플릿과 학습 가능한 템플릿을 결합해 더 잘 맞춘 텍스트 프롬프트를 생성합니다. DefLoc 모듈은 초기 이상치 위치 확인을 위해 Grounding DINO 모델을 사용하며, 다중 스케일 변형 가능한 교차 모달 상호 작용(MDCI) 모듈을 통해 다양한 크기와 형태의 이상치 지역을 찾아냅니다. 추가적으로, 이 방법은 위치 향상 패치 매칭 접근법도 설계하여 몇 샷 이상치 탐지가 가능합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, FiLo++는 기존 방법들에 비해 상당한 성능 향상을 이루었습니다. 예를 들어, VisA 데이터셋의 제로샷 시나리오에서 FiLo++는 이미지 수준 AUC 84.5%와 픽셀 수준 AUC 96.2%를 달성하였습니다. 이러한 결과는 제로샷 및 몇 샷 이상치 탐지 및 위치 지정을 모두 효과적으로 수행할 수 있음을 보여줍니다.



### One-D-Piece: Image Tokenizer Meets Quality-Controllable Compression (https://arxiv.org/abs/2501.10064)
Comments:
          Our Project Page: this https URL

- **What's New**: 본 연구에서 제안된 One-D-Piece는 기존의 고정 길이 토큰화 방법을 초월하여 가변 길이 이미지 토큰화를 지원하는 혁신적인 방법입니다. 이 방법은 'Tail Token Drop'이라는 규제 기법을 통해 핵심 정보를 토큰 시퀀스의 시작 부분에 집중시켜, 재구성 품질을 유지하면서도 효율성을 극대화할 수 있습니다. 이를 통해 다양한 애플리케이션에서 필요한 유연한 압축이 가능해집니다.

- **Technical Details**: One-D-Piece는 입력 이미지의 정보를 효과적으로 캡처하기 위해 이미지의 해상도와 토큰의 수를 동적으로 조정할 수 있도록 고안되었습니다. 이 기법은 Transformer 기반 모델과의 통합을 간소화하고 연산 복잡성을 줄이는 데 기여합니다. 또한, 기존의 이미지 압축 알고리즘과 달리, 이 모델은 Neural Networks에서 직접 활용 가능한 방식으로 설계되었습니다.

- **Performance Highlights**: One-D-Piece는 낮은 토큰 수에서도 높은 재구성 품질을 유지하며 다양한 다운스트림 작업에서 우수한 성능을 보여주었습니다. 특히 이미지 분류, 객체 검출, 의미론적 분할 및 깊이 추정과 같은 여러 컴퓨터 비전 작업에서 기존의 이미지 압축 방법보다 뛰어난 성능을 발휘했습니다. 이러한 결과는 One-D-Piece가 다양한 애플리케이션에 적합한 유연성을 갖고 있음을 시사합니다.



### LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks (https://arxiv.org/abs/2501.10040)
Comments:
          12 pages, 8 figures, Remote sensing

- **What's New**: 이 논문에서는 원거리 센싱(RS) 비주얼 작업을 위해 설계된 LWGANet이라는 경량 백본 네트워크를 소개합니다. LWGANet은 새로운 Lightweight Group Attention (LWGA) 모듈을 통합하여 비슷한 조건의 다양한 물체를 효과적으로 처리하는 능력을 갖추고 있습니다. 이 모듈은 다중 스케일의 객체에서 공간 정보를 효과적으로 추출하며, 이는 정밀한 특징 추출을 가능하게 합니다.

- **Technical Details**: LWGA 모듈은 입력된 피처 맵을 여러 하위 모듈로 나누어 각 스케일에서의 특징 추출과 경량 계산 효율성 간의 균형을 유지합니다. 다양한 피처 추출 방식을 포함하여, point attention, local attention, medium-range attention, global attention을 통해 대규모 변화에 노출된 객체의 특징을 정제하여 최적화된 피처 표현을 제공합니다. 이와 더불어, 문서에서는 sparsity 기술을 접목하여 경제적인 자원에서의 복잡성을 회피하도록 설계되었습니다.

- **Performance Highlights**: LWGANet은 12개의 데이터셋을 통해 검증되었으며, 장면 분류, 방향성 객체 탐지, 의미론적 세분화, 변화 탐지와 같은 네 가지 주요 RS 비주얼 작업에서 SOTA(최고 성능) 결과를 달성했습니다. LWGANet은 고성능과 저복잡성 간의 최적 균형을 유지하며, 자원 제한 환경에서도 강력한 RS 이미지 처리 기능을 제공하는 새로운 솔루션으로 자리 매김했습니다.



### X-Dyna: Expressive Dynamic Human Image Animation (https://arxiv.org/abs/2501.10021)
Comments:
          Project page:this https URL Code:this https URL

- **What's New**: X-Dyna는 인간 이미지 애니메이션을 위한 새로운 제로샷(zero-shot) 확산 기반(diffusion-based) 파이프라인입니다. 이 접근법은 다른 사람의 드라이빙 비디오에서 파생된 신체 움직임과 얼굴 표정을 사용하여 단일 인간 이미지를 애니메이션화합니다. X-Dyna는 기존의 인간 포즈 제어 방법의 단점을 보완하여 동적 세부 사항을 잃지 않도록 하고, 생생한 인간 비디오 애니메이션의 사실성을 향상시킵니다.

- **Technical Details**: X-Dyna의 핵심은 Dynamics-Adapter라는 경량 모듈로, 참조 외관(context of appearance)을 확산(backbone)의 공간적 주의(attention)에 통합합니다. 이 모듈은 모션 모듈의 유동적이고 복잡한 동적 세부 사항 생성 능력을 유지하면서도 동적 디테일의 손실을 방지합니다. 또한, 로컬 제어 모듈을 도입해 개체 분리(문맥 단절) 표정을 캡처하여 정확한 표정 전이를 통해 애니메이션 장면의 사실성을 높입니다.

- **Performance Highlights**: X-Dyna는 900시간의 인간 춤 및 자연 장면 비디오로 학습되어 우수한 포즈 및 표정 전이와 함께 사실적인 인간과 장면 동적 생성을 수행합니다. 다양한 벤치마크에서 평가한 결과, X-Dyna는 최첨단 방법들보다 양적 및 질적으로 뛰어난 성능을 보여주며, 생동감 있는 동적 표현력과 시각적 품질에서 우수함을 입증했습니다.



### Textoon: Generating Vivid 2D Cartoon Characters from Text Descriptions (https://arxiv.org/abs/2501.10020)
- **What's New**: 이 기술 보고서에서는 Textoon이라는 혁신적인 방법을 소개하여 텍스트 설명을 기반으로 다양한 2D 만화 캐릭터를 Live2D 포맷으로 생성하는 방식을 제시합니다. Textoon은 첨단 언어(large language models) 및 비전 모델(vision models)을 활용하여 문자 의도를 이해하고, 1분 이내에 놀랍고 상호작용이 가능한 2D 캐릭터를 생성할 수 있습니다. 이는 기존의 수작업 과정 없이 가능한 연속적인 생성 프로세스를 통해 이루어집니다.

- **Technical Details**: Textoon은 다양한 Live2D 모델을 생성하기 위해 텍스트 설명을 파싱(p parsing)하여 사용자의 입력에 대한 세부 정보를 정확하게 추출하는 기능을 갖추고 있습니다. 각 구성 요소는 종합적인 캐릭터 템플릿으로 합성되며, 색상 및 질감은 text-to-image 모델에 의해 결정됩니다. 또한, 사용자가 생성된 결과에 불만족할 경우 특정 세부 사항을 수정할 수 있는 기능도 포함되어 있습니다.

- **Performance Highlights**: Textoon은 1분 이내에 사용자 맞춤형 2D 캐릭터를 생성할 수 있도록 도와주며, 이 과정에서 정확도는 90% 이상을 기록합니다. 특히, Live2D 모델의 얼굴 애니메이션 메커니즘을 리팩토링하여 캐릭터의 얼굴 표현력을 크게 향상시켰습니다. 이 기술은 특히 모바일 및 웹 애플리케이션과 같은 제한된 처리 능력을 가진 기기에서도 효율적으로 작동할 수 있도록 설계되었습니다.



### DiffuEraser: A Diffusion Model for Video Inpainting (https://arxiv.org/abs/2501.10018)
Comments:
          11pages, 13figures

- **What's New**: DiffuEraser는 최신의 비디오 인페인팅 모델로, 안정적인 확산 모델(stable diffusion) 기반의 영상 생성 기술을 활용해 세밀하고 일관된 구조로 마스크된 영역을 채우는 것을 목표로 합니다. 기존의 접근 방식이 큰 마스크를 처리할 때 모호함이나 시간적 불일치 문제를 겪는 반면, 이 모델은 향상된 생성 능력을 통해 이러한 문제를 완화합니다. 또한, 초기화를 위한 사전 정보와 약한 조건을 도입하여 노이즈 아티팩트를 줄이고 환각 현상을 억제합니다.

- **Technical Details**: DiffuEraser는 BrushNet을 기반으로 한 영상 인페인팅 모델로, 이미지 인페인팅 구성 요소에 동작 모듈을 통합하여 마스킹된 이미지의 특징을 추출합니다. 모델 아키텍처는 기본적인 노이즈 제거 UNet과 보조적 BrushNet으로 구성되며, BrushNet은 마스킹된 이미지와 노이즈 있는 잠재 변수를 사용하여 조건부 입력을 처리합니다. 또한, 시간적 일관성을 개선하기 위해 자가 주의(self-attention) 및 교차 주의(cross-attention) 후속 레이어에 시간적 주의 메커니즘을 도입합니다.

- **Performance Highlights**: 실험 결과, DiffuEraser는 콘텐츠 완전성과 시간적 일관성 모두에서 최신 기법들을 능가하며, 효율성도 수용 가능한 수준을 유지합니다. 이 모델은 오래된 시퀀스 추론을 통해도 일관된 결과를 생성하며, 필요한 디테일과 구조적 완전성을 보장합니다. 전반적으로 DiffuEraser는 비디오 인페인팅의 새로운 가능성을 열어주는 혁신적인 접근 방식으로 평가됩니다.



### Mitigating Hallucinations on Object Attributes using Multiview Images and Negative Instructions (https://arxiv.org/abs/2501.10011)
Comments:
          2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 본 논문은 현재 인기 있는 Large Vision-Language Models (LVLMs)가 겪고 있는 Hallucinations on Object Attributes (HoOA) 문제를 해결하기 위한 새로운 방법을 제안합니다. 3D 생성 기술의 발전을 활용하여, 단일 이미지에서 생성된 3D 표현으로부터 샘플링된 다중 관점(multiview) 이미지를 LVLMs의 시각적 프롬프트로 사용함으로써, 보다 다양한 시각 정보를 제공합니다. 이 접근 방식은 LVLM의 성능 향상에 기여합니다.

- **Technical Details**: 제안된 방법은 Multiview Image Augmented VLM (MIAVLM)으로, 여기에는 Multiview Attributes Perceiver (MAP) 서브모듈이 포함되어 있어 입력 이미지 순서의 영향을 제거하면서 다중 관점 이미지에서 시각 정보를 Large Language Models (LLMs)와 정렬합니다. 이를 통해 LVLM의 성능을 더욱 향상시키며, 특히 다중 관점 이미지가 LVLMs에서 수행하는 세부 속성 인식의 정확성을 높이는 데 도움을 줍니다. 또한, 부정적 지침(negative instructions)을 도입하여 LVLMs가 "예" 응답에 편향되는 문제를 완화하고자 하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 LVLMs의 비정확한 속성 판단 문제를 효과적으로 개선하는 것을 보여주었습니다. 다양한 실험을 통해 MIAVLM은 기존 모델 대비 성능에서 우수한 결과를 나타냈으며, 다중 관점 이미지가 LVLMs의 인식 능력을 강화하는 데 중요한 역할을 함을 입증하였습니다. 이러한 결과는 LVLMs의 활용 가능성을 더욱 확대하는 데 기여할 것으로 기대됩니다.



### Deep Learning for Early Alzheimer Disease Detection with MRI Scans (https://arxiv.org/abs/2501.09999)
- **What's New**: 이번 연구는 알츠하이머병(Alzheimer's Disease, AD)의 진단 정확성을 향상시키기 위해 심층 학습 모델의 비교를 다루고 있습니다. 특히, CNN(Convolutional Neural Network), Bayesian CNN, U-net 모델을 검토하며 OASIS brain MRI 데이터 세트를 활용하고 있습니다. 모델 평가의 신뢰성을 보장하기 위해 데이터 불균형 문제를 해결하였고, 민감도, 특이도 및 계산 효율성을 고려하여 각 모델의 장단점을 분석했습니다.

- **Technical Details**: 연구는 세 가지 심층 학습 모델(CNN, Bayesian CNN, U-Net)을 사용하여 MRI 스캔 결과를 분석합니다. 데이터 세트의 균형을 유지하기 위해 SMOTE-Tomek 기법을 적용하였고, 모델의 정확도, 정밀도, 재현율 및 F1 점수를 평가하였습니다. Bayesian CNN은 95% 이상의 정확도를 달성했고, Grad-CAM(Gradient-weighted Class Activation Mapping)을 통해 모델 예측에 기여한 뇌의 주요 영역을 시각화하여 해석 가능성을 높였습니다.

- **Performance Highlights**: Bayesian CNN 모델은 95% 이상의 높은 정확도를 기록하여 가장 우수한 성능을 보였습니다. CNN과 U-Net 모델도 후속 순위를 차지하며 성능을 입증했습니다. Grad-CAM을 활용하여 모델이 집중한 뇌 영역을 시각화함으로써, AI를 통한 조기 진단 가능성을 제시하였습니다. 향후 연구는 종양학자와 협력하여 뇌 MRI의 마스킹 방법을 개발하고, AD 조기 예측을 위한 핵심 변수를 식별하는 방향으로 진행될 예정입니다.



### Multi-Modal Attention Networks for Enhanced Segmentation and Depth Estimation of Subsurface Defects in Pulse Thermography (https://arxiv.org/abs/2501.09994)
Comments:
          Pulse thermography, infrared thermography, defect segmentation, multi-modal networks, attention mechanism

- **What's New**: 이 논문에서는 Pulse Thermography (PT) 검사를 위한 AI 기반의 다중 모달 주의 기반 융합 네트워크인 PT-Fusion을 소개합니다. PT-Fusion은 PCA(주성분 분석)와 TSR(열영상 신호 재구성) 모달리티를 통합하여 결함 분할과 깊이 추정을 동시에 수행할 수 있도록 설계되었습니다. 이 연구는 기존의 기술적 제약을 극복하고, 특히 결함 탐지의 성능을 향상시키는 접근 방식을 제안하여 주목받고 있습니다.

- **Technical Details**: PT-Fusion 네트워크는 결함 분할과 깊이 추정에 필요한 두 가지 주요 모듈인 Encoder Attention Fusion Gate (EAFG)와 Attention Enhanced Decoding Block (AEDB)을 사용하여 PCA와 TSR 데이터를 융합하는 방식으로 작동합니다. 이 모듈들은 서로 다른 정보인 PCA와 TSR의 의미론적 특성을 동적으로 학습하여 성능을 최적화합니다. 또한, 새로운 데이터 증강 기법이 도입되어 열영상 시퀀스에서 무작위 데이터 샘플링을 통해 PT 데이터셋의 부족 문제를 완화하고자 하였습니다.

- **Performance Highlights**: 시험 결과, PT-Fusion은 U-Net, Attention U-Net 및 3D-CNN과 같은 기존의 최신 PT 검사 모델에 비해 결함 분할 및 깊이 추정 정확도에서 10% 이상의 성능 향상을 보였습니다. 이 성과는 PT-Fusion의 새로운 접근 방식이 실제 산업 환경에서의 비파괴 검사(NDT) 성능을 극대화하는 데 기여할 수 있음을 보여줍니다.



### RichSpace: Enriching Text-to-Video Prompt Space via Text Embedding Interpolation (https://arxiv.org/abs/2501.09982)
- **What's New**: 이번 연구에서는 텍스트를 비디오로 생성하는 데 있어 중요성이 상대적으로 간과되었던 텍스트 임베딩(text embedding)의 최적화 방법을 제안합니다. 특히, 임베딩 공간에서의 보간(interpolation)을 통해 최적의 텍스트 임베딩을 선택함으로써 비디오 생성 모델의 성능을 개선할 수 있다고 주장합니다. 이러한 접근 방식은 기존의 다수의 텍스트 인코더를 사용하는 방법과 대조를 이루며, 계산 비용을 줄이는 동시에 효과적인 결과를 도출할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 수직각 발 임베딩(perpendicular foot embeddings)과 코사인 유사성(cosine similarity) 알고리즘을 활용하여 텍스트 임베딩 공간 내에서 최적의 텍스트 임베딩을 찾는 새로운 방법을 소개합니다. 이를 통해 비디오 생성 모델이 원하는 비디오를 생성할 수 있는 능력을 극대화하는 것을 목표로 하고 있습니다. 또한, CogvideoX 모델의 각 구성 요소에 대한 공식적인 정의와 문제 정의를 제공하여, 향후 연구자들이 더욱 쉽게 이 알고리즘을 적용할 수 있도록 돕고 있습니다.

- **Performance Highlights**: 실험 결과, 최적의 텍스트 임베딩을 선택하는 것이 비디오 생성 모델이 원하는 비디오를 생성하는 데 큰 영향을 미친다는 것을 확인했습니다. 본 연구의 제안에 따른 알고리즘은 비디오 품질 향상은 물론 복잡한 텍스트 프롬프트를 보다 효과적으로 처리하는 능력을 보여주었습니다. 이러한 성과는 텍스트에서 비디오로의 전환이 갖는 잠재력을 뒷받침하며, 향후 다양한 응용 프로그램에서 활용될 수 있는 기반을 제공합니다.



### Aneumo: A Large-Scale Comprehensive Synthetic Dataset of Aneurysm Hemodynamics (https://arxiv.org/abs/2501.09980)
- **What's New**: 이번 연구에서는 뇌동맥류(Intracranial Aneurysm, IA)에 대한 포괄적인 혈역학적 데이터셋을 구축했습니다. 이 데이터셋은 466개의 실제 뇌동맥류 모델과 10,000개의 합성 모델을 포함하여 뇌동맥류의 형태적 특징을 분석하는 데 유용합니다. 특히, 데이터셋에는 의학 이미지와 유사한 세분화 마스크 파일도 제공되어 유의미한 분석을 지원합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 8개의 일정한 유량(steady-state flow rates)에서 측정된 혈역학적 데이터로 구성되어 있으며, 유속(flow velocity), 압력(pressure), 벽 전단 응력(wall shear stress)과 같은 중요한 매개변수가 포함되어 있습니다. 이 데이터는 뇌동맥류의 발생 원인(pathogenesis) 및 임상 예측(clinical prediction) 연구에 있어 귀중한 자원입니다. 또한, 466개의 뇌동맥류가 없는 모델과 9,534개의 변형된 뇌동맥류 모델이 포함되어 있어 다양한 형태적 분석이 가능합니다.

- **Performance Highlights**: 이 데이터셋은 뇌동맥류의 병리적 특성과 혈역학적 메커니즘을 이해하는 데 중요한 기여를 할 것으로 기대됩니다. 이를 통해 뇌동맥류 관련 연구의 심화와 클리닉에서의 예측 가능성을 높일 수 있습니다. 연구자는 이 데이터셋을 활용하여 뇌동맥류의 보다 심층적인 연구를 진행할 수 있습니다.



### GaussianAvatar-Editor: Photorealistic Animatable Gaussian Head Avatar Editor (https://arxiv.org/abs/2501.09978)
Comments:
          Accepted to 3DV 2025. [Project Link](this https URL)

- **What's New**: GaussianAvatar-Editor는 표현, 포즈 및 시점을 완전히 제어할 수 있는 애니메이션 가능한 Gaussian 헤드 아바타의 텍스트 기반 편집을 위한 혁신적인 프레임워크입니다. 3D Gaussian 편집이 정적이었던 반면, 4D 애니메이션 가능한 아바타를 편집하는 데는 모션 가림 및 시공간 일관성과 관련된 도전 과제가 존재합니다. 이를 해결하기 위해 가시적인 Gaussian의 혼합 가중치를 강화하고 비가시적인 Gaussian의 영향을 억제하는 가중 알파 혼합 방정식(WABE)을 제안합니다.

- **Technical Details**: GaussianAvatar-Editor는 비가시적인 부분에서 발생하는 모션 가림 문제를 해결하기 위해 Gaussian alpha blending에서 새로운 활성화 함수를 적용하고, 편집 품질 향상 및 4D 일관성을 보장하기 위해 조건적 적대적 학습(conditional adversarial learning)을 편집 과정에 통합합니다. 이러한 접근 방식은 편집된 결과를 개선하고 애니메이션 전반에 걸쳐 일관성을 유지하는 데 도움을 줍니다. 우리의 방법은 포토리얼리스틱한 4D 애니메이션 편집을 수행하며, 다양한 주제에 대한 실험을 통해 그 효과iveness를 입증합니다.

- **Performance Highlights**: GaussianAvatar-Editor는 새로운 뷰, 포즈 및 표현에서 기존의 방법에 비해 일관되게 우수한 성능을 보이며, 고품질 편집과 시공간 일관성을 보장합니다. 노이즈가 많은 씬이나 복잡한 애니메이션 시나리오에서도 탁월한 결과를 제공합니다. 모든 실험 결과와 코드가 제공되는 프로젝트 링크에서 더 많은 정보를 확인할 수 있습니다.



### Discrete Prior-based Temporal-coherent Content Prediction for Blind Face Video Restoration (https://arxiv.org/abs/2501.09960)
- **What's New**: 이 논문에서는 복잡하고 알려지지 않은 열화(Degradation)를 겪은 비디오에서 고충실도(High-fidelity) 얼굴 비디오 복원(BFVR)의 새로운 접근 방식인 DP-TempCoh(Discrete Prior-based Temporal-Coherent content prediction transformer)를 제안합니다. 이 모델은 고품질 콘텐츠를 합성하기 위해 비정형(Discrete) 비주얼 프라이어를 기반으로 하는 공간-시간 인식 콘텐츠 예측 모듈과 모션 통계 조정 모듈을 포함하고 있습니다. 이러한 구성 요소들은 비디오에서의 시간적 동질성을 향상시켜 복원된 결과의 일관성을 높이는 데 기여합니다.

- **Technical Details**: DP-TempCoh 모델은 두 가지 주요 모듈로 구성되어 있습니다. 첫 번째 모듈은 비주얼 프라이어 기반의 콘텐츠 예측 모듈로, 열화된 비디오 프레임의 맥락 정보를 조건으로 고품질 콘텐츠의 특성을 생성합니다. 두 번째 모듈은 모션 프라이어에 기반한 통계 조정 모듈로, 교차 프레임의 평균 및 분산을 조정하여 예측된 콘텐츠가 실제 얼굴 비디오의 통계와 일치하도록 조정합니다.

- **Performance Highlights**: 실험을 통해 DP-TempCoh가 기존의 최첨단 방법들에 비해 우수한 성능을 보임을 확인하였습니다. 복원된 비디오에서 얼굴의 동적 일관성을 유지하면서도 탁월한 품질의 얼굴 이미지를 생성할 수 있음을 입증하였습니다. 특히, 실제 세계의 비디오 복원에서도 뛰어난 성능을 발휘하는 것으로 나타났습니다.



### Surface-SOS: Self-Supervised Object Segmentation via Neural Surface Representation (https://arxiv.org/abs/2501.09947)
Comments:
          Accepted by TIP

- **What's New**: 이 논문은 Self-supervised Object Segmentation (SOS) 알고리즘을 제안하며, 주석이 없는 객체 분할을 가능하게 합니다. 특히, 다중 카메라 입력 조건에서 각 보기(view) 간의 구조적, 질감적, 그리고 기하학적 일관성을 활용하여 세밀한 객체 분할을 달성합니다. Surface representation 기반의 Self-supervised Object Segmentation(Surface-SOS)라는 새로운 프레임워크를 통해 다중 시점 이미지를 사용한 3D 표면 표현으로 객체를 분할하는 방법을 제안합니다.

- **Technical Details**: Surface-SOS는 복잡한 장면을 위해 고품질 기하학적 표면을 모델링하는 새로운 장면 표현 체계를 설계합니다. 이 체계는 Signed Distance Function(SDF)을 사용하여 장면을 두 개의 보완적인 신경망 표현 모듈로 분해합니다. 또한, 다중 보기의 비주석 이미지에서 단일 보기 분할을 개선하기 위해 조잡한 분할 마스크를 추가 입력으로 도입합니다.

- **Performance Highlights**: 여러 표준 벤치마크(LLFF, CO3D, BlendedMVS, TUM 및 여러 실제 장면)에서 실험한 결과, Surface-SOS는 NeRF 기반의 방법들보다 항상 보다 세밀한 객체 마스크를 제공하며, 감독된 단일 보기 기준선도 크게 초월하는 성능을 보입니다. 이 연구는 대량의 주석 데이터와 강력한 제약에 대한 의존성을 줄이는 최초의 Self-supervised 접근 방식으로 평가됩니다.



### A Multi-Scale Feature Extraction and Fusion Deep Learning Method for Classification of Wheat Diseases (https://arxiv.org/abs/2501.09938)
- **What's New**: 본 연구에서는 밀(wheat) 질병의 식별 및 분류의 어려움을 다루며, 특히 밀 느슨한 깔개(millet loose smut), 잎녹병(leaf rust), 왕관 및 뿌리 썩음(crown and root rot) 질병에 중점을 둡니다. 이번 연구는 멀티 스케일 피쳐 추출(multi-scale feature extraction)과 고급 이미지 세분화(image segmentation) 기술을 결합한 혁신적인 접근 방식을 제안하여 질병 분류 정확도를 향상시킵니다.

- **Technical Details**: 제안된 방법론은 Xception, Inception V3, ResNet 50과 같은 신경망(neural network) 모델을 사용하여 2020년 대규모 밀 질병 분류 데이터셋으로 훈련됩니다. 이 과정에서 투표(voting) 및 스태킹(stacking)과 같은 여러 머신 비전 분류기(ensemble of machine vision classifiers)를 통합합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법론은 최신 방법들과 비교해 99.75%의 높은 정확도를 달성했습니다. 특히, Xception 딥러닝 앙상블 모델은 가장 높은 정확도를 보여주었습니다.



### IE-Bench: Advancing the Measurement of Text-Driven Image Editing for Human Perception Alignmen (https://arxiv.org/abs/2501.09927)
- **What's New**: 이 논문에서는 텍스트 기반 이미지 편집(Text-driven image editing)의 평가 방법을 개선하기 위해 IE-Bench라는 새로운 벤치마크 세트를 제안합니다. IE-Bench는 다양한 소스 이미지, 편집 프롬프트 및 결과를 포함하는 데이터베이스로, 이를 통해 인간의 주관적 평가를 반영한 Mean Opinion Scores (MOS)를 제공합니다. 또한, 텍스트 기반 이미지 편집의 품질을 평가하기 위해 IE-QA라는 다중 모달 소스 인식 품질 평가 방법을 소개합니다.

- **Technical Details**: IE-Bench는 이미지 편집에서 텍스트와 소스 이미지 간의 관계를 동적으로 모델링하는 다중 모달 방법을 필요로 한다는 점에 중점을 둡니다. IE-DB는 다양한 소스-프롬프트-타겟 케이스를 수집하고 MOS를 제공하며, 이는 이미지 편집을 위한 최초의 이미지 품질 평가 데이터셋으로 간주됩니다. IE-QA는 편집된 이미지의 품질, 이미지와 텍스트 간의 관계, 소스와 타겟 이미지 간의 연결성을 포함하여 여러 차원을 고려한 포괄적인 평가를 제공합니다.

- **Performance Highlights**: IE-QA는 기존의 IQA 메트릭스와 비교하여 주관적 정렬에서 현저한 이점을 보여주며, 텍스트 기반 이미지 편집 작업에서 효과적인 성능을 입증합니다. 이러한 연구 결과는 텍스트 기반 이미지 편집 평가의 새로운 지평을 열며, 관련 데이터와 코드가 공개될 예정입니다. IE-Bench의 결과는 향후 텍스트 기반 이미지 편집의 질을 향상시키는 데 기여할 것으로 기대됩니다.



### TalkingEyes: Pluralistic Speech-Driven 3D Eye Gaze Animation (https://arxiv.org/abs/2501.09921)
- **What's New**: 본 연구는 음성 기반 3D 얼굴 애니메이션에서 종종 간과되는 눈의 시선(eye gaze) 애니메이션 생성 방식을 제안합니다. 'TalkingEyes'라는 새로운 데이터 기반 방법을 통해, 음성과 조화를 이루는 다양한 3D 눈 시선 운동을 생성할 수 있도록 합니다. 이 방법은 약 14시간의 음성-메쉬 시퀀스가 포함된 오디오-시선 데이터셋을 구축하여, 고품질의 눈 또는 얼굴 움직임을 동기화할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 저품질의 인터넷 비디오에서 먼저 3D 눈 시선 맞춤과 3D 얼굴 재구성을 수행하는 'LightGazeFit'이라는 경량화된 방법을 제안합니다. 그런 다음, 시선 운동과 머리 운동을 두 개의 개별 잠재 공간(latent spaces)에서 모델링하여 음성을 통해 생성합니다. 이러한 접근 방식은 눈동자의 회전 범위가 머리보다 작다는 생리학적 지식에 기반하여 서로 다른 운동의 다양성을 고려합니다.

- **Performance Highlights**: 'TalkingEyes'는 음성 입력만으로 눈 시선 운동, 눈 깜빡임, 머리 운동 및 얼굴 운동을 복합적으로 생성할 수 있는 첫 번째 3D 애니메이션 아바타입니다. 제안된 방법의 우수성은 정량적 및 정성적 평가를 통해 입증되었으며, 다양한 자연스러운 3D 눈 시선 운동을 생동감 있게 생성할 수 있습니다. 이를 통해 또한 기존 방법들보다 불확실성을 감소시키고, 더 다채로운 움직임을 생성하는 데 중점을 두고 있습니다.



### FoundationStereo: Zero-Shot Stereo Matching (https://arxiv.org/abs/2501.09898)
- **What's New**: FoundationStereo는 1백만 개의 고해상도 합성 스테레오 이미지 페어로 구성된 대규모 훈련 데이터셋을 기반으로 하는 새로운 스테레오 깊이 추정 모델로, 기존의 도메인별 파인 튜닝 없이도 강력한 제로샷 제너럴리제이션(zero-shot generalization)을 달성합니다. 이 모델은 사진 현실감을 유지하면서 다채로운 데이터를 포함하고 있으며, 자동 자가 검증 파이프라인을 통해 모호한 샘플을 제거하여 데이터 품질과 모델의 강인성을 높였습니다.

- **Technical Details**: FoundationStereo는 사이드 튜닝 피처 백본을 사용하여 DepthAnythingV2 모델의 리치한 단안 선행 지식을 스테레오 설정에 맞게 조정합니다. 또한, Attentive Hybrid Cost Volume(AHCF) 모듈은 3D Axial-Planar Convolution(APC) 필터링과 Disparity Transformer(DT)를 포함하여 모든 차이에 대해 자가 주의를 수행하고 볼륨 피처 집합을 위한 수용 영역을 늘려 개선된 성능을 제공합니다. 이러한 혁신들을 통해 스테레오 깊이 추정에서 강력한 성능을 발휘합니다.

- **Performance Highlights**: FoundationStereo는 기존의 도메인별 파인 튜닝 모델과 비교할 때 우수한 성능을 보이며, 실제 데이터에 대한 적용 시에도 기존 방법들을 크게 능가합니다. 제로샷 일반화 능력 덕분에, 이 모델은 다양한 도메인에서의 정확하고 안정적인 성능을 유지하며, 향후 실제 애플리케이션에서 경쟁력을 갖출 수 있는 새로운 기준을 제공할 것으로 기대됩니다.



### FLORA: Formal Language Model Enables Robust Training-free Zero-shot Object Referring Analysis (https://arxiv.org/abs/2501.09887)
- **What's New**: 본 논문에서는 FLORA(Formal Language for Object Referring and Analysis)라는 훈련이 필요 없는 새로운 프레임워크를 소개합니다. FLORA는 대형 언어 모델(LLMs)의 추론 능력을 활용하여 객체 지칭 분석(Object Referring Analysis, ORA)을 지원합니다. 이 프레임워크는 비주얼 그라운딩 감지기를 통한 제로샷(zero-shot) ORA 성능을 향상시키며, 정교화된 언어 모델과 베이지안 추론 접근 방식을 결합하여 구현됩니다.

- **Technical Details**: FLORA는 LLM의 출력물을 규제하기 위한 공식 언어 모델(formal language model)을 포함하고 있으며, 구조적 설명의 세부 정보를 해석하고 추론하는 확률적 프레임워크를 제공합니다. 이를 통해 객체 설명에 대한 효과적인 해석을 달성하면서 추가적인 훈련 없이 제로샷 ORA 성능을 높입니다. 구체적으로, FLM은 복잡한 구문을 규칙에 기반한 구조로 형식화하여 객체 탐지의 가능성을 계산합니다.

- **Performance Highlights**: FLORA 프레임워크는 다양한 적재 데이터셋에서 기존 pretrained grounding detectors의 제로샷 성능을 약 45% 향상시킵니다. 여러 도전적인 데이터셋에 대한 포괄적인 평가를 통해, FLORA가 현재의 최신 제로샷 방법들보다 일관되게 우수한 성과를 보여줍니다. 이러한 결과는 FLORA가 다중 모달 비주얼 이해 및 추론에서 활용될 수 있는 새로운 기반을 제공함을 의미합니다.



### Semi-Supervised Image-Based Narrative Extraction: A Case Study with Historical Photographic Records (https://arxiv.org/abs/2501.09884)
Comments:
          This paper has been accepted for oral presentation in the findings track of the 47th European Conference on Information Retrieval (ECIR 2025). Source code and experiments are available at this https URL

- **What's New**: 이 논문은 역사적 사진 기록에서 내러티브(narrative)를 추출하기 위한 반지도 학습(semi-supervised) 접근법을 제시합니다. 기존의 비지도(text-based) 방식에 이미지를 적용하여 심층 학습(deep learning) 기술을 활용한 시각적 특성 추출 및 유사성 계산을 목표로 하고 있습니다. 특히, 1928년 볼리비아의 Sacambaya Expedition에서 캡처한 Robert Gerstmann의 사진을 담은 ROGER 데이터셋에 본 방법을 적용했습니다.

- **Technical Details**: 저자는 원래의 비지도 내러티브 맵 알고리즘을 이미지 데이터에 맞게 조정하며, 전문가 주석(annotator)이 제공한 부분 레이블(partial labels)을 통해 의미 있는 내러티브를 추출합니다. 이 과정에서는 시각적 특징(feature) 추출, 유사성 계산, 내러티브 구조 구성 등의 중요한 단계가 포함됩니다. 또한, 동적 시간 왜곡(Dynamic Time Warping, DTW) 알고리즘을 사용하여 전문가가 만든 기준과 추출된 내러티브를 비교합니다.

- **Performance Highlights**: 연구 결과는 10개 이상의 이미지를 포함한 긴 타임라인의 경우, 내러티브 맵 접근법이 무작위 샘플링(random sampling)보다 우수함을 보여줍니다. 전문가 평가를 통해 추출된 내러티브의 역사적 정확성 및 일관성이 확인되었습니다. 이러한 연구는 역사 연구자와 디지털 인문학 연구자에게 새로운 도구를 제공하여 대규모 이미지 컬렉션에서 의미 있는 내러티브를 탐색하고 이해하는 데 기여합니다.



### ASTRA: A Scene-aware TRAnsformer-based model for trajectory prediction (https://arxiv.org/abs/2501.09878)
- **What's New**: 이번 논문에서는 ASTRA(장면 인식 트랜스포머 기반 경로 예측 모델)를 제안합니다. 이 모델은 장면 맥락, 공간 역학, 사회적 상호작용 및 시간 진행을 통합하여 보행자 경로 예측의 정확성을 높이도록 설계되었습니다. 또한, 경량화된 설계와 경량화된 패러미터 구성으로 기존 모델보다 7배 적은 파라미터를 가지고 있습니다.

- **Technical Details**: ASTRA는 U-Net 기반의 피쳐 추출기를 활용하여 장면 대표성을 포착하고, 그래프 인식 트랜스포머 인코더를 통해 사회적 상호작용을 캡쳐합니다. 이 모델은 결정론적(prediction) 및 확률적(stochastic) 결과를 모두 생성할 수 있으며, 확률적 예측은 조건부 변분 오토인코더(Conditional Variational Auto-Encoder, CVAE)를 통해 생성됩니다. 또한, 손실 함수에 대한 가중 패널티를 도입하여 다양한 상태에서 우수한 예측 성능을 발휘합니다.

- **Performance Highlights**: ASTRA는 ETH-UCY 데이터셋에서 결정론적 27%, 확률적 10%의 평균적인 성능 향상을 보였으며, PIE 데이터셋에서는 26% 향상되었습니다. 이러한 성과 외에도, 이 모델은 다양한 관점의 일반화 능력을 갖추고 있어 Bird's Eye View(BEV) 및 Ego-Vehicle View(EVV)에서 모두 적용 가능합니다.



### CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation (https://arxiv.org/abs/2501.09838)
Comments:
          Accepted in the 2025 WACV workshop GeoCV

- **What's New**: 본 연구에서는 CrossModalityDiffusion라는 새로운 모듈형 프레임워크를 제안하여 다양한 감지 모달리티 간에서 이미지를 생성합니다. 이 프레임워크는 장면 기하학에 대한 사전 정보 없이도 다양한 시점에서 이미지를 합성할 수 있는 가능성을 제공합니다. 다수의 입력 이미지를 처리하여 장면 구조를 인코드하는 기하학적 인식 피처 볼륨을 생성합니다.

- **Technical Details**: CrossModalityDiffusion는 모달리티별 인코더를 사용하여 여러 이미지를 입력으로 받고, 입력 카메라 위치에 상대적인 장면 구조를 암시하는 피처 볼륨을 생성합니다. 이 피처 볼륨은 입력 모달리티를 통합하는 공통 공간에서 배치되며, 부피 렌더링 기법을 사용하여 새로운 관점에서 피처 이미지를 렌더링합니다. 이렇게 생성된 피처 이미지는 특정 모달리티를 위한 확산 모델의 조건부 입력으로 활용됩니다.

- **Performance Highlights**: ShapeNet 자동차 데이터셋에서 CrossModalityDiffusion의 성능을 검증한 결과, 다양한 이미징 모달리티 간에 정확하고 일관된 새로운 관점을 효과적으로 합성할 수 있음을 보여주었습니다. 본 연구는 다양한 이미지 감지 모달리티 간의 새로운 시점 합성 문제를 해결하는 데 중점을 두고 있으며, 이전 모델에 비해 일반화된 성능을 나타냅니다.



### EraseBench: Understanding The Ripple Effects of Concept Erasure Techniques (https://arxiv.org/abs/2501.09833)
Comments:
          11 pages main; 9 pages supplemental material

- **What's New**: 이번 연구는 개념 지우기(Concept Erasure) 기법의 새로운 평가 기준인 EraseBENCH를 소개합니다. 이 벤치마크는 전체 100개 이상의 다양한 개념과 1,000개 이상의 맞춤형 프롬프트로 구성되어 있어 개념 지우기 방식의 효과를 보다 심층적으로 평가할 수 있습니다. 연구 결과, 현재의 최첨단 기술들조차 이미지 품질을 유지하는 데 어려움을 겪고 있으며, 이는 실제 적용을 위한 신뢰성에 큰 차이가 있음을 보여줍니다.

- **Technical Details**: 이 연구는 개념의 시각적 유사성, 이항 관계, 의미론적 관계를 중점적으로 조사하며, 이러한 상호 연결된 관계가 개념 얽힘(Concept Entanglement)을 불러일으켜 이미지 품질 저하나 예기치 않은 결과를 초래한다고 주장합니다. EraseBENCH는 이러한 다양한 평가 차원을 아우르는 것으로, 실질적인 지우기 성능을 평가할 수 있도록 설계되었습니다. 이 작업은 많은 개념 지우기 기법이 품질 유지를 유지하는 데 어려움을 겪고 있음을 밝혀내어 추가 연구의 필요성을 강조합니다.

- **Performance Highlights**: EraseBENCH를 통해 테스트한 결과, 현행 개념 지우기 기법들이 품질과 정합성을 유지하지 못하고 있으며, 이는 민감한 애플리케이션에서 사용할 수 있는 수준이 아님을 시사합니다. 연구팀은 시각적으로 유사한 개념이나 이항적으로 관련된 개념을 다룰 때 현행 기법들이 특히 취약하다는 점을 강조했습니다. 이러한 결과는 개념 지우기 기법의 신뢰성과 견고함에서 상당한 격차가 있음을 보여줍니다.



### PIXELS: Progressive Image Xemplar-based Editing with Latent Surgery (https://arxiv.org/abs/2501.09826)
- **What's New**: 최근의 영상 편집을 위한 언어 안내 드리프트 모델에 대한 발전은 사용자들이 원하는 변경사항을 정확하게 표현하기 위해 번거로운 프롬프트 엔지니어링에 의해 제약을 받는 경우가 많습니다. 본 논문에서는 평범한 예시를 기반으로 한 맞춤형 편집이 가능하도록 하는 PIXELS라는 새로운 프레임워크를 도입합니다. 이 방법은 사용자가 다수의 참조 이미지를 자유롭게 활용하여 미세한 조정이 가능하도록 지원하며, 기존의 TTI 모델을 다시 훈련하거나 미세 조정할 필요 없이 이러한 조정을 수행할 수 있습니다.

- **Technical Details**: PIXELS는 사용자가 각 픽셀의 '편집 가능성'(editability)을 효율적이고 동시에 정의할 수 있도록 해 주며, 여러 개의 이미지 참조를 사용하여 매끄러운 편집을 성취합니다. 이 방법은 기존의 이미지 생성 모델의 추론 과정에서 동작하여, 이미지의 특정 영역을 지역적으로 조정할 수 있도록 해줍니다. 이로 인해 다양한 참조 이미지를 통해 세밀한 수정 및 선택적 변경이 가능해지며, 사용자가 원하는 편집을 보다 효과적으로 제어할 수 있는 여지를 제공합니다.

- **Performance Highlights**: PIXELS는 다른 기존 방법들과 비교했을 때 높은 품질의 편집을 효율적으로 수행할 수 있음을 입증하였습니다. 정량적 지표와 사용자 평가 모두에서 매우 유의미한 개선을 나타내었습니다. 이러한 접근법은 이미지 생성 모델을 사용하는 다양한 사용자들에게 전문적인 편집 기능을 더 쉽게 사용할 수 있도록 변모시킬 잠재력이 있습니다.



### Generalized Single-Image-Based Morphing Attack Detection Using Deep Representations from Vision Transformer (https://arxiv.org/abs/2501.09817)
- **What's New**: 이번 논문에서는 얼굴 인식 시스템(Face Recognition Systems, FRS)에 대한 심각한 위협을 방지하기 위해, 모핑 공격 탐지를 위한 새로운 알고리즘인 S-MAD(Single-image-based Morphing Attack Detection)를 제안합니다. 특히, Vision Transformer (ViT) 구조를 활용하여 이미지의 로컬(local) 및 글로벌(global) 정보를 효과적으로 통합하는 방법을 탐구합니다. 이는 다양한 유형의 감지 알고리즘(MAD) 중에서도 특히 단일 이미지에 기반한 공격 탐지 기술을 발전시키는 데 중요한 기여를 합니다.

- **Technical Details**: 제안된 S-MAD 알고리즘은 MTCNN을 사용하여 얼굴 영역을 식별하고, 384x384 픽셀로 리사이즈한 후 입력 이미지를 32x32 픽셀 크기의 패치(patch)로 분할합니다. 각 패치는 선형 프로젝션을 통해 임베딩(embedding)으로 변환되며, 자기 주의 메커니즘(self-attention mechanism)을 포함하는 24층의 트랜스포머 인코더(transformer encoder)를 통해 처리됩니다. 이 과정에서 여러 개의 관점에서 특징을 추출할 수 있어, 다양한 모핑 공격에 대한 탐지 능력이 향상됩니다.

- **Performance Highlights**: 논문에서 제안한 S-MAD 알고리즘은 여러 공개 데이터셋을 대상으로 한 실험을 통해 성능 평가를 진행하였습니다. 교차 데이터셋 테스트에서의 성능 개선이 입증되었으며, 인트라 데이터셋 테스트(intra-dataset testing)에서는 유사한 성능을 보였습니다. 기존의 딥러닝 구조로 개발된 다른 최신 알고리즘들과 비교하여, 제안한 알고리즘이 디지털 입력에서 일반화된 성능을 나타내는 것으로 분석되었습니다.



### Lossy Compression with Pretrained Diffusion Models (https://arxiv.org/abs/2501.09815)
- **What's New**: 이번 연구는 DiffC 알고리즘을 Stable Diffusion 모델에 적용하여, pretrained 모델들이 탁월한 lossy 이미지 압축기로서의 능력을 갖고 있음을 입증합니다. 우리는 간단한 우회 방법을 도입하여 DiffC의 첫 번째 완전 구현을 수행하였고, 이는 10초 이내에 이미지를 압축하고 복원할 수 있습니다. 우리의 방법은 추가적인 훈련 없이도 경쟁력 있는 성능을 보여주며, 초저비트 전송에서의 state-of-the-art generative 압축 방법들과 비교할 만한 성능을 제공합니다.

- **Technical Details**: DiffC 알고리즘은 pretrained diffusion 모델을 사용하여 손실 압축을 수행하는 원리적 알고리즘으로, 이전 연구에서 제안되었습니다. 본 연구에서는 DiffC를 Stable Diffusion 1.5, 2, XL, Flux-dev와 같은 플래그십 오픈 소스 모델에 적용하는 최초의 사례가 됩니다. 우리는 denoising 일정 최적화를 통한 알고리즘 개선 방안을 제안하며, 이는 압축률-왜곡 곡선을 향상시키고 인코딩 시간을 10초 이내로 줄입니다.

- **Performance Highlights**: DiffC는 추가 훈련 없이 제로샷(zero-shot)으로 작동하며, 원하는 비트 전송률로 이미지를 쉽게 인코딩할 수 있는 특성을 가지고 있습니다. 비록 Diffusion 모델들이 원래 이미지 압축을 위한 것이 아닐지라도, 우리는 HiFiC, MS-ILLM, DiffEIC, PerCo와 같은 목적 특화 최신 압축 방법들과 비교하여 경쟁력 있는 압축 성능을 달성했습니다.



### SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation (https://arxiv.org/abs/2501.09782)
Comments:
          An extension of SMPLer-X [arXiv:2309.17448]. Homepage: this https URL

- **What's New**: 이 연구는 Expressive Human Pose and Shape Estimation (EHPS) 작업을 위한 데이터 및 모델 스케일링의 중요성을 조사합니다. 40개의 다양한 EHPS 데이터셋을 체계적으로 분석하고 이를 통해 수집한 인사이트를 바탕으로 모델의 훈련 체계를 최적화합니다. 특히, 대량의 데이터를 이용해 SMPLer-X와 SMPLest-X라는 새로운 파라미터 모델을 개발하여 다양한 테스트 벤치마크에서 우수한 성능을 보여줍니다.

- **Technical Details**: 연구는 파라미터 모델의 설정에 있어 SMPL-X와 같은 기존 모델들을 활용합니다. 모델 스케일링을 위해 vision transformers를 사용하고, SMPLer-X와 SMPLest-X를 통해 간소화된 아키텍처를 채택하여 강력한 전반적인 성능을 발휘합니다. 이 연구에서는 Mean Primary Error (MPE)라는 새로운 지표를 제안하여 모델 성능을 평가하며, 이를 통해 다채로운 데이터의 활용을 강조합니다.

- **Performance Highlights**: SMPLer-X와 SMPLest-X는 여러 대회에서 이전의 모든 기록을 뒤엎고, AGORA 데이터셋에서 NMVE가 96.2 mm, 손 MVE는 31.1 mm에 도달하여 SOTA 성능을 달성하였습니다. 이 연구를 통해 데이터 확장 및 모델 크기 확장이 EHPS 성능을 유의미하게 향상시키며, 이는 다른 미지의 시나리오로의 효과적인 전이 가능성을 보여줍니다. 종합적으로 매개 모델들은 여러 테스트 벤치마크에서 뛰어난 성능을 유지하고 있습니다.



### VideoWorld: Exploring Knowledge Learning from Unlabeled Videos (https://arxiv.org/abs/2501.09781)
Comments:
          Code and models are released at: this https URL

- **What's New**: 이번 연구는 비언어적 입력만으로 복잡한 지식을 학습할 수 있는 딥 생성 모델의 가능성을 탐구합니다. 기존의 언어 기반 모델에 대한 초점에서 벗어나, 비디오 데이터로만 학습하는 비디오 생성 모델인 VideoWorld를 개발하였습니다. 이 모델은 Go 게임과 로봇 제어 작업에서 지식 습득 능력을 평가하여 새로운 지식을 학습하는 방법을 제시합니다.

- **Technical Details**: VideoWorld는 라벨이 없는 비디오 데이터를 기반으로 하는 자동 회귀 비디오 생성 모델입니다. 여기에는 VQ-VAE 및 자동 회귀 트랜스포머 기술이 사용되며, 비디오 프레임을 이산적으로 변환하여 다음 token 예측 패러다임을 통해 학습합니다. 또한 Latent Dynamics Model (LDM)를 도입하여 비디오 학습의 효율성과 효과성을 동시에 향상시키는 메커니즘을 제공합니다.

- **Performance Highlights**: VideoWorld는 3억 개의 파라미터로 구성된 모델로, RL 에이전트인 KataGO에 대해 5단 전문 레벨에 도달했습니다. 또한 CALVIN 및 RLBench 로봇 시나리오에서 다양한 제어 작업을 수행하며 다수의 환경에서 일반화 능력을 보여줍니다. 이러한 결과는 비디오 데이터에서 효과적으로 지식을 습득하고 복잡한 작업을 마스터하는 모델의 능력을 강조합니다.



### New Fashion Products Performance Forecasting: A Survey on Evolutions, Models and Emerging Trends (https://arxiv.org/abs/2501.10324)
Comments:
          Accepted at the Springer Nature Computer Science journal

- **What's New**: 이번 논문은 빠르게 변화하는 패션 산업의 지속 가능성을 향상시키기 위해 머신러닝과 딥러닝을 활용한 새로운 패션 제품 성능 예측(New Fashion Products Performance Forecasting, NFPPF)에 대한 체계적인 검토를 제공합니다. NFPPF 문제는 역사가 거의 없는 신제품의 미래 판매를 예측하는 도전 과제를 다루고 있으며, 과거 데이터에 크게 의존하지 않는 모델의 필요성을 강조합니다. 이 연구에서는 NFPPF 전략을 광범위하게 분석하고, 최신 방법론과 데이터를 제시하여 기존 문헌의 공백을 메우려는 기초가 됩니다.

- **Technical Details**: NFPPF 문제는 소비자 경향과 패션 트렌드의 변동성으로 인해 복잡합니다. 전통적인 시계열 예측 모델은 주로 과거 판매 데이터를 사용하므로 신제품 예측에는 한계가 있습니다. 새로운 지도학습 모델들은 제품의 색상, 스타일 및 마케팅 정보를 적극 활용하여 소비자의 관심 변화를 반영하는 보다 정교한 예측을 가능하게 합니다. 이를 통해 신제품 예측을 위한 혁신적인 접근법이 필요하며, 특히 비주얼 데이터의 통합은 정확도를 크게 향상시킵니다.

- **Performance Highlights**: 딥러닝 모델은 이미지 데이터를 포함하여 다양한 원천의 정보를 반영하여 높은 예측 정확도를 달성합니다. 예를 들어, VISUELLE 데이터셋은 5,577개의 제품에 대한 판매 데이터를 제공하며, 이는 NFPPF 연구의 표준이 되고 있습니다. 이러한 모델의 적용은 재고 관리, 마케팅 전략 조정 등 여러 결정 과정에서 효율성을 높이고, 결과적으로 패션 산업의 지속 가능성을 증대시키는 데 기여합니다.



### Robust Egoistic Rigid Body Localization (https://arxiv.org/abs/2501.10219)
- **What's New**: 이 논문은 외부 인프라 없이 대상의 형상을 사전 지식 없이 추정하는 로버스트하고 자립적인 경직체 위치 추정(Rigid Body Localization: RBL) 문제를 다룹니다. 제안된 방법은 대상이 동일한 형상이나 같은 수의 랜드마크 포인트를 가질 필요가 없음을 강조하며, 실제 무선 통신 신호를 통해 위치 정보를 효과적으로 확보할 수 있는 가능성을 보여줍니다. 또한, 논문에서는 데이터의 불완전성을 고려하여 다양한 조건 하에서의 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 이 연구에서 다루는 시스템 모델은 N개의 랜드마크 포인트로 구성된 경직체의 위치를 3D 공간에서 정의합니다. 이 경 rigid body의 형상을 나타내는 형상 매트릭스는 랜드마크 포인트들을 열 형태로 콜렉션하여 구성됩니다. 또한, 두 가지 방식으로 해당 경 rigid body 간의 변환 벡터를 추정하는 방법이 소개되며, 이들은 불완전한 관측 조건에서도 견고성을 유지합니다.

- **Performance Highlights**: 제안된 기법들은 완전한 정보 조건 및 불완전한 조건 하에서 RMSE(root mean square error) 성능 면에서 기존의 최신 기법들(State-of-the-Art: SotA)에 비해 유리함을 입증했습니다. 특히, 첫 번째 변환 벡터 추정 방법이 존재하는 기술들과 비교하여 상당히 우수한 성능을 보이며, 데이터 손실의 경우에도 적절한 대안을 제공합니다. 경 rigid body Orientation을 추정하는 세 번째 방법도 두 경 rigid body 간의 회전 매트릭스를 효과적으로 추정하는 능력을 강조합니다.



### FECT: Classification of Breast Cancer Pathological Images Based on Fusion Features (https://arxiv.org/abs/2501.10128)
- **What's New**: 이번 연구는 유방암 조직 분류를 위한 새로운 모델인 FECT(Edges, Cells, and Tissues의 융합 특징)를 제안합니다. 기존 방법들이 단일 세포 또는 조직 특징에 의존하는 경향이 있는 반면, 이 모델은 미세한 형태학적 특징을 고려하여 설계되었습니다. 결과적으로, 이 연구는 유방암 진단을 위해 보다 정확한 기계 학습 접근법을 제공합니다.

- **Technical Details**: FECT 모델은 ResMTUNet 구조와 주의(attention) 기반의 집계기(aggregator)를 사용하여, 다양한 특징들을 추출하고 통합합니다. 이 접근법은 각 세포 및 조직의 경계(edge), 세포(cell), 조직(tissue) 정보를 함께 고려하는 점에서 독창적입니다. 모델의 설계는 병리학자(pathologist)의 진단 접근 방식과 일치하여 실용성을 높입니다.

- **Performance Highlights**: BRACS 데이터셋에서 광범위한 테스트를 실시한 결과, 이 모델은 기존의 최신 방법들보다 높은 분류 정확도(classification accuracy)와 F1 점수를 기록했습니다. 또한, 이 모델의 특징 융합(fusion) 과정은 해석 가능성을 높이며 향후 임상 응용 분야에서도 중요한 역할을 할 가능성이 있습니다.



### Universal Actions for Enhanced Embodied Foundation Models (https://arxiv.org/abs/2501.10105)
Comments:
          Preprint

- **What's New**: 이 논문에서는 다양한 로봇 플랫폼에서 일반화 가능한 행동을 포착하는 새로운 프레임워크인 UniAct를 소개합니다. UniAct는 서로 다른 행동 공간의 이질성을 해소하고, 다양한 환경과 임무에 적응할 수 있도록 설계된 보편적인 작업 공간인 Universal Action Space에서 작동합니다. 이번 Framework는 전통적인 이질적 행동 공간 문제를 해결하는 데 중요한 발전을 가져올 것으로 기대됩니다.

- **Technical Details**: UniAct는 Vision Language Model (VLM)을 사용하여 벡터 양자화된 코드북 형태로 보편적인 행동 공간을 구성합니다. 이 설정은 행동의 원시적인 패턴을 추출하여 다양한 로봇에서 구현 가능한 공통적인 행동을 인식하는 데 도움을 줍니다. 더불어, 각 로봇 플랫폼에 특화된 행동 명령으로 변환할 수 있는 효율적인 이질적 디코더를 통해 빠른 적응이 가능해집니다.

- **Performance Highlights**: UniAct-0.5B 모델은 7B 파라미터를 가진 기존의 최첨단 모델보다 14배 더 작은 규모임에도 불구하고 다양한 과제에서 월등한 성능을 보여주었습니다. 특히, 새로운 로봇 플랫폼에 빠르게 적응할 수 있는 능력을 강조하며, 보편적인 행동을 채택한 것이 얼마나 큰 이점을 가져오는지 입증하고 있습니다. 이 연구는 Universal Action Space 내에서의 효과적인 행동 모델 개발이 전통적인 이질적 공간에서 개발하는 것보다 훨씬 더 유리하다는 것을 보여줍니다.



### SpatialCoT: Advancing Spatial Reasoning through Coordinate Alignment and Chain-of-Thought for Embodied Task Planning (https://arxiv.org/abs/2501.10074)
Comments:
          13 pages, 6 figures

- **What's New**: 이 연구는 embodied AI(임바디드 AI)에서의 spatial reasoning(공간 추론) 문제를 해결하기 위한 새로운 접근법인 SpatialCoT를 제안합니다. 기존의 방법들이 복잡한 임바디드 작업에서는 한계가 있었던 반면, SpatialCoT는 Vision-Language Models (VLMs)의 본질적인 사고 및 추론 능력을 활용하여 이러한 문제를 해결하고자 합니다. 본 연구는 두 가지 단계로 구성되어 있으며, vision-language 입력을 공간 좌표와 정렬하는 bi-directional alignment(양방향 정렬)와 언어 모델의 사고 능력을 활용하는 chain-of-thought spatial grounding(사고의 연쇄 공간 기초)를 포함합니다.

- **Technical Details**: SpatialCoT 방법론은 특히 복잡한 환경에서의 세밀한 작업을 처리하기 위해 설계되었습니다. 이 방법은 spatial coordinate bi-directional alignment를 통해 vision-language 입력을 공간 좌표와 정렬하며, chain-of-thought 접근 방식을 통해 고급 공간 추론을 지원합니다. 이 두 가지 단계는 서로 보완적으로 작용하여 VLMs의 공간 추론 능력을 극대화합니다.

- **Performance Highlights**: SpatialCoT는 시뮬레이션과 실제 환경 모두에서 어려운 네비게이션과 조작 작업에 대해 평가되었습니다. 실험 결과, SpatialCoT는 이전의 최첨단 방법들보다 두 작업 모두에서 상당한 성과 향상을 보여주었습니다. 이러한 성과는 VLMs의 사고 및 추론 능력을 효과적으로 활용함으로써 달성되었습니다.



### Explainable artificial intelligence (XAI): from inherent explainability to large language models (https://arxiv.org/abs/2501.09967)
- **What's New**: 이 논문은 기계 학습 모델의 결정 과정을 설명할 수 있는 방법인 설명 가능한 인공지능(Explainable AI, XAI) 기술의 발전을 포괄적으로 다룹니다. 특히, 대규모 언어 모델(LLM)과 비전-언어 모델(VLM)을 활용하여 다른 기계 학습 모델의 설명 가능성을 자동화하거나 개선하는 방법에 대해서도 논의합니다. 이러한 접근은 모델의 결정과 행동에 대한 의미 있는 고수준 설명을 제공하는 데 도움을 줍니다.

- **Technical Details**: 기계 학습 모델의 해석 가능성은 중요하며, 이는 인풋과 아웃풋 간의 관계를 강조하거나 내부 아키텍처를 통해 모델의 예측 과정을 이해할 수 있는 것이 필요합니다. 본 논문에서는 설명 가능성의 두 가지 주요 유형인 post-hoc(모델 훈련 후 적용) 해석 가능성과 ante-hoc(훈련 또는 설계 시 해석 가능성 부여) 해석 가능성을 설명합니다. 각 방식은 장단점이 있으며, 특히 post-hoc 방식은 종종 모델의 진정한 예측 논리를 따르지 않을 수 있어 주의가 필요합니다.

- **Performance Highlights**: 설명 가능한 인공지능(XAI) 기술을 통해 기계 학습 시스템의 투명성을 높이고, 의사 결정자의 신뢰도를 증가시켜 중요한 애플리케이션에서의 채택을 촉진합니다. 또한, XAI 방법은 모델의 편향을 발견하고 모든 이해 관계자에게 공정한 결정을 보장하는 데 기여합니다. 최종적으로, XAI는 기계 학습 모델의 안전성을 개선하고, 부적절한 예측 결과를 방지하는 데 필수적인 역할을 합니다.



### Physics-informed DeepCT: Sinogram Wavelet Decomposition Meets Masked Diffusion (https://arxiv.org/abs/2501.09935)
- **What's New**: 이 논문에서는 Sparse-view X-ray computed tomography (SVCT) 복원을 위한 새로운 접근 방식인 SWARM(Sinogram-based Wavelet random decomposition And Random mask diffusion Model)을 제안합니다. SWARM은 랜덤 마스크 전략을 도입하여 제한된 훈련 샘플 공간을 효과적으로 확장합니다. 이를 통해 모델은 데이터의 다양성을 학습하고 불확실성에 대한 이해를 높일 수 있습니다.

- **Technical Details**: SWARM은 sinogram(시노그램) 및 wavelet(웨이블릿) 고주파의 두 가지 차별화된 확산 모델을 사용하여, 이미지 구조의 글로벌 정보와 세부 특성을 동시에 포착합니다. 랜덤 마스크 삽입 기법을 통해 샘플의 다양성을 높이고 모델의 일반화 능력을 향상시키며, 웨이블릿 변환을 통해 신호의 고주파 성분에 랜덤하게 접근합니다.

- **Performance Highlights**: 실험 결과, SWARM은 다양한 데이터셋에서 정량적 및 정성적 성능 모두에서 경쟁 방법들보다 우수한 결과를 보여줍니다. 이 방식은 특히 고주파 대역의 세부 정보 캡처와 같은 응용 분야에서도 효과적이며, 전체 이미지의 일관성을 확보하면서 세밀한 디테일을 개선합니다.



### ForestProtector: An IoT Architecture Integrating Machine Vision and Deep Reinforcement Learning for Efficient Wildfire Monitoring (https://arxiv.org/abs/2501.09926)
Comments:
          Accepted for publication in the proceedings of the 11th International Conference on Automation, Robotics, and Applications (ICARA 2025)

- **What's New**: 이번 연구는 저비용의 숲 화재 탐지 시스템을 제안합니다. 이 시스템은 360° 시야에서 연기를 원거리에서 모니터링할 수 있는 컴퓨터 비전 기능을 갖춘 중앙 게이트웨이 장치를 중심으로 구성됩니다. 또 다른 특징은 Deep Reinforcement Learning (DRL) 에이전트가 카메라의 방향을 동적으로 조절하여 실시간 센서 데이터를 활용한다는 점입니다.

- **Technical Details**: 제안된 시스템은 IoT 센서 노드와 중앙 게이트웨이로 이루어져 있습니다. 각 센서 노드는 온도, 습도, 기압, 연기 및 수치를 감지하는 센서를 사용하여 환경 조건을 모니터링합니다. 중앙 게이트웨이는 NVIDIA Jetson Nano 카드에서 LoRa 프로토콜을 통해 데이터를 수집하고, DRL 에이전트를 사용하여 카메라 시점을 제어합니다. 이 시스템은 MongoDB 데이터베이스에 저장된 데이터를 기반으로 실시간 시각화를 제공합니다.

- **Performance Highlights**: 저비용의 IoT 솔루션을 통해 대규모 지역에서 화재 탐지의 정확도를 향상시킬 수 있습니다. 제안된 시스템은 false positive를 줄이는 데 초점을 맞추고 있으며, 실시간으로 데이터를 전송하여 신속한 반응이 가능합니다. 또한 이 시스템은 AWS EC2 인스턴스에서 운영되며, WhatsApp을 통해 긴급 알림을 받을 수 있는 기능도 제공합니다.



### SLIM: Sim-to-Real Legged Instructive Manipulation via Long-Horizon Visuomotor Learning (https://arxiv.org/abs/2501.09905)
- **What's New**: 이번 연구에서는 강화 학습(Reinforcement Learning, RL)으로 훈련된 저비용 쿼드펙드 조작 시스템을 소개합니다. 이 시스템은 고급 정책을 위한 계층적 디자인과 함께 다양한 과업을 해결하기 위한 점진적 정책 확장 접근 방식을 포함하고 있습니다. 실제 환경에서 길고 복잡한 조작 작업을 효과적으로 수행하며, 단 하나의 RGB 카메라로도 높은 성공률을 달성합니다.

- **Technical Details**: SLIM(Sim-to-Real Legged Instructive Manipulation)은 19 자유도(Degree of Freedom)로 구성된 쿼드펙드 로봇으로, 고급 비주얼-모터 조작 정책과 저급 쿼드펙드 제어기를 결합한 계층적 설계 구조를 사용합니다. 이 시스템은 개인화된 정보(privileged information) 접근을 통해 학생 정책이 학생 정책을 증류(distill)하도록 훈련하여 다양한 작업을 수행합니다. 또한, SLIM은 언어 지침에 따라 로봇에게 다양한 작업을 지시할 수 있도록 설계되었습니다.

- **Performance Highlights**: SLIM은 시뮬레이션과 실제 환경 모두에서 높은 성공률과 빠른 작업 완료 시간을 달성했으며, 실내 외부 환경에 걸쳐 다양한 작업을 수행할 수 있는 강력한 성과를 보입니다. 특히, 비싸고 고성능 하드웨어가 아니라도 잘 작동하며, 시뮬레이션에서 연습한 내용을 실제로 원활하게 전달할 수 있는 능력을 보여줍니다. 본 연구 결과는 저비용 하드웨어를 사용하여도 다양한 환경에서 높은 성능을 입증했습니다.



### Detection of Vascular Leukoencephalopathy in CT Images (https://arxiv.org/abs/2501.09863)
- **What's New**: 이번 연구는 인공지능(AI)이 뇌의 작은 혈관 질환인 백질병(leukoencephalopathy)의 진단에서 중요한 역할을 할 수 있음을 제시합니다. 특히, 혈관성 치매 및 뇌출혈의 주요 원인으로 알려진 이 질환의 진단에 AI를 적용한 첫 번째 사례로 주목받고 있습니다.

- **Technical Details**: 연구진은 약 1200명의 환자의 축방향 뇌 CT 스캔 데이터를 사용하여, 이진 질병 분류를 위한 convolutional neural networks (CNNs)를 훈련했습니다. 다양한 환자의 생리학적 특성으로 인한 스캔 크기의 불일치를 해결하기 위해 데이터를 통일된 크기로 처리하고, 모델의 정확도를 높이기 위해 세 가지 전처리 방법을 적용했습니다.

- **Performance Highlights**: 비교된 네 가지 신경망 구조 중 ConvNext 모델이 전처리 없이 98.5%의 최고 정확도를 기록하며, 3D convolution을 적용한 모델들을 능가했습니다. Grad-CAM heatmap을 통해 모델의 결정 과정을 시각화함으로써, 스캔에서 집중한 영역을 강조하여 AI의 진단 정확성 향상을 입증했습니다.



New uploads on arXiv(cs.AI)

### Large language models for automated scholarly paper review: A survey (https://arxiv.org/abs/2501.10326)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 시대에 자동 학술 논문 리뷰(ASPR)의 전체적인 관점을 제공하고자 하였습니다. 특히 ASPR의 도입과 동시 존재 단계에서 LLMs의 발전과 적용을 고찰합니다. LLMs가 ASPR의 기술적 병목 문제를 해결하는 방법과 새로운 기술과 데이터 세트의 응용을 리뷰합니다.

- **Technical Details**: LLMs는 수억 또는 수천억 개의 매개변수를 갖고 있어 자연어의 복잡성과 다양성을 모델링하는 데 적합합니다. Closed-source LLM과 open-source LLM 두 가지 개발 경향이 있으며, 각각 중요한 장점과 단점을 가지고 있습니다. Closed-source 모델은 안정성과 성능이 뛰어나지만 투명성이 부족한 반면, open-source 모델은 커스터마이징이 용이하고 커뮤니티 참여도가 높지만 특정 도메인 지식 이해가 부족합니다.

- **Performance Highlights**: 연구 결과에 따르면 현재 ASPR에서 가장 많이 사용되는 LLM은 OpenAI의 GPT-4로, 문헌에서 가장 높은 빈도로 언급되었습니다. GPT 시리즈는 특히 사용 용이성과 성능 덕분에 널리 사용되며, Llama 시리즈는 상대적으로 적은 비율을 차지하고 있습니다. 이러한 결과는 ASPR 구현에 있어 LLM의 성능과 커뮤니티의 지지가 중요한 요소임을 시사합니다.



### An Ontology for Social Determinants of Education (SDoEd) based on Human-AI Collaborative Approach (https://arxiv.org/abs/2501.10300)
Comments:
          Accepted in CONSORTIUM FOR COMPUTING SCIENCES IN COLLEGES

- **What's New**: 이 논문은 교육의 사회적 결정 요인(Social Determinants of Education, SDoEd)에 대한 표준화된 프레임워크의 부재를 해결하기 위해 새로운 SDoEd 온톨로지를 소개합니다. 연구자들은학습자들의 삶의 상황과 교육 성과 간의 상호작용을 정밀하게 개념화하기 위한 연구를 진행했습니다. 이 온톨로지는 ChatGPT-3.5의 제안을 활용하여 개발되었으며, 동료 평가를 거친 연구 논문을 통해 검증되었습니다.

- **Technical Details**: SDoEd 온톨로지는 총 231개의 도메인 개념, 10개의 객체 속성(object properties), 24개의 데이터 속성(data properties)으로 구성되어 있습니다. 이온톨로지는 교육 분야의 전문가들에 의해 평가되었고, 표준 온톨로지 평가 소프트웨어를 통해 검증되었습니다. 이는 학생들의 삶의 상황과 교육적 성취 간의 관계를 체계적으로 이해하기 위한 기초 자료로 기능합니다.

- **Performance Highlights**: 이 연구는 SDoEd에 대해 기존의 연구와 구별될 수 있는 체계적이고 정량적인 접근 방식을 제시합니다. 이를 통해 교육 정책 개발 및 연구자들이 학생들의 사회적 결정 요인을 이해하고 분석하는 데 기여할 것으로 기대됩니다. 논문에서 제시된 첫 번째 버전의 온톨로지는 교육 분야의 실천 및 연구의 기초로 활용될 수 있습니다.



### Temporal Causal Reasoning with (Non-Recursive) Structural Equation Models (https://arxiv.org/abs/2501.10190)
- **What's New**: 최근 인공지능 연구자들과 철학자들 사이에서 인과 추론(causal reasoning)에 대한 관심이 높아지고 있습니다. 본 논문에서는 Structural Equation Models (SEM)을 사용하여 실제 인과관계를 설명하는 새로운 해석을 제안합니다. SEM은 외부 변수의 동적 변화를 내부 변수의 동적 변화로 변환하는 메커니즘으로 간주되며, 이를 통해 반사적 인과 추론을 기존의 시간 논리(formal logic) 체계와 결합할 수 있습니다.

- **Technical Details**: 논문에서는 SEM을 기반으로 한 실제 인과 추론(actual causality)과 시간적 추론(temporal reasoning) 결합을 위한 새로운 방식을 소개합니다. 이 모델에서는 외부 변수와 내부 변수 간의 의존관계를 정적(static)으로 이해하는 대신, 외부 변화의 동적 변화가 내부 변화로 어떻게 전이되는지를 시간적 맥락에서 처리합니다. 또한, 새로운 시간적 논리 구조인 CPLTL을 도입함으로써 시스템의 과거와 미래에 대한 진술을 표현할 수 있습니다.

- **Performance Highlights**: CBTLT 프레임워크는 특히 상호 의존적인 과정(mutually dependent processes)와 피드백 루프(feedback loops)에 대한 인과 분석을 수행할 수 있는 유용한 도구로 평가됩니다. 기존의 연구들이 일반적으로 재귀적(recursive) 인과 모델만을 다루었던 반면, 본 연구에서는 순환(cycle) 의존 그래프를 허용하여 기술적인 난제를 피할 수 있는 가능성을 보여줍니다. 마지막으로, 제안된 모델에 대한 효율적인 검증(model-checking) 절차도 제시됩니다.



### Generative Artificial Intelligence: Implications for Biomedical and Health Professions Education (https://arxiv.org/abs/2501.10186)
- **What's New**: 이 논문에서는 Generative AI(생성적 인공지능)가 생물의학과 건강 분야에서 교육 및 전문 업무에 미친 영향을 다루고 있습니다. 특히, 대형 언어 모델(LLMs)이 의학 시험, 임상 질문 답변, 임상 사례 해결 등에서 인간과 유사한 성능을 보여주었다는 점이 강조됩니다. 또한, Generative AI는 학술 과정 및 평가에서도 우수한 성과를 보이고 있습니다.

- **Technical Details**: 이 리뷰는 LLM의 성공 사례를 정리하고, 교육 맥락에서의 도전 과제를 소개합니다. 그 중에서도 전문 업무에서의 지식과 기술 습득을 저해할 수 있는 요소들이 논의됩니다. 또한, 교육에서 LLM 사용 시 단점을 극복하기 위한 최고의 실천 사례에 대한 추천도 포함되어 있습니다.

- **Performance Highlights**: 비록 교육에서 Generative AI 사용에 도전 과제가 존재하지만, 생물의학과 건강 분야의 모든 학생 및 교수는 이러한 기술의 효과적인 사용을 이해하고 능숙하게 다룰 수 있어야 한다고 강조합니다. 이 연구는 Generative AI의 교육적 활용을 통해 전문 인력 양성에 기여할 수 있는 가능성을 제시하고 있습니다.



### CSSDM Ontology to Enable Continuity of Care Data Interoperability (https://arxiv.org/abs/2501.10160)
Comments:
          6 pages, 5 figures, Published in: 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)

- **What's New**: 이 논문은 디지털 기술의 발전과 최근의 글로벌 팬데믹 상황을 바탕으로 헬스케어 서비스의 효율성을 높이는 방안에 대해 집중하고 있습니다. 특히, 기존의 디지털 전환 프로그램을 통합하는 액션 플랜을 검토하여 지속 가능한 헬스케어 솔루션의 기초를 마련하고 있습니다. 개인화된 홈 케어를 통해 과밀한 병원 환경에서 치료를 피할 수 있는 가능성도 제시합니다.

- **Technical Details**: 우리는 데이터 추출, 변환 및 적재(ETL)를 위한 반자동(process) 프로세스를 구현하는 방법론을 소개합니다. 이 과정에서 사용되는 공통 의미 표준 데이터 모델(Common Semantic Standardized Data Model, CSSDM)은 ISO 13940 ContSys의 공식 온톨로지를 기반으로 하며 FHIR 기반의 사양을 통합하여 KGs(지식 그래프)를 생성하는 데 필요한 구조적 속성을 지원합니다. CSSDM은 데이터 조화(harmonization)와 연결(linking)을 촉진하여 상호 운용성(interoperability)에 대한 대안을 제공합니다.

- **Performance Highlights**: 이 방법론은 헬스케어 정보 시스템을 개발하는 기업과 클라우드 기반 헬스 서비스 간의 새로운 협력 양식을 촉진합니다. 따라서 여러 이해관계자들에게 고품질 데이터와 정보를 공유할 수 있는 접근을 제공합니다. 이러한 데이터 흐름은 의료 서비스 제공자 간의 안전하고 신뢰할 수 있는 연결을 가능하게 하여 궁극적으로 헬스케어 환경의 개선에 기여할 것입니다.



### Topology-Driven Attribute Recovery for Attribute Missing Graph Learning in Social Internet of Things (https://arxiv.org/abs/2501.10151)
Comments:
          Accepted by IEEE Internet of Things Journal

- **What's New**: 이번 논문에서는 Topology-Driven Attribute Recovery (TDAR) 프레임워크를 제안하여 Attribute Missing Graphs (AMGs)에서의 속성 복구 문제를 해결합니다. TDAR는 그래프의 topological data를 활용하여 속성을 회복하는 방식으로, 기존의 방법과 비교해 노이즈를 줄이고 정보 전파를 효율적으로 개선합니다. 본 연구는 공공 데이터셋에 대한 광범위한 실험을 통해 TDAR가 기존 최첨단 방법보다 현저히 우수한 성능을 보여줌을 입증하였습니다.

- **Technical Details**: TDAR는 크게 세 가지 핵심 솔루션을 통해 AMGs의 복잡한 관계를 처리합니다: Topology-Aware Attribute Propagation (TAAP), Embedding Space Propagation Confidence (ESPC), Node Homogeneity Score (NHS) 및 Non-Linkage Similarity Calibration (NLSC)입니다. TAAP는 노드의 속성을 Dirichlet 에너지 최소화 과정으로 보고, 글로벌 전파와 초기화 기법을 통해 결측 속성을 예측합니다. ESPC는 노드의 topological 위치에 따라 동적인 주의 가중치를 부여하여 정보 전파의 오류를 보완하며, NHS와 NLSC는 이웃 노드 간의 동질성을 평가하고 비연결 노드 간 잘못된 유사성을 보정하여 올바른 구조적 해석을 보장합니다.

- **Performance Highlights**: TDAR는 그래프의 전반적인 연결성을 활용해 속성 복구의 질을 높이며, 특히 노드 분류 및 클러스터링 작업에서 사실상 전통적 방법들에 비해 뛰어난 성능을 보입니다. 본 프레임워크는 노드 종류에 따른 다양한 중요성을 반영하여 복잡한 topological 관계를 지능적으로 모델링합니다. 이러한 방식 덕분에 TDAR는 실험을 통해 매력적인 결과를 도출하며,AMGs의 도전적인 문제를 해결하는 데 필요한 강력한 솔루션을 제시합니다.



### Exploring the Impact of Generative Artificial Intelligence in Education: A Thematic Analysis (https://arxiv.org/abs/2501.10134)
- **What's New**: 최근의 생성형 인공지능(Generative Artificial Intelligence, GenAI) 기술 발전은 교육 분야에 많은 변화를 가져왔습니다. 특히 ChatGPT와 Bard와 같은 대형 언어 모델(Large Language Models, LLMs)은 반복 작업을 자동화하고 개인화된 교육 콘텐츠를 생성하는 데 활용될 수 있습니다. 그러나 이러한 도구의 책임감 있는 통합을 보장하기 위해 교육 분야에서 지침과 정책이 필요합니다.

- **Technical Details**: 이 연구에서는 교육 분야 전문가들로부터 제출받은 여섯 개의 에세이에 대한 주제 분석(thematic analysis)이 수행되었습니다. 이를 통해 GenAI 모델 이용의 이점과 단점을 통찰하고, 교수진 견해에 대한 Exploratory Data Analysis (EDA)를 통해 추가적인 통찰을 도출했습니다. 이 과정에서 에세이들로부터 파생된 여러 주제가 발견되었습니다.

- **Performance Highlights**: 연구 결과,GenAI 도구의 여러 이점과 단점이 밝혀졌으며, 학생들이 이러한 도구를 책임감 있고 윤리적인 방식으로 이용할 수 있도록 하기 위한 제안이 제시되었습니다. 특히 AI 도구가 교육에서의 개인화된 튜터 역할을 수행할 수 있지만,학업 무결성과 일반적인 기술의 남용 가능성에 대한 우려도 고려해야 한다는 점이 강조되었습니다.



### Infrastructure for AI Agents (https://arxiv.org/abs/2501.10114)
- **What's New**: 이 논문에서는 AI 시스템이 개방된 환경에서 상호작용을 계획하고 실행할 수 있는 방법에 대해 다루고 있습니다. 특히, 'agent infrastructure'라는 개념을 도입하여 이러한 AI 에이전트가 기존 제도(legal and economic systems) 및 행위자(digital service providers, humans, other AI agents)와 상호작용하도록 설계된 기술 시스템과 공유 프로토콜을 제안하고 있습니다.

- **Technical Details**: 에이전트 인프라스트럭처는 세 가지 주요 기능, 즉 특정 에이전트 또는 행위자에 대한 행동 및 정보 귀속(attributing actions), 에이전트의 상호작용 형태(shape interactions), 그리고 유해한 행동 탐지 및 수정(detecting and remedying harmful actions)을 제공합니다. 이러한 기능을 통해, 우리는 사용자 인증(OpenID)과 같은 기존 시스템을 활용하여 에이전트의 책임성을 향상시키고 효율성을 높일 수 있는 다양한 방법을 논의합니다.

- **Performance Highlights**: 에이전트 인프라스트럭처는 웹 서비스와의 상호작용을 안전하게 수행할 수 있도록 지원하며, 잘못된 거래나 서비스 중단과 같은 부작용을 방지하는 데 중요한 역할을 합니다. 이를 통해 AI 에이전트의 사용이 확대될 수 있으며, 이는 결국 사회가 더 진보된 AI 에이전트를 수용할 수 있는 기반을 마련할 것입니다.



### LLM Reasoner and Automated Planner: A new NPC approach (https://arxiv.org/abs/2501.10106)
Comments:
          15 pages, 7 figures, extended version of the homonymous paper submitted to the Catalan Conference on Artificial Intelligent (CCIA) 2025

- **What's New**: 이 논문에서는 고전적인 행동 트리(Behaviour Trees)와 같은 전통적인 기술에서 발생하는 문제점을 인식하고, 결정을 내리는 데 LLM(Large Language Model)을 통합한 새로운 아키텍처를 제안합니다. 이 아키텍처는 주어진 문제에 대해 합리적이고 인간과 유사한 응답을 생성함으로써, 다양한 상황에서의 의사결정 능력을 갖춘 지능형 에이전트를 목표로 합니다. 특히, 이 시스템은 비상 시뮬레이션과 같은 실질적인 응용 프로그램에서 높은 유연성을 제공합니다.

- **Technical Details**: 제안된 시스템은 LLM을 사용하여 환경 상태에 기반한 목표를 결정하고, 고전적인 자동 계획(Automated Planning, AP) 알고리즘을 사용하여 이를 달성하기 위한 실행 가능한 계획을 생성합니다. 이 시스템의 아키텍처는 Reasoner 모듈(Large Language Model을 사용하여 목표를 생성하는), Planner 모듈(선택된 목표를 달성하기 위한 계획을 생성하는), 그리고 환경과의 인터페이스 모듈로 구성되어 있습니다. 이러한 모듈 모두가 협력하여 LLM의 강점을 극대화하고, 그 단점을 최소화합니다.

- **Performance Highlights**: 실 구현 예로는 '소방관 문제(FireFighter Problem)'가 제시되며, 여기서 에이전트는 소화기와 사람, 안전 구역 등의 요소와 상호작용합니다. 에이전트는 주어진 정보를 바탕으로 목표를 설정하고 실행 가능한 행동 계획을 세우며, 일련의 작업을 통해 목표를 달성하도록 설계되었습니다. 이 시스템은 다양한 역할과 성격 특성을 가진 에이전트를 시뮬레이션할 수 있어, 비상 상황에서의 인간처럼 그럴듯한 행동을 보여줄 수 있습니다.



### A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks (https://arxiv.org/abs/2501.10069)
- **What's New**: 최근 LLM(inference)의 test-time compute를 향상시키기 위한 search화가 부각되고 있습니다. 그러나 현재의 프레임워크는 task 정의, LLM 프로파일링, search 절차의 세 가지 주요 측면에서 상이한 관점을 채택하고 있어 비교가 어렵습니다. 이 서베이는 이러한 측면을 통합하여 명확한 비교를 가능하게하고, 기존 검색 알고리즘과의 차이를 강조합니다.

- **Technical Details**: 우리는 임무를 MDP(Markov Decision Process) 구조로 표준화하고, 일반적인 MDP 해결에 사용되는 구성 요소인 정책(polices), 가치 함수(value functions), 전환 모델(transition models)으로 LLM 프로파일링과 프롬프트를 모듈화할 수 있음을 보여줍니다. LLM-Profiled Roles (LMPRs)의 세 가지 유형을 정의하여 구조와 기능의 차이를 비교합니다. 이러한 분류는 검색 방법을 채택하거나 확장할 때의 유연성을 증대시키고, 오버헤드를 최소화합니다.

- **Performance Highlights**: 우리는 11개의 프레임워크를 개별적으로 검토하고, 이러한 프레임워크가 기존 검색 알고리즘에서 벗어나거나 이를 향상시키는 방식을 분석했습니다. 특히, LLM이 전통적인 검색 프로세스를 수정 또는 강화하는 방법을 심층적으로 이해할 수 있게 합니다. 성능과 효율성 측면에서도 이 방법을 비판적으로 검토하며, 기존 연구들에 대한 참조를 통해 더 나은 적용 가능성을 탐구합니다.



### AirRAG: Activating Intrinsic Reasoning for Retrieval Augmented Generation via Tree-based Search (https://arxiv.org/abs/2501.10053)
Comments:
          17 pages, 14 figures

- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 방식의 새로운 패턴인 AirRAG를 제안합니다. AirRAG는 시스템 분석(System Analysis)과 효율적인 추론(Action)에 대한 통합을 통해 본래의 내재적 추론 능력을 활성화하고 Monte Carlo Tree Search (MCTS)를 통해 특정 작업의 솔루션 공간을 확장합니다. 이 접근법은 다섯 가지 기본 추론 행동을 설계하여 보다 넓고 깊은 tree-based reasoning space를 생성합니다.

- **Technical Details**: AirRAG의 핵심은 MCTS와 self-consistency 검증을 통한 효율적인 추론 경로 생성을 가능하게 하며, 다양한 문제 상황에서 사용할 수 있는 다섯 가지 기본 추론 행동을 포함합니다. 이러한 행동은 복잡한 문제 해결에 필요한 프로세스와 병렬 쿼리 또한 효율적으로 처리할 수 있습니다. 실험 결과, AirRAG는 기존의 iterative 및 recursive RAG에 비해 획기적인 성능 향상을 보이고 있습니다.

- **Performance Highlights**: AirRAG는 복잡한 QA 데이터셋에서 상당한 성능 향상을 보여주어 그 효과성을 입증했습니다. 또한, AirRAG의 유연하고 경량화된 구조는 다른 고급 기술과의 통합을 용이하게 하며, 이를 통해 더욱 많은 추론 계산을 주요 행동에 적용해 추가적인 성능 개선을 이루었습니다.



### Spatiotemporal Prediction of Secondary Crashes by Rebalancing Dynamic and Static Data with Generative Adversarial Networks (https://arxiv.org/abs/2501.10041)
- **What's New**: 이 연구에서는 급작스러운 교통 이벤트 분석과 예측에서 흔히 발생하는 데이터 불균형 문제를 해결하기 위해 VarFusiGAN-Transformer라는 하이브리드 모델을 제안합니다. 이 모델은 초래 충돌(secondary crashes)과 같은 특정 상황에서의 데이터 생성 정확도를 높이고, 붐비는 교통 상황에 대한 예측을 향상시키는 데 목표를 두고 있습니다.

- **Technical Details**: VarFusiGAN-Transformer 모델은 Long Short-Term Memory (LSTM) 네트워크를 사용하여 다변량(long-time series) 데이터 생성을 강화합니다. 이 모델은 정적(static) 데이터 생성기와 보조 판별기(auxiliary discriminator)를 사용하여 동적(dynanic) 및 정적 특성의 공동 분포(joint distribution)를 모델링합니다. 예측 모듈은 초래 충돌의 발생과 시공간(spatiotemporal) 분포를 동시에 예측할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 모델은 기존 방법들과 비교했을 때 데이터 생성의 충실도(fidelity)와 예측 정확도를 개선하는 데 있어 우수한 성능을 보여줍니다. VarFusiGAN-Transformer는 복잡한 교통 사고 데이터의 특성을 더 잘 처리하여 예측의 일반화 능력을 향상시킵니다.



### Enhancing Crash Frequency Modeling Based on Augmented Multi-Type Data by Hybrid VAE-Diffusion-Based Generative Neural Networks (https://arxiv.org/abs/2501.10017)
- **What's New**: 본 연구에서는 교통 사고 발생 빈도 모델링에 대한 차별화된 접근 방식을 제안합니다. 전통적인 방법들이 가진 한계를 극복하기 위해 Hybrid VAE-Diffusion(하이브리드 VAE-디퓨전) 신경망을 활용하여 제로 관측값(zero observations)을 줄이고 다양한 데이터 타입을 처리할 수 있도록 설계했습니다. 이는 교통 안전과 관련된 데이터의 정확성을 향상시키는 데 기여할 수 있습니다.

- **Technical Details**: 이 모델은 여러 유형의 테이블 형식 데이터(count, ordinal, nominal, real-valued variables)를 처리할 수 있으며, 제로 관측값 문제를 해결하는 데 중점을 두고 있습니다. 합성 데이터(synthetic data)의 품질은 유사성(similarity), 정확성(accuracy), 다양성(diversity), 구조적 일관성(structural consistency) 등의 지표로 평가됩니다. 또한 이 모델의 예측 성능은 전통적인 통계 모델과 비교되어 더 우수한 결과를 보여줍니다.

- **Performance Highlights**: Hybrid VAE-Diffusion 모델은 모든 지표에서 기준 모델들보다 성능이 우수하다는 결과를 도출하였습니다. 이러한 성과는 교통 사고 빈도를 보다 정확하게 예측하고 교통 안전을 향상시키는 데 필수적인 데이터 증강(data augmentation)에 기여할 수 있음을 시사합니다. 이 연구 결과는 정책 결정 및 자원 배분의 효과성을 높이는 데 중요한 인사이트를 제공합니다.



### ForestProtector: An IoT Architecture Integrating Machine Vision and Deep Reinforcement Learning for Efficient Wildfire Monitoring (https://arxiv.org/abs/2501.09926)
Comments:
          Accepted for publication in the proceedings of the 11th International Conference on Automation, Robotics, and Applications (ICARA 2025)

- **What's New**: 이번 연구는 저비용의 숲 화재 탐지 시스템을 제안합니다. 이 시스템은 360° 시야에서 연기를 원거리에서 모니터링할 수 있는 컴퓨터 비전 기능을 갖춘 중앙 게이트웨이 장치를 중심으로 구성됩니다. 또 다른 특징은 Deep Reinforcement Learning (DRL) 에이전트가 카메라의 방향을 동적으로 조절하여 실시간 센서 데이터를 활용한다는 점입니다.

- **Technical Details**: 제안된 시스템은 IoT 센서 노드와 중앙 게이트웨이로 이루어져 있습니다. 각 센서 노드는 온도, 습도, 기압, 연기 및 수치를 감지하는 센서를 사용하여 환경 조건을 모니터링합니다. 중앙 게이트웨이는 NVIDIA Jetson Nano 카드에서 LoRa 프로토콜을 통해 데이터를 수집하고, DRL 에이전트를 사용하여 카메라 시점을 제어합니다. 이 시스템은 MongoDB 데이터베이스에 저장된 데이터를 기반으로 실시간 시각화를 제공합니다.

- **Performance Highlights**: 저비용의 IoT 솔루션을 통해 대규모 지역에서 화재 탐지의 정확도를 향상시킬 수 있습니다. 제안된 시스템은 false positive를 줄이는 데 초점을 맞추고 있으며, 실시간으로 데이터를 전송하여 신속한 반응이 가능합니다. 또한 이 시스템은 AWS EC2 인스턴스에서 운영되며, WhatsApp을 통해 긴급 알림을 받을 수 있는 기능도 제공합니다.



### GenSC-6G: A Prototype Testbed for Integrated Generative AI, Quantum, and Semantic Communication (https://arxiv.org/abs/2501.09918)
Comments:
          SUBMITTED FOR PUBLICATION IN IEEE COMMUNICATIONS MAGAZINE

- **What's New**: 이번 논문에서는 생성적 인공지능(Generative AI), 양자 컴퓨팅(Quantum Computing), 및 의미론적 통신(Semantic Communication)을 통합하여 새로운 6세대(6G) 애플리케이션을 지원하는 종합 데이터셋인 GenSC-6G를 개발한 프로토타입 테스트베드에 대해 소개합니다. GenSC-6G 데이터셋은 노이즈가 추가된 합성 데이터로 설계되어 의미론적 디코딩(semanic decoding), 분류(classification), 로컬라이제이션(localization) 작업에 최적화되어 다양한 AI 기반 통신 애플리케이션에 대한 유연성을 크게 향상시킵니다.

- **Technical Details**: GenSC-6G 데이터셋은 분류(classification), 분할(segmentation), 객체 탐지(object detection), 그리고 엣지 기반 언어 모델링(edge LLM) 작업을 지원하도록 세심하게 조직되어 있습니다. 각 머신러닝(mL) 또는 의미론적 작업은 독립적인 진실 데이터 수집(ground-truth data collection)과 연결된 여러 수집의 조합과 관련이 있으며, 이를 통해 모델이 다양한 연결된 작업 간에 효과적으로 훈련되고 평가될 수 있도록 보장합니다.

- **Performance Highlights**: 실제 활용 사례를 통해 경량 분류, 의미론적 업샘플링(semantic upsampling), 노이즈 조건에서의 엣지 기반 언어 추론의 성능을 평가하는 기회를 제공합니다. GenSC-6G 데이터셋은 6G 네트워크의 증가하는 요구에 맞춘 목표 지향적 통신 시스템을 개발하기 위한 확장 가능하고 강력한 자원으로 기능합니다.



### Towards A Litmus Test for Common Sens (https://arxiv.org/abs/2501.09913)
- **What's New**: 이번 논문은 안전하고 유익한 인공지능(AI)을 위한 경로를 구상하는 일련의 작업 중 두 번째로, 'Common Sense Is All You Need'의 개념적 통찰을 바탕으로 합니다. 공통 상식(common sense)에 대한 보다 공식적인 리트머스 테스트(litmus test)를 제안하며, 최소한의 사전 지식(minimal prior knowledge, MPK) 제약과 대각선 또는 뤼델 스타일의 주장을 결합한 접근 방식을 채택합니다. 이러한 접근법은 고급 AI 시스템이 지식의 간극을 숨기기 위해 거짓 정보를 의도적으로 생성하는데 대해 논의하고, 이러한 경향이 안전과 신뢰를 어떻게 저해할 수 있는지를 강조합니다.

- **Technical Details**: 논문은 최소한의 사전 지식(MPK)과 대각선 논증을 결합하여 AI의 진정한 공통 상식을 평가하는 리트머스 테스트 설계를 제안합니다. 이는 데이터가 아닌 직관적인 문제 해결을 중심으로 하며, 특히 Abstraction and Reasoning Corpus (ARC)에서의 적용 가능성을 강조합니다. 이 테스트는 AI가 진정으로 새로운 개념을 처리할 수 있는지를 진단할 뿐만 아니라, 이를 통해 윤리적이고 신뢰할 수 있는 AI의 토대를 마련하는 데 기여하고자 합니다.

- **Performance Highlights**: 이 연구에서 목표하는 리트머스 테스트는 AI가 직면한 문제에 대해 새로운 개념을 발명하는 능력을 평가합니다. 이러한 접근은 모델이 대규모 패턴만으로 해결할 수 없는 본질적인 작업에 도전함으로써, 변화를 독창적으로 창출할 수 있는지를 판단합니다. 이를 통한 AI의 일반화 능력을 증진시켜, 안전하고 유익한 AI 시스템을 구축하는 데 기여할 것으로 기대됩니다.



### Evolving Deeper LLM Thinking (https://arxiv.org/abs/2501.09891)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 추론 시간 컴퓨팅을 확장하는 진화적 탐색 전략인 Mind Evolution을 제안합니다. 이 접근법은 언어 모델을 활용하여 후보 응답을 생성, 재조합 및 정제하여 자연어 계획 작업에서 Best-of-N 및 Sequential Revision 같은 다른 추론 전략들보다 월등한 성능을 보입니다. Mind Evolution은 평가자를 통해 피드백을 받아 자유롭게 솔루션 후보를 진화시키며, LLM을 직접 최적화합니다.

- **Technical Details**: Mind Evolution은 유전적 탐색 전략을 채택하며, 이는 자연선택에 영감을 받은 메타 휴리스틱 방법입니다. 각 후보 솔루션은 '적합성' 함수를 기준으로 평가되며, 높은 품질의 솔루션 비율을 증가시키도록 진화합니다. 이 과정은 독립적으로 생성된 후보 솔루션들의 집단에서 시작하여, 개체의 유전적 표현을 돌연변이 및 재조합하여 새로운 솔루션을 생성하는 방식입니다.

- **Performance Highlights**: 여러 실험에서 Mind Evolution을 활용한 Gemini 1.5 Flash는 TravelPlanner에서 95.6%, Meeting Planning에서는 85%의 성공률을 기록했습니다. 추가적으로, unsolved 문제 인스턴스는 Gemini 1.5 Pro를 통해 100%의 성공률을 달성하며, 이는 LLM 최적화의 새로운 기준을 제시합니다. 마지막으로, 새로운 벤치마크 문제인 StegPoet에서는 87%의 성공률을 달성하여 자연어 도메인에서도 탐색의 적용 가능성을 입증했습니다.



### Exploring the Implementation of AI in Early Onset Interviews to Help Mitigate Bias (https://arxiv.org/abs/2501.09890)
- **What's New**: 이 논문은 초기 단계의 채용 인터뷰에서 인공지능(AI)을 활용하여固有(bias)를 줄이는 방법을 연구합니다. 특히, 감정적 편향(sentiment bias)을 줄이는데 초점을 맞추었습니다. 전통적인 면접관들은 여러 가지 편향에 영향을 받을 수 있으며, 이는 비포함적인 채용 관행을 초래하고 다양성이 낮은 인력을 형성하게 됩니다.

- **Technical Details**: 연구는 현재 시장에서 사용되고 있는 다양한 AI介入(intervention)을 분석합니다. 다중 모드 플랫폼(multimodal platforms)과 상호작용 후보자 평가 도구(interactive candidate assessment tools) 등을 살펴보며 AI가 초기 채용 과정에서 어떻게 활용되고 있는지를 파악합니다. 또한, 기존에 개발된 독특한 AI 시스템을 사용하여 면접 역학을 분석하고, 감정보다 기술과 지식에 중점을 두고 있습니다.

- **Performance Highlights**: AI는 감정 기반의 편향을 41.2% 효과적으로 최소화했다는 결과를 도출하였습니다. 이는 기업의 채용 프로세스에서 공정성과 효율성을 향상시킬 수 있는 혁신적인 가능성을 보여줍니다. 이 연구는 AI가 채용 과정에서의 편향 문제를 해결하는 데 있어서 중요한 역할을 할 수 있음을 제시합니다.



### 3rd Workshop on Maritime Computer Vision (MaCVi) 2025: Challenge Results (https://arxiv.org/abs/2501.10343)
Comments:
          Part of the MaCVi 2025 workshop

- **What's New**: 해양 컴퓨터 비전에 대한 제3회 워크숍(MaCVi 2025)은 무인 수상 차량(Unmanned Surface Vehicles, USV) 및 수중 시스템에 초점을 맞추고 있습니다. 이 보고서는 700개 이상의 제출물에서 도출된 통계적 및 정성적 분석 결과를 종합적으로 제시합니다. 데이터셋, 평가 코드 및 리더보드는 누구나 접근할 수 있도록 공개되었습니다.

- **Technical Details**: 해양 환경의 고유한 도전 과제를 해결하기 위해 독창적인 비전 알고리즘 개발이 요구됩니다. 특히, 거리 추정(Distance Estimation) 및 객체 탐지(Object Detection) 기술이 필요하며, 참가자들은 약 3,000개의 레이블링된 훈련 샘플이 포함된 데이터셋을 활용해야 합니다. 우수한 모델은 ONNX 포맷으로 내보내야 하며, 제출 제한은 챌린지마다 하루 1-3회로 설정됩니다.

- **Performance Highlights**: 참여자들은 모델의 성능을 평가하기 위한 다양한 지표를 사용해야 하며, 여기에는 객체 탐지의 평균 정밀도(AP)와 거리 추정 정확도가 포함됩니다. 거리 예측의 정확도를 평가하기 위해 절대 오차(absolute error)와 상대 오차(relative error)도 평가합니다. 제출된 모델은 50만 개의 파라미터를 초과해서는 안 되며, 총 60개의 제출물이 6개 팀으로부터 접수되었습니다.



### Agent4Edu: Generating Learner Response Data by Generative Agents for Intelligent Education Systems (https://arxiv.org/abs/2501.10332)
Comments:
          Accepted by AAAI2025

- **What's New**: 이 논문에서는 개인화된 학습을 위한 새로운 시뮬레이터인 Agent4Edu를 소개합니다. 본 시스템은 대규모 언어 모델(LLMs)의 발전을 활용하여 학습자 반응 데이터를 시뮬레이션하고, 실제 학습자의 학습 패턴을 반영합니다. 이로 인해 기존의 오프라인 메트릭과 온라인 성과 간의 간극을 해소하여, 지능형 학습 시스템에서의 개인화된 학습 효율을 높이는 데 기여할 것으로 기대됩니다.

- **Technical Details**: Agent4Edu는 학습자 프로파일, 메모리, 행동 모듈을 갖춘 LLM 기반의 생성 에이전트를 특징으로 합니다. 각 에이전트는 실제 반응 데이터를 기반으로 초기화되며, 메모리 모듈은 학습 경험과 반성을 통합해 학습 상태를 요약합니다. 행동 모듈은 학습 알고리즘과 상호작용하여 수동적이 아닌 능동적인 학습 반응을 생성할 수 있도록 지원합니다.

- **Performance Highlights**: 우리는 Agent4Edu의 효과를 평가하기 위해 포괄적인 실험을 수행했으며, 에이전트와 인간 학습자 간의 일관성을 검토했습니다. 실험 결과, Agent4Edu는 기존의 학습자 시뮬레이션 기법보다 더 신뢰할 수 있는 반응 데이터와 문제 해결 능력을 시뮬레이션하는 데 성공했습니다. 이로 인해 개인화된 학습 알고리즘의 향상을 확인할 수 있었습니다.



### Hierarchical Autoregressive Transformers: Combining Byte-~and Word-Level Processing for Robust, Adaptable Language Models (https://arxiv.org/abs/2501.10322)
- **What's New**: 이 논문에서는 텍스트를 처리하기 위한 새로운 Tokenization 방법을 제안합니다. 기존의 subword tokenizers의 한계를 극복하기 위해 계층적(혹은 hierarchical) 모델 아키텍처를 활용하여, 문자 수준(character-level)과 단어 수준(word-level) 처리를 결합하였습니다. 이는 고정된 vocabulary에 의존하지 않으면서도 단어 수준 Tokenization의 장점을 유지합니다. 이를 통해 다양한 언어와 도메인에 유연하게 대응할 수 있는 NLP 시스템을 구축할 수 있습니다.

- **Technical Details**: 제안된 아키텍처는 텍스트를 단어 단위로 분할한 후, 각 단어의 문자를 작은 문자 수준 인코더를 통해 단어 임베딩(word embedding)으로 변환합니다. 그 다음, 이들 단어 임베딩은 더 큰 백본(backbone) 모델에 의해 처리되고, 최종 출력은 소형 문자 수준 디코더를 통해 문자로 디코딩됩니다. 이러한 계층적 구조는 사전 학습된 모델이 새로운 도메인이나 언어의 텍스트에 적용될 때 발생하는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 최대 70억 개의 파라미터를 가진 계층적 트랜스포머(hierarchical transformers)가 기존의 subword tokenizer 기반 모델과 동등한 성능을 보이는 동시에 입력 변동성에 대한 강인성(robustness)이 크게 향상됨을 보여주었습니다. 또한, 새로운 언어에 대한 지속적 사전 학습(pretraining)中 모델이 약 두 배 더 빠르게 훈련되고 목표 언어에서 탁월한 성능을 발휘하며 이전에 획득한 지식을 더 많이 유지하는 것으로 나타났습니다.



### SEANN: A Domain-Informed Neural Network for Epidemiological Insights (https://arxiv.org/abs/2501.10273)
- **What's New**: 이번 연구에서는 전통적인 통계 방법이 아닌 SEANN(요약 효과 적응 신경망)이라는 혁신적인 방법론을 소개합니다. SEANN은 Deep Neural Networks(DNNs)에 Pooled Effect Sizes(PES)라는 형태의 도메인 특화 지식을 통합해 예측 성능을 향상시키고자 합니다. 연구에서는 이 방법의 적용이 기존의 도메인 지식이 없는 신경망보다 예측 성능의 일반화 가능성을 크게 향상시킨다고 보고합니다.

- **Technical Details**: SEANN은 PES를 통해 DNN 학습 과정에 직접적으로 통합합니다. PES는 메타 분석 연구에서 자주 발견되며, 과학적 합의를 정량화한 형태입니다. 이 연구에서는 PES를 통해 모델의 일반화 능력을 개선하고, 과학적으로 타당한 관계를 도출할 수 있도록 맞춤형 손실 함수(custom loss function)를 설계하여 이를 적용했습니다.

- **Performance Highlights**: 실험 결과, SEANN은 제한적이고 노이즈가 많은 데이터 환경에서 예측 성능의 신뢰성을 크게 향상시켰습니다. SEANN은 DNN의 한계를 극복하고, 과거 연구에서 얻어진 도메인 지식을 결합함으로써 잠재적으로 더 정확한 위험 지표를 계산할 수 있음을 보여줍니다. 이러한 성과를 통해 SEANN은 역학 연구 및 관련 분야에서 기대되는 새로운 기법으로 자리매김할 가능성을 보여줍니다.



### Unsupervised Rhythm and Voice Conversion of Dysarthric to Healthy Speech for ASR (https://arxiv.org/abs/2501.10256)
Comments:
          Accepted at ICASSP 2025 Satellite Workshop: Workshop on Speech Pathology Analysis and DEtection (SPADE)

- **What's New**: 본 논문에서는 디사르트리아(dysarthria) 환자의 음성을 일반적인 음성으로 변환하기 위한 새로운 접근법인 Rhythm and Voice (RnV) 변환 프레임워크를 제안합니다. 기존 방법들과 달리, 이 연구는 비지도 학습(self-supervised learning) 기반의 음성 표현을 통해 리듬(rhythm)과 음성 변환(voice conversion)을 수행하여 디사르트리아 음성을 다룹니다. 특히, 이 연구는 대규모 ASR 모델에 대해 추가적인 세밀 조정 없이도 성능 향상을 확인했습니다.

- **Technical Details**: Rhythm and Voice 변환 프레임워크는 SSL 음성 인코더를 사용하여 출처 음성에서 특성을 추출하고, 타겟 화자의 리듬 특성을 모델링하기 위해 클러스터링 기술을 적용합니다. kNN-VC 모델을 통해 음성을 변환하고, 이후에 학습된 vocoder를 사용하여 변환된 음성을 파형으로 복원합니다. 이 접근법은 각 화자의 고유한 리듬을 효과적으로 전달할 수 있도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, Torgo 코퍼스의 심각한 디사르트리아 화자에 대해 리듬 변환이 ASR 성능 향상에 특히 효과적임을 입증했습니다. ASR은 웅장한 모델인 Whisper를 활용하였고, 여러 시험에서 디사르트리아 음성을 일반 화자의 음성으로 변환하는 데 성공하였습니다. 이 연구는 이러한 기술이 말하기 능력이 제한된 화자들에 대한 지원 기술 개발에 기여할 것임을 강조합니다.



### Random-Key Algorithms for Optimizing Integrated Operating Room Scheduling (https://arxiv.org/abs/2501.10243)
Comments:
          38 pages, Preprint submitted to Applied Soft Computing

- **What's New**: 이번 연구는 Random-Key Optimizer (RKO) 개념을 도입하여 효율적인 수술실 예약 문제를 해결하기 위한 새로운 접근법을 제안하고 있습니다. 이 연구는 기존 문헌을 기반으로 새로운 실제 사례에 대한 엄격한 테스트를 통해 이 개념의 유효성을 입증하였으며, 수술, 장비, 그리고 회의실 가용성 제약을 통합한 복잡한 조합 최적화 문제를 포함합니다. 이를 통해 수술 일정을 개선하고 자원 활용도를 최적화하고자 하였습니다.

- **Technical Details**: RKO 접근법은 문제의 해결책을 연속 공간의 점으로 설정한 뒤, 이를 디코더라는 결정론적 함수에 의해 문제 해결 공간으로 매핑합니다. 이 연구에서는 Biased Random-Key Genetic Algorithm (BRKGA)와 Q-Learning, Simulated Annealing, Iterated Local Search를 혼합하여 활용하는 메타휴리스틱을 설계하였으며, 이는 모든 문제에 대해 단일 디코더 함수를 사용하도록 최적화되었습니다. 또한, 제안된 메타휴리스틱은 하한식과 함께 최적성을 비교하여 평가합니다.

- **Performance Highlights**: 결과적으로, 본 연구는 문헌에서 제안된 기존 방법보다 더 개선된 하한 및 상한을 보여주었으며, 실제 세계의 제약이 많은 상황에서도 효율적으로 수술 일정을 생성하는 능력을 입증했습니다. 최종적으로, 이 방법은 자원 할당을 최적화하고 환자 대기 시간을 줄이며 전체 운영 효율성을 향상시키는 실질적인 솔루션을 제공하며, 병원 운영에 유용한 통찰을 제공합니다.



### Challenges and recommendations for Electronic Health Records data extraction and preparation for dynamic prediction modelling in hospitalized patients -- a practical guid (https://arxiv.org/abs/2501.10240)
- **What's New**: 이 논문은 전자 건강 기록(EHR) 데이터를 활용한 동적 예측 모델 개발 과정에서 데이터 추출과 준비 단계에서 발생하는 40가지 이상의 도전을 소개합니다. 또한 이러한 문제를 해결하기 위한 구체적인 권장 사항을 제시하여 연구자 및 데이터 추출 엔지니어들이 실질적인 도움을 받을 수 있도록 합니다. 이 연구는 예측 모델의 품질과 실제 적용 가능성을 향상시키기 위한 지침을 제공합니다.

- **Technical Details**: EHR 데이터 분석은 데이터 수집, 추출, 준비의 세 단계를 포함하여 예측 모델이 구축됩니다. 데이터는 복잡한 관계형 데이터베이스에서 구조화된 형식으로 추출되고, OMOP CDM(Observational Medical Outcomes Partnership Common Data Model) 같은 표준을 이용하여 정리됩니다. 데이터 준비 과정에서는 tidyverse, tidymodels, pandas와 같은 도구 세트를 사용해 모델이 사용 가능한 형식으로 전환됩니다.

- **Performance Highlights**: 이 연구는 동적 예측 모델의 개발을 위한 실용적인 접근법을 제공하며, 데이터 품질 평가와 관련한 새로운 프레임워크인 METRIC을 도입하여 예측 연구에 특화된 방법론을 제안합니다. 연구자들이 데이터 품질 문제를 구조적 또는 비구조적으로 평가하는 방법을 다루며, 데이터 추출 및 준비의 중요성을 강조합니다. 최종 데이터셋을 예측 작업에 적합하게 만들기 위한 단계를 정확히 다루는 것이 본 논문의 주요 기여 중 하나입니다.



### Good things come in small packages: Should we adopt Lite-GPUs in AI infrastructure? (https://arxiv.org/abs/2501.10187)
Comments:
          5+ pages, 4 figures

- **What's New**: 최근 AI 수요 증가에 따라 GPU 설계자들은 단일, 복합적인 패키지에 더 많은 계산 능력과 메모리를 조합하려고 하고 있습니다. 그러나 최신 GPU는 포장, 수율, 냉각의 한계를 보이고 있으며, 이에 따라 AI 클러스터의 확장 가능성에 대한 불확실성이 커지고 있습니다. 본 논문은 Lite-GPU란 개념을 통해 이러한 문제점을 해결하며, 효율적으로 연결된 대규모 Lite-GPU 클러스터를 설계할 것을 제안하고 있습니다.

- **Technical Details**: Lite-GPU는 단일, 작은 compute die를 가진 GPU로, 기존 GPU에 비해 낮은 생산 비용과 높은 대역폭 대비 컴퓨팅 비율을 자랑합니다. 최근 코팩 패키징(optics) 기술을 통해 AI 작업 부하를 더 많은 Lite-GPU에 분배하는 통신 문제를 극복할 수 있으며, 이는 차세대 데이터 센터의 효율성을 크게 개선할 것입니다. 특히, 최신 기술 덕분에 비약적인 대역폭 증가와 함께 광학 인터커넥트(optical interconnects)의 발전이 기대됩니다.

- **Performance Highlights**: Lite-GPU 클러스터는 기존 GPU와 비교하여 더욱 높은 성능 효율성을 제공할 잠재력을 가지고 있으며, 특히 I/O 집약적인 연산에서 현재 GPU와 유사하거나 더 나은 성능을 달성할 수 있는 가능성을 보여주고 있습니다. 제조 비용 또한 낮추어질 것으로 예상되며, 이러한 Lite-GPU의 설치와 운영은 데이터 센터 관리에서의 문제 해결을 포함한 다양한 기회를 창출할 수 있습니다.



### A Simple but Effective Closed-form Solution for Extreme Multi-label Learning (https://arxiv.org/abs/2501.10179)
Comments:
          10pages, Accepted at ECIR25

- **What's New**: 이 논문은 극단 멀티 레이블 학습(Extreme Multi-Label Learning, XML)의 문제를 해결하기 위해 리지 회귀(ridge regression)를 사용한 새로운 접근 방식을 제안합니다. 기존 모델들이 많은 하이퍼파라미터를 포함하여 조정이 복잡한 반면, 이 방법은 단일 하이퍼파라미터만으로 모델을 최적화하는 간단한 솔루션을 제공합니다. 또한, 이 방법은 저빈도 레이블에 대한 예측 성능을 개선하여 중요한 정보를 포함할 수 있는 가능성을 지니고 있습니다. 실험 결과는 제안된 방법이 기존 모델들에 비해 경쟁력 있는 성능을 나타냄을 보여줍니다.

- **Technical Details**: XML 데이터셋은 각 인스턴스의 특징 벡터와 레이블 벡터로 구성됩니다. 제안된 리지 회귀 방법은 최소제곱법과 L2 정규화(L2 regularization)를 기반으로 하며, 저빈도 레이블에 대한 예측을 위해 라벨별 가중치(label-specific weights)를 통합하는 방식을 제공합니다. 이 모델은 단일 하이퍼파라미터만 사용하므로 구현이 단순하고 해석하기 용이합니다. 리지 회귀를 통해 XML 작업을 모델링할 때, 데이터 포인트가 적은 저빈도 레이블의 예측이 중요한 고려사항입니다.

- **Performance Highlights**: 제안된 리지 회귀 모델은 다양한 XML 기준 데이터셋에서 실험되었으며, 그 결과 기존의 다양한 하이퍼파라미터를 가진 모델들과 비슷하거나 더 나은 성능을 달성했습니다. 특히, 저빈도 레이블의 예측 성능이 상당히 개선되었으며, 기존 방법과 비교해 거의 변경 없이 구현될 수 있음을 보여주었습니다. 이러한 접근은 XML 작업을 보다 간단하게 만들어 주며, 향후 연구 및 응용 가능성을 제시합니다.



### Region-wise stacking ensembles for estimating brain-age using MRI (https://arxiv.org/abs/2501.10153)
Comments:
          version1

- **What's New**: 이 연구에서는 구조적 자기공명영상(Structural MRI) 데이터를 이용한 예측 모델링의 새로운 접근 방식을 소개합니다. 전통적인 방법인 보간(resampling)이나 평균화(averaging)로 인한 정보 손실을 극복하기 위해, 새로운 두 단계 스태킹 앙상블(two-level stacking ensemble, SE) 방법을 제안합니다. 이 방법은 개별적으로 얻어진 생리학적 정보로부터 개인의 나이를 보다 정확하게 예측할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 SE 접근법은 두 개의 수준으로 구성되어 있습니다. 첫 번째 수준은 복셀(voxel) 단위의 정보를 기반으로 한 지역 모델을 통해 개인의 나이를 예측하고, 두 번째 수준의 모델이 최종 예측을 수행합니다. 이를 통해 데이터 융합(data fusion) 시나리오에 대한 구체적인 실험을 통해 다양한 데이터 세트에서 회색 물질 볼륨(Gray matter volume, GMV)을 입력으로 하여 수행됩니다.

- **Performance Highlights**: 연구 결과, MEA(Mean Absolute Error), R2, 상관관계(correlation), 예측 바이어스(prediction bias) 등의 지표에서 SE 방법이 지역 평균(region-wise averages)에 비해 우수한 성능을 보였습니다. 특히, 외부 샘플 예측이 이루어진 첫 번째 단계 모델과 독립적이며 특정 장소에서 훈련된 두 번째 단계 모델을 사용할 때 가장 높은 성능을 기록했습니다(MAE=4.75 vs baseline regional mean GMV MAE=5.68). 이 접근법은 나이 예측의 신뢰성을 높이며 데이터의 개인 정보를 더욱 안전하게 보호합니다.



### Dual Debiasing: Remove Stereotypes and Keep Factual Gender for Fair Language Modeling and Translation (https://arxiv.org/abs/2501.10150)
- **What's New**: 이번 연구에서는 성별 편향을 줄이면서 사실적인 성별 정보를 보존하는 새로운 방법인 Dual Debiasing Algorithm through Model Adaptation (2DAMA)를 도입합니다. 기존의 연구에서는 성별 대표성을 식별하고 사회적 편향을 완화하는 데 집중했으나, 2DAMA는 언어 모델의 성별 정보를 공평하게 표현하는 것을 목표로 합니다. 이 방법은 특히 번역에서 고전적인 편향 경향을 완화하는 데 기여하며, 언어 처리 작업에서 유용한 사실적 성별 단서를 보존할 수 있습니다.

- **Technical Details**: 2DAMA는 사전 훈련된 언어 모델에서 편향 신호를 효과적으로 제거할 수 있는 알고리즘적 구성 요소를 포함합니다. 핵심 아이디어는 편향을 줄이면서도 모델이 사실적인 성별 정보를 유지하도록 하는 것입니다. 이 방법은 기존의 DAMA와 LEACE 알고리즘을 결합하여 모델 성능을 손상시키지 않고 해로운 편향의 인코딩을 완화합니다.

- **Performance Highlights**: 실험 결과, 2DAMA는 LLM의 성별 편향을 효과적으로 줄이는 데 성공했으며, 이를 통해 고전적인 성별 고정관념이 언어 모델 훈련 중에 어떻게 증폭되는지를 보여줍니다. 또한 2DAMA는 네 개의 다양한 모델에서 성별 표현의 패턴을 분석하여 각기 다른 디자인 선택이 편향 제거에 미치는 영향을 탐구합니다. 이 연구의 결과는 LLM의 다국적 성별 편향을 완화하는 데 기여할 것으로 기대됩니다.



### Enhancing UAV Path Planning Efficiency Through Accelerated Learning (https://arxiv.org/abs/2501.10141)
Comments:
          This paper was accepted in this https URL conference but it is not available from the conference yet

- **What's New**: 본 논문에서는 무인 항공기(UAV)의 경로 계획 알고리즘을 위한 새로운 기법들을 제안합니다. 특히, DDPG 및 TD3 알고리즘의 개선을 위한 다양한 기법들을 통합함으로써 학습 시간을 단축시킬 수 있는 방법을 모색하고 있습니다. PCA 기반의 차원 축소 기법, 샘플 조합, 우선 경험 재생(Prioritized Experience Replay, PER), 손실 계산의 통합 방법 등이 포함되어 있습니다.

- **Technical Details**: 시스템 모델에서 UAV는 무선 중계기로서 지상 사용자에게 최소 품질 서비스를 제공하기 위해 동작합니다. 사용자는 UAV와 연결되며, UAV는 지상 기지국(BS)과 무선 백홀 링크를 통해 연결됩니다. 이 시스템은 사용자 위치를 기반으로 경로 손실 및 그림자 효과를 측정해 채널 전파 모델을 계산합니다.

- **Performance Highlights**: 제안된 방식은 전통적인 TD3 알고리즘 대비 훈련에 필요한 수렴 에피소드를 약 4배 줄이는 성과를 보여주었습니다. 이러한 성과는 UAV 통신 중계 문제를 해결하는 데 있어서 더욱 효율적이고 효과적인 학습을 가능하게 합니다. 이로 인해 무선 통신의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores (https://arxiv.org/abs/2501.10139)
- **What's New**: 이 논문에서는 표준 conformal prediction의 한계를 극복하는 새로운 대안을 제시합니다. 특히 잘못된 예측에 대해 과신하는 분류기의 예측에서 조건부 커버리지(conditional coverage)를 보장하는 알고리즘을 개발했습니다. 이 알고리즘은 모델의 신뢰도와 Bayes 최적 분류기와의 차이를 기록하는 신뢰 점수를 기반으로 합니다.

- **Technical Details**: 제안된 방법은 miscoverage 사건을 분석하여 신뢰도를 바탕으로 한 변수 집합을 만들어, 이를 통해 예측 집합을 형성합니다. 새로운 conformal prediction 변형은 신뢰도(Confidence)와 Bayes 분류기와의 불일치를 측정하는 비모수 신뢰 점수(Trust)를 포함하여 두 가지 변수에 대해 조건부 커버리지를 조정합니다. 이 접근 방식은 고차원 특성 공간에서도 효과적으로 근사 조건부 커버리지를 달성합니다.

- **Performance Highlights**: 다양한 이미지 데이터셋(ImageNet, Places365 등)에서 평가한 결과, 제안된 방법이 표준 conformal prediction과 비교하여 조건부 커버리지 향상에 기여함을 보여주었습니다. 특히, skin condition classification을 위한 Fitzpatrick 17k 데이터셋에서 피부 타입에 상관없이 커버리지를 개선하는 성과를 이뤘습니다. 이러한 결과는 의료 결정에서 예측의 신뢰성을 높일 수 있는 중요한 기초 자료로 활용될 수 있습니다.



### Spatio-temporal Graph Learning on Adaptive Mined Key Frames for High-performance Multi-Object Tracking (https://arxiv.org/abs/2501.10129)
- **What's New**: 이번 논문에서는 다중 객체 추적(Multi-Object Tracking)의 기존 한계를 극복하기 위해 적응형 키 프레임 마이닝 전략을 제안합니다. 이를 통해 강화 학습(Reinforcement Learning)을 활용하여 비디오를 동적으로 분할하고, 객체 간의 공간적 및 시간적 관계를 효과적으로 캡처할 수 있게 되었습니다. 또한, Intra-Frame Feature Fusion(IFF) 모듈을 도입하여 서로 다른 객체 간의 정보 교환을 촉진합니다.

- **Technical Details**: 키 프레임 추출(KFE) 모듈은 Q-learning에 기반해 설계되며, 최적의 분할 전략과 보상 메커니즘을 통해 비디오 세그먼트를 적응적으로 나누는 역할을 합니다. 이 과정에서 GCN(Graph Convolutional Network)을 사용하여 한 프레임 내의 객체 및 주변 객체 간의 정보 상호작용을 극대화하고, 이를 통해 객체 별로 고유 식별력을 높입니다. 이러한 기법들은 기존의 그래프 기반 방법과는 다른 접근 방식을 보여줍니다.

- **Performance Highlights**: 제안된 알고리즘은 MOT17 데이터셋에서 68.6 HOTA, 81.0 IDF1, 66.6 AssA 및 893 IDS의 성능을 기록하며 효과성과 정확성을 입증합니다. 전략적인 단기 및 장기 연관성을 모두 모델링하는 통합된 트래커로, 다양한 시나리오에 적응할 수 있는 포괄적인 해결책을 제공합니다. 실험 결과, 제안된 방식이 기존 방법들에 비해 개선된 성능을 보여주는 것으로 나타났습니다.



### BBPOS: BERT-based Part-of-Speech Tagging for Uzbek (https://arxiv.org/abs/2501.10107)
- **What's New**: 본 논문은 저자들에 의해 이전에 시험받지 않은 단일 언어 우즈벡 BERT 모델을 부분 품사 태깅(task) 작업에 적용하여 저자들이 처음으로 공개한 우즈벡어 UPOS 태깅 벤치마크 데이터셋을 소개합니다. 이 모델의 평균 정확도는 91%에 달하며, 다국어 mBERT 및 규칙 기반 태거보다 우수한 성능을 보입니다.

- **Technical Details**: 우즈벡어는 역사적 이유로 인해 라틴 문자와 키릴 문자를 혼합하여 사용하는 형태로, 다양한 크기와 품질의 사전 학습된 BERT 기반 모델들이 존재하지만, 공개된 벤치마크 데이터셋이 부재하여 평가가 미비했습니다. 저자들은 POS 태깅을 위해 500개의 문장을 포함한 새로운 데이터셋을 제작하였으며, 이 데이터셋은 UPOS 태그를 사용하여 수동으로 주석이 달렸습니다.

- **Performance Highlights**: 실험 결과, 단일 언어 BERT 모델들은 평균적으로 90% 이상의 정확도 및 84% 이상의 F1 점수를 기록했습니다. 특히, TahrirchiBERT보다 데이터 양이 10배 적게 학습된 UzBERT가 두 메트릭에서 약간 우수한 성능을 보였으며, 이는 사전 학습 데이터의 품질 차이에 기인할 수 있습니다.



### Universal Actions for Enhanced Embodied Foundation Models (https://arxiv.org/abs/2501.10105)
Comments:
          Preprint

- **What's New**: 이 논문에서는 다양한 로봇 플랫폼에서 일반화 가능한 행동을 포착하는 새로운 프레임워크인 UniAct를 소개합니다. UniAct는 서로 다른 행동 공간의 이질성을 해소하고, 다양한 환경과 임무에 적응할 수 있도록 설계된 보편적인 작업 공간인 Universal Action Space에서 작동합니다. 이번 Framework는 전통적인 이질적 행동 공간 문제를 해결하는 데 중요한 발전을 가져올 것으로 기대됩니다.

- **Technical Details**: UniAct는 Vision Language Model (VLM)을 사용하여 벡터 양자화된 코드북 형태로 보편적인 행동 공간을 구성합니다. 이 설정은 행동의 원시적인 패턴을 추출하여 다양한 로봇에서 구현 가능한 공통적인 행동을 인식하는 데 도움을 줍니다. 더불어, 각 로봇 플랫폼에 특화된 행동 명령으로 변환할 수 있는 효율적인 이질적 디코더를 통해 빠른 적응이 가능해집니다.

- **Performance Highlights**: UniAct-0.5B 모델은 7B 파라미터를 가진 기존의 최첨단 모델보다 14배 더 작은 규모임에도 불구하고 다양한 과제에서 월등한 성능을 보여주었습니다. 특히, 새로운 로봇 플랫폼에 빠르게 적응할 수 있는 능력을 강조하며, 보편적인 행동을 채택한 것이 얼마나 큰 이점을 가져오는지 입증하고 있습니다. 이 연구는 Universal Action Space 내에서의 효과적인 행동 모델 개발이 전통적인 이질적 공간에서 개발하는 것보다 훨씬 더 유리하다는 것을 보여줍니다.



### Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics (https://arxiv.org/abs/2501.10100)
- **What's New**: 이 연구에서는 복잡하고 부분적으로 관찰 가능하며 확률적(dynamics) 동작을 정확히 포착하는 새로운 세계 모델(world model) 학습 프레임워크를 제안합니다. 이 방법은 domain-specific inductive biases에 의존하지 않고, 이중 자기 회귀 메커니즘(dual-autoregressive mechanism)과 자기 지도 학습(self-supervised training)을 사용하여 장기 예측(Long-horizon prediction)을 신뢰성 있게 수행할 수 있도록 설계되었습니다. 저자는 이 프레임워크를 통해 로봇 제어의 효율성을 높이고 다양한 작업(tasks)에 걸쳐 적응력을 보장하고자 합니다.

- **Technical Details**: 제안된 방법은 지금까지 존재하던 많은 도메인 고유(특화된) 기반의 inductive biases를 통해 개선된 그리드 구조가 아닌, 더욱 일반화된(world models) 학습 프레임워크를 제공합니다. 특히 저자는 PPO(Proximal Policy Optimization)와 같은 정책 최적화(policy optimization) 프레임워크를 제시합니다. 실험 결과, 제안 된 방법이 발생하는 노이즈에 강한 견고성을 보이며, 여러 작업을 통해 우수한 성능을 발휘했습니다.

- **Performance Highlights**: 이제까지의 여러 실험에서, 제안된 접근 방식은 기존의 최첨단(state-of-the-art) 방법들보다 우수한 autoregressive prediction 정확성을 보였으며, ANYmal D 하드웨어에서의 전이에서도 성과를 거두었습니다. 이러한 정책들은 최소한의 시뮬레이션에서 실제로 이동(transferred) 되더라도 견고한 성능을 유지했습니다. 이 연구는 시뮬레이션과 실제 환경 간의 격차(sim-to-real gap)를 해결하는 데 중요한 기여를 하며, 모델 기반 강화학습(model-based reinforcement learning)의 발전을 이끌고 있습니다.



### landmarker: a Toolkit for Anatomical Landmark Localization in 2D/3D Images (https://arxiv.org/abs/2501.10098)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문에서는 의료 이미지에서 해부학적 랜드마크(localization) 식별의 정확도를 향상시키기 위해 개발된 새로운 Python 패키지 'landmarker'를 소개합니다. 이 패키지는 PyTorch를 기반으로 하며, 다양한 이미지를 위한 전처리 파이프라인과 여러 방법론을 지원하여 연구 및 개발 프로세스를 간소화합니다. 또한, 랜드마크 식별의 정확성을 높이며, 특히 의료 영상 분야에서 중요한 외과 절차 및 진단을 위한 필수 도구입니다.

- **Technical Details**: 'landmarker'는 두 가지 주된 방법론인 heatmap regression과 coordinate regression을 포함합니다. 이 패키지는 다양한 데이터셋과 이미지 포맷(NIfTI, DICOM 등)을 지원하며, PyTorch의 데이터셋 클래스를 상속하여 유연한 데이터 로딩 기능을 제공합니다. 사용자는 static 및 adaptive heatmap regression 접근 방식을 통해 랜드마크를 학습하고, 여러 복잡한 문제를 해결할 수 있는 도구를 갖추게 됩니다.

- **Performance Highlights**: landmarker는 높은 정확도 및 사용자 맞춤형 구현 가능성을 통해 연구자와 실무자들에게 최신 알고리즘을 적용할 수 있게 돕습니다. 기존의 pose estimation 도구들은 의료 영상의 고유한 요구사항을 충족하지 못하는 반면, landmarker는 모듈식 구성을 통해 다양한 낯선 알고리즘을 실험하고 구현할 수 있는 유연성을 제공합니다.



### How Do Programming Students Use Generative AI? (https://arxiv.org/abs/2501.10091)
Comments:
          preprint; accepted to ACM International Conference on the Foundations of Software Engineering (FSE) 2025

- **What's New**: 이 연구는 프로그래밍 학습자들이 ChatGPT와 같은 생성 AI 도구를 사용하는 방식을 탐구하고 그 영향력을 분석하는 것을 목표로 합니다. 37명의 프로그래밍 학생을 대상으로 한 실험에서는, 학생들이 어떻게 이 도구에 의존하며 문제 해결에 접근하는지에 대한 주요 통찰을 제공합니다. 연구 결과, 많은 학생이 생성된 해결책을 단순히 요청하는 경향이 있으며, 이는 비판적 사고 능력이 약화될 우려를 낳고 있습니다.

- **Technical Details**: 연구는 프로그래밍 입문 과정의 과제를 수행하는 학생들에게 ChatGPT에 монитор링된 접근을 제공하는 실험을 포함합니다. 학생들의 상호작용 로그를 분석하여 사용된 전략을 이해하고, 이들에게 발생한 여러 오류 패턴을 추적합니다. 또한 연구는 사용자의 의도에 따라 생성 AI에게 요청된 명령어의 패턴을 평가하고 결과를 기준으로 합격 또는 불합격으로 분류합니다.

- **Performance Highlights**: 결과적으로, 많은 학생이 ChatGPT의 출력을 과도하게 신뢰하며 문제 해결 수업의 진전을 저해하고 있는 상태입니다. 이 연구는 학생들이 코드 이해 및 실수를 스스로 학습하는 대신, 오류가 있는 생성 코드를 ChatGPT로 수정하려는 악순환에 빠지는 경향을 보였습니다. 또한, 일반적으로 생성 AI를 자주 사용하는 학생들은 솔루션 생성을 요청할 가능성이 높았으며, 이는 프로그래머 본인의 자율성과 생산성 감소에 대한 우려를 뒷받침합니다.



### Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework (https://arxiv.org/abs/2501.10075)
Comments:
          This work has been submitted to the IEEE Transactions on Geoscience and Remote Sensing journal for possible publication

- **What's New**: 본 연구는 고해상도 RGB 이미지 쌍과 의미론적 분할 맵을 포함하는 새로운 RSICC 데이터셋인 SECOND-CC를 소개합니다. 이 데이터셋은 6,041 쌍의 바이템포랄 이미지와 30,205개의 문장을 포함하여 이미지 간의 차이를 설명합니다. 또한, MModalCC라는 멀티모달 프레임워크를 제안하여 고급 주의 메커니즘을 통합하여 변화 설명의 정확도를 높입니다.

- **Technical Details**: SECOND-CC 데이터셋은 다양한 실제 시나리오를 반영하여 다양한 이미지 해상도 및 등록 오류 문제를 해결하도록 설계되었습니다. MModalCC는 Cross-Modal Cross Attention (CMCA) 및 Multimodal Gated Cross Attention (MGCA) 메커니즘을 통해 시멘틱 맵과 RGB 이미지로부터 의미 있는 특징을 통합하고 있습니다. 이후 실시된 세부적인 아블레이션 연구 및 주의 시각화는 이러한 접근 방식의 효과를 입증했습니다.

- **Performance Highlights**: MModalCC는 기존 RSICC 최신 방법인 RSICCformer, Chg2Cap, PSNet에 비해 BLEU4 점수에서 4.6%, CIDEr 점수에서 9.6%의 향상을 보여주었습니다. 이러한 성과는 MModalCC의 혁신적인 설계와 데이터셋의 품질 덕분이며, 향후 연구를 위한 오픈 소스 제공도 계획하고 있습니다.



### SpatialCoT: Advancing Spatial Reasoning through Coordinate Alignment and Chain-of-Thought for Embodied Task Planning (https://arxiv.org/abs/2501.10074)
Comments:
          13 pages, 6 figures

- **What's New**: 이 연구는 embodied AI(임바디드 AI)에서의 spatial reasoning(공간 추론) 문제를 해결하기 위한 새로운 접근법인 SpatialCoT를 제안합니다. 기존의 방법들이 복잡한 임바디드 작업에서는 한계가 있었던 반면, SpatialCoT는 Vision-Language Models (VLMs)의 본질적인 사고 및 추론 능력을 활용하여 이러한 문제를 해결하고자 합니다. 본 연구는 두 가지 단계로 구성되어 있으며, vision-language 입력을 공간 좌표와 정렬하는 bi-directional alignment(양방향 정렬)와 언어 모델의 사고 능력을 활용하는 chain-of-thought spatial grounding(사고의 연쇄 공간 기초)를 포함합니다.

- **Technical Details**: SpatialCoT 방법론은 특히 복잡한 환경에서의 세밀한 작업을 처리하기 위해 설계되었습니다. 이 방법은 spatial coordinate bi-directional alignment를 통해 vision-language 입력을 공간 좌표와 정렬하며, chain-of-thought 접근 방식을 통해 고급 공간 추론을 지원합니다. 이 두 가지 단계는 서로 보완적으로 작용하여 VLMs의 공간 추론 능력을 극대화합니다.

- **Performance Highlights**: SpatialCoT는 시뮬레이션과 실제 환경 모두에서 어려운 네비게이션과 조작 작업에 대해 평가되었습니다. 실험 결과, SpatialCoT는 이전의 최첨단 방법들보다 두 작업 모두에서 상당한 성과 향상을 보여주었습니다. 이러한 성과는 VLMs의 사고 및 추론 능력을 효과적으로 활용함으로써 달성되었습니다.



### Accelerating Large Language Models through Partially Linear Feed-Forward Network (https://arxiv.org/abs/2501.10054)
- **What's New**: 대형 언어 모델(LLM)의 압축 문제를 해결하기 위해 TARDIS라는 새로운 접근법을 제안합니다. 이 방법은 선형 근사를 통해 비선형 활성화 함수를 처리하여 FFN(Feed-Forward Network) 블록의 매개변수를 최대 87.5%까지 줄이는 데 기여합니다. TARDIS는 복잡한 비선형 활성화(예: GELU)를 효율적으로 근사할 수 있으며, 경량화하면서도 높은 정확도를 유지합니다.

- **Technical Details**: TARDIS의 핵심 아이디어는 LLM의 활성화 함수 입력 값이 좁은 범위에 집중되어 있다는 점입니다. 이와 같은 분포는 '핫' 입력 범위에서 비선형 활성화 함수를 부분적으로 선형 근사할 수 있게 해줍니다. 특히, 비선형 함수의 근사가 이루어지는 동안, 비정상 입력에 대해서는 온라인 예측기를 통해 원래 계산으로 돌아가는 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: TARDIS는 FFN 매트릭스를 최대 80%까지 실제로 줄일 수 있으며, 최신 pruning 방법들보다 65% 높은 정확도를 기록합니다. 7B 모델을 활용한 실험 결과, TARDIS는 vLLM 서빙 시스템에서 총 추론 속도를 1.6배, HuggingFace 구현에서 1.4배 가속화하는 성능을 보여주었습니다. 이 모든 것들이 단지 10.9%의 정확도 손실을 감수하고 이루어진다는 점에서 매우 인상적입니다.



### Virtual Nodes Improve Long-term Traffic Prediction (https://arxiv.org/abs/2501.10048)
- **What's New**: 본 연구는 전통적인 spatio-temporal graph neural networks(ST-GNNs)의 한계를 극복하기 위해 가상 노드(virtual nodes)를 통합한 새로운 프레임워크를 제안합니다. 이 방법은 병목 현상(over-squashing problem)을 줄이고, 전체 그래프의 정보를 단일 GNN 레이어 내에서 집계할 수 있도록 돕습니다. 실험 결과, 가상 노드를 이용함으로써 장기 예측(long-term prediction) 정확성이 크게 향상되었음을 보여주고 있습니다.

- **Technical Details**: 제안된 모델은 반적응(adaptive) 인접 행렬을 구성하여 가상 노드를 통합합니다. 이 행렬은 거리 기반과 적응형 인접 행렬을 결합하여, 모델이 지리적 정보를 활용하면서 동시에 데이터에서 태스크 특화(feature-specific) 특성을 학습할 수 있게 합니다. 이러한 구조는 GNN의 메시지 전달(message-passing) 과정에서 전역 정보(global information) 통합을 용이하게 하여 장기 의존성(long-range dependencies)을 효과적으로 포착할 수 있도록 합니다.

- **Performance Highlights**: 가상 노드를 포함한 모델은 장기 예측 정확도를 상당히 향상시켰으며, 레이어별 민감도(layer-wise sensitivity)도 증가하여 장거리 노드 간의 연결을 개선하고 있습니다. 또한, 도로 네트워크 열 지도(heat maps)를 통해 주요 교차로와 고교통 지역에 대한 가시성을 개선하여, 모델이 어떻게 정보를 처리하는지에 대한 설명 가능성(explainability)도 제공하고 있습니다.



### Automatic Speech Recognition for Sanskrit with Transfer Learning (https://arxiv.org/abs/2501.10024)
Comments:
          Paper has been accepted at the 4th International Conference on Computer, Communication, Control & Information Technology (C3IT), Hooghly, India, 2024, pp. 1-5

- **What's New**: 산스크리트어(Sanskrit)는 인류의 가장 오래된 언어 중 하나로, 방대한 양의 문헌을 가지고 있습니다. 그러나 디지털 콘텐츠가 매우 제한적이기 때문에 AI 시스템의 훈련에 필요한 자료가 부족합니다. 본 연구에서는 OpenAI의 Whisper 모델을 활용하여 전이 학습(transfer learning) 기법을 통해 산스크리트어를 위한 자동 음성 인식(Automatic Speech Recognition, ASR) 모델을 개발했습니다.

- **Technical Details**: 이 모델은 하이퍼파라미터 최적화를 거쳐 Vaksancayah 데이터셋에서 15.42%의 단어 오류율(Word Error Rate, WER)을 달성했습니다. 전통적인 통계적 접근이 산스크리트어의 복잡한 언어 구조에 한계를 가져오기에, 본 연구에서는 변환기 아키텍처(transformer architecture)를 적용하였습니다. 이를 통해 언어적 복잡성을 더 효과적으로 처리할 수 있었습니다.

- **Performance Highlights**: 개발된 ASR 모델은 텍스트 전사 및 발음 연습 등에 활용될 수 있으며, 온라인 데모를 통해 공공에게 공개하여 성능 평가에 기여하고 있습니다. 이 모델은 산스크리트어 학습의 접근성을 높이고, 현대 기술 지원을 통해 산스크리트어 교육을 개선하는 데 중요한 발판이 될 것입니다.



### Mitigating Hallucinations on Object Attributes using Multiview Images and Negative Instructions (https://arxiv.org/abs/2501.10011)
Comments:
          2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 본 논문은 현재 인기 있는 Large Vision-Language Models (LVLMs)가 겪고 있는 Hallucinations on Object Attributes (HoOA) 문제를 해결하기 위한 새로운 방법을 제안합니다. 3D 생성 기술의 발전을 활용하여, 단일 이미지에서 생성된 3D 표현으로부터 샘플링된 다중 관점(multiview) 이미지를 LVLMs의 시각적 프롬프트로 사용함으로써, 보다 다양한 시각 정보를 제공합니다. 이 접근 방식은 LVLM의 성능 향상에 기여합니다.

- **Technical Details**: 제안된 방법은 Multiview Image Augmented VLM (MIAVLM)으로, 여기에는 Multiview Attributes Perceiver (MAP) 서브모듈이 포함되어 있어 입력 이미지 순서의 영향을 제거하면서 다중 관점 이미지에서 시각 정보를 Large Language Models (LLMs)와 정렬합니다. 이를 통해 LVLM의 성능을 더욱 향상시키며, 특히 다중 관점 이미지가 LVLMs에서 수행하는 세부 속성 인식의 정확성을 높이는 데 도움을 줍니다. 또한, 부정적 지침(negative instructions)을 도입하여 LVLMs가 "예" 응답에 편향되는 문제를 완화하고자 하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 LVLMs의 비정확한 속성 판단 문제를 효과적으로 개선하는 것을 보여주었습니다. 다양한 실험을 통해 MIAVLM은 기존 모델 대비 성능에서 우수한 결과를 나타냈으며, 다중 관점 이미지가 LVLMs의 인식 능력을 강화하는 데 중요한 역할을 함을 입증하였습니다. 이러한 결과는 LVLMs의 활용 가능성을 더욱 확대하는 데 기여할 것으로 기대됩니다.



### Adaptive Spatiotemporal Augmentation for Improving Dynamic Graph Learning (https://arxiv.org/abs/2501.10010)
Comments:
          2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 본 논문에서는 동적 그래프 신경망(Dynamic GNN)의 성능을 향상시키기 위한 STAA(SpatioTemporal Activity-Aware Random Walk Diffusion)라는 새로운 방법을 제안합니다. 기존의 방법들은 최신 엣지가 더 중요하다는 가정을 기반으로 하여 노이즈를 포착하는 경향이 있습니다. STAA는 공간적 및 시간적 차원에서 노이즈 엣지를 식별하고 행동의 변동성을 활용하여 노이즈 영향을 줄입니다. 이를 통해 보다 신뢰할 수 있는 스파시타임 정보(spatiotemporal information)를 활용할 수 있게 됩니다.

- **Technical Details**: STAA는 그래프 웨이브렛 계수(Graph Wavelet Coefficients)를 사용하여 노드의 위치 및 시계열 변화를 평가합니다. 공간 도메인에서는 그래프 웨이브렛 계수를 통해 중요 토폴로지 위치를 분석하고, 시간 도메인에서는 엣지의 진화를 평가합니다. 그런 다음 랜덤 워크(Random Walk)를 통해 최근 엣지에 대한 노드의 선호를 줄이고 초기 엣지에 대한 시간적 걷기 확률을 증가시켜 궁극적으로 노이즈 엣지의 가중치를 줄입니다. 이 과정을 통해 STAA는 동적 GNN 학습을 위한 보강 인접 행렬을 생성합니다.

- **Performance Highlights**: 여러 데이터셋에서의 실험 결과 STAA는 노드 분류(node classification) 및 링크 예측(link prediction) 작업에서 다른 동적 그래프 증강 방법들보다 우수한 성능을 보였습니다. 특히, STAA는 GNN이 동적 그래프의 스파시타임 정보를 효과적으로 활용할 수 있도록 지원하여 전반적인 성능 향상을 가져왔습니다. 이러한 결과는 STAA의 효용성을 입증하며, 다양하고 도전적인 동적 그래프 상황에서도 유리한 효과를 나타내는 데 큰 기여를 합니다.



### Deep Learning for Early Alzheimer Disease Detection with MRI Scans (https://arxiv.org/abs/2501.09999)
- **What's New**: 이번 연구는 알츠하이머병(Alzheimer's Disease, AD)의 진단 정확성을 향상시키기 위해 심층 학습 모델의 비교를 다루고 있습니다. 특히, CNN(Convolutional Neural Network), Bayesian CNN, U-net 모델을 검토하며 OASIS brain MRI 데이터 세트를 활용하고 있습니다. 모델 평가의 신뢰성을 보장하기 위해 데이터 불균형 문제를 해결하였고, 민감도, 특이도 및 계산 효율성을 고려하여 각 모델의 장단점을 분석했습니다.

- **Technical Details**: 연구는 세 가지 심층 학습 모델(CNN, Bayesian CNN, U-Net)을 사용하여 MRI 스캔 결과를 분석합니다. 데이터 세트의 균형을 유지하기 위해 SMOTE-Tomek 기법을 적용하였고, 모델의 정확도, 정밀도, 재현율 및 F1 점수를 평가하였습니다. Bayesian CNN은 95% 이상의 정확도를 달성했고, Grad-CAM(Gradient-weighted Class Activation Mapping)을 통해 모델 예측에 기여한 뇌의 주요 영역을 시각화하여 해석 가능성을 높였습니다.

- **Performance Highlights**: Bayesian CNN 모델은 95% 이상의 높은 정확도를 기록하여 가장 우수한 성능을 보였습니다. CNN과 U-Net 모델도 후속 순위를 차지하며 성능을 입증했습니다. Grad-CAM을 활용하여 모델이 집중한 뇌 영역을 시각화함으로써, AI를 통한 조기 진단 가능성을 제시하였습니다. 향후 연구는 종양학자와 협력하여 뇌 MRI의 마스킹 방법을 개발하고, AD 조기 예측을 위한 핵심 변수를 식별하는 방향으로 진행될 예정입니다.



### Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models (https://arxiv.org/abs/2501.09997)
- **What's New**: 이 연구에서는 기존의 LLM에서 발생하는 환각(hallucination) 문제를 해결하기 위해 새로운 Attention-Guided SElf-Reflection (AGSER) 방법을 제안합니다. AGSER 방법은 입력 쿼리를 주의 깊은(attentive) 쿼리와 비주몰한(non-attentive) 쿼리로 분리하여 처리합니다. 이 방식을 통해 환각 탐지의 효율성을 높이고, 계산 복잡성을 줄이는 효과도 있습니다.

- **Technical Details**: AGSER는 LLM의 주의(attention) 기여도를 활용하여 입력 쿼리를 처리합니다. 각 쿼리는 LLM을 통해 별도로 처리되며, 생성된 응답과 원래 응답 간의 일관성 점수를 계산합니다. 두 개의 일관성 점수 차이를 계산하여 환각의 정도를 추정합니다. AGSER는 3번의 LLM 실행만으로 결과를 도출할 수 있으며, 이는 기존의 접근 방식과 비교했을 때 계산 비용을 크게 줄이는 것입니다.

- **Performance Highlights**: 실험 결과, AGSER는 4개의 잘 알려진 LLM에 대해 다양한 환각 벤치마크에서 기존 방법보다 뛰어난 성능을 보여줍니다. 연구자들은 AGSER 접근법이 환각 탐지에서 최첨단 성과를 달성했다고 보고했습니다. 이를 통해 LLM의 신뢰성을 높이는 데 기여할 것으로 기대하고 있습니다.



### Fast energy-aware OLSR routing in VANETs by means of a parallel evolutionary algorithm (https://arxiv.org/abs/2501.09996)
- **What's New**: 이 연구는 차량 네트워크에서 OLSR 라우팅 프로토콜의 전력 소비를 줄이는 문제를 다루고 있습니다. 최근 에너지 효율성을 고려한 통신 프로토콜이 모바일 네트워크 구축 시 중요한 연구 주제가 되고 있습니다. 본 논문에서는 병렬 진화 알고리즘(parallel evolutionary algorithm)을 활용하여 에너지 효율적인 OLSR 구성(configurations)을 빠르게 자동으로 탐색하는 방법론을 제안합니다.

- **Technical Details**: 제안된 방법론은 기존 구성보다 전력 소비 측면에서 큰 개선을 보여줍니다. 실험 분석(experimental analysis)을 통해 QoS(Quality of Service)의 손실 없이도 성능 향상을 달성할 수 있음을 입증하였습니다. 이 연구는 차량 통신 환경에 적합한 새로운 에너지 절약 방법을 제공합니다.

- **Performance Highlights**: 전력 소모에서 표준 구성 대비 유의미한 개선이 이루어졌으며, QoS의 저하 없이 성능을 개선하였습니다. 이러한 결과는 에너지 효율성 향상을 위해 자동차 네트워크의 라우팅 프로토콜을 최적화할 수 있는 가능성을 시사합니다.



### Multi-Modal Attention Networks for Enhanced Segmentation and Depth Estimation of Subsurface Defects in Pulse Thermography (https://arxiv.org/abs/2501.09994)
Comments:
          Pulse thermography, infrared thermography, defect segmentation, multi-modal networks, attention mechanism

- **What's New**: 이 논문에서는 Pulse Thermography (PT) 검사를 위한 AI 기반의 다중 모달 주의 기반 융합 네트워크인 PT-Fusion을 소개합니다. PT-Fusion은 PCA(주성분 분석)와 TSR(열영상 신호 재구성) 모달리티를 통합하여 결함 분할과 깊이 추정을 동시에 수행할 수 있도록 설계되었습니다. 이 연구는 기존의 기술적 제약을 극복하고, 특히 결함 탐지의 성능을 향상시키는 접근 방식을 제안하여 주목받고 있습니다.

- **Technical Details**: PT-Fusion 네트워크는 결함 분할과 깊이 추정에 필요한 두 가지 주요 모듈인 Encoder Attention Fusion Gate (EAFG)와 Attention Enhanced Decoding Block (AEDB)을 사용하여 PCA와 TSR 데이터를 융합하는 방식으로 작동합니다. 이 모듈들은 서로 다른 정보인 PCA와 TSR의 의미론적 특성을 동적으로 학습하여 성능을 최적화합니다. 또한, 새로운 데이터 증강 기법이 도입되어 열영상 시퀀스에서 무작위 데이터 샘플링을 통해 PT 데이터셋의 부족 문제를 완화하고자 하였습니다.

- **Performance Highlights**: 시험 결과, PT-Fusion은 U-Net, Attention U-Net 및 3D-CNN과 같은 기존의 최신 PT 검사 모델에 비해 결함 분할 및 깊이 추정 정확도에서 10% 이상의 성능 향상을 보였습니다. 이 성과는 PT-Fusion의 새로운 접근 방식이 실제 산업 환경에서의 비파괴 검사(NDT) 성능을 극대화하는 데 기여할 수 있음을 보여줍니다.



### RichSpace: Enriching Text-to-Video Prompt Space via Text Embedding Interpolation (https://arxiv.org/abs/2501.09982)
- **What's New**: 이번 연구에서는 텍스트를 비디오로 생성하는 데 있어 중요성이 상대적으로 간과되었던 텍스트 임베딩(text embedding)의 최적화 방법을 제안합니다. 특히, 임베딩 공간에서의 보간(interpolation)을 통해 최적의 텍스트 임베딩을 선택함으로써 비디오 생성 모델의 성능을 개선할 수 있다고 주장합니다. 이러한 접근 방식은 기존의 다수의 텍스트 인코더를 사용하는 방법과 대조를 이루며, 계산 비용을 줄이는 동시에 효과적인 결과를 도출할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 수직각 발 임베딩(perpendicular foot embeddings)과 코사인 유사성(cosine similarity) 알고리즘을 활용하여 텍스트 임베딩 공간 내에서 최적의 텍스트 임베딩을 찾는 새로운 방법을 소개합니다. 이를 통해 비디오 생성 모델이 원하는 비디오를 생성할 수 있는 능력을 극대화하는 것을 목표로 하고 있습니다. 또한, CogvideoX 모델의 각 구성 요소에 대한 공식적인 정의와 문제 정의를 제공하여, 향후 연구자들이 더욱 쉽게 이 알고리즘을 적용할 수 있도록 돕고 있습니다.

- **Performance Highlights**: 실험 결과, 최적의 텍스트 임베딩을 선택하는 것이 비디오 생성 모델이 원하는 비디오를 생성하는 데 큰 영향을 미친다는 것을 확인했습니다. 본 연구의 제안에 따른 알고리즘은 비디오 품질 향상은 물론 복잡한 텍스트 프롬프트를 보다 효과적으로 처리하는 능력을 보여주었습니다. 이러한 성과는 텍스트에서 비디오로의 전환이 갖는 잠재력을 뒷받침하며, 향후 다양한 응용 프로그램에서 활용될 수 있는 기반을 제공합니다.



### Aneumo: A Large-Scale Comprehensive Synthetic Dataset of Aneurysm Hemodynamics (https://arxiv.org/abs/2501.09980)
- **What's New**: 이번 연구에서는 뇌동맥류(Intracranial Aneurysm, IA)에 대한 포괄적인 혈역학적 데이터셋을 구축했습니다. 이 데이터셋은 466개의 실제 뇌동맥류 모델과 10,000개의 합성 모델을 포함하여 뇌동맥류의 형태적 특징을 분석하는 데 유용합니다. 특히, 데이터셋에는 의학 이미지와 유사한 세분화 마스크 파일도 제공되어 유의미한 분석을 지원합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 8개의 일정한 유량(steady-state flow rates)에서 측정된 혈역학적 데이터로 구성되어 있으며, 유속(flow velocity), 압력(pressure), 벽 전단 응력(wall shear stress)과 같은 중요한 매개변수가 포함되어 있습니다. 이 데이터는 뇌동맥류의 발생 원인(pathogenesis) 및 임상 예측(clinical prediction) 연구에 있어 귀중한 자원입니다. 또한, 466개의 뇌동맥류가 없는 모델과 9,534개의 변형된 뇌동맥류 모델이 포함되어 있어 다양한 형태적 분석이 가능합니다.

- **Performance Highlights**: 이 데이터셋은 뇌동맥류의 병리적 특성과 혈역학적 메커니즘을 이해하는 데 중요한 기여를 할 것으로 기대됩니다. 이를 통해 뇌동맥류 관련 연구의 심화와 클리닉에서의 예측 가능성을 높일 수 있습니다. 연구자는 이 데이터셋을 활용하여 뇌동맥류의 보다 심층적인 연구를 진행할 수 있습니다.



### GVMGen: A General Video-to-Music Generation Model with Hierarchical Attentions (https://arxiv.org/abs/2501.09972)
Comments:
          Accepted by the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 본 논문에서는 다양한 스타일의 비디오에 높은 관련성을 가진 음악을 생성하기 위한 General Video-to-Music Generation model (GVMGen)을 제안합니다. GVMGen은 공간적 및 시간적 차원에서 비디오 특징을 음악과 정합시키기 위해 계층적 주의 기법(hierarchical attentions)을 활용하여, 중복성을 최소화하면서 관련 있는 특징을 보존합니다. 특히, 제안된 모델은 각기 다른 비디오 입력에 대해 다중 스타일의 음악을 생성할 수 있는 범용성을 지닌 것이 특징입니다.

- **Technical Details**: GVMGen은 개별 변수를 명시적으로 정의하기보다는 시각적 특징을 추출하는 공간적 자기 주의 기법(spatial self-attention)과 음악 특징으로 변환하는 공간적 및 시간적 교차 주의 기법(cross-attention)을 채택하여 이루어집니다. 이러한 임 implicit feature extraction 및 정합은 다양한 비디오와 음악 스타일에 적합하게 작동하며, 제로샷(zero-shot) 상황에서도 성능을 유지합니다. 또한, 글로벌 및 로컬 음악-비디오 정합성을 평가하기 위한 새로운 객관적 메트릭을 제안합니다.

- **Performance Highlights**: 실험 결과 GVMGen은 음악-비디오 정합성 및 음악의 풍부함 면에서 우수한 성능을 보여 주목할 만합니다. 제안된 모델은 MIDI 형식의 음악이 아닌 중국 및 서양 스타일의 다중 트랙 파형 음악을 생성할 수 있어 음악 생성의 풍부함 및 완전성을 지속적으로 개선하고 있습니다. 또한 GVMGen은 제로샷 상황에서도 고품질 음악 생성을 가능하게 하여, 다양한 응용 프로그램에서 우수한 성과를 나타냅니다.



### Explainable artificial intelligence (XAI): from inherent explainability to large language models (https://arxiv.org/abs/2501.09967)
- **What's New**: 이 논문은 기계 학습 모델의 결정 과정을 설명할 수 있는 방법인 설명 가능한 인공지능(Explainable AI, XAI) 기술의 발전을 포괄적으로 다룹니다. 특히, 대규모 언어 모델(LLM)과 비전-언어 모델(VLM)을 활용하여 다른 기계 학습 모델의 설명 가능성을 자동화하거나 개선하는 방법에 대해서도 논의합니다. 이러한 접근은 모델의 결정과 행동에 대한 의미 있는 고수준 설명을 제공하는 데 도움을 줍니다.

- **Technical Details**: 기계 학습 모델의 해석 가능성은 중요하며, 이는 인풋과 아웃풋 간의 관계를 강조하거나 내부 아키텍처를 통해 모델의 예측 과정을 이해할 수 있는 것이 필요합니다. 본 논문에서는 설명 가능성의 두 가지 주요 유형인 post-hoc(모델 훈련 후 적용) 해석 가능성과 ante-hoc(훈련 또는 설계 시 해석 가능성 부여) 해석 가능성을 설명합니다. 각 방식은 장단점이 있으며, 특히 post-hoc 방식은 종종 모델의 진정한 예측 논리를 따르지 않을 수 있어 주의가 필요합니다.

- **Performance Highlights**: 설명 가능한 인공지능(XAI) 기술을 통해 기계 학습 시스템의 투명성을 높이고, 의사 결정자의 신뢰도를 증가시켜 중요한 애플리케이션에서의 채택을 촉진합니다. 또한, XAI 방법은 모델의 편향을 발견하고 모든 이해 관계자에게 공정한 결정을 보장하는 데 기여합니다. 최종적으로, XAI는 기계 학습 모델의 안전성을 개선하고, 부적절한 예측 결과를 방지하는 데 필수적인 역할을 합니다.



### AIRCHITECT v2: Learning the Hardware Accelerator Design Space through Unified Representations (https://arxiv.org/abs/2501.09954)
Comments:
          Accepted to DATE 2025

- **What's New**: 본 논문은 AIrchitect v2라는 새로운 DSE 기술을 제안하여, 이전 접근 방식의 한계를 극복하고 더 정확하고 일반화 가능한 학습 기반 DSE를 가능하게 합니다. 이 모델은 고유의 encoder-decoder transformer 구조를 사용하여 복잡한 설계 공간을 균일한 중간 표현으로 인코딩하고, 분류(classification)와 회귀(regression)의 이점을 혼합한 새로운 통합 표현을 활용합니다. 이는 대규모 DSE 공간을 탐색할 수 있도록 도와주며, 정확도를 유지하게 됩니다.

- **Technical Details**: AIrchitect v2는 contrastive learning을 활용하여 입력 피처 표현을 균일하고 부드러운 임베딩 공간으로 학습 및 인코딩합니다. 논문에서 제안한 Unified Ordinal Vectors라고 불리는 통합 표현은 DSE를 위해 분류와 회귀의 장점을 결합하여 성능을 높입니다. 연구 결과, AIrchitect v2는 기존 기술 대비 평균 15%의 개선을 보여주며, 10^5개의 실제 DNN 워크로드에 대해 최적 설계 포인트를 발견하는 데 있어 우수함을 입증합니다.

- **Performance Highlights**: AIrchitect v2는 예측된 하드웨어 아키텍처에서 추론 성능을 약 1.7배 개선하는 성과를 보여주었습니다. 이는 서로 다른 모델 아키텍처에 대한 일반화 가능성을 강조하며, 모르는 모델 워크로드에서 성능을 평가하는 데도 유리한 결과를 냈습니다. 또한, 이 연구는 MAESTRO 기반의 DSE 훈련 데이터셋을 공개하여, 학습 기반 DSE 연구 분야의 발전에 기여하고자 합니다.



### MultiPruner: Balanced Structure Removal in Foundation Models (https://arxiv.org/abs/2501.09949)
- **What's New**: 최근 대형 사전 학습 모델(Large Pre-trained Models, LPM) 프루닝(pruning) 기술이 크게 발전하였습니다. 새로운 알고리즘인 MultiPruner는 기존 BlockPruner와 비교하여 다차원(multi-dimensional) 반복적(pruning) 방법을 통해 모델을 효과적으로 압축하는 방안을 제시합니다. 이 방법은 잔여 블록(residual blocks), MLP(Multilayer Perceptrons) 채널 및 주의 헤드(attention heads)의 3가지 차원에서 압축을 수행하여, 전반적인 모델 성능을 향상시키는 데 기여하고 있습니다.

- **Technical Details**: MultiPruner는 Transformer의 구조적 균형을 유지하면서 세밀한 방식으로 모델을 압축하기 위해 설계되었습니다. 이 접근법은 각 차원에서 평균적인 성능을 유지하는 것을 목표로 하며, 블록 프루닝(block pruning) 후에도 모델의 구조를 고려하여 추가적인 프루닝을 실시합니다. MultiPruner는 세 가지 주요 요소에 대해 압축을 적용하는데, 이는 잔여 Transformer 블록, MLP 채널, Attention heads입니다.

- **Performance Highlights**: MultiPruner는 대규모 사전 학습 모델에 대해 다양한 실험을 통해 그 효과를 입증하였습니다. 본 알고리즘은 타깃 비율(target ratio)을 유지하면서 높은 성능의 프루닝 구성을 찾아내는 데 성공하며, 제로샷(zero-shot) 평가에서 우수한 정확도를 보였습니다. 실험 결과는 MultiPruner가 구조적 균형을 유지하면서도 효율적인 모델을 생성함을 보여주었습니다.



### AI Explainability for Power Electronics: From a Lipschitz Continuity Perspectiv (https://arxiv.org/abs/2501.09948)
- **What's New**: 이 논문에서는 전력 전자(PE) 분야에서 AI 수학적 설명 가능성의 중요성을 강조하고 있습니다. 기존에는 이론적 엄밀성이 부족하여 미션 크리티컬(mission-critical) 애플리케이션에서의 채택이 어려웠습니다. 저자들은 일반적인 프레임워크를 제안하여 Lipschitz 연속성(Lipschitz continuity) 관점에서 추론 안정성 및 훈련 수렴성을 평가합니다.

- **Technical Details**: 추론 안정성은 입력 변동성(perturbations) 아래에서도 일관된 출력을 보장하며, 이는 강력한 실시간 제어 및 고장 진단에 필수적입니다. 훈련 수렴성은 안정적인 학습 동역학을 보장하여 PE 상황에서의 정확한 모델링을 촉진합니다. Lipschitz 인식 학습 속도 선택 전략이 도입되어 수렴을 가속화하면서 과도한 발진(overshoots)과 진동(oscillations)을 완화합니다.

- **Performance Highlights**: 저자들은 제안된 Lipschitz 지향 프레임워크의 타당성을 입증하기 위해 최첨단 물리 기반(neural network) 신경망의 수학적 설명 가능성을 검증하며, 이중 능동 브리지 변환기(dual-active-bridge converters)에 대한 경험적 사례 연구를 통해 입증하였습니다. 이 논문은 PE 커뮤니티가 수학적 설명 가능성을 받아들여 신뢰할 수 있는 AI 솔루션의 변혁적인 시대를 맞이하길 촉구하는 메시지를 전달합니다.



### Client-Centric Federated Adaptive Optimization (https://arxiv.org/abs/2501.09946)
- **What's New**: 이 논문에서는 Client-Centric Federated Adaptive Optimization (CC-FAO)라는 새로운 연합 학습 접근 방식을 제안합니다. 이 방법은 클라이언트의 데이터 프라이버시를 유지하면서도 비동기화된 서버 집계 및 이질적인 로컬 컴퓨팅을 지원하여 실제적인 연합 학습 시스템의 특징을 반영합니다. 기존의 연합 학습 연구가 대부분 비현실적인 가정에 기반한 데 반해 이 프레임워크는 더 많은 자율성을 클라이언트에게 부여합니다.

- **Technical Details**: 본 연구에서 제안하는 CC-FAO는 여러 고유 기능을 포함하고 있습니다: (1) 클라이언트는 필요할 때만 참여하고, 각 클라이언트는 장치에 따라 다르게 설정된 로컬 에폭 수를 자율적으로 결정할 수 있습니다. (2) 비동기 집계 방식으로 인해 각 클라이언트는 구식의 글로벌 모델 뷰를 이용하면서 작업할 수 있습니다. (3) 서버에서의 글로벌 최적화가 클라이언트의 로컬 업데이트와 동시에 이루어집니다.

- **Performance Highlights**: 예비 실험 결과, CC-FedAdam, CC-FedAdagrad, CC-FedAMS와 같은 제안된 방법들이 FedAvg 및 그 변형보다 월등한 성능을 보여주었습니다. 이 개선은 통계적 및 시스템 이질성에 따른 여러 수준에서도 일관되게 나타났습니다. 또한, 하이퍼파라미터 조정에 대한 분석을 통해 본 방법이 더 나은 결과를 얻기 위해 쉽게 조정 가능하다는 것을 확인하였습니다.



### HEART: Achieving Timely Multi-Model Training for Vehicle-Edge-Cloud-Integrated Hierarchical Federated Learning (https://arxiv.org/abs/2501.09934)
Comments:
          14 pages, 6 figures,

- **What's New**: 최근 AI와 IoV(Internet of Vehicles)의 발전은 효율적인 기계 학습 솔루션에 대한 필요성을 강조하고 있으며, 이로 인해 Hierarchical Federated Learning(VEC-HFL)의 발달이 촉진되고 있다. 본 연구는 다중 모델 학습 환경에서의 여러 도전에 대응하기 위한 첫 발걸음으로, 다양한 ML 작업 간의 균형 잡힌 훈련을 보장하고 전 세계적인 훈련 지연 시간을 최소화하는 프레임워크를 제안한다. 특히, hybrid synchronous-asynchronous aggregation rule을 도입하여 효율적으로 모델 학습을 진행한다.

- **Technical Details**: 제안된 Hybrid Evolutionary And gReedy allocaTion (HEART) 방법은 두 단계로 구성되어 있다. 첫 번째 단계에서는 개선된 Particle Swarm Optimization(PSO)와 Genetic Algorithms(GA)를 결합한 하이브리드 휴리스틱 접근을 통해 균형 잡힌 작업 스케줄링을 달성하며, 두 번째 단계에서는 지정된 작업의 훈련 우선 순위를 결정하기 위해 저복잡도 그리디 알고리즘을 활용한다. 이러한 구조는 VEC-HFL의 다중 모델 학습 과정을 최적화하는 데 기여한다.

- **Performance Highlights**: 실제 데이터셋을 사용한 실험 결과, HEART는 기존 방법들에 비해 시간이 효율적이며 커뮤니케이션 비용에서도 우수한 성능을 보여준다. 이 연구는 다이나믹 IoV 환경에서 HEART의 효율적 수행 가능성을 강조하며, 실제 적용 및 확장성에 대한 잠재력을 잘 드러낸다.



### Steering Large Language Models with Feature Guided Activation Additions (https://arxiv.org/abs/2501.09929)
Comments:
          7 maintext pages, 14 appendix pages

- **What's New**: 본 논문에서는 Feature Guided Activation Additions(FGAA)라는 새로운 활성화 조정(activation steering) 방법을 소개합니다. FGAA는 Sparse Autoencoder(SAE)와 Contrastive Activation Addition(CAA)의 통찰력을 활용하여 모델의 행동을 보다 효과적으로 제어할 수 있도록 설계되었습니다. 기존 방법들의 부족한 정밀성과 해석 가능성을 개선하며, 복잡한 조정 작업에서도 일관된 출력 품질을 유지합니다.

- **Technical Details**: FGAA는 SAE의 잠재 공간에서 작동하며, 원하는 SAE 특성을 선택하기 위해 최적화 기법을 사용합니다. 이 방법은 라벨이 있는 데이터로부터 긍정적(positive) 및 부정적(negative) 예시를 통해 대조적 차이를 계산하여 탐지된 특징을 기반으로 유용한 조정 벡터를 생성합니다. FGAA는 밀도 필터링(density filtering), BOS(Beginning Of Sequence) 피처 제거, 그리고 상위 특성 선택(top-k selection)이라는 세 가지 중요한 필터링 단계를 통해 특성 벡터를 변형하여 조정 벡터를 생성합니다.

- **Performance Highlights**: GFWWA는 Gemma-2-2B 및 Gemma-2-9B 모델에서 다양한 조정 작업을 수행하는 동안 기존 CAA, SAE 디코더 조정 및 SAE-TS보다 우수한 성능을 보여줍니다. 연구 결과, 조정 규모와 모델의 일반적인 능력 간의 중요한 상쇄 관계가 존재함을 발견하였으며, 이는 모든 테스트된 조정 방법에 걸쳐 일관됩니다. FGAA는 조정 효과와 출력 일관성을 개선하며, 복잡한 조정 작업에서의 성능이 특히 두드러집니다.



### Dialogue Benchmark Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented LLMs (https://arxiv.org/abs/2501.09928)
Comments:
          The paper is publsihed in SIGMOD 2025

- **What's New**: 이 논문에서는 Chatty-Gen이라는 새로운 다단계 Retrieval-Augmented Generation (RAG) 플랫폼을 소개합니다. Chatty-Gen은 지식 그래프(KG)를 사용하여 특정 도메인에 맞춘 고품질 대화 벤치마크(dialogue benchmark)를 자동으로 생성하는 혁신적인 시스템입니다. 이 플랫폼은 생성 과정의 복잡한 단계를 관리 가능하도록 쪼개고, 단계 간의 자동 검증을 위해 assertion rules를 사용합니다. 이를 통해 대화의 일관성을 유지하고, 시간 소모가 큰 재시작을 방지합니다.

- **Technical Details**: Chatty-Gen은 KG에서 샘플링된 대표 엔티티를 기반으로 대화의 맥락(context)을 확보합니다. 생성 과정은 처음에 특정 엔티티를 샘플링한 후, 관련 있는 서브그래프(subgraph)를 추출하여 대화 맥락을 유지합니다. 텍스트 환경을 인간이 읽을 수 있는 형태로 변환하고, 대화의 흐름을 유지하는 주요 질문을 통해답변 생성을 위해 SPARQL 쿼리를 자동으로 생성합니다. 이 시스템은 KG에 종속되지 않는 설계를 갖추고 있어 다양한 도메인 및 KG에서 사용될 수 있습니다.

- **Performance Highlights**: Chatty-Gen은 DBpedia, YAGO, DBLP와 같은 여러 실제 KG에서 성능 평가를 진행했으며, 다양한 상업적 및 오픈소스 LLM과의 호환성을 입증했습니다. 특히, Chatty-Gen은 Maestro와 비교하여 대화 벤치마크 생성에서 99%의 시간 효율성을 보여주었으며, 30시간이 소요되던 작업을 불과 10분으로 단축시켰습니다. 이러한 결과는 Chatty-Gen이 고품질 KG 기반 대화 벤치마크 생성을 위한 비용 효율적이고 다재다능한 솔루션임을 입증합니다.



### IE-Bench: Advancing the Measurement of Text-Driven Image Editing for Human Perception Alignmen (https://arxiv.org/abs/2501.09927)
- **What's New**: 이 논문에서는 텍스트 기반 이미지 편집(Text-driven image editing)의 평가 방법을 개선하기 위해 IE-Bench라는 새로운 벤치마크 세트를 제안합니다. IE-Bench는 다양한 소스 이미지, 편집 프롬프트 및 결과를 포함하는 데이터베이스로, 이를 통해 인간의 주관적 평가를 반영한 Mean Opinion Scores (MOS)를 제공합니다. 또한, 텍스트 기반 이미지 편집의 품질을 평가하기 위해 IE-QA라는 다중 모달 소스 인식 품질 평가 방법을 소개합니다.

- **Technical Details**: IE-Bench는 이미지 편집에서 텍스트와 소스 이미지 간의 관계를 동적으로 모델링하는 다중 모달 방법을 필요로 한다는 점에 중점을 둡니다. IE-DB는 다양한 소스-프롬프트-타겟 케이스를 수집하고 MOS를 제공하며, 이는 이미지 편집을 위한 최초의 이미지 품질 평가 데이터셋으로 간주됩니다. IE-QA는 편집된 이미지의 품질, 이미지와 텍스트 간의 관계, 소스와 타겟 이미지 간의 연결성을 포함하여 여러 차원을 고려한 포괄적인 평가를 제공합니다.

- **Performance Highlights**: IE-QA는 기존의 IQA 메트릭스와 비교하여 주관적 정렬에서 현저한 이점을 보여주며, 텍스트 기반 이미지 편집 작업에서 효과적인 성능을 입증합니다. 이러한 연구 결과는 텍스트 기반 이미지 편집 평가의 새로운 지평을 열며, 관련 데이터와 코드가 공개될 예정입니다. IE-Bench의 결과는 향후 텍스트 기반 이미지 편집의 질을 향상시키는 데 기여할 것으로 기대됩니다.



### Study on a Fast Solver for Combined Field Integral Equations of 3D Conducting Bodies Based on Graph Neural Networks (https://arxiv.org/abs/2501.09923)
Comments:
          10 pages,11 figures

- **What's New**: 이 논문에서는 그래프 신경망(GNNs)을 기반으로 한 빠른 해법(GraphSolver)을 제안하여 3D 전도체의 결합장 적분방정식(CFIEs)을 해결합니다. Rao-Wilton-Glisson(RWG) 기초 함수가 3D 물체의 지오메트리를 정확하게 표현하기 위해 사용되며, 각 RWG 함수는 그래프에서 노드로 간주되어 전류의 흐름을 가능하게 합니다.

- **Technical Details**: GraphSolver는 변환된 그래프를 사용하여 각 노드에서의 표면 전류 밀도의 x, y, z 성분의 실수 및 허수 부분을 직접 예측합니다. 네트워크 아키텍처는 업샘플링 완전 연결 네트워크(FCN), 그래프 컨볼루션 네트워크(GCN), 여섯 개의 다운샘플링 FCN으로 구성됩니다. 이 접근법은 다양한 기하학적 복잡성을 가진 3D 전도체에 대해 CFIEs를 해결하는 데 성공적임을 보여줍니다.

- **Performance Highlights**: Numerical results demonstrate the efficacy of GraphSolver in handling geometrically complex targets like missiles and airplanes, outperforming traditional methods in computational efficiency. Additionally, the implementation 코드 및 훈련된 모델 파라미터 파일이 공개되어 있어, 연구자들이 쉽게 접근하여 사용할 수 있습니다.



### SLIM: Sim-to-Real Legged Instructive Manipulation via Long-Horizon Visuomotor Learning (https://arxiv.org/abs/2501.09905)
- **What's New**: 이번 연구에서는 강화 학습(Reinforcement Learning, RL)으로 훈련된 저비용 쿼드펙드 조작 시스템을 소개합니다. 이 시스템은 고급 정책을 위한 계층적 디자인과 함께 다양한 과업을 해결하기 위한 점진적 정책 확장 접근 방식을 포함하고 있습니다. 실제 환경에서 길고 복잡한 조작 작업을 효과적으로 수행하며, 단 하나의 RGB 카메라로도 높은 성공률을 달성합니다.

- **Technical Details**: SLIM(Sim-to-Real Legged Instructive Manipulation)은 19 자유도(Degree of Freedom)로 구성된 쿼드펙드 로봇으로, 고급 비주얼-모터 조작 정책과 저급 쿼드펙드 제어기를 결합한 계층적 설계 구조를 사용합니다. 이 시스템은 개인화된 정보(privileged information) 접근을 통해 학생 정책이 학생 정책을 증류(distill)하도록 훈련하여 다양한 작업을 수행합니다. 또한, SLIM은 언어 지침에 따라 로봇에게 다양한 작업을 지시할 수 있도록 설계되었습니다.

- **Performance Highlights**: SLIM은 시뮬레이션과 실제 환경 모두에서 높은 성공률과 빠른 작업 완료 시간을 달성했으며, 실내 외부 환경에 걸쳐 다양한 작업을 수행할 수 있는 강력한 성과를 보입니다. 특히, 비싸고 고성능 하드웨어가 아니라도 잘 작동하며, 시뮬레이션에서 연습한 내용을 실제로 원활하게 전달할 수 있는 능력을 보여줍니다. 본 연구 결과는 저비용 하드웨어를 사용하여도 다양한 환경에서 높은 성능을 입증했습니다.



### ASTRA: A Scene-aware TRAnsformer-based model for trajectory prediction (https://arxiv.org/abs/2501.09878)
- **What's New**: 이번 논문에서는 ASTRA(장면 인식 트랜스포머 기반 경로 예측 모델)를 제안합니다. 이 모델은 장면 맥락, 공간 역학, 사회적 상호작용 및 시간 진행을 통합하여 보행자 경로 예측의 정확성을 높이도록 설계되었습니다. 또한, 경량화된 설계와 경량화된 패러미터 구성으로 기존 모델보다 7배 적은 파라미터를 가지고 있습니다.

- **Technical Details**: ASTRA는 U-Net 기반의 피쳐 추출기를 활용하여 장면 대표성을 포착하고, 그래프 인식 트랜스포머 인코더를 통해 사회적 상호작용을 캡쳐합니다. 이 모델은 결정론적(prediction) 및 확률적(stochastic) 결과를 모두 생성할 수 있으며, 확률적 예측은 조건부 변분 오토인코더(Conditional Variational Auto-Encoder, CVAE)를 통해 생성됩니다. 또한, 손실 함수에 대한 가중 패널티를 도입하여 다양한 상태에서 우수한 예측 성능을 발휘합니다.

- **Performance Highlights**: ASTRA는 ETH-UCY 데이터셋에서 결정론적 27%, 확률적 10%의 평균적인 성능 향상을 보였으며, PIE 데이터셋에서는 26% 향상되었습니다. 이러한 성과 외에도, 이 모델은 다양한 관점의 일반화 능력을 갖추고 있어 Bird's Eye View(BEV) 및 Ego-Vehicle View(EVV)에서 모두 적용 가능합니다.



### From Explainability to Interpretability: Interpretable Policies in Reinforcement Learning Via Model Explanation (https://arxiv.org/abs/2501.09858)
Comments:
          Accepted to Deployable AI (DAI) Workshop at the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 본 연구에서는 설명 가능성(explainability)과 해석 가능성(interpretability) 간의 간극을 메우기 위한 새로운 모델 비의존적 접근 방식을 제안합니다. 이를 위해 Shapley 값을 활용하여 복잡한 심층 강화 학습 정책을 투명한 표현으로 변환합니다. 제안한 방법은 지역적 설명을 넘어 정책 해석에 Shapley 값을 사용하여 해석 가능한 정책을 생성하는 것을 목표로 합니다.

- **Technical Details**: 강화 학습(RL)은 보상 함수에 의해 정의된 최상의 결과로 의사 결정을 학습하는 중요한 기계 학습 기술입니다. DNN(Deep Neural Networks)을 사용하는 RL 모델은 종종 '블랙 박스'로 간주되어 해석이 어렵습니다. 본 연구는 모델 해석을 위해 Shapley 값을 활용하는 방법을 도입하여, 고성능의 해석 가능한 정책을 생성하는 구조를 제시합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 기존의 심층 강화 학습 알고리즘에 대해 평가되었으며, 두 개의 고전적인 제어 환경에서 성능을 검증했습니다. 결과는 제안된 접근 방식이 원래 모델의 성능을 유지하는 동시에 더 안정적인 해석 가능한 정책을 생성함을 보여주었습니다. 이러한 결과는 신뢰성 있고 이해하기 쉬운 RL 모델의 개발에 기여할 것입니다.



### CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation (https://arxiv.org/abs/2501.09838)
Comments:
          Accepted in the 2025 WACV workshop GeoCV

- **What's New**: 본 연구에서는 CrossModalityDiffusion라는 새로운 모듈형 프레임워크를 제안하여 다양한 감지 모달리티 간에서 이미지를 생성합니다. 이 프레임워크는 장면 기하학에 대한 사전 정보 없이도 다양한 시점에서 이미지를 합성할 수 있는 가능성을 제공합니다. 다수의 입력 이미지를 처리하여 장면 구조를 인코드하는 기하학적 인식 피처 볼륨을 생성합니다.

- **Technical Details**: CrossModalityDiffusion는 모달리티별 인코더를 사용하여 여러 이미지를 입력으로 받고, 입력 카메라 위치에 상대적인 장면 구조를 암시하는 피처 볼륨을 생성합니다. 이 피처 볼륨은 입력 모달리티를 통합하는 공통 공간에서 배치되며, 부피 렌더링 기법을 사용하여 새로운 관점에서 피처 이미지를 렌더링합니다. 이렇게 생성된 피처 이미지는 특정 모달리티를 위한 확산 모델의 조건부 입력으로 활용됩니다.

- **Performance Highlights**: ShapeNet 자동차 데이터셋에서 CrossModalityDiffusion의 성능을 검증한 결과, 다양한 이미징 모달리티 간에 정확하고 일관된 새로운 관점을 효과적으로 합성할 수 있음을 보여주었습니다. 본 연구는 다양한 이미지 감지 모달리티 간의 새로운 시점 합성 문제를 해결하는 데 중점을 두고 있으며, 이전 모델에 비해 일반화된 성능을 나타냅니다.



### Bridging Language Barriers in Healthcare: A Study on Arabic LLMs (https://arxiv.org/abs/2501.09825)
- **What's New**: 본 논문은 다국어 이해와 의료 지식을 모두 갖춘 대형 언어 모델(LLM) 개발의 도전을 조사합니다. 단순히 의료 데이터를 번역하는 것으로는 목표 언어에서의 임상 작업에서 강력한 성능을 보장할 수 없음을 보여줍니다. 실험 결과, 훈련 데이터의 최적 언어 혼합이 다양한 의료 작업에 따라 상이하게 나타났습니다. 우리는 신중하게 조정된 언어 비율을 가진 대형 모델이 모국어 임상 작업에서 우수한 성능을 달성한다는 사실을 발견했습니다.

- **Technical Details**: 본 연구에서는 아랍어 의료 작업에 대한 LLM의 성능을 평가하기 위해 Llama3.1 모델을 중점적으로 다룹니다. 기존 LLM의 번역, 패러프레이징 및 합성 데이터 생성 기법을 활용하여 아랍어 의료 데이터셋을 보강하는 방법을 탐구합니다. 우리는 다양한 원본 및 합성 아랍어 의료 데이터의 혼합을 사용하여 Llama 3.1을 세부 조정(fine-tuning)하였으며, 이는 다양한 임상 작업에서 모델 성능에 미치는 영향을 분석하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 모든 모델 계열에서 대형 언어 모델이 아랍어 의료 기준에서 제한된 성능을 보인 것으로 나타났습니다. Llama3.1은 영어에서는 높은 정확도를 기록했지만 아랍어로는 큰 성능 저하가 있었습니다. 반면 Qwen2.5 모델은 아랍어 작업에 대해 상대적으로 더 나은 성능을 보였지만 여전히 최적 수준에 미치지 못했습니다. 향후 연구는 아랍어 성능 향상 및 영어 능력 달성을 위한 전략을 탐구하는 데 집중할 예정입니다.



### Generalized Single-Image-Based Morphing Attack Detection Using Deep Representations from Vision Transformer (https://arxiv.org/abs/2501.09817)
- **What's New**: 이번 논문에서는 얼굴 인식 시스템(Face Recognition Systems, FRS)에 대한 심각한 위협을 방지하기 위해, 모핑 공격 탐지를 위한 새로운 알고리즘인 S-MAD(Single-image-based Morphing Attack Detection)를 제안합니다. 특히, Vision Transformer (ViT) 구조를 활용하여 이미지의 로컬(local) 및 글로벌(global) 정보를 효과적으로 통합하는 방법을 탐구합니다. 이는 다양한 유형의 감지 알고리즘(MAD) 중에서도 특히 단일 이미지에 기반한 공격 탐지 기술을 발전시키는 데 중요한 기여를 합니다.

- **Technical Details**: 제안된 S-MAD 알고리즘은 MTCNN을 사용하여 얼굴 영역을 식별하고, 384x384 픽셀로 리사이즈한 후 입력 이미지를 32x32 픽셀 크기의 패치(patch)로 분할합니다. 각 패치는 선형 프로젝션을 통해 임베딩(embedding)으로 변환되며, 자기 주의 메커니즘(self-attention mechanism)을 포함하는 24층의 트랜스포머 인코더(transformer encoder)를 통해 처리됩니다. 이 과정에서 여러 개의 관점에서 특징을 추출할 수 있어, 다양한 모핑 공격에 대한 탐지 능력이 향상됩니다.

- **Performance Highlights**: 논문에서 제안한 S-MAD 알고리즘은 여러 공개 데이터셋을 대상으로 한 실험을 통해 성능 평가를 진행하였습니다. 교차 데이터셋 테스트에서의 성능 개선이 입증되었으며, 인트라 데이터셋 테스트(intra-dataset testing)에서는 유사한 성능을 보였습니다. 기존의 딥러닝 구조로 개발된 다른 최신 알고리즘들과 비교하여, 제안한 알고리즘이 디지털 입력에서 일반화된 성능을 나타내는 것으로 분석되었습니다.



### Enhancing Generalization in Chain of Thought Reasoning for Smaller Models (https://arxiv.org/abs/2501.09804)
- **What's New**: 이 논문은 Chain-of-Thought (CoT) 추론을 소형 언어 모델(small language models)에서 효과적으로 수행하기 위한 혁신적인 접근법을 제안합니다. PRompt-Assisted Domain-Adversarial fine-tuning (PRADA) 프레임워크는 다양한 CoT 도메인을 통합하여 소형 LLM의 일반화 능력을 개선하는 방법을 모색합니다. 이를 통해 복잡한 태스크에 대한 대처를 개선하고, 보다 강력한 CoT 일반화를 가능하게 합니다.

- **Technical Details**: PRADA는 (1) 대형 teacher 모델이 다양한 CoT 추론 응답을 생성하도록 유도하며, (2) prompt learning layers를 통해 소형 모델에 도메인 독립적 지식을 습득하도록 돕고, (3) 도메인 적대적 미세 조정을 통해 모델이 불변 도메인 특징을 학습하게끔 합니다. 이러한 접근법은 CoT 지식 증류 과정에서 발생할 수 있는 일반화 열화를 해결하기 위한 것입니다.

- **Performance Highlights**: PRADA는 12개의 다양한 도메인을 대상으로 하는 실험에서 기존 CoT 지식 증류 방법들보다 뛰어난 성능을 보였습니다. 특히 PRADA를 활용한 소형 LLM은 새로운 미지의 도메인에서도 뛰어난 일반화 능력을 발휘하며, 모델의 설명 가능성을 높이는 결과를 보였습니다.



### Multiple Choice Questions: Reasoning Makes Large Language Models (LLMs) More Self-Confident Even When They Are Wrong (https://arxiv.org/abs/2501.09775)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 답변에 대한 자신감이 모델이 직접 답변할 때와 이유를 먼저 제공한 후에 답변할 때 어떻게 달라지는지를 연구합니다. 모델들이 이유를 제시할 때 더 높은 자신감을 보이며, 이는 정답이든 오답이든 관계없이 관찰됩니다. 이러한 결과는 LLM의 동작 방식뿐만 아니라 인간의 인지 구조와도 연결되어 설명됩니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식을 사용하여 LLM에 질문을 제시합니다. 첫 번째 접근은 직접 답변을 요구하는 방법이고, 두 번째 접근은 문제 해결을 위한 단계적 사고(Chain of Thought, CoT)를 요구하는 것입니다. 이 연구는 57개의 카테고리와 15,000개 이상의 질문을 포함하는 MMLU(Massive Multitask Language Understanding) 벤치마크를 사용하여 여러 LLM 모델을 평가합니다.

- **Performance Highlights**: 모델들은 대개 CoT 방식으로 질문에 대답할 때 정답을 선택할 때의 확률이 증가하는 경향을 보입니다. 또한, 이유를 제공한 경우 모든 모델에서 선택한 옵션에 대한 자신감이 높아지는 게 관찰되었습니다. 이 결과는 모델의 정확성을 향상시키고, 선택한 옵션에 대한 확신을 증가시켜, LLM의 응답 평가 시 이러한 확률 추정치를 적용하는 데 주의를 기울여야 함을 시사합니다.



### EVAL: EigenVector-based Average-reward Learning (https://arxiv.org/abs/2501.09770)
Comments:
          Accepted at the AAAI-25 8th Workshop on Generalization in Planning. arXiv admin note: text overlap with arXiv:2501.09080

- **What's New**: 이번 논문에서는 평균 보상(RL) 문제를 해결하기 위한 새로운 알고리즘, EigenVector 기반 평균 보상 학습(EVAL)을 소개합니다. EVAL은 엔트로피_regularization(entropy regularization)을 포함하여 평균 보상 목표 함수에 대한 일상적인 접근 방식을 확장합니다. 이 연구는 기존의 할인(discount) 기반 접근과는 달리, 다양한 환경에서의 안정성과 수렴 속도 측면에서 더 나은 성능을 발휘하는 것을 보입니다.

- **Technical Details**: 이 논문에서는 강화 학습에서 엔트로피 정규화(entropy regularization)된 평균 보상 문제를 해결하는 새로운 방법을 제안합니다. EVAL 알고리즘은 신경망(neural networks)을 사용한 함수 근사(function approximation)에 기반하여 일반적인 설정에서도 적용 가능하도록 확장됩니다. 기본적으로 EVAL은 최적 정책(optimal policy)과 평균 보상률(average reward-rate)을 단일 행렬(matrix) 특성을 통해 접근할 수 있도록 해줍니다.

- **Performance Highlights**: 전통적인 제어 벤치마크에서 EVAL 알고리즘을 실험적으로 평가한 결과, 본 방법이 기존의 알고리즘과 비교하여 안정성 및 수렴 속도 측면에서 유리하다는 것을 발견했습니다. 특히, 평균 보상 문제를 엔트로피 정규화 없이 해결할 수 있는 방법을 제공하여 다양한 상황에 대응할 수 있는 가능성을 보여줍니다. 이러한 접근 방식은 기존 DQN 설정의 최소한의 변경으로 구현 가능하여 연구자들이 쉽게 접근할 수 있도록 돕습니다.



### Can Large Language Models Predict the Outcome of Judicial Decisions? (https://arxiv.org/abs/2501.09768)
- **What's New**: 이 논문은 아랍어 법정 판단 예측(Arabic Legal Judgment Prediction, LJP)에 대한 연구를 진행하며, 사우디 상업 법원의 판결 데이터를 수집하여 새로운 LJP 데이터셋을 개발했습니다. 특히, 기존의 고급 모델이 아랍어에서 낮은 성능을 보이는 문제를 해결하기 위해 아랍어 특화 모델을 벤치마킹했습니다. 또한 데이터셋, 구현 코드 및 모델을 공개하여 향후 연구의 기반을 마련했습니다.

- **Technical Details**: 연구에서는 LLaMA-3.2-3B 및 LLaMA-3.1-8B와 같은 최첨단 오픈 소스 LLM 모델을 다양한 설정(Zero-shot, One-shot, Fine-tuning)에서 평가했습니다. 이 과정에서 BLEU와 ROUGE와 같은 양적 평가 지표와 Coherence, 법적 언어, 명확성과 같은 질적 평가를 결합한 포괄적인 평가 프레임워크를 사용했습니다. QLoRA를 활용한 파인튜닝을 통해 작은 모델들이 특정 작업에서 더 큰 모델과 유사한 성능을 내도록 하는 방법을 찾았습니다.

- **Performance Highlights**: 결과적으로 파인튜닝된 소형 모델이 작업 특화 컨텍스트에서 대형 모델과 유사한 성능을 달성하는 반면, 자원 효율성은 더 크게 개선되었습니다. 프롬프트 엔지니어링과 파인튜닝이 모델 출력에 미치는 영향도 조사하여 성능 변동성과 지시 민감성에 대한 통찰력을 제공했습니다. 이 연구는 아랍어 법적 NLP의 발전을 위한 강력한 평가 프레임워크를 제시하여 LJP 연구를 촉진할 것으로 기대됩니다.



### LeMo: Enabling LEss Token Involvement for MOre Context Fine-tuning (https://arxiv.org/abs/2501.09767)
- **What's New**: 이번 논문에서는 LeMo라는 새로운 LLM(대형 언어 모델) 미세 조정 시스템을 제안합니다. LeMo는 긴 문맥 상황에서 고유한 토큰 수준의 희소성 메커니즘인 Contextual Token Sparsity를 활용하여 메모리와 계산 효율성을 최적화합니다. 기존의 미세 조정 방법들은 활성화 메모리 문제를 해결하지 못했으나, LeMo는 정보 기반 토큰 제거 및 패턴 예측 등을 통해 이러한 문제를 극복합니다.

- **Technical Details**: LeMo는 세 가지 주요 기술로 구성됩니다: (1) Token Elimination은 동적으로 중복 토큰을 식별하여 계산에서 제외합니다. (2) Pattern Prediction은 훈련된 예측기를 활용하여 희소성 패턴을 추정합니다. (3) Kernel Optimization은 토큰 선택 및 패딩 동안 불필요한 메모리 이동을 제거하고, 세그먼트 기반의 그래디언트 계산 방법으로 활성화 메모리의 피크를 줄이는 방식을 채택합니다.

- **Performance Highlights**: LeMo는 다양한 LLM 아키텍처에 호환되는 엔드 투 엔드 미세 조정 시스템으로 구현되었습니다. 평가 결과, LeMo는 메모리 사용량을 최대 1.93배 줄이고, 속도를 최대 1.36배 향상시켜 최신 미세 조정 시스템보다 뛰어난 성능을 보였습니다. 이를 통해 긴 문맥 시퀀스를 처리하면서도 적은 자원을 요구하는 혁신적인 접근법을 제시하고 있습니다.



### Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning (https://arxiv.org/abs/2501.09766)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 도구 사용 능력을 향상시키기 위해 외부 도구와 결합하는 새로운 방법론을 제시합니다. 복잡한 상황에서 모델의 부족한 성능(Deficiency)을 극복하기 위해 반복적인 강화 세부 조율 전략(iTool)을 도입하며, 이를 통해 모델이 복잡한 도구 사용 시나리오에서도 효율적으로 학습할 수 있도록 합니다. 또한, 고난이도 데이터에서 학습할 수 있도록 더 쉬운 난이도에서 시작하는 Warm-up SFT 전략을 사용합니다.

- **Technical Details**: LLMs가 도구를 사용하는 과정은 사용자 질문에 답하기 위해 적절한 함수 선택 및 도구 호출을 수행하는 것입니다. 논문에서는 정책 모델의 피드백을 기반으로 부족 데이터를 식별하고, 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS)을 통해 세밀한 선호 쌍을 수집합니다. 이를 통해 모델을 Ground Truth에 맞도록 조정하고 부족한 부분을 misalign하게 만드는 정책을 업데이트하여 효과적으로 부조화를 해소합니다.

- **Performance Highlights**: iTool은 개수로만 7억 개의 매개변수를 가진 모델임에도 불구하고, 많은 대규모 공개 모델보다 뛰어난 성능을 보이며, 최고의 비공개 모델과도 경쟁할 수 있습니다. 실험 결과, 데이터 규모가 증가해도 적절한 훈련 성과를 유지하며 복잡한 도구 사용 시나리오에서 두각을 나타내고 있습니다. 이러한 결과는 반복적인 강화 세부 조율 전략의 성공적인 적용을 잘 보여줍니다.



### Enhancing the De-identification of Personally Identifiable Information in Educational Data (https://arxiv.org/abs/2501.09765)
Comments:
          14 pages, 1 figure; This work has been submitted to the IEEE for possible publication

- **What's New**: 본 연구는 개인 식별 정보(PII) 탐지를 위한 비용 효율적이고 효과적인 솔루션으로 GPT-4o-mini 모델을 조사합니다. 우리는 프롬프트(prompting)와 파인튜닝(fine-tuning) 접근 방식을 비교하고, Microsoft Presidio 및 Azure AI Language와 같은 기존 프레임워크와 GPT-4o-mini의 성능을 비교합니다. 두 개의 공개 데이터셋인 CRAPII와 TSCC에 대한 평가 결과, 파인튜닝된 GPT-4o-mini 모델이 PII 탐지에서 우수한 성과를 거두었음을 보여주었습니다.

- **Technical Details**: PII는 개인을 식별할 수 있는 정보로, 학습 기술이 점점 더 교육에서 중요한 역할을 하면서 보호가 필수적입니다. 특히, 최근의 대형 언어 모델(LLMs)의 발전은 PII 탐지를 향상시킬 기회를 제공합니다. 본 연구에서는 고급 AI 모델을 사용하여 Named Entity Recognition (NER) 과정을 개선하여 PII 보호를 증대시키는 방법을 탐구합니다.

- **Performance Highlights**: 파인튜닝된 GPT-4o-mini 모델은 CRAPII 데이터셋에서 0.9589의 리콜(recall) 점수를 달성했고, 정밀도(precision) 점수는 삼중 상승했습니다. 또한 Azure AI Language의 약 10분의 1으로 계산 비용을 줄이는 성과를 보였습니다. 이러한 결과는 교육 데이터에서 PII 탐지를 위한 정확하고 비용 효율적인 도구로서 파인튜닝된 GPT-4o-mini의 잠재력을 강조합니다.



### VERITAS: Verifying the Performance of AI-native Transceiver Actions in Base-Stations (https://arxiv.org/abs/2501.09761)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 AI-native 수신기의 성능을 향상시키기 위한 새로운 프레임워크인 VERITAS를 제안합니다. VERITAS는 AI-native 송수신기가 배포된 후 무선 채널에서 발생할 수 있는 분포 변화(distribution shifts)를 지속적으로 감지하고, 필요 시 재학습(retraining)을 자동으로 수행하는 시스템입니다. 이 프레임워크는 더 넓은 환경에서도 AI 기반 수신기의 신뢰성을 높이고 통신 오버헤드를 줄이는 것을 목표로 합니다.

- **Technical Details**: VERITAS는 5G 파일럿 신호를 사용하여 보조 신경망(auxiliary neural network)을 통해 무선 채널을 모니터링합니다. 여기서 채널 프로파일(channel profile), 송신기 속도(transmitter speed), 및 지연 확산(delay spread)와 같은 요소들을 감지하며, 이러한 요소에 변화가 생기면 전통적인 수신기(reference receiver)를 동시에 작동시킵니다. 이후 AI-native 수신기와 전통적 수신기 간의 비트 확률(bit probabilities)을 비교하여 재학습이 필요한지를 결정합니다.

- **Performance Highlights**: VERITAS는 채널 프로파일에서는 99%, 송신기 속도에서는 97%, 지연 확산에서는 69%의 높은 정확도로 변화를 감지할 수 있습니다. 이러한 감지 후, 각 변동 요소에 대해 86%, 93.3%, 94.8%의 경우에 신속하게 재학습을 시작하는 것으로 평가되었습니다. 이러한 성과는 AI-native 수신기가 다양한 환경에서도 안정성을 유지할 수 있도록 보장합니다.



### Towards Large Reasoning Models: A Survey on Scaling LLM Reasoning Capabilities (https://arxiv.org/abs/2501.09686)
Comments:
          36 pages, 5 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 복잡한 추론 작업을 다루기 위한 새로운 접근 방식을 제시합니다. 특히, '사고(thought)' 개념을 도입해 LLM이 인간의 추론 과정을 모방할 수 있도록 하고, 강화 학습(reinforcement learning)을 통해 LLM의 학습 능력을 증대시키는 경향을 강조합니다. 이러한 접근 방식을 통해, LLM의 추론 정확성을 크게 향상시킬 수 있는 새로운 연구 경계를 탐구하고 있습니다.

- **Technical Details**: LLM의 훈련에서는 트랜스포머(transformer) 아키텍처가 사용되며, 대규모 텍스트 코퍼스에서 사전 훈련(pretraining)을 거치게 됩니다. 최근의 연구는 사람의 주석 없이 LLM 주도 검색 알고리즘을 통해 자동으로 추론 궤적을 생성하는 방법을 강조하며, 이는 LLM의 추론 능력을 대폭 확장합니다. 또한, 과정 보상 모델(Process Reward Models, PRMs)을 통해 LLM의 훈련과 추론에서 더욱 효율적으로 작동하도록 지원합니다.

- **Performance Highlights**: 최근 연구에서는 테스트 시간 동안 LLM이 '더 많은 토큰'을 사용하여 사고를 할 수 있도록 장려하면 추론 정확성이 크게 향상될 수 있음을 보여줍니다. 이 연구는 LLM이 목표에 맞는 추론 단계를 생성할 수 있도록 하고, 논문에서 제안하는 RL 기반 훈련 및 검색 기반 테스트 시간 확장 방법은 LLM의 추론 능력을 최대한 활용하는 데 기여하고 있습니다. 'OpenAI의 o1 시리즈'는 이러한 접근 방법의 효과를 입증하는 중요한 이정표로 자리 잡고 있습니다.




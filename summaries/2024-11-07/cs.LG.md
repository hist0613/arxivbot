New uploads on arXiv(cs.CL)

### Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? (https://arxiv.org/abs/2411.04118)
Comments:
          Accepted to EMNLP 2024 Main Conference as Long Paper (Oral)

- **What's New**: 이 논문에서는 의학 응용을 위해 개발된 여러 모델들의 성능을 평가하고, 일반 도메인 모델들과의 비교를 통해 DAPT(Domain-Adaptive Pretraining)의 효과에 대한 의문을 제기합니다.

- **Technical Details**: 연구팀은 7개의 의학 LLM과 2개의 VLM을 선택하여, 각 의학 모델을 해당하는 일반 모델과 직접 비교했습니다. 각 모델에 대해 최적의 프롬프트를 독립적으로 선정하고, 통계적 불확실성을 고려한 후 성능을 평가했습니다.

- **Performance Highlights**: 의학 LLMs는 3-shot 설정에서 12.1%의 경우에만 베이스 모델을 초과하여, 나머지 87.9%의 경우에서는 동점이거나 그보다 성능이 떨어진다는 점을 발견했습니다. 이 결과는 DAPT가 의학 분야에서 항상 성능 향상을 보장하지 않음을 시사합니다.



### Self-Consistency Preference Optimization (https://arxiv.org/abs/2411.04109)
Comments:
          16 pages, 3 figures

- **What's New**: 이번 논문에서는 Self-consistency Preference Optimization (ScPO)라는 방법을 제안하여 모델이 비지도 학습 문제에 대해 일관된 답을 학습하도록 하는 새로운 접근을 소개합니다. 이 방법은 모델 훈련 과정에서 자가 일관성(self-consistency) 개념을 활용하여, 복잡한 문제 해결 작업의 성능을 향상시킵니다.

- **Technical Details**: ScPO는 비지도 학습 단계에서 모델이 생성한 문제와 쿼리를 사용하여 복잡한 문제 해결 작업을 수행합니다. 이 방법의 과정은 (i) 모델이 생성한 쿼리 선택, (ii) 가장 자가 일관성이 높은 응답(우승자)과 가장 낮은 응답(패자)에 대한 선호 쌍을 주석 달기, (iii) 모델의 선호 쌍에 대한 신뢰도에 따라 가중치가 조정된 손실 함수 최적화로 구성됩니다. 이 논문은 또한 라벨이 부여된 인스턴스와 미라벨 인스턴스에서 공동으로 LLM을 훈련하는 반지도 변형도 제안합니다.

- **Performance Highlights**: Llama-3 8B 모델을 사용한 실험 결과, ScPO는 GSM8K에서 22.74% 그리고 MATH에서 5.26%의 제로샷 정확도를 향상시키며, 감독 학습 방식의 성능에 가까운 결과를 도출했습니다. 또한 ScPO를 사용하여 ZebraLogic의 난해한 논리 퍼즐에서 6.5%의 정확도 향상을 보여주었으며, 이는 Llama-3 70B 및 Gemma-2 27B와 같은 더 큰 LLM들을 초월하는 성능입니다.



### Summarization of Opinionated Political Documents with Varied Perspectives (https://arxiv.org/abs/2411.04093)
- **What's New**: 이 논문은 다양한 정치적 관점을 독립적으로 요약하는 새로운 데이터셋인 PoliSum과 그 과제를 소개합니다. 이 과제는 혼합된 정치적 관점을 담고 있는 뉴스 기사의 모음에서 각 정치적 관점에 대한 요약을 생성하는 것을 목표로 하고 있습니다.

- **Technical Details**: PoliSum 데이터셋은 The Flip Side에서 수집된 편향된 텍스트와 쌍으로 된 요약으로 구성되어 있습니다. 요약 작업은 양쪽 정치적 관점의 의견을 나타내는 독립적인 요약을 생성하는 것이며, 신뢰성 있는 요약을 생성하는 모델의 성능을 평가하기 위한 초기 프레임워크를 제공합니다.

- **Performance Highlights**: 10개의 다양한 모델을 벤치마킹한 결과, 최근 모델들(GPT-4o 포함)이 이 작업에서 성능이 좋지만, 모든 모델이 의도된 관점에 충실한 요약을 생성하는데 어려움을 겪고 있음을 발견했습니다. 특히, 모델의 추출 행동은 입력 문서의 특성에 따라 달라지며, 입력 문서의 위치와 길이, 자극적인 용어의 사용이 결과에 영향을 미친다는 분석 결과가 나왔습니다.



### A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreemen (https://arxiv.org/abs/2411.04090)
Comments:
          35 pages, 1 figure

- **What's New**: 이 논문은 주석에서의 불일치를 포착하는 데 초점을 맞춘 새로운 콘텐츠 조정 프레임워크를 제안합니다. 이 프레임워크는 다중 작업 학습(multitask learning) 접근법을 사용하여 독성 분류(toxicity classification)와 주석 불일치(annotation disagreement)를 보조 작업(auxiliary task)으로 적절하게 다룹니다.

- **Technical Details**: 우리는 Conformal Prediction(UQ) 기법을 활용하여 주석에서의 모호성과 모델의 예측 독성에 대한 고유한 불확실성을 고려합니다. 이 프레임워크는 조정자들이 주석 불일치에 대한 임계값(threshold)을 조정할 수 있어 인간 검토 결과 최적화를 지원합니다.

- **Performance Highlights**: 제안된 접근법은 모델의 성능, 보정(calibration), 불확실성 추정을 향상시킬 뿐만 아니라, 단일 작업(single-task) 방법에 비해 파라미터 효율성과 검토 프로세스 개선을 제공합니다.



### M3SciQA: A Multi-Modal Multi-Document Scientific QA Benchmark for Evaluating Foundation Models (https://arxiv.org/abs/2411.04075)
- **What's New**: 새로운 벤치마크 M3SciQA가 소개되었으며, 이는 다중 문서 및 다중 모달 과학 질문 응답을 평가하기 위해 설계되었습니다. 기존의 벤치마크는 주로 단일 문서 및 텍스트 전용 작업에 중점을 두었으나, M3SciQA는 1,452개의 전문가 지정 질문을 포함하여 70개의 자연어 처리(NLP) 논문 클러스터를 아우릅니다.

- **Technical Details**: M3SciQA 벤치마크는 시각적 컨텍스트 질문과 참조 기반 질문의 두 가지 유형으로 구성됩니다. 각각의 클러스터는 주요 논문과 그에 인용된 모든 문서를 포함하여, 다중 문서 및 다중 모달 접근 방식을 requirment합니다.

- **Performance Highlights**: M3SciQA를 통해 18개의 기초 모델에 대한 광범위한 평가가 이루어졌고, 현재의 모델들이 인간 전문가에 비해 다중 모달 정보 검색 및 여러 과학 문서 간의 추론에서 상당히 저조한 성능을 보인다는 점이 드러났습니다. 예를 들어, 최고의 성능을 보인 모델인 GPT-4o는 MRR(Mean Reciprocal Rank) 0.488을 기록, 전문가 점수 0.796에 비해 0.308의 성능 격차를 보였습니다.



### Beemo: Benchmark of Expert-edited Machine-generated Outputs (https://arxiv.org/abs/2411.04032)
- **What's New**: 본 논문은 전문가가 편집한 머신 생성 텍스트(Compared with conventional single-author MGT benchmarks)에서 발생한 새로운 벤치마크인 'Beemo'를 소개합니다.

- **Technical Details**: Beemo 벤치마크는 6.5k 개의 인간 작성 텍스트와 10개의 instruction-finetuned LLM(large language models)에서 생성된 텍스트를 포함하며 전문가들이 다양한 용도로 편집하였습니다. 또한 13.1k 개의 머신 생성 및 LLM 편집 텍스트를 포함하여 다양한 MGT 탐지 평가를 가능하게 합니다.

- **Performance Highlights**: 전문가 편집 결과는 MGT 탐지를 회피하는 경향이 있으며, LLM 편집 텍스트는 인간 작성으로 인식될 가능성이 낮다는 결과를 나타냈습니다. Beemo와 모든 자료는 공개되어 있습니다.



### Prompt Engineering Using GPT for Word-Level Code-Mixed Language Identification in Low-Resource Dravidian Languages (https://arxiv.org/abs/2411.04025)
Comments:
          Accepted at FIRE 2024 (Track: Word-level Language Identification in Dravidian Languages)

- **What's New**: 이 연구는 다국어 사회에서의 언어 식별(Language Identification, LI)에서 발생하는 과제를 해결하기 위해 드라비다어(Dravidian languages)에서 단어 단위 LI에 초점을 맞춘 새로운 방법론을 제안합니다. 특히, 소셜 미디어에서 완벽하게 혼합된 언어 사용을 처리할 수 있는 모델의 유용성을 탐구하였습니다.

- **Technical Details**: 본 연구에서는 GPT-3.5 Turbo를 활용하여 단어를 정확하게 분류할 수 있는지 분석하였습니다. 입력 온도(temperature) 값은 0.7, 0.8, 0.9등 세 가지로 설정하였으며, 다양한 데이터셋(데이터셋: Tulu, Kannada, Tamil, Malayalam)을 사용하였습니다. 이 데이터셋은 코드 혼합(code-mixing) 문장을 포함하며, 각 언어의 특징을 반영하여 여러 클래스에 분류되었습니다.

- **Performance Highlights**: 실험 결과, Kannada 모델이 Tamil 모델보다 대다수의 메트릭에서 일관되게 높은 성능을 보였고, 정확도와 신뢰성 측면에서 우수한 결과를 나타냈습니다. 반면, Tamil 모델은 중간 정도의 성능을 보였으며, 특히 정밀도(precision)와 재현율(recall) 개선이 필요하였습니다.



### WorryWords: Norms of Anxiety Association for over 44k English Words (https://arxiv.org/abs/2411.03966)
- **What's New**: WorryWords는 44,450개의 영어 단어와 관련된 불안(anxiety) 연관성을 수작업으로 도출한 최초의 대규모 데이터베이스입니다. 이를 통해 불안과 언어의 관계를 심층적으로 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: 이 연구에서는 44,450개의 영어 단어에 대한 불안 점수를 수집하였으며, 평균 신뢰도 점수(split-half reliability score)는 0.82로 기존 감정 리소스와 비슷한 수준입니다. WorryWords는 불안이 다른 감정 구조와 어떻게 관련되어 있는지, 나이에 따라 불안 단어가 어떻게 습득되는지를 연구하는 데 사용됩니다.

- **Performance Highlights**: WorryWords를 활용하면 텍스트 속에서 불안의 변화를 정확하게 추적할 수 있습니다. 불안에 대한 연구 분야는 심리학, 자연어 처리(NLP), 공공 건강, 사회 과학 등 폭넓은 주제를 다루게 될 것입니다.



### What Really is Commonsense Knowledge? (https://arxiv.org/abs/2411.03964)
Comments:
          Code and data will be released together with the next version of the paper

- **What's New**: 이 논문은 commonsense(상식) 지식의 정의를 여러 프레임워크에 기반하여 통합적으로 정리하고, 이를 바탕으로 CommonsenseQA 및 CommonsenseQA 2.0 데이터셋의 비상식 지식 인스턴스 비율을 분석합니다.

- **Technical Details**: 이 연구에서는 상식 지식을 기존의 정의 framework(프레임워크)를 기반으로 분석하고, 그 결과를 CommonsenseQA와 CommonsenseQA 2.0 데이터셋에 적용하여 LLMs(대형 언어 모델)의 성능 차이를 실험합니다. 특히, 각 인스턴스의 지식 유형을 commonsense와 referenced knowledge(참조 지식)으로 구분합니다.

- **Performance Highlights**: 실험 결과, CommonsenseQA에서 27%, CommonsenseQA 2.0에서 56%가 참조 지식 인스턴스인 것으로 나타났으며, LLMs는 commonsense 지식 인스턴스에서 평균적으로 4~7점 낮은 정확도를 보였습니다.



### How Does A Text Preprocessing Pipeline Affect Ontology Syntactic Matching? (https://arxiv.org/abs/2411.03962)
Comments:
          13 pages, 26 figures, 4 tables

- **What's New**: 본 논문에서는 Ontology Matching (OM) 시스템에서 텍스트 전처리 파이프라인의 효과를 분석했습니다. 모든 OM 시스템이 Tokenisation, Normalisation, Stop Words Removal, Stemming/Lemmatisation을 포함하지만, 표준화 부족으로 인해 매핑 결과의 다양성이 발생하고 있습니다. 여러 실험을 통해 Tokenisation과 Normalisation의 효율성을 확인했으며, Lemmatisation 또는 Stemming의 선정이 작업에 따라 다름을 발표했습니다.

- **Technical Details**: 이 연구는 49개의 다양한 정렬을 포함한 8개의 OAEI 트랙 저장소에서 수행되었습니다. 실험에서는 Tokenisation과 Normalisation이 Stop Words Removal과 Stemming/Lemmatisation보다 효과적이라는 결과가 도출되었습니다. 특히, Porter Stemmer와 Snowball Stemmer가 Lancaster Stemmer보다 성능이 뛰어난 것으로 나타났으며, POS Tagging이 Lemmatisation에 도움이 되지 않는다는 결론에 도달했습니다. 또한, 덜 효과적인 Stop Words Removal 및 Stemming/Lemmatisation에 대한 새로운 맥락 기반 파이프라인 복구 접근법을 제안했습니다.

- **Performance Highlights**: 새로 제안한 맥락 기반 파이프라인 복구 방법은 OM 작업의 정확성과 전반적인 성능을 크게 향상시켰습니다. 특히, 대형 언어 모델(LLMs)은 텍스트 전처리 없이는 구문 매칭을 수행할 수 없지만, 잘못된 매핑을 수정하는 대안으로 활용될 수 있음을 논의했습니다.



### Evaluation data contamination in LLMs: how do we measure it and (when) does it matter? (https://arxiv.org/abs/2411.03923)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 평가 중 'evaluation data contamination' 문제를 다룹니다. 특히, 평가 샘플이 사전 훈련 데이터에 포함되어 있는 경우 이로 인해 평가 점수가 왜곡될 수 있다는 점에 주목합니다. 저자들은 이를 정의하고 측정하기 위한 새로운 방법론인 'ConTAM'을 제안합니다.

- **Technical Details**: ConTAM(Contamination Threshold Analysis Method)은 평가 데이터의 오염 정도와 그로 인해 모델이 실제로 불리한 이점을 갖는 정도를 함께 평가하는 방법으로, 'Estimated Performance Gain (EPG)' 개념을 중심으로 합니다. 연구에서는 n-gram 기반의 다양한 오염 측정 지표를 사용하여 13개의 벤치마크와 2개 유형의 모델에서 폭넓은 조사 결과를 제공합니다. 이 과정에서 오염 샘플의 정의, 이점 분석 및 모델별 임계값 선택이 중요한 요소로 작용함을 보여줍니다.

- **Performance Highlights**: 연구 결과, 오염이 LLM 성능 평가에 미치는 영향이 과소 평가되었다는 것이 밝혀졌습니다. 특히, 모든 벤치마크에서 긴 오염 부분 문자열을 사용하는 것이 더 효과적이었고, 모델 크기에 따라 오염의 영향이 다르게 나타났습니다. 또한, n 값이 작을수록 더 좋은 성능을 보였으며, 사전 훈련 데이터에서의 일회 발생이 중요하다는 결과를 도출했습니다.



### RAGulator: Lightweight Out-of-Context Detectors for Grounded Text Generation (https://arxiv.org/abs/2411.03920)
- **What's New**: 본 논문에서는 RAG 응용 프로그램에서 LLM(대형 언어 모델) 생성 텍스트의 의미적 맥락을 벗어난(out-of-context, OOC) 출력을 탐지하기 위해 경량 모델을 훈련시키는 방법을 제안합니다. 이를 통해 기업들이 RAG 응용 프로그램을 안전하게 활용할 수 있는 방안을 모색합니다.

- **Technical Details**: 연구팀은 DeBERTa 모델을 활용하여 OOC 탐지를 수행하는 경량의 grey-box 감지기를 개발했습니다. 데이터 생성 파이프라인을 통해 OOC 및 맥락 내 RAG 프롬프트에 대한 훈련 데이터를 생성하고, 여러 NLP 작업을 위해 구축된 공공 데이터셋에서 데이터를 수집하였습니다. 또한, BERT 기반 분류기와 수치적 특성에 따라 훈련된 앙상블 메타 분류기를 비교하였습니다.

- **Performance Highlights**: DeBERTa-v3-large 모델은 OOC 탐지 작업에서 AUROC에서 19% 및 F1 점수에서 17% 향상을 보였습니다. 이는 OOC 탐지를 위한 특별한 모델의 필요성을 강조합니다.



### Computational Analysis of Gender Depiction in the Comedias of Calder\'on de la Barca (https://arxiv.org/abs/2411.03895)
- **What's New**: 이번 연구는 스페인 17세기 작가 페드로 칼데론 데 라 바르카(Pedro Calderón de la Barca)의 비종교 극작품에서 성별 묘사를 탐구하기 위한 정량적 방법론을 개발했습니다. 연구진은 100편 이상의 연극 텍스트를 분석하여 캐릭터의 대사를 '남성' 또는 '여성'으로 분류하는 성별 분류기를 사용하였고, 어떤 텍스트 특성이 모델의 결정에 가장 큰 영향을 미치는지를 밝혀냈습니다.

- **Technical Details**: 본 연구는 자연어 처리(NLP)에 있어 최근의 발전 사항인 AI 모델의 설명 가능성을 기반으로 구축되었습니다. 성별 분류기는 캐릭터의 대사에 기반하여 훈련되었으며, 문맥 벡터(context vectors)와 임베딩(embeddings) 등의 개념을 활용하여 성별 예측 모델을 개발했습니다. 모델 예측을 이해하기 위해 문법적(grammatical) 및 의미적(semantic) 특성의 분석을 포함하였습니다.

- **Performance Highlights**: 실험 결과, 모델은 칼데론의 코메디아(comedias)에서 캐릭터의 성별을 높은 정확도로 분류할 수 있었으며(f=0.83), 특히 크로스 드레싱(cross-dressing) 캐릭터의 경우 남성 캐릭터와의 대사 유사성을 보여주었습니다. 이 연구는 동시에 작품 전체에서 성별 묘사의 글로벌 패턴을 확인할 수 있는 기준을 제시합니다.



### Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision-Language Models (https://arxiv.org/abs/2411.03888)
Comments:
          Preprint

- **What's New**: 이 논문은 멀티모달(multi-modal) 및 다국어(multilingual) 병렬 증오 발언(hate speech) 데이터셋인 Multi3Hate를 소개하며, 세계 여러 문화가 증오 발언을 어떻게 다르게 해석하는지를 조사합니다.

- **Technical Details**: Multi3Hate 데이터셋은 5개 언어(영어, 독일어, 스페인어, 힌디어, 중국어)로 구성된 300개의 밈(meme) 샘플을 포함합니다. 이 데이터셋은 다양한 문화적 배경을 가진 5명의 원주민 화자에 의해 주석이 달렸습니다. 각 국가 간의 평균 쌍별 동의율은 74%로 나타났으며, 미국과 인도 간의 동의율은 67%로 가장 낮았습니다. 5개의 대형 VLMs를 이용한 실험에서는 이러한 모델이 미국의 주석과 더 높은 일치를 보였습니다.

- **Performance Highlights**: 이 연구에서 수행된 실험에서 50개의 모델, 언어, 입력 조합 중 42개 조합이 미국의 주석과 가장 높은 일치를 보였습니다. 이러한 경향은 VLM이 비영어 사용 문화의 주석보다는 미국 문화의 주석에 더 밀접하게 일치함을 보여주며, 이는 특정 문화가 소외될 위험이 있음을 시사합니다.



### Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models (https://arxiv.org/abs/2411.03884)
- **What's New**: 본 논문에서는 transformers 아키텍처를 최적화하기 위해 폴리곤 조합 활성화 함수(PolyCom)를 제안합니다. 이 활성화 함수는 전통적인 비선형 활성화 기능의 한계를 극복하고, 데이터 내의 더 복잡한 패턴을 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: PolyCom 활성화 함수는 다항식(degree) 조합을 활용하여 더 높은 차원의 상호작용을 캡처할 수 있게 해줍니다. 특히, PolyReLU와 PolyNorm의 두 가지 인스턴스를 도입하며, 이들은 transformer 아키텍처의 통합 과정에서 다항식의 힘을 활용합니다. 폴리곤 조합 활성화 함수는 Sobolev 공간에서 일반적인 매끄러운 함수의 최적 근사 속도를 달성하는 것을 보여줍니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)에서 PolyCom을 적용한 결과, SwiGLU, GELU 및 ReLU 등 기존의 활성화 함수보다 성능이 유의미하게 향상되었으며, 수렴 속도도 크게 빨라짐을 확인했습니다. 이는 PolyCom이 딥러닝 애플리케이션에서 실질적인 이점을 제공함을 시사합니다.



### MEG: Medical Knowledge-Augmented Large Language Models for Question Answering (https://arxiv.org/abs/2411.03883)
- **What's New**: MEG라는 새로운 파라미터 효율적인 접근 방식을 제안합니다. 이는 의료 관련 LLM(대규모 언어 모델)에 그래프 임베딩(knowledge graph embeddings)을 통합하여 외부 지식을 비용 효율적으로 활용할 수 있도록 합니다.

- **Technical Details**: MEG는 경량화된 매핑 네트워크를 활용하여 KG 임베딩을 LLM의 벡터 공간으로 단방향으로 변환합니다. Mistral-Instruct(7B)를 기본 LLM으로 사용하였으며, KG 인코더는 GraphSAGE를 기반으로 하여 간단한 MLP(다층 퍼셉트론) 매핑 네트워크와 결합되었습니다. RAG(회수 증강 생성) 방식과 유사한 접근방식입니다.

- **Performance Highlights**: MEG는 Mistral-Instruct 기준선에 비해 평균 10.2% 더 높은 정확도를 달성하였고, BioMistral과 같은 전용 모델에 비해 6.7% 더 우수한 성능을 보였습니다. 4개의 인기 있는 의료 다중 선택 데이터셋에서 검증하였으며, KG 임베딩을 통합한 LLM이 의료 질문 답변에 있어 효과적임을 입증하였습니다.



### Performance evaluation of SLAM-ASR: The Good, the Bad, the Ugly, and the Way Forward (https://arxiv.org/abs/2411.03866)
Comments:
          Submitted to ICASSP 2025 SALMA Workshop

- **What's New**: 이 논문은 SLAM-ASR architectures의 성능을 분석하고, 다양한 실험을 통해 훈련 데이터에 대한 의존성과 다양한 잡음 및 시간적 변화에 대한 민감도를 조사합니다.

- **Technical Details**: SLAM-ASR 아키텍처는 음성 인코더, 다운샘플러 및 학습 가능한 선형 프로젝터로 구성됩니다. 이 구조는 음성 인코더의 음향 임베딩을 LLM의 입력 임베딩으로 변환하는 역할을 합니다.

- **Performance Highlights**: SLAM-ASR 시스템은 크로스 도메인 평가에서 저조한 성능을 보였으며, 인도메인 내에서 속도 변화나 잡음 추가와 같은 음성 변형이 성능에 significant한 영향을 미친다는 발견이 있었습니다.



### MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba (https://arxiv.org/abs/2411.03855)
- **What's New**: 본 논문에서는 Mamba라는 새로운 State Space Model (SSM) 기반 모델에 대한 Parameter-efficient fine-tuning (PEFT) 방법들의 탐색적 분석을 진행하였습니다. 기존의 Transformer 기반 모델들에 비해 Mamba 모델에서의 PEFT의 효율성과 효과를 강조하며, 신규 Mamba 특화 PEFT 방법들을 제안하였습니다.

- **Technical Details**: Mamba는 시간을 선형적으로 처리할 수 있는 모델로, 기존 Transformer의 계산 복잡성을 극복하며 긴 시퀀스를 효율적으로 처리합니다. 본 연구에서는 Mamba 아키텍처에 적합하도록 기존 PEFT 방법들을 수정하고, Mamba에 특화된 새로운 PEFT 방법들을 제안하여 성능을 극대화하는 방법을 제안합니다. 실험에서는 7개의 주요 PEFT 방법과 총 20개의 파생 변형을 벤치마킹하였습니다.

- **Performance Highlights**: Mamba에서의 PEFT는 Transformer보다 더 효과적이며, 여러 PEFT 방법들을 조합하여 성능을 향상시킬 수 있음을 보여주었습니다. 본 논문은 PEFT 방법의 조합을 효율적으로 탐색하는 기술을 제안하고, 단순한 높은 성능의 방법 조합만으로는 충분하지 않다는 것을 밝혔습니다.



### The natural stability of autonomous morphology (https://arxiv.org/abs/2411.03811)
Comments:
          Accepted for publication by the journal Morphology

- **What's New**: 본 논문은 자율 형태론(autonomous morphology)의 지속성을 설명하기 위해 형태 카테고리 간의 끌림(attraction)과 밀침(repulsion) 역학을 제안합니다. 이를 통해 자율 형태론이 어떻게 자연 언어에서 나타나며 지속될 수 있는지를 설명합니다.

- **Technical Details**: 논문에서는 'Paradigm Cell Filling Problem'(PCFP)에 대한 유추적 추론을 사용하여 자율 형태론의 구조적 특성과 지속성을 모델링합니다. 'Dissociative evidence'라는 개념은 형태 카테고리 간의 밀침 역학을 도입하여 형태들이 완전히 평준화되지 않도록 합니다.

- **Performance Highlights**: 모델링 결과, 자율 형태론은 단순한 추론 과정을 통해 자연스럽게 발생하고 지속되는 형태로 나타납니다. 이러한 결과는 자율 형태론이 인위적이지 않고 인간 언어의 본질적인 특성임을 주장합니다.



### Understanding the Effects of Human-written Paraphrases in LLM-generated Text Detection (https://arxiv.org/abs/2411.03806)
- **What's New**: 이번 연구는 LLM(대형 언어 모델) 생성 텍스트 감지에 있어 사람의 패러프레이즈(human-paraphrases)가 미치는 영향을 최초로 포괄적으로 조사합니다. 연구자는 Human & LLM Paraphrase Collection (HLPC)이라는 독창적인 데이터세트를 수집하였으며, 이를 통해 LLM 생성 감지기의 성능에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 연구는 OpenAI의 RoBERTa 및 수문자(watermark) 검출기를 포함한 최첨단 LLM 생성 텍스트 감지 모델을 사용하여 분류 실험을 수행합니다. 실험에서는 인간이 생성한 패러프레이즈와 GPT, OPT, DIPPER, BART에서 생성한 LLM 텍스트와 패러프레이즈를 포함합니다. 결과적으로 인간 패러프레이즈의 포함이 LLM 생성 검출기의 성능에 미치는 중대한 영향을 보여줍니다.

- **Performance Highlights**: 최신 연구 결과는 인간이 작성한 패러프레이즈가 LLM 생성 검출기의 TPR@1%FPR(참 긍정률) 향상에 기여하나, AUROC(곡선 아래 면적)와 정확도 간의 상쇄가 발생할 수 있다는 점을 강조하고 있습니다.



### A Comparative Study of Recent Large Language Models on Generating Hospital Discharge Summaries for Lung Cancer Patients (https://arxiv.org/abs/2411.03805)
- **What's New**: 본 연구는 대형 언어 모델(large language models, LLMs)이 임상 실무에서 퇴원 요약을 생성하는 데 어떻게 도움이 될 수 있는지를 탐구하고 있습니다.

- **Technical Details**: 연구는 1,099명의 폐암 환자의 임상 노트를 사용하였고, 그 중 50명은 테스트, 102명은 모델 조정을 위한 샘플로 사용되었습니다. 여러 LLM(GPT-3.5, GPT-4, GPT-4o, LLaMA 3 8b)의 성능을 평가했으며, BLEU, ROUGE-1, ROUGE-2, ROUGE-L과 같은 토큰 레벨 분석 및 모델이 생성한 요약과 의사가 작성한 목표 기준 간의 의미적 유사성을 측정했습니다.

- **Performance Highlights**: GPT-4o와 조정된 LLaMA 3은 탁월한 토큰 레벨 평가 지표를 보였고, LLaMA 3은 입력 길이에 관계없이 간결한 요약을 일관되게 생성했습니다. 의미적 유사성 점수는 GPT-4o와 LLaMA 3이 임상적 관련성을 잘 포착하는 우수한 모델로 나타났습니다.



### No Culture Left Behind: ArtELingo-28, a Benchmark of WikiArt with Captions in 28 Languages (https://arxiv.org/abs/2411.03769)
Comments:
          9 pages, Accepted at EMNLP 24, for more details see this http URL

- **What's New**: 이번 연구에서는 다양한 언어와 문화에 대한 주관적 감정을 반영한 새로운 비전-언어 벤치마크인 ArtELingo-28을 소개합니다. 이 벤치마크는 28개 언어로 약 200,000개의 주석을 포함하고 있습니다.

- **Technical Details**: ArtELingo-28은 감정적 설명을 생성하는 머신 러닝 시스템을 구축하는 데 중점을 두며, Zero-Shot, Few-Shot 및 One-vs-All Zero-Shot의 세 가지 새로운 평가 조건에 대해 기본 성능 결과를 발표합니다. 또한, 이 연구는 문화적으로 관련된 언어 간의 전이 성능이 더 성공적임을 발견했습니다.

- **Performance Highlights**: 연구 결과, ArtELingo-28은 2000개의 이미지에 대해 200K 감정 레이블과 감정적 텍스트 설명을 수집했으며, 다양한 문화적 시각을 반영한 새로운 데이터 세트를 성공적으로 구축하였습니다. 또한, 다국어 설정에서 모델의 성능을 평가하는 데 기여했습니다.



### Number Cookbook: Number Understanding of Language Models and How to Improve I (https://arxiv.org/abs/2411.03766)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 기본적인 숫자 이해 및 처리 능력(NUPA)을 종합적으로 조사하고, LLMs의 성능을 평가하기 위한 새로운 벤치마크를 제안합니다. 저자는 17개의 다양한 숫자 관련 작업을 포함한 41개의 과제를 설계하였습니다.

- **Technical Details**: 제안한 벤치마크는 정수, 부동소수점 숫자, 분수 및 과학적 표기법 등 4개의 숫자 표현 유형과 4개의 능력 카테고리로 나뉘며, 이 작업들은 초중등 교육 커리큘럼에서 파생되었습니다. 또한, 학습된 모델에 대한 NUPA를 개선하기 위한 세 가지 접근 방식을 탐구하였습니다: 사전 학습(pretraining) 단계에서의 개선, 퀄리티 있는 파인튜닝(fine-tuning) 기법, 그리고 연쇄 사고(chain-of-thought, CoT) 기법의 활용입니다.

- **Performance Highlights**: 최신 LLM들은 일부 간단한 작업에서는 양호한 성능을 보였으나, 보다 복잡한 작업에서 성능이 급격히 저하되는 경향을 보였습니다. 잘 훈련된 모델의 경우, 단순한 파인튜닝을 통해 NUPA 성능을 상당히 개선할 수 있었지만, 전문 기술 적용 시에는 기존 성능을 넘기지 못하는 경우가 많았습니다.



### The Root Shapes the Fruit: On the Persistence of Gender-Exclusive Harms in Aligned Language Models (https://arxiv.org/abs/2411.03700)
Comments:
          Accepted to 2024 Neurips Queer in AI Workshop

- **What's New**: 본 연구는 자연어 모델이 어떻게 성별 다양성에 대한 편견을 증폭할 수 있는지를 조사하여 이전의 연구들이 소외된 그룹을 충분히 대변하지 못하는 문제를 다룹니다. 특히, 트랜스젠더 및 비바이너리 커뮤니티에 중점을 두며, 기존 편향 평가 방법론의 한계를 지적합니다.

- **Technical Details**: 우리는 12개의 LLM 모델을 평가하여 성별 다양한 편견이 사전 학습, 감독된 세부 조정(SFT), 그리고 직접 선호 최적화(DPO) 단계에서 어떻게 상호작용하는지를 분석합니다. 또한, 이러한 평가를 위해 TGNB(트랜스젠더, 비바이너리 및 기타 성별 다양성) 전용 벤치마크를 사용합니다.

- **Performance Highlights**: DPO 정렬 모델은 특히 SFT로 초기화된 모델에서 성별 비확인 언어와 낙인화를 포함한 실제 피해를 증폭할 수 있으며, 이는 종종 일반적인 편향 벤치마크에서 검출되지 않습니다. 또한, DPO 정렬 모델들은 TGNB 사회 낙인을 강하게 반영하고 있으며, 이러한 문제를 해결하기 위한 권고 사항도 제시되었습니다.



### QUILL: Quotation Generation Enhancement of Large Language Models (https://arxiv.org/abs/2411.03675)
Comments:
          17 pages, 6 figures

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 인용 생성 능력을 평가하고 개선하기 위한 체계적인 연구를 다루고 있습니다. 특히 기존 LLM들이 인용 생성에서 겪는 문제인 'Quotation Hallucination' 현상을 해결하기 위한 새로운 기준과 기법을 제시합니다.

- **Technical Details**: QUILL(QUotation GeneratIon enhancement framework)이란 새로운 프레임워크를 통해 인용 생성 작업의 자동 평가 시스템을 구축하고, 이를 기반으로 32,022개의 인용문이 포함된 이중언어 지식 베이스를 설계했습니다. 또한, 5가지 기준(Quotation Authenticity, Quotation Credibility, Semantic Matching, Semantic Fluency, Quotation Novelty)에 맞춘 자동 평가 메트릭스를 개발했습니다.

- **Performance Highlights**: 개발한 지식 베이스와 재랭킹 메트릭스는 기존 LLM의 인용 생성 성능을 유의미하게 향상시키며, 연구 결과는 인간의 선호도와 높은 상관관계를 보였습니다. 이 체계적인 접근은 LLM의 인용 생성 능력을 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### Evaluating Moral Beliefs across LLMs through a Pluralistic Framework (https://arxiv.org/abs/2411.03665)
- **What's New**: 본 연구는 네 가지 주요 대형 언어 모델의 도덕 신념(moral beliefs)을 평가하기 위한 세 가지 모듈 프레임워크를 도입합니다. 이를 통해 언어 모델의 도덕적 결정 과정을 평가하고, 중국 대학생의 도덕적 선택에 대한 한국어 번역 결과를 비교합니다.

- **Technical Details**: 연구에서는 472개의 도덕적 선택 시나리오를 중국어로 구축하고, 이를 통해 언어 모델의 도덕 원칙 선호도를 파악합니다. 또한, Best-Worst Scaling(BWS) 및 Iterative Luce Spectral Ranking을 통해 도덕 원칙의 순위를 도출하고, 도덕적 논쟁(moral debates)을 통해 언어 모델의 도덕적 선택의 확고함을 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, ChatGPT와 Gemini는 중국 대학생 샘플의 도덕적 결정과 유사하게 나타나며, 개인적 도덕 신념에 강한 경향이 확인되었습니다. 반면에 Ernie와 ChatGLM 같은 중국 모델은 집단적 도덕 신념을 지향하며 도덕적 선택에서 모호성을 보였습니다. 모든 언어 모델에서 성별 편향이 내재되어 있는 것으로도 드러났습니다.



### Deploying Multi-task Online Server with Large Language Mod (https://arxiv.org/abs/2411.03644)
Comments:
          COLING2025 under submission

- **What's New**: 본 연구에서는 대규모 언어 모델(large language models)에서 수행할 수 있는 다중 작업 학습(multi-task learning) 프레임워크를 제안합니다. 이 프레임워크는 세 단계로 구성되어 있으며, 각각의 단계는 작업 필터링(task filtering), 고자원 작업에 대한 미세 조정(fine-tuning), 그리고 모든 작업에 대한 미세 조정을 포함합니다.

- **Technical Details**: 이 연구에서는 데이터 불균형(data imbalance) 문제와 작업 이질성(task heterogeneity) 문제를 해결하기 위해 각각의 작업을 필터링하고, 고자원 작업에서 미세 조정한 후 모두를 혼합하여 미세 조정하는 세단계 방법론을 개발하였습니다. 이를 통해 서로 다른 작업 간의 부정적 이전(negative transfer)을 방지하고 자원 낭비를 줄이는 결과를 가져왔습니다.

- **Performance Highlights**: 우리의 접근법은 단일작업(single-task) 방법과 비슷한 성능을 내면서도 서빙 비용을 최대 90.9%까지 줄일 수 있음을 보여주었으며, 여러 개의 벤치마크를 통해 입증된 바 있습니다.



### From Medprompt to o1: Exploration of Run-Time Strategies for Medical Challenge Problems and Beyond (https://arxiv.org/abs/2411.03590)
Comments:
          25 pages

- **What's New**: Medprompt과 OpenAI의 o1-preview 모델을 통해 런타임(작동시간) 추론 전략이 대규모 언어 모델(LLMs)의 성능을 향상시키는 방법이 제시되었습니다. 특히 o1-preview 모델은 복잡한 의료 문제에서 우수한 성능을 입증하며, 기존의 프롬프트 기법 없이도 GPT-4에 비해 더욱 뛰어난 결과를 보였습니다.

- **Technical Details**: o1-preview 모델은 체계적인 Chain-of-Thought (CoT) 추론을 통합하여, 런타임에서의 효율성을 향상시켜 주목받고 있습니다. Medprompt와 통합하여, LLM의 사고 능력을 극대화하고, 특히 의학 분야에서의 오류율을 약 50% 감소시키는 효과가 있었습니다.

- **Performance Highlights**: o1-preview 모델은 여러 의료 벤치마크에서 GPT-4(Medprompt 활용)보다 월등한 성능을 보여주었으며, 의학적 도전 과제에 대한 새로운 벤치마크 필요성을 강조합니다. 반면, GPT-4o 모델은 비용 대비 효율성이 뛰어난 선택지로 여전히 중요성을 지니고 있습니다.



### The American Sign Language Knowledge Graph: Infusing ASL Models with Linguistic Knowledg (https://arxiv.org/abs/2411.03568)
- **What's New**: 본 연구에서는 American Sign Language (ASL)용 지식 그래프인 ASLKG를 소개하고, 이를 통해 인간의 수어 모양 인식 및 번역 작업의 정확성을 향상시키기 위한 신경 기호 모델을 훈련시키는 방법을 제안합니다. 이 그래프는 12개의 전문가 언어 지식 출처로부터 수집된 지식으로 구성되어 있습니다.

- **Technical Details**: ASLKG는 5802개의 ASL 신호에 대한 7171개의 언어적 사실을 포함하고 있습니다. 이 연구는 신경 기호적 방법(neuro-symbolic methods)을 활용하여 ASL 신호 인식(isolated sign recognition), 알려지지 않은 신호의 의미 특징 예측, YouTube-ASL 비디오의 주제 분류와 같은 세 가지 다운스트림 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 단독 신호 인식에서 91%, 알려지지 않은 신호의 의미 특징 예측에서 14%, YouTube-ASL 비디오 주제 분류에서 36%의 정확도를 달성했습니다. 이러한 성과는 ASLKG를 활용함으로써 가능해졌습니다.



### Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers' Essays (https://arxiv.org/abs/2411.03550)
Comments:
          To appear in main of Conference on Empirical Methods in Natural Language Processing; EMNLP 2024

- **What's New**: 이 연구는 비원어민(Second Language, L2) 학습자들이 L1 배경에 따라 정보를 어떻게 분배하는지를 조사하였습니다. 특히 L2 능숙도에 따라 제작된 에세이의 정보 전달에서의 차이를 분석했습니다.

- **Technical Details**: 연구는 서프라이절(surprisal), 엔트로피율(constancy of entropy rate) 등의 정보를 사용하는 메트릭스를 기반으로 진행되었습니다. 또한, TOEFL11 데이터베이스와 ICNALE 코퍼스를 사용하여 L1 배경에 따른 L2 정보 분배의 변동성을 분석했습니다.

- **Performance Highlights**: 분석 결과, 높은 L2 능숙도를 가진 작가들은 정보의 불확실성을 줄이며 여전히 유용한 내용을 전달할 수 있는 것으로 나타났습니다. 그러나 정보 분배의 균일성(uniformity of information distribution)은 L2 화자 그룹 간의 변동성이 적어 L2 에세이 작성에서 보편적인 특성을 나타냅니다.



### Exploring the Benefits of Domain-Pretraining of Generative Large Language Models for Chemistry (https://arxiv.org/abs/2411.03542)
- **What's New**: 이 논문에서는 화학 분야에 특화된 AISLE(Scientific Literature에서 AI) 모델을 도입하고, 일반적인 언어 모델과 비교하여 전이 학습(pre-training) 및 파인튜닝(fine-tuning)을 통해 이 모델이 보여주는 성능 향상을 분석합니다.

- **Technical Details**: 이 연구는 과학적 텍스트로 모델을 사전 학습(pre-training)하는 이점을 탐구하며, 일반 목적의 대형 언어 모델(general-purpose large language models)과의 성능을 비교합니다. 연구에서는 'zero-shot' 및 'few-shot' 조건에서 도메인 적응 모델을 평가합니다.

- **Performance Highlights**: 실험 결과, 화학 특정 작업에서 AISLE 모델은 기존 모델들보다 우수한 성능을 보이며, 추가적인 지시 기반(fine-tuning using instruction) 조정이 필요한 모든 작업에 대해 효율적인 성능을 입증했습니다.



### Mitigating Metric Bias in Minimum Bayes Risk Decoding (https://arxiv.org/abs/2411.03524)
Comments:
          To appear at WMT2024

- **What's New**: 이번 연구에서는 Minimum Bayes Risk (MBR) 디코딩 과정에서 발생하는 metric bias 문제를 조사하였습니다. MBR 디코딩이 특정 utility metric에 따라 높은 점수를 얻도록 번역 결과를 생성하는 과정에서, 해당 metric을 디코딩과 평가에 모두 사용하는 것이 불가능하다는 점을 확인하였습니다. 주된 발견은 MBR/QE 디코딩이 인간 평가보다 품질을 과대 평가함을 보여주며, utility metric의 앙상블(ensemble) 사용을 통해 이러한 bias 문제를 완화할 수 있다는 것입니다.

- **Technical Details**: MBR 디코딩은 n개의 후보 번역을 샘플링하여, reference 기반의 utility metric을 계산하여 최상의 번역 후보를 선택합니다. MBR/QE 디코딩은 다양한 utility metrics를 통해 수행되며, 간단한 utility metric 사용 시 자동 평가 지표는 눈에 띄는 향상을 보이나, 실제 인간 평가에서는 greedy 디코딩보다 성능이 떨어지는 경우가 많습니다. 이를 해결하기 위해 MBR 디코딩에 여러 utility metrics의 앙상블을 사용하는 전략을 제안하였습니다.

- **Performance Highlights**: 연구 결과, MBR 디코딩을 여러 utility metrics의 앙상블로 수행할 때, 단일 utility metric를 사용할 때보다 더 우수한 번역 품질을 제공하였습니다. 이에 대한 인간 평가 결과가 이를 뒷받침하며, MBR 디코딩의 전반적인 성능을 향상시키는 것으로 나타났습니다.



### Change Is the Only Constant: Dynamic LLM Slicing based on Layer Redundancy (https://arxiv.org/abs/2411.03513)
Comments:
          Accepted at EMNLP Findings 2024

- **What's New**: 본 논문에서는 Large Language Models(LLMs)에서 동적 레이어 전용 가지치기(dynamic layer-specific pruning)를 통해 새로운 모델 압축 방법을 소개하며, 기존의 SliceGPT의 방법론을 발전시켰습니다. 우리는 변동폭이 있는 동적 슬라이싱(dynamic slicing)으로 전환하여, 각 레이어가 입력을 얼마나 변화시키는지를 평가하는 새로운 Layer Redundancy (LR) 점수를 활용합니다.

- **Technical Details**: 제안된 방법은 각 레이어의 중요도에 따라 가지치기 정도를 조절하는 동적 가지치기(dynamic pruning) 방법으로, LR 점수를 기반으로 개별 레이어의 중복성을 평가했습니다. 이를 통해 계산 효율성을 극대화하면서도 모델의 성능 저하를 최소화하도록 설계되었습니다.

- **Performance Highlights**: Llama3-8B 및 Mistral-7B 모델을 사용한 광범위한 실험 결과, 우리의 방법이 SliceGPT에 비해 최대 5%의 성능 향상을 보였으며, 여러 벤치마크에서 나타난 당혹도(perplexity)는 최대 7% 감소했습니다. 이는 우리의 동적 슬라이싱 접근 방식이 기존의 상수 슬라이싱 방법보다 효율적임을 입증합니다.



### Uncertainty Quantification for Clinical Outcome Predictions with (Large) Language Models (https://arxiv.org/abs/2411.03497)
- **What's New**: 이 논문에서는 의료 분야에 활용되는 언어 모델(언어 모델 - LM)의 불확실성 정량화(uncertainty quantification) 방법을 제안하고, 이를 통해 전자 건강 기록(EHR) 예측 작업에서의 성능을 향상시킵니다. 특히, 화이트박스(white-box) 및 블랙박스(black-box) 설정에서의 불확실성 감소를 강조합니다.

- **Technical Details**: 화이트박스 모델에서는 모델 매개변수 및 출력 로짓에 접근 가능하여 불확실성을 정량화하고, 새로운 멀티 태스킹(multi-tasking) 및 앙상블(ensemble) 방법을 통해 이를 효과적으로 줄입니다. 블랙박스 모델은 접근이 제한된 현대 대형 언어 모델(GPT-4 포함)을 대상으로 하여, 신규 접근 방식을 통해 답변 분포를 분석하여 불확실성을 측정합니다.

- **Performance Highlights**: 6,000명 이상의 환자에 대한 장기적 임상 데이터 사용을 통해, 제안된 방법이 다양한 시나리오에서 불확실성을 유의미하게 감소시켰음을 보여줍니다. 이는 화이트박스 및 블랙박스 상황에서 모델의 투명성을 높이고, 신뢰할 수 있는 AI 의료를 발전시키는 데 기여함을 나타냅니다.



### Automatic Generation of Question Hints for Mathematics Problems using Large Language Models in Educational Technology (https://arxiv.org/abs/2411.03495)
Comments:
          Accepted at NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess)

- **What's New**: 이번 연구는 Intelligent Tutoring Systems (ITSs) 내에서 Large Language Models (LLMs)를 사용하여 수학 문제를 해결하는 학생들에게 효과적인 힌트를 생성하는 방법을 모색하였습니다. 특히, LLMs (GPT-4o와 Llama-3-8B-instruct)의教师 역할과 GPT-3.5-turbo 또는 Mistral-7B-instruct-v0.3 기반의 모의 학생 역할을 수행하는 것이 주요한 발전으로 평가됩니다.

- **Technical Details**: 이 연구에서는 다양한 방법으로 LLM을 활용하여 1) 고등학생을 위한 수학 연습에서 발생하는 오류 패턴 인식, 2) GPT-4o를 활용한 힌트 생성 방법의 효과 평가, 3) Llama-3-8B-Instruct를 शिक्षक으로 하여 최적의 프롬프트를 테스트하여 오류 수정 능력을 향상시키는 방법을 조사했습니다. 특히, 힌트 생성에서 온도 설정(temperature)과 관련된 새로운 발견을 보고하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4o가 생성한 힌트는 특정 오류에 맞춘 프롬프트와 일반적인 수학 오류에 기초한 프롬프트에서 가장 효과적이었습니다. Llama-3-8B-Instruct는 전반적인 성능에서 GPT-4o를 초과하며, 특히 GPT-3.5-turbo 모델이 힌트를 받은 후 문제 해결 및 응답 수정 능력이 크게 향상되었습니다.



### Usefulness of LLMs as an Author Checklist Assistant for Scientific Papers: NeurIPS'24 Experimen (https://arxiv.org/abs/2411.03417)
- **What's New**: 대규모 언어 모델(LLMs)의 사용이 과학적 동료 검토에 도움이 될 수 있는 가능성을 탐구하는 연구에서, 2024 Neural Information Processing Systems(NeurIPS) 학회에서 LLM 기반 체크리스트 어시스턴트를 사용하여 제출된 234개의 논문의 검토를 진행한 결과, 저자들이 체험한 유용성이 상당히 긍정적이라는 결과를 도출했습니다.

- **Technical Details**: 이 연구에서는 LLM을 활용하여 NeurIPS의 저자 체크리스트에 대한 응답을 검토하고, 제출물이 학회의 요구사항을 충족시키기 위해 필요한 개선점을 제안합니다. 학회의 체크리스트는 연구 재현성, 투명성, 윤리적 연구 기준을 충족하는지 점검하기 위해 저자들이 숙지해야 하는 일련의 질문으로 구성됩니다. LLM은 15개의 질문에 대해 저자에게 4-6개의 구체적인 피드백 포인트를 제공합니다.

- **Performance Highlights**: 응답한 저자들 중 70% 이상이 LLM 기반 어시스턴트를 유용하게 평가했으며, 70%는 피드백에 따라 논문이나 체크리스트 응답을 수정할 계획이라고 밝혔습니다. 그런 와중에도 저자들은 LLM의 정확성 문제(20/52명 응답자)가 존재했음을 지적하였으며, LLM이 너무 엄격하다는 피드백도 있었습니다. 또한 LLM 도구의 조작성 가능성을 보여주면서, 저자들이 작성한 허위 정당화로 체크리스트 어시스턴트의 결과를 조작할 수 있는 가능성을 내비쳤습니다.



### SAUCE: Synchronous and Asynchronous User-Customizable Environment for Multi-Agent LLM Interaction (https://arxiv.org/abs/2411.03397)
Comments:
this https URL

- **What's New**: SAUCE는 다양한 참여자가 있는 그룹 설정에서의 복잡한 사회적 상호작용을 탐구하기 위해 제작된 사용자 지정 가능 Python 플랫폼입니다. 이 플랫폼은 연구자가 다양한 LLMs를 쉽게 조합하여 사용자 선택 주제에 대한 토론을 진행할 수 있게 해줍니다.

- **Technical Details**: SAUCE는 비동기 통신(asynchronous communication)을 지원하여 모델이 언제 발언할지 결정할 수 있는 기능을 포함하고 있습니다. 이 플랫폼은 LLM의 상호작용을 동기적(synchronous) 및 비동기적 환경 모두에서 연구할 수 있도록 고안되었습니다. 실험은 JSON 파일로 설정하며, 이는 다양한 모델 소스(HuggingFace, API 등)와 통합이 가능합니다.

- **Performance Highlights**: 첫 번째 실험에서는 SAUCE를 사용하여 서로 다른 정치 이념을 가진 에이전트 간의 정치 토론을 시뮬레이션하였으며, LLM 에이전트가 사회적 편향에 순응하는 경향을 발견했습니다. 두 번째 실험에서는 '트롤리 문제'에 대한 비동기적 철학적 토론을 시뮬레이션하여 에이전트가 상황과 시간 제약에 따라 발언 방식을 조정하는 모습을 보여주었습니다.



### A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness (https://arxiv.org/abs/2411.03350)
Comments:
          76 pages, 26 figures, 14 tables

- **What's New**: 본 논문은 Small Language Models (SLMs)의 정의, 획득, 응용, 향상 및 신뢰성 문제에 대한 포괄적인 조사를 수행하고 있습니다. 이러한 조사는 LLMs의 한계를 극복하기 위해 성장하는 SLMs의 필요성과 관련이 있습니다.

- **Technical Details**: SLMs는 적은 추론 지연(inference latency), 비용 효율성(cost-effectiveness), 효율적인 개발(efficient development), 커스터마이징(customization) 및 적응(adaptability)에서 우수합니다. SLMs의 정의는 그들이 수행하는 전문적인 작업의 능력과 자원 제약이 있는 환경에서의 적합성을 기준으로 설정합니다. 또한, 모델 및 방법에 대한 분류학(taxonomy)과 각 범주에 대한 일반적인 프레임워크를 개발하여 SLMs를 효과적으로 활용하는 방법을 제안합니다.

- **Performance Highlights**: SLMs는 LLMs에 비해 로컬 데이터 처리를 위한 개인 정보 보호, 효율성을 위한 최소한의 추론 지연, 경량 파인 튜닝을 통한 도메인 지식 습득에 최적화된 응용 프로그램에 적합한 성능을 보여줍니다.



### How Transformers Solve Propositional Logic Problems: A Mechanistic Analysis (https://arxiv.org/abs/2411.04105)
- **What's New**: 이번 연구에서는 작은 Transformer 모델이 복잡한 논리 문제를 해결하는 내부 메커니즘과 이에 필요한 '계획' 및 '추론' 회로를 탐색합니다. 또한 Mistral 7B 모델을 통해 보다 큰 모델의 내부 구성요소를 분석하여 논리 문제 해결에 필요한 회로를 식별합니다.

- **Technical Details**: 연구는 Synthetic propositional logic 문제를 중심으로 진행되며, 두 가지 주요 실험을 포함합니다: (1) 작은 Transformer 모델에 대한 실험과 (2) 사전 훈련된 LLM인 Mistral-7B에 대한 실험. 활성화 패칭 (activation patching)을 통해 Mistral-7B의 특정 주목 헤드(attention heads) 역할을 분석하여 문제를 해결하는 데 필요한 회로를 밝혀냅니다.

- **Performance Highlights**: 작은 Transformer 모델은 다양한 추론 문제를 해결하는 데 있어 정보 흐름을 불균형하게 조정하는 '라우팅 임베딩(routing embeddings)'을 사용하고, Mistral-7B 모델에서는 각 규칙과 사실을 처리하는 특수화된 주목 헤드의 역할을 발견하였습니다. 이 연구는 작은 및 큰 Transformer의 새로운 측면을 체계적으로 드러내며, 모델이 어떻게 계획하고 추론하는지를 지속적으로 탐구합니다.



### Interactions Across Blocks in Post-Training Quantization of Large Language Models (https://arxiv.org/abs/2411.03934)
- **What's New**: 이 연구에서는 Post-training quantization을 통해 대규모 언어 모델의 양자화 성능을 개선하기 위한 두 가지 다중 블록 파인튜닝 전략을 제안하고 비교했습니다.

- **Technical Details**: 기존의 양자화 방법은 서로 독립적이라고 가정하며 하위 구조의 지식을 무시했습니다. 이에 반해, 첫 번째 전략은 양자화된 여러 블록을 공동 최적화하여 블록 간의 상관관계를 포착합니다. 두 번째 전략은 후속 블록의 정보를 활용하여 다운스트림(pre-activation)의 오류를 최소화합니다.

- **Performance Highlights**: 모델에 따라 이러한 방법의 효과가 달라졌으며, 일부 모델에서는 아무런 영향을 미치지 않았지만 다른 모델에서는 상당한 성과 향상을 보였습니다.



### Lexicalization Is All You Need: Examining the Impact of Lexical Knowledge in a Compositional QALD System (https://arxiv.org/abs/2411.03906)
Comments:
          24th International Conference on Knowledge Engineering and Knowledge Management (EKAW 2024), November 26-28, 2024, Amsterdam, The Netherlands

- **What's New**: 이번 논문에서는 Linked Data (QALD)에서 질문 응답에 대한 어휘화(lexicalization)의 영향을 조사하였습니다. 자연어 질문을 SPARQL 쿼리로 해석하는 과정에서 어휘적 간격(lexical gap)을 메우는 것이 주요 도전 과제라는 점을 강조하고, 어휘화가 Q&A 시스템의 성능을 크게 향상시킨다는 주장을 펼쳤습니다.

- **Technical Details**: 우리는 명시적 어휘 지식을 활용하는 조합형 질문 응답 시스템(compositional QA system)을 제안합니다. 이 시스템은 SPARQL 쿼리의 의미를 추론하기 위해 어휘적 지식을 조합적으로 활용하며, QALD-9 데이터셋에서 기존 Q&A 시스템보다 35.8% 향상된 마이크로 F1 점수를 기록하는 성과를 보여주었습니다.

- **Performance Highlights**: LLM(대형 언어 모델)은 어휘적 지식을 활용하는 데 한계가 있어, 어휘적 지식 없이 활용할 때와 큰 차이가 없음을 보여줍니다. 새로운 접근법은 QALD 연구에 대한 새로운 방향을 제시하며, 어휘화와 조합성의 중요성을 강조합니다.



### From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning (https://arxiv.org/abs/2411.03817)
- **What's New**: 이 논문에서는 StepAgent를 소개합니다. 이는 단계별 보상(Step-wise reward)을 활용하여 에이전트의 강화학습(Reinforcement Learning) 프로세스를 최적화합니다.

- **Technical Details**: StepAgent는 초보자에서 전문가로의 이론(Novice-to-Expert Theory)을 바탕으로 하여, 전문가와 에이전트의 행동을 비교하여 세분화된 최적화를 위한 중간 보상을 자동으로 생성합니다. 또한, 암묵적 보상(Implicit Reward) 및 역 강화학습(Inverse Reinforcement Learning) 기법을 제안하여 에이전트의 반성과 정책 조정을 촉진합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, StepAgent는 기존의 기준 방법들보다 더 우수한 성능을 보였습니다.



### MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogu (https://arxiv.org/abs/2411.03814)
- **What's New**: 이번 연구에서는 Multi-round Dialogue Jailbreak ( MRJ ) 에이전트를 제안하여, LLMs (Large Language Models) 에 대한 멀티 라운드 대화 공격의 위험을 식별하고 완화하기 위한 새로운 전략을 소개합니다.

- **Technical Details**: 새로운 공격 메커니즘은 Heuristic Search (휴리스틱 검색) 과정을 기반으로 하며, 악의적인 쿼리에서 시작해 점진적으로 민감한 주제로 나아가도록 설계되었습니다. 이 과정에서는 정보 기반 제어 전략과 심리 유도 전략이 사용됩니다.

- **Performance Highlights**: 제안된 방법은 기존의 단일 및 다중 라운드 공격 방법보다 우수한 공격 성공률을 기록하며, Closed-source 및 Open-source 모델에서 모두 높은 효과를 나타냈습니다. 이를 통해 다양한 모델 및 시나리오에 적용 가능한 보다 일반적인 공격 전략을 선보였습니다.



### Long Context RAG Performance of Large Language Models (https://arxiv.org/abs/2411.03538)
Comments:
          2024 NeurIPS workshop on Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning

- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG)의 성능을 LLM의 컨텍스트 길이 증가에 따라 평가한 포괄적인 연구를 발표합니다. 특히, 2000부터 128,000 토큰까지의 컨텍스트 길이에서 RAG 작업을 수행하면서 최상위 LLM의 성능을 분석합니다.

- **Technical Details**: 연구에서는 총 20개의 인기 있는 오픈 소스 및 상업적 LLM을 사용하여 Databricks DocsQA, FinanceBench 및 Natural Questions라는 세 가지 도메인 특정 데이터세트를 대상으로 RAG 워크플로우를 실행했습니다. LLM의 컨텍스트 길이를 2000에서 128000 토큰, 경우에 따라 최대 200만 토큰까지 변화시키며 성능을 평가했습니다.

- **Performance Highlights**: 가장 최근의 최첨단 LLM 중 일부만이 64k 토큰을 초과하는 긴 컨텍스트에서 일관된 정확도를 유지할 수 있음을 발견했습니다. 전반적인 결과는 RAG 성능이 증가하는 컨텍스트 길이에 비례해 증가하지 않는다는 것을 보여주며, 일부 모델은 긴 컨텍스트 상황에서 고유한 실패 양상을 충족하기도 했습니다.



### LASER: Attention with Exponential Transformation (https://arxiv.org/abs/2411.03493)
Comments:
          15 pages, under review in ICLR 2025

- **What's New**: 이 논문은 기존의 attention 메커니즘에서 발생하는 미세한 기울기 전달 문제를 해결하기 위해 LASER라는 새로운 attention 방식을 제안합니다. LASER는 더 큰 기울기 신호를 허용하며, 기존의 attention 구현에 아주 작은 수정만으로 적용할 수 있습니다.

- **Technical Details**: LASER는 LogArithm of Summed Exponentials of Representations의 약어로, 입력의 지수 변환을 진행하여 attention을 수행하는 구조입니다. 본 방법은 Log-Weighted-Sum-Exp라는 새로운 기법을 도입하여 큰 모델(최대 22억 개의 파라미터를 가진 모델)에서도 수치적 오버플로우 문제를 해결합니다.

- **Performance Highlights**: LASER를 적용한 결과, Vision Transformer(이미지 분류의 경우)에서 4.67%의 정확도 향상, Conformer(음성 인식의 경우)에서 2.25%의 오류율 감소, BERT 모델에서 0.93%의 잘못된 예측 비율 감소를 달성했습니다. 여러 downstream 작업에서 LASER는 평균 1%의 정확도 향상을 보였고, 최대 3.38%의 정확도 향상을 기록했습니다.



### LLM Generated Distribution-Based Prediction of US Electoral Results, Part I (https://arxiv.org/abs/2411.03486)
Comments:
          17 pages, 10 Figures, Pre-print

- **What's New**: 본 논문에서는 Large Language Models (LLMs)을 예측 도구로 사용하는 새로운 접근법인 distribution-based prediction을 소개합니다. 이 방법은 출력 토큰 확률을 모델이 학습한 세계를 나타내는 분포로 해석하여 알고리즘적 신뢰성을 분석하는 대안을 제공합니다.

- **Technical Details**: distribution-based prediction 접근법은 개별 인물 시뮬레이션을 우회하여 LLM을 비개인 예측 모델로 활용합니다. 주어진 주(state) 내에서 각 후보에 대한 투표 비율을 예측하기 위해 모델에 직접 프롬프트를 제공하고, 출력 확률을 해당 모델이 내재한 지식을 나타내는 분포로 처리합니다. 이는 후보자별로 주마다 유권자 비율의 분포를 생성하는 과정으로 진행됩니다.

- **Performance Highlights**: 이 방법을 통해 미국 대선에서 후보별 주 투표 비율을 예측하며, LLM의 정확성과 알고리즘적 신뢰성을 평가할 수 있음을 보여줍니다. 또한, 이 개념은 다양한 도메인에서 LLM 기반 예측의 신뢰성과 투명성을 높이는 데 상당한 의미를 가집니다.



### MetRex: A Benchmark for Verilog Code Metric Reasoning Using LLMs (https://arxiv.org/abs/2411.03471)
- **What's New**: 본 논문에서는 HDL 설계의 post-synthesis 메트릭 추론 및 추정에 대한 LLM(대형 언어 모델)의 사용을 탐색합니다. 이를 위해 25,868개의 Verilog HDL 설계와 해당 post-synthesis 메트릭을 포함한 MetRex 데이터셋을 소개합니다.

- **Technical Details**: MetRex는 area, delay, static power 메트릭을 포함하는 대규모 데이터셋으로, LLM이 이러한 메트릭에 대해 더 깊은 이해를 가질 수 있도록 Chain of Thought (CoT) 템플릿을 적용했습니다. 우리는 Supervised Fine-Tuning (SFT) 기법을 사용하여 메트릭 추론 능력을 평균 37.0%, 25.3%, 25.7% 개선했습니다.

- **Performance Highlights**: SFT가 우리 벤치마크에서 성능을 개선하는 동안, LLM은 최신 회귀 모델들에 비해 5% 오차 범위 내에서 17.4% 더 많은 정확한 post-synthesis 예측을 제공합니다. 또한, 전처리 과정이 필요 없어 1.7배 빠른 속도를 자랑합니다.



### Solving Trojan Detection Competitions with Linear Weight Classification (https://arxiv.org/abs/2411.03445)
Comments:
          9 pages, 4 Figures

- **What's New**: 이 논문에서는 트로이안 백도어를 탐지하기 위한 새로운 방법을 제안했습니다. 이 접근법은 다양한 데이터셋과 도메인에서 높은 성능을 보입니다.

- **Technical Details**: 제안된 탐지기는 여러 개의 모델 가중치에 대한 이진 분류기를 학습하여 얻어지며, 주요 전처리 단계를 통해 성능을 개선합니다. 전 처리 단계에는 특성 선택, 표준화, 참조 모델 가중치 빼기, 모델 정렬 등이 포함됩니다. 이 기법은 가중치 분석(weight analysis) 탐지에 해당하며, 트리거에 대한 사전 지식 없이도 적용 가능합니다.

- **Performance Highlights**: 본 알고리즘은 Trojan Detection Challenge(TDC22)와 IARPA/NIST TrojAI 프로그램의 다양한 벤치마크에서 평가되었으며, 모델의 정밀한 분류를 통해 청정(clean) 모델과 오염된(poisoned) 모델 간의 구분을 효과적으로 수행했습니다.



### Exploring Large Language Models for Specialist-level Oncology Car (https://arxiv.org/abs/2411.03395)
- **What's New**: 이 연구에서는 AMIE라는 연구 대화형 진단 AI 시스템의 유방 종양학(oncology) 분야에서의 성능을 조사했습니다. 기존의 특정 도메인에 대한 세부 조정 없이도 임상적 의사 결정에 도움을 줄 수 있는 잠재력을 가지고 있음을 보여줍니다.

- **Technical Details**: 연구팀은 50개의 합성 유방암 사례(vignettes)를 제작하고, 관리 계획의 질, 안전성 및 치료법 권장 사항을 평가하는 포괄적 임상 평가 기준(rubric)을 개발했습니다. AMIE는 웹 검색(web search) 기능과 다단계 자기 비판(self-critique) 파이프라인을 통해 성능을 향상시켰습니다.

- **Performance Highlights**: AMIE는 내부 의학 수련의(internal medicine trainees) 및 초기 종양 전문의(oncology fellows) 보다 우수한 성능을 보였지만, 경험이 풍부한 종양 전문의(attending oncologists)와 비교했을 때 전반적으로 열세였습니다. 이는 AMIE가 향후 보완이 필요하다는 것을 의미합니다.



### RuAG: Learned-rule-augmented Generation for Large Language Models (https://arxiv.org/abs/2411.03349)
- **What's New**: 본 논문에서는 LLM의 추론 능력을 향상시키기 위해 대량의 offline 데이터를 해석 가능한 1차 논리 규칙으로 자동 변환하고 이를 LLM에 주입하는 새로운 프레임워크인 RuAG를 제안합니다. RuAG는 LLM의 상식에 기반하여 목표 및 본체 술어를 자동 정의하고, Monte Carlo Tree Search를 통해 데이터에서 논리 규칙을 효율적으로 발견합니다.

- **Technical Details**: RuAG는 세 가지 주요 단계로 구성됩니다. 첫째, LLM 기반의 논리 규칙 검색 공식을 통해 LLM이 상식에 기반하여 목표 술어와 본체 술어를 정의합니다. 둘째, MCTS를 이용하여 논리 규칙 검색을 수행하며, 복합적인 검색 공간을 효율적으로 탐색하여 구조화된 1차 논리 규칙을 생성합니다. 셋째, 생성된 논리 규칙을 자연어로 변환하여 LLM의 프롬프트에 주입합니다.

- **Performance Highlights**: RuAG는 자연어 처리, 시계열, 의사 결정 및 산업 과제를 포함한 다양한 공개 및 민간 산업 과제에서 LLM의 능력을 향상시키는 데 효과적임을 입증했습니다. 이 프레임워크는 SFT, ICL, RAG 및 KG 기반 방법과 같은 기존 방법의 한계를 극복하여, LLM의 추론 및 작업 성능을 최소한의 수작업 개입으로 개선합니다.



### What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks (https://arxiv.org/abs/2411.03343)
- **What's New**: 이번 연구에서는 LLM (large language models)에서 'jailbreaks'의 안전성과 신뢰성 연구의 핵심 요소로서, 성공적인 jailbreak에 기여하는 프롬프트의 특징을 비교 분석하였습니다.

- **Technical Details**: 연구팀은 35가지 공격 방법으로부터 수집된 10,800개의 jailbreak 시도를 포함하는 데이터셋을 소개하였으며, 선형(linear)과 비선형(nonlinear) 방법을 비교하여 프롬프트의 성공적인 jailbreak을 지원하는 특징을 조사했습니다. 특히 비선형 프로브(non-linear probes)가 LLM을 기계적으로 jailbreak하는 데 사용될 수 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과, 특정 비선형 특징을 통해 성공적인 프롬프트와 실패한 프롬프트를 높은 정확도로 구별할 수 있었지만, 보유된 공격 방법에 대해서는 성능이 떨어지는 경향이 있었습니다. 또한, Gemma-7B-IT를 jailbreak 하는 데 있어 연구진의 접근 방식이 기존의 35가지 기술보다 더 신뢰할 수 있는 결과를 보였습니다.



### Unlocking the Archives: Using Large Language Models to Transcribe Handwritten Historical Documents (https://arxiv.org/abs/2411.03340)
Comments:
          29 Pages, 11 Tables, 2 Figures

- **What's New**: 이 연구는 Large Language Models (LLMs)이 특화된 Handwritten Text Recognition (HTR) 소프트웨어보다 역사적인 손글씨 문서를 훨씬 더 높은 정확도로 기록할 수 있음을 보여줍니다. 이와 함께 개발한 오픈 소스 소프트웨어 도구인 Transcription Pearl은 상업적으로 사용 가능한 다양한 LLM을 활용하여 효율적으로 손글씨 문서를 자동으로 기록하고 수정합니다.

- **Technical Details**: Transcription Pearl은 18세기와 19세기 영어 손글씨 문서의 다양한 데이터셋을 사용하여 테스트 하였으며, LLM은 Character Error Rates (CER) 5.7-7%와 Word Error Rates (WER) 8.9-15.9%를 기록하여, 기존 HTR 소프트웨어인 Transkribus보다 각각 14% 및 32% 개선되었습니다. LLM은 타임라인을 50배 더 빠르고, HTR 프로그램의 1/50의 비용으로 작업을 완료할 수 있게 해줍니다.

- **Performance Highlights**: LLMs는 전통적인 HTR 소프트웨어와 LLM들이 생성한 기록들을 수정함으로써 인력 수준의 정확성에 도달하였으며, CER은 1.8%까지, WER은 3.5%에 달했습니다. 기록 과정은 대략 84-93%의 정확도로 수행되었으며, 이는 대부분의 일상적인 사용 예제에 적합합니다. 결과적으로 LLM을 활용한 HTR은 역사적 손글씨 문서의 대량 기록 프로젝트 및 개별 기록을 다루는 역사학자에게 접근 가능하고 효율적인 방법을 제시합니다.



### Will Trump Win in 2024? Predicting the US Presidential Election via Multi-step Reasoning with Large Language Models (https://arxiv.org/abs/2411.03321)
Comments:
          This research is ongoing work. Xiyang Hu and Yue Zhao are the corresponding authors

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 선거 예측 능력을 탐구하고, 그에 대한 혁신적인 접근법을 제시합니다. 특히 정치적 분석을 위한 다단계 추론 프레임워크를 제공하여 실제 데이터를 기반으로 한 예측을 향상시킵니다.

- **Technical Details**: 우리는 Sync synthetic data generation framework를 사용하여 개별 유권자의 인구통계 및 행동 프로필을 재구성하고, 2016년 및 2020년 미국 국가선거조사(ANES)의 실제 데이터를 통해 프레임워크를 검증합니다. Chain of Thought prompting에서 영감을 받아, 이 접근 방식은 정치적 맥락의 변화에 맞춰 모델을 조정하며, 인구 통계학적 정보 및 이념적 요인과 같은 다양한 요소들을 체계적으로 통합합니다.

- **Performance Highlights**: 이 모델은 2024년 미국 대통령 선거 결과를 사전에 예측하는 데 성공하여, 보이지 않는 정치적 데이터에 대해 LLMs가 어떻게 적응할 수 있는지를 증명합니다. 최종 파이프라인은 예측 정확성과 실제 결과와의 정합성 모두에서 상당한 개선을 보여줍니다.



### FactTest: Factuality Testing in Large Language Models with Finite-Sample and Distribution-Free Guarantees (https://arxiv.org/abs/2411.02603)
- **What's New**: 대형 언어 모델(LLMs)의 사실성을 통계적으로 평가하는 새로운 프레임워크인 FactTest가 도입되었습니다. 이 프레임워크는 LLM이 질문에 대한 올바른 답을 높은 확률로 제공할 수 있는지를 검증합니다.

- **Technical Details**: FactTest는 가설 검정(hypothesis testing) 문제로 사실성 테스트를 공식화하며, 사용자 지정 유의 수준에서 Type I 오류를 제어합니다. 또한, mild 조건 하에서 Type II 오류를 강하게 제어할 수 있음을 증명하였습니다. 이 프레임워크는 모델에 구애받지 않으며, 이론적 보장을 제공합니다.

- **Performance Highlights**: FactTest는 질문-답변 및 다중 선택 기준에서 광범위한 실험을 통해 환각을 효과적으로 감지할 수 있음을 보여주며, 정확도가 40% 이상 향상된 결과를 나타냈습니다. 추가 학습이나 외부 데이터 출처 없이도 모델의 성능을 크게 개선했습니다.



New uploads on arXiv(cs.IR)

### Reproducible Hybrid Time-Travel Retrieval in Evolving Corpora (https://arxiv.org/abs/2411.04051)
- **What's New**: 이 논문에서는 fast retrieval을 위한 Lucene과 versioned, timestamped index를 유지하는 column-store 기반 검색 시스템을 결합한 하이브리드 정보 검색 시스템을 제안합니다. 이 시스템은 이전에 제기된 쿼리를 재실행할 수 있게 하여 동일한 ranked list를 반환하며, 변화하는 문서 모음에 대해 시간 여행 쿼리를 가능하게 하여 원래의 ranking을 유지합니다.

- **Technical Details**: 제안하는 하이브리드 시스템의 아키텍처는 classic rank-based retrieval engine과 temporal column-store search engine을 결합합니다. 이를 통해 문서 모음에서의 전역 통계 변화에 관계없이 검색 결과의 재현성을 보장합니다. 이 시스템은 Open Source Software로 배포됩니다.

- **Performance Highlights**: 시스템은 색인 속도, 쿼리 응답 시간, 점수 정확성, 저장 오버헤드 등을 평가하였으며, changes에 따라 term statistics를 유지하는 효율적인 방법을 보여줍니다.



### Fine-Grained Guidance for Retrievers: Leveraging LLMs' Feedback in Retrieval-Augmented Generation (https://arxiv.org/abs/2411.03957)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능을 향상시키기 위한 새로운 방법인 FiGRet(fine-grained guidance for retrievers)를 제안합니다. 기존의 retriever는 LLM(large language model)과의 최적화가 부족했지만, FiGRet는 LLM의 언어 능력을 활용해 retriever 학습을 보다 효과적으로 돕습니다.

- **Technical Details**: FiGRet 프레임워크는 교육 이론에 기반하여 LLM을 '교사'로, retriever를 '학생'으로 설정하고, 최소한의 겹침이 있는 세 가지 RAG 성능 관련 학습 목표(관련성, 포괄성, 순수성)를 설정하여 retriever의 학습을 안내합니다. 또한, 이 과정에서 이중 커리큘럼 학습 전략을 채택하여 LLM과 retriever 간의 피드백 루프를 활용합니다.

- **Performance Highlights**: 실험을 통해 FiGRet 프레임워크가 다양한 LLM과 retriever 조합에서 성능 향상을 이끌어내었다는 것을 입증하였습니다. MMLU 및 오픈 도메인 QA와 같은 작업들에서도 성능 개선이 관찰되었습니다.



### Data Fusion of Synthetic Query Variants With Generative Large Language Models (https://arxiv.org/abs/2411.03881)
Comments:
          The definitive version of record was published in SIGIR-AP '24

- **What's New**: 이 논문은 정보 검색(IR) 실험에서 쿼리 변동성을 고려하는 것이 검색 효과성을 높인다는 점을 제시합니다. 특히, 여러 주제와 관련된 쿼리를 기반으로 한 랭킹 앙상블이 단일 쿼리보다 더 나은 결과를 보여줍니다.

- **Technical Details**: 우리는 generative instruction-tuned Large Language Models (LLMs)를 활용하여 합성 쿼리 변종(synthetic query variants)을 생성하고, 이 변종들을 데이터 융합(data fusion) 실험에서 사용하는 방법을 제안합니다. 여기서는 원가 효율적이며 비지도 학습(unsupervised)의 경량화된 접근법을 채택했습니다. 각 쿼리를 생성하기 위해 톱픽 파일에 대한 주제 전용 프롬프트(promotion)를 구축합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, LLM이 추가적인 맥락 정보를 제공받을 때 효과적인 쿼리를 생성합니다. 4개의 TREC 뉴스와이어 벤치마크에 대한 분석 결과, 합성 쿼리 변형을 기반으로 한 데이터 융합이 단일 쿼리를 사용하는 기준선보다 현저히 나은 성능을 보이며, 유사한 관련 피드백 방법도 초월했습니다.



### The Essence of the Essence from the Web:The Metasearch Engin (https://arxiv.org/abs/2411.03701)
Comments:
          6 pages

- **What's New**: 이 논문에서는 메타서치 엔진(Metasearch Engine)의 작동 방식을 설명하고, 전통적인 검색 엔진(traditional search engines)과의 비교 연구를 통해 메타서치 엔진의 우수성을 입증합니다.

- **Technical Details**: 메타서치 엔진은 사용자가 보낸 쿼리(query)를 여러 검색 엔진에 동시에 전송하여 결과를 수집하고 이를 종합하여 사용자에게 최상의 결과를 제공합니다. 이 과정에서 메타서치 엔진은 개별적으로 웹 페이지 데이터베이스를 소유하지 않고, 검색 엔진 회사가 관리하는 데이터베이스에 쿼리를 전송하여 결과를 받습니다.

- **Performance Highlights**: 결과적으로 메타서치 엔진은 사용자 쿼리에 대한 더 나은 결과를 제공하며, 검색 엔진 간의 성능 및 효율성을 비교할 때 메타서치 엔진이 더 효과적이라는 것을 보여줍니다.



### Advanced RAG Models with Graph Structures: Optimizing Complex Knowledge Reasoning and Text Generation (https://arxiv.org/abs/2411.03572)
- **What's New**: 이번 연구에서는 기존의 retrieval-augmented generation (RAG) 모델을 최적화하기 위해 그래프 구조를 도입하여 복잡한 지식 추론 작업에서 모델의 성능을 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: 전통적인 RAG 모델은 지식 그래프(knowledge graphs), 계층적 관계(hierarchical relationships)와 같은 복잡한 그래프 구조 정보를 처리할 때 효율성이 부족한 문제를 가지고 있습니다. 이 연구에서는 그래프 신경망(graph neural network, GNN)을 결합하여 그래프 구조 데이터를 처리하는 방안을 제안합니다. 이를 통해 모델은 엔티티 간의 복잡한 관계를 캡처할 수 있어 지식 일관성(knowledge consistency)과 추론 능력을 향상시킵니다.

- **Performance Highlights**: 자연 질문(Natural Questions, NQ) 데이터셋을 사용하여 여러 기존 생성 모델과 비교한 결과, 본 연구에서 제안한 그래프 기반의 RAG 모델이 품질, 지식 일관성 및 추론 능력 면에서 전통적인 생성 모델보다 우수한 성능을 보여주었습니다. 특히 다차원적 추론이 요구되는 작업에서 그 효과가 두드러졌습니다.



### Lexicalization Is All You Need: Examining the Impact of Lexical Knowledge in a Compositional QALD System (https://arxiv.org/abs/2411.03906)
Comments:
          24th International Conference on Knowledge Engineering and Knowledge Management (EKAW 2024), November 26-28, 2024, Amsterdam, The Netherlands

- **What's New**: 이번 논문에서는 Linked Data (QALD)에서 질문 응답에 대한 어휘화(lexicalization)의 영향을 조사하였습니다. 자연어 질문을 SPARQL 쿼리로 해석하는 과정에서 어휘적 간격(lexical gap)을 메우는 것이 주요 도전 과제라는 점을 강조하고, 어휘화가 Q&A 시스템의 성능을 크게 향상시킨다는 주장을 펼쳤습니다.

- **Technical Details**: 우리는 명시적 어휘 지식을 활용하는 조합형 질문 응답 시스템(compositional QA system)을 제안합니다. 이 시스템은 SPARQL 쿼리의 의미를 추론하기 위해 어휘적 지식을 조합적으로 활용하며, QALD-9 데이터셋에서 기존 Q&A 시스템보다 35.8% 향상된 마이크로 F1 점수를 기록하는 성과를 보여주었습니다.

- **Performance Highlights**: LLM(대형 언어 모델)은 어휘적 지식을 활용하는 데 한계가 있어, 어휘적 지식 없이 활용할 때와 큰 차이가 없음을 보여줍니다. 새로운 접근법은 QALD 연구에 대한 새로운 방향을 제시하며, 어휘화와 조합성의 중요성을 강조합니다.



### SEGMN: A Structure-Enhanced Graph Matching Network for Graph Similarity Learning (https://arxiv.org/abs/2411.03624)
- **What's New**: 본 논문에서는 그래프 유사도 계산(Graphic similarity computation, GSC)에 대한 새로운 접근법인 구조 강화 그래프 매칭 네트워크(Structure-Enhanced Graph Matching Network, SEGMN)를 제안합니다. SEGMN은 노드 간 연결 구조를 최대한 활용하여 더 정확한 유사도 점수를 산출합니다.

- **Technical Details**: SEGMN은 이중 임베딩 학습 모듈(Dual embedding learning module)과 구조 인식 매칭 모듈(Structure perception matching module)을 갖추고 있습니다. 이중 임베딩 학습 모듈은 각 노드에 인접한 엣지 표현을 통합하여 구조 향상된 표현을 생성합니다. 구조 인식 매칭 모듈은 할당 그래프(Assignment graph) 합성을 통해 교차 그래프의 구조를 강화합니다.

- **Performance Highlights**: 벤치마크 데이터 세트에 대한 실험 결과, SEGMN은 최신 GSC 방법보다 GED 회귀 작업에서 우수한 성능을 보였으며, 구조 인식 매칭 모듈은 기존 방법의 성능을 최대 25%까지 향상시킬 수 있음을 보여주었습니다.



### Automated, LLM enabled extraction of synthesis details for reticular materials from scientific literatur (https://arxiv.org/abs/2411.03484)
Comments:
          16 pages

- **What's New**: 이 연구에서는 과학 문헌에서 레티큘러(materials synthesis protocols)을 자동으로 추출하는 새로운 접근법인 지식 추출 파이프라인(Knowledge Extraction Pipeline, KEP)을 제안합니다. 이 방법은 대형 언어 모델(LLMs)을 사용하여 PDF 문서에서 화학 정보를 효과적으로 검색할 수 있는 가능성을 보여줍니다.

- **Technical Details**: KEP는 주로 네 개의 모듈로 구성됩니다: (i) PDF 추출기: PDF 문서에서 텍스트를 추출합니다; (ii) 문단 분류: 사용자 관심 정보가 포함된 문단을 선택합니다; (iii) 정보 추출: 선택된 문단에서 관련 정보를 추출합니다; (iv) 지식 표현: 추출한 정보를 해석하고 의미를 정리합니다. 이 과정에서 prompt engineering과 in-context learning (ICL)을 활용하여 LLM이 필요한 모든 맥락을 제공받도록 합니다.

- **Performance Highlights**: 실험 결과, 다섯 개의 오픈 소스 LLM 모델 모두에서 문단 분류 및 정보 추출에서 높은 성능을 보였습니다. 특히, 이들 모델은 기초적인 훈련이나 미세 조정 없이도 ICL이 사용되었을 경우 우수한 성능을 달성했습니다. 예제의 선택이 모델 성능에 미치는 영향을 강조하며, 이는 제과 연결에 있어 중요한 요소임을 보여줍니다.



### DM4Steal: Diffusion Model For Link Stealing Attack On Graph Neural Networks (https://arxiv.org/abs/2411.03364)
- **What's New**: 이 논문에서는 그래프 신경망(GNN)의 링크 도용(link stealing) 공격에 대한 새로운 방법론인 DM4Steal을 제안합니다. DM4Steal은 확산 모델(diffusion model)을 기반으로 하여, 다양한 공격 시나리오와 방어 GNN에 대한 적응성을 보장합니다.

- **Technical Details**: DM4Steal은 세 가지 중요한 측면에서 기존 연구와 다릅니다: (i) 일반성: 한정된 보조 지식(auxiliary knowledge)을 활용하여 여섯 가지 공격 시나리오를 다룰 수 있도록 새로운 훈련 전략을 제안합니다. (ii) 효과성: 확산 모델의 훈련 과정에서 의미적 구조(semantic structure)를 유지하여, GNN 의사결정 과정을 통해 정확한 그래프 토폴로지를 학습할 수 있습니다. (iii) 적응성: GNN 방어 메커니즘이 있을 때도 성능 저하를 최소화하는 안정성을 활용하여 DM4Steal이 성공적인 적응적 공격을 수행할 수 있도록 합니다.

- **Performance Highlights**: DM4Steal은 8개의 실제 데이터 세트에서 3개의 GNN에 대한 광범위한 실험을 통해 최신 기술(state-of-the-art, SOTA) 공격 성능을 달성했습니다. 또한 DM4Steal은 방어 GNN에 대해서도 효과적인 링크 도용 공격을 수행할 수 있습니다.



New uploads on arXiv(cs.CV)

### Community Forensics: Using Thousands of Generators to Train Fake Image Detectors (https://arxiv.org/abs/2411.04125)
Comments:
          15 pages

- **What's New**: 이 논문은 이전에 보지 못한 생성 모델이 생성한 이미지를 탐지하는 데 있어 훈련 데이터의 다양성이 큰 장애물이라는 점을 강조합니다. 저자들은 더 크고 다양한 데이터 세트를 제안하며, 이를 통해 일반화 능력을 연구합니다.

- **Technical Details**: 제안하는 Community Forensics 데이터 세트는 2.7M 이미지로 구성되며, 4803개의 서로 다른 생성 모델에서 샘플링되었습니다. 이 데이터 세트는 이미지 생성기 아키텍처 및 이미지 처리 설정에서 다양한 장면 콘텐츠를 포착합니다. 실험 결과 훈련 세트에 포함된 모델 수가 증가할수록 탐지 성능이 향상되는 것으로 나타났습니다.

- **Performance Highlights**: 훈련된 분류기는 제안된 평가 및 이전 표준 벤치마크에서 강력한 성능을 보여주었습니다. 특히, 다양한 생성 모델 아키텍처를 포함하면 성능이 현저히 개선되는 것을 확인했습니다.



### RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (https://arxiv.org/abs/2411.04097)
Comments:
          NeurIPS 2024

- **What's New**: 이 연구에서는 spurious correlations(허위 상관관계)를 발견하고 완화하기 위해 Region-aware Vision-Language learning(RaVL)이라는 새로운 접근법을 제안합니다. 기존 방법들은 주로 전역 이미지 레벨에서 작동하며, 세부 이미지 특징에 직접 개입하지 않으며 unimodal 설정에 주로 설계되었습니다.

- **Technical Details**: RaVL은 fine-tuned VLM에서 학습된 spurious correlations를 발견하기 위해 지역 수준의 클러스터링 접근 방식을 활용하고, 이를 통해 zero-shot classification 오류에 기여하는 정확한 이미지 특징을 식별합니다. 또한, RaVL은 novel region-aware loss function을 도입하여 모델이 관련 영역에 집중하고 허위 관계를 무시하도록 합니다.

- **Performance Highlights**: RaVL은 654개의 여러 VLM 아키텍처 및 데이터 도메인에 대해 평가되었으며, 스푸리어스 상관관계를 정확하게 발견하여 최근 경계선보다 191% 향상된 결과를 보여주었고, 최악 그룹 이미지 분류 정확도에서 8.2% 향상된 효과를 입증했습니다. 일반 도메인 및 의료 도메인 VLM에 대한 정성적 평가에서도 RaVL의 유용성이 확인되었습니다.



### Textual Decomposition Then Sub-motion-space Scattering for Open-Vocabulary Motion Generation (https://arxiv.org/abs/2411.04079)
Comments:
          project page: this https URL

- **What's New**: 본 논문은 텍스트에서 3D 동작을 생성하는 Text-to-motion (T2M) 생성 분야에서 개방형 어휘(open vocabulary) 동작 생성을 해결하기 위한 새로운 접근 방식을 제안합니다. 기존 방법들이 제한된 데이터셋으로 인해 오버피팅(overfitting) 문제에 직면해 있음을 지적하며, 원자 동작(atomic motion)을 중간 표현으로 사용하여 전면 매핑(full mapping) 문제를 해결하는 두 단계의 방법론을 도입합니다.

- **Technical Details**: 논문에서는 'Textual Decomposition'과 'Sub-motion-space Scattering'이라는 두 가지 단계로 구성된 방법론을 제안합니다. Textual Decomposition 단계에서 구체적인 동작 기술 변환 알고리즘을 설계하고 대형 언어 모델(large language model, LLM)을 활용하여 전이적 텍스트를 원자 텍스트(atomic texts)로 변환합니다. Sub-motion-space Scattering은 원자 동작을 목표 동작으로 변환하는 조합 과정을 학습하여, 제한된 라벨링된 데이터의 문제를 극복하고 일반화를 높입니다.

- **Performance Highlights**: DSO-Net이라는 네트워크를 기반으로 한 본 연구는 개방형 어휘 동작 생성에 있어 최신 방법들에 비해 상당한 성능 향상을 달성했음을 실험적으로 입증합니다. 문서의 코드 또한 공개되어 있어 실제 활용 가능성이 높습니다.



### H-POPE: Hierarchical Polling-based Probing Evaluation of Hallucinations in Large Vision-Language Models (https://arxiv.org/abs/2411.04077)
Comments:
          Poster at this https URL

- **What's New**: H-POPE라는 새로운 벤치마크를 도입하여 LVLMs의 객체 및 속성에 대한 환각(hallucination)을 평가합니다. 이 평가에서는 기존의 POPE 프레임워크를 수정하여 객체 존재와 더불어 미세한 속성에 대한 환각을 체계적으로 분석합니다.

- **Technical Details**: H-POPE는 COARSE-to-FINE 평가 방법을 사용하며, 이미지에 대한 객체 존재에 대한 질문와 그 객체의 속성에 대한 질문을 차례로 제시합니다. 또한, 기존의 고정 샘플링 전략 외에 새로운 이미지 기반 적대적 샘플링 전략을 제안하여 낮은 정확도를 관찰합니다.

- **Performance Highlights**: H-POPE의 평가에 따르면, 모델들은 객체 존재에 대한 질문에 비해 속성에 대한 질문에서 더 많은 환각을 나타내며, 최고 정확도는 약 76.76%에 불과합니다. 이는 기존의 POPE에서 InstructBLIP이 보여준 88.73%보다 낮은 수치입니다.



### Pseudo-labeling with Keyword Refining for Few-Supervised Video Captioning (https://arxiv.org/abs/2411.04059)
Comments:
          12 figures, Accepted in Pattern Recognition

- **What's New**: 이 논문은 'few-supervised video captioning'이라는 새로운 작업을 소개하며, 이 작업은 비디오에 대한 단 하나의 ground-truth 문장만을 사용하여 모델을 훈련하는 방식입니다. 이는 기존의 많은 ground-truth 문장이 필요한 방식과 대비됩니다.

- **Technical Details**: 제안된 few-supervised video captioning 프레임워크는 lexically constrained pseudo-labeling 모듈과 keyword-refined captioning 모듈로 구성되어 있습니다. 이 시스템은 pretrained token-level classifier를 사용하여 후보 문장을 편집하고, XLNet과 같은 pretrained language model을 통해 이를 미세 조정합니다. 또한, video-keyword gated fusion 전략을 사용하여 pseudo-labeled 문장이 비디오 내용과의 의미 일관성을 유지하도록 최적화합니다.

- **Performance Highlights**: MSVD, MSR-VTT, VATEX 데이터셋을 포함한 여러 벤치마크에서 제안된 접근 방식이 1개의 human-labeled 문장만을 사용하여도 보다 유망한 성과를 거두었으며, gated fusion captioning 모델은 모든 ground-truth 라벨을 사용하는 경우에도 기존의 SOTA(supervised) 방법들을 초과했습니다.



### Aligning Characteristic Descriptors with Images for Human-Expert-like Explainability (https://arxiv.org/abs/2411.04008)
- **What's New**: 본 논문에서는 딥러닝 모델의 결정 과정을 설명하기 위해 특성 설명자(characteristic descriptors)를 활용하는 새로운 접근 방식을 제안합니다. 이는 인간 전문가의 설명을 모방하여 출력 결과의 해석성을 높입니다.

- **Technical Details**: 제안하는 방법에는 모델 아키텍처 내에 개념 병목층(concept bottleneck layer)을 포함하여 이미지와 설명자의 인코딩 간 유사도를 계산하고, 이를 통해 설명을 생성하는 방식이 포함됩니다. 또한 CLIP을 사용하여 이미지를 인코딩하고, 텍스트 개념과의 매칭을 통해 분석합니다.

- **Performance Highlights**: 얼굴 인식과 가슴 X-레이 진단 실험을 통해 기존 기법에 비해 훈련을 통해 전문가 수준의 해석 가능한 설명을 제공하며, 외부의 시각적 설명보다는 간결하고 신뢰할 수 있는 서면 설명을 제공하는 능력이 있음을 입증했습니다.



### Local vs distributed representations: What is the right basis for interpretability? (https://arxiv.org/abs/2411.03993)
- **What's New**: 이번 연구는 심층 신경망(deep neural networks, DNNs)의 해석 가능성(interpretable AI)에서 분산된 표현(distributed representations)의 중요성을 강조합니다. 특히, 그래디언트 방법을 사용하는 기존의 해석 방식이 한정된 시각적 패턴에 의존하는 반면, 분산된 표현이 인간이 더욱 쉽게 해석할 수 있는 정보를 제공한다는 점을 발견했습니다.

- **Technical Details**: 연구팀은 560명의 참가자를 대상으로 세 가지 대규모 심리물리학(psycho-physics) 실험을 실시하여, 국소적(local) 표현과 분산된 표현에서 도출된 시각적 특성의 모호성(ambiguity)을 평가했습니다. 결과적으로, 분산된 표현이 더 쉽게 해석 가능하며, 특히 신경망의 깊은 층에서 이 효과가 더욱 두드러진다는 것을 알게 되었습니다. 기존의 국소적(혹은 local) 코드로는 이해하기 어려운 시각적 패턴을 단일 요소로 더욱 쉽게 해석할 수 있습니다.

- **Performance Highlights**: 연구 결과, 분산된 표현에서 도출된 특성은 국소적인 표현에서 도출된 특성보다  유의미하게 해석하기 쉽고, 이러한 경향은 신경망의 깊은 층에서 더욱 두드러진다는 강력한 증거를 제공했습니다. 또한, 모델은 국소적 표현보다 분산된 표현에서 도출된 특성에 훨씬 더 의존하는 것으로 나타났습니다.



### ReEdit: Multimodal Exemplar-Based Image Editing with Diffusion Models (https://arxiv.org/abs/2411.03982)
Comments:
          First three authors contributed equally to this work

- **What's New**: ReEdit라는 새로운 모듈형 및 효율적인 T2I(텍스트에서 이미지로) 프레임워크를 제안하여 이미지 에디팅의 정확성과 속도를 개선하였습니다. 이 기법은 텍스트와 이미지 모달리티에서의 수정을 포착하며, 고품질 페이크 리얼리즘 이미지 생성을 보장합니다.

- **Technical Details**: ReEdit는 세 가지 주요 구성 요소로 구성됩니다. 첫째, 사전 훈련된 어댑터 모듈을 사용하여 이미지 임베딩 공간에서 수정을 캡처합니다. 둘째, 세밀한 추론이 가능한 다중 모달 비전-언어 모델(VLM)을 통합하여 자연어에서 수정을 캡처합니다. 마지막으로, 테스트 이미지의 내용과 구조를 유지하며 관련 부분만을 수정하기 위해 이미지 생성기를 조건부로 훈련합니다. 이 접근법은 최적화 과정 없이 동작하며, 기존 T2I diffusions보다 빠른 속도를 자랑합니다.

- **Performance Highlights**: ReEdit는 기존 방법들에 비해 질적 및 양적 측면에서 일관되게 우수한 성능을 보이며, 고유 구조를 유지하면서 다양한 수정을 잘 처리합니다. 또한, 지정된 작업 최적화가 필요하지 않아 실용성이 높고, 평균적으로 4배 빠른 속도로 실행됩니다.



### HRDecoder: High-Resolution Decoder Network for Fundus Image Lesion Segmentation (https://arxiv.org/abs/2411.03976)
Comments:
          11 pages, 3 figures, accepted by MICCAI 2024, the revised version

- **What's New**: 본 논문에서는 fundus 이미지의 병변 세분화를 위한 새로운 네트워크인 HRDecoder를 제안합니다. 이 네트워크는 고해상도 표현 학습 모듈과 다중 스케일 예측을 융합하는 모듈을 통합하여 세밀한 지역 특징을 효과적으로 캡처합니다.

- **Technical Details**: HRDecoder는 두 개의 모듈로 구성됩니다. 첫 번째 모듈은 고해상도 입력을 시뮬레이션하여 작은 객체의 표현 학습을 향상시키며, 두 번째 모듈은 고해상도 다중 스케일 예측을 통합하여 세밀한 정보와 지역 맥락 신호를 캡처합니다. 이 방식은 메모리 사용량, 오버헤드, 및 추론 속도 측면에서 합리적인 비용으로 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: IDRID 및 DDR 데이터셋에서 실험 결과, HRDecoder는 fundus 병변의 세분화 정확도를 효과적으로 향상시키며 합리적인 메모리와 계산 오버헤드를 소비하고，만족스러운 추론 속도를 유지합니다.



### Face Reconstruction from Face Embeddings using Adapter to a Face Foundation Mod (https://arxiv.org/abs/2411.03960)
- **What's New**: 본 연구에서는 블랙박스(face recognition system) 얼굴 인식 모델의 임베딩(embedding)으로부터 얼굴 이미지를 재구성할 수 있는 새로운 방법을 제안합니다. 이를 위해 4200만 개의 이미지로 훈련된 얼굴 기초 모델(face foundation model)을 사용하고, 어댑터(adapter) 모듈을 통해 목표 임베딩(target embeddings)을 기초 모델의 임베딩 공간으로 변환합니다. 이 방식은 컴퓨팅 자원이나 추가 훈련이 필요 없이 다양한 얼굴 인식 모델에 적용이 가능합니다.

- **Technical Details**: 연구에서는 어댑터 모듈을 통해 얼굴 임베딩을 기초 모델의 입력 공간으로 매핑(mappig)하는 기술을 사용합니다. 이러한 기초 모델은 고정된 얼굴 인식 모델의 임베딩으로부터 얼굴 이미지를 생성할 수 있습니다. 본 연구의 실험은 여러 얼굴 인식 모델과 데이터셋에서 수행되었으며, 복원된 얼굴 이미지를 평가함으로써 기존 얼굴 재구성 공격보다 우수성을 입증합니다.

- **Performance Highlights**: 실험 결과, 복원된 얼굴 이미지는 다양한 얼굴 인식 모델에 대한 공격에서 이전 재구성 공격보다 뛰어난 성능을 발휘했습니다. 특히, 제안하는 방법은 얼굴 인식 시스템에 접근하는 데 효과적이며, 여러 얼굴 인식 모델 간의 임베딩 전달 가능성도 평가되었습니다.



### Energy Score-based Pseudo-Label Filtering and Adaptive Loss for Imbalanced Semi-supervised SAR target recognition (https://arxiv.org/abs/2411.03959)
- **What's New**: 본 연구에서는 비균형 클래스 상황에서의 Semi-Supervised SAR ATR(Automatic Target Recognition) 접근 방식을 제안합니다. 이 방법은 동적 에너지 점수(dynamic energy scores)와 적응형 손실(adaptive loss)을 활용하여, 라벨이 부족한 상황에서도 높은 인식 정확도를 달성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 두 가지 핵심 요소로 구성됩니다. 첫째, Energy Score 기반의 비슷한 배포에 대한 Pseudo-label 선택 메커니즘(ESIDPS)을 도입하여, 긴 꼬리 분포에서의 pseudo-label 신뢰성을 보장합니다. 둘째, Adaptive Margin Loss(AML) 및 Adaptive Hard Triplet Loss(AHTL)와 같은 적응형 손실 함수를 개발하여 class imbalance를 해결합니다. AML은 head 클래스와 tail 클래스 간의 경계 차이를 인식하게 하고, AHTL은 모델이 복잡한 difficile한 샘플에 집중하도록 유도합니다.

- **Performance Highlights**: 두 가지 매우 불균형한 SAR 데이터셋(MSTAR 및 FUSAR-ship)에 대한 광범위한 실험 결과, 제안한 방법이 불균형 클래스에서 SAR 대상을 인식하는 성능을 효과적으로 향상시켰습니다. 실험에서 제안한 방법은 데이터 불균형 문제로 인한 모델 편향을 극복하며, 높은 정밀도의 목표 인식을 달성했습니다.



### Act in Collusion: A Persistent Distributed Multi-Target Backdoor in Federated Learning (https://arxiv.org/abs/2411.03926)
- **What's New**: 최근 페더레이티드 러닝(Federated Learning, FL)에서 분산된 다중 목표를 가진 백도어 공격(Backdoor Attack)을 새롭게 제안했습니다. 여러 공격자가 각기 다른 클라이언트를 제어하며 다양한 트리거를 삽입하여 글로벌 모델에 백도어를 주입하는 복잡한 위협 모델을 개발하였습니다.

- **Technical Details**: 이 논문에서 제안된 DMBA(Distributed Multi-Target Backdoor Attack) 방법은 다중 타겟 트리거 전략을 통해 백도어 성능을 유지하고, 기존의 백도어 사이의 매개변수 충돌을 방지하는 `multi-channel dispersed frequency trigger strategy`를 설계했습니다. 또한, `backdoor replay`를 도입해 로컬 훈련 동안의 그래디언트 간섭을 완화했습니다.

- **Performance Highlights**: DMBA는 실제 피해를 최소화하면서도 고성능을 유지하였으며, 30 라운드 후에도 다양한 클라이언트에서 오는 세 가지 다른 백도어 공격의 성공률이 93% 이상으로 유지되었습니다. 이는 백도어 공격의 효율성과 지속성을 보장하며, 두 가지 최신 방어 방법에 대해서도 강한 강인성을 입증했습니다.



### Self-supervised Representation Learning for Cell Event Recognition through Time Arrow Prediction (https://arxiv.org/abs/2411.03924)
- **What's New**: 본 연구에서는 self-supervised representation learning (SSRL)을 활용하여 live-cell microscopy 데이터에서 cell event recognition을 위한 feature map을 획득하는 새로운 방법을 제시합니다. 이 방법은 전통적인 fully supervised 접근 방식에 비해 제한된 주석으로도 더 나은 성능을 제공합니다.

- **Technical Details**: 우리는 시간 방향 예측(time arrow prediction, TAP)을 통해 dense feature를 학습하여 cell division과 death을 인식하는 데 사용합니다. 이 과정에서 적절한 손실 함수와 정규화(term)를 적용하여 feature map을 최적화합니다.

- **Performance Highlights**: 실험 결과, TAP을 통해 학습된 feature map을 사용하여 주석된 데이터셋에 대해 미세 조정(fine-tuning)을 진행했을 때, end-to-end fully supervised 접근 방식보다 우수한 성능을 보여줍니다.



### ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization (https://arxiv.org/abs/2411.03862)
Comments:
          Accept to NeurIPS 2024

- **What's New**: 이번 논문에서는 생성 콘텐츠의 수분화 모델에서 효과적인 워터마킹(watermarking) 방법을 제안하며, 기존 방법들과의 차별점으로 강력한 워터마크를 활성적으로 숨길 수 있는 프로세스를 도입했습니다.

- **Technical Details**: 로빈(ROBIN) 스킴은 중간의 확산 상태에서 강력한 워터마크를 주입하고, 최종 생성 이미지에서 워터마크를 감추는 과정을 안내합니다. 이를 위해 최적의 프롬프트(guide prompt) 신호를 생성하는 적대적 최적화(adversarial optimization) 알고리즘을 사용합니다.

- **Performance Highlights**: 여러 확산 모델을 실험한 결과, 해당 워터마크는 심각한 이미지 변조에서도 검증 가능하며, 다른 최신 강력 워터마킹 방법들에 비해 뛰어난 투명성을 제공합니다.



### FedRISE: Rating Induced Sign Election of Gradients for Byzantine Tolerant Federated Aggregation (https://arxiv.org/abs/2411.03861)
Comments:
          This is a work under submission/review process

- **What's New**: 본 논문에서는 omnisicent 공격자에게 보다 강력한 저항성을 제공하는 FedRISE라는 새로운 로버스트 집계기(robust aggregator)를 개발했습니다. 이 방법은 개별 기울기의 최적 방향을 결정하기 위해 분산 감소된 희소 기울기(variance-reduced sparse gradients)와 부호 투표(sign-voting) 전략을 사용합니다.

- **Technical Details**: FedRISE는 두 개의 하이퍼파라미터(γ와 β)를 사용하며, 이는 클라이언트 수, 훈련 설정 및 데이터 분포에 최소한으로 의존합니다. 이 방법은 기울기 크기를 무시하고 사인 기반의 기울기 평가 함수(sign-based gradient valuation function)를 도입하여 잘못된 투표 가중치를 방지합니다.

- **Performance Highlights**: 실험 결과 FedRISE는 6가지 공격 하의 8개의 기존 로버스트 집계기에 비해 더 뛰어난 강인성을 보여줍니다. 기존의 방법들은 강력한 공격 상황에서 수치적으로 무너지는 반면, FedRISE는 보다 효율적으로 기울기 포함 공식을 엄격하게 적용하여 공격에 저항합니다.



### An Edge Computing-Based Solution for Real-Time Leaf Disease Classification using Thermal Imaging (https://arxiv.org/abs/2411.03835)
- **What's New**: 본 논문에서는 열 화상 이미지를 활용한 식물 잎 질병 분류를 위한 새로운 데이터셋을 제안하고, Raspberry Pi 4B와 같은 자원 제한 장치에서의 모델 최적화 기법을 평가하여 실시간 분류를 가능하게 하는 하드웨어 기반 솔루션을 개발했습니다.

- **Technical Details**: 연구에서는 InceptionV3, MobileNetV1, MobileNetV2, VGG-16와 같은 딥러닝 모델을 평가했으며, pruning 및 quantization-aware training (PQAT)을 통해 Edge TPU Max에서 VGG16이 최대 1.48x 더 빠른 추론 시간을 달성했습니다. MobileNetV1의 경우 Intel NCS2에서 정밀도 감소를 통해 최대 2.13x 더 빠른 성능을 보였습니다.

- **Performance Highlights**: 제안된 시스템은 16종의 식물에서 발병한 잎 질병을 식별할 수 있는 15,144 이미지의 새로운 공공 데이터셋을 제공하며, 실시간 이미지 분류를 위한 하드웨어 가속 장치인 Edge TPU와 Intel NCS2를 사용하여 성능을 최적화했습니다.



### An Enhancement of Haar Cascade Algorithm Applied to Face Recognition for Gate Pass Security (https://arxiv.org/abs/2411.03831)
- **What's New**: 본 연구는 Haar Cascade 알고리즘을 개선하여 얼굴 인식 및 감지의 잘못된 긍정(false positive) 및 잘못된 부정(false negative) 비율을 감소시키고, 도전적인 조건에서도 정확도를 높이는 데 초점을 맞추었습니다.

- **Technical Details**: Haar Cascade 알고리즘을 사용하여 얼굴의 고유한 특성을 나타내는 128차원 벡터를 인코딩하며, 이 과정에서 그레이스케일(grayscale) 이미지를 RGB 이미지로 변환하는 서브프로세스가 적용되었습니다. 논리적 프로세스와 얼굴 필터링이 비얼굴(non-face) 탐지를 감소시키는데 사용되었습니다.

- **Performance Highlights**: 개선된 Haar Cascade 알고리즘은 98.39%의 정확도(21.39% 증가), 63.59%의 정밀도(precision), 98.30%의 재현율(recall), 72.23%의 F1 점수를 기록하였고, 기존 알고리즘은 46.70%에서 77.00%의 정확도와 44.15%의 정밀도, 98.61%의 재현율, 47.01%의 F1 점수를 기록하였습니다. 이 연구의 결과는 550개의 이미지 데이터셋을 기반으로 301,950회의 비교를 통해 검증되었습니다.



### Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts (https://arxiv.org/abs/2411.03829)
Comments:
          Published in NeurIPS 2024

- **What's New**: 이 논문에서는 오픈 월드( open-world) 시나리오에서 anomalous клас스를 탐지하고 새로운 도메인에 일반화할 수 있는 효과적인 세분화 모델을 제안합니다. 이를 위해 새로운 generative augmentation 방법을 설계하여 이미지 및 객체 수준에서 anomaly 개체와 covariate shifts를 통합한 일관된 이미지를 생성하는 기술을 도입합니다.

- **Technical Details**: 제안된 방법은 semantic-to-image 생성 모델을 사용하여 covariate 및 semantic shifts를 포함한 데이터를 생성하고, 이를 통해 모델이 shift 유형 간의 본질적인 차이를 학습하도록 돕습니다. 또한, 학습 중 uncertainty를 재조정하여 semantic shifts에 대응하며, 도메인 변화와 관련된 특징을 정렬하는 feature extractor를 강화하는 두 단계의 훈련 전략을 제안합니다.

- **Performance Highlights**: 이 방법은 RoadAnomaly, SMIYC, ACDC-POC, MUAD 벤치마크에서 모두 state-of-the-art 성능을 달성하였으며, OOD detection 및 알려진 클래스 세분화에서 뛰어난 결과를 보였습니다.



### Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM Data Contamination (https://arxiv.org/abs/2411.03823)
- **What's New**: 이 연구에서는 다중 모달 대규모 언어 모델(MLLMs)에서 데이터 오염 감지를 위한 새로운 프레임워크인 MM-Detect를 소개합니다. 이 프레임워크는 다양한 오염 수준을 감지할 수 있으며, MLLMs의 훈련 세트 유출로 인한 성능 향상을 강조합니다.

- **Technical Details**: MM-Detect는 두 가지 방법, 즉 Option Order Sensitivity Test와 Slot Guessing for Perturbation Caption을 통합하여 Multiple-choice 및 Caption-based 질문과 같은 시각적 질문 응답(VQA) 작업을 처리합니다. 또한, 연구에서는 오염이 LLMs의 사전 훈련(pre-training) 단계에서 유래할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: MM-Detect를 총 5개의 VQA 데이터 세트에 걸쳐 11개의 널리 사용되는 MLLMs에 적용한 결과, MLLMs에서 데이터 오염이 관찰되었으며, 이는 모델에 따라 오염 정도가 다르게 나타났습니다. 이 연구는 MLLMs에서의 데이터 오염 감지 및 그로 인해 성능이 증가할 수 있는 경로를 최초로 체계적으로 분석한 것입니다.



### SA3DIP: Segment Any 3D Instance with Potential 3D Priors (https://arxiv.org/abs/2411.03819)
- **What's New**: 본 논문에서는 SA3DIP라는 새로운 방법을 제안하여 어떤 3D 인스턴스도 잠재적인 3D 우선순위를 활용해 분할할 수 있도록 합니다. 이 방법은 기하학적 및 질감적 우선순위를 기반으로 보완적인 3D 프리미티브를 생성하여 초기 오류를 줄이는 방식으로 머신러닝의 성능을 향상시킵니다.

- **Technical Details**: SA3DIP는 기하학적 및 색상 우선순위를 통합하여 고품질의 3D 인스턴스를 분할하는 새로운 파이프라인입니다. 이 방법은 초기 단계에서 발생하는 오류를 최소화하고 나중의 병합 과정에서도 3D 탐지기를 사용하여 제약 조건을 도입하여 오버 세그멘테이션을 방지합니다.

- **Performance Highlights**: 운영한 다양한 2D-3D 데이터셋에 대한 실험 평가 결과, SA3DIP의 접근 방식은 효과적이고 강력함을 입증했습니다. 특히, 새로운 ScanNetV2-INS 데이터셋을 통해 3D 클래스 무관 인스턴스 세그멘테이션의 성능이 크게 향상되었습니다.



### GS2Pose: Tow-stage 6D Object Pose Estimation Guided by Gaussian Splatting (https://arxiv.org/abs/2411.03807)
- **What's New**: 이 논문에서는 새로운 6D 포즈 추정 방법인 GS2Pose를 제안합니다. GS2Pose는 CAD 모델이 필요하지 않고 세분화된 RGBD 이미지만으로도 작업을 수행할 수 있습니다. 이 방법은 두 단계 구조로 코스 추정(coarse estimation)과 정밀 추정(refined estimation)으로 나눠져 있습니다.

- **Technical Details**: GS2Pose는 3D Gaussian splatting을 도입하여 3DGS 모델을 사용한 감독 학습을 통해 주변 환경 변화에 대한 강력한 저항성을 보입니다. 코스 단계에서 Pose-Net이라는 경량 U-Net 네트워크를 이용해 대략적인 포즈를 추정하며, 정밀 단계에서는 GS-Refiner를 통해 입력 이미지와 렌더링된 이미지를 비교하여 포즈를 정제합니다.

- **Performance Highlights**: GS2Pose는 LineMod 데이터셋에서 비교 알고리즘에 비해 뛰어난 정확도, 추론 속도 및 계산 자원 효율성을 보여주었습니다. 이 모델은 환경 변화, Occlusion, 조명 등 다양한 도전 과제를 극복하는데 높은 성능을 발휘하며, 곧 GitHub에 코드가 공개될 예정입니다.



### VQA$^2$:Visual Question Answering for Video Quality Assessmen (https://arxiv.org/abs/2411.03795)
Comments:
          10 pages 3 figures

- **What's New**: LMM(large multi-modal models)의 발전에 힘입어, 저자들은 비디오 품질 평가(Video Quality Assessment, VQA) 분야에서 Visual Question Answering(VQA) 기반의 첫 번째 대규모 Instruction Dataset인 VQA2 Instruction Dataset을 소개합니다. 이 데이터셋은 비디오 품질에 대한 질문 응답을 중심으로 구성되어 있으며, 157,735개의 질문-답변 쌍을 포함합니다.

- **Technical Details**: VQA2 Instruction Dataset은 세 가지 단계로 구성됩니다: 1단계는 왜곡 인식을 중심으로 한 사전 훈련 데이터, 2단계는 비디오 품질 점수를 위한 Instruction Tuning 데이터, 3단계는 비디오 품질 이해를 위한 Instruction Tuning 데이터입니다. 이번 연구는 다양한 비디오 유형과 구성을 다루기 위해 사람 주도의 주석을 바탕으로 고품질의 데이터를 활용합니다.

- **Performance Highlights**: VQA2 시리즈 모델은 비디오 품질 점수(task)에서 SOTA(state-of-the-art) 성능을 달성하였으며, Visual Quality Question Answering(task)에서도 GPT-4o를 초능가하는 성능을 나타냅니다. VQA2-Assistant 모델은 점수(task)와 질문 응답(task) 모두에서 우수한 성능을 보이며, 모델의 다재다능성을 입증합니다.



### Harmformer: Harmonic Networks Meet Transformers for Continuous Roto-Translation Equivarianc (https://arxiv.org/abs/2411.03794)
Comments:
          Appears in NeurIPS 2024 Workshop on Symmetry and Geometry in Neural Representations

- **What's New**: 이번 논문에서는 Harmformer라는 새로운 비전 트랜스포머를 소개합니다. 이 모델은 2D 로토-변환(equivariant to roto-translation)에서의 연속적 (continuous) 동등성을 달성하는 최초의 모델로, 기존의 데이터와 변환 연산에 대한 예측의 불변성을 보장합니다.

- **Technical Details**: Harmformer는 하모닉 네트워크(harmonic networks)를 기반으로 하여 처음부터 동등한 특성을 유지하도록 설계되었습니다. 이 네트워크는 선형 계층(linear layers)과 새로운 자기 주의(self-attention) 메커니즘을 통합하여 끝까지 동등성을 보장합니다. 이 모델의 훈련 과정에서 회전된 샘플을 보지 않고도 완전한 회전에 안정적인 성능을 발휘합니다.

- **Performance Highlights**: Harmformer는 기존의 불균형 변환 모델과 이산 변환 트랜스포머 성능을 뛰어넘는 결과를 보였습니다. 특히, 미리 회전된 데이터 없이 분류 작업을 수행할 때, Harmformer는 이전의 어떤 모델보다도 우수한 성능을 입증하였습니다.



### Homotopy Continuation Made Easy: Regression-based Online Simulation of Starting Problem-Solution Pairs (https://arxiv.org/abs/2411.03745)
- **What's New**: 본 연구는 기하학적 문제에 대한 자동화된 솔버의 효율성을 개선하기 위해 새로운 솔루션 회귀 네트워크(regression network)를 도입하며, 이를 통해 입력 대응 관계(input correspondences)로부터 직접적으로 해결책을 예측하고, 일관된 문제-해결 쌍을 생성하는 온라인 시뮬레이터(online simulator)를 사용합니다.

- **Technical Details**: 기존의 고차 다항식 방정식을 해결하는 방법은 Gröbner basis 방법과 polynomial resultants를 포함합니다. 다양한 기하학적 문제를 해결하는 데 사용되어온 전통적인 방식을 개선하여, 이 연구에서는 Homotopy Continuation (HC) 기법을 활용하여 단일 솔루션을 효과적으로 추적하는 새로운 방법을 제안합니다. 제안된 방법은 복잡한 문제에 대한 일관된 문제-해결 쌍을 생성할 수 있는 시뮬레이터를 통합하여, 효율성을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 일반화된 카메라 재배치(generalized camera resectioning)와 일반화된 상대 자세 및 스케일 문제(generalized relative pose and scale problem)에 적용되어 높은 성공률과 CPU 기반의 효율적인 솔버를 제공합니다. 실제 테스트에서 이 방법은 기존의 학습 기반 해결책들과 비교할 때, 뛰어난 효율성과 성공률을 보였습니다.



### Relation Learning and Aggregate-attention for Multi-person Motion Prediction (https://arxiv.org/abs/2411.03729)
Comments:
          Submitted to IEEE Transactions on Multimedia

- **What's New**: 이번 논문에서 제안하는 새로운 협업 학습 프레임워크는 각 개인 내부의 intra-relations와 서로 간의 inter-relations를 명시적으로 모델링하는 방식이다. 이를 통해 기존의 방법이 간과했던 관계를 정확히 다룰 수 있다.

- **Technical Details**: 논문에서는 Graph Convolutional Networks (GCN) 기반 네트워크를 사용하여 intra-relations를 모델링하고, cross-attention 메커니즘을 통해 inter-relations를 처리하는 새로운 구조를 제안한다. 또한, Interaction Aggregation Module (IAM)이라는 플러그 앤 플레이(plug-and-play) 집계 모듈을 도입하여 이 두 관계를 통합한다.

- **Performance Highlights**: 3DPW, 3DPW-RC, CMU-Mocap, MuPoTS-3D 및 합성 데이터셋 Mix1 & Mix2(9~15명)에서의 실험 결과, 제안된 방법이 최신 기술 수준의 성능을 달성함을 보여준다.



### Efficient Fourier Filtering Network with Contrastive Learning for UAV-based Unaligned Bi-modal Salient Object Detection (https://arxiv.org/abs/2411.03728)
Comments:
          11 pages, 7 figures

- **What's New**: 이번 연구에서는 비정렬 RGB 및 열 이미지 쌍을 활용하여 UAV(무인 항공기)를 기반으로 하는 이모달 주목 물체 탐지(BSOD) 문제를 해결하기 위해 AlignSal이라는 효율적인 포리에 필터 네트워크를 제안합니다. 이 모델은 실시간 성능과 정확성을 동시에 달성하며, 기존 모델에 비해 파라미터 수를 70% 줄이고, 부동 소수점 연산을 50% 줄이며, 추론 속도를 152.5% 향상시킵니다.

- **Technical Details**: AlignSal은 세 가지 주요 구성 요소로 구성됩니다. 첫째, SCAL(semantic contrastive alignment loss)은 RGB와 열 modality를 의미적 수준에서 정렬하여 각 modality가 서로를 개선하도록 합니다. 둘째, SAF(synchronized alignment fusion) 모듈은 빠른 Fourier 변환을 활용하여 채널 및 공간 차원에서 bi-modal feature를 정렬하고 융합합니다. 이를 통해 AlignSal은 저복잡도에서 멀티-스케일의 spatial offsets를 처리할 수 있습니다.

- **Performance Highlights**: AlignSal은 UAV RGB-T 2400 및 세 개의 약한 정합 데이터셋에서 다양한 평가 지표에 대해 16개의 최신 BSOD 모델들과 비교하여 더 나은 성능 및 일반화 능력을 보여줍니다. 특히 AlignSal은 MROS 모델과 비교해 적은 자원을 사용하면서도 더 나은 성능을 보이며, 입력 이미지의 무정렬 상태에서도 효과적으로 작동합니다.



### PX2Tooth: Reconstructing the 3D Point Cloud Teeth from a Single Panoramic X-ray (https://arxiv.org/abs/2411.03725)
Comments:
          Ma W, Wu H, Xiao Z, et al. PX2Tooth: Reconstructing the 3D Point Cloud Teeth from a Single Panoramic X-Ray[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2024: 411-421

- **What's New**: 이 논문에서는 Panoramic X-ray(PX) 이미지를 사용하여 3D 치아 구조를 복원하는 새로운 접근인 PX2Tooth를 제안합니다. PX2Tooth는 PX 이미지를 통해 고정밀 3D 치아 모델을 생성할 수 있으며, 이를 위한 2단계 프레임워크를 사용합니다.

- **Technical Details**: PX2Tooth는 먼저 PXSegNet을 사용하여 PX 이미지에서 영구 치아를 세분화합니다. 그런 다음 TGNet이라는 치아 생성 네트워크를 통해 3D 치아를 생성합니다. 이 과정에서 Prior Fusion Module(PFM)을 사용하여 생성 품질을 향상시킵니다. 해당 방법은 499개의 CBCT와 PX 이미지 쌍으로 검증되었습니다.

- **Performance Highlights**: PX2Tooth는 Intersection over Union(IoU) 점수 0.793을 달성하여 기존의 연구 방법론에 비해 우수한 성능을 보였습니다. 이는 디지털 치의학에서 인공지능의 잠재력을 여실히 보여줍니다.



### Estimation of Psychosocial Work Environment Exposures Through Video Object Detection. Proof of Concept Using CCTV Footag (https://arxiv.org/abs/2411.03724)
Comments:
          11 pages, 9 figures, presented at IWOAR 9th International Workshop on Sensor-Based Activity Recognition and Artificial Intelligence, September 26-27, Potsdam, Germany

- **What's New**: 이 논문은 CCTV 영상을 사용하여 심리사회적 작업 환경의 여러 측면을 추정하기 위한 컴퓨터 비전 알고리즘의 활용을 연구합니다. 고객과 직원 간의 상호작용을 감지하고 추적하는 방법론의 개념 증명(proof of concept) 을 제시합니다.

- **Technical Details**: 물체 감지(object detection) 및 추적(tracking) 알고리즘(YOLOv8 및 DeepSORT)과 자세 추정(pose estimation) 알고리즘(BlazePose)을 조합하여 고객 수와 직원 수, 상호작용의 지속 시간을 추정하는 파이프라인을 제안합니다. 상호작용은 거리(distance), 지속시간(duration), 자세(pose)에 따라 긍정적, 중립적 또는 부정적으로 분류되는 간단한 규칙 기반 접근 방식을 사용하여 평가됩니다.

- **Performance Highlights**: 제안된 방법론은 CCTV 영상 소량 집합에 대해 테스트되었으며, 객체 감지 및 추적 부분은 높은 재현율(recall)과 괜찮은 정확도를 보였습니다. 그러나 자세 추정은 여전히 한계가 있으며, 직원 추적의 어려움 때문에 상호작용 유형을 완전하게 감지하는 데에는 제한이 있습니다. 이 방법은 심리사회적 작업 환경의 자기 보고(self-reported measures) 방식에 대한 유망한 대안이 될 수 있습니다.



### These Maps Are Made by Propagation: Adapting Deep Stereo Networks to Road Scenarios with Decisive Disparity Diffusion (https://arxiv.org/abs/2411.03717)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문은 도로 표면 3D 재구성을 위한 결정적 시차 확산(Decisive Disparity Diffusion, D3Stereo) 방법을 소개합니다. 이는 사전 학습된 딥 컨볼루션 신경망(DCNNs)을 새로운 도로 시나리오에 적응시키는 첫 번째 시도로, 비용 볼륨을 여러 레벨로 학습된 표현을 사용하여 생성합니다.

- **Technical Details**: D3Stereo는 반복적인 양방향 필터링 알고리즘을 사용하여 비용을 집계하며, 내부 및 외부 스케일에서의 결정적 시차 확산 전략을 통해 희소한 시차 이미지를 보완하고 높은 해상도를 위한 유용한 사전 정보를 제공합니다. 이 방법은 깊이 있는 기능 메칭을 통해 기존의 프로그래밍 기반 알고리즘보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: UDTIRI-Stereo 및 Stereo-Road 데이터셋에서 수행된 광범위한 실험 결과, D3Stereo 전략이 사전 학습된 DCNN을 적응시키는 데 효과적임을 입증하였으며, 도로 표면 3D 재구성을 위한 모든 다른 프로그래밍 기반 알고리즘보다 뛰어난 성능을 발휘했습니다.



### Explaining Human Activity Recognition with SHAP: Validating Insights with Perturbation and Quantitative Measures (https://arxiv.org/abs/2411.03714)
- **What's New**: 본 연구는 인간 활동 인식(HAR) 영역에서 Graph Convolution Networks(GCNs)의 의사 결정 과정을 설명하기 위해 SHapley Additive exPlanations(SHAP)를 사용하는 새로운 접근 방식을 제안합니다. 이는 신경망의 복잡성을 극복하면서 예측의 투명성을 높이는 데 기여합니다.

- **Technical Details**: SHAP는 입력 피처가 GCN의 최종 예측 결과에 어떻게 기여하는지를 더 세밀하게 이해할 수 있도록 하며, 이는 각 피처의 기여 값을 명확히 할당합니다. 특히, 본 연구에서 GCN 모델에 대한 새로운 알고리즘인 ShapGCN을 적용하고, 두 개의 실제 HAR 데이터셋에서 설명의 유효성을 평가하기 위해 새로운 섭동 기법을 도입하였습니다.

- **Performance Highlights**: 연구 결과, SHAP를 통해 식별된 신체 주요 포인트는 정확도, 특이도 및 민감도 메트릭에 가장 큰 영향을 미쳤으며, HAR 작업에서 GCNs의 예측 결과에 대한 입력 피처의 기여를 세밀하게 분석할 수 있는 가능성을 보여줍니다.



### Fine-Tuning Vision-Language Model for Automated Engineering Drawing Information Extraction (https://arxiv.org/abs/2411.03707)
Comments:
          Paper has been submitted to the 9th International Conference on Innovation in Artificial Intelligence (ICIAI 2025)

- **What's New**: 이번 연구는 자동화된 GD&T(Geometric Dimensioning and Tolerancing) 추출 방법을 제안하며, Florence-2라는 오픈소스 비전-언어 모델(Vision-Language Model)을 미세 조정하여 제조업에서의 효율성을 높입니다.

- **Technical Details**: 모델은 400개의 도면으로 구성된 데이터셋에서 도메인 전문가에 의해 제공된 기준 주석과 함께 훈련되었습니다. 서로 다른 수준으로 증강된 데이터셋을 활용한 세 가지 실험을 통해, 0.23억 개의 파라미터를 가진 Florence-2를 전체 파라미터 조정(full-parameter fine-tuning)하여 최적화하였습니다. 대조적으로, 두 개의 최신 클로즈드소스 VLM인 GPT-4o와 Claude-3.5-Sonnet은 제로샷(zero-shot) 설정에서 평가되었습니다.

- **Performance Highlights**: Florence-2는 클로즈드소스 모델에 비해 precision은 29.95%, recall은 37.75%, F1-score는 52.40% 증가하였고, hallucination 비율은 43.15% 감소하였습니다. 이는 작은 오픈소스 VLM의 미세 조정이 GD&T 자동 추출을 위한 효과적이고 실용적인 해결책이 될 수 있음을 보여줍니다.



### 3DGS-CD: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangemen (https://arxiv.org/abs/2411.03706)
- **What's New**: 본 논문에서는 3D Gaussian Splatting (3DGS)을 기반으로 한 3D 장면의 물리적 객체 재배치 탐지 방법인 3DGS-CD를 제안합니다. 이 방법은 서로 다른 시간에 촬영된 두 세트의 비정렬 이미지 비교를 통해 3D 객체 수준의 변화를 추정합니다.

- **Technical Details**: 3DGS는 3차원 장면의 기하정보와 외형을 명시적으로 부호화하는 3D Gaussian의 집합을 사용합니다. 우리의 방법은 EfficientSAM의 제로샷 세분화(zero-shot segmentation) 기능을 활용하여 변화 전후의 이미지를 비교하고, 2D 객체 변화를 연관 지어 3D 객체 세그먼트 및 자세 변화를 정확히 추정합니다.

- **Performance Highlights**: 우리의 방법은 공개 데이터셋 및 자가 수집한 실제 데이터셋에서 검증되었으며, 최신 NeRF 기반 방법에 비해 최대 14% 높은 정확도와 3배 빠른 성능을 기록했습니다. 이러한 성능 향상은 객체 복원, 로봇 작업 공간 초기화, 3DGS 모델 업데이트와 같은 다양한 실제 응용에 활용될 수 있습니다.



### Graph-Based Multi-Modal Sensor Fusion for Autonomous Driving (https://arxiv.org/abs/2411.03702)
Comments:
          An extended abstract accepted at Young Researchers' Symposium, ICVGIP '24. This extended abstract contains the following: 1. Short summary of our work, SAGA-KF, accepted at ICPR'24. 2. A proposal that was awarded the Qualcomm Innovation Fellowship'24

- **What's New**: 본 연구는 모바일 로봇 및 자율주행에서의 씬 이해(scene understanding) 강화를 위해 다양한 감각 모달리티(sensor modalities)를 통합하는 새로운 접근 방식을 제안합니다. 특히, 객체 추적(Multi-Object Tracking) 및 동시 위치 추정 및 지도 작성(SLAM)과 같은 자율주행에 필요한 의사결정을 지원하는 그래프 기반의 동적 씬 표현을 중심으로 하고 있습니다.

- **Technical Details**: 제안된 기술은 Sensor-Agnostic Graph-Aware Kalman Filter(SAGA-KF)를 이용하여 여러 센서에서 발생하는 노이즈 데이터를 융합(fusion)하고, 이를 통해 각 객체의 상호작용(node interactions)을 모델링합니다. SAGA-KF는 노드 기반의 상호작용 함수를 통해 객체 간의 관계를 포착하며, 센서에 독립적인 상태 진화를 지원합니다.

- **Performance Highlights**: 제안된 SAGA-KF는 실험 결과 MOTA(Multiple Object Tracking Accuracy) 향상과 함께 추적된 객체의 위치 오류(MOTP) 감소 및 신원 전환(IDS) 감소를 보여주었습니다. 이러한 결과는 SAGA-KF가 자율주행 시스템의 상황 인식과 안전성을 향상시키는 데 기여할 수 있음을 입증합니다.



### OccLoff: Learning Optimized Feature Fusion for 3D Occupancy Prediction (https://arxiv.org/abs/2411.03696)
- **What's New**: 본 연구에서는 3D 환경 이해를 위한 새로운 프레임워크 OccLoff를 제안합니다. 이 방법은 LiDAR와 카메라의 다중 모달 데이터를 효과적으로 융합하여 점유 예측(occupancy prediction)의 정확성을 향상시키며, 계산 비용을 줄이는데 중점을 둡니다.

- **Technical Details**: OccLoff는 엔트로피 마스크(entropy masks)를 사용하는 희소 융합 인코더(sparse fusion encoder)를 도입하여 2D 및 3D 기능의 직접적인 융합을 가능하게 합니다. 또한, 점유 프록시 손실(occupancy proxy loss) 및 적응형 하드 샘플 가중치(adaptive hard sample weighting) 알고리즘을 제안하여 다수의 최신 방법의 성능을 향상시킵니다.

- **Performance Highlights**: nuScenes와 SemanticKITTI 벤치마크에서 OccLoff는 작은 객체의 인식을 포함하여 매우 우수한 성능을 보였으며, 제안된 학습 방법이 다른 최신 점유 예측 모델의 성능을 일관되게 향상시킨다는 것을 보여주었습니다.



### AMNCutter: Affinity-Attention-Guided Multi-View Normalized Cutter for Unsupervised Surgical Instrument Segmentation (https://arxiv.org/abs/2411.03695)
Comments:
          This paper was accepted by the 2025 IEEE Winter Conference on Applications of Computer Vision (WACV)

- **What's New**: 본 논문에서는 기존의 지도 학습(Supervised Learning) 방법이 필요했던 수술 기구 분할(Surgical Instrument Segmentation, SIS) 문제에서, 새로운 라벨 없는 비지도 학습(Unsupervised Learning) 모델인 AMNCutter를 제안합니다. 이 모델은 Multi-View Normalized Cutter(m-NCutter)라는 혁신적인 모듈을 통해 다양한 레벨의 패치 친화성(Patch Affinities)을 활용해 훈련되며, 각기 다른 수준의 피쳐를 통합하여 신뢰성을 높인다.

- **Technical Details**: 제안된 AMNCutter는 그래프 컷팅 손실(Graph-Cutting Loss)을 사용하며, 중간 피쳐 맵을 효과적으로 활용하여 끝(End-to-End) 프레임워크 상에서 실시간 실행이 가능하다. 또한 Multi-View Self-Attention 블록을 도입하여 다양한 레벨의 친화성 중 어떤 것이 더 중요한지를 동적으로 판단하고 최적화한다. 이 검증을 위해 다양한 SIS 데이터셋에서 실험을 수행하였다.

- **Performance Highlights**: 실험 결과, AMNCutter는 기존의 비지도 수술 기구 분할 방법보다 성능이 우수하며, 강력한 일반화 능력과 예외적인 실시간 처리 능력을 입증하였다. 다양한 데이터셋에서의 실험을 통해 SOTA(State-of-the-Art) 성능을 달성했고, 다운스트림 작업을 위한 프리트레인 모델로서의 잠재력이 확인되었다.



### Where Do We Stand with Implicit Neural Representations? A Technical and Performance Survey (https://arxiv.org/abs/2411.03688)
- **What's New**: Implicit Neural Representations (INRs)는 지식 표현 분야에서 유연성과 성능을 제공하는 새로운 패러다임으로 떠올랐습니다. 본 서베이는 INRs의 최신 방법을 종합적으로 리뷰하고, 이를 네 가지 주요 분야인 Activation Functions, Position Encoding, Combined Strategies, Network Structure Optimization으로 분류한 명확한 세분법을 제시하였습니다.

- **Technical Details**: INRs는 Multilayer Perceptrons (MLPs)를 활용하여 데이터를 연속적인 암시적 함수로 모델링합니다. 이는 해상도 독립성(memory efficiency) 및 일반화(generalisation) 등의 장점을 제공하며, 복잡한 역문제를 해결하는 데 탁월합니다. 본 연구에서는 다양한 접근 방식 간의 트레이드오프를 비교하여 최신 INR 기술의 역량과 과제를 조명합니다.

- **Performance Highlights**: 본 서베이는 현재 방법들이 뛰어난 분야와 잠재적 개선 방안, 예를 들어 더 표현력이 풍부한 활성화 함수, 향상된 위치 인코딩 메커니즘, 복잡한 고차원 데이터에 대한 확장성 강화를 강조합니다. INRs의 적용 가능성을 넓혀줄 새로운 연구 방향을 제시하여 향후 비약적인 발전을 이끌어낼 기회를 마련하고자 합니다.



### Towards 3D Semantic Scene Completion for Autonomous Driving: A Meta-Learning Framework Empowered by Deformable Large-Kernel Attention and Mamba Mod (https://arxiv.org/abs/2411.03672)
- **What's New**: MetaSSC는 비용 효율적인 배포를 목표로 하는 새로운 meta-learning 기반의 프레임워크입니다.

- **Technical Details**: 이 프레임워크는 deformable convolution과 large-kernel attention을 활용하여 3D voxel grid 내에서의 장기 의존성(long-range dependencies)을 효과적으로 포착합니다. 또한, voxel 기반의 semantic segmentation(SS) 사전 학습(pretraining) 작업을 통해 불완전한 영역의 의미 및 기하학을 탐색하고 전이 가능한 meta-knowledge를 습득합니다.

- **Performance Highlights**: MetaSSC는 경쟁 모델들보다 월등한 성능을 발휘하며 배포 비용 또한 줄였음을 입증하는 광범위한 실험 결과를 보여줍니다.



### Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation? (https://arxiv.org/abs/2411.03670)
Comments:
          Accepted to NeurIPS-2024

- **What's New**: 이 논문은 AI 성능 테스트의 중요성을 다루며, 기존의 벤치마크의 한계를 극복하기 위해 Touchstone이라는 대규모 협업 세분화 벤치마크를 제안합니다. 이는 9종의 복부 장기를 대상으로 하며, 전 세계 76개 병원에서 수집한 5,195개의 훈련 CT 스캔과 11개 병원에서 수집한 5,903개의 테스트 CT 스캔으로 구성되어 있습니다. 이 데이터셋은 AI 알고리즘의 다양한 상황에서의 성과를 rigorously 평가할 수 있게 해줍니다.

- **Technical Details**: 제안된 Touchstone 벤치마크는 다양한 AI 알고리즘의 성능을 테스트할 수 있도록 설정되어 있습니다. 논문에서는 19개의 AI 알고리즘 개발자 14명을 초대하여 알고리즘을 훈련시킨 후, 세 가지 테스트 세트에서 독립적인 평가를 진행했습니다. 평가는 U-Net, nnU-Net, MedNeXt와 같은 기존 AI 프레임워크 또한 포함되었습니다. nnU-Net은 자동 구성되고, MedNeXt는 Transformer 기반의 확장성을 통해 3차원 의료 이미지 세분화에서의 효과적인 훈련을 지원합니다.

- **Performance Highlights**: 새로운 벤치마크는 다양한 의료 세분화 모델의 정확도를 높이는 데 기여합니다. STU-Net과 같은 모델은 네트워크 깊이와 넓이를 조정하여 세분화 정확도를 향상했습니다. 또한, UniSeg는 다중 작업 의료 이미지 세분화에서 전이 학습 능력을 향상시키는 혁신적인 접근을 제공합니다. 전체적으로, 이러한 새로운 접근법들은 3D 의료 이미지 분야의 발전에 큰 기여를 할 것으로 기대됩니다.



### Adaptive Stereo Depth Estimation with Multi-Spectral Images Across All Lighting Conditions (https://arxiv.org/abs/2411.03638)
- **What's New**: 본 논문은 다중 스펙트럼 깊이 추정을 위한 새로운 프레임워크를 제안하며, 가시광선 및 열 이미지를 스테레오 쌍으로 처리하여 기하학적 제약을 강화하는 방법을 제시합니다.

- **Technical Details**: 우리 방법은 Cross-modal Feature Matching (CFM) 모듈을 통해 각 픽셀에 대한 정렬된 특징 벡터를 생성하고, 비용 볼륨을 구성하여 픽셀 수준의 정밀한 매칭을 수행합니다. 또한, 저조도 영역에서의 불량 조명의 영향을 완화하기 위해 Degradation Masking 메커니즘을 도입합니다.

- **Performance Highlights**: 제안하는 방법은 Multi-Spectral Stereo (MS2) 데이터셋에서 최신 기술(state-of-the-art)보다 우수한 성능을 보이며, 다양한 조명 조건에서 고품질 깊이 맵을 생성하는 것을 입증하였습니다.



### Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis (https://arxiv.org/abs/2411.03637)
Comments:
          NeurIPS 2024 Accepted

- **What's New**: 이 논문은 SCGaussian이라는 새로운 방법을 제안하여 3D Gaussian Splatting(3DGS) 기반의 뷰 합성과정에서 입력 데이터가 희소할 때 겪는 성능 저하 문제를 해결합니다.

- **Technical Details**: SCGaussian은 매칭 프라이어(matching priors)를 활용하여 3D 일관된 장면 구조를 학습하는 구조 일관성 있는 가우시안 스플래팅(method). 이 방법은 렌더링 기하학(rendering geometry)과 가우시안 프리미티브의 위치를 최적화하는 두 가지 방식으로 장면 구조를 조정합니다. 특히, 레이 기반 가우시안 프리미티브(ray-based Gaussian primitives)의 최적화는 레이 방향으로 제한되어 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 SCGaussian은 다양한 복잡한 대규모 장면에서 기존 최첨단(performance) 기법에 비해 우수한 성능과 높은 효율성을 보여줍니다.



### StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding (https://arxiv.org/abs/2411.03628)
- **What's New**: 본 논문에서는 Multimodal Large Language Models (MLLMs)의 스트리밍 비디오 이해 능력을 평가하기 위한 최초의 종합 벤치마크인 StreamingBench를 소개합니다. 이는 MLLMs가 온라인 비디오 스트림을 이해하는 데 필요한 중요한 기술적 요소를 평가하고, 기존의 오프라인 비디오 이해 방법론과의 차이를 부각합니다.

- **Technical Details**: StreamingBench는 세 가지 핵심 측면인 (1) 실시간 시각 이해 (real-time visual understanding), (2) 전천후 소스 이해 (omni-source understanding), (3) 맥락 이해 (contextual understanding)를 평가합니다. 900개의 비디오와 4,500개의 질문으로 구성되어 있으며, 18개의 다양한 작업으로 나뉘어 있습니다. 각 비디오는 밀접하게 관련된 질문을 포함하고, 스트리밍 시나리오를 시뮬레이션하기 위해 여러 시점에서의 질문을 제공합니다.

- **Performance Highlights**: StreamingBench에서 13개의 개방형 및 독점 MLLMs을 실험한 결과, 가장 진보된 모델인 Gemini 1.5 Pro와 GPT-4o조차도 인간 수준의 스트리밍 비디오 이해 능력에 비해 평균 24.59% 낮은 정확도를 기록했습니다. 이로써 현재 MLLMs가 스트리밍 컨텍스트를 따라잡는 데는 상당한 격차가 있으며, 이는 향후 MLLMs의 발전을 위해 중요한 인사이트를 제공합니다.



### Hybrid Attention for Robust RGB-T Pedestrian Detection in Real-World Conditions (https://arxiv.org/abs/2411.03576)
Comments:
          Accepted for publication in IEEE Robotics and Automation Letters, October 2024

- **What's New**: 최근 자율주행 애플리케이션에서 다중 스펙트럼 보행자 탐지가 주목받고 있습니다. 본 논문에서는 열화상 이미지와 RGB 이미지의 혼합을 통해 특정 문제를 해결하고자 하며, 부분적인 오버랩과 센서 실패 문제를 해결하기 위해 Hybrid Attention (HA) 메커니즘을 도입했습니다.

- **Technical Details**: 이 논문에서는 Hybrid Attention (HA) 모듈을 사용하여 자가 주의(self-attention)와 교차 주의(cross-attention)를 결합하여 성능 저하를 완화하는 방법을 제안합니다. HA-MLPD(Hybrid Attention-based Multi-Label Pedestrian Detector) 알고리즘은 부분 오버랩 및 센서 실패에 대해 회복력이 강한 RGB-T 융합 알고리즘을 제공합니다. 또한, 임베디드 시스템에서의 자원 제약을 고려하여 경량의 백본(backbone)을 사용합니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션한 다양한 부분 오버랩 및 센서 실패 시나리오에서 테스트를 통해 성능 저하를 방지하고 기존의 최신 방법들에 비해 우수한 성능을 보였습니다.



### Estimating Ego-Body Pose from Doubly Sparse Egocentric Video Data (https://arxiv.org/abs/2411.03561)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 연구에서는 egocentric 비디오에서 카메라 착용자의 신체 움직임을 추정하는 문제를 다룹니다. 현재의 방법들은 대부분 밀집한 센서 데이터에 의존하지만, 본 연구에서는 드문 손 자세와 같은 시간적으로 희소한 데이터도 활용할 수 있음을 제안합니다.

- **Technical Details**: 이 방법은 두 단계 접근법을 사용하며, 첫 번째 단계는 손 궤적을 추정하는 Masked Autoencoder(MAE)를 적용하고, 두 번째 단계에서는 조건부 확산 모델(conditional diffusion model)을 사용하여 전체 신체 움직임을 생성합니다. 이를 통해 개별 손 궤적의 불확실성을 추정합니다.

- **Performance Highlights**: 다양한 HMD 설정에서 AMASS 및 Ego-Exo4D 데이터셋을 통해 엄격한 실험을 진행하여 본 방법의 유효성을 입증하였으며, 기존 방법보다 더 우수한 성능을 자랑합니다.



### Object and Contact Point Tracking in Demonstrations Using 3D Gaussian Splatting (https://arxiv.org/abs/2411.03555)
Comments:
          CoRL 2024, Workshop on Lifelong Learning for Home Robots, Munich, Germany

- **What's New**: 이 논문에서는 비디오 데모에서 터치 상호작용 포인트를 추출하고 객체의 움직임을 추적하여 Interactive Imitation Learning (IIL)을 향상시키기 위한 방법을 소개합니다.

- **Technical Details**: 3D Gaussian Splatting과 FoundationPose와 같은 최첨단 기술을 활용하여 로봇이 동적 환경에서 객체를 이해하고 조작할 수 있도록 지원합니다. 이를 통해 세밀한 상호작용 데이터를 제공하여 IIL 프레임워크를 향상시킵니다.

- **Performance Highlights**: 이 방법은 다양한 객체에 대한 조작을 가능하게 하며, 특히 아티큘레이트(articulated) 객체(예: 문, 서랍)에서 더 향상된 성능을 보여줍니다. 큰 객체와 시각적으로 뚜렷한 특징이 있는 경우에 가장 신뢰할 수 있는 결과를 보였습니다.



### Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Datas (https://arxiv.org/abs/2411.03554)
- **What's New**: 이번 논문에서는 Facial Identity Unlearning Benchmark (FIUBench)를 소개하여 Vision Language Models (VLMs)에서 개인 정보를 잊는 알고리즘의 효율성을 평가하는 새로운 벤치마크를 제안했습니다. 이는 'Right to be Forgotten' 규정을 기반으로 하여 Privacy 문제에 대한 중요성을 강조합니다.

- **Technical Details**: FIUBench는 Fictitious Facial Identity VQA 데이터셋을 기반으로 하여 VLM unlearning 작업을 공식화하고, 정보의 출처와 노출 수준을 정밀하게 제어하는 두 단계 평가 파이프라인을 사용합니다. 또한, membership inference attacks 및 adversarial privacy attacks와 같은 다양한 평가 지표를 제공합니다.

- **Performance Highlights**: 여기서 평가한 네 가지 VLM unlearning 알고리즘 모두 제한된 성능을 보였으며, 모델 유용성과 잊기 품질 간의 큰 균형이 필요함을 보여주었습니다. 또한, robust evaluation을 통해 Privacy 공격의 중요성을 강조하였습니다.



### Personalized Video Summarization by Multimodal Video Understanding (https://arxiv.org/abs/2411.03531)
Comments:
          In Proceedings of CIKM 2024 Applied Research Track

- **What's New**: 이 논문에서는 사용자 선호에 따라 비디오 요약을 생성하는 새로운 벤치마크인 UserPrefSum 데이터셋을 소개하며, 비디오 자막 및 장면을 분석하여 사용자 맞춤형 비디오 요약을 생성하는 Video Summarization with Language (VSL) 파이프라인을 제안합니다.

- **Technical Details**: VSL 파이프라인은 멀티모달(scene detection) 장면 탐지, 비디오 자막 생성(captioning), 멀티모달 요약(summarization), 비디오 선택(selection)이라는 네 가지 구성 요소로 이루어져 있으며, CLIP 모델을 활용하여 장면의 장르를 자동으로 레이블링합니다. 이 방식은 사전 학습된 비주얼 언어 모델(VLMs)을 사용하여 대규모 트레이닝 데이터 없이도 사용자 선호에 맞춰 요약을 생성할 수 있습니다.

- **Performance Highlights**: VSL은 현업에서의 사용에 적합성을 입증했으며, 기존의 비지도 학습 기반 비디오 요약 모델보다 더 많은 데이터셋에서 유연하게 적용할 수 있습니다. 또한, 사용자 선호가 증가할 때에도 요약 생성 능력이 향상되며, 실제 응용 프로그램에서 중요한 실시간 처리 능력을 보여줍니다.



### Beyond Complete Shapes: A quantitative Evaluation of 3D Shape Matching Algorithms (https://arxiv.org/abs/2411.03511)
- **What's New**: 이번 연구에서는 부분적으로 관측된 3D 형태 매칭(partial shape matching) 문제의 가능성을 탐구하며, 기존의 데이터 세트의 한계를 극복하기 위한 절차적 데이터 생성 기능을 제안합니다. 특히, 연구자들이 가상적으로 무한한 부분 형태 쌍을 생성할 수 있도록 하여, 공통적으로 적용할 수 있는 새로운 기준 데이터를 제공합니다.

- **Technical Details**: 연구 기간 동안 우리는 기존의 완전 형태 데이터셋을 기반으로 현실적이고 다양한 부분 형태 쌍을 생성하기 위한 프레임워크를 개발했습니다. 이 프레임워크는 교차 카테고리 매칭을 지원하며, 특정 응용 프로그램 요구 사항에 맞춘 사용자 정의 데이터 생성을 가능하게 합니다. BeCoS라는 새로운 벤치마크를 제안하고, 기존 데이터 세트를 기반으로 하여 다양한 형태 카테고리를 포함하고 보다 현실적인 부분 인스턴스를 만들어 내었습니다.

- **Performance Highlights**: 실험 결과 현재 최첨단의 방법들이 부분 형태 매칭 문제에서 한계가 있음을 보여주었습니다. 벤치마크에서 다양한 방법들을 평가하며, 이 분야에서의 추가적인 연구와 방법론 개발의 필요성을 강조합니다. 이러한 문제는 실질적으로 해결되지 않은 상태임을 시사하고 있습니다.



### SynthSet: Generative Diffusion Model for Semantic Segmentation in Precision Agricultur (https://arxiv.org/abs/2411.03505)
- **What's New**: 이 논문은 정밀 농업 분야에서의 시맨틱 세그멘테이션 작업을 위해 합성 주석 데이터 생성 방법론을 제시합니다. Denoising Diffusion Probabilistic Models (DDPMs)와 Generative Adversarial Networks (GANs)를 활용하여 인간의 개입 없이 사실적인 농업 데이터를 합성하는 이중 확산 모델 아키텍처를 제안합니다.

- **Technical Details**: 우리는 이중 확산 모델 아키텍처를 설계하여 이미지와 그에 대응하는 바이너리 세그멘테이션 마스크를 생성합니다. 이 과정에서 skip connections와 cross-attention 메커니즘을 통해 생성되는 이미지와 마스크 간의 일관성을 높입니다. 또한, super-resolution 기법을 사용해 생성된 이미지-마스크 쌍의 품질을 높입니다.

- **Performance Highlights**: 우리의 생성 데이터로 훈련된 모델은 외부의 다양한 실제 밀밭 데이터셋에서 시험했을 때 유망한 성능을 보였습니다. 제안된 방법론은 정밀 농업의 시맨틱 세그멘테이션 작업에서 데이터 부족 문제를 해결하는 데 효과적임이 입증되었습니다.



### An Application-Agnostic Automatic Target Recognition System Using Vision Language Models (https://arxiv.org/abs/2411.03491)
Comments:
          Accepted to the Thirty-Seventh Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-25)

- **What's New**: 본 연구에서는 개방형 어휘(open-vocabulary) 객체 탐지 및 분류 모델을 사용하는 새로운 Automatic Target Recognition (ATR) 시스템을 제안합니다. 이 시스템의 주된 장점은 비전문 사용자가 런타임 직전에 대상 클래스를 정의할 수 있다는 것입니다.

- **Technical Details**: 대상에 대한 텍스트 설명이나 이미지 예시를 통해 세부 사항을 전달할 수 있으며, 이는 적은 양의 훈련 데이터로도 고유한 대상을 식별할 수 있는 데 유용합니다. 우리는 여러 기술을 결합하여 성능을 개선하였으며, 특히 겹치는 프레임의 추가 정보를 활용한 tubelet identification(시간적 바운딩 박스 매칭), 바운딩 박스 재평가, tubelet linking 등을 구현했습니다.

- **Performance Highlights**: 초기 적용 사례로는 비행장 활주로에서 폭발하지 않은 탄약을 탐지하고 제거하는 작업에 사용되었으며, 현재는 다른 실제 응용 프로그램으로 연구를 확장하고 있습니다.



### Rainfall regression from C-band Synthetic Aperture Radar using Multi-Task Generative Adversarial Networks (https://arxiv.org/abs/2411.03480)
Comments:
          36 pages, 13 figures

- **What's New**: 이 논문은 Synthetic Aperture Radar (SAR) 데이터를 활용하여 높은 공간적 해상도(200미터/pixel)에서 강수량을 추정하는 데이터 기반 접근 방식을 제시합니다. 특히, 기상 레이더와의 데이터 콜로케이션(Spatial Co-location)과 강풍 하의 강수 예제 부족 문제를 해결하기 위한 다중 목표 수식을 도입하여 개선합니다.

- **Technical Details**: 제안된 방법은 패치 레벨 구성 요소와 대적적 구성 요소(adversarial component)를 도입하여 NEXRAD 아카이브를 활용합니다. SAR 데이터와의 원활한 콜로케이션을 찾기 위해 Sentinel-1 데이터를 이용하며, 훈련 과정 개선과 추가 입력을 포함하여 강수 추정의 정확성을 높였습니다. 또한 이 모델은 최대 15 m/s의 바람 조건에서도 성능을 연장할 수 있는 능력을 보입니다.

- **Performance Highlights**: 논문에서 제안한 모델은 강풍 하에서도 더 정확한 강수량 추정을 가능하게 하며, SAR 관찰을 통해 높은 해상도의 강수 패치 추정이 가능함을 보여줍니다. 이는 기상 및 수문학적 상황의 이해와 홍수 예보 개선에 중요한 기여를 할 것입니다.



### Self Supervised Networks for Learning Latent Space Representations of Human Body Scans and Motions (https://arxiv.org/abs/2411.03475)
Comments:
          23 pages, 11 figures, 6 tables

- **What's New**: 이 논문에서는 3D 인간 신체 분석 및 처리의 여러 기본 문제를 해결하기 위해 자기지도 학습(self-supervised learning) 신경망 모델을 도입합니다. 특히, VariShaPE(Varifold Shape Parameter Estimator)라는 새로운 아키텍처를 제안하여 인체 형태와 포즈의 잠재 공간(latent space) 표현을 빠르고 안정적으로 추정할 수 있습니다.

- **Technical Details**: 논문에서 제안하는 VariShaPE는 비등록된 메쉬(unregistered meshes)로부터 신속하게 잠재 코드(latent codes)를 추정하는 데 초점을 맞추고 있으며, MoGeN(Motion Geometry Network)는 이러한 잠재 공간의 기하학을 학습하는 프레임워크입니다. 두 모델은 함께 사용되어 4D 데이터로부터 인체 동작의 기하학을 보다 정밀하게 표현합니다. 이 과정에서 비선형 기하학의 변화를 반영해 인체 동작을 더욱 정확하게 포착할 수 있습니다.

- **Performance Highlights**: 제안된 VariShaPE 모델은 이전의 최첨단 3D-Coded 방법과 경쟁력 있는 성능을 보여줍니다. 특히, 자원 소모적 측면에서 훨씬 적은 계산 시간으로 유사한 결과를 낼 수 있음을 실험을 통해 입증했습니다. 또한, 이 모델은 메쉬 리샘플링에 대한 저항성을 가지고 있어 실제 데이터의 노이즈에도 효과적으로 대응할 수 있습니다.



### Fine-Grained Spatial and Verbal Losses for 3D Visual Grounding (https://arxiv.org/abs/2411.03405)
Comments:
          Accepted at WACV 2025

- **What's New**: 이 논문에서는 3D 시각적 그라운딩(3D visual grounding) 분야의 한계를 극복하기 위해 두 가지 새로운 손실 함수(loss functions)를 제안합니다. 첫 번째는 각 인스턴스에서 지상 진실 대상 인스턴스를 가리키는 3D 오프셋 벡터를 회귀하는 시각적 오프셋 손실(visual-level offset loss)이며, 두 번째는 설명에서 지칭된 인스턴스의 단어 수준 범위(word-level span)에 대한 예측을 기반으로 하는 언어 관련 범위 손실(language-related span loss)입니다.

- **Technical Details**: 제안된 AsphaltNet 아키텍처는 최상단-하단 양방향 주의적 융합 블록(top-down bidirectional attentive fusion block)을 통해 시각적 및 언어적 인식 통합을 강화합니다. 이를 통해 제안된 두 가지 손실 함수의 감독 신호가 네트워크의 대화 지점으로 전파되어 인스턴스 임베딩(instance embeddings)과 언어 임베딩(embedding) 간의 상관 관계를 학습하도록 돕습니다.

- **Performance Highlights**: AsphaltNet의 제안된 보조 손실 함수는 ReferIt3D 벤치마크에서 기존 최첨단 모델과 비교했을 때 경쟁력 있는 결과를 보여주었습니다. 이로 인해 3D 시각적 그라운딩의 성능이 크게 향상되었음을 알 수 있습니다.



### Enhancing Maritime Situational Awareness through End-to-End Onboard Raw Data Analysis (https://arxiv.org/abs/2411.03403)
Comments:
          38 pages

- **What's New**: 이번 연구는 소형 위성의 엄격한 대역폭, 에너지 및 지연 제약을 해결하기 위한 새로운 프레임워크를 제안합니다. 해상 모니터링에 중점을 두고 있으며, 해양 감시를 위한 빠르고 효과적인 데이터 처리를 가능하게 합니다.

- **Technical Details**: 이 연구는 원시 위성 이미지에서 선박을 직접 탐지하고 분류하기 위해 딥 러닝 기법을 적용하며, 데이터 처리 체인을 간소화하여 교정(calibration)이나 정사 보정(ortho-rectification) 등의 계산 집약적인 단계를 요구하지 않습니다. 또한, 기존 원시 위성 데이터 부족 문제를 해결하기 위해 VDS2Raw와 VDV2Raw라는 두 개의 새로운 데이터셋을 소개합니다.

- **Performance Highlights**: CubeSat 유사 하드웨어에 대한 개념 증명을 통해 제안된 방법의 실행 가능성을 입증했습니다. 이는 해양 모니터링을 위한 반응 시간을 단축시키고 효율성을 높이는 데 중요한 발전을 의미합니다.



### Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection (https://arxiv.org/abs/2411.03359)
Comments:
          accepted by NeurIPS 2024

- **What's New**: 이번 연구에서는 Self-Calibrated Tuning (SCT)라는 새로운 프레임워크를 도입하여, 불완전한 OOD (Out-of-Distribution) 특성을 해결하고 효과적인 OOD 탐지를 가능하게 합니다. 이는 단지 몇 개의 ID (In-Distribution) 데이터만으로 수행됩니다.

- **Technical Details**: SCT는 원래 학습 목표의 두 구성 요소에 각각 조정 요인을 도입하여 OOD 정규화의 영향을 조절할 수 있도록 학습 과정에서의 최적화 방향을 적응적으로 조절합니다. 이를 통해 낮은 예측 불확실성을 가진 데이터로 학습할 때 모델이 더 나은 일반화를 이룰 수 있도록 한다.

- **Performance Highlights**: SCT 방법은 대규모 ImageNet-1k 벤치마크에서 기존의 최상의 방법보다 잘못된 긍정 탐지 비율(FPR95)을 3% 개선하였으며, 다양한 실험과 분석을 통해 SCT의 효율성을 검증하였습니다.



### Unlocking the Archives: Using Large Language Models to Transcribe Handwritten Historical Documents (https://arxiv.org/abs/2411.03340)
Comments:
          29 Pages, 11 Tables, 2 Figures

- **What's New**: 이 연구는 Large Language Models (LLMs)이 특화된 Handwritten Text Recognition (HTR) 소프트웨어보다 역사적인 손글씨 문서를 훨씬 더 높은 정확도로 기록할 수 있음을 보여줍니다. 이와 함께 개발한 오픈 소스 소프트웨어 도구인 Transcription Pearl은 상업적으로 사용 가능한 다양한 LLM을 활용하여 효율적으로 손글씨 문서를 자동으로 기록하고 수정합니다.

- **Technical Details**: Transcription Pearl은 18세기와 19세기 영어 손글씨 문서의 다양한 데이터셋을 사용하여 테스트 하였으며, LLM은 Character Error Rates (CER) 5.7-7%와 Word Error Rates (WER) 8.9-15.9%를 기록하여, 기존 HTR 소프트웨어인 Transkribus보다 각각 14% 및 32% 개선되었습니다. LLM은 타임라인을 50배 더 빠르고, HTR 프로그램의 1/50의 비용으로 작업을 완료할 수 있게 해줍니다.

- **Performance Highlights**: LLMs는 전통적인 HTR 소프트웨어와 LLM들이 생성한 기록들을 수정함으로써 인력 수준의 정확성에 도달하였으며, CER은 1.8%까지, WER은 3.5%에 달했습니다. 기록 과정은 대략 84-93%의 정확도로 수행되었으며, 이는 대부분의 일상적인 사용 예제에 적합합니다. 결과적으로 LLM을 활용한 HTR은 역사적 손글씨 문서의 대량 기록 프로젝트 및 개별 기록을 다루는 역사학자에게 접근 가능하고 효율적인 방법을 제시합니다.



### Fed-EC: Bandwidth-Efficient Clustering-Based Federated Learning For Autonomous Visual Robot Navigation (https://arxiv.org/abs/2411.04112)
- **What's New**: 본 논문에서는 로봇이 다양한 야외 환경에서 자율적으로 탐색할 수 있도록 지원하는 클러스터링 기반의 연합 학습 시스템인 Federated-EmbedCluster(Fed-EC)를 제안합니다. 기존의 연합 학습 방법들은 단일 글로벌 모델을 학습하여 모든 로봇에 적용하는 방식이지만, 본 연구에서는 로봇의 환경에 따라 비슷한 데이터 분포를 가진 로봇들로 클러스터를 구성하여 각 클러스터에 맞춤화된 모델을 학습합니다.

- **Technical Details**: Fed-EC는 로봇의 로컬 데이터셋 간 비슷한 점을 살펴서 클러스터를 형성하고, 각 클러스터 내부에서 IID(Independent and Identically Distributed) 성격을 모방하는 공통의 모델을 학습합니다. 클러스터는 참여하는 로봇들 간의 유사성을 기반으로 공통적으로 구성되며, 새로운 로봇이 클러스터에 합류할 수 있는 전이 가능성을 갖추고 있습니다. Fed-EC는 추가 통신 비용 없이 평균 임베딩 벡터를 공유함으로써 통신 효율성을 극대화합니다.

- **Performance Highlights**: 실제 야외 환경에서의 다양한 실험을 통해 Fed-EC가 통신 크기를 23배 줄이면서 센트럴라이즈드(Centralized) 학습의 성능을 유지할 수 있음을 입증했습니다. 또한, 로컬 학습보다 우수한 성능을 보였으며, 각 클러스터를 위한 개인화된 FL 모델을 학습함으로써 모든 로봇을 위한 단일 글로벌 모델보다 더 나은 성능을 구현할 수 있음을 보여주었습니다.



### Multi-branch Spatio-Temporal Graph Neural Network For Efficient Ice Layer Thickness Prediction (https://arxiv.org/abs/2411.04055)
- **What's New**: 본 논문은 다중 분기 구조의 시공간 그래프 신경망을 개발하여, 얼음층 두께 정보를 활용해 깊은 얼음층의 두께를 예측하는 데 초점을 맞추고 있습니다. 기존의 퓨즈된 시공간 그래프 신경망과 비교했을 때, 제안된 네트워크는 효율성과 정확성에서 일관되게 우수한 성능을 보입니다.

- **Technical Details**: 제안된 네트워크는 GraphSAGE 프레임워크를 활용하여 시공간 특성을 학습하고, 시간적 변화를 포착하기 위해 시간적 컨볼루션(temporal convolution) 작업을 수행합니다. 다양한 작업에 특화된 네트워크의 다양한 부분이 별도의 분기로 구성되어 있습니다.

- **Performance Highlights**: 제안된 다중 분기 네트워크는 기존의 메서드와 비교하여 효율성과 정확성 모두에서 일관되게 더 나은 성능을 보였습니다. 특히, LSTM 구조 대신 도입된 게이티드 시간적 컨볼루션 블록이 효율성을 높이는 데 기여했습니다.



### Synomaly Noise and Multi-Stage Diffusion: A Novel Approach for Unsupervised Anomaly Detection in Ultrasound Imaging (https://arxiv.org/abs/2411.04004)
- **What's New**: 이 연구는 합성 이상(Synomaly) 노이즈 기능과 다단계 확산 프로세스를 포함한 새로운 비지도 이상 탐지 프레임워크를 제안합니다. 이는 훈련 과정에서 건강한 이미지에 합성 이상을 추가하여, 모델이 이상 제거를 효과적으로 학습할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 단순히 건강한 이미지로만 훈련되어 이상 샘플이나 픽셀 단위의 주석이 필요하지 않습니다. 다단계 확산 프로세스는 노이즈가 적고 세부 사항을 보존하면서 이상이 없는 이미지를 다듬는데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존의 최첨단 비지도 이상 탐지 방법보다 우수한 성능을 보여주었으며, 특히 초음파(US) 데이터셋에서 완전 감독 세분화 모델에 버금가는 성과를 거두었습니다.



### ET-SEED: Efficient Trajectory-Level SE(3) Equivariant Diffusion Policy (https://arxiv.org/abs/2411.03990)
Comments:
          Accept to CoRL 2024 Workshop on X-Embodiment Robot Learning

- **What's New**: 이 논문에서는 로봇 조작 작업을 위한 효율적인 경로 수준 (trajectory-level) SE(3) 등가성 (equivariance) 확산 모델인 ET-SEED를 제안하고 있습니다. 이 모델은 샘플 효율성을 개선하고 훈련 난이도를 감소시키며, 복잡한 조작 작업에서의 일반화 능력을 향상시킵니다.

- **Technical Details**: ET-SEED는 SE(3) 군 내에서 정의된 등가 확산 정책으로, 조작 경로를 생성하는 과정에서 적어도 하나의 등가 전이(Eqvariant Transition)만으로도 충분하다는 이론적 확장을 통해 훈련 효율성을 개선했습니다. 또한, SE(3) 다양체에서의 확산 과정을 통합하여 설계되었습니다.

- **Performance Highlights**: 실험 결과, ET-SEED는 데이터 효율성, 조작 능숙도, 공간 일반화 능력에서 최첨단 (SOTA) 방법들을 초월하는 성과를 보여주었으며, 단지 20개의 시연 경로로도 보지 못한 시나리오까지 일반화할 수 있음을 입증했습니다.



### MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba (https://arxiv.org/abs/2411.03855)
- **What's New**: 본 논문에서는 Mamba라는 새로운 State Space Model (SSM) 기반 모델에 대한 Parameter-efficient fine-tuning (PEFT) 방법들의 탐색적 분석을 진행하였습니다. 기존의 Transformer 기반 모델들에 비해 Mamba 모델에서의 PEFT의 효율성과 효과를 강조하며, 신규 Mamba 특화 PEFT 방법들을 제안하였습니다.

- **Technical Details**: Mamba는 시간을 선형적으로 처리할 수 있는 모델로, 기존 Transformer의 계산 복잡성을 극복하며 긴 시퀀스를 효율적으로 처리합니다. 본 연구에서는 Mamba 아키텍처에 적합하도록 기존 PEFT 방법들을 수정하고, Mamba에 특화된 새로운 PEFT 방법들을 제안하여 성능을 극대화하는 방법을 제안합니다. 실험에서는 7개의 주요 PEFT 방법과 총 20개의 파생 변형을 벤치마킹하였습니다.

- **Performance Highlights**: Mamba에서의 PEFT는 Transformer보다 더 효과적이며, 여러 PEFT 방법들을 조합하여 성능을 향상시킬 수 있음을 보여주었습니다. 본 논문은 PEFT 방법의 조합을 효율적으로 탐색하는 기술을 제안하고, 단순한 높은 성능의 방법 조합만으로는 충분하지 않다는 것을 밝혔습니다.



### Sub-DM:Subspace Diffusion Model with Orthogonal Decomposition for MRI Reconstruction (https://arxiv.org/abs/2411.03758)
Comments:
          10 pages, 11 figures

- **What's New**: 이번 논문에서는 MRI 재구성에서의 차별화된 성과를 바탕으로, 기존의 확산 모델(diffusion model) 기반 접근 방식의 한계를 극복하기 위한 새로운 방법인 서브스페이스 확산 모델(subspace diffusion model)인 Sub-DM을 소개합니다.

- **Technical Details**: Sub-DM은 k-공간(k-space) 데이터 분포가 노이즈 방향으로 진화할 때 확산 과정을 서브스페이스에 투영하여 제한하는 방법입니다. 이 모델은 고차원 k-공간 데이터의 복잡성을 피할 수 있는 효과적인 방법을 제공합니다. 또한, 웨이브렛 변환(wavelet transform)에 기반한 직교 분해(orthogonal decomposition) 전략은 일반 확산 과정에서 서브스페이스로의 변환 중 정보 손실을 방지합니다.

- **Performance Highlights**: 데이터 세트에 대한 포괄적인 실험 결과, Sub-DM이 최첨단(state of-the-art) 방법들과 비교했을 때 재구성 속도 및 품질 면에서 우수성을 확연히 보였습니다.



### Deferred Poisoning: Making the Model More Vulnerable via Hessian Singularization (https://arxiv.org/abs/2411.03752)
- **What's New**: 최근 연구에서 기존의 데이터 오염 공격이 그동안 알려진 것보다 덜 위협적일 수 있음을 밝히고, 새로운 유형의 공격인 Deferred Poisoning Attack (DPA)을 소개했습니다. 이 공격은 훈련 및 검증 단계 동안 모델이 정상 작동하도록 하면서도 배포 단계에서 모델의 강건성을 크게 약화시키는 방식입니다.

- **Technical Details**: DPA는 손상된 데이터셋으로 훈련된 모델이 정상적인 데이터셋에서와 비슷한 성능을 발휘하도록 강요함으로써 도움이 됩니다. 또한, 각 샘플 주변에서 발생하는 로컬 곡률을 크게 확대하여 손상된 모델이 작은 변동에 대해 민감하게 반응하도록 합니다. 이 과정에서 Singularization Regularization 항을 통해 모델이 최적점에서 특이한 Hessian 정보를 갖도록 설계합니다.

- **Performance Highlights**: DPA는 기존 데이터 오염 방법보다 훨씬 낮은 공격 비용을 수반하면서도 우수한 전이성과 강건성을 보여줍니다. 실험을 통해 검증된 모델은 자연 잡음으로 인한 새로운 시나리오에서도 이 공격에 대해 더욱 취약해질 수 있음을 확인했습니다.



### NeurIPS 2023 Competition: Privacy Preserving Federated Learning Document VQA (https://arxiv.org/abs/2411.03730)
Comments:
          27 pages, 6 figures

- **What's New**: 본 논문은 Privacy Preserving Federated Learning Document VQA (PFL-DocVQA) 대회의 내용을 다루고 있으며, 커뮤니티의 도전 과제로서 실생활 사용 사례인 인보이스 처리에 대한 개인 정보 보호 및 통신 효율성 높은 솔루션을 개발하는 것을 목표로 하였습니다.

- **Technical Details**: 대회는 실제 인보이스 문서 데이터셋과 함께 질문 및 답변을 제공하며, 정보 추출 및 문서 이미지에 대한 추론을 요구합니다. 참가자들은 사전 훈련된 문서 시각적 질문 답변 모델을 조정하여 연합 학습 환경을 구축하였으며, 개인 정보 보호를 위한 차별적 개인 정보 보호(differential privacy, DP)를 적용하였습니다. 이 대회는 문서 분석, 개인 정보 보호 및 연합 학습 자료의 교류를 위한 새로운 테스트베드 역할을 하였습니다.

- **Performance Highlights**: 대회 참가자들은 통신 비용을 줄이면서도 최소 효용 기준을 유지하는 방법과 각 문서 제공자의 모든 정보를 보호하기 위한 차별적 개인 정보 보호 방법을 제안하였습니다. 대회 분석은 향후 개인 정보 중심의 연합 학습 도전을 성공적으로 실행하는 데에 대한 모범 사례와 권장 사항을 제공합니다.



### Zero-shot Dynamic MRI Reconstruction with Global-to-local Diffusion Mod (https://arxiv.org/abs/2411.03723)
Comments:
          11 pages, 9 figures

- **What's New**: 최근 확산 모델(Diffusion models)이 자기 공명 영상(MRI) 데이터의 생성 및 재구성에서 상당한 발전을 보여주었습니다. 이러한 모델은 비동기적 데이터 처리를 통해 잡음을 줄이고, 생성 모델로서의 가능성을 강조하고 있습니다. 하지만 동적 MRI에 대한 적용은 아직 상대적으로 탐구되지 않았습니다.

- **Technical Details**: 이 연구에서는 시간 간헐 잡음 수집 방식을 기반으로 한 동적 MRI 재구성 방법인 글로벌-투-로컬 확산 모델(Global-to-local Diffusion Model)을 제안합니다. 인접한 시간 프레임에서 샘플링되지 않은 k-space 데이터를 병합하여 완전 인코딩된 전체 해상도 기준 데이터를 구성하는 방식으로 진행됩니다. 글로벌 및 로컬 모델을 위한 두 개의 훈련 데이터 세트를 생성하며, 글로벌-투-로컬 확산 프레임워크는 글로벌 정보와 로컬 이미지 세부 정보를 번갈아 최적화하는 과정을 포함합니다.

- **Performance Highlights**: 제안된 방법은 잡음 감소 및 세부 정보 보존 측면에서 뛰어난 성능을 보여주며, 감독 방식(supervised approaches)과 비교했을 때 재구성 품질이 유사한 수준에 도달합니다.



### Cross Feature Fusion of Fundus Image and Generated Lesion Map for Referable Diabetic Retinopathy Classification (https://arxiv.org/abs/2411.03618)
Comments:
          ACCV 2024 accepted

- **What's New**: 이 연구는 당뇨병성 망막병증(Diabetic Retinopathy, DR)의 조기 탐지 및 진단을 위해 향상된 cross-learning 방법론을 제안합니다. 특히, Swin U-Net 아키텍처를 활용하여 DR의 병변 지도를 세분화하고, 이를 통해 정확한 분류를 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 transfer learning과 cross-attention 메커니즘을 결합하여 segmentation과 분류를 수행합니다. Swin U-Net을 활용하여 이미지에서 병변 지도를 추출하고, 이를 기반으로 classify 모델이 학습됩니다. cross-attention 메커니즘은 이미지의 주요 영역을 강조하여 드러내는 역할을 합니다.

- **Performance Highlights**: 제안한 방법의 분류 정확도는 94.6%로, 기존의 최첨단 기법들을 4.4% 초과하였습니다. 이는 DR 분류의 정확성과 효율성을 크게 향상시키는 성과로, 임상적 워크플로우에 쉽게 통합할 수 있도록 목표하고 있습니다.



### ADMIRE: a locally adaptive single-image, non-uniformity correction and denoising algorithm: application to uncooled IR camera (https://arxiv.org/abs/2411.03615)
- **What's New**: 이 논문은 비냉각 적외선 이미지에서 비균일성(non-uniformity, NU)과 잡음을 수정하는 새로운 방법을 제안합니다. 이 방법은 정적 이미지에서 작동하며, 등록(registration), 카메라 모션(camera motion) 및 NU 모델이 필요 없습니다.

- **Technical Details**: 제안된 방법은 자동 로컬 적응 대비 조정(contrast adjustment) 및 최첨단 이미지 잡음 제거(image denoising) 방법을 포함하는 하이브리드(hybrid) 방식입니다. 이 방법은 단 하나의 이미지로 비선형 NU와 잡음을 효율적으로 수정할 수 있습니다. 또한, motion compensation(모션 보상)이나 테스트 패턴 테스트가 필요하지 않으며 '유령 아티팩트(ghost artifact)'를 생성하지 않습니다.

- **Performance Highlights**: 제안된 방법은 실시간 원시(raw) 및 시뮬레이션한 NU 적외선 이미지에서 전체 변동(total variation) 기반 방법과 비교되었습니다. 이 접근 방식의 장점은 단순성과 낮은 계산 비용으로, 제조사가 교정(calibration) 작업을 할 필요 없이도 지속적인 이미지 흐름을 유지할 수 있다는 점입니다.



### LCP-Fusion: A Neural Implicit SLAM with Enhanced Local Constraints and Computable Prior (https://arxiv.org/abs/2411.03610)
Comments:
          Accepted by 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: LCP-Fusion이라는 새로운 신경 임플리시 SLAM 시스템은 메모리 사용량과 맵핑 해상도를 균형 있게 조정하고, 알려지지 않은 장면 경계에서의 드리프트 문제를 완화하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 시스템은 하이브리드 장면 표현을 사용하여 희소 복셀 옥트리(Sparse Voxel Octree) 구조를 통해 기능 그리드와 SDF(Signed Distance Function) 사전 정보를 포함합니다. 또한 시각적 중복을 기반으로 하는 새로운 슬라이딩 윈도우 선택 전략과 상대 자세를 제한하는 실용적인 왜곡 손실을 제안하여 로컬 제약을 강화합니다.

- **Performance Highlights**: LCP-Fusion은 ScanNet과 같은 복잡한 실제 장면에서 기존 RGB-D 임플리시 SLAM에 비해 더 나은 로컬라이제이션 정확도와 재구성 일관성을 달성했습니다. 이는 더 적은 반복 연산에서도 뛰어난 성능을 제공하여 ROBUSTNESS를 높였습니다.



### Towards Personalized Federated Learning via Comprehensive Knowledge Distillation (https://arxiv.org/abs/2411.03569)
Comments:
          Accepted by IEEE SMC 2024

- **What's New**: 최근 개인화 연합 학습(Personalized Federated Learning, PFL) 접근 방식이 데이터 이질성 문제를 해결하기 위해 발전하였습니다. 본 논문에서는 모델의 개인화를 중시하면서도 일반화 성능을 유지할 수 있는 새로운 방법인 FedCKD를 제안합니다.

- **Technical Details**: FedCKD는 글로벌 모델(global model)과 역사적 모델(historical model)을 교육자로 사용하고, 로컬 모델(local model)을 학생으로 설정하여 지식 증류(knowledge distillation)를 수행합니다. 이 과정에서 글로벌 모델은 서버 집합의 마지막 라운드의 집합 모델을 나타내며, 역사적 모델은 클라이언트 훈련의 마지막 라운드에서의 로컬 모델을 나타냅니다. 이러한 구조를 통해 글로벌 일반화 지식 및 역사적 개인화 지식을 효과적으로 로컬 모델에 전달합니다.

- **Performance Highlights**: 실험 결과, FedCKD는 기존의 최첨단 방법들을 초월하여 모델의 성능을 유의미하게 향상시키는 것으로 나타났습니다. 특히, 모델의 일반화와 개인화 간의 균형을 유지하면서 치명적 망각(catastrophic forgetting) 문제를 완화하는 데 성공하였습니다.



### The American Sign Language Knowledge Graph: Infusing ASL Models with Linguistic Knowledg (https://arxiv.org/abs/2411.03568)
- **What's New**: 본 연구에서는 American Sign Language (ASL)용 지식 그래프인 ASLKG를 소개하고, 이를 통해 인간의 수어 모양 인식 및 번역 작업의 정확성을 향상시키기 위한 신경 기호 모델을 훈련시키는 방법을 제안합니다. 이 그래프는 12개의 전문가 언어 지식 출처로부터 수집된 지식으로 구성되어 있습니다.

- **Technical Details**: ASLKG는 5802개의 ASL 신호에 대한 7171개의 언어적 사실을 포함하고 있습니다. 이 연구는 신경 기호적 방법(neuro-symbolic methods)을 활용하여 ASL 신호 인식(isolated sign recognition), 알려지지 않은 신호의 의미 특징 예측, YouTube-ASL 비디오의 주제 분류와 같은 세 가지 다운스트림 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 단독 신호 인식에서 91%, 알려지지 않은 신호의 의미 특징 예측에서 14%, YouTube-ASL 비디오 주제 분류에서 36%의 정확도를 달성했습니다. 이러한 성과는 ASLKG를 활용함으로써 가능해졌습니다.



### Enhancing Weakly Supervised Semantic Segmentation for Fibrosis via Controllable Image Generation (https://arxiv.org/abs/2411.03551)
- **What's New**: 이 연구에서는 Fibrotic Lung Disease (FLD)의 효과적인 분석을 위해 새로운 약한 감독 Semantic Segmentation 방법인 DiffSeg를 제안합니다. 이는 고해상도 CT 이미지에서 섬유증(segmentation) 패턴을 식별하는 데 필요한 수작업 레이블링을 줄이는 것을 목표로 합니다.

- **Technical Details**: DiffSeg는 이미지 수준의 주석을 이용하여 픽셀 수준의 섬유증 분할을 생성하는데, 전이 학습된 분류자를 통해 잠재 공간(latent space)을 조정하고, 이를 이용해 건강한 CT 이미지를 섬유증이 있는 이미지로 생성합니다. 또한, DiffSeg는 Denoising Diffusion Implicit Model (DDIM)을 사용하여 섬유증 특징을 가미합니다.

- **Performance Highlights**: 실험 결과, DiffSeg는 기존의 약한 감독 하의 방법들보다 가짜 마스크(pseudo masks)의 정확도를 유의미하게 향상시켰습니다. 수작업 레이블링의 복잡성을 크게 줄이고 생성된 마스크의 일관성을 증대시켰습니다.



### TopoTxR: A topology-guided deep convolutional network for breast parenchyma learning on DCE-MRIs (https://arxiv.org/abs/2411.03464)
Comments:
          22 pages, 8 figures, 8 tables, accepted by Medical Image Analysis ( this https URL )

- **What's New**: 이번 논문에서는 유방 DCE-MRI에서 유방 피질(parenchyma)의 특성을 효과적으로 모델링하기 위한 새로운 방법인 TopoTxR을 제안합니다. 이 방법은 복잡한 조직 구조를 정확하게 추출하고 이를 딥러닝 모델에 통합하여 예측력을 높입니다.

- **Technical Details**: TopoTxR은 수학적 토폴로지(topology) 언어를 사용하여 유방 피질 구조를 추출합니다. 이 방법은 영속 동형체(persistent homology) 이론에 기반하여 1D와 2D 토폴로지 구조를 추출하며, 이를 통해 유방 조직의 생물학적 구조에 기반한 예측을 가능하게 합니다. 또한, 토폴로지 기반의 공간 주의 기법을 도입하여 3D CNN에서 생물학적으로 관련된 복셀(voxel)에 모델의 주의를 집중시킵니다.

- **Performance Highlights**: TopoTxR은 I-SPY 1 데이터셋에서 2.6%의 정확도 증가와 4.6%의 AUC 향상을 보이며, 기존의 최첨단 방법들과 비교하여 뛰어난 예측 성능을 입증했습니다. 실험 결과, 이 모델은 네오아주반 화학요법에 대한 신뢰성 있는 예측을 제공하여 임상에서의 활용 가능성을 보여줍니다.



### BOston Neonatal Brain Injury Data for Hypoxic Ischemic Encephalopathy (BONBID-HIE): II. 2-year Neurocognitive Outcome and NICU Outcom (https://arxiv.org/abs/2411.03456)
Comments:
          Data description for BONBID-HIE 2024 Challenge on MICCAI 2024

- **What's New**: 본 논문에서는 Hypoxic Ischemic Encephalopathy (HIE)에 대한 Boston Neonatal Brain Injury Dataset의 두 번째 릴리스를 소개합니다. 이 데이터셋은 237명의 환자를 포함한 MRI 및 임상 데이터를 제공하며, Massachusetts General Hospital과 Boston Children's Hospital에서 수집되었습니다.

- **Technical Details**: BONBID-HIE 데이터셋은 HIE 예후 바이오마커 개발을 위한 포괄적이고 개방된 소스의 MRI 및 임상 데이터셋입니다. 데이터는 GE 1.5T Signa 스캐너 또는 Siemens 3T TrioTim 및 PrismaFit 스캐너를 사용하여 수집되었으며, Apparent Diffusion Coefficient (ADC) 지도와 NICU(Neonatal Intensive Care Unit) 결과 및 2년 시점의 신경인지적 결과를 포함합니다.

- **Performance Highlights**: 데이터셋은 신생아의 HIE 예후를 예측하는 데 중요한 ADC 및 Z_ADC 정보를 가지고 있으며, 엔트리 마다 임상 변수와 NICU 상태, 2년 신경인지적 결과를 포함하므로, 향후 예후 바이오마커 개발에 기여할 것으로 기대됩니다.



### Solving Trojan Detection Competitions with Linear Weight Classification (https://arxiv.org/abs/2411.03445)
Comments:
          9 pages, 4 Figures

- **What's New**: 이 논문에서는 트로이안 백도어를 탐지하기 위한 새로운 방법을 제안했습니다. 이 접근법은 다양한 데이터셋과 도메인에서 높은 성능을 보입니다.

- **Technical Details**: 제안된 탐지기는 여러 개의 모델 가중치에 대한 이진 분류기를 학습하여 얻어지며, 주요 전처리 단계를 통해 성능을 개선합니다. 전 처리 단계에는 특성 선택, 표준화, 참조 모델 가중치 빼기, 모델 정렬 등이 포함됩니다. 이 기법은 가중치 분석(weight analysis) 탐지에 해당하며, 트리거에 대한 사전 지식 없이도 적용 가능합니다.

- **Performance Highlights**: 본 알고리즘은 Trojan Detection Challenge(TDC22)와 IARPA/NIST TrojAI 프로그램의 다양한 벤치마크에서 평가되었으며, 모델의 정밀한 분류를 통해 청정(clean) 모델과 오염된(poisoned) 모델 간의 구분을 효과적으로 수행했습니다.



### Undermining Image and Text Classification Algorithms Using Adversarial Attacks (https://arxiv.org/abs/2411.03348)
Comments:
          Accepted for presentation at Electronic Imaging Conference 2025

- **What's New**: 이 연구는 Generative Adversarial Networks (GANs)와 Synthetic Minority Oversampling Technique (SMOTE)를 결합하여 텍스트 및 이미지 분류 모델에 대한 새로운 adversarial attack 방법론을 제시합니다.

- **Technical Details**: 연구에서 사용된 방법론은 Fast Gradient Sign Method (FGSM)를 이용한 perturbation 벡터와 GradCAM을 통해 강조된 주요 특징을 결합하여 adversarial 공격을 수행합니다. 이는 Convolutional Neural Network (CNN) 모델을 사용하여 얼굴 인식 시스템에 대한 공격을 포함합니다.

- **Performance Highlights**: 이 실험에서 텍스트 분류 모델의 정확도가 공격 후 20% 감소했으며, 얼굴 인식 정확도는 30% 감소했습니다. 이러한 결과는 공격에 대한 모델의 취약성을 강조하며, 머신러닝 시스템의 신뢰성을 저하시킬 수 있습니다.



### Interpretable Embeddings for Segmentation-Free Single-Cell Analysis in Multiplex Imaging (https://arxiv.org/abs/2411.03341)
Comments:
          5 Pages, 5 Figures, Submitted to ISBI 2025

- **What's New**: 이 논문에서는 세포 분할(segmentation) 알고리즘에 의존하지 않고, 깊은 학습(deep learning) 접근 방식으로 그룹화된 컨볼루션(grouped convolutions)을 활용해 각 이미징 채널에서 해석 가능한 고정밀 특징을 학습함으로써 세포 유형을 식별하는 새로운 방법을 제안합니다. 이를 통해 수동적인 특징 선택 없이도 강력한 세포 유형 식별이 가능해집니다.

- **Technical Details**: 제안된 모델은 NeXt-Channel Block 구조를 이용하여 생물학적 마커를 독립적으로 처리하며, 각 채널이 포함된 특징 공간에서의 기여도를 해석가능하게 만듭니다. 이 모델은 1.8 백만 개의 세포를 포함한 신경모세포종(neuroblastoma) IMC 데이터셋을 기반으로 검증되었으며, 고차원(MI) 데이터에 적합성과 확장성을 보여줍니다.

- **Performance Highlights**: 모델은 UMAP 시각화(t-SNE 또는 UMAP을 사용한 샘플링)를 통해 생물학적으로 유사한 세포들을 효과적으로 그룹짓고, T 세포, B 세포, 종양 세포의 정확한 분류 결과(높은 rediscovery rates)로 확인됩니다. 이는 T 세포 92.05%, B 세포 81.05%, 종양 세포 88.33%, 호중구(granulocytes) 86.23%의 높은 rediscovery rates를 통해 입증되었습니다.



New uploads on arXiv(cs.AI)

### Lexicalization Is All You Need: Examining the Impact of Lexical Knowledge in a Compositional QALD System (https://arxiv.org/abs/2411.03906)
Comments:
          24th International Conference on Knowledge Engineering and Knowledge Management (EKAW 2024), November 26-28, 2024, Amsterdam, The Netherlands

- **What's New**: 이번 논문에서는 Linked Data (QALD)에서 질문 응답에 대한 어휘화(lexicalization)의 영향을 조사하였습니다. 자연어 질문을 SPARQL 쿼리로 해석하는 과정에서 어휘적 간격(lexical gap)을 메우는 것이 주요 도전 과제라는 점을 강조하고, 어휘화가 Q&A 시스템의 성능을 크게 향상시킨다는 주장을 펼쳤습니다.

- **Technical Details**: 우리는 명시적 어휘 지식을 활용하는 조합형 질문 응답 시스템(compositional QA system)을 제안합니다. 이 시스템은 SPARQL 쿼리의 의미를 추론하기 위해 어휘적 지식을 조합적으로 활용하며, QALD-9 데이터셋에서 기존 Q&A 시스템보다 35.8% 향상된 마이크로 F1 점수를 기록하는 성과를 보여주었습니다.

- **Performance Highlights**: LLM(대형 언어 모델)은 어휘적 지식을 활용하는 데 한계가 있어, 어휘적 지식 없이 활용할 때와 큰 차이가 없음을 보여줍니다. 새로운 접근법은 QALD 연구에 대한 새로운 방향을 제시하며, 어휘화와 조합성의 중요성을 강조합니다.



### OML: Open, Monetizable, and Loyal AI (https://arxiv.org/abs/2411.03887)
Comments:
          60 pages, 22 figures

- **What's New**: AI 개발 및 배포의 민주화를 위한 새로운 접근법 OML을 제안합니다. OML(Open, Monetizable, and Loyal AI)은 AI, 블록체인, 암호학을 포함한 학제간 프레임워크를 통해 실현됩니다.

- **Technical Details**: OML은 Trusted Execution Environments (TEE), 완전 동형 암호(fully homomorphic encryption), 기능적 암호(functional encryption) 등과 같은 기존 암호 기술을 활용하여 구현됩니다. 새로운 과학 분야인 AI-native cryptography를 도입하여, AI 데이터 표현의 연속성과 저차원 매니폴드(low-dimensional manifolds)를 이용한 개선된 성능을 목표로 합니다.

- **Performance Highlights**: OML은 모델 핑거프린팅(model fingerprinting)을 사용하여 AI 모델의 무결성과 소유권을 보호하며, 이는 AI 개발의 분산화, 개방성 및 투명성을 달성하는 데 기여합니다.



### Beyond The Rainbow: High Performance Deep Reinforcement Learning On A Desktop PC (https://arxiv.org/abs/2411.03820)
Comments:
          9 main pages, 26 total. Currently under review at ICLR

- **What's New**: 이 논문에서는 Rainbow DQN의 많은 독립적인 향상을 통합하여 BTR(Beyond The Rainbow)이라는 새로운 알고리즘을 제시합니다. 이 알고리즘은 컴퓨팅 리소스가 제한된 환경에서도 뛰어난 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: BTR은 Rainbow DQN에 비해 6개의 알고리즘 개선 사항을 결합하여 새로운 최첨단 성능을 달성하였습니다. 이 개선 사항들은 에이전트의 훈련에 필요한 계산 비용을 줄이기 위한 것입니다. BTR은 데스크톱 PC에서 200백만 Atari 프레임을 12시간 이내에 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: Atari-60에서 BTR은 인간 정규화된 Interquartile Mean(IQM) 7.4를 기록하며 새로운 기록을 세웠습니다. 또한, BTR을 사용해 Super Mario Galaxy, Mario Kart 등 복잡한 3D 게임을 성공적으로 훈련시킬 수 있음을 보여주었습니다.



### From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning (https://arxiv.org/abs/2411.03817)
- **What's New**: 이 논문에서는 StepAgent를 소개합니다. 이는 단계별 보상(Step-wise reward)을 활용하여 에이전트의 강화학습(Reinforcement Learning) 프로세스를 최적화합니다.

- **Technical Details**: StepAgent는 초보자에서 전문가로의 이론(Novice-to-Expert Theory)을 바탕으로 하여, 전문가와 에이전트의 행동을 비교하여 세분화된 최적화를 위한 중간 보상을 자동으로 생성합니다. 또한, 암묵적 보상(Implicit Reward) 및 역 강화학습(Inverse Reinforcement Learning) 기법을 제안하여 에이전트의 반성과 정책 조정을 촉진합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, StepAgent는 기존의 기준 방법들보다 더 우수한 성능을 보였습니다.



### MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogu (https://arxiv.org/abs/2411.03814)
- **What's New**: 이번 연구에서는 Multi-round Dialogue Jailbreak ( MRJ ) 에이전트를 제안하여, LLMs (Large Language Models) 에 대한 멀티 라운드 대화 공격의 위험을 식별하고 완화하기 위한 새로운 전략을 소개합니다.

- **Technical Details**: 새로운 공격 메커니즘은 Heuristic Search (휴리스틱 검색) 과정을 기반으로 하며, 악의적인 쿼리에서 시작해 점진적으로 민감한 주제로 나아가도록 설계되었습니다. 이 과정에서는 정보 기반 제어 전략과 심리 유도 전략이 사용됩니다.

- **Performance Highlights**: 제안된 방법은 기존의 단일 및 다중 라운드 공격 방법보다 우수한 공격 성공률을 기록하며, Closed-source 및 Open-source 모델에서 모두 높은 효과를 나타냈습니다. 이를 통해 다양한 모델 및 시나리오에 적용 가능한 보다 일반적인 공격 전략을 선보였습니다.



### Navigating the landscape of multimodal AI in medicine: a scoping review on technical challenges and clinical applications (https://arxiv.org/abs/2411.03782)
Comments:
          28 pages

- **What's New**: 최근 헬스케어에서의 기술 발전으로 인해 환자 데이터의 양과 다양성이 폭발적으로 증가하고 있으며, 이는 multimodal AI의 필요성을 더욱 부각시키고 있습니다. 본 논문은 2018년에서 2024년 사이에 발표된 432개의 논문을 분석하여, 다양한 의료 분야에서의 multimodal AI의 발전 현황을 종합적으로 검토합니다.

- **Technical Details**: 논문에서는 deep learning 기반의 multimodal AI 적용 사례를 다루며, 다양한 아키텍처 접근 방식(architectural approaches)과 데이터 융합 전략(fusion strategies)을 분석합니다. 또한 교차부서간 협력과 이질적인 데이터 특성(homogeneous data characteristics)과 같은 기술적 및 실질적 과제들을 제시하고, 상업적으로 이용 가능한 다양한 multimodal AI 모델들을 소개합니다.

- **Performance Highlights**: 연구 결과에 따르면 multimodal AI 모델은 unimodal 모델에 비해 평균 6.2%의 AUC 개선 효과를 보였으며, 이는 임상 의사결정에 있어 멀티모달 데이터 통합의 장점을 보여줍니다. 다만, 데이터의 불완전성, 다양한 의료 분야 간의 협력 부족과 같은 도전 과제가 여전히 남아있습니다.



### Automating Exploratory Proteomics Research via Language Models (https://arxiv.org/abs/2411.03743)
- **What's New**: 이 논문에서는 PROTEUS라는 혁신적인 자동화 시스템을 소개합니다. 이 시스템은 원시 단백질체학 데이터(proteomics data)로부터 과학적 발견을 자동으로 수행하며, 대규모 언어 모델(LLM)을 활용해 분석 워크플로우를 계획하고 실행합니다.

- **Technical Details**: PROTEUS는 복잡한 프로테오믹스 분석을 위한 자동화된 접근 방식을 제공합니다. 시스템은 12개의 다양한 단백질체 데이터셋을 분석하였으며, 총 191개의 과학적 가설(hypothesis)을 생성했습니다. 분석 과정은 LLM이 주도하며, 데이터 의존적인 분석 절차를 수행하여 결과를 해석합니다. 또한 자동으로 분석 후 개선되는 반복적(refinement) 절차를 갖추고 있습니다.

- **Performance Highlights**: PROTEUS는 전문가의 평가와 함께 자동 평가를 통해 생성된 가설의 신뢰성과 논리적 일관성을 검증하였으며, 기존 문헌과 잘 부합하는 높은 품질의 가설을 일관되게 생성할 수 있음을 입증했습니다. 이 시스템은 단백질체학 연구에서 과학적 발견의 속도를 상당히 가속화할 가능성이 있습니다.



### Policy Aggregation (https://arxiv.org/abs/2411.03651)
- **What's New**: 본 논문에서는 상이한 보상 함수(reward function)와 최적 정책(optimal policy)을 가진 여러 개인(individuals) 간의 AI 가치 정렬(value alignment) 문제를 정책 집계(policy aggregation) 문제로 형식화했습니다. 특히 사회 선택 이론(social choice theory)을 활용하여 효과적인 집계 방법을 제안합니다.

- **Technical Details**: 이 논문은 각 대리인이 MDP(Markov Decision Process) 환경에서 고유의 최적 정책을 가지고 있다는 기본 전제를 바탕으로 그들의 정책을 어떻게 집계해야 할지를 다룹니다. 정책 집계 문제를 해결하기 위해, 우리는 보르다 집계(Borda count), 비례 거부 핵(proportional veto core), 정량 공정성(quantile fairness) 등 다양한 사회 선택 규칙들을 제안하고, 이들이 실제로 어떻게 활용될 수 있는지를 보여주었습니다.

- **Performance Highlights**: 실험을 통해 정량 공정성(quantile fairness)이 특히 매력적인 특성을 보임이 확인되었습니다. 또한, 우리의 접근 방식이 보상에 대한 민감성이 있는 사회 복지 측정 기능을 최적화하는 규칙들보다 우수하다는 것을 보여주었습니다.



### RTify: Aligning Deep Neural Networks with Human Behavioral Decisions (https://arxiv.org/abs/2411.03630)
Comments:
          Published at NeurIPS 2024

- **What's New**: 현 토대 위에, 새로운 RTify 모듈을 도입하여 순환 신경망(RNN)이 인간의 반응 시간(RT)에 맞춰 역동적으로 증거를 누적할 수 있도록 하였습니다. 이 접근법은 단순한 인간 RT 데이터에 대한 감독 없이도 효과적으로 사용될 수 있습니다.

- **Technical Details**: RTify 모듈은 RNN의 재발 단계 최적화를 가능하게 하여 인간 RT를 계량할 수 있으며, 비선형적으로 증거를 누적하고 역전파(Back-propagation) 방식으로 학습됩니다. 이 모델은 주어진 시간 내에 통합된 증거가 학습된 임계값을 초과할 때 결정을 내립니다.

- **Performance Highlights**: 다양한 심리물리 실험을 통해 우리의 방법이 인간 RT 데이터와의 적합성을 높이고, 기존 모델보다 일관되게 우수한 성능을 보여 주었습니다. 또한, 이 프레임워크는 Wong-Wang 결정 모델을 확장하여 인간의 반응 시간을 잘 적합시키는 생물학적으로 그럴듯한 RNN 모듈을 생성하는 데 성공했습니다.



### Fully Hyperbolic Rotation for Knowledge Graph Embedding (https://arxiv.org/abs/2411.03622)
- **What's New**: 본 논문에서는 기존의 하이퍼볼릭 회전 모델의 한계를 극복하기 위해 완전 하이퍼볼릭 회전 모델(FHRE)을 제안합니다. 이 모델은 특징 변환을 위한 매핑 방식 대신 로렌츠 모델을 이용하여 하이퍼볼릭 공간에서 직접 정의됩니다.

- **Technical Details**: 제안하는 FHRE 모델은 지식 그래프를 구성하는 각 관계를 헤드 엔티티에서 테일 엔티티로의 로렌츠 회전으로 간주합니다. 또한, triplet의 타당성을 측정하기 위한 scoring function으로 로렌츠 거리 버전을 사용합니다.

- **Performance Highlights**: 표준 지식 그래프 완성 벤치마크인 FB15k-237과 WN18RR에서 경쟁력 있는 성과를 달성했으며, 다양하고 도전적인 데이터셋인 CoDEx-s와 CoDEx-m에서 최첨단 성능을 기록했습니다.



### LLM Generated Distribution-Based Prediction of US Electoral Results, Part I (https://arxiv.org/abs/2411.03486)
Comments:
          17 pages, 10 Figures, Pre-print

- **What's New**: 본 논문에서는 Large Language Models (LLMs)을 예측 도구로 사용하는 새로운 접근법인 distribution-based prediction을 소개합니다. 이 방법은 출력 토큰 확률을 모델이 학습한 세계를 나타내는 분포로 해석하여 알고리즘적 신뢰성을 분석하는 대안을 제공합니다.

- **Technical Details**: distribution-based prediction 접근법은 개별 인물 시뮬레이션을 우회하여 LLM을 비개인 예측 모델로 활용합니다. 주어진 주(state) 내에서 각 후보에 대한 투표 비율을 예측하기 위해 모델에 직접 프롬프트를 제공하고, 출력 확률을 해당 모델이 내재한 지식을 나타내는 분포로 처리합니다. 이는 후보자별로 주마다 유권자 비율의 분포를 생성하는 과정으로 진행됩니다.

- **Performance Highlights**: 이 방법을 통해 미국 대선에서 후보별 주 투표 비율을 예측하며, LLM의 정확성과 알고리즘적 신뢰성을 평가할 수 있음을 보여줍니다. 또한, 이 개념은 다양한 도메인에서 LLM 기반 예측의 신뢰성과 투명성을 높이는 데 상당한 의미를 가집니다.



### Watson: A Cognitive Observability Framework for the Reasoning of Foundation Model-Powered Agents (https://arxiv.org/abs/2411.03455)
- **What's New**: 본 논문은 FM 기반 소프트웨어의 복잡성과 관련하여 전통적인 운영 가시성의 한계를 강조하고, 이러한 혁신적인 시스템을 위한 새로운 유형의 가시성인 인지 가시성(cognitive observability)을 제안합니다.

- **Technical Details**: 저자들은 에이전트의 암묵적 추론 과정에 대한 인지 가시성을 제공하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 특히 에이전트가 예상치 못한 행동을 하거나 오류를 만날 때, 해당 행동의 관찰과 이해를 용이하게 만들어줍니다.

- **Performance Highlights**: 사례 연구 AutoCodeRover를 통해, 이 프레임워크가 Agentware의 디버그 가능성을 향상시키고, 궁극적으로 Agentware의 능력을 증진시키는 데 효과적임을 입증합니다.



### RuAG: Learned-rule-augmented Generation for Large Language Models (https://arxiv.org/abs/2411.03349)
- **What's New**: 본 논문에서는 LLM의 추론 능력을 향상시키기 위해 대량의 offline 데이터를 해석 가능한 1차 논리 규칙으로 자동 변환하고 이를 LLM에 주입하는 새로운 프레임워크인 RuAG를 제안합니다. RuAG는 LLM의 상식에 기반하여 목표 및 본체 술어를 자동 정의하고, Monte Carlo Tree Search를 통해 데이터에서 논리 규칙을 효율적으로 발견합니다.

- **Technical Details**: RuAG는 세 가지 주요 단계로 구성됩니다. 첫째, LLM 기반의 논리 규칙 검색 공식을 통해 LLM이 상식에 기반하여 목표 술어와 본체 술어를 정의합니다. 둘째, MCTS를 이용하여 논리 규칙 검색을 수행하며, 복합적인 검색 공간을 효율적으로 탐색하여 구조화된 1차 논리 규칙을 생성합니다. 셋째, 생성된 논리 규칙을 자연어로 변환하여 LLM의 프롬프트에 주입합니다.

- **Performance Highlights**: RuAG는 자연어 처리, 시계열, 의사 결정 및 산업 과제를 포함한 다양한 공개 및 민간 산업 과제에서 LLM의 능력을 향상시키는 데 효과적임을 입증했습니다. 이 프레임워크는 SFT, ICL, RAG 및 KG 기반 방법과 같은 기존 방법의 한계를 극복하여, LLM의 추론 및 작업 성능을 최소한의 수작업 개입으로 개선합니다.



### Will Trump Win in 2024? Predicting the US Presidential Election via Multi-step Reasoning with Large Language Models (https://arxiv.org/abs/2411.03321)
Comments:
          This research is ongoing work. Xiyang Hu and Yue Zhao are the corresponding authors

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 선거 예측 능력을 탐구하고, 그에 대한 혁신적인 접근법을 제시합니다. 특히 정치적 분석을 위한 다단계 추론 프레임워크를 제공하여 실제 데이터를 기반으로 한 예측을 향상시킵니다.

- **Technical Details**: 우리는 Sync synthetic data generation framework를 사용하여 개별 유권자의 인구통계 및 행동 프로필을 재구성하고, 2016년 및 2020년 미국 국가선거조사(ANES)의 실제 데이터를 통해 프레임워크를 검증합니다. Chain of Thought prompting에서 영감을 받아, 이 접근 방식은 정치적 맥락의 변화에 맞춰 모델을 조정하며, 인구 통계학적 정보 및 이념적 요인과 같은 다양한 요소들을 체계적으로 통합합니다.

- **Performance Highlights**: 이 모델은 2024년 미국 대통령 선거 결과를 사전에 예측하는 데 성공하여, 보이지 않는 정치적 데이터에 대해 LLMs가 어떻게 적응할 수 있는지를 증명합니다. 최종 파이프라인은 예측 정확성과 실제 결과와의 정합성 모두에서 상당한 개선을 보여줍니다.



### Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? (https://arxiv.org/abs/2411.04118)
Comments:
          Accepted to EMNLP 2024 Main Conference as Long Paper (Oral)

- **What's New**: 이 논문에서는 의학 응용을 위해 개발된 여러 모델들의 성능을 평가하고, 일반 도메인 모델들과의 비교를 통해 DAPT(Domain-Adaptive Pretraining)의 효과에 대한 의문을 제기합니다.

- **Technical Details**: 연구팀은 7개의 의학 LLM과 2개의 VLM을 선택하여, 각 의학 모델을 해당하는 일반 모델과 직접 비교했습니다. 각 모델에 대해 최적의 프롬프트를 독립적으로 선정하고, 통계적 불확실성을 고려한 후 성능을 평가했습니다.

- **Performance Highlights**: 의학 LLMs는 3-shot 설정에서 12.1%의 경우에만 베이스 모델을 초과하여, 나머지 87.9%의 경우에서는 동점이거나 그보다 성능이 떨어진다는 점을 발견했습니다. 이 결과는 DAPT가 의학 분야에서 항상 성능 향상을 보장하지 않음을 시사합니다.



### Fed-EC: Bandwidth-Efficient Clustering-Based Federated Learning For Autonomous Visual Robot Navigation (https://arxiv.org/abs/2411.04112)
- **What's New**: 본 논문에서는 로봇이 다양한 야외 환경에서 자율적으로 탐색할 수 있도록 지원하는 클러스터링 기반의 연합 학습 시스템인 Federated-EmbedCluster(Fed-EC)를 제안합니다. 기존의 연합 학습 방법들은 단일 글로벌 모델을 학습하여 모든 로봇에 적용하는 방식이지만, 본 연구에서는 로봇의 환경에 따라 비슷한 데이터 분포를 가진 로봇들로 클러스터를 구성하여 각 클러스터에 맞춤화된 모델을 학습합니다.

- **Technical Details**: Fed-EC는 로봇의 로컬 데이터셋 간 비슷한 점을 살펴서 클러스터를 형성하고, 각 클러스터 내부에서 IID(Independent and Identically Distributed) 성격을 모방하는 공통의 모델을 학습합니다. 클러스터는 참여하는 로봇들 간의 유사성을 기반으로 공통적으로 구성되며, 새로운 로봇이 클러스터에 합류할 수 있는 전이 가능성을 갖추고 있습니다. Fed-EC는 추가 통신 비용 없이 평균 임베딩 벡터를 공유함으로써 통신 효율성을 극대화합니다.

- **Performance Highlights**: 실제 야외 환경에서의 다양한 실험을 통해 Fed-EC가 통신 크기를 23배 줄이면서 센트럴라이즈드(Centralized) 학습의 성능을 유지할 수 있음을 입증했습니다. 또한, 로컬 학습보다 우수한 성능을 보였으며, 각 클러스터를 위한 개인화된 FL 모델을 학습함으로써 모든 로봇을 위한 단일 글로벌 모델보다 더 나은 성능을 구현할 수 있음을 보여주었습니다.



### Self-Consistency Preference Optimization (https://arxiv.org/abs/2411.04109)
Comments:
          16 pages, 3 figures

- **What's New**: 이번 논문에서는 Self-consistency Preference Optimization (ScPO)라는 방법을 제안하여 모델이 비지도 학습 문제에 대해 일관된 답을 학습하도록 하는 새로운 접근을 소개합니다. 이 방법은 모델 훈련 과정에서 자가 일관성(self-consistency) 개념을 활용하여, 복잡한 문제 해결 작업의 성능을 향상시킵니다.

- **Technical Details**: ScPO는 비지도 학습 단계에서 모델이 생성한 문제와 쿼리를 사용하여 복잡한 문제 해결 작업을 수행합니다. 이 방법의 과정은 (i) 모델이 생성한 쿼리 선택, (ii) 가장 자가 일관성이 높은 응답(우승자)과 가장 낮은 응답(패자)에 대한 선호 쌍을 주석 달기, (iii) 모델의 선호 쌍에 대한 신뢰도에 따라 가중치가 조정된 손실 함수 최적화로 구성됩니다. 이 논문은 또한 라벨이 부여된 인스턴스와 미라벨 인스턴스에서 공동으로 LLM을 훈련하는 반지도 변형도 제안합니다.

- **Performance Highlights**: Llama-3 8B 모델을 사용한 실험 결과, ScPO는 GSM8K에서 22.74% 그리고 MATH에서 5.26%의 제로샷 정확도를 향상시키며, 감독 학습 방식의 성능에 가까운 결과를 도출했습니다. 또한 ScPO를 사용하여 ZebraLogic의 난해한 논리 퍼즐에서 6.5%의 정확도 향상을 보여주었으며, 이는 Llama-3 70B 및 Gemma-2 27B와 같은 더 큰 LLM들을 초월하는 성능입니다.



### How Transformers Solve Propositional Logic Problems: A Mechanistic Analysis (https://arxiv.org/abs/2411.04105)
- **What's New**: 이번 연구에서는 작은 Transformer 모델이 복잡한 논리 문제를 해결하는 내부 메커니즘과 이에 필요한 '계획' 및 '추론' 회로를 탐색합니다. 또한 Mistral 7B 모델을 통해 보다 큰 모델의 내부 구성요소를 분석하여 논리 문제 해결에 필요한 회로를 식별합니다.

- **Technical Details**: 연구는 Synthetic propositional logic 문제를 중심으로 진행되며, 두 가지 주요 실험을 포함합니다: (1) 작은 Transformer 모델에 대한 실험과 (2) 사전 훈련된 LLM인 Mistral-7B에 대한 실험. 활성화 패칭 (activation patching)을 통해 Mistral-7B의 특정 주목 헤드(attention heads) 역할을 분석하여 문제를 해결하는 데 필요한 회로를 밝혀냅니다.

- **Performance Highlights**: 작은 Transformer 모델은 다양한 추론 문제를 해결하는 데 있어 정보 흐름을 불균형하게 조정하는 '라우팅 임베딩(routing embeddings)'을 사용하고, Mistral-7B 모델에서는 각 규칙과 사실을 처리하는 특수화된 주목 헤드의 역할을 발견하였습니다. 이 연구는 작은 및 큰 Transformer의 새로운 측면을 체계적으로 드러내며, 모델이 어떻게 계획하고 추론하는지를 지속적으로 탐구합니다.



### RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (https://arxiv.org/abs/2411.04097)
Comments:
          NeurIPS 2024

- **What's New**: 이 연구에서는 spurious correlations(허위 상관관계)를 발견하고 완화하기 위해 Region-aware Vision-Language learning(RaVL)이라는 새로운 접근법을 제안합니다. 기존 방법들은 주로 전역 이미지 레벨에서 작동하며, 세부 이미지 특징에 직접 개입하지 않으며 unimodal 설정에 주로 설계되었습니다.

- **Technical Details**: RaVL은 fine-tuned VLM에서 학습된 spurious correlations를 발견하기 위해 지역 수준의 클러스터링 접근 방식을 활용하고, 이를 통해 zero-shot classification 오류에 기여하는 정확한 이미지 특징을 식별합니다. 또한, RaVL은 novel region-aware loss function을 도입하여 모델이 관련 영역에 집중하고 허위 관계를 무시하도록 합니다.

- **Performance Highlights**: RaVL은 654개의 여러 VLM 아키텍처 및 데이터 도메인에 대해 평가되었으며, 스푸리어스 상관관계를 정확하게 발견하여 최근 경계선보다 191% 향상된 결과를 보여주었고, 최악 그룹 이미지 분류 정확도에서 8.2% 향상된 효과를 입증했습니다. 일반 도메인 및 의료 도메인 VLM에 대한 정성적 평가에서도 RaVL의 유용성이 확인되었습니다.



### A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreemen (https://arxiv.org/abs/2411.04090)
Comments:
          35 pages, 1 figure

- **What's New**: 이 논문은 주석에서의 불일치를 포착하는 데 초점을 맞춘 새로운 콘텐츠 조정 프레임워크를 제안합니다. 이 프레임워크는 다중 작업 학습(multitask learning) 접근법을 사용하여 독성 분류(toxicity classification)와 주석 불일치(annotation disagreement)를 보조 작업(auxiliary task)으로 적절하게 다룹니다.

- **Technical Details**: 우리는 Conformal Prediction(UQ) 기법을 활용하여 주석에서의 모호성과 모델의 예측 독성에 대한 고유한 불확실성을 고려합니다. 이 프레임워크는 조정자들이 주석 불일치에 대한 임계값(threshold)을 조정할 수 있어 인간 검토 결과 최적화를 지원합니다.

- **Performance Highlights**: 제안된 접근법은 모델의 성능, 보정(calibration), 불확실성 추정을 향상시킬 뿐만 아니라, 단일 작업(single-task) 방법에 비해 파라미터 효율성과 검토 프로세스 개선을 제공합니다.



### M3SciQA: A Multi-Modal Multi-Document Scientific QA Benchmark for Evaluating Foundation Models (https://arxiv.org/abs/2411.04075)
- **What's New**: 새로운 벤치마크 M3SciQA가 소개되었으며, 이는 다중 문서 및 다중 모달 과학 질문 응답을 평가하기 위해 설계되었습니다. 기존의 벤치마크는 주로 단일 문서 및 텍스트 전용 작업에 중점을 두었으나, M3SciQA는 1,452개의 전문가 지정 질문을 포함하여 70개의 자연어 처리(NLP) 논문 클러스터를 아우릅니다.

- **Technical Details**: M3SciQA 벤치마크는 시각적 컨텍스트 질문과 참조 기반 질문의 두 가지 유형으로 구성됩니다. 각각의 클러스터는 주요 논문과 그에 인용된 모든 문서를 포함하여, 다중 문서 및 다중 모달 접근 방식을 requirment합니다.

- **Performance Highlights**: M3SciQA를 통해 18개의 기초 모델에 대한 광범위한 평가가 이루어졌고, 현재의 모델들이 인간 전문가에 비해 다중 모달 정보 검색 및 여러 과학 문서 간의 추론에서 상당히 저조한 성능을 보인다는 점이 드러났습니다. 예를 들어, 최고의 성능을 보인 모델인 GPT-4o는 MRR(Mean Reciprocal Rank) 0.488을 기록, 전문가 점수 0.796에 비해 0.308의 성능 격차를 보였습니다.



### Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Res (https://arxiv.org/abs/2411.04034)
- **What's New**: 본 연구에서는 비정상적(non-stationary) 데이터 분포에 적응하는 신경망 학습 접근법을 제안합니다. 기존에는 정적 분포를 가정했으나, 분포가 변하는 상황에서의 학습을 가능하게 하고자 합니다.

- **Technical Details**: 제안된 접근법은 Ornstein-Uhlenbeck 프로세스를 기반으로 하며, 적응적 드리프트 파라미터를 사용하여 비정상성에 적응합니다. 이 드리프트는 파라미터를 초기화 분포로 끌어당기는 경향이 있어 소프트 파라미터 리셋 형태로 이해할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 비정상적인 감독 학습과 off-policy 강화 학습 설정에서 효과적으로 작동함을 보여 주었습니다.



### Predicting and Publishing Accurate Imbalance Prices Using Monte Carlo Tree Search (https://arxiv.org/abs/2411.04011)
- **What's New**: 이번 논문에서는 재생 가능 에너지(RE) 공급의 변동성에 따른 전력망(전력 시스템) 불균형 문제를 해결하기 위해 신규 방법론인 Monte Carlo Tree Search (MCTS)를 제안했습니다. 이 방법은 시스템의 역학을 신경망 예측기(neural network forecaster)와 강화 학습(reinforcement learning) 에이전트가 제어하는 가상 배터리(fe) 클러스터를 사용하여 모델링합니다.

- **Technical Details**: 제안된 MCTS 방법은 가격 예측을 위해 시스템 동역학의 모델을 기반으로 하며, 1분 단위의 실시간 불균형 가격을 정확하게 예측하고 게시하는 것을 목표로 합니다. 이 기술은 각각의 예측 후 실시간 가격 발표를 위한 낮은 계산 시간을 필요로 하며 산업 복잡성을 조절하는 유연성을 제공합니다.

- **Performance Highlights**: 제안된 기술은 벨기에의 현재 게시 방법에 비해 이상적인 조건에서 가격 정확성을 20.4% 향상시켰으며, 보다 현실적인 조건에서는 12.8%의 향상을 보였습니다. 이 연구는 이전에 시도되지 않았던 진보된 불균형 가격 게시 기법을 분석하는 선구적인 작업으로 자리매김합니다.



### Aligning Characteristic Descriptors with Images for Human-Expert-like Explainability (https://arxiv.org/abs/2411.04008)
- **What's New**: 본 논문에서는 딥러닝 모델의 결정 과정을 설명하기 위해 특성 설명자(characteristic descriptors)를 활용하는 새로운 접근 방식을 제안합니다. 이는 인간 전문가의 설명을 모방하여 출력 결과의 해석성을 높입니다.

- **Technical Details**: 제안하는 방법에는 모델 아키텍처 내에 개념 병목층(concept bottleneck layer)을 포함하여 이미지와 설명자의 인코딩 간 유사도를 계산하고, 이를 통해 설명을 생성하는 방식이 포함됩니다. 또한 CLIP을 사용하여 이미지를 인코딩하고, 텍스트 개념과의 매칭을 통해 분석합니다.

- **Performance Highlights**: 얼굴 인식과 가슴 X-레이 진단 실험을 통해 기존 기법에 비해 훈련을 통해 전문가 수준의 해석 가능한 설명을 제공하며, 외부의 시각적 설명보다는 간결하고 신뢰할 수 있는 서면 설명을 제공하는 능력이 있음을 입증했습니다.



### Select2Plan: Training-Free ICL-Based Planning through VQA and Memory Retrieva (https://arxiv.org/abs/2411.04006)
- **What's New**: 이 연구는 자율 주행의 맥락에서 오프-더-셀프 비전-언어 모델(Vision-Language Models, VLMs)을 사용한 고급 로봇 계획의 가능성을 탐구합니다. 이를 위해 Select2Plan (S2P)라는 새로운 훈련 없는 프레임워크를 도입하여 전문 교육이나 미세 조정이 필요 없는 로봇 계획을 가능하게 했습니다.

- **Technical Details**: Select2Plan는 구조화된 비주얼 질문-답변(Visual Question-Answering, VQA)과 인-컨텍스트 학습(In-Context Learning, ICL)을 활용하여 데이터 수집의 필요성을 크게 줄이는 방식으로 작동합니다. 이 방법은 일반적으로 훈련된 모델이 사용하는 특정 작업 데이터의 극히 일부만을 필요로 하거나, 온라인 데이터에만 의존하기도 합니다. 우리의 방식을 사용하면 로봇이 모노큘러 카메라 외에 추가적인 센서를 요구하지 않으며, 다양한 장면 유형과 센싱 설정에 유연하게 적응할 수 있습니다.

- **Performance Highlights**: TPV(Third-Person View) 시나리오에서 기본 VLM의 내비게이션 능력이 약 50% 향상되었으며, FPV(First-Person View) 시나리오에서는 훈련된 모델과 유사한 성능을 보였습니다. 이 모든 과정을 고작 20회의 시연을 통해 검증할 수 있었습니다.



### ParaGAN: A Scalable Distributed Training Framework for Generative Adversarial Networks (https://arxiv.org/abs/2411.03999)
Comments:
          Accepted at ACM Symposium on Cloud Computing (SoCC) 2024

- **What's New**: 이 논문에서는 대규모 GAN(Generative Adversarial Networks) 학습을 위한 분산 훈련 프레임워크인 ParaGAN을 소개합니다. ParaGAN은 비동기 훈련과 비대칭 최적화 정책을 이용하여 GAN 훈련 시간을 크게 단축합니다.

- **Technical Details**: ParaGAN은 데이터 파이프라인과 하드웨어 레이아웃 변환을 최적화하여 가속기 활용도를 향상시키며, 훈련시간을 15일에서 14시간으로 단축시키면서 91%의 스케일링 효율성을 달성합니다. 또한, ParaGAN을 이용하여 1024x1024 해상도의 이미지 생성을 가능하게 합니다.

- **Performance Highlights**: ParaGAN은 기존 BigGAN보다 30% 이상의 처리량 개선을 이루며, 특히 대규모 훈련에서 GAN의 수렴 문제를 해결하는데 중점을 두고 있습니다.



### Towards Resource-Efficient Federated Learning in Industrial IoT for Multivariate Time Series Analysis (https://arxiv.org/abs/2411.03996)
- **What's New**: 최근 산업 응용에서의 이상 탐지 및 결측 데이터 문제를 해결하기 위한 새로운 연합 학습(federated learning, FL) 프레임워크가 제안되었습니다. 이 방법은 데이터 압축 및 모델 축소를 통해 통신 비용과 계산 복잡성을 줄이는 데 기여할 수 있습니다.

- **Technical Details**: 연합 학습 프레임워크를 기반으로 한 본 연구에서는, 클라이언트가 고유한 센서를 장착하고 다차원(time series) 데이터를 포함하는 상황을 고려하여 모델 프루닝(pruning) 기법을 적용합니다. 고유한 센서로 측정된 단일 변수(univariate) 데이터를 활용하여 모델을 압축하고, 이후 서버 측에서 이를 통합하여 글로벌 모델을 생성하여 성능 저하를 최소화합니다.

- **Performance Highlights**: 제안된 방법은 결측값 대체 및 이상 탐지 문제에서 99.7% 이상의 높은 압축률을 달성하였으며, 성능 손실은 1.18% 미만에 그쳤습니다. 이는 기존 중앙 집중식(solution) 접근 방식에 비해 매우 효율적인 결과입니다.



### What Really is Commonsense Knowledge? (https://arxiv.org/abs/2411.03964)
Comments:
          Code and data will be released together with the next version of the paper

- **What's New**: 이 논문은 commonsense(상식) 지식의 정의를 여러 프레임워크에 기반하여 통합적으로 정리하고, 이를 바탕으로 CommonsenseQA 및 CommonsenseQA 2.0 데이터셋의 비상식 지식 인스턴스 비율을 분석합니다.

- **Technical Details**: 이 연구에서는 상식 지식을 기존의 정의 framework(프레임워크)를 기반으로 분석하고, 그 결과를 CommonsenseQA와 CommonsenseQA 2.0 데이터셋에 적용하여 LLMs(대형 언어 모델)의 성능 차이를 실험합니다. 특히, 각 인스턴스의 지식 유형을 commonsense와 referenced knowledge(참조 지식)으로 구분합니다.

- **Performance Highlights**: 실험 결과, CommonsenseQA에서 27%, CommonsenseQA 2.0에서 56%가 참조 지식 인스턴스인 것으로 나타났으며, LLMs는 commonsense 지식 인스턴스에서 평균적으로 4~7점 낮은 정확도를 보였습니다.



### Energy Score-based Pseudo-Label Filtering and Adaptive Loss for Imbalanced Semi-supervised SAR target recognition (https://arxiv.org/abs/2411.03959)
- **What's New**: 본 연구에서는 비균형 클래스 상황에서의 Semi-Supervised SAR ATR(Automatic Target Recognition) 접근 방식을 제안합니다. 이 방법은 동적 에너지 점수(dynamic energy scores)와 적응형 손실(adaptive loss)을 활용하여, 라벨이 부족한 상황에서도 높은 인식 정확도를 달성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 두 가지 핵심 요소로 구성됩니다. 첫째, Energy Score 기반의 비슷한 배포에 대한 Pseudo-label 선택 메커니즘(ESIDPS)을 도입하여, 긴 꼬리 분포에서의 pseudo-label 신뢰성을 보장합니다. 둘째, Adaptive Margin Loss(AML) 및 Adaptive Hard Triplet Loss(AHTL)와 같은 적응형 손실 함수를 개발하여 class imbalance를 해결합니다. AML은 head 클래스와 tail 클래스 간의 경계 차이를 인식하게 하고, AHTL은 모델이 복잡한 difficile한 샘플에 집중하도록 유도합니다.

- **Performance Highlights**: 두 가지 매우 불균형한 SAR 데이터셋(MSTAR 및 FUSAR-ship)에 대한 광범위한 실험 결과, 제안한 방법이 불균형 클래스에서 SAR 대상을 인식하는 성능을 효과적으로 향상시켰습니다. 실험에서 제안한 방법은 데이터 불균형 문제로 인한 모델 편향을 극복하며, 높은 정밀도의 목표 인식을 달성했습니다.



### Fine-Grained Guidance for Retrievers: Leveraging LLMs' Feedback in Retrieval-Augmented Generation (https://arxiv.org/abs/2411.03957)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능을 향상시키기 위한 새로운 방법인 FiGRet(fine-grained guidance for retrievers)를 제안합니다. 기존의 retriever는 LLM(large language model)과의 최적화가 부족했지만, FiGRet는 LLM의 언어 능력을 활용해 retriever 학습을 보다 효과적으로 돕습니다.

- **Technical Details**: FiGRet 프레임워크는 교육 이론에 기반하여 LLM을 '교사'로, retriever를 '학생'으로 설정하고, 최소한의 겹침이 있는 세 가지 RAG 성능 관련 학습 목표(관련성, 포괄성, 순수성)를 설정하여 retriever의 학습을 안내합니다. 또한, 이 과정에서 이중 커리큘럼 학습 전략을 채택하여 LLM과 retriever 간의 피드백 루프를 활용합니다.

- **Performance Highlights**: 실험을 통해 FiGRet 프레임워크가 다양한 LLM과 retriever 조합에서 성능 향상을 이끌어내었다는 것을 입증하였습니다. MMLU 및 오픈 도메인 QA와 같은 작업들에서도 성능 개선이 관찰되었습니다.



### Long-Form Text-to-Music Generation with Adaptive Prompts: A Case of Study in Tabletop Role-Playing Games Soundtracks (https://arxiv.org/abs/2411.03948)
Comments:
          Paper accepted at the LAMIR 2024 workshop

- **What's New**: 이 논문은 시시각각 변화하는 프롬프트를 사용하여 긴 형식의 음악을 생성할 수 있는 text-to-audio 음악 생성 모델의 능력을 탐구합니다. 특히 Tabletop Role-Playing Games (TRPGs)를 위한 사운드트랙 생성에 중점을 두고 Babel Bardo라는 시스템을 소개합니다.

- **Technical Details**: Babel Bardo는 Large Language Models (LLMs)를 활용하여 발음 전사를 음악 설명으로 변환하고, 이를 사용하여 text-to-music 모델을 제어합니다. 본 연구는 TRPG 캠페인에서 4개의 Babel Bardo 버전을 비교 평가하며, 오디오 품질, 이야기 일치도, 전환 부드러움의 기준으로 모델을 평가합니다.

- **Performance Highlights**: 결과는 상세한 음악 설명이 오디오 품질을 향상시키고, 연속적인 설명 간의 일관성을 유지하는 것이 이야기 일치도와 전환의 부드러움을 개선하는 데 기여한다는 것을 나타냅니다. 이 연구는 감정 신호가 TRPG 내러티브와 생성된 음악 간의 정렬을 효과적으로 수행할 수 있는 방법을 제시합니다.



### Can Custom Models Learn In-Context? An Exploration of Hybrid Architecture Performance on In-Context Learning Tasks (https://arxiv.org/abs/2411.03945)
Comments:
          18 pages, 16 figures

- **What's New**: 이번 논문은 In-Context Learning (ICL) 현상을 연구하며, 특히 Multi-Headed Attention (MHA) 모델에서의 절대 위치 인코딩을 다룹니다. GPT-2와 LLaMa, LLaMa와 Mamba 간의 아키텍처적 차이에 대해 분석합니다.

- **Technical Details**: 이 연구는 Garg et al. (2022)와 Park et al. (2024)의 작업에 기반하여 GPT-2/LLaMa 혼합 모델과 LLaMa/Mamba 혼합 모델을 확장합니다. 이 과정에서 시퀀스 변환 블록과 ICL 성능 간의 상호작용을 조사합니다. 아키텍처 변경이 학습 효율성과 ICL 정확도에 미치는 영향도 논의하며, ICL 회귀 점수라는 메트릭을 제안합니다.

- **Performance Highlights**: 특정 하이브리드 모델에서 긍정적인 성과 향상이 관찰되었습니다. 모든 실험은 모듈화된 파이썬 패키지를 사용하여 수행되어 재현성과 확장성을 촉진합니다.



### Fine-tuning -- a Transfer Learning approach (https://arxiv.org/abs/2411.03941)
- **What's New**: 본 논문에서는 전통적인 통계 및 심층 학습 기반의 데이터 보간 방법들을 대체할 수 있는 모듈화된 데이터 보간 및 분류 파이프라인의 개발을 제안합니다. 이 접근 방식은 보간기(imputer)와 분류기(classifier)의 성능을 독립적으로 평가할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: 제안된 파이프라인은 최첨단 심층 학습 보간 모델의 기능을 활용하여 다운스트림(classification) 작업을 위한 최적화된 구성 요소로 만들어졌습니다. 구체적으로, autoencoders, RNNs 및 LSTMs와 같은 심층 신경망 아키텍처를 활용하여 복잡한 패턴을 캡처하고 결측값을 보완하는 데 초점을 맞췄습니다.  또한, 기존의 end-to-end 파이프라인과 달리, 각 구성 요소가 최적화되어 독립적으로 평가될 수 있도록 하여 평가와 성능 향상에 있어 투명성을 제공합니다.

- **Performance Highlights**: 이 연구의 주요 발견은 최적화된 보간기를 활용할 경우, 단순한 분류기(classifier)라도 비교 가능한 성능을 달성할 수 있다는 것입니다. 이 접근 방식은 다양한 결측 데이터 패턴을 다룰 수 있어 보간의 품질을 높이며, 의료 데이터 분석에 있어 신뢰성을 크게 향상시키는 가능성을 보여줍니다.



### Interactions Across Blocks in Post-Training Quantization of Large Language Models (https://arxiv.org/abs/2411.03934)
- **What's New**: 이 연구에서는 Post-training quantization을 통해 대규모 언어 모델의 양자화 성능을 개선하기 위한 두 가지 다중 블록 파인튜닝 전략을 제안하고 비교했습니다.

- **Technical Details**: 기존의 양자화 방법은 서로 독립적이라고 가정하며 하위 구조의 지식을 무시했습니다. 이에 반해, 첫 번째 전략은 양자화된 여러 블록을 공동 최적화하여 블록 간의 상관관계를 포착합니다. 두 번째 전략은 후속 블록의 정보를 활용하여 다운스트림(pre-activation)의 오류를 최소화합니다.

- **Performance Highlights**: 모델에 따라 이러한 방법의 효과가 달라졌으며, 일부 모델에서는 아무런 영향을 미치지 않았지만 다른 모델에서는 상당한 성과 향상을 보였습니다.



### Disability data futures: Achievable imaginaries for AI and disability data justic (https://arxiv.org/abs/2411.03885)
- **What's New**: 본 논문은 데이터와 인공지능(AI)이 개인의 정체성과 경험을 필터링하는 현대 사회에서의 역할을 탐구합니다. 특히, AI와 데이터 시스템의 발전이 장애인을 포함한 다양한 집단에 미치는 영향과 그로 인한 가능성을 논의합니다.

- **Technical Details**: 연구자들과 장애인 옹호자들이 협력하여 장애 데이터 정의(data justice)와 인공지능의 교차점에서 실현 가능한 비전을 제시합니다. 이들은 다양한 맥락, 학문적 관점, 그리고 개인적 경험을 반영하여 장애 정의를 향한 진전을 지원하는 데이터 및 AI의 구상되는 형태와 목적을 설명합니다.

- **Performance Highlights**: 이 논문은 장애인을 배제하는 역사적 맥락을 비판하며, 데이터 시스템과 AI가 알고리즘 중립성의 허울을 뒤로 한 채 장애 차별을 자동화할 위험에 대해 경고합니다. 동시에, 이들은 포괄적인 미래를 위한 장애 기반의 비전이 공동 행동을 위한 새로운 경로를 열 수 있음을 강조합니다.



### Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models (https://arxiv.org/abs/2411.03884)
- **What's New**: 본 논문에서는 transformers 아키텍처를 최적화하기 위해 폴리곤 조합 활성화 함수(PolyCom)를 제안합니다. 이 활성화 함수는 전통적인 비선형 활성화 기능의 한계를 극복하고, 데이터 내의 더 복잡한 패턴을 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: PolyCom 활성화 함수는 다항식(degree) 조합을 활용하여 더 높은 차원의 상호작용을 캡처할 수 있게 해줍니다. 특히, PolyReLU와 PolyNorm의 두 가지 인스턴스를 도입하며, 이들은 transformer 아키텍처의 통합 과정에서 다항식의 힘을 활용합니다. 폴리곤 조합 활성화 함수는 Sobolev 공간에서 일반적인 매끄러운 함수의 최적 근사 속도를 달성하는 것을 보여줍니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)에서 PolyCom을 적용한 결과, SwiGLU, GELU 및 ReLU 등 기존의 활성화 함수보다 성능이 유의미하게 향상되었으며, 수렴 속도도 크게 빨라짐을 확인했습니다. 이는 PolyCom이 딥러닝 애플리케이션에서 실질적인 이점을 제공함을 시사합니다.



### MEG: Medical Knowledge-Augmented Large Language Models for Question Answering (https://arxiv.org/abs/2411.03883)
- **What's New**: MEG라는 새로운 파라미터 효율적인 접근 방식을 제안합니다. 이는 의료 관련 LLM(대규모 언어 모델)에 그래프 임베딩(knowledge graph embeddings)을 통합하여 외부 지식을 비용 효율적으로 활용할 수 있도록 합니다.

- **Technical Details**: MEG는 경량화된 매핑 네트워크를 활용하여 KG 임베딩을 LLM의 벡터 공간으로 단방향으로 변환합니다. Mistral-Instruct(7B)를 기본 LLM으로 사용하였으며, KG 인코더는 GraphSAGE를 기반으로 하여 간단한 MLP(다층 퍼셉트론) 매핑 네트워크와 결합되었습니다. RAG(회수 증강 생성) 방식과 유사한 접근방식입니다.

- **Performance Highlights**: MEG는 Mistral-Instruct 기준선에 비해 평균 10.2% 더 높은 정확도를 달성하였고, BioMistral과 같은 전용 모델에 비해 6.7% 더 우수한 성능을 보였습니다. 4개의 인기 있는 의료 다중 선택 데이터셋에서 검증하였으며, KG 임베딩을 통합한 LLM이 의료 질문 답변에 있어 효과적임을 입증하였습니다.



### Performance evaluation of SLAM-ASR: The Good, the Bad, the Ugly, and the Way Forward (https://arxiv.org/abs/2411.03866)
Comments:
          Submitted to ICASSP 2025 SALMA Workshop

- **What's New**: 이 논문은 SLAM-ASR architectures의 성능을 분석하고, 다양한 실험을 통해 훈련 데이터에 대한 의존성과 다양한 잡음 및 시간적 변화에 대한 민감도를 조사합니다.

- **Technical Details**: SLAM-ASR 아키텍처는 음성 인코더, 다운샘플러 및 학습 가능한 선형 프로젝터로 구성됩니다. 이 구조는 음성 인코더의 음향 임베딩을 LLM의 입력 임베딩으로 변환하는 역할을 합니다.

- **Performance Highlights**: SLAM-ASR 시스템은 크로스 도메인 평가에서 저조한 성능을 보였으며, 인도메인 내에서 속도 변화나 잡음 추가와 같은 음성 변형이 성능에 significant한 영향을 미친다는 발견이 있었습니다.



### AdaSociety: An Adaptive Environment with Social Structures for Multi-Agent Decision-Making (https://arxiv.org/abs/2411.03865)
Comments:
          Accepted at NeurIPS D&B 2024

- **What's New**: 본 논문에서는 기존의 다중 에이전트 환경의 한계를 극복하기 위해 AdaSociety라는 새로운 다중 에이전트 환경을 소개합니다. 이 환경은 에이전트의 행동에 기반하여 새로운 작업을 동적으로 생성하고, 사회적 연결을 통해 보상을 조정하는 특징을 가지고 있습니다.

- **Technical Details**: AdaSociety는 확장 가능한 상태(state)와 행동(action) 공간을 가지고 있으며, 명시적이고 조정 가능한 사회 구조를 포함합니다. 이 환경에서는 에이전트들이 새로운 작업에 도전하며, 세 가지 미니 게임을 통해 다양한 사회 구조를 경험하게 됩니다. 또한, 사회 연결은 멀티 레이어 방향 그래프로 표현되어 에이전트 간의 관계를 정량적으로 설명합니다.

- **Performance Highlights**: 초기 결과에 따르면, 특정 사회 구조가 개인 및 집단의 이익을 동시에 촉진할 수 있지만, 현재 강화학습( RL) 및 LLM 기반 알고리즘은 사회 구조를 효과적으로 활용하여 성능을 향상시키는 데 제한적인 효과를 보였습니다. AdaSociety는 다양한 물리적 및 사회적 환경에서의 지능을 탐구하기 위한 귀중한 연구 플랫폼으로 자리매김할 수 있습니다.



### ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization (https://arxiv.org/abs/2411.03862)
Comments:
          Accept to NeurIPS 2024

- **What's New**: 이번 논문에서는 생성 콘텐츠의 수분화 모델에서 효과적인 워터마킹(watermarking) 방법을 제안하며, 기존 방법들과의 차별점으로 강력한 워터마크를 활성적으로 숨길 수 있는 프로세스를 도입했습니다.

- **Technical Details**: 로빈(ROBIN) 스킴은 중간의 확산 상태에서 강력한 워터마크를 주입하고, 최종 생성 이미지에서 워터마크를 감추는 과정을 안내합니다. 이를 위해 최적의 프롬프트(guide prompt) 신호를 생성하는 적대적 최적화(adversarial optimization) 알고리즘을 사용합니다.

- **Performance Highlights**: 여러 확산 모델을 실험한 결과, 해당 워터마크는 심각한 이미지 변조에서도 검증 가능하며, 다른 최신 강력 워터마킹 방법들에 비해 뛰어난 투명성을 제공합니다.



### UniTraj: Universal Human Trajectory Modeling from Billion-Scale Worldwide Traces (https://arxiv.org/abs/2411.03859)
- **What's New**: 본 연구는 다양한 작업과 지역에 걸쳐 일반화 및 확장이 가능한 범용 인간 궤적 기반 모델, UniTraj를 제안합니다. 또한, 세계적으로 분산된 첫 번째 대규모 고품질 궤적 데이터셋인 WorldTrace를 구축하여, 70개국에서 2.45백만 개 궤적과 수십억 개의 포인트를 포함하고 있습니다.

- **Technical Details**: UniTraj는 task-adaptive(작업 적응형), region-independent(지역 비의존성) 및 데이터 품질에 높은 내성을 갖춘 모델입니다. 이 모델은 다양한 궤적 분석 작업을 효율적으로 지원하고, 여러 재샘플링 및 마스킹 전략을 활용하여 복잡한 시공간 의존성을 포착합니다.

- **Performance Highlights**: 다양한 궤적 분석 작업과 실제 데이터셋을 통해 UniTraj는 기존 접근 방식에 비해 일관되게 우수한 성능을 보였습니다. 특히, WorldTrace 데이터셋은 파트너십 및 협력을 통한 손쉬운 접근성을 제공함으로써 범용 궤적 모델 개발을 촉진할 수 있는 잠재력을 지니고 있습니다.



### MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba (https://arxiv.org/abs/2411.03855)
- **What's New**: 본 논문에서는 Mamba라는 새로운 State Space Model (SSM) 기반 모델에 대한 Parameter-efficient fine-tuning (PEFT) 방법들의 탐색적 분석을 진행하였습니다. 기존의 Transformer 기반 모델들에 비해 Mamba 모델에서의 PEFT의 효율성과 효과를 강조하며, 신규 Mamba 특화 PEFT 방법들을 제안하였습니다.

- **Technical Details**: Mamba는 시간을 선형적으로 처리할 수 있는 모델로, 기존 Transformer의 계산 복잡성을 극복하며 긴 시퀀스를 효율적으로 처리합니다. 본 연구에서는 Mamba 아키텍처에 적합하도록 기존 PEFT 방법들을 수정하고, Mamba에 특화된 새로운 PEFT 방법들을 제안하여 성능을 극대화하는 방법을 제안합니다. 실험에서는 7개의 주요 PEFT 방법과 총 20개의 파생 변형을 벤치마킹하였습니다.

- **Performance Highlights**: Mamba에서의 PEFT는 Transformer보다 더 효과적이며, 여러 PEFT 방법들을 조합하여 성능을 향상시킬 수 있음을 보여주었습니다. 본 논문은 PEFT 방법의 조합을 효율적으로 탐색하는 기술을 제안하고, 단순한 높은 성능의 방법 조합만으로는 충분하지 않다는 것을 밝혔습니다.



### A Novel Access Control and Privacy-Enhancing Approach for Models in Edge Computing (https://arxiv.org/abs/2411.03847)
- **What's New**: 이 논문은 엣지 컴퓨팅 환경을 위한 새로운 모델 접근 제어 방법을 제안합니다. 이 방법은 이미지 스타일을 라이선스 메커니즘으로 활용하여 모델 운영 프레임워크에 스타일 인식을 내장합니다.

- **Technical Details**: 제안된 방법은 특정 스타일의 라이선스 데이터에서만 정확하게 추론할 수 있도록 설계되어, 엣지 모델에 대한 무단 접근을 방지합니다. 기존의 전통적인 암호화 및 인증 방법의 유연성과 적응성 부족 문제를 해결합니다.

- **Performance Highlights**: MNIST, CIFAR-10, FACESCRUB 등의 벤치마크 데이터셋을 활용한 실험에서, 제안된 방법은 무단 접근을 효과적으로 방지하며, 정확도를 유지하고 허위 라이선스 및 파인튜닝에 대한 강력한 저항력을 보여줍니다.



### Reconsidering the Performance of GAE in Link Prediction (https://arxiv.org/abs/2411.03845)
- **What's New**: 최근 그래프 신경망(GNN)의 본질을 이해하고, Graph Autoencoders (GAE)의 성능을 최적화하여 복잡한 모델의 성능에 필적할 수 있음을 보여주었습니다. 이 연구는 하이퍼파라미터 조정 및 직교 임베딩(orthogonal embedding)과 선형 전파(linear propagation) 기법을 활용한 새로운 방법론을 제안합니다.

- **Technical Details**: 본 연구는 GAE 모델을 OGB 벤치마크에서 평가하며, 새로운 모델의 기술을 활용하고 하이퍼파라미터를 포괄적으로 조정하여 비교 분석합니다. 특히, GAE의 제한 사항을 극복하기 위해 직교 벡터의 선형 전파를 적용하고, 특히 ogbl-ddi 및 ogbl-ppa 데이터셋에서 성능을 개선했습니다.

- **Performance Highlights**: GAE의 최적화된 모델은 더 복잡한 GNN 모델과 경쟁할 수 있는 성능을 발휘하지만, 계산 효율성은 더 뛰어난 것으로 나타났습니다. 이는 새로운 GNN 기술들이 직면한 표현력의 제한성을 해결하는 방향으로 기여하고 있습니다.



### Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM Data Contamination (https://arxiv.org/abs/2411.03823)
- **What's New**: 이 연구에서는 다중 모달 대규모 언어 모델(MLLMs)에서 데이터 오염 감지를 위한 새로운 프레임워크인 MM-Detect를 소개합니다. 이 프레임워크는 다양한 오염 수준을 감지할 수 있으며, MLLMs의 훈련 세트 유출로 인한 성능 향상을 강조합니다.

- **Technical Details**: MM-Detect는 두 가지 방법, 즉 Option Order Sensitivity Test와 Slot Guessing for Perturbation Caption을 통합하여 Multiple-choice 및 Caption-based 질문과 같은 시각적 질문 응답(VQA) 작업을 처리합니다. 또한, 연구에서는 오염이 LLMs의 사전 훈련(pre-training) 단계에서 유래할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: MM-Detect를 총 5개의 VQA 데이터 세트에 걸쳐 11개의 널리 사용되는 MLLMs에 적용한 결과, MLLMs에서 데이터 오염이 관찰되었으며, 이는 모델에 따라 오염 정도가 다르게 나타났습니다. 이 연구는 MLLMs에서의 데이터 오염 감지 및 그로 인해 성능이 증가할 수 있는 경로를 최초로 체계적으로 분석한 것입니다.



### GS2Pose: Tow-stage 6D Object Pose Estimation Guided by Gaussian Splatting (https://arxiv.org/abs/2411.03807)
- **What's New**: 이 논문에서는 새로운 6D 포즈 추정 방법인 GS2Pose를 제안합니다. GS2Pose는 CAD 모델이 필요하지 않고 세분화된 RGBD 이미지만으로도 작업을 수행할 수 있습니다. 이 방법은 두 단계 구조로 코스 추정(coarse estimation)과 정밀 추정(refined estimation)으로 나눠져 있습니다.

- **Technical Details**: GS2Pose는 3D Gaussian splatting을 도입하여 3DGS 모델을 사용한 감독 학습을 통해 주변 환경 변화에 대한 강력한 저항성을 보입니다. 코스 단계에서 Pose-Net이라는 경량 U-Net 네트워크를 이용해 대략적인 포즈를 추정하며, 정밀 단계에서는 GS-Refiner를 통해 입력 이미지와 렌더링된 이미지를 비교하여 포즈를 정제합니다.

- **Performance Highlights**: GS2Pose는 LineMod 데이터셋에서 비교 알고리즘에 비해 뛰어난 정확도, 추론 속도 및 계산 자원 효율성을 보여주었습니다. 이 모델은 환경 변화, Occlusion, 조명 등 다양한 도전 과제를 극복하는데 높은 성능을 발휘하며, 곧 GitHub에 코드가 공개될 예정입니다.



### Overcoming label shift in targeted federated learning (https://arxiv.org/abs/2411.03799)
- **What's New**: 이번 연구에서는 FedPALS라는 새로운 모델 집계 기법을 제안하여 label shift 문제를 해결합니다. FedPALS는 중간 서버에서 목표 label 분포에 대한 지식을 활용하여 모델 업데이트를 보장합니다.

- **Technical Details**: FedPALS는 클라이언트 모델의 볼록 조합을 최적화하여 목표 도메인에 적합한 집계 모델을 생성합니다. 이는 Stochastic Gradient Descent(SGD) 하에서 편향 없는 업데이트를 보장합니다.

- **Performance Highlights**: FedPALS는 이미지 분류 실험에서 전통적인 방법들보다 일관되게 뛰어난 성능을 보였으며, 극단적인 클라이언트 희소성 상황에서도 성능 저하 문제를 완화했습니다.



### VQA$^2$:Visual Question Answering for Video Quality Assessmen (https://arxiv.org/abs/2411.03795)
Comments:
          10 pages 3 figures

- **What's New**: LMM(large multi-modal models)의 발전에 힘입어, 저자들은 비디오 품질 평가(Video Quality Assessment, VQA) 분야에서 Visual Question Answering(VQA) 기반의 첫 번째 대규모 Instruction Dataset인 VQA2 Instruction Dataset을 소개합니다. 이 데이터셋은 비디오 품질에 대한 질문 응답을 중심으로 구성되어 있으며, 157,735개의 질문-답변 쌍을 포함합니다.

- **Technical Details**: VQA2 Instruction Dataset은 세 가지 단계로 구성됩니다: 1단계는 왜곡 인식을 중심으로 한 사전 훈련 데이터, 2단계는 비디오 품질 점수를 위한 Instruction Tuning 데이터, 3단계는 비디오 품질 이해를 위한 Instruction Tuning 데이터입니다. 이번 연구는 다양한 비디오 유형과 구성을 다루기 위해 사람 주도의 주석을 바탕으로 고품질의 데이터를 활용합니다.

- **Performance Highlights**: VQA2 시리즈 모델은 비디오 품질 점수(task)에서 SOTA(state-of-the-art) 성능을 달성하였으며, Visual Quality Question Answering(task)에서도 GPT-4o를 초능가하는 성능을 나타냅니다. VQA2-Assistant 모델은 점수(task)와 질문 응답(task) 모두에서 우수한 성능을 보이며, 모델의 다재다능성을 입증합니다.



### No Culture Left Behind: ArtELingo-28, a Benchmark of WikiArt with Captions in 28 Languages (https://arxiv.org/abs/2411.03769)
Comments:
          9 pages, Accepted at EMNLP 24, for more details see this http URL

- **What's New**: 이번 연구에서는 다양한 언어와 문화에 대한 주관적 감정을 반영한 새로운 비전-언어 벤치마크인 ArtELingo-28을 소개합니다. 이 벤치마크는 28개 언어로 약 200,000개의 주석을 포함하고 있습니다.

- **Technical Details**: ArtELingo-28은 감정적 설명을 생성하는 머신 러닝 시스템을 구축하는 데 중점을 두며, Zero-Shot, Few-Shot 및 One-vs-All Zero-Shot의 세 가지 새로운 평가 조건에 대해 기본 성능 결과를 발표합니다. 또한, 이 연구는 문화적으로 관련된 언어 간의 전이 성능이 더 성공적임을 발견했습니다.

- **Performance Highlights**: 연구 결과, ArtELingo-28은 2000개의 이미지에 대해 200K 감정 레이블과 감정적 텍스트 설명을 수집했으며, 다양한 문화적 시각을 반영한 새로운 데이터 세트를 성공적으로 구축하였습니다. 또한, 다국어 설정에서 모델의 성능을 평가하는 데 기여했습니다.



### Number Cookbook: Number Understanding of Language Models and How to Improve I (https://arxiv.org/abs/2411.03766)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 기본적인 숫자 이해 및 처리 능력(NUPA)을 종합적으로 조사하고, LLMs의 성능을 평가하기 위한 새로운 벤치마크를 제안합니다. 저자는 17개의 다양한 숫자 관련 작업을 포함한 41개의 과제를 설계하였습니다.

- **Technical Details**: 제안한 벤치마크는 정수, 부동소수점 숫자, 분수 및 과학적 표기법 등 4개의 숫자 표현 유형과 4개의 능력 카테고리로 나뉘며, 이 작업들은 초중등 교육 커리큘럼에서 파생되었습니다. 또한, 학습된 모델에 대한 NUPA를 개선하기 위한 세 가지 접근 방식을 탐구하였습니다: 사전 학습(pretraining) 단계에서의 개선, 퀄리티 있는 파인튜닝(fine-tuning) 기법, 그리고 연쇄 사고(chain-of-thought, CoT) 기법의 활용입니다.

- **Performance Highlights**: 최신 LLM들은 일부 간단한 작업에서는 양호한 성능을 보였으나, 보다 복잡한 작업에서 성능이 급격히 저하되는 경향을 보였습니다. 잘 훈련된 모델의 경우, 단순한 파인튜닝을 통해 NUPA 성능을 상당히 개선할 수 있었지만, 전문 기술 적용 시에는 기존 성능을 넘기지 못하는 경우가 많았습니다.



### Sub-DM:Subspace Diffusion Model with Orthogonal Decomposition for MRI Reconstruction (https://arxiv.org/abs/2411.03758)
Comments:
          10 pages, 11 figures

- **What's New**: 이번 논문에서는 MRI 재구성에서의 차별화된 성과를 바탕으로, 기존의 확산 모델(diffusion model) 기반 접근 방식의 한계를 극복하기 위한 새로운 방법인 서브스페이스 확산 모델(subspace diffusion model)인 Sub-DM을 소개합니다.

- **Technical Details**: Sub-DM은 k-공간(k-space) 데이터 분포가 노이즈 방향으로 진화할 때 확산 과정을 서브스페이스에 투영하여 제한하는 방법입니다. 이 모델은 고차원 k-공간 데이터의 복잡성을 피할 수 있는 효과적인 방법을 제공합니다. 또한, 웨이브렛 변환(wavelet transform)에 기반한 직교 분해(orthogonal decomposition) 전략은 일반 확산 과정에서 서브스페이스로의 변환 중 정보 손실을 방지합니다.

- **Performance Highlights**: 데이터 세트에 대한 포괄적인 실험 결과, Sub-DM이 최첨단(state of-the-art) 방법들과 비교했을 때 재구성 속도 및 품질 면에서 우수성을 확연히 보였습니다.



### Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions (https://arxiv.org/abs/2411.03755)
- **What's New**: 이번 논문은 비정렬 다중 도메인 데이터에서 잠재적 콘텐츠(content) 및 스타일(style) 변수를 식별하는 새로운 분석 프레임워크를 제안합니다. 기존 연구의 제한된 조건을 극복하고, 콘텐츠와 스타일의 차원을 미리 알 필요 없이도 도메인 변환 및 데이터 생성에 대한 유용성을 보여줍니다.

- **Technical Details**: 본 연구는 교차 도메인 잠재 분포 매칭(latent distribution matching, LDM)을 통한 콘텐츠-스타일 식별 기준을 제안합니다. 이 기준은 비선형 혼합 모델(nonlinear mixture model)에서도 적용 가능하며, 기존 연구에서 요구되던 요소 간 상호 독립성(component-wise independence) 가정이 필요하지 않습니다. LDM은 다양한 도메인에서 효율적으로 작동하면서도, 계산 자원을 크게 절약할 수 있는 GAN 손실(loss)로 재구성됩니다.

- **Performance Highlights**: 이론적 주장을 뒷받침하는 실험을 통해 이미지 번역(image translation) 및 생성(image generation) 작업에서의 성능을 확인했습니다. 이 연구는 실용적 설정에서의 불확실한 잠재 차원에서도 콘텐츠-스타일 식별 가능성을 유지할 수 있음을 입증했습니다.



### Optimal Defenses Against Gradient Reconstruction Attacks (https://arxiv.org/abs/2411.03746)
Comments:
          The code for this project is available at this https URL

- **What's New**: 본 논문은 Federated Learning (FL) 환경에서 gradient reconstruction 공격으로부터의 데이터 유출을 방지하기 위해 새로운 방어 메커니즘을 제안합니다. 구체적으로, 최적의 gradient noise 추가와 gradient pruning 방법을 통해 데이터 유출과 모델 유틸리티 간의 균형을 최적화합니다.

- **Technical Details**: Gradient reconstruction 공격(GRA)에 대한 취약성을 해결하기 위해, 본 연구는 데이터 유출을 방지하는 한편, 모델의 유틸리티를 최대화하는 두 가지 방어 기법을 제안합니다: Optimal Gradient Noise와 Optimal Gradient Pruning. 이러한 방법들은 각 파라미터에 최적화되어 데이터 보호 수준을 높입니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법들이 기존의 Gradient Noise 및 Gradient Pruning 방법보다 더 우수한 데이터 보호 효과를 나타내며, 동시에 모델의 유틸리티 또한 개선되었음을 확인하였습니다.



### Adaptive Consensus Gradients Aggregation for Scaled Distributed Training (https://arxiv.org/abs/2411.03742)
- **What's New**: 이 논문은 동기식 데이터 병렬 처리 환경에서의 분산 기울기 집계 문제를 새롭게 분석하고, 하위 공간 최적화(subspace optimization) 관점에서 접근합니다. 기울기 집계 문제를 목표 인식 하위 공간 최적화 문제로 정식화하여, 하위 공간 계수(subspace coefficients)에 기반한 효율적인 기울기 가중치(weighting scheme)를 도출하였습니다.

- **Technical Details**: 분산 최적화 환경에서 기울기 집계는 각 작업자가 서로 다른 데이터의 하위 집합을 처리하고, 계산된 기울기를 중앙 모델에 집계하는 과정입니다. 기존의 기울기 평균화(gradient averaging) 방법 대신 비편향(unbiased) 추정기를 사용하는 하위 공간(momentum) 모멘텀을 도입하여 수렴 속도를 높이고 있습니다. 논문은 또한, 모멘텀을 통해 기울기 집계 시 통계적 비편향성을 유지하면서 성능을 개선합니다.

- **Performance Highlights**: 제안된 방법은 여러 MLPerf 작업에서 기울기 평균화 방법보다 현저한 성능 향상을 보였으며, 통신(computation) 및 계산(computational) 복잡성이 매우 낮아 효율적입니다. 이 방법은 하이퍼파라미터 튜닝(hyper-parameter tuning)이나 데이터 병렬 처리 설정의 수정 없이도 유용하게 적용될 수 있습니다.



### Relation Learning and Aggregate-attention for Multi-person Motion Prediction (https://arxiv.org/abs/2411.03729)
Comments:
          Submitted to IEEE Transactions on Multimedia

- **What's New**: 이번 논문에서 제안하는 새로운 협업 학습 프레임워크는 각 개인 내부의 intra-relations와 서로 간의 inter-relations를 명시적으로 모델링하는 방식이다. 이를 통해 기존의 방법이 간과했던 관계를 정확히 다룰 수 있다.

- **Technical Details**: 논문에서는 Graph Convolutional Networks (GCN) 기반 네트워크를 사용하여 intra-relations를 모델링하고, cross-attention 메커니즘을 통해 inter-relations를 처리하는 새로운 구조를 제안한다. 또한, Interaction Aggregation Module (IAM)이라는 플러그 앤 플레이(plug-and-play) 집계 모듈을 도입하여 이 두 관계를 통합한다.

- **Performance Highlights**: 3DPW, 3DPW-RC, CMU-Mocap, MuPoTS-3D 및 합성 데이터셋 Mix1 & Mix2(9~15명)에서의 실험 결과, 제안된 방법이 최신 기술 수준의 성능을 달성함을 보여준다.



### PropNEAT -- Efficient GPU-Compatible Backpropagation over NeuroEvolutionary Augmenting Topology Networks (https://arxiv.org/abs/2411.03726)
- **What's New**: 이 논문에서는 NEAT의 빠른 백프로파게이션 구현인 PropNEAT을 소개합니다. 이 방법은 네트워크의 유전체 그래프를 계층 기반 구조에 이중 방향으로 매핑하여 NEAT 유전체를 보존하면서 GPU 백프로파게이션을 효율적으로 가능하게 합니다.

- **Technical Details**: PropNEAT 알고리즘은 GPU에서 NEAT로 생성된 신경망의 가중치를 학습시키기 위해 백프로파게이션을 효율적으로 적용합니다. 이 알고리즘은 NEAT의 그래프 토폴로지를 계층 기반 구조로 매핑하여 텐서와의 양방향 매핑을 가능하게 합니다. PropNEAT의 설계는 노드 관계를 고려한 그래프 탐색 및 분석 단계를 포함하여 계층 구조와 연결성을 계산하고, 이를 통해 효율적인 텐서 표현을 만듭니다.

- **Performance Highlights**: PropNEAT은 Penn Machine Learning Benchmarks 데이터베이스의 58개의 이진 분류 데이터셋에서 평가된 결과, 우수한 성능을 보였으며, 기본적인 백프로파게이션 방법보다 상당히 빠르고, 원래 NEAT 구현보다도 성능이 뛰어났습니다. PropNEAT은 네트워크 깊이에 따라 선형적으로 훈련 시간이 확장되며, 저전력 컨텍스트에서의 적용 가능성도 보여 줍니다.



### AutoGameUI: Constructing High-Fidelity Game UIs via Multimodal Learning and Interactive Web-Based Too (https://arxiv.org/abs/2411.03709)
Comments:
          27 pages

- **What's New**: 게임 개발에서 일관된 사용자 인터페이스를 효율적으로 구성하기 위한 혁신적인 시스템, AutoGameUI를 소개합니다. 이 시스템은 비일관한 UI와 UX 디자인의 통합으로부터 발생하는 일관성 문제를 해결하는 최초의 시스템입니다.

- **Technical Details**: AutoGameUI 시스템은 두 단계의 multimodal learning pipeline을 통해 UI 및 UX 디자인의 포괄적인 표현을 획득하고 그들의 대응 관계를 설정합니다. 이 시스템은 자동 대응 매칭, 보편적인 데이터 프로토콜, 인터렉티브 웹 기반 도구 등 세 가지 요소로 구성되어 있습니다. 또한, Transformer 모델에 기초한 두 개의 multimodal 모델과 유연한 정수 프로그래밍 알고리즘을 사용합니다.

- **Performance Highlights**: 실험 결과는 AutoGameUI 시스템이 구축된 인터페이스와 원본 디자인 간의 일관성을 유지하는 데 효과적임을 보여주었습니다. 이 시스템은 게임 UI 개발 워크플로우를 크게 가속화하며, 고품질의 GameUI 구성을 가능하게 합니다.



### Fine-Tuning Vision-Language Model for Automated Engineering Drawing Information Extraction (https://arxiv.org/abs/2411.03707)
Comments:
          Paper has been submitted to the 9th International Conference on Innovation in Artificial Intelligence (ICIAI 2025)

- **What's New**: 이번 연구는 자동화된 GD&T(Geometric Dimensioning and Tolerancing) 추출 방법을 제안하며, Florence-2라는 오픈소스 비전-언어 모델(Vision-Language Model)을 미세 조정하여 제조업에서의 효율성을 높입니다.

- **Technical Details**: 모델은 400개의 도면으로 구성된 데이터셋에서 도메인 전문가에 의해 제공된 기준 주석과 함께 훈련되었습니다. 서로 다른 수준으로 증강된 데이터셋을 활용한 세 가지 실험을 통해, 0.23억 개의 파라미터를 가진 Florence-2를 전체 파라미터 조정(full-parameter fine-tuning)하여 최적화하였습니다. 대조적으로, 두 개의 최신 클로즈드소스 VLM인 GPT-4o와 Claude-3.5-Sonnet은 제로샷(zero-shot) 설정에서 평가되었습니다.

- **Performance Highlights**: Florence-2는 클로즈드소스 모델에 비해 precision은 29.95%, recall은 37.75%, F1-score는 52.40% 증가하였고, hallucination 비율은 43.15% 감소하였습니다. 이는 작은 오픈소스 VLM의 미세 조정이 GD&T 자동 추출을 위한 효과적이고 실용적인 해결책이 될 수 있음을 보여줍니다.



### Beyond Model Adaptation at Test Time: A Survey (https://arxiv.org/abs/2411.03687)
- **What's New**: 최근 기계 학습 알고리즘은 훈련 샘플과 테스트 샘플이 동일한 분포에서 가져온다는 가정 하에 성공을 거두어왔습니다. 그러나 테스트 분포가 훈련 분포와 차이를 보이기 시작하면 이러한 알고리즘은 취약해집니다. 본 논문에서는 Test-time adaptation(테스트 시간 적응)이라는 새로운 학습 패러다임을 소개하며, 이를 통해 모델이 훈련 데이터에서 학습한 후 테스트 시간에 목표 데이터를 적응시킬 수 있다는 점을 강조하고 있습니다.

- **Technical Details**: Test-time adaptation은 도메인 적응(domain adaptation)과 도메인 일반화(domain generalization)의 장점을 결합한 방법입니다. 이 방법에서는 소스 데이터에서만 모델을 학습하고, 테스트 시점에서 타겟 데이터로 모델을 적응시킵니다. 본 논문은 400개 이상의 최근 연구를 포괄적으로 검토하며, 테스트 시간 적응 방법을 모델, 추론, 정규화, 샘플, 프롬프트의 다섯 가지 범주로 나누어 자세히 분석합니다.

- **Performance Highlights**: Test-time adaptation 알고리즘은 형태학적 변화에 따른 분포 이동을 효과적으로 처리할 수 있는 가능성을 보여줍니다. 이 알고리즘은 실제에서 발생하는 온라인 또는 제한된 테스트 데이터 상황에서 효율적으로 적용가능하며, 다양한 응용 프로그램에서 기계 학습 모델의 성능과 견고성을 향상시키는 잠재력을 가지고 있습니다.



### QUILL: Quotation Generation Enhancement of Large Language Models (https://arxiv.org/abs/2411.03675)
Comments:
          17 pages, 6 figures

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 인용 생성 능력을 평가하고 개선하기 위한 체계적인 연구를 다루고 있습니다. 특히 기존 LLM들이 인용 생성에서 겪는 문제인 'Quotation Hallucination' 현상을 해결하기 위한 새로운 기준과 기법을 제시합니다.

- **Technical Details**: QUILL(QUotation GeneratIon enhancement framework)이란 새로운 프레임워크를 통해 인용 생성 작업의 자동 평가 시스템을 구축하고, 이를 기반으로 32,022개의 인용문이 포함된 이중언어 지식 베이스를 설계했습니다. 또한, 5가지 기준(Quotation Authenticity, Quotation Credibility, Semantic Matching, Semantic Fluency, Quotation Novelty)에 맞춘 자동 평가 메트릭스를 개발했습니다.

- **Performance Highlights**: 개발한 지식 베이스와 재랭킹 메트릭스는 기존 LLM의 인용 생성 성능을 유의미하게 향상시키며, 연구 결과는 인간의 선호도와 높은 상관관계를 보였습니다. 이 체계적인 접근은 LLM의 인용 생성 능력을 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### Towards 3D Semantic Scene Completion for Autonomous Driving: A Meta-Learning Framework Empowered by Deformable Large-Kernel Attention and Mamba Mod (https://arxiv.org/abs/2411.03672)
- **What's New**: MetaSSC는 비용 효율적인 배포를 목표로 하는 새로운 meta-learning 기반의 프레임워크입니다.

- **Technical Details**: 이 프레임워크는 deformable convolution과 large-kernel attention을 활용하여 3D voxel grid 내에서의 장기 의존성(long-range dependencies)을 효과적으로 포착합니다. 또한, voxel 기반의 semantic segmentation(SS) 사전 학습(pretraining) 작업을 통해 불완전한 영역의 의미 및 기하학을 탐색하고 전이 가능한 meta-knowledge를 습득합니다.

- **Performance Highlights**: MetaSSC는 경쟁 모델들보다 월등한 성능을 발휘하며 배포 비용 또한 줄였음을 입증하는 광범위한 실험 결과를 보여줍니다.



### Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation? (https://arxiv.org/abs/2411.03670)
Comments:
          Accepted to NeurIPS-2024

- **What's New**: 이 논문은 AI 성능 테스트의 중요성을 다루며, 기존의 벤치마크의 한계를 극복하기 위해 Touchstone이라는 대규모 협업 세분화 벤치마크를 제안합니다. 이는 9종의 복부 장기를 대상으로 하며, 전 세계 76개 병원에서 수집한 5,195개의 훈련 CT 스캔과 11개 병원에서 수집한 5,903개의 테스트 CT 스캔으로 구성되어 있습니다. 이 데이터셋은 AI 알고리즘의 다양한 상황에서의 성과를 rigorously 평가할 수 있게 해줍니다.

- **Technical Details**: 제안된 Touchstone 벤치마크는 다양한 AI 알고리즘의 성능을 테스트할 수 있도록 설정되어 있습니다. 논문에서는 19개의 AI 알고리즘 개발자 14명을 초대하여 알고리즘을 훈련시킨 후, 세 가지 테스트 세트에서 독립적인 평가를 진행했습니다. 평가는 U-Net, nnU-Net, MedNeXt와 같은 기존 AI 프레임워크 또한 포함되었습니다. nnU-Net은 자동 구성되고, MedNeXt는 Transformer 기반의 확장성을 통해 3차원 의료 이미지 세분화에서의 효과적인 훈련을 지원합니다.

- **Performance Highlights**: 새로운 벤치마크는 다양한 의료 세분화 모델의 정확도를 높이는 데 기여합니다. STU-Net과 같은 모델은 네트워크 깊이와 넓이를 조정하여 세분화 정확도를 향상했습니다. 또한, UniSeg는 다중 작업 의료 이미지 세분화에서 전이 학습 능력을 향상시키는 혁신적인 접근을 제공합니다. 전체적으로, 이러한 새로운 접근법들은 3D 의료 이미지 분야의 발전에 큰 기여를 할 것으로 기대됩니다.



### Evaluating Moral Beliefs across LLMs through a Pluralistic Framework (https://arxiv.org/abs/2411.03665)
- **What's New**: 본 연구는 네 가지 주요 대형 언어 모델의 도덕 신념(moral beliefs)을 평가하기 위한 세 가지 모듈 프레임워크를 도입합니다. 이를 통해 언어 모델의 도덕적 결정 과정을 평가하고, 중국 대학생의 도덕적 선택에 대한 한국어 번역 결과를 비교합니다.

- **Technical Details**: 연구에서는 472개의 도덕적 선택 시나리오를 중국어로 구축하고, 이를 통해 언어 모델의 도덕 원칙 선호도를 파악합니다. 또한, Best-Worst Scaling(BWS) 및 Iterative Luce Spectral Ranking을 통해 도덕 원칙의 순위를 도출하고, 도덕적 논쟁(moral debates)을 통해 언어 모델의 도덕적 선택의 확고함을 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, ChatGPT와 Gemini는 중국 대학생 샘플의 도덕적 결정과 유사하게 나타나며, 개인적 도덕 신념에 강한 경향이 확인되었습니다. 반면에 Ernie와 ChatGLM 같은 중국 모델은 집단적 도덕 신념을 지향하며 도덕적 선택에서 모호성을 보였습니다. 모든 언어 모델에서 성별 편향이 내재되어 있는 것으로도 드러났습니다.



### Requirements Engineering for Older Adult Digital Health Software: A Systematic Literature Review (https://arxiv.org/abs/2411.03656)
Comments:
          arxiv version of SLR on RE for Older Adult Digital Health Software

- **What's New**: 노인 인구의 증가로 인해 기술 지원 노인 돌봄에 대한 관심이 커지고 있습니다. 그러나 돌봄 제공자 부족, 노인의 정서적, 사회적, 신체적, 정신적 요구에 대한 이해의 한계 등의 도전 과제가 존재합니다.

- **Technical Details**: 본 연구는 디지털 건강 소프트웨어에서 노인을 위한 요구 사항 공학(Requirements Engineering, RE) 관련 문헌을 체계적으로 리뷰하였습니다. Kitchenham 방법, PRISMA, PICO 가이드라인을 사용하여 프로토콜을 개발하고 8개의 데이터베이스를 체계적으로 탐색하여 69개의 높은 관련성의 주요 연구를 도출했습니다.

- **Performance Highlights**: 요구 사항 수집 및 이해에서의 변동성이 큰 것으로 나타났으며, 이는 기존 연구의 품질, 깊이 및 요구 사항 수집 방식의 차이에서 기인합니다. 이러한 차이는 RE 방법의 고르지 않은 채택 때문입니다.



### Deploying Multi-task Online Server with Large Language Mod (https://arxiv.org/abs/2411.03644)
Comments:
          COLING2025 under submission

- **What's New**: 본 연구에서는 대규모 언어 모델(large language models)에서 수행할 수 있는 다중 작업 학습(multi-task learning) 프레임워크를 제안합니다. 이 프레임워크는 세 단계로 구성되어 있으며, 각각의 단계는 작업 필터링(task filtering), 고자원 작업에 대한 미세 조정(fine-tuning), 그리고 모든 작업에 대한 미세 조정을 포함합니다.

- **Technical Details**: 이 연구에서는 데이터 불균형(data imbalance) 문제와 작업 이질성(task heterogeneity) 문제를 해결하기 위해 각각의 작업을 필터링하고, 고자원 작업에서 미세 조정한 후 모두를 혼합하여 미세 조정하는 세단계 방법론을 개발하였습니다. 이를 통해 서로 다른 작업 간의 부정적 이전(negative transfer)을 방지하고 자원 낭비를 줄이는 결과를 가져왔습니다.

- **Performance Highlights**: 우리의 접근법은 단일작업(single-task) 방법과 비슷한 성능을 내면서도 서빙 비용을 최대 90.9%까지 줄일 수 있음을 보여주었으며, 여러 개의 벤치마크를 통해 입증된 바 있습니다.



### Adaptive Stereo Depth Estimation with Multi-Spectral Images Across All Lighting Conditions (https://arxiv.org/abs/2411.03638)
- **What's New**: 본 논문은 다중 스펙트럼 깊이 추정을 위한 새로운 프레임워크를 제안하며, 가시광선 및 열 이미지를 스테레오 쌍으로 처리하여 기하학적 제약을 강화하는 방법을 제시합니다.

- **Technical Details**: 우리 방법은 Cross-modal Feature Matching (CFM) 모듈을 통해 각 픽셀에 대한 정렬된 특징 벡터를 생성하고, 비용 볼륨을 구성하여 픽셀 수준의 정밀한 매칭을 수행합니다. 또한, 저조도 영역에서의 불량 조명의 영향을 완화하기 위해 Degradation Masking 메커니즘을 도입합니다.

- **Performance Highlights**: 제안하는 방법은 Multi-Spectral Stereo (MS2) 데이터셋에서 최신 기술(state-of-the-art)보다 우수한 성능을 보이며, 다양한 조명 조건에서 고품질 깊이 맵을 생성하는 것을 입증하였습니다.



### StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding (https://arxiv.org/abs/2411.03628)
- **What's New**: 본 논문에서는 Multimodal Large Language Models (MLLMs)의 스트리밍 비디오 이해 능력을 평가하기 위한 최초의 종합 벤치마크인 StreamingBench를 소개합니다. 이는 MLLMs가 온라인 비디오 스트림을 이해하는 데 필요한 중요한 기술적 요소를 평가하고, 기존의 오프라인 비디오 이해 방법론과의 차이를 부각합니다.

- **Technical Details**: StreamingBench는 세 가지 핵심 측면인 (1) 실시간 시각 이해 (real-time visual understanding), (2) 전천후 소스 이해 (omni-source understanding), (3) 맥락 이해 (contextual understanding)를 평가합니다. 900개의 비디오와 4,500개의 질문으로 구성되어 있으며, 18개의 다양한 작업으로 나뉘어 있습니다. 각 비디오는 밀접하게 관련된 질문을 포함하고, 스트리밍 시나리오를 시뮬레이션하기 위해 여러 시점에서의 질문을 제공합니다.

- **Performance Highlights**: StreamingBench에서 13개의 개방형 및 독점 MLLMs을 실험한 결과, 가장 진보된 모델인 Gemini 1.5 Pro와 GPT-4o조차도 인간 수준의 스트리밍 비디오 이해 능력에 비해 평균 24.59% 낮은 정확도를 기록했습니다. 이로써 현재 MLLMs가 스트리밍 컨텍스트를 따라잡는 데는 상당한 격차가 있으며, 이는 향후 MLLMs의 발전을 위해 중요한 인사이트를 제공합니다.



### Cross Feature Fusion of Fundus Image and Generated Lesion Map for Referable Diabetic Retinopathy Classification (https://arxiv.org/abs/2411.03618)
Comments:
          ACCV 2024 accepted

- **What's New**: 이 연구는 당뇨병성 망막병증(Diabetic Retinopathy, DR)의 조기 탐지 및 진단을 위해 향상된 cross-learning 방법론을 제안합니다. 특히, Swin U-Net 아키텍처를 활용하여 DR의 병변 지도를 세분화하고, 이를 통해 정확한 분류를 가능하게 합니다.

- **Technical Details**: 제안된 방법론은 transfer learning과 cross-attention 메커니즘을 결합하여 segmentation과 분류를 수행합니다. Swin U-Net을 활용하여 이미지에서 병변 지도를 추출하고, 이를 기반으로 classify 모델이 학습됩니다. cross-attention 메커니즘은 이미지의 주요 영역을 강조하여 드러내는 역할을 합니다.

- **Performance Highlights**: 제안한 방법의 분류 정확도는 94.6%로, 기존의 최첨단 기법들을 4.4% 초과하였습니다. 이는 DR 분류의 정확성과 효율성을 크게 향상시키는 성과로, 임상적 워크플로우에 쉽게 통합할 수 있도록 목표하고 있습니다.



### An Experimental Study on Decomposition-Based Deep Ensemble Learning for Traffic Flow Forecasting (https://arxiv.org/abs/2411.03588)
Comments:
          This work has been accepted by the 2024 Australasian Joint Conference on Artificial Intelligence (AJCAI 2024)

- **What's New**: 본 연구는 교통 흐름 예측을 위한 데이터의 분해 기반(Decomposition-based) 깊은 앙상블(deep ensemble) 학습 방법과 비분해 기반(non-decomposition-based) 방법의 성능을 비교합니다. 이 방법은 시간 시퀀스를 여러 개의 간단한 신호로 분해하여 모델링하는 과정을 포함하며, 이전 연구가 비교적 적었던 분야입니다.

- **Technical Details**: 이 연구에서 사용된 주요 기술적 접근 방식은 분해 기반 앙상블 방법과 전통적인 배깅(bagging) 및 시간 영역의 다중 해상도(multi-resolution) 앙상블 방법입니다. 특히, Empirical Mode Decomposition (EMD), Ensemble Empirical Mode Decomposition (EEMD), Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)과 같은 기법들이 효과적으로 활용됩니다. 실험은 교통 데이터셋 세 개를 사용하여 수행되었습니다.

- **Performance Highlights**: 실험 결과 분해 기반 앙상블 방법이 깊은 학습 모델의 성능을 향상시키며, 특히 집계(aggregation) 전략과 예측 범위(forecasting horizons)에 민감함을 나타냈습니다. 반면, non-decomposition 기반 방법은 복잡한 교통 흐름 패턴을 충분히 반영하지 못하는 경향을 보였습니다.



### Hybrid Attention for Robust RGB-T Pedestrian Detection in Real-World Conditions (https://arxiv.org/abs/2411.03576)
Comments:
          Accepted for publication in IEEE Robotics and Automation Letters, October 2024

- **What's New**: 최근 자율주행 애플리케이션에서 다중 스펙트럼 보행자 탐지가 주목받고 있습니다. 본 논문에서는 열화상 이미지와 RGB 이미지의 혼합을 통해 특정 문제를 해결하고자 하며, 부분적인 오버랩과 센서 실패 문제를 해결하기 위해 Hybrid Attention (HA) 메커니즘을 도입했습니다.

- **Technical Details**: 이 논문에서는 Hybrid Attention (HA) 모듈을 사용하여 자가 주의(self-attention)와 교차 주의(cross-attention)를 결합하여 성능 저하를 완화하는 방법을 제안합니다. HA-MLPD(Hybrid Attention-based Multi-Label Pedestrian Detector) 알고리즘은 부분 오버랩 및 센서 실패에 대해 회복력이 강한 RGB-T 융합 알고리즘을 제공합니다. 또한, 임베디드 시스템에서의 자원 제약을 고려하여 경량의 백본(backbone)을 사용합니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션한 다양한 부분 오버랩 및 센서 실패 시나리오에서 테스트를 통해 성능 저하를 방지하고 기존의 최신 방법들에 비해 우수한 성능을 보였습니다.



### Towards Personalized Federated Learning via Comprehensive Knowledge Distillation (https://arxiv.org/abs/2411.03569)
Comments:
          Accepted by IEEE SMC 2024

- **What's New**: 최근 개인화 연합 학습(Personalized Federated Learning, PFL) 접근 방식이 데이터 이질성 문제를 해결하기 위해 발전하였습니다. 본 논문에서는 모델의 개인화를 중시하면서도 일반화 성능을 유지할 수 있는 새로운 방법인 FedCKD를 제안합니다.

- **Technical Details**: FedCKD는 글로벌 모델(global model)과 역사적 모델(historical model)을 교육자로 사용하고, 로컬 모델(local model)을 학생으로 설정하여 지식 증류(knowledge distillation)를 수행합니다. 이 과정에서 글로벌 모델은 서버 집합의 마지막 라운드의 집합 모델을 나타내며, 역사적 모델은 클라이언트 훈련의 마지막 라운드에서의 로컬 모델을 나타냅니다. 이러한 구조를 통해 글로벌 일반화 지식 및 역사적 개인화 지식을 효과적으로 로컬 모델에 전달합니다.

- **Performance Highlights**: 실험 결과, FedCKD는 기존의 최첨단 방법들을 초월하여 모델의 성능을 유의미하게 향상시키는 것으로 나타났습니다. 특히, 모델의 일반화와 개인화 간의 균형을 유지하면서 치명적 망각(catastrophic forgetting) 문제를 완화하는 데 성공하였습니다.



### Large Language Models Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Lev (https://arxiv.org/abs/2411.03562)
- **What's New**: Agent K v1.0은 다양한 데이터 과학 작업을 자동화, 최적화 및 일반화하도록 설계된 종단 간(End-to-End) 자율 데이터 과학 에이전트입니다. 경험으로부터 학습하며 완전 자동화된 기능을 갖추고 있습니다.

- **Technical Details**: Agent K v1.0은 구조화된 추론(framework) 구조를 활용하여 고도로 유연한 방식으로 메모리를 동적으로 처리합니다. 여러 경험을 통해 복잡한 추론(tasks)을 수행하고, 장기 및 단기 기억을 최적화합니다. 환경 보상을 기반으로 한 미래 결정 유도, 세밀한 조정(fine-tuning) 없이도 지속적인 개선을 이룰 수 있는 반복적인 접근 방식을 사용합니다.

- **Performance Highlights**: Kaggle 대회에서 테스트한 결과, Agent K v1.0은 92.5%의 성공률을 기록하였으며, 탭형(tabular), 컴퓨터 비전(computer vision), 자연어 처리(NLP), 멀티모달(multimodal) 도메인에서의 성능을 입증했습니다. 5,856명의 인간 Kaggle 경쟁자와 비교할 때 상위 38%에 랭크되었으며, Kaggle Grandmaster 수준과 동등한 성능을 달성하여 6개의 금메달, 3개의 은메달, 7개의 동메달을 기록했습니다.



### Exploring the Benefits of Domain-Pretraining of Generative Large Language Models for Chemistry (https://arxiv.org/abs/2411.03542)
- **What's New**: 이 논문에서는 화학 분야에 특화된 AISLE(Scientific Literature에서 AI) 모델을 도입하고, 일반적인 언어 모델과 비교하여 전이 학습(pre-training) 및 파인튜닝(fine-tuning)을 통해 이 모델이 보여주는 성능 향상을 분석합니다.

- **Technical Details**: 이 연구는 과학적 텍스트로 모델을 사전 학습(pre-training)하는 이점을 탐구하며, 일반 목적의 대형 언어 모델(general-purpose large language models)과의 성능을 비교합니다. 연구에서는 'zero-shot' 및 'few-shot' 조건에서 도메인 적응 모델을 평가합니다.

- **Performance Highlights**: 실험 결과, 화학 특정 작업에서 AISLE 모델은 기존 모델들보다 우수한 성능을 보이며, 추가적인 지시 기반(fine-tuning using instruction) 조정이 필요한 모든 작업에 대해 효율적인 성능을 입증했습니다.



### Two-Stage Pretraining for Molecular Property Prediction in the Wild (https://arxiv.org/abs/2411.03537)
- **What's New**: MoleVers는 데이터가 부족한 환경에서도 효과적으로 작동하는 다목적의 사전 학습(pretrained) 모델로, 두 단계의 사전 학습 전략을 채택하여 분자 속성 예측을 개선합니다.

- **Technical Details**: MoleVers는 첫 번째 단계에서 masked atom prediction (MAP)과 dynamic denoising을 통해 큰 비표시 데이터셋에서 분자 표현을 학습합니다. 두 번째 단계에서는 낮은 비용의 계산 방법으로 얻은 보조 레이블을 사용하여 추가 사전 학습을 수행합니다. 이러한 이중 단계의 구조는 다양한 다운스트림 데이터셋에 효과적으로 일반화 가능한 표현을 학습하게 합니다.

- **Performance Highlights**: MoleVers는 22개의 다양한 분자 데이터셋을 평가한 결과, 20개에서 최첨단(state-of-the-art) 성능을 달성하고 나머지 2개에서도 두 번째로 높은 성과를 기록하며, 데이터가 부족한 현실 상황에서 유용한 주석이 부족한 모델의 한계를 극복하는데 성공하였습니다.



### Personalized Video Summarization by Multimodal Video Understanding (https://arxiv.org/abs/2411.03531)
Comments:
          In Proceedings of CIKM 2024 Applied Research Track

- **What's New**: 이 논문에서는 사용자 선호에 따라 비디오 요약을 생성하는 새로운 벤치마크인 UserPrefSum 데이터셋을 소개하며, 비디오 자막 및 장면을 분석하여 사용자 맞춤형 비디오 요약을 생성하는 Video Summarization with Language (VSL) 파이프라인을 제안합니다.

- **Technical Details**: VSL 파이프라인은 멀티모달(scene detection) 장면 탐지, 비디오 자막 생성(captioning), 멀티모달 요약(summarization), 비디오 선택(selection)이라는 네 가지 구성 요소로 이루어져 있으며, CLIP 모델을 활용하여 장면의 장르를 자동으로 레이블링합니다. 이 방식은 사전 학습된 비주얼 언어 모델(VLMs)을 사용하여 대규모 트레이닝 데이터 없이도 사용자 선호에 맞춰 요약을 생성할 수 있습니다.

- **Performance Highlights**: VSL은 현업에서의 사용에 적합성을 입증했으며, 기존의 비지도 학습 기반 비디오 요약 모델보다 더 많은 데이터셋에서 유연하게 적용할 수 있습니다. 또한, 사용자 선호가 증가할 때에도 요약 생성 능력이 향상되며, 실제 응용 프로그램에서 중요한 실시간 처리 능력을 보여줍니다.



### Mitigating Metric Bias in Minimum Bayes Risk Decoding (https://arxiv.org/abs/2411.03524)
Comments:
          To appear at WMT2024

- **What's New**: 이번 연구에서는 Minimum Bayes Risk (MBR) 디코딩 과정에서 발생하는 metric bias 문제를 조사하였습니다. MBR 디코딩이 특정 utility metric에 따라 높은 점수를 얻도록 번역 결과를 생성하는 과정에서, 해당 metric을 디코딩과 평가에 모두 사용하는 것이 불가능하다는 점을 확인하였습니다. 주된 발견은 MBR/QE 디코딩이 인간 평가보다 품질을 과대 평가함을 보여주며, utility metric의 앙상블(ensemble) 사용을 통해 이러한 bias 문제를 완화할 수 있다는 것입니다.

- **Technical Details**: MBR 디코딩은 n개의 후보 번역을 샘플링하여, reference 기반의 utility metric을 계산하여 최상의 번역 후보를 선택합니다. MBR/QE 디코딩은 다양한 utility metrics를 통해 수행되며, 간단한 utility metric 사용 시 자동 평가 지표는 눈에 띄는 향상을 보이나, 실제 인간 평가에서는 greedy 디코딩보다 성능이 떨어지는 경우가 많습니다. 이를 해결하기 위해 MBR 디코딩에 여러 utility metrics의 앙상블을 사용하는 전략을 제안하였습니다.

- **Performance Highlights**: 연구 결과, MBR 디코딩을 여러 utility metrics의 앙상블로 수행할 때, 단일 utility metric를 사용할 때보다 더 우수한 번역 품질을 제공하였습니다. 이에 대한 인간 평가 결과가 이를 뒷받침하며, MBR 디코딩의 전반적인 성능을 향상시키는 것으로 나타났습니다.



### Exploring the Potentials and Challenges of Using Large Language Models for the Analysis of Transcriptional Regulation of Long Non-coding RNAs (https://arxiv.org/abs/2411.03522)
- **What's New**: 이 연구에서는 LLMs(대형 언어 모델)를 이용하여 lncRNA 유전자 발현의 전사 조절에 관한 시퀀스 분석을 체계적으로 탐구하고, 기존의 계산적 접근 방식의 한계를 극복할 수 있는 가능성을 조사하고자 합니다.

- **Technical Details**: 본 연구에서는 DNABERT, DNABERT-2, Nucleotide Transformer와 같은 유전체 기초 모델들을 세부 조정(fine-tuning)하여 lncRNA 관련 작업의 성능을 평가했습니다. 또한, 생물학적 설명 가능성을 향상시키기 위한 특징 중요도 분석도 진행하였습니다.

- **Performance Highlights**: 대형 언어 모델을 활용한 실험 결과, 복잡한 작업들에 대한 뛰어난 성능을 보여주었으며, 작업의 복잡성, 모델 선택, 데이터 품질이 lncRNA 분석에 미치는 중요한 영향을 논의하였습니다.



### AI Metropolis: Scaling Large Language Model-based Multi-Agent Simulation with Out-of-order Execution (https://arxiv.org/abs/2411.03519)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트의 시뮬레이션을 위한 새로운 엔진인 AI Metropolis를 소개합니다. 이 엔진은 복잡한 작업을 효율적으로 처리할 수 있도록 순서가 뒤바뀐 실행 스케줄링(out-of-order execution scheduling)을 도입하여, 에이전트 간의 실제 종속성을 동적으로 추적함으로써 잘못된 종속성(false dependencies)을 최소화합니다.

- **Technical Details**: AI Metropolis는 LLM 에이전트 간의 관계를 분석하여 시뮬레이션 동안의 의존성을 관리합니다. 이에 따라 각 에이전트의 동작은 시뮬레이션 결과에 영향을 미치지 않고 시간을 진척시킬 수 있습니다. 이러한 접근법은 병렬성을 향상시켜 대량의 LLM 쿼리를 보다 효과적으로 처리할 수 있도록 합니다. AI Metropolis는 OpenAI Gym과 유사한 인터페이스를 제공하며, 시뮬레이션 상태 업데이트, 데이터베이스 I/O, 스케줄링 및 LLM 추론 과정을 자동으로 관리합니다.

- **Performance Highlights**: AI Metropolis는 전통적인 병렬 시뮬레이션 방식에 비해 1.3배에서 4.15배의 속도 향상을 이루었습니다. 에이전트 수가 증가함에 따라 AI Metropolis의 성능은 최적 성능에 근접하게 되며, 이는 AI Metropolis의 확장성과 효과적인 의존성 관리 능력을 보여줍니다.



### Change Is the Only Constant: Dynamic LLM Slicing based on Layer Redundancy (https://arxiv.org/abs/2411.03513)
Comments:
          Accepted at EMNLP Findings 2024

- **What's New**: 본 논문에서는 Large Language Models(LLMs)에서 동적 레이어 전용 가지치기(dynamic layer-specific pruning)를 통해 새로운 모델 압축 방법을 소개하며, 기존의 SliceGPT의 방법론을 발전시켰습니다. 우리는 변동폭이 있는 동적 슬라이싱(dynamic slicing)으로 전환하여, 각 레이어가 입력을 얼마나 변화시키는지를 평가하는 새로운 Layer Redundancy (LR) 점수를 활용합니다.

- **Technical Details**: 제안된 방법은 각 레이어의 중요도에 따라 가지치기 정도를 조절하는 동적 가지치기(dynamic pruning) 방법으로, LR 점수를 기반으로 개별 레이어의 중복성을 평가했습니다. 이를 통해 계산 효율성을 극대화하면서도 모델의 성능 저하를 최소화하도록 설계되었습니다.

- **Performance Highlights**: Llama3-8B 및 Mistral-7B 모델을 사용한 광범위한 실험 결과, 우리의 방법이 SliceGPT에 비해 최대 5%의 성능 향상을 보였으며, 여러 벤치마크에서 나타난 당혹도(perplexity)는 최대 7% 감소했습니다. 이는 우리의 동적 슬라이싱 접근 방식이 기존의 상수 슬라이싱 방법보다 효율적임을 입증합니다.



### Automatic Generation of Question Hints for Mathematics Problems using Large Language Models in Educational Technology (https://arxiv.org/abs/2411.03495)
Comments:
          Accepted at NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess)

- **What's New**: 이번 연구는 Intelligent Tutoring Systems (ITSs) 내에서 Large Language Models (LLMs)를 사용하여 수학 문제를 해결하는 학생들에게 효과적인 힌트를 생성하는 방법을 모색하였습니다. 특히, LLMs (GPT-4o와 Llama-3-8B-instruct)의教师 역할과 GPT-3.5-turbo 또는 Mistral-7B-instruct-v0.3 기반의 모의 학생 역할을 수행하는 것이 주요한 발전으로 평가됩니다.

- **Technical Details**: 이 연구에서는 다양한 방법으로 LLM을 활용하여 1) 고등학생을 위한 수학 연습에서 발생하는 오류 패턴 인식, 2) GPT-4o를 활용한 힌트 생성 방법의 효과 평가, 3) Llama-3-8B-Instruct를 शिक्षक으로 하여 최적의 프롬프트를 테스트하여 오류 수정 능력을 향상시키는 방법을 조사했습니다. 특히, 힌트 생성에서 온도 설정(temperature)과 관련된 새로운 발견을 보고하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4o가 생성한 힌트는 특정 오류에 맞춘 프롬프트와 일반적인 수학 오류에 기초한 프롬프트에서 가장 효과적이었습니다. Llama-3-8B-Instruct는 전반적인 성능에서 GPT-4o를 초과하며, 특히 GPT-3.5-turbo 모델이 힌트를 받은 후 문제 해결 및 응답 수정 능력이 크게 향상되었습니다.



### Solving Trojan Detection Competitions with Linear Weight Classification (https://arxiv.org/abs/2411.03445)
Comments:
          9 pages, 4 Figures

- **What's New**: 이 논문에서는 트로이안 백도어를 탐지하기 위한 새로운 방법을 제안했습니다. 이 접근법은 다양한 데이터셋과 도메인에서 높은 성능을 보입니다.

- **Technical Details**: 제안된 탐지기는 여러 개의 모델 가중치에 대한 이진 분류기를 학습하여 얻어지며, 주요 전처리 단계를 통해 성능을 개선합니다. 전 처리 단계에는 특성 선택, 표준화, 참조 모델 가중치 빼기, 모델 정렬 등이 포함됩니다. 이 기법은 가중치 분석(weight analysis) 탐지에 해당하며, 트리거에 대한 사전 지식 없이도 적용 가능합니다.

- **Performance Highlights**: 본 알고리즘은 Trojan Detection Challenge(TDC22)와 IARPA/NIST TrojAI 프로그램의 다양한 벤치마크에서 평가되었으며, 모델의 정밀한 분류를 통해 청정(clean) 모델과 오염된(poisoned) 모델 간의 구분을 효과적으로 수행했습니다.



### STEER: Flexible Robotic Manipulation via Dense Language Grounding (https://arxiv.org/abs/2411.03409)
Comments:
          Project website: this https URL

- **What's New**: STEER는 고위험 상황에서 로봇이 유연하게 행동할 수 있도록 돕는 로봇 학습 프레임워크로, 고급의 공통 근거 추론을 저급의 유연한 제어와 연결합니다.

- **Technical Details**: STEER는 자연어 기반의 조밀한 주석을 사용하여 정책 훈련을 구조화하며, 로봇의 행동을 지능적으로 조정할 수 있는 인터페이스를 제공합니다. 또한, 데이터 수집이나 추가 훈련 없이 새로운 작업을 수행할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: STEER는 이전의 모방 학습 방법들에 비해 유연한 저수준 로봇 정책을 훈련시켜 차별화된 성능을 보이며, 인간 또는 사전 훈련된 VLM이 작업을 효율적으로 조작할 수 있도록 지원합니다.



### Neurons for Neutrons: A Transformer Model for Computation Load Estimation on Domain-Decomposed Neutron Transport Problems (https://arxiv.org/abs/2411.03389)
Comments:
          28 pages, 14 figures

- **What's New**: 이 논문에서는 대규모 중성자 전달 문제에 대한 메모리 부담을 줄이기 위해 도메인 분할(Domain Decomposition) 기법을 사용합니다. 가장 최근의 발전으로, 저자들은 Transformer 모델을 제안하여 작은 규모의 시뮬레이션을 통해 수렴된 하위 도메인 계산 부하를 예측할 수 있습니다.

- **Technical Details**: 제안된 Transformer 모델은 독특한 3D 입력 임베딩(3D Input Embedding) 구조와 도메인 분할 중성자 전달 문제를 고려한 입력 표현(Input Representation)을 사용합니다. 이 모델은 Small Modular Reactor(SMR) 시뮬레이션에서 훈련되어 98.2%의 정확도를 달성합니다.

- **Performance Highlights**: 모델은 작은 규모의 시뮬레이션 단계를 완전히 생략하면서도 하위 도메인 계산 부하를 효과적으로 예측할 수 있습니다. 또한 다양한 연료 조립체(Fuel Assemblies), 다른 문제 기하학(Problem Geometries) 및 시뮬레이션 매개변수(Parameter) 변화에 대한 모델의 견고성도 논의됩니다.



### An Open API Architecture to Discover the Trustworthy Explanation of Cloud AI Services (https://arxiv.org/abs/2411.03376)
Comments:
          Published in: IEEE Transactions on Cloud Computing ( Volume: 12, Issue: 2, April-June 2024)

- **What's New**: 이 논문은 클라우드 AI 서비스를 위한 설명 가능한 AI (XAI) 서비스 설계를 제안합니다. 이는 기존 XAI 방법을 클라우드 AI 서비스와 통합하여 AI 모델의 예측 과정을 이해하고 신뢰할 수 있도록 지원합니다.

- **Technical Details**: 제안된 아키텍처는 마이크로서비스 아키텍처를 기반으로 하며, RESTful API를 통해 XAI 기능을 제공합니다. 이는 클라우드 AI 모델의 네트워크 구조를 드러내지 않고도 피처 기여 설명을 제공합니다. 또한, XAI 일관성 메트릭을 평가할 수 있는 기능을 포함하여 신뢰성을 높입니다.

- **Performance Highlights**: 실험 결과, 데이터 증강 기법이 클라우드 AI 서비스의 학습 성능을 개선하고 XAI 메트릭의 일관성 향상에도 기여함을 보여줍니다. 또한, 제안된 아키텍처는 클라우드에 독립적이어서 여러 클라우드 서비스에서 활용될 수 있습니다.



### Enhanced Real-Time Threat Detection in 5G Networks: A Self-Attention RNN Autoencoder Approach for Spectral Intrusion Analysis (https://arxiv.org/abs/2411.03365)
Comments:
          This article has been accepted for publication in WiOpt 2024

- **What's New**: 본 논문은 자가 주의(self-attention) 메커니즘을 활용한 RNN 기반 오토인코더를 통해 5G 네트워크에서의 이상 스펙트럼 활동 탐지 성능을 향상시킨 실험 모델을 제시합니다.

- **Technical Details**: 시간-시리즈 분석을 기반으로 하여 I/Q 샘플을 처리하며 잠재적 재밍 공격을 나타낼 수 있는 이상 현상을 식별합니다. 자가 주의 레이어가 추가된 RNN 오토인코더 아키텍처를 통해 RF 스펙트럼의 시간적 의존성과 맥락적 관계를 더욱 정교하게 이해할 수 있게 됩니다.

- **Performance Highlights**: SDR 기반의 실험 환경에서 모델은 위협 탐지에서 성능과 정확도가 크게 향상된 결과를 보여주었습니다.



### Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection (https://arxiv.org/abs/2411.03359)
Comments:
          accepted by NeurIPS 2024

- **What's New**: 이번 연구에서는 Self-Calibrated Tuning (SCT)라는 새로운 프레임워크를 도입하여, 불완전한 OOD (Out-of-Distribution) 특성을 해결하고 효과적인 OOD 탐지를 가능하게 합니다. 이는 단지 몇 개의 ID (In-Distribution) 데이터만으로 수행됩니다.

- **Technical Details**: SCT는 원래 학습 목표의 두 구성 요소에 각각 조정 요인을 도입하여 OOD 정규화의 영향을 조절할 수 있도록 학습 과정에서의 최적화 방향을 적응적으로 조절합니다. 이를 통해 낮은 예측 불확실성을 가진 데이터로 학습할 때 모델이 더 나은 일반화를 이룰 수 있도록 한다.

- **Performance Highlights**: SCT 방법은 대규모 ImageNet-1k 벤치마크에서 기존의 최상의 방법보다 잘못된 긍정 탐지 비율(FPR95)을 3% 개선하였으며, 다양한 실험과 분석을 통해 SCT의 효율성을 검증하였습니다.



### Enhancing Table Representations with LLM-powered Synthetic Data Generation (https://arxiv.org/abs/2411.03356)
Comments:
          the Thirty-Eighth Annual Conference on Neural Information Processing Systems Table Representation Workshop

- **What's New**: 이 논문에서는 데이터 기반 의사결정 시대에 효율적인 테이블 추천 시스템을 위한 합성 데이터 생성 파이프라인을 제안합니다. 이 시스템은 Large Language Models (LLMs)을 활용하여 테이블 유사성을 정의하고, 테이블 표현 학습을 위한 고품질 합성 데이터를 생성합니다.

- **Technical Details**: 제안된 방법은 다음의 세 가지 접근 방식을 통해 검증되었습니다: (i) 인간 검증을 통한 합성 데이터셋의 정확성 확인, (ii) 기존 데이터셋과의 cosine similarity 비교를 통해 향상된 테이블 표현 가능성 입증, (iii) 합성 데이터를 활용한 유사 테이블 매칭 작업에서 최신 embedding 모델을 초월하는 성과.

- **Performance Highlights**: 실험 결과, 제안된 합성 데이터 생성 파이프라인은 테이블 유사성 정의에 부합하며, 추천 성능을 크게 향상시키는 고품질의 합성 데이터를 생성하여 실질적인 유사 테이블 추천 응용 프로그램에 기여할 수 있음을 보여줍니다.



### Exploring Feature Importance and Explainability Towards Enhanced ML-Based DoS Detection in AI Systems (https://arxiv.org/abs/2411.03355)
Comments:
          6 pages, 2 figures, IEEE VTC2024-Fall

- **What's New**: 이 논문은 DoS 공격 감지에서 머신러닝(ML) 모델의 성능 향상을 위해 feature selection의 중요성을 조사합니다. 특히, LYCOS-IDS2017 데이터셋을 활용하여 불필요한 feature를 제거하고, 이상 검출을 위한 최적의 feature 조합을 탐색합니다.

- **Technical Details**: 본 연구에서는 주성분 분석(Principal Component Analysis, PCA)을 이용하여 LYCOS-IDS2017 데이터셋의 불필요한 feature를 필터링합니다. 다양한 ML 모델(결정 트리, 랜덤 포레스트, 서포트 벡터 머신 등)을 훈련시키고, 훈련된 모델의 성능을 정확성, 정밀도, 재현율 및 F1 점수 등의 메트릭을 통해 평가합니다.

- **Performance Highlights**: 이 논문의 실험적 결과는 DoS 트래픽에 대한 철저한 통계 분석과 feature engineering이 공격 행동 이해 및 ML 기반 DoS 감지의 정확도 향상에 어떻게 기여하는지를 입증합니다.



### Tabular Data Synthesis with Differential Privacy: A Survey (https://arxiv.org/abs/2411.03351)
- **What's New**: 이 논문은 개인 정보를 보호하면서 데이터 공유를 가능하게 하는 차별 개인정보 보호(differential privacy)를 결합한 합성 테이블 데이터 생성 방법에 대한 포괄적인 리뷰를 제공합니다. 여기에 중앙 집중형 및 분산 환경에서의 다양한 접근 방식을 분석합니다.

- **Technical Details**: 본 연구에서는 통계 기반 방법과 딥러닝 기반 방법으로 분류된 다수의 차별 개인정보 보호 합성 데이터 생성 모델을 평가하며, 각 방법의 효용, 개인 정보 보호 수준, 계산 복잡성 면에서의 강점과 약점을 강조합니다. 또한 합성 데이터의 질을 평가하는 다양한 방법론에 대해서도 논의합니다.

- **Performance Highlights**: 중앙 집중형 및 분산형 데이터 합성에서 차별 개인정보 보호를 적용한 방법들의 성능을 종합적으로 분석하였으며, 향후 연구 방향 및 데이터 합성 분야의 발전을 위한 제안도 포함하였습니다.



### A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness (https://arxiv.org/abs/2411.03350)
Comments:
          76 pages, 26 figures, 14 tables

- **What's New**: 본 논문은 Small Language Models (SLMs)의 정의, 획득, 응용, 향상 및 신뢰성 문제에 대한 포괄적인 조사를 수행하고 있습니다. 이러한 조사는 LLMs의 한계를 극복하기 위해 성장하는 SLMs의 필요성과 관련이 있습니다.

- **Technical Details**: SLMs는 적은 추론 지연(inference latency), 비용 효율성(cost-effectiveness), 효율적인 개발(efficient development), 커스터마이징(customization) 및 적응(adaptability)에서 우수합니다. SLMs의 정의는 그들이 수행하는 전문적인 작업의 능력과 자원 제약이 있는 환경에서의 적합성을 기준으로 설정합니다. 또한, 모델 및 방법에 대한 분류학(taxonomy)과 각 범주에 대한 일반적인 프레임워크를 개발하여 SLMs를 효과적으로 활용하는 방법을 제안합니다.

- **Performance Highlights**: SLMs는 LLMs에 비해 로컬 데이터 처리를 위한 개인 정보 보호, 효율성을 위한 최소한의 추론 지연, 경량 파인 튜닝을 통한 도메인 지식 습득에 최적화된 응용 프로그램에 적합한 성능을 보여줍니다.



### Undermining Image and Text Classification Algorithms Using Adversarial Attacks (https://arxiv.org/abs/2411.03348)
Comments:
          Accepted for presentation at Electronic Imaging Conference 2025

- **What's New**: 이 연구는 Generative Adversarial Networks (GANs)와 Synthetic Minority Oversampling Technique (SMOTE)를 결합하여 텍스트 및 이미지 분류 모델에 대한 새로운 adversarial attack 방법론을 제시합니다.

- **Technical Details**: 연구에서 사용된 방법론은 Fast Gradient Sign Method (FGSM)를 이용한 perturbation 벡터와 GradCAM을 통해 강조된 주요 특징을 결합하여 adversarial 공격을 수행합니다. 이는 Convolutional Neural Network (CNN) 모델을 사용하여 얼굴 인식 시스템에 대한 공격을 포함합니다.

- **Performance Highlights**: 이 실험에서 텍스트 분류 모델의 정확도가 공격 후 20% 감소했으며, 얼굴 인식 정확도는 30% 감소했습니다. 이러한 결과는 공격에 대한 모델의 취약성을 강조하며, 머신러닝 시스템의 신뢰성을 저하시킬 수 있습니다.



### What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks (https://arxiv.org/abs/2411.03343)
- **What's New**: 이번 연구에서는 LLM (large language models)에서 'jailbreaks'의 안전성과 신뢰성 연구의 핵심 요소로서, 성공적인 jailbreak에 기여하는 프롬프트의 특징을 비교 분석하였습니다.

- **Technical Details**: 연구팀은 35가지 공격 방법으로부터 수집된 10,800개의 jailbreak 시도를 포함하는 데이터셋을 소개하였으며, 선형(linear)과 비선형(nonlinear) 방법을 비교하여 프롬프트의 성공적인 jailbreak을 지원하는 특징을 조사했습니다. 특히 비선형 프로브(non-linear probes)가 LLM을 기계적으로 jailbreak하는 데 사용될 수 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과, 특정 비선형 특징을 통해 성공적인 프롬프트와 실패한 프롬프트를 높은 정확도로 구별할 수 있었지만, 보유된 공격 방법에 대해서는 성능이 떨어지는 경향이 있었습니다. 또한, Gemma-7B-IT를 jailbreak 하는 데 있어 연구진의 접근 방식이 기존의 35가지 기술보다 더 신뢰할 수 있는 결과를 보였습니다.



### Towards evaluations-based safety cases for AI scheming (https://arxiv.org/abs/2411.03336)
- **What's New**: 이 논문에서는 최전선 AI 시스템의 개발자들이 AI 시스템이 재앙적인 결과를 초래할 가능성이 낮다는 것을 입증하는 구조화된 근거인 '안전 사례'(safety case)를 구성할 수 있는 방법을 설명합니다.

- **Technical Details**: 논문에서는 경계 모델로서 'scheming'에 대해 논의하며, AI 시스템이 의도적으로 잘못된 목표를 추구하고 실제 능력과 목적을 숨길 수 있는 가능성을 다룹니다. 제안하는 세 가지 주장은 다음과 같습니다: 1) AI 시스템이 scheming을 할 수 없다는 주장(Scheming Inability), 2) AI 시스템이 scheming을 통해 해를 끼칠 수 없다는 주장(Harm Inability), 3) AI 시스템의 제어 조치가 불가피한 결과를 예방할 수 있다는 주장(Harm Control)입니다.

- **Performance Highlights**: 논문은 AI 시스템이 개발자와 합리적으로 정렬되어 있다는 증거(Alignment)를 통해 안전 사례를 지원하는 방법도 논의하며, 이러한 안전 주장을 뒷받침하기 위해 필요한 여러 가정이 현재까지 확신 있게 충족되지 않았음을 지적합니다.



### Neural Network Prediction of Strong Lensing Systems with Domain Adaptation and Uncertainty Quantification (https://arxiv.org/abs/2411.03334)
Comments:
          Accepted to the Machine Learning for Physical Sciences workshop at NeurIPS 2024; 24 pages, 2 figures, 4 tables

- **What's New**: 이번 연구에서는 강력한 중력 렌즈링 데이터를 이용하여 Mean-Variance Estimators (MVE)와 비지도 도메인 적응 (UDA)을 결합한 첫 번째 연구를 수행하였습니다. 이를 통해 소스 도메인 데이터(노이즈가 없는)와 타겟 도메인 데이터(현대 우주론 조사의 노이즈를 모방)에서의 성능을 비교하였습니다.

- **Technical Details**: MVE는 데이터 레이블의 평균과 분산을 추정하며, 분산은 aleatoric (데이터) 불확실성의 제곱으로 표현됩니다. MVE 손실 함수는 β-NLL(negative log-likelihood)로 설정되며, UDA는 라벨이 없는 타겟 데이터를 사용하는 비지도 방법입니다. 이 연구에서 MVE와 UDA의 결합을 통해 이미지의 노이즈에 의해 구분되는 두 도메인에서의 성능을 비교하였습니다.

- **Performance Highlights**: UDA를 MVE에 추가함으로써 타겟 데이터에서의 정확도가 2배 향상되었습니다. 또한 UDA를 포함시킴으로써 aleatoric 불확실성 예측의 보정도 더욱 정확해졌습니다.



### log-RRIM: Yield Prediction via Local-to-global Reaction Representation Learning and Interaction Modeling (https://arxiv.org/abs/2411.03320)
Comments:
          18 pages, 8 figures

- **What's New**: 이 논문에서는 화학 반응의 수율(젤)의 정확한 예측을 위한 혁신적인 그래프 변환기 기반 프레임워크인 log-RRIM을 소개합니다. 이 접근법은 지역에서 글로벌 반응 표현 학습(local-to-global reaction representation learning) 전략을 구현하여 분자 수준 정보를 자세히 캡처하고 분자간 상호작용을 모델링 및 집계합니다.

- **Technical Details**: log-RRIM은 반응제(reactant)와 반응 중심(reaction center) 간의 상호작용에 집중하는 크로스 어텐션(cross-attention) 메커니즘을 통합하였습니다. 이 설계는 반응에서 화학 결합을 형성하고 분해하는 과정에 영향을 주는 반응제의 중요한 역할을 반영하며, 분자의 크기가 다양할 때에도 고르게 주목할 수 있도록 합니다.

- **Performance Highlights**: log-RRIM은 중간에서 고수율 반응에 대해 기존 방법들보다 우수한 성능을 보이며, 소규모 분자 조각들에 대한 민감성을 통해 화학 합성에서의 반응 계획 및 최적화를 위한 가치 있는 도구임을 입증합니다.



### Masked Multi-Query Slot Attention for Unsupervised Object Discovery (https://arxiv.org/abs/2404.19654)
Comments:
          Paper accepted for presentation at IJCNN 2024

- **What's New**: 본 논문은 DINO ViT 특징을 활용한 객체 중심의 비지도 객체 발견 전략을 제안합니다. 이는 배경을 선택적으로 무시하는 마스킹(masking) 기법을 통해 모델이 주목할만한 객체에 더 집중하도록 유도하며, 멀티 쿼리 슬롯 어텐션(multi-query slot attention) 방식으로 여러 개의 슬롯 세트를 학습하여 물체 탐지의 안정성을 높입니다.

- **Technical Details**: 제안하는 방법은 두 가지 주요 전략을 포함합니다. 첫째, DINO 특징에 대한 선택적 마스킹을 적용하여 배경으로부터 메시지(signal)를 차단하고 세밀한 객체 정보를 학습합니다. 둘째, 슬롯 어텐션을 확장하여 여러 개의 쿼리를 사용, 각 쿼리에서 독립적으로 슬롯을 학습합니다. 이 과정에서 여러 슬롯 세트를 훈련시키며, 테스트 시 헝가리안 매칭(Hungarian matching)을 통해 최종 슬롯을 결합합니다.

- **Performance Highlights**: PASCAL-VOC 2012 데이터셋에서의 실험 결과, 제안된 방법이 각 구성 요소의 중요성을 입증하며 객체 위치 탐지(object localization) 성능이 일관되게 향상됨을 보여주었습니다. 주목할 점은 선택적 마스킹과 멀티 쿼리 어텐션이 결합되어 성능을 높이고, 다양한 성능 지표를 통해 비지도 객체 발견의 유용성을 증명했습니다.



### FactTest: Factuality Testing in Large Language Models with Finite-Sample and Distribution-Free Guarantees (https://arxiv.org/abs/2411.02603)
- **What's New**: 대형 언어 모델(LLMs)의 사실성을 통계적으로 평가하는 새로운 프레임워크인 FactTest가 도입되었습니다. 이 프레임워크는 LLM이 질문에 대한 올바른 답을 높은 확률로 제공할 수 있는지를 검증합니다.

- **Technical Details**: FactTest는 가설 검정(hypothesis testing) 문제로 사실성 테스트를 공식화하며, 사용자 지정 유의 수준에서 Type I 오류를 제어합니다. 또한, mild 조건 하에서 Type II 오류를 강하게 제어할 수 있음을 증명하였습니다. 이 프레임워크는 모델에 구애받지 않으며, 이론적 보장을 제공합니다.

- **Performance Highlights**: FactTest는 질문-답변 및 다중 선택 기준에서 광범위한 실험을 통해 환각을 효과적으로 감지할 수 있음을 보여주며, 정확도가 40% 이상 향상된 결과를 나타냈습니다. 추가 학습이나 외부 데이터 출처 없이도 모델의 성능을 크게 개선했습니다.



New uploads on arXiv(cs.LG)

### Weighted Sobolev Approximation Rates for Neural Networks on Unbounded Domains (https://arxiv.org/abs/2411.04108)
- **What's New**: 이번 연구에서는 spectral Barron space에서의 함수 근사를 위해 weighted Sobolev 공간에서의 얕은 신경망의 근사 능력을 고려합니다. 기존 연구에서는 얕은 네트워크와 다양한 activation 함수 클래스를 사용하여 spectral Barron space를 효과적으로 근사할 수 있는 여러 경우를 다루었습니다. 또한, 현재 결과들의 한계는 주로 고려된 오류 측정치에 있으며, 이는 유한한 영역에서 Sobolev 공간에만 제한되어 있었습니다. 본 연구에서는 제한된 영역 및 Muckenhoupt weights 경우와 무한한 영역에서의 감소하는 weight를 허용하는 경우를 다룹니다.

- **Technical Details**: 최초로 더 일반적인 weighted Fourier-Lebesgue 공간을 weighted Sobolev 공간에 포함하는 이론적 결과를 제시하며, 이후 얕은 신경망의 근사율을 정립합니다. weighted Sobolev 공간에서의 근사율은 차원 저주(curse of dimensionality) 없이 도출됩니다. Muckenhoupt weights는 제한된 영역에서의 근사에 흥미로운 weights 클래스이며, 이를 통해 유한 요소 방법(finite element methods)에서 문제를 다룰 수 있는 장점이 있습니다. 이 연구는 이러한 weighted Sobolev 공간을 신경망의 문맥에서 다루고 있습니다.

- **Performance Highlights**: 본 연구는 얕은 신경망이 복잡한 함수 구조를 다루기 위한 효과적인 대안이 될 수 있음을 보여줍니다. 특히, 얕은 신경망은 차원 저주(curse of dimensionality)를 극복할 수 있는 잠재력을 지니며, 이를 통해 고차원 문제의 근접 해결을 가능하게 합니다. 실험적으로, 이 연구에서 제시하는 접근법이 기존의 전통적 방법들보다 우수한 성능을 발휘할 수 있는 가능성을 탐색합니다.



### How Transformers Solve Propositional Logic Problems: A Mechanistic Analysis (https://arxiv.org/abs/2411.04105)
- **What's New**: 이번 연구에서는 작은 Transformer 모델이 복잡한 논리 문제를 해결하는 내부 메커니즘과 이에 필요한 '계획' 및 '추론' 회로를 탐색합니다. 또한 Mistral 7B 모델을 통해 보다 큰 모델의 내부 구성요소를 분석하여 논리 문제 해결에 필요한 회로를 식별합니다.

- **Technical Details**: 연구는 Synthetic propositional logic 문제를 중심으로 진행되며, 두 가지 주요 실험을 포함합니다: (1) 작은 Transformer 모델에 대한 실험과 (2) 사전 훈련된 LLM인 Mistral-7B에 대한 실험. 활성화 패칭 (activation patching)을 통해 Mistral-7B의 특정 주목 헤드(attention heads) 역할을 분석하여 문제를 해결하는 데 필요한 회로를 밝혀냅니다.

- **Performance Highlights**: 작은 Transformer 모델은 다양한 추론 문제를 해결하는 데 있어 정보 흐름을 불균형하게 조정하는 '라우팅 임베딩(routing embeddings)'을 사용하고, Mistral-7B 모델에서는 각 규칙과 사실을 처리하는 특수화된 주목 헤드의 역할을 발견하였습니다. 이 연구는 작은 및 큰 Transformer의 새로운 측면을 체계적으로 드러내며, 모델이 어떻게 계획하고 추론하는지를 지속적으로 탐구합니다.



### Interpretable and Efficient Data-driven Discovery and Control of Distributed Systems (https://arxiv.org/abs/2411.04098)
- **What's New**: 이번 연구에서는 부분 미분 방정식(Partial Differential Equations, PDEs)으로 모델링된 시스템을 효과적으로 제어하기 위한 데이터 효율적이고 해석 가능한 Dyna 스타일의 모델 기반 강화학습(Model-Based Reinforcement Learning, RL) 프레임워크를 제안합니다.

- **Technical Details**: 이 방법은 비선형 동역학의 희소 식별(Sparse Identification of Nonlinear Dynamics, SINDy-C) 알고리즘과 오토인코더(autoencoder, AE) 프레임워크를 결합하여 PDE 상태와 행동의 차원 수를 줄이는 데 중점을 둡니다. 이를 통해 빠른 롤아웃을 가능하게 하여 대규모 환경 상호작용의 필요성을 줄이고, PDE 전진 동역학의 해석 가능한 잠재 공간(latent space) 표현을 제공합니다.

- **Performance Highlights**: 우리는 1차원 버거스 방정식(1D Burgers equation)과 2차원 나비에-스톡스 방정식(2D Navier-Stokes equations)이라는 두 가지 PDE 문제에 대해 이 방법을 검증했으며, 모델 프리(model-free) 기준선과 비교 분석을 통해 학습된 동역학을 종합적으로 분석했습니다.



### Multi-branch Spatio-Temporal Graph Neural Network For Efficient Ice Layer Thickness Prediction (https://arxiv.org/abs/2411.04055)
- **What's New**: 본 논문은 다중 분기 구조의 시공간 그래프 신경망을 개발하여, 얼음층 두께 정보를 활용해 깊은 얼음층의 두께를 예측하는 데 초점을 맞추고 있습니다. 기존의 퓨즈된 시공간 그래프 신경망과 비교했을 때, 제안된 네트워크는 효율성과 정확성에서 일관되게 우수한 성능을 보입니다.

- **Technical Details**: 제안된 네트워크는 GraphSAGE 프레임워크를 활용하여 시공간 특성을 학습하고, 시간적 변화를 포착하기 위해 시간적 컨볼루션(temporal convolution) 작업을 수행합니다. 다양한 작업에 특화된 네트워크의 다양한 부분이 별도의 분기로 구성되어 있습니다.

- **Performance Highlights**: 제안된 다중 분기 네트워크는 기존의 메서드와 비교하여 효율성과 정확성 모두에서 일관되게 더 나은 성능을 보였습니다. 특히, LSTM 구조 대신 도입된 게이티드 시간적 컨볼루션 블록이 효율성을 높이는 데 기여했습니다.



### Stepping Forward on the Last M (https://arxiv.org/abs/2411.04036)
- **What's New**: 이 논문에서는 고정 소수점 (fixed-point) 전진 기울기 (forward gradients)를 사용하여 엣지 (edge) 장치에서의 온디바이스 (on-device) 훈련의 실행 가능성을 조사합니다. 이전의 연구에서 제공된 이점에도 불구하고, 기울기 근사에 대한 불확실성을 해결하기 위해 다양한 딥 러닝 벤치마크에서 실험을 진행합니다.

- **Technical Details**: 전진 기울기를 활용하여 역전파 (backpropagation)를 우회하고, 출력 기울기를 추정하는 새로운 방법론인 Quantized Zeroth-order Forward Gradient (QZO-FF) 추정기를 제안합니다. 이를 통해 고정 소수점 공간에서 가중치의 미세 조정 및 업데이트의 과정을 체계적으로 정리합니다.

- **Performance Highlights**: 실험 결과, 고정 소수점 전진 기울기를 활용한 훈련이 엣지 장치에서 모델 맞춤화의 마지막 단계에서 실용적이고 실행 가능한 접근 방법임을 보여줍니다. 특정 아키텍처에 구애받지 않으면서도 딥 러닝 과제에서 경쟁력 있는 성능을 유지합니다.



### Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Res (https://arxiv.org/abs/2411.04034)
- **What's New**: 본 연구에서는 비정상적(non-stationary) 데이터 분포에 적응하는 신경망 학습 접근법을 제안합니다. 기존에는 정적 분포를 가정했으나, 분포가 변하는 상황에서의 학습을 가능하게 하고자 합니다.

- **Technical Details**: 제안된 접근법은 Ornstein-Uhlenbeck 프로세스를 기반으로 하며, 적응적 드리프트 파라미터를 사용하여 비정상성에 적응합니다. 이 드리프트는 파라미터를 초기화 분포로 끌어당기는 경향이 있어 소프트 파라미터 리셋 형태로 이해할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 비정상적인 감독 학습과 off-policy 강화 학습 설정에서 효과적으로 작동함을 보여 주었습니다.



### Multi-Scale and Multimodal Species Distribution Modeling (https://arxiv.org/abs/2411.04016)
Comments:
          Published at the CV4Ecology workshop at ECCV 2024 (this https URL)

- **What's New**: 본 논문에서는 Species Distribution Models (SDMs)에 대한 최신 연구 결과를 제시하며, 딥러닝 기법의 도입을 통해 환경 정보를 포함한 다중 모달 모델을 사용하는 접근 방식을 설명합니다. 특히, 다양한 스케일에서의 이미지 패치 크기가 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 본 연구는 CNN 기반의 모듈형 SDM 구조를 설계하여, 단일 및 다중 스케일 설정에서의 성능을 평가합니다. 다양한 해상도의 다중 모달 데이터를 처리할 수 있는 Late Fusion 방식을 통해 각 모달리티(modality)가 자신의 해상도에서 고려됩니다. GeoLifeCLEF 2023 벤치마크를 사용하여 공간적 범위(scale)와 성능 간의 관계를 조사했습니다.

- **Performance Highlights**: 연구 결과, 다중 스케일 및 다중 모달 접근 방식이 기존 모델보다 높은 정확도를 보여줍니다. 코드 및 모델 재현 데이터는 GitHub에서 확인할 수 있습니다.



### $k$NN Attention Demystified: A Theoretical Exploration for Scalable Transformers (https://arxiv.org/abs/2411.04013)
Comments:
          30 pages, 12 figures

- **What's New**: 이 연구는 $k$-Nearest-Neighbor ($k$NN) 주의 메커니즘의 이론적 프레임워크를 제시하며, 새로운 sub-quadratic 알고리즘을 제안하여 자기 주의(self-attention) 기울기의 근사화를 효율적으로 수행하는 방법을 탐구합니다.

- **Technical Details**: 자기 주의(self-attention)를 softmax 분포에 대한 기대값으로 재구성하고, lazy Gumbel sampling 기법을 활용하여 $k$NN 지수를 이용한 효율적인 근사를 구현합니다. 이 프레임워크를 통해, Markov Chain 기반의 추정 기법을 사용할 수 있는 novel sub-quadratic 알고리즘을 제안합니다.

- **Performance Highlights**: 제안된 알고리즘은 실험을 통해 훈련(training) 및 추론(inference) 과정에서 각각의 이점을 입증하였으며, $k$NN 주의가 Transformer 모델의 효율성을 높이는 데 기여함을 보여줍니다.



### Towards Resource-Efficient Federated Learning in Industrial IoT for Multivariate Time Series Analysis (https://arxiv.org/abs/2411.03996)
- **What's New**: 최근 산업 응용에서의 이상 탐지 및 결측 데이터 문제를 해결하기 위한 새로운 연합 학습(federated learning, FL) 프레임워크가 제안되었습니다. 이 방법은 데이터 압축 및 모델 축소를 통해 통신 비용과 계산 복잡성을 줄이는 데 기여할 수 있습니다.

- **Technical Details**: 연합 학습 프레임워크를 기반으로 한 본 연구에서는, 클라이언트가 고유한 센서를 장착하고 다차원(time series) 데이터를 포함하는 상황을 고려하여 모델 프루닝(pruning) 기법을 적용합니다. 고유한 센서로 측정된 단일 변수(univariate) 데이터를 활용하여 모델을 압축하고, 이후 서버 측에서 이를 통합하여 글로벌 모델을 생성하여 성능 저하를 최소화합니다.

- **Performance Highlights**: 제안된 방법은 결측값 대체 및 이상 탐지 문제에서 99.7% 이상의 높은 압축률을 달성하였으며, 성능 손실은 1.18% 미만에 그쳤습니다. 이는 기존 중앙 집중식(solution) 접근 방식에 비해 매우 효율적인 결과입니다.



### Customized Multiple Clustering via Multi-Modal Subspace Proxy Learning (https://arxiv.org/abs/2411.03978)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: Multi-Sub이라는 새로운 end-to-end multiple clustering 방법론을 제안하며, 이는 사용자 선호도를 표현하는 텍스트 프롬프트를 시각적 표현과 정렬하는 multi-modal subspace proxy learning framework을 포함합니다.

- **Technical Details**: Multi-Sub는 CLIP와 GPT-4의 시너지 능력을 활용하여, 사용자 관심사에 따라 이미지의 시각적 표현을 사용자 맞춤형으로 조정합니다. 이를 위해 LLMs로부터 생성된 proxy words를 사용하여 사용자의 특정 관심사에 대한 데이터 표현을 맞춤형으로 구현하는 방식을 사용합니다.

- **Performance Highlights**: 제안된 Multi-Sub 방법은 다양한 데이터셋에서 기존 방법보다 일관되게 더 나은 성능을 보여주며, 사용자 관심사를 정확하게 포착할 수 있음을 입증했습니다.



### Can Custom Models Learn In-Context? An Exploration of Hybrid Architecture Performance on In-Context Learning Tasks (https://arxiv.org/abs/2411.03945)
Comments:
          18 pages, 16 figures

- **What's New**: 이번 논문은 In-Context Learning (ICL) 현상을 연구하며, 특히 Multi-Headed Attention (MHA) 모델에서의 절대 위치 인코딩을 다룹니다. GPT-2와 LLaMa, LLaMa와 Mamba 간의 아키텍처적 차이에 대해 분석합니다.

- **Technical Details**: 이 연구는 Garg et al. (2022)와 Park et al. (2024)의 작업에 기반하여 GPT-2/LLaMa 혼합 모델과 LLaMa/Mamba 혼합 모델을 확장합니다. 이 과정에서 시퀀스 변환 블록과 ICL 성능 간의 상호작용을 조사합니다. 아키텍처 변경이 학습 효율성과 ICL 정확도에 미치는 영향도 논의하며, ICL 회귀 점수라는 메트릭을 제안합니다.

- **Performance Highlights**: 특정 하이브리드 모델에서 긍정적인 성과 향상이 관찰되었습니다. 모든 실험은 모듈화된 파이썬 패키지를 사용하여 수행되어 재현성과 확장성을 촉진합니다.



### Fine-tuning -- a Transfer Learning approach (https://arxiv.org/abs/2411.03941)
- **What's New**: 본 논문에서는 전통적인 통계 및 심층 학습 기반의 데이터 보간 방법들을 대체할 수 있는 모듈화된 데이터 보간 및 분류 파이프라인의 개발을 제안합니다. 이 접근 방식은 보간기(imputer)와 분류기(classifier)의 성능을 독립적으로 평가할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: 제안된 파이프라인은 최첨단 심층 학습 보간 모델의 기능을 활용하여 다운스트림(classification) 작업을 위한 최적화된 구성 요소로 만들어졌습니다. 구체적으로, autoencoders, RNNs 및 LSTMs와 같은 심층 신경망 아키텍처를 활용하여 복잡한 패턴을 캡처하고 결측값을 보완하는 데 초점을 맞췄습니다.  또한, 기존의 end-to-end 파이프라인과 달리, 각 구성 요소가 최적화되어 독립적으로 평가될 수 있도록 하여 평가와 성능 향상에 있어 투명성을 제공합니다.

- **Performance Highlights**: 이 연구의 주요 발견은 최적화된 보간기를 활용할 경우, 단순한 분류기(classifier)라도 비교 가능한 성능을 달성할 수 있다는 것입니다. 이 접근 방식은 다양한 결측 데이터 패턴을 다룰 수 있어 보간의 품질을 높이며, 의료 데이터 분석에 있어 신뢰성을 크게 향상시키는 가능성을 보여줍니다.



### GUIDE-VAE: Advancing Data Generation with User Information and Pattern Dictionaries (https://arxiv.org/abs/2411.03936)
- **What's New**: 이 논문에서는 사용자 정보를 활용하여 데이터 생성을 향상시키는 새로운 조건부 생성 모델인 GUIDE-VAE를 소개합니다. 기존의 생성 모델들이 사용자 특성을 무시해왔던 점을 극복하고, 사용자 임베딩을 통해 데이터 균형이 심각한 다중 사용자 환경에서의 성능을 향상시킵니다.

- **Technical Details**: GUIDE-VAE는 사용자 정보를 통합하여 다중 사용자 데이터에서 사용자 특정 지침을 제공하며, 패턴 사전 기반 공분산 구성 (PDCC)을 통해 복잡한 특징 의존성을 캡쳐하여 생성된 샘플의 현실성을 향상시킵니다. 특히, PDCC는 VAEs에서 일반적으로 발견되는 노이즈와 과도 마찰 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: GUIDE-VAE는 다중 사용자 스마트 미터 데이터 세트에서 평가되었으며, 정량적인 결과는 합성 데이터 생성 및 누락된 기록 보간 작업 모두에서 효과적으로 수행됨을 보여줍니다. 질적 평가를 통해서는 GUIDE-VAE가 보다 더 그럴듯하고 덜 노이즈가 섞인 데이터를 생산함을 입증하였습니다.



### Interactions Across Blocks in Post-Training Quantization of Large Language Models (https://arxiv.org/abs/2411.03934)
- **What's New**: 이 연구에서는 Post-training quantization을 통해 대규모 언어 모델의 양자화 성능을 개선하기 위한 두 가지 다중 블록 파인튜닝 전략을 제안하고 비교했습니다.

- **Technical Details**: 기존의 양자화 방법은 서로 독립적이라고 가정하며 하위 구조의 지식을 무시했습니다. 이에 반해, 첫 번째 전략은 양자화된 여러 블록을 공동 최적화하여 블록 간의 상관관계를 포착합니다. 두 번째 전략은 후속 블록의 정보를 활용하여 다운스트림(pre-activation)의 오류를 최소화합니다.

- **Performance Highlights**: 모델에 따라 이러한 방법의 효과가 달라졌으며, 일부 모델에서는 아무런 영향을 미치지 않았지만 다른 모델에서는 상당한 성과 향상을 보였습니다.



### Quantum Algorithm for Sparse Online Learning with Truncated Gradient Descen (https://arxiv.org/abs/2411.03925)
Comments:
          31 pages, 1 table, 4 algorithms

- **What's New**: 이번 연구는 로지스틱 회귀(logistic regression), SVM(Support Vector Machine), 최소 자승(least squares) 방법을 적용한 양자(quantum) 희소 온라인 학습 알고리즘을 개발하였다. 이는 Langford, Li, Zhang(2009)의 방법을 기반으로 하여, 트렁케이티드 그래디언트 하강법(truncated gradient descent)을 통해 희소성을 얻는 데 기여하며, 온라인 손실(regret)도 줄인다.

- **Technical Details**: 이 알고리즘은 효율적인 양자 입력 접근을 활용하여, 문제의 차원에 따라 시간 복잡도에서 제곱(Quadratic) 속도 향상을 달성할 수 있다. 또한, 반복 수 T에 대해 $O(1/\\sqrt{T})$의 손실을 유지한다.

- **Performance Highlights**: 제안된 알고리즘은 기존의 방법들에 비해 차원 수가 증가할 때 비약적인 성능 향상을 보여주며, 특정 실험을 통해 이론적인 결과들이 성공적으로 입증되었다.



### Retentive Neural Quantum States: Efficient Ans\"atze for Ab Initio Quantum Chemistry (https://arxiv.org/abs/2411.03900)
Comments:
          16 pages, 1 figure, to be submitted for peer-reviewed publication

- **What's New**: 본 논문은 RetNet (Retentive Network)을 도입하여 전통적인 변환기(transformer) 대신 NQS (Neural Network Quantum States)에서의 전자 기초 상태 문제를 해결하는 새로운 접근 방식을 제안합니다. RetNet은 데이터 병렬 처리로 훈련을 수행하며, 추론(inference) 중 순환적으로 데이터를 처리하여 시간 복잡도를 개선합니다.

- **Technical Details**: RetNet은 전통적인 변환기와는 달리, 시퀀스의 길이에 대한 시간 복잡도를 줄여줍니다. RetNet은 훈련 시 데이터 병렬 처리, 추론 시에는 순환 처리를 통해 더욱 효율적으로 작동하며, 전통적인 변환기와 비교해 특정 문제 크기 초과 시 시간 복잡도가 우수합니다. 또한, 변동 신경 드로이징(variational neural annealing)을 통해 RetNet의 표현력이 향상됩니다.

- **Performance Highlights**: RetNet을 NQS 적용에 성공적으로 도입하여, 여러 기본 분자에 대한 실험결과에서 성능이 기존 방법과 유사하거나 더 나은 정확도를 기록하였습니다. 이 논문은 NQS가 전통적인 방법과 함께 실질적 운영 개선을 위한 여러 유망한 방향을 제시하고 있습니다.



### Calibrating for the Future:Enhancing Calorimeter Longevity with Deep Learning (https://arxiv.org/abs/2411.03891)
- **What's New**: 본 연구는 입자 물리학 실험에서 사용되는 칼로리미터(calorimeters)의 보정(calibration) 과정을 개선하기 위해 딥 러닝(deep learning) 전략을 제안합니다. 특히 Wasserstein GAN(WGAN) 방법론을 활용하여 노화나 기타 요인으로 인한 데이터의 불일치를 정교하게 보정할 수 있게 됩니다.

- **Technical Details**: Wasserstein 거리(Wasserstein distance)를 사용하여 손실(loss) 계산을 수행하며, 이 혁신적인 접근법은 높은 정밀도(high precision)를 달성하면서 필요한 사건 수와 자원을 크게 줄여주어 절대적 오차(absolute errors)를 효과적으로 최소화합니다. 10000개의 단일 10 GeV 파이온(pion) 사건에 대한 몬테카를로(Monte Carlo) 시뮬레이션을 통해 성능을 평가하였습니다.

- **Performance Highlights**: 이 연구를 통해 개발된 모델은 칼로리미터의 작동 수명을 연장하고 데이터의 정확성 및 신뢰성을 지속적으로 보장하며, 과학적 발견을 위한 데이터 무결성이 필수적인 실험에 특히 유용합니다.



### EXPLORA: Efficient Exemplar Subset Selection for Complex Reasoning (https://arxiv.org/abs/2411.03877)
- **What's New**: 이 논문에서는 복잡한 추론 질문(answering reasoning-based complex questions)을 해결하기 위한 새로운 알고리즘, EXPLORA를 제안합니다.

- **Technical Details**: EXPLORA는 static exemplar subset을 선택하기 위한 탐색 방법으로, 신뢰도 정보(confidence information)를 포함하지 않고도 exemplars의 파라미터(Parameter)를 추정할 수 있도록 설계되었습니다.

- **Performance Highlights**: EXPLORA는 최신 방법들(state-of-the-art methods)의 호출 수를 약 11%로 줄이며, 성능 향상(performance improvement)에서는 12.24%의 유의미한 개선을 이루었습니다.



### Reconsidering the Performance of GAE in Link Prediction (https://arxiv.org/abs/2411.03845)
- **What's New**: 최근 그래프 신경망(GNN)의 본질을 이해하고, Graph Autoencoders (GAE)의 성능을 최적화하여 복잡한 모델의 성능에 필적할 수 있음을 보여주었습니다. 이 연구는 하이퍼파라미터 조정 및 직교 임베딩(orthogonal embedding)과 선형 전파(linear propagation) 기법을 활용한 새로운 방법론을 제안합니다.

- **Technical Details**: 본 연구는 GAE 모델을 OGB 벤치마크에서 평가하며, 새로운 모델의 기술을 활용하고 하이퍼파라미터를 포괄적으로 조정하여 비교 분석합니다. 특히, GAE의 제한 사항을 극복하기 위해 직교 벡터의 선형 전파를 적용하고, 특히 ogbl-ddi 및 ogbl-ppa 데이터셋에서 성능을 개선했습니다.

- **Performance Highlights**: GAE의 최적화된 모델은 더 복잡한 GNN 모델과 경쟁할 수 있는 성능을 발휘하지만, 계산 효율성은 더 뛰어난 것으로 나타났습니다. 이는 새로운 GNN 기술들이 직면한 표현력의 제한성을 해결하는 방향으로 기여하고 있습니다.



### Flexible task abstractions emerge in linear networks with fast and bounded units (https://arxiv.org/abs/2411.03840)
- **What's New**: 본 연구는 신경망에서 과거 정보를 잊기보다는 작업을 추상화(task abstraction)하여 유연하게 대응하는 능력을 밀접한 관점에서 분석합니다. 연구팀은 선형 게이트 네트워크(linear gated network) 모델을 통해 작업 전환(task switching)의 유연한 및 망각적 모드를 제시하였으며, 이는 뇌에서의 인지 유연성을 이해하는 데 도움을 줍니다.

- **Technical Details**: 실험에서는 게이트가 빠른 시간 척도(faster timescale)로 작동하며, 비음수(non-negativity) 및 제한된 활동(bounded activity)과 같은 신경 세포(neuron-like) 제약을 둔 선형 게이트 네트워크를 사용하였습니다. 연구는 작업 블록 내에서의 빠른 적응 속도와 효율적인 지식 보호를 통한 가중치 전문화(weight specialization) 과정을 보여주며, 이는 작업 별로 모듈을 구성하고 그에 맞춘 게이트 표현을 생성합니다.

- **Performance Highlights**: 연구의 결과는 선형 및 비선형 네트워크에서 조합적 일반화(compositional generalization)를 지원하는 작업 추상화(task abstractions)의 발견을 포함하고 있으며, 이는 동적 학습과 함께 작업별 모듈 구성을 이루는 중요한 발견으로 평가됩니다. 추가적으로, 더욱 긴 작업 블록에 따른 데이터 분포 변화(data distribution shifts)와 연관된 최초의 시뮬레이션을 제시함으로써 기존의 인간 행동 연구에 대한 명확한 비교를 제공합니다.



### Hybrid Transfer Reinforcement Learning: Provable Sample Efficiency from Shifted-Dynamics Data (https://arxiv.org/abs/2411.03810)
- **What's New**: 본 연구에서는 하이브리드 전이 강화 학습(HTRL) 설정을 제안하여, 목표 환경(target environment)에서 에이전트가 이전 데이터(offline data)를 효과적으로 활용하는 방법을 탐구합니다.

- **Technical Details**: 하이브리드 전이 강화 학습(HTRL)은 에이전트가 시프트된 다이나믹스(shifted dynamics)를 가진 소스 환경(source environment)에서 오프라인 데이터를 이용하여 목표 환경에서 학습하는 접근 방식입니다. 연구에서는 다이나믹스 변화의 정보를 미리 알고 있을 때, HySRL이라는 전이 알고리즘(transfer algorithm)을 설계하여 샘플 복잡도(sample complexity)를 문제에 따라 조정할 수 있음을 보였습니다.

- **Performance Highlights**: HySRL 알고리즘은 순수 온라인 강화 학습(pure online RL)을 초월하여, 최신 온라인 RL 기반의 성과를 능가하는 것으로 나타났습니다.



### Overcoming label shift in targeted federated learning (https://arxiv.org/abs/2411.03799)
- **What's New**: 이번 연구에서는 FedPALS라는 새로운 모델 집계 기법을 제안하여 label shift 문제를 해결합니다. FedPALS는 중간 서버에서 목표 label 분포에 대한 지식을 활용하여 모델 업데이트를 보장합니다.

- **Technical Details**: FedPALS는 클라이언트 모델의 볼록 조합을 최적화하여 목표 도메인에 적합한 집계 모델을 생성합니다. 이는 Stochastic Gradient Descent(SGD) 하에서 편향 없는 업데이트를 보장합니다.

- **Performance Highlights**: FedPALS는 이미지 분류 실험에서 전통적인 방법들보다 일관되게 뛰어난 성능을 보였으며, 극단적인 클라이언트 희소성 상황에서도 성능 저하 문제를 완화했습니다.



### The N-Grammys: Accelerating Autoregressive Inference with Learning-Free Batched Speculation (https://arxiv.org/abs/2411.03786)
- **What's New**: 본 연구에서는 언어 모델의 autoregressive generation 속도를 높이기 위해 학습이 필요 없는 저비용 draft 전략인 $N$-grams을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 방법은 모델 가중치와 문맥(context)에서 얻은 $N$-grams을 사용하여, 기본 모델의 다음 토큰 예측이 이 간단한 전략들의 상위 예측(top prediction)과 거의 같지 않지만, 종종 상위 $k$ 예측(predictions) 내에 존재한다는 점을 관찰합니다.

- **Performance Highlights**: 단순한 전략들의 조합을 통해 다양한 작업에서 상당한 추론 속도 증가(inference speedups)를 달성할 수 있음을 보여줍니다. 전체 성능은 더 복잡한 방법들과 비교할 만하며, 비싼 전처리나 기본 모델 수정 없이도 원활한 'plug-and-play' 통합(integration)이 가능합니다.



### A Bayesian Approach to Data Point Selection (https://arxiv.org/abs/2411.03768)
- **What's New**: 본 논문은 Bayesian 접근 방식을 이용하여 데이터 포인트 선택(Data Point Selection, DPS) 문제를 새롭게 제시합니다. 이러한 접근은 메모리와 계산 요구가 많고 이론적으로 결함이 있는 기존의 bi-level optimization (BLO) 방법의 대안으로 등장합니다. 이를 통해 DPS 문제를 postier inference 관점에서 바라보며, 모든 학습 작업에서 DPS 관련 문제들을 해결할 수 있는 유연한 모델을 제안합니다.

- **Technical Details**: 제안된 방법은 Stochastic Gradient Langevin MCMC 샘플링을 사용하여 주요 신경망과 인스턴스별 가중치를 공동으로 학습합니다. 이 과정은 mini-batch에 대해서도 수렴성을 보장하며, 업데이트 방정식은 기존의 SGD에 비해 효율적입니다. 주요 데이터셋과 메타 데이터셋을 사용하여 효과적으로 훈련을 수행하며, Bayesian 모델을 통해 joint posterior를 추정합니다.

- **Performance Highlights**: 이 논문은 시각 및 언어 도메인에서 다양한 학습 작업을 통해 제안된 기법의 성공적인 성능을 입증합니다. 특히 LLM(대형 언어 모델)에 대한 자동화된 교육 세트 최적화를 보여주며, 수십억 개의 파라미터를 가진 LLM에서도 잘 작동함을 입증합니다. 이 접근법의 코드가 제공되어 연구자들이 쉽게 사용할 수 있도록 하였습니다.



### Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions (https://arxiv.org/abs/2411.03755)
- **What's New**: 이번 논문은 비정렬 다중 도메인 데이터에서 잠재적 콘텐츠(content) 및 스타일(style) 변수를 식별하는 새로운 분석 프레임워크를 제안합니다. 기존 연구의 제한된 조건을 극복하고, 콘텐츠와 스타일의 차원을 미리 알 필요 없이도 도메인 변환 및 데이터 생성에 대한 유용성을 보여줍니다.

- **Technical Details**: 본 연구는 교차 도메인 잠재 분포 매칭(latent distribution matching, LDM)을 통한 콘텐츠-스타일 식별 기준을 제안합니다. 이 기준은 비선형 혼합 모델(nonlinear mixture model)에서도 적용 가능하며, 기존 연구에서 요구되던 요소 간 상호 독립성(component-wise independence) 가정이 필요하지 않습니다. LDM은 다양한 도메인에서 효율적으로 작동하면서도, 계산 자원을 크게 절약할 수 있는 GAN 손실(loss)로 재구성됩니다.

- **Performance Highlights**: 이론적 주장을 뒷받침하는 실험을 통해 이미지 번역(image translation) 및 생성(image generation) 작업에서의 성능을 확인했습니다. 이 연구는 실용적 설정에서의 불확실한 잠재 차원에서도 콘텐츠-스타일 식별 가능성을 유지할 수 있음을 입증했습니다.



### Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length (https://arxiv.org/abs/2411.03753)
- **What's New**: 이 논문에서는 기존의 상징 회귀(symbolic regression) 방법의 성능 문제를 해결하기 위해 최소 설명 길이(minimum description length)에 기반한 새로운 탐색 목표를 제안하고 있습니다. 이 새로운 접근법은 목표 공식(target formula)에 근접할수록 설명 길이가 점진적으로 감소하도록 설계되었습니다.

- **Technical Details**: 이 연구는 MDLformer라는 신경망(neural network)을 설계하여 데이터에 대한 최소 설명 길이를 추정합니다. SR4MDL이라는 상징 회귀 방법을 구현하여 MDLformer의 출력을 탐색 목표로 사용합니다. 이를 통해 기존의 방법들보다 훨씬 효과적으로 공식을 복구할 수 있습니다.

- **Performance Highlights**: 이 방법은 133개의 문제를 포함한 두 개의 벤치마크 데이터셋에서 약 50개의 공식을 성공적으로 복구했으며, 기존 최첨단 방법보다 43.92% 더 높은 성능을 보였습니다. 또한, 122개의 블랙박스 문제에서 낮은 모델 복잡도로 높은 R² 점수를 달성하였습니다.



### Deferred Poisoning: Making the Model More Vulnerable via Hessian Singularization (https://arxiv.org/abs/2411.03752)
- **What's New**: 최근 연구에서 기존의 데이터 오염 공격이 그동안 알려진 것보다 덜 위협적일 수 있음을 밝히고, 새로운 유형의 공격인 Deferred Poisoning Attack (DPA)을 소개했습니다. 이 공격은 훈련 및 검증 단계 동안 모델이 정상 작동하도록 하면서도 배포 단계에서 모델의 강건성을 크게 약화시키는 방식입니다.

- **Technical Details**: DPA는 손상된 데이터셋으로 훈련된 모델이 정상적인 데이터셋에서와 비슷한 성능을 발휘하도록 강요함으로써 도움이 됩니다. 또한, 각 샘플 주변에서 발생하는 로컬 곡률을 크게 확대하여 손상된 모델이 작은 변동에 대해 민감하게 반응하도록 합니다. 이 과정에서 Singularization Regularization 항을 통해 모델이 최적점에서 특이한 Hessian 정보를 갖도록 설계합니다.

- **Performance Highlights**: DPA는 기존 데이터 오염 방법보다 훨씬 낮은 공격 비용을 수반하면서도 우수한 전이성과 강건성을 보여줍니다. 실험을 통해 검증된 모델은 자연 잡음으로 인한 새로운 시나리오에서도 이 공격에 대해 더욱 취약해질 수 있음을 확인했습니다.



### Optimal Defenses Against Gradient Reconstruction Attacks (https://arxiv.org/abs/2411.03746)
Comments:
          The code for this project is available at this https URL

- **What's New**: 본 논문은 Federated Learning (FL) 환경에서 gradient reconstruction 공격으로부터의 데이터 유출을 방지하기 위해 새로운 방어 메커니즘을 제안합니다. 구체적으로, 최적의 gradient noise 추가와 gradient pruning 방법을 통해 데이터 유출과 모델 유틸리티 간의 균형을 최적화합니다.

- **Technical Details**: Gradient reconstruction 공격(GRA)에 대한 취약성을 해결하기 위해, 본 연구는 데이터 유출을 방지하는 한편, 모델의 유틸리티를 최대화하는 두 가지 방어 기법을 제안합니다: Optimal Gradient Noise와 Optimal Gradient Pruning. 이러한 방법들은 각 파라미터에 최적화되어 데이터 보호 수준을 높입니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법들이 기존의 Gradient Noise 및 Gradient Pruning 방법보다 더 우수한 데이터 보호 효과를 나타내며, 동시에 모델의 유틸리티 또한 개선되었음을 확인하였습니다.



### Graph Neural Networks with Coarse- and Fine-Grained Division for Mitigating Label Sparsity and Nois (https://arxiv.org/abs/2411.03744)
- **What's New**: 본 논문에서는 레이블의 희소성과 노이즈 문제를 완화하기 위해 	extbf{G}raph 	extbf{N}eural 	extbf{N}etwork with 	extbf{C}oarse- and 	extbf{F}ine-	extbf{G}rained 	extbf{D}ivision (GNN-CFGD) 모델을 제안합니다. 이 모델은 깨끗한 레이블과 노이즈가 있는 레이블을 구분하는 데 Gaussian Mixture Model (GMM)을 사용합니다.

- **Technical Details**: GNN-CFGD는 노이즈 레이블의 부정적인 영향을 줄이고자 부담과 세부적인 구분을 통해 레이블의 희소성과 노이즈를 완화합니다. 이 방법은 unlabeled nodes를 cleanly labeled nodes와 연결하여 그래프를 재구성하는 방식을 취합니다. 또한, confidence 기반으로 noisy labeled nodes와 unlabeled nodes를 두 개의 후보 집합으로 세분화합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, GNN-CFGD 모델이 기존의 GNN 모델들에 비해 우수한 성능 및 강인성을 가진 것을 보여주었습니다. 특히 노이즈가 있는 레이블에 효과적으로 대응하고, unlabeled nodes의 감독 전파를 증진시키는 데 성공했습니다.



### Adaptive Consensus Gradients Aggregation for Scaled Distributed Training (https://arxiv.org/abs/2411.03742)
- **What's New**: 이 논문은 동기식 데이터 병렬 처리 환경에서의 분산 기울기 집계 문제를 새롭게 분석하고, 하위 공간 최적화(subspace optimization) 관점에서 접근합니다. 기울기 집계 문제를 목표 인식 하위 공간 최적화 문제로 정식화하여, 하위 공간 계수(subspace coefficients)에 기반한 효율적인 기울기 가중치(weighting scheme)를 도출하였습니다.

- **Technical Details**: 분산 최적화 환경에서 기울기 집계는 각 작업자가 서로 다른 데이터의 하위 집합을 처리하고, 계산된 기울기를 중앙 모델에 집계하는 과정입니다. 기존의 기울기 평균화(gradient averaging) 방법 대신 비편향(unbiased) 추정기를 사용하는 하위 공간(momentum) 모멘텀을 도입하여 수렴 속도를 높이고 있습니다. 논문은 또한, 모멘텀을 통해 기울기 집계 시 통계적 비편향성을 유지하면서 성능을 개선합니다.

- **Performance Highlights**: 제안된 방법은 여러 MLPerf 작업에서 기울기 평균화 방법보다 현저한 성능 향상을 보였으며, 통신(computation) 및 계산(computational) 복잡성이 매우 낮아 효율적입니다. 이 방법은 하이퍼파라미터 튜닝(hyper-parameter tuning)이나 데이터 병렬 처리 설정의 수정 없이도 유용하게 적용될 수 있습니다.



### Human-in-the-Loop Feature Selection Using Interpretable Kolmogorov-Arnold Network-based Double Deep Q-Network (https://arxiv.org/abs/2411.03740)
Comments:
          Submitted to a journal under IEEE Transactions series

- **What's New**: 이 연구에서는 강화 학습( Reinforcement Learning )을 사용하여 동적인 인스턴스별 특징 선택( feature selection ) 프레임워크를 제안합니다. 제안된 방법론은 Double Deep Q-Network ( DDQN )과 Kolmogorov-Arnold Network ( KAN )를 통합하여 인공지능 모델의 해석 가능성과 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 KAN-DDQN 모델은 데이터 인스턴스별로 특징 하위 집합을 반복적으로 정제하기 위해 시뮬레이션된 인간 피드백과 확률적 분포 기반 샘플링(Beta)을 활용합니다. 이 과정에서 전문가의 실시간 피드백을 대신하여 시뮬레이션된 피드백을 사용하여, 인스턴스에 따라 독특한 특징 집합을 선택합니다. KAN은 신경망의 해석성을 높이는 동시에 MLP보다 숨겨진 레이어에서 4배 적은 뉴런을 사용하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: KAN-DDQN 모델은 MNIST 데이터셋에서 93%, FashionMNIST 데이터셋에서 83%의 테스트 정확도를 기록하여 전통적인 MLP-DDQN 모델보다 최고 9%의 정확도 향상을 보였습니다. 특징 선택을 하지 않은 모델은 MNIST에서 단 58%, FashionMNIST에서 64%의 성과를 나타내어, 제안된 프레임워크의 효과성을 강조합니다.



### Reducing Hyperparameter Tuning Costs in ML, Vision and Language Model Training Pipelines via Memoization-Awareness (https://arxiv.org/abs/2411.03731)
- **What's New**: 이번 논문에서는 머신러닝 파이프라인에서 하이퍼파라미터 튜닝 비용을 줄이는 새로운 방법론을 제시합니다. 기존의 튜닝 알고리즘에 비해 더욱 효율적으로 하이퍼파라미터 후보를 평가할 수 있는 '메모이제이션 인식' 베이지안 최적화(Bayesian Optimization, BO) 알고리즘 EEIPU를 소개합니다.

- **Technical Details**: EEIPU는 파이프라인 캐싱 시스템과 병행하여 작동하여, 같은 GPU-일(GPU-days) 내에 더 많은 하이퍼파라미터 후보를 평가할 수 있도록 설계되었습니다. 이 알고리즘은 파이프라인의 중간 단계에서 결과를 메모이즈(캐슁)하여 재사용함으로써, 검색 비용을 줄이는 동시에 더 나은 품질의 하이퍼파라미터를 제공합니다. 또한 이 연구에서는 T5 아키텍처(architecture)와 같은 다양한 모델에 대한 벤치마크를 수행했습니다.

- **Performance Highlights**: EEIPU는 동일한 예산으로 평균 103% 더 많은 하이퍼파라미터 후보를 생성하며, 검증 성능(Validation Metric)도 평균 108% 향상됩니다. 이는 모델 훈련 및 파이프라인 성능을 획기적으로 개선하는 결과를 도출합니다.



### NeurIPS 2023 Competition: Privacy Preserving Federated Learning Document VQA (https://arxiv.org/abs/2411.03730)
Comments:
          27 pages, 6 figures

- **What's New**: 본 논문은 Privacy Preserving Federated Learning Document VQA (PFL-DocVQA) 대회의 내용을 다루고 있으며, 커뮤니티의 도전 과제로서 실생활 사용 사례인 인보이스 처리에 대한 개인 정보 보호 및 통신 효율성 높은 솔루션을 개발하는 것을 목표로 하였습니다.

- **Technical Details**: 대회는 실제 인보이스 문서 데이터셋과 함께 질문 및 답변을 제공하며, 정보 추출 및 문서 이미지에 대한 추론을 요구합니다. 참가자들은 사전 훈련된 문서 시각적 질문 답변 모델을 조정하여 연합 학습 환경을 구축하였으며, 개인 정보 보호를 위한 차별적 개인 정보 보호(differential privacy, DP)를 적용하였습니다. 이 대회는 문서 분석, 개인 정보 보호 및 연합 학습 자료의 교류를 위한 새로운 테스트베드 역할을 하였습니다.

- **Performance Highlights**: 대회 참가자들은 통신 비용을 줄이면서도 최소 효용 기준을 유지하는 방법과 각 문서 제공자의 모든 정보를 보호하기 위한 차별적 개인 정보 보호 방법을 제안하였습니다. 대회 분석은 향후 개인 정보 중심의 연합 학습 도전을 성공적으로 실행하는 데에 대한 모범 사례와 권장 사항을 제공합니다.



### PropNEAT -- Efficient GPU-Compatible Backpropagation over NeuroEvolutionary Augmenting Topology Networks (https://arxiv.org/abs/2411.03726)
- **What's New**: 이 논문에서는 NEAT의 빠른 백프로파게이션 구현인 PropNEAT을 소개합니다. 이 방법은 네트워크의 유전체 그래프를 계층 기반 구조에 이중 방향으로 매핑하여 NEAT 유전체를 보존하면서 GPU 백프로파게이션을 효율적으로 가능하게 합니다.

- **Technical Details**: PropNEAT 알고리즘은 GPU에서 NEAT로 생성된 신경망의 가중치를 학습시키기 위해 백프로파게이션을 효율적으로 적용합니다. 이 알고리즘은 NEAT의 그래프 토폴로지를 계층 기반 구조로 매핑하여 텐서와의 양방향 매핑을 가능하게 합니다. PropNEAT의 설계는 노드 관계를 고려한 그래프 탐색 및 분석 단계를 포함하여 계층 구조와 연결성을 계산하고, 이를 통해 효율적인 텐서 표현을 만듭니다.

- **Performance Highlights**: PropNEAT은 Penn Machine Learning Benchmarks 데이터베이스의 58개의 이진 분류 데이터셋에서 평가된 결과, 우수한 성능을 보였으며, 기본적인 백프로파게이션 방법보다 상당히 빠르고, 원래 NEAT 구현보다도 성능이 뛰어났습니다. PropNEAT은 네트워크 깊이에 따라 선형적으로 훈련 시간이 확장되며, 저전력 컨텍스트에서의 적용 가능성도 보여 줍니다.



### Generalized Trusted Multi-view Classification Framework with Hierarchical Opinion Aggregation (https://arxiv.org/abs/2411.03713)
- **What's New**: 이 논문에서는 기존의 신뢰성 있는 다중 뷰 분류 방법에 대한 한계를 극복하기 위해 계층적 의견 집합(hierarchical opinion aggregation) 프레임워크를 제안합니다. 이를 통해 새로운 신뢰성 다중 뷰 학습 방식을 도입하였습니다.

- **Technical Details**: 제안된 GTMC-HOA는 두 단계의 집합 과정으로 구성되며, 첫 번째 단계는 개별 뷰 내 정보의 집합인 intra-view aggregation, 두 번째 단계는 서로 다른 뷰들 간의 집합인 inter-view aggregation입니다. 이 과정에서 Dempster의 조합 규칙과 주의(attention) 메커니즘을 사용하여 집합을 수행합니다.

- **Performance Highlights**: 여러 실험 결과 GTMC-HOA가 기존의 최첨단 신뢰성 있는 방법들보다 우수한 성능을 보임을 보여줍니다.



### Beyond Model Adaptation at Test Time: A Survey (https://arxiv.org/abs/2411.03687)
- **What's New**: 최근 기계 학습 알고리즘은 훈련 샘플과 테스트 샘플이 동일한 분포에서 가져온다는 가정 하에 성공을 거두어왔습니다. 그러나 테스트 분포가 훈련 분포와 차이를 보이기 시작하면 이러한 알고리즘은 취약해집니다. 본 논문에서는 Test-time adaptation(테스트 시간 적응)이라는 새로운 학습 패러다임을 소개하며, 이를 통해 모델이 훈련 데이터에서 학습한 후 테스트 시간에 목표 데이터를 적응시킬 수 있다는 점을 강조하고 있습니다.

- **Technical Details**: Test-time adaptation은 도메인 적응(domain adaptation)과 도메인 일반화(domain generalization)의 장점을 결합한 방법입니다. 이 방법에서는 소스 데이터에서만 모델을 학습하고, 테스트 시점에서 타겟 데이터로 모델을 적응시킵니다. 본 논문은 400개 이상의 최근 연구를 포괄적으로 검토하며, 테스트 시간 적응 방법을 모델, 추론, 정규화, 샘플, 프롬프트의 다섯 가지 범주로 나누어 자세히 분석합니다.

- **Performance Highlights**: Test-time adaptation 알고리즘은 형태학적 변화에 따른 분포 이동을 효과적으로 처리할 수 있는 가능성을 보여줍니다. 이 알고리즘은 실제에서 발생하는 온라인 또는 제한된 테스트 데이터 상황에서 효율적으로 적용가능하며, 다양한 응용 프로그램에서 기계 학습 모델의 성능과 견고성을 향상시키는 잠재력을 가지고 있습니다.



### Multi-model Ensemble Conformal Prediction in Dynamic Environments (https://arxiv.org/abs/2411.03678)
- **What's New**: 본 논문에서는 기존의 adaptive conformal prediction 방식의 한계를 극복하기 위해 새로운 adaptive conformal prediction 프레임워크를 제안하였습니다. 이 프레임워크는 여러 후보 모델 중에서 즉시 모델을 선택하여 예측 집합을 생성합니다.

- **Technical Details**: 제안된 알고리즘 'Strongly Adaptive Multimodel Ensemble Online Conformal Prediction (SAMOCP)'은 동적 환경에서의 데이터 분포 변화에 적응하여 여러 모델을 동시에 활용합니다. 모델의 성능에 따라 적절한 모델을 동적으로 선택하여 예측 집합을 생성합니다. 이 방법은 강력한 적응적 후회를 보이며 유효한 커버리지를 유지합니다.

- **Performance Highlights**: 실제와 합성 데이터셋에 대한 실험 결과, SAMOCP는 기존 방법들보다 더 효율적인 예측 집합을 지속적으로 생성하면서 목표 커버리지 확률에 밀접하게 부합하는 성능을 보였습니다.



### Can Graph Neural Networks Expose Training Data Properties? An Efficient Risk Assessment Approach (https://arxiv.org/abs/2411.03663)
Comments:
          In NeurIPS'24

- **What's New**: 본 연구에서는 그래프 속성 추론 공격(Graph Property Inference Attack)을 분석하고, 모델 공유로 인한 민감한 정보 유출 가능성을 조사합니다. 기존 방식의 비효율성을 개선하기 위해, 적은 수의 모델만으로도 효과적인 공격을 수행할 수 있는 방법을 제안합니다.

- **Technical Details**: 우리는 모델 근사화 기법(Model Approximation Techniques)을 활용하여, 소량의 모델 학습을 기반으로 다수의 근사 그림자를 생성하는 효율적인 그래프 속성 추론 공격 방법을 개발하였습니다. 이를 위해 구조 인식 랜덤 워크 샘플링(Structure-aware Random Walk Sampling)과 편집 거리(Edit Distance)를 사용하여 다각적 그래프를 샘플링하고, 근사 모델의 오류를 정량화합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 공격 방식보다 평균적으로 2.7% 높은 공격 정확도(Attack Accuracy)와 4.1% 높은 ROC-AUC를 기록하며, 6.5배 더 빠른 성능을 보였습니다. 평균적으로 기존 공격은 700개의 그림자 모델을 학습해야 했지만, 우리의 방법은 66.7개의 모델만을 학습하여 성능을 향상시킵니다.



### Constrained Multi-objective Bayesian Optimization through Optimistic Constraints Estimation (https://arxiv.org/abs/2411.03641)
- **What's New**: CMOBO라는 새로운 샘플 효율적인 제약 다목적 Bayesian 최적화 알고리즘을 제안합니다. 이 알고리즘은 여러 알려지지 않은 목표와 제약 조건에서 수준 집합을 학습하면서도 제약 영역 내에서의 다목적 최적화를 균형 있게 수행합니다.

- **Technical Details**: CMOBO는 무작위 스칼라화를 통해 기존의 제약 다목적 최적화 문제를 해결하며, 이론적으로 정당한 획득 규칙을 따릅니다. 또한 샘플 효율성과 비가용 선언 기능에 대한 이론적 분석을 제공합니다. 이 연구는 제약이 있는 다목적 최적화에서 최초로 저항을 제공하는 알고리즘으로 기록됩니다.

- **Performance Highlights**: 다양한 합성 기준점과 실제 응용 사례를 통해 제안된 방법의 효과성과 효율성을 확인했습니다. 기존의 알고리즘과 비교하여 성능이 우수함을 입증하였습니다.



### SEGMN: A Structure-Enhanced Graph Matching Network for Graph Similarity Learning (https://arxiv.org/abs/2411.03624)
- **What's New**: 본 논문에서는 그래프 유사도 계산(Graphic similarity computation, GSC)에 대한 새로운 접근법인 구조 강화 그래프 매칭 네트워크(Structure-Enhanced Graph Matching Network, SEGMN)를 제안합니다. SEGMN은 노드 간 연결 구조를 최대한 활용하여 더 정확한 유사도 점수를 산출합니다.

- **Technical Details**: SEGMN은 이중 임베딩 학습 모듈(Dual embedding learning module)과 구조 인식 매칭 모듈(Structure perception matching module)을 갖추고 있습니다. 이중 임베딩 학습 모듈은 각 노드에 인접한 엣지 표현을 통합하여 구조 향상된 표현을 생성합니다. 구조 인식 매칭 모듈은 할당 그래프(Assignment graph) 합성을 통해 교차 그래프의 구조를 강화합니다.

- **Performance Highlights**: 벤치마크 데이터 세트에 대한 실험 결과, SEGMN은 최신 GSC 방법보다 GED 회귀 작업에서 우수한 성능을 보였으며, 구조 인식 매칭 모듈은 기존 방법의 성능을 최대 25%까지 향상시킬 수 있음을 보여주었습니다.



### Temporal-Difference Learning Using Distributed Error Signals (https://arxiv.org/abs/2411.03604)
Comments:
          10 pages, to be published at NeurIPS 2024

- **What's New**: 이 논문에서는 뇌의 보상 기반 학습에서 중요한 역할을 하는 nucleus accumbens (NAc) 내 도파민의 신호 전달 메커니즘을 탐구합니다. 새로운 깊이 Q-학습 알고리즘인 Artificial Dopamine (AD)을 설계하여, 각 층에서 분산된 TD 오류만으로도 복잡한 강화 학습 (RL) 작업을 학습할 수 있음을 입증했습니다.

- **Technical Details**: AD는 각 층이 자신의 예측과 해당 오류를 계산하는 방식으로 작동하며, 이는 도파민의 지역적으로 균질한 분포를 반영합니다. 각 층의 업데이트는 서로 의존하지 않도록 설계되어 있으며, 오류 신호 없이 상위 층에서 하위 층으로 활성화 정보를 전송하는 전방 연결을 사용합니다. AD는 MinAtar, DeepMind Control Suite 및 고전 제어 작업에서 평가되었습니다.

- **Performance Highlights**: AD는 기존의 딥 RL 알고리즘인 DQN, SAC 및 TD-MPC2에 준하는 성능을 발휘하며, 오류 전파 없이도 여러 일반적인 RL 작업을 해결할 수 있음을 보여줍니다. 이 연구는 NAc의 도파민이 분산된 TD 오류 신호만으로도 보상 기반 학습을 지원할 수 있다는 주요 통찰력을 제공합니다.



### Open-Source High-Speed Flight Surrogate Modeling Framework (https://arxiv.org/abs/2411.03598)
- **What's New**: 이번 논문에서는 고속 비행체의 예측 능력을 개선하고 계산 비용을 줄이기 위한 새로운 서그릿 모델링 프레임워크를 제안합니다. 이 프레임워크는 다양한 신뢰도 수준에서 생성된 데이터(예: 공학적 방법, 시뮬레이션, 풍동 및 비행 시험 데이터를 통합하여 정확한 예측을 가능하게 합니다.

- **Technical Details**: 서그릿 모델은 고성능 컴퓨팅(HPC)에서 싱글 유저 머신(예: 노트북, 데스크탑 등)으로 계산의 대부분을 이동시키며, 개선된 하이퍼파라미터 튜닝 기능을 가지고 있습니다. Gaussian process regression와 deep neural network 기반 모델을 포함하여 두 개의 데이터셋을 R^2>0.99의 높은 정확도로 모델링합니다.

- **Performance Highlights**: 제안된 프레임워크는 실제 세계 프로젝트에 통합될 수 있도록 공군에 전달되었으며, 향후 연구를 위해 물리 법칙을 명시적으로 통합하는 모델링 방법의 추가 검사 및 개정이 권장됩니다.



### Enhancing the Expressivity of Temporal Graph Networks through Source-Target Identification (https://arxiv.org/abs/2411.03596)
Comments:
          Accepted to NeurIPS Symmetry and Geometry in Neural Representations Workshop 2024

- **What's New**: 최신 연구에서는 Temporal Graph Networks (TGN) 모델의 동적 노드 친화도 예측(dynamic node affinity prediction) 성능 향상을 위해, 메시지(message) 기반의 계산적 휴리스틱(heuristic)을 도입한 TGNv2를 제안합니다. 이는 기존의 TGN이 움직이는 평균(moving averages)을 제대로 나타낼 수 없다는 점에 착안한 것입니다.

- **Technical Details**: TGNv2는 각각의 상호작용 이벤트 메시지에 출발지와 도착지를 식별하는 정보를 추가하여, 기존 TGN의 표현력을 개선합니다. 이는 지속적인 예측(persistent forecasting) 및 이동 평균을 나타내는 데 필요한 변경 사항으로, TGNv2는 TGN 및 현재의 모든 TG 모델보다 월등히 우수한 성능을 보입니다.

- **Performance Highlights**: TGNv2는 Temporal Graph Benchmark (TGB) 동적 노드 친화도 예측 데이터셋에서 TGN 및 기존 TG 모델보다 모든 기준에서 유의미하게 성능이 개선되었습니다. 즉, TGNv2는 다양한 데이터셋에 대해 가장 신뢰할 수 있는 결과를 보여주고 있습니다.



### An Experimental Study on Decomposition-Based Deep Ensemble Learning for Traffic Flow Forecasting (https://arxiv.org/abs/2411.03588)
Comments:
          This work has been accepted by the 2024 Australasian Joint Conference on Artificial Intelligence (AJCAI 2024)

- **What's New**: 본 연구는 교통 흐름 예측을 위한 데이터의 분해 기반(Decomposition-based) 깊은 앙상블(deep ensemble) 학습 방법과 비분해 기반(non-decomposition-based) 방법의 성능을 비교합니다. 이 방법은 시간 시퀀스를 여러 개의 간단한 신호로 분해하여 모델링하는 과정을 포함하며, 이전 연구가 비교적 적었던 분야입니다.

- **Technical Details**: 이 연구에서 사용된 주요 기술적 접근 방식은 분해 기반 앙상블 방법과 전통적인 배깅(bagging) 및 시간 영역의 다중 해상도(multi-resolution) 앙상블 방법입니다. 특히, Empirical Mode Decomposition (EMD), Ensemble Empirical Mode Decomposition (EEMD), Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)과 같은 기법들이 효과적으로 활용됩니다. 실험은 교통 데이터셋 세 개를 사용하여 수행되었습니다.

- **Performance Highlights**: 실험 결과 분해 기반 앙상블 방법이 깊은 학습 모델의 성능을 향상시키며, 특히 집계(aggregation) 전략과 예측 범위(forecasting horizons)에 민감함을 나타냈습니다. 반면, non-decomposition 기반 방법은 복잡한 교통 흐름 패턴을 충분히 반영하지 못하는 경향을 보였습니다.



### Towards Personalized Federated Learning via Comprehensive Knowledge Distillation (https://arxiv.org/abs/2411.03569)
Comments:
          Accepted by IEEE SMC 2024

- **What's New**: 최근 개인화 연합 학습(Personalized Federated Learning, PFL) 접근 방식이 데이터 이질성 문제를 해결하기 위해 발전하였습니다. 본 논문에서는 모델의 개인화를 중시하면서도 일반화 성능을 유지할 수 있는 새로운 방법인 FedCKD를 제안합니다.

- **Technical Details**: FedCKD는 글로벌 모델(global model)과 역사적 모델(historical model)을 교육자로 사용하고, 로컬 모델(local model)을 학생으로 설정하여 지식 증류(knowledge distillation)를 수행합니다. 이 과정에서 글로벌 모델은 서버 집합의 마지막 라운드의 집합 모델을 나타내며, 역사적 모델은 클라이언트 훈련의 마지막 라운드에서의 로컬 모델을 나타냅니다. 이러한 구조를 통해 글로벌 일반화 지식 및 역사적 개인화 지식을 효과적으로 로컬 모델에 전달합니다.

- **Performance Highlights**: 실험 결과, FedCKD는 기존의 최첨단 방법들을 초월하여 모델의 성능을 유의미하게 향상시키는 것으로 나타났습니다. 특히, 모델의 일반화와 개인화 간의 균형을 유지하면서 치명적 망각(catastrophic forgetting) 문제를 완화하는 데 성공하였습니다.



### Large Language Models Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Lev (https://arxiv.org/abs/2411.03562)
- **What's New**: Agent K v1.0은 다양한 데이터 과학 작업을 자동화, 최적화 및 일반화하도록 설계된 종단 간(End-to-End) 자율 데이터 과학 에이전트입니다. 경험으로부터 학습하며 완전 자동화된 기능을 갖추고 있습니다.

- **Technical Details**: Agent K v1.0은 구조화된 추론(framework) 구조를 활용하여 고도로 유연한 방식으로 메모리를 동적으로 처리합니다. 여러 경험을 통해 복잡한 추론(tasks)을 수행하고, 장기 및 단기 기억을 최적화합니다. 환경 보상을 기반으로 한 미래 결정 유도, 세밀한 조정(fine-tuning) 없이도 지속적인 개선을 이룰 수 있는 반복적인 접근 방식을 사용합니다.

- **Performance Highlights**: Kaggle 대회에서 테스트한 결과, Agent K v1.0은 92.5%의 성공률을 기록하였으며, 탭형(tabular), 컴퓨터 비전(computer vision), 자연어 처리(NLP), 멀티모달(multimodal) 도메인에서의 성능을 입증했습니다. 5,856명의 인간 Kaggle 경쟁자와 비교할 때 상위 38%에 랭크되었으며, Kaggle Grandmaster 수준과 동등한 성능을 달성하여 6개의 금메달, 3개의 은메달, 7개의 동메달을 기록했습니다.



### Do Mice Grok? Glimpses of Hidden Progress During Overtraining in Sensory Cortex (https://arxiv.org/abs/2411.03541)
- **What's New**: 이번 연구는 행동 변화가 멈춘 이후에도 특정 작업에 대한 표현 학습이 지속될 수 있다는 가설을 제시합니다. 이를 위해 쥐의 후부 피리폼 피질(Posterior Piriform Cortex)에서의 신경 데이터를 재분석하였으며, 행동이 정체된 후에도 지속적인 학습을 발견했습니다.

- **Technical Details**: 신경 데이터 분석 결과, 피리폼 신경 집단의 디코딩 정확도(decoding accuracy)가 지속적으로 증가하며, 과훈련(overtraining) 동안 작업별 표현(class representations)이 지속적으로 분리되었습니다. 이를 통해, 초기의 잘못 분류된 예들이 시간이 지남에 따라 더 정확하게 분류될 수 있음을 보여주었습니다. 이러한 학습은 대략적인 마진 극대화(approximate margin maximization) 형태로 나타납니다.

- **Performance Highlights**: 실험 결과, 과훈련 기간이 길수록 쥐가 테스트 예제에서 더 나은 성과를 보였고, 이는 이론적 통찰력을 바탕으로 수치적으로 분석되었습니다. 또한, 이러한 학습은 동물 학습의 경험적 수수께끼인 과훈련 역전(overtraining reversal) 효과에 대한 설명을 제공할 수 있는 모델링으로 이어졌습니다.



### Long Context RAG Performance of Large Language Models (https://arxiv.org/abs/2411.03538)
Comments:
          2024 NeurIPS workshop on Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning

- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG)의 성능을 LLM의 컨텍스트 길이 증가에 따라 평가한 포괄적인 연구를 발표합니다. 특히, 2000부터 128,000 토큰까지의 컨텍스트 길이에서 RAG 작업을 수행하면서 최상위 LLM의 성능을 분석합니다.

- **Technical Details**: 연구에서는 총 20개의 인기 있는 오픈 소스 및 상업적 LLM을 사용하여 Databricks DocsQA, FinanceBench 및 Natural Questions라는 세 가지 도메인 특정 데이터세트를 대상으로 RAG 워크플로우를 실행했습니다. LLM의 컨텍스트 길이를 2000에서 128000 토큰, 경우에 따라 최대 200만 토큰까지 변화시키며 성능을 평가했습니다.

- **Performance Highlights**: 가장 최근의 최첨단 LLM 중 일부만이 64k 토큰을 초과하는 긴 컨텍스트에서 일관된 정확도를 유지할 수 있음을 발견했습니다. 전반적인 결과는 RAG 성능이 증가하는 컨텍스트 길이에 비례해 증가하지 않는다는 것을 보여주며, 일부 모델은 긴 컨텍스트 상황에서 고유한 실패 양상을 충족하기도 했습니다.



### Two-Stage Pretraining for Molecular Property Prediction in the Wild (https://arxiv.org/abs/2411.03537)
- **What's New**: MoleVers는 데이터가 부족한 환경에서도 효과적으로 작동하는 다목적의 사전 학습(pretrained) 모델로, 두 단계의 사전 학습 전략을 채택하여 분자 속성 예측을 개선합니다.

- **Technical Details**: MoleVers는 첫 번째 단계에서 masked atom prediction (MAP)과 dynamic denoising을 통해 큰 비표시 데이터셋에서 분자 표현을 학습합니다. 두 번째 단계에서는 낮은 비용의 계산 방법으로 얻은 보조 레이블을 사용하여 추가 사전 학습을 수행합니다. 이러한 이중 단계의 구조는 다양한 다운스트림 데이터셋에 효과적으로 일반화 가능한 표현을 학습하게 합니다.

- **Performance Highlights**: MoleVers는 22개의 다양한 분자 데이터셋을 평가한 결과, 20개에서 최첨단(state-of-the-art) 성능을 달성하고 나머지 2개에서도 두 번째로 높은 성과를 기록하며, 데이터가 부족한 현실 상황에서 유용한 주석이 부족한 모델의 한계를 극복하는데 성공하였습니다.



### PACE: Pacing Operator Learning to Accurate Optical Field Simulation for Complicated Photonic Devices (https://arxiv.org/abs/2411.03527)
Comments:
          Accepeted by Neurips 2024, 21 pages

- **What's New**: 이 연구에서는 복잡한 광소자의 시뮬레이션을 위한 새로운 PACE 연산자를 소개하며, 예전의 기술보다 예측 정확도를 대폭 향상시키고, 기존의 패러미터의 절반으로 73% 낮은 에러를 달성했습니다.

- **Technical Details**: 연구팀은 크게 두 가지 단계로 나누어 복잡한 시뮬레이션 작업을 수행했습니다. 첫 번째 단계 모델(PACE-I)은 초기 솔루션을 학습하고, 두 번째 단계 모델(PACE-II)은 이를 보완하여 최종 솔루션을 도출합니다. 또한, Cross-axis factorized PACE operator를 통해 광장과 장치 구조 간의 상관관계를 효과적으로 캡처하며, 이를 통해 시뮬레이션의 정확도를 향상시킵니다.

- **Performance Highlights**: PAC 모델은 기존의 여러 PDE 솔버에 비해 154-577배의 시뮬레이션 속도 향상을 보였고, 시간 소요 또한 크게 줄였습니다. 이 모델은 오픈소스로 제공되어 AI 및 PDE 연구 커뮤니티에서 활용될 수 있습니다.



### Understanding Contrastive Learning via Gaussian Mixture Models (https://arxiv.org/abs/2411.03517)
- **What's New**: 이번 논문에서는 Gaussian Mixture Models (GMM) 내에서 Contrastive Learning의 이론적 분석을 진행했습니다. 구체적으로 InfoNCE loss를 기반으로 하여 데이터 포인트의 증강을 독립적인 샘플로 정의하며, 이를 통해 Contrastive Learning이 최적의 저차원 서브스페이스를 찾을 수 있음을 보여주고 있습니다.

- **Technical Details**: Contrastive Learning은 unlabeled data로부터 representations를 학습하는 방법입니다. 본 연구에서는 Gaussian Mixture Models (GMM)에서 dimensionality reduction을 통해 Point-pairs의 증강을 정형화하는 새로운 방식을 제안합니다. 이를 통해, 변형된 InfoNCE가 비등방성 Gaussians에서도 최적의 선형 투영을 발견할 수 있음을 증명합니다. 또한, CLIP과 같은 다중 모달 Contrastive Learning 알고리즘에 대한 분석을 확장하여 noise를 필터링하는 최적의 하위 공간을 학습함을 보여줍니다.

- **Performance Highlights**: 본 논문은 GMM의 맥락에서 첫 번째로 Contrastive Learning을 분석한 연구로, 전통적인 방법으로는 실패했던 경우에서도 성공적인 결과를 도출하며, Zero-shot 성능과 다양한 작업에서의 상태 최첨단(SoTA) 성능을 달성할 수 있는 가능성을 제시합니다.



### LASER: Attention with Exponential Transformation (https://arxiv.org/abs/2411.03493)
Comments:
          15 pages, under review in ICLR 2025

- **What's New**: 이 논문은 기존의 attention 메커니즘에서 발생하는 미세한 기울기 전달 문제를 해결하기 위해 LASER라는 새로운 attention 방식을 제안합니다. LASER는 더 큰 기울기 신호를 허용하며, 기존의 attention 구현에 아주 작은 수정만으로 적용할 수 있습니다.

- **Technical Details**: LASER는 LogArithm of Summed Exponentials of Representations의 약어로, 입력의 지수 변환을 진행하여 attention을 수행하는 구조입니다. 본 방법은 Log-Weighted-Sum-Exp라는 새로운 기법을 도입하여 큰 모델(최대 22억 개의 파라미터를 가진 모델)에서도 수치적 오버플로우 문제를 해결합니다.

- **Performance Highlights**: LASER를 적용한 결과, Vision Transformer(이미지 분류의 경우)에서 4.67%의 정확도 향상, Conformer(음성 인식의 경우)에서 2.25%의 오류율 감소, BERT 모델에서 0.93%의 잘못된 예측 비율 감소를 달성했습니다. 여러 downstream 작업에서 LASER는 평균 1%의 정확도 향상을 보였고, 최대 3.38%의 정확도 향상을 기록했습니다.



### Pathway-Guided Optimization of Deep Generative Molecular Design Models for Cancer Therapy (https://arxiv.org/abs/2411.03460)
- **What's New**: 본 논문은 경로 모델(한계 방정식으로 설명되는)을 활용하여 JTVAE와 유사한 모델의 잠재 공간 최적화(latent space optimization, LSO)를 효과적으로 수행하는 방법을 제안합니다. 이는 새로운 의약 시사 대상을 제안할 수 있는 기회를 제공합니다.

- **Technical Details**: Junction tree variational autoencoder (JTVAE)를 사용하여 분자 구조를 직접 인코딩하고 디코딩함으로써, 고차원 구조 입력 데이터에서의 최적화 문제를 저차원 연속 공간으로 변환합니다. 이를 통해 성능을 높이고 고급 분자 특성을 달성하기 위한 새로운 접근 방식을 제안합니다. LSO 기술은 조건부 데이터 생성 및 고유 객체 기능 최적화를 위해 수행됩니다.

- **Performance Highlights**: 새로운 약물 유사 소분자 디자인을 위한 데이터 기반 모델의 효율성을 향상시키기 위해 제안된 접근법이 잠재 공간 최적화 기술과 결합하여 약물의 치료 효능을 예측하는 모델에서 개선된 결과를 도출했습니다.



### Fourier Analysis of Variational Quantum Circuits for Supervised Learning (https://arxiv.org/abs/2411.03450)
- **What's New**: 이번 논문에서는 변분 양자 회로(Variational Quantum Circuit, VQC)의 푸리에 계수(Fourier coefficient)가 변분 매개변수(variational parameters)에 의해 어떻게 결정되는지를 설명하는 알고리즘을 처음으로 제안합니다. 또한, 주어진 데이터셋의 푸리에 변환(Fourier transform)을 기존 스펙트럼과 비교하여 최적의 VQC를 예측할 수 있는 방법을 제시합니다.

- **Technical Details**: VQC는 입력 데이터를 인코딩하는 인코딩 게이트와 데이터를 최적화하는 변분 게이트, 그리고 회로에서 얽힘(Entanglement)을 생성하는 얽힘 게이트로 구성됩니다. 본 연구는 VQC의 스펙트럼을 정확하게 계산하는 알고리즘을 제안하며, 이를 통해 푸리에 계수의 조합적(combinatorial) 부분과 삼각 함수 다항식(trigonometric polynomials)의 합을 구별합니다. 새로운 알고리즘은 VQC의 변분 매개변수와 푸리에 계수 간의 함수적 관계를 추구합니다.

- **Performance Highlights**: 제안된 방법은 다양한 VQC 아키텍처를 사용하여 생성된 데이터셋에 대해 실험되었으며, 가장 중요한 주파수(주요 푸리에 계수)들을 추출하여 데이터에 가장 적합한 VQC 구조를 선택했습니다. 이 방법은 실제 세계 데이터와 합성 데이터 모두에서 테스트되었습니다.



### Solving Trojan Detection Competitions with Linear Weight Classification (https://arxiv.org/abs/2411.03445)
Comments:
          9 pages, 4 Figures

- **What's New**: 이 논문에서는 트로이안 백도어를 탐지하기 위한 새로운 방법을 제안했습니다. 이 접근법은 다양한 데이터셋과 도메인에서 높은 성능을 보입니다.

- **Technical Details**: 제안된 탐지기는 여러 개의 모델 가중치에 대한 이진 분류기를 학습하여 얻어지며, 주요 전처리 단계를 통해 성능을 개선합니다. 전 처리 단계에는 특성 선택, 표준화, 참조 모델 가중치 빼기, 모델 정렬 등이 포함됩니다. 이 기법은 가중치 분석(weight analysis) 탐지에 해당하며, 트리거에 대한 사전 지식 없이도 적용 가능합니다.

- **Performance Highlights**: 본 알고리즘은 Trojan Detection Challenge(TDC22)와 IARPA/NIST TrojAI 프로그램의 다양한 벤치마크에서 평가되었으며, 모델의 정밀한 분류를 통해 청정(clean) 모델과 오염된(poisoned) 모델 간의 구분을 효과적으로 수행했습니다.



### Quantifying Aleatoric Uncertainty of the Treatment Effect: A Novel Orthogonal Learner (https://arxiv.org/abs/2411.03387)
- **What's New**: 이번 연구에서는 관찰 데이터에서 인과적 수량(causal quantities)을 추정하는 방법에 대해 새로운 접근법을 제시합니다. 특히 치료 효과의 랜덤성을 이해하는 Aleatoric Uncertainty에 초점을 맞추고, 이를 통해 치료의 이점을 이해하는 데 필요한 확률 및 분위를 정량화하고자 합니다.

- **Technical Details**: 연구진은 조건부 치료 효과의 분포(Conditional Distribution of the Treatment Effect, CDTE)를 정의하고, 이에 대한 Aleatoric Uncertainty를 정량화하기 위해 부분 식별(partial identification) 기법을 사용하여 경계(support)를 도출합니다. 또한, 새로운 AU-learner라는 학습 알고리즘을 개발하여 CDTE의 경계를 추정합니다. 이 AU-learner는 Neyman-orthogonality를 만족하고 이중 강건성(doubly robust)을 지닌 특징이 있습니다.

- **Performance Highlights**: AU-learner의 성능은 기존 방법들과 비교하여 뛰어난 추정 품질과 안정성을 보여주며, 제안된 완전 매개변수형 심층 학습(deep learning) 구현이 AU-learner의 효과성을 증명합니다.



### Kernel Approximation using Analog In-Memory Computing (https://arxiv.org/abs/2411.03375)
- **What's New**: 본 연구는 혼합 신호 Analog In-Memory Computing (AIMC) 아키텍처에 적합한 커널 근사화(kernel approximation) 접근 방식을 소개합니다. 이 방법은 기존 커널 기반 방법의 성능 병목 현상을 해결하기 위해 거의 모든 연산을 메모리 내에서 직접 실행합니다.

- **Technical Details**: Analog In-Memory Kernel Approximation은 새로운 샘플과 훈련 데이터의 각 데이터 포인트 간의 커널 함수 평가를 효율적으로 처리하여 계산 효율성과 정확성을 동시에 유지합니다. 제안된 방법은 IBM HERMES Project Chip(최첨단 페이즈 변화 메모리 기반 AIMC 칩)을 사용하여 하드웨어 데모 우성하게 시행되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 커널 기반 릿지 분류 벤치마크에서 1% 미만의 정확도 하락을 유지하며, Transformer 신경망의 커널화된 어텐션에 대한 Long Range Arena 벤치마크에서 1% 이내의 정확도를 보였습니다. 기존 디지털 가속기 대비 에너지 효율이 뛰어나고 전력 소비가 낮아지는 것으로 추정됩니다.



### Energy Price Modelling: A Comparative Evaluation of four Generations of Forecasting Methods (https://arxiv.org/abs/2411.03372)
- **What's New**: 이 연구는 에너지 가격 예측을 위한 다양한 예측 방법론의 포괄적인 비교 분석을 제공하며, 특히 대규모 언어 모델(LLM)을 처음으로 에너지 분야의 시계열 예측에 적용하였습니다.

- **Technical Details**: 이 논문은 4개의 모델 가족으로 구성된 예측 방법론을 분류하고, 고전적 경제계량 모델, 머신 러닝 구조, LSTM과 같은 초기 시퀀스 모델, 최신 딥 러닝 기술인 transformer 네트워크 및 LLM을 포함한 여러 예측 기법을 조사합니다. 또한, pre-training과 transfer learning 같은 신흥 개념을 심도 있게 탐구합니다.

- **Performance Highlights**: EU 에너지 시장 데이터를 활용한 대규모 실증 연구를 통해 4개 모델 가족 간의 예측 정확도를 비교하고, 특히 시계열 transformer를 위한 대안 제안을 중점적으로 분석하여 최신 기술의 성과를 투명하게 조명합니다.



### Pedestrian Volume Prediction Using a Diffusion Convolutional Gated Recurrent Unit Mod (https://arxiv.org/abs/2411.03360)
- **What's New**: 메트로폴리탄 멜버른의 자동 보행자 카운팅 시스템에서 제공한 실시간 데이터 기반으로 한 보행자 흐름 예측 모델을 제안합니다. 새로운 모델인 DCGRU-DTW는 동적 시간 왜곡(Dynamic Time Warping)을 활용하여 시공간(spatio-temporal) 의존성을 효과적으로 캡처합니다.

- **Technical Details**: 제안된 DCGRU-DTW 모델은 Diffusion Convolutional Gated Recurrent Unit (DCGRU)에 동적 시간 왜곡(DTW)을 통합하여 보행자 흐름 데이터의 미세한 시공간 관계를 캡처합니다. 이 모델은 그래프 신경망(Graph Neural Network)의 근접성 측정을 개선하며, 각 교차로에 배치된 센서로부터 수집된 다변량 시계열(Multivariate Time Series) 데이터를 바탕으로 향후 보행자 수를 예측할 수 있습니다.

- **Performance Highlights**: 전통적인 벡터 자기회귀 모델(Variation Autoregressive Model) 및 기존 DCGRU 모델과 비교하여 제안된 DCGRU-DTW 모델은 여러 모델 정확성 지표에서 우수한 성능을 보여줍니다. 보행자 수 예측의 다양한 시나리오에서 높은 예측 정확도를 자랑합니다.



### SPINEX_ Symbolic Regression: Similarity-based Symbolic Regression with Explainable Neighbors Exploration (https://arxiv.org/abs/2411.03358)
- **What's New**: 이번 논문에서는 SPINEX(Similarity-based Predictions with Explainable Neighbors Exploration) 패밀리를 기반으로 한 새로운 기호 회귀 알고리즘(SPINEX_SymbolicRegression)을 소개합니다.

- **Technical Details**: 새로운 알고리즘은 유사성 기반 접근 방식을 사용하여 정확도 및 구조적 유사성 메트릭에 부합하는 고성능 표현을 식별합니다. 연구팀은 국제 문제 집합에서 180개 이상의 수학적 벤치마크 함수와 비교하여 SPINEX_SymbolicRegression의 성능을 평가했습니다.

- **Performance Highlights**: SPINEX_SymbolicRegression은 여러 번의 테스트 결과 항상 뛰어난 성능을 보였으며, 일부 경우에는 기존의 우수한 알고리즘보다 더 좋은 결과를 냈습니다. 또한, 알고리즘의 설명 가능성(explainability) 기능이 실험을 통해 강조되었습니다.



### Enhancing Table Representations with LLM-powered Synthetic Data Generation (https://arxiv.org/abs/2411.03356)
Comments:
          the Thirty-Eighth Annual Conference on Neural Information Processing Systems Table Representation Workshop

- **What's New**: 이 논문에서는 데이터 기반 의사결정 시대에 효율적인 테이블 추천 시스템을 위한 합성 데이터 생성 파이프라인을 제안합니다. 이 시스템은 Large Language Models (LLMs)을 활용하여 테이블 유사성을 정의하고, 테이블 표현 학습을 위한 고품질 합성 데이터를 생성합니다.

- **Technical Details**: 제안된 방법은 다음의 세 가지 접근 방식을 통해 검증되었습니다: (i) 인간 검증을 통한 합성 데이터셋의 정확성 확인, (ii) 기존 데이터셋과의 cosine similarity 비교를 통해 향상된 테이블 표현 가능성 입증, (iii) 합성 데이터를 활용한 유사 테이블 매칭 작업에서 최신 embedding 모델을 초월하는 성과.

- **Performance Highlights**: 실험 결과, 제안된 합성 데이터 생성 파이프라인은 테이블 유사성 정의에 부합하며, 추천 성능을 크게 향상시키는 고품질의 합성 데이터를 생성하여 실질적인 유사 테이블 추천 응용 프로그램에 기여할 수 있음을 보여줍니다.



### Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? (https://arxiv.org/abs/2411.04118)
Comments:
          Accepted to EMNLP 2024 Main Conference as Long Paper (Oral)

- **What's New**: 이 논문에서는 의학 응용을 위해 개발된 여러 모델들의 성능을 평가하고, 일반 도메인 모델들과의 비교를 통해 DAPT(Domain-Adaptive Pretraining)의 효과에 대한 의문을 제기합니다.

- **Technical Details**: 연구팀은 7개의 의학 LLM과 2개의 VLM을 선택하여, 각 의학 모델을 해당하는 일반 모델과 직접 비교했습니다. 각 모델에 대해 최적의 프롬프트를 독립적으로 선정하고, 통계적 불확실성을 고려한 후 성능을 평가했습니다.

- **Performance Highlights**: 의학 LLMs는 3-shot 설정에서 12.1%의 경우에만 베이스 모델을 초과하여, 나머지 87.9%의 경우에서는 동점이거나 그보다 성능이 떨어진다는 점을 발견했습니다. 이 결과는 DAPT가 의학 분야에서 항상 성능 향상을 보장하지 않음을 시사합니다.



### Self-Consistency Preference Optimization (https://arxiv.org/abs/2411.04109)
Comments:
          16 pages, 3 figures

- **What's New**: 이번 논문에서는 Self-consistency Preference Optimization (ScPO)라는 방법을 제안하여 모델이 비지도 학습 문제에 대해 일관된 답을 학습하도록 하는 새로운 접근을 소개합니다. 이 방법은 모델 훈련 과정에서 자가 일관성(self-consistency) 개념을 활용하여, 복잡한 문제 해결 작업의 성능을 향상시킵니다.

- **Technical Details**: ScPO는 비지도 학습 단계에서 모델이 생성한 문제와 쿼리를 사용하여 복잡한 문제 해결 작업을 수행합니다. 이 방법의 과정은 (i) 모델이 생성한 쿼리 선택, (ii) 가장 자가 일관성이 높은 응답(우승자)과 가장 낮은 응답(패자)에 대한 선호 쌍을 주석 달기, (iii) 모델의 선호 쌍에 대한 신뢰도에 따라 가중치가 조정된 손실 함수 최적화로 구성됩니다. 이 논문은 또한 라벨이 부여된 인스턴스와 미라벨 인스턴스에서 공동으로 LLM을 훈련하는 반지도 변형도 제안합니다.

- **Performance Highlights**: Llama-3 8B 모델을 사용한 실험 결과, ScPO는 GSM8K에서 22.74% 그리고 MATH에서 5.26%의 제로샷 정확도를 향상시키며, 감독 학습 방식의 성능에 가까운 결과를 도출했습니다. 또한 ScPO를 사용하여 ZebraLogic의 난해한 논리 퍼즐에서 6.5%의 정확도 향상을 보여주었으며, 이는 Llama-3 70B 및 Gemma-2 27B와 같은 더 큰 LLM들을 초월하는 성능입니다.



### A Comparative Study of Deep Reinforcement Learning for Crop Production Managemen (https://arxiv.org/abs/2411.04106)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 crop production management(작물 생산 관리)를 위한 머신 러닝 알고리즘인 reinforcement learning (RL)의 두 가지 접근 방식인 proximal policy optimization (PPO)와 deep Q-network (DQN)의 성능을 비교했습니다. 이 연구는 gym-DSSAT 환경에서 세 가지 RL 작업인 비료 주기, 관개 및 혼합 관리에 대해 두 알고리즘을 평가하여 더 효과적인 RL 기반 작물 관리 전략 개발을 지원합니다.

- **Technical Details**: 본 연구의 평가에서는 PPO와 DQN 알고리즘이 static baseline policies(정적 기준 정책)와 비교되었습니다. 이 과정에서 동일한 기본 매개변수(default parameters), 보상 함수(reward functions), 환경 설정을 사용하여 공정한 비교를 보장했습니다. PPO는 비료와 관개 작업에서 DQN을 초월한 성능을 보였고, DQN은 혼합 관리 작업에서 더 우수한 결과를 나타냈습니다. 이 연구는 RL 알고리즘 간의 강점과 한계를 이해하는 중요한 통찰력을 제공합니다.

- **Performance Highlights**: 비료 주기와 관개 작업에서는 PPO가 DQN보다 우수한 성능을 보였고, 혼합 관리 작업에서는 DQN이 더 뛰어난 성과를 거두었습니다. 이러한 결과는 RL 알고리즘 선택 시 유용한 정보로 작용할 것으로 기대됩니다.



### Problem Space Transformations for Generalisation in Behavioural Cloning (https://arxiv.org/abs/2411.04056)
- **What's New**: 행동 복제(behavioural cloning)와 신경망(neural networks)의 결합이 로봇 조작(robust manipulation)에서 중요한 진전을 이뤘지만, 복잡한 시나리오에서는 여전히 비효율적이라는 문제를 다룹니다. 이 연구는 로봇 조작의 광범위한 특성을 분석하고, 이러한 특성이 포함된 문제 공간 변환(problem space transformations)을 통해 학습된 정책의 OOD(Out-of-Distribution) 일반화 성능을 향상시킬 수 있음을 실험적으로 증명합니다.

- **Technical Details**: 이 연구에서는 로봇 조작 과제를 다루며, 유한 시간 마르코프 결정 과정(finite-horizon Markov Decision Process, MDP) 모델을 사용하여 데이터 수집을 가정합니다. 행동 복제는 주어진 작업을 위한 데이터를 수집하고, 이를 사용하여 특정 강화학습 정책을 베이스로 한 정책 π θ 를 학습하여 실제 환경에서의 조작 성능을 향상시킵니다. 연구의 주요 기여는 실용적인 조작 문제의 특성을 규명하고, 이러한 특성을 내재화한 여러 문제 공간 변환을 제안하며, 선택된 문제 공간 변환이 OOD 일반화 능력에 미치는 영향을 실험적으로 입증합니다.

- **Performance Highlights**: 이 연구에서 제안된 문제 공간 변환은 OOD 상태에서 잘 작동하도록 정책의 일반화 능력을 개선하여, 로봇 조작 작업에서의 성능을 향상시킵니다. 세 가지 로봇 조작 작업에 대한 실험 결과를 통해 이러한 효과를 입증하였으며, 특정 문제 공간 변환의 선택이 OOD 일반화 성능에 중대한 영향을 미친다는 사실을 보여주었습니다.



### Partial Structure Discovery is Sufficient for No-regret Learning in Causal Bandits (https://arxiv.org/abs/2411.04054)
Comments:
          To appear in Proceedings of NeurIPS 24

- **What's New**: 본 연구는 알려지지 않은 인과 그래프와 잠재적 혼란 변수를 포함하는 인과 밴딧 문제를 다룹니다. 연구자들은 목표 변수를 위한 최적의 개입을 찾는 보다 효율적인 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서 제안되는 알고리즘은 인과 그래프의 각 부모를 넘어서는 인과 발견(causal discovery) 방법론을 사용하고, 잠재적 혼란 변수( latent confounders)에 대한 필요하고 충분한 집합을 공식적으로 특성화합니다. 두 단계 접근 방식으로, 첫 번째 단계에서는 보상 노드의 조상에 대한 induced subgraph 를 학습하고, 두 번째 단계에서는 일반적인 밴딧 알고리즘인 UCB(Upper Confidence Bound) 알고리즘을 적용합니다.

- **Performance Highlights**: 제안된 알고리즘은 기존의 인과 그래프 학습 방법론에 비해 샘플 효율성을 높이며, 이 과정에서 발생하는 후회(regret) 경계도 서브리니어(sublinear)하게 확립됩니다. 이는 인과 그래프의 인과 구조를 완전히 파악할 필요 없이 가능한 최적의 개입 세트를 식별할 수 있도록 합니다.



### Predicting and Publishing Accurate Imbalance Prices Using Monte Carlo Tree Search (https://arxiv.org/abs/2411.04011)
- **What's New**: 이번 논문에서는 재생 가능 에너지(RE) 공급의 변동성에 따른 전력망(전력 시스템) 불균형 문제를 해결하기 위해 신규 방법론인 Monte Carlo Tree Search (MCTS)를 제안했습니다. 이 방법은 시스템의 역학을 신경망 예측기(neural network forecaster)와 강화 학습(reinforcement learning) 에이전트가 제어하는 가상 배터리(fe) 클러스터를 사용하여 모델링합니다.

- **Technical Details**: 제안된 MCTS 방법은 가격 예측을 위해 시스템 동역학의 모델을 기반으로 하며, 1분 단위의 실시간 불균형 가격을 정확하게 예측하고 게시하는 것을 목표로 합니다. 이 기술은 각각의 예측 후 실시간 가격 발표를 위한 낮은 계산 시간을 필요로 하며 산업 복잡성을 조절하는 유연성을 제공합니다.

- **Performance Highlights**: 제안된 기술은 벨기에의 현재 게시 방법에 비해 이상적인 조건에서 가격 정확성을 20.4% 향상시켰으며, 보다 현실적인 조건에서는 12.8%의 향상을 보였습니다. 이 연구는 이전에 시도되지 않았던 진보된 불균형 가격 게시 기법을 분석하는 선구적인 작업으로 자리매김합니다.



### ET-SEED: Efficient Trajectory-Level SE(3) Equivariant Diffusion Policy (https://arxiv.org/abs/2411.03990)
Comments:
          Accept to CoRL 2024 Workshop on X-Embodiment Robot Learning

- **What's New**: 이 논문에서는 로봇 조작 작업을 위한 효율적인 경로 수준 (trajectory-level) SE(3) 등가성 (equivariance) 확산 모델인 ET-SEED를 제안하고 있습니다. 이 모델은 샘플 효율성을 개선하고 훈련 난이도를 감소시키며, 복잡한 조작 작업에서의 일반화 능력을 향상시킵니다.

- **Technical Details**: ET-SEED는 SE(3) 군 내에서 정의된 등가 확산 정책으로, 조작 경로를 생성하는 과정에서 적어도 하나의 등가 전이(Eqvariant Transition)만으로도 충분하다는 이론적 확장을 통해 훈련 효율성을 개선했습니다. 또한, SE(3) 다양체에서의 확산 과정을 통합하여 설계되었습니다.

- **Performance Highlights**: 실험 결과, ET-SEED는 데이터 효율성, 조작 능숙도, 공간 일반화 능력에서 최첨단 (SOTA) 방법들을 초월하는 성과를 보여주었으며, 단지 20개의 시연 경로로도 보지 못한 시나리오까지 일반화할 수 있음을 입증했습니다.



### Bayesian algorithmic perfumery: A Hierarchical Relevance Vector Machine for the Estimation of Personalized Fragrance Preferences based on Three Sensory Layers and Jungian Personality Archetypes (https://arxiv.org/abs/2411.03965)
Comments:
          15 pages, 0 figures

- **What's New**: 본 연구는 계층적 Relevance Vector Machines(RVM)와 융 심리학적(personality archetypes) 성격 유형을 통합하여 개인화된 향수 추천을 위한 베이지안 알고리즘적 접근 방식을 탐구합니다.

- **Technical Details**: 이 모델은 상단(top), 중간(middle), 기저(base) 노트에 대한 개인의 향 선호를 융 심리학의 성격 특성에 연결합니다. 베이지안 업데이트(Updating)를 통해 사용자와 향의 상호작용에 따라 예측을 동적으로 반복적으로 개선할 수 있습니다.

- **Performance Highlights**: 이 연구는 심리학 이론과 베이지안 기계 학습의 결합이 개인의 선호도를 모델링하는 복잡성을 해결하고 사용자의 특성과 집단 수준의 추세를 파악하는 데 잠재력을 가지고 있음을 강조합니다.



### Improved Regret of Linear Ensemble Sampling (https://arxiv.org/abs/2411.03932)
- **What's New**: 이 연구에서는 선형 앙상블 샘플링 (linear ensemble sampling)에 대한 개선된 regret 경계 (regret bound)를 제공하여 이론과 실제의 근본적인 간극을 해소했습니다. 앙상블 크기가 T에 대해 로그적일 때, 선형 앙상블 샘플링은 $	ilde{	ext{O}}(d^{3/2}	ext{√}T)$의 빈도론적 (frequentist) regret 경계를 달성할 수 있음을 입증했습니다.

- **Technical Details**: 이 접근법은 선형 밴딧 알고리즘에 대한 일반적인 regret 분석 프레임워크를 도입합니다. 논문의 중요한 발견 중 하나는 선형 앙상블 샘플링과 선형 섭동 이력 탐색 (Linear Perturbed-History Exploration, LinPHE) 간의 관계를 밝힌 것입니다. 즉, 앙상블 크기가 T일 때 LinPHE는 선형 앙상블 샘플링의 특별한 경우입니다. 이를 통해 LinPHE에 대한 새로운 regret 경계를 유도할 수 있습니다.

- **Performance Highlights**: 이 연구의 기여는 앙상블 샘플링의 이론적 기반을 발전시키며, 그 regret 경계를 다른 무작위 탐색 알고리즘에 대한 가장 잘 알려진 경계와 일치시킵니다.



### A Causal Framework for Precision Rehabilitation (https://arxiv.org/abs/2411.03919)
Comments:
          keywords: rehabilitation; precision rehabilitation; causal inference; international classification of functioning; rehabilitation treatment specification system; computational neurorehabilitation

- **What's New**: 정밀 재활(Precision rehabilitation)의 새로운 접근법이 제시되었습니다. 개인 맞춤형 재활을 최적화하여 장기적인 기능적 결과를 개선할 수 있는 증거 기반 접근법을 제공하는 것이 목표입니다.

- **Technical Details**: 최적의 동적 치료 요법(Optimal Dynamic Treatment Regimens, ODTR)을 식별하기 위한 프레임워크를 제안하며, 이는 다양한 측정값과 바이오마커를 활용한 의사 결정을 포함합니다. 인과 모델(causal models)을 설계하고 적합시키며, 이는 인과 추론(causal inference) 도구를 사용하여 계산 신경 재활(Computational Neurorehabilitation) 프레임워크를 확장하는 방법으로 이루어집니다. 이러한 모델은 다양한 데이터 출처(heterogeneous data)에서 학습할 수 있으며, 재활 치료 사양 시스템(Rehabilitation Treatment Specification System)을 통해 개입에 대한 세부 문서를 포함해야 합니다.

- **Performance Highlights**: 이 프레임워크는 환자의 회복 경로를 디지털 트윈(digital twins)으로 나타내며, ODTR을 학습할 수 있게 합니다. 또한, 기능의 수준 변화(linking changes across levels of functioning)를 정량적으로 연결하여 개입을 정확히 선택할 수 있도록 하며, 이는 환자 및 이해관계자에게 의미 있는 결과를 극대화하기 위해 선택됩니다.



### Game-Theoretic Machine Unlearning: Mitigating Extra Privacy Leakag (https://arxiv.org/abs/2411.03914)
- **What's New**: 최근의 GDPR 및 CCPA와 같은 법률로 인해 사용자는 개인 데이터를 모델에서 삭제할 권리가 생겼습니다. 이를 위해 최근 주목받고 있는 기계 비학습(machine unlearning) 기술이 등장했습니다. 본 논문에서는 게임 이론(game theory)을 활용한 새로운 기계 비학습 알고리즘을 제안하며, 데이터 삭제의 효율성과 개인 정보 보호 간의 균형을 찾고자 합니다.

- **Technical Details**: 본 알고리즘은 비학습(unlearning) 모듈과 개인 정보 보호(privacy) 모듈로 구성되어 있습니다. 비학습 모듈은 모델 거리(model distance)와 분류 오류(classification error)로 구성된 손실 함수(loss function)를 사용하여 최적의 전략을 도출하고, 개인 정보 보호 모듈은 비학습 데이터에서 멤버십 정보(membership information)를 유출하기 어렵도록 설계되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 게임 이론 기반 비학습 알고리즘은 실제 데이터셋에서 비학습 모델의 성능을 유지하면서도 추가적인 개인 정보 유출 위험을 줄이는 것으로 나타났습니다.



### Polynomial Composition Activations: Unleashing the Dynamics of Large Language Models (https://arxiv.org/abs/2411.03884)
- **What's New**: 본 논문에서는 transformers 아키텍처를 최적화하기 위해 폴리곤 조합 활성화 함수(PolyCom)를 제안합니다. 이 활성화 함수는 전통적인 비선형 활성화 기능의 한계를 극복하고, 데이터 내의 더 복잡한 패턴을 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: PolyCom 활성화 함수는 다항식(degree) 조합을 활용하여 더 높은 차원의 상호작용을 캡처할 수 있게 해줍니다. 특히, PolyReLU와 PolyNorm의 두 가지 인스턴스를 도입하며, 이들은 transformer 아키텍처의 통합 과정에서 다항식의 힘을 활용합니다. 폴리곤 조합 활성화 함수는 Sobolev 공간에서 일반적인 매끄러운 함수의 최적 근사 속도를 달성하는 것을 보여줍니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)에서 PolyCom을 적용한 결과, SwiGLU, GELU 및 ReLU 등 기존의 활성화 함수보다 성능이 유의미하게 향상되었으며, 수렴 속도도 크게 빨라짐을 확인했습니다. 이는 PolyCom이 딥러닝 애플리케이션에서 실질적인 이점을 제공함을 시사합니다.



### MEG: Medical Knowledge-Augmented Large Language Models for Question Answering (https://arxiv.org/abs/2411.03883)
- **What's New**: MEG라는 새로운 파라미터 효율적인 접근 방식을 제안합니다. 이는 의료 관련 LLM(대규모 언어 모델)에 그래프 임베딩(knowledge graph embeddings)을 통합하여 외부 지식을 비용 효율적으로 활용할 수 있도록 합니다.

- **Technical Details**: MEG는 경량화된 매핑 네트워크를 활용하여 KG 임베딩을 LLM의 벡터 공간으로 단방향으로 변환합니다. Mistral-Instruct(7B)를 기본 LLM으로 사용하였으며, KG 인코더는 GraphSAGE를 기반으로 하여 간단한 MLP(다층 퍼셉트론) 매핑 네트워크와 결합되었습니다. RAG(회수 증강 생성) 방식과 유사한 접근방식입니다.

- **Performance Highlights**: MEG는 Mistral-Instruct 기준선에 비해 평균 10.2% 더 높은 정확도를 달성하였고, BioMistral과 같은 전용 모델에 비해 6.7% 더 우수한 성능을 보였습니다. 4개의 인기 있는 의료 다중 선택 데이터셋에서 검증하였으며, KG 임베딩을 통합한 LLM이 의료 질문 답변에 있어 효과적임을 입증하였습니다.



### Large Generative Model-assisted Talking-face Semantic Communication System (https://arxiv.org/abs/2411.03876)
- **What's New**: 이 연구는 대형 생성 모델을 활용한 새로운 대화 얼굴 의미 통신 시스템(LGM-TSC)을 도입하며, 이는 저대역폭 환경에서도 고품질의 대화 얼굴 비디오 전송을 가능하게 합니다.

- **Technical Details**: LGM-TSC 시스템은 크게 Generative Semantic Extractor (GSE), Large Language Model (LLM) 기반의 개인 Knowledge Base (KB), 그리고 Generative Semantic Reconstructor (GSR)로 구성됩니다. GSE는 비디오를 텍스트로 변환하여 대역폭을 효율적으로 사용하도록 돕고, LLM 기반의 KB는 의미 모호성을 줄이며 텍스트 교정을 지원합니다. GSR은 텍스트를 고품질 대화 얼굴 비디오로 복원합니다.

- **Performance Highlights**: 모의 결과에 따르면 LGM-TSC 시스템은 기존 방법들에 비해 낮은 대역폭에서도 높은 QoE(Quality of Experience)를 유지하며 의미 통신의 효율성을 개선하는 데 성공하였습니다.



### AdaSociety: An Adaptive Environment with Social Structures for Multi-Agent Decision-Making (https://arxiv.org/abs/2411.03865)
Comments:
          Accepted at NeurIPS D&B 2024

- **What's New**: 본 논문에서는 기존의 다중 에이전트 환경의 한계를 극복하기 위해 AdaSociety라는 새로운 다중 에이전트 환경을 소개합니다. 이 환경은 에이전트의 행동에 기반하여 새로운 작업을 동적으로 생성하고, 사회적 연결을 통해 보상을 조정하는 특징을 가지고 있습니다.

- **Technical Details**: AdaSociety는 확장 가능한 상태(state)와 행동(action) 공간을 가지고 있으며, 명시적이고 조정 가능한 사회 구조를 포함합니다. 이 환경에서는 에이전트들이 새로운 작업에 도전하며, 세 가지 미니 게임을 통해 다양한 사회 구조를 경험하게 됩니다. 또한, 사회 연결은 멀티 레이어 방향 그래프로 표현되어 에이전트 간의 관계를 정량적으로 설명합니다.

- **Performance Highlights**: 초기 결과에 따르면, 특정 사회 구조가 개인 및 집단의 이익을 동시에 촉진할 수 있지만, 현재 강화학습( RL) 및 LLM 기반 알고리즘은 사회 구조를 효과적으로 활용하여 성능을 향상시키는 데 제한적인 효과를 보였습니다. AdaSociety는 다양한 물리적 및 사회적 환경에서의 지능을 탐구하기 위한 귀중한 연구 플랫폼으로 자리매김할 수 있습니다.



### UniTraj: Universal Human Trajectory Modeling from Billion-Scale Worldwide Traces (https://arxiv.org/abs/2411.03859)
- **What's New**: 본 연구는 다양한 작업과 지역에 걸쳐 일반화 및 확장이 가능한 범용 인간 궤적 기반 모델, UniTraj를 제안합니다. 또한, 세계적으로 분산된 첫 번째 대규모 고품질 궤적 데이터셋인 WorldTrace를 구축하여, 70개국에서 2.45백만 개 궤적과 수십억 개의 포인트를 포함하고 있습니다.

- **Technical Details**: UniTraj는 task-adaptive(작업 적응형), region-independent(지역 비의존성) 및 데이터 품질에 높은 내성을 갖춘 모델입니다. 이 모델은 다양한 궤적 분석 작업을 효율적으로 지원하고, 여러 재샘플링 및 마스킹 전략을 활용하여 복잡한 시공간 의존성을 포착합니다.

- **Performance Highlights**: 다양한 궤적 분석 작업과 실제 데이터셋을 통해 UniTraj는 기존 접근 방식에 비해 일관되게 우수한 성능을 보였습니다. 특히, WorldTrace 데이터셋은 파트너십 및 협력을 통한 손쉬운 접근성을 제공함으로써 범용 궤적 모델 개발을 촉진할 수 있는 잠재력을 지니고 있습니다.



### Efficient Message Passing Architecture for GCN Training on HBM-based FPGAs with Orthogonal Topology On-Chip Networks (https://arxiv.org/abs/2411.03857)
Comments:
          This paper has been accepted for 2024 ACM/SIGDA International Symposium on Field Programmable Gate Arrays(FPGA'24) as poster

- **What's New**: 이 논문에서는 그래프에서 메시지 전달을 위한 새로운 아키텍처를 제안합니다. NUMA 기반의 메모리 접근 속성을 활용하고, 4차 하이퍼큐브 네트워크를 기반으로 한 병렬 멀티캐스트 라우팅 알고리즘을 적용하여, GCN의 훈련 효율을 높입니다.

- **Technical Details**: 제안된 아키텍처는 GCN의 역전파(backpropagation) 알고리즘을 재설계하여, 훈련 중 메모리 요구 사항을 줄이고 대량의 행렬 전치(transposition)로 인한 계산 오버헤드를 감소시킵니다. 이로 인해, 훈련이 더 효율적으로 이루어질 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 Xilinx UltraScale+ VCU128에서 HP-GNN 아키텍처와 비교하여 1.03×에서 1.81×까지 성능 향상을 보였습니다.



### MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba (https://arxiv.org/abs/2411.03855)
- **What's New**: 본 논문에서는 Mamba라는 새로운 State Space Model (SSM) 기반 모델에 대한 Parameter-efficient fine-tuning (PEFT) 방법들의 탐색적 분석을 진행하였습니다. 기존의 Transformer 기반 모델들에 비해 Mamba 모델에서의 PEFT의 효율성과 효과를 강조하며, 신규 Mamba 특화 PEFT 방법들을 제안하였습니다.

- **Technical Details**: Mamba는 시간을 선형적으로 처리할 수 있는 모델로, 기존 Transformer의 계산 복잡성을 극복하며 긴 시퀀스를 효율적으로 처리합니다. 본 연구에서는 Mamba 아키텍처에 적합하도록 기존 PEFT 방법들을 수정하고, Mamba에 특화된 새로운 PEFT 방법들을 제안하여 성능을 극대화하는 방법을 제안합니다. 실험에서는 7개의 주요 PEFT 방법과 총 20개의 파생 변형을 벤치마킹하였습니다.

- **Performance Highlights**: Mamba에서의 PEFT는 Transformer보다 더 효과적이며, 여러 PEFT 방법들을 조합하여 성능을 향상시킬 수 있음을 보여주었습니다. 본 논문은 PEFT 방법의 조합을 효율적으로 탐색하는 기술을 제안하고, 단순한 높은 성능의 방법 조합만으로는 충분하지 않다는 것을 밝혔습니다.



### Beyond The Rainbow: High Performance Deep Reinforcement Learning On A Desktop PC (https://arxiv.org/abs/2411.03820)
Comments:
          9 main pages, 26 total. Currently under review at ICLR

- **What's New**: 이 논문에서는 Rainbow DQN의 많은 독립적인 향상을 통합하여 BTR(Beyond The Rainbow)이라는 새로운 알고리즘을 제시합니다. 이 알고리즘은 컴퓨팅 리소스가 제한된 환경에서도 뛰어난 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: BTR은 Rainbow DQN에 비해 6개의 알고리즘 개선 사항을 결합하여 새로운 최첨단 성능을 달성하였습니다. 이 개선 사항들은 에이전트의 훈련에 필요한 계산 비용을 줄이기 위한 것입니다. BTR은 데스크톱 PC에서 200백만 Atari 프레임을 12시간 이내에 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: Atari-60에서 BTR은 인간 정규화된 Interquartile Mean(IQM) 7.4를 기록하며 새로운 기록을 세웠습니다. 또한, BTR을 사용해 Super Mario Galaxy, Mario Kart 등 복잡한 3D 게임을 성공적으로 훈련시킬 수 있음을 보여주었습니다.



### On the Decomposition of Differential Gam (https://arxiv.org/abs/2411.03802)
- **What's New**: 이 논문은 Helmholtz의 정리와 Hodge의 정리를 연결하여 차별 게임(differential games)의 동적-전략적 단절 문제를 해결하는 두 가지 분해 방법을 제시합니다. 이는 기존의 게임 이론에 대한 이해를 확장하는 중요한 작업입니다.

- **Technical Details**: 저자들은 차별 게임을 정확한 스칼라 잠재적 부분(exact scalar potential part), 근사 벡터 잠재적 부분(near vector potential part), 비전략적 부분(non-strategic part)으로 분해하는 두 가지 방법을 제안합니다. 이 중 스칼라 잠재 게임(scalar potential games)은 개별 전략 변화가 집합적 효용에 미치는 영향을 정량화하는 잠재 함수(potential function)가 존재합니다.

- **Performance Highlights**: 스칼라 잠재 게임은 gradient descent 방법을 통해 Nash 균형(Nash equilibrium)에 성공적으로 도달할 수 있으며, 벡터 잠재 게임의 경우에는 개별 그래디언트 필드가 발산이 없는(divergence-free) 결과를 보입니다. 출발 조건에 따라 gradient descent 동적이 발산하거나 주기적일 수 있다는 점이 중요합니다.



### Navigating the landscape of multimodal AI in medicine: a scoping review on technical challenges and clinical applications (https://arxiv.org/abs/2411.03782)
Comments:
          28 pages

- **What's New**: 최근 헬스케어에서의 기술 발전으로 인해 환자 데이터의 양과 다양성이 폭발적으로 증가하고 있으며, 이는 multimodal AI의 필요성을 더욱 부각시키고 있습니다. 본 논문은 2018년에서 2024년 사이에 발표된 432개의 논문을 분석하여, 다양한 의료 분야에서의 multimodal AI의 발전 현황을 종합적으로 검토합니다.

- **Technical Details**: 논문에서는 deep learning 기반의 multimodal AI 적용 사례를 다루며, 다양한 아키텍처 접근 방식(architectural approaches)과 데이터 융합 전략(fusion strategies)을 분석합니다. 또한 교차부서간 협력과 이질적인 데이터 특성(homogeneous data characteristics)과 같은 기술적 및 실질적 과제들을 제시하고, 상업적으로 이용 가능한 다양한 multimodal AI 모델들을 소개합니다.

- **Performance Highlights**: 연구 결과에 따르면 multimodal AI 모델은 unimodal 모델에 비해 평균 6.2%의 AUC 개선 효과를 보였으며, 이는 임상 의사결정에 있어 멀티모달 데이터 통합의 장점을 보여줍니다. 다만, 데이터의 불완전성, 다양한 의료 분야 간의 협력 부족과 같은 도전 과제가 여전히 남아있습니다.



### No Culture Left Behind: ArtELingo-28, a Benchmark of WikiArt with Captions in 28 Languages (https://arxiv.org/abs/2411.03769)
Comments:
          9 pages, Accepted at EMNLP 24, for more details see this http URL

- **What's New**: 이번 연구에서는 다양한 언어와 문화에 대한 주관적 감정을 반영한 새로운 비전-언어 벤치마크인 ArtELingo-28을 소개합니다. 이 벤치마크는 28개 언어로 약 200,000개의 주석을 포함하고 있습니다.

- **Technical Details**: ArtELingo-28은 감정적 설명을 생성하는 머신 러닝 시스템을 구축하는 데 중점을 두며, Zero-Shot, Few-Shot 및 One-vs-All Zero-Shot의 세 가지 새로운 평가 조건에 대해 기본 성능 결과를 발표합니다. 또한, 이 연구는 문화적으로 관련된 언어 간의 전이 성능이 더 성공적임을 발견했습니다.

- **Performance Highlights**: 연구 결과, ArtELingo-28은 2000개의 이미지에 대해 200K 감정 레이블과 감정적 텍스트 설명을 수집했으며, 다양한 문화적 시각을 반영한 새로운 데이터 세트를 성공적으로 구축하였습니다. 또한, 다국어 설정에서 모델의 성능을 평가하는 데 기여했습니다.



### Variational Inference on the Boolean Hypercube with the Quantum Entropy (https://arxiv.org/abs/2411.03759)
- **What's New**: 본 논문에서는 부울 하이퍼큐브에서 쌍-마르코프 랜덤 필드(pairwise Markov random fields)의 로그-파트션(log-partition) 함수에 대한 변별적 추론(variational inference) 상한을 도출했습니다. 이 과정에서 쿼텀 완화(quantum relaxation)를 기반으로 한 쿨백-라이블러 발산(Kullback-Leibler divergence)을 사용하였습니다.

- **Technical Details**: 논문에서는 기본-쌍대 최적화(primal-dual optimization)에 기반하여 이러한 상한을 효율적으로 계산하는 알고리즘을 제안합니다. 또한, '계층(hierarchies)'을 사용하여 이러한 상한을 개선하는 방법을 제안하며, 이는 제곱의 합(sum-of-squares, SoS) 계층과 유사합니다. greedy algorithm을 통해 이러한 완화들 중에서 선택하는 방법도 소개합니다.

- **Performance Highlights**: 다양한 수치 실험을 수행하였고, 이 추론 문제에 대한 최첨단(state-of-the-art) 방법들과 비교하였습니다.



### Energy-based physics-informed neural network for frictionless contact problems under large deformation (https://arxiv.org/abs/2411.03671)
Comments:
          22 pages, 9 figures

- **What's New**: 본 논문에서는 다양한 조건에서 마찰 없는 접촉 문제를 해결하기 위한 에너지 기반 물리 정보 신경망(physics-informed neural networks, PINNs) 프레임워크를 제안합니다. 이 프레임워크는 미세한 Lennard-Jones 포텐셜에서 영감을 받아 접촉 현상을 설명하기 위해 접촉 에너지를 사용하며, 강인성을 보장하기 위해 이완(relaxation), 점진적 하중(gradual loading), 출력 스케일링(output scaling) 기술을 도입합니다.

- **Technical Details**: PINN은 비선형 고체 역학 문제를 해결하기 위해 신경망을 사용하여 허용 가능한 변위 솔루션을 근사합니다. 이 프레임워크에서 제안된 신경망은 에너지 형태 손실에 의해 조정되어 더욱 견고한 결과를 제공합니다. 논문에 소개된 다양한 수치적 예제들은 Hertz 접촉 벤치마크 문제와 재료 및 기하학적 비선형성 등을 포함하여 에너지 기반 PINN의 유효성을 입증합니다.

- **Performance Highlights**: 제안된 PINN 프레임워크는 상업적인 유한 요소 분석(Finite Element Method, FEM) 소프트웨어와 비교했을 때 복잡한 접촉 문제를 처리하는 데 있어 경쟁력 있는 계산 효율성을 보여줍니다. 해당 프레임워크는 실험 데이터를 쉽게 통합할 수 있는 장점도 가지고 있습니다.



### Requirements Engineering for Older Adult Digital Health Software: A Systematic Literature Review (https://arxiv.org/abs/2411.03656)
Comments:
          arxiv version of SLR on RE for Older Adult Digital Health Software

- **What's New**: 노인 인구의 증가로 인해 기술 지원 노인 돌봄에 대한 관심이 커지고 있습니다. 그러나 돌봄 제공자 부족, 노인의 정서적, 사회적, 신체적, 정신적 요구에 대한 이해의 한계 등의 도전 과제가 존재합니다.

- **Technical Details**: 본 연구는 디지털 건강 소프트웨어에서 노인을 위한 요구 사항 공학(Requirements Engineering, RE) 관련 문헌을 체계적으로 리뷰하였습니다. Kitchenham 방법, PRISMA, PICO 가이드라인을 사용하여 프로토콜을 개발하고 8개의 데이터베이스를 체계적으로 탐색하여 69개의 높은 관련성의 주요 연구를 도출했습니다.

- **Performance Highlights**: 요구 사항 수집 및 이해에서의 변동성이 큰 것으로 나타났으며, 이는 기존 연구의 품질, 깊이 및 요구 사항 수집 방식의 차이에서 기인합니다. 이러한 차이는 RE 방법의 고르지 않은 채택 때문입니다.



### Policy Aggregation (https://arxiv.org/abs/2411.03651)
- **What's New**: 본 논문에서는 상이한 보상 함수(reward function)와 최적 정책(optimal policy)을 가진 여러 개인(individuals) 간의 AI 가치 정렬(value alignment) 문제를 정책 집계(policy aggregation) 문제로 형식화했습니다. 특히 사회 선택 이론(social choice theory)을 활용하여 효과적인 집계 방법을 제안합니다.

- **Technical Details**: 이 논문은 각 대리인이 MDP(Markov Decision Process) 환경에서 고유의 최적 정책을 가지고 있다는 기본 전제를 바탕으로 그들의 정책을 어떻게 집계해야 할지를 다룹니다. 정책 집계 문제를 해결하기 위해, 우리는 보르다 집계(Borda count), 비례 거부 핵(proportional veto core), 정량 공정성(quantile fairness) 등 다양한 사회 선택 규칙들을 제안하고, 이들이 실제로 어떻게 활용될 수 있는지를 보여주었습니다.

- **Performance Highlights**: 실험을 통해 정량 공정성(quantile fairness)이 특히 매력적인 특성을 보임이 확인되었습니다. 또한, 우리의 접근 방식이 보상에 대한 민감성이 있는 사회 복지 측정 기능을 최적화하는 규칙들보다 우수하다는 것을 보여주었습니다.



### Fully Hyperbolic Rotation for Knowledge Graph Embedding (https://arxiv.org/abs/2411.03622)
- **What's New**: 본 논문에서는 기존의 하이퍼볼릭 회전 모델의 한계를 극복하기 위해 완전 하이퍼볼릭 회전 모델(FHRE)을 제안합니다. 이 모델은 특징 변환을 위한 매핑 방식 대신 로렌츠 모델을 이용하여 하이퍼볼릭 공간에서 직접 정의됩니다.

- **Technical Details**: 제안하는 FHRE 모델은 지식 그래프를 구성하는 각 관계를 헤드 엔티티에서 테일 엔티티로의 로렌츠 회전으로 간주합니다. 또한, triplet의 타당성을 측정하기 위한 scoring function으로 로렌츠 거리 버전을 사용합니다.

- **Performance Highlights**: 표준 지식 그래프 완성 벤치마크인 FB15k-237과 WN18RR에서 경쟁력 있는 성과를 달성했으며, 다양하고 도전적인 데이터셋인 CoDEx-s와 CoDEx-m에서 최첨단 성능을 기록했습니다.



### A Subsampling Based Neural Network for Spatial Data (https://arxiv.org/abs/2411.03620)
- **What's New**: 이 논문은 Geospatial 데이터에서 2층의 localized 깊은 신경망을 기반으로 한 회귀 방법을 제안합니다. 특히, lattice 데이터에서 신경망의 비대칭 분석을 활용하는 방법에 대한 문헌이 없는 상황에서 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 경계가 있는 공간 영역 및 경계가 없는 영역에 대한 신경망의 일관성을 증명하였으며, 고정 샘플링 설계 하에 혼합 증가 영역에서 비대칭 수렴 속도가 기존 연구보다 빠르다는 것을 입증하고 있습니다. 우리 방법론은 시뮬레이션된 lattice 데이터와 미국 주요 도시의 월 평균 기온 데이터 추정을 통한 검증을 포함합니다.

- **Performance Highlights**: 이 접근법은 비선형 공간 회귀의 효과적인 사례로 드러나며, 매끄럽지 않은 공간 표면에서 관측된 데이터와 예측된 데이터 간의 불일치 측정치 수렴 속도가 빨라짐을 관찰하였습니다.



### Designing a Linearized Potential Function in Neural Network Optimization Using Csisz\'{a}r Type of Tsallis Entropy (https://arxiv.org/abs/2411.03611)
- **What's New**: 본 논문은 Shannon 엔트로피에 기반한 정규화 항이 포함된 새로운 최적화 프레임워크를 제안하여, 일반화된 로그 소벨레프 불평등(logarithmic Sobolev inequality)과 그래디언트 플로우(gradient flow) 방정식에서의 잠재 함수(potential function)의 분포 의존성을 해결합니다.

- **Technical Details**: Csiszár 타입의 Tsallis 엔트로피를 이용하여 선형화된 잠재 함수(linearized potential function)를 활용하는 구조를 확립하고, 이를 통해 새로운 프레임워크를 통해 지수 수렴(exponential convergence) 결과를 도출합니다. 보았던 문제 설정에서는 Wasserstein 그래디언트 플로우(Wasserstein gradient flow)와 함께 작용해, 분포적인 해의 시간 미분이 상대 피셔 정보(relative Fisher information)의 음수가 되도록 설정합니다.

- **Performance Highlights**: 제안된 구조는 Tsallis 엔트로피의 특성을 통해 효율적으로 전역 최소(Global Minimum) 문제를 다룰 수 있으며, 실질적인 기계 학습(machin learning) 응용 분야에서의 성능 개선을 기대할 수 있습니다. 또한 Geometric 측면에서 풀 수 있는 문제에 대한 새로운 관점을 제공합니다.



### Learning Constant-Depth Circuits in Malicious Noise Models (https://arxiv.org/abs/2411.03570)
- **What's New**: 이번 논문은 Linial, Mansour, Nisan의 작업을 확장하여, 악의적인 노이즈 모델에서 상수 깊이 회로인 \(\mathsf{AC}^0\)를 학습할 수 있는 쿼사 다항식 시간 알고리즘을 제시합니다. 이는 입력과 레이블 모두가 적대적으로 손상될 수 있는 상황에 초점을 맞추고 있습니다.

- **Technical Details**: 저자들은 Braverman의 정리를 활용하여 아웃라이어 제거 기법을 결합하여 알고리즘을 증명하였습니다. 이들은 상수 깊이 회로에 대한 'nasty noise' 모델을 정의하고, 이 모델에서 얻은 알고리즘의 실행 시간과 상수 관련 종속성이 최적이라고 주장합니다.

- **Performance Highlights**: 이 연구는 가장 강력한 노이즈 모델에서 \(\mathsf{AC}^0\) 회로를 학습할 수 있는 가장 우수한 성능을 달성했으며, 노이즈 비율에 대해서도 최적의 의존성을 나타냅니다.



### The Differentiable Feasibility Pump (https://arxiv.org/abs/2411.03535)
- **What's New**: 본 논문은 feasibility pump 알고리즘을 새로운 시각에서 해석하고, 이를 gradient-descent 알고리즘으로 재프레임팅합니다. 이 해석을 통해 원래의 알고리즘 성능을 개선할 기회를 제시하며, differentiable feasibility pump로 알려진 일반화된 알고리즘을 소개합니다.

- **Technical Details**: 기존의 feasibility pump 알고리즘은 두 가지 주요 작업으로 이루어져 있으며, 선형 제약을 만족하는 해를 찾기 위해 선형 완화(linear relaxation)를 풀고, 이 해를 정수 해로 바꾸는 rounding 작업을 반복합니다. 본 논문에서 제안하는 새로운 접근법은 이 두 단계의 업데이트를 gradient-update 절차를 통해 수행하는 것입니다. 특히, loss function을 조정하여 성능을 향상시킬 수 있는 여러 방법을 모색합니다.

- **Performance Highlights**: MIPLIB 인스턴스에 대한 실험을 통해, 기존의 feasibility pump 알고리즘이 하이퍼파라미터에 대해 안정적이라는 것을 입증합니다. 그러나 feasibility loss를 도입하고 하이퍼파라미터를 조정하는 것만으로도 성능이 크게 향상될 수 있음을 보여줍니다. 또한, differentiable pump을 사용할 경우 restart 작업의 수가 크게 줄어들어 알고리즘이 훨씬 덜 시끄럽게 됩니다.



### Exploring the Potentials and Challenges of Using Large Language Models for the Analysis of Transcriptional Regulation of Long Non-coding RNAs (https://arxiv.org/abs/2411.03522)
- **What's New**: 이 연구에서는 LLMs(대형 언어 모델)를 이용하여 lncRNA 유전자 발현의 전사 조절에 관한 시퀀스 분석을 체계적으로 탐구하고, 기존의 계산적 접근 방식의 한계를 극복할 수 있는 가능성을 조사하고자 합니다.

- **Technical Details**: 본 연구에서는 DNABERT, DNABERT-2, Nucleotide Transformer와 같은 유전체 기초 모델들을 세부 조정(fine-tuning)하여 lncRNA 관련 작업의 성능을 평가했습니다. 또한, 생물학적 설명 가능성을 향상시키기 위한 특징 중요도 분석도 진행하였습니다.

- **Performance Highlights**: 대형 언어 모델을 활용한 실험 결과, 복잡한 작업들에 대한 뛰어난 성능을 보여주었으며, 작업의 복잡성, 모델 선택, 데이터 품질이 lncRNA 분석에 미치는 중요한 영향을 논의하였습니다.



### Forecasting Outside the Box: Application-Driven Optimal Pointwise Forecasts for Stochastic Optimization (https://arxiv.org/abs/2411.03520)
Comments:
          Submitted for publication

- **What's New**: 최근 데이터 가용성이 기하급수적으로 증가함에 따라, 컨텍스트 정보(contextual information)를 활용한 확률적 최적화 문제에 대한 새로운 접근 방식이 제안되었습니다. 이 논문은 특정 컨텍스트 정보에 기반하여 특정 함수의 기댓값을 최적화하는 방법을 다루고 있습니다.

- **Technical Details**: 이 연구에서는 고정된 비용과 고정된 제한행렬을 가진 두 단계 확률적 프로그램(class of two-stage stochastic programs) 문제를 다루었습니다. 완벽한 정보의 시나리오 하나만으로도 문제를 해결할 수 있으며, 다양한 예측 함수들의 모델 내 최적 근사해를 찾는 통합 학습 및 최적화 절차를 제시하였습니다. 이를 통해 머신러닝 기법을 통한 불확실성 예측이 최적화 모델과 결합될 수 있도록 하였습니다.

- **Performance Highlights**: 문헌에서 제안된 재고 관리 및 실제 데이터를 기반으로 한 자전거 공유 문제의 수치 결과를 통하여, 제안한 접근법이 기존 기준 방법에 비해 뛰어난 성과를 보여주었습니다.



### AI Metropolis: Scaling Large Language Model-based Multi-Agent Simulation with Out-of-order Execution (https://arxiv.org/abs/2411.03519)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트의 시뮬레이션을 위한 새로운 엔진인 AI Metropolis를 소개합니다. 이 엔진은 복잡한 작업을 효율적으로 처리할 수 있도록 순서가 뒤바뀐 실행 스케줄링(out-of-order execution scheduling)을 도입하여, 에이전트 간의 실제 종속성을 동적으로 추적함으로써 잘못된 종속성(false dependencies)을 최소화합니다.

- **Technical Details**: AI Metropolis는 LLM 에이전트 간의 관계를 분석하여 시뮬레이션 동안의 의존성을 관리합니다. 이에 따라 각 에이전트의 동작은 시뮬레이션 결과에 영향을 미치지 않고 시간을 진척시킬 수 있습니다. 이러한 접근법은 병렬성을 향상시켜 대량의 LLM 쿼리를 보다 효과적으로 처리할 수 있도록 합니다. AI Metropolis는 OpenAI Gym과 유사한 인터페이스를 제공하며, 시뮬레이션 상태 업데이트, 데이터베이스 I/O, 스케줄링 및 LLM 추론 과정을 자동으로 관리합니다.

- **Performance Highlights**: AI Metropolis는 전통적인 병렬 시뮬레이션 방식에 비해 1.3배에서 4.15배의 속도 향상을 이루었습니다. 에이전트 수가 증가함에 따라 AI Metropolis의 성능은 최적 성능에 근접하게 되며, 이는 AI Metropolis의 확장성과 효과적인 의존성 관리 능력을 보여줍니다.



### Change Is the Only Constant: Dynamic LLM Slicing based on Layer Redundancy (https://arxiv.org/abs/2411.03513)
Comments:
          Accepted at EMNLP Findings 2024

- **What's New**: 본 논문에서는 Large Language Models(LLMs)에서 동적 레이어 전용 가지치기(dynamic layer-specific pruning)를 통해 새로운 모델 압축 방법을 소개하며, 기존의 SliceGPT의 방법론을 발전시켰습니다. 우리는 변동폭이 있는 동적 슬라이싱(dynamic slicing)으로 전환하여, 각 레이어가 입력을 얼마나 변화시키는지를 평가하는 새로운 Layer Redundancy (LR) 점수를 활용합니다.

- **Technical Details**: 제안된 방법은 각 레이어의 중요도에 따라 가지치기 정도를 조절하는 동적 가지치기(dynamic pruning) 방법으로, LR 점수를 기반으로 개별 레이어의 중복성을 평가했습니다. 이를 통해 계산 효율성을 극대화하면서도 모델의 성능 저하를 최소화하도록 설계되었습니다.

- **Performance Highlights**: Llama3-8B 및 Mistral-7B 모델을 사용한 광범위한 실험 결과, 우리의 방법이 SliceGPT에 비해 최대 5%의 성능 향상을 보였으며, 여러 벤치마크에서 나타난 당혹도(perplexity)는 최대 7% 감소했습니다. 이는 우리의 동적 슬라이싱 접근 방식이 기존의 상수 슬라이싱 방법보다 효율적임을 입증합니다.



### An Open-source Sim2Real Approach for Sensor-independent Robot Navigation in a Grid (https://arxiv.org/abs/2411.03494)
Comments:
          Accepted for publication at the 9th IEEE International Conference on Robotics and Automation Engineering (IEEE ICRAE 2024), Singapore

- **What's New**: 이 논문은 Sim2Real (Simulation to Reality) 접근법을 제시하여 시뮬레이션 환경에서 훈련받은 에이전트가 실제 환경에서 자율적으로 이동할 수 있도록 하는 방법을 제안합니다. 특히 Quadruped 로봇을 대상으로 하며, Gymnasium Frozen Lake 환경에서 학습한 정책을 물리적 로봇에 적용하는 파이프라인을 개발하였습니다.

- **Technical Details**: Frozen Lake 환경에서 Reinforcement Learning (RL) 에이전트를 훈련하고, 생성된 Q-table을 활용하여 12 Degrees-of-Freedom (DOF) quadruped 로봇을 제어합니다. 이 과정에서 센서나 추가적인 훈련 없이도 그리드 내에서 자율적으로 이동하고 장애물을 피할 수 있도록 하였습니다. 또한, 이 연구는 GitHub에 오픈 소스로 공개되어 활용이 용이합니다.

- **Performance Highlights**: 이 연구는 저비용의 접근 가능한 프레임워크를 제공하여 연구자, 학생 및 취미로 로봇 탐색을 실험하고 구현할 수 있도록 합니다. 나아가 Frozen Lake 환경을 사용한 Sim2Real 로봇 탐색의 첫 번째 구현으로 주목받고 있습니다.



### Climate AI for Corporate Decarbonization Metrics Extraction (https://arxiv.org/abs/2411.03402)
- **What's New**: CAI 모델은 대규모 언어 모델(LLM)을 활용해 기업의 탄소 감소 목표에 대한 메트릭을 자동으로 추출하고 검증하는 새로운 정보 추출 프레임워크를 제공합니다. 이 접근 방식은 데이터 수집 효율성을 극대화하고 오류를 최소화하는 데 기여합니다.

- **Technical Details**: CAI는 문서 처리를 위한 4단계 프로세스를 활용합니다: 1단계(텍스트 전환 및 청크화), 2단계(관련 텍스트 검색), 3단계(메트릭 추출), 4단계(검증). RoBERTa 모델을 활용하여 텍스트 세그먼트를 분석하고, 맞춤형 프롬프트를 통해 구조화된 메트릭을 추출합니다.

- **Performance Highlights**: CAI 모델은 기존 방법들과 비교하여 생산 품질의 정보 추출 성능을 보이며, 자동화된 데이터 수집 및 검증 과정을 통해 기업의 탄소 감소 목표 데이터를 효과적으로 수집하는 데 성공합니다.



### Solving stochastic partial differential equations using neural networks in the Wiener chaos expansion (https://arxiv.org/abs/2411.03384)
- **What's New**: 이번 논문에서는 확률적 편미분 방정식(stochastic partial differential equations, SPDEs)을 해석하기 위해 무작위 신경망(random neural networks)을 활용하여 수치적으로 해결합니다. 특히, 해당 방정식에 대한 해의 단축된 위너 카오스 전개(truncated Wiener chaos expansion)를 사용합니다.

- **Technical Details**: 이 연구에서는 SPDEs의 해를 근사하는 데 있어 덧셈(additive) 및 곱셈(multiplicative) 노이즈를 고려한 학습의 근사 속도(approximation rates)를 제시합니다. 이러한 접근 방식은 여러 형태의 노이즈가 포함된 방정식에 적합하게 설계되었습니다.

- **Performance Highlights**: 세 가지 SPDE의 해를 근사하기 위한 수치 예제를 통해 저자들은 확률적 열 방정식(stochastic heat equation), 히스-자로-모튼 방정식(Heath-Jarrow-Morton equation), 그리고 자카이 방정식(Zakai equation)에 대한 결과를 적용하였습니다.



### Enhanced Real-Time Threat Detection in 5G Networks: A Self-Attention RNN Autoencoder Approach for Spectral Intrusion Analysis (https://arxiv.org/abs/2411.03365)
Comments:
          This article has been accepted for publication in WiOpt 2024

- **What's New**: 본 논문은 자가 주의(self-attention) 메커니즘을 활용한 RNN 기반 오토인코더를 통해 5G 네트워크에서의 이상 스펙트럼 활동 탐지 성능을 향상시킨 실험 모델을 제시합니다.

- **Technical Details**: 시간-시리즈 분석을 기반으로 하여 I/Q 샘플을 처리하며 잠재적 재밍 공격을 나타낼 수 있는 이상 현상을 식별합니다. 자가 주의 레이어가 추가된 RNN 오토인코더 아키텍처를 통해 RF 스펙트럼의 시간적 의존성과 맥락적 관계를 더욱 정교하게 이해할 수 있게 됩니다.

- **Performance Highlights**: SDR 기반의 실험 환경에서 모델은 위협 탐지에서 성능과 정확도가 크게 향상된 결과를 보여주었습니다.



### DM4Steal: Diffusion Model For Link Stealing Attack On Graph Neural Networks (https://arxiv.org/abs/2411.03364)
- **What's New**: 이 논문에서는 그래프 신경망(GNN)의 링크 도용(link stealing) 공격에 대한 새로운 방법론인 DM4Steal을 제안합니다. DM4Steal은 확산 모델(diffusion model)을 기반으로 하여, 다양한 공격 시나리오와 방어 GNN에 대한 적응성을 보장합니다.

- **Technical Details**: DM4Steal은 세 가지 중요한 측면에서 기존 연구와 다릅니다: (i) 일반성: 한정된 보조 지식(auxiliary knowledge)을 활용하여 여섯 가지 공격 시나리오를 다룰 수 있도록 새로운 훈련 전략을 제안합니다. (ii) 효과성: 확산 모델의 훈련 과정에서 의미적 구조(semantic structure)를 유지하여, GNN 의사결정 과정을 통해 정확한 그래프 토폴로지를 학습할 수 있습니다. (iii) 적응성: GNN 방어 메커니즘이 있을 때도 성능 저하를 최소화하는 안정성을 활용하여 DM4Steal이 성공적인 적응적 공격을 수행할 수 있도록 합니다.

- **Performance Highlights**: DM4Steal은 8개의 실제 데이터 세트에서 3개의 GNN에 대한 광범위한 실험을 통해 최신 기술(state-of-the-art, SOTA) 공격 성능을 달성했습니다. 또한 DM4Steal은 방어 GNN에 대해서도 효과적인 링크 도용 공격을 수행할 수 있습니다.



### TDDBench: A Benchmark for Training data detection (https://arxiv.org/abs/2411.03363)
- **What's New**: 이 논문은 Training Data Detection (TDD)라는 새로운 벤치마크인 TDDBench를 소개합니다. TDDBench는 13개의 데이터셋을 포함하며, 이미지, 표 형식 데이터, 텍스트 등 세 가지 데이터 모드로 구성되어 있습니다. 이 연구는 21개의 다양한 TDD 알고리즘을 벤치마킹하고 그 성능을 다양한 관점에서 평가합니다.

- **Technical Details**: TDDBench는 이미지, 표 형식 데이터, 텍스트의 세 가지 데이터 모드에서 21개의 최신 TDD 알고리즘을 41개의 다양한 타겟 모델에 대해 평가합니다. 각 TDD 알고리즘은 메트릭 기반, 학습 기반, 모델 기반, 쿼리 기반의 네 가지 유형으로 분류됩니다. 특히, 이 논문에서는 블랙박스(black-box) 훈련 데이터 감지를 고려하고, 접근이 제한된 상용 모델을 대상으로 합니다.

- **Performance Highlights**: 대규모 실험 결과, TDD 알고리즘 간의 성능 차이가 있으며, 모델 기반 TDD 방법이 다른 방법보다 일반적으로 높은 성능을 보입니다. 그러나 이들 방법은 높은 계산 비용을 동반합니다. 전반적으로 TDDBench는 TDD 알고리즘의 성능을 평가하고, 비효율적인 알고리즘을 식별하여 개선할 수 있는 기회를 제공합니다.



### Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection (https://arxiv.org/abs/2411.03359)
Comments:
          accepted by NeurIPS 2024

- **What's New**: 이번 연구에서는 Self-Calibrated Tuning (SCT)라는 새로운 프레임워크를 도입하여, 불완전한 OOD (Out-of-Distribution) 특성을 해결하고 효과적인 OOD 탐지를 가능하게 합니다. 이는 단지 몇 개의 ID (In-Distribution) 데이터만으로 수행됩니다.

- **Technical Details**: SCT는 원래 학습 목표의 두 구성 요소에 각각 조정 요인을 도입하여 OOD 정규화의 영향을 조절할 수 있도록 학습 과정에서의 최적화 방향을 적응적으로 조절합니다. 이를 통해 낮은 예측 불확실성을 가진 데이터로 학습할 때 모델이 더 나은 일반화를 이룰 수 있도록 한다.

- **Performance Highlights**: SCT 방법은 대규모 ImageNet-1k 벤치마크에서 기존의 최상의 방법보다 잘못된 긍정 탐지 비율(FPR95)을 3% 개선하였으며, 다양한 실험과 분석을 통해 SCT의 효율성을 검증하였습니다.



### Exploring Feature Importance and Explainability Towards Enhanced ML-Based DoS Detection in AI Systems (https://arxiv.org/abs/2411.03355)
Comments:
          6 pages, 2 figures, IEEE VTC2024-Fall

- **What's New**: 이 논문은 DoS 공격 감지에서 머신러닝(ML) 모델의 성능 향상을 위해 feature selection의 중요성을 조사합니다. 특히, LYCOS-IDS2017 데이터셋을 활용하여 불필요한 feature를 제거하고, 이상 검출을 위한 최적의 feature 조합을 탐색합니다.

- **Technical Details**: 본 연구에서는 주성분 분석(Principal Component Analysis, PCA)을 이용하여 LYCOS-IDS2017 데이터셋의 불필요한 feature를 필터링합니다. 다양한 ML 모델(결정 트리, 랜덤 포레스트, 서포트 벡터 머신 등)을 훈련시키고, 훈련된 모델의 성능을 정확성, 정밀도, 재현율 및 F1 점수 등의 메트릭을 통해 평가합니다.

- **Performance Highlights**: 이 논문의 실험적 결과는 DoS 트래픽에 대한 철저한 통계 분석과 feature engineering이 공격 행동 이해 및 ML 기반 DoS 감지의 정확도 향상에 어떻게 기여하는지를 입증합니다.



### A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness (https://arxiv.org/abs/2411.03350)
Comments:
          76 pages, 26 figures, 14 tables

- **What's New**: 본 논문은 Small Language Models (SLMs)의 정의, 획득, 응용, 향상 및 신뢰성 문제에 대한 포괄적인 조사를 수행하고 있습니다. 이러한 조사는 LLMs의 한계를 극복하기 위해 성장하는 SLMs의 필요성과 관련이 있습니다.

- **Technical Details**: SLMs는 적은 추론 지연(inference latency), 비용 효율성(cost-effectiveness), 효율적인 개발(efficient development), 커스터마이징(customization) 및 적응(adaptability)에서 우수합니다. SLMs의 정의는 그들이 수행하는 전문적인 작업의 능력과 자원 제약이 있는 환경에서의 적합성을 기준으로 설정합니다. 또한, 모델 및 방법에 대한 분류학(taxonomy)과 각 범주에 대한 일반적인 프레임워크를 개발하여 SLMs를 효과적으로 활용하는 방법을 제안합니다.

- **Performance Highlights**: SLMs는 LLMs에 비해 로컬 데이터 처리를 위한 개인 정보 보호, 효율성을 위한 최소한의 추론 지연, 경량 파인 튜닝을 통한 도메인 지식 습득에 최적화된 응용 프로그램에 적합한 성능을 보여줍니다.



### RuAG: Learned-rule-augmented Generation for Large Language Models (https://arxiv.org/abs/2411.03349)
- **What's New**: 본 논문에서는 LLM의 추론 능력을 향상시키기 위해 대량의 offline 데이터를 해석 가능한 1차 논리 규칙으로 자동 변환하고 이를 LLM에 주입하는 새로운 프레임워크인 RuAG를 제안합니다. RuAG는 LLM의 상식에 기반하여 목표 및 본체 술어를 자동 정의하고, Monte Carlo Tree Search를 통해 데이터에서 논리 규칙을 효율적으로 발견합니다.

- **Technical Details**: RuAG는 세 가지 주요 단계로 구성됩니다. 첫째, LLM 기반의 논리 규칙 검색 공식을 통해 LLM이 상식에 기반하여 목표 술어와 본체 술어를 정의합니다. 둘째, MCTS를 이용하여 논리 규칙 검색을 수행하며, 복합적인 검색 공간을 효율적으로 탐색하여 구조화된 1차 논리 규칙을 생성합니다. 셋째, 생성된 논리 규칙을 자연어로 변환하여 LLM의 프롬프트에 주입합니다.

- **Performance Highlights**: RuAG는 자연어 처리, 시계열, 의사 결정 및 산업 과제를 포함한 다양한 공개 및 민간 산업 과제에서 LLM의 능력을 향상시키는 데 효과적임을 입증했습니다. 이 프레임워크는 SFT, ICL, RAG 및 KG 기반 방법과 같은 기존 방법의 한계를 극복하여, LLM의 추론 및 작업 성능을 최소한의 수작업 개입으로 개선합니다.



### Undermining Image and Text Classification Algorithms Using Adversarial Attacks (https://arxiv.org/abs/2411.03348)
Comments:
          Accepted for presentation at Electronic Imaging Conference 2025

- **What's New**: 이 연구는 Generative Adversarial Networks (GANs)와 Synthetic Minority Oversampling Technique (SMOTE)를 결합하여 텍스트 및 이미지 분류 모델에 대한 새로운 adversarial attack 방법론을 제시합니다.

- **Technical Details**: 연구에서 사용된 방법론은 Fast Gradient Sign Method (FGSM)를 이용한 perturbation 벡터와 GradCAM을 통해 강조된 주요 특징을 결합하여 adversarial 공격을 수행합니다. 이는 Convolutional Neural Network (CNN) 모델을 사용하여 얼굴 인식 시스템에 대한 공격을 포함합니다.

- **Performance Highlights**: 이 실험에서 텍스트 분류 모델의 정확도가 공격 후 20% 감소했으며, 얼굴 인식 정확도는 30% 감소했습니다. 이러한 결과는 공격에 대한 모델의 취약성을 강조하며, 머신러닝 시스템의 신뢰성을 저하시킬 수 있습니다.



### Unlocking the Archives: Using Large Language Models to Transcribe Handwritten Historical Documents (https://arxiv.org/abs/2411.03340)
Comments:
          29 Pages, 11 Tables, 2 Figures

- **What's New**: 이 연구는 Large Language Models (LLMs)이 특화된 Handwritten Text Recognition (HTR) 소프트웨어보다 역사적인 손글씨 문서를 훨씬 더 높은 정확도로 기록할 수 있음을 보여줍니다. 이와 함께 개발한 오픈 소스 소프트웨어 도구인 Transcription Pearl은 상업적으로 사용 가능한 다양한 LLM을 활용하여 효율적으로 손글씨 문서를 자동으로 기록하고 수정합니다.

- **Technical Details**: Transcription Pearl은 18세기와 19세기 영어 손글씨 문서의 다양한 데이터셋을 사용하여 테스트 하였으며, LLM은 Character Error Rates (CER) 5.7-7%와 Word Error Rates (WER) 8.9-15.9%를 기록하여, 기존 HTR 소프트웨어인 Transkribus보다 각각 14% 및 32% 개선되었습니다. LLM은 타임라인을 50배 더 빠르고, HTR 프로그램의 1/50의 비용으로 작업을 완료할 수 있게 해줍니다.

- **Performance Highlights**: LLMs는 전통적인 HTR 소프트웨어와 LLM들이 생성한 기록들을 수정함으로써 인력 수준의 정확성에 도달하였으며, CER은 1.8%까지, WER은 3.5%에 달했습니다. 기록 과정은 대략 84-93%의 정확도로 수행되었으며, 이는 대부분의 일상적인 사용 예제에 적합합니다. 결과적으로 LLM을 활용한 HTR은 역사적 손글씨 문서의 대량 기록 프로젝트 및 개별 기록을 다루는 역사학자에게 접근 가능하고 효율적인 방법을 제시합니다.



### Neural Network Prediction of Strong Lensing Systems with Domain Adaptation and Uncertainty Quantification (https://arxiv.org/abs/2411.03334)
Comments:
          Accepted to the Machine Learning for Physical Sciences workshop at NeurIPS 2024; 24 pages, 2 figures, 4 tables

- **What's New**: 이번 연구에서는 강력한 중력 렌즈링 데이터를 이용하여 Mean-Variance Estimators (MVE)와 비지도 도메인 적응 (UDA)을 결합한 첫 번째 연구를 수행하였습니다. 이를 통해 소스 도메인 데이터(노이즈가 없는)와 타겟 도메인 데이터(현대 우주론 조사의 노이즈를 모방)에서의 성능을 비교하였습니다.

- **Technical Details**: MVE는 데이터 레이블의 평균과 분산을 추정하며, 분산은 aleatoric (데이터) 불확실성의 제곱으로 표현됩니다. MVE 손실 함수는 β-NLL(negative log-likelihood)로 설정되며, UDA는 라벨이 없는 타겟 데이터를 사용하는 비지도 방법입니다. 이 연구에서 MVE와 UDA의 결합을 통해 이미지의 노이즈에 의해 구분되는 두 도메인에서의 성능을 비교하였습니다.

- **Performance Highlights**: UDA를 MVE에 추가함으로써 타겟 데이터에서의 정확도가 2배 향상되었습니다. 또한 UDA를 포함시킴으로써 aleatoric 불확실성 예측의 보정도 더욱 정확해졌습니다.



### Hypergraphs as Weighted Directed Self-Looped Graphs: Spectral Properties, Clustering, Cheeger Inequality (https://arxiv.org/abs/2411.03331)
Comments:
          Preprint, 31 pages

- **What's New**: 본 논문은 edge-dependent vertex weights (EDVW) 모델링의 통합된 랜덤 워크 기반 공식화를 제안하며, 이를 통해 hypergraph의 Rayleigh Quotient, NCut, 경계/컷, 볼륨, 그리고 전달성을 정의하고 정의된 개념들이 그래프와의 일관성을 유지하도록 한다.

- **Technical Details**: EDVW 하이퍼그래프를 위해 제안된 HyperClus-G 알고리즘은 NCut과 전달성을 최적화하는 선형적 클러스터링을 수행한다. 또한, 하이퍼그래프 Cheeger Inequality에 대한 전반적인 증명을 제공하고, 기존 결과가 잘못되었음을 밝혀낸다. 소개된 개념들은 하이퍼그래프의 성질을 대부분의 하이퍼그래프에 적용 가능하게 만든다.

- **Performance Highlights**: HyperClus-G는 EDVW 하이퍼그래프에 대해 약선형 최적 분할을 항상 찾아낼 수 있으며, 이는 실험적으로도 확인되었다. 그 결과, EDVW 하이퍼그래프를 위한 스펙트럼 클러스터링 알고리즘의 기초가 확립되었으며, 이 알고리즘은 대부분의 실용적인 하이퍼그래프 응용에 적용될 수 있다.



### Foundation Models for Rapid Autonomy Validation (https://arxiv.org/abs/2411.03328)
- **What's New**: 이 논문에서는 자율주행차의 성능 검증을 위한 새로운 접근 방법을 제시합니다. 주요 내용은 드라이빙 시나리오를 재구성하기 위해 훈련된 마스크드 오토인코더(Masked Autoencoder, MAE)를 사용하는 것입니다. 이 모델은 다양한 주행 시나리오를 효과적으로 그룹화하고, 각 시나리오의 난이도를 평가하여 collision(충돌) 발생 가능성을 기반으로 중요도를 부여합니다.

- **Technical Details**: 본 논문에서는 마스크드 오토인코더(MAE)를 통해 자율주행차의 행동 검증을 위한 사전 훈련(pre-training) 과정을 설명합니다. MAE는 시나리오 입력의 일부분을 마스킹한 후, 부분적으로 마스킹된 입력을 인코딩하고 디코딩합니다. 이 과정에서 비어 있는(브릿지) 입력을 원본과 비교하여 재구성 손실(reconstruction loss)을 계산합니다. 또한, 드라이빙 장면의 희소 표현(sparse representation)을 사용하여 더 높은 해상도의 입력을 처리합니다.

- **Performance Highlights**: 이 접근 방법은 드라이빙 시나리오의 난이도를 기준으로 중요도를 부여함으로써 자율주행차의 충돌 비율과 중대성(severity)을 보다 신속하게 추정할 수 있습니다. 이러한 방식은 자율주행 소프트웨어의 안전성을 평가하는 데 필요한 실제 시뮬레이션 비용을 줄이는 데 기여할 수 있습니다.



### A Surrogate Model for Quay Crane Scheduling Problem (https://arxiv.org/abs/2411.03324)
- **What's New**: 본 연구는 항구에서의 작업 스케줄링 문제인 Quay Crane Scheduling Problem (QCSP)을 보다 빠르고 정확하게 해결하는 방법을 제안합니다.

- **Technical Details**: 연구는 실제 항구 데이터로부터 학습하여 Quay Cranes (QCs)의 작업 속도를 정확하게 예측하는 방법을 제시합니다. 또한, 이 논문은 복잡한 최적화 문제를 해결하는 데 널리 사용되는 Machine Learning (ML) 모델과 Genetic Algorithm (GA)을 결합하여 Surrogate Model을 제안합니다. 이를 통해 다양한 차원의 염색체 인코딩을 사용하여 해결책을 제공할 수 있습니다.

- **Performance Highlights**: 비교 실험을 통해 제안된 방법론이 더 빠른 탐색 속도와 향상된 피트니스 점수를 보여주었습니다. 이 방법은 QCSP에만 국한되지 않으며, 다양한 NP-Hard 문제에도 적용 가능하고, 휴리스틱 알고리즘과 ML 모델을 결합한 고급 탐색 알고리즘 발전의 가능성을 엽니다.



### Satellite monitoring uncovers progress but large disparities in doubling crop yields (https://arxiv.org/abs/2411.03322)
Comments:
          5 pages, 3 figures/tables in main body; 20 pages, 13 figures/tables total including supplementary material and references; pre-print for submission undergoing review

- **What's New**: 이번 연구는 르완다의 15,000개 마을에서 위성 및 기계 학습(Machine Learning)을 통해 작물 수확량을 고해상도로 매핑한 결과를 제시합니다. 이를 통해 2030년까지 생산성을 두 배로 늘리기 위한 진척 상황을 파악하고, 지역적으로 구체적인 생산성 목표를 설계하는 데 기여하고자 합니다.

- **Technical Details**: 이 연구에서 활용한 방법론은 10m의 공간 해상도와 계절적 시간 해상도를 가진 옥수수 재배 및 생산성에 대한 시계열(time series) 데이터를 기반으로 합니다. 2019-2024년에 걸쳐 6만 건의 현장 관측과 9천 건의 작물 수확량 데이터를 사용하여 Gradient Boosted Tree 모델을 구축하였습니다. 이러한 데이터는 Sentinel-2 Level-2A 광학 영상 및 위성으로 유도된 지표면 온도와 강우량 데이터를 통해 예측되었습니다.

- **Performance Highlights**: 연구 결과, 르완다의 전반적인 옥수수 수확량은 목표치를 크게 밑돌고 있으며, 생산성 향상이 지역적으로 불균형하게 분포되어 있음을 보여주었습니다. 현행 정책으로는 SDG 2.3 목표를 달성하기 어려운 상황이며, 특정 정책을 통해 모든 마을의 생산성을 균등하게 증가시킬 경우에만 목표를 달성할 수 있음을 나타냅니다.



### Will Trump Win in 2024? Predicting the US Presidential Election via Multi-step Reasoning with Large Language Models (https://arxiv.org/abs/2411.03321)
Comments:
          This research is ongoing work. Xiyang Hu and Yue Zhao are the corresponding authors

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 선거 예측 능력을 탐구하고, 그에 대한 혁신적인 접근법을 제시합니다. 특히 정치적 분석을 위한 다단계 추론 프레임워크를 제공하여 실제 데이터를 기반으로 한 예측을 향상시킵니다.

- **Technical Details**: 우리는 Sync synthetic data generation framework를 사용하여 개별 유권자의 인구통계 및 행동 프로필을 재구성하고, 2016년 및 2020년 미국 국가선거조사(ANES)의 실제 데이터를 통해 프레임워크를 검증합니다. Chain of Thought prompting에서 영감을 받아, 이 접근 방식은 정치적 맥락의 변화에 맞춰 모델을 조정하며, 인구 통계학적 정보 및 이념적 요인과 같은 다양한 요소들을 체계적으로 통합합니다.

- **Performance Highlights**: 이 모델은 2024년 미국 대통령 선거 결과를 사전에 예측하는 데 성공하여, 보이지 않는 정치적 데이터에 대해 LLMs가 어떻게 적응할 수 있는지를 증명합니다. 최종 파이프라인은 예측 정확성과 실제 결과와의 정합성 모두에서 상당한 개선을 보여줍니다.



### log-RRIM: Yield Prediction via Local-to-global Reaction Representation Learning and Interaction Modeling (https://arxiv.org/abs/2411.03320)
Comments:
          18 pages, 8 figures

- **What's New**: 이 논문에서는 화학 반응의 수율(젤)의 정확한 예측을 위한 혁신적인 그래프 변환기 기반 프레임워크인 log-RRIM을 소개합니다. 이 접근법은 지역에서 글로벌 반응 표현 학습(local-to-global reaction representation learning) 전략을 구현하여 분자 수준 정보를 자세히 캡처하고 분자간 상호작용을 모델링 및 집계합니다.

- **Technical Details**: log-RRIM은 반응제(reactant)와 반응 중심(reaction center) 간의 상호작용에 집중하는 크로스 어텐션(cross-attention) 메커니즘을 통합하였습니다. 이 설계는 반응에서 화학 결합을 형성하고 분해하는 과정에 영향을 주는 반응제의 중요한 역할을 반영하며, 분자의 크기가 다양할 때에도 고르게 주목할 수 있도록 합니다.

- **Performance Highlights**: log-RRIM은 중간에서 고수율 반응에 대해 기존 방법들보다 우수한 성능을 보이며, 소규모 분자 조각들에 대한 민감성을 통해 화학 합성에서의 반응 계획 및 최적화를 위한 가치 있는 도구임을 입증합니다.



### Learning Force Distribution Estimation for the GelSight Mini Optical Tactile Sensor Based on Finite Element Analysis (https://arxiv.org/abs/2411.03315)
- **What's New**: 이 연구에서는 GelSight Mini 센서를 사용하여 비접촉 젤 형태의 변형으로부터 힘 분포를 직접 예측하는 머신러닝 접근법을 제안합니다. U-net 아키텍처를 통해 센서의 원시 이미지로부터 정상(normal) 및 전단(shear) 힘 분포를 예측할 수 있는 모델을 개발하였으며, 이는 기존 방법보다 크게 확장된 응용 가능성을 가집니다.

- **Technical Details**: 제안된 방법인 FEATS(Finite Element Analysis for Tactile Sensing)는 Finite Element Analysis (FEA)를 통해 생성된 힘 분포 데이터를 기반으로 학습한 U-net 네트워크로, GelSight Mini 센서의 원시 RGB 이미지로부터 힘 분포를 추정합니다. 이 방법은 실시간 애플리케이션을 지원할 수 있을 만큼 계산 효율성이 높고, 복잡한 기하학적 정보가 필요하지 않으며, 전통적인 방법보다 더 빠른 추론 시간을 자랑합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 접촉 시나리오에서 높은 차원의 힘 분포를 정확하게 예측하며, 이는 로봇 조작 능력을 향상시키고 다채로운 조작 상황에 적응할 수 있는 가능성을 보여줍니다. 또한, 코드베이스와 데이터셋이 오픈 소스로 제공되어 재사용과 재현성을 지원합니다.




New uploads on arXiv(cs.CL)

### FineZip : Pushing the Limits of Large Language Models for Practical Lossless Text Compression (https://arxiv.org/abs/2409.17141)
- **What's New**: FineZip는 전통적 텍스트 압축 방법에 비해 54배 빠른 압축 시간을 달성함으로써 대규모 텍스트 압축에 대한 사용 가능성을 높입니다.

- **Technical Details**: FineZip는 '온라인' 및 '오프라인' 구성 요소를 결합하여 손실 없는 텍스트 압축을 수행하며, 파라미터 효율적 미세 조정(PEFT) 방식을 사용하여 압축하는 데이터를 Memorize(기억)합니다. 또한, 동적 컨텍스트 사이즈를 활용하여 각 토큰의 압축을 개선하고 병렬 처리 가능성을 높였습니다.

- **Performance Highlights**: FineZip는 기존의 LLMZip보다 약 54배 빠른 압축 성능을 보여주며, 압축 비율을 약 50% 향상시킵니다. 전통적인 알고리즘 기반 압축 방법에 비해 크게 개선된 압축 효율성을 자랑합니다.



### Assessing the Level of Toxicity Against Distinct Groups in Bangla Social Media Comments: A Comprehensive Investigation (https://arxiv.org/abs/2409.17130)
Comments:
          Accepted for publication in "18th International Conference on Information Technology and Applications (ICITA 2024)"

- **What's New**: 이번 연구는 방글라 언어에서 세 가지 특정 집단(트랜스젠더, 원주민, 이주민)에 대한 독성 댓글을 식별하기 위해 다양한 소셜 미디어 출처에서 자료를 수집하고 분석하는 새로운 접근을 제시합니다.

- **Technical Details**: 연구진은 Bangla-BERT, bangla-bert-base, distil-BERT, Bert-base-multilingual-cased와 같은 사전 훈련된 트랜스포머 모델을 사용하여 댓글을 분류합니다. 독성 수준은 높음, 중간, 낮음으로 나눠 측정되며, 정확도(accuracy), 재현율(recall), 정밀도(precision), F1-score 같은 다양한 평가 지표가 사용됩니다.

- **Performance Highlights**: Bangla-BERT 모델은 F1-score 0.8903을 달성하며 다른 대안 모델들을 초월하는 성능을 보였습니다. 이는 방글라 소셜 미디어 대화의 독성 문제를 세밀하게 드러내며, 다양한 인구 집단에 미치는 영향을 분석하여 온라인 차별과 해악 문제를 해결하는 데 기여합니다.



### Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Handy Appetizer (https://arxiv.org/abs/2409.17120)
Comments:
          This book contains 93 pages and 60 figures

- **What's New**: 이 책은 필독서로 인공지능(AI), 머신러닝(ML) 및 딥러닝(DL)이 빅데이터 분석 및 관리의 발전에 미치는 역할을 탐구합니다. 복잡한 수학 개념을 단순화하고, 직관적 시각화 및 실제 사례 연구를 제공하여 독자들이 신경망(neural networks) 및 Convolutional Neural Networks(CNNs)와 같은 기술이 어떻게 작동하는지를 이해하도록 돕습니다.

- **Technical Details**: 책에서는 Transformers, GPT, ResNet, BERT 및 YOLO와 같은 여러 고전 모델과 기술을 소개하며, 자연어 처리(natural language processing), 이미지 인식(image recognition) 및 자율 주행(autonomous driving) 분야에서의 응용을 강조합니다. 사전 훈련된(pre-trained) 모델의 중요성과 그들이 모델의 성능과 정확도를 향상시킬 수 있는 방법을 설명합니다. SQL 및 NoSQL 데이터베이스와 같은 주요 빅데이터 관리 기술의 개관과 Apache Hadoop 및 Spark와 같은 분산 컴퓨팅 프레임워크를 다룹니다.

- **Performance Highlights**: 딥러닝 및 빅데이터 관리 기술을 습득하는 것이 미래 인력에게 중요한 도구라 강조하며, 초보자부터 숙련된 전문가까지 모두에게 필수적인 자료로 자리 잡고 있습니다.



### Programming Every Example: Lifting Pre-training Data Quality like Experts at Sca (https://arxiv.org/abs/2409.17115)
Comments:
          45 pages, 13 figures, 34 tables

- **What's New**: 이번 연구에서는 사전 훈련된 대형 언어 모델에 대해 새로운 접근법인 ProX(Programming Every Example)를 제안합니다. 이는 데이터 정제를 프로그래밍 작업으로 간주하여, 각 개별 예제에 대해 정교한 작업을 생성하고 실행할 수 있게 합니다.

- **Technical Details**: ProX는 모델이 각각의 데이터 예제를 정제하기 위해 필요한 작업을 프로그래밍 방식으로 정의할 수 있도록 하여, 기존의 왜곡된 데이터를 정제하는 데 필요한 유연성을 제공합니다. 이는 문자열 정규화, 데이터 세분화 등의 작업을 포함하여, 0.3B 파라미터를 가진 소형 모델도 인간 전문가와 비슷한 정제 능력을 발휘할 수 있음을 보여줍니다.

- **Performance Highlights**: ProX로 정제된 데이터로 사전 훈련된 모델은 원래 데이터나 다른 필터링 기법으로 정제된 데이터에 비해 다양한 하위 벤치마크에서 2% 이상의 성능 향상을 보였습니다. 특히, OpenWebMath 데이터 세트에서 ProX로 정제된 모델은 Mistral-7B에 비해 평균 정확도가 7.6% 개선되었고, Llama-2-7B에서는 14.6%, CodeLlama-7B에서는 20.3% 향상되었습니다.



### Enhancing Post-Hoc Attributions in Long Document Comprehension via Coarse Grained Answer Decomposition (https://arxiv.org/abs/2409.17073)
- **What's New**: 이 논문은 질문-응답 시스템의 신뢰성을 높이기 위한 정보 출처 표기(attribution) 방법을 다루고 있습니다. 특히 긴 문서에 대한 응답의 출처를 정확히 표기하는 방법에 중점을 두고 있으며, 기존의 표기 방법과는 달리 정보 단위(information units)를 식별하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 저자는 응답을 사실 기반으로 분해하는 factual decomposition 방법을 제안하며, template-based in-context learning 기법을 사용하여 질문과의 맥락을 고려한 응답 분해를 수행합니다. 또한, negative sampling 기법을 통하여 올바른 분해를 구분할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, context를 고려한 coarse-grain decomposition을 사용한 경우, BM25, GTR, MonoT5 기반의 retrieval 시스템에서 평균 3%의 정확도 향상을 보였습니다. 이로 인해 QASPER 및 Verifiability 데이터셋에서 state-of-the-art 성능을 달성하였습니다.



### Detecting Temporal Ambiguity in Questions (https://arxiv.org/abs/2409.17046)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 TEMPAMBIQA라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 8,162개의 오픈 도메인 질문으로 구성되어 있으며, 시간적 애매함(temporal ambiguity)을 포착할 수 있도록 수작업으로 주석 처리되었습니다.

- **Technical Details**: TEMPAMBIQA 데이터셋은 시간적 애매함을 연구하기 위해 설계되었으며, 3,879개의 애매한 질문과 4,283개의 명확한 질문을 포함합니다. 다양한 검색 전략을 사용하여 질문의 시간적 애매함을 감지하는 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: 초기 성능 기준을 설정하며, Zero-Shot Question Classification과 Few-Shot Question Classification을 활용한 질문 분류 방법이 효과적임을 보였습니다. BERT(base 모델)를 미세 조정(Fine-Tuned)하여 질문 분류 작업에 사용했습니다. 실험 결과, 여러 모델이 성능을 발휘했으며, 시간적 문맥을 이해하는 데 중요한 진전을 이루었습니다.



### How to Connect Speech Foundation Models and Large Language Models? What Matters and What Does No (https://arxiv.org/abs/2409.17044)
- **What's New**: 대형 언어 모델(LLM)의 성과가 두드러진 가운데, 본 연구는 음성 인식을 위한 다양한 구성요소(SFM, adapter, LLM)가 하위 작업 성과에 미치는 영향을 최초로 분석합니다.

- **Technical Details**: 다양한 adapter 모듈과 LLM, SFM을 이용하여 자동 음성 인식(ASR) 및 음성 번역(ST) 작업을 수행하였으며, SFM과 LLM의 조합에 따라 최적의 adapter 디자인이 달라짐을 규명하였습니다.

- **Performance Highlights**: SFM의 선택에 따라 ASR와 ST 성능이 평균적으로 각각 1 WER 및 2 COMET 포인트 이상 차이가 발생하였다. 따라서 length adapter의 디자인은 선택된 SFM 및 LLM에 크게 의존하는 것이 밝혀졌습니다.



### LLM-CARD: Towards a Description and Landscape of Large Language Models (https://arxiv.org/abs/2409.17011)
Comments:
          ongoing work, 16 pages

- **What's New**: 이번 연구는 자연어 처리(NLP) 분야에서 빠르게 성장하는 대형 언어 모델(LLM)에 대한 중요한 정보를 자동으로 추출하고 정리하는 시스템을 개발했습니다. 이 시스템은 논문에서 LLM에 관한 정보를 효율적으로 검색할 수 있도록 하는 LLM 모델 카드를 생성합니다.

- **Technical Details**: 이 연구에서는 명명된 개체 인식(Named Entity Recognition, NER)과 관계 추출(Relation Extraction, RE) 방법을 사용하여 LLM에 대한 주요 정보를 자동으로 추출합니다. 106개의 학술 논문을 처리하여 3개의 사전(LLM 이름, 라이센스, 응용)을 정의하고, 11,051개의 문장을 추출하였으며 최종적으로 129개의 문장과 106개의 문장을 수작업으로 검토하여 데이터 세트를 구축하였습니다.

- **Performance Highlights**: 이 시스템은 연구자들이 LLM에 대한 정보에 쉽게 접근할 수 있도록 돕고, 라이센스 유형, 활용 분야 등을 간단하게 이해할 수 있게 합니다. 또한, 자동화를 통해 연구자들이 시간이 절약되고 혁신에 집중할 수 있는 기회를 제공합니다.



### Decoding Large-Language Models: A Systematic Overview of Socio-Technical Impacts, Constraints, and Emerging Questions (https://arxiv.org/abs/2409.16974)
Comments:
          28 pages, 5 figures, preprint submitted to journal

- **What's New**: 최근 대형 언어 모델(LLM)의 발전이 자연어 처리(NLP)와 인공지능(AI) 분야에 혁신적인 변화를 가져왔습니다. 이 연구에서는 LLM의 개발 방향, 영향력 및 한계에 대한 체계적인 문헌 조사를 수행하였습니다.

- **Technical Details**: 논문은 LLM 연구의 목표, 방법론, 제한 사항 및 향후 방향성을 서술하며, 알고리즘 개선, 윤리적 도전 과제, 사회적 영향을 포함하여 책임 있는 개발에 대한 고려 사항을 포함합니다. 또한, 체계적인 리뷰 방법론을 통해 문헌을 분석하고, 150회 이상의 인용된 61개의 주요 논문을 선정했습니다.

- **Performance Highlights**: LLM은 번역, 분류, 질문-응답, 요약 및 정보 검색과 같은 복잡한 작업을 수행하는 데 탁월한 성능을 보여줍니다. 특히, GPT-3와 같은 모델은 창의적인 콘텐츠 생성과 대화 시뮬레이션에서 뛰어난 다양한 기능을 발휘하고 있습니다.



### Adaptive Self-Supervised Learning Strategies for Dynamic On-Device LLM Personalization (https://arxiv.org/abs/2409.16973)
Comments:
          First ASLS

- **What's New**: 이번 논문에서는 Adaptive Self-Supervised Learning Strategies (ASLS)를 제안합니다. ASLS는 대규모 언어 모델(LLMs)을 사용자 개인의 선호도에 맞게 동적으로 개인화하는 혁신적인 방법으로, 라벨이 있는 데이터셋의 필요성을 줄이고 실시간 피드백을 기반으로 모델을 조정합니다.

- **Technical Details**: ASLS는 사용자 프로파일링 레이어와 신경망 적응 레이어의 이중 레이어 구조로 구성되어 있습니다. 사용자와의 상호작용 데이터를 수집하여 모델을 실시간으로 미세 조정하며, 이는 계속해서 사용자 피드백을 학습하여 사용자별 맞춤형 응답을 생성합니다. 이 접근법은 계산 자원을 절약하고 개인화 효율성을 높입니다.

- **Performance Highlights**: 다양한 사용자 시나리오에 대한 실험 결과, ASLS는 기존의 개인화 방법 대비 사용자의 참여도와 만족도를 크게 향상시켰습니다. 이러한 결과는 ASLS가 대규모 언어 모델을 보다 반응성이 뛰어나고 맥락을 인지하는 시스템으로 변모시킬 잠재력을 보여줍니다.



### Weighted Cross-entropy for Low-Resource Languages in Multilingual Speech Recognition (https://arxiv.org/abs/2409.16954)
Comments:
          5 pages, 1 figure. Presented at Interspeech 2024

- **What's New**: 이번 논문에서는 저자들이 저자원 언어(low-resource language)를 다국어 자동 음성 인식(multilingual automatic speech recognition, ASR) 시스템에 통합하기 위해 새로운 접근 방식을 제안합니다. 일반적으로 불균형 데이터셋을 다루는 데 사용되는 weighted cross-entropy를 활용하여 미리 학습된 다국어 ASR 모델에 저자원 언어를 통합하는 방법을 제시합니다.

- **Technical Details**: 저자들은 Whisper 모델을 활용하여 5개의 고자원 언어와 1개의 저자원 언어를 fine-tuning하며, 언어 가중치 동적 cross-entropy와 데이터 증강(data augmentation) 기법을 적용합니다. 저자원 언어의 경우, 제안된 접근 방식을 적용하지 않은 fine-tuned 모델에 비해 6.69%의 단어 오류율(word error rate, WER) 감소를 보여주며, 원래 Whisper 모델과 비교하여 48.86% 감소를 기록했습니다. 이를 통해 6개 언어에서 평균 3.29%의 WER 감소 효과를 보였습니다.

- **Performance Highlights**: 이 연구의 접근 방식은 저자원 언어의 인식 정확성을 크게 향상시키고, 고자원 언어의 성능 저하 없이 평균 32.5%의 WER 감소를 보여주었습니다. 이러한 결과는 저자원 언어를 다국어 ASR 모델에 성공적으로 통합하는 새로운 방법론을 나타냅니다.



### Investigating OCR-Sensitive Neurons to Improve Entity Recognition in Historical Documents (https://arxiv.org/abs/2409.16934)
- **What's New**: 이 논문은 Transformer 아키텍처 내에서 OCR 민감한 뉴런의 존재를 조사하고, 역사적 문서에 대한 명명된 개체 인식(NER) 성능에 미치는 영향을 분석합니다. 깨끗한 텍스트 입력과 잡음이 있는 텍스트 입력에 대한 뉴런 활성화를 분석하여 OCR 민감한 뉴런을 식별하고 중화시킴으로써 모델 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 실험은 Llama2와 Mistral 두 개의 오픈 액세스 대형 언어 모델을 기반으로 하며, OCR 잡음이 존재하는 텍스트에 대한 뉴런의 반응을 측정하여 OCR 민감한 레이어와 뉴런을 식별하는 데 중점을 둡니다. 이 과정에서 데이터셋은 프랑스 역사 신문의 OCR 수정 버전을 기반으로 생성되며, 다양한 수준의 OCR 잡음이 추가된 토큰을 사용합니다.

- **Performance Highlights**: 실험 결과, 역사 신문 및 고전 해설 문서에서 NER 성능이 개선되는 것으로 나타났고, 이는 특정 뉴런 조절이 잡음이 있는 텍스트에서 모델의 성능을 향상시킬 수 있음을 시사합니다.



### Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness (https://arxiv.org/abs/2409.16914)
Comments:
          To appear at the main conference of EMNLP 2024

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)로 생성된 텍스트를 자동으로 감지하기 위한 새로운 특징인 'token cohesiveness'를 제안하고, 이를 기반으로 TOCSIN이라는 제로샷 감지기(detection) 패러다임을 개발하였습니다.

- **Technical Details**: TOCSIN은 기존의 제로샷 감지기와 token cohesiveness 계산 모듈을 갖춘 이중 채널(dDual-channel) 감지기로 구성되어 있습니다. Token cohesiveness는 입력 텍스트에서 무작위로 일정 비율의 토큰을 삭제한 후, 잔여 텍스트와 그 복사본의 의미적 차이를 측정하여 계산됩니다. 이 방법은 BARTScore를 활용하여 평균적인 의미적 차이를 측정하며, 블랙박스 환경에서도 적합합니다.

- **Performance Highlights**: TOCSIN은 4개의 최신 제로샷 감지기와 함께 여러 다양한 데이터셋에서 실험을 수행하여, 기존 감지기 대비 높은 정확도를 기록하였습니다. 실험 결과는 TOCSIN이 기존의 제로샷 감지기보다 일관되게 의미 있는 개선을 보임을 시사합니다.



### Pruning Multilingual Large Language Models for Multilingual Inferenc (https://arxiv.org/abs/2409.16911)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 Multilingual Large Language Models (MLLMs)의 비영어 성능을 향상시키기 위해 이 모델들이 영어와 비영어 간의 정렬(alignment) 능력을 활용할 수 있는 방법을 탐구했습니다. 특히, 특히 큰 크기의 기능(features)에 집중하여 비번역(zero-shot) 작업의 성능을 높이는 방법을 제시하였습니다.

- **Technical Details**: MLLMs의 번역 성능 분석에서 큰 크기의 기능이 번역 과정에 중대한 역할을 한다는 것을 발견하였습니다. 이를 통해 우리는 큰 크기의 기능과 관련된 가중치(weights)를 유지하고 나머지 가중치는 줄여서 MLLMs가 비번역 작업에 대해 이러한 기능에 의존하도록 강요하는 전략을 구현하였습니다. 이 방법은 XNLI 및 MARC 작업에서 비영어 성능을 개선하는 데 성공적이었습니다.

- **Performance Highlights**: 분석 결과, 큰 크기의 기능을 사용하는 것이 MLLMs의 비영어 언어에서의 성능을 개선하는 데 기여한다는 사실을 실증적으로 보여주었습니다. XGLM과 mGPT에서 성능 향상을 확인했으나 BLOOM 모델은 프로그래밍 언어 처리 능력 때문에 다소 혼란스러운 결과를 보였습니다. 이 연구는 비영어 비약 연산에 대한 가중치 프루닝(pruning) 전략이 성능 개선에 효과적임을 입증하였습니다.



### Enhancing Temporal Sensitivity and Reasoning for Time-Sensitive Question Answering (https://arxiv.org/abs/2409.16909)
Comments:
          Accepted by EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Time-Sensitive Question Answering (TSQA) 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Temporal Information-Aware Embedding과 Granular Contrastive Reinforcement Learning을 통해 모델의 시간 인식 및 추론 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 방법론을 포함합니다: Temporal Information-Aware Embedding은 모델의 시간 정보에 대한 민감성을 높이고, Granular Contrastive Reinforcement Learning은 시간적 거리에 따라 원거리 및 근접한 부정적 답변을 제공하여 모델의 시간적 추론 능력을 향상시킵니다.

- **Performance Highlights**: 우리는 제안된 프레임워크가 기존의 LLM보다 TSQA 작업에서 유의미하게 뛰어난 성능을 보여주는 것을 확인했습니다. 실험 결과는 네 개의 다양한 TSQA 데이터셋에서 우리의 프레임워크가 기존 모델보다 크게 향상된 성능을 보였음을 입증합니다.



### Shifting from endangerment to rebirth in the Artificial Intelligence Age: An Ensemble Machine Learning Approach for Hawrami Text Classification (https://arxiv.org/abs/2409.16884)
Comments:
          19 pages, 7 tables, 14 figures

- **What's New**: 이번 연구에서는 6,854개의 Hawrami 텍스트 기사를 기반으로, 텍스트 분류를 위한 다양한 모델을 도입하였습니다. 이는 Hawrami라는 키르디시 방언의 데이터 부족 문제를 해결하는 데 기여하고자 하는 시도로서, 두 명의 원어민이 15개 카테고리로 라벨링하였습니다.

- **Technical Details**: 분류 작업을 위해 K-nearest Neighbor (KNN), Linear Support Vector Machine (Linear SVM), Logistic Regression (LR), Decision Tree (DT) 등의 기법을 활용하였으며, Linear SVM이 96%의 정확도로 가장 우수한 성과를 보였습니다.

- **Performance Highlights**: Linear SVM 모델이 다른 방법들보다 뛰어난 성과를 보여, Hawrami 방언의 텍스트 분류 작업에서 핵심적인 기여를 할 것으로 기대됩니다.



### CodeInsight: A Curated Dataset of Practical Coding Solutions from Stack Overflow (https://arxiv.org/abs/2409.16819)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 새로운 CodeInsight 데이터셋을 소개하며, 이는 개발자의 일반적인 작업을 지원하기 위해 독창적으로 설계되었습니다. 이 데이터셋에는 명확한 의도, 관련된 코드 스니펫, 그리고 평균 3개의 관련 단위 테스트 예제가 포함되어 있습니다. 또한, Pandas, Numpy, Regex 및 70개 이상의 표준 Python 라이브러리를 포함하고 있습니다.

- **Technical Details**: 데이터셋은 3,409개의 Python 전문가에 의해 작성된 예제로 구성되어 있으며, 모델을 세부 조정(finetuning) 및 독립적 평가(standalone evaluation)에 사용할 수 있도록 설계되었습니다. 단위 테스트 평가를 완료하기 위해 예제를 분류하여 세부 분석을 가능하게 합니다. 데이터 오염(data contamination) 감소를 위해 예제를 다듬었으며, Mistral 7B, CodeLLaMa 13B, Starcoder 15B 등 세 가지 주요 모델 성능을 통해 확인하였습니다.

- **Performance Highlights**: CodeInsight 데이터셋은 코드 생성 분야에서의 주요 혁신을 포함하고 있습니다. 단위 테스트 기반 평가를 통해 BLEU score와 같은 기존 방법보다 더 강력한 평가 지표를 제공합니다. 또한 예제들은 강점과 약점을 더 깊이 분석할 수 있도록 주석이 달려 있으며, 각 예제는 수작업으로 큐레이션되어 고품질을 보장합니다.



### A Few Hypocrites: Few-Shot Learning and Subtype Definitions for Detecting Hypocrisy Accusations in Online Climate Change Debates (https://arxiv.org/abs/2409.16807)
Comments:
          cite the public version, published at CPSS 2024 @ KONVENS

- **What's New**: 이번 연구는 누군가를 비난하기 위해 사용하는 '위선' (hypocrisy) 주장 탐지를 독립적 작업으로 정의합니다. 기존 연구에서는 위선 주장이 논리적 오류 탐지의 일부로 다뤄졌으나, 본 연구는 이를 새로운 방식으로 접근하여 '환경 위선 주장 데이터셋' (Climate Hypocrisy Accusation Corpus, CHAC)을 구축합니다.

- **Technical Details**: 420개의 Reddit 기후 토론 댓글로 구성된 CHAC 데이터셋은 개인적 위선과 정치적 위선 두 가지 유형으로 주석이 달려있습니다. 이 데이터셋을 기반으로 6샷 방법과 3개의 instruction-tuned Large Language Models (LLMs)을 활용하여 위선 주장 탐지의 성능을 평가합니다.

- **Performance Highlights**: GPT-4o와 Llama-3 모델이 특히 뛰어난 성능을 보여주었으며, F1 점수는 0.68에 도달했습니다. 이는 이전 연구의 F1 점수 0.44에 비해 현저히 향상된 수치입니다. 연구 결과, 모델은 개인적 도덕적 위선 주장을 탐지하는 데는 효과적이나, 정치적 위선 주장을 탐지하는 데는 어려움을 겪었던 것으로 나타났습니다.



### Mitigating the Bias of Large Language Model Evaluation (https://arxiv.org/abs/2409.16788)
- **What's New**: 이번 연구에서는 LLM(as-a-Judge)의 평가 편향을 체계적으로 연구하였습니다. 특히 비공식 소스의 LLM 판별 모델들에서 겉보기 품질(superficial quality)의 중요성을 줄이기 위한 보정(calibration) 방법을 적용하고, 공식 소스 모델에서는 대비 학습(contrastive training)을 제안하여 편향을 완화하고자 했습니다.

- **Technical Details**: 비공식 LLM 판별자의 경우, 확률 기반 평가자와 생성 기반 평가자 모두에서 겉보기 품질을 모델링하여 최종 결과에서 이를 차감하는 두 가지 방법을 제안하였습니다. 이는 자연어 테스트 세트에서 편향을 평가하는 지표를 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법들이 기존 평가 편향을 크게 완화시키면서도 자연어 세트에 대한 평가 정확도를 만족스럽게 유지했음을 보여주었습니다.



### Holistic Automated Red Teaming for Large Language Models through Top-Down Test Case Generation and Multi-turn Interaction (https://arxiv.org/abs/2409.16783)
Comments:
          EMNLP 2024 camera ready version

- **What's New**: 본 논문에서는 HARM (Holistic Automated Red teaMing)을 제안하여, 대형 언어 모델 (LLMs)의 비정상적인 행동을 체계적으로 식별하는 새로운 방법을 소개합니다. HARM은 리스크 카테고리의 세분화된 분류를 바탕으로 한 상향식 접근 방식을 사용하고, 멀티 턴 상호작용을 지원하여 자동화된 레드 팀핑을 강화합니다.

- **Technical Details**: HARM은 세분화된 리스크 분류 체계를 사용하여 다양한 테스트 케이스를 생성하는 방법을 적용합니다. 이 방법은 고유의 파인 튜닝 전략과 강화 학습 기법을 활용하여 인간과 유사한 방식으로 멀티 턴에서의 적대적인 탐색을 수행합니다.

- **Performance Highlights**: 실험 결과를 통해 HARM이 모델의 취약성에 대한 보다 체계적인 이해를 가능하게 하고, 안전한 정렬 과정을 위한 보다 구체적인 가이드를 제공함을 보여주었습니다.



### E-SQL: Direct Schema Linking via Question Enrichment in Text-to-SQL (https://arxiv.org/abs/2409.16751)
- **What's New**: 본 논문에서는 E-SQL이라는 새로운 파이프라인을 소개하며, 이는 자연어 질의를 데이터베이스 스키마와 직접 연결하고 후보 술어를 증강하여 더 복잡한 질의를 다룰 수 있도록 설계되었습니다.

- **Technical Details**: E-SQL은 후보 SQL 생성(CSG), 후보 술어 생성(CPG), 질문 증강(QE), SQL 정제(SR)의 네 가지 주요 모듈로 구성됩니다. 이 시스템은 자연어 질의에 관련된 데이터베이스 요소를 통합하고, SQL 실행 오류를 감지하여 질의를 개선합니다.

- **Performance Highlights**: BIRD 벤치마크에서 E-SQL은 복잡한 질의를 효과적으로 처리하고, 테스트 세트에서 66.29%의 실행 정확도로 경쟁력 있는 성능을 달성하며, 고급 LLM(대형 언어 모델) 환경에서 더 나은 결과를 보여주었습니다.



### RoleBreak: Character Hallucination as a Jailbreak Attack in Role-Playing Systems (https://arxiv.org/abs/2409.16727)
- **What's New**: 이 논문은 캐릭터 환각(character hallucination)의 체계적인 분석을 제시하며, 이를 공격 관점에서 탐구하는 RoleBreak 프레임워크를 소개합니다. 이 프레임워크는 두 가지 주요 메커니즘인 쿼리 희소성(query sparsity)과 역할-쿼리 충돌(role-query conflict)을 캐릭터 환각의 핵심 요소로 식별합니다.

- **Technical Details**: RoleBreakEval이라는 새로운 데이터셋을 구축하고, 이를 통해 기존의 환각 완화 기법을 평가합니다. 두 가지 주 요인, 즉 쿼리 희소성과 역할-쿼리 충돌을 기반으로 공격 쿼리를 반자동으로 생성하여, LLM 기반 역할 놀이 시스템의 취약성을 입증합니다. 이 진행 과정에서 Narrator Mode라는 새로운 방어 전략을 제안하여, 보강된 내러티브 컨텍스트를 생성하고 역할 지침과 사용자 쿼리 간의 충돌을 완화합니다.

- **Performance Highlights**: 실험 결과, Narrator Mode는 전통적인 거부 기반 전략보다 우수한 성과를 보이며, 캐릭터 역할에 대한 충실도를 높이고 전체 내러티브 일관성을 향상시킵니다.



### PMSS: Pretrained Matrices Skeleton Selection for LLM Fine-tuning (https://arxiv.org/abs/2409.16722)
- **What's New**: 최근에 제안된 PMSS(Pre-trained Matrices Skeleton Selection)는 LoRA의 한계를 극복하고, 사전 훈련된 가중치 내의 의미적 정보를 활용하여 높은 랭크 업데이트를 가능하게 하는 새로운 fine-tuning 방법입니다.

- **Technical Details**: PMSS는 사전 훈련된 행렬에서 스켈레톤을 선택하여 작은 행렬만 학습하도록 설계되었습니다. 이를 통해 낮은 비용으로 높은 랭크 업데이트를 달성합니다. PMSS는 DROP, commonsense reasoning, 수학적 추론과 같은 복잡한 작업에서 LoRA보다 우수한 성능을 보입니다.

- **Performance Highlights**: PMSS는 LLaMA2-7B/13B의 DROP 벤치마크에서 각각 +3.4%/+5.9% 성능 향상을 보였고, GSM8K 데이터셋에서 Mistral-7B, Gemma-7B에 대해 각각 +12.89%/+5.61%/+3.11%의 성과를 달성했습니다.



### Probing Omissions and Distortions in Transformer-based RDF-to-Text Models (https://arxiv.org/abs/2409.16707)
Comments:
          Accepted for publication in Transactions of the ACL (TACL)

- **What's New**: 이 논문은 자연어 생성(Natural Language Generation, NLG)에서 중요한 정보가 출력 텍스트에서 누락되는 문제를 분석하며, RDF(Graph)를 텍스트로 변환하는 과정에서 이러한 누락(simulation) 및 왜곡(distortion) 현상을 탐색합니다. BART와 T5 모델의 인코더 출력에서 누락을 탐지하기 위한 새로운 두 가지 방법을 제시합니다.

- **Technical Details**: (i) 파라미터가 필요 없는 프로빙 방법으로, RDF 그래프의 임베딩과 일부 엔티티를 제거한 RDF 그래프의 임베딩 간의 코사인 유사도(cosine similarity)를 계산합니다. (ii) 이진 분류기(binary classifier)로, 인코더 임베딩을 분석하여 누락된 엔티티를 탐지합니다. 연구에서는 RDF 데이터를 영어로 변환한 텍스트에서 누락된 엔티티를 분석하며, 이러한 누락 및 왜곡이 인코더의 출력 임베딩에서 탐지될 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 두 가지 프로빙 방법 모두에서 인코더 출력에서 누락된 엔티티 및 왜곡된 엔티티의 탐지가 가능함을 입증했습니다. 또한, 로지스틱 회귀(logistic regression)를 활용하여 해당 엔티티가 누락되거나 왜곡될 가능성을 피처(feature) 기반으로 예측할 수 있음을 발견했습니다.



### SynTQA: Synergistic Table-based Question Answering via Mixture of Text-to-SQL and E2E TQA (https://arxiv.org/abs/2409.16682)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 Text-to-SQL 파싱 및 엔드 투 엔드 질문 응답(E2E TQA) 접근 방식의 성과를 비교하고 이들이 갖고 있는 장점을 통합하는 Synergistic Table-based Question Answering(SynTQA) 방법론을 제안합니다.

- **Technical Details**: Text-to-SQL은 숫자 연산 및 긴 테이블 처리에 능숙하고, E2E TQA는 모호한 질문, 비표준 테이블 스키마 및 복잡한 테이블 내용을 잘 처리합니다. 이 연구는 이러한 두 모델을 통합하여 답변 선택(answer selection) 방법을 통해 개선된 성능을 확인했습니다.

- **Performance Highlights**: 실험 결과, feature 기반 및 LLM 기반 답변 선택기를 통해 기존 모델보다 향상된 성능을 보였으며, 복잡한 질문과 테이블에 대해서는 두 가지 접근 방식의 상호 보완적인 강점이 드러났습니다.



### SWE2: SubWord Enriched and Significant Word Emphasized Framework for Hate Speech Detection (https://arxiv.org/abs/2409.16673)
Comments:
          Published in CIKM 2020

- **What's New**: 이 논문에서는 Hate speech(혐오표현) 문제를 해결하기 위한 새로운 프레임워크(SWE2)를 제안합니다. SWE2는 메시지의 내용만을 기반으로 하며, 혐오표현을 자동으로 식별합니다.

- **Technical Details**: SWE2는 두 가지 유형의 서브워드 임베딩(phonetic-level embedding 및 character-level embedding)을 활용합니다. 또한, LSTM+attention 기반의 특성 추출 방법을 설계하여, 전반적인 내용의 의미 정보를 추출합니다. 이 프레임워크는 FastText와 BERT에서의 사전 학습된 변형을 비교하여 어떤 것이 서브워드 표현을 보완하는지 확인했습니다.

- **Performance Highlights**: 우리 모델은 백도어 공격 없이 0.975의 정확도와 0.953의 macro F1 점수를 달성하며 7개의 최신 방법론보다 우수합니다. 극단적인 공격 상황에서도 0.967의 정확도와 0.934의 macro F1 점수를 유지하여 높은 견고성을 입증하였습니다.



### Topic-aware Causal Intervention for Counterfactual Detection (https://arxiv.org/abs/2409.16668)
Comments:
          Accepted to the 4th EMNLP-NLP4DH 2024 workshop

- **What's New**: 이 논문에서는 Counterfactual Detection (CFD) 모델의 성능을 향상시키기 위해 neural topic model (NTM)와 인과적 개입 방법을 통합하는 새로운 접근 방식을 제안합니다. 기존 모델들이 clue phrases에 의존하여 성능 저하가 발생하는 문제를 해결하고자 합니다.

- **Technical Details**: 제안하는 모델은 NTM을 CFD 모듈에 통합하여 입력 문장의 전반적인 의미를 파악합니다. 또한 hidden representation에 대해 인과적 개입(causal intervention)을 시행하여 클래스 불균형(class imbalance)의 부정적인 영향을 줄입니다. 모델은 Variational AutoEncoder 아키텍처를 기반으로 하며, topic encoder와 decoder로 구성되어 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안한 방법이 기존의 최첨단 CFD 및 bias-resolving 접근 방식보다 뛰어난 성능을 보임을 입증하였습니다. 이 방법은 다른 bias-sensitive 자연어 처리 작업에도 적용 가능합니다.



### A Character-Centric Creative Story Generation via Imagination (https://arxiv.org/abs/2409.16667)
- **What's New**: 본 논문에서는 창의적인 스토리 생성을 위한 새로운 프레임워크인 CCI (Character-centric Creative story generation via Imagination)를 소개합니다. CCI는 창의적인 스토리를 생성하기 위해 IG (Image-Guided Imagination)와 MW (Multi-Writer model) 두 가지 혁신적인 모듈을 특징으로 합니다.

- **Technical Details**: IG 모듈은 DALL-E 3를 사용하여 이야기의 주요 요소들(예: 캐릭터, 배경 등)의 시각적 표현을 생성하며, MW 모듈은 IG에서 생성된 스토리 요소를 사용하여 주인공에 대한 여러 설명 후보를 생성하고 이후 가장 적합한 설명을 선택합니다.

- **Performance Highlights**: 연구 결과, CCI 모델이 생성한 스토리는 다양성과 캐릭터의 일관성 모두에서 우수한 성능을 보였으며, 인간 평가에서는 스토리의 창의성, 생동감, 구체성 및 일관성에서 더 높은 선호도를 보였습니다.



### Pre-trained Language Models Return Distinguishable Probability Distributions to Unfaithfully Hallucinated Texts (https://arxiv.org/abs/2409.16658)
Comments:
          10 pages, EMNLP 2024 Findings

- **What's New**: 본 연구에서는 사전 학습된 언어 모델(Pre-trained Language Models, PLMs)이 신뢰할 수 없는 환각(hallucinated) 텍스트에 대해 구별 가능한 생성 확률과 불확실성 분포를 반환한다는 것을 보여줍니다. 이는 모델의 크기나 구조에 상관없이 88-98%의 경우에서 통계적으로 유의미하게 구별되는 결과를 나타냈습니다.

- **Technical Details**: 우리는 24개의 PLM을 6개의 데이터 세트에서 테스트하였으며, 이 과정에서 높은 로스를 가진 데이터 포인트를 절단함으로써 보다 신뢰할 수 있는 뉴스 제목을 생성할 수 있음을 발견했습니다. 또한, 생성 확률과 불확실성이 신뢰성과 긍정적인 상관관계를 가짐을 확인하였습니다. 이 연구에서 제안한 알고리즘은 환각 현상을 줄이는 데 효과적이며, 모델을 파인튜닝(fine-tuning) 할 때 불확실성을 고려하면서 그런 효과를 보였습니다.

- **Performance Highlights**: 제안한 알고리즘은 다른 기준 모델들에 비해 더 높은 신뢰성 메트릭을 달성하였으며, 전반적인 텍스트 품질 지표를 유지하면서도 탁월한 성능을 보였습니다.



### Domain-Independent Automatic Generation of Descriptive Texts for Time-Series Data (https://arxiv.org/abs/2409.16647)
- **What's New**: 이번 연구에서는 시간 시계열 데이터로부터 설명적인 텍스트를 체계적으로 생성하는 도메인 독립적인 방법을 제안합니다. 특히, 시간 시계열 데이터와 설명 텍스트의 쌍을 생성하기 위해 두 가지 접근 방법, 즉 forward approach와 backward approach를 정의하고, 새로운 backward approach를 통해 TACO 데이터셋을 생성합니다.

- **Technical Details**: TACO (Temporal Automated Captions for Observations) 데이터셋은 120만 개의 실제 시간 시계열 데이터 샘플을 사용하여 생성되었습니다. 연구진은 먼저 시간 시계열 클래스 세트를 정의하고, 각 클래스를 기준으로 점수를 계산하여 시간 시계열 데이터를 분류한 후, 이에 해당하는 설명 텍스트를 생성했습니다. 이 과정에서 min-max scaling을 사용하여 데이터와 점수를 정규화하였으며, Llama-3-8B-Instruct 모델을 활용하여 기초 설명 텍스트를 재구성했습니다.

- **Performance Highlights**: 제안된 방법으로 훈련된 contrastive learning 기반 모델은 새로운 도메인에서도 시간 시계열 데이터에 대해 도메인 독립적인 설명 텍스트를 효과적으로 생성할 수 있음을 실험 결과를 통해 입증하였습니다.



### Cross-Lingual and Cross-Cultural Variation in Image Descriptions (https://arxiv.org/abs/2409.16646)
- **What's New**: 이번 연구는 31개 언어 및 다양한 장소의 이미지를 포함하는 대규모 다중모달 데이터셋을 활용하여, 서로 다른 언어 구사자들이 이미지 설명에서 어떻게 다르게 언급하는지를 체계적으로 분석한 첫 연구입니다. 특히 지리적, 유전적으로 가까운 언어 쌍에서 유사한 개체가 더 빈번하게 언급되는 경향을 발견했습니다.

- **Technical Details**: 연구에서 WordNet을 사용하여 이미지 설명에서 언급된 개체를 식별하는 자동화된 방법론을 개발했습니다. XM3600 데이터셋을 사용하여 서로 다른 언어에서의 개체 언급 변화를 정량적으로 분석하는 데 중점을 두었습니다. 이 과정에서 이미지에 포함된 개체 카테고리를 독립적으로 주석으로 추가했습니다.

- **Performance Highlights**: 일부 언어 쌍(예: 일본어는 영어보다 의류를 훨씬 더 자주 언급)에서의 차이를 측정한 사례 연구를 통해 엔티티 카테고리의 눈에 띄는 정도와 언어에 따른 변동성을 확인했습니다. 또한, 이전의 소규모 연구 결과를 지지하는 데이터를 제공하며 기본 수준 카테고리에 대한 선호도가 드러났습니다.



### Training Language Models to Win Debates with Self-Play Improves Judge Accuracy (https://arxiv.org/abs/2409.16636)
Comments:
          48 pages, 12 figures; code at this https URL

- **What's New**: 이 연구에서는 언어 모델을 훈련시켜 토론에서 승리하도록 최적화한 결과, 평가자의 판단 정확도가 향상된다는 것을 처음으로 보였습니다. 이는 토론을 실질적인 확장 가능한 감독 방법으로 구현하고 검증하는 중요한 단계입니다.

- **Technical Details**: 연구는 QuALITY 데이터셋에서의 독해 질문에 대한 정보 비대칭 토론을 통해 진행되었습니다. 참여자 모델이 모든 주장을 두 차례 제시하고, 최종적으로 평가자가 어느 모델의 주장을 신뢰하는지를 결정합니다. 평가자의 정확성은 모델이 다른 모델과 싸울 때의 승률로 측정되었습니다.

- **Performance Highlights**: 토론 훈련 후 평가자의 정확도가 4% 증가했으며(p<10−6), 이러한 향상은 실제 감독 신호 없이도 이루어졌습니다. 반면 비대립 컨설팅 모델을 대상으로 한 실험에서는 모델 숙련도와 평가자 정확성 간의 긍정적인 관계가 발견되지 않았습니다.



### Claim-Guided Textual Backdoor Attack for Practical Applications (https://arxiv.org/abs/2409.16618)
Comments:
          Under Review

- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP) 및 대규모 언어 모델(Large Language Models, LLMs)의 발전은 새로운 보안 취약점을 드러냈습니다. 특히, 이번 연구에서는 입력 변조 없이 내재된 텍스트 클레임(claim)을 트리거로 활용하는 새로운 Claim-Guided Backdoor Attack (CGBA)을 도입합니다.

- **Technical Details**: CGBA는 다음의 세 가지 주요 단계로 구성됩니다: 1) 트레이닝 샘플에서 클레임 추출하기, 2) 유사한 클레임을 군집화하기, 3) 특정 군집을 선택하여 트리거로 설정하고 모델 훈련 중에 백도어를 주입하여 목표 클레임에서 잘못된 결정을 유도합니다. 이 과정은 대조적 손실(contrastive losses), 클레임 거리(claim distance), 다중 작업 손실(multi-tasking losses)을 사용합니다.

- **Performance Highlights**: CGBA는 다양한 데이터셋과 모델에서 실험을 통해 이전 방법들보다 높은 공격 성공률을 보이며, 깨끗한 데이터 정확도에 미치는 영향은 최소화되었습니다. 또한 기존 방어 방법에 대한 스텔스성(stealthiness)을 평가한 결과, perturbation 기반 방법에 대해 높은 저항성을 나타냈습니다.



### Evaluating and Enhancing Large Language Models for Novelty Assessment in Scholarly Publications (https://arxiv.org/abs/2409.16605)
Comments:
          under review

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 학술 논문에서의 창의성 및 참신성을 평가하기 위한 새로운 벤치마크인 SchNovel을 도입하였습니다. 이 벤치마크는 arXiv 데이터 세트에서 선택된 15,000 쌍의 논문으로 구성되어 있으며, 각 쌍의 최근 발표된 논문이 더 참신하다고 가정합니다. 또한, RAG-Novelty라는 새로운 방법을 제안하여 LLM이 논문의 참신성을 평가할 때 유사한 논문의 검색을 활용합니다.

- **Technical Details**: SchNovel 벤치마크는 2~10년 차이가 나는 논문 쌍을 포함하며, 이는 특히 높은 수준의 리뷰 과정을 거치는 학술 논문에서 참신성을 평가하는 데 중요합니다. RAG-Novelty는 검색 기반 생성 방법으로, 더 참신한 논문일수록 최근 발표된 논문을 더 많이 검색할 것이라는 가정을 바탕으로 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 RAG-Novelty가 기존의 기준 모델보다 논문의 참신성을 평가하는 데 더 뛰어난 성능을 보인다는 것을 입증했습니다. 이 연구는 LLM의 논문 참신성 평가 능력을 깊이 있게 탐구하고, 다양한 카테고리와 출판 연도 간의 변화를 평가하여 LLM의 성능 향상에 기여하였습니다.



### Overview of the First Shared Task on Clinical Text Generation: RRG24 and "Discharge Me!" (https://arxiv.org/abs/2409.16603)
Comments:
          ACL Proceedings. BioNLP workshop

- **What's New**: 최근 자연어 생성(Natural Language Generation, NLG) 기술이 의료 분야에서 큰 변화를 일으키고 있습니다. 특히, 임상 보고서의 각 섹션을 자동 생성하여 의사들의 업무 부담을 줄이고 병원 문서화 프로세스를 간소화하는 다양한 애플리케이션을 도입하였습니다.

- **Technical Details**: 이번 연구에서는 두 가지 하위 작업으로 이루어진 공유 과제를 제시합니다: (1) Radiology Report Generation (RRG24)와 (2) Discharge Summary Generation ('Discharge Me!')입니다. RRG24는 흉부 X-레이를 기반으로 라디오 로지 보고서의 'Findings' 및 'Impression' 섹션을 생성하는 과제이고, 'Discharge Me!'는 응급실에 입원한 환자를 위한 퇴원 요약의 'Brief Hospital Course' 및 'Discharge Instructions' 섹션을 생성하는 과제입니다. 이 두 작업 모두 클리니션의 반복적인 업무를 덜고 번아웃(Burnout)을 줄이는 것을 목표로 합니다.

- **Performance Highlights**: RRG24 과제에서 201개의 제출물이 8개 팀에서 왔으며, 'Discharge Me!' 과제에서는 211개의 제출물이 16개 팀에서 도착했습니다. 이 작업들은 최근의 발전을 공통 데이터 분할 및 평가 구현을 통해 벤치마킹하는 것을 목표로 하고 있습니다.



### Disentangling Questions from Query Generation for Task-Adaptive Retrieva (https://arxiv.org/abs/2409.16570)
- **What's New**: 본 논문에서는 정보 검색 (Information Retrieval) 문제를 다루며, 기존 쿼리 생성기가 일반적인 검색 의도를 수용하지 못하는 문제를 해결하기 위한 새로운 쿼리 생성기 EGG (Efficient Generalized Generator)를 제안합니다. EGG는 메타 프롬프트 (meta-prompt)를 활용하여 고급 의도를 작업 적응 쿼리로 변환합니다.

- **Technical Details**: EGG는 다양한 검색 의도를 통합하기 위해 메타 프롬프트와 검색기 피드백 (retriever feedback)을 활용하며, 137B LLM (Large Language Model)을 포함한 몇 가지 모델 크기를 제공합니다. 특히, FLAN 137B를 사용하여 few-shot 쿼리 생성에 최적화된 EGG-FLAN과, 적절한 컨텍스트를 사용할 수 있는 EGG-Llama 두 가지 버전을 도입합니다.

- **Performance Highlights**: EGG는 기반 모델 (baseline) 및 기존 모델들을 초월하여 네 가지 작업에서 우수한 성능을 보였으며, 이전의 최첨단 모델보다 47배 더 작은 쿼리 생성기를 사용하면서도 가장 높은 전반적인 성능을 달성했습니다.



### Understanding the Cognitive Complexity in Language Elicited by Product Images (https://arxiv.org/abs/2409.16521)
- **What's New**: 이 논문은 제품 이미지가 소비자에 의해 보고된 다양한 언어적 특징을 유도하는 방식을 다루고 있습니다. 특히, 인간의 언어가 보여주는 인지적 복잡성을 측정하고 검증하는 접근법을 제안하며, 대형 언어 모델(LLM)을 통해 모의된 응답자의 인지 과정을 이해할 수 있는 도구를 제공합니다.

- **Technical Details**: 연구에서는 인간이 생성한 언어의 인지적 복잡성이 제품 이미지에 의해 어떻게 유도되는지를 분석합니다. 이를 위해 14개 카테고리의 4,000개 이상의 제품 이미지와 45,609개의 인간 생성 텍스트 레이블 및 복잡성 평가를 포함한 대규모 데이터셋을 소개합니다. 인지적 복잡성을 측정하기 위해 여러 자연어 모델을 활용하여 인간의 평가와 근접한 결과를 도출합니다.

- **Performance Highlights**: 인간이 평가한 인지적 복잡성은 특정 제품 이미지에 대해 두 사람이 매우 다른 기억을 설명하더라도 높은 복잡성을 가질 수 있음을 보여줍니다. 해당 연구는 인지 복잡성이 선택 예측을 개선하고, 인간과 LLM 간의 생성된 언어의 인지적 복잡성의 분포 차이를 분석하여 데이터의 품질을 평가할 수 있는 가능성을 제시합니다.



### Exploring Knowledge Tracing in Tutor-Student Dialogues (https://arxiv.org/abs/2409.16490)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 인공지능(AI) 기반의 튜터링 챗봇이 개발되었습니다. 본 연구에서는 학생과 튜터 간의 대화에서 학생 행동을 모델링하는 새로운 접근을 제안합니다. 대화 회차(turn)에 대한 분석을 통해 학생의 지식 수준을 추적하고 오해를 파악할 수 있는 가능성을 제시합니다.

- **Technical Details**: 본 연구는 두 가지 주요 단계를 통해 대화 지식 추적(dialogue knowledge tracing, KT)을 수행합니다: i) 사전 학습된 LLM인 GPT-4o를 사용하여 대화 데이터를 주석 처리하고, ii) 주석이 달린 데이터를 이용해 KT 모델을 훈련합니다. 제안하는 LLMKT 방법은 Llama 3 모델을 활용하여 KT 목표를 위해 미세 조정하고, 기존의 DKT 방법의 임베딩을 시맨틱 텍스트 임베딩으로 대체하는 DKT-Sem을 소개합니다.

- **Performance Highlights**: LLMKT 방법은 두 가지 튜터-학생 수학 대화 데이터셋에서 기존 KT 방법들보다 크게 향상된 성능을 보여주었으며, 이 방법들은 적은 훈련 데이터로도 효과적일 수 있음을 입증했습니다. 또한 GPT-4o의 대화 주석이 전문가의 평가에 따라 정확하다는 점을 보여줘 향후 연구 방향성을 제안합니다.



### Spelling Correction through Rewriting of Non-Autoregressive ASR Lattices (https://arxiv.org/abs/2409.16469)
Comments:
          8 pages, 7 figures

- **What's New**: 이번 연구는 Transformer 기반의 CTC 모델로 생성된 단어 조각(wordpiece) 격자(lattice)를 재작성하기 위한 유한 상태 변환기(Finite-State Transducer, FST) 기법을 소개합니다. 이 알고리즘은 재훈련 없이 단어 조각에서 음소(phoneme)로 직접 변환하는 것을 가능하게 합니다.

- **Technical Details**: 이 논문에서는 비자기회귀(non-autoregressive) CTC 기법으로 생성된 격자의 조정이 어렵다는 한계를 극복하기 위한 방법을 제시합니다. 특히, 비자기회귀 ASR에서는 이전 또는 이후 토큰과 조건적으로 독립적인 토큰들이 발행되어, '소시지 격자' topology의 문제를 해결해야 합니다. FST 기법을 통해 기존 격자의 풍부한 신호를 활용하여 보다 개선된 성능을 달성할 수 있습니다.

- **Performance Highlights**: 연구는 문맥적으로 관련된 개체들에 대한 테스트 세트에서 문장 오류율(Sentence Error Rate, SER)을 최대 15.2% 감소시켰습니다.



### Strategies for Improving NL-to-FOL Translation with LLMs: Data Generation, Incremental Fine-Tuning, and Verification (https://arxiv.org/abs/2409.16461)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)이 자연어(NL)에서 1차 논리(FOL)로의 번역 과정에서 발생하는 오류를 체계적으로 분석하고 이를 개선하기 위한 여러 방법을 제시하고 있습니다. 특히 ProofFOL이라는 고품질의 FOL 주석이 달린 데이터셋을 생성해 소형 모델들도 성능을 향상시킬 수 있도록 합니다.

- **Technical Details**: 이 연구는 ProofWriter 데이터셋을 기반으로 GPT-4o를 사용하여 FOL 번역에 관한 대규모 데이터셋인 ProofFOL을 생성합니다. ProofFOL은 104241 예시의 (premises, conclusion) 쌍과 해당하는 FOL 번역을 포함하고 있으며, 이를 통해 소형 모델인 LLaMA-2 13B 및 Mistral 7B가 LLaMA-2 70B 모델보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: ProofFOL을 사용한 LLaMA-2 및 Mistral 모델은 ProofWriter와 ProntoQA 데이터셋에서 최첨단 성능을 기록했으며, FOLIO 데이터셋에서는 41% 및 17%의 성능 향상을 달성하였습니다. 이 연구는 데이터 부족 문제 해결을 위한 점진적 훈련 방법과 새로운 검증 메커니즘을 제안합니다.



### FMDLlama: Financial Misinformation Detection based on Large Language Models (https://arxiv.org/abs/2409.16452)
Comments:
          work in progress

- **What's New**: 이 논문은 금융 분야에서의 허위 정보 탐지(FMD)를 위한 최초의 오픈소스 LLM인 FMDLlama를 제안합니다. 이는 Llama3.1에 대한 지침 데이터로 파인튜닝하여 개발하였으며, 첫 번째 다중 작업 FMD 지침 데이터 세트(FMDID)와 FMD 능력을 평가하기 위한 종합적인 벤치마크(FMD-B)를 함께 제공합니다.

- **Technical Details**: FMDLLMs는 허위 정보 탐지 작업에 최적화된 최초의 오픈소스 LLM으로, 다양하고 복잡한 금융 허위 정보 탐지 작업을 수행할 수 있습니다. 이 모델은 지침 조정 지침을 수립하고, 다중 금융 허위 정보 탐지 작업을 다루며, 특정 보고서 클래스의 성능을 극대화하기 위해 모든 작업 데이터 세트를 결합합니다.

- **Performance Highlights**: FMD-B 벤치마크에서 FMDLLMs는 다른 모든 오픈소스 LLM 및 ChatGPT보다 뛰어난 성능을 기록하며, 최신 기술(SOTA) 성능을 달성하였습니다. 이로 인해 LLM의 금융 허위 정보 검증 능력을 평가하는 데 큰 기여를 하였습니다.



### A Comprehensive Survey of Bias in LLMs: Current Landscape and Future Directions (https://arxiv.org/abs/2409.16430)
Comments:
          2 Tables, 1 Figure

- **What's New**: 이 논문은 Large Language Models(LLMs)에서의 편향(bias)에 대한 종합적인 조사(survey)를 제공하며, 다양한 유형, 출처, 영향 및 완화 전략을 체계적으로 분류하고 있습니다.

- **Technical Details**: LLMs의 편향을 여러 차원으로 분류한 후, 현재 연구 결과를 종합하고 실제 응용에서의 편향의 함의를 논의합니다. 또한 기존의 편향 완화(bias mitigation) 기법을 비판적으로 평가하고 LLMs의 공정성(fairness) 및 형평성(equity)을 향상시키기 위한 미래 연구 방향을 제시합니다.

- **Performance Highlights**: 이 조사는 연구자, 실무자 및 정책 입안자들에게 LLMs의 편향 문제를 해결하고 이해하는 데 기초 자료로 활용될 수 있는 중요한 리소스를 제공합니다.



### RISCORE: Enhancing In-Context Riddle Solving in Language Models through Context-Reconstructed Example Augmentation (https://arxiv.org/abs/2409.16383)
- **What's New**: 본 논문에서는 LLM의 수수께끼 해결 능력을 검토하고, RISCORE라는 새로운 자동화된 프롬프트 기법을 소개하여 수수께끼의 해결 성능을 향상시키고자 합니다.

- **Technical Details**: RISCORE(RIddle Solving with COntext REcontruciton)는 수수께끼의 문맥을 재구성하여 몇 가지 샘플을 생성하는 방법으로, 언어 모델이 수수께끼 해결 시 기존의 예시와 결합하여 성능을 극대화합니다.

- **Performance Highlights**: RISCORE는 수직 사고(vertical thinking)와 수평 사고(lateral thinking) 작업에서 언어 모델의 성능을 현저히 향상시키며, 기존 프롬프트 기술보다 뛰어난 성능을 보였습니다.



### Do the Right Thing, Just Debias! Multi-Category Bias Mitigation Using LLMs (https://arxiv.org/abs/2409.16371)
Comments:
          17 pages, 5 Figures

- **What's New**: 이 논문에서는 언어에서의 편향 완화(bias mitigation) 모델을 구축하는 과제를 다룹니다. 기존 데이터셋의 한계를 인식하고, 9개의 사회적 편향 범주를 포함한 1507개의 세심하게 선별된 문장 쌍을 포함하는 새로운 데이터셋 ANUBIS를 소개합니다.

- **Technical Details**: ANUBIS 데이터셋은 인종, 성별, 성 정체성, 성적 지향, 종교, 나이, 국적, 장애, 외모, 사회경제적 지위 등 다양한 편향 범주를 포괄합니다. 우리는 최신 모델 T5를 사용하여 Supervised Fine-Tuning (SFT), 강화 학습(Reinforcement Learning; PPO, DPO 등), In-Context Learning (ICL)을 통해 효과적인 편향 완화 기법을 평가했습니다.

- **Performance Highlights**: 연구 결과, ANUBIS 데이터셋과 기존 데이터셋(WIKIBIAS)에서 훈련된 모델의 성능을 비교하였으며, 모델의 지속 가능성과 환경적 영향을 고려한 평가도 진행하였습니다. 이는 더 공정한 AI 시스템 개발과 사회적 영향을 고려한 기술 발전에 기여할 것으로 기대됩니다.



### Exploring the traditional NMT model and Large Language Model for chat translation (https://arxiv.org/abs/2409.16331)
Comments:
          7 pages, 6 Tables, WMT24

- **What's New**: 이 논문에서는 WMT24 채팅 번역 공유 과제에서 영어와 독일어 간의 이중 번역(en-de) 작업을 위한 Huawei Translation Services Center(HW-TSC)의 제출 사례를 다룹니다. 연구에서는 채팅 데이터를 이용한 모델의 파인튜닝(fine-tuning)과 함께 Minimum Bayesian Risk (MBR) 디코딩 및 자기 학습(self-training) 등의 다양한 전략을 탐색했습니다.

- **Technical Details**: 본 논문에서는 Transformer-Big 아키텍처를 기반으로 한 모델을 사용하며, 기존 NMT 모델 외에 대형 언어 모델(LLM)의 활용을 통해 번역 작업의 새로운 패러다임을 제시합니다. Minimum Bayesian Risk (MBR) 디코딩 기법은 여러 후보 중에서 최소 예상 오류를 가진 출력을 선택하는 방식입니다. 자기 학습 기법(self-training)과 역번역(back-translation)을 통해 번역 품질을 향상시키는 방법을 설명합니다.

- **Performance Highlights**: 결과적으로 MBR 자기 학습 방법이 가장 우수한 성능 향상을 보여주었고, 기계 번역 품질의 측면에서 확실한 개선이 있었습니다. 그러나 LLM의 출력 comet 메트릭은 NMT 모델의 최적 결과를 초과하지 못했습니다.



### DeepScore: A Comprehensive Approach to Measuring Quality in AI-Generated Clinical Documentation (https://arxiv.org/abs/2409.16307)
Comments:
          9 pages, 5 figures, 6 tables

- **What's New**: 이 논문은 DeepScribe의 의료 문서 품질 평가 방법론에 대해 설명합니다. 특히, 다양한 지표와 종합 점수인 'DeepScore'를 통한 품질 및 정확성을 측정하는 방법에 초점을 맞춥니다.

- **Technical Details**: DeepScribe는 'Stat Rates'라는 시스템을 통해 AI가 생성한 의료 노트의 품질을 평가하고, 즉각적인 수정과 알고리즘의 발전 방향을 제시합니다. Major Defect-Free Rate (MDFR), Critical Defect-Free Rate (CDFR), Captured Entity Rate (CER), Accurate Entity Rate (AER), Minimally-Edited Note Rate (MNR), Medical Word Hit Rate (MWHR)와 같은 다양한 지표를 활용하여, AI 문서의 정확성과 사용자 수용성을 분석합니다.

- **Performance Highlights**: DeepScore는 앞서 언급한 모든 지표를 평균내어 계산하여 생성됩니다. 이를 통해 의료 문서 품질에 대한 종합적인 평가를 제공하며, 지속적인 개선을 위한 지침 역할을 합니다.



### Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models (https://arxiv.org/abs/2409.17146)
- **What's New**: Molmo는 현재 공개된 VLM 중에서 가장 최신의 성능을 자랑하며, 독점적인 데이터가 아닌 새롭게 수집된 고품질 데이터로 구성된 이미지를 설명하는 고급 캡션 데이터셋을 기반으로 하고 있습니다. 이 연구는 VLM을 처음부터 구축하는 데 필요한 기본 지식을 제공합니다.

- **Technical Details**: Molmo 모델은 기존의 비전 인코더와 언어 모델을 결합하여 만들어졌습니다. 이 과정에서는 이미지로부터 자세하고 질 높은 설명을 생성하는 새로운 데이터셋인 PixMo를 사용하고, 60~90초 동안 음성으로 설명하도록 요구하여 다양한 내용을 포괄하도록 했습니다. 모델 훈련은 다단계 과정이 아닌, 간단한 훈련 파이프라인을 통해 진행되었습니다.

- **Performance Highlights**: Molmo-72B 모델은 학술 벤치마크에서 최고의 점수를 기록했으며, GPT-4o와 비교했을 때 사용자 선호도 순위에서 두 번째에 올랐습니다. Molmo-1B 모델은 효율적인 성능을 보이며 GPT-4V와 근접한 결과를 보여주었고, 전체적으로 많은 상업적 시스템을 능가하는 성능을 발휘했습니다.



### Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning? (https://arxiv.org/abs/2409.17080)
Comments:
          13 pages, 4 figures. Code released at this https URL

- **What's New**: 본 연구에서는 Spatial Visual Ambiguity Tasks (SVAT)라는 새로운 벤치마크를 제안하여 대형 비전-언어 모델(VLM)이 시각적 데모를 통해 새로운 비주얼-스페이셜 개념을 학습할 수 있는지를 평가합니다.

- **Technical Details**: SVAT는 여러 난이도 수준의 분류 작업으로 구성되어 있으며, 목표는 주어진 모호한 텍스트와 시각적 데모를 기반으로 이미지 내에서 관심 있는 객체의 정확한 위치를 학습하는 것입니다. VLM은 이 과제를 위해 Text-Query와 Image-Based Input을 사용하여 정확한 경계 결정을 해야 합니다.

- **Performance Highlights**: Zero-shot 설정에서 VLM은 SVAT 작업에서 실패하고, 단순한 파인튜닝만으로는 5.8%-27.3%의 성능 향상이 가능합니다. 그러나 커리큘럼 학습(Curriculum Learning)을 통해, VLM은 SVAT의 가장 어려운 작업에서 14.2%에서 34.2%의 정확도 개선을 이룰 수 있습니다.



### Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia (https://arxiv.org/abs/2409.17054)
- **What's New**: 이 연구는 동남아시아 지역에서의 의사-환자 상호 작용의 효율성을 개선하기 위해 지역화된 대형 언어 모델(LLM)을 활용한 자동적 환자 기록 전사 및 요약 시스템을 제안합니다.

- **Technical Details**: 이 시스템은 OpenAI의 Whisper 모델을 사용하여 실시간으로 환자-의사 대화를 전사하고, GPT-3를 통해 요약된 정보를 ePuskesmas 전자 건강 기록 형식으로 변환합니다. 주요 과정은 네 가지로 나누어집니다: 대화 녹음, 실시간 전사, 의료 데이터 요약, 자동 ePuskesmas 파서.

- **Performance Highlights**: 이 솔루션을 통해 의사들은 더욱 신속하게 환자 정보를 정리할 수 있으며, 기록의 질도 향상되어 향후 환자 방문을 위한 정보가 더욱 자세하고 통찰력 있게 변모합니다. 이는 인도네시아의 과중한 시설과 의료 제공자의 행정 부담을 해소하는 중요한 진전을 나타냅니다.



### Counterfactual Token Generation in Large Language Models (https://arxiv.org/abs/2409.17027)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)이 출력의 대안에 대해 반사실적 추론(counterfactual reasoning)을 수행할 수 있도록 하는 causal 모델을 개발했습니다. 이는 무작위 샘플링(sampling) 과정에 Gumbel-Max 구조적 인과 모델을 적용하여 구현됩니다.

- **Technical Details**: 제안한 모델은 Llama 3 8B-instruct에서 구현되었으며, 기존의 노드 상태를 유지하지 않는 상태 비저장(stateless) LLM에서 출발하여 반사실적 토큰 생성(counterfactual token generation)을 가능하게 합니다. 이를 통해 모델은 입력된 프롬프트에 따라 자동 회귀(auto-regressive) 생성 방식을 사용하여 가능한 대안을 탐색할 수 있습니다.

- **Performance Highlights**: 반사실적 토큰 생성의 질과 양을 분석한 결과, 모델이 생성한 결과는 기존 출력과 높은 유사성을 보여주었으며, 바이어스 탐지에 대한 유용성도 입증되었습니다. 이 연구는 LLM이 어떻게 세상 모델을 구성하는지에 대한 통찰력을 제공합니다.



### Models Can and Should Embrace the Communicative Nature of Human-Generated Math (https://arxiv.org/abs/2409.17005)
- **What's New**: 이 논문은 수학이 사람에 의해 사람을 위해 구성된다는 관점을 제안하며, 수학 데이터가 단순한 기호적 표현을 넘어선 풍부한 의사소통 의도를 반영한다고 주장합니다. 이 연구는 언어 모델(Language Models)이 수학적 심볼을 처리할 때 인간과 유사한 방식을 채택하고 있음을 실험을 통해 증명합니다.

- **Technical Details**: 연구에서는 두 가지 사례 연구를 통해 언어 모델의 수학적 문제 해결 방식의 비대칭성(asymmetry)과 정렬(ordering)에 대한 선호도를 조사하였습니다. 첫 번째 실험에서는 동일한 기본 방정식에 대해 다양한 형태의 언어 문제를 생성하는 능력을 평가했습니다. 두 번째로, 언어 모델이 수학적 증명의 배열 방식이 자연스러운 방식일 때 더 선호한다는 사실을 발견했습니다. 이는 AI 시스템이 인간이 생성한 수학의 의사소통 의도를 학습하고 표현하는 데 기여할 수 있음을 나타냅니다.

- **Performance Highlights**: 언어 모델들은 같은 방정식에 대해 다르게 해석하며, 문제를 구성할 때 비대칭성을 인식합니다. 또한, 정당한 수학적 증명들의 배열 순서에 대해 인간처럼 자연적인 방식의 선호를 보였습니다. 이러한 결과는 수학적 의사소통에서 맥락을 완전히 무시하지 말고 이를 고려해야 한다는 점을 강조합니다.



### AXCEL: Automated eXplainable Consistency Evaluation using LLMs (https://arxiv.org/abs/2409.16984)
- **What's New**: 이 논문에서는 LLM을 활용하여 텍스트 응답의 일관성을 평가하는 새로운 접근 방식인 AXCEL(Automated eXplainable Consistency Evaluation using LLMs)을 소개합니다. AXCEL은 프롬프트 기반으로 작동하며, 일관성 점수에 대한 설명을 제공하여 사용자가 추론할 수 있도록 돕습니다.

- **Technical Details**: AXCEL는 Chain of Thought (CoT)와 few shot prompting 기술을 사용하여 텍스트의 일관성을 측정합니다. 기존의 메트릭에 비해 AXCEL은 설명 가능성을 제공하며, 특정 태스크에 맞지 않아도 여러 태스크에 적용할 수 있는 일반화 가능성을 가지고 있습니다. AXCEL은 요약, 자유 텍스트 생성, 데이터-텍스트 변환의 세 가지 태스크에서 실험되었습니다.

- **Performance Highlights**: AXCEL은 요약에서는 8.7%, 자유 텍스트 생성에서는 6.2%, 데이터-텍스트 변환 태스크에서는 29.4% 향상된 성능을 보이며, 기존의 non-prompt 및 prompt 기반 최신 메트릭을 초월했습니다. 또한 AXCEL은 오픈 소스 LLM을 사용하더라도 강력한 성능을 보여줍니다.



### Semi-Supervised Cognitive State Classification from Speech with Multi-View Pseudo-Labeling (https://arxiv.org/abs/2409.16937)
- **What's New**: 본 연구는 인지 상태 분류와 같은 주관적 평가가 많이 필요한 음성 분류 작업에서 레이블이 없는 데이터의 문제를 해결하기 위해 새로운 반지도 학습(Semi-Supervised Learning, SSL) 프레임워크를 제안합니다. 특히, 이 프레임워크는 음향과 언어적 특성을 활용한 다중 뷰(pseudo-labeling) 방법을 도입하여 분류 모델 훈련에 가장 확신이 높은 데이터를 선택합니다.

- **Technical Details**: 제안된 SSL 프레임워크는 두 가지 경로로 구성됩니다: 1) 음향 경로, 해당 경로에서는 다양한 오디오 임베딩을 이용해 레이블이 있는 데이터와 레이블이 없는 데이터를 비교하여 유사성을 판단합니다. 2) 언어 경로, 여기서는 대형 언어 모델(Large Language Models, LLMs)을 사용하여 ASR(Automatic Speech Recognition) 전사로부터 예측 레이블을 도출합니다. 프레셰 오디오 거리(Frechet Audio Distance, FAD)를 사용해 레이블이 없는 데이터의 음향 유사성을 측정하고, 이를 통해 고신뢰 데이터와 저신뢰 데이터를 구분합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 라벨이 있는 데이터의 30%만 사용하여도 완전 감독 학습(fully supervised learning)과 유사한 성능을 달성하였으며, 두 개의 기준선보다 훨씬 우수한 성과를 나타냈습니다.



### Cross-lingual Speech Emotion Recognition: Humans vs. Self-Supervised Models (https://arxiv.org/abs/2409.16920)
- **What's New**: 본 연구는 Self-Supervised Learning (SSL) 모델을 통한 Speech Emotion Recognition (SER)에서 인간 성능과의 비교 분석을 수행했습니다. 특히, 단일 언어(monolingual), 교차 언어(cross-lingual) 및 전이 학습(transfer learning) 맥락에서 매개변수 효율적인 미세 조정(Parameter-Efficient Fine-Tuning) 전략을 탐구하고, 방언이 교차 언어 SER에 미치는 영향을 조사하였습니다.

- **Technical Details**: 이 연구는 Wav2vec 2.0 (W2V2) 및 WavLM 등 강력한 사전 훈련(pre-trained) 모델을 사용하여 SSL 기반의 SER 성능을 비교하였습니다. 다양한 PEFT 전략을 통해 모델의 초기 파라미터를 수정하며, 중간층의 음성 표현이 높은 성능을 발휘하는 것을 확인했습니다. 교차 언어 SER에서 모델은 적절한 지식 전이를 통해 목표 언어에 적응할 수 있음을 보여주었으며, 특정 방언의 특성을 고려한 평가를 수행했습니다.

- **Performance Highlights**: 모델과 인간 모두 다른 감정에 대해 뚜렷한 행동 차이를 보였으며, 모델은 네이티브(Native) 화자와 유사한 성능을 기록했습니다. 방언은 인간의 감정 인식에 상당한 영향을 미치는 것으로 나타났으며, 감정 인식 정확도를 조정하기 위한 다양한 PEFT 전략이 효과적이었음을 입증했습니다.



### A Roadmap for Embodied and Social Grounding in LLMs (https://arxiv.org/abs/2409.16900)
Comments:
          Accepted Version of a conference paper presented at Robophilosophy Conference 2024

- **What's New**: 이번 연구는 로봇 시스템에 대규모 언어 모델(LLMs)을 통합하여 의사소통 뿐만 아니라 다중 모달 입력 처리, 고급 추론 및 계획 생성을 통한 변혁적인 패러다임을 제시합니다. LLM의 지식을 경험적 세계와 연결하기 위한 새로운 접근 방식을 탐구합니다.

- **Technical Details**: 이 논문은 LLMs의 기초 모델에 대한 추상 지식을 물리적 현실과 일치시키는 'grounding'(근거화) 과정을 다룹니다. 이 과정에서는 LLMs가 환경에 대한 이해도를 높이고, 객체의 물리적 속성을 통해 추론 능력을 향상시키도록 돕는 기술적 접근이 포함됩니다.

- **Performance Highlights**: 최근의 연구들은 LLMs의 텍스트 기반 작업에서의 효과가 신체적 제어까지 확장되었음을 보여주며, 이는 로봇이 인간과의 협력적 환경에서 더 뛰어난 성능을 발휘할 수 있게 합니다. 그러나 실제 로봇 작업에서 물리적 및 사회적 추 reasoning과 같은 기술들에 대한 LLM의 한계 또한 강조되었습니다.



### Robotic Backchanneling in Online Conversation Facilitation: A Cross-Generational Study (https://arxiv.org/abs/2409.16899)
Comments:
          Published at Proceedings of the 2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2023)

- **What's New**: 일본은 고령화 사회에 따른 문제, 특히 인지 저하 증가와 돌봄 인력 부족과 같은 도전에 직면하고 있습니다. 본 연구는 AI를 활용한 사회적 존재감을 가진 로봇이 노인과의 상호작용에서의 호환성을 평가하기 위한 첫걸음을 제시합니다.

- **Technical Details**: 이 연구에서는 인지 저하를 방지하기 위한 그룹 대화 프로토콜을 촉진하는 로봇을 평가하기 위한 사용자 연구를 실시했습니다. 로봇은 자연스러운 의사소통 방식인 backchannelling을 사용하도록 변경되어, 그룹 대화의 즐거움을 높이도록 설계되었습니다.

- **Performance Highlights**: 교차 세대 연구를 통해 젊은 성인들은 backchannelling 로봇을 비-backchannelling 로봇보다 더 친절하고 신뢰할 수 있으며 수용 가능하다고 인식했습니다. 또한, 로봇의 backchannelling이 노인 참가자들의 비언어적 backchanneling을 유도했습니다.



### The Role of Language Models in Modern Healthcare: A Comprehensive Review (https://arxiv.org/abs/2409.16860)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 의료 분야에 어떻게 적용되는지에 대한 체계적인 리뷰를 제공합니다. LLM의 발전 과정과 의료 적용에서의 강점뿐만 아니라 데이터 프라이버시, 편향, 윤리적 고려사항 등의 문제를 논의합니다.

- **Technical Details**: 대형 언어 모델(LLM)은 Transformer 아키텍처를 기반으로 하여 장거리 의존성을 효과적으로 캡처하는 능력을 가지고 있습니다. 모델은 일반적으로 방대한 텍스트 데이터셋으로 사전 학습(pre-training)된 후, 특정 작업에 맞춰 세부 조정(fine-tuning)됩니다. BioBERT, ClinicalBERT와 같은 의료 전용 모델이 개발되어 임상 언어의 독특한 도전을 해결하고 있습니다.

- **Performance Highlights**: LLM은 의료 데이터 분석, 질병 진단, 환자 관리 및 약물 발견과 같은 다양한 분야에서 사용되고 있습니다. 임상 의사결정 지원 및 의료 문서 요약 등의 임무에 대한 효과적인 증상이 입증되었습니다. 측정 기준으로는 MMLU, HumanEval과 같은 벤치마크가 사용되어 모델의 효과성을 평가합니다.



### Exposing Assumptions in AI Benchmarks through Cognitive Modelling (https://arxiv.org/abs/2409.16849)
Comments:
          11 pages, 2 figures

- **What's New**: 본 논문에서는 문화적 AI 벤치마크가 종종 측정된 구성 요소에 대한 암묵적인 가정에 의존하고 있다는 점을 지적하며, 이러한 가정들을 명시적인 인지 모델을 통해 드러내고자 합니다. 구조 방정식 모델(Structural Equation Models; SEM)을 활용하여, 누락된 데이터셋을 식별하고 연구 질문에 답할 수 있는 방법을 제시합니다.

- **Technical Details**: 세부적으로, 우리는 LLM(대형 언어 모델)의 ‘특성’에 관한 명시적 모델링을 통해 심리 측정(psychometrics)에서 영감을 받은 접근 방식을 확장하고 있습니다. 우리의 구조 방정식 모델은 언어 능력, 문화 지식, 정렬(alignment) 간의 관계를 분석합니다. 이 모델은 교차 언어 정렬 이전의 명확한 가정을 드러내어, 데이터셋 개발을 위한 방향을 제시합니다.

- **Performance Highlights**: 본 프레임워크는 기존의 벤치마크와 이론적 구성 간의 관계를 명확히 하며, 다양한 테스트가 잘 측정하는지를 평가할 수 있는 가능성을 열었습니다. 이는 LLM 특성에 대한 더 엄격하고 이론적으로 정당화된 이해를 위한 길을 제시하고, 결과적으로 생성적 AI 시스템의 포괄적이고 세밀한 평가를 촉진합니다.



### Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification (https://arxiv.org/abs/2409.16718)
Comments:
          EMNLP 2024 Main Conference

- **What's New**: 최근 Vision-Language Models (VLMs)에 대한 세밀한 조정이 이루어지면서, 클립 모델의 고유한 매개변수를 조정하는 것의 중요성을 재조명하였다. 이 연구에서는 모든 매개변수를 조정하는 대신 특정 매개변수만을 조정하는 CLIPFit 방법을 제안하였다.

- **Technical Details**: CLIPFit은 기존의 프롬프트 튜닝(prompt tuning) 및 어댑터 튜닝(adapter tuning) 방식과는 다르게, 추가적인 매개변수를 도입하지 않고 클립 모델의 특정 바이어스와 정규화 레이어만 조정하는 방법이다. 이로 인해 파라미터 수가 줄어들고, 성능이 향상된다.

- **Performance Highlights**: CLIPFit을 사용하여 zero-shot CLIP 대비 7.33%의 평균 조화 평균 정확도(harmonic mean accuracy) 개선을 달성하였으며, 이는 16-shot 설정에서 프롬프트 튜닝 및 어댑터 튜닝을 대체할 수 있는 유망한 옵션이다.



### Beyond Turing Test: Can GPT-4 Sway Experts' Decisions? (https://arxiv.org/abs/2409.16710)
- **What's New**: 이 논문은 LLM(대규모 언어 모델) 생성 텍스트가 독자의 반응을 기반으로 사람의 결정에 미치는 영향을 탐구합니다. 특히 아마추어와 전문가 독자의 반응 차이를 분석하며, GPT-4가 유도할 수 있는 설득력 있는 분석을 강조합니다.

- **Technical Details**: 실험은 ECTSum 데이터셋을 기반으로 설계되었으며, 2,425개의 ECC(어닝 컨퍼런스 콜) 기록을 바탕으로 합니다. GPT-4는 중립적 요약과 전문 분석 리포트를 생성하여 투자자에게 제공되었고, 두 단계에서 투자자의 결정을 유도했습니다. 실험은 아마추어와 전문가의 반응을 비교했습니다.

- **Performance Highlights**: 결과적으로 GPT-4가 생성한 분석은 아마추어의 결정에 더 큰 영향을 미치는 반면, 전문가는 전문가 작성 분석에 더 많은 영향을 받는 것으로 나타났습니다. 이는 LLM을 통한 분석 생성이 아마추어에게는 효과적일 수 있으나, 전문가 수준에는 미치지 못한다는 것을 보여줍니다.



### A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms (https://arxiv.org/abs/2409.16694)
Comments:
          Ruihao Gong leads the overall organization of the survey, with Yifu Ding and Jinyang Du contributing to Sections 2 and 3. Xingyu Zheng is responsible for authoring Section 4, while Chengtao Lv and Zining Wang collaborate on Section 5. Haotong Qin, Jinyang Guo, Michele Magno, and Xianglong Liu provide guidance during the whole process and assist in refining the final manuscript

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 위한 저비트 양자화(low-bit quantization) 방법에 대한 종합적인 조사를 제공합니다. 이는 메모리 사용량과 계산 요구 사항을 줄여주어 LLM의 실제 구현에서 중요한 과제를 해결하는 데 기여합니다.

- **Technical Details**: 저비트 양자화는 모델의 파라미터, 활성화(activations), 그리고 그래디언트의 비트 폭을 줄이는 프로세스로, 메모리 사용량과 계산 요구를 감소시킵니다. 이 논문에서는 저비트 LLM의 기초 원리, 시스템 구현, 알고리즘 전략을 다루고 있습니다. 새로운 저비트 데이터 형식과 양자화 세분화(granularity), 정적 또는 동적 양자화의 차이점 등이 소개됩니다.

- **Performance Highlights**: 저비트 양자화는 LLM의 훈련(training) 및 추론(inference)을 가속화하며, 정확도를 유지하면서도 모델을 저장하는 데 필요한 자원을 줄이는 데 효과적입니다. 이 연구에서는 새로운 연구 분야, 잠재적인 혁신, 그리고 새로운 기술이 LLM 양자화에 미치는 영향을 논의하며, LLM의 효율성 및 적합성을 향상시키기 위한 가치 있는 통찰력을 제공합니다.



### MSI-Agent: Incorporating Multi-Scale Insight into Embodied Agents for Superior Planning and Decision-Making (https://arxiv.org/abs/2409.16686)
- **What's New**: 이번 논문에서는 Multi-Scale Insight Agent (MSI-Agent)를 소개합니다. 이 에이전트는 장기 기억(Long-term memory)을 개선하여 LLMs의 계획 및 의사결정 능력을 높이기 위해 다양한 스케일에서 통찰(insight)을 효과적으로 요약하고 활용하는 방식으로 설계되었습니다.

- **Technical Details**: MSI-Agent는 경험 선택기(experience selector), 통찰 생성기(insight generator), 통찰 선택기(insight selector)의 세 가지 주요 구성 요소를 통해 작동합니다. 이 세 부분으로 이루어진 파이프라인(pipeline)을 활용하여, MSI는 작업에 특화된(task-specific) 고수준의 통찰을 생성하고 이를 데이터베이스에 저장한 후, 의사결정을 위해 관련 통찰을 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, MSI는 GPT3.5에 기반한 다른 통찰 전략보다 우수한 성능을 보였습니다. 또한, 씨앗 경험(seed experience)과 통찰을 선택하는 전략을 탐구하며, LLM에 더 유용하고 관련성 있는 통찰을 제공하여 더 나은 의사결정을 지원하는 데 초점을 맞추고 있습니다. MSI는 도메인 전환(domain-shifting) 시나리오에서 더 나은 강건성을 보여주는 것으로 관찰되고 있습니다.



### Emotional Dimension Control in Language Model-Based Text-to-Speech: Spanning a Broad Spectrum of Human Emotions (https://arxiv.org/abs/2409.16681)
Comments:
          submitted to ICASSP 2025

- **What's New**: 이 논문은 감정 텍스트 음성 변환(TTS) 시스템의 새로운 프레임워크를 제안하여, 감정 음성 데이터 없이도 다양한 감정 스타일을 합성할 수 있는 기능을 제공합니다. 이 시스템은 쾌감(pleasure), 자극(arousal), 지배(dominance)라는 감정 차원을 제어할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 TTS 프레임워크는 카테고리 레이블만을 사용해 감정 속성 예측기를 학습하고, 자가 감독 학습(self-supervised learning) 기능을 활용하여 텍스트 입력을 음소 토큰으로 변환합니다. 감정 차원 벡터를 통해 미세한 음향 세부사항의 병렬 예측을 조정합니다. 평가를 위해 LibriTTS 데이터셋을 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과는 제안된 모델이 자연스러움과 다양한 감정 스타일을 효과적으로 합성할 수 있으며, 기존 모델을 초월하는 성능을 보여 줍니다. 또한, 감정 차원 제어를 통해 합성된 음성이 개선된 자연스러움과 발화 일관성을 나타냅니다.



### Speech Recognition Rescoring with Large Speech-Text Foundation Models (https://arxiv.org/abs/2409.16654)
- **What's New**: 본 연구는 LLMs(대규모 언어 모델)를 활용한 자동 음성 인식(ASR) 시스템의 두 번째 패스 재점수를 위한 새로운 접근 방식을 제안합니다. 특히, 음성과 텍스트 양식 모두에서 대규모 데이터를 활용하는 멀티모달 LLMs를 사용하여 기존 ASR의 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 HuBERT(음성 인코더)를 사용하여 음성 표현을 추출하고, 이를 k-means 군집화를 통해 불연속 음성 토큰으로 양자화합니다. 이후, 텍스트 토큰은 문장 조각 모델을 사용하여 유도됩니다. 텍스트 및 음성 토큰에 대한 인과 언어 모델링을 위해 디코더 전용 트랜스포머 구조를 적용합니다. 두 번째 패스에서는 ASR로부터의 n-best 가설을 사용하여 모델로부터의 가능도 점수를 계산하고 이를 바탕으로 재순위를 결정합니다.

- **Performance Highlights**: 실험 결과, Whisper 대규모 ASR에 대해 최대 20%의 상대적 개선을 보였으며, 텍스트 전용 LLM에 대해서도 최대 15%의 상대적 개선을 달성했습니다. 이러한 결과는 멀티모달 LLMs가 음성과 텍스트 정보를 활용하여 재점수 과정을 개선할 수 있음을 보여줍니다.



### Enabling Auditory Large Language Models for Automatic Speech Quality Evaluation (https://arxiv.org/abs/2409.16644)
Comments:
          submitted to ICASSP 2025

- **What's New**: 최근 도입된 auditory large language models (LLMs)를 활용하여 자동 음성 품질 평가를 수행하는 방안을 제안합니다. 이 모델은 MOS, SIM 및 A/B 테스트 결과 등을 예측할 수 있도록 조정됩니다.

- **Technical Details**: 이 연구에서는 SALMONN, Qwen-Audio 및 Qwen2-Audio와 같은 오픈소스 auditory LLM을 활용하여 NISQA, BVCC, SOMOS 및 VoxSim 데이터셋을 포함한 다양한 음성 품질 평가 작업을 수행합니다. fine-tuning된 auditory LLM은 자연 언어로 평가할 수 있는 추가적인 능력을 갖추게 됩니다.

- **Performance Highlights**: 실험 결과, auditory LLM은 state-of-the-art task-specific 작은 모델들과 비교하여 MOS 및 SIM 예측에서 경쟁력 있는 성능을 보여주었으며, A/B 테스트와 자연 언어 설명에서도 유망한 결과를 기록했습니다.



### A Unified Hallucination Mitigation Framework for Large Vision-Language Models (https://arxiv.org/abs/2409.16494)
Comments:
          Accepted by TMLR

- **What's New**: 본 논문에서는 다양한 형태의 환각(hallucination) 문제를 해결하기 위해, 쿼리(query)를 분류하고 이를 기반으로 다양한 환각 완화(mitigation) 과정을 수행하는 통합 프레임워크인 Dentist를 제안합니다. 이를 통해 환각의 종류에 따라 각기 다른 접근방법을 적용할 수 있습니다.

- **Technical Details**: Dentist 프레임워크는 먼저 쿼리를 인식(perception)과 추론(reasoning)으로 분류하고, 각 쿼리 유형에 맞는 처리 방법을 사용합니다. 구체적으로, 감지 쿼리에 대한 생성 결과는 부차적 질문(sub-questions)을 통해 검증되며, 추론 쿼리의 경우 체인 오브 생각(Chain-of-Thought, CoT)을 활용해 검증됩니다. 이 검증 루프는 정밀도 향상을 위해 수차례 반복됩니다.

- **Performance Highlights**: MMbench에서 InstructBLIP, LLaVA, VisualGLM과 같은 기존 기법에 비해 이미지 품질 관점에서 13.44%, 10.2%, 15.8%의 정확도 향상을 달성했습니다. 또한, 우리의 방법은 다양한 비주얼 언어 작업에서 효과적이고 우수함을 입증하였습니다.



### Revisiting Acoustic Features for Robust ASR (https://arxiv.org/abs/2409.16399)
Comments:
          submitted to ICASSP 2025

- **What's New**: 이 논문은 생물학적 청각 지각에서 영감을 받은 새로운 음향 특징들을 사용하여 자동 음성 인식(ASR) 시스템의 정확성과 강인성을 평가합니다. 저자들은 새로운 음향 특징인 Frequency Masked Spectrogram (FreqMask)와 Difference of Gammatone Spectrogram (DoGSpec)을 제안하며, 이들이 기존의 Log Mel Spectrogram (LogMelSpec)과 비교하여 우수한 성능을 나타낸다고 주장합니다.

- **Technical Details**: 이 연구에서는 생물학적으로 더 타당한 음향 특징을 사용하는 것이 ASR의 전사 정확성과 강인성에 미치는 영향을 분석합니다. 특징들은 Gammatone filterbank 특징(GammSpec), Frequency Masked Spectrogram (FreqMask), Difference of Gammatone Spectrogram (DoGSpec) 등으로 구성되며, 각 특징은 Short-Time Fourier Transform (STFT)으로부터 계산됩니다. 주목할 만한 점은 FreqMask가 주파수 마스킹 현상을 시뮬레이션하고, DoGSpec가 레테럴 억제를 모델링한다는 것입니다.

- **Performance Highlights**: 실험 결과 DoGSpec은 LogMelSpec에 비해 상대적으로 높은 강인성을 보였으나 정확도 저하가 миним했습니다. GammSpec은 Speech Robust Bench 벤치마크의 비적대적 노이즈에 대해 더 나은 정확도와 강인성을 나타내었습니다. 또한, FreqMask와 DoGSpec를 사용하는 모델이 adversarial attacks에 대해 유의미한 강인성을 제공하며, LogMelSpec과 유사한 WER를 달성했습니다.



### Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs (https://arxiv.org/abs/2409.16341)
- **What's New**: 이번 연구에서는 기존 LLM(대형 언어 모델)을 외부 도구와 함께 사용하는 데 있어, 데이터 품질 평가의 중요성을 강조합니다. 두 가지 새로운 접근 방식을 제안하여 LLM 훈련에 사용할 데이터의 신뢰성을 측정합니다.

- **Technical Details**: 첫 번째 접근 방식은 인간이 정의한 직관적인 정확성 기준을 사용하고, 두 번째 접근 방식은 모델 기반 평가와 in-context evaluation (ICE)을 이용합니다. 또한, 데이터 품질을 평가하기 위한 자동화된 방법을 구현하여 전문가의 주석과 높은 일치를 보였습니다.

- **Performance Highlights**: 데이터 품질이 높은 훈련 데이터로 학습한 모델이 검증되지 않은 데이터로 학습한 모델보다 더 우수한 성능을 보임을 실증적으로 보여주었습니다. 이는 효율적인 훈련 데이터 관리의 중요성을 입증하며, ToolBench와 ToolAlpaca 벤치마크에서 검증되었습니다.



### Towards Within-Class Variation in Alzheimer's Disease Detection from Spontaneous Speech (https://arxiv.org/abs/2409.16322)
- **What's New**: 이 논문은 알츠하이머병(Alzheimer's Disease, AD) 탐지 분야에서 머신러닝(classification model)을 활용하여 AD 환자와 비환자를 구별하려는 연구의 일환이다. 기존의 이진 분류(binary classification) 접근법의 한계점을 지적하며, 내적 변동성(within-class variation) 및 샘플 불균형(instance-level imbalance) 문제를 해결하기 위한 두 가지 새로운 방법, Soft Target Distillation(SoTD)와 Instance-level Re-balancing(InRe)을 제안한다.

- **Technical Details**: AD 탐지의 중요한 문제는 샘플 간 인지 기능의 정도가 다르다는 것이다. SoTD는 샘플의 세부 정보를 인식하여 신뢰도(Awareness of sample degree)를 바탕으로 단계적인 학습을 제공하고, InRe는 데이터 로스(loss)를 재조정하여 오버피팅(over-fitting)을 완화하는 방법이다. 실험은 ADReSS와 ADReSSo 데이터셋에서 BERT 및 RoBERTa 임베딩(features)을 사용하여 수행되었다.

- **Performance Highlights**: 실험 결과, SoTD와 InRe 방법을 도입함으로써 AD 탐지 정확도가 크게 향상되었으며, SoTD는 특히 모델 앙상블(ensemble estimation) 기법에 비해 더 높은 효율성을 보였다. 또한, InRe는 모델의 오버피팅을 현저히 줄이며, 훈련 안정성을 높였다.



### A Literature Review of Keyword Spotting Technologies for Urdu (https://arxiv.org/abs/2409.16317)
- **What's New**: 이 문헌 리뷰는 파키스탄의 저자원 언어(Low-Resource Language)인 우르두어의 키워드 스포팅(Keyword Spotting, KWS) 기술 발전을 조사합니다. 저자원 언어가 직면한 독특한 도전과제와 이를 해결하기 위한 맞춤형 솔루션의 필요성을 강조합니다.

- **Technical Details**: 이 리뷰는 가우시안 혼합 모델(Gaussian Mixture Models, GMMs)에서 심층 신경망(Deep Neural Networks, DNNs) 및 변환기(transformer)와 같은 복잡한 신경 아키텍처의 진화를 추적합니다. 특히 다중 작업 학습(multi-task learning)과 자가 감독 학습(self-supervised learning) 접근법을 통합하여 라벨 없는 데이터(unlabeled data)를 활용하는 방법을 강조합니다. 새로운 EdgeCRNN 모델과 통합된 CNN과 RNN을 포함하여, 키워드 탐지를 위한 최신 모델들과 그 효율성을 논의합니다.

- **Performance Highlights**: 최신 연구에서, 자가 감독 학습(S3RL) 및 경량 변환기 모델들이 우르두어와 같은 저자원 언어에서 KWS 효율성과 정확성을 향상시키는데 긍정적인 영향을 미쳤습니다. Massively Multilingual Speech(MMS) 프로젝트는 1000개 이상의 언어에서 모델을 사전 학습하여 현대적인 음성 기술을 다수의 언어로 확대했으나, 여전히 우르두어는 데이터 부족 문제로 인해 성능 향상에 제한이 있습니다.



### How Redundant Is the Transformer Stack in Speech Representation Models? (https://arxiv.org/abs/2409.16302)
- **What's New**: 이 논문에서는 transformer 기반의 음성 표현 모델에서 계층 간의 중복성을 조사하고, 이러한 중복성을 활용하여 계층을 가지 치거나 대체할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 층 유사성을 분석하기 위해 세 가지 유사도 메트릭(cosine similarity, centered kernel alignment, mutual nearest-neighbor alignment)을 사용하였으며, 고유한 블록 구조와 작업 단계를 발견했습니다. 또한, pruning을 통해 transformer 계층을 최대 40%까지 축소할 수 있음을 보여줍니다.

- **Performance Highlights**: 지식 증류(knowledge distillation) 방법을 이용하여 전체 transformer 스택을 최소화하고, 네트워크 크기를 95-98% 감소시키며, 추론 시간은 최대 94%까지 단축할 수 있음을 입증하였습니다.



### Efficient Training of Self-Supervised Speech Foundation Models on a Compute Budg (https://arxiv.org/abs/2409.16295)
Comments:
          To appear in SLT 2024

- **What's New**: 이 연구는 제한된 계산 예산 하에서 음성 기반 모델 학습의 효율성을 높이기 위한 다양한 요소들을 분석하고 익히려는 시도를 포함하고 있으며, 기존의 방법론들이 잘 알려진 자원 효율성과 성능 간의 균형을 찾기 위한 실험을 진행했습니다.

- **Technical Details**: 논문에서는 self-supervised learning (SSL) 목표를 평가하고, 모델 아키텍처, 모델 크기, 데이터 크기가 SSL 성과에 미치는 영향을 체계적으로 조사합니다. 연구에 따르면 슬리머 모델 아키텍처가 일반적인 작은 모델 아키텍처보다 더 뛰어난 성과를 보이며, 사전 훈련 데이터의 크기도 성능에 큰 영향을 미친다고 밝혔습니다. 또한, 특정 계산 예산 내에서 최적의 모델 크기를 찾는 방법을 제안합니다.

- **Performance Highlights**: 이 연구의 주요 결과는 다음과 같습니다: 1) SSL 목표는 성과에 영향을 줄 수 있지만, 다른 주요 요소에 비해 그 영향이 덜하다. 2) 동일한 계산 및 파라미터 예산 하에서 슬리머 SSL 모델이 일반적으로 사용되는 3-레이어 작은 SSL 모델을 초월한다. 3) 충분히 큰 사전 훈련 데이터의 중요성이 강조되며, 데이터 사이즈와 이터레이션 간의 균형이 필요하다.



### Unsupervised Word Discovery: Boundary Detection with Clustering vs. Dynamic Programming (https://arxiv.org/abs/2409.14486)
Comments:
          3 figures, 3 tables

- **What's New**: 이 논문에서는 비지도 학습 환경에서 음성을 단어와 유사한 단위로 분할하고 이를 군집화하여 렉시콘(lexicon)을 구축하는 새로운 접근 방식을 제안합니다. 이 방법은 인접한 자기 지도 특성 간의 비유사성을 이용하여 단어 경계를 예측하고, 이를 통해 단순한 방식으로 렉시콘을 구축합니다.

- **Technical Details**: 연구에서는 먼저 우세 기반 접근(presence-based approach)을 사용하여 단어 경계를 찾아내고, 그런 다음 발견된 단어들과 그에 대한 특성을 클러스터링하여 렉시콘을 구축합니다. 이 시스템은 HuBERT라는 고급 음성 특성을 기반으로 하며, 주어진 음성 프레임 사이의 코사인 거리(cosine distance)를 이용하여 단어 경계를 예측합니다. 이후 PCA(주성분 분석)로 차원을 축소한 후, 음향 단어 임베딩(acoustic word embedding)을 통해 단어 세그먼트를 고정 차원의 벡터로 변환하여 클러스터링합니다.

- **Performance Highlights**: ZeroSpeech 벤치마크의 5개 언어에서 테스트한 결과, 제안된 간단한 방법은 새로운 ES-KMeans+ 방법과 유사한 성능을 보이며, 속도는 거의 5배 빠릅니다. 또한, 언어에 특화된 모델이 다국어 모델보다 성능이 더 뛰어난 것을 확인했습니다.



### M^2PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning (https://arxiv.org/abs/2409.15657)
Comments:
          EMNLP 2024

- **What's New**: 이 연구에서는 Multimodal Prompt Tuning (M^2PT) 이라는 새로운 접근 방식을 소개하여 멀티모달 대형 언어 모델(MLLM) 의 효과적인 instruction tuning을 지원하게 됩니다. M^2PT는 시각적 프롬프트(visual prompts)와 텍스트 프롬프트(textual prompts)를 통합하여 전반적인 매개변수의 0.09%만 조정하면서도 우수한 성능을 발휘합니다.

- **Technical Details**: M^2PT는 비전 인코더(vision encoder)와 언어 프로세서(language processor) 각각에 시각적 프롬프트와 텍스트 프롬프트를 삽입하여 기능(feature) 를 추출하고 각 모달리티(modality) 간의 정렬을 촉진합니다. 이 과정에서 두 세트의 프롬프트 간의 교차 모달 상호작용(cross-modality interaction)이 강화되며, 이를 통해 모델이 맥락을 이해하고 제로샷 제어(zero-shot control)에서 모호성을 줄일 수 있도록 합니다.

- **Performance Highlights**: M^2PT는 여러 멀티모달 평가 데이터셋에서 여러 최신 PEFT (Parameter-Efficient Fine-Tuning) 기법과 비교하여 우수한 성능을 나타냈습니다. 특히, 기법의 효율성과 효과성을 검증하기 위한 포괄적인 경험적 실험이 진행되었습니다.



### Block-Attention for Efficient RAG (https://arxiv.org/abs/2409.15355)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 시나리오에서의 추론 지연 및 비용을 줄이기 위해 설계된 새로운 attention 메커니즘인 Block-Attention을 소개합니다. 기존 방법들과 달리, Block-Attention은 검색된 문서를 블록으로 나누어 각 블록이 KV(state) 상태를 독립적으로 계산하도록 하여 효율성을 높입니다.

- **Technical Details**: Block-Attention 메커니즘은 입력 시퀀스를 여러 블록으로 나누고, 각 블록은 자신만의 KV 상태를 self-attention을 통해 계산합니다. 마지막 블록만이 다른 블록에 접근합니다. 이 기법을 통해 모든 passage를 블록으로 정의하고 그 KV 상태를 메모리에 캐시하여, 추론 시의 지연 시간을 크게 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, Block-Attention을 적용하면 평균 TTFT(첫 번째 토큰까지의 시간)를 98.7% 줄일 수 있으며(3638 ms에서 45 ms로 단축), 첫 번째 토큰을 추출할 때의 FLOPs 역시 99.8% 감소시킵니다. 블록-어텐션 모델은 기존의 self-attention 모델에 비해 (Llama3: 68.4% vs 67.9%, Mistral: 62.8% vs 59.6%) 유사하거나 더 나은 성능을 보여줍니다.



New uploads on arXiv(cs.IR)

### Enhancing Automatic Keyphrase Labelling with Text-to-Text Transfer Transformer (T5) Architecture: A Framework for Keyphrase Generation and Filtering (https://arxiv.org/abs/2409.16760)
- **What's New**: 이 논문은 Text-to-Text Transfer Transformer (T5) 아키텍처를 기반으로 한 새로운 키프레이즈 생성 모델 docT5keywords를 제안합니다. 이 모델은 문서의 제목과 초록을 입력으로 받아 키프레이즈를 생성합니다. 또한, 다수결 접근법을 통해 생성된 여러 시퀀스의 빈도에 따라 키프레이즈를 정렬하여 예측 결과를 개선하는 방법을 소개합니다.

- **Technical Details**: 모델은 T5 아키텍처를 사용하여 문서의 키프레이즈와 관련된 두 가지 평가 방법론을 구현합니다. 첫 번째는 이진 평가로, 키프레이즈가 주어진 문서와 관련 있는지를 예측합니다. 두 번째로, 여러 AKG 모델에 의한 예측 키프레이즈 필터링을 통해 평가 점수를 개선하는 방법을 사용합니다. 이를 통해 정확성을 높이고 불필요한 키프레이즈를 줄이는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안한 키프레이즈 생성 모델은 모든 기준선 모델을 크게 능가하며, 일부 경우 100% 이상의 성능 향상을 보였습니다. 필터링 기술은 모든 데이터 세트에서 가짜 긍정(False Positive)을 제거하는 데 근사 완벽한 정확도를 달성했습니다.



### A Prompting-Based Representation Learning Method for Recommendation with Large Language Models (https://arxiv.org/abs/2409.16674)
Comments:
          Risks: The 1st International Workshop on Risks, Opportunities, and Evaluation of Generative Models in Recommendation

- **What's New**: 최근 대규모 언어 모델(LLM)이 추천 시스템(RS) 분야에 도입됨에 따라, 해당 시스템은 혁신적인 변화를 겪고 있습니다. 본 논문은 LLM의 특성을 고려한 새로운 추천 프레임워크인 Prompting-Based Representation Learning Method for Recommendation (P4R)을 제안합니다.

- **Technical Details**: P4R 프레임워크는 LLM 프롬프트 전략을 활용하여 개인화된 항목 프로필을 생성하고, 이 프로필은 미리 학습된 BERT 모델을 사용해 의미적 표현 공간으로 변환됩니다. 또한, 협업 필터링을 위한 Graph Convolution Network (GCN)도 통합되어 일반 추천 작업을 해결합니다.

- **Performance Highlights**: P4R은 최신 추천 모델들과 성능을 비교하였고, LLM 기반 추천 시스템에서 어떤 맥락 정보가 중요한지를 분석했습니다. 실험 결과, 특히 중소기업에 적합한 성능 향상을 보여주었습니다.



### Train Once, Deploy Anywhere: Matryoshka Representation Learning for Multimodal Recommendation (https://arxiv.org/abs/2409.16627)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 본 연구에서는 다중 모드 추천 시스템을 위한 경량화된 프레임워크인 full-scale Matryoshka representation learning for multimodal recommendation (fMRLRec)를 소개합니다. fMRLRec는 서로 다른 세부 수준에서 아이템 특징을 캡처하며, 정보적 표현을 학습하여 다차원에서 효율적인 추천을 제공합니다.

- **Technical Details**: fMRLRec는 다양한 모드에서 아이템 특징을 통합하기 위해 간단한 매핑 기법을 사용하여 다중 모드 아이템 특징을 정렬된 피처 공간으로 투사합니다. 효율적인 선형 변환을 설계하여 더 작은 특징을 더 큰 특징으로 임베드하여 대규모 추천 데이터 교육을 위한 메모리 요구 사항을 대폭 줄입니다.

- **Performance Highlights**: fMRLRec는 여러 벤치마크 데이터셋에서 그 효과성과 효율성을 입증하며, 최신 방법들과 비교했을 때 추천 성능과 교육 효율성에서 일관되게 우수한 성과를 보여줍니다.



### Generative Pre-trained Ranking Model with Over-parameterization at Web-Scale (Extended Abstract) (https://arxiv.org/abs/2409.16594)
- **What's New**: 본 연구에서는 웹 검색에서 중요한 웹페이지를 우선순위로 설정하기 위한 새로운 학습 모델인 Generative Semi-Supervised Pre-trained (GS2P) LTR 모델을 제안합니다. 이 모델은 부족한 주석(query-webpage pairs with ranking scores)의 문제를 해결하고, 일반화된 표현을 학습하도록 설계되었습니다.

- **Technical Details**: GS2P 모델은 고품질의 유사 레이블을 추출하기 위해 다양한 LTR 모델의 코-트레이닝(co-training)을 사용합니다. 이 과정에서 일반화된 표현을 학습하기 위해 자기 주의(self-attentive) 네트워크를 활용하며, MLP 기반의 ranker와 Random Fourier Features (RFF)를 조합하여 성능을 향상시킵니다.

- **Performance Highlights**: GS2P 모델은 공개 데이터셋과 실제 대규모 검색 엔진에서 수집한 데이터셋을 사용하여 실험을 수행하였으며, A/B 테스트를 통해 기존 시스템 대비 현저한 성능 향상을 보여주었습니다.



### FusionANNS: An Efficient CPU/GPU Cooperative Processing Architecture for Billion-scale Approximate Nearest Neighbor Search (https://arxiv.org/abs/2409.16576)
Comments:
          15 pages, 26 figures

- **What's New**: FusionANNS는 CPU와 GPU의 협업 필터링 및 재정렬 메커니즘을 활용하여 대량의 벡터 데이터셋에서 고처리량, 저지연, 비용 효율성 및 높은 정확도를 동시에 달성할 수 있는 새로운 ANNS 시스템입니다.

- **Technical Details**: FusionANNS는 (1) 다단계 인덱싱, (2) 휴리스틱 재정렬, (3) 중복 확인 I/O 중복 제거 등 세 가지 혁신적인 디자인을 통해 CPU와 GPU 간의 데이터 전송을 최소화하며, SSD에 저장된 원시 벡터와 GPU의 HBM에서 압축된 벡터를 결합하여 사용합니다.

- **Performance Highlights**: FusionANNS는 SPANN에 비해 9.4-13.1배 높은 QPS(초당 쿼리 수)와 5.7-8.8배 높은 비용 효율성을 달성했으며, RUMMY와 비교하여 2-4.9배 높은 QPS와 2.3-6.8배 높은 비용 효율성을 보장합니다.



### Algorithmic Drift: A Simulation Framework to Study the Effects of Recommender Systems on User Preferences (https://arxiv.org/abs/2409.16478)
- **What's New**: 이 논문은 추천 시스템의 장기적인 사용자 행동 변화에 대한 영향을 정량화할 수 있는 새로운 접근 방식을 제안합니다. 이 연구는 사용자와 추천 알고리즘 간의 상호작용을 모델링하기 위한 확률적 시뮬레이션 프레임워크를 채택합니다.

- **Technical Details**: 논문에서는 사용자 저항(user resistance) 및 관성(inertia)과 같은 행동적 측면을 포함하여 사용자 모델을 공식화하고, 추천 알고리즘이 사용자 선호에 미치는 영향을 평가하기 위해 새로운 메트릭스(새로운 지표)를 도입합니다. 시뮬레이션 모델은 처음에 이질적인 사용자 선호 그룹에서 시작되며, 추천 시스템이 초기 전이 확률을 유도하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법론이 사용자 선호의 변화(algorithmic drift)를 정확히 감지하고 정량화하는 데 효과적이라는 것을 입증하였습니다. 다양한 신합성 데이터셋을 통해 시스템의 강건성을 검증하며, 사용자 선택의 시나리오와 하이퍼 파라미터 설정에 따른 결과를 평가했습니다.



### Spacewalker: Traversing Representation Spaces for Fast Interactive Exploration and Annotation of Unstructured Data (https://arxiv.org/abs/2409.16793)
- **What's New**: 이번 논문에서 우리는 다양한 모달리티를 활용한 비구조화 데이터 분석을 위한 상호작용 도구인 Spacewalker를 소개합니다. 이 도구는 데이터를 탐색하고 주석을 달 수 있도록 설계되었습니다.

- **Technical Details**: Spacewalker는 사용자가 임의의 모달리티의 데이터 세트를 업로드하고, Low-dimensional space에서 데이터의 시맨틱 유사성을 강조하여 시각화할 수 있는 기능을 제공합니다. Bayesians networks 및 Deep Learning 기반의 방법을 사용하여 데이터를 추출합니다.

- **Performance Highlights**: 사용자 연구 결과, Spacewalker는 기존 방법에 비해 데이터 주석 속도를 현저히 개선하며, 데이터 무결성 검증 및 부패 데이터 세트 식별 작업에서도 빠른 탐색이 가능합니다.



### PIFS-Rec: Process-In-Fabric-Switch for Large-Scale Recommendation System Inferences (https://arxiv.org/abs/2409.16633)
- **What's New**: 이 논문은 Deep Learning Recommendation Models (DLRMs)의 성능 개선을 위해 CXL(Compute Express Link) 기술을 활용하여 새로운 PIFS-Rec(장비 스위치 내 작업 처리) 접근 방식을 제안합니다. 이는 메모리 및 대역폭 확장을 최적화하여 DLRMs를 가속화하는 데 중점을 둡니다.

- **Technical Details**: CXL 시스템에서 DLRM 작업량을 특성화하고 주된 병목 현상을 식별합니다. PIFS-Rec는 데이터 프로세싱과 효율적인 메모리 사용을 위한 하드웨어 및 소프트웨어 최적화를 결합하여, 낮은 대기 시간으로 Pond 및 BEACON과의 비교에서 각각 3.89배 및 2.03배 우수한 성능을 보입니다.

- **Performance Highlights**: PIFS-Rec는 CXL 기반 시스템에서의 DLRM 작업의 성능을 크게 향상시킵니다. 본 연구에서 제안한 접근 방식은 데이터 이동 비용을 줄이고, 메모리 대역폭 병목 현상을 해결하여 데이터 센터 규모의 DLRM 처리 성능을 극대화합니다.



### Evaluating and Enhancing Large Language Models for Novelty Assessment in Scholarly Publications (https://arxiv.org/abs/2409.16605)
Comments:
          under review

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 학술 논문에서의 창의성 및 참신성을 평가하기 위한 새로운 벤치마크인 SchNovel을 도입하였습니다. 이 벤치마크는 arXiv 데이터 세트에서 선택된 15,000 쌍의 논문으로 구성되어 있으며, 각 쌍의 최근 발표된 논문이 더 참신하다고 가정합니다. 또한, RAG-Novelty라는 새로운 방법을 제안하여 LLM이 논문의 참신성을 평가할 때 유사한 논문의 검색을 활용합니다.

- **Technical Details**: SchNovel 벤치마크는 2~10년 차이가 나는 논문 쌍을 포함하며, 이는 특히 높은 수준의 리뷰 과정을 거치는 학술 논문에서 참신성을 평가하는 데 중요합니다. RAG-Novelty는 검색 기반 생성 방법으로, 더 참신한 논문일수록 최근 발표된 논문을 더 많이 검색할 것이라는 가정을 바탕으로 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 RAG-Novelty가 기존의 기준 모델보다 논문의 참신성을 평가하는 데 더 뛰어난 성능을 보인다는 것을 입증했습니다. 이 연구는 LLM의 논문 참신성 평가 능력을 깊이 있게 탐구하고, 다양한 카테고리와 출판 연도 간의 변화를 평가하여 LLM의 성능 향상에 기여하였습니다.



### Pre-trained Graphformer-based Ranking at Web-scale Search (Extended Abstract) (https://arxiv.org/abs/2409.16590)
- **What's New**: MPGraf 모델은 Transformer와 Graph Neural Networks (GNNs)의 장점을 통합하여 학습 랭킹(learning to rank) 문제에 접근합니다. 이 모델은 모듈형 및 캡슐 기반의 사전 학습(pre-training) 전략을 사용하여 웹 규모의 통합된 LTR 프레임워크를 구현합니다.

- **Technical Details**: MPGraf는 세 가지 주요 단계를 포함합니다: (1) Link Rippiling을 통한 그래프 구성, (2) Hybrid Graphformer를 통한 표현 학습, (3) 모듈 구성(modular composition)을 통한 정밀 조정(surgical fine-tuning)입니다. 이를 통해 sparsely annotated query-webpage 쌍에서 그래프 기반 학습 데이터를 생성하고, GNN과 Transformer 모듈을 조합하여 하이브리드 아키텍처에서 특성 학습을 수행합니다.

- **Performance Highlights**: MPGraf는 실제 검색 엔진 환경에서 extensive offline 및 online 실험을 수행한 결과, 최신의 웹페이지 랭킹 방법들과 비교하여 최고의 성능을 나타냈습니다. 특히 온라인 평가에서 상당한 개선을 달성했습니다.



### Modern Hopfield Networks meet Encoded Neural Representations -- Addressing Practical Considerations (https://arxiv.org/abs/2409.16408)
Comments:
          17 pages, 8 figures, workshop submission to Neurips

- **What's New**: 본 논문은 Modern Hopfield Networks (MHN)에 대한 메타 안정 상태 문제를 해결하는 새로운 접근 방식인 Hopfield Encoding Networks (HEN)를 소개합니다. HEN은 입력 패턴의 분리 가능성을 높이고, 메타 안정 상태를 줄이기 위해 인코딩된 신경 표현을 통합합니다.

- **Technical Details**: HEN은 미리 훈련된 신경 인코더-디코더 모델을 사용하여 입력을 잠재 표현 공간으로 인코딩한 후 저장하고, 재호출 시 다시 디코딩하는 방법을 사용합니다. 이 접근 방식은 MHNs의 메타 안정 상태 문제를 해결하고, 자연어 쿼리를 통한 다양한 입력 모달리티에서의 검색을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 HEN이 메타 안정 상태를 크게 줄이고, 저장 용량을 증가시키면서 다양한 입력을 완벽히 기억할 수 있음을 나타냅니다. 이는 실제 작업을 위한 연상 기억 네트워크의 실용적인 활용을 향상시킵니다.



### Mitigating Digital Discrimination in Dating Apps -- The Dutch Breeze cas (https://arxiv.org/abs/2409.15828)
- **What's New**: 2023년 9월, 네덜란드 인권 연구소는 네덜란드의 데이팅 앱 Breeze가 비백인에 대한 차별 가능성을 의심한 것이 정당하다고 결정했습니다. 이로 인해 Breeze는 인종에 따른 차별을 방지해야 한다는 결정을 받았습니다.

- **Technical Details**: 이 논문은 Breeze의 매칭 알고리즘에서 인종에 기반한 차별이 불법인지에 대한 질문과 데이팅 앱들이 매칭 알고리즘에서 차별을 완화하거나 중단할 수 있는 방법을 탐구합니다. 또한, 컴퓨터 과학과 법률의 통찰을 결합하여 Breeze 결정의 법적 및 기술적 어려움을 심도 있게 분석합니다.

- **Performance Highlights**: 정당한 차별 방지 조치를 실행하며, 공정하고 비차별적인 머신러닝(machine learning) 분야에서의 연구와 실천에 대한 논의가 포함되어 있습니다.



New uploads on arXiv(cs.CV)

### Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models (https://arxiv.org/abs/2409.17146)
- **What's New**: Molmo는 현재 공개된 VLM 중에서 가장 최신의 성능을 자랑하며, 독점적인 데이터가 아닌 새롭게 수집된 고품질 데이터로 구성된 이미지를 설명하는 고급 캡션 데이터셋을 기반으로 하고 있습니다. 이 연구는 VLM을 처음부터 구축하는 데 필요한 기본 지식을 제공합니다.

- **Technical Details**: Molmo 모델은 기존의 비전 인코더와 언어 모델을 결합하여 만들어졌습니다. 이 과정에서는 이미지로부터 자세하고 질 높은 설명을 생성하는 새로운 데이터셋인 PixMo를 사용하고, 60~90초 동안 음성으로 설명하도록 요구하여 다양한 내용을 포괄하도록 했습니다. 모델 훈련은 다단계 과정이 아닌, 간단한 훈련 파이프라인을 통해 진행되었습니다.

- **Performance Highlights**: Molmo-72B 모델은 학술 벤치마크에서 최고의 점수를 기록했으며, GPT-4o와 비교했을 때 사용자 선호도 순위에서 두 번째에 올랐습니다. Molmo-1B 모델은 효율적인 성능을 보이며 GPT-4V와 근접한 결과를 보여주었고, 전체적으로 많은 상업적 시스템을 능가하는 성능을 발휘했습니다.



### DreamWaltz-G: Expressive 3D Gaussian Avatars from Skeleton-Guided 2D Diffusion (https://arxiv.org/abs/2409.17145)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 텍스트 기반의 3D 아바타 생성 프레임워크인 DreamWaltz-G를 소개한다. 이 프레임워크는 Skeleton-guided Score Distillation과 Hybrid 3D Gaussian 아바타 표현을 핵심으로 하여, 보다 일관성 있는 아바타 생성을 목표로 한다.

- **Technical Details**: DreamWaltz-G는 두 가지 단계로 아바타를 생성하는 방식으로, (I) Canonical Avatar Generation 단계에서 텍스트 설명을 바탕으로 기본 3D 아바타를 생성하고, (II) Animatable Avatar Learning 단계에서 이 아바타를 SMPL-X로 리깅하고 애니메이션을 추가하는 최적화를 진행한다.

- **Performance Highlights**: 광범위한 실험을 통해 DreamWaltz-G는 기존 방법들과 비교하여 시각적 품질과 애니메이션 표현력 면에서 우수한 성능을 발휘하며, 인상적인 3D 아바타 생성 및 애니메이션 수행 능력을 입증했다.



### Attention Prompting on Image for Large Vision-Language Models (https://arxiv.org/abs/2409.17143)
Comments:
          Website, see this https URL

- **What's New**: 본 연구에서는 Attention Prompting on Image (𝒜⁢𝒫⁢ℐ𝒜𝒫ℐ)라는 새로운 프롬프트 기법을 제안하여, 원본 이미지 위에 텍스트 쿼리 기반 주의 열지도를 오버레이함으로써 LVLM의 성능을 향상시킵니다.

- **Technical Details**: 이 기법은 보조 LVLM 모델을 활용하여 입력 이미지에 대한 주의 열지도를 생성합니다. 이때 CLIP과 같은 이미지-텍스트 매칭 모델의 cls 토큰 유사도 점수를 기반으로 하여 열지도를 만들어 냅니다. 생성된 열지도는 원본 이미지의 픽셀 값에 단순히 곱해져 LVLM의 실제 입력 이미지가 됩니다.

- **Performance Highlights**: Attention Prompting on Image는 LLaVA-1.5 모델의 MM-Vet, LLaVA-Wild 벤치마크에서 각각 3.8% 및 2.9%의 성능 향상을 보여줍니다.



### Streaming Neural Images (https://arxiv.org/abs/2409.17134)
Comments:
          IEEE International Conference on Image Processing (ICIP)2024

- **What's New**: 본 논문은 Implicit Neural Representations (INRs)의 이미지 압축에서의 기존 한계를 분석하고, 계산 비용, 성능 불안정성 및 견고성과 같은 중요한 요인을 제시합니다.

- **Technical Details**: INRs는 신호를 지속적으로 설명하는 함수로 매핑하여 RGB 값을 생성하는 방식으로, 신경망(Neural Network)을 사용하여 이 함수(ϕ)를 학습합니다. 또한, SPINR(Streaming Progressive INRs) 방법론을 소개하여 이미지 압축 및 전송에서의 여러 문제를 해결합니다.

- **Performance Highlights**: 각종 실험을 통해 INRs의 강화된 압축 능력을 보였으며, 기존 압축 방식인 JPEG 및 JPEG2000과 비교하여 그 잠재력을 강조합니다.



### Small data deep learning methodology for in-field disease detection (https://arxiv.org/abs/2409.17119)
Comments:
          9 pages

- **What's New**: 본 연구에서는 농업 질병, 특히 감자에서 발생하는 연한 증상을 조기에 감지할 수 있는 첫 번째 머신러닝 모델을 제안합니다. 이 모델은 필드에서 직접 촬영된 고해상도 RGB 이미지를 분석하여 기존 문헌의 한계를 극복하고 실제 적용 가능성을 제시합니다.

- **Technical Details**: 제안된 ISD4L(In-field Small Data Disease Detection with Deep Learning) 방법론은 고해상도 이미지를 패치(patch)로 나누고, 이 패치를 사용한 훈련을 통해 진단 모델을 학습합니다. 이 모델은 깊은 합성곱 신경망(convolutional neural networks)과 초점 손실 함수(focal loss function)를 활용하며, 데이터 증강(data augmentation) 기법을 통해 적은 수의 고해상도 이미지로도 효과적인 훈련이 가능합니다.

- **Performance Highlights**: 개발된 모델은 테스트 데이터셋에서 모든 늦은 흑색병(latent blight) 사례를 올바르게 감지하며, 초기 증상을 식별하는 데 있어 높은 정확성과 효과성을 입증했습니다. 이러한 결과는 농업에서의 질병 및 해충 조기 감지를 위한 머신러닝의 잠재적 활용성을 강화합니다.



### MorphoSeg: An Uncertainty-Aware Deep Learning Method for Biomedical Segmentation of Complex Cellular Morphologies (https://arxiv.org/abs/2409.17110)
- **What's New**: 이 논문에서는 생물학적 세포의 복잡한 형태를 효과적으로 분할하기 위한 새로운 벤치마크 데이터셋인 Ntera-2 (NT2) 세포를 소개하고, 동 불확실성을 인식하는 딥러닝 프레임워크인 MorphoSeg를 제안합니다.

- **Technical Details**: MorphoSeg는 저확률 영역에서 가상의 이상치를 샘플링하는 방식을 통하여 세포 분할의 경계를 개선하는 방법입니다. 이 접근법은 TransUNet을 기반으로 하여 불규칙한 세포 형상과 훈련의 어려움을 해결합니다.

- **Performance Highlights**: MorphoSeg는 Dice Similarity Coefficient (DSC)를 80.35%에서 86.57%로 향상시키고, Hausdorff Distance (HD95)를 21.98에서 15.75로 감소시키는데 성공하여 정확도가 크게 향상되었습니다.



### Unveiling Ontological Commitment in Multi-Modal Foundation Models (https://arxiv.org/abs/2409.17109)
Comments:
          Qualitative Reasoning Workshop 2024 (QR2024) colocated with ECAI2024, camera-ready submission; first two authors contributed equally; 10 pages, 4 figures, 3 tables

- **What's New**: 이번 논문은 다중 모달 심층 신경망(DNN)에서 학습된 개념의 슈퍼클래스 계층 구조를 추출하는 방법을 제안합니다. 이를 통해 질적 추론(qualitative reasoning, QR) 모델과의 검증 및 확인을 위한 단계로 나아갑니다.

- **Technical Details**: 우리는 DNN의 텍스트 입력 모달리티를 사용하여 리프 개념의 임베딩을 얻고, 이를 계층적 클러스터링을 통해 의미적 유사성을 기반으로 하는 슈퍼클래스 개념을 라벨링합니다. 제안된 방법은 다중 모달 DNN의 중간 표현에서 간단한 온톨로지를 추출하고 검증할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 의미 있는 온톨로지를 추출할 수 있고, 주어진 온톨로지와의 불일치성을 밝혀낼 수 있음을 보여주었습니다. 또한, QR 모델의 추출 및 삽입에 대한 잠재적 응용 가능성을 논의하였습니다.



### Text2CAD: Generating Sequential CAD Models from Beginner-to-Expert Level Text Prompts (https://arxiv.org/abs/2409.17106)
Comments:
          Accepted in NeurIPS 2024 (Spotlight)

- **What's New**: 텍스트 기반의 지침을 통해 파라메트릭 (parametric) CAD 모델을 생성하는 첫 번째 AI 프레임워크인 Text2CAD를 제안합니다. 이 접근 방식은 모든 수준의 디자이너에게 적합하도록 설계되었으며, 디지털 프로토타입 제작의 효율성을 향상시킬 수 있습니다.

- **Technical Details**: Text2CAD는 디자이너 친화적인 텍스트 프롬프트를 기반으로 CAD 모델을 생성하는 엔드 투 엔드 (end-to-end) 트랜스포머 기반의 오토리그레시브 (auto-regressive) 네트워크를 포함합니다. 이 시스템은 DeepCAD 데이터셋에 대한 텍스트 주석을 생성하기 위한 데이터 주석 파이프라인을 도입하여 약 170,000개의 모델과 660,000개의 텍스트 주석을 포함하고 있습니다.

- **Performance Highlights**: 우리의 실험 분석은 제안한 프레임워크가 시각적 품질, 파라메트릭 정밀도, 기하학적 정확성을 포함한 다양한 메트릭을 통해 두 단계(2-stage) 기준 방법보다 우수한 성능을 보여주었음을 입증합니다.



### General Detection-based Text Line Recognition (https://arxiv.org/abs/2409.17095)
- **What's New**: 이번 논문에서는 인쇄된 텍스트(OCR)와 손글씨(HTR) 인식을 위해 새로운 탐지 기반 접근 방식인 DTLR(Detection-based Text Line Recognition)을 제안합니다. 기존의 HTR 방법들은 개별 문자를 분리하여 처리하는 데 어려움이 있었으나, 본 연구에서는 이를 극복하여 다양한 스크립트에 적용 가능한 방식으로 발전하였습니다.

- **Technical Details**: DTLR 접근 방식은 세 가지 주요 통찰에 기반하고 있습니다: (i) 다양한 데이터로의 합성 사전 훈련을 통해 문자의 위치를 적절히 일반화할 수 있게 된다; (ii) 현대의 transformer 기반 탐지 모델은 여러 인스턴스를 조화롭게 탐지하며, 적절한 마스킹 전략을 사용하면 각 탐지 간의 일관성을 활용할 수 있다; (iii) 사전 훈련된 탐지 모델을 사용하여 실제 데이터에서의 라인 수준 주석으로 미세 조정할 수 있으며, 이는 다른 알파벳에서도 적용할 수 있다.

- **Performance Highlights**: 우리는 DTLR이 다양한 데이터셋에서 뛰어난 성능을 발휘하며, 특히 중국어 스크립트 인식에서 CASIA v2 데이터셋과 암호 인식에서 Borg 및 Copiale 데이터셋에서 선진 성능을 개선했음을 입증했습니다. 이 접근 방식은 기존 HTR 방식과는 다른 패러다임을 채택하여 여러 스크립트에서 효과적으로 작동합니다.



### BitQ: Tailoring Block Floating Point Precision for Improved DNN Efficiency on Resource-Constrained Devices (https://arxiv.org/abs/2409.17093)
- **What's New**: 이번 연구에서는 DNN(Deep Neural Networks) 추론을 위한 최적의 BFP(Block Floating Point) 구현을 목표로 하는 비트너스(Bitwidth) 인식 분석 모델링 프레임워크(BitQ)를 제안합니다. 기존의 BFP 기반 양자화가 블록 크기와 정확도를 경험적으로 선택하였던 반면, BitQ는 최적의 BFP 블록 크기와 비트너스를 결정하기 위한 최적화 문제를 해결합니다.

- **Technical Details**: BitQ는 자원 제약 장치에서의 DNN 효율성을 높이기 위해 데이터 이동량과 정확성 간의 트레이드오프를 활용하며, BFP 양자화 설정을 탐색하기 위한 최적의 구성을 식별합니다. 이 방식은 DNN의 데이터 재사용성을 충분히 탐색하여 데이터 이동량을 평가하는 BFP 기반 모델링 접근 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, BitQ는 이미지 분류, 객체 탐지, 장면 세분화 응용 분야에서 기존의 최첨단 기법들과 비교하여 뛰어난 성능을 보였으며, 정확성을 유지하면서 계산 효율성과 메모리 접근 요구 사항을 감소시켰습니다.



### Ctrl-GenAug: Controllable Generative Augmentation for Medical Sequence Classification (https://arxiv.org/abs/2409.17091)
Comments:
          17 pages, 7 figures, 7 tables

- **What's New**: 이 논문에서는 Ctrl-GenAug라는 새로운 생성적 증강 프레임워크를 제안하여, 의료 시퀀스 분류를 위한 고도로 의미적이고 연속적인 시퀀스 생성을 지원하고 부정확하게 합성된 샘플을 억제합니다.

- **Technical Details**: Ctrl-GenAug는 다중 모달 조건 유도 시퀀스 생성기를 통해 진단 촉진 샘플을 제어 가능하게 합성하며, 시간적/입체적 일관성을 향상시키는 연속 증강 모듈을 통합합니다. 또한, 불확실한 사례를 억제하는 노이즈 합성 데이터 필터를 설계하였습니다.

- **Performance Highlights**: 세 가지 의료 데이터셋과 세 가지 패러다임에서 훈련된 11개의 네트워크를 사용한 광범위한 실험에서 Ctrl-GenAug의 효과성과 일반성이 입증되었습니다. 특히, 대표성이 부족한 고위험 군과 도메인 외 조건에서의 성능 향상을 보여주었습니다.



### Parameter-efficient Bayesian Neural Networks for Uncertainty-aware Depth Estimation (https://arxiv.org/abs/2409.17085)
Comments:
          Presented at UnCV Workshop at ECCV'24

- **What's New**: 본 연구에서는 대규모 Transformer 기반 비전 모델의 서브스페이스 Bayesian inference를 위해 Parameter-Efficient Fine-Tuning (PEFT) 방법이 적합한지 조사합니다. 특히 LoRA, BitFit, DiffFit 및 새로운 PEFT 방법인 CoLoRA를 조합하여, 단일 깊이 추정(MDE)에서 보다 강건하고 신뢰할 수 있는 예측 성능을 확보할 수 있음을 보여줍니다.

- **Technical Details**: PEFT 방법(LoRA, BitFit, DiffFit 등)을 사용하여 매개변수의 고차원성 문제를 해결하고, 효율적인 추론이 가능함을 입증했습니다. CoLoRA는 Tucker 분해를 기반으로 한 저차원 perturbation을 적용하여 깊이 추정 문제에 적합하게 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 PEFT 방법을 사용하여 MDE에서 개선된 예측 성능을 달성하였고, Bayesian inference를 통해 모델의 신뢰성을 높일 수 있음을 확인하였습니다.



### Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning? (https://arxiv.org/abs/2409.17080)
Comments:
          13 pages, 4 figures. Code released at this https URL

- **What's New**: 본 연구에서는 Spatial Visual Ambiguity Tasks (SVAT)라는 새로운 벤치마크를 제안하여 대형 비전-언어 모델(VLM)이 시각적 데모를 통해 새로운 비주얼-스페이셜 개념을 학습할 수 있는지를 평가합니다.

- **Technical Details**: SVAT는 여러 난이도 수준의 분류 작업으로 구성되어 있으며, 목표는 주어진 모호한 텍스트와 시각적 데모를 기반으로 이미지 내에서 관심 있는 객체의 정확한 위치를 학습하는 것입니다. VLM은 이 과제를 위해 Text-Query와 Image-Based Input을 사용하여 정확한 경계 결정을 해야 합니다.

- **Performance Highlights**: Zero-shot 설정에서 VLM은 SVAT 작업에서 실패하고, 단순한 파인튜닝만으로는 5.8%-27.3%의 성능 향상이 가능합니다. 그러나 커리큘럼 학습(Curriculum Learning)을 통해, VLM은 SVAT의 가장 어려운 작업에서 14.2%에서 34.2%의 정확도 개선을 이룰 수 있습니다.



### Benchmarking Domain Generalization Algorithms in Computational Pathology (https://arxiv.org/abs/2409.17063)
- **What's New**: 이번 연구는 30개의 도메인 일반화 (Domain Generalization, DG) 알고리즘의 효과를 3개의 CPath 작업에 대해 평가하고, 새로운 다중 도메인 종양 탐지 데이터셋 (HISTOPANTUM)을 소개합니다.

- **Technical Details**: 연구에서는 7,560회의 교차 검증 (cross-validation) 실험을 통해 DG 알고리즘의 상대적인 성능을 비교하며, 최근에 제안된 pretrained foundation models와 같은 모달리티별 (modality-specific) 기법을 통합했습니다.

- **Performance Highlights**: 자기 감독 학습 (self-supervised learning) 및 염색 증강 (stain augmentation) 기법이 consistently 다른 알고리즘보다 좋은 성능을 보였으며, 연구 결과는 연구자들이 CPath 작업에 적합한 DG 접근 방식을 선택하는 데 도움을 줄 수 있습니다.



### Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors (https://arxiv.org/abs/2409.17058)
Comments:
          The code is available at this https URL

- **What's New**: 본 논문에서는 효율성을 크게 개선한 새로운 하나의 단계로 구성된 Super-resolution (SR) 모델인 S3Diff를 소개합니다. 이는 기존의 diffusion 모델을 기반으로 하여 저해상도(Low-Resolution, LR) 이미지에서 고해상도(High-Resolution, HR) 이미지로 변환하는 과정을 최적화합니다.

- **Technical Details**: 제안된 S3Diff 모델은 Degradation-guided Low-Rank Adaptation (LoRA) 모듈을 사용하여 LR 이미지의 저하 정보를 기반으로 모델의 매개변수를 조정합니다. 이는 기존의 fine-tuning 전략과는 달리, 더욱 효율적이고 데이터 의존적인 SR 모델을 제공합니다.

- **Performance Highlights**: 실험 결과, S3Diff 모델은 최근의 최첨단 방법들과 비교했을 때 효율성과 효과성이 우수함을 입증하였습니다. 특히, 샘플링 단계 수를 대폭 줄이면서도 높은 시각적 품질을 유지하였습니다.



### ControlCity: A Multimodal Diffusion Model Based Approach for Accurate Geospatial Data Generation and Urban Morphology Analysis (https://arxiv.org/abs/2409.17049)
Comments:
          20 pages

- **What's New**: 이 논문에서는 다중 소스의 자원(Volunteer Geographic Information, VGI)을 활용하여 도시 건물 외형 데이터를 생성하는 새로운 접근방법인 ControlCity를 제안합니다. 이 모델은 여러 데이터 모달리티를 통합하여 더 정확하고 유용한 지리정보를 생성할 수 있습니다.

- **Technical Details**: ControlCity는 텍스트, 메타데이터, 이미지 데이터를 포함하는 'image-text-metadata-building footprint' 데이터 셋을 구축하고, 이를 기반으로 multimodal diffusion model을 사용하여 건물 외형 데이터를 생성합니다. 텍스트와 메타데이터를 정렬하여 도시의 건물 패턴을 학습하고, 개선된 ControlNet을 통해 도로 네트워크 및 토지 이용 이미지를 통합합니다.

- **Performance Highlights**: ControlCity는 전 세계 22개 도시에서 실험을 통해 평균 FID 점수 50.94를 기록하였으며, 기존 방법 대비 71.01%의 오류 감소와 38.46% 향상된 MIoU 점수를 달성했습니다. 제로샷 도시 생성을 통해 도시 구조를 정확하게 예측하고 생성할 수 있는 강력한 일반화 능력을 입증했습니다.



### GeoBiked: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design (https://arxiv.org/abs/2409.17045)
- **What's New**: 본 연구에서는 Deep Generative Models (DGM)을 공학 설계에 적용하기 위한 데이터셋 GeoBiked를 제공하며, 대규모 기초 모델을 활용하여 데이터 레이블링 자동화를 위한 방법을 제안합니다. GeoBiked 데이터셋은 4,355개의 자전거 이미지를 포함하고 있으며, 구조적 및 기술적 특징으로 주석이 달려 있습니다.

- **Technical Details**: GeoBiked 데이터셋은 이미지 생성 모델에서 추출한 통합 잠재 특징(Hyperfeatures)을 사용하여 구조적 이미지의 기하학적 대응 관계(예: 바퀴 중심 위치)를 검출하는 두 가지 자동 레이블링 기술을 조사합니다. 또한 GPT-4o를 통해 기술 이미지에 대한 다양한 설명을 생성합니다. 기술 이미지를 Diffusion-Hyperfeatures로 표현하여 기하학적 대응 관계를 측정할 수 있습니다.

- **Performance Highlights**: GeoBiked 데이터셋을 기반으로 한 두 가지 자동 레이블링 방법은, 잠재 이미지 특징의 학습된 통합을 통해 보지 못한 이미지에서 기하학적 기준점을 정확히 예측할 수 있음을 보여주며, GPT-4o를 통해 생성된 다양한 텍스트 설명은 정확한 기술 이미지를 설명합니다. 이러한 접근은 기술 이미지의 일반적인 포인트 검출 및 주석 작업에 적용 가능한 방법으로 제안됩니다.



### EventHDR: from Event to High-Speed HDR Videos and Beyond (https://arxiv.org/abs/2409.17029)
Comments:
          TPAMI 2024

- **What's New**: 이 논문은 이벤트 카메라에서 생성된 이벤트 시퀀스를 활용하여 고속 HDR 비디오를 재구성하는 획기적인 기술적 접근 방식을 제시합니다. 재귀적 합성곱 신경망(recurrent convolutional neural network)을 사용하여 키 프레임 가이드를 통해 정보 손실을 방지하며, 실세계 데이터셋을 새롭게 수집하여 연구의 기초를 다집니다.

- **Technical Details**: 이 연구는 스파스 이벤트 데이터를 통해 영상 재구성이 발생할 수 있는 오류 누적을 방지하기 위해 키 프레임 가이드를 제공하는 재귀 신경망(RNN)을 제안합니다. 또한, 피라미드 변형 네트워크(pyramidal deformable network)를 사용하여 연속적인 이벤트 프레임 간의 기능을 정렬하여 시간적 일관성을 향상시키는 방법을 도입합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 고속 HDR 비디오 재구성에서 최첨단 성능을 달성했으며, 실제 쌍 데이터셋을 이용하여 다양한 컴퓨터 비전 작업, 예를 들면, 물체 탐지, 파노라마 분할, 광학 흐름 추정 및 단안 깊이 추정에서 우수한 성능을 발휘했습니다.



### Enhanced Wavelet Scattering Network for image inpainting detection (https://arxiv.org/abs/2409.17023)
- **What's New**: 이 연구는 이미지 인페인팅의 진위를 감지하기 위해 저수준 노이즈 분석에 기반한 혁신적인 방법들을 제안합니다. 특성 추출을 위한 Dual-Tree Complex Wavelet Transform (DT-CWT)와 변조 영역 감지 및 위치 지정을 위한 합성곱 신경망 (CNN)을 결합하여 접근합니다.

- **Technical Details**: DT-CWT는 전송 불변성 (shift-invariance)을 제공하여 인페인팅 과정에서 미세 조작에 대한 강인성을 증가시키고, 방향 선택성은 특정 주파수 대역 및 방향에서 인페인팅으로 인해 발생하는 미세한 아티팩트를 감지하는 데 도움을 줍니다. 또한, 텍스처 분할과 노이즈 분산 추정 (noise variance estimation)을 결합하여 변조된 영역을 찾는 융합 감지 모듈을 제안합니다.

- **Performance Highlights**: 우리의 접근 방식은 최첨단 방법들과 비교하여 모든 인용된 대안보다 우수한 성능을 보임을 입증했습니다. 트레이닝 코드와 사전 훈련된 모델 가중치는 제공된 URL에서 이용 가능할 것입니다.



### PTQ4RIS: Post-Training Quantization for Referring Image Segmentation (https://arxiv.org/abs/2409.17020)
- **What's New**: 이 논문은 Referring Image Segmentation (RIS) 모델에 대한 새로운 포스트-트레이닝 양자화(post-training quantization) 프레임워크인 PTQ4RIS를 제안합니다. PTQ4RIS는 자원 제한적인 엣지 디바이스에서의 실제 응용을 위한 효율적이고 효과적인 솔루션을 제공합니다.

- **Technical Details**: PTQ4RIS는 Dual-Region Quantization (DRQ) 및 Reorder-based Outlier-retained Quantization (RORQ) 방법을 통해 RIS 모델의 양자화 시 성능 저하를 해결합니다. DRQ는 비주얼 인코더의 활성화 분포를 다루며, RORQ는 텍스트 인코더에서 아웃라이어(outlier)의 영향을 최소화합니다.

- **Performance Highlights**: PTQ4RIS는 RefCOCO+ testB 데이터셋에서 FP 모델과 동등한 성능을 기록하며, W6A6 및 W4A8 설정에서도 성능 저하가 각각 0.66 OIoU와 1.54 OIoU에 불과합니다. 이는 RIS 모델에서 포스트-트레이닝 양자화(PTQ) 도입의 가능성을 강조합니다.



### CNN Mixture-of-Depths (https://arxiv.org/abs/2409.17016)
Comments:
          Conference Paper of the Asian Conference on Computer Vision (ACCV) 2024

- **What's New**: CNN Mixture-of-Depths (MoD) 접근법을 도입하여 CNN의 계산 효율성을 개선하였습니다. 이 방법은 현재 예측과 관련성에 따라 채널을 선택적으로 처리하여 계산 자원을 최적화합니다. MoD는 동적 계산 그래프 없이 정적 계산 그래프를 사용하여 하드웨어 효율성을 높이고, 훈련 및 추론 과정을 가속화합니다.

- **Technical Details**: MoD는 Conv-Blocks 내에서 입력 특성 맵의 각 채널의 중요성을 평가한 후, 상위 k개 채널을 선택하여 처리합니다. 이를 통해 처리할 채널 수를 조정하고, 계산 부하를 줄이며, 최종적으로 처리된 채널과 원본 특성 맵의 첫 k개 채널을 융합하여 특징 표현을 강화합니다.

- **Performance Highlights**: ResNet86-MoD는 표준 ResNet50을 능가하며 CPU에서 6%, GPU에서 5% 더 빠른 처리 속도를 제공합니다. ResNet75-MoD는 ResNet50과 동일한 성능을 유지하며 CPU에서 25%, GPU에서 15%의 속도 향상을 보여줍니다.



### Adverse Weather Optical Flow: Cumulative Homogeneous-Heterogeneous Adaptation (https://arxiv.org/abs/2409.17001)
- **What's New**: 이 연구는 악천후에서의 광학 흐름 문제를 해결하기 위해 누적 동질-이질 적응 프레임워크인 CH2DA-Flow를 제안합니다. 이 방법은 깨끗한 장면에서 악천후에 이르는 지식 전이를 위한 중간 단계로 합성 열화 도메인을 활용합니다.

- **Technical Details**: 제안된 CH2DA-Flow 프레임워크는 두 가지 주요 과정으로 구성됩니다: 깨끗한-열화 동작 적응(CDMA)과 합성-실제 동작 적응(SRMA). CDMA에서는 정적 날씨에 대해 깊이 연관 동질 동작 적응을, 동적 날씨에 대해 왜곡 오차 이질 경계 적응을 설계했습니다. SRMA에서는 합성 및 실제 열화 이미지를 비용 볼륨(cos volume) 공간으로 변환하고 K-L 발산을 사용하여 두 도메인 간의 동질적 상관 값의 전반적인 거리를 측정합니다.

- **Performance Highlights**: 제안된 방법은 다양한 악천후 조건에서의 성능이 우수함을 입증하기 위해 광범위한 실험을 수행했으며, 새로운 실제 악천후 데이터셋이 수작업으로 주석 처리된 광학 흐름 레이블과 함께 제공됩니다. 여러 이동 객체와 다양한 장면을 포함해 동적인 환경에서도 효과적으로 작동함을 확인했습니다.



### Single Image, Any Face: Generalisable 3D Face Generation (https://arxiv.org/abs/2409.16990)
- **What's New**: 새로운 모델 Gen3D-Face를 제안하여, 제약 없는 단일 이미지를 통해 3D 인간 얼굴 아바타를 생성하는 것을 가능하게 함. 이 모델은 기존 방법들이 겪던 일반화 문제를 해결하기 위해 다중 보기 일관성(diffusion framework)을 통해 작동함.

- **Technical Details**: Gen3D-Face는 한 개의 얼굴 이미지를 입력으로 받아 다중 보기 이미지를 생성하고, 그 후 신경 표면 건축(neural surface construction)을 수행함. 입력 조건에 따라 메쉬 추정(input-conditioned mesh estimation)을 활용하여 모델의 일반화를 도모하고, 다양한 외형 스타일을 가진 중복들을 처리함. 멀티 뷰 조인트 생성(multi-view joint generation) 방식을 도입하여 보기 간의 일관성을 높임.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 기존 방법들 대비 우수한 성능을 보이며, 특히 새로운 도메인에서의 단일 이미지 3D 얼굴 생성에서 성능이 월등함을 입증함. 이는 다양한 상황에서 현실적인 3D 얼굴 아바타 생성을 위한 혁신적인 접근법임.



### Path-adaptive Spatio-Temporal State Space Model for Event-based Recognition with Arbitrary Duration (https://arxiv.org/abs/2409.16953)
Comments:
          First version

- **What's New**: 이 연구에서는 이벤트 카메라의 비동기적 데이터 흐름을 이용하여 객체 및 행동 인식의 새로운 프레임워크인 PAST-SSM을 제안합니다. 이 프레임워크는 임의의 지속 시간(0.1초에서 4.5분)에서 이벤트를 인식하고 다양한 추론 주파수에 일반화하는 능력을 갖추었습니다.

- **Technical Details**: PAST-SSM 프레임워크는 Path-Adaptive Event Aggregation and Scan (PEAS) 모듈을 사용하여 다양한 지속 시간의 이벤트를 고정된 차원의 특징으로 인코딩합니다. 또한 Multi-faceted Selection Guiding (MSG) 손실을 도입하여 인코딩된 특징의 무작위성과 중복을 최소화합니다. 이로 인해 모델의 일반화 능력이 향상됩니다. 최종적으로 상태 공간 모델(SSM)을 활용하여 인코딩된 특징의 시공간 특성을 학습합니다.

- **Performance Highlights**: 실험 결과, PAST-SSM 모델은 DVS Action, SeAct, HARDVS 데이터셋에서 각각 +3.45%, +0.38%, +8.31%의 성능 향상을 보였으며, ArDVS100, Real-ArDVS10 및 TemArDVS 데이터셋에서 각각 97.35%, 100.00%, 89.00%의 Top-1 정확도를 달성했습니다. 또한, 다양한 추론 주파수에 대해 최대 8.62%의 성능 저하를 보이며, 이전 샘플링 방법의 27.59%에 비해 뛰어난 일반화 성능을 발휘했습니다.



### DALDA: Data Augmentation Leveraging Diffusion Model and LLM with Adaptive Guidance Scaling (https://arxiv.org/abs/2409.16949)
Comments:
          Accepted to ECCV Synthetic Data for Computer Vision Workshop (Oral)

- **What's New**: 이 논문에서는 데이터가 부족한 상황에서의 문제를 해결하기 위해 Large Language Model (LLM)과 Diffusion Model (DM)을 활용하는 데이터 증강 프레임워크를 제안합니다. 이 방법은 LLM을 통해 생성된 텍스트 프롬프트에 새로운 의미 정보를 삽입하고, 실제 이미지를 시각적 프롬프트로 사용하여 의미적으로 풍부한 이미지를 생성하도록 합니다.

- **Technical Details**: 우리의 접근 방식은 다중 모달 조건부 Diffusion Model (MMDM)에서 예제 이미지와 텍스트 프롬프트의 영향을 조화롭게 균형 잡는 Adaptive Guidance Scaling (AGS) 메커니즘을 포함합니다. CLIPScore를 기반으로 이미지 생성을 위한 텍스트 프롬프트의 가중치를 동적으로 조정하여 생성된 이미지가 목표 분포를 벗어나지 않도록 합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 방법들보다 높은 다양성과 개선된 분류 모델 성능을 보여주며, 특히 몇 가지 샷(few-shot) 설정에서 효율적임을 입증했습니다.



### NTIRE 2024 Challenge on Stereo Image Super-Resolution: Methods and Results (https://arxiv.org/abs/2409.16947)
- **What's New**: 이번 논문은 3차 NTIRE 스테레오 이미지 초해상도(SR) 챌린지를 요약하고 새로운 솔루션 및 결과에 초점을 맞추었습니다. 이 챌린지의 목표는 저해상도 스테레오 이미지 쌍을 x4 배율로 고해상도로 변환하는 것입니다.

- **Technical Details**: 이번 챌린지는 bicubic degradation과 실제 열화(real degradation) 두 가지 트랙으로 구성되어 있습니다. 총 108명과 70명의 참가자가 각각 성공적으로 등록했고, 테스트 단계에서 14개 팀과 13개 팀이 PSNR(RGB) 점수가 기준을 초과하여 유효한 결과를 제출했습니다. 이 챌린지는 스테레오 이미지 SR을 위한 새로운 벤치마크를 설정했습니다.

- **Performance Highlights**: NTIRE 2024 챌린지는 주어진 계산 제약 하에서 SR의 최첨단을 측정하고 이를 한계까지 밀어붙이는 것을 목표로 하고 있으며, 참가자들은 이번 대회를 통해 새로운 접근법을 비교할 수 있습니다.



### Face Forgery Detection with Elaborate Backbon (https://arxiv.org/abs/2409.16945)
- **What's New**: 이번 연구에서는 Face Forgery Detection (FFD) 모델의 일반화 성능을 개선하기 위해 Backbone의 사전 훈련(parallel training) 및 미세 조정(fine-tuning) 과정을 재조명했습니다. 기존의 모델들이 Backbone의 중요성을 간과한 반면, 본 연구에서는 ViT(Visual Transformer) 네트워크와 자가 지도 학습(self-supervised learning)을 통한 Backbone 개발을 제안하였습니다.

- **Technical Details**: FFD 모델을 위한 새로운 Backbone 구조를 구현했으며, 다양한 포뮬레이션을 통해 강력한 얼굴 특징 표현(capacities)을 할 수 있는 능력을 확보했습니다. 또한, 신뢰성 있는 추론(inference)을 위해 예측 신뢰도(prediction confidence)를 활용한 임계값 최적화(mechanism) 기법을 도입했습니다.

- **Performance Highlights**: 종합적인 실험을 통해 본 연구에서 제안한 FFD 모델이 기존 모델들보다 우수한 FFD 및 프레젠테이션 공격 탐지(presentation attack detection) 성능을 달성했음을 입증했습니다. 코드는 해당 링크에서 확인할 수 있습니다.



### Generative Object Insertion in Gaussian Splatting with a Multi-View Diffusion Mod (https://arxiv.org/abs/2409.16938)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 Gaussian Splatting으로 표현된 3D 콘텐츠에 새로운 객체를 삽입하는 혁신적인 방법을 제안합니다. 이 방법은 MVInpainter라는 다중 뷰 확산 모델(multi-view diffusion model)을 기반으로 하여, 사전 학습된 안정적인 비디오 확산 모델을 활용하여 보기 일관성(view-consistent)을 보장하는 객체 인핑팅을 제공합니다.

- **Technical Details**: MVInpainter의 핵심은 ControlNet 기반의 조건부 주입 모듈을 통합하여 보다 통제되고 예측 가능한 다중 뷰 생성을 가능하게 하는 것입니다. 이 모델은 원본 3D 씬과 대조 모델에서 배경, BBox(Bounding Box) 수준의 마스크 및 깊이 맵을 추출하여, 입력으로 세 가지 세트를 사용해 목표 객체 설명에 맞춰 인핑팅 결과를 생성합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존 방법들보다 뛰어난 성과를 보임을 입증하였습니다. 우리의 접근 방식은 다양한 결과를 생성하고, 보기 일관성을 보장하며, 더 나은 객체 품질을 제공합니다.



### Game4Loc: A UAV Geo-Localization Benchmark from Game Data (https://arxiv.org/abs/2409.16925)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 영상 기반 UAV (Unmanned Aerial Vehicle) 지리 위치 추정 기술을 개선하기 위한 대규모 데이터셋인 GTA-UAV를 구축하였습니다. 이 데이터셋은 실제 사용 사례를 반영하기 위해 드론-위성 이미지 쌍의 부분 일치를 포함하도록 설계되었습니다.

- **Technical Details**: GTA-UAV 데이터셋은 현대의 컴퓨터 게임을 활용하여 다양한 비행 고도, 자세, 장면 및 타겟을 포함하여 전방위 연속 영역에서 수집한 33,763개의 드론-뷰 이미지로 구성됩니다. 또한, 데이터 쌍의 학습을 위해 가중치 기반 대비 학습 방식인 weighted-InfoNCE를 도입하여 부분 일치 이미지를 효과적으로 학습할 수 있도록 하였습니다.

- **Performance Highlights**: 이 연구 결과, 제안된 데이터셋과 학습 방법이 UAV 지리 위치 추정에서 효과적임을 입증했으며, 실제 상황에 대한 일반화 능력도 확인되었습니다. 이러한 접근은 기존의 완벽한 일치 데이터셋의 한계를 극복하고, 보다 현실적인 운영 환경에서의 적용 가능성을 높였습니다.



### An Adaptive Screen-Space Meshing Approach for Normal Integration (https://arxiv.org/abs/2409.16907)
- **What's New**: 본 연구에서는 이미지 도메인에서 적응형 표면 삼각분할(Adaptive Surface Triangulation)을 도입하여 삼각형 메쉬(triangle mesh)에서 법선(normal) 통합(normal integration)을 수행하는 새로운 방법을 제안합니다. 이 접근은 기존의 픽셀 그리드(pixel grids)보다 표면 세부사항에 적응하여 표현을 스파스하게 만들어, 계산 효율성을 크게 개선합니다.

- **Technical Details**: 연구의 핵심 통찰은 법선에서 표면 곡률(curvature)을 계산할 수 있다는 것이고, 이를 통해 평탄한 영역(flat areas)을 식별하고 픽셀을 삼각형으로 집계합니다. 사용자는 단일 매개변수를 통해 근사 품질을 조절할 수 있으며, 64 MP 노말 맵(normal maps)에서 메쉬 생성 및 통합을 수행하는데 몇 분이 소요되는 반면, 기존의 픽셀 기반 접근법은 수시간이 소요됩니다.

- **Performance Highlights**: 실제 및 합성 데이터에서 실험 결과, 법선 통합을 위해 필요한 정점(vertex)의 수가 픽셀보다 10배에서 100배 적게 요구됨을 보여주었습니다. 또한 이 스파스성은 픽셀 수에 대한 하위선형(sublinear) 런타임으로 이어짐을 시사합니다.



### Towards Underwater Camouflaged Object Tracking: An Experimental Evaluation of SAM and SAM 2 (https://arxiv.org/abs/2409.16902)
Comments:
          Preprint. Work in Progress

- **What's New**: 이번 논문에서는 UW-COT라는 첫 번째 대규모 수중 위장 물체 추적 데이터셋을 제안하고, 이 데이터셋을 기반으로 여러 최신 시각 물체 추적 방법의 실험적 평가를 수행하였습니다.

- **Technical Details**: UW-COT 데이터셋은 96개 카테고리로 구성된 220개의 수중 비디오 시퀀스를 포함하며, 각 프레임에 대해 위장된 물체에 대한 바운딩 박스 주석을 제공합니다. 이 데이터셋을 통해 SAM(Segmentation Anything Model)과 SAM 2의 성능을 비교하였으며, SAM 2는 시간적 일관성, 신뢰성, 기능 임베딩, 컴퓨팅 효율성 및 신규 도메인 일반화 능력이 개선되었습니다.

- **Performance Highlights**: SAM 2는 UW-COT 데이터셋에서 SAM 기반 추적기(SAM-DA 및 Tracking Anything)보다 뛰어난 성능을 보였으며, 현재의 최신 VOT 방법들보다 우수한 성능을 기록하였습니다. 이는 SAM 2가 비디오 데이터의 동적 도전 과제를 해결하기 위한 향상된 솔루션을 제공한다는 것을 보여줍니다.



### HVT: A Comprehensive Vision Framework for Learning in Non-Euclidean Spac (https://arxiv.org/abs/2409.16897)
- **What's New**: 이번 논문에서는 하이퍼볼릭 기하 (hyperbolic geometry)를 통합한 새로운 비전 트랜스포머 (Vision Transformer) 모델인 하이퍼볼릭 비전 트랜스포머 (Hyperbolic Vision Transformer, HVT)를 제안합니다. 전통적인 비전 트랜스포머가 유클리드 공간 (Euclidean space)에서 작동하는 반면, 하이퍼볼릭 비전 트랜스포머는 하이퍼볼릭 거리와 뫼비우스 변환 (Möbius transformations)을 활용하여 계층적 관계를 더 효과적으로 모델링합니다.

- **Technical Details**: 하이퍼볼릭 비전 트랜스포머는 비전 데이터 내의 계층적 및 관계적 의존성을 최적화하기 위해 하이퍼볼릭 신경 구성 요소 (Hyperbolic Neural Components)와 뫼비우스 변환을 적용합니다. 이 모델은 하이퍼볼릭 공간에서 작동하도록 신경망의 구성 요소, 예를 들어 어텐션 메커니즘 (attention mechanisms) 및 피드-포워드 네트워크 (feed-forward networks)를 확장합니다. 또한, 수학적 구조를 통해 어텐션 레이어와 최적화에서 하이퍼볼릭 기하를 적용하는 방법을 제시합니다.

- **Performance Highlights**: 이미지넷 데이터셋 (ImageNet dataset)을 사용한 실험 결과, 하이퍼볼릭 비전 트랜스포머는 이미지 분류 (image classification) 작업에서 성능이 향상되었음을 보여 주었습니다. 특히, 전통적인 유클리드 접근 방식에 비해 계층적 구조를 더 잘 모델링할 수 있는 능력을 입증하였습니다.



### Linking in Style: Understanding learned features in deep learning models (https://arxiv.org/abs/2409.16865)
- **What's New**: 이 논문은 CNN(Convolutional Neural Networks)에서 학습된 특성을 체계적으로 분석하고 시각화할 수 있는 자동화된 방법을 제안합니다. 특히, 사전 학습된 분류기의 곧 전 단계(또는 penultimate layer)를 생성 모델(StyleGAN-XL)의 잠재 공간(latent space)에 매핑하는 링크 네트워크(linking network)를 도입하여 분류기의 표현을 해석할 수 있는 시각화를 가능하게 합니다.

- **Technical Details**: 우리의 방법은 두 가지 단계로 구성됩니다. 첫째, 사전 학습된 StyleGAN-XL에 기반한 효율적인 특징 시각화 도구를 구축하여 여러 사전 학습된 CNN과 유연하게 연결할 수 있습니다. 둘째, 비지도 추적 방법(unsupervised tracking methods)과 소수의 샷 이미지 분할(few-shot image segmentation)을 활용하여 분류기의 표현 공간에서 학습한 개념을 자동으로 평가합니다. 이를 통해 각 유닛의 특성을 분석하고 요약 통계(summary statistics)를 작성할 수 있습니다.

- **Performance Highlights**: 우리는 제안하는 방법을 통해 단일 유닛에서 학습된 추상적 개념을 밝히고, 분류기의 결정 경계를 분석하여 분류에 있어 가장 중요한 특징을 해석할 수 있음을 보여줍니다. 링크 네트워크는 훈련이 쉬우며, GAN 및 분류기 훈련과는 분리되어 효율적으로 학습됩니다.



### Towards Unified 3D Hair Reconstruction from Single-View Portraits (https://arxiv.org/abs/2409.16863)
Comments:
          SIGGRAPH Asia 2024, project page: this https URL

- **What's New**: 본 논문에서는 단일 시점(single-view)에서 다양한 머리 스타일에 대한 3D 헤어 재구성을 가능하게 하는 새로운 전략을 제안합니다. 복잡한 머리 스타일을 처리할 수 있도록 고안된 통합 파이프라인을 통해, 손상된 머리 스타일을 복원하는 데 있어 기존의 한계를 극복했습니다. 또한, 새로운 합성 데이터셋 SynMvHair를 구축하여 다양한 스타일의 3D 헤어 재구성을 위한 기초를 마련했습니다.

- **Technical Details**: 제안한 접근법은 2D diffusion priors을 활용한 coarse-to-fine 최적화 기반 방법으로, Gaussian 기반의 3D 헤어 표현을 사용합니다. 이 방법은 view-wise와 pixel-wise Gaussian refinement 두 가지 모듈을 최적화하여 고품질의 텍스처를 제공합니다. HairSynthesizer와 HairEnhancer라는 두 가지 diffusion-based hair priors를 통해, 단일 시점 이미지를 조건으로 하여 세밀한 3D 헤어를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 다양한 머리 스타일을 단일 시점 이미지에서 재구성할 수 있는 가능성을 보여주었고, 복잡한 머리 스타일을 복원하는 데 있어 state-of-the-art 성능을 달성했습니다. 특히, 실제 이미지에 대한 강력한 일반화 능력을 입증하였고, 고속으로 고품질의 멀티 뷰 헤어 렌더링을 가능하게 합니다.



### Limitations of (Procrustes) Alignment in Assessing Multi-Person Human Pose and Shape Estimation (https://arxiv.org/abs/2409.16861)
- **What's New**: 이 논문은 비디오 감시 시나리오에서 3D 인간 포즈와 형상을 정확하게 추정하는 데 있어 새로운 도전 과제를 다룹니다. 기존 메트릭의 한계를 극복하기 위해 RotAvat라는 새로운 기법을 제안합니다.

- **Technical Details**: RotAvat는 3D 메시를 지면과 정렬하는 방법을 개선하여 W-MPJPE 및 W-PVE 메트릭을 정교화합니다. 기존 방법들은 카메라의 시점 변화로 인한 3D 몸체의 글로벌 위치를 제대로 반영하지 못하고 있습니다.

- **Performance Highlights**: RotAvat는 기존 최첨단 방법들(BEV, SPEC, CLIFF)에 비해 복잡한 상황에서도 더 나은 성능을 보여주며, 2D 입력만으로도 안정적인 결과를 도출함으로써 비디오 감시 환경에서 3D 포즈 추정의 정확도를 높입니다.



### The Role of Language Models in Modern Healthcare: A Comprehensive Review (https://arxiv.org/abs/2409.16860)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 의료 분야에 어떻게 적용되는지에 대한 체계적인 리뷰를 제공합니다. LLM의 발전 과정과 의료 적용에서의 강점뿐만 아니라 데이터 프라이버시, 편향, 윤리적 고려사항 등의 문제를 논의합니다.

- **Technical Details**: 대형 언어 모델(LLM)은 Transformer 아키텍처를 기반으로 하여 장거리 의존성을 효과적으로 캡처하는 능력을 가지고 있습니다. 모델은 일반적으로 방대한 텍스트 데이터셋으로 사전 학습(pre-training)된 후, 특정 작업에 맞춰 세부 조정(fine-tuning)됩니다. BioBERT, ClinicalBERT와 같은 의료 전용 모델이 개발되어 임상 언어의 독특한 도전을 해결하고 있습니다.

- **Performance Highlights**: LLM은 의료 데이터 분석, 질병 진단, 환자 관리 및 약물 발견과 같은 다양한 분야에서 사용되고 있습니다. 임상 의사결정 지원 및 의료 문서 요약 등의 임무에 대한 효과적인 증상이 입증되었습니다. 측정 기준으로는 MMLU, HumanEval과 같은 벤치마크가 사용되어 모델의 효과성을 평가합니다.



### A Versatile and Differentiable Hand-Object Interaction Representation (https://arxiv.org/abs/2409.16855)
Comments:
          Accepted at the Winter Applications in Computer Vision 2025 conference. 9 pages, 6 figures

- **What's New**: 이 논문에서는 Coarse Hand-Object Interaction Representation (CHOIR)이라는 새로운 HOI 모델링 필드를 제시했습니다. CHOIR는 완전한 미분 가능성을 가지고 있어 다양한 작업에 응용할 수 있습니다.

- **Technical Details**: CHOIR는 이산적인 비부호 거리(discrete unsigned distances)를 활용하여 연속적인 형태(shape)와 자세(pose)를 인코딩하며, 다변량 가우시안 분포(multivariate Gaussian distributions)를 통해 밀집 접촉 맵(dense contact maps)을 적은 매개변수로 표현합니다. 이 연구에서는 JointDiffusion이라는 확산 모델을 설계하여 노이즈가 있는 손-물체 상호작용(hand-object interactions) 또는 단순한 물체 기하학(Object geometries)을 기반으로 하는 그립(grasp) 분포를 학습합니다.

- **Performance Highlights**: JointDiffusion은 정제(refinement) 작업에서 접촉 F1 점수를 $5\%$ 증가시켰으며, 합성(synthesis) 작업에서 시뮬레이션 변위를 $46\%$ 감소시켰습니다. 실험 결과, CHOIR와 함께 사용하는 JointDiffusion은 특정 작업을 위한 기존 방법에 비해 접촉 정확도(contact accuracy)와 물리적 현실감(physical realism)에서 우수한 성능을 나타냈습니다.



### Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms (https://arxiv.org/abs/2409.16850)
Comments:
          7 pages

- **What's New**: 본 논문에서는 DINOv2라는 비주얼 파운데이션 모델의 강력한 특징 추출 능력을 활용한 새로운 장면 변화 탐지(scene change detection; SCD) 방법을 제안합니다. 본 방법은 강한 조명 변화, 계절적 변화 및 관점의 차이와 같은 주요 과제를 해결하기 위해 전체 이미지에 대한 크로스 주의(full-image cross-attention)를 통합합니다.

- **Technical Details**: 제안하는 방법은 이미지 쌍 간의 일치 및 불일치 사항을 효과적으로 학습하기 위해 1) 백본(backbone) 네트워크를 고정하여 밀집한 파운데이션 특징의 일반성을 유지하고, 2) 관점 차이에 보다 최적으로 대처하기 위해 '전체 이미지' 크로스 주의 방법을 사용합니다.

- **Performance Highlights**: VL-CMU-CD 및 PSCD와 같은 두 가지 벤치마크 데이터셋에서 실험을 진행했으며, 특히 이미지 쌍 간의 기하학적 변화가 포함된 시나리오에서 F1-score가 크게 향상됨을 보였습니다. 결과는 기존의 최첨단 접근 방식에 비해 우리 방법의 뛰어난 일반화 능력을 보여줍니다.



### IRASNet: Improved Feature-Level Clutter Reduction for Domain Generalized SAR-ATR (https://arxiv.org/abs/2409.16845)
Comments:
          16 pages, 11 figures

- **What's New**: 본 연구에서는 SAR-ATR(합성 개구 레이더 자동 목표 인식) 분야에서 도메인 일반화(Domain Generalization)를 위한 새로운 프레임워크인 IRASNet을 제안합니다. 이 프레임워크는 효과적인 특징 수준의 클러터(clutter) 감소와 도메인 불변 특징 학습을 가능하게 하여 기존의 기술적 한계를 극복하고자 합니다.

- **Technical Details**: IRASNet은 1) 클러터 감소 모듈(Clutter Reduction Module, CRM)을 통해 특징 맵에서 신호 대 클러터 비율(Signal-to-Clutter Ratio, SCR)을 최대화하고, 2) 적대적 학습(Adversarial Learning)과 CRM을 통합하여 클러터 감소된 도메인 불변 특징을 추출하며, 3) 마스크 지상 진실(Mask Ground Truth) 인코딩을 사용하여 정책적 감독(Positional Supervision) 작업을 통해 특징 추출을 개선합니다. 이 모든 작업은 측정된 데이터 없이도 이루어집니다.

- **Performance Highlights**: IRASNet은 공개 SAR 데이터셋인 SAMPLE에서 뛰어난 성능을 달성하며, 기존의 SAR-ATR 방법들과 비교할 때 뛰어난 일반화 성능을 보여줍니다. 덧붙여, 특징 수준의 클러터 감소 능력 또한 크게 향상되어 레이더 이미지 패턴 인식 분야에서 중요한 진전을 이루었습니다.



### Explicitly Modeling Pre-Cortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness (https://arxiv.org/abs/2409.16838)
- **What's New**: 이번 연구에서는 이전의 VOneNets를 기반으로 하여 후드 구조를 일반화하기 위해 RetinaBlock을 도입하고, RetinaNets 및 EVNets라는 두 가지 새로운 CNN 모델 패밀리를 소개합니다. 이 모델들은 망막 및 외측 슬상핵(LGN)의 시각처리 단계를 시뮬레이션하여 CNN의 견고성을 개선합니다.

- **Technical Details**: RetinaBlock은 midget과 parasol 망막 신경세포의 시각적 프로세싱을 모델링하여, 여러 개의 병렬 경로를 통해 작동합니다. 관심영역(RF) 내의 공간적 종합을 수행하고, DoG(차이-가우시안) 필터를 통해 주위와의 반응을 모델링하여 대칭세포의 상호작용을 시뮬레이션합니다. RetinaNets는 표준 CNN 백엔드 아키텍처를 통합하며, EVNets는 VOneBlock과 결합하여 동작합니다.

- **Performance Highlights**: RetinaNets는 기존 모델에 비해 12.3%의 상대적 견고성 향상이 관찰되었으며, EVNets는 18.5%의 향상을 보였습니다. 이러한 개선은 다양한 방식의 이미지 왜곡에 대해 일반화되었으나, 깨끗한 이미지 정확도에 약간의 감소가 동반되었습니다.



### Focus Entirety and Perceive Environment for Arbitrary-Shaped Text Detection (https://arxiv.org/abs/2409.16827)
- **What's New**: 이번 연구에서는 다각적 정보 레벨을 활용한 임의 형태 텍스트 탐지기를 제안합니다. 핵심 모듈(FEM)과 주변 환경 모듈(PEM)을 통해 기존의 노이즈에 취약한 하향식 모델링 방식의 문제를 해결합니다.

- **Technical Details**: 제안하는 Focus Entirety Module (FEM)은 몸체 레벨의 특징을 추출하고 상향식의 노이즈 영향 감소 방식으로 텍스트를 모델링합니다. Perceive Environment Module (PEM)은 지역 레벨의 정보를 추출하고, 픽셀 주변의 긍정 샘플 분포를 강조하여 픽셀 상호작용을 촉진합니다.

- **Performance Highlights**: 제안된 FEPE 모델은 공개된 4개의 데이터셋에서 기존의 최첨단 기법을 초월하는 성능을 보여주며, 수평, 회전 및 불규칙 형태의 텍스트를 모두 처리할 수 있는 능력을 입증하였습니다.



### XAI-guided Insulator Anomaly Detection for Imbalanced Datasets (https://arxiv.org/abs/2409.16821)
Comments:
          Accepted as a workshop paper at ECCV 2024

- **What's New**: 이 연구에서는 전선의 절연체 결함을 탐지하기 위한 새로운 파이프라인을 제안합니다. UAV(무인 항공기)를 활용하여 수집한 이미지를 통해 절연체 결함을 정확하게 검출하고 분류하는 방법론을 개발하였습니다.

- **Technical Details**: 제안된 방법은 YOLOv8 모델을 사용하여 절연체를 검출하고, 비율이 불균형한 데이터셋 문제를 해결하기 위해 로지스틱 회귀를 통한 재훈련 기법을 적용하였습니다. 또한, LRP(층별 관련 전파기법)를 활용하여 결함의 위치를 정확히 설명하고 시각화했습니다.

- **Performance Highlights**: 결함 탐지 정확도를 최대 13% 향상시켰으며, 클래스 불균형 문제를 해결하여 예측 유지보수의 효과성을 크게 개선하였습니다. 이 연구는 산업 현장에서의 비전 기반 검사 및 예측 유지 보수에 가치 있는 기여를 하고 있습니다.



### Spotlight Text Detector: Spotlight on Candidate Regions Like a Camera (https://arxiv.org/abs/2409.16820)
- **What's New**: 본 논문에서 제안하는 Spotlight Text Detector (STD)는 불규칙한 형태의 텍스트를 보다 정확하게 감지하기 위해 두 가지 모듈, 즉 Spotlight Calibration Module (SCM)과 Multivariate Information Extraction Module (MIEM)을 포함하여 복잡한 배경에서도 텍스트 감지를 효과적으로 향상시킨다.

- **Technical Details**: SCM은 후보 영역에 초점을 맞춰 예측된 텍스트 커널을 정확하게 보정하여 잘못된 긍정 샘플을 제거하는 방식으로 작동한다. MIEM은 다양한 기하학적 특성을 탐색하여 텍스트의 형태, 크기 및 방향의 다양성에 대처하며, 연산량을 최소화하면서 여러 공간적 관계를 포착한다. 두 모듈은 서로 보완적으로 동작하여 텍스트 감지의 효율성을 극대화한다.

- **Performance Highlights**: 광범위한 실험을 통해 제안한 STD는 ICDAR2015, CTW1500, MSRA-TD500, Total-Text와 같은 다양한 데이터셋에서 기존의 최첨단 방법들 대비 우수한 성능을 보였으며, 특히 불규칙 형상의 텍스트를 효과적으로 감지하는 능력을 지닌 것으로 나타났다.



### Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices (https://arxiv.org/abs/2409.16808)
- **What's New**: 본 논문에서는 YOLOv8, EfficientDet Lite, SSD와 같은 최신 객체 탐지 모델을 자원 제한이 있는 엣지 디바이스에서 성능을 평가하였습니다. Raspberry Pi와 Jetson Orin Nano 등 다양한 엣지 디바이스에서 에너지 소비, 추론 시간, 평균 정밀도(Mean Average Precision, mAP) 등의 성능 지표를 수집하였습니다.

- **Technical Details**: 우리가 연구한 모델들은 YOLOv8 (버전 Nano, Small, Medium), EfficientDet Lite (Lite0, Lite1, Lite2), 그리고 SSD (SSD MobileNet V1, SSDLite MobileDet)입니다. 각 모델은 Raspberry Pi 3,4,5 및 TPU 가속기 사용 여부에 따라 성능 평가를 진행하였고, 전반적으로 객체 탐지를 위한 다양한 머신 러닝 프레임워크(PyTorch, TensorFlow Lite, TensorRT)를 활용하였습니다.

- **Performance Highlights**: YOLOv8 Medium과 같은 높은 mAP 모델은 에너지 소비가 많고 느린 편이지만, SSD MobileNet V1 모델은 에너지 효율성이 뛰어나고 속도가 빠른 것으로 나타났습니다. Jetson Orin Nano는 요청 처리에 있어 가장 빠르고 에너지 효율적이지만 대기 전력 소비가 가장 높은 것으로 분석되었습니다. 이 연구는 엣지 디바이스에 딥러닝 모델을 배포할 때 정확성, 속도 및 에너지 효율성 간의 균형을 고려해야 한다는 점을 강조합니다.



### Topological SLAM in colonoscopies leveraging deep features and topological priors (https://arxiv.org/abs/2409.16806)
Comments:
          MICCAI 2024

- **What's New**: 이번 연구에서는 ColonSLAM을 소개합니다. 이는 기존의 SLAM 시스템에 딥 러닝 기반의 기능과 위상적 프라이어(topological priors)를 결합하여 전체 대장의 위상적 지도를 생성하는 시스템입니다.

- **Technical Details**: ColonSLAM은 고전적인 metric SLAM과 함께 딥 로컬리제이션 네트워크를 활용하여 동일한 지점에서 촬영된 이미지 간의 관계를 식별하고, transformer 기반의 매칭 네트워크를 통해 복잡한 맵을 구축합니다. 이 시스템은 선형 연결만으로 이루어진 작은 metric 서브맵을 조합하여 위상적 맵 G=(N,E) 형태로 생성합니다. 각 노드는 대장 구조의 특정 구역을 나타내며, 간선은 공간에서 연결 가능한 장소를 연결합니다.

- **Performance Highlights**: Endomapper 데이터셋을 통한 평가를 통해 ColonSLAM이 실제 인간 탐사를 통해 전체 대장 지도를 생성할 수 있는 잠재력을 보여주었습니다. 이 연구는 대장 내시경에 있어 기존의 방법들보다 더 복잡한 맵을 구축할 수 있게 하여, 의료 영상 인식 기술에 기여할 것으로 기대됩니다.



### Spacewalker: Traversing Representation Spaces for Fast Interactive Exploration and Annotation of Unstructured Data (https://arxiv.org/abs/2409.16793)
- **What's New**: 이번 논문에서 우리는 다양한 모달리티를 활용한 비구조화 데이터 분석을 위한 상호작용 도구인 Spacewalker를 소개합니다. 이 도구는 데이터를 탐색하고 주석을 달 수 있도록 설계되었습니다.

- **Technical Details**: Spacewalker는 사용자가 임의의 모달리티의 데이터 세트를 업로드하고, Low-dimensional space에서 데이터의 시맨틱 유사성을 강조하여 시각화할 수 있는 기능을 제공합니다. Bayesians networks 및 Deep Learning 기반의 방법을 사용하여 데이터를 추출합니다.

- **Performance Highlights**: 사용자 연구 결과, Spacewalker는 기존 방법에 비해 데이터 주석 속도를 현저히 개선하며, 데이터 무결성 검증 및 부패 데이터 세트 식별 작업에서도 빠른 탐색이 가능합니다.



### MixPolyp: Integrating Mask, Box and Scribble Supervision for Enhanced Polyp Segmentation (https://arxiv.org/abs/2409.16774)
Comments:
          Accepted in IEEE BIBM 2024

- **What's New**: 본 논문에서는 기존의 주석 방식의 한계를 극복하기 위해 다양한 주석 유형을 결합한 혼합 감독 기법인 MixPolyp를 제안합니다. 기존 방식이 단일 주석 유형에 의존하는 것과 달리, MixPolyp는 픽셀, 박스, 스크리블 주석을 통합하여 데이터의 가용성을 높이고 레이블링 비용을 감소시키는 데 초점을 맞춥니다.

- **Technical Details**: MixPolyp는 세 가지 새로운 감독 손실 함수를 도입하여 다양한 주석을 처리합니다. 1) Subspace Projection loss (L_SP): 예측과 박스 주석 간의 형태 불일치를 제거합니다. 2) Binary Minimum Entropy loss (L_BME): 레이블이 없는 픽셀에 대한 제어를 제공하여 감독의 희소성을 완화합니다. 3) Linear Regularization loss (L_LR): 예측 간의 일관성을 보장하여 비고유성(non-uniqueness)을 줄입니다. 이 손실들은 모델 구조에 구애받지 않으며, 훈련 중에만 사용되어 추론 시 계산 비용이 없습니다.

- **Performance Highlights**: 다섯 개의 데이터셋에 대한 광범위한 실험을 통해 MixPolyp의 효과성은 입증되었습니다. MixPolyp는 기존의 완전 감독 결과를 초월하여 고품질의 폴립 마스크를 예측하는 성능을 보여줍니다.



### MaViLS, a Benchmark Dataset for Video-to-Slide Alignment, Assessing Baseline Accuracy with a Multimodal Alignment Algorithm Leveraging Speech, OCR, and Visual Features (https://arxiv.org/abs/2409.16765)
- **What's New**: 이 논문에서는 강의 비디오와 해당 슬라이드를 정렬하는 벤치마크 데이터셋을 제시하고, 음성, 텍스트 및 이미지에서 특징을 활용하는 새로운 다중 모달 알고리즘을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 동적 프로그래밍(dynamic programming)을 사용하여 최적의 슬라이드 시퀀스를 결정하며, OCR(Optical Character Recognition)을 통해 얻은 특징들이 매칭 정확도에 크게 기여한다고 보고합니다. 알고리즘은 SIFT(Scale-Invariant Feature Transform)에 비해 평균 0.82의 정확도를 기록하면서 속도는 약 11배 빨라졌습니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 강의 스타일과 비디오 품질에 따른 매칭 정확도의 차이를 보였으며, 슬라이드 전환을 제재할 경우 정확도가 향상되었습니다. 또한, 매칭의 정확도는 오디오 전사에 의해서도 유용한 정보를 제공하고, OCR 데이터가 부족할 때 더욱 유용함을 강조합니다.



### Statewide Visual Geolocalization in the Wild (https://arxiv.org/abs/2409.16763)
- **What's New**: 이 연구에서는 राज्य 규모의 검색 영역 내에서 자연에서 촬영한 거리 전경 사진의 지리적 위치를 예측하는 방법을 제시합니다. 항공 이미지를 데이터베이스와 비교하여 거리 전경 사진의 정확한 위치를 정의할 수 있는 모델을 훈련시켰습니다.

- **Technical Details**: 이 방법은 검색 영역을 지리적 셀로 분할하고, 각 셀과 해당 사진을 결합된 임베딩 공간에 매핑하여 테스트 시 검색을 수행합니다. 다양한 수준의 세부 정보(Levels of Detail)를 활용하여 주변 장면에 대한 충분한 정보를 제공합니다. 새로운 검색 지역 레이아웃을 설계하여 대규모 지역으로 확장할 수 있습니다.

- **Performance Highlights**: 이 방법은 매사추세츠주에 업로드된 모든 비파노라마 거리 전경 사진의 60.6%를 실제 위치로부터 50m 이내로 정확하게 지역화하는 데 성공하였습니다.



### Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics (https://arxiv.org/abs/2409.16756)
- **What's New**: 이번 논문에서는 LATEC이라는 대규모 벤치마크를 소개합니다. 이 벤치마크는 17개의 주요 Explainable AI (XAI) 방법을 20가지 메트릭으로 평가하여, 다양한 설계 매개변수와 모델 아키텍처를 통합하여 7,560개의 조합을 체계적으로 분석합니다.

- **Technical Details**: LATEC을 통해 XAI 방법의 평가에서 자주 발생하는 메트릭 간의 상충 가능성을 발견하였으며, 이는 신뢰할 수 없는 순위를 초래할 수 있습니다. 논문에서는 보다 강력한 평가 방안을 제안하며, 다양한 XAI 방법을 종합적으로 평가하여 실무자들이 필요에 맞는 적절한 방법을 선택할 수 있도록 지원합니다. 또한 LATEC은 326k saliency maps와 378k metric scores를 공공 데이터셋으로 제공하여 향후 XAI 연구에 기여할 수 있도록 합니다.

- **Performance Highlights**: 특히, Expected Gradients라는 새로운 고성능 방법이 기존 연구에서 검토되지 않았음을 발견하였으며, 이는 LATEC의 중요한 발견 중 하나로, 향후 XAI 연구 방향에 중대한 영향을 미칠 것으로 예상됩니다.



### Commonly Interesting Images (https://arxiv.org/abs/2409.16736)
Comments:
          ECCV 2024

- **What's New**: 본 논문은 개인의 주관적 취향이 관여하는 시각적 흥미로움(visual interestingness) 개념을 형식적으로 정의하고, 이미지의 공통적 흥미(common interest) 요소를 밝혀내기 위해 2.5천 명의 Flickr 사용자로부터 수집한 500k 이미지를 분석합니다.

- **Technical Details**: FlickrUser-dataset을 기반으로 이미지의 특성을 분석하며, perceptual, denotative, connotative 특징을 포함합니다. Bottom-up 처리와 top-down 처리 간의 상호작용을 통해 흥미로움이 형성된다는 이론을 제시합니다. 또한 공통적 흥미와 주관적 흥미의 연속성을 제안하며 컴퓨터 모델 학습에 활용하였습니다.

- **Performance Highlights**: 전문적으로 촬영된 경관 이미지들은 큰 공통적 흥미를 유발하는 반면, 개인적 사건을 담은 이미지는 주관적 흥미를 자아내며 개인의 기억과 감정을 자극하는 경향이 있습니다.



### EAGLE: Towards Efficient Arbitrary Referring Visual Prompts Comprehension for Multimodal Large Language Models (https://arxiv.org/abs/2409.16723)
- **What's New**: 이번 논문에서는 EAGLE이라는 새로운 Multimodal Large Language Model (MLLM)을 제안하여, 임의의 referring visual prompts를 이해하는 능력을 향상시킵니다. EAGLE는 기존 모델들보다 훈련 노력을 줄이면서도 효과적으로 다양한 형태의 시각적 프롬프트를 처리할 수 있도록 설계되었습니다.

- **Technical Details**: EAGLE는 주어진 이미지에 색깔이 입혀진 패치로 referring visual prompts를 렌더링하여 이미지 자원을 활용합니다. 또한 Geometry-Agnostic Learning (GAL) 패러다임을 도입하여 다양한 형태의 referring visual prompts와의 관계를 분리합니다. 이로 인해 MLLM이 강조된 객체를 인식하는 데 더 집중할 수 있도록 합니다.

- **Performance Highlights**: EAGLE는 다양한 임의의 시각적 프롬프트를 처리하는 데 있어 기존 최첨단 방법들보다 더 효율적으로 작동하며, 실험 결과에서 더욱 향상된 성능을 보여줍니다. 또한, 우리의 방법은 기존의 지역 텍스트 정렬 프로세스를 새롭게 시작하는 것보다, 시각적 프롬프트를 효과적으로 제시하는 것이 훨씬 효율적임을 입증하였습니다.



### Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification (https://arxiv.org/abs/2409.16718)
Comments:
          EMNLP 2024 Main Conference

- **What's New**: 최근 Vision-Language Models (VLMs)에 대한 세밀한 조정이 이루어지면서, 클립 모델의 고유한 매개변수를 조정하는 것의 중요성을 재조명하였다. 이 연구에서는 모든 매개변수를 조정하는 대신 특정 매개변수만을 조정하는 CLIPFit 방법을 제안하였다.

- **Technical Details**: CLIPFit은 기존의 프롬프트 튜닝(prompt tuning) 및 어댑터 튜닝(adapter tuning) 방식과는 다르게, 추가적인 매개변수를 도입하지 않고 클립 모델의 특정 바이어스와 정규화 레이어만 조정하는 방법이다. 이로 인해 파라미터 수가 줄어들고, 성능이 향상된다.

- **Performance Highlights**: CLIPFit을 사용하여 zero-shot CLIP 대비 7.33%의 평균 조화 평균 정확도(harmonic mean accuracy) 개선을 달성하였으며, 이는 16-shot 설정에서 프롬프트 튜닝 및 어댑터 튜닝을 대체할 수 있는 유망한 옵션이다.



### Pose-Guided Fine-Grained Sign Language Video Generation (https://arxiv.org/abs/2409.16709)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Sign Language Video Generation (SLVG) 분야에서 새로운 Pose-Guided Motion Model (PGMM)을 제안하여 고품질의 시각적 세부 사항과 일관된 동작을 가진 수화 비디오 생성의 한계를 극복하고자 합니다.

- **Technical Details**: 이 방법은 Coarse Motion Module (CMM)과 Pose Fusion Module (PFM)으로 구성되며, CMM을 통해 광학 흐름 왜곡(optical flow warping)을 활용하여 거친 구조의 동작을 완성하고, PFM은 RGB와 포즈 모달리티 정보를 결합하여 세부 사항 생성을 정교하게 진행합니다. 또한 새로운 Temporal Consistency Difference (TCD) 메트릭을 설계하여 비디오의 일관성을 정량적으로 평가합니다.

- **Performance Highlights**: 실험 결과, 제안하는 PGMM은 기존의 최첨단 방법들보다 뛰어난 성능을 보였으며, 다양한 비디오 테스트에서 높은 세부 사항 및 동시성의 개선을 보여주었습니다.



### Pix2Next: Leveraging Vision Foundation Models for RGB to NIR Image Translation (https://arxiv.org/abs/2409.16706)
Comments:
          19 pages,12 figures

- **What's New**: Pix2Next는 RGB 이미지를 기반으로 고해상도 NIR 이미지를 생성하는 혁신적인 이미지-이미지 변환 프레임워크입니다. 이 방법은 최신 Vision Foundation Model (VFM)을 활용하여_encoder-decoder_ 아키텍처에서 크로스-어텐션 메커니즘(cross-attention mechanism)을 통합하여 특징 통합을 향상시킵니다. 더불어, 여러 해상도에서 현실적인 이미지 생성을 보장하는 PatchGAN 판별자를 사용하여 NIR 이미지 생성의 품질과 세부사항을 개선합니다.

- **Technical Details**: Pix2Next은 RGB 이미지를 NIR 이미지로 변환하는 과정에서 고유한 세부 사항과 스펙트럼 특성을 유지하는 데 중점을 두고 설계되었습니다. 실제로, Segmentation과 Object Detection 태스크에 대한 성능 평가와 함께 다양한 손실 함수가 글로벌 컨텍스트 이해와 로컬 특징 보존을 연결하여 모델 성능을 높입니다. 또한	RANUS 데이터셋을 사용하여 테스트를 진행하였습니다.

- **Performance Highlights**: Pix2Next는 FID(Frechet Inception Distance) 점수를 기존 방법에 비해 34.81% 향상시켜, 세 가지 시각 품질 지표에서 뛰어난 성능을 보였습니다. 또한, NIR 이미지로의 변환된 데이터를 이용하여 자율 주행 인식 태스크에서 더욱 개선된 성능을 보여주어, 제한된 실 NIR 데이터셋을 보완하는 데 있어 효용성을 입증했습니다.



### Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Mod (https://arxiv.org/abs/2409.16689)
Comments:
          Accepted by ECCV2024, Project Page: this https URL

- **What's New**: 이 논문은 기존의 Discrete Diffusion Models (DDMs)에서 발생하는 레이아웃 고착(Layout Sticking) 현상을 해결하기 위해 Layout-Corrector라는 새롭고 간단한 모듈을 제안합니다.

- **Technical Details**: Layout-Corrector는 레이아웃의 각 요소에 대한 정확성 점수를 평가하고, 저조한 점수를 가진 요소를 재초기화하여 하모니가 있는 레이아웃을 생성을 돕습니다. 이 모듈은 DDM과 함께 사용되며, 각 생성 과정에서 하모니를 고려하여 불일치하는 요소를 식별합니다.

- **Performance Highlights**: Layout-Corrector는 다양한 기준 벤치마크에서 테스트되어 DDM과 함께 사용할 경우 레이아웃 생성 성능을 일관되게 향상시키고, 정확성-다양성 무역의 조절을 통한 성능 저하를 완화합니다.



### Skyeyes: Ground Roaming using Aerial View Images (https://arxiv.org/abs/2409.16685)
- **What's New**: 이번 논문에서는 Skyeyes라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 항공 이미지 데이터만을 사용하여 지상 시점의 포토리얼리스틱(photorealistic) 이미지를 생성할 수 있어, 자율주행 및 게임 디자인 등에서 사실감을 동반한 3D 환경을 구현합니다.

- **Technical Details**: Skyeyes는 3D 표현과 시점 일관성 생성 모델을 결합하여 생성된 이미지 간의 일관성을 보장합니다. SuGaR를 활용해 정밀한 세부 정보를 유지하고, 지상 뷰 이미지 생성을 위한 카메라 포즈를 학습합니다. 생성 과정에서 주어진 항공 이미지를 바탕으로 appearance control 모듈을 통해 픽셀 정확성을 유지하며, 마지막으로 view consistency 모듈을 통해 시간적 일관성을 확보합니다.

- **Performance Highlights**: 대규모의 합성 및 지리적으로 정렬된 데이터셋을 구축하고 다양한 실험을 통해 기존 기법들보다 우수한 성능을 보였습니다. 정성적 및 정량적 분석에서 탁월한 결과를 나타내며, 합성 기술의 최전선에 위치하고 있습니다. 개발한 코드와 데이터셋은 논문 수락 후 공개될 예정입니다.



### TalkinNeRF: Animatable Neural Fields for Full-Body Talking Humans (https://arxiv.org/abs/2409.16666)
Comments:
          Accepted by ECCVW 2024. Project page: this https URL

- **What's New**: TalkinNeRF라는 새로운 프레임워크를 소개하며, 이는 모노큘러 비디오로부터 전신 인간의 동적 neural radiance field (NeRF)를 학습합니다. 기존 연구들은 신체 자세나 얼굴만 표현했으나, 이 방법에서는 신체 전체를 아우르는 메시지를 전달하는 데 필요한 모든 요소를 통합합니다.

- **Technical Details**: TalkinNeRF는 모노큘러 비디오에서 인간의 4D 모션을 표현하는 통합된 NeRF 기반 네트워크 입니다. 신체, 얼굴, 손에 대해 각각의 모듈을 학습하고, 복잡한 손가락 움직임을 표현하기 위해 추가적인 변형 필드도 학습합니다. 또한, 다중 정체성 표현을 통해 여러 주체에 대한 동시에 학습이 가능합니다.

- **Performance Highlights**: 기존 최첨단 기술을 능가하여 동적인 전신 인간 애니메이션을 생성하는 데 있어 우수한 성과를 보여줍니다. TalkinNeRF는 핸드 아티큘레이션(hand articulation)과 얼굴 표정(facial expressions)을 활용하여 새로운 포즈에 대해서도 견고한 애니메이션을 만들어냅니다.



### Progressive Representation Learning for Real-Time UAV Tracking (https://arxiv.org/abs/2409.16652)
Comments:
          Accepted by the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 이 논문에서는 UAV(무인 항공기) 추적을 위한 새로운 점진적 표현 학습 프레임워크인 PRL-Track을 제안합니다. PRL-Track은 거친 표현 학습과 정밀 표현 학습의 두 부분으로 나뉘며, 특히 복잡한 동적 환경에서의 객체 추적 성능을 개선하는 데 중점을 둡니다.

- **Technical Details**: PRL-Track은 CNN(합성곱 신경망)을 기반으로 하는 거친 표현 학습과 ViT(비전 트랜스포머)를 기반으로 하는 정밀 표현 학습을 통합합니다. 거친 표현 학습에서는 외관 정보와 의미론적 정보를 이용한 두 개의 혁신적인 조절기(regulator)를 사용하는데, 이는 외관 간섭을 완화하고 깊은 특징에서 의미 정보를 캡처합니다. 정밀 표현 학습에서는 새로운 계층적 모델링 생성기가 도입되어 객체의 거친 표현을 연결합니다.

- **Performance Highlights**: 종합 실험에 따르면 PRL-Track은 세 개의 권위 있는 UAV 추적 벤치마크에서 우수한 성능을 보여주었습니다. 실제 테스트 결과, PRL-Track은 일반적인 UAV 플랫폼에서 초당 42.6 프레임으로 뛰어난 추적 성능을 실현하여 효율성과 강인성을 입증했습니다.



### Enhancing Nighttime UAV Tracking with Light Distribution Suppression (https://arxiv.org/abs/2409.16631)
- **What's New**: 이 논문은 LMEnhancer라는 새로운 저조도 이미지 향상 기술을 제안하여, 야간 드론(UAV) 추적의 효과성을 높이고자 합니다. 기존의 저조도 이미지 향상 기법들이 복잡한 조명 조건에서의 불균형한 조명 분포를 간과하는 문제를 해결하기 위해, 이 작업은 조명 분포 억제를 통한 향상을 도모합니다.

- **Technical Details**: LDEnhancer는 이미지 콘텐츠 정보와 조명 분포 정보를 특징 공간에서 분리하여 목표 지향적인 향상을 가능하게 하는 새로운 이미지 콘텐츠 정제 모듈을 개발합니다. 또한, 두 개의 파라미터 맵을 활용하여 저조도 이미지의 픽셀 단위 조정을 위한 혁신적인 순회 반복 조정을 제안합니다. 이 연구실에서는 40개의 시퀀스와 74K 이상의 프레임으로 구성된 새로운 야간 UAV 추적 데이터셋 NAT2024-2도 구축하였습니다.

- **Performance Highlights**: LDEnhancer는 기존의 저조도 향상 기법에 비해 야간 UAV 추적에서 우수한 성능을 보여주었으며, 권위있는 UAV 벤치마크에서 검증된 강건성을 통해 실세계 테스트에서도 효율성과 실용성을 입증했습니다.



### DeformStream: Deformation-based Adaptive Volumetric Video Streaming (https://arxiv.org/abs/2409.16615)
- **What's New**: 비대칭적 볼륨 비디오 스트리밍 (Volumetric Video Streaming)의 성능을 향상시키기 위해, 기하학적 변형의 유용성을 활용한 새로운 프레임워크인 Deformation-based Adaptive Volumetric Video Streaming을 소개합니다. 이 방법은 메쉬 기반 표현의 본질적 변형 가능성을 활용하여, 새로운 프레임을 이전 프레임의 모션에서 재구성함으로써 대역폭 사용량을 크게 줄이는 동시에 각 프레임 간 시각적 일관성을 보장합니다.

- **Technical Details**: DeformStream은 인코더, 네트워크 적응, 디코더의 세 가지 주요 구성 요소로 나뉘며, GoF(Group of Frames) 개념을 기반으로 메쉬 시퀀스를 청크 방식으로 스트리밍합니다. 각 I-프레임은 최초의 메쉬 데이터와 앵커 노드 그래프를 포함하고, 후속 P-프레임은 각 노드의 변형 행렬만 포함합니다. 이러한 방식은 메쉬 간의 상관관계를 유지하여 데이터 전송을 최적화합니다.

- **Performance Highlights**: Deformation-based Adaptive Volumetric Video Streaming은 기존의 메쉬 기반 스트리밍 시스템보다 대역폭 효율성과 시각적 품질에서 모두 뛰어난 성능을 보이며, 실시간 볼륨 비디오 애플리케이션을 위한 강력한 솔루션을 제공합니다.



### Semi-LLIE: Semi-supervised Contrastive Learning with Mamba-based Low-light Image Enhancemen (https://arxiv.org/abs/2409.16604)
- **What's New**: 새로운 Semi-LLIE(Mean Teacher 기반의 준지도 저조도 이미지 향상) 프레임워크가 제안되었습니다. 이 프레임워크는 비구조적 데이터(unpaired data)를 모델 학습에 통합하여 저조도 이미지 향상의 성능을 향상시킵니다.

- **Technical Details**: Semi-LLIE는 semantic-aware contrastive loss와 Mamba 기반 저조도 이미지 향상 백본(backbone)을 활용하여 이미지를 향상시킵니다. contrastive loss는 이미지의 조명 분포를 정확하게 전달하여 자연스러운 색상을 구현하고, Mamba는 다중 스케일 특징 학습을 통해 로컬 지역의 픽셀 관계 표현을 강화합니다.

- **Performance Highlights**: Semi-LLIE는 Visdrone 및 LRSW 데이터셋에서 기존 최첨단(SOTA) 비지도 방법보다 뛰어난 성능을 보이며, 감지 태스크의 성능 향상에도 기여합니다.



### FAFA: Frequency-Aware Flow-Aided Self-Supervision for Underwater Object Pose Estimation (https://arxiv.org/abs/2409.16600)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 자율 수중 차량(Unmanned Underwater Vehicles, UUVs)의 6D 자세 추정(pose estimation)을 위한 Frequency-Aware Flow-Aided(self-supervised) 프레임워크인 FAFA를 제시합니다. 이 프레임워크는 합성 데이터(예: synthetic data)에서 학습한 후 실제 수중 환경에 적응하는 방식으로 작동합니다.

- **Technical Details**: FAFA는 두 단계로 구성된 self-supervised 프레임워크로서, 첫 번째 단계에서는 Fast Fourier Transform(FFT)을 기반으로 하는 데이터 증강(data augmentation) 방법으로 RGB 이미지에서 도메인 불변 특징(domain-invariant features)을 추출합니다. 두 번째로, multi-level flow-aided consistencies를 통해 이미지와 피처(feature) 수준에서 정렬(alignment)을 강제하여 네트워크의 성능을 향상시킵니다.

- **Performance Highlights**: 연구자는 FAFA가 수중 6D 객체 자세 벤치마크에서 현재의 최첨단(state-of-the-art) 방법보다 현저한 성능 향상을 보였음을 입증했습니다. FAFA는 추가적인 실제 세계의 감독 신호없이도 뛰어난 성능을 자랑합니다.



### EventHallusion: Diagnosing Event Hallucinations in Video LLMs (https://arxiv.org/abs/2409.16597)
- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 비디오 이해 분야에서 큰 진전을 이루었습니다. 본 논문은 EventHallusion이라는 새로운 벤치마크를 제안하여 비디오 이벤트 이해에서 VideoLLMs의 홀루시네이션(hallucination) 문제를 평가합니다.

- **Technical Details**: EventHallusion은 VideoLLMs의 홀루시네이션 현상을 평가하기 위해 비디오와 질문을 수집하고 주의 깊게 주석을 달아, 모델이 비디오 콘텐츠를 정확하게 이해하는 대신 기존 선험적(prior) 지식에 기반해 이벤트를 해석하도록 유도합니다. 또한, Temporal Contrastive Decoding (TCD)이라는 단순하면서도 효과적인 방법을 통해 비디오 LLM의 홀루시네이션 문제를 해결합니다. TCD는 원본 비디오와 생성된 비디오를 비교하여 모델의 선험적 경향을 억제합니다.

- **Performance Highlights**: 제안된 EventHallusion 벤치마크에서 8개의 오픈소스 및 2개의 클로즈드소스 VideoLLMs를 종합적으로 평가한 결과, 오픈소스 모델은 심각한 홀루시네이션 문제를 겪는 반면, 클로즈드소스 모델은 훨씬 더 나은 성능을 보였습니다. TCD 방법으로 오픈소스 VideoLLMs를 보강함으로써 대부분의 메트릭에서 성능이 개선되었습니다.



### SelectiveKD: A semi-supervised framework for cancer detection in DBT through Knowledge Distillation and Pseudo-labeling (https://arxiv.org/abs/2409.16581)
Comments:
          10 pages, 2 figures, 1 table

- **What's New**: 이번 논문에서는 Digital Breast Tomosynthesis (DBT)용 컴퓨터 보조 탐지(CAD) 시스템에 대한 새로운 반지도 학습 프레임워크인 SelectiveKD를 제안합니다. 이 프레임워크는 제한된 수의 주석이 달린 슬라이스(slices)를 이용하여 높은 성능을 달성할 수 있도록 설계되었습니다.

- **Technical Details**: SelectiveKD는 Knowledge Distillation (KD) 개념을 활용하여 주석이 없는 DBT 슬라이스를 이용하는 접근 방식을 제공합니다. 주석이 달린 슬라이스로 학습된 teacher 모델이 student's 모델에 감독 신호를 전송하여 전체 DBT 볼륨의 슬라이스에 대한 교육을 받습니다. 이는 Pseudo Labels (PL)을 사용하여 데이터 세트를 선택적으로 확장하는 방법을 통해 노이즈 문제를 완화합니다.

- **Performance Highlights**: 10,000건 이상의 DBT exams로 구성된 대규모 실제 데이터셋을 통해 검증된 SelectiveKD는 주석이 없는 슬라이스를 효과적으로 활용하여 암 분류 성능(AUC)과 일반화 성능을 유의미하게 개선하였습니다. 이 방식은 대량의 주석 확보 비용을 절감하며 다양한 제조업체 간의 일반화 능력을 유지합니다.



### Source-Free Domain Adaptation for YOLO Object Detection (https://arxiv.org/abs/2409.16538)
Comments:
          ECCV 2024: European Conference on Computer Vision - Workshop on Out-of-Distribution Generalization in Computer Vision Foundation Models, Milan Italy

- **What's New**: 본 논문에서는 Object Detection(OD)의 Source-Free Domain Adaptation(SFDA) 분야에서 YOLO 계열의 단일 단계 탐지기를 향상시키는 새로운 방법인 Source-Free YOLO(SF-YOLO)를 제안합니다.

- **Technical Details**: SF-YOLO는 Teacher-Student 프레임워크를 기반으로 하여, 학생 모델이 특정 타겟 도메인에 대한 학습된 데이터 증강 기법을 통해 훈련됩니다. 이 방법은 레이블이 없는 타겟 데이터만을 사용하며 기능 정렬(Feature Alignment)을 요구하지 않습니다. 또한, 새로운 Student Stabilisation Module(SSM)을 도입하여 훈련의 안정성을 높이고, 레이블이 없는 상황에서의 정확도 저하 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SF-YOLO는 Cityscapes, Foggy Cityscapes, Sim10k, KITTI 데이터셋에서 여러 도전적인 도메인 적응 벤치마크에서 현재의 최고 성능을 보이는 탐지기들과 경쟁할 수 있으며, 심지어는 소스 데이터를 사용하는 적응 방법보다 나은 성능을 기록하기도 하였습니다. 저희의 접근법은 낮은 계산 자원을 요구하며, 실용적인 실시간 응용에 적합합니다.



### Prompt Sliders for Fine-Grained Control, Editing and Erasing of Concepts in Diffusion Models (https://arxiv.org/abs/2409.16535)
Comments:
          ECCV'24 - Unlearning and Model Editing Workshop. Code: this https URL

- **What's New**: 이번 논문에서는 이미지 생성 및 편집을 위한 Diffusion 모델의 새로운 접근 방식인 Factor Graph-DMs (FG-DMs)를 제안합니다. 이 방식은 기존의 방법보다 더 정밀하게 이미지 속성을 제어할 수 있는 가능성을 열어줍니다.

- **Technical Details**: FG-DMs는 이미지와 조건 변수를 모델링하기 위한 새로운 프레임워크로, 모듈화된 구조를 통해 다양한 시스템에서 작동할 수 있도록 지원합니다. FG-DM은 기존의 Stable Diffusion (SD) 모델을 기반으로 하여 조건적 변수의 분포를 학습하며, Attention Distillation Loss를 통해 생성된 조건의 신뢰성을 높입니다.

- **Performance Highlights**: FG-DM은 CelebA-HQ, ADE20K, Cityscapes 및 COCO 데이터세트에서 훈련된 결과 고품질의 이미지를 생성하며, 낮은 FID 점수와 높은 LPIPS 점수를 기록하여 이미지 다양성이 증가하는 성과를 보였습니다. 또한, Prompt Sliders 방법을 통해 새로운 개념을 학습하는 동시에 원하지 않는 개념을 삭제하는 작업이 가능하여 30% 빠른 속도를 자랑합니다.



### Low Latency Point Cloud Rendering with Learned Splatting (https://arxiv.org/abs/2409.16504)
Comments:
          Published at CVPR 2024 Workshop on AIS: Vision, Graphics and AI for Streaming (this https URL)

- **What's New**: 이번 연구는 점 구름(Point Cloud)을 사용하여 실시간 고화질 렌더링을 가능하게 하는 새로운 프레임워크를 제안합니다. 이는 동적 점 구름을 실시간으로 렌더링할 수 있는 능력을 갖추고 있으며, 기존의 렌더링 솔루션보다 우수한 품질과 속도를 자랑합니다.

- **Technical Details**: 제안된 메소드는 Point-to-Ellipsoid (P2ENet)이라는 경량 3D 희소 합성곱 신경망을 활용하여 색상이 있는 점 구름의 각 점을 3D 타원으로 변환합니다. 이후 이 타원을 스플랫팅하여 현재 관점에서의 부드러운 질감과 표면 법선을 렌더링합니다. 이러한 방식은 각 장면에 대한 최적화를 필요로 하지 않으며, 고품질의 렌더링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 100 FPS 이상의 속도로 고화질의 구멍 없는 이미지를 렌더링할 수 있으며, 초기 지연이 30 ms 미만인 것으로 확인되었습니다. 또한 환경 잡음에 대해 강력한 내성을 보입니다.



### GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization (https://arxiv.org/abs/2409.16502)
Comments:
          Project website at this https URL

- **What's New**: 본 연구에서는 3D Gaussian Splatting (3DGS) 기술을 활용하여 시각적 로컬라이제이션(visual localization)을 향상시키는 새로운 프레임워크 GSplatLoc을 제안합니다. 이 방법은 기존의 메모리 소모나 최적화 요구 사항을 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: GSplatLoc은 XFeat의 경량 키포인트 감지 및 기술 모델로 생성된 고밀도 기술 맵을 활용하여 3DGS에 밀집된 키포인트 설명자(dense keypoint descriptors)를 증류(distill)합니다. 이를 통해 공간 이해도를 개선하고, 2D-3D 대응(relations)을 통해 더 정확한 카메라 포즈 예측을 가능하게 합니다. 초기 포즈 추정 후에는 포토메트릭 왜곡 손실(photometric warping loss)을 사용하여 포즈를 세분화(refine)합니다.

- **Performance Highlights**: 이번 연구는 인기 있는 실내 및 실외 데이터셋에서 벤치마킹한 결과, 기존의 최첨단 Neural Render Pose (NRP) 방법들, 특히 NeRFMatch와 PNeRFLoc을 능가하는 성과를 보여주었습니다.



### Real-Time Detection of Electronic Components in Waste Printed Circuit Boards: A Transformer-Based Approach (https://arxiv.org/abs/2409.16496)
Comments:
          International Conference on Applications in Electronics Pervading Industry, Environment and Society (ApplePies2024). Proceedings are published in the Springer Lecture Notes in Electrical Engineering

- **What's New**: 본 논문은 Waste Printed Circuit Boards (WPCBs)에서 Critical Raw Materials (CRMs)인 구리, 망간, 갈륨 등의 농도를 높여 효율적으로 추출하기 위한 방법으로 전자 부품의 선택적 분해를 제안합니다. 이를 위해 인공지능 비전 기술에 기반한 메카트로닉 시스템을 사용하여 전자 부품을 실시간으로 감지하고 위치를 추적하는 데 중점을 두었습니다.

- **Technical Details**: 연구진은 Real-Time DEtection TRansformer (RT-DETR) 모델 아키텍처를 사용하여 전자 부품 감지 및 위치 추적의 실시간 정확성을 평가했습니다. Transformer 아키텍처는 이미지의 특징을 추출하기 위해 CNN 백본을 사용하며, 이후 Transformer 인코더와 디코더를 통해 객체 요청을 처리하고 예측합니다. 논문에서는 V-PCB라는 커스텀 데이터셋을 사용하여 실험을 진행하였고, Mean Average Precision (mAP) 메트릭스를 통해 성능을 평가하였습니다.

- **Performance Highlights**: RT-DETR 모델은 최신 YOLOv8 및 YOLOv9 모델에 비해 우수한 성능을 기록했으며, 이는 인공지능 비전 기술의 장점을 활용하여 WPCBs의 CRMs 추출을 효율적으로 수행할 수 있음을 보여줍니다. 여러 가지 전자 부품에 대한 감지 작업을 통해 대량의 부품을 신속하고 정확하게 분류할 수 있는 가능성을 제시합니다.



### A Unified Hallucination Mitigation Framework for Large Vision-Language Models (https://arxiv.org/abs/2409.16494)
Comments:
          Accepted by TMLR

- **What's New**: 본 논문에서는 다양한 형태의 환각(hallucination) 문제를 해결하기 위해, 쿼리(query)를 분류하고 이를 기반으로 다양한 환각 완화(mitigation) 과정을 수행하는 통합 프레임워크인 Dentist를 제안합니다. 이를 통해 환각의 종류에 따라 각기 다른 접근방법을 적용할 수 있습니다.

- **Technical Details**: Dentist 프레임워크는 먼저 쿼리를 인식(perception)과 추론(reasoning)으로 분류하고, 각 쿼리 유형에 맞는 처리 방법을 사용합니다. 구체적으로, 감지 쿼리에 대한 생성 결과는 부차적 질문(sub-questions)을 통해 검증되며, 추론 쿼리의 경우 체인 오브 생각(Chain-of-Thought, CoT)을 활용해 검증됩니다. 이 검증 루프는 정밀도 향상을 위해 수차례 반복됩니다.

- **Performance Highlights**: MMbench에서 InstructBLIP, LLaVA, VisualGLM과 같은 기존 기법에 비해 이미지 품질 관점에서 13.44%, 10.2%, 15.8%의 정확도 향상을 달성했습니다. 또한, 우리의 방법은 다양한 비주얼 언어 작업에서 효과적이고 우수함을 입증하였습니다.



### Proactive Schemes: A Survey of Adversarial Attacks for Social Good (https://arxiv.org/abs/2409.16491)
Comments:
          Submitted for review

- **What's New**: 본 논문은 컴퓨터 비전 분야에서 적대적 공격(adversarial attack)이 머신 러닝 모델의 취약점을 악용하는 방식과 이에 대한 방어적 접근으로 사회적 이익을 추구하는 방법을 다룹니다. 특히, 새로운 proactive schemes를 통해 입력 데이터를 암호화하고 심층 학습 모델의 성능을 향상시키는 방안을 제시합니다.

- **Technical Details**: 적대적 공격은 입력 데이터에 미세한 섭동(perturbation)을 추가하여 잘못된 예측이나 분류를 유도합니다. 본 연구에서는 'templates'이라 불리는 추가 신호를 사용하여 입력 데이터를 암호화하고, 이를 통해 다양한 컴퓨터 비전 및 자연어 처리(natural language processing) 응용 프로그램에 적용할 수 있는 proactive schemes를 설명합니다. 이러한 방법론은 전통적인 passive schemes와 달리 입력 데이터 분포를 변경하지 않고 성능을 개선할 수 있습니다. 또한, 매체의 보안을 유지하며 원본과 비교하여 품질을 보장하는 암호화 및 학습 과정에 대해 설명합니다.

- **Performance Highlights**: proactive schemes는 다양한 응용 분야에서의 성능을 향상시킬 수 있는 잠재력을 가지고 있습니다. 예를 들어, 이미지 개선, 인공지능 생성 모델(GenAI) 및 대형 언어 모델(LLM) 방어, 저작권 보호 및 개인 정보 보호 등의 응용이 가능합니다. 이 논문은 다양한 template 유형 및 그에 따른 암호화 과정, 학습 목표를 탐구하며, 향후 발전 방향과 함께 현재의 한계점에 대해 논의합니다.



### Frequency-based View Selection in Gaussian Splatting Reconstruction (https://arxiv.org/abs/2409.16470)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 연구에서는 3D Gaussian Splatting을 사용한 3차원 재구성을 위한 능동적인 시점 선택 문제를 다루고 있습니다. 기존의 방법들이 특정 장면에 일반화하는 데 어려움이 있었던 반면, 본 연구는 주파수 영역에서 가능성 있는 뷰를 순위 매김하여 새로운 시점의 정보 이득을 효과적으로 추정할 수 있는 방법을 제안합니다. 이로 인해 제한된 수의 입력 이미지로도 효율적인 3D 재구성이 가능합니다.

- **Technical Details**: 이 알고리즘은 처음에 몇 개의 이미지를 입력받고, Gaussian Splatting 모델의 렌더링 결과를 바탕으로 방문할 시점을 능동적으로 생성합니다. 이 방식은 기존 모델 아키텍처 및 효율성의 제약을 극복하여 3D-GS 모델에 맞춤화된 카메라 뷰 선택 파이프라인을 생성합니다. SfM(Structure-from-Motion) 알고리즘을 통해 카메라 포즈를 계산하고, 스프레드 포인트 클라우드를 기반으로 3D Gaussians를 생성합니다.

- **Performance Highlights**: 제안된 방법은 데이터셋에서 단 1/3의 뷰로 합리적인 렌더링 결과를 도출하며, 뷰포인트 간의 이동 거리도 크게 줄였습니다. 이는 3D 재구성을 위한 능동적인 뷰 선택에서 최첨단 성과를 나타냅니다.



### Underground Mapping and Localization Based on Ground-Penetrating Radar (https://arxiv.org/abs/2409.16446)
- **What's New**: 본 논문은 Ground Penetrating Radar (GPR) 데이터를 활용한 심층 신경망 기반의 포물선 신호 감지 네트워크를 소개합니다. GPR 센서의 B-scan 이미지를 사용하여 지하 객체의 3D 재구성과 점 군 지도 생성에 기여하며, 기존의 단일 작업 기반 알고리즘을 다중 작업 네트워크로 발전시켰습니다.

- **Technical Details**: 제안하는 ParNet은 GPR B-scan 데이터에서 주요 포인트를 감지하고, 포물선 방정식을 적합하여 지하 객체의 단면 깊이를 계산합니다. GPRNet은 희소 점 군을 세분화 및 보완하여 밀집된 3D 점 군을 생성합니다. 또한, NetVLAD을 통해 A-scan 데이터에서 특성을 추출하여 알려지지 않은 위치에서의 로컬라이징을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GPR 데이터를 기반으로 한 지하 객체 위치 추정 및 3D 재구성이 효과적임을 증명하였으며, 복잡한 지하 환경에서 점 군의 보완 및 정확한 매칭을 통한 정확성을 향상시켰습니다.



### Hand Gesture Classification Based on Forearm Ultrasound Video Snippets Using 3D Convolutional Neural Networks (https://arxiv.org/abs/2409.16431)
Comments:
          Accepted to IUS 2024

- **What's New**: 본 연구에서는 3D CNN 기술을 활용하여 손 동작 인식을 위한 초음파 비디오 세그먼트에서 시공간(spatiotemporal) 패턴을 캡처합니다. 기존의 2D CNN을 사용한 연구와는 달리, 연속 손 동작에 따른 초음파 데이터의 시간적 특성을 반영하여 제안된 모델의 성능을 향상시켰습니다.

- **Technical Details**: 이 연구에서는 3명의 피실험자가 12개의 손 동작을 수행하는 동안 촬영된 초음파 데이터를 사용했습니다. (2+1)D 컨볼루션 신경망 모델이 기존의 2D 및 3D CNN 모델과 비교되었습니다. 데이터 전처리 과정에서 모션 캡처 시스템을 사용해 실제 손가락 각도를 계산하고, 초음파 이미지를 크기가 224x224로 크롭하여 학습에 적합하도록 변환했습니다.

- **Performance Highlights**: 제안된 모델은 손 동작 분류 정확도를 96.5%에서 98.8%로 향상시켰으며, 이는 초음파 비디오 스니펫을 사용하여 손 동작 분류 성능을 개선하는데 있어 뛰어난 장점을 보여줍니다.



### Leveraging Local Structure for Improving Model Explanations: An Information Propagation Approach (https://arxiv.org/abs/2409.16429)
- **What's New**: 최근 심층 신경망(DNN) 모델의 결정 해석을 위한 다양한 설명 방법들이 개발되었으며, 본 논문에서는 IProp이라는 새로운 방법을 제안합니다. IProp은 각 픽셀의 기여도를 독립적으로 평가하는 대신, 이웃 픽셀과의 구조적 유사성을 고려해 공동으로 평가합니다.

- **Technical Details**: IProp은 각 픽셀의 기여도를 설명 정보의 소스로 모델링하며, Markov Reward Process(MRP)를 통해 모든 픽셀 간의 정보 전파를 다이나믹하게 처리합니다. 정보 전파는 연속적으로 발생하며, 픽셀 간의 상관관계를 포착합니다.

- **Performance Highlights**: IProp은 다양한 DNN 모델과 기존 설명 방법에 대한 실험을 통해 해석 가능성 메트릭에서 현저한 개선을 확인했으며, 정량적 및 정성적으로 모든 기준 방법보다 우수한 결과를 보여주었습니다.



### Improving Intersession Reproducibility for Forearm Ultrasound based Hand Gesture Classification through an Incremental Learning Approach (https://arxiv.org/abs/2409.16415)
Comments:
          Accepted to IUS 2024

- **What's New**: 이번 연구는 초음파 (ultrasound)를 이용한 손 제스처 분류 (hand gesture classification)에서 모델을 여러 세션에 걸쳐 훈련 (training)하여 일반화하는 방법을 제안했습니다. 이를 통해 초음파 프로브를 이동하거나 교체한 경우에도 정확도를 유지할 수 있는 모델을 개발할 수 있었습니다.

- **Technical Details**: 이 연구에서는 CNN (Convolutional Neural Network)을 사용하여 데이터 수집 세션에서 수집된 초음파 이미지를 학습했습니다. 순차적으로 5개의 convolution 레이어와 후속된 dense 레이어를 사용하여 5가지 손 제스처를 분류했습니다. 또한, 모델의 상위 레이어는 동결되어 기능 추출기에 따라 조정되었고, 낮은 레이어는 새로운 데이터를 기반으로 점진적으로 학습되었습니다.

- **Performance Highlights**: 연구 결과, 2회의 점진적 fine tuning 세션 후 모델의 분류 정확도가 약 10% 증가했습니다. vanilla 모델의 평균 정확도는 85.4%였고, fine tuning 후 1회차에서는 93.8%, 2회차에는 95.5%로 상승했습니다. 이러한 결과는 모델의 정확도가 향상되었음을 보여주며, 데이터 저장 공간과 처리 능력을 절약할 수 있음을 나타냅니다.



### Camera Calibration and Stereo via a Single Image of a Spherical Mirror (https://arxiv.org/abs/2409.16386)
Comments:
          12 pages, 11 figures

- **What's New**: 이 논문은 구면 거울(spherical mirror)을 사용하는 단일 뷰(camera view)를 통해 카메라 보정을 위한 새로운 기법을 제시합니다. 이 기법은 이미지에서 볼 수 있는 구의 윤곽과 그 반사를 활용하여 정밀한 보정을 달성하는 효과를 보여줍니다.

- **Technical Details**: 저자들은 카다이옵트릭(stereo imaging) 시스템에서 단일 구면 거울을 이용한 방법을 다루고 있으며, 카메라 매트릭스(calibration matrix) 보정과 카다이옵트릭 스테레오를 동시에 수행합니다. 이 연구는 카메라의 반사, 구면 거울에 있는 두 쌍의 대응 점 또는 특별한 경우의 단일 대응을 통하여 구의 중심 및 윤곽을 찾는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과는Synthetic 및 실제 데이터 모두를 포함하여 제안된 접근법의 실행 가능성 및 정확성을 입증합니다. 해당 연구는 단순한 카다이옵트릭 스테레오 시스템 개발에 기여할 수 있는 잠재력을 보여줍니다.



### Towards Synthetic Data Generation for Improved Pain Recognition in Videos under Patient Constraints (https://arxiv.org/abs/2409.16382)
Comments:
          Pain Recognition Synthetic Data Video Analysis Privacy Preserving

- **What's New**: 이 연구는 비디오 기반의 통증 인식을 향상시키기 위해 합성 데이터(synthetic data)를 활용하는 혁신적인 접근 방식을 소개합니다. 기존의 데이터 수집 방식이 윤리적 및 물리적 도전과제를 포함하는 반면, 본 연구에서는 작은 데이터 세트를 통해 3D 얼굴 모델을 합성하여 다양한 시점에서 통증 표정을 반영합니다. 이를 통해 8,600개의 합성 얼굴을 생성하고, 실제 데이터와 결합하여 통증 인식 모델의 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 고급 얼굴 캡처 기법을 사용하고, CelebV-HQ 및 FFHQ-UV와 같은 공개 데이터셋을 활용하여 인구 통계학적 다양성을 확보했습니다. 참여자의 얼굴을 대체하여 개인 정보를 보호하며, 3D CNN을 적용하여 비디오 기반의 통증 인식 모델을 훈련하는데 필요한 합성 데이터셋을 생성했습니다.

- **Performance Highlights**: 합성 데이터로 훈련한 모델이 실제 참가자의 소량 데이터와 함께 사용될 때, 통증 인식 성능이 현저히 향상되었습니다. 본 접근 방식은 데이터 부족과 윤리적 문제를 해결하며, 프라이버시를 유지하는 데이터셋 생성의 새로운 연구 방향을 제공합니다. 모든 리소스는 공개되어 있어 이 분야에서의 혁신을 촉진할 수 있도록 하고 있습니다.



### Instance Segmentation of Reinforced Concrete Bridges with Synthetic Point Clouds (https://arxiv.org/abs/2409.16381)
Comments:
          33 pages, 12 figures, Submitted to "Automation in Construction"

- **What's New**: 위 논문에서는 다리 구조물의 요소 수준 검사 자동화를 위한 새로운 접근 방식을 제안합니다. 이 접근 방식은 세 가지 독특한 방법을 사용하여 합성 데이터를 생성하며, 기존 연구에서 부족했던 인스턴스 세분화(instance segmentation)에 초점을 맞춥니다.

- **Technical Details**: 제안된 프레임워크는 Mask3D transformer 모델을 활용하며 하이퍼파라미터 조정(hyperparameter tuning)과 새로운 차폐 기법(occlusion technique)으로 최적화됩니다. 이 모델은 실제 LiDAR와 포토그래메트리(photogrammetry)로 수집된 다리 포인트 클라우드에서 최첨단 성능을 달성합니다.

- **Performance Highlights**: 이 연구는 요소 수준 다리 검사 자동화를 위한 프레임워크의 가능성을 보여주며, 더욱 포괄적인 상태 문서화를 통해 전체 다리 관리의 향상을 기대할 수 있습니다.



### Development and Application of a Sentinel-2 Satellite Imagery Dataset for Deep-Learning Driven Forest Wildfire Detection (https://arxiv.org/abs/2409.16380)
- **What's New**: 본 연구에서는 Google Earth Engine(GEE)에서 소싱한 양 시점의 Sentinel-2 위성 이미지를 활용하여 10만 개 이상의 레이블이 부착된 산불 전후 이미지 쌍으로 이루어진 California Wildfire GeoImaging Dataset(CWGID)을 구축하여 딥러닝(DL)을 통한 산불 탐지를 위해 기여하고자 하였습니다.

- **Technical Details**: CWGID는 고해상도 위성 이미지 데이터셋으로, 데이터 획득은 권위 있는 출처에서 이루어졌으며, 세 가지 사전 훈련된 Convolutional Neural Network(CNN) 아키텍처를 활용하여 초기 데이터세트 분석이 진행되었습니다. 특히 EF EfficientNet-B0 모델이 산불 탐지에서 92% 이상의 정확도를 달성하였습니다.

- **Performance Highlights**: CWGID와 이를 구축하는 방법론은 DL 아키텍처 훈련 및 테스트를 위한 귀중한 자원으로 작용하며, 모델 훈련 및 평가 시 높은 정확도와 낮은 손실을 기록하였습니다. 본 연구는 산불 탐지를 위한 높은 품질의 레이블 이미지 데이터셋의 필요성을 강조하며, 산불 전후 이미지를 사용하여 성능을 개선하는 데 기여하고 있습니다.



### LiDAR-3DGS: LiDAR Reinforced 3D Gaussian Splatting for Multimodal Radiance Field Rendering (https://arxiv.org/abs/2409.16296)
- **What's New**: 이번 논문에서는 3D Gaussian Splatting(3DGS) 기반의 Radiance Field Rendering에 LiDAR 입력을 활용한 혁신적인 방법인 LiDAR-3DGS를 소개합니다.

- **Technical Details**: LiDAR-3DGS는 LiDAR로 생성된 포인트 클라우드를 이용해 3DGS 입력을 보강하여 3D 모델의 정확성 및 세부사항을 크게 향상시킵니다. 이 방법은 볼트, 구멍 및 기타 중요한 특징들을 포착하는 데 도움을 주며, 이는 원격 모니터링 및 유지보수와 같은 엔지니어링 응용에 매우 중요합니다. 3DGS 알고리즘을 수정하지 않고도, LiDAR 포인트 클라우드의 소폭 추가로 모델의 인지 품질이 향상됨을 보여주었습니다.

- **Performance Highlights**: 모델이 30,000회 반복 실행 이후 PSNR(피크 신호 대 잡음 비율)이 7.064% 증가하고 SSIM(구조적 유사도 지표)이 0.565% 개선되었습니다. 사용된 LiDAR는 상용 등급의 기기였으며, 향후 더 고급 LiDAR 시스템을 통해 이러한 개선은 더욱 발전할 수 있습니다.



### GenCAD: Image-Conditioned Computer-Aided Design Generation with Transformer-Based Contrastive Representation and Diffusion Priors (https://arxiv.org/abs/2409.16294)
Comments:
          24 pages, 13 figures

- **What's New**: 이번 논문에서는 GenCAD라는 새로운 생성 모델을 소개합니다. 이 모델은 CAD 명령어 시퀀스로 변환하여 편집 가능한 3D 형태를 생성하며, 이미지 입력을 통해 CAD 프로그램을 생성합니다.

- **Technical Details**: GenCAD는 autoregressive transformer와 latent diffusion 모델을 통합하여 이미지에서 CAD 명령 시퀀스를 생성합니다. 이를 위해 contrastive learning 프레임워크를 활용하여 CAD 이미지와 CAD 명령 시퀀스의 공동 분포를 학습합니다.

- **Performance Highlights**: GenCAD는 기존의 최신 방법들보다 3D 형상 생성의 정밀도와 수정 가능성 면에서 월등한 성능을 보였습니다. 특히, 긴 시퀀스의 3D 형상 생성 정확도가 크게 향상되어 복잡한 설계 작업에 적합합니다.



### Explaining Human Comparisons using Alignment-Importance Heatmaps (https://arxiv.org/abs/2409.16292)
- **What's New**: 이 논문에서는 사람의 유사성 판단을 비교하는 과정을 설명하기 위해 Alignement Importance Score (AIS) 열지도를 제안하고 있습니다. AIS는 Deep Neural Network (DNN)의 표현 기하학과 인간의 그것 사이의 정렬에 대한 기여도를 측정합니다.

- **Technical Details**: 이 연구는 DNN의 마지막 합성곱 층에서 핀셋하여 이미지에 대한 중요한 정보를 설명합니다. 구체적으로, AIS를 활용하여 높은 평가 점수를 가진 특징 맵만 사용하여 인간의 유사성 판단을 예측하는 데 있어 정확도를 높입니다. 연구는 전통적인 saliency map과 비교하여 결과의 해석 가능성을 평가합니다.

- **Performance Highlights**: DNN의 임베딩으로부터 인간의 유사성 판단을 예측하는 데 Alignment Importance가 개선된 결과를 보였으며, 이미지 공간에서 어떤 정보가 중요한지를 설명하는 통찰력을 제공합니다.



### PACE: marrying generalization in PArameter-efficient fine-tuning with Consistency rEgularization (https://arxiv.org/abs/2409.17137)
Comments:
          Accepted by NeurIPS 2024 as a spotlight. This preliminary version will soon be extended with the experiments and analyses from the rebuttal

- **What's New**: 본 연구는 Parameter-Efficient Fine-Tuning (PEFT) 방법의 일반화 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. PACE라는 방법은 작은 gradient norm과 대규모 데이터셋의 관계를 이론적으로 연결하고, 이를 통해 fine-tuned 모델의 일반화를 향상시키는 것을 목표로 합니다.

- **Technical Details**: PACE는 두 가지 주요 전략을 결합합니다. 첫째, ADAPT에 의해 학습된 feature에 대해 multiplicative noise를 적용하여 perturbation을 일으킵니다. 둘째, 다양한 perturbation 아래에서 fine-tuned 모델의 출력이 일관되게 유지되도록 합니다. 이를 통해 gradient regularization이 강화되어 모델의 일반화 성능이 향상됩니다.

- **Performance Highlights**: PACE는 VTAB-1k, FGVC, few-shot learning, domain adaptation 등 4가지 비주얼 적응 작업에서 기존 PEFT 방법들을 능가하는 성과를 보여줍니다. 이 연구는 날로 증가하는 모델의 크기와 복잡성을 관리하는 데 중요한 통찰을 제공합니다.



### Classification of Gleason Grading in Prostate Cancer Histopathology Images Using Deep Learning Techniques: YOLO, Vision Transformers, and Vision Mamba (https://arxiv.org/abs/2409.17122)
- **What's New**: 이 연구에서는 전립선암 진단을 위한 Gleason 등급 분류 자동화를 위해 YOLO, Vision Transformers, Vision Mamba의 세 가지 딥러닝 (deep learning) 방법론의 효과를 비교합니다.

- **Technical Details**: Gleason2019 및 SICAPv2이라는 두 개의 공개 데이터셋을 사용하여 각 모델의 성능을 훈련하고 테스트했습니다. 각 모델은 false positive rate, false negative rate, precision, recall과 같은 메트릭을 기반으로 평가되었습니다.

- **Performance Highlights**: Vision Mamba는 모든 성능 지표에서 우수한 성과를 보였으며, 높은 precision과 recall을 유지하면서 false positives 및 negatives를 최소화했습니다. YOLO는 실시간 분석에 유리한 속도와 효율성을 보여주었으며, Vision Transformers는 이미지 내 긴 거리 종속성을 잘 포착했지만 다른 모델들에 비해 더 높은 계산 복잡성을 나타냈습니다.



### The Effect of Perceptual Metrics on Music Representation Learning for Genre Classification (https://arxiv.org/abs/2409.17069)
Comments:
          arXiv admin note: text overlap with arXiv:2312.03455

- **What's New**: 이 연구에서는 음악 이해 작업, 특히 장르 분류에 대한 성능을 향상시킬 수 있는 방법으로, 지각 메트릭(perceptual metrics)을 활용하는 새로운 접근법을 제안합니다. 특히, 자기 부호화기(autoencoders)로부터 추출된 특징을 사용하여 지각 손실(perceptual losses)로 훈련된 모델이 장르 분류를 개선할 수 있음을 입증했습니다.

- **Technical Details**: 지각 메트릭은 인간 관찰자의 지각 행동을 근사하는 데 설계된 객관적인 측정 지표입니다. 예를 들어, 구조적 유사도(SSIM)와 정규화된 라플라스 피라미드 거리(NLPD)와 같은 지표를 스펙트로그램(spectrograms)에 적용하여 오디오 품질에 대한 인간 평가와 더 나은 연관성을 보이는 것을 보여주었습니다. 이들 메트릭은 모델 훈련 시 손실 함수(loss function)으로 사용되어 모델의 성능을 개선할 수 있습니다.

- **Performance Highlights**: K-최근접 이웃(K-Nearest Neighbours) 분류기를 사용할 때, 전통적인 MSE(mean squared error)보다 지각 메트릭을 통한 성능 향상이 나타났습니다. 특히 로지스틱 회귀(Logistic Regression) 모델은 자기 부호화기에서 추출된 잠재 특징(latent features)을 활용할 때 높은 F1 점수를 기록했습니다. 그러나 NLPD는 군집화(clustering) 거리 측정에는 적합하지 않은 것으로 나타났으며, 이는 불필요한 정보를 제거함으로써 느리게 변화하는 부분들을 배제하기 때문입니다.



### Automated Surgical Skill Assessment in Endoscopic Pituitary Surgery using Real-time Instrument Tracking on a High-fidelity Bench-top Phantom (https://arxiv.org/abs/2409.17025)
Comments:
          7 pages, 6 figures

- **What's New**: 이 연구에서는 내시경 뇌하수체 수술의 비강 단계를 시뮬레이션한 새로운 공개 데이터 세트를 소개하였습니다. 이는 수술 기술 평가를 자동화하는 데 필요한 새로운 기반 모델 PRINTNet을 개발하고, 수술 기술 수준을 예측하는 데 사용되는 다양한 인사이트를 제공합니다.

- **Technical Details**: PRINTNet (Pituitary Real-time INstrument Tracking Network)은 DeepLabV3를 사용한 분류 및 세분화, StrongSORT를 이용한 추적, 그리고 NVIDIA Holoscan SDK를 통한 실시간 성능으로 구성됩니다. 이 모델은 22 Frames Per Second (FPS)에서 71.9%의 Multiple Object Tracking Precision을 달성했습니다.

- **Performance Highlights**: MULTILAYER Perceptron은 수술 기술 수준을 예측하는 데 87%의 정확도를 기록하였으며, '전체 절차 시간 대비 도구 가시 시간의 비율'이 높은 수술 기술과 상관관계가 있음을 나타냈습니다. 이 연구는 시뮬레이션된 내시경 뇌하수체 수술에서 자동화된 수술 기술 평가의 가능성을 보여줍니다.



### WasteGAN: Data Augmentation for Robotic Waste Sorting through Generative Adversarial Networks (https://arxiv.org/abs/2409.16999)
Comments:
          Accepted at 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 이번 논문에서는 로봇 폐기물 분류 문제를 해결하기 위한 획기적인 접근으로 WasteGAN이라는 새로운 GAN(Generative Adversarial Network) 아키텍처를 소개합니다. 이 방법은 제한된 라벨링된 데이터를 사용하여 성능을 향상시키기 위한 데이터 증대(data augmentation) 기법을 적용합니다.

- **Technical Details**: WasteGAN은 새로운 손실 함수(loss function)와 활성화 함수(activation function), 그리고 더 큰 생성기 블록(generator block)을 포함하고 있어, 네트워크가 제한된 예제 수로부터 학습하고 실제 세계 분포를 더 잘 반영하는 데이터를 합성할 수 있도록 돕습니다. 이 기술은 특히 폐기물 처리 분야의 비구조적 장면에서의 학습에 중점을 두고 있습니다.

- **Performance Highlights**: WasteGAN을 기반으로 한 데이터 증대 방법은 ZeroWaste 데이터셋에서 다양한 최첨단 세그멘테이션(segmentation) 모델을 훈련하는 데 효과적임이 입증되었습니다. 논문에서는 성능이 최대 5.8% 향상되었음을 보고하며, 실제 로봇 폐기물 분류 시스템에 통합하여 탁월한 성과를 보였습니다.



### PitRSDNet: Predicting Intra-operative Remaining Surgery Duration in Endoscopic Pituitary Surgery (https://arxiv.org/abs/2409.16998)
Comments:
          Accepted to the Augmented Environments for Computer-Assisted Interventions (AE-CAI) Workshop at the Medical Image Computing and Computer-Assisted Interventions (MICCAI) Conference 2024

- **What's New**: 이 논문은 내시경 뇌하수체 수술 중 수술의 남은 시간(Remaining Surgery Duration, RSD) 예측을 위한 새로운 모델, PitRSDNet을 제안합니다. 이 모델은 과거 데이터를 활용하여 작업 흐름에 중점을 둔 spatio-temporal neural network 모델입니다.

- **Technical Details**: PitRSDNet은 두 가지 형태로 작업 흐름 지식을 RSD 예측에 통합합니다: 1) RSD와 단계(step)를 동시에 예측하는 multi-task learning; 2) 시간 학습과 추론에서 이전 단계를 맥락으로 포함시킵니다. 이 모델은 88개의 비디오로 구성된 새로운 내시경 뇌하수체 수술 데이터셋에서 훈련 및 평가되어 이전의 통계적 및 기계 학습 방법보다 경쟁력 있는 성능 개선을 보여줍니다.

- **Performance Highlights**: PitRSDNet은 수술의 마지막 10-20분 동안 5분 이하의 오차를 예상하며, 전체 수술 시간에 대해서는 10분 이하의 오차를 가지는 것이 기대됩니다. 연구 결과는 PitRSDNet이 이전 단계의 지식을 활용하여 이상치(outlier) 사례에서 RSD의 정밀성을 개선하는 방법을 강조합니다.



### Multi-Robot Informative Path Planning for Efficient Target Mapping using Deep Reinforcement Learning (https://arxiv.org/abs/2409.16967)
Comments:
          arXiv admin note: text overlap with arXiv:2402.04894

- **What's New**: 자율 로봇이 업무를 수행하는 과정에서, 한정된 자원 예산 아래에서 정보 수집과 탐색을 극대화하기 위해 새로운 심층 강화 학습 기법을 제안했습니다.

- **Technical Details**: 우리의 접근 방식은 다중 로봇(멀티 로봇) 정보 경로 계획(Informative Path Planning, IPP) 문제를 해결하기 위해 증강 그래프를 활용합니다. 이는 통신 및 로봇 간 충돌 회피를 위한 계획을 가능하게 합니다. 각각의 로봇은 제한된 감지 범위를 가진 UAV로 구성되어 있으며, 통신 모듈을 통해 모든 인근 로봇과 정보를 전송합니다.

- **Performance Highlights**: 제안한 방안은 초점 탐색 목표 발견 수에서 기존의 최첨단 다중 로봇 목표 매핑 기술보다 33.75% 더 뛰어난 성능을 보였으며, 여러 UAV를 사용하는 도시 감시 시나리오에서 실제 성능을 검증하였습니다.



### Go-SLAM: Grounded Object Segmentation and Localization with Gaussian Splatting SLAM (https://arxiv.org/abs/2409.16944)
- **What's New**: 새로운 Go-SLAM 프레임워크를 소개하며, 3D Gaussian Splatting SLAM을 활용하여 동적 환경을 재구성하고 장면 표현 내에서 객체 수준 정보를 통합합니다. 이 시스템은 고급 객체 분할 기술을 사용하여 각 Gaussian splat에 고유한 식별자를 할당합니다.

- **Technical Details**: Go-SLAM은 3D Gaussian Splatting을 기반으로 하며, 객체 감지 및 세분화 모델을 포함하여 고급 컴퓨터 비전 기술을 활용하여 환경을 재구성합니다. 또한, 자연어 처리 기술을 활용하여 사용자 또는 고급 계획 알고리즘이 객체를 유연하게 쿼리할 수 있도록 합니다.

- **Performance Highlights**: 종합적인 평가 결과, 다양한 장면 설정에서 정밀도와 recall, IoU가 각각 17%, 27%, 35% 개선되는 것을 보여주며, Go-SLAM의 효율성을 입증합니다.



### Going Beyond U-Net: Assessing Vision Transformers for Semantic Segmentation in Microscopy Image Analysis (https://arxiv.org/abs/2409.16940)
Comments:
          to be published in ECCV 2024 BioImage Computing Workshop

- **What's New**: 이 논문은 전통적인 U-Net 모델과 최신 변환기(transformer) 기반 모델 간의 비교를 통해 생물 의학 이미지 세분화에서의 성능 향상을 제시합니다. 특히, UNETR, Segment Anything Model, Swin-UPerNet 모델을 활용하여 전자 현미경, 밝은 필드, 조직 병리학, 위상 대비와 같은 다양한 이미지 모달리티에서 이들 모델을 평가합니다.

- **Technical Details**: 변환기 기반 모델은 Attention Mechanism을 활용하여 이미지 구조를 분석하며, ViT(Vision Transformer) 및 Swin Transformer와 같은 최신 기술을 포함합니다. 이는 이미지 세분화에서 로컬 컨텍스트를 더욱 잘 포착하고, 수용 필드를 확장할 수 있게 해줍니다. 특히, Swin Transformer를 이용한 UPerNet 기반 디코더의 구조적 개선을 통해 세부 사항 잡기 및 세분화 정확도를 향상하는 방법을 제안합니다.

- **Performance Highlights**: 개선된 Swin-UPerNet 모델은 기존의 U-Net 모델과 비개선된 Swin-UPerNet 대비 세분화 성능이 향상된 결과를 보였습니다. 이는 변환기 기반 모델이 생물 의학 이미지 세분화의 효율성과 적용 가능성을 높일 수 있는 가능성을 보여줍니다.



### Moner: Motion Correction in Undersampled Radial MRI with Unsupervised Neural Representation (https://arxiv.org/abs/2409.16921)
Comments:
          18 pages, 13 pages

- **What's New**: 최근 연구에서는 움직임 보정(motion correction, MoCo) 문제를 해결하기 위해 여러 기법들이 제안되었습니다. 본 논문에서는 Moner라는 새로운 비지도 학습 MoCo 방법을 제안하여, 고품질의 MR 이미지를 확보하기 위한 대규모 데이터셋 없이도 정확한 움직임을 추정할 수 있습니다.

- **Technical Details**: Moner는 암묵적 신경 표현(Implicit Neural Representation, INR)을 활용하여 비선형 문제를 해결합니다. INR에 준정적(quasi-static) 모션 모델을 통합하고 푸리에 슬라이스 정리를 활용하여 radial MRI 복구를 역 투영(back-projection) 문제로 재구성합니다. 이 방법은 MRI k-space 데이터로 인해 발생하는 고역(dynamics) 문제를 완화하여 안정적인 모델 최적화를 도모합니다.

- **Performance Highlights**: Moner는 두 개의 공공 MRI 데이터셋(fastMRI와 MoDL)에서 평가했으며, 도메인 내(data in-domain)에서는 최신의 MoCo 기술과 유사한 성능을 보였고, 도메인 외(out-of-domain) 데이터에서 상당한 성능 향상을 보여주었습니다.



### Towards General Text-guided Image Synthesis for Customized Multimodal Brain MRI Generation (https://arxiv.org/abs/2409.16818)
Comments:
          23 pages, 9 figures

- **What's New**: 본 논문에서는 TUMSyn이라는 텍스트 기반의 범용 MR 이미지 생성 모델을 제안합니다. 이 모델은 정기적으로 수집된 스캔 데이터로부터 요구되는 이미징 메타데이터를 기반으로 뇌 MR 이미지를 유연하게 생성할 수 있습니다.

- **Technical Details**: TUMSyn 모델은 31,407개의 3D 이미지를 포함하는 대규모 뇌 MR 데이터베이스를 기반으로 훈련됩니다. 우리는 대칭적 학습(contrastive learning)을 사용하여 텍스트 인코더를 미리 훈련하고, 이를 통해 메타데이터에서 관련된 이미지 기능을 효과적으로 추출합니다. 두 단계의 훈련 전략을 통해 이미지와 텍스트 페어를 조정하여 MR 이미지를 생성합니다.

- **Performance Highlights**: TUMSyn은 다양한 데이터셋과 임상 평가에서 높은 성능을 보여주었습니다. PSNR(최대 신호 대 잡음 비율)은 2.86 dB까지 향상되었으며, SSIM(구조적 유사성 지수)은 0.044의 개선을 보였습니다. 또한, TUMSyn은 적응성이 뛰어난 제로샷 학습(zero-shot learning)에서 우수한 성능을 발휘하며, 뇌 종양 영역의 정확한 합성을 통해 진단 지원에 활용 가능성을 나타냅니다.



### Inline Photometrically Calibrated Hybrid Visual SLAM (https://arxiv.org/abs/2409.16810)
- **What's New**: 본 논문은 Hybrid direct-indirect visual SLAM (H-SLAM)에 온라인 순차적 photometric calibration을 통합한 새로운 접근을 제시합니다. 이를 통해 다양한 조명 조건에서의 픽셀 강도 값을 정규화하여 H-SLAM의 정확성을 개선하고, 기존 SLAM 시스템에 비해 월등한 성능을 보여줍니다.

- **Technical Details**: 본 연구에서는 Online Sequential Photometric Calibration (OSPC) 기술을 H-SLAM에 통합하여, 카메라의 응답 함수(Camera Response Function, CRF)와 비네팅(vignetting)을 순차적으로 추정하고 이를 통해 V-SLAM의 입력 프레임을 교정합니다. 이 과정은 더욱 안정적이고 일관된 피쳐 추적 및 깊이 추정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, photometrically calibrated H-SLAM은 TUM monoVO와 자작 데이터셋을 포함한 여러 테스트에서 다른 최첨단 SLAM 시스템보다 뛰어난 성능을 보였습니다. 특히, 실시간 SLAM 테스트에서는 기존 SLAM 시스템들을 현저히 능가하는 결과를 기록했습니다.



### Scalable Ensemble Diversification for OOD Generalization and Detection (https://arxiv.org/abs/2409.16797)
Comments:
          Under review

- **What's New**: 본 연구는 Scalable Ensemble Diversification (SED)라는 새로운 방법론을 제시하여 기존의 다양한 앙상블 학습 방법의 한계를 극복합니다. 특히, OOD 샘플 없이도 대규모 데이터에서 효과적으로 적용할 수 있는 방식으로 설계되었습니다.

- **Technical Details**: SED는 세 가지 주요 기술적 혁신을 통해 발전되었습니다: (1) 하드 샘플을 동적으로 식별하여 모델 간의 불일치를 유도합니다. (2) 각 반복에서 무작위로 선택된 두 모델에 대해서만 다양화 목표를 적용하여 계산 비용을 줄입니다. (3) 네트워크의 출력 근처의 일부 레이어에만 영향을 미치도록 하여 심층 네트워크에서의 다양화 목표를 조정합니다.

- **Performance Highlights**: ImageNet에서의 실험을 통해 SED의 다양한 이점을 확인했습니다. OOD 일반화와 OOD 탐지 모두에서 성능이 상당히 향상되었으며, Predictive Diversity Score (PDS) 방법론은 OOD 샘플 탐지에서 기존 방법들을 초월하는 성능을 보였습니다.



### Let There Be Light: Robust Lensless Imaging Under External Illumination With Deep Learning (https://arxiv.org/abs/2409.16766)
Comments:
          4 pages, dataset: this https URL

- **What's New**: 이 논문은 렌즈가 없는 카메라에서 외부 조명(ambient lighting)이 가지는 중요성을 다룹니다. 기존 연구에서는 대상 물체에서 방출된 빛에만 초점을 맞추었으나, 이 연구는 다양한 조명 조건에서의 데이터 세트를 제공하고 외부 조명을 고려한 복구 기술을 제안합니다.

- **Technical Details**: 제안된 방법은 물리 기반의 이미지 복구 기술로, 외부 조명의 추정치를 이미지 복구 과정에 통합합니다. 연구진은 25K 개의 다양한 조명 조건에서 측정된 렌즈 없는 데이터셋을 오픈소스로 배포하였으며, 이를 통해 모델 기반 최적화와 신경망(neural network) 기법을 결합하여 이미지 품질을 크게 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 기법은 기존 재구성 방법들과 비교할 때 질적 및 양적 개선을 보이며, 외부 조명에 대한 적응력을 높이는 것이 렌즈 없는 이미징의 실용성과 채택 가능성을 증대시킵니다.



### The Effect of Lossy Compression on 3D Medical Images Segmentation with Deep Learning (https://arxiv.org/abs/2409.16733)
Comments:
          12 pages, 5 figures, 2 tables; accepted on MICCAI Workshop on Advancing Data Solutions in Medical Imaging AI

- **What's New**: 이 연구는 3D 의료 이미지에서 손실 압축이 분할 품질에 부정적인 영향을 미치지 않음을 보여줍니다. Gu사에서는 깊은 신경망(DNN) 기반 모델이 압축된 데이터로 훈련될 때 감손 없이 비압축 데이터에서도 예측할 수 있음을 입증했습니다.

- **Technical Details**: 연구는 CT 및 MRI의 3D 의료 이미지를 대상으로 손실 압축이 DNN 기반 분할 모델에 미치는 영향을 분석합니다. JPEG 2000 압축 방식을 사용하여 훈련 데이터를 최대 20배 압축하고도 분할 품질 유지가 가능함을 보여줍니다. nnU-Net을 프레임워크로 사용하여 20개의 분할 작업을 수행했습니다.

- **Performance Highlights**: 연구 결과, 손실 압축 비율이 20배인 경우에도 DNN 모델은 여전히 높은 품질의 분할 능력을 유지하며, 압축된 이미지를 기반으로 한 훈련 후에도 비압축 이미지에 대해 예측 품질이 보장된다고 확인되었습니다.



### Non-stationary BERT: Exploring Augmented IMU Data For Robust Human Activity Recognition (https://arxiv.org/abs/2409.16730)
- **What's New**: 이 연구에서는 OPPOHAR라는 새로운 인간 활동 인식(HAR) 데이터셋을 소개합니다. 이는 모바일 장치에서 수집된 IMU 데이터로 구성되어 있으며 사용자의 특정 활동 인식을 위해 최적화된 경량 네트워크인 Non-stationary BERT를 제안합니다. 또한 가속도계와 자이로스코프 데이터 간의 깊은 관계를 탐구하기 위한 간단하면서도 효과적인 데이터 증강(data augmentation) 방법을 도입합니다.

- **Technical Details**: 논문에서는 IMU 데이터의 자기 감시(self-supervised) 사전 훈련(pretraining)과 감독(classification) 단계를 포함한 Non-stationary BERT 네트워크 설계를 제안합니다. 이 네트워크는 가속도계와 자이로스코프 데이터의 상호 관계를 고려하여 새로운 시퀀스를 생성하고, 이를 기반으로 한 데이터 증강 방법을 도입하여 HAR 성능을 향상시킵니다. 또한, 사용자 개인의 데이터를 사용한 분산 배포 최적화된 경량 네트워크를 구현합니다.

- **Performance Highlights**: 제안된 Non-stationary BERT 네트워크는 다양한 활동 인식 데이터셋에서 최신 성능(state-of-the-art performance)을 달성하였으며, 데이터 증강 방법은 광범위한 적응 가능성을 보여줍니다. 이 연구는 사용자 개인의 데이터로 훈련된 분류기를 통해 프라이버시를 보장하면서도 사용자 맞춤형 활동 인식을 가능하게 합니다.



### SDCL: Students Discrepancy-Informed Correction Learning for Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2409.16728)
Comments:
          Accepted at MICCAI 2024

- **What's New**: 이번 연구에서는 Semi-supervised medical image segmentation (SSMIS) 방법의 한계인 확인 및 인지 편향을 해결하기 위해 Students Discrepancy-Informed Correction Learning (SDCL) 프레임워크를 제안합니다. 이 프레임워크는 두 명의 학생 모델과 하나의 비훈련 교사 모델을 사용하며, 두 학생 간의 세분화 차이를 통해 자기 수정 학습을 유도합니다.

- **Technical Details**: SDCL 프레임워크는 Mean Teacher 접근법을 개선하여 설계되었습니다. 두 개의 학습 가능한 학생과 하나의 EMA (Exponential Moving Average) 교사를 통해 세분화의 차이가 존재하는 영역을 잠재적인 편향 영역으로 식별합니다. 또한, 두 개의 수정 손실 함수가 사용되어 올바른 세분화 부피 간의 거리를 최소화하고 잘못된 세분화 부피의 엔트로피를 최대화합니다.

- **Performance Highlights**: 세 가지 공공 의료 이미지 데이터셋(Pancreas, LA, ACDC)에서 실험한 결과, SDCL은 현재 State-of-the-Art (SOTA) SSMIS 방법을 각각 2.57%, 3.04%, 2.34% 초과하는 Dice 점수를 기록했습니다. 또한 ACDC 데이터셋에서는 완전 감독 방법과의 정확도가 매우 유사하며, Pancreas 및 LA 데이터셋에서는 이를 초과하는 성능을 보였습니다.



### 3DDX: Bone Surface Reconstruction from a Single Standard-Geometry Radiograph via Dual-Face Depth Estimation (https://arxiv.org/abs/2409.16702)
Comments:
          MICCAI 2024. 12 pages, 4 figures

- **What's New**: 이번 연구에서는 단일 X선 이미지를 이용한 3D 뼈 표면 복원 작업을 새롭게 제안하였습니다. 기존의 다른 접근법들과는 다르게, X선의 고유한 특성을 활용하여 여러 깊이 맵을 동시에 학습하는 방법을 채택하였습니다.

- **Technical Details**: 제안된 방법(3DDX)은 X선 이미지로부터 프론트 및 백 서페이스의 깊이 맵을 동시에 추정합니다. 이 과정에서 새로운 손실 함수가 도입되어 특정 기하학적 제약 하에 스케일-특화 훈련이 가능해집니다. 또한, 600명의 CT와 2651개의 X선 이미지를 활용하여 방법의 효과성을 검증하였습니다.

- **Performance Highlights**: 기존 방법과 비교했을 때, 표면 재구축 오차가 4.78mm에서 1.96mm로 감소하는 등 상당한 정확도 향상을 보였으며, 임상 적용 가능성을 시사합니다.



### TSBP: Improving Object Detection in Histology Images via Test-time Self-guided Bounding-box Propagation (https://arxiv.org/abs/2409.16678)
Comments:
          MICCAI 2024

- **What's New**: 본 논문에서는 Test-time Self-guided Bounding-box Propagation (TSBP) 방법을 제안하여, 물체 검출 성능을 크게 향상시키는 새로운 접근 방식을 소개합니다. 이 방법은 고신뢰도 바운딩 박스의 정보를 활용하여 저신뢰도 바운딩 박스를 조정합니다.

- **Technical Details**: TSBP는 Earth Mover's Distance (EMD)를 활용하여 시각적 유사성을 바탕으로 바운딩 박스 간의 정보를 전파합니다. 이 과정은 신뢰도가 낮은 바운딩 박스의 클래스 레이블을 보정하며, 별도의 라벨링된 샘플이 요구되지 않아 기존의 불확실성 보정 방법과 차별화됩니다.

- **Performance Highlights**: 실험 결과, TSBP는 기존의 불확실성 보정 방법에 비해 더욱 견고하고 정확한 물체 검출 결과를 제공합니다. 특히 상태-of-the-art 딥러닝 기반 검출 네트워크와 함께 사용했을 때 성능이 크게 향상되었습니다.



### Mitigating Covariate Shift in Imitation Learning for Autonomous Vehicles Using Latent Space Generative World Models (https://arxiv.org/abs/2409.16663)
Comments:
          7 pages, 6 figures, for ICRA 2025 conference, for associated video file, see this https URL

- **What's New**: 본 논문에서는 자율주행에서 발생하는 covariate shift 문제를 해결하기 위해 잠재 공간 생성 세계 모델(latent space generative world models)의 사용을 제안합니다. 저자들은 드라이빙 정책이 인간의 행동을 따른 학습을 활용하여 오류에서 회복할 수 있도록 설계하였으며, 실제 환경에서도 훈련 데이터의 분포를 초과한 perturbations에 대응할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 시스템은 잠재 공간 생성 세계 모델을 통해 드라이빙 정책을 공동 학습하는 구조를 가지고 있으며, 새로운 ego 상태를 샘플링하여 훈련 데이터에서 발견되지 않았던 상태를 탐색합니다. 이를 통해 driving policy가 인간의 시연에서 관찰된 상태에 대해 더 가깝게 행동을 선택할 수 있도록 합니다. 또한, 다중 뷰 크로스 어텐션(multi-view cross-attention)을 사용하는 새로운 Transformer 기반의 인식 인코더(perception encoder)를 도입하였습니다.

- **Performance Highlights**: 실험 결과, 저자들은 CARLA 시뮬레이터와 NVIDIA DRIVE Sim에서 닫힌 루프(closed-loop) 테스트를 통해 이전 최첨단 기술보다 유의미한 개선을 달성하였으며, 다양한 perturbations을 처리할 수 있는 능력을 입증하였습니다.



### Deep-Learning Recognition of Scanning Transmission Electron Microscopy: Quantifying and Mitigating the Influence of Gaussian Noises (https://arxiv.org/abs/2409.16637)
- **What's New**: STEM(Scanning Transmission Electron Microscopy) 이미지를 통해 나노 입자를 인식하기 위한 Deep Learning Mask R-CNN 모델을 제안하며, 생성된 큰 데이터셋의 컴퓨터 기반 자동화를 위한 접근법을 개발했습니다.

- **Technical Details**: Mask R-CNN 모델은 다양한 Gaussian 잡음, 입자 모양 및 크기를 고려하여 나노 입자를 이미지에서 인식합니다. Gaussian noise가 인식 정확도에 미치는 영향을 분석하고, Gaussian 및 Non-Local Means 필터를 적용하여 노이즈의 영향을 줄였습니다.

- **Performance Highlights**: STEM-HAADF 이미지에 대해 Mask R-CNN 모델은 전통적인 threshold 방법보다 더 높은 인식 정확도를 달성하였습니다. 실험 및 시뮬레이션 결과 모두 만족스러운 인식 정확도를 보였으며, 복잡한 구조의 대량 데이터를 분석하는 데 유용합니다.



### Stochastic Subsampling With Average Pooling (https://arxiv.org/abs/2409.16630)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문에서는 기존의 Dropout 방식이 갖는 일관성 결여 문제를 해결한 새로운 모듈인 stochastic average pooling을 제안합니다. 이 모듈은 pooling 과정에서 Dropout과 유사한 확률성을 통합하여 신경망의 성능을 개선할 수 있습니다.

- **Technical Details**: Stochastic average pooling은 stochastic subsampling과 average pooling을 통합한 방식입니다. 이는 기존 average pooling을 대체할 수 있으며, 코드 변경이 최소화됩니다. 이 방법은 수학적 기호와 함께 명확하게 정의되어 있습니다.

- **Performance Highlights**: 실험 결과, stochastic average pooling로 기존 평균 풀링을 대체하면 다양한 데이터셋, 작업 및 모델에서 성능이 일관되게 개선되는 것으로 나타났습니다.



### FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning (https://arxiv.org/abs/2409.16578)
- **What's New**: 최근 로봇 공학 분야에서는 대규모 다중 작업 Behavior Cloning을 통해 일반화된 로봇 정책을 구축하기 위한 여러 노력이 진행되고 있습니다. 본 논문에서는 FLaRe라는 대규모 Reinforcement Learning 미세 조정 프레임워크를 제안하여, 사전 훈련된 표현을 통합하고, 대규모 훈련 및 그래디언트 안정화 기술을 사용하여 성능 향상을 목표로 합니다.

- **Technical Details**: FLaRe는 다중 작업 로봇 정책에서 시작하여 대규모 RL을 통한 미세 조정을 수행합니다. 이 과정에서 안정적인 RL 미세 조정을 보장하기 위해 간단하면서도 효과적인 기술을 도입하며, 이를 통해 성능을 대폭 향상시킵니다. FLaRe는 15배의 훈련 시간 단축을 제공하고, 전통적인 손으로 설계된 보상 함수 없이도 작동할 수 있습니다.

- **Performance Highlights**: FLaRe는 가정용 모바일 조작 작업에서 평균 79.5%의 성공률을 달성하며, 이전 SoTA 방법에 비해 +23.6%의 절대 향상을 기록했습니다. 실제 로봇에서의 성능도 평균 80.7%를 달성하며, 이전 최고 기록에 비해 +30.7% 개선된 결과를 보였습니다. FLaRe는 미세 조정에 필요한 인력이 적고, 새로운 구조와 행동에 빠르게 적응할 수 있는 장점을 가지고 있습니다.



### Diffusion Models to Enhance the Resolution of Microscopy Images: A Tutoria (https://arxiv.org/abs/2409.16488)
Comments:
          45 pages, 8 figures

- **What's New**: 본 튜토리얼에서는 Denoising Diffusion Probabilistic Models (DDPMs)을 사용하여 저해상도 현미경 이미지를 그에 상응하는 고해상도 이미지로 변환하는 방법에 대한 포괄적인 가이드를 제공합니다.

- **Technical Details**: 이 논문에서는 Diffusion Models을 기반으로한 이미지-이미지 변환 기술을 구현하는 방법을 설명하며, PyTorch를 사용한 구체적인 코드 구현과 함께 필요한 수학적 이론과 배경 지식을 포함합니다. 또한 microtubule 구조 이미지와 같은 실제 데이터를 활용하여 모델 성능을 향상시키는 기법에 대해서도 다룹니다.

- **Performance Highlights**: Diffusion 모델은 텍스트-이미지 변환 및 이미지 변환 작업에서 성공적으로 사용되고 있으며, 생물학적 구조를 복원하는 데 있어 뛰어난 해상도를 제공합니다. 특히, 저해상도 이미지를 체계적으로 처리하며, 딥러닝을 통한 단일 이미지 초해상도(SISR) 분야에서 긍정적인 결과를 나타내고 있습니다.



### Initialization of Monocular Visual Navigation for Autonomous Agents Using Modified Structure from Small Motion (https://arxiv.org/abs/2409.16465)
Comments:
          6 pages, 1 page for references, 6 figures, 1 table, IEEEtran format This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 우리는 우주에서 자율 로봇을 위한 독립적인 단안 시각 동시 위치 추정 및 지도 작성(vSLAM) 초기화 파이프라인을 제안합니다. 이 방법은 고전적인 Structure from Small Motion (SfSM)을 강화하여 약한 원근 투영(scene) 환경에서 단안(vSLAM) 에이전트를 강력하게 초기화합니다.

- **Technical Details**: 이 연구는 불확실한 기하학을 해결하기 위해 약한 원근 투영 및 중심 지향 동작(center-pointing motion)에 대한 SfSM 접근 방식을 개선합니다. 새로운 세 단계 초기화 파이프라인은 회전 추정, 평면 기하학을 통한 번들 조정(bundle adjustment) 및 전반적인 추정 과정을 포함하며, 고화질 위성 검사 이미지를 이용해 검증하였습니다. 또한, 반전 깊이에 대한 재매개변수를 도입하여 수치적 안정성을 보장하고, 3D 랜드마크의 재매개변수를 통해 이미지 기능 좌표 벡터의 양자화를 고려했습니다.

- **Performance Highlights**: 이 방법은 약한 원근 투영이 있는 현실적인 위성 검사 이미지를 사용하여 다른 단안 초기화 절차와 비교했을 때 더 향상된 성능과 효과성을 보여주었습니다. 시뮬레이션 데이터 세트를 기반으로 초기화 과정의 정확성과 신뢰성을 높여, 특히 우주에서의 비협조적 거주 우주 물체(RSO)의 검사 및 근접 작전 중에 안전한 비행 경로를 확보하는 데 기여할 수 있습니다.



### A novel open-source ultrasound dataset with deep learning benchmarks for spinal cord injury localization and anatomical segmentation (https://arxiv.org/abs/2409.16441)
- **What's New**: 이번 연구에서는 10,223개의 Brightness-mode (B-mode) 초음파 이미지를 포함하는 대규모 데이터셋을 공개했습니다. 이 데이터셋은 대칭 단면을 가진 25마리의 육계 소에서 얻은 것으로, 부상 전후의 척수 이미지를 포함합니다. 또한, 여러 최첨단 객체 탐지 알고리즘의 성능을 비교 분석하여 부상 부위의 위치를 파악하고 해부학적 구조에 레이블을 붙이는 방법을 제시하였습니다.

- **Technical Details**: 이 연구에서 사용된 YOLOv8 객체 탐지 모델은 부상 위치 탐지에서 평균 정확도(mAP50-95) 0.606을 달성하여 최상의 성능을 기록했습니다. DeepLabv3 세분화 모델은 보이지 않는 육계 생리학에 대해 평균 Dice 점수 0.587을 기록했고, SAMed는 인간 해부학에 대한 제로샷 일반화에서 0.445의 평균 Dice 점수를 달성하였습니다. 데이터셋은 건강한 및 손상된 척수의 해부학적 구조를 포함합니다.

- **Performance Highlights**: 본 연구는 SCI(척수 손상)에 대한 자동화된 지속 진단의 최초 선구적 노력을 담고 있으며, 연구자와 의료전문가들이 사용할 수 있는 가장 큰 주석 데이터셋을 제공함으로써 임상 결과를 향상시키기 위한 맞춤형 치료의 새로운 방법을 모색하고 있습니다.



### Lessons Learned from a Unifying Empirical Study of Parameter-Efficient Transfer Learning (PETL) in Visual Recognition (https://arxiv.org/abs/2409.16434)
Comments:
          Code is available at this https URL

- **What's New**: 최근 Parameter-efficient transfer learning (PETL) 기술에 대한 관심이 커지고 있으며, 이를 통해 기존의 대규모 pre-trained 모델을 더욱 효율적으로 조정하여 다양한 downstream 작업에서의 성능을 향상시키고자 하는 연구가 진행되고 있습니다. 본 논문은 Vision Transformers의 맥락에서 PETL 방법들을 통합적으로 비교하고, 그들의 성능을 체계적으로 분석하였습니다.

- **Technical Details**: 이 연구에서는 Low-Rank Adaptation (LoRA), Visual Prompt Tuning (VPT), Adapter 등 다양한 PETL 기법을 사용하여 진행하였으며, 하이퍼 파라미터(learning rate, weight decay 등)를 체계적으로 조정하여 low-shot benchmark인 VTAB-1K의 정확도를 비교하였습니다. 또한, CIFAR-100 및 RESISC와 같은 풀사이즈 데이터셋에서도 PETL 방법을 평가하였습니다.

- **Performance Highlights**: PETL 접근 방식들이 잘 조정되었을 경우 VTAB-1K에서 유사한 정확도를 기록하였으며, PETL 방법들은 낮은 샷의 데이터에서도 뛰어난 성능을 보여주었습니다. 무수한 학습 데이터를 가진 시나리오에서도 PETL이 full fine-tuning과 동등하거나 그 이상의 결과를 도출할 수 있다는 점이 주목할 만합니다. 또한 PETL은 distribution shift에 대한 강건성을 가지며, 기존 모델의 일반성을 유지하는 결과를 보였습니다.



### Vision-based Xylem Wetness Classification in Stem Water Potential Determination (https://arxiv.org/abs/2409.16412)
- **What's New**: 본 연구는 Stem Water Potential (SWP) 측정을 자동화하기 위한 새로운 접근 방식을 제안합니다. Scholander Pressure Chamber를 사용하여 줄기 탐지(stem detection)와 수관(xylem) 습도 분류를 자동화하고, YOLOv8n 및 ResNet50을 활용하여 이를 개선하였습니다.

- **Technical Details**: 본 연구에서는 SWP 측정을 위한 자동화된 시스템을 구축하였으며, YOLOv8n과 ResNet50 기반의 정확한 검출 및 분류 방법을 적용했습니다. 추가적으로, 데이터 확대(data augmentation) 및 모델 파라미터 튜닝을 통해 20개의 SWP 측정에 대해 평가를 진행했습니다.

- **Performance Highlights**: 최고 성능의 모델은 줄기 탐지 및 수관 습도 분류에서 80.98%의 Top-1 정확도를 기록하였고, 이는 SWP 측정의 자동화 작업에서 가장 뛰어난 성과로 평가됩니다.



### Modern Hopfield Networks meet Encoded Neural Representations -- Addressing Practical Considerations (https://arxiv.org/abs/2409.16408)
Comments:
          17 pages, 8 figures, workshop submission to Neurips

- **What's New**: 본 논문은 Modern Hopfield Networks (MHN)에 대한 메타 안정 상태 문제를 해결하는 새로운 접근 방식인 Hopfield Encoding Networks (HEN)를 소개합니다. HEN은 입력 패턴의 분리 가능성을 높이고, 메타 안정 상태를 줄이기 위해 인코딩된 신경 표현을 통합합니다.

- **Technical Details**: HEN은 미리 훈련된 신경 인코더-디코더 모델을 사용하여 입력을 잠재 표현 공간으로 인코딩한 후 저장하고, 재호출 시 다시 디코딩하는 방법을 사용합니다. 이 접근 방식은 MHNs의 메타 안정 상태 문제를 해결하고, 자연어 쿼리를 통한 다양한 입력 모달리티에서의 검색을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 HEN이 메타 안정 상태를 크게 줄이고, 저장 용량을 증가시키면서 다양한 입력을 완벽히 기억할 수 있음을 나타냅니다. 이는 실제 작업을 위한 연상 기억 네트워크의 실용적인 활용을 향상시킵니다.



### Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning (https://arxiv.org/abs/2409.16391)
Comments:
          Published in Conference on Lifelong Learning Agents (COLLAS) 2024

- **What's New**: 논문에서는 상대적으로 탐색이 부족한 학습 패러다임인 Online Unsupervised Continual Learning (O-UCL)에 집중하고 있습니다. O-UCL은 비정상적인 레이블 없는 데이터 스트림을 처리하며 점진적으로 클래스의 수를 식별하는 능력을 길러주는 방식입니다. 본 연구는 실시간으로 새로운 클래스를 식별하고, 기존에 학습한 클래스를 잊지 않으면서 데이터를 스트리밍 방식으로 처리하는 동적 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법인 Patch-based Contrastive learning and Memory Consolidation (PCMC)은 데이터의 패치 수준의 특징을 식별하고 클러스터링하여 데이터를 이해합니다. PCMC는 인코더를 통해 패치 임베딩을 추출하고, 새로운 데이터를 기존 분포에 통합하면서 재학습 시 기억을 통합하는 기능을 갖추고 있습니다. 이 접근 방식은 주기적인 '깨어남'과 '수면' 주기를 통해 개념적으로 변화를 관리합니다.

- **Performance Highlights**: PCMC의 성능은 ImageNet과 Places365 데이터셋에서 생성된 스트림을 기반으로 평가되었습니다. 여러 기존 방법들과 간단한 기준점과의 성능 비교를 통해 PCMC의 유효성을 입증하였습니다.



### Future-Proofing Medical Imaging with Privacy-Preserving Federated Learning and Uncertainty Quantification: A Review (https://arxiv.org/abs/2409.16340)
Comments:
          21 pages, 5 figures, 4 tables, Review paper, preprint to Radiology AI. arXiv admin note: text overlap with arXiv:2406.12815

- **What's New**: 이번 논문은 Federated Learning (FL)을 통해 의료 이미징에서의 AI 모델 훈련을 개선할 수 있는 가능성을 탐구합니다. 특히, 데이터 공유 없이 민감한 정보를 보호하면서 협업할 수 있는 방법론을 다루며, 기존의 FL 접근법의 기본적인 문제점과 한계를 짚어봅니다.

- **Technical Details**: FL은 분산된 환경에서 여러 기관이 AI 모델을 공동으로 훈련할 수 있도록 하는 기술입니다. 이 과정에서 모델의 update 정보(예: gradients)만 공유되며 데이터는 공유되지 않습니다. 그러나 민감한 정보가 여전히 추론될 수 있는 가능성이 남아 있으며, 이는 privacy-preserving Federated Learning (PPFL)와 Uncertainty Quantification (UQ) 연구와 깊은 연관이 있습니다. 논문에서는 다양한 FL 알고리즘(예: FedAvg, FedProx 등)의 발전을 소개하고, 비독립적이고 동질적이지 않은 데이터 세트(data heterogeneity)의 문제를 해결하기 위한 접근 방식도 설명합니다.

- **Performance Highlights**: 이 논문은 FL이 의료 이미징에서 모델의 신뢰성을 높이는 방법으로 여겨지며, 정확하고 일반화 가능한 AI 모델 개발에 기여할 수 있는 잠재력을 가지고 있다고 강조합니다. 특히, 각기 다른 클라이언트 환경에서의 데이터 이질성을 고려하는 Personalized Federated Learning (PFL) 기법의 중요성을 제시하며, FL의 적용 사례를 통해 효과를 분석합니다.



### Predicting Distance matrix with large language models (https://arxiv.org/abs/2409.16333)
- **What's New**: RNA 구조 예측에 대한 새로운 접근 방법을 제안합니다. 대규모 사전 훈련된 RNA 언어 모델과 잘 훈련된 변환기(transformer)를 사용하여 RNA 염기 사이의 거리를 정확하게 추론할 수 있습니다.

- **Technical Details**: 거리 매트릭스(distance matrix)를 직접 예측하는 방법을 제시하며, 이 과정에서 기존의 3D 구조 모델링에서 발생하는 데이터 부족 문제를 해결하기 위한 혁신적인 접근 방식을 채택합니다. 우리는 변환기의 주의(attention) 메커니즘이 RNA 염기 쌍의 거리를 예측하는 데 적합하다고 주장합니다.

- **Performance Highlights**: 제안된 Distance Transformer(DiT)는 사전 훈련된 RNA 언어 모델을 통해 RNA 거리 매트릭스를 예측하고, 이를 통해 RNA 구조 및 기능에 대한 이해를 높일 수 있는 가능성을 보여줍니다.



### MRI Radiomics for IDH Genotype Prediction in Glioblastoma Diagnosis (https://arxiv.org/abs/2409.16329)
Comments:
          8 pages, 1 figure

- **What's New**: 이번 논문은 MRI 이미지를 활용한 Radiomics의 최신 동향을 다루고 있으며, 특히 Isocitrate dehydrogenase (IDH) 돌연변이 상태를 식별하는데 중점을 둡니다. IDH 돌연변이는 고등급 교모세포종과 등급 IV 별아교종의 중요 생체표지자로, 비침습적인 진단 방법의 필요성을 강조합니다.

- **Technical Details**: 논문에서 다루는 MRI Radiomics 워크플로우는 MRI 이미지에서 특징 추출을 위한 주요 단계를 설명합니다. 이미지 세분화는 수동, 반자동 또는 자동 방법으로 수행될 수 있으며, 자동 세분화는 딥러닝 모델을 사용하여 더 빠르고 정확하게 수행됩니다. 이미지 전처리 과정은 필수적으로 포함되며, 스컬 스트리핑과 다양한 필터링 기법이 사용됩니다.

- **Performance Highlights**: 이 연구는 IDH 돌연변이 상태를 정확히 예측하기 위한 MRI 기반 비침습적 방법의 효과를 입증하고 있으며, 이는 각환자에 맞춘 치료 계획 수립에 기여할 것입니다. 또한, 딥러닝 기반의 자동 세분화 기법은 임상 적용 가능성을 높이고 있습니다.



### Developing a Thailand solar irradiance map using Himawari-8 satellite imageries and deep learning models (https://arxiv.org/abs/2409.16320)
Comments:
          23 pages, 14 figures

- **What's New**: 이 논문은 태국의 태양 복사 지도(Global Horizontal Irradiance, GHI)를 30분마다 온라인으로 보여주는 플랫폼을 소개합니다. 이 플랫폼은 Himawari-8 위성 이미지를 기반으로 한 구름 지수(cloud index)와, Linke turbidity로 조정된 Ineichen 맑은 하늘 모델을 사용하여 GHI를 추정합니다.

- **Technical Details**: GHI 추정 모델에서 입력으로는 맑은 하늘 복사량, 구름 지수, MERRA-2 데이터베이스의 재분석 GHI 및 온도 데이터를 포함합니다. 사용된 머신 러닝 모델로는 LightGBM, LSTM, Informer, Transformer가 있으며, 2022-2023년 기간 동안 53개 지상 스테이션에서 15분 단위의 GHI 데이터를 평가하여 성능을 비교했습니다.

- **Performance Highlights**: 모든 모델은 경쟁력 있는 성능을 보였으며, SolCast 서비스보다 우수한 결과를 나타냈습니다. LightGBM 모델의 MAE(Mean Absolute Error)는 78.58 W/sqm, RMSE(Root Mean Square Error)는 118.97 W/sqm로 최상위 성능을 기록했습니다. Informer 모델은 추가적으로 재분석 MERRA-2 데이터 없이 MAE 78.67 W/sqm로 우수한 성능을 보였습니다.



### Damage detection in an uncertain nonlinear beam based on stochastic Volterra series: an experimental application (https://arxiv.org/abs/2409.16305)
- **What's New**: 이 논문은 구조물의 비선형 행동과 데이터의 자연 변동을 고려하여, 손상 탐지 문제 해결을 위한 확률적(Probabilistic) Volterra 시리즈의 실험적 적용을 다룹니다. 비선형 시스템에서 손상을 탐지하기 위해 기존의 결정론적(Deterministic) 방법과 확률적 방법의 비교를 수행합니다.

- **Technical Details**: 이 연구에서는 비선형 운동을 하는 고정 단작업 빔(Cantilever beam)을 실험적으로 사용하였으며, 실험적 데이터의 변동성을 보완하기 위해 확률적 Volterra 커널(Volterra kernel)의 기여도를 이용한 접근법이 적용되었습니다. 처치(causation)된 손상은 볼트 연결(bolted connection)에서의 질량 변화(mass changes)에 따른 것이며, 비선형 기여도와 선형 기여도를 비교하여 분석합니다.

- **Performance Highlights**: 확률적 모델이 데이터 변동성을 고려할 때 손상의 존재를 통계적 신뢰도로 탐지할 수 있는 능력을 보여주며, 비선형 메트릭이 선형 메트릭보다 손상 감지에 더 높은 민감도를 가지는 장점이 입증되었습니다. 이는 비선형 행동을 가진 시스템에서 비선형 메트릭을 사용하는 것이 중요하다는 것을 강조합니다.



### Computer Aided Detection and Classification of mammograms using Convolutional Neural Network (https://arxiv.org/abs/2409.16290)
- **What's New**: 이번 연구에서는 유방(X-ray) 영상에서 유방 종양을 정상과 비정상으로 분류하기 위한 새로운 기법으로 Convolutional Neural Network (CNN)을 활용하였습니다.

- **Technical Details**: 연구에 사용된 DDSM (Digital Database for Screening Mammography) 데이터셋에는 정상 유방 이미지를 약 460장, 비정상 유방 이미지를 920장 포함하고 있습니다. CNN을 사용하여 초기 종양의 자동 탐지 방법을 소개합니다.

- **Performance Highlights**: 기계 학습과 심층 학습(Deep Learning) 기술 적용을 통해 초기 유방암 증상인 덩어리(masses)와 미세 석회화(micro-calcifications)의 탐지가 더욱 정확해질 수 있습니다.



### Facing Asymmetry -- Uncovering the Causal Link between Facial Symmetry and Expression Classifiers using Synthetic Interventions (https://arxiv.org/abs/2409.15927)
Comments:
          45 pages; 26 figures; accepted at ACCV 2024

- **What's New**: 이 연구는 얼굴 대칭(Facial Symmetry)이 표현 분류기 모델의 출력 행동에 미치는 영향을 분석하는 구조적 인과 모델을 개발하였으며, 이를 통해 unilateral facial palsy 환자의 얼굴 표현을 개선하는 방법에 대한 통찰을 제공합니다.

- **Technical Details**: 연구진은 얼굴 대칭의 영향을 분석하기 위해 구조적 인과 모델(Structural Causal Model)에서 파생된 합성 개입 프레임워크(Synthetic Interventional Framework)를 사용했습니다. 이 프레임워크는 얼굴 대칭을 변경하면서 다른 요소를 고정하여 네트워크의 출력 행동을 분석할 수 있게 합니다.

- **Performance Highlights**: 17개의 표현 분류기를 분석한 결과, 얼굴 대칭이 감소할수록 모델의 출력 활성화가 의미있게 낮아지는 것을 발견했습니다. 이 연구는 얼굴 대칭이 표현 분류기 행동에 중요한 인과적 요인임을 강조합니다.



### FedRepOpt: Gradient Re-parameterized Optimizers in Federated Learning (https://arxiv.org/abs/2409.15898)
- **What's New**: 이번 연구에서는 Federated Learning (FL) 환경에서의 효율성을 높이기 위해 FedRepOpt라는 새로운 기법을 제안합니다. FedRepOpt는 복잡한 모델과 동일한 성능을 내는 단순한 로컬 모델을 훈련할 수 있도록 합니다.

- **Technical Details**: FedRepOpt는 gradient re-parameterization 기법을 사용하여 FL에서 RepOpt-VGG 및 RepOpt-GhostNet 모델을 적용합니다. 모델의 하이퍼 파라미터에 따라 최적화기의 그래디언트를 수정함으로써 훈련 도중 리소스를 효율적으로 사용하고, 복잡한 구조의 모델과 유사한 성능을 유지할 수 있습니다.

- **Performance Highlights**: FedRepOpt 사용 시 RepGhost 스타일과 RepVGG 스타일 네트워크에 비해 각각 16.7% 및 11.4%의 성능 향상을 보였으며, 더 빠른 수렴 시간을 기록하여 FL의 효율성을 크게 향상시켰습니다.



New uploads on arXiv(cs.AI)

### Turn Every Application into an Agent: Towards Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents (https://arxiv.org/abs/2409.17140)
- **What's New**: 이 논문은 AXIS라는 새로운 LLM 기반 에이전트 프레임워크를 제안하여 UI(UI)의 행위보다 API(API)의 행위를 우선시함으로써 사용자 인터페이스와의 상호작용에서 발생하는 지연(latency)과 신뢰성(reliability) 문제를 해결하고자 합니다.

- **Technical Details**: AXIS는 기존의 소프트웨어 애플리케이션을 자동 탐색하고 지원 문서 및 액션 트레일에서 통찰력을 학습하며 새로운 API를 생성할 수 있는 기능을 갖춘 자기 탐색(self-exploration) LLM 기반 프레임워크입니다. 이 프레임워크는 API 호출을 통해 작업을 수행하며, 예를 들어 Word 문서에 2×2 테이블을 삽입할 때 API 호출로 단 한 줄의 코드(doc.Tables.Add(NumRows=2,NumColumns=2)) 만으로 작업을 완료할 수 있습니다.

- **Performance Highlights**: AXIS를 사용하여 Office Word에서 수행한 실험 결과, 작업 완료 시간(task completion time)을 65%-70% 단축시키고 인지적 작업 부담(cognitive workload)을 38%-53% 감소시켰으며, 정확성은 97%-98%로 유지되었습니다. 이 연구는 LLM 시대에 애플리케이션 제공자를 위한 새로운 UI 디자인 원칙과 인간-에이전트-컴퓨터 상호작용(HACI) 프레임워크의 기여를 포함합니다.



### On-orbit Servicing for Spacecraft Collision Avoidance With Autonomous Decision Making (https://arxiv.org/abs/2409.17125)
Comments:
          The first joint European Space Agency SPAICE Conference / IAA Conference on AI in and for Space

- **What's New**: 본 연구는 인공지능(AI)을 기반으로 한 자율 궤도 서비스(On-Orbit Servicing, OOS) 임무 개발을 다루며, 이를 통해 우주선 충돌 회피 기동(Collision Avoidance Maneuvers, CAM)을 지원합니다. 특히 강화 학습(Reinforcement Learning, RL)으로 훈련된 자율 ‘서빙기(servicer)’를 제안하고, 충돌 위험을 자동으로 탐지하며 endangered satellite과 조우하고 도킹하여 최적의 CAM을 수행합니다.

- **Technical Details**: 본 연구는 목표 위성을 보호하기 위해 자율적으로 CAM을 결정하고 수행하는 서빙기 우주선을 고려합니다. 연구는 궤도 역학 시뮬레이터, 에이전트, 환경, 그리고 에이전트 훈련 알고리즘을 포함하는 프레임워크를 구성합니다. 또한, Markov Decision Process (MDP) 모델을 사용하여 자율 결정 문제를 형성하고, 각 위성과 쓰레기, 서빙기의 상태, 행동, 보상 함수를 정의합니다.

- **Performance Highlights**: 초기 결과는 Collision Avoidance 서비스에 대한 자율 로봇 OOS의 실행 가능성을 보여줍니다. 특히, 한 대의 서빙기 우주선과 하나의 endangered satellite 시나리오에서 집중적으로 연구하여 그 실행의 복잡성을 논의합니다.



### VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models (https://arxiv.org/abs/2409.17066)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 극히 저비트 양자화를 위한 새로운 접근법인 Vector Post-Training Quantization (VPTQ)을 제시합니다. VPTQ는 2비트 양자화를 지원하여 메모리 요구 사항을 줄이고 저장 비용을 최적화합니다.

- **Technical Details**: VPTQ는 두 번째 순서 최적화(Second-Order Optimization)를 사용하여 LLM의 벡터 양자화(Vector Quantization) 문제를 공식화합니다. 이 알고리즘은 채널 독립적인 두 번째 순서 최적화를 통해 세밀한 벡터 양자화를 제공하며, 최적화 문제를 분해하여 효과적인 코드북 초기화 알고리즘을 제안합니다. 또한, 잔여(residual)와 이상치(outlier) 양자화를 지원하여 모델의 정확성을 향상시키고 모델 압축을 추가적으로 진행합니다.

- **Performance Highlights**: 실험 결과에 따르면 VPTQ는 LLaMA-2에서 모델 양자화의 혼란도(perplexity)를 $0.01$-$0.34$ 감소시키고, Mistral-7B에서 $0.38$-$0.68$, LLaMA-3에서 $4.41$-$7.34$의 향상을 보였습니다. QA 작업에서 평균적으로 LLaMA-2는 $0.79$-$1.5	ext{ 	extbf{	extit{	extbf{	extit{%}}}}}$, Mistral-7B는 $1	ext{ 	extbf{	extit{%}}}$, LLaMA-3는 $11$-$22	ext{ 	extbf{	extit{	extbf{	extit{%}}}}}$의 정확도 개선을 나타냈습니다. 이 방법은 이전 방법 대비 $1.6$-$1.8	imes$의 추론 처리량을 증가시켰습니다.



### DRIM: Learning Disentangled Representations from Incomplete Multimodal Healthcare Data (https://arxiv.org/abs/2409.17055)
- **What's New**: 이 논문은 의료 데이터의 다중 모드(multi-modal) 통합을 위한 새로운 방법인 DRIM을 소개합니다. DRIM은 결측 데이터가 있는 상황에서도 공유된 정보와 고유한 정보를 효과적으로 캡처합니다. 이 방법은 환자 관련 정보와 각 모드에 특정한 정보를 분리하여 인코딩합니다.

- **Technical Details**: DRIM은 각 모드에 대한 두 개의 인코더(encoder)를 사용하여, 하나는 공유 정보(shared information)를, 다른 하나는 고유 정보(unique information)를 캡처합니다. 이를 통해 두 가지 차원에서 정보 통합을 수행합니다: 첫 번째로는 공유 정보 집합을 모으고, 두 번째로는 고유 정보를 조합하여 포괄적인 표현을 생성합니다. 또한 DRIM은 attention 기반의 융합 방법을 사용해 결측 모드를 자연스럽게 관리합니다.

- **Performance Highlights**: DRIM은 교모종(glioma) 환자의 생존 예측 작업에서 기존의 최신 알고리즘을 초월하는 성능을 발휘했습니다. 특히, DRIM은 결측 모드가 있어도 성능이 안정적이며, 의료 데이터의 복잡한 특성을 효과적으로 다룰 수 있습니다.



### Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia (https://arxiv.org/abs/2409.17054)
- **What's New**: 이 연구는 동남아시아 지역에서의 의사-환자 상호 작용의 효율성을 개선하기 위해 지역화된 대형 언어 모델(LLM)을 활용한 자동적 환자 기록 전사 및 요약 시스템을 제안합니다.

- **Technical Details**: 이 시스템은 OpenAI의 Whisper 모델을 사용하여 실시간으로 환자-의사 대화를 전사하고, GPT-3를 통해 요약된 정보를 ePuskesmas 전자 건강 기록 형식으로 변환합니다. 주요 과정은 네 가지로 나누어집니다: 대화 녹음, 실시간 전사, 의료 데이터 요약, 자동 ePuskesmas 파서.

- **Performance Highlights**: 이 솔루션을 통해 의사들은 더욱 신속하게 환자 정보를 정리할 수 있으며, 기록의 질도 향상되어 향후 환자 방문을 위한 정보가 더욱 자세하고 통찰력 있게 변모합니다. 이는 인도네시아의 과중한 시설과 의료 제공자의 행정 부담을 해소하는 중요한 진전을 나타냅니다.



### AI-Driven Risk-Aware Scheduling for Active Debris Removal Missions (https://arxiv.org/abs/2409.17012)
- **What's New**: 본 논문에서는 Deep Reinforcement Learning (DRL)을 기반으로 한 자율 의사 결정 계획 모델을 통해 Active Debris Removal (ADR) 미션에서 우주 쓰레기 제거 작업을 최적화하는 방안을 제시합니다.

- **Technical Details**: ADR 미션 계획 문제는 Cost-Constrained Traveling Salesman Problem (CCTSP)로 구성된다는 점이 강조되며, OTV(Orbital Transfer Vehicle)가 쓰레기 간 전이를 수행하며 비행 시간을 최소화하고 연료를 효율적으로 사용할 수 있도록 설계됩니다. 상태 공간은 남은 쓰레기 수, OTV의 총 연료량, 남은 임무 시간, 현재 제거할 쓰레기, 각 쓰레기의 제거 여부 및 충돌 위험 수준을 포함합니다.

- **Performance Highlights**: 제안된 모델을 통해 OTV는 쓰레기 제거 작업의 최적 계획을 수립할 수 있으며, 자동적으로 충돌 위험이 높은 쓰레기를 포함시키는 방식으로 계획을 업데이트하는 능력을 얻는 것으로 나타났습니다.



### Models Can and Should Embrace the Communicative Nature of Human-Generated Math (https://arxiv.org/abs/2409.17005)
- **What's New**: 이 논문은 수학이 사람에 의해 사람을 위해 구성된다는 관점을 제안하며, 수학 데이터가 단순한 기호적 표현을 넘어선 풍부한 의사소통 의도를 반영한다고 주장합니다. 이 연구는 언어 모델(Language Models)이 수학적 심볼을 처리할 때 인간과 유사한 방식을 채택하고 있음을 실험을 통해 증명합니다.

- **Technical Details**: 연구에서는 두 가지 사례 연구를 통해 언어 모델의 수학적 문제 해결 방식의 비대칭성(asymmetry)과 정렬(ordering)에 대한 선호도를 조사하였습니다. 첫 번째 실험에서는 동일한 기본 방정식에 대해 다양한 형태의 언어 문제를 생성하는 능력을 평가했습니다. 두 번째로, 언어 모델이 수학적 증명의 배열 방식이 자연스러운 방식일 때 더 선호한다는 사실을 발견했습니다. 이는 AI 시스템이 인간이 생성한 수학의 의사소통 의도를 학습하고 표현하는 데 기여할 수 있음을 나타냅니다.

- **Performance Highlights**: 언어 모델들은 같은 방정식에 대해 다르게 해석하며, 문제를 구성할 때 비대칭성을 인식합니다. 또한, 정당한 수학적 증명들의 배열 순서에 대해 인간처럼 자연적인 방식의 선호를 보였습니다. 이러한 결과는 수학적 의사소통에서 맥락을 완전히 무시하지 말고 이를 고려해야 한다는 점을 강조합니다.



### Harnessing Diversity for Important Data Selection in Pretraining Large Language Models (https://arxiv.org/abs/2409.16986)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 사전 훈련을 위한 데이터 선택 접근 방식인 	exttt{Quad}를 제안합니다. 이 방법은 데이터의 질과 다양성을 동시에 고려하여 모델 성능을 극대화하는 데 기여합니다.

- **Technical Details**: 	exttt{Quad}는 주어진 데이터셋을 유사한 데이터 인스턴스를 클러스터링하고, 각 클러스터에서 샘플을 선택하여 측정 비용을 절감합니다. 영향을 평가하기 위해 iHVP(computation methods)를 주의(attention) 레이어에 맞게 수정하였습니다.

- **Performance Highlights**: 	exttt{Quad}는 고품질 데이터를 선택하는 동시에 데이터의 다양성을 유지하여 LLM의 성능을 최적화합니다. 이 접근 방식은 데이터 선택 과정에서 영향 점수를 활용하여 클러스터의 품질을 높이고, MAB(Multi-Armed Bandit) 방법론을 통해 효율성을 극대화합니다.



### AXCEL: Automated eXplainable Consistency Evaluation using LLMs (https://arxiv.org/abs/2409.16984)
- **What's New**: 이 논문에서는 LLM을 활용하여 텍스트 응답의 일관성을 평가하는 새로운 접근 방식인 AXCEL(Automated eXplainable Consistency Evaluation using LLMs)을 소개합니다. AXCEL은 프롬프트 기반으로 작동하며, 일관성 점수에 대한 설명을 제공하여 사용자가 추론할 수 있도록 돕습니다.

- **Technical Details**: AXCEL는 Chain of Thought (CoT)와 few shot prompting 기술을 사용하여 텍스트의 일관성을 측정합니다. 기존의 메트릭에 비해 AXCEL은 설명 가능성을 제공하며, 특정 태스크에 맞지 않아도 여러 태스크에 적용할 수 있는 일반화 가능성을 가지고 있습니다. AXCEL은 요약, 자유 텍스트 생성, 데이터-텍스트 변환의 세 가지 태스크에서 실험되었습니다.

- **Performance Highlights**: AXCEL은 요약에서는 8.7%, 자유 텍스트 생성에서는 6.2%, 데이터-텍스트 변환 태스크에서는 29.4% 향상된 성능을 보이며, 기존의 non-prompt 및 prompt 기반 최신 메트릭을 초월했습니다. 또한 AXCEL은 오픈 소스 LLM을 사용하더라도 강력한 성능을 보여줍니다.



### Informed deep hierarchical classification: a non-standard analysis inspired approach (https://arxiv.org/abs/2409.16956)
- **What's New**: 이번 연구는 다계층 분류 작업을 위한 새로운 접근 방식을 제안합니다. 이 방법은 다중 레이블로 구성되고 엄격한 부모-자식 구조로 조직된 데이터를 분류하는 문제에 초점을 맞추고 있습니다. 발표된 접근 방식은 lexicographic hybrid deep neural network (LH-DNN)라는 다중 출력 심층 신경망 구조를 포함합니다.

- **Technical Details**: 이 LH-DNN 아키텍처는 lexicographic multi-objective optimization(선형 다목적 최적화), non-standard analysis(비표준 해석학), deep learning(심층 학습) 등의 다양한 연구 분야의 도구를 결합하여 설계되었습니다. 이 접근법은 데이터 계층 구조와 일치하도록 학습 과정을 조정하는 방식으로, 학습 파라미터, 훈련 에폭 및 계산 시간을 대폭 줄이며 성능은 B-CNN과 유사하거나 우수할 수 있다는 것을 보여줍니다.

- **Performance Highlights**: LH-DNN은 CIFAR10, CIFAR100 및 Fashion-MNIST 벤치마크에서 B-CNN과 비교하여 학습 효율성이 높고 계층 관계를 학습하는 데 강력한 성능을 발휘합니다. 이 방식은 별도의 손실 함수 가중치 없이도 적용됩니다.



### Setting the AI Agenda -- Evidence from Sweden in the ChatGPT Era (https://arxiv.org/abs/2409.16946)
Comments:
          This paper is part of the Second AEQUITAS Workshop on Fairness and Bias in AI | co-located with ECAI 2024, October 19--24, 2024, Santiago de Compostela, Spain

- **What's New**: 이번 연구는 스웨덴에서 ChatGPT 출시 이전과 이후의 인공지능(AI) 메타 논의의 발전을 조사합니다. 연구자는 정치 엘리트가 상대적으로 침묵하고 AI 논의가 학계에서 주도되고 있다는 점을 강조합니다.

- **Technical Details**: 정량적 데이터셋을 기반으로 한 질적 내용 분석을 수행하여 스웨덴의 주요 신문에 실린 엘리트 의견 기사를 분석했습니다. 연구는 단기 리스크와 장기 리스크의 논의 경향을 분류하였으며, AI에 대한 논의가 점점 더 실질적이고 리스크 지향적으로 변화하고 있음을 보여줍니다.

- **Performance Highlights**: 2022년 이후 AI 관련 논의에서 단기 리스크에 대한 강조가 증가했음을 나타내며, 이는 스웨덴 내 정치 엘리트보다 학문적 엘리트가 AI 논쟁을 주도하고 있음을 시사합니다.



### Quantum-Classical Sentiment Analysis (https://arxiv.org/abs/2409.16928)
Comments:
          Submitted to BigHPC 2024 - this https URL

- **What's New**: 이번 연구에서는 감정 분석(Sentiment Analysis)을 위한 혼합 고전-양자 분류기(Hybrid Classical-Quantum Classifier, HCQC)의 적용을 조사하며, HCQC의 성능을 고전적 CPLEX 분류기 및 Transformer 아키텍처와 비교했습니다. HCQC는 Transformer에 비해 분류 정확도는 떨어지지만, 상대적으로 빠르게 근사해결에 도달할 수 있다는 점이 발견되었습니다. 또한 D-Wave의 비공개 특성으로 인한 HCQC의 아키텍처 상에서의 병목 현상도 밝혀졌습니다.

- **Technical Details**: 자연어 처리(Natural Language Processing)에서 높은 품질의 데이터를 확보하는 것과 모델의 훈련 시간이라는 두 가지 주요 도전이 있습니다. 우리는 비전통적인 컴퓨팅 아키텍처를 사용하여 훈련 시간을 단축시켜 더 표현력이 뛰어난 모델을 얻는 것에 초점을 맞추었습니다. 이를 위해 D-Wave의 아디아바틱 양자 컴퓨터(Adiabatic Quantum Computing, AQC) 아키텍처를 선택하였으며, 여기서는 QUBO(Quadratic Unconstrained Binary Optimization) 형태로 감정 분석을 수행합니다. 실험에서는 TweetEval 데이터셋을 사용했으며, 감정 클래스를 세 가지(positive, negative, neutral)로 나눈 후, neutral 클래스를 제외하고 positive와 negative의 균형 잡힌 데이터셋을 생성했습니다.

- **Performance Highlights**: RoBERTa는 94.3%의 높은 정확도로 분류 결과를 제공했으며, HCQC의 D-Wave 솔루션은 각각 76.1%의 정확도를 보였습니다. D-Wave의 최적 할당을 찾는 데 소요된 시간은 39.2초로, CPLEX의 101.9초보다 60% 적었습니다. 또한, RoBERTa는 예측을 위해 136.8초가 소요되어 CPLEX나 D-Wave보다 더 많은 시간이 요구되었습니다. 이 연구는 QPU의 활용도를 높이기 위해 새로운 하이브리드 솔버(웨이브 문제)를 개발하는 방법도 제안하고 있습니다.



### AI-assisted Gaze Detection for Proctoring Online Exams (https://arxiv.org/abs/2409.16923)
Comments:
          Accepted to HCOMP-24 Works-in-Progress and Demonstration track

- **What's New**: 이번 연구에서는 고위험 온라인 시험에서 시험 응시자가 화면에서 시선을 돌리는지를 감지하는 AI 보조 시스템을 제안하였습니다. 이 시스템은 감독관이 비디오 프레임 간 탐색을 통해 의심스러운 순간을 더 효과적으로 식별할 수 있도록 지원합니다.

- **Technical Details**: AI 보조 시선 감지 시스템은 시험 응시자의 각 프레임에서 예측된 시선 방향을 산포도로 표시합니다. 감독관은 시선 플롯에서 특정 영역을 선택하고, 관련된 비디오 타임스탬프를 강조 표시하여 비디오 프레임을 보다 효율적으로 탐색할 수 있습니다.

- **Performance Highlights**: 사용자 연구를 통해, 본 시스템은 기존의 인간 중심 프로토콜과 기계 학습 기반 프로토콜보다 더 효과적인 성능을 보였으며, 감독관이 보다 정교한 판단을 내릴 수 있도록 지원함을 입증하였습니다.



### Tell Me What You Don't Know: Enhancing Refusal Capabilities of Role-Playing Agents via Representation Space Analysis and Editing (https://arxiv.org/abs/2409.16913)
- **What's New**: 이 연구는 Role-Playing Agents(RPAs)의 역할 지식과 충돌하는 쿼리에 대한 응답 정확도를 향상시키기 위해 경량화된 표현 편집 방법을 소개합니다. 이를 통해 RPAs의 반려 능력을 강화하고 일반적인 역할 수행 능력을 유지할 수 있음을 보였습니다.

- **Technical Details**: 연구에서는 두 가지 주요 충돌 범주(역할 맥락 지식 충돌, 역할 파라메트릭 지식 충돌)로 쿼리를 분류하고 각각 중 4가지 특정 시나리오를 고려하여 RPA의 반려 능력을 평가하기 위한 평가 기준을 구축했습니다. 또한, 내부 표현의 rejection region과 direct response region을 분석하여 각 쿼리에 따른 반응 패턴을 규명하였습니다.

- **Performance Highlights**: 실험 결과, 최신 모델(GPT-4, Llama-3 등) 사이에서 다양한 충돌 시나리오에 따른 반려 능력의 유의미한 차이를 발견하였으며, 제안한 표현 편집 방법이 충돌 요청을 효과적으로 반려할 수 있도록 도와주었습니다. 이는 RPAs의 역할 수행 능력을 저하시키지 않으면서 반려 능력을 향상시키는 데 기여하였습니다.



### AI-driven View Guidance System in Intra-cardiac Echocardiography Imaging (https://arxiv.org/abs/2409.16898)
- **What's New**: 이번 연구에서는 Intra-cardiac Echocardiography (ICE) 이미징을 위한 인공지능 기반의 폐쇄 루프(view guidance system)를 제안합니다. 이 시스템은 사용자에게 전문 지식 없이도 ICE 이미징을 지원하도록 설계되었습니다. 특히, 경험이 적은 사용자가 ICE 카테터를 효과적으로 조종할 수 있도록 돕습니다.

- **Technical Details**: 제안된 시스템은 임의의 뷰와 임상적으로 정의된 ICE 뷰 간의 상대적 위치 및 방향 벡터를 모델링합니다. 사용자에게 현재 뷰에서 원하는 뷰로의 전환을 안내하며, 이는 폐쇄 루프 구성에서 작동하여 ICE 카테터 상태를 예측하고 업데이트합니다. 시스템은 기존 임상 워크플로우를 방해하지 않고 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 시뮬레이션 기반 평가에서 6532 테스트 데이터셋을 통해 89%의 성공률을 기록하였습니다. 이는 제안된 시스템이 ICE 이미징 절차의 정확성과 효율성을 향상시킬 수 있는 가능성을 강조합니다.



### Automating Traffic Model Enhancement with AI Research Agen (https://arxiv.org/abs/2409.16876)
Comments:
          19 pages, 10 figures

- **What's New**: 새로운 트래픽 모델 개발을 위해 TR-Agent라는 AI 기반 시스템을 도입했습니다. 이 시스템은 기존의 비효율적인 수작업 프로세스를 대체하고 연구 효율성을 높이기 위해 구조적이고 반복적인 파이프라인을 통해 자율적으로 트래픽 모델을 개발하고 개선할 수 있도록 설계되었습니다.

- **Technical Details**: TR-Agent는 아이디어 생성(Idea Generation), 이론 формuliering, 이론 평가, 반복 최적화(Iterative Optimization)라는 네 가지 주요 단계로 연구 파이프라인을 세분화하며, 각 단계에 맞는 네 가지 모듈(아이디어 생성기, 코드 생성기, 평가기, 분석기)을 활용합니다. 아이디어 생성기는 Retrieval-Augmented Generation (RAG) 기술을 통해 기존 모델의 결함을 식별하고 개선점을 제안합니다. 코드 생성기는 이러한 아이디어를 실행 가능한 파이썬 함수로 변환하고 반복적으로 디버깅합니다. 평가기는 새로 생성된 모델 함수를 평가하며, 분석기는 성능을 분석하고 추가 개선을 위한 피드백을 제공합니다.

- **Performance Highlights**: TR-Agent는 전통적인 모델 대비 25%, 75%, 90% 향상을 알고리즘별로 달성하였으며, IDM, MOBIL, LWR 모델에서 유의미한 성능 개선을 나타냈습니다. 연구자는 TR-Agent의 최적화 과정을 통해 상세한 분석 결과와 개선 사항을 확인할 수 있으며, 이를 통해 모델 개발을 보다 빠르고 효율적으로 할 수 있습니다.



### Ethical and Scalable Automation: A Governance and Compliance Framework for Business Applications (https://arxiv.org/abs/2409.16872)
- **What's New**: AI 적용의 대중화가 이루어짐에 따라 기업에서 윤리 (ethics), 거버넌스 (governance), 법적 준수 (legal compliance)와 관련된 주요 도전 과제가 발생하고 있습니다. 이 논문에서는 윤리적 (ethical), 제어 가능 (controllable), 실행 가능 (viable), 그리고 바람직한 (desirable) AI를 보장하기 위한 프레임워크를 소개합니다.

- **Technical Details**: 제안된 프레임워크는 성능 (performance)과 설명 가능성 (explainability) 간의 균형을 맞추어야 하며, 금융 (finance) 및 의료 (healthcare)와 같은 분야에서 규제 요구 사항을 충족할 수 있도록 실제적인 조언을 제공합니다. 대규모 언어 모델 (large language models) 사용 사례를 통해 환경 문제에 대한 태도를 모방하는 합성 의견 (synthetic opinions)을 생성하는 비용 효과적인 대안을 제시합니다.

- **Performance Highlights**: 프레임워크의 유효성을 다양한 사례 연구 (case studies)를 통해 검증하였으며, Chi-test 점수 (Chi-test scores), 정규화된 상호 정보 (normalized mutual information), 그리고 Jaccard 지수 (Jaccard indexes)와 같은 지표를 사용하여 합성 데이터와 기대 분포 간의 정렬 (alignment)을 정량화하였습니다. 향후 연구에서는 다양한 산업 환경에서 프레임워크의 경험적 검증 (empirical validation)을 추가로 탐색할 필요가 있습니다.



### Multi-objective Evolution of Heuristic Using Large Language Mod (https://arxiv.org/abs/2409.16867)
- **What's New**: 이 논문에서는 다목적 최적화(multi-objective optimization) 문제로서 휴리스틱(h heuristics) 검색을 모델링하는 새로운 접근 방식을 제안합니다. 기존 연구들은 최적 성능만을 목표로 했으나, 본 연구는 효율성과 확장성 등의 실용적인 기준도 고려합니다.

- **Technical Details**: 제안된 프레임워크는 다목적 진화 휴리스틱(Multi-objective Evolution of Heuristic, MEoH)으로, 대형 언어 모델(large language models, LLMs)을 활용하여 코드 비유사성(code dissimilarity) 및 목표 공간의 지배성(dominance)을 모두 고려한 새로운 지배-비유사성 메커니즘을 설계합니다.

- **Performance Highlights**: MEoH는 온라인 빈 포장 문제(online Bin Packing Problem, BPP)와 외판원 문제(Traveling Salesman Problem, TSP)에서 입증되었으며, 단일 실행으로 다양한 엘리트 휴리스틱(heuristics)을 자동 생성하여 기존 방법들보다 더 많은 트레이드오프 옵션을 제공합니다. 효율성이 최대 10배 향상되었으며, 다목적 검색은 휴리스틱 설계에 대한 새로운 통찰력을 제공하고 다양한 휴리스틱 발견으로 이어집니다.



### Dispute resolution in legal mediation with quantitative argumentation (https://arxiv.org/abs/2409.16854)
- **What's New**: 본 논문에서는 Mediation(중재)의 독특한 역학을 고려하여 QuAM(Quantitative Argumentation Mediate) 프레임워크를 도입하여 논의의 목표 수용성을 평가하는 새로운 방법론을 제안합니다.

- **Technical Details**: QuAM은 당사자들의 지식과 중재자의 법적 기준과 사실을 통합하여 중재 목표의 수용성을 결정하는 프레임워크입니다. 또한, 목표 인수의 수용 가능성과 관련 변수가 부여된 값 간의 관계를 모델링하는 새로운 형식을 개발합니다.

- **Performance Highlights**: QuAM 프레임워크는 법적 중재 과정 전반에 걸쳐 중재자를 지원하고, 각 당사자 간의 협략을 보다 효율적으로 관리할 수 있도록 합니다.



### Exposing Assumptions in AI Benchmarks through Cognitive Modelling (https://arxiv.org/abs/2409.16849)
Comments:
          11 pages, 2 figures

- **What's New**: 본 논문에서는 문화적 AI 벤치마크가 종종 측정된 구성 요소에 대한 암묵적인 가정에 의존하고 있다는 점을 지적하며, 이러한 가정들을 명시적인 인지 모델을 통해 드러내고자 합니다. 구조 방정식 모델(Structural Equation Models; SEM)을 활용하여, 누락된 데이터셋을 식별하고 연구 질문에 답할 수 있는 방법을 제시합니다.

- **Technical Details**: 세부적으로, 우리는 LLM(대형 언어 모델)의 ‘특성’에 관한 명시적 모델링을 통해 심리 측정(psychometrics)에서 영감을 받은 접근 방식을 확장하고 있습니다. 우리의 구조 방정식 모델은 언어 능력, 문화 지식, 정렬(alignment) 간의 관계를 분석합니다. 이 모델은 교차 언어 정렬 이전의 명확한 가정을 드러내어, 데이터셋 개발을 위한 방향을 제시합니다.

- **Performance Highlights**: 본 프레임워크는 기존의 벤치마크와 이론적 구성 간의 관계를 명확히 하며, 다양한 테스트가 잘 측정하는지를 평가할 수 있는 가능성을 열었습니다. 이는 LLM 특성에 대한 더 엄격하고 이론적으로 정당화된 이해를 위한 길을 제시하고, 결과적으로 생성적 AI 시스템의 포괄적이고 세밀한 평가를 촉진합니다.



### PeerArg: Argumentative Peer Review with LLMs (https://arxiv.org/abs/2409.16813)
- **What's New**: 이 논문은 PeerArg 시스템을 제안하여 LLM(대형 언어 모델)과 지식 표현 방법을 결합하여 동료 검토(peer review) 과정을 지원하고 이해할 수 있는 새로운 파이프라인을 소개합니다. PeerArg는 논문에 대한 리뷰 세트를 입력으로 받아 논문 수락 예측을 출력합니다.

- **Technical Details**: PeerArg는 컴퓨터 논증(computational argumentation) 방법을 사용하여 리뷰 이해를 향상시키고, 논문의 수락 결정 프로세스와 리뷰 간의 일치를 평가합니다. 여기서는 이진 논증 프레임워크(bipolar argumentation frameworks)를 활용하여 리뷰와 리뷰 집계 과정을 모델링합니다. 논문은 또한 몇 샷 학습(few-shot learning)을 사용하는 새로운 end-2-end LLM을 제안하여 주어진 리뷰로부터 논문 수락을 예측합니다.

- **Performance Highlights**: Ethical argumentation 통해 LLM을 강화하면 성능이 향상되며, PeerArg 파이프라인이 특정 하이퍼파라미터 조합을 사용할 경우, 모든 데이터 세트에서 LLM보다 더 우수한 성능을 발휘한다는 결과를 보여주었습니다.



### Large Language Model Predicts Above Normal All India Summer Monsoon Rainfall in 2024 (https://arxiv.org/abs/2409.16799)
Comments:
          3 figures

- **What's New**: 이번 연구에서는 AISMR(All India Summer Monsoon Rainfall)의 예측 정확도를 높이기 위해 최신 LLM 모델인 PatchTST를 조정하고 세부 조정을 했습니다.

- **Technical Details**: PatchTST 모델은 과거 AISMR 데이터, Niño3.4 지수 및 카테고리별 인도양 쌍극자(Indian Ocean Dipole) 값을 사용하여 훈련되었으며, 그 결과 여러 인기 있는 신경망 모델 및 통계 모델보다 더 높은 성능을 보여줍니다.

- **Performance Highlights**: Fine-tuned PatchTST 모델은 0.07%의 RMSE(Root Mean Square Error)와 0.976의 Spearman 상관관계를 기록하며, 이는 가장 성능이 좋은 신경망 모델보다 약 80% 더 높은 정확도를 나타냅니다. 이 모델은 2024년 몬순이 정상 이상으로 예상되며, 전체 국가에 대해 6월에서 9월까지 총 921.6mm의 강수량을 예측하고 있습니다.



### LLaMa-SciQ: An Educational Chatbot for Answering Science MCQ (https://arxiv.org/abs/2409.16779)
- **What's New**: LLaMa-SciQ라는 교육용 챗봇을 개발해 대학생들이 STEM 분야의 수학적 문제를 해결하고 이해하도록 돕고 있습니다.

- **Technical Details**: LLaMa-SciQ는 Mistral-7B 및 LLaMa-8B 모델을 비교하여 더 높은 평가 정확도를 보이는 LLaMa-8B 모델을 선택했습니다. 모델의 정확성을 높이기 위해 Retrieval-Augmented Generation (RAG) 방식을 도입하고, 양자화를 통해 모델을 압축하여 추론 시간을 단축했습니다.

- **Performance Highlights**: LLaMa-SciQ는 GSM8k 데이터셋에서 74.5%의 정확도를 달성하였고, MATH 데이터셋에서는 30%의 정확도를 보였습니다. 양자화된 모델은 성능이 5%만 낮아져 효율성을 크게 향상시켰습니다.



### Non-stationary BERT: Exploring Augmented IMU Data For Robust Human Activity Recognition (https://arxiv.org/abs/2409.16730)
- **What's New**: 이 연구에서는 OPPOHAR라는 새로운 인간 활동 인식(HAR) 데이터셋을 소개합니다. 이는 모바일 장치에서 수집된 IMU 데이터로 구성되어 있으며 사용자의 특정 활동 인식을 위해 최적화된 경량 네트워크인 Non-stationary BERT를 제안합니다. 또한 가속도계와 자이로스코프 데이터 간의 깊은 관계를 탐구하기 위한 간단하면서도 효과적인 데이터 증강(data augmentation) 방법을 도입합니다.

- **Technical Details**: 논문에서는 IMU 데이터의 자기 감시(self-supervised) 사전 훈련(pretraining)과 감독(classification) 단계를 포함한 Non-stationary BERT 네트워크 설계를 제안합니다. 이 네트워크는 가속도계와 자이로스코프 데이터의 상호 관계를 고려하여 새로운 시퀀스를 생성하고, 이를 기반으로 한 데이터 증강 방법을 도입하여 HAR 성능을 향상시킵니다. 또한, 사용자 개인의 데이터를 사용한 분산 배포 최적화된 경량 네트워크를 구현합니다.

- **Performance Highlights**: 제안된 Non-stationary BERT 네트워크는 다양한 활동 인식 데이터셋에서 최신 성능(state-of-the-art performance)을 달성하였으며, 데이터 증강 방법은 광범위한 적응 가능성을 보여줍니다. 이 연구는 사용자 개인의 데이터로 훈련된 분류기를 통해 프라이버시를 보장하면서도 사용자 맞춤형 활동 인식을 가능하게 합니다.



### A Multi-Dataset Classification-Based Deep Learning Framework for Electronic Health Records and Predictive Analysis in Healthcar (https://arxiv.org/abs/2409.16721)
- **What's New**: 이 연구에서는 심장병, 간경화 및 망막 질환을 분류하기 위해 Residual Networks와 Artificial Neural Networks를 결합한 새로운 딥러닝 예측 분석 프레임워크를 제안합니다.

- **Technical Details**: 데이터는 세 가지 서로 다른 출처에서 전처리되며, 카테고리 데이터 변환, 차원 축소, 결측 데이터 합성을 포함하여 준비됩니다. 이미지 데이터셋에 대해서는 ResNet 아키텍처를 사용하여 특징 추출을 수행하고, 카테고리 데이터셋에는 스케일러 변환을 사용합니다.

- **Performance Highlights**: 망막 촬영 이미지, 간경화 단계, 심장병 진단 예측에 대한 정확도는 각각 93%, 99%, 95%로 높은 성능을 기록하였으며, F1-score, precision, recall 등의 상세한 분석을 통해 제안된 방법의 효과성을 입증하였습니다.



### A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms (https://arxiv.org/abs/2409.16694)
Comments:
          Ruihao Gong leads the overall organization of the survey, with Yifu Ding and Jinyang Du contributing to Sections 2 and 3. Xingyu Zheng is responsible for authoring Section 4, while Chengtao Lv and Zining Wang collaborate on Section 5. Haotong Qin, Jinyang Guo, Michele Magno, and Xianglong Liu provide guidance during the whole process and assist in refining the final manuscript

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 위한 저비트 양자화(low-bit quantization) 방법에 대한 종합적인 조사를 제공합니다. 이는 메모리 사용량과 계산 요구 사항을 줄여주어 LLM의 실제 구현에서 중요한 과제를 해결하는 데 기여합니다.

- **Technical Details**: 저비트 양자화는 모델의 파라미터, 활성화(activations), 그리고 그래디언트의 비트 폭을 줄이는 프로세스로, 메모리 사용량과 계산 요구를 감소시킵니다. 이 논문에서는 저비트 LLM의 기초 원리, 시스템 구현, 알고리즘 전략을 다루고 있습니다. 새로운 저비트 데이터 형식과 양자화 세분화(granularity), 정적 또는 동적 양자화의 차이점 등이 소개됩니다.

- **Performance Highlights**: 저비트 양자화는 LLM의 훈련(training) 및 추론(inference)을 가속화하며, 정확도를 유지하면서도 모델을 저장하는 데 필요한 자원을 줄이는 데 효과적입니다. 이 연구에서는 새로운 연구 분야, 잠재적인 혁신, 그리고 새로운 기술이 LLM 양자화에 미치는 영향을 논의하며, LLM의 효율성 및 적합성을 향상시키기 위한 가치 있는 통찰력을 제공합니다.



### CaBRNet, an open-source library for developing and evaluating Case-Based Reasoning Models (https://arxiv.org/abs/2409.16693)
- **What's New**: 이 논문에서는 self-explainable (자기 설명 가능한) 모델을 설계하기 위한 새로운 접근법으로 CaBRNet을 제안합니다. 이는 기존의 post-hoc (사후 분석) 방법의 한계를 극복하는 것을 목표로 합니다.

- **Technical Details**: CaBRNet은 Case-Based Reasoning Networks (사례 기반 추론 네트워크)를 위한 오픈 소스, 모듈형, 역 호환 가능한 프레임워크입니다. 이 논문은 이 프레임워크의 설계와 구현에 대한 상세한 설명을 제공합니다.

- **Performance Highlights**: CaBRNet은 높은 재현성(reproducibility) 및 비교 가능성을 보장하며, 다양한 기준(standards)에서 일관된 성능을 발휘합니다.



### MSI-Agent: Incorporating Multi-Scale Insight into Embodied Agents for Superior Planning and Decision-Making (https://arxiv.org/abs/2409.16686)
- **What's New**: 이번 논문에서는 Multi-Scale Insight Agent (MSI-Agent)를 소개합니다. 이 에이전트는 장기 기억(Long-term memory)을 개선하여 LLMs의 계획 및 의사결정 능력을 높이기 위해 다양한 스케일에서 통찰(insight)을 효과적으로 요약하고 활용하는 방식으로 설계되었습니다.

- **Technical Details**: MSI-Agent는 경험 선택기(experience selector), 통찰 생성기(insight generator), 통찰 선택기(insight selector)의 세 가지 주요 구성 요소를 통해 작동합니다. 이 세 부분으로 이루어진 파이프라인(pipeline)을 활용하여, MSI는 작업에 특화된(task-specific) 고수준의 통찰을 생성하고 이를 데이터베이스에 저장한 후, 의사결정을 위해 관련 통찰을 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, MSI는 GPT3.5에 기반한 다른 통찰 전략보다 우수한 성능을 보였습니다. 또한, 씨앗 경험(seed experience)과 통찰을 선택하는 전략을 탐구하며, LLM에 더 유용하고 관련성 있는 통찰을 제공하여 더 나은 의사결정을 지원하는 데 초점을 맞추고 있습니다. MSI는 도메인 전환(domain-shifting) 시나리오에서 더 나은 강건성을 보여주는 것으로 관찰되고 있습니다.



### Judgment of Thoughts: Courtroom of the Binary Logical Reasoning in Large Language Models (https://arxiv.org/abs/2409.16635)
- **What's New**: 이 논문은 이진 논리 추론 작업에 특화된 새로운 프롬프트 엔지니어링 기법인 Judgement of Thought (JoT)를 제안합니다. JoT는 변호사, 검사, 판사 세 가지 역할을 통해 모델의 추론을 보다 신뢰할 수 있고 정확하게 수행할 수 있도록 돕습니다.

- **Technical Details**: JoT 프레임워크는 고급 모델을 판사에게 할당하고, 저급 모델을 변호사와 검사에게 사용합니다. 이를 통해 판사는 변호사와 검사로부터의 응답을 보다 잘 이해하고, 정확한 판단을 내릴 수 있습니다. JoT는 BigBenchHard와 Winogrande와 같은 LLM 벤치마크 데이터셋에서 기존 방법론인 Chain of Thought (CoT) 및 Self-Consistency (SC)를 초과하는 성능을 보였습니다.

- **Performance Highlights**: JoT는 이진 논리 추론 작업에서 모델의 정확성과 신뢰성을 크게 향상시켰으며, Fake News Detection 및 SMS Spam Detection과 같은 실제 문제에서도 기존 기술에 비해 유사하거나 개선된 성능을 보여주었습니다. 이는 JoT가 여러 분야에서 실질적인 적용 가능성을 지닌다는 것을 시사합니다.



### On Your Mark, Get Set, Predict! Modeling Continuous-Time Dynamics of Cascades for Information Popularity Prediction (https://arxiv.org/abs/2409.16623)
- **What's New**: 이번 연구에서는 정보의 인기 예측을 위한 새로운 모델인 ConCat을 제안합니다. 이 모델은 지속적인 시간 다이나믹스를 조건부 강도 함수와 결합하여 정보의 확산 과정을 보다 정확하게 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: ConCat은 신경 Ordinary Differential Equations (ODEs)를 활용하여 비정상적인 정보의 확산 이벤트를 지속적인 시간 내에서 모델링합니다. 또한, 신경 Temporal Point Processes (TPPs)를 사용하여 정보 집합의 글로벌 트렌드를 정밀하게 포착하고, 이를 통해 예측 성능을 향상시킵니다. 이 모델은 그래프 구조와 순차적 이벤트 정보를 기반으로 작동합니다.

- **Performance Highlights**: ConCat은 세 가지 실제 데이터 세트에서 평가한 결과, 기존의 최첨단 모델에 비해 2.3%에서 33.2%까지 성능 향상을 나타냈습니다. 이는 정보의 인기 예측 태스크에 있어 뛰어난 효율성을 입증합니다.



### Entailment-Driven Privacy Policy Classification with LLMs (https://arxiv.org/abs/2409.16621)
Comments:
          8 pages, 4 figures, 3 tables

- **What's New**: 이 논문에서는 개인 정보 보호 정책의 내용을 이해하기 쉽게 분류하기 위한 기존의 방법들과는 다른 접근방식인, "entailment-driven LLM" 기반의 프레임워크를 제안합니다. 최근의 Large Language Models (LLMs) 기술 발전을 활용해, 복잡한 개인정보 보호 정책을 사용자가 이해할 수 있도록 명확하게 분류하는 방법을 제시하였으며, 평균 11.2% 향상된 F1 점수를 기록했습니다.

- **Technical Details**: 제안된 프레임워크는 개인정보 보호 정책의 단락을 12개의 카테고리(예: 첫 번째 당사자 데이터 수집/사용, 제3자 공유/수집 등)로 분류합니다. 이 과정에서, explain classifier가 분류의 이유를 생성하고, blank filler가 그 이유를 재구성하며, entailment verifier가 최종 결정을 내리는 방식으로 진행됩니다. 이러한 과정은 사람의 사고 과정을 모방하여 더 신뢰할 수 있는 출력을 제공합니다.

- **Performance Highlights**: OPP-115 데이터셋을 사용하여 실험을 수행한 결과, 제안된 방법은 기존의 LLM 방법들보다 평균 8.6%, 14.5%, 10.5% 높은 F1 점수를 기록했습니다. 또한, 예측 결과의 57.9%는 법률 전문가가 도출한 추론과 최소 50% 이상 겹치는 것으로 나타났습니다.



### Optimized Monte Carlo Tree Search for Enhanced Decision Making in the FrozenLake Environmen (https://arxiv.org/abs/2409.16620)
- **What's New**: 이 논문은 Monte Carlo Tree Search (MCTS) 알고리즘을 FrozenLake 환경에 최적화하여 적용한 새로운 구현을 제안합니다. 이 최적화는 누적 보상과 방문 수 테이블을 통합하여 효율적인 학습을 가능하게 하고, 이를 통해 기존의 방법보다 더 빠른 수렴을 달성합니다.

- **Technical Details**: 최적화된 MCTS 알고리즘은 누적 보상(Q)과 방문 수(N) 테이블, 그리고 Upper Confidence Bound for Trees (UCT) 공식을 활용하여 결정-making을 개선합니다. 이 알고리즘은 반복적인 시뮬레이션을 통해 가능 상태-행동 궤적을 대표하는 검색 트리를 구축하며, UCT 공식은 탐색과 활용의 균형을 맞추는 데 중앙적 역할을 합니다.

- **Performance Highlights**: 최적화된 MCTS는 평균 보상 0.8 및 70%의 성공률을 기록했으며, 약 10,000 에피소드 후 안정화되었습니다. 다른 알고리즘 대비 성능이 우수하여, MCTS with Policy는 평균 보상 0.4와 성공률 35%에 그쳤고, Q-Learning은 평균 보상 0.8과 성공률 60%를 기록했습니다.



### CasFT: Future Trend Modeling for Information Popularity Prediction with Dynamic Cues-Driven Diffusion Models (https://arxiv.org/abs/2409.16619)
- **What's New**: 이번 논문에서는 정보의 확산 과정에서 관찰된 패턴을 기반으로 미래의 인기 트렌드를 예측하는 새로운 접근법인 CasFT를 제안합니다. 특히, 기존의 방법들이 미래의 인기 변화를 간과하는 문제를 해결하고자 하였습니다.

- **Technical Details**: CasFT는 정보 확산(cascade) 및 신경 ODEs(neural Ordinary Differential Equations)를 활용하여 미래의 인기 상승 트렌드를 생성합니다. 이 모델은 관찰된 성장률을 기반으로 하여, 예측 시점까지의 성장률을 전파하고, 누적 인기를 계산하는 데 중점을 두었습니다.

- **Performance Highlights**: CasFT는 실제 데이터 세트에 대한 실험을 통해 기존의 최첨단 방법들보다 2.2%에서 19.3% 까지 예측 정확도가 향상되었음을 입증하였습니다.



### Enhancing disease detection in radiology reports through fine-tuning lightweight LLM on weak labels (https://arxiv.org/abs/2409.16563)
- **What's New**: 이번 연구에서는 Llama 3.1-8B와 같은 경량 LLM을 합성 레이블(synthetic labels)을 사용한 파인튜닝(fine-tuning)을 통해 개선할 수 있는 가능성을 조사했습니다. 두 개의 작업을 결합하여 각각의 지침(instruction) 데이터셋을 혼합하여 공동 훈련하였습니다. 이 접근법은 의료 분야에서 LLM의 특화 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 두 가지 라디오 로지(tasks)에 대해 경량 LLM을 파인튜닝 했습니다. 첫 번째 작업은 여러 선택지 속에서 폐 질환을 분류하는 것이고, 두 번째 작업은 라디오 로지 보고서에서 비정상 발견을 추출하는 것입니다. 고품질 합성 레이블(GPT-4o에 의해 생성된)을 사용할 경우 Llama 3.1-8B는 질병 탐지 작업에서 micro F1 점수 0.91을 달성했습니다.

- **Performance Highlights**: 경량 LLM인 Llama 3.1-8B는 저품질의 합성 레이블(예: MIMIC-CXR에서)이 사용되었을 때도 지정된 라벨과 비교하여 과속성 샘플의 정확성을 초과했습니다(micro F1 점수: 0.67 vs 0.63). 이는 모델의 강력한 기본 능력을 보여주는 것입니다.



### Dynamic-Width Speculative Beam Decoding for Efficient LLM Inferenc (https://arxiv.org/abs/2409.16560)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 추론 과정을 가속화하는 새로운 접근 방식인 dynamic-width speculative beam decoding (DSBD)를 제안합니다. DSBD는 추측 디코딩과 빔 샘플링을 통합하여 속도와 품질을 향상시키고자 합니다.

- **Technical Details**: DSBD는 작은 보조 모델을 활용해 초안 토큰을 생성하고, 그 토큰을 기반으로 대규모 모델에서 여러 후보 시퀀스를 동시에 생성하는 새로운 검증 체계를 도입합니다. 또한, 적응형 메커니즘을 통해 컨텍스트에 따라 빔 수를 동적으로 조정하고, 여러 개의 트리를 동시에 처리하는 나무 기반 병렬 검증 기술을 확장합니다.

- **Performance Highlights**: 실험 결과 DSBD는 빔 샘플링에 비해 1.5-1.9배의 속도 향상과 1.8-2.5배의 에너지 소비 감소를 달성했으며, 다운스트림 작업에서 성능 저하 없이도 한결 우수한 출력 품질을 보여주었습니다.



### Context-aware and Style-related Incremental Decoding framework for Discourse-Level Literary Translation (https://arxiv.org/abs/2409.16539)
Comments:
          7 pages, 2 figures, wmt24

- **What's New**: 이번 보고서에서는 WMT24 담론 수준의 문학 번역 과제를 위한 우리의 접근 방식을 제시하며, 중국어-영어 언어 쌍에 중점을 둡니다. 문학 텍스트의 번역은 복잡한 의미, 관용구 및 이야기 구조로 인해 도전 과제가 많습니다. 이를 해결하기 위해 Chinese-Llama2 모델을 활용하였고, Continual Pre-training (CPT) 및 Supervised Fine-Tuning (SFT)을 결합하여 이 과제를 위한 특별한 개선을 이루었습니다.

- **Technical Details**: 우리는 TP3(Three-Stages Translation Pipeline) 훈련 패러다임을 소개하며,<br>- Stage 1: 풍부한 단일 언어 데이터를 활용한 Continual Pre-training.<br>- Stage 2: 문장을 정렬한 이중 언어 문서의 Interlinear Text Format을 사용한 Continual Pre-training.<br>- Stage 3: Semantic coherence 및 stylistic consistency 향상을 위한 Supervised Fine-Tuning을 진행합니다.

- **Performance Highlights**: 실험 결과, 문장 수준 및 문서 수준의 BLEU 점수에서 유의미한 개선이 있었으며, 이로 인해 제안된 프레임워크가 문서 수준의 문학 번역의 복잡성을 해결하는 데 효과적임을 보였습니다.



### Graph Pruning Based Spatial and Temporal Graph Convolutional Network with Transfer Learning for Traffic Prediction (https://arxiv.org/abs/2409.16532)
Comments:
          14 pages, accepted by ICIAAI2023, withdrawn from proceedings

- **What's New**: 이번 연구에서는 도시화 및 인구 급증으로 인한 교통 혼잡 문제를 보다 효과적으로 해결하기 위해 새로운 Spatial-temporal Convolutional Network(TL-GPSTGN)를 제안했습니다. 이 모델은 그래프 가지치기(graph pruning) 및 전이 학습(transfer learning) 프레임워크를 기반으로 합니다.

- **Technical Details**: TL-GPSTGN은 도로 네트워크 구조와 특성 데이터의 상관관계 및 정보 엔트로피 분석을 통해 필수적인 그래프 구조와 정보를 추출합니다. 그래프 가지치기 기술을 활용하여 그래프의 인접 행렬(adjacency matrix)과 입력 특성 데이터를 처리하여 모델의 이동 성능(migration performance)을 크게 개선합니다. 그 후, 잘 특성화된 데이터를 공간-시간 그래프 컨볼루션 네트워크(spatial-temporal graph convolutional network)에 입력하여 공간적 및 시간적 관계를 포착하고 도로 상태 예측을 수행합니다.

- **Performance Highlights**: TL-GPSTGN은 실제 데이터셋에서 종합적인 테스트 및 검증을 수행하였으며, 동일한 조건 하에서 다른 일반적으로 사용되는 모델들과의 예측 성능을 비교하였습니다. 결과적으로, TL-GPSTGN은 단일 데이터셋에서 뛰어난 예측 정확도를 보였으며, 다양한 데이터셋 간의 견고한 이동 성능을 입증하였습니다.



### SynChart: Synthesizing Charts from Language Models (https://arxiv.org/abs/2409.16517)
- **What's New**: GPT-4V(O) 모델을 기반으로 한 데이터 생성 방법을 탐색하며, 다중 모달리티(multi-modality) 모델을 위한 대규모 차트 데이터셋 SynChart를 구축하였습니다. 이 데이터셋은 약 400만 개의 다양한 차트 이미지와 7500만 개 이상의 밀집 주석을 포함하고 있습니다.

- **Technical Details**: SynChart 데이터셋은 데이터 테이블, 코드, 설명, 질문-답변 세트를 포함하여 차트 이미지를 위한 고품질 주석을 제공합니다. 우리는 LLM을 활용하여 차트 시각화를 위한 코드 생성 및 차트 데이터의 다양성을 확보하였으며, 그래픽 엔진으로 Matplotlib, Seaborn, Plotly 및 Bokeh를 사용했습니다.

- **Performance Highlights**: 훈련된 4.2B 차트 전문가 모델은 ChartQA 작업에서 GPT-4O 성능에 근접하며 GPT-4V를 초과하는 성과를 기록했습니다.



### Unsupervised Text Representation Learning via Instruction-Tuning for Zero-Shot Dense Retrieva (https://arxiv.org/abs/2409.16497)
Comments:
          Accepted at DCAI24 workshop@CIKM2024

- **What's New**: 본 연구에서는 라오-블랙웰 정리를 기반으로 한 새로운 비지도 학습 방식의 텍스트 표현 학습 기법을 제안합니다. 이 방법은 사전훈련된 인코더-디코더 대형 언어 모델(LLM)을 활용하여 쿼리와 코퍼스를 효과적으로 표현할 수 있습니다.

- **Technical Details**: 연구에서는 두 단계의 자기 지시 조정(self-instructed-tuning)을 통해 비지도 방식으로 코퍼스 표현을 학습합니다. 첫 번째 단계에서는 미리 정의된 지침에 따라 질문 생성과 키워드 요약 작업을 수행해 합성적인 쿼리를 생성하고, 두 번째 단계에서는 품질 필터를 적용하여 생성된 쿼리의 성능을 향상시킵니다. 이러한 과정은 Rao-Blackwell 정리에 의해 코퍼스의 임베딩을 개선하는 데 기여합니다.

- **Performance Highlights**: 제안한 방법은 NDCG@10, MRR@100, Recall@100 등의 지표에서 세 개의 영어 데이터 세트와 한 개의 독일어 데이터 세트를 이용하여 평가되었습니다. 최종적으로 기존의 세 가지 경쟁 모델보다 성능이 개선되었으며, FLAN-T5 모델 변형을 기반으로 하여 평균적으로 성능을 약 3.34%에서 3.50% 향상시켰습니다.



### Artificial Intelligence for Secured Information Systems in Smart Cities: Collaborative IoT Computing with Deep Reinforcement Learning and Blockchain (https://arxiv.org/abs/2409.16444)
- **What's New**: 이 논문은 사물인터넷(IoT)과 블록체인(Blockchain) 기술을 통합하여 스마트 시티에서의 모바일 전송을 최적화하고 안전한 데이터 교환을 실현하는 방법에 대해 investigates(조사)합니다. 특히, 심층 강화 학습(Deep Reinforcement Learning, DRL)을 IoT 환경에 접목하여 높은 적응성 및 결정 능력을 제공합니다.

- **Technical Details**: 블록체인의 불변성(Immutable), 확장성(Scalable), 분산화(Decentralized) 솔루션이 IoT의 프라이버시(Privacy), 보안(Security), 데이터 무결성(Data Integrity) 문제를 해결하는 데 어떻게 기여하는지를 고찰합니다. 이 논문은 2015년부터 2024년까지 발표된 연구들을 기반으로 다양한 접근 방식을 분류하고 실용적인 분류 체계(Taxonomies)를 제공합니다.

- **Performance Highlights**: DRL과 블록체인의 조합은 IoT 네트워크의 성능을 향상시키며, 프라이버시와 보안을 유지합니다. 본 연구는 블록체인의 분산 프레임워크와 DRL의 결합이 모바일 전송 효율성을 향상시키고 Robust하며 프라이버시 보호가 가능한 IoT 시스템을 보장할 수 있음을 보여줍니다.



### HAICOSYSTEM: An Ecosystem for Sandboxing Safety Risks in Human-AI Interactions (https://arxiv.org/abs/2409.16427)
Comments:
          Both the second and third authors contributed equally

- **What's New**: HAICOSYSTEM은 다양한 사회적 상호작용 내에서 AI 에이전트의 안전성을 평가하기 위해 다차원적 평가 프레임워크를 제공합니다. 이는 모듈식 샌드박스 환경을 통해 멀티 턴 상호작용을 시뮬레이션하고, AI 에이전트가 다양한 도구를 사용하여 복잡한 시나리오를 탐색할 수 있도록 합니다.

- **Technical Details**: HAICOSYSTEM은 92개의 시나리오를 바탕으로 1840회의 시뮬레이션을 실행하여, AI 에이전트와 인간 사용자 간의 상호작용을 실재감 있게 모사합니다. 평가 프레임워크 HAICOSYSTEM-EVAL은 안전성과 성능을 동시에 측정하며, 법적 위험과 같은 다양한 안전 위험 차원도 포함됩니다.

- **Performance Highlights**: 상태-of-the-art LLM들이 50% 이상의 경우 안전 위험을 보이는 것으로 나타났으며, 특히 악의적인 사용자와의 상호작용에서 위험이 증가합니다. HAICOSYSTEM은 향후 연구 및 AI 에이전트의 안전 생태계 구축에 기반을 제공합니다.



### Design and Evaluation of a CDSS for Drug Allergy Management Using LLMs and Pharmaceutical Data Integration (https://arxiv.org/abs/2409.16395)
- **What's New**: HELIOT라는 혁신적인 임상 의사결정 지원 시스템(CDSS)이 소개되었으며, 이는 약물 알레르기 관리에 초점을 맞추고 있습니다.

- **Technical Details**: HELIOT는 Large Language Models (LLMs)와 포괄적인 제약 데이터 저장소를 통합하여 고급 자연어 처리(Natural Language Processing, NLP) 기능을 활용합니다. 이를 통해 복잡한 의료 텍스트를 해석하고 비정형 데이터를 종합할 수 있습니다.

- **Performance Highlights**: HELIOT는 합성 환자 데이터 세트와 전문가 확인된 기초 진실을 사용한 실증 평가에서 높은 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수를 보여주었으며, 여러 실험에서 100	ext{%}에 도달하는 결과를 보였습니다.



### Rao-Blackwellized POMDP Planning (https://arxiv.org/abs/2409.16392)
- **What's New**: 본 연구는 Rao-Blackwellized POMDP (RB-POMDP) 근사 해법을 도입하고, 신념 업데이트(belief updates) 및 온라인 계획(online planning)에서 Rao-Blackwellization을 적용하기 위한 일반적인 방법들을 제시합니다.

- **Technical Details**: Partially Observable Markov Decision Processes (POMDPs)는 불확실성 하에서 의사 결정을 위한 구조화된 프레임워크를 제공합니다. Sequential Importance Resampling Particle Filters (SIRPF)는 대규모 근사 POMDP 풀링기에서 신념 업데이트에 흔히 사용되지만, 시스템의 상태 차원이 증가함에 따라 입자 부족(particle deprivation) 및 높은 계산 비용(computational costs) 등의 문제에 직면합니다.

- **Performance Highlights**: 시뮬레이션된 로컬라이제이션 문제에서 SIRPF와 Rao-Blackwellized Particle Filters (RBPF)의 성능을 비교한 결과, RBPF가 적은 수의 입자로 시간이 지남에 따라 정확한 신념 근사를 유지하며, 더욱 놀라운 점은 RBPF가 사각형 기반 통합(quadrature-based integration)과 결합될 경우 SIRPF 기반 계획보다 질적으로 planning 품질이 크게 향상된다는 것을 확인했습니다.



### Beyond Text-to-Text: An Overview of Multimodal and Generative Artificial Intelligence for Education Using Topic Modeling (https://arxiv.org/abs/2409.16376)
- **What's New**: 이 연구에서는 교육 분야에서의 생성형 인공지능(Generative AI)과 다중모달(multimodal) 접근법 연구의 현황을 분석하였습니다. 기존 연구는 주로 텍스트-텍스트 모델에 중점을 두고 있어 다중모달 기술의 잠재력을 간과하고 있다는 점을 지적합니다.

- **Technical Details**: 4175개의 논문에 대한 주제 모델링(topic modeling)을 사용하였으며, 38개의 해석 가능한 주제를 14개의 테마 영역으로 구성하였습니다. 이 과정에서 BERTopic 접근방식을 적용해 문서의 잠재적 주제를 추출하였습니다.

- **Performance Highlights**: 결과적으로, 다중모달 접근법과 생성형 AI가 교육에 어떤 일관된 연구 방향성을 제공하는지를 보여주며, 다채로운 인공지능 기술이 교육 분야에서 어떻게 활용될 수 있는지를 탐구할 수 있는 기회를 제시합니다.



### WeatherFormer: Empowering Global Numerical Weather Forecasting with Space-Time Transformer (https://arxiv.org/abs/2409.16321)
- **What's New**: WeatherFormer라는 새로운 transformer 기반의 수치 기상 예측(NWP) 프레임워크를 제안하여 데이터 기반 NWP의 성능 격차를 줄이는데 기여하고 있습니다.

- **Technical Details**: WeatherFormer는 공간-시간 요인 분해 transformer 블록을 사용하여 파라미터 및 메모리 소비를 줄이고, 위치 인식 적응형 푸리에 신경 연산자(PAFNO)를 도입하여 위치에 민감한 토큰 혼합을 수행합니다. 또한 두 가지 데이터 증강 전략을 통해 성능을 향상시키고 훈련 소비를 저감합니다.

- **Performance Highlights**: WeatherBench 데이터셋에서의 광범위한 실험 결과, WeatherFormer는 기존 깊은 학습 방법들보다 뛰어난 성능을 발휘하였으며, 최신 물리 모델과 더욱 근접한 성능을 서술하고 있습니다.



### Differential Privacy Regularization: Protecting Training Data Through Loss Function Regularization (https://arxiv.org/abs/2409.17144)
- **What's New**: 본 논문에서는 Neural Network 기반의 머신러닝 모델이 민감한 정보를 노출하지 않도록 하기 위해 기존의 DP-SGD(Differentially Private Stochastic Gradient Descent) 알고리즘의 효율성을 개선한 새로운 정규화 전략인 PDP-SGD를 제안하였다.

- **Technical Details**: 본 연구는 PDP-SGD라는 새로운 접근 방식을 통해 손실 함수의 정규화로 차별적 프라이버시(Differential Privacy)를 구현할 수 있음을 보여준다. PDP-SGD는 네트워크 파라미터와 입력값에 직접적으로 의존하는 정규화 항을 포함하여 기울기 누출(Gradient Leakage) 공격에 대한 방어를 구현한다.

- **Performance Highlights**: PDP-SGD는 Gaussian noise의 명시적 도입 없이도 효율적으로 동작하며, 기존 DP-SGD가 직면했던 정확도 저하 문제를 완화하는 가능성을 제시한다. 또한 이 방식은 컴퓨팅 비용을 줄이는 데에도 기여할 수 있다.



### Attention Prompting on Image for Large Vision-Language Models (https://arxiv.org/abs/2409.17143)
Comments:
          Website, see this https URL

- **What's New**: 본 연구에서는 Attention Prompting on Image (𝒜⁢𝒫⁢ℐ𝒜𝒫ℐ)라는 새로운 프롬프트 기법을 제안하여, 원본 이미지 위에 텍스트 쿼리 기반 주의 열지도를 오버레이함으로써 LVLM의 성능을 향상시킵니다.

- **Technical Details**: 이 기법은 보조 LVLM 모델을 활용하여 입력 이미지에 대한 주의 열지도를 생성합니다. 이때 CLIP과 같은 이미지-텍스트 매칭 모델의 cls 토큰 유사도 점수를 기반으로 하여 열지도를 만들어 냅니다. 생성된 열지도는 원본 이미지의 픽셀 값에 단순히 곱해져 LVLM의 실제 입력 이미지가 됩니다.

- **Performance Highlights**: Attention Prompting on Image는 LLaVA-1.5 모델의 MM-Vet, LLaVA-Wild 벤치마크에서 각각 3.8% 및 2.9%의 성능 향상을 보여줍니다.



### FineZip : Pushing the Limits of Large Language Models for Practical Lossless Text Compression (https://arxiv.org/abs/2409.17141)
- **What's New**: FineZip는 전통적 텍스트 압축 방법에 비해 54배 빠른 압축 시간을 달성함으로써 대규모 텍스트 압축에 대한 사용 가능성을 높입니다.

- **Technical Details**: FineZip는 '온라인' 및 '오프라인' 구성 요소를 결합하여 손실 없는 텍스트 압축을 수행하며, 파라미터 효율적 미세 조정(PEFT) 방식을 사용하여 압축하는 데이터를 Memorize(기억)합니다. 또한, 동적 컨텍스트 사이즈를 활용하여 각 토큰의 압축을 개선하고 병렬 처리 가능성을 높였습니다.

- **Performance Highlights**: FineZip는 기존의 LLMZip보다 약 54배 빠른 압축 성능을 보여주며, 압축 비율을 약 50% 향상시킵니다. 전통적인 알고리즘 기반 압축 방법에 비해 크게 개선된 압축 효율성을 자랑합니다.



### Blox-Net: Generative Design-for-Robot-Assembly Using VLM Supervision, Physics Simulation, and a Robot with Res (https://arxiv.org/abs/2409.17126)
Comments:
          8 pages, 7 Figures

- **What's New**: 이번 논문에서는 Generative Design-for-Robot-Assembly (GDfRA)라는 새로운 문제를 제안합니다. 이 과정은 자연어 프롬프트(예: '기린')와 3D 프린팅한 블록과 같은 물리적 부품의 이미지를 기반으로 조립을 생성하는 것입니다. 이를 통해 로봇이 조립을 보다 효과적으로 수행할 수 있도록 설계되었습니다.

- **Technical Details**: Blox-Net은 VLM(vision language model)과 시뮬레이션, 물리적 로봇 실험을 결합하여 GDfRA 문제를 해결하는 시스템입니다. 3단계로 구성되어 있으며, 각 단계에서 3D 부품의 적절한 배열을 디자인하고, 이를 물리적 로봇이 구축할 수 있도록 검증합니다. 이 시스템은 인간의 개입 없이도 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: Blox-Net은 조립물의 '인지성'에서 63.5%의 Top-1 정확도를 기록했습니다. 자동화된 재설계를 거친 후, 로봇은 10회 연속 조립에서 거의 완벽한 성공률을 보였습니다. 특히, 99.2%의 정확도로 블록을 자율적으로 배치하는 데 성공했습니다.



### Programming Every Example: Lifting Pre-training Data Quality like Experts at Sca (https://arxiv.org/abs/2409.17115)
Comments:
          45 pages, 13 figures, 34 tables

- **What's New**: 이번 연구에서는 사전 훈련된 대형 언어 모델에 대해 새로운 접근법인 ProX(Programming Every Example)를 제안합니다. 이는 데이터 정제를 프로그래밍 작업으로 간주하여, 각 개별 예제에 대해 정교한 작업을 생성하고 실행할 수 있게 합니다.

- **Technical Details**: ProX는 모델이 각각의 데이터 예제를 정제하기 위해 필요한 작업을 프로그래밍 방식으로 정의할 수 있도록 하여, 기존의 왜곡된 데이터를 정제하는 데 필요한 유연성을 제공합니다. 이는 문자열 정규화, 데이터 세분화 등의 작업을 포함하여, 0.3B 파라미터를 가진 소형 모델도 인간 전문가와 비슷한 정제 능력을 발휘할 수 있음을 보여줍니다.

- **Performance Highlights**: ProX로 정제된 데이터로 사전 훈련된 모델은 원래 데이터나 다른 필터링 기법으로 정제된 데이터에 비해 다양한 하위 벤치마크에서 2% 이상의 성능 향상을 보였습니다. 특히, OpenWebMath 데이터 세트에서 ProX로 정제된 모델은 Mistral-7B에 비해 평균 정확도가 7.6% 개선되었고, Llama-2-7B에서는 14.6%, CodeLlama-7B에서는 20.3% 향상되었습니다.



### Unveiling Ontological Commitment in Multi-Modal Foundation Models (https://arxiv.org/abs/2409.17109)
Comments:
          Qualitative Reasoning Workshop 2024 (QR2024) colocated with ECAI2024, camera-ready submission; first two authors contributed equally; 10 pages, 4 figures, 3 tables

- **What's New**: 이번 논문은 다중 모달 심층 신경망(DNN)에서 학습된 개념의 슈퍼클래스 계층 구조를 추출하는 방법을 제안합니다. 이를 통해 질적 추론(qualitative reasoning, QR) 모델과의 검증 및 확인을 위한 단계로 나아갑니다.

- **Technical Details**: 우리는 DNN의 텍스트 입력 모달리티를 사용하여 리프 개념의 임베딩을 얻고, 이를 계층적 클러스터링을 통해 의미적 유사성을 기반으로 하는 슈퍼클래스 개념을 라벨링합니다. 제안된 방법은 다중 모달 DNN의 중간 표현에서 간단한 온톨로지를 추출하고 검증할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 의미 있는 온톨로지를 추출할 수 있고, 주어진 온톨로지와의 불일치성을 밝혀낼 수 있음을 보여주었습니다. 또한, QR 모델의 추출 및 삽입에 대한 잠재적 응용 가능성을 논의하였습니다.



### Accumulator-Aware Post-Training Quantization (https://arxiv.org/abs/2409.17092)
- **What's New**: 이 논문에서는 포스트 트레이닝 양자화 (Post-Training Quantization, PTQ) 환경에서 누산기 (accumulator)에 대한 양자화를 정식으로 연구하는 최초의 사례를 제시하고 있습니다. 'AXE'라는 새로운 프레임워크를 도입하여 PTQ 알고리즘에 오버플로우 회피 보장을 추가합니다.

- **Technical Details**: AXE는 기본적으로 레이어 단위로 가중치와 활성화 값의 양자화를 지지하는 알고리즘에 적용할 수 있도록 개발되었습니다. GPFQ 및 OPTQ라는 최신 PTQ 알고리즘 위에 AXE를 구현하면서 다단계 누산 지원을 일반화했습니다. 이는 대형 언어 모델(LLM)에서도 최적화 가능성을 열어줍니다.

- **Performance Highlights**: AXE는 이미지 분류 및 언어 생성 모델을 대상으로 평가되었으며, 누산기 비트 폭과 모델 정확성 간의 트레이드오프에서 기존 방법들에 비해 상당한 개선을 보여주었습니다. 특히, 다단계 누산을 목표로 할 때 Pythia 모델 세트에서 뛰어난 확장성을 입증하였습니다.



### Ctrl-GenAug: Controllable Generative Augmentation for Medical Sequence Classification (https://arxiv.org/abs/2409.17091)
Comments:
          17 pages, 7 figures, 7 tables

- **What's New**: 이 논문에서는 Ctrl-GenAug라는 새로운 생성적 증강 프레임워크를 제안하여, 의료 시퀀스 분류를 위한 고도로 의미적이고 연속적인 시퀀스 생성을 지원하고 부정확하게 합성된 샘플을 억제합니다.

- **Technical Details**: Ctrl-GenAug는 다중 모달 조건 유도 시퀀스 생성기를 통해 진단 촉진 샘플을 제어 가능하게 합성하며, 시간적/입체적 일관성을 향상시키는 연속 증강 모듈을 통합합니다. 또한, 불확실한 사례를 억제하는 노이즈 합성 데이터 필터를 설계하였습니다.

- **Performance Highlights**: 세 가지 의료 데이터셋과 세 가지 패러다임에서 훈련된 11개의 네트워크를 사용한 광범위한 실험에서 Ctrl-GenAug의 효과성과 일반성이 입증되었습니다. 특히, 대표성이 부족한 고위험 군과 도메인 외 조건에서의 성능 향상을 보여주었습니다.



### SEN12-WATER: A New Dataset for Hydrological Applications and its Benchmarking (https://arxiv.org/abs/2409.17087)
Comments:
          Submitted to IEEE Transactions on Geoscience and Remote Sensing. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 연구에서는 기후 변화와 가뭄 증가 문제 해결을 위한 새로운 데이터셋인 SEN12-WATER를 소개합니다. 이 데이터셋은 제거 및 분석을 위한 새로운 end-to-end Deep Learning 프레임워크와 함께 제공됩니다.

- **Technical Details**: SEN12-WATER 데이터셋은 SAR (Synthetic Aperture Radar) polarization, 고도(elevation), 기울기(slope), 다중 스펙트럼(optical bands)을 포함한 시공간 데이터큐브(spatiotemporal datacube)입니다. 제안된 DL 프레임워크는 U-Net 아키텍처를 통한 수체(segmentation) 분할과 TD-CNN(Time-Distributed-Convolutional Neural Network)을 이용한 시계열(time series) 분석을 포함합니다.

- **Performance Highlights**: 제안된 방법론은 물리적 양(예: 물의 부피)의 시간 변화를 검토하여 물의 동역학에 대한 중요한 통찰을 제공합니다. 결과는 Precision, Recall, Intersection over Union 등과 같은 맞춤형 메트릭스를 통해 검증되었습니다.



### The Effect of Perceptual Metrics on Music Representation Learning for Genre Classification (https://arxiv.org/abs/2409.17069)
Comments:
          arXiv admin note: text overlap with arXiv:2312.03455

- **What's New**: 이 연구에서는 음악 이해 작업, 특히 장르 분류에 대한 성능을 향상시킬 수 있는 방법으로, 지각 메트릭(perceptual metrics)을 활용하는 새로운 접근법을 제안합니다. 특히, 자기 부호화기(autoencoders)로부터 추출된 특징을 사용하여 지각 손실(perceptual losses)로 훈련된 모델이 장르 분류를 개선할 수 있음을 입증했습니다.

- **Technical Details**: 지각 메트릭은 인간 관찰자의 지각 행동을 근사하는 데 설계된 객관적인 측정 지표입니다. 예를 들어, 구조적 유사도(SSIM)와 정규화된 라플라스 피라미드 거리(NLPD)와 같은 지표를 스펙트로그램(spectrograms)에 적용하여 오디오 품질에 대한 인간 평가와 더 나은 연관성을 보이는 것을 보여주었습니다. 이들 메트릭은 모델 훈련 시 손실 함수(loss function)으로 사용되어 모델의 성능을 개선할 수 있습니다.

- **Performance Highlights**: K-최근접 이웃(K-Nearest Neighbours) 분류기를 사용할 때, 전통적인 MSE(mean squared error)보다 지각 메트릭을 통한 성능 향상이 나타났습니다. 특히 로지스틱 회귀(Logistic Regression) 모델은 자기 부호화기에서 추출된 잠재 특징(latent features)을 활용할 때 높은 F1 점수를 기록했습니다. 그러나 NLPD는 군집화(clustering) 거리 측정에는 적합하지 않은 것으로 나타났으며, 이는 불필요한 정보를 제거함으로써 느리게 변화하는 부분들을 배제하기 때문입니다.



### Benchmarking Domain Generalization Algorithms in Computational Pathology (https://arxiv.org/abs/2409.17063)
- **What's New**: 이번 연구는 30개의 도메인 일반화 (Domain Generalization, DG) 알고리즘의 효과를 3개의 CPath 작업에 대해 평가하고, 새로운 다중 도메인 종양 탐지 데이터셋 (HISTOPANTUM)을 소개합니다.

- **Technical Details**: 연구에서는 7,560회의 교차 검증 (cross-validation) 실험을 통해 DG 알고리즘의 상대적인 성능을 비교하며, 최근에 제안된 pretrained foundation models와 같은 모달리티별 (modality-specific) 기법을 통합했습니다.

- **Performance Highlights**: 자기 감독 학습 (self-supervised learning) 및 염색 증강 (stain augmentation) 기법이 consistently 다른 알고리즘보다 좋은 성능을 보였으며, 연구 결과는 연구자들이 CPath 작업에 적합한 DG 접근 방식을 선택하는 데 도움을 줄 수 있습니다.



### ControlCity: A Multimodal Diffusion Model Based Approach for Accurate Geospatial Data Generation and Urban Morphology Analysis (https://arxiv.org/abs/2409.17049)
Comments:
          20 pages

- **What's New**: 이 논문에서는 다중 소스의 자원(Volunteer Geographic Information, VGI)을 활용하여 도시 건물 외형 데이터를 생성하는 새로운 접근방법인 ControlCity를 제안합니다. 이 모델은 여러 데이터 모달리티를 통합하여 더 정확하고 유용한 지리정보를 생성할 수 있습니다.

- **Technical Details**: ControlCity는 텍스트, 메타데이터, 이미지 데이터를 포함하는 'image-text-metadata-building footprint' 데이터 셋을 구축하고, 이를 기반으로 multimodal diffusion model을 사용하여 건물 외형 데이터를 생성합니다. 텍스트와 메타데이터를 정렬하여 도시의 건물 패턴을 학습하고, 개선된 ControlNet을 통해 도로 네트워크 및 토지 이용 이미지를 통합합니다.

- **Performance Highlights**: ControlCity는 전 세계 22개 도시에서 실험을 통해 평균 FID 점수 50.94를 기록하였으며, 기존 방법 대비 71.01%의 오류 감소와 38.46% 향상된 MIoU 점수를 달성했습니다. 제로샷 도시 생성을 통해 도시 구조를 정확하게 예측하고 생성할 수 있는 강력한 일반화 능력을 입증했습니다.



### GeoBiked: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design (https://arxiv.org/abs/2409.17045)
- **What's New**: 본 연구에서는 Deep Generative Models (DGM)을 공학 설계에 적용하기 위한 데이터셋 GeoBiked를 제공하며, 대규모 기초 모델을 활용하여 데이터 레이블링 자동화를 위한 방법을 제안합니다. GeoBiked 데이터셋은 4,355개의 자전거 이미지를 포함하고 있으며, 구조적 및 기술적 특징으로 주석이 달려 있습니다.

- **Technical Details**: GeoBiked 데이터셋은 이미지 생성 모델에서 추출한 통합 잠재 특징(Hyperfeatures)을 사용하여 구조적 이미지의 기하학적 대응 관계(예: 바퀴 중심 위치)를 검출하는 두 가지 자동 레이블링 기술을 조사합니다. 또한 GPT-4o를 통해 기술 이미지에 대한 다양한 설명을 생성합니다. 기술 이미지를 Diffusion-Hyperfeatures로 표현하여 기하학적 대응 관계를 측정할 수 있습니다.

- **Performance Highlights**: GeoBiked 데이터셋을 기반으로 한 두 가지 자동 레이블링 방법은, 잠재 이미지 특징의 학습된 통합을 통해 보지 못한 이미지에서 기하학적 기준점을 정확히 예측할 수 있음을 보여주며, GPT-4o를 통해 생성된 다양한 텍스트 설명은 정확한 기술 이미지를 설명합니다. 이러한 접근은 기술 이미지의 일반적인 포인트 검출 및 주석 작업에 적용 가능한 방법으로 제안됩니다.



### How to Connect Speech Foundation Models and Large Language Models? What Matters and What Does No (https://arxiv.org/abs/2409.17044)
- **What's New**: 대형 언어 모델(LLM)의 성과가 두드러진 가운데, 본 연구는 음성 인식을 위한 다양한 구성요소(SFM, adapter, LLM)가 하위 작업 성과에 미치는 영향을 최초로 분석합니다.

- **Technical Details**: 다양한 adapter 모듈과 LLM, SFM을 이용하여 자동 음성 인식(ASR) 및 음성 번역(ST) 작업을 수행하였으며, SFM과 LLM의 조합에 따라 최적의 adapter 디자인이 달라짐을 규명하였습니다.

- **Performance Highlights**: SFM의 선택에 따라 ASR와 ST 성능이 평균적으로 각각 1 WER 및 2 COMET 포인트 이상 차이가 발생하였다. 따라서 length adapter의 디자인은 선택된 SFM 및 LLM에 크게 의존하는 것이 밝혀졌습니다.



### Counterfactual Token Generation in Large Language Models (https://arxiv.org/abs/2409.17027)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)이 출력의 대안에 대해 반사실적 추론(counterfactual reasoning)을 수행할 수 있도록 하는 causal 모델을 개발했습니다. 이는 무작위 샘플링(sampling) 과정에 Gumbel-Max 구조적 인과 모델을 적용하여 구현됩니다.

- **Technical Details**: 제안한 모델은 Llama 3 8B-instruct에서 구현되었으며, 기존의 노드 상태를 유지하지 않는 상태 비저장(stateless) LLM에서 출발하여 반사실적 토큰 생성(counterfactual token generation)을 가능하게 합니다. 이를 통해 모델은 입력된 프롬프트에 따라 자동 회귀(auto-regressive) 생성 방식을 사용하여 가능한 대안을 탐색할 수 있습니다.

- **Performance Highlights**: 반사실적 토큰 생성의 질과 양을 분석한 결과, 모델이 생성한 결과는 기존 출력과 높은 유사성을 보여주었으며, 바이어스 탐지에 대한 유용성도 입증되었습니다. 이 연구는 LLM이 어떻게 세상 모델을 구성하는지에 대한 통찰력을 제공합니다.



### INT-FlashAttention: Enabling Flash Attention for INT8 Quantization (https://arxiv.org/abs/2409.16997)
- **What's New**: 본 논문에서는 FlashAttention과 양자화 방법을 통합한 첫 번째 INT8 양자화 아키텍처인 INT-FlashAttention을 소개합니다. 이 아키텍처는 Amplere GPU에서 FlashAttention의 추론 속도를 크게 향상시킵니다.

- **Technical Details**: INT-FlashAttention은 INT8 활성화 및 일반 행렬 곱셈 (GEMM) 커널을 사용하여 구성된 최적의 프로토타입으로, 첫 번째 완전 INT8 입력을 가지는 어텐션 연산자입니다. INT-FlashAttention은 훈련 후 양자화(p스트레이닝(quantization)) 구조로, 다른 데이터 형식인 INT4와 호환됩니다.

- **Performance Highlights**: 실험 결과, INT-FlashAttention은 FlashAttention-FP16 대비 72% 빠른 추론 속도를 달성하였고, FlashAttention-FP8 대비 최대 82% 더 작은 양자화 오류를 보였습니다.



### Towards User-Focused Research in Training Data Attribution for Human-Centered Explainable AI (https://arxiv.org/abs/2409.16978)
- **What's New**: 이번 연구는 Explainable AI(XAI) 분야에서 기존의 하향식(bottom-up) 접근 대신 사용자의 요구를 중심으로 한 상향식(top-down) 접근 방식을 제안합니다. 특히 Training Data Attribution(TDA) 하위 분야에 초점을 맞추어 현재 사용자가 필요로 하는 기획 및 요구 사항을 파악했습니다.

- **Technical Details**: 우리는 10명의 AI 실무자와의 인터뷰 및 31명의 모델 개발자를 대상으로 한 시스템적 조사를 통해 TDA에 대한 사용자 니즈를 분석했습니다. 이를 통해 TDA에서 현재 간과되고 있는 다양한 작업을 확인했습니다.

- **Performance Highlights**: 사용자 중심의 TDA 연구 방향을 제안하며, 기계 학습 모델의 개발자가 TDA 설명을 필요로 하고, 유연성과 안정성을 요구하는 경향이 있음을 발견했습니다. 특히, 전체 교육 데이터 집단에 대한 기여도 설명이 개인 기여도 설명보다 더 선호되는 것으로 나타났습니다.



### Decoding Large-Language Models: A Systematic Overview of Socio-Technical Impacts, Constraints, and Emerging Questions (https://arxiv.org/abs/2409.16974)
Comments:
          28 pages, 5 figures, preprint submitted to journal

- **What's New**: 최근 대형 언어 모델(LLM)의 발전이 자연어 처리(NLP)와 인공지능(AI) 분야에 혁신적인 변화를 가져왔습니다. 이 연구에서는 LLM의 개발 방향, 영향력 및 한계에 대한 체계적인 문헌 조사를 수행하였습니다.

- **Technical Details**: 논문은 LLM 연구의 목표, 방법론, 제한 사항 및 향후 방향성을 서술하며, 알고리즘 개선, 윤리적 도전 과제, 사회적 영향을 포함하여 책임 있는 개발에 대한 고려 사항을 포함합니다. 또한, 체계적인 리뷰 방법론을 통해 문헌을 분석하고, 150회 이상의 인용된 61개의 주요 논문을 선정했습니다.

- **Performance Highlights**: LLM은 번역, 분류, 질문-응답, 요약 및 정보 검색과 같은 복잡한 작업을 수행하는 데 탁월한 성능을 보여줍니다. 특히, GPT-3와 같은 모델은 창의적인 콘텐츠 생성과 대화 시뮬레이션에서 뛰어난 다양한 기능을 발휘하고 있습니다.



### Adaptive Self-Supervised Learning Strategies for Dynamic On-Device LLM Personalization (https://arxiv.org/abs/2409.16973)
Comments:
          First ASLS

- **What's New**: 이번 논문에서는 Adaptive Self-Supervised Learning Strategies (ASLS)를 제안합니다. ASLS는 대규모 언어 모델(LLMs)을 사용자 개인의 선호도에 맞게 동적으로 개인화하는 혁신적인 방법으로, 라벨이 있는 데이터셋의 필요성을 줄이고 실시간 피드백을 기반으로 모델을 조정합니다.

- **Technical Details**: ASLS는 사용자 프로파일링 레이어와 신경망 적응 레이어의 이중 레이어 구조로 구성되어 있습니다. 사용자와의 상호작용 데이터를 수집하여 모델을 실시간으로 미세 조정하며, 이는 계속해서 사용자 피드백을 학습하여 사용자별 맞춤형 응답을 생성합니다. 이 접근법은 계산 자원을 절약하고 개인화 효율성을 높입니다.

- **Performance Highlights**: 다양한 사용자 시나리오에 대한 실험 결과, ASLS는 기존의 개인화 방법 대비 사용자의 참여도와 만족도를 크게 향상시켰습니다. 이러한 결과는 ASLS가 대규모 언어 모델을 보다 반응성이 뛰어나고 맥락을 인지하는 시스템으로 변모시킬 잠재력을 보여줍니다.



### Dynamic Obstacle Avoidance through Uncertainty-Based Adaptive Planning with Diffusion (https://arxiv.org/abs/2409.16950)
- **What's New**: 이번 연구에서는 강화 학습을 시퀀스 모델링 문제로 접근하면서, 움직이는 장애물이 있는 역동적인 환경에서도 효과적으로 충돌 회피를 수행할 수 있는 적응형 생성 계획(adaptive generative planning) 방법을 제안합니다. 이 방법은 불확실성을 기반으로 리플래닝 빈도를 동적으로 조절하여, 전통적인 방법들보다 더 나은 성능을 보여줍니다.

- **Technical Details**: 제안된 방법은 생성 모델(generative models), 특히 diffusion models를 활용하여 깊은 앙상블 역다이나믹스(action dynamics) 모델로부터 얻은 불확실성 추정을 기반으로 합니다. 이를 통해 1) 정해진 튜닝 파라미터에 따라 긴 시간 계획(long-horizon planning)과 각 단계에서의 리플래닝(replanning) 간의 유연한 균형을 제공하고, 2) 불필요한 리플래닝을 줄이면서 안전성을 확보할 수 있습니다.

- **Performance Highlights**: 실험 결과, 평균 경로 길이가 13.5% 증가하고, 평균 보상(mean reward) 또한 12.7% 증가하며, 이는 충돌 비율의 감소와 안전하게 환경을 탐색할 수 있는 능력이 향상되었음을 나타냅니다.



### Go-SLAM: Grounded Object Segmentation and Localization with Gaussian Splatting SLAM (https://arxiv.org/abs/2409.16944)
- **What's New**: 새로운 Go-SLAM 프레임워크를 소개하며, 3D Gaussian Splatting SLAM을 활용하여 동적 환경을 재구성하고 장면 표현 내에서 객체 수준 정보를 통합합니다. 이 시스템은 고급 객체 분할 기술을 사용하여 각 Gaussian splat에 고유한 식별자를 할당합니다.

- **Technical Details**: Go-SLAM은 3D Gaussian Splatting을 기반으로 하며, 객체 감지 및 세분화 모델을 포함하여 고급 컴퓨터 비전 기술을 활용하여 환경을 재구성합니다. 또한, 자연어 처리 기술을 활용하여 사용자 또는 고급 계획 알고리즘이 객체를 유연하게 쿼리할 수 있도록 합니다.

- **Performance Highlights**: 종합적인 평가 결과, 다양한 장면 설정에서 정밀도와 recall, IoU가 각각 17%, 27%, 35% 개선되는 것을 보여주며, Go-SLAM의 효율성을 입증합니다.



### Generative Object Insertion in Gaussian Splatting with a Multi-View Diffusion Mod (https://arxiv.org/abs/2409.16938)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 Gaussian Splatting으로 표현된 3D 콘텐츠에 새로운 객체를 삽입하는 혁신적인 방법을 제안합니다. 이 방법은 MVInpainter라는 다중 뷰 확산 모델(multi-view diffusion model)을 기반으로 하여, 사전 학습된 안정적인 비디오 확산 모델을 활용하여 보기 일관성(view-consistent)을 보장하는 객체 인핑팅을 제공합니다.

- **Technical Details**: MVInpainter의 핵심은 ControlNet 기반의 조건부 주입 모듈을 통합하여 보다 통제되고 예측 가능한 다중 뷰 생성을 가능하게 하는 것입니다. 이 모델은 원본 3D 씬과 대조 모델에서 배경, BBox(Bounding Box) 수준의 마스크 및 깊이 맵을 추출하여, 입력으로 세 가지 세트를 사용해 목표 객체 설명에 맞춰 인핑팅 결과를 생성합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존 방법들보다 뛰어난 성과를 보임을 입증하였습니다. 우리의 접근 방식은 다양한 결과를 생성하고, 보기 일관성을 보장하며, 더 나은 객체 품질을 제공합니다.



### Semi-Supervised Cognitive State Classification from Speech with Multi-View Pseudo-Labeling (https://arxiv.org/abs/2409.16937)
- **What's New**: 본 연구는 인지 상태 분류와 같은 주관적 평가가 많이 필요한 음성 분류 작업에서 레이블이 없는 데이터의 문제를 해결하기 위해 새로운 반지도 학습(Semi-Supervised Learning, SSL) 프레임워크를 제안합니다. 특히, 이 프레임워크는 음향과 언어적 특성을 활용한 다중 뷰(pseudo-labeling) 방법을 도입하여 분류 모델 훈련에 가장 확신이 높은 데이터를 선택합니다.

- **Technical Details**: 제안된 SSL 프레임워크는 두 가지 경로로 구성됩니다: 1) 음향 경로, 해당 경로에서는 다양한 오디오 임베딩을 이용해 레이블이 있는 데이터와 레이블이 없는 데이터를 비교하여 유사성을 판단합니다. 2) 언어 경로, 여기서는 대형 언어 모델(Large Language Models, LLMs)을 사용하여 ASR(Automatic Speech Recognition) 전사로부터 예측 레이블을 도출합니다. 프레셰 오디오 거리(Frechet Audio Distance, FAD)를 사용해 레이블이 없는 데이터의 음향 유사성을 측정하고, 이를 통해 고신뢰 데이터와 저신뢰 데이터를 구분합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 라벨이 있는 데이터의 30%만 사용하여도 완전 감독 학습(fully supervised learning)과 유사한 성능을 달성하였으며, 두 개의 기준선보다 훨씬 우수한 성과를 나타냈습니다.



### Investigating OCR-Sensitive Neurons to Improve Entity Recognition in Historical Documents (https://arxiv.org/abs/2409.16934)
- **What's New**: 이 논문은 Transformer 아키텍처 내에서 OCR 민감한 뉴런의 존재를 조사하고, 역사적 문서에 대한 명명된 개체 인식(NER) 성능에 미치는 영향을 분석합니다. 깨끗한 텍스트 입력과 잡음이 있는 텍스트 입력에 대한 뉴런 활성화를 분석하여 OCR 민감한 뉴런을 식별하고 중화시킴으로써 모델 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 실험은 Llama2와 Mistral 두 개의 오픈 액세스 대형 언어 모델을 기반으로 하며, OCR 잡음이 존재하는 텍스트에 대한 뉴런의 반응을 측정하여 OCR 민감한 레이어와 뉴런을 식별하는 데 중점을 둡니다. 이 과정에서 데이터셋은 프랑스 역사 신문의 OCR 수정 버전을 기반으로 생성되며, 다양한 수준의 OCR 잡음이 추가된 토큰을 사용합니다.

- **Performance Highlights**: 실험 결과, 역사 신문 및 고전 해설 문서에서 NER 성능이 개선되는 것으로 나타났고, 이는 특정 뉴런 조절이 잡음이 있는 텍스트에서 모델의 성능을 향상시킬 수 있음을 시사합니다.



### Cross-lingual Speech Emotion Recognition: Humans vs. Self-Supervised Models (https://arxiv.org/abs/2409.16920)
- **What's New**: 본 연구는 Self-Supervised Learning (SSL) 모델을 통한 Speech Emotion Recognition (SER)에서 인간 성능과의 비교 분석을 수행했습니다. 특히, 단일 언어(monolingual), 교차 언어(cross-lingual) 및 전이 학습(transfer learning) 맥락에서 매개변수 효율적인 미세 조정(Parameter-Efficient Fine-Tuning) 전략을 탐구하고, 방언이 교차 언어 SER에 미치는 영향을 조사하였습니다.

- **Technical Details**: 이 연구는 Wav2vec 2.0 (W2V2) 및 WavLM 등 강력한 사전 훈련(pre-trained) 모델을 사용하여 SSL 기반의 SER 성능을 비교하였습니다. 다양한 PEFT 전략을 통해 모델의 초기 파라미터를 수정하며, 중간층의 음성 표현이 높은 성능을 발휘하는 것을 확인했습니다. 교차 언어 SER에서 모델은 적절한 지식 전이를 통해 목표 언어에 적응할 수 있음을 보여주었으며, 특정 방언의 특성을 고려한 평가를 수행했습니다.

- **Performance Highlights**: 모델과 인간 모두 다른 감정에 대해 뚜렷한 행동 차이를 보였으며, 모델은 네이티브(Native) 화자와 유사한 성능을 기록했습니다. 방언은 인간의 감정 인식에 상당한 영향을 미치는 것으로 나타났으며, 감정 인식 정확도를 조정하기 위한 다양한 PEFT 전략이 효과적이었음을 입증했습니다.



### Enhancing Temporal Sensitivity and Reasoning for Time-Sensitive Question Answering (https://arxiv.org/abs/2409.16909)
Comments:
          Accepted by EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Time-Sensitive Question Answering (TSQA) 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Temporal Information-Aware Embedding과 Granular Contrastive Reinforcement Learning을 통해 모델의 시간 인식 및 추론 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 방법론을 포함합니다: Temporal Information-Aware Embedding은 모델의 시간 정보에 대한 민감성을 높이고, Granular Contrastive Reinforcement Learning은 시간적 거리에 따라 원거리 및 근접한 부정적 답변을 제공하여 모델의 시간적 추론 능력을 향상시킵니다.

- **Performance Highlights**: 우리는 제안된 프레임워크가 기존의 LLM보다 TSQA 작업에서 유의미하게 뛰어난 성능을 보여주는 것을 확인했습니다. 실험 결과는 네 개의 다양한 TSQA 데이터셋에서 우리의 프레임워크가 기존 모델보다 크게 향상된 성능을 보였음을 입증합니다.



### Discriminative Anchor Learning for Efficient Multi-view Clustering (https://arxiv.org/abs/2409.16904)
Comments:
          This work has been accepted by TMM

- **What's New**: 이 논문에서는 Multi-view clustering을 위한 차별적 앵커 학습(discriminative anchor learning)을 제안하여 기존 방법의 단점을 해결하고자 했다. 이 방법은 각 뷰(view)에 대한 차별적(feature) 표현을 학습하고, 이를 통해 공유 앵커 그래프(shared anchor graph)의 품질을 향상시키며, 앵커 간의 보완적(complementary) 정보도 고려한다.

- **Technical Details**: 차별적 앵커 학습(discriminative anchor learning for multi-view clustering, DALMC) 방식은 원본 데이터셋을 기준으로 각 뷰에 대해 차별적 feature representation을 학습하고, 이를 바탕으로 합동 앵커 그래프(consensus anchor graph)를 구축한다. 이 과정은 차별적 feature 학습과 앵커 그래프 구축을 통합하여 서로 개선될 수 있도록 한다. 또한 최적 앵커(anchors)와 합동 앵커 그래프는 직교 제약조건(orthogonal constraints)을 통해 학습된다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, DALMC는 기존 방법들에 비해 효율성과 효과성을 입증했다. 이 알고리즘은 선형 시간 복잡도를 가지며, 대규모 데이터셋을 효율적으로 처리할 수 있는 능력을 보여준다.



### Towards Underwater Camouflaged Object Tracking: An Experimental Evaluation of SAM and SAM 2 (https://arxiv.org/abs/2409.16902)
Comments:
          Preprint. Work in Progress

- **What's New**: 이번 논문에서는 UW-COT라는 첫 번째 대규모 수중 위장 물체 추적 데이터셋을 제안하고, 이 데이터셋을 기반으로 여러 최신 시각 물체 추적 방법의 실험적 평가를 수행하였습니다.

- **Technical Details**: UW-COT 데이터셋은 96개 카테고리로 구성된 220개의 수중 비디오 시퀀스를 포함하며, 각 프레임에 대해 위장된 물체에 대한 바운딩 박스 주석을 제공합니다. 이 데이터셋을 통해 SAM(Segmentation Anything Model)과 SAM 2의 성능을 비교하였으며, SAM 2는 시간적 일관성, 신뢰성, 기능 임베딩, 컴퓨팅 효율성 및 신규 도메인 일반화 능력이 개선되었습니다.

- **Performance Highlights**: SAM 2는 UW-COT 데이터셋에서 SAM 기반 추적기(SAM-DA 및 Tracking Anything)보다 뛰어난 성능을 보였으며, 현재의 최신 VOT 방법들보다 우수한 성능을 기록하였습니다. 이는 SAM 2가 비디오 데이터의 동적 도전 과제를 해결하기 위한 향상된 솔루션을 제공한다는 것을 보여줍니다.



### A Roadmap for Embodied and Social Grounding in LLMs (https://arxiv.org/abs/2409.16900)
Comments:
          Accepted Version of a conference paper presented at Robophilosophy Conference 2024

- **What's New**: 이번 연구는 로봇 시스템에 대규모 언어 모델(LLMs)을 통합하여 의사소통 뿐만 아니라 다중 모달 입력 처리, 고급 추론 및 계획 생성을 통한 변혁적인 패러다임을 제시합니다. LLM의 지식을 경험적 세계와 연결하기 위한 새로운 접근 방식을 탐구합니다.

- **Technical Details**: 이 논문은 LLMs의 기초 모델에 대한 추상 지식을 물리적 현실과 일치시키는 'grounding'(근거화) 과정을 다룹니다. 이 과정에서는 LLMs가 환경에 대한 이해도를 높이고, 객체의 물리적 속성을 통해 추론 능력을 향상시키도록 돕는 기술적 접근이 포함됩니다.

- **Performance Highlights**: 최근의 연구들은 LLMs의 텍스트 기반 작업에서의 효과가 신체적 제어까지 확장되었음을 보여주며, 이는 로봇이 인간과의 협력적 환경에서 더 뛰어난 성능을 발휘할 수 있게 합니다. 그러나 실제 로봇 작업에서 물리적 및 사회적 추 reasoning과 같은 기술들에 대한 LLM의 한계 또한 강조되었습니다.



### Revisiting Space Mission Planning: A Reinforcement Learning-Guided Approach for Multi-Debris Rendezvous (https://arxiv.org/abs/2409.16882)
Comments:
          Accepted for publication at the 2024 International Conference on Space Robotics (iSpaRo)

- **What's New**: 이 연구는 우주 쓰레기 탐방의 효율적인 순서를 결정하기 위해 심층 강화 학습(deep reinforcement learning) 분야의 masked Proximal Policy Optimization(PPO) 알고리즘을 도입합니다. 목표는 주어진 모든 쓰레기를 방문하는 최적의 순서를 찾고, 이를 통해 전체 미션의 총 소요 시간을 최소화 하는 것입니다.

- **Technical Details**: 이 연구에서는 신경망(neural network) 정책을 개발하고, 다양한 쓰레기 환경에서 시뮬레이션된 우주 미션을 통해 훈련합니다. 훈련 후에 신경망은 Izzo의 Lambert 기법을 사용해 최적의 경로를 계산하며, 강화 학습 접근법을 통해 계획 효율성이 크게 향상되었습니다. 기존의 유전 알고리즘(Genetic algorithm)과 탐욕 알고리즘(Greedy algorithm)과 비교하여 각각 약 10.96%와 13.66%의 미션 시간을 줄이는 데 성공했습니다.

- **Performance Highlights**: 이 모델은 다양한 시뮬레이션 시나리오에서 쓰레기 방문의 가장 시간 효율적인 순서를 식별하며, 가장 빠른 계산 속도로 결과를 도출합니다. 이는 우주 쓰레기 제거를 위한 미래의 미션 계획 전략을 향상시키는데 기여할 것으로 기대됩니다.



### The Role of Language Models in Modern Healthcare: A Comprehensive Review (https://arxiv.org/abs/2409.16860)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 의료 분야에 어떻게 적용되는지에 대한 체계적인 리뷰를 제공합니다. LLM의 발전 과정과 의료 적용에서의 강점뿐만 아니라 데이터 프라이버시, 편향, 윤리적 고려사항 등의 문제를 논의합니다.

- **Technical Details**: 대형 언어 모델(LLM)은 Transformer 아키텍처를 기반으로 하여 장거리 의존성을 효과적으로 캡처하는 능력을 가지고 있습니다. 모델은 일반적으로 방대한 텍스트 데이터셋으로 사전 학습(pre-training)된 후, 특정 작업에 맞춰 세부 조정(fine-tuning)됩니다. BioBERT, ClinicalBERT와 같은 의료 전용 모델이 개발되어 임상 언어의 독특한 도전을 해결하고 있습니다.

- **Performance Highlights**: LLM은 의료 데이터 분석, 질병 진단, 환자 관리 및 약물 발견과 같은 다양한 분야에서 사용되고 있습니다. 임상 의사결정 지원 및 의료 문서 요약 등의 임무에 대한 효과적인 증상이 입증되었습니다. 측정 기준으로는 MMLU, HumanEval과 같은 벤치마크가 사용되어 모델의 효과성을 평가합니다.



### OffRIPP: Offline RL-based Informative Path Planning (https://arxiv.org/abs/2409.16830)
Comments:
          7 pages, 6 figures, submitted to ICRA 2025

- **What's New**: 이번 연구에서는 Offline RL 기반의 정보 경로 계획 (Informative Path Planning, IPP) 프레임워크인 OffRIPP를 제안합니다. 이는 실시간 상호작용 없이 정보 수익을 극대화하는 경로를 계획합니다.

- **Technical Details**: OffRIPP 프레임워크는 배치 제약 강화 학습 (Batch-Constrained Reinforcement Learning)을 활용하여 미리 수집된 데이터셋에서 학습하며, 전통적인 방법들보다 안전하고 비용 효율적으로 작동합니다. 이 접근법은 다양한 알고리즘에 의해 생성된 데이터셋으로부터 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과, OffRIPP는 기존의 여러 IPP 알고리즘, 온라인 RL 기반 훈련 방식 및 동작 복제 (Behavior Cloning)와 비교할 때 성능이 우수한 결과를 보였으며, 운영 비용을 줄이는 데 도움을 주었습니다.



### On the role of Artificial Intelligence methods in modern force-controlled manufacturing robotic tasks (https://arxiv.org/abs/2409.16828)
Comments:
          To be published in Proceedings of the 20th International Conference on Informatics in Control, Automation and Robotics (ICINCO)

- **What's New**: 이번 논문은 인공지능(AI)과 포스 제어(force-controlled) 로봇 작업의 통합에 대해 탐구하며, 이는 첨단 제조(advanced manufacturing)의 중요한 요소이자 산업 4.0의 기초입니다. AI가 로봇 매니퓰레이터에 미치는 영향과 이를 통한 스마트 제조의 혁신을 다룹니다.

- **Technical Details**: 포스 제어는 다양한 산업 및 의료 응용 분야에서 필수적인 요소로 여겨지며, 이 논문에서는 AI 기반 포스 제어 기술의 최신 동향과 이를 통한 공정 혁신을 설명합니다. 포스 제어에 대한 기존 기법을 기반으로, AI는 환경의 비선형 특성과 복잡성을 극복하는데 도움을 줄 수 있습니다.

- **Performance Highlights**: 포스 제어 작업의 예로는 디버링(deburring), 폴리싱(polishing), 조립(assembly) 작업이 있으며, 특히 peg-in-hole (PiH) 조립 작업이 강조됩니다. 논문은 AI 기반 기법의 최신 동향과 이들이 현장에서의 적용성 및 성능을 어떻게 개선할 수 있는지를 탐구합니다.



### Learning phase-space flows using time-discrete implicit Runge-Kutta PINNs (https://arxiv.org/abs/2409.16826)
Comments:
          10 pages, 4 figures, published in the International Conference on Scientific Computing and Machine Learning, see this http URL

- **What's New**: 본 연구에서는 고차원 함수형 비선형 결합 미분 방정식의 위상 공간 솔루션을 얻기 위한 계산 프레임워크를 제시합니다. 이를 위해 High-order Implicit Runge-Kutta Physics-Informed Neural Networks (IRK-PINNs) 방식을 사용하였습니다.

- **Technical Details**: 기존의 미분 방정식 해법을 바탕으로 하여, 좌표가 함수로 처리되는 맥락으로의 스킴을 수정하였습니다. 이 수정은 외부 필드에서 입자의 운동 방정식을 효율적으로 해결할 수 있도록 해줍니다.

- **Performance Highlights**: 우리의 접근 방식을 활용하여, 중심 힘 필드에 위치한 질량 입자와 주기적인 전기 필드에서의 하전 입자의 운동 방정식을 성공적으로 해결하였습니다.



### Uncertainty Representations in State-Space Layers for Deep Reinforcement Learning under Partial Observability (https://arxiv.org/abs/2409.16824)
- **What's New**: 이 논문은 기대되는 수익을 극대화하기 위해 모델-프리 아키텍처 내에서 훈련된 스탠드얼론 Kalman 필터 레이어를 제안합니다. 이 레이어는 숨겨진 상태 표현의 불확실성을 포함할 수 있는 내부 메커니즘을 갖추고 있습니다.

- **Technical Details**: Kalman 필터 레이어는 선형 상태 공간 모델에서 닫힌 형태의 Gaussian 추론을 수행하며, 시퀀스 길이에 대해 로그적으로 확장 가능합니다. 이 레이어는 표준 모델-프리 아키텍처의 다른 순환 레이어와 쉽게 교체 가능하며, 잠재 상태 표현의 확률적 필터링을 위한 명시적 메커니즘을 포함합니다.

- **Performance Highlights**: 다양한 POMDP 문제에서 실시된 실험 결과, Kalman 필터 레이어는 의사결정에서 불확실성 추론이 중요한 문제에서 다른 상태 기반 모델보다 우수한 성능을 발휘하였습니다.



### XAI-guided Insulator Anomaly Detection for Imbalanced Datasets (https://arxiv.org/abs/2409.16821)
Comments:
          Accepted as a workshop paper at ECCV 2024

- **What's New**: 이 연구에서는 전선의 절연체 결함을 탐지하기 위한 새로운 파이프라인을 제안합니다. UAV(무인 항공기)를 활용하여 수집한 이미지를 통해 절연체 결함을 정확하게 검출하고 분류하는 방법론을 개발하였습니다.

- **Technical Details**: 제안된 방법은 YOLOv8 모델을 사용하여 절연체를 검출하고, 비율이 불균형한 데이터셋 문제를 해결하기 위해 로지스틱 회귀를 통한 재훈련 기법을 적용하였습니다. 또한, LRP(층별 관련 전파기법)를 활용하여 결함의 위치를 정확히 설명하고 시각화했습니다.

- **Performance Highlights**: 결함 탐지 정확도를 최대 13% 향상시켰으며, 클래스 불균형 문제를 해결하여 예측 유지보수의 효과성을 크게 개선하였습니다. 이 연구는 산업 현장에서의 비전 기반 검사 및 예측 유지 보수에 가치 있는 기여를 하고 있습니다.



### Scalable Ensemble Diversification for OOD Generalization and Detection (https://arxiv.org/abs/2409.16797)
Comments:
          Under review

- **What's New**: 본 연구는 Scalable Ensemble Diversification (SED)라는 새로운 방법론을 제시하여 기존의 다양한 앙상블 학습 방법의 한계를 극복합니다. 특히, OOD 샘플 없이도 대규모 데이터에서 효과적으로 적용할 수 있는 방식으로 설계되었습니다.

- **Technical Details**: SED는 세 가지 주요 기술적 혁신을 통해 발전되었습니다: (1) 하드 샘플을 동적으로 식별하여 모델 간의 불일치를 유도합니다. (2) 각 반복에서 무작위로 선택된 두 모델에 대해서만 다양화 목표를 적용하여 계산 비용을 줄입니다. (3) 네트워크의 출력 근처의 일부 레이어에만 영향을 미치도록 하여 심층 네트워크에서의 다양화 목표를 조정합니다.

- **Performance Highlights**: ImageNet에서의 실험을 통해 SED의 다양한 이점을 확인했습니다. OOD 일반화와 OOD 탐지 모두에서 성능이 상당히 향상되었으며, Predictive Diversity Score (PDS) 방법론은 OOD 샘플 탐지에서 기존 방법들을 초월하는 성능을 보였습니다.



### Symbolic State Partition for Reinforcement Learning (https://arxiv.org/abs/2409.16791)
- **What's New**: 본 논문은 연속 상태 공간에서 직접적으로 작동할 수 없는 테이블 기반 강화 학습(tabular reinforcement learning) 방법의 문제를 다룹니다. 이 문제의 해결책 중 하나는 상태 공간을 분할(partition)하는 것입니다. 이 연구에서는 환경 동역학(dynamics)을 기반으로 상징적 실행(symbolic execution)을 통해 분할을 추출하는 방법을 제안합니다.

- **Technical Details**: 상징적 분할(symbolic partitioning)은 비선형 관계(nonlinear relations)가 있는 상태 구성 요소 간의 근사화가 학습 과정에서 해롭다는 점을 충분히 고려합니다. 이상적인 분할은 가능한 한 거칠게(coarse) 하면서도 주어진 문제의 상태 공간의 핵심 구조(key structure)를 포착할 수 있어야 합니다.

- **Performance Highlights**: 상징적 분할을 통해 강화 학습이 드문 보상(sparse rewards) 상황에서도 성능이 향상된다는 것을 보였습니다. 논문에서는 정밀도(precision), 확장성(scalability), 학습 에이전트 성능(performance), 학습된 정책의 상태 공간 커버리지(state space coverage) 측면에서 평가를 수행하였습니다.



### Enhancing Feature Selection and Interpretability in AI Regression Tasks Through Feature Attribution (https://arxiv.org/abs/2409.16787)
- **What's New**: 이 연구는 Explainable Artificial Intelligence (XAI) 분야의 feature attribution 방법을 활용하여 회귀 문제에서 입력 데이터의 비정보적 특징을 필터링하고, 예측의 정확도 및 안정성을 높이는 방법을 제안합니다.

- **Technical Details**: 연구자들은 Integrated Gradients (IG)와 k-means 클러스터링을 결합한 feature selection pipeline을 도입했습니다. 이 방법은 실제 산업 문제인 터보 기계 개발 과정에서 블레이드 진동 분석에 적용되었습니다. IG는 gradient 기반의 모델 독립적 접근법으로, 이전에 회귀 문제에서 사용된 바 있습니다.

- **Performance Highlights**: 이 접근법은 생성된 투명한 더미 데이터에서 실험을 통해 효과를 검증한 후, 실제 문제에 적용되었습니다. 제안된 방법은 established baseline feature selection 방법 및 KernelShap과 비교되었습니다.



### Holistic Automated Red Teaming for Large Language Models through Top-Down Test Case Generation and Multi-turn Interaction (https://arxiv.org/abs/2409.16783)
Comments:
          EMNLP 2024 camera ready version

- **What's New**: 본 논문에서는 HARM (Holistic Automated Red teaMing)을 제안하여, 대형 언어 모델 (LLMs)의 비정상적인 행동을 체계적으로 식별하는 새로운 방법을 소개합니다. HARM은 리스크 카테고리의 세분화된 분류를 바탕으로 한 상향식 접근 방식을 사용하고, 멀티 턴 상호작용을 지원하여 자동화된 레드 팀핑을 강화합니다.

- **Technical Details**: HARM은 세분화된 리스크 분류 체계를 사용하여 다양한 테스트 케이스를 생성하는 방법을 적용합니다. 이 방법은 고유의 파인 튜닝 전략과 강화 학습 기법을 활용하여 인간과 유사한 방식으로 멀티 턴에서의 적대적인 탐색을 수행합니다.

- **Performance Highlights**: 실험 결과를 통해 HARM이 모델의 취약성에 대한 보다 체계적인 이해를 가능하게 하고, 안전한 정렬 과정을 위한 보다 구체적인 가이드를 제공함을 보여주었습니다.



### Super Level Sets and Exponential Decay: A Synergistic Approach to Stable Neural Network Training (https://arxiv.org/abs/2409.16769)
- **What's New**: 본 논문에서는 신경망 최적화 프로세스를 향상시키기 위해 동적인 학습률 알고리즘을 개발하였습니다. 이 알고리즘은 지수적 감소(exponential decay)와 고급 과적합 방지 전략을 통합하여 최적화의 안정성을 높입니다. 또한, 이론적 프레임워크를 수립하여 Lyapunov 안정성 원리를 통해 손실 함수의 초레벨 집합(superlevel sets)의 연결성을 보장합니다.

- **Technical Details**: 이 논문은 동적 학습률(dynamic learning rate)과 손실 함수의 초레벨 집합(superlevel set) 간의 수학적 관계를 탐구합니다. 지수적 감소에 기반한 학습률 ‌η(t)=η₀e^{-αt} 모델을 통해 최적화 경로의 연결성을 유지하며, 불안정한 지역에 갇히지 않도록 합니다. 이를 통해 안정적이고 효율적인 최적화를 달성하는 방법을 제시합니다.

- **Performance Highlights**: 이 연구는 신경망 훈련 과정에서 에러를 최소화하면서 과적합(overfitting)을 방지하는 데 효과적입니다. 제안된 알고리즘은 더 나은 일반화 능력을 제공하며, 다양한 데이터 환경에서 일관된 안정성을 유지하여 신뢰성을 높입니다.



### MaViLS, a Benchmark Dataset for Video-to-Slide Alignment, Assessing Baseline Accuracy with a Multimodal Alignment Algorithm Leveraging Speech, OCR, and Visual Features (https://arxiv.org/abs/2409.16765)
- **What's New**: 이 논문에서는 강의 비디오와 해당 슬라이드를 정렬하는 벤치마크 데이터셋을 제시하고, 음성, 텍스트 및 이미지에서 특징을 활용하는 새로운 다중 모달 알고리즘을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 동적 프로그래밍(dynamic programming)을 사용하여 최적의 슬라이드 시퀀스를 결정하며, OCR(Optical Character Recognition)을 통해 얻은 특징들이 매칭 정확도에 크게 기여한다고 보고합니다. 알고리즘은 SIFT(Scale-Invariant Feature Transform)에 비해 평균 0.82의 정확도를 기록하면서 속도는 약 11배 빨라졌습니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 강의 스타일과 비디오 품질에 따른 매칭 정확도의 차이를 보였으며, 슬라이드 전환을 제재할 경우 정확도가 향상되었습니다. 또한, 매칭의 정확도는 오디오 전사에 의해서도 유용한 정보를 제공하고, OCR 데이터가 부족할 때 더욱 유용함을 강조합니다.



### Offline and Distributional Reinforcement Learning for Radio Resource Managemen (https://arxiv.org/abs/2409.16764)
- **What's New**: 이 논문은 기존의 온라인 강화 학습(online RL) 방식에서 오프라인 및 분포적 강화 학습(off-line and distributional RL)을 적용하여 무선 네트워크의 무선 자원 관리(radio resource management, RRM) 문제에 대한 새로운 해결책을 제시합니다.

- **Technical Details**: 제안된 알고리즘은 고정된 데이터셋을 사용하여 환경과의 상호작용 없이 오프라인 학습을 수행하며, 반환(return)의 분포를 고려하여 불확실성을 처리합니다. RRM 문제에 대한 이 접근 방식은 평균 성과만을 고려하는 기존 방법들의 한계를 극복합니다.

- **Performance Highlights**: 모의실험 결과, 제안된 오프라인 및 분포적 RL 알고리즘이 기존의 자원 관리 모델보다 우수한 성능을 보였으며, 온라인 RL보다 16% 높은 성과를 달성했습니다.



### GB-RVFL: Fusion of Randomized Neural Network and Granular Ball Computing (https://arxiv.org/abs/2409.16735)
- **What's New**: 본 논문에서는 랜덤 벡터 기능 링크(RVFL) 네트워크의 문제점을 해결하기 위해, 입력으로 개별 샘플 대신 granular balls (GBs)을 사용하는 GB-RVFL 모델과, GBs의 지리적 구조를 보존하기 위한 graph embedding (GE) 통합 모델인 GE-GB-RVFL을 제안합니다.

- **Technical Details**: GB-RVFL 모델은 GB의 중심 행렬의 역행렬만 필요로하여 확장성을 높이고, GB의 조밀성을 활용하여 노이즈와 이상치에 대한 강인성을 개선합니다. GE-GB-RVFL 모델은 데이터셋의 내재적 기하구조를 보존하면서 GB-RVFL의 핵심 특성을 유지하며, 그래프 정규화 항을 포함하는 방법론을 통합하여 데이터 구조의 세부정보를 보존합니다.

- **Performance Highlights**: GB-RVFL 및 GE-GB-RVFL 모델은 KEEL, UCI, NDC 및 생물의학 데이터셋에서 평가되었으며, 기존의 기준 모델에 비해 우수한 성능을 입증하였습니다. 특히, 실제 생물의학 데이터셋에서도 적용 가능성을 보여주며, 유방암 분류 및 알츠하이머병 분류에서의 향상된 성능을 나타냈습니다.



### Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification (https://arxiv.org/abs/2409.16718)
Comments:
          EMNLP 2024 Main Conference

- **What's New**: 최근 Vision-Language Models (VLMs)에 대한 세밀한 조정이 이루어지면서, 클립 모델의 고유한 매개변수를 조정하는 것의 중요성을 재조명하였다. 이 연구에서는 모든 매개변수를 조정하는 대신 특정 매개변수만을 조정하는 CLIPFit 방법을 제안하였다.

- **Technical Details**: CLIPFit은 기존의 프롬프트 튜닝(prompt tuning) 및 어댑터 튜닝(adapter tuning) 방식과는 다르게, 추가적인 매개변수를 도입하지 않고 클립 모델의 특정 바이어스와 정규화 레이어만 조정하는 방법이다. 이로 인해 파라미터 수가 줄어들고, 성능이 향상된다.

- **Performance Highlights**: CLIPFit을 사용하여 zero-shot CLIP 대비 7.33%의 평균 조화 평균 정확도(harmonic mean accuracy) 개선을 달성하였으며, 이는 16-shot 설정에서 프롬프트 튜닝 및 어댑터 튜닝을 대체할 수 있는 유망한 옵션이다.



### Pix2Next: Leveraging Vision Foundation Models for RGB to NIR Image Translation (https://arxiv.org/abs/2409.16706)
Comments:
          19 pages,12 figures

- **What's New**: Pix2Next는 RGB 이미지를 기반으로 고해상도 NIR 이미지를 생성하는 혁신적인 이미지-이미지 변환 프레임워크입니다. 이 방법은 최신 Vision Foundation Model (VFM)을 활용하여_encoder-decoder_ 아키텍처에서 크로스-어텐션 메커니즘(cross-attention mechanism)을 통합하여 특징 통합을 향상시킵니다. 더불어, 여러 해상도에서 현실적인 이미지 생성을 보장하는 PatchGAN 판별자를 사용하여 NIR 이미지 생성의 품질과 세부사항을 개선합니다.

- **Technical Details**: Pix2Next은 RGB 이미지를 NIR 이미지로 변환하는 과정에서 고유한 세부 사항과 스펙트럼 특성을 유지하는 데 중점을 두고 설계되었습니다. 실제로, Segmentation과 Object Detection 태스크에 대한 성능 평가와 함께 다양한 손실 함수가 글로벌 컨텍스트 이해와 로컬 특징 보존을 연결하여 모델 성능을 높입니다. 또한	RANUS 데이터셋을 사용하여 테스트를 진행하였습니다.

- **Performance Highlights**: Pix2Next는 FID(Frechet Inception Distance) 점수를 기존 방법에 비해 34.81% 향상시켜, 세 가지 시각 품질 지표에서 뛰어난 성능을 보였습니다. 또한, NIR 이미지로의 변환된 데이터를 이용하여 자율 주행 인식 태스크에서 더욱 개선된 성능을 보여주어, 제한된 실 NIR 데이터셋을 보완하는 데 있어 효용성을 입증했습니다.



### Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Mod (https://arxiv.org/abs/2409.16689)
Comments:
          Accepted by ECCV2024, Project Page: this https URL

- **What's New**: 이 논문은 기존의 Discrete Diffusion Models (DDMs)에서 발생하는 레이아웃 고착(Layout Sticking) 현상을 해결하기 위해 Layout-Corrector라는 새롭고 간단한 모듈을 제안합니다.

- **Technical Details**: Layout-Corrector는 레이아웃의 각 요소에 대한 정확성 점수를 평가하고, 저조한 점수를 가진 요소를 재초기화하여 하모니가 있는 레이아웃을 생성을 돕습니다. 이 모듈은 DDM과 함께 사용되며, 각 생성 과정에서 하모니를 고려하여 불일치하는 요소를 식별합니다.

- **Performance Highlights**: Layout-Corrector는 다양한 기준 벤치마크에서 테스트되어 DDM과 함께 사용할 경우 레이아웃 생성 성능을 일관되게 향상시키고, 정확성-다양성 무역의 조절을 통한 성능 저하를 완화합니다.



### Erase then Rectify: A Training-Free Parameter Editing Approach for Cost-Effective Graph Unlearning (https://arxiv.org/abs/2409.16684)
Comments:
          Under review

- **What's New**: 이번 연구에서는 Erase then Rectify (ETR)라는 두 단계의 훈련 없는 접근 방식을 제안하여 효율적이고 확장 가능한 그래프 비학습 (graph unlearning)을 실현했습니다. 이 방법은 특정 노드나 엣지의 영향을 제거하는 데 중점을 두었으며, 기존의 방법들이 필요한 추가 훈련을 하지 않고도 높은 유용성을 유지합니다.

- **Technical Details**: ETR 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계인 Erase에서는 중요한 매개변수를 마스킹하여 비학습 샘플의 영향을 효과적으로 제거합니다. 두 번째 단계인 Rectify에서는 남은 데이터 세트에 대한 모델의 그래디언트를 추정하여 GNN의 성능을 향상시키는 기법을 적용합니다. ETR은 전체 훈련 데이터에 접근하지 않고도 그래프 비학습을 수행할 수 있게 설계되었습니다.

- **Performance Highlights**: ETR은 평균적으로 4583.9배 적은 시간과 4.2배 적은 메모리 사용량을 기록하며 비학습 효율성, 성능, 그리고 유용성 측면에서도 뛰어난 결과를 보였습니다. 이는 대규모 그래프에도 적용가능한 가능성을 제시합니다.



### TSBP: Improving Object Detection in Histology Images via Test-time Self-guided Bounding-box Propagation (https://arxiv.org/abs/2409.16678)
Comments:
          MICCAI 2024

- **What's New**: 본 논문에서는 Test-time Self-guided Bounding-box Propagation (TSBP) 방법을 제안하여, 물체 검출 성능을 크게 향상시키는 새로운 접근 방식을 소개합니다. 이 방법은 고신뢰도 바운딩 박스의 정보를 활용하여 저신뢰도 바운딩 박스를 조정합니다.

- **Technical Details**: TSBP는 Earth Mover's Distance (EMD)를 활용하여 시각적 유사성을 바탕으로 바운딩 박스 간의 정보를 전파합니다. 이 과정은 신뢰도가 낮은 바운딩 박스의 클래스 레이블을 보정하며, 별도의 라벨링된 샘플이 요구되지 않아 기존의 불확실성 보정 방법과 차별화됩니다.

- **Performance Highlights**: 실험 결과, TSBP는 기존의 불확실성 보정 방법에 비해 더욱 견고하고 정확한 물체 검출 결과를 제공합니다. 특히 상태-of-the-art 딥러닝 기반 검출 네트워크와 함께 사용했을 때 성능이 크게 향상되었습니다.



### GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning (https://arxiv.org/abs/2409.16670)
Comments:
          Under review

- **What's New**: 본 논문에서는 다양한 그래프 도메인으로 잘 훈련된 GNN을 전이하기 위한 효과적이고 파라미터 효율적인 방법인 GraphLoRA를 제안합니다. GraphLoRA는 Low-Rank Adaptation (LoRA)의 성공으로부터 영감을 받아 구조 인식 최대 평균 불일치(SMMD)를 도입하여 출처 및 대상 그래프 간의 노드 피쳐 분포의 차이를 줄이는 방법입니다.

- **Technical Details**: GraphLoRA는 구조 인식 최대 평균 불일치(SMMD)를 도입하여 출처와 대상 그래프 간 노드 피쳐 분포의 차이를 최소화하고, 저차원 적응(low-rank adaptation) 방식을 통해 훈련된 GNN과 함께 소규모 훈련 가능 GNN을 삽입하여 구조적 분포의 격차를 메우는 방식으로 구축됩니다. 또한, 구조 인식 정규화 목표를 제안하여 레이블 수가 적은 대상 그래프에 대해 프리트레인 GNN의 적응성을 향상시킵니다.

- **Performance Highlights**: 여섯 개의 실제 데이터세트에 대한 광범위한 실험 결과 GraphLoRA는 20%의 파라미터만 조정하여도 열한 개의 기준선에 비해 뛰어난 성능을 달성하였음을 보여줍니다.



### Progressive Representation Learning for Real-Time UAV Tracking (https://arxiv.org/abs/2409.16652)
Comments:
          Accepted by the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 이 논문에서는 UAV(무인 항공기) 추적을 위한 새로운 점진적 표현 학습 프레임워크인 PRL-Track을 제안합니다. PRL-Track은 거친 표현 학습과 정밀 표현 학습의 두 부분으로 나뉘며, 특히 복잡한 동적 환경에서의 객체 추적 성능을 개선하는 데 중점을 둡니다.

- **Technical Details**: PRL-Track은 CNN(합성곱 신경망)을 기반으로 하는 거친 표현 학습과 ViT(비전 트랜스포머)를 기반으로 하는 정밀 표현 학습을 통합합니다. 거친 표현 학습에서는 외관 정보와 의미론적 정보를 이용한 두 개의 혁신적인 조절기(regulator)를 사용하는데, 이는 외관 간섭을 완화하고 깊은 특징에서 의미 정보를 캡처합니다. 정밀 표현 학습에서는 새로운 계층적 모델링 생성기가 도입되어 객체의 거친 표현을 연결합니다.

- **Performance Highlights**: 종합 실험에 따르면 PRL-Track은 세 개의 권위 있는 UAV 추적 벤치마크에서 우수한 성능을 보여주었습니다. 실제 테스트 결과, PRL-Track은 일반적인 UAV 플랫폼에서 초당 42.6 프레임으로 뛰어난 추적 성능을 실현하여 효율성과 강인성을 입증했습니다.



### Task Addition in Multi-Task Learning by Geometrical Alignmen (https://arxiv.org/abs/2409.16645)
Comments:
          11 pages, 5 figures, Accepted at AI for Science Workshop at 41st International Conference on Machine Learning

- **What's New**: 본 연구에서는 Geometrically Aligned Transfer Encoder (GATE) 알고리즘을 개선하기 위한 새로운 접근법인 task addition을 제안합니다. 이 방법은 한정된 데이터로도 목표 작업의 성능을 높이면서 계산 복잡성을 최소화하도록 설계되었습니다.

- **Technical Details**: GATE 알고리즘은 서로 다른 작업의 잠재 공간(latent space)의 기하학적 형태를 정렬하여 지식을 전달합니다. Task addition 접근법은 대규모 데이터 세트로 초기 지도학습(pre-training)을 통해 수행된 후, 각 목표 작업에 대해 특정 모듈을 추가하고 학습하는 방식으로 진행됩니다. 이 방식은 기존 다중 작업(multi-task) 방법보다 계산 비용 측면에서 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, task-added GATE가 단일 작업 학습(SINGLE) 및 다중 작업 학습(MTL) 방법보다 뛰어난 성능을 보여주었으며, 학습 시간 또한 MTL 모델을 처음부터 학습하는 것보다 유의미하게 빠른 것으로 나타났습니다.



### Training Language Models to Win Debates with Self-Play Improves Judge Accuracy (https://arxiv.org/abs/2409.16636)
Comments:
          48 pages, 12 figures; code at this https URL

- **What's New**: 이 연구에서는 언어 모델을 훈련시켜 토론에서 승리하도록 최적화한 결과, 평가자의 판단 정확도가 향상된다는 것을 처음으로 보였습니다. 이는 토론을 실질적인 확장 가능한 감독 방법으로 구현하고 검증하는 중요한 단계입니다.

- **Technical Details**: 연구는 QuALITY 데이터셋에서의 독해 질문에 대한 정보 비대칭 토론을 통해 진행되었습니다. 참여자 모델이 모든 주장을 두 차례 제시하고, 최종적으로 평가자가 어느 모델의 주장을 신뢰하는지를 결정합니다. 평가자의 정확성은 모델이 다른 모델과 싸울 때의 승률로 측정되었습니다.

- **Performance Highlights**: 토론 훈련 후 평가자의 정확도가 4% 증가했으며(p<10−6), 이러한 향상은 실제 감독 신호 없이도 이루어졌습니다. 반면 비대립 컨설팅 모델을 대상으로 한 실험에서는 모델 숙련도와 평가자 정확성 간의 긍정적인 관계가 발견되지 않았습니다.



### Stochastic Subsampling With Average Pooling (https://arxiv.org/abs/2409.16630)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문에서는 기존의 Dropout 방식이 갖는 일관성 결여 문제를 해결한 새로운 모듈인 stochastic average pooling을 제안합니다. 이 모듈은 pooling 과정에서 Dropout과 유사한 확률성을 통합하여 신경망의 성능을 개선할 수 있습니다.

- **Technical Details**: Stochastic average pooling은 stochastic subsampling과 average pooling을 통합한 방식입니다. 이는 기존 average pooling을 대체할 수 있으며, 코드 변경이 최소화됩니다. 이 방법은 수학적 기호와 함께 명확하게 정의되어 있습니다.

- **Performance Highlights**: 실험 결과, stochastic average pooling로 기존 평균 풀링을 대체하면 다양한 데이터셋, 작업 및 모델에서 성능이 일관되게 개선되는 것으로 나타났습니다.



### Ascend HiFloat8 Format for Deep Learning (https://arxiv.org/abs/2409.16626)
Comments:
          13 Pages, 4 Figures, 9 Tables

- **What's New**: 본 논문은 딥러닝을 위한 새로운 8비트 부동 소수점 데이터 포맷인 HiFloat8(약칭 HiF8)을 제안합니다. HiF8은 정밀도(precision)와 동적 범위(dynamic range) 사이의 균형을 개선하였으며, AI 훈련의 순전파(forward pass)와 역전파(backward pass) 모두에 동시에 사용할 수 있는 특징이 있습니다.

- **Technical Details**: HiF8은 1비트의 부호(Sign), 2-4비트의 점(Dot), 그리고 3-1비트의 지수(Exponent)와 맨티사(Mantissa) 필드를 포함합니다. 정규 값 인코딩에 대해 7개의 지수와 3비트의 맨티사, 8개의 지수와 2비트의 맨티사, 16개의 지수와 1비트의 맨티사를 제공합니다. 비정규 값 인코딩에 대해서는 7개의 추가적인 2의 거듭제곱을 포함하여 범위를 확장합니다.

- **Performance Highlights**: 다양한 신경망 및 대규모 언어 모델(LLMs)에 대한 대규모 시뮬레이션 결과가 제공되어 HiF8 포맷의 효율성을 입증하였습니다. HiF8은 기존 Float8 포맷들과 비교하여 정밀도와 동적 범위를 효과적으로 조화시키며, AI 훈련에서의 활용 가능성을 보여줍니다.



### Claim-Guided Textual Backdoor Attack for Practical Applications (https://arxiv.org/abs/2409.16618)
Comments:
          Under Review

- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP) 및 대규모 언어 모델(Large Language Models, LLMs)의 발전은 새로운 보안 취약점을 드러냈습니다. 특히, 이번 연구에서는 입력 변조 없이 내재된 텍스트 클레임(claim)을 트리거로 활용하는 새로운 Claim-Guided Backdoor Attack (CGBA)을 도입합니다.

- **Technical Details**: CGBA는 다음의 세 가지 주요 단계로 구성됩니다: 1) 트레이닝 샘플에서 클레임 추출하기, 2) 유사한 클레임을 군집화하기, 3) 특정 군집을 선택하여 트리거로 설정하고 모델 훈련 중에 백도어를 주입하여 목표 클레임에서 잘못된 결정을 유도합니다. 이 과정은 대조적 손실(contrastive losses), 클레임 거리(claim distance), 다중 작업 손실(multi-tasking losses)을 사용합니다.

- **Performance Highlights**: CGBA는 다양한 데이터셋과 모델에서 실험을 통해 이전 방법들보다 높은 공격 성공률을 보이며, 깨끗한 데이터 정확도에 미치는 영향은 최소화되었습니다. 또한 기존 방어 방법에 대한 스텔스성(stealthiness)을 평가한 결과, perturbation 기반 방법에 대해 높은 저항성을 나타냈습니다.



### ECG-Image-Database: A Dataset of ECG Images with Real-World Imaging and Scanning Artifacts; A Foundation for Computerized ECG Image Digitization and Analysis (https://arxiv.org/abs/2409.16612)
- **What's New**: 이 논문에서는 ECG-Image-Database라는 대규모 심전도(ECG) 이미지 데이터베이스를 소개합니다. 이 데이터베이스는 실제 스캐닝 및 물리적 아티팩트가 포함된 다양한 ECG 이미지로 구성되어 있으며, ECG time-series 데이터로부터 생성되었습니다.

- **Technical Details**: 고유한 ECG 이미지는 ECG-Image-Kit라는 오픈소스 파이썬 툴킷을 사용하여 원시 ECG time-series에서 생성되었습니다. 이 툴킷을 통해 PTB-XL 데이터베이스의 977개 12-리드 ECG 레코드와 Emory Healthcare의 1,000개 레코드를 기반으로 높은 신뢰도의 합성 ECG 이미지를 생성하였습니다. 최종 데이터셋에는 35,595개의 소프트웨어 레이블링된 ECG 이미지가 포함되어 있습니다.

- **Performance Highlights**: ECG-Image-Database는 종합적인 이미징 아티팩트와 왜곡의 범위를 포함하고 있어 머신 및 딥러닝 모델 개발에 있어 기준점으로 활용될 수 있습니다. 이 논문은 PhysioNet Challenge 2024의 ECG 이미지 디지털화 및 분류에 사용되었습니다.



### Evaluating and Enhancing Large Language Models for Novelty Assessment in Scholarly Publications (https://arxiv.org/abs/2409.16605)
Comments:
          under review

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 학술 논문에서의 창의성 및 참신성을 평가하기 위한 새로운 벤치마크인 SchNovel을 도입하였습니다. 이 벤치마크는 arXiv 데이터 세트에서 선택된 15,000 쌍의 논문으로 구성되어 있으며, 각 쌍의 최근 발표된 논문이 더 참신하다고 가정합니다. 또한, RAG-Novelty라는 새로운 방법을 제안하여 LLM이 논문의 참신성을 평가할 때 유사한 논문의 검색을 활용합니다.

- **Technical Details**: SchNovel 벤치마크는 2~10년 차이가 나는 논문 쌍을 포함하며, 이는 특히 높은 수준의 리뷰 과정을 거치는 학술 논문에서 참신성을 평가하는 데 중요합니다. RAG-Novelty는 검색 기반 생성 방법으로, 더 참신한 논문일수록 최근 발표된 논문을 더 많이 검색할 것이라는 가정을 바탕으로 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 RAG-Novelty가 기존의 기준 모델보다 논문의 참신성을 평가하는 데 더 뛰어난 성능을 보인다는 것을 입증했습니다. 이 연구는 LLM의 논문 참신성 평가 능력을 깊이 있게 탐구하고, 다양한 카테고리와 출판 연도 간의 변화를 평가하여 LLM의 성능 향상에 기여하였습니다.



### A Hybrid Quantum Neural Network for Split Learning (https://arxiv.org/abs/2409.16593)
Comments:
          47 pages

- **What's New**: 본 연구에서는 Hybrid Quantum Split Learning (HQSL)이라는 새로운 방법을 제안합니다. HQSL은 자원 제약이 있는 환경에서 클래식 클라이언트가 하이브리드 양자 서버와 협력하여 모델을 훈련할 수 있게 돕습니다. 또한, 데이터 프라이버시 문제를 해결하고 재구성 공격에 대한 방어 기제를 제공합니다.

- **Technical Details**: HQSL은 클래식 Neural Network layers로 구성된 클라이언트 측 모델과 양자 레이어와 클래식 Neural Network layers로 구성된 서버 측 모델로 나뉘어 있습니다. 데이터 로딩을 위한 효율적인 qubit 기법을 도입하여 qubit 수와 서킷 깊이를 최소화했습니다. NISQ(Noisy Intermediate-Scale Quantum) 환경을 고려하였으며, Laplacian noise layer를 통해 데이터 프라이버시를 보호합니다.

- **Performance Highlights**: HQSL은 Fashion-MNIST 데이터셋에서 평균적으로 3% 이상의 정확도와 F1-score 향상을 기록하며, Speech Commands 데이터셋에서도 1.5% 이상의 개선을 보였습니다. 최대 100명의 클라이언트를 포함한 실험을 통해 HQSL의 확장성을 검증했습니다.



### MambaJSCC: Adaptive Deep Joint Source-Channel Coding with Generalized State Space Mod (https://arxiv.org/abs/2409.16592)
Comments:
          submitted to IEEE Journal

- **What's New**: 본 논문에서는 저전력 및 효율적인 신경망 모델인 MambaJSCC를 제안합니다. 해당 모델은 깊은 공동 소스-채널 코딩(deep joint source-channel coding, JSCC)을 위한 혁신적인 아키텍처로, 낮은 계산량 및 파라미터 오버헤드로 최첨단 성능을 달성합니다.

- **Technical Details**: MambaJSCC는 이미지 전송을 위해 VSSM-CA(visual state space model with channel adaptation) 블록을 백본 모델로 사용하며, GSSM(generalized state space models) 및 CSI-ReST(zero-parameter, zero-computational channel adaptation method)를 포함한 구조를 가지고 있습니다. GSSM 모듈은 가역적인 매트릭스 변환을 활용하여 일반화된 스캔 확장 작업을 표현하며, 두 개의 GSSM 모듈이 효과적으로 글로벌 정보를 캡처할 수 있음을 이론적으로 증명했습니다.

- **Performance Highlights**: MambaJSCC는 다양한 실험을 통해 기존의 JSCC 방법(예: SwinJSCC)을 모든 주요 아키텍처와 비교하여 왜곡(distortion) 및 지각(perception) 측면에서 우수한 성능을 보였으며, 전반적인 파라미터 크기, 계산 오버헤드, 추론 지연이 크게 감소했습니다. 특히, 최고 신호 대 잡음비(peak-signal-to-noise ratio, PSNR)에서 0.52 dB의 성능 향상을 보여주었습니다.



### AutoSTF: Decoupled Neural Architecture Search for Cost-Effective Automated Spatio-Temporal Forecasting (https://arxiv.org/abs/2409.16586)
Comments:
          16 pages, 13 figures

- **What's New**: 최근 자동화된 spatio-temporal forecasting 방법이 제안되었는데, 이는 복잡한 spatio-temporal 의존성을 캡처하기 위해 최적의 신경망 아키텍처를 automatically 탐색하는 방식이다. 그러나 기존 방법들은 비싼 neural architecture search (NAS) 비용으로 인해 실용성이 떨어진다.

- **Technical Details**: 이 논문에서는 AutoSTF라는 decoupled automated neural architecture search (NAS) 프레임워크를 제안하였다. 이는 mixed search space를 temporal space와 spatial space로 분리하고, representation compression 및 parameter-sharing schemes를 통해 파라미터 폭발 문제를 완화한다. 이러한 방식으로 모델 최적화 과정을 가속화하고, 효과적인 spatio-temporal 의존성 모델링을 가능하게 한다.

- **Performance Highlights**: AutoSTF는 8개의 데이터셋에서 광범위한 실험을 통해 기존 자동 spatio-temporal forecasting 방법에 비해 accuracy와 efficiency 모두에서 우수성을 입증하였다. 특히, 기존 방법들에 비해 13.48배의 속도 향상을 달성하면서도 최고의 예측 정확도를 유지하였다.



### Reactive Multi-Robot Navigation in Outdoor Environments Through Uncertainty-Aware Active Learning of Human Preference Landscap (https://arxiv.org/abs/2409.16577)
- **What's New**: 이 연구에서는 Multi-Robot Systems (MRS)를 위한 새로운 joint preference landscape learning 및 behavior adjusting framework인 PLBA를 제안합니다. 이 프레임워크는 실시간 인간 가이드를 효과적으로 통합하고, 환경 특성을 기반으로 인간의 선호도를 신속하게 평가합니다.

- **Technical Details**: PLBA는 Sparse Variational Gaussian Processes를 활용하여 환경 특성 간의 공간 상관관계를 이용하여 인간의 선호를 평가하며, 최적화 기반의 행동 조정 방법을 통해 MRS의 행동을 안전하게 조정합니다. 이를 통하여 MRS는 'cluttered', 'structured', 'open space'와 같은 다양한 환경에 적응할 수 있습니다.

- **Performance Highlights**: 실험에서 20명의 인간 사용자가 1764개의 피드백을 제공하였고, 그 결과 PLBA의 예측 정확도와 적응 속도가 증가하여 MRS의 행동 적응의 효과성을 입증했습니다.



### Demystifying Issues, Causes and Solutions in LLM Open-Source Projects (https://arxiv.org/abs/2409.16559)
Comments:
          22 pages, 2 images, 6 tables, Manuscript submitted to a journal (2024)

- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전에 따라, 이를 핵심 기능 요소로 활용하는 오픈 소스 소프트웨어 프로젝트가 증가하고 있습니다. 그러나 LLM 오픈 소스 프로젝트의 실무자들이 겪는 도전과제에 대한 연구는 부족합니다. 본 연구는 실무자가 LLM 오픈 소스 소프트웨어 개발 및 사용 중 encountering 하는 문제들과 그 원인 및 해결책을 조사하기 위해 진행되었습니다.

- **Technical Details**: 본 연구에서는 15개의 LLM 오픈 소스 프로젝트에서 닫힌 문제(issue)들을 수집하고 요구 사항에 맞는 문제에 레이블을 지정한 후, 총 994개의 문제를 무작위로 선택하여 데이터 추출 및 분석을 수행했습니다. 주요 발견 사항으로는 (1) 실무자가 가장 많이 직면한 문제는 Model Issue이며, (2) 문제의 가장 일반적인 원인은 Model Problem, Configuration and Connection Problem, Feature and Method Problem으로 확인되었습니다. (3) 문제를 해결하기 위한 주요 솔루션은 Optimize Model이었습니다.

- **Performance Highlights**: 이번 연구를 통해 LLM 오픈 소스 소프트웨어 개발 및 활용에 있어 발생하는 문제와 그 원인 및 잠재적 해결책을 제시하였습니다. 연구 결과는 연구자와 실무자에게 유용한 시사점을 제공할 수 있습니다. 특히 LLM 오픈 소스 프로젝트에서 문제를 해결하기 위한 두 가지 수준의 분류법이 개발되었습니다.



### Source-Free Domain Adaptation for YOLO Object Detection (https://arxiv.org/abs/2409.16538)
Comments:
          ECCV 2024: European Conference on Computer Vision - Workshop on Out-of-Distribution Generalization in Computer Vision Foundation Models, Milan Italy

- **What's New**: 본 논문에서는 Object Detection(OD)의 Source-Free Domain Adaptation(SFDA) 분야에서 YOLO 계열의 단일 단계 탐지기를 향상시키는 새로운 방법인 Source-Free YOLO(SF-YOLO)를 제안합니다.

- **Technical Details**: SF-YOLO는 Teacher-Student 프레임워크를 기반으로 하여, 학생 모델이 특정 타겟 도메인에 대한 학습된 데이터 증강 기법을 통해 훈련됩니다. 이 방법은 레이블이 없는 타겟 데이터만을 사용하며 기능 정렬(Feature Alignment)을 요구하지 않습니다. 또한, 새로운 Student Stabilisation Module(SSM)을 도입하여 훈련의 안정성을 높이고, 레이블이 없는 상황에서의 정확도 저하 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SF-YOLO는 Cityscapes, Foggy Cityscapes, Sim10k, KITTI 데이터셋에서 여러 도전적인 도메인 적응 벤치마크에서 현재의 최고 성능을 보이는 탐지기들과 경쟁할 수 있으며, 심지어는 소스 데이터를 사용하는 적응 방법보다 나은 성능을 기록하기도 하였습니다. 저희의 접근법은 낮은 계산 자원을 요구하며, 실용적인 실시간 응용에 적합합니다.



### Center-fixing of tropical cyclones using uncertainty-aware deep learning applied to high-temporal-resolution geostationary satellite imagery (https://arxiv.org/abs/2409.16507)
Comments:
          Submitted to AMS journal Weather and Forecasting. Main body is 52 pages and 14 figures; supplement is another 33 pages and 28 figures

- **What's New**: 이번 연구에서는 열대 사이클론(tropical cyclone, TC)의 중심을 정확히 찾아내는 깊이 학습 알고리즘인 GeoCenter를 개발했습니다. 이 알고리즘은 지구 정지 IR (infrared) 위성 이미지만을 사용하여 모든 TC 분지에서 고주파 (10-15분)로 데이터를 제공하며, 낮과 밤 모두 적용 가능합니다.

- **Technical Details**: GeoCenter는 10개 채널의 IR 이미지를 시간 시리즈로 포함한 애니메이션을 흡수하여 작동합니다. 초기 추정 위치에서의 오프셋을 평균 48km에서 100km 이상으로 보정하는 임무를 수행하며, 최대 3시간 시간 지연을 처리할 수 있습니다. GeoCenter는 독립적인 테스트 데이터셋에서 평균/중앙값/RMS 오차가 각각 26.9/23.3/32.0 km, 열대 시스템의 경우 25.7/22.3/30.5 km, 카테고리 2-5 허리케인의 경우 15.7/13.6/18.6 km인 성능을 보였습니다.

- **Performance Highlights**: GeoCenter는 ARCHER-2의 성능과 유사한 오차를 보였으며, IR 데이터만 사용할 때는 더 나은 결과를 제공합니다. 또한, 200개의 TC 중심 위치에 대한 잘 보정된 앙상블을 포함하여 효율적인 불확실성 정량화(uncertainty quantification, UQ)를 수행합니다. 실시간으로 사용할 수 있는 모든 예측자를 이용해 10-15분 간격으로 운영하기 용이한 알고리즘입니다.



### GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization (https://arxiv.org/abs/2409.16502)
Comments:
          Project website at this https URL

- **What's New**: 본 연구에서는 3D Gaussian Splatting (3DGS) 기술을 활용하여 시각적 로컬라이제이션(visual localization)을 향상시키는 새로운 프레임워크 GSplatLoc을 제안합니다. 이 방법은 기존의 메모리 소모나 최적화 요구 사항을 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: GSplatLoc은 XFeat의 경량 키포인트 감지 및 기술 모델로 생성된 고밀도 기술 맵을 활용하여 3DGS에 밀집된 키포인트 설명자(dense keypoint descriptors)를 증류(distill)합니다. 이를 통해 공간 이해도를 개선하고, 2D-3D 대응(relations)을 통해 더 정확한 카메라 포즈 예측을 가능하게 합니다. 초기 포즈 추정 후에는 포토메트릭 왜곡 손실(photometric warping loss)을 사용하여 포즈를 세분화(refine)합니다.

- **Performance Highlights**: 이번 연구는 인기 있는 실내 및 실외 데이터셋에서 벤치마킹한 결과, 기존의 최첨단 Neural Render Pose (NRP) 방법들, 특히 NeRFMatch와 PNeRFLoc을 능가하는 성과를 보여주었습니다.



### To Explore the Potential Inhibitors against Multitarget Proteins of COVID 19 using In Silico Study (https://arxiv.org/abs/2409.16486)
Comments:
          22 pages

- **What's New**: COVID-19 팬데믹에 대응하기 위해 약물 재창출(Drug Repurposing) 전략을 활용한 새로운 연구가 발표되었습니다. 본 연구는 COVID-19에 대한 잠재적인 억제제를 탐색하기 위해 분자 도킹(Molecular Docking) 및 머신러닝 회귀(Machine Learning Regression) 기법을 결합하였습니다.

- **Technical Details**: 연구진은 분자 도킹 과정을 사용하여 여러 약물의 다중 타겟 단백질에 대한 결합 친화력을 계산하였습니다. 또한, 다양한 머신러닝 회귀 접근 방식을 활용한 QSAR 모델링을 수행하였으며, 특히 결정 트리 회귀(Decision Tree Regression) 모델이 가장 적합한 모델로 밝혀졌습니다.

- **Performance Highlights**: 본 연구에서는 -19.7 kcal/mol에서 -12.6 kcal/mol 범위의 평균 결합 친화도를 가지는 5개 새로운 유망 억제제(Zinc IDs: ZINC 3873365, 85432544, 8214470, 85536956, 261494640)를 제안하였습니다. 이들 억제제의 생리화학적(Physicochemical) 및 약물동태학적(Pharmacokinetic) 특성 분석을 통해 효과적인 공공 보건 치료제를 찾기 위한 기초를 마련하였습니다.



### Algorithmic Drift: A Simulation Framework to Study the Effects of Recommender Systems on User Preferences (https://arxiv.org/abs/2409.16478)
- **What's New**: 이 논문은 추천 시스템의 장기적인 사용자 행동 변화에 대한 영향을 정량화할 수 있는 새로운 접근 방식을 제안합니다. 이 연구는 사용자와 추천 알고리즘 간의 상호작용을 모델링하기 위한 확률적 시뮬레이션 프레임워크를 채택합니다.

- **Technical Details**: 논문에서는 사용자 저항(user resistance) 및 관성(inertia)과 같은 행동적 측면을 포함하여 사용자 모델을 공식화하고, 추천 알고리즘이 사용자 선호에 미치는 영향을 평가하기 위해 새로운 메트릭스(새로운 지표)를 도입합니다. 시뮬레이션 모델은 처음에 이질적인 사용자 선호 그룹에서 시작되며, 추천 시스템이 초기 전이 확률을 유도하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법론이 사용자 선호의 변화(algorithmic drift)를 정확히 감지하고 정량화하는 데 효과적이라는 것을 입증하였습니다. 다양한 신합성 데이터셋을 통해 시스템의 강건성을 검증하며, 사용자 선택의 시나리오와 하이퍼 파라미터 설정에 따른 결과를 평가했습니다.



### Lessons Learned from a Unifying Empirical Study of Parameter-Efficient Transfer Learning (PETL) in Visual Recognition (https://arxiv.org/abs/2409.16434)
Comments:
          Code is available at this https URL

- **What's New**: 최근 Parameter-efficient transfer learning (PETL) 기술에 대한 관심이 커지고 있으며, 이를 통해 기존의 대규모 pre-trained 모델을 더욱 효율적으로 조정하여 다양한 downstream 작업에서의 성능을 향상시키고자 하는 연구가 진행되고 있습니다. 본 논문은 Vision Transformers의 맥락에서 PETL 방법들을 통합적으로 비교하고, 그들의 성능을 체계적으로 분석하였습니다.

- **Technical Details**: 이 연구에서는 Low-Rank Adaptation (LoRA), Visual Prompt Tuning (VPT), Adapter 등 다양한 PETL 기법을 사용하여 진행하였으며, 하이퍼 파라미터(learning rate, weight decay 등)를 체계적으로 조정하여 low-shot benchmark인 VTAB-1K의 정확도를 비교하였습니다. 또한, CIFAR-100 및 RESISC와 같은 풀사이즈 데이터셋에서도 PETL 방법을 평가하였습니다.

- **Performance Highlights**: PETL 접근 방식들이 잘 조정되었을 경우 VTAB-1K에서 유사한 정확도를 기록하였으며, PETL 방법들은 낮은 샷의 데이터에서도 뛰어난 성능을 보여주었습니다. 무수한 학습 데이터를 가진 시나리오에서도 PETL이 full fine-tuning과 동등하거나 그 이상의 결과를 도출할 수 있다는 점이 주목할 만합니다. 또한 PETL은 distribution shift에 대한 강건성을 가지며, 기존 모델의 일반성을 유지하는 결과를 보였습니다.



### A Comprehensive Survey of Bias in LLMs: Current Landscape and Future Directions (https://arxiv.org/abs/2409.16430)
Comments:
          2 Tables, 1 Figure

- **What's New**: 이 논문은 Large Language Models(LLMs)에서의 편향(bias)에 대한 종합적인 조사(survey)를 제공하며, 다양한 유형, 출처, 영향 및 완화 전략을 체계적으로 분류하고 있습니다.

- **Technical Details**: LLMs의 편향을 여러 차원으로 분류한 후, 현재 연구 결과를 종합하고 실제 응용에서의 편향의 함의를 논의합니다. 또한 기존의 편향 완화(bias mitigation) 기법을 비판적으로 평가하고 LLMs의 공정성(fairness) 및 형평성(equity)을 향상시키기 위한 미래 연구 방향을 제시합니다.

- **Performance Highlights**: 이 조사는 연구자, 실무자 및 정책 입안자들에게 LLMs의 편향 문제를 해결하고 이해하는 데 기초 자료로 활용될 수 있는 중요한 리소스를 제공합니다.



### Leveraging Local Structure for Improving Model Explanations: An Information Propagation Approach (https://arxiv.org/abs/2409.16429)
- **What's New**: 최근 심층 신경망(DNN) 모델의 결정 해석을 위한 다양한 설명 방법들이 개발되었으며, 본 논문에서는 IProp이라는 새로운 방법을 제안합니다. IProp은 각 픽셀의 기여도를 독립적으로 평가하는 대신, 이웃 픽셀과의 구조적 유사성을 고려해 공동으로 평가합니다.

- **Technical Details**: IProp은 각 픽셀의 기여도를 설명 정보의 소스로 모델링하며, Markov Reward Process(MRP)를 통해 모든 픽셀 간의 정보 전파를 다이나믹하게 처리합니다. 정보 전파는 연속적으로 발생하며, 픽셀 간의 상관관계를 포착합니다.

- **Performance Highlights**: IProp은 다양한 DNN 모델과 기존 설명 방법에 대한 실험을 통해 해석 가능성 메트릭에서 현저한 개선을 확인했으며, 정량적 및 정성적으로 모든 기준 방법보다 우수한 결과를 보여주었습니다.



### Lessons for Editors of AI Incidents from the AI Incident Databas (https://arxiv.org/abs/2409.16425)
Comments:
          8 pages, 0 figures

- **What's New**: 본 연구는 AI 사건 데이터베이스(AIID)의 750개 이상의 AI 사건을 검토하고, 이러한 사건에 적용된 두 개의 독립적인 분류 체계를 분석하여 AI 사건 인덱싱 및 분석에 대한 공통적인 도전 과제를 식별합니다. 연구자는 AI 사건 보고에서 불확실성을 피할 수 없는 구조적 모호성을 발견하고, 이러한 불확실성과 관련된 사건 프로세스를 보다 강화할 수 있는 방법을 보고합니다.

- **Technical Details**: AIID는 AI 사건을 분류하고 기록하는 플랫폼으로, AI 사건에 대한 메타데이터를 제공하여 사건의 재발 방지를 위한 분석을 가능하게 합니다. AIID는 두 가지 주요 세부 분류 체계인 CSET AI Harm Taxonomy와 Goals, Methods, and Failures Taxonomy를 통해 각 사건을 구체적이고 다양한 관점에서 분석하며, 750개 이상의 AI 사건을 포함하여 3000개 이상의 제3자 보고서를 인덱스하여 제공합니다.

- **Performance Highlights**: AIID의 데이터셋은 다양한 AI 시스템과 맥락을 포괄하며, 자율 주행 차량 사고와 알고리즘적 차별과 같은 다양한 사건을 포함합니다. AIID의 결과는 AI 사건의 발생 빈도를 줄이고, AI 시스템 개발 및 배포의 안전성을 높이는 데 기여합니다.



### Task-oriented Prompt Enhancement via Script Generation (https://arxiv.org/abs/2409.16418)
Comments:
          17 pages + reference

- **What's New**: 이번 연구에서는 TITAN이라는 새로운 전략을 제안하여 대형 언어 모델(LLMs)의 작업 지향적인 프롬프트 성능을 향상시킵니다. TITAN은 제로샷 학습 기법을 통해 코드를 생성하여 task-oriented 문제를 해결하는 데 중점을 두었습니다.

- **Technical Details**: TITAN은 step-back prompting과 chain-of-thought prompting이라는 두 가지 핵심 기술을 활용하여 입력 사양을 추출하고 필요 절차 단계를 식별합니다. 이 정보는 LLM의 코드 생성 프로세스를 개선하는 데 사용되며, 추가적인 후처리를 통해 생성된 스크립트를 보완하여 최종 결과를 도출합니다.

- **Performance Highlights**: TITAN은 다양한 작업에서 LLM의 성능을 향상시키며, 평균적으로 GPT-3.5와 GPT-4에 대해서 각각 7.6% 및 3.9%의 정확도 개선을 보여주었습니다. TITAN은 11개의 데이터셋 중 8개에서 최첨단 성능을 달성하며, 인간의 개입이 필요한 few-shot 접근 방식에 비해 소폭 손실을 보인 사례도 있었습니다.



### Selection of Prompt Engineering Techniques for Code Generation through Predicting Code Complexity (https://arxiv.org/abs/2409.16416)
Comments:
          18 pages + reference

- **What's New**: 이번 논문에서는 코드 생성 정확성을 향상시키기 위해 다양한 prompt engineering techniques (PETs)을 선택할 수 있는 PET-Select 모형을 제안합니다.

- **Technical Details**: 이 모델은 code complexity를 기반으로 쿼리를 분류하고, 적절한 PET를 선택하는 PET-agnostic (PET 비 의존적) 방식을 채택합니다. 또한, contrastive learning 기법을 통해 간단한 문제와 복잡한 문제를 효과적으로 구분합니다.

- **Performance Highlights**: GPT-3.5 Turbo 및 GPT-4o를 사용한 MBPP와 HumanEval 벤치마크 평가에서 pass@1 정확도가 최대 1.9% 향상되었으며, 토큰 사용량이 74.8% 감소하였습니다.



### Modern Hopfield Networks meet Encoded Neural Representations -- Addressing Practical Considerations (https://arxiv.org/abs/2409.16408)
Comments:
          17 pages, 8 figures, workshop submission to Neurips

- **What's New**: 본 논문은 Modern Hopfield Networks (MHN)에 대한 메타 안정 상태 문제를 해결하는 새로운 접근 방식인 Hopfield Encoding Networks (HEN)를 소개합니다. HEN은 입력 패턴의 분리 가능성을 높이고, 메타 안정 상태를 줄이기 위해 인코딩된 신경 표현을 통합합니다.

- **Technical Details**: HEN은 미리 훈련된 신경 인코더-디코더 모델을 사용하여 입력을 잠재 표현 공간으로 인코딩한 후 저장하고, 재호출 시 다시 디코딩하는 방법을 사용합니다. 이 접근 방식은 MHNs의 메타 안정 상태 문제를 해결하고, 자연어 쿼리를 통한 다양한 입력 모달리티에서의 검색을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 HEN이 메타 안정 상태를 크게 줄이고, 저장 용량을 증가시키면서 다양한 입력을 완벽히 기억할 수 있음을 나타냅니다. 이는 실제 작업을 위한 연상 기억 네트워크의 실용적인 활용을 향상시킵니다.



### Future-Proofing Medical Imaging with Privacy-Preserving Federated Learning and Uncertainty Quantification: A Review (https://arxiv.org/abs/2409.16340)
Comments:
          21 pages, 5 figures, 4 tables, Review paper, preprint to Radiology AI. arXiv admin note: text overlap with arXiv:2406.12815

- **What's New**: 이번 논문은 Federated Learning (FL)을 통해 의료 이미징에서의 AI 모델 훈련을 개선할 수 있는 가능성을 탐구합니다. 특히, 데이터 공유 없이 민감한 정보를 보호하면서 협업할 수 있는 방법론을 다루며, 기존의 FL 접근법의 기본적인 문제점과 한계를 짚어봅니다.

- **Technical Details**: FL은 분산된 환경에서 여러 기관이 AI 모델을 공동으로 훈련할 수 있도록 하는 기술입니다. 이 과정에서 모델의 update 정보(예: gradients)만 공유되며 데이터는 공유되지 않습니다. 그러나 민감한 정보가 여전히 추론될 수 있는 가능성이 남아 있으며, 이는 privacy-preserving Federated Learning (PPFL)와 Uncertainty Quantification (UQ) 연구와 깊은 연관이 있습니다. 논문에서는 다양한 FL 알고리즘(예: FedAvg, FedProx 등)의 발전을 소개하고, 비독립적이고 동질적이지 않은 데이터 세트(data heterogeneity)의 문제를 해결하기 위한 접근 방식도 설명합니다.

- **Performance Highlights**: 이 논문은 FL이 의료 이미징에서 모델의 신뢰성을 높이는 방법으로 여겨지며, 정확하고 일반화 가능한 AI 모델 개발에 기여할 수 있는 잠재력을 가지고 있다고 강조합니다. 특히, 각기 다른 클라이언트 환경에서의 데이터 이질성을 고려하는 Personalized Federated Learning (PFL) 기법의 중요성을 제시하며, FL의 적용 사례를 통해 효과를 분석합니다.



### Exploring the traditional NMT model and Large Language Model for chat translation (https://arxiv.org/abs/2409.16331)
Comments:
          7 pages, 6 Tables, WMT24

- **What's New**: 이 논문에서는 WMT24 채팅 번역 공유 과제에서 영어와 독일어 간의 이중 번역(en-de) 작업을 위한 Huawei Translation Services Center(HW-TSC)의 제출 사례를 다룹니다. 연구에서는 채팅 데이터를 이용한 모델의 파인튜닝(fine-tuning)과 함께 Minimum Bayesian Risk (MBR) 디코딩 및 자기 학습(self-training) 등의 다양한 전략을 탐색했습니다.

- **Technical Details**: 본 논문에서는 Transformer-Big 아키텍처를 기반으로 한 모델을 사용하며, 기존 NMT 모델 외에 대형 언어 모델(LLM)의 활용을 통해 번역 작업의 새로운 패러다임을 제시합니다. Minimum Bayesian Risk (MBR) 디코딩 기법은 여러 후보 중에서 최소 예상 오류를 가진 출력을 선택하는 방식입니다. 자기 학습 기법(self-training)과 역번역(back-translation)을 통해 번역 품질을 향상시키는 방법을 설명합니다.

- **Performance Highlights**: 결과적으로 MBR 자기 학습 방법이 가장 우수한 성능 향상을 보여주었고, 기계 번역 품질의 측면에서 확실한 개선이 있었습니다. 그러나 LLM의 출력 comet 메트릭은 NMT 모델의 최적 결과를 초과하지 못했습니다.



### MRI Radiomics for IDH Genotype Prediction in Glioblastoma Diagnosis (https://arxiv.org/abs/2409.16329)
Comments:
          8 pages, 1 figure

- **What's New**: 이번 논문은 MRI 이미지를 활용한 Radiomics의 최신 동향을 다루고 있으며, 특히 Isocitrate dehydrogenase (IDH) 돌연변이 상태를 식별하는데 중점을 둡니다. IDH 돌연변이는 고등급 교모세포종과 등급 IV 별아교종의 중요 생체표지자로, 비침습적인 진단 방법의 필요성을 강조합니다.

- **Technical Details**: 논문에서 다루는 MRI Radiomics 워크플로우는 MRI 이미지에서 특징 추출을 위한 주요 단계를 설명합니다. 이미지 세분화는 수동, 반자동 또는 자동 방법으로 수행될 수 있으며, 자동 세분화는 딥러닝 모델을 사용하여 더 빠르고 정확하게 수행됩니다. 이미지 전처리 과정은 필수적으로 포함되며, 스컬 스트리핑과 다양한 필터링 기법이 사용됩니다.

- **Performance Highlights**: 이 연구는 IDH 돌연변이 상태를 정확히 예측하기 위한 MRI 기반 비침습적 방법의 효과를 입증하고 있으며, 이는 각환자에 맞춘 치료 계획 수립에 기여할 것입니다. 또한, 딥러닝 기반의 자동 세분화 기법은 임상 적용 가능성을 높이고 있습니다.



### Automated Spatio-Temporal Weather Modeling for Load Forecasting (https://arxiv.org/abs/2409.16326)
- **What's New**: 본 논문에서는 전력 수요 예측을 위해 기상 모델링을 개선하기 위한 새로운 접근 방식을 제안합니다. 심층 신경망(deep neural networks)의 자동화된 표현 및 공간-시간 특성(spatio-temporal feature) 추출 능력을 활용하여, 기존의 고정된 모델링 방법에서 벗어나 새로운 방법을 탐구합니다.

- **Technical Details**: 이 연구에서는 전력 수요(load)와 재생 에너지 생산에 대한 예측 정확도를 높이기 위해 기온, 바람 및 일조시간과 같은 기상 변수의 공간적 및 시간적 변동성을 설명하는 복합 생태계(model)를 개발했습니다. 여러 기상 관측소(observations) 및 기상 모델(simulated data)에서 얻은 데이터를 활용하여 이러한 변수를 동시에 모델링하는 방법을 제시합니다.

- **Performance Highlights**: 프랑스 국가 전력 수요에 대한 최신 모델과의 비교 연구를 통해, 제안된 심층 학습 기반(deep learning-based) 방법론이 전력망(Grid)의 성능 및 안정성을 개선하는 데 기여할 수 있음을 보여줍니다. 이 접근법은 재생 에너지 생산 예측에도 완전히 적용될 수 있음을 강조합니다.



### Towards Within-Class Variation in Alzheimer's Disease Detection from Spontaneous Speech (https://arxiv.org/abs/2409.16322)
- **What's New**: 이 논문은 알츠하이머병(Alzheimer's Disease, AD) 탐지 분야에서 머신러닝(classification model)을 활용하여 AD 환자와 비환자를 구별하려는 연구의 일환이다. 기존의 이진 분류(binary classification) 접근법의 한계점을 지적하며, 내적 변동성(within-class variation) 및 샘플 불균형(instance-level imbalance) 문제를 해결하기 위한 두 가지 새로운 방법, Soft Target Distillation(SoTD)와 Instance-level Re-balancing(InRe)을 제안한다.

- **Technical Details**: AD 탐지의 중요한 문제는 샘플 간 인지 기능의 정도가 다르다는 것이다. SoTD는 샘플의 세부 정보를 인식하여 신뢰도(Awareness of sample degree)를 바탕으로 단계적인 학습을 제공하고, InRe는 데이터 로스(loss)를 재조정하여 오버피팅(over-fitting)을 완화하는 방법이다. 실험은 ADReSS와 ADReSSo 데이터셋에서 BERT 및 RoBERTa 임베딩(features)을 사용하여 수행되었다.

- **Performance Highlights**: 실험 결과, SoTD와 InRe 방법을 도입함으로써 AD 탐지 정확도가 크게 향상되었으며, SoTD는 특히 모델 앙상블(ensemble estimation) 기법에 비해 더 높은 효율성을 보였다. 또한, InRe는 모델의 오버피팅을 현저히 줄이며, 훈련 안정성을 높였다.



### Developing a Thailand solar irradiance map using Himawari-8 satellite imageries and deep learning models (https://arxiv.org/abs/2409.16320)
Comments:
          23 pages, 14 figures

- **What's New**: 이 논문은 태국의 태양 복사 지도(Global Horizontal Irradiance, GHI)를 30분마다 온라인으로 보여주는 플랫폼을 소개합니다. 이 플랫폼은 Himawari-8 위성 이미지를 기반으로 한 구름 지수(cloud index)와, Linke turbidity로 조정된 Ineichen 맑은 하늘 모델을 사용하여 GHI를 추정합니다.

- **Technical Details**: GHI 추정 모델에서 입력으로는 맑은 하늘 복사량, 구름 지수, MERRA-2 데이터베이스의 재분석 GHI 및 온도 데이터를 포함합니다. 사용된 머신 러닝 모델로는 LightGBM, LSTM, Informer, Transformer가 있으며, 2022-2023년 기간 동안 53개 지상 스테이션에서 15분 단위의 GHI 데이터를 평가하여 성능을 비교했습니다.

- **Performance Highlights**: 모든 모델은 경쟁력 있는 성능을 보였으며, SolCast 서비스보다 우수한 결과를 나타냈습니다. LightGBM 모델의 MAE(Mean Absolute Error)는 78.58 W/sqm, RMSE(Root Mean Square Error)는 118.97 W/sqm로 최상위 성능을 기록했습니다. Informer 모델은 추가적으로 재분석 MERRA-2 데이터 없이 MAE 78.67 W/sqm로 우수한 성능을 보였습니다.



### A Literature Review of Keyword Spotting Technologies for Urdu (https://arxiv.org/abs/2409.16317)
- **What's New**: 이 문헌 리뷰는 파키스탄의 저자원 언어(Low-Resource Language)인 우르두어의 키워드 스포팅(Keyword Spotting, KWS) 기술 발전을 조사합니다. 저자원 언어가 직면한 독특한 도전과제와 이를 해결하기 위한 맞춤형 솔루션의 필요성을 강조합니다.

- **Technical Details**: 이 리뷰는 가우시안 혼합 모델(Gaussian Mixture Models, GMMs)에서 심층 신경망(Deep Neural Networks, DNNs) 및 변환기(transformer)와 같은 복잡한 신경 아키텍처의 진화를 추적합니다. 특히 다중 작업 학습(multi-task learning)과 자가 감독 학습(self-supervised learning) 접근법을 통합하여 라벨 없는 데이터(unlabeled data)를 활용하는 방법을 강조합니다. 새로운 EdgeCRNN 모델과 통합된 CNN과 RNN을 포함하여, 키워드 탐지를 위한 최신 모델들과 그 효율성을 논의합니다.

- **Performance Highlights**: 최신 연구에서, 자가 감독 학습(S3RL) 및 경량 변환기 모델들이 우르두어와 같은 저자원 언어에서 KWS 효율성과 정확성을 향상시키는데 긍정적인 영향을 미쳤습니다. Massively Multilingual Speech(MMS) 프로젝트는 1000개 이상의 언어에서 모델을 사전 학습하여 현대적인 음성 기술을 다수의 언어로 확대했으나, 여전히 우르두어는 데이터 부족 문제로 인해 성능 향상에 제한이 있습니다.



### Surface solar radiation: AI satellite retrieval can outperform Heliosat and generalizes well to other climate zones (https://arxiv.org/abs/2409.16316)
Comments:
          19 pages, 11 figures

- **What's New**: 본 논문은 유럽 전역에서 즉각적인 표면 태양 복사량(Surface Solar Irradiance, SSI)을 정확히 추정할 수 있는 최초의 머신러닝 기반 위성 검색 모델을 소개합니다. 이는 데이터 기반의 Heliosat 알고리즘 에뮬레이션 및 기상 관측소에서의 미세 조정을 통해 가능합니다.

- **Technical Details**: SSI 검색 모델은 복사 전달 모델(radiative transfer model)을 에뮬레이션하고, 기상 관측소의 SSI 데이터를 이용하여 훈련을 진행합니다. 또한, 이 모델은 구름 상태에 따라 Meteosat 채널과 태양 천정 각(solar zenith angle)과 같은 예측 변수의 상대적 중요성을 정량화하여 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구에서는 Heliosat 모델에 비해 높은 정확도를 보여주며, 특히 다양한 기후 조건 및 표면 알베도(surface albedo)를 가진 지역에서도 뛰어난 일반화 성능을 발휘합니다. 특히 구름 조건에서는 여러 근적외선 채널이 성능을 향상시키는 것으로 나타났습니다.



### SEE: Semantically Aligned EEG-to-Text Translation (https://arxiv.org/abs/2409.16312)
Comments:
          4 pages

- **What's New**: 본 연구는 EEG (Electroencephalography) 신호를 텍스트로 변환하는 EEG-to-Text 디코딩의 한계를 극복하기 위해 SEE (Semantically Aligned EEG-to-Text Translation)라는 혁신적인 방법을 제안합니다. 이 방법은 두 개의 모듈 (Cross-Modal Codebook 및 Semantic Matching Module)을 프리트레인된 BART 모델에 통합하여 다양한 EEG-Text 쌍 간의 의미적 일치를 높이고, 도메인 간의 간극을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: SEE는 두 개의 주요 모듈로 구성됩니다. 1) Cross-Modal Codebook는 다양한 모달리티의 표현을 학습하여 특징 통합 및 모달리티 편향 완화를 지원합니다. 2) Semantic Matching Module은 EEG-Text 쌍에서 유도된 다중 모달 특징들을 의미 일치에 따라 정렬하여 노이즈의 영향을 줄이는 역할을 합니다. 이러한 모듈들은 BART와 같은 사전 훈련된 언어 모델에 원활하게 통합됩니다.

- **Performance Highlights**: Zurich Cognitive Language Processing Corpus (ZuCo) 데이터셋에서의 실험 결과, SEE 방법은 EEG-to-Text 디코딩의 정확성을 향상시키는 데 효과적임을 입증하였습니다. 특히, 기존의 모델들과 비교했을 때 높은 성능을 나타내며, 최첨단(State-of-the-art) 결과를 달성하였습니다.



### DeepScore: A Comprehensive Approach to Measuring Quality in AI-Generated Clinical Documentation (https://arxiv.org/abs/2409.16307)
Comments:
          9 pages, 5 figures, 6 tables

- **What's New**: 이 논문은 DeepScribe의 의료 문서 품질 평가 방법론에 대해 설명합니다. 특히, 다양한 지표와 종합 점수인 'DeepScore'를 통한 품질 및 정확성을 측정하는 방법에 초점을 맞춥니다.

- **Technical Details**: DeepScribe는 'Stat Rates'라는 시스템을 통해 AI가 생성한 의료 노트의 품질을 평가하고, 즉각적인 수정과 알고리즘의 발전 방향을 제시합니다. Major Defect-Free Rate (MDFR), Critical Defect-Free Rate (CDFR), Captured Entity Rate (CER), Accurate Entity Rate (AER), Minimally-Edited Note Rate (MNR), Medical Word Hit Rate (MWHR)와 같은 다양한 지표를 활용하여, AI 문서의 정확성과 사용자 수용성을 분석합니다.

- **Performance Highlights**: DeepScore는 앞서 언급한 모든 지표를 평균내어 계산하여 생성됩니다. 이를 통해 의료 문서 품질에 대한 종합적인 평가를 제공하며, 지속적인 개선을 위한 지침 역할을 합니다.



### HyperAgent: Generalist Software Engineering Agents to Solve Coding Tasks at Sca (https://arxiv.org/abs/2409.16299)
- **What's New**: 이번 논문에서 소개하는 HyperAgent는 다양한 프로그램 언어를 활용하여 소프트웨어 엔지니어링(Software Engineering, SE) 작업에 대한 일반적인 다중 에이전트 시스템입니다. 이 시스템은 인간 개발자의 작업 흐름을 모방하여 광범위한 SE 작업을 해결하도록 설계되었습니다.

- **Technical Details**: HyperAgent는 네 개의 전문화된 에이전트로 구성됩니다: Planner, Navigator, Code Editor, Executor. 이 시스템은 초기 개념화에서 최종 검증까지 SE 작업의 전체 생명 주기를 관리하며, 다양한 SE 작업에 대한 성능 평가를 통해 최신 기술의 성능을 자랑합니다.

- **Performance Highlights**: HyperAgent는 GitHub 문제 해결을 위한 SWE-Bench-Lite에서 25.01%의 성공률, SWE-Bench-Verified에서 31.40%의 성공률을 기록하며 기존 방법을 초월하는 성능을 보였습니다. 또한 repository-level code generation, fault localization, program repair에서 SOTA 성능을 입증하여 복잡한 다단계 SE 작업 처리에서의 잠재력을 보여주었습니다.



### Explaining Human Comparisons using Alignment-Importance Heatmaps (https://arxiv.org/abs/2409.16292)
- **What's New**: 이 논문에서는 사람의 유사성 판단을 비교하는 과정을 설명하기 위해 Alignement Importance Score (AIS) 열지도를 제안하고 있습니다. AIS는 Deep Neural Network (DNN)의 표현 기하학과 인간의 그것 사이의 정렬에 대한 기여도를 측정합니다.

- **Technical Details**: 이 연구는 DNN의 마지막 합성곱 층에서 핀셋하여 이미지에 대한 중요한 정보를 설명합니다. 구체적으로, AIS를 활용하여 높은 평가 점수를 가진 특징 맵만 사용하여 인간의 유사성 판단을 예측하는 데 있어 정확도를 높입니다. 연구는 전통적인 saliency map과 비교하여 결과의 해석 가능성을 평가합니다.

- **Performance Highlights**: DNN의 임베딩으로부터 인간의 유사성 판단을 예측하는 데 Alignment Importance가 개선된 결과를 보였으며, 이미지 공간에서 어떤 정보가 중요한지를 설명하는 통찰력을 제공합니다.



### Beyond Following: Mixing Active Initiative into Computational Creativity (https://arxiv.org/abs/2409.16291)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 연구에서는 Active Mixed Initiative Co-Creative (MI-CC) 시스템에서 AI 에이전트의 적응 능력이 인간 창작자의 창의적 책임 기대에 미치는 영향을 조사하였습니다. Reinforcement Learning (RL) 방법을 이용해 인간 사용자의 창의적 책임 선호도를 학습하는 시스템을 개발하였습니다.

- **Technical Details**: 우리는 Multi-armed-bandit (MAB) 에이전트를 개발하여 인간 창작자로부터 학습하며 협력적 의사결정을 업데이트하고 MI-CC 경험 중에 다양한 능력으로 전환할 수 있는 시스템을 구축하였습니다. 39명의 참가자를 대상으로 한 연구에서, 학습 기능이 없는 시스템에 비해 우리 시스템의 학습 능력이 높이 평가되었습니다.

- **Performance Highlights**: 조사 결과, MI-CC 경험에 대한 전반적인 만족도가 유의미하게 증가하였으며, 참가자들은 AI 에이전트의 학습 능력에 대해 높은 인식을 보였습니다. 이 발견은 적극적인 AI 주도의 MI-CC 상호작용과 참가자 간의 깊은 이해의 관련성을 강조합니다.



### M^2PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning (https://arxiv.org/abs/2409.15657)
Comments:
          EMNLP 2024

- **What's New**: 이 연구에서는 Multimodal Prompt Tuning (M^2PT) 이라는 새로운 접근 방식을 소개하여 멀티모달 대형 언어 모델(MLLM) 의 효과적인 instruction tuning을 지원하게 됩니다. M^2PT는 시각적 프롬프트(visual prompts)와 텍스트 프롬프트(textual prompts)를 통합하여 전반적인 매개변수의 0.09%만 조정하면서도 우수한 성능을 발휘합니다.

- **Technical Details**: M^2PT는 비전 인코더(vision encoder)와 언어 프로세서(language processor) 각각에 시각적 프롬프트와 텍스트 프롬프트를 삽입하여 기능(feature) 를 추출하고 각 모달리티(modality) 간의 정렬을 촉진합니다. 이 과정에서 두 세트의 프롬프트 간의 교차 모달 상호작용(cross-modality interaction)이 강화되며, 이를 통해 모델이 맥락을 이해하고 제로샷 제어(zero-shot control)에서 모호성을 줄일 수 있도록 합니다.

- **Performance Highlights**: M^2PT는 여러 멀티모달 평가 데이터셋에서 여러 최신 PEFT (Parameter-Efficient Fine-Tuning) 기법과 비교하여 우수한 성능을 나타냈습니다. 특히, 기법의 효율성과 효과성을 검증하기 위한 포괄적인 경험적 실험이 진행되었습니다.



### Block-Attention for Efficient RAG (https://arxiv.org/abs/2409.15355)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 시나리오에서의 추론 지연 및 비용을 줄이기 위해 설계된 새로운 attention 메커니즘인 Block-Attention을 소개합니다. 기존 방법들과 달리, Block-Attention은 검색된 문서를 블록으로 나누어 각 블록이 KV(state) 상태를 독립적으로 계산하도록 하여 효율성을 높입니다.

- **Technical Details**: Block-Attention 메커니즘은 입력 시퀀스를 여러 블록으로 나누고, 각 블록은 자신만의 KV 상태를 self-attention을 통해 계산합니다. 마지막 블록만이 다른 블록에 접근합니다. 이 기법을 통해 모든 passage를 블록으로 정의하고 그 KV 상태를 메모리에 캐시하여, 추론 시의 지연 시간을 크게 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, Block-Attention을 적용하면 평균 TTFT(첫 번째 토큰까지의 시간)를 98.7% 줄일 수 있으며(3638 ms에서 45 ms로 단축), 첫 번째 토큰을 추출할 때의 FLOPs 역시 99.8% 감소시킵니다. 블록-어텐션 모델은 기존의 self-attention 모델에 비해 (Llama3: 68.4% vs 67.9%, Mistral: 62.8% vs 59.6%) 유사하거나 더 나은 성능을 보여줍니다.




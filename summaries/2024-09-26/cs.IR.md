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




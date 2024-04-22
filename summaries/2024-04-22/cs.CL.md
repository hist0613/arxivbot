### Sample Design Engineering: An Empirical Study of What Makes Good  Downstream Fine-Tuning Samples for LLMs (https://arxiv.org/abs/2404.13033)
Comments: 23 pages, 12 figures, 14 tables

- **What's New**: 새롭게 도입된 Sample Design Engineering (SDE) 방법론은 Large Language Models (LLMs)를 위한 세부 튜닝에 있어 입력(input), 출력(output), 그리고 사고 과정(reasoning)의 설계를 개선함으로써 모델의 성능을 향상시키는 데 중점을 둔다. 이 연구는 Multi-Aspect Sentiment Analysis (MASA), Event Extraction, Nested Entity Recognition과 같은 복잡한 하류 작업(downstream tasks)에 대해 다양한 SDE 옵션들의 효과를 평가하고, 최적의 전략을 개발함으로써 폭넓은 LLM 사용 환경에 적용 가능성을 탐구한다.

- **Technical Details**: 이 논문은 입력, 출력 및 추론 디자인 옵션을 세 가지 주요 그룹으로 분류하고, 이를 통해 훈련 샘플의 구조가 LLM의 세부 튜닝 성능에 미치는 영향을 조사한다. 주요 실험은 ID (in-domain) 및 OOD (out-of-domain) 환경에서 실행되었으며, 이를 통해 다양한 SDE 옵션이 성능에 미치는 영향을 광범위하게 분석한다. 또한, 본 연구에서는 LoRA (Low-Rank Adaptation) 기법을 사용하여 여러 가지 parameter-efficient fine-tuning (PEFT) 방법 중 하나로 채택하였다. 이를 통해, 내장된 prompt/output perplexity, zero-shot 및 ICL (in-context learning) 능력을 분석하고 PE (Prompt Engineering) 전략이 SDE 전략으로 직접적으로 번역되지 않을 수도 있다는 결과를 도출하였다.

- **Performance Highlights**: 연구 결과, 예를 들어, 작업 지시의 위치를 변경하는 것만으로도 성능에 유의미한 차이를 가져오고, 언급되지 않은 대상에 대한 placeholder를 추가하면 성능이 크게 향상되는 등의 흥미로운 결과를 발견했다. 특히 Integrated SDE strategy (ES-SDE)는 다양한 상황에서 기존의 휴리스틱(huristic) 샘플 디자인을 넘어서는 일관된 우수성을 보여주며, ES-SDE는 다양한 훈련 크기, 디코딩 무작위성(decoding randomness), 지시 변화(instruction variation)에 걸쳐 안정적인 효과를 확인할 수 있었다.



### Stronger Random Baselines for In-Context Learning (https://arxiv.org/abs/2404.13020)
- **What's New**: 새로운 '최대 랜덤 베이스라인(maximum random baseline)'이 소개되었습니다. 이는 언어 모델의 인-컨텍스트(in-context) 학습 분류 성능을 평가할 때 직면한 난제들을 해결하기 위해 고안되었습니다. 기존에는 표준 랜덤 베이스라인(standard random baseline)만 사용되었지만, 새로운 방식은 재사용되는 검증 세트(validation set)와 작은 데이터 셋의 문제를 고려해 보다 강력한 방법을 제안합니다.

- **Technical Details**: 제안된 '최대 랜덤 베이스라인'은 다수의 랜덤 분류자(random classifiers)간 예상된 최대 정확도(maximum accuracy)를 기반으로 합니다. 이 방식은 견고한 베이스라인을 제공함으로써, 일부 few-shot 결과들이 표준 베이스라인을 넘어서도 실제로는 이 강력한 랜덤 베이스라인을 넘지 못하는 경우가 20% 이상임을 밝혀내고 있습니다. 또한, 홀드-아웃 테스트 세트(held-out test set)가 사용 가능할 경우, 이 더 강력한 베이스라인이 홀드-아웃 성능의 더 좋은 예측자가 되어 불필요한 테스트 세트 평가를 피할 수 있습니다.

- **Performance Highlights**: '최대 랜덤 베이스라인'은 16개의 BIG-bench Lite 작업들에 대해 적용된 여섯 가지 양자화된(language models) 언어 모델들을 사용한 프롬프트 데모 선택에서 표준 베이스라인보다 우수한 예측 성능을 보였습니다. 이는 더 높은 예측 정확도를 제공하며, 실제 언어 모델의 성능을 정확하게 평가할 수 있는 방법을 제공합니다.



### Towards Reliable Latent Knowledge Estimation in LLMs: In-Context  Learning vs. Prompting Based Factual Knowledge Extraction (https://arxiv.org/abs/2404.12957)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)에서 내포된 지식을 평가하는 새로운 접근법을 제안합니다. 이 방법은 이전에 제시되었던 프롬프트 기반 방법의 신뢰성 문제를 피하면서, 인컨텍스트 학습(In-Context Learning, ICL) 능력을 활용하여 LLM이 얼마나 많은 사실을 알고 있는지를 추정합니다. 이러한 접근은 개념적으로 단순하며 적용하기도 쉬워, LLM 내부에 내포된 지식을 더 많이 밝힐 수 있었습니다.

- **Technical Details**: 저자들은 인컨텍스트 학습 기반 지식 추정기(IC-LKE)를 소개하고, 다양한 디자인 선택이 지식 추정 성능에 미치는 영향을 조사합니다. IC-LKE는 다양한 크기와 종류의 오픈소스 LLMs를 대상으로 실험을 진행해, 모델에 따라 알고 있는 사실의 정도가 다르다는 것을 발견하였습니다. 특히, Llama(2), Gemma, Mistral 등의 모델 제품군은 다른 제품군보다 더 많은 사실을 알고 있는 경향이 있었습니다.

- **Performance Highlights**: IC-LKE는 기존의 인간 생성 프롬프트나 기계 추출 프롬프트를 사용한 지식 추정 방법들과 비교하여 우수한 성능을 보였습니다. 특히, 이전 방법들은 관계 및 LLM 특화 디자인이었으나, IC-LKE는 관계에 독립적이며 다양한 LLM에 적용 가능한 범용성을 갖추고 있습니다. 49개의 오픈소스 LLM을 사용하여 50개의 관계와 20,000개의 사실에 대한 평가를 수행한 결과, 모델별, 제품군별로 알고 있는 사실의 차이가 명확히 드러났으며, 학습된 모델들이 제공하는 사실 지식의 양에 있어서 미세조정이 감소하는 경향을 보였습니다.



### MAiDE-up: Multilingual Deception Detection of GPT-generated Hotel  Reviews (https://arxiv.org/abs/2404.12938)
- **What's New**: 이 논문에서는 LLMs(Large Language Models)의 성능 향상과 일반적인 사용 증가로 인해 속임수 리뷰가 점점 흔해지고 있음을 다루고 있습니다. 기존의 연구들이 진실된 리뷰와 속임수적 인간 리뷰를 구별하는 모델 개발에 집중해 왔지만, 실제 리뷰와 AI가 작성한 가짜 리뷰를 구별하는 것에 대한 정보는 많지 않습니다. 또한, 대부분의 연구가 영어에 초점을 맞추고 있으며, 다른 언어에 대한 연구는 매우 제한적입니다. 본 논문에서는 10,000개의 실제 호텔 리뷰와 10,000개의 AI 생성된 가짜 호텔 리뷰로 구성된, 열 가지 언어로 균형 잡힌 MAiDE-up 데이터셋을 소개하고 공개합니다.

- **Technical Details**: MAiDE-up 데이터셋을 활용하여, AI 가짜 호텔 리뷰와 실제 호텔 리뷰를 비교하고, 속임수 탐지 모델 성능에 영향을 미치는 요인을 식별하는 광범위한 언어학적 분석을 수행했습니다. 연구팀은 감정(sentiment), 위치(location), 언어(language)의 세 가지 주요 차원에서 호텔 리뷰에 대한 속임수 탐지를 위한 여러 모델의 효과를 탐구했습니다.

- **Performance Highlights**: 분석 결과, 이 세 가지 차원이 AI 생성된 가짜 리뷰를 탐지하는 데 얼마나 잘 작동하는지에 큰 영향을 미친다는 것을 발견했습니다. 각기 다른 차원은 모델이 가짜 리뷰를 식별하는 능력에 상이한 영향을 미칩니다.



### Cross-cultural Inspiration Detection and Analysis in Real and  LLM-generated Social Media Data (https://arxiv.org/abs/2404.12933)
- **What's New**: 이 연구는 인도와 영국의 영감을 주는 콘텐츠를 비교하고 AI(인공 지능)가 생성한 영감을 주는 게시물과 실제 영감을 주는 게시물을 비교하는 첫 번째 연구입니다. 'InspAIred' 데이터셋은 인도와 영국에서 AI로 생성된 콘텐츠와 실제 콘텐츠 모두를 포함하고 있으며, 이를 통해 문화적 영감의 차이를 분석합니다.

- **Technical Details**: 연구자들은 Reddit에서 수집한 2,000개의 실제 영감을 주는 게시물과 2,000개의 비영감 게시물, 그리고 GPT-4 모델을 사용하여 생성한 2,000개의 영감을 주는 게시물을 포함하는 InspAIred 데이터셋을 개발했습니다. 데이터는 인도와 영국에서 균등하게 수집되었습니다. 연구팀은 이 데이터를 사용하여 계산 언어학 분석(computational linguistic analyses)을 수행하고, 문화 간(content across cultures), 데이터 출처(data sources) 간 콘텐츠를 비교하는 동시에 기계 학습 모델을 활용해 영감을 감지하는 효율성을 평가했습니다.

- **Performance Highlights**: 이 연구는 LLM(대규모 언어 모델)을 사용하여 인도와 영국의 영감과 관련된 문화적 지식을 탐구했으며, 기계 학습 techniques을 이용하여 각 문화에서 실제 사용자가 작성한 Reddit 게시물과 AI가 생성한 게시물을 비교했습니다. 초기 결과는 AI가 실제 영감을 주는 게시물과 유사한 수준의 영감을 제공할 수 있음을 보여주며, 이는 AI 콘텐츠 생성의 새로운 가능성을 열어줍니다.



### Enabling Natural Zero-Shot Prompting on Encoder Models via  Statement-Tuning (https://arxiv.org/abs/2404.12897)
- **What's New**: 이 논문에서는 인코더만을 사용하는 언어 모델의 성능을 개선하기 위해 새로운 기법인 'Statement-Tuning'을 제안합니다. 이 방법은 자연어 문장으로 NLU (Natural Language Understanding) 작업을 명시하고, 이를 바탕으로 진리값 (True or False)을 분류하여 다양한 태스크에서의 일반화 능력을 향상시키는 것을 목표로 합니다. Statement-Tuning은 특히 소규모 모델에서도 대규모 언어 모델(LLMs)과 경쟁할 수 있는 성능을 보여준다는 점에서 주목할 만한 발전입니다.

- **Technical Details**: Statement-Tuning은 인코더 모델을 이용하여, 주어진 문장의 진리값을 분류하도록 학습하는 방식입니다. 이 기법은 다양한 NLU 태스크를 자연어 문장으로 전환하고, 이를 통해 학습을 진행합니다. 사용된 인코더 모델인 RoBERTa는 다양한 다운스트림 데이터와 문장 템플릿을 사용하여 학습되어, 새로운 태스크에 대해 zero-shot 및 few-shot 성능을 보여줍니다. 연구는 16,000개의 학습 문장 예제를 통해 충분한 수의 문장과 문장 템플릿, 그리고 태스크 다양성이 모델의 일반화 능력에 긍정적 영향을 준다는 것을 밝혀냈습니다.

- **Performance Highlights**: 실험 결과, Statement-Tuning은 상당히 적은 파라미터를 사용하면서도 대규모 언어 모델(LLMs)과 경쟁할 수 있는 성능을 달성하였습니다. 특히, zero-shot 및 few-shot 시나리오에서 뛰어난 일반화 능력을 보여주며, 몇 가지 태스크에서는 소수의 데이터셋 (예: 1000 문장) 만으로도 충분한 성능을 나타내었습니다. 또한, 태스크와 문장의 다양성이 성능 향상에 중요한 역할을 하는 것으로 나타났습니다.



### Unlocking Multi-View Insights in Knowledge-Dense Retrieval-Augmented  Generation (https://arxiv.org/abs/2404.12879)
- **What's New**: 이 연구에서는 지식이 풍부한 도메인(법률 및 의학 분야)에서의 정보 검색 정확도를 향상시키기 위해 다중 관점을 고려한 새로운 RAG (Retrieval-Augmented Generation) 프레임워크인 MVRAG를 소개합니다. 이 프레임워크는 다중 도메인 관점에서 의도를 인식하고 쿼리를 재작성하여 높은 정밀도의 검색을 가능하게 합니다. 이는 향상된 해석 가능성(interpretability)과 신뢰성(reliability)으로 이어지며 LLM (Large Language Models)의 적용 범위를 확장할 잠재력을 가집니다.

- **Technical Details**: MVRAG 프레임워크는 세 가지 주요 단계로 구성됩니다: 의도 인식(Intention Recognition), 쿼리 재작성(Query Rewriting), 및 검색 증강(Retrieval Augmentation). 의도 인식 단계에서 LLM은 쿼리의 의도를 식별하고 각 전문적 관점에 따라 중요도를 평가합니다. 쿼리 재작성에서는 각 관점에 맞춰 수정된 쿼리로 문서를 검색하고, 검색 증강 단계에서는 검색된 문서를 재순위하고 통합하여 최종 응답을 생성합니다. 이러한 방식은 전통적인 텍스트 기반 유사성을 넘어서 전문적 쿼리의 다중 관점을 충실히 반영하는 데 중점을 둡니다.

- **Performance Highlights**: 실험을 통해 법률 및 의료 사례 검색에서 MVRAG가 회상률(recall) 및 정밀도(precision)에서 유의미한 개선을 보였습니다. 다중 관점 검색 접근 방식은 RAG 작업에서 다차원적인 정보를 활용하여 LLMs의 적용을 가속화하는 데 기여합니다. 이는 지식-밀집 도메인에서의 응답 생성의 정확성과 관련성을 크게 향상시키는 결과를 가져왔습니다.



### How Does the Textual Information Affect the Retrieval of Multimodal  In-Context Learning? (https://arxiv.org/abs/2404.12866)
- **What's New**: 다양한 언어 이해 작업에서 단일 모델을 사용할 수 있는 멀티모달 인코텍스트 학습(M-ICL)의 새로운 전략으로, MSIER(Multimodal Supervised In-context Examples Retrieval)이 등장했습니다. 이 연구는 텍스트 정보의 중요성과 멀티모달 학습 환경에서 예시 선택에 대한 미묘한 영향을 평가하고, 향상된 성능을 위해 모달리티(modality)를 고려한 예시 선택 방법을 제안합니다.

- **Technical Details**: MSIER 프레임워크는 네트워크를 사용하여 예시를 선택하고 멀티모달 인코텍스트 학습의 효율을 향상시키기 위해 설계되었습니다. 이러한 접근 방식은 이미지 캡션(image captioning), 시각적 질문 대답(visual question answering), 순위 분류(rank classification) 등 세 가지 작업에서 검증되었습니다. 주어진 테스트 예(instance)에 대해 학습된 예(memory) 중 유사도(similarity metric)에 따라 적합한 예를 선택하는 방식이 구현되어 있습니다.

- **Performance Highlights**: MSIER은 이미지와 텍스트를 모두 고려하여 인코텍스트 예시를 선택함으로써 면밀하게 예시 선택 성능을 높였습니다. 실험 결과, 이 방법은 상위의 성능을 달성하며, 이는 멀티모달 태스크에서 인코텍스트 학습의 효율성을 크게 향상시킵니다. 또한, 다양한 모달리티가 좋은 인코텍스트 예를 선택하는 데 중요한 역할을 한다는 귀중한 통찰을 제공합니다.



### TartuNLP @ SIGTYP 2024 Shared Task: Adapting XLM-RoBERTa for Ancient and  Historical Languages (https://arxiv.org/abs/2404.12845)
Comments: 11 pages, 3 figures

- **What's New**: SIGTYP 2024 공유 작업에 제출된 연구에서는 고대 및 역사적 언어에 대한 Word Embedding 평가를 위해 어댑터(Adapter) 기반의 접근 방식을 사용하여 단순하고 통일된 방법을 개발하였습니다. 이 방법은 파라미터 효율적인 미세 조정(Parameter-Efficient Fine-Tuning)을 기반으로 하며, 다양한 언어와 태스크에 일관되게 적용되었습니다.

- **Technical Details**: 연구팀은 언어별 및 태스크별 어댑터를 스택 형태로 미세 조정(Fine-Tuning)하였으며, XLM-RoBERTa를 사용하여 다양한 과제에 대응하였습니다. 어댑터 기반 접근법은 언어 모델을 새로운 목표 언어에 맞게 조정하는데 있어 매우 효과적이었으며, 모델의 대규모 파라미터를 변경할 필요 없이 태스크 특화 학습이 가능하도록 지원했습니다.

- **Performance Highlights**: 이 접근법을 사용하여 16개 언어에 대한 형태학적 주석, 품사 태깅, 표제어 추출, 문자 및 단어 수준의 결핍 채우기 등의 다양한 NLP 태스크를 수행했습니다. 결과적으로 SIGTYP 2024 공유 작업에서 전체 3개 제출 중 2위를 차지하였으며, 단어 수준에서의 갭 채우기(Gap-Filling) 과제에서는 1위를 달성하였습니다.



### LiMe: a Latin Corpus of Late Medieval Criminal Sentences (https://arxiv.org/abs/2404.12829)
Comments: to be published in: LT4HALA@LREC-COLING 2024

- **What's New**: 이 논문에서는 라틴어 법률 문서의 새로운 데이터셋인 LiMe 데이터셋을 소개합니다. 이 데이터셋은 중세 시대 밀라노의 법정에서 기록된 325개의 법률 문서로 구성되어 있으며, 이는 고급 언어 모델을 위한 훈련 자료로 사용될 수 있습니다. 특히, 이 데이터셋은 전문가들에 의해 자세히 주석(annotation)이 달린 채로 제공되어, 마스크(masked) 언어 모델 및 지도학습(supervised learning) 작업에 사용될 수 있습니다.

- **Technical Details**: LiMe 데이터셋은 Libri sententiarum potestatis Mediolani라는 중세 시대 문서집에서 추출되었으며, 이 문서들은 법률적, 역사적 분석뿐만 아니라 언어 모델링에까지 활용될 수 있는 풍부한 정보를 담고 있습니다. 문서는 디지털화 과정을 거쳐 체계적인 정보 추출 및 엔티티(entity), 관계(relation) 태깅 등의 상세한 주석 작업을 완료하였습니다. 또한, 이 데이터셋은 통계학적 및 기계학습(machine learning) 방법론을 적용하는 데에 있어 유용한 예시들을 제공합니다.

- **Performance Highlights**: 이 연구를 통해 제공되는 LiMe 데이터셋은 기존의 라틴어 데이터 자원에 비해 향상된 양질의 주석 정보를 제공함으로써, 라틴어 텍스트에 대한 벡터 표현(vector representations)을 생성하는 언어 모델의 성능 개선에 기여할 가능성이 높습니다. LiMe 데이터셋을 사용한 초기 실험에서는 주석된 정보를 활용하여 더 정밀한 언어 처리 및 분석이 가능함을 보여주었습니다.



### CT-ADE: An Evaluation Benchmark for Adverse Drug Event Prediction from  Clinical Trial Results (https://arxiv.org/abs/2404.12827)
- **What's New**: 새롭게 도입된 CT-ADE 데이터셋은 임상 시험 결과에서 추출한 12,000개 이상의 인스턴스를 포함하여 ADE(Adverse Drug Events, 부작용 건강사고)의 예측 모델링을 강화하기 위해 개발되었습니다. 이 데이터셋은 약물, 환자 집단 및 맥락 정보를 통합하여 단일약물치료(monopharmacy treatments)에서 다중 라벨 ADE 분류 작업을 위한 포괄적인 자원을 제공합니다.

- **Technical Details**: CT-ADE는 높은 수준의 정제를 거쳐 MedDRA (Medical Dictionary for Regulatory Activities) 온톨로지의 시스템 장기 클래스(system organ class) 수준에서 표준화된 주석을 제공합니다. 이를 통해 ADE의 복잡한 특성을 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: 기본 모델을 사용한 초기 분석에서 CT-ADE는 73.33%의 F1 점수(F1 score)와 81.54%의 균형 정확도(balanced accuracy)를 달성하여 ADE 예측에서의 높은 잠재력을 보여주었습니다. 이는 향후 약물 개발과 환자의 안전을 향상시키는 데 중요한 도구가 될 것입니다.



### REXEL: An End-to-end Model for Document-Level Relation Extraction and  Entity Linking (https://arxiv.org/abs/2404.12788)
Comments: Accepted at NAACL Industry Track 2024

- **What's New**: REXEL이라는 새로운 E2E (end-to-end) 모델을 소개하여, 문서 수준의 cIE (closed information extraction) 작업을 대폭 향상 시켰습니다. 이 모델은 한 번의 전달(forward pass)로 멘션 감지(mention detection, MD), 엔터티 타이핑(entity typing, ET), 엔티티 식별(entity disambiguation, ED), 코어퍼런스 해결(coreference resolution, Coref) 및 문서 수준 관계 분류(document-level relation classification, RC)를 수행합니다.

- **Technical Details**: REXEL은 중간 임베딩 표현을 통해 5가지 하위 작업을 통합하며, 이는 각 작업이 서로의 정보를 활용하여 정확도를 크게 향상시킵니다. 이 모듈식 아키텍처는 단일 문서에 대해 단일 전달로 사실을 추출할 수 있게 하여, 웹 규모에서의 효율적인 배치를 가능하게 합니다.

- **Performance Highlights**: REXEL은 유사한 설정에서 경쟁 모델보다 평균 11배 빠른 속도로 실행되며, 각 개별 하위 작업과 다양한 공동 작업 조합에 최적화 되었을 때 경쟁 작업보다 평균 6 F1 포인트 이상 우수한 성능을 보여줍니다. 또한, REXEL은 E2E 관계 추출(RE) 설정에서 기존 기준을 평균 6 F1 포인트 이상 개선합니다.



### AutoCrawler: A Progressive Understanding Web Agent for Web Crawler  Generation (https://arxiv.org/abs/2404.12753)
Comments: 18 pages, 5 figures

- **What's New**: 이 연구에서는 LLMs와 크롤러를 결합하는 새로운 패러다임을 소개하고, 수직 정보 웹 페이지를 대상으로 한 크롤러 생성 작업을 제안합니다. 이는 다양하고 변화하는 웹 환경을 보다 효율적으로 처리할 수 있는 능력을 크롤러에 제공합니다.

- **Technical Details**: AutoCrawler는 HTML의 계층 구조를 활용하여 점진적으로 이해할 수 있는 두 단계 프레임워크를 제안합니다. 이 프레임워크는 top-down 및 step-back 작업을 통해 잘못된 작업에서 학습하고 HTML을 지속적으로 가지치기하여 더 나은 작업 생성을 도모합니다. 실험 결과는 다양한 LLMs를 사용하여 프레임워크의 효과를 입증합니다.

- **Performance Highlights**: AutoCrawler는 구조화된 웹 페이지 추출 작업에 사용되며, 기존의 IE 작업 평가를 실행 가능한 평가로 변환함으로써 크롤러 생성 작업의 효율성을 측정합니다. 이 평가는 정확도(precision), 재현율(recall), F1 점수를 바탕으로 실행 경로의 성공 비율을 더 중요하게 봅니다.



### Beyond Human Norms: Unveiling Unique Values of Large Language Models  through Interdisciplinary Approaches (https://arxiv.org/abs/2404.12744)
Comments: 16 pages, work in progress

- **What's New**: 이 연구는 LLM(Large Language Models)의 고유한 가치 시스템(Value System)을 탐구하고 재구성하는 새로운 프레임워크, ValueLex를 제안합니다. 기존 연구들이 인간 중심의 가치 시스템에 의존하는 것과 달리, ValueLex는 LLM 고유의 가치를 발견하고 평가하기 위해 설계되었습니다. 이를 통해 LLM의 다양한 모델 크기, 훈련 방법, 데이터 소스에 걸쳐 가치 성향을 측정합니다.

- **Technical Details**: ValueLex는 Lexical Hypothesis를 기반으로 하여 LLM에서 가치에 대한 표현을 추출하고, 이를 요인 분석(Factor Analysis) 및 의미론적 클러스터링(Semantic Clustering)을 통해 정제합니다. 이 과정을 통해 '능력(Competence)', '성격(Character)', '진실성(Integrity)'의 세 가지 주요 가치 차원을 식별하였으며, 각 차원은 특정한 하위 차원을 가집니다. 또한, 연구는 30개 이상의 LLM에 대해 프로젝티브 테스트(Projective Tests)를 개발하여 가치 성향을 평가했습니다.

- **Performance Highlights**: 이 프레임워크를 통해 LLM이 가장 높게 평가하는 가치는 '능력'으로 나타났으며, 훈련 방법에 따라 가치의 우선순위에 차이가 있음을 발견했습니다. 예를 들어, 바닐라(Vanilla) 사전 훈련된 모델은 뚜렷한 가치 성향을 보이지 않았지만, 지시-튜닝(Instruction-tuning)은 차원 간의 일관성을 향상시키고, 정렬(Alignment)은 가치의 다양성을 증가시켰습니다. 또한, 더 큰 모델은 '능력'을 선호하는 경향이 크며, 다른 차원에 비해 약간의 손실을 보였습니다.



### Relevant or Random: Can LLMs Truly Perform Analogical Reasoning? (https://arxiv.org/abs/2404.12728)
- **What's New**: 이 연구는 Large Language Models(LLMs)이 문제 해결을 위해 유사 사례를 자체 생성하는 방법의 효과를 탐구합니다. 특히, 연구자들은 유의미한 예시(self-generated relevant examples)와 무작위 예시(random examples)가 LLMs의 문제 해결 능력에 미치는 영향을 비교하고 있습니다. 놀랍게도, 무작위 예시가 유의미한 예시와 비교하여 동등하거나 더 나은 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: LLMs의 아날로지 추론(analogical reasoning) 능력을 평가하기 위해 GSM8K, MATH, BIG-Bench Hard 등 다양한 추론 과제에서 실험을 진행하였습니다. 특히, GPT-3.5(turbo) 및 Llama 2-Chat 모델을 사용하여 실험하였으며, 문제에 대한 유의미한 예시와 무작위 예시 모두를 생성하도록 요청하고 그 결과를 비교 분석하였습니다. 실험 결과, 자체 생성된 예시의 정확도가 LLMs의 성능에 중대한 영향을 미치는 주요 요소임을 확인하였습니다.

- **Performance Highlights**: GSM8K에서는 무작위 생물학적 예시를 사용했을 때 4%의 성능 향상을 보였습니다. 이러한 결과는 자체 생성된 예시의 유의미함보다는 정확성이 더 중요할 수 있음을 시사합니다. 또한, 연구진은 추론 비용을 크게 줄인 두 개의 새로운 방법을 개발했으며, 이 방법들이 기존 방법보다 우수한 성능을 보였습니다.



### Evaluating Character Understanding of Large Language Models via  Character Profiling from Fictional Works (https://arxiv.org/abs/2404.12726)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large language models, LLMs)이 허구적인 캐릭터들을 이해하고 상호작용하는 능력을 평가하는 새로운 방법을 제안합니다. 특히, 작품 속 캐릭터 프로필을 요약하는 '캐릭터 프로파일링' 작업을 통해 이를 평가하며, 이는 역할 연기 에이전트(Role-Playing Agents, RPAs) 개발에 널리 사용되고 있지만 충분히 연구된 바가 없습니다.

- **Technical Details**: 연구팀은 문학 전문가들로부터 구성된 CroSS 데이터셋을 구축하고, 이를 사용하여 다양한 요약 방법과 LLMs를 통해 생성된 캐릭터 프로파일을 평가하였습니다. 평가는 'ground truth' 참조문서와의 비교뿐만 아니라, 다운스트림 태스크(Downstream tasks)에서의 적용 가능성을 통해서도 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 여러 요약 방법들과 LLMs를 사용했을 때 약속된 결과를 얻었으며, 이는 LLMs의 캐릭터 이해 능력을 강하게 입증합니다. 이러한 결과는 LLMs가 허구적 캐릭터의 성격과 특성을 효과적으로 파악하고 재현할 수 있음을 보여줍니다.



### Enabling Ensemble Learning for Heterogeneous Large Language Models with  Deep Parallel Collaboration (https://arxiv.org/abs/2404.12715)
Comments: 12 pages, 5 figures

- **What's New**: 이 논문에서는 대규모 언어 모델 (LLMs)의 앙상블을 통해 예측 퓨전(prediction fusion)을 구현하는 새로운 프레임워크, DEEPEN을 제안하고 있습니다. DEEPEN은 모든 모델의 확률 분포를 평균화하여 최종 토큰을 결정하는 훈련이 필요 없는 앙상블 방법론입니다. 주요 챌린지인 다양한 LLMs 간의 어휘(vocabulary) 불일치는 상대적 표현(relative representation) 이론을 기반으로 해결합니다.

- **Technical Details**: DEEPEN은 각 모델의 확률 분포를 일반화된 공간으로 매핑한 후 이를 집계하여, 최종적으로 한 모델의 확률 공간으로 다시 매핑하는 검색 기반 역변환(search-based inverse transformation)을 수행합니다. 이를 통해 LLMs 사이의 협력을 개선하며, 텍스트 응답이 아닌 내부 표현(probability distributions)을 통신 수단으로 활용합니다.

- **Performance Highlights**: DEEPEN은 6B에서 70B에 이르는 다양한 LLMs를 포함한 앙상블에서 실험되었으며, 학습 시험, 추론, 지식 질의응답(Knowledge-QA) 등 여섯 가지 벤치마크에서 일관된 성능 향상을 보여주었습니다. 이는 DEEPEN이 상이한 언어 모델의 결합을 통해 텍스트 생성 작업에서의 효과를 입증합니다.



### Neural Semantic Parsing with Extremely Rich Symbolic Meaning  Representations (https://arxiv.org/abs/2404.12698)
Comments: This manuscript has been submitted to Computational Linguistics journal on 2024-03-15

- **What's New**: 본 논문에서는 기존의 신경 의미 분석기(neural semantic parsers)가 직면한 문제점을 해결하기 위해 새로운 '분류학적(taxonomical)' 의미 표현 방식을 제안합니다. 이 방식은 어휘 온톨로지(lexical ontology)의 계층적 구조를 활용하여, 개념을 그들이 속한 분류학적 계층에 따라 표현합니다. 이는 의미 정보를 풍부하게 하고 해석 가능성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 전통적인 신경 의미 분석기와 새로 도입된 분류학적 의미 분석기를 비교 분석하였습니다. 분류학적 모델은 어휘 밖의 개념(out-of-vocabulary concepts)에 대해 더 우수한 성능을 보였지만, 표준 평가 지표(standard metrics)를 사용할 때는 전통 모델에 비해 다소 낮은 성능을 보였습니다. 이는 분류학적인 표현 방식이 의미 분석에서 데이터 기반의 분포적 의미(data-driven distributional meanings)와 지식 기반의 심볼릭 표현(knowledge-based symbolic representations)을 효과적으로 결합할 수 있는 가능성을 시사합니다.

- **Performance Highlights**: 새로운 '분류학적' 의미 분석 방법은 어휘 밖 개념을 처리하는 능력에서 탁월함을 보였습니다. 그러나 표준 평가 방식에 따른 성능은 전통적인 방식에 약간 못 미치는 결과를 나타냈습니다. 이는 새로운 시스템이 특정 조건 하에서 기존 방법보다 우수할 수 있음을 보여주면서도, 일반적인 평가에서는 여전히 개선이 필요함을 지적합니다.



### SOS-1K: A Fine-grained Suicide Risk Classification Dataset for Chinese  Social Media Analysis (https://arxiv.org/abs/2404.12659)
- **What's New**: 이 연구는 중국 소셜 미디어에서 자살 위험도를 탐지하기위해 개발된 새로운 데이터셋(SOS-1K)을 소개합니다. 이 데이터셋은 자살의도, 자살방법, 시급성 등을 포함하는 세밀한 자살 위험 분류에 초점을 맞추고 있습니다. 특히, 이 연구는 BERT와 같은 사전 학습된 모델들을 사용하여 높은 및 낮은 자살 위험을 구분하는 데 유용함을 보여줍니다. 또한, 데이터 불균형 문제를 해결하기 위한 전통적 및 LLM 기반 데이터 증강 기법도 탐구되었습니다.

- **Technical Details**: 이 연구에서는 Transformer의 인코더와 양방향 처리를 사용하는 BERT 모델을 포함하여 여러 사전 훈련된 모델들을 평가하였습니다. BERT는 언어의 복잡한 뉘앙스를 이해하기 위해 Masked Language Model(MLM)과 Next Sentence Prediction(NSP) 작업을 수행합니다. RoBERTa, ELECTRA, MacBERT, NeZha, ERNIE 3.0 및 중국어를 위한 MentalBERT와 같은 다양한 변형 모델들도 사용되었습니다. 데이터 증강을 위해 동의어 교체와 왕복 번역과 같은 기법이 사용되었으며, 이러한 방법들은 모델이 더 잘 일반화하도록 도왔습니다.

- **Performance Highlights**: 실험 결과, 최고의 모델은 높은 자살 위험 감별에서 88.39%의 F1 점수를 달성했습니다. 그러나 세밀하게 분류된 자살 위험도에서는 여전히 만족스럽지 못한 50.89%의 가중치 F1 점수를 나타냈습니다. 데이터 증강은 F1 점수를 최대 4.65% 포인트 향상시키는데 도움이 되었습니다. 특히, 중국어 정신 건강 분야에 적합한 MentalBERT 모델은 두 가지 모든 분류 작업에서 우수한 성능을 보였습니다.



### Cooperative Sentiment Agents for Multimodal Sentiment Analysis (https://arxiv.org/abs/2404.12642)
- **What's New**: 이 논문에서는 각각의 모달리티(sessing modality)로부터 동적 감정 변화(dynamic sentiment variations)를 강조하는 감정 에이전트(Sentiment Agents)를 이용한 새로운 다중 모달 표현 학습(Multimodal Representation Learning, MRL) 방법을 제안합니다. 해당 방법은 감정 에이전트 설립(Sentiment Agents Establishment, SAE) 단계와 감정 에이전트 협력(Sentiment Agents Cooperation, SAC) 단계로 이루어져 있으며, 독립적인 정책 모델(policy model)을 사용하여 각 모달리티 내의 중요 속성을 파악하고 최적화합니다.

- **Technical Details**: 제안된 Co-SA(Cooperative Sentiment Agents)는 모달리티-감정 연계 해제(Modality-Sentiment Disentanglement, MSD) 모듈과 심층 위상 공간 재구성(Deep Phase Space Reconstruction, DPSR) 모듈을 포함합니다. MSD는 원시 입력으로부터 감정 특성(sentiment features)을 분리하고, DPSR은 단시간 및 장시간 관찰 사이의 관계를 설정하고 시간에 따른 감정 변화를 강조하는 데 사용됩니다. 상호 보상 메커니즘(mutual rewarding mechanism)을 통해 개별 에이전트들은 최적화된 다중 모달 표현 학습 전략에 도달하게 됩니다.

- **Performance Highlights**: Co-SA는 다중 모달 감정 분석(Multimodal Sentiment Analysis, MSA)과 다중 모달 감정 인식(Multimodal Emotion Recognition, MER) 작업에 적용되었습니다. 실험 결과 Co-SA는 다양한 교차 모달 특성(cross-modal features)을 탐색하는 데 뛰어난 성능을 보였으며, 이는 공통 및 보완적인 측면(both common and complementary aspects)을 포함합니다.



### Efficient infusion of self-supervised representations in Automatic  Speech Recognition (https://arxiv.org/abs/2404.12628)
Comments: Accepted to ENLSP workshop, NeurIPS 2023

- **What's New**: 이 연구에서는 사전 훈련된 자기지도 학습(SSL) 모델에서 추출된 정보를 효율적으로 자동 음성 인식(ASR: Automatic Speech Recognition) 아키텍처에 통합하기 위한 두 가지 새로운 접근 방식을 제안합니다. 첫 번째 방법은 '프레임별 추가(framewise addition)'이며, 두 번째 방법은 '크로스-어텐션 메커니즘(cross-attention mechanism)'을 활용하는 것입니다. 두 방식 모두 표준 인코더-디코더 형태의 컨포머(conformer) 시스템과 비슷한 크기를 유지하면서도, 훈련 단계에서 SSL 모델을 사용하지 않으므로 훈련 속도가 향상됩니다.

- **Technical Details**: 제안된 방식은 기존의 결합 CTC-어텐션(CTC-Attention) 프레임워크를 수정하여 SSL 모델에서 추출된 오디오 표현을 ASR 시스템에 쉽게 통합할 수 있도록 합니다. 이를 위해 인코더 부분에서만 변경을 가하며, 이 변경을 통해 fbank 생성기에서 생성된 표현과 SSL 모델에서 생성된 표현을 모두 처리할 수 있습니다. 이렇게 생성된 오디오 표현은 인코더에 의해 컨텍스트화되어 디코더에서 최종 텍스트를 예측하는 데 사용됩니다. 또한, 인코딩 과정에서의 차원 축소와 레이어 정규화(llayer normalization)를 통해 모델의 일반화 성능을 개선합니다.

- **Performance Highlights**: Librispeech 및 Tedlium 데이터셋에 대한 실험에서 제안하는 방법은 기존의 기준 모델 대비 뛰어난 성능 향상을 보여줍니다. 특히, 훈련 속도가 기존 모델 대비 현저히 개선되었으며, 이는 대규모 SSL 모델을 훈련 단계에서 사용하지 않기 때문입니다. 이와 함께 진행된 상세한 분석 및 탈락 연구(ablation study)는 이러한 접근 방식의 유효성을 더욱 입증합니다.



### CORI: CJKV Benchmark with Romanization Integration -- A step towards  Cross-lingual Transfer Beyond Textual Scripts (https://arxiv.org/abs/2404.12618)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 연구에서는 교차 언어 전이(Cross-lingual transfer)가 소스 언어 선택에 어떤 영향을 받는지를 탐구함으로써, 영어가 아닌 밀접하게 관련된 언어를 소스로 사용할 때의 중요성을 강조합니다. 또한, 중국어, 일본어, 한국어, 베트남어로 구성된 밀접한 접촉 언어 그룹인 CJKV에 대한 새로운 벤치마크 데이터셋을 개발하여 언어 접촉에 대한 심층 연구를 장려합니다.

- **Technical Details**: 연구는 교차 학습 목표(Contrastive Learning objectives)를 통해 로마자 표기법을 텍스트 스크립트 너머로 통합함으로써, 이러한 언어들 간의 접촉을 포괄적으로 포착하고 교차 언어 표현(Cross-lingual representations)을 향상시키는 방법을 제안합니다. 이를 통해 효과적인 제로샷 교차 언어 전이(Zero-shot cross-lingual transfer)를 달성하였습니다.

- **Performance Highlights**: 제안된 기법은 CJKV 언어 군에 대한 교차 언어 전이 성능을 향상시켰으며, 특히 로마자 표기법을 사용하는 것이 언어 간의 더 나은 표현 학습에 기여함을 보여줍니다.



### Parameter Efficient Diverse Paraphrase Generation Using Sequence-Level  Knowledge Distillation (https://arxiv.org/abs/2404.12596)
Comments: Published in: 2024 5th International Conference on Advancements in Computational Sciences (ICACS) with IEEE

- **What's New**: 이 연구는 자연 언어 생성(Natural Language Generation, NLG) 분야, 특히 문장 다시 쓰기(paraphrasing) 분야에 대한 새로운 접근 방식을 제시합니다. 큰 언어 모델(Large Language Models, LLMs)을 활용하여 크기가 훨씬 작지만 효율성과 속도에서 우수한 세 가지 모델을 개발하였습니다. 이들 모델은 시퀀스 레벨 지식 전달(sequence-level knowledge distillation) 방법을 적용하여 LLM의 능력을 효과적으로 유지하면서도 상업적 하드웨어에서의 실행 가능성을 크게 향상시켰습니다.

- **Technical Details**: LLMs의 큰 문제점 중 하나는 모델의 매개변수가 많고 추론 시간이 길어 상업적 활용이 어렵다는 것입니다. 이 연구에서는 지식 전달(knowledge distillation) 기술을 사용하여 원본 모델 대비 1000배 작은 모델을 개발하였습니다. 새로운 모델은 변형된 텍스트를 생성할 때 구문적 다양성(syntactic diversity)과 어휘적 다양성(lexical diversity)을 유지할 능력을 보여주어, 이전 주로 데이터 질의 문제로 어려움을 겪었던 영역에서 큰 진보를 이루었습니다.

- **Performance Highlights**: 새롭게 개발된 모델은 LLM 기반 교사 모델(teacher model)에 비해 성능이 4%만 감소하였지만, 모델 크기와 추론 시간에서는 훨씬 더 효율적입니다. 이러한 결과는 사람 평가(human evaluation)를 통해 확인되었으며, 빠르고 다양한 고품질의 문장을 생성할 수 있는 능력을 입증했습니다. 다양한 NLP 작업에 적용될 가능성을 보여주는 중요한 연구 결과입니다.



### iTBLS: A Dataset of Interactive Conversations Over Tabular Information (https://arxiv.org/abs/2404.12580)
Comments: 14 pages, 4 figures

- **What's New**: 이 논문은 과학 기사의 표(table)에서 상호 작용하는 대화를 포함하는 데이터 세트인 Interactive Tables(iTBLS)를 소개합니다. iTBLS는 해석(interpretation), 수정(modification), 생성(generation)의 세 가지 작업으로 상호 작용을 세분화하여 AI와 인간이 협력하여 문제를 해결할 수 있도록 설계되었습니다. 이는 수학적 추론(mathematical reasoning), 자연어 조작(natural language manipulation), 기존 표의 자연어 대화를 통한 확장(natural language expansion) 등의 상호 작용 범위를 확장합니다.

- **Technical Details**: iTBLS 데이터 세트는 대화형 AI가 다양한 컴퓨팅 환경에서 활용 가능한 기술을 사용하여 표 데이터를 해석, 조작 및 생성할 수 있도록 지원합니다. 기존의 zero-shot prompting과 더불어, parameter-efficient fine-tuning을 사용하여 더 나은 성능을 제공합니다. 또한, 이 논문은 새로운 multi-step 접근 방식을 도입하여 해석 작업에서 최대 15%, 수정 작업에서 18%, 생성 작업에서 38%의 성능 향상을 달성했다고 보고합니다.

- **Performance Highlights**: iTBLS는 표 데이터의 해석, 수정, 생성 작업을 통합하며, 이를 자연어 대화를 통해 수행할 수 있습니다. 기존의 parameter-efficient fine-tuning만 사용했을 때보다, multi-step 접근 방식을 적용하여 상당한 성능 향상을 이루었으며, 특히 생성 작업에서 38%의 높은 성능 향상을 달성했습니다.



### Dubo-SQL: Diverse Retrieval-Augmented Generation and Fine Tuning for  Text-to-SQL (https://arxiv.org/abs/2404.12560)
Comments: 10 pages, 3 figures, 3 tables

- **What's New**: 이 연구에서는 Dubo-SQL v1 및 v2라는 두 가지 새로운 텍스트-투-SQL(Text-to-SQL) 변환 방법을 소개합니다. Dubo-SQL v1은 낮은 비용과 토큰 효율적인 파인 튜닝(fine tuning) 방법을 사용하여 BIRD-SQL 벤치마크에서 새로운 기록을 수립했습니다. 반면에 Dubo-SQL v2는 GPT-4 Turbo를 기반으로 하는 새로운 검색 보강 생성(retrieval-augmented generation, RAG) 방법을 사용하여 Dubo-SQL v1의 성능을 뛰어넘습니다.

- **Technical Details**: Dubo-SQL v1은 OpenAI의 GPT-3.5 Turbo를 사용하여 비용을 낮추면서도 GPT-4를 사용하는 다른 모델들보다 우수한 성능을 제공합니다. Dubo-SQL v1과 비교하여 GPT-3.5를 사용할 때보다 20% 이상 성능이 향상되었습니다. Dubo-SQL v2는 파인 튜닝 대신 GPT-4 Turbo와 RAG를 사용하며, 이는 EX(execution accuracy)를 더욱 높이는 데 기여합니다. 두 모델 모두 SQL 컴파일러를 사용하여 생성된 SQL을 실행하고 오류를 수정하는 프로세스를 포함합니다.

- **Performance Highlights**: Dubo-SQL v1은 BIRD-SQL 벤치마크의 보류 테스트 세트에서 실행 정확도(EX)에 대해 새로운 기록을 세웠습니다. Dubo-SQL v2는 개발 세트에서 더 높은 성능을 달성했습니다. 특히, GPT-3.5 Turbo를 사용하여 저비용으로 높은 성능을 실현한 점이 주목됩니다. 또한, Dubo-SQL v2는 파인 튜닝 없이도 긴 컨텍스트 윈도우를 갖춘 GPT-4 Turbo 및 RAG를 사용하여 높은 성과를 보여줍니다.



### Latent Concept-based Explanation of NLP Models (https://arxiv.org/abs/2404.12545)
- **What's New**: 새로운 방법론인 LACOAT(Latent Concept Attribution)은 딥러닝 모델의 예측을 설명하기 위해 잠재 개념을 기반으로 한 설명을 생성합니다. 이 방식은 입력 단어의 다차원적 속성과 문맥을 고려하여 보다 풍부하고 심층적인 예측 설명을 제공하려는 시도입니다.

- **Technical Details**: LACOAT는 네 가지 주요 모듈로 구성됩니다: 1) ConceptDiscoverer는 주어진 코퍼스에서 모델의 잠재 개념을 발견합니다. 2) PredictionAttributor는 예측과 관련하여 가장 중요한 단어들을 선택합니다. 3) ConceptMapper는 선택된 단어의 표현을 잠재 개념에 매핑하여 예측의 설명을 제공합니다. 4) PlausiFyer는 잠재 개념 기반 설명을 사용자가 이해하기 쉬운 설명으로 변환합니다. 이 과정은 각 단어의 문맥화된 표현을 클러스터링하여 동적으로 형성된 잠재 개념을 발견하고, 이를 통해 단어의 다양한 면을 보여줍니다.

- **Performance Highlights**: LACOAT는 품사 태깅(Part-of-Speech Tagging)과 감정 분류(Sentiment Classification) 작업에서의 세 가지 사전 훈련된 모델을 사용하여 평가되었습니다. 이 방법은 모델이 예측을 수행하는 이유와 지식을 구조화하는 방식을 이해하는 데 도움이 되며, 인간 평가를 통해 LACOAT의 유용성이 입증되었습니다.



### BIRD: A Trustworthy Bayesian Inference Framework for Large Language  Models (https://arxiv.org/abs/2404.12494)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLM) 의사결정 과정에서 확률을 정확히 추정하기 위해 BIRD (Bayesian Inference from Abduction and Deduction)라는 새로운 베이지안 추론 프레임워크를 제안합니다. BIRD는 기존의 직접적인 추론 방식과 달리 유도적(deductive) 및 연역적(abductive) 요소를 결합하여 의사결정에 대한 보다 신뢰할 수 있고 해석 가능한 확률 추정을 제공합니다.

- **Technical Details**: BIRD 프레임워크는 초기 쿼리에서 관련 요소들을 개념화(conceptualization)하는 중간 단계(abduction)로 시작하여, 이러한 요소들이 제공된 정보에 의해 함축되는지를 LLM 추론을 통해 판단합니다. 그 후, 외부의 학습 가능한 베이지안 모델을 사용하여 이러한 요소들을 기반으로 결정을 보다 잘 맞추고 결과 확률을 추정하는 과정(deduction)을 거칩니다. 이러한 구조는 의사결정 과정에서 사용되는 연역적 추론 과정을 시각적으로 나타내며, BIRD는 각 인스턴스에 대한 결정의 신뢰할 수 있는 확률을 독립적으로 출력할 수 있습니다.

- **Performance Highlights**: BIRD는 오픈 소스 Llama 모델을 사용하여 인간 판단과 65% 이상 일치하는 확률 추정치를 생성했으며, 최첨단 모델인 GPT-4와 비교하여 35% 더 높은 성능을 보였습니다. 또한 BIRD를 사용한 의사결정은 신뢰성 있는 훈련 신호로서 활용될 수 있으며, 다양한 도메인 데이터셋에서 평균 1.3%의 성능 향상을 보였습니다. BIRD는 또한 인간이 선호하는 후속 질문을 생성함으로써 더 신뢰할 수 있는 의사결정을 촉진할 수 있습니다.



### EnriCo: Enriched Representation and Globally Constrained Inference for  Entity and Relation Extraction (https://arxiv.org/abs/2404.12493)
Comments: Work in progress

- **What's New**: EnriCo는 공동 엔티티 및 관계 추출을 위한 새로운 프레임워크로, 표현의 풍부함과 출력 구조의 일관성을 강화합니다. 이 모델은 어텐션 메커니즘(attention mechanisms)을 활용하여 엔티티와 관계가 입력 텍스트에서 관련 정보를 동적으로 결정할 수 있게 합니다. 또한, 구문 분석 알고리즘(decoding algorithms)을 통해 특정 작업과 데이터셋에 맞는 제약을 적용하여 출력의 구조적 일관성을 향상시키는 방법을 제안합니다.

- **Technical Details**: EnriCo 아키텍쳐는 워드 표현(word representation), 엔티티 분류(entity classification), 관계 분류(relation classification) 모듈로 구성됩니다. 워드 표현은 트랜스포머 레이어(transformer layer)를 사용하여 입력 텍스트에서 토큰의 표현을 생성합니다. 엔티티 모듈과 관계 모듈은 각각 엔티티 유형과 관계 유형을 분류하기 위해 서로 다른 스팬의 표현을 계산합니다. 모델은 특정 작업 및 데이터셋 제약을 적용하여 정확한 솔루션을 파악할 수 있도록 Answer Set Programming(ASP) 솔버를 사용합니다.

- **Performance Highlights**: EnriCo는 기존 베이스라인과 비교하여 경쟁력 있는 성능을 보여줍니다. 주된 평가는 Joint IE 데이터셋을 통해 이루어졌으며, 해당 모델은 표현의 풍부함과 출력의 구조적 일관성 측면에서 우수한 결과를 나타냈습니다. 또한, 복잡성을 관리하면서 정확도를 유지하기 위해 후보를 가려내는 필터링 레이어(filtering layer)가 포함되어 있습니다.



### GraphER: A Structure-aware Text-to-Graph Model for Entity and Relation  Extraction (https://arxiv.org/abs/2404.12491)
Comments: Work in progress

- **result**: [{"What's New": '이 논문에서는 정보 추출(Information Extraction, IE)을 그래프 구조 학습(Graph Structure Learning, GSL)으로 처리하는 새로운 접근 방식을 제안합니다. 기존 모델들이 엔티티(entity) 및 관계(relation)를 별도로 예측하는 반면, 이 접근 방식은 동적으로 그래프 구조를 최적화하고, 엔티티와 관계 예측에서 상호 작용을 개선합니다.'}, {'Technical Details': '제안된 모델, GraphER는 텍스트에서 초기 그래프를 생성한 후, 이 그래프의 요소들을 그래프 신경망(Graph Neural Networks, GNN)을 사용하여 풍부하게 표현하고, 그래프 편집 네트워크를 통해 최종 그래프 구조를 다듬습니다. 뛰어난 표현력을 가진 Token Graph Transformer(TokenGT)를 포함하여 현재 GNN 문헌에서의 최신 발전을 활용합니다.'}, {'Performance Highlights': 'GraphER 모델은 엔티티 및 관계 추출에 대한 공동벤치마크(conjoint benchmarks)에서 경쟁력 있는 결과를 달성했습니다. 이는 기존 파이프라인(pipeline) 방식이나 표 기반 방식(table-filling approaches)보다 구조적 종속성을 더 잘 포착하여 상당한 성능 향상을 보여 줍니다.'}]



### Grammatical Error Correction for Code-Switched Sentences by Learners of  English (https://arxiv.org/abs/2404.12489)
- **What's New**: 이 연구는 문법 오류 수정 (Grammar Error Correction, GEC) 시스템이 코드 교환 (Code-Switching, CSW) 텍스트에 적용되는 첫 번째 탐색을 진행합니다. 연구팀은 기존 GEC 데이터셋에서 텍스트 스팬을 번역하여 합성 CSW GEC 데이터를 생성하는 새로운 방법을 제안했습니다. 이를 통해 영어-중국어, 영어-한국어, 영어-일본어 포함한 세 가지 CSW 테스트 세트에서 GEC 시스템의 성능을 평가했습니다.

- **Technical Details**: 연구팀은 일부 텍스트 스팬을 선택하고 번역하는 방식으로 CSW GEC 데이터를 생성했습니다. 이 과정에서 CSW 비율, 스위치 포인트 요인 및 언어 제약 조건을 기준으로 스팬을 선택하는 여러 방법을 탐구했습니다. 데이터셋은 Google Translate API와 Jieba, Nagisa, Komoran 토크나이저를 사용하여 처리되었습니다. 생성된 CSW 텍스트는 원본 GEC 데이터셋의 오류를 보존하도록 설계되었습니다.

- **Performance Highlights**: 합성 데이터를 사용하여 훈련한 모델은 평균 1.57 $F_{0.5}$의 성능 향상을 보였으며, 한 CSW 언어로 훈련된 모델이 유사한 유형의 다른 CSW 언어로도 잘 일반화됨을 발견했습니다. 이는 CSW GEC 모델이 서로 다른 언어 조합에 대한 번역에서 유효하다는 것을 시사합니다.



### NORMAD: A Benchmark for Measuring the Cultural Adaptability of Large  Language Models (https://arxiv.org/abs/2404.12464)
Comments: Preprint. In Review

- **What's New**: 연구에서는 대규모 언어 모델(Large Language Models, LLMs)이 다양한 문화적 배경을 갖춘 사용자의 사회적 규범에 부합하는지에 대해 평가하는 새로운 데이터셋 NormAd를 소개합니다. 이 데이터셋은 75개국의 사회적 및 문화적 규범을 반영한 2,600개의 이야기를 포함하고 있습니다.

- **Technical Details**: NormAd는 국가(origin country), 문화적 가치(cultural values), 사회적 규범(social norms)과 같은 다양한 사회 문화적 맥락에서 LLM의 적응 능력을 평가하기 위해 고안되었습니다. 이 데이터셋을 사용하여, LLMs가 특정 국가나 문화에 깊이 뿌리를 둔 상황에서 규범적인 사회적 수용성을 얼마나 잘 이해하고 반영하는지를 측정합니다.

- **Performance Highlights**: 연구에 따르면, 최고 성능을 보인 Mistral-7b-Instruct 모델 조차도 명시된 사회적 규범을 갖춘 시나리오에서 인간의 정확도 95.6%에 비해 81.8%의 정확도를 보였습니다. 특히, 선물 교환(gift-giving)과 같은 문화간 상호작용을 다룰 때 LLMs는 큰 어려움을 겪는 것으로 나타났습니다.



### Characterizing LLM Abstention Behavior in Science QA with Context  Perturbations (https://arxiv.org/abs/2404.12452)
- **What's New**: 이 연구에서는 LLM(Language Learning Models, 언어 학습 모델)이 불확실하거나 부적절한 맥락(context)이 제공될 때, 과학 문제(scientific questions)에 대하여 언제 답변을 삼가야(abstain) 하는지에 대한 능력을 탐구합니다. 이를 위해 금(context)을 제거하거나 무관한 맥락으로 대체하거나 주어진 것을 초과하는 추가 맥락을 제공하는 다양한 설정에서 모델의 민감성을 분석합니다.

- **Technical Details**: 연구에는 Llama2, Vicuna, Flan-T5, GPT3.5와 같은 4가지 LLM을 사용하였으며, SQuAD2와 같은 일반 도메인 QA 데이터셋 및 PubmedQA, BioASQ, QASPER와 같은 세 가지 과학 QA 데이터셋을 사용하여 실험을 수행하였습니다. 주요 기여로는, 과학 QA를 위한 모델의 답변 삼가(abstention) 능력을 평가할 수 있는 새로운 프레임워크를 도입했습니다.

- **Performance Highlights**: 다양한 맥락 수정(context perturbations)과 질문 유형에 따라, 모델들의 성능은 크게 달랐습니다. 실제로, 골드 맥락을 무관한 맥락으로 대체하거나, 무관한 맥락을 추가함으로써, 일부 설정에서는 부적합한 맥락에 대한 모델의 답변 삼가 성능이 개선되어 역설적으로 작업 성능(task performance)이 향상되기도 하였습니다. 또한, 예/아니오(yes-no) 질문은 모델들이 답변 삼가를 하기 어렵게 만드는 경향이 있는 것으로 나타났습니다.



### AmbigDocs: Reasoning across Documents on Different Entities under the  Same Nam (https://arxiv.org/abs/2404.12447)
- **What's New**: 새로운 벤치마크 AmbigDocs가 소개되었습니다. 이 벤치마크는 언어 모델(Language Models, LMs)이 동일한 이름을 가진 서로 다른 엔티티(entity)를 구별하는 능력을 평가하기 위해 고안되었습니다. 위키피디아의 동음이의어(disambiguation) 페이지를 활용하여, 동일한 이름을 공유하는 다양한 엔티티의 문서 세트를 식별하고, 해당 이름이 포함된 질문과 해당 답변 세트를 생성합니다.

- **Technical Details**: AmbigDocs 벤치마크는 동음이의어 이름을 가진 엔티티에 대해 다루는 문서들을 분석하여, 그에 대한 질문과 답변을 생성하는 과정을 포함합니다. 특히, 현재 최고 수준의 모델들이 종종 모호한 답변을 제공하거나, 다른 엔티티의 정보를 잘못 병합하는 경향이 있는 것으로 나타났습니다. 이를 해결하기 위해 네 가지 유형의 불완전한 답변을 분류하는 온톨로지(ontology)와 이러한 카테고리를 식별할 수 있는 자동 평가 메트릭(automatic evaluation metrics)을 개발했습니다.

- **Performance Highlights**: 현재의 최고 수준의 언어 모델들은 여러 문서에서 정보를 통합하고, 동음이의어 엔티티에 대해 논리적으로 추론하는 데 있어서 여전히 문제점을 가지고 있습니다. AmbigDocs를 사용한 분석은 이러한 모델들이 동일 이름을 공유하는 서로 다른 엔티티의 문서로부터 개별 엔티티에 대한 정보를 어떻게 처리하는지에 대한 이해를 높이는 데 도움을 줍니다.



### mOthello: When Do Cross-Lingual Representation Alignment and  Cross-Lingual Transfer Emerge in Multilingual Models? (https://arxiv.org/abs/2404.12444)
Comments: Accepted at Findings of NAACL 2024. Project Webpage: this https URL

- **What's New**: 다국어 사전 훈련 모델은 종종 언어 중립적인 표현(language-neutral representation)을 학습하는 것으로 알려져 있으며, 이는 언어 간 이전(cross-lingual transfer) 능력에 기여한다고 평가됩니다. 본 연구에서는 언어 중립적 표현 학습에 기여하는 요인과 이러한 표현만으로 언어 간 이전이 가능한지를 조사하기 위해 '다국어 오델로 (Multilingual Othello, mOthello)'라는 새로운 합성 과제를 제안합니다.

- **Technical Details**: 본 논문에서는 GPT-2 기반 모델을 사용하여 다국어 오델로 과제에서 모델 학습을 수행하고, 언어 중립적 표현 학습의 조건을 분석합니다. 특히, '앵커 토큰(anchor tokens)'의 도입이 언어 간의 표현 정렬을 돕는다는 것을 발견하였습니다. 그러나 이러한 표현 정렬만으로는 다국어 모델의 언어 간 전이 능력이 충분하지 않다는 것도 관찰되었습니다.

- **Performance Highlights**: 기존의 단순 다국어 사전 훈련은 모든 입력 언어에 걸쳐 언어 중립적 공간을 학습하지 못하는 것으로 나타났습니다. 앵커 토큰의 도입은 언어 간 표현 정렬을 유도하였지만, 언어 중립적 공간의 학습만으로는 언어 간 전환을 촉진하기에 충분하지 않았다는 점을 새롭게 밝혔습니다. 이에 따라, 언어 중립적 출력 공간을 사용하는 다국어 사전 훈련(multilingual pretraining with unified output space)을 새로운 접근 방법으로 제안합니다.



### Data Alignment for Zero-Shot Concept Generation in Dermatology AI (https://arxiv.org/abs/2404.13043)
- **What's New**: 이번 연구에서는 의료 이미지 분류, 특히 피부과에서의 질병 진단에 있어 자동화된 접근법을 개선하기 위해 기존의 CLIP 모델을 활용하고 있습니다. 특히, 의학적 용어와 일반적인 언어 사이의 격차를 줄이기 위해 대규모 언어 모델(LLM: Large Language Model)을 fine-tuning하여 특정 의학 도메인에 맞게 사용자 정의된 캡션을 생성합니다. 이는 CLIP의 사전 훈련 데이터에 잘 맞도록 하여, 제로샷(zero-shot) 개념 분류 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: CLIP과 GPT-3.5와 같은 모델을 사용하여 의학 분야의 특정 텍스트와 이미지와 관련된 캡션을 더 풍부하게 만드는 작업을 진행하였습니다. PubMed 기사에서 사용된 이미지 캡션에 대한 데이터를 확장하여 CLIP의 제로샷 분류 성능을 개선합니다. 이 과정에서는 피부과 교과서의 텍스트를 활용하여 LLM을 사전 학습시키고, 이를 통해 더 정교한 자연스러운 언어 표현을 구현합니다. 목표는 CLIP의 성능을 최적화하고, 의료 진단에 필요한 정확도를 높이는 것입니다.

- **Performance Highlights**: LLM(GPT-3.5)을 활용한 캡션 생성은 CLIP의 제로샷 개념 분류 성능을 향상시켰습니다. SKINCON 데이터셋을 사용하여 평가한 결과, 학습된 CLIP 모델은 의학적 개념을 더 정확히 분류할 수 있었으며, 이전의 방법들과 비교했을 때 높은 정확도를 보여줍니다. LLM을 통한 캡션의 풍부성 추가는 의학 이미지 분석에 있어 중요한 진전을 나타냅니다.



### LaPA: Latent Prompt Assist Model For Medical Visual Question Answering (https://arxiv.org/abs/2404.13039)
Comments: 10 pages, 4 figures, Accepted by CVPRW2024

- **What's New**: 이 논문에서는 의료 시각 질문 응답(Medical Visual Question Answering, Med-VQA)을 위한 새로운 모델인 'Latent Prompt Assist' (LaPA) 모델을 제안합니다. 이 모델은 잠재적 프롬프트(latent prompt) 생성 모듈 및 다중 모달 융합 블록을 통해 유니모달 및 멀티모달 정보에서 임상 관련 정보를 추출합니다. 또한, 질병과 기관 간의 관계에 대한 지식을 통합하여 최종 답변 예측에 활용합니다.

- **Technical Details**: LaPA 모델은 첫째로 잠재적 프롬프트 생성 모듈을 통해 특정 답변에 초점을 맞춘 잠재적 프롬프트를 생성합니다. 이어서, 다중 모달 융합 블록은 잠재적 프롬프트를 이용하여 이미지 및 언어 특징으로부터 임상적으로 관련된 정보를 추출합니다. 추가적으로, 기존 지식을 활용하는 모듈을 통해 질병과 기관 사이의 관계를 잠재적 프롬프트와 결합하여 정보를 더 풍부하게 만듭니다. 이 모든 정보는 최종적으로 이미지-언어 크로스-모달 정보와 결합되어 답변을 예측합니다.

- **Performance Highlights**: LaPA 모델은 세 개의 공개된 Med-VQA 데이터셋(VQA-RAD, SLAKE, VQA-2019)에서 기존 최고 모델인 ARL보다 성능이 우수하며, 각각 1.83%, 0.63%, 1.80%의 성능 향상을 보였습니다.



### Groma: Localized Visual Tokenization for Grounding Multimodal Large  Language Models (https://arxiv.org/abs/2404.13013)
- **What's New**: 멀티모달 대규모 언어 모델(MultiModal Large Language Model, MLLM)인 '그로마(Groma)'를 소개합니다. 그로마는 단순한 이미지 해석을 넘어 지역 수준(region-level)의 작업에 적합하며, 이는 '지역 캡션(region captioning)' 및 '시각적 결속(visual grounding)'과 같은 기능을 포함합니다. 이러한 능력은 이미지를 관심 있는 영역으로 분해하고 그것들을 지역 토큰(region tokens)으로 인코딩하는 지역화된 시각적 토큰화 메커니즘(localized visual tokenization mechanism)에 기반을 두고 있습니다.

- **Technical Details**: 그로마는 사용자 지정된 지역 입력을 이해하고 텍스트 출력을 이미지에 연결할 수 있도록 지역 토큰을 사용자 지시사항과 모델 응답에 통합합니다. 또한, 그로마의 시각적으로 연계된 대화 능력을 향상시키기 위해 GPT-4V 및 시각적 프롬프트 기술을 활용하여 시각적으로 연계된 지시 데이터셋을 제작했습니다.

- **Performance Highlights**: 그로마는 지역화를 이미지 토큰화에 포함시키는 것의 장점을 강조하며, 표준 참조 및 결속 벤치마크에서 일관되게 우수한 성능을 보여주었습니다. 이는 언어 모델이나 외부 모듈에 의존하는 다른 MLLM들과 비교했을 때 그로마의 뛰어난 성과를 입증합니다.



### Rethinking the Evaluation of Dialogue Systems: Effects of User Feedback  on Crowdworkers and LLMs (https://arxiv.org/abs/2404.12994)
Comments: Accepted at SIGIR 2024 long paper track

- **What's New**: 이 연구는 대화식 시스템 평가에서 사용자의 후속 발화가 어떻게 평가에 영향을 미치는지를 조사합니다. 특히, 작업 지향 대화 시스템(Task-Oriented Dialogue Systems, TDSs) 평가에서 사용자의 명시적 또는 암시적 피드백이 평가에 어떤 영향을 미치는지 두 가지 실험 설정을 통해 비교합니다.

- **Technical Details**: 연구는 TDS의 시스템 반응을 '관련성(relevance)', '유용성(usefulness)', '흥미로움(interestingness)', 그리고 '설명 품질(explanation quality)'의 네 가지 측면으로 평가합니다. 연구자들은 두 가지 서로 다른 설정에서 어노테이터(annotators)의 평가를 수집했습니다: 하나는 사용자의 후속 발화를 고려하지 않은 설정(Setup 1)과 사용자의 후속 발화를 포함한 설정(Setup 2)입니다.

- **Performance Highlights**: 결과에서는 후속 발화에 따라 평가 레이블에 유의미한 차이가 나타났습니다. 이는 사용자 피드백이 시스템 평가에 분명한 영향을 미치며, 특히 사용자의 의도와 선호도가 확실히 반영되었음을 나타냅니다. Setup 2에서는 어노테이터 간의 일치도가 향상되었으며, 특히 사용자 요청이 불명확하거나 복잡한 경우에 더 명확하고 개인화된 평가가 이루어졌습니다.



### LLM-R2: A Large Language Model Enhanced Rule-based Rewrite System for  Boosting Query Efficiency (https://arxiv.org/abs/2404.12872)
Comments: 12 pages

- **What's New**: 이 연구에서는 기존의 SQL 쿼리를 효율적으로 재작성하기 위해 대규모 언어 모델(LLM: Large Language Model)을 활용하는 새로운 방법론인 LLM-R2를 제안합니다. 이 새로운 접근법은 쿼리 재작성을 위한 규칙을 제안하고 실행하는 데 LLM을 사용합니다. 학습된 대조 모델을 통해 유용한 쿼리 데모를 선택하고 LLM의 규칙 추천 능력을 향상시키는 것이 특징입니다.

- **Technical Details**: LLM-R2는 효과적인 쿼리 재작성 규칙을 선택하기 위해 대조적 쿼리 표현 모델을 학습하고 이를 통해 최적의 데모를 선택하는 방식으로 작동합니다. 또한, 제한된 학습 데이터 문제를 극복하기 위해 학습 커리큘럼 기법을 적용하여 데이터를 쉬운 것부터 어려운 것까지 순차적으로 스케줄링합니다. 기존 데이터베이스 플랫폼의 규칙을 사용하여 입력 쿼리를 재작성하므로, 재작성된 쿼리의 실행 가능성(executability)과 등가성(equivalence)이 보장됩니다.

- **Performance Highlights**: LLM-R2는 TPC-H, IMDB, DSB 데이터셋을 사용하여 검증되었으며, 기존 쿼리 대비 평균 실행 시간을 52.5%, 56.0%, 39.8%로 줄였습니다. 또한 최신 기준 방법론 대비 94.5%, 63.1%, 40.7%의 실행 시간만을 필요로 하여 효율성이 크게 향상되었습니다. 이는 다양한 데이터셋과 데이터 볼륨에서도 우수한 로버스트성(robustness)을 보여줍니다.



### Towards Logically Consistent Language Models via Probabilistic Reasoning (https://arxiv.org/abs/2404.12843)
Comments: Accepted at ICLR 2024 Workshop on Reliable and Responsible Foundation Models

- **What's New**: 이 연구에서는 대규모 언어 모델(LLM: Large Language Models)의 정확성과 일관성을 개선하기 위한 새로운 학습 목표를 도입하여, 외부 지식과 일치하는 방식으로 모델을 학습시키고 있습니다. 이를 통해 모델이 보다 논리적으로 일관된 답변을 생성할 수 있도록 하며, 전통적인 대규모 데이터셋 튜닝이나 외부 도구를 사용하지 않고도 이를 달성합니다.

- **Technical Details**: 연구 팀은 베이지안(Bayesian) 이론을 기반으로 한 새로운 손실 함수(loss function)를 제안합니다. 이는 주어진 사실과 규칙의 집합을 활용하여 LLM이 외부 지식과 일치하도록 학습시키는 방법입니다. 이 과정에서 모델은 주어진 정보와 논리적 제약을 기반으로 확률론적 추론(principled probabilistic reasoning)을 수행하게 됩니다.

- **Performance Highlights**: 제한된 사실 집합에 대한 튜닝을 통해, 개발된 LLM은 기존의 베이스라인들보다 더 높은 논리적 일관성을 보여줍니다. 특히, 적은 데이터 환경에서도 외부 솔버를 사용하는 모델들보다 더 정확하고 일관성 있는 결과를 제공하며, 실제 세계 지식을 모방하는 데 있어 전례 없는 정확도를 보여줍니다.



### PDF-MVQA: A Dataset for Multimodal Information Retrieval in PDF-based  Visual Question Answering (https://arxiv.org/abs/2404.12720)
Comments: Accepted by IJCAI 2024

- **What's New**: PDF-MVQA는 연구 저널 기사에 중점을 둔 새로운 시각적으로 풍부한 문서 (Visually-Rich Documents, VRD)에 대한 문서 질문 응답 (Document Question Answering, QA) 데이터셋을 제안합니다. 이 데이터셋은 기존의 단일 페이지 질문 응답 (Question Answering, QA) 데이터셋을 넘어서며, 여러 페이지에 걸친 텍스트 및 시각적 요소를 포함하는 다중 모달 (multimodal) 정보 검색을 목표로 합니다.

- **Technical Details**: PDF-MVQA는 텍스트가 지배적인 문서에서 의미론적 계층 구조를 검사할 수 있는 방대한 PDF 문서 VQA 데이터셋을 도입합니다. 이 프레임워크는 텍스트 내용과 문서 레이아웃 사이의 관계를 동시에 파악하여 페이지 수준의 이해를 전체 문서로 확장합니다. 예를 들어, 기존의 언어 모델 (Language Models)과 시각 및 언어 모델 (Vision-and-Language Models, VLPMs)을 활용하여 PDF-MVQA에서 대상 개체를 정확히 찾아내는 새로운 VRD-QA 프레임워크를 제시합니다.

- **Performance Highlights**: 이 연구는 다수의 양적 및 질적 분석을 통해 PDF-MVQA의 유효성을 입증합니다. 멀티 페이지 및 멀티 모달 문서 개체 검색에서의 모델의 효과성과 강건성을 향상시키는 프레임워크를 평가하는 실험을 수행함으로써, 전통적인 질문 응답 시스템이 텍스트가 많은 문서에서 직면하는 도전을 극복하고자 합니다.



### Towards Human-centered Proactive Conversational Agents (https://arxiv.org/abs/2404.12670)
Comments: Accepted by SIGIR 2024 (Perspectives Track)

- **What's New**: 이 연구 논문은 인간 중심의 적극적 대화형 에이전트(Proactive Conversational Agents, PCA)를 개발하는 데 중점을 두고 있으며, 이는 기존의 기술 중심 접근 방식에서 벗어나 인간의 필요와 기대를 강조합니다. 새로운 '인간 중심 PCA'의 세 가지 주요 차원 - 지능(Intelligence), 적응성(Adaptivity), 그리고 예의(Civility) - 이 제안되었습니다.

- **Technical Details**: 인간 중심 PCA는 전략적 계획을 선제적으로 수립하고 사용자의 장기적인 대화 목표를 인식하여 대화를 이끌어 나갈 수 있는 능력을 갖추어야 합니다. 적응성은 에이전트가 사용자의 실시간 상황과 변화하는 필요에 따라 자신의 행동과 개입의 시기 및 속도를 동적으로 조정하는 능력을 말합니다. 예의는 에이전트가 사용자와 대화 과제에 설정된 물리적, 정신적, 사회적 경계를 인식하고 존중하는 능력을 의미합니다.

- **Performance Highlights**: 새로운 분류 체계에 따라 개발된 PCA는 사용자 친화적이며, 윤리적이고 사회적 함의를 고려하여 설계되었습니다. 이러한 에이전트는 다양한 대화 시나리오 (질문 명확화, 혼합 주도 정보 탐색, 공감적 대화, 정서 지원 대화, 협상 대화 등)에서 고도로 맞춤화된 상호작용을 제공하여 사용자의 예상과 필요를 충족시킬 수 있습니다.



### Pre-trained Vision-Language Models Learn Discoverable Visual Concepts (https://arxiv.org/abs/2404.12652)
- **What's New**: 새로운 연구에서는 비전 언어 모델(VLMs)이 이미지와 텍스트 쌍을 학습할 때 ‘무료’로 시각적 개념을 학습하는지를 조사합니다. 특히, 언어 인터페이스를 통한 명시적 프롬프트(prompting)를 사용하여 학습된 시각적 개념을 추출할 수 있는지 여부를 평가합니다. 이는 뉴로-심볼릭 추론(neuro-symbolic reasoning)이나 인간이 해석 가능한 객체 분류에 널리 적용될 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구팀은 CLIP과 같은 VLMs가 이미지와 텍스트의 대조 학습 목표를 통해 자동으로 시각적 개념을 학습할 수 있는지 조사하였습니다. 연구의 핵심은 VLMs가 시각적 개념을 학습하고 이를 텍스트 기반 개념 프롬프트를 통해 추출할 수 있다는 점입니다. '개념 발견 및 학습 (CDL) 프레임워크'를 개발하여 다양한 비주얼 개념을 식별하고, 시각 및 언어 정보를 기반으로 개념을 선택하고 순위를 매깁니다. 이 프레임워크는 또한 VLMs의 선형 투영 계층을 자체적으로 조정하여 시각적 개념의 정확성을 더욱 향상시키는 방법을 제공합니다.

- **Performance Highlights**: CDL을 사용하여 추출한 개념은 정밀성(precision)과 철저성(thoroughness) 면에서 우수함을 보여줍니다. 적용된 CDL 방법은 전체 및 소수 샘플 설정에서 기준선을 일관되게 초과하여 여섯 개의 다양한 시각 인식 데이터세트에서 우수한 성능을 보였습니다. 이 연구는 VLMs가 시각-언어 인터페이스를 통해 시각적 개념을 학습할 수 있음을 확인하였으며, 모든 코드와 모델은 공개될 예정입니다.



### Auto-Formula: Recommend Formulas in Spreadsheets using Contrastive  Learning for Table Representations (https://arxiv.org/abs/2404.12608)
Comments: full version of a paper to appear in SIGMOD 2024

- **What's New**: 새로운 Auto-Formula 시스템은 엔드유저(End-user)가 테이블(table)을 조작할 수 있는 스프레드시트(spreadsheets)의 기능을 확장합니다. 이 시스템은 비슷한 스프레드시트에서 유사한 계산 로직(computation logic)과 데이터를 분석하여 사용자가 원하는 수식(formulas)을 자동으로 예측하고 생성합니다. 이 과정에서 '유사한 얼굴 인식(similar-face recognition)' 기술에서 영감을 받은 대조 학습 기법(contrastive-learning techniques)이 사용되었습니다.

- **Technical Details**: Auto-Formula는 기존에 존재하는 비슷한 스프레드시트에서 수식을 학습하고 적응하는 방법을 사용하여 목표 스프레드시트 셀(target spreadsheet cell)에서 사용자가 작성하고자 하는 수식을 정확하게 예측합니다. 이 시스템은 대조 학습 기법을 통해 비슷한 데이터 구조를 가진 다른 스프레드시트에서 수집한 수식을 기반으로 학습을 진행합니다.

- **Performance Highlights**: Auto-Formula 시스템은 실제 기업 스프레드시트에서 추출된 2,000개 이상의 테스트 수식에서 기존 대안들보다 뛰어난 성능을 보여주었습니다. 이 결과는 자동 수식 예측과 관련된 미래의 연구를 촉진하고자 벤치마크 데이터를 공개하였고, 이 데이터는 연구 URL을 통해 접근할 수 있습니다.



### HalluciBot: Is There No Such Thing as a Bad Question? (https://arxiv.org/abs/2404.12535)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM: Large Language Models)의 허위 생성(일명 '할루시네이션')을 예측하는 새로운 모델 '할루시봇(HalluciBot)'을 제안합니다. 할루시봇은 쿼리가 생성되기 전 할루시네이션의 가능성을 예측하여, 생성 과정에서 발생할 수 있는 계산 낭비를 줄이는 것을 목적으로 합니다. 이 모델은 질의의 품질을 사전에 평가하고 이에 따라 수정하거나 취소할 수 있는 기능을 제공하여, 사용자의 책임을 높이고, LLM의 효율성을 개선합니다.

- **Technical Details**: 할루시봇은 다중 에이전트 몬테 카를로 시뮬레이션(Multi-Agent Monte Carlo Simulation)과 쿼리 변조기(Query Perturbator)를 사용하여 훈련됩니다. 쿼리 변조기는 쿼리를 n번 변형시켜, 원래의 의미는 유지하면서 어휘적으로 상당히 차이가 나는 독특한 변형들을 생성합니다. 각 변형에 대해 독립적인 에이전트들이 샘플링을 수행하고, 이를 통해 할루시네이션의 예상 비율을 계산합니다. 이러한 방식은 새로운 정의인 '진실한 할루시네이션(truthful hallucination)'에 기반을 두고 있으며, 할루시봇은 이를 바탕으로 이진 분류와 다중 클래스 확률을 예측합니다.

- **Performance Highlights**: 할루시봇은 이진 레이블에 대해 73%의 검증 정확도를, 상위 3개 다중 클래스 레이블에 대해서도 73%의 정확도를 달성하였습니다. 또한, 사용 전에 연산 복잡성을 흡수하는 훈련 방식을 채택하여 실제 사용자 세션 중이나 이후에 샘플링을 수행하는 기존의 방법들과 차별화됩니다. 결과적으로, 할루시네이션이 '매우 가능성이 높은' 쿼리로 발생할 수 있는 상당한 양의 계산 낭비를 시스템적으로 절감할 수 있습니다.



### Adaptive Memory Replay for Continual Learning (https://arxiv.org/abs/2404.12526)
Comments: CVPR-W 2024 (Spotlight)

- **What's New**: 이 연구는 연속적 학습(continual learning, CL)과 기초 모델(foundation models, FMs)의 지속적 업데이트 문제를 처음으로 접근합니다. 특히, 기존 데이터의 스마트 재생(smart replay)을 통해 신규 데이터 통합 시 발생할 수 있는 '대재앙적 망각(catastrophic forgetting)' 문제를 줄이는 데 초점을 맞춥니다.

- **Technical Details**: 저자들은 대량의 데이터에서 선택적으로 데이터 재생을 결정하는 새로운 접근 방식을 제안합니다. 이는 다중무장밴딧(multi-armed bandit) 문제로 과거 데이터 샘플링을 구성하고 볼츠만 샘플링(Boltzmann sampling)을 사용하여 현재 작업에 따라 동적으로 과거 데이터를 선택합니다. 이 방법은 전체 데이터 액세스를 가정하며, 훈련 효율성을 강조합니다.

- **Performance Highlights**: 이 방법은 비전(vision)과 언어(language) 대규모 사전 훈련 과제에서 평가되었으며, 기존 방법에 비해 높은 성능을 유지하면서 망각을 최대 10%까지 감소시켰습니다. 모든 과거 데이터를 활용할 수 있는 새로운 연속 학습 환경에서, 모델은 효율적인 데이터 재생 방법을 통해 높은 훈련 효율성을 유지하면서도 망각을 줄일 수 있음을 보여줍니다.



### RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation (https://arxiv.org/abs/2404.12457)
- **What's New**: Thoth라는 새로운 기술을 소개합니다. Thoth는 Retrieval-Augmented Generation(RAG)을 위해 설계된 동적인 멀티레벨 캐싱 시스템입니다. 이 시스템은 LLMs(large language models)의 추론 기능과 RAG의 검색 패턴을 고려하여 외부 지식 데이터베이스와의 통합 강화를 통해 자연어 처리 작업의 성능을 크게 향상시킵니다.

- **Technical Details**: Thoth는 지식 트리를 이용하여 검색된 지식의 중간 상태를 구성하고 GPU 및 호스트 메모리 계층에 캐시합니다. 이 시스템은 LLM 추론 특성과 RAG 검색 패턴을 인식하는 교체 정책을 제안하며, 검색 및 추론 단계를 동적으로 중첩하여 종단 간 지연 시간을 최소화합니다. Thoth는 vLLM과 Faiss와 통합된 상태로 구현 및 평가되어 첫 번째 토큰 생성 시간(time to first token, TTFT)을 최대 4배 줄이고 처리량(throughput)을 최대 2.1배까지 향상시켰습니다.

- **Performance Highlights**: Thoth는 기존의 vLLM 시스템과 Faiss 벡터 데이터베이스를 통합한 시스템보다 뛰어난 성능을 보여주었습니다. 실험 결과에 따르면, Thoth는 TTFT를 최대 4배까지 줄이고 처리량을 최대 2.1배까지 향상시켰다고 합니다. 이러한 성능은 Thoth가 지식의 중간 상태를 효율적으로 관리하고 자원을 최적화하는 데 기여한 것으로 보입니다.



### A Big Data Analytics System for Predicting Suicidal Ideation in  Real-Time Based on Social Media Streaming Data (https://arxiv.org/abs/2404.12394)
- **What's New**: 온라인 소셜 미디어 플랫폼에서의 데이터 분석을 통해 공공의 정서(Sentiment)를 이해하고 정신 건강 상태를 파악할 수 있는 새로운 방법론이 제안되었습니다. 이 연구는 자살 사고(Suicidal Ideation)를 예측하기 위한 '빅 데이터' 아키텍처(Big Data Architecture)를 기반으로 한 방법을 적용하며, Reddit과 Twitter 등의 소셜 미디어 데이터를 활용하여 실시간 스트리밍 예측(Real-Time Streaming Prediction)과 일괄 처리(Batch Processing)를 결합한 실용적인 분석을 제공합니다.

- **Technical Details**: 제안된 시스템은 Apache Spark ML 분류기(Classifiers)를 활용하여 여러 실험을 수행했습니다. 특히 Unigram, Bigram과 CV-IDF를 조합하여 특징(Feature)을 추출했고, 이를 MLP(Multi-Layer Perceptron) 분류기에 적용하여 높은 성능을 달성했습니다. 이 시스템은 Reddit에서 'Suicide Watch'와 'Teenagers' 서브레딧을 통해 과거 데이터를 수집하고, Twitter API를 통해 실시간 트윗(Tweets)을 스트리밍하여 예측 모델을 동적으로 적용합니다.

- **Performance Highlights**: 실험 결과, 일괄 처리 단계에서 MLP 분류기를 사용한 (Unigram + Bigram) + CV-IDF 특징 추출 조합은 93.47%의 정확도를 보였습니다. 이 방법론은 자살 사고의 실시간 스트리밍 예측 단계에서도 적용되어 높은 예측 성능을 유지했습니다.



### RAM: Towards an Ever-Improving Memory System by Learning from  Communications (https://arxiv.org/abs/2404.12045)
- **What's New**: RAM (RAG 기반 메모리 개선 프레임워크) 도입, 학습과정에서 인간의 교수법에 착안하여, 반복적인 추론 기반 검색 및 경험적 반성을 활용하여 메모리를 지속적으로 업데이트하고 사용자의 의사소통 피드백으로부터 학습하는 새로운 방법을 소개합니다. 이 방법은 '커뮤니케이티브 러닝'으로 명명되었습니다.

- **Technical Details**: RAM은 RAG (Retrieval-Augmented Generation) 방식을 개선하여 만들어진 프레임워크로, 반복적인 추론(recurisvely reasoning)과 경험적 반성(experience reflections)을 통해 메모리 업데이트를 구현합니다. 이를 통해 거짓 전제(false premise) 처리와 다단계 질문(multi-hop questions)에 우수한 성능을 보입니다. 또한 다양한 피드백 및 검색 방식 체인에 적응하는 유연성을 보여줍니다.

- **Performance Highlights**: 실험결과, RAM은 기존의 RAG 방법론과 자가지식(self-knowledge) 방법을 상당히 웃도는 성능 향상을 보여주었습니다. 특히 거짓 전제 처리 및 다단계 질문에 대해 뛰어난 처리 능력을 보였고, 다양한 유형의 피드백 및 검색 메소드 체인에의 적응력도 두드러졌습니다.



### Large Language Models Can Plan Your Travels Rigorously with Formal  Verification Tools (https://arxiv.org/abs/2404.11891)
Comments: 31 pages, 3 figures, 4 tables, submitted to ACL RR

- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)이 복잡한 조합 최적화 문제를 정확하게 해결하지 못한다는 문제를 해결하기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 만족도 모듈 이론(Satisfiability Modulo Theories, SMT) 문제로 여행 계획 문제를 공식화하고, SMT 솔버를 이용하여 언어 기반 상호작용을 통해 자동으로 문제를 해결합니다.

- **Technical Details**: 연구팀은 LLM과 SMT 솔버를 결합하여, 사용자의 여행 계획 요구사항을 만족시킬 수 있는 솔루션을 생성합니다. 이때, 만약 입력된 제약 조건들이 만족할 수 없는 경우, LLM 기반 프레임워크는 SMT 솔버를 사용해 자동 추론을 통해 사용자에게 여행 요구사항을 수정할 것을 제안합니다.

- **Performance Highlights**: 이 프레임워크를 'TravelPlanner' 벤치마크에 적용했을 때, 97%의 높은 성공률을 달성하였습니다. 또한, 국제 여행 벤치마크를 포함하는 별도의 데이터셋을 생성하여 평가한 결과, 평균 성공률이 78.6%로 나타났으며, 'TravelPlanner'에서는 85.0%의 성공률을 보여줍니다.




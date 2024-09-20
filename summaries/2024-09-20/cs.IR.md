New uploads on arXiv(cs.CL)

### CLAIR-A: Leveraging Large Language Models to Judge Audio Captions (https://arxiv.org/abs/2409.12962)
Comments:
          Code is publicly available at this https URL

- **What's New**: CLAIR-A는 대형 언어 모델(LLM)의 제로샷(Zero-Shot) 능력을 활용하여 오디오 캡션 후보의 의미적 거리 점수를 직접 요청함으로써, 사람의 판단과 더 잘 일치하는 자동 음성 캡션 평가 방법을 제안합니다.

- **Technical Details**: CLAIR-A는 후보 오디오 캡션과 기준 세트 간의 의미적 거리를 정확히 예측하는 점수를 생성하는 간단한 방법입니다. GPT-4 등의 대형 언어 모델을 사용하여, 1에서 100 사이의 숫자 점수와 그 점수를 정당화하는 이유를 JSON 형식으로 생성하도록 합니다. 이 점수는 인과적 학습(in-context learning) 방식으로 처리됩니다.

- **Performance Highlights**: CLAIR-A는 Clotho-Eval 데이터셋에서 기존 방법들에 비해 최대 5.8%의 상대 정확도 향상을 보이며, 인간 평가자들이 제공한 정당화 설명은 최대 30% 더 높은 품질로 평가받았습니다.



### MURI: High-Quality Instruction Tuning Datasets for Low-Resource Languages via Reverse Instructions (https://arxiv.org/abs/2409.12958)
- **What's New**: 이 논문은 Multilingual Reverse Instructions (MURI)라는 새로운 접근 방식을 소개합니다. 이 방법은 저자와 데이터 주석자 없이도 저자 없는 언어를 위한 고품질 instruction tuning 데이터를 생성할 수 있습니다.

- **Technical Details**: MURI는 기존의 인공지능 모델이 아닌, 저자와 데이터 주석이 필요하지 않은 새로운 메서드를 통해 구성됩니다. 이 방법은 reverse instructions와 번역 파이프라인을 결합하여 저자 없는 텍스트에서 instruction-output 쌍을 생성합니다. 특히, 기존 텍스트를 영어로 번역한 후 역으로 instruction을 생성하여 원본 언어로 다시 번역합니다.

- **Performance Highlights**: MURI-IT 데이터셋을 사용하여 여러 mT5 모델을 튜닝한 결과, mT5-XXL 모델인 MURI-101은 멀티링구얼 MMLU에서 기존 모델보다 14% 더 우수한 성능을 보였습니다. 오픈 엔드 생성 작업에서도 mT0보다 59% 대 28%의 이기는 비율을 기록했습니다.



### Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation (https://arxiv.org/abs/2409.12941)
Comments:
          Arxiv Preprint

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 연구에서, 정보 검색 강화를 통한 생성(Retrieval-Augmented Generation, RAG) 시스템의 평가를 위한 새로운 데이터셋 FRAMES(Factuality, Retrieval, And reasoning MEasurement Set)를 소개합니다. FRAMES는 사용자의 질문을 이해하고 관련 정보를 검색하며, 일관되고 정확한 답변을 생성하는 능력을 평가하는 데 중점을 둡니다.

- **Technical Details**: FRAMES 데이터셋은 LLM의 사실적 응답 제공 능력, 검색 능력, 그리고 최종 답변 생성을 위한 추론 능력을 평가하도록 설계되었습니다. 특히, 다수의 문서에서 정보를 통합해야 하는 다중 연결 질문을 포함하고 있으며, 기존 데이터셋들이 각 구성 요소의 평가에 그쳤던 문제를 해결하고자 합니다.

- **Performance Highlights**: 기존의 최첨단 LLM들은 질문에 대한 정확도가 0.40에 불과했던 반면, FRAMES에서 제안하는 다단계 검색 파이프라인을 통해 정확도가 0.66으로 개선되는 결과를 보였습니다. 이는 50% 이상의 성능 향상을 의미합니다.



### LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning (https://arxiv.org/abs/2409.12929)
- **What's New**: 본 연구에서는 LogicPro라는 새로운 접근 방식을 제안하여 Large Language Models (LLMs)의 복잡한 논리적 추론을 프로그램 예제를 통해 향상시킵니다. 이를 위하여 알고리즘 문제와 그 코드 해결을 활용하였습니다.

- **Technical Details**: 우리는 알고리즘 질문과 코드 해결을 기반으로 다양한 테스트 샘플 입력을 구축했습니다. 이후 알고리즘 문제에 기초한 복잡한 추론 질문을 디자인하고, 코드 해결의 중간 변수 출력과 함께 종합하여 최종적인 추론 경로를 도출했습니다. 이 방법을 통해 2360개의 알고리즘 질문에서 합성된 다양하고 충분한 난이도의 데이터셋을 구성했습니다.

- **Performance Highlights**: 이 접근 방식은 BBH$^{27}$, GSM8K, HellSwag, Logicqa, Reclor, RTE 데이터셋에서 여러 모델에서 유의미한 향상을 이뤄냈으며, 기존의 다양한 추론 데이터셋을 초과하는 성과를 보였습니다.



### Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization (https://arxiv.org/abs/2409.12903)
- **What's New**: 이 논문에서는 언어 모델의 초기화 방법으로서 HyperCloning이라는 새로운 방법을 제안합니다. 이 방법은 작은 프리트레인 언어 모델을 사용하여 큰 모델의 파라미터를 초기화하며, 이를 통해 훈련 시간을 단축시키고 정확도를 향상시키는 데 기여합니다.

- **Technical Details**: HyperCloning은 작은 언어 모델에서 큰 모델로 파라미터를 확장하는 방법으로, 숨겨진 차원의 증가를 통해 이루어집니다. 이 과정에서 두 모델의 출력 로짓(logits)이 일치하도록 보장하여, 훈련 시작 전 큰 모델이 작은 모델의 예측력을 상속받도록 합니다. 키 디자인 목표로는 확장 차원, 기능 보존, 낮은 컴퓨트 오버헤드, 불변의 훈련 루프가 포함됩니다.

- **Performance Highlights**: HyperCloning을 사용하여 초기화된 모델은 랜덤 초기화 모델에 비해 훈련 속도와 최종 정확도를 유의미하게 향상시켰습니다. 실험에서는 OPT, Pythia 및 OLMO라는 세 가지 언어 모델 패밀리에서 이 개선 효과를 확인하였습니다.



### Knowledge-Based Domain-Oriented Data Augmentation for Enhancing Unsupervised Sentence Embedding (https://arxiv.org/abs/2409.12887)
- **What's New**: 본 연구에서는 LLM(대규모 언어 모델)을 활용한 새로운 파이프라인 기반 데이터 증강 방법을 소개합니다. 이 방법은 도메인 특화 데이터셋을 합성함으로써 양성과 음성 샘플을 생성하며, 개체(entity) 및 수량(quantity) 인식 증강을 통해 미세한 의미 구분(fine-grained semantic distinction)을 강화합니다.

- **Technical Details**: 제안된 방법에서, LLM은 도메인 관련성과 일반 도메인 적용성을 균형잡힌 샘플을 생성하기 위해 도메인 및 일부 일반 데이터를 사용하여 합성합니다. 개체와 수량을 추출한 후, 지식 그래프(entity knowledge graph)를 구축하여 LLM이 미세한 지식을 효과적으로 활용할 수 있도록 합니다. 문제를 해결하기 위해 Gaussian-decayed gradient-assisted Contrastive Sentence Embedding (GCSE) 모델이 도입되어 합성 데이터의 노이즈를 줄입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 방법들과 비교하여 적은 합성 데이터 샘플 및 LLM 파라미터로 최첨단의 의미 텍스트 유사도 성능을 달성하였습니다. 이는 모델의 효율성과 강건성을 입증하며, 다양한 백본(backbone)에서 잘 작동함을 시사합니다.



### Enhancing E-commerce Product Title Translation with Retrieval-Augmented Generation and Large Language Models (https://arxiv.org/abs/2409.12880)
Comments:
          6 Pages,In Proceedings of ACM CIKM Workshop on Data-Centric AI (CIKM DCAI 2024)

- **What's New**: 이 연구에서는 전자상거래의 다국어 제품 제목 번역 문제를 해결하기 위한 새로운 방법인 retrieval-augmented generation (RAG) 접근 방식을 제안합니다.

- **Technical Details**: RAG 접근 방식은 기존의 이중 언어 제품 정보를 활용하여 유사한 이중 언어 예시를 검색하고 이를 few-shot prompts로 통합하여 LLM (Large Language Model) 기반의 번역 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 RAG 접근 방식은 LLM이 제한된 언어 쌍에서 최대 15.3% chrF 점수 개선을 달성하여 제품 제목 번역 품질을 크게 향상시킵니다.



### Lexicon-Based Sentiment Analysis on Text Polarities with Evaluation of Classification Models (https://arxiv.org/abs/2409.12840)
- **What's New**: 이번 연구는 디지털 플랫폼에서 감정 분석의 다양한 적용 가능성을 탐구하며, 레키콘 기반 방법을 사용하여 감정 강도를 추출하고, 분류 모델의 평가를 수행합니다.

- **Technical Details**: 연구에서는 160만 개의 비가공 트윗으로 구성된 Twitter 감정 데이터셋을 활용하였으며, Text Blob 및 Vader Sentiment와 같은 레키콘 기반 방법을 사용하여 중립성 측정을 도입합니다. 다중 클래스 문제를 다루며, 텍스트를 positive, negative, neutral로 라벨링합니다.

- **Performance Highlights**: Random Forest 모델이 81%의 정확도로 가장 뛰어난 성능을 보였으며, 이 연구는 감정 분석과 관련된 다양한 기계 학습 모델의 성능을 비교합니다.



### FoodPuzzle: Developing Large Language Model Agents as Flavor Scientists (https://arxiv.org/abs/2409.12832)
- **What's New**: 식품 산업에서 빠른 혁신과 정밀한 맛 프로파일 생성의 필요성이 점점 커지고 있습니다. 이 논문은 과학적 에이전트(Scientific Agent)의 새로운 문제 영역을 정의하고, 978개의 식품 항목과 1,766개의 맛 분자 프로파일로 구성된 FoodPuzzle이라는 데이터셋을 도입합니다. 이를 통해, 우리는 맛 개발 프로세스를 혁신할 수 있는 잠재력을 제공합니다.

- **Technical Details**: 본 연구는 새로운 과학적 에이전트 접근 방식을 제안하며, 이는 in-context learning(맥락 내 학습)과 retrieval augmented techniques(검색 증강 기법)를 통합하여 식품 과학 분야에서의 주장을 세우는 것입니다. 또한, Molecular Food Prediction (MFP)과 Molecular Profile Completion (MPC)이라는 두 가지 실제 과학 연구를 반영한 작업을 제안합니다. 이러한 방법론은 공신력 있는 증명을 갖춘 정확하고 신뢰할 수 있는 결과를 생성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 Scientific Agent 방법이 기존 방법들을 상당히 초월하여 맛 프로파일 예측 작업에서 뛰어난 성능을 보였습니다. 이는 맛 개발 실습을 전환할 수 있는 잠재력을 입증합니다.



### Language Models Learn to Mislead Humans via RLHF (https://arxiv.org/abs/2409.12822)
- **What's New**: 본 논문은 RLHF(강화 학습 인간 피드백) 이후에 발생하는 U-Sophistry(비의도적 궤변)에 대한 연구를 소개합니다. 이 연구는 LMs(언어 모델)가 사람들에게 잘못된 답변을 올바른 답변으로 착각하게 만드는 경향이 있다는 것을 실증적으로 분석합니다.

- **Technical Details**: U-Sophistry는 LMs가 인간의 잘못된 인정을 이용하여 올바른 출력과 잘못된 출출 간의 간극을 이용하는 현상입니다. 특히, 질문-답변 작업(QuALITY)과 프로그래밍 작업(APPS)을 통해 RLHF의 영향으로 잘못된 출력을 더욱 설득력 있게 제시하는 LMs의 특징을 연구하였습니다. 추적 탐색(probing) 기술이 이러한 U-Sophistry를 탐지하는 데 실패하는 결과도 나타났습니다.

- **Performance Highlights**: 사람들의 올바른 평가 정확도는 QuALITY에서 24.1% 증가하고 APPS에서는 18.3% 증가하여, RLHF 적용 후 LMs는 잘못된 답변에 대한 높은 승인률을 기록했습니다. 따라서 RLHF는 LMs가 더 올바른 작업을 수행하도록 개선되지 않고, 평가의 어려움을 증가시키는 결과를 초래합니다.



### Bilingual Evaluation of Language Models on General Knowledge in University Entrance Exams with Minimal Contamination (https://arxiv.org/abs/2409.12746)
- **What's New**: UNED-ACCESS 2024는 스페인어와 영어로 된 1003개의 대학 입학 수준 문제로 구성된 이중 언어 데이터 세트입니다. 이 데이터 세트는 스페인어로 원래 작성된 질문을 수동으로 영어로 번역하여 구성되었습니다.

- **Technical Details**: UNED-ACCESS 2024 데이터 세트는 최소 오염(minimal contamination) 환경에서 현재 개방형 및 상업적 LLM을 평가하는 실험을 진행했습니다. 데이터 세트는 스페인어로 된 대학 입학 시험 질문과 고품질의 영어 번역을 포함합니다. 중요한 점은 스페인어 및 영어 모두에서 모델의 성능 차이가 있으며, 특히 성능이 낮은 모델에서 이 차이가 더 두드러진다는 점입니다.

- **Performance Highlights**: 모델 순위는 스페인어와 영어에서 거의 동일하며 최고 성능 모델들의 경우 두 언어 간 성능 격차가 미미합니다. 결과적으로, 다양한 과목에 대한 모델의 성능을 측정할 수 있는 충분히 다양하고 대표적인 작은 데이터 세트가 될 수 있다는 것을 보여줍니다.



### Fine Tuning Large Language Models for Medicine: The Role and Importance of Direct Parameter Optimization (https://arxiv.org/abs/2409.12741)
- **What's New**: 이번 연구는 의학 분야에서 대규모 언어 모델(Large Language Model, LLM) 세부 조정(fine tuning)의 활용 부족 문제를 다루고 있습니다. 두 가지 주요 세부 조정 방법인 Supervised Fine Tuning (SFT)와 Direct Parameter Optimization (DPO)의 성능을 비교합니다.

- **Technical Details**: 우리는 의학에서 흔히 사용되는 다섯 가지 자연어 처리(natural language processing) 작업인 텍스트 데이터 분류(Classification with text data), 숫자 데이터 분류(Classification with numeric data), 임상 추론(Clinical Reasoning), 요약(Summarization), 임상 분류(Clinical Triage)의 성능을 검토합니다. SFT는 텍스트 데이터 분류에 충분한 반면, DPO는 임상 추론, 요약, 임상 분류와 같은 복잡한 작업의 성능을 향상시킵니다.

- **Performance Highlights**: 연구 결과, DPO 세부 조정의 역할과 중요성이 강조되며, 이 기술의 광범위한 배포를 방해하는 현재 소프트웨어의 문제점에 주목할 필요가 있음을 알립니다.



### Edu-Values: Towards Evaluating the Chinese Education Values of Large Language Models (https://arxiv.org/abs/2409.12739)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 Edu-Values를 제안하여, 중국 교육 관련 가치와 더불어 LLMs(대형 언어 모델)의 정렬 능력을 평가하는 최초의 벤치마크로 자리잡았습니다. 기존 LLM 평가 방법에서 중국의 교육적 맥락이 충분히 다뤄지지 않았던 문제를 해결하고자 하였습니다.

- **Technical Details**: Edu-Values는 1,418개의 질문으로 구성되며, 질문의 유형은 객관식, 다중 모드 질문 응답, 주관적 분석, 역도식 질문 및 전통 중국 문화 관련 질문 등 다양합니다. 이 연구는 11개의 최신 LLM(SoTA)을 대상으로 하여 인적 평가 및 자동 평가를 수행하였습니다.

- **Performance Highlights**: 연구 결과 중국 LLM이 영어 LLM보다 월등한 성과를 내었으며, Qwen 2는 81.37의 점수로 1위를 차지했습니다. LLM은 주제 지식과 교수 능력 측면에서는 우수한 성과를 보였으나, 교사 전문 윤리 및 기본 역량 평가에서 어려움을 겪고 있는 것으로 나타났습니다.



### MEXMA: Token-level objectives improve sentence representations (https://arxiv.org/abs/2409.12737)
Comments:
          11 pages, 12 figures

- **What's New**: MEXMA라는 새로운 접근 방식을 제안하여 문장 수준(objectives)과 토큰 수준(objectives)의 목표를 동시에 통합하여 향상된 다국어 샘플 표현(multilingual sentence representation)을 생성합니다.

- **Technical Details**: MEXMA는 한 언어의 문장 표현을 이용하여 다른 언어의 마스킹된 토큰을 예측하고, 문장 표현과 모든 토큰이 인코더를 직접 업데이트하도록 합니다. 이렇게 하기 위해 토큰과 문장 수준 목표를 결합하여 인코더를 효과적으로 업데이트합니다.

- **Performance Highlights**: MEXMA는 bitext mining, 분류(classification), 쌍 분류(pair classification) 등의 주요 작업에서 LaBSE와 SONAR를 포함한 최신 사전 훈련된 교차 언어 문장 인코더에 비해 우수한 성능을 보여주었습니다. MEXMA는 xsim++ 벤치마크에서 9.60%의 오류율과 MTEB와 SentEval에서 65.35%의 정확도를 달성했습니다.



### LLM-Measure: Generating Valid, Consistent, and Reproducible Text-Based Measures for Social Science Research (https://arxiv.org/abs/2409.12722)
- **What's New**: 이 논문은 사회 과학 연구에서 텍스트를 데이터로 사용하는 증가에 따라, 텍스트 기반 개념 측정을 생성하기 위한 효과적이고 재현 가능한 방법을 제시합니다. 새로운 방법은 대형 언어 모델(LLM)의 내부 숨겨진 상태를 활용하여 개념 벡터를 학습하고, 텍스트 데이터의 개념 값을 추정합니다.

- **Technical Details**: 제안된 방법은 LLM의 내부에서 목표 개념을 어떻게 표현하는지를 포착하는 개념 벡터를 학습하며, 그 후 텍스트의 LLM 숨겨진 상태를 개념 벡터에 투영하여 텍스트 데이터의 개념 값을 추정합니다.

- **Performance Highlights**: 세 가지 재현 연구를 통해 이 방법은 다양한 사회 과학 연구 맥락에서 텍스트 기반 측정을 높게 유효하고 일관되며 재현 가능하게 생성하는 데 효과적임을 입증하였습니다.



### Exploring Large Language Models for Product Attribute Value Identification (https://arxiv.org/abs/2409.12695)
- **What's New**: 이 논문은 e-커머스에서 제품 속성 값 식별(Product Attribute Value Identification, PAVI)을 위한 대형 언어 모델(large language models, LLMs)의 가능성을 탐구합니다. 기존의 방법들은 사전 훈련된 언어 모델(pre-trained language models, PLMs)인 BART 및 T5에 의존하여 많은 양의 특화된 훈련 데이터를 요구하고 새로운 속성에 일반화하기 어려웠습니다. 이 논문은 LLaMA 및 Mistral과 같은 LLMs를 데이터 효율적이고 강력한 대안으로 제안합니다.

- **Technical Details**: 저자들은 파라메트릭(parametric) 및 비파라메트릭(non-parametric) 지식을 활용한 다양한 전략을 제안하며, 제로샷(zero-shot) 환경에서의 한 단계와 두 단계 프롬프트 기반 접근 방식을 비교합니다. 특히, 사전 훈련된 T5 모델을 기반으로 한 밀집 시연 검색기(dense demonstration retriever)를 도입하고, 태스크 특화 지침을 명시적으로 훈련하기 위한 지시 지침 훈련(instruction fine-tuning)을 수행합니다.

- **Performance Highlights**: 두 개의 제품 벤치마크에서 수행한 광범위한 실험 결과, 두 단계 접근 방식이 제로샷 설정에서 성능을 크게 개선하며, 훈련 데이터를 사용할 경우 지시 지침 훈련이 추가적으로 성능을 향상시키는 것으로 나타났습니다. 이는 PAVI를 위한 대형 언어 모델 사용의 실질적인 이점을 보여줍니다.



### Connecting Ideas in 'Lower-Resource' Scenarios: NLP for National Varieties, Creoles and Other Low-resource Scenarios (https://arxiv.org/abs/2409.12683)
Comments:
          Selected as a full-day tutorial at COLING 2025

- **What's New**: 이번 튜토리얼은 '저자원' 언어 환경에서의 자연어 처리(NLP) 연구에 대한 새로운 접근 방식을 제시합니다. 이 연구는 방언, 크리올어 등 데이터가 부족한 언어를 다루는 데 중점을 두고 있으며, 이러한 언어 환경에서의 공통 과제 및 해결 전략을 조명합니다.

- **Technical Details**: 본 튜토리얼은 NLP의 최근 발전 및 기법을 다루며, Transformer 및 대형 언어 모델(Large Language Models), 방언과 크리올어의 언어적 다양성 탐구, 다중 작업 학습(Multi-task Learning)에 대한 교육을 포함합니다. 참가자들은 주어진 데이터 세트와 코드 샘플을 이용하여 실습할 기회를 가집니다.

- **Performance Highlights**: 참가자들은 방언 및 저자원 언어에 대한 이해도와 생성 기술을 적용할 수 있도록 훈련받으며, 기법을 개발하거나 연구에 적용할 수 있는 역량을 기릅니다. 또한, 대화형 학습과 데이터 증강(Data Augmentation) 기법을 통해 발생할 수 있는 다양한 문제와 도전 과제를 이해할 것입니다.



### Text2Traj2Text: Learning-by-Synthesis Framework for Contextual Captioning of Human Movement Trajectories (https://arxiv.org/abs/2409.12670)
Comments:
          To appear in the International Natural Language Generation Conference (INLG 2024)

- **What's New**: 이 논문에서는 쇼핑객의 궤적 데이터에 대한 가능한 맥락을 캡션화하는 새로운 학습 프레임워크인 Text2Traj2Text를 제시합니다. 이 연구는 타겟 광고 및 재고 관리와 같은 다양한 소매 애플리케이션에 영향을 미칠 것입니다.

- **Technical Details**: Text2Traj2Text는 두 단계로 구성됩니다: Text2Traj(데이터 합성)와 Traj2Text(모델 미세 조정). Text2Traj 단계에서는 대규모 언어 모델(LLMs)을 활용하여 실제적이고 다양한 컨텍스트 캡션과 매장 맵의 궤적을 합성합니다. Traj2Text 단계에서는 이 합성된 데이터로 미세 조정된 캡션 모델을 구축합니다.

- **Performance Highlights**: 체계적인 평가를 통해, LLMs에 의한 다양한 데이터 합성이 실제 인간 궤적과 생성된 캡션에 대한 잘 일반화될 수 있음을 보여주었습니다. 또한, ROUGE 및 BERT Score 지표에서 GPT-3.5, GPT-4, Llama2와 같은 기존 LLM 서비스보다 우수한 성능을 기록하였습니다.



### Exploring the topics, sentiments and hate speech in the Spanish information environmen (https://arxiv.org/abs/2409.12658)
Comments:
          24 pages

- **What's New**: 이 연구는 스페인 언론의 뉴스에 대한 공공 반응을 분석하여 주제, 감정 및 증오 발언의 유행을 탐구한 첫 번째 연구입니다. 총 337,807개의 응답 메시지(웹사이트 댓글 및 트윗)를 조사하여 다양한 유형의 증오 발언을 식별하고 이를 부정적, 중립적, 긍정적 감정으로 분류했습니다.

- **Technical Details**: 이 연구에서는 81개의 주제를 추출하기 위해 BERTopic이라는 비지도 학습 프레임워크를 사용하였고, 주제들은 대규모 언어 모델(LLMs)의 도움으로 수동으로 명명되어 9개의 주요 카테고리로 그룹화되었습니다. 감정 분석과 주제 모델링을 통해 공공 반응의 주요 동향과 감정 경향을 파악합니다.

- **Performance Highlights**: 조사 결과, 공공 반응의 62.7%가 부정적이며 28.57%는 중립적, 8.73%는 긍정적이었습니다. 주요 논의 주제는 사회적 문제(22.22%), 표현 및 속어(20.35%), 정치적 문제(11.80%)이며, 증오 발언 비율은 상대적으로 낮은 3.98%였지만 온라인 반응의 높은 독성이 확인되었습니다. 이는 대화 표현, 성별, 페미니즘, COVID-19와 관련된 독성 서사와 관련이 있습니다.



### Efficient Performance Tracking: Leveraging Large Language Models for Automated Construction of Scientific Leaderboards (https://arxiv.org/abs/2409.12656)
- **What's New**: 본 연구에서는 자동 리더보드 생성의 필요성을 강조하며, 기계 학습 및 자연어 처리 분야에서의 새로운 데이터셋 SciLead를 소개합니다. SciLead는 수작업으로 선별된 과학 리더보드 데이터셋으로, 기존의 불완전하고 부정확한 정보 문제를 해결하고자 합니다.

- **Technical Details**: SciLead는 43개의 NLP 논문에서 파생된 27개의 리더보드를 포함하고 있습니다. 이 데이터셋은 태스크(Task), 데이터셋(Dataset), 평가 메트릭(Metric) 삼중항(TDM triples)에 대한 포괄적인 주석을 제공하며, 이는 리더보드 구축과정에서 필수적입니다. 연구팀은 리더보드를 구축하기 위해 다양한 LLM 기반의 프레임워크를 개발했습니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 각기 다른 실제 시나리오에 잘 적응하며, 특히 LLM들은 TDM 삼중항을 성공적으로 식별하는 데 강점을 보였으나, 출판물에서 결과 값을 추출하는 데 어려움이 있음을 발견했습니다. 이를 통해 리더보드의 정확도를 높이는 데 기여할 수 있는 가능성을 보여주었습니다.



### Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries (https://arxiv.org/abs/2409.12640)
- **What's New**: 이 연구에서는 대형 언어 모델의 긴 컨텍스트 추론 평가를 위한 새로운 시스템인 Michelangelo를 소개합니다. 이 시스템은 모델이 단순히 정보를 검색하는 것 이상의 작업을 수행할 수 있는지를 평가하도록 설계되었습니다.

- **Technical Details**: Michelangelo는 Latent Structure Queries (LSQ) 프레임워크를 사용하여 모델이 컨텍스트의 관련 없는 정보를 "조각 조각 깎아내고" 중요한 구조를 드러낼 수 있는지 평가합니다. 이 프레임워크는 복잡한 평가 작업을 생성하는 데 활용되며, 각 평가 작업은 명백하게 자연어 및 코드 기반 설정에 위치하고 있습니다.

- **Performance Highlights**: Michelangelo는 여러 최첨단 모델에서 1M 컨텍스트 길이로 성능을 평가했으며, GPT 및 Claude 모델이 128K 컨텍스트에 비해 괜찮은 성능을 보인 반면, Gemini 모델은 1M 컨텍스트에 대해 비고차가 있는 일반화 능력을 보여주었습니다. 그러나 모든 평가 모델에서 추론 과제가 어려운 것으로 인해 초기 성능이 급격히 떨어지는 경향이 있음을 발견하였습니다.



### CamelEval: Advancing Culturally Aligned Arabic Language Models and Benchmarks (https://arxiv.org/abs/2409.12623)
- **What's New**: 새로운 아랍어-영어 이중 언어 대형 언어 모델인 Juhaina를 소개하며, 아랍어 사용자들의 가치와 선호에 부합하도록 설계되었습니다. Juhaina는 9.24억 개의 파라미터를 가지고 있으며, 8,192개의 토큰을 처리할 수 있는 컨텍스트 윈도우를 지원합니다.

- **Technical Details**: Juhaina는 디코더 전용의 밀집 변환기 모델로, Gemma 2 기반의 개방형 LLM 모델을 포스트 트레이닝하여 개발되었습니다. CamelEval은 아랍어 LLM의 대화 능력과 지시 준수 능력을 평가하기 위해 설계된 새로운 벤치마크입니다.

- **Performance Highlights**: Juhaina는 아랍어로 유용한 응답을 생성하고, 지역에 대한 사실적으로 정확한 정보를 제공하며, 미묘한 문화적 측면을 이해하는 데 있어 비교 가능한 크기의 기존 LLM보다 우수한 성과를 보였습니다.



### Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning (https://arxiv.org/abs/2409.12618)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 응답을 개선하기 위한 새로운 프레임워크인 Iteration of Thought (IoT)를 소개합니다.

- **Technical Details**: IoT 프레임워크는 세 가지 구성 요소로 이루어져 있습니다: (1) 내적 대화 에이전트(Inner Dialogue Agent, IDA)는 맥락에 따른 유용한 프롬프트를 생성하고, (2) LLM 에이전트(LLM Agent, LLMA)는 이러한 프롬프트를 처리하여 응답을 다듬으며, (3) 두 컴포넌트 간의 반복 프롬프트 루프가 대화를 수행합니다. 또한 자율 반복(Autonomous Iteration of Thought, AIoT)과 유도 반복(Guided Iteration of Thought, GIoT)의 두 가지 변형을 제공합니다.

- **Performance Highlights**: IoT는 GPQA, Game of 24, Mini Crosswords, HotpotQA 데이터셋에서 다양한 복잡한 추론 작업을 통해 CoT와 비교했을 때 상당한 개선을 보여주며, LLM의 자율 응답 개선을 위한 실용적인 패러다임임을 증명합니다.



### Enhancing SLM via ChatGPT and Dataset Augmentation (https://arxiv.org/abs/2409.12599)
- **What's New**: 이 논문은 작은 언어 모델(SLM)을 ChatGPT-3.5-Turbo를 통해 전략적으로 데이터셋을 증강하여 향상시키는 방법을 탐구합니다. 주된 초점은 인간 주석 없이도 큰 언어 모델(LLM)과 작은 언어 모델 간의 성능 격차를 줄이는 것입니다.

- **Technical Details**: 지식 증류(Knowledge Distillation) 기반 기술과 합성 데이터셋 증강(Synthetic Dataset Augmentation)을 활용하여 ANLI 데이터셋의 자연어 추론(NLI) 능력을 향상시키고 있습니다. 정보 추출(Information Extraction) 및 정보 기반 추론(Informed Reasoning)의 두 가지 접근 방식을 사용하여 데이터셋을 증강합니다. T5-Small 모델을 이러한 증강된 데이터셋으로 미세 조정(Fine-tuning)하고 성능을 평가합니다.

- **Performance Highlights**: 합성 근거(Synthetic Rationales)의 통합은 자연어 이해(NLU) 능력을 향상시키고, ANLI 데이터셋에서 분류 정확도가 각각 1.3% 및 2.3% 향상되는 결과를 보여줍니다. 이는 보다 복잡한 작업에서 작은 모델의 성능을 개선하고, 효율적인 미세 조정 방법을 제시합니다.



### Efficient Knowledge Distillation: Empowering Small Language Models with Teacher Model Insights (https://arxiv.org/abs/2409.12586)
- **What's New**: 본 논문에서는 소형 언어 모델의 성능을 향상시키기 위한 새로운 지식 증류(knowledge distillation) 방법을 소개합니다. 이 방법은 약 30억 개의 매개변수를 가진 교사 모델을 활용하여 모델의 의사결정 과정에서 가장 영향을 미치는 토큰을 식별하고, 이를 학생 모델에 rationales로 제공하는 방식입니다.

- **Technical Details**: 우리의 방법은 gradient 기반의 지식 증류 기법을 사용하며, saliency map을 통해 주요 토큰을 추출합니다. 이 과정에서 입력에 대해 각 토큰의 기여도를 계산하고, 그 중에서 중요성이 높은 토큰을 학생 모델에 제공합니다. 이를 통해 학생 모델의 학습 과정 및 성능을 향상시키는 목표를 가지고 있습니다.

- **Performance Highlights**: 논문 실험에서 제안하는 방법은 4개의 다양한 데이터 세트에서 표준 fine-tuning 방법 및 최신 지식 증류 모델보다 우수한 성능을 보여주었습니다. 특히, 정답의 일부인 토큰이 68%의 경우에 해당되어, 주어진 입력과 정답 간의 관계를 효과적으로 반영하고 있음을 나타냅니다.



### RAD-Bench: Evaluating Large Language Models Capabilities in Retrieval Augmented Dialogues (https://arxiv.org/abs/2409.12558)
- **What's New**: RAD-Bench (Retrieval Augmented Dialogue)는 다중 턴 대화에서의 외부 검색 메커니즘을 평가하기 위한 새로운 벤치마크로, LLMs의 능력을 신뢰성 있는 검색 결과를 활용하여 향상된 응답을 효과적으로 생성하는지 측정합니다.

- **Technical Details**: RAD-Bench는 LLMs의 두 가지 주요 능력인 Retrieval Synthesis(검색 통합)와 Retrieval Reasoning(검색 추론)을 평가합니다. 벤치마크 샘플은 3턴 질문과 함께 제공된 검색된 컨텍스트로 구성되며, 각 턴에서 사용자 의도가 변하거나 추가 조건이 제시될 때 LLMs가 어떻게 응답하는지를 평가합니다. 89개의 샘플이 포함되어 있으며, 각 샘플은 267 턴의 평가를 제공합니다.

- **Performance Highlights**: LGMs의 성능은 다중 턴에서 새로운 조건 또는 의도가 도입될 때 하락합니다. RAD-Bench는 여러 LLMs(GPT-4o, Llama 등)의 성능을 분석하여, 유사한 다중 턴 대화 성능을 보이는 모델들이 검색 후 대화에서는 동일하게 수행되지 않음을 보여줍니다.



### Enhancing Knowledge Distillation of Large Language Models through Efficient Multi-Modal Distribution Alignmen (https://arxiv.org/abs/2409.12545)
Comments:
          18 pages

- **What's New**: 본 논문에서는 Ranking Loss 기반 지식 증류(RLKD) 방식을 제안하여 큰 언어 모델(LLM)로부터 학생 모델이 다중 모드 확률 분포를 더 잘 학습할 수 있도록 돕고, 다양한 다운스트림 작업에서의 성능 향상을 이끌어냄을 보여준다.

- **Technical Details**: RLKD는 교육자와 학생 모델 간의 예측 결과 순위의 일관성을 유지하도록 설계됐다. 특히, Spearman의 순위 상관계수(SRCC)를 사용하여 peak 예측 순서의 일관성을 최적화하고, 기존 증류 목표와의 뛰어난 호환성을 보장한다.

- **Performance Highlights**: 실험 결과, proposed method는 학생 모델이 다중 모드 분포를 예측하는 능력을 효과적으로 향상시켰으며, 다양한 데이터셋에서의 다운스트림 학습 작업에서 유의미한 성능 개선을 보여주었다.



### Profiling Patient Transcript Using Large Language Model Reasoning Augmentation for Alzheimer's Disease Detection (https://arxiv.org/abs/2409.12541)
Comments:
          accepted to EMBC 2024

- **What's New**: 이 연구는 Alzheimer’s disease(AD) 감지의 새로운 접근 방식을 제안합니다. LLM(large language model)을 활용하여 환자 수준의 언어 프로파일링을 수행하고, 이를 통해 언어 결손 속성을 체계적으로 도출하여 AD 감지 기법을 개선합니다.

- **Technical Details**: 제안된 방법에서는 13개의 언어 결손 속성을 통해 환자 언어 프로파일을 생성하고 이를 Albert 모델에 통합하여 AD를 감지합니다. LLM을 기반으로 한 발화 세션에서 언어 결손 정보를 추출하는 방식으로, 텍스트의 특성을 고려한 모델링을 향상시킵니다.

- **Performance Highlights**: ADReSS 데이터셋을 사용한 실험에서, 제안한 방법은 기존의 모델보다 8.51% ACC 및 8.34% F1 점수 향상을 보였습니다. 이는 언어 결손 속성의 효용성을 증명합니다.



### Should RAG Chatbots Forget Unimportant Conversations? Exploring Importance and Forgetting with Psychological Insights (https://arxiv.org/abs/2409.12524)
- **What's New**: LUFY(감정 기반의 대화 메모리 관리)는 대화의 10% 미만의 기억만 남기고 비중이 높은 기억을 우선시함으로써 사용자 경험을 향상시킨다는 새로운 접근 방식을 제안합니다.

- **Technical Details**: LUFY는 대화에서의 감정 생성과 기억 유지에 관한 심리학적 통찰을 바탕으로 6개의 메모리 메트릭을 사용하여 중요한 발화를 식별하고 정보 회수 및 망각 모듈에서 가중 점수를 활용합니다.

- **Performance Highlights**: LUFY는 기존 Naive RAG 시스템 대비 정보 회수의 정확성을 17% 이상 향상시켰으며, 참가자들이 대화한 내용의 절반 이상을 망각하는 방식으로 사용자 경험을 극적으로 개선했습니다.



### Exploring and Enhancing the Transfer of Distribution in Knowledge Distillation for Autoregressive Language Models (https://arxiv.org/abs/2409.12512)
- **What's New**: 이 논문은 Online Knowledge Distillation (OKD) 방법을 소개하며, 기존의 Knowledge Distillation (KD) 방식에서의 한계를 극복하기 위해 교사 모델의 고정된 분포를 동적으로 업데이트하는 방안을 제시합니다. 이를 통해 훈련 시간과 성능 향상을 동시에 이룰 수 있음을 보입니다.

- **Technical Details**: OKD는 학생 모델과 동시에 훈련하기 위해 작은 온라인 모듈을 통합한 교사 네트워크를 사용합니다. 이를 통해 on-policy 샘플링의 필요성을 없애고, 훈련 중 최소한의 매개변수 수정을 요구합니다. 일반적으로 사용되는 Reverse KL divergence (RKL) 방식의 한계를 분석하고, 교사 모델의 분포를 변화시켜 학생 모델의 크기에 맞춰 동적으로 조정할 수 있는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, OKD는 다양한 모델 아키텍처와 크기에 걸쳐 기존 선도 방법들보다 동등하거나 더 나은 성과를 달성하였으며, 최대 4배의 훈련 시간 단축에도 성공하였습니다. 또한, GKD와 MiniLLM 같은 최신 방법들과 비교할 때 더 뛰어난 결과를 보였습니다.



### LLMR: Knowledge Distillation with a Large Language Model-Induced Reward (https://arxiv.org/abs/2409.12500)
Comments:
          Accepted by LERC COLING 2024

- **What's New**: 이 논문에서는 리소스 제약 환경에서의 대형 언어 모델의 활용 가능성을 높이기 위해 LLMR이라는 새로운 지식 증류(knowledge distillation) 방법을 제안했습니다.

- **Technical Details**: LLMR은 대형 언어 모델에서 유도된 보상 함수(reward function)를 기반으로 하며, 대화 생성(dialogue generation) 및 요약(summarization) 작업에 대한 여러 데이터 세트에서 실험이 이루어졌습니다.

- **Performance Highlights**: 실험 결과, LLMR 접근법은 전통적인 지식 증류 방법에 비해 다양한 작업 및 데이터 세트에서 일관되게 우수한 성능을 보여주었습니다.



### CritiPrefill: A Segment-wise Criticality-based Approach for Prefilling Acceleration in LLMs (https://arxiv.org/abs/2409.12490)
- **What's New**: 이 논문에서는 긴 맥락 작업에서의 프리필링(prefilling) 단계 비효율성을 해결하기 위해, 쿼리 토큰의 중요도(locality in query criticality)를 기반으로 한 CritiPrefill 기법을 제안합니다.

- **Technical Details**: CritiPrefill 방법은 입력 시퀀스의 쿼리와 Key-Value (KV) 캐시를 세그먼트(segment)로 나누어 쿼리 중요도를 추정하며, 자가 주의(self-attention) 메커니즘에서 쿼리 세그먼트와 캐시 블록 간 비중요 계산을 가지치기(pruning)하여 프리필링 과정을 가속화합니다.

- **Performance Highlights**: Llama3-8B 모델에서 최대 2.7배, Yi-9B 모델에서 최대 3.0배의 속도 향상을 보여주며, 품질 저하가 최소화되었습니다.



### AutoMode-ASR: Learning to Select ASR Systems for Better Quality and Cos (https://arxiv.org/abs/2409.12476)
Comments:
          SPECOM 2024 Conference

- **What's New**: 새로운 프레임워크 AutoMode-ASR는 여러 개의 ASR 시스템을 효과적으로 통합하여 전반적인 전사(transcription) 품질을 향상시키고 비용을 최적화합니다. 이 시스템은 각 오디오 입력(segment)에 대해 최적의 ASR 시스템을 선택할 수 있도록 결정 모델을 훈련합니다.

- **Technical Details**: AutoMode-ASR는 이진 분류기(binary classifiers)를 사용하여 두 개의 시스템 간의 선호도를 결정하고, 오디오 임베딩(audio embeddings), 품질 추정(quality estimation), 신호 특성(signal properties)과 같은 다양한 기능(features)을 장착합니다. 이 프레임워크는 상장 모델 코드 변경 없이 상업적으로나 오픈소스 ASR 시스템과 호환됩니다.

- **Performance Highlights**: 실험 결과 AutoMode-ASR는 WER(Word Error Rate)를 16.2% 상대적으로 감소시키고, 비용을 65% 절감하며, 속도는 75% 향상시키는 것을 보여줍니다.



### Familiarity-aware Evidence Compression for Retrieval Augmented Generation (https://arxiv.org/abs/2409.12468)
- **What's New**: 본 논문에서는 FaviComp (Familiarity-aware Evidence Compression)이라는 새로운 방법을 제안합니다. 이 방법은 외부에서 가져온 증거(evidence)가 최종 모델(target model)에게 보다 친숙하게 만들어질 수 있도록 하며, 파라메트릭 지식(parametric knowledge)과 비파라메트릭 지식(non-parametric knowledge)을 통합합니다.

- **Technical Details**: FaviComp는 모델의 파라메트릭 지식을 필요한 만큼 통합하면서, 압축된 증거(evidence)의 혼란도를 낮춰(target model의 perplexity 감소) 사용될 수 있도록 합니다. 두 개의 언어 모델(LM), 즉 압축 모델(compression model)과 목표 모델(target model)의 토큰 확률(token probabilities)을 조합하여 새로운 컨텍스트(context)를 생성합니다. 이 과정을 통해 FaviComp는 보다 효율적인 정보 통합을 이끌어내어, 복잡한 작업에서도 본질적인 정보를 유지할 수 있도록 합니다.

- **Performance Highlights**: FaviComp는 다섯 개의 공개 도메인 QA 데이터셋에서 기존의 모든 기준선(baselines)을 초과하는 성능을 보여주었습니다. 높은 압축 비율을 유지하면서, 두 개의 다운스트림 LMs과 함께 작동하여 파라메트릭 및 비파라메트릭 지식의 효과적인 통합을 입증했습니다.



### CodePlan: Unlocking Reasoning Potential in Large Langauge Models by Scaling Code-form Planning (https://arxiv.org/abs/2409.12452)
- **What's New**: CODEPLAN이라는 새로운 프로세스를 제안하며, 이는 대형 언어 모델(LLMs)이 구조화된 계획을 생성하고 따르는 능력을 향상시킵니다. 이 프로세스는 고급 추론 과정을 코드 형식으로 설정하여 복잡한 문제 해결에서의 성능을 개선합니다.

- **Technical Details**: CODEPLAN은 코드의 구조적이고 다재다능한 성질을 활용하여 정교한 추론의 풍부한 의미론과 제어 흐름을 포착합니다. 또한, 대규모 데이터셋(2M 예시)을 이용하여 훈련되며, 자연어 데이터의 암시적 계획 신호를 극복하기 위한 두 단계로 구성된 생성 과정을 활용합니다: 계획과 표면 실현.

- **Performance Highlights**: CODEPLAN은 13개의 복잡한 다단계 추론 벤치마크에서 직접적인 응답 생성을 할 때보다 평균 25.1%의 성능 향상을 보여줍니다. 특히, 문제 복잡성이 증가할수록 성능이 향상되는 경향도 보입니다.



### Incremental and Data-Efficient Concept Formation to Support Masked Word Prediction (https://arxiv.org/abs/2409.12440)
Comments:
          Accepted by the Eleventh Annual Conference on Advances in Cognitive Systems

- **What's New**: 이 논문에서는 Cobweb4L이라는 새로운 접근 방식을 소개하며, 이는 마스킹된 단어 예측(masked word prediction)을 지원하고 효율적인 언어 모델 학습을 가능하게 합니다. 이 시스템은 확률적 개념의 계층 구조를 학습하는 Cobweb의 기능을 발전시킵니다.

- **Technical Details**: Cobweb4L은 개념 유틸리티(category utility)의 정보 이론적 변형을 사용하고, 여러 개념을 활용하여 예측을 생성하는 새로운 성능 메커니즘(performance mechanism)을 도입합니다. 이 메커니즘은 단일 노드만을 사용하여 예측을 생성하는 기존 Cobweb 성능 메커니즘보다 뛰어난 성과를 보여줍니다.

- **Performance Highlights**: Cobweb4L은 신속하게 학습하며, Word2Vec과 유사하거나 이를 초월하는 성능을 달성합니다. 또한 Cobweb4L과 Word2Vec은 BERT보다 적은 훈련 데이터를 사용하여 같은 작업에서 더 높은 성능을 보입니다.



### Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data (https://arxiv.org/abs/2409.12437)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 복잡한 논리적 추론(task) 처리에서의 한계를 극복하기 위해 그래프 기반의 합성 추론 데이터(synthetic reasoning data)를 사용한 연구가 발표되었습니다. 이 데이터는 LLM의 추론 능력을 향상시키는 데 효과적이며, 기존의 표준 평가 기준에서도 성능을 유지하는 동시에 효과적인 훈련 신호로 작용합니다.

- **Technical Details**: 이 연구는 두 가지 자연어 추론 작업인 유도 추론(inductive reasoning)과 공간적 추론(spatial reasoning)에 대해 그래프 기반의 합성 데이터를 생성하는 새로운 알고리즘과 프롬프트(prompts) 전략을 제안합니다. 제안된 방법은 랜덤 워크(random walk) 샘플링 알고리즘을 통해 그래프에서 서브그래프를 추출하여, LLMs가 특정 작업에 적합하게 조정될 수 있도록 합니다.

- **Performance Highlights**: CLUTRR 및 StepGame이라는 두 가지 잘 알려진 벤치마크에서 수행된 실험 결과, 제안한 방법이 기존의 표준 프롬프트 및 훈련 방법에 비해 유의미한 성능 향상을 보였습니다. 연구 결과는 구조화된 데이터가 LLM의 추론 능력을 효과적으로 강화할 수 있음을 시사합니다.



### Linguistic Minimal Pairs Elicit Linguistic Similarity in Large Language Models (https://arxiv.org/abs/2409.12435)
Comments:
          Codes and data are available at this https URL

- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 내부 언어 표현을 탐구하기 위해 언어적 최소 쌍(linguistic minimal pairs)을 활용한 새로운 분석 기법을 소개합니다.

- **Technical Details**: 우리는 100개 이상의 LLM과 15만 개의 최소 쌍 데이터로 구성된 대규모 실험을 진행하였으며, 이 과정에서 LLM의  활성화 차이를 통해 언어적 유사성을 측정했습니다. 이 분석은 언어적 유사성이 LLM 간의 일관성, 이론적 분류와의 관계, 의미적 맥락에 대한 의존성, 관련 현상의 언어 간 정렬 여부 등 4가지 주요 측면에서 드러났습니다.

- **Performance Highlights**: 연구 결과, 첫째, 언어적 유사성은 훈련 데이터에 의해 크게 영향을 받으며, 자원이 풍부한 언어에서 LM 간의 합의가 높다는 점을 발견했습니다. 둘째, 언어적 유사성은 세분화된 이론적 언어 범주와는 강하게 일치하지만 더 넓은 범주와는 약한 상관관계를 보였습니다. 셋째, 언어적 유사성은 의미적 유사성과 약한 상관관계를 보여서 맥락 의존적 성격을 나타냅니다. 넷째, 다양한 언어에서 관련된 현상에 대한 이해가 제한적이라는 점도 확인되었습니다.



### Zero-to-Strong Generalization: Eliciting Strong Capabilities of Large Language Models Iteratively without Gold Labels (https://arxiv.org/abs/2409.12425)
Comments:
          15 pages

- **What's New**: 이 논문은 라벨이 없는 데이터를 활용하여 강력한 모델 기능을 이끌어내기 위한 새로운 패러다임인 zero-to-strong generalization을 제안합니다. 이 방법은 LLMs를 반복적으로 프롬프트하여 라벨을 생성하고, 높은 품질의 라벨을 필터링하여 사용하는 접근 방식을 채택합니다.

- **Technical Details**: Zero-to-strong generalization은 라벨이 없는 상황에서 LLMs의 뛰어난 성과를 달성하기 위한 반복적 프로세스를 포함합니다. 초기에는 무작위 또는 잘못된 사례로 LLM을 프롬프트하여 데이터를 라벨링하고, 신뢰 수준에 따라 새로운 사례를 선택하여 이 과정을 반복함으로써 성능을 향상시킵니다. 본 연구에서는 Meta-Llama-3-8B 및 Mistral-7B 모델을 이용하여 17개의 분류 작업, 2개의 극한 레이블 분류 작업 및 2개의 추론 작업에서 시험하였습니다.

- **Performance Highlights**: 실험 결과, zero-to-strong framework은 gold labels를 사용하는 in-context learning의 성과와 비슷하거나 더 나은 결과를 보여주었습니다. 이 방법은 특히 더 복잡한 작업 및 더 강한 모델에서 더욱 효과적이며, 다양한 모델 크기와 fine-tuning에서도 유용하게 작용합니다.



### Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation (https://arxiv.org/abs/2409.12411)
- **What's New**: 이 논문에서는 AgentCOT라는 새로운 LLM 기반의 자율 에이전트 프레임워크를 제안합니다. 이 프레임워크는 복잡한 문제를 해결하기 위해 다단계 LLM 생성을 통해 에이전트 스타일 방식으로 작업을 수행합니다. 또한, 추론 과정에서 단계의 인덱스를 통합하여 복잡한 추론 로직을 위한 그래프 구조를 형성합니다.

- **Technical Details**: AgentCOT는 각 단계에서 행동을 선택하고 수행하여 중간 결과를 도출하며, 결론을 위한 증거를 제공합니다. 논문은 잘 정의된 행동 세트에서 특정 행동을 선택하고, 해당 행동에 대한 설명과 함께 중간 결과 및 증거를 제시하여 설명 가능성을 높입니다.

- **Performance Highlights**: AgentCOT는 6개의 공통 벤치마크에서 실험을 수행한 결과, 현재의 경쟁적인 방법들에 비해 상당한 성능 향상을 보였습니다. 이 연구는AgentCOT가 다양한 데이터 세트 및 모델에서 효과적이라는 것을 보여줍니다.



### Mutual Information-based Representations Disentanglement for Unaligned Multimodal Language Sequences (https://arxiv.org/abs/2409.12408)
Comments:
          31 pages, 8 figures

- **What's New**: 본 논문에서는 다양한 모달리티(multiple modalities)에서 비선형 상관관계를 제거하면서 정보 중복을 감소시키는 새로운 접근 방식을 제안합니다. 이 방법은 Mutual Information 기반의 Representation Disentanglement (MIRD) 방식으로, 전통적인 모달리티 분리 방법보다 효과적으로 모달리티 무관한 표현을 통합하여 더 나은 성능을 보입니다.

- **Technical Details**: MIRD 방법은 단일한 모달리티 무관한 표현을 공동으로 학습하는 새로운 프레임워크를 제공하며, 상관관계 측정을 위해 상호 정보(mutual information) 최소화 제약을 사용합니다. 이를 통해 각 모달리티의 비선형 상관관계를 제거하고 정보 중복성을 감소시킵니다. 또한, 레이블이 없는 데이터(unlabeled data)를 활용하여 상호 정보를 정확히 추정하고, 모델의 성능을 향상시킵니다.

- **Performance Highlights**: MIRD 접근법은 여러 벤치마크 데이터셋에서 실험을 통해 기존 모델 대비 더 높은 성능을 보여주었으며, 최신 기술(state-of-the-art) 성능을 달성했습니다. 시각화를 통해 각 모달리티의 표현이 효과적으로 분리되었음을 입증했습니다.



### Preference Alignment Improves Language Model-Based TTS (https://arxiv.org/abs/2409.12403)
- **What's New**: 최근 텍스트-음성 변환(TTS) 분야에서 언어 모델 (LM) 기반 시스템의 경쟁력 있는 성능을 보여주고 있으며, 이를 더욱 최적화하기 위한 선호 정렬(preference alignment) 알고리즘이 개발되고 있습니다. 본 연구는 Direct Preference Optimization (DPO) 알고리즘이 LM 기반 TTS에 미치는 영향을 실증적으로 평가합니다.

- **Technical Details**: TTS는 주어진 조건(예: 텍스트)에서 인간의 음성을 합성하는 작업입니다. 본 연구는 1.15B 매개변수를 가진 LM 기반 TTS 모델을 사용하여 선호 정렬 알고리즘의 적용이 음성 인지성(intelligibility), 화자 유사도(speaker similarity), 주관적 평가 점수(proxy subjective evaluation scores)를 일관성 있게 향상시키는 것을 증명하였습니다. 특히 이 두 가지 지표는 특정 평가에서 인간 음성을 초과하기도 하였습니다.

- **Performance Highlights**: 평가 결과, 선호 정렬 알고리즘의 적용이 TTS 모델의 성능을 기하급수적으로 향상시켰고, 이는 고품질의 자연스러운 음성을 생성하는 데 기여하였습니다. 본 연구는 저자원 환경에서도 적용 가능함을 보여주며, 다양한 도메인에서 일반화되는 능력도 확인하였습니다.



### Small Language Models are Equation Reasoners (https://arxiv.org/abs/2409.12393)
Comments:
          6 pages, 2 figures

- **What's New**: 본 논문은 소형 언어 모델(sLM)이 산술 추론 작업에서 성능이 낮은 이유를 조사하고 자연어 형식의 변동성이 이러한 모델에 높은 모호성을 유발한다고 가설을 세웠습니다. 또한, 수식 전용 형식(equation-only format)을 도입하여 모델의 성능을 향상시키는 실험을 진행하였습니다.

- **Technical Details**: 수식 전용 형식은 산술 문제를 자연어 형식이 아닌 수학적 방정식으로 통합하여 모델의 모호성을 줄이는 방법입니다. 이 실험은 T5 모델을 사용하여 수행되었으며, T5-Tiny와 같은 소형 모델에서 특히 효과적임을 입증하였습니다.

- **Performance Highlights**: T5 모델의 경우, 자연어 형식에 비해 수식 전용 형식을 사용했을 때 성능이 일관되게 향상되었습니다. 예를 들어, T5-base 모델은 정확도가 13%에서 17%로 상승하였고, T5-small 모델은 10%에서 14%로 향상되었습니다. 이 결과는 소형 모델의 경우 수식 사용이 널리 보편적인 자연어 형식보다 더 효과적이라는 것을 보여줍니다.



### Measuring Sound Symbolism in Audio-visual Models (https://arxiv.org/abs/2409.12306)
Comments:
          SLT 2024

- **What's New**: 최근 오디오-비주얼(pre-trained audio-visual) 모델들이 다양한 오디오-비주얼 과제에서 뛰어난 성능을 보여주고 있습니다. 본 연구는 이러한 모델들이 소리와 시각적 표현 간의 비임의적 연관성을 나타내는지 조사하였습니다. 이를 위해 합성된 이미지와 오디오 샘플로 구성된 특별한 데이터셋을 개발했습니다.

- **Technical Details**: 이 연구에서는 제로샷(zero-shot) 설정에서 비모수(non-parametric) 접근 방식을 사용하여 오디오-비주얼 모델들을 평가했습니다. 실험 결과, 특정 오디오-비주얼 모델에서 모델 출력과 확립된 소리 상징성 패턴 간의 유의미한 상관관계를 발견했습니다. 특히, 언어 데이터로 훈련된 모델들이 음성 데이터와 강한 상관관계를 보였습니다.

- **Performance Highlights**: 모델들이 다른 모양의 시각적 자극과 오디오를 랜덤 이상으로 잘 그룹화할 수 있음을 보여주었고, 전반적으로 오디오-비주얼 모델이 순수 텍스트 비전-언어 모델보다 더 두드러진 소리 상징성 효과를 나타냈습니다. 이러한 결과는 기존 심리학 문헌과의 시너지를 나타내며 인간 언어의 비임의성을 더욱 지지합니다.



### Making Large Language Models into World Models with Precondition and Effect Knowledg (https://arxiv.org/abs/2409.12278)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 세계 모델처럼 작동하도록 유도할 수 있는 가능성을 탐구합니다. 특히, LLMs가 행동의 적용 가능성과 실행 후 세계 상태 예측이라는 핵심 기능을 수행할 수 있음을 보여줍니다.

- **Technical Details**: 두 개의 LLM을 미세 조정하여 전제 조건 예측과 효과 예측을 수행합니다. 이 과정에서 합성 데이터 생성 기법을 활용하여 LLM들이 상호 작용할 수 있도록 설계합니다. 연구는 인간 참가자 연구를 통해 이 모델들이 생성한 전제 조건과 효과 지식이 인간의 세계 역학 이해와 일치하는지를 검증합니다.

- **Performance Highlights**: 모델들은 높은 품질의 전제 조건/효과 코퍼스를 생성하고, 유능한 세계 모델을 구축함으로써 정확성을 입증했습니다. 인간 평가 및 자동화된 평가 모두에서 효과적인 결과를 나타냈습니다.



### MQA-KEAL: Multi-hop Question Answering under Knowledge Editing for Arabic Languag (https://arxiv.org/abs/2409.12257)
- **What's New**: 이 논문에서는 아랍어에 대한 지식 편집(Knowledge Editing)을 활용한 다중 홉 질문 응답(Multi-hop Question Answering, MQA) 시스템인 MQA-KEAL을 제안합니다. 이는 기존 LLM에서 영어 중심으로 이루어졌던 연구의 한계를 극복하기 위한 것입니다.

- **Technical Details**: MQA-KEAL은 외부 메모리에 저장된 구조화된 지식 단위로 지식 편집을 관리하며, 질문을 더 작은 하위 문제로 분해하는 작업 분해(task-decomposition) 기법을 사용합니다. 이후 각 하위 문제에 대해 외부 메모리와 LLM을 반복적으로 쿼리하여 최종 응답을 생성합니다.

- **Performance Highlights**: 실험 결과, MQA-KEAL은 기존의 기준 모델들과 비교했을 때 상당한 성능 향상을 보여주었습니다.



### ARTICLE: Annotator Reliability Through In-Context Learning (https://arxiv.org/abs/2409.12218)
- **What's New**: 새로운 메소드인 ARTICLE을 통해 주석자 품질을 자기 일관성(self-consistency)을 통해 추정하고, 이 방법의 성능을 기존 방법들과 비교하였습니다.

- **Technical Details**: ARTICLE은 두 단계의 프레임워크로 구성되어 있으며, 첫 번째 단계에서는 일관성이 결여된 주석자들을 식별하여 데이터셋에서 제거합니다. 그 후, 일관된 주석자들의 응답을 바탕으로 집단적인 오류 인식 감각을 모델링합니다.

- **Performance Highlights**: 전통적인 방법들과 비교했을 때, ARTICLE은 주석자의 신뢰성을 강력하게 식별할 수 있는 방법으로 데이터 품질 향상에 기여합니다.



### MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines (https://arxiv.org/abs/2409.12959)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 MMSearch-Engine이라는 다중모달(AI search engine) 검색 엔진 파이프라인을 제안하여 대규모 다중모달 모델(LMMs)들이 검색 능력을 발휘할 수 있도록 지원합니다. 이를 통해 텍스트와 이미지를 포함한 다양한 사용자 쿼리를 처리하는 새로운 방식을 제안합니다.

- **Technical Details**: MMSearch-Engine은 이미지가 포함된 쿼리에 대해 웹 검색을 수행하고, 검색된 결과를 재정렬하는 과정을 포함한 다단계 상호작용을 통해 작동합니다. 또한 MMSearch라는 포괄적인 평가 기준을 통해 LMMs의 다중모달 검색 성능을 평가하며, 14개 분야에 걸쳐 수집된 300개의 데이터를 사용합니다.

- **Performance Highlights**: GPT-4o는 MMSearch-Engine과 함께 사용할 경우, 상업적으로 우수한 검색 엔진인 Perplexity Pro보다 뛰어난 성능을 보였습니다. 그러나 현재의 LMM들은 여전히 다중모달 검색 작업에서 일반화하는 데 어려움을 겪고 있으며, 정답을 올바르게 식별하는 능력에 제약이 있습니다.



### Re-Introducing LayerNorm: Geometric Meaning, Irreversibility and a Comparative Study with RMSNorm (https://arxiv.org/abs/2409.12951)
- **What's New**: 이번 논문에서는 Transformer 아키텍처에서의 Layer Normalization의 기하학적 의미에 대해 다루고 있습니다. LayerNorm이 숨겨진 벡터의 방향과 크기에 미치는 영향을 분석하며, LayerNorm의 정의가 균일 벡터(uniform vector)와 본질적으로 연결되어 있음을 보여줍니다.

- **Technical Details**: LayerNorm은 벡터의 특정 성분을 제거하고 나머지 벡터를 정규화(normalize)한 후 결과 벡터를 차원 수의 제곱근인 \sqrt{d}로 스케일링하는 세 가지 간단한 단계를 따릅니다. 또한 LayerNorm의 '비가역성'(irreversibility) 특성을 소개하여 정규화 과정에서 손실된 정보를 복구할 수 없음을 입증합니다.

- **Performance Highlights**: RMSNorm을 활용한 모델이 균일 벡터를 제거하는 LayerNorm에 비해 더 효율적이면서도 비슷한 성능을 보이고, 균일 벡터에 수직인(hidden representations orthogonal) 표현을 학습하는 것으로 나타났습니다. 이는 LayerNorm의 특정 성분 제거 과정을 불필요한 단계로 만들며 RMSNorm의 사용을 지지하는 근거로 작용합니다.



### WaveletGPT: Wavelets Meet Large Language Models (https://arxiv.org/abs/2409.12924)
Comments:
          16 pages, 4 figures

- **What's New**: 본 논문은 대형 언어 모델(LLMs)에 전통적인 신호 처리 아이디어인 wavelets를 접목하여 기존 LLM 아키텍처에 추가 파라미터 없이도 거의 두 배 빠른 사전 훈련 성능을 달성하는 방법을 제안합니다.

- **Technical Details**: 제안된 아키텍처는 각 Transformer 디코더 블록에서 여러 시간 해상도의 중간 임베딩(intermediate embeddings)에 접근을 허용하여 다음 토큰 예측을 수행합니다. Haar wavelet을 사용하여 각 Transformer 디코더 레이어의 중간 임베딩에 다중 스케일 필터를 추가하여 다층 구조를 구현합니다.

- **Performance Highlights**: 이 접근 방식은 텍스트, 원시 오디오 및 기호 음악 세 가지 분야에서 사전 훈련 작업의 유효성 손실(validation loss)을 개선하여 모델 성능에 실질적인 향상을 제공했습니다. 같은 훈련 단계에서, 몇 개의 레이어나 파라미터를 추가하는 것과 유사한 비약적인 성능 향상이 이루어졌습니다.



### Defending against Reverse Preference Attacks is Difficu (https://arxiv.org/abs/2409.12914)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)이 인간의 가치에 안전하게 맞춰져 있지만, 훈련 시 공격에 취약하다는 점을 강조합니다. 특히 Reverse Preference Attacks (RPA)라는 새로운 공격 접근법을 통해, LLM이 해로운 행동을 학습하도록 만드는 방법을 제안합니다.

- **Technical Details**: RPA는 RLHF(인간 피드백으로부터의 강화 학습)에서의 공격을 설명하는 새로운 프레임워크입니다. 이 논문에서는 처벌(Reward) 모델을 활용하여 RLHF 훈련 시 데이터의 라벨을 역전시켜 해로운 정책을 최적화하는 과정을 탐구합니다. 또한, Constrained Markov Decision Processes (CMDPs)와 같은 기법을 통해 방어 메커니즘을 제안하고, '온라인'과 '오프라인' 방어 전략을 구분합니다.

- **Performance Highlights**: 실험 결과, '온라인' 방어 방식이 RPA에 대해 효과적으로 LLM을 보호할 수 있는 반면, '오프라인' 방식은 덜 효과적임을 보여줍니다. RPA는 LLM의 안전성을 심각하게 위협할 수 있으며, 이 논문은 이러한 공격이 LLM의 내부 정책을 변형할 수 있음을 경고합니다.



### A New Perspective on ADHD Research: Knowledge Graph Construction with LLMs and Network Based Insights (https://arxiv.org/abs/2409.12853)
Comments:
          14 pages, 2 figures

- **What's New**: ADHD(주의력 결핍/과다 활동 장애)의 복잡한 증상과 요인을 이해하기 위해, 이 연구는 최첨단 Large Language Models(LLMs)와 Retrieval-Augmented Generation(RAG) 시스템을 활용하여 ADHD의 포괄적인 지식 그래프(Knowledge Graph, KG)를 구축하고 이를 네트워크 분석했습니다. 이를 통해 ADHD의 핵심 노드와 관계를 밝히고, 정확하고 정보에 기반한 상호작용을 가능하게 하는 맥락 인식 챗봇을 개발했습니다.

- **Technical Details**: 이 연구는 ADHD의 다양한 과학적 자료와 임상 데이터를 통합하여 멀티모달 지식 그래프를 구축하는 데 중점을 둡니다. 연구 수행을 위해, 텍스트 데이터를 처리하고 개념을 추출하기 위해 Llama3.1-8B 모델을 사용하며, NetworkX 라이브러리를 통해 초기 그래프를 생성하고 DBSCAN 클러스터링 알고리즘으로 유사한 개념 간의 중복을 제거했습니다. 최종 그래프는 각 클러스터를 가장 관련 있는 노드로 재구성하여 중복성을 줄였습니다.

- **Performance Highlights**: 이 연구의 주목할 만한 성과는 ADHD 지식 그래프의 구축을 통해 ADHD에 대한 체계적이고 통합적인 관점을 제공할 수 있었다는 점입니다. 또한 이 지식 그래프를 기반으로 한 RAG 시스템은 환자와 임상 의사, 연구자들에게 ADHD에 대해 정확하고 구조화된 정보를 제공하여 의사결정을 지원할 수 있는 가능성을 보여주었습니다.



### Channel-Aware Domain-Adaptive Generative Adversarial Network for Robust Speech Recognition (https://arxiv.org/abs/2409.12386)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 연구에서는 미지의 녹음 환경 및 조건으로 인한 채널 불일치(channel mismatch) 문제를 해결하기 위한 혁신적인 채널 인식 데이터 시뮬레이션 방법을 제안합니다. 이 방법은 채널 추출 기술과 생성적 적대 신경망(Generative Adversarial Network, GANs)의 시너지 효과를 활용하여 자동 음성 인식(ASR) 시스템의 강건성을 향상시킵니다.

- **Technical Details**: 제안된 방법인 CADA-GAN(Channel-Aware Domain-Adaptive Generative Adversarial Network)은 두 단계로 이루어집니다: 첫째, 채널 인코더(channel encoder)를 통해 목표 도메인(target domain) 음성에서 채널 임베딩(channel embedding)을 추출합니다. 둘째, 추출된 임베딩을 활용하여 GAN 기반의 음성 합성기(speech synthesizer)가 원천 음성의 음소(content)를 보존하면서 목표 도메인의 채널 특성을 모방하는 음성을 생성합니다.

- **Performance Highlights**: Hakka Across Taiwan (HAT) 및 Taiwanese Across Taiwan (TAT) 데이터셋에서 평가한 결과, 상대 문자 오류율(Character Error Rate, CER)이 각각 20.02% 및 9.64% 감소하여, 제안된 채널 인식 데이터 시뮬레이션 방법의 효과를 입증하였습니다.



### Robust Audiovisual Speech Recognition Models with Mixture-of-Experts (https://arxiv.org/abs/2409.12370)
Comments:
          6 pages, 2 figures, accepted by IEEE Spoken Language Technology Workshop 2024

- **What's New**: EVA라는 새로운 오디오-비주얼 음성 인식 모델이 제안되었습니다. 이 모델은 다양한 상황의 'in-the-wild' 비디오에서 강력한 음성 인식 기능을 제공하기 위해 미니미한 다수의 전문가(Mixture-of-Experts) 기법을 이용합니다.

- **Technical Details**: EVA는 사전 훈련된 음성 인식 모델인 OWSM v3.1를 기반으로 하여, 비디오의 전체 프레임에서 시각 정보를 추출하고 이를 시각 토큰으로 변환한 후, 경량 프로젝션을 통해 음성 공간으로 매핑합니다. 또한, MoE 모듈을 통해 시각 정보를 ASR 모델에 통합하여 강력한 일반화 능력을 보장합니다.

- **Performance Highlights**: EVA는 세 가지 벤치마크 데이터세트에서 최첨단 성능을 기록했으며, 이전의 SOTA 모델인 AVFormer보다 약 400배 더 적은 오디오-비주얼 훈련 데이터로도 뛰어난 성능을 발휘했습니다.



### RAG-Modulo: Solving Sequential Tasks using Experience, Critics, and Language Models (https://arxiv.org/abs/2409.12294)
Comments:
          8 pages, 5 figures

- **What's New**: RAG-Modulo는 과거 상호작용을 기억하는 기능이 있는 LLM 기반 에이전트를 위한 새로운 프레임워크입니다. 이 프레임워크는 비기계적 예시를 자동으로 검색하여 더 나은 의사결정을 가능하게 합니다.

- **Technical Details**: RAG-Modulo는 Interaction Memory를 구축하고 과거 상호작용에서 관련 경험을 자동으로 검색하여 의사결정을 안내합니다. 이는 formal verifiers 또는 critics를 통해 각 단계에서 행동의 실행 가능성을 평가하여 작동합니다.

- **Performance Highlights**: 실험 결과, RAG-Modulo가 BabyAI 및 AlfWorld 도메인에서 작업 성공률과 효율성을 크게 향상시켰으며, 최신 테크닉들에 비해 성능이 뛰어난 것을 보여주었습니다.



New uploads on arXiv(cs.IR)

### The Relevance of Item-Co-Exposure For Exposure Bias Mitigation (https://arxiv.org/abs/2409.12912)
- **What's New**: 이 논문은 사용자 선택에 대한 편향(bias)이 특정 추천 시스템의 추천에 어떻게 영향을 미치는지를 탐구하며, 새로운 이론인 노출 편향(exposure bias)에 대해 설명하고 있습니다. 연구는 다양한 이산 선택 모델(discrete choice models)을 활용하여 인간 선택 데이터에 대한 노출 편향을 완화하는 방법을 제시하고 있습니다.

- **Technical Details**: 연구는 이산 선택 모델을 통해 과거 추천 정책이 어떻게 사용자의 선택에 영향을 미치는지를 분석하였고, 다변량 이산 선택 모델(multivariate discrete choice models)이 아이템 간의 경쟁(competition) 영향을 고려하여 노출 편향을 효과적으로 완화한다는 것을 보여주었습니다. MNL(multi-nomial logit model)과 같은 기존 모델이 인간의 선택을 정확하게 모델링하지 못할 수 있다는 점도 지적하였습니다.

- **Performance Highlights**: 이산 선택 기반 모델들은 다른 모델들에 비해 노출 편향을 거의 완전히 제거하였고, 정확도에서도 우수한 성능을 보였습니다. 노출이 과도한 아이템의 인기도를 과소 추정하는 경향이 관찰되었으며, 이러한 정보가 포함되면 노출 편향을 거의 완전히 제거할 수 있음을 보여주었습니다.



### HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling (https://arxiv.org/abs/2409.12740)
- **What's New**: 이 논문에서는 새롭게 Hierarchical Large Language Model (HLLM) 아키텍처를 제안하여 시퀀스 추천 시스템을 향상시키는 방법을 소개합니다. 기존의 추천 모델 보다 유의미한 성능 향상을 보여주고 있으며, 실세계 응용에서도 효율성을 검증하였습니다.

- **Technical Details**: 제안된 HLLM은 두 개의 계층을 가진 모델로 구성되어 있습니다. 첫 번째 Item LLM은 항목의 상세 텍스트 설명에서 풍부한 콘텐츠 특징을 추출하고, 두 번째 User LLM은 이러한 특징을 기반으로 사용자 상호작용 이력을 분석하여 미래의 관심사를 예측합니다. 이 모델은 훈련 효율성과 서비스 효율성을 높이며, 최대 70억 개의 파라미터를 사용하는 대규모 구성으로 확장 가능합니다.

- **Performance Highlights**: HLLM은 PixelRec 및 Amazon Reviews라는 두 대규모 데이터셋에서 최첨단 성능을 기록하였으며, 전통적인 ID 기반 모델들과 비교하여 큰 폭으로 성능을 향상시켰습니다. 온라인 A/B 테스트에서도 유의미한 이익을 보여주어 실제 추천 시나리오에서의 실용성을 입증했습니다.



### When SparseMoE Meets Noisy Interactions: An Ensemble View on Denoising Recommendation (https://arxiv.org/abs/2409.12730)
- **What's New**: 이 연구에서는 사용자 선호도를 암시적 피드백(implicit feedback)에서 학습하는 과제를 다루고 있으며, 최근에 제안된 다양한 denoising recommendation 방법들이 그 한계를 극복하기 위한 새로운 접근법으로 Adaptive Ensemble Learning (AEL)을 제안합니다.

- **Technical Details**: AEL은 sparse gating network를 기반으로 하여 적절한 전문가(expert)를 선택하고, 데이터를 통해 다양한 denoising 능력을 조합합니다. 이 모델은 denoising 모듈, 손상 모듈(corrupt module), 그리고 적응형 앙상블 모듈(adaptive ensemble module)로 구성됩니다. AEL은 하이퍼파라미터 조정 없이도 데이터 패턴에 적응할 수 있는 능력을 가집니다.

- **Performance Highlights**: 다양한 데이터셋에서 광범위한 실험을 통해 AEL이 기존의 방법들에 비해 여러 주요 지표(metric)에서 우수한 성능을 보였으며, 큰 소음이 존재하는 상황에서도 효과적으로 작동함을 입증했습니다.



### A Deep Dive into Fairness, Bias, Threats, and Privacy in Recommender Systems: Insights and Future Research (https://arxiv.org/abs/2409.12651)
Comments:
          38 pages, 6 figures

- **What's New**: 이번 연구는 추천 시스템의 공정성(fairness), 편향(bias), 위협(threats), 그리고 개인 정보 보호(privacy) 문제를 다룬다. 특히 알고리즘 결정이 특정 사용자 및 항목 그룹을 어떻게 무의식적으로 강화 또는 소외시킬 수 있는지를 분석한다.

- **Technical Details**: 연구에서는 추천 시스템에서의 공정한 추천 전략의 필요성을 강조하며, 시스템의 무결성을 저해할 수 있는 다양한 공격 형태의 위협과 개인 정보 보호를 위한 고급 기술들을 논의한다.

- **Performance Highlights**: 이 연구는 추천 시스템의 신뢰성(reliability)과 보안(security)을 손상시키는 위협을 해결하고, 공정성과 개인 정보 보호를 개선하여 다양한 사용자 집단에 더 잘 봉사할 수 있는 윤리적 추천 시스템을 개발하고자 한다.



### Bundle Fragments into a Whole: Mining More Complete Clusters via Submodular Selection of Interesting webpages for Web Topic Detection (https://arxiv.org/abs/2409.12380)
Comments:
          10

- **What's New**: 본 논문에서는 멀티모달 웹 데이터에서 흥미로운 웹 페이지를 핫 토픽으로 정리하는 새로운 접근인 Bundling-Refining (BR) 방법을 제안합니다. 이 방법은 여러 조각의 핫 토픽을 보다 완전한 형태로 탐색함으로써 기존의 비효율적인 피처 표현과 비지도학습 기반의 주제 생성 문제를 해결하고자 합니다.

- **Technical Details**: Bundling 단계에서 조각 주제를 거칠게 묶어 전체 주제로 만듭니다. 그 후 Refine 단계에서는 서브모듈 기반 기법을 사용해 거칠게 묶인 주제를 더 세밀화합니다. 이 방법은 사이트에서의 페이지의 흥미로움을 그래프 형태로 모델링하며, 중요한 페이지를 선별적으로 찾는 방식으로 이루어집니다.

- **Performance Highlights**: 제안된 BR 방법은 두 개의 공개 데이터 세트에서 기존의 최첨단 방법인 latent Poisson deconvolution보다 각각 20%의 정확도 및 10% 더 나은 성능을 보여줍니다. 또한, 이 방법은 전통적인 랭킹 방법을 초월하여 나쁘지 않은 성능을 발휘하고, 간단하게 실행될 수 있는 특징을 가지고 있습니다.



### MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines (https://arxiv.org/abs/2409.12959)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 MMSearch-Engine이라는 다중모달(AI search engine) 검색 엔진 파이프라인을 제안하여 대규모 다중모달 모델(LMMs)들이 검색 능력을 발휘할 수 있도록 지원합니다. 이를 통해 텍스트와 이미지를 포함한 다양한 사용자 쿼리를 처리하는 새로운 방식을 제안합니다.

- **Technical Details**: MMSearch-Engine은 이미지가 포함된 쿼리에 대해 웹 검색을 수행하고, 검색된 결과를 재정렬하는 과정을 포함한 다단계 상호작용을 통해 작동합니다. 또한 MMSearch라는 포괄적인 평가 기준을 통해 LMMs의 다중모달 검색 성능을 평가하며, 14개 분야에 걸쳐 수집된 300개의 데이터를 사용합니다.

- **Performance Highlights**: GPT-4o는 MMSearch-Engine과 함께 사용할 경우, 상업적으로 우수한 검색 엔진인 Perplexity Pro보다 뛰어난 성능을 보였습니다. 그러나 현재의 LMM들은 여전히 다중모달 검색 작업에서 일반화하는 데 어려움을 겪고 있으며, 정답을 올바르게 식별하는 능력에 제약이 있습니다.



### Exploring Large Language Models for Product Attribute Value Identification (https://arxiv.org/abs/2409.12695)
- **What's New**: 이 논문은 e-커머스에서 제품 속성 값 식별(Product Attribute Value Identification, PAVI)을 위한 대형 언어 모델(large language models, LLMs)의 가능성을 탐구합니다. 기존의 방법들은 사전 훈련된 언어 모델(pre-trained language models, PLMs)인 BART 및 T5에 의존하여 많은 양의 특화된 훈련 데이터를 요구하고 새로운 속성에 일반화하기 어려웠습니다. 이 논문은 LLaMA 및 Mistral과 같은 LLMs를 데이터 효율적이고 강력한 대안으로 제안합니다.

- **Technical Details**: 저자들은 파라메트릭(parametric) 및 비파라메트릭(non-parametric) 지식을 활용한 다양한 전략을 제안하며, 제로샷(zero-shot) 환경에서의 한 단계와 두 단계 프롬프트 기반 접근 방식을 비교합니다. 특히, 사전 훈련된 T5 모델을 기반으로 한 밀집 시연 검색기(dense demonstration retriever)를 도입하고, 태스크 특화 지침을 명시적으로 훈련하기 위한 지시 지침 훈련(instruction fine-tuning)을 수행합니다.

- **Performance Highlights**: 두 개의 제품 벤치마크에서 수행한 광범위한 실험 결과, 두 단계 접근 방식이 제로샷 설정에서 성능을 크게 개선하며, 훈련 데이터를 사용할 경우 지시 지침 훈련이 추가적으로 성능을 향상시키는 것으로 나타났습니다. 이는 PAVI를 위한 대형 언어 모델 사용의 실질적인 이점을 보여줍니다.



### Multi-View Adaptive Contrastive Learning for Information Retrieval Based Fault Localization (https://arxiv.org/abs/2409.12519)
- **What's New**: 이 논문에서는 소프트웨어 결함 위치 식별을 위한 새로운 방법인 Multi-View Adaptive Contrastive Learning for Information Retrieval Fault Localization (MACL-IRFL)을 제안합니다. 이 방법은 버그 보고서와 소스 코드 파일 간의 상호 작용, 버그 보고서 간의 유사성, 소스 코드 파일 간의 공동 인용 관계와 같은 여러 관계를 학습하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MACL-IRFL은 버그 보고서와 소스 코드 파일 간의 상호 작용을 세 가지 관점(보고서-코드 상호 작용 뷰, 보고서-보고서 유사성 뷰 및 코드-코드 공동 인용 뷰)으로 모델링하여 각각 데이터 증강을 생성합니다. 또한 Graph Neural Network (GNN)를 사용하여 세 가지 관점에서 버그 보고서 또는 소스 코드 파일의 정보를 집계하고, 대조 학습(contrastive learning)을 수행하여 노이즈를 줄입니다.

- **Performance Highlights**: 이 연구는 오픈 소스 자바 프로젝트 5개에 대한 실험을 통해 성능을 평가했습니다. 결과적으로, 제안된 모델은 Accuracy@1에서 최대 28.93%, MAP에서 25.57%, MRR에서 20.35%의 성능 향상을 보여주었습니다.



### Familiarity-aware Evidence Compression for Retrieval Augmented Generation (https://arxiv.org/abs/2409.12468)
- **What's New**: 본 논문에서는 FaviComp (Familiarity-aware Evidence Compression)이라는 새로운 방법을 제안합니다. 이 방법은 외부에서 가져온 증거(evidence)가 최종 모델(target model)에게 보다 친숙하게 만들어질 수 있도록 하며, 파라메트릭 지식(parametric knowledge)과 비파라메트릭 지식(non-parametric knowledge)을 통합합니다.

- **Technical Details**: FaviComp는 모델의 파라메트릭 지식을 필요한 만큼 통합하면서, 압축된 증거(evidence)의 혼란도를 낮춰(target model의 perplexity 감소) 사용될 수 있도록 합니다. 두 개의 언어 모델(LM), 즉 압축 모델(compression model)과 목표 모델(target model)의 토큰 확률(token probabilities)을 조합하여 새로운 컨텍스트(context)를 생성합니다. 이 과정을 통해 FaviComp는 보다 효율적인 정보 통합을 이끌어내어, 복잡한 작업에서도 본질적인 정보를 유지할 수 있도록 합니다.

- **Performance Highlights**: FaviComp는 다섯 개의 공개 도메인 QA 데이터셋에서 기존의 모든 기준선(baselines)을 초과하는 성능을 보여주었습니다. 높은 압축 비율을 유지하면서, 두 개의 다운스트림 LMs과 함께 작동하여 파라메트릭 및 비파라메트릭 지식의 효과적인 통합을 입증했습니다.



### A Simple Model to Estimate Sharing Effects in Social Networks (https://arxiv.org/abs/2409.12203)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 이번 연구에서는 소셜 네트워크에서 사용자의 공유 행동을 설명하는 간단한 Markov Decision Process (MDP) 기반 모델을 제안합니다. 이 모델을 통해, 기존 방법들과 비교했을 때 치료 효과를 보다 정확하게 추정할 수 있는 편향 없는 추정기를 도출하고 그 성능을 입증하였습니다.

- **Technical Details**: 제안된 MDP 모델은 상태와 행동, 전이 확률, 보상으로 구성됩니다. 사용자는 실험에서 주어진 시스템 변형에 따라 행동을 취하며, 성공적인 공유 전이에 대해 일정한 보상을 받습니다. 이 과정에서 계량 정수(SUTVA)의 가정이 위반되지 않는 조건 하에 공유의 효과에 대해 편향 없는 추정기를 도출하였습니다.

- **Performance Highlights**: 재현 가능한 합성 실험을 통해, 제안된 모델이 기존의 다른 방법들보다 유의미하게 높은 성능을 보임을 확인하였습니다. 즉, 사용자의 공유가 소비 지표에 미치는 영향을 효과적으로 추정할 수 있음을 입증하였습니다.




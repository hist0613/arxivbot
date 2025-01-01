New uploads on arXiv(cs.CL)

### Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs (https://arxiv.org/abs/2412.21187)
Comments:
          Work in progress

- **What's New**: 이 논문은 OpenAI의 o1 모델과 유사한 모델들이 직면한 과도한 사고(overthinking) 문제를 심층적으로 분석합니다. 이 모델들은 단순한 문제를 해결하는 데 지나치게 많은 계산 자원을 할당하는 경향이 있습니다. 저자들은 이러한 문제를 해결하기 위해 새로운 효율성 메트릭을 제안하고, 자원의 합리적 사용을 평가하는 방법을 통해 과도한 사고를 줄이려는 전략을 제안합니다.

- **Technical Details**: 이 연구에서는 o1 모델의 출력 패턴을 분석하여 과도한 사고의 존재를 정량적으로 감지하고, 자체 훈련(self-training)과 부트스트래핑(bootstrapping) 설정을 통해 효율적인 사고를 유도합니다. 제안된 메트릭은 결과 효율성과 과정 효율성을 모두 고려하여, 적은 자원으로도 정확한 답변을 도출할 수 있는 모델의 성능을 평가합니다. 실험은 MATH500, GSM8K, ASDIV와 같은 다양한 난이도의 테스트셋에서 진행되었습니다.

- **Performance Highlights**: 실험 결과, 저자들은 제안된 방법이 수학 추론 작업에서 눈에 띄는 성과를 거두었음을 입증하였습니다. 특히, 모델의 성능을 유지하면서 계산 오버헤드를 효과적으로 줄인 것으로 나타났습니다. 결과적으로, 저자들은 과도한 사고 문제를 % 비율로 줄이는데 성공했으며, 이는 향후 모델 설계 및 효율성 개선에 기여할 것입니다.



### Facilitating large language model Russian adaptation with Learned Embedding Propagation (https://arxiv.org/abs/2412.21140)
Comments:
          Preprint version of an article published in the Journal of Language and Education. Copyright held by the owner/author(s). Publication rights licensed to the Journal of Language and Education

- **What's New**: 본 논문에서는 Learned Embedding Propagation (LEP)이라는 새로운 어댑테이션 파이프라인을 제안하고 있습니다. 기존의 instruction-tuning 과정 없이도 새로운 언어 지식을 기존의 instruct-tuned 모델에 직접 삽입할 수 있는 방법입니다. 이 접근법은 데이터 요구량을 줄이고, 언어 특정 모델의 비용을 줄이며, 더 나은 성능을 제공합니다.

- **Technical Details**: LEP는 기존 모델의 지식에 의존하여 언어 적응을 실시하며, 데이터의 양을 최소화하는 방법을 탐구합니다. 연구자들은 BPE, Unigram, Extension, Optimization과 같은 다양한 토큰화 방법을 실험하고, 이러한 방법들이 러시아어의 형태론(linguistic morphology)과 얼마나 잘 일치하는지를 평가했습니다. 또한 새로운 임베딩을 initialization 하는 방법도 제시하여, 이전 토큰화된 데이터의 평균을 사용하는 기법이 효과적임을 보여줍니다.

- **Performance Highlights**: LEP는 Mistral-7B 및 LLaMa-3-8B 모델에 대해 4가지 러시아어 토큰화 변형을 실험하며, 기존의 instruction-tuning 방법들과 경쟁력을 갖추고 있다는 것을 입증했습니다. 실험 결과, LEP는 OpenChat 3.5 및 LLaMa-3-8B-Instruct와 비슷한 수준의 성능을 달성했으며, 후속 조정을 통해 더 나은 과제 해결 능력을 나타냈습니다.



### Exploring and Controlling Diversity in LLM-Agent Conversation (https://arxiv.org/abs/2412.21102)
Comments:
          Accepted for the AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration

- **What's New**: 본 논문에서는 Multi-Agent communication에서 다양성(Diversity)을 제어하고 탐구하는 새로운 방법인 Adaptive Prompt Pruning (APP)를 제안합니다. APP는 발화 생성 프롬프트의 내용을 동적으로 조정하여 다양성을 단일 파라미터인 λ(램다)를 이용해 제어합니다. 결과적으로, APP는 모델과 데이터셋 전반에서 효과적으로 출력의 다양성을 조절한다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 다양한 설계를 통해 프롬프트 내용과 출력 다양성 간의 관계를 포괄적으로 분석했습니다. 주요 구성 요소로는 환경 설명, 에이전트 프로필, 대화 이력 등이 있으며, 이들은 출력의 다양성에 영향을 미칩니다. APP는 temperature sampling 및 top-p sampling과 같은 기존의 다양성 관리 기술과 호환되어, 이론적으로도 다재다능한 도구입니다.

- **Performance Highlights**: APP는 대화 생성 후 정보 누락으로 인한 불일치를 해소하기 위해 수정 단계(correction step)를 도입하여, 다양성 증진과 출력 일관성(consistency) 간의 균형을 효과적으로 맞춥니다. 실험 결과, 다양한 프롬프트 구조가 출력의 다양성에 미치는 영향을 조사한 바, 구성 요소의 순서와 길이가 다양성에 중대한 영향을 미친다는 사실을 발견하였습니다.



### Efficient Multi-Task Inferencing with a Shared Backbone and Lightweight Task-Specific Adapters for Automatic Scoring (https://arxiv.org/abs/2412.21065)
Comments:
          Accepted by AAAI-iRAISE Workshop

- **What's New**: 본 논문은 인공지능(AI)이 교육에서 확장 가능하고 효율적인 프레임워크를 구축하기 위한 접근법을 제시하고 있습니다. 경량의 LoRA 어댑터를 사용하여 27개의 상호 배타적인 과제의 자동 채점 문제를 해결하기 위한 공유 백본 모델 구조를 제안하였습니다. 이 방법으로 GPU 메모리 소비를 60% 줄이고, 추론 시간도 40% 개선하는 등의 효율성을 달성했습니다.

- **Technical Details**: 제안된 프레임워크는 공유 백본 모델과 태스크 별 경량 모듈로 구성되며, LoRA(저순위 적응) 레이어와 세부 조정된 분류 헤드를 사용합니다. 각 과제는 고유한 LoRA 어댑터나 분류 헤드를 가지며, 이를 통해 모델의 메모리 사용을 최소화하면서도 효율적으로 과제에 적응할 수 있도록 합니다. 이 프레임워크는 Hugging Face Transformers 라이브러리를 기반으로 구현되어 신속한 로드와 적응이 가능하게 설계되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 27개의 상호 배타적인 과제에서 높은 정확도를 보여주며, 기존의 다중 모델 배포 방식에 비해 비용과 배포 시간을 크게 줄였습니다. 또한, 전통적인 모델보다 평균 QWK 점수에서 경쟁력 있는 성과를 보이며, 자동화된 채점 시스템의 공정성과 투명성을 유지하는 잠재력을 강조합니다. 정량적 성과는 QWK와 F1 점수를 포함한 여러 메트릭스를 사용하여 평가되었습니다.



### GePBench: Evaluating Fundamental Geometric Perception for Multimodal Large Language Models (https://arxiv.org/abs/2412.21036)
- **What's New**: 새로운 벤치마크인 GePBench를 소개합니다. MLLMs(다양한 모달 대형 언어 모델)의 기하학적 인식 능력을 평가하는 데 중점을 둡니다. 연구 결과 현재 최고 성능의 MLLMs조차도 기하학적 인지 과제에서 현저한 결함을 보이며, 이는 다중 모달 응용을 위한 기초적인 인식 기술의 중요성을 강조합니다.

- **Technical Details**: GePBench는 20,000개의 이미지와 250,000개의 객관식 질문으로 구성되어 있으며, 기하학적 도형에 대한 공간 인식과 형태 이해와 같은 핵심 역량을 평가합니다. 이 벤치마크는 위치, 크기, 존재, 개수, 참조 및 관계와 같은 6가지 주요 차원을 중심으로 개발되었습니다. GePBench의 구조적 데이터 합성 엔진을 통해 생성된 질문은 기하학적 도형을 기반으로 하고, 다수의 MLLMs를 통해 평가됩니다.

- **Performance Highlights**: MLLMs는 GePBench의 평가 결과에서 기하학적 인식에 심각한 한계를 보였습니다. 예를 들어, Gemini-1.5-pro는 객관식 질문에서 평균 69.4%의 정확도로 기하학적 도형의 크기를 판단하는 데 어려움을 보였습니다. 반면 LLaVA-GeP 모델은 GePBench에서의 데이터로 훈련받아 다양한 하위 작업에서 주목할 만한 개선을 보여줍니다.



### Plancraft: an evaluation dataset for planning with LLM agents (https://arxiv.org/abs/2412.21033)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트를 위한 멀티 모달 평가 데이터셋인 Plancraft를 소개합니다. Plancraft는 Minecraft의 crafting GUI를 기반으로 하여, 텍스트 전용 및 멀티 모달 인터페이스를 포함하고 있습니다. 이를 통해 도구 사용과 Retrieval Augmented Generation (RAG) 평가를 위해 Minecraft Wiki를 활용하며, 현대 에이전트 아키텍처의 다양한 구성 요소를 실험하고 있습니다.

- **Technical Details**: Plancraft는 다양한 복잡성과 길이의 계획 문제를 포함하고 있으며, 일부는 고의적으로 해결할 수 없도록 설계되었습니다. 이를 통해 LLM 및 VLM(비주얼 언어 모델)의 계획 문제를 평가하고, LLM 기반 에이전트의 성능을 수작업 계획자와 비교합니다. 데이터셋에서는 텍스트 전용 모델과 멀티 모달 에이전트를 평가하고, 전문 계획안에 대한 미세 조정의 영향을 실험합니다.

- **Performance Highlights**: Plancraft의 성과 지표는 성공률뿐만 아니라 LLM의 계획이 수작업 솔루션에 얼마나 가까운지를 평가합니다. 기존의 LLM 평가는 성공률에만 집중했으나, Plancraft는 실제 상황의 복잡성을 포착하기 위해 더 세분화된 평가를 제공합니다. 연구 결과, LLM들은 Plancraft가 제시하는 계획 문제에서 어려움을 겪고 있으며, 이들의 능력을 향상시키기 위한 제안도 포함되어 있습니다.



### MapQaTor: A System for Efficient Annotation of Map Query Datasets (https://arxiv.org/abs/2412.21015)
Comments:
          13 pages, 35 figures

- **What's New**: 이 논문에서는 기존의 지도 API와의 통합을 통해 map-based question-answering (QA) 데이터셋을 효율적으로 생성할 수 있는 웹 애플리케이션인 MapQaTor를 소개합니다. 이 플랫폼은 사용자들이 다양한 지도 서비스를 쉽게 연결하고, 지리적 데이터를 수집하고 시각화할 수 있도록 설계되었습니다. 또한, MapQaTor는 지도의 사용 비효율성과 불일치 문제를 해결하고자 하며, 데이터 정확성을 높이기 위해 API 응답을 캐싱합니다.

- **Technical Details**: MapQaTor는 plug-and-play 아키텍처를 갖춘 웹 기반 플랫폼으로, 다양한 지도 API와의 통합을 지원합니다. 사용자는 질문과 답변 쌍을 설계하고 JSON 형식으로 데이터셋을 내보낼 수 있으며, 이 과정은 10단계의 주요 단계를 통해 설명됩니다. 이 시스템은 사용자가 자연어 기반의 QA 작업을 수행할 수 있도록 지원하여 복잡한 지리적 질문 처리를 용이하게 합니다.

- **Performance Highlights**: MapQaTor는 수작업 방법에 비해 데이터 주석 처리 속도를 최소 30배 향상시키는 효율성을 보여줍니다. 이를 통해 지리적 자원의 개발은 물론, LLM들이 지리 정보를 이해하는 능력을 향상시키는 데 중요한 기여를 하고 있습니다. 논문은 GitHub에 코드를 공개하였으며, 다양한 공동 작업자에게 열려 있습니다.



### Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale via Principled Criteria (https://arxiv.org/abs/2412.21006)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 중간 추론 경로를 줄이는 새로운 방법인 문장 단위의 근거 감소 훈련 프레임워크를 제안합니다. 이 접근법은 중복된 추론 문장을 제거하는데 목표를 두고 있으며, 기존의 토큰 단위 감소 방식과는 달리 모델의 성능을 유지하면서 생성 길이를 줄일 수 있음을 입증합니다.

- **Technical Details**: 제안된 방법은 ‘verbosity’라는 기준을 활용하여 비필요한 근거 문장을 제거하고, 이를 통해 LLM의 추론 능력을 보존하면서도 근거 생성을 줄이는 제안입니다. 초기 추론 단계에서의 근거 문장들이 중첩성을 유발함을 경험적으로 증명하고, 문장 단위로 중복성을 평가하는 방법을 도입하여 LLM 성능 저하 없이 효율적인 경로 감소를 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 다양한 모델과 태스크에서 평균 17.15%의 생성 비용이 감소하는 것으로 나타났습니다. 이는 LLMs의 근거 문장을 효율적으로 관리함으로써 보다 효과적인 성능을 유지할 수 있음을 시사하며, 실제 적용 시 효율성 및 안정성을 높일 수 있는 방법론으로 주목받고 있습니다.



### Plug-and-Play Training Framework for Preference Optimization (https://arxiv.org/abs/2412.20996)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 연구에서는 수학적 추론과 같은 정확성 요구가 높은 작업에서의 성능 향상을 위해 새로운 훈련 프레임워크를 제안합니다. 기존의 Preference Optimization (PO) 방법들은 훈련 샘플의 난이도 차이를 제대로 반영하지 못하는 한계를 가지고 있습니다. 이 연구에 제시된 방법은 여러 번의 샘플링을 통해 출력 분포를 분석하고, 훈련 샘플에 서로 다른 가중치를 부여하여 모델이 도전적인 예제에 더 집중하도록 합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 단계로 구성됩니다. 첫 번째 단계는 데이터 수집으로, 동일한 질문에 대해 모델의 응답을 여러 번 샘플링하여 출력 분포를 이해합니다. 두 번째 단계에서는 모델의 성능에 따라 각 훈련 샘플의 가중치를 조정하는 메트릭을 설계하며, 세 번째 단계에서는 얻은 가중치를 활용하여 더 어려운 샘플에 초점을 맞추도록 훈련 과정을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크가 수학적 추론 작업에 일관된 성능 향상을 달성하며 다양한 PO 방법과 성공적으로 통합됨을 입증하였습니다. 이 연구는 LLM의 대화 생성 및 질문 응답에서 인간 선호와의 정렬을 높이는 데 기여할 수 있는 유연하고 효과적인 방법을 제시합니다.



### KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation (https://arxiv.org/abs/2412.20995)
Comments:
          23 pages, 6 figures

- **What's New**: 이번 연구에서는 Knowledge graph Assisted Reasoning Path Aggregation (KARPA)라는 새로운 프레임워크를 제안합니다. KARPA는 LLM의 글로벌 계획(Planning) 능력을 활용하여 더 효율적이고 정확한 지식 그래프(QG) 추론을 가능하게 합니다. 기존의 KG 기반 질문 응답 방식의 제한점을 극복하며 별도의 학습 없이 다양한 LLM 아키텍처에 적응할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: KARPA는 세 가지 단계로 구성됩니다. 첫 번째로 LLM의 글로벌 계획 능력을 사용하여 관계 경로(relation path)를 사전 계획(pre-planning)하고, 두 번째로 임베딩(embedding) 모델을 통해 의미적으로 관련성 있는 경로를 매칭합니다. 마지막으로 이러한 경로에 대한 추론(reasoning)을 통해 답변을 생성합니다.

- **Performance Highlights**: KARPA는 KGQA 작업에서 최고 수준의 성능(state-of-the-art performance)을 달성하며, 높은 효율성과 정확성을 보여줍니다. 실험 결과에 따르면 KARPA는 기존 방법에 비해 뛰어난 성능을 발휘하여 KGQA의 새로운 가능성을 제시합니다. 또한 코드가 Github에 공개될 예정입니다.



### DoTA: Weight-Decomposed Tensor Adaptation for Large Language Models (https://arxiv.org/abs/2412.20891)
Comments:
          12 pages, 6 figures

- **What's New**: 논문에서 제안된 Weight-Decomposed Tensor Adaptation (DoTA)은 기존의 랜덤 초기화 방법 대신, 사전 훈련된 가중치의 Matrix Product Operator (MPO) 분해를 활용하여 LLMs의 효과적인 초기화를 구현합니다. 이는 고차원 구조를 효과적으로 캡처하여 모델의 성능을 향상시킵니다. 또한, QDoTA라는 4비트 양자화된 버전도 제안하여 메모리 소모를 줄이는 동시에 성능을 유지합니다.

- **Technical Details**: 본 논문에서는 텐서 대수(tensor algebra)의 기초 개념을 간략히 소개합니다. N차 텐서(order-N tensor)는 벡터와 행렬의 일반화로서, 다차원 배열로 표현됩니다. 텐서의 수축(tensor contraction) 과정은 두 텐서의 지수 쌍을 더하여 고차원 텐서 간의 곱셈을 일반화합니다.

- **Performance Highlights**: 실험 결과, DoTA는 랜덤 초기화 방법보다 적은 매개변수로 더 나은 validation loss 곡선을 기록하며, 일부 작업에서는 전체 미세 조정(full fine-tuning)보다 뛰어난 성능을 보입니다. QDoTA는 commonsense reasoning 작업에서 DoTA와 유사한 성능을 유지하면서 메모리 사용을 현저히 줄이는 성과를 보여주었습니다.



### Enhancing Annotated Bibliography Generation with LLM Ensembles (https://arxiv.org/abs/2412.20864)
- **What's New**: 이 연구는 전통적인 주석 목록 생성 방식에 비해 더욱 향상된 방법을 제안합니다. 특히, 다양한 역할을 수행하는 여러 LLM(대형 언어 모델)을 활용하여 모델 성능을 개선하는 체계적인 방법론을 채택하였습니다. 본 접근 방식은 텍스트 생성, 평가, 요약 과정에서의 LLM들의 협력을 통해 이루어집니다.

- **Technical Details**: 제안된 방법은 서로 다른 LLM 매개변수를 통해 텍스트의 다양성을 확보하고, 평가를 위한 LLM이 정확성, 관련성 및 일관성을 평가하는 구조로 되어 있습니다. 이 후 여러 결합 전략을 통해 선택된 응답은 요약 및 중복 제거 기법으로 통합 및 정제됩니다. 이러한 체계적인 접근 방식은 학문적인 작업에서의 LLM의 역할을 더욱 다양하게 확장할 수 있게 합니다.

- **Performance Highlights**: 예비 실험 결과, LLM 앙상블의 결합된 출력을 사용했을 때 개별 응답에 비해 일관성과 관련성이 개선됨을 보여주었습니다. 이 연구는 주석 품질이 38% 향상되었고 내용 중복이 51% 감소하였음을 밝히고 있습니다. 이를 통해 복잡한 학문적 작업의 자동화 가능성을 제시하면서도 높은 품질 기준을 유지할 수 있다는 점이 강조되었습니다.



### Are LLMs Really Not Knowledgable? Mining the Submerged Knowledge in LLMs' Memory (https://arxiv.org/abs/2412.20846)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 지식 보유 및 표현 간의 차이를 분석하고, 새로운 지표인 Hits@k를 제안합니다. 연구 결과, LLM이 잘못된 답변을 생성하더라도 높은 확률의 토큰에서 올바른 지식을 유지한다는 사실을 밝혔습니다. 이를 바탕으로, SkipUnsure라는 방법을 개발하여 모델이 내부적으로 보유한 지식을 활용해 답변 정확성을 높이는 방안을 제시합니다.

- **Technical Details**: 논문은 LLM의 출력 분석을 통해 지식 저장과 표현 간의 체계적 간극을 강조합니다. Hits@k 메트릭은 모델의 정확한 표현과 관계없이 지식 보유를 평가하기 위해 개발되었습니다. 연구에 사용된 데이터셋은 DBPedia와 IMDB로, 다양한 모델 사용 시 결과에 대한 세부적인 실험을 통해 지식 패턴의 효용성을 입증했습니다.

- **Performance Highlights**: 실험 결과, DBPedia에서 LLaMA3-8b 모델은 Hits@1에서 17.2%의 정확도를 보였지만, Hits@5에서는 57.9%로 더 많은 저장된 지식을 드러냈습니다. SkipUnsure 방법을 통해 DBPedia에서 11.8%, IMDB에서 6.3%의 정확도 향상을 달성하여 지식 기반 질문 응답의 실질적인 개선을 보여주었습니다.



### Disentangling Preference Representation and Text Generation for Efficient Individual Preference Alignmen (https://arxiv.org/abs/2412.20834)
Comments:
          Coling 2025

- **What's New**: 이 연구에서는 개인의 피드백에 따라 LLM(대형 언어 모델)을 개인화하는 새로운 접근 방식을 제시합니다. 특히, 개인의 선호도를 효율적으로 정렬할 수 있는 유연한 패러다임을 도입하며, 이는 텍스트 생성과 선호 표현을 분리하여 수행됩니다. 이러한 접근 방식은 기존의 PEFT(파라미터 효율 미세 조정) 방식보다 80%에서 90%까지 훈련 시간을 단축시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다: 첫 번째 단계에서는 반응 표현을 위한 잠재 인코더와 이를 LLM에 공급하는 잠재 어댑터를 훈련합니다. 두 번째 단계에서는 개인의 피드백에 따라 개인화된 잠재 표현을 생성하기 위해 잠재 인코더를 미세 조정합니다. 추가적으로, 개인의 피드백을 통해 수집된 정보를 활용하여 맞춤형 텍스트 생성을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LoRA 기반의 방법과 P-Tuning 기반의 방법에 비해 경쟁력 있는 정렬 품질을 달성합니다. 예를 들어, LoRA 기반 DPO의 실험 결과는 52.4에서 80.8로 개선된 반면, 제안된 Latent DPO는 52.5에서 83.3로 향상되었습니다. 이는 지속적으로 높은 품질을 유지하면서도 계산 비용과 시간을 크게 절감할 수 있음을 보여줍니다.



### Attributing Culture-Conditioned Generations to Pretraining Corpora (https://arxiv.org/abs/2412.20760)
- **What's New**: 이번 연구는 언어 모델이 문화에 따라 편향된 출력을 생성하는 방식에 대한 새로운 통찰력을 제공합니다. 본 연구에서는 MEMOed 프레임워크(MEMOrization from pretraining document)를 통해 모델이 특정 문화와 관련된 심볼(symbol)을 어떻게 기억하는지를 분석하고, 이러한 기억이 문화적 편향의 원인임을 밝혀냅니다.

- **Technical Details**: MEMOed 프레임워크는 두 가지 단계로 구성되어 있습니다: 1) 문화-심볼 연관성에 기여한 문서를 식별하고, 2) 기여 문서의 비율이 충분히 높은 경우 해당 심볼을 기억된 것으로 분류합니다. 이를 통해 110개 문화에 대한 음식 및 의류 주제로 생성된 심볼에서 기억된 연관성이 46% 및 26%를 차지함을 발견했습니다.

- **Performance Highlights**: 이 연구는 모델이 기억된 고빈도 심볼에 의존하므로, 문화에 대한 정확한 지식을 고르게 회상하지 못하고 있다는 점을 강조합니다. 이러한 결과는 문화 생성에서 편향을 완화하기 위한 더 나은 접근 방식을 모색하는 데 유용할 것입니다. 향후 MEMOed 프레임워크와 관련된 연구가 진행됨으로써, 모델의 성능이 사전 훈련 데이터에 어떻게 의존하는지를 더 깊이 이해할 수 있게 되기를 바랍니다.



### Depression and Anxiety Prediction Using Deep Language Models and Transfer Learning (https://arxiv.org/abs/2412.20741)
- **What's New**: 이 연구는 디지털 스크리닝 및 모니터링 애플리케이션이 정신 건강 상태 관리에 어떻게 도움을 줄 수 있는지를 탐구합니다. 16,000개 사용자 상호작용에서 수집한 대화형 음성을 통해 우울증(depression)과 불안(anxiety)의 탐지를 위한 깊은 언어 모델(deep language models)의 활용이 보고되었습니다. 이 연구는 PHQ-8과 GAD-7 결과를 바탕으로 한 데이터 레이블을 사용하여 진행되었습니다.

- **Technical Details**: 이 연구에서는 이진 분류(binary classification)에서 0.86에서 0.79 AUC의 성능을 보여주었습니다. 두 가지 상태가 모두 존재하거나 전혀 존재하지 않을 때 최고의 성능이 나타났으며, 이 결과는 데이터 스큐(data skew)와는 관련이 없다는 것을 보여줍니다. 또한, 연구진은 단어 시퀀스(underlying word sequence) 신호가 불안보다 우울증에 대해 더 뚜렷하게 나타날 수 있음을 발견했습니다.

- **Performance Highlights**: 최고 성능의 이진 분류 결과에 따르면, 조건 및 동시 발생(co-occurrence)에 따라 성능이 달라진다고 합니다. 특히, 사용자 간의 상태가 모두 동일하거나 전무할 때 가장 우수한 결과를 얻었습니다. 이로 인해 디지털 도구가 정신 건강 상태의 조기 탐지 및 관리에 있어 잠재적인 유용성을 제공할 수 있음을 시사합니다.



### Align Attention Heads Before Merging Them: An Effective Way for Converting MHA to GQA (https://arxiv.org/abs/2412.20677)
Comments:
          12 pages, 4 figures

- **What's New**: 본 연구에서는 MHA(멀티헤드 어텐션) 모델을 GQA(그룹 쿼리 어텐션) 모델로 전환하는 저비용 방법을 제안합니다. 이 방법은 L0 정규화(regularization) 마스크를 통해 중복된 파라미터를 점진적으로 제거하는 방식으로 구성됩니다. 또한, 어텐션 헤드 간의 유사성을 높이기 위해 모델 변환을 적용하여 모델 소거 훈련(pruning training)의 성능을 개선합니다.

- **Technical Details**: 저자들은 L0 마스크를 활용하여 원래의 키-값 프로젝션 매트릭스를 새로운 것으로 변환하며, 이 과정에서 계산 불변성(computational invariance) 개념을 적용합니다. 이는 어텐션 헤드의 매트릭스를 변경하지 않고, 유사한 KV 캐시간의 유사성을 측정하여 그룹화하는 방법으로 작동합니다. 실험을 통해 LLaMA2-7B 모델을 GQA 구조로 변환하는 것에서 성능 저하 없이 최대 87.5%의 키-값 헤드를 압축할 수 있음을 입증했습니다.

- **Performance Highlights**: 제안된 알고리즘은 LLaMA2-7B 모델을 GQA-4, GQA-8, GQA-16 등으로 압축했을 때, 전체 모델 성능에 비해 유의미한 감소 없이 수행되었습니다. 실험 결과는 모델의 효율성을 크게 개선할 수 있으며, 비교 실험을 통해 기존 방법보다 우수한 성능을 나타냅니다. 이 연구는 LLM(대규모 언어 모델)의 KV 캐시 압축과 관련하여 새로운 관점을 제시합니다.



### Knowledge Editing for Large Language Model with Knowledge Neuronal Ensemb (https://arxiv.org/abs/2412.20637)
Comments:
          26 pages, 5 figures, 2 tables

- **What's New**: 이 논문에서는 실시간으로 변화하는 실제 지식을 효과적으로 수정하기 위해 새로운 지식 수정 방법인 Knowledge Neuronal Ensemble (KNE)를 제안합니다. KNE는 특정 지식을 인코딩하는 뉴런 그룹을 나타내며, 이는 파라미터 로컬라이제이션에서 발생하는 잦은 수정 문제를 완화합니다. 본 방법은 각 계층에서 파라미터에 대한 기울기 기여 점수를 계산하여 정확한 파라미터 로컬라이제이션을 보장합니다.

- **Technical Details**: KNE 방법은 지식 신경 집합체(knowledge neuronal ensemble)를 사용하여 파라미터 업데이트 중 동적 상호작용을 보장합니다. 이는 지식 신경 집합체와 관련된 기울기와 손실만을 계산하고, 오류 역전파(error backpropagation)을 통해 파라미터 간의 협동적 업데이트를 수행합니다. 또한, 최적화된 업데이트 전략을 통해 수정해야 하는 파라미터의 수를 전체 모델의 약 1%로 줄임으로써 계산 비용을 크게 감소시킵니다.

- **Performance Highlights**: 세 가지 널리 사용되는 지식 수정 데이터셋에서 KNE 방법이 기존의 최상위 기준 방법보다 뛰어난 성능을 보여줍니다. 특히, KNE는 정확한 지식 수정을 크게 향상시키고, 이동성과 로컬리티(metric) 지표에서도 최상의 성능을 달성합니다. 또한, 주요 레이어에 대한 지식 신경 집합체를 수정하는 것이 이전 방법들과 비교했을 때 우수한 결과를 도출함을 시사합니다.



### NLP-based Regulatory Compliance -- Using GPT 4.0 to Decode Regulatory Documents (https://arxiv.org/abs/2412.20602)
Comments:
          accepted for presentation at Georg Nemetschek Institute Symposium & Expo on Artificial Intelligence for the Built World - Munich, Germany. 12 Sept 2024

- **What's New**: 이번 연구는 GPT-4.0와 같은 대형 언어 모델(LLM)이 규제 문서의 의미적 복잡성을 이해하고, 불일치 및 모순을 탐지하는 데 큰 가능성을 보여준다는 점에서 새로운 사실을 제시합니다. 연구진은 규제 요구사항 내의 충돌을 식별하는 GPT-4.0의 능력을 평가하기 위해 인공적으로 모호성과 모순을 주입한 선택된 말뭉치를 분석했습니다.

- **Technical Details**: 실험은 정확도(precision), 재현율(recall), F1 점수와 같은 메트릭(metrics)을 사용하여 진행되었으며, 결과는 인간 전문가들에 의해 검증되었습니다. 연구는 건축가 및 규정 준수 엔지니어들과 협력하여 설계된 데이터로 수행되었습니다.

- **Performance Highlights**: GPT-4.0은 규제 요구 사항에서 불일치를 효과적으로 탐지하는 성능을 보였으며, 이는 규제 준수 프로세스를 향상시킬 수 있는 잠재력을 나타냅니다. 하지만 더 크고 도메인별로 미세 조정된 데이터셋을 통한 추가 테스트가 필요하며, 향후 연구에서는 자동 충돌 해결 및 산업 파트너와의 파일럿 프로젝트를 통한 실제 적용 가능성을 탐색할 예정입니다.



### GliLem: Leveraging GliNER for Contextualized Lemmatization in Estonian (https://arxiv.org/abs/2412.20597)
Comments:
          Accepted to NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이번 연구에서는 에스토니아어를 위한 새로운 하이브리드 형태소 분석 시스템인 GliLem을 소개합니다. GliLem은 기존의 매우 높은 정확도를 자랑하는 규칙 기반 형태소 분석기 Vabamorf에 외부의 불확실성 제거 모듈을 결합하여 성능을 향상시킵니다. 이 외부 모듈은 GliNER이라는 오픈 어휘 NER 모델을 기반으로 하여, 자연어 내 텍스트 스팬을 텍스트 레이블과 일치시킬 수 있는 기능을 제공합니다. 이를 통해 Vabamorf의 형태소 분석 정확도를 10% 향상시키고, 이러한 개선사항이 정보 검색(Information Retrieval, IR) 작업에 미치는 영향을 평가합니다.

- **Technical Details**: Vabamorf는 에스토니아어를 위해 개발된 규칙 기반의 형태소 분석 시스템으로, 각 단어 토큰에 대한 형태소 분석 및 레마 후보들을 생성합니다. 기존의 HMM 기반 불확실성 제거기는 단지 단어의 직전 맥락만을 고려하므로, 문맥에서의 후보 후보군을 줄이는 데 한계가 있습니다. GliNER는 다양한 자연어 레이블을 처리할 수 있는 오픈 어휘 NER 모델로, Vabamorf의 레마 후보군을 평가하는 데 활용됩니다. 이를 통해 기존 HMM 기반 불확실성 제거기의 정확도를 89%에서 97.7%로 크게 향상시킵니다.

- **Performance Highlights**: 리트리벌 설정에서 Vabamorf의 형태소 분석 성능을 비교하면서, 레마화가 단순한 어간 추출(stemming)보다 IR 메트릭에서 약 10% 향상된 결과를 보여줍니다. GliNER을 사용한 불확실성 제거는 이러한 성능을 추가로 1% 가량 향상시켜 주며, 이를 통해 개선된 레마 의사결정의 중요성을 강조합니다. 최종적으로, GliLem의 개발은 에스토니아어를 위한 첫 번째 IR 데이터셋을 생성하고, 효과적인 형태소 분석이 정보 검색 작업에 미치는 긍정적인 영향을 입증합니다.



### Controlling Out-of-Domain Gaps in LLMs for Genre Classification and Generated Text Detection (https://arxiv.org/abs/2412.20595)
Comments:
          The 31st International Conference on Computational Linguistics

- **What's New**: 이번 연구는 최신 대규모 언어 모델(LLMs, 예: GPT-4)이 이전 연구에서 발견된 것과 동일한 도메인 외 성능 격차(out-of-domain performance gap)를 겪고 있음을 보여줍니다. 특히, 우리는 두 가지 비주제별 분류 작업인 장르 분류와 생성된 텍스트 탐지에서 이 성능 저하가 어떻게 발생하는지를 시연합니다. 우리는 In-Context Learning (ICL) 접근 방식으로 서로 다른 도메인에서 성능 저하를 경험하며, 이를 해결하기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 예제 도메인에 따라 분류 성능이 달라지는 현상을 제어하기 위한 방법을 소개합니다. 이 접근 방식은 텍스트의 주제적 특성을 배제하고 스타일적 속성에 초점을 맞추도록 모델을 유도하는 데 효과적입니다. 특히, 장르 분류와 AI 생성 텍스트 탐지라는 두 가지 작업에서 수행하였으며, 각 작업의 OOD 격차를 최대 20%까지 줄이는 데 성공했습니다.

- **Performance Highlights**: 우리의 연구 결과는 기존의 Chain-of-Thought (CoT) 방법론이 충분히 효과적이지 않음을 보여주며, 제안한 접근 방식이 도메인 전이 성능을 지속적으로 향상시킨다는 점에서 의미가 큽니다. 특히, 소수의 예제(few-shot) 설정에서도 두 작업에서 각각 7%와 20%까지 성능 격차를 줄였습니다. 이러한 결과는 LLM들이 다른 도메인에서 잘 작동할 수 있는 가능성을 열어줍니다.



### Towards Neural No-Resource Language Translation: A Comparative Evaluation of Approaches (https://arxiv.org/abs/2412.20584)
- **What's New**: 이 논문은 디지털 표현이 거의 없는 언어인 no-resource 언어의 기계 번역 문제를 탐구하고 있다. 특히 번역 전용 모델의 세밀한 조정(fine-tuning), 대형 언어 모델(LLM)에서의 체인 오프 리즈닝 프롬프트(chain-of-reasoning prompting), 그리고 직접 프롬프트(direct prompting)라는 세 가지 접근 방식을 통해 문제를 다룬다. 사례 연구로 Owens Valley Paiute 언어를 사용하여, 기존의 저자원(low-resource) 언어 번역 접근 방식이 no-resource 언어에 적합하지 않음을 보여준다.

- **Technical Details**: 이 연구에서는 no-resource 언어 번역을 위한 세 가지 신경망 방법론을 엄격히 평가한다. 파라미터 조정으로는 PaLM 모델을 사용하며, QLoRA라는 메모리 효율적인 방법으로 대형 언어 모델을 적응시키는 방식이다. 이를 통해 주어진 문구와 없는 문구에 대한 번역 출력을 생성하며, 다양한 평가 지표(BLEU, ROUGE, METEOR, TER)를 통해 번역의 정확성과 의미 충실도를 평가한다.

- **Performance Highlights**: 체인 오프 리즈닝 프롬프트 방식은 no-resource 언어 번역에서 뛰어난 성능을 보여주었으며, BLEU 값 평균이 0.48로 나타났다. 특히 데이터 셋의 크기가 커질수록 BLEU, ROUGE 및 METEOR와 같은 성능 지표가 일관되게 향상되었다. 직접 프롬프트 방식도 실험에서 괄목할 성과를 보이며，根据 제공된 기법의 능력에 따라 통계적으로 입증할 수 있는 번역 품질을 확보하였다.



### Counterfactual Samples Constructing and Training for Commonsense Statements Estimation (https://arxiv.org/abs/2412.20563)
Comments:
          14 pages, 4 figures

- **What's New**: 이번 논문에서는 언어 모델의 plausibility estimation (PE) 능력을 향상시키기 위한 새로운 방법, 즉 Commonsense Counterfactual Samples Generating (CCSG)를 제안합니다. 기존의 PE 모델들은 언어적 편향 때문에 일반 상식 지식에 효율적으로 대응하지 못하는 경우가 많았습니다. CCSG는 주요 단어를 대체하고 문장 내에서 저수준의 dropout을 도입하여 counterfactual samples를 생성함으로써 모델의 언어 설명 가능성 및 상식 민감성을 강화합니다.

- **Technical Details**: CCSG 방법은 기존의 사전 학습된 지식에 의존하지 않으며, 모델이 관련 언어 영역에 집중할 수 있도록 돕는 중재자로 작용합니다. 이 방법은 low-level dropout과 word-piece 교체를 사용하여 counterfactual samples 생성기를 개발합니다. 또한, 생성된 samples는 문장 수준의 대조 학습(contrastive training) 프레임워크에 통합되어 모델의 학습 과정을 개선합니다.

- **Performance Highlights**: 아홉 개의 다양한 데이터셋에서 실험 결과, CCSG는 기존의 최첨단 방법들과 비교하였을 때 3.07%의 성능 향상을 보였습니다. 이러한 결과를 통해 제안하는 CCSG 방법이 PE 작업에서의 편향을 줄이고 성능 기준을 새롭게 설정하는 효과를 발휘함을 확인하였습니다.



### SAFE-MEME: Structured Reasoning Framework for Robust Hate Speech Detection in Memes (https://arxiv.org/abs/2412.20541)
Comments:
          28 pages, 15 figures, 6 tables

- **What's New**: 이번 연구에서는 기존의 한계점들을 극복하기 위해 두 가지 새로운 멀티모달 혐오 발언 데이터셋인 MHS와 MHS-Con을 소개합니다. 이러한 데이터셋은 일반적인 상황과 혼란스러운 상황에서의 미세한 혐오 표현을 포착합니다. 또한, SAFE-MEME라는 새로운 프레임워크를 제안하여 Q&A 스타일의 추론(사고 체계)과 계층적 분류를 통해 meme에서의 혐오 발언 탐지를 강화하고 있습니다.

- **Technical Details**: SAFE-MEME는 구조적 추론을 기반으로 하여, 다양한 단계의 질문-답변 접근법을 통해 복잡한 추론을 개선합니다. SAFE-MEME-QA는 MHS와 MHS-Con 데이터셋에서 각각 기존 기준보다 평균 5% 및 4% 성능 향상을 달성했습니다. SAFE-MEME-H는 먼저 콘텐츠가 혐오적인지 여부를 판단하고, 그러한 경우에만 더욱 세부적으로 명시적 또는 암시적 혐오 표현으로 분류하는 계층적 접근을 채택하고 있습니다.

- **Performance Highlights**: SAFE-MEME의 구성 요소 중 하나인 SAFE-MEME-H는 MHS 데이터셋에서 평균 6%의 성능 향상을 달성하였으며, 혼란스러운 경우에서만 기존 멀티모달 기준을 초과하였습니다. 단일 레이어 어댑터를 조정하는 접근 방식이 기존의 완전 조정된 모델보다 더 효과적으로 판별력을 높였습니다. 하지만, Q&A 설정을 활용한 완전 조정 접근법은 혼란스러운 사례에 보다 효과적으로 작용한다고 밝혀졌습니다.



### Cut the Deadwood Out: Post-Training Model Purification with Selective Module Substitution (https://arxiv.org/abs/2412.20476)
Comments:
          preprint

- **What's New**: 이번 연구에서는 Greedy Module Substitution (GMS)라는 새로운 방법을 제안합니다. GMS는 백도어(backdoor) 공격이 있는 모델에서 중요한 모듈을 식별하고 대체하여 모델을 정화하는 방식입니다. 이 접근법은 깨끗한 데이터셋이나 보조 모델에 대한 의존성을 줄여 더욱 효율적인 방어 수단을 제공합니다.

- **Technical Details**: GMS는 백도어 경로에 중요한 역할을 하는 'deadwood' 모듈을 탐지하고 교체하여 모델을 수리하는 과정입니다. 기존의 모델 정화 방법들과는 달리, GMS는 깨끗한 데이터셋이나 깨끗한 보조 모델에 대한 의존성을 최소화하며, 대신 Proxy 모델을 활용하여 대체합니다. 이를 통해 파라미터 단위 편집 대신 모듈 단위 정화를 실현합니다.

- **Performance Highlights**: GMS는 RoBERTa-large 모델을 대상으로 한 다양한 설정에서 효과적으로 작동하며, 특히 LWS와 같은 통상적인 공격에 대해 높은 효과를 보였습니다. 예를 들어, SST-2에서 GMS는 LWS의 공격 성공률(ASR)을 9.7%로 줄이는 데 성공했습니다. 이는 기존의 방법보다 현저히 낮은 값으로, GMS의 강력한 방어 성능을 입증합니다.



### Utilizing Multimodal Data for Edge Case Robust Call-sign Recognition and Understanding (https://arxiv.org/abs/2412.20467)
- **What's New**: 이 논문에서는 공중 교통 관제(ATC)에서의 통신 자동화 향상을 위한 새로운 기법인 CallSBERT 모델을 제안합니다. 이 모델은 기존의 CRU(콜사인 인식 및 이해) 모델보다 더 작고 훈련이 빨리 진행되며, 특히 대처하기 어려운 엣지 케이스에서 성능이 향상됩니다. 또한, 콜사인 명령 회복 모델(CCR)을 통해 엣지 케이스의 정확도를 15%까지 증대시킵니다.

- **Technical Details**: CallSBERT 모델은 SBERT 기반의 아키텍처로 24.6M 파라미터를 가지고 있으며, 기존의 EncDec 모델보다 훈련 속도가 4배가량 빨라집니다. 이는 모델 입력 크기를 줄이고, 코사인 유사도를 통해 감지된 다수의 콜사인 정보를 제공하는 것과 관련이 있습니다. CCR은 콜사인 인식과 명령 분류를 결합하여 CRU의 강 robustness를 높입니다.

- **Performance Highlights**: 제안된 CCR 모델은 전체 운영 범위에서 정확도를 획기적으로 향상시킵니다. 다양한 엣지 케이스에서 훈련할 때, 모델이 실제 작업 환경에서 성능 저하를 피할 수 있도록 도와줍니다. 실험 결과, 평균 높은 정확도를 보여주며 엣지 케이스에 특히 강력합니다.



### Enhancing Entertainment Translation for Indian Languages using Adaptive Context, Style and LLMs (https://arxiv.org/abs/2412.20440)
Comments:
          Accepted to AAAI'25

- **What's New**: 본 논문은 엔터테인먼트 분야에서의 신경 기계 번역(NMT)의 도전적인 작업을 다룹니다. 특히 대화의 맥락과 스타일을 유지하면서 목표 언어로 번역하는 방법을 제안합니다. 기존의 NMT 시스템들이 단일 문장을 고립적으로 번역하는 반면, 본 연구에서는 대화의 전체 맥락을 고려한 번역을 강조합니다.

- **Technical Details**: 본 연구에서는 Context And Style Aware Translation (CASAT) 알고리즘을 제안하여, 맥락과 스타일 인식을 통해 대화 내용을 더욱 풍부하게 번역합니다. 이 알고리즘은 시간 의존적인 맥락 정보를 세션으로 나누어 추출하고, 이를 바탕으로 LLM(대형 언어 모델)을 활용하여 번역을 생성합니다. 추가 정보 없이도 문화적 이해를 제공할 수 있는 특징을 가지고 있습니다.

- **Performance Highlights**: 제안된 방법은 여러 숫자적 실험을 통해 높은 효과성을 입증하며, 기존의 LLM들에 비해 COMET 점수가 유의미하게 향상됨을 보여줍니다. 또한, 본 알고리즘은 높은 승률을 유지하며 기초 LLM들보다 일관되게 뛰어난 성능을 나타냅니다.



### Integrating Natural Language Processing Techniques of Text Mining Into Financial System: Applications and Limitations (https://arxiv.org/abs/2412.20438)
Comments:
          6 pages, 5 figures, 1 table

- **What's New**: 이번 연구는 2018년부터 2023년까지의 기간 동안 금융 시스템의 여러 구성 요소에서 자연어 처리(Natural Language Processing, NLP) 기법으로서 텍스트 마이닝의 활용을 탐구하였습니다. 자산 가격 설정(asset pricing), 기업 금융(corporate finance), 파생상품(derivatives), 위험 관리(risk management) 및 공공 금융(public finance) 등 다양한 분야에서의 적용 사례를 다루고 있습니다. 연구는 특정 문제를 논의하며 이를 해결할 필요성을 강조하고 있습니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론은 주로 확률적(probabilistic) 모델과 벡터 공간(vector-space) 모델의 결합입니다. 정보 처리에 가장 많이 사용되는 기술은 정보 분류(information classification) 기법이며, 롱-숏 메모리(long-short term memory)와 양방향 인코더(bidirectional encoder) 모델이 많이 사용됩니다. 연구 결과, 자산 가격 설정에 대한 집중적인 관심과 함께 새로운 알고리즘이 개발되고 있다는 점도 확인되었습니다.

- **Performance Highlights**: 금융 텍스트를 분석해야 하는 연구자들에게 엔지니어링 관점에서의 경로를 제시하고 있으며, 텍스트 마이닝에 관련된 데이터 품질, 맥락 적응(context-adaption), 모델 해석 가능성(model interpretability) 등의 과제를 해결해야 한다고 주장합니다. 이러한 문제들을 해결함으로써, 금융 분석 및 예측을 향상시키기 위한 고급 자연어 처리 모델 및 기법의 통합이 가능할 것입니다.



### Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection (https://arxiv.org/abs/2412.20414)
Comments:
          6 pages, 1 table, AICCONF '24: Cognitive Models and Artificial Intelligence Conference, Istanbul, Turkey

- **What's New**: 이번 논문은 다국어 텍스트에서 지리적 데이터(geospatial data)의 추출 및 분석을 위한 최첨단 자연어 처리(NLP) 방법론과 대규모 언어 모델(LLMs) 통합의 중요성을 다룹니다. 특히, 다양한 NLP 모델(SpaCy, XLM-RoBERTa, mLUKE, GeoLM)과 OpenAI의 GPT 3.5 및 GPT 4를 평가하여 다국어 지리 엔티티 탐지의 맥락에서 이들의 성능을 분석합니다.

- **Technical Details**: 논문에서는 영어, 러시아어, 아랍어의 Telegram 채널에서 수집한 데이터셋을 활용하여 모델의 성능을 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1 scores) 등의 지표를 통해 평가합니다. 각 모델의 장점과 도전 과제를 드러내어 다양한 언어 환경에서 정확한 지리 엔티티 식별(geo-entity identification)의 복잡성을 강조합니다.

- **Performance Highlights**: 실험 결과, 각 모델의 성능 차이를 명확히 하여 지리적 참조 식별에 대한 효과성을 확인하였습니다. 이를 통해 고급 NLP 도구의 개선 및 개발 방향이 제시되어 지리적 분석 및 글로벌 보안(global security) 적용 분야의 발전에 기여하고자 합니다.



### Multi-Objective Large Language Model Unlearning (https://arxiv.org/abs/2412.20412)
- **What's New**: 최근 대형 언어 모델(LLM)에서의 머신 언렀닝(machine unlearning)이 큰 주목을 받고 있으며, 이는 모델의 불필요한 행동을 효과적으로 제거하는 방법을 제공합니다. 본 논문에서는 LLM 언렀닝에서의 Gradient Ascent(GA) 접근 방식을 탐구하며, 대상 데이터의 예측 확률을 감소시켜 영향력을 제거하는 프로액티브한 방법을 제시합니다. 이를 통해 우리는 기존의 문제를 해결하는 Multi-Objective Large Language Model Unlearning(MOLLM) 알고리즘을 제안하여, 모델의 유용성을 유지하면서 대상 데이터를 잊도록 모델을 업데이트합니다.

- **Technical Details**: MOLLM 알고리즘은 LLM 언렀닝을 다중 목표 최적화 문제로 정의하고, 크로스 엔트로피 손실을 언렀닝 버전으로 수정하여 그래디언트 폭발 문제를 해결합니다. 이 프로세스에서는 공통적인 하강 방향(common descent direction)을 계산하여 모델이 목표 데이터를 잊으면서도 LLM의 유용성을 보존할 수 있도록 합니다. 또한, KL 다이버전스(Kullback-Leibler divergence)와 크로스 엔트로피 손실을 통합하여 모델의 성능을 유지하는 방향으로 나아갑니다.

- **Performance Highlights**: 실험 결과, MOLLM은 기존 SOTA GA 기반 LLM 언렀닝 방법들보다 언렀닝 효과와 모델 유용성 유지 측면에서 우수한 성능을 보였습니다. SafeRLHF 데이터셋을 사용해 검증한 결과, 제안된 방법은 언렀닝 효율성과 유틸리티 보존 간의 균형을 잘 유지 көрс합니다. 이는 앞으로의 LLM 언렀닝 연구에 중요한 참고 자료가 될 것입니다.



### Natural Language Fine-Tuning (https://arxiv.org/abs/2412.20382)
- **What's New**: 이 논문에서는 자연 언어를 활용한 자연어 파인튜닝(Natural Language Fine-Tuning, NLFT)이라는 새로운 기법을 소개합니다. NLFT는 특정 도메인에서 제한된 데이터를 가지고 파인튜닝을 가능하게 하며, 모델의 성능을 크게 향상시킵니다. 기존 방법들과 달리 NLFT는 파인튜닝 과정에서 자연어의 지침을 사용하고, 토큰 수준에서의 최적화를 통해 훈련 효율성을 높입니다.

- **Technical Details**: NLFT는 자연어를 감독 신호로 사용하여 데이터를 효과적으로 활용합니다. 이 방법은 또한 warm-up 과정이 필요 없는 최소 데이터 파인튜닝 방법론을 제안하여 기존의 ReFT 방식보다 반복훈련 수를 줄입니다. NLFT는 O(n) 알고리즘 복잡성을 유지하면서 문자 단위의 확률 변화를 평가하고, saliency 토큰을 할당하여 손실 함수를 정제합니다.

- **Performance Highlights**: GSM8K 데이터셋에서 NLFT는 50개의 데이터 샘플만으로 SFT보다 219% 높은 정확도를 달성했습니다. 또한 NLFT는 ReFT보다 훈련 시간과 메모리 사용의 효율성을 각각 78.27%와 92.24% 줄였습니다. 이러한 결과는 NLFT가 리소스가 제한된 환경에서 효과적으로 응용될 수 있는 가능성을 보여줍니다.



### LLM2: Let Large Language Models Harness System 2 Reasoning (https://arxiv.org/abs/2412.20372)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 한계점을 극복하기 위해 LLM2라는 새로운 프레임워크를 소개합니다. LLM2는 인간의 인지 이론인 이중 과정(The dual-process theory)에서 영감을 받아 LLM(시스템 1)과 프로세스 기반 검증기(시스템 2)를 결합하였습니다. 이를 통해 LLM은 출력 후보를 생성하고, 검증기는 적절한 피드백을 제공하여 바람직한 결과와 그렇지 않은 결과를 구분하는 역할을 수행합니다.

- **Technical Details**: LLM2는 훈련 단계에서 쌍 비교 손실(pairwise comparison loss)을 이용하여 바람직한 토큰과 바람직하지 않은 토큰을 구분하도록 최적화된 프로세스 기반 검증기를 포함하고 있습니다. 이 검증기는 각 후보에 대한 시의적절한 피드백을 제공하여 LLM이 생성하는 후보의 품질을 향상시킵니다. 이를 위해 고안된 토큰 품질 탐색 전략을 통해 생성된 합성(process-supervision) 데이터를 사용합니다.

- **Performance Highlights**: 실험 결과, LLM2는 GSM8K 및 MATH와 같은 수학적 추론 데이터셋에서 성능 향상을 보여주었습니다. 예를 들어, Llama3-1B 모델의 경우 GSM8K에서 정확도가 50.3에서 57.8로 (+7.5) 향상되었고, LLM2에 자가 일관성(self-consistency)을 결합했을 때 주요 정확도가 56.2에서 70.2로 (+14.0) 증가했습니다. 이러한 결과는 합성 과정 감독 데이터의 효과성과 가능성을 잘 보여줍니다.



### HindiLLM: Large Language Model for Hind (https://arxiv.org/abs/2412.20357)
- **What's New**: 이번 연구에서는 힌디어 및 다른 인도 언어를 위한 고성능 언어 모델인 HindiLLM-Small 및 HindiLLM-Medium을 사전 훈련했습니다. 두 단계의 과정으로 비지도 사전 훈련과 지도 미세 조정을 수행하여, 텍스트 분류, 감정 분석 등 다양한 작업에서 사용될 수 있는 모델을 구축했습니다. 이를 통해 더욱 효과적으로 인도 언어를 처리할 수 있는 기반을 마련했습니다.

- **Technical Details**: 이 연구에서는 힌디어 텍스트를 위한 큰 데이터세트를 생성하고 BPE(Byte-Pair Encoding) 알고리즘을 사용하여 토크나이저를 훈련했습니다. 이후 라벨 없이 제공된 데이터를 이용한 사전 훈련을 통해 힌디어 모델의 기초를 형성했습니다. 그리고 감정 분석, 텍스트 분류, 자연어 추론과 같은 다양한 작업에 맞춰 모델을 미세 조정했습니다.

- **Performance Highlights**: 평가 결과, HindiLLM 기반의 미세 조정 모델이 여러 언어 관련 작업에서 기존 모델들보다 성능이 우수함을 보여주었습니다. 특히, 다양한 인지적 태스크와 함께 주어진 라벨된 데이터셋에서 높은 정확도를 보였습니다. 이는 힌디어 및 다른 인도 언어에서 NLP의 발전을 이끄는 데 중요한 기여를 할 것으로 기대됩니다.



### Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain (https://arxiv.org/abs/2412.20309)
- **What's New**: 이 연구는 Retrieval Augmented Generation (RAG)을 통해 대규모 언어 모델의 신뢰도에 미치는 영향을 분석합니다. 특히 의료 분야에서 RAG의 구성 및 모델에 따라 신뢰도와 정확성이 어떻게 달라지는지를 평가하고 있습니다. 또한, 문서의 배치가 신뢰도에 어떤 영향을 미치는지도 조사하여, 최신 정보의 활용과 함께 RAG의 가능성을 최대한 발휘하고자 합니다.

- **Technical Details**: 연구는 Medical Information Retrieval Augmented Generation Evaluation (MIRAGE) 디자인에 따라 질문 프롬프트와 답변 옵션을 결합하여 입력을 포맷합니다. 모델의 예측 확률로부터 신뢰도를 계산하여 Expected Calibration Error (ECE) 및 Adaptive Calibration Error (ACE) 점수를 산출하며, RAG 사용 여부에 따라 시나리오를 시뮬레이션하여 결과를 분석합니다. 또한, 문서 삽입 위치에 대한 최적화를 위해 세 가지 패턴을 평가합니다: 질문 이전(Pre-Question), 질문 후(After-Question), 답변 옵션 후(After-Choice).

- **Performance Highlights**: 의료 데이터셋을 분석한 결과, 구성 요소에 따라 RAG가 신뢰도와 정확성에 미치는 영향은 상이합니다. 유의미한 점은 관련 문서를 제공할 때 성능이 향상되며, irrelevant 문서 삽입 시 성능 저하가 발생한다는 점입니다. 이로 인해, 올바른 답변을 지원하는 문서의 신중한 선택이 결과의 신뢰성을 높이기 위해 필수적임을 강조하고 있습니다.



### No Preference Left Behind: Group Distributional Preference Optimization (https://arxiv.org/abs/2412.20299)
- **What's New**: 이번 논문에서는 사람들 사이의 의견 차이를 반영하는 새로운 프레임워크인 Group Distribution Preference Optimization (GDPO)을 제안합니다. GDPO는 기존의 Direct Preference Optimization (DPO)이 다루지 못했던 그룹 내부의 분산된 선호도(Distributional Preferences)를 효과적으로 캡처하는 방법을 소개하여, 개인의 믿음(Beliefs)을 고려한 선호 정렬(preference alignment)을 통해 더 포괄적인 결과를 이끌어냅니다.

- **Technical Details**: GDPO는 두 가지 주요 목표를 최적화하여 선호도 간의 충돌을 해결합니다. 첫째, 그룹의 믿음 분포(Belief Distribution)를 통계적으로 추정하여 모델이 다양한 믿음을 생성하도록 유도합니다. 둘째, 믿음 조건 선호 정렬(Belief-conditioned Preference Alignment)을 도입해 각 선호가 대응하는 믿음에 따라 데이터 쌍을 구성하며, 기존의 DPO와 유사한 방식으로 훈련을 시작합니다.

- **Performance Highlights**: 실험 결과, GDPO는 DPO에 비해 그룹 분포 선호도에 잘 정렬되는 것으로 나타났습니다. 특히, GPT-2 Large 및 Pythia-2.8B 모델을 사용하여 여러 데이터셋에서 검증한 결과, DPO는 목표 분포와의 정렬성에서 분명한 한계를 보였으나, GDPO는 일관되게 이 간격을 줄이며 우수한 성능을 보였습니다. 이러한 성과는 pluralistic alignment 분야에서 중요한 진전을 의미합니다.



### Scoring with Large Language Models: A Study on Measuring Empathy of Responses in Dialogues (https://arxiv.org/abs/2412.20264)
Comments:
          Accepted by IEEE BigData 2024

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)가 공감(score empathy) 점수를 부여하는 능력을 탐구합니다. LLMs는 사람의 대화 패턴을 인식하여 공감을 이해하고 점수를 매기는 능력을 기르지만, 인간의 평가와 비교하여 그 정확성을 확인하는 것이 필요합니다. 이러한 분석을 통해 LLM의 공감 점수 산출 과정에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 LLMs의 공감 카운터 기능을 향상시키기 위해 새로운 프레임워크를 개발했습니다. 이 프레임워크는 MITI(Motivational Interviewing Treatment Integrity) 코드와 대화의 다양한 특성을 반영하여 공감을 평가합니다. 또한, 대화의 임베딩(embeddings)과 MITI 코드의 조합을 통해 분류기를 훈련시키고, 그 성능을 검증합니다.

- **Performance Highlights**: 결과적으로, 임베딩만 사용했을 때도 기본 LLM의 성능에 근접한 결과를 보였으며, MITI 코드와 LLM에 의해 평가된 명시적 서브팩터를 활용했을 때는 세밀 조정된 LLM과 동등한 정확도를 달성했습니다. 이러한 성과는 LLM이 공감을 얼마나 잘 이해하고 측정하는지를 조명하는 데 기여합니다.



### ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty (https://arxiv.org/abs/2412.20251)
- **What's New**: 본 논문에서는 LLM이 필요한 지식의 빈도에 따라 성능이 어떻게 달라지는지를 명확하게 보여주는 ComparisonQA 벤치마크를 소개합니다. 본 벤치마크는 283K 개의 추상 질문으로 구성되어 있으며, 각각 고빈도 및 저빈도의 엔티티 쌍으로 구성이 됩니다. 또한, LLM의 지식 강인성을 평가하기 위해 정확성과 불확실성을 활용한 2단계 방법론을 설계했습니다.

- **Technical Details**: 제안된 ComparisonQA 벤치마크는 고빈도와 저빈도 엔티티를 사용하는 질문 쌍을 통해 지식의 빈도를 비교할 수 있는 제어 가능한 분석을 제공합니다. 각 데이터는 두 엔티티에 대해 동일한 추상 질문을 포함하며, 이는 DBpedia에서의 관계 수에 의해 엔티티의 빈도를 정의합니다. 또한, 저품질 질문 및 세멘틱 숏컷을 제거하기 위한 방법으로 ComparisonQA-Hard 서브셋을 생성하였습니다.

- **Performance Highlights**: 실험 결과 LLM은 저빈도 지식에 대한 강인성이 매우 낮으며, 특히 GPT-4o는 모든 테스트된 LLM 중에서 가장 저조한 성과를 보였습니다. 이는 LLM의 성능이 필요한 지식의 빈도와 밀접하게 관련되어 있음을 나타냅니다. 또한, 불확실성을 기반으로 한 자동 필터링 방법이 질문의 질을 개선하는 데 효과적임을 보여주었습니다.



### LLM Reasoning Engine: Specialized Training for Enhanced Mathematical Reasoning (https://arxiv.org/abs/2412.20227)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 수학적 추론 작업에서의 성능을 향상시키기 위한 새로운 방법론을 제안합니다. 기존의 접근 방식들이 데이터 부족 문제에 의존하는 경우가 많았던 반면, 제안하는 방법은 질문 재구성 전략(question paraphrase strategy)을 포함하여 수학적 질문의 다양한 언어적 형태를 활용하여 일반화 능력을 향상시키고자 합니다. 더불어, 특화된 훈련 목표(training objectives)를 통해 모델의 학습 과정을 길잡이로 삼아 수학 개념과 추론 과정에 대한 이해도를 높이고자 합니다.

- **Technical Details**: 우리의 접근 방식은 질문 재구성을 결합한 것이다. 이 기법은 GPT-4 모델을 사용하여 데이터 세트 내에서 각 질문에 대한 대체 어구를 생성하며, 이를 통해 수학 문제를 해결하는 데 일반적으로 사용되는 보다 다양한 언어적 구조와 표현에 모델을 노출시킵니다. 훈련의 유효성을 평가하기 위해 Llama, Llama2, Mistral, Mixtral과 같은 다양한 LLM을 사용하여 GSM8K, MATH, GSM8K_Hard, SVAMP와 같은 네 가지 수학적 추론 데이터 세트에서 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 LLM의 수학적 추론 작업에서 성능을 향상시키는 데 효과적임을 입증했습니다. 특히, 기존 데이터 세트를 최대한 활용하기 위해 데이터 증강(data augmentation) 기술을 활용하여 LLM의 효율성을 높였습니다. 이러한 결과는 수학적 문제 해결을 위한 LLM의 진전을 나타내며, 향후 머신러닝 분야의 발전과 수학적 추론 능력이 필요한 실제 응용 프로그램 개발에 큰 의미를 가질 것으로 기대됩니다.



### AfriHG: News headline generation for African Languages (https://arxiv.org/abs/2412.20223)
Comments:
          Accepted to AfricaNLP Workshop at ICLR 2024

- **What's New**: 이 논문은 아프리카에서 널리 사용되는 16개 언어를 다룬 뉴스 헤드라인 생성 데이터셋인 AfriHG를 소개합니다. 이는 XLSum 및 MasakhaNEWS 데이터셋을 결합하여 생성되었습니다. 연구 결과, 아프리카 중심의 seq2seq 모델인 AfriTeVa V2가 다국어 mT5-base 모델보다 성능이 뛰어난 것으로 나타났습니다. 또한, 313M 파라미터의 AfriTeVa V2와 13B 파라미터의 Aya-101 LLM을 비교했을 때, AfriTeVa V2가 경쟁력을 갖추었다는 점을 보여주었습니다.

- **Technical Details**: 본 논문에서는 NLP에서 텍스트 요약이라는 주제를 다뤘습니다. 구술 요약(extractive summarization)과 추상적 요약(abstractive summarization)의 두 가지 주요 유형의 텍스트 요약에 초점을 맞췄으며, 뉴스 헤드라인 생성이 추상적 요약의 특별한 형태로 간주됨을 강조했습니다. AfriHG 데이터셋은 BBC, VOA, Isolezwe의 기사-헤드라인 쌍으로 구성되어 있으며, 아프리카에서 널리 사용되는 언어를 포함합니다.

- **Performance Highlights**: 실험 결과, AfriTeVa V2-base 모델은 헤드라인 생성에서 mT5-base 모델보다 일관되게 우수한 성능을 보였습니다. 비라틴 문자로 된 언어들에서 낮은 Rouge 점수를 기록하는 경향이 있지만, 언어 및 스크립트에 따른 토크나이저 사용으로 성능이 개선될 수 있음을 발견했습니다. 마지막으로, Aya LLM과 AfriTeVa V2의 성능이 비슷하게 나타났으며, 이는 훈련 데이터가 풍부할 때 미세 조정(fine-tune)의 장점을 보여줍니다.



### YAD: Leveraging T5 for Improved Automatic Diacritization of Yor\`ub\'a Tex (https://arxiv.org/abs/2412.20218)
Comments:
          Accepted at AfricaNLP Workshop at ICLR 2024

- **What's New**: 이번 연구에서는 Yorùbá 자동 다이아크리티제이션(YAD)에 대한 벤치마크 데이터셋을 제시하여 Yorùbá 다이아크리티제이션 시스템 평가를 가능하게 하였습니다. 또한, 우리는 영문 텍스트를 요루바어로 변환하는 T5 모델을 사전 훈련(Fine-tuning)하였고, 이 모델이 여러 다국어로 훈련된 T5 모델들을 초월하는 성능을 보여주었습니다. 마지막으로 더 많은 데이터와 더 큰 모델이 요루바어의 다이아크리티제이션 성능 향상에 기여한다는 점을 입증했습니다.

- **Technical Details**: 본 연구에서는 요루바어를 위한 T5 모델을 사전 훈련하였으며, 특히 Transformer 아키텍처를 이용하여 텍스트를 텍스트로 변환하는 과정에서의 성능 향상을 중점적으로 다루었습니다. 모델 훈련을 위해 사용된 데이터셋은 요루바어의 다이아크리티제이션 작업에 특별히 설계되었으며, 다양한 멀티링구얼(Multilingual) T5 모델과 비교하여 성능을 평가하였습니다. 데이터의 양(Data size)과 모델의 크기(Model size)가 이 작업에 큰 영향을 미친다는 연구 결과를 도출했습니다.

- **Performance Highlights**: YAD 벤치마크 데이터셋은 요루바어 다이아크리티제이션 시스템의 평가를 위한 최초의 표준화된 기준으로 자리잡을 것으로 기대됩니다. T5 모델은 다이아크리티제이션 정확도(accuracy)에서 현저한 개선을 보여주었으며, 이전 멀티링구얼 모델들에 비해 뛰어난 성능을 나타냈습니다. 이러한 결과는 향후 요루바어 자연어 처리(NLP) 분야에서의 연구와 애플리케이션의 여지를 확장할 것으로 보입니다.



### Decoding Emotion: Speech Perception Patterns in Individuals with Self-reported Depression (https://arxiv.org/abs/2412.20213)
- **What's New**: 이번 연구는 인도 인구 내에서 자가 보고된 우울증과 감정적 음성 인식 간의 관계를 조사한 것으로, PANAS와 PHQ-9를 이용하여 기분과 우울증을 평가하였습니다. 연구는 감정적으로 자극된 음성에 대한 개인들의 감정 반응을 분석하며, 특히 우울증 상태가 이 인식에 미치는 영향을 다루고 있습니다. 이전 연구 결과와는 달리, 우울증 집단에서 긍정적인 감정 반응 감소가 관찰되지 않았다는 점이 주목할 만합니다.

- **Technical Details**: 본 연구에서는 감정적 음성 자극에 대한 감정 반응을 분석하기 위해 대학생 97명을 대상으로 했습니다. 음성 자극은 IEMOCAP 데이터베이스에서 취득하였으며, 네 가지 감정 범주(슬픔, 행복, 분노, 중립)에 따라 세트가 구성되었습니다. 음성 자극은 조작적으로 분류된 라벨과 차원적 주석을 가지고 있어 감정 인식 측정의 신뢰성을 높였습니다.

- **Performance Highlights**: 연구 결과, 우울증 집단은 중립 감정 음성을 제외하고는 어떤 정서 자극에서도 뚜렷한 차이를 보이지 않았습니다. 이는 당초의 예상과 달리 우울증 상태에서 긍정적 감정 반응이 감소하지 않았음을 보여줍니다. 슬픔이나 분노를 묘사하는 음성 자극에서의 감정 반응은 모든 감정 인식 측정 방법에서 일관성을 나타내며, 이는 음성 자극에 대한 감정적 반응의 복잡성을 시사합니다.



### Building a Rich Dataset to Empower the Persian Question Answering Systems (https://arxiv.org/abs/2412.20212)
- **What's New**: 이번 연구에서는 페르시아어를 위한 포괄적인 오픈 도메인 데이터셋인 NextQuAD를 제시합니다. 이 데이터셋은 7,515개의 문맥과 23,918개의 질문 및 답변으로 구성되어 있습니다. 기존의 모델들이 영어에 집중된 반면, 이 연구는 자원이 부족한 언어에 대한 문제 해결을 위한 기초 자료로 활용될 것입니다.

- **Technical Details**: NextQuAD 데이터셋은 BERT 기반의 질문 응답 모델로 활용되며, ParsBERT와 XLM-RoBERTa 두 가지 사전 훈련된 언어 모델이 사용됩니다. 이 두 모델의 결과는 평균 로그 확률(mean logits)로 앙상블(ensemble)됩니다. 개발 세트에서의 평가는 0.95의 Exact Match (EM) 및 0.97의 Fl_score를 나타냅니다.

- **Performance Highlights**: NextQuAD로 훈련된 모델은 PersianQA 및 ParSQuAD의 두 데이터셋과 비교되었습니다. 결과적으로 PersianQA에서는 EM이 0.39, ParSQuAD-manual에서는 0.14 증가한 반면, ParSQuAD-automatic에서는 0.007의 소폭 감소가 발생했습니다. 이러한 성과는 NextQuAD가 페르시아어 질문 응답 시스템의 성능을 개선할 수 있음을 보여줍니다.



### Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering (https://arxiv.org/abs/2412.20145)
- **What's New**: 이 논문에서는 Multi-Agent Collaboration with Tool use (MACT)이라는 새로운 프레임워크를 제안합니다. MACT는 닫힌 소스 모델(closed-source models) 또는 파인 튜닝(fine-tuning)이 필요하지 않으며, 계획 에이전트(planning agent)와 코딩 에이전트(coding agent)가 협력하여 복잡한 질문에 응답합니다. 기존의 TQA 시스템들과 비교하여 MACT는 오픈 웨이트(open-weight) 모델만을 사용하여도 우수한 성능을 발휘합니다.

- **Technical Details**: MACT 프레임워크는 메모리, 계획 에이전트, 코딩 에이전트, 도구 집합으로 구성됩니다. 두 개의 에이전트는 효율 최적화 모듈을 통해 협업하며, 여러 단계의 행동 생성과 선택을 반복 수행합니다. 각 에이전트는 각기 다른 LLM으로 구현될 수 있으며, 이를 통해 다양한 추론 방식에 적합하게 설계되었습니다.

- **Performance Highlights**: MACT는 네 가지 TQA 벤치마크 중 세 개에서 이전의 SoTA 시스템보다 뛰어난 성능을 보였습니다. 특히, MACT는 GPT-4와 유사한 성능을 기록했으며, 파인 튜닝을 필요로 하지 않고 오픈 웨이트 모델로도 충분히 경쟁력 있는 결과를 도출했습니다. 효율성 최적화 모듈을 통해 최대 33%의 반복 횟수를 절약하면서도 성능 저하 없이 작업할 수 있음을 입증했습니다.



### M-MAD: Multidimensional Multi-Agent Debate Framework for Fine-grained Machine Translation Evaluation (https://arxiv.org/abs/2412.20127)
Comments:
          Work in progress. Code and data are available at this https URL

- **What's New**: 이 논문에서는 Multidimensional Multi-Agent Debate (M-MAD)라는 새로운 프레임워크를 제안하며, 이를 통해 기계 번역(MT) 평가에서 LLM(as-a-judge) 접근법을 개선합니다. M-MAD는 평가 기준을 분리하고 여러 에이전트 간의 논쟁을 통해 LLM의 추론 능력을 극대화합니다. 이 새로운 접근법은 기존 방법들보다 더 강력한 성능을 보이며, 다차원 평가로 신뢰성 있는 결과를 제공합니다.

- **Technical Details**: M-MAD는 MQM(다차원 품질 메트릭) 기준을 독립된 평가 차원으로 나누고 각 차원 내에서 다중 에이전트 간의 논쟁을 수행합니다. 이러한 과정은 LLM의 지식과 협동 능력을 활용하여, 최종적으로 토론 결과를 종합하여 종합적인 평가 판단을 생성합니다. 이 접근법은 세 가지 주요 단계로 구성되며, 이는 평가 차원 분할, 다중 에이전트 논쟁, 최종 판단을 포함합니다.

- **Performance Highlights**: M-MAD는 기존 LLM-as-a-judge 방법들과 비교하여 모든 시스템 수준에서 우수한 성능을 발휘하며, GPT-4o mini와 같은 하위 최적 모델로도 최신 참조 기반 자동 메트릭과 유사한 성과를 달성합니다. 종합 실험 결과 M-MAD는 세그먼트 수준 성능에서도 개선된 결과를 보이며, 기존 메트릭들과의 비교에서 높은 일치도를 나타냅니다. 이러한 성과는 LLM-as-a-judge 접근법의 변혁적 잠재력을 보여줍니다.



### Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Datas (https://arxiv.org/abs/2412.20072)
Comments:
          ICASSP 2025

- **What's New**: 이 연구는 기존의 공개된 텍스트와 표 데이터를 독립적으로 처리하는 데 탁월한 성능을 보이는 대형 언어 모델(LLMs)의 능력을 혼합된 장문(홀드) 문서의 정보 추출에 적용하는 방법을 제안합니다. Automated Information Extraction(AIE) 프레임워크를 통해 LLM이 HLD에서 정보를 효과적으로 추출할 수 있는 방법을 모색합니다. 이는 HLD의 문서 처리를 성공적으로 수행하기 위해 다양한 기술적 접근을 포함합니다.

- **Technical Details**: AIE 프레임워크는 세분화(Segmentation), 검색(Retrieval), 요약(Summarization), 추출(Extraction)의 네 가지 모듈로 구성됩니다. Segmentation 모듈은 HLD를 관리 가능한 세그먼트로 나누며, Retrieval 모듈은 임베딩 기반 검색을 통해 키워드와 관련된 세그먼트를 효율적으로 검색합니다. 요약 모듈은 LLM을 사용하여 검색된 세그먼트에서 핵심 정보를 요약하며, 최종적으로 Extraction 모듈은 LLM을 통해 숫자 값을 정확하게 추출합니다.

- **Performance Highlights**: AIE는 세 가지 데이터셋에 대한 실험에서 기본 LLM 기반 접근 방식보다 일관되게 높은 성능을 보였으며, 특히 더 엄격한 기준에서는 그 차이가 더욱 두드러졌습니다. AIE는 다양한 키워드 집합 간의 모호성을 처리하는 데에도 우수함을 보였으며, '수익'과 '총 순자산'과 같은 개념들 간에 의미를 구별하는 데 특히 효과적이었습니다. 실험 결과는 AIE의 구조적 접근이 LLM을 통해 복잡한 데이터에서 정보를 추출하는 데 매우 유용하다는 사실을 입증했습니다.



### Comparative Analysis of Listwise Reranking with Large Language Models in Limited-Resource Language Contexts (https://arxiv.org/abs/2412.20061)
- **What's New**: 본 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 자원 제한이 있는 아프리카 언어에 대한 리스트 방식 재정렬(listwise reranking) 성능을 평가합니다. 우리는 RankGPT3.5, Rank4o-mini, RankGPTo1-mini, RankClaude-sonnet과 같은 독점 모델을 비교했습니다.

- **Technical Details**: 이 연구에서는 평가 지표로 nDCG@10과 MRR@100을 사용해 BM25-DT와 같은 기존의 기준 방법에 비해 성능을 비교합니다. 결과적으로 LLMs는 대부분의 평가 지표에서 전통적인 방법들과 비교하여 우수한 성능을 보였음을 확인했습니다.

- **Performance Highlights**: 이 연구의 결과는 LLMs가 자원이 부족한 언어의 재정렬 작업을 향상시키는 데에 큰 잠재력을 가지고 있음을 보여줍니다. 또한, 본 연구에서는 저비용 솔루션을 제공하여 관련 분야의 연구에 기여할 수 있는 가능성을 제시합니다.



### "My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguis (https://arxiv.org/abs/2412.20057)
Comments:
          Under review at ARR

- **What's New**: 이 논문은 겸손한 과시(humblebragging)를 자동으로 탐지하는 작업을 처음으로 소개하고, 4-튜플 정의를 통해 이 작업을 체계화합니다. 최근 대화형 AI와 소셜 미디어의 발전으로 인해 겸손한 과시에 대한 연구의 필요성이 대두되고 있는데, 이를 통해 인간 언어의 복잡성을 더욱 깊이 이해하려 합니다. 또한, 3,340개의 겸손한 과시 사례로 구성된 HB24라는 새 데이터셋을 개발하고 공개하여 연구를 지원합니다.

- **Technical Details**: 논문에서는 겸손한 과시의 정의를 명확히 하고, 기계 학습(machine learning), 심층 학습(deep learning) 및 최신 대형 언어 모델(large language models, LLMs)의 성능을 비교 평가합니다. 이를 통해 자동 탐지 기술의 정확성을 높이며, 기존의 심리학적 연구와 협력하여 겸손한 과시의 사회적 맥락을 탐구합니다. 연구 결과, 겸손한 과시는 사람들에게도 탐지가 쉽지 않으며, 최상의 모델이 F1-score 0.88을 기록했습니다.

- **Performance Highlights**: 연구에서는 겸손한 과시 탐지가 인간에게도 어려움이 있다는 점을 강조하며, 이 분야에 대한 이해를 심화하는 계기를 제공합니다. 다양한 기계 학습 및 심층 학습 기법이 실험되었으며, 그 성능은 인간의 탐지 능력과 비교됩니다. 이 결과는 향후 사람의 감정 분석(sentiment analysis), 의도 인식(intent recognition) 및 대화 이해(dialogue understanding)와 같은 분야에서도 응용될 수 있는 가능성을 제시합니다.



### STAYKATE: Hybrid In-Context Example Selection Combining Representativeness Sampling and Retrieval-based Approach -- A Case Study on Science Domains (https://arxiv.org/abs/2412.20043)
- **What's New**: 이번 연구에서는 STAYKATE라는 정적-동적 하이브리드 선택 방법론을 제안하며, 이는 액티브 러닝의 대표성 샘플링 원칙과 일반적인 검색 기반 접근법을 결합합니다. 과학 정보 추출에서 중요한 이는 적절한 인-context 예제 선택이 성능에 미치는 영향을 강조합니다. 실험 결과 STAYKATE는 기존의 지도 학습 방법 및 선택 방법들을 초월하는 성과를 확인했습니다.

- **Technical Details**: STAYKATE는 정적(static) 및 동적(dynamic) 예제 선택 전략을 통합하여 작동합니다. 정적 예제는 변하지 않는 고정된 샘플이며, 동적 예제는 테스트 사례와 유사한 샘플을 선택합니다. 이 과정에서 한정된 자원으로 효과적인 예제를 선택하는 것이 중요하며, 과학 영역의 NER에 필요한 전문 지식과 비용 문제를 해결하고자 합니다.

- **Performance Highlights**: 세 가지 도메인 특화 데이터셋에서 STAYKATE의 성능이 가장 뛰어난 것으로 나타났습니다. 이 연구의 결과는 일반적인 오류를 완화하고, 데이터에서 미세한 뉘앙스를 구분하는 모델의 능력을 향상시키는 데 기여하고 있습니다. 이를 통해 STAYKATE는 LLM이 다양한 과학적 NER 작업에서 효과적으로 활용될 수 있음을 보여줍니다.



### OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System (https://arxiv.org/abs/2412.20005)
Comments:
          Work in progress

- **What's New**: OneKE는 웹과 PDF 형식의 책에서 지식을 추출할 수 있는 도커화된 스키마 기반의 지식 추출 시스템으로 소개됩니다. 이 시스템은 다양한 도메인을 지원하며, 여러 개의 에이전트를 설계하여 각자의 역할을 수행하도록 만들어졌습니다. 또한, 구성 가능한 지식 기반이 오류 디버깅과 개선을 돕습니다.

- **Technical Details**: OneKE는 실제 데이터에서 지식을 추출하는 데 필요한 복잡성을 해결하기 위해 설계되었습니다. 스키마 에이전트는 다양한 데이터 유형에 대한 분석을 수행하고, 추출 에이전트는 곧추 사용 가능한 LLM을 이용해 지식을 추출하며, 반영 에이전트는 오류를 디버깅하는 역할을 담당합니다. 이를 통해 다양한 데이터 포맷 및 길이에 유연하게 대응할 수 있는 기능이 있습니다.

- **Performance Highlights**: OneKE는 CrossNER 및 NYT-11-HRL 데이터셋에서 평가되었으며, 두 개의 주요 작업인 NER과 RE 모두에서 성능 향상을 보였습니다. 특히, 추출 에이전트의 사례 검색 방법이 가장 큰 성과를 달성하고 있으며, 복잡한 RE 작업에서는 중간 추론 단계가 더 중요한 것으로 나타났습니다. 이 시스템은 사용자 정의 스키마와의 통합을 통해 더욱 강력한 지식 추출 성능을 발휘합니다.



### Bridging Context Gaps: Enhancing Comprehension in Long-Form Social Conversations Through Contextualized Excerpts (https://arxiv.org/abs/2412.19966)
Comments:
          Accepted at COLING 2025

- **What's New**: 이번 연구는 소규모 기록된 대화에서 이해력을 향상시키는 방법에 중점을 두고 있습니다. 사람들을 연결하고 중요한 사회적 문제에 대한 개인의 이야기를 나눌 수 있는 공간을 제공하기 위해 이러한 대화를 활용합니다. 우리는 대화에서 강조된 발췌(text excerpt)를 후속 대화에서 공유하는 방식을 제안하여 관련 문제에 대한 집단적 이해를 촉진하고자 합니다.

- **Technical Details**: 본 연구에서 다루는 주요 도전 과제는 한 대화에서 발췌한 내용을 다른 맥락에서 공유할 때 발생하는 원본 대화의 맥락이나 핵심 요소의 부족입니다. 이러한 문제는 대화가 길어지고 주제가 풍부해질수록 더욱 악화됩니다. 우리는 Large Language Models (LLMs)를 활용하여 이러한 발췌에 사회적으로 관련된 맥락을 제공하는 방법을 탐색했습니다. 이는 이해력, 가독성(readability), 그리고 공감(empathy) 향상에 도움을 줍니다.

- **Performance Highlights**: 우리 연구는 주관적 및 객관적 평가를 통해 이해력의 상당한 개선을 보여주었습니다. LLMs는 유용한 맥락을 제공할 수 있지만, 중요한 사회적 측면을 포착하는 데 어려움을 겪습니다. 이를 위해 Human-annotated Salient Excerpts (HSE) 데이터셋을 공개하여 향후 연구를 지원하고, 맥락이 풍부한 발췌가 더 집중적이고 포괄적인 대화 요약을 제공할 수 있음을 입증하였습니다.



### Assessing Text Classification Methods for Cyberbullying Detection on Social Media Platforms (https://arxiv.org/abs/2412.19928)
Comments:
          15 pages, 10 figures, 7 tables

- **What's New**: 이번 연구는 사이버 괴롭힘(cyberbullying) 감지 시스템의 성능 문제를 해결하기 위해 기존 텍스트 분류 기법(text classification techniques)을 비교 연구하였습니다. 연구는 BERT, RoBERTa, XLNet, DistilBERT, GPT-2.0과 같은 대형 언어 모델(large language models)을 활용하여 소셜 미디어 플랫폼에서의 사이버 괴롭힘 탐지 성능을 평가하였습니다.

- **Technical Details**: 연구 결과 BERT는 정확도(Accuracy) 95%, 정밀도(Precision) 95%, 재현율(Recall) 95%, F1 점수(F1 Score) 95%를 기록하며 뛰어난 성능을 보였습니다. 또한, 오류율(Error Rate) 5%, 추론 시간(Inference Time) 0.053초, RAM 사용량(RAM Usage) 35.28MB, CPU/GPU 사용량(CPU/GPU Usage) 0.4%, 에너지 소비(Energy Consumption) 0.000263 kWh로 시간 효율성과 вычис적인 자원(c) 사용 측면에서 균형을 이루고 있습니다.

- **Performance Highlights**: 일반적인 생성적 AI(generative AI) 모델들은 테스트된 벤치마크에서 빈번하게 미세 조정된 모델에 비해 뛰어난 성능을 보이지 않았습니다. 그러나 기존 모델들을 특정 데이터셋과 작업에 맞춰 전략적으로 조정하고 미세 조정함으로써 최첨단(state-of-the-art) 성능을 여전히 달성할 수 있음을 확인하였습니다.



### Right vs. Right: Can LLMs Make Tough Choices? (https://arxiv.org/abs/2412.19926)
- **What's New**: 이번 연구는 LLMs(대형 언어 모델)가 윤리적 딜레마를 어떻게 처리하는지를 체계적으로 평가합니다. 연구팀은 1,730개의 윤리적 딜레마 데이터셋을 구축하고, 20개의 LLM을 대상으로 네 가지 주요 가치 쌍에 대한 선호도를 분석했습니다. 이 연구에서는 LLM들이 어떤 윤리적 가치에 보다 민감하게 반응하는지, 그리고 이들 간의 일관성을 어떻게 유지하는지를 살펴봅니다.

- **Technical Details**: 연구에서는 LLM의 윤리적 결정 과정에서의 민감성(sensitivity), 다양한 가치 간의 일관성(consistency), 결과에 대한 고려(consideration of consequences), 명시적으로 또는 암묵적으로 제시된 도덕적 가치 선호에 조화하는 능력을 평가합니다. 이러한 평가를 바탕으로 LLM들이 주요 가치 쌍에 대해 어떤 선호를 가지고 있는지, 그리고 그들이 선택한 행동에 대해 얼마나 신념을 유지하는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: 실험 결과, LLM들은 진실(truth)이 충성(loyalty)보다 우선시되고, 공동체(community)가 개인(individual)보다 더 중요하며, 단기(short-term)보다 장기(long-term) 관련 사항을 우선시한다는 사실이 확인되었습니다. 또한, LLM들은 부정적인 결과가 주어지더라도 의무에 따라 선택을 유지하는 경향이 있으며, 명시적인 가이드라인이 예시보다 더 효과적으로 도덕적 선택을 이끌어낸다고 합니다.



### HADES: Hardware Accelerated Decoding for Efficient Speculation in Large Language Models (https://arxiv.org/abs/2412.19925)
Comments:
          Accepted to ICCEA 2025

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 성능과 에너지 효율성을 향상시키는 새로운 접근법인 Hardware Accelerated Decoding (HADES)을 소개합니다. 기존 문헌에서는 다루어지지 않았던 하드웨어 수준에서의 speculative decoding 지원을 포함하여, LLM 가속기(Accelerator)의 설계를 다룹니다.

- **Technical Details**: HADES는 LLM의 연산 효율성을 크게 향상시키기 위해 speculative decoding 기술을 활용합니다. 이 접근법은 LLM의 대규모와 복잡성으로 인한 계산적 도전을 해결하도록 설계되었습니다.

- **Performance Highlights**: 이 연구는 speculative decoding이 LLM의 작동 효율성을 어떻게 개선할 수 있는지를 보여주며, 이로 인해 보다 발전된 실용적 응용 가능성을 제시합니다. HADES는 LLM의 활용 범위를 넓히는데 기여할 것입니다.



### Evaluate Summarization in Fine-Granularity: Auto Evaluation with LLM (https://arxiv.org/abs/2412.19906)
- **What's New**: 이 논문에서는 정보 소화의 필요성으로 인해 요약의 정확하고 객관적인 평가를 위한 새로운 방법론인 SumAutoEval을 제안합니다. 기존의 ROUGE나 임베딩 유사도 기반 방법들이 인적 평가와 잘 상관되지 않고 비직관적임을 보완하고자 합니다. SumAutoEval은 완전성(Completeness), 정확성(Correctness), 정렬(Alignment), 가독성(Readability) 등 4가지 주요 차원에서 객관적인 점수를 제공합니다.

- **Technical Details**: 제안된 평가 방법은 세 단계로 구성됩니다: 1. 추출(Extraction) - 단문 형태로 노트를 분리하고 각 문장이 고유한 정보를 포함하도록 합니다. 2. 자기 검증(Self-Verification) - 생성된 결과를 검토하고 유사한 개체를 통합합니다. 3. 참고 출처 확인(Reference Sourcing) - 원본 문구를 활용하여 개체를 검증합니다. 이러한 과정은 공정하고 정량적인 방식으로 요약의 품질을 평가할 수 있도록 합니다.

- **Performance Highlights**: SumAutoEval는 요약의 품질 이해를 강화하며, 인적 평가와 더 나은 상관관계를 입증합니다. 이 접근법은 모델이나 프롬프트의 변화에 관계없이 점수의 안정성과 신뢰성을 확보합니다. 따라서, 요약 평가 상황에서 객관적이고 생산적인 결과를 도출할 수 있는 가능성을 제시합니다.



### GaLore$+$: Boosting Low-Rank Adaptation for LLMs with Cross-Head Projection (https://arxiv.org/abs/2412.19820)
- **What's New**: 최근의 저랭크 훈련 방법인 GaLore는 대형 언어 모델(LLM)의 최적화에 필요한 메모리를 상당히 줄여주었습니다. 그러나 GaLore는 특이값 분해(SVD) 단계에서 전체 훈련 시간의 80% 이상을 소비하는 한계가 있습니다. 이를 해결하기 위해 본 논문에서는 다중 헤드 주의력을 위한 크로스 헤드 저랭크 프로젝션을 사용하는 GaLore$+$를 제안합니다.

- **Technical Details**: GaLore$+$는 크로스 헤드 저랭크 프로젝션을 활용하여 시간을 절약하며 신속한 SVD 추정을 달성합니다. 이 방법은 다중 주의력 헤드의 쿼리 또는 키 변환의 프로젝션 행렬을 공유하여 저랭크 프로젝션의 계산 복잡성을 O(h³)에서 O(h)로 줄입니다. 또한, 스파스 코딩된 잔여값을 적용함으로써 저랭크 근사가 초래하는 오류를 감소시킵니다.

- **Performance Highlights**: 실험 결과, GaLore$+$는 산술 추론 및 자연어 생성 데이터셋에서 기존의 최첨단 저랭크 적응 방법들보다 우수한 성능을 보여줍니다. GaLore$+$는 기존의 GaLore에 비해 약 4배의 빠른 미세 조정 속도를 달성하였습니다. 이러한 성과는 언어 모델의 훈련 비용을 크게 절감할 수 있는 가능성을 제시합니다.



### Distributed Mixture-of-Agents for Edge Inference with Large Language Models (https://arxiv.org/abs/2412.21200)
- **What's New**: Mixture-of-Agents (MoA)는 여러 개의 대형 언어 모델(LLM)이 협력적으로 작업하여 퍼포먼스를 향상시키는 방법으로 최근 제안되었습니다. 이는 여러 LLM이 개별 엣지 장치에서 Operation하여 중앙 집중식 서버 없이도 사용자 프롬프트에 대한 응답을 개선하도록 합니다. 본 논문은 정보 교환을 위한 분산 설정에서 MoA 아키텍처를 연구하며, 사용자에 맞춘 LLM을 통해 더 정교한 답변을 생성하는 방법을 모색합니다.

- **Technical Details**: 이 시스템 모델에서는 n명의 사용자가 각자 엣지 장치에서 LLM을 호스팅하고, 이 LLM은 다른 LLM으로부터 받은 프롬프트를 통해 협력을 이루어냅니다. 각 LLM은 FCFS(First-Come-First-Served) 큐에서 프롬프트를 저장하고 처리하여 응답을 생성합니다. 이 과정에서 큐의 안정성을 유지하는 것은 메모리 제약 조건을 고려하며, 프롬프트 생성 속도와 평균 추론 시간을 분석하여 시스템 성능을 평가합니다.

- **Performance Highlights**: 실험을 통해 다양한 MoA 구성의 성능 차이를 검증하였으며, 특정 MoA 설정이 AlpacaEval 2.0 벤치마크 기준에서 더 높은 품질의 응답을 생성하는 것으로 나타났습니다. 이 논문은 LLM의 협업 처리와 모듈 간의 상호작용을 통해 시스템의 최적화를 달성하고, 큐의 안정성 조건을 이론적으로 산출하고 실험적으로 검증하였음을 보여줍니다.



### HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking Code Generation (https://arxiv.org/abs/2412.21199)
- **What's New**: 이번 논문에서는 self-invoking code generation이라는 새로운 과제를 도입하여 LLMs의 점진적 추론 및 문제 해결 능력을 평가합니다. 기존의 벤치마크에 기반하여 더 복잡한 문제를 해결하는 세 가지 새로운 벤치마크인 HumanEval Pro, MBPP Pro, 및 BigCodeBench-Lite Pro를 제안합니다. 연구 결과에 따르면 대부분의 LLMs가 전통적인 코드 생성 작업에서는 높은 성능을 보이지만 self-invoking 작업에서는 성능이 저하되는 경향을 보입니다.

- **Technical Details**: 이 연구는 LLMs가 기본 문제를 해결하고 그 결과를 활용하여 더 복잡한 문제를 해결하도록 요구하는 구조로 되어 있습니다. 이를 통해 LLMs의 문제 해결 능력을 보다 현실적으로 평가할 수 있습니다. 벤치마크 구성은 Deepseek-V2.5를 활용하여 생성된 self-invoking 문제와 후보 솔루션, 테스트 입력을 포함하여 수행됩니다.

- **Performance Highlights**: 실험적으로 LLM들 간의 성능 격차가 눈에 띄며, 예를 들어 o1-mini 모델은 HumanEval에서 96.2%의 통과율을 기록했지만 HumanEval Pro에서는 76.2%로 낮아지는 모습을 보였습니다. 또한 instruction-tuned 모델들이 전통적인 코드 생성 작업에 비해 self-invoking 코드 생성 작업에서 효율성이 떨어짐을 발견하였습니다. 이러한 결과는 LLMs의 코드 추론 능력 향상에 대한 필요성을 강조합니다.



### Two-component spatiotemporal template for activation-inhibition of speech in ECoG (https://arxiv.org/abs/2412.21178)
- **What's New**: 이 연구에서는 여러 피험자를 대상으로 다중 채널 고밀도 전두전기적 측정법 (ECoG)을 통해 연령대 간의 음성 활동의 평균적인 힘을 계산하였다. 특히 음성 운동 중 관찰되는 베타 주파수 활동과 고주파 감마 활동 간의 반상관 관계를 각 ECoG 채널 간 비교하였으며, 이는 운동 제어에 대한 새로운 통찰을 제공한다.

- **Technical Details**: 연구에서는 ECoG 데이터를 사용하여 감마 (Gamma) 및 베타 (Beta) 대역에서의 주파수 활동을 분석하였다. 특히, 주요 성분 분석 (PCA)을 통해 ECoG 채널의 밴드파워를 모델링하고, 스페이셔 템포럴 관계를 확인하기 위해 창 기반 상관 분석을 수행하였다. 결과적으로, 두 가지 구성 요소가 음성 운동 중 SMC 활동을 나타내기에 충분하다는 점이 확인되었다.

- **Performance Highlights**: 이 모델은 SMC의 활동에 대한 두 가지 구성 요소의 활성화-억제 구조를 유사하게 재현하였다. 연구 결과는 최근 전체 신체의 운동 제어, 억제 및 자세 조절에 있어 복잡한 상호작용을 보여주는 지역 SMC의 활성화 구역과 유사하게 나타났다. 특히, 세 번째 주요 성분은 모든 피험자 간에 비유의미한 상관관계를 보여 주목할 만하다.



### Aviary: training language agents on challenging scientific tasks (https://arxiv.org/abs/2412.21154)
- **What's New**: 이 논문에서는 언어 에이전트를 위한 확장 가능한 툴인 Aviary를 소개합니다. Aviary는 언어 기반의 부분 관측 마르코프 의사결정 과정(language-grounded partially observable Markov decision processes), 즉 언어 의사결정 프로세스(language decision processes)로 에이전트를 정의합니다. 새로운 환경을 통해 DNA 조작, 과학적 문헌의 질문 답변, 단백질 안정성 공학과 같은 과학적 문제를 해결할 수 있도록 설계되었습니다.

- **Technical Details**: Aviary는 언어 에이전트의 다양한 구성 요소와 최적화 방법들을 모듈 방식으로 교환할 수 있는 소프트웨어 패키지로, stochastic computation graphs의 개념을 도입합니다. 이 논문에서는 언어 에이전트를 강화하는 과정을 통해 에이전트의 작업 효율성을 높이는 여러 최적화 알고리즘을 설정합니다. 구체적으로, 전문가 반복(expert iteration) 방식 및 인퍼런스-타임 다수결 샘플링을 통해 작은 오픈 소스 LLM 기반의 에이전트가 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, Aviary의 언어 에이전트는 전문가 및 최첨단 LLM보다 최대 100배 더 낮은 인퍼런스 비용으로 여러 과제를 수행할 수 있습니다. 특히 DNA 구성물 조작 및 과학적 문헌 질문 답변 환경에서 작은 모델이 동작을 수행할 수 있어 강력한 성능을 보여주었습니다. 이러한 결과들은 해당 연구의 적용 가능성을 높이고 언어 기반 에이전트의 활용 가능성을 한층 확장시킵니다.



### Training Software Engineering Agents and Verifiers with SWE-Gym (https://arxiv.org/abs/2412.21139)
Comments:
          Code at this https URL

- **What's New**: SWE-Gym은 실제 소프트웨어 공학(SWE) 에이전트를 교육하기 위한 최초의 환경으로, 2,438개의 실제 Python 작업을 포함하고 있습니다. 각 작업은 실행할 수 있는 런타임 환경, 단위 테스트 및 자연어로 지정된 작업을 포함하여, 실질적인 엔지니어링 문제를 다룰 수 있도록 설계되었습니다. SWE-Gym을 사용하여 언어 모델 기반 SWE 에이전트를 교육함으로써, SWE-Bench Verified 및 Lite 테스트 세트에서 해결 비율이 최대 19% 향상되었습니다.

- **Technical Details**: SWE-Gym은 11개의 인기 있는 오픈 소스 저장소에서 가져온 2,438개의 Python 작업으로 구성되며, 이를 통해 LMs을 에이전트 및 검증기로 훈련할 수 있는 유용한 환경을 제공합니다. OPEN-HANDS 에이전트 프레임워크를 기반으로 32B Qwen-2.5 모델을 조정하여 SWE-Gym의 에이전트-환경 상호작용 경로에서 샘플링한 491개의 경로를 사용하여 상당한 향상을 달성했습니다.

- **Performance Highlights**: SWE-Gym을 활용한 강화 학습을 통해, SWE-Bench Verified에서 32.0%와 SWE-Bench Lite에서 26.0%의 성과를 기록하며 공개 가중치 시스템 중 새로운 최첨단 상태를 세웠습니다. 이는 대규모 샘플링을 통해 계속해서 개선되고 있으며, 성능 개선은 샘플 수의 증가와 결합하여 더욱 확실해졌습니다.



### TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization (https://arxiv.org/abs/2412.21037)
Comments:
this https URL

- **What's New**: 이번 논문에서는 515M 매개변수를 가진 효율적인 텍스트-오디오(TTA) 생성 모델인 TangoFlux를 소개합니다. TangoFlux는 단일 A40 GPU에서 3.7초 만에 최대 30초 짜리 44.1kHz 오디오를 생성할 수 있습니다. TTA 모델의 정렬 과정에서 발생하는 문제를 해결하기 위해 새로운 접근법인 CLAP-Ranked Preference Optimization (CRPO)를 제안하여 오디오 생성의 품질을 대폭 향상시켰습니다. 모든 코드와 모델이 오픈 소스로 공개되어 후속 연구를 지원합니다.

- **Technical Details**: TangoFlux는 FluxTransformer 블록으로 구성되며, 이는 Diffusion Transformer(DiT) 및 Multimodal Diffusion Transformer(MMDiT)에 따라 조건부 텍스트 프롬프트와 지속 시간 임베딩을 사용하여 44.1kHz에서 최대 30초의 오디오를 생성합니다. TangoFlux의 훈련 파이프라인은 세 단계로 나뉘며, 이는 사전 훈련(pre-training), 미세 조정(fine-tuning), 그리고 선호 최적화(preference optimization)로 구성됩니다. 또한, CRPO를 통해 새로운 합성 데이터를 반복적으로 생성하고 선호 쌍을 구성하여 오디오 생성 결과의 품질을 높입니다.

- **Performance Highlights**: TangoFlux는 기존의 오디오 선호 데이터셋 대비 우수한 성능을 자랑하며, 벤치마크와 비분배(out-of-distribution) 프롬프트에서 최첨단 성능을 달성합니다. 이 모델은 생성된 오디오의 품질을 높이기 위해 사용된 수정된 손실 함수(modified loss function)을 통한 반복 최적화가 효과적이라는 것을 입증했습니다. 또한, TangoFlux는 비독점 데이터로 훈련되어 다양한 길이의 오디오 생성을 지원하는 뛰어난 능력을 보여줍니다.



### Efficiently Serving LLM Reasoning Programs with Certaindex (https://arxiv.org/abs/2412.20993)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 효율적인 추론을 위한 Dynasor라는 새로운 시스템을 제안합니다. Dynasor는 각 추론 쿼리와 관련된 입력/출력 요청의 일정을 제어할 수 있어, 다양한 쿼리의 난이도에 따라 동적으로 컴퓨트(계산 자원)를 할당합니다. 이를 통해 정확도와 비용을 균형 있게 조절하면서, 응답 지연(latency)과 같은 성능 목표를 충족시킵니다.

- **Technical Details**: Dynasor는 'certaindex'라는 새로운 프록시 변수를 사용하여 LLM의 추론 진행 상태를 측정합니다. 이는 LLM이 최종 정답에 얼마나 확신을 가지고 있는지를 평가할 수 있도록 도와주며, 더 많은 컴퓨트를 할당해야 할 지 아니면 줄여야 할지를 결정하는 데 사용됩니다. Dynasor는 추론 쿼리에 대한 컴퓨트를 조정할 뿐만 아니라 요청 일정도 최적화하여 시스템의 효율성을 높입니다.

- **Performance Highlights**: Dynasor는 다양한 데이터셋과 알고리즘에서 평가된 결과, 배치 처리(batch processing)에서 최대 50%의 컴퓨트를 줄이면서 동일한 정확도로 도달할 수 있다는 점이 입증되었습니다. 온라인 서비스에서는 이전보다 3.3배 더 많은 쿼리 요청을 처리하거나, 4.7배 더 엄격한 지연 시간 SLOs를 유지할 수 있어서, 효율성과 성능 모두를 향상시켰습니다.



### Enhancing Multimodal Emotion Recognition through Multi-Granularity Cross-Modal Alignmen (https://arxiv.org/abs/2412.20821)
Comments:
          ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)

- **What's New**: 이 논문에서는 멀티모달 감정 인식(Multi-modal Emotion Recognition, MER)을 위해 새로운 Multi-Granularity Cross-Modal Alignment (MGCMA) 프레임워크를 도입했습니다. MGCMA는 분포 기반, 인스턴스 기반, 토큰 기반의 정렬 모듈을 포함하여 감정 정보를 다층적으로 인식할 수 있도록 설계되었습니다. 이는 감정 표현의 복잡성과 모호성을 해결하기 위한 접근을 제공함으로써, MER의 성능 한계를 극복하고자 합니다.

- **Technical Details**: MGCMA 프레임워크는 3개의 주요 모듈로 구성됩니다. 첫째, 분포 기반 정렬 모듈은 코스 그레인(corresponding grain) 정렬을 위해 분포 수준의 대조 학습(distribution-level contrastive learning)을 사용합니다. 둘째, 토큰 기반 정렬 모듈은 셀프 어텐션(self-attention) 및 크로스 어텐션(cross-attention) 메커니즘을 활용하여 음성과 텍스트 사이의 로컬 교차 모달 표현을 정확히 일치시키고자 합니다. 셋째, 인스턴스 기반 정렬 모듈은 특정 음성-텍스트 쌍 간의 매핑 관계를 학습합니다.

- **Performance Highlights**: IEMOCAP 데이터셋에서 실시한 실험 결과, 제안한 MGCMA 프레임워크는 기존의 최신 기술보다 우수한 성능을 보였습니다. 무게 정확도(Weighted Accuracy, WA)는 78.87%, 비무게 정확도(Unweighted Accuracy, UA)는 80.24%를 달성하여, 감정 인식의 정확성이 크게 향상됨을 입증했습니다.



### HUNYUANPROVER: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving (https://arxiv.org/abs/2412.20735)
- **What's New**: HunyuanProver는 Hunyuan 7B에서 파인튜닝된 언어 모델로, LEAN4를 활용한 대화형 자동 정리 증명(auto theorem proving)을 위한 새로운 프레임워크를 제시합니다. 이 모델은 데이터 희소성 문제를 해결하기 위해 저비용으로 데이터를 반복적으로 합성하는 확장 가능한 프레임워크를 디자인하였으며, 효과적인 'System 2 thinking'을 가능하게 하는 가이드 트리 탐색 알고리즘도 포함하고 있습니다. HunyuanProver는 주요 기준 점에서 최첨단 성능(state-of-the-art, SOTA)을 기록하고 있으며, 커뮤니티에 기여하기 위해 30,000개의 합성된 인스턴스의 데이터셋을 오픈소스할 계획입니다.

- **Technical Details**: HunyuanProver는 두 가지 핵심 모듈로 구성되어 있습니다: 확장 가능한 증명 데이터 생성기와 가이드 트리 탐색 알고리즘입니다. 증명 데이터 생성기는 오픈소스 데이터를 활용하여 초기 자동 형식화(autoformalizer)와 증명기(prover) 모델을 훈련합니다. 그런 다음, 새로운 증명 기초 데이터는 각 반복(iteration)에서 생성되어 증명기를 훈련시키며, 트리 탐색 알고리즘과 여러 비평(critic) 모델이 사용되어 복잡한 정리 증명 작업을 해결하기 위한 '느린 사고(slow thinking)'를 수행합니다.

- **Performance Highlights**: HunyuanProver는 miniF2F 기준 점에서 68.4%의 정확도를 기록하며 기존 최고 기록인 65.9%를 초과하였습니다. 모델은 miniF2F-test에서 4개의 IMO 진술(imo_1960_p2, imo_1962_p2, imo_1964_p2, imo_1983_p6)을 증명했습니다. 또한 데이터 양에 따른 파인튜닝 효과와 비평 모델을 활용한 트리 탐색의 유용성이 강조되었습니다.



### ChartAdapter: Large Vision-Language Model for Chart Summarization (https://arxiv.org/abs/2412.20715)
- **What's New**: 이번 연구에서는 차트 요약(chart summarization)을 위한 새로운 경량 변환기 모듈인 ChartAdapter를 제안합니다. ChartAdapter는 차트 데이터에서 암묵적인 의미를 추출하기 위한 학습 가능한 쿼리 벡터를 사용하고, 비전-언어 생성 학습을 향상시키기 위한 크로스 모달 정렬 프로젝터를 통합합니다. 이 모듈은 차트 요약을 위한 최적화된 end-to-end 학습을 가능하게 하며, 190,618개의 샘플로 구성된 대규모 데이터셋을 개발하여 효과적인 훈련을 지원합니다.

- **Technical Details**: ChartAdapter는 시각 인코더(visual encoder)와 언어 모델(language model) 간의 협력을 촉진하여 차트 요약을 효율적으로 수행합니다. 이 모듈은 크로스 모달 프로젝터(cross-modal projector), 잠재 텍스트 임베딩(latent textual embeddings), 크로스 모달 상호작용 레이어(cross-modal interaction layer), 암묵적 의미 디코더 레이어(implicit semantic decoder layer) 등 네 가지 기본 구성 요소로 이루어져 있습니다. 이러한 구성 요소들은 다중 모달 임베딩을 정렬하고 통합하는 데 기여하며, 시각적 정보를 추출하고 크로스 모달 상호작용을 촉진하여 최종적으로 텍스트 요약으로 디코딩합니다.

- **Performance Highlights**: 실험 결과, 제안된 ChartAdapter는 기존의 최첨단 모델보다 차트 요약의 질을 크게 향상시킴을 보여줍니다. 표준 Chart-to-Text 테스트 세트에서 실시된 실험은 ChartAdapter의 효과성을 입증하며, 주요 구성 요소들이 차트 이해 및 해석의 발전 가능성을 높임을 강조합니다. 최종적으로, 차트 요약을 위한 맞춤형 LLM 기반 접근의 잠재력이 크게 부각되었습니다.



### UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design (https://arxiv.org/abs/2412.20694)
- **What's New**: 이번 논문에서는 NP-hard 문제 해결을 위한 새로운 방법론인 UBER(불확실성 기반 진화법)를 소개합니다. UBER는 기존의 FunSearch 프레임워크 위에 불확실성을 통합하여 LLM(대형 언어 모델)과 EA(진화 알고리즘)의 결합을 강화합니다. 이는 자동 휴리스틱 설계의 효율성을 크게 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: UBER는 두 가지 주요 혁신을 도입합니다. 첫째, UIEP(Uncertainty-Inclusive Evolution Process)는 탐색-활용(balance of exploration and exploitation)을 적응적으로 조정하여 더 효과적인 문제 해결을 가능하게 합니다. 둘째, UIIS(Uncertainty-Inclusive Island Reset) 전략은 인구 다양성을 유지하여 알고리즘의 성능을 높입니다.

- **Performance Highlights**: 광범위한 NP-complete 문제에 대한 실험을 통해 UBER는 FunSearch에 비해 상당한 성능 개선을 보여줍니다. 이 연구는 LLM과 EA의 상호 작용에 대한 새로운 방향을 제시하며, 자동 휴리스틱 설계 분야에 기여할 것입니다.



### The Impact of Prompt Programming on Function-Level Code Generation (https://arxiv.org/abs/2412.20545)
Comments:
          CodePromptEval dataset and replication package on GitHub: this https URL

- **What's New**: 이 연구에서는 7072개의 프롬프트로 구성된 CodePromptEval 데이터세트를 소개하여 다섯 가지 프롬프트 기법(예: few-shot, persona 등)과 그 조합이 LLM의 코드 생성의 정확성, 유사성, 품질에 미치는 영향을 평가합니다. 여러 프롬프트 기법의 조합이 반드시 결과 개선으로 이어지지 않는다는 것을 발견했습니다. 또한 프롬프트 기법 사용 시 정확성과 품질 간의 트레이드오프를 관찰했습니다.

- **Technical Details**: LLMs는 소프트웨어 엔지니어들에 의해 코드 생성 및 완성 작업에 널리 사용되고 있지만, 비관련 코드 생성이나 오류와 같은 한계가 여전히 존재합니다. 본 연구는 LLM 생성 코드의 정확성과 품질을 평가하기 위해 프롬프트 기법(특히 few-shot, function signature 등)의 조합을 전면적으로 실험합니다. CodePromptEval 데이터세트를 기반으로 GPT-4o, Llama3, Mistral와 같은 LLM를 사용하여 함수 생성을 진행하고, 그 결과를 기반 진리와 비교하여 평가합니다.

- **Performance Highlights**: 연구 결과, 특정 프롬프트 기법이 LLM의 코드 생성에 중요한 영향을 미치지만, 여러 기법의 조합이 필수적으로 성과를 향상시키는 것은 아니라는 점을 확인했습니다. GPT-4o는 세 가지 모델 중 가장 높은 정확도를 보였으나 성과의 차이는 약 5%에 불과했습니다. 코드 생성에서 함수 서명이나 few-shot 예제만으로도 정확성이 크게 향상될 수 있음을 알 수 있었고, 코드의 품질과는 종종 상반된 결과를 나타냅니다.



### ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding (https://arxiv.org/abs/2412.20504)
- **What's New**: 이번 연구에서는 비디오 이해에서 Video Large Language Models (VideoLLMs)의 한계를 극복하기 위해 새로운 방법인 ReTaKe를 제안합니다. 기존 VideoLLM들은 긴 비디오를 처리하는 데 어려움을 겪고 있으며, ReTaKe는 두 가지 새로운 모듈인 DPSelect와 PivotKV를 통해 이를 해결합니다. 이 방법은 비디오에서 시간적 시각 중복과 지식 중복을 모두 효과적으로 모델링하여 긴 비디오 이해를 지원합니다.

- **Technical Details**: ReTaKe는 훈련 없는 방식으로, DPSelect는 비주얼 특징에 기반하여 키프레임을 선택하고, PivotKV는 선택된 키프레임을 피벗으로 사용하여 저주의 높은 점수를 가진 비피벗 토큰의 KV 캐시 압축을 수행합니다. 이러한 접근은 기존의 방법들보다 더 정교하게 비디오의 정보 밀도를 높이고, 긴 비디오에서의 이해도를 향상시킵니다. 실험 결과, ReTaKe는 기존 VideoLLM보다 4배 긴 비디오 시퀀스를 처리할 수 있으며, 성능 저하는 1% 미만입니다.

- **Performance Highlights**: 다양한 벤치마크 테스트에서 ReTaKe는 MiniCPM-V2.6-8B 및 LLaVA-OneVision-7B와 같은 유사 크기의 VideoLLM을 3%-5% 향상시켰으며, 더욱 큰 모델들인 InternVL2-34B 및 VideoLLaMA 2-72B와 맞먹거나 초과하는 성과를 보였습니다. DPSelect와 PivotKV의 효과를 검증하는 아블레이션 연구에서는, DPSelect가 행동 추론과 속성 인식 작업에서 중요한 저수준 시각 세부 정보를 더 잘 보존하는 것을 입증했습니다.



### A Multidisciplinary Approach to Telegram Data Analysis (https://arxiv.org/abs/2412.20406)
Comments:
          7 pages, 1 table, 2 figures, 24th International Multidisciplinary Scientific GeoConference SGEM 2024

- **What's New**: 이번 논문은 텔레그램에서 수집한 데이터를 분석하여 사이버 위협에 대한 조기 경고 정보 생성에 기여하는 다학제적 접근법을 제시합니다. 해커 집단(hacktivist groups)이 사이버 공격에 대한 정보를 전파하는 방식이 증가함에 따라, 관련 위험을 식별하기 위한 효과적인 데이터 분석 방법의 필요성이 커졌습니다.

- **Technical Details**: 본 연구에서는 신경망 아키텍처(neural network architectures)와 전통적인 기계 학습 알고리즘(traditional machine learning algorithms)의 조합을 사용하여 텔레그램 데이터 내 사이버 위협을 분류하고 식별합니다. 또한 감정 분석(sentiment analysis)과 개체 인식(entity recognition) 기법을 도입하여 전달된 정보의 성격과 맥락에 대한 심층적인 통찰력을 제공합니다.

- **Performance Highlights**: 연구는 각 방법이 사이버 위협을 탐지하고 분류하는 효과를 평가하며, 성능 비교와 개선 가능성을 도출합니다. 다양한 분석 도구를 활용하여 사이버 위협의 조기 경고 시스템을 강화하고 잠재적인 보안 침해에 대한 보다 적극적인 대응을 가능하게 하는 것을 목표로 하고 있습니다.



### Enhancing Code LLMs with Reinforcement Learning in Code Generation (https://arxiv.org/abs/2412.20367)
- **What's New**: 본 논문은 강화 학습(Reinforcement Learning, RL)이 코드 생성 및 최적화에서의 역할을 체계적으로 조망합니다. 특히 컴파일러 최적화, 자원 할당 및 프레임워크 및 도구 개발에서 RL의 중요한 적용 사례를 강조하고 있습니다. 이를 통해 연구자들이 RL의 힘을 활용하여 코드 생성 및 최적화 기법을 발전시키는 방안을 제시하고 있습니다.

- **Technical Details**: 코드 LLM의 코드 생성은 자연어 설명에서 실행 가능한 코드를 생성하는 핵심 작업입니다. 강화 학습(RL) 방법은 주어진 환경에서 최적의 전략을 학습할 수 있으며, LLM이 진화하는 환경에 더 유연하게 적응할 수 있도록 도와줍니다. 특히 PPO와 Actor-Critic 프레임워크와 같은 정책 기반 방법이 강화 학습의 주요 포인트로 강조되며, 이는 행동 실행자와 평가자로 구성되어 있습니다.

- **Performance Highlights**: 강화 학습 기반 피드백(RLHF)는 대규모 언어 모델을 인간의 선호도와 특정 작업 목표에 맞게 조정하는 데 중요한 알고리즘 전략으로 자리 잡고 있습니다. 코드 생성의 맥락에서 unit test 피드백을 보상 신호로 활용함으로써, 생성된 코드가 실제 기능 요구 사항에 더 가까워지는 결과를 기대할 수 있습니다. 이를 통해 RL 기반 방법이 코드 생성에서 더욱 정밀하고 적합한 결과를 생성할 수 있는 가능성을 보여줍니다.



### On the Compositional Generalization of Multimodal LLMs for Medical Imaging (https://arxiv.org/abs/2412.20070)
- **What's New**: 이번 연구는 Multimodal Large Language Models (MLLMs)이 의료 분야에서 이미지 일반화의 능력을 탐구하는 데 중요한 역할을 하고 있음을 강조합니다. MLLMs의 한계는 불충분한 데이터에 기인하므로, 어떤 종류의 이미지가 MLLMs의 일반화에 유용한지 이해할 필요가 있습니다. 본 연구는 Composition Generalization (CG)를 사용하여 MLLMs가 의료 이미지를 이해할 수 있는 방법을 제안하며, Med-MAT라는 새로운 데이터셋을 개발했습니다.

- **Technical Details**: Med-MAT는 106개의 의료 데이터셋을 조합하여 생성된 데이터셋으로, 각 데이터는 Modality, Anatomical Area, 그리고 Task의 세 가지 요소인 MAT-Triplet으로 명확하게 정의되었습니다. 이 데이터셋은 의료 이미지를 기반으로 CG를 탐구할 수 있는 기회를 제공합니다. 연구자들은 데이터의 관련성을 분석하기 위해 MLLMs의 성능을 비교하고, CG가 다양한 이미지 유형에서 모델의 일반화 성능에 미치는 영향을 조사하였습니다.

- **Performance Highlights**: 실험 결과, MLLMs는 CG를 활용하여 이전에 보지 못한 의료 이미지를 이해할 수 있으며, CG가 멀티태스크 훈련에서 나타나는 일반화의 주요 요인 중 하나라는 점이 확인되었습니다. CG는 데이터 수가 적은 경우에도 유용하게 작용해 다양한 MLLMs 구조에서 일관된 성능을 제공함을 보여주었습니다. 이러한 결과는 의료 영역에서 MLLMs의 성공적인 활용 가능성을 제시합니다.



### The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Suppor (https://arxiv.org/abs/2412.20068)
- **What's New**: 본 논문에서는 정신 건강 지원을 위한 새로운 심리 평가 접근 방식을 기반으로 한 대화형 AI 시스템인 RACLETTE를 개발하였습니다. 이 시스템은 사용자의 감정 상태를 이해하고 공감적 반응을 생성하는 데 있어 기존의 최신 기술보다 우수한 성능을 보입니다. 또한, 사용자의 감정 프로필을 구축하는 과정에서 기존의 대화 기술을 증진시키는 가능성을 보여줍니다.

- **Technical Details**: RACLETTE는 3턴 구조를 채택하여 사용자의 감정을 다음 토큰 예측으로 훈련시키고, Mistral 7B 모델의 생성 능력을 활용하여 공감적으로 반응합니다. 사용자의 감정 프로필은 대화 중에 실시간으로 업데이트되며, 이는 시스템이 사용자의 감정 상태를 보다 정확히 이해할 수 있도록 합니다. 이러한 감정 프로필은 다양한 정신 장애와 통계적으로 비교될 수 있는 설명 가능한 지표로 활용됩니다.

- **Performance Highlights**: 연구 결과, RACLETTE는 감정 분류 작업에서 높은 정확도를 보이며, 공감적 응답 생성의 질 또한 탁월함을 입증했습니다. 감정 프로필을 통해 사용자 간의 다양한 감정 패턴을 인식하고 이를 기반으로 초기 진단 및 검사에 활용할 수 있는 가능성을 제시합니다. 이러한 접근 방식은 정신 건강 문제의 조기 발견과 진단에 기여할 것으로 예상됩니다.



### BaiJia: A Large Scale Role-Playing Agent Corpus of Chinese Historical Charcaters (https://arxiv.org/abs/2412.20024)
- **What's New**: BaiJia라는 혁신적인 대규모 역사적 캐릭터 역할 놀이 에이전트 데이터 집합이 소개되었습니다. 이 데이터 집합은 다양한 중국 역사 인물에 대한 정보를 통합하여 LLMs(대형 언어 모델)가 AI 주도 역사 역할 놀이에 참여할 수 있도록 설계되었습니다. BaiJia는 과거의 산재된 역사 텍스트 기록의 문제를 해결하며, 인물의 전기적 정보나 문학적 작품 등을 포함하고 있습니다.

- **Technical Details**: BaiJia 데이터 집합은 데이터 수집, 대화 생성 및 모델 평가의 세 가지 주요 단계로 구성됩니다. 이 과정에서 우리는 다섯 개의 주요 중국 왕조(Tang, Song, Yuan, Ming, Qing)에서 총 19,281명의 역사적 인물의 이력서를 수집하고, 이를 바탕으로 대화를 생성하였습니다. 또한, LLMs의 역할 놀이 능력을 평가하기 위해 15개의 주제 측면에서 질문 데이터 집합을 구축하여 평가 기준을 마련했습니다.

- **Performance Highlights**: BaiJia 데이터 집합을 활용한 여러 LLM의 실험 결과, LLM에 인물 이력서 정보를 통합함으로써 역할놀이 능력이 각 평가 차원에서 유의미하게 향상되었습니다. 특히, 역할 놀이에 특화된 LLM조차도 역사적 캐릭터 표현에 한계를 보였지만, BaiJia는 이러한 한계를 극복하는 데 기여하고 있음을 입증하였습니다. 우리의 데이터 집합은 맥락적으로 일관되고 사실적인 대화를 생성하는 데 향상된 도움을 주며, Character Consistency 및 Culture & Historical Appropriateness 차원에서 가장 큰 개선을 보였습니다.



### From Generalist to Specialist: A Survey of Large Language Models for Chemistry (https://arxiv.org/abs/2412.19994)
Comments:
          COLING2025,We maintain an up-to-date Github repository at: this https URL

- **What's New**: 이 논문에서는 화학(Langage Model for Chemistry) 분야에서 대형 언어 모델(LLM)의 사용을 개선하기 위한 방법론을 소개하고 있습니다. 특히, 화학 도메인 특별한 지식 및 다중 모달(multi-modal) 정보를 LLM에 통합하는 방법을 설명합니다. 기존 연구에서 다뤄지지 않았던 화학 중심 LLM에 대한 체계적인 조사를 제공하여, 연구자들이 화학 LLM의 발전을 따라갈 수 있도록 돕고자 합니다.

- **Technical Details**: LLM의 일반적인 문제점으로는 도메인 지식 부족, 다중 모달 데이터의 비인식 등이 있습니다. 많은 LLM들이 인터넷에서 수집된 웹 데이터로 사전 학습되기 때문에 화학 관련 데이터는 부족합니다. 이로 인해 강화학습(RLHF) 및 특정 작업에 대한 세부 조정이 불충분하여 화학에 대한 응용력이 떨어집니다.

- **Performance Highlights**: 논문에서는 다양한 화학적 데이터 다루기 위한 새로운 모델을 제안하고 기존 모델들이 화학 관련 작업에서 성능이 미흡하다는 것을 지적합니다. 또한 여러 화학 분석 도구와 기법을 통해 연구를 가속화할 수 있는 잠재력을 탐구하며, 향후 연구를 위한 기대되는 방향성을 제시합니다. 이를 통해 화학 분야의 혁신적인 응용 사례를 발굴하고 연구자들에게 통찰을 제공합니다.



New uploads on arXiv(cs.IR)

### Unsupervised dense retrieval with conterfactual contrastive learning (https://arxiv.org/abs/2412.20756)
- **What's New**: 이 논문에서는 Dense Retrieval Models의 강인성을 높이기 위해 세부적인 관련성을 잡아내는 민감도를 향상시키는 방법을 제안합니다. 이 모델은 문서의 주요 구절이 수정될 때 높은 분산을 보이면서도 불필요한 구절이 수정될 때는 낮은 분산을 유지해야 합니다. 이는 모델이 공격에 대해 강인하게 만들고, 논문과 쿼리 간의 관련성을 분석할 수 있게 도와줍니다.

- **Technical Details**: 제안된 방법은 Shapley value에 기반하여 문서의 주요 구절, 즉 counterfactual passage를 추출하는 방식을 포함합니다. 이후, 이 counterfactual passages에 기반하여 비지도 학습 과제를 도입하여 Dense Retrieval Models의 학습 과정을 개선합니다. 실험 결과, 이 방법은 패시지 수준의 관련성 주석 없이도 주요 구절을 추출할 수 있으며, 적대적 공격에 대한 강인성도 향상되었습니다.

- **Performance Highlights**: 이 논문에서 제안한 정규화된 Dense Retrieval Models는 최신 적대적 공격 방어 방법들보다 더 나은 강인성을 보여줍니다. 이 모델은 쿼리 관련성과의 관계에서 중요한 구절을 학습함으로써, 전반적인 성능 향상과 함께 다양한 IR 작업에서의 효과성을 입증했습니다. 특히 이 연구는 Dense Retrieval 분야에서의 저항성과 해석 가능성을 높이는 데 기여할 것으로 기대됩니다.



### AmalREC: A Dataset for Relation Extraction and Classification Leveraging Amalgamation of Large Language Models (https://arxiv.org/abs/2412.20427)
Comments:
          18 pages, 5 Figures

- **What's New**: 본 연구는 기존의 관계 분류 및 추출 데이터셋의 한계를 극복하기 위한 새로운 데이터셋을 제안합니다. 이 데이터셋은 255개의 관계 유형을 포함하고 있으며, 15,000개의 문장으로 이루어진 테스트 세트와 약 150,000개의 문장으로 구성된 훈련 세트를 가지고 있습니다. 이를 통해 관계의 다양성과 복잡성이 크게 향상되었습니다.

- **Technical Details**: 이 연구는 관계 튜플로부터 문장을 생성하기 위해 5단계의 다면적 파이프라인을 활용하며, LLM(대형 언어 모델) 및 템플릿 기반 생성 방법을 결합합니다. 문장 평가 지수(SEI)를 도입하여 문장의 문법적 정확성, 유창성, 인간 공감, 정확성 및 복잡성을 평가하고, SEI를 활용하여 최적의 문장을 선택하는 SEI-Ranker 모듈을 포함합니다.

- **Performance Highlights**: 제안된 데이터셋은 최첨단 모델(TANL, SPERT 등)을 활용하여 관계 추출 및 분류 작업을 평가하며, LLM 기반 분류 기술의 성과를 비교합니다. 이 연구는 데이터셋이 품질있는 문장 생성을 위한 최적화된 방법론임을 입증하며, 다양한 LLM들이 사용되었습니다.



### Introducing Semantic Capability in LinkedIn's Content Search Engin (https://arxiv.org/abs/2412.20366)
- **What's New**: 이번 논문에서는 LinkedIn의 새로운 콘텐츠 검색 엔진의 설계를 소개하고 있습니다. 이는 기존의 단순한 키워드 기반 검색에서 벗어나 자연어 쿼리를 처리할 수 있는 의미적(semantic) 기능을 갖추고 있습니다. 특히 사용자가 점차 길고 복잡한 쿼리를 사용하고 있는 기존 추세를 반영하여, 이 새로운 검색 엔진은 보다 높은 품질의 검색 결과를 제공하는 데 초점을 맞추었습니다.

- **Technical Details**: LinkedIn의 콘텐츠 검색 엔진은 두 개의 주요 레이어로 구성되어 있습니다: 검색(retrieval) 레이어와 다단계 순위(ranking) 레이어입니다. 검색 레이어에는 키워드 기반 선택을 위한 TBR(Token Based Retriever)와 AI 모델을 사용하는 EBR(Embedding Based Retriever)이 포함되어 있습니다. EBR은 다국어 텍스트 임베딩 모델인 multilingual-e5를 사용해 쿼리와 게시물 간의 의미론적 일치를 가능하게 하며, 포스트 임베딩을 사전 계산하여 실시간 처리 효율성을 극대화합니다.

- **Performance Highlights**: 새로운 검색 엔진의 목표는 사용자 쿼리에 대해 양질의 게시물들을 반환하는 것입니다. 이를 위해 두 가지 측정 기준인 온토픽 비율(on-topic rate)과 긴 체류(Long-dwells)를 활용합니다. 이 기준들은 사용자의 참여도와 관련된 중요한 지표로, 검색 엔진은 이러한 목표를 극대화할 수 있도록 설계되었습니다. 다단계 순위 레이어는 실시간으로 후보 게시물을 점수화하여 최종 검색 결과를 제공하며, 사용자와 게시물 간의 상호작용을 최적화할 수 있도록 돕습니다.



### Topic-Aware Knowledge Graph with Large Language Models for Interoperability in Recommender Systems (https://arxiv.org/abs/2412.20163)
Comments:
          Accepted by The 40th ACM/SIGAPP Symposium On Applied Computing(SAC) 2025

- **What's New**: 이 논문에서는 지식 그래프를 활용한 추천 시스템에서 LLMs(대형 언어 모델)를 통해 일반 및 특정 주제를 추출하는 새로운 접근 방식이 제안됩니다. 이 방법은 사용자와 아이템 간의 데이터 희소성 및 콜드 스타트 문제를 해결하는데 도움을 주며, 다양한 추천 시스템 간의 일관성을 향상시킵니다. 특히, LLMs를 활용하여 기존의 불일치 문제를 해결하고, 사용하는 맥락 정보에서 보다 구체적인 아이템 특성을 포착합니다.

- **Technical Details**: 논문에서는 첫째, 아이템의 사이드 정보에서 일반 주제를 반복적으로 추출하여 업데이트하고, 둘째, 맥락 정보를 활용하여 특정 주제를 추출합니다. 그 후, 특정 주제 추출 과정에서 발생하는 동의어 문제를 해결하기 위해 정제 알고리즘이 적용됩니다. 이 과정을 통해 일관성을 유지하며, 아이템의 세부 특성과 사용자 선호를 더욱 깊이 이해할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과는 다양한 지식 그래프를 통해 추천 성능이 크게 향상됨을 보여줍니다. 이 연구는 사이드 및 맥락 정보에서 추출한 일반 및 특정 주제를 기반으로 추천 품질을 개선했다는 점에서 의미가 있습니다. 또한, 제안된 방법은 표준화된 메타그래프를 바탕으로 하여, 다양한 도메인 간 일관되고 상호 운용 가능한 접근 방식을 보장하고 있습니다.



### A Contrastive Pretrain Model with Prompt Tuning for Multi-center Medication Recommendation (https://arxiv.org/abs/2412.20040)
Comments:
          accepted by TOIS

- **What's New**: 이 논문은 다중 센터(multi-center) 약물 추천 시스템의 새로운 가능성을 탐구합니다. 연구진은 기존의 연구들이 주로 단일 병원에 집중하고 있음을 지적하며, 여러 병원에서 수집된 데이터를 활용하는 실용적인 모델을 제안합니다. 특히, 적은 기록을 가진 작은 병원에서도 유용한 정보를 추출할 수 있도록 하는 방법에 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서 제안하는 TEMPT 모델은 두 단계의 사전 학습(pretraining)과 프롬프트 튜닝(prompt tuning)으로 구성되어 있습니다. 사전 학습 단계에서는 마스크 예측(mask prediction)과 대조(constrastive) 과제를 통해 일반적인 의학 지식을 습득하며, 특정 병원의 정보를 캡처하기 위한 프롬프트 튜닝 기법을 개발하였습니다. 이를 통해 모델은 데이터의 이질성을 더 잘 학습하고, 재학습 중 발생할 수 있는 재난적 망각(catasrophic forgetting) 문제를 완화할 수 있습니다.

- **Performance Highlights**: eICU 데이터셋을 통해 수행한 실험 결과, 제안된 모델은 기존의 약물 추천 시스템들에 비해 뛰어난 효과를 보였습니다. 다수의 병원 데이터를 통합한 모델은 작은 병원에서도 적절한 약물 세트를 추천하는 데 있어 큰 개선을 이루었습니다. 이러한 연구 결과는 다중 센터 설정에서의 약물 추천 시스템 개발에 기여할 것으로 기대됩니다.



### Invariant debiasing learning for recommendation via biased imputation (https://arxiv.org/abs/2412.20036)
- **What's New**: 이번 연구에서는 유저의 변동 정보를 통합하여 편향 없는 추천을 향상시키는 경량화된 지식 증류 프레임워크(KD-Debias)를 제안합니다. 기존의 invariant learning 방법에서는 정보 손실이 발생하여 모델 성능이 저하되는 문제를 파악하였습니다. 여기에 대한 해결책으로, 변동 정보를 사용하여 invariant 정보를 보완하는 방식을 채택하여 추천의 정확성을 높였습니다.

- **Technical Details**: KD-Debias는 거리 인식 지식 증류 과정에서 변동 정보를 불어넣어 invariant 사용자 선호도를 개선합니다. 이 과정은 변동 정보의 분포가 invariant 정보의 분포와 얼마나 다른지를 고려하여 주의를 기울입니다. 또한, matrix factorization (MF) 기반의 student 모델은 파라미터 수를 50% 이하로 줄이며 효율성을 높이고 일반화 능력을 증진시킵니다.

- **Performance Highlights**: 실험 결과, Yahoo!R3, Coat, MIND의 세 가지 공개 데이터 세트에서 변동 정보를 활용한 우리의 방법이 편향 제거 추천 모형 중에서 최고의 성능을 달성했습니다. KD-Debias는 최신 unsupervised debiasing 모델과 비교하여 일반적인 추천 성능을 크게 개선하며, 모델의 학습 파라미터 수를 대폭 줄였습니다.



### Towards Identity-Aware Cross-Modal Retrieval: a Dataset and a Baselin (https://arxiv.org/abs/2412.21009)
Comments:
          Accepted as full paper at ECIR 2025

- **What's New**: 최근 딥러닝의 발전은 CLIP과 같은 모델을 통해 콘텐츠 기반 검색 방식의 개선에 크게 기여했습니다. 하지만 이러한 접근법은 특정 개인을 식별하는 데 어려움을 겪고 있으며, 특히 훈련 데이터에 없는 도메인 특정 엔터티와 긴 꼬리 개념을 처리하는 데에 한계가 있습니다. 본 논문에서는 자연어 쿼리에 따라 특정 맥락에서 사람의 이미지를 검색하는 '정체성 인식 교차 모드 검색(identity-aware cross-modal retrieval)' 작업을 탐구합니다.

- **Technical Details**: 우리는 COCO 데이터셋에서 파생된 새로운 데이터셋인 COCO-PFS를 소개합니다. 이 데이터셋은 VGGFace2의 딥페이크 생성 얼굴로 보강된 이미지로 구성되어 있으며, 모델 훈련 및 평가에 필요한 대규모 데이터셋 부족 문제를 해결하고자 합니다. 다양한 CLIP 변형 모델을 실험하여 Identity-aware CLIP(Id-CLIP) 아키텍처를 제안하며, 이는 특화된 미세 조정을 통해 경쟁력 있는 검색 성능을 달성합니다.

- **Performance Highlights**: 실험 결과, Id-CLIP은 전통적인 CLIP 모델보다 정체성을 인식하는 데 더 뛰어난 성능을 보였습니다. 이를 통해 우리는 긴 꼬리 정체성과 맥락의 뉘앙스를 인식할 수 있는 견고한 교차 모드 검색 시스템의 개발 기초를 다졌습니다. 본 연구는 각기 다른 CLIP 모델을 비교하여 이 새로운 도전 과제에 대한 유의미한 통찰을 제공합니다.



### Rise of Generative Artificial Intelligence in Scienc (https://arxiv.org/abs/2412.20960)
Comments:
          26 pages, 4 tables, 1 figures, 1 appendix figure

- **What's New**: 최근 생성 인공지능(Generative Artificial Intelligence, GenAI)가 과학 연구에서 도구로 급속히 확산되고 있습니다. 본 연구는 OpenAlex를 활용한 실증 분석을 통해 2017년부터 2023년까지의 GenAI 및 기타 AI 학술 출판물의 성장을 조사합니다. 연구 결과는 GenAI가 컴퓨터 과학을 넘어 다양한 과학 분야에서 사용되고 있음을 보여줍니다.

- **Technical Details**: 분석은 GenAI 출판물의 성장 패턴, 연구 분야 내 확산, 그리고 생성 AI에 대한 과학 연구의 지리적 분포를 포함합니다. 연구팀의 규모와 국제 협력 또한 조사하여 GenAI가 다른 AI 기술과 비교할 때 다른 협력 양상을 보이는지를 검토하였습니다. 분석 결과 GenAI 연구팀은 다른 AI 분야에 비해 다소 작은 규모를 가지는 경향이 있습니다.

- **Performance Highlights**: 미국 연구자들은 전 세계 GenAI 출판물의 거의 40%를 차지하며, 중국이 그 뒤를 잇고 있습니다. 여러 중소형 선진 경제국들도 연구 출판물에서 비교적 높은 수준의 GenAI 활용을 보이고 있습니다. 또한 최근의 지정학적 긴장에도 불구하고 GenAI 연구는 다른 AI 기술과 유사한 수준의 국제 협력을 지속하고 있습니다.



### Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema (https://arxiv.org/abs/2412.20942)
Comments:
          Presented at HI-AI@KDD, Human-Interpretable AI Workshop at the KDD 2024, 26th of August 2024, Barcelona, Spain

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)을 활용한 온톨로지 기반의 지식 그래프(KG) 구축 접근법을 제안합니다. 이 방법은 Competency Questions(CQ)를 생성해 지식의 범위를 파악하고, 이러한 CQs에서 관계를 추출하여 위키데이터(Wikidata)의 대응 관계로 교체하는 과정을 포함합니다. 이는 KG의 일관성과 해석 가능성을 보장하고, 최소한의 인간 개입으로 고품질의 KG를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 접근법은 네 가지 주요 단계로 구성됩니다: 1) Competency Question 생성, 2) 관계 추출 및 온톨로지 매칭, 3) 온톨로지 형식 지정, 4) KG 구축입니다. 이 과정은 LLM을 통해 시작되며, 지식 도메인에 맞는 질문을 생성해 KG 구축 작업의 범위를 정의합니다. 이어서 CQ에서 추출된 속성을 위키데이터의 속성과 매칭하여 해당 속성을 정제합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 평가 결과, 제안된 방법은 전통적인 KG 구축 방식에 비해 높은 품질의 KG를 생성하는 것이 입증되었습니다. 이 방법은 해석 가능성과 유용성을 보장하며, 위키데이터와의 상호 운용성을 통해 기존 지식 기반과의 통합을 용이하게 합니다. 따라서 이 연구는 KG 구축에 있어 확장 가능하고 효율적인 방법론을 제시합니다.



### Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection (https://arxiv.org/abs/2412.20414)
Comments:
          6 pages, 1 table, AICCONF '24: Cognitive Models and Artificial Intelligence Conference, Istanbul, Turkey

- **What's New**: 이번 논문은 다국어 텍스트에서 지리적 데이터(geospatial data)의 추출 및 분석을 위한 최첨단 자연어 처리(NLP) 방법론과 대규모 언어 모델(LLMs) 통합의 중요성을 다룹니다. 특히, 다양한 NLP 모델(SpaCy, XLM-RoBERTa, mLUKE, GeoLM)과 OpenAI의 GPT 3.5 및 GPT 4를 평가하여 다국어 지리 엔티티 탐지의 맥락에서 이들의 성능을 분석합니다.

- **Technical Details**: 논문에서는 영어, 러시아어, 아랍어의 Telegram 채널에서 수집한 데이터셋을 활용하여 모델의 성능을 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1 scores) 등의 지표를 통해 평가합니다. 각 모델의 장점과 도전 과제를 드러내어 다양한 언어 환경에서 정확한 지리 엔티티 식별(geo-entity identification)의 복잡성을 강조합니다.

- **Performance Highlights**: 실험 결과, 각 모델의 성능 차이를 명확히 하여 지리적 참조 식별에 대한 효과성을 확인하였습니다. 이를 통해 고급 NLP 도구의 개선 및 개발 방향이 제시되어 지리적 분석 및 글로벌 보안(global security) 적용 분야의 발전에 기여하고자 합니다.



### Left-handed representation in top 100 male professional tennis players: Multi-disciplinary perspectives (https://arxiv.org/abs/2412.20360)
Comments:
          The original work citation (in APA): Bačić, B., & Ghazala, A. (2016). Left-handed representation in top 100 male professional tennis players: Multi-disciplinary perspectives. Symposium conducted at the meeting of the First New Zealand Text Mining Workshop (TMNZ 2016) in conjunction with the 8th Asian Conference on Machine Learning (ACML 2016), Hamilton, New Zealand

- **What's New**: 이 연구는 왼손잡이 테니스 선수들이 일반 인구의 왼손잡이 비율보다 과대대표된다는 주장을 데이터 분석을 통해 뒷받침합니다. 과거 수십 년간의 데이터에 기반하여, 이는 부모와 코치들이 자녀가 왼손잡이로 테니스를 시작할지, 오른손잡이로 시작할지 결정하는 데 도움이 될 수 있습니다.

- **Technical Details**: 연구는 ATP 웹사이트에서 제공한 지난 1985년부터 2016년까지의 상위 100위 테니스 선수 데이터를 분석했습니다. 이 데이터 분석을 통해 남성 인구의 약 10%가 왼손잡이라는 일반적인 추정치에 반해, 엘리트 테니스 선수 중 왼손잡이 비율은 약 15%로 확인되었습니다.

- **Performance Highlights**: 이 결과는 왼손잡이 선수의 과대대표화를 보여주며, 이는 테니스 코칭, 전략적 게임 개념, 미디어 분석 및 테니스 장비 제조에 유익한 통찰력을 제공합니다. 이러한 분석은 또한 왼손잡이에 대한 사실과 통계의 이해를 높이는 데 기여할 것입니다.



### OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System (https://arxiv.org/abs/2412.20005)
Comments:
          Work in progress

- **What's New**: OneKE는 웹과 PDF 형식의 책에서 지식을 추출할 수 있는 도커화된 스키마 기반의 지식 추출 시스템으로 소개됩니다. 이 시스템은 다양한 도메인을 지원하며, 여러 개의 에이전트를 설계하여 각자의 역할을 수행하도록 만들어졌습니다. 또한, 구성 가능한 지식 기반이 오류 디버깅과 개선을 돕습니다.

- **Technical Details**: OneKE는 실제 데이터에서 지식을 추출하는 데 필요한 복잡성을 해결하기 위해 설계되었습니다. 스키마 에이전트는 다양한 데이터 유형에 대한 분석을 수행하고, 추출 에이전트는 곧추 사용 가능한 LLM을 이용해 지식을 추출하며, 반영 에이전트는 오류를 디버깅하는 역할을 담당합니다. 이를 통해 다양한 데이터 포맷 및 길이에 유연하게 대응할 수 있는 기능이 있습니다.

- **Performance Highlights**: OneKE는 CrossNER 및 NYT-11-HRL 데이터셋에서 평가되었으며, 두 개의 주요 작업인 NER과 RE 모두에서 성능 향상을 보였습니다. 특히, 추출 에이전트의 사례 검색 방법이 가장 큰 성과를 달성하고 있으며, 복잡한 RE 작업에서는 중간 추론 단계가 더 중요한 것으로 나타났습니다. 이 시스템은 사용자 정의 스키마와의 통합을 통해 더욱 강력한 지식 추출 성능을 발휘합니다.



### ERPA: Efficient RPA Model Integrating OCR and LLMs for Intelligent Document Processing (https://arxiv.org/abs/2412.19840)
Comments:
          6 pages , 2 figures, 1 algorithm

- **What's New**: 이 논문에서는 이민 업무 내 ID 데이터 추출을 개선하고 Optical Character Recognition (OCR) 작업을 최적화하기 위해 설계된 혁신적인 Robotic Process Automation (RPA) 모델인 ERPA를 소개합니다. 기존 RPA 솔루션은 대량의 문서를 처리할 때 성능 한계에 직면하는 경우가 많아 비효율성을 초래했습니다. ERPA는 대규모 언어 모델(Large Language Models, LLMs)을 통합하여 추출된 텍스트의 정확성과 명확성을 개선하고, 애매한 문자 및 복잡한 구조를 효과적으로 처리하여 이 문제를 해결합니다.

- **Technical Details**: ERPA는 고급 OCR 기술과 LLM을 통합한 향상된 RPA 프레임워크로, 문서 처리 워크플로우에서 텍스트 추출의 정확성과 적응성을 높입니다. 이 모델은 다양한 문서 형식에서 텍스트 콘텐츠를 추출하기 위해 최첨단 OCR 시스템을 활용하고, LLM을 적용하여 추출된 데이터를 정제하고 검증하여, 모호하거나 복잡한 구조를 다룰 때 더 나은 정확성을 보장합니다. 또한 ERPA는 다양한 문서 레이아웃에 동적으로 적응하여 ID, 여권, 비자 및 증명서와 같은 이민 문서를 실시간으로 원활하게 처리할 수 있습니다.

- **Performance Highlights**: ERPA는 기존의 RPA 플랫폼에 비해 처리 시간을 93% 감소시켜 ID 데이터 추출을 단 9.94초 안에 완료하는 성과를 보여줍니다. 이 시스템은 자동화된 고사양, 문서 집약적 작업 흐름을 혁신할 가능성을 지니고 있으며, 이민 서비스와 같은 고볼륨 환경에서의 처리 속도와 정확성을 동시에 개선하여 새로운 표준을 제시합니다.



### From Interests to Insights: An LLM Approach to Course Recommendations Using Natural Language Queries (https://arxiv.org/abs/2412.19312)
Comments:
          17 pages, 9 figures

- **What's New**: 이 연구에서는 미국 대학의 학생들에게 더 나은 강의 추천을 제공하기 위해 대규모 언어 모델(LLM) 기반의 혁신적인 강의 추천 시스템을 탐구합니다. 이 시스템은 사용자 질문에 기반하여 '이상적인' 강의 설명을 생성하고, 결과를 벡터로 변환해 실제 강의를 찾는 방식으로 작동합니다. 강의 선택 과정에서의 불평등 해소를 목표로 하며, 캠퍼스 내에서의 파일럿 시스템을 배포할 계획도 포함되어 있습니다.

- **Technical Details**: 이 시스템은 Retrieval Augmented Generation (RAG) 방법을 적용하여 강의 설명의 데이터 집합에서 관련 정보를 검색합니다. RAG는 정보 검색과 생성형 AI를 결합한 접근법으로, 정보를 식별하는 리트리버(retriever)와 정보를 처리하는 생성기(generator)로 구성됩니다. 이 시스템은 상당한 대화적 요소를 포함하여 사용자와의 상호작용을 강화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 연구는 기존의 추천 시스템들이 직면한 여러 제한 사항을 해결하고, 강의 추천의 품질과 공정성을 평가하는 초기 테스트 결과를 포함합니다. 또한, 다양한 예시 프롬프트에서의 반응을 분석하여 이 시스템이 제공할 수 있는 개선된 사용자 경험을 보여줍니다. 아틀라스 플랫폼을 통한 파일럿 서비스의 구현을 통해 실제 학생들에게 이점을 제공하고자 합니다.



### LLM-assisted Vector Similarity Search (https://arxiv.org/abs/2412.18819)
- **What's New**: 본 논문에서는 데이터 검색 요구가 점점 복잡해짐에 따라 전통적인 검색 방법이 nuance (뉘앙스)와 conceptual (개념적) 쿼리를 처리하는 데 한계를 보임을 설명합니다. 이를 해결하기 위해, vector similarity search (벡터 유사도 검색)와 Large Language Models (LLMs)을 결합한 하이브리드 접근 방식을 제안합니다. 이 두 단계의 솔루션은 첫 번째 단계에서 벡터 유사도 검색을 활용하여 잠재적인 일치를 선별하고, 두 번째 단계에서 LLM을 사용하여 결과의 맥락 인식 순위를 매깁니다.

- **Technical Details**: 제안된 접근 방식은 먼저 벡터 유사도 검색을 통해 기본 쿼리를 처리하고, 이후 LLM을 통해 복잡한 쿼리에서 발생할 수 있는 constraints (제약 조건), negations (부정) 및 conceptual requirements (개념적 요구)를 고려하여 결과를 재순위합니다. 실험 결과, 벡터 유사도 검색은 간단한 쿼리에 효과적이지만, LLM을 활용한 방법은 더욱 복잡한 쿼리에서도 뛰어난 성과를 나타냅니다.

- **Performance Highlights**: 자연어 이해 능력을 활용하여 복잡한 작업의 검색 결과 정확성을 향상시키면서도 효율성을 유지하는 것이 이 방법의 큰 장점입니다. 논문에서는 실제 응용 사례를 논의하며, 다양한 데이터셋과 사용 사례를 위한 이 기술을 개선하고 확장하기 위한 향후 연구 방향도 제안합니다.



New uploads on arXiv(cs.CV)

### PERSE: Personalized 3D Generative Avatars from A Single Portra (https://arxiv.org/abs/2412.21206)
Comments:
          Project Page: this https URL

- **What's New**: PERSE는 참조 초상화(reference portrait)로부터 애니메이션 제작이 가능한 개인화된 생성 아바타를 구축하는 방법입니다. 이 모델은 사용자 고유의 정체성을 유지하면서 얼굴 속성(facial attribute)을 독립적으로 제어할 수 있는 연속적이고 분리된 잠재 공간(latent space)에서 얼굴 속성을 편집할 수 있게 합니다. 또한, PERSE는 대규모 합성 2D 비디오 데이터셋을 생성하여 관련 감정 변화와 특정 얼굴 속성의 변화를 포함한 비디오를 만듭니다.

- **Technical Details**: PERSE의 기법은 3D Gaussian Splatting을 기반으로 하여 연속적이고 분리된 잠재 공간에서 직관적인 얼굴 속성 조작을 학습합니다. 이 잠재 공간에서는 얼굴 속성을 부드럽게 전환시키기 위해 보강 기법인 잠재 공간 정규화(latent space regularization)를 도입하여 보조(supervision)로서 보간된 2D 얼굴을 사용합니다. 또한, 고품질의 포토리얼리스틱(photorealistic) 2D 비디오를 생성하기 위한 새로운 파이프라인(pipeline)을 제안합니다.

- **Performance Highlights**: PERSE는 이전 접근 방식에 비해 고품질의 아바타를 생성하며, 고유한 정체성을 유지하면서 보간된 속성을 출력을 가능하게 합니다. 이 시스템은 사용자가 원하는 얼굴 속성을 원하는 대로 조작할 수 있도록 지원하며, 매우 자연스러운 비디오 결과물을 제공합니다. 결과적으로, 이 방법은 개인 맞춤형 아바타 생성에서 혁신적인 발전을 이루었습니다.



### Action-Agnostic Point-Level Supervision for Temporal Action Detection (https://arxiv.org/abs/2412.21205)
Comments:
          AAAI-25. Technical appendices included. 15 pages, 3 figures, 11 tables

- **What's New**: 이번 논문에서 제안된 action-agnostic point-level (AAPL) supervision은 일부분만 주석이 달린 데이터셋으로 정확한 행동 인스턴스 탐지를 이루기 위해 개발되었습니다. 이 방법은 주석자가 비디오의 모든 행동 인스턴스를 찾지 않고도 샘플링한 비디오 프레임을 주석할 수 있도록 합니다. 또한, AAPL 레이블을 효과적으로 활용할 수 있는 탐지 모델과 학습 방법도 제안됩니다.

- **Technical Details**: AAPL supervision의 주석 파이프라인은 비행 동작을 무관하게 프레임을 샘플링하고, 그 후에 수동으로 주석을 다는 두 단계로 구성됩니다. 또한, AAPL 레이블은 단일 임의의 시간 지점만 지정하는 기존의 포인트 레벨 주석과 달리, 다수의 레이블을 사용하여 행동 인스턴스에 대한 보다 포괄적인 정보를 전달할 수 있습니다. 비록 AAPL supervision이 모든 행동 인스턴스를 탐지하지 못할 가능성이 있지만, 이는 전체적인 성능 향상을 위한 유효한 방법으로 고려됩니다.

- **Performance Highlights**: AAPL supervision을 사용하여 다양한 데이터셋(THUMOS '14, FineAction, GTEA, BEOID, ActivityNet 1.3)에서 실시된 실험 결과는, 제안된 접근 방식이 비디오 수준과 포인트 수준 주석 모두에서 이전 방법들과 경쟁하거나 이를 초월하는 성능을 보여주었습니다. 특히, 주석 비용과 탐지 성능 간의 균형을 시각적으로 비교했으며, 주석이 일부 달린 프레임으로만 훈련해도 이전 연구와 비슷한 경쟁력 있는 결과를 얻을 수 있음을 발견했습니다.



### A Large-Scale Study on Video Action Dataset Condensation (https://arxiv.org/abs/2412.21197)
- **What's New**: 본 연구는 비디오 데이터 세트 압축(video dataset condensation)이라는 분야를 탐구하며, 기존의 이미지 데이터 세트 압축에서 발전된 기법을 비디오 데이터에 적용합니다. 특히, 실험을 통해 샘플 다양성이 비디오 데이터 세트 압축에서 시간 다양성보다 더 중요하다는 흥미로운 사실을 발견하였으며, 단순한 슬라이드 창 샘플링(slide-window sampling) 방식이 효과적임을 확인했습니다.

- **Technical Details**: 연구에서는 데이터 세트 압축 알고리즘을 크게 샘플 선택(sample selection)과 데이터 세트 증류(dataset distillation)로 분류합니다. 샘플 선택은 원본 데이터 세트에서 샘플을 선택하여 압축된 데이터 세트를 생성하는 반면, 데이터 세트 증류는 프록시(Proxy) 목표를 최적화함으로써 데이터를 합성합니다. 이 연구는 비디오 데이터에 적합한 다양한 방법을 탐색하고, 평가 프로토콜을 수립하여 공정한 비교를 수행하였습니다.

- **Performance Highlights**: 세 개의 주요 액션 인식 데이터 세트(HMDB51, UCF101, Kinetics-400)에서 실험을 수행하여 모든 데이터 세트에서 최첨단(State-of-the-art) 성능을 달성했습니다. 연구 결과, 샘플 선택 방법이 대부분의 상황에서 데이터 세트 증류 방법보다 일반적으로 더 우수하게 작동함을 확인하였고, 데이터 세트 압축의 중요성과 그 잠재력을 강조했습니다.



### What Makes for a Good Stereoscopic Image? (https://arxiv.org/abs/2412.21127)
- **What's New**: 본 논문에서는 SCOPE(Stereoscopic COntent Preference Evaluation)라는 새로운 데이터 세트를 소개합니다. 이 데이터 세트는 실제 및 합성의 입체 이미지로 구성되어 있으며, 다양한 일반적 지각 왜곡(perceptual distortions)과 아티팩트(artifacts)를 포함하고 있습니다. 또한, 사용자 선호도에 대한 주석(annotation)을 VR 헤드셋에서 수집한 결과, 다양한 헤드셋에서 사용자 선호도가 일관됨을 보여줍니다. 그리고 이 데이터 세트를 기반으로 훈련된 새로운 모델인 iSQoE는 기존 방법들과 비교해 인간의 선호도와 더 잘 맞는 결과를 나타냅니다.

- **Technical Details**: 입체 품질 경험(SQoE)을 평가하기 위해, 우리는 SCOPE 데이터 세트를 활용하여 두 가지 왜곡(distortions)을 겪는 입체 이미지의 다양한 변형을 생성합니다. 이 과정에서 2400개의 샘플이 수집되었고, 103명의 주석자가 각 이미지 쌍에 대해 선호도를 평가하였습니다. iSQoE 모델은 이러한 데이터 세트로 훈련되어, 단순한 이미지 품질 평가 도구(IQA)를 넘어서는 복합적인 평가를 가능하게 합니다.

- **Performance Highlights**: iSQoE 모델은 단일 이미지 품질 평가 도구보다 입체 콘텐츠의 사용자 경험을 더 잘 반영하는 성능을 보여주었습니다. 이 모델은 다양한 입체 합성 방법을 평가하는 데 효과적으로 사용되며, 훈련 데이터에 포함되지 않은 왜곡 유형과 강도에 대해서도 일반화하는 능력을 입증하였습니다. 이 연구는 입체 이미지의 평가 방법론을 개선하기 위한 광범위한 노력을 보여주는 중요한 사례입니다.



### Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation (https://arxiv.org/abs/2412.21117)
- **What's New**: 이번 연구에서는 텍스트를 3D 객체 및 장면으로 변환하는 3D-aware latent diffusion 모델인 Prometheus를 소개합니다. 본 모델은 다중 뷰(Multi-view) 및 피드 포워드(Feed-forward)를 통해 3D Gaussian 생성 방식을 활용하며, 이는 2D 데이터에서의 일반화 가능성을 크게 향상시킵니다. 또한 RGB-D 잠재 공간(RGB-D latent space)을 도입하여 외관(Appearance)과 기하학(Geometry) 정보를 분리하여 보다 효율적인 3D 생성이 이루어질 수 있도록 합니다.

- **Technical Details**: Prometheus 모델은 두 가지 훈련 단계로 나뉘며, 첫 번째 단계에서는 3D Gaussian Variational Autoencoder (GS-VAE)를 통해 다중 뷰 또는 단일 뷰 RGB-D 이미지에서 압축된 잠재 공간을 학습합니다. 이 과정에서 단일 뷰 이미지를 카메라 포즈(Camera pose)와 텍스트 프롬프트(Text prompt)에 따라 조건화하여 다중 뷰 RGB-D 잠재 코드를 공동 예측합니다. Stable Diffusion 모델을 활용하여 적은 수정으로 본 연구의 모델을 구축하였으며, 다수의 다중 뷰 및 단일 뷰 데이터셋을 사용하여 훈련을 진행하였습니다.

- **Performance Highlights**: 실험 결과, Prometheus 모델은 3D Gaussian 재구성과 텍스트를 기반으로 한 3D 생성을 모두 훌륭하게 수행함을 입증하였습니다. 특히, 본 모델은 다양한 3D 객체 및 장면에 대해 우수한 일반화 능력을 보여주며, 수초 내에 3D 장면을 생성할 수 있는 효율성을 발휘합니다. 이러한 성능은 현재의 2D 이미지 생성 모델의 한계를 극복하는데 기여할 것으로 기대됩니다.



### Vinci: A Real-time Embodied Smart Assistant based on Egocentric Vision-Language Mod (https://arxiv.org/abs/2412.21080)
- **What's New**: Vinci는 실시간으로 작동하는 인공지능 기술로, 사용자의 관점에서 환경을 지속적으로 관찰하며 자연어 대화로 상호작용할 수 있는 스마트 어시스턴트입니다. 이는 스마트폰 및 웨어러블 카메라와 같은 휴대용 기기에서 동작하도록 설계되었습니다. 사용자들은 음성 명령으로 Vinci를 활성화하고 hands-free 방식으로 질문하거나 요청할 수 있습니다.

- **Technical Details**: Vinci는 다양한 모듈로 구성되어 있으며, 실시간으로 비디오 스트림을 처리하는 입력 모듈과 핵심 모델인 EgoVideo-VL을 통한 응답 생성 기능을 포함합니다. 또한, 사용자 활동 및 환경 사건의 종합적인 로그를 유지하는 메모리 모듈과, 상세한 시각적 안내를 제공하기 위해 비디오 생성 모듈을 통해 어떻게 할 것인지에 대한 단계별 데모를 생성합니다.

- **Performance Highlights**: Vinci는 실시간 성능과 사용자 간의 매끄러운 상호작용을 제공하며, 사용자의 질문에 대한 현재 및 과거 정보를 바탕으로 강력하고 맥락 기반의 응답을 생성합니다. 또한, 생성된 비디오 안내는 사용자에게 실질적인 작업 지침을 제시하여 보다 효율적인 사용자 경험을 제공합니다.



### Edicho: Consistent Image Editing in the Wild (https://arxiv.org/abs/2412.21079)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 다양한 이미지에서 일관된 편집 수행의 필요성을 강조하며, 기존의 훈련 기반 접근 방식 대신, 훈련이 필요 없는 확산 모델(diffusion models) 기반의 솔루션인 Edicho를 제안합니다. Edicho는 명시적인 이미지 상응(일치) 관계를 활용하여 편집 방향을 정하고, 주목 조작 모듈(attention manipulation module)과 분류기 없는 안내(classifier-free guidance, CFG) 비백색화 전략을 포함합니다. 이러한 방법은 다양한 편집 방법에서 적용 가능하며, 강력한 성능을 입증합니다.

- **Technical Details**: Edicho는 이미지 쌍 간의 명시적 상응 관계를 추출한 후, 훈련된 편집 모델을 활용하여 편집을 수행합니다. 이 과정에서 확산 모델의 자기 주의(self-attention) 메커니즘을 강화하여 소스 이미지에서 타겟 이미지로 특징을 효과적으로 전달합니다. CFG 계산을 수정하여 상응 관계를 반영함으로써 편집의 일관성을 향상시키고, 특성을 융합하여 편집 품질을 유지합니다. 이를 통해 다양한 조명 및 배경 조건에서도 안정적으로 작동할 수 있는 능력을 확보하고 있습니다.

- **Performance Highlights**: 제안된 Edicho 방법은 다양한 실험을 통해 양적 및 질적 평가에서 우수한 성능을 보였습니다. 특히 이미지 편집의 일관성을 유지하면서도 고화질 이미지를 생성하는 데 성공하였으며, 다양한 실제 상황에서의 응용 가능성을 보여줍니다. 이 방법은 개인화된 콘텐츠 제작, 신규 개념 학습을 위한 커스터마이즈된 모델 학습 등 여러 실제 응용에서 유용합니다. 코드가 공개될 예정으로, 향후 연구에 기여할 것으로 기대됩니다.



### Varformer: Adapting VAR's Generative Prior for Image Restoration (https://arxiv.org/abs/2412.21063)
- **What's New**: 본 논문은 VAR(Variable Autoregressive)라는 새로운 이미지 생성 패러다임을 소개하며, 이를 통해 이미지 복원(image restoration)에서의 효율성을 강조한다. VAR는 다음 단계 예측(next-scale prediction) 접근 방식을 통해 디퓨전(diffusion) 모델을 능가하는 생성을 조명하며, 높은 품질의 이미지 복원 기능을 제공한다. 또한 VarFormer 프레임워크를 통해 다양한 손상의 복원 작업에서 우수한 일반화 능력을 발휘할 수 있음을 보여준다.

- **Technical Details**: VAR는 전통적인 오토 리그레시브(autoregressive) 방법과 차별화된 새로운 시각적 오토 리그레시브 모델링 패러다임을 도입하고, "다음 토큰(prediction)" 대신 "다음 스케일(prediction)"을 예측한다. 이 과정은 구조적 열화(structural degradation) 및 수학적 불일치를 해결하여 고도로 구조화된 이미지를 생성하는 데 더 최적화되어 있다. VarFormer는 이러한 여러 스케일 분포 정렬(prior) 정보를 통합하여 다양한 복원 작업을 단일 모델 내에서 처리한다.

- **Performance Highlights**: VarFormer는 여섯 가지 이미지 복원 작업에서 놀라운 성과를 보여주었으며, 특히 저조도 이미지 향상(low-light enhancement), 노이즈 제거(denoising), 흐림 제거(deblurring) 및 저체적 해소(dehazing) 등의 작업에서 효과적이다. 실험 결과, VarFormer는 기존의 다중 작업 이미지 복원 방법들보다 월등한 성능을 발휘함으로써, 복원 분야의 새로운 가능성을 열어준다. 다양한 손상 유형에 대해  높은 품질의 기능 분포를 포착하며, 훈련 비용 또한 낮출 수 있는 접근 방식을 제안한다.



### VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation (https://arxiv.org/abs/2412.21059)
Comments:
          27 pages

- **What's New**: 이 연구에서는 이미지와 비디오 생성 모델을 인간의 선호도와 정렬하기 위한 새로운 전략인 VisionReward를 제안합니다. 이 모델은 인간의 선호도를 다차원적으로 분해하여 여러 판단 질문으로 구성된 보상을 생성합니다. 연구진은 VisionReward가 기존 비디오 평가 방법인 VideoScore보다 17.2% 더 나은 성능을 보이며, 이미지 및 비디오 스코어링 방법을 크게 뛰어넘는다고 주장합니다.

- **Technical Details**: VisionReward는 이미지와 비디오의 각 선호도 요인을 분해하여 3백만 개의 질문으로 구성된 데이터셋을 사용합니다. 이를 위해 연구자들은 다양한 판단 질문을 포함하는 체크리스트를 만들고, 이를 통해 인간의 선호도를 예측하는 다차원적 보상 모델을 구축합니다. 이 과정에서 시각-언어 생성 모델인 CogVLM2와 강화를 통해 머신러닝 사례를 다룹니다.

- **Performance Highlights**: VisionReward는 특히 비디오 평가에서 높은 정확성과 해석 가능성을 나타냅니다. 이 모델은 기존의 선호도 예측 방법을 능가하며, 기계 지표와 인간 평가 모두에서 우수한 성과를 보입니다. 또한, Multi-Objective Preference Optimization(MPO) 알고리즘을 통해 안정적인 최적화를 이루어냅니다.



### E2EDiff: Direct Mapping from Noise to Data for Enhanced Diffusion Models (https://arxiv.org/abs/2412.21044)
Comments:
          technical report, to be further updated

- **What's New**: 이번 논문에서는 확산 모델(diffusion models)이 직면하고 있는 여러 한계를 극복하기 위한 혁신적인 엔드 투 엔드 교육 프레임워크를 제안합니다. 기존 방법들이 소음 예측(noise prediction)에 집중하는 것과 달리, 제안하는 방법은 최종 재구성을 직접 최적화하여 훈련과 샘플링 프로세스를 일치시킵니다. 이를 통해 훈련-샘플링 간의 격차를 해소하고, 정보 유출(information leakage)을 줄이면서 인지 손실(perceptual losses)과 적대적 손실(adversarial losses)을 통합할 수 있게 됩니다.

- **Technical Details**: 제안하는 접근법은 훈련 프로세스를 순수한 가우시안 소음(pure Gaussian noise)에서 목표 데이터 분포(target data distribution)로의 직접 매핑(direct mapping)으로 간주함으로써 정보 유출 문제를 해결합니다. 확산 모델의 근본적인 구조적 한계를 극복하기 위해, 최종 재구성을 직접 최적화하는 방식을 채택하여 훈련 및 샘플링 과정 간의 모순을 해소합니다. 또한, 퍼셉션 품질(perceptual quality) 및 의미적 일관성(semantic consistency)을 강화하기 위한 다양한 손실 함수를 통합할 수 있게 됨을 강조합니다.

- **Performance Highlights**: COCO30K와 HW30K와 같은 벤치마크에서 진행한 광범위한 실험 결과, 제안된 방법이 전통적인 확산 모델에 비해 일관되게 우수한 성능을 보임을 확인했습니다. Fréchet Inception Distance (FID)와 CLIP score에서 기존의 최첨단 성능을 초과하였으며, 샘플링 단계가 줄어들어도 강력한 성과를 보였습니다. 이러한 결과는 엔드 투 엔드 훈련이 확산 기반 생성 모델을 더욱 견고하고 효율적인 솔루션으로 발전시킬 잠재력을 지니고 있음을 강조합니다.



### Visual Style Prompt Learning Using Diffusion Models for Blind Face Restoration (https://arxiv.org/abs/2412.21042)
Comments:
          Published at Pattern Recognition; 13 pages, 11 figures

- **What's New**: 이 논문은 시각 스타일 프롬프트 학습 프레임워크를 소개하여, 사전 훈련된 생성 모델의 잠재 공간에서 시각 프롬프트를 생성하여 바이든 얼굴 복원 과정을 안내합니다. 기존의 방법이 세부 사항을 재현하는 데 한계를 보인 반면, 이 방법은 향상된 성능을 제공합니다. 스타일 조정 집계 변환 계층을 추가하여 시각 프롬프트의 활용을 극대화하고, 정보가 풍부한 패턴 추출을 촉진합니다.

- **Technical Details**: 본 연구는 Diffusion Probabilistic Models (DMs)를 활용하여 저하된 이미지로부터 깨끗한 잠재 표현을 추정하는 방법을 탐구하며, 얼굴 특징 추출을 개선합니다. 또한, 시각 프롬프트를 기반으로 다중 스케일 convolutional kernel을 조정하여 기능 추출을 최적화하는 방법도 다룹니다. 스타일GAN의 잠재 표현을 이용하여 다양한 얼굴 속성을 생성하고 이후 복원 과정을 지원합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 제안한 방법이 네 가지 공개 데이터 세트에서 상태-of-the-art (SOTA) 방법들과 비교하여 높은 품질의 바이든 얼굴 복원을 달성함을 입증하였습니다. 또한, 얼굴 랜드마크 감지 및 감정 인식 같은 관련 작업에 대한 응용으로도 그 효과성을 보여주었습니다. 본 방법은 복원 품질 향상뿐만 아니라 다양한 얼굴 관련 작업에서도 유용함을 강조했습니다.



### Towards Identity-Aware Cross-Modal Retrieval: a Dataset and a Baselin (https://arxiv.org/abs/2412.21009)
Comments:
          Accepted as full paper at ECIR 2025

- **What's New**: 최근 딥러닝의 발전은 CLIP과 같은 모델을 통해 콘텐츠 기반 검색 방식의 개선에 크게 기여했습니다. 하지만 이러한 접근법은 특정 개인을 식별하는 데 어려움을 겪고 있으며, 특히 훈련 데이터에 없는 도메인 특정 엔터티와 긴 꼬리 개념을 처리하는 데에 한계가 있습니다. 본 논문에서는 자연어 쿼리에 따라 특정 맥락에서 사람의 이미지를 검색하는 '정체성 인식 교차 모드 검색(identity-aware cross-modal retrieval)' 작업을 탐구합니다.

- **Technical Details**: 우리는 COCO 데이터셋에서 파생된 새로운 데이터셋인 COCO-PFS를 소개합니다. 이 데이터셋은 VGGFace2의 딥페이크 생성 얼굴로 보강된 이미지로 구성되어 있으며, 모델 훈련 및 평가에 필요한 대규모 데이터셋 부족 문제를 해결하고자 합니다. 다양한 CLIP 변형 모델을 실험하여 Identity-aware CLIP(Id-CLIP) 아키텍처를 제안하며, 이는 특화된 미세 조정을 통해 경쟁력 있는 검색 성능을 달성합니다.

- **Performance Highlights**: 실험 결과, Id-CLIP은 전통적인 CLIP 모델보다 정체성을 인식하는 데 더 뛰어난 성능을 보였습니다. 이를 통해 우리는 긴 꼬리 정체성과 맥락의 뉘앙스를 인식할 수 있는 견고한 교차 모드 검색 시스템의 개발 기초를 다졌습니다. 본 연구는 각기 다른 CLIP 모델을 비교하여 이 새로운 도전 과제에 대한 유의미한 통찰을 제공합니다.



### FPGA-based Acceleration of Neural Network for Image Classification using Vitis AI (https://arxiv.org/abs/2412.20974)
- **What's New**: 최근 몇 년 동안, Convolutional Neural Networks (CNNs)는 컴퓨터 비전 분야에서 널리 사용되고 있습니다. 본 연구는 Xilinx Zynq UltraScale+ MPSoC ZCU104 FPGA 평가 보드를 활용하여 CIFAR-10 데이터셋에 대한 이미지 분류를 가속화하는 방법을 제시합니다. 이러한 전용 하드웨어의 사용은 CNN의 성능을 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문에서는 Vitis-AI를 사용하여 CNN의 이미지를 처리하는 방법을 설명합니다. 연구 결과는 CPU와 GPU 기준과 비교했을 때 3.33-5.82배 더 높은 처리량(throughput)과 3.39-6.30배 더 높은 에너지 효율성을 달성했습니다. 이러한 성능 향상은 고급 하드웨어 가속을 통해 가능함을 보여줍니다.

- **Performance Highlights**: CNN의 가속화는 이미지 분류 외에도 깊이 추정(depth estimation) 및 3D 재구성(3D reconstruction) 등 다운스트림 작업에 필요한 2D 피처(feature) 추출 잠재력을 발견했습니다. 이 연구는 향후 컴퓨터 비전 기술의 성능을 더욱 개선할 수 있는 방법을 제안하고 있습니다.



### Hierarchical Banzhaf Interaction for General Video-Language Representation Learning (https://arxiv.org/abs/2412.20964)
Comments:
          Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

- **What's New**: 최근 멀티모달(Multimodal) 표현 학습이 인공지능 분야에서 중요한 역할을 하고 있으며, 특히 비디오-언어 표상 학습(Video-language representation learning)의 필요성이 강조되고 있습니다. 본 연구는 비디오 텍스트 쌍의 세부적인 상호작용을 통해 더욱 세밀한 멀티모달 학습을 위한 새로운 접근법을 소개합니다. 영상과 텍스트를 게임 플레이어로 모델링하여 불확실성을 처리하고, 이를 위해 다변량 협력 게임 이론(Multivariate Cooperative Game Theory)을 적용한 점이 특징입니다.

- **Technical Details**: 우리는 Hierarchical Banzhaf Interaction(HBI)를 설계하여 비디오 프레임과 텍스트 단어 간의 세밀한 대응 관계를 계층적 관점에서 모사합니다. HBI는 협력 게임 내에서 동맹의 협력 정도를 측정하기 위해 Banzhaf Interaction 지수를 활용하며, 비디오 클립과 텍스트 단어 간의 상호작용을 더욱 정밀하게 수행할 수 있도록 합니다. 또한, 단일 모드(Single-modal)와 교차 모드(Cross-modal) 구성 요소의 융합을 통해 표현을 재구성하여 양쪽의 장점을 살려 더 세밀한 정합을 이루도록 합니다.

- **Performance Highlights**: HBI V2 프레임워크는 텍스트-비디오 검색, 비디오 질문 응답(VideoQA), 비디오 자막 생성 등 다양한 다운스트림 작업에서 획기적인 성과를 기록했습니다. 이 방법은 기존 HBI와 다른 최신 알고리즘보다 모든 다운스트림 작업에서 뛰어난 성과를 보였습니다. 또한, 본 연구는 멀티모달 표현 학습에 게임 프로세스를 처음으로 도입하여 세밀한 비디오-언어 학습의 새로운 가능성을 제시하고 있습니다.



### Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering (https://arxiv.org/abs/2412.20927)
Comments:
          6 pages, 3 figures, under review

- **What's New**: 이번 논문에서는 기존의 멀티모달 대형 언어 모델(MLLM)들이 시각적 세부 사항을 정확하게 캡처하지 못하는 문제를 해결하기 위해 새로운 멀티모달 검색 증강 생성(RAG) 프레임워크를 제안합니다. 이 프레임워크는 구조화된 장면 그래프를 도입하여 객체 식별 및 공간 이해를 향상시키고, 특히 겹치거나 작은 객체가 있는 복잡한 장면에서도 시각적 설명의 정확성을 높입니다. 또한, VG-150 데이터셋 및 AUG 데이터셋에서의 실험을 통해 기존 MLLM들을 지속적으로 초월하는 성능을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 모듈로 구성됩니다: (1) 멀티모달 RAG 구성, (2) 의미론적 향상 프롬프트, (3) LLM 기반 VQA입니다. 멀티모달 RAG 구성 모듈은 입력 이미지에서 객체를 식별하고 속성(범주, 수량, 위치 등)을 정의하여 장면 그래프를 생성합니다. 먼저 Faster-RCNN을 사용하여 객체 제안 및 관련 특성을 생성하고, 각 객체의 속성을 평가하여 시각적 이해를 개선합니다.

- **Performance Highlights**: 제안된 접근 방식은 VQA 작업에서 더 나은 객체의 정확한 수량, 위치 및 관계를 인식하는 능력을 보여줍니다. 실험 결과, 우리의 모델은 VG-150과 AUG 데이터셋을 통해 기존 MLLM들과 비교했을 때, 특히 세밀한 객체 수량 및 관계를 요구하는 경우에서 뛰어난 정확성을 달성하였습니다. 따라서 이 프레임워크는 복잡한 시각적 상황에서도 더 우수한 결과를 제공합니다.



### HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization (https://arxiv.org/abs/2412.20924)
Comments:
          Accepted by IEEE Transactions on Medical Imaging

- **What's New**: 이번 논문에서는 'HisynSeg'라는 새로운 약한 지도 하의 의미 구분(segmentation) 프레임워크를 제안합니다. 이 프레임워크는 이미지 혼합 합성을 기반으로 하여 기존의 CAM 기반 접근법의 단점을 극복하고, 약한 감독 작업을 완전 감독 작업으로 변환합니다. 이를 통해, 픽셀 정확도가 크게 개선되었습니다.

- **Technical Details**: HisynSeg에서는 Mosaic 변환 및 Bézier 마스크 생성을 기반으로 하는 두 가지 이미지 혼합 합성 전략이 도입됩니다. 또한 생성된 이미지의 진정성을 보장하기 위한 이미지 필터링 모듈이 개발되었습니다. 마지막으로, 자가 감독(self-supervised) 일관성 정규화(consistency regularization)를 통해 실제 이미지를 모델 훈련에 통합하여 과적합(overfitting) 문제를 완화합니다.

- **Performance Highlights**: 세 가지 데이터셋에서 수행된 실험 결과, HisynSeg 프레임워크는 최신 기술(SOTA)과 비교하여 성능이 크게 향상되었습니다. 특히 약한 감독 학습 환경에서의 픽셀 수준 세그멘테이션 성능이 현저히 개선된 것으로 나타났습니다. 이러한 성과는 실제 병리학적 이미지를 통한 자동화된 진단의 잠재력을 더욱 높이는 데 기여합니다.



### Low-Light Image Enhancement via Generative Perceptual Priors (https://arxiv.org/abs/2412.20916)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 기존의 Low-Light Image Enhancement (LLIE) 접근 방식에 비해 혁신적인 Generative Perceptual Priors (GPP-LLIE)를 도입하여 저조도 이미지 개선의 새로운 프레임워크를 제안합니다. 이는 Vision-Language Models (VLMs)을 활용하여 다양한 시각적 특성을 평가하고, 이러한 평가를 통해 얻은 perceptual priors를 기반으로 LLIE를 지원합니다. 이를 통해 이전의 방법들이 놓쳤던 시각적 디테일을 효과적으로 복원할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 본 논문에서는 자가 생성된 perceptual priors를 LLIE 모델에 통합하기 위해 transformer 기반의 backbone을 사용하고, 새로운 layer normalization (GPP-LN) 및 attention mechanism (LPP-Attn)을 개발하였습니다. 이 모델은 전 세계적 및 지역적 perceptual priors를 적용하여 개선 프로세스를 안내하며, LLIE에서의 성능을 높입니다. 국내외 다양한 실험에서 이 방법이 우수한 성능을 보이며 SOTA(State Of The Art) 방법들을 초월하는 결과를 입증하였습니다.

- **Performance Highlights**: 우리 모델은 LL 데이터셋과 현실 세계의 데이터에서 뛰어난 일반화 성능을 보이며, 다양한 테스트 환경에서도 아름답고 세부적인 개선 결과를 제공합니다. GPP-LLIE는 다른 LLIE 모델에도 적용 가능하며, 이를 통해 더욱 향상된 결과를 달성할 수 있습니다. 이러한 성능은 LLIE 분야에서의 새로운 가능성을 보여주며, 다양한 실제 상황에 적합한 이미지 개선 솔루션을 제공할 것으로 기대됩니다.



### TiGDistill-BEV: Multi-view BEV 3D Object Detection via Target Inner-Geometry Learning Distillation (https://arxiv.org/abs/2412.20911)
Comments:
          13 pages, 8 figures. arXiv admin note: substantial text overlap with arXiv:2212.13979

- **What's New**: TiGDistill-BEV는 LiDAR와 카메라 데이터를 효과적으로 통합하여 다중 뷰 3D 객체 검출의 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이 방법은 다양한 모달리티( modalities)에서 지식(knowledge)을 증류하여 카메라 기반 탐지기를 개선하며, 내부 기하학(internal geometry)을 통한 학습을 통해 더욱 정밀한 객체 인식을 가능하게 합니다. 특히, 내부 깊이 감독 모듈(inner-depth supervision module)과 내부 특징 BEV 증류 모듈(inner-feature BEV distillation module)을 도입하여 객체의 공간 구조를 더 잘 이해할 수 있도록 합니다.

- **Technical Details**: TiGDistill-BEV의 주요 구성 요소로는 두 가지 핵심 모듈이 포함됩니다. 첫째, 내부 깊이 감독 모듈은 객체 내의 저수준 상대 깊이 관계를 학습하여 탐지기가 객체의 공간 구조를 깊이 이해할 수 있도록 합니다. 둘째, 내부 특징 BEV 증류 모듈은 미리 훈련된 다중 모달 탐지기로부터 고수준 의미를 효과적으로 전이하는 역할을 합니다. 이러한 구조는 특정 포인트를 샘플링하여 카메라 기반 탐지기에 특징의 유사성을 학습하게 함으로써 모달리티 간의 거리(domain gap)를 줄입니다.

- **Performance Highlights**: nuScenes 기준에 대한 광범위한 실험을 통해 TiGDistill-BEV는 카메라 기반 탐지기에 보다 나은 성능을 부여하여 최첨단 62.8% NDS를 기록하며, 이전 방법들을 큰 차이로 초과하는 성과를 나타냈습니다. BEVDepth는 nuScenes val 세트에서 +2.3% NDS와 +2.4% mAP에서 성능 향상을 보여주며, 테스트 세트에서는 +3.9% NDS와 +4.8% mAP의 추가 향상이 있습니다. 이러한 결과는 TiGDistill-BEV의 효과성을 강조합니다.



### WalkVLM:Aid Visually Impaired People Walking by Vision Language Mod (https://arxiv.org/abs/2412.20903)
- **What's New**: 이 논문에서는 약 2억 명이 시각적 장애를 겪는 현상을 감안하여 인공지능 기술을 활용한 보행 보조 시스템의 필요성을 제기합니다. 기존 방법들이 자가 구축된 질문-답변 데이터셋을 기반으로 한 반면, 저자들은 이를 극복하고 보행 안내를 위한 통합된 훈련 및 테스트 기준을 제시하였습니다. 특히, 1만 2천 개 비디오-매뉴얼 주석 쌍이 포함된 Walking Awareness Dataset (WAD)을 새롭게 개발하여 다양한 장애인을 위한 연구 기반을 마련하였습니다.

- **Technical Details**: WalkVLM 모델은 계층적 계획을 위해 'chain of thought' 방식을 활용하여 간결하고 유익한 알림을 생성하고, 시간 인지 적응 예측을 통해 알림의 시간적 중복성을 줄입니다. 이 방식은 실시간 비디오 스트리밍에서 정보를 제공하며, 시각 장애인들에게 보다 유리한 조건의 보행 지원을 가능하게 합니다. 이를 통해 VLM들이 갖는 응답 중복 및 낮은 추론 효율성 문제를 해결하고자 했습니다.

- **Performance Highlights**: 실험 결과, WalkVLM은 다른 VLM 모델들과 비교하여 더 간결한 알림을 생성하고 비디오 스트리밍 처리에서 뛰어난 시간 적응성을 발휘하는 것으로 나타났습니다. 이 연구는 시각 장애인을 위한 실질적인 보행 안내 응용처의 발전을 위한 중요한 기초를 마련합니다. 저자들은 새로운 기준을 설정하고 오후 VLM을 통해 보행 안내를 가능하게 한 첫 번째 연구로 자리매김하고자 합니다.



### ILDiff: Generate Transparent Animated Stickers by Implicit Layout Distillation (https://arxiv.org/abs/2412.20901)
- **What's New**: 본 논문에서는 ILDiff라는 새로운 방법을 제안하여 애니메이션 스티커의 투명 채널을 생성합니다. 이전의 비디오 매팅(video matting) 및 확산 기반(diffusion-based) 방법의 단점을 해결하며, 특히 세미 오픈 영역(semi-open area) 처리와 시계열 정보(temporal information) 고려의 부족 문제를 해결하고자 합니다. 여기에는 SAM(Segmentation-Aware Matting) 접근법을 통해 암시적인 레이아웃 정보(implicit layout information)를 추가하여 더 정교한 결과를 제공합니다.

- **Technical Details**: ILDiff는 임시 레이아웃 증류(implicit layout distillation) 방법을 이용하여 투명 애니메이션 채널을 생성합니다. 이 방법은 임시 모델링(temporal modeling) 구조를 갖추어 시계열 처리 능력을 부여함으로써 기존 확산 기반 방법들이 직면했던 지역 후레임 떨림(local flickering) 문제를 개선합니다. 또한, TASD(Transparent Animated Sticker Dataset)라는 0.32M 고품질 샘플로 구성된 데이터셋을 구축하여 관련 분야에 지원을 제공합니다.

- **Performance Highlights**: 실험 결과, ILDiff는 Matting Anything 및 Layer Diffusion과 같은 기존 방법들과 비교하여 더 섬세하고 부드러운 투명 채널을 생성하는 것으로 나타났습니다. 이 연구는 고품질 투명 애니메이션 스티커 생성의 가능성을 열어주며, 향후 다양한 애플리케이션에서 활용될 수 있습니다. 제공되는 코드와 데이터셋은 연구자들에게 추가적인 지원을 제공할 것입니다.



### DDIM sampling for Generative AIBIM, a faster intelligent structural design framework (https://arxiv.org/abs/2412.20899)
Comments:
          the 10th International Conference on Innovative Production and Construction (IPC 2024), Perth, Australia. this https URL

- **What's New**: 본 연구에서는 구조 설계 파이프라인인 Generative AIBIM의 문제를 해결하기 위해 새로운 방법론을 제시합니다. 기존의 physics-based conditional diffusion model (PCDM)은 설계 생성을 위해 1000회의 반복이 필요했으나, 본 논문은 이를 가속화하는 denoising diffusion implicit model (DDIM)을 도입하였습니다. 이를 통해, 원래의 PCDM을 기반으로 한 설계 생성 과정을 최대 100배 빨라지게 하는 기술을 개발하였습니다.

- **Technical Details**: DDIM 샘플링은 PCDM의 최적화 프로세스에 최적화된 새로운 기법으로, 기존 DDIM 수식을 수정하여 적용되었습니다. DDIM은 denoising diffusion probabilistic model (DDPM)에서 파생된 모델이며, PCDM의 고유한 과정에 맞춰 설계되었습니다. 이 연구에서는 DDIM이 기존 PCDM의 처리 속도를 현저히 개선하는 방안을 실험적으로 입증하였습니다.

- **Performance Highlights**: 실험 결과, DDIM 샘플링을 적용한 PCDM은 시각적 품질을 유지하면서도 원래 모델보다 100배 빠른 속도로 설계를 생성할 수 있음을 보여주었습니다. 이는 구조 설계의 지능적 생성 속도를 크게 향상시키며, 기계 학습 이론에 대한 깊은 이해가 없는 연구자들에게도 매우 유용한 도구가 될 것입니다. 본 연구는 DDIM의 내용을 재조직하여 실제 사용에 초점을 맞추었으며, 이는 많은 연구자들에게 기여할 것으로 기대됩니다.



### Towards Compatible Fine-tuning for Vision-Language Model Updates (https://arxiv.org/abs/2412.20895)
Comments:
          preprint

- **What's New**: 이번 논문에서는 효율적인 fine-tuning 방법이 Foundation Model의 최신 업데이트와의 호환성 문제를 간과하고 있다는 점을 지적합니다. 저자들은 Class-conditioned Context Optimization (ContCoOp)라는 새로운 접근 방식을 제안하여 이를 해결하고자 합니다. 이 방법은 쉽게 추가할 수 있는 learnable prompts와 class embeddings를 통합하여 업데이트된 모델에서도 효과적으로 작동하도록 만들어집니다.

- **Technical Details**: ContCoOp는 attention layer를 통해 learnable prompts에 클래스 정보를 통합하여 텍스트 인코더에 입력하기 전에 이러한 프롬프트가 동적으로 변경될 수 있도록 합니다. 이를 통해 모델 업데이트로 인해 발생하는 embedding 공간 변화에 적응할 수 있으며, 이는 여러 데이터셋에서 수행된 실험에서 발견되었습니다. 기존의 shallow-layer 방법과 deep-layer 방법의 비교 분석을 통해 shallow-layer 방법이 업데이트된 모델에서 더 나은 호환성을 보여준다는 것을 증명했습니다.

- **Performance Highlights**: 15개의 다양한 데이터셋에서 실시된 실험 결과, ContCoOp는 기존 방법들보다 월등한 호환성을 보여주었습니다. 이 접근 방식은 종합적인 out-of-distribution 일반화를 입증하였으며, 업데이트된 모델에서도 지속적인 효과를 유지할 수 있음을 나타냅니다. 향후 fine-tuning 방법의 설계에 중요한 단서를 제공할 수 있을 것으로 기대됩니다.



### LiDAR-Camera Fusion for Video Panoptic Segmentation without Video Training (https://arxiv.org/abs/2412.20881)
Comments:
          Accepted by 2024 International Conference on Intelligent Computing and its Emerging Applications

- **What's New**: 이 논문은 자율주행 차량에서 사용되는 panoptic segmentation(파놉틱 세그멘테이션)의 새로운 접근 방식을 제안합니다. LiDAR(라이더)와 이미지 데이터의 융합(fusion)을 통해 PS(파놉틱 세그멘테이션)와 VPS(비디오 파놉틱 세그멘테이션)의 성능을 향상시키는 기능 융합 모듈을 도입했습니다. 또한, 제안된 모델이 비디오 데이터에 대한 학습 없이도 고품질의 VPS 결과를 제공할 수 있음을 보여주고 있습니다.

- **Technical Details**: 이 연구는 3D 데이터가 카메라 기반 장면 인식(scene perception)에 미치는 영향을 조사하는 데 집중했습니다. LiDAR와 이미지 데이터를 결합하여 PS와 VPS의 성능을 높이는 방식으로 기능 융합 모듈을 설계했습니다. 이 모델은 두 가지 간단한 수정 사항을 활용하여, 비디오 데이터 없이도 더 높은 품질의 VPS를 구현할 수 있도록 하였습니다.

- **Performance Highlights**: 결과적으로, 이 방법은 이미지와 비디오 파놉틱 세그멘테이션의 평가 지표에서 각각 최대 5포인트의 상당한 개선을 보였습니다. 이를 통해 자율주행 차량의 환경 인식 능력을 향상시키는 데 기여할 것으로 기대됩니다. 또한, 융합 및 모델 수정으로 인한 성능 향상이 입증되었습니다.



### Attention Is All You Need For Mixture-of-Depths Routing (https://arxiv.org/abs/2412.20875)
Comments:
          22 pages, 19 figures

- **What's New**: 본 논문에서는 Mixture-of-Depths (MoD) 모델의 개선된 라우팅 메커니즘인 A-MoD를 소개합니다. A-MoD는 이전 레이어의 attention map을 활용하여 현재 레이어에서의 라우팅 결정을 내리며, 추가적인 trainable parameter 없이도 효율적인 훈련을 가능하게 합니다. 이를 통해 기존 MoD 모델의 성능을 높이고, ImageNet에서 2% 높은 정확도를 기록했습니다. 또한, A-MoD는 빠른 transfer learning을 지원합니다.

- **Technical Details**: A-MoD는 라우팅 메커니즘을 혁신적으로 변화시킵니다. 전통적인 MoD 모델에서는 추가적인 네트워크 레이어를 활용해야 했지만, A-MoD는 기존의 attention map을 이용하여 토큰의 중요도를 평가합니다. 이러한 접근 방식은 훈련 과정에서의 노이즈를 줄이고, 기존 pretrained transformer 모델에 쉽게 적용할 수 있습니다. 라우팅 점수가 기존의 라우터에 비해 더 나은 상관관계를 보입니다.

- **Performance Highlights**: A-MoD는 여러 모델 크기와 작업에서 표준 라우팅보다 일관되게 더 나은 성능을 나타냅니다. MoD를 비추어 본 경우, A-MoD는 전통적인 모델에 비해 FLOPs 및 성능 모두에서 유리함을 입증하였습니다. 또한, A-MoD를 사용할 경우 더 빠른 수렴을 보여주며, transfer learning에서의 성능 향상을 가져왔습니다.



### LINK: Adaptive Modality Interaction for Audio-Visual Video Parsing (https://arxiv.org/abs/2412.20872)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 연구에서는 다중 모달 학습에 있어 비동기적 시각 및 청각 모달리티 간의 상호 작용을 조정하는 새로운 방법인 Learning Interaction method for Non-aligned Knowledge (LINK)를 제안합니다. 기존 방법들이 모달리티 간의 불일치를 무시했던 한계를 극복하고, 비동기적 신호 간의 상관관계를 동적으로 조정하여 이벤트 예측 성능을 높입니다.

- **Technical Details**: LINK는 각 모달리티의 입력을 동적으로 조정하여 비동기적 모달리티 간의 기여도를 균형 있게 만듭니다. 이를 구현하기 위해 유사도 기반의 손실 함수를 사용하여 서로 다른 샘플에 가중치를 적용하고, 모달리티 특정 개선을 위해 temporal-spatial attention을 활용합니다. 또한, CLIP 및 CLAP에서 생성된 유니모달의 pseudo-labels를 제약으로 사용하여 예측 정확도를 향상시킵니다.

- **Performance Highlights**: 우리의 모델은 LLP 데이터셋에서 기존의 방법보다 뛰어난 성능을 기록하였으며, 특히 시각 및 청각 이벤트가 비동기적으로 나타나는 상황에서도 효과적으로 이벤트를 파악할 수 있음을 입증했습니다. 이러한 성과는 LINK의 세밀한 조정 및 모달리티 간 상호작용 최적화의 결과입니다.



### SoftPatch+: Fully Unsupervised Anomaly Classification and Segmentation (https://arxiv.org/abs/2412.20870)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2403.14233

- **What's New**: 본 논문은 노이즈가 포함된 데이터로 수행되는 완전 비지도 산업 이상 탐지(unsupervised anomaly detection)를 처음으로 다룹니다. 기존의 비지도 이상 탐지 알고리즘은 깨끗한 데이터 세트에서 잘 작동하지만, 실제 환경에서는 노이즈가 문제로 작용합니다. 이러한 한계를 극복하기 위한 메모리 기반 비지도 AD 방법론인 SoftPatch와 SoftPatch+를 제안합니다.

- **Technical Details**: SoftPatch와 SoftPatch+ 방법은 패치 레벨에서 데이터를 효과적으로 디노이즈(denoise)하는 기능을 가지고 있습니다. 노이즈 구분자(noise discriminators)를 사용하여 패치 레벨의 노이즈 제거를 위한 이상치 점수(outlier scores)를 생성하고, 이 점수를 메모리 뱅크(memory bank)에 저장하여 이상 탐지 경계를 부드럽게 합니다. 이러한 방식으로 정상 데이터의 모델링 능력을 유지하며, 코어셋(core set)에서의 과신(overconfidence) 문제를 완화할 수 있습니다.

- **Performance Highlights**: 다양한 노이즈 시나리오에서 실시한 포괄적인 실험 결과, SoftPatch와 SoftPatch+는 MVTecAD, ViSA, BTAD 벤치마크에서 기존의 최첨단 AD 방법보다 우수한 성능을 보였습니다. 특히 SoftPatch+는 10%에서 40%까지의 높은 노이즈 수준에서도 견고한 성능을 발휘하여 현실 세계의 산업 검사 시나리오에 특히 유용합니다. 논문에서 제안한 방법의 코드는 제공된 URL에서 확인할 수 있습니다.



### Dual-Space Augmented Intrinsic-LoRA for Wind Turbine Segmentation (https://arxiv.org/abs/2412.20838)
Comments:
          Authors Shubh Singhal and Raül Pérez-Gonzalo contributed equally to this work. Accepted to ICASSP 2025

- **What's New**: 이번 연구에서는 풍력 터빈 블레이드의 이미지 분할(segmentation) 정확도를 높이기 위해 Intrinsic LoRA 방법을 확장하고, 새로운 이중 공간 증강(dual-space augmentation) 전략을 제안합니다. 이 방법은 이미지 레벨과 잠재 공간(latent-space) 증강을 통합하여, 선형 보간(linear interpolation)과 노이즈 기반의 확률적 모델을 통해 강화된 이미지 분할 결과를 공헌합니다. 현재의 최첨단 기법들을 초월하는 성능을 보여주어 자가 점검 및 손상 감지 시스템의 효과를 향상시킵니다.

- **Technical Details**: 이 논문에서는 Intrinsic LoRA 방법을 사용하여 이미지 분할을 위한 세부적인 과정을 설명합니다. 초기 Stable Diffusion 모델을 적응하여 WTB 분할을 위한 인코더와 디코더 구조를 적용하였으며, 잠재 공간에서의 변형 벡터의 최적화를 통해 지도 학습을 활성화했습니다. 또한 이미지 및 잠재 공간에서의 이중 증강 접근 방식을 통해 이미지 분할 성능을 극대화하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 WTB 이미지 분할 기법에 비해 현저하게 높은 정확도를 기록했습니다. 특히, 이중 증강 기법을 통해 모델의 학습 안정성과 성능을 동시에 개선하였습니다. 이러한 성과는 풍력 에너지 분야에서의 자동화된 평가 시스템 구축에 기여할 것으로 기대됩니다.



### Inclusion 2024 Global Multimedia Deepfake Detection: Towards Multi-dimensional Facial Forgery Detection (https://arxiv.org/abs/2412.20833)
Comments:
          Inclusion 2024 Global Multimedia Deepfake Detection Competition Top Team Technical Report

- **What's New**: 이번 논문에서는 Inclusion 2024와 동시에 열린 Global Multimedia Deepfake Detection 대회를 소개하고 있습니다. 이 대회는 이미지 및 오디오-비디오 조작 탐지를 목표로 하며, 전 세계의 1500개 팀이 참여하여 약 5000개의 유효 결과를 제출했습니다. 논문에서는 2개 트랙에서 상위 3개 팀의 솔루션을 제시하여 이미지 및 오디오-비디오 위조 탐지 분야의 연구를 촉진하고자 합니다.

- **Technical Details**: 이 대회에서는 Multi-dimensional Facial Forgery (MultiFF) 데이터셋을 도입하여 이미지 및 오디오-비디오 위조 탐지 과제를 위한 대규모 벤치마크를 제공합니다. MultiFFI는 90만 개 이상의 이미지로 구성되어 있으며 80개 이상의 생성 알고리즘을 사용하여 생성됩니다. MultiFFV 데이터셋은 60만 개 이상의 비디오 소스를 포함하고 다양한 얼굴 공격 기술의 예제를 제공합니다.

- **Performance Highlights**: 대회 중 최종 성과 평가에서는 AUC (Area under the Curve)를 주 평가 지표로 사용하여 각각의 모델 성능을 측정합니다. 최종 20개 팀은 최종 단계인 Phase 3에 진출하게 되며, 각 팀의 순위는 공공 테스트 세트와 숨겨진 테스트 세트, 기술 보고서의 가중 점수를 바탕으로 결정됩니다. 이러한 성과들은 깊은 합성 기술의 빠른 발전에 맞춰 지속적으로 진행될 것입니다.



### ReFlow6D: Refraction-Guided Transparent Object 6D Pose Estimation via Intermediate Representation Learning (https://arxiv.org/abs/2412.20830)
- **What's New**: 이번 논문에서는 투명 객체의 6D 포즈 추정을 위한 새로운 방법인 ReFlow6D를 제안합니다. 이 방법은 RGB 이미지 입력만을 사용하여 투명 객체의 6D 포즈를 정확하게 추정할 수 있도록 설계되었습니다. ReFlow6D는 빛의 굴절을 이용하여 독특한 중간 표현을 생성하며, 이는 환경 변화에 독립적입니다.

- **Technical Details**: ReFlow6D는 투명 객체를 통해 전달되는 빛의 경로 변형을 모델링하여 새로운 refractive-intermediate representation을 사용합니다. 이러한 중간 표현은 물체의 기하학적 특징에 의존하지 않으며, RGB 이미지 공간에서의 변화에 강력한 특성을 가집니다. 이 방법은 학습된 Patch-PnP 기법을 통해 투명 객체의 6D 포즈를 직접 회귀하도록 설계되었습니다.

- **Performance Highlights**: 실험 평가 결과, ReFlow6D는 TOD 및 Trans32K-6D 데이터셋에서 기존의 최신 방법들보다 뛰어난 성능을 보여주었습니다. 로봇 잡기 실험을 통해 ReFlow6D의 포즈 추정 정확도가 실제 로봇 작업에 효과적으로 적용됨을 입증하였습니다.



### Length-Aware DETR for Robust Moment Retrieva (https://arxiv.org/abs/2412.20816)
- **What's New**: 본 논문은 비디오 순간 검색(Video Moment Retrieval, MR) 분야에서 단기 순간의 검색 정확성을 향상시키기 위한 두 가지 새로운 기술, MomentMix와 Length-Aware Decoder (LAD)를 제안합니다. 최근의 DETR 기반 모델들이 단기 순간에 대해 성능 저하 문제를 보이는 가운데, 이 연구는 이러한 문제를 해결하기 위한 방법론을 제공하여 주목을 받습니다. 데이터 분석을 통해 단기 순간의 특징 다양성이 제한적이라는 점을 발견하였고, 이를 해결하기 위한 출시된 혁신적 기술들이 설명됩니다.

- **Technical Details**: MomentMix는 ForegroundMix와 BackgroundMix라는 두 가지 데이터 증강 전략을 활용하여 비디오의 전경(가장 관련이 큰 정보)과 배경(관련이 적은 정보)의 특징 표현을 향상시키는 방법입니다. ForegroundMix는 다양한 순간의 전경 특징을 결합하여 새로운 전경을 만들며, BackgroundMix는 다른 비디오의 배경 조각을 사용하여 다양한 전경-배경 조합을 생성합니다. 또한 Length-Aware Decoder는 순간의 길이에 따른 예측 정확성을 개선하도록 모델을 구조화하여, 더 나은 센터(center) 및 길이(length) 예측을 가능하게 합니다.

- **Performance Highlights**: 제안된 접근법은 기존의 DETR 기반 방법들과 비교하여 QVHighlights, TACoS, Charades-STA와 같은 여러 데이터셋에서 성능이 크게 향상됨을 보여줍니다. 특히 QVHighlights에서는 mAP이 9.36% 증가하여 41.22에서 45.08로 상승하였고, R1@0.7에서도 유의미한 개선을 달성하였습니다. 이는 단기 순간의 검색 성능을 비약적으로 향상시킴으로써 MR 분야에서 중요한 한 걸음을 내딛었다고 할 수 있습니다.



### Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability (https://arxiv.org/abs/2412.20807)
Comments:
          9 pages, 6 figures, accepted by 2025ICASSP

- **What's New**: 이 논문에서는 목표 공격(targeted attack)의 전이 가능성(transferability)을 향상시키기 위한 새로운 방법론을 제안합니다. 기존의 방법론들이 최적화 경로(optimization trajectory)를 무시하고 끝점만을 이용하는 반면, 이 논문은 최적화 경로의 평균화(averaging)를 통해 조작된 적대적 예시(adversarial example)를 안정적으로 중앙 영역으로 이동시키는 방안을 제시합니다.

- **Technical Details**: 제안된 방법은 기존의 적대적 공격과 결합되어 다양한 공격 시나리오에서 비교됩니다. 기존의 fine-tuning 기법들이 손실 표면(loss surface)의 평탄한 영역 주변에서 진동하는 경향을 보이는 반면, 이 방법은 경로의 평균화를 통해 이를 극복하고자 합니다. 이는 목표 공격의 향상된 전이 가능성을 제공하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 기존의 fine-tuning 기법들과 비교할 때 목표 전이 가능성(targeted transferability)을 크게 향상시킵니다. 이 방법은 최신 적대적 공격 기법들과 통합되어 뛰어난 성능을 나타내며, 실제 코드도 제공됩니다.



### Frequency-aware Event Cloud Network (https://arxiv.org/abs/2412.20803)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 Event Cloud 표현을 활용하는 FECNet라는 새로운 네트워크를 제안합니다. FECNet은 Event Cloud의 장점을 최대화하며, 폴라리티 정보도 포괄적으로 다룹니다. 뿐만 아니라, 주파수(domain) 분석을 활용하여 긴 연속 이벤트로부터 공간-시간 특성을 효과적으로 캡처합니다.

- **Technical Details**: FECNet은 이벤트 기반의 그룹 및 샘플링 모듈(G & S)을 혁신적으로 재설계하여 폴라리티 정보를 포함한 지역 특징을 효과적으로 추출합니다. 주파수-인지 모듈(Frequency-aware modules)은 컨볼루션을 해다르 곱(Hadamard product)으로 대체하여 많은 MACs를 줄이고, 긴 시간 종속성을 포착하는 데 효율적입니다. 이를 통해 FECNet은 경량화된 네트워크 아키텍처로 설계되었습니다.

- **Performance Highlights**: FECNet은 객체 분류, 동작 인식 및 인간 자세 추정과 같은 다양한 작업에서 9개 데이터 세트에 대해 테스트하여 우수한 성능을 입증했습니다. 이 실험의 결과는 FECNet이 기존 방법에 비해 효율성과 효과성을 모두 갖추었다는 점을 보여줍니다. 또한, DVS128 Gesture 데이터셋에서의 성능 향상도 특별히 강조되었습니다.



### Generalize Your Face Forgery Detectors: An Insertable Adaptation Module Is All You Need (https://arxiv.org/abs/2412.20801)
Comments:
          ICASSP2025 accepted

- **What's New**: 이 연구에서는 기존의 얼굴 진위 판별기(face forgery detector)의 단점을 해결하기 위해, 학습 단계에서 보지 못한 위조(face forgery)에 일반화할 수 있는 새로운 적응 모듈(adaptation module)을 소개합니다. 이 적응 모듈은 온라인에서 라벨이 없는 테스트 데이터만을 이용하여 기존의 탐지기를 수정 없이 조정할 수 있습니다. 이를 통해 다양한 위조 단서와 도메인 간의 간극을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법에서는 기존의 탐지기가 생성하는 출력만을 처리하는 적응 모듈이 포함되어 있으며, 이 모듈은 메모리 뱅크(memory bank)를 유지하여 평가 중에 역사적 정보를 보존합니다. 특히, 학습 가능한 클래스 프로토타입 기반 분류기(class prototype-based classifier)를 통해 테스트 데이터에 대한 예측을 생성하며, 이를 바탕으로 최근접 특징 보정기(nearest feature calibrator)를 사용하여 예측 정확도를 더욱 향상시킵니다. 이 과정에서 모델의 주요 구조는 변경되지 않으며, 기존의 탐지기를 고정 상태로 유지합니다.

- **Performance Highlights**: 다양한 데이터셋을 통해 수행된 실험 결과, 제안된 적응 모듈은 기존의 최첨단 방법들에 비해 월등한 일반화를 달성하였습니다. 또한, 이 모듈은 다양한 탐지기와 결합하여 전체 성능을 향상시키는 플러그 앤 플레이(component) 방식으로 작동하여 실용적인 적용 가능성을 높입니다. 이에 따라 얼굴 위조 탐지의 연구 분야에 기여할 것으로 기대됩니다.



### VMix: Improving Text-to-Image Diffusion Model with Cross-Attention Mixing Contro (https://arxiv.org/abs/2412.20800)
Comments:
          Codes and models are available at this https URL

- **What's New**: 이 논문에서는 Cross-Attention Value Mixing Control (VMix) Adapter라는 새로운 미적 어댑터를 제안하여 텍스트-이미지 생성에서 높은 미적 품질을 달성하는 방법을 다룹니다. VMix는 입력 텍스트 프롬프트를 콘텐츠 설명과 미적 설명으로 분리하고, 미적 조건을 denoising 과정에 통합하여 응용할 수 있도록 설계되었습니다. 이를 통해 기존 diffusion 모델의 미적 표현력을 향상시키는 동시에 이미지-텍스트 정렬을 유지합니다. VMix는 재학습 없이도 다양한 커뮤니티 모델에 쉽게 적용 가능하다는 점이 핵심적인 장점입니다.

- **Technical Details**: VMix는 두 가지 전문 모듈을 포함하여 미적 레이블을 U-Net 아키텍처에 통합합니다. 첫 번째 모듈은 미적 임베딩 초기화 모듈로, 미적 텍스트 데이터를 전처리하여 이미지와 일치하는 임베딩으로 초기화합니다. 두 번째 모듈인 크로스-어텐션 믹싱 컨트롤 모듈은 주의 맵을 직접 변경하지 않고 이미지-텍스트 정렬에 대한 부정적인 영향을 최소화하는 것을 목표로 합니다. 이러한 체계적인 접근 방식을 통해 VMix는 다양한 미적 치수에서 생성된 이미지와 실제 이미지 간의 품질 격차를 효과적으로 해소합니다.

- **Performance Highlights**: VMix는 광범위한 실험을 통해 최신 기술들과 비교했을 때 뚜렷한 성능 향상을 보여주었습니다. 특히, ControlNet, IP-Adapter 및 LoRA와 같은 커뮤니티 모듈과의 뛰어난 호환성을 유지합니다. 이러한 결과는 VMix가 다양한 미적 조건을 충족하며, 생성된 이미지의 미적 품질을 크게 향상시킬 수 있음을 시사합니다. 결과적으로, VMix는 이미지 생성의 미적 성능을 높이는 데 기여하며 커뮤니티의 창의적인 가능성을 확장합니다.



### Sample Correlation for Fingerprinting Deep Face Recognition (https://arxiv.org/abs/2412.20768)
- **What's New**: 최근 얼굴 인식 분야에서는 딥 러닝 기술의 발전으로 인해 놀라운 혁신이 이루어졌습니다. 그러나 모델 훔치기 공격에 대한 우려가 커지고 있으며, 이는 모델 소유자의 지적 재산권에 심각한 위협이 되고 있습니다. 이러한 문제를 해결하기 위해, SAC(Sample Correlation)이라는 새로운 모델 훔치기 탐지 방법을 제안하였습니다.

- **Technical Details**: SAC 방법은 샘플 간의 쌍별 관계를 이용하여 모델 훔치기 공격을 탐지합니다. 특히 SAC-JC는 JPEG 압축된 샘플을 사용하여 상관 행렬을 계산하고, 이를 기반으로 훔쳐진 모델을 검증합니다. 다양한 공격 방식에서 SAC-JC의 유효성을 검증하였으며, 그림 인식 데이터셋 등 여러 실험을 통해 기존 방법보다 우수한 성능을 발휘함을 보여주었습니다.

- **Performance Highlights**: 실험 결과 SAC-JC는 얼굴 인증 및 감정 인식 과제에서 가장 높은 AUC, p-value 및 F1 점수를 기록하며 다양한 모델 훔치기 공격에 대해 성공적으로 방어함을 입증하였습니다. 또한, 객체 인식 데이터셋인 Tiny-ImageNet과 CIFAR10에서도 SAC-JC의 우수성을 확인하였습니다. 이 방법은 대체 모델 훈련 없이 계산적 부담을 크게 줄이며, 약 34393배 더 빠른 성능을 제공합니다.



### KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences (https://arxiv.org/abs/2412.20767)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 3D 모델을 재구성하는 새로운 프레임워크인 KeyGS를 제안합니다. 이 방법은 Structure-from-Motion(SfM) 기법을 사용하여 초기에 대략적인 카메라 포즈를 신속하게 획득한 후, 3D Gaussian Splatting(3DGS)을 활용하여 포즈를 정제합니다. KeyGS는 효율적인 카메라 포즈 추정과 함께 긴 훈련 시간을 단축시키는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: KeyGS는 기존 방법들보다 카메라 포즈의 정확도를 향상시키기 위해 조합된 정제 과정을 도입합니다. 일반적으로 SfM과 여러 외부 소프트웨어를 통한 카메라 포즈 추정이 필요하지만, KeyGS는 불필요한 매칭 모델 없이 몇 초 만에 카메라 포즈를 얻습니다. 또한, 이 방법은 고주파 신호에 의한 오류를 방지하도록 설계된 coarse-to-fine frequency-aware densification 기법을 통합하여 재구성 과정의 질을 높입니다.

- **Performance Highlights**: 이 접근법은 훈련 시간을 몇 시간에서 몇 분으로 단축시키면서, 더 높은 정확도의 새로운 뷰 합성을 달성하고 기존 방법들과 비교해 카메라 포즈 추정의 신뢰성을 크게 개선합니다. KeyGS의 구현을 통해 다양한 환경에서도 정확한 3D 재구성이 가능합니다. 이 연구의 결과는 3D 재구성을 위한 실시간 렌더링과 효율성을 극대화하는 데 기여할 것으로 보입니다.



### Unforgettable Lessons from Forgettable Images: Intra-Class Memorability Matters in Computer Vision Tasks (https://arxiv.org/abs/2412.20761)
- **What's New**: 이 논문은 인트라 클래스 기억력(intra-class memorability)을 도입하며, 이는 동일한 객체 클래스 내에서 특정 이미지가 더 기억에 남는 이유를 탐구합니다. 연구진은 연속 인식 태스크를 사용하여 특정 개체가 더 기억에 남는 이유를 설명하기 위해 인간 행동 실험을 설계하였습니다. 또한, 새로운 메트릭인 인트라 클래스 기억력 점수(ICMscore)를 통해 이미지 기억력을 정량화했습니다.

- **Technical Details**: 인트라 클래스 기억력 점수(ICMscore)는 이미지 반복 프레젠테이션 간의 시간 간격을 포함하여 각 개별 인스턴스의 기억력을 측정하는 첨단 메트릭입니다. 이 연구에서는 1,000명의 참가자로부터 수집된 2,500개의 이미지에 대한 응답을 기반으로 한 인트라 클래스 기억력 데이터셋(ICMD)을 제안하고, 이를 통해 인공지능 모델을 교육하고 테스트했습니다. 또한, 기억력이 높은 이미지가 AI의 이미지 인식 및 지속적 학습 작업에서 성능 저하를 유발하는 것을 발견했습니다.

- **Performance Highlights**: ICMD 데이터셋을 기반으로 한 기억력 예측 모델은 인트라 클래스 기억력의 대표적인 특성을 효과적으로 포착할 수 있음을 보여줍니다. 연구 결과, 기억력 높은 이미지(big memorability, HM)가 더 눈에 띄지만, 실제로는 이미지 인식과 지속적 학습 작업에서 AI의 성능을 저하시킬 수 있음을 발견했습니다. 이를 통해, 이미지를 기억력 기준으로 조절할 수 있는 혁신적인 이미지 확산 모델의 활용 가능성을 제시했습니다.



### Are Vision-Language Models Truly Understanding Multi-vision Sensor? (https://arxiv.org/abs/2412.20750)
Comments:
this https URL. arXiv admin note: text overlap with arXiv:2408.12114

- **What's New**: 최근 대규모 Vision-Language Models (VLMs)의 발전으로 시각적 입력과 텍스트 간의 정렬이 이루어졌고, 이는 컴퓨터 비전 작업에서 성능 향상을 가져왔습니다. 하지만 현재 VLM이 다양한 다중 비전 센서 데이터인 열 화상, 깊이 및 X-레이 정보를 효과적으로 이해하지 못한다는 한계가 있습니다. 이러한 문제를 해결하기 위해 Multi-vision Sensor Perception and Reasoning (MS-PR)이라는 새로운 벤치마크를 제안합니다.

- **Technical Details**: 이 논문에서는 VLM이 각 센서의 고유한 물리적 특성을 이해하지 못하고 다중 비전 센서 이미지를 처리하는 데의 한계를 보여줍니다. 저자들은 Multi-vision Sensor Perception and Reasoning (MS-PR) 테스트를 통해 VLM의 센서 기반 추론 능력을 평가하고, Diverse Negative Attributes (DNA) 최적화를 도입하여 다중 비전 센서 작업에서 깊은 추론이 가능하도록 합니다. 이를 통해 VLM은 이미지와 센서 데이터 간의 핵심 정보 격차를 해소하는 데 도움이 됨을 입증했습니다.

- **Performance Highlights**: 대규모 VLM을 평가한 결과, 대부분의 최신 VLM들이 센서 추론에서 다소 결점을 보이는 것으로 나타났습니다. 그러나 DNA 최적화를 적용한 VLM은 다중 비전 센서 추론 작업에서 성능이 크게 향상되었습니다. 이 작업은 최근 자율주행차, 보안 시스템, 의료 이미지 진단 등 다양한 응용 분야에서 센서 특이적 정확도가 얼마나 중요한지를 강조합니다.



### Solar Filaments Detection using Active Contours Without Edges (https://arxiv.org/abs/2412.20749)
Comments:
          6 pages, 2 figures

- **What's New**: 이 논문에서는 H-alpha 전체 디스크 태양 이미지에서 태양 필라멘트를 감지하기 위해 액티브 컨투어(Active Contours without Edges, ACWE) 기반 알고리즘이 제안되었습니다. 이 알고리즘은 이미지 전처리, 분할, 후처리의 세 가지 주요 단계로 구성됩니다. 새로운 접근 방식은 객체의 경계를 정확하게 감지하여 태양 이미지에서 최적의 성능을 발휘합니다.

- **Technical Details**: 제안된 알고리즘은 에너지 함수에 따라 변형할 수 있는 경계를 태양 이미지 위에 초기화하고, 원하는 객체의 경계에 도달하면 에너지 함수가 감소하며 경계가 더 이상 발전하지 않도록 설계되었습니다. 이 과정은 이미지 전처리, 분할, 후기 처리를 포함하여 각 단계를 체계적으로 수행합니다.

- **Performance Highlights**: 성능 분석 결과, 제안된 알고리즘은 기존의 클래식(Object Detection) 알고리즘보다 뛰어난 성능을 보였습니다. 이 연구에서 제안한 방법은 몇 가지 벤치마크 데이터셋에 적용되어 비교되었으며, 객체 감지에서의 우수성을 입증하였습니다.



### UniRS: Unifying Multi-temporal Remote Sensing Tasks through Vision Language Models (https://arxiv.org/abs/2412.20742)
Comments:
          12 pages, 5 figures

- **What's New**: 본 논문에서는 다양한 유형의 시각 입력을 통합하여 다중 시간 원격 감지 작업을 수행하는 최초의 비전-언어 모델인 UniRS를 소개합니다. UniRS는 단일 이미지, 이중 시간 이미지 쌍 및 비디오를 입력으로 지원하여 통합된 프레임워크 내에서 포괄적인 원격 감지 분석을 가능하게 합니다. 이러한 새로운 접근 방식은 원격 감지 분야에서 다중 작업 통합 연구의 패러다임을 제시합니다.

- **Technical Details**: UniRS는 다양한 시각 입력 유형을 수용하기 위해 통합된 시각 표현 접근 방식을 채택하고 있으며, 이중 시간 이미지 쌍의 경우 변화를 추출하는 모듈을 맞춤 설계하여 공간적 및 시간적 특징을 향상시킵니다. 이 모델은 또한 LLM의 사전 지식을 활용하여 추론 과정에서의 효율성을 높이기 위해 프롬프트 증강 메커니즘을 설계하였습니다. 이를 통해 UniRS는 다양한 원격 감지 작업에서 풍부한 시공간 특징을 학습할 수 있습니다.

- **Performance Highlights**: UniRS는 시각 질문 응답, 변화 캡셔닝 및 비디오 장면 분류와 같은 다양한 작업에서 최신 성능(SOTA)을 달성하고 있습니다. 특히, RSVQA-HR 테스트 세트에서 제로샷 성능이 이전의 SOTA VLM을 초과했으며, LEVIR-CC 및 ERA 테스트 세트에서도 전통적인 분류 모델보다 월등한 성능을 보였습니다. 이 결과는 UniRS의 다재다능함과 원격 감지 문제를 해결하는 효과성을 입증합니다.



### Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study (https://arxiv.org/abs/2412.20733)
Comments:
          The original work citation: Bačić, B., Claudiu Vasile, Feng, C., & Ciucă, M. G. (2024, 13-15 Dec.). Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study. Presented at the Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024), Sydney, NSW

- **What's New**: 이번 논문은 개인 정보를 보호하는 빅데이터 분석 헬스케어 플랫폼을 구축하기 위한 기여를 목표로 하고 있습니다. 이를 위해 환자들의 동영상이나 시계열 데이터 처리를 가능하게 하는 새로운 접근 방식을 제안하고 있습니다. Google MediaPipe를 활용하여 모바일 동영상을 개인 정보 보호 진단 시계열 데이터로 변환하는 과정을 소개합니다.

- **Technical Details**: 연구에서는 실제 무릎 재활 동영상 데이터셋을 사용하였으며, 다양한 운동을 정밀하게 분석하기 위한 알고리즘을 개발했습니다. 이 알고리즘은 환자의 운동 비디오에 스틱 피겨 요소를 오버레이하고, CSV 파일 형식으로 무릎 각도 추정을 업데이트하여 시계열 플롯을 생성합니다. 또한, 사전 정의된 무릎 각도 매개변수를 통해 문제를 시각적으로 나타내는 동영상이 가능해집니다.

- **Performance Highlights**: 제안된 적응형 알고리즘은 재활 프로그램 준수를 높이고 운동 세트 및 반복 수를 정확히 측정할 수 있습니다. 동영상의 측면 및 전면 보기에서 모든 운동을 91.67%에서 100%까지 정확히 식별할 수 있으며, 다양한 무릎 운동 패턴에 대한 투명한 알고리즘 설계는 해석 가능한 AI의 발전에 기여합니다. 이러한 연구는 향후 개인 정보 보호를 고려한 오픈 소스 개발로 이어질 것으로 기대됩니다.



### Dialogue Director: Bridging the Gap in Dialogue Visualization for Multimodal Storytelling (https://arxiv.org/abs/2412.20725)
- **What's New**: 이번 연구에서 우리는 Dialogue Visualization이라는 새로운 작업을 제안하며 대화 중심의 스크립트를 동적이고 다각적인 스토리보드로 변환하는 방법론을 소개합니다. 이 방법은 스크립트의 빈약한 설명 세부정보와 다양한 시각적 표현의 복잡성 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: Dialogue Director는 스크립트 디렉터, 촬영 감독, 스토리보드 제작자로 구성된 훈련 없는 다중 모달 프레임워크입니다. 이 프레임워크는 대규모 다중 모달 모델과 확산 기반 아키텍처를 활용하여 스크립트 이해, 물리적 맥락 이해, 영화 원칙 통합을 개선하는 다양한 기술을 사용합니다.

- **Performance Highlights**: 실험 결과, Dialogue Director는 스크립트 해석, 물리적 세계 이해 및 영화 원칙 적용에서 최신 기술보다 우수한 성능을 보이며 대화 기반 스토리 비주얼화의 품질과 조정 가능성을 크게 향상시켰습니다.



### 4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives (https://arxiv.org/abs/2412.20720)
Comments:
          Journal extension of ICLR 2024

- **What's New**: 이번 연구에서는 동적 장면을 스페이셜-템포럴 4D 볼륨 학습 문제로 재구성하였습니다. 이를 통해 동적 장면을 표현하기 위한 명시적 모델을 제공하고 최소한의 가정 하에 다양하게 활용 가능한 동적 장면 학습 프레임워크를 제안합니다. 4D Gaussian Splatting (4DGS) 모델을 채택하여 장면의 기하학적 및 외형적 속성을 포함한 정보를 완전히 캡처할 수 있도록 하였습니다.

- **Technical Details**: 4DGS 모델은 비대칭 타원체로 매개변수화된 4D Gaussian을 사용하여 시간에 따른 색상 변화를 모델링합니다. 이를 통해 자율적이고 유연하게 스페이셜 및 템포럴 정보의 통합을 달성하며, Spherindrical Harmonics를 통해 동적 장면의 외형 진화를 표현합니다. 고해상도, 포토리얼리스틱(realistic) 새로운 시각을 실시간으로 렌더링할 수 있는 첫 번째 솔루션입니다.

- **Performance Highlights**: 4DGS 모델은 동적 장면 기반 작업(예: 새로운 시각 합성, 콘텐츠 생성 등)에서 시각 품질과 효율성 면에서 기존 방법론보다 우수성을 입증하였습니다. 추가로, 이 모델은 저용량 변형을 통해 메모리 풋프린트(memory footprint)를 줄이고, 과적합(overfitting)의 위험을 완화하는 성능 개선을 포함합니다. 여러 동적 장면 관련 과제에서의 성능을 비교하며 다양한 상황(예: 단일 객체, 실내 및 실외 장면)에서의 적용 가능성을 보여주었습니다.



### M$^3$oralBench: A MultiModal Moral Benchmark for LVLMs (https://arxiv.org/abs/2412.20718)
- **What's New**: 최근 대규모 기초 모델, 특히 대규모 언어 모델(LLMs)과 대규모 비전-언어 모델(LVLMs)이 법률, 금융, 의료 등 다양한 분야에서 필수적인 도구로 자리잡고 있습니다. 이러한 모델의 일상적인 통합이 증가함에 따라, 인간의 가치와 도덕적 경계를 존중하는 출력을 보장하기 위한 도덕적 평가(moral evaluation)의 필요성이 대두되고 있습니다. 이전 연구들은 주로 LLMs에 집중해 도덕적 데이터셋과 텍스트 모드에 제한된 벤치마크를 제안했습니다.

- **Technical Details**: 이에 따라, 우리는 LVLMs를 위한 최초의 다중 모달 도덕 벤치마크(MultiModal Moral Benchmark)인 M$^3$oralBench를 소개합니다. M$^3$oralBench는 도덕 기초 시나리오(Moral Foundations Vignettes, MFVs)를 확장하고 텍스트-이미지 확산 모델인 SD3.0을 사용하여 관련 시나리오 이미지를 생성합니다. 이 벤치마크는 도덕 기초 이론(Moral Foundations Theory, MFT)의 여섯 가지 도덕적 기초에 따른 도덕적 평가를 실시하며, 도덕 판단(moral judgement), 도덕 분류(moral classification), 도덕적 반응(moral response) 과제를 포함하여 다중 모달 도덕 이해 및 추론 모델 성능을 종합적으로 평가합니다.

- **Performance Highlights**: 10개의 인기 있는 오픈 소스 및 클로즈드 소스 LVLM에 대해 광범위한 실험을 진행한 결과, M$^3$oralBench는 현재 모델의 현저한 도덕적 한계를 드러내는 도전적인 벤치마크임을 입증했습니다. 우리의 벤치마크는 공개 가능하며, 이를 통해 LVLMs의 도덕적 성능을 개선하기 위한 기초 데이터로 활용될 수 있습니다.



### HFI: A unified framework for training-free detection and implicit watermarking of latent diffusion model generated images (https://arxiv.org/abs/2412.20704)
- **What's New**: 본 논문에서는 기존의 AI 생성 이미지 탐지 방법들이 훈련 데이터에 의존하는 한계를 극복하기 위해 훈련이 필요 없는 설정(training-free setup)에서의 탐지 방법을 제안합니다. 특히, Latent Diffusion Models (LDMs)로 생성된 이미지의 탐지 과제가 현실적으로 어려운 점을 강조하며, 고주파 정보의 왜곡(aliased) 정도를 측정하는 새로운 접근 방식을 도입합니다. 이 방법은 High-frequency influence (HFI)라고 불리며, 현재의 탐지 기준을 뛰어넘는 성능을 보여줍니다.

- **Technical Details**: HFI는 LDM의 오토인코더(autoencoder)를 다운샘플링-업샘플링 커널(downsampling-upsampling kernel)로 간주하여, 재구성된 이미지에서 고주파 정보의 왜곡을 측정합니다. 이 방법은 훈련 없이도 효율적으로 작동하며, 다양한 생성 모델에 의해 생성된 이미지 탐지에서 다른 훈련 없이 사용 가능한 방법들을 일관되게 초월하는 성능을 보입니다. HFI는 고주파 성분을 강조하여 배경 정보의 영향을 줄이면서 효율적인 탐지를 가능하게 합니다.

- **Performance Highlights**: HFI는 매우 도전적인 AI 생성 이미지 탐지 벤치마크에서 기존의 훈련 없는 방법들보다 우수한 성능을 발휘합니다. 또한, HFI는 특정 LDM 모델에서 생성된 이미지를 효과적으로 트레이싱(tracing)하여 이미지 소유권을 구분할 수 있는 신뢰할 수 있는 불확실성 지표를 제공합니다. HFI는 입력 최적화(input optimization)에 기반한 최신 방법보다 속도면에서도 현저하게 빠른 성능을 보여주며, 경쟁력 있는 결과를 얻고 있습니다.



### Open-Set Object Detection By Aligning Known Class Representations (https://arxiv.org/abs/2412.20701)
Comments:
          Accepted to WACV'24

- **What's New**: 본 연구는 기존의 Open-Set Object Detection (OSOD) 방식을 개선하기 위해 새롭고 효과적인 semantic clustering 기반 접근법을 제안합니다. 이를 통해 모델이 객체 간의 의미적 경계를 더 잘 이해하도록 하고, unknown class와 known class 간의 misclassification을 줄입니다. 새로운 class decorrelation 모듈과 object focus 모듈을 도입하여 객체의 위치 및 지리적 특징을 학습함으로써 unknown 객체에 대한 탐지 성능을 높입니다.

- **Technical Details**: OSOD의 주요 문제는 unknown class의 misclassification입니다. 이를 해결하기 위해 본 연구에서는 semantic clustering 모듈을 도입하여 region proposal feature와 semantic class embedding을 정렬하고, orthogonality constraint를 적용하여 클러스터 간의 구분을 강화합니다. 추가적으로, object focus loss를 활용하여 centerness와 classification-based objectness를 결합하여 unknown 객체 탐지의 효과를 극대화합니다.

- **Performance Highlights**: 제안하는 모델은 MS-COCO 및 PASCAL VOC 데이터셋에서 기존의 최첨단 방법들과 비교하여 significant improvement를 보여주었습니다. 실험 결과, unknown 객체를 정확히 탐지할 수 있는 성능을 입증하였으며, 예를 들어 'zebra'를 올바르게 unknown 객체로 식별하는 성과를 보여주었습니다. 이를 통해 본 연구의 접근법이 open-set 물체 탐지 분야에서 유망함을 입증합니다.



### Learning to Rank Pre-trained Vision-Language Models for Downstream Tasks (https://arxiv.org/abs/2412.20682)
- **What's New**: 이 논문은 기존의 비지도 학습 (unsupervised learning) 환경에서 비전-언어 모델 선택 (Vision-Language Model Selection)의 문제를 제기합니다. 기존의 VLM 선택 방법은 주로 주석이 있는 데이터셋에 의존해왔지만, 본 연구에서는 주석이 없는 데이터셋과 클래스 이름만을 사용하여 VLM을 선택하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 새롭게 제안된 Visual-tExtual Graph Alignment (VEGA) 방법은 비전과 텍스트 모달리티 간의 정렬을 측정하여 VLM의 다운스트림 성능을 평가합니다. VEGA는 각 모달리티에서 정의된 두 개의 그래프를 사용하여, 노드 간 및 엣지 간의 유사성을 결합하여 VLM의 성능 점수를 계산합니다. 이 방법은 클러스터링과 유사도 측정을 통해 이미지와 해당 클래스 이름 간의 관계를 효과적으로 모델링합니다.

- **Performance Highlights**: 논문에서 수행한 광범위한 실험 결과, VEGA는 VLM의 성능 예측을 위한 신뢰할 수 있는 지표임을 입증했습니다. VEGA는 다양한 응용 시나리오에서 일관되게 신뢰할 수 있는 성능 추정치를 제공하며, 비지도 데이터를 사용한 VLM 선택에 대한 새로운 가능성을 열어줍니다.



### Recurrence-based Vanishing Point Detection (https://arxiv.org/abs/2412.20666)
Comments:
          WACV 2025

- **What's New**: 이번 논문에서는 Recurrence-based Vanishing Point Detection (R-VPD)라는 새로운 비지도 학습 기반의 소실점 탐지 방법을 제시합니다. 이 방법은 이미지 내에서 발견된 암시적 선(implicit lines)과 명시적 선(explicit lines)을 이용하여 소실점을 탐지합니다. 또한, 저자들은 3200개의 합성 이미지와 1400개의 실제 이미지로 구성된 두 개의 benchmark 데이터셋을 제공합니다.

- **Technical Details**: R-VPD 방법은 Recurring Patterns (RPs)로부터 생성된 시각적 단어들을 이용하여 암시적 선을 구성하고, 이 선들을 통해 소실점을 추정하는 방식입니다. 기존의 VPD 방법들은 보통 명시적 선에 의존하지만, R-VPD는 암시적 선을 활용함으로써 더 다양한 이미지에서 소실점을 탐지할 수 있습니다. 이 과정에는 Euclidean 거리 기반의 특성 매칭 행렬과 계층적 클러스터링(hierarchical clustering) 기법이 사용됩니다.

- **Performance Highlights**: R-VPD는 합성 이미지 데이터셋에서 모든 기존 방법들을 초과하는 성능을 발휘하며, 실제 이미지에서는 고전적인 방법들보다 뛰어난 성과를 보입니다. 특히, 비지도 학습 방법으로 개발된 이 기법은 기존의 지도 학습 접근 방식에 근접한 성능을 기록하여, 소실점 탐지 분야에서 새로운 가능성을 제시합니다.



### SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection (https://arxiv.org/abs/2412.20665)
- **What's New**: 이 논문은 원격 탐사에서 다중 모달 데이터셋과 다중 작업 객체 탐지(M2Det)라는 새로운 작업을 도입합니다. M2Det은 각기 다른 센서 모달리티로부터 수평 및 경사진 객체를 정확하게 탐지할 수 있도록 설계되었습니다. 기존의 객체 탐지 모델은 단일 데이터셋에만 의존하는 반면, 본 연구는 다양한 모달리티 간의 지식을 공유하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 SM3Det(Single Model for Multi-Modal datasets and Multi-Task object Detection)이라는 통합 모델을 제안하여 M2Det 작업의 문제를 해결합니다. 이 모델은 grid-level sparse Mixture of Experts(MoE) 구조를 통해 각기 다른 모달리티의 특징을 보존하며 지식을 공동으로 학습할 수 있게 합니다. 또한, 적응형 학습 속도 조정을 통한 일관성과 동기화 최적화 전략을 통합하여 다양한 모달리티와 작업 간의 학습 난이도 차이를 효과적으로 처리합니다.

- **Performance Highlights**: SM3Det 모델은 기존 개별 데이터셋에서 훈련된 전문 모델들을 일관되게 능가하는 성능을 보였으며, 경량화된 변형 또한 많은 파라미터를 줄이면서 뛰어난 성능을 발휘합니다. 이 모델은 다양한 백본과 탐지기에 적응할 수 있는 강력한 일반화 능력을 보여주며, 대규모 실험을 통해 제안된 단일 모델의 효과성을 입증했습니다.



### Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner (https://arxiv.org/abs/2412.20662)
- **What's New**: 본 논문은 사전 학습된 비전 대형 언어 모델(VLLMs)의 테이블 인식 적용에 대한 연구 격차를 해소하고자 합니다. 특히, 기존의 VLLM 기반 방법들과 달리, Fine-tuning 없이 프롬프트 기반의 패러다임을 이용해 테이블 인식을 다루는 새로운 접근 방식을 제안합니다. 이를 위해 다양한 계층 구조를 갖춘 벤치마크를 설계하고 실험을 통해 비전 모델의 성능을 향상시키기 위한 프레임워크인 Neighbor-Guided Toolchain Reasoner (NGTR)를 제안합니다.

- **Technical Details**: NGTR 프레임워크는 저품질 입력 이미지의 문제를 해결하기 위해 여러 경량 모델을 통합하여 시각 처리 작업을 수행합니다. 이 프레임워크는 이웃 검색 메커니즘을 활용하여 각 입력 인스턴스에 대해 유사한 이웃을 검색하고, 이를 통해 생성된 도구 호출 계획을 안내합니다. 또한, 각 단계의 도구 호출 과정에서 반영(reflection) 모듈을 포함시켜 도구 선택을 지속적으로 개선함으로써 VLLMs가 더 높은 품질의 구조화된 데이터를 생성할 수 있도록 합니다.

- **Performance Highlights**: 본 연구의 실험 결과, NGTR 프레임워크는 기존의 VLLM 기반 접근 방식의 테이블 인식 성능을 상당히 향상시킨 것으로 나타났습니다. 특히, 기존의 전통적인 모델들에 비해 VLLMs는 경쟁력 있는 정확도를 보여주었지만, 여전히 성능 차이를 보여 주었습니다. 이번 연구는 VLLMs의 다양한 공공 테이블 인식 데이터셋에서 가능한 성능 경계를 예비적으로 밝히고, 향후 연구의 기초가 될 수 있는 중요한 관점을 제시합니다.



### Diffgrasp: Whole-Body Grasping Synthesis Guided by Object Motion Using a Diffusion Mod (https://arxiv.org/abs/2412.20657)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 고유한 전신 그립(whole-body grasping) 모션 시퀀스를 생성하는 새로운 프레임워크인 DiffGrasp를 제안했습니다. 기존 연구들은 손잡이 자세 없이 인간의 상호작용 모션 시퀀스를 생성하거나 정적 그립 포즈만 모델링하는 데 그쳤으나, 우리의 방법은 이는 복잡한 손 움직임과 객체의 모션 시퀀스 간의 관계를 단일 diffusion 모델 내에서 공동 모델링합니다. 이를 통해 생생한 그립 동작과 신체 각 부분의 조화를 보장할 수 있습니다.

- **Technical Details**: DiffGrasp는 손, 몸, 객체 간의 복잡한 움직임을 모델링하기 위한 새로운 조건부 확산 모델을 사용합니다. 이 모델은 손의 접촉을 인지하고 매우 자연스러운 그립 결과를 생성할 수 있도록 훈련하는 데 두 가지 새로운 접촉 인식 손실(contact-aware losses)을 도입합니다. 이러한 방식을 통해 고유한 객체의 위치를 이해하고 더 자연스러운 그립 포즈를 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과, DiffGrasp는 최신 기술(state-of-the-art)과 비교했을 때 더 뛰어난 성능을 보였습니다. 우리의 프레임워크는 다양한 객체 형태를 위한 자연스럽고 안정적이며 현실적인 전신 그립 결과를 생성할 수 있으며, 시간이 지남에 따라 매끄러운 모션을 유지합니다. 이러한 혁신을 통해 DiffGrasp는 인간과 객체 간의 상호작용을 효과적으로 모델링하고 있습니다.



### Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis (https://arxiv.org/abs/2412.20651)
- **What's New**: 이 논문에서는 Latent Drift (LD)라는 새로운 접근 방식을 제안하여, 의료 이미징에서의 분포 이동(distribution shift) 문제를 완화할 수 있도록 diffusion 모델을 개선합니다. 이는 대규모 데이터셋의 접근이 제한된 의료 이미징에서 중요한 역할을 합니다. Latent Drifting은 반사상(counterfactual) 이미지 생성을 가능하게 하여, 성별, 나이, 질병의 유무 등의 변수를 반영하는 이미지 생성을 지원합니다.

- **Technical Details**: Latent Drift는 기존의 pre-trained 모델에 대한 미세 조정(fine-tuning)을 용이하게 하여, 데이터의 분포 변화에 따른 도전을 극복할 수 있게 해줍니다. 이 방법은 어떤 미세 조정 방식에도 적용 가능할 뿐만 아니라, 추론(inference) 시간에도 조건으로 사용될 수 있습니다. 본 연구에서는 Latent Drift를 활용하여 세 가지 공개 장기적 벤치마크(brain MRI, chest X-ray) 데이터셋에서 반사상 이미지를 생성하는 성능을 평가했습니다.

- **Performance Highlights**: 진행한 실험 결과, Latent Drift는 다양한 시나리오에서 성능 향상을 보였으며, 여러 미세 조정 방식과 결합했을 때 특히 유의미한 향상을 나타냈습니다. 이러한 결과는 의료 이미징 분야에서 diffusion 모델의 잠재력을 증명하는 중요한 증거가 됩니다. 이 연구 결과물의 소스 코드는 논문이 수락된 후 공개될 예정입니다.



### Enhancing Visual Representation for Text-based Person Searching (https://arxiv.org/abs/2412.20646)
- **What's New**: 이 논문에서는 텍스트 기반 인물 검색(text-based person search) 문제를 해결하기 위해 Visual Feature Enhanced Text-based Person Search (VFE-TPS) 모델을 제안합니다. 이는 기존의 unimodal 데이터로 사전 훈련된 인코더를 넘어, CLIP 모델을 활용하여 텍스트와 이미지 간의 효과적인 교차 모달(feature) 정렬을 도모합니다. 또한, Text Guided Masked Image Modeling (TG-MIM) 및 Identity Supervised Global Visual Feature Calibration (IS-GVFC)이라는 보조 작업을 통해 모델의 시각적 이해력을 크게 향상시킵니다.

- **Technical Details**: VFE-TPS 모델은 cross-modal retrieval 및 person re-identification을 결합한 새로운 접근 방식을 채택하고 있습니다. TG-MIM 작업은 masked image modeling 기술을 활용하여 텍스트 쿼리를 기반으로 지역적인 시각적 세부 정보를 강화하고, IS-GVFC 작업은 자연스러운 pedestrian ID 주석을 활용하여 시각적 특성의 분포 일치 손실을 최소화하여 인물 정체성 혼동 문제를 해결합니다. 이 모델은 두 가지 보조 작업을 통해 CLIP 모델의 지식을 성공적으로 전이 및 적응시킵니다.

- **Performance Highlights**: 실험 결과, VFE-TPS 모델은 기존 방법을 일정 비율로 초과하여 Rank-1 정확도가 약 1%에서 9% 향상되는 결과를 보여주었습니다. 이는 텍스트 기반 인물 검색 작업에 있어 기존 모델보다 현저한 성과를 낸 것을 뜻합니다. 다양한 벤치마크에서 실시된 실험은 모델의 효과성과 우수성을 명확히 입증하고 있습니다.



### YOLO-UniOW: Efficient Universal Open-World Object Detection (https://arxiv.org/abs/2412.20645)
- **What's New**: 이번 논문에서는 Open-World Object Detection의 새로운 패러다임인 Universal Open-World Object Detection(Uni-OWD)을 소개합니다. YOLO-UniOW라는 새로운 모델을 통해 데이터를 이질적으로 결합하는 cross-modality fusion의 부담을 줄이며, 효율성(efficiency)과 범용성(versatility), 성능(performance) 측면에서 크게 발전합니다. 또한, Wildcard Learning 전략을 도입하여 점진적인 학습(incremental learning) 없이도 새로운 카테고리의 동적 확장을 가능하게 하였습니다.

- **Technical Details**: YOLO-UniOW는 클립(clip) 잠재 공간에서 가벼운 정렬(lightweight alignment)과 Adaptive Decision Learning을 통해 계산 비용이 높은 cross-modality fusion을 대체합니다. 이 방식은 모델의 일반화(generalization)를 해치지 않으면서도 효율적인 탐지(detection)를 가능하게 합니다. Wildcard Learning 전략은 분포 외(out-of-distribution) 객체를 'unknown'으로 탐지하며, 이를 통해 오픈 월드 환경에서의 새로운 카테고리에 원활하게 적응할 수 있는 능력을 부여합니다.

- **Performance Highlights**: YOLO-UniOW는 LVIS에서 34.6 AP와 30.0 APr의 성능을 달성하며, 69.6 FPS의 추론 속도(inference speed)를 기록했습니다. 또한 M-OWODB, S-OWODB, nuScenes 데이터세트에서 벤치마크 성능을 선보이며 오픈 월드 객체 탐지(open-world object detection) 분야에서 독보적인 성과를 입증했습니다. 관련 코드와 모델은 제공된 URL에서 확인할 수 있습니다.



### Slow Perception: Let's Perceive Geometric Figures Step-by-step (https://arxiv.org/abs/2412.20631)
- **What's New**: 이 논문에서는 'slow perception' (SP)이라는 새로운 개념을 도입하여 모델이 복잡한 기하학적 구조를 인간처럼 점진적으로 인식하도록 유도합니다. 특히, SP는 기하학적 도형의 인식을 단순한 점과 선의 조합으로 분해하고, 각 선을 스텝별로 세심하게 추적하는 두 가지 단계로 구성됩니다. 이러한 접근 방식은 모델의 시각적 인식 속도를 늦추면서도 정확성을 높일 수 있음을 보여줍니다.

- **Technical Details**: SP는 두 가지 주요 단계로 나뉩니다: 첫 번째는 인식 분해(perception decomposition)로, 복잡한 기하학적 도형을 기본 단위로 나누어 기하학적 표현을 통합합니다. 두 번째는 인식 흐름(perception flow)으로, 각 선을 단계별로 추적하는 방식을 통해 정확한 선의 정의를 제공합니다. 여기서 각 선은 시작점에서 종료점에 이르는 일련의 점으로 표현되며, 이를 위해 'perceptual ruler'를 사용합니다.

- **Performance Highlights**: 논문에서는 SP가 F1-score를 6% 향상시키는 효과가 있음을 보였습니다. 또한, 인식 흐름 방식이 라인 예측의 정확성을 일관되게 향상시키며, perceptual ruler의 길이가 줄어들수록 각 선 segment의 예측되는 계산 비용이 증가하면서 성능이 점진적으로 개선된다는 흥미로운 결과를 발견했습니다. 또한, 작성한 200,000개의 합성 데이터 샘플과 중학교 시험지에서 수집한 480개의 실제 기하학적 도형 데이터세트를 개방하여 커뮤니티 개발을 촉진할 예정입니다.



### HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models (https://arxiv.org/abs/2412.20622)
- **What's New**: 본 논문에서는 LVLMs (Large Vision-Language Models)에서 발생하는 객체 환각(object hallucination) 문제를 해결하기 위한 새로운 벤치마크인 HALLUCINOGEN을 제안합니다. 기존 벤치마크는 주로 단순한 개체 확인 프롬프트에 의존했으나, HALLUCINOGEN은 다양한 맥락적 추론(prompt) 능력을 활용하여 LVLMs의 객체 식별 정확성을 평가합니다. 이와 함께, MED-HALLUCINOGEN이라는 의료 분야에 맞춤 형태의 환각 공격을 도입하여, 의료 이미지에서의 LVLMs 성능을 검증하고자 합니다.

- **Technical Details**: HALLUCINOGEN 벤치마크에서는 60,000개의 이미지-프롬프트 조합을 포함한 3,000개의 시각-객체 쌍을 통해 LVLMs의 정확한 객체 식별 능력을 검토합니다. 두 가지 유형의 환각 공격인 정확한(object hallucination) 및 암시적(implicit) 공격을 구분하여 평가합니다. 메디컬 분야에서의 환각 성능을 측정하기 위해 NIH Chest X-rays 데이터셋을 이용한 MED-HALLUCINOGEN을 설계하였으며, 이를 통해 감독이 필요한 문맥에서 LVLMs의 신뢰성을 평가하고자 합니다.

- **Performance Highlights**: 여덟 개의 LVLM과 두 가지 환각 완화 전략에 대한 평가 결과, HALLUCINOGEN 및 MED-HALLUCINOGEN의 환각 공격에 대해 대부분의 SOTA LVLM들은 무작위 추측에 가까운 성능을 보임을 확인하였습니다. 특히 LVLM이 Chain-of-Thought (CoT) 추론을 사용할 경우 환각 현상이 더욱 증가하는 경향이 있음을 보여주었습니다. 이를 통해, 현재 LVLM들은 심각한 환각 공격에 취약하다는 것을 입증하였습니다.



### FreqMixFormerV2: Lightweight Frequency-aware Mixed Transformer for Human Skeleton Action Recognition (https://arxiv.org/abs/2412.20621)
Comments:
          IEEE FG2025

- **What's New**: 이 논문은 FreqMixFormer를 기반으로 한 FreqMixFormerV2라는 경량화된 인체 동작 인식 모델을 제안합니다. 이 모델은 주파수 도메인 분석(frequency-domain analysis)을 활용하여 미세한 동작을 인식할 수 있도록 설계되었습니다. 특히, 모델 복잡성을 줄이면서도 성능은 유지하였다는 점이 주요 혁신입니다.

- **Technical Details**: FreqMixFormerV2는 혼합 주의 블록(mixed attention block)의 구조를 단순화해 두 개의 채널 입력(input)을 사용합니다. 이는 계산량을 줄이면서 필수적인 고유의 뼈대 정보를 유지할 수 있도록 합니다. 또한, 새로운 고주파 및 저주파 연산자(high-low frequency operator)를 도입하여 주파수 계수를 효율적으로 조정하는 기능이 추가되었습니다.

- **Performance Highlights**: 세 가지 표준 데이터세트(NTU RGB+D, NTU RGB+D 120, NW-UCLA)에서 FreqMixFormerV2는 원래 모델의 60% 파라미터로 유사한 성능을 기록하였으며, 정확도 손실은 0.8%에 불과했습니다. 이러한 결과는 효율성과 정확성 간의 우수한 균형을 보여줍니다.



### Do Current Video LLMs Have Strong OCR Abilities? A Preliminary Study (https://arxiv.org/abs/2412.20613)
Comments:
          Accepted by CoLing 2025 (The 31st International Conference on Computational Linguistics)

- **What's New**: 본 논문에서는 비디오 콘텐츠에서 문자 정보를 정확하게 추출하고 이해하는 비디오 기반 광학 문자 인식(Video OCR)의 성능을 평가하기 위한 새로운 벤치마크를 소개합니다. 이 벤치마크는 1,028개의 비디오와 2,961개의 질문-답변 쌍으로 구성되어 있으며, 비디오 내 다중 모드 모델의 Video OCR 성능을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 벤치마크는 6개의 별도 하위 작업을 통해 여러 주요 과제를 제안합니다: (1) 텍스트 콘텐츠 인식 및 기본 시각적 속성, (2) 비디오 내 OCR 객체의 의미적 및 공간적 이해, (3) 동적 움직임 탐지 및 시간적 위치 지정입니다. 연구진은 이미지 LLMs의 OCR 능력을 수동 수정과 통합하여 반자동 방식으로 이 벤치마크를 개발하였으며, 효율성, 비용 및 데이터 품질 간의 균형을 맞추었습니다.

- **Performance Highlights**: 이 자원은 비디오 LLMs 연구를 진전시키고 비디오 LLMs의 OCR 능력을 향상시킬 필요성을 강조하는 데 도움이 될 것입니다. 최종적으로 이 벤치마크는 연구자들이 활용할 수 있도록 공개될 예정입니다.



### Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond) (https://arxiv.org/abs/2412.20596)
Comments:
          Code can be found at: this https URL

- **What's New**: 최근 이미지 복원 과제에서 사전 학습된 확산 모델(Diffusion Model, DM)과 데이터 충실도 가이드를 이용하여 '제로샷'(zero-shot) 복원 방법이 각광받고 있습니다. 이 연구는 4개의 Neural Function Evaluations (NFEs)만으로 기존의 방법보다 우수한 성능을 발휘하는 새로운 제로샷 복원 방안을 제안합니다. 이 접근은 우수한 초기화, 역투영 가이드, 그리고 새로운 노이즈 주입 메커니즘을 통해 이뤄집니다.

- **Technical Details**: 본문에서는 이미지 복원의 여러 작업을 해결하기 위한 여러 가지 기법이 소개되며, 특히 Consistency Models (CMs)를 활용합니다. 노이즈 주입 메커니즘은 두 가지 주요 구성요소를 통해 반복 횟수를 줄이는 데 기여합니다. 첫째, 복원 작업에서의 노이즈 수준을 주입 노이즈 수준과 분리하여 조정하며, 둘째, 주입하는 노이즈의 양을 확률적 노이즈와 추정된 노이즈로 분할하여 샘플링을 개선합니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 이미지 슈퍼 해상도, 블러 제거(deblurring), 그리고 인페인팅(inpainting) 등의 인기 있는 작업에서 기존 제로샷 방법들보다 더 우수한 성과를 보여줍니다. 특히, 노이즈 주입 기법은 CMs 이외에도 기존 가이드 DM 방법의 성능 저하를 완화하는 데에도 효과적임을 증명하였습니다.



### Enhancing autonomous vehicle safety in rain: a data-centric approach for clear vision (https://arxiv.org/abs/2412.20565)
Comments:
          16 pages, 16 figures, 2 tables

- **What's New**: 본 연구는 비가 오는 날씨에서 자율주행차(AV)의 내비게이션 문제를 해결하기 위해 최신 딥러닝 기술을 활용하여 비로 인한 시각적 방해를 제거하는 비전 모델을 개발합니다. 이 모델은 자동차 캠코더의 실시간 피드를 처리하여 맑고 비가 오지 않는 장면과 유사한 비주얼을 생성합니다. 새로운 배치 전략을 통해 비가 내리는 장면에서 높은 주파수의 비 패턴을 효과적으로 구분하여 모델 학습과 추론 성능을 향상시키는 점에서 차별점을 가지고 있습니다.

- **Technical Details**: 이 연구에서는 Car Learning to Act (CARLA) 시뮬레이션 환경을 활용하여 비 오는 이미지와 그에 상응하는 맑은 이미지를 포함하는 종합적인 데이터셋을 생성합니다. 모델 아키텍처는 전통적인 encoder-decoder 구조를 기반으로 하며, skip connection 및 concatenation 연산을 포함하여 고해상도 이미지를 처리할 수 있게 확장됩니다. 두 가지 새로운 배치 기법을 도입하여 비가 내리는 패턴과 배경 장면을 효과적으로 처리한 결과, 모델은 안정성 및 추론 성능을 향상시킵니다.

- **Performance Highlights**: 결과적으로 개발한 모델은 steering module과 통합하여 비가 오는 날씨에서도 운전 성능을 크게 향상시킵니다. 비가 내리는 조건과 맑은 조건에서의 스티어링 성능을 비교한 결과, 이 모델을 통해 AV의 내비게이션 안전성과 신뢰성이 크게 개선됨을 보여주었습니다. 모델의 성능을 정량적으로 평가하기 위해 PilotNet을 사용하였으며, 여기에 의해 예측된 스티어링 각도가 맑은 이미지에서와 유사한 결과를 나타냅니다.



### Exploiting Aggregation and Segregation of Representations for Domain Adaptive Human Pose Estimation (https://arxiv.org/abs/2412.20538)
Comments:
          accepted by the 2025 IEEE International Conference on Automatic Face and Gesture Recognition (FG 2025)

- **What's New**: 우리의 연구는 헬스케어, 가상현실 등 다양한 분야에 응용되는 인체 자세 추정을 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 도메인 적응 과정에서 도메인 불변(domain-invariant)과 도메인 특정(domain-specific) 컴포넌트로 특징을 분리하여 성능을 향상시킵니다. 또한 여러 키포인트 간의 관계를 세분화하여 측정 차이를 최소화하는 새로운 메커니즘을 도입했습니다.

- **Technical Details**: 인체 자세 추정(HPE)에서 도메인 적응의 중요성을 설명하며, 기존 방법들이 도메인 불변 및 도메인 특정 요소를 혼합하여 사용함으로써 발생하는 성능 저하를 지적합니다. 우리의 프레임워크는 이러한 요소들을 명확히 구분하고, 도메인 불변 특징을 집계(aggregation)하며 도메인 특정 특징을 분리(segregation)합니다. 다양한 벤치마크에 대한 실험을 통해 이 방법의 효과성을 입증했습니다.

- **Performance Highlights**: 다양한 기준에서 수행된 실험 결과, 제안된 방법이 기존의 최첨단 방법들보다 한층 더 우수한 성능을 보였다는 것을 확인했습니다. 특히, Human3.6M, LSP, H3D, FreiHand 등의 데이터셋에서 두각을 나타내며, 도메인 간 격차 문제를 효과적으로 해결했습니다. 이 연구는 미래의 HPE 분야에 큰 기여를 할 것으로 기대됩니다.



### MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks (https://arxiv.org/abs/2412.20522)
- **What's New**: 이번 연구에서는 3D Gaussian Splatting(3DGS)의 메모리 소모 문제를 효과적으로 해결하기 위해 MaskGaussian을 도입했습니다. 기존의 수동으로 설정된 기준이나 학습된 마스크를 사용하는 방법과 달리, 이 방법은 Gaussians를 영구적으로 제거하기보다는 확률적으로 존재하는 개체로 모델링하여 동적으로 활용합니다. 또한, masked-rasterization 기법을 통해 사용되지 않는 Gaussians도 경량화된 이미지 생성을 지원하며, 이들의 존재 가능성을 조정할 수 있습니다.

- **Technical Details**: MaskGaussian은 Gaussian의 존재 확률에 따라 이를 샘플링하여 사용하며, 이는 역전파가 가능하여 Gaussians의 기여도를 동적으로 평가할 수 있도록 합니다. 구체적으로, 전송 감쇠(transmittance attenuation)와 색상 합산(color accumulation) 단계에서 마스크를 적용하여, 비샘플링된 Gaussians도 그래디언트를 받을 수 있게 하여 존재 확률을 조정하도록 합니다. 이 과정에서 MaskGaussian은 기존의 수동 마스크 생성 방식의 단점을 극복하여 Gaussians의 변화하는 중요성을 반영합니다.

- **Performance Highlights**: 실험 결과, MaskGaussian 기법은 이전의 가지치기 방법보다 60% 이상의 Gaussian을 줄이면서도 PSNR(peak signal-to-noise ratio) 손실이 단 0.02에 불과하여 더 나은 렌더링 품질을 제공합니다. 다양한 데이터셋에서 평균 62.4%, 67.7%, 75.3%의 가지치기 비율을 기록하며, Mip-NeRF360, Tanks & Temples, Deep Blending의 렌더링 속도는 각각 2.05×, 2.19×, 3.16× 향상되었습니다. 이 연구 결과는 MaskGaussian의 우수성을 강조하며 향후 3DGS 마스크 적용에 대한 새로운 대안을 제시합니다.



### Can Robots "Taste" Grapes? Estimating SSC with Simple RGB Sensors (https://arxiv.org/abs/2412.20521)
- **What's New**: 이 연구는 테이블 포도 재배에서 수확을 위해 포도 품질을 평가하는 방법에 대해 다룹니다. 특히, RGB 센서를 이용하여 Soluble Solid Content (SSC) 및 색상을 추정할 수 있는 가능성을 탐구하고, 로봇-assisted harvesting을 위한 저비용 솔루션을 제안합니다. 이 연구는 2021-2022년 여름 동안 포도 이미지와 SSC, 색상 레이블을 수집하여 알고리즘 솔루션을 평가하였습니다.

- **Technical Details**: 연구에서는 RGB 카메라와 같은 저비용 수동 센서를 사용하여 농작물의 내부 품질을 평가하는 방법을 제시합니다. 고급 알고리즘 기술을 통해 포도 색상과 함께 SSC를 추정할 수 있는 방법론이 개발되었습니다. 실험에서는 다양한 RGB 카메라에서 수집한 포도 이미지로부터 SSC 추정을 위한 경량화된 알고리즘과 딥러닝을 기반으로 한 방법이 검증되었습니다.

- **Performance Highlights**: 제안된 방법론은 인간의 성능에 가까운 수준으로 SSC 추정이 가능함을 보여주었습니다. 측정 결과는 RGB 센서를 활용한 비파괴적이고 비용 효율적인 품질 평가의 가능성을 강조합니다. 이 연구는 농업 상황에서 로봇 시스템에 통합 가능한 신뢰성 있는 수확 지원을 위한 기초 자료를 제공합니다.



### DPBridge: Latent Diffusion Bridge for Dense Prediction (https://arxiv.org/abs/2412.20506)
- **What's New**: 본 연구에서는 DPBridge라는 새로운 생성 프레임워크를 제안합니다. 이는 기존의 diffusion 모델들의 한계를 극복하기 위해 dense prediction 과제를 이미지-조건 생성 문제로 재구성하고, 입력 이미지와 해당하는 dense map 간의 직접적인 매핑을 확립합니다. 이를 통해 성능 저하 및 느린 추론 속도와 같은 문제를 해결하고자 하였습니다.

- **Technical Details**: DPBridge는 학습된 이미지 diffusion 백본을 활용하여 dense prediction 작업을 처리하는 latent diffusion bridge 모델입니다. 본 모델은 이미지와 dense signal map 간의 latent 분포의 diffusion bridge 과정을 구성하며, 이는 중간 샘플들이 쌍으로 이루어진 데이터의 볼록 조합이 되도록 합니다. 이러한 방식은 점진적으로 입력 이미지에서 시작해 생성할 신호 맵으로의 변환을 수행합니다.

- **Performance Highlights**: DPBridge는 다양한 벤치마크에서 feed-forward 및 diffusion 기반 접근 방식과 비교하여 경쟁력 있는 성능을 달성하였습니다. 이를 통해 DPBridge의 효과성과 적응력을 강조하며, depth estimation, surface normal prediction 및 semantic segmentation과 같은 대표적인 dense prediction 작업에서의 성과를 확인할 수 있었습니다.



### ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding (https://arxiv.org/abs/2412.20504)
- **What's New**: 이번 연구에서는 비디오 이해에서 Video Large Language Models (VideoLLMs)의 한계를 극복하기 위해 새로운 방법인 ReTaKe를 제안합니다. 기존 VideoLLM들은 긴 비디오를 처리하는 데 어려움을 겪고 있으며, ReTaKe는 두 가지 새로운 모듈인 DPSelect와 PivotKV를 통해 이를 해결합니다. 이 방법은 비디오에서 시간적 시각 중복과 지식 중복을 모두 효과적으로 모델링하여 긴 비디오 이해를 지원합니다.

- **Technical Details**: ReTaKe는 훈련 없는 방식으로, DPSelect는 비주얼 특징에 기반하여 키프레임을 선택하고, PivotKV는 선택된 키프레임을 피벗으로 사용하여 저주의 높은 점수를 가진 비피벗 토큰의 KV 캐시 압축을 수행합니다. 이러한 접근은 기존의 방법들보다 더 정교하게 비디오의 정보 밀도를 높이고, 긴 비디오에서의 이해도를 향상시킵니다. 실험 결과, ReTaKe는 기존 VideoLLM보다 4배 긴 비디오 시퀀스를 처리할 수 있으며, 성능 저하는 1% 미만입니다.

- **Performance Highlights**: 다양한 벤치마크 테스트에서 ReTaKe는 MiniCPM-V2.6-8B 및 LLaVA-OneVision-7B와 같은 유사 크기의 VideoLLM을 3%-5% 향상시켰으며, 더욱 큰 모델들인 InternVL2-34B 및 VideoLLaMA 2-72B와 맞먹거나 초과하는 성과를 보였습니다. DPSelect와 PivotKV의 효과를 검증하는 아블레이션 연구에서는, DPSelect가 행동 추론과 속성 인식 작업에서 중요한 저수준 시각 세부 정보를 더 잘 보존하는 것을 입증했습니다.



### MR-Occ: Efficient Camera-LiDAR 3D Semantic Occupancy Prediction Using Hierarchical Multi-Resolution Voxel Representation (https://arxiv.org/abs/2412.20480)
Comments:
          11 pages, 5 figures, 9 tables

- **What's New**: 이 논문에서는 자율 주행의 3D 인식 기술에 있어 카메라와 LiDAR를 조합한 새로운 접근법인 MR-Occ를 제안합니다. MR-Occ는 Hierarchical Voxel Feature Refinement, Multi-scale Occupancy Decoder, Pixel to Voxel Fusion Network의 세 가지 핵심 구성 요소를 통해 3D semantic occupancy 예측에 대한 정확성과 효율성을 높입니다. 이 방법은 기존 접근 방식에 비해 +5.2% IoU와 +5.3% mIoU 향상을 보여주며, 효율적인 계산 자원 사용을 보장합니다.

- **Technical Details**: MR-Occ는 LiDAR의 고해상도 3D 위치 정보를 활용하여 점유 예측을 수행하고, 카메라의 풍부한 의미 정보를 통해 객체 분류를 개선합니다. 우리의 Hierarchical Voxel Feature Refinement(HVFR) 방법은 중요한 복셀의 해상도를 선택적으로 증가시켜 효율성을 높이며, Multi-scale Occupancy Decoder(MOD)는 'occluded' 클래스를 도입하여 가려진 영역을 효과적으로 처리합니다. 마지막으로, Pixel to Voxel Fusion Network(PVF-Net)는 densified LiDAR 기능을 통해 카메라 및 LiDAR 데이터의 효과적인 결합을 가능하게 합니다.

- **Performance Highlights**: MR-Occ는 nuScenes-Occupancy 데이터셋에서 최첨단 성능을 달성하며, 이전의 최상의 방법인 OccGen [3]에 비해 +5.2%의 IoU 및 +5.3%의 mIoU를 기록합니다. 이 접근법은 더 적은 매개변수와 FLOPs를 요구하며, SemanticKITTI 데이터셋에서도 뛰어난 성능을 보여 다양한 3D semantic occupancy 벤치마크에서 효과성과 일반성을 검증합니다.



### Toward Scene Graph and Layout Guided Complex 3D Scene Generation (https://arxiv.org/abs/2412.20473)
Comments:
          13 pages, 12 figures

- **What's New**: 최근 객체 중심 텍스트-3D 생성 기술이 눈부신 성과를 나타내고 있으나, 복합적인 3D 장면을 생성하는 데는 여전히 많은 도전 과제가 존재합니다. 기존의 방법들은 주로 score distillation sampling(SDS) 방식에 기반하고 있어, 특정 상호작용이 필요한 다 객체의 조작이 어렵습니다. 이 문제를 해결하기 위해 우리는 Scene Graph and Layout Guided 3D Scene Generation(GraLa3D)이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: GraLa3D는 텍스트 프롬프트를 바탕으로 복합적인 3D 장면을 모델링하는 데 LLM을 활용하며, 장면 그래프와 레이아웃 바운딩 박스 정보를 사용합니다. 이 접근법은 단일 객체 노드와 조합 슈퍼 노드로 구성된 장면 그래프를 독특하게 만들어, 객체 간의 관계와 상호작용을 더욱 명확하게 설명합니다. Node-to-3D Generation 단계에서는 이러한 슈퍼 노드를 통해 객체 간의 상호작용을 잘 처리하고, 객체가 서로 얽히지 않도록 설계했습니다.

- **Performance Highlights**: 우리는 GraLa3D가 기존의 한계를 극복하고, 텍스트 프롬프트와 일치하는 복잡한 3D 장면을 성공적으로 생성함을 실험적으로 확인했습니다. 이 프레임워크는 위치와 크기를 정확히 반영할 뿐만 아니라, 상호작용을 생생하게 표현하며 최종적으로 모든 객체를 공간적으로 재배치하여 일관된 3D 장면을 구현합니다.



### JADE: Joint-aware Latent Diffusion for 3D Human Generative Modeling (https://arxiv.org/abs/2412.20470)
- **What's New**: 이 논문에서 제안하는 JADE는 3D 인간 모델링을 위한 새로운 생성적 프레임워크로, 인체의 형태 변화를 세밀하게 제어할 수 있도록 설계되었습니다. JADE의 핵심 통찰력은 인체를 뼈대 구조와 표면 기하학으로 나누는 joint-aware latent representation을 도입하여, 기하학적 및 의미론적 해석이 가능하도록 하는 것입니다. 이를 통해 사용자는 인체를 더 유연하게 조작할 수 있게 됩니다.

- **Technical Details**: JADE는 각각의 관절에 대한 인코딩을 통해 인체 모델이 joint token의 시퀀스로 설명될 수 있도록 하는 joint-aware latent representation을 도입합니다. 이러한 disentangled latent space 디자인은 local sampling 및 관절에 연관된 인체 부분의 편집을 자연스럽게 가능하게 합니다. 또한, Transformer 기반 아키텍처를 통해 관절 간의 피쳐를 융합해 모델의 표현력을 향상시킵니다.

- **Performance Highlights**: JADE는 여러 공공 데이터셋에서 여러 작업 수행 시 기존 방법들과 비교하여 autoencoding 재구성 정확도, 편집 가능성 및 생성 품질에 있어 우수한 성능을 보임을 입증했습니다. 본 연구에서는 매끄럽고 신뢰성 높은 인간 형태를 생성하고 다양한 형태 편집 및 변형 애플리케이션을 가능하게 하며, 그 유용성을 강조했습니다.



### Single-image reflection removal via self-supervised diffusion models (https://arxiv.org/abs/2412.20466)
- **What's New**: 이 연구에서는 투명한 표면에서 촬영한 이미지에서 반사를 제거하기 위한 새로운 혼합 접근 방식을 제안합니다. 제안하는 방법은 paired training data 없이 단일 이미지에서 효과적으로 반사를 제거하는 데 초점을 맞추고 있습니다. 연구의 핵심은 두 개의 네트워크, Reflective Removal Network (RRN)와 Reflective Synthesis Network (RSN)을 활용하여 비선형 주의 메커니즘을 통해 분리된 구성 요소를 재합성하는 것입니다.

- **Technical Details**: 제안된 방법은 Denoising Diffusion Probabilistic Models (DDPM)을 활용하여 반사 이미지와 전송 이미지를 분해하는 과정을 모델링합니다. RRN은 주어진 참조 이미지에서 반사를 제거하기 위한 전송 이미지를 복원하며, RSN은 이 전송 이미지를 기반으로 다시 원본 이미지를 재구성합니다. 본 연구는 Museum Reflection Removal (MRR) 데이터셋을 포함하여 여러 데이터셋에서 광범위한 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 최첨단 기술들보다 우월한 성과를 보여주었으며, PSNR, SSIM, LPIPS 및 RAM 지표에서 개선된 수치를 기록했습니다. 특히, SIR2, FRR, MRR 데이터셋에서 각각 0.50-3.84 dB의 PSNR 증가와 0.005-0.074의 SSIM 개선을 달성했습니다. 이러한 성과는 실제 세계의 복잡한 반사 문제를 다루는데 있어 제안된 방법이 매우 효과적임을 나타냅니다.



### Cross-Modal Fusion and Attention Mechanism for Weakly Supervised Video Anomaly Detection (https://arxiv.org/abs/2412.20455)
Comments:
          Accepted to CVPR'24 MULA Workshop

- **What's New**: 이번 논문에서는 최신 연구 방향인 약한 감독 비디오 이상 탐지(WS-VAD)에 대한 새로운 다중 모달 프레임워크를 제안합니다. 이 프레임워크는 비디오에서 폭력 및 누드와 같은 이상현상을 효과적으로 탐지하기 위한 권장 사항을 제공합니다. 특히, Cross-modal Fusion Adapter (CFA)라는 새로운 융합 메커니즘을 도입하여 시각 모달리티에 관련된 오디오-비주얼 특성을 역동적으로 선택 및 향상시킵니다.

- **Technical Details**: 제안된 방법은 비디오 수준의 이진 레이블을 사용하여 이상 이벤트를 학습합니다. CFA 모듈은 각각의 모달리티의 기여도를 동적으로 조정하여 시각 모달리티에 대한 오디오 특성의 중요성을 우선시합니다. 더욱이, Hyperbolic Lorentzian Graph Attention (HLGAtt) 메커니즘을 통해 정상 및 비정상 표현 간의 계층적 관계를 효과적으로 캡처하여 이상 특성을 더 정확하게 구별합니다.

- **Performance Highlights**: 제안된 모델은 폭력 및 누드 탐지의 벤치마크 데이터세트에서 기존의 최첨단(SOTA) 방법보다 우수한 성능을 보여주었습니다. 여러 실험을 통해 모델의 유효성을 입증하며, 시각 모달리티와 오디오 모달리티 간의 상호작용을 개선하여 탐지 정확도를 높이고 있습니다.



### Image Augmentation Agent for Weakly Supervised Semantic Segmentation (https://arxiv.org/abs/2412.20439)
- **What's New**: 이 논문에서는 Weakly-Supervised Semantic Segmentation (WSSS)의 새로운 접근 방식인 Image Augmentation Agent (IAA)를 제안합니다. IAA는 대규모 언어 모델(LLMs)과 확산 모델(diffusion models)을 활용해 추가적인 이미지를 자동으로 생성하여 WSSS의 성능을 개선할 수 있음을 보여줍니다. 이 방법은 데이터 생성 관점에서 WSSS를 강화하며, 동적인 프롬프트 품질 평가를 통해 이미지의 일관성을 보장합니다.

- **Technical Details**: IAA는 ControlNet과 GPT-4o를 통합하여 추가적인 이미지를 생성하는 구조로 이루어져 있습니다. 프롬프트 자가 개선(self-refinement) 메커니즘을 통해 LLM의 출력 품질을 높이고, 생성된 이미지의 품질과 클래스 분포를 보장하기 위해 분산 생성 과정에 온라인 필터를 추가합니다. 이 방식으로 다채롭고 품질 높은 합성 데이터가 생성될 수 있습니다.

- **Performance Highlights**: PASCAL VOC 2012 및 MS COCO 2014 데이터셋에서 실험한 결과, 제안된 IAA 방법이 기존의 최첨단 WSSS 접근법보다 뛰어난 성능을 보여주었습니다. WSSS 성능 향상에 기여하는 이 연구는 제한된 훈련 데이터를 바탕으로 더욱 다양한 이미지 생성을 가능하게 하여 의미론적 패턴의 이해를 돕습니다.



### ESVQA: Perceptual Quality Assessment of Egocentric Spatial Videos (https://arxiv.org/abs/2412.20423)
Comments:
          7 pages, 3 figures

- **What's New**: 이 논문에서는 egocentric spatial videos의 품질 평가를 위한 새로운 데이터베이스인 'ESVQAD'를 소개합니다. ESVQAD는 600개의 다양하고 고품질의 영상과 그에 대한 평균 평가 점수(MOS)를 포함하여, immersiveness와 engagement를 향상시키는 품질 평가 문제를 다룹니다. 또한, 'ESVQAnet'이라는 다차원 binocular feature fusion 모델을 제안하여, 모션 및 의미 정보와 결합하여 인식 품질을 예측합니다.

- **Technical Details**: 제안된 ESVQAnet은 binocular 비디오에서 공간, 모션, 의미 특징을 추출하여 품질 점수를 예측합니다. 공간 특징 추출기는 VSSD 블록과 MSA 블록을 사용하여 다층적 방법으로 공간 특징을 추출하며, 모션 특징 추출기는 3D-CNN을 사용하고, 의미 특징 추출기는 CLIP 비주얼 인코더의 변환기를 활용하여 영상을 분석합니다. 최종적으로, 다층 퍼셉트론(MLP)을 이용하여 여러 특징들을 통합하여 품질 점수를 생성합니다.

- **Performance Highlights**: ESVQAnet은 ESVQAD에서 16개의 최신 비디오 품질 평가 모델(VQA models)보다 뛰어난 성능을 보여줍니다. 연구 결과는 ESVQAnet이 전통적인 VQA 작업에서도 강력한 일반화 능력을 발휘함을 증명하였습니다. 이와 함께, 신뢰성 있는 MOS 배포와 여러 참가자에 대한 시험 결과를 통해 높은 품질의 평가 점수를 확보하였습니다.



### Bringing Objects to Life: 4D generation from 3D objects (https://arxiv.org/abs/2412.20422)
- **What's New**: 최근 생성 모델의 발전으로 텍스트 프롬프트를 통해 제어된 4D 콘텐츠(움직이는 3D 객체)를 생성할 수 있게 되었습니다. 이 방법은 가상 세계, 미디어 및 게임과 같은 응용 분야에서 큰 잠재력을 가지고 있지만, 기존 방법들은 생성된 콘텐츠의 외형과 기하학적 형태에 대한 제한된 제어만을 제공합니다. 본 연구에서는 사용자 제공 3D 객체를 애니메이션화하여 4D 생성을 유도하는 새로운 방법을 제안합니다.

- **Technical Details**: 본 연구는 3D 메시를 기반으로 '정적' 4D Neural Radiance Field (NeRF)를 생성한 후, 텍스트 프롬프트에 의해 구동되는 이미지-비디오 확산 모델을 사용하여 객체를 애니메이션화합니다. 이는 원래 객체의 정체성을 유지하면서 사용자 제공 텍스트에 따라 맞춤형 애니메이션을 생성할 수 있게 합니다. 또한, 보다 현실적인 움직임을 위해 점진적인 시점 선택 프로토콜과 마스크된 Score Distillation Sampling (SDS) 손실을 도입하여 실제적인 동작을 증진시킵니다.

- **Performance Highlights**: 모델의 평가 결과, 시간적 일관성, 프롬프트 준수성, 시각적 충실도 면에서 기존 방법들을 능가하는 성능을 보였습니다. 우리의 방법은 LPIPS 점수 기준으로 기존 방법에 비해 최대 세 배 향상된 정체성 보존 능력을 달성하였으며, 동적 콘텐츠와 시각적 품질 간의 균형을 효과적으로 유지합니다. 이러한 개선 사항은 불황에 대한 현실적인 4D 장면 생성에 기여하며 동적 풍부함을 증가시킵니다.



### Diff4MMLiTS: Advanced Multimodal Liver Tumor Segmentation via Diffusion-Based Image Synthesis and Alignmen (https://arxiv.org/abs/2412.20418)
- **What's New**: 이번 연구에서는 Diff4MMLiTS라는 혁신적이고 효과적인 다중 모달 간 종양 분할 프레임워크를 소개합니다. 이 프레임워크는 기존의 엄격하게 정렬된 다중 모달 CT 데이터에 의존하지 않고, 미리 등록된 CT 데이터와 생성된 합성 데이터를 활용하여 종양 분할 문제를 해결합니다. 이 방법은 다양한 다중 모달 데이터 세트에 적응 가능하며, 종양 등록의 어려움을 데이터 합성 문제로 변환하는 새로운 파이프라인을 갖추고 있습니다.

- **Technical Details**: Diff4MMLiTS는 세 가지 주요 구성 요소로 이루어져 있습니다: 정상 CT 생성기(Normal CT Generator, NCG) 모듈, 다중 모달 CT 합성기(Multimodal CT Synthesizer, MCS) 모듈, 다중 모달 세그멘터(Multimodal Segmenter, MS) 모듈입니다. 이 구성 요소들은 순차적으로 훈련되며, 각 단계의 출력을 다음 단계의 입력으로 사용합니다. NCG 모듈은 종양 데이터 합성을 위한 기반으로 쌍으로 정렬된 정상 CT 데이터를 생성하고, MCS 모듈은 생성된 이미지를 바탕으로 종양이 포함된 CT 이미지를 합성합니다.

- **Performance Highlights**: Diff4MMLiTS는 다양한 공개 및 내부 데이터 세트에서 실험을 통해 기존의 최첨단 다중 모달 분할 기법보다 우수한 성능을 보여주었습니다. 이 방법은 다중 모달 데이터의 정렬 문제를 해결함으로써 종양 분할 성능을 극대화하는 것을 목표로 하고 있으며, 실험 결과가 이를 입증하고 있습니다. 코드 및 합성 데이터 세트는 공개될 예정입니다.



### EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers (https://arxiv.org/abs/2412.20413)
Comments:
          24 pages, 18 figures

- **What's New**: 이 논문에서는 최신의 흐름 기반 T2I 프레임워크 내에서 개념 삭제(concept erasure)를 다루기 위해 특별히 개발된 최초의 방법인 	extbf{EraseAnything}을 소개합니다. 이 방법은 LoRA 기반 매개변수 조정과 주의 맵 정규화(attention map regularizer)를 활용하여 원하지 않는 활성화를 선택적으로 억제합니다. 또한, 자가 대조 학습(self-contrastive learning) 전략을 제안하여 원하지 않는 개념을 제거하는 과정에서 관련 없는 개념의 성능이 해치지 않도록 보장합니다.

- **Technical Details**: EraseAnything는 개념 삭제를 이원 최적화(bi-level optimization) 문제로 공식화합니다. 이 과정에서 유용한 데이터셋을 활용하여 특정 개념의 활성화를 줄이는 조정을 수행합니다. 첫 번째로는 LoRA를 미세 조정하는 방법, 그리고 주의 맵 정규화를 통해 모델 내부의 상세한 구조를 탐색하여 주의 맵을 개선합니다. 또한, 개념 삭제와 관련된 비유의어 및 관련 없는 개념을 활용하여 모델이 생성하는 주의 맵에 페널티를 부여하는 자가 대조 손실(self-contrastive loss)을 구성합니다.

- **Performance Highlights**: 실험 결과에 따르면, EraseAnything은 최신 T2I 패러다임에서 이전 방법들이 남긴 연구 격차를 성공적으로 메우며 다양한 개념 삭제 작업에서 최첨단 성능을 달성합니다. 이 연구는 새로운 흐름 기반 T2I 구조가 가진 도전 과제를 극복하고, 개념 삭제의 효과성과 안정성을 동시에 만족하는 방법을 제시합니다. 이를 통해 독립적인 개념을 제거하면서도 이미지 생성 품질을 유지하는 데 성공했습니다.



### Open-Sora: Democratizing Efficient Video Production for A (https://arxiv.org/abs/2412.20404)
- **What's New**: 이번 논문은 Open-Sora라는 오픈 소스 비디오 생성 모델을 소개하며, 이는 고품질 비디오 콘텐츠 생성을 위해 설계되었습니다. Open-Sora는 텍스트-이미지 생성, 텍스트-비디오 생성 및 이미지-비디오 생성 등 다양한 시각적 생성 작업을 지원합니다. 특히 Spatial-Temporal Diffusion Transformer (STDiT)라는 효율적인 확산 프레임워크를 도입하여 공간 및 시간 주의를 분리하여 비디오 생성의 유연성을 제공합니다.

- **Technical Details**: Open-Sora는 전량 오픈 소스로 제공되며, 30M개의 비디오 클립과 3M개의 이미지로 구성된 데이터셋을 사용합니다. 이 모델은 15초 길이의 비디오를 최대 720p 해상도로 생성할 수 있으며, 각각의 비디오 클립을 효과적으로 처리하기 위해 PySceneCut를 활용하여 장면을 탐지하고 원시 비디오를 고품질 비디오-텍스트 쌍으로 변환하는 전체 파이프라인을 구축하였습니다. 모델 아키텍처는 Sora의 보고서를 따르며, 3D 오토인코더를 사용하여 비디오를 압축합니다.

- **Performance Highlights**: Open-Sora 모델은 다양한 비디오 생성 작업에서 인상적인 결과를 보여주었으며, 텍스트-비디오 및 이미지-비디오 작업에서의 모션 동력학을 제어할 수 있는 기능이 뛰어납니다. 각 배포 버전(1.0, 1.1, 1.2)에 대한 온라인 보고서도 제공되며, 특히 1.2 버전에서는 시간적 이해 시스템이 개선되었습니다. 이 모델은 AI 콘텐츠 생성 커뮤니티 내에서 혁신과 창의성, 포괄성을 촉진하는 것을 목표로 하고 있습니다.



### Defending Multimodal Backdoored Models by Repulsive Visual Prompt Tuning (https://arxiv.org/abs/2412.20392)
- **What's New**: 최근 연구에서는 CLIP와 같은 다중 모드 대조 학습 모델들이 백도어 공격에 취약하다는 사실이 밝혀졌습니다. 이러한 취약점은 모델이 클래스와 무관한 특징을 과도하게 인코딩하는 데 기인합니다. 이 연구에서는 이를 해결하기 위한 새로운 방어 접근법인 Repulsive Visual Prompt Tuning (RVPT)을 제안합니다. RVPT는 손상된 데이터 없이 소수의 샘플로도 효과적인 방어 기능을 제공하며, 클린 정확도를 유지하면서 불필요한 특징을 제거합니다.

- **Technical Details**: RVPT는 특수하게 설계된 심층 시각 프롬프트 튜닝과 특징을 밀어내는 손실 함수(feature-repelling loss)를 사용하여 클래스와 무관한 특징(CIFs)을 제거합니다. 이 방식은 모델의 시각적 특징이 입력 변화에 대한 민감도를 감소시키도록 설계되어 있습니다. 연구에서는 Perturbation Resistivity (PR)라는 메트릭을 도입하여 모델의 민감도를 정량화하고, PR이 낮을수록 모델은 백도어 공격에 더 취약하다는 사실을 보여주었습니다. RVPT는 CLIP의 0.27%의 파라미터만을 조정하여 기존 최첨단 기법과 비교해 공격 성공률을 크게 감소시키는 효과를 확인하였습니다.

- **Performance Highlights**: RVPT는 다수의 데이터셋과 백도어 공격에 걸쳐 효과적으로 작동하며, CLIP 모델 대비 PR 값을 증가시키고 클래스와 무관한 특징을 줄이는 데 성공했습니다. 이 연구의 실험 결과에 따르면 RVPT는 기존의 공격 방법에 대해 성공률을 67.53%에서 2.76%로 줄였으며, 한 개 클래스당 한 샷만으로도 뛰어난 방어 능력을 발휘하는 것이 입증되었습니다. 따라서, RVPT는 백도어 공격에 대한 강력한 방어 대책으로 자리잡을 것으로 기대됩니다.



### MetricDepth: Enhancing Monocular Depth Estimation with Deep Metric Learning (https://arxiv.org/abs/2412.20390)
- **What's New**: 본 연구는 MetricDepth라는 새로운 방법을 제안합니다. MetricDepth는 깊이 추정(monocular depth estimation)의 성능을 향상시키기 위해 심층 메트릭 학습(deep metric learning)을 통합한 독창적인 방식입니다. 이 방법은 차별 기반 샘플 식별을 통해, 깊이 차이를 기준으로 특징 샘플을 다양한 샘플 유형으로 구분하여 깊이 추정 모델의 기능 정규화를 가능하게 합니다.

- **Technical Details**: MetricDepth에서는 깊이 추정 작업에서 클래스를 기반으로 한 샘플 식별 기법의 한계를 극복하기 위해, 차별 기반 샘플 식별을 설계했습니다. 특징 샘플의 깊이 차이를 기준으로 긍정적 샘플과 부정적 샘플로 구분하며, 이를 통해 깊이 추정의 정규화 과정을 효과적으로 수행할 수 있습니다. 또한, 다중 범위 전략을 도입하여 부정적 샘플의 정규화 효과를 극대화하고, 부정적 샘플을 깊이 차이에 따라 더 세부적인 하위 그룹으로 나누어 다양한 정규화 공식을 적용합니다.

- **Performance Highlights**: 다양한 MDE 데이터셋인 NYU Depth V2와 KITTI에서 MetricDepth의 효과성을 검증하기 위한 실험을 수행하였습니다. 실험 결과, MetricDepth를 통합함으로써 선택된 모델들의 성능이 모두 유의미하게 향상되었음을 보여주었습니다. 이러한 결과는 MetricDepth의 효과성과 다재다능성을 입증합니다.



### PTQ4VM: Post-Training Quantization for Visual Mamba (https://arxiv.org/abs/2412.20386)
Comments:
          Accepted at WACV 2025

- **What's New**: Visual Mamba는 전통적인 Mamba 모델의 비전을 위한 확장으로, 이미지 데이터를 처리하는 데 뛰어난 성능을 보입니다. 그러나 기존 포스트-트레이닝 양자화(PTQ) 방법을 사용했을 때 품질 저하가 발생하는 문제를 발견했습니다. 이를 해결하기 위해 PTQ4VM을 제안하며, Per-Token Static (PTS) 양자화와 Joint Learning of Smoothing Scale and Step Size (JLSS) 두 가지 기술을 기반으로 합니다.

- **Technical Details**: PTQ4VM은 Visual Mamba의 양자화 과정을 개선하는 포스트-트레이닝 양자화 방법입니다. PTS는 토큰별 분산을 처리하도록 설계되었으며, JLSS는 출력 특성 맵에서 최소한의 차이를 유지하며 양자화 매개변수를 최적화합니다. 이 두 가지 방법은 양자화 후에도 모델의 기능을 보존하면서 높은 처리량을 보장하도록 세심하게 설계되었습니다.

- **Performance Highlights**: 실험 결과 PTQ4VM를 적용한 Visual Mamba는 FP16에 비해 정확도 손실이 거의 없고, GPU에서 최대 1.83배의 속도를 향상시키는 성능을 달성했습니다. 이는 다양한 Visual Mamba 백본에 정확성과 효율성을 동시에 제공할 수 있는 가능성을 보여줍니다. PTQ4VM은 기존 모델의 양자화를 15분 이내에 수행할 수 있어 실용성도 뚜렷합니다.



### Breaking Fine-Grained Classification Barriers with Cost-Free Data in Few-Shot Class-Incremental Learning (https://arxiv.org/abs/2412.20383)
Comments:
          29 pages

- **What's New**: 본 논문에서는 기존의 세분화 분류(fine-grained classification) 접근 방식의 한계를 극복하기 위해 새로운 학습 패러다임을 제안합니다. 이 방법은 모델이 표준 학습 단계 이후에도 시스템 운영 중에 발생하는 비용 없는 데이터를 활용할 수 있게 합니다. 이를 통해 기존 데이터로부터의 연관성을 넘어 새로운 데이터를 효과적으로 학습할 수 있는 가능성을 열었습니다.

- **Technical Details**: 제안된 방식인 EXPloring and EXPloiting (EXP2) 전략은 최종 분류 결과를 얻기 전에 클래스 템플릿에 따라 대표적인 추론(inference) 데이터 샘플을 탐색하고 이를 활용하여 분류기를 최적화합니다. 종래의 FSCIL(세분화된 클래스 증가 학습) 프레임워크 내에서 제안되지만, 이 이론은 FSCIL에 국한되지 않고 향후 다양한 설정으로 확장될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 EXP2 방법이 세분화 이미지 분류 문제에 대한 일반적인 효과를 나타내는 것으로 보입니다. 특히, 최신 기술들은 기존 데이터로부터 추가적인 지식을 탐색하고 활용함으로써 동적 데이터 변화에 대한 적응력을 강화할 수 있음을 보여줍니다. 따라서 이 연구는 실세계 시나리오에서 세분화 분류 문제의 깊은 이해와 탐구를 위한 지침을 제공합니다.



### Prot\'eg\'e: Learn and Generate Basic Makeup Styles with Generative Adversarial Networks (GANs) (https://arxiv.org/abs/2412.20381)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문은 디지털 메이크업 애플리케이션의 한계를 극복하기 위해 'Protégé'라는 새로운 메이크업 애플리케이션을 제안합니다. 기존의 시스템들은 수동적이고 제한된 방법으로 메이크업을 추천하거나 이전 이미지를 기반으로 메이크업을 이식하는데 그쳤지만, Protégé는 GANs(Generative Adversarial Networks)를 활용하여 개인화된 메이크업 스타일을 자동으로 생성합니다. 이는 사용자와 전문가의 지식에 의존하지 않고 인튜이티브한 방식으로 메이크업을 만들어낼 수 있는 가능성을 열어줍니다.

- **Technical Details**: Protégé는 기존의 이미지 인페인팅(image inpainting) 기술을 재구성하여 메이크업 생성 기능을 통합합니다. 이 시스템은 기본 메이크업 데이터셋에서 학습하여 개인 얼굴에 맞춘 독창적이고 다양한 메이크업을 생성하는데 초점을 맞추고 있습니다. GANs를 사용함으로써 데이터셋에 특화된 메이크업 스타일을 학습하고 적용할 수 있도록 설계되었습니다. 이로 인해 전문가의 규칙이나 제한된 옵션에 얽매이지 않고, 보다 자연스럽고 개인화된 메이크업을 생성할 수 있습니다.

- **Performance Highlights**: Protégé는 전통적인 메이크업 응용 프로그램과 비교할 때 학습 및 메이크업 생성의 혁신적인 성장 가능성을 보여주며, 이용자에게 필요한 메이크업 스타일을 인지적으로 이해하고 생성하는 능력을 가집니다. 실험을 통해 Protégé는 다양한 메이크업 스타일의 생성 능력을 입증하며, 디지털 메이크업 기술에서의 중요한 발전을 나타냅니다. 이러한 기능들은 소비자들이 개인화된 메이크업 경험을 통해 더욱 향상된 미적 만족을 느낄 수 있도록 돕습니다.



### Tri-Ergon: Fine-grained Video-to-Audio Generation with Multi-modal Conditions and LUFS Contro (https://arxiv.org/abs/2412.20378)
Comments:
          AAAI 2025 Accepted

- **What's New**:  Tri-Ergon은 비디오 장면에 대응하는 사실적인 소리를 생성하기 위해 시각적 비디오 기능을 활용하는 확산 기반(V2A) 모델입니다. 기존의 V2A 모델이 생성된 오디오에 대한 세밀한 제어를 제공하지 못하는 단점을 해결하기 위해, Tri-Ergon은 텍스트, 청각 및 픽셀 수준의 시각적 프롬프트를 포함하여 정교하고 의미 있는 오디오 합성을 가능하게 합니다. 이 모델은 아날로그적 소리 효과(Foley workflows)에서 비디오와 오디오 간의 복잡한 상관관계를 효과적으로 해결하기 위해 LUFS(Loudness Units relative to Full Scale) 임베딩을 도입하였습니다.

- **Technical Details**:  본 연구에서는 비디오 프레임으로부터 세맨틱적으로 관련이 있으며 시간적으로 동기화된 오디오를 생성하는 V2A 작업에 초점을 맞추었습니다. Tri-Ergon은 텍스트, 청각 및 시각 프롬프트를 활용하여 오디오 생성 프로세스를 조정하는 다중 모달 조건을 포함하여 정밀한 음량 제어 메커니즘을 도입합니다. LUFS는 인간의 청각 인식에 밀접하게 연관된 표준화된 메트릭을 제공하여, 오디오 생성 과정 전반에 걸쳐 세밀한 음량 제어를 가능하게 합니다. 이 모델은 44.1 kHz의 고충실도 스테레오 오디오 클립을 생성할 수 있으며, 최대 60초 길이까지 다양하게 생성할 수 있습니다.

- **Performance Highlights**:  Tri-Ergon은 기존의 V2A 방법보다 뛰어난 성능을 보여줍니다. 이전 모델들이 일반적으로 고정된 시간의 모노 오디오를 생성하는 데 그쳤던 반면, Tri-Ergon은 60초까지의 다양한 길이의 스테레오 오디오를 생성할 수 있습니다. 연구 결과, Tri-Ergon은 고충실도의 오디오를 제공하며, 다중 모달 STEM 접근 방식을 통해 생성된 오디오의 품질을 높였습니다. 질적 및 양적 결과 모두에서 기존 최첨단 모델을 초월하는 성능을 입증하였습니다.



### FairDiffusion: Enhancing Equity in Latent Diffusion Models via Fair Bayesian Perturbation (https://arxiv.org/abs/2412.20374)
Comments:
          The data and code are made publicly available at this https URL

- **What's New**: 본 연구에서는 의료 텍스트-이미지 확산 모델의 공정성에 대한 최초의 포괄적인 연구를 제시합니다. 특히 Stable Diffusion 모델을 사용하여 성별, 인종 및 민족에 따라 이미지 생성 품질의 편향을 평가하고, FairDiffusion라는 공정성을 고려한 새로운 모델을 도입하여 공정성을 개선하고자 하였습니다. 또한, FairGenMed라는 첫 번째 데이터셋을 구축하여 의료 생성 모델의 공정성을 연구하는 데 필요한 기초 자료를 제공합니다.

- **Technical Details**: FairDiffusion 모델은 Bayesian Optimization 기반의 접근 방식을 사용하여 각 인구 집단의 샘플을 적절히 변형하여 학습 과정에서 공정성을 확보합니다. 연구에서는 SLO fundus 이미지, 피부병변 이미지 및 흉부 X-ray 등 세 가지 의료 이미징 모달리티에서 FairDiffusion의 효과를 평가하였으며, FID 및 AUC와 같은 지표를 통해 이미지 생성 품질과 임상 특징의 의미적 상관관계 모두에서 개선된 성과를 보여주었습니다.

- **Performance Highlights**: 실험 결과, FairDiffusion은 기존 Stable Diffusion보다 모든 보호 속성에서의 생성 품질 및 의미적 상관관계에서 유의미한 개선을 보였습니다. 예를 들어, 안과 SLO fundus 데이터셋에서 FairDiffusion은 Black 및 Hispanic 집단에 대해 각각 7.84와 11.79라는 FID 개선을 달성했습니다. 이러한 긍정적인 결과는 FairDiffusion이 다양한 의료 이미징 모달리티에서의 공정성 문제를 해결하는 데 효과적임을 입증합니다.



### Differential Evolution Integrated Hybrid Deep Learning Model for Object Detection in Pre-made Dishes (https://arxiv.org/abs/2412.20370)
- **What's New**: 본 논문에서는 사전 제작 요리(pre-made dishes) 산업에서 재료를 선택하고 품질을 평가하기 위한 객체 탐지(object detection) 기술의 중요성을 강조합니다. 기존의 객체 탐지 방법들은 재료의 중첩 또는 가림(overlapping occlusion), 유사성(similarity), 조명 부족(insufficient light) 등으로 인해 한계가 있었습니다. 이러한 문제를 해결하기 위해, 차별 진화 통합 하이브리드 딥러닝(Differential Evolution Integrated Hybrid Deep Learning, DEIHDL) 모델을 제안합니다.

- **Technical Details**: DEIHDL 모델은 세 가지 주요 요소로 구성됩니다. 첫째, 세 가지 YOLO 기반(Object detection에 쓰이는 프레임워크) 및 transformer 기반 기반 모델을 각각 개발하여 재료 탐지의 다양성을 높입니다. 둘째, Differential Evolution을 통해 최적화된 자기 조정 가중치(self-adjusting weights)를 사용하여 세 가지 기본 모델을 통합합니다. 셋째, 가중치 박스 통합(weighted boxes fusion) 전략을 적용하여 통합 시 세 모델의 신뢰도를 평가합니다.

- **Performance Highlights**: DEIHDL은 제안된 접근 방식으로 인해 복잡한 사전 제작 요리 장면에서 높은 정확도로 객체 탐지를 수행합니다. 실제 데이터셋을 통한 광범위한 실험을 통해 DEIHDL 모델이 기본 모델들보다 객체 탐지에서 현저히 향상된 성능을 보인다는 것을 확인했습니다. 이 연구는 사전 제작 요리 산업에서의 객체 탐지 기술의 새로운 방향을 제시합니다.



### Exploring the Magnitude-Shape Plot Framework for Anomaly Detection in Crowded Video Scenes (https://arxiv.org/abs/2412.20363)
Comments:
          21 pages, 4 figures, 10 tables

- **What's New**: 이 연구는 비디오 이상 탐지에서 기능 데이터 분석(Functional Data Analysis) 프레임워크를 탐구하고, Magnitude-Shape (MS) Plot의 적용을 중심으로 합니다. Autoencoder를 사용하여 비정상 프레임에 대해 높은 재구성 오류를 발생시키고, 정상 프레임에서는 낮은 오류를 유지하는 정상 행동 패턴을 학습합니다. MS-Plot은 이 데이터를 통해 이상 탐지를 개선하는 통계적으로 유의미하고 해석 가능한 프레임워크를 제공합니다.

- **Technical Details**: 비디오 이상 탐지의 복잡성은 대량의 공간-시간 데이터(Spatiotemporal data)에서 발생하는 상호작용을 포착해야하는 점에서 비롯됩니다. MS-Plot은 각 프레임의 재구성 오류 매트릭스를 다변량 기능 데이터(Multivariate Functional Data)로 간주하여 두 가지 핵심 차원인 크기(magnitude)와 형태(shape)의 편차를 분석합니다. 이 방식은 이상 탐지의 정확성을 크게 향상시켰습니다.

- **Performance Highlights**: 제안된 방법론은 UCSD Ped2 및 CUHK Avenue라는 두 가지 널리 사용되는 벤치마크 데이터셋에서 평가되었으며, 전통적인 단일변량 기능 감지기들보다 우수한 성능을 보였습니다. MS-Plot 기반 프레임워크는 복잡한 비디오 장면에서 효과적인 이상 탐지를 위한 잠재력을 보여주며, 최근의 여러 최신 방법들과 비교했을 때 성과가 뛰어납니다.



### Deep Learning in Image Classification: Evaluating VGG19's Performance on Complex Visual Data (https://arxiv.org/abs/2412.20345)
- **What's New**: 이번 연구는 VGG19 심층 합성곱 신경망(deep convolutional neural network)을 기반으로 한 폐렴 X-ray 이미지의 자동 분류 방법을 탐구합니다. 기존 모델인 SVM, XGBoost, MLP, ResNet50과의 비교를 통해 폐렴 진단에 대한 적용 효과를 평가하였습니다. 이 연구는 자동화된 의료 이미지 처리 기술의 응용 및 발전을 촉진할 수 있는 기초를 마련합니다.

- **Technical Details**: VGG19 모델은 정확도(accuracy) 92%, AUC(Area Under Curve) 0.95, F1 score 0.90, 재현율(recall rate) 0.87 포인트에서 뛰어난 성능을 보였습니다. 반면, ResNet50은 일부 지표에서 우수하지만 재현율과 F1 score에서는 VGG19보다 미흡한 성능을 보입니다. 전통적인 머신 러닝 모델인 SVM과 XGBoost는 이미지 분류 작업에서 명백히 한계를 드러내며, 복잡한 의료 이미지 분석 작업에서는 상대적으로 저조한 성능을 나타냅니다.

- **Performance Highlights**: 이 연구 결과는 심층 학습(deep learning), 특히 합성곱 신경망(convolutional neural network)이 폐렴 X-ray 이미지 분석과 같은 의료 이미지 분류 작업에서 상당한 장점을 가지고 있음을 보여줍니다. 효율적이고 정확한 자동 진단 지원을 제공하여 폐렴의 조기 발견에 강력한 기술적 지원을 제공합니다. 따라서 이 연구는 폐렴 진단 시스템의 발전을 위한 기초 자료로서 중요한 의미가 있습니다.



### Contrastive Conditional Alignment based on Label Shift Calibration for Imbalanced Domain Adaptation (https://arxiv.org/abs/2412.20337)
Comments:
          accepted by ICPR 2024

- **What's New**: 이 논문에서는 기존의 Unsupervised Domain Adaptation (UDA) 방법들이 covariate shift에 주로 초점을 맞추어, covariate shift와 label shift가 동시에 존재하는 불균형 도메인 적응(IDA) 문제 해결에 한계가 있음을 지적합니다. 새로운 방법인 Contrastive Conditional Alignment based on Label Shift Calibration (CCA-LSC)을 제안하여, label shift를 보정함으로써 기존 UDA 및 IDA 방법보다 뛰어난 성능을 발휘함을 증명세합니다. 특히, CCA-LSC는 domain invariance와 class discriminability를 학습하고, label shift를 기반으로 타겟 샘플의 예측을 조정하여 오류 누적을 방지합니다.

- **Technical Details**: CCA-LSC 방법은 크게 두 가지 단계로 나누어져 있습니다. 첫 단계에서는 domain adversarial learning과 sample-weighted moving average centroid alignment 등을 사용하여 두 도메인의 조건부 분포를 정렬합니다. 두 번째 단계에서는 타겟 도메인의 label distribution을 추정하고, label shift 메트릭스를 기반으로 타겟 샘플의 분류 예측을 보정하여 실제 타겟 데이터 분포와의 일관성을 높입니다. 이러한 과정들은 불균형 도메인 적응 문제를 해결하기 위한 필수적인 요소로 작용합니다.

- **Performance Highlights**: 실험 결과, CCA-LSC는 OfficeHome과 DomainNet 데이터셋에서 label shift와 covariate shift가 존재하는 상황에서도 기존의 UDA 및 IDA 방법을 초월하는 성능을 보였습니다. 특히, CCA-LSC에 의해 확보된 pseudo-label들은 분류기의 출력으로 직접 얻은 pseudo-label들보다 지속적으로 더 우수한 성능을 나타내며, 이는 두 도메인의 정밀한 정렬을 촉진함을 시사합니다. 따라서 CCA-LSC는 불균형 도메인 적응 문제를 해결하는 데 있어 매우 효과적인 전략임을 보여줍니다.



### Dual-Level Precision Edges Guided Multi-View Stereo with Accurate Planarization (https://arxiv.org/abs/2412.20328)
Comments:
          Accepted by AAAI25

- **What's New**: 이 논문에서는 DPE-MVS 방법을 제안하여 저조도 텍스처 영역에서의 재구성 정확도를 향상시킵니다. 이는 이중 수준의 정밀 엣지 정보(dual-level precision edge information)를 도입하여 평면 모델의 견고성을 높이는 방식입니다. 추가로 엣지 정보를 활용하여 전통적인 PatchMatch MVS의 샘플링 전략을 개선하며 적응형 패치 크기 조정 접근법을 제안합니다.

- **Technical Details**: DPE-MVS 기법은 정밀한 엣지(fine edge)와 거친 엣지(coarse edge)의 이중 수준 정보를 기반으로 하여, 저조도 텍스처 영역의 평면 모델 구축을 지원합니다. fine edge는 평면 구조를 제약하는 역할을 하며, coarse edge는 더 넓은 범위의 앵커(anchor)를 선택할 수 있도록 돕습니다. 이러한 기법을 통하여 매칭 비용 계산을 최적화하여 더욱 견고한 매칭을 달성할 수 있습니다.

- **Performance Highlights**: DPE-MVS 방법은 ETH3D 및 Tanks & Temples 벤치마크에서 최첨단 성능을 달성하였으며, ETH3D에서는 모든 기존 방법을 초과하는 성능을 보였습니다. 이를 통해 저조도 텍스처 지역에서의 재구성 성능 향상에 대한 강력한 증거를 제시합니다. 본 연구는 전통적인 MVS 접근 방식을 개선하는 데 기여하며, 실제 애플리케이션에 적용 가능성을 높입니다.



### Motion Transfer-Driven intra-class data augmentation for Finger Vein Recognition (https://arxiv.org/abs/2412.20327)
Comments:
          5 Pages

- **What's New**: 이번 연구에서는 전통적인 데이터 증강 방법의 한계를 극복하기 위해 새로운 모션 전송(motion transfer, MT) 모델을 제안합니다. 이 모델은 실제 손가락 자세와 회전 움직임을 모델링하여 손가락 정맥 이미지 데이터 증강을 수행합니다. 이를 통해 손가락 정맥 인식에서 발생하는 오버피팅(overfitting) 문제를 완화하고, 데이터의 부족 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 모델은 먼저 키 포인트(key point) 감지기를 이용해 원본 및 전이 손가락 정맥 이미지의 주요 점과 자세 맵을 추출합니다. 이후 밀집 모션 모듈(dense motion module)을 사용하여 모션 옵티컬 플로우(optical flow)를 추정하고, 이를 통해 이미지 생성 모듈(image generation module)에서 목표 자세에 맞는 이미지를 생성하도록 합니다. 이러한 과정을 통해 보다 자연스럽고 다양한 손가락 자세를 재현할 수 있습니다.

- **Performance Highlights**: 세 개의 공공 손가락 정맥 데이터베이스에서 수행한 실험 결과, 제안된 모션 전송 모델이 인식 정확도를 효과적으로 향상시킬 수 있음을 보여줍니다. 이는 손가락 정맥 인식 기술의 발전에 기여할 수 있는 중대한 발견으로, 실제 환경에서의 적용 가능성을 높입니다.



### Transformer-Based Contrastive Meta-Learning For Low-Resource Generalizable Activity Recognition (https://arxiv.org/abs/2412.20290)
- **What's New**: 본 논문에서는 사람 활동 인식(HAR) 분야의 일반화를 위한 새로운 접근법인 TACO를 제안합니다. TACO는 transformer 기반의 대조적 메타 학습(contrastive meta-learning) 기법으로, 다양한 사용자 및 시나리오 간의 분포 변화(distribution shifts, DS) 문제를 해결하는 데 중점을 둡니다. 특히, TACO는 훈련 중 가상의 목표 도메인을 합성하여 모델의 일반화 가능성을 명확하게 고려하여 DS를 완화합니다.

- **Technical Details**: 이 연구에서는 HAR의 성능 향상을 위해 transformer의 주의 메커니즘(attention mechanism)을 활용하고, 메타 최적화(meta-optimization) 과정에서 감독된 대조 손실(supervised contrastive loss)을 강화합니다. 데이터 다양성(expanding data diversity)은 세 가지 방법으로 달성되며, 여기에는 회전(rotation), 시간 왜곡(time warping), 난잡성(jittering) 등이 포함됩니다. 이를 통해 TACO는 제한된 훈련 샘플로도 효과적인 도메인 불변 및 클래스 판별 표현을 학습합니다.

- **Performance Highlights**: TACO는 여러 저자원 분포 이동 시나리오에서 평균 4.08%의 정확도 향상을 기록하며, 기존의 도메인 일반화 방법들보다 더 나은 성능을 보입니다. 이는 TACO가 HAR 알고리즘 중 처음으로 감독된 대조 학습(supervised contrastive learning)을 메타 학습에 통합한 것이며, 모델의 세분화된 표현 학습을 강화하게 됩니다. 이러한 성과는 사용자와 환경에 대한 높은 다양성을 반영한 데이터 공간 확장을 통해 얻어진 것입니다.



### Few-shot Algorithm Assuranc (https://arxiv.org/abs/2412.20275)
- **What's New**: 이번 논문에서는 이미지를 분류하는 작업에서 나타나는 왜곡에 대한 모델 보장을 다루고 있습니다. 이미지를 왜곡에 대한 정확도가 정해진 임계값을 초과하는지 여부를 예측하는 새로운 분류기를 제안했습니다. 이 방법은 Level Set Estimation (LSE) 알고리즘에 기반하고 있으며, 적은 샘플로도 학습할 수 있도록 확장되었습니다.

- **Technical Details**: 제안된 방법은 왜곡 수준을 입력으로 받아 모델의 정확도를 예측하는 이진 분류 문제로 구성되어 있습니다. 새로운 Conditional Variational Autoencoder 모델을 사용하여 실 이미지에서 합성 이미지를 생성하고, 두 가지 새로운 손실 함수를 도입하였습니다. 이렇게 생성된 합성 이미지를 통해 더 다양한 왜곡 수준을 학습할 수 있습니다.

- **Performance Highlights**: 다섯 개의 벤치마크 이미지 데이터셋에서 엄청난 실험을 수행하여, 제안된 분류 방법이 강력한 기준선보다 현저하게 더 우수한 성능을 보임을 입증했습니다. 이는 이미지 왜곡에 대한 모델의 보장을 향상시키고, 실제 응용에서의 신뢰도 있는 분류 성능을 확보하는 데 기여할 것입니다.



### Plastic Waste Classification Using Deep Learning: Insights from the WaDaBa Datas (https://arxiv.org/abs/2412.20232)
Comments:
          18 pages, 12 figures

- **What's New**: 이번 연구에서는 플라스틱 폐기물 관리의 효율성을 높이기 위해 딥러닝(deep learning) 기술의 가능성을 탐구하였습니다. 특히, 합성곱 신경망(CNNs) 및 YOLO(You Only Look Once)와 같은 객체 탐지(object detection) 모델을 통해 플라스틱 분류와 재활용 과제를 해결하고자 하였습니다.

- **Technical Details**: 연구는 WaDaBa 데이터셋을 사용하여 YOLO-11m 모델이 98.03%의 정확도와 0.990의 mAP50(mean Average Precision at 50%)를 달성했다고 보고하였습니다. 또한, YOLO-11n은 이와 유사한 성능을 보였으며 mAP50에서는 0.992로 더 나은 결과를 나타냈습니다. 경량 모델인 YOLO-10n은 훈련 속도가 빨랐으나 정확도는 낮았고, MobileNet V2는 97.12%의 정확도를 보였지만 객체 탐지에서는 부족한 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 딥러닝 모델이 플라스틱 폐기물 분류 방식을 혁신할 수 있는 잠재력을 지니고 있음을 강조하고 있습니다. YOLO 모델들이 가장 효과적인 성과를 보였고, 이들 모델의 정확성과 컴퓨팅 효율성을 균형 있게 조절함으로써 폐기물 관리 및 재활용에서 스케일이 가능하고 영향력 있는 솔루션을 창출할 수 있다는 점에서 큰 의미가 있습니다.



### Towards Real-Time 2D Mapping: Harnessing Drones, AI, and Computer Vision for Advanced Insights (https://arxiv.org/abs/2412.20210)
Comments:
          7 pages, 7 figures, 1 table

- **What's New**: 이 프로젝트는 드론 이미지를 머신 러닝(machie learning) 및 컴퓨터 비전(computer vision)과 통합하여 실시간 2D 매핑(real-time 2D mapping) 시스템을 소개합니다. 이 시스템은 서베일런스(surveillance), 정찰(reconnaissance), 목표 추적(target tracking)과 같은 군사 작전을 위한 정확하고 시기적절한 지리적 데이터 제공을 목적으로 합니다. 자동화된 기능 탐지(feature detection), 이미지 매칭(image matching) 및 스티칭(stitching)을 통해 고해상도 지도를 신속하게 생성합니다.

- **Technical Details**: 시스템은 Python으로 구현되었으며, OpenCV를 활용하여 이미지 처리를 수행하고, NumPy를 이용하여 효율적인 계산을 지원합니다. ORB(Oriented FAST and Rotated BRIEF)를 사용하여 기능을 탐지하고, FLANN(Fast Library for Approximate Nearest Neighbors)를 통해 정확한 키포인트 매칭(keypoint matching)을 보장합니다. 동차 변환(homography transformations)을 통해 겹치는 이미지를 정렬하여 실시간으로 왜곡 없는 지도를 생성합니다.

- **Performance Highlights**: 테스트 결과, 이 시스템은 전통적인 방법에 비해 속도와 정확성에서 상당한 개선을 이루었습니다. 다양한 조명 조건 및 거친 지형에서도 뛰어난 성능을 보이며, 이는 우주항공(aerospace) 및 방위(defense) 시나리오에서 매우 효과적입니다. 이 자동화된 접근법은 수동 개입을 제거하여 동적인 환경에서 실시간 업데이트를 가능하게 하여 상황 인식(situational awareness) 및 의사결정(decision-making)을 향상시킵니다.



### Towards Visual Grounding: A Survey (https://arxiv.org/abs/2412.20206)
Comments:
          TPAMI under review. We keep tracing related works at this https URL

- **What's New**: 이 논문에서는 Visual Grounding(시각적 기초)의 발전 역사와 현재의 동향을 체계적으로 정리하고 있습니다. 최근 몇 년간 Grounding 기술이 크게 발전하면서 새로운 개념들과 챌린지가 등장했으며, 이들에 대한 포괄적인 분석을 제공합니다. 두 가지 주요 섹션인 기술적 세부사항과 성능의 하이라이트를 통해 선행 연구들과의 구분을 명확하게 하고 향후 연구 방향의 기초를 마련합니다.

- **Technical Details**: 논문은 Visual Grounding을 정의하고 그 과정에서 인공지능 모델이 언어적 표현과 시각적 요소 간의 본질적 관계를 확립하는 방법을 설명합니다. 또한, 연구에서 사용되는 데이터 유형—이미지, 참조 표현(referring expression), 바운딩 박스(bounding box)—을 나열하고, 이러한 요소들의 부족으로 인해 발생하는 문제들을 논의합니다. 연구는 또한 LSTM과 CNN에서 Transformer 기반의 모델로의 전환과 같은 알고리즘 발전을 다루며, 이러한 변화가 Visual Grounding 연구에 미친 영향을 설명합니다.

- **Performance Highlights**: Visual Grounding의 연구는 데이터를 이용한 실험 설정의 다양성과 데이터셋의 제약으로 인해 여러 도전 과제에 직면해 있습니다. 특히, RefCOCO/+/g 데이터셋은 거의 10년 동안 중요한 평가 기준으로 남아 있었으나, 성능 향상에 한계를 보이고 있습니다. LLMs(대형 언어 모델)의 출현으로 기존 데이터셋의 유용성에 대한 의문이 제기되며, 이는 향후 Visual Grounding 연구의 방향성을 새롭게 설정할 수 있는 기회를 제공합니다.



### Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems (https://arxiv.org/abs/2412.20201)
Comments:
          IEEE TETC-CS (Under review)

- **What's New**: 본 논문은 TCVADS(Two-stage Cross-modal Video Anomaly Detection System)라는 새로운 시스템을 제안합니다. 이 시스템은 weak supervision learning을 활용하여 스마트 시티 모니터링에서 발생하는 이상 현상을 효과적으로 탐지합니다. 특히, 기존의 복잡한 멀티모달 접근 방식의 한계를 극복하고 실시간성과 해석 가능성을 동시에 만족하도록 설계되었습니다.

- **Technical Details**: TCVADS는 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 비디오 프레임에서 특징을 추출한 후, 이를 시간 시계열 분석 모듈에 입력하여 teacher model로 작동시킵니다. 이후, knowledge distillation을 통해 얻어진 통찰력을 기반으로 단순화된 convolutional network(student model)이 이진 분류를 수행합니다.

- **Performance Highlights**: 실험 결과, TCVADS는 기존 방법들보다 모델 성능, 탐지 효율성 및 해석 가능성에서 상당한 개선을 보였습니다. 마지막 단계에서는 CLIP을 활용한 cross-modal contrastive learning을 통해 텍스트와 이미지 간의 관계를 강화하며, 더욱 정교한 분류 결과를 도출합니다. 이러한 결과는 스마트 시티 모니터링 응용 프로그램에 매우 중요한 기여를 합니다.



### Mining Platoon Patterns from Traffic Videos (https://arxiv.org/abs/2412.20177)
Comments:
          This submission is an extended technical report version of a paper currently under revision for the VLDB conference. In accordance with PVLDB guidelines, some sentences in the paper are highlighted in blue to indicate changes made during the revision process, specifically for the benefit of VLDB reviewers

- **What's New**: 이 논문에서는 도시 규모의 비디오 데이터에서 공동 이동 패턴(co-movement patterns)을 발견하는 새로운 접근 방식을 제안합니다. 기존의 패턴 정의는 연속 카메라에서 물체들이 나타나야 하며 정확한 궤적이 필요하다는 강한 가정에 기초하고 있었습니다. 그러나 이로 인해 객체가 올바르게 식별되지 않을 경우 패턴 탐지가 어려워지는 문제를 해결하고자합니다.

- **Technical Details**: 논문에서는 VConvoy로 알려진 기존 패턴으로부터 새로운 완화된 정의인 VPlatoon을 제안합니다. 이 새로운 패턴은 최소 m개의 객체가 각 최소 k개의 카메라를 통해 그리고 최대 d개의 누락된 카메라를 허용하며 이동할 수 있습니다. 이를 통해 객체들이 잠시 공통 경로에서 이탈하더라도 패턴을 검출할 수 있는 가능성을 확보하였습니다.

- **Performance Highlights**: MaxGrowth라는 새로운 효율적인 패턴 추출 프레임워크가 제안되었으며, 이는 후보 패턴의 검증 비용이 없는 특징을 가지고 있습니다. 실험 결과, MaxGrowth는 기존의 알고리즘보다 최대 두 배 더 빠르게 동작하며, 실제 비디오 데이터셋에서도 높은 정확도를 나타내어 궤적 복구에 대한 한계가 있다는 점을 보완하였습니다.



### On dataset transferability in medical image classification (https://arxiv.org/abs/2412.20172)
- **What's New**: 이 논문에서는 의료 영상 분류를 위한 새로운 전이 가능성 추정 방법을 제안합니다. 기존의 전이 가능성 메트릭이 일반 자연 영상 데이터셋에 최적화되어 있었던 반면, 본 연구는 의료 영상 특성에 맞춘 새로운 접근 방식으로, 피처 품질과 기울기를 결합해 전이 성능을 평가합니다. 특히, 이 방식은 의료 영상에서 자주 발생하는 자기 소스 편향을 해결하여 선행 학습 모델을 더 효과적으로 재사용할 수 있게 합니다.

- **Technical Details**: 의료 영상을 위한 전이 가능성 평가는 기존 이미지넷(ImageNet) 방식에서 벗어나, 다른 데이터셋을 소스 모델로 사용할 때의 적합성과 유연성을 함께 고려합니다. 연구팀은 두 가지 새로운 시나리오, 즉 의료 영상 분류를 위한 소스 데이터셋 전이 가능성과 교차 도메인 전이를 통해 그 효과를 검증하였습니다. 또한, 15개의 소스 데이터셋과 9개의 CNN 구조를 사용한 성능 벤치마킹 결과도 제공합니다.

- **Performance Highlights**: 제안된 전이 가능성 메트릭은 의료 영상 분류와 교차 도메인 전이 모두에서 기존 방법들을 초월하는 성능을 보여주었습니다. 특히, 공개된 의료 데이터셋을 활용한 연구 결과는 이미지넷 사전 학습이 아닌 더 적합한 데이터셋을 사용하는 것이 성능 향상에 기여함을 입증하였습니다. 이러한 결과는 향후 의료 영상 전이 가능성 추정 연구를 더욱 촉진하는 기초 자료로 활용될 것입니다.



### Geo-ConvGRU: Geographically Masked Convolutional Gated Recurrent Unit for Bird-Eye View Segmentation (https://arxiv.org/abs/2412.20171)
- **What's New**: 최근의 연구에서 CNN(Convolutional Neural Network)은 다양한 컴퓨터 비전 작업에 큰 영향을 미쳤으나, 긴 거리 종속성을 모델링하는 데 한계를 보여왔습니다. 본 논문에서는 3D CNN의 한계를 지적하고, 이를 개선하기 위한 Geo-ConvGRU라는 모듈을 제안합니다. 이 모듈은 Bird's-Eye View(BEV) 분할 작업에 최적화되어 있으며, 시공간 종속성을 처리하는 데 효과적입니다.

- **Technical Details**: 제안한 Geo-ConvGRU는 ConvGRU(Convolutional Gated Recurrent Unit)를 이용하여 시계열 정보를 처리합니다. 3D CNN 레이어를 ConvGRU로 대체하여 모델의 메모리 요구를 줄이고, 공간 구속을 통해 잡음을 억제하기 위해 지리적 마스크를 통합했습니다. 이를 통해 보다 정확한 예측이 가능하며, 차량의 위치 추적 성능이 향상되었습니다.

- **Performance Highlights**: NuScenes 데이터셋에서 수행한 실험을 통해 Geo-ConvGRU가 기존의 최첨단 방법들보다 BEV 의미 분할, 미래 인스턴스 분할, 인식 맵 예측에서 각각 1.3%, 0.9%, 0.8%의 성능 향상을 이루었다고 보고되었습니다. 이 결과들은 자율 주행 시나리오에서 제안한 방법의 우수성을 입증합니다.



### Conformal Risk Control for Pulmonary Nodule Detection (https://arxiv.org/abs/2412.20167)
- **What's New**: 이 논문은 폐암 선별검사를 위한 폐 결절 탐지 사례 연구를 제공하며, conformal risk control (CRC)라는 불확실성 정량화 기법을 통해 고도 탐지 모델의 기능을 강화합니다. 의료 분야에서 예측 불확실성을 이해하는 것은 의사 결정의 신뢰성과 투명성을 보장하는 데 필수적입니다. CRC는 예측 세트를 생성하여 사용자에게 비정상 제안과 함께 안정적인 통계적 보장을 제공하는 매력적인 도구로 설명됩니다.

- **Technical Details**: CRC는 예측 세트의 크기를 예측 불확실성과 연결하여 불확실성을 신호하는 인간의 경향을 모방합니다. 이는 역학적으로 해석 가능한 문구를 제공하여 비전문가인 방사선의사도 쉽게 이해할 수 있도록 합니다. 우리의 모델은 3명 이상의 방사선의사가 주석을 단 결절에서 평균 91.35%의 민감도를 달성하며, 이는 기존 방사선의사의 성과와 비교했을 때 경쟁력 있는 수치입니다.

- **Performance Highlights**: 우리의 모델은 평균 2.25의 허위 양성을 초래하며, 이는 일반적인 방사선의사보다 높은 민감도를 기록합니다. 예측 세트에서의 합의 구성은 방사선의 사이의 일치 정도에 따라 성능에 영향을 미치는 요소로 작용하며, 결론적으로 방사선의사 간의 합의 수가 낮을수록 예측 세트가 커짐을 알 수 있습니다. 이러한 특성은 안전성이 중요한 의료 환경에서 AI 모델의 배포에 매우 유용한 방향성을 제시합니다.



### StyleAutoEncoder for manipulating image attributes using pre-trained StyleGAN (https://arxiv.org/abs/2412.20164)
- **What's New**: 이 논문에서는 이미지의 속성을 조작하는 데 사용되는 새로운 경량 AutoEncoder 모듈인 StyleAutoEncoder(StyleAE)를 소개합니다. StyleAE는 사전 훈련된 생성 모델에 플러그인으로 작용하며, 제한된 계산 자원으로도 깊은 생성 모델을 효율적으로 훈련할 수 있는 비용 효과적인 솔루션을 제공합니다. StyleGAN과 결합하여 실험한 결과, StyleAE는 최신의 알고리즘과 동일한 수준의 성능을 보여주면서도 보다 간단하고 빠르며 유연한 설계가 가능합니다.

- **Technical Details**: 논문에서는 StyleGAN과 AutoEncoder 아키텍처를 토대로 하는 StyleAE 접근법을 설명합니다. StyleGAN은 고품질 이미지를 생성할 수 있는 최첨단 생성 모델로 두 가지 주요 요소로 구성되어 있습니다. StyleAE는 이러한 구조를 통해 StyleGAN의 잠재 공간을 분리하여 속성을 조절할 수 있는 가능성을 제시하며, AutoEncoder를 활용하여 머지 수 있는 속성을 간편하게 변형할 수 있게 합니다.

- **Performance Highlights**: 실험을 통해 StyleAE가 인간 및 동물의 얼굴 이미지를 포함한 데이터셋에서 기존의 플로우 기반 모델과 동등하거나 더 나은 성능을 나타내며, 속성 조작에 있어 우수한 결과를 보여주었습니다. 또한, StyleAE는 계산적으로 효율적이며 대량의 훈련 데이터를 요구하지 않아서 다양한 응용 프로그램에서 실용적인 사용이 기대됩니다. 이러한 결과는 StyleGAN 및 다른 생성 모델의 잠재 공간 조작 효과를 향상할 수 있는 가능성을 보여줍니다.



### Multi-Modality Driven LoRA for Adverse Condition Depth Estimation (https://arxiv.org/abs/2412.20162)
- **What's New**: 이번 논문에서는 악화된 날씨 조건에서의 깊이 추정 문제를 해결하기 위해 Multi-Modality Driven LoRA (MMD-LoRA)라는 혁신적인 접근 방법을 제안합니다. MMD-LoRA는 저랭크 적응 기법(Low-Rank Adaptation, LoRA)과 대조 학습(Contrastive Learning)을 결합하여 소스 도메인에서 타겟 도메인으로의 효과적인 미세 조정을 지원합니다. 특히, Prompt Driven Domain Alignment (PDDA)와 Visual-Text Consistent Contrastive Learning (VTCCL) 두 가지 핵심 구성 요소를 포함하고 있습니다.

- **Technical Details**: 제안된 PDDA는 텍스트 임베딩을 활용하여 시멘틱적으로 관련된 시각적 표현을 추정하도록 돕습니다. 또한, 임베딩 층에 저랭크 분해 행렬을 통합하여 추가적인 타겟 이미지를 사용하지 않고도 타겟 도메인 시각 특성을 캡처합니다. VTCCL은 CLIP 기반의 텍스트 인코더와 확산 모델 기반의 이미지 인코더 사이의 정합성을 향상시키며, 다양한 날씨 조건 간의 표현을 효과적으로 분리합니다.

- **Performance Highlights**: MMD-LoRA는 nuScenes와 Oxford RobotCar 데이터셋에서 광범위한 실험을 통해 최첨단 성능을 달성하였습니다. 이는 다양한 악화된 환경에 대한 적응성과 효율성을 강조합니다. 따라서, 이 연구는 자율주행 차가 까다로운 환경에서 안전하게 작동하도록 하는 데 중요한 기여를 할 것으로 기대됩니다.



### UniRestorer: Universal Image Restoration via Adaptively Estimating Image Degradation at Proper Granularity (https://arxiv.org/abs/2412.20157)
Comments:
          28 pages, 20 figures

- **What's New**: 최근 모든 이미지 복원(all-in-one image restoration) 분야에서 상당한 발전이 이루어졌습니다. 기존의 방법들은 일반적으로 degradation-agnostic 또는 degradation-aware 두 가지 방식으로 나뉘지만, 이들 각각은 특정 문제점이 존재합니다. 우리는 UniRestorer라는 새로운 접근 방식을 통해 향상된 복원 성능을 제공하고자 합니다.

- **Technical Details**: UniRestorer는 degradation 공간에 대한 계층적 클러스터링을 수행하고, 다중 규모의 mixtures-of-experts (MoE) 복원 모델을 훈련합니다. 이 모델은 degradation과 granularity(세분성) 추정치를 함께 사용하여 적절한 전문가를 선택합니다. 이로 인해 UniRestorer는 degradation-specific restoration을 가능하게 하고, degradation 추정 오류에 대한 로버스트성을 높입니다.

- **Performance Highlights**: 實험 결과 UniRestorer는 최신 all-in-one 방법들에 비해 크게 우수한 성능을 보이며, 특정 단일 작업 모델과의 성능 차이를 줄이는 데 유망한 결과를 나타냈습니다. 이 모델의 코드와 사전 훈련된 모델은 공개될 예정입니다.



### Distilled Transformers with Locally Enhanced Global Representations for Face Forgery Detection (https://arxiv.org/abs/2412.20156)
Comments:
          Accepted by Pattern Recognition

- **What's New**: 최근 딥페이크(Deepfake) 감지 기술은 얼굴 이미지의 진위 여부를 판별하는 데 집중하고 있습니다. 본 연구에서는 기존 CNN 기반 모델들이 지역적인 조작 패턴에 취약한 반면, 변환기 기반 탐지기는 글로벌 의존성을 모델링하는 데 개선되었으나 여전히 지역적 조작 아티팩트를 탐구하는 데 한계가 있다는 점을 강조합니다. 이러한 한계를 극복하기 위해, 본 논문에서는 Distilled Transformer Network (DTN)를 제안하여 로컬과 글로벌 위조 흔적을 동시에 캡처할 수 있도록 합니다.

- **Technical Details**: 제안된 DTN은 Mixture of Expert (MoE) 모듈을 사용하여 다양한 위조 임베딩을 학습하도록 설계되었습니다. 또한 Locally-Enhanced Vision Transformer (LEVT) 모듈을 통해 로컬에서 향상된 글로벌 표현을 학습하여 섬세한 위조 아티팩트를 탐색합니다. Attention collapse 문제를 해결하기 위해 Lightweight Multi-Attention Scaling (MAS) 모듈이 도입되어 다양한 Attention maps의 선택을 통해 모델의 성능을 극대화하고, Deepfake Self-Distillation (DSD) 기법을 통해 풍부한 soft label 정보를 제공합니다.

- **Performance Highlights**: 방대한 실험 결과를 통해, 제안된 방법은 5개의 딥페이크 데이터셋에서 최신의 기술들보다 우수한 성능을 보였습니다. DTN은 다양한 위조 패턴을 포괄적으로 탐지하는 능력을 강화하여, 얼굴 조작의 진위를 보다 잘 판별할 수 있도록 합니다. 나아가, 제안된 MoE와 LEVT 모듈은 위조 흔적 탐지에서의 일반화 가능성을 크게 향상시킵니다.



### DEGSTalk: Decomposed Per-Embedding Gaussian Fields for Hair-Preserving Talking Face Synthesis (https://arxiv.org/abs/2412.20148)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 논문에서는 긴 머리를 가진 인물의 Talking Face 영상 합성 문제를 해결하기 위해 DEGSTalk라는 새로운 방법을 제안합니다. DEGSTalk는 Deformable Pre-Embedding Gaussian Fields(변형 가능한 사전 내장 가우시안 필드)를 사용하여 사실적인 Talking Face를 생성하며, 동적 얼굴 영역과 미세한 표정을 정확히 캡처할 수 있도록 설계되었습니다. 또한, Dynamic Hair-Preserving Portrait Rendering 기술을 적용하여 긴 머리의 움직임을 더욱 사실적으로 재현합니다.

- **Technical Details**: DEGSTalk는 3D Gaussian Splatting(3DGS)에 기반을 두고 있으며, 각 Gaussian 원시 요소는 중심, 스케일링, 회전 쿼터니온, 불투명도, 색상 등으로 정의됩니다. 이 방법은 사전 내장 가우시안의 변형을 통해 동적 얼굴 영역을 정확하게 표현하고, 긴 머리의 복잡한 상호작용을 처리하는 데 필요한 기술을 제공합니다. 즉, 사전 내장 가우시안 원시 요소를 동적으로 조정하여 사실성과 적응력을 강화합니다.

- **Performance Highlights**: DEGSTalk는 기존의 방법들에 비해 현실감과 합성 품질이 개선되었으며, 복잡한 얼굴의 동적 처리와 긴 머리의 보존에서 뛰어난 성능을 보여줍니다. 이 기술은 영화 제작, 가상 현실(VR), 증강 현실(AR) 등 다양한 분야에서 활용될 수 있는 가능성을 제시하고 있습니다. 코드와 자료는 공개될 예정이므로, 더 많은 연구자들이 이 기술을 활용할 수 있기를 기대합니다.



### Cross-Modal Mapping: Eliminating the Modality Gap for Few-Shot Image Classification (https://arxiv.org/abs/2412.20110)
- **What's New**: 현재의 few-shot image classification (소수 샷 이미지 분류) 방법들은 사전 훈련된 Vision-Language Models (비전-언어 모델), 예를 들어 CLIP를 기반으로 성과를 이루어냈습니다. 그러나 이러한 접근법들에는 시각적 및 텍스트적 특성을 직접 클래스 프로토타입(class prototypes)으로 활용한다는 한계가 있습니다. 본 논문에서는 이러한 모달리티 갭(modality gap)을 해결하기 위해 Cross-Modal Mapping (CMM) 방법을 제안하였습니다.

- **Technical Details**: Cross-Modal Mapping (CMM) 방법은 선형 변환(linear transformation)을 사용하여 이미지 특징을 텍스트 특징 공간으로 매핑(mapping)합니다. 이를 통해 두 모달리티가 동일한 특징 공간(feature space) 내에서 비교 가능하게 만듭니다. 또한, 우리는 triplet loss 함수를 도입하여 이미지 특성과 클래스 텍스트 특성 간의 공간적 관계를 최적화하여, 텍스트 특징이 이미지 특성의 클래스 프로토타입 자연스럽게 기능하도록 합니다.

- **Performance Highlights**: 실험 결과는 CMM 기법이 기존 방법들에 비해 평균 3.5% 개선된 성능을 나타내며, 4개의 배포 변화(distribution shift) 기준에서도 경쟁력 있는 성능을 보입니다. 이러한 성과는 pretrained vision-language models에서 모달리티 갭을 없애는 데 있어 Cross-Modal Mapping의 효과를 보여줍니다.



### ST$^3$: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming (https://arxiv.org/abs/2412.20105)
Comments:
          Accepted to AAAI2025

- **What's New**: 본 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 효율적인 추론을 위한 새로운 프레임워크인 Spatial-Temporal Visual Token Trimming ($\textbf{ST}^{3}$)을 제안합니다. MLLM의 주의(attention) 메커니즘에서 발견된 중복 시각 토큰과 부분 주의 계산을 활용하여, 비효율적인 과정을 개선합니다. 이 프레임워크는 기존 사전 학습된 MLLM에 원활하게 통합될 수 있어, 훈련 없이도 성능을 유지하며 추론 속도를 두 배로 증가시킵니다.

- **Technical Details**: 본 연구에서 제안하는 $\textbf{ST}^{3}$는 두 가지 주요 구성 요소로 이루어져 있습니다: Progressive Visual Token Pruning (PVTP)는 각 층에서 무관심한 시각 토큰을 제거하고, Visual Token Annealing (VTA)는 생성된 토큰의 길이에 따라 현재 층에서 유지되는 시각 토큰의 수를 동적으로 조정합니다. 이러한 접근법은 MLLM의 토큰 프루닝(token pruning) 기법과 다르게 동작하여, 문맥에 따라 가장 중요한 정보를 효과적으로 추출합니다. 이는 멀티모달 작업에서 효율성을 높이며, 이전 방법들보다 적은 리소스로 더 나은 성능을 제공합니다.

- **Performance Highlights**: 제안된 방식은 LLaVA 모델에서 50% 이상의 FLOPs 감소를 달성하고, 추론 속도를 2배로 증가시킵니다. 다양한 데이터셋에서 일관된 성능을 유지하며, 메모리 사용량 또한 줄어들어 실시간 응용 프로그램에서도 효과적입니다. 이러한 성능 개선은 훨씬 더 큰 모델에 쉽게 적용될 수 있기 때문에, 향후 멀티모달 AI 발전에 기여할 가능성이 큽니다.



### SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis (https://arxiv.org/abs/2412.20104)
- **What's New**: 이 논문에서는 VR/AR 및 인간 애니메이션에서 중요한 문제인 현실적인 인간-객체 상호작용 모션 합성을 다룹니다. 특히, 여러 인간과 객체가 포함된 다체 구조에서의 모션 동기화를 위한 새로운 방법인 SyncDiff를 제안합니다. 기존 작업들이 제한된 수의 신체 구성에 초점을 둔 반면, SyncDiff는 여러 몸체의 복잡한 상호작용을 처리할 수 있는 일반적인 접근 방식을 제공합니다.

- **Technical Details**: SyncDiff는 단일 확산 모델을 사용해 다체 모션의 결합 분포를 캡처하며, 주파수 도메인 모션 분해 기법을 통해 모션의 정밀도를 높입니다. 이 모델은 데이터 샘플 점수와 정렬 점수를 동시에 추정하며, 시각적 모델 내에서 신체 모션 간의 상호작용을 그래픽적으로 모델링합니다. 이로 인해, 모든 신체의 개인적 모션과 상대적 모션을 포함하는 다차원 표현에서 모션 동기화를 명확하게 처리할 수 있습니다.

- **Performance Highlights**: SyncDiff는 네 가지 데이터 세트에서 내려받은 여러 인체 동작 구성에 대한 실험을 통해 기존의 최첨단 모션 합성 방법들보다 우수한 성능을 입증했습니다. 다양한 설정에서 모션의 조정과 세부적인 동작 충실도를 지속적으로 향상시키며, 다체 상호작용의 정밀한 조화를 가능하게 합니다. 이 연구는 향후 VR/AR 및 로봇 학습 분야에서 방대한 응용 가능성을 제공합니다.



### An archaeological Catalog Collection Method Based on Large Vision-Language Models (https://arxiv.org/abs/2412.20088)
Comments:
          4 pages,4 figures,www source track

- **What's New**: 이번 논문은 고고학적 카탈로그 수집을 위한 새로운 방법을 제안합니다. 특히 Large Vision-Language Models(VLMs)를 기반으로 한 접근 방식을 통해 기존의 기술적 문제를 해결하고자 합니다. 제안된 방법은 문서 위치 파악, 블록 이해 및 블록 일치화의 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: 제안된 방법에서는 첫째, 오픈 세트 객체 탐지 모델을 사용하여 입력 카탈로그 내에서 문서 블록을 위치 파악합니다. 둘째, 이 블록들을 처리하여 특성(attribute)으로 설명합니다. 마지막으로, 외래 키(linkage) 및 이분 그래프 매칭을 기반으로 일치 규칙을 구현하여 각 모달의 데이터를 완전하게 일치시킵니다.

- **Performance Highlights**: 다바구(Dabagou) 및 미아오지구(Miaozigou) 도자기 카탈로그를 사용한 실험을 통해 제안된 방법의 유효성을 입증하였습니다. 비교 실험 결과는 기존 VLM 기반 데이터 수집 방법의 한계를 극복할 수 있는 신뢰할 수 있는 솔루션으로 제시됩니다.



### Enhancing Marine Debris Acoustic Monitoring by Optical Flow-Based Motion Vector Analysis (https://arxiv.org/abs/2412.20085)
Comments:
          8 pages, conference

- **What's New**: 이 논문은 해양 쓰레기 모니터링을 위한 새로운 방법론인 optical flow 기반의 접근법을 제안합니다. 기존의 방법들은 사전 클래스 레이블에 의존했으나, 이 연구는 이러한 종속성을 없애고 해양 쓰레기의 시계열 정보를 활용하여 성능을 향상시키고자 합니다. 이는 해양 생태계에 위협이 되는 플라스틱 오염 문제를 효과적으로 다루기 위한 중요한 발전으로 볼 수 있습니다.

- **Technical Details**: 이 연구에서 사용된 acoustic camera는 고해상도 전방형 소나(FLS)로 알려져 있으며, 해양 환경의 탁한 물에서도 영향을 받지 않기 때문에 효과적으로 작동합니다. 그러나 소나 이미지는 시각적 관점에 따라 목표물의 외관이 변화하고, 신호 대 잡음비(signal-to-noise ratio)가 낮으며, 질감(texture)이 약한 이미지 왜곡 문제가 발생합니다. 이 논문은 이러한 어려움에 대응하기 위한 optical flow 기술을 기반으로 한 새로운 방법론을 제안합니다.

- **Performance Highlights**: 제안된 방법은 순환 수조에서의 실험을 통해 검증되었으며, 해양 쓰레기 모니터링에서의 가능성과 견고성을 입증했습니다. 이 접근법은 해양 쓰레기의 공간적(spatial) 및 시간적(temporal) 분포에 대한 새로운 통찰력을 제공할 가능성을 보여주고 있습니다. 따라서 이 연구는 해양 쓰레기 문제 해결을 위한 실질적인 기여를 할 것으로 기대됩니다.



### STNMamba: Mamba-based Spatial-Temporal Normality Learning for Video Anomaly Detection (https://arxiv.org/abs/2412.20084)
- **What's New**: 이번 논문에서는 Mamba 기반의 경량화된 네트워크 STNMamba를 제안합니다. STNMamba는 다중 스케일 시각 공간 상태 블록(MS-VSSB)과 채널 인식 시각 공간 상태 블록(CA-VSSB)을 이용하여 공간-시간의 정상 패턴을 효과적으로 학습할 수 있도록 설계되었습니다. 특히, 이 연구는 Mamba를 비디오 이상 탐지(VAD) 작업에 적용한 최초의 시도로, 기존 기술들의 계산 비용을 낮추면서 경쟁력 있는 성능을 보였습니다.

- **Technical Details**: STNMamba 네트워크는 두 개의 인코더 아키텍처로 구성되어 있습니다. 공간 인코더는 MS-VSSB를 통해 다중 스케일의 외형 특징을 추출하며, 시간 인코더는 CA-VSSB를 사용하여 중요한 모션 패턴을 포착합니다. 또한, 공간-시간 상호작용 모듈(STIM)을 통해 다양한 레벨의 공간 및 시간 정보를 통합하여 내재된 공간-시간 일관성을 효과적으로 모델링합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에 대한 실험 결과, STNMamba는 적은 매개변수와 낮은 계산 비용으로 경쟁력 있는 성능을 달성했습니다. 또한, 기존 방법들에 비해 공간-시간 패턴의 효과적인 학습을 통해 노멀 패턴과의 간극을 증가시켜 비정상상의 탐지 성능을 더욱 강화했습니다.



### MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing (https://arxiv.org/abs/2412.20082)
- **What's New**: 이번 논문에서는 MambaVO라는 새로운 시각적 오도메트리(Visual Odometry) 시스템이 제안됩니다. 이 시스템은 프레임 간의 매칭 품질을 개선하고 포즈(poses) 추정을 강화하기 위해 Mamba 기반의 모듈을 사용합니다. MambaVO는 안정적인 초기화, Mamba 기반의 시퀀셜 매칭 정제, 그리고 매끄러운 훈련을 통하여 기존의 Deep VO 방법들이 가진 여러 한계를 해결합니다.

- **Technical Details**: MambaVO의 핵심 구성 요소는 세 가지입니다: 첫째, Point-Frame Graph (PFG)는 관찰 관계를 캡처하여 안정적인 초기화 및 시퀀스 기반 매칭 정제를 가능하게 합니다. 둘째, Geometric Initialization Module (GIM)은 반밀집 반경화(semi-dense matching)와 PnP 방법을 사용하여 픽셀 대응을 예측하고 포즈를 초기화합니다. 셋째, Geometric Mamba Module (GMM)은 Mamba 블록을 통해 픽셀 매칭을 정제하며, 역사적 토큰과 융합된 특성을 활용하여 포즈 최적화를 수행합니다.

- **Performance Highlights**: MambaVO와 그 확장 버전인 MambaVO++은 EuRoC, TUM-RGBD, KITTI 및 TartanAir와 같은 여러 벤치마크에서 최첨단 성능을 보이며, 실시간 성능과 낮은 GPU 메모리 소모를 유지합니다. 특히, MambaVO는 기존의 Learning to Optimize 방법들에 비해 정확한 포즈 추정과 신뢰성을 확보하며, 글로벌 최적화를 위한 루프 클로저 모듈을 추가하여 MambaVO++로 확장됩니다.



### On the Compositional Generalization of Multimodal LLMs for Medical Imaging (https://arxiv.org/abs/2412.20070)
- **What's New**: 이번 연구는 Multimodal Large Language Models (MLLMs)이 의료 분야에서 이미지 일반화의 능력을 탐구하는 데 중요한 역할을 하고 있음을 강조합니다. MLLMs의 한계는 불충분한 데이터에 기인하므로, 어떤 종류의 이미지가 MLLMs의 일반화에 유용한지 이해할 필요가 있습니다. 본 연구는 Composition Generalization (CG)를 사용하여 MLLMs가 의료 이미지를 이해할 수 있는 방법을 제안하며, Med-MAT라는 새로운 데이터셋을 개발했습니다.

- **Technical Details**: Med-MAT는 106개의 의료 데이터셋을 조합하여 생성된 데이터셋으로, 각 데이터는 Modality, Anatomical Area, 그리고 Task의 세 가지 요소인 MAT-Triplet으로 명확하게 정의되었습니다. 이 데이터셋은 의료 이미지를 기반으로 CG를 탐구할 수 있는 기회를 제공합니다. 연구자들은 데이터의 관련성을 분석하기 위해 MLLMs의 성능을 비교하고, CG가 다양한 이미지 유형에서 모델의 일반화 성능에 미치는 영향을 조사하였습니다.

- **Performance Highlights**: 실험 결과, MLLMs는 CG를 활용하여 이전에 보지 못한 의료 이미지를 이해할 수 있으며, CG가 멀티태스크 훈련에서 나타나는 일반화의 주요 요인 중 하나라는 점이 확인되었습니다. CG는 데이터 수가 적은 경우에도 유용하게 작용해 다양한 MLLMs 구조에서 일관된 성능을 제공함을 보여주었습니다. 이러한 결과는 의료 영역에서 MLLMs의 성공적인 활용 가능성을 제시합니다.



### MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration (https://arxiv.org/abs/2412.20066)
- **What's New**: 이번 논문에서는 새로운 Mamba 기반 이미지 복원 모델인 MaIR을 제안합니다. MaIR은 Nested S-shaped Scanning strategy (NSS)와 Sequence Shuffle Attention block (SSA)를 통합하여 자연 이미지에서의 지역성(locality)과 연속성(continuity)을 보존하면서 이미지 복원을 수행합니다. 특히 NSS는 스트라이프 기반의 스캐닝 영역을 통해 지역성을 보존하고, S-형 스캐닝 경로를 통해 연속성을 유지합니다.

- **Technical Details**: 이 모델은 크게 두 개의 블록으로 구성됩니다. NSS는 이미지 처리 시 발생하는 지역성과 연속성의 손실을 방지하고, SSA는 주어진 입력의 다양한 시퀀스 간의 의존성을 계산하여 이를 집계하는 역할을 합니다. MaIR의 구조는 세 가지 단계, 즉 얕은 특징 추출 단계, 깊은 특징 추출 단계, 재구성 단계로 나뉘며, 각 단계는 Residual Mamba Group (RMG)으로 구성되어 있습니다.

- **Performance Highlights**: MaIR은 14개의 도전적인 데이터세트에서 40개의 베이스라인을 초과하는 성능을 달성하며, 이미지 초해상도, 노이즈 제거, 블러 제거, 안개 제거 작업에서 최첨단 성능을 기록합니다. 이러한 결과로, MaIR은 자연 이미지의 복원 작업에서 지역성과 연속성을 잘 보존하면서 장거리 의존성을 효과적으로 포착하는 새로운 대안이 되고 있습니다.



### VELoRA: A Low-Rank Adaptation Approach for Efficient RGB-Event based Recognition (https://arxiv.org/abs/2412.20064)
Comments:
          In Peer Review

- **What's New**: 본 논문은 RGB-Event 인식을 위한 새로운 파라미터 효율적인 미세 조정(Parameter Efficient Fine-Tuning, PEFT) 전략인 VELoRA를 제안합니다. 기존의 무거운 모델 미세 조정 방식에서 벗어나 LoRA와 같은 경량 미세 조정 방법을 사용하여 성능과 효율의 균형을 맞추려 합니다. 이를 통해 RGB 및 이벤트 카메라를 활용한 멀티모달 작업의 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: 이 논문은 먼저 RGB 프레임과 이벤트 스트림을 통해 프레임 차이를 추출하고, 이를 통해 모션 정보를 캡처합니다. 제안된 LoRA 튜닝 방법을 통해 RGB와 이벤트 데이터를 구분하여 인코딩하고, 이들 간의 상호작용을 통해 특징을 재구성하여 멀티모달 특징 학습을 수행합니다. 마지막으로, 이렇게 통합된 특징들을 분류하는 네트워크에 입력하여 효율적인 미세 조정을 진행합니다.

- **Performance Highlights**: 본 연구는 여러 이벤트 기반 패턴 인식 벤치마크 데이터 세트에서 VELoRA의 효과를 검증하며, 종합적인 실험 결과를 통해 이 전략이 기존 모델들보다 더 높은 성능을 발휘하는 것을 보여줍니다. 특히, 제안된 모델은 계산 비용과 정확도 간의 보다 나은 균형을 제공하며, 다양한 다운스트림 태스크에 유용하게 활용될 수 있습니다.



### MADiff: Text-Guided Fashion Image Editing with Mask Prediction and Attention-Enhanced Diffusion (https://arxiv.org/abs/2412.20062)
- **What's New**: 이 연구에서는 패션 도메인에서 텍스트 안내 이미지 편집을 위한 새로운 모델 MADiff를 제안합니다. MADiff는 MaskNet과 Attention-Enhanced Diffusion Model의 두 가지 주요 구성 요소로 구성되어 있습니다. MaskNet은 편집 영역을 정확히 식별하기 위해 경량 UNet을 사용하여 전경 영역, DensePose와 대규모 언어 모델로부터 마스크 프롬프트를 입력받아 편집 영역의 마스크를 예측합니다. Attention-Enahnced Diffusion Model은 편집 강도를 강화하기 위해 Noise Map과 Attention Map, MaskNet의 마스크를 결합하여 정제된 Noise Map을 생성합니다.

- **Technical Details**: MADiff 모델의 첫 번째 단계는 MaskNet을 사용하여 입력 이미지와 목표 프롬프트로부터 편집 영역의 마스크를 예측하는 것입니다. 이 과정에서 Graphonomy, DensePose, LLAMA3-8b를 활용하여 전경 영역과 DensePose 맵을 추출합니다. 두 번째 단계는 Attention Processor를 통해 정제된 노이즈 맵을 만들어 편집이 이뤄지도록 하는 것입니다. 이 방식은 가벼운 구조와 공간 주의 레이어를 통해 텍스트 정보를 통합하여 정확한 편집을 가능하게 합니다.

- **Performance Highlights**: Fashion-E 데이터셋에서의 광범위한 실험 결과, 제안된 MADiff 모델이 기존 최첨단 모델들보다 편집 영역 예측 정확도와 편집 강도를 크게 향상시킴을 입증했습니다. 패션 이미지 편집을 위한 기준이 부족한 상황에서, 이 연구는 29380개의 이미지-텍스트 쌍을 포함한 Fashion-E 데이터셋을 구축하였으며, 이로 인해 다양한 패션 편집 작업에서 모델을 평가할 수 있도록 지원합니다.



### AI-based Wearable Vision Assistance System for the Visually Impaired: Integrating Real-Time Object Recognition and Contextual Understanding Using Large Vision-Language Models (https://arxiv.org/abs/2412.20059)
Comments:
          N-A

- **What's New**: 이번 연구에서는 시각 장애인을 위한 혁신적인 착용형 비전 지원 시스템을 소개합니다. 이 시스템은 Raspberry Pi 4 Model B(8GB RAM)에 연결된 모자에 장착된 카메라와 인공지능(AI) 기술을 활용하여 사용자가 실시간으로 피드백을 받을 수 있도록 합니다. 기존의 접근 방식에 비해 맥락적으로 풍부한 환경 정보를 제공하는 것이 특징입니다.

- **Technical Details**: 제안된 시스템은 새로운 사람이나 물체를 인식하는 사용자 친화적인 절차를 갖추고 있으며, 사용자들이 새로운 개체에 대한 데이터를 추가하여 인식 정확성을 시간에 따라 개선할 수 있도록 돕습니다. 또한, 대형 비전 언어 모델(LVLM)을 사용하여 사용자의 환경에 있는 물체에 대한 상세한 설명을 제공합니다. 거리 센서를 통해 사용자가 물체와 충돌할 위험이 있을 때 경고음을 발생시켜 안전성을 높입니다.

- **Performance Highlights**: 포괄적인 평가가 진행되었으며, 제안된 AI 기반 솔루션은 전통적인 지원 기술과 비교하여 유의미한 성과를 보여주었습니다. 하드웨어와 AI를 혁신적으로 결합한 이 시스템은 시각 장애인 커뮤니티가 직면한 주요 문제를 해결하기 위한 보조 기술의 중요한 발전으로 평가됩니다.



### GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting (https://arxiv.org/abs/2412.20056)
Comments:
          11 pages, 2 figures. Code available at this https URL

- **What's New**: GSplatLoc은 3D Gaussian Splatting의 미분 가능한 렌더링 기능을 활용하여 초정밀 카메라 위치 추정을 가능하게 하는 새로운 방법입니다. 이 방법은 기존 3D Gaussian 장면에서 렌더링한 깊이 맵과 관찰된 깊이 이미지 간의 불일치를 최소화하는 그래디언트 기반 최적화 문제로 포즈 추정을 공식화합니다. GSplatLoc은 Replica 데이터셋에서 0.01cm 이내의 평행 이동 오류와 거의 없는 회전 오류를 달성하며 기존 방법보다 월등한 성능을 보입니다.

- **Technical Details**: GSplatLoc은 카메라 포즈의 파생물에 대한 포괄적인 이론적 분석을 기반으로 한 GPU 가속 프레임워크를 제공합니다. 이 방법은 3D Gaussian 장면에 대해 카메라 포즈 추정을 위한 렌더링 과정의 미분 가능한 성질을 완전히 활용하는 새로운 최적화 접근 방식을 제안합니다. 또한 기존의 Gaussian Splatting SLAM 프레임워크나 로컬화에 중점을 둔 다른 딥 러닝 작업에 원활하게 통합될 수 있어 효율적인 장면 표현 및 카메라 포즈 추정을 가능하게 합니다.

- **Performance Highlights**: GSplatLoc은 Replica 및 TUM RGB-D 데이터셋에서 도전적인 실내 환경에서도 견고함을 강조하며, 복잡한 카메라 움직임을 처리할 수 있는 능력을 보여줍니다. 기존 SLAM 기술을 활용하면서도 더욱 향상된 포즈 추정 결과를 보이며 높은 정밀도를 요구하는 애플리케이션에 중요한 영향을 미치는 새로운 벤치마크를 설정합니다. 이 방법은 로봇 공학 및 증강 현실과 같은 정확한 실시간 로컬화를 필요로 하는 응용 분야에서의 활용 가능성을 넓힙니다.



### SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection (https://arxiv.org/abs/2412.20047)
Comments:
          Technical Report

- **What's New**: 최근 비전 모델의 발전에도 불구하고, 적은 예시로부터 학습하는 데 여전히 어려움을 겪고 있습니다. 본 논문에서는 자연적인 장기 꼬리(long-tailed) 분포를 따르는 객체 탐지(object detection) 작업에 중점을 두고, 외부의 대규모 레이블이 있는 데이터베이스에 의존하지 않고도 학습할 수 있는 효율적인 방법을 제안합니다. 저자들은 SimLTD 프레임워크를 제안하며, 이는 사전 학습(pre-training), 전이 학습(transfer learning), 및 미세 조정(fine-tuning)의 세 가지 단계를 포함합니다.

- **Technical Details**: SimLTD 프레임워크는 데이터 풍부한 헤드 클래스(head classes)에 대해 사전 학습를 수행하고, 제한된 데이터가 존재하는 테일 클래스(tail classes)에 대해 전이 학습을 진행하며, 마지막으로 헤드 및 테일 클래스를 모두 포함한 샘플 집합에 대해 미세 조정하는 방식으로 작동합니다. 이 과정에서 라벨이 없는 이미지를 보조 데이터로 활용하여 적은 수의 레이블로도 학습을 강화하게 됩니다. 이 방법은 메타 학습(meta-learning)이나 지식 증류(knowledge distillation)와 같은 복잡한 기법 없이도 구현될 수 있습니다.

- **Performance Highlights**: 제안하는 SimLTD는 LVIS v1 벤치마크에서 기존의 방법들보다 뛰어난 성능을 보여주며 새로운 최첨단 성과를 수립하였습니다. SimLTD는 다양한 백본(backbone) 및 탐지기(detector)와 호환 가능하고, 간단하면서도 직관적인 설계를 가지고 있습니다. 연구자들은 SimLTD가 향후 장기 꼬리 문제를 다루기 위한 강력한 기준선으로 작용할 것으로 기대하고 있습니다.



### Enhancing Diffusion Models for Inverse Problems with Covariance-Aware Posterior Sampling (https://arxiv.org/abs/2412.20045)
- **What's New**: 논문에서는 새로운 방법인 covariance-aware diffusion posterior sampling (CA-DPS)를 제안하여, 기존의 denoising diffusion probabilistic models (DDPMs)을 활용하여 노이즈가 있는 선형 역문제를 해결하는 접근법을 향상시킵니다. 기존 방법들이 개별 모델 훈련을 요구하는 반면, CA-DPS는 추가적인 훈련 없이도 상태를 추정할 수 있는 우수한 방법으로 자리잡습니다. 이를 통해 높은 성능의 신호 복원이 가능하며, 이는 딥러닝 훈련의 복잡성을 덜어줍니다.

- **Technical Details**: CA-DPS의 핵심은 반전 프로세스의 공분산을 공식화한 것입니다. 이 공분산을 기반으로 한 근사식은 기존의 사전 훈련된 DDPM에서 쉽게 구할 수 있습니다. 논문에서는 Tweedie 공식을 사용하여 Conditional Mean을 계산하고, 이를 통해 공분산을 보정함으로써 신뢰할 수 있는 Likelihood 근사를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, CA-DPS는 하이퍼파라미터 조정 없이도 재구성 성능을 유의미하게 향상시키는 것을 보여주었습니다. 이는 기존의 모델들에 비해 노이즈 환경에서도 뛰어난 결과를 기록하며, 이미지, 비디오 및 오디오 합성 분야에서의 가능성을 더욱 확장합니다. 또한, 이 접근 방식은 시간 의존성 문제를 우회하여 효율성을 증가시킵니다.



### DAVE: Diverse Atomic Visual Elements Dataset with High Representation of Vulnerable Road Users in Complex and Unpredictable Environments (https://arxiv.org/abs/2412.20042)
- **What's New**: 이번 연구에서는 DAVE라는 새로운 데이터세트를 소개합니다. DAVE는 복잡하고 예측할 수 없는 환경에서 Vulnerable Road Users (VRUs)의 고비율 표현을 평가하기 위해 설계되었습니다. 기존의 데이터세트들이 서구 트래픽에 집중한 반면, DAVE는 아시아의 복잡한 트래픽 시나리오를 반영하여 실제 도로 안전성을 높이는 데 기여하고자 합니다.

- **Technical Details**: DAVE는 16개의 다양한 행위자 카테고리(동물, 인간, 차량 등)와 16개의 복잡한 행동 유형(컷인, 지그재그 이동, U턴 등)으로 수작업으로 주석이 달린 1300만 개 이상의 바운딩 박스(bboxes)를 포함합니다. 또한, VRUs는 DAVE 인스턴스의 41.13%를 차지하여, Waymo의 23.14%에 비해 상당히 높은 비율을 기록하고 있습니다. 이 데이터세트는 다양한 기상 조건, 시간대, 도로 시나리오 및 교통 밀도를 반영하여 수집되었습니다.

- **Performance Highlights**: DAVE는 비디오 인식 연구의 가치 있는 자료를 제공하며, 기존 방법들이 DAVE에서 평가될 때 성능이 저하된다는 실험 결과를 보였습니다. 이는 DAVE가 미래의 비디오 인식 모델 개발에 중요한 역할을 할 수 있음을 강조합니다. DAVE의 다양한 환경과 복잡한 행동 인식 요구는 비디오 분석 알고리즘의 능력을 향상시키는 데 기여할 것입니다.



### Maintain Plasticity in Long-timescale Continual Test-time Adaptation (https://arxiv.org/abs/2412.20034)
- **What's New**: 논문은 지속적 테스트 시 도메인 적응(CTTA)에서 모델의 플라스틱성(plasticity)이라는 중요한 특성을 탐구합니다. 모델이 비정상적인 환경에서 예측을 지속적으로 조정할 수 있는 능력인 플라스틱성은 장기적인 적응에 필수적입니다. 연구진은 Adaptive Shrink-Restore (ASR)라는 단순하면서도 효과적인 정책을 제안하여 모델의 플라스틱성을 보존하는 방법을 소개합니다.

- **Technical Details**: 플라스틱성의 감소는 label flip의 변화와 밀접한 관계가 있으며, 이를 기반으로 ASR은 가중치 재초기화를 수행합니다. 이 과정에서 적응 간격은 label flipping을 기준으로 결정되며, 현재 모델 가중치를 축소한 후 소스 모델의 가중치로 복원하는 방식입니다. 연구는 여러 CTTA 벤치마크에서 ASR의 효과성을 검증하였습니다.

- **Performance Highlights**: ASR 방식은 기존 CTTA 방법들보다 뛰어난 성능을 보여주었습니다. 실험을 통해, 리셋 정책이 없는 경우 모델의 플라스틱성이 급격히 감소하는 반면, 리셋 정책을 사용할 경우 성능이 안정적으로 유지된다는 것을 알 수 있었습니다. 이는 대부분의 CTTA 방법들이 지속적인 적응 과정에서 플라스틱성을 잃는다는 것을 시사합니다.



### A Robust Adversarial Ensemble with Causal (Feature Interaction) Interpretations for Image Classification (https://arxiv.org/abs/2412.20025)
- **What's New**: 이번 논문에서는 이미지를 분류하기 위해 분류 특성과 생성 모델을 결합한 심층 앙상블 모델을 제안합니다. 이 접근법은 공격에 대한 내구성을 높이기 위해 변분 베이즈(variational Bayes)를 활용하며, 흰 상자(adversarial) 공격에 대한 뛰어난 내구성을 보여줍니다. 향상된 모델 해석 가능성과 공격에 대한 저항성을 평가하기 위해 반사적 메트릭과 특성 상호 작용 기반 메트릭을 사용했습니다.

- **Technical Details**: 모델 아키텍처는 두 가지 주요 구성 요소로 구성됩니다: 하위 수준에서 사전 훈련된 분류적(feature) 네트워크와 상위 수준의 생성적 분류기 네트워크입니다. 생성 모델은 공격 입력 분포를 모델링함으로써 공격에 대한 내구성을 증대시킵니다. 특성 추출을 위한 하위 네트워크는 VGG나 ResNet와 같은 사전 훈련된 CNN 모델을 사용할 수 있습니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 데이터세트에서의 광범위한 실험 결과, 제안된 모델은 공격 성공률을 크게 감소시키며 강력한 공격에 대한 저항성을 입증했습니다. 또한, Tiny-ImageNet 데이터세트를 사용한 예비 결과는 제안된 접근법이 더 복잡한 데이터세트에 대해서도 확장 가능하다는 것을 입증합니다.



### Adversarial Robustness for Deep Learning-based Wildfire Detection Models (https://arxiv.org/abs/2412.20006)
- **What's New**: 이번 연구에서는 DNN 기반의 야생화재 감지 모델의 적대적 강인성을 평가하기 위한 최초의 모델 비구속 프레임워크인 WARP(Wildfire Adversarial Robustness Procedure)를 제안합니다. WARP는 이미지 다양성의 한계를 해결하기 위해 전역 및 지역 적대적 공격 방법을 사용하며, 이를 통해 기존 모델의 단점에 대한 인사이트를 제공합니다. 연구 결과, Transformer 모델이 CNN에 비해 전역 노이즈에 민감하다는 것이 드러났습니다.

- **Technical Details**: WARP는 Gaussian 노이즈와 패치 노이즈 주입을 이용하여 이미지의 적대적 강인성을 평가합니다. 이 모델 비구속 프레임워크는 야생화재 감지에 특화된 적대적 테스트를 설계하며, 일반적인 랜덤 노이즈 대신 연기와 구름의 구분을 고려합니다. 이를 통해 CNN과 Transformer 두 가지 주요 DNN 아키텍처의 특성과 취약점을 폭넓게 분석할 수 있었습니다.

- **Performance Highlights**: WARP의 분석 결과, CNN 및 Transformer 모델 모두 연기 감지 시 구름 이미지 노출에 취약하다는 점이 확인되었습니다. 또한, WARP 분석을 통해 제안된 데이터 증강 전략은 야생화재 감지 모델의 강인성을 개선하는 데 중요한 기초 단계가 될 것입니다. 이러한 발견은 모델의 실제적인 적용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### Learning Adaptive and View-Invariant Vision Transformer with Multi-Teacher Knowledge Distillation for Real-Time UAV Tracking (https://arxiv.org/abs/2412.20002)
- **What's New**: 이번 연구에서는 AVTrack이라는 새로운 어댑티브 컴퓨테이션 프레임워크를 소개하여, 실시간 UAV(무인 항공기) 추적을 위한 transformer 블록을 선택적으로 활성화함으로써 성능과 효율성의 균형을 이루고자 합니다. 또한, 다중 뷰에서 얻은 상호정보(Mutual Information, MI)를 극대화하여 뷰 불변(feature representation) 표현을 학습함으로써, UAV 추적의 정확성을 높입니다. 이 방법은 AVTrack-MD라는 개선된 트래커를 구현하여 여러 트래킹 모델의 출력을 통합하고 정제하는 다중 교수 지식 증류(Multi-teacher Knowledge Distillation) 프레임워크를 포함시켰습니다.

- **Technical Details**: AVTrack의 핵심 요소인 Activation Module(AM)은 transformer 블록에서 활성화 확률을 생성하여 블록의 활성화 여부를 결정하는 기능을 합니다. AM은 효율성을 높이기 위해 색깔 대비와 같은 간단한 기능을 활용하여 비구조적 접근 작업을 줄이고 계산 시간을 단축합니다. 또한, MI 극대화 전략을 통해 두 가지 다른 뷰에서의 특징을 조합하여 뷰 불변 표현을 학습하는 과정이 포함됩니다. 마지막으로, AVTrack-MD의 구조는 줄어든 ViT 블록 수를 가지며, 여러 교수 모델로부터 다양한 지식을 받아들이면서 작은 크기의 학생 모델로 전달합니다.

- **Performance Highlights**: AVTrack-MD는 UAV 추적 벤치마크에서 진행된 광범위한 실험을 통해 AVTrack의 기준 성능을 뛰어넘거나 동등한 성능을 보였으며, 모델 복잡성을 줄여 평균 추적 속도를 17% 증가시키는 결과를 가져왔습니다. 제안된 방법은 학생 모델이 효과적으로 일반화되고 성능을 개선할 수 있도록, 특히 소음이 많은 조건에서 강력한 성능을 발휘했습니다. 이러한 결과는 다양한 시나리오에 적용 가능한 UAV 추적의 새로운 방향성을 제시합니다.



### Comprehensive Review of EEG-to-Output Research: Decoding Neural Signals into Images, Videos, and Audio (https://arxiv.org/abs/2412.19999)
Comments:
          15 pages. Submitted as a conference paper to IntelliSys 2025

- **What's New**: 이 논문은 EEG(뇌파 검사)가 현대 신경과학에서 중요한 역할을 하며, 최근의 머신러닝 및 생성 모델의 발전이 EEG를 활용하여 인지 경험을 재구성하는 데 어떻게 도움을 주는지를 검토한다. 연구자들은 EEG-출력 연구의 최신 동향을 체계적으로 분석하고, 주요 격차와 기회를 강조하며, GANs(Generative Adversarial Networks), VAEs(Variational Autoencoders), Transformers와 같은 고급 모델의 잠재력을 탐구하고 있다. 한편, 연구 분야의 표준화된 데이터셋 필요성과 주제 간 일반화의 중요성을 강조한다.

- **Technical Details**: 이 논문은 PRISMA 가이드라인을 따르며, 1800개의 연구를 시스템적으로 분석하여 최첨단 생성 방법, 평가 메트릭, 데이터 과제를 다룬다. 연구에서는 EEG 신호에서 인지 출력으로의 재구성을 다루는 다양한 생성 모델을 검토하며, EEG 신호 처리의 혁신을 가져온 딥러닝 기술의 발전을 조명한다. EEG2Image와 같은 연구가 EEG 특성을 추출하고 이미지를 합성하는 두 단계 접근 방식을 포함하여 대한 인사이트를 제공한다.

- **Performance Highlights**: EEG를 사용한 생성 모델의 성과는 감각 처리 및 인지 표현에 대한 깊은 통찰을 제공할 수 있는 능력으로 더욱 강화된다. 여기서 중요한 성과로는 EEG2Image가 있으며, 이는 EEG 신호에서 시각적 자극에 해당하는 이미지를 재구성하는 데 뚜렷한 진전을 보여준다. 그러나 여전히 EEG 신호의 고유한 노이즈와 변동성, 그리고 공간적 해상도의 한계가 신뢰성 및 재현 가능성을 저해하고 있어 이러한 도전 과제가 해결되어야 한다.



### FashionFAE: Fine-grained Attributes Enhanced Fashion Vision-Language Pre-training (https://arxiv.org/abs/2412.19997)
- **What's New**: 이번 연구에서는 패션 도메인에 특화된 Fine-grained Attributes Enhanced VLP (FashionFAE)를 제안합니다. 기존의 Vision-Language Pre-training (VLP) 모델이 패션 아이템의 섬세한 특성을 간과하던 문제를 해결하기 위해 텍스트 및 이미지 모달리티에서의 미세한 속성을 동시에 추출하는 방법론을 도입했습니다. 이는 패션의 세부 속성인 재질과 질감을 강조하는 새로운 접근 방식입니다.

- **Technical Details**: FashionFAE의 구조는 Vision Transformer 기반의 이미지 인코더와 BERT 기반의 트랜스포머 모듈로 구성됩니다. 주요 작업인 Attribute-Emphasized Text Prediction (AETP)와 Attribute-Promoted Image Reconstruction (APIR)를 통해 패션 도메인에서의 미세한 정보를 효과적으로 활용할 수 있도록 설계되었습니다. AETP와 APIR 작업이 결합되어 모델의 성능을 극대화하며, 패션 데이터 세트의 다양한 속성을 포괄적으로 분석할 수 있게 만듭니다.

- **Performance Highlights**: 실험 결과, FashionFAE는 State-Of-The-Art (SOTA) 방법들에 비해 2.9% 및 5.2%의 향상을 기록하며 retrieval 성능에서 우수성을 입증했습니다. 또한 인식 작업에서 평균 1.6%의 성능 개선을 달성했습니다. 이러한 성과는 Fashion Domain에서의 모델 적용 가능성을 더욱 높이고, 패션 관련 데이터의 분석과 활용을 촉진할 것입니다.



### An Ordinary Differential Equation Sampler with Stochastic Start for Diffusion Bridge Models (https://arxiv.org/abs/2412.19992)
Comments:
          9 pages, 5 figures, This work has been submitted to the IEEE for possible publication

- **What's New**: 이 연구는 Diffusion bridge 모델의 느린 추론 속도 문제를 해결하기 위해 ODE Sampler with a Stochastic Start (ODES3)를 제안합니다. 기존의 Stochastic Differential Equation (SDE) 샘플러에 비해 고차 Ordinary Differential Equation (ODE) 솔버를 사용하여 초기 이미지를 부드럽게 변환하는 방법에 중점을 두고 있습니다. 이 접근법은 기존의 모델들과 호환되며 추가적인 학습 없이도 뛰어난 성능을 발휘합니다.

- **Technical Details**: ODS3는 먼저 손상된 이미지에서 중간 표현으로의 전환을 위해 posterior sampling을 수행하고, 이후 Heun의 2차 솔버를 사용하여 확률 흐름 ODE(PF-ODE)를 해결합니다. 이 과정은 손상된 이미지에서 생성 경로로의 부드러운 전환을 보장하며, discretization 오류를 줄입니다. 이 방법은 사전 학습된 Diffusion bridge 모델과 완벽히 호환되며, 추가적인 훈련이 필요하지 않습니다.

- **Performance Highlights**: 이 샘플러는 슈퍼 해상도, JPEG 복원 및 이미지 번역 태스크에서 기존 최첨단 방법들보다 높은 시각적 품질과 Frechet Inception Distance (FID) 점수를 기록했습니다. 여러 데이터셋에서 수행된 실험 결과, 제안된 방법은 기존 Diffusion bridge 모델의 원래 샘플러보다 성능이 향상된 것으로 나타났습니다.



### MAKIMA: Tuning-free Multi-Attribute Open-domain Video Editing via Mask-Guided Attention Modulation (https://arxiv.org/abs/2412.19978)
- **What's New**: MAKIMA는 기존의 텍스트-투-이미지(text-to-image) 모델을 기반으로 하여 튜닝(tuning) 없이 다중 속성 비디오 편집을 가능하게 하는 혁신적인 프레임워크이다. 이 접근법은 비디오의 구조와 외관 정보를 보존하면서 마스크 기반의 주의(modulation)를 통해 세밀한 편집을 지원한다. 또, 특징 전파(feature propagation) 메커니즘을 도입하여 계산 효율성을 높였다.

- **Technical Details**: MAKIMA는 Mutual Spatial-Temporal Self-Attention 기법을 적용하여 편집된 객체의 주의 분배 문제를 해결한다. 특히 Mask-guided Attention Modulation을 통해 관련 지역의 주의력을 강화하고 비관련 지역의 방해를 억제함으로써 보다 정밀한 편집을 수행한다. 이러한 구조 덕분에 비디오 편집의 품질과 효율성을 동시에 향상시킬 수 있다.

- **Performance Highlights**: 실험 결과, MAKIMA는 기존의 다중 속성 비디오 편집 방법 대비 우수한 편집 정확성과 시간적 일관성을 나타냈다. 특히, 연산 효율성을 유지하면서도, 원하는 속성을 각각 정확히 반영하여 편집할 수 있는 능력을 입증하였다. 이러한 성과는 MAKIAM의 핵심 기술들이 효과적으로 작용했음을 보여준다.



### DepthMamba with Adaptive Fusion (https://arxiv.org/abs/2412.19964)
- **What's New**: 이 연구에서는 다중 시점(depth estimation) 시스템이 실제 환경에서 자주 발생하는 노이즈가 있는 카메라 포즈를 다루는 새로운 견고성(robustness) 벤치마크를 제안합니다. 기존의 다중 시점 시스템이 이상적인 카메라 포즈에 의존하고 있다는 점을 지적하고, 이에 대응하는 새로운 방법론을 소개합니다.

- **Technical Details**: 제안된 네트워크 아키텍처는 두 개의 브랜치 채널을 결합하여 단일 시점(single-view)과 다중 시점(multi-view)에서의 깊이 추정 결과를 융합합니다. 특히 mamba라는 특징 추출 백본(backbone)을 도입하고, 두 브랜치 간의 가장 강력한 추정 결과를 적응적으로 선택하는 attention 기반의 융합 방법을 제안했습니다.

- **Performance Highlights**: 제안된 방법은 동적 객체 및 텍스처가 없는 지역을 포함한 도전적인 장면에서 우수한 성능을 보여줍니다. 아블레이션 연구(ablation studies) 결과는 백본과 융합 방법의 효과를 입증하며, KITTI 및 DDAD와 같은 도전적인 벤치마크에서 기존의 최첨단(state-of-the-art) 방법들과 비교해 경쟁력 있는 성능을 달성했습니다.



### ErgoChat: a Visual Query System for the Ergonomic Risk Assessment of Construction Workers (https://arxiv.org/abs/2412.19954)
Comments:
          32 pages, 8 figures

- **What's New**: 이 연구는 건설 근로자의 자세 관련 인체공학적 위험을 평가하기 위해 인터랙티브한 비주얼 쿼리 시스템을 도입하였습니다. 기존의 전통적인 인체공학적 위험 평가(ERA) 방법은 상호작용 피드백을 제공하지 않지만, 새로운 시스템은 이미지 입력을 기반으로 질문에 답하고 텍스트 설명을 생성할 수 있는 기능을 갖추고 있습니다. 이 연구는 또한 이러한 방법론을 훈련하고 테스트할 수 있는 데이터셋을 제안합니다.

- **Technical Details**: 이 시스템은 비주얼 질문 답변(VQA) 기능과 이미지 캡셔닝(IC) 기능을 포함하고 있습니다. VQA는 근로자가 노출된 자세 관련 인체공학적 위험에 대한 비주얼 쿼리에 응답하며, IC는 이미지에서 이러한 위험에 대한 텍스트 설명을 생성합니다. 체계적인 테스트 결과, VQA 기능은 96.5%의 정확도를 달성하였고, IC 성능 평가에서는 아홉 가지 지표와 인간 전문가의 평가를 통해 새로운 접근법이 일반 데이터셋만으로 훈련된 동일 아키텍처의 방법보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 연구 결과, 비주얼 질문 답변 기능이 96.5%의 높은 정확도를 기록했으며, 이미지 캡셔닝 측면에서도 새로운 시스템이 기존 방법보다 우수한 성능을 발휘했습니다. 이는 AI 기술을 활용한 인터랙티브한 인체공학적 위험 평가(ERA)의 새로운 방향성을 제시하며, 향후 연구 및 개발에 큰 기여를 할 것으로 기대됩니다.



### Zero-shot Hazard Identification in Autonomous Driving: A Case Study on the COOOL Benchmark (https://arxiv.org/abs/2412.19944)
- **What's New**: 이 논문은 자율주행에서 Out-Of-Label 위험 요소를 감지하고 분류하기 위한 COOOL 대회에 제출한 내용을 소개합니다. 우리는 (i) 운전자의 반응 감지, (ii) 위험 물체 식별, (iii) 위험 설명 작업을 포함하는 세 가지 핵심 작업을 통합하는 접근 방식을 제안합니다. 마지막으로, 새로운 데이터셋과 메트릭스를 활용하여 머신러닝 영역에서의 성능을 평가하고, 32개 팀 중 2위를 기록했습니다.

- **Technical Details**: 운전자의 반응 감지를 위한 방법으로는 커널 기반의 변화점 감지와 광학 흐름 동역학을 사용하여 동작 패턴을 분석했습니다. 위험 식별에서는 사전 훈련된 ViT 모델을 활용한 객체 분류와 근접 기반 전략을 결합했습니다. 위험 설명 작업에서는 맞춤형 프롬프트를 사용해 MOLMO 비전-언어 모델을 활용하여 낮은 해상도의 희귀한 위험에 대한 정확하고 문맥에 맞는 설명을 생성했습니다.

- **Performance Highlights**: 제안된 파이프라인은 기본 모델을 상당히 초월하여 상대 오류를 33% 줄였습니다. 이러한 결과는 자율주행 기술의 안전성을 높이기 위한 작업의 일환으로 중요한 의미를 갖습니다. 또한, COOOL 대회에서 제공하는 종합적인 평가 메트릭스를 통해 실제 시나리오에서의 성능을 효과적으로 반영하였습니다.



### Not all Views are Created Equal: Analyzing Viewpoint Instabilities in Vision Foundation Models (https://arxiv.org/abs/2412.19920)
Comments:
          8 pages + 3 pages of references. 8 figures, 3 tables

- **What's New**: 본 연구는 기초 모델(foundational models)의 시점 안정성(viewpoint stability)을 분석하고, 시점 변화에 대한 감도를 정의하여 3D 추론 과제에서 일반화 갭을 발생시키는 주요 변화를 설명합니다. 특히 사고 시점(accidental viewpoints)에서 물체의 실제 3D 구조가 가려지는 상황을 고려하여 기초 모델의 반응을 조사합니다. 이 연구를 통해, 우리는 이러한 기초 모델들이 사고 시점을 지속적으로 인코딩하나, OOD(out-of-distribution) 시점에 대한 해석은 내재된 편향에 따라 달라진다는 것을 밝혀냈습니다.

- **Technical Details**: 연구에서는 기초 모델의 출력 피처를 통해 시점 불안정성을 인식하고 분류하는 방법론을 제시하며, 이는 외부 이미지 접근 없이 진행됩니다. 이를 통해 불확실한 응답 플래그 설정이 가능해지며, 프라이버시 센시티브 애플리케이션에 유리합니다. 아울러 사고 시점과 다른 불안정한 시점을 피처를 통해 분류하는 방법도 제안합니다.

- **Performance Highlights**: 정량적 및 정성적 평가를 통해 모델의 시점 불안정성 영향과 모델 강인성을 저해하는 특정 조건을 식별합니다. 실험 결과, 기초 모델은 사고 불안정한 시점을 인코딩하는 데 있어 일관성을 보였으나 OOD 시점 해석은 모델 간 변동이 있었습니다. 이를 통해 다양한 시청 조건에서 안정적인 성능을 지원하기 위해 강력한 피처 표현의 필요성을 강조하고 있습니다.



### Char-SAM: Turning Segment Anything Model into Scene Text Segmentation Annotator with Character-level Visual Prompts (https://arxiv.org/abs/2412.19917)
- **What's New**: 이 논문에서는 Char-SAM이라는 새로운 자동 주석 파이프라인을 도입하여 Segment Anything Model (SAM)을 저비용의 문자 수준 주석 도구로 변환합니다. Char-SAM은 기존의 텍스트 검출 데이터셋을 이용하여 더 세분된 문자 수준의 주석을 생성하는 Character Bounding-box Refinement (CBR) 모듈과 문자 글리프 정보를 사용하는 Character Glyph Refinement (CGR) 모듈로 구성되어 있습니다. 이러한 접근 방식은 SAM의 bbox-to-mask 기능을 완전히 활용하여 고품질의 장면 텍스트 분할 주석을 자동으로 생성합니다.

- **Technical Details**: Char-SAM은 주어진 이미지와 단어 단위의 바운딩 박스를 입력으로 받아 픽셀 수준의 주석으로 정제하는 작업을 수행합니다. CBR 모듈은 세부적인 문자 수준의 바운딩 박스를 생성하고, CGR 모듈은 입력된 문자 범주에 해당하는 글리프 정보를 사용하여 긍정적 및 부정적 포인트 프롬프트를 생성합니다. 이 과정에서 SAM의 세분화 기능을 통해 텍스트에 대한 보다 정확한 세분화 마스크를 생성하는 것을 목표로 합니다.

- **Performance Highlights**: 광범위한 실험 결과, Char-SAM은 TextSeg 데이터셋에서 경쟁력 있는 성능을 보였습니다. 특히, Char-SAM은 훈련이 필요 없는 구조로서, 실제 세계 데이터셋인 COCO-Text 및 MLT17에서 고품질의 문자 수준 텍스트 세그멘테이션 데이터셋을 생성했습니다. 이러한 성능은 실제 애플리케이션에서의 사용 가능성을 크게 높입니다.



### Leveraging Scene Geometry and Depth Information for Robust Image Deraining (https://arxiv.org/abs/2412.19913)
Comments:
          12 pages, 5 figures, 10 tables

- **What's New**: 이 연구는 다중 네트워크를 통합한 새로운 학습 프레임워크를 제안합니다. 기존의 단일 네트워크 접근 방식을 뛰어넘어, 레인 디렛(deraining)과 깊이 정보(depth information)를 동시에 처리하는 두 개의 서포트 네트워크를 포함하여 더 효과적으로 이미지 처리를 수행합니다. 이로 인해 자율주행 차량의 물체 탐지(object detection) 성능을 개선할 수 있습니다.

- **Technical Details**: 이 방법은 Derain AutoEncoder(DerainAE) 모델을 활용하여 레인 아티펙트를 효과적으로 처리합니다. 또한 깊이 네트워크(DepthNet)를 통합하여 이미지 간의 중요한 구조 정보를 유지하며, 레인과 클리어(clear) 이미지 간의 일관된 특징을 잡아냅니다. 두 가지 형태의 감독(supervision)을 통해 모델은 레인을 제거하면서도 장면의 내재적 특징을 보존합니다.

- **Performance Highlights**: 다양한 야외 데이터셋에 대한 광범위한 실험을 통해, 이 방법은 비 오는 날의 아티펙트를 효과적으로 제거하면서도 중요한 이미지 세부사항을 보존하는 성능을 입증했습니다. 특히, 물체 탐지 작업에서 뛰어난 성능을 보였습니다. 이러한 결과는 기상 변화가 심한 환경에서도 영상 기반 자율 주행의 신뢰성과 기능성을 향상시키는 데 기여할 것으로 기대됩니다.



### YOLO-MST: Multiscale deep learning method for infrared small target detection based on super-resolution and YOLO (https://arxiv.org/abs/2412.19878)
- **What's New**: 이 논문은 군사 및 다양한 응용 분야의 요구를 충족하기 위해 저위조 및 고정밀의 적외선 소형 목표 탐지 알고리즘 개발에 집중하고 있습니다. 기존 딥러닝 방법의 한계를 극복하기 위해, 이미지 초해상도 기술 및 다중 스케일 관측을 결합한 새로운 딥러닝 네트워크 YOLO-MST를 제안합니다. 이는 복잡한 배경에서 소형 목표의 탐지를 개선하기 위한 최첨단 접근법입니다.

- **Technical Details**: YOLO-MST 네트워크는 YOLOv5 모델을 기반으로 하며, 기존 SPPF 모듈을 MSFA 모듈로 대체하여 다중 스케일의 특성 정보를 효과적으로 추출합니다. 더불어, 네트워크의 목 부분에서 대형 목표 출력 부분을 삭제하고, 소형 목표 탐지에 더 집중할 수 있도록 구조를 최적화했습니다. 마지막으로, 예측 헤드에 동적 특징 융합을 수행하는 DyHead 탐지 헤드를 추가하여 고난이도 탐지 문제를 해결합니다.

- **Performance Highlights**: 제안된 방법은 SIRST 및 IRIS 두 공개 데이터셋에서 각각 96.4% 및 99.5%의 mAP@0.5 탐지율을 기록하였습니다. YOLO-MST는 기존의 SOTA(target detection methods)보다 더 효과적으로 탐지의 누락, 위조 및 저정밀 문제를 해결하는 성능을 보여주었습니다.



### Image Classification with Deep Reinforcement Active Learning (https://arxiv.org/abs/2412.19877)
- **What's New**: 최근 딥러닝 연구에서 이미지 분류 작업의 성능이 저조하다는 점이 강조되며, 특히 라벨이 부여된 데이터의 부족이 문제로 지적되고 있습니다. 본 연구에서는 Markov Decision Process (MDP)를 기반으로 한 새로운 적응형 액티브 러닝 방법인 Deep Reinforcement Active Learning (DRAL)을 제안하였습니다. 이 방법은 액티브 러닝과 딥 강화 학습을 결합하여, 샘플 선택 전략을 동적으로 조정할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: DRAL 모델은 액티브 러닝 문제를 MDP로 설정하여, 현재 상태에서 가장 정보성이 높은 샘플을 선택합니다. 먼저, 불확실성 기준을 통해 샘플을 정렬한 다음, 최상위 샘플들을 선택하여 액터 네트워크에 피드합니다. 이 액터는 각 샘플에 대해 라벨링할지 말지를 결정하며, 선택된 샘플의 효과는 분류기의 정확도 변화로 평가됩니다. 이런 방식으로, DRAL은 고전적인 핸드크래프트(손으로 제작된) 전략보다 효율적인 동적 선택 전략을 제공합니다.

- **Performance Highlights**: 세 가지 이미지 분류 벤치마크에서 광범위한 실험을 통해, 제안된 DRAL 방법이 기존의 여러 액티브 러닝 전략들보다 우수한 성능을 나타났습니다. 특히, 새로운 샘플 선택 전략이 불확실성 기준에 기반하여 분류 성능을 향상시켰다는 점이 인상적입니다. 따라서, DRAL은 제한된 라벨링 예산으로 최적의 학습 모델을 구축하는 데 기여할 가능성을 보이고 있습니다.



### Neighbor Does Matter: Density-Aware Contrastive Learning for Medical Semi-supervised Segmentation (https://arxiv.org/abs/2412.19871)
- **What's New**: 이 논문에서는 다기관 반지도 세분화(semisupervised segmentation)에서 부족한 레이블과 부드러운 조직의 낮은 대조를 해결하기 위해 Density-Aware Contrastive Learning (DACL) 방안을 제안합니다. 기존의 반지도 방법들이 개별 데이터 샘플에 의존하는 것을 벗어나, 특성 공간(feature space)의 기하학적 구조를 활용하여 인접 정보를 추출합니다. 특히, 고밀도 긍정 표본을 사용하여 군집 중심을 근사하는 새로운 접근법을 통해 클래스 내 응집도를 높이려는 목표를 설정하고 있습니다.

- **Technical Details**: DACL 전략은 레이블이 지정된 샘플과 레이블이 없는 샘플을 활용하여 밀도 기반 이웃 그래프(density-aware neighbor graphs)를 구축합니다. 이 방법은 로컬 밀도를 기반으로 희소한 영역을 찾아내고, 최소의 대조 손실(contrastive loss)을 유지함으로써 클래스 내 밀집도를 증가시키는 장점이 있습니다. 또한, Label-guided co-training과 밀도 기반 기하학적 규제를 결합하여 레이블이 없는 데이터에 대한 감독 신호를 보완하여 제공합니다.

- **Performance Highlights**: Multi-Organ Segmentation Challenge 데이터셋을 기반으로 한 실험에서, 제안된 DACL 방법이 최신 최첨단 방식들을 능가하는 성능을 달성했습니다. 이는 반지도 다기관 세분화 문제 해결을 위한 새로운 접근법으로, 데이터 부족과 피처 희소성 문제를 해결할 효능을 입증했습니다. 셀프 학습(self-supervised learning) 방법들과의 비교에서도 현저한 성과를 보여 주목을 받고 있습니다.



### Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales (https://arxiv.org/abs/2412.19867)
- **What's New**: 이번 연구에서는 diffusion 모델의 quantization(양자화)을 개선하기 위해 group-wise quantization(그룹 단위 양자화)의 영향을 조사합니다. 기존의 coarser-grained post-training quantization(후속 훈련 양자화) 기법은 품질 손실이 커서 대규모 모델에 적합하지 않았습니다. 이 연구는 Winograd convolution(윈로가드 합성곱)을 활용한 8비트 fully quantized(전량자화된) 모델이 품질 손실을 최소화하며, 기존 기법에 비해 더 나은 성능을 제공할 수 있음을 입증합니다.

- **Technical Details**: 이 연구에서는 group-wise quantization을 통해 diffusion 모델의 가중치 및 활성화를 양자화하여 이미지 생성 품질을 유지하는 데 집중합니다. Winograd 변환 매트릭스의 scale 파라미터만 조정하여 드문 값의 범위를 줄이는 방법을 제안, 이는 데이터 특정 훈련 데이터 없이 수행할 수 있습니다. 우리의 연구는 학습 데이터에 의존하지 않는 일반화 성능을 보장하므로, foundation 모델에 대한 전이 가능한 성능을 제공합니다.

- **Performance Highlights**: Winograd convolutions가 적용된 우리 8비트 fully quantized diffusion 모델은 다른 최첨단 방법들과 비교하여 ImageNet 데이터셋에서 ResNet-34로 수행했을 때 정확성이 2.56% 향상되었습니다. 또한, 최적화된 커널이 표준 합성곱에 비해 31.3%의 실행 성능 개선을 이루어냈으며, 이는 실제 edge나 모바일 장치에서 diffusion 모델을 효과적으로 배포하는 데 기여할 수 있습니다.



### UniAvatar: Taming Lifelike Audio-Driven Talking Head Generation with Comprehensive Motion and Lighting Contro (https://arxiv.org/abs/2412.19860)
- **What's New**: 최근 오디오 입력을 사용한 초상 이미지 애니메이션 생성이 주목받고 있습니다. UniAvatar라는 새로운 기법이 다양한 모션과 조명 조건을 광범위하게 제어할 수 있도록 설계되었습니다. 이 방법은 FLAME 모델을 활용하여 3D 모션의 세부 정보를 유지하면서도 정밀한 픽셀 단위의 제어를 가능하게 합니다. 또한, 모션뿐만 아니라 전반적인 조명 제어도 지원하여 다채로운 조명 조건에서의 생성이 가능합니다.

- **Technical Details**: UniAvatar는 모션 신호와 조명 조건을 독립적으로 제어하기 위해 두 가지 신호를 활용합니다. FLAME 모델을 통해 여러 모션 표현을 단일 이미지에 통합하고, 3D Motion 모듈을 통해 모션 정보를 추출하여 픽셀 단위의 제어를 지원합니다. 조명 제어를 위해 Illumination-aware Rendering 기술을 사용하여 전 세계 조명 조건을 생성하고, Masked-Cross-Source Sampling 전략을 통해 조명의 영향을 극대화합니다.

- **Performance Highlights**: 방법론적 실험을 통해 UniAvatar는 기존의 방법들보다 넓은 범위의 모션 제어와 조명 제어에서 성능이 우수함을 입증했습니다. 본 연구는 얼굴 애니메이션 분야에서 처음으로 다양한 모션과 조명 환경을 동시에 제어할 수 있는 방법을 제안하며, 두 가지 새로운 데이터셋인 DH-FaceDrasMvVid-100과 DH-FaceReliVid-200을 공개할 예정입니다. 이 데이터셋은 발화 중의 주요 헤드 움직임과 다양한 조명 상황을 담고 있습니다.



### Fusion of Deep Learning and GIS for Advanced Remote Sensing Image Analysis (https://arxiv.org/abs/2412.19856)
- **What's New**: 이번 논문은 원격 감지 이미지 분석을 위한 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 Convolutional Neural Networks (CNNs)와 Long Short-Term Memory (LSTM) 네트워크를 Geographic Information Systems (GIS)와 결합하여 spatial data analysis의 정확성과 효율성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 모델 파라미터를 미세 조정하기 위해 Particle Swarm Optimization (PSO)과 Genetic Algorithms (GA)와 같은 최적화 알고리즘을 구현하였습니다. 이 과정에서 추정 성능 지표가 개선되어, 분류 정확도가 78%에서 92%로 증가하고, 예측 오차는 12%에서 6%로 감소하였습니다. 또한, 모델의 시간적 정확도는 75%에서 88%로 향상되었습니다.

- **Performance Highlights**: 연구 결과, GIS와의 통합은 공간 분석을 풍부하게 하고 지리적 특성 간의 관계를 깊이 이해하는 데 기여하였습니다. 연구 결과는 고급 딥러닝 방법과 GIS, 최적화 전략을 결합하여 원격 감지 응용 프로그램을 크게 발전시킬 수 있음을 보여 줍니다.



### Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation (https://arxiv.org/abs/2412.19853)
- **What's New**: 이 논문은 이미지 생성에서 콘텐츠 충실도(content fidelity)와 예술적 스타일(artistic style) 간의 균형을 유지하는 문제를 해결하는 새로운 방법을 제안합니다. 기존의 Denoising Diffusion Probabilistic Models (DDPMs)에서 발생하는 스타일과 콘텐츠 간의 불일치를 최소화하기 위해 주의(attention) 레이어의 민감도를 분석하여 특정 스타일 요소에 해당하는 레이어를 식별합니다. 이러한 민감한 레이어에만 조건부 입력을 주입함으로써 스타일과 콘텐츠 간의 미세 조정을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 Denoising Diffusion Probabilistic Models (DDPMs)의 주의 레이어에서 민감도를 분석하는 새로운 방법론을 개발했습니다. 연구자는 고정된 콘텐츠 주제를 갖고 다양한 스타일적 요소를 변별하여 민감한 레이어를 신속히 파악하고, 이러한 레이어에 조건부 입력을 적용하여 생성 품질을 향상시켰습니다. 실험을 통해서는 복잡한 조건이 모델 훈련 중 미비하게 표현된 경우에 발생하는 여러 문제를 발견하고, 이를 개선하는 효과를 입증했습니다.

- **Performance Highlights**: 새롭게 제안된 방식은 다중 조건부 입력을 효과적으로 활용할 수 있도록 하여 생성 이미지의 품질을 현저히 향상시켰습니다. 조건부 입력을 적절히 조절함으로써 스타일의 자유도와 일관성을 높이고, 과도한 제약에서 발생하는 아티팩트를 줄이는 성과를 굉장히 잘 보여줍니다. 이 연구는 다양한 예술적 스타일에 대한 조건부 생성에서 더 나은 품질과 다재다능성을 제공하는 새로운 접근법을 제시합니다.



### 3D Face Reconstruction With Geometry Details From a Single Color Image Under Occluded Scenes (https://arxiv.org/abs/2412.19849)
Comments:
          arXiv admin note: text overlap with arXiv:2412.18920

- **What's New**: 이번 연구에서는 3D 얼굴 복원 기술의 발전을 위해 새로운 방법론을 제안합니다. 기존의 방법들은 여러 가지 폐색(occlusion) 시나리오에서 동시에 일반화하는 데 어려움을 겪었지만, 본 논문에서는 bump mapping 기법을 도입하여 중간 수준의 디테일을 coarse 3D 얼굴 모델에 추가했습니다. 또한, 여러 종류의 장애물에 대한 대응을 종합적으로 처리할 수 있는 통합 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 폐색 영역이 있는 경우 완전한 얼굴 특징을 가진 2D 얼굴을 합성하고, 두 번째 단계에서는 제한되지 않은 정면 이미지를 기반으로 한 세부 3D 형태 복원 모듈이 작동합니다. 이에 따라 Pixel-level 인식이 중요한 단계로 작용하며, Fully Convolutional Networks에 기반한 얼굴 분석 기법을 활용하여 세부 정보를 추출합니다.

- **Performance Highlights**: 저자들은 광범위한 실험과 비교를 통해 제안된 방법이 폐색 장면에서도 고품질의 복원 결과를 생성할 수 있음을 입증했습니다. 특히, 개선된 손실 함수(loss function)를 통해 안경을 포함하는 폐색 장면에서도 기존 방법들보다 더 정확한 결과를 도출할 수 있었습니다. 이러한 결과는 실제 이미지에서 최첨단의 질적 성능을 나타내며, 다양한 응용 프로그램에서의 활용 가능성을 제시합니다.



### Generative Landmarks Guided Eyeglasses Removal 3D Face Reconstruction (https://arxiv.org/abs/2412.19848)
Comments:
          arXiv admin note: text overlap with arXiv:2412.18920

- **What's New**: 이번 논문에서는 단일 이미지에서 안경을 제거하며 3D 얼굴을 재구성하는 새로운 방법을 제안합니다. 기존의 얼굴 재구성 방법들은 안경 제거를 자동으로 수행하지 못하며, 실제 환경에서는 이러한 솔루션이 적합하지 않습니다. 우리는 효율적으로 안경 영역을 식별하고 지능적으로 제거하는 과정에서 혁신적인 접근 방식을 모색했습니다.

- **Technical Details**: 우리의 제안된 방법은 얼굴 구조의 2D 추정을 통해 3D 텍스처 작성 과정을 포함합니다. 깊은 학습 아키텍처를 활용하여 단일 2D 이미지에서 3DMM(3D Morphable Model) 표현을 직접 회귀합니다. 또한, 얼굴 파싱 작업을 통합하여 재구성 품질을 향상시키는 방법을 보여줍니다.

- **Performance Highlights**: 광범위한 실험을 통해 기존 3D 얼굴 재구성 작업과 비교하여 우리의 방법이 우수한 규제 능력을 갖추고 있음을 입증했습니다. 우리의 방법은 실제 환경에서 고품질의 재구성 성능을 보이며 최신의 품질 성과를 달성하였습니다.



### Symbolic Disentangled Representations for Images (https://arxiv.org/abs/2412.19847)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 ArSyD(Architecture for Symbolic Disentanglement)를 제안합니다. ArSyD는 각 생성 요인을 객체 표현과 동일한 차원의 벡터로 표현합니다. 이를 통해 생성 요인 벡터 표현의 중첩으로 객체 표현을 얻으며, 이러한 표현을 기호 분리 표현(symbolic disentangled representation)이라고 합니다.

- **Technical Details**: ArSyD는 Hyperdimensional Computing(HDC) 원리를 기반으로 하며, 여기서 기호는 고차원 벡터인 하이퍼 벡터로 표현됩니다. 분리는 구성적으로 이루어지며, 학습 중 기반 분포에 대한 추가 가정을 두지 않고 약한 감독 방식으로 이미지를 복원하도록 모델을 학습합니다. ArSyD는 Fixed Seed Vectors를 사용하여 객체의 피처 값을 생성하며, Attention 메커니즘을 적용하여 객체 속성의 조작을 가능하게 합니다.

- **Performance Highlights**: ArSyD는 dSprites와 CLEVR 데이터셋에서 학습된 기호 분리 표현을 분석하며, 다양한 차원의 잠재 표현을 사용하여 방법의 비교를 허용하는 새로운 분리 메트릭을 제안합니다. 실험 결과 ArSyD는 여러 객체를 포함한 장면에서도 효과적으로 작동하며, 객체의 속성을 통제 가능한 방식으로 편집할 수 있습니다.



### A Review of Latent Representation Models in Neuroimaging (https://arxiv.org/abs/2412.19844)
Comments:
          31 pages, 4 figures

- **What's New**: 이 논문은 Neuroimaging 데이터의 복잡성을 관리하기 위해 Autoencoders, Generative Adversarial Networks (GANs), Latent Diffusion Models (LDMs)와 같은 잠재 표현 모델(latent representation models)이 점점 더 활용되고 있음을 강조합니다. 이러한 모델은 고차원 neuroimaging 데이터를 낮은 차원의 잠재 공간(latent space)으로 축소하여 뇌 기능과 관련된 주요 패턴을 식별하는 데 중점을 두고 있습니다. 또한, 이 리뷰는 임상 응용 이야기부터 뇌의 기본 메커니즘 탐색에 이르기까지 이러한 모델의 다면적인 활용을 논의합니다.

- **Technical Details**: Neuroimaging 데이터의 고차원성은 다양한 예측 및 원인 분석을 어렵게 만듭니다. manifold hypothesis는 실제 고차원 데이터가 저차원 매니폴드(manifold) 상에 존재한다고 제안하여 데이터를 효과적으로 해석하는 데 도움이 됩니다. Principal Component Analysis (PCA)와 Multi-Dimensional Scaling (MDS) 같은 전통적인 차원 축소 기술은 선형 가정에 기반하기 때문에 복잡한 비선형 관계를 포착하는 데 한계가 있습니다. 따라서 Deep Neural Networks (DNNs)는 비선형 관계를 잘 모델링할 수 있는 새로운 방법으로 주목받고 있으며, Autoencoders, GANs, LDMs가 중요한 역할을 하고 있습니다.

- **Performance Highlights**: 잠재 생성 모델(latent generative models)은 고차원 데이터의 효율적인 해석에 기여하며, Neuroimaging의 다양한 임상적인 응용 및 기초 연구에 대한 통찰을 제공합니다. Autoencoder(AE)는 입력 데이터를 커다란 잠재 공간에서 인코딩한 후 원래 형태로 재구성하여 차원 축소(feature extraction) 및 잡음 제거(denoising) 작업에 효과적입니다. 이러한 모델들은 데이터의 본질적인 특성 및 변화를 잘 포착하여 robust한 분포(characterization of its distribution)를 가능하게 합니다.



### Multimodal joint prediction of traffic spatial-temporal data with graph sparse attention mechanism and bidirectional temporal convolutional network (https://arxiv.org/abs/2412.19842)
- **What's New**: 이 연구에서는 다양한 교통 수단 간의 joint prediction을 위한 새로운 방법론인 GSABT(Graph Sparse Attention Mechanism with Bidirectional Temporal Convolutional Network)를 제안합니다. 기존 연구는 단일 교통 수단에 대한 예측에 초점을 맞추었으나, 본 연구는 다중 모드 간의 시간적 및 공간적 특성을 통합하여 예측의 정확성을 높이려 하였습니다. 멀티모달 그래프와 자기 주의 가중치를 활용하여 공간적 로컬 특성을 포착하고, Bidirectional Temporal Convolutional Network를 통해 시간적 기능상관을 강화하는 혁신적 접근을 채택하였습니다.

- **Technical Details**: 제안된 GSABT 모델은 공간적 특성을 위해 graph sparse attention 메커니즘과 Top-U sparse attention을 사용하여 글로벌 특성을 추출합니다. 또한, Bidirectional Temporal Convolutional Network(BiTCN)를 통해 각 교통 수단의 시간적 기능을 양방향으로 동시에 추출합니다. 이 접근은 서로 다른 교통 모드 간의 시간적 특성을 효과적으로 추출할 수 있어 예측 성능을 극대화합니다.

- **Performance Highlights**: GSABT 모델은 세 개의 실제 데이터셋을 대상으로 한 광범위한 실험에서 최첨단 예측 성능을 달성하였습니다. 특히, GSABT는 기존의 단일 교통 수단 예측 모델들에 비해 뛰어난 일반화 능력을 발휘하며, 멀티모달 교통 예측의 가능성을 새롭게 열어주는 결과를 보였습니다. 이를 통해 지능형 교통 시스템에서의 의사결정과 최적화에 기여할 수 있는 잠재력을 지닌다고 할 수 있습니다.



### FlameGS: Reconstruct flame light field via Gaussian Splatting (https://arxiv.org/abs/2412.19841)
- **What's New**: 본 논문에서는 전통적인 ART(Adaptive Resonance Theory) 알고리즘의 시간 소모적이고 계산 집약적인 문제를 해결하기 위해, 불꽃 시뮬레이션 기술에서 영감을 받아 새로운 불꽃 표현 방법을 제안합니다. 이 방법은 불꽃의 발광 과정을 모델링하고 2D 프로젝션 이미지를 감독으로 활용하여 성능을 향상시킵니다.

- **Technical Details**: 제안된 모델은 실제 이미지와 예측된 2D 프로젝션 간의 평균 구조 유사도 지수(Structural Similarity Index)가 0.96에 달하며, 피크 신호 대 잡음 비율(Peak Signal-to-Noise Ratio)은 39.05입니다. 이를 통해 전통적인 알고리즘에 비해 약 34배의 계산 시간을 절약하고 약 10배의 메모리 사용량을 줄이는 효과를 검증했습니다.

- **Performance Highlights**: 본 연구는 실험(validation)을 통해 제안된 모델이 효율성과 성능 면에서 전통적인 방법보다 우수한 결과를 달성했음을 보여줍니다. 이러한 개선은 불꽃 연소 진단 분야에 적용 가능성을 높이며, 연산 비용과 시간 절약의 중요성을 강조합니다.



### ERPA: Efficient RPA Model Integrating OCR and LLMs for Intelligent Document Processing (https://arxiv.org/abs/2412.19840)
Comments:
          6 pages , 2 figures, 1 algorithm

- **What's New**: 이 논문에서는 이민 업무 내 ID 데이터 추출을 개선하고 Optical Character Recognition (OCR) 작업을 최적화하기 위해 설계된 혁신적인 Robotic Process Automation (RPA) 모델인 ERPA를 소개합니다. 기존 RPA 솔루션은 대량의 문서를 처리할 때 성능 한계에 직면하는 경우가 많아 비효율성을 초래했습니다. ERPA는 대규모 언어 모델(Large Language Models, LLMs)을 통합하여 추출된 텍스트의 정확성과 명확성을 개선하고, 애매한 문자 및 복잡한 구조를 효과적으로 처리하여 이 문제를 해결합니다.

- **Technical Details**: ERPA는 고급 OCR 기술과 LLM을 통합한 향상된 RPA 프레임워크로, 문서 처리 워크플로우에서 텍스트 추출의 정확성과 적응성을 높입니다. 이 모델은 다양한 문서 형식에서 텍스트 콘텐츠를 추출하기 위해 최첨단 OCR 시스템을 활용하고, LLM을 적용하여 추출된 데이터를 정제하고 검증하여, 모호하거나 복잡한 구조를 다룰 때 더 나은 정확성을 보장합니다. 또한 ERPA는 다양한 문서 레이아웃에 동적으로 적응하여 ID, 여권, 비자 및 증명서와 같은 이민 문서를 실시간으로 원활하게 처리할 수 있습니다.

- **Performance Highlights**: ERPA는 기존의 RPA 플랫폼에 비해 처리 시간을 93% 감소시켜 ID 데이터 추출을 단 9.94초 안에 완료하는 성과를 보여줍니다. 이 시스템은 자동화된 고사양, 문서 집약적 작업 흐름을 혁신할 가능성을 지니고 있으며, 이민 서비스와 같은 고볼륨 환경에서의 처리 속도와 정확성을 동시에 개선하여 새로운 표준을 제시합니다.



### Multi-View Fusion Neural Network for Traffic Demand Prediction (https://arxiv.org/abs/2412.19839)
- **What's New**: 이 연구는 교통 수요 예측에서 공간적 및 시간적 특징을 효과적으로 추출하기 위해 multi-view fusion neural network(MVFN) 접근 방식을 제안합니다. 기존의 고정된 공간 그래프와 통합된 시간 모델링 메커니즘이 가진 한계를 극복하며, 공간적 지역 특징과 전역 특징을 동시에 추출할 수 있는 방법론을 개발하였습니다.

- **Technical Details**: 이 모델은 그래프 합성곱 신경망(Graph Convolutional Network, GCN)과 코사인 재가중선형 주의 메커니즘(Cosine Re-weighting Linear Attention, CLA)을 결합하여 공간적 특징을 추출합니다. 또한, 다중 채널 분리형 시간 합성곱 네트워크(Multi-Channel Separable Temporal Convolutional Network, MSTCN)를 통해 고립된 시간적 특징을 고려하여 예측의 정확성을 향상시킵니다.

- **Performance Highlights**: 제안된 모델은 두 개의 실제 교통 수요 데이터셋에서 검증되었으며, 기존의 예측 방법들보다 높은 예측 정확도를 달성했습니다. 이는 도시에서의 차세대 지능형 교통 시스템에 대한 강력한 기여를 할 것으로 기대됩니다.



### Multi-atlas Ensemble Graph Neural Network Model For Major Depressive Disorder Detection Using Functional MRI Data (https://arxiv.org/abs/2412.19833)
Comments:
          17 pages, 2 figures, 10 tables

- **What's New**: 이 연구는 앙상블 기반의 그래프 신경망(GNN) 모델을 개발하여 rs-fMRI 이미지를 분석하고 주요 우울 장애(MDD)를 진단하는 데 중요한 특징들을 탐지하는 데 목적을 두고 있습니다. 기존 연구들이 단일 뇌 영역 세분화 템플릿을 사용한 데 반해, 본 연구는 여러 뇌 아틀라스의 특징을 결합하여 뇌의 복잡성을 포착하고 보다 정확한 진단을 목표로 하고 있습니다. 또한, 대규모 다기관 MDD 데이터셋을 통해 모델의 효과를 입증하였습니다.

- **Technical Details**: 모델은 네 개의 GNN으로 구성되며, 각각이 다른 뇌 아틀라스에서 파생된 기능적 연결망(FCN)을 통해 훈련됩니다. 이는 단일 아틀라스 기반 모델에 비해 뇌의 복잡성을 더 잘 포착하고 구별된 특징을 선별할 수 있는 장점을 제공합니다. 또한, 합성 소수 민족 과대표집(SMOTE) 기법을 적용하여 훈련 데이터의 다양성을 높였습니다.

- **Performance Highlights**: 본 연구의 모델은 모든 검증 절차에서 75.80%의 정확도, 88.89%의 민감도, 61.84%의 특이도, 71.29%의 정밀도, 79.12%의 F1-score를 기록하며 우수한 성능을 보였습니다. 이러한 성과는 MDD 진단의 정확성을 향상시키고, 정신 건강 문제 조기 탐지에 기여할 것으로 기대됩니다.



### Vitron: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing (https://arxiv.org/abs/2412.19806)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 최근 비전 대형 언어 모델(LLMs)의 발전이 주목할 만한 성과를 내고 있지만, 여전히 여러 도전 과제가 존재합니다. 본 논문에서는 VITRON을 소개하는데, 이는 정적 이미지와 동적 비디오의 이해, 생성, 분할, 편집을 위한 보편적인 픽셀 수준의 비전 LLM입니다. VITRON은 LLM 백본을 기반으로 이미지 및 비디오 인코더를 통합하여 다양한 비전 작업을 지원하도록 설계되었습니다.

- **Technical Details**: VITRON은 최첨단 시각 전문 모듈을 활용하여 비전 종단 작업을 포괄하는 기능을 제공합니다. 이를 통해 낮은 수준에서 높은 수준까지 비주얼 인식 및 생성 작업을 수행할 수 있습니다. 또한 새로운 하이브리드 방법을 통해 LLM과 백엔드 모듈 간의 효과적이고 정밀한 메시지 전송을 보장하며, 텍스트 지시와 신호 임베딩을 동시에 통합합니다.

- **Performance Highlights**: 12개 비주얼 작업 및 22개 데이터셋을 통해 VITRON의 광범위한 역량이 입증되었습니다. 특히, 다양한 비전 작업 간의 시너지를 극대화하는 작업 불변의 미세한 비주얼 특징을 학습하는 크로스-태스크 시너지 모듈이 중요한 역할을 합니다. 이러한 결과는 보다 통합된 다중 모달 일반 모델을 개발할 수 있는 큰 잠재력을 보여줍니다.



### UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI (https://arxiv.org/abs/2412.20977)
Comments:
          Project page: this http URL

- **What's New**: UnrealZoo는 Unreal Engine 기반의 고급스러운 3D 가상 세계의 방대한 컬렉션으로, 복잡한 오픈 월드의 다양성과 변동성을 반영하여 만들어졌다. 사용자는 손쉽게 사용할 수 있는 Python API를 통해 데이터를 수집하고, 환경을 증대시키며, 분산 학습 및 벤치마킹 등의 다양한 응용 프로그램을 활용할 수 있다. 이를 통해 Embodied AI 에이전트의 연구를 발전시키고 다양한 실험을 지원하는 것이 가능하다.

- **Technical Details**: UnrealZoo는 100개의 고품질의 현실감 넘치는 장면과 다양한 playable entities를 포함하고 있다. 이러한 환경은 아티스트에 의해 세밀하게 설계되어 실제 조명, 질감 및 동역학을 모방한다. 또한, UnrealCV의 통신 효율성을 최적화하고 다양한 요구사항에 맞춘 Gym 인터페이스를 제공하여 사용자와의 상호작용을 향상시킨다.

- **Performance Highlights**: UnrealZoo의 실험 결과는 Embodied AI 에이전트의 일반화 및 강인성을 높이고, 동적 요인을 처리하기 위한 저지연 폐쇄 루프 제어의 필요성을 강조한다. 특정 태스크로는 embodied visual navigation과 tracking을 사용하여, 복잡한 동적 환경에서 에이전트의 성능을 평가하였다. 이러한 다양한 훈련 환경은 RL 기반 및 VLM 기반 에이전트가 오픈 월드에서 직면하는 한계를 이해하는 데 중요한 통찰을 제공한다.



### Fine-Tuning TransMorph with Gradient Correlation for Anatomical Alignmen (https://arxiv.org/abs/2412.20822)
- **What's New**: 이 논문에서는 뇌 MRI 등록에서의 unsupervised deep learning 접근 방식의 진전을 다루고 있습니다. 제안된 방법은 사전 훈련된 TransMorph 모델을 미세 조정하여 해석학적 정확성을 향상시키고, 해부학적 정렬을 유지하는 간섭을 줄이는 것에 중점을 두고 있습니다. FAdam optimizer와 gradient correlation을 포함하여 매끄럽고 구조적으로 일관된 변형을 달성하는 데 중점을 두었습니다.

- **Technical Details**: 이 방법론은 특히 Fisher Adam (FAdam) optimizer와 gradient correlation (GC) 유사도 측정을 포함합니다. FAdam은 훈련 중 최적화를 개선하기 위해 조정된 모멘텀과 편향 보정, 적응형 그래디언트 스케일링을 사용하는 Adam 변형입니다. 또한, GC는 뇌 MRI 간의 구조적 일관성을 유지하는 데 도움이 되며, 고차원 이미지를 비교하는 데 유용한 유사도 측정법입니다.

- **Performance Highlights**: 실험 결과, FAdam+GC 모델이 Dice 계수와 95% Hausdorff 거리(HdDist95)에서 약한 개선을 보였으며, 비변형 볼륨(NDV)은 0.27%로 현저하게 낮아져 매끄러운 변형을 나타냅니다. 평가된 모델 간에는 유사한 등록 성능이 있지만 FAdam+GC가 하는 gradient correlation 덕분에 경계 부위에서 보다 나은 정렬을 보여 주며, 이는 HdDist95 지표에 긍정적인 영향을 미쳤습니다.



### A Tale of Two Imperatives: Privacy and Explainability (https://arxiv.org/abs/2412.20798)
Comments:
          Work in progress

- **What's New**: 이번 논문은 인공지능(AI) 분야에서의 프라이버시(Right-to-Privacy, RTP)와 설명 가능성(Right-to-Explanation, RTE)의 복합적 문제를 다룹니다. Differential Privacy (DP)가 개인정보 보호 머신러닝의 금본위로 자리 잡은 이유와, 포스트-호크(post-hoc) 설명 기법들이 모델 감사에 어떻게 기여할 수 있는지를 탐구합니다. 특히, 이 논문은 고위험(high-stakes) 어플리케이션에서 RTP와 RTE를 효과적으로 통합하는 방법을 제안합니다.

- **Technical Details**: 이 연구는 DP 모델과 다양한 포스트-호크 설명 기법의 상호작용을 분석하여, 이들 각각의 특성과 제한 사항을 명확하게 합니다. 연구에서는 3종의 CNN 모델, 6개의 ε (epsilon) 값, 그리고 5개의 인기 있는 포스트-호크 설명 방법을 사용하여 실험을 진행하며 이들이 RTP 조건을 만족하는지를 검토합니다. 이를 통해 RTP와 RTE의 결합이 고위험 상황에서 어떻게 효과적으로 이루어질 수 있는지를 제시합니다.

- **Performance Highlights**: 연구의 주요 성과는 DP 모델과 포스트-호크 설명 기법들의 조화로운 작동 가능성을 탐구하고, 이를 통해 고위험 AI 시스템에서도 신뢰성과 효율성을 확보할 수 있는 기초를 제공한다는 점입니다. 또한, 이 논문은 의료 기록 같은 프라이버시가 중요한 분야에서 환자와 의사 모두가 신뢰할 수 있는 예측을 제공하는 것이 필수적임을 강조합니다. 결과적으로, RTP와 RTE를 만족하는 산업 소프트웨어 파이프라인 예시를 도출하여 실질적인 적용 가능성을 제시합니다.



### Residual Connection Networks in Medical Image Processing: Exploration of ResUnet++ Model Driven by Human Computer Interaction (https://arxiv.org/abs/2412.20709)
- **What's New**: 이 논문은 ResNet과 Unet++를 결합한 하이브리드 모델인 ResUnet++를 제안합니다. 이 모델은 뇌종양 탐지 및 위치 확인을 개선하고, 의사와 의료 이미지 시스템 간의 원활한 상호작용을 촉진하도록 설계되었습니다. 기존의 연구에서는 의료 이미지 처리를 위한 CNN의 활용은 있었지만, 인간-컴퓨터 상호작용(Human-Computer Interaction, HCI)과의 통합에 대한 연구는 거의 없었습니다.

- **Technical Details**: ResUnet++는 다운샘플링(downsampling) 및 업샘플링(upsampling) 단계 모두에서 잔여 블록(residual block)을 통합하여 중요한 이미지 특징이 보존되도록 합니다. HCI 원칙을 도입함으로써, 모델은 직관적이고 실시간 피드백을 제공하여 의사가 종양 위치 결과를 효과적으로 시각화하고 상호작용할 수 있도록 지원합니다. 이는 임상 환경에서 정보에 기반한 의사결정을 촉진하고 작업 흐름의 효율성을 높입니다.

- **Performance Highlights**: ResUnet++는 LGG 세분화 데이터셋에서 평가되었으며, Jaccard Loss 98.17%를 기록했습니다. 이 결과는 강력한 세분화 성능을 보여주며, 실제 응용 가능성을 시사합니다. 이 모델은 고급 의료 이미지 기술과 HCI를 접목함으로써, 대화형 진단 도구 개발의 기초를 제공하고, 의사의 신뢰도 및 결정 정확성, 환자 결과 향상에 기여할 수 있을 것으로 기대됩니다.



### Prototypical Distillation and Debiased Tuning for Black-box Unsupervised Domain Adaptation (https://arxiv.org/abs/2412.20670)
- **What's New**: 최근 연구에서는 소스 데이터를 통한 레이블링의 높은 비용을 회피하기 위한 방법으로 source-free domain adaptation에 대한 관심이 증가하고 있습니다. 본 논문은 black-box domain adaptation이라는 새로운 설정을 제안하며, 이는 모델이 API를 통해서만 접근될 수 있고 레이블과 신뢰도 값을 예측하는 데 중점을 둡니다. 이를 위해, 원시 예측과 타겟 영역에서 파생된 프로토타입을 활용하여 사용자 맞춤형 타겟 모델을 증류하는 방법론을 개발하였습니다.

- **Technical Details**: 본 논문에서 제안한 ProDDing(Prototype Distillation and Debiased Tuning)은 두 단계로 구성된 프레임워크입니다. 첫 번째 단계에서는 소스 모델로부터의 원시 예측과 타겟 도메인으로부터의 프로토타입을 이용해 지식을 증류합니다. 두 번째 단계에서는 특정 클래스에 편향된 로짓을 패널티를 주면서 증류된 모델을 미세 조정합니다. 아울러 adaptive label smoothing 기법을 도입하여 제안된 소스 예측으로부터의 신뢰도를 최적화합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 통해 ProDDing이 기존 black-box 도메인 적응 방법보다 월등히 뛰어난 성능을 보이는 것으로 확인되었습니다. 특히, 레이블 예측정보만 제공되는 hard-label black-box 도메인 적응 시나리오에서도 ProDDing이 상당한 성과를 기록하여 그 유용성을 입증하였습니다. 본 연구는 black-box 도메인 적응의 현실적이고 도전적인 상황을 설정하여 높은 수준의 성능을 달성하였습니다.



### Conformable Convolution for Topologically Aware Learning of Complex Anatomical Structures (https://arxiv.org/abs/2412.20608)
- **What's New**: 본 연구에서는 기존의 컴퓨터 비전 기술의 한계를 극복하기 위해 'Conformable Convolution'이라는 새로운 합성곱 층을 제안합니다. 이는 의학 이미지 분석에서 필수적인 복잡한 위상 구조를 명시적으로 유지하는 데 중점을 둡니다. 제안된 'Topological Posterior Generator (TPG)' 모듈을 통해 높은 위상적 중요성을 가진 영역에 적응 가능한 커널 오프셋을 중점을 두어 적용할 수 있도록 하였습니다.

- **Technical Details**: Conformable Convolution은 위상적 선호를 학습하여 이미지 내에서 중요한 구조의 연속성과 연결성을 캡처하는 데 기여합니다. 이 모델은 Persistent Homology를 활용하여 피처 맵을 변환한 후, 큐빅 컴플렉스에서 주요 위상적 특징을 식별합니다. 제안한 모듈들은 현재의 다양한 아키텍처에 쉽게 통합될 수 있으며, 구조의 위상적 연속성을 보장하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과는 제안된 프레임워크가 다양한 데이터셋에서 세분화 성능을 향상시키며 위상적 속성을 효과적으로 보존하고 있음을 보여줍니다. 특히, 망막 혈관, 대장암 세포 및 신경 전자현미경 등의 다양한 구조에 대한 분석 결과에서 강력한 성능을 입증하였습니다. Quantitative 및 qualitative 메트릭에서 기존 SOTA 모델과 비교하여 동등하거나 더 좋은 결과를 거두고 있습니다.



### Segmentation of Muscularis Propria in Colon Histopathology Images Using Vision Transformers for Hirschsprung's Diseas (https://arxiv.org/abs/2412.20571)
Comments:
          To be published in the CMBEC47/ACCES26 Joint Conference

- **What's New**: 이번 연구에서는 Hirschsprung's disease (HD)에서 근육층(muscularis propria) 분할(segmentation)을 위해 Vision Transformers (ViTs)를 적용하고 그 성능을 CNN 및 얕은 학습(shallow learning) 방법과 비교했습니다. ViTs는 자기 주의(self-attention) 메커니즘 덕분에 최근 강력한 딥러닝(deep learning) 접근법으로 부각되고 있습니다. 이러한 접근법은 병리학 이미지 분석의 자동화를 가능하게 하여 병리학자의 업무를 효율적으로 지원할 수 있습니다.

- **Technical Details**: 연구의 주요 목표는 calretinin 염색(histopathology images)된 조직 슬라이드에서 근육층의 정확한 분할을 구현하는 것으로, ViT 모델은 DICE 점수(dice score) 89.9%와 Plexus Inclusion Rate (PIR) 100%를 달성했습니다. 이는 CNN 모델의 DICE 점수(89.2%; PIR 96.0%) 및 k-평균(k-means) 클러스터링 방법(80.7%; PIR 77.4%)보다 뛰어난 성과입니다. 이러한 결과는 ViTs가 HD 관련 이미지 분석을 위한 유망한 도구임을 입증합니다.

- **Performance Highlights**: ViT 모델은 높은 정확성과 함께 JPEG 처리와 같은 이미지 전처리(preprocessing)의 필요성을 줄여주는 장점을 가지고 있습니다. 병리학적 이미지 분석에서 ViTs를 활용하면 인력과 자원의 효율성을 높일 수 있습니다. 이러한 혜택을 통해 HD의 조기 진단 및 관리에 기여할 수 있는 가능성이 큽니다.



### KVC-onGoing: Keystroke Verification Challeng (https://arxiv.org/abs/2412.20530)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2401.16559, arXiv:2311.06000

- **What's New**: 이 논문에서는 연구자들이 표준 실험 프로토콜을 이용해 시스템을 벤치마킹할 수 있는 Keystroke Verification Challenge - onGoing (KVC-onGoing)을 소개합니다. KVC-onGoing은 185,000명 이상의 주제로부터 수집된 대규모 공개 데이터베이스인 Aalto University Keystroke databases를 기반으로 합니다. 이를 통해 사용자는 데스크탑 및 모바일 시나리오에 대한 타이핑 행동을 분석하여 고차원적인 성능을 평가할 수 있습니다.

- **Technical Details**: Keystroke Dynamics (KD)는 인간의 타이핑 행동을 의미하며, 일반적으로 생체 인식 특성으로 간주됩니다. 본 연구는 KD 기반 생체 인증의 표준 실험 프로토콜과 벤치마크를 제안하며, 이는 한층 향상된 정확성 및 다양한 해석을 제공합니다. 또한, 185,000명 이상의 참여자로 구성된 데이터셋을 활용하여 가정용 키보드 및 모바일 키보드에서 수집된 타이핑 데이터의 특징을 분석합니다.

- **Performance Highlights**: KVC-onGoing의 평가 세트에서는 Equal Error Rate (EER)가 3.33% 및 False Non-Match Rate (FNMR)가 11.96%에 달하며, 이는 기존의 최고 성능을 크게 개선한 수치입니다. 모바일 환경에서는 EER이 3.61% 및 FNMR이 17.44%로 평가되었습니다. 이러한 결과는 행동 기반 생체 인식 시스템의 우수한 판별력을 나타내며, 연구자들에게 새로운 벤치마킹 기준을 제공해 줍니다.



### Multimodal Variational Autoencoder: a Barycentric View (https://arxiv.org/abs/2412.20487)
Comments:
          AAAI 2025

- **What's New**: 이번 논문은 멀티모달 인코딩을 위한 변분 오토인코더(variational autoencoder, VAE)의 새로운 이론적 접근을 제시합니다. 기존의 전문가 모델(연합 전문가 모델)의 한계점을 극복하고, 바리센터(barycenter) 개념을 통해 더욱 유연한 표현 방식을 제안합니다. 특히 이 방법은 누락된 모달리티(missing modality)에서도 강력한 성능을 발휘할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 저자들은 바리센터를 통해 비대칭 가중치 KL 발산(asymmetric weighted KL divergence)을 최소화하는 방법을 제안하며, 이는 기존의 전문가 모델을 대표하는 특수한 경우로 볼 수 있습니다. 논문에서 제안하는 새로운 바리센터는 다양한 발산 유형(divergences)을 고려하여 두 전문가 모델(PoE, MoE)을 확장합니다. 특히, 2-바리센터(Wasserstein barycenter)로 정의된 Wasserstein 거리를 활용하여 모달리티 특화 및 비특화 표현을 보다 잘 보존한다고 설명하고 있습니다.

- **Performance Highlights**: 세 가지 멀티모달 벤치마크에서의 경험적 연구를 통해 제안된 방법의 효과성을 검증했습니다. 실험 결과, 새로운 바리센터 기반의 VAE가 기존 방법보다 더 우수한 성능을 보였으며, 미존재 모달리티 환경에서도 강력한 일반화 능력을 발휘했습니다. 이러한 연구는 향후 다양한 멀티모달 학습 응용 분야에 중요한 기여를 할 것으로 기대됩니다.



### Unlocking adaptive digital pathology through dynamic feature learning (https://arxiv.org/abs/2412.20430)
Comments:
          49 pages, 14 figures

- **What's New**: 본 논문에서는 혁신적인 동적 특성 학습 방법인 PathFiT을 소개하고 있습니다. 이 방법은 다양한 병리학 기반 모델에 쉽게 추가하여 그들의 적응성을 높이고, 실질적인 병리학적 요구를 충족할 수 있도록 합니다.

- **Technical Details**: PathFiT은 20 테라바이트 이상의 디지털 병리학 벤치마크를 구성하여 28개의 H
dash;E 염색 작업과 Masson's Trichrome 염색 및 면역형광 이미지를 포함한 7개의 전문 이미징 작업을 검토합니다. 이 방법은 임상 응용 프로그램의 다양성에 따라 원활하게 통합되며, 범용 특성을 보유한 모델들이 더욱 진화할 수 있도록 합니다.

- **Performance Highlights**: PathFiT을 사용하여 대표적인 병리학 기반 모델에 적용한 결과, 35개의 작업 중 34개에서 최첨단 성능을 시연하였고, 23개 작업에서 상당한 개선을 이루었습니다. 전문 이미징 작업에서는 10.20%의 성능 향상을 보이며, PathFiT의 뛰어난 성능과 다재다능함이 계산병리학의 새로운 가능성을 열어나갑니다.



### Election of Collaborators via Reinforcement Learning for Federated Brain Tumor Segmentation (https://arxiv.org/abs/2412.20253)
- **What's New**: 이 연구는 유연한 데이터 선택을 통해 데이터 프라이버시를 유지하면서도 다양한 데이터를 활용하는 새로운 연합 학습 모델인 RL-HSimAgg를 제안합니다. 이 알고리즘은 강화 학습(Reinforcement Learning) 및 유사성 가중 집계(Similarity-weighted aggregation)를 사용하여 아웃라이어 데이터를 관리합니다. 특히, Epsilon-greedy와 upper confidence bound(UCB) 방법을 적용하여 협력자 선택을 최적화하고 모델의 일반화를 향상시키는 방법을 보여주고 있습니다.

- **Technical Details**: 연구 결과는 연합 뇌 병변 세분화(Federated Brain Lesion Segmentation) 작업에 적용되었습니다. 이로써 glioblastoma 환자의 다중 매개변수 자기 공명 영상(multi-parametric magnetic resonance imaging, mpMRI) 데이터셋을 사용하여 강화 학습 기반 협력자 관리의 효율성을 분석했습니다. 연구진은 Intel의 Federated Learning OpenFL 프레임워크와 U-Net CNN 아키텍처를 통해 협력적인 세분화 모델을 교육하였고, 각 라운드마다 참여 사이트의 20%가 모델 업데이트에 기여하도록 구성했습니다.

- **Performance Highlights**: RL-HSimAgg의 실험 결과, UCB 협력자는 모든 평가 지표에서 Epsilon-greedy 방법보다 더 높은 성능을 보였습니다. 특히, 치명적인 종양의 세분화(Dice score)는 0.7334로 기록되었으며, 종양의 다양한 부분에서도 유사한 성과를 보였습니다. 이러한 결과는 연합 학습 환경에서 모델의 견고함과 유연성을 향상시키기 위한 RL 기반 협력자 관리 기법의 유용성을 입증합니다.



### Recommender Engine Driven Client Selection in Federated Brain Tumor Segmentation (https://arxiv.org/abs/2412.20250)
- **What's New**: 이번 연구에서는 Federated Learning (FL) 프로세스를 최적화하기 위한 효율적인 클라이언트 선택 프로토콜을 제안합니다. 이를 위해 비부정적 행렬 분해 (NNMF) 기반의 추천 엔진을 도입하고, 콘텐츠 기반 및 협업 필터링을 혼합한 하이브리드 집계 접근 방식을 사용합니다. 또한, 새로운 협력자 추천 방식을 통해 데이터 제한으로 인한 콜드 스타트 문제를 해결하며, FL 프로세스의 정밀성과 효율성을 크게 향상시킵니다.

- **Technical Details**: 연구에 사용된 데이터셋은 1251명의 교모세포종(GBM) 환자로부터 수집된 다중 매개변수 자기공명영상(mpMRI) 스캔으로 구성됩니다. Intel의 Federated Learning(OpenFL) 프레임워크를 활용하여 데이터 프라이버시를 유지하며 협업 모델을 학습합니다. 새로운 추천 엔진 시스템은 NNMF 전략을 적용하여 과거 성과 데이터를 분석하고, 협력자의 기여도를 평가하여 최적의 협력자를 선택합니다.

- **Performance Highlights**: 제안된 연합 모델은 외부 검증 세트에서 각각 증식 종양(ET), 종양 중심部(TC), 전체 종양(WT) 세분화 작업에 대해 0.7298, 0.7424, 0.8218의 Dice 점수를 기록했습니다. 협력자 선택의 정확성을 높이기 위해 NNMF 기반의 추천 엔진이 통합되어 FL 모델의 전반적인 효율성과 효과를 크게 향상시킴을 보여주었습니다. 이 연구는 특히 뇌 종양 세분화와 같은 특정 작업에 맞는 협력자를 선택함으로써 FL 네트워크의 효율성을 높일 수 있음을 입증합니다.



### Enhancing Transfer Learning for Medical Image Classification with SMOTE: A Comparative Study (https://arxiv.org/abs/2412.20235)
Comments:
          Accepted in 27th International Conference on Computer and Information Technology (ICCIT) 2024

- **What's New**: 본 논문은 Transfer Learning (TL)의 의료 영상에서 다중 레이블 이미지 분류 적용을 확장하고 향상시킵니다. 특히, 뇌 종양 분류 및 당뇨망막병증 단계를 탐지하는 데 중점을 두었으며, 사전 훈련된 모델을 사용하여 의료 이미징에서의 효과를 평가합니다. 연구 결과 TL 모델이 뇌 종양 분류에서 뛰어난 성능을 보였지만, 당뇨망막병증 탐지에서는 클래스 불균형으로 인해 성능이 저하됨을 발견했습니다.

- **Technical Details**: 이 연구에서는 MobileNet, Xception, InceptionV3, ResNet50, DenseNet201의 5개 사전 훈련된 모델을 사용하여 Brain Tumor MRI 및 APTOS 2019의 두 데이터 세트에서 TL 기법의 효과를 분석합니다. 특히, SMOTE (Synthetic Minority Over-sampling Technique)와 전통적인 머신 러닝(ML) 방법을 결합하여 데이터 불균형 문제를 해결하도록 했습니다. 이로 인해 정확도는 1.97%, 재현율(민감도)은 5.43%, 특이도는 0.72% 향상되었습니다.

- **Performance Highlights**: 본 연구에서 제시된 결과는 TL과 리샘플링 기법, ML 방법을 결합하여 데이터 불균형 문제를 해결하고 분류 성능을 향상시키는 필요성을 강조합니다. 특히, 뇌 종양 분류에서의 TL 모델은 거의 최적의 메트릭을 달성했으며, 이를 통해 의료 이미지 분석의 정확성과 신뢰성을 높이고, 환자 결과를 개선하는 데 기여할 수 있는 가능성을 제시합니다.



### Self-Calibrated Dual Contrasting for Annotation-Efficient Bacteria Raman Spectroscopy Clustering and Classification (https://arxiv.org/abs/2412.20060)
- **What's New**: 이 논문에서는 라만 산란(Raman scattering)을 이용한 병원균 진단에 있어서 주목할 만한 기술로, 스펙트럼 해석을 위한 새로운 기법인 Self-Calibrated Dual Contrasting (SCDC) 방법을 제안합니다. 이 방법은 적은 수의 주석(annotation)으로도 효과적으로 작동하며, 두 가지 관점에서 스펙트럼을 표현하는 것을 목표로 합니다.

- **Technical Details**: SCDC 방법은 두 가지 다른 하위 공간에서 스펙트럼을 표현하는 이중 대조 학습(dual contrastive learning) 접근 방식을 사용합니다. 여기에서 임베딩(embedding) 관점은 인스턴스 수준 정보를 포착하고, 카테고리(category) 관점은 카테고리 수준 정보를 반영합니다. 이러한 구조를 통해 라만 분광학 인식에서 비지도 및 준지도 학습 조건에서 적용 가능한 판별 표현(discriminative representations)을 얻습니다.

- **Performance Highlights**: 대규모 박테리아 라만 분광 데이터셋 세 개를 통해 식별 작업을 검증한 결과, SCDC 방법은 주석이 거의 없거나 5% 또는 10%의 적은 양으로 강력한 인식 성능을 발휘했습니다. 이는 제안된 방법이 주석 효율적인 임상 환경에서 바이오 스펙트럼 식별 가능성을 시사합니다.



### Uncertainty Quantified Deep Learning and Regression Analysis Framework for Image Segmentation of Skin Cancer Lesions (https://arxiv.org/abs/2412.20007)
Comments:
          Presented at the 2024 IEEE International Conference on Machine Learning and Applications (ICMLA), accepted for publication and in press by IEEE

- **What's New**: 이 연구는 피부 병변 분할을 위한 두 가지 딥 러닝 모델(DLMs)을 제안하고 있습니다. 하나는 처음부터 훈련된 모델이고, 다른 하나는 전이 학습(transfer learning)을 기반으로 하며, 몬테 카를로 드롭아웃(Monte Carlo dropout) 또는 베이즈백전파(Bayes-by-backprop)로 불확실성 추정을 수행합니다. 연구는 DLM의 예측 신뢰성을 향상시키기 위해 픽셀 수준의 불확실성 추정을 처음으로 도입하였습니다.

- **Technical Details**: 연구 방법론으로는 DLM의 성능을 향상시키기 위한 불확실성 정량화를 포함하여, 몬테 카를로 드롭아웃(MCD)과 베이즈백전파(BBP) 기법이 사용되었습니다. 이를 통해 임상에서의 이미지 분할 정확성을 높이고, 피부 병변에 대한 위험 평가를 보다 효율적으로 수행할 수 있게 되었습니다. 다양한 피부 병변을 분할하기 위해 기존의 VGG-UNet 구조를 기반으로 하여 훈련되었습니다.

- **Performance Highlights**: 이 연구의 분석 결과, Dice 점수와 DLM 성능 간의 높은 상관관계가 발견되었습니다(p < 0.05). DLM은 임상적으로 중요한 피부 지역의 병변을 분할하는 데 있어 높은 신뢰성과 해석 가능성을 제공하며, 제안된 리니어 회귀 모델을 통해 효율적인 불확실성 추정 작업을 가능하게 했습니다. 이러한 접근법은 피부 이미지 진단 및 예측에서 중요한 역할을 할 수 있을 것으로 기대됩니다.



### SegKAN: High-Resolution Medical Image Segmentation with Long-Distance Dependencies (https://arxiv.org/abs/2412.19990)
- **What's New**: 이 논문에서는 SegKAN이라는 혁신적인 모델을 제안합니다. SegKAN은 전통적인 embedding 모듈을 개선하고 새로운 convolutional 네트워크 구조를 채택하여 이미지 노이즈를 부드럽게 하며, Patch 블록 간의 관계를 공간적에서 시간적 관계로 변환하여 기존 Vision Transformer 모델의 문제를 해결합니다. 실험 결과, 기존 최신 기술 모델 대비 Dice 점수가 1.78% 향상되었습니다.

- **Technical Details**: SegKAN 모델은 3D 이미지를 여러 3D 패치로 나누고, 패치 간의 공간적 관계를 시계열 네트워크를 통해 순차적 관계로 변환합니다. 이를 통해 전통적인 Vision Transformer 모델에서 패치의 위치적 관계를 효과적으로 캡처할 수 있습니다. 또한, FKAC(Fourier-based KAN Convolution)를 통해 로컬 영역의 노이즈를 부드럽게 하여 네트워크 학습 중 그래디언트 폭발을 방지하고 모델의 안정성을 높입니다.

- **Performance Highlights**: 제안된 SegKAN 모델은 Hepatic Vessel 데이터셋에서 실험한 결과, 현재의 최첨단 방법인 TransUNet보다 우수한 성과를 나타냈습니다. 이 모델은 Transformer 기반 아키텍처 내에서 맥락 모델링 능력을 더욱 최적화하여 장거리 분할 성능을 개선하는 데 기여합니다. 이러한 성과는 복잡하고 고해상도의 의료 영상에서 혈관 구조를 효과적으로 분할할 수 있는 가능성을 보여줍니다.



### Standard-Deviation-Inspired Regularization for Improving Adversarial Robustness (https://arxiv.org/abs/2412.19947)
- **What's New**: 이 연구는 Adversarial Training (AT) 기법의 효율성을 높이기 위해 표준 편차에 기반한 정규화 항(SDI measure)을 제안합니다. 기존 AT가 취약기를 훈련시키는 데 효과적인 반면, 이번 연구는 SDI 정규화를 통해 더욱 강력한 공격에 대한 DNN의 강인성을 향상시키는 방법을 제시합니다. 연구를 통해 SDI 측정이 adversarial 예제를 작성하는 데 유용하다는 실험 결과도 보여줍니다.

- **Technical Details**: 연구는 AT의 내부 최대화 단계가 모델 출력 확률의 변별력을 극대화하려는 시도와 유사하다는 점을 강조합니다. 또한 SDI 손실 줄이기를 AT의 외부 최소화 단계와 연관 지어 설명합니다. SDI 측정은 cross-entropy, entropy 또는 Kullback-Leibler divergence와 같은 기존의 정보 이론적 손실들에 의존하지 않으며, 이러한 특성으로 인해 SDI는 기존 AT 변형과 통합될 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: 제안된 SDI 정규화 항의 적용은 Auto 공격과 CW 공격에 대한 기존 AT 변형의 강인성을 더욱 개선하는 결과를 가져왔습니다. 또한, SDI 정규화가 기존 AT 변형의 일반화 성능 역시 향상시켜, adversarial 훈련 중에 관찰되지 않은 공격에 대해 더 나은 성능을 보였습니다. 마지막으로 SDI 메트릭을 사용하여 생성된 adversarial 예제는 cross-entropy 손실 및 KL divergence를 사용하여 생성된 것들과 비교되어, 더 높은 성공률을 보여주었습니다.



### RoboSignature: Robust Signature and Watermarking on Network Attacks (https://arxiv.org/abs/2412.19834)
- **What's New**: 이번 연구에서는 생성 모델의 워터마킹 기능과 관련된 취약점을 파악하고 이를 해결하기 위한 새로운 알고리즘을 제시합니다. 기존의 Stable Signature 방식은 적절한 워터마크를 숨기기 위해 LDM의 디코더를 세밀하게 조정하지만, 우리는 모델이 의도한 워터마크를 삽입하지 못하게 하는 적대적 파인튜닝 공격의 가능성을 강조합니다. 이 새로운 공격은 모델이 잘못된, 임의의 워터마크를 심는 경우를 발생시켜 많은 기존 방법들의 취약성을 드러냅니다.

- **Technical Details**: 우리는 LDM(잠재적 확산 모델)을 통해 생성된 이미지에 탐지할 수 없는 워터마크를 삽입하는 알고리즘을 제안합니다. WF 서명이라는 방법을 이용해 디코더를 세밀하게 조정하여 모든 생성된 이미지에 고유한 워터마크를 루트(root)하는 방식입니다. 그러나 이 디코더는 네트워크 수준 공격에 노출될 수 있으며, 랜덤 키 생성 기반의 새로운 적대적 공격을 통해 의도된 워터마크 대신 잘못된 워터마크를 삽입하도록 모델을 혼란스럽게 만들 수 있습니다.

- **Performance Highlights**: 제안된 방법은 LDM과 같은 생성 시스템의 취약점을 예방하고 대응하는 데 중요한 임무를 강조합니다. 또한, Tamirisa et al.의 LLM을 위한 접근 방식을 LDM에 맞춰 조정하여 워터마크 삽입 프로세스를 강화하고 있습니다. 이는 실용적인 응용을 위한 모델의 기본 이미지 생성 능력을 유지하면서도 적대적 파인튜닝 공격에 대한 저항력을 높이는 데 기여합니다.



### Quantum Implicit Neural Compression (https://arxiv.org/abs/2412.19828)
Comments:
          5 pages, 4 figures

- **What's New**: 최근의 연구에서는 양자 신경망(Quantum Neural Networks, QNN)을 활용하여 신호 압축의 새로운 방법인 양자 INR(Quantum Implicit Neural Representation, quINR)을 도입했습니다. 이 접근 방식은 전통적인 비압축 코덱에 비해 이미지 압축 성능을 향상시킬 수 있으며, 특히 고주파 세부 사항의 정확도를 개선하는 것을 목표로 하고 있습니다. 실험 결과, quINR은 기존의 압축 방법들에 비해 최대 1.2dB까지 향상된 성능을 보였습니다.

- **Technical Details**: quINR 구조는 특성 임베딩과 QNN을 결합하여 좌표 값 변환을 위한 최적의 매핑을 학습합니다. 이 과정에서 평균 제곱 오차(Mean Squared Error, MSE)를 손실 함수로 사용하는데, 이는 각 좌표의 신호 값으로의 매핑을 학습하기 위해 설계되었습니다. 또한, 최적의 파라미터 세트는 압축된 형태로 저장되어, 디코더에서 신호를 재구성하는 데 사용됩니다.

- **Performance Highlights**: 실험을 통해 KITTI LiDAR 데이터셋 및 Kodak 컬러 이미지 데이터셋에서 quINR을 이용한 압축이 기존의 방법들보다 우수한 코딩 효율을 제공하는 것으로 나타났습니다. 특히, 압축률과 왜곡 성능 평가에서 quINR은 전통적인 코덱 및 기존 INR 기반 방법들보다 더 나은 성능을 보여주었습니다. 이러한 결과는 quINR이 고주파 세부 사항을 잘 복원할 수 있는 가능성을 보여줍니다.



New uploads on arXiv(cs.AI)

### Aviary: training language agents on challenging scientific tasks (https://arxiv.org/abs/2412.21154)
- **What's New**: 이 논문에서는 언어 에이전트를 위한 확장 가능한 툴인 Aviary를 소개합니다. Aviary는 언어 기반의 부분 관측 마르코프 의사결정 과정(language-grounded partially observable Markov decision processes), 즉 언어 의사결정 프로세스(language decision processes)로 에이전트를 정의합니다. 새로운 환경을 통해 DNA 조작, 과학적 문헌의 질문 답변, 단백질 안정성 공학과 같은 과학적 문제를 해결할 수 있도록 설계되었습니다.

- **Technical Details**: Aviary는 언어 에이전트의 다양한 구성 요소와 최적화 방법들을 모듈 방식으로 교환할 수 있는 소프트웨어 패키지로, stochastic computation graphs의 개념을 도입합니다. 이 논문에서는 언어 에이전트를 강화하는 과정을 통해 에이전트의 작업 효율성을 높이는 여러 최적화 알고리즘을 설정합니다. 구체적으로, 전문가 반복(expert iteration) 방식 및 인퍼런스-타임 다수결 샘플링을 통해 작은 오픈 소스 LLM 기반의 에이전트가 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, Aviary의 언어 에이전트는 전문가 및 최첨단 LLM보다 최대 100배 더 낮은 인퍼런스 비용으로 여러 과제를 수행할 수 있습니다. 특히 DNA 구성물 조작 및 과학적 문헌 질문 답변 환경에서 작은 모델이 동작을 수행할 수 있어 강력한 성능을 보여주었습니다. 이러한 결과들은 해당 연구의 적용 가능성을 높이고 언어 기반 에이전트의 활용 가능성을 한층 확장시킵니다.



### On Parallel External-Memory Bidirectional Search (https://arxiv.org/abs/2412.21104)
Comments:
          10 pages, includes conference paper and appendix

- **What's New**: 이 논문은 Parallelization and External Memory (PEM) 기법을 결합하여 단일 방향 및 양방향 최적 탐색 알고리즘의 성능을 향상시키는 새로운 프레임워크를 제시하고 있습니다. 기존의 연구는 주로 단일 방향 알고리즘에 초점을 맞추었던 반면, 본 연구는 Bidirectional Heuristic Search (
BiHS) 알고리즘 BAE*의 PEM 변형인 PEM-BAE*를 제안합니다.

- **Technical Details**: PEM-BAE*는 상태 공간의 크기를 확장할 수 있도록 설계된 프로세스를 활용하여, 문제가 어려운 경우의 평가를 가능하게 합니다. BiHS는 두 개의 열린 리스트 (Open List)를 유지하며, 시작 노드에서 목표 노드까지의 최소비용 경로를 찾는 데 초점을 두고 있습니다. BiHS 알고리즘은 일반적으로 탐색 방향과 관련된 f, g, h 값을 나타내는 다양한 변수들을 사용합니다.

- **Performance Highlights**: 실험 결과 PEM-BAE*는 A* 및 MM 알고리즘의 PEM 변형, 그리고 IDA*의 병렬 변형보다 뛰어난 성능을 보여주었습니다. 특히 PEM-BAE*는 강력한 휴리스틱을 사용할 때조차도 복잡한 문제를 해결하는 데 유리하다는 점이 강조되었습니다. 이러한 결과는 양방향 탐색 알고리즘이 여러 분야에서 단일 방향 탐색 알고리즘보다 우수함을 보여주는 중요한 성과입니다.



### UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI (https://arxiv.org/abs/2412.20977)
Comments:
          Project page: this http URL

- **What's New**: UnrealZoo는 Unreal Engine 기반의 고급스러운 3D 가상 세계의 방대한 컬렉션으로, 복잡한 오픈 월드의 다양성과 변동성을 반영하여 만들어졌다. 사용자는 손쉽게 사용할 수 있는 Python API를 통해 데이터를 수집하고, 환경을 증대시키며, 분산 학습 및 벤치마킹 등의 다양한 응용 프로그램을 활용할 수 있다. 이를 통해 Embodied AI 에이전트의 연구를 발전시키고 다양한 실험을 지원하는 것이 가능하다.

- **Technical Details**: UnrealZoo는 100개의 고품질의 현실감 넘치는 장면과 다양한 playable entities를 포함하고 있다. 이러한 환경은 아티스트에 의해 세밀하게 설계되어 실제 조명, 질감 및 동역학을 모방한다. 또한, UnrealCV의 통신 효율성을 최적화하고 다양한 요구사항에 맞춘 Gym 인터페이스를 제공하여 사용자와의 상호작용을 향상시킨다.

- **Performance Highlights**: UnrealZoo의 실험 결과는 Embodied AI 에이전트의 일반화 및 강인성을 높이고, 동적 요인을 처리하기 위한 저지연 폐쇄 루프 제어의 필요성을 강조한다. 특정 태스크로는 embodied visual navigation과 tracking을 사용하여, 복잡한 동적 환경에서 에이전트의 성능을 평가하였다. 이러한 다양한 훈련 환경은 RL 기반 및 VLM 기반 에이전트가 오픈 월드에서 직면하는 한계를 이해하는 데 중요한 통찰을 제공한다.



### Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema (https://arxiv.org/abs/2412.20942)
Comments:
          Presented at HI-AI@KDD, Human-Interpretable AI Workshop at the KDD 2024, 26th of August 2024, Barcelona, Spain

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)을 활용한 온톨로지 기반의 지식 그래프(KG) 구축 접근법을 제안합니다. 이 방법은 Competency Questions(CQ)를 생성해 지식의 범위를 파악하고, 이러한 CQs에서 관계를 추출하여 위키데이터(Wikidata)의 대응 관계로 교체하는 과정을 포함합니다. 이는 KG의 일관성과 해석 가능성을 보장하고, 최소한의 인간 개입으로 고품질의 KG를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 접근법은 네 가지 주요 단계로 구성됩니다: 1) Competency Question 생성, 2) 관계 추출 및 온톨로지 매칭, 3) 온톨로지 형식 지정, 4) KG 구축입니다. 이 과정은 LLM을 통해 시작되며, 지식 도메인에 맞는 질문을 생성해 KG 구축 작업의 범위를 정의합니다. 이어서 CQ에서 추출된 속성을 위키데이터의 속성과 매칭하여 해당 속성을 정제합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 평가 결과, 제안된 방법은 전통적인 KG 구축 방식에 비해 높은 품질의 KG를 생성하는 것이 입증되었습니다. 이 방법은 해석 가능성과 유용성을 보장하며, 위키데이터와의 상호 운용성을 통해 기존 지식 기반과의 통합을 용이하게 합니다. 따라서 이 연구는 KG 구축에 있어 확장 가능하고 효율적인 방법론을 제시합니다.



### HUNYUANPROVER: A Scalable Data Synthesis Framework and Guided Tree Search for Automated Theorem Proving (https://arxiv.org/abs/2412.20735)
- **What's New**: HunyuanProver는 Hunyuan 7B에서 파인튜닝된 언어 모델로, LEAN4를 활용한 대화형 자동 정리 증명(auto theorem proving)을 위한 새로운 프레임워크를 제시합니다. 이 모델은 데이터 희소성 문제를 해결하기 위해 저비용으로 데이터를 반복적으로 합성하는 확장 가능한 프레임워크를 디자인하였으며, 효과적인 'System 2 thinking'을 가능하게 하는 가이드 트리 탐색 알고리즘도 포함하고 있습니다. HunyuanProver는 주요 기준 점에서 최첨단 성능(state-of-the-art, SOTA)을 기록하고 있으며, 커뮤니티에 기여하기 위해 30,000개의 합성된 인스턴스의 데이터셋을 오픈소스할 계획입니다.

- **Technical Details**: HunyuanProver는 두 가지 핵심 모듈로 구성되어 있습니다: 확장 가능한 증명 데이터 생성기와 가이드 트리 탐색 알고리즘입니다. 증명 데이터 생성기는 오픈소스 데이터를 활용하여 초기 자동 형식화(autoformalizer)와 증명기(prover) 모델을 훈련합니다. 그런 다음, 새로운 증명 기초 데이터는 각 반복(iteration)에서 생성되어 증명기를 훈련시키며, 트리 탐색 알고리즘과 여러 비평(critic) 모델이 사용되어 복잡한 정리 증명 작업을 해결하기 위한 '느린 사고(slow thinking)'를 수행합니다.

- **Performance Highlights**: HunyuanProver는 miniF2F 기준 점에서 68.4%의 정확도를 기록하며 기존 최고 기록인 65.9%를 초과하였습니다. 모델은 miniF2F-test에서 4개의 IMO 진술(imo_1960_p2, imo_1962_p2, imo_1964_p2, imo_1983_p6)을 증명했습니다. 또한 데이터 양에 따른 파인튜닝 효과와 비평 모델을 활용한 트리 탐색의 유용성이 강조되었습니다.



### Predicting Long Term Sequential Policy Value Using Softer Surrogates (https://arxiv.org/abs/2412.20638)
Comments:
          23 pages, 1 figure

- **What's New**: 이 논문은 교육, 의료 및 온라인 상거래와 같은 분야에서 새 정책의 장기 가치를 추정하기 위해 단기 데이터만 사용하고자 합니다. 특히, 새로운 행동 정책과 과거 정책의 데이터를 결합하여 새로운 정책의 가치를 평가하는 새로운 두 개의 추정기를 소개합니다. 이 연구는 짧은 시간 내에 정책의 성능을 신속하게 추정할 수 있는 가능성을 탐색합니다.

- **Technical Details**: 연구자들은 새로운 정책의 단기 관찰만으로 장기 성과를 추정하기 위한 문헌을 참조하며, 다중 단계 지평의 가치 추정 작업에 대한 새로운 접근 방식을 제안합니다. 제안된 두 가지 추정기 중 하나는 doubly robust 형식에 기반하고 있으며, 특정 조건 하에 역사적 행동 정책에 대한 데이터를 사용하는 방법론을 개발했습니다. 이러한 추정기는 주어진 짧은 관찰 기반에서 기대되는 미래 수익을 예측하는 데 중점을 두고 있습니다.

- **Performance Highlights**: HIV 치료와 패혈증 치료에 대한 현실적인 시뮬레이터에서의 실험 결과, 이 방법들은 전체 지평을 기다리는 것보다 10배 더 빠르게 유용한 추정치를 제공할 수 있음을 입증하였습니다. 짧은 관찰 데이터를 통한 추정에도 불구하고 정책의 성과를 비교하는 데 충분히 정확한 결과를 도출할 수 있다는 점에서, AI 안전 응용을 위한 중요성을 강조하고 있습니다.



### The intrinsic motivation of reinforcement and imitation learning for sequential tasks (https://arxiv.org/abs/2412.20573)
Comments:
          Habilitation thesis

- **What's New**: 이번 연구는 강화 학습(reinforcement learning)과 모방 학습(imitation learning) 간의 새롭고 통합적인 접근 방식을 제안하고 있습니다. 학습자가 자율적으로 학습 커리큘럼을 선택할 수 있도록 하는 내재적 동기(intrinsic motivation)의 공통 공식을 도입하여, 다양한 작업을 배우는 데 도움을 줍니다. 이 접근 방식의 독창성은 학습자가 데이터를 단순히 수동적으로 수집하는 것이 아니라, 언제 누군가에게 도움을 요청할지 능동적으로 결정한다는 점입니다.

- **Technical Details**: 연구는 지속적인 학습(lifelong learning)과 열린 학습(open-ended learning)에 중점을 두고 있으며, 이는 무한한 차원의 상태(state), 행동(action), 및 작업(task) 공간을 포함하는 다중 작업 학습(multi-task learning)으로 특징지어집니다. 또한, 인간의 시연을 활용하는 사회적 안내된 내재적 동기(socially guided intrinsic motivation) 프레임워크를 개발하여, 학습자가 최적의 튜터로부터 시연 요청을 통해 다양한 작업을 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 모델은 학습자의 튜토리얼 질에 더 강해지며, 필요한 시연 수를 줄임으로써 빠르게 학습합니다. 연구는 인간의 행동 관찰을 통해 어떻게 학습할 수 있는지를 탐구하며, 신경망을 통해 복잡한 모터 기술(motor skills)과 행동을 학습하는 새로운 경로를 개척하고 있습니다. 이 연구 결과는 자율 로봇 개발에 있어 매우 중요한 기여를 할 수 있을 것으로 기대됩니다.



### Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning (https://arxiv.org/abs/2412.20505)
Comments:
          4 pages, 2 figures, accepted by The 1st Workshop on AI for Urban Planning (AAAI 2025's Workshop)

- **What's New**: 이번 연구에서는 도시화 과정에서의 도전 과제를 해결하기 위해 사이클형 도시 계획(Cyclical Urban Planning, CUP)을 제안합니다. 이 새로운 패러다임은 대형 언어 모델(LLMs)을 활용하여 도시 계획을 지속적으로 생성하고 평가하며, 정교화하는 순환 프로세스를 포함합니다. 이러한 접근 방식은 전통적인 방법의 한계를 극복하고 도시 환경의 변화에 유연하게 대응할 수 있는 가능성을 제시합니다.

- **Technical Details**: CUP 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Planning, 여기서 LLM 에이전트가 맥락 데이터에 기반하여 도시 계획을 생성하고 수정합니다; (2) Living, 도시 환경에서 거주자의 행동과 상호작용을 시뮬레이션하는 과정입니다; (3) Judging은 계획의 효과성을 평가하고 개선을 위한 피드를 제공합니다. 이러한 반복적인 프로세스를 통해 도시 계획의 효율성을 높이는 방법을 모색합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험을 통해 제안한 프레임워크의 유효성을 입증했습니다. 실험 결과, CUP는 지속적이고 적응적인 계획 프로세스의 구현을 가능하게 하여 도시의 역동적인 요구에 보다 잘 부응할 수 있는 잠재력을 보여주었습니다. 특히, 베이징의 휴롱관 커뮤니티를 사례 연구로 활용하여 도시 계획 결과를 개선하는 데 기여할 수 있음을 보였습니다.



### A Comprehensive Framework for Reliable Legal AI: Combining Specialized Expert Systems and Adaptive Refinemen (https://arxiv.org/abs/2412.20468)
Comments:
          16 pages and 5 figures

- **What's New**: 이 논문에서는 법률 전문 분야에서 인공지능(AI)의 역할이 어떻게 진화하고 있는지에 대해 논의하며, 문서 검토, 연구 및 계약 초안 작성과 같은 작업을 간소화할 잠재력에 주목하고 있습니다. 그러나 AI 모델의 '환각(hallucinations)' 문제 즉, 부정확하거나 잘못된 정보를 생성하는 이슈가 존재하여 법률 분야에서의 신뢰성을 저해하고 있습니다.

- **Technical Details**: 이 논문은 AI 기반 법률 서비스를 향상시키기 위해 지식 기반 아키텍처와 전문가 시스템의 혼합을 결합한 새로운 프레임워크를 제안합니다. 이 프레임워크는 특정 법률 분야에 초점을 맞춘 전문 모듈을 활용하여 보다 정확하고 상황에 맞는 법률 조언 및 분석을 제공합니다. 또한, Retrieval-Augmented Generation (RAG), Knowledge Graphs (KG), Reinforcement Learning from Human Feedback (RLHF)와 같은 고급 AI 기술을 활용하여 시스템의 정확성을 높입니다.

- **Performance Highlights**: 제안된 접근 방식은 기존 AI 모델 대비 성능 향상을 나타내며, 법률 작업 수행에서 개선된 성능을 보이고 더 접근 가능하고 저렴한 법률 서비스를 제공할 수 있는 확장 가능한 솔루션을 제시합니다. 실험 결과는 AI 모델이 법적 기준에 부합하며, 법정 사안에 대한 의존도를 줄이면서 보다 신뢰할 수 있는 법률 조언을 제공할 수 있음을 보여줍니다.



### High-fidelity social learning via shared episodic memories enhances collaborative foraging through mnemonic convergenc (https://arxiv.org/abs/2412.20271)
Comments:
          15 pages, 5 figures

- **What's New**: 이번 연구에서는 집단 식량 탐사에서의 사건 기반 기억(episodic memory)과 사회 학습(social learning) 간의 상호 관계를 탐구합니다. 특히, 행동 시퀀스를 완전히 공유할 수 있는 Sequential Episodic Control (SEC) 에이전트를 사용하여 사회 학습의 빈도와 정확성이 집단 식량 탐사 성능에 미치는 영향을 분석합니다. 또한, 사회 학습이 집단 내 사건 기억의 내용 및 분포에 미치는 효과를 조명합니다.

- **Technical Details**: 본 연구에서 사용된 Sequential Episodic Control (SEC) 에이전트는 기억이나 사례 기반 학습 프로세스를 모델링하고 있으며, 특히 포유류의 해마(hippocampus)에 의해 지원됩니다. 실험에서는 에이전트가 서로 간에 행동 시퀀스를 공유하고, 기억 길이의 변화를 통해 사회 학습의 효과를 분석합니다. 또한 전송 속도(transfer rate)와 전송 노이즈(transfer noise)를 조절하여 사회 학습 과정에서의 속도와 정확성 간의 균형을 살펴봅니다.

- **Performance Highlights**: 고충실도(high-fidelity) 사회 학습이 자원의 수집 효율성과 분포를 일관되게 향상시키는 반면, 저충실도(low-fidelity) 학습은 효과적인 결과를 내지 못합니다. 기록된 결과는 고충실도 사회 학습이 기억 집단 정렬과 공평한 자원 분배를 촉진함을 보여주며, 저충실도 조건에서는 기억 다양성을 증가시키지만 성과 향상으로 이어지지 않는 것으로 나타났습니다. 최적의 사건 기억 길이를 식별하여 성과가 정체되는 사례도 발견하였습니다.



### BaiJia: A Large Scale Role-Playing Agent Corpus of Chinese Historical Charcaters (https://arxiv.org/abs/2412.20024)
- **What's New**: BaiJia라는 혁신적인 대규모 역사적 캐릭터 역할 놀이 에이전트 데이터 집합이 소개되었습니다. 이 데이터 집합은 다양한 중국 역사 인물에 대한 정보를 통합하여 LLMs(대형 언어 모델)가 AI 주도 역사 역할 놀이에 참여할 수 있도록 설계되었습니다. BaiJia는 과거의 산재된 역사 텍스트 기록의 문제를 해결하며, 인물의 전기적 정보나 문학적 작품 등을 포함하고 있습니다.

- **Technical Details**: BaiJia 데이터 집합은 데이터 수집, 대화 생성 및 모델 평가의 세 가지 주요 단계로 구성됩니다. 이 과정에서 우리는 다섯 개의 주요 중국 왕조(Tang, Song, Yuan, Ming, Qing)에서 총 19,281명의 역사적 인물의 이력서를 수집하고, 이를 바탕으로 대화를 생성하였습니다. 또한, LLMs의 역할 놀이 능력을 평가하기 위해 15개의 주제 측면에서 질문 데이터 집합을 구축하여 평가 기준을 마련했습니다.

- **Performance Highlights**: BaiJia 데이터 집합을 활용한 여러 LLM의 실험 결과, LLM에 인물 이력서 정보를 통합함으로써 역할놀이 능력이 각 평가 차원에서 유의미하게 향상되었습니다. 특히, 역할 놀이에 특화된 LLM조차도 역사적 캐릭터 표현에 한계를 보였지만, BaiJia는 이러한 한계를 극복하는 데 기여하고 있음을 입증하였습니다. 우리의 데이터 집합은 맥락적으로 일관되고 사실적인 대화를 생성하는 데 향상된 도움을 주며, Character Consistency 및 Culture & Historical Appropriateness 차원에서 가장 큰 개선을 보였습니다.



### Action-Agnostic Point-Level Supervision for Temporal Action Detection (https://arxiv.org/abs/2412.21205)
Comments:
          AAAI-25. Technical appendices included. 15 pages, 3 figures, 11 tables

- **What's New**: 이번 논문에서 제안된 action-agnostic point-level (AAPL) supervision은 일부분만 주석이 달린 데이터셋으로 정확한 행동 인스턴스 탐지를 이루기 위해 개발되었습니다. 이 방법은 주석자가 비디오의 모든 행동 인스턴스를 찾지 않고도 샘플링한 비디오 프레임을 주석할 수 있도록 합니다. 또한, AAPL 레이블을 효과적으로 활용할 수 있는 탐지 모델과 학습 방법도 제안됩니다.

- **Technical Details**: AAPL supervision의 주석 파이프라인은 비행 동작을 무관하게 프레임을 샘플링하고, 그 후에 수동으로 주석을 다는 두 단계로 구성됩니다. 또한, AAPL 레이블은 단일 임의의 시간 지점만 지정하는 기존의 포인트 레벨 주석과 달리, 다수의 레이블을 사용하여 행동 인스턴스에 대한 보다 포괄적인 정보를 전달할 수 있습니다. 비록 AAPL supervision이 모든 행동 인스턴스를 탐지하지 못할 가능성이 있지만, 이는 전체적인 성능 향상을 위한 유효한 방법으로 고려됩니다.

- **Performance Highlights**: AAPL supervision을 사용하여 다양한 데이터셋(THUMOS '14, FineAction, GTEA, BEOID, ActivityNet 1.3)에서 실시된 실험 결과는, 제안된 접근 방식이 비디오 수준과 포인트 수준 주석 모두에서 이전 방법들과 경쟁하거나 이를 초월하는 성능을 보여주었습니다. 특히, 주석 비용과 탐지 성능 간의 균형을 시각적으로 비교했으며, 주석이 일부 달린 프레임으로만 훈련해도 이전 연구와 비슷한 경쟁력 있는 결과를 얻을 수 있음을 발견했습니다.



### Adversarial Attack and Defense for LoRa Device Identification and Authentication via Deep Learning (https://arxiv.org/abs/2412.21164)
- **What's New**: 이 논문은 LoRa 네트워크의 보안을 강화하기 위해 딥 러닝(Deep Learning)을 활용하는 새로운 접근 방식을 제안합니다. 특히, LoRa 장치 식별과 합법적인 장치와 악성 장치 분류라는 두 가지 중요한 작업을 동시에 수행할 수 있는 다중 작업(Multi-task) 분류기를 개발했습니다. 또한, 합법적인 LoRa 신호를 학습하여 악성 신호를 생성하는 방법을 채택했습니다.

- **Technical Details**: 연구에서는 심층 신경망(DNNs)으로 합성 곱 신경망(Convolutional Neural Network, CNN)과 피드포워드 신경망(Feedforward Neural Network, FNN) 아키텍처를 사용하여 LoRa 신호를 분석하는 두 가지 작업을 구현합니다. 논문에서는 입력 샘플을 조작하기 위한 다양한 적대적 공격(Adversarial Attacks) 기법인 Fast Gradient Sign Method (FGSM)을 사용하여 DNN 모델의 취약성을 평가합니다. 또한, 적대적 훈련(Adversarial Training)을 통한 방어 메커니즘도 제안하고 있습니다.

- **Performance Highlights**: 결과적으로, LoRa 신호 분류 작업이 적대적 공격에 얼마나 취약한지를 수치적으로 정량화하며, 다양한 공격 유형 및 방어 전략들이 모델의 정확도에 미치는 영향을 분석합니다. 본 연구는 IoT 애플리케이션의 보안을 강화하기 위한 보강 방안을 제공하며, 모델의 일반화 성능을 저하시키지 않고 공격에 대한 저항력을 높이는 방법을 보여줍니다.



### Open RAN-Enabled Deep Learning-Assisted Mobility Management for Connected Vehicles (https://arxiv.org/abs/2412.21161)
Comments:
          Accepted for publication in ICOIN 2025

- **What's New**: 최근 이 논문에서 제안하는 Connected Vehicles (CVs)의 개선 방법은 Open RAN (O-RAN) 아키텍처와 딥 러닝 모델을 활용하여 품질(QoS) 저하를 방지하는 것입니다. 기존의 3GPP 핸드오버(HO) 방식에 비해 더 나은 성능과 낮은 지연 시간을 달성하는 새로운 프레임워크의 효과를 보여주었습니다.

- **Technical Details**: 이 연구에서 도입한 O-RAN Software Community (OSC) 플랫폼을 통해 xApps를 개발하고, OMNeT++ 시뮬레이터와 통합하여 실제 데이터셋을 기반으로 한 성능 평가를 실시했습니다. 특히, near-Real-Time RIC와 gNodeB 간의 통신을 통해 통합된 경량화된 네트워크 관리와 손쉬운 핸드오버 결정을 이루어냈습니다.

- **Performance Highlights**: 실험 결과, 도시 시나리오에서의 비디오 스트리밍 및 OTA 업데이트를 포함한 다양한 사용 사례에서 제안된 xApp이 표준 3GPP HO 절차에 비해 지연시간과 통신 품질에서 현저히 개선된 성과를 보였습니다. 이러한 결과는 CV 서비스의 안정성과 효율성을 크게 향상시킬 잠재력을 지니고 있습니다.



### PyG-SSL: A Graph Self-Supervised Learning Toolk (https://arxiv.org/abs/2412.21151)
- **What's New**: Graph Self-Supervised Learning (SSL)은 최근 몇 년간 중요한 연구 분야로 떠올랐습니다. 자가 지도 학습을 통해 복잡한 그래프의 구조와 특성을 학습할 수 있는 능력을 가진 그래프 SSL 모델은 성능 향상, 일반화 개선, 강건성 증가를 이루어냈습니다. 그러나 이러한 방법의 현재 구현은 복잡한 그래프 구조, 일관성이 없는 평가 지표, 재현성 문제로 인해 초보자와 실무자가 접근하는 데 큰 도전이 되고 있습니다.

- **Technical Details**: 이 논문에서는 PyTorch를 기반으로 한 Graph SSL 툴킷인 PyG-SSL을 소개합니다. 이 툴킷은 데이터셋 로딩, 하이퍼파라미터(configuration) 설정, 모델 학습 및 다양한 다운스트림 작업에 대한 성능 평가를 포함하는 통합 프레임워크를 제공합니다. 또한, 다양한 그래프 데이터셋에 대한 최적의 하이퍼파라미터와 초보자 친화적인 튜토리얼도 제공하여 결과 재현을 용이하게 합니다.

- **Performance Highlights**: PyG-SSL은 다양한 그래프 SSL 알고리즘을 지원하며, 동종 그래프뿐만 아니라 이종 그래프와 분자 그래프도 포함됩니다. 기존의 다른 툴킷들과 비교했을 때, 우리의 툴킷은 사용자 친화적인 튜토리얼과 다양한 그래프 향상 기법을 제공하여 초보자들이 접근하는 데 어려움이 없도록 만드는 데 중점을 두고 있습니다. 또한, 수많은 그래프 자가 지도 학습 방법의 성능을 평가하여 특정 작업에 적합한 방법 선택에 대한 통찰력을 제공합니다.



### Facilitating large language model Russian adaptation with Learned Embedding Propagation (https://arxiv.org/abs/2412.21140)
Comments:
          Preprint version of an article published in the Journal of Language and Education. Copyright held by the owner/author(s). Publication rights licensed to the Journal of Language and Education

- **What's New**: 본 논문에서는 Learned Embedding Propagation (LEP)이라는 새로운 어댑테이션 파이프라인을 제안하고 있습니다. 기존의 instruction-tuning 과정 없이도 새로운 언어 지식을 기존의 instruct-tuned 모델에 직접 삽입할 수 있는 방법입니다. 이 접근법은 데이터 요구량을 줄이고, 언어 특정 모델의 비용을 줄이며, 더 나은 성능을 제공합니다.

- **Technical Details**: LEP는 기존 모델의 지식에 의존하여 언어 적응을 실시하며, 데이터의 양을 최소화하는 방법을 탐구합니다. 연구자들은 BPE, Unigram, Extension, Optimization과 같은 다양한 토큰화 방법을 실험하고, 이러한 방법들이 러시아어의 형태론(linguistic morphology)과 얼마나 잘 일치하는지를 평가했습니다. 또한 새로운 임베딩을 initialization 하는 방법도 제시하여, 이전 토큰화된 데이터의 평균을 사용하는 기법이 효과적임을 보여줍니다.

- **Performance Highlights**: LEP는 Mistral-7B 및 LLaMa-3-8B 모델에 대해 4가지 러시아어 토큰화 변형을 실험하며, 기존의 instruction-tuning 방법들과 경쟁력을 갖추고 있다는 것을 입증했습니다. 실험 결과, LEP는 OpenChat 3.5 및 LLaMa-3-8B-Instruct와 비슷한 수준의 성능을 달성했으며, 후속 조정을 통해 더 나은 과제 해결 능력을 나타냈습니다.



### Exploring and Controlling Diversity in LLM-Agent Conversation (https://arxiv.org/abs/2412.21102)
Comments:
          Accepted for the AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration

- **What's New**: 본 논문에서는 Multi-Agent communication에서 다양성(Diversity)을 제어하고 탐구하는 새로운 방법인 Adaptive Prompt Pruning (APP)를 제안합니다. APP는 발화 생성 프롬프트의 내용을 동적으로 조정하여 다양성을 단일 파라미터인 λ(램다)를 이용해 제어합니다. 결과적으로, APP는 모델과 데이터셋 전반에서 효과적으로 출력의 다양성을 조절한다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 다양한 설계를 통해 프롬프트 내용과 출력 다양성 간의 관계를 포괄적으로 분석했습니다. 주요 구성 요소로는 환경 설명, 에이전트 프로필, 대화 이력 등이 있으며, 이들은 출력의 다양성에 영향을 미칩니다. APP는 temperature sampling 및 top-p sampling과 같은 기존의 다양성 관리 기술과 호환되어, 이론적으로도 다재다능한 도구입니다.

- **Performance Highlights**: APP는 대화 생성 후 정보 누락으로 인한 불일치를 해소하기 위해 수정 단계(correction step)를 도입하여, 다양성 증진과 출력 일관성(consistency) 간의 균형을 효과적으로 맞춥니다. 실험 결과, 다양한 프롬프트 구조가 출력의 다양성에 미치는 영향을 조사한 바, 구성 요소의 순서와 길이가 다양성에 중대한 영향을 미친다는 사실을 발견하였습니다.



### Towards Effective Discrimination Testing for Generative AI (https://arxiv.org/abs/2412.21052)
Comments:
          38 pages, 9 tables, 8 figures

- **What's New**: 이 논문은 Generative AI (GenAI) 모델의 차별 행동을 규제하는 데 있어 새로운 도전과제를 제기합니다. 현재의 편향성 평가 방법과 규제 목표 간의 중요한 격차가 존재하여, 실제로는 차별적이지만 보고된 바에 따르면 공정한 GenAI 시스템의 배포를 허용할 가능성을 지적합니다. 이를 해결하기 위해, 저자들은 법률 및 기술 문헌을 연결하여 차별 평가 방법의 불일치를 식별하고 있으며, 네 개의 사례 연구를 통해 이러한 문제가 어떻게 실제 배포에서 차별적 결과를 초래하는지를 보여줍니다.

- **Technical Details**: Generative AI 모델은 전통적인 ML 분류 모델과는 달리, 출력 결과가 쉽게 할당 결정에 매핑되지 않아 전통적인 차별 방지 원칙인 disparate impact를 적용하기 어렵습니다. 또한, GenAI의 출력 변동성이 높아 성능 및 편향 측정이 매우 가변적으로 나타납니다. 사용자가 모델의 (hyper)parameters를 조정하거나 모델을 수정할 수 있는 기능 또한 차별 평가는 복잡해지며, 전통적인 법적 틀과 편향성 테스트 접근 방식의 효과를 저하시키고 있습니다.

- **Performance Highlights**: 저자들은 각 사례 연구에서 공정성 테스트 기준이 규제 목표와 일치하지 않음을 보여주며, 특정 그룹에서 채용 후보자를 공정하게 선발하지 못할 수 있음을 증명합니다. 또한, 인기 있는 편향 테스트 기술의 변동성이 불공정한 모델이 현재의 보고 기준을 통과할 수 있게 함을 지적합니다. 이 논문은 GenAI 시스템의 성과 평가 및 차별 방지 문제를 해결하기 위한 실질적인 조정 방안을 제시하며, 향후 연구가 이러한 우려를 완화하는 데 기여할 수 있기를 희망합니다.



### Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defens (https://arxiv.org/abs/2412.21051)
Comments:
          7 pages; In submission

- **What's New**: 이 논문에서는 Generative Foundation Models(GFMs), 특히 Large Language Models(LLMs)의 발전이 클라우드 보안의 새로운 해결책이 될 수 있음을 제시합니다. LLM-PD라는 새로운 능동적 방어 아키텍처를 설계하여 다양한 사이버 위협을 예방하는 방법을 설명하고 있으며, 기존 방어 방법과의 비교를 통해 방어 효과성과 효율성을 보여줍니다.

- **Technical Details**: LLM-PD는 데이터 수집, 평가, 결정, 배치 및 피드백의 다섯 가지 핵심 구성 요소로 이루어져 있습니다. 이 아키텍처는 기존의 보안 기능과 LLMs의 통합을 통해 클라우드 네트워크에서 지능적이고 효율적인 보안을 제공합니다. 특히, LLM의 강력한 언어 이해 능력과 데이터 분석, 행동 계획 기능을 이용하여 클라우드 환경에서 다양한 공격 시나리오에 능동적으로 대응합니다.

- **Performance Highlights**: 실험 결과 LLM-PD는 다양한 DoS 공격에 대한 높은 생존율, 시간 효율성, 성공률을 자랑합니다. 특히, 기존의 다른 방법들과 비교했을 때 뛰어난 성공률을 기록하였으며, 클라우드에서 보안 방어에 대한 새로운 가능성을 제시합니다. 이 대표 사례는 LLM-PD의 다양한 클라우드 보안 시나리오에 대한 적응력을 강조하고 있습니다.



### TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization (https://arxiv.org/abs/2412.21037)
Comments:
this https URL

- **What's New**: 이번 논문에서는 515M 매개변수를 가진 효율적인 텍스트-오디오(TTA) 생성 모델인 TangoFlux를 소개합니다. TangoFlux는 단일 A40 GPU에서 3.7초 만에 최대 30초 짜리 44.1kHz 오디오를 생성할 수 있습니다. TTA 모델의 정렬 과정에서 발생하는 문제를 해결하기 위해 새로운 접근법인 CLAP-Ranked Preference Optimization (CRPO)를 제안하여 오디오 생성의 품질을 대폭 향상시켰습니다. 모든 코드와 모델이 오픈 소스로 공개되어 후속 연구를 지원합니다.

- **Technical Details**: TangoFlux는 FluxTransformer 블록으로 구성되며, 이는 Diffusion Transformer(DiT) 및 Multimodal Diffusion Transformer(MMDiT)에 따라 조건부 텍스트 프롬프트와 지속 시간 임베딩을 사용하여 44.1kHz에서 최대 30초의 오디오를 생성합니다. TangoFlux의 훈련 파이프라인은 세 단계로 나뉘며, 이는 사전 훈련(pre-training), 미세 조정(fine-tuning), 그리고 선호 최적화(preference optimization)로 구성됩니다. 또한, CRPO를 통해 새로운 합성 데이터를 반복적으로 생성하고 선호 쌍을 구성하여 오디오 생성 결과의 품질을 높입니다.

- **Performance Highlights**: TangoFlux는 기존의 오디오 선호 데이터셋 대비 우수한 성능을 자랑하며, 벤치마크와 비분배(out-of-distribution) 프롬프트에서 최첨단 성능을 달성합니다. 이 모델은 생성된 오디오의 품질을 높이기 위해 사용된 수정된 손실 함수(modified loss function)을 통한 반복 최적화가 효과적이라는 것을 입증했습니다. 또한, TangoFlux는 비독점 데이터로 훈련되어 다양한 길이의 오디오 생성을 지원하는 뛰어난 능력을 보여줍니다.



### Plancraft: an evaluation dataset for planning with LLM agents (https://arxiv.org/abs/2412.21033)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트를 위한 멀티 모달 평가 데이터셋인 Plancraft를 소개합니다. Plancraft는 Minecraft의 crafting GUI를 기반으로 하여, 텍스트 전용 및 멀티 모달 인터페이스를 포함하고 있습니다. 이를 통해 도구 사용과 Retrieval Augmented Generation (RAG) 평가를 위해 Minecraft Wiki를 활용하며, 현대 에이전트 아키텍처의 다양한 구성 요소를 실험하고 있습니다.

- **Technical Details**: Plancraft는 다양한 복잡성과 길이의 계획 문제를 포함하고 있으며, 일부는 고의적으로 해결할 수 없도록 설계되었습니다. 이를 통해 LLM 및 VLM(비주얼 언어 모델)의 계획 문제를 평가하고, LLM 기반 에이전트의 성능을 수작업 계획자와 비교합니다. 데이터셋에서는 텍스트 전용 모델과 멀티 모달 에이전트를 평가하고, 전문 계획안에 대한 미세 조정의 영향을 실험합니다.

- **Performance Highlights**: Plancraft의 성과 지표는 성공률뿐만 아니라 LLM의 계획이 수작업 솔루션에 얼마나 가까운지를 평가합니다. 기존의 LLM 평가는 성공률에만 집중했으나, Plancraft는 실제 상황의 복잡성을 포착하기 위해 더 세분화된 평가를 제공합니다. 연구 결과, LLM들은 Plancraft가 제시하는 계획 문제에서 어려움을 겪고 있으며, 이들의 능력을 향상시키기 위한 제안도 포함되어 있습니다.



### Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale via Principled Criteria (https://arxiv.org/abs/2412.21006)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 중간 추론 경로를 줄이는 새로운 방법인 문장 단위의 근거 감소 훈련 프레임워크를 제안합니다. 이 접근법은 중복된 추론 문장을 제거하는데 목표를 두고 있으며, 기존의 토큰 단위 감소 방식과는 달리 모델의 성능을 유지하면서 생성 길이를 줄일 수 있음을 입증합니다.

- **Technical Details**: 제안된 방법은 ‘verbosity’라는 기준을 활용하여 비필요한 근거 문장을 제거하고, 이를 통해 LLM의 추론 능력을 보존하면서도 근거 생성을 줄이는 제안입니다. 초기 추론 단계에서의 근거 문장들이 중첩성을 유발함을 경험적으로 증명하고, 문장 단위로 중복성을 평가하는 방법을 도입하여 LLM 성능 저하 없이 효율적인 경로 감소를 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 다양한 모델과 태스크에서 평균 17.15%의 생성 비용이 감소하는 것으로 나타났습니다. 이는 LLMs의 근거 문장을 효율적으로 관리함으로써 보다 효과적인 성능을 유지할 수 있음을 시사하며, 실제 적용 시 효율성 및 안정성을 높일 수 있는 방법론으로 주목받고 있습니다.



### LEASE: Offline Preference-based Reinforcement Learning with High Sample Efficiency (https://arxiv.org/abs/2412.21001)
Comments:
          14 pages, 4 figures

- **What's New**: 이번 논문에서 제안한 LEASE 알고리즘은 오프라인 환경에서의 선호 기반 강화 학습(PbRL)의 샘플 효율성을 높이기 위한 혁신적인 접근 방식을 제공합니다. LEASE는 학습된 전이 모델을 활용하여 라벨이 없는 데이터에서 선호 데이터를 생성하고, 불확실성 인식 메커니즘을 통해 올바른 보상 모델을 유지할 수 있도록 설계되었습니다. 이를 통해 기존 방법들보다 적은 양의 선호 데이터로도 비슷한 성능을 달성할 수 있습니다.

- **Technical Details**: LEASE 알고리즘은 데이터 선택 메커니즘을 통해 높은 신뢰도와 낮은 불확실성을 가진 데이터를 선별합니다. 이 선별 과정을 통해 잘못된 라벨링이 발생할 가능성을 줄이고, 보상 모델의 안정성을 향상시킵니다. 또한, 상태-행동 쌍을 기반으로 한 보상 모델의 일반화 경계를 개발하여 이론적으로 보상 정확도에 영향을 미치는 요소들을 분석합니다.

- **Performance Highlights**: 실험 결과, LEASE는 D4RL 벤치마크에서 온라인 상호작용 없이도 비슷한 성능을 발휘하는 것으로 나타났습니다. LEASE는 이전의 방법들에 비해 적은 양의 데이터로도 우수한 성능을 보이며, 이론적으로도 정책 개선의 보장을 제공합니다. 이는 오프라인 PbRL 분야에서 중요한 진전을 나타내며, 향후 연구에도 큰 기여를 할 것으로 기대됩니다.



### KARPA: A Training-free Method of Adapting Knowledge Graph as References for Large Language Model's Reasoning Path Aggregation (https://arxiv.org/abs/2412.20995)
Comments:
          23 pages, 6 figures

- **What's New**: 이번 연구에서는 Knowledge graph Assisted Reasoning Path Aggregation (KARPA)라는 새로운 프레임워크를 제안합니다. KARPA는 LLM의 글로벌 계획(Planning) 능력을 활용하여 더 효율적이고 정확한 지식 그래프(QG) 추론을 가능하게 합니다. 기존의 KG 기반 질문 응답 방식의 제한점을 극복하며 별도의 학습 없이 다양한 LLM 아키텍처에 적응할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: KARPA는 세 가지 단계로 구성됩니다. 첫 번째로 LLM의 글로벌 계획 능력을 사용하여 관계 경로(relation path)를 사전 계획(pre-planning)하고, 두 번째로 임베딩(embedding) 모델을 통해 의미적으로 관련성 있는 경로를 매칭합니다. 마지막으로 이러한 경로에 대한 추론(reasoning)을 통해 답변을 생성합니다.

- **Performance Highlights**: KARPA는 KGQA 작업에서 최고 수준의 성능(state-of-the-art performance)을 달성하며, 높은 효율성과 정확성을 보여줍니다. 실험 결과에 따르면 KARPA는 기존 방법에 비해 뛰어난 성능을 발휘하여 KGQA의 새로운 가능성을 제시합니다. 또한 코드가 Github에 공개될 예정입니다.



### Conservation-informed Graph Learning for Spatiotemporal Dynamics Prediction (https://arxiv.org/abs/2412.20962)
- **What's New**: 본 논문에서는 보존 법칙(consservation law)을 고려한 그래프 신경망(CiGNN)을 제안합니다. 이 모델은 한정된 학습 데이터를 바탕으로 복잡한 시공간 역학(spatiotemporal dynamics)을 학습할 수 있는 투명한 학습 프레임워크입니다. CiGNN은 일반적인 보존 법칙에 의해 설계되며, 이는 대칭(symmetry) 원리에 따라 정보 전파를 가능하게 합니다.

- **Technical Details**: CiGNN은 다중 스케일 공간을 통해 보존정보와 비보존정보를 전달합니다. 이 네트워크는 잠재적 시간 통합기(latent time integrator)를 설계하여 긴 시간 예측의 안정성과 정확성을 향상시킵니다. 또한, 각 SymMPNN 레이어에서 추출된 잠재 표현의 순차적 의존성을 모델링하여 시공간 동적의 정밀한 학습을 도모합니다.

- **Performance Highlights**: CiGNN은 합성 데이터(synthetic datasets)와 실제 데이터를 통해 다양한 시공간 시스템에서 높은 성능을 입증하였습니다. 제한된 학습 데이터로도 놀라운 정확도와 일반화 능력을 보여주며, 복잡한 시공간 역학의 예측 학습에 손쉽게 적용 가능합니다. 기존의 여러 대표 모델보다 우수한 성능을 자랑합니다.



### Rise of Generative Artificial Intelligence in Scienc (https://arxiv.org/abs/2412.20960)
Comments:
          26 pages, 4 tables, 1 figures, 1 appendix figure

- **What's New**: 최근 생성 인공지능(Generative Artificial Intelligence, GenAI)가 과학 연구에서 도구로 급속히 확산되고 있습니다. 본 연구는 OpenAlex를 활용한 실증 분석을 통해 2017년부터 2023년까지의 GenAI 및 기타 AI 학술 출판물의 성장을 조사합니다. 연구 결과는 GenAI가 컴퓨터 과학을 넘어 다양한 과학 분야에서 사용되고 있음을 보여줍니다.

- **Technical Details**: 분석은 GenAI 출판물의 성장 패턴, 연구 분야 내 확산, 그리고 생성 AI에 대한 과학 연구의 지리적 분포를 포함합니다. 연구팀의 규모와 국제 협력 또한 조사하여 GenAI가 다른 AI 기술과 비교할 때 다른 협력 양상을 보이는지를 검토하였습니다. 분석 결과 GenAI 연구팀은 다른 AI 분야에 비해 다소 작은 규모를 가지는 경향이 있습니다.

- **Performance Highlights**: 미국 연구자들은 전 세계 GenAI 출판물의 거의 40%를 차지하며, 중국이 그 뒤를 잇고 있습니다. 여러 중소형 선진 경제국들도 연구 출판물에서 비교적 높은 수준의 GenAI 활용을 보이고 있습니다. 또한 최근의 지정학적 긴장에도 불구하고 GenAI 연구는 다른 AI 기술과 유사한 수준의 국제 협력을 지속하고 있습니다.



### HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization (https://arxiv.org/abs/2412.20924)
Comments:
          Accepted by IEEE Transactions on Medical Imaging

- **What's New**: 이번 논문에서는 'HisynSeg'라는 새로운 약한 지도 하의 의미 구분(segmentation) 프레임워크를 제안합니다. 이 프레임워크는 이미지 혼합 합성을 기반으로 하여 기존의 CAM 기반 접근법의 단점을 극복하고, 약한 감독 작업을 완전 감독 작업으로 변환합니다. 이를 통해, 픽셀 정확도가 크게 개선되었습니다.

- **Technical Details**: HisynSeg에서는 Mosaic 변환 및 Bézier 마스크 생성을 기반으로 하는 두 가지 이미지 혼합 합성 전략이 도입됩니다. 또한 생성된 이미지의 진정성을 보장하기 위한 이미지 필터링 모듈이 개발되었습니다. 마지막으로, 자가 감독(self-supervised) 일관성 정규화(consistency regularization)를 통해 실제 이미지를 모델 훈련에 통합하여 과적합(overfitting) 문제를 완화합니다.

- **Performance Highlights**: 세 가지 데이터셋에서 수행된 실험 결과, HisynSeg 프레임워크는 최신 기술(SOTA)과 비교하여 성능이 크게 향상되었습니다. 특히 약한 감독 학습 환경에서의 픽셀 수준 세그멘테이션 성능이 현저히 개선된 것으로 나타났습니다. 이러한 성과는 실제 병리학적 이미지를 통한 자동화된 진단의 잠재력을 더욱 높이는 데 기여합니다.



### WalkVLM:Aid Visually Impaired People Walking by Vision Language Mod (https://arxiv.org/abs/2412.20903)
- **What's New**: 이 논문에서는 약 2억 명이 시각적 장애를 겪는 현상을 감안하여 인공지능 기술을 활용한 보행 보조 시스템의 필요성을 제기합니다. 기존 방법들이 자가 구축된 질문-답변 데이터셋을 기반으로 한 반면, 저자들은 이를 극복하고 보행 안내를 위한 통합된 훈련 및 테스트 기준을 제시하였습니다. 특히, 1만 2천 개 비디오-매뉴얼 주석 쌍이 포함된 Walking Awareness Dataset (WAD)을 새롭게 개발하여 다양한 장애인을 위한 연구 기반을 마련하였습니다.

- **Technical Details**: WalkVLM 모델은 계층적 계획을 위해 'chain of thought' 방식을 활용하여 간결하고 유익한 알림을 생성하고, 시간 인지 적응 예측을 통해 알림의 시간적 중복성을 줄입니다. 이 방식은 실시간 비디오 스트리밍에서 정보를 제공하며, 시각 장애인들에게 보다 유리한 조건의 보행 지원을 가능하게 합니다. 이를 통해 VLM들이 갖는 응답 중복 및 낮은 추론 효율성 문제를 해결하고자 했습니다.

- **Performance Highlights**: 실험 결과, WalkVLM은 다른 VLM 모델들과 비교하여 더 간결한 알림을 생성하고 비디오 스트리밍 처리에서 뛰어난 시간 적응성을 발휘하는 것으로 나타났습니다. 이 연구는 시각 장애인을 위한 실질적인 보행 안내 응용처의 발전을 위한 중요한 기초를 마련합니다. 저자들은 새로운 기준을 설정하고 오후 VLM을 통해 보행 안내를 가능하게 한 첫 번째 연구로 자리매김하고자 합니다.



### ILDiff: Generate Transparent Animated Stickers by Implicit Layout Distillation (https://arxiv.org/abs/2412.20901)
- **What's New**: 본 논문에서는 ILDiff라는 새로운 방법을 제안하여 애니메이션 스티커의 투명 채널을 생성합니다. 이전의 비디오 매팅(video matting) 및 확산 기반(diffusion-based) 방법의 단점을 해결하며, 특히 세미 오픈 영역(semi-open area) 처리와 시계열 정보(temporal information) 고려의 부족 문제를 해결하고자 합니다. 여기에는 SAM(Segmentation-Aware Matting) 접근법을 통해 암시적인 레이아웃 정보(implicit layout information)를 추가하여 더 정교한 결과를 제공합니다.

- **Technical Details**: ILDiff는 임시 레이아웃 증류(implicit layout distillation) 방법을 이용하여 투명 애니메이션 채널을 생성합니다. 이 방법은 임시 모델링(temporal modeling) 구조를 갖추어 시계열 처리 능력을 부여함으로써 기존 확산 기반 방법들이 직면했던 지역 후레임 떨림(local flickering) 문제를 개선합니다. 또한, TASD(Transparent Animated Sticker Dataset)라는 0.32M 고품질 샘플로 구성된 데이터셋을 구축하여 관련 분야에 지원을 제공합니다.

- **Performance Highlights**: 실험 결과, ILDiff는 Matting Anything 및 Layer Diffusion과 같은 기존 방법들과 비교하여 더 섬세하고 부드러운 투명 채널을 생성하는 것으로 나타났습니다. 이 연구는 고품질 투명 애니메이션 스티커 생성의 가능성을 열어주며, 향후 다양한 애플리케이션에서 활용될 수 있습니다. 제공되는 코드와 데이터셋은 연구자들에게 추가적인 지원을 제공할 것입니다.



### Holistic Construction Automation with Modular Robots: From High-Level Task Specification to Execution (https://arxiv.org/abs/2412.20867)
- **What's New**: 이 논문은 건설 분야의 분산 로봇 자동화를 위한 포괄적인 프레임워크를 제안합니다. 사용자는 그래픽 인터페이스를 통해 로봇의 행동을 지정하고 모니터링할 수 있습니다. 이 접근 방식은 건물 정보 모델링(Building Information Modelling, BIM)을 통합하여 실세계에서의 자동 실행을 가능하게 합니다. 또한, 모듈식 로봇 구성 요소를 활용하여 건설 작업의 특정 요구 사항에 신속하게 적응할 수 있도록 설계되었습니다.

- **Technical Details**: 프레임워크는 고급 지침을 자율적으로 분해하고(BIM 데이터를 활용하여 로봇 팔의 최적 모듈 구성을 식별), 온라인 경로 계획을 수행하며, 모바일 베이스의 위치 오류에 대한 적응을 포함합니다. 작업자가 필요한 것은 작업의 명세와 로봇 모듈의 조립뿐입니다. 이 논문은 모바일 로봇 조작에 따른 정확성과 유연성을 강조하며, 시뮬레이션을 통한 최적화 후 실제 로봇 조작을 검증합니다.

- **Performance Highlights**: 실험 검증 결과, 제안된 접근 방식은 자율적인 드릴링 작업을 성공적으로 수행할 수 있음을 보여줍니다. 이 연구는 과거 건설 산업에서 로봇 사용의 부족을 해결할 수 있는 가능성을 제시합니다. 또한, 고급 제어 시스템이 통합되어 있으며, 이는 로봇의 실제 동작에서 발생할 수 있는 오류를 최소화하는 데 도움을 줍니다.



### Enhancing Annotated Bibliography Generation with LLM Ensembles (https://arxiv.org/abs/2412.20864)
- **What's New**: 이 연구는 전통적인 주석 목록 생성 방식에 비해 더욱 향상된 방법을 제안합니다. 특히, 다양한 역할을 수행하는 여러 LLM(대형 언어 모델)을 활용하여 모델 성능을 개선하는 체계적인 방법론을 채택하였습니다. 본 접근 방식은 텍스트 생성, 평가, 요약 과정에서의 LLM들의 협력을 통해 이루어집니다.

- **Technical Details**: 제안된 방법은 서로 다른 LLM 매개변수를 통해 텍스트의 다양성을 확보하고, 평가를 위한 LLM이 정확성, 관련성 및 일관성을 평가하는 구조로 되어 있습니다. 이 후 여러 결합 전략을 통해 선택된 응답은 요약 및 중복 제거 기법으로 통합 및 정제됩니다. 이러한 체계적인 접근 방식은 학문적인 작업에서의 LLM의 역할을 더욱 다양하게 확장할 수 있게 합니다.

- **Performance Highlights**: 예비 실험 결과, LLM 앙상블의 결합된 출력을 사용했을 때 개별 응답에 비해 일관성과 관련성이 개선됨을 보여주었습니다. 이 연구는 주석 품질이 38% 향상되었고 내용 중복이 51% 감소하였음을 밝히고 있습니다. 이를 통해 복잡한 학문적 작업의 자동화 가능성을 제시하면서도 높은 품질 기준을 유지할 수 있다는 점이 강조되었습니다.



### About rectified sigmoid function for enhancing the accuracy of Physics-Informed Neural Networks (https://arxiv.org/abs/2412.20851)
Comments:
          9 pages, 1 figure, 2 tables, 4 algthorithms. arXiv admin note: substantial text overlap with arXiv:2412.19235

- **What's New**: 이번 연구는 일반적으로 물리 문제를 해결하기 위해 수정된 활성화 함수와 단일 숨겨진 층을 가진 신경망의 사용에 초점을 맞추고 있습니다. 새롭게 제안된 rectified sigmoid 함수는 ODE(Ordinary Differential Equations)로 설명되는 물리 문제를 보다 효과적으로 해결할 수 있도록 설계되었습니다. 또한, 데이터 기반 초기화 및 무기울기 신경망 적합 방법이 제시되어 이 신경망의 학습 및 적용 가능성을 높이고 있습니다.

- **Technical Details**: 연구에서는 rectified sigmoid 활성화 함수를 활용하여 ODE 문제를 해결하기 위한 알고리즘을 상세히 설명합니다. PINN(Physics-Informed Neural Network) 접근 방법을 통해 알려지지 않은 해를 신경망으로 근사하며, 여러 신경망 구성요소의 가중치와 편향을 최적화합니다. 연구에서 제안한 방법은 기존의 sigmoid 활성화 함수에 비해 높은 정확성을 제공하며, 다양한 물리 문제 해결에 적용되었습니다.

- **Performance Highlights**: Numerical experiments에서 rectified sigmoid 활성화 함수를 사용하는 신경망이 기존의 sigmoid 활성화 함수보다 물리 문제 해결 정확도가 우수하다는 결과를 확인했습니다. 이 연구는 다양한 물리적 시스템을 다루며, 특히 harmonic oscillator, relativistic slingshot, 및 Lorentz system을 통한 실험이 포함되었습니다. 이러한 결과는 수정된 활성화 함수가 물리 문제 해결에 있어 매우 효과적임을 입증하고 있습니다.



### Analog Alchemy: Neural Computation with In-Memory Inference, Learning and Routing (https://arxiv.org/abs/2412.20848)
- **What's New**: 이 논문은 신경 연산(neural computation)의 발전에 따라 이상적인 신경 하드웨어(neural hardware)를 재고하는 필요성을 강조합니다. 특히 기존의 von Neumann 아키텍처(von Neumann architecture)의 메모리와 연산의 분리로 인한 에너지 효율의 병목현상을 문제 삼고 있습니다. 저자는 메모리스티브 장치(memristive devices)를 활용한 대안적인 방법을 탐구하며, 이를 통해 지능형 시스템의 구축 가능성을 모색합니다.

- **Technical Details**: 저자는 메모리스티브 장치의 독특한 물리적 동역학(physical dynamics)을 활용하여 추론(inference), 학습(learning), 라우팅(routing)을 수행하는 방법을 제시합니다. 경량의 매개변수 파악을 위해 그래디언트 기반 학습(gradient-based learning) 원칙에 의해 필요한 함수를 선택하고, 효율적인 배선을 위한 커넥토믹스(connectomics) 원리를 분석합니다. 비이상성(non-idealities)과 아날로그 물리에서의 잡음(noise)에도 불구하고, 저자는 이러한 환경에서의 지역 학습(local learning)의 적응성에 대한 하드웨어적 증거를 제공합니다.

- **Performance Highlights**: 본 연구는 메모리스티브 기반의 새로운 재료 조합과 회로 블록을 통해 크레딧 할당 문제(credit assignment problem) 해결과 아날로그 교차 바(analog crossbars) 간의 효율적인 라우팅을 지원하는 방법을 다룹니다. 이 접근 방식은 스케일러블 아키텍처(scalable architectures)에 대한 적용 가능성을 보여줍니다. 결국, 제안된 방법은 신경 연산의 에너지 효율성과 성능 개선에 기여할 수 있는 잠재력을 지니고 있습니다.



### Dual-Space Augmented Intrinsic-LoRA for Wind Turbine Segmentation (https://arxiv.org/abs/2412.20838)
Comments:
          Authors Shubh Singhal and Raül Pérez-Gonzalo contributed equally to this work. Accepted to ICASSP 2025

- **What's New**: 이번 연구에서는 풍력 터빈 블레이드의 이미지 분할(segmentation) 정확도를 높이기 위해 Intrinsic LoRA 방법을 확장하고, 새로운 이중 공간 증강(dual-space augmentation) 전략을 제안합니다. 이 방법은 이미지 레벨과 잠재 공간(latent-space) 증강을 통합하여, 선형 보간(linear interpolation)과 노이즈 기반의 확률적 모델을 통해 강화된 이미지 분할 결과를 공헌합니다. 현재의 최첨단 기법들을 초월하는 성능을 보여주어 자가 점검 및 손상 감지 시스템의 효과를 향상시킵니다.

- **Technical Details**: 이 논문에서는 Intrinsic LoRA 방법을 사용하여 이미지 분할을 위한 세부적인 과정을 설명합니다. 초기 Stable Diffusion 모델을 적응하여 WTB 분할을 위한 인코더와 디코더 구조를 적용하였으며, 잠재 공간에서의 변형 벡터의 최적화를 통해 지도 학습을 활성화했습니다. 또한 이미지 및 잠재 공간에서의 이중 증강 접근 방식을 통해 이미지 분할 성능을 극대화하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 WTB 이미지 분할 기법에 비해 현저하게 높은 정확도를 기록했습니다. 특히, 이중 증강 기법을 통해 모델의 학습 안정성과 성능을 동시에 개선하였습니다. 이러한 성과는 풍력 에너지 분야에서의 자동화된 평가 시스템 구축에 기여할 것으로 기대됩니다.



### Disentangling Preference Representation and Text Generation for Efficient Individual Preference Alignmen (https://arxiv.org/abs/2412.20834)
Comments:
          Coling 2025

- **What's New**: 이 연구에서는 개인의 피드백에 따라 LLM(대형 언어 모델)을 개인화하는 새로운 접근 방식을 제시합니다. 특히, 개인의 선호도를 효율적으로 정렬할 수 있는 유연한 패러다임을 도입하며, 이는 텍스트 생성과 선호 표현을 분리하여 수행됩니다. 이러한 접근 방식은 기존의 PEFT(파라미터 효율 미세 조정) 방식보다 80%에서 90%까지 훈련 시간을 단축시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다: 첫 번째 단계에서는 반응 표현을 위한 잠재 인코더와 이를 LLM에 공급하는 잠재 어댑터를 훈련합니다. 두 번째 단계에서는 개인의 피드백에 따라 개인화된 잠재 표현을 생성하기 위해 잠재 인코더를 미세 조정합니다. 추가적으로, 개인의 피드백을 통해 수집된 정보를 활용하여 맞춤형 텍스트 생성을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LoRA 기반의 방법과 P-Tuning 기반의 방법에 비해 경쟁력 있는 정렬 품질을 달성합니다. 예를 들어, LoRA 기반 DPO의 실험 결과는 52.4에서 80.8로 개선된 반면, 제안된 Latent DPO는 52.5에서 83.3로 향상되었습니다. 이는 지속적으로 높은 품질을 유지하면서도 계산 비용과 시간을 크게 절감할 수 있음을 보여줍니다.



### Fine-Tuning TransMorph with Gradient Correlation for Anatomical Alignmen (https://arxiv.org/abs/2412.20822)
- **What's New**: 이 논문에서는 뇌 MRI 등록에서의 unsupervised deep learning 접근 방식의 진전을 다루고 있습니다. 제안된 방법은 사전 훈련된 TransMorph 모델을 미세 조정하여 해석학적 정확성을 향상시키고, 해부학적 정렬을 유지하는 간섭을 줄이는 것에 중점을 두고 있습니다. FAdam optimizer와 gradient correlation을 포함하여 매끄럽고 구조적으로 일관된 변형을 달성하는 데 중점을 두었습니다.

- **Technical Details**: 이 방법론은 특히 Fisher Adam (FAdam) optimizer와 gradient correlation (GC) 유사도 측정을 포함합니다. FAdam은 훈련 중 최적화를 개선하기 위해 조정된 모멘텀과 편향 보정, 적응형 그래디언트 스케일링을 사용하는 Adam 변형입니다. 또한, GC는 뇌 MRI 간의 구조적 일관성을 유지하는 데 도움이 되며, 고차원 이미지를 비교하는 데 유용한 유사도 측정법입니다.

- **Performance Highlights**: 실험 결과, FAdam+GC 모델이 Dice 계수와 95% Hausdorff 거리(HdDist95)에서 약한 개선을 보였으며, 비변형 볼륨(NDV)은 0.27%로 현저하게 낮아져 매끄러운 변형을 나타냅니다. 평가된 모델 간에는 유사한 등록 성능이 있지만 FAdam+GC가 하는 gradient correlation 덕분에 경계 부위에서 보다 나은 정렬을 보여 주며, 이는 HdDist95 지표에 긍정적인 영향을 미쳤습니다.



### Length-Aware DETR for Robust Moment Retrieva (https://arxiv.org/abs/2412.20816)
- **What's New**: 본 논문은 비디오 순간 검색(Video Moment Retrieval, MR) 분야에서 단기 순간의 검색 정확성을 향상시키기 위한 두 가지 새로운 기술, MomentMix와 Length-Aware Decoder (LAD)를 제안합니다. 최근의 DETR 기반 모델들이 단기 순간에 대해 성능 저하 문제를 보이는 가운데, 이 연구는 이러한 문제를 해결하기 위한 방법론을 제공하여 주목을 받습니다. 데이터 분석을 통해 단기 순간의 특징 다양성이 제한적이라는 점을 발견하였고, 이를 해결하기 위한 출시된 혁신적 기술들이 설명됩니다.

- **Technical Details**: MomentMix는 ForegroundMix와 BackgroundMix라는 두 가지 데이터 증강 전략을 활용하여 비디오의 전경(가장 관련이 큰 정보)과 배경(관련이 적은 정보)의 특징 표현을 향상시키는 방법입니다. ForegroundMix는 다양한 순간의 전경 특징을 결합하여 새로운 전경을 만들며, BackgroundMix는 다른 비디오의 배경 조각을 사용하여 다양한 전경-배경 조합을 생성합니다. 또한 Length-Aware Decoder는 순간의 길이에 따른 예측 정확성을 개선하도록 모델을 구조화하여, 더 나은 센터(center) 및 길이(length) 예측을 가능하게 합니다.

- **Performance Highlights**: 제안된 접근법은 기존의 DETR 기반 방법들과 비교하여 QVHighlights, TACoS, Charades-STA와 같은 여러 데이터셋에서 성능이 크게 향상됨을 보여줍니다. 특히 QVHighlights에서는 mAP이 9.36% 증가하여 41.22에서 45.08로 상승하였고, R1@0.7에서도 유의미한 개선을 달성하였습니다. 이는 단기 순간의 검색 성능을 비약적으로 향상시킴으로써 MR 분야에서 중요한 한 걸음을 내딛었다고 할 수 있습니다.



### Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability (https://arxiv.org/abs/2412.20807)
Comments:
          9 pages, 6 figures, accepted by 2025ICASSP

- **What's New**: 이 논문에서는 목표 공격(targeted attack)의 전이 가능성(transferability)을 향상시키기 위한 새로운 방법론을 제안합니다. 기존의 방법론들이 최적화 경로(optimization trajectory)를 무시하고 끝점만을 이용하는 반면, 이 논문은 최적화 경로의 평균화(averaging)를 통해 조작된 적대적 예시(adversarial example)를 안정적으로 중앙 영역으로 이동시키는 방안을 제시합니다.

- **Technical Details**: 제안된 방법은 기존의 적대적 공격과 결합되어 다양한 공격 시나리오에서 비교됩니다. 기존의 fine-tuning 기법들이 손실 표면(loss surface)의 평탄한 영역 주변에서 진동하는 경향을 보이는 반면, 이 방법은 경로의 평균화를 통해 이를 극복하고자 합니다. 이는 목표 공격의 향상된 전이 가능성을 제공하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 기존의 fine-tuning 기법들과 비교할 때 목표 전이 가능성(targeted transferability)을 크게 향상시킵니다. 이 방법은 최신 적대적 공격 기법들과 통합되어 뛰어난 성능을 나타내며, 실제 코드도 제공됩니다.



### A Tale of Two Imperatives: Privacy and Explainability (https://arxiv.org/abs/2412.20798)
Comments:
          Work in progress

- **What's New**: 이번 논문은 인공지능(AI) 분야에서의 프라이버시(Right-to-Privacy, RTP)와 설명 가능성(Right-to-Explanation, RTE)의 복합적 문제를 다룹니다. Differential Privacy (DP)가 개인정보 보호 머신러닝의 금본위로 자리 잡은 이유와, 포스트-호크(post-hoc) 설명 기법들이 모델 감사에 어떻게 기여할 수 있는지를 탐구합니다. 특히, 이 논문은 고위험(high-stakes) 어플리케이션에서 RTP와 RTE를 효과적으로 통합하는 방법을 제안합니다.

- **Technical Details**: 이 연구는 DP 모델과 다양한 포스트-호크 설명 기법의 상호작용을 분석하여, 이들 각각의 특성과 제한 사항을 명확하게 합니다. 연구에서는 3종의 CNN 모델, 6개의 ε (epsilon) 값, 그리고 5개의 인기 있는 포스트-호크 설명 방법을 사용하여 실험을 진행하며 이들이 RTP 조건을 만족하는지를 검토합니다. 이를 통해 RTP와 RTE의 결합이 고위험 상황에서 어떻게 효과적으로 이루어질 수 있는지를 제시합니다.

- **Performance Highlights**: 연구의 주요 성과는 DP 모델과 포스트-호크 설명 기법들의 조화로운 작동 가능성을 탐구하고, 이를 통해 고위험 AI 시스템에서도 신뢰성과 효율성을 확보할 수 있는 기초를 제공한다는 점입니다. 또한, 이 논문은 의료 기록 같은 프라이버시가 중요한 분야에서 환자와 의사 모두가 신뢰할 수 있는 예측을 제공하는 것이 필수적임을 강조합니다. 결과적으로, RTP와 RTE를 만족하는 산업 소프트웨어 파이프라인 예시를 도출하여 실질적인 적용 가능성을 제시합니다.



### Frequency-Masked Embedding Inference: A Non-Contrastive Approach for Time Series Representation Learning (https://arxiv.org/abs/2412.20790)
Comments:
          This paper has been accepted by AAAI-2025 main track

- **What's New**: 이 논문은 Frequency-masked Embedding Inference (FEI)라는 새로운 비대조적(self-supervised) 시간 시계열 표현 학습 프레임워크를 제안합니다. FEI는 정량적 데이터 샘플 쌍의 필요성을 완전히 없애고, 주파수 마스킹 프롬프트를 활용하여 임베딩 공간에서의 연속적인 의미 모델링을 가능하게 합니다. 기존 대조 학습의 한계를 벗어나기 위한 접근 방식으로, 시간 시계열의 연속적인 의미 관계를 탐색하는 것이 주요 특징입니다.

- **Technical Details**: FEI의 구조는 두 가지 추론 분기로 구성되어 있습니다. 첫 번째는 마스킹 프롬프트를 기반으로 한 타겟 임베딩 추론이고, 두 번째는 타겟 시리즈를 프롬프트로 사용하는 마스크 추론입니다. 이를 통해 FEI는 특정 주파수 샘플을 임베딩 공간에서 직접 추론하므로, 비대조적 모델링 방식을 활용한 연속적인 임베딩 공간을 구축할 수 있습니다.

- **Performance Highlights**: 제안된 FEI는 8개의 광범위하게 사용되는 시간 시계열 데이터셋에서 분류 및 회귀 작업을 위한 실험 결과, 기존의 대조 기반 방법들에 비해 현저히 우수한 일반화 성능을 보였습니다. 이 논문은 비대조적 학습의 가능성과 효과를 입증하며, 시간 시계열 자기 지도 표현 학습에 대한 새로운 통찰을 제공합니다.



### SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity (https://arxiv.org/abs/2412.20787)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)의 성능을 평가하기 위한 새로운 벤치마크 데이터셋인 SecBench를 제안합니다. SecBench는 사이버 보안 도메인에서 LLM을 평가하기 위해 설계된 다차원 벤치마킹 데이터셋으로, 다양한 형식의 질문(객관식 질문(MCQ) 및 주관식 질문(SAQ))과 여러 언어(중국어 및 영어), 다양한 하위 도메인에서의 질문들을 포함합니다. 기존 벤치마크는 일반적인 성능 평가에 초점을 맞추었으나, SecBench는 특정 전문가 도메인, 특히 사이버 보안에 대한 평가에 중점을 두고 있습니다.

- **Technical Details**: SecBench는 두 가지 판단 수준(지식 보유(KR) 및 논리적 추론(LR))으로 LLM의 능력을 평가합니다. 이 데이터셋은 44,823개의 MCQ와 3,087개의 SAQ로 구성되어 있으며, 이는 고품질 데이터와 다양한 대회 및 공모전을 통해 수집된 데이터입니다. 평가 과정에서는 강력한 LLM인 GPT-4를 사용하여 데이터 라벨링과 SAQ의 자동 채점을 수행했으며, 총 13개의 최신 LLM이 평가에 활용되었습니다.

- **Performance Highlights**: SecBench는 사이버 보안 분야에서 LLM의 가장 포괄적이고 광범위한 평가를 가능하게 합니다. 13개의 State-of-the-Art LLM에 대한 벤치마킹 결과는 SecBench의 유용성을 정성적 및 정량적으로 입증하였습니다. 이 데이터셋은 LLM의 강점을 파악하고 성능을 개선할 수 있는 중요한 도구로 자리잡을 것으로 기대됩니다.



### Sample Correlation for Fingerprinting Deep Face Recognition (https://arxiv.org/abs/2412.20768)
- **What's New**: 최근 얼굴 인식 분야에서는 딥 러닝 기술의 발전으로 인해 놀라운 혁신이 이루어졌습니다. 그러나 모델 훔치기 공격에 대한 우려가 커지고 있으며, 이는 모델 소유자의 지적 재산권에 심각한 위협이 되고 있습니다. 이러한 문제를 해결하기 위해, SAC(Sample Correlation)이라는 새로운 모델 훔치기 탐지 방법을 제안하였습니다.

- **Technical Details**: SAC 방법은 샘플 간의 쌍별 관계를 이용하여 모델 훔치기 공격을 탐지합니다. 특히 SAC-JC는 JPEG 압축된 샘플을 사용하여 상관 행렬을 계산하고, 이를 기반으로 훔쳐진 모델을 검증합니다. 다양한 공격 방식에서 SAC-JC의 유효성을 검증하였으며, 그림 인식 데이터셋 등 여러 실험을 통해 기존 방법보다 우수한 성능을 발휘함을 보여주었습니다.

- **Performance Highlights**: 실험 결과 SAC-JC는 얼굴 인증 및 감정 인식 과제에서 가장 높은 AUC, p-value 및 F1 점수를 기록하며 다양한 모델 훔치기 공격에 대해 성공적으로 방어함을 입증하였습니다. 또한, 객체 인식 데이터셋인 Tiny-ImageNet과 CIFAR10에서도 SAC-JC의 우수성을 확인하였습니다. 이 방법은 대체 모델 훈련 없이 계산적 부담을 크게 줄이며, 약 34393배 더 빠른 성능을 제공합니다.



### KeyGS: A Keyframe-Centric Gaussian Splatting Method for Monocular Image Sequences (https://arxiv.org/abs/2412.20767)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 3D 모델을 재구성하는 새로운 프레임워크인 KeyGS를 제안합니다. 이 방법은 Structure-from-Motion(SfM) 기법을 사용하여 초기에 대략적인 카메라 포즈를 신속하게 획득한 후, 3D Gaussian Splatting(3DGS)을 활용하여 포즈를 정제합니다. KeyGS는 효율적인 카메라 포즈 추정과 함께 긴 훈련 시간을 단축시키는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: KeyGS는 기존 방법들보다 카메라 포즈의 정확도를 향상시키기 위해 조합된 정제 과정을 도입합니다. 일반적으로 SfM과 여러 외부 소프트웨어를 통한 카메라 포즈 추정이 필요하지만, KeyGS는 불필요한 매칭 모델 없이 몇 초 만에 카메라 포즈를 얻습니다. 또한, 이 방법은 고주파 신호에 의한 오류를 방지하도록 설계된 coarse-to-fine frequency-aware densification 기법을 통합하여 재구성 과정의 질을 높입니다.

- **Performance Highlights**: 이 접근법은 훈련 시간을 몇 시간에서 몇 분으로 단축시키면서, 더 높은 정확도의 새로운 뷰 합성을 달성하고 기존 방법들과 비교해 카메라 포즈 추정의 신뢰성을 크게 개선합니다. KeyGS의 구현을 통해 다양한 환경에서도 정확한 3D 재구성이 가능합니다. 이 연구의 결과는 3D 재구성을 위한 실시간 렌더링과 효율성을 극대화하는 데 기여할 것으로 보입니다.



### Attributing Culture-Conditioned Generations to Pretraining Corpora (https://arxiv.org/abs/2412.20760)
- **What's New**: 이번 연구는 언어 모델이 문화에 따라 편향된 출력을 생성하는 방식에 대한 새로운 통찰력을 제공합니다. 본 연구에서는 MEMOed 프레임워크(MEMOrization from pretraining document)를 통해 모델이 특정 문화와 관련된 심볼(symbol)을 어떻게 기억하는지를 분석하고, 이러한 기억이 문화적 편향의 원인임을 밝혀냅니다.

- **Technical Details**: MEMOed 프레임워크는 두 가지 단계로 구성되어 있습니다: 1) 문화-심볼 연관성에 기여한 문서를 식별하고, 2) 기여 문서의 비율이 충분히 높은 경우 해당 심볼을 기억된 것으로 분류합니다. 이를 통해 110개 문화에 대한 음식 및 의류 주제로 생성된 심볼에서 기억된 연관성이 46% 및 26%를 차지함을 발견했습니다.

- **Performance Highlights**: 이 연구는 모델이 기억된 고빈도 심볼에 의존하므로, 문화에 대한 정확한 지식을 고르게 회상하지 못하고 있다는 점을 강조합니다. 이러한 결과는 문화 생성에서 편향을 완화하기 위한 더 나은 접근 방식을 모색하는 데 유용할 것입니다. 향후 MEMOed 프레임워크와 관련된 연구가 진행됨으로써, 모델의 성능이 사전 훈련 데이터에 어떻게 의존하는지를 더 깊이 이해할 수 있게 되기를 바랍니다.



### Solar Filaments Detection using Active Contours Without Edges (https://arxiv.org/abs/2412.20749)
Comments:
          6 pages, 2 figures

- **What's New**: 이 논문에서는 H-alpha 전체 디스크 태양 이미지에서 태양 필라멘트를 감지하기 위해 액티브 컨투어(Active Contours without Edges, ACWE) 기반 알고리즘이 제안되었습니다. 이 알고리즘은 이미지 전처리, 분할, 후처리의 세 가지 주요 단계로 구성됩니다. 새로운 접근 방식은 객체의 경계를 정확하게 감지하여 태양 이미지에서 최적의 성능을 발휘합니다.

- **Technical Details**: 제안된 알고리즘은 에너지 함수에 따라 변형할 수 있는 경계를 태양 이미지 위에 초기화하고, 원하는 객체의 경계에 도달하면 에너지 함수가 감소하며 경계가 더 이상 발전하지 않도록 설계되었습니다. 이 과정은 이미지 전처리, 분할, 후기 처리를 포함하여 각 단계를 체계적으로 수행합니다.

- **Performance Highlights**: 성능 분석 결과, 제안된 알고리즘은 기존의 클래식(Object Detection) 알고리즘보다 뛰어난 성능을 보였습니다. 이 연구에서 제안한 방법은 몇 가지 벤치마크 데이터셋에 적용되어 비교되었으며, 객체 감지에서의 우수성을 입증하였습니다.



### Advancing Parkinson's Disease Progression Prediction: Comparing Long Short-Term Memory Networks and Kolmogorov-Arnold Networks (https://arxiv.org/abs/2412.20744)
- **What's New**: 본 연구는 콜모고로프-아르놀드 네트워크(KAN)을 활용하여 파킨슨병(PD) 진행 상황을 예측하는 혁신적인 접근 방식을 제안합니다. KAN은 전통적인 선형 모델과는 달리 스플라인 매개변수화된 일변수 함수를 활용하여 동적 학습이 가능하게 합니다. 이 연구는 KAN의 성능을 평가하고 전통적인 회귀 모델 및 딥러닝 아키텍처와 비교하여 최고의 예측 방법을 규명하고자 합니다.

- **Technical Details**: 연구의 주요 방법론은 파킨슨병 진행 예측을 위한 MDS-UPDRS 점수를 기반으로 하며, 데이터 세트 설명에서 시작하여 데이터 전처리, 피처 추출, 모델 작동 방식 및 모델 매개변수에 대한 자세한 설명을 포함합니다. 특히 KAN은 작고 간단한 네트워크 크기를 통해도 동일하거나 더 우수한 성과를 거둘 수 있도록 설계되었습니다. 이를 통해 모델의 정확도와 해석 가능성을 개선하게 됩니다.

- **Performance Highlights**: 연구 결과, KAN은 병의 진행 상태를 예측하는 데 있어 다른 접근 방식보다 뛰어난 성과를 나타냈습니다. 다양한 방법론을 비교하고 KAN의 뛰어난 성능을 강조함으로써, 임상 데이터에서 복잡한 시간적 패턴을 효과적으로 모델링하고 해석할 수 있는 가능성을 보여줍니다. 이러한 발전은 PD 관리에서의 AI와 머신러닝의 잠재력을 강조하고, 향후 치료 전략과 환자 관리 개선을 위한 기초를 마련합니다.



### Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study (https://arxiv.org/abs/2412.20733)
Comments:
          The original work citation: Bačić, B., Claudiu Vasile, Feng, C., & Ciucă, M. G. (2024, 13-15 Dec.). Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study. Presented at the Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024), Sydney, NSW

- **What's New**: 이번 논문은 개인 정보를 보호하는 빅데이터 분석 헬스케어 플랫폼을 구축하기 위한 기여를 목표로 하고 있습니다. 이를 위해 환자들의 동영상이나 시계열 데이터 처리를 가능하게 하는 새로운 접근 방식을 제안하고 있습니다. Google MediaPipe를 활용하여 모바일 동영상을 개인 정보 보호 진단 시계열 데이터로 변환하는 과정을 소개합니다.

- **Technical Details**: 연구에서는 실제 무릎 재활 동영상 데이터셋을 사용하였으며, 다양한 운동을 정밀하게 분석하기 위한 알고리즘을 개발했습니다. 이 알고리즘은 환자의 운동 비디오에 스틱 피겨 요소를 오버레이하고, CSV 파일 형식으로 무릎 각도 추정을 업데이트하여 시계열 플롯을 생성합니다. 또한, 사전 정의된 무릎 각도 매개변수를 통해 문제를 시각적으로 나타내는 동영상이 가능해집니다.

- **Performance Highlights**: 제안된 적응형 알고리즘은 재활 프로그램 준수를 높이고 운동 세트 및 반복 수를 정확히 측정할 수 있습니다. 동영상의 측면 및 전면 보기에서 모든 운동을 91.67%에서 100%까지 정확히 식별할 수 있으며, 다양한 무릎 운동 패턴에 대한 투명한 알고리즘 설계는 해석 가능한 AI의 발전에 기여합니다. 이러한 연구는 향후 개인 정보 보호를 고려한 오픈 소스 개발로 이어질 것으로 기대됩니다.



### M$^3$oralBench: A MultiModal Moral Benchmark for LVLMs (https://arxiv.org/abs/2412.20718)
- **What's New**: 최근 대규모 기초 모델, 특히 대규모 언어 모델(LLMs)과 대규모 비전-언어 모델(LVLMs)이 법률, 금융, 의료 등 다양한 분야에서 필수적인 도구로 자리잡고 있습니다. 이러한 모델의 일상적인 통합이 증가함에 따라, 인간의 가치와 도덕적 경계를 존중하는 출력을 보장하기 위한 도덕적 평가(moral evaluation)의 필요성이 대두되고 있습니다. 이전 연구들은 주로 LLMs에 집중해 도덕적 데이터셋과 텍스트 모드에 제한된 벤치마크를 제안했습니다.

- **Technical Details**: 이에 따라, 우리는 LVLMs를 위한 최초의 다중 모달 도덕 벤치마크(MultiModal Moral Benchmark)인 M$^3$oralBench를 소개합니다. M$^3$oralBench는 도덕 기초 시나리오(Moral Foundations Vignettes, MFVs)를 확장하고 텍스트-이미지 확산 모델인 SD3.0을 사용하여 관련 시나리오 이미지를 생성합니다. 이 벤치마크는 도덕 기초 이론(Moral Foundations Theory, MFT)의 여섯 가지 도덕적 기초에 따른 도덕적 평가를 실시하며, 도덕 판단(moral judgement), 도덕 분류(moral classification), 도덕적 반응(moral response) 과제를 포함하여 다중 모달 도덕 이해 및 추론 모델 성능을 종합적으로 평가합니다.

- **Performance Highlights**: 10개의 인기 있는 오픈 소스 및 클로즈드 소스 LVLM에 대해 광범위한 실험을 진행한 결과, M$^3$oralBench는 현재 모델의 현저한 도덕적 한계를 드러내는 도전적인 벤치마크임을 입증했습니다. 우리의 벤치마크는 공개 가능하며, 이를 통해 LVLMs의 도덕적 성능을 개선하기 위한 기초 데이터로 활용될 수 있습니다.



### UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design (https://arxiv.org/abs/2412.20694)
- **What's New**: 이번 논문에서는 NP-hard 문제 해결을 위한 새로운 방법론인 UBER(불확실성 기반 진화법)를 소개합니다. UBER는 기존의 FunSearch 프레임워크 위에 불확실성을 통합하여 LLM(대형 언어 모델)과 EA(진화 알고리즘)의 결합을 강화합니다. 이는 자동 휴리스틱 설계의 효율성을 크게 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: UBER는 두 가지 주요 혁신을 도입합니다. 첫째, UIEP(Uncertainty-Inclusive Evolution Process)는 탐색-활용(balance of exploration and exploitation)을 적응적으로 조정하여 더 효과적인 문제 해결을 가능하게 합니다. 둘째, UIIS(Uncertainty-Inclusive Island Reset) 전략은 인구 다양성을 유지하여 알고리즘의 성능을 높입니다.

- **Performance Highlights**: 광범위한 NP-complete 문제에 대한 실험을 통해 UBER는 FunSearch에 비해 상당한 성능 개선을 보여줍니다. 이 연구는 LLM과 EA의 상호 작용에 대한 새로운 방향을 제시하며, 자동 휴리스틱 설계 분야에 기여할 것입니다.



### Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner (https://arxiv.org/abs/2412.20662)
- **What's New**: 본 논문은 사전 학습된 비전 대형 언어 모델(VLLMs)의 테이블 인식 적용에 대한 연구 격차를 해소하고자 합니다. 특히, 기존의 VLLM 기반 방법들과 달리, Fine-tuning 없이 프롬프트 기반의 패러다임을 이용해 테이블 인식을 다루는 새로운 접근 방식을 제안합니다. 이를 위해 다양한 계층 구조를 갖춘 벤치마크를 설계하고 실험을 통해 비전 모델의 성능을 향상시키기 위한 프레임워크인 Neighbor-Guided Toolchain Reasoner (NGTR)를 제안합니다.

- **Technical Details**: NGTR 프레임워크는 저품질 입력 이미지의 문제를 해결하기 위해 여러 경량 모델을 통합하여 시각 처리 작업을 수행합니다. 이 프레임워크는 이웃 검색 메커니즘을 활용하여 각 입력 인스턴스에 대해 유사한 이웃을 검색하고, 이를 통해 생성된 도구 호출 계획을 안내합니다. 또한, 각 단계의 도구 호출 과정에서 반영(reflection) 모듈을 포함시켜 도구 선택을 지속적으로 개선함으로써 VLLMs가 더 높은 품질의 구조화된 데이터를 생성할 수 있도록 합니다.

- **Performance Highlights**: 본 연구의 실험 결과, NGTR 프레임워크는 기존의 VLLM 기반 접근 방식의 테이블 인식 성능을 상당히 향상시킨 것으로 나타났습니다. 특히, 기존의 전통적인 모델들에 비해 VLLMs는 경쟁력 있는 정확도를 보여주었지만, 여전히 성능 차이를 보여 주었습니다. 이번 연구는 VLLMs의 다양한 공공 테이블 인식 데이터셋에서 가능한 성능 경계를 예비적으로 밝히고, 향후 연구의 기초가 될 수 있는 중요한 관점을 제시합니다.



### Overcoming Class Imbalance: Unified GNN Learning with Structural and Semantic Connectivity Representations (https://arxiv.org/abs/2412.20656)
- **What's New**: 본 논문에서는 class imbalance(클래스 불균형) 문제를 해결하기 위한 새로운 Unified Graph Neural Network Learning (Uni-GNN) 프레임워크를 제안합니다. 기존 GNN의 한계를 극복하기 위해, 구조적(connectivity) 및 의미적(semantic) 연결 표현을 통합하여 노드 분류의 정확도를 향상시키고자 합니다. 또한, unlabelled(라벨 없는) 노드의 잠재력을 활용하기 위한 balanced pseudo-label 생성 메커니즘을 도입하여 학습 데이터셋 내 minority classes(소수 클래스)의 라벨 수를 증가시킵니다.

- **Technical Details**: Uni-GNN 프레임워크는 입력 그래프의 구조적(connectivity) 연결 기반 구조를 구축하고, 노드 임베딩 간의 유사성을 바탕으로 의미적(semantic) 연결을 도출합니다. 각 연결의 유형에 대해 메시지 패싱(layer)을 통해 정보 전파를 수행하고, 이로 인해 포괄적인 구조적 및 의미적 표현을 얻을 수 있습니다. 이를 통해 노드 임베딩의 전파 범위를 표준 이웃을 넘어 비인접한 구조적 노드 및 의미적으로 유사한 노드까지 확장합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 실시한 실험 결과, 제안된 Uni-GNN 프레임워크가 기존의 클래스 불균형 그래프 학습의 첨단 방법들에 비해 우수한 성능을 학습할 수 있음을 강조하고 있습니다. Uni-GNN은 class imbalance 문제를 효과적으로 해결하여 다양한 응용 분야에서 그래프 신경망의 활용성을 높이는 데 기여할 것으로 기대됩니다.



### Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis (https://arxiv.org/abs/2412.20651)
- **What's New**: 이 논문에서는 Latent Drift (LD)라는 새로운 접근 방식을 제안하여, 의료 이미징에서의 분포 이동(distribution shift) 문제를 완화할 수 있도록 diffusion 모델을 개선합니다. 이는 대규모 데이터셋의 접근이 제한된 의료 이미징에서 중요한 역할을 합니다. Latent Drifting은 반사상(counterfactual) 이미지 생성을 가능하게 하여, 성별, 나이, 질병의 유무 등의 변수를 반영하는 이미지 생성을 지원합니다.

- **Technical Details**: Latent Drift는 기존의 pre-trained 모델에 대한 미세 조정(fine-tuning)을 용이하게 하여, 데이터의 분포 변화에 따른 도전을 극복할 수 있게 해줍니다. 이 방법은 어떤 미세 조정 방식에도 적용 가능할 뿐만 아니라, 추론(inference) 시간에도 조건으로 사용될 수 있습니다. 본 연구에서는 Latent Drift를 활용하여 세 가지 공개 장기적 벤치마크(brain MRI, chest X-ray) 데이터셋에서 반사상 이미지를 생성하는 성능을 평가했습니다.

- **Performance Highlights**: 진행한 실험 결과, Latent Drift는 다양한 시나리오에서 성능 향상을 보였으며, 여러 미세 조정 방식과 결합했을 때 특히 유의미한 향상을 나타냈습니다. 이러한 결과는 의료 이미징 분야에서 diffusion 모델의 잠재력을 증명하는 중요한 증거가 됩니다. 이 연구 결과물의 소스 코드는 논문이 수락된 후 공개될 예정입니다.



### NetFlowGen: Leveraging Generative Pre-training for Network Traffic Dynamics (https://arxiv.org/abs/2412.20635)
- **What's New**: 이번 연구에서는 NetFlowGen이라는 프레임워크를 제안하여, 라벨이 없는 NetFlow 데이터를 기반으로 네트워크 트래픽의 동적 특성을 포착하는 일반-purpose 머신 러닝 모델을 사전 학습합니다. 이 모델은 다양한 다운스트림 작업을 위해 소량의 라벨만으로 미세 조정(fine-tuning)될 수 있으며, 다양한 네트워킹 태스크에 효과적으로 적용됩니다.

- **Technical Details**: NetFlowGen은 ISP에서 수집된 대규모 네트워크 트래픽 데이터를 활용하여 템플릿화된 피처 임베딩 파이프라인을 생성합니다. 이를 통해 IP, 타임스탬프, 포트, 전송 프로토콜, 패킷, 바이트 등 다양한 트래픽 피처를 통합하여 전처리된 모델 내에서 일반화된 공간으로 변환하여 학습하게 됩니다. 모델은 주로 Transformer 아키텍처를 사용하고, 주어진 속성에 따른 다음 타임스텝을 예측하는 생성적(pre-training) 과정을 따릅니다.

- **Performance Highlights**: 제안된 프레임워크는 DDoS 공격 감지 등 실제 작업에 대한 적응성과 트래픽의 동적 특성을 효과적으로 포착하는 성과를 보여줍니다. 실험 결과, 86개의 트래픽 피처가 고려되었으며, 제한된 라벨 데이터로도 네트워크 보안의 다양한 공격 탐지 작업에서 개선된 성능이 나타났습니다. NetFlowGen의 초기 결과는 자가 감독(pre-training) 학습 프레임워크가 네트워킹 연구에서 유망한 이점을 제공함을 시사합니다.



### HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models (https://arxiv.org/abs/2412.20622)
- **What's New**: 본 논문에서는 LVLMs (Large Vision-Language Models)에서 발생하는 객체 환각(object hallucination) 문제를 해결하기 위한 새로운 벤치마크인 HALLUCINOGEN을 제안합니다. 기존 벤치마크는 주로 단순한 개체 확인 프롬프트에 의존했으나, HALLUCINOGEN은 다양한 맥락적 추론(prompt) 능력을 활용하여 LVLMs의 객체 식별 정확성을 평가합니다. 이와 함께, MED-HALLUCINOGEN이라는 의료 분야에 맞춤 형태의 환각 공격을 도입하여, 의료 이미지에서의 LVLMs 성능을 검증하고자 합니다.

- **Technical Details**: HALLUCINOGEN 벤치마크에서는 60,000개의 이미지-프롬프트 조합을 포함한 3,000개의 시각-객체 쌍을 통해 LVLMs의 정확한 객체 식별 능력을 검토합니다. 두 가지 유형의 환각 공격인 정확한(object hallucination) 및 암시적(implicit) 공격을 구분하여 평가합니다. 메디컬 분야에서의 환각 성능을 측정하기 위해 NIH Chest X-rays 데이터셋을 이용한 MED-HALLUCINOGEN을 설계하였으며, 이를 통해 감독이 필요한 문맥에서 LVLMs의 신뢰성을 평가하고자 합니다.

- **Performance Highlights**: 여덟 개의 LVLM과 두 가지 환각 완화 전략에 대한 평가 결과, HALLUCINOGEN 및 MED-HALLUCINOGEN의 환각 공격에 대해 대부분의 SOTA LVLM들은 무작위 추측에 가까운 성능을 보임을 확인하였습니다. 특히 LVLM이 Chain-of-Thought (CoT) 추론을 사용할 경우 환각 현상이 더욱 증가하는 경향이 있음을 보여주었습니다. 이를 통해, 현재 LVLM들은 심각한 환각 공격에 취약하다는 것을 입증하였습니다.



### Towards Explaining Uncertainty Estimates in Point Cloud Registration (https://arxiv.org/abs/2412.20612)
- **What's New**: 이번 연구는 Iterative Closest Point (ICP) 알고리즘에 대한 새로운 접근 방식을 제안합니다. 최근 Explainable AI의 발전을 활용하여, 확률적 ICP 방법이 생성한 출력의 불확실성을 설명할 수 있는 방법을 제시합니다. 이 방법은 Kernel SHAP을 기반으로 하여, 센서 노이즈, 가림 현상, 불확실한 환경 등 ICP의 일반적인 불확실성 원인에 중요성을 부여합니다.

- **Technical Details**: 이 연구에서는 ICP가 두 점 구름(포인트 클라우드) 간의 변환을 추정하는 알고리즘이라는 점을 강조합니다. ICP는 초기 변환 추정치를 바탕으로 반복적으로 매칭된 포인트 간의 유클리드 거리(Euclidean distance)를 최소화합니다. 불확실성이 존재하는 상황에서, SHAP 커널을 사용하여 모델에 구애받지 않는 설명 모듈을 개발하고 불확실성 원인과 추정된 불확실성 간의 연관성을 식별하여, 최종적으로 더 낮은 불확실성을 달성할 수 있도록 연구했습니다.

- **Performance Highlights**: 실험 결과, 제안된 설명 방법이 불확실성의 원인을 적절하게 설명할 수 있음을 보여주었습니다. 이는 로봇들이 언제, 왜 실패했는지를 인간이 이해할 수 있는 방식으로 알려줄 수 있는 기초를 제공하는 결과로 이어집니다. 이러한 접근은 ICP 알고리즘의 직관적 이해를 높여줄 뿐 아니라, 로봇의 환경 인식 능력을 향상시킬 것으로 기대됩니다.



### MATEY: multiscale adaptive foundation models for spatiotemporal physical systems (https://arxiv.org/abs/2412.20601)
- **What's New**: 새로운 연구에서는 spatiotemporal physical systems에 대한 정밀한 표현을 제공하기 위해 MATEY라는 multiscale adaptive foundation model을 제안합니다. 이 모델은 두 가지 adaptive tokenization 방식을 도입하여 로컬 특징에 따라 패치 크기를 동적으로 조정합니다. 두 방법 모두 계산 효율성을 개선하고, 비슷한 정확도를 유지하면서도 시퀀스 길이를 최소화하는 장점을 제공합니다.

- **Technical Details**: MATEY 모델은 adaptive mesh refinement (AMR) 기법에서 영감을 받아 각 시스템의 로컬 특징에 기반하여 패치 크기를 동적으로 조정하는 adaptive tokenization 방법을 사용합니다. 이 모델에서는 axial attention 기법을 기반으로 한 spatiotemporal attention 방식도 채택하여 긴 spatiotemporal 시퀀스를 효과적으로 분해합니다. 이러한 접근법을 통해 MATEY는 물리적 시스템의 고해상도 데이터에 대한 성능을 개선합니다.

- **Performance Highlights**: 모델의 성능은 두 개의 서로 다른 물리적 환경에서 검증되었으며, PDEBench 데이터(미분 방정식 벤치마크)에서 사전훈련된 모델이 무작위 초기화된 모델보다 더 우수한 성능을 보였습니다. 특히 저데이터 환경에서는 사전훈련된 모델이 더욱 뛰어난 결과를 나타냈습니다. MATEY는 전반적인 정확도를 증가시키면서도 필요한 훈련 시간과 모델 가중치를 줄일 수 있는 가능성을 보여줍니다.



### Controlling Out-of-Domain Gaps in LLMs for Genre Classification and Generated Text Detection (https://arxiv.org/abs/2412.20595)
Comments:
          The 31st International Conference on Computational Linguistics

- **What's New**: 이번 연구는 최신 대규모 언어 모델(LLMs, 예: GPT-4)이 이전 연구에서 발견된 것과 동일한 도메인 외 성능 격차(out-of-domain performance gap)를 겪고 있음을 보여줍니다. 특히, 우리는 두 가지 비주제별 분류 작업인 장르 분류와 생성된 텍스트 탐지에서 이 성능 저하가 어떻게 발생하는지를 시연합니다. 우리는 In-Context Learning (ICL) 접근 방식으로 서로 다른 도메인에서 성능 저하를 경험하며, 이를 해결하기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 예제 도메인에 따라 분류 성능이 달라지는 현상을 제어하기 위한 방법을 소개합니다. 이 접근 방식은 텍스트의 주제적 특성을 배제하고 스타일적 속성에 초점을 맞추도록 모델을 유도하는 데 효과적입니다. 특히, 장르 분류와 AI 생성 텍스트 탐지라는 두 가지 작업에서 수행하였으며, 각 작업의 OOD 격차를 최대 20%까지 줄이는 데 성공했습니다.

- **Performance Highlights**: 우리의 연구 결과는 기존의 Chain-of-Thought (CoT) 방법론이 충분히 효과적이지 않음을 보여주며, 제안한 접근 방식이 도메인 전이 성능을 지속적으로 향상시킨다는 점에서 의미가 큽니다. 특히, 소수의 예제(few-shot) 설정에서도 두 작업에서 각각 7%와 20%까지 성능 격차를 줄였습니다. 이러한 결과는 LLM들이 다른 도메인에서 잘 작동할 수 있는 가능성을 열어줍니다.



### Kryptonite-N: Machine Learning Strikes Back (https://arxiv.org/abs/2412.20588)
- **What's New**: 이번 논문에서는 Quinn et al의 "Kryptonite-N" 데이터셋을 반박하며, 다양한 모델들이 이 데이터셋에서 좋은 성과를 낸다고 주장합니다. 데이터 탐색과 실험 디자인을 통해 Universal Function Approximation의 유효성을 증명하며, 기존의 주장에 대한 비판을 포함하고 있습니다.

- **Technical Details**: Kryptonite-N 데이터셋을 다루기 위해 파라메트릭 모델을 학습하고, 이를 최적화하는 과정에서 각 특성의 표준화(standardization)를 적용합니다. 이론적으로, 신경망(Neural Network)은 비선형적인 관계를 모델링하는 데 효과적이므로, 이를 통해 데이터셋을 해결하는 과정을 설명하고 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 충분한 다항식 확장과 L1 정규화가 적용될 경우, Kryptonite 데이터셋에 대해 성능 좋은 결과를 도출할 수 있음을 보여줍니다. 이러한 성과는 기존의 "Kryptonite-N" 논문에서 주장된 비판을 극복하는 데 기여하며, 머신러닝 모델에 대한 신뢰성을 높이는 결과로 이어집니다.



### Bridging the Gap: A Decade Review of Time-Series Clustering Methods (https://arxiv.org/abs/2412.20582)
- **What's New**: 본 논문에서는 전통적인 시계열 클러스터링 방법과 최신 딥러닝 기반 알고리즘 간의 간극을 메우는 포괄적인 분류체계(taxonomy)를 제시합니다. 또한, 다양한 분야에 걸쳐 시계열 데이터의 변별력을 높이고 이러한 복잡한 데이터에서 숨겨진 패턴을 발견하는 데 기여하는 클러스터링 기술의 발전을 다룹니다. 이를 통해 향후 시계열 클러스터링 연구에 대한 통찰을 제공합니다.

- **Technical Details**: 시계열 데이터는 차원에 따라 univariate, multivariate, tensor 필드 등으로 나뉘며, 데이터 수집 시 샘플링 전략에 따라 정규와 비정규로 분류됩니다. 시계열 클러스터링의 문제 정의와 클러스터링 프로세스의 일반적인 파이프라인을 설명하며, 이러한 과정은 후속 섹션에서 제안된 새로운 분류체계와 밀접한 관련이 있습니다. 이 논문에서는 기존의 클러스터링 알고리즘인 k-Means의 한계와 동적 시간 왜곡(DTW)에 의한 새로운 접근 방식에 대해서도 설명합니다.

- **Performance Highlights**: 딥러닝 시대에 의해 등장한 다양한 클러스터링 방법들은 공통적으로 비슷한 특징을 공유하는 데이터 샘플을 그룹화할 수 있는 가능성을 제공합니다. 이는 데이터의 본질적인 왜곡에 강인한 방법을 제안하며, 효율적인 병렬 처리와 고급 GPU 자원을 활용하여 훈련 및 배포 시간을 획기적으로 단축시는 기술적 진전을 포함합니다. 이러한 방식은 차원 공간에서의 데이터 표현력을 높이고 기존 클러스터링 작업의 성능을 개선할 수 있는 기회를 제공합니다.



### A Survey on Time-Series Distance Measures (https://arxiv.org/abs/2412.20574)
- **What's New**: 이 연구는 7가지 범주로 분류된 100가지 이상의 최첨단 거리 측정치를 종합적으로 고려하고 있습니다. 이를 통해 단일 변수(univariate) 및 다변수(multivariate) 경우의 응용 및 차별성을 다루며, 시계열 거리 측정의 혁신적인 발전을 위한 기초를 제시합니다.

- **Technical Details**: 시간 시리즈 데이터의 원활한 처리를 위해 거리 함수(distance functions)는 신호 간 비유사성을 정의하는 데 필수적입니다. 본 연구에서는 Euclidean Distance(유클리드 거리) 및 Dynamic Time Warping(DTW)와 같이 다양한 측정 기능을 포함하는 거리 척도를 설명하며, 비선형적인 형태의 변형을 처리하기 위한 복잡성 및 최적화 전략을 논의합니다.

- **Performance Highlights**: Shape-based Distance(SBD)는 Fast Fourier Transform(FFT)를 활용하여 시간 복잡성을 𝒪⁢(n⁢log⁡(n))으로 줄이고, 이는 시간 민감한 작업에서의 적용을 용이하게 합니다. 이 연구는 또한 다양한 거리 측정의 상대적 강점과 약점을 밝혀내고, 적합한 측정 방법 선택의 중요성을 강조합니다.



### Segmentation of Muscularis Propria in Colon Histopathology Images Using Vision Transformers for Hirschsprung's Diseas (https://arxiv.org/abs/2412.20571)
Comments:
          To be published in the CMBEC47/ACCES26 Joint Conference

- **What's New**: 이번 연구에서는 Hirschsprung's disease (HD)에서 근육층(muscularis propria) 분할(segmentation)을 위해 Vision Transformers (ViTs)를 적용하고 그 성능을 CNN 및 얕은 학습(shallow learning) 방법과 비교했습니다. ViTs는 자기 주의(self-attention) 메커니즘 덕분에 최근 강력한 딥러닝(deep learning) 접근법으로 부각되고 있습니다. 이러한 접근법은 병리학 이미지 분석의 자동화를 가능하게 하여 병리학자의 업무를 효율적으로 지원할 수 있습니다.

- **Technical Details**: 연구의 주요 목표는 calretinin 염색(histopathology images)된 조직 슬라이드에서 근육층의 정확한 분할을 구현하는 것으로, ViT 모델은 DICE 점수(dice score) 89.9%와 Plexus Inclusion Rate (PIR) 100%를 달성했습니다. 이는 CNN 모델의 DICE 점수(89.2%; PIR 96.0%) 및 k-평균(k-means) 클러스터링 방법(80.7%; PIR 77.4%)보다 뛰어난 성과입니다. 이러한 결과는 ViTs가 HD 관련 이미지 분석을 위한 유망한 도구임을 입증합니다.

- **Performance Highlights**: ViT 모델은 높은 정확성과 함께 JPEG 처리와 같은 이미지 전처리(preprocessing)의 필요성을 줄여주는 장점을 가지고 있습니다. 병리학적 이미지 분석에서 ViTs를 활용하면 인력과 자원의 효율성을 높일 수 있습니다. 이러한 혜택을 통해 HD의 조기 진단 및 관리에 기여할 수 있는 가능성이 큽니다.



### Enhancing autonomous vehicle safety in rain: a data-centric approach for clear vision (https://arxiv.org/abs/2412.20565)
Comments:
          16 pages, 16 figures, 2 tables

- **What's New**: 본 연구는 비가 오는 날씨에서 자율주행차(AV)의 내비게이션 문제를 해결하기 위해 최신 딥러닝 기술을 활용하여 비로 인한 시각적 방해를 제거하는 비전 모델을 개발합니다. 이 모델은 자동차 캠코더의 실시간 피드를 처리하여 맑고 비가 오지 않는 장면과 유사한 비주얼을 생성합니다. 새로운 배치 전략을 통해 비가 내리는 장면에서 높은 주파수의 비 패턴을 효과적으로 구분하여 모델 학습과 추론 성능을 향상시키는 점에서 차별점을 가지고 있습니다.

- **Technical Details**: 이 연구에서는 Car Learning to Act (CARLA) 시뮬레이션 환경을 활용하여 비 오는 이미지와 그에 상응하는 맑은 이미지를 포함하는 종합적인 데이터셋을 생성합니다. 모델 아키텍처는 전통적인 encoder-decoder 구조를 기반으로 하며, skip connection 및 concatenation 연산을 포함하여 고해상도 이미지를 처리할 수 있게 확장됩니다. 두 가지 새로운 배치 기법을 도입하여 비가 내리는 패턴과 배경 장면을 효과적으로 처리한 결과, 모델은 안정성 및 추론 성능을 향상시킵니다.

- **Performance Highlights**: 결과적으로 개발한 모델은 steering module과 통합하여 비가 오는 날씨에서도 운전 성능을 크게 향상시킵니다. 비가 내리는 조건과 맑은 조건에서의 스티어링 성능을 비교한 결과, 이 모델을 통해 AV의 내비게이션 안전성과 신뢰성이 크게 개선됨을 보여주었습니다. 모델의 성능을 정량적으로 평가하기 위해 PilotNet을 사용하였으며, 여기에 의해 예측된 스티어링 각도가 맑은 이미지에서와 유사한 결과를 나타냅니다.



### Attacks on the neural network and defense methods (https://arxiv.org/abs/2412.20529)
- **What's New**: 이 논문은 오디오 데이터로 훈련된 신경망(neural network)에 대한 공격의 사용 및 이러한 공격에 대한 가능한 방어 방법을 논의합니다. FGSM, PGD, CW 공격과 데이터 중독(data poisoning)에 대해 다룹니다.

- **Technical Details**: 보호 기술 측면에서 Art-IBM과 advertorch 라이브러리가 검토됩니다. 이러한 도구들은 신경망의 취약성을 완화하고 공격으로부터 시스템을 보호하는 데 사용될 수 있습니다.

- **Performance Highlights**: 최종적으로, 공격 적용 범위 내에서 얻어진 정확도(metrics)를 제시합니다. 이를 통해 신경망이 얼마나 공격에 강한지를 평가할 수 있는 지표를 제공합니다.



### Game Theory and Multi-Agent Reinforcement Learning : From Nash Equilibria to Evolutionary Dynamics (https://arxiv.org/abs/2412.20523)
Comments:
          22 pages

- **What's New**: 이번 논문에서는 복잡한 다중 에이전트 시스템에 대한 고급 주제를 탐구하며, 특히 Multi-Agent Reinforcement Learning (MARL)의 네 가지 기본적인 도전 과제를 다룹니다. 이들 과제는 비정상성(non-stationarity), 부분 관측성(partial observability), 대규모 에이전트 인구 scalability와 분산 학습(decentralized learning)입니다. 게임 이론의 개념을 통합하여 이러한 도전 과제를 해결하는 알고리즘적 진전을 분석합니다.

- **Technical Details**: MARL에서의 비정상성은 여러 에이전트의 정책 변화가 환경의 역학에 영향을 미치기 때문에 발생합니다. 각 에이전트의 정책 업데이트가 환경을 변화시켜 다른 에이전트에게는 끊임없이 움직이는 목표 설정을 의미합니다. 논문에서는 특히 Deep Q-Networks (DQN)와 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)와 같은 최신 MARL 알고리즘을 소개하며, 이들이 협력적 및 경쟁적 상호작용을 처리할 수 있도록 발전된 방법을 제시합니다.

- **Performance Highlights**: 이 연구는 게임 이론과 MARL의 통합이 복잡하고 동적인 환경에서 다중 에이전트 시스템의 효과성과 강인성을 어떻게 향상시킬 수 있는지를 보여줍니다. 비정상성, 부분 관측성, scalability, 그리고 분산 협력의 문제를 해결하기 위한 지속적인 연구가 금융, 로봇공학 등의 분야에서 상당한 발전을 이끌 것으로 기대됩니다. 전체적으로, MARL의 이론적 기초와 실제적 함의를 강조하여 다중 에이전트 시스템의 효율성을 크게 향상시킬 수 있는 방법을 제시하고 있습니다.



### Goal-Conditioned Data Augmentation for Offline Reinforcement Learning (https://arxiv.org/abs/2412.20519)
- **What's New**: 이번 연구에서는 오프라인 강화 학습(offline reinforcement learning)에서 데이터 품질 향상을 위해 Goal-conditioned Data Augmentation(GODA)라는 새로운 방법을 소개합니다. GODA는 목표 지향적(diffusion-based) 데이터 증강 기법으로, 이전의 수집된 데이터를 활용하여 더 높은 품질의 샘플을 생성합니다. 이 방법은 기존 데이터를 바탕으로 특정 목표(return-oriented)를 조건으로 하여 새로운 데이터를 생성함으로써 제한된 최적 시나리오의 유용성을 극대화합니다.

- **Technical Details**: GODA는 최근 발전한 생성적인 모델링을 활용하고 있으며, 목표인지 샘플을 선별하기 위한 다양한 선택 메커니즘과 제어 가능한 스케일링 기법을 도입합니다. 연구에서 제안하는 'goal'은 return-to-go(RTG)로 정의되며, 이는 특정 시점까지의 누적 보상을 나타내어 향후 보상을 명확히 예측할 수 있게 해줍니다. GODA는 입력 정보 처리를 위한 새로운 적응형 게이트 조건 방식도 도입하여 목표 지향적 인도를 향상시킵니다.

- **Performance Highlights**: GODA는 D4RL 벤치마크 및 실제 세계 과제인 교통 신호 제어(TSC) 작업에서 실험하여 데이터 품질을 향상시키는 데 효과적임을 입증하였습니다. 기존의 최첨단 데이터 증강 방법 대비 우수한 성능을 보였으며, 특히 제한된 데이터 세트를 처리하는데 있어 GODA의 기여도가 큽니다. 이러한 평가를 통해 GODA는 강화 학습 기반의 방법들이 실제 세계 시나리오에서 더욱 적용될 수 있도록 하는 데 기여할 수 있음을 확인하였습니다.



### Dive into Time-Series Anomaly Detection: A Decade Review (https://arxiv.org/abs/2412.20512)
- **What's New**: 최근 데이터 수집 기술의 발전과 함께 스트리밍 데이터의 양과 속도가 증가하면서 시계열 분석의 필요성이 강조되고 있습니다. 특히, 시계열 이상 탐지는 사이버 보안, 금융 시장, 법 집행, 의료와 같은 다양한 분야에서 매우 중요한 활동으로 부각되고 있습니다. 이 논문에서는 전통적인 통계적 방법 대신 머신 러닝 알고리즘의 증가에 따라 시계열 이상 탐신을 위한 연구 방법론의 구조적 특징을 정리하였습니다.

- **Technical Details**: 이 논문은 시계열 맥락에서 이상 탐지 솔루션을 프로세스 중심의 분류법(process-centric taxonomy) 아래 그룹화하고 요약합니다. 또한 이상 탐지 방법의 원래 분류를 제공하며, 문헌 메타 분석을 통해 시계열 이상 탐지 연구의 일반적인 경향을 정리합니다. 이를 통해 데이터 생성의 복잡성과 측정 시스템의 불완전성이 시계열 데이터에서 이상 현상(anomalies)을 발생시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 시간 시리즈의 이상 탐지에 대한 연구는 60년 이상의 역사를 가지며, 다양한 이상 패턴을 분석하는 과정은 상당히 도전적인 작업입니다. 이 논문은 기존의 일반적인 이상 탐지 방법론과는 달리 시계열의 특정 특성을 고려하여 설계된 방법들에 중점을 두고 있습니다. 이를 통해 IOT(Internet-of-Things) 애플리케이션의 폭발적인 증가로 인해 시계열 데이터 내에서 많은 이상이 발생할 것임을 예측하고 있습니다.



### Stratify: Unifying Multi-Step Forecasting Strategies (https://arxiv.org/abs/2412.20510)
Comments:
          30 pages, 9 figures, journal

- **What's New**: 이번 논문에서는 다단계 예측(multi-step forecasting, MSF)을 위한 새로운 매개변수화된 프레임워크인 Stratify를 소개합니다. Stratify는 기존의 여러 예측 전략을 통합하고 새로운 전략을 도입하여 예측 전략 선택의 체계적 탐색을 가능하게 합니다. 연구 결과에 따르면, Stratify의 새로운 전략이 기존의 모든 전략에 비해 성능이 향상되었음을 보여줍니다.

- **Technical Details**: Stratify는 기존의 단일 출력 전략을 다중 출력(multi-output) 전략으로 전환 가능하게 하는 새로운 접근 방식을 제시합니다. Stratify의 매개변수화 프레임워크를 통해 반드시 최적의 전략을 선택해야 하는 현업의 시행착오를 감소시키고, 각 작업과 데이터셋에 대한 예측 전략 탐색을 용이하게 합니다. 또한, 다단계 예측 전략의 성능이 상대적으로 부드럽게 나타나 최적화의 가능성을 더욱 효과적으로 시사합니다.

- **Performance Highlights**: 정밀한 실험 평가를 통해 Stratify는 18개의 벤치마크 데이터셋과 10, 20, 40, 80의 다양한 예측 지평선에서 기존의 최첨단 전략보다 일관되게 높은 성능을 보였습니다. 실험에서 1080회의 실험 중 84% 이상의 비율로 Stratify의 새로운 전략이 기존 전략보다 뛰어난 성능을 기록하였습니다. 이러한 결과는 MSF 전략의 포괄적인 벤치마킹을 가능하게 하며, 코드도 제공되어 결과 재현이 가능합니다.



### A Multiparty Homomorphic Encryption Approach to Confidential Federated Kaplan Meier Survival Analysis (https://arxiv.org/abs/2412.20495)
Comments:
          40 pages

- **What's New**: 이 논문에서는 헬스케어 데이터를 활용한 연구에 있어, 민감한 환자 기록을 결합하기 힘든 문제를 해결하기 위해 
다수 당사자 동형 암호(Multiparty Homomorphic Encryption) 기반의 프레임워크를 제안합니다. 특히 본 프레임워크는 
프라이버시를 유지하면서 여러 기관에서의 Kaplan–Meier 생존 분석을 지원하며, 이론 모델과 공격 완화 방안도 포함되어 있습니다. 다른 연구들과 비교했을 때, 우리 프레임워크는 암호화된 생존 추정치가 중앙 집중식 결과와 유사하게 일치하도록 보장하며, 유틸리티 손실 한계를 공식적으로 제시합니다.

- **Technical Details**: 본 연구는 데이터 암호화를 위한 CKKS(Chevalier et al.) 암호화를 사용하여 생존 시간 예측의 정밀도를 
향상시킵니다. 또한, 데이터 overlapping이 발생하는 경우의 재구성 공격의 위험을 정량화하고, 이에 대한 방어 메커니즘을 구축하여 
프라이버시와 확장성을 보장합니다. 논문에서는 다양한 데이터셋에서의 실험을 통해 전반적인 성능을 검증하며,
암호화된 생존 곡선과 비암호화 생존 곡선 간의 오차가 미미함을 입증하였습니다.

- **Performance Highlights**: 실험 결과, NCCTG 폐암 및 합성 유방암 데이터셋을 활용한 결과, 평균 절대 오차(MAE)와 
제곱근 평균 제곱 오차(RMSE)가 낮음을 확인하였습니다. 또한, 로그 순위 및 정밀도 테스트에서 
암호화된 분석과 비암호화된 분석 간의 유의미한 차이가 없음을 보여주어 통계적 유효성을 유지하는 것으로 나타났습니다. 
비록 계산 비용이 8-19배 증가했지만, 본 연구의 동형 암호화는 중간 규모 배포에서의 실현 가능성을 확인하여 
안전성과 실행 속도 간의 균형을 잘 이뤘습니다.



### Integrating Natural Language Processing Techniques of Text Mining Into Financial System: Applications and Limitations (https://arxiv.org/abs/2412.20438)
Comments:
          6 pages, 5 figures, 1 table

- **What's New**: 이번 연구는 2018년부터 2023년까지의 기간 동안 금융 시스템의 여러 구성 요소에서 자연어 처리(Natural Language Processing, NLP) 기법으로서 텍스트 마이닝의 활용을 탐구하였습니다. 자산 가격 설정(asset pricing), 기업 금융(corporate finance), 파생상품(derivatives), 위험 관리(risk management) 및 공공 금융(public finance) 등 다양한 분야에서의 적용 사례를 다루고 있습니다. 연구는 특정 문제를 논의하며 이를 해결할 필요성을 강조하고 있습니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론은 주로 확률적(probabilistic) 모델과 벡터 공간(vector-space) 모델의 결합입니다. 정보 처리에 가장 많이 사용되는 기술은 정보 분류(information classification) 기법이며, 롱-숏 메모리(long-short term memory)와 양방향 인코더(bidirectional encoder) 모델이 많이 사용됩니다. 연구 결과, 자산 가격 설정에 대한 집중적인 관심과 함께 새로운 알고리즘이 개발되고 있다는 점도 확인되었습니다.

- **Performance Highlights**: 금융 텍스트를 분석해야 하는 연구자들에게 엔지니어링 관점에서의 경로를 제시하고 있으며, 텍스트 마이닝에 관련된 데이터 품질, 맥락 적응(context-adaption), 모델 해석 가능성(model interpretability) 등의 과제를 해결해야 한다고 주장합니다. 이러한 문제들을 해결함으로써, 금융 분석 및 예측을 향상시키기 위한 고급 자연어 처리 모델 및 기법의 통합이 가능할 것입니다.



### Multi-Scenario Reasoning: Unlocking Cognitive Autonomy in Humanoid Robots for Multimodal Understanding (https://arxiv.org/abs/2412.20429)
Comments:
          The main text is 5 pages, 2 figures, and 3 tables

- **What's New**: 이번 연구는 휴머노이드 로봇의 인지 자율성을 향상시키기 위한 다중 시나리오 추론 아키텍처를 제안합니다. 이 아키텍처는 시뮬레이션 기반의 실험 설계를 사용하여 다중 모달(visual, auditory, tactile) 이해의 기술적 한계를 해결하고자 하였습니다. 연구는 'Maha'라는 시뮬레이터를 구축하여 수행되었습니다.

- **Technical Details**: 제안된 아키텍처는 다중 모달 합성을 채택하여 다양한 감각 데이터를 통합하고, 이를 통해 동적 환경에서의 크로스 모달(모달 간) 상호작용 전략 탐색을 지원합니다. 연구에서는 시뮬레이터 'Maha'를 통해 다중 모달 데이터의 실현 가능성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 아키텍처는 다중 모달 데이터를 효과적으로 처리하고, 휴머노이드 로봇의 복잡한 환경에서의 상호작용 능력을 향상시킬 수 있음을 보여주었습니다. 이는 동적 환경 내에서의 휴머노이드 로봇의 행동 및 판단능력을 개선하는 데 기여할 것으로 기대됩니다.



### Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection (https://arxiv.org/abs/2412.20414)
Comments:
          6 pages, 1 table, AICCONF '24: Cognitive Models and Artificial Intelligence Conference, Istanbul, Turkey

- **What's New**: 이번 논문은 다국어 텍스트에서 지리적 데이터(geospatial data)의 추출 및 분석을 위한 최첨단 자연어 처리(NLP) 방법론과 대규모 언어 모델(LLMs) 통합의 중요성을 다룹니다. 특히, 다양한 NLP 모델(SpaCy, XLM-RoBERTa, mLUKE, GeoLM)과 OpenAI의 GPT 3.5 및 GPT 4를 평가하여 다국어 지리 엔티티 탐지의 맥락에서 이들의 성능을 분석합니다.

- **Technical Details**: 논문에서는 영어, 러시아어, 아랍어의 Telegram 채널에서 수집한 데이터셋을 활용하여 모델의 성능을 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1 scores) 등의 지표를 통해 평가합니다. 각 모델의 장점과 도전 과제를 드러내어 다양한 언어 환경에서 정확한 지리 엔티티 식별(geo-entity identification)의 복잡성을 강조합니다.

- **Performance Highlights**: 실험 결과, 각 모델의 성능 차이를 명확히 하여 지리적 참조 식별에 대한 효과성을 확인하였습니다. 이를 통해 고급 NLP 도구의 개선 및 개발 방향이 제시되어 지리적 분석 및 글로벌 보안(global security) 적용 분야의 발전에 기여하고자 합니다.



### Multi-Objective Large Language Model Unlearning (https://arxiv.org/abs/2412.20412)
- **What's New**: 최근 대형 언어 모델(LLM)에서의 머신 언렀닝(machine unlearning)이 큰 주목을 받고 있으며, 이는 모델의 불필요한 행동을 효과적으로 제거하는 방법을 제공합니다. 본 논문에서는 LLM 언렀닝에서의 Gradient Ascent(GA) 접근 방식을 탐구하며, 대상 데이터의 예측 확률을 감소시켜 영향력을 제거하는 프로액티브한 방법을 제시합니다. 이를 통해 우리는 기존의 문제를 해결하는 Multi-Objective Large Language Model Unlearning(MOLLM) 알고리즘을 제안하여, 모델의 유용성을 유지하면서 대상 데이터를 잊도록 모델을 업데이트합니다.

- **Technical Details**: MOLLM 알고리즘은 LLM 언렀닝을 다중 목표 최적화 문제로 정의하고, 크로스 엔트로피 손실을 언렀닝 버전으로 수정하여 그래디언트 폭발 문제를 해결합니다. 이 프로세스에서는 공통적인 하강 방향(common descent direction)을 계산하여 모델이 목표 데이터를 잊으면서도 LLM의 유용성을 보존할 수 있도록 합니다. 또한, KL 다이버전스(Kullback-Leibler divergence)와 크로스 엔트로피 손실을 통합하여 모델의 성능을 유지하는 방향으로 나아갑니다.

- **Performance Highlights**: 실험 결과, MOLLM은 기존 SOTA GA 기반 LLM 언렀닝 방법들보다 언렀닝 효과와 모델 유용성 유지 측면에서 우수한 성능을 보였습니다. SafeRLHF 데이터셋을 사용해 검증한 결과, 제안된 방법은 언렀닝 효율성과 유틸리티 보존 간의 균형을 잘 유지 көрс합니다. 이는 앞으로의 LLM 언렀닝 연구에 중요한 참고 자료가 될 것입니다.



### Natural Language Fine-Tuning (https://arxiv.org/abs/2412.20382)
- **What's New**: 이 논문에서는 자연 언어를 활용한 자연어 파인튜닝(Natural Language Fine-Tuning, NLFT)이라는 새로운 기법을 소개합니다. NLFT는 특정 도메인에서 제한된 데이터를 가지고 파인튜닝을 가능하게 하며, 모델의 성능을 크게 향상시킵니다. 기존 방법들과 달리 NLFT는 파인튜닝 과정에서 자연어의 지침을 사용하고, 토큰 수준에서의 최적화를 통해 훈련 효율성을 높입니다.

- **Technical Details**: NLFT는 자연어를 감독 신호로 사용하여 데이터를 효과적으로 활용합니다. 이 방법은 또한 warm-up 과정이 필요 없는 최소 데이터 파인튜닝 방법론을 제안하여 기존의 ReFT 방식보다 반복훈련 수를 줄입니다. NLFT는 O(n) 알고리즘 복잡성을 유지하면서 문자 단위의 확률 변화를 평가하고, saliency 토큰을 할당하여 손실 함수를 정제합니다.

- **Performance Highlights**: GSM8K 데이터셋에서 NLFT는 50개의 데이터 샘플만으로 SFT보다 219% 높은 정확도를 달성했습니다. 또한 NLFT는 ReFT보다 훈련 시간과 메모리 사용의 효율성을 각각 78.27%와 92.24% 줄였습니다. 이러한 결과는 NLFT가 리소스가 제한된 환경에서 효과적으로 응용될 수 있는 가능성을 보여줍니다.



### A Deep Subgrouping Framework for Precision Drug Repurposing via Emulating Clinical Trials on Real-world Patient Data (https://arxiv.org/abs/2412.20373)
Comments:
          To be published in KDD 2025

- **What's New**: 이번 연구에서는 STEDR라는 새로운 약물 재사용 프레임워크를 소개합니다. 이 프레임워크는 하위 집단 분석(subgroup analysis)과 치료 효과 추정(treatment effect estimation)을 통합하여 기존의 방법들이 간과한 특정 하위 집단에 대한 영향력을 조명합니다. STEDR는 실제 환자 데이터를 기반으로 여러 임상 시험을 모방하는 방식으로 재사용 후보약물을 식별하며, 알츠하이머병(Alzheimer's Disease) 연구에 활용되었습니다.

- **Technical Details**: STEDR는 복잡한 환자 데이터를 처리하기 위해 이중 수준의 주의(attention) 메커니즘을 설계하여 고차원 정보를 효과적으로 인코딩합니다. 이 프레임워크는 변분 오토인코더(Variational Auto-Encoder, VAE)를 활용하여 환자 집단 내의 이질적인 하위 집단을 식별합니다. 하위 집단의 치료 효과를 추정하기 위해 관측 가능한 공변량(observational covariates)과 하위 집단 특성을 같은 맥락에서 동시 분석합니다.

- **Performance Highlights**: STEDR는 기존 방법들보다 약물 재사용 후보를 식별하는 데 더 우수한 성능을 보였습니다. 연구 결과, 8백만 이상의 환자 데이터를 기반으로 한 대규모 데이터베이스에서 알츠하이머병에 효과적인 14개의 약물 후보를 발견하였습니다. 또한, STEDR는 임상적으로 중요한 환자 하위 집단을 효과적으로 특성화할 수 있어 정밀 약물 재사용을 위한 가능성을 보여줍니다.



### LLM2: Let Large Language Models Harness System 2 Reasoning (https://arxiv.org/abs/2412.20372)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 한계점을 극복하기 위해 LLM2라는 새로운 프레임워크를 소개합니다. LLM2는 인간의 인지 이론인 이중 과정(The dual-process theory)에서 영감을 받아 LLM(시스템 1)과 프로세스 기반 검증기(시스템 2)를 결합하였습니다. 이를 통해 LLM은 출력 후보를 생성하고, 검증기는 적절한 피드백을 제공하여 바람직한 결과와 그렇지 않은 결과를 구분하는 역할을 수행합니다.

- **Technical Details**: LLM2는 훈련 단계에서 쌍 비교 손실(pairwise comparison loss)을 이용하여 바람직한 토큰과 바람직하지 않은 토큰을 구분하도록 최적화된 프로세스 기반 검증기를 포함하고 있습니다. 이 검증기는 각 후보에 대한 시의적절한 피드백을 제공하여 LLM이 생성하는 후보의 품질을 향상시킵니다. 이를 위해 고안된 토큰 품질 탐색 전략을 통해 생성된 합성(process-supervision) 데이터를 사용합니다.

- **Performance Highlights**: 실험 결과, LLM2는 GSM8K 및 MATH와 같은 수학적 추론 데이터셋에서 성능 향상을 보여주었습니다. 예를 들어, Llama3-1B 모델의 경우 GSM8K에서 정확도가 50.3에서 57.8로 (+7.5) 향상되었고, LLM2에 자가 일관성(self-consistency)을 결합했을 때 주요 정확도가 56.2에서 70.2로 (+14.0) 증가했습니다. 이러한 결과는 합성 과정 감독 데이터의 효과성과 가능성을 잘 보여줍니다.



### Safe Multiagent Coordination via Entropic Exploration (https://arxiv.org/abs/2412.20361)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구는 제약이 있는 다중 에이전트 강화 학습(Constrained Multiagent Reinforcement Learning)에서 팀 제약을 정의하고, 이를 통해 팀 수준에서의 안전성 개념을 확인합니다. 제안된 방법인 엔트로픽 탐색(E2C)은 에이전트들이 협력하여 안전하고 효과적인 행동을 학습하고 탐색을 촉진하기 위해 관찰 엔트로피(Observation Entropy)를 최대화하는 기법입니다. 실험 결과, E2C는 기존의 일반적인 제약 없는 알고리즘 및 제약 있는 기준 대비 작업 성능에서 동등하거나 우수한 성과를 보이면서, 불안전한 행동을 최대 50%까지 감소시켰습니다.

- **Technical Details**: 본 연구에서 다루는 제약이 있는 다중 에이전트 강화 학습(MARL)은 연구에 나온 수학적 모델로 표현되며, 이는 에이전트들 간의 협력을 최적화합니다. 모든 에이전트는 결합된 행동을 통해 환경의 상태를 변화시키며, 고유한 보상에 따라 행동을 선택합니다. E2C는 카운트 기반 및 k-최근접 이웃(k-nearest neighbor) 근사를 활용하여 관찰 엔트로피를 추정하고, 이 과정에서 에이전트들이 높은 성과를 내기 위해 최적의 행동을 선택하도록 보상합니다.

- **Performance Highlights**: E2C 알고리즘은 복잡한 협력 작업에서 제약 조건을 성공적으로 만족시키며, 이전의 제약이 있는 MARL 기법들이 실패했던 고위험 환경에서도 높은 보상 행동을 학습합니다. 제안한 알고리즘은 여섯 가지 잘 알려진 MARL 도메인에서 실험되어, 그 성능이 기존의 제약 없는 알고리즘과 비교하여 뛰어남을 입증했습니다. 또한, 팀 제약을 도입함으로써 정책 향상에 대한 하한을 제공하고, 협력 에이전트의 성공적인 학습을 도모합니다.



### EmoReg: Directional Latent Vector Modeling for Emotional Intensity Regularization in Diffusion-based Voice Conversion (https://arxiv.org/abs/2412.20359)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이번 논문에서는 감정 목소리 변환(Emotional Voice Conversion, EVC) 프레임워크에서 감정 강도를 정규화하는 방법을 제안합니다. 기존 방식들이 감정 강도를 조절하는 데 있어 한계점을 보였던 반면, 새로운 방법론은 Self-supervised learning (SSL)에 기반한 특징 표현과 비지도 방향 잠재 벡터 모델링(Directional Latent Vector Modeling, DVM)을 사용하여 보다 정확한 감정 표현을 달성합니다. 이를 통해 특정 감정 및 강도로 변환된 고품질의 음성을 생성할 수 있습니다.

- **Technical Details**: EVC는 주어진 음성 발화에서 소스 감정 상태를 목표 감정으로 변환하는 것을 목표로 하며, 이 과정에서 언어적 내용을 보존합니다. 본 연구에서는 Diffusion 기반 EVC 프레임워크에서 Emotion embeddings를 조정하여 목표 감정 강도를 실현하는 방향으로 접근합니다. 기존의 이산 감정 표현에서 감정 강도 조절이 더 용이한 반면, 연속 감정 모델링은 고급 감정 데이터베이스의 필요성으로 인해 구현에 어려움이 있습니다.

- **Performance Highlights**: 제안된 EmoReg 방법의 효과는 영어 및 힌디어에 대한 주관적인 및 객관적인 평가에서 최신 스타일(state-of-the-art, SOTA) 기준과 비교하여 입증되었습니다. 본 연구는 EVC의 감정 강도 정규화를 위한 첫 번째 고품질 접근 방식을 제시하였으며, 이는 음성 합성 및 AI 기반 더빙의 복잡한 요구를 충족시키는 데 기여할 것으로 기대됩니다. 향후 이 연구는 다양한 응용 분야에서 감정 표현의 품질 향상에 기여할 수 있습니다.



### HindiLLM: Large Language Model for Hind (https://arxiv.org/abs/2412.20357)
- **What's New**: 이번 연구에서는 힌디어 및 다른 인도 언어를 위한 고성능 언어 모델인 HindiLLM-Small 및 HindiLLM-Medium을 사전 훈련했습니다. 두 단계의 과정으로 비지도 사전 훈련과 지도 미세 조정을 수행하여, 텍스트 분류, 감정 분석 등 다양한 작업에서 사용될 수 있는 모델을 구축했습니다. 이를 통해 더욱 효과적으로 인도 언어를 처리할 수 있는 기반을 마련했습니다.

- **Technical Details**: 이 연구에서는 힌디어 텍스트를 위한 큰 데이터세트를 생성하고 BPE(Byte-Pair Encoding) 알고리즘을 사용하여 토크나이저를 훈련했습니다. 이후 라벨 없이 제공된 데이터를 이용한 사전 훈련을 통해 힌디어 모델의 기초를 형성했습니다. 그리고 감정 분석, 텍스트 분류, 자연어 추론과 같은 다양한 작업에 맞춰 모델을 미세 조정했습니다.

- **Performance Highlights**: 평가 결과, HindiLLM 기반의 미세 조정 모델이 여러 언어 관련 작업에서 기존 모델들보다 성능이 우수함을 보여주었습니다. 특히, 다양한 인지적 태스크와 함께 주어진 라벨된 데이터셋에서 높은 정확도를 보였습니다. 이는 힌디어 및 다른 인도 언어에서 NLP의 발전을 이끄는 데 중요한 기여를 할 것으로 기대됩니다.



### Distilling Desired Comments for Enhanced Code Review with Large Language Models (https://arxiv.org/abs/2412.20340)
Comments:
          12 pages, 9 figures

- **What's New**: 이 논문에서는 코드 리뷰를 위해 맞춤형 데이터 세트를 통해 LLMs(대형 언어 모델)를 미세 조정할 수 있는 새로운 데이터 증류 방법인 Desiview를 제안합니다. 기존의 방법들이 데이터 세트의 품질과 효과성에서 한계를 보인 데 반해, Desiview는 고품질의 DRCs(Desired Review Comments)를 자동으로 식별하여 보다 효율적인 미세 조정을 가능하게 합니다. 이 방법은 코드 리뷰 데이터 세트에서 DRCs를 효과적으로 추출하여 새로운 모델 Desiview4FT 및 Desiview4FA를 구축합니다.

- **Technical Details**: Desiview는 코드 리뷰 데이터 세트에서 DRCs를 식별하여 미세 조정에 적합한 데이터 세트를 자동으로 구성합니다. 논문에서는 LLaMA 3 및 LLaMA 3.1 시리즈를 사용하여 Desiview4FT 모델을 구축하고, 이후 KTO 정렬을 통해 Desiview4FA 모델을 개발합니다. 이 과정에서 Desiview는 Precision, Recall, Accuracy와 F1 점수에서 각각 88.93%, 80.37%, 86.67%, 84.44%를 기록하며 최신 기술을 초월하는 성과를 보여주었습니다.

- **Performance Highlights**: Desiview4FT와 Desiview4FA 모델 모두 기존 LLM 모델보다 코드의 문제를 더 정확하게 식별하고 리뷰 코멘트를 더 잘 생성하여 DRCs의 품질을 향상시킵니다. 인간 평가 결과에 따르면, 두 모델은 더 정확한 문제 지적과 더 적절한 코멘트 생성을 보여주어, 코드 품질 보증 과정에서의 유용성을 강조합니다. 이러한 결과는 Desiview 방법이 LLM 기반 코드 리뷰 시스템의 성과를 크게 개선할 수 있음을 보여줍니다.



### Mind the Data Gap: Bridging LLMs to Enterprise Data Integration (https://arxiv.org/abs/2412.20331)
Comments:
          CIDR'25

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 성과가 공개 데이터에 기반한 현재 기준보다 실제 기업 데이터에서는 저하된다는 점을 강조합니다. 기존 기준은 웹에서 수집된 데이터를 기반으로 하고 있으며, 이러한 기준에서 높은 성과를 보이는 LLM의 실제 사용에서는 저조한 결과를 초래합니다. 저자들은 이를 해결하기 위해 새로운 벤치마크 데이터셋인 GOBY Benchmark를 출시하였고, 이를 통해 LLM의 성능을 기업 데이터에서도 높일 수 있는 방법을 제시합니다.

- **Technical Details**: GOBY Benchmark는 이벤트 프로모션과 마케팅 환경으로부터 유도된 데이터셋으로, 400만 개 이상의 이벤트를 포함합니다. 이 데이터셋은 전문 개발자들에 의해 작성된 워퍼를 활용하여 웹 페이지와 API를 관계형 테이블로 변환하여 구성되었습니다. 연구자들은 LLM 성능을 높이기 위해 계층적 주석(hierarchical annotation), 런타임 클래스 학습(runtime class-learning), 온톨로지 합성(ontology synthesis) 기술을 제안하며, 이러한 기술이 적용될 경우 성과가 공공 데이터와 대등해지는 것을 보였습니다.

- **Performance Highlights**: GOBY Benchmark의 성과를 분석한 결과, LLM 기반 접근 방식이 기존의 공공 데이터 기준보다 낮은 성과를 보였습니다. 저자들은 LLM이 실제 기업 데이터에서 잘 작동하도록 하기 위해 보다 세밀한 접근 방법이 필요하다고 주장합니다. 이러한 벤치마크의 필요성은 LLM 기반 기술들이 결과의 품질 문제로 인해 많은 기업 프로젝트에서 포기되고 있다는 점에서 더욱 부각됩니다.



### Protein Structure Prediction in the 3D HP Model Using Deep Reinforcement Learning (https://arxiv.org/abs/2412.20329)
Comments:
          15 pages, 9 figures

- **What's New**: 본 논문에서는 3D Hydrophobic-Polar (HP) 격자 모델에서 단백질 구조 예측을 위한 두 가지 새로운 딥러닝 아키텍처를 제시합니다. 36개 잔기를 가진 단백질의 경우, 고정된 랜덤 프로젝션과 학습 가능한 딥 레이어를 결합한 하이브리드 레저보어 모델이 최적의 형태를 달성하며, 학습 에피소드를 25% 줄이는 성과를 보였습니다. 더 긴 서열에는 다중 헤드 어텐션이 포함된 장기 단기 메모리 네트워크(LSTM)를 사용하여 최적의 에너지 값을 재현했습니다.

- **Technical Details**: 연구는 단백질 접힘 과정을 Markov Decision Process (MDP)로 모델링하고 레저보어 컴퓨팅 및 LSTM 네트워크로 이루어진 두 가지 아키텍처를 제안합니다. 단백질의 각 아미노산은 격자 구조의 점을 점유하며, 아미노산 간 거리 및 각도와 같은 다양한 제약이 부여됩니다. 이를 통해 H-H 접촉 수를 극대화하는 최적의 형태를 찾는 것이 목표가 됩니다.

- **Performance Highlights**: 제안된 아키텍처들은 기존 방법들에 비해 훈련 효율성을 지속적으로 향상시키면서 최적의 형태를 일관되게 달성하는 것으로 나타났습니다. 이 논문은 단백질 접힘 문제에 레저보어 컴퓨팅을 처음으로 적용하고 있으며, LSTM 아키텍처는 먼 거리의 아미노산 간 상호작용 모델링을 효과적으로 수행합니다. 실험 결과는 우리의 접근법이 최신 상태의 기법들과 비교할 때 월등한 성과를 보여준 것을 강조합니다.



### Hypergraph-Based Dynamic Graph Node Classification (https://arxiv.org/abs/2412.20321)
Comments:
          Accepted in ICASSP 2025

- **What's New**: 본 논문은 동적 그래프에서의 노드 분류를 위한 새로운 모델인 Hypergraph-Based Multi-granularity Dynamic Graph Node Classification (HYDG)을 제안합니다. 기존의 RNN과 self-attention 기반 방법들이 노드의 다양한 동적 변화를 충분히 잘 반영하지 못했던 문제를 해결하기 위해, 개별 및 집단 수준의 하이퍼그래프를 활용하여 노드의 특징을 효과적으로 모델링합니다. HYDG는 노드 간의 복잡한 상관관계를 다차원적으로 포착하며, 이전의 정적인 그래프 모델의 한계를 극복합니다.

- **Technical Details**: HYDG 모델은 기본 노드 표현을 생성하기 위해 GNN(backbone) 구조를 사용합니다. 그런 다음 하이퍼엣지를 통해 개별 노드와 집단 노드의 다양한 시간 범위에서의 관계를 모델링합니다. 이 과정에서 단순한 정보 전파(Information propagation)가 아니라, 가중치가 부여된 정보 전파를 통해 더 나은 노드 표현을 생성하여 다차원적인 종속성을 효과적으로 캡처합니다.

- **Performance Highlights**: 실험을 통해 HYDG는 다섯 개의 실제 동적 그래프 데이터셋에서 우수한 성능을 보이며, 기존의 기준 모델들보다 일관되게 더 나은 결과를 나타냅니다. 이로써 제안된 모델이 동적 그래프에서의 노드 분류 작업에 있어서 중요하고 효율적인 접근 방식을 제공함을 확인할 수 있습니다. 이 논문은 실제 응용 분야에서의 다양한 변화를 반영할 수 있는 가능성을 보여줍니다.



### EXAdam: The Power of Adaptive Cross-Moments (https://arxiv.org/abs/2412.20302)
- **What's New**: 이 논문에서는 EXAdam($\textbf{EX}$tended $\textbf{Adam}$)이라는 새로운 최적화 알고리즘을 소개합니다. EXAdam은 기존 Adam 옵티마이저를 기반으로 하여 세 가지 주요 개선점을 포함하고 있습니다: (1) 모멘트 추정을 개선하기 위한 새로운 디바이셔닝(debiasing) 항, (2) 현재 손실 경량에 대한 응답성을 높이는 그래디언트 기반의 가속 메커니즘, (3) 훈련 과정 동안 학습률의 지속적인 증가를 허용하는 동적 스텝 사이즈 공식입니다.

- **Technical Details**: EXAdam의 방법론에서, 소위 마법의 기법으로 불리는 모멘텀(momentum)은 여러 반복 최적화 알고리즘에서 중요한 역할을 하며, 기존의 Adam 옵티마이저의 한계를 극복하는 데 초점을 맞추고 있습니다. EXAdam에서는 전통적인 Adam의 관성을 유지하면서 새로운 디바이셔닝 모멘트 추정 m~~(m)과 v~~(v)를 도입하여, 두 번째 모멘트가 업데이트에 미치는 영향을 고려하여 최적화의 안정성을 높이도록 설계되었습니다. 이는 특히 고차원의 손실 경량을 다룰 때 유용하게 작용할 수 있습니다.

- **Performance Highlights**: 실험적으로, EXAdam은 CIFAR-10 데이터셋에서 CNN 모델을 훈련할 때 기존 Adam에 비해 48.07% 더 빠른 수렴 속도를 기록하고, 훈련 정확도는 4.6%, 검증 정확도는 4.13% 향상시켰습니다. 이러한 결과는 EXAdam의 효용성을 입증하는 것이며, 다양한 작업에 대해 추가적인 실험을 통해 그 효과를 더욱 평가할 필요가 있습니다. EXAdam은 머신러닝 응용 프로그램의 폭넓은 개선 가능성 덕분에 적응형 최적화 기법의 중요한 발전을 나타냅니다.



### Transformer-Based Contrastive Meta-Learning For Low-Resource Generalizable Activity Recognition (https://arxiv.org/abs/2412.20290)
- **What's New**: 본 논문에서는 사람 활동 인식(HAR) 분야의 일반화를 위한 새로운 접근법인 TACO를 제안합니다. TACO는 transformer 기반의 대조적 메타 학습(contrastive meta-learning) 기법으로, 다양한 사용자 및 시나리오 간의 분포 변화(distribution shifts, DS) 문제를 해결하는 데 중점을 둡니다. 특히, TACO는 훈련 중 가상의 목표 도메인을 합성하여 모델의 일반화 가능성을 명확하게 고려하여 DS를 완화합니다.

- **Technical Details**: 이 연구에서는 HAR의 성능 향상을 위해 transformer의 주의 메커니즘(attention mechanism)을 활용하고, 메타 최적화(meta-optimization) 과정에서 감독된 대조 손실(supervised contrastive loss)을 강화합니다. 데이터 다양성(expanding data diversity)은 세 가지 방법으로 달성되며, 여기에는 회전(rotation), 시간 왜곡(time warping), 난잡성(jittering) 등이 포함됩니다. 이를 통해 TACO는 제한된 훈련 샘플로도 효과적인 도메인 불변 및 클래스 판별 표현을 학습합니다.

- **Performance Highlights**: TACO는 여러 저자원 분포 이동 시나리오에서 평균 4.08%의 정확도 향상을 기록하며, 기존의 도메인 일반화 방법들보다 더 나은 성능을 보입니다. 이는 TACO가 HAR 알고리즘 중 처음으로 감독된 대조 학습(supervised contrastive learning)을 메타 학습에 통합한 것이며, 모델의 세분화된 표현 학습을 강화하게 됩니다. 이러한 성과는 사용자와 환경에 대한 높은 다양성을 반영한 데이터 공간 확장을 통해 얻어진 것입니다.



### How To Think About End-To-End Encryption and AI: Training, Processing, Disclosure, and Consen (https://arxiv.org/abs/2412.20231)
- **What's New**: 본 논문은 종단 간 암호화(End-to-End Encryption, E2EE) 시스템의 인공지능(AI) 모델 통합의 호환성 분석에 대한 심도 있는 연구입니다. E2EE는 데이터의 기밀성과 개인 정보 보호를 보장하는 데 중요한 역할을 하고 있지만, 최근 AI 모델의 확산은 이러한 보장을 흔들 수 있는 잠재적 위험을 안고 있습니다. 이 연구는 AI 보조 도구의 통합과 E2EE 데이터를 AI 모델 교육에 사용하는 두 가지 측면에서 보안 영향을 탐구합니다.

- **Technical Details**: 이 논문은 E2EE의 핵심 기밀성 및 무결성 속성을 식별하고, AI 기능 통합을 위한 다양한 기술 구성을 평가합니다. E2EE 콘텐츠를 AI 모델에 입력하기 위한 여러 접근 방식(추론이나 교육을 위한 경우 포함)의 효과성을 분석하고, 각 구성 방식이 E2EE의 주요 보장을 유지할 수 있는 능력을 평가합니다. 이 과정에서, E2EE 데이터가 AI 모델 훈련에 사용될 수 없음을 강조하며, 특정 요구 조건을 따를 경우에만 E2EE 콘텐츠 처리가 가능함을 보고합니다.

- **Performance Highlights**: AI 기능을 사용하기 위해 E2EE 콘텐츠를 처리하는 경우, 사용자 측에서 가능한 한 많은 처리를 우선시해야 하며, 제3자는 어떤 E2EE 콘텐츠도 보거나 사용할 수 없어야 합니다. 이 연구는 E2EE 보안 유지를 위한 기술적 설계 선택을 우선시해야 하며, 서비스 제공자가 E2EE 보안을 정확히 반영해야 할 필요성을 강조합니다. 최종적으로, AI 기능의 기본 작동 방식 및 사용자 동의를 요청하는 방법에 대한 모범 사례를 제시하며, AI의 배포와 E2EE 간의 긴장 관계에 대한 심도 있는 논의가 이어지기를 기대합니다.



### Leveraging Large Language Models for Enhancing Autonomous Vehicle Perception (https://arxiv.org/abs/2412.20230)
Comments:
          4 pages

- **What's New**: 이번 연구는 자율주행차(Autonomous Vehicles, AV)의 인식 시스템에 대형 언어 모델(Large Language Models, LLMs)을 통합하는 혁신적인 접근 방식을 제시합니다. 이 통합은 동적 환경에서의 과제 해결, 센서 융합(sensor fusion) 및 맥락적 추론(contextual reasoning)을 용이하게 하여 자율주행 기술의 정확성 및 신뢰성을 크게 향상시킵니다. 실험 결과는 LLMs가 AV의 인식 시스템을 안전하고 스마트하게 만들어줄 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 세 가지 핵심 구성 요소로 이루어진 시스템을 제안합니다. 첫 번째는 센서 데이터 처리 모듈로, 카메라 및 LiDAR와 같은 다양한 센서의 입력을 처리합니다. 두 번째 구성 요소인 LLM 통합 계층은 자연어 형식으로 데이터를 변환하여 고차원의 판단을 가능하게 합니다. 마지막으로 의사 결정 지원 모듈이 LLM의 출력을 AV의 제어 시스템에 전달하여 보다 지능적인 의사 결정을 지원합니다.

- **Performance Highlights**: 실험은 KITTI 및 nuScenes와 같은 데이터셋을 활용하여 AV 인식 시스템의 성능을 평가하였습니다. LLM 통합의 영향을 검토하기 위해 기존 AV 시스템과 비교하며, 객체 인식 정확도, 의사 결정 반응 시간 및 맥락 이해 점수를 주요 성과 지표로 설정했습니다. 이 연구의 결과는 자율주행차 인식 시스템의 신뢰성과 적응력을 크게 향상시키는 LLM의 잠재력을 입증합니다.



### Decoding Emotion: Speech Perception Patterns in Individuals with Self-reported Depression (https://arxiv.org/abs/2412.20213)
- **What's New**: 이번 연구는 인도 인구 내에서 자가 보고된 우울증과 감정적 음성 인식 간의 관계를 조사한 것으로, PANAS와 PHQ-9를 이용하여 기분과 우울증을 평가하였습니다. 연구는 감정적으로 자극된 음성에 대한 개인들의 감정 반응을 분석하며, 특히 우울증 상태가 이 인식에 미치는 영향을 다루고 있습니다. 이전 연구 결과와는 달리, 우울증 집단에서 긍정적인 감정 반응 감소가 관찰되지 않았다는 점이 주목할 만합니다.

- **Technical Details**: 본 연구에서는 감정적 음성 자극에 대한 감정 반응을 분석하기 위해 대학생 97명을 대상으로 했습니다. 음성 자극은 IEMOCAP 데이터베이스에서 취득하였으며, 네 가지 감정 범주(슬픔, 행복, 분노, 중립)에 따라 세트가 구성되었습니다. 음성 자극은 조작적으로 분류된 라벨과 차원적 주석을 가지고 있어 감정 인식 측정의 신뢰성을 높였습니다.

- **Performance Highlights**: 연구 결과, 우울증 집단은 중립 감정 음성을 제외하고는 어떤 정서 자극에서도 뚜렷한 차이를 보이지 않았습니다. 이는 당초의 예상과 달리 우울증 상태에서 긍정적 감정 반응이 감소하지 않았음을 보여줍니다. 슬픔이나 분노를 묘사하는 음성 자극에서의 감정 반응은 모든 감정 인식 측정 방법에서 일관성을 나타내며, 이는 음성 자극에 대한 감정적 반응의 복잡성을 시사합니다.



### Building a Rich Dataset to Empower the Persian Question Answering Systems (https://arxiv.org/abs/2412.20212)
- **What's New**: 이번 연구에서는 페르시아어를 위한 포괄적인 오픈 도메인 데이터셋인 NextQuAD를 제시합니다. 이 데이터셋은 7,515개의 문맥과 23,918개의 질문 및 답변으로 구성되어 있습니다. 기존의 모델들이 영어에 집중된 반면, 이 연구는 자원이 부족한 언어에 대한 문제 해결을 위한 기초 자료로 활용될 것입니다.

- **Technical Details**: NextQuAD 데이터셋은 BERT 기반의 질문 응답 모델로 활용되며, ParsBERT와 XLM-RoBERTa 두 가지 사전 훈련된 언어 모델이 사용됩니다. 이 두 모델의 결과는 평균 로그 확률(mean logits)로 앙상블(ensemble)됩니다. 개발 세트에서의 평가는 0.95의 Exact Match (EM) 및 0.97의 Fl_score를 나타냅니다.

- **Performance Highlights**: NextQuAD로 훈련된 모델은 PersianQA 및 ParSQuAD의 두 데이터셋과 비교되었습니다. 결과적으로 PersianQA에서는 EM이 0.39, ParSQuAD-manual에서는 0.14 증가한 반면, ParSQuAD-automatic에서는 0.007의 소폭 감소가 발생했습니다. 이러한 성과는 NextQuAD가 페르시아어 질문 응답 시스템의 성능을 개선할 수 있음을 보여줍니다.



### Towards Real-Time 2D Mapping: Harnessing Drones, AI, and Computer Vision for Advanced Insights (https://arxiv.org/abs/2412.20210)
Comments:
          7 pages, 7 figures, 1 table

- **What's New**: 이 프로젝트는 드론 이미지를 머신 러닝(machie learning) 및 컴퓨터 비전(computer vision)과 통합하여 실시간 2D 매핑(real-time 2D mapping) 시스템을 소개합니다. 이 시스템은 서베일런스(surveillance), 정찰(reconnaissance), 목표 추적(target tracking)과 같은 군사 작전을 위한 정확하고 시기적절한 지리적 데이터 제공을 목적으로 합니다. 자동화된 기능 탐지(feature detection), 이미지 매칭(image matching) 및 스티칭(stitching)을 통해 고해상도 지도를 신속하게 생성합니다.

- **Technical Details**: 시스템은 Python으로 구현되었으며, OpenCV를 활용하여 이미지 처리를 수행하고, NumPy를 이용하여 효율적인 계산을 지원합니다. ORB(Oriented FAST and Rotated BRIEF)를 사용하여 기능을 탐지하고, FLANN(Fast Library for Approximate Nearest Neighbors)를 통해 정확한 키포인트 매칭(keypoint matching)을 보장합니다. 동차 변환(homography transformations)을 통해 겹치는 이미지를 정렬하여 실시간으로 왜곡 없는 지도를 생성합니다.

- **Performance Highlights**: 테스트 결과, 이 시스템은 전통적인 방법에 비해 속도와 정확성에서 상당한 개선을 이루었습니다. 다양한 조명 조건 및 거친 지형에서도 뛰어난 성능을 보이며, 이는 우주항공(aerospace) 및 방위(defense) 시나리오에서 매우 효과적입니다. 이 자동화된 접근법은 수동 개입을 제거하여 동적인 환경에서 실시간 업데이트를 가능하게 하여 상황 인식(situational awareness) 및 의사결정(decision-making)을 향상시킵니다.



### Injecting Explainability and Lightweight Design into Weakly Supervised Video Anomaly Detection Systems (https://arxiv.org/abs/2412.20201)
Comments:
          IEEE TETC-CS (Under review)

- **What's New**: 본 논문은 TCVADS(Two-stage Cross-modal Video Anomaly Detection System)라는 새로운 시스템을 제안합니다. 이 시스템은 weak supervision learning을 활용하여 스마트 시티 모니터링에서 발생하는 이상 현상을 효과적으로 탐지합니다. 특히, 기존의 복잡한 멀티모달 접근 방식의 한계를 극복하고 실시간성과 해석 가능성을 동시에 만족하도록 설계되었습니다.

- **Technical Details**: TCVADS는 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 비디오 프레임에서 특징을 추출한 후, 이를 시간 시계열 분석 모듈에 입력하여 teacher model로 작동시킵니다. 이후, knowledge distillation을 통해 얻어진 통찰력을 기반으로 단순화된 convolutional network(student model)이 이진 분류를 수행합니다.

- **Performance Highlights**: 실험 결과, TCVADS는 기존 방법들보다 모델 성능, 탐지 효율성 및 해석 가능성에서 상당한 개선을 보였습니다. 마지막 단계에서는 CLIP을 활용한 cross-modal contrastive learning을 통해 텍스트와 이미지 간의 관계를 강화하며, 더욱 정교한 분류 결과를 도출합니다. 이러한 결과는 스마트 시티 모니터링 응용 프로그램에 매우 중요한 기여를 합니다.



### Federated Unlearning with Gradient Descent and Conflict Mitigation (https://arxiv.org/abs/2412.20200)
Comments:
          To be published in the Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이번 연구에서는 Federated Unlearning (FU) 방식에 있어 주목할 만한 새로운 접근인 Federated Unlearning with Orthogonal Steepest Descent (FedOSD)를 제안합니다. FedOSD는 기존 방법에서 발생할 수 있는 gradient conflicts를 피하고, 모델 유틸리티 감소를 최소화하는 방법입니다. 또한, unlearning 후 모델의 성능 회복을 위한 gradient projection 전략을 도입하여, 학습 완료 후에도 데이터 삭제의 성과를 유지합니다.

- **Technical Details**: FedOSD는 효율적인 unlearning을 위한 새로운 Cross-Entropy 손실을 설계하여 gradient ascent의 수렴 문제를 해결합니다. 비활성 클라이언트의 gradient와 충돌하지 않도록 계산된 orthogonal steepest descent 방향을 활용하여, 특정 클라이언트의 데이터를 효율적으로 삭제할 수 있습니다. 이 과정은 모델 유틸리티 감소를 최소화하고, unlearning 성과를 유지하는데 중점을 두고 진행됩니다.

- **Performance Highlights**: 다양한 FL 시나리오에서 실시된 실험 결과, FedOSD는 기존의 SOTA FU 알고리즘과 비교하여 우수한 unlearning 성과와 뛰어난 모델 유틸리티를 달성했습니다. 모델의 유틸리티 감소를 방지하면서도 효율적인 데이터 삭제가 가능함을 보여줍니다. 따라서, FedOSD는 데이터 프라이버시 및 보안을 강화하는 데 기여할 수 있는 방안으로 주목받고 있습니다.



### Lower bounds on transformers with infinite precision (https://arxiv.org/abs/2412.20195)
- **What's New**: 이번 논문에서는 무한 정밀도의 1-layer softmax transformers에 대한 하한(lower bound)을 VC 차원 기법을 통해 증명하였습니다. 하한은 특정 작업에 대해 특정 수의 매개변수로 이 작업을 수행할 수 없는 경우를 나타냅니다. 이를 통해 1-layer softmax transformers가 특정 작업에서 낮은 성능을 보이는 이론적 이유를 설명하고 있습니다.

- **Technical Details**: 연구자들은 이전 작업에서 통신 복잡도(communication complexity)를 사용하여 1-layer softmax transformers의 하한을 개발하였습니다. 본 논문에서는 정밀도가 아닌 출력 MLP의 크기(output MLP size)에 대한 가정을 세워 무한 정밀도의 transformer에 대한 하한을 도출합니다. 또한, 함수 조합(function composition) 및 SUM2 작업을 포함하여 두 가지 특정 작업에 대한 하한을 제시합니다.

- **Performance Highlights**: 이 연구에서 제안된 하한은 각 작업에 대해 embedding dimension 또는 출력 MLP의 크기가 n^{
Ω(1)} 이상이어야 함을 보여줍니다. 특히, palindrome 인식 작업(palindrome recognition task)은 무한 정밀도를 가정할 때 일정한 embedding dimension과 출력 MLP 크기로 해결될 수 있음을 보여줍니다. 이 연구는 1-layer softmax transformers의 성능 한계를 명확히 하고, 향후 연구 방향에 대한 기초 자료를 제공합니다.



### Imitation Learning from Suboptimal Demonstrations via Meta-Learning An Action Ranker (https://arxiv.org/abs/2412.20193)
- **What's New**: 본 논문에서는 전문가의 수많은 시연이 필요하다는 제약을 해결하기 위해 ILMAR(아크션 랭커를 통한 메타 학습에 의한 모방 학습)이라는 새로운 접근 방식을 제안합니다. ILMAR은 제한된 수의 전문가 시연과 보조 시연을 활용하여 가중치 행동 클로닝(weighted behavior cloning)을 적용합니다. 이 모델은 보조 시연으로부터의 지식을 선택적으로 통합하기 위해 이점 함수(advantage function)를 활용합니다.

- **Technical Details**: ILMAR은 새로운 메타 목표(meta-goal)를 도입하여 현재 정책과 전문가 정책 간의 거리를 최소화하여 성능을 높입니다. 이를 통해, 최적 시연(suboptimal demonstrations)으로부터의 학습을 효율적으로 수행하며, 비전문가 시연에서의 높은 품질을 보장합니다. ILMAR 접근 방식은 부정확한 시연을 학습할 때 발생하는 오류 문제를 경감시키고, 훈련 중에 고품질 비전문가 시연을 유지하도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 ILMAR은 기존의 최첨단 모방 학습 알고리즘에 비해 우수한 성능을 발휘하는 것이 입증되었습니다. 성능은 각기 다른 상황에서도 전문가 및 추가 시연을 포함한 다양한 작업에서 경쟁력이 있으며, 특히 전문가 시연이 충분하지 않을 때 그 장점이 더욱 두드러집니다. ILMAR은 보조 데이터셋에서의 개선된 학습을 통해 더 나은 정책을 생성하는 데 성공했습니다.



### Real-time Calibration Model for Low-cost Sensor in Fine-grained Time series (https://arxiv.org/abs/2412.20170)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 저비용 센서의 오차를 줄이기 위한 새로운 보정 모델, TESLA(Transformer for effective sensor calibration utilizing logarithmic-binned attention)를 제안합니다. TESLA는 고성능 딥러닝 모델과 로그 바인딩을 도입하여 비선형 컴포넌트를 캘리브레이션하고 감지합니다. 이 방법은 하드웨어 제약이 있는 시스템에서도 일관되게 실시간 캘리브레이션을 수행할 수 있게 해줍니다.

- **Technical Details**: TESLA는 세 가지 주요 요구 사항을 충족시키며, 이는 정밀한 센서 유형 처리, 실시간 캘리브레이션의 일관성 확보, 하드웨어 제약 수용이 포함됩니다. 또한, TESLA는 로그 바인닝을 사용하여 주목 복잡성을 최소화하고, 멀티 뷰 임베딩과 특징별 집계를 통해 지역 및 글로벌 시계열 패턴을 보존합니다. 이 모델은 기존 모델들과 비교했을 때 선형 모델과 유사한 속도로 작동하면서도 더 높은 정확도를 제공합니다.

- **Performance Highlights**: 실험 결과 TESLA는 정확성, 캘리브레이션 속도 및 에너지 효율성에서 기존의 딥러닝 및 선형 모델을 능가하는 성능을 보였습니다. TESLA는 저비용 센서에서 고품질 센서와 유사한 정확도를 달성하며, 세 가지 요구 사항을 동시에 해결합니다. 이러한 결과는 TESLA가 실제 IoT 시스템에서도 충분히 적용 가능하다는 것을 시사합니다.



### LoL-PIM: Long-Context LLM Decoding with Scalable DRAM-PIM System (https://arxiv.org/abs/2412.20166)
Comments:
          15 pages, 12 figures

- **What's New**: 이 논문에서는 LoL-PIM이라는 하드웨어-소프트웨어 공조 설계를 통해 긴 컨텍스트를 가진 대형 언어 모델(LLM)을 가속화하는 새로운 다중 노드 PIM 아키텍처를 제안합니다. 기존의 PIM 시스템의 한계인 고정적인 메모리 관리 방식 대신, LoL-PIM은 동적 메모리 관리(DMA)를 도입하여 다양한 KV-cache 크기를 지원하고, 더 높은 처리량을 실현할 수 있도록 합니다. 이러한 접근은 메모리 용량과 대역폭의 문제를 해결하면서 더욱 다양한 컨텍스트 길이를 유연하게 처리할 수 있게 합니다.

- **Technical Details**: LoL-PIM 시스템은 여러 PIM 모듈로 구성된 노드를 포함하여 긴 컨텍스트 LLM의 가속화를 위한 충분한 메모리 용량과 대역폭을 제공합니다. 이 시스템은 KV-cache와 가중치 행렬에 대한 PIM 인식 분할을 통해 PIM 모듈 내에서의 시퀀스 병렬성을 가능하게 하여 대역폭 이용률을 극대화합니다. 또한, LoL-PIM은 ‘ping-pong’ I/O 버퍼링을 도입하여 PIM 계산과 I/O 데이터 이동의 겹침을 통해 새로운 병목 현상을 완화하고 이를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: LoL-PIM은 7B에서 72B 파라미터를 가진 인기 있는 긴 컨텍스트 LLM에서의 스케일러블 디코딩 성능을 평가하였고, 최대 32K까지의 토큰 길이를 지원하는 다양한 긴 컨텍스트 작업을 통해 성능을 검증하였습니다. 결과적으로, LoL-PIM은 기존 GPU 및 상용 AiMX 시스템에 비해 각각 8.54배 및 4.74배의 토큰 생성 처리량을 달성하며, 실제 애플리케이션에서 LLM의 더 효율적인 배치를 가능하게 하는 잠재력을 보여줍니다.



### StyleAutoEncoder for manipulating image attributes using pre-trained StyleGAN (https://arxiv.org/abs/2412.20164)
- **What's New**: 이 논문에서는 이미지의 속성을 조작하는 데 사용되는 새로운 경량 AutoEncoder 모듈인 StyleAutoEncoder(StyleAE)를 소개합니다. StyleAE는 사전 훈련된 생성 모델에 플러그인으로 작용하며, 제한된 계산 자원으로도 깊은 생성 모델을 효율적으로 훈련할 수 있는 비용 효과적인 솔루션을 제공합니다. StyleGAN과 결합하여 실험한 결과, StyleAE는 최신의 알고리즘과 동일한 수준의 성능을 보여주면서도 보다 간단하고 빠르며 유연한 설계가 가능합니다.

- **Technical Details**: 논문에서는 StyleGAN과 AutoEncoder 아키텍처를 토대로 하는 StyleAE 접근법을 설명합니다. StyleGAN은 고품질 이미지를 생성할 수 있는 최첨단 생성 모델로 두 가지 주요 요소로 구성되어 있습니다. StyleAE는 이러한 구조를 통해 StyleGAN의 잠재 공간을 분리하여 속성을 조절할 수 있는 가능성을 제시하며, AutoEncoder를 활용하여 머지 수 있는 속성을 간편하게 변형할 수 있게 합니다.

- **Performance Highlights**: 실험을 통해 StyleAE가 인간 및 동물의 얼굴 이미지를 포함한 데이터셋에서 기존의 플로우 기반 모델과 동등하거나 더 나은 성능을 나타내며, 속성 조작에 있어 우수한 결과를 보여주었습니다. 또한, StyleAE는 계산적으로 효율적이며 대량의 훈련 데이터를 요구하지 않아서 다양한 응용 프로그램에서 실용적인 사용이 기대됩니다. 이러한 결과는 StyleGAN 및 다른 생성 모델의 잠재 공간 조작 효과를 향상할 수 있는 가능성을 보여줍니다.



### Topic-Aware Knowledge Graph with Large Language Models for Interoperability in Recommender Systems (https://arxiv.org/abs/2412.20163)
Comments:
          Accepted by The 40th ACM/SIGAPP Symposium On Applied Computing(SAC) 2025

- **What's New**: 이 논문에서는 지식 그래프를 활용한 추천 시스템에서 LLMs(대형 언어 모델)를 통해 일반 및 특정 주제를 추출하는 새로운 접근 방식이 제안됩니다. 이 방법은 사용자와 아이템 간의 데이터 희소성 및 콜드 스타트 문제를 해결하는데 도움을 주며, 다양한 추천 시스템 간의 일관성을 향상시킵니다. 특히, LLMs를 활용하여 기존의 불일치 문제를 해결하고, 사용하는 맥락 정보에서 보다 구체적인 아이템 특성을 포착합니다.

- **Technical Details**: 논문에서는 첫째, 아이템의 사이드 정보에서 일반 주제를 반복적으로 추출하여 업데이트하고, 둘째, 맥락 정보를 활용하여 특정 주제를 추출합니다. 그 후, 특정 주제 추출 과정에서 발생하는 동의어 문제를 해결하기 위해 정제 알고리즘이 적용됩니다. 이 과정을 통해 일관성을 유지하며, 아이템의 세부 특성과 사용자 선호를 더욱 깊이 이해할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과는 다양한 지식 그래프를 통해 추천 성능이 크게 향상됨을 보여줍니다. 이 연구는 사이드 및 맥락 정보에서 추출한 일반 및 특정 주제를 기반으로 추천 품질을 개선했다는 점에서 의미가 있습니다. 또한, 제안된 방법은 표준화된 메타그래프를 바탕으로 하여, 다양한 도메인 간 일관되고 상호 운용 가능한 접근 방식을 보장하고 있습니다.



### Stable-TTS: Stable Speaker-Adaptive Text-to-Speech Synthesis via Prosody Prompting (https://arxiv.org/abs/2412.20155)
Comments:
          Accepted by ICASSP 2025

- **What's New**: Stable-TTS는 고품질의 사전 샘플을 활용하여 적은 양의 소음이 있는 목표 음성 샘플에서도 일관된 프로소디를 보장하는 새로운 화자 적응형 TTS 프레임워크입니다. 이 모델은 프로소디 인코더와 Prosoody Language Model(PLM)을 통합하여 프로소디 생성의 일관성을 유지하고, 과적합을 방지하기 위해 prior-preservation loss를 적용합니다. 또한, 연결된 여러 사전 샘플을 사용하여 목표 화자의 음색을 효과적으로 포착합니다.

- **Technical Details**: Stable-TTS는 다섯 가지 핵심 모듈로 구성되며, 텍스트 인코더 Tᵗ, 프로소디 인코더 Pᵖ, 분산 조정기 Vᵛ, 음색 인코더 Sₛ, 확산 모델로 이루어져 있습니다. 이 모델은 멜-스펙트로그램 형태의 음성과 그에 상응하는 텍스트로 구성된 훈련 데이터셋을 사용하여 훈련됩니다. 특히, 프로소디 인코더는 프로소디 표현을 고정된 이산 프로소디 코드 집합으로 변환하는 벡터 양자화 계층을 포함하고 있습니다.

- **Performance Highlights**: 체계적인 실험을 통해 Stable-TTS가 청취 가능성(intelligibility), 자연스러움(naturalness), 화자 유사성(speaker similarity) 면에서 우수한 성능을 발휘함을 입증했습니다. 특히, 소음이 포함된 제한적인 샘플에서도 효과적인 고품질 음성을 생성하여 기존 모델들이 직면했던 문제점을 극복했습니다. 이 연구는 개인화된 음성 인식 서비스와 같은 다양한 애플리케이션에 기여할 것으로 기대됩니다.



### TradingAgents: Multi-Agents LLM Financial Trading Framework (https://arxiv.org/abs/2412.20138)
Comments:
          Multi-Agent AI in the Real World, AAAI 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 활용한 다중 에이전트 시스템을 통해 실제 거래 회사의 협업 역학을 모사하는 새로운 주식 거래 프레임워크인 TradingAgents를 제안합니다. 이 프레임워크는 기본적인 분석가, 감정 분석가, 기술 분석가 등 다양한 역할을 가진 에이전트를 포함하여 트레이더들이 역사적 데이터를 기반으로 통찰력을 종합하고 정보에 기반한 결정을 내리도록 지원합니다. 연구는 과거 금융 데이터를 통해 시스템의 성능을 평가하고 기타 기준 모델과 비교하여 개선된 누적 수익률과 샤프 비율을 달성했음을 보여줍니다.

- **Technical Details**: TradingAgents 프레임워크는 전문 트레이딩 팀의 의사 결정 프로세스를 모사합니다. 이 시스템은 기본적인 요소와 기술적 지표를 평가하는 Bull 및 Bear 에이전트를 포함하여, 유기적인 의사 소통을 통한 협업을 강화하기 위해 조직 구조에서 영감을 얻었습니다. 구조화된 출력을 포함한 하이브리드 접근 방식을 통해 에이전트 간의 논의 및 협업을 용이하게 하고, 복잡한 의사 결정 과정을 효율적으로 수립합니다.

- **Performance Highlights**: 실험 결과, TradingAgents는 누적 수익률, 샤프 비율 및 최대 낙폭 지표에서 기준 모델 대비 탁월한 성과를 보였습니다. 이러한 성과는 멀티 에이전트 LLM 프레임워크가 금융 거래의 동적이고 협업적인 환경에서 소비자 수익률을 향상시킬 수 있음을 입증합니다. 이러한 연구 결과는 금융 분야에 있어 LLM을 통한 다중 에이전트 시스템의 가능성을 강조합니다.



### M-MAD: Multidimensional Multi-Agent Debate Framework for Fine-grained Machine Translation Evaluation (https://arxiv.org/abs/2412.20127)
Comments:
          Work in progress. Code and data are available at this https URL

- **What's New**: 이 논문에서는 Multidimensional Multi-Agent Debate (M-MAD)라는 새로운 프레임워크를 제안하며, 이를 통해 기계 번역(MT) 평가에서 LLM(as-a-judge) 접근법을 개선합니다. M-MAD는 평가 기준을 분리하고 여러 에이전트 간의 논쟁을 통해 LLM의 추론 능력을 극대화합니다. 이 새로운 접근법은 기존 방법들보다 더 강력한 성능을 보이며, 다차원 평가로 신뢰성 있는 결과를 제공합니다.

- **Technical Details**: M-MAD는 MQM(다차원 품질 메트릭) 기준을 독립된 평가 차원으로 나누고 각 차원 내에서 다중 에이전트 간의 논쟁을 수행합니다. 이러한 과정은 LLM의 지식과 협동 능력을 활용하여, 최종적으로 토론 결과를 종합하여 종합적인 평가 판단을 생성합니다. 이 접근법은 세 가지 주요 단계로 구성되며, 이는 평가 차원 분할, 다중 에이전트 논쟁, 최종 판단을 포함합니다.

- **Performance Highlights**: M-MAD는 기존 LLM-as-a-judge 방법들과 비교하여 모든 시스템 수준에서 우수한 성능을 발휘하며, GPT-4o mini와 같은 하위 최적 모델로도 최신 참조 기반 자동 메트릭과 유사한 성과를 달성합니다. 종합 실험 결과 M-MAD는 세그먼트 수준 성능에서도 개선된 결과를 보이며, 기존 메트릭들과의 비교에서 높은 일치도를 나타냅니다. 이러한 성과는 LLM-as-a-judge 접근법의 변혁적 잠재력을 보여줍니다.



### SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis (https://arxiv.org/abs/2412.20104)
- **What's New**: 이 논문에서는 VR/AR 및 인간 애니메이션에서 중요한 문제인 현실적인 인간-객체 상호작용 모션 합성을 다룹니다. 특히, 여러 인간과 객체가 포함된 다체 구조에서의 모션 동기화를 위한 새로운 방법인 SyncDiff를 제안합니다. 기존 작업들이 제한된 수의 신체 구성에 초점을 둔 반면, SyncDiff는 여러 몸체의 복잡한 상호작용을 처리할 수 있는 일반적인 접근 방식을 제공합니다.

- **Technical Details**: SyncDiff는 단일 확산 모델을 사용해 다체 모션의 결합 분포를 캡처하며, 주파수 도메인 모션 분해 기법을 통해 모션의 정밀도를 높입니다. 이 모델은 데이터 샘플 점수와 정렬 점수를 동시에 추정하며, 시각적 모델 내에서 신체 모션 간의 상호작용을 그래픽적으로 모델링합니다. 이로 인해, 모든 신체의 개인적 모션과 상대적 모션을 포함하는 다차원 표현에서 모션 동기화를 명확하게 처리할 수 있습니다.

- **Performance Highlights**: SyncDiff는 네 가지 데이터 세트에서 내려받은 여러 인체 동작 구성에 대한 실험을 통해 기존의 최첨단 모션 합성 방법들보다 우수한 성능을 입증했습니다. 다양한 설정에서 모션의 조정과 세부적인 동작 충실도를 지속적으로 향상시키며, 다체 상호작용의 정밀한 조화를 가능하게 합니다. 이 연구는 향후 VR/AR 및 로봇 학습 분야에서 방대한 응용 가능성을 제공합니다.



### RFPPO: Motion Dynamic RRT based Fluid Field - PPO for Dynamic TF/TA Routing Planning (https://arxiv.org/abs/2412.20098)
Comments:
          2024 IEEE Intelligent Vehicles Symposium

- **What's New**: 이 논문에서는 기존의 로컬 다이나믹 경로 계획(algo) 알고리즘의 한계를 극복하기 위해 새로운 방법론을 제안합니다. 이 방법론은 대형 및 중형 고정익 항공기의 동적 장애물 회피를 동시에 충족할 수 있도록 설계되었습니다. Motion Dynamic RRT 기반 Fluid Field 및 PPO를 통한 동적 경로 계획을 통해 실시간 성능과 장거리 계획을 실현할 수 있습니다.

- **Technical Details**: 연구에서는 인공 포텐셜 필드(Artificial Potential Field)와 방해 유동( disturbance flow fields) 알고리즘을 통해 근접 정책 경량(proximal policy gradient) 알고리즘의 동작 및 상태 공간을 재설계하였습니다. 이러한 재설계는 항공기 동역학 모델을 기반으로 하여 상태 전이 과정을 구축하고, 장애물 회피 및 안전 비행 전략을 유도하는 보상 함수(reward function)를 설계합니다.

- **Performance Highlights**: 실제 DEM 데이터에서의 실험 결과, 제안된 알고리즘은 동적 제약 조건을 준수하는 충돌 없는 궤적 계획을 통해 장거리 비행 과제를 성공적으로 완료할 수 있음을 보여줍니다. 이러한 접근은 사전 글로벌 계획 없이도 동적 장애물 회피 및 지형 따르기 기능을 효율적으로 수행하도록 도와줍니다.



### From Worms to Mice: Homeostasis Maybe All You Need (https://arxiv.org/abs/2412.20090)
Comments:
          11 pages, 6 figures

- **What's New**: 이 논문은 기계 학습의 신경망에서 영감을 받아 단순한 XOR 모티프를 제안합니다. 이 모티프는 흥분성(excitatory) 및 억제성(inhibitory) 연결을 포함하며 생물의 신경 회로에서 가소성(plasticity)의 한 형태로 작용할 수 있음을 주장합니다. XOR 모티프는 입력 신호와 참조 신호 간의 차이를 신호화하여 학습 손실 함수의 기초를 제공하며, 신호의 전파를 멈춤으로써 항상성을 유지합니다.

- **Technical Details**: 핵심 모티프는 흥분성 신경세포와 억제성 신경세포의 비율이 4:1입니다. 이 연구는 C. elegans, 여러 Drosophila 신경망, 그리고 쥐의 V1 시각 피질에서 XOR 모티프의 존재를 조사했습니다. 연구팀은 이 모티프를 특수한 신경세포 연결을 요구하며 확인했습니다.

- **Performance Highlights**: XOR 모티프는 C. elegans의 접속망에서 722개의 엄격한 XOR 모티프와 134개의 '진짜' XOR 모티프를 발견했습니다. Drosophila와 쥐의 V1 시각 피질에서도 유사한 모티프가 확인되어 흥미로운 결과를 도출했습니다. 이 연구는 생물학적 신경 가소성(biological neural plasticity)의 근본적인 규제 메커니즘을 제안하며, 기계 학습 모델과의 유사성을 강조합니다.



### An archaeological Catalog Collection Method Based on Large Vision-Language Models (https://arxiv.org/abs/2412.20088)
Comments:
          4 pages,4 figures,www source track

- **What's New**: 이번 논문은 고고학적 카탈로그 수집을 위한 새로운 방법을 제안합니다. 특히 Large Vision-Language Models(VLMs)를 기반으로 한 접근 방식을 통해 기존의 기술적 문제를 해결하고자 합니다. 제안된 방법은 문서 위치 파악, 블록 이해 및 블록 일치화의 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: 제안된 방법에서는 첫째, 오픈 세트 객체 탐지 모델을 사용하여 입력 카탈로그 내에서 문서 블록을 위치 파악합니다. 둘째, 이 블록들을 처리하여 특성(attribute)으로 설명합니다. 마지막으로, 외래 키(linkage) 및 이분 그래프 매칭을 기반으로 일치 규칙을 구현하여 각 모달의 데이터를 완전하게 일치시킵니다.

- **Performance Highlights**: 다바구(Dabagou) 및 미아오지구(Miaozigou) 도자기 카탈로그를 사용한 실험을 통해 제안된 방법의 유효성을 입증하였습니다. 비교 실험 결과는 기존 VLM 기반 데이터 수집 방법의 한계를 극복할 수 있는 신뢰할 수 있는 솔루션으로 제시됩니다.



### On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs (https://arxiv.org/abs/2412.20087)
Comments:
          101 pages, 3 figures

- **What's New**: 이번 연구는 Common Vulnerability Scoring System (CVSS)과 같은 기존의 취약성 지표들이 Large Language Models (LLMs)에 대한 Adversarial Attacks (AAs) 평가에 얼마나 효과적인지를 조사합니다. 특히, 전반적인 및 특정 지표 요소들이 취약성 점수를 결정하는 데 미치는 영향을 탐구하며, 이러한 지표의 향상을 위한 새로운 관점을 제공합니다. 연구 결과는 기존 지표들이 다양한 공격에 대해 극히 최소한의 변동성을 보인다는 것을 보여주어, AAs 평가에 있어 더 유연하고 일반화된 지표의 필요성을 강조하고 있습니다.

- **Technical Details**: 이 연구는 56개의 적대적 공격에 대한 취약성 점수의 변동 계수를 계산하고 비교하는 정량적 접근 방식을 채택했습니다. 공격은 다양한 연구 논문에서 수집하였으며, 온라인 데이터베이스를 통해 획득되었습니다. 세 가지 개별 LLM을 통해 평가된 값들을 평균하여 점수를 결정했습니다.

- **Performance Highlights**: 결과는 기존의 점수 시스템이 서로 다른 공격에 대해 엄청난 변동성을 보이지 않음을 나타내며, 이는 많은 지표 요소들이 LLM에 대한 적대적 공격을 평가하는 데 적합하지 않다는 것을 시사합니다. 이 연구는 또한 현재의 취약성 지표들이 특히 고정된 값들을 사용하는 경우에 LLM의 AAs를 평가하는 데 한계가 있음을 강조하며, 더욱 발전된 평가 프레임워크 개발을 위한 새로운 가능성을 열어줍니다.



### MAFT: Efficient Model-Agnostic Fairness Testing for Deep Neural Networks via Zero-Order Gradient Search (https://arxiv.org/abs/2412.20086)
Comments:
          Accepted by ICSE24

- **What's New**: 이 논문에서는 기존의 백색상자(white-box) 방법과 비교하여 더욱 효과적이고 효율적인 새로운 블랙박스(black-box) 개별 공정성 테스트 방법인 MAFT(Model-Agnostic Fairness Testing)를 제안합니다. MAFT는 알고리즘이나 아키텍처에 구애받지 않고 딥 러닝(Deep Learning) 모델의 차별을 식별하고 해결할 수 있도록 설계되었습니다. 기존 방법이 복잡한 절차에 의존하는 반면, MAFT는 경량 절차를 사용하여 스케일과 적용성을 크게 향상시킵니다.

- **Technical Details**: MAFT는 기울기 추정(gradient estimation)과 속성 변형(attribute perturbation) 같은 경량 기술을 활용하여, 딥 뉴럴 네트워크(DNN)의 내부 작동에 대한 접근 없이도 공정성 테스트를 수행할 수 있는 모델 불가지론적(model-agnostic) 방법입니다. 이 방법은 기존의 ADF(Adversarial Discrimination Finder)와 EIDIG(Efficient Individual Discriminatory Instances Generator)의 워크플로우를 통해 효율성을 유지하면서, 모델에 대한 의존성을 제거하여 블랙박스 접근 방식으로 전환했습니다. 또한, MAFT는 동일한 효과성을 유지하면서 대규모 네트워크에서의 적합성을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: MAFT는 기존의 블랙박스 방법에 비해 효과성과 효율성 측면에서 각각 약 14.69배 및 32.58배 향상된 성능을 보입니다. 특히, MAFT는 AEQUITAS 및 SG보다 각각 1369.42% 및 3158.43% 향상된 성능을 보여주어 블랙박스 공정성 테스트 분야에서 다목적성을 입증합니다. 실험 결과, MAFT는 최첨단 백색상자 메서드인 EIDIG와 비슷한 수준의 성능을 나타내며, 기존 블랙박스 방법들보다 월등한 효율성을 제공합니다.



### Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Datas (https://arxiv.org/abs/2412.20072)
Comments:
          ICASSP 2025

- **What's New**: 이 연구는 기존의 공개된 텍스트와 표 데이터를 독립적으로 처리하는 데 탁월한 성능을 보이는 대형 언어 모델(LLMs)의 능력을 혼합된 장문(홀드) 문서의 정보 추출에 적용하는 방법을 제안합니다. Automated Information Extraction(AIE) 프레임워크를 통해 LLM이 HLD에서 정보를 효과적으로 추출할 수 있는 방법을 모색합니다. 이는 HLD의 문서 처리를 성공적으로 수행하기 위해 다양한 기술적 접근을 포함합니다.

- **Technical Details**: AIE 프레임워크는 세분화(Segmentation), 검색(Retrieval), 요약(Summarization), 추출(Extraction)의 네 가지 모듈로 구성됩니다. Segmentation 모듈은 HLD를 관리 가능한 세그먼트로 나누며, Retrieval 모듈은 임베딩 기반 검색을 통해 키워드와 관련된 세그먼트를 효율적으로 검색합니다. 요약 모듈은 LLM을 사용하여 검색된 세그먼트에서 핵심 정보를 요약하며, 최종적으로 Extraction 모듈은 LLM을 통해 숫자 값을 정확하게 추출합니다.

- **Performance Highlights**: AIE는 세 가지 데이터셋에 대한 실험에서 기본 LLM 기반 접근 방식보다 일관되게 높은 성능을 보였으며, 특히 더 엄격한 기준에서는 그 차이가 더욱 두드러졌습니다. AIE는 다양한 키워드 집합 간의 모호성을 처리하는 데에도 우수함을 보였으며, '수익'과 '총 순자산'과 같은 개념들 간에 의미를 구별하는 데 특히 효과적이었습니다. 실험 결과는 AIE의 구조적 접근이 LLM을 통해 복잡한 데이터에서 정보를 추출하는 데 매우 유용하다는 사실을 입증했습니다.



### On the Compositional Generalization of Multimodal LLMs for Medical Imaging (https://arxiv.org/abs/2412.20070)
- **What's New**: 이번 연구는 Multimodal Large Language Models (MLLMs)이 의료 분야에서 이미지 일반화의 능력을 탐구하는 데 중요한 역할을 하고 있음을 강조합니다. MLLMs의 한계는 불충분한 데이터에 기인하므로, 어떤 종류의 이미지가 MLLMs의 일반화에 유용한지 이해할 필요가 있습니다. 본 연구는 Composition Generalization (CG)를 사용하여 MLLMs가 의료 이미지를 이해할 수 있는 방법을 제안하며, Med-MAT라는 새로운 데이터셋을 개발했습니다.

- **Technical Details**: Med-MAT는 106개의 의료 데이터셋을 조합하여 생성된 데이터셋으로, 각 데이터는 Modality, Anatomical Area, 그리고 Task의 세 가지 요소인 MAT-Triplet으로 명확하게 정의되었습니다. 이 데이터셋은 의료 이미지를 기반으로 CG를 탐구할 수 있는 기회를 제공합니다. 연구자들은 데이터의 관련성을 분석하기 위해 MLLMs의 성능을 비교하고, CG가 다양한 이미지 유형에서 모델의 일반화 성능에 미치는 영향을 조사하였습니다.

- **Performance Highlights**: 실험 결과, MLLMs는 CG를 활용하여 이전에 보지 못한 의료 이미지를 이해할 수 있으며, CG가 멀티태스크 훈련에서 나타나는 일반화의 주요 요인 중 하나라는 점이 확인되었습니다. CG는 데이터 수가 적은 경우에도 유용하게 작용해 다양한 MLLMs 구조에서 일관된 성능을 제공함을 보여주었습니다. 이러한 결과는 의료 영역에서 MLLMs의 성공적인 활용 가능성을 제시합니다.



### The Emotional Spectrum of LLMs: Leveraging Empathy and Emotion-Based Markers for Mental Health Suppor (https://arxiv.org/abs/2412.20068)
- **What's New**: 본 논문에서는 정신 건강 지원을 위한 새로운 심리 평가 접근 방식을 기반으로 한 대화형 AI 시스템인 RACLETTE를 개발하였습니다. 이 시스템은 사용자의 감정 상태를 이해하고 공감적 반응을 생성하는 데 있어 기존의 최신 기술보다 우수한 성능을 보입니다. 또한, 사용자의 감정 프로필을 구축하는 과정에서 기존의 대화 기술을 증진시키는 가능성을 보여줍니다.

- **Technical Details**: RACLETTE는 3턴 구조를 채택하여 사용자의 감정을 다음 토큰 예측으로 훈련시키고, Mistral 7B 모델의 생성 능력을 활용하여 공감적으로 반응합니다. 사용자의 감정 프로필은 대화 중에 실시간으로 업데이트되며, 이는 시스템이 사용자의 감정 상태를 보다 정확히 이해할 수 있도록 합니다. 이러한 감정 프로필은 다양한 정신 장애와 통계적으로 비교될 수 있는 설명 가능한 지표로 활용됩니다.

- **Performance Highlights**: 연구 결과, RACLETTE는 감정 분류 작업에서 높은 정확도를 보이며, 공감적 응답 생성의 질 또한 탁월함을 입증했습니다. 감정 프로필을 통해 사용자 간의 다양한 감정 패턴을 인식하고 이를 기반으로 초기 진단 및 검사에 활용할 수 있는 가능성을 제시합니다. 이러한 접근 방식은 정신 건강 문제의 조기 발견과 진단에 기여할 것으로 예상됩니다.



### VELoRA: A Low-Rank Adaptation Approach for Efficient RGB-Event based Recognition (https://arxiv.org/abs/2412.20064)
Comments:
          In Peer Review

- **What's New**: 본 논문은 RGB-Event 인식을 위한 새로운 파라미터 효율적인 미세 조정(Parameter Efficient Fine-Tuning, PEFT) 전략인 VELoRA를 제안합니다. 기존의 무거운 모델 미세 조정 방식에서 벗어나 LoRA와 같은 경량 미세 조정 방법을 사용하여 성능과 효율의 균형을 맞추려 합니다. 이를 통해 RGB 및 이벤트 카메라를 활용한 멀티모달 작업의 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: 이 논문은 먼저 RGB 프레임과 이벤트 스트림을 통해 프레임 차이를 추출하고, 이를 통해 모션 정보를 캡처합니다. 제안된 LoRA 튜닝 방법을 통해 RGB와 이벤트 데이터를 구분하여 인코딩하고, 이들 간의 상호작용을 통해 특징을 재구성하여 멀티모달 특징 학습을 수행합니다. 마지막으로, 이렇게 통합된 특징들을 분류하는 네트워크에 입력하여 효율적인 미세 조정을 진행합니다.

- **Performance Highlights**: 본 연구는 여러 이벤트 기반 패턴 인식 벤치마크 데이터 세트에서 VELoRA의 효과를 검증하며, 종합적인 실험 결과를 통해 이 전략이 기존 모델들보다 더 높은 성능을 발휘하는 것을 보여줍니다. 특히, 제안된 모델은 계산 비용과 정확도 간의 보다 나은 균형을 제공하며, 다양한 다운스트림 태스크에 유용하게 활용될 수 있습니다.



### CrossSpeech++: Cross-lingual Speech Synthesis with Decoupled Language and Speaker Generation (https://arxiv.org/abs/2412.20048)
- **What's New**: 이 논문은 CrossSpeech++를 제안하여 다양한 언어에서 자연스러운 음성을 생성하면서도 동일한 화자 정체성을 유지하는 문제를 다룹니다. 특히, 언어와 화자 정보의 얽힘 문제를 효과적으로 분리하고 이를 통해 크로스링구얼(Cross-lingual) 음성 합성의 품질을 획기적으로 향상시킵니다. 이 방법은 언어 의존 생성기와 화자 의존 생성기로 음성 생성 파이프라인을 두 부분으로 나누어 각 정보를 독립적으로 처리합니다.

- **Technical Details**: CrossSpeech++는 음성 생성 과정에서 언어 정보를 포함한 음향 특징(output acoustic feature)과 화자 정보의 분리를 통해 자연스러운 음성 합성을 목표로 합니다. 언어 의존 생성기(Language-dependent Generator, LDG)와 화자 의존 생성기(Speaker-dependent Generator, SDG)가 각각 음성의 언어적 변화를 담당하고 화자 속성을 모델링합니다. LDG는 세 가지 구성 요소(Mix Dynamic Speaker Layer Normalization, Language-Dependent Variance, Linguistic Adaptor)를 포함하고 있으며, SDG는 Dynamic Speaker Layer Normalization과 Speaker-Dependent Variance를 통해 화자 특성을 강화합니다.

- **Performance Highlights**: 실험 결과, CrossSpeech++는 주관적 및 객관적 평가 지표 모두에서 크로스링구얼 음성 합성 품질을 기존 방법들보다 상당히 향상했습니다. 특히, 음향 샘플을 통해 생성된 음성이 더 자연스럽고 정교하다는 것이 입증되었습니다. 이 논문에 제시된 기법은 다국어 음성 처리에 있어 중요한 발전을 제공하며 다양한 응용 분야에서의 활용이 기대됩니다.



### Enhancing Diffusion Models for Inverse Problems with Covariance-Aware Posterior Sampling (https://arxiv.org/abs/2412.20045)
- **What's New**: 논문에서는 새로운 방법인 covariance-aware diffusion posterior sampling (CA-DPS)를 제안하여, 기존의 denoising diffusion probabilistic models (DDPMs)을 활용하여 노이즈가 있는 선형 역문제를 해결하는 접근법을 향상시킵니다. 기존 방법들이 개별 모델 훈련을 요구하는 반면, CA-DPS는 추가적인 훈련 없이도 상태를 추정할 수 있는 우수한 방법으로 자리잡습니다. 이를 통해 높은 성능의 신호 복원이 가능하며, 이는 딥러닝 훈련의 복잡성을 덜어줍니다.

- **Technical Details**: CA-DPS의 핵심은 반전 프로세스의 공분산을 공식화한 것입니다. 이 공분산을 기반으로 한 근사식은 기존의 사전 훈련된 DDPM에서 쉽게 구할 수 있습니다. 논문에서는 Tweedie 공식을 사용하여 Conditional Mean을 계산하고, 이를 통해 공분산을 보정함으로써 신뢰할 수 있는 Likelihood 근사를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, CA-DPS는 하이퍼파라미터 조정 없이도 재구성 성능을 유의미하게 향상시키는 것을 보여주었습니다. 이는 기존의 모델들에 비해 노이즈 환경에서도 뛰어난 결과를 기록하며, 이미지, 비디오 및 오디오 합성 분야에서의 가능성을 더욱 확장합니다. 또한, 이 접근 방식은 시간 의존성 문제를 우회하여 효율성을 증가시킵니다.



### Calibre: Towards Fair and Accurate Personalized Federated Learning with Self-Supervised Learning (https://arxiv.org/abs/2412.20020)
Comments:
          ICDCS camera-ready paper, Code repo: this https URL

- **What's New**: 이 논문에서는 개인화된 페더레이티드 러닝(FL)에서 기존 모델 메커니즘의 한계를 극복하기 위해 새로운 프레임워크인 Calibre를 제안합니다. Calibre는 자기 지도 학습(SSL)을 활용하여 전 세계 모델이 공정성(fairness)과 정확성을 유지하면서 개인화된 모델을 최적화할 수 있도록 조정하는 메커니즘을 포함합니다. 이를 통해 각 클라이언트가 자신만의 데이터 특성에 맞는 고품질의 개인화된 모델을 훈련할 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 단계를 포함하는데, 첫째로 SSL을 이용해 전 세계 모델을 훈련하고, 둘째로 클라이언트가 이 모델을 피처 추출기로 활용해 개인화된 모델을 학습할 수 있게 합니다. 특히, 고객 특정 프로토타입 손실(client-specific prototype loss)과 프로토타입을 기반으로 한 집계 알고리즘(aggregation algorithm)을 도입하여 SSL의 표현력을 캘리브레이션(calibrate)합니다. 이 과정에서 각 클라이언트는 자신의 샘플과 관련된 프로토타입 간의 평균 거리를 계산하여 지역적 분산(local divergence rate)을 측정하는 방식을 사용합니다.

- **Performance Highlights**: 광범위한 비-i.i.d. 환경을 통해 실시된 실험에서 Calibre는 평균 정확도(mean accuracy)와 공정성(fairness) 모두에 대해 최첨단 성능을 달성했습니다. 특히, CIFAR-10, CIFAR-100, STL-10 데이터셋을 활용한 실험에서 경량화된 개별 모델인 선형 분류기(linear classifier)를 사용했음에도 불구하고, 좋은 성능을 발휘했습니다. 또한, Calibre는 훈련 과정에 참여하지 않은 새로운 클라이언트에서도 잘 일반화되는 특성을 가지고 있습니다.



### ProtCLIP: Function-Informed Protein Multi-Modal Learning (https://arxiv.org/abs/2412.20014)
- **What's New**: 이 논문은 단백질 서열과 생물학적 설명을 정렬하는 멀티 모달리티 사전 학습(paradigm) 접근법을 통해 일반적인 단백질 표현을 학습하고 여러 다운스트림 애플리케이션에서 유망한 성과를 달성했습니다. 이를 위해, 저자들은 ProtAnno라는 대규모 단백질-텍스트 쌍 데이터셋을 구성하고 기능 정보에 기반한 새롭고 효과적인 사전 학습 패러다임을 도입했습니다. 이 논문은 단백질의 정적 및 동적 기능 세그먼트를 명시적으로 모델링하여 세분화된 정보를 주입하는 방법을 제안합니다.

- **Technical Details**: 이 연구는 ProtAnno 데이터셋을 구축하여, 샘플 신뢰도와 속성 커버리지를 기반으로 선택 확률을 결정하는 특성 기반 샘플링 전략(property-driven sampling strategy)을 사용합니다. 이 전략은 대규모 노이즈 데이터의 데이터 품질과 양을 균형 있게 조절합니다. 또한, 제안된 방법은 CLIP 손실( loss)를 활용하여 단백질의 기능 메커니즘을 효과적으로 모델링하며, 세그먼트별 사전 학습 목표를 통해 단백질의 세분화된 정보도 포착합니다.

- **Performance Highlights**: ProtCLIP는 22개의 단백질 벤치마크에서 획기적인 성과를 달성하며, 특히 크로스-모달 변환 벤치마크에서 평균 75% 향상을 보였습니다. GO-CC 및 GO-BP 단백질 기능 예측에서도 각각 59.9% 및 39.7%의 성능 개선을 이룩했습니다. 이러한 결과는 ProtCLIP이 단백질 멀티 모달리티 기초 모델로서의 뛰어난 잠재력을 검증합니다.



### OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System (https://arxiv.org/abs/2412.20005)
Comments:
          Work in progress

- **What's New**: OneKE는 웹과 PDF 형식의 책에서 지식을 추출할 수 있는 도커화된 스키마 기반의 지식 추출 시스템으로 소개됩니다. 이 시스템은 다양한 도메인을 지원하며, 여러 개의 에이전트를 설계하여 각자의 역할을 수행하도록 만들어졌습니다. 또한, 구성 가능한 지식 기반이 오류 디버깅과 개선을 돕습니다.

- **Technical Details**: OneKE는 실제 데이터에서 지식을 추출하는 데 필요한 복잡성을 해결하기 위해 설계되었습니다. 스키마 에이전트는 다양한 데이터 유형에 대한 분석을 수행하고, 추출 에이전트는 곧추 사용 가능한 LLM을 이용해 지식을 추출하며, 반영 에이전트는 오류를 디버깅하는 역할을 담당합니다. 이를 통해 다양한 데이터 포맷 및 길이에 유연하게 대응할 수 있는 기능이 있습니다.

- **Performance Highlights**: OneKE는 CrossNER 및 NYT-11-HRL 데이터셋에서 평가되었으며, 두 개의 주요 작업인 NER과 RE 모두에서 성능 향상을 보였습니다. 특히, 추출 에이전트의 사례 검색 방법이 가장 큰 성과를 달성하고 있으며, 복잡한 RE 작업에서는 중간 추론 단계가 더 중요한 것으로 나타났습니다. 이 시스템은 사용자 정의 스키마와의 통합을 통해 더욱 강력한 지식 추출 성능을 발휘합니다.



### Adaptive Parameter-Efficient Federated Fine-Tuning on Heterogeneous Devices (https://arxiv.org/abs/2412.20004)
- **What's New**: 이 논문에서는 Federated fine-tuning (FedFT) 방법을 통해 사전 훈련된 언어 모델을 효율적으로 미세 조정할 수 있는 새로운 접근 방식을 제안하고 있습니다. 기존의 parameter-efficient fine-tuning 방법들이 갖는 제약을 극복하기 위해, LoRA(저랭크 적응) 계층을 조정하여 리소스 소비를 줄이고 비슷한 성능으로 결과를 도출할 수 있음을 보여줍니다. 새로운 FEDFT 프레임워크인 LEGEND는 LoRA 계층의 깊이(LoRA depth)와 순위를 최적화하여 이질적인 장치 환경에서 효율적인 미세 조정을 가능하게 합니다.

- **Technical Details**: LEGEND 프레임워크는 LoRA 계층을 통해 미세 조정을 수행하며, 계층의 깊이와 순위 분포를 조절하여 리소스 소비와 성능 사이의 균형을 맞춥니다. 이 연구에서 제안된 알고리즘은 개별 장치의 제약을 감안하여 LoRA 깊이와 순위를 각각 어떻게 설정할지를 결정합니다. 80대의 상용 장치로 구성된 물리적 플랫폼에서 실험을 실시하여, LEGEND의 성능을 평가하였으며, 기존 솔루션과 비교해 빠른 수렴 속도를 보여주는 결과를 도출하였습니다.

- **Performance Highlights**: LEGEND 프레임워크는 미세 조정 속도를 1.5-2.8배 향상시켰으며, 목표 정확도에 도달할 때 통신 비용을 약 42.3% 절감했습니다. 이러한 결과는 리소스가 제한된 장치에서도 효과적으로 적용 가능하다는 것을 입증하며, 프레임워크의 효율성을 강조합니다. 이 새로운 접근 방식은 실제 무선 환경에서 이질적인 장치들 간의 협업적 학습을 가능케 하여, 언어 처리의 발전에 기여할 것으로 기대됩니다.



### Comprehensive Review of EEG-to-Output Research: Decoding Neural Signals into Images, Videos, and Audio (https://arxiv.org/abs/2412.19999)
Comments:
          15 pages. Submitted as a conference paper to IntelliSys 2025

- **What's New**: 이 논문은 EEG(뇌파 검사)가 현대 신경과학에서 중요한 역할을 하며, 최근의 머신러닝 및 생성 모델의 발전이 EEG를 활용하여 인지 경험을 재구성하는 데 어떻게 도움을 주는지를 검토한다. 연구자들은 EEG-출력 연구의 최신 동향을 체계적으로 분석하고, 주요 격차와 기회를 강조하며, GANs(Generative Adversarial Networks), VAEs(Variational Autoencoders), Transformers와 같은 고급 모델의 잠재력을 탐구하고 있다. 한편, 연구 분야의 표준화된 데이터셋 필요성과 주제 간 일반화의 중요성을 강조한다.

- **Technical Details**: 이 논문은 PRISMA 가이드라인을 따르며, 1800개의 연구를 시스템적으로 분석하여 최첨단 생성 방법, 평가 메트릭, 데이터 과제를 다룬다. 연구에서는 EEG 신호에서 인지 출력으로의 재구성을 다루는 다양한 생성 모델을 검토하며, EEG 신호 처리의 혁신을 가져온 딥러닝 기술의 발전을 조명한다. EEG2Image와 같은 연구가 EEG 특성을 추출하고 이미지를 합성하는 두 단계 접근 방식을 포함하여 대한 인사이트를 제공한다.

- **Performance Highlights**: EEG를 사용한 생성 모델의 성과는 감각 처리 및 인지 표현에 대한 깊은 통찰을 제공할 수 있는 능력으로 더욱 강화된다. 여기서 중요한 성과로는 EEG2Image가 있으며, 이는 EEG 신호에서 시각적 자극에 해당하는 이미지를 재구성하는 데 뚜렷한 진전을 보여준다. 그러나 여전히 EEG 신호의 고유한 노이즈와 변동성, 그리고 공간적 해상도의 한계가 신뢰성 및 재현 가능성을 저해하고 있어 이러한 도전 과제가 해결되어야 한다.



### From Generalist to Specialist: A Survey of Large Language Models for Chemistry (https://arxiv.org/abs/2412.19994)
Comments:
          COLING2025,We maintain an up-to-date Github repository at: this https URL

- **What's New**: 이 논문에서는 화학(Langage Model for Chemistry) 분야에서 대형 언어 모델(LLM)의 사용을 개선하기 위한 방법론을 소개하고 있습니다. 특히, 화학 도메인 특별한 지식 및 다중 모달(multi-modal) 정보를 LLM에 통합하는 방법을 설명합니다. 기존 연구에서 다뤄지지 않았던 화학 중심 LLM에 대한 체계적인 조사를 제공하여, 연구자들이 화학 LLM의 발전을 따라갈 수 있도록 돕고자 합니다.

- **Technical Details**: LLM의 일반적인 문제점으로는 도메인 지식 부족, 다중 모달 데이터의 비인식 등이 있습니다. 많은 LLM들이 인터넷에서 수집된 웹 데이터로 사전 학습되기 때문에 화학 관련 데이터는 부족합니다. 이로 인해 강화학습(RLHF) 및 특정 작업에 대한 세부 조정이 불충분하여 화학에 대한 응용력이 떨어집니다.

- **Performance Highlights**: 논문에서는 다양한 화학적 데이터 다루기 위한 새로운 모델을 제안하고 기존 모델들이 화학 관련 작업에서 성능이 미흡하다는 것을 지적합니다. 또한 여러 화학 분석 도구와 기법을 통해 연구를 가속화할 수 있는 잠재력을 탐구하며, 향후 연구를 위한 기대되는 방향성을 제시합니다. 이를 통해 화학 분야의 혁신적인 응용 사례를 발굴하고 연구자들에게 통찰을 제공합니다.



### An Ordinary Differential Equation Sampler with Stochastic Start for Diffusion Bridge Models (https://arxiv.org/abs/2412.19992)
Comments:
          9 pages, 5 figures, This work has been submitted to the IEEE for possible publication

- **What's New**: 이 연구는 Diffusion bridge 모델의 느린 추론 속도 문제를 해결하기 위해 ODE Sampler with a Stochastic Start (ODES3)를 제안합니다. 기존의 Stochastic Differential Equation (SDE) 샘플러에 비해 고차 Ordinary Differential Equation (ODE) 솔버를 사용하여 초기 이미지를 부드럽게 변환하는 방법에 중점을 두고 있습니다. 이 접근법은 기존의 모델들과 호환되며 추가적인 학습 없이도 뛰어난 성능을 발휘합니다.

- **Technical Details**: ODS3는 먼저 손상된 이미지에서 중간 표현으로의 전환을 위해 posterior sampling을 수행하고, 이후 Heun의 2차 솔버를 사용하여 확률 흐름 ODE(PF-ODE)를 해결합니다. 이 과정은 손상된 이미지에서 생성 경로로의 부드러운 전환을 보장하며, discretization 오류를 줄입니다. 이 방법은 사전 학습된 Diffusion bridge 모델과 완벽히 호환되며, 추가적인 훈련이 필요하지 않습니다.

- **Performance Highlights**: 이 샘플러는 슈퍼 해상도, JPEG 복원 및 이미지 번역 태스크에서 기존 최첨단 방법들보다 높은 시각적 품질과 Frechet Inception Distance (FID) 점수를 기록했습니다. 여러 데이터셋에서 수행된 실험 결과, 제안된 방법은 기존 Diffusion bridge 모델의 원래 샘플러보다 성능이 향상된 것으로 나타났습니다.



### Delayed Random Partial Gradient Averaging for Federated Learning (https://arxiv.org/abs/2412.19987)
- **What's New**: 본 논문은 지연된 랜덤 부분 경량 평균(Delayed Random Partial Gradient Averaging, DPGA)을 제안하여 연합 학습(Federated Learning, FL)의 통신 병목을 동시에 해결합니다. DPGA는 클라이언트가 지역 모델의 일부 경량만을 서버와 공유하여 대역폭(bandwidth) 문제를 해결하고, 통신 중에도 지역 계산을 지속하도록 하여 대기 시간을 감소시킵니다. 이러한 접근 방식은 비독립 동일 분포(non-IID) 데이터셋인 CIFAR-10/100를 사용하여 그 효과성을 입증합니다.

- **Technical Details**: DPGA에서는 클라이언트가 각 통신 라운드에서 공유하는 경량의 크기를 동적 업데이트 비율을 기반으로 결정합니다. 업데이트 비율은 범위 내에서 조정되며, 클라이언트의 지역 연산 성능을 개선하게 됩니다. 또한, 클라이언트는 부분 경량 통신 중에도 계속적으로 지역 계산을 수행하여 시스템의 총 실행 시간을 줄입니다.

- **Performance Highlights**: 실험 결과, DPGA는 정확도(accuracy), 통신 비용(communication cost), 실행 시간(run time) 측면에서 최신 방법들(state-of-the-art)보다 우수한 성능을 보였습니다. 이는 클라이언트와 서버 간의 효과적인 정보 교환을 통해 이루어집니다. 따라서 DPGA는 FL 환경에서 발생하는 다양한 통신 이슈를 효과적으로 해결합니다.



### The Fifth International Verification of Neural Networks Competition (VNN-COMP 2024): Summary and Results (https://arxiv.org/abs/2412.19985)
Comments:
          Report on the results of VNN-COMP 2024. arXiv admin note: substantial text overlap with arXiv:2312.16760, arXiv:2212.10376

- **What's New**: 이번 논문은 제5회 국제 신경망 검증 대회(VNN-COMP 2024)에 대한 요약을 제공합니다. 이 대회는 AI 검증(SAIV) 국제 심포지움 및 컴퓨터 지원 검증(CAV) 국제 회의와 함께 개최되었습니다. 매년 개최되는 VNN-COMP의 목적은 신경망 검증 도구들의 객관적인 비교를 촉진하고, 도구 인터페이스의 표준화를 장려하며, 신경망 검증 커뮤니티를 결집시키는 것입니다.

- **Technical Details**: 대회에서는 네트워크에 대한 표준화된 포맷인 ONNX와 명세서 VNN-LIB가 정의되었습니다. 도구 평가를 위해 AWS 인스턴스를 기반으로 한 자동 평가 파이프라인이 사용되었으며, 참가자들은 최종 테스트 세트가 공개되기 전에 도구 파라미터를 선택했습니다. 2024년 버전에서는 8개 팀이 12개의 일반 벤치마크와 8개의 확장 벤치마크 세트를 대상으로 참가했습니다.

- **Performance Highlights**: 이 보고서는 이번 대회의 규칙, 벤치마크, 참여 도구, 결과 및 학습된 교훈을 요약합니다. 각 도구의 성능을 비교하는 데 있어, 공정하고 일관된 평가 방법이 도입되어 신뢰성 있는 결과를 도출했습니다. 이러한 신경망 검증 도구들의 발전은 향후 AI 안전성을 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### Will you donate money to a chatbot? The effect of chatbot anthropomorphic features and persuasion strategies on willingness to dona (https://arxiv.org/abs/2412.19976)
Comments:
          13 pages, 2 figures

- **What's New**: 본 논문은 챗봇(personified chatbot)의 인격화(personification)와 설득 전략(persuasion strategies)이 사용자 인식과 기부 가능성에 미치는 영향의 인과 메커니즘을 조사합니다. 특히, 챗봇이 비인격화(non-personified chatbot)된 상태에서 감정적(emotional) 또는 논리적(logical) 설득 방식을 적용한 실험을 통해, 개인화된 챗봇이 사용자에게 미치는 영향을 분석하였습니다.

- **Technical Details**: 이 연구는 2(인격화된 vs. 비인격화된 챗봇) x 2(감정적 vs. 논리적 설득 전략) 디자인의 준실험적 연구를 진행하였으며, 총 76명의 참가자가 비영리 자선 단체를 대표하는 챗봇과 상호작용했습니다. 연구 결과, 인격화된 챗봇과의 상호작용은 인지된 인류성(perceived anthropomorphism)을 불러일으켰으나, 기부 의향(donation likelihood)에는 긍정적인 영향을 미치지 못했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 일반적으로 사용되는 인격화된 특성은 기부 상황에서 AI 에이전트에 대한 부정적인 태도를 유도했습니다. 이에 따라, 논리적 설득 방식을 결합한 비인격화된 챗봇이 선호된다는 결과가 나타났습니다. 이러한 발견은 챗봇 상호작용의 일관성(consistency)과 인간 대 인간 상호작용(human-human engagement)의 유사성을 강조하며, AI 시스템에 대한 최근 규제를 고려할 때, 챗봇의 정체성 연구에 대한 전환의 중요성을 논의합니다.



### MobileNetV2: A lightweight classification model for home-based sleep apnea screening (https://arxiv.org/abs/2412.19967)
- **What's New**: 이번 연구는 심전도(ECG) 및 호흡 신호에서 추출한 특징을 활용하여 경량화된 신경망 모델을 제안하여 조기 폐쇄성 수면 무호흡증(OSA) 스크리닝을 개선합니다. 이 모델은 수면 단계 예측을 위한 특징 스펙트로그램과 호흡 이상 검출을 통합하여 아페네-하이포네아 지수(AHI)를 더욱 정확하게 계산합니다. 모델은 Apnea-ECG, UCDDB 데이터셋 및 MIT-BIH 폴리솜노그래피 데이터베이스의 세 가지 공개 데이터베이스에서 검증되었으며, OSA 탐지 정확도가 0.978로 나타났습니다.

- **Technical Details**: 제안된 방법은 ECG 기록과 혈중 산소 데이터를 포함하는 Apnea-ECG 데이터베이스, PSG 데이터를 포함하는 UCDDB 데이터셋, 다채널 기록이 포함된 MIT-BIH 폴리솜노그래피 데이터베이스의 데이터를 사용하여 검증되었습니다. ECG 데이터는 1분 단위로 나누어져 슬라이딩 윈도우 기법으로 처리되며, AHI를 계산하기 위해 수면 시간을 Sleep(S)과 Non-Sleep(NS)으로 구분합니다. MobileNet 모델은 깊이 분리형 합성을 기반으로 하여 모바일 애플리케이션에 적합한 경량화된 구조를 제공합니다.

- **Performance Highlights**: 모델의 성능은 UCDDB 데이터셋에서 수면 단계 분류 시 모든 단계에서 ROC-AUC 값이 0.85를 초과하였으며, 수면의 recall은 0.906, REM 및 Wake 상태의 specificity는 각각 0.956 및 0.937로 높은 정확도를 보였습니다. 호흡 사건 분류 정확도는 0.969에 달하며, ROC-AUC는 0.98로 확인되었습니다. 이러한 결과는 가정 및 웨어러블 건강 모니터링 시스템에서 OSA 스크리닝의 확산 가능성을 강조하고 있습니다.



### Bridging Context Gaps: Enhancing Comprehension in Long-Form Social Conversations Through Contextualized Excerpts (https://arxiv.org/abs/2412.19966)
Comments:
          Accepted at COLING 2025

- **What's New**: 이번 연구는 소규모 기록된 대화에서 이해력을 향상시키는 방법에 중점을 두고 있습니다. 사람들을 연결하고 중요한 사회적 문제에 대한 개인의 이야기를 나눌 수 있는 공간을 제공하기 위해 이러한 대화를 활용합니다. 우리는 대화에서 강조된 발췌(text excerpt)를 후속 대화에서 공유하는 방식을 제안하여 관련 문제에 대한 집단적 이해를 촉진하고자 합니다.

- **Technical Details**: 본 연구에서 다루는 주요 도전 과제는 한 대화에서 발췌한 내용을 다른 맥락에서 공유할 때 발생하는 원본 대화의 맥락이나 핵심 요소의 부족입니다. 이러한 문제는 대화가 길어지고 주제가 풍부해질수록 더욱 악화됩니다. 우리는 Large Language Models (LLMs)를 활용하여 이러한 발췌에 사회적으로 관련된 맥락을 제공하는 방법을 탐색했습니다. 이는 이해력, 가독성(readability), 그리고 공감(empathy) 향상에 도움을 줍니다.

- **Performance Highlights**: 우리 연구는 주관적 및 객관적 평가를 통해 이해력의 상당한 개선을 보여주었습니다. LLMs는 유용한 맥락을 제공할 수 있지만, 중요한 사회적 측면을 포착하는 데 어려움을 겪습니다. 이를 위해 Human-annotated Salient Excerpts (HSE) 데이터셋을 공개하여 향후 연구를 지원하고, 맥락이 풍부한 발췌가 더 집중적이고 포괄적인 대화 요약을 제공할 수 있음을 입증하였습니다.



### DepthMamba with Adaptive Fusion (https://arxiv.org/abs/2412.19964)
- **What's New**: 이 연구에서는 다중 시점(depth estimation) 시스템이 실제 환경에서 자주 발생하는 노이즈가 있는 카메라 포즈를 다루는 새로운 견고성(robustness) 벤치마크를 제안합니다. 기존의 다중 시점 시스템이 이상적인 카메라 포즈에 의존하고 있다는 점을 지적하고, 이에 대응하는 새로운 방법론을 소개합니다.

- **Technical Details**: 제안된 네트워크 아키텍처는 두 개의 브랜치 채널을 결합하여 단일 시점(single-view)과 다중 시점(multi-view)에서의 깊이 추정 결과를 융합합니다. 특히 mamba라는 특징 추출 백본(backbone)을 도입하고, 두 브랜치 간의 가장 강력한 추정 결과를 적응적으로 선택하는 attention 기반의 융합 방법을 제안했습니다.

- **Performance Highlights**: 제안된 방법은 동적 객체 및 텍스처가 없는 지역을 포함한 도전적인 장면에서 우수한 성능을 보여줍니다. 아블레이션 연구(ablation studies) 결과는 백본과 융합 방법의 효과를 입증하며, KITTI 및 DDAD와 같은 도전적인 벤치마크에서 기존의 최첨단(state-of-the-art) 방법들과 비교해 경쟁력 있는 성능을 달성했습니다.



### ErgoChat: a Visual Query System for the Ergonomic Risk Assessment of Construction Workers (https://arxiv.org/abs/2412.19954)
Comments:
          32 pages, 8 figures

- **What's New**: 이 연구는 건설 근로자의 자세 관련 인체공학적 위험을 평가하기 위해 인터랙티브한 비주얼 쿼리 시스템을 도입하였습니다. 기존의 전통적인 인체공학적 위험 평가(ERA) 방법은 상호작용 피드백을 제공하지 않지만, 새로운 시스템은 이미지 입력을 기반으로 질문에 답하고 텍스트 설명을 생성할 수 있는 기능을 갖추고 있습니다. 이 연구는 또한 이러한 방법론을 훈련하고 테스트할 수 있는 데이터셋을 제안합니다.

- **Technical Details**: 이 시스템은 비주얼 질문 답변(VQA) 기능과 이미지 캡셔닝(IC) 기능을 포함하고 있습니다. VQA는 근로자가 노출된 자세 관련 인체공학적 위험에 대한 비주얼 쿼리에 응답하며, IC는 이미지에서 이러한 위험에 대한 텍스트 설명을 생성합니다. 체계적인 테스트 결과, VQA 기능은 96.5%의 정확도를 달성하였고, IC 성능 평가에서는 아홉 가지 지표와 인간 전문가의 평가를 통해 새로운 접근법이 일반 데이터셋만으로 훈련된 동일 아키텍처의 방법보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 연구 결과, 비주얼 질문 답변 기능이 96.5%의 높은 정확도를 기록했으며, 이미지 캡셔닝 측면에서도 새로운 시스템이 기존 방법보다 우수한 성능을 발휘했습니다. 이는 AI 기술을 활용한 인터랙티브한 인체공학적 위험 평가(ERA)의 새로운 방향성을 제시하며, 향후 연구 및 개발에 큰 기여를 할 것으로 기대됩니다.



### Standard-Deviation-Inspired Regularization for Improving Adversarial Robustness (https://arxiv.org/abs/2412.19947)
- **What's New**: 이 연구는 Adversarial Training (AT) 기법의 효율성을 높이기 위해 표준 편차에 기반한 정규화 항(SDI measure)을 제안합니다. 기존 AT가 취약기를 훈련시키는 데 효과적인 반면, 이번 연구는 SDI 정규화를 통해 더욱 강력한 공격에 대한 DNN의 강인성을 향상시키는 방법을 제시합니다. 연구를 통해 SDI 측정이 adversarial 예제를 작성하는 데 유용하다는 실험 결과도 보여줍니다.

- **Technical Details**: 연구는 AT의 내부 최대화 단계가 모델 출력 확률의 변별력을 극대화하려는 시도와 유사하다는 점을 강조합니다. 또한 SDI 손실 줄이기를 AT의 외부 최소화 단계와 연관 지어 설명합니다. SDI 측정은 cross-entropy, entropy 또는 Kullback-Leibler divergence와 같은 기존의 정보 이론적 손실들에 의존하지 않으며, 이러한 특성으로 인해 SDI는 기존 AT 변형과 통합될 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: 제안된 SDI 정규화 항의 적용은 Auto 공격과 CW 공격에 대한 기존 AT 변형의 강인성을 더욱 개선하는 결과를 가져왔습니다. 또한, SDI 정규화가 기존 AT 변형의 일반화 성능 역시 향상시켜, adversarial 훈련 중에 관찰되지 않은 공격에 대해 더 나은 성능을 보였습니다. 마지막으로 SDI 메트릭을 사용하여 생성된 adversarial 예제는 cross-entropy 손실 및 KL divergence를 사용하여 생성된 것들과 비교되어, 더 높은 성공률을 보여주었습니다.



### Towards Strong AI: Transformational Beliefs and Scientific Creativity (https://arxiv.org/abs/2412.19938)
- **What's New**: 이번 논문에서는 강인공지능(strong AI)의 개념과 관련하여 천문학 및 물리학 역사에서의 주요 발견을 탐구합니다. 특히, 해왕성의 발견과 과학 혁명(scientific revolutions) 개념을 철학자들의 관점에서 분석합니다. 또한, 과학적 창의성을 모델링하기 위한 기반으로 변환신념(Transformational Belief, TB) 프레임워크를 도입합니다.

- **Technical Details**: TB 프레임워크는 약한 신념(weak beliefs)을 이론적 및 통계적 방식으로 정형화한 것입니다. 이 프레임워크는 통계 과학(statistical science)에서의 사례를 통해 과학적 창의성의 이해와 분석, 촉진을 위한 잠재력을 보여줍니다. 이는 강인공지능 개발을 위한 중요한 기반이 될 수 있습니다.

- **Performance Highlights**: TB 프레임워크는 창의성을 이해하고 지원하는 데 유망한 토대를 제공하여, 강인공지능의 개발을 위한 새로운 길을 여는 잠재력을 가지고 있습니다. 논문의 결론에서는 향후 연구 방향과 발전 가능성에 대한 고찰이 포함되어 있습니다.



### Hidformer: Transformer-Style Neural Network in Stock Price Forecasting (https://arxiv.org/abs/2412.19932)
Comments:
          12 pages, 6 figures, 4 tables

- **What's New**: 이 연구는 Transformer 기반의 신경망을 주식 가격 예측에 적용하는 방법을 탐구합니다. 머신러닝 기법과 금융 시장 분석의 교차점을 중점적으로 다루고 있으며, Hidformer 모델의 발전을 주목합니다. 이번 연구는 변형된 Hidformer 모델이 주식 가격 예측에서도 뛰어난 성능을 발휘하는지를 평가하는 것을 목표로 합니다.

- **Technical Details**: Hidformer 모델은 금융 컨텍스트에서의 시계열(data) 분석을 위해 설계된 Transformer 모델의 한 예로, 기술적 분석(technical analysis)의 원리와 고급 머신러닝 개념을 통합하여 주식 가격 예측의 정확성을 높입니다. 연구에서는 이 모델의 성능을 평가하기 위해 여러 기준(criteria)을 사용하여 검증을 실시합니다. 이러한 접근법은 기존의 알고리즘적 거래 전략을 향상시키는 데 기여할 수 있습니다.

- **Performance Highlights**: 연구 결과는 Transformer 아키텍처가 금융 시계열 예측에 활용될 수 있는 가능성을 강조합니다. Hidformer의 성능은 실험을 통해 구체적으로 평가되었으며, 알고리즘적 거래와 인간 의사결정(human decision making)을 포함한 다양한 응용 분야에서의 잠재력을 보여주었습니다. 이로 인해 금융 시장 예측의 정확성이 향상될 것으로 기대됩니다.



### Pivoting B2B platform business models: From platform experimentation to multi-platform integration to ecosystem envelopmen (https://arxiv.org/abs/2412.19931)
- **What's New**: 이 논문은 제조업에서 디지털 서비스화(digital servitization) 변모의 경향과 플랫폼 비즈니스 모델(platform business models, BMs)의 전략적 변화에 대해 논의합니다. 제조업체가 B2B 플랫폼 개발 시 자주 직면하는 실패 사례를 다루며, 이에 대한 해결책으로 3단계 회전 프레임워크를 제시합니다. 특히, 에너지 분야의 특정 사례를 통해 플랫폼 전략의 발전을 명확하게 설명합니다.

- **Technical Details**: 연구는 자산 기반(product sales)을 중시하던 제조사가 고객 여정(customer journey) 전략을 중심으로 AI 기술을 통한 다중 플랫폼 통합으로 전환하는 과정을 탐구합니다. 초기의 독립적인 B2B 플랫폼에서 점차 통합된 플랫폼으로 나아가며, 데이터 기반(data-driven) 서비스 제공의 필요성을 강조합니다. 이는 플랫폼 BM의 성숙 단계로, 더욱 포괄적이고 복합적인 서비스를 가능하게 합니다.

- **Performance Highlights**: 최종적으로, 이 연구는 제조업체가 외부 이해관계자와의 협력을 통해 데이터 주도(data-driven) 플랫폼 생태계를 구축하는 방안을 제시합니다. 연구 결과는 디지털 서비스화와 B2B 플랫폼 BM 개발의 효율성을 향상시키기 위한 점진적 접근법과 전략적 회전(pivoting)의 중요성을 강조합니다. 이러한 발견은 제조업체가 경쟁 우위를 얻는 데 실질적인 기여를 할 것입니다.



### Modeling Continuous Spatial-temporal Dynamics of Turbulent Flow with Test-time Refinemen (https://arxiv.org/abs/2412.19927)
Comments:
          14 pages

- **What's New**: 이 논문에서는 저해상도의 대형 와동 시뮬레이션(LES) 데이터를 활용해 고충실도 직접 수치 시뮬레이션(DNS) 데이터를 복원하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 물리적 지식을 활용하여 흐름 동역학을 모델링하는 데 초점을 맞추고 있습니다. 기존의 초해상도(super-resolution) 기술과 달리, 제안된 방법은 테스트 단계에서만 LES 데이터를 사용하여 물리적 제약을 적용하고 누적 재구성 오류를 줄이는 방식을 채택합니다.

- **Technical Details**: 논문에서 제안된 방법은 두 가지 주요 구성요소로 구성되어 있습니다: 저하 기반 정제(degradation-based refinement)와 연속 공간 전이 유닛(Continuous Spatial Transition Unit, CSTU)입니다. CSTU 구조는 난류 흐름의 연속적인 시간적 변화와 공간적 동역학을 포착하기 위해 설계되었습니다. 또한, 이 방법은 암묵적 신경 표현(Implicit Neural Representation, INR) 기법을 통합하여 다양한 해상도에서 흐름 데이터를 효과적으로 재구성하는 기능을 제공합니다.

- **Performance Highlights**: 제안된 SR-TR 방법은 두 가지 난류 흐름 데이터 세트, 즉 강제 등방성 난류(FIT) 흐름과 테일러-그린 소용돌이(TGV) 흐름에서 성능 평가를 받았습니다. 이 결과는 시간에 따른 재구성 성능 및 다양한 해상도에 걸친 성능의 일관성을 입증합니다. 각 방법론의 구성 요소의 효과는 정성적 및 정량적으로 보여졌습니다.



### HADES: Hardware Accelerated Decoding for Efficient Speculation in Large Language Models (https://arxiv.org/abs/2412.19925)
Comments:
          Accepted to ICCEA 2025

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 성능과 에너지 효율성을 향상시키는 새로운 접근법인 Hardware Accelerated Decoding (HADES)을 소개합니다. 기존 문헌에서는 다루어지지 않았던 하드웨어 수준에서의 speculative decoding 지원을 포함하여, LLM 가속기(Accelerator)의 설계를 다룹니다.

- **Technical Details**: HADES는 LLM의 연산 효율성을 크게 향상시키기 위해 speculative decoding 기술을 활용합니다. 이 접근법은 LLM의 대규모와 복잡성으로 인한 계산적 도전을 해결하도록 설계되었습니다.

- **Performance Highlights**: 이 연구는 speculative decoding이 LLM의 작동 효율성을 어떻게 개선할 수 있는지를 보여주며, 이로 인해 보다 발전된 실용적 응용 가능성을 제시합니다. HADES는 LLM의 활용 범위를 넓히는데 기여할 것입니다.



### Identifying Cocoa Pollinators: A Deep Learning Datas (https://arxiv.org/abs/2412.19915)
Comments:
          The manuscript introduces the first cocoa pollination dataset and an example analysis with YOLOv8 models

- **What's New**: 이 논문은 코코아 산업에서 수확량 향상을 위한 수분 매개체(pollination-related) 연구가 제한적이라는 점을 강조합니다. 새로운 임베디드 하드웨어(embedded hardware)와 AI 기반 데이터 분석을 통해 코코아 꽃 방문자에 대한 정보를 심화하고 있으며, 이를 위해 다수의 이미지를 포함하는 데이터셋을 최초로 제시합니다.

- **Technical Details**: 이 데이터셋에는 Ceratopogonidae, Formicidae, Aphididae, Araneae, Encyrtidae와 같은 다양한 꽃 방문자 이미지를 포함하여 총 5,792장의 이미지가 수록되어 있습니다. 이 데이터셋은 중국 하이난 성의 코코아 농장에서 두 해 동안 임베디드 카메라를 통해 수집된 2300만 장의 이미지 중에서 선별되었습니다. YOLOv8 모델을 사용하여 학습 세트의 배경 이미지 비율을 조절하면서 최적의 성능을 발휘하는 모델을 찾는 방법을 예시로 보여줍니다.

- **Performance Highlights**: 중형 YOLOv8 모델이 배경 이미지 비율이 8%인 조건에서 최상의 성과를 달성하였습니다(F1 Score 0.71, mAP50 0.70). 또한, 이 데이터셋은 저대비 이미지와 어려운 검출 대상을 가진 이미지에서 딥 러닝(deep learning) 모델 아키텍처의 성능을 비교하는 데 유용합니다. 본 연구는 수분 모니터링 프로젝트를 통한 지속 가능한 코코아 생산을 촉진하기 위한 향후 노력에 기여할 수 있습니다.



### Leveraging Scene Geometry and Depth Information for Robust Image Deraining (https://arxiv.org/abs/2412.19913)
Comments:
          12 pages, 5 figures, 10 tables

- **What's New**: 이 연구는 다중 네트워크를 통합한 새로운 학습 프레임워크를 제안합니다. 기존의 단일 네트워크 접근 방식을 뛰어넘어, 레인 디렛(deraining)과 깊이 정보(depth information)를 동시에 처리하는 두 개의 서포트 네트워크를 포함하여 더 효과적으로 이미지 처리를 수행합니다. 이로 인해 자율주행 차량의 물체 탐지(object detection) 성능을 개선할 수 있습니다.

- **Technical Details**: 이 방법은 Derain AutoEncoder(DerainAE) 모델을 활용하여 레인 아티펙트를 효과적으로 처리합니다. 또한 깊이 네트워크(DepthNet)를 통합하여 이미지 간의 중요한 구조 정보를 유지하며, 레인과 클리어(clear) 이미지 간의 일관된 특징을 잡아냅니다. 두 가지 형태의 감독(supervision)을 통해 모델은 레인을 제거하면서도 장면의 내재적 특징을 보존합니다.

- **Performance Highlights**: 다양한 야외 데이터셋에 대한 광범위한 실험을 통해, 이 방법은 비 오는 날의 아티펙트를 효과적으로 제거하면서도 중요한 이미지 세부사항을 보존하는 성능을 입증했습니다. 특히, 물체 탐지 작업에서 뛰어난 성능을 보였습니다. 이러한 결과는 기상 변화가 심한 환경에서도 영상 기반 자율 주행의 신뢰성과 기능성을 향상시키는 데 기여할 것으로 기대됩니다.



### Evaluate Summarization in Fine-Granularity: Auto Evaluation with LLM (https://arxiv.org/abs/2412.19906)
- **What's New**: 이 논문에서는 정보 소화의 필요성으로 인해 요약의 정확하고 객관적인 평가를 위한 새로운 방법론인 SumAutoEval을 제안합니다. 기존의 ROUGE나 임베딩 유사도 기반 방법들이 인적 평가와 잘 상관되지 않고 비직관적임을 보완하고자 합니다. SumAutoEval은 완전성(Completeness), 정확성(Correctness), 정렬(Alignment), 가독성(Readability) 등 4가지 주요 차원에서 객관적인 점수를 제공합니다.

- **Technical Details**: 제안된 평가 방법은 세 단계로 구성됩니다: 1. 추출(Extraction) - 단문 형태로 노트를 분리하고 각 문장이 고유한 정보를 포함하도록 합니다. 2. 자기 검증(Self-Verification) - 생성된 결과를 검토하고 유사한 개체를 통합합니다. 3. 참고 출처 확인(Reference Sourcing) - 원본 문구를 활용하여 개체를 검증합니다. 이러한 과정은 공정하고 정량적인 방식으로 요약의 품질을 평가할 수 있도록 합니다.

- **Performance Highlights**: SumAutoEval는 요약의 품질 이해를 강화하며, 인적 평가와 더 나은 상관관계를 입증합니다. 이 접근법은 모델이나 프롬프트의 변화에 관계없이 점수의 안정성과 신뢰성을 확보합니다. 따라서, 요약 평가 상황에서 객관적이고 생산적인 결과를 도출할 수 있는 가능성을 제시합니다.



### A Fully Hardware Implemented Accelerator Design in ReRAM Analog Computing without ADCs (https://arxiv.org/abs/2412.19869)
- **What's New**: 이 논문은 ReRAM 기반의 새로운 신경망 가속기 아키텍처인 RACA를 제안합니다. RACA는 Stochastic Binary Neural Networks (SBNNs)를 활용하여 Sigmoid 및 SoftMax 활성화 함수를 하드웨어적으로 구현합니다. 기존의 DAC와 ADC를 제거함으로써 하드웨어 복잡성을 줄이고 에너지 효율성을 높입니다.

- **Technical Details**: 제안된 RACA 아키텍처는 ReRAM 교차 배열을 활용하여 전압 신호 입력을 처리하고, 노이즈 전류를 이용해 Sigmoid 함수의 확률적 이진화 과정을 수행합니다. SoftMax 함수는 Winner-Takes-All(WTA) 규칙을 사용하여 다중 클래스 분류 결과를 결정합니다. 이러한 방법은 비선형 활성화의 필요성을 최소화하여 전체 에너지 소비를 줄입니다.

- **Performance Highlights**: 실험 결과, RACA는 전통적인 아키텍처에 비해 성능 지표에서 두각을 나타내며, 추론 정확도는 손상되지 않았습니다. 제안된 설계는 에너지 소비 및 하드웨어 효율 측면에서 현저한 개선을 보여주며, 대규모 신경망 처리에 적합한 솔루션을 제공합니다.



### Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales (https://arxiv.org/abs/2412.19867)
- **What's New**: 이번 연구에서는 diffusion 모델의 quantization(양자화)을 개선하기 위해 group-wise quantization(그룹 단위 양자화)의 영향을 조사합니다. 기존의 coarser-grained post-training quantization(후속 훈련 양자화) 기법은 품질 손실이 커서 대규모 모델에 적합하지 않았습니다. 이 연구는 Winograd convolution(윈로가드 합성곱)을 활용한 8비트 fully quantized(전량자화된) 모델이 품질 손실을 최소화하며, 기존 기법에 비해 더 나은 성능을 제공할 수 있음을 입증합니다.

- **Technical Details**: 이 연구에서는 group-wise quantization을 통해 diffusion 모델의 가중치 및 활성화를 양자화하여 이미지 생성 품질을 유지하는 데 집중합니다. Winograd 변환 매트릭스의 scale 파라미터만 조정하여 드문 값의 범위를 줄이는 방법을 제안, 이는 데이터 특정 훈련 데이터 없이 수행할 수 있습니다. 우리의 연구는 학습 데이터에 의존하지 않는 일반화 성능을 보장하므로, foundation 모델에 대한 전이 가능한 성능을 제공합니다.

- **Performance Highlights**: Winograd convolutions가 적용된 우리 8비트 fully quantized diffusion 모델은 다른 최첨단 방법들과 비교하여 ImageNet 데이터셋에서 ResNet-34로 수행했을 때 정확성이 2.56% 향상되었습니다. 또한, 최적화된 커널이 표준 합성곱에 비해 31.3%의 실행 성능 개선을 이루어냈으며, 이는 실제 edge나 모바일 장치에서 diffusion 모델을 효과적으로 배포하는 데 기여할 수 있습니다.



### Fusion of Deep Learning and GIS for Advanced Remote Sensing Image Analysis (https://arxiv.org/abs/2412.19856)
- **What's New**: 이번 논문은 원격 감지 이미지 분석을 위한 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 Convolutional Neural Networks (CNNs)와 Long Short-Term Memory (LSTM) 네트워크를 Geographic Information Systems (GIS)와 결합하여 spatial data analysis의 정확성과 효율성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 모델 파라미터를 미세 조정하기 위해 Particle Swarm Optimization (PSO)과 Genetic Algorithms (GA)와 같은 최적화 알고리즘을 구현하였습니다. 이 과정에서 추정 성능 지표가 개선되어, 분류 정확도가 78%에서 92%로 증가하고, 예측 오차는 12%에서 6%로 감소하였습니다. 또한, 모델의 시간적 정확도는 75%에서 88%로 향상되었습니다.

- **Performance Highlights**: 연구 결과, GIS와의 통합은 공간 분석을 풍부하게 하고 지리적 특성 간의 관계를 깊이 이해하는 데 기여하였습니다. 연구 결과는 고급 딥러닝 방법과 GIS, 최적화 전략을 결합하여 원격 감지 응용 프로그램을 크게 발전시킬 수 있음을 보여 줍니다.



### Symbolic Disentangled Representations for Images (https://arxiv.org/abs/2412.19847)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 ArSyD(Architecture for Symbolic Disentanglement)를 제안합니다. ArSyD는 각 생성 요인을 객체 표현과 동일한 차원의 벡터로 표현합니다. 이를 통해 생성 요인 벡터 표현의 중첩으로 객체 표현을 얻으며, 이러한 표현을 기호 분리 표현(symbolic disentangled representation)이라고 합니다.

- **Technical Details**: ArSyD는 Hyperdimensional Computing(HDC) 원리를 기반으로 하며, 여기서 기호는 고차원 벡터인 하이퍼 벡터로 표현됩니다. 분리는 구성적으로 이루어지며, 학습 중 기반 분포에 대한 추가 가정을 두지 않고 약한 감독 방식으로 이미지를 복원하도록 모델을 학습합니다. ArSyD는 Fixed Seed Vectors를 사용하여 객체의 피처 값을 생성하며, Attention 메커니즘을 적용하여 객체 속성의 조작을 가능하게 합니다.

- **Performance Highlights**: ArSyD는 dSprites와 CLEVR 데이터셋에서 학습된 기호 분리 표현을 분석하며, 다양한 차원의 잠재 표현을 사용하여 방법의 비교를 허용하는 새로운 분리 메트릭을 제안합니다. 실험 결과 ArSyD는 여러 객체를 포함한 장면에서도 효과적으로 작동하며, 객체의 속성을 통제 가능한 방식으로 편집할 수 있습니다.



### Unveiling Secrets of Brain Function With Generative Modeling: Motion Perception in Primates & Cortical Network Organization in Mic (https://arxiv.org/abs/2412.19845)
Comments:
          This is my PhD Dissertation, defended on November 3, 2023

- **What's New**: 이번 논문에서는 신경과학의 질문을 해결하기 위한 생성 모델링(generative modeling)의 두 가지 주요 프로젝트가 소개됩니다. 첫 번째 프로젝트는 계층적 변동 오토인코더(hierarchical VAE)를 사용하여 외부 세계의 특징을 신경세포가 인코딩하는 방법을 탐구합니다. 두 번째 프로젝트는 자발적인 뇌 활동(spontaneous brain activity)의 시공간 구조를 조사하며, 이 구조가 뇌 상태를 어떻게 반영하는지를 다룹니다.

- **Technical Details**: 첫 번째 프로젝트에서는 현대의 생성 모델인 변동 오토인코더와 헬름홀츠의 '무의식적 추론으로서의 지각(perception as unconscious inference)'을 결합하여 계층적 VAE 모델을 개발했습니다. 이 모델은 원숭이 시각 피질의 신경세포가 움직임 자극에 반응하는 방식을 모방하는 능력을 테스트하며, 무감독 학습(unsupervised learning) 방식으로 망막 운동 입력의 인과적 요인(causal factors)을 파악합니다. 두 번째 프로젝트에서는 동시에 진행된 fMRI와 Ca2+ 이미징 데이터를 사용하여 마우스 피질의 겹치는 커뮤니티(overlapping communities)를 분석합니다.

- **Performance Highlights**: 첫 번째 프로젝트의 결과는 계층적 VAE가 원숭이 뇌와 유사한 방식으로 움직임을 인식함을 보여줍니다. 또한, 이 모델은 다양한 커뮤니티에 속하는 피질 지역(cortical regions)의 비율 등을 분석하여 fMRI와 Ca2+ 신호로부터 유추된 네트워크 간의 유사성과 차이점을 드러냅니다. 이 연구는 뇌가 세계를 이해하는 데 있어 계층적 추론(hierarchical inference)이 중요함을 시사하며, 향후 연구의 방향성에 대한 기초를 제공합니다.



### A Review of Latent Representation Models in Neuroimaging (https://arxiv.org/abs/2412.19844)
Comments:
          31 pages, 4 figures

- **What's New**: 이 논문은 Neuroimaging 데이터의 복잡성을 관리하기 위해 Autoencoders, Generative Adversarial Networks (GANs), Latent Diffusion Models (LDMs)와 같은 잠재 표현 모델(latent representation models)이 점점 더 활용되고 있음을 강조합니다. 이러한 모델은 고차원 neuroimaging 데이터를 낮은 차원의 잠재 공간(latent space)으로 축소하여 뇌 기능과 관련된 주요 패턴을 식별하는 데 중점을 두고 있습니다. 또한, 이 리뷰는 임상 응용 이야기부터 뇌의 기본 메커니즘 탐색에 이르기까지 이러한 모델의 다면적인 활용을 논의합니다.

- **Technical Details**: Neuroimaging 데이터의 고차원성은 다양한 예측 및 원인 분석을 어렵게 만듭니다. manifold hypothesis는 실제 고차원 데이터가 저차원 매니폴드(manifold) 상에 존재한다고 제안하여 데이터를 효과적으로 해석하는 데 도움이 됩니다. Principal Component Analysis (PCA)와 Multi-Dimensional Scaling (MDS) 같은 전통적인 차원 축소 기술은 선형 가정에 기반하기 때문에 복잡한 비선형 관계를 포착하는 데 한계가 있습니다. 따라서 Deep Neural Networks (DNNs)는 비선형 관계를 잘 모델링할 수 있는 새로운 방법으로 주목받고 있으며, Autoencoders, GANs, LDMs가 중요한 역할을 하고 있습니다.

- **Performance Highlights**: 잠재 생성 모델(latent generative models)은 고차원 데이터의 효율적인 해석에 기여하며, Neuroimaging의 다양한 임상적인 응용 및 기초 연구에 대한 통찰을 제공합니다. Autoencoder(AE)는 입력 데이터를 커다란 잠재 공간에서 인코딩한 후 원래 형태로 재구성하여 차원 축소(feature extraction) 및 잡음 제거(denoising) 작업에 효과적입니다. 이러한 모델들은 데이터의 본질적인 특성 및 변화를 잘 포착하여 robust한 분포(characterization of its distribution)를 가능하게 합니다.



### RoboSignature: Robust Signature and Watermarking on Network Attacks (https://arxiv.org/abs/2412.19834)
- **What's New**: 이번 연구에서는 생성 모델의 워터마킹 기능과 관련된 취약점을 파악하고 이를 해결하기 위한 새로운 알고리즘을 제시합니다. 기존의 Stable Signature 방식은 적절한 워터마크를 숨기기 위해 LDM의 디코더를 세밀하게 조정하지만, 우리는 모델이 의도한 워터마크를 삽입하지 못하게 하는 적대적 파인튜닝 공격의 가능성을 강조합니다. 이 새로운 공격은 모델이 잘못된, 임의의 워터마크를 심는 경우를 발생시켜 많은 기존 방법들의 취약성을 드러냅니다.

- **Technical Details**: 우리는 LDM(잠재적 확산 모델)을 통해 생성된 이미지에 탐지할 수 없는 워터마크를 삽입하는 알고리즘을 제안합니다. WF 서명이라는 방법을 이용해 디코더를 세밀하게 조정하여 모든 생성된 이미지에 고유한 워터마크를 루트(root)하는 방식입니다. 그러나 이 디코더는 네트워크 수준 공격에 노출될 수 있으며, 랜덤 키 생성 기반의 새로운 적대적 공격을 통해 의도된 워터마크 대신 잘못된 워터마크를 삽입하도록 모델을 혼란스럽게 만들 수 있습니다.

- **Performance Highlights**: 제안된 방법은 LDM과 같은 생성 시스템의 취약점을 예방하고 대응하는 데 중요한 임무를 강조합니다. 또한, Tamirisa et al.의 LLM을 위한 접근 방식을 LDM에 맞춰 조정하여 워터마크 삽입 프로세스를 강화하고 있습니다. 이는 실용적인 응용을 위한 모델의 기본 이미지 생성 능력을 유지하면서도 적대적 파인튜닝 공격에 대한 저항력을 높이는 데 기여합니다.



### Multi-atlas Ensemble Graph Neural Network Model For Major Depressive Disorder Detection Using Functional MRI Data (https://arxiv.org/abs/2412.19833)
Comments:
          17 pages, 2 figures, 10 tables

- **What's New**: 이 연구는 앙상블 기반의 그래프 신경망(GNN) 모델을 개발하여 rs-fMRI 이미지를 분석하고 주요 우울 장애(MDD)를 진단하는 데 중요한 특징들을 탐지하는 데 목적을 두고 있습니다. 기존 연구들이 단일 뇌 영역 세분화 템플릿을 사용한 데 반해, 본 연구는 여러 뇌 아틀라스의 특징을 결합하여 뇌의 복잡성을 포착하고 보다 정확한 진단을 목표로 하고 있습니다. 또한, 대규모 다기관 MDD 데이터셋을 통해 모델의 효과를 입증하였습니다.

- **Technical Details**: 모델은 네 개의 GNN으로 구성되며, 각각이 다른 뇌 아틀라스에서 파생된 기능적 연결망(FCN)을 통해 훈련됩니다. 이는 단일 아틀라스 기반 모델에 비해 뇌의 복잡성을 더 잘 포착하고 구별된 특징을 선별할 수 있는 장점을 제공합니다. 또한, 합성 소수 민족 과대표집(SMOTE) 기법을 적용하여 훈련 데이터의 다양성을 높였습니다.

- **Performance Highlights**: 본 연구의 모델은 모든 검증 절차에서 75.80%의 정확도, 88.89%의 민감도, 61.84%의 특이도, 71.29%의 정밀도, 79.12%의 F1-score를 기록하며 우수한 성능을 보였습니다. 이러한 성과는 MDD 진단의 정확성을 향상시키고, 정신 건강 문제 조기 탐지에 기여할 것으로 기대됩니다.



### Back To The Future: A Hybrid Transformer-XGBoost Model for Action-oriented Future-proofing Nowcasting (https://arxiv.org/abs/2412.19832)
- **What's New**: 이 논문은 고전 영화 'Back to the Future'에서 영감을 받아 현재 행동과 미래 결과 간의 관계를 혁신적으로 재구성한 적응형(nowcasting) 접근 방식을 탐구합니다. 이를 통해 미래에 대한 예측 통찰력을 활용하여 현재 조건을 조정하는 방식으로, 기존의 예측 방식에서 벗어난 새로운 모델을 제안합니다.

- **Technical Details**: 제안된 두 단계 모델은 미래를 예측하는 Transformers의 예측 능력과 해석 가능성과 효율성이 높은 XGBoost의 의사 결정 능력을 통합합니다. 이러한 조합은 미래 예측과 현재 적응의 연속성을 제공하여 더욱 향상된 결과를 도출할 수 있도록 합니다.

- **Performance Highlights**: 기상 데이터셋을 사용한 실험을 통해 본 프레임워크가 보다 정확한 예측을 달성하고, 실시간 적용을 위한 실행 가능한 개입을 안내하는 데 있어 장점을 입증했습니다. 이 연구는 전통적인 예측 방법에 비해 유의미한 성과를 보이며, 실용성을 강조합니다.



### A Unified Framework for Context-Aware IoT Management and State-of-the-Art IoT Traffic Anomaly Detection (https://arxiv.org/abs/2412.19830)
- **What's New**: 본 논문은 IoT 관리 작업을 위한 통합 프레임워크를 제안합니다. 이 프레임워크는 컨텍스트 기반 대형 언어 모델(context-driven large language models, LLMs)과 미세 조정된 이상 탐지 모듈(anomaly detection module)을 조합하여 IoT 시스템에서의 장치 관리, 문제 해결 및 보안 강화를 지원합니다. 아울러, 주목할 만한 성과는 LLM을 기반으로 한 응답의 정확성과 신뢰성을 높이기 위해 관련된 컨텍스트 정보를 통합하는 것입니다.

- **Technical Details**: 이 프레임워크는 RAG( retrieval-augmented generation)으로 향상된 LLM과 미세 조정된 BERT 모델을 결합하여 IoT 트래픽에서의 이상 탐지 및 문서 기반 질문 응답을 수행합니다. 시스템 디자인은 두 가지 단계인 인덱싱 및 쿼리 처리로 나뉘며, 각 단계는 도메인 특화된 응답을 제공하고 실시간으로 사이버 위협 및 이상을 탐지할 수 있게 합니다.

- **Performance Highlights**: 프레임워크는 Edge-IIoTset 데이터셋에서 99.87%의 정확도를 달성한 BERT 모델을 통합하여 IoT 네트워크의 보안 및 성능을 강화합니다. 또한, RAG를 통한 컨텍스트 증강은 BLEU, ROUGE, METEOR, BERTScore 지표에서 성능 상승을 확인하며, 이는 IoT 관리 작업에서 LLM의 응답 정확성과 관련성을 중대하게 향상시키는 결과를 보여줍니다.



### AnalogXpert: Automating Analog Topology Synthesis by Incorporating Circuit Design Expertise into Large Language Models (https://arxiv.org/abs/2412.19824)
- **What's New**: 이 논문에서는 AnalogXpert라는 새로운 LLM 기반 에이전트를 제안하여 실제 아날로그 회로의 토폴로지 합성 문제를 해결하고자 합니다. 기존 연구들은 애매한 설계 요구사항을 입력으로 삼아 이상적인 모델을 출력하는 반면, 이 연구는 보다 구체적인 구조적 요구사항과 장치 수준 모델을 중요시합니다. 

- **Technical Details**: 연구팀은 아날로그 토폴로지를 SPICE 코드로 표현하고, 차단 라이브러리를 통해 설계 공간을 줄이는 방안을 도입했습니다. 문제를 블록 선택과 블록 연결의 두 개의 하위 작업으로 나누어 CoT(Chain of Thought) 및 인컨텍스트 학습 기법을 사용하여 실제 설계 프로세스를 모방합니다. 또한, 초기 설계에서의 오류를 점진적으로 수정하는 교정 전략을 소개하여 인간 디자이너의 iterative(반복적) 확인 과정을 반영합니다.

- **Performance Highlights**: AnalogXpert는 합성 데이터셋에서 40%, 실제 데이터셋에서 23%의 성공률을 달성하여, GPT-4o의 3%와 비교할 때 현저히 높은 성능을 보입니다. 이 연구는 실질적으로 아날로그 회로 설계의 복잡성을 해결할 수 있는 중요한 초석이 될 것으로 기대됩니다.



### A Survey on Large Language Models for Communication, Network, and Service Management: Application Insights, Challenges, and Future Directions (https://arxiv.org/abs/2412.19823)
- **What's New**: 최근 통신 네트워크의 진화는 Network and Service Management (NSM) 전략의 필요성을 더욱 강조하고 있습니다. 이 논문은 다양한 통신 네트워크 분야에서 Large Language Models (LLMs)의 통합을 조사하며, LLMs가 NSM 작업을 자동화할 수 있는 잠재력을 다룹니다. 기존의 단일 네트워크 도메인에 대한 연구와는 달리, 이 연구는 모바일 네트워크, 차량 네트워크, 클라우드 기반 네트워크 등 여러 도메인에 걸쳐 LLM의 활용을 탐구합니다.

- **Technical Details**: 이 논문은 LLM의 기초 지식을 제공하고, transformer 아키텍처, 일반 목적 및 도메인 특화 LLM, 모델의 사전 학습(pre-training) 및 미세 조정(fine-tuning) 과정을 자세히 설명합니다. NSM 작업에서 LLM의 다양한 응용 프로그램을 새롭게 정의한 분류 체계에 따라 분류하며, 네트워크 모니터링과 보고, AI 기반 네트워크 계획 등 다양한 관리 작업을 포괄합니다. 이러한 연구는 NSM 작업에 대한 기존 문헌을 체계적으로 검토하고, 기술적인 디테일을 보강합니다.

- **Performance Highlights**: 이 연구는 LLM이 네트워크 운영 자동화, 이상 탐지, 예측 유지보수 등의 작업에서 성능을 향상시킬 수 있음을 강조합니다. 예를 들어, Mekrache et al.의 연구는 LLM을 통해 인간과 기계 간의 원활한 통신을 지원하고, ConnectGPT는 센서 데이터를 분석하여 교통 관리의 효율성을 높이며 안전 메시지를 자동 생성하는 방안을 제안합니다. 그러나 LLM의 NSM 적용 및 커스터마이징은 아직 초기 단계에 있으며, 이 연구는 이러한 분야에 대한 포괄적인 탐색의 필요성을 제시합니다.



### Nanoscaling Floating-Point (NxFP): NanoMantissa, Adaptive Microexponents, and Code Recycling for Direct-Cast Compression of Large Language Models (https://arxiv.org/abs/2412.19821)
Comments:
          12 pages, 12 figures

- **What's New**: 이 논문에서는 최신 대형 언어 모델(LLMs)의 성능을 향상시키기 위해 새로운 데이터 포맷인 Nanoscaling Floating-Point (NxFP)를 제안합니다. NxFP는 Micro Scaling (MxFP)에서 발생하는 문제에 대한 해결책을 제시하며, 특히 6 비트 이하에서의 생성을 개선합니다. 이 연구는 MxFP의 성능 저하 문제를 해결하고 더 나은 정확도와 메모리 감축을 목표로 합니다.

- **Technical Details**: Nanoscaling Floating-Point (NxFP)는 NanoMantissa, Adaptive Microexponent, Code Recycling의 세 가지 기술을 포함합니다. 이 기술들은 각각 큰 값의 정확한 근사를 돕고, 적절한 포맷을 제공하며, 코드 낭비를 최소화하는 역할을 수행합니다. NxFP는 기존의 MxFP에 비해 향상된 정확도와 메모리 발자국을 달성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, NxFP는 다양한 현대 LLMs에서 Perplexity를 최대 0.64까지 개선하고 MMLU 벤치마크에서 정확도를 최대 30% 향상시켰습니다. 또한 NxFP는 MxFP와 비슷한 perplexity를 유지하면서도 메모리 발자국을 최대 16%까지 줄일 수 있음을 보여주었습니다. 이는 NxFP가 실제 LLMs의 성능에서 더 나은 결과를 낼 수 있음을 나타냅니다.



### GaLore$+$: Boosting Low-Rank Adaptation for LLMs with Cross-Head Projection (https://arxiv.org/abs/2412.19820)
- **What's New**: 최근의 저랭크 훈련 방법인 GaLore는 대형 언어 모델(LLM)의 최적화에 필요한 메모리를 상당히 줄여주었습니다. 그러나 GaLore는 특이값 분해(SVD) 단계에서 전체 훈련 시간의 80% 이상을 소비하는 한계가 있습니다. 이를 해결하기 위해 본 논문에서는 다중 헤드 주의력을 위한 크로스 헤드 저랭크 프로젝션을 사용하는 GaLore$+$를 제안합니다.

- **Technical Details**: GaLore$+$는 크로스 헤드 저랭크 프로젝션을 활용하여 시간을 절약하며 신속한 SVD 추정을 달성합니다. 이 방법은 다중 주의력 헤드의 쿼리 또는 키 변환의 프로젝션 행렬을 공유하여 저랭크 프로젝션의 계산 복잡성을 O(h³)에서 O(h)로 줄입니다. 또한, 스파스 코딩된 잔여값을 적용함으로써 저랭크 근사가 초래하는 오류를 감소시킵니다.

- **Performance Highlights**: 실험 결과, GaLore$+$는 산술 추론 및 자연어 생성 데이터셋에서 기존의 최첨단 저랭크 적응 방법들보다 우수한 성능을 보여줍니다. GaLore$+$는 기존의 GaLore에 비해 약 4배의 빠른 미세 조정 속도를 달성하였습니다. 이러한 성과는 언어 모델의 훈련 비용을 크게 절감할 수 있는 가능성을 제시합니다.



### ChipAlign: Instruction Alignment in Large Language Models for Chip Design via Geodesic Interpolation (https://arxiv.org/abs/2412.19819)
- **What's New**: 최근 대규모 언어 모델(LLMs)에서의 발전은 칩 설계와 같은 다양한 영역으로의 응용을 확장시켰습니다. 새로운 모델 ChipAlign은 훈련이 필요 없는 모델 병합(strategy) 기법을 활용하여 일반 지침 정렬 LLM과 칩 전용 LLM의 장점을 결합하여 지침 정렬 능력을 개선합니다.

- **Technical Details**: ChipAlign은 두 개의 서로 다른 LLM을 Riemannian manifold(리만 다양체) 상의 두 점으로 보고, geodesic interpolation(측지선 보간법)을 사용하여 효과적으로 가중치를 병합합니다. 이 과정에서 ChipAlign은 입력 모델의 강력한 지침 정렬 및 칩 전문 지식을 유지한 통합 모델을 생성합니다.

- **Performance Highlights**: ChipAlign의 결과는 기존 칩 LLM의 지침 추적 능력을 26.6% 개선한 것으로 나타났으며, OpenROAD QA 벤치마크에서는 3.9%, 생산 수준의 칩 QA 벤치마크에서는 8.25%의 성능 향상을 달성했습니다. 이는 ChipNeMo와 같은 최첨단 모델에 비해 상당한 개선을 보여줍니다.



### Predicting Human Brain States with Transformer (https://arxiv.org/abs/2412.19814)
Comments:
          11 pages, 4 figures, MICCAI MMMI workshop in press

- **What's New**: 이번 연구에서는 fMRI (functional magnetic resonance imaging) 데이터를 이용하여 뇌의 resting states를 예측할 수 있는 가능성을 조사하였습니다. 특히, self-attention과 transformer 아키텍처의 성공적 적용을 바탕으로 하여, 대규모의 고품질 fMRI 데이터를 활용하고 있습니다. 이러한 접근 방식은 뇌의 기능적 메커니즘을 이해하는 데 중요한 진전을 이룰 수 있습니다.

- **Technical Details**: 이 논문에서는 HCP (human connectome project)로부터 수집된 fMRI 데이터를 사용하여, 21.6초 동안의 뇌 상태를 기반으로 5.04초의 예측을 수행했습니다. 예측 과정은 self-attention 기법을 활용하였으며, 이로 인해 모델이 뇌의 기능적 연결 구조(functional connectome)를 반영하는 fMRI 상태를 생성할 수 있음을 보여주었습니다. 이러한 예측 모델은 뇌 상태의 동적 변화를 이해하는 데 기여할 수 있습니다.

- **Performance Highlights**: 모델은 예측 시간이 길어질수록 오류가 누적되는 경향이 있지만, 생성된 fMRI 뇌 상태는 여전히 유의미한 결과를 보입니다. 초기 결과들이 뇌의 기능적 조직 구성(functional organization)을 학습하는 generative 모델의 가능성을 보여주며, 향후 연구에 큰 기대를 모으고 있습니다.



### LINKs: Large Language Model Integrated Management for 6G Empowered Digital Twin NetworKs (https://arxiv.org/abs/2412.19811)
Comments:
          Accepted by The 2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)

- **What's New**: 이 논문에서는 디지털 트윈(DT)과 6G 네트워크 관리에 있어 대형 언어 모델(LLM)의 통합을 통해 네트워크 관리의 새로운 접근 방식을 제시합니다. LINKs라는 프레임워크를 통해 LLM을 활용하여 데이터 검색 및 통신 효율성을 최적화하고, 자동화된 방식으로 자원 관리(RRM)를 수행합니다. 개념적으로, 이 프레임워크는 레이지 로딩 전략을 통해 전송 지연을 최소화하도록 설계되었습니다.

- **Technical Details**: 이 시스템 모델은 IoT 센서로부터 데이터를 수집하고 LLM을 사용하여 중앙 집중식 서버에서 관리하도록 설계되었습니다. 6G 네트워크에서는 소프트웨어 정의 네트워킹(SDN) 기술을 활용하여 센서와 중앙 노드 간의 통신을 관리하며, 센서들은 데이터 전송이 요구될 때까지 로컬에 데이터를 저장합니다. 이러한 구조에서는 전송 지연을 최소화하며, 최적의 RRM을 위한 수치 최적화 문제로 데이터 검색 작업이 변환됩니다.

- **Performance Highlights**: 시뮬레이션 결과는 6G 강화를 통해 DT와 LLM 기술의 통합이 데이터 계획 및 네트워크 관리 성능을 향상시킬 수 있음을 보여줍니다. 이 연구는 LLM이 IoT 환경에서의 복잡한 통신 시나리오를 효율적으로 처리하는 능력을 입증하며, 네트워크 자원 관리의 새로운 자동화 전략을 강조합니다. 전체적으로 LLM의 도입은 더욱 지능적이고 자동화된 네트워크 관리를 위한 새로운 가능성을 열어줍니다.



### AI-driven Automation as a Pre-condition for Eudaimonia (https://arxiv.org/abs/2412.19808)
- **What's New**: 이 논문은 '일의 미래'에 대한 논의를 인간의 번영(eudaimonia) 관점에서 재조명합니다. 자동화가 일자리 손실에 대한 위협이 아니라 오히려 인간의 번영을 촉진할 수 있다는 점을 강조하고 있습니다. 이는 고전 철학의 새로운 해석을 바탕으로 하여, AI에 의해 주도되는 자동화의 긍정적인 측면에 주목하는 접근법입니다.

- **Technical Details**: 연구는 니체 아리스토텔레스 철학을 활용하며, AI의 도입이 어떻게 인간의 여가 참여를 증가시킬 수 있는지를 분석합니다. 또한, 연구는 미덕 법학(virtue jurisprudence)의 개념을 통해 이러한 변화가 현재의 법적 체계에 어떤 함의를 가질 수 있는지를 탐구합니다.

- **Performance Highlights**: 종합적으로, 이 연구는 자동화가 무조건적인 위협이 아닐 뿐만 아니라 인간의 가치 있는 활동을 새로운 차원으로 끌어올릴 수 있는 기회를 제공한다고 주장합니다. 이 관점에서, 자동화는 인간 삶의 질을 향상시키고 법적 제도에도 긍정적인 변화를 일으킬 수 있음을 제시합니다.



### exLong: Generating Exceptional Behavior Tests with Large Language Models (https://arxiv.org/abs/2405.14619)
Comments:
          ICSE 2025 (camera ready)

- **What's New**: 이번 논문에서는 exLong이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 프로그래밍 언어에서 예외적인 동작 검사(EBT)를 자동으로 생성하는 기능을 가지고 있습니다. exLong은 CodeLlama를 기반으로 하여, 예외처리 논리를 이해하도록 특수하게 조정된 대형 언어 모델입니다. 또한, 이 논문은 개발자들이 일반적으로 '행복한 경로(happy paths)'에 집중하는 경향이 있다는 점에 주목하며 이러한 간극을 해소하기 위한 솔루션을 제공합니다.

- **Technical Details**: exLong은 메서드 호출과 관람 조건, 비예외 동작 테스트(non-EBT)와 같은 컨텍스트를 포함하여, 특정 throw 문을 유도하는 추적(trace)을 분석합니다. 이 모델은 두 가지 사용 사례를 평가하여, 첫째는 개발자가 특정 메서드와 throw 문을 선택하여 생성하는 EBT의 경우이고, 둘째는 전체 코드베이스에 대하여 EBT를 자동으로 생성하는 경우입니다. EBT의 생성 과정에서 exLong은 대규모 언어 모델에서 효과적인 테스트 생성이 가능함을 보여주었습니다.

- **Performance Highlights**: exLong은 기존의 모델(CAT-LM 및 GPT-3.5)과 비교했을 때, 각각 83.8% 및 9.9% 더 많은 실행 가능한 EBT를 생성하는 데 성공했습니다. 또한, Randoop 및 EvoSuite와 같은 분석 기반 도구들과 비교했을 때도 더 우수한 성능을 보였습니다. exLong이 자동 생성한 23개의 EBT는 여러 오픈 소스 프로젝트에도 기여하였으며, 이는 exLong의 실제적 유용성을 강조합니다.



### LLM-assisted Vector Similarity Search (https://arxiv.org/abs/2412.18819)
- **What's New**: 본 논문에서는 데이터 검색 요구가 점점 복잡해짐에 따라 전통적인 검색 방법이 nuance (뉘앙스)와 conceptual (개념적) 쿼리를 처리하는 데 한계를 보임을 설명합니다. 이를 해결하기 위해, vector similarity search (벡터 유사도 검색)와 Large Language Models (LLMs)을 결합한 하이브리드 접근 방식을 제안합니다. 이 두 단계의 솔루션은 첫 번째 단계에서 벡터 유사도 검색을 활용하여 잠재적인 일치를 선별하고, 두 번째 단계에서 LLM을 사용하여 결과의 맥락 인식 순위를 매깁니다.

- **Technical Details**: 제안된 접근 방식은 먼저 벡터 유사도 검색을 통해 기본 쿼리를 처리하고, 이후 LLM을 통해 복잡한 쿼리에서 발생할 수 있는 constraints (제약 조건), negations (부정) 및 conceptual requirements (개념적 요구)를 고려하여 결과를 재순위합니다. 실험 결과, 벡터 유사도 검색은 간단한 쿼리에 효과적이지만, LLM을 활용한 방법은 더욱 복잡한 쿼리에서도 뛰어난 성과를 나타냅니다.

- **Performance Highlights**: 자연어 이해 능력을 활용하여 복잡한 작업의 검색 결과 정확성을 향상시키면서도 효율성을 유지하는 것이 이 방법의 큰 장점입니다. 논문에서는 실제 응용 사례를 논의하며, 다양한 데이터셋과 사용 사례를 위한 이 기술을 개선하고 확장하기 위한 향후 연구 방향도 제안합니다.



### From Interests to Insights: An LLM Approach to Course Recommendations Using Natural Language Queries (https://arxiv.org/abs/2412.19312)
Comments:
          17 pages, 9 figures

- **What's New**: 이 연구에서는 미국 대학의 학생들에게 더 나은 강의 추천을 제공하기 위해 대규모 언어 모델(LLM) 기반의 혁신적인 강의 추천 시스템을 탐구합니다. 이 시스템은 사용자 질문에 기반하여 '이상적인' 강의 설명을 생성하고, 결과를 벡터로 변환해 실제 강의를 찾는 방식으로 작동합니다. 강의 선택 과정에서의 불평등 해소를 목표로 하며, 캠퍼스 내에서의 파일럿 시스템을 배포할 계획도 포함되어 있습니다.

- **Technical Details**: 이 시스템은 Retrieval Augmented Generation (RAG) 방법을 적용하여 강의 설명의 데이터 집합에서 관련 정보를 검색합니다. RAG는 정보 검색과 생성형 AI를 결합한 접근법으로, 정보를 식별하는 리트리버(retriever)와 정보를 처리하는 생성기(generator)로 구성됩니다. 이 시스템은 상당한 대화적 요소를 포함하여 사용자와의 상호작용을 강화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 연구는 기존의 추천 시스템들이 직면한 여러 제한 사항을 해결하고, 강의 추천의 품질과 공정성을 평가하는 초기 테스트 결과를 포함합니다. 또한, 다양한 예시 프롬프트에서의 반응을 분석하여 이 시스템이 제공할 수 있는 개선된 사용자 경험을 보여줍니다. 아틀라스 플랫폼을 통한 파일럿 서비스의 구현을 통해 실제 학생들에게 이점을 제공하고자 합니다.



### Scaling Capability in Token Space: An Analysis of Large Vision Language Mod (https://arxiv.org/abs/2412.18387)
- **What's New**: 본 연구는 비전 언어 모델에서 비전 토큰의 수와 성능 간의 관계를 분석하여 새로운 접근 방식을 제시합니다. 이전의 연구와 달리, 비전 토큰에 대한 스케일링 성능이 비슷하게 유지되며, 사용자의 질문과 비전 토큰을 융합하는 메커니즘이 성능 개선에 기여하는 것을 입증합니다. 이를 통해 비전 언어 모델의 효율성을 높이는 방법을 제시하며, 다양한 벤치마크를 통해 검증된 결과를 공유합니다.

- **Technical Details**: 비전 토큰 수 (N_l)에 따른 성능은 S(N_l) ≈ (c/N_l)^{α}와 같은 관계로 정의되며, 여기서 c는 작업 관련 스칼라, α는 성능 변화 비율을 나타냅니다. 연구에서는 여러 비전 언어 모델들을 대상으로 이론 분석과 실험적 평가를 진행하였으며, 질문 융합 (fusion mechanism)의 효과를 탐구했습니다. 비전 언어 모델에서 고해상도 이미지를 효율적으로 처리하기 위한 다양한 토큰화 전략도 논의됩니다.

- **Performance Highlights**: 연구 결과, 비전 언어 모델의 성능은 비전 토큰 수의 증가에 비례하여 확장 가능한 경향을 보이며, 융합된 질문은 특히 작업 특화된 경우 성능을 향상시킵니다. 다양한 작업과 도메인에서 15개 이상의 벤치마크를 통하여 이러한 효과가 유효함을 입증하였습니다. 따라서, 비전 토큰의 수와 질이 모델의 전반적인 성능에 미치는 영향을 강조하며, 효율성과 성능 간의 균형이 중요하다는 점을 강조합니다.




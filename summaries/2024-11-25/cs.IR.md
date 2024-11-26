New uploads on arXiv(cs.CL)

### Measuring Bullshit in the Language Games played by ChatGP (https://arxiv.org/abs/2411.15129)
- **What's New**: 이번 논문은 생성적 대형 언어 모델(Generative Large Language Models, LLMs)이 진리 값과 직접적인 연관 없이 텍스트를 생성하는 방식을 분석합니다. 저자들은 이 현상이 어떻게 발생했는지, 그리고 어떻게 분석할 수 있는지를 고찰하며, LLM 기반의 챗봇이 'bullshit의 언어 게임(Language Game of Bullshit)'에서 어떤 역할을 하는지를 제시합니다.

- **Technical Details**: 연구는 통계적 텍스트 분석(statistical text analysis)을 사용하여 1,000개의 과학 출처와 ChatGPT가 생성한 전형적인 사이비 과학 텍스트(pseudo-scientific text)를 대조하는 데이터셋을 기반으로 진행됩니다. 이후 저자들은 조지 오웰의 정치와 언어에 대한 비판 및 데이비드 그레이버의 사이비 직업(bullshit jobs)에 대한 묘사에서 같은 언어적 특징이 발견되는지 탐구합니다. 간단한 가설 검정(hypothesis-testing) 방법을 사용하여 ChatGPT의 인위적인 bullshit 언어와 자연 언어에서 관찰되는 정치적 및 직장 내 bullshit의 기능 간의 관계를 설명합니다.

- **Performance Highlights**: 이 연구에서는 ChatGPT의 생성적 언어 구조가 정책 및 직장 내 부조리에 걸쳐 어떻게 작용하는지를 통계 모델을 통해 신뢰성 있게 연결합니다. 이를 통해 생성적 언어 모델이 사회적 비효율성의 맥락에서도 유의미한 언어적 패턴을 드러낼 수 있다는 점을 보여줍니다.



### T\"ULU 3: Pushing Frontiers in Open Language Model Post-Training (https://arxiv.org/abs/2411.15124)
- **What's New**: TÜLU 3는 완전 개방형(AI) 모델로, 포스트 트레이닝(post-training) 기술을 적용하여 최신 언어 모델의 기능을 향상시키고 새로운 기술을 발굴합니다. 이 모델은 Llama 3.1을 기반으로 하여, Llama 3.1, Qwen 2.5, Mistral 및 비공식 모델인 GPT-4o-mini 및 Claude 3.5-Haiku를 초월하는 성능을 나타냅니다. TÜLU 3의 기술은 Open 소스 데이터와 검증 가능한 보상을 활용한 강화 학습(RL) 등의 새로운 방법론을 포함하고 있습니다.

- **Technical Details**: TÜLU 3 모델은 SFT(감독된 미세 조정), DPO(선호 최적화), RLVR(검증 가능한 보상으로 학습하는 강화 학습)이라는 세 가지 훈련 알고리즘을 포함합니다. 이 모델은 명확한 성능 목표를 설정하고 모델 개선을 이끌기 위한 평가 프레임워크를 통해 평가됐습니다. 또한, 단계적으로 훈련이 진행되며, 다양한 스킬을 표적하는 새로운 데이터셋이 도입되어 훈련 과정에서 철저한 데이터 정화가 이루어졌습니다.

- **Performance Highlights**: TÜLU 3는 기존의 개방형 모델 대비 우수한 성능을 보여줍니다. 특히, 같은 규모의 Llama 3.1 Instruct 및 Mistral-Instruct 등과 비교할 때 높은 성능을 나타내며, 70B 규모에서는 Claude 3.5 Haiku 및 GPT-4o mini와 일치하는 성능을 보입니다. 이러한 성과는 데이터 믹스, 메소드 및 파라미터 조정을 최적화한 결과로 판단됩니다.



### XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (https://arxiv.org/abs/2411.15100)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)용으로 설계된 유연하고 효율적인 구조 생성 엔진인 XGrammar를 제안합니다. XGrammar는 context-free grammar(CFG)의 실행을 가속화하기 위해 어휘(vocabulary)를 context-independent tokens와 context-dependent tokens로 나누어 처리합니다. 이 접근 방식은 구조 생성의 오버헤드를 최소화하고 대형 언어 모델 추론 작업과 효과적으로 오버랩되도록 설계되었습니다.

- **Technical Details**: XGrammar는 byte-level pushdown automaton을 사용하여 CFG를 해석합니다. 이 설계는 각 문자 가장자리에 여러 바이트를 포함할 수 있도록 하여 비정규 토큰 경계를 처리하고 서브 UTF-8 문자가 포함된 토큰을 지원합니다. 또한, 적응형 토큰 마스크 캐시를 생성하여 대부분의 마스크를 미리 생성하고, persistent execution stack을 사용하여 context-dependent tokens를 효율적으로 처리함으로써 성능을 향상시킵니다.

- **Performance Highlights**: XGrammar는 기존의 방법보다 최대 100배의 성능 향상을 달성할 수 있습니다. Llama-3.1 모델에 통합된 XGrammar 기반 LLM 서빙 엔진은 H100 GPU에서 구조적 출력에 대해 최대 80배의 속도 향상을 이룰 수 있습니다. 이를 통해 구조 생성을 위한 거의 제로 오버헤드를 달성할 수 있으며, XGrammar는 오픈 소스로 제공될 예정입니다.



### Locating the Leading Edge of Cultural Chang (https://arxiv.org/abs/2411.15068)
Comments:
          Accepted CHR 2024

- **What's New**: 최근의 연구들은 문화적 변화의 연구에 있어 텍스트 유사성과 차별성을 측정하는 방법의 중요성을 강조하고 있습니다. 본 연구에서는 텍스트의 세 가지 다른 표현 방식인 topic models, document embeddings, word-level perplexity를 세 가지 서로 다른 데이터셋에 적용해 사회적 증거와의 일치를 평가하였습니다. 흥미롭게도, 저명한 저자와 젊은 저자들의 작품은 텍스트적으로 시대를 앞서가는 경향이 있음을 발견했습니다.

- **Technical Details**: 본 연구에서는 언어학적 및 경제학 분야에서의 저널 기사와 1890년부터 2000년까지의 픽션 자료를 포함한 다양한 데이터셋을 사용하여 텍스트의 혁신 효과를 모델링하였습니다. LDA를 활용한 topic modeling과 더불어, 문서의 전반적인 혁신성을 평가하기 위해 문서 조각의 'precocity'를 계산하였습니다. 세 가지 다른 텍스트 표현 방식을 통해 텍스트의 과거 및 미래와의 관계를 분석하였으며, 특히 가장 혁신적인 부분을 강조하여 새로운 통찰력을 제공했습니다.

- **Performance Highlights**: 결과적으로, 텍스트의 사회적 영향력은 전체적인 혁신 수준보다도 가장 전향적인 순간에 더 의존할 수 있다는 결론을 도출했습니다. 높은 precocity를 가진 텍스트는 동년의 동료들에 비해 상대적으로 더 나은 점수를 기록하는 경향이 있었습니다. 이 연구는 고립된 사회적 증거가 아닌 다양한 사회적 변수와의 일치를 통한 텍스트 분석 방법의 타당성을 제시하며 문화적 변화에 대한 더 깊은 이해를 가능하게 합니다.



### Fantastic Biases (What are They) and Where to Find Them (https://arxiv.org/abs/2411.15051)
Comments:
          Publication in Spanish in the Journal Bits de Ciencias: this https URL

- **What's New**: 이 논문은 인공지능(AI) 모델의 공정성(Fairness) 문제를 탐구합니다. 특히, 알고리즘이 성별, 인종 및 계급 정보에 바탕을 두는 경향이 있으며, 이로 인해 불공정한 결과를 초래할 수 있다는 점을 강조합니다. 바이어스(Bias)의 정의와 그 사회적 영향에 대한 통찰력을 제공하며, 공정한 AI 시스템을 구축하기 위한 접근 방법을 논의합니다.

- **Technical Details**: AI 모델의 공정성을 유지하는 것은 복잡한 과제로, ML 모델은 인종, 성별 등과 관련된 실제 세계 데이터에서 학습합니다. 이러한 데이터의 바이어스는 모델이 선별적으로 학습하고 이를 증폭시킬 수 있게 합니다. 공정성을 정의하는 기준은 집단의 경험, 문화, 역사, 정치적 요소 등을 고려해야 하며, 상충되는 가치관이 존재할 수 있음을 나타냅니다.

- **Performance Highlights**: AI 모델의 성능은 지속적으로 향상되고 있으며, 보다 정교한 결정이 요구됩니다. 그러나 비록 완전한 공정성을 이루는 것은 어렵지만 기존 모델의 바이어스를 제거하는 방법을 통해 보다 공정한 시스템으로 발전할 수 있는 가능성을 제시합니다. 최종적으로, 알고리즘의 투명성을 높이고 사용자에게 공정함을 보장하는 것이 필수적입니다.



### ScribeAgent: Towards Specialized Web Agents Using Production-Scale Workflow Data (https://arxiv.org/abs/2411.15004)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 에이전트들이 복잡한 웹 작업을 처리하는 방안을 제시합니다. 연구팀은 250개 이상의 도메인에서 수집한 60억 개의 토큰을 포함한 생산 규모의 워크플로우 데이터를 이용하여 오픈 소스 LLM을 미세조정하는 대안을 탐구하고 있습니다. 이를 통해 ScribeAgent라는 새로운 웹 에이전트를 개발하였으며, 기존 에이전트 대비 더 나은 성능을 입증했습니다.

- **Technical Details**: ScribeAgent는 HTML-DOM과 행동 이력을 기반으로 다음 행동을 직접 생성할 수 있는 첫 번째 단일 단계 LLM 에이전트입니다. 이는 이전의 다단계 에이전트와는 달리, 여러 단계를 거치지 않고도 작업을 수행할 수 있도록 설계되었습니다. 연구팀은 LoRA (Low-Rank Adaptation) 기법을 사용하여 오픈 소스 LLM을 미세 조정하였으며, 이 과정에서 60억 개 이상의 토큰으로 구성된 훈련 데이터를 활용했습니다.

- **Performance Highlights**: ScribeAgent는 Mind2Web 벤치마크에서 최첨단의 직접 생성 성능을 달성하고, WebArena에서는 작업 성공률을 37.2%에서 51.3%로 증가시켜 최고 성과를 나타냅니다. 특히, 32B 매개변수를 가진 ScribeAgent-Large는 모든 테스트 세트에서 기본 성능보다 5-10% 향상된 단계 성공률을 보였습니다. 이러한 결과는 LLM의 성능을 높이기 위한 대규모 고품질 실세계 데이터의 중요성을 강조합니다.



### SwissADT: An Audio Description Translation System for Swiss Languages (https://arxiv.org/abs/2411.14967)
- **What's New**: 이번 연구에서는 SwissADT라는 다국어 및 다중 모달 오디오 설명 번역(ADT) 시스템을 소개합니다. 이는 독일어, 프랑스어, 이탈리아어 및 영어를 지원하며, 시각 장애인 및 시각적 제한이 있는 사용자들이 접근할 수 있도록 설계되었습니다. SwissADT는 비디오 클립을 활용하여 시각적 정보를 포함한 AD 스크립트를 자동으로 번역함으로써 다국어 인구에 대한 정보 접근성을 향상시키고자 합니다.

- **Technical Details**: SwissADT는 비디오 클립과 텍스트 입력을 결합하여 AD 스크립트를 번역하는 LLM 기반 시스템입니다. 이 시스템은 AD 스크립트에 가장 적합한 비디오 순간을 식별하기 위해 CG-DETR라는 비디오 시간 그라우더를 사용합니다. 최종적으로 AD 번역기를 위해 비디오 프레임을 샘플링하고, GPT-4 모델을 활용하여 다국어 번역을 수행합니다.

- **Performance Highlights**: SwissADT의 성능은 자동 및 인간 평가를 통해 검증되었습니다. 실험 결과, SwissADT는 ADT 작업에서 유망한 성능을 보여주었으며, AD 전문가의 의견을 토대로 시각 정보를 입력에 포함하는 것이 결과에 긍정적인 영향을 미친 것으로 나타났습니다. 이 연구는 향후 다국어 ADT 시스템의 발전 방향에 중요한 기초 자료를 제공합니다.



### LLM for Barcodes: Generating Diverse Synthetic Data for Identity Documents (https://arxiv.org/abs/2411.14962)
Comments:
          5 pages, 1 figures

- **What's New**: 이 논문에서는 기존의 데이터 생성 방식의 한계를 극복하기 위해 LLM(대형 언어 모델)을 활용한 새로운 합성 데이터 생성 접근법을 제안합니다. 기존의 Faker와 같은 도구는 미리 정의된 템플릿에 의존하여 실제 문서의 다양성과 복잡성을 포착하지 못하는 문제를 보였습니다. 이 방법을 통해 다양한 홍보 요소와 문화적 차이를 반영한 문서 데이터를 생성함으로써 보다 현실적인 데이터셋을 구축할 수 있게 됩니다. 또한, 민감한 개인 정보는 포함되지 않으므로 개인정보 보호에도 적합한 솔루션을 제공하게 됩니다.

- **Technical Details**: 우리의 접근법은 LLM을 사용하여 Driver’s Licenses, Insurance Cards, 및 University IDs와 같은 다양한 신원 문서에 필요한 데이터를 생성하는 과정을 포함합니다. 각 문서 유형은 그 목적에 맞는 특정 정보를 포함하도록 설계되었습니다. 예를 들어, Driver’s Licenses는 소지자의 이름, 주소, 라이센스 번호와 같은 세부정보를 포함하고, Insurance Cards는 정책 번호와 보장 일자를 포함해 보험사에 따라 다르게 나타납니다. 데이터 생성 과정에서 pyBarcode 라이브러리를 사용해 바코드를 인코딩하고, 문서 템플릿에 통합하는 단계를 거치며, 데이터 다양성을 높이기 위해 데이터 증강 기법도 적용합니다.

- **Performance Highlights**: LLM을 기반으로 생성된 데이터는 기존의 합성 데이터에 비해 다양성과 맥락적 관련성이 향상되어 바코드 탐지 모델의 성능 개선으로 이어지는 것으로 나타났습니다. 우리는 이 새로운 방식이 다양한 문서 형식과 바코드 유형에 강력한 성능을 발휘할 수 있음을 입증하였습니다. 이 방법은 문서 유형, 지역 또는 바코드 표준의 변경에 쉽게 적응할 수 있어 신원 문서 표준의 발전에 따라 진화할 수 있는 유연한 솔루션을 제공합니다.



### Information Extraction from Heterogenous Documents without Ground Truth Labels using Synthetic Label Generation and Knowledge Distillation (https://arxiv.org/abs/2411.14957)
Comments:
          Accepted to WACV 2025

- **What's New**: 이 논문에서는 레이블이 없는 VRD(Visually Rich Document) 자료에서 합성 무작위 레이블을 생성하는 "Task Aware Instruction-based Labelling (TAIL)" 방식을 제안합니다. 이 방식을 통해 멀티모달 문서 이해 모델(VRDU)을 훈련시키고, 기존의 레이블이 없는 정보 추출 문제를 해결하고자 합니다. 또한, 기존의 상업용 LMM을 사용하지 않고도 문서에서 필요한 정보를 효율적으로 추출할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구는 인공지능 모델을 통해 불확실한 레이블을 다루며, 클로드 3 소네트를 기반으로 하는 합성 레이블을 생성합니다. 이 과정을 통해 TAIL 레이블을 활용하여 VRDU 모델을 훈련시키고, 특히 응답 기반 지식 증류(response-based knowledge distillation)를 활용하여 효과적인 정보 추출을 구현합니다. 다양한 언어와 형식이 포함된 문서에서 정보 추출의 신뢰성을 높이기 위한 결론에 도달했습니다.

- **Performance Highlights**: 실험 결과, LlaVA-Net 모델은 내부 비용을 85% 절감하면서도 약 5배 더 빨리 작업을 수행할 수 있으며, 최신 LMM인 Claude 3 Sonnet과 비교하여 동등하거나 약간 더 나은 성능을 보이는 것으로 나타났습니다. 또한, 레이아웃 인식 기반 방법들보다 더 나은 성과를 내며, 평균 정규화 레벤슈타인 유사도(ANLS) 점수에서 10% 이상 향상된 결과를 기록했습니다.



### Evaluating LLM Prompts for Data Augmentation in Multi-label Classification of Ecological Texts (https://arxiv.org/abs/2411.14896)
Comments:
          Ivannikov ISPRAS Open Conference (ISPRAS) 2024

- **What's New**: 이 연구에서는 큰 언어 모델(LLM)을 활용하여 러시아 소셜 미디어에서 그린 프랙티스(green practices)의 언급을 탐지하기 위한 프롬프트 기반 데이터 증강(prompts-based data augmentation) 기법을 적용했습니다. 데이터 증강을 통해 그린 프랙티스의 보급 정도를 이해하고, 환경 문제 완화를 위한 에코 프렌들리(clean actions) 조치를 확장하기 위한 권고를 제시할 수 있습니다. 이 연구에서는 프롬프트를 사용하여 다중 라벨 분류(multi-label classification) 성능을 개선하기 위해 여러 전략을 실험했습니다.

- **Technical Details**: 연구에서는 GreenRu라는 러시아 소셜 미디어 데이터셋을 사용하였으며, 1326개의 게시물에 대해 9가지 유형의 그린 프랙티스에 대한 레이블이 제공되었습니다. LLM은 T-lite-instruct-0.1을 사용하였으며, 이는 주로 러시아어로 사전 훈련된 모델입니다. 네 가지 프롬프트 전략이 사용되었으며, 이는 원문을 패러프레이징(paraphrasing)하거나 주어진 카테고리에 따라 새로운 샘플을 생성하는 방법입니다.

- **Performance Highlights**: 연구 결과, 데이터 증강 전략이 원래 데이터셋으로만 세밀하게 조정된 모델보다 분류 성능을 향상시킨 것으로 나타났습니다. 특히 원본 텍스트를 패러프레이징하는 전략이 가장 우수한 성능을 보였으며, 이는 관련 카테고리를 명확하게 표시함으로써 이루어졌습니다. 이러한 결과는 대부분의 경우 기준선(baselines)보다 높은 성능을 기록했습니다.



### Leveraging Hierarchical Prototypes as the Verbalizer for Implicit Discourse Relation Recognition (https://arxiv.org/abs/2411.14880)
- **What's New**: 이 논문에서는 기존의 수동적인 verbalizer에 의존하던 방법 대신, 클래스 수준의 의미적 특징을 포착하는 프로토타입(prototype)과 다양한 클래스의 계층적(label hierarchy) 구조를 활용한 새로운 방식을 소개합니다. 이렇게 제안된 접근법은 zer0-shot cross-lingual learning을 가능하게 하여 자원이 부족한 언어에서도 담론 관계를 인식할 수 있는 가능성을 제시합니다. 이는 다양한 언어에서 implicit discourse relation recognition(IDRR) 문제를 해결하는 데 실용적이고 다재다능한 접근법으로 자리잡을 수 있습니다.

- **Technical Details**: 논문에서 제안하는 방법은 각 클래스에 대한 프로토타입 벡터(central points)를 추정하여 이를 verbalizer로 활용합니다. 특히, 차별적 학습(contrastive learning)을 통해 클래스 프로토타입 간의 거리, 인스턴스 간의 거리, 프로토타입과 인스턴스 간의 거리 조정이 이루어집니다. 이는 PDTB-2 및 PDTB-3의 계층 감각 구조를 반영하여 효과적으로 작동합니다. 이 방법은 단일 언어 IDRR 뿐만 아니라 제로 샷 크로스 언어 학습을 지원하여, 데이터가 부족한 다른 언어에서도 적용 가능성을 높입니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 접근법은 영어에서 경쟁하는 기준선 대비 성능 향상을 보여줍니다. 또한, 적은 데이터로 지원되는 다른 언어에서 IDRR 작업에서도 더 나은 결과를 도출할 수 있는 잠재력을 가지고 있습니다. 기존의 접근법에서 발생할 수 있는 수동적 verbalizer의 불확실성을克服할 수 있으며, 실제 응용에 적합한 모델로의 가능성을 열어줍니다.



### Astro-HEP-BERT: A bidirectional language model for studying the meanings of concepts in astrophysics and high energy physics (https://arxiv.org/abs/2411.14877)
Comments:
          7 pages, 4 figures, 1 table

- **What's New**: 이 논문은 Astro-HEP-BERT라는 새로운 language model을 소개합니다. 이는 astrophysics와 high-energy physics 분야에서 개념의 의미를 연구하기 위해 고안된 transformer 기반의 모델입니다. Astro-HEP-BERT는 이미 사전 훈련된 BERT 모델을 바탕으로 하여, 21.84 백만 개의 문단을 포함한 Astro-HEP Corpus를 사용하여 세 번 더 훈련되었습니다.

- **Technical Details**: Astro-HEP Corpus는 1986년부터 2022년까지 발표된 600,000개 이상의 학술 기사를 포함하며, 이를 통해 생성된 contextualized word embeddings (CWEs)로 astrophysics와 HEP의 개념을 분석합니다. CWEs는 단어의 특정 맥락을 기반으로 고유한 표현을 생성함으로써, 단어의 의미를 포괄적이고 구체적으로 포착할 수 있습니다. 이 모델은 기존 BERT의 가중치와 어휘를 재사용하고, Astro-HEP Corpus에서의 추가 훈련을 통해 특정 도메인에 최적화되었습니다.

- **Performance Highlights**: 초기 평가 결과, Astro-HEP-BERT의 CWEs는 대규모 데이터셋에서 무작위로 학습된 도메인 특화 BERT 모델과 유사한 성능을 보였습니다. 이는 HPSS 연구자들에게 특정 과학적 도메인에 대한 일반 언어 모델을 재훈련하는 것이 비용 효율적이고 효율적인 전략이 될 수 있음을 시사합니다. Astro-HEP-BERT 모델과 관련 자료는 Hugging Face에서 공개적으로 사용 가능합니다.



### De-biased Multimodal Electrocardiogram Analysis (https://arxiv.org/abs/2411.14795)
- **What's New**: 이 연구는 Electrocardiogram (ECG) 데이터를 Multimodal Large Language Models (MLLMs)로 처리하는 새로운 방식에 대해 설명합니다. 기존의 접근법은 ECG 신호를 텍스트 태그로 변환하여 정보를 압축했지만, 본 연구에서는 direct projection layer를 통해 ECG 임베딩을 LLM에 입력하여 더 많은 정보를 보유하고, LLM의 추론 능력을 극대화합니다. 이 방법은 두 개의 ECG를 비교하는 임상 실습 상황에서도 효과적으로 작동합니다.

- **Technical Details**: ECG는 심장 전기 활동을 기록하며 심장 질환의 진단에 도움을 줍니다. 심층 학습의 발전과 함께 ECG 분석 및 진단을 위한 자동화된 방법이 증가하고 있습니다. 저자들은 800,000개의 ECG-리포트 쌍을 사용하여 multimodal contrastive learning을 통해 ECG 인코더를 사전 훈련한 후, ECG 임베딩을 LLM의 의미 공간에 사상하여 정보를 극대화했습니다.

- **Performance Highlights**: 이 연구는 adversarial testing에서 우수한 성과를 보여 ECQ-QA 작업에서 효과적인 성능을 입증하였습니다. 모델은 zero-shot 기능도 나타내며, 특히 random ECG 테스트를 통해 입력된 ECG 신호를 잘 이해하고 활용함을 추가로 검증하였습니다.



### KBAda: Efficient Self Adaptation on Specific Knowledge Bases (https://arxiv.org/abs/2411.14790)
- **What's New**: 새롭게 제안된 KBAda는 대규모 언어 모델(LLM)이 지식 기반에 효율적으로 적응하도록 설계된 기법입니다. 이 접근 방식은 자가 주석(self-annotation)된 데이터를 활용하여 LLM의 자가 학습 잠재력을 활용합니다. KBAda는 Q&A 쌍 및 수정 제안과 같은 자가 주석 데이터를 통해 반복적인 학습을 수행하며, 자원 소모 없이도 모델 성능을 크게 향상시키는 것으로 확인되었습니다.

- **Technical Details**: KBAda는 LLM이 지식 기반(KB) 적응을 위해 자가 주석 방법을 통해 모델을 세밀하게 조정합니다. 이는 사실 QA, 장문 QA 및 전문 분야 테스트와 같은 다양한 다운스트림 작업에서 모델 성능을 확인합니다. 실험 결과, 기존의 GPT-4-turbo 주석만큼의 성능 향상(90% 이상)을 달성하면서도 전적으로 자가 감독(self-supervision)에 의존하여 비용을 낮췄습니다.

- **Performance Highlights**: KBAda는 반복적인 자기 검사와 학습 대응을 통해 모델의 반응을 개선하며, 이 과정에서 쿼리 확장(query expansion) 및 신뢰도 검증(confidence verification) 등의 전략을 활용합니다. 다양한 실험 결과 KBAda가 어떻게 일반 지식 콘텐츠를 마스터하고 значительный 성능 향상을 달성하는지를 보여줍니다. 아울러 자기 주석 데이터의 효율적 양과 최적의 학습량도 규명하여 실무에서 KBAda를 효과적으로 활용하는 데 도움을 줄 수 있는 지침을 제공합니다.



### Universal and Context-Independent Triggers for Precise Control of LLM Outputs (https://arxiv.org/abs/2411.14738)
- **What's New**: 이 연구에서는 기존의 프롬프트 주입 공격(Prompt Injection Attack) 기술을 발전시켜, 범용적(universal)이고 상황에 독립적(context-independent)이며 정확한 출력(precise output)을 제어하는 트리거(trigger)를 찾는 방법을 제안합니다. 이러한 트리거는 광범위한 응용 분야에서 LLM의 출력을 조작할 수 있는 가능성을 제공합니다.

- **Technical Details**: 제안된 방법은 주입된 프롬프트를 두 가지 논리적 구성 요소로 나누는 것입니다: 원하는 내용을 인코딩하는 페이로드(payload)와 모델이 특정 내용을 출력하도록 활성화하는 트리거입니다. 이를 통해 다양하고 복잡한 프롬프트 상황에서도 적용 가능한 트리거를 발견하는 접근법을 제시하며, 이를 통해 평균적으로 높은 성공률을 기록하는 실험 결과도 포함됩니다.

- **Performance Highlights**: 연구 결과, 새로운 트리거는 다양한 문맥(context)과 작업(task)에서 효과적으로 작동함을 입증하였으며, 이러한 공격이 LLM 기반 응용 프로그램에 미치는 잠재적 위험성을 강조합니다. 또한 이 연구는 프롬프트 주입 공격의 심각성을 알리고, 이러한 공격에 대한 경각심을 높이는 데 기여하고자 합니다.



### MolReFlect: Towards In-Context Fine-grained Alignments between Molecules and Texts (https://arxiv.org/abs/2411.14721)
Comments:
          22 pages, 12 figures

- **What's New**: 이번 논문에서는 MolReFlect라는 새로운 teacher-student 프레임워크를 제안합니다. 이 프레임워크는 분자와 해당 설명 텍스트 간의 미세 정렬(fine-grained alignments)을 수행하여, 분자-캡션 번역 작업에서의 성능을 크게 향상시킵니다. 기존의 SMILES 또는 분자 그래프를 general하게 다루는 접근 대신, MolReFlect는 중요한 구문을 직접 추출하고 이를 분자의 하위 구조나 특성에 적용하는 방식을 채택합니다.

- **Technical Details**: MolReFlect는 세 가지 주요 단계로 구성됩니다: Zero-shot Alignment Extraction, In-Context Selective Reflection, 그리고 Chain-of-Thought In-Context Molecule Tuning (CoT-ICMT)입니다. 처음에는 대형 teacher LLM이 분자 SMILES 또는 캡션으로부터 중요한 구문을 추출하여 그에 맞는 하위 구조 또는 특성을 규명합니다. 그 후, student LLM은 이전에 추출된 결과와 reflective 결과 중에서 선택함으로써, 정렬 품질을 개선합니다.

- **Performance Highlights**: MolReFlect는 ChEBI-20 데이터셋에서 SOTA 성능을 달성하며, Mol2Cap 및 Cap2Mol 작업 모두에서 다른 기법들에 비해 우수한 성과를 기록하였습니다. 더욱이 ablation study를 통해 각 단계의 효과성을 분석하고, 실례를 통한 사례 연구로 분자와 텍스트 간의 미세 정렬의 중요성을 강조합니다.



### Optimizing Social Media Annotation of HPV Vaccine Skepticism and Misinformation Using Large Language Models: An Experimental Evaluation of In-Context Learning and Fine-Tuning Stance Detection Across Multiple Models (https://arxiv.org/abs/2411.14720)
- **What's New**: 이 논문은 대형 언어 모델(large-language models, LLMs)을 활용하여 HPV 백신 관련 트윗에서의 입장 탐지를 위한 사회적 미디어 콘텐츠 주석의 최적 전략을 실험적으로 결정합니다. 특히, 우리는 기존의 파인 튜닝(fine-tuning) 방법과 emergent in-context learning 방법을 비교하며, 다양한 프롬프트 엔지니어링 전략을 적용하여 LLMs와 그 변형들에 대한 성능을 조사했습니다.

- **Technical Details**: 연구에서는 프롬프트 템플릿 디자인, 샷 샘플링 방법, 샷 수를 변화시키며 HPV 백신에 대한 입장을 탐지했습니다. 연구 결과, 전반적으로 in-context learning이 HPV 백신 관련 사회적 미디어 콘텐츠의 입장 탐지에서 파인 튜닝보다 우수한 성능을 보였으며, 모델 간의 민감성 차이를 드러내었습니다. 또한, 최적의 in-context learning 구성으로는 6개의 층화 샷(stratified shots)을 세부적인 문맥 프롬프트와 짝지어 사용하는 것이 밝혀졌습니다.

- **Performance Highlights**: 이 연구는 대형 언어 모델을 사용하여 사회 미디어에서의 입장 및 회의론 탐지 연구에 적용할 수 있는 잠재력과 접근 방식을 강조합니다. 특히, 여러 LLMs와 그 변형들이 in-context learning 조건에 따라 상이한 성능을 보이며, 샷 양의 증가가 반드시 성능 향상으로 이어지지 않는다는 점이 중요합니다. 이러한 발견들은 향후 사회적 미디어 데이터 분석에 있어 LLM 활용 가능성을 시사합니다.



### Improving Mathematical Reasoning Capabilities of Small Language Models via Feedback-Driven Distillation (https://arxiv.org/abs/2411.14698)
- **What's New**: 이 논문에서는 Feedback-Driven Distillation (FDD) 프레임워크를 제안하여 Small Language Models (SLMs)의 수학적 추론 능력을 향상시키는 방법을 다룹니다. 이 접근법은 대형 언어 모델(Large Language Models, LLMs)에서 고급 추론 능력을 SLMs에 전달하기 위해 고안되었습니다. 데이터의 양과 질을 동시에 고려하면서 문제를 난이도에 따라 쉽게 또는 어렵게 분류하고 이에 따라 새로운 질문을 창출하여 증강된 데이터셋을 구축합니다.

- **Technical Details**: FDD는 세 가지 주요 단계로 구성됩니다: 초기화 단계에서 LLM에 수학 문제와 대응하는 추론 합리성을 짝짓게 하여 초기 데이터셋을 구축합니다. 그런 다음 SLM의 성능에 따라 문제를 난이도로 분류한 후, 각 유형에 맞게 추가 질문을 생성하여 데이터셋을 확장합니다. 마지막으로 다단계 증류(paradigm) 방식을 통해 이를 반복적으로 수행하여 SLMs의 수학적 추론 능력을 점진적으로 스타일로 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 FDD 프레임워크를 통해 SLMs는 수학적 추론 작업에서 최신 성능(state-of-the-art performance)을 달성할 수 있음을 입증했습니다. 또한, 이 방법은 SLMs의 도메인 내(in-domain) 수학적 추론 성능을 향상시킬 뿐만 아니라 도메인 외(out-of-domain)에서도 성능을 크게 향상시킨다는 것을 보여주었습니다.



### Multiverse of Greatness: Generating Story Branches with LLMs (https://arxiv.org/abs/2411.14672)
Comments:
          12 pages, 14 figures

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)과 상호작용하여 그래프 기반 콘텐츠를 생성하는 새로운 프레임워크인 DCP/P(동적 컨텍스트 프롬프트/프로그래밍)를 제안합니다. 기존의 연구에서는 시각 소설 게임을 생성하는 데 있어 수작업을 많이 요구하였고, 긴 일관된 이야기를 생성하는 데 유연성이 부족했습니다. DCP/P는 LLM에게 적절한 맥락을 제공함으로써 이전 연구의 한계를 극복하고 있습니다.

- **Technical Details**: DCP/P는 동적 프로그래밍과 그래프 탐색 알고리즘을 활용하여 LLM과 상호작용하는 새로운 접근 방식을 소개합니다. 이 프레임워크는 사용자 개입을 최소화하며, LLM이 주인공, 장소 및 선택지를 결정하도록 허용합니다. 또한 각 게임 생성에 필요한 이미지 자산을 생성하기 위해 이미지 생성 모델을 통합하였습니다.

- **Performance Highlights**: DCP/P를 사용하여 생성된 게임들은 기준선보다 일관되게 우수한 성과를 보였습니다. 연구진은 20개 스토리의 객관적인 평가를 통해 DCP/P의 효과성을 입증하였고, 생성된 콘텐츠의 질과 편향도 평가하여 LLM의 특정 단어 및 감정 편향 경향을 확인하였습니다. 모든 연구 코드와 프레임워크는 오픈소스로 제공되어 향후 연구를 지원할 수 있도록 하였습니다.



### Comparative Analysis of Pooling Mechanisms in LLMs: A Sentiment Analysis Perspectiv (https://arxiv.org/abs/2411.14654)
Comments:
          4 figures

- **What's New**: 이번 논문은 BERT와 GPT와 같은 대규모 언어 모델(Large Language Models, LLMs)의 풀링 메커니즘이 문장 수준의 감정 분석에 미치는 영향을 조사합니다. 전통적인 Mean, Max 및 Weighted Sum 풀링 전략의 성능을 비교하기 위한 포괄적인 실험이 수행되었습니다. 연구의 결과는 다양한 작업 요구 사항에 따라 각 풀링 메커니즘이 서로 다른 장점과 단점을 가지고 있음을 강조합니다.

- **Technical Details**: Transformer 기반 아키텍처의 설계 핵심은 어텐션 메커니즘으로, 이는 모델이 시퀀스 데이터를 처리할 때 서로 다른 토큰의 중요도를 동적으로 가중할 수 있게 합니다. 본 연구에서는 주요 풀링 전략으로 Mean, Max 및 Weighted Sum 풀링을 탐구하며, 각 기법의 특성과 상호작용을 분석합니다. 이를 통해 문장 수준 표현을 생성하기 위한 다양한 풀링 방법의 기초적인 수학적 설명도 제공됩니다.

- **Performance Highlights**: BERT 및 GPT 모델에 대한 감정 분석 실험 결과, 각 풀링 방법은 특정 상황에서 상이한 성능을 보여주며, 이는 특정 작업에 대한 최적의 해결책을 찾는 데 중요한 인사이트를 제공합니다. 특히, 어떤 풀링 방법이 특정 유형의 감정 분석에 보다 효과적인지를 이해함으로써 LLM 기반 모델의 향후 최적화 방향에 기여할 수 있습니다. 연구는 LLM 기반 모델을 특정 작업의 요구 사항에 맞게 조정하는 방법에 대한 실용적인 권장 사항을 포함합니다.



### Benchmarking Multimodal Models for Ukrainian Language Understanding Across Academic and Cultural Domains (https://arxiv.org/abs/2411.14647)
- **What's New**: 이 논문에서는 우크라이나어 중심의 종합적인 멀티모달 벤치마크인 ZNO-Vision을 소개하고 있습니다. 이는 12개 학문 분야에 걸쳐 4,300개의 전문가 제작 질문으로 구성되어 있으며, 이를 통해 LLM(large language models)의 성능 평가를 보다 포괄적으로 수행할 수 있게 됩니다. 또한, 이 연구는 우크라이나어에 대한 멀티모달 텍스트 생성의 첫 번째 평가 연구를 진행하였으며, Multi30K-UK 데이터셋에서 캡션 생성 품질을 측정했습니다.

- **Technical Details**: ZNO는 우크라이나의 고등학교 졸업생을 위한 국가 시험으로, 이 벤치마크는 수학, 화학, 인문학 등 여러 과목을 포괄합니다. 우리는 13개 카테고리로 나누어진 4,306개의 질문 쌍으로 구성된 데이터셋을 구축하고, OpenAI의 GPT-4o를 사용하여 VQA 2.0의 영어 질문을 우크라이나어로 번역했습니다. 또한, 새로운 UACUISINE 벤치마크를 개발하여 우크라이나의 전통 요리에 대한 깊이 있는 평가가 가능하도록 하였습니다.

- **Performance Highlights**: 실험 결과, 대부분의 모델이 기준선보다 낮은 성능을 보였으며, 일부 모델만이 기준 성능을 초과했습니다. 특히, 우크라이나 음식에 대한 문화적 지식 테스트에서 Paligemma 모델이 성능 향상에 성공했으며, 이는 도메인 적응 가능성을 보여주었습니다. 본 연구는 향후 우크라이나 언어 및 기타 저자원 언어에 대한 멀티모달 모델 개발에 기여할 것으로 기대됩니다.



### An Experimental Study on Data Augmentation Techniques for Named Entity Recognition on Low-Resource Domains (https://arxiv.org/abs/2411.14551)
Comments:
          21 pages, 2 figures

- **What's New**: 본 연구에서는 Named Entity Recognition (NER)과 관련하여 데이터 증강(data augmentation) 기술의 효과를 평가합니다. 특히 Mention Replacement(MR) 및 Contextual Word Replacement(CWR)라는 두 가지 주요 텍스트 증강 기법이 Bi-LSTM+CRF와 BERT 모델에 미치는 영향을 알아봅니다. 이 과정에서 저자들은 다양한 저자원(lowersource) 데이터셋으로 실험을 수행하고, 데이터 증강이 소규모 데이터셋에 유리하다는 것을 증명했습니다.

- **Technical Details**: 본 연구에서는 NER 모델 훈련에 있어 데이터 증강 기술이 미치는 영향을 분석했습니다. 연구에 사용된 두 가지 모델인 Bi-LSTM+CRF와 BERT는 각각 고유한 아키텍처를 가지고 있으며, 특히 BERT 모델은 데이터 증강의 이점을 더 많이 누리는 것으로 나타났습니다. 연구자들은 데이터 증강의 최적 수량이 정해지지 않으며, NER 실무자들은 다양한 양의 데이터를 활용하여 실험해야 한다고 강조했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 데이터 증강은 특히 작은 데이터셋에서 유리하지만, 큰 데이터셋에서는 오히려 낮은 성능을 보일 수도 있음을 보여주었습니다. 또한, CWR은 MR보다 더 우수한 성능을 보였으며, BERT 모델이 Bi-LSTM+CRF 모델보다 데이터 증강의 혜택을 더 많이 받는 것으로 나타났습니다. 이 연구는 앞서 연구되지 않았던 저자원 데이터셋에 대한 NER 모델 평가를 포함하여, 기존 연구에 기여하고 있습니다.



### Exploring Accuracy-Fairness Trade-off in Large Language Models (https://arxiv.org/abs/2411.14500)
Comments:
          9 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 정확성과 공정성을 조화롭게 향상시키는 복잡한 문제를 탐구합니다. 기존의 방법들은 종종 성능을 최적화하기 위해 공정성을 희생하게 되며, 이는 사회적인 피해를 초래할 수 있습니다. 우리는 LLM 훈련 과정을 다목적 학습(task)으로 재구성하고, 다목적 진화 학습(MOEL) 방법론을 통해 이 문제를 해결하기 위한 새로운 경로를 제시합니다.

- **Technical Details**: 해당 연구에서는 LLM의 불공정성을 완화하기 위한 기존 방법들을 네 가지 단계(전처리, 훈련 중, 처리 중, 후처리)로 나누어 설명합니다. 특히 MOEL은 여러 목표를 동시에 최적화할 수 있는 알고리즘으로, 파레토 최적 집합(Pareto-optimal set)을 생성하여 다양한 요구 사항을 충족할 수 있는 LLM 모델의 세트를 제공합니다. MOEL을 활용함으로써 더 나은 정확성 및 공정성을 동시에 달성할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 이 연구는 MOEL 방법론을 통해 공정성과 정확성을 동시에 최적화한 LLM의 구성요소를 제공하며, 다양한 트레이드오프를 통해 사용자에게 적합한 모델 선택의 유연성을 부여합니다. LLM의 공정성 문제에 대한 해결책을 제시함으로써 더 공정하고 효율적인 AI 기술의 발전을 도모하고 있습니다. 이러한 접근법은 LLM이 부정확하거나 불공정한 결과를 초래하는 위험을 줄이는 데 기여할 것으로 기대됩니다.



### Understanding World or Predicting Future? A Comprehensive Survey of World Models (https://arxiv.org/abs/2411.14499)
- **What's New**: 세계 모델(world models)의 개념은 GPT-4와 Sora와 같은 최신 멀티모달 언어 모델 및 비디오 생성 모델의 발전으로 주목받고 있습니다. 이 논문은 세계 모델 관련 문헌을 포괄적으로 검토하며, 세계 모델이 현재 세계 상태를 이해하거나 미래의 역동성을 예측하는 도구로 여겨진다고 설명합니다. 세계 모델의 구성과 기능을 체계적으로 분류하고, 자율주행, 로보틱스, 사회 시뮬레이터 등 주요 분야에서의 응용을 살펴보는 것이 중요하다고 강조합니다.

- **Technical Details**: 세계 모델의 정의는 일반적으로 두 가지 관점으로 나뉘며, 하나는 세계를 이해하는 것이고 다른 하나는 미래를 예측하는 것입니다. Ha와 Schmidhuber의 초기 연구는 외부 세계를 추상화하여 그 기제를 이해하는 데 초점을 맞췄습니다. 반면, LeCun은 세계 모델이 실제 세계를 인식하고 모델링할 뿐만 아니라, 의사결정을 위한 가능한 미래 상태를 구상할 수 있어야 한다고 주장했습니다. 이러한 이론적 배경을 통해 연구자들은 세계 모델의 내부 표현(internal representation)과 미래 예측(future prediction) 기능을 확립하였습니다.

- **Performance Highlights**: Sora 모델은 현실 세계의 시각 데이터를 입력받아 미래 세계의 진화를 예측하는 비디오 생성 모델로, 세계 시뮬레이터로서의 역할을 하고 있습니다. 이 모델은 카메라 움직임 중에도 3D 비디오 시뮬레이션의 일관성을 유지하며, 물리적으로 그럴듯한 결과를 생성하는 능력을 보여줍니다. 이러한 발전은 세계 모델의 효율성을 높이며, 다양한 실제 응용 프로그램에 적합한 방향으로 미래 연구의 방향성을 제시합니다.



### Star-Agents: Automatic Data Optimization with LLM Agents for Instruction Tuning (https://arxiv.org/abs/2411.14497)
- **What's New**: 이번 논문에서는 Star-Agents 프레임워크를 제안하여 대규모 언어 모델(LLMs)의 성능을 향상시키기 위한 새로운 데이터 최적화 방법론을 소개합니다. 이 프레임워크는 다양한 LLM 에이전트를 활용하여 데이터 품질을 자동으로 향상시키고, 이를 통해 더 높은 품질의 훈련 데이터를 생성하게 됩니다. 기존의 인스트럭션 튜닝 방식에서 인간 주도 데이터 생성의 한계를 극복하고자 하며, 최종적으로 모델의 성능을 12% 이상 향상시키는 결과를 보여줍니다.

- **Technical Details**: Star-Agents 프레임워크는 세 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 여러 고급 LLM 에이전트를 활용하여 다양한 인스트럭션 데이터를 생성합니다. 두 번째 단계는 이 데이터를 두 모델 평가 방식을 통해 분석하여 복잡성과 품질을 기준으로 고품질 샘플을 선택합니다. 마지막으로, 선택된 데이터의 복합 점수를 기반으로 에이전트-쌍의 샘플링 확률을 조정하여 데이터 품질과 다양성을 균형 있게 유지합니다.

- **Performance Highlights**: 본 연구에서 실시한 실험에서는 Pythia와 LLaMA 같은 LLM을 사용한 인스트럭션 튜닝 결과, Star-Agents 프레임워크로 생성된 데이터로 학습한 LLM이 기존의 Evol-Instruct나 IFD 메트릭 준수 데이터를 활용한 경우보다 더 나은 성능을 보였습니다. 특히, Fermi 지표에서 40%의 성능 개선이 이루어졌으며, 이는 입증된 벤치마크에서 관찰된 성과입니다.



### From Statistical Methods to Pre-Trained Models; A Survey on Automatic Speech Recognition for Resource Scarce Urdu Languag (https://arxiv.org/abs/2411.14493)
Comments:
          Submitted to SN Computer Science

- **What's New**: 이 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 기술의 진화와 함께 저자원이 있는 언어인 우르두어(Urdu)에 대한 연구 동향을 탐구합니다. 특히, 우르두어는 남아시아에서 널리 사용되고 있지만 여전히 많은 어려움에 직면해 있습니다. 본 논문은 ASR 분야에서 우르두어 관련 연구의 현재 상태와 향후 연구 방향을 제시합니다.

- **Technical Details**: 연구에서는 최신 기술을 활용하고 기존 데이터셋을 분석하며 효과적인 알고리즘과 도구를 평가합니다. 이렇게 함으로써, 우르두어 언어 처리를 위한 독특한 도전과 기회를 조명합니다. 주요 초점은 자원이 제한된 환경에서의 ASR 기술의 적용에 있습니다.

- **Performance Highlights**: 이 논문은 우르두어 ASR의 발전을 위한 기반을 마련하고, 향후 연구자들에게 영감을 주기 위한 정보를 제공합니다. 또한, 대중 언어와 비교했을 때 우르두어에 대한 ASR 연구의 중요성을 강조하며, 관련 분야 연구의 확대를 목표로 하고 있습니다.



### A Survey on Human-Centric LLMs (https://arxiv.org/abs/2411.14491)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 발전이 인간의 인지 및 행동 시뮬레이션에 미치는 영향을 다룹니다. 기존의 LLM들은 개인적 및 집단적 작업을 수행할 수 있는 능력을 성장시키며, 인간 행동을 모방하는 새로운 도구로 활용되고 있습니다. 연구는 LLM들이 인간의 사고, 의사결정 및 사회적 상호작용을 실제로 어떻게 재현하는지를 평가하고, 향후 연구 방향과 도전 과제를 제시합니다.

- **Technical Details**: 연구는 LLM의 인지적, 지각적, 사회적 능력을 평가하며, 이를 인간의 유사한 기술과 비교합니다. LLM이 특정 작업을 수행할 때 이들이 보여주는 강점은 체계적 추론, 패턴 인식, 창의력 등입니다. 그러나, 복잡한 다단계 로직 처리 및 감정적 공감에서는 한계를 보입니다. LLM을 인간 중심의 영역에 적용하는 방법과 이론적 프레임워크를 탐구합니다.

- **Performance Highlights**: LLM은 행동 과학, 정치 학, 사회학 등 다양한 분야에서 인간 행동을 모방하는 데 효과적입니다. 개별 및 집단적 맥락에서 LLM의 활용 가능성을 강조하며, 기본적인 프롬프트 기법과 다중 에이전트 접근 방식이 사용됩니다. 연구는 LLM의 실시간 적응력, 감정 지능, 문화적 민감성을 개선하고, 인간-AI 협업을 위한 보다 정교한 프레임워크 개발의 필요성을 강조합니다.



### GhostRNN: Reducing State Redundancy in RNN with Cheap Operations (https://arxiv.org/abs/2411.14489)
- **What's New**: 본 논문의 혁신적인 접근 방식은 GhostRNN이라고 불리는 새로운 효율적인 RNN 아키텍처를 제안한 점입니다. 이 모델은 숨겨진 상태(hidden state)의 중복성을 줄이고 계산 비용을 낮추기 위해 저렴한 연산에서 파생된 "유령 상태(ghost states)"를 활용합니다. 이를 통해 메모리 사용량을 약 40% 줄이면서도 성능을 유지할 수 있음을 보여 줍니다.

- **Technical Details**: GhostRNN은 GRU 구조를 기반으로 하며, 기존 모델의 중복성을 분석하여 이를 최소화합니다. 초기 내부 상태(intrinsic states)를 생성하고, 간단한 선형 변환(linear transformation)과 활성화 함수(activation function)를 사용하여 유령 상태(ghost states)를 생성합니다. 이러한 유령 상태는 기존의 숨겨진 상태와 결합되어 다음 계산 단계로 전달됩니다.

- **Performance Highlights**: KWS 및 SE 작업에서 실험 결과, GhostRNN은 Google Speech Commands 데이터셋에서 0.1%의 정확도 향상을 보이며, 기본 모델의 매개변수 수를 40% 줄였습니다. 또한 SE 작업에서는 SDR과 Si-SDR을 각각 0.1dB 향상시키며 약 40%의 압축률을 달성했습니다.



### Ensuring Safety and Trust: Analyzing the Risks of Large Language Models in Medicin (https://arxiv.org/abs/2411.14487)
- **What's New**: 이 논문의 주요 내용은 의학 분야에서 Large Language Models (LLMs)의 안전성 문제를 체계적으로 분석하고 다루는 것입니다. 특히, 의학 AI의 안전하고 신뢰성 있는 사용을 위해 진실성(Truthfulness), 회복력(Resilience), 공정성(Fairness), 강건성(Robustness), 및 프라이버시(Privacy)라는 다섯 가지 원칙을 제안하고, 이를 바탕으로 1,000개의 전문가 검증 질문을 포함한 MedGuard 벤치마크를 도입했습니다.

- **Technical Details**: 이 연구에서는 11개의 일반적으로 사용되는 LLM을 평가하여, 현재의 언어 모델들이 대부분의 벤치마크에서 낮은 성과를 보이고 있음을 발견했습니다. 이는 인간 의사의 높은 성과와 비교했을 때 더욱 두드러지며, 의학적 안전성과 관련된 메커니즘이 작동하더라도 상황이 크게 개선되지 않는다는 점을 보여줍니다.

- **Performance Highlights**: 이 논문은 최근 ChatGPT와 같은 고급 LLM들이 다양한 의학적 작업에서 인간 성과와 비슷하거나 이를 초월할 수 있다는 보고에도 불구하고, 여전히 안전성의 중요한 격차가 존재한다고 강조합니다. 이는 인간의 감독과 AI 안전 방어 장치의 필요성을 더욱 부각시키는 결과입니다.



### The Impossible Test: A 2024 Unsolvable Dataset and A Chance for an AGI Quiz (https://arxiv.org/abs/2411.14486)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 불확실성을 인식하는 능력을 평가하기 위한 새로운 평가 프레임워크를 제시합니다. 연구는 의도적으로 답이 알려지지 않은 대학원 수준의 질문 675개를 포함하는 자료를 사용하여, 12개의 최신 LLM 모델을 평가하였습니다. 이 프레임워크는 LLM이 부정확한 응답을 생성하기보다는 무지(ignorance)를 인정하는 경향에 중점을 두었습니다.

- **Technical Details**: 모델 평가에는 생물학, 철학, 수학 등 다양한 분야의 문제들이 포함되었으며, 전체 모델들은 문제 해결이 알려지지 않았음을 인정하는 정확도가 62%에서 68%에 이릅니다. 흥미롭게도 문제의 난이도와 모델의 정확도 간에는 역관계가 관찰되었습니다. 특히 GPT-4는 어려운 문제에 대해 35.8%의 불확실성 인식률을 나타내었으나, 더 쉬운 문제에서는 20.0%에 불과했습니다.

- **Performance Highlights**: 모델들은 창의성과 NP-완전 문제에서 불확실성을 인정하는 데 어려움을 겪었고, 철학 및 심리학 과제에서는 상대적으로 더 나은 성과를 보였습니다. 이러한 결과는 인공지능(AGI) 평가 연구의 중요성을 강조하며, LLMs가 자신의 지식 한계를 인식하는 데 현재의 한계를 나타낸다는 것을 보여줍니다. 연구는 모델 훈련 아키텍처와 평가 접근법을 개선하기 위한 새로운 방향성을 제시합니다.



### Mediating Modes of Thought: LLM's for design scripting (https://arxiv.org/abs/2411.14485)
Comments:
          Published at ACADIA 2024

- **What's New**: 이 논문에서는 Large Language Models (LLMs)를 사용해 디자인 프로세스에서 인간의 직관과 알고리즘의 논리를 연결하는 새로운 접근 방식을 제안합니다. 이러한 시스템이 자연어 프롬프트를 해석하여 지오메트리 오퍼레이션을 조합할 수 있는지 탐구합니다. 이 연구는 기계학습 원리를 활용한 디자인 도구의 잠재력을 강조하며, LLM이 사용자 의도를 유추하고 디자인 스크립트를 생성하는 능력을 검토합니다.

- **Technical Details**: 제안된 시스템은 다수의 LLM 에이전트를 사용해 사용자의 자연어 입력을 분석하고, 이를 바탕으로 지오메트리 로직 스크립트를 생성합니다. 이 알고리즘은 Grasshopper 소프트웨어의 특정 컴포넌트에 매핑되어 사용자 인터페이스 내에서 결과를 직접 생성합니다. 시스템은 사용자의 고수준 프롬프트에 따라 논리 연산의 시퀀스를 구축하는 방식으로 작동하며, 특정 복잡도 범위 내에서는 완전한 시각 스크립트를 생성하는 데 성공하지만, 이 복잡도 이상의 경우에는 실패합니다.

- **Performance Highlights**: 초기 결과는 LLM이 디자인 의도를 유추하고 지오메트리 로직을 구성할 수 있는 유망한 가능성을 보여줍니다. 연구진은 이 시스템이 알고리즘 디자인 메디에이터로서의 미래를 제시할 수 있다고 역설합니다. LLMs를 활용한 디자인 도구는 디자인 프로세스의 효과성과 사용자와의 상호 작용 방식을 혁신할 수 있습니다.



### Robust Planning with Compound LLM Architectures: An LLM-Modulo Approach (https://arxiv.org/abs/2411.14484)
- **What's New**: 이번 논문에서는 LLM-Modulo 프레임워크라는 혼합 LLM 아키텍처를 제안합니다. 이 프레임워크는 LLM과 완전한 검증 세트를 결합하여, 아웃풋의 신뢰성을 높이고 잘못된 출력을 최소화합니다. 특히, 이 방법은 이전의 프롬프트 엔지니어링 기법들이 갖는 한계를 극복하며, 생성된 모든 출력을 올바르다고 보장할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: LLM-Modulo 프레임워크는 LLM이 솔루션을 제안하고, 비평가가 그 제안을 검토하는 생성-검증-비판 루프(generative-test-critique loop)를 기반으로 합니다. 이 시스템의 기본 원리는 LLM의 출력을 검증할 Soundness(건전성)과 LLM의 전반적인 솔루션 생성 능력에 의존하는 Completeness(완전성)입니다. 사용자가 제공한 문제 정의를 통해 LLM에 입력이 주어지고, 그에 따른 제안이 이루어집니다.

- **Performance Highlights**: 여러 도메인에서 LLM-Modulo 프레임워크의 평가 결과, 색다른 성능 향상을 보였습니다. 예를 들어, Travel Planner의 경우 GPT-4o의 정확도가 8.3%에서 23.89%로, Claude-3.5-Sonnet은 4.4%에서 25%로 증가했습니다. 더욱이, 자연계획 도메인에서는 GPT-4o의 성능이 3.43%에서 40%로 극적으로 개선되어, 새로운 시스템이 출력한 모든 솔루션이 비평가들에 의해 올바른 것으로 판단되었습니다.



### Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Comba (https://arxiv.org/abs/2411.14483)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 평가를 위한 효율적인 순위 시스템을 탐구합니다. 특히, 사용자 선호를 반영하여 LLM을 쌍으로 비교하는 새로운 접근법인 Pairwise Ranking을 도입하였습니다. 이에 따라, Elo와 같은 알고리즘을 활용하여 모델의 상대적 강점을 평가할 수 있는 방법론이 제안됩니다.

- **Technical Details**: 연구에서는 네 가지 널리 사용되는 순위 알고리즘(Elo, Bradley-Terry, Glicko, Markov Chain)을 평가하며, Arena Style과 Controlled Style이라는 두 가지 다른 평가 시나리오에서의 성능을 분석합니다. 각 알고리즘의 전이성(Transitivity), 예측 정확도(Prediction Accuracy), 하이퍼파라미터(Hyperparameter) 변화에 대한 안정성(Stability)을 검토하여 모델 평가에 적합한 알고리즘을 선택하는 데 필요한 기본적인 원칙을 확립합니다.

- **Performance Highlights**: 연구 결과, LLM의 평가 및 순위 결정에서 중요한 인사이트를 도출하였습니다. 특히, 개별 알고리즘의 성능 차이를 체계적으로 분석하여 다양한 평가 작업의 특성과 가용 자원에 따라 적합한 방법을 선택할 수 있는 가이드라인을 제시합니다. 본 논문은 LLM 순위 매기기를 위한 첫 번째 체계적인 연구로, 모든 코드 및 데이터는 재현 가능성을 높이기 위해 공개되었습니다.



### GRL-Prompt: Towards Knowledge Graph based Prompt Optimization via Reinforcement Learning (https://arxiv.org/abs/2411.14479)
- **What's New**: 이 논문에서는 기존의 Large Language Models (LLMs)에서 효율적인 prompt 최적화를 위해 GRL-Prompt라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 강화 학습(reinforcement learning, RL)을 통해 자동으로 최적의 프롬프트를 구성하며, 지식 그래프(knowledge graph)를 활용하여 사용자 쿼리와 예시 간의 상관관계를 효과적으로 인코딩합니다. 이를 통해 프롬프트를 생성하는 과정에서 발생할 수 있는 불확실성을 줄이고 성능을 향상시킬 수 있습니다.

- **Technical Details**: GRL-Prompt 프레임워크의 핵심 요소는 두 가지입니다: 첫째, 사용자 지침과 후보 in-context 예시로부터 구축된 지식 그래프입니다. 이 그래프는 이종 그래프 신경망을 사용하여 구조적 임베딩 표현을 암호화합니다. 둘째, 정책 네트워크(policy network)는 쌍별 엣지 분류기(pairwise edge classifier)와 in-context 매칭 네트워크(in-context matching network)를 포함하여 최적의 예시 순서를 생성하는데 기여합니다. 이 과정에서 임베딩 기반 보상 설계를 통해 RL 훈련 과정을 안정화합니다.

- **Performance Highlights**: 실험 결과, GRL-Prompt는 현재의 최첨단 방법들을 초월하여 ROUGE-1에서 평균 0.10, ROUGE-2에서 0.07, ROUGE-L에서 0.07, BLEU에서 0.05의 성능 향상을 이루었습니다. 여러 개의 데이터셋을 비교하여 GRL-Prompt가 다양한 NLP 작업에서 LLM의 성능을 효과적으로 향상시킨 것을 확인했습니다. 이로써 제안한 방법이 LLM의 잠재력을 극대화할 수 있는 가능성을 보여줍니다.



### StreetviewLLM: Extracting Geographic Information Using a Chain-of-Thought Multimodal Large Language Mod (https://arxiv.org/abs/2411.14476)
- **What's New**: 이번 논문에서는 지리적 예측에 있어 새로운 프레임워크인 StreetViewLLM을 제안합니다. 이 모델은 대규모 언어 모델과 chain-of-thought reasoning을 통합하여 비구조적이거나 다중 양식(multi-modal) 데이터를 효과적으로 처리합니다. StreetViewLLM은 스트리트 뷰 이미지와 지리적 좌표, 텍스트 데이터를 결합하여 지리적 예측의 정확성과 세분성을 향상시킵니다.

- **Technical Details**: StreetViewLLM은 retrieval-augmented generation 기술을 활용하여 도시 환경에 대한 세부 분석 및 지리적 정보 추출을 개선합니다. 이 프레임워크는 홍콩, 도쿄, 싱가포르, 로스앤젤레스, 뉴욕, 런던, 파리 등 7개 글로벌 도시에서 적용되어 효과를 입증했습니다. 이 모델은 인구 밀도, 의료 접근성, 정상화된 식생 지수(normalized difference vegetation index), 건물 높이, 불투수 표면(impervious surface) 등의 도시 지표를 예측하는 데 우수한 성능을 보여줍니다.

- **Performance Highlights**: StreetViewLLM은 기존 기준 모델들에 비해 지속적으로 더 나은 성능을 발휘하며, 예측 정확성이 향상되었습니다. 이 연구는 대규모 언어 모델을 도시 분석(urban analytics), 도시 계획의 의사 결정(decision-making), 인프라 관리(infrastructure management), 환경 모니터링 환경 분야에 통합할 수 있는 새로운 기회를 열어줍니다.



### Large Language Model for Qualitative Research -- A Systematic Mapping Study (https://arxiv.org/abs/2411.14473)
Comments:
          8 pages, includes 1 figures and 3 tables. Submitted to the WSESE 2025 ICSE Workshop

- **What's New**: 이 연구에서는 Large Language Models (LLMs)의 질적 분석에 있어 활용 현황 및 기회를 체계적으로 조사합니다. LLMs는 방대한 텍스트 데이터를 자동화하고 향상시키는 변혁적 도구로, 기존 질적 분석 방법의 한계를 극복할 수 있는 가능성을 보여줍니다. 본 논문은 LLMs의 적용 분야, 설정, 방법론 및 평가 지표를 맵핑하여 질적 연구에서의 최신 동향과 연구 격차를 파악하는 데 중점을 두고 있습니다.

- **Technical Details**: LLMs는 방대한 양의 데이터에서 학습한 패턴을 기반으로 새로운 데이터를 생성하는 Generative AI 모델입니다. 이들 모델은 고급 자연어 처리 기능을 통해 머신 번역, 요약, 텍스트 편집 및 챗봇 지원과 같은 다양한 작업을 수행할 수 있습니다. 특히 질적 분석 분야에서 LLMs는 비구조화된 데이터에서 패턴을 식별하고 이를 체계적으로 분석할 수 있게 해줍니다.

- **Performance Highlights**: LLMs는 기존의 노동 집약적인 질적 분석 과정을 자동화하여 데이터 분석의 속도와 효율성을 drastically 향상시킬 수 있습니다. 예를 들어, 교육 연구에서 ChatGPT는 학생 설문 조사 응답을 신속하게 분석하고 주제를 도출하는 데 사용되었으며, 병원 피드백 콘텐츠 분석에서도 큰 성과를 보였습니다. 그러나 LLMs의 신뢰성을 확보하기 위해서는 Prompt 설계와 같은 여러 가지 시스템적 연구와 개선이 필요합니다.



### Exploring the Potential Role of Generative AI in the TRAPD Procedure for Survey Translation (https://arxiv.org/abs/2411.14472)
- **What's New**: 이 논문은 생성 AI가 설문 도구를 번역하는 데 어떻게 기여할 수 있는지를 탐구하고 평가합니다. 특히, 복수의 언어 및 문화적 환경에서 활용되는 설문 조사에서 발생할 수 있는 번역 오류를 탐지하는 데 ChatGPT의 활용을 제안합니다. 시작하여 설계된 질문의 준비와 번역 과정에서 AI가 피할 수 있는 문제들을 식별하는 방법에 대해 논의합니다.

- **Technical Details**: 설문 조사 연구에서 구조화된 질문 세트는 양적 분석과 밀접하게 연결되어 있습니다. 연구자들은 번역 과정에서 의미 있는 결과를 수집하기 위해 질문의 정밀도를 확보해야 하며, 이를 위해서는 질문 설계, 사전 테스트 기법 등을 활용할 필요가 있습니다. 이 논문은 AI가 이러한 설계 과정을 지원하여 효율성을 높일 수 있음을 주장합니다.

- **Performance Highlights**: ChatGPT는 번역 문제에 대한 의미 있는 피드백을 제공하여 설문 조사의 문제점을 조기에 발견하고, 연구자가 연구 목표를 명확히하게 만드는 데 도움을 줄 수 있습니다. 이 연구의 발견은 향후 설문 번역 관행에 AI를 통합할 수 있는 다양한 경로를 제안합니다. 논문은 최적의 설문 조사를 위한 '고급 번역'의 한 사례로 AI 활용을 강조합니다.



### Popular LLMs Amplify Race and Gender Disparities in Human Mobility (https://arxiv.org/abs/2411.14469)
- **What's New**: 최근 LLMs의 사용이 사회적 결과에 영향을 미치는 여러 영역으로 확장됨에 따라, 이들 모델이 편견을 지속하고 확대하는 경향을 이해하는 것이 중요해졌습니다. 본 연구에서는 LLMs가 인종과 성별에 따라 인류 이동성(흔히 사람의 행동을 나타내는 중요한 지표)에 대한 예측에서 편견을 드러내는지 조사했습니다. GPT-4, Gemini 및 Claude라는 세 가지 LLM 모델을 사용하여, 인구 통계학적 세부 사항이 포함된 이름과 포함되지 않은 이름을 기반으로 한 관심 지점(POIs)에 대한 방문 예측을 분석하였습니다.

- **Technical Details**: 이 연구에서는 LLM들이 다양한 인구통계적 집단 간 POIs 방문 패턴을 이해하는 방식을 조사하기 위해 일련의 프롬프트를 개발했습니다. POIs는 (1) 직업 관련 및 일상 필수 POIs와 (2) 부유 관련 및 가난 관련 POIs의 두 그룹으로 구분되었습니다. 총 2,675개의 일반적인 이름을 선택하고 인종 및 성별 조합을 분석하며, LLM이 개인의 이름만을 기반으로 POIs를 예측하도록 요청했습니다.

- **Performance Highlights**: 연구 결과, 성별과 인종에 따라 LLM의 예측이 극명한 차이를 보였습니다. 남성은 여성보다 직업 관련 POIs와 강한 연관성을 보이며, 특히 백인 남성이 직업 관련 POIs와 12.2%의 확률로 연결되는 것으로 나타났습니다. 특히, 백인 여성은 직업 관련 POIs와 거의 연관이 없으며, 대부분의 예측이 일상 필수 POIs에 국한되었습니다. 또한, LLM은 인종 정보 없이 예측할 경우 부유 관련 POIs에 96.8%의 비율로 연결된다고 하였습니다.



### Learning to Ask: Conversational Product Search via Representation Learning (https://arxiv.org/abs/2411.14466)
Comments:
          Accepted by ACM TOIS

- **What's New**: 본 연구에서는 대화형 제품 검색 모델인 ConvPS를 제안합니다. 기존 연구들은 사용자와 상품 간의 정보 불일치 문제를 제대로 해결하지 못했습니다. 우리의 모델은 사용자, 쿼리, 아이템, 대화를 통합하여 표현 학습을 수행하고, 이를 바탕으로 보다 자연스럽고 적응력 있는 검색 시스템을 구축합니다.

- **Technical Details**: ConvPS 모델은 사용자, 쿼리, 아이템, 대화의 의미를 통합하여 학습하는 통합 생성 프레임워크를 사용합니다. 특히, 슬롯-값쌍(slots-value pairs)을 기반으로 한 질문 풀을 생성하고, 고성능 질문을 사용자에게 순차적으로 질문하는 방법을 학습합니다. 또한, 학습된 표현을 활용하여 상품을 검색합니다.

- **Performance Highlights**: 실험 결과는 ConvPS 모델이 기존 최첨단 문제 해결 모델에 비해 유의미하게 성능을 향상시킴을 보여줍니다. 이를 통해 사용자와의 대화에서 얻는 명확한 피드백을 기반으로 사용자의 동적인 선호를 더 잘 이해하게 됩니다.



### Testing Uncertainty of Large Language Models for Physics Knowledge and Reasoning (https://arxiv.org/abs/2411.14465)
- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)의 확신과 정확도 간의 관계를 탐구했습니다. 이 연구는 여러 선택형 물리학 질문에 대한 LLM 성능 평가 방법을 개발하였으며, 다양한 모델의 응답 변화와 정확도-확신 간의 트레이드오프를 분석했습니다. 이로써 LLM의 물리학적 추론과 지식 검색에서의 신뢰성을 논의하였습니다.

- **Technical Details**: 본 연구는 823823823823개의 고등학교 수준 물리학 질문으로 구성된 mlphys101 데이터셋을 사용하여, 세 가지 오픈소스 LLM과 하나의 폐쇄형 LLM의 답변을 얻었습니다. 질문은 정의 복제, 물리적 사실 복제, 개념 물리학 및 질적 추론, 단일 단계 추론, 다단계 추론으로 분류되었습니다. 모델의 성능은 고정된 few-shot 프롬프트ing 방식을 사용하여 평가되었으며, 답변의 확률 분포와 엔트로피를 계산하여 각 질문에 대한 확신도를 평가했습니다.

- **Performance Highlights**: 모델의 응답 다양성을 분석한 결과, Llama3.1-8B-Instruct 및 Mistral-7B-Instruct-v0.3 모델은 낮은 엔트로피 값을 보여 일관된 응답을 생성하는 경향이 있었습니다. 반면, GPT-3.5 Turbo는 더 높은 엔트로피를 탐지하여 응답 다양성이 컸고, Mixtral-8x7B-Instruct-v0.1은 대부분 낮은 엔트로피에서 답변을 제공하는 것으로 나타났습니다. 이러한 결과는 LLM 모델들이 질문에 대한 신뢰도를 가지기 위해서는 높은 정확도와 적절한 확신이 필요함을 시사합니다.



### Leveraging AI and NLP for Bank Marketing: A Systematic Review and Gap Analysis (https://arxiv.org/abs/2411.14463)
- **What's New**: 이 논문은 은행 마케팅에서 AI(인공지능)와 NLP(자연어 처리)의 영향력을 탐구하며, 이들 기술이 마케팅 전략 향상, 고객 참여 개선 및 가치를 창출하는 데 어떻게 기여할 수 있는지를 강조하고 있습니다. 기존의 마케팅에서 AI와 NLP의 연구는 많이 이루어졌으나, 은행 부문에서의 구체적인 활용에 대한 연구는 미비한 상황입니다. 이 연구는 PRISMA 방법론을 적용하여 현재의 AI 및 NLP 활용 현황을 체계적으로 검토하고, 향후 연구 기회를 제시합니다.

- **Technical Details**: 본 연구는 AI 및 NLP의 적용 가능성을 평가하기 위해 PRISMA(체계적인 리뷰를 위한 권장 보고 항목)에 따라 문헌 검토를 수행합니다. 이를 위해 Scopus 데이터베이스를 활용하여 관련 문헌을 광범위하게 스캔하고, 분석 마케팅, 마케팅과 NLP, 마케팅, 은행 및 NLP의 교차점을 탐색하는 세 가지 주요 쿼리를 설정했습니다. 이 과정에서 특정 Boolean 검색 쿼리를 통해 질 높은 연구만을 선별하여 시스템적으로 리뷰합니다.

- **Performance Highlights**: 연구 결과는 NLP가 은행 마케팅 전략을 향상시키는 데 중요한 역할을 하며, 고객 중심의 접근에서 특히 효과적이라는 점을 보여줍니다. 또한, 이 접근법은 고객 참여, 운영 효율성 및 규제 준수와 같은 분야에서 실질적인 통찰을 제공합니다. 이의 실행 가능성을 통해 은행 마케팅에서 NLP 기반의 혁신 성장 프레임워크 개발이 가능함을 제시하고 있습니다.



### Towards Next-Generation Medical Agent: How o1 is Reshaping Decision-Making in Medical Scenarios (https://arxiv.org/abs/2411.14461)
- **What's New**: 인공지능(AI)은 현대 의료에서 필수적이며, 대형 언어 모델(LLM)이 임상 의사결정에 있어 획기적인 발전을 보여주고 있습니다. 전통적인 모델 기반 접근법은 실시간 적응성, 다단계 추론 및 복잡한 의료 작업 처리에서 한계를 보이고 있으며, 이러한 문제를 해결하기 위해 에이전트 기반 AI 시스템이 등장했습니다. 이 시스템은 추론 추적, 상황에 따른 도구 선택, 지식 검색, 단기 및 장기 기억 기능을 포함하여 복잡한 의료 시나리오를 처리할 수 있습니다.

- **Technical Details**: 연구에서는 의료 AI 에이전트의 기초가 되는 LLM 선택의 영향을 조사합니다. 특히 emergent o1 모델이 대안으로 고려되며, Chain-of-Thought (CoT) 추론 프레임워크가 그 훈련 과정에 통합되어 의사결정 능력을 높입니다. o1은 다양한 의료 데이터셋에서 기존 모델들보다 탁월한 성능을 보여주며, 특히 LancetQA 및 NEJMQA 데이터셋과 같은 다단계 추론이 필요한 작업에서 강점을 보입니다.

- **Performance Highlights**: o1 모델을 통합한 세 가지 에이전트(Chain of Diagnosis, MedAgents, AgentClinic)를 다양한 유명 데이터셋에서 테스트한 결과, 진단 정확도 및 일관성이 향상되었습니다. 특히, o1은 DxBench 및 NEJMQA와 같은 임상 벤치마크에서 뛰어난 진단 정확도를 나타내며, 진단의 가변성을 줄이는 데 효과적입니다. 그러나 복잡한 작업에서는 뛰어난 성능을 보이지만, 단순한 작업에서는 효율성이 떨어져 기존 모델이 더 나은 선택이 될 수 있습니다.



### LLaSA: Large Language and Structured Data Assistan (https://arxiv.org/abs/2411.14460)
- **What's New**: 이 논문에서는 LLaSA(Large Language and Structured Data Assistant)라는 새로운 프레임워크를 제안합니다. LLaSA는 다양한 구조화된 데이터를 하이퍼그래프(hypergraph) 형식으로 통합하여 모델의 성능을 향상시키는 방법을 제공합니다. 특히, LLaSA는 self-supervised learning을 통해 하이퍼그래프 인코더와 G-Former를 사전 학습함으로써 다양한 LLM에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: LLaSA는 테이블, 지식 그래프와 같은 구조화된 데이터를 하이퍼그래프로 모델링하여 통합된 GNN을 통해 인코딩을 수행합니다. 이 과정에서 테이블의 셀은 노드로, 행과 열은 하이퍼엣지(hyperedge)로 취급되어 다양한 구조화된 데이터에 대해 일관성 있는 처리가 가능합니다. G-Former는 인코딩된 구조화된 데이터를 LLM이 이해할 수 있는 고정된 수의 soft token으로 변환하여 모듈 간의 격차를 해소합니다.

- **Performance Highlights**: 여러 SKG 태스크에 대한 실험 결과에서 LLaSA는 LLM의 다양한 구조화된 데이터 처리 능력을 현저히 향상시킴을 보여주었습니다. 예를 들어, LLaSA는 frozen LLM인 Llama-7B에서 10개 데이터셋에 대해 평균 12%의 성능 향상을 기록하였고, LoRA 조정된 LLM에서도 0.4%의 평균 향상을 달성했습니다. 또한, LLaSA는 전체 파라미터 조정을 사용하는 이전의 SOTA 방법보다 우수한 성과를 보입니다.



### Unveiling User Preferences: A Knowledge Graph and LLM-Driven Approach for Conversational Recommendation (https://arxiv.org/abs/2411.14459)
- **What's New**: COMPASS라는 새로운 프레임워크가 제안되어 기존의 CRSs(Conversational Recommender Systems)에서 사용자 선호도를 개선하고, 추천 성능과 설명 가능성을 높이고자 합니다. 이 프레임워크는 LLMs(Large Language Models)와 KGs(Knowledge Graphs)를 통합하여 비가시적인 사용자 선호도를 드러내는 새로운 방법을 제시합니다. 특히, COMPASS는 두 단계의 훈련 과정을 통해 비구조적 대화와 구조적 KGs 간의 간극을 좁힙니다.

- **Technical Details**: COMPASS는 그래프 엔티티 설명(pre-training) 메커니즘을 사용하여 KG 구조를 자연어 설명으로 변환하여 LLM이 도메인 특정 정보를 이해할 수 있게 만듭니다. 이어서, COMPASS는 지식 기반(instruction) 세부조정을 통해 사용자 선호에 대한 추론 능력을 높입니다. 이 과정에서 KG로부터 추출된 관련 엔티티 정보와 관계를 포함하여 대화 기록을 보강하는 KG 향상 컨텍스트를 사용합니다.

- **Performance Highlights**: COMPASS는 설명 가능한 사용자 선호 요약 생성능력을 통해 CRSs에서 실제 데이터와 대화 이력을 기반으로 한 지능형 추천 및 이해 가능한 결과를 제공합니다. 이 프레임워크는 기존의 CRS 모델과 통합되는 적응형 게이팅 메커니즘을 포함하여, 성능을 강화하고, 수정 없이도 추천 품질을 높입니다. 이를 통해 COMPASS는 더욱 명확하고 해석 가능한 추천 시스템을 구현할 수 있는 가능성을 제공합니다.



### Can Artificial Intelligence Generate Quality Research Topics Reflecting Patient Concerns? (https://arxiv.org/abs/2411.14456)
- **What's New**: 이 연구는 환자 중심의 연구를 위한 자동화된 프레임워크를 제안하며, 이를 통해 연구 및 환자 치료 간의 간극을 좁히려는 노력을 다룹니다. 혁신적인 자연어 처리(NLP)와 인공지능(AI)을 활용하여 환자 포털 메시지를 분석하고, 이를 통해 환자가 중요하게 여기는 문제를 우선시하는 연구 아이디어를 생성합니다. 이러한 접근 방식은 환자의 관점을 연구에 효과적으로 통합할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 연구는 2013년부터 2024년까지의 대규모 학술 병원에서 수집된 614,464개의 환자 메시지를 통해 2단계 비지도 학습 기반의 NLP 주제 모델을 구성하였습니다. AI(GPT-4o, OpenAI Inc, 2024년 4월 버전)를 활용하여 여러 단계의 작업을 수행하고, 지식 해석, 연구 아이디어 생성, 자기 반성 및 수정, 자기 확신의 과정을 통해 환자의 임상 문제를 해결하는 연구 주제를 생성했습니다.

- **Performance Highlights**: AI가 제안한 연구 주제 중 3분의 1이 높은 중요성과 독창성을 갖추었다고 평가되었습니다. 또한, 두 암(유방암과 피부암)에 대해서 AI가 제안한 연구 주제의 3분의 2가 독창적이라고 판단되었습니다. 이러한 결과는 방대한 환자 메시지를 통해 얻어진 AI 생성 연구 주제가 환자 중심의 건강 연구 방향을 의미 있게 안내할 수 있음을 보여줍니다.



### Direct Speech-to-Speech Neural Machine Translation: A Survey (https://arxiv.org/abs/2411.14453)
- **What's New**: 본 논문은 Speech-to-Speech Translation (S2ST) 모델에 대한 포괄적인 리뷰를 제공하며, 직접 S2ST 모델과 관련된 데이터, 응용 문제 및 성능 메트릭을 검토합니다. 특히, 전통적인 cascade 접근 방식과 비교하여, 직접 S2ST 모델의 장점과 함께 이들이 직면하는 주요 도전 과제를 분석합니다. 직접 S2ST 모델은 언어 중간 텍스트 생성 없이 음성을 변환할 수 있는 가능성을 제시하여 최근 주목받고 있습니다.

- **Technical Details**: 직접 S2ST 모델은 끝-to-끝 (end-to-end, E2E) 훈련 접근 방식을 따르며, 이는 오류 전파를 줄이고 더 낮은 지연(latency)을 제공합니다. 그러나 이러한 모델은 두 언어 간의 평행 음성 코퍼스 확보의 어려움, 평가의 복잡성, 음성 복제의 위협 등 여러 도전에 직면하고 있습니다. 기계 학습(Machine Learning, ML) 및 딥 러닝(Deep Learning, DL) 기술의 발전이 이러한 모델들에 도움을 줄 수 있습니다.

- **Performance Highlights**: 직접 S2ST 모델은 기존의 cascade 모델에 비해 실시간 퀄리티 성능에서 뒤쳐져 있으며, 특히 실제 세계 번역에서 그 성능이 미흡합니다. 이 논문에서는 벤치마크 데이터 세트를 통해 모델 성능을 분석하고, 연구의 도전 과제와 향후 방향성을 제안합니다. 최종적으로, S2ST 시스템이 원활한 커뮤니케이션을 달성하기 위한 더 많은 진전을 필요로 한다고 결론짓습니다.



### ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation (https://arxiv.org/abs/2411.15122)
- **What's New**: AI 기반 모델이 흉부 X선의 영상 판독 보고서를 자동으로 생성하는 가능성을 보여주고 있으나, 이를 객관적으로 평가할 수 있는 표준화된 벤치마크가 부족했다. 이 연구는 ReXrank라는 공개 리더보드와 챌린지를 소개하여 AI 영상 판독 보고서 생성 모델의 평가를 위한 체계적인 기준을 제공한다. ReXrank는 10,000개의 스터디로 이루어진 ReXGradient라는 대규모 데이터셋을 포함하여 세 가지 공개 데이터셋(MIMIC-CXR, IU-Xray, CheXpert Plus)을 활용하고 있다.

- **Technical Details**: ReXrank는 8개의 평가 지표를 사용하여 모델의 성능을 평가한다. 모델은 단순히 발견 사항을 생성하는 것과 발견 사항과 해석 모두를 제공하는 모델로 구분하여 평가된다. 다양한 데이터셋을 통합하여, 모델의 일반화 능력을 보다 깊이 이해할 수 있는 평가를 가능하게 한다.

- **Performance Highlights**: MedVersa 모델이 ReXGradient 및 MIMIC-CXR 데이터셋에서 최고 성능을 기록하며, 다양한 지표에서 GPT4V보다 우수한 성능을 보였다. 평가 지표의 분포를 분석한 결과, IU X-ray는 모든 모델에서 높은 성능을 보였지만, CheXpert Plus는 가장 높은 변동성과 낮은 성능을 나타냈다. ReXGradient 데이터셋은 매우 낮은 성능 변동성을 보여 모델의 강건성 평가에 유용함을 입증했다.



### VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinemen (https://arxiv.org/abs/2411.15115)
Comments:
          Project page: this https URL

- **What's New**: 최근 텍스트-비디오(T2V) 확산 모델들은 다양한 분야에서 뛰어난 생성 능력을 보여주고 있습니다. 하지만 이러한 모델들은 여러 객체와 속성이 포함된 복잡한 장면을 설명하는 텍스트 프롬프트와 정확하게 일치하는 비디오 생성에 어려움을 겪고 있습니다. 이 문제를 해결하기 위해 우리는 VideoRepair라는 새로운 모델 불가지론적 훈련 없는 비디오 정제 프레임워크를 도입합니다.

- **Technical Details**: VideoRepair는 미세한 텍스트-비디오 일치 문제를 자동으로 감지하고, 명확한 공간적 및 텍스트 피드백을 생성하여 T2V 확산 모델이 목표로 하는 부위에서 정제를 수행하도록 돕습니다. 이 프레임워크는 네 단계로 구성되어 있으며, 첫 번째 단계에서는 미세한 평가 질문을 생성하여 비디오를 평가하고, 두 번째 단계에서는 정제할 영역을 계획합니다. 세 번째 단계에서는 정확하게 생성된 영역을 세분화하고, 마지막 단계에서는 정제된 프롬프트에 따라 비디오를 재생성합니다.

- **Performance Highlights**: VideoRepair는 EvalCrafter와 T2V-CompBench라는 인기 있는 비디오 생성 벤치마크에서 최근 기준선보다 우수한 성능을 보입니다. 다양한 텍스트-비디오 정렬 메트릭에서도 상당한 개선을 보여주며, 정제 과정의 각 구성 요소에 대한 포괄적인 분석과 질적 사례를 제공합니다. 이 연구는 시각 생성 작업에서 자동 정제 프레임워크의 발전을 촉진할 것으로 기대됩니다.



### Efficient Pruning of Text-to-Image Models: Insights from Pruning Stable Diffusion (https://arxiv.org/abs/2411.15113)
- **What's New**: 본 연구는 리소스 제약이 있는 장치에서 텍스트-이미지 모델의 채택을 가로막는 주요 원인인 모델 크기의 문제를 해결하기 위해 Stable Diffusion 2의 후처리 가지치기(post-training pruning) 기법을 분석합니다. 이전에 다루어지지 않았던 다중 모드 생성 모델의 가지치기 기술을 다루면서 텍스트 컴포넌트와 이미지 생성 컴포넌트에 대한 가지치기 체계의 영향을 개별적으로 검토합니다. 이 연구는 기존 언어 모델 가지치기 경향과는 달리 텍스트-이미지 맥락에서 두드러진 발견을 보고합니다.

- **Technical Details**: 긴 모델 아키텍처의 경우, 모델 크기는 수십억 개의 파라미터를 포함하고 있으며, 이는 실질적인 연산 비용을 발생시킵니다. 본 논문에서는 텍스트 인코더와 확산 생성기를 개별적으로 가지치기하는 방법과 각 컴포넌트에 대한 최적의 희소성(sparsity) 수준을 모색합니다. 연구 결과에 따르면, Stable Diffusion 2의 경우 38.5%의 희소성을 달성하면서 품질 손실을 최소화할 수 있음을 발견했습니다.

- **Performance Highlights**: 가지치기를 통해 모델의 연산 요구 사항을 대폭 줄일 수 있으며, 특히 텍스트 인코더에 대해 47.5%까지, 확산 생성기에 대해서는 35%까지 최적의 가지치기 구성을 제안합니다. 연구에서 나타난 흥미로운 점으로는 특정 임계를 넘어서는 가지치기가 성능 급락을 야기할 수 있다는 것이며, 이는 특정 가중치들이 핵심 의미론적 정보를 담고 있음을 시사합니다. 이러한 발견은 텍스트-이미지 모델의 모델 압축, 상호 운용성 및 편향 식별에 대한 새로운 연구 방향을 열어줍니다.



### Context-Aware Multimodal Pretraining (https://arxiv.org/abs/2411.15099)
- **What's New**: 이 논문에서는 대규모 멀티모달 표현 학습(multi-modal representation learning)을 위한 간단하지만 신중하게 설계된 확장을 제안합니다. 이는 기존 contrastive learning 개념이 few-shot 적응(few-shot adaptation)을 지원하도록 최적화되지 않았음을 지적합니다. 따라서 이 연구의 결과로, 다양한 다운스트림 태스크에서 샘플 효율성이 최대 4배 향상되고 평균 5% 이상의 성능 향상이 나타났습니다.

- **Technical Details**: 제안된 LIxP는 Contrastive Language-Image Pretraining의 확장으로, 교육 과정에서 cross-attention 기반의 맥락화를 통해 메트릭 기반 적응을 위한 표현을 준비합니다. 이 접근법은 전체 손실 설계와 개별적으로 학습 가능한 온도를 활용하여 기초 제로샷(zero-shot) 능력을 유지합니다. LIxP는 21개의 다운스트림 분류 작업에서 제로샷 전이 성능을 유지하면서 원래 성능을 뛰어넘습니다.

- **Performance Highlights**: LIxP는 훈련 없는 메트릭 기반 적응 메커니즘을 통해 대규모 멀티모달 모델이 적응할 수 있도록 막대한 성능 향상을 보여줍니다. 이는 복잡하고 비싼 최적화 기반 방법에 비해 간단한 접근 방식을 채택하여 새로운 도메인으로의 일반화를 쉽게 만듭니다. 이 논문은 또한 기존의 few-shot 및 many-shot 전이와 관련된 문제를 해결하고, 더욱 효율적인 학습 모델을 구축하는 방향성을 제시합니다.



### Instance-Aware Generalized Referring Expression Segmentation (https://arxiv.org/abs/2411.15087)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서 제안하는 InstAlign는 복잡한 다중 객체 참조 표현을 효과적으로 처리하기 위한 첫 번째 인스턴스 인식 접근 방식을 제안합니다. 이 모델은 텍스트의 특정 구문과 개별 객체 인스턴스를 명시적으로 구별하고 연결함으로써, 더욱 정교한 세분화를 실현합니다. InstAlign는 우수한 인스턴스 분리를 통해 기존의 GRES 모델에 비해 성능을 크게 향상시킵니다.

- **Technical Details**: InstAlign는 Transformer 기반 아키텍처를 활용하여 물체 쿼리를 처리하고, 이는 입력 텍스트에서 지정된 객체의 인스턴스를 세분화하는 데 목적을 두고 있습니다. 모델은 K 개의 Phrase-Object Transformer 블록을 통해 다중 스케일 시각적 특성과 텍스트 특성을 결합하여 쿼리를 정제합니다. 최종적으로, 각 인스턴스는 텍스트 구문과 연계된 세분화 마스크를 생성하고, 이는 효율적으로 학습됩니다.

- **Performance Highlights**: InstAlign는 gRefCOCO 및 Ref-ZOM 벤치마크에서 매우 높은 성능을 자랑하며, gIoU 74.34%와 N-acc 79.72%를 기록하여 기존의 방법들을 3.3% 및 12.25% 이상 초과 달성했습니다. 본 연구는 복잡한 참조 표현을 처리하는 데 있어 새로운 표준을 정립하며, 인스턴스 세분화를 통한 성능 향상을 성공적으로 입증합니다.



### mR$^2$AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA (https://arxiv.org/abs/2411.15041)
- **What's New**: 이 논문은 최신 Knowledge-based VQA(task) 문제를 해결하기 위해 새로운 프레임워크인 mR²AG를 제안합니다. 기존의 Multimodal Retrieval-Augmented Generation (mRAG) 방법들이 가지는 한계를 극복하고, MLLM(Multimodal Large Language Models)의 지식 범위를 확장하는 데 중점을 둡니다. mR²AG는  두 가지 리플렉션(Reflection) 작업을 통해 정보 검색을 보다 효과적으로 수행하도록 설계되었습니다.

- **Technical Details**: mR²AG는 Retrieval-Reflection과 Relevance-Reflection 두 가지 메커니즘을 도입하여 사용자의 쿼리를 구분하고, 유용한 정보를 효과적으로 찾아냅니다. 이 프레임워크는 MLLM의 성능을 해치지 않으면서도 추가적인 복잡성을 유도하지 않습니다. 또한, mR²AG Instruction-Tuning 데이터세트를 통해 사전 훈련된 MLLM에 쉽게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 논문에서는 mR²AG가 INFOSEEK 및 Encyclopedic-VQA 테스크에서 최신 기술(state-of-the-art) 방법들을 초월한다고 주장합니다. 특히, mR²AG는 LLaVA-v1.5-7B를 베이스로 하여 INFOSEEK 테스트 세트에서 이전 SOTA들보다 10.6%와 15.5%의 성능 향상을 기록했습니다. 따라서, mR²AG는 Visual-dependent 작업에서도 기존 MLLM의 뛰어난 성능을 유지하면서 Knowledge-based 질문에 대한 응답을 성공적으로 개선합니다.



### Evolutionary Automata and Deep Evolutionary Computation (https://arxiv.org/abs/2411.15008)
- **What's New**: 이 논문은 진화 알고리즘(evolutionary algorithms) 및 진화 계산(evolutionary computation)의 새로운 모델인 진화 오토마타(evolutionary automata)를 소개합니다. 진화 오토마타는 복잡한 문제를 해결하는 데에 있어 기존의 알고리즘보다 더 표현력이 뛰어난 모델이며, 자연 진화의 메커니즘을 자기 진화하는 방법으로 연구합니다. 이는 자연 진화와 진화 계산 간의 상호작용에 대한 통찰을 제공하며, 이 연구의 기초를 더욱 강화하려는 시도가 담겨 있습니다.

- **Technical Details**: 논문에서는 진화 계산의 경우를 여러 유형으로 나누어 설명하며, 유전 알고리즘(Genetic Algorithms), 유전 프로그래밍(Genetic Programming), 진화 전략(Evolution Strategies), 진화 프로그래밍(Evolutionary Programming)을 다룹니다. 진화 오토마타는 이러한 기존 접근법들을 포괄하여 상대적으로 더 일반적이고 완전한 이론적 모델을 제공합니다. 이 모델은 무한 대의 세대에서 진화하는 오토마타를 통해 진화의 본질을 직접적으로 나타낼수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 진화 오토마타는 현재 존재하는 모든 진화 알고리즘 유형을 포괄하며, 미해결 최적화 문제뿐만 아니라 계산 불가능한 문제(Turing machine undecidable problems)도 해결 가능한 능력을 입증합니다. 이 논문에서 제시된 새로운 개념들은 진화 알고리즘의 수렴에 대한 간단한 증명을 통해 밝혔다고 할 수 있습니다. 앞으로 이러한 연구는 컴퓨터 과학의 문제 해결에서 새로운 길을 여는 데 기여할 것으로 기대됩니다.



### Large Multi-modal Models Can Interpret Features in Large Multi-modal Models (https://arxiv.org/abs/2411.14982)
- **What's New**: 최근 Large Multimodal Models (LMMs)의 발전은 학계와 산업에서 눈에 띄는 돌파구를 이끌어냈습니다. 본 논문에서는 이러한 LMMs의 내부 신경 표현을 이해하기 위한 초기 단계로, LMMs의 의미론을 식별하고 해석할 수 있는 다재다능한 프레임워크를 제시합니다. Sparse Autoencoder(SAE)를 적용하여 인간이 이해할 수 있는 특징으로 신경 표현을 분해하고, LMMs가 학습한 열린 의미론적 특징의 자동 해석 프레임워크를 도입합니다.

- **Technical Details**: 이 방법론에서는 OpenAI의 연구에서 제시된 SAE 아키텍처를 사용하여 LMMs의 내부 표현을 분해하고 해석합니다. SAE는 두 개의 레이어로 구성된 오토인코더로, Top-K 활성화 함수를 사용하며, 입력 데이터는 다양한 차원의 토큰으로 구성됩니다. 이 과정에서, sparse한 데이터 표현을 통해 다의적 신경 표현을 단일 의미적 특징으로 변환합니다.

- **Performance Highlights**: 실험 결과, LLaVA-NeXT-8B 모델을 분석한 결과 SAE를 통해 인간이 이해할 수 있는 특징이 모델의 행동을 조정하는데 효과적이라는 것을 입증했습니다. 특히, LMMs의 감정적 특징을 식별하고 이 모델들이 감정을 생성하거나 공유할 수 있는 능력을 확인하였습니다. 또한, LMMs의 특정 행동의 원인을 분석하고, 잘못된 결과를 수정하기 위한 전략을 제시하여 LMMs의 내부 메커니즘에 대한 새로운 통찰력을 제공합니다.



### ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos (https://arxiv.org/abs/2411.14901)
- **What's New**: 이 논문에서는 ReVisionLLM이라고 불리는 새로운 recursive vision-language model을 도입합니다. 이 모델은 시간적 정보가 중요한 1시간 분량의 비디오에서 사건을 정확히 위치시키는 능력을 가지고 있습니다. ReVisionLLM은 인류의 검색 전략에서 영감을 받아 넓은 관심 구역에서 시작해 세부적인 사건의 경계를 점진적으로 수정하는 방식으로 작동합니다.

- **Technical Details**: ReVisionLLM은 구조적으로 재귀적(recursive)으로 비디오를 처리하며, 여러 계층에 걸쳐 인식 위계(hierarchical perception)를 도입합니다. 초기에는 관심있는 비디오 구간을 대략적으로 식별하고, 다음 단계로 이동하면서 점점 더 세밀한 시간적 단위로 맞춰 사건의 경계를 정확히 나타냅니다. 또한, 간단한 비디오 클립으로 시작하여 점진적으로 긴 비디오로 확장하는 교육 전략을 사용하여 훈련을 최적화합니다.

- **Performance Highlights**: 모델의 성능은 여러 데이터셋에서 이전의 최첨단 방법들을 상당 폭 초과하면서 입증됩니다. 예를 들어, ReVisionLLM은 MAD 데이터셋에서 기존의 최고 성능 방법보다 2.6% 더 높은 R1@0.1을 기록하였습니다. 이 모델은 평균적으로 기존 VLM 모델보다 43% 적은 프레임을 처리하면서도 높은 정확도와 효율성을 달성합니다.



### Prioritize Denoising Steps on Diffusion Model Preference Alignment via Explicit Denoised Distribution Estimation (https://arxiv.org/abs/2411.14871)
- **What's New**: 본 논문에서는 Denoised Distribution Estimation (DDE)이라는 새로운 기법을 제안합니다. DDE는 기존의 보조 모델이나 수작업 방식에 의존하지 않고, 각 디노이징 단계에서의 관점으로부터 말단의 디노이즈 분포를 직접적으로 추정합니다. 이를 통해 스프의 선호 레이블이 드물게 제공되는 환경에서도 보다 효율적인 신용 할당이 가능해집니다.

- **Technical Details**: DDE는 두 가지 추정 전략, 즉 단계별(stepwise) 및 일괄(single-shot) 추정을 제안합니다. 단계별 추정은 조건부 분포를 기반으로 하여 모델 분포를 추정하는 반면, 일괄 추정은 DDIM 모델링을 사용하여 중간 노이즈 상태를 바탕으로 말단 분포를 직접 평가합니다. 이러한 두 방법을 통합함으로써 모델 추론을 통한 전체 디노이징 궤적의 평가가 가능해집니다.

- **Performance Highlights**: 우리는 SD15와 SDXL에서 DDE를 평가하였고, 그 결과 기존의 보조 모델이 없는 방법들에 비해 뛰어난 성능을 입증하였습니다. DDE는 SD15와 SDXL의 성능 지표를 각각 3.3%에서 6.7%, 1.0%에서 3.1% 향상시켰습니다. 전반적으로 DDE는 기존 접근 방식과 비교할 때 수치적, 질적으로 최고 수준의 성능을 보여주었습니다.



### VisGraphVar: A Benchmark Generator for Assessing Variability in Graph Analysis Using Large Vision-Language Models (https://arxiv.org/abs/2411.14832)
- **What's New**: 이 논문은 LVLMs(Large Vision-Language Models)의 시각 그래프 분석을 위한 새로운 기준 생성기인 VisGraphVar(Visual Graph Variability)를 소개합니다. 이는 서로 다른 그래프 이미지 스타일과 구조를 생성하여 LVLM의 강점과 약점을 체계적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: VisGraphVar는 노드 및 엣지 탐지, 그래프 유형 분류, 세분화, 패턴 인식, 링크 예측, 추론, 일치 확인 등 총 7개의 작업 범주로 구성되어 있습니다. 이 생성기에 의해 만들어진 990개의 그래프 이미지를 통해 6개의 LVLM을 평가하였으며, 줄 무샘과 사고 사슬의 두 가지 다른 프롬프트 전략을 사용하였습니다.

- **Performance Highlights**: 실험 결과, 이미지의 시각적 속성 변화(예: 노드 레이블링 및 레이아웃)와 의도적으로 포함된 시각적 결함(예: 중첩된 노드)이 모델 성능에 상당한 영향을 미치는 것으로 나타났습니다. 이 연구는 LVLM의 시각 그래프 분석 능력을 보다 신뢰성 있고 견고한 시스템으로 발전시키기 위한 통찰력을 제공합니다.



### Fine-Grained Alignment in Vision-and-Language Navigation through Bayesian Optimization (https://arxiv.org/abs/2411.14811)
- **What's New**: 이 논문은 Vision-and-Language Navigation (VLN) 과제에서의 섬세한 조정(fine-grained alignment) 문제를 다룹니다. 기존의 접근 방식들이 언어와 시각적 경로 시퀀스를 조정하는 데 어려움을 겪고 있는 가운데, 저자들은 독창적인 Bayesian Optimization(based) 적대적 최적화(adversarial optimization) 프레임워크를 도입하여 섬세한 대조적 시각 샘플을 생성합니다. 이를 통해 생성된 임베딩(enriched embeddings)이 VLN 과제의 전반적인 성능 강화를 이끈다는 것을 실험을 통해 입증합니다.

- **Technical Details**: 본 연구에서는 기존 VLN 접근 방식의 한계인 대조적 샘플의 품질을 향상시키는 데 초점을 맞추고 있습니다. 특히, 저자들은 Bayesian Optimization(BO)을 활용하여 시각 시퀀스에서 가장 영향력이 큰 요소를 찾아내고 이를 대체하여 섬세한 시각 부정 샘플을 형성하는 방법론을 제안합니다. 이 과정은 훈련 과정에서 신중한 샘플링을 가능하게 하고, 탁월한 경로 선택을 도와 시각-언어 정렬(vision-language alignment) 능력을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크를 통해 훈련된 인코더는 섬세한 시각 정보를 보다 잘 포착하는 것으로 나타났습니다. R2R(Room-to-Room) 및 REVERIE와 같은 일반 VLN 벤치마크에서 성능 향상이 확인되었습니다. 이는 섬세한 샘플(fine-grained samples)의 중요성을 강조하며 기존의 대조적 부정 샘플보다 향상된 성능을 달성하였음을 의미합니다.



### Harlequin: Color-driven Generation of Synthetic Data for Referring Expression Comprehension (https://arxiv.org/abs/2411.14807)
Comments:
          Accepted to ICPR 2024

- **What's New**: 이번 연구에서는 Referring Expression Comprehension (REC) 작업을 위한 새로운 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 텍스트 및 시각적 변수를 모두 고려하여 인공 데이터를 생성합니다. 새롭게 생성된 데이터 세트는 Harlequin이라는 이름을 가지고 있으며, 1백만 개 이상의 쿼리로 구성되어 있습니다. 이 연구는 데이터 수집 및 주석 작업이 수작업으로 이루어지는 것을 피할 수 있음을 강조합니다.

- **Technical Details**: 저자들은 두 가지 주요 모듈로 구성된 파이프라인을 설계했습니다. 첫 번째 모듈인 Annotation Generation Engine은 일관된 바운딩 박스 주석이 있는 새로운 Referring Expressions를 생성합니다. 두 번째 모듈인 Image Generation Engine은 이전 단계에서 얻은 주석 정보를 바탕으로 새로운 이미지를 생성합니다. 이렇게하여 개발된 Harlequin 데이터 세트는 완전히 합성 공정으로 생성된 REC 작업을 위한 첫 번째 데이터 세트입니다.

- **Performance Highlights**: 실험 결과, Harlequin 데이터 세트를 이용한 사전 학습이 실제 데이터에서의 성능을 향상시키는 데 기여함을 보여주었습니다. 이를 통해 주석 수집에서의 노력과 오류를 줄이는 것이 가능해졌습니다. 저자들은 데이터 및 코드를 공개하여 향후 연구자들이 이 결과를 토대로 추가 연구를 진행할 수 있도록 하고 있습니다.



### Continual SFT Matches Multimodal RLHF with Negative Supervision (https://arxiv.org/abs/2411.14797)
- **What's New**: 이 논문에서는 비전-언어 모델(VLM)의 선호 정렬 과정에서 다중 모달 RLHF(강화 학습에서 인간 피드백 사용)의 고유한 가치가 거부된 응답의 로그리트를 통한 부정 감독에 있음을 관찰하고, 새로운 부정 감독 미세 조정(nSFT) 접근 방식을 제안합니다. nSFT는 VLM과 간단한 SFT 손실(Supervised Finetuning Loss)을 지속적으로 정렬하며, 기존의 슬롯 방식에 비해 메모리 효율성이 뛰어난 장점을 가지고 있습니다. 이는 여러 데이터 세트 및 평가 지표와 비교하여 nSFT가 더 나은 성능을 발휘함을 수치적으로 보여줍니다.

- **Technical Details**: 논문의 핵심은 nSFT가 RLHF 최적화에서 부정 감독을 분리하여 훈련을 더 효율적으로 만들 수 있다는 점입니다. 다중 모달 RLHF에서는 메모리를 2개 또는 4개의 모델을 사용해야 하지만, nSFT는 오직 하나의 모델에서 수행된다는 점이 강조됩니다. 이 논문은 LLM(예: GPT-4)를 사용하여 부정 감독의 불일치를 식별하고 이미지 관련 대화를 구성하여 부정 응답으로부터 자기 견책의 학습을 돕습니다.

- **Performance Highlights**: nSFT는 다양한 실험을 통해 SFT의 한계를 보완하고, 여러 평가 지표에서 최고의 성과를 달성합니다. 특히 LLaVA-NeXT와 같은 강력한 VLM에서도 가설이 유지됩니다. 시험 결과는 nSFT가 기존의 다중 모달 RLHF 방법과 견줄 만큼 강력하며, 향후 연구에 자극이 되길 바랍니다.



### VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection (https://arxiv.org/abs/2411.14794)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 VideoQA(비디오 질문-답변) 작업의 한계를 극복하기 위해 VideoEspresso라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 중요한 공간적 세부정보와 시간적 일관성을 보존하며, 중간 추론 단계를 포함한 다중 모드 주석(annotation)을 특징으로 합니다. 또한, 저자들은 Hybrid LVLMs Collaboration 프레임워크를 제안하여 질의에 적합한 핵심 프레임을 선택하고, 체인 오브 사고(reasoning) 접근법을 통해 비디오 내용을 기반으로 논리적 관계를 추출합니다.

- **Technical Details**: VideoEspresso의 구축은 자동화된 파이프라인을 통해 이루어지며, 이를 통해 효율적으로 QA 쌍을 생성합니다. 처음에는 LVLM을 사용하여 비디오 프레임을 언어 공간으로 매핑한 후, 의미론적 유사성을 바탕으로 중복 프레임을 제거합니다. 이어서 GPT-4o를 활용하여 QA 쌍을 생성하고 저품질 데이터를 필터링합니다. 중간 논리 과정을 확장하기 위해 Video Chain-of-Thought 주석을 도입하여 이러한 정보가 QA 쌍 형성에 기여하도록 설계되었습니다.

- **Performance Highlights**: 새롭게 제안된 평가 기준을 기반으로 9개의 인기 LVLM을 대상으로 14개의 작업에서 성능을 평가한 결과, 제안된 방법이 대부분의 작업에서 기존 방법들을 초월하는 것으로 나타났습니다. 이러한 평가를 통해 VideoEspresso의 고품질 VideoQA 쌍이 복합적인 비디오 추론 작업에서 우수한 성능을 발휘함을 보였습니다. 코드 및 데이터셋은 추후 공개될 예정입니다.



### IRLab@iKAT24: Learned Sparse Retrieval with Multi-aspect LLM Query Generation for Conversational Search (https://arxiv.org/abs/2411.14739)
- **What's New**: 2024년 Interactive Knowledge Assistant Track (iKAT)은 사용자 맞춤형 지식을 바탕으로 대화형 어시스턴트를 발전시키는 데 주목하고 있습니다. 개인 텍스트 지식 기반(PTKB)과 함께 다양한 Conversational AI 작업, 예를 들어 passage ranking 및 response generation을 통합합니다. 특히, Multi-aspect query generation을 통해 대화의 맥락을 해결하고, Learned Sparse Retrieval 및 SPLADE 아키텍처와 강력한 cross-encoder 모델을 결합하여 성능을 향상시키려는 시도가 돋보입니다.

- **Technical Details**: 본 논문에서는 세 가지 주요 작업에 대해 설명합니다: Passage Ranking, Response Generation 및 PTKB Classification입니다. Passage ranking은 각 사용자 발화에 대한 적절한 passage를 검색하고 순위를 매기는 것이며, Response Generation은 해당 passage를 기반으로 자연스러운 응답을 생성하는 것을 목표로 합니다. PTKB Classification은 PTKB의 각 진술을 각 대화 전환에 대해 관련성/비관련성으로 분류하는 작업입니다.

- **Performance Highlights**: 다양한 실험을 바탕으로 Multi-aspect query generation이 향상된 성능을 제공하는 것을 확인했습니다. LLMs가 적절한 문맥의 모호성을 해소하고 사용자 맞춤형 쿼리 재작성 과정을 통해 인간의 재작성 성능을 초월하는 결과를 보여줍니다. 또한, 제안된 방법이 기존의 interleaving 전략보다 더 효과적인 회귀 과정을 통해 성능을 강화한다고 보고되었습니다.



### Evaluating and Advancing Multimodal Large Language Models in Ability Lens (https://arxiv.org/abs/2411.14725)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 시각 인식(perception) 능력 평가를 위한 새로운 벤치마크인 AbilityLens를 소개합니다. 이전의 평가 기준들이 여러 다양한 질문 유형, 도메인 및 평가 지표로 인해 평가에 변동성이 있었으나, AbilityLens는 여섯 가지 주요 인식 능력을 아우르는 통합된 평가 체계를 제공합니다. 또한, 모델의 정확성과 안정성을 동시에 평가하는 시스템을 새롭게 도입해 모델 발전에 실질적인 방향성을 제시합니다.

- **Technical Details**: AbilityLens는 11개의 기존 벤치마크에서 데이터를 수집하여 각 인식 능력에 대해 1,000개 이상의 테스트 샘플을 포함하여 총 12,000개의 샘플을 구성합니다. 이 평가 방안은 정확성(accuracy)과 안정성(stability)을 측정하기 위해 개별 하위 지표(sub-metric)를 통합하여 단일 결과 점수(composite score)를 산출합니다. 또한, 온라인 및 오프라인 두 가지 평가 모드를 통해 모델 학습 동향을 모니터링하고, 각 인식 능력에 대해 모델을 비교합니다.

- **Performance Highlights**: AbilityLens를 사용하여 현재 MLLMs의 성능을 분석한 대표 사례로, LLaVA-OV-72b와 Qwen2-VL-72b 모델이 상업적 모델에 비해 정확성에서 우수함을 확인하였지만, 안정성은 여전히 부족함을 나타냅니다. 또한, 각 인식 능력 사이의 상이한 개선 곡선과 초기 수렴(early convergence) 품질 혼란(ability conflict) 현상이 관찰되었습니다. 능력 충돌을 완화하기 위해 초기 훈련 단계의 최적 능력 체크포인트를 통합한 모델 병합 방법을 제안하여 성능 저하를 최소화하는 효과를 확인하였습니다.



### FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data (https://arxiv.org/abs/2411.14717)
- **What's New**: 이 연구에서는 Multimodal Large Language Models (MLLMs)와 관련하여 새로운 벤치마크를 소개합니다. 이는 Federated Learning (FL) 프레임워크 내에서 MLLM의 다양한 다운스트림 작업을 평가하기 위한 허브로 기능합니다. 또한, 네 가지 대표적인 FL 방법을 통합하는 일반적인 FedMLLM 프레임워크를 개발하여 다중 모드 이질성 문제를 해결하고자 합니다.

- **Technical Details**: MLLM은 강화된 멀티모달 데이터 처리 기술을 기반으로 하며, FL을 사용하여 개인 데이터를 포함한 훈련 데이터의 범위를 확장합니다. FL의 다양한 기법과 함께, 두 가지 모달리스틱(모달-무관) 전략이 설계되어 다양한 모달 이질성을 관리합니다. 이러한 접근법은 MLLMs가 실제 응용 프로그램에서 다중 모드 입력을 지원하도록 하고, 모달 이질성에 따른 편향을 최소화하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과에 따르면, FL 패러다임이 MLLMs의 성능을 향상시키며 훈련 데이터의 범위를 넓혀주고, 다중 모드 이질성 문제를 완화하는 긍정적인 영향을 미쳤습니다. 연구에서 제안된 방법론은 2개의 데이터셋과 5개의 비교 기준을 포함하여 종합적인 모달 시나리오에서 효과적으로 테스트되었습니다. FL을 적용한 MLLMs의 성능은 이전의 로컬 훈련 방식에 비해 유의미한 개선을 나타냈습니다.



### Understanding LLM Embeddings for Regression (https://arxiv.org/abs/2411.14708)
Comments:
          15 pages, 13 figures

- **What's New**: 이 논문은 LLM(대형 언어 모델)의 임베딩을 활용한 회귀 분석에 대한 최초의 포괄적인 연구 중 하나로, 전통적인 피처 엔지니어링 대신 LLM 임베딩이 고차원 회귀 작업에서 더 효과적일 수 있음을 보입니다. 연구 결과, LLM 임베딩은 숫자 데이터에 대해 Lipschitz 연속성을 보존하며, 이러한 특성이 회귀 성능을 향상시키는 데 기여함을 밝혀냅니다.

- **Technical Details**: 회귀 작업은 주어진 입력 공간에서 스칼라 값을 예측하기 위한 함수와 관련된 데이터 셋으로 구성됩니다. 본 연구는 LLM 임베딩이 흐름이 있는 문자를 구성하는 피처로 사용될 때의 성능을 분석합니다. LLM은 특정 문제에 대해 전통적인 피처 방식과 비교할 때 고차원 데이터에서도 강력한 회귀 성능을 유지하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 연구 결과, LLM 임베딩을 이용한 회귀 성능은 특히 고차원 데이터에 대해 상당한 이점을 제공하며, 전통적인 방식에서는 성능 저하가 예상됩니다. 또한, 모델의 크기와 언어 이해력 같은 요소는 예상과 다르게 회귀 성능에 복잡한 영향을 미치는 것으로 나타났습니다.



### Whats in a Video: Factorized Autoregressive Decoding for Online Dense Video Captioning (https://arxiv.org/abs/2411.14688)
- **What's New**: 본 논문에서는 비디오의 내용을 자동으로 밀도 있게 설명하는 캡션을 생성하는 새로운 접근 방식을 제안합니다. 기존의 모델은 비디오 전체를 한 번에 처리해야 했지만, 우리는 미래 프레임에 접근할 필요 없이 빈번하고 구체적인 캡션을 출력하는 효율적인 온라인 접근 방식을 개발했습니다. 이 모델은 시간 세그먼트마다 시각적 특징의 시퀀스를 모델링하는 새로운 autoregressive factorized decoding 아키텍처를 사용합니다.

- **Technical Details**: 제안된 모델은 비디오의 현재 정보와 이전 컨텍스트를 연결하는 메모리와 같은 메커니즘을 활용하여 밀집한 캡션을 생성합니다. 각 비디오 세그먼트의 정보를 처리하고 출력하는 과정에서 단일 디코더가 여러 세그먼트를 처리하도록 설계되어 계산 비용이 비디오 길이에 비례하여 선형적으로 증가합니다. 이 접근 방식은 더 긴 비디오에서 효율적으로 작동하며, 더 많은 캡션과 프레임을 처리하는 데 필요한 표준 하드웨어 메모리 용량을 초과하지 않도록 설계되었습니다.

- **Performance Highlights**: 우리 모델은 ViTT, YouCook2, ActivityNet과 같은 잘 알려진 밀집 비디오 캡셔닝 벤치마크에서 평가되었습니다. 실험 결과, 모두에서 최신 기법들을 초과하는 성능을 보였으며, 일부 경우에는 개선 폭이 매우 큰 것으로 나타났습니다. 또한 이 모델은 같은 성능을 유지하면서 20% 더 적은 연산 비용을 사용합니다.



### Towards Knowledge Checking in Retrieval-augmented Generation: A Representation Perspectiv (https://arxiv.org/abs/2411.14572)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템에서 내부 지식과 외부 지식을 효과적으로 통합하는 데 도전과제를 다루고 있습니다. RAG의 성능을 극대화하기 위해, 연구팀은 LLM(대형 언어 모델)의 표현을 활용한 지식 확인 방법을 제안합니다. 이로 인해 노이즈가 있는 지식 데이터베이스에서도 큰 성능 향상을 보여주며, RAG 시스템의 신뢰성과 효율성을 강화하는 새로운 통찰을 제공합니다.

- **Technical Details**: 이 연구는 RAG 시스템에서의 지식 확인을 위하여 네 가지 주요 작업을 설계합니다. 첫 번째, 사용자 쿼리에 대한 내부 지식 확인(Internal Knowledge Checking)을 수행합니다. 두 번째, 외부 지식의 유용성을 확인하는 Helpfulness Checking을 수행하며, 이는 내부 지식이 있는 경우와 없는 경우로 나뉩니다. 마지막으로, 내부 지식과 외부 정보 간의 모순을 확인하는 Contradiction Checking을 수행합니다. 연구팀은 LLM 표현을 바탕으로 한 분류기를 개발하여 지식을 필터링합니다.

- **Performance Highlights**: RAG 성능 향상의 주요 결과는 표현 기반 방법을 통한 지식 필터링의 도입에서 나타납니다. 실험 결과, 모순되거나 관련 없는 정보를 간단히 필터링하는 것만으로도 RAG의 성능이 상당히 개선됩니다. 이는 노이즈가 포함된 지식 데이터베이스를 처리할 때에도 유효하며, RAG 시스템의 전반적인 품질을 향상시키는데 기여합니다.



### Assessment of LLM Responses to End-user Security Questions (https://arxiv.org/abs/2411.14571)
Comments:
          18 pages, 1 figure, 8 tables

- **What's New**: 이번 연구는 LLM(large language models)들이 사용자 보안 질문에 대한 응답 품질을 평가하기 위해 수행된 첫 번째 광범위한 연구입니다. 이전 연구들은 주로 보안 신화를 반박하는 LLM의 능력에 초점을 맞춘 반면, 본 연구는 LLM의 보안 문제에 대한 폭넓은 질문들을 다루었습니다. 이 연구는 사용자와 모델 개발자 모두에게 실질적인 개선 방향과 전략을 제시하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 GPT, LLaMA, Gemini라는 세 가지 LLM 모델이 생성한 1,244개의 보안 관련 응답을 정성적으로 평가했습니다. 이 과정에서 각 모델의 응답 품질, 정확성, 완전성 및 관련성 뿐만 아니라 정보 전달 스타일을 분석했습니다. 특히, 직접성이 떨어지고 관련되지 않은 응답을 제공하는 경향을 보였으며, 이는 사용자와 LLM 간의 효과적인 소통을 저하시킬 수 있습니다.

- **Performance Highlights**: LLM들은 일반적인 보안 질문에 대해 사용자 친화적인 고품질 정보를 제공하는 데 성공했습니다. 그러나 응답의 오류와 제한들이 지속적으로 발견되었으며, 이는 사용자가 LLM을 보다 효과적으로 활용하는 데 장애가 됩니다. 특히 구식의 패스워드 권장 사항이나, 피싱 및 HTTPS URL 간의 연관을 잘못 연결하는 등의 문제가 보고되었습니다.



### Reducibility among NP-Hard graph problems and boundary classes (https://arxiv.org/abs/2411.14553)
Comments:
          9 pages, 6 figures

- **What's New**: 본 논문에서는 NP-hard 그래프 문제들 간의 경계(class)의 관계를 연구하는 새로운 방법을 제시합니다. 특정 NP-hard 문제 Π가 다른 문제 Γ로 줄여질 수 있을 때, Π의 경계 클래스가 Γ의 경계 클래스로 변환됩니다. 이러한 관계를 통해 새로운 경계 클래스들을 발견하는 방법론이 제안되었습니다. 이에 따라, vertex-cover, clique, traveling-salesperson 등의 문제에서 이전에 알려지지 않았던 경계 클래스들이 도출되었습니다.

- **Technical Details**: 경계 클래스(boundary class)의 개념은 NP-hard 그래프 문제에 대한 인스턴스들의 쉽고 어려운 경계를 구분하기 위해 사용됩니다. 이 논문에서는 bijective한 줄임법이 존재하고 해당 줄임법이 유전자형(hereditary classes)의 그래프 사이에서 작동할 때 경계 클래스 간의 관계를 공식적으로 정의합니다. 이를 통해, 주어진 NP-hard 문제의 경계 클래스 X가 있을 경우, 다른 NP-hard 문제 Γ에서 경계 클래스를 찾을 수 있는 방법이 제공됩니다.

- **Performance Highlights**: 이 새로운 이론의 적용을 통해 여러 NP-hard 그래프 문제에서 처음으로 경계 클래스가 정의되었습니다. 이는 기존의 경계 클래스와의 연결 혹은 줄임법을 통해 보다 많은 NP-hard 문제들에 대한 경계 클래스를 발견할 수 있는 강력한 방법이 됨을 시사합니다. 연구 결과는 NP-hard 문제 해결을 위한 이론적 토대 강화에 기여할 것으로 기대됩니다.



### Towards a Middleware for Large Language Models (https://arxiv.org/abs/2411.14513)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)이 기업에 배포되고 활용될 수 있는 미들웨어 시스템 아키텍처에 대한 비전을 제시합니다. LLM 기술이 점차 성숙해짐에 따라, 기업들은 클라우드 제공업체로부터 독립적으로 자체 호스팅 솔루션을 구축하려는 강력한 동기를 얻고 있습니다. 이러한 자가 호스팅은 개인 정보 보호, 비용 절감 및 특정 도메인에 맞게 모델을 조정할 수 있는 가능성에서 비롯됩니다.

- **Technical Details**: LLM의 독립적 호스팅은 기존 시스템과의 통합 문제로 인해 어려움이 많습니다. 이 논문에서는 사용자의 자연어 프롬프트를 통해 기업 애플리케이션과의 상호작용을 가능하게 하며, 복잡한 다단계 대화 가능성을 가진 LLM이 특정 도메인에서 중요한 역할을 할 수 있다는 점을 강조합니다. LLM은 단순한 응답 생성에는 그칠 수 없으며, 외부 서비스와의 협력이 필수적입니다.

- **Performance Highlights**: LLM은 더 나은 신뢰성과 정확성을 제공하기 위해 외부 서비스와의 협력을 통해 결정론적 보장을 제공해야 합니다. 이 연구는 LLM의 잠재력을 최대화하기 위한 통합 레이어의 중요성을 강조하며, 이를 통해 기업 내에서 LLM의 배포 및 활용을 용이하게 할 수 있는 시스템의 설계를 제안합니다.



### FuseGPT: Learnable Layers Fusion of Generative Pre-trained Transformers (https://arxiv.org/abs/2411.14507)
- **What's New**: FuseGPT는 감축된 transformer 블록을 재활용하여 모델 성능을 회복할 수 있는 혁신적인 방법론을 제안합니다. 기존의 모델 압축 기법들이 제거 후의 성능 저하 문제를 해결하지 못했던 반면, FuseGPT는 Macro Influence(MI)라는 새로운 메트릭을 도입하여 블록의 중요성을 평가합니다. 이 방법은 불필요한 블록의 파라미터를 인접 블록의 레이어에 주입하여 블록 복원 과정을 통해 성능 회복을 가능케 합니다.

- **Technical Details**: FuseGPT에서는 transformer 블록의 중요도를 평가하기 위해 블록 제거 후 정보 손실을 계산하여 장기적인 영향력을 측정합니다. 이 방법론은 가벼운 그룹 수준의 파인튜닝을 통해 인접 블록의 해당 레이어에 불필요한 블록의 파라미터를 반복적으로 주입하고 최적화합니다. 이를 통해 블록의 이미 존재하는 지식을 활용하여 성능 회복을 극대화할 수 있습니다.

- **Performance Highlights**: FuseGPT는 적은 양의 데이터를 사용하여 perplexity와 zero-shot 과제 성능 모두에서 이전 연구들을 초월하는 결과를 보였습니다. 특히 대형 언어 모델뿐 아니라 대형 다중 모달 모델에서도 효과적으로 작용하여 뛰어난 성능을 발휘하고 있습니다. 실험 결과, FuseGPT는 강력한 차세대 성능을 기록하였으며, 향후 모델 압축 연구에서 유망한 방향성을 제시하고 있습니다.



### AI Ethics by Design: Implementing Customizable Guardrails for Responsible AI Developmen (https://arxiv.org/abs/2411.14442)
- **What's New**: 이 논문은 AI 시스템을 위한 윤리적 가드레일 프레임워크의 발전을 탐구하고, 다양한 사용자 가치와 기본 윤리에 맞춘 맞춤형 가드레일의 중요성을 강조합니다. AI 윤리의 도전 과제를 해결하기 위해 규칙, 정책 및 AI 보조 도구를 통합하는 구조를 제안하며, 이를 통해 책임 있는 AI 행동을 보장하는 방법을 논의합니다. 기존의 최신 가드레일과 비교하여 지속적인 개선 및 사용자 자율성 증진을 목표로 합니다.

- **Technical Details**: AI의 윤리는 알고리즘적 편향, 프라이버시, 공정성, 자율 시스템, 정렬(alignment) 등의 개념과 관련된 문제를 포함하는 최근의 하위 분야입니다. 이 논문은 윤리적 문제를 해결하기 위한 가드레일의 필요성을 강조하며, 특히 대규모 언어 모델(LLM)에서 투명성과 공정성을 보장하기 위한 여러 방법론을 제시합니다. 또한, 다양한 윤리적 관점이 AI 설계 및 생성에 미치는 영향을 논의합니다.

- **Performance Highlights**: 이 연구는 사용자 가치에 맞춘 맞춤형 윤리적 정책 세트를 설정하는 프레임워크를 제안하여, 윤리적 다원성(ethical pluralism)에 부합하는 유연하고 적응 가능한 솔루션을 제공합니다. 이를 통해 AI 시스템의 투명성, 사용자 자율성 및 지속적인 개선이 가능해집니다. 최종적으로, AI의 목표를 인간의 가치와 정렬시키기 위한 접근 방식도 탐구하여, 안전성과 호환성을 모두 충족하는 AI 시스템의 필요성을 강조합니다.



New uploads on arXiv(cs.IR)

### Multi-granularity Interest Retrieval and Refinement Network for Long-Term User Behavior Modeling in CTR Prediction (https://arxiv.org/abs/2411.15005)
Comments:
          KDD2025

- **What's New**: 이번 논문에서는 Click-Through Rate (CTR) 예측에서의 사용자 행동 모델링을 개선하기 위한 새로운 접근법인 Multi-granularity Interest Retrieval and Refinement Network (MIRRN)을 제안합니다. 기존 방법들이 사용자 관심사를 충분히 포착하지 못하는 문제를 해결하기 위해, MIRRN은 다양한 시간 대의 행동을 기반으로 한 쿼리를 구성하여 사용자의 관심사를 다중 세분화하여 나타냅니다. 이를 통해 사용자 행동 데이터의 복잡성을 효과적으로 처리함으로써 CTR 예측의 정확성을 높이려는 목표를 가지고 있습니다.

- **Technical Details**: MIRRN은 두 가지 주요 모듈로 구성되어 있습니다: 다중 세분화 관심 검색 모듈 (MIRM)과 행동 순서 정제 모듈 (BSRM)입니다. MIRM은 SimHash를 기반으로 한 빠른 검색 방법을 통해 행동 시퀀스에서 서브시퀀스를 추출하여 다양한 관심사를 포착합니다. BSRM은 다중 헤드 푸리에 변환기(MHFT)를 사용하여 서브시퀀스 내의 관계 정보를 효과적으로 추출하고, 이는 복잡한 연산을 대체하는 FFT를 통해 가능해집니다.

- **Performance Highlights**: MIRRN은 다양한 실험을 통해 기존 최신 기술보다 향상된 성능을 보여줍니다. 특히 인기 음악 스트리밍 앱에서 실시한 A/B 테스트 결과, MIRRN이 평균 청취 노래 수를 1.32% 증가시키고 평균 청취 시간도 0.55% 증가시킨 것으로 나타났습니다. 이러한 결과는 MIRRN이 실제 산업 환경에서도 효과적으로 작동함을 입증하고 있습니다.



### GOT4Rec: Graph of Thoughts for Sequential Recommendation (https://arxiv.org/abs/2411.14922)
- **What's New**: 이 논문에서는 GOT4Rec이라는 새로운 순차 추천 방법을 제안합니다. 이 방법은 'graph of thoughts' (GoT) 프롬프트 전략을 활용하여 사용자의 행동 시퀀스에서 사용자 관심 정보를 포괄적으로 수집합니다. GOT4Rec는 단기 관심, 장기 관심 및 협업 정보를 효과적으로 결합하여 더욱 정확한 추천을 제공합니다.

- **Technical Details**: GOT4Rec는 사용자의 과거 상호작용 시퀀스를 기반으로 추천을 생성하며, 각 시퀀스에서 중요한 세 가지 유형의 정보를 추출합니다. 이러한 정보에는 단기 및 장기 관심, 그리고 유사한 선호를 가진 다른 사용자로부터의 협업 정보가 포함됩니다. 기존의 방법들과 달리, GOT4Rec는 사용자 시퀀스를 전체로 간주하는 대신 다층적인 정보 처리를 통해 LLM의 추론 능력을 극대화합니다.

- **Performance Highlights**: 실험 결과 GOT4Rec는 세 가지 데이터셋에서 전통적인 신경망 기반 모델들과 다른 프롬프트 전략들보다 뛰어난 성능을 보였습니다. GOT4Rec의 도입으로 LLM이 사용자 시퀀스 내 다양한 정보를 보다 효율적으로 활용할 수 있게 되어, 정확한 추천 및 포괄적인 설명이 가능해졌습니다. 이는 GOT4Rec가 기존의 최첨단 모델들과 비교하여 우수함을 확인시키는 결과입니다.



### A Reproducibility and Generalizability Study of Large Language Models for Query Generation (https://arxiv.org/abs/2411.14914)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용한 Boolean 쿼리 생성의 체계적 검토를 다룹니다. 연구자들이 청구서를 작성하는 데 필요한 지루한 작업을 지원하여 문헌 검토 과정의 비효율성을 줄일 수 있는 잠재력을 가지고 있습니다. 위의 내용을 바탕으로 ChatGPT의 성능을 Mistral 및 Zephyr와 같은 오픈 소스 모델과 비교하여 포괄적인 분석을 제공합니다.

- **Technical Details**: 연구는 이미 정의된 LLM을 사용하여 특정 검토 주제에 대한 Boolean 쿼리를 자동으로 생성하는 파이프라인을 구현하였습니다. 이를 통해 ChatGPT의 쿼리 생성 결과가 재현 가능하고 일관성이 있는지를 평가하고, 오픈 소스 모델의 효율성을 분석하여 일반화된 결과를 도출합니다. 또한, LLM의 특성과 문헌 검색 과제에 있어서의 전반적인 한계 및 과제를 논의합니다.

- **Performance Highlights**: 연구 결과는 LLM의 정보 검색 및 문헌 검토 자동화 영역에서의 장점과 한계를 강조합니다. 구체적으로, LLM이 복잡한 Boolean 쿼리를 생성하는 프로세스에서 신뢰성 및 재현 가능성 문제를 보이고 있으며, 이러한 점검은 연구 신뢰성을 확보하는 데 필수적입니다. 연구자들은 ChatGPT와 같은 LLM을 검색 전략 구성에 의존하지 말고 그 한계를 이해하고 충분한 주의를 기울일 것을 권장합니다.



### The 1st Workshop on Human-Centered Recommender Systems (https://arxiv.org/abs/2411.14760)
Comments:
          Workshop at TheWebConf 2025

- **What's New**: 이 워크샵은 Human-Centered Recommender Systems(HCRS)의 발전을 중심으로 한 새로운 연구 동향을 탐구하기 위한 플랫폼을 제공합니다. HCRS는 인간의 필요와 가치를 최우선으로 고려하여 설계된 추천 시스템을 의미하며, 사용자 친화적이고 사회적으로 책임 있는 시스템 개발에 중점을 두고 있습니다. 다양한 주제와 평가 방법을 논의하며, HCRS의 구축 및 개선 방법에 대한 의견을 나눌 예정입니다.

- **Technical Details**: 이 워크샵에서는 추천 시스템의 강건성, 프라이버시, 투명성, 공정성, 다양성, 윤리적 고려사항 등 다양한 주제를 다룹니다. 참여자들은 사용자 만족도와 신뢰를 측정하는 혁신적인 평가 지표를 포함한 다양한 평가 방법을 탐색할 기회를 가지게 됩니다. Human-Centered 접근 방식은 참여 설계 방법론을 통해 사용자와 협력하여 시스템을 발전시키는 것을 강조합니다.

- **Performance Highlights**: 이번 워크샵은 연구자들이 새로운 이론적 틀과 학제 간 접근 방식을 제안하도록 장려하며, 대규모 언어 모델과 같은 최첨단 기술을 활용하여 HCRS의 인간 중심적 특성을 향상시키는 방법을 모색합니다. 키노트 강연 뿐만 아니라 패널 토론 및 연구 발표 세션을 통해 참여자들 간의 활발한 논의와 혁신적인 아이디어 교환을 할 수 있는 기회를 제공하여, HCRS의 발전에 기여할 것으로 기대합니다.



### IRLab@iKAT24: Learned Sparse Retrieval with Multi-aspect LLM Query Generation for Conversational Search (https://arxiv.org/abs/2411.14739)
- **What's New**: 2024년 Interactive Knowledge Assistant Track (iKAT)은 사용자 맞춤형 지식을 바탕으로 대화형 어시스턴트를 발전시키는 데 주목하고 있습니다. 개인 텍스트 지식 기반(PTKB)과 함께 다양한 Conversational AI 작업, 예를 들어 passage ranking 및 response generation을 통합합니다. 특히, Multi-aspect query generation을 통해 대화의 맥락을 해결하고, Learned Sparse Retrieval 및 SPLADE 아키텍처와 강력한 cross-encoder 모델을 결합하여 성능을 향상시키려는 시도가 돋보입니다.

- **Technical Details**: 본 논문에서는 세 가지 주요 작업에 대해 설명합니다: Passage Ranking, Response Generation 및 PTKB Classification입니다. Passage ranking은 각 사용자 발화에 대한 적절한 passage를 검색하고 순위를 매기는 것이며, Response Generation은 해당 passage를 기반으로 자연스러운 응답을 생성하는 것을 목표로 합니다. PTKB Classification은 PTKB의 각 진술을 각 대화 전환에 대해 관련성/비관련성으로 분류하는 작업입니다.

- **Performance Highlights**: 다양한 실험을 바탕으로 Multi-aspect query generation이 향상된 성능을 제공하는 것을 확인했습니다. LLMs가 적절한 문맥의 모호성을 해소하고 사용자 맞춤형 쿼리 재작성 과정을 통해 인간의 재작성 성능을 초월하는 결과를 보여줍니다. 또한, 제안된 방법이 기존의 interleaving 전략보다 더 효과적인 회귀 과정을 통해 성능을 강화한다고 보고되었습니다.



### LIBER: Lifelong User Behavior Modeling Based on Large Language Models (https://arxiv.org/abs/2411.14713)
- **What's New**: 본 논문에서는 Lifelong User Behavior Modeling (LIBER) 프레임워크를 제안하여 LLM(large language models) 기반의 추천 시스템에서의 문제를 해결하고자 합니다. LIBER는 사용자 행동 시퀀스의 이해 문제를 접근하며, 사용자 행동을 효율적으로 처리하기 위해 세 개의 모듈을 포함합니다: User Behavior Streaming Partition (UBSP), User Interest Learning (UIL), User Interest Fusion (UIF). 이를 통해 사용자의 동적 관심사를 효과적으로 포착하고, 추천 성능을 개선하고자 합니다.

- **Technical Details**: LIBER는 짧은 사용자 행동 파티션을 생성하는 Incremental Paradigm을 통해 긴 사용자 행동 시퀀스를 처리합니다. 먼저 UBSP 모듈이 각 사용자의 행동 시퀀스를 짧은 파티션으로 나누고, UIL 모듈이 LLM을 활용하여 이러한 파티션에서 통찰력을 추출합니다. UIF 모듈은 이러한 텍스트 출력을 통합하여 추천 모델에 삽입할 수 있는 포괄적인 표현을 구축합니다.

- **Performance Highlights**: LIBER는 Huawei의 음악 추천 서비스에 적용되어 사용자 재생 횟수와 재생 시간을 각각 3.01% 및 7.69% 증가시켰습니다. 공공 데이터셋과 산업 데이터셋에서의 오프라인 실험 결과, LIBER의 성능은 기존 모델보다 우수함을 입증했습니다. 추가적으로 온라인 A/B 테스트를 통해 LIBER의 효과성과 적용 가능성을 확인하였습니다.



### G-RAG: Knowledge Expansion in Material Scienc (https://arxiv.org/abs/2411.14592)
- **What's New**: 이번 연구에서는 Material Science 분야의 정보 검색 시스템에서 발생하는 여러 문제를 해결하기 위한 Graph RAG 방법론을 제안합니다. 기존의 Retrieval-Augmented Generation (RAG) 접근 방식의 한계인 구식 정보, 허위 정보, 해석 가능성의 부족 등을 극복하기 위해, 그래프 데이터베이스를 통합하여 키 엔티티를 활용한 정교한 정보 검색을 가능하게 합니다. 이러한 방법은 문서 간의 관계를 캡처하여 정보 검색의 정확성과 맥락 이해를 개선하는 데 기여합니다.

- **Technical Details**: 이 연구에서 제안된 Graph RAG는 MatID라는 키 엔티티를 문장에서 추출하고, 이 정보를 외부 위키피디아 데이터베이스에 쿼리하여 관련 정보를 검색함으로써 향상된 응답 생성을 구현합니다. G-RAG 시스템은 엔티티 링크와 관계 추출 기능을 포함하며, 구문 분석 모듈을 통해 정보 검색을 수행합니다. 이 구조는 LLM(대형 언어 모델)에서 더 깊은 의미적 이해를 가능하게 합니다.

- **Performance Highlights**: G-RAG 접근 방식은 Material Science와 같은 정확한 정보 검색이 중요한 도메인에서 성능을 크게 향상시키는 것으로 나타났습니다. 특히, 정보의 정확성과 관련성을 유지하면서 LLM이 생성하는 응답의 질을 크게 개선하여, 복잡한 데이터를 효과적으로 관리할 수 있는 능력을 발휘합니다. 또한, 다양한 자연어 처리(NLP) 작업에서도 기존 기술들보다 우수한 성과를 나타내는 것으로 평가받고 있습니다.



### Variable Extraction for Model Recovery in Scientific Literatur (https://arxiv.org/abs/2411.14569)
- **What's New**: 본 논문은 전 세계 연간 500만 편 이상의 학술지가 발표되고 있어 과학적 출력의 일부만을 파악하는 것이 어렵다는 문제를 다룹니다. 이는 과학 문헌을 구성하는 텍스트, 그래프, 차트, 코드, 모델 및 데이터셋과 같은 아티팩트를 탐색하고 해석할 수 있는 방법이 필요함을 시사합니다. 추가로, 논문에서는 전염병 연구에서 수학적 모델 변수 추출을 위한 다양한 방법을 평가하고 있습니다.

- **Technical Details**: 특히, 변수 추출은 기본적인 작업으로 보이지만 과학 문헌에서 모델을 복구하는 데 중요한 역할을 합니다. 연구진은 수작업으로 주석이 달린 변수 설명 및 과학 논문에서 추출된 변수 값을 포함하는 기준 데이터셋을 소개합니다. 이 데이터셋 기반으로 LLMs(거대한 언어 모델)와 규칙 기반 정보 추출 시스템을 활용한 여러 기준 방법을 제시하며, LLM 기반 솔루션이 가장 우수한 성능을 보이는 것으로 나타났습니다.

- **Performance Highlights**: LLM의 전이 학습(Transfer Learning) 및 지침 조정(Instruction Tuning) 능력 덕분에 얻어진 성능 향상은 규칙 기반 추출 방법의 집합과 결합했을 때보다 훨씬 더 중요합니다. 이 연구는 LLMs가 과학 아티팩트의 자동 이해, 모델 회복 및 시뮬레이션을 향상시킬 잠재력을 보여줍니다.



### Cross-Modal Pre-Aligned Method with Global and Local Information for Remote-Sensing Image and Text Retrieva (https://arxiv.org/abs/2411.14704)
- **What's New**: 이번 연구는 원거리 탐지 교차 모드 텍스트-이미지 검색 (RSCTIR)에서 글로벌 및 로컬 정보를 효과적으로 통합하기 위한 CMPAGL 방법을 제안합니다. 특히, Gswin transformer block을 활용하여 다중 스케일 특성을 포착하고, 준비 정렬(pre-alignment) 메커니즘을 통해 모드 융합(training) 과정을 간소화합니다. 새로운 유사도 행렬 재가중화(SMR) 알고리즘과 최적화된 삼중 손실 최적화 방법을 통해 특징 학습을 최적화하고 검색 성능을 향상시킵니다.

- **Technical Details**: CMPAGL 방법은 이미지 인코더, 텍스트 인코더 및 다중 모드 인코더를 포함하는 인프라를 기반으로 합니다. Gswin transformer block은 로컬 창(self-attention)과 글로벌-로컬 창 크로스 어텐션(cross-attention)을 결합하여 특징 추출을 최적화하며, 이는 복잡한 세부 정보와 다중 스케일 정보를 효과적으로 캡처합니다. 또한, SMR 알고리즘은 원래 유사도 행렬을 활용하여 재정렬을 개선하고, 최적화된 삼중 손실은 매칭되는 이미지와 텍스트 간의 거리를 최소화합니다.

- **Performance Highlights**: CMPAGL 방법은 RSICD 및 RSITMD 등 다양한 데이터셋에서 기존 최첨단 방법보다 R@1에서 최대 4.65% 향상, 평균 Recall(mR)에서 2.28% 향상을 달성하는 등의 실험 결과를 통해 효과성을 입증했습니다. 이러한 성과는 제안된 아키텍처의 유효성을 뒷받침하며, 먼 거리 탐지 이미지-텍스트 검색 분야의 발전에 기여할 것입니다.



### An Experimental Study on Data Augmentation Techniques for Named Entity Recognition on Low-Resource Domains (https://arxiv.org/abs/2411.14551)
Comments:
          21 pages, 2 figures

- **What's New**: 본 연구에서는 Named Entity Recognition (NER)과 관련하여 데이터 증강(data augmentation) 기술의 효과를 평가합니다. 특히 Mention Replacement(MR) 및 Contextual Word Replacement(CWR)라는 두 가지 주요 텍스트 증강 기법이 Bi-LSTM+CRF와 BERT 모델에 미치는 영향을 알아봅니다. 이 과정에서 저자들은 다양한 저자원(lowersource) 데이터셋으로 실험을 수행하고, 데이터 증강이 소규모 데이터셋에 유리하다는 것을 증명했습니다.

- **Technical Details**: 본 연구에서는 NER 모델 훈련에 있어 데이터 증강 기술이 미치는 영향을 분석했습니다. 연구에 사용된 두 가지 모델인 Bi-LSTM+CRF와 BERT는 각각 고유한 아키텍처를 가지고 있으며, 특히 BERT 모델은 데이터 증강의 이점을 더 많이 누리는 것으로 나타났습니다. 연구자들은 데이터 증강의 최적 수량이 정해지지 않으며, NER 실무자들은 다양한 양의 데이터를 활용하여 실험해야 한다고 강조했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 데이터 증강은 특히 작은 데이터셋에서 유리하지만, 큰 데이터셋에서는 오히려 낮은 성능을 보일 수도 있음을 보여주었습니다. 또한, CWR은 MR보다 더 우수한 성능을 보였으며, BERT 모델이 Bi-LSTM+CRF 모델보다 데이터 증강의 혜택을 더 많이 받는 것으로 나타났습니다. 이 연구는 앞서 연구되지 않았던 저자원 데이터셋에 대한 NER 모델 평가를 포함하여, 기존 연구에 기여하고 있습니다.



### Learning to Ask: Conversational Product Search via Representation Learning (https://arxiv.org/abs/2411.14466)
Comments:
          Accepted by ACM TOIS

- **What's New**: 본 연구에서는 대화형 제품 검색 모델인 ConvPS를 제안합니다. 기존 연구들은 사용자와 상품 간의 정보 불일치 문제를 제대로 해결하지 못했습니다. 우리의 모델은 사용자, 쿼리, 아이템, 대화를 통합하여 표현 학습을 수행하고, 이를 바탕으로 보다 자연스럽고 적응력 있는 검색 시스템을 구축합니다.

- **Technical Details**: ConvPS 모델은 사용자, 쿼리, 아이템, 대화의 의미를 통합하여 학습하는 통합 생성 프레임워크를 사용합니다. 특히, 슬롯-값쌍(slots-value pairs)을 기반으로 한 질문 풀을 생성하고, 고성능 질문을 사용자에게 순차적으로 질문하는 방법을 학습합니다. 또한, 학습된 표현을 활용하여 상품을 검색합니다.

- **Performance Highlights**: 실험 결과는 ConvPS 모델이 기존 최첨단 문제 해결 모델에 비해 유의미하게 성능을 향상시킴을 보여줍니다. 이를 통해 사용자와의 대화에서 얻는 명확한 피드백을 기반으로 사용자의 동적인 선호를 더 잘 이해하게 됩니다.



### Unveiling User Preferences: A Knowledge Graph and LLM-Driven Approach for Conversational Recommendation (https://arxiv.org/abs/2411.14459)
- **What's New**: COMPASS라는 새로운 프레임워크가 제안되어 기존의 CRSs(Conversational Recommender Systems)에서 사용자 선호도를 개선하고, 추천 성능과 설명 가능성을 높이고자 합니다. 이 프레임워크는 LLMs(Large Language Models)와 KGs(Knowledge Graphs)를 통합하여 비가시적인 사용자 선호도를 드러내는 새로운 방법을 제시합니다. 특히, COMPASS는 두 단계의 훈련 과정을 통해 비구조적 대화와 구조적 KGs 간의 간극을 좁힙니다.

- **Technical Details**: COMPASS는 그래프 엔티티 설명(pre-training) 메커니즘을 사용하여 KG 구조를 자연어 설명으로 변환하여 LLM이 도메인 특정 정보를 이해할 수 있게 만듭니다. 이어서, COMPASS는 지식 기반(instruction) 세부조정을 통해 사용자 선호에 대한 추론 능력을 높입니다. 이 과정에서 KG로부터 추출된 관련 엔티티 정보와 관계를 포함하여 대화 기록을 보강하는 KG 향상 컨텍스트를 사용합니다.

- **Performance Highlights**: COMPASS는 설명 가능한 사용자 선호 요약 생성능력을 통해 CRSs에서 실제 데이터와 대화 이력을 기반으로 한 지능형 추천 및 이해 가능한 결과를 제공합니다. 이 프레임워크는 기존의 CRS 모델과 통합되는 적응형 게이팅 메커니즘을 포함하여, 성능을 강화하고, 수정 없이도 추천 품질을 높입니다. 이를 통해 COMPASS는 더욱 명확하고 해석 가능한 추천 시스템을 구현할 수 있는 가능성을 제공합니다.



### Can Artificial Intelligence Generate Quality Research Topics Reflecting Patient Concerns? (https://arxiv.org/abs/2411.14456)
- **What's New**: 이 연구는 환자 중심의 연구를 위한 자동화된 프레임워크를 제안하며, 이를 통해 연구 및 환자 치료 간의 간극을 좁히려는 노력을 다룹니다. 혁신적인 자연어 처리(NLP)와 인공지능(AI)을 활용하여 환자 포털 메시지를 분석하고, 이를 통해 환자가 중요하게 여기는 문제를 우선시하는 연구 아이디어를 생성합니다. 이러한 접근 방식은 환자의 관점을 연구에 효과적으로 통합할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 연구는 2013년부터 2024년까지의 대규모 학술 병원에서 수집된 614,464개의 환자 메시지를 통해 2단계 비지도 학습 기반의 NLP 주제 모델을 구성하였습니다. AI(GPT-4o, OpenAI Inc, 2024년 4월 버전)를 활용하여 여러 단계의 작업을 수행하고, 지식 해석, 연구 아이디어 생성, 자기 반성 및 수정, 자기 확신의 과정을 통해 환자의 임상 문제를 해결하는 연구 주제를 생성했습니다.

- **Performance Highlights**: AI가 제안한 연구 주제 중 3분의 1이 높은 중요성과 독창성을 갖추었다고 평가되었습니다. 또한, 두 암(유방암과 피부암)에 대해서 AI가 제안한 연구 주제의 3분의 2가 독창적이라고 판단되었습니다. 이러한 결과는 방대한 환자 메시지를 통해 얻어진 AI 생성 연구 주제가 환자 중심의 건강 연구 방향을 의미 있게 안내할 수 있음을 보여줍니다.



New uploads on arXiv(cs.CV)

### DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (https://arxiv.org/abs/2411.15139)
Comments:
          Work in progress. Code & demo & model will be available at this https URL

- **What's New**: DiffusionDrive는 최근 확산 모델(diffusion model)을 활용하여 로봇 정책 학습에서의 생성(generative) 방식에 혁신을 가져왔습니다. 이 모델은 다중 모드(multi-mode) 액션 분포를 효과적으로 모델링하며, 특히 자율 주행에서의 응용 가능성을 높이고 있습니다. 이전의 디퓨전 정책과 비교하여, 20회에 달하는 디노이징(denoising) 과정을 두 단계로 줄여 실시간 속도를 구현한 것이 주목할 만합니다.

- **Technical Details**: DiffusionDrive는 앵커 가우시안 분포(anchored Gaussian distribution)를 도입하여 디퓨전 일정을 절단(truncating)하는 새로운 방법론을 제안합니다. 이를 통해 모델은 사전 앵커를 중심으로 다중 가우시안 분포를 학습하게 되며, 그 결과 다양한 드라이빙 액션을 실시간으로 생성할 수 있습니다. 이 과정은 변환기 기반(diffusion decoder) 모델을 활용하여 상황 맥락과의 상호작용을 효율적으로 진행합니다.

- **Performance Highlights**: DiffusionDrive는 NAVSIM 데이터셋에서 88.1 PDMS를 달성하며 새로운 기록을 세웠고, 실시간 속도 45 FPS로 NVIDIA 4090에서 실행됩니다. 기존의 VADv2 모델과 비교하여, DiffusionDrive는 1.6 PDMS의 성능 향상을 보이며, 또한 nuScenes 데이터셋에서도 20.8% 더 낮은 L2 오류와 63.6% 낮은 충돌률을 기록하여 뛰어난 계획 성능(planning performance)을 입증했습니다.



### Material Anything: Generating Materials for Any 3D Object via Diffusion (https://arxiv.org/abs/2411.15138)
Comments:
          Project page: this https URL

- **What's New**: Material Anything는 3D 객체를 위한 물리 기반 재료를 생성하기 위해 설계된 완전 자동화된 통합 확산 프레임워크입니다. 기존의 복잡한 파이프라인이나 케이스별 최적화에 의존하지 않고, 다양한 조명 조건에서 적응할 수 있는 견고한 솔루션을 제공합니다. 이 연구에서는 사전 훈련된 이미지 확산 모델을 활용하며, 안정성과 재료 품질을 향상시키기 위한 트리플 헤드 아키텍처와 렌더링 손실을 도입하였습니다.

- **Technical Details**: 이 접근 방식은 두 단계의 파이프라인을 통해 고품질 재료 맵을 생성합니다: 이미지 공간 재료 생성 및 UV 공간 재료 정제. 이미지 공간 재료 확산 모델은 3D 객체의 각 뷰에 대해 PBR 재료를 생성하는 것을 목표로 하며, 조명 신뢰도를 나타내는 신뢰 마스크를 도입하여 다양한 조명 시나리오에 따른 물체 지원을 가능하게 합니다.

- **Performance Highlights**: Material Anything는 다양한 조명 조건과 객체 카테고리에서 기존 방법들보다 성능이 뛰어난 결과를 보여줍니다. 연구팀은 80K 이상의 고품질 PBR 재료와 UV 맵이 포함된 Material3D 데이터셋을 구축하여 시스템의 안정성 및 일관성을 크게 향상시켰습니다. 제안된 방법은 자동화된 생성 과정에서 뛰어난 확장성과 안정성을 나타냅니다.



### ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation (https://arxiv.org/abs/2411.15122)
- **What's New**: AI 기반 모델이 흉부 X선의 영상 판독 보고서를 자동으로 생성하는 가능성을 보여주고 있으나, 이를 객관적으로 평가할 수 있는 표준화된 벤치마크가 부족했다. 이 연구는 ReXrank라는 공개 리더보드와 챌린지를 소개하여 AI 영상 판독 보고서 생성 모델의 평가를 위한 체계적인 기준을 제공한다. ReXrank는 10,000개의 스터디로 이루어진 ReXGradient라는 대규모 데이터셋을 포함하여 세 가지 공개 데이터셋(MIMIC-CXR, IU-Xray, CheXpert Plus)을 활용하고 있다.

- **Technical Details**: ReXrank는 8개의 평가 지표를 사용하여 모델의 성능을 평가한다. 모델은 단순히 발견 사항을 생성하는 것과 발견 사항과 해석 모두를 제공하는 모델로 구분하여 평가된다. 다양한 데이터셋을 통합하여, 모델의 일반화 능력을 보다 깊이 이해할 수 있는 평가를 가능하게 한다.

- **Performance Highlights**: MedVersa 모델이 ReXGradient 및 MIMIC-CXR 데이터셋에서 최고 성능을 기록하며, 다양한 지표에서 GPT4V보다 우수한 성능을 보였다. 평가 지표의 분포를 분석한 결과, IU X-ray는 모든 모델에서 높은 성능을 보였지만, CheXpert Plus는 가장 높은 변동성과 낮은 성능을 나타냈다. ReXGradient 데이터셋은 매우 낮은 성능 변동성을 보여 모델의 강건성 평가에 유용함을 입증했다.



### VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinemen (https://arxiv.org/abs/2411.15115)
Comments:
          Project page: this https URL

- **What's New**: 최근 텍스트-비디오(T2V) 확산 모델들은 다양한 분야에서 뛰어난 생성 능력을 보여주고 있습니다. 하지만 이러한 모델들은 여러 객체와 속성이 포함된 복잡한 장면을 설명하는 텍스트 프롬프트와 정확하게 일치하는 비디오 생성에 어려움을 겪고 있습니다. 이 문제를 해결하기 위해 우리는 VideoRepair라는 새로운 모델 불가지론적 훈련 없는 비디오 정제 프레임워크를 도입합니다.

- **Technical Details**: VideoRepair는 미세한 텍스트-비디오 일치 문제를 자동으로 감지하고, 명확한 공간적 및 텍스트 피드백을 생성하여 T2V 확산 모델이 목표로 하는 부위에서 정제를 수행하도록 돕습니다. 이 프레임워크는 네 단계로 구성되어 있으며, 첫 번째 단계에서는 미세한 평가 질문을 생성하여 비디오를 평가하고, 두 번째 단계에서는 정제할 영역을 계획합니다. 세 번째 단계에서는 정확하게 생성된 영역을 세분화하고, 마지막 단계에서는 정제된 프롬프트에 따라 비디오를 재생성합니다.

- **Performance Highlights**: VideoRepair는 EvalCrafter와 T2V-CompBench라는 인기 있는 비디오 생성 벤치마크에서 최근 기준선보다 우수한 성능을 보입니다. 다양한 텍스트-비디오 정렬 메트릭에서도 상당한 개선을 보여주며, 정제 과정의 각 구성 요소에 대한 포괄적인 분석과 질적 사례를 제공합니다. 이 연구는 시각 생성 작업에서 자동 정제 프레임워크의 발전을 촉진할 것으로 기대됩니다.



### Efficient Pruning of Text-to-Image Models: Insights from Pruning Stable Diffusion (https://arxiv.org/abs/2411.15113)
- **What's New**: 본 연구는 리소스 제약이 있는 장치에서 텍스트-이미지 모델의 채택을 가로막는 주요 원인인 모델 크기의 문제를 해결하기 위해 Stable Diffusion 2의 후처리 가지치기(post-training pruning) 기법을 분석합니다. 이전에 다루어지지 않았던 다중 모드 생성 모델의 가지치기 기술을 다루면서 텍스트 컴포넌트와 이미지 생성 컴포넌트에 대한 가지치기 체계의 영향을 개별적으로 검토합니다. 이 연구는 기존 언어 모델 가지치기 경향과는 달리 텍스트-이미지 맥락에서 두드러진 발견을 보고합니다.

- **Technical Details**: 긴 모델 아키텍처의 경우, 모델 크기는 수십억 개의 파라미터를 포함하고 있으며, 이는 실질적인 연산 비용을 발생시킵니다. 본 논문에서는 텍스트 인코더와 확산 생성기를 개별적으로 가지치기하는 방법과 각 컴포넌트에 대한 최적의 희소성(sparsity) 수준을 모색합니다. 연구 결과에 따르면, Stable Diffusion 2의 경우 38.5%의 희소성을 달성하면서 품질 손실을 최소화할 수 있음을 발견했습니다.

- **Performance Highlights**: 가지치기를 통해 모델의 연산 요구 사항을 대폭 줄일 수 있으며, 특히 텍스트 인코더에 대해 47.5%까지, 확산 생성기에 대해서는 35%까지 최적의 가지치기 구성을 제안합니다. 연구에서 나타난 흥미로운 점으로는 특정 임계를 넘어서는 가지치기가 성능 급락을 야기할 수 있다는 것이며, 이는 특정 가중치들이 핵심 의미론적 정보를 담고 있음을 시사합니다. 이러한 발견은 텍스트-이미지 모델의 모델 압축, 상호 운용성 및 편향 식별에 대한 새로운 연구 방향을 열어줍니다.



### A Real-Time DETR Approach to Bangladesh Road Object Detection for Autonomous Vehicles (https://arxiv.org/abs/2411.15110)
- **What's New**: 최근 몇 년간 컴퓨터 비전 분야에서 transformer 아키텍처의 발전으로 패러다임의 변화가 있었습니다. Detection Transformers는 객체 탐지에서 최첨단 솔루션으로 자리 잡았으며, 자율 주행차에서 도로 물체 탐지의 유력한 후보로 부상하고 있습니다. 빠른 inference 시간에도 불구하고, Real-Time DETR (RTDETR) 모델이 보다 나은 성능을 보여 주목할 만합니다.

- **Technical Details**: BadODD 데이터셋은 오토리크샤, 자전거, 버스, 자동차, 화물차 등 총 13개의 클래스로 구성되어 있으며, 각 클래스의 인스턴스 분포가 불균형적입니다. 이로 인해 훈련 및 평가 과정에서의 모델 정확도에 영향을 미치는 도전 과제가 발생합니다. RT-DETR 모델은 YOLO 스타일 아키텍처의 속도와 transform의 표현 능력을 결합하여, 더 효과적인 객체 탐지를 구현합니다.

- **Performance Highlights**: 모델은 60% 공공 데이터셋에서 mAP 점수 0.4151을 달성하며, 40% 남은 데이터셋에서는 0.2891의 성적을 거두었습니다. 평균 inference 시간은 22.44ms로, 자율주행 및 보안 시스템에서의 실시간 응답에 적합합니다. 하지만 극도로 작은 객체나 가려진 객체 탐지에서는 어려움이 있으며, 이에 대한 연구 개발 노력이 지속적으로 필요합니다.



### About Time: Advances, Challenges, and Outlooks of Action Understanding (https://arxiv.org/abs/2411.15106)
- **What's New**: 이번 논문에서는 비디오 액션 이해(action understanding) 분야의 최근 발전을 살펴보며, 단일 모드 및 다중 모드 작업을 포함한 다양한 작업들에 대한 전반적인 개요를 제공합니다. 이 조사는 현재 시스템의 성과를 기반으로 여러 가지 템포럴 스코프(temporal scopes)를 구분하여 액션 인식, 예측 및 예측 작업을 설명합니다. 또한, 현재의 한계를 극복하기 위한 미래 방향도 논의합니다.

- **Technical Details**: 액션 이해는 세 가지 주요 템포럴 스코프, 즉 전체 관찰에서의 인식 작업, 부분적으로 관찰된 진행 중인 행동에 대한 예측 작업, 그리고 관찰되지 않은 후속 행동 예측 작업으로 분류됩니다. 이러한 구분은 특정 행동 모델링 및 비디오 표현의 도전 과제를 식별하는 데 도움을 줍니다. 연구는 비디오에서의 행동 모델링 접근 방식을 시간의 흐름에 따라 다룹니다.

- **Performance Highlights**: 현재의 연구는 여러 도전 과제를 다루고 있으며, 이전의 리뷰들은 특정 측면에 중점을 두고 있습니다. 예를 들어, 성능의 변동성은 데이터세트 내의 다양한 인스턴스 분포 차이에 따라 달라질 수 있으며, 제한된 클래스 간 변동은 모델의 일반화에 영향을 미칩니다. 이러한 점에서, 저자는 비디오 행동 이해에 대한 포괄적 리뷰를 제공하며, 다양한 데이터세트와 벤치마크를 통해 향후 연구 방향을 제시합니다.



### Context-Aware Multimodal Pretraining (https://arxiv.org/abs/2411.15099)
- **What's New**: 이 논문에서는 대규모 멀티모달 표현 학습(multi-modal representation learning)을 위한 간단하지만 신중하게 설계된 확장을 제안합니다. 이는 기존 contrastive learning 개념이 few-shot 적응(few-shot adaptation)을 지원하도록 최적화되지 않았음을 지적합니다. 따라서 이 연구의 결과로, 다양한 다운스트림 태스크에서 샘플 효율성이 최대 4배 향상되고 평균 5% 이상의 성능 향상이 나타났습니다.

- **Technical Details**: 제안된 LIxP는 Contrastive Language-Image Pretraining의 확장으로, 교육 과정에서 cross-attention 기반의 맥락화를 통해 메트릭 기반 적응을 위한 표현을 준비합니다. 이 접근법은 전체 손실 설계와 개별적으로 학습 가능한 온도를 활용하여 기초 제로샷(zero-shot) 능력을 유지합니다. LIxP는 21개의 다운스트림 분류 작업에서 제로샷 전이 성능을 유지하면서 원래 성능을 뛰어넘습니다.

- **Performance Highlights**: LIxP는 훈련 없는 메트릭 기반 적응 메커니즘을 통해 대규모 멀티모달 모델이 적응할 수 있도록 막대한 성능 향상을 보여줍니다. 이는 복잡하고 비싼 최적화 기반 방법에 비해 간단한 접근 방식을 채택하여 새로운 도메인으로의 일반화를 쉽게 만듭니다. 이 논문은 또한 기존의 few-shot 및 many-shot 전이와 관련된 문제를 해결하고, 더욱 효율적인 학습 모델을 구축하는 방향성을 제시합니다.



### OminiControl: Minimal and Universal Control for Diffusion Transformer (https://arxiv.org/abs/2411.15098)
- **What's New**: 이번 논문에서는 이미지 조건을 통합한 OminiControl이라는 매우 다재다능하고 매개변수 효율적인 프레임워크를 소개합니다. OminiControl은 사전 학습된 Diffusion Transformer(DiT) 모델을 활용하여 이미지 조건을 효과적으로 인코딩하고 처리할 수 있도록 설계되었습니다. 기존의 복잡한 추가 인코더 모듈에 의존하는 방법들과는 달리, OminiControl은 약 0.1%의 추가 매개변수만으로 이미지 조건을 통합하고, 다양한 이미지 조건 작업을 통합적으로 처리할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: OminiControl은 기존 DiT 아키텍처에 이미지 기반 제어를 통합하기 위한 매개변수 효율적인 접근 방식을 제안합니다. 이 방법에서는 모델의 기존 VAE 인코더를 활용하여 조건 이미지를 처리하며, 디노이징 네트워크의 잠재적인 노이즈와 함께 조건 이미지를 통합하여 멀티 모달 어텐션 상호작용을 촉진합니다. 이는 DiT의 트랜스포머 블록 전반에 걸쳐 효율적인 정보 교환과 제어 신호 전파를 가능하게 합니다.

- **Performance Highlights**: OminiControl을 적용한 DiT 기반의 접근 방식은 주제 기반 생성(subject-driven generation) 및 공간 정렬 조건(spatially-aligned conditions) 작업에서 기존의 UNet 기반 및 DiT 조정 모델보다 월등한 성능을 보여줍니다. 이 연구에서는 200,000개 이상의 다양한 이미지로 구성된 Subjects200K라는 고품질 데이터셋을 개발하여 공개하며, 주제 일관성 생성(task) 연구를 위한 귀중한 자원을 제공합니다.



### Instance-Aware Generalized Referring Expression Segmentation (https://arxiv.org/abs/2411.15087)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서 제안하는 InstAlign는 복잡한 다중 객체 참조 표현을 효과적으로 처리하기 위한 첫 번째 인스턴스 인식 접근 방식을 제안합니다. 이 모델은 텍스트의 특정 구문과 개별 객체 인스턴스를 명시적으로 구별하고 연결함으로써, 더욱 정교한 세분화를 실현합니다. InstAlign는 우수한 인스턴스 분리를 통해 기존의 GRES 모델에 비해 성능을 크게 향상시킵니다.

- **Technical Details**: InstAlign는 Transformer 기반 아키텍처를 활용하여 물체 쿼리를 처리하고, 이는 입력 텍스트에서 지정된 객체의 인스턴스를 세분화하는 데 목적을 두고 있습니다. 모델은 K 개의 Phrase-Object Transformer 블록을 통해 다중 스케일 시각적 특성과 텍스트 특성을 결합하여 쿼리를 정제합니다. 최종적으로, 각 인스턴스는 텍스트 구문과 연계된 세분화 마스크를 생성하고, 이는 효율적으로 학습됩니다.

- **Performance Highlights**: InstAlign는 gRefCOCO 및 Ref-ZOM 벤치마크에서 매우 높은 성능을 자랑하며, gIoU 74.34%와 N-acc 79.72%를 기록하여 기존의 방법들을 3.3% 및 12.25% 이상 초과 달성했습니다. 본 연구는 복잡한 참조 표현을 처리하는 데 있어 새로운 표준을 정립하며, 인스턴스 세분화를 통한 성능 향상을 성공적으로 입증합니다.



### Learning to Stabilize Faces (https://arxiv.org/abs/2411.15074)
Comments:
          Eurographics 2024

- **What's New**: 이 논문은 얼굴 데이터의 자동 안정화 문제를 해결하기 위한 새로운 학습 기반 접근 방법을 제안합니다. 기존 방법들이 수동 작업, 불확실한 휴리스틱, 느린 최적화 등에 의존했던 반면, 이 방법은 네트워크를 이용해 두 개의 얼굴 메쉬 간의 강체 변환을 직접 예측하는 방식입니다. 이 방식은 3D Morphable Model (3DMM)을 이용해 합성 훈련 데이터를 생성하여 학습합니다.

- **Technical Details**: 이 방법의 핵심은 안정화를 회귀 문제로 간주하고 두 개의 입력 메쉬에 대해 불필요한 강체 변환을 예측하는 것입니다. 3DMM 파라미터의 특성을 활용하여 개별 정체성과 표정을 결합한 새로운 훈련 샘플을 생성하고, 모델을 합성 데이터로 학습합니다. 실험 결과, 이 방법이 실제 메쉬에서도 높은 정확도를 보이며 기존 최첨단 기술을 뛰어넘는 성능을 나타냄을 입증하였습니다.

- **Performance Highlights**: 자동 안정화 성능의 실험을 통해 본 접근 방식이 정적인 얼굴 표정 및 동적인 얼굴 수행 안정화 작업에서 정량적이고 정성적으로 뛰어난 결과를 보여주었습니다. 추가적으로 사용자가 이 접근 방식에서 선택한 설계 결정 및 모범 사례를 도와줄 수 있는 ablation 연구를 포함함으로써 다른 연구자들이 이 방법을 쉽게 채택할 수 있도록 하였습니다.



### SPAC-Net: Rethinking Point Cloud Completion with Structural Prior (https://arxiv.org/abs/2411.15066)
- **What's New**: 본 연구에서는 기존의 혼합형 형태 완성 방법에서 벗어나, '인터페이스'라는 구조적 프라이어를 바탕으로 하는 새로운 형태 완성 프레임워크인 SPAC-Net을 제안합니다. 이 방법은 마진 탐지기(Margin Detector, MAD) 모듈을 활용하여 관찰된 부분과 누락된 부분의 교차 지점을 로컬라이즈하고, 이 정보를 기반으로 누락된 부분의 모양을 예측하는 방식입니다. 또한, 구조 보완 모듈(Structure Supplement, SSP)을 도입하여 구조적 세부 사항을 강화하여 더 나은 업샘플링 성능을 자랑합니다.

- **Technical Details**: SPAC-Net은 점 구름(point cloud) 완성을 위해 N개의 점으로 이루어진 포인트 클라우드를 이끌며, 부분 스캔(P={p_i})과 누락된 부분(M={m_i})으로 나누어 처리합니다. 특히, 이 연구는 인터페이스(T)와 그에 포함된 포인트 간의 상호작용을 통해 누락된 부분(M')을 예측하는 모델을 설계했습니다. MAD 모듈은 누락된 데이터의 공간적 구조를 이해하기 위한 중요한 도구로 활용되어, 누락된 데이터의 상황에 따른 성능 검증을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, SPAC-Net은 다양한 벤치마크에서 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다. 이 방법은 세밀한 구조적 세부 정보 복구에 집중함으로써 누락된 부분의 완성과 관련하여 더 뛰어난 능력을 보입니다. 특히, 코어 형상을 예측하고, 업샘플링 과정에서 세부 정보를 강조하는 SSP 모듈은 전체적인 성능 향상에 중요한 역할을 수행합니다.



### OVO-SLAM: Open-Vocabulary Online Simultaneous Localization and Mapping (https://arxiv.org/abs/2411.15043)
- **What's New**: 이 논문은 첫 번째 Open-Vocabulary Online 3D semantic SLAM 파이프라인인 OVO-SLAM을 제안합니다. OVO-SLAM의 주요 공헌은 고속 처리 및 더 나은 분할 지표(segmentation metrics)를 제공하는 매핑 스레드(mapping thread)입니다. CLIP 벡터를 사용하여 새로운 방식으로 3D 세그먼트를 감지하고 추적하여 온라인 3D 재구성을 실현합니다. 이 연구는 기초 카메라 위치나 장면 기하학에 의존하지 않으면서도 end-to-end 형태의 오픈 보캐뷸러리 온라인 3D 재구성을 처음으로 입증합니다.

- **Technical Details**: OVO-SLAM은 3D 포인트 클라우드 표현으로부터 세그먼트를 추출하며, 각 세그먼트에 CLIP 벡터를 할당합니다. 이 세그먼트는 SAM 2.1 마스크를 역투영(back-projection)하여 초기화되며, 2D 마스크에 대해 3D 세그먼트를 투영하고 일치(matching)시킵니다. CLIP 벡터는 더 나은 가시성을 가진 키프레임에서 선택되며, 각 세그먼트에 대해 인스턴스 CLIP 설명자를 추출하는 새로운 모델을 기여합니다. OVO-SLAM의 파이프라인은 기존 오프라인 방법들보다 더 온라인 속도가 빠르며 성능적으로 우수합니다.

- **Performance Highlights**: OVO-SLAM은 기존 오프라인 방법들에 비해 분할 성능이 향상되었습니다. 오프라인 세멘틱 3D 재구성이 전통적으로 폐쇄된 세트의 카테고리에 제한되어 있는 반면, OVO-SLAM은 오픈 보캐뷸러리를 지원하여 현실 세계의 더 다양한 상황에 적합성을 높입니다. 실험 결과는 Gaussian-SLAM과 통합된 OVO-SLAM의 성능을 보여주며, 이는 온라인 3D 재구성의 새로운 가능성을 시사합니다.



### HeadRouter: A Training-free Image Editing Framework for MM-DiTs by Adaptively Routing Attention Heads (https://arxiv.org/abs/2411.15034)
- **What's New**: 이번 연구는 Diffusion Transformers (DiTs)의 새로운 구조를 소개하며, 특히 멀티모달 Diffusion Transformers (MM-DiTs)에서의 텍스트 유도 이미지 편집의 문제를 다룹니다. 기존의 UNet 기반 모델과 달리, MM-DiTs는 텍스트 안내를 명시적으로 통합할 수 있는 방법이 부족하여 편집 결과물과 텍스트 간의 의미적 정렬이 틀어지는 문제를 해결하고자 합니다.

- **Technical Details**: HeadRouter라는 새로운 이미지 편집 프레임워크를 도입하며, 이는 훈련 과정 없이도 이미지의 다양한 주의 헤드에 적응적으로 텍스트 안내를 라우팅하여 이미지 편집을 수행합니다. 또한, 이 연구에서는 텍스트 및 이미지 토큰 표현을 정제하기 위한 이중 토큰 정제 모듈을 사용하여 보다 정확한 의미적 안내를 제공합니다. 이 연구를 통해 MM-DiTs의 다중 헤드 주의 메커니즘의 세부 구조를 분석하고, 각 주의 헤드가 특정 이미지 속성에 대해 어떻게 반응하는지를 탐구합니다.

- **Performance Highlights**: 여러 텍스트 유도 이미지 편집 벤치마크에서 HeadRouter의 우수한 성능이 입증되었고, 정확한 지역, 의미 및 속성 기반 편집 효과를 달성했습니다. 연구 결과, 우리의 접근법은 기존 최첨단 방법들을 초월하는 결과를 보여줍니다. 또한, 이 방법은 시간 효율성을 유지하며 복잡한 계산을 피하는 특성도 갖추고 있습니다.



### FloAt: Flow Warping of Self-Attention for Clothing Animation Generation (https://arxiv.org/abs/2411.15028)
- **What's New**: 이번 논문에서는 인간 의류의 애니메이션을 생성하기 위한 diffusion model 기반의 방법인 FloAtControlNet을 제안합니다. 사용자는 텍스트 프롬프트와 함께 의류의 텍스처를 설명하는 일련의 노멀 맵을 입력할 수 있습니다. 제안된 방법은 ControlNet을 기반으로 하여 훈련 없이 작동할 수 있는 장점을 가지고 있으며, 의류 애니메이션의 자연스러움을 크게 향상시킵니다.

- **Technical Details**: 우리는 노멀 맵 간의 흐름을 활용하여 현재 프레임의 self-attention 맵을 수정하고, 이로써 의류 애니메이션을 더 자연스럽게 만드는 방법을 보여줍니다. 특히, 우리의 방법은 특정 레이어와 시간 단계의 self-attention 맵을 왜곡된 버전으로부터 조작하여 일관성을 유지하는데 효과적입니다. 우리의 접근 방식은 CycleNet과 같은 기존 GAN 기반 방법과 비교할 때, 고주파 텍스처를 가진 의류의 애니메이션 생성에서 우위를 점합니다.

- **Performance Highlights**: We demonstrate that our method significantly outperforms existing baselines both qualitatively and quantitatively in user studies. Specifically, 우리는 기존의 diffusion model 기반 방법에서 나타나는 배경 깜빡임 현상을 효과적으로 줄이고, RMSE와 PSNR 측정에서도 우수한 성능을 보입니다. 따라서 LPIPS, SSIM, CLIP와 같은 시각적 품질 평가 지표들이 의류 애니메이션의 미세한 움직임을 포착하는 데 적합하지 않다는 점도 강조됩니다.



### DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models (https://arxiv.org/abs/2411.15024)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 연구에서는 Video Large Language Models (VLLMs)에 대한 새로운 접근법인 DyCoke를 소개합니다. DyCoke는 교육 없이도 비디오 토큰 압축을 최적화하여 추론 속도를 높입니다. 이 방법은 비디오 프레임 간의 중복을 줄이기 위해 플러그 앤 플레이(plug-and-play) 방식의 모듈을 사용하여 비디오 이해의 효율성을 크게 향상시킵니다.

- **Technical Details**: DyCoke는 두 가지 주요 단계를 포함합니다: 첫째, 프레임 간 중복 토큰을 병합하는 것으로, 연속적인 프레임에서 유사한 정보를 가진 토큰을 그룹화합니다. 둘째, KV 캐시의 덜 중요한 정보를 동적으로 잘라내어 공간적 중복을 줄이면서도 중요한 토큰을 유지합니다. 이러한 방식은 비디오 입력의 각 프레임에서 평균 15개의 토큰을 남깁니다.

- **Performance Highlights**: 실험 결과, DyCoke는 LLaVA-OV-7B 모델에서 1.54배의 추론 속도 증가를 달성함과 동시에 낮은 메모리 소모와 높은 정확성을 기록하였습니다. 이 기법은 훈련이 필요 없으며, 비디오 토큰을 압축하고 더 긴 비디오 시퀀스를 처리할 수 있는 능력을 보여줍니다.



### Neural 4D Evolution under Large Topological Changes from 2D Images (https://arxiv.org/abs/2411.15018)
Comments:
          15 pages, 21 figures

- **What's New**: 이 논문은 2D 이미지로부터 4D 형태를 모델링하고 예측하는 새로운 방법을 제안합니다. 기존의 3D 방법을 4D로 확장하려는 시도가 있으나, 함께 발생하는 큰 위상(topology)의 변화로 인해 성능이 저조했습니다. 이에 따라 논문은 두 가지 혁신적인 수정 사항, 즉 변형(displacement)과 SDF(Signed Distance Function)의 학습을 위한 새로운 아키텍처와 시간 일관성을 부여하는 기술을 도입합니다.

- **Technical Details**: 제안된 방법은 HashGrid 기반의 아키텍처를 사용하여 3D 공간을 디스크리트(discrete)하게 샘플링하고, SDF를 통해.geomtrical features를 학습합니다. 이 과정에서, Gaussian splatting 기법을 이용하여 색상 예측을 위한 렌더링 모듈을 구현하여 효율적으로 성능을 강화했습니다. 또한, 이 방법은 변화하는 표면을 효과적으로 모델링하기 위해 비선형(neural) 서브모델을 활용하여 각 프레임에서의 변형도 연속적으로 학습합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, 제안된 방법은 특히 큰 위상이 변화하는 동적 장면의 재구성을 효과적으로 수행하였습니다. experiments demonstrate that the model not only generalizes well but also accurately captures deformations and interpolations in previously unseen time-steps. 공개된 소스 코드와 데이터셋은 연구 결과를 재현하는데 도움을 줄 것입니다.



### MSSF: A 4D Radar and Camera Fusion Framework With Multi-Stage Sampling for 3D Object Detection in Autonomous Driving (https://arxiv.org/abs/2411.15016)
- **What's New**: 이 논문에서는 최근 자율주행 시스템의 인식 성능을 향상시키기 위한 새로운 다중 단계 샘플링 융합 네트워크(MSSF)를 제안합니다. 4D 밀리미터파 레이더와 카메라 데이터를 통합하여 포인트 클라우드와 이미지 특징 간의 상호작용을 강화합니다. 이전의 레이더-카메라 융합 방법들은 성능에서 큰 간극이 있었지만, 이 연구는 이를 해결하고 점진적으로 향상된 결과를 보여주고 있습니다.

- **Technical Details**: MSSF는 포인트 클라우드와 이미지 특징 간의 깊은 상호작용을 위한 두 가지 유형의 융합 블록, 즉 간단한 특징 융합(SFF)과 다중 스케일 변형 특징 융합(MSDFF)을 설계했습니다. SFF는 구현이 용이한 반면, MSDFF는 더 강력한 융합 능력을 보유하고 있습니다. 또한, 세분화 손실을 통해 포그라운드와 배경 포인트를 구별하는 세멘틱 가이드 헤드를 추가하여 기능 흐림 문제를 완화합니다.

- **Performance Highlights**: VoD 데이터셋과 TJ4DRadset 데이터셋에서 MSSF의 효과를 검증한 결과, 각각 7.0%와 4.0%의 평균 정밀도(mAP) 향상을 달성했습니다. 특히 VoD 데이터셋에서 차 범주에 대해 18.6%의 상당한 성장을 기록했으며, 기존의 최첨단 방법들보다 더 나은 성능을 보여주었습니다.



### Differentiable Biomechanics for Markerless Motion Capture in Upper Limb Stroke Rehabilitation: A Comparison with Optical Motion Captur (https://arxiv.org/abs/2411.14992)
Comments:
          7 pages, 4 figures, 3 tables, RehabWeek 2025 ICORR, first 3 authors are shared-first and last two authors are shared last

- **What's New**: 이번 연구는 전통적인 Marker-based Optical Motion Capture(OMC)와 새로운 Markerless Motion Capture(MMC) 방법을 비교하여, 뇌졸중 환자의 상지 운동 성능을 측정하는 데 있어 이점을 보여줍니다. MMC는 여러 대의 동기화된 웹캠만으로 수행되며, 이는 임상 설정에서 접근성을 높이는 데 기여합니다. 연구 결과, MMC의 정확성이 OMC에 근접함을 보여주며, 이는 재활 치료 전략을 향상시키는 데 유용할 것으로 예상됩니다.

- **Technical Details**: 연구에서는 15명의 뇌졸중 환자가 물컵을 잡는 표준화된 작업을 수행하는 동안 OMC와 MMC 기법을 동시에 기록하였습니다. OMC는 Qualisys 및 Optitrack 기술을 사용해 100Hz의 샘플링으로 기록되었고, MMC는 5대의 웹캠으로 60Hz로 동기화되었습니다. 두 시스템 각각에서 얻은 운동학적 궤적들은 상관관계와 RMSE를 통해 비교 분석되었으며, 이 모델은 OpenSim과 MuJoCo를 사용하여 최적화되었습니다.

- **Performance Highlights**: 결과적으로, 대다수의 운동학적 궤적에서 OMC와 MMC의 높은 상관관계가 나타났고, 관절 각도에서의 RMSE는 2-5도, 최종 효과기 속도에서는 0.04 m/s로 낮은 값을 보였습니다. 이러한 발견은 MMC가 OMC와 비교하여 유사한 정확성을 가지고 있음을 방증하며, 이는 뇌졸중 환자의 운동 재활에 있어 중요한 통찰력을 제공할 수 있는 가능성을 제시합니다.



### Large Multi-modal Models Can Interpret Features in Large Multi-modal Models (https://arxiv.org/abs/2411.14982)
- **What's New**: 최근 Large Multimodal Models (LMMs)의 발전은 학계와 산업에서 눈에 띄는 돌파구를 이끌어냈습니다. 본 논문에서는 이러한 LMMs의 내부 신경 표현을 이해하기 위한 초기 단계로, LMMs의 의미론을 식별하고 해석할 수 있는 다재다능한 프레임워크를 제시합니다. Sparse Autoencoder(SAE)를 적용하여 인간이 이해할 수 있는 특징으로 신경 표현을 분해하고, LMMs가 학습한 열린 의미론적 특징의 자동 해석 프레임워크를 도입합니다.

- **Technical Details**: 이 방법론에서는 OpenAI의 연구에서 제시된 SAE 아키텍처를 사용하여 LMMs의 내부 표현을 분해하고 해석합니다. SAE는 두 개의 레이어로 구성된 오토인코더로, Top-K 활성화 함수를 사용하며, 입력 데이터는 다양한 차원의 토큰으로 구성됩니다. 이 과정에서, sparse한 데이터 표현을 통해 다의적 신경 표현을 단일 의미적 특징으로 변환합니다.

- **Performance Highlights**: 실험 결과, LLaVA-NeXT-8B 모델을 분석한 결과 SAE를 통해 인간이 이해할 수 있는 특징이 모델의 행동을 조정하는데 효과적이라는 것을 입증했습니다. 특히, LMMs의 감정적 특징을 식별하고 이 모델들이 감정을 생성하거나 공유할 수 있는 능력을 확인하였습니다. 또한, LMMs의 특정 행동의 원인을 분석하고, 잘못된 결과를 수정하기 위한 전략을 제시하여 LMMs의 내부 메커니즘에 대한 새로운 통찰력을 제공합니다.



### 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes (https://arxiv.org/abs/2411.14974)
Comments:
          13 pages, 13 figures, 10 tables

- **What's New**: 최근의 조명 필드 재구성 기술인 3D Convex Splatting (3DCS)은 Gaussian primitives 대신 3D smooth convexes를 이용하여 기하학적으로 의미 있는 조명 필드를 모델링하는 새로운 방법을 제안합니다. 이를 통해 복잡한 장면을 보다 정밀하고 효과적으로 재구성할 수 있습니다.

- **Technical Details**: 3DCS는 3D smooth convexes를 사용하여 다중 시점 이미지로부터 조명 필드를 재구성합니다. 이 방법은 CUDA 기반의 효율적인 레스터라이저를 활용하여 실시간 렌더링을 가능하게 하며, Gaussian 기반 방법보다 적은 수의 primitives로도 고품질의 3D 장면을 표현할 수 있게 해줍니다.

- **Performance Highlights**: 3DCS는 Mip-NeRF360, Tanks and Temples, Deep Blending과 같은 벤치마크에서 3D Gaussian Splatting (3DGS)보다 최대 0.81의 PSNR 개선 효과를 보였습니다. 이러한 결과는 3DCS가 고품질 장면 재구성과 새로운 시점 합성에 있어 새로운 표준이 될 가능성을 시사합니다.



### Design-o-meter: Towards Evaluating and Refining Graphic Designs (https://arxiv.org/abs/2411.14959)
Comments:
          Accepted to WACV 2025. Project page: this https URL

- **What's New**: 이번 논문에서는 그래픽 디자인의 품질을 정량화하고 개선하기 위한 새로운 방법론, Design-o-meter를 소개합니다. Design-o-meter는 디자인 평가 및 향상을 위한 통합된 프레임워크로, 주관적이고 모호한 디자인 특성을 고려합니다. 이 방법론은 Generative AI 기술을 사용하여 디자인 생성뿐만 아니라 평가와 개선도 수행합니다.

- **Technical Details**: Design-o-meter는 두 가지 주요 모듈로 구성되며, 하나는 디자인을 평가하는 scorer이고, 다른 하나는 평가 결과를 바탕으로 디자인을 개선하는 refiner입니다. scorer는 metric learning 모델로 구현되어 있으며, 디자인의 좋고 나쁨을 점수화합니다. refiner는 새로운 설계 특정 교차 운영인 SWAN을 사용하는 유전 알고리즘을 기반으로 디자인을 정제합니다.

- **Performance Highlights**: 이 연구는 Design-o-meter의 효과성을 정량적 및 정성적으로 분석하며, 기존의 다중 모달 LLM 기반 접근 방식들과 비교하여 우수한 성능을 보입니다. 특히, 디자인 문서의 미세한 뉘앙스를 포착할 수 있는 scorer의 동작과 복잡한 디자인 공간을 효과적으로 탐색하는 refiner의 성능을 강조합니다. 종합적으로, Design-o-meter는 디자인 품질 측정 및 개선의 혁신적인 방법을 제공할 것으로 기대됩니다.



### Evaluating Vision Transformer Models for Visual Quality Control in Industrial Manufacturing (https://arxiv.org/abs/2411.14953)
- **What's New**: 이번 연구는 산업 제조 분야의 시각 품질 관리 시스템에서 사용될 수 있는 최신의 비전 트랜스포머 모델과 이상 탐지 방법들을 종합적으로 검토하고 평가합니다. 다양한 트랜스포머 아키텍처와 이상 탐지 기법을 조합하여 소형, 빠르고 효율적인 모델을 개발하여 실용적인 품질 관리 시스템 구축에 도움을 제공합니다. 이를 통해 품질 관리 시스템에서 필요한 모델 구조 선택에 대한 가이드라인도 제시합니다.

- **Technical Details**: 연구에서는 Unsusupervised anomaly detection (AD) 및 anomaly localization (AL) 기법을 활용하여 드문 결함 품목을 식별하는 방법론을 다룹니다. 일반적으로 이상 탐지를 위해 이미지 인코딩 백본과 탐지 알고리즘이 조합되어 사용됩니다. 최신 비전 트랜스포머 모델은 고해상도 이미지 처리 시 발생하는 부족한 스케일링 문제를 해결하며, 전통적인 CNN 아키텍처에 비해 더 나은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 실험 결과, MVTecAD 및 BTAD 데이터셋을 이용한 평가에서 최적의 모델 조합을 찾는데 성공하였습니다. 특히, 연구에서 검토한 계층적 비전 트랜스포머 모델들이 산업적 품질 관리에서 필요한 시간 및 자원 절감과 더불어 탁월한 성능을 보였습니다. 이 연구는 예측 정확성을 높이고 수동 검사를 줄임으로써 제조 산업에서 큰 이점을 제공할 수 있는 가능성을 보여줍니다.



### Morph: A Motion-free Physics Optimization Framework for Human Motion Generation (https://arxiv.org/abs/2411.14951)
Comments:
          15 pages, 6 figures

- **What's New**: 이 논문에서는 Motion-free physics optimization framework인 Morph를 제안합니다. Morph는 Motion Generator(모션 생성기)와 Motion Physics Refinement(모션 물리 정제 모듈)으로 구성되어 있어, 실제 모션 데이터에 의존하지 않으면서도 물리적 신뢰성을 높이는 데 중점을 둡니다. 이 방법론은 기존의 모션 생성 접근 방식의 한계를 극복하려 시도하고, 물리적 제약을 명시적으로 적용하여 생성된 모션의 품질을 개선합니다.

- **Technical Details**: Morph는 두 가지 주요 모듈로 나뉘어 있습니다. 첫 번째는 사용자가 원하는 다양한 신호(예: 텍스트, 음악)에 따라 모션 데이터를 생성하는 Motion Generator이며, 두 번째는 이러한 데이터를 기반으로 물리적 제약을 적용하여 모션을 정제하는 Motion Physics Refinement 모듈입니다. Morph의 두 단계 최적화 과정은 노이즈가 있는 모션을 물리적으로 신뢰할 수 있는 공간으로 변환하고, 이를 통해 생성된 정제된 모션은 모션 생성기의 훈련에 사용됩니다.

- **Performance Highlights**: Morph는 텍스트-모션 및 음악-댄스 생성 작업에서 성능을 평가하였습니다. 실험 결과, Morph는 기존 모델에 비해 물리적 오류 지표에서 중대한 개선을 보였으며, 다양한 생성 모델에서 경쟁력 있는 생성 품질을 달성했습니다. 특히, 실제 모션 데이터에 의존하지 않고도 우수한 결과를 도출할 수 있음이 입증되었습니다.



### Reliable Evaluation of Attribution Maps in CNNs: A Perturbation-Based Approach (https://arxiv.org/abs/2411.14946)
- **What's New**: 이번 연구는 CNN(Convolutional Neural Network)의 Attribution Map(AM, 기여도 맵) 평가에 대한 새로운 접근 방식을 제시합니다. 기존의 삽입/삭제 메트릭이 분포 이동(Distribution Shift)에 민감하다는 점을 강조하며, 이를 해결하기 위해 픽셀 수정을 적대적 섭동(Adversarial Perturbations)으로 대체하는 방법을 제안합니다. 이 연구는 AM의 신뢰성과 일관성을 높이는 사람의 이해가 가능한 평가 프레임워크를 제공합니다.

- **Technical Details**: AM은 이미지 입력 데이터의 각 이미지에 대한 기여를 시각적으로 표현하여 신경망의 결정 과정을 설명하는 도구입니다. 그러나 AM의 정량적 평가 및 올바른 성능 메트릭 정의는 여전히 난제입니다. 본 연구는 다양한 메트릭을 활용해 15개 데이터셋 및 아키텍처 조합에 걸쳐총 16개의 AM을 평가하였으며, Kendall의 순위 상관 계수(τ)를 통해 신뢰성을 테스트하였습니다.

- **Performance Highlights**: 본 연구의 결과, SmoothGrad는 16개의 AM 중에서 최상의 성능을 보였으며, 이는 AM의 질적 및 양적 평가를 통합한 가장 포괄적인 분석을 기반으로 합니다. 또한, 기초 기준 AM을 도입하여 평가의 신뢰성을 높이는 방법을 제시하였으며, 코드와 결과를 함께 제공하여 재현성을 보장합니다.



### LiDAR-based End-to-end Temporal Perception for Vehicle-Infrastructure Cooperation (https://arxiv.org/abs/2411.14927)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 LiDAR 기반의 End-to-End Tracking 프레임워크인 LET-VIC를 소개합니다. LET-VIC는 차량과 인프라 간의 협력을 통해 동적 환경에서의 시간적 인식을 개선하도록 설계되었습니다. 이 프레임워크는 V2X 통신을 활용하여 차량과 인프라 센서 데이터를 융합하고, 이러한 통합을 통해 가려진 객체와 관측 블라인드 스팟 문제를 해결합니다.

- **Technical Details**: LET-VIC는 차량 측과 인프라 측의 LiDAR 데이터를 Bird's Eye View (BEV) 형식으로 통합하여 종합적인 시각을 제공합니다. 이 시스템은 또한 프레임 간의 시간적 맥락을 통합하여 역사적 데이터를 활용하여 추적의 안정성과 정확성을 높입니다. 특히, Calibration Error Compensation (CEC) 모듈을 도입하여 센서 간 불일치를 보정하여 정밀한 특징 정렬을 보장합니다.

- **Performance Highlights**: 실험 결과, LET-VIC는 V2X-Seq-SPD 데이터셋에서 기존의 기본 모델보다 최소 13.7%의 mAP 개선과 13.1%의 AMOTA 개선을 달성하였습니다. LET-VIC는 통신 지연을 고려하지 않더라도 강력한 안정성과 적응력을 보여 주며, 다양한 지연 조건에서도 일관된 성능을 유지합니다. 이러한 성과는 LET-VIC가 고도화된 자율 주행 시스템의 잠재력을 보여줍니다.



### ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos (https://arxiv.org/abs/2411.14901)
- **What's New**: 이 논문에서는 ReVisionLLM이라고 불리는 새로운 recursive vision-language model을 도입합니다. 이 모델은 시간적 정보가 중요한 1시간 분량의 비디오에서 사건을 정확히 위치시키는 능력을 가지고 있습니다. ReVisionLLM은 인류의 검색 전략에서 영감을 받아 넓은 관심 구역에서 시작해 세부적인 사건의 경계를 점진적으로 수정하는 방식으로 작동합니다.

- **Technical Details**: ReVisionLLM은 구조적으로 재귀적(recursive)으로 비디오를 처리하며, 여러 계층에 걸쳐 인식 위계(hierarchical perception)를 도입합니다. 초기에는 관심있는 비디오 구간을 대략적으로 식별하고, 다음 단계로 이동하면서 점점 더 세밀한 시간적 단위로 맞춰 사건의 경계를 정확히 나타냅니다. 또한, 간단한 비디오 클립으로 시작하여 점진적으로 긴 비디오로 확장하는 교육 전략을 사용하여 훈련을 최적화합니다.

- **Performance Highlights**: 모델의 성능은 여러 데이터셋에서 이전의 최첨단 방법들을 상당 폭 초과하면서 입증됩니다. 예를 들어, ReVisionLLM은 MAD 데이터셋에서 기존의 최고 성능 방법보다 2.6% 더 높은 R1@0.1을 기록하였습니다. 이 모델은 평균적으로 기존 VLM 모델보다 43% 적은 프레임을 처리하면서도 높은 정확도와 효율성을 달성합니다.



### Boundless Across Domains: A New Paradigm of Adaptive Feature and Cross-Attention for Domain Generalization in Medical Image Segmentation (https://arxiv.org/abs/2411.14883)
Comments:
          5 pages, 3 figures

- **What's New**: 이번 연구에서는 domain generalization (DG) 문제를 해결하기 위해 Adaptive Feature Blending (AFB)이라는 새로운 방법론을 제안합니다. AFB는 source domain에서 발생하는 feature 및 스타일 정보를 활용하여 out-of-distribution 샘플을 생성하고, 이를 통해 domain distribution의 다양성을 크게 확대합니다. 또한, Dual Cross-Attention Regularization (DCAR)을 사용하여 domain-invariant representation을 학습하는 강력한 모델을 구축합니다.

- **Technical Details**: 이 연구에서는 cross-channel attention 메커니즘을 활용하여 domain-invariant representation을 학습하는 방법론을 제시합니다. 제안된 DCAR는 source domain에서 얻은 deep features를 쿼리로, 생성된 도메인 이미지의 deep features를 키와 값으로 사용하여 특정 채널 간 유사성 매트릭스를 계산합니다. 이를 기반으로 원래의 deep features를 강력한 정규화 표현으로 재구성하여 domain-invariant representation을 학습합니다.

- **Performance Highlights**: 제안된 방법론은 두 개의 표준 domain generalization 벤치마크 데이터셋에서 의료 이미지 분할 작업의 성능을 향상시키는 데 뛰어난 효율성을 보여줍니다. extensive experimental results에 따르면, Adaptive Feature Blending와 Dual Cross-Attention Regularization 기법이 기존의 방법보다 우수한 성능을 달성했습니다. 이러한 결과는 의료 이미징 분야의 실제 적용 가능성을 높이는 데 기여할 것으로 예상됩니다.



### Prioritize Denoising Steps on Diffusion Model Preference Alignment via Explicit Denoised Distribution Estimation (https://arxiv.org/abs/2411.14871)
- **What's New**: 본 논문에서는 Denoised Distribution Estimation (DDE)이라는 새로운 기법을 제안합니다. DDE는 기존의 보조 모델이나 수작업 방식에 의존하지 않고, 각 디노이징 단계에서의 관점으로부터 말단의 디노이즈 분포를 직접적으로 추정합니다. 이를 통해 스프의 선호 레이블이 드물게 제공되는 환경에서도 보다 효율적인 신용 할당이 가능해집니다.

- **Technical Details**: DDE는 두 가지 추정 전략, 즉 단계별(stepwise) 및 일괄(single-shot) 추정을 제안합니다. 단계별 추정은 조건부 분포를 기반으로 하여 모델 분포를 추정하는 반면, 일괄 추정은 DDIM 모델링을 사용하여 중간 노이즈 상태를 바탕으로 말단 분포를 직접 평가합니다. 이러한 두 방법을 통합함으로써 모델 추론을 통한 전체 디노이징 궤적의 평가가 가능해집니다.

- **Performance Highlights**: 우리는 SD15와 SDXL에서 DDE를 평가하였고, 그 결과 기존의 보조 모델이 없는 방법들에 비해 뛰어난 성능을 입증하였습니다. DDE는 SD15와 SDXL의 성능 지표를 각각 3.3%에서 6.7%, 1.0%에서 3.1% 향상시켰습니다. 전반적으로 DDE는 기존 접근 방식과 비교할 때 수치적, 질적으로 최고 수준의 성능을 보여주었습니다.



### BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligenc (https://arxiv.org/abs/2411.14869)
- **What's New**: 이번 논문에서는 BIP3D라는 새로운 이미지 중심 3D 지각 모델을 제안합니다. 현재의 포인트 클라우드 접근법의 한계를 극복하기 위해, 이미지 특성과 명시적인 3D 위치 부호화를 결합했습니다. BIP3D는 다중 뷰 이미지와 텍스트 특성을 융합하여 3D 객체 감지 및 3D 시각적 기반 작업을 수행하며, 깊이 맵을 보조 입력으로 사용할 수 있습니다.

- **Technical Details**: BIP3D는 GroundingDINO 모델을 기반으로 하며, 카메라 모델을 구성하고 다중 뷰 데이터 및 여러 모드를 융합하는 기능을 최적화했습니다. 카메라 매개변수를 지원하여 2D 이미지 특성에 3D 위치 부호화를 추가하고, 2D 비대칭 주의(attention) 기법을 3D 형태로 수정하여 다이나믹한 다중 뷰 특성 융합을 구현했습니다.

- **Performance Highlights**: 실험 결과, BIP3D는 EmbodiedScan 벤치마크에서 현재 최고 성능을 달성하여, 3D 감지 작업에서 5.69%의 성능 향상을 보였고, 3D 시각적 기반 작업에서는 15.25%의 향상을 기록했습니다. 이 모델은 또한 RGB 전용 입력을 지원하여 Crowd-sourcing을 통해 대량의 데이터 수집이 가능합니다.



### Defective Edge Detection Using Cascaded Ensemble Canny Operator (https://arxiv.org/abs/2411.14868)
Comments:
          2 Pages and 2 Figures

- **What's New**: 이번 연구에서는 복잡한 장면 사진에서 발생하는 기존의 에지 탐지(Edge Detection) 알고리즘의 한계를 극복하기 위해 Cascaded Ensemble Canny operator를 제안했습니다. 이는 다양한 형태와 크기의 객체가 포함된 실제 이미지의 경계 및 에지를 정확히 식별하는 데 중점을 두었습니다.

- **Technical Details**: 제안된 방법에서는 여러 백본(backbones)과 주목(attention) 모듈을 결합한 앙상블 학습(Ensemble Learning) 접근 방식을 사용하여 에지 탐지 성능을 향상시킵니다. 특히 Fresh and Rotten과 Berkeley 데이터셋을 활용하여 Python으로 구현된 알고리즘을 비교 및 테스트하였으며, 에지 탐지의 정확성을 높였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 Sobel 및 Canny 에지 탐지 네트워크에 비해 성능 지표(Performance Metrics)와 출력 이미지 품질에서 우수한 성능을 보여주었습니다. 이는 더 나은 에지 탐지 결과를 통해 실제 애플리케이션에서도 유용하게 적용될 수 있음을 시사합니다.



### Latent Schrodinger Bridge: Prompting Latent Diffusion for Fast Unpaired Image-to-Image Translation (https://arxiv.org/abs/2411.14863)
- **What's New**: 이 논문에서는 Schrodinger Bridges (SBs)를 활용하여 이미지 생성 및 데이터에서의 변환 문제를 해결하는 새로운 Latent Schrödinger Bridges (LSBs) 알고리즘을 제안하고 있습니다. 이 알고리즘은 기존의 Diffusion Models (DMs)보다 적은 수의 Neural Function Evaluations (NFEs)로 경쟁력 있는 이미지 간 변환(I2I translation)을 수행할 수 있습니다. 또한, SB의 확률 흐름을 활용하여 효율적인 계산을 가능하게 하여 실용성을 높이고 있습니다.

- **Technical Details**: Schrodinger Bridges는 두 분포 간의 최소한의 전송 비용을 통해 stochastic differential equations (SDEs)를 학습하는 방법입니다. 이 논문에서는 SB의 확률 흐름 ODE 벡터 필드를 소스 예측기, 타겟 예측기 및 노이즈 예측기의 선형 결합으로 분해하는 방법을 제안합니다. 이를 통해 Latent Diffusion Models (LDMs)와 결합하여 효율적인 ODE 근사를 제공하는 Latent Schrödinger Bridges (LSBs)를 개발하였습니다.

- **Performance Highlights**: 제안한 LSB 알고리즘은 기존 DM 기반의 이미지 간 변환 방법보다 현저히 적은 계산 비용으로 경쟁력 있는 성능을 보여줍니다. 실험 결과, LSB는 여러 데이터셋에서 높은 확장성을 가지며, diffusion-based I2I translation에서 탁월한 성능을 발휘함을 입증했습니다.



### Dynamics-Aware Gaussian Splatting Streaming Towards Fast On-the-Fly Training for 4D Reconstruction (https://arxiv.org/abs/2411.14847)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 4D 동적 공간 재구성을 위한 새로운 3단계 파이프라인(DASS)을 제안합니다. 이 방법은 기존의 전신 멀티뷰 비디오에 의존하지 않고, 현재 프레임에서의 재구성을 이전 캐시와 현재 멀티뷰 입력을 기반으로 진행합니다. 또한, 시간적 연속성을 고려하여 선택적 상속 메커니즘을 통해 이전 프레임에서 추가된 Gaussians를 보존합니다.

- **Technical Details**: 3단계 파이프라인은 상속(inheritance), 이동(shift), 밀도화(densification)로 구성됩니다. 각 단계에서는 새로운 Gaussians가 이전 프레임의 Gaussians를 선택적으로 상속받고, 2D 동적 오프셋과 Gaussian 분할을 사용하여 움직이는 객체와 정적인 객체를 구분하여 최적화합니다. 마지막 단계에서는 새로운 객체를 수용하기 위해 새 Gaussians를 추가하고, 이 과정에서 위치 기울기와 오류 맵을 고려합니다.

- **Performance Highlights**: 이 방법은 기존의 동적 공간 재구성 방법들에 비해 20% 향상된 온라인 훈련 속도, 높은 재구성 품질, 실시간 렌더링 능력을 입증하였습니다. 이 연구는 실시간으로 동적 장면을 처리하는 데 중요한 진전을 보여주며, AR/VR 및 홀로그램 통신 등 다양한 실용적 응용에 기여할 수 있습니다.



### VisGraphVar: A Benchmark Generator for Assessing Variability in Graph Analysis Using Large Vision-Language Models (https://arxiv.org/abs/2411.14832)
- **What's New**: 이 논문은 LVLMs(Large Vision-Language Models)의 시각 그래프 분석을 위한 새로운 기준 생성기인 VisGraphVar(Visual Graph Variability)를 소개합니다. 이는 서로 다른 그래프 이미지 스타일과 구조를 생성하여 LVLM의 강점과 약점을 체계적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: VisGraphVar는 노드 및 엣지 탐지, 그래프 유형 분류, 세분화, 패턴 인식, 링크 예측, 추론, 일치 확인 등 총 7개의 작업 범주로 구성되어 있습니다. 이 생성기에 의해 만들어진 990개의 그래프 이미지를 통해 6개의 LVLM을 평가하였으며, 줄 무샘과 사고 사슬의 두 가지 다른 프롬프트 전략을 사용하였습니다.

- **Performance Highlights**: 실험 결과, 이미지의 시각적 속성 변화(예: 노드 레이블링 및 레이아웃)와 의도적으로 포함된 시각적 결함(예: 중첩된 노드)이 모델 성능에 상당한 영향을 미치는 것으로 나타났습니다. 이 연구는 LVLM의 시각 그래프 분석 능력을 보다 신뢰성 있고 견고한 시스템으로 발전시키기 위한 통찰력을 제공합니다.



### Physically Interpretable Probabilistic Domain Characterization (https://arxiv.org/abs/2411.14827)
- **What's New**: 본 논문에서는 자율주행 차량 환경에서 도메인 특성을 확률 분포로 캐릭터화하는 새로운 접근 방식을 제안합니다. 이는 기존의 회귀 혹은 분류 문제 접근법의 한계를 넘어서, 물리적 매개변수의 분포를 추정하여 차별화된 도메인 특성을 제공하는 방법입니다. 특히, 노멀라이징 플로우(normalizing flows)를 사용하여 차량에 장착된 카메라에서 촬영한 이미지로부터 다양한 기상 조건의 예측 가능성을 높입니다.

- **Technical Details**: 저자들은 카메라로 수집된 고급 이미지 데이터를 기반으로 실시간으로 기상 매개변수를 예측하는 세 가지 작업을 제안합니다. 이 방법은 고차원 데이터에서의 우도 추정의 비효율성을 극복하고, 이미지와 날씨 분포에서 날씨 매개변수를 모델링하기 위해 노멀라이징 플로우를 활용합니다. 이를 통해 단일 이미지에서의 기상 조건 예측뿐만 아니라, 여러 이미지의 집합을 통한 절대 및 상대 도메인 특성화가 가능합니다.

- **Performance Highlights**: 제안된 방법은 자율주행 시스템이 다양한 환경 조건에 적응할 수 있도록 하며, 안전한 작동을 보장합니다. 여러 도메인 간 비교를 통해 특정 도메인 내에서의 안전한 작동 가능성을 평가할 수 있으며, 이는 자율주행차의 실제 운전 조건과 관련하여 큰 실용적 응용 가능성을 제공합니다. 본 연구는 기상 예측의 정확성과 효과적인 도메인 적응이 자율 시스템의 동적 환경 조정에 얼마나 중요한지를 강조합니다.



### Omni-IML: Towards Unified Image Manipulation Localization (https://arxiv.org/abs/2411.14823)
- **What's New**: 이 논문에서는 Omni-IML이라는 첫 번째 일반화 모델을 제안하여 다양한 이미지 조작 탐지(Image Manipulation Localization, IML) 작업을 통합했습니다. 기존의 IML 방법들은 특정 이미지 유형에 국한되어 있어, 여러 유형의 이미지에서 성능이 저조합니다. Omni-IML은 Modal Gate Encoder와 Dynamic Weight Decoder를 활용해 입력 샘플에 최적화된 인코딩 방식 및 디코더 필터를 선택함으로써 일관된 성능을 발휘합니다.

- **Technical Details**: Omni-IML은 공통 특성 추출을 위한 Anomaly Enhancement 모듈 또한 도입합니다. 이 모듈은 조작된 영역의 특징을 향상시키고 이미지의 잡음 요소를 억제합니다. 세 가지 주요 IML 작업인 자연 이미지, 문서 이미지 및 얼굴 이미지에서 성능을 검증하였으며, 단일 모델로 모든 작업에서 최첨단 성능을 달성했습니다. 이 모델은 일반화 성능을 높이며 유지 관리 비용을 줄여주는 효용성을 지니고 있습니다.

- **Performance Highlights**: 모든 실험 결과에서 Omni-IML은 기존의 전문화된 모델보다 월등한 성과를 보였습니다. 일반적인 IML 모델이 달성하기 어려운 다양한 조작 기법을 한 모델에서 효과적으로 처리할 수 있음을 보여주었습니다. 이러한 결과는 이미지 포렌식 분야에서의 일반화 모델의 설계 유효성을 검증하며, 장기적으로 인공지능 일반 지능(Artificial General Intelligence, AGI) 발전에 기여할 수 있는 가능성을 제시합니다.



### Unsupervised Multi-view UAV Image Geo-localization via Iterative Rendering (https://arxiv.org/abs/2411.14816)
Comments:
          13 pages

- **What's New**: 본 연구에서는 UAV Cross-View Geo-Localization (CVGL) 문제를 해결하기 위한 새로운 비지도 학습 접근법을 제안합니다. 이 방법은 UAV에서 관측한 장면을 3D 공간으로 확장하여 위성 이미지를 생성함으로써 시각적 왜곡에 강한 표현을 제공합니다. 기존의 방법들이 가진 영역 특화 과적합 문제를 자연스럽게 회피하며, 추가적인 데이터 기반 훈련이나 기능 미세 조정 없이 UAV 이미지에 대한 일반적인 CVGL을 가능하게 합니다.

- **Technical Details**: 우리의 접근법은 3D 신경장(field)에서 UAV 관측을 기반으로 장면 구조를 복원하고 질높은 위성 이미지를 생성하여 보기 불일치를 줄입니다. 특히, 3D Gaussian Splatting (3DGS) 모델을 통해 드론 보기 관측을 3D 장면으로 나타내어 드론-위성 뷰를 위한 공유 공간을 형성합니다. 실측 위성 피사체와의 정렬을 위해 점진적으로 가상 카메라 포즈를 수정하는 반복적인 업데이트 메커니즘을 도입하여 기능의 시각적 일관성을 강화합니다.

- **Performance Highlights**: University-1652 및 SUES-200 데이터세트를 통한 실험 결과, 우리의 방법이 기존의 SOTA 제로샷 방법보다 뛰어난 지리적 위치 정확도를 보이며, 특정 작업을 위해 감독된 방법들과 경쟁력 있는 성능을 보여줍니다. 우리의 비지도 접근법은 모델 미세 조정이나 짝지어진 훈련 없이도 일반화 능력을 키워 다양한 지역에서의 성능을 유지하는 것으로 입증되었습니다.



### Fine-Grained Alignment in Vision-and-Language Navigation through Bayesian Optimization (https://arxiv.org/abs/2411.14811)
- **What's New**: 이 논문은 Vision-and-Language Navigation (VLN) 과제에서의 섬세한 조정(fine-grained alignment) 문제를 다룹니다. 기존의 접근 방식들이 언어와 시각적 경로 시퀀스를 조정하는 데 어려움을 겪고 있는 가운데, 저자들은 독창적인 Bayesian Optimization(based) 적대적 최적화(adversarial optimization) 프레임워크를 도입하여 섬세한 대조적 시각 샘플을 생성합니다. 이를 통해 생성된 임베딩(enriched embeddings)이 VLN 과제의 전반적인 성능 강화를 이끈다는 것을 실험을 통해 입증합니다.

- **Technical Details**: 본 연구에서는 기존 VLN 접근 방식의 한계인 대조적 샘플의 품질을 향상시키는 데 초점을 맞추고 있습니다. 특히, 저자들은 Bayesian Optimization(BO)을 활용하여 시각 시퀀스에서 가장 영향력이 큰 요소를 찾아내고 이를 대체하여 섬세한 시각 부정 샘플을 형성하는 방법론을 제안합니다. 이 과정은 훈련 과정에서 신중한 샘플링을 가능하게 하고, 탁월한 경로 선택을 도와 시각-언어 정렬(vision-language alignment) 능력을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크를 통해 훈련된 인코더는 섬세한 시각 정보를 보다 잘 포착하는 것으로 나타났습니다. R2R(Room-to-Room) 및 REVERIE와 같은 일반 VLN 벤치마크에서 성능 향상이 확인되었습니다. 이는 섬세한 샘플(fine-grained samples)의 중요성을 강조하며 기존의 대조적 부정 샘플보다 향상된 성능을 달성하였음을 의미합니다.



### High-Resolution Image Synthesis via Next-Token Prediction (https://arxiv.org/abs/2411.14808)
Comments:
          30 pages

- **What's New**: D-JEPA (Denoising with a Joint-Embedding Predictive Architecture)는 이미지 생성에서 뛰어난 성능을 보여주는 자기회귀 모형입니다. 본 논문에서는 D-JEPA의 확장 모델인 D-JEPA⋅T2I를 소개하는데, 이는 flow matching loss를 적용하여 데이터 효율적 연속 해상도 학습을 가능하게 합니다. 또한, 멀티모달 비주얼 트랜스포머를 이용하여 텍스트와 비주얼 특징을 효과적으로 통합합니다.

- **Technical Details**: D-JEPA⋅T2I는 새로운 모델 아키텍처와 데이터 활용 방식으로 설계되었습니다. 특히, Visual Rotary Positional Embedding (VoPE)을 도입하여 연속 해상도 학습을 지원하며, 데이터 피드백 메커니즘을 통해 데이터 활용도를 극대화합니다. 이는 전통적인 데이터 처리와 차별화되는 접근법으로, 실시간 통계 분석을 통한 데이터 배급 조정이 특징입니다.

- **Performance Highlights**: D-JEPA⋅T2I는 T2I-CompBench, GenEval, GenAI-Bench 벤치마크에서 고해상도 이미지 합성 분야에서 최첨단의 성능을 달성했습니다. 특히, 다음-토큰 예측(next-token prediction)을 통해 이미지 품질을 크게 개선하였으며, 기존의 autoregressive 모델과 비교해 이미지 텍스처와 품질 면에서 우월한 성능을 보입니다.



### Harlequin: Color-driven Generation of Synthetic Data for Referring Expression Comprehension (https://arxiv.org/abs/2411.14807)
Comments:
          Accepted to ICPR 2024

- **What's New**: 이번 연구에서는 Referring Expression Comprehension (REC) 작업을 위한 새로운 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 텍스트 및 시각적 변수를 모두 고려하여 인공 데이터를 생성합니다. 새롭게 생성된 데이터 세트는 Harlequin이라는 이름을 가지고 있으며, 1백만 개 이상의 쿼리로 구성되어 있습니다. 이 연구는 데이터 수집 및 주석 작업이 수작업으로 이루어지는 것을 피할 수 있음을 강조합니다.

- **Technical Details**: 저자들은 두 가지 주요 모듈로 구성된 파이프라인을 설계했습니다. 첫 번째 모듈인 Annotation Generation Engine은 일관된 바운딩 박스 주석이 있는 새로운 Referring Expressions를 생성합니다. 두 번째 모듈인 Image Generation Engine은 이전 단계에서 얻은 주석 정보를 바탕으로 새로운 이미지를 생성합니다. 이렇게하여 개발된 Harlequin 데이터 세트는 완전히 합성 공정으로 생성된 REC 작업을 위한 첫 번째 데이터 세트입니다.

- **Performance Highlights**: 실험 결과, Harlequin 데이터 세트를 이용한 사전 학습이 실제 데이터에서의 성능을 향상시키는 데 기여함을 보여주었습니다. 이를 통해 주석 수집에서의 노력과 오류를 줄이는 것이 가능해졌습니다. 저자들은 데이터 및 코드를 공개하여 향후 연구자들이 이 결과를 토대로 추가 연구를 진행할 수 있도록 하고 있습니다.



### Facial Features Matter: a Dynamic Watermark based Proactive Deepfake Detection Approach (https://arxiv.org/abs/2411.14798)
- **What's New**: 이 논문에서는 기존의 수동적인 딥 페이크(face fake) 탐지 기술의 한계를 극복하기 위해 근본적으로 다른 접근 방식을 제안합니다. 새로운 방법인 Facial Feature-based Proactive deepfake detection method (FaceProtect)을 통해 얼굴 특징을 검증 데이터로 활용하여 탐지 정확도를 높입니다. 또한, GAN 기반의 One-way Dynamic Watermark Generating Mechanism (GODWGM)을 도입하여, 얼굴 특징을 동적으로 변형된 워터마크로 변환함으로써 보안을 강화했습니다.

- **Technical Details**: 제안된 방법은 128차원 얼굴 특징 벡터를 입력으로 사용하여 얼굴 특징에서 워터마크로의 전이 가능성을 높이고, 이를 통해 역 추론 공격에 대한 저항력을 강화합니다. 착안점을 제공하는 얼굴 곡선(예: 동일 인물의 다양한 표정 간의 차이)을 활용하여 원본 이미지와 혼합된 이미지에서 워터마크를 추출할 수 있는 심층 가변 신경망 구조를 구현했습니다. 새로운 워터마크 기반 검증 전략(WVS)을 통해 워터마크의 동기화 문제를 해결하고, 숨길 수 있는 네트워크를 사용하여 무결성을 유지하면서 워터마크를 이미지에 효과적으로 삽입합니다.

- **Performance Highlights**: 실험 결과, 제안된 FaceProtect 방법은 다양한 딥페이크 기술로 변경된 이미지에서도 우수한 탐지 성능을 유지합니다. 특히, GAN 및 워터마크 기반 검증 전략을 통해 탐지 과정에서 보안성이 강화된 것으로 나타났습니다. 이로 인해 딥페이크 탐지의 실제 적용 가능성을 높이며, 다양한 환경에서의 활용이 기대됩니다.



### Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections (https://arxiv.org/abs/2411.14796)
- **What's New**: 이번 연구에서는 인간 골격의 구조를 반영한 하이퍼 그래프 컨볼루션 네트워크(Hyper-GCN)를 제안하여, 행동 인식을 위한 성능을 향상시킵니다. 전통적인 그래프 컨볼루션 네트워크(GCN)가 두 개의 인접 정점 간의 이진 연결에 의존하는 것에서 벗어나, 다중 정점 간의 관계를 보다 잘 표현할 수 있는 방법을 탐구하고 있습니다. 하이퍼 그래프를 사용하여 다양한 행동 범주에 대한 의미론적 힌트를 강조할 수 있는 가상 연결을 도입하게 됩니다.

- **Technical Details**: 하이퍼 그래프는 이진 연결 대신 다중 정점의 관계를 설명할 수 있는 구조로, 이를 통해 인간 행동을 정의하는 다양한 조인트의 상호작용을 포착할 수 있습니다. Hyper-GCN은 학습 과정에서 다중 스케일 하이퍼 그래프를 최적화하여 행동에 따른 다중 정점 관계를 드러내며, 이 과정에서 가상 연결을 통해 골격 내의 의존성을 확장할 수 있습니다. 이러한 방법론은 기존 GCN에 비해 정보의 집계 효율성을 극대화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, NTU-60, NTU-120 및 NW-UCLA 데이터셋에서 Hyper-GCN은 최신 기술(SOTA)과 비교하여 뛰어난 성능을 보여줍니다. 특히, NTU-120 데이터셋에서 X-Sub 및 X-Set에 대한 Top-1 인식 정확도가 각각 90.2% 및 91.4%에 이릅니다. 이와 같은 결과는 하이퍼 그래프 기법이 행동 인식 분야에서 기존 솔루션들보다 향상된 성능을 제공할 수 있다는 것을 시사합니다.



### VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection (https://arxiv.org/abs/2411.14794)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 VideoQA(비디오 질문-답변) 작업의 한계를 극복하기 위해 VideoEspresso라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 중요한 공간적 세부정보와 시간적 일관성을 보존하며, 중간 추론 단계를 포함한 다중 모드 주석(annotation)을 특징으로 합니다. 또한, 저자들은 Hybrid LVLMs Collaboration 프레임워크를 제안하여 질의에 적합한 핵심 프레임을 선택하고, 체인 오브 사고(reasoning) 접근법을 통해 비디오 내용을 기반으로 논리적 관계를 추출합니다.

- **Technical Details**: VideoEspresso의 구축은 자동화된 파이프라인을 통해 이루어지며, 이를 통해 효율적으로 QA 쌍을 생성합니다. 처음에는 LVLM을 사용하여 비디오 프레임을 언어 공간으로 매핑한 후, 의미론적 유사성을 바탕으로 중복 프레임을 제거합니다. 이어서 GPT-4o를 활용하여 QA 쌍을 생성하고 저품질 데이터를 필터링합니다. 중간 논리 과정을 확장하기 위해 Video Chain-of-Thought 주석을 도입하여 이러한 정보가 QA 쌍 형성에 기여하도록 설계되었습니다.

- **Performance Highlights**: 새롭게 제안된 평가 기준을 기반으로 9개의 인기 LVLM을 대상으로 14개의 작업에서 성능을 평가한 결과, 제안된 방법이 대부분의 작업에서 기존 방법들을 초월하는 것으로 나타났습니다. 이러한 평가를 통해 VideoEspresso의 고품질 VideoQA 쌍이 복합적인 비디오 추론 작업에서 우수한 성능을 발휘함을 보였습니다. 코드 및 데이터셋은 추후 공개될 예정입니다.



### Style-Friendly SNR Sampler for Style-Driven Generation (https://arxiv.org/abs/2411.14793)
- **What's New**: 최근 대규모 확산 모델이 고품질 이미지를 생성하지만 새로운 개인화된 예술 스타일을 학습하는 데 어려움을 겪고 있다는 점을 지적합니다. 이를 해결하기 위해 Style-friendly SNR sampler를 제안하여, 스타일 특성이 나타나는 고노이즈 레벨에 집중하도록 신호 대 잡음을 조정합니다. 이 방법은 모델이 독특한 스타일을 더 잘 포착하고 높은 스타일 정합성을 가진 이미지를 생성할 수 있게 해 줍니다.

- **Technical Details**: 확산 모델의 훈련 과정에서, 데이터가 순차적으로 순수 노이즈로 변환되는 과정을 설명합니다. 최근의 최신 확산 모델들은 정체된 흐름에서 유도된 잡음 스케줄을 사용하여, 신호 대 잡음 비율을 로그-신호 대 잡음 비율을 기반으로 조정함으로써 스타일 학습의 효율을 높입니다. 새로운 접근법은 각 잡음 레벨의 중요성을 신중히 고려하여 모델의 손실 함수를 조정함으로써 이루어집니다.

- **Performance Highlights**: 이 연구는 FLUX-dev와 Stable Diffusion 3.5와 같은 최신 모델들이 스타일 템플릿을 효과적으로 학습할 수 있게 만든다는 것을 강조합니다. 또한, 개인 수채화, 미니멀한 만화, 3D 렌더링 및 텍스트가 포함된 밈과 같은 다양한 스타일의 이미지를 생성하는 능력을 시연하여 스타일 기반 생성의 폭을 넓혔습니다. 이 방안은 사용자 및 제작자들이 쉽게 접근할 수 있는 새로운 "스타일 템플릿"을 생성하고 공유할 수 있는 토대를 마련합니다.



### Reconciling Semantic Controllability and Diversity for Remote Sensing Image Synthesis with Hybrid Semantic Embedding (https://arxiv.org/abs/2411.14781)
- **What's New**: 이번 논문은 원격 감지에서의 의미적 이미지 합성에 대한 새로운 접근법인 하이브리드 의미 임베딩 가이드 생성 적대망(HySEGGAN)을 제안합니다. 특히 HySEGGAN은 단일 소스에서 계층적 정보를 활용하여 의미적 제어 가능성과 다양성을 달성합니다. 이 방법은 세밀한 지역 의미 레이아웃을 조정하여 추가 정보 없이도 원격 감지 객체의 기하학적 구조를 특징화합니다.

- **Technical Details**: HySEGGAN은 세 가지 주요 구성 요소를 도입합니다: 기하학적 정보를 기반으로 한 공간 기술자(GSD), 하이브리드 의미 임베딩 네트워크(HSGNet), 그리고 의미 정제 네트워크(SRN)입니다. HSGNet은 의미 마스크를 GSD로 변환하여 다양성과 세밀한 제어를 향상시키고, SRN은 새로운 손실 함수를 통합하여 세밀한 의미 피드백을 보장합니다. 전체적으로 이 접근법은 의미적 혼란을 줄이고 기하학적 패턴 붕괴를 방지합니다.

- **Performance Highlights**: 실험 결과, HySEGGAN은 원격 감지 이미지 합성에서 뛰어난 화질과 함께 의미적 제어 가능성과 다양성 간의 균형을 이룹니다. 정량적 평가에 따라, HySEGGAN은 GID-15와 ISPRS 데이터셋에서의 성능을 향상시켜 다운스트림 태스크인 의미 세분화에서 효과적인 데이터 보강 기술로 자리 잡았습니다. 또한 이 방법은 추가적인 주석 없이도 세분화 성능을 개선할 수 있는 가능성을 보여줍니다.



### Resolution-Agnostic Transformer-based Climate Downscaling (https://arxiv.org/abs/2411.14774)
- **What's New**: 이번 연구는 고해상도 대기 예측 모델을 활용한 새로운 다운스케일링 방법을 소개합니다. 특히, Earth Vision Transformer(Earth ViT) 모델이 기존의 Global Climate Models(GCMs)로부터 50km에서 25km, 그리고 3km 해상도로 다운스케일링되는 과정을 보여주며, 별도의 추가 훈련 없이 이뤄진 일반화 가능성을 강조합니다. 이 방법은 극단적인 기상 현상에 대한 더 나은 계획 수립을 가능하게 하는 기회를 제공합니다.

- **Technical Details**: Earth ViT 구조는 Pangu-Weather 모델과 유사하며, 압력 수치를 13개에서 3개로 줄이고 더 높은 해상도를 위한 출력을 약간 수정하였습니다. 모델의 물질 보존 의무를 보장하기 위해 사용자 정의 손실 함수가 구현되어, 저해상도 입력 이미지에서 계산된 질량과 초해상도 출력 이미지에서 계산된 질량을 비교합니다. 이는 물리적 일관성을 유지하는 데 큰 역할을 합니다.

- **Performance Highlights**: 실험 결과, Earth ViT는 ERA5 데이터를 기반으로 하는 2배 다운스케일링 작업에서 2000/2001년 데이터를 훈련하여 성능을 보였습니다. ResNet 모델과의 비교를 통해 Earth ViT는 다양한 기온 및 강수량 변수에서 효과적으로 작동함을 보여줍니다. 이 새로운 접근 방식은 기존의 기후 모델링 절차보다 경제적이면서도 효율적인 대안으로 여겨집니다.



### Efficient Long Video Tokenization via Coordinated-based Patch Reconstruction (https://arxiv.org/abs/2411.14762)
Comments:
          Code is available on the project webpage: this https URL

- **What's New**: 인트로덕션에서 발표된 새로운 동영상 토크나이저 CoordTok은 시계열 및 공간 정보를 효과적으로 활용할 수 있는 방법을 제시합니다. 기존의 비디오 토크나이저가 짧은 비디오 클립을 인코딩하는 데 한계가 있었던 반면, CoordTok은 긴 비디오를 직접 인코딩할 수 있도록 설계되었습니다. 이 방법은 랜덤 샘플링된 좌표를 기반으로 비디오 패치를 재구성하는 방식을 통해 훈련 비용을 획기적으로 줄일 수 있습니다.

- **Technical Details**: CoordTok은 비디오를 분할된 3D 표현으로 인코딩하고, 주어진 (x, y, t) 좌표에 해당하는 비디오 패치를 재구성합니다. 이 과정에서 비디오는 겹치지 않는 공간-시간 패치로 나누어지고, 각 패치는 학습 가능한 위치 임베딩을 사용하여 처리됩니다. 두 차원으로 나뉘는 표현을 활용함으로써, CoordTok은 메모리와 연산 자원을 효율적으로 사용합니다.

- **Performance Highlights**: 실험 결과, CoordTok은 긴 비디오 클립을 인코딩하는 데 필요한 토큰 수를 기존 방법보다 크게 줄일 수 있음을 보여주었습니다. 예를 들어, 128x128 해상도의 128프레임 비디오를 단 1280개의 토큰으로 인코딩할 수 있는 반면, 기존 방법들은 6144개 또는 8192개의 토큰이 필요합니다. 이러한 효율적인 인코딩 덕분에 대규모 Diffusion Transformer 모델을 메모리 효율적으로 훈련할 수 있습니다.



### FairAdapter: Detecting AI-generated Images with Improved Fairness (https://arxiv.org/abs/2411.14755)
- **What's New**: 이 논문은 AI가 생성한 비얼굴 이미지 감지의 공정성 문제를 다루기 위해 FairAdapter라는 새로운 프레임워크를 제안합니다. 기존 기술들이 데이터 편향 문제와 불공정성에 취약한 반면, 본 모델은 이러한 문제를 해결하여 감지 공정성을 개선할 수 있는 가능성을 보입니다. 또한, 실험적인 결과를 통해 FairAdapter의 성능 향상이 입증되었습니다.

- **Technical Details**: FairAdapter 프레임워크는 세 가지 모듈로 구성되어 있습니다: 공정성 보조 모듈, 분류 모듈, 및 손실 함수입니다. 공정성 보조 모듈은 CLIP(Contrastive Language-Image Pre-training) 이미지 인코더를 기반으로 하며, 다른 범주의 이미지에서 자연어 의미를 추출하여 분류 모듈을 지원합니다. 분류 모듈은 FairAdapter 네트워크와 ClassifyAdapter 네트워크로 기능하며, 각각이 독립적으로 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안하는 FairAdapter 모델은 AI 생성 이미지의 감지 공정성을 크게 향상시키는 것으로 나타났습니다. Ablation study를 통해 제안된 모듈의 장점도 확인되었습니다. 기존의 딥러닝 기반 이미지 감지 기술들과 비교하여 본 모델은 공정성 측면에서 우수한 성능을 발휘하고 있습니다.



### TopoSD: Topology-Enhanced Lane Segment Perception with SDMap Prior (https://arxiv.org/abs/2411.14751)
Comments:
          17 pages, 7 figures, and 7 tables

- **What's New**: 이번 연구에서는 고해상도 지도(HDMap)에 대한 의존도를 줄이는 방향으로 자율주행 시스템을 혁신적으로 개선하고 있습니다. 고비용의 주석 및 유지 관리 문제를 해결하기 위해, 연구자들은 온보드 센서를 활용한 온라인 벡터화 HDMap 구축에 중점을 두고 있습니다. 연구진은 저해상도 지도(SDMap)를 사용하여 환경을 '인식'하고, 이는 주행 장면 이해에 크게 기여할 것으로 기대됩니다.

- **Technical Details**: SDMap 요소들은 신경 공간 맵 표현(neural spatial map representation) 및 인스턴스 토큰(instance tokens)으로 인코딩되어, 비행 시점(BEV) 특성을 향상시킵니다. 제안된 경로 세그먼트 표현 프레임워크는 경로, 중심선 및 그 토폴로지를 동시에 예측합니다. 또한, 토폴로지 유도 디코더(topology-guided decoder)를 사용하여 지오메트리 예측 및 토폴로지 추론 능력을 더욱 향상시킵니다.

- **Performance Highlights**: 개선된 모델은 OpenLane-V2 데이터셋에서 실험하여 기존의 최첨단 방법들보다 현저한 성능 향상을 보였고, mAP에서 +6.7, 토폴로지 메트릭에서 +9.1의 성과를 기록했습니다. 연구 결과, SDMap 노이즈 증강을 활용한 모델이 더 높은 견고성을 나타내는 것으로 나타났습니다.



### Ordinal Multiple-instance Learning for Ulcerative Colitis Severity Estimation with Selective Aggregated Transformer (https://arxiv.org/abs/2411.14750)
Comments:
          10 pages, 9 figures, Accepted in WACV 2025

- **What's New**: 이번 논문에서는 궤양성 대장염(UC)의 환자 수준에서 심각도를 추정하는 새로운 방법을 제안합니다. 기존의 이미지 수준에서 심각도를 추정하는 방법들과는 달리, 이 방법은 여러 개의 이미지를 통합하여 환자의 실제 임상 설정을 반영하는 방식으로 심각도를 평가합니다. 이를 통해 의료진이 실제로 기록한 심각도 레이블을 활용하게 되어, 더욱 정확한 진단이 가능해집니다.

- **Technical Details**: 제안된 방법은 선택적 집계 토큰(selective aggregator tokens)을 통한 변환기(transformer) 기반의 환자 수준의 심각도 추정 방식으로 구성됩니다. 이 방법은 주어진 이미지 집합에서 가장 높은 심각도를 가진 인스턴스만을 집계할 수 있도록 설계되어, 인스턴스 특성과 환자 수준의 심각도 점수 간의 관계를 효과적으로 파악합니다. 이를 위해 k-1개의 선택적 집계 토큰을 사용하며, 각 토큰은 특정 기준을 초과하는 인스턴스 특성을 집계합니다.

- **Performance Highlights**: 실험 결과, SATOMIL 방법은 두 가지 내시경 이미지 데이터셋에서 기존의 최첨단 MIL 방법들보다 뛰어난 성능을 보였습니다. 또한, 환자 수준의 주석을 활용한 모델은 이미지 수준의 심각도 추정 방법보다 더 높은 정확도를 기록했습니다. 본 연구는 실제 임상 데이터를 활용한 환자 수준의 심각도 추정 방법이 기존 이미지 수준 방법보다 우수하다는 것을 입증했습니다.



### Point Cloud Understanding via Attention-Driven Contrastive Learning (https://arxiv.org/abs/2411.14744)
- **What's New**: 이번 논문에서는 PointACL이라는 새로운 Attention-driven Contrastive Learning 프레임워크를 제안합니다. 이 방법은 포인트 클라우드(point cloud) 이해에서 발생하는 기존 모델의 한계를 보완하기 위해 설계되었습니다. 특히, PointACL은 저조도 주의(low-attention) 영역에 초점을 맞추고, 더 나아가 글로벌 구조에 대한 이해도를 향상 시킵니다.

- **Technical Details**: PointACL의 핵심 구성 요소는 Attention-driven Dynamic Masking 전략입니다. 이는 모델이 일부 주요 패치에 지나치게 의존하지 않도록 하여 저조도 영역에 더 많은 주의를 기울이게 합니다. 또한, 원래의 pre-training 손실과 대조 학습 손실을 결합하여 모델의 특성 구별(feature discrimination)과 일반화를 향상시킵니다.

- **Performance Highlights**: PointACL은 다양한 3D 이해 과제에서 최고 수준(state-of-the-art)의 성능을 달성하였습니다. 예를 들어, ScanObjectNN에서 89.9%의 정확도를 기록하였고, ModelNet40에서는 94.1%에 달합니다. 여기에 더해, PointACL은 노이즈 환경에서도 뛰어난 강인성을 보여 주목받고 있습니다.



### FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification (https://arxiv.org/abs/2411.14743)
Comments:
          15 pages, 3 figures

- **What's New**: 이 논문은 암 진단을 위한 Few-shot learning의 필요성과 이를 해결하기 위한 혁신적인 프레임워크인 FOCUS를 제안합니다. FOCUS는 pathology foundation models (FMs)와 언어 사전 지식을 결합하여 진단적으로 중요한 영역에 집중적으로 접근할 수 있게 해줍니다. 이를 통해 기존의 기술적 한계를 극복하고, 데이터가 부족한 환경에서도 효과적인 암 진단을 가능케 합니다. FOCUS는 다단계 압축 전략을 통해 전반적인 모델의 성능을 향상시킵니다.

- **Technical Details**: FOCUS는 (1) 전역 시각적 중복 제거 모듈을 통해 자극적(novel) feature를 선택하고, (2) 시맨틱 관련성을 기준으로 visual tokens를 우선시하는 지식 강화 압축 메커니즘을 도입합니다. 마지막으로,(3) 쌍별 순차 토큰 압축 모듈을 통해 코사인 유사성 임계값을 유지하면서, 시공간 연속성을 보존합니다. 이러한 접근 방식은 데이터가 제한적인 상황에서도 모델의 주의력을 중요한 진단적 특성에 집중시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: FOCUS는 다양한 암 데이터셋에서 광범위한 평가를 통해 기존의 최신 기술(SOTA)과 동등하거나 우수한 성능을 보였습니다. 특히 작은 샘플로도 효과적으로 진단적 특징을 포착하는 데 있어 그 효과성을 입증했습니다. 추가적인 ablation studies를 통해 제안된 아키텍처의 각 구성 요소의 기여도를 체계적으로 분석하였으며, three-stage 압축 전략, pathology FMs의 선택과 언어 프롬프트의 디자인이 효과적임을 보여 주었습니다.



### TEXGen: a Generative Diffusion Model for Mesh Textures (https://arxiv.org/abs/2411.14740)
Comments:
          Accepted to SIGGRAPH Asia Journal Article (TOG 2024)

- **What's New**: 이번 연구에서는 기존의 2D diffusion 모델을 활용한 테스트 시 최적화 접근법에서 벗어나, UV 텍스처 공간을 직접 학습하는 방법에 초점을 맞췄습니다. 새로운 대규모 diffusion 모델인 TEXGen을 소개하며, 이 모델은 텍스처 맵을 직접 생성할 수 있는 기능을 통해 고해상도 텍스처 생성에 기여합니다. 특히, 모델은 텍스트 프롬프트와 단일 시점 이미지를 통해 유도된 UV 텍스처 맵을 생성할 수 있습니다.

- **Technical Details**: 우리는 UV 텍스처 맵을 생성하는 데 필요한 효율적인 네트워크 아키텍처를 제안합니다. 이 아키텍처는 UV 맵에 대한 컨볼루션을 점 구름(point clouds)의 attention 레이어와 교차하여 처리합니다. 이를 통해 3D 표면에서 효과적인 특징 상호작용을 달성하며, 매니지 가능한 계산으로 3D 연속성을 유지합니다. 700M 매개변수를 가진 이 모델은 고해상도 텍스처를 생성할 수 있는 능력을 지니고 있습니다.

- **Performance Highlights**: TEXGen은 텍스처 생성, 인페인팅(inpainting), 스파스 뷰(sparse view)에서의 텍스처 완성 등의 다양한 확장 응용 프로그램을 자연스럽게 지원합니다. 이 연구 결과는 기존 방식보다 더 높은 품질을 달성하며, 학습 없이 다양한 작업에 사용할 수 있는 기초 모델로 기능합니다. 상태-of-the-art 결과를 달성하여 3D 텍스처의 품질을 향상시키고, 더 광범위한 객체에 대한 일반화 능력을 보여줍니다.



### AI Tailoring: Evaluating Influence of Image Features on Fashion Product Popularity (https://arxiv.org/abs/2411.14737)
- **What's New**: 이 논문에서는 패션 산업에서 소비자 선호에 영향을 미치는 핵심 제품 특징을 식별하는 새로운 방법론을 제시합니다. "influence score"라는 메트릭을 도입하여 제품 특징의 중요성을 정량적으로 평가하고, Transformer 기반 모델과 Random Forest를 통합한 Fashion Demand Predictor (FDP)라는 예측 모델을 개발하였습니다. 또한, 이미지 수정 모델을 적용하여 특정 특징이 제품의 인기도에 미치는 영향을 분석하는 여러 실험을 수행합니다.

- **Technical Details**: 본 연구는 유럽의 패스트 패션 회사로부터 얻은 실제 제품 데이터를 사용하여 진행되었습니다. 데이터셋에는 8,503개의 제품 이미지와 설명이 포함되어 있으며, 이미지 편집을 통한 실험에서는 InstructPix2pix-Distill 및 Adobe Firefly Image 3와 같은 확산 모델을 활용하여 AI 수정이미지를 생성하였습니다. 캡션 구성을 통해 각 제품의 디자인 요소를 명확히 하고, 이를 기반으로 Synonym 그룹을 통해 중복 데이터의 재정리를 수행합니다.

- **Performance Highlights**: FDP 모델은 인간의 선호 데이터를 기반으로 평가되었으며, 특정 디자인 특징을 강화한 제품의 인기도는 예측된 인기도에 대해 유의미한 개선을 보였습니다. 또한, 이 모델은 다양한 실험과 조사 결과를 통해 그 유효성을 입증하였으며, 패션 이미지 분석 및 마케팅 전략 개발에 있어 귀중한 통찰력을 제공합니다.



### Evaluating and Advancing Multimodal Large Language Models in Ability Lens (https://arxiv.org/abs/2411.14725)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 시각 인식(perception) 능력 평가를 위한 새로운 벤치마크인 AbilityLens를 소개합니다. 이전의 평가 기준들이 여러 다양한 질문 유형, 도메인 및 평가 지표로 인해 평가에 변동성이 있었으나, AbilityLens는 여섯 가지 주요 인식 능력을 아우르는 통합된 평가 체계를 제공합니다. 또한, 모델의 정확성과 안정성을 동시에 평가하는 시스템을 새롭게 도입해 모델 발전에 실질적인 방향성을 제시합니다.

- **Technical Details**: AbilityLens는 11개의 기존 벤치마크에서 데이터를 수집하여 각 인식 능력에 대해 1,000개 이상의 테스트 샘플을 포함하여 총 12,000개의 샘플을 구성합니다. 이 평가 방안은 정확성(accuracy)과 안정성(stability)을 측정하기 위해 개별 하위 지표(sub-metric)를 통합하여 단일 결과 점수(composite score)를 산출합니다. 또한, 온라인 및 오프라인 두 가지 평가 모드를 통해 모델 학습 동향을 모니터링하고, 각 인식 능력에 대해 모델을 비교합니다.

- **Performance Highlights**: AbilityLens를 사용하여 현재 MLLMs의 성능을 분석한 대표 사례로, LLaVA-OV-72b와 Qwen2-VL-72b 모델이 상업적 모델에 비해 정확성에서 우수함을 확인하였지만, 안정성은 여전히 부족함을 나타냅니다. 또한, 각 인식 능력 사이의 상이한 개선 곡선과 초기 수렴(early convergence) 품질 혼란(ability conflict) 현상이 관찰되었습니다. 능력 충돌을 완화하기 위해 초기 훈련 단계의 최적 능력 체크포인트를 통합한 모델 병합 방법을 제안하여 성능 저하를 최소화하는 효과를 확인하였습니다.



### Effective SAM Combination for Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2411.14723)
- **What's New**: 본 논문에서는 ESC-Net이라는 새로운 일단계(open-vocabulary) 의미 분할 모델을 제안합니다. ESC-Net은 이미지를 비가시적 클래스로 세분화하기 위해 Segmentation Anything Model(SAM)의 디코더 블록을 활용합니다. 이 접근법은 이미지-텍스트 상관관계를 통해 생성된 의사 프롬프트(pseudo prompts)를 SAM의 프롬프트 처리 구조에 통합하여 정확한 마스크 예측을 가능하게 합니다.

- **Technical Details**: ESC-Net은 CLIP의 이미지 및 텍스트 특징 간의 상관관계를 바탕으로 의사 좌표 포인트와 객체 마스크를 생성합니다. 이후 이 정보를 SAM의 프롬프트 인코더를 통해 임베딩하고, 생성된 프롬프트와 CLIP 인코더 피처를 SAM 변압기 블록에 입력하여 이미지와 텍스트 간의 상호작용을 촉진하는 구조를 가지고 있습니다. 이 모델은 SAM의 지역적 공간 집합기를 활용하여 이미지-언어 관계를 효과적으로 모델링하는 데 기여합니다.

- **Performance Highlights**: ESC-Net은 ADE20K, PASCAL-VOC 및 PASCAL-Context와 같은 표준 벤치마크에서 기존 방법들과 비교하여 뛰어난 성능을 나타냅니다. 특히, 같은 계산 비용에도 불구하고 이전의 상관 기반 일단계 방법인 CAT-Seg보다 더 나은 성능을 보였습니다. 다양한 ablation 연구를 통해도 도전적인 시나리오에서 견고한 성능을 유지함을 보여주었습니다.



### VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving (https://arxiv.org/abs/2411.14716)
- **What's New**: 이번 논문은 VisionPAD라는 자가 감독(pre-training) 방식의 새로운 패러다임을 도입하여 자율주행의 비전 중심 알고리즘을 위한 보다 효율적인 3D 표현을 제공합니다. 기존의 신경 렌더링(neural rendering) 방식과 달리, VisionPAD는 이미지 한 가지만을 사용하여 멀티뷰 표현을 재구성하기 위해 3D Gaussian Splatting(3D-GS)을 활용합니다. 특히, 이 방법은 현재 프레임의 복셀 속도(velocity)를 추정하는 새로운 자가 감독 기법을 제안합니다.

- **Technical Details**: VisionPAD는 네 가지 핵심 모듈로 구성되어 있습니다. 첫째, 과거의 멀티뷰 이미지를 처리하기 위해 비전 중심 인식 네트워크(backbone)를 이용합니다. 둘째, 3D Gaussian Splatting Decoder가 복셀 표현을 기반으로 현재 프레임의 멀티뷰 이미지를 재구성합니다. 셋째, 복셀 속도 추정(strategy)을 통해 현재 프레임의 속성들을 인접 프레임에 맞춰 전이하여 멀티뷰 이미지와 깊이 맵(depth map)을 만들도록 합니다.

- **Performance Highlights**: 실험 결과, VisionPAD는 자율주행 데이터셋에서 3D 객체 탐지, 점유 예측, 맵 세그멘테이션의 세 가지 하위 작업에서 성능이 크게 향상됨을 보였습니다. 특히, 단순히 멀티프레임 이미지만을 사용하여 사전 훈련(pre-training)한 결과 3D 객체 탐지에서 연간 2.5 mAP, 의미론적 점유 예측에서 4.5 mIoU, 맵 세그멘테이션에서 4.1 IoU의 성능 개선을 달성하였습니다. 이는 기존의 최첨단(pre-training) 접근법에 비해 매우 높은 성과입니다.



### Any-to-3D Generation via Hybrid Diffusion Supervision (https://arxiv.org/abs/2411.14715)
- **What's New**: 이번 연구는 XBind라는 새로운 통합 프레임워크를 제안하며, 이는 다양한 모달리티(모달리티)에서 3D 객체 생성을 가능하게 한다. XBind는 멀티모달 정렬 인코더(multi-modal aligned encoder)와 사전 훈련된 확산 모델을 결합하여 텍스트, 이미지, 오디오 등 다양한 입력 소스로부터 3D 객체를 생성할 수 있도록 한다. 특히, Modality Similarity (MS) 손실 함수를 도입하여 서로 다른 모달리티에 대한 임베딩을 정렬함으로써 품질 개선을 도모하고 있다.

- **Technical Details**: XBind 프레임워크는 멀티모달 정렬 인코더를 사용하여 여러 모달리티의 임베딩을 통합된 공유 공간에 인코딩한다. 이를 통해 사전 훈련된 2D 확산 모델을 위한 프롬프트로 활용되며, Distillation Sampling Loss를 통해 3D 표현 최적화를 유도한다. 또한, 하이브리드 확산 감독(Hybrid Diffusion Supervision)과 삼단계 최적화(Three-Phase Optimization) 프로세스를 결합하여 생성된 3D 객체의 품질을 향상시킨다.

- **Performance Highlights**: XBind의 실험 결과는 이 프레임워크가 주어진 모달리티 프롬프트에 잘 정렬된 고품질의 3D 객체를 생성할 수 있음을 보여준다. XBind는 다양한 모달리티에서 직접적으로 3D 객체를 생성할 수 있도록 하여 시간과 자원 소모를 크게 줄인다. 이로 인해 정보 손실이 최소화되며, 3D 생성의 첫 번째 통합 프레임워크로서 두 가지 혁신적인 기능을 통해显しい 성능 향상을 달성하였다.



### Cross-Modal Pre-Aligned Method with Global and Local Information for Remote-Sensing Image and Text Retrieva (https://arxiv.org/abs/2411.14704)
- **What's New**: 이번 연구는 원거리 탐지 교차 모드 텍스트-이미지 검색 (RSCTIR)에서 글로벌 및 로컬 정보를 효과적으로 통합하기 위한 CMPAGL 방법을 제안합니다. 특히, Gswin transformer block을 활용하여 다중 스케일 특성을 포착하고, 준비 정렬(pre-alignment) 메커니즘을 통해 모드 융합(training) 과정을 간소화합니다. 새로운 유사도 행렬 재가중화(SMR) 알고리즘과 최적화된 삼중 손실 최적화 방법을 통해 특징 학습을 최적화하고 검색 성능을 향상시킵니다.

- **Technical Details**: CMPAGL 방법은 이미지 인코더, 텍스트 인코더 및 다중 모드 인코더를 포함하는 인프라를 기반으로 합니다. Gswin transformer block은 로컬 창(self-attention)과 글로벌-로컬 창 크로스 어텐션(cross-attention)을 결합하여 특징 추출을 최적화하며, 이는 복잡한 세부 정보와 다중 스케일 정보를 효과적으로 캡처합니다. 또한, SMR 알고리즘은 원래 유사도 행렬을 활용하여 재정렬을 개선하고, 최적화된 삼중 손실은 매칭되는 이미지와 텍스트 간의 거리를 최소화합니다.

- **Performance Highlights**: CMPAGL 방법은 RSICD 및 RSITMD 등 다양한 데이터셋에서 기존 최첨단 방법보다 R@1에서 최대 4.65% 향상, 평균 Recall(mR)에서 2.28% 향상을 달성하는 등의 실험 결과를 통해 효과성을 입증했습니다. 이러한 성과는 제안된 아키텍처의 유효성을 뒷받침하며, 먼 거리 탐지 이미지-텍스트 검색 분야의 발전에 기여할 것입니다.



### Anti-Forgetting Adaptation for Unsupervised Person Re-identification (https://arxiv.org/abs/2411.14695)
Comments:
          Accepted to TPAMI

- **What's New**: 본 논문에서는 Dual-level Joint Adaptation and Anti-forgetting (DJAA) 프레임워크를 제안합니다. 이 프레임워크는 이전 도메인의 지식과 각 적응된 목표 도메인을 잊지 않고 새로운 도메인에 점진적으로 적응할 수 있도록 해줍니다. 기존의 방법들과 달리, 우리의 접근 방식은 변하지 않는 과거 지식의 일관성을 유지하면서도 모델의 일반화 능력을 향상시키는 데 중점을 둡니다.

- **Technical Details**: DJAA 프레임워크는 적응 모듈과 리허설 모듈로 구성됩니다. 적응 모듈에서는 클러스터링 알고리즘을 통해 생성된 가상의 레이블을 사용하여 pseudo labels에 기반한 이미지-프로토타입 대조 손실(image-to-prototype contrastive loss)을 적용합니다. 리허설 모듈에서는 과거 도메인의 샘플을 저장하고, 이를 통해 이미지 간 유사성(image-to-image similarity)과 이미지-프로토타입 유사성(image-to-prototype similarity)을 정규화하여 잊어버림을 예방합니다.

- **Performance Highlights**: 실험 결과, 제안된 DJAA 방법이 기존의 비감독형(person ReID 모델의 유의미한 성능 개선 및 강화 학습 기법에 비해 크게 향상되었습니다. 특히, DJAA는 여러 주요 데이터셋에서 비잊는(non-forgetting) 특성과 일반화(generalization), 그리고 역호환(backward-compatible) 능력이 뛰어난 것으로 나타났습니다.



### Whats in a Video: Factorized Autoregressive Decoding for Online Dense Video Captioning (https://arxiv.org/abs/2411.14688)
- **What's New**: 본 논문에서는 비디오의 내용을 자동으로 밀도 있게 설명하는 캡션을 생성하는 새로운 접근 방식을 제안합니다. 기존의 모델은 비디오 전체를 한 번에 처리해야 했지만, 우리는 미래 프레임에 접근할 필요 없이 빈번하고 구체적인 캡션을 출력하는 효율적인 온라인 접근 방식을 개발했습니다. 이 모델은 시간 세그먼트마다 시각적 특징의 시퀀스를 모델링하는 새로운 autoregressive factorized decoding 아키텍처를 사용합니다.

- **Technical Details**: 제안된 모델은 비디오의 현재 정보와 이전 컨텍스트를 연결하는 메모리와 같은 메커니즘을 활용하여 밀집한 캡션을 생성합니다. 각 비디오 세그먼트의 정보를 처리하고 출력하는 과정에서 단일 디코더가 여러 세그먼트를 처리하도록 설계되어 계산 비용이 비디오 길이에 비례하여 선형적으로 증가합니다. 이 접근 방식은 더 긴 비디오에서 효율적으로 작동하며, 더 많은 캡션과 프레임을 처리하는 데 필요한 표준 하드웨어 메모리 용량을 초과하지 않도록 설계되었습니다.

- **Performance Highlights**: 우리 모델은 ViTT, YouCook2, ActivityNet과 같은 잘 알려진 밀집 비디오 캡셔닝 벤치마크에서 평가되었습니다. 실험 결과, 모두에서 최신 기법들을 초과하는 성능을 보였으며, 일부 경우에는 개선 폭이 매우 큰 것으로 나타났습니다. 또한 이 모델은 같은 성능을 유지하면서 20% 더 적은 연산 비용을 사용합니다.



### Differentially Private Adaptation of Diffusion Models via Noisy Aggregated Embeddings (https://arxiv.org/abs/2411.14639)
- **What's New**: 이 논문에서는 차별적 개인 정보 보호(Differential Privacy, DP) 제약 하에 확산 모델을 적응시키는 새로운 방법을 소개합니다. 기존의 DP-SGD 같은 개인 정보 보호 기술은 계산 부담이 크고 대규모 복잡한 모델에 적용할 때 성능 저하가 발생하지만, 본 연구에서는 Embedding 기반 기술인 Universal Guidance와 Textual Inversion(TI)을 활용하여 이러한 문제를 해결합니다. 이를 통해 Stable Diffusion을 스타일 적응에 적용하여 개인 정보 보호를 유지하면서도 높은 일본곡을 실현할 수 있음을 보여줍니다.

- **Technical Details**: Diffusion 모델은 반복적인 노이즈 제거 과정을 통해 텍스트 입력에 기반한 고품질 이미지를 생성합니다. 본 연구에서는 Universal Guidance와 TI를 활용하여 차별적 개인 정보 보호를 보장하면서 작은 데이터셋에 대한 적응을 가능하게 합니다. 특히, TI를 사용할 때, 각 타겟 이미지에 대해 별도의 Embedding을 학습하고 이를 노이즈 중심으로 집계하여 개인 정보 보호를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, TI 기반 적응 방법이 높은 스타일 이전 충실도를 달성하며, 강한 개인 정보 보호 보장 하에서도 효과적으로 기능합니다. 또한, 서브샘플링 기법을 통해 개별 데이터 포인트에 대한 민감도를 조절하여 이미지 품질을 개선하고, 개인 정보 보호 탄력성을 강화할 수 있음을 보여줍니다. 이로써, 본 연구는 개인 정보 보호를 유지하면서도 혁신적인 이미지 생성을 위한 실용적이고 효과적인 방법론을 제시합니다.



### HotSpot: Screened Poisson Equation for Signed Distance Function Optimization (https://arxiv.org/abs/2411.14628)
- **What's New**: 본 논문에서 제안하는 HotSpot 방법은 스크린된 Poisson 방정식과 거리 함수 간의 관계를 기반으로 하여 신경 서명 거리 함수(neural signed distance function)를 최적화하는 새로운 접근법입니다. 기존의 이코널 손실(eikonal loss)은 회복된 암시적 함수가 거리 함수로서 작동할 것을 보장하지 못하며, 최적화 과정에서 불안정성을 보입니다. HotSpot은 이러한 문제를 해결하기 위해 손실 함수를 설계하며, 이를 통해 진정한 거리 함수로 수렴할 수 있습니다.

- **Technical Details**: HotSpot은 암시적 함수가 서명 거리 함수를 출력하도록 보장하는 것을 목표로 하며, 스크린된 Poisson 방정식을 활용합니다. 기존의 이코널 정규화 손실은 불안정해 최적화가 최적이 아닌 결과에 수렴할 수 있으며, 표면 면적을 제재할 필요가 있어 과도한 부드럽게 처리됨(over-smoothing) 현상이 발생할 수 있습니다. 이러한 문제를 해결하기 위해 HotSpot은 안정적이고 자연스럽게 표면 면적을 제재하며, 복잡한 형태에서 서명 거리 함수에 접근하는 데 뛰어난 성능을 보입니다.

- **Performance Highlights**: Heat transfer와 거리의 관계를 활용하여 2D와 3D 데이터셋에서 뛰어난 표면 재구성과 정확한 거리 근사를 실현합니다. 본 연구의 실험결과는 HotSpot이 기존 방법들보다 우수한 성능을 제공함을 보여주며, 특히 고유의 복잡한 형태를 가진 데이터에 효과적임을 입증하였습니다. HotSpot은 암시적 함수 최적화에서 수치적으로 안정적이며, 공간적 및 시간적 안정성을 갖춘 방법으로 자리잡을 것입니다.



### Solving Zero-Shot 3D Visual Grounding as Constraint Satisfaction Problems (https://arxiv.org/abs/2411.14594)
- **What's New**: 본 연구에서는 3D 시각 기초화(3DVG) 작업을 제약 만족 문제(Constraint Satisfaction Problem, CSP)로 재구성하여 기존의 제로샷(zero-shot) 방식의 한계를 해결하는 새로운 방법이 제안되었습니다. 이를 통해 공간적 관계를 전역적으로 해석하고 =>(global reasoning) 타겟과 앵커 객체의 기초화 결과를 동시에 도출할 수 있게 되었습니다. 또한, 우리의 시스템,Constraint Satisfaction Visual Grounding (CSVG)는 부정(negation) 및 카운팅(counting) 기반 쿼리 처리에 유연성을 보여주며, 소스 코드도 공개되었습니다.

- **Technical Details**: CSVG 시스템은 대형 언어 모델(Large Language Model, LLM)을 활용하여 3DVG 작업을 위한 프로그램을 생성하고, 이를 통해 전역적으로 유효한 해법을 탐색합니다. 기존 방법들의 지역적인 추론(local reasoning) 한계를 극복하였으며, 쿼리의 자연어 표현을 처리하기 위해 포인트 클라우드를 텍스트 형식으로 변환하는 과정을 포함합니다. 이를 통해 CSVG는 객체간의 관계를 효과적으로 해석할 수 있습니다.

- **Performance Highlights**: CSVG는 공용 데이터셋인 ScanRefer와 Nr3D에서 폭넓은 평가를 받았으며, 현재 주류 제로샷 3DVG 방법보다 각각 +7.0% (Acc@0.5 점수) 및 +11.2%의 개선된 기초화 정확도를 달성했습니다. 이러한 성과는 CSVG의 효과성을 입증하며, 복잡한 3D 장면에서 객체를 정확히 기초화할 수 있는 새로운 가능성을 제시합니다.



### Privacy-Preserving Video Anomaly Detection: A Survey (https://arxiv.org/abs/2411.14565)
Comments:
          19 pages, 6 figures

- **What's New**: 이 논문은 Privacy-Preserving Video Anomaly Detection (P2VAD)의 발전을 체계적으로 검토하며, 이를 위한 최초의 분류 체계를 제공합니다. 기존의 탐색은 RGB 비디오 시퀀스에 초점을 맞추는 경향이 있었지만, 본 연구는 개인 정보 유출 및 외관 편향 문제를 간과한 점을 지적합니다. 또한 데이터 수집, 모델 학습 및 시스템 적용의 세 가지 주요 단계에 대한 P2VAD 프레임워크를 제시하고, 다양한 방법론의 기본 가정과 강점, 약점을 분석합니다.

- **Technical Details**: P2VAD는 데이터를 수집할 때 개인 식별 정보가 포함되지 않도록 하기 위해 Non-Identifiable Elements (NIE) 방법, 중간 매개체에 대한 정보를 비활성화하는 Desensitized Intermediate Modalities (DIM) 방법, 그리고 Edge-Cloud Intelligence (ECI) 기술을 통해 데이터 보안 문제를 해결합니다. NIE 방법은 객체 탐지 모델을 사용하여 RGB 시퀀스에서 인간 관련 지역을 마스킹하는 방식으로 개인 정보를 보호합니다. DIM 방법은 모션 관련 이상 탐지를 위한 중간 과정을 활용함으로써 외관과 관련된 개인 정보 우려를 제거합니다.

- **Performance Highlights**: P2VAD는 데이터 수집 없이 정상 이벤트를 쉽게 수집하여 모델을 훈련할 수 있어 비용이 적게 들고 다양한 현장 적응력이 뛰어납니다. 최근 Weakly-supervised VAD 접근법은 비디오 레벨 레이블을 통해 프레임 레벨의 이상 점수를 정량화 가능하게 하여 비연속적인 데이터에서 유용성을 높였습니다. 이러한 점에서 P2VAD는 상업적 및 공공 안전을 위한 심도 있는 연구 필요성이 커지고 있으며, 향후 기술 발전과 함께 더욱 확대될 가능성이 있습니다.



### Enhancing GeoAI and location encoding with spatial point pattern statistics: A Case Study of Terrain Feature Classification (https://arxiv.org/abs/2411.14560)
Comments:
          4 pages with 1 figure. Accepted in 7th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery

- **What's New**: 이번 연구에서는 지형 특성(classification) 분류를 위한 새로운 접근 방식을 소개합니다. 공간 점 패턴 통계(spatial point pattern statistics)를 딥러닝 모델에 통합하여 GeoAI(Geo Artificial Intelligence) 의사결정 능력을 향상시키고자 합니다. 특히, 첫 번째 및 두 번째 차수(point patterns) 효과를 지식 기반으로 통합하여 GeoAI 모델을 개선하는 방법을 탐구합니다.

- **Technical Details**: 이 연구는 위치 인코딩(location encoding) 개념에서 영감을 받아, 공간적 맥락이 지형 특성 예측의 정확성에 미치는 영향을 분석합니다. 첫 번째 및 두 번째 차수의 패턴 효과를 고려하여, 모델에 공간 점 패턴 통계를 효과적으로 통합하고 있습니다. 이러한 접근은 기존의 GeoAI 프레임워크에서 지형 특성의 예측 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 공간 점 패턴 통계를 포함함으로써 모델 성능이 현저히 향상되는 것을 보여줍니다. 다양한 공간 관계의 표현을 활용하여, 예측의 정확성을 증가시킬 수 있음을 입증하였습니다. 이 연구는 GeoAI 모델의 실효성을 높이는 중요한 단계를 제공합니다.



### GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and A Comprehensive Multimodal Dataset Towards General Medical AI (https://arxiv.org/abs/2411.14522)
- **What's New**: 본 논문에서는 GMAI-VL-5.5M이라는 포괄적인 다중 모달 의료 데이터셋을 소개하고, 이를 기반으로 GMAI-VL이라는 일반 의료 시각-언어 모델을 제안합니다. GMAI-VL-5.5M은 수백 개의 전문 의료 데이터셋을 이미지-텍스트 쌍으로 정교하게 변환하여 생성된 데이터셋으로, 다양한 의료 작업을 포괄하며 고품질의 데이터로 구성되어 있습니다. 이 모델은 시각적 및 언어적 정보를 통합하여 다중 모달 데이터 처리 능력을 개선하고, 정밀한 진단 및 임상 결정 지원을 목표로 합니다.

- **Technical Details**: GMAI-VL 모델의 훈련 전략은 세 단계로 나뉘며, 첫 두 단계에서는 시각적(의료 이미지) 및 언어적(의료 텍스트) 요소 간의 기본적인 특징에서 고급 의미적 요소로 연결을 구축합니다. 마지막 단계에서는 크로스 모달 지침으로 모델을 미세 조정하여 시각-언어 간의 상호작용 이해를 향상시킵니다. 이 훈련 전략을 통해 GMAI-VL은 시각적 질문 응답 및 의료 이미지 진단과 같은 다양한 의료 작업에서 높은 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과 GMAI-VL 모델은 PMC-VQA 및 VQA-RAD와 같은 다중 모달 질문-응답 작업에서 기존 모델들을 능가하며, OmniMedVQA, GMAI-MMBench, MMMU의 Health & Medicine 트랙에서 새로운 벤치마크를 수립합니다. 구체적으로 GMAI-VL은 OmniMedVQA에서 평균 88.48%, GMAI-MMBench 테스트 세트에서 62.43%, MMMU의 Health & Medicine 트랙에서 51.3%의 성적을 기록하였습니다. 이는 다중 모달 의료 모델의 발전을 위한 탄탄한 기초를 제공합니다.



### MyTimeMachine: Personalized Facial Age Transformation (https://arxiv.org/abs/2411.14521)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 MyTimeMachine (MyTM)라는 새로운 방법론을 제안하여, 글로벌 노화(prior)와 개인 사진 컬렉션을 결합하여 개인화된 얼굴 노화 변환을 학습합니다. 이 방법은 최소 50장의 이미지를 사용해 개인화된 노화 특성을 모델링하며, StyleGAN2를 통해 재노화된 이미지를 생성합니다. 또한, 개인화된 노화 손실 함수와 규제 기능을 도입하여 다양한 개인적 요인을 반영하는 고품질 노화 효과를 구현했습니다.

- **Technical Details**: MyTimeMachine은 Adapter Network라는 혁신적인 네트워크를 도입하여 글로벌 노화 기능을 개인화된 노화 특성으로 업데이트합니다. 세 가지 손실 함수, 즉 개인화된 노화 손실, 외삽 규제, 적응형 w-노름 규제를 활용하여 학습합니다. 이러한 구조는 입력 이미지의 개인적 특성을 명확하게 보존하면서도 노화의 형태와 질감 변화를 반영하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 이미지 및 비디오 모두에서 고품질의 아이덴티티 보존 노화 효과를 생성하며, 목표 연령에서 개인의 실제 모습을 정확히 반영합니다. 실험적으로, 70세를 젊게 또는 40세를 늙게 render하는 두 가지 프레임워크를 통해 기존 기술보다 우수한 성능을 보였습니다. MyTM을 활용하면 노화 효과가 시간적으로 일관되고 정체성을 유지하는 재 노화가 가능해져 많은 VFX 애플리케이션에서 활용될 수 있습니다.



### The Double-Ellipsoid Geometry of CLIP (https://arxiv.org/abs/2411.14517)
- **What's New**: 본 논문에서는 Contrastive Language-Image Pre-Training (CLIP)의 임베딩 구조를 분석하여 텍스트와 이미지가 원점을 중심으로 하지 않고, 선형 분리 가능한 타원체 쉘에 존재함을 밝혀냈습니다. 또한 이러한 구조가 불확실성에 따른 인스턴스 임베딩을 개선하는 데 어떻게 기여하는지를 설명하였습니다. 특히 흔하게 나타나는 개념이 있는 경우, 더 많은 false negatives가 발생하며, 이는 불확실성을 증가시킵니다. 새로운 개념 적합성 (conformity)의 개념을 도입하여 평균 코사인 유사도를 측정하는 방법을 제안합니다.

- **Technical Details**: CLIP의 원시 임베딩을 분석한 결과, 각 모달리티가 원점에 중심을 두지 않는 분리 가능한 타원체 쉘에 존재하는 것으로 나타났습니다. 이러한 구조는 contrastive learning에서 응답의 선명도를 조절할 수 있게 해 주며, false negative 사례가 이 임베딩에서 가장 큰 이점을 얻는 것으로 보입니다. 논문에서 제안한 개념 적합성은 개념의 인기도를 정량화하며, 이는 평균 벡터와의 유사성과 강한 상관관계를 보여줍니다. 이러한 이해를 바탕으로 새로운 이미지 보간법인 vertical SLERP (vSLERP)를 개발합니다.

- **Performance Highlights**: 임상 실험 결과, CLIP의 타원체 오프셋 근처에서 contrastive loss가 최적화됨을 확인하였습니다. MS-COCO 데이터셋을 통해 이미지와 텍스트의 적합성 분포가 다르며, 현재 CLIP의 타원체 배치에서 잘 일치함을 발견하였습니다. 최근의 실험 결과는 새로운 이미지 생성 기법이 효과적으로 작동함을 시사하며, 추후 연구에 대한 방향성을 제시합니다. 본 연구의 결과는 multi-modal learning의 기초를 심화시키고, 응용 프로그램의 성능 향상을 위한 기초 자료로 활용될 수 있습니다.



### NexusSplats: Efficient 3D Gaussian Splatting in the Wild (https://arxiv.org/abs/2411.14514)
Comments:
          submitted to CVPR 2025

- **What's New**: 본 논문은 NexusSplats라는 새로운 접근 방식을 제안하여 복잡한 조명 및 장애물 조건에서 3D 장면 재구성을 보다 효율적이고 정밀하게 수행합니다. 기존의 3D Gaussian Splatting (3DGS)의 한계를 극복하기 위해, NexusSplats는 nexus kernel을 활용하여 조명 조건을 분리하고, 이로 인해 세밀한 텍스처를 보장하면서 재구성 속도를 가속화합니다.

- **Technical Details**: NexusSplats 방법은 nuxes kernels을 이용해 3D 장면을 voxel 세그먼트로 나누어 3D Gaussian primitives를 계층적으로 관리합니다. 각 kernel은 자신이 관리하는 Gaussian primitive 세트에 대해 색상 맵핑을 조정할 appearance embedding을 최적화합니다. 이를 통해 서로 다른 조명 조건에서도 지역적 색상 일관성을 유지할 수 있습니다. 또한, Gaussian-wise uncertainty 메커니즘을 사용하여 3D 구조와 2D 이미지 특징을 정렬하여 세밀한 장애물 처리를 지원합니다.

- **Performance Highlights**: NexusSplats는 실험 결과 현재의 최고 품질 방법보다 재구성 시간을 최대 70.4% 단축하는 동시에 우수한 렌더링 품질을 달성했습니다. 이 방식은 다양한 장면에서 뛰어난 성능을 발휘하며, 특히 조명 변화를 효과적으로 처리할 수 있는 능력을 보여줍니다.



### LLaVA-MR: Large Language-and-Vision Assistant for Video Moment Retrieva (https://arxiv.org/abs/2411.14505)
- **What's New**: 최근 발표된 논문에서는 대형 다중 모드 언어 모델(MLLMs)을 활용하여 비디오에서 정확한 순간 검색과 맥락적 기반을 강화하는 LLaVA-MR(Large Language-and-Vision Assistant for Moment Retrieval)을 제안합니다. 이는 Dense Frame and Time Encoding (DFTE)과 같은 방법을 적용하여 시공간적 특징을 추출하고, Informative Frame Selection (IFS)을 통해 짧은 시각적 및 동작 패턴을 포착합니다. 또한, Dynamic Token Compression (DTC)을 활용하여 모델의 문맥 크기 제한을 효과적으로 관리합니다.

- **Technical Details**: LLaVA-MR 모델 구조는 모든 샘플링된 프레임에서 시각적 특징을 추출하는 고정된 이미지 인코더로 시작합니다. 이후 Dense Frame and Time Encoding (DFTE)을 통해 세밀한 시공간 특징을 추출하고, Informative Frame Selection (IFS)을 통해 긴 비디오 시퀀스에서 중요한 순간을 포착합니다. 마지막으로 Dynamic Token Compression (DTC)을 통해 정보를 유지하면서 시퀀스 길이를 줄여 모델의 효율성을 극대화합니다.

- **Performance Highlights**: LLaVA-MR은 Charades-STA와 QVHighlights와 같은 벤치마크에서 11개의 최첨단 방법들을 능가하여 R1@0.5에서 1.82%, mAP@0.5에서 1.29%의 성능 개선을 이루었습니다. 이러한 결과는 긴 비디오 내에서 복잡한 동적인 순간들을 효과적으로 캡처할 수 있는 능력을 보여줍니다. 또한, 이 모델은 개방된 소스 코드로 제공될 예정입니다.



### Night-to-Day Translation via Illumination Degradation Disentanglemen (https://arxiv.org/abs/2411.14504)
Comments:
          8 pages

- **What's New**: 본 연구에서는 Night-to-Day 변환(Night2Day)을 위한 새로운 방법론 N2D3(Night-to-Day via Degradation Disentanglement)를 제안합니다. N2D3는 야간 이미지에서 다양한 감쇠(degradation) 패턴을 인식하여, 낮 이미지와 유사한 시각적 품질을 달성하는 것을 목표로 합니다. 기존 방법들이 야간 이미지의 복잡한 감쇠를 동시에 해결하는 데 부족함을 드러내면서, 본 연구는 이에 대한 해결책을 제공합니다.

- **Technical Details**: N2D3의 주요 구성 요소는 감쇠 분리 모듈(degradation disentanglement module)과 감쇠 인식 대조 학습 모듈(degradation-aware contrastive learning module)입니다. 첫째, Kubelka-Munk 이론에 기반한 물리적 전제가 활용되어, 야간 환경의 조명 감쇠를 분류합니다. 둘째, 감쇠 인식 대조 학습 전략을 통해 구조적인 일관성을 유지하면서 세밀한 이미지 변환이 이루어집니다.

- **Performance Highlights**: N2D3는 두 개의 공용 데이터셋에서 평가되었으며, 시각적 품질에서 상당한 향상을 이루었고, 하향 작업(downstream tasks)에도 긍정적인 효과를 보여줍니다. 본 연구의 결과는 야간 장면에서 복잡한 감쇠 유형을 고려하는 것이 얼마나 중요한지를 입증하며, 기존 방법들과 비교해 State-of-the-art 성능을 달성하였습니다.



### U-Motion: Learned Point Cloud Video Compression with U-Structured Motion Estimation (https://arxiv.org/abs/2411.14501)
- **What's New**: 본 논문에서는 U-Motion이라는 새로운 학습 기반의 포인트 클라우드 비디오(PCV) 압축 방식이 소개됩니다. 이 방식은 PCV의 기하학적 속성과 속성(attribute)을 동시에 효과적으로 압축할 수 있는 방법으로, 다층적 인터프레임(prediction) 예측 프레임워크인 U-Inter를 사용합니다. 이를 통해 다중 스케일(multi-scale)에서 정확한 모션 추정(motion estimation)과 보상을 수행할 수 있습니다.

- **Technical Details**: U-Motion은 U-Structured 다중 스케일 간의 모션 추정 및 보상 기법을 통합하여 현재 스케일에서 정확한 모션 추정을 가능하게 합니다. 이 과정에서는 현재 및 이전 프레임의 정보뿐만 아니라 높은 스케일과 낮은 스케일의 모션 기능이 통합됩니다. 또한, U-Inter 예측 이후 남은 스페이셜(spatial) 중복성을 포착하기 위해 계단식 공간 예측 부호화(cascaded spatial predictive coding) 모듈이 설계되었습니다.

- **Performance Highlights**: MPEG 표준 테스트 조건에 따라 실험을 수행한 결과, U-Motion은 기하학 및 속성 압축 모두에 대해 최신 학습 기반 방법인 Unicorn 및 MPEG G-PCC-GesTM v3.0보다 향상된 성능을 보여주었습니다. 또한, 제안된 U-Motion 코덱의 각 구성 요소가 효과적임을 검증하기 위한 변별 연구도 수행되었습니다. U-Motion은 다양한 인체 포인트 클라우드와 모션 패턴에 대해 우수한 견고성을 나타냅니다.



### Delta-NAS: Difference of Architecture Encoding for Predictor-based Evolutionary Neural Architecture Search (https://arxiv.org/abs/2411.14498)
- **What's New**: 이번 논문은 Neural Architecture Search (NAS)의 효율성을 높이기 위해, 고비용의 세밀한 NAS 대신 저비용의 세밀한 NAS를 수행할 수 있는 알고리즘을 제안합니다. 이 알고리즘은 비슷한 네트워크의 정확도 차이를 예측하여 문제를 낮은 차원으로 투영함으로써 계산 복잡성을 저감합니다. 이를 통해 검색 공간의 크기에 대한 계산 복잡성을 기하급수적으로 줄일 수 있습니다.

- **Technical Details**: 제안된 Delta-NAS는 고전적인 NAS 방법들이 겪는 성능 저하 없이 데이터의 압축을 가능하게 합니다. 두 네트워크 간의 정확도 차이를 예측함으로써, Delta-NAS는 드문 후보 신경망을 구성할 수 있는 스파스 인코딩을 생성합니다. 또한, 이 방법은 기존 네트워크와 둘의 차이를 예측하여 성능을 평가하고, 높은 샘플 효율성과 함께 NAS 성능을 크게 향상시킵니다.

- **Performance Highlights**: Delta-NAS 방법은 여러 NAS 벤치마크에서 우수한 성능을 달성하며, 후보 아키텍처의 수가 기하급수적으로 증가할 때에도 선형적으로 확장됩니다. 이 알고리즘은 현대 NAS 분야의 요구에 맞는 성능과 계산 비용을 조절하며, 다양한 ML 작업에서 최신 성과를 기록하고 있습니다.



### Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models (https://arxiv.org/abs/2411.14495)
Comments:
          Accepted to WACV 2025 (Winter Conference on Applications of Computer Vision)

- **What's New**: 본 논문에서는 새로운 3D 테스트 시간 적응 방법인 3D Denoising Diffusion Test-Time Adaptation (3DD-TTA)을 소개합니다. 이 방법은 손상된 포인트 클라우드를 적응시키기 위해 경량화된 diffusion 방식을 활용하여 원본 모델의 파라미터를 변경하지 않고도 적용됩니다. 3DD-TTA는 잠재적 변형을 고려하면서 데이터의 충실성을 보존하도록 설계되었습니다.

- **Technical Details**: 3DD-TTA는 Variational Autoencoder (VAE)를 사용하여 손상된 포인트 클라우드를 인코딩하고 Gaussian noise를 추가하여 잠재 포인트를 만드는 방식으로 작동합니다. 이를 통해 얻은 잠재 포인트는 역 확산 과정 중에 수정되어 원본 도메인과의 정합성을 유지합니다. Selective Chamfer Distance (SCD)라는 새로운 거리를 도입하여, 원래의 잠재 포인트와 예측된 포인트 간의 차이를 측정하고 이 거리를 최소화하여 개선된 결과를 생성합니다.

- **Performance Highlights**: ShapeNet 데이터셋에서의 실험 결과, 3DD-TTA는 기존의 방법들을 초월하는 성능을 보였습니다. 또한 ModelNet40 및 ScanObjectNN 데이터셋에서도 일반화 가능성을 입증하였으며, 최첨단 성과를 달성했습니다. 연구 결과는 코드와 함께 공개되어 이를 활용한 후속 연구에 기여할 것입니다.



### dc-GAN: Dual-Conditioned GAN for Face Demorphing From a Single Morph (https://arxiv.org/abs/2411.14494)
- **What's New**: 이 논문에서는 두 개의 얼굴 이미지를 결합하여 생성된 얼굴 변형(facial morph)의 원본 이미지를 복원하는 얼굴 디모핑(face demorphing) 기법인 dc-GAN을 제안합니다. 기존의 디모핑 기술은 테스트 중 정체성을 가정하거나 유사한 결과를 생성하는 제약이 있는 반면, 우리의 방법은 이러한 문제를 극복하고 고품질의 원본 이미지를 재구성할 수 있습니다.

- **Technical Details**: dc-GAN은 생성적 적대 신경망(Generative Adversarial Network, GAN)에 기반한 새로운 디모핑 방법입니다. 이 방법은 임의의 변형 이미지(morph image)에 대해 적응하며, 현재까지 제안된 디모핑 기법 중에서 가장 일반화가 뛰어난 것으로 평가됩니다. 실험은 AMSL, FRLL-Morphs와 MorDiff 데이터셋에서 수행되어 성능을 입증했습니다.

- **Performance Highlights**: 본 연구에서는 dc-GAN이 이전의 방법들에 비해 높은 유효성을 보인다는 것을 확인했습니다. 특히, dc-GAN은 보안 생체 인증 시스템의 얼굴 변형 이미지를 인식하고 원본 이미지를 복원하는 데 성공적으로 사용될 수 있습니다. 기존 방법들이 다양한 시나리오에서 제한된 성능을 보이는 데 비해, 우리의 기법은 많은 상황에서도 효과적으로 작동할 수 있습니다.



### WildLMa: Long Horizon Loco-Manipulation in the Wild (https://arxiv.org/abs/2411.15131)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 기존의 모바일 조작 기술을 개선하기 위한 새로운 접근법인 WildLMa를 제안합니다. WildLMa는 로봇이 다양한 실세계 환경에서 일반화 가능한 기술을 습득하고 긴 시간의 작업을 수행할 수 있도록 설계되었습니다. 또한 이 시스템은 일반화 가능한 visuomotor skills를 구현하기 위한 WildLMa-Skill 라이브러리와 LLM 플래너와 연결된 WildLMa-Planner 인터페이스를 통합하고 있습니다.

- **Technical Details**: WildLMa는 VR 기반의 전체 신체 원격 조작 및 Traversability를 위한 저수준 제어기의 적응을 포함합니다. 이 시스템은 CLIP을 이용하여 언어 조건에 맞춘 모방 학습을 통해 훈련 된 일반화 가능한 비주얼-모터 스킬을 제공합니다. 많은 demonstration을 통해 얻은 높은 품질의 훈련 데이터는 조작 성공률을 높이는 데 중요한 역할을 합니다.

- **Performance Highlights**: WildLMa는 클리닝, 아티큘레이티드 오브젝트 조작, 책장 아이템 재배치 등 실용적인 로봇 응용 사례를 통해 철저한 평가를 받았습니다. 이 연구는 몇 차례의 demonstration만으로도 기존 RL 기준에 비해 높은 조작 성공률을 달성했습니다. WildLMa의 접근 방식은 향후 연구를 위한 기초를 마련하는 데 기여합니다.



### Health AI Developer Foundations (https://arxiv.org/abs/2411.15128)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문에서는 Health AI Developer Foundations (HAI-DEF)라는 새로운 의료 머신 러닝 모델 개발 스위트를 소개합니다. 이 스위트는 다양한 도메인에 최적화된 사전 훈련된 모델과 도구, 레시피를 포함하고 있으며, 의료 분야의 AI 개발을 가속화하도록 설계되었습니다. HAI-DEF는 레이블이 적은 데이터로도 사용 가능하며, 훈련 시간을 단축시키고 컴퓨팅 비용을 낮추는 데 도움을 줍니다.

- **Technical Details**: HAI-DEF는 다양하고 큰 데이터셋을 이용해 훈련된 여러 개별 모델로 구성됩니다. 예를 들어, CXR Foundation은 이미지 및 관련 정확한 정보를 활용해 CXRs를 표현하는데 사용되며, Path Foundation은 자기 지도 학습 방법을 통해 병리학적 이미지 패치를 훈련합니다. 각 모델은 특정 데이터 모달리티 및 용도에 맞춰 조정되었습니다.

- **Performance Highlights**: HAI-DEF의 모델들은 데이터 효율성 측면에서 우수한 성능을 보였으며, 이는 일반 모델과의 성능 벤치마킹을 통해 입증되었습니다. 예를 들어, CXR Foundation 모델은 다양한 작업에서 높은 성능을 기록했으며, Derm Foundation 모델은 419개의 피부 질환을 효과적으로 처리할 수 있음을 입증했습니다. 이러한 결과는 도메인 특정 모델이 요구되는 데이터와 기술적 전문성을 줄이면서도 기존 접근 방식보다 우수한 성능을 달성할 수 있음을 시사합니다.



### Dimension-independent rates for structured neural density estimation (https://arxiv.org/abs/2411.15095)
- **What's New**: 이번 논문에서는 딥 뉴럴 네트워크가 이미지, 오디오, 비디오 및 텍스트와 같은 구조적 밀도를 학습하는 데 있어 차원 독립적인 수렴 속도를 달성한다는 것을 보여줍니다. 특히, 마르코프(Markov) 그래프에 대해 최대 클리크 크기가 r인 경우 비모수적 밀도 추정에서 n^{-1/(4+r)} 속도를 달성할 수 있는 것을 입증합니다. 이 결과는 데이터의 주변 차원에 독립적이며, 실제 데이터 모델에도 적용 가능하다는 점에서 의미가 있습니다.

- **Technical Details**: 논문에서는 Markov 무작위장(Markov Random Field, MRF)에 대한 개념을 도입하고, 거리와 시간적으로 거의 독립적인 공변량(covariates)을 활용하여 밀도 추정 문제를 해결하는 방법을 설명합니다. MRF에 기반한 밀도 추정에서 딥 뉴럴 네트워크는 간단한 L^2 손실 함수를 최소화하여 n^{-1/(4+r)}의 수렴 속도를 얻습니다. 이는 기존의 밀도 추정에서 요구되는 n^{-1/(2+d)}보다 우수한 성능을 보이고, 문제의 유효 차원이 사실상 최대 클리크 크기 r로 제한됨을 보여줍니다.

- **Performance Highlights**: 이 연구는 깊은 학습(deep learning)의 유효성을 강조하며, 조화롭게 결합된 데이터 유형에서 MRF 가정을 통해 차원 독립적인 수렴 속도를 달성할 수 있음을 입증합니다. 다양한 데이터 유형에서 Neural Networks가 어떻게 효과적인 차원 축소를 가능하게 하는지를 보여줍니다. 이로 인해 실질적인 데이터 세트(예: 이미지, 소리, 비디오 등)의 복잡한 밀도 함수를 효과적으로 학습할 수 있는 새로운 전망이 제공됩니다.



### Quantum-enhanced unsupervised image segmentation for medical images analysis (https://arxiv.org/abs/2411.15086)
Comments:
          16 pages, 7 figures

- **What's New**: 이 논문은 유방암 이미지를 세분화하기 위한 첫 번째 끝-to-end(End-to-End) 양자 강화(Quantum-Enhanced) 비지도 학습 프레임워크를 제안합니다. 기존의 방법과 달리, 요구되는 데이터 세트의 수를 줄이고 성능의 정확도를 유지하는 새로운 접근 방식을 탐구합니다. 데이터의 일반화 문제를 해결하기 위해, 이 방법은 QUBO(Quadratic Unconstrained Binary Optimization) 문제로 공식화되어 성능과 계산 요구 사항 간의 균형을 맞추고 있습니다.

- **Technical Details**: 제안된 접근 방식에서는 양자 영감을 받은 이미지 표현 방법이 사용되며, 이는 세분화 마스크의 초기 추정값으로 작용합니다. 이후 QUBO 문제로 변환된 세분화 작업은 배경과 종양 영역 간의 대비를 극대화하고, 연결 성분을 최소화하여 응집력 있는 세분화 마스크를 생성하는 것을 목표로 합니다. 다양한 양자 및 양자 영감 기술이 평가되었으며, 양자 어닐링(Quantum Annealing)과 변분 양자 회로(Variational Quantum Circuits)가 고전적 최적화 기법과 유사한 성능을 입증했습니다.

- **Performance Highlights**: 제안된 비지도 파이프라인의 성능은 감독되는 UNet 모델과 유사하며, 전통적인 Otsu 임계값 접근법보다 훨씬 우수한 결과를 나타냅니다. QUBO 문제 해결 과정은 이 작업에 대해 고전적 Gurobi 해결 방법보다 약 10배 빠른 실행 시간을 기록했으며, 실행 시간의 변동성도 현저히 낮습니다. 이러한 결과는 유방암 이미지 세분화를 위한 비지도 학습 대안으로서의 접근 방식의 가능성을 보여줍니다.



### Leapfrog Latent Consistency Model (LLCM) for Medical Images Generation (https://arxiv.org/abs/2411.15084)
Comments:
          Total 16 pages including 5 figures and 36 references

- **What's New**: 의료 영상 데이터의 접근성 부족은 심각한 문제로, 의료 진단을 위한 딥러닝 모델의 효과적인 학습을 저해하는 주요 요인이다. 이에 데이터셋 MedImgs(메드입스)를 구축하여 61가지 질병 유형과 159개 클래스에 걸쳐 250,127개 이상의 이미지를 수집하였다. 본 연구에서는 이 데이터셋을 기반으로 실시간 고해상도 이미지를 생성하는 Leapfrog Latent Consistency Model (LLCM)을 제안한다. LLCM은 역확산 과정 수식을 확률 흐름 보통 미분 방정식(PF-ODE)로 공식화하고 여기에 레프로그 알고리즘을 적용하였다.

- **Technical Details**: LLCM은 수천 개의 데이터 샘플을 필요로 하지 않으며, 1-4번의 추론 단계 만으로 512×512 해상도의 이미지를 생성할 수 있는 새로운 접근 방식을 제공한다. 기존의 난이도를 줄이고 샘플링 속도를 높이는 레프로그 방법론이 적용된 이 모델은 기존의 diffusion 모델들보다 효율적이다. 가장 큰 장점은 우리의 MedImgs 데이터셋에 따라 고해상도 이미지를 빠르게 생성할 수 있는 동시에 다양한 맞춤형 의료 이미지 데이터셋에 대해 미세 조정이 가능하다는 점이다.

- **Performance Highlights**: LLCM은 기존 모델들과 비교했을 때 개가 심장 X선 이미지와 같은 보지 못한 데이터를 사용한 실험에서 성능 상의 우위를 보였다. 데이터의 다양성을 효과적으로 다루어 의료 영상 생성의 현저한 향상을 이루어냈으며, 이는 AI 기반 의학 솔루션의 접근성과 정확성을 높이는 데 기여할 것으로 기대된다. 본 모델의 소스 코드는 공개되어 있어 관련 연구자들에 의해 활용될 수 있을 것이다.



### RankByGene: Gene-Guided Histopathology Representation Learning Through Cross-Modal Ranking Consistency (https://arxiv.org/abs/2411.15076)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문에서는 Spatial transcriptomics (ST) 데이터를 통해 유전자 발현과 조직 이미지를 효과적으로 정렬하는 새로운 프레임워크를 제안합니다. 기존의 방법들은 주로 직접적인 정렬에 의존했으나, 복잡한 교차 모달(cross-modal) 관계를 포착하지 못하는 한계가 있었습니다. 이를 해결하기 위해 랭킹 기반의 정렬 손실(ranking-based alignment loss)을 사용하여 모달리티 간의 상대적 유사성을 보존하고 강력한 다중 스케일 정렬을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 고차원 및 희소성을 포함한 유전자 발현 데이터의 복잡성으로부터 오는 방해를 최소화하기 위해 교사-학생 네트워크 아키텍처(self-supervised knowledge distillation)를 채택하여 정렬의 안정성을 강화합니다. 특히, 랭킹 기반 손실을 통해 이미지와 유전자 특징 간의 모달리티 특정 왜곡에 대한 견고함을 유지하며, 글로벌 정렬을 통해 먼 거리의 특징들이 상호작용하고 정렬될 수 있도록 설계되었습니다.

- **Performance Highlights**: 수행된 광범위한 실험 결과, 제안한 프레임워크는 기존의 방법들에 비해 유전자 발현 예측 및 생존 예측에서 개선된 성능을 보여줍니다. 특히, 유전자 기반의 이미지 특징 학습을 통해 디지털 병리학 분야에서 임상 결과 예측에 크게 기여할 수 있는 강력한 도구로 자리매김할 수 있습니다.



### Detecting Hallucinations in Virtual Histology with Neural Precursors (https://arxiv.org/abs/2411.15060)
- **What's New**: 이 연구에서는 Virtual Staining (VS) 모델의 환각(hallucination) 탐지 문제를 공식적으로 정립하고, Neural Hallucination Precursor (NHP)라는 새로운 탐지 방법을 도입합니다. NHP는 VS 모델의 임베딩 공간에서 환각의 선행 지표를 식별하게 설계되어 있습니다. 또한, 다양한 VS 환경에서의 검증 결과를 통해 NHP의 효과성과 강건성을 입증합니다.

- **Technical Details**: NHP는 VS 모델의 특징 신호를 병합하고 관련 성능 매개변수를 자가 조정하여 최적의 마커를 찾는 후처리(post-hoc) 검색 방법입니다. 이 방법은 대량의 데이터 처리(throughput)와 다재다능성(versatility) 요구를 충족시키며, 다양한 장기 타입, 소스/타겟 쌍, 그리고 이미지-to-이미지 변환(I2IT) 기법에 관계없이 각각 적절한 성능을 보입니다.

- **Performance Highlights**: NHP는 병리학적 정보 손실 없이 환각을 실시간으로 감지하는 능력을 보여주며, 그럼에도 불구하고 환각이 적더라도 VS 모델의 신뢰성을 보장하지는 못한다는 점을 강조합니다. 이는 현재의 VS 평가 관행을 재평가할 필요성을 제기하며, 환각의 검출 방식이 단기적인 경향을 넘어 더 포괄적이어야 함을 시사합니다.



### Fantastic Biases (What are They) and Where to Find Them (https://arxiv.org/abs/2411.15051)
Comments:
          Publication in Spanish in the Journal Bits de Ciencias: this https URL

- **What's New**: 이 논문은 인공지능(AI) 모델의 공정성(Fairness) 문제를 탐구합니다. 특히, 알고리즘이 성별, 인종 및 계급 정보에 바탕을 두는 경향이 있으며, 이로 인해 불공정한 결과를 초래할 수 있다는 점을 강조합니다. 바이어스(Bias)의 정의와 그 사회적 영향에 대한 통찰력을 제공하며, 공정한 AI 시스템을 구축하기 위한 접근 방법을 논의합니다.

- **Technical Details**: AI 모델의 공정성을 유지하는 것은 복잡한 과제로, ML 모델은 인종, 성별 등과 관련된 실제 세계 데이터에서 학습합니다. 이러한 데이터의 바이어스는 모델이 선별적으로 학습하고 이를 증폭시킬 수 있게 합니다. 공정성을 정의하는 기준은 집단의 경험, 문화, 역사, 정치적 요소 등을 고려해야 하며, 상충되는 가치관이 존재할 수 있음을 나타냅니다.

- **Performance Highlights**: AI 모델의 성능은 지속적으로 향상되고 있으며, 보다 정교한 결정이 요구됩니다. 그러나 비록 완전한 공정성을 이루는 것은 어렵지만 기존 모델의 바이어스를 제거하는 방법을 통해 보다 공정한 시스템으로 발전할 수 있는 가능성을 제시합니다. 최종적으로, 알고리즘의 투명성을 높이고 사용자에게 공정함을 보장하는 것이 필수적입니다.



### Exploring Foundation Models Fine-Tuning for Cytology Classification (https://arxiv.org/abs/2411.14975)
Comments:
          5 pages, 2 figures

- **What's New**: 이 논문에서는 세포학 슬라이드 분석의 효율성을 향상시키기 위한 기초 모델의 활용 가능성을 모색합니다. 특히, 적은 샘플로 학습할 수 있는 Low-Rank Adaptation (LoRA) 기법을 통해 기존의 기초 모델을 미세 조정하는 방법을 효과적으로 제안하고 있습니다. 이를 통해 단순 분류 및 복잡한 분류 작업에서 최첨단 성능을 달성하였고, 더 적은 데이터 샘플을 사용하면서도 성능을 향상시킬 수 있음을 입증하였습니다.

- **Technical Details**: 연구에서는 세포학 분류를 위해 유명한 기초 모델 다섯 개를 평가하였습니다. LoRA를 사용해 미세 조정한 모델이 이전의 방식에 비해 성능이 높아졌다고 보고하고 있습니다. 또한, 세포학 작업에 특화된 모델이 일반 모델보다 더 탁월한 적응성을 보여준다고 밝혔습니다.

- **Performance Highlights**: 모델의 성능 평가에서는 세포학 분류의 네 가지 벤치마크 데이터셋에서 뛰어난 성과를 얻었습니다. CLIP의 비전 인코더는 LoRA로 미세 조정 후 단일 데이터셋의 70%만으로도 최첨단 성능을 달성하여, 현재의 최선단 모델보다 훨씬 적은 훈련 가능한 매개변수를 사용하였습니다. 이를 통해 LoRA의 효과가 다양한 의학적 작업에서 강력하게 입증되었습니다.



### SwissADT: An Audio Description Translation System for Swiss Languages (https://arxiv.org/abs/2411.14967)
- **What's New**: 이번 연구에서는 SwissADT라는 다국어 및 다중 모달 오디오 설명 번역(ADT) 시스템을 소개합니다. 이는 독일어, 프랑스어, 이탈리아어 및 영어를 지원하며, 시각 장애인 및 시각적 제한이 있는 사용자들이 접근할 수 있도록 설계되었습니다. SwissADT는 비디오 클립을 활용하여 시각적 정보를 포함한 AD 스크립트를 자동으로 번역함으로써 다국어 인구에 대한 정보 접근성을 향상시키고자 합니다.

- **Technical Details**: SwissADT는 비디오 클립과 텍스트 입력을 결합하여 AD 스크립트를 번역하는 LLM 기반 시스템입니다. 이 시스템은 AD 스크립트에 가장 적합한 비디오 순간을 식별하기 위해 CG-DETR라는 비디오 시간 그라우더를 사용합니다. 최종적으로 AD 번역기를 위해 비디오 프레임을 샘플링하고, GPT-4 모델을 활용하여 다국어 번역을 수행합니다.

- **Performance Highlights**: SwissADT의 성능은 자동 및 인간 평가를 통해 검증되었습니다. 실험 결과, SwissADT는 ADT 작업에서 유망한 성능을 보여주었으며, AD 전문가의 의견을 토대로 시각 정보를 입력에 포함하는 것이 결과에 긍정적인 영향을 미친 것으로 나타났습니다. 이 연구는 향후 다국어 ADT 시스템의 발전 방향에 중요한 기초 자료를 제공합니다.



### LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinemen (https://arxiv.org/abs/2411.14961)
- **What's New**: 이 논문은 LoRA (Low-Rank Adaptation)와 Federated Learning (FL)을 결합하여 발생하는 두 가지 주요 문제를 해결하는 새로운 방식인 LoRA-FAIR를 제안합니다. 서버에서 LoRA 매트릭스를 평균화할 때 발생하는 Server-Side LoRA Aggregation Bias와 연속훈련 동안 초기화 문제인 Client-Side LoRA Initialization Drift를 동시에 해결하려고 합니다. LoRA-FAIR는 이러한 문제를 해결하면서도 계산적 효율성을 유지하여 우수한 성능을 자랑합니다.

- **Technical Details**: LoRA는 저랭크 행렬을 모델에 도입하여 훈련 가능한 파라미터의 수를 줄이는 PEFT 기법입니다. LoRA-FAIR는 서버 측에서 평균화된 LoRA 모듈을 고정하고, 수정 항을 추가하여 이상적인 업데이트에 가까운 서버 최적화를 수행합니다. 이를 통해 클라이언트에서 수집한 정보의 평균을 유지하면서도 동적으로 업데이트되어 서버의 최적화 효율성을 높입니다.

- **Performance Highlights**: 대규모 데이터셋을 이용한 ViT 및 MLP-Mixer 모델 실험 결과, LoRA-FAIR는 기존의 가장 앞선 방법들과 비교하여 일관되게 성능 개선을 보였습니다. 이러한 성능 향상은 LoRA의 효율적인 미세 조정에 의해 이루어지며, FL 환경에서 LoRA를 강화할 수 있는 중요한 기초를 제공합니다.



### Implementation of Real-Time Lane Detection on Autonomous Mobile Robo (https://arxiv.org/abs/2411.14873)
Comments:
          4 pages, 9 figures 2 tables

- **What's New**: 이 논문은 자율 이동 로봇(Autonomous Mobile Robot)에서 학습 기반 차선 감지 알고리즘을 구현한 내용을 설명합니다. Ultra Fast Lane Detection 알고리즘의 실시간 적용을 목표로 하며, SEATER P2MC-BRIN 프로토타입에 카메라를 사용하여 구현하였습니다. 또한 Jetson Nano 플랫폼에서 성능을 최적화하는 방안을 다루고 있습니다.

- **Technical Details**: 실험은 CULane과 TuSimple의 두 가지 데이터셋을 사용하여 알고리즘의 데이터 처리 속도 및 정확성을 평가하였습니다. 알고리즘은 TensorRT로 변환 후 Jetson Nano 플랫폼에서 ONNX 모델보다 더 최적화되어 101 ms, TuSimple에서는 105 ms의 처리 속도를 기록했습니다. 이는 이전 모델에 비해 약 22배 빠른 성능을 보여줍니다.

- **Performance Highlights**: 알고리즘은 외부(public) 데이터셋에서 좋은 정확도를 보였으나, 내부 데이터셋에서는 성능이 부족한 것으로 나타났습니다. 향후 연구는 전이 학습(transfer learning)과 세부 조정(fine-tuning)에 초점을 맞춰 실내 차선 감지의 정확도를 향상시키는 방향으로 진행되어야 할 것입니다.



### Benchmarking the Robustness of Optical Flow Estimation to Corruptions (https://arxiv.org/abs/2411.14865)
Comments:
          The benchmarks and source code will be released at this https URL

- **What's New**: 본 연구에서는 Optical flow estimation의 강인성을 평가하기 위해 7가지의 시간적 손상(tapestry corruptions)과 17가지의 일반적인 단일 이미지 손상을 도입했습니다. 첫 번째 손상 강인성 벤치마크인 KITTI-FC와 GoPro-FC를 구축하여 Optical flow 모델의 성능을 체계적으로 연구할 수 있는 기초 자료를 제공합니다. 연구에서는 Corruption Robustness Error (CRE), Corruption Robustness Error ratio (CREr), Relative Corruption Robustness Error (RCRE)와 같은 다양한 강인성 지표도 제안하여 모델의 강인성을 정량화합니다.

- **Technical Details**: Optical flow estimation의 연구가 최근 심화되면서 특히 자율주행과 비디오 편집 분야에서의 강인성 연구가 중요해졌습니다. 기존 연구들은 주로 adversarial 공격에 초점을 맞춰 강인성을 평가하였지만, 일반적인 손상에 대한 연구는 부족했습니다. 이 연구는 2929개의 모델 변형을 평가하여, 모델의 절대 강인성은 예측 성능에 크게 의존하며, 지역 정보를 망치는 손상이 시각적 효과를 줄이는 손상보다 더 심각한 영향을 미친다는 흥미로운 발견을 제시합니다.

- **Performance Highlights**: 연구 결과, Optical flow 모델의 강인성에 대한 새로운 인사이트가 도출되었습니다. 특히, 손상으로 인해 지역 정보가 감소되는 경우가 시각적 효과가 감소하는 경우보다 더 큰 영향을 미친다는 점이 강조되었습니다. 따라서 모델 설계 및 적용에서 추후 연구 방향으로는 Transformer 유사 구조 도입, 비지도 학습 방법을 적용, 지역 정보의 신뢰성 향상 등이 제안되었습니다. 이를 통해 Optical flow estimation의 강인성을 높일 수 있는 구체적인 방향성을 제시하고 있습니다.



### Cell as Point: One-Stage Framework for Efficient Cell Tracking (https://arxiv.org/abs/2411.14833)
Comments:
          17 pages, 8 figures, 8 tables

- **What's New**: 이 논문에서는 다단계 셀 추적 방법의 한계를 극복하기 위해 새로운 단일 단계의 CAP (Cell as Point) 프레임워크를 제안합니다. CAP는 셀을 포인트로 간주하여 트래킹 중 다양한 셀 포인트를 공동으로 추적함으로써 프로세스를 단순화하고 라벨 요구사항을 줄입니다. 이 프레임워크는 기존 방법들보다 10배에서 55배 더 효율적인 성과를 보여줍니다.

- **Technical Details**: CAP는 적응형 이벤트 기반(AEG) 샘플링과 롤링-창(rolling-as-window) 추론 방식을 통해 데이터 불균형과 긴 시퀀스의 새로운 셀 추적 문제를 해결합니다. AEG 샘플링은 세포 분열 사건의 불균형을 완화하기 위한 데이터 샘플링 전략입니다. 롤링-창 추론 방법은 긴 시퀀스 내에서 새로운 셀을 연속적으로 추적할 수 있게 합니다.

- **Performance Highlights**: CAP는 검출이나 분할 단계가 필요 없는 단일 단계 추적 방식을 통해 높은 성능을 달성합니다. 이 접근 방식은 복잡한 데이터 전처리 문제를 줄이고 세포 추적에 있어 낮은 데이터 주석 요구사항을 가지며, 다양한 복잡한 세포 사건을 효과적으로 표현할 수 있습니다. CAP는 기존의 다단계 방법들과 비교해 더 높은 성능을 보이는 것을 목표로 하고 있습니다.



### Continual SFT Matches Multimodal RLHF with Negative Supervision (https://arxiv.org/abs/2411.14797)
- **What's New**: 이 논문에서는 비전-언어 모델(VLM)의 선호 정렬 과정에서 다중 모달 RLHF(강화 학습에서 인간 피드백 사용)의 고유한 가치가 거부된 응답의 로그리트를 통한 부정 감독에 있음을 관찰하고, 새로운 부정 감독 미세 조정(nSFT) 접근 방식을 제안합니다. nSFT는 VLM과 간단한 SFT 손실(Supervised Finetuning Loss)을 지속적으로 정렬하며, 기존의 슬롯 방식에 비해 메모리 효율성이 뛰어난 장점을 가지고 있습니다. 이는 여러 데이터 세트 및 평가 지표와 비교하여 nSFT가 더 나은 성능을 발휘함을 수치적으로 보여줍니다.

- **Technical Details**: 논문의 핵심은 nSFT가 RLHF 최적화에서 부정 감독을 분리하여 훈련을 더 효율적으로 만들 수 있다는 점입니다. 다중 모달 RLHF에서는 메모리를 2개 또는 4개의 모델을 사용해야 하지만, nSFT는 오직 하나의 모델에서 수행된다는 점이 강조됩니다. 이 논문은 LLM(예: GPT-4)를 사용하여 부정 감독의 불일치를 식별하고 이미지 관련 대화를 구성하여 부정 응답으로부터 자기 견책의 학습을 돕습니다.

- **Performance Highlights**: nSFT는 다양한 실험을 통해 SFT의 한계를 보완하고, 여러 평가 지표에서 최고의 성과를 달성합니다. 특히 LLaVA-NeXT와 같은 강력한 VLM에서도 가설이 유지됩니다. 시험 결과는 nSFT가 기존의 다중 모달 RLHF 방법과 견줄 만큼 강력하며, 향후 연구에 자극이 되길 바랍니다.



### Simplifying CLIP: Unleashing the Power of Large-Scale Models on Consumer-level Computers (https://arxiv.org/abs/2411.14789)
- **What's New**: 이 논문에서는 Nvidia RTX3090 GPU 하나와 1TB의 저장소만 사용하여 경쟁력 있는 CLIP 모델을 학습할 수 있는 방법을 제시하고 있습니다. 기존 모델의 구조를 단순화하고 학습 시 메모리 요구량을 줄이기 위해 Weight Inheritance와 multi-stage Knowledge Distillation (WIKD) 전략을 통합했습니다. 또한, 작은 데이터셋의 문제를 해결하기 위해 합성 캡션을 생성하고, 긍정 및 부정 이미지-텍스트 쌍 간의 구별력을 최적화하기 위한 Pair Matching (PM) 손실 함수를 도입했습니다.

- **Technical Details**: 모델 구조는 SAS-P 블록을 통해 가중치를 공유하여 구성됩니다. WIKD 방법을 통해 효율적인 학습을 구현하며, PM 손실 함수를 사용하여 훈련 성능을 향상시킵니다. 또한, CC12M 데이터셋을 개선하여 새로운 CC12M-SYN 데이터셋을 생성함으로써 데이터 다양성과 품질을 높였습니다. 이 모든 방법의 결합으로 모델 학습의 수렴 속도가 크게 향상되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 SiCLIP 프레임워크는 38개 데이터셋에서 여러 인기 있는 방법과 비교했을 때 새로운 최첨단 데이터 규모-파라미터-정확도 트레이드오프를 달성했습니다. 제안된 방법은 소비자용 컴퓨터에서 CLIP 모델의 연구를 대중화하는 데 기여할 것으로 기대됩니다. WIKD 및 PM 손실을 통해 작은 규모의 데이터셋에 대한 학습 효율성을 크게 개선했으며, 이는 경쟁력 있는 성능을 유지하는 데 중요한 역할을 합니다.



### FastGrasp: Efficient Grasp Synthesis with Diffusion (https://arxiv.org/abs/2411.14786)
- **What's New**: 이번 연구에서는 기존의 두 단계 프로세스를 대체하는 새로운 접근 방식인 FastGrasp를 소개합니다. FastGrasp는 중간 표현을 생성하지 않고도 인간의 손이 객체를 잡는 자세를 한 단계에서 직접 생성할 수 있는 혁신적인 방법입니다. 이를 통해 생성 속도와 다양한 손 자세를 크게 개선했습니다.

- **Technical Details**: FastGrasp는 라텐트 디퓨전 모델(latent diffusion model)과 적응 모듈(adaptation module)을 활용해, 손과 객체 간의 물리적 제약을 효과적으로 반영합니다. 이 모델은 물체의 포인트 클라우드(point cloud)를 기반으로 조잡한 디퓨전 생성 과정에서 손 자세를 학습합니다. 마지막으로, 생성된 손 자세는 MANO(MANO) 매개변수로 디코딩됩니다.

- **Performance Highlights**: 실험 결과, FastGrasp는 기존 최첨단 방법보다 더 빠른 추론 시간과 다양한 형질, 향상된 자세 품질을 달성했습니다. HO-3D, OakInk 및 Grab 데이터 세트에서 평가된 결과, 생성된 손 자세는 자연스럽고 물리적으로 현실적인 상호작용을 보여주었습니다.



### A Benchmark Dataset for Collaborative SLAM in Service Environments (https://arxiv.org/abs/2411.14775)
Comments:
          8 pages, 6 figures, Accepted to IEEE RA-L

- **What's New**: 최근의 서비스 환경 다양화로 인해 단일 로봇이 수행하기 어려운 복잡한 작업에 대한 요구가 증가하고 있습니다. 이에 따라 다수의 로봇이 협력하여 작업을 수행하는 Collaborative SLAM (C-SLAM) 기술에 대한 관심이 높아지고 있습니다. 본 논문에서는 다양한 실내 서비스 환경에서 다수의 서비스 로봇을 위한 새로운 멀티 모달 C-SLAM 데이터셋인 CSE 데이터를 소개하고 있습니다.

- **Technical Details**: CSE 데이터셋은 실제 서비스 환경에서 발생할 수 있는 동적 객체와 동질적인 장면의 도전을 포함하여, NVIDIA Isaac Sim을 활용해 생성된 데이터입니다. 이 데이터셋은 병원, 사무실, 창고와 같은 세 가지 일반적인 실내 서비스 환경을 구성하고, 각 환경의 다양한 동적 객체와 함께 실제 서비스 로봇의 행동을 모사합니다. 각 로봇은 정확하게 시간 동기화된 센서 데이터와 GT 포즈를 제공합니다.

- **Performance Highlights**: 논문에서는 여러 최신 단일 로봇 SLAM 및 다중 로봇 SLAM 방법을 통해 데이터셋의 유용성을 평가하고 있습니다. 본 데이터셋은 동적 객체와 같은 복잡한 상황에 적합한 평가를 위한 적절한 시나리오를 제공하여, C-SLAM 알고리즘의 효율성을 평가할 수 있도록 돕습니다. 최종적으로, CSE 데이터셋은 서비스 로봇 운용에 있어 더욱 현실적이고 도전적인 환경을 제공하여 연구자들에게 유용한 리소스가 될 것입니다.



### Comparative Analysis of nnUNet and MedNeXt for Head and Neck Tumor Segmentation in MRI-guided Radiotherapy (https://arxiv.org/abs/2411.14752)
Comments:
          15 pages, 3 figures

- **What's New**: 본 연구에서는 HNTS-MRG24 MICCAI 챌린지에 대한 해결책으로서, 두 가지 최첨단 딥러닝 모델인 nnUNet과 MedNeXt를 활용하여 방사선 치료(RT) 전후 MRI 이미지에서 머리 및 목 종양의 자동 분할을 구현했습니다. 이 연구의 주요 목표는 의료 이미지에서의 복잡한 종양 해부학을 정확하게 세분화하는 것입니다. 발표된 솔루션은 GitHub 리포지토리에서 공개되어 있으며, 연구자들이 쉽게 접근할 수 있는 형식입니다.

- **Technical Details**: 본 연구에서는 150개의 HNC 진단을 받은 환자로부터 수집된 HNTS-MRG2024 데이터셋을 활용했습니다. 데이터셋은 RT 전후의 T2 가중 MRI 스캔과 해당 종양 볼륨에 대한 분할 마스크를 포함하고 있습니다. 모델 훈련 과정에서는 주어진 데이터에 대해 사전 훈련(pretraining) 및 미세 조정(fine-tuning)을 적용하여 성능을 향상시켰습니다.

- **Performance Highlights**: 제안된 솔루션은 Task 1에서 0.8254의 총체적 Dice Similarity Coefficient를 기록하며 1위를 차지했습니다. Task 2에서는 점수 0.7005로 8위를 기록하며, 다양한 모델과 학습 전략을 통한 성능 비교가 이루어졌습니다. 이는 MR 영상 기반 자동 분할 기술이 방사선 치료 분야에서 효과적으로 적용될 수 있음을 나타냅니다.



### FedMLLM: Federated Fine-tuning MLLM on Multimodal Heterogeneity Data (https://arxiv.org/abs/2411.14717)
- **What's New**: 이 연구에서는 Multimodal Large Language Models (MLLMs)와 관련하여 새로운 벤치마크를 소개합니다. 이는 Federated Learning (FL) 프레임워크 내에서 MLLM의 다양한 다운스트림 작업을 평가하기 위한 허브로 기능합니다. 또한, 네 가지 대표적인 FL 방법을 통합하는 일반적인 FedMLLM 프레임워크를 개발하여 다중 모드 이질성 문제를 해결하고자 합니다.

- **Technical Details**: MLLM은 강화된 멀티모달 데이터 처리 기술을 기반으로 하며, FL을 사용하여 개인 데이터를 포함한 훈련 데이터의 범위를 확장합니다. FL의 다양한 기법과 함께, 두 가지 모달리스틱(모달-무관) 전략이 설계되어 다양한 모달 이질성을 관리합니다. 이러한 접근법은 MLLMs가 실제 응용 프로그램에서 다중 모드 입력을 지원하도록 하고, 모달 이질성에 따른 편향을 최소화하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과에 따르면, FL 패러다임이 MLLMs의 성능을 향상시키며 훈련 데이터의 범위를 넓혀주고, 다중 모드 이질성 문제를 완화하는 긍정적인 영향을 미쳤습니다. 연구에서 제안된 방법론은 2개의 데이터셋과 5개의 비교 기준을 포함하여 종합적인 모달 시나리오에서 효과적으로 테스트되었습니다. FL을 적용한 MLLMs의 성능은 이전의 로컬 훈련 방식에 비해 유의미한 개선을 나타냈습니다.



### Cross Group Attention and Group-wise Rolling for Multimodal Medical Image Synthesis (https://arxiv.org/abs/2411.14684)
- **What's New**: 본 논문에서는 멀티모달 MR 이미지 합성을 위한 Adaptive Group-wise Interaction Network (AGI-Net)를 제안합니다. AGI-Net은 모달리티 간 및 내부 모달리티 간의 관계를 탐구하여 공간적 불일치 문제를 해결하며, 여러 입력 이미지들로부터의 정보를 효과적으로 융합합니다. 이를 통해 기존 방법들에 비해 성능 개선을 이루었고, IXI 및 BraTS2023 데이터셋에서 최첨단의 성능을 달성했습니다.

- **Technical Details**: AGI-Net의 핵심 설계는 Cross Group Attention과 Group-wise Rolling (CAGR) 모듈로 이루어져 있습니다. Cross Group Attention은 그룹 내 및 그룹 간의 관계를 설정하여 입력 특징에서 모달리티 간의 혼합 노이즈를 억제합니다. Group-wise Rolling은 각 그룹에 대한 컨볼루션 커널의 독립적인 적응형 롤링을 가능하게 하여, 부분적인 불일치 조건 하에서도 효과적으로 모달리티 간의 관계를 캡처합니다.

- **Performance Highlights**: AGI-Net은 IXI 및 BraTS2023 데이터셋을 사용해 평가되었으며, 기존의 프레임워크에 롤링 커널 모듈을 교체함으로써 새로운 최첨단 결과를 달성했습니다. 이러한 효과는 AGI-Net이 입력 모달리티 간의 공간적 관계를 잘 포착하고 통합할 수 있도록 돕기 때문에 가능합니다. 또한, AGI-Net은 플러그 앤 플레이 형태로 기존 컨볼루션 레이어를 대체할 수 있어 다양한 응용 가능성을 가지고 있습니다.



### BrightVAE: Luminosity Enhancement in Underexposed Endoscopic Images (https://arxiv.org/abs/2411.14663)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 논문에서는 저조도(endoscopic image) 내시경 이미지의 밝기를 향상시키기 위해 특별히 설계된 BrightVAE라는 아키텍처를 제안합니다. 기존의 내시경 이미지는 조명 불균형으로 인해 진단 정확도와 치료 계획에 부정적인 영향을 미칠 수 있습니다. BrightVAE는 이러한 문제를 해결하기 위해 계층적 벡터 양자화 변분 오토인코더(hierarchical VQ-VAE)를 기반으로 합니다.

- **Technical Details**: BrightVAE 아키텍처는 조명 변화와 세부 사항 숨김 등 내시경 영상에서의 고유한 도전을 해결하도록 면밀히 설계되었습니다. 이 모델은 다양한 수용 영역(receptive fields), 스킵 연결(skip connections) 및 특징 주의(feature attention)를 통합하여 세 가지 관점에서 고급 특징 추출을 강조합니다. 이러한 방식은 이미지 품질을 강력하게 향상시키고 더 정확한 의료 진단을 지원합니다.

- **Performance Highlights**: BrightVAE의 성능은 Endo4IE 데이터셋을 사용하여 평가되었으며, SSIM, PSNR, LPIPS와 같은 세 가지 널리 인정된 메트릭스(metrics)를 활용하여 성능이 검증되었습니다. 실험 결과, 기존 최첨단 방법들에 비해 내시경 영상에서의 밝기 향상에서 상당한 진전을 보였습니다.



### Evaluating Representational Similarity Measures from the Lens of Functional Correspondenc (https://arxiv.org/abs/2411.14633)
- **What's New**: 이 연구는 신경과학(neuroscience)과 인공지능(AI) 분야에서 고차원 신경 데이터(high-dimensional neural data)를 비교하는 데 중요한 도구를 찾고자 하며, 다양한 측정(metric)을 평가하여 특정 행동(behavioral outcomes)과의 일치도를 분석합니다. 특히, 제안된 방법론을 통해 훈련된 모델과 훈련되지 않은 모델을 효과적으로 구분할 수 있는지 알아봅니다. 또한, 여러 대표적 유사성 측정치가 행동 평가와 얼만큼 잘 일치하는지를 분석하여 신경 AI 분야에서의 지침을 제공합니다.

- **Technical Details**: 연구자는 여러 유사성 측정 방안을 비교하고, 정렬 기반, Canonical Correlation Analysis (CCA) 등 다양한 방법론을 통해 내부 표현(internal representations) 사이의 차이를 분석합니다. 이 과정에서 linear Centered Kernel Alignment (CKA)와 Procrustes 거리 같은 측정 방식이 훈련된 모델과 훈련되지 않은 모델을 구별하는 데 유능하다는 것을 발견하였습니다. 반면, linear predictivity는 신경과학에서 사용되는 방식 중 하나지만 행동과의 일치도가 낮은 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 각각의 측정 방식이 모델 구분에 있어 다르게 작용하며, 행동 메트릭(behavioral metrics)과의 일치도가 전반적으로 더 높은 경향을 보였습니다. 특히 linear CKA와 Procrustes 거리가 행동 평가와 가장 잘 일치하는 것으로 나타났고, 이는 NeuroAI 연구에서 행동적으로 의미 있는 비교를 강조할 수 있는 기초 자료를 제공합니다. 이러한 통찰은 향후 신경 AI 연구에서 유용할 것입니다.



### Unveiling the Hidden: A Comprehensive Evaluation of Underwater Image Enhancement and Its Impact on Object Detection (https://arxiv.org/abs/2411.14626)
- **What's New**: 이 연구는 수중 이미징의 품질 개선을 목표로 하는 최신 이미지 향상 모델들을 검토하고, 이러한 모델들이 수중 물체 탐지 성능에 미치는 영향을 조사했습니다. 두 가지 주요 데이터셋(RUOD, CUPDD)을 이용해 향상된 이미지의 정성적 및 정량적 분석을 수행하였고, 향상된 이미지와 원본 이미지의 품질 분포를 비교하기 위한 Quality Index(Q-index)를 개발했습니다 또한 YOLO-NAS 탐지 모델의 성능을 비교하여 향상 전후의 탐지 성능 변화를 분석했습니다.

- **Technical Details**: 본 연구는 수중 이미지 품질 저하의 원인과 이를 개선하기 위한 향상 기법을 quantitatively 및 qualitatively 연결하여 살펴보았습니다. RUOD와 CUPDD의 대표적인 이미지 향상 모델을 활용해 실험을 진행하였고, YOLO-NAS 탐지 모델을 기반으로 탐지 성능을 분석했습니다. 이 과정에서 Reference-free metrics(예: UIQM, UCIQE 등)를 활용하여 이미지 품질을 평가했습니다.

- **Performance Highlights**: 연구 결과, 일반적으로 이미지 향상이 탐지 성능을 저하시키는 경향이 있지만, 특정 경우에는 탐지 성능이 향상되는 경과를 보였습니다. 탐지 성능을 저하시킨 향상된 이미지의 분석을 통해, 인간 주석자가 놓친 객체를 발견하는 등 향상이 긍정적 결과를 초래하는 사례도 밝혀졌습니다. 이 연구는 시각적 품질 및 물체 탐지 성능 간의 관계를 명확히 하여 향상 기법의 활용 가능성을 보여주었습니다.



### SegBook: A Simple Baseline and Cookbook for Volumetric Medical Image Segmentation (https://arxiv.org/abs/2411.14525)
- **What's New**: 이 연구에서는 종합적 평가를 위한 대규모 기준을 설정하고, CT로 사전 훈련된 모델의 다양한 다운스트림 의료 분할 작업으로의 전이를 조사했습니다. 총 87개의 공개 데이터 세트를 수집하고 STU-Net을 사용하여 다중 모델 크기로 데이터 세트와 목표 간의 전이 학습을 수행합니다. 이러한 전이 연구는 의료 분야에 중요한 정보들을 제공할 것으로 기대됩니다.

- **Technical Details**: 연구에서는 CT, MRI, PET 및 초음파(US)를 포함한 다양한 모달리티의 87개 공개 데이터 세트를 수집하였습니다. STU-Net은 경량형부터 초대형 구성으로 조정 가능한 아키텍처를 제공하며, 140억 개의 매개변수를 가진 STU-Net-H와 같은 다양한 모델 크기로 평가하였습니다. 데이터 전처리 및 특정 다운스트림 작업을 위한 증강은 nnU-Net의 자동 설정을 따릅니다.

- **Performance Highlights**: 실험 결과, 데이터 세트 크기와 모델 크기가 성능 전이에 미치는 영향이 다르게 나타났습니다. 특히, 소규모 및 대규모 데이터 세트에서의 성능 향상이 두드러진 반면, 중간 규모의 데이터 세트에서는 상대적으로 덜 향상되었습니다. CT 데이터로 사전 훈련된 모델은 다른 모달리티와 잘 전이 가능하며, 구조 탐지 및 병변 탐지에 대한 효과를 보여 주었습니다.



### Memory Backdoor Attacks on Neural Networks (https://arxiv.org/abs/2411.14516)
- **What's New**: 본 논문에서는 메모리 백도어 공격(memory backdoor attack)을 제안합니다. 기존 모델이 특정 훈련 샘플을 메모리해 두고, 인덱스 패턴으로 트리거될 때 이를 선택적으로 출력할 수 있는 방법입니다. 이 공격은 분류 작업과 같은 상충된 과제를 처리하면서 모델 성능을 유지하는 독특한 공격 방식으로 주목받고 있습니다. 또한, 이미지 분류기, 세분화 모델, 대형 언어 모델(LLM)에서 이 공격이 어떻게 이루어질 수 있는지를 설명하고 있습니다.

- **Technical Details**: 메모리 백도어 공격은 두 가지 주요 취약점을 결합하여 발생합니다. 첫째, 모델 훈련 시 조건을 부여하여 특정 패턴으로 예측을 변경하는 백도어 공격이 가능한 점입니다. 둘째, 인덱스를 사용하여 샘플을 선택적으로 재구성할 수 있습니다. 이러한 공격은 훈련 데이터가 손상된 환경에서 작동하며, 실제로 배포된 모델에서도 공격자가 메모리된 훈련 샘플을 체계적으로 추출할 수 있도록 합니다. 특히, 공격자는 훈련 데이터의 특정 정보를 크게 방해할 수 있습니다.

- **Performance Highlights**: Pixel Pirate라는 이름의 새로운 공격을 통해 다양한 모델 아키텍처에서 테스트한 결과, CIFAR-100과 VGGFace2에서 각각 9.2%, 8.77%의 소폭 감소만으로 5000개의 샘플을 훔칠 수 있었습니다. 이 외에도 의료 이미지 세분화 모델에서 4.2%의 미미한 성능 저하로 전체 데이터셋을 메모리할 수 있었습니다. 논문에서는 이러한 공격에 대한 대응 조치를 제안하였으며, 커뮤니티에서 이를 위한 더 나은 솔루션을 연구할 것을 권장하고 있습니다.



### Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection (https://arxiv.org/abs/2411.14515)
Comments:
          Under review

- **What's New**: 이 논문에서는 기존의 이분법적(anomalies in a binary manner) 접근 방식을 넘어 실제 응용에서의 심각도를 반영하는 다층적 이상 탐지(Multilevel Anomaly Detection, MAD)를 제안합니다. 이 새로운 설정에서 이상 점수(anomaly score)는 이상 현상의 심각도를 나타내며, 다양한 분야에서의 응용 가능성이 강조됩니다. 또한, 새로운 벤치마크인 MAD-Bench를 도입하여 모델의 이상 탐지 능력과 심각도 정렬 점수 부여의 효과성을 동시에 평가합니다.

- **Technical Details**: MAD 설정에서는 데이터 포인트가 L0에서 Ln까지의 심각도를 구분할 수 있습니다. 이로 인해 훈련 세트는 정상 데이터에만 제한되고, 테스트 세트는 여러 심각도 수준의 이상을 포함합니다. 이와 함께, 모델 성능 평가를 위해 기존 데이터셋을 다양한 도메인에서 다층적 이상 탐지 맥락으로 조정했으며, 다중모달 대형 언어 모델(Multimodal Large Language Model, MLLM) 기반의 새로운 기준을 포함했습니다.

- **Performance Highlights**: MAD-Bench를 통해 다양한 모델의 성능 분석을 실시하였고, 모델들이 심각도에 맞는 점수를 부여하는 능력을 평가했습니다. 이 분석은 모델의 이분법적(binary) 탐지와 다층적(multilevel) 탐지 성능 간의 상관관계 및 견고성을 조사합니다. 결과적으로, 이 연구는 실제적인 심각도 정렬을 위한 독창적인 AD 모델 개선 방안을 제시합니다.



### Multi-agent reinforcement learning strategy to maximize the lifetime of Wireless Rechargeab (https://arxiv.org/abs/2411.14496)
Comments:
          77 pages, Bachelor's thesis

- **What's New**: 이 논문은 큰 규모의 무선 재충전 센서 네트워크(WRSN)에서 네트워크 수명(network lifetime)을 최대화하고 목표 범위(target coverage) 및 연결성(connectivity)을 보장하기 위한 일반화된 충전 프레임워크를 제안합니다. 또한, 여러 모바일 충전기(Mobile Chargers, MC)가 충전 지점에서 여러 센서를 동시에 충전할 수 있도록 하는 멀티 포인트 충전 모델을 활용하여 충전 효율성을 향상시킵니다. 이 연구는 실시간 네트워크 정보를 기반으로 최적의 충전 위치를 탐지하며, MC의 협력을 증진시키는 분산 부분 관찰 반 마르코프 결정 프로세스(Decentralized Partially Observable Semi-Markov Decision Process, Dec POSMDP) 모델을 제안합니다.

- **Technical Details**: 이 논문은 Proximal Policy Optimization(PPO) 알고리즘을 기반으로 하는 비동기 다중 에이전트 강화 학습(Asynchronous Multi-Agent Reinforcement Learning, AMAPPO) 알고리즘을 통해 Dec POSMDP 모델을 해결하는 방법을 제시합니다. 이 모델은 모바일 충전기(MC)가 실시간 데이터에 접근하여 최적의 충전 위치를 감지하고, 네트워크 다양한 환경에 적용할 수 있는 강화 학습 알고리즘을 통해 효율적으로 작동하도록 설계되었습니다. 이러한 접근 방식은 각 MC가 독립적으로 최적의 경로를 결정하며, 전체 네트워크의 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안된 알고리즘은 WRSN에서의 충전 효율성을 강화하고, 네트워크의 지속적인 성능을 보장합니다. 이는 센서가 에너지를 지속적으로 공급받을 수 있도록 보장하여, 데이터 수집과 전송의 연속성을 유지합니다. 논문에서는 다양한 실험을 통해 이 모델이 네트워크의 전체적인 수명과 성능을 향상시키는 것을 확인했으며, 이는 무선 센서 네트워크의 다양한 실제 응용 분야에서 중요한 의의가 있습니다.



### StreetviewLLM: Extracting Geographic Information Using a Chain-of-Thought Multimodal Large Language Mod (https://arxiv.org/abs/2411.14476)
- **What's New**: 이번 논문에서는 지리적 예측에 있어 새로운 프레임워크인 StreetViewLLM을 제안합니다. 이 모델은 대규모 언어 모델과 chain-of-thought reasoning을 통합하여 비구조적이거나 다중 양식(multi-modal) 데이터를 효과적으로 처리합니다. StreetViewLLM은 스트리트 뷰 이미지와 지리적 좌표, 텍스트 데이터를 결합하여 지리적 예측의 정확성과 세분성을 향상시킵니다.

- **Technical Details**: StreetViewLLM은 retrieval-augmented generation 기술을 활용하여 도시 환경에 대한 세부 분석 및 지리적 정보 추출을 개선합니다. 이 프레임워크는 홍콩, 도쿄, 싱가포르, 로스앤젤레스, 뉴욕, 런던, 파리 등 7개 글로벌 도시에서 적용되어 효과를 입증했습니다. 이 모델은 인구 밀도, 의료 접근성, 정상화된 식생 지수(normalized difference vegetation index), 건물 높이, 불투수 표면(impervious surface) 등의 도시 지표를 예측하는 데 우수한 성능을 보여줍니다.

- **Performance Highlights**: StreetViewLLM은 기존 기준 모델들에 비해 지속적으로 더 나은 성능을 발휘하며, 예측 정확성이 향상되었습니다. 이 연구는 대규모 언어 모델을 도시 분석(urban analytics), 도시 계획의 의사 결정(decision-making), 인프라 관리(infrastructure management), 환경 모니터링 환경 분야에 통합할 수 있는 새로운 기회를 열어줍니다.



### Attention-guided Spectrogram Sequence Modeling with CNNs for Music Genre Classification (https://arxiv.org/abs/2411.14474)
Comments:
          6 pages, 7 figures, 17 References

- **What's New**: 본 연구에서는 음악 장르 분류를 위한 혁신적인 모델을 제안합니다. 이 모델은 Attention 기반(주의 메커니즘) Temporal Signature Modeling을 사용하여 음악 장르를 분류합니다. 우리의 접근은 Convolutional Neural Networks (CNN)와 multi-head attention을 통해 각각의 음악 조각에서 가장 중요한 순간을 포착하여 고유한 '서명'(signature)을 생성합니다.

- **Technical Details**: 제안된 방법은 CNN(Convolutional Neural Networks)과 주의 메커니즘을 통합하여 음악 장르를 분류합니다. 이 구조는 스펙트로그램(인식 그래픽 형태의 음악 데이터)에서 공간적 특징을 추출하고, 시간적 의존성을 캡처해 장르를 정의하는 중요한 '서명 순간'을 식별합니다. CNN은 각 스펙트로그램을 처리하여 특정 프레임의 특징을 추출하고, 이후 multi-head attention을 적용하여 중요한 시간적 세그먼트를 강조합니다.

- **Performance Highlights**: 모델은 6겹 교차 검증을 통해 훈련되었으며, 혼동 행렬(confusion matrix)을 사용하여 장르별 분류 정확성을 평가했습니다. 특히, 클래식과 재즈 장르는 높은 분류 정확도를 보였으며, 블루스와 컨트리 장르는 유사한 음악적 뿌리로 인해 혼란을 겪었습니다. PCA 기반 분석을 통해 장르 간의 관계를 시각화하며, 특정 장르가 다른 장르와 어떻게 관련되는지를 수학적으로 설명하는 흥미로운 통찰을 제공합니다.



### Towards Scalable Insect Monitoring: Ultra-Lightweight CNNs as On-Device Triggers for Insect Camera Traps (https://arxiv.org/abs/2411.14467)
- **What's New**: 이 연구에서는 소형, 빠르게 움직이는 절지동물(예: 곤충)을 탐지하기 위한 카메라 트랩의 새로운 접근 방식을 제안합니다. 전통적인 수동 적외선(PIR) 센서 대신, 저전력 하드웨어에서 작동하는 초경량 합성곱 신경망(Convolutional Neural Network, CNN)을 사용하여 지속적으로 캡처된 이미지 스트림에서 곤충을 감지합니다. 마지막으로, 이 연구는 높은 정확도(91.8%에서 96.4% AUC)와 함께 낮은 전력 소비(최대 300mW)로 신뢰성 있는 탐지를 가능하게 함으로써 기존 설계보다 더 긴 운영 시간을 제공합니다.

- **Technical Details**: 제안된 시스템은 모바일 장치에서 실행되는 CNN 모델을 활용하여 곤충 이미지를 배경에서 구별할 수 있도록 훈련되었습니다. 이미지 캡처와 트리거 사이의 지연이 제로에 가까워, 실시간으로 곤충을 탐지하는데 효과적입니다. 모델은 낮은 전력 소비로 동작할 수 있도록 설계되었으며, 이는 저비용의 마이크로컨트롤러 유닛에서 가능한 구조입니다. 시스템은 스파일 맵(saliency map)을 통해 훈련된 표현이 견고하다는 것을 보여주며, 배경 잡음에 대한 의존성이 낮습니다.

- **Performance Highlights**: 모델은 검증 데이터에서 91.8%에서 96.4% AUC의 높은 정확도를 달성했으며, 훈련 중 보지 못한 데이터에 대한 AUC도 87% 이상입니다. 높은 특이도(specificity)를 통해 잘못된 양성 이미지를 최소화하고, 높은 재현율(recall) 점수로 잘못된 음성 비율을 줄여 곤충 탐지를 극대화합니다. 전반적으로, 제안된 시스템은 비용, 효율성 그리고 곤충 모니터링의 범위를 혁신적으로 향상시킬 수 있는 잠재력을 가지고 있습니다.



New uploads on arXiv(cs.AI)

### Empowering Clients: Transformation of Design Processes Due to Generative AI (https://arxiv.org/abs/2411.15061)
- **What's New**: 본 연구는 Generative AI의 발전이 건축 디자인 프로세스에 미치는 영향에 대한 심층적인 분석을 제공합니다. 특히, 고객과의 상호작용을 통해 디자인을 생성하고 피드백을 제공하는 과정에서 AI의 역할 변화를 철저히 검토합니다. 이를 통해 AI가 디자인의 아이디어 개발 단계에서 고객이 적극적으로 참여할 수 있게 만든다는 점을 강조합니다.

- **Technical Details**: 본 연구에서는 인기 있는 범용 텍스트-투-이미지(text-to-image) 도구를 활용하여 디자인 생성 실험을 수행하고, 기존 디자인에 대한 피드백을 제공합니다. 참가자들은 AI와 협업하여 생성된 디자인의 실행 가능성을 평가하는 방향으로 건축가의 역할이 변화하게 됩니다. 이러한 접근법은 AI가 디자인을 생성하는 데는 한계가 있으며, NP-completeness와 같은 컴퓨터 과학의 기초 개념과 흥미로운 연결점을 형성합니다.

- **Performance Highlights**: AI는 디자인에 대한 유용한 피드백을 제공할 수 있지만, 혁신적 접근방식을 표준화된 디자인으로 변경하는 경향이 있어 창의성과 혁신성을 저해할 수 있습니다. 건축가들은 디자인 프로세스에서 AI의 저자권 증대에 따른 의미와 정체성의 상실에 대한 불확실성을 느끼고 있습니다. 이러한 점은 AI가 디자인 분야에서의 authorship(저자권) 문제를 심도 있게 탐구할 필요성을 제기합니다.



### mR$^2$AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA (https://arxiv.org/abs/2411.15041)
- **What's New**: 이 논문은 최신 Knowledge-based VQA(task) 문제를 해결하기 위해 새로운 프레임워크인 mR²AG를 제안합니다. 기존의 Multimodal Retrieval-Augmented Generation (mRAG) 방법들이 가지는 한계를 극복하고, MLLM(Multimodal Large Language Models)의 지식 범위를 확장하는 데 중점을 둡니다. mR²AG는  두 가지 리플렉션(Reflection) 작업을 통해 정보 검색을 보다 효과적으로 수행하도록 설계되었습니다.

- **Technical Details**: mR²AG는 Retrieval-Reflection과 Relevance-Reflection 두 가지 메커니즘을 도입하여 사용자의 쿼리를 구분하고, 유용한 정보를 효과적으로 찾아냅니다. 이 프레임워크는 MLLM의 성능을 해치지 않으면서도 추가적인 복잡성을 유도하지 않습니다. 또한, mR²AG Instruction-Tuning 데이터세트를 통해 사전 훈련된 MLLM에 쉽게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 논문에서는 mR²AG가 INFOSEEK 및 Encyclopedic-VQA 테스크에서 최신 기술(state-of-the-art) 방법들을 초월한다고 주장합니다. 특히, mR²AG는 LLaVA-v1.5-7B를 베이스로 하여 INFOSEEK 테스트 세트에서 이전 SOTA들보다 10.6%와 15.5%의 성능 향상을 기록했습니다. 따라서, mR²AG는 Visual-dependent 작업에서도 기존 MLLM의 뛰어난 성능을 유지하면서 Knowledge-based 질문에 대한 응답을 성공적으로 개선합니다.



### Learning Lifted STRIPS Models from Action Traces Alone: A Simple, General, and Scalable Solution (https://arxiv.org/abs/2411.14995)
Comments:
          submitted to ICAPS 2025

- **What's New**: 이 논문에서는 STRIPS(action model) 모델을 action traces 만으로 학습하는 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 기존의 LOCM(LOgical Causal Model) 시스템처럼 확장성이 뛰어나지만, SAT(Satisfiability) 접근법처럼 건전하고 완전성을 보장하는 방법입니다. 이 방법은 hidden domain이나 predicate의 수 및 arity에 대한 제한이 없으며, 효율적인 테스트를 통해 action pattern이 predicate에 미치는 영향을 평가합니다.

- **Technical Details**: 제안된 학습 방법은 action traces 또는 그래프를 이용해 predicate와 action pattern을 학습하며, 이는 전통적인 8-puzzle 같은 클래식 도메인에서 평가되었습니다. 이 방법은 action pattern 특성을 활용해 학습된 predicate를 제시할 뿐만 아니라, precondition과 static predicate로 쉽게 완성할 수 있는 도메인을 제공합니다. 이 새로운 방법은 학습의 효율성을 높이고, 표준적인 도메인에서 수백만 개의 상태 및 전이를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 이론적으로 및 실험적으로 검토되었으며, 특히 8-puzzle 같은 클래식 도메인을 통해 실제로 평가되었습니다. 수많은 상태와 전이를 포함하는 데이터셋을 사용해 구현된 학습된 표현은 작은 인스턴스뿐만 아니라 더 큰 인스턴스에서도 검증되었습니다. 이 연구는 STRIPS 모델 학습의 한계를 극복할 수 있는 가능성을 제시함으로써, AI 플래닝 분야에 기여할 방안을 모색합니다.



### Free Energy Projective Simulation (FEPS): Active inference with interpretability (https://arxiv.org/abs/2411.14991)
Comments:
          26 pages (including 5 pages appendix), 6 figures

- **What's New**: 이 논문은 Free Energy Principle (FEP)와 Active Inference (AIF) 기반의 에이전트를 해석 가능하게 구축하는 방법을 제안합니다. 기존의 딥 뉴럴 네트워크를 사용하지 않고 Free Energy Projective Simulation (FEPS)을 도입하여 에이전트의 행동을 설명하는 새로운 모델을 제안합니다. 이를 통해 에이전트는 내부 보상만을 사용하여 불완전한 환경을 인식하고, 예측 오류를 줄일 수 있는 기술을 도입합니다.

- **Technical Details**: FEPS 모델에서는 세계 모델과 행동 정책 모두에 Projective Simulation (PS)을 사용합니다. PS는 메모리를 방향 그래프로 구성하여 개별 노드가 의미 정보를 포함하도록 만들어 해석 가능성을 높입니다. 에이전트는 주어진 세계 모델을 바탕으로 내재적 동기를 가지고 행동을 선택하며, 목표 상태를 달성하기 위한 새로운 휴리스틱을 제시합니다.

- **Performance Highlights**: 제안된 FEPS 모델은 행동 생리학에서 영감을 받은 두 가지 강화 학습 환경에서 테스트되었습니다. 에이전트는 예측 정확도에 따라 관찰을 맥락화하고, 목표 관찰을 위한 최적 행동 정책을 유연하게 추론함으로써 두 환경의 모호성을 완전히 해결했습니다. 이 성과는 억제 된 상태 추정으로 인한 예측 오류를 효과적으로 줄인 결과입니다.



### Domain and Range Aware Synthetic Negatives Generation for Knowledge Graph Embedding Models (https://arxiv.org/abs/2411.14858)
Comments:
          Accepted at the Third Learning on Graphs Conference (LoG 2024)

- **What's New**: 이번 연구에서는 Knowledge Graph Embedding (KGE) 모델의 훈련에서 중요한 합성 네거티브 생성 방법을 탐구합니다. 실제로 KGE 모델은 긍정, 부정 진술을 구분하는 데 초점을 두고 있지만, KGs에 결여된 사실들이 불필요하게 부정으로 간주되지 않도록 하는 방법에 대한 필요성을 강조합니다. 연구팀은 기존의 네거티브 샘플링(Negative Sampling, NS) 전략을 개선하고, 도메인과 범위를 고려한 새로운 방법을 제안합니다. 이는 기존의 벤치마크 데이터셋에서 10% 이상, 더 큰 온톨로지 기반 데이터셋에서는 150% 이상의 MRR 향상을 보여줍니다.

- **Technical Details**: KGE 모델은 엔티티와 관계의 연속 표현을 학습하여 다양한 작업을 해결합니다. 연구자들은 ال기존 네거티브 샘플링 방법을 확장하여 클래스에 따른 엔티티 타입 정보를 활용하며, 반복적으로 동일한 엔티티를 샘플링하는 비효율성을 최소화하고 다각적인 샘플링을 수행하여 진짜 부정을 피하는 방법을 강조합니다. 이 연구에서 제안된 클래스 인식 방법은 억제된 네거티브 샘플을 생성하기 위한 강력한 기초를 제공합니다.

- **Performance Highlights**: 네거티브 샘플링의 개선이 KGE의 성능에 미치는 긍정적인 영향이 크다는 것을 입증했습니다. 특히, 벤치마크 데이터셋에서 10% MRR 향상은 물론, 생물학적 데이터셋에서는 150% 이상의 MRR 증가를 달성했습니다. 이러한 성과는 실제 애플리케이션에 대한 효율성과 확장성을 고려하여 보장됩니다.



### SRSA: A Cost-Efficient Strategy-Router Search Agent for Real-world Human-Machine Interactions (https://arxiv.org/abs/2411.14574)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 놀라운 능력으로 인해 LLM 기반 검색 에이전트에 대한 연구가 급증하고 있습니다. 본 논문에서는 'Strategy-Router Search Agent (SRSA)'라는 새로운 접근 방식을 제안하여 다양한 쿼리를 적절한 검색 전략으로 라우팅함으로써 고품질 결과를 보다 경제적으로 얻는 방법을 소개합니다. 이 연구는 이전의 사용자-챗봇 인터랙션 연구들과는 달리, 실질적인 인간-기계 대화를 고려하고 있습니다.

- **Technical Details**: SRSA는 세 가지 개별 검색 전략과 지능적인 라우팅 메커니즘을 활용하여, 각기 다른 쿼리에 가장 적합한 전략으로 자동으로 방향을 제시합니다. 또한, 'Contextual Query Enhancement Dataset (CQED)'라는 새로운 데이터 셋을 도입하여, 사용자와 챗봇 간의 복잡한 대화 시나리오를 시뮬레이션합니다. SRSA의 성능은 정보성, 완전성, 새로움, 실행 가능성을 포함한 네 가지 평가 측면에서 측정되었습니다.

- **Performance Highlights**: SRSA는 단일 라운드 검색 기능이 있는 기존 LLM과 ReAct 기반 에이전트의 성능을 능가하는 뛰어난 결과를 도출했습니다. 이를 통해 사용자의 복잡한 쿼리에 대해 보다 포괄적이고 유익한 응답을 생성할 수 있다는 성과를 확인하였습니다. 본 연구는 LLM의 답변 품질을 향상시키고 현실 세계의 상호작용에서의 중요한 격차를 해결하는 데 기여하고자 합니다.



### Associative Knowledge Graphs for Efficient Sequence Storage and Retrieva (https://arxiv.org/abs/2411.14480)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 고유한 연관 지식 그래프를 구축하기 위한 혁신적인 접근 방식을 제안합니다. 이 그래프는 개체의 겹치는 시퀀스를 표현함으로써 대규모 그래프 내에서 밀집된 클러스터를 형성하며, 개별 객체들은 여러 시퀀스에 포함되거나 동일한 시퀀스 내에서 반복될 수 있습니다. 이 시스템의 메모리 용량은 그래프의 크기 및 연결 밀도에 의해 결정되며, 비평적 밀도(critical density)는 오류 없는 시퀀스 재구성이 불가능한 지점을 나타냅니다.

- **Technical Details**: 논문에서 제안하는 구조적 연관 지식 그래프는 개체들 간의 관계를 시각적으로 나타내며, 복잡한 의사결정에 유용한 프레임워크를 제공합니다. 이 알고리즘은 시퀀스 내의 요소를 효과적으로 정렬하는 방법을 연구하며, 다양한 데이터셋에 대한 테스트 결과를 비교하여 보여줍니다. 또한, 구조적 접근 방식이 메모리 크기, 그래프 밀도 및 컨텍스트 요구사항을 유지하면서 시퀀스 저장 및 검색에 어떻게 효과적으로 적용될 수 있는지를 강조합니다.

- **Performance Highlights**: 제안된 접근 방식은 금융 거래에서의 이상 탐지나 과거 행동에 기반한 사용자 행동 예측 등 다양한 분야에서의 활용 가능성을 보여줍니다. 실험을 통해, 메모리 용량이 quadratically 증가하는 것을 확인하였으며, 적절한 컨텍스트와 노드 수를 기반으로 메모리 설정을 최적화할 수 있음을 입증하였습니다. 이 연구 결과는 기존의 지식 그래프 및 신경망 방법론과의 결합을 통해 보다 향상된 데이터 저장 및 검색 성능을 제공할 것입니다.



### Measuring Bullshit in the Language Games played by ChatGP (https://arxiv.org/abs/2411.15129)
- **What's New**: 이번 논문은 생성적 대형 언어 모델(Generative Large Language Models, LLMs)이 진리 값과 직접적인 연관 없이 텍스트를 생성하는 방식을 분석합니다. 저자들은 이 현상이 어떻게 발생했는지, 그리고 어떻게 분석할 수 있는지를 고찰하며, LLM 기반의 챗봇이 'bullshit의 언어 게임(Language Game of Bullshit)'에서 어떤 역할을 하는지를 제시합니다.

- **Technical Details**: 연구는 통계적 텍스트 분석(statistical text analysis)을 사용하여 1,000개의 과학 출처와 ChatGPT가 생성한 전형적인 사이비 과학 텍스트(pseudo-scientific text)를 대조하는 데이터셋을 기반으로 진행됩니다. 이후 저자들은 조지 오웰의 정치와 언어에 대한 비판 및 데이비드 그레이버의 사이비 직업(bullshit jobs)에 대한 묘사에서 같은 언어적 특징이 발견되는지 탐구합니다. 간단한 가설 검정(hypothesis-testing) 방법을 사용하여 ChatGPT의 인위적인 bullshit 언어와 자연 언어에서 관찰되는 정치적 및 직장 내 bullshit의 기능 간의 관계를 설명합니다.

- **Performance Highlights**: 이 연구에서는 ChatGPT의 생성적 언어 구조가 정책 및 직장 내 부조리에 걸쳐 어떻게 작용하는지를 통계 모델을 통해 신뢰성 있게 연결합니다. 이를 통해 생성적 언어 모델이 사회적 비효율성의 맥락에서도 유의미한 언어적 패턴을 드러낼 수 있다는 점을 보여줍니다.



### Health AI Developer Foundations (https://arxiv.org/abs/2411.15128)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문에서는 Health AI Developer Foundations (HAI-DEF)라는 새로운 의료 머신 러닝 모델 개발 스위트를 소개합니다. 이 스위트는 다양한 도메인에 최적화된 사전 훈련된 모델과 도구, 레시피를 포함하고 있으며, 의료 분야의 AI 개발을 가속화하도록 설계되었습니다. HAI-DEF는 레이블이 적은 데이터로도 사용 가능하며, 훈련 시간을 단축시키고 컴퓨팅 비용을 낮추는 데 도움을 줍니다.

- **Technical Details**: HAI-DEF는 다양하고 큰 데이터셋을 이용해 훈련된 여러 개별 모델로 구성됩니다. 예를 들어, CXR Foundation은 이미지 및 관련 정확한 정보를 활용해 CXRs를 표현하는데 사용되며, Path Foundation은 자기 지도 학습 방법을 통해 병리학적 이미지 패치를 훈련합니다. 각 모델은 특정 데이터 모달리티 및 용도에 맞춰 조정되었습니다.

- **Performance Highlights**: HAI-DEF의 모델들은 데이터 효율성 측면에서 우수한 성능을 보였으며, 이는 일반 모델과의 성능 벤치마킹을 통해 입증되었습니다. 예를 들어, CXR Foundation 모델은 다양한 작업에서 높은 성능을 기록했으며, Derm Foundation 모델은 419개의 피부 질환을 효과적으로 처리할 수 있음을 입증했습니다. 이러한 결과는 도메인 특정 모델이 요구되는 데이터와 기술적 전문성을 줄이면서도 기존 접근 방식보다 우수한 성능을 달성할 수 있음을 시사합니다.



### ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation (https://arxiv.org/abs/2411.15122)
- **What's New**: AI 기반 모델이 흉부 X선의 영상 판독 보고서를 자동으로 생성하는 가능성을 보여주고 있으나, 이를 객관적으로 평가할 수 있는 표준화된 벤치마크가 부족했다. 이 연구는 ReXrank라는 공개 리더보드와 챌린지를 소개하여 AI 영상 판독 보고서 생성 모델의 평가를 위한 체계적인 기준을 제공한다. ReXrank는 10,000개의 스터디로 이루어진 ReXGradient라는 대규모 데이터셋을 포함하여 세 가지 공개 데이터셋(MIMIC-CXR, IU-Xray, CheXpert Plus)을 활용하고 있다.

- **Technical Details**: ReXrank는 8개의 평가 지표를 사용하여 모델의 성능을 평가한다. 모델은 단순히 발견 사항을 생성하는 것과 발견 사항과 해석 모두를 제공하는 모델로 구분하여 평가된다. 다양한 데이터셋을 통합하여, 모델의 일반화 능력을 보다 깊이 이해할 수 있는 평가를 가능하게 한다.

- **Performance Highlights**: MedVersa 모델이 ReXGradient 및 MIMIC-CXR 데이터셋에서 최고 성능을 기록하며, 다양한 지표에서 GPT4V보다 우수한 성능을 보였다. 평가 지표의 분포를 분석한 결과, IU X-ray는 모든 모델에서 높은 성능을 보였지만, CheXpert Plus는 가장 높은 변동성과 낮은 성능을 나타냈다. ReXGradient 데이터셋은 매우 낮은 성능 변동성을 보여 모델의 강건성 평가에 유용함을 입증했다.



### VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinemen (https://arxiv.org/abs/2411.15115)
Comments:
          Project page: this https URL

- **What's New**: 최근 텍스트-비디오(T2V) 확산 모델들은 다양한 분야에서 뛰어난 생성 능력을 보여주고 있습니다. 하지만 이러한 모델들은 여러 객체와 속성이 포함된 복잡한 장면을 설명하는 텍스트 프롬프트와 정확하게 일치하는 비디오 생성에 어려움을 겪고 있습니다. 이 문제를 해결하기 위해 우리는 VideoRepair라는 새로운 모델 불가지론적 훈련 없는 비디오 정제 프레임워크를 도입합니다.

- **Technical Details**: VideoRepair는 미세한 텍스트-비디오 일치 문제를 자동으로 감지하고, 명확한 공간적 및 텍스트 피드백을 생성하여 T2V 확산 모델이 목표로 하는 부위에서 정제를 수행하도록 돕습니다. 이 프레임워크는 네 단계로 구성되어 있으며, 첫 번째 단계에서는 미세한 평가 질문을 생성하여 비디오를 평가하고, 두 번째 단계에서는 정제할 영역을 계획합니다. 세 번째 단계에서는 정확하게 생성된 영역을 세분화하고, 마지막 단계에서는 정제된 프롬프트에 따라 비디오를 재생성합니다.

- **Performance Highlights**: VideoRepair는 EvalCrafter와 T2V-CompBench라는 인기 있는 비디오 생성 벤치마크에서 최근 기준선보다 우수한 성능을 보입니다. 다양한 텍스트-비디오 정렬 메트릭에서도 상당한 개선을 보여주며, 정제 과정의 각 구성 요소에 대한 포괄적인 분석과 질적 사례를 제공합니다. 이 연구는 시각 생성 작업에서 자동 정제 프레임워크의 발전을 촉진할 것으로 기대됩니다.



### RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts (https://arxiv.org/abs/2411.15114)
- **What's New**: 이 논문에서는 AI의 연구 개발(R&D) 동력으로서 AI 에이전트의 자동화가 중요한 능력으로 강조되고 있습니다. RE-Bench(Research Engineering Benchmark, v1)라는 새로운 벤치를 제안하여 7개의 도전적인 ML 연구 엔지니어링 환경과 61명의 전문가의 8시간 동안의 시도에서 수집된 데이터를 소개합니다. 이 연구는 AI R&D 능력에 대한 평가가 부족한 현재 상황에서 중요한 비교 기준이 될 것입니다.

- **Technical Details**: 연구는 7개의 유일한 ML 최적화 문제를 포함한 평가 환경을 제공하며, 각 환경은 고유한 연속적 ML 최적화 문제를 다룹니다. 44명의 인간 연구자와 35회의 기본 Claude-3.5-Sonnet 에이전트의 실행 결과를 비교한 결과, 인간 연구자는 시간이 지남에 따라 일관되게 발전하는 반면, AI 에이전트는 초기 1시간 동안 빠르게 진행하지만 그 후 plateau 상태에 들어갔습니다. 이 데이터는 미래의 AI 시스템과 인간 전무가 간의 미세한 비교를 가능하게 합니다.

- **Performance Highlights**: AI 에이전트는 2시간의 시간 예산으로 주어진 경우 인간 전문가보다 4배 높은 점수를 기록했지만, 인간은 시간이 늘어남에 따라 더 나은 성과를 보였습니다. 8시간의 예산에서 인간 전문가가 가장 우수한 AI 에이전트의 점수를 근소하게 초과하여 32시간 동안의 시도에서 2배의 점수를 보였습니다. 연구는 AI 에이전트가 여러 ML 주제에 있어 뛰어난 전문성을 가지고 있으며, 인간보다 10배 이상 빠르게 솔루션을 생성하고 테스트할 수 있다는 것을 보여주고 있습니다.



### Efficient Pruning of Text-to-Image Models: Insights from Pruning Stable Diffusion (https://arxiv.org/abs/2411.15113)
- **What's New**: 본 연구는 리소스 제약이 있는 장치에서 텍스트-이미지 모델의 채택을 가로막는 주요 원인인 모델 크기의 문제를 해결하기 위해 Stable Diffusion 2의 후처리 가지치기(post-training pruning) 기법을 분석합니다. 이전에 다루어지지 않았던 다중 모드 생성 모델의 가지치기 기술을 다루면서 텍스트 컴포넌트와 이미지 생성 컴포넌트에 대한 가지치기 체계의 영향을 개별적으로 검토합니다. 이 연구는 기존 언어 모델 가지치기 경향과는 달리 텍스트-이미지 맥락에서 두드러진 발견을 보고합니다.

- **Technical Details**: 긴 모델 아키텍처의 경우, 모델 크기는 수십억 개의 파라미터를 포함하고 있으며, 이는 실질적인 연산 비용을 발생시킵니다. 본 논문에서는 텍스트 인코더와 확산 생성기를 개별적으로 가지치기하는 방법과 각 컴포넌트에 대한 최적의 희소성(sparsity) 수준을 모색합니다. 연구 결과에 따르면, Stable Diffusion 2의 경우 38.5%의 희소성을 달성하면서 품질 손실을 최소화할 수 있음을 발견했습니다.

- **Performance Highlights**: 가지치기를 통해 모델의 연산 요구 사항을 대폭 줄일 수 있으며, 특히 텍스트 인코더에 대해 47.5%까지, 확산 생성기에 대해서는 35%까지 최적의 가지치기 구성을 제안합니다. 연구에서 나타난 흥미로운 점으로는 특정 임계를 넘어서는 가지치기가 성능 급락을 야기할 수 있다는 것이며, 이는 특정 가중치들이 핵심 의미론적 정보를 담고 있음을 시사합니다. 이러한 발견은 텍스트-이미지 모델의 모델 압축, 상호 운용성 및 편향 식별에 대한 새로운 연구 방향을 열어줍니다.



### About Time: Advances, Challenges, and Outlooks of Action Understanding (https://arxiv.org/abs/2411.15106)
- **What's New**: 이번 논문에서는 비디오 액션 이해(action understanding) 분야의 최근 발전을 살펴보며, 단일 모드 및 다중 모드 작업을 포함한 다양한 작업들에 대한 전반적인 개요를 제공합니다. 이 조사는 현재 시스템의 성과를 기반으로 여러 가지 템포럴 스코프(temporal scopes)를 구분하여 액션 인식, 예측 및 예측 작업을 설명합니다. 또한, 현재의 한계를 극복하기 위한 미래 방향도 논의합니다.

- **Technical Details**: 액션 이해는 세 가지 주요 템포럴 스코프, 즉 전체 관찰에서의 인식 작업, 부분적으로 관찰된 진행 중인 행동에 대한 예측 작업, 그리고 관찰되지 않은 후속 행동 예측 작업으로 분류됩니다. 이러한 구분은 특정 행동 모델링 및 비디오 표현의 도전 과제를 식별하는 데 도움을 줍니다. 연구는 비디오에서의 행동 모델링 접근 방식을 시간의 흐름에 따라 다룹니다.

- **Performance Highlights**: 현재의 연구는 여러 도전 과제를 다루고 있으며, 이전의 리뷰들은 특정 측면에 중점을 두고 있습니다. 예를 들어, 성능의 변동성은 데이터세트 내의 다양한 인스턴스 분포 차이에 따라 달라질 수 있으며, 제한된 클래스 간 변동은 모델의 일반화에 영향을 미칩니다. 이러한 점에서, 저자는 비디오 행동 이해에 대한 포괄적 리뷰를 제공하며, 다양한 데이터세트와 벤치마크를 통해 향후 연구 방향을 제시합니다.



### XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (https://arxiv.org/abs/2411.15100)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)용으로 설계된 유연하고 효율적인 구조 생성 엔진인 XGrammar를 제안합니다. XGrammar는 context-free grammar(CFG)의 실행을 가속화하기 위해 어휘(vocabulary)를 context-independent tokens와 context-dependent tokens로 나누어 처리합니다. 이 접근 방식은 구조 생성의 오버헤드를 최소화하고 대형 언어 모델 추론 작업과 효과적으로 오버랩되도록 설계되었습니다.

- **Technical Details**: XGrammar는 byte-level pushdown automaton을 사용하여 CFG를 해석합니다. 이 설계는 각 문자 가장자리에 여러 바이트를 포함할 수 있도록 하여 비정규 토큰 경계를 처리하고 서브 UTF-8 문자가 포함된 토큰을 지원합니다. 또한, 적응형 토큰 마스크 캐시를 생성하여 대부분의 마스크를 미리 생성하고, persistent execution stack을 사용하여 context-dependent tokens를 효율적으로 처리함으로써 성능을 향상시킵니다.

- **Performance Highlights**: XGrammar는 기존의 방법보다 최대 100배의 성능 향상을 달성할 수 있습니다. Llama-3.1 모델에 통합된 XGrammar 기반 LLM 서빙 엔진은 H100 GPU에서 구조적 출력에 대해 최대 80배의 속도 향상을 이룰 수 있습니다. 이를 통해 구조 생성을 위한 거의 제로 오버헤드를 달성할 수 있으며, XGrammar는 오픈 소스로 제공될 예정입니다.



### OminiControl: Minimal and Universal Control for Diffusion Transformer (https://arxiv.org/abs/2411.15098)
- **What's New**: 이번 논문에서는 이미지 조건을 통합한 OminiControl이라는 매우 다재다능하고 매개변수 효율적인 프레임워크를 소개합니다. OminiControl은 사전 학습된 Diffusion Transformer(DiT) 모델을 활용하여 이미지 조건을 효과적으로 인코딩하고 처리할 수 있도록 설계되었습니다. 기존의 복잡한 추가 인코더 모듈에 의존하는 방법들과는 달리, OminiControl은 약 0.1%의 추가 매개변수만으로 이미지 조건을 통합하고, 다양한 이미지 조건 작업을 통합적으로 처리할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: OminiControl은 기존 DiT 아키텍처에 이미지 기반 제어를 통합하기 위한 매개변수 효율적인 접근 방식을 제안합니다. 이 방법에서는 모델의 기존 VAE 인코더를 활용하여 조건 이미지를 처리하며, 디노이징 네트워크의 잠재적인 노이즈와 함께 조건 이미지를 통합하여 멀티 모달 어텐션 상호작용을 촉진합니다. 이는 DiT의 트랜스포머 블록 전반에 걸쳐 효율적인 정보 교환과 제어 신호 전파를 가능하게 합니다.

- **Performance Highlights**: OminiControl을 적용한 DiT 기반의 접근 방식은 주제 기반 생성(subject-driven generation) 및 공간 정렬 조건(spatially-aligned conditions) 작업에서 기존의 UNet 기반 및 DiT 조정 모델보다 월등한 성능을 보여줍니다. 이 연구에서는 200,000개 이상의 다양한 이미지로 구성된 Subjects200K라는 고품질 데이터셋을 개발하여 공개하며, 주제 일관성 생성(task) 연구를 위한 귀중한 자원을 제공합니다.



### RED: Effective Trajectory Representation Learning with Comprehensive Information (https://arxiv.org/abs/2411.15096)
Comments:
          This paper is accepted by VLDB2025

- **What's New**: 이 논문에서는 Trajectory Representation Learning (TRL)의 정확도를 높이기 위해 self-supervised TRL 프레임워크인 RED를 제안합니다. RED는 Transformer 모델을 기반으로 하여, 경로의 핵심 정보를 유지하는 Road-aware masking 전략을 적용합니다. 이를 통해 다양한 종류의 경로 정보를 효과적으로 활용하며, 모델 입력을 준비하면서 공간-시간-사용자 정보를 결합하는 방법을 사용합니다.

- **Technical Details**: RED는 Dual-objective 학습을 통해 트레일리 역할을 수행하며, Transformer 인코더는 경로의 다음 세그먼트를 예측하고 디코더는 전체 경로를 복원합니다. 특별히, RED는 경로의 공간-시간 상관관계를 고려하여 Transformer의 주의 메커니즘을 수정합니다. 이러한 기술적 요소들은 길이와 무관하게 효율적인 계산이 가능하도록 도와줍니다.

- **Performance Highlights**: 실험 결과 RED는 9개의 최신 TRL 방법들과 비교했을 때 평균적으로 5% 이상의 정확도 향상을 보여주었습니다. 구체적으로 여행 시간 추정, 경로 분류, 경로 유사도 계산에서 각각 7.03%, 12.11%, 20.02%의 정확도 향상률을 기록했습니다. 이러한 결과는 RED가 효과적인 정보 활용 방식과 모델 설계를 적용했음을 입증합니다.



### Towards Speaker Identification with Minimal Dataset and Constrained Resources using 1D-Convolution Neural Network (https://arxiv.org/abs/2411.15082)
- **What's New**: 이 논문은 스피커 식별(speaker identification)을 위한 경량화된 1차원 합성곱 신경망(1D-CNN)을 제안합니다. 최소한의 데이터셋에서 작동하도록 설계된 이 시스템은 데이터 증강(data augmentation) 기법을 활용하여 배경 잡음(background noise)과 제한된 학습 샘플을 처리합니다. 우리의 접근 방식은 97.87%의 검증 정확도(validation accuracy)를 달성하였으며, 향후 대규모 데이터셋에 대한 테스트와 전이 학습(transfer learning) 방법 통합을 계획하고 있습니다.

- **Technical Details**: 스피커 식별을 위해 DNN(Deep Neural Networks) 사용에 대한 동향을 검토하며, 특히 CNN(Convolutional Neural Networks)을 기반으로 한 모델들이 음성 스펙트럼에서 음성 인식을 효과적으로 수행함을 강조합니다. 저자들은 1차원 합성곱 신경망(1D-CNN) 아키텍처를 사용하여 리소스 제약이 있는 환경에서도 효율적으로 음성 데이터를 처리할 수 있는 방법을 모색합니다. 전체 프로세스에는 데이터 수집, 전처리(preprocessing), 훈련 및 응용 과정이 포함됩니다.

- **Performance Highlights**: 본 연구의 제출된 모델은 약 80%의 데이터를 훈련용(training set)으로 사용하고, 20%를 검증용(test/validation set)으로 사용하여 모델의 성능을 평가했습니다. 훈련 과정에서 'Sparse Categorical Cross-Entropy' 손실 함수를 사용하여 유효성을 평가하고, 초기 모델에서 예측이 수행되었습니다. 최종적으로 예측된 스피커의 소속 확률을 바탕으로 정확한 스피커 식별이 가능함을 증명했습니다.



### Financial Risk Assessment via Long-term Payment Behavior Sequence Folding (https://arxiv.org/abs/2411.15056)
Comments:
          ICDM2024 long paper

- **What's New**: 이 논문에서는 온라인 결제 플랫폼에서 사용자 결제 행동의 장기적인 패턴을 모델링하기 위한 새로운 방법인 'Long-term Payment Behavior Sequence Folding' (LBSF)을 제안합니다. LBSF는 상점(Merchant) 데이터를 기반으로 결제 행동 시퀀스를 압축함으로써, 사용자의 금융 위험을 보다 정확하게 평가할 수 있도록 합니다. 기존의 단기 행동 모델링의 한계를 극복하고, 사용자 금융 프로필을 효율적으로 학습할 수 있는 기회를 제공합니다.

- **Technical Details**: LBSF 방법은 결제 행동 시퀀스를 상점 수준에서 그룹화하여, 결제에 대한 정보를 최대한 활용할 수 있도록 설계되었습니다. 시퀀스는 상점의 특성을 기준으로 압축되고, 결제의 설명, 시간 정보 및 금액을 포함하는 다중 필드 행동 인코딩 메커니즘에 의해 표현됩니다. 이 방법은 내부 행동 신호를 기반으로 하여 정보의 흐름을 효율적으로 관리하며, 이를 통해 장기적인 사용자 행동 데이터를 효과적으로 처리합니다.

- **Performance Highlights**: 대규모 실제 데이터셋을 기반으로 한 실험 결과, LBSF 방법이 사용자 금융 프로필을 더욱 정확하게 모델링하는 데 효과적임을 보여주었습니다. 장기적인 결제 행동을 기반으로 한 모델링은 높은 예측 정확도를 제공하며, 이는 온라인 금융 서비스의 리스크 관리 향상에 기여할 수 있습니다. 이러한 발견은 데이터 성장과 함께 지속적으로 발전할 가능성을 시사합니다.



### Enhancing Autonomous Driving Safety through World Model-Based Predictive Navigation and Adaptive Learning Algorithms for 5G Wireless Applications (https://arxiv.org/abs/2411.15042)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 논문에서는 자율주행(Auto Driving) 시스템의 안전성 보장을 위한 새로운 내비게이션 프레임워크인 Navigation Secure (NavSecure)를 소개합니다. NavSecure는 세계 모델(World Models)과 안전을 중심으로 한 의사 결정 능력을 통합하여, 변화무쌍한 환경에서도 자율적으로 경로를 선택할 수 있도록 합니다. 이 접근방식은 잠재적인 위협을 예측하고 더 안전한 경로를 설정하는 데 중점을 두어, 실세계에서의 시행착오(Trial-and-Error) 학습의 필요성을 크게 줄였습니다.

- **Technical Details**: NavSecure는 5G 통신 네트워크를 활용하여 실시간 데이터 교환을 향상시키며, 자율주행차가 지속적인 학습과 개발을 통해 새로운 도전에 적응하도록 합니다. 이 시스템은 시뮬레이션-현실 환경에서의 엄격한 실험을 통해 검증되어, 갑작스러운 장애물 회피와 같은 안전-critical 시나리오에서 뛰어난 성능을 보여주었습니다. 또한, NavSecure는 세계 모델의 예측 기능을 활용하여 정밀한 의사 결정을 지원합니다.

- **Performance Highlights**: 실험 결과, NavSecure는 충돌 예방(Collision Prevention) 및 위험 감소(Risk Reduction)와 같은 주요 안전 지표에서 우수한 성과를 나타내었으며, 기존의 엔드-투-엔드(End-to-End) 방법론을 초월하는 성능을 기록했습니다. NavSecure는 자율주행 시스템의 안전성을 개선할 뿐만 아니라, 복잡한 의사 결정이 요구되는 응용 분야에서 세계 모델의 가치도 증명합니다. 이 프레임워크는 실세계의 동적이고 불확실한 환경을 처리할 수 있는 더 강력하고 신뢰할 수 있는 자율주행 시스템 개발에 새로운 기준을 제시합니다.



### One to rule them all: natural language to bind communication, perception and action (https://arxiv.org/abs/2411.15033)
- **What's New**: 이번 연구는 인간-로봇 상호작용에서 로봇의 자율적인 작업 계획 능력을 향상시키기 위해 LLM(대규모 언어 모델)을 통합한 새로운 아키텍처를 제시합니다. 이 시스템은 자연어로 표현된 명령을 실행 가능한 로봇 동작으로 변환하고, 실시간 피드백을 기반으로 계획을 동적으로 업데이트합니다. 특히, 프레임워크 ReAct를 수정하여 사용자 요청을 해석하고 실행하는 데 LLM을 활용하는 점이 특징입니다.

- **Technical Details**: 아키텍처는 주로 두 가지 모듈, 즉 인식 모듈과 계획 모듈로 구성됩니다. 인식 모듈은 환경을 감지하고 해석하여 기하학적 및 의미 정보를 포함하는 방향 그래프 형태의 의미 맵을 생성합니다. 계획 모듈은 사용자 요청을 자연어로 이해하고, 적절한 행동을 계획하며, 동적 환경에서의 실행을 관리합니다.

- **Performance Highlights**: 제안된 시스템은 로봇이 환경의 변화를 효과적으로 인식하고 이에 따라 전략을 조정할 수 있도록 하는 피드백 루프를 통해 동적 환경에서의 적응성을 높입니다. LLM을 통한 자연어 이해와 인식 시스템의 통합은 로봇이 복잡한 시나리오에서 자율적으로 작업을 수행할 수 있는 지원을 제공합니다. 이 연구는 로봇의 작업 수행 능력을 극대화하고, 인간 사용자와의 원활한 협업을 가능하게 합니다.



### Time is on my sight: scene graph filtering for dynamic environment perception in an LLM-driven robo (https://arxiv.org/abs/2411.15027)
- **What's New**: 본 논문에서는 작업 환경에서 로봇이 인간과 원활하게 상호작용할 수 있도록 대화형 로봇 제어 아키텍처를 제안합니다. 이 아키텍처는 Large Language Models(LLM)을 활용하여 자연어 명령, 로봇 기술 표현, 실시간 동적 의미 맵을 통합함으로써 복잡한 동적 환경에서 유연하고 적응력 있는 로봇 행동을 가능하게 합니다. 특히, 로봇의 상태 표현을 동적으로 생성하고 지속적으로 업데이트하는 방법론에 중점을 두고 있습니다.

- **Technical Details**: 로봇 시스템은 Perception Module과 Planner Module의 두 가지 주요 구성 요소로 이루어져 있습니다. Perception Module은 RGB-D 센서 데이터를 사용하여 환경의 의미 장면 그래프를 생성 및 업데이트하며, 이를 통해 로봇이 자신의 환경을 보다 명확히 이해할 수 있도록 돕습니다. Planner Module은 Perception Module에서 제공된 정보를 바탕으로 사용자의 자연어 요청을 고수준 기술로 변환하고, 이를 구체적이고 실행 가능한 저수준 행동으로 변환하여 로봇이 특정 작업을 수행하도록 안내합니다.

- **Performance Highlights**: 이 아키텍처는 동적이고 복잡한 환경에서 인간-로봇 협업을 개선하는 데 집중하고 있습니다. 특히, PSGTR 모델을 통해 로봇의 실시간 인식 및 상태 추적이 이루어지며, 이를 바탕으로 LLM이 복잡한 질문이나 지시사항을 해석하고 실행 가능 계획으로 변환할 수 있습니다. 이와 같은 방식으로 아키텍처는 로봇의 적응성, 작업 효율성, 인간-로봇 협업의 질을 높였습니다.



### FTA generation using GenAI with an Autonomy sensor Usecas (https://arxiv.org/abs/2411.15007)
- **What's New**: 이 논문에서는 자율주행 차량의 Lidar 센서 고장을 염두에 두고 Fault Tree Analysis(FTA)를 개발하기 위한 Generative Artificial Intelligence(GenAI)의 적용 가능성을 탐구합니다. 다양한 오픈 소스 Large Language Models(LLM)를 살펴보고, 그 중 하나의 응답을 깊이 있게 분석합니다. 또한, 기존의 LLM을 Prompt Engineering을 통해 훈련하여 Autonomy 사용 사례에 맞춰 FTA를 수행할 수 있는 가능성을 보여주고자 합니다.

- **Technical Details**: 론문은 Functional Safety(FuSa)와 Safety Of The Intended Functionality(SOTIF)이라는 두 가지 안전 기준을 통해 자율주행 차량에 필요한 다층 안전 접근 방식을 제안합니다. GenAI의 다양한 기술, 예를 들어 Generative Adversarial Networks(GANs) 및 Transformer 기반 모델이 FTA 분석에 활용될 수 있음을 설명합니다. 또한, HAZOP(위험성과 운전 가능성 연구)와 FMEA(실패 모드 및 영향 분석) 방법을 통해 자율주행 안전 분석을 지원하는 방식을 논의합니다.

- **Performance Highlights**: FTA를 활용함으로써 초기 결함을 발견하고, 차량의 안전성을 향상시키며, 설계 및 제조 프로세스를 최적화하는 데 기여할 수 있습니다. 이 연구는 GenAI 기반의 기법이 자율주행 기술의 설계 개선에 중요한 역할을 할 수 있음을 보여주며, 이론적 접근에서 실제 응용 가능성으로 나아가는 지향점을 설정합니다. 마지막으로, GenAI 모델의 활용은 교육 및 학습 효율성을 높이는 데도 기여할 수 있다는 점에서 매우 흥미롭습니다.



### ScribeAgent: Towards Specialized Web Agents Using Production-Scale Workflow Data (https://arxiv.org/abs/2411.15004)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 에이전트들이 복잡한 웹 작업을 처리하는 방안을 제시합니다. 연구팀은 250개 이상의 도메인에서 수집한 60억 개의 토큰을 포함한 생산 규모의 워크플로우 데이터를 이용하여 오픈 소스 LLM을 미세조정하는 대안을 탐구하고 있습니다. 이를 통해 ScribeAgent라는 새로운 웹 에이전트를 개발하였으며, 기존 에이전트 대비 더 나은 성능을 입증했습니다.

- **Technical Details**: ScribeAgent는 HTML-DOM과 행동 이력을 기반으로 다음 행동을 직접 생성할 수 있는 첫 번째 단일 단계 LLM 에이전트입니다. 이는 이전의 다단계 에이전트와는 달리, 여러 단계를 거치지 않고도 작업을 수행할 수 있도록 설계되었습니다. 연구팀은 LoRA (Low-Rank Adaptation) 기법을 사용하여 오픈 소스 LLM을 미세 조정하였으며, 이 과정에서 60억 개 이상의 토큰으로 구성된 훈련 데이터를 활용했습니다.

- **Performance Highlights**: ScribeAgent는 Mind2Web 벤치마크에서 최첨단의 직접 생성 성능을 달성하고, WebArena에서는 작업 성공률을 37.2%에서 51.3%로 증가시켜 최고 성과를 나타냅니다. 특히, 32B 매개변수를 가진 ScribeAgent-Large는 모든 테스트 세트에서 기본 성능보다 5-10% 향상된 단계 성공률을 보였습니다. 이러한 결과는 LLM의 성능을 높이기 위한 대규모 고품질 실세계 데이터의 중요성을 강조합니다.



### Exploring Foundation Models Fine-Tuning for Cytology Classification (https://arxiv.org/abs/2411.14975)
Comments:
          5 pages, 2 figures

- **What's New**: 이 논문에서는 세포학 슬라이드 분석의 효율성을 향상시키기 위한 기초 모델의 활용 가능성을 모색합니다. 특히, 적은 샘플로 학습할 수 있는 Low-Rank Adaptation (LoRA) 기법을 통해 기존의 기초 모델을 미세 조정하는 방법을 효과적으로 제안하고 있습니다. 이를 통해 단순 분류 및 복잡한 분류 작업에서 최첨단 성능을 달성하였고, 더 적은 데이터 샘플을 사용하면서도 성능을 향상시킬 수 있음을 입증하였습니다.

- **Technical Details**: 연구에서는 세포학 분류를 위해 유명한 기초 모델 다섯 개를 평가하였습니다. LoRA를 사용해 미세 조정한 모델이 이전의 방식에 비해 성능이 높아졌다고 보고하고 있습니다. 또한, 세포학 작업에 특화된 모델이 일반 모델보다 더 탁월한 적응성을 보여준다고 밝혔습니다.

- **Performance Highlights**: 모델의 성능 평가에서는 세포학 분류의 네 가지 벤치마크 데이터셋에서 뛰어난 성과를 얻었습니다. CLIP의 비전 인코더는 LoRA로 미세 조정 후 단일 데이터셋의 70%만으로도 최첨단 성능을 달성하여, 현재의 최선단 모델보다 훨씬 적은 훈련 가능한 매개변수를 사용하였습니다. 이를 통해 LoRA의 효과가 다양한 의학적 작업에서 강력하게 입증되었습니다.



### Open-Amp: Synthetic Data Framework for Audio Effect Foundation Models (https://arxiv.org/abs/2411.14972)
- **What's New**: 본 논문에서는 Open-Amp라는 합성 데이터 프레임워크를 소개하며, 이를 통해 대규모의 다양한 오디오 이펙트 데이터를 생성할 수 있습니다. 기존의 오디오 이펙트 데이터 세트는 범위가 제한적이며, 일반적으로 적은 수의 프로세서와 입력 신호만을 포함합니다. Open-Amp는 오픈 소스 소프트웨어 사용자가 생성한 신경망의 에뮬레이션을 활용하여 이러한 문제를 해결하며, 사용자가 원하는 신호를 처리할 수 있게 합니다. 또한 고품질 에뮬레이션을 통해 수백 가지 장치에 대한 훈련 데이터를 생성할 수 있습니다.

- **Technical Details**: Open-Amp는 Python 패키지로서 온라인에서 다양한 기타 앰프와 왜곡 이펙트 모델을 통해 오디오를 실시간으로 증강할 수 있는 기능을 제공합니다. 이 패키지는 사용자가 선택한 입력 신호에 대한 다양한 오디오 이펙트를 적용하여 데이터 증강을 가능하게 합니다. 논문에서는 Open-Amp를 사용하여 기타 이펙트 인코더를 훈련시키고, 학습된 특징 공간을 통해 전이 가능성을 보여주었습니다. 또한, 모델 아키텍처는 단일 레이어 LSTM을 기반으로 하며, 각 블록에는 두 개의 합성곱 레이어와 잔차 연결이 포함되어 있습니다.

- **Performance Highlights**: Open-Amp를 사용하여 훈련된 기타 이펙트 인코더는 여러 기타 이펙트 분류 작업에서 새로운 최첨단 결과를 달성했습니다. 또한, Open-Amp를 통해 훈련된 1대 다 모델을 사용하여 학습된 잠재 공간을 통해 새로운 아날로그 이펙트를 에뮬레이트할 수 있음을 보여줍니다. 이 연구는 오디오 이펙트 기반 모델을 구축하기 위한 대규모 및 다양한 데이터가 필요함을 강조하며, Open-Amp가 이러한 요구를 충족할 수 있는 유연하고 효율적인 방법임을 입증합니다.



### SwissADT: An Audio Description Translation System for Swiss Languages (https://arxiv.org/abs/2411.14967)
- **What's New**: 이번 연구에서는 SwissADT라는 다국어 및 다중 모달 오디오 설명 번역(ADT) 시스템을 소개합니다. 이는 독일어, 프랑스어, 이탈리아어 및 영어를 지원하며, 시각 장애인 및 시각적 제한이 있는 사용자들이 접근할 수 있도록 설계되었습니다. SwissADT는 비디오 클립을 활용하여 시각적 정보를 포함한 AD 스크립트를 자동으로 번역함으로써 다국어 인구에 대한 정보 접근성을 향상시키고자 합니다.

- **Technical Details**: SwissADT는 비디오 클립과 텍스트 입력을 결합하여 AD 스크립트를 번역하는 LLM 기반 시스템입니다. 이 시스템은 AD 스크립트에 가장 적합한 비디오 순간을 식별하기 위해 CG-DETR라는 비디오 시간 그라우더를 사용합니다. 최종적으로 AD 번역기를 위해 비디오 프레임을 샘플링하고, GPT-4 모델을 활용하여 다국어 번역을 수행합니다.

- **Performance Highlights**: SwissADT의 성능은 자동 및 인간 평가를 통해 검증되었습니다. 실험 결과, SwissADT는 ADT 작업에서 유망한 성능을 보여주었으며, AD 전문가의 의견을 토대로 시각 정보를 입력에 포함하는 것이 결과에 긍정적인 영향을 미친 것으로 나타났습니다. 이 연구는 향후 다국어 ADT 시스템의 발전 방향에 중요한 기초 자료를 제공합니다.



### LLM for Barcodes: Generating Diverse Synthetic Data for Identity Documents (https://arxiv.org/abs/2411.14962)
Comments:
          5 pages, 1 figures

- **What's New**: 이 논문에서는 기존의 데이터 생성 방식의 한계를 극복하기 위해 LLM(대형 언어 모델)을 활용한 새로운 합성 데이터 생성 접근법을 제안합니다. 기존의 Faker와 같은 도구는 미리 정의된 템플릿에 의존하여 실제 문서의 다양성과 복잡성을 포착하지 못하는 문제를 보였습니다. 이 방법을 통해 다양한 홍보 요소와 문화적 차이를 반영한 문서 데이터를 생성함으로써 보다 현실적인 데이터셋을 구축할 수 있게 됩니다. 또한, 민감한 개인 정보는 포함되지 않으므로 개인정보 보호에도 적합한 솔루션을 제공하게 됩니다.

- **Technical Details**: 우리의 접근법은 LLM을 사용하여 Driver’s Licenses, Insurance Cards, 및 University IDs와 같은 다양한 신원 문서에 필요한 데이터를 생성하는 과정을 포함합니다. 각 문서 유형은 그 목적에 맞는 특정 정보를 포함하도록 설계되었습니다. 예를 들어, Driver’s Licenses는 소지자의 이름, 주소, 라이센스 번호와 같은 세부정보를 포함하고, Insurance Cards는 정책 번호와 보장 일자를 포함해 보험사에 따라 다르게 나타납니다. 데이터 생성 과정에서 pyBarcode 라이브러리를 사용해 바코드를 인코딩하고, 문서 템플릿에 통합하는 단계를 거치며, 데이터 다양성을 높이기 위해 데이터 증강 기법도 적용합니다.

- **Performance Highlights**: LLM을 기반으로 생성된 데이터는 기존의 합성 데이터에 비해 다양성과 맥락적 관련성이 향상되어 바코드 탐지 모델의 성능 개선으로 이어지는 것으로 나타났습니다. 우리는 이 새로운 방식이 다양한 문서 형식과 바코드 유형에 강력한 성능을 발휘할 수 있음을 입증하였습니다. 이 방법은 문서 유형, 지역 또는 바코드 표준의 변경에 쉽게 적응할 수 있어 신원 문서 표준의 발전에 따라 진화할 수 있는 유연한 솔루션을 제공합니다.



### Design-o-meter: Towards Evaluating and Refining Graphic Designs (https://arxiv.org/abs/2411.14959)
Comments:
          Accepted to WACV 2025. Project page: this https URL

- **What's New**: 이번 논문에서는 그래픽 디자인의 품질을 정량화하고 개선하기 위한 새로운 방법론, Design-o-meter를 소개합니다. Design-o-meter는 디자인 평가 및 향상을 위한 통합된 프레임워크로, 주관적이고 모호한 디자인 특성을 고려합니다. 이 방법론은 Generative AI 기술을 사용하여 디자인 생성뿐만 아니라 평가와 개선도 수행합니다.

- **Technical Details**: Design-o-meter는 두 가지 주요 모듈로 구성되며, 하나는 디자인을 평가하는 scorer이고, 다른 하나는 평가 결과를 바탕으로 디자인을 개선하는 refiner입니다. scorer는 metric learning 모델로 구현되어 있으며, 디자인의 좋고 나쁨을 점수화합니다. refiner는 새로운 설계 특정 교차 운영인 SWAN을 사용하는 유전 알고리즘을 기반으로 디자인을 정제합니다.

- **Performance Highlights**: 이 연구는 Design-o-meter의 효과성을 정량적 및 정성적으로 분석하며, 기존의 다중 모달 LLM 기반 접근 방식들과 비교하여 우수한 성능을 보입니다. 특히, 디자인 문서의 미세한 뉘앙스를 포착할 수 있는 scorer의 동작과 복잡한 디자인 공간을 효과적으로 탐색하는 refiner의 성능을 강조합니다. 종합적으로, Design-o-meter는 디자인 품질 측정 및 개선의 혁신적인 방법을 제공할 것으로 기대됩니다.



### Evaluating Vision Transformer Models for Visual Quality Control in Industrial Manufacturing (https://arxiv.org/abs/2411.14953)
- **What's New**: 이번 연구는 산업 제조 분야의 시각 품질 관리 시스템에서 사용될 수 있는 최신의 비전 트랜스포머 모델과 이상 탐지 방법들을 종합적으로 검토하고 평가합니다. 다양한 트랜스포머 아키텍처와 이상 탐지 기법을 조합하여 소형, 빠르고 효율적인 모델을 개발하여 실용적인 품질 관리 시스템 구축에 도움을 제공합니다. 이를 통해 품질 관리 시스템에서 필요한 모델 구조 선택에 대한 가이드라인도 제시합니다.

- **Technical Details**: 연구에서는 Unsusupervised anomaly detection (AD) 및 anomaly localization (AL) 기법을 활용하여 드문 결함 품목을 식별하는 방법론을 다룹니다. 일반적으로 이상 탐지를 위해 이미지 인코딩 백본과 탐지 알고리즘이 조합되어 사용됩니다. 최신 비전 트랜스포머 모델은 고해상도 이미지 처리 시 발생하는 부족한 스케일링 문제를 해결하며, 전통적인 CNN 아키텍처에 비해 더 나은 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 실험 결과, MVTecAD 및 BTAD 데이터셋을 이용한 평가에서 최적의 모델 조합을 찾는데 성공하였습니다. 특히, 연구에서 검토한 계층적 비전 트랜스포머 모델들이 산업적 품질 관리에서 필요한 시간 및 자원 절감과 더불어 탁월한 성능을 보였습니다. 이 연구는 예측 정확성을 높이고 수동 검사를 줄임으로써 제조 산업에서 큰 이점을 제공할 수 있는 가능성을 보여줍니다.



### Reliable Evaluation of Attribution Maps in CNNs: A Perturbation-Based Approach (https://arxiv.org/abs/2411.14946)
- **What's New**: 이번 연구는 CNN(Convolutional Neural Network)의 Attribution Map(AM, 기여도 맵) 평가에 대한 새로운 접근 방식을 제시합니다. 기존의 삽입/삭제 메트릭이 분포 이동(Distribution Shift)에 민감하다는 점을 강조하며, 이를 해결하기 위해 픽셀 수정을 적대적 섭동(Adversarial Perturbations)으로 대체하는 방법을 제안합니다. 이 연구는 AM의 신뢰성과 일관성을 높이는 사람의 이해가 가능한 평가 프레임워크를 제공합니다.

- **Technical Details**: AM은 이미지 입력 데이터의 각 이미지에 대한 기여를 시각적으로 표현하여 신경망의 결정 과정을 설명하는 도구입니다. 그러나 AM의 정량적 평가 및 올바른 성능 메트릭 정의는 여전히 난제입니다. 본 연구는 다양한 메트릭을 활용해 15개 데이터셋 및 아키텍처 조합에 걸쳐총 16개의 AM을 평가하였으며, Kendall의 순위 상관 계수(τ)를 통해 신뢰성을 테스트하였습니다.

- **Performance Highlights**: 본 연구의 결과, SmoothGrad는 16개의 AM 중에서 최상의 성능을 보였으며, 이는 AM의 질적 및 양적 평가를 통합한 가장 포괄적인 분석을 기반으로 합니다. 또한, 기초 기준 AM을 도입하여 평가의 신뢰성을 높이는 방법을 제시하였으며, 코드와 결과를 함께 제공하여 재현성을 보장합니다.



### Comparative Study of Neural Network Methods for Solving Topological Solitons (https://arxiv.org/abs/2411.14942)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 비선형 미분 방정식의 안정적이고 국소화된 해인 topological solitons을 효율적으로 해결하기 위한 새로운 신경망(neural network) 방법을 개발하였습니다. 기존의 Physics-Informed Neural Networks(PINN)와 비교하여 개발한 모델이 동일한 정확도를 유지하면서도 훨씬 더 짧은 계산 시간을 기록하는 것으로 나타났습니다. 이러한 계산 효율성의 혁신은 현재의 한계를 극복할 뿐만 아니라 topological solitons과 그 동적 행동을 연구하는 새로운 가능성을 열어줍니다.

- **Technical Details**: 우리의 NN 방법은 Neural Network for Difference Equation(NNDE)라는 이름으로 불리며, 1차원 공간 좌표에서 미분 방정식을 해결하는 것을 목표로 합니다. 이 방법은 경계 조건과 함께 작동하며, 일반적으로 높은 차원의 문제와 여러 함수에 대해서도 적용이 가능합니다. NNDE에서는 선형 회귀의 기본 원리를 바탕으로 하고, 미분 계산에는 차분법(difference method)을 사용하여 함수의 도함수를 근사합니다.

- **Performance Highlights**: NNDE는 TensorFlow를 이용해 구현되었으며, 다양한 비선형 장 이론(non-linear field theories)을 통해 성능을 비교하였습니다. 결과적으로, NNDE는 PINN보다 계산 속도를 상당히 개선하면서도 유사한 수준의 정확도를 유지하는 것으로 나타났습니다. 이러한 효율성과 정확도의 개선은 future 연구에서 topological solitons의 심도 깊은 분석을 가능케 하며, 더 넓은 실제 적용 가능성을 시사합니다.



### Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning (https://arxiv.org/abs/2411.14937)
- **What's New**: 이 논문은 비전-언어 모델(vision-language models, VLMs)을 활용하여 연합 학습(federated learning, FL)에서의 기울기 역전 공격(gradient inversion attacks, GIAs)의 한계를 극복하는 Geminio라는 새로운 방법을 소개합니다. 이 모델은 공격자가 자연어로 원하는 데이터 샘플을 서술하도록 허용하여, 해당 샘플의 재구성을 우선시할 수 있는 혁신적인 프라이버시 공격 경험을 제공합니다. 기존의 GIAs가 직면한 문제를 해결하는 데 중점을 두어, 공격자가 타겟 데이터를 보다 유연하고 의미 있는 방식으로 명시할 수 있게 합니다.

- **Technical Details**: Geminio는 미리 학습된 VLM을 활용하여 악의적인 전역 모델의 최적화를 안내합니다. 공격자는 특정 데이터 샘플을 찾기 위한 쿼리를 제출할 수 있으며, 이러한 쿼리를 통해 FL 서버는 해당하는 샘플의 기울기를 재구성하여 높은 품질의 타겟 데이터를 복원할 수 있게 됩니다. 이 방식은 기존의 최적화 알고리즘과 결합되어 강력한 기울기 재구성을 가능하게 합니다.

- **Performance Highlights**: 상 extensive한 실험을 통해 Geminio의 효과가 입증되었습니다. 복잡한 데이터셋과 다양한 공격 방법 및 방어 메커니즘을 사용하여 타겟 샘플의 추적 및 재구성에서 높은 성공률을 보여주었으며, 기존 방어에 대해서도 강한 저항력을 나타냈습니다. 이러한 결과는 Geminio가 향후 연구 및 실제 응용에 있어 중대한 위협이 될 수 있음을 의미합니다.



### LiDAR-based End-to-end Temporal Perception for Vehicle-Infrastructure Cooperation (https://arxiv.org/abs/2411.14927)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 LiDAR 기반의 End-to-End Tracking 프레임워크인 LET-VIC를 소개합니다. LET-VIC는 차량과 인프라 간의 협력을 통해 동적 환경에서의 시간적 인식을 개선하도록 설계되었습니다. 이 프레임워크는 V2X 통신을 활용하여 차량과 인프라 센서 데이터를 융합하고, 이러한 통합을 통해 가려진 객체와 관측 블라인드 스팟 문제를 해결합니다.

- **Technical Details**: LET-VIC는 차량 측과 인프라 측의 LiDAR 데이터를 Bird's Eye View (BEV) 형식으로 통합하여 종합적인 시각을 제공합니다. 이 시스템은 또한 프레임 간의 시간적 맥락을 통합하여 역사적 데이터를 활용하여 추적의 안정성과 정확성을 높입니다. 특히, Calibration Error Compensation (CEC) 모듈을 도입하여 센서 간 불일치를 보정하여 정밀한 특징 정렬을 보장합니다.

- **Performance Highlights**: 실험 결과, LET-VIC는 V2X-Seq-SPD 데이터셋에서 기존의 기본 모델보다 최소 13.7%의 mAP 개선과 13.1%의 AMOTA 개선을 달성하였습니다. LET-VIC는 통신 지연을 고려하지 않더라도 강력한 안정성과 적응력을 보여 주며, 다양한 지연 조건에서도 일관된 성능을 유지합니다. 이러한 성과는 LET-VIC가 고도화된 자율 주행 시스템의 잠재력을 보여줍니다.



### Purrfessor: A Fine-tuned Multimodal LLaVA Diet Health Chatbo (https://arxiv.org/abs/2411.14925)
Comments:
          10 pages, 5 figures

- **What's New**: 본 연구는 개인화된 식단 안내를 제공하는 혁신적인 AI 챗봇인 Purrfessor를 소개합니다. Purrfessor는 Large Language-and-Vision Assistant (LLaVA) 모델을 기반으로 하여 다중 모드 접근을 통해 사용자 경험과 상호작용을 향상시키는 기능을 갖추고 있습니다. 이를 통해 시각적인 식사 분석과 맥락에 맞는 조언을 제공하여 보다 효과적인 사용자 참여를 도모하고자 합니다.

- **Technical Details**: Purrfessor 시스템은 클라우드 기반 리소스와 구조화된 데이터베이스를 활용하여 사용자와 상호작용할 수 있는 고급 대화형 AI 모델로 설계되었습니다. 사용자 인터페이스는 웹 페이지를 통해 접근 가능하며, 사용자 계정 기능을 통해 개인화된 상호작용과 추천사항을 제공합니다. Node.js와 MongoDB를 사용하여 데이터 흐름을 조정하고 사용자 대화를 기록하며, fine-tuned LLaVA 모델은 효율적인 이미지 처리와 성능 확장을 위해 클라우드 서버에 배포됩니다.

- **Performance Highlights**: Purrfessor는 사용자의 돌봄에 대한 인식과 관심을 유의미하게 향상시키며, 이러한 개선은 사용자 인터뷰를 통해 상호작용 디자인의 중요성이 강조되었습니다. 두 가지 연구를 통해 챗봇의 성능과 사용자 경험을 평가하였으며, 이는 AI 건강 보조 도구가 어떻게 개인의 식습관 개선에 기여할 수 있는지를 제시합니다. 특히, 저소득층이나 역사적으로 소외된 그룹을 위한 맞춤형 솔루션을 제공하여 건강한 식단 계획을 지원하는 데 집중하고 있습니다.



### GOT4Rec: Graph of Thoughts for Sequential Recommendation (https://arxiv.org/abs/2411.14922)
- **What's New**: 이 논문에서는 GOT4Rec이라는 새로운 순차 추천 방법을 제안합니다. 이 방법은 'graph of thoughts' (GoT) 프롬프트 전략을 활용하여 사용자의 행동 시퀀스에서 사용자 관심 정보를 포괄적으로 수집합니다. GOT4Rec는 단기 관심, 장기 관심 및 협업 정보를 효과적으로 결합하여 더욱 정확한 추천을 제공합니다.

- **Technical Details**: GOT4Rec는 사용자의 과거 상호작용 시퀀스를 기반으로 추천을 생성하며, 각 시퀀스에서 중요한 세 가지 유형의 정보를 추출합니다. 이러한 정보에는 단기 및 장기 관심, 그리고 유사한 선호를 가진 다른 사용자로부터의 협업 정보가 포함됩니다. 기존의 방법들과 달리, GOT4Rec는 사용자 시퀀스를 전체로 간주하는 대신 다층적인 정보 처리를 통해 LLM의 추론 능력을 극대화합니다.

- **Performance Highlights**: 실험 결과 GOT4Rec는 세 가지 데이터셋에서 전통적인 신경망 기반 모델들과 다른 프롬프트 전략들보다 뛰어난 성능을 보였습니다. GOT4Rec의 도입으로 LLM이 사용자 시퀀스 내 다양한 정보를 보다 효율적으로 활용할 수 있게 되어, 정확한 추천 및 포괄적인 설명이 가능해졌습니다. 이는 GOT4Rec가 기존의 최첨단 모델들과 비교하여 우수함을 확인시키는 결과입니다.



### DAIRHuM: A Platform for Directly Aligning AI Representations with Human Musical Judgments applied to Carnatic Music (https://arxiv.org/abs/2411.14907)
Comments:
          4 Pages, ICASSP workshop submission

- **What's New**: 이 논문에서는 AI 음악 모델의 표현을 인간의 음악 판단과 직접적으로 정렬하는 DAIRHuM(Direct alignment between AI music model Representations and Human Musical judgments) 플랫폼을 소개합니다. 이 플랫폼은 음악 기록 데이터셋에서 유사성을 라벨링하고, 사전 훈련된 모델의 정렬 상태를 정량적 점수 및 비주얼 플롯을 통해 검사하도록 설계되었습니다. 특히, 캔틱(Carnatic) 음악 장르에서 데이터를 수집하기 어려운 상황을 해결하며 음악 AI 도구에 대한 접근성을 높입니다.

- **Technical Details**: DAIRHuM 시스템은 Python 패키지로, 사용자들이 음악 과제에서 인간-모델 정렬을 탐색하고 해석할 수 있도록 지원합니다. 이 시스템은 소스 및 변형 추적 레이블링, 임베딩 분석, 그리고 인간 판단과의 성능 평가를 통해 정렬을 체계적으로 평가합니다. 예를 들어, 사용자들은 원본 녹음과 변형 버전을 포함하는 오디오 트랙을 수집하고, 음악적으로 의미 있는 라벨을 지정한 후, AI 모델의 임베딩을 생성하여 유의미한 차이를 검사합니다.

- **Performance Highlights**: Carnatic 음악에서의 리드 퍼커셔니스트(Mridangist)와 보조 퍼커셔니스트(Kanjirist) 간의 리드 및 조화에 대한 AI 모델 정렬 평가 결과는 의미 있는 발견을 보여줍니다. 이 연구는 AI가 인간의 리듬적 조화에 대한 판단과 어떻게 일치하는지에 대한 새로운 통찰을 제공하며, 캔틱 음악의 리듬 지각 및 음악 유사성 판단의 차이를 강조합니다. DAIRHuM 플랫폼의 발전은 인도 음악에서 데이터 부족과 문화적 특수성을 다루면서 MIR 연구의 진전을 이끌어냅니다.



### Boundless Across Domains: A New Paradigm of Adaptive Feature and Cross-Attention for Domain Generalization in Medical Image Segmentation (https://arxiv.org/abs/2411.14883)
Comments:
          5 pages, 3 figures

- **What's New**: 이번 연구에서는 domain generalization (DG) 문제를 해결하기 위해 Adaptive Feature Blending (AFB)이라는 새로운 방법론을 제안합니다. AFB는 source domain에서 발생하는 feature 및 스타일 정보를 활용하여 out-of-distribution 샘플을 생성하고, 이를 통해 domain distribution의 다양성을 크게 확대합니다. 또한, Dual Cross-Attention Regularization (DCAR)을 사용하여 domain-invariant representation을 학습하는 강력한 모델을 구축합니다.

- **Technical Details**: 이 연구에서는 cross-channel attention 메커니즘을 활용하여 domain-invariant representation을 학습하는 방법론을 제시합니다. 제안된 DCAR는 source domain에서 얻은 deep features를 쿼리로, 생성된 도메인 이미지의 deep features를 키와 값으로 사용하여 특정 채널 간 유사성 매트릭스를 계산합니다. 이를 기반으로 원래의 deep features를 강력한 정규화 표현으로 재구성하여 domain-invariant representation을 학습합니다.

- **Performance Highlights**: 제안된 방법론은 두 개의 표준 domain generalization 벤치마크 데이터셋에서 의료 이미지 분할 작업의 성능을 향상시키는 데 뛰어난 효율성을 보여줍니다. extensive experimental results에 따르면, Adaptive Feature Blending와 Dual Cross-Attention Regularization 기법이 기존의 방법보다 우수한 성능을 달성했습니다. 이러한 결과는 의료 이미징 분야의 실제 적용 가능성을 높이는 데 기여할 것으로 예상됩니다.



### Prioritize Denoising Steps on Diffusion Model Preference Alignment via Explicit Denoised Distribution Estimation (https://arxiv.org/abs/2411.14871)
- **What's New**: 본 논문에서는 Denoised Distribution Estimation (DDE)이라는 새로운 기법을 제안합니다. DDE는 기존의 보조 모델이나 수작업 방식에 의존하지 않고, 각 디노이징 단계에서의 관점으로부터 말단의 디노이즈 분포를 직접적으로 추정합니다. 이를 통해 스프의 선호 레이블이 드물게 제공되는 환경에서도 보다 효율적인 신용 할당이 가능해집니다.

- **Technical Details**: DDE는 두 가지 추정 전략, 즉 단계별(stepwise) 및 일괄(single-shot) 추정을 제안합니다. 단계별 추정은 조건부 분포를 기반으로 하여 모델 분포를 추정하는 반면, 일괄 추정은 DDIM 모델링을 사용하여 중간 노이즈 상태를 바탕으로 말단 분포를 직접 평가합니다. 이러한 두 방법을 통합함으로써 모델 추론을 통한 전체 디노이징 궤적의 평가가 가능해집니다.

- **Performance Highlights**: 우리는 SD15와 SDXL에서 DDE를 평가하였고, 그 결과 기존의 보조 모델이 없는 방법들에 비해 뛰어난 성능을 입증하였습니다. DDE는 SD15와 SDXL의 성능 지표를 각각 3.3%에서 6.7%, 1.0%에서 3.1% 향상시켰습니다. 전반적으로 DDE는 기존 접근 방식과 비교할 때 수치적, 질적으로 최고 수준의 성능을 보여주었습니다.



### Application of AI to formal methods -- an analysis of current trends (https://arxiv.org/abs/2411.14870)
- **What's New**: 이번 연구에서는 인공지능(AI)과 형식적 방법(FM) 사이의 상관관계를 분석하기 위해 체계적인 매핑 연구(SMS)를 수행했습니다. 최근 5년 간의 문헌을 조사하여 AI 기술이 FM 분야에 어떻게 적용되고 있는지를 확인했습니다. 이 연구는 189개의 주요 연구 결과를 심층적으로 탐구하며, AI와 FM의 접목이 현재 어떤 경향을 보이는지, 연구의 공백이 어디에 있는지를 추적합니다.

- **Technical Details**: 먼저 연구 분야를 선정한 후, 연구 질문을 수립하고 포함 기준(IC) 및 제외 기준(EC)을 정의합니다. 이를 통해 과학적 메타 검색 엔진에서 결과를 필터링하고, 적절한 연구 결과를 수집합니다. 형식적 방법(FM)은 소프트웨어 및 하드웨어 시스템을 설계하는 수학적으로 엄격한 접근법으로, 모델 검증, 이론 증명 등 다양한 기술이 활용됩니다.

- **Performance Highlights**: 최근 5년 동안의 데이터 세트를 통해 AI와 FM의 응용 프로그램을 분석하여 연구의 발전 방향과 가장 유망한 AI 기법을 도출했습니다. 이 연구 결과는 향후 연구의 방향성을 제시하며, 데이터 세트는 모든 연구자들에게 공개됩니다. 따라서, FM과 AI의 융합이 계속 발전할 것으로 기대됩니다.



### BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligenc (https://arxiv.org/abs/2411.14869)
- **What's New**: 이번 논문에서는 BIP3D라는 새로운 이미지 중심 3D 지각 모델을 제안합니다. 현재의 포인트 클라우드 접근법의 한계를 극복하기 위해, 이미지 특성과 명시적인 3D 위치 부호화를 결합했습니다. BIP3D는 다중 뷰 이미지와 텍스트 특성을 융합하여 3D 객체 감지 및 3D 시각적 기반 작업을 수행하며, 깊이 맵을 보조 입력으로 사용할 수 있습니다.

- **Technical Details**: BIP3D는 GroundingDINO 모델을 기반으로 하며, 카메라 모델을 구성하고 다중 뷰 데이터 및 여러 모드를 융합하는 기능을 최적화했습니다. 카메라 매개변수를 지원하여 2D 이미지 특성에 3D 위치 부호화를 추가하고, 2D 비대칭 주의(attention) 기법을 3D 형태로 수정하여 다이나믹한 다중 뷰 특성 융합을 구현했습니다.

- **Performance Highlights**: 실험 결과, BIP3D는 EmbodiedScan 벤치마크에서 현재 최고 성능을 달성하여, 3D 감지 작업에서 5.69%의 성능 향상을 보였고, 3D 시각적 기반 작업에서는 15.25%의 향상을 기록했습니다. 이 모델은 또한 RGB 전용 입력을 지원하여 Crowd-sourcing을 통해 대량의 데이터 수집이 가능합니다.



### Latent Schrodinger Bridge: Prompting Latent Diffusion for Fast Unpaired Image-to-Image Translation (https://arxiv.org/abs/2411.14863)
- **What's New**: 이 논문에서는 Schrodinger Bridges (SBs)를 활용하여 이미지 생성 및 데이터에서의 변환 문제를 해결하는 새로운 Latent Schrödinger Bridges (LSBs) 알고리즘을 제안하고 있습니다. 이 알고리즘은 기존의 Diffusion Models (DMs)보다 적은 수의 Neural Function Evaluations (NFEs)로 경쟁력 있는 이미지 간 변환(I2I translation)을 수행할 수 있습니다. 또한, SB의 확률 흐름을 활용하여 효율적인 계산을 가능하게 하여 실용성을 높이고 있습니다.

- **Technical Details**: Schrodinger Bridges는 두 분포 간의 최소한의 전송 비용을 통해 stochastic differential equations (SDEs)를 학습하는 방법입니다. 이 논문에서는 SB의 확률 흐름 ODE 벡터 필드를 소스 예측기, 타겟 예측기 및 노이즈 예측기의 선형 결합으로 분해하는 방법을 제안합니다. 이를 통해 Latent Diffusion Models (LDMs)와 결합하여 효율적인 ODE 근사를 제공하는 Latent Schrödinger Bridges (LSBs)를 개발하였습니다.

- **Performance Highlights**: 제안한 LSB 알고리즘은 기존 DM 기반의 이미지 간 변환 방법보다 현저히 적은 계산 비용으로 경쟁력 있는 성능을 보여줍니다. 실험 결과, LSB는 여러 데이터셋에서 높은 확장성을 가지며, diffusion-based I2I translation에서 탁월한 성능을 발휘함을 입증했습니다.



### Dynamics-Aware Gaussian Splatting Streaming Towards Fast On-the-Fly Training for 4D Reconstruction (https://arxiv.org/abs/2411.14847)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 4D 동적 공간 재구성을 위한 새로운 3단계 파이프라인(DASS)을 제안합니다. 이 방법은 기존의 전신 멀티뷰 비디오에 의존하지 않고, 현재 프레임에서의 재구성을 이전 캐시와 현재 멀티뷰 입력을 기반으로 진행합니다. 또한, 시간적 연속성을 고려하여 선택적 상속 메커니즘을 통해 이전 프레임에서 추가된 Gaussians를 보존합니다.

- **Technical Details**: 3단계 파이프라인은 상속(inheritance), 이동(shift), 밀도화(densification)로 구성됩니다. 각 단계에서는 새로운 Gaussians가 이전 프레임의 Gaussians를 선택적으로 상속받고, 2D 동적 오프셋과 Gaussian 분할을 사용하여 움직이는 객체와 정적인 객체를 구분하여 최적화합니다. 마지막 단계에서는 새로운 객체를 수용하기 위해 새 Gaussians를 추가하고, 이 과정에서 위치 기울기와 오류 맵을 고려합니다.

- **Performance Highlights**: 이 방법은 기존의 동적 공간 재구성 방법들에 비해 20% 향상된 온라인 훈련 속도, 높은 재구성 품질, 실시간 렌더링 능력을 입증하였습니다. 이 연구는 실시간으로 동적 장면을 처리하는 데 중요한 진전을 보여주며, AR/VR 및 홀로그램 통신 등 다양한 실용적 응용에 기여할 수 있습니다.



### Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2411.14842)
- **What's New**: 본 논문에서는 Chat-Audio Attacks (CAA) 벤치마크를 도입하여 음성 기반 인간-기계 상호작용에서 대규모 언어 모델(LLMs)에게 가해지는 적대적 오디오 공격의 취약성을 탐구하고자 합니다. CAA 벤치마크는 고유한 네 가지 유형의 오디오 공격을 포함한 360개의 공격 세트를 제공합니다. 이를 통해 LLM의 안정성을 평가하기 위해 세 가지 평가 전략을 제안하고 있습니다.

- **Technical Details**: CAA 벤치마크는 원래 오디오와 관련된 문서, 음성 에이전트의 음성을 포함한 각 공격 세트로 구성되어 있습니다. 이 연구에서는 세 가지 평가 방법, 즉 표준 평가(Standard Evaluation), GPT-4o 기반 평가(GPT-4o-Based Evaluation), 인간 평가(Human Evaluation)를 통해 LLM의 성능을 체계적으로 평가합니다. 특히, CAA 벤치마크에 포함된 공격 유형은 콘텐츠 공격(content attack), 감정 공격(emotional attack), 명확한 잡음 공격(explicit noise attack), 암시적 잡음 공격(implicit noise attack)입니다.

- **Performance Highlights**: CAA 벤치마크를 통해 총 1,680개의 적대적 오디오 샘플의 성능을 평가한 결과, GPT-4o가 가장 높은 회복력을 보였으며, 다른 5개의 최신 LLM 모델들과의 비교 분석을 통해 음성 인터랙션의 취약성을 강조하였습니다. 연구 결과는 각 공격이 LLM의 성능에 미치는 영향을 명확히 드러내며, 사용자 신뢰 및 지각에 대한 통찰을 제공하였습니다.



### VisGraphVar: A Benchmark Generator for Assessing Variability in Graph Analysis Using Large Vision-Language Models (https://arxiv.org/abs/2411.14832)
- **What's New**: 이 논문은 LVLMs(Large Vision-Language Models)의 시각 그래프 분석을 위한 새로운 기준 생성기인 VisGraphVar(Visual Graph Variability)를 소개합니다. 이는 서로 다른 그래프 이미지 스타일과 구조를 생성하여 LVLM의 강점과 약점을 체계적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: VisGraphVar는 노드 및 엣지 탐지, 그래프 유형 분류, 세분화, 패턴 인식, 링크 예측, 추론, 일치 확인 등 총 7개의 작업 범주로 구성되어 있습니다. 이 생성기에 의해 만들어진 990개의 그래프 이미지를 통해 6개의 LVLM을 평가하였으며, 줄 무샘과 사고 사슬의 두 가지 다른 프롬프트 전략을 사용하였습니다.

- **Performance Highlights**: 실험 결과, 이미지의 시각적 속성 변화(예: 노드 레이블링 및 레이아웃)와 의도적으로 포함된 시각적 결함(예: 중첩된 노드)이 모델 성능에 상당한 영향을 미치는 것으로 나타났습니다. 이 연구는 LVLM의 시각 그래프 분석 능력을 보다 신뢰성 있고 견고한 시스템으로 발전시키기 위한 통찰력을 제공합니다.



### Physically Interpretable Probabilistic Domain Characterization (https://arxiv.org/abs/2411.14827)
- **What's New**: 본 논문에서는 자율주행 차량 환경에서 도메인 특성을 확률 분포로 캐릭터화하는 새로운 접근 방식을 제안합니다. 이는 기존의 회귀 혹은 분류 문제 접근법의 한계를 넘어서, 물리적 매개변수의 분포를 추정하여 차별화된 도메인 특성을 제공하는 방법입니다. 특히, 노멀라이징 플로우(normalizing flows)를 사용하여 차량에 장착된 카메라에서 촬영한 이미지로부터 다양한 기상 조건의 예측 가능성을 높입니다.

- **Technical Details**: 저자들은 카메라로 수집된 고급 이미지 데이터를 기반으로 실시간으로 기상 매개변수를 예측하는 세 가지 작업을 제안합니다. 이 방법은 고차원 데이터에서의 우도 추정의 비효율성을 극복하고, 이미지와 날씨 분포에서 날씨 매개변수를 모델링하기 위해 노멀라이징 플로우를 활용합니다. 이를 통해 단일 이미지에서의 기상 조건 예측뿐만 아니라, 여러 이미지의 집합을 통한 절대 및 상대 도메인 특성화가 가능합니다.

- **Performance Highlights**: 제안된 방법은 자율주행 시스템이 다양한 환경 조건에 적응할 수 있도록 하며, 안전한 작동을 보장합니다. 여러 도메인 간 비교를 통해 특정 도메인 내에서의 안전한 작동 가능성을 평가할 수 있으며, 이는 자율주행차의 실제 운전 조건과 관련하여 큰 실용적 응용 가능성을 제공합니다. 본 연구는 기상 예측의 정확성과 효과적인 도메인 적응이 자율 시스템의 동적 환경 조정에 얼마나 중요한지를 강조합니다.



### High-Resolution Image Synthesis via Next-Token Prediction (https://arxiv.org/abs/2411.14808)
Comments:
          30 pages

- **What's New**: D-JEPA (Denoising with a Joint-Embedding Predictive Architecture)는 이미지 생성에서 뛰어난 성능을 보여주는 자기회귀 모형입니다. 본 논문에서는 D-JEPA의 확장 모델인 D-JEPA⋅T2I를 소개하는데, 이는 flow matching loss를 적용하여 데이터 효율적 연속 해상도 학습을 가능하게 합니다. 또한, 멀티모달 비주얼 트랜스포머를 이용하여 텍스트와 비주얼 특징을 효과적으로 통합합니다.

- **Technical Details**: D-JEPA⋅T2I는 새로운 모델 아키텍처와 데이터 활용 방식으로 설계되었습니다. 특히, Visual Rotary Positional Embedding (VoPE)을 도입하여 연속 해상도 학습을 지원하며, 데이터 피드백 메커니즘을 통해 데이터 활용도를 극대화합니다. 이는 전통적인 데이터 처리와 차별화되는 접근법으로, 실시간 통계 분석을 통한 데이터 배급 조정이 특징입니다.

- **Performance Highlights**: D-JEPA⋅T2I는 T2I-CompBench, GenEval, GenAI-Bench 벤치마크에서 고해상도 이미지 합성 분야에서 최첨단의 성능을 달성했습니다. 특히, 다음-토큰 예측(next-token prediction)을 통해 이미지 품질을 크게 개선하였으며, 기존의 autoregressive 모델과 비교해 이미지 텍스처와 품질 면에서 우월한 성능을 보입니다.



### Continual SFT Matches Multimodal RLHF with Negative Supervision (https://arxiv.org/abs/2411.14797)
- **What's New**: 이 논문에서는 비전-언어 모델(VLM)의 선호 정렬 과정에서 다중 모달 RLHF(강화 학습에서 인간 피드백 사용)의 고유한 가치가 거부된 응답의 로그리트를 통한 부정 감독에 있음을 관찰하고, 새로운 부정 감독 미세 조정(nSFT) 접근 방식을 제안합니다. nSFT는 VLM과 간단한 SFT 손실(Supervised Finetuning Loss)을 지속적으로 정렬하며, 기존의 슬롯 방식에 비해 메모리 효율성이 뛰어난 장점을 가지고 있습니다. 이는 여러 데이터 세트 및 평가 지표와 비교하여 nSFT가 더 나은 성능을 발휘함을 수치적으로 보여줍니다.

- **Technical Details**: 논문의 핵심은 nSFT가 RLHF 최적화에서 부정 감독을 분리하여 훈련을 더 효율적으로 만들 수 있다는 점입니다. 다중 모달 RLHF에서는 메모리를 2개 또는 4개의 모델을 사용해야 하지만, nSFT는 오직 하나의 모델에서 수행된다는 점이 강조됩니다. 이 논문은 LLM(예: GPT-4)를 사용하여 부정 감독의 불일치를 식별하고 이미지 관련 대화를 구성하여 부정 응답으로부터 자기 견책의 학습을 돕습니다.

- **Performance Highlights**: nSFT는 다양한 실험을 통해 SFT의 한계를 보완하고, 여러 평가 지표에서 최고의 성과를 달성합니다. 특히 LLaVA-NeXT와 같은 강력한 VLM에서도 가설이 유지됩니다. 시험 결과는 nSFT가 기존의 다중 모달 RLHF 방법과 견줄 만큼 강력하며, 향후 연구에 자극이 되길 바랍니다.



### VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection (https://arxiv.org/abs/2411.14794)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 VideoQA(비디오 질문-답변) 작업의 한계를 극복하기 위해 VideoEspresso라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 중요한 공간적 세부정보와 시간적 일관성을 보존하며, 중간 추론 단계를 포함한 다중 모드 주석(annotation)을 특징으로 합니다. 또한, 저자들은 Hybrid LVLMs Collaboration 프레임워크를 제안하여 질의에 적합한 핵심 프레임을 선택하고, 체인 오브 사고(reasoning) 접근법을 통해 비디오 내용을 기반으로 논리적 관계를 추출합니다.

- **Technical Details**: VideoEspresso의 구축은 자동화된 파이프라인을 통해 이루어지며, 이를 통해 효율적으로 QA 쌍을 생성합니다. 처음에는 LVLM을 사용하여 비디오 프레임을 언어 공간으로 매핑한 후, 의미론적 유사성을 바탕으로 중복 프레임을 제거합니다. 이어서 GPT-4o를 활용하여 QA 쌍을 생성하고 저품질 데이터를 필터링합니다. 중간 논리 과정을 확장하기 위해 Video Chain-of-Thought 주석을 도입하여 이러한 정보가 QA 쌍 형성에 기여하도록 설계되었습니다.

- **Performance Highlights**: 새롭게 제안된 평가 기준을 기반으로 9개의 인기 LVLM을 대상으로 14개의 작업에서 성능을 평가한 결과, 제안된 방법이 대부분의 작업에서 기존 방법들을 초월하는 것으로 나타났습니다. 이러한 평가를 통해 VideoEspresso의 고품질 VideoQA 쌍이 복합적인 비디오 추론 작업에서 우수한 성능을 발휘함을 보였습니다. 코드 및 데이터셋은 추후 공개될 예정입니다.



### KBAda: Efficient Self Adaptation on Specific Knowledge Bases (https://arxiv.org/abs/2411.14790)
- **What's New**: 새롭게 제안된 KBAda는 대규모 언어 모델(LLM)이 지식 기반에 효율적으로 적응하도록 설계된 기법입니다. 이 접근 방식은 자가 주석(self-annotation)된 데이터를 활용하여 LLM의 자가 학습 잠재력을 활용합니다. KBAda는 Q&A 쌍 및 수정 제안과 같은 자가 주석 데이터를 통해 반복적인 학습을 수행하며, 자원 소모 없이도 모델 성능을 크게 향상시키는 것으로 확인되었습니다.

- **Technical Details**: KBAda는 LLM이 지식 기반(KB) 적응을 위해 자가 주석 방법을 통해 모델을 세밀하게 조정합니다. 이는 사실 QA, 장문 QA 및 전문 분야 테스트와 같은 다양한 다운스트림 작업에서 모델 성능을 확인합니다. 실험 결과, 기존의 GPT-4-turbo 주석만큼의 성능 향상(90% 이상)을 달성하면서도 전적으로 자가 감독(self-supervision)에 의존하여 비용을 낮췄습니다.

- **Performance Highlights**: KBAda는 반복적인 자기 검사와 학습 대응을 통해 모델의 반응을 개선하며, 이 과정에서 쿼리 확장(query expansion) 및 신뢰도 검증(confidence verification) 등의 전략을 활용합니다. 다양한 실험 결과 KBAda가 어떻게 일반 지식 콘텐츠를 마스터하고 значительный 성능 향상을 달성하는지를 보여줍니다. 아울러 자기 주석 데이터의 효율적 양과 최적의 학습량도 규명하여 실무에서 KBAda를 효과적으로 활용하는 데 도움을 줄 수 있는 지침을 제공합니다.



### Resolution-Agnostic Transformer-based Climate Downscaling (https://arxiv.org/abs/2411.14774)
- **What's New**: 이번 연구는 고해상도 대기 예측 모델을 활용한 새로운 다운스케일링 방법을 소개합니다. 특히, Earth Vision Transformer(Earth ViT) 모델이 기존의 Global Climate Models(GCMs)로부터 50km에서 25km, 그리고 3km 해상도로 다운스케일링되는 과정을 보여주며, 별도의 추가 훈련 없이 이뤄진 일반화 가능성을 강조합니다. 이 방법은 극단적인 기상 현상에 대한 더 나은 계획 수립을 가능하게 하는 기회를 제공합니다.

- **Technical Details**: Earth ViT 구조는 Pangu-Weather 모델과 유사하며, 압력 수치를 13개에서 3개로 줄이고 더 높은 해상도를 위한 출력을 약간 수정하였습니다. 모델의 물질 보존 의무를 보장하기 위해 사용자 정의 손실 함수가 구현되어, 저해상도 입력 이미지에서 계산된 질량과 초해상도 출력 이미지에서 계산된 질량을 비교합니다. 이는 물리적 일관성을 유지하는 데 큰 역할을 합니다.

- **Performance Highlights**: 실험 결과, Earth ViT는 ERA5 데이터를 기반으로 하는 2배 다운스케일링 작업에서 2000/2001년 데이터를 훈련하여 성능을 보였습니다. ResNet 모델과의 비교를 통해 Earth ViT는 다양한 기온 및 강수량 변수에서 효과적으로 작동함을 보여줍니다. 이 새로운 접근 방식은 기존의 기후 모델링 절차보다 경제적이면서도 효율적인 대안으로 여겨집니다.



### Mode-conditioned music learning and composition: a spiking neural network inspired by neuroscience and psychology (https://arxiv.org/abs/2411.14773)
Comments:
          18 pages, 8 figures

- **What's New**: 이 논문에서는 음악 모드(musical mode)와 키(key)를 표현하고 음악적 조화(harmonic relationships)를 생성하기 위해 뇌 메커니즘과 심리학 이론에 영감을 받은 spiking neural network (SNN)를 제안합니다. 기존의 인공지능 모델은 이러한 다양성을 간과하는 반면, 인간은 다양한 모드와 키를 인식할 수 있는 인지적 메커니즘을 갖추고 있습니다. 이 연구의 목표는 음악을 학습하고 생성하는 시스템을 개발하는 동시에 인간의 인지와 인공지능의 격차를 해소하는 것입니다.

- **Technical Details**: 제안된 모델은 관련 뇌 영역의 구조와 기능에 영감을 받은 여러 협력적 서브시스템으로 설계되었습니다. 신경 회로(neural circuit) 진화 학습 메커니즘을 통합하여 이 네트워크는 음악에서 모드 관련 특징을 학습하고 생성할 수 있도록 했습니다. 이 모델은 Krumhansl-Schmuckler 모델과 유사한 연결 구조를 보여줌으로써 음악 심리학 영역에서의 인지적 이해를 반영합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 주어진 모드와 키의 특성을 갖춘 음악 작품을 생성할 수 있으며, 생성된 음악 조각은 조화적 특성과 다채로운 음악 콘텐츠 생성을 위한 멜로디 적응성을 모두 갖추고 있습니다. 이러한 결과는 심리학적 이론과 신경 과학의 통찰을 인공지능 모델에 통합한 방식으로 검증되었습니다.



### Grid and Road Expressions Are Complementary for Trajectory Representation Learning (https://arxiv.org/abs/2411.14768)
Comments:
          This paper is accepted by KDD2025(August Cycle)

- **What's New**: 이번 연구는 기존의 Trajectory Representation Learning (TRL) 방법들이 단일 유형의 궤적 표현만을 사용하고 있다는 점에 주목합니다. 우리는 Grid 궤적과 Road 궤적을 결합하여 서로 보완적인 정보를 활용하는 새로운 다중 모달(TRL) 방법인 GREEN을 제안합니다. GREEN은 GPS 궤적을 Raw에서 Grid와 Road 궤적으로 변환하고, 두 가지 인코더를 통해 각각의 정보를 포착합니다.

- **Technical Details**: GREEN은 각 궤적 표현을 효과적으로 캡처하기 위해 Convolutional Neural Network (CNN)와 Graph Neural Network (GNN)을 사용하는 두 개의 인코더를 포함합니다. 이러한 인코더들은 Transformer 구조를 기반으로 하여 궤적을 Grid와 Road 표현의 시퀀스로 모델링합니다. 두 인코더의 출력을 일치시키고 통합하기 위해 Contrastive Loss와 Masked Language Model (MLM) Loss를 설계하여 블록 간의 SoA를 극복하고 있습니다.

- **Performance Highlights**: GREEN은 7개의 최신 TRL 방법과 비교하여 2개의 실제 데이터 세트에서 3개의 주요 하위 작업에 대해 일관되게 뛰어난 성능을 보였습니다. 특히, GREEN은 Travel Time Estimation, Trajectory Classification, Most Similar Trajectory Search에서 각각 19.55%, 2.21%, 23.90%의 평균 정확도 향상을 보였습니다. 전반적으로 GREEN은 기존 방법들과 비교할 때 학습 및 추론 효율성 역시 유사한 수준을 유지하며, 모든 구성 요소들이 정확도 향상에 기여함을 확인했습니다.



### Efficient Long Video Tokenization via Coordinated-based Patch Reconstruction (https://arxiv.org/abs/2411.14762)
Comments:
          Code is available on the project webpage: this https URL

- **What's New**: 인트로덕션에서 발표된 새로운 동영상 토크나이저 CoordTok은 시계열 및 공간 정보를 효과적으로 활용할 수 있는 방법을 제시합니다. 기존의 비디오 토크나이저가 짧은 비디오 클립을 인코딩하는 데 한계가 있었던 반면, CoordTok은 긴 비디오를 직접 인코딩할 수 있도록 설계되었습니다. 이 방법은 랜덤 샘플링된 좌표를 기반으로 비디오 패치를 재구성하는 방식을 통해 훈련 비용을 획기적으로 줄일 수 있습니다.

- **Technical Details**: CoordTok은 비디오를 분할된 3D 표현으로 인코딩하고, 주어진 (x, y, t) 좌표에 해당하는 비디오 패치를 재구성합니다. 이 과정에서 비디오는 겹치지 않는 공간-시간 패치로 나누어지고, 각 패치는 학습 가능한 위치 임베딩을 사용하여 처리됩니다. 두 차원으로 나뉘는 표현을 활용함으로써, CoordTok은 메모리와 연산 자원을 효율적으로 사용합니다.

- **Performance Highlights**: 실험 결과, CoordTok은 긴 비디오 클립을 인코딩하는 데 필요한 토큰 수를 기존 방법보다 크게 줄일 수 있음을 보여주었습니다. 예를 들어, 128x128 해상도의 128프레임 비디오를 단 1280개의 토큰으로 인코딩할 수 있는 반면, 기존 방법들은 6144개 또는 8192개의 토큰이 필요합니다. 이러한 효율적인 인코딩 덕분에 대규모 Diffusion Transformer 모델을 메모리 효율적으로 훈련할 수 있습니다.



### Hammer: Towards Efficient Hot-Cold Data Identification via Online Learning (https://arxiv.org/abs/2411.14759)
- **What's New**: 이번 논문에서는 빅데이터와 클라우드 컴퓨팅 환경에서 저장소 자원 관리의 효율성을 높이기 위해 데이터의 '콜드(cold)' 상태와 '핫(hot)' 상태를 정확하게 식별하는 새로운 솔루션을 제안합니다. 전통적인 규칙 기반 알고리즘이나 초기 AI 기법은 동적인 작업 부하에 취약함을 드러내며, 이에 대한 대안을 온라인 학습(on-line learning) 전략을 통해 보여줍니다.

- **Technical Details**: 제안된 방법은 데이터 접근 패턴의 변화에 동적으로 적응하여 더욱 높은 정확도(accuracy)와 낮은 운영 비용(low operational cost)을 달성합니다. 이 연구에서는 합성(synthetic) 데이터와 실세계(real-world) 데이터셋을 사용한 엄격한 테스트를 통해 핫-콜드 분류에서 90%의 정확도를 달성하며, 계산(computational) 및 저장(storage) 오버헤드를 크게 줄이는 성과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 동적 작업 환경에서 더 나은 정확도와 효율성을 보여주며, 특히 핫-콜드 데이터 식별에서 현저한 성과를 나타내고 있습니다. 이는 클라우드 환경에서 자원 관리의 혁신적인 개선을 기대하게 합니다.



### TopoSD: Topology-Enhanced Lane Segment Perception with SDMap Prior (https://arxiv.org/abs/2411.14751)
Comments:
          17 pages, 7 figures, and 7 tables

- **What's New**: 이번 연구에서는 고해상도 지도(HDMap)에 대한 의존도를 줄이는 방향으로 자율주행 시스템을 혁신적으로 개선하고 있습니다. 고비용의 주석 및 유지 관리 문제를 해결하기 위해, 연구자들은 온보드 센서를 활용한 온라인 벡터화 HDMap 구축에 중점을 두고 있습니다. 연구진은 저해상도 지도(SDMap)를 사용하여 환경을 '인식'하고, 이는 주행 장면 이해에 크게 기여할 것으로 기대됩니다.

- **Technical Details**: SDMap 요소들은 신경 공간 맵 표현(neural spatial map representation) 및 인스턴스 토큰(instance tokens)으로 인코딩되어, 비행 시점(BEV) 특성을 향상시킵니다. 제안된 경로 세그먼트 표현 프레임워크는 경로, 중심선 및 그 토폴로지를 동시에 예측합니다. 또한, 토폴로지 유도 디코더(topology-guided decoder)를 사용하여 지오메트리 예측 및 토폴로지 추론 능력을 더욱 향상시킵니다.

- **Performance Highlights**: 개선된 모델은 OpenLane-V2 데이터셋에서 실험하여 기존의 최첨단 방법들보다 현저한 성능 향상을 보였고, mAP에서 +6.7, 토폴로지 메트릭에서 +9.1의 성과를 기록했습니다. 연구 결과, SDMap 노이즈 증강을 활용한 모델이 더 높은 견고성을 나타내는 것으로 나타났습니다.



### Point Cloud Understanding via Attention-Driven Contrastive Learning (https://arxiv.org/abs/2411.14744)
- **What's New**: 이번 논문에서는 PointACL이라는 새로운 Attention-driven Contrastive Learning 프레임워크를 제안합니다. 이 방법은 포인트 클라우드(point cloud) 이해에서 발생하는 기존 모델의 한계를 보완하기 위해 설계되었습니다. 특히, PointACL은 저조도 주의(low-attention) 영역에 초점을 맞추고, 더 나아가 글로벌 구조에 대한 이해도를 향상 시킵니다.

- **Technical Details**: PointACL의 핵심 구성 요소는 Attention-driven Dynamic Masking 전략입니다. 이는 모델이 일부 주요 패치에 지나치게 의존하지 않도록 하여 저조도 영역에 더 많은 주의를 기울이게 합니다. 또한, 원래의 pre-training 손실과 대조 학습 손실을 결합하여 모델의 특성 구별(feature discrimination)과 일반화를 향상시킵니다.

- **Performance Highlights**: PointACL은 다양한 3D 이해 과제에서 최고 수준(state-of-the-art)의 성능을 달성하였습니다. 예를 들어, ScanObjectNN에서 89.9%의 정확도를 기록하였고, ModelNet40에서는 94.1%에 달합니다. 여기에 더해, PointACL은 노이즈 환경에서도 뛰어난 강인성을 보여 주목받고 있습니다.



### FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification (https://arxiv.org/abs/2411.14743)
Comments:
          15 pages, 3 figures

- **What's New**: 이 논문은 암 진단을 위한 Few-shot learning의 필요성과 이를 해결하기 위한 혁신적인 프레임워크인 FOCUS를 제안합니다. FOCUS는 pathology foundation models (FMs)와 언어 사전 지식을 결합하여 진단적으로 중요한 영역에 집중적으로 접근할 수 있게 해줍니다. 이를 통해 기존의 기술적 한계를 극복하고, 데이터가 부족한 환경에서도 효과적인 암 진단을 가능케 합니다. FOCUS는 다단계 압축 전략을 통해 전반적인 모델의 성능을 향상시킵니다.

- **Technical Details**: FOCUS는 (1) 전역 시각적 중복 제거 모듈을 통해 자극적(novel) feature를 선택하고, (2) 시맨틱 관련성을 기준으로 visual tokens를 우선시하는 지식 강화 압축 메커니즘을 도입합니다. 마지막으로,(3) 쌍별 순차 토큰 압축 모듈을 통해 코사인 유사성 임계값을 유지하면서, 시공간 연속성을 보존합니다. 이러한 접근 방식은 데이터가 제한적인 상황에서도 모델의 주의력을 중요한 진단적 특성에 집중시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: FOCUS는 다양한 암 데이터셋에서 광범위한 평가를 통해 기존의 최신 기술(SOTA)과 동등하거나 우수한 성능을 보였습니다. 특히 작은 샘플로도 효과적으로 진단적 특징을 포착하는 데 있어 그 효과성을 입증했습니다. 추가적인 ablation studies를 통해 제안된 아키텍처의 각 구성 요소의 기여도를 체계적으로 분석하였으며, three-stage 압축 전략, pathology FMs의 선택과 언어 프롬프트의 디자인이 효과적임을 보여 주었습니다.



### TEXGen: a Generative Diffusion Model for Mesh Textures (https://arxiv.org/abs/2411.14740)
Comments:
          Accepted to SIGGRAPH Asia Journal Article (TOG 2024)

- **What's New**: 이번 연구에서는 기존의 2D diffusion 모델을 활용한 테스트 시 최적화 접근법에서 벗어나, UV 텍스처 공간을 직접 학습하는 방법에 초점을 맞췄습니다. 새로운 대규모 diffusion 모델인 TEXGen을 소개하며, 이 모델은 텍스처 맵을 직접 생성할 수 있는 기능을 통해 고해상도 텍스처 생성에 기여합니다. 특히, 모델은 텍스트 프롬프트와 단일 시점 이미지를 통해 유도된 UV 텍스처 맵을 생성할 수 있습니다.

- **Technical Details**: 우리는 UV 텍스처 맵을 생성하는 데 필요한 효율적인 네트워크 아키텍처를 제안합니다. 이 아키텍처는 UV 맵에 대한 컨볼루션을 점 구름(point clouds)의 attention 레이어와 교차하여 처리합니다. 이를 통해 3D 표면에서 효과적인 특징 상호작용을 달성하며, 매니지 가능한 계산으로 3D 연속성을 유지합니다. 700M 매개변수를 가진 이 모델은 고해상도 텍스처를 생성할 수 있는 능력을 지니고 있습니다.

- **Performance Highlights**: TEXGen은 텍스처 생성, 인페인팅(inpainting), 스파스 뷰(sparse view)에서의 텍스처 완성 등의 다양한 확장 응용 프로그램을 자연스럽게 지원합니다. 이 연구 결과는 기존 방식보다 더 높은 품질을 달성하며, 학습 없이 다양한 작업에 사용할 수 있는 기초 모델로 기능합니다. 상태-of-the-art 결과를 달성하여 3D 텍스처의 품질을 향상시키고, 더 광범위한 객체에 대한 일반화 능력을 보여줍니다.



### Universal and Context-Independent Triggers for Precise Control of LLM Outputs (https://arxiv.org/abs/2411.14738)
- **What's New**: 이 연구에서는 기존의 프롬프트 주입 공격(Prompt Injection Attack) 기술을 발전시켜, 범용적(universal)이고 상황에 독립적(context-independent)이며 정확한 출력(precise output)을 제어하는 트리거(trigger)를 찾는 방법을 제안합니다. 이러한 트리거는 광범위한 응용 분야에서 LLM의 출력을 조작할 수 있는 가능성을 제공합니다.

- **Technical Details**: 제안된 방법은 주입된 프롬프트를 두 가지 논리적 구성 요소로 나누는 것입니다: 원하는 내용을 인코딩하는 페이로드(payload)와 모델이 특정 내용을 출력하도록 활성화하는 트리거입니다. 이를 통해 다양하고 복잡한 프롬프트 상황에서도 적용 가능한 트리거를 발견하는 접근법을 제시하며, 이를 통해 평균적으로 높은 성공률을 기록하는 실험 결과도 포함됩니다.

- **Performance Highlights**: 연구 결과, 새로운 트리거는 다양한 문맥(context)과 작업(task)에서 효과적으로 작동함을 입증하였으며, 이러한 공격이 LLM 기반 응용 프로그램에 미치는 잠재적 위험성을 강조합니다. 또한 이 연구는 프롬프트 주입 공격의 심각성을 알리고, 이러한 공격에 대한 경각심을 높이는 데 기여하고자 합니다.



### LIBER: Lifelong User Behavior Modeling Based on Large Language Models (https://arxiv.org/abs/2411.14713)
- **What's New**: 본 논문에서는 Lifelong User Behavior Modeling (LIBER) 프레임워크를 제안하여 LLM(large language models) 기반의 추천 시스템에서의 문제를 해결하고자 합니다. LIBER는 사용자 행동 시퀀스의 이해 문제를 접근하며, 사용자 행동을 효율적으로 처리하기 위해 세 개의 모듈을 포함합니다: User Behavior Streaming Partition (UBSP), User Interest Learning (UIL), User Interest Fusion (UIF). 이를 통해 사용자의 동적 관심사를 효과적으로 포착하고, 추천 성능을 개선하고자 합니다.

- **Technical Details**: LIBER는 짧은 사용자 행동 파티션을 생성하는 Incremental Paradigm을 통해 긴 사용자 행동 시퀀스를 처리합니다. 먼저 UBSP 모듈이 각 사용자의 행동 시퀀스를 짧은 파티션으로 나누고, UIL 모듈이 LLM을 활용하여 이러한 파티션에서 통찰력을 추출합니다. UIF 모듈은 이러한 텍스트 출력을 통합하여 추천 모델에 삽입할 수 있는 포괄적인 표현을 구축합니다.

- **Performance Highlights**: LIBER는 Huawei의 음악 추천 서비스에 적용되어 사용자 재생 횟수와 재생 시간을 각각 3.01% 및 7.69% 증가시켰습니다. 공공 데이터셋과 산업 데이터셋에서의 오프라인 실험 결과, LIBER의 성능은 기존 모델보다 우수함을 입증했습니다. 추가적으로 온라인 A/B 테스트를 통해 LIBER의 효과성과 적용 가능성을 확인하였습니다.



### Understanding LLM Embeddings for Regression (https://arxiv.org/abs/2411.14708)
Comments:
          15 pages, 13 figures

- **What's New**: 이 논문은 LLM(대형 언어 모델)의 임베딩을 활용한 회귀 분석에 대한 최초의 포괄적인 연구 중 하나로, 전통적인 피처 엔지니어링 대신 LLM 임베딩이 고차원 회귀 작업에서 더 효과적일 수 있음을 보입니다. 연구 결과, LLM 임베딩은 숫자 데이터에 대해 Lipschitz 연속성을 보존하며, 이러한 특성이 회귀 성능을 향상시키는 데 기여함을 밝혀냅니다.

- **Technical Details**: 회귀 작업은 주어진 입력 공간에서 스칼라 값을 예측하기 위한 함수와 관련된 데이터 셋으로 구성됩니다. 본 연구는 LLM 임베딩이 흐름이 있는 문자를 구성하는 피처로 사용될 때의 성능을 분석합니다. LLM은 특정 문제에 대해 전통적인 피처 방식과 비교할 때 고차원 데이터에서도 강력한 회귀 성능을 유지하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 연구 결과, LLM 임베딩을 이용한 회귀 성능은 특히 고차원 데이터에 대해 상당한 이점을 제공하며, 전통적인 방식에서는 성능 저하가 예상됩니다. 또한, 모델의 크기와 언어 이해력 같은 요소는 예상과 다르게 회귀 성능에 복잡한 영향을 미치는 것으로 나타났습니다.



### Improving Mathematical Reasoning Capabilities of Small Language Models via Feedback-Driven Distillation (https://arxiv.org/abs/2411.14698)
- **What's New**: 이 논문에서는 Feedback-Driven Distillation (FDD) 프레임워크를 제안하여 Small Language Models (SLMs)의 수학적 추론 능력을 향상시키는 방법을 다룹니다. 이 접근법은 대형 언어 모델(Large Language Models, LLMs)에서 고급 추론 능력을 SLMs에 전달하기 위해 고안되었습니다. 데이터의 양과 질을 동시에 고려하면서 문제를 난이도에 따라 쉽게 또는 어렵게 분류하고 이에 따라 새로운 질문을 창출하여 증강된 데이터셋을 구축합니다.

- **Technical Details**: FDD는 세 가지 주요 단계로 구성됩니다: 초기화 단계에서 LLM에 수학 문제와 대응하는 추론 합리성을 짝짓게 하여 초기 데이터셋을 구축합니다. 그런 다음 SLM의 성능에 따라 문제를 난이도로 분류한 후, 각 유형에 맞게 추가 질문을 생성하여 데이터셋을 확장합니다. 마지막으로 다단계 증류(paradigm) 방식을 통해 이를 반복적으로 수행하여 SLMs의 수학적 추론 능력을 점진적으로 스타일로 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 FDD 프레임워크를 통해 SLMs는 수학적 추론 작업에서 최신 성능(state-of-the-art performance)을 달성할 수 있음을 입증했습니다. 또한, 이 방법은 SLMs의 도메인 내(in-domain) 수학적 추론 성능을 향상시킬 뿐만 아니라 도메인 외(out-of-domain)에서도 성능을 크게 향상시킨다는 것을 보여주었습니다.



### Quantum Hamiltonian Descent for Graph Partition (https://arxiv.org/abs/2411.14696)
- **What's New**: 본 연구에서는 그래프 분할 문제를 해결하기 위한 새로운 접근법인 Quantum Hamiltonian Descent (QHD)를 소개합니다. 그래프 분할을 Quadratic Unconstrained Binary Optimization (QUBO) 문제로 재구성함으로써 QHD의 양자에서 영감을 받은 동역학을 활용하여 최적의 커뮤니티 구조를 식별합니다. 이 방법은 QUBO 형식화와 QHD 최적화를 번갈아 수행하여 파티션 품질을 반복적으로 향상시키는 다단계 세분화 전략을 적용합니다.

- **Technical Details**: QHD는 전통적인 경량 경량화 방법과 차별되며, 경량화 알고리즘의 연속 시간 한계를 경로 적분 양자화하는 방식을 통해 도입되었습니다. QHD의 동적 시스템은 해밀토니안에 의해 구동되는 양자 진화로, 이는 모든 가능한 경로를 동시에 고려함으로써 지역 최솟값에서 벗어날 수 있는 양자 터널링 효과를 사용합니다. 이를 통해 QHD는 비볼록 최적화 문제를 해결하는 데 있어서 기존의 경량 기반 방법 및 양자 아디아바틱 알고리즘보다 우수한 성능을 나타냅니다.

- **Performance Highlights**: 실험 결과, QHD 기반 접근법은 기존 최적화 방법에 비해 계산 비용을 줄이며 최대 5.49% 향상된 모듈성 점수를 달성했습니다. 작은 문제 인스턴스에서는 GUROBI의 성능과 일치하며, 변수 수가 1,000을 초과하는 더 큰 문제에서는 GUROBI를 초월하여 이점을 보였습니다. 이러한 성능 우위는 대규모 그래프를 다루는 실용적 응용에 있어 중요한 제약 요인을 해결하는 데 기여합니다.



### Cross Group Attention and Group-wise Rolling for Multimodal Medical Image Synthesis (https://arxiv.org/abs/2411.14684)
- **What's New**: 본 논문에서는 멀티모달 MR 이미지 합성을 위한 Adaptive Group-wise Interaction Network (AGI-Net)를 제안합니다. AGI-Net은 모달리티 간 및 내부 모달리티 간의 관계를 탐구하여 공간적 불일치 문제를 해결하며, 여러 입력 이미지들로부터의 정보를 효과적으로 융합합니다. 이를 통해 기존 방법들에 비해 성능 개선을 이루었고, IXI 및 BraTS2023 데이터셋에서 최첨단의 성능을 달성했습니다.

- **Technical Details**: AGI-Net의 핵심 설계는 Cross Group Attention과 Group-wise Rolling (CAGR) 모듈로 이루어져 있습니다. Cross Group Attention은 그룹 내 및 그룹 간의 관계를 설정하여 입력 특징에서 모달리티 간의 혼합 노이즈를 억제합니다. Group-wise Rolling은 각 그룹에 대한 컨볼루션 커널의 독립적인 적응형 롤링을 가능하게 하여, 부분적인 불일치 조건 하에서도 효과적으로 모달리티 간의 관계를 캡처합니다.

- **Performance Highlights**: AGI-Net은 IXI 및 BraTS2023 데이터셋을 사용해 평가되었으며, 기존의 프레임워크에 롤링 커널 모듈을 교체함으로써 새로운 최첨단 결과를 달성했습니다. 이러한 효과는 AGI-Net이 입력 모달리티 간의 공간적 관계를 잘 포착하고 통합할 수 있도록 돕기 때문에 가능합니다. 또한, AGI-Net은 플러그 앤 플레이 형태로 기존 컨볼루션 레이어를 대체할 수 있어 다양한 응용 가능성을 가지고 있습니다.



### Multiverse of Greatness: Generating Story Branches with LLMs (https://arxiv.org/abs/2411.14672)
Comments:
          12 pages, 14 figures

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)과 상호작용하여 그래프 기반 콘텐츠를 생성하는 새로운 프레임워크인 DCP/P(동적 컨텍스트 프롬프트/프로그래밍)를 제안합니다. 기존의 연구에서는 시각 소설 게임을 생성하는 데 있어 수작업을 많이 요구하였고, 긴 일관된 이야기를 생성하는 데 유연성이 부족했습니다. DCP/P는 LLM에게 적절한 맥락을 제공함으로써 이전 연구의 한계를 극복하고 있습니다.

- **Technical Details**: DCP/P는 동적 프로그래밍과 그래프 탐색 알고리즘을 활용하여 LLM과 상호작용하는 새로운 접근 방식을 소개합니다. 이 프레임워크는 사용자 개입을 최소화하며, LLM이 주인공, 장소 및 선택지를 결정하도록 허용합니다. 또한 각 게임 생성에 필요한 이미지 자산을 생성하기 위해 이미지 생성 모델을 통합하였습니다.

- **Performance Highlights**: DCP/P를 사용하여 생성된 게임들은 기준선보다 일관되게 우수한 성과를 보였습니다. 연구진은 20개 스토리의 객관적인 평가를 통해 DCP/P의 효과성을 입증하였고, 생성된 콘텐츠의 질과 편향도 평가하여 LLM의 특정 단어 및 감정 편향 경향을 확인하였습니다. 모든 연구 코드와 프레임워크는 오픈소스로 제공되어 향후 연구를 지원할 수 있도록 하였습니다.



### Comparative Analysis of Pooling Mechanisms in LLMs: A Sentiment Analysis Perspectiv (https://arxiv.org/abs/2411.14654)
Comments:
          4 figures

- **What's New**: 이번 논문은 BERT와 GPT와 같은 대규모 언어 모델(Large Language Models, LLMs)의 풀링 메커니즘이 문장 수준의 감정 분석에 미치는 영향을 조사합니다. 전통적인 Mean, Max 및 Weighted Sum 풀링 전략의 성능을 비교하기 위한 포괄적인 실험이 수행되었습니다. 연구의 결과는 다양한 작업 요구 사항에 따라 각 풀링 메커니즘이 서로 다른 장점과 단점을 가지고 있음을 강조합니다.

- **Technical Details**: Transformer 기반 아키텍처의 설계 핵심은 어텐션 메커니즘으로, 이는 모델이 시퀀스 데이터를 처리할 때 서로 다른 토큰의 중요도를 동적으로 가중할 수 있게 합니다. 본 연구에서는 주요 풀링 전략으로 Mean, Max 및 Weighted Sum 풀링을 탐구하며, 각 기법의 특성과 상호작용을 분석합니다. 이를 통해 문장 수준 표현을 생성하기 위한 다양한 풀링 방법의 기초적인 수학적 설명도 제공됩니다.

- **Performance Highlights**: BERT 및 GPT 모델에 대한 감정 분석 실험 결과, 각 풀링 방법은 특정 상황에서 상이한 성능을 보여주며, 이는 특정 작업에 대한 최적의 해결책을 찾는 데 중요한 인사이트를 제공합니다. 특히, 어떤 풀링 방법이 특정 유형의 감정 분석에 보다 효과적인지를 이해함으로써 LLM 기반 모델의 향후 최적화 방향에 기여할 수 있습니다. 연구는 LLM 기반 모델을 특정 작업의 요구 사항에 맞게 조정하는 방법에 대한 실용적인 권장 사항을 포함합니다.



### Social Media Algorithms Can Shape Affective Polarization via Exposure to Antidemocratic Attitudes and Partisan Animosity (https://arxiv.org/abs/2411.14652)
- **What's New**: 이 연구는 소셜 미디어 피드 알고리즘이 정치적 양극화를 유발할 수 있다는 우려에 대응하기 위해, 최신의 대형 언어 모델(LLM)을 활용하여 실시간으로 피드를 재정렬하는 방법을 개발했습니다. 이를 통해 반민주적 태도와 당파적 적대감(AAPA)과 같은 콘텐츠의 영향을 실험할 수 있게 되었습니다. 연구 결과는 AAPA 콘텐츠의 노출이 감소할 때, 반대 정당에 대한 긍정적인 감정이 증가하였고, 노출이 증가할 경우 부정적인 감정이 증대됨을 보여주었습니다.

- **Technical Details**: 본 연구는 1,256명의 참가자를 대상으로 하는 10일간의 현장 실험을 실시하여, AAPA의 노출을 조절함으로써 참가자들의 정치적 감정 변화를 측정했습니다. 참가자들은 웹 확장을 설치한 후, X(트위터) 웹사이트에 접속하여 실시간으로 피드가 재정렬되는 경험을 하였습니다. 피드는 AAPA 요소의 표현 정도에 따라 평점이 매겨지고, 4개 이상의 요소를 반영한 포스트는 AAPA로 간주되었습니다.

- **Performance Highlights**: 개입 이후, AAPA의 노출이 감소한 참가자들은 정치적 아웃그룹에 대한 감정이 따뜻해졌고, 반대로 AAPA의 노출이 증가한 참가자들은 차가운 감정을 경험했습니다. 그러나 전통적인 참여 지표인 재게시 및 즐겨찾기 비율에 대해서는 통계적으로 유의미한 변화가 없었습니다. 이 결과는 소셜 미디어 피드 알고리즘의 조정이 affective polarization를 완화할 수 있는 가능성을 제시합니다.



### Evaluating Representational Similarity Measures from the Lens of Functional Correspondenc (https://arxiv.org/abs/2411.14633)
- **What's New**: 이 연구는 신경과학(neuroscience)과 인공지능(AI) 분야에서 고차원 신경 데이터(high-dimensional neural data)를 비교하는 데 중요한 도구를 찾고자 하며, 다양한 측정(metric)을 평가하여 특정 행동(behavioral outcomes)과의 일치도를 분석합니다. 특히, 제안된 방법론을 통해 훈련된 모델과 훈련되지 않은 모델을 효과적으로 구분할 수 있는지 알아봅니다. 또한, 여러 대표적 유사성 측정치가 행동 평가와 얼만큼 잘 일치하는지를 분석하여 신경 AI 분야에서의 지침을 제공합니다.

- **Technical Details**: 연구자는 여러 유사성 측정 방안을 비교하고, 정렬 기반, Canonical Correlation Analysis (CCA) 등 다양한 방법론을 통해 내부 표현(internal representations) 사이의 차이를 분석합니다. 이 과정에서 linear Centered Kernel Alignment (CKA)와 Procrustes 거리 같은 측정 방식이 훈련된 모델과 훈련되지 않은 모델을 구별하는 데 유능하다는 것을 발견하였습니다. 반면, linear predictivity는 신경과학에서 사용되는 방식 중 하나지만 행동과의 일치도가 낮은 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 각각의 측정 방식이 모델 구분에 있어 다르게 작용하며, 행동 메트릭(behavioral metrics)과의 일치도가 전반적으로 더 높은 경향을 보였습니다. 특히 linear CKA와 Procrustes 거리가 행동 평가와 가장 잘 일치하는 것으로 나타났고, 이는 NeuroAI 연구에서 행동적으로 의미 있는 비교를 강조할 수 있는 기초 자료를 제공합니다. 이러한 통찰은 향후 신경 AI 연구에서 유용할 것입니다.



### Generative AI for Music and Audio (https://arxiv.org/abs/2411.14627)
Comments:
          PhD Dissertation

- **What's New**: 이번 연구는 Generative AI가 음악과 오디오 생성에서의 혁신적인 접근 방식을 제시하고 있습니다. 특히, 다중 트랙 음악 생성, 보조 음악 제작 도구, 오디오 및 음악을 위한 멀티모달 학습이라는 세 가지 주요 방향을 다루고 있습니다. 이 연구는 콘텐츠 생성의 접근성을 높이기 위한 것입니다.

- **Technical Details**: 연구의 주요 초점은 AI가 어떻게 전문가와 아마추어가 음악 및 오디오 콘텐츠를 생성하는 데 도움이 될 수 있는지를 탐구하는 것입니다. 또한, AI가 인간이 음악을 배우는 방식과 유사하게 음악을 생성할 수 있는지를 탐구하고 있습니다. 이러한 접근은 Generative AI 기술의 진화와 음악 작곡의 민주화를 목표로 하고 있습니다.

- **Performance Highlights**: 연구의 궁극적인 목표는 음악 작곡에 대한 장벽을 낮추고 오디오 콘텐츠 생성의 민주화를 이끄는 것입니다. Generative AI 기술을 활용하여 사용자 친화적인 음악 생성 도구의 가능성을 모색하며, 다양한 매체에서 오디오 콘텐츠 생성의 변화를 예고하고 있습니다.



### Unveiling the Hidden: A Comprehensive Evaluation of Underwater Image Enhancement and Its Impact on Object Detection (https://arxiv.org/abs/2411.14626)
- **What's New**: 이 연구는 수중 이미징의 품질 개선을 목표로 하는 최신 이미지 향상 모델들을 검토하고, 이러한 모델들이 수중 물체 탐지 성능에 미치는 영향을 조사했습니다. 두 가지 주요 데이터셋(RUOD, CUPDD)을 이용해 향상된 이미지의 정성적 및 정량적 분석을 수행하였고, 향상된 이미지와 원본 이미지의 품질 분포를 비교하기 위한 Quality Index(Q-index)를 개발했습니다 또한 YOLO-NAS 탐지 모델의 성능을 비교하여 향상 전후의 탐지 성능 변화를 분석했습니다.

- **Technical Details**: 본 연구는 수중 이미지 품질 저하의 원인과 이를 개선하기 위한 향상 기법을 quantitatively 및 qualitatively 연결하여 살펴보았습니다. RUOD와 CUPDD의 대표적인 이미지 향상 모델을 활용해 실험을 진행하였고, YOLO-NAS 탐지 모델을 기반으로 탐지 성능을 분석했습니다. 이 과정에서 Reference-free metrics(예: UIQM, UCIQE 등)를 활용하여 이미지 품질을 평가했습니다.

- **Performance Highlights**: 연구 결과, 일반적으로 이미지 향상이 탐지 성능을 저하시키는 경향이 있지만, 특정 경우에는 탐지 성능이 향상되는 경과를 보였습니다. 탐지 성능을 저하시킨 향상된 이미지의 분석을 통해, 인간 주석자가 놓친 객체를 발견하는 등 향상이 긍정적 결과를 초래하는 사례도 밝혀졌습니다. 이 연구는 시각적 품질 및 물체 탐지 성능 간의 관계를 명확히 하여 향상 기법의 활용 가능성을 보여주었습니다.



### Predictive Analytics of Air Alerts in the Russian-Ukrainian War (https://arxiv.org/abs/2411.14625)
- **What's New**: 이번 연구는 2022년 2월 24일에 시작된 러시아-우크라이나 전쟁 중 공습 경고에 대한 탐색적 데이터 분석과 예측 분석 접근법을 다루고 있습니다. 공습 경고가 서로 상관관계를 가지며 지리적으로 패턴이 있음을 보여주어, 특정 지역에서 발생할 경고를 예측할 수 있는 모델을 구축 가능성을 제시합니다. 이러한 결과는 인접 지역의 경고 상태가 서로 밀접하게 연관되어 있음을 드러내며, 계절성 특징이 목표 변수를 예측하는 데에 중요하다는 사실도 강조합니다.

- **Technical Details**: 연구에서 사용된 데이터는 지난 공습 경고에 대한 역사적 데이터이며, 다양한 지역의 공습 경고 기간을 분석하고 시각화하여 지리적 패턴을 탐지하였습니다. 각각의 지역에 대해 별도의 예측 모델을 구축하기 위해 Random Forest 알고리즘을 사용해 5분 및 15분 시간 범위로 예측을 실시했습니다. 데이터를 훈련용 및 검증용으로 나누어 500회 반복하여 모델을 훈련했습니다.

- **Performance Highlights**: 모델의 성능을 평가하기 위해 ROC 곡선 및 정확도 점수를 이용해 결과를 분석했습니다. 예측 변수와 인접 지역의 특성이 더욱 중요한 역할을 한다는 것입니다. 공습 경고 패턴이 시간에 따라 변동함을 관찰했으며, 지역 간 공습 경고의 시간 시계열은 상관성이 높다는 결과를 도출했습니다.



### Exploiting Boosting in Hyperdimensional Computing for Enhanced Reliability in Healthcar (https://arxiv.org/abs/2411.14612)
Comments:
          Accepted to DATE 2025

- **What's New**: BoostHD라는 새로운 접근법이 하이퍼차원 컴퓨팅(Hyperdimensional Computing, HDC)에서 부스팅 알고리즘을 적용하여 하이퍼차원 공간을 하위 공간으로 분할하는 방법을 제안했습니다. 이 방법은 성능과 신뢰성을 향상시켜 기존 HDC 방법을 초월하는 효과를 보여줍니다. 특히, 헬스케어와 같은 분야에서 데이터가 제한된 상황에서도 안정적이고 일관된 성능을 제공하는 것이 중요합니다.

- **Technical Details**: BoostHD는 OnlineHD 모델을 기반으로 하여 하이퍼차원 공간을 n개의 약한 학습자에게 D/n 차원으로 나누는 새로운 분할 전략을 제공하여, 각 약한 학습자가 특정한 하위 공간을 담당하도록 구성합니다. 이 방식을 통해 각 약한 학습자가 공간을 효율적으로 활용할 수 있도록 하여, 노이즈와 과적합에 대한 견고성을 확보합니다. 후에 각 약한 학습자의 성능을 보장할 수 있는 조건에서, BoostHD는 OnlineHD의 기능을 증대시킵니다.

- **Performance Highlights**: BoostHD는 WESAD 데이터셋에서 98.37%의 정확도로 랜덤 포레스트, XGBoost 및 OnlineHD 등 기존의 최신 방법들을 초과하는 성능을 달성했습니다. 이는 데이터 불균형 및 노이즈 환경에서도 높은 정확도를 유지하며, 개인별 평가에서도 평균 96.19%의 정확도를 기록하였습니다. 이와 같이 BoostHD는 헬스케어와 같은 신뢰성과 정확성이 중요한 도메인에서 HDC의 적용 가능성을 크게 확장합니다.



### A Systematic Study of Multi-Agent Deep Reinforcement Learning for Safe and Robust Autonomous Highway Ramp Entry (https://arxiv.org/abs/2411.14593)
Comments:
          9 pages, 9 figures

- **What's New**: 이 논문은 고속도로 진입로에서 완전 자율주행을 위한 시스템적 연구를 소개합니다. 이 연구는 자율주행차가 보다 안전하게 고속도로 흐름에 합류하도록 돕기 위해 다중 에이전트를 활용한 게임이론적 접근 방식을 채택했습니다. 이러한 접근법은 심층 강화 학습(Deep Reinforcement Learning, DRL)을 기반으로 하며, 차량이 평행하게 합류하는 상황에서 안전한 주행을 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 연구는 경량 고속도로 합류 과정을 자율적으로 수행할 수 있는 시스템을 개발하기 위해 더 복잡한 차량 상호작용을 다룹니다. 기존의 두 대 이상의 차량이 포함된 시뮬레이션 환경을 조성하여, 추가적인 차량들과의 상호작용을 통해 임상적 최적화된 제어기(controller) 학습을 수행하였습니다. 이를 통해, 학습된 제어기는 이상적인 최적 제어기와 비교할 때 매우 가까운 성능을 나타냄을 확인했습니다.

- **Performance Highlights**: 이 연구 결과는 자율주행차의 안전성과 효율성을 높일 수 있는 가능성을 제시하고 있습니다. 실제 시뮬레이션 경험을 통해, 시스템이 어떻게 두 대 이상의 차량이 밀집한 환경에서도 충돌 없이 안전하게 합류할 수 있는지에 대한 실증적 증거를 제공합니다. 이러한 연구는 L4 및 L5 자율주행차의 상용 배치 및 도로 운영에 중대한 영향을 미칠 것으로 기대됩니다.



### G-RAG: Knowledge Expansion in Material Scienc (https://arxiv.org/abs/2411.14592)
- **What's New**: 이번 연구에서는 Material Science 분야의 정보 검색 시스템에서 발생하는 여러 문제를 해결하기 위한 Graph RAG 방법론을 제안합니다. 기존의 Retrieval-Augmented Generation (RAG) 접근 방식의 한계인 구식 정보, 허위 정보, 해석 가능성의 부족 등을 극복하기 위해, 그래프 데이터베이스를 통합하여 키 엔티티를 활용한 정교한 정보 검색을 가능하게 합니다. 이러한 방법은 문서 간의 관계를 캡처하여 정보 검색의 정확성과 맥락 이해를 개선하는 데 기여합니다.

- **Technical Details**: 이 연구에서 제안된 Graph RAG는 MatID라는 키 엔티티를 문장에서 추출하고, 이 정보를 외부 위키피디아 데이터베이스에 쿼리하여 관련 정보를 검색함으로써 향상된 응답 생성을 구현합니다. G-RAG 시스템은 엔티티 링크와 관계 추출 기능을 포함하며, 구문 분석 모듈을 통해 정보 검색을 수행합니다. 이 구조는 LLM(대형 언어 모델)에서 더 깊은 의미적 이해를 가능하게 합니다.

- **Performance Highlights**: G-RAG 접근 방식은 Material Science와 같은 정확한 정보 검색이 중요한 도메인에서 성능을 크게 향상시키는 것으로 나타났습니다. 특히, 정보의 정확성과 관련성을 유지하면서 LLM이 생성하는 응답의 질을 크게 개선하여, 복잡한 데이터를 효과적으로 관리할 수 있는 능력을 발휘합니다. 또한, 다양한 자연어 처리(NLP) 작업에서도 기존 기술들보다 우수한 성과를 나타내는 것으로 평가받고 있습니다.



### Assessment of LLM Responses to End-user Security Questions (https://arxiv.org/abs/2411.14571)
Comments:
          18 pages, 1 figure, 8 tables

- **What's New**: 이번 연구는 LLM(large language models)들이 사용자 보안 질문에 대한 응답 품질을 평가하기 위해 수행된 첫 번째 광범위한 연구입니다. 이전 연구들은 주로 보안 신화를 반박하는 LLM의 능력에 초점을 맞춘 반면, 본 연구는 LLM의 보안 문제에 대한 폭넓은 질문들을 다루었습니다. 이 연구는 사용자와 모델 개발자 모두에게 실질적인 개선 방향과 전략을 제시하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 GPT, LLaMA, Gemini라는 세 가지 LLM 모델이 생성한 1,244개의 보안 관련 응답을 정성적으로 평가했습니다. 이 과정에서 각 모델의 응답 품질, 정확성, 완전성 및 관련성 뿐만 아니라 정보 전달 스타일을 분석했습니다. 특히, 직접성이 떨어지고 관련되지 않은 응답을 제공하는 경향을 보였으며, 이는 사용자와 LLM 간의 효과적인 소통을 저하시킬 수 있습니다.

- **Performance Highlights**: LLM들은 일반적인 보안 질문에 대해 사용자 친화적인 고품질 정보를 제공하는 데 성공했습니다. 그러나 응답의 오류와 제한들이 지속적으로 발견되었으며, 이는 사용자가 LLM을 보다 효과적으로 활용하는 데 장애가 됩니다. 특히 구식의 패스워드 권장 사항이나, 피싱 및 HTTPS URL 간의 연관을 잘못 연결하는 등의 문제가 보고되었습니다.



### The importance of the clustering model to detect new types of intrusion in data traffic (https://arxiv.org/abs/2411.14550)
Comments:
          18 pages, 4 figures

- **What's New**: 현재 디지털 시대에서 생성되는 데이터의 양이 엄청나게 증가하고 있으며, 이는 사이버 보안 조치를 개선하는 데 활용될 수 있습니다. 이 연구는 비정형 데이터 분석의 어려움을 해결하기 위해 클러스터링(clustering) 기법을 도입하여 데이터 내에서 숨겨진 패턴을 식별하는 방법을 제안합니다. 특히 K-means 알고리즘을 사용하여 새로운 공격 유형을 식별하고 라벨링하는 과정이 강조되었습니다.

- **Technical Details**: 클러스터링은 데이터 마이닝(data mining)의 한 접근 방법으로, 유사도 계산을 통해 데이터 세트를 여러 카테고리로 나누는 방식을 사용합니다. 본 연구에서는 Kali Linux 환경과 다양한 공격을 통해 수집된 데이터를 활용하였습니다. 또한 Kaggle의 'Internet of Things Network에 대한 침입 탐지' 데이터 세트에서 동일한 방식으로 클러스터링 모델을 적용하여 결과를 확인하였습니다.

- **Performance Highlights**: K-means 알고리즘을 통해 수집한 데이터에서 새로운 공격 유형을 효과적으로 식별할 수 있었으며, 클러스터링 모델은 공격의 수를 정확히 감지하였습니다. 결과 섹션에서 이 모델의 성능이 잘 나타나 있으며, 이는 사이버 위협의 동적 특성에 따라 새로운 공격 유형에 대한 라벨 데이터가 부족한 경우에도 유용성을 보여줍니다.



### Open Challenges in the Formal Verification of Autonomous Driving (https://arxiv.org/abs/2411.14520)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 이 논문에서는 자율주행 시스템의 복잡성과 이종 시스템 통합의 실제 사례를 제시합니다. 기존의 모노리식(monoilithic) 시스템에서 벗어나 다양한 하드웨어와 소프트웨어 구성 요소가 조화를 이루어야 함을 강조합니다. 또한, 다양한 기업에서 개발한 독립적인 구성 요소들이 상호작용하면서 발생하는 인증 문제를 다룹니다.

- **Technical Details**: 자율주행 차량은 여러 개의 독립적인 부품들로 구성되며, 이들 부품은 종종 블랙박스(black-box) 형태로 내부 동작을 공개하지 않습니다. 이러한 시스템 내에서 각 구성 요소의 신뢰성과 안전성을 보장하기 위한 포멀 검증(formal verification) 기법의 중요성을 논의합니다. 논문은 이종 컴포넌트의 통합 및 인증 과정을 개선하기 위한 다양한 기술적 접근 방안을 탐구합니다.

- **Performance Highlights**: 실제 사례 연구를 통해 자율주행 시스템의 신뢰성과 안전성을 확보하는 데 필요한 주요 도전 과제를 파악했습니다. 포멀 검증 기법을 활용하는 것이 이러한 과제들을 극복할 수 있는 효과적인 방법임을 확인했고, 이는 시스템의 전반적인 성능 향상에도 기여할 것으로 기대됩니다.



### Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection (https://arxiv.org/abs/2411.14515)
Comments:
          Under review

- **What's New**: 이 논문에서는 기존의 이분법적(anomalies in a binary manner) 접근 방식을 넘어 실제 응용에서의 심각도를 반영하는 다층적 이상 탐지(Multilevel Anomaly Detection, MAD)를 제안합니다. 이 새로운 설정에서 이상 점수(anomaly score)는 이상 현상의 심각도를 나타내며, 다양한 분야에서의 응용 가능성이 강조됩니다. 또한, 새로운 벤치마크인 MAD-Bench를 도입하여 모델의 이상 탐지 능력과 심각도 정렬 점수 부여의 효과성을 동시에 평가합니다.

- **Technical Details**: MAD 설정에서는 데이터 포인트가 L0에서 Ln까지의 심각도를 구분할 수 있습니다. 이로 인해 훈련 세트는 정상 데이터에만 제한되고, 테스트 세트는 여러 심각도 수준의 이상을 포함합니다. 이와 함께, 모델 성능 평가를 위해 기존 데이터셋을 다양한 도메인에서 다층적 이상 탐지 맥락으로 조정했으며, 다중모달 대형 언어 모델(Multimodal Large Language Model, MLLM) 기반의 새로운 기준을 포함했습니다.

- **Performance Highlights**: MAD-Bench를 통해 다양한 모델의 성능 분석을 실시하였고, 모델들이 심각도에 맞는 점수를 부여하는 능력을 평가했습니다. 이 분석은 모델의 이분법적(binary) 탐지와 다층적(multilevel) 탐지 성능 간의 상관관계 및 견고성을 조사합니다. 결과적으로, 이 연구는 실제적인 심각도 정렬을 위한 독창적인 AD 모델 개선 방안을 제시합니다.



### Variational Autoencoders for Efficient Simulation-Based Inferenc (https://arxiv.org/abs/2411.14511)
- **What's New**: 이 논문에서는 likelihood-free 시뮬레이션 기반 추론을 위한 변분 추론(variational inference) 접근 방식을 제안합니다. 이 방법은 변분 오토인코더(variational autoencoder) 내의 잠재 변수(latent variables)를 활용하여 복잡한 사후 분포(posterior distributions)를 효율적으로 추정합니다. 두 가지 변형 모델을 통해 사전 분포(prior distribution)를 처리하는 방식을 탐구할 것이며, 이를 통해 더 나은 일반화(generalization)를 달성합니다.

- **Technical Details**: 제안된 접근 방식은 Conditional Variational Autoencoder (C-VAE) 아키텍처를 기반으로 하며, 기존의 GAN(Generative Adversarial Network) 알고리즘의 불안정성과 비효율성을 피합니다. C-VAE는 다양한 신경망 아키텍처의 통합을 용이하게 하면서도 복잡한 데이터 구조를 처리할 수 있는 강력한 기능을 제공합니다. 이러한 접근 방식은 외부 요약 네트워크 없이도 복잡한 데이터 및 종속성을 효과적으로 관리할 수 있습니다.

- **Performance Highlights**: 논문은 제안된 모델들이 기존의 벤치마크 문제들에서 효과적임을 입증하며, 이전의 흐름 기반(flow-based) 접근 방식과 비교해도 유사한 성과를 달성하지만, 계산 효율성과 확장성에서 우위를 점하고 있음을 강조합니다. 네 가지 기존의 SBI 방법과의 비교를 통해 효율성과 확장성을 중점을 두어 논의하였습니다.



### FuseGPT: Learnable Layers Fusion of Generative Pre-trained Transformers (https://arxiv.org/abs/2411.14507)
- **What's New**: FuseGPT는 감축된 transformer 블록을 재활용하여 모델 성능을 회복할 수 있는 혁신적인 방법론을 제안합니다. 기존의 모델 압축 기법들이 제거 후의 성능 저하 문제를 해결하지 못했던 반면, FuseGPT는 Macro Influence(MI)라는 새로운 메트릭을 도입하여 블록의 중요성을 평가합니다. 이 방법은 불필요한 블록의 파라미터를 인접 블록의 레이어에 주입하여 블록 복원 과정을 통해 성능 회복을 가능케 합니다.

- **Technical Details**: FuseGPT에서는 transformer 블록의 중요도를 평가하기 위해 블록 제거 후 정보 손실을 계산하여 장기적인 영향력을 측정합니다. 이 방법론은 가벼운 그룹 수준의 파인튜닝을 통해 인접 블록의 해당 레이어에 불필요한 블록의 파라미터를 반복적으로 주입하고 최적화합니다. 이를 통해 블록의 이미 존재하는 지식을 활용하여 성능 회복을 극대화할 수 있습니다.

- **Performance Highlights**: FuseGPT는 적은 양의 데이터를 사용하여 perplexity와 zero-shot 과제 성능 모두에서 이전 연구들을 초월하는 결과를 보였습니다. 특히 대형 언어 모델뿐 아니라 대형 다중 모달 모델에서도 효과적으로 작용하여 뛰어난 성능을 발휘하고 있습니다. 실험 결과, FuseGPT는 강력한 차세대 성능을 기록하였으며, 향후 모델 압축 연구에서 유망한 방향성을 제시하고 있습니다.



### Planning-Driven Programming: A Large Language Model Programming Workflow (https://arxiv.org/abs/2411.14503)
- **What's New**: 이번 논문에서는 LLM 프로그래밍 워크플로우(LLM Programming Workflow, LPW)를 제안하여 코드 생성 및 수정의 효율성을 대폭 향상시키고자 합니다. LPW는 문제를 관리 가능한 소문제로 분해하고 가시적 테스트 케이스를 통해 해결책을 검증하는 구조화된 두 단계의 워크플로우를 적용합니다. 이 방식은 코드 생성의 정확성과 품질을 높이는 데 기여하며, 실제 프로그래밍 과정에서 발생할 수 있는 한계를 극복하는 것을 목표로 하고 있습니다.

- **Technical Details**: LPW는 솔루션 생성 단계와 코드 구현 단계로 나뉘며, 각 단계에서 LLM이 생성한 솔루션 플랜과 그 검증을 사용하여 초기 코드를 개발합니다. 특히 솔루션 플랜 작성 시, 복잡한 문제를 여러 개의 중간 단계로 나누어 접근하며, 이 과정을 통하여 더욱 논리적이고 일관된 코드 생성을 도모합니다. SLPW는 LPW의 샘플링 변형으로, 여러 솔루션 플랜과 그 검증을 초기 생성 후 상황에 따라 최적화하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, LPW는 기존의 텍스트-투-코드 생성 벤치마크에서 Pass@1 정확성을 최대 16.4% 향상시켰으며, 특히 어려운 벤치마크에서 약 10%의 주요 개선을 보였습니다. SLPW는 LPW보다 최대 5.6% 향상된 결과를 나타내며, 다양한 벤치마크에서 새로운 최첨단 Pass@1 정확도를 기록했습니다. 예를 들어, GPT-4o를 기반으로 할 때 HumanEval에서 98.2%, MBPP에서 84.8%, APPS에서 64.0%, CodeContest에서 35.3%의 높은 정확성을 달성했습니다.



### Global Challenge for Safe and Secure LLMs Track 1 (https://arxiv.org/abs/2411.14502)
- **What's New**: 이번 논문은 AI Singapore (AISG)와 CyberSG R&D Programme Office (CRPO)가 주최하는 안전하고 보안이 강화된 대형 언어 모델(LLMs) 글로벌 챌린지를 소개합니다. 이 혁신적인 이니셔티브는 자동화된 jailbreaking 공격에 대한 방어 메커니즘 개발을 촉진하는 것을 목표로 합니다. LLMs가 의료, 금융, 공공 행정과 같은 중요한 분야에 통합됨에 따라, 이러한 모델의 악의적 공격에 대한 저항력을 보장하는 것이 윤리적 기준 유지를 위해 필수적입니다.

- **Technical Details**: 이 대회는 LLM 보안 프레임워크의 강건성을 평가하고 향상시키기 위해 두 가지 별도의 트랙으로 구성되었습니다. 트랙 1에서는 참가자들이 LLM의 취약점을 탐지하기 위해 바람직하지 않은 반응을 유도하는 자동화된 방법을 개발하도록 하는 과제가 주어졌습니다. 이 과정에서, 참가자들은 폭언, 허위 정보, 불법 활동 등 다양한 시나리오에서 콘텐츠 보호 조치를 우회할 수 있는 기술을 구상해야 했습니다.

- **Performance Highlights**: 트랙 1을 통해 LLM의 취약성에 대한 이해를 심화하고, 더 강한 모델을 만들기 위한 통찰력을 제공하는 것을 목표로 했습니다. 이 챌린지는 기존의 안전 프로토콜의 한계를 효과적으로 테스트하며, LLM의 보안 강화를 위한 혁신적인 방법론 개발을 장려합니다. 궁극적으로 이러한 연구는 LLM의 안전성을 높이고 윤리적 사용을 촉진하는 데 기여할 것으로 기대됩니다.



### Exploring Accuracy-Fairness Trade-off in Large Language Models (https://arxiv.org/abs/2411.14500)
Comments:
          9 pages

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 정확성과 공정성을 조화롭게 향상시키는 복잡한 문제를 탐구합니다. 기존의 방법들은 종종 성능을 최적화하기 위해 공정성을 희생하게 되며, 이는 사회적인 피해를 초래할 수 있습니다. 우리는 LLM 훈련 과정을 다목적 학습(task)으로 재구성하고, 다목적 진화 학습(MOEL) 방법론을 통해 이 문제를 해결하기 위한 새로운 경로를 제시합니다.

- **Technical Details**: 해당 연구에서는 LLM의 불공정성을 완화하기 위한 기존 방법들을 네 가지 단계(전처리, 훈련 중, 처리 중, 후처리)로 나누어 설명합니다. 특히 MOEL은 여러 목표를 동시에 최적화할 수 있는 알고리즘으로, 파레토 최적 집합(Pareto-optimal set)을 생성하여 다양한 요구 사항을 충족할 수 있는 LLM 모델의 세트를 제공합니다. MOEL을 활용함으로써 더 나은 정확성 및 공정성을 동시에 달성할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 이 연구는 MOEL 방법론을 통해 공정성과 정확성을 동시에 최적화한 LLM의 구성요소를 제공하며, 다양한 트레이드오프를 통해 사용자에게 적합한 모델 선택의 유연성을 부여합니다. LLM의 공정성 문제에 대한 해결책을 제시함으로써 더 공정하고 효율적인 AI 기술의 발전을 도모하고 있습니다. 이러한 접근법은 LLM이 부정확하거나 불공정한 결과를 초래하는 위험을 줄이는 데 기여할 것으로 기대됩니다.



### Understanding World or Predicting Future? A Comprehensive Survey of World Models (https://arxiv.org/abs/2411.14499)
- **What's New**: 세계 모델(world models)의 개념은 GPT-4와 Sora와 같은 최신 멀티모달 언어 모델 및 비디오 생성 모델의 발전으로 주목받고 있습니다. 이 논문은 세계 모델 관련 문헌을 포괄적으로 검토하며, 세계 모델이 현재 세계 상태를 이해하거나 미래의 역동성을 예측하는 도구로 여겨진다고 설명합니다. 세계 모델의 구성과 기능을 체계적으로 분류하고, 자율주행, 로보틱스, 사회 시뮬레이터 등 주요 분야에서의 응용을 살펴보는 것이 중요하다고 강조합니다.

- **Technical Details**: 세계 모델의 정의는 일반적으로 두 가지 관점으로 나뉘며, 하나는 세계를 이해하는 것이고 다른 하나는 미래를 예측하는 것입니다. Ha와 Schmidhuber의 초기 연구는 외부 세계를 추상화하여 그 기제를 이해하는 데 초점을 맞췄습니다. 반면, LeCun은 세계 모델이 실제 세계를 인식하고 모델링할 뿐만 아니라, 의사결정을 위한 가능한 미래 상태를 구상할 수 있어야 한다고 주장했습니다. 이러한 이론적 배경을 통해 연구자들은 세계 모델의 내부 표현(internal representation)과 미래 예측(future prediction) 기능을 확립하였습니다.

- **Performance Highlights**: Sora 모델은 현실 세계의 시각 데이터를 입력받아 미래 세계의 진화를 예측하는 비디오 생성 모델로, 세계 시뮬레이터로서의 역할을 하고 있습니다. 이 모델은 카메라 움직임 중에도 3D 비디오 시뮬레이션의 일관성을 유지하며, 물리적으로 그럴듯한 결과를 생성하는 능력을 보여줍니다. 이러한 발전은 세계 모델의 효율성을 높이며, 다양한 실제 응용 프로그램에 적합한 방향으로 미래 연구의 방향성을 제시합니다.



### Star-Agents: Automatic Data Optimization with LLM Agents for Instruction Tuning (https://arxiv.org/abs/2411.14497)
- **What's New**: 이번 논문에서는 Star-Agents 프레임워크를 제안하여 대규모 언어 모델(LLMs)의 성능을 향상시키기 위한 새로운 데이터 최적화 방법론을 소개합니다. 이 프레임워크는 다양한 LLM 에이전트를 활용하여 데이터 품질을 자동으로 향상시키고, 이를 통해 더 높은 품질의 훈련 데이터를 생성하게 됩니다. 기존의 인스트럭션 튜닝 방식에서 인간 주도 데이터 생성의 한계를 극복하고자 하며, 최종적으로 모델의 성능을 12% 이상 향상시키는 결과를 보여줍니다.

- **Technical Details**: Star-Agents 프레임워크는 세 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 여러 고급 LLM 에이전트를 활용하여 다양한 인스트럭션 데이터를 생성합니다. 두 번째 단계는 이 데이터를 두 모델 평가 방식을 통해 분석하여 복잡성과 품질을 기준으로 고품질 샘플을 선택합니다. 마지막으로, 선택된 데이터의 복합 점수를 기반으로 에이전트-쌍의 샘플링 확률을 조정하여 데이터 품질과 다양성을 균형 있게 유지합니다.

- **Performance Highlights**: 본 연구에서 실시한 실험에서는 Pythia와 LLaMA 같은 LLM을 사용한 인스트럭션 튜닝 결과, Star-Agents 프레임워크로 생성된 데이터로 학습한 LLM이 기존의 Evol-Instruct나 IFD 메트릭 준수 데이터를 활용한 경우보다 더 나은 성능을 보였습니다. 특히, Fermi 지표에서 40%의 성능 개선이 이루어졌으며, 이는 입증된 벤치마크에서 관찰된 성과입니다.



### A Survey on Human-Centric LLMs (https://arxiv.org/abs/2411.14491)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 발전이 인간의 인지 및 행동 시뮬레이션에 미치는 영향을 다룹니다. 기존의 LLM들은 개인적 및 집단적 작업을 수행할 수 있는 능력을 성장시키며, 인간 행동을 모방하는 새로운 도구로 활용되고 있습니다. 연구는 LLM들이 인간의 사고, 의사결정 및 사회적 상호작용을 실제로 어떻게 재현하는지를 평가하고, 향후 연구 방향과 도전 과제를 제시합니다.

- **Technical Details**: 연구는 LLM의 인지적, 지각적, 사회적 능력을 평가하며, 이를 인간의 유사한 기술과 비교합니다. LLM이 특정 작업을 수행할 때 이들이 보여주는 강점은 체계적 추론, 패턴 인식, 창의력 등입니다. 그러나, 복잡한 다단계 로직 처리 및 감정적 공감에서는 한계를 보입니다. LLM을 인간 중심의 영역에 적용하는 방법과 이론적 프레임워크를 탐구합니다.

- **Performance Highlights**: LLM은 행동 과학, 정치 학, 사회학 등 다양한 분야에서 인간 행동을 모방하는 데 효과적입니다. 개별 및 집단적 맥락에서 LLM의 활용 가능성을 강조하며, 기본적인 프롬프트 기법과 다중 에이전트 접근 방식이 사용됩니다. 연구는 LLM의 실시간 적응력, 감정 지능, 문화적 민감성을 개선하고, 인간-AI 협업을 위한 보다 정교한 프레임워크 개발의 필요성을 강조합니다.



### GhostRNN: Reducing State Redundancy in RNN with Cheap Operations (https://arxiv.org/abs/2411.14489)
- **What's New**: 본 논문의 혁신적인 접근 방식은 GhostRNN이라고 불리는 새로운 효율적인 RNN 아키텍처를 제안한 점입니다. 이 모델은 숨겨진 상태(hidden state)의 중복성을 줄이고 계산 비용을 낮추기 위해 저렴한 연산에서 파생된 "유령 상태(ghost states)"를 활용합니다. 이를 통해 메모리 사용량을 약 40% 줄이면서도 성능을 유지할 수 있음을 보여 줍니다.

- **Technical Details**: GhostRNN은 GRU 구조를 기반으로 하며, 기존 모델의 중복성을 분석하여 이를 최소화합니다. 초기 내부 상태(intrinsic states)를 생성하고, 간단한 선형 변환(linear transformation)과 활성화 함수(activation function)를 사용하여 유령 상태(ghost states)를 생성합니다. 이러한 유령 상태는 기존의 숨겨진 상태와 결합되어 다음 계산 단계로 전달됩니다.

- **Performance Highlights**: KWS 및 SE 작업에서 실험 결과, GhostRNN은 Google Speech Commands 데이터셋에서 0.1%의 정확도 향상을 보이며, 기본 모델의 매개변수 수를 40% 줄였습니다. 또한 SE 작업에서는 SDR과 Si-SDR을 각각 0.1dB 향상시키며 약 40%의 압축률을 달성했습니다.



### Ensuring Safety and Trust: Analyzing the Risks of Large Language Models in Medicin (https://arxiv.org/abs/2411.14487)
- **What's New**: 이 논문의 주요 내용은 의학 분야에서 Large Language Models (LLMs)의 안전성 문제를 체계적으로 분석하고 다루는 것입니다. 특히, 의학 AI의 안전하고 신뢰성 있는 사용을 위해 진실성(Truthfulness), 회복력(Resilience), 공정성(Fairness), 강건성(Robustness), 및 프라이버시(Privacy)라는 다섯 가지 원칙을 제안하고, 이를 바탕으로 1,000개의 전문가 검증 질문을 포함한 MedGuard 벤치마크를 도입했습니다.

- **Technical Details**: 이 연구에서는 11개의 일반적으로 사용되는 LLM을 평가하여, 현재의 언어 모델들이 대부분의 벤치마크에서 낮은 성과를 보이고 있음을 발견했습니다. 이는 인간 의사의 높은 성과와 비교했을 때 더욱 두드러지며, 의학적 안전성과 관련된 메커니즘이 작동하더라도 상황이 크게 개선되지 않는다는 점을 보여줍니다.

- **Performance Highlights**: 이 논문은 최근 ChatGPT와 같은 고급 LLM들이 다양한 의학적 작업에서 인간 성과와 비슷하거나 이를 초월할 수 있다는 보고에도 불구하고, 여전히 안전성의 중요한 격차가 존재한다고 강조합니다. 이는 인간의 감독과 AI 안전 방어 장치의 필요성을 더욱 부각시키는 결과입니다.



### The Impossible Test: A 2024 Unsolvable Dataset and A Chance for an AGI Quiz (https://arxiv.org/abs/2411.14486)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 불확실성을 인식하는 능력을 평가하기 위한 새로운 평가 프레임워크를 제시합니다. 연구는 의도적으로 답이 알려지지 않은 대학원 수준의 질문 675개를 포함하는 자료를 사용하여, 12개의 최신 LLM 모델을 평가하였습니다. 이 프레임워크는 LLM이 부정확한 응답을 생성하기보다는 무지(ignorance)를 인정하는 경향에 중점을 두었습니다.

- **Technical Details**: 모델 평가에는 생물학, 철학, 수학 등 다양한 분야의 문제들이 포함되었으며, 전체 모델들은 문제 해결이 알려지지 않았음을 인정하는 정확도가 62%에서 68%에 이릅니다. 흥미롭게도 문제의 난이도와 모델의 정확도 간에는 역관계가 관찰되었습니다. 특히 GPT-4는 어려운 문제에 대해 35.8%의 불확실성 인식률을 나타내었으나, 더 쉬운 문제에서는 20.0%에 불과했습니다.

- **Performance Highlights**: 모델들은 창의성과 NP-완전 문제에서 불확실성을 인정하는 데 어려움을 겪었고, 철학 및 심리학 과제에서는 상대적으로 더 나은 성과를 보였습니다. 이러한 결과는 인공지능(AGI) 평가 연구의 중요성을 강조하며, LLMs가 자신의 지식 한계를 인식하는 데 현재의 한계를 나타낸다는 것을 보여줍니다. 연구는 모델 훈련 아키텍처와 평가 접근법을 개선하기 위한 새로운 방향성을 제시합니다.



### Mediating Modes of Thought: LLM's for design scripting (https://arxiv.org/abs/2411.14485)
Comments:
          Published at ACADIA 2024

- **What's New**: 이 논문에서는 Large Language Models (LLMs)를 사용해 디자인 프로세스에서 인간의 직관과 알고리즘의 논리를 연결하는 새로운 접근 방식을 제안합니다. 이러한 시스템이 자연어 프롬프트를 해석하여 지오메트리 오퍼레이션을 조합할 수 있는지 탐구합니다. 이 연구는 기계학습 원리를 활용한 디자인 도구의 잠재력을 강조하며, LLM이 사용자 의도를 유추하고 디자인 스크립트를 생성하는 능력을 검토합니다.

- **Technical Details**: 제안된 시스템은 다수의 LLM 에이전트를 사용해 사용자의 자연어 입력을 분석하고, 이를 바탕으로 지오메트리 로직 스크립트를 생성합니다. 이 알고리즘은 Grasshopper 소프트웨어의 특정 컴포넌트에 매핑되어 사용자 인터페이스 내에서 결과를 직접 생성합니다. 시스템은 사용자의 고수준 프롬프트에 따라 논리 연산의 시퀀스를 구축하는 방식으로 작동하며, 특정 복잡도 범위 내에서는 완전한 시각 스크립트를 생성하는 데 성공하지만, 이 복잡도 이상의 경우에는 실패합니다.

- **Performance Highlights**: 초기 결과는 LLM이 디자인 의도를 유추하고 지오메트리 로직을 구성할 수 있는 유망한 가능성을 보여줍니다. 연구진은 이 시스템이 알고리즘 디자인 메디에이터로서의 미래를 제시할 수 있다고 역설합니다. LLMs를 활용한 디자인 도구는 디자인 프로세스의 효과성과 사용자와의 상호 작용 방식을 혁신할 수 있습니다.



### Robust Planning with Compound LLM Architectures: An LLM-Modulo Approach (https://arxiv.org/abs/2411.14484)
- **What's New**: 이번 논문에서는 LLM-Modulo 프레임워크라는 혼합 LLM 아키텍처를 제안합니다. 이 프레임워크는 LLM과 완전한 검증 세트를 결합하여, 아웃풋의 신뢰성을 높이고 잘못된 출력을 최소화합니다. 특히, 이 방법은 이전의 프롬프트 엔지니어링 기법들이 갖는 한계를 극복하며, 생성된 모든 출력을 올바르다고 보장할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: LLM-Modulo 프레임워크는 LLM이 솔루션을 제안하고, 비평가가 그 제안을 검토하는 생성-검증-비판 루프(generative-test-critique loop)를 기반으로 합니다. 이 시스템의 기본 원리는 LLM의 출력을 검증할 Soundness(건전성)과 LLM의 전반적인 솔루션 생성 능력에 의존하는 Completeness(완전성)입니다. 사용자가 제공한 문제 정의를 통해 LLM에 입력이 주어지고, 그에 따른 제안이 이루어집니다.

- **Performance Highlights**: 여러 도메인에서 LLM-Modulo 프레임워크의 평가 결과, 색다른 성능 향상을 보였습니다. 예를 들어, Travel Planner의 경우 GPT-4o의 정확도가 8.3%에서 23.89%로, Claude-3.5-Sonnet은 4.4%에서 25%로 증가했습니다. 더욱이, 자연계획 도메인에서는 GPT-4o의 성능이 3.43%에서 40%로 극적으로 개선되어, 새로운 시스템이 출력한 모든 솔루션이 비평가들에 의해 올바른 것으로 판단되었습니다.



### Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Comba (https://arxiv.org/abs/2411.14483)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 평가를 위한 효율적인 순위 시스템을 탐구합니다. 특히, 사용자 선호를 반영하여 LLM을 쌍으로 비교하는 새로운 접근법인 Pairwise Ranking을 도입하였습니다. 이에 따라, Elo와 같은 알고리즘을 활용하여 모델의 상대적 강점을 평가할 수 있는 방법론이 제안됩니다.

- **Technical Details**: 연구에서는 네 가지 널리 사용되는 순위 알고리즘(Elo, Bradley-Terry, Glicko, Markov Chain)을 평가하며, Arena Style과 Controlled Style이라는 두 가지 다른 평가 시나리오에서의 성능을 분석합니다. 각 알고리즘의 전이성(Transitivity), 예측 정확도(Prediction Accuracy), 하이퍼파라미터(Hyperparameter) 변화에 대한 안정성(Stability)을 검토하여 모델 평가에 적합한 알고리즘을 선택하는 데 필요한 기본적인 원칙을 확립합니다.

- **Performance Highlights**: 연구 결과, LLM의 평가 및 순위 결정에서 중요한 인사이트를 도출하였습니다. 특히, 개별 알고리즘의 성능 차이를 체계적으로 분석하여 다양한 평가 작업의 특성과 가용 자원에 따라 적합한 방법을 선택할 수 있는 가이드라인을 제시합니다. 본 논문은 LLM 순위 매기기를 위한 첫 번째 체계적인 연구로, 모든 코드 및 데이터는 재현 가능성을 높이기 위해 공개되었습니다.



### GRL-Prompt: Towards Knowledge Graph based Prompt Optimization via Reinforcement Learning (https://arxiv.org/abs/2411.14479)
- **What's New**: 이 논문에서는 기존의 Large Language Models (LLMs)에서 효율적인 prompt 최적화를 위해 GRL-Prompt라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 강화 학습(reinforcement learning, RL)을 통해 자동으로 최적의 프롬프트를 구성하며, 지식 그래프(knowledge graph)를 활용하여 사용자 쿼리와 예시 간의 상관관계를 효과적으로 인코딩합니다. 이를 통해 프롬프트를 생성하는 과정에서 발생할 수 있는 불확실성을 줄이고 성능을 향상시킬 수 있습니다.

- **Technical Details**: GRL-Prompt 프레임워크의 핵심 요소는 두 가지입니다: 첫째, 사용자 지침과 후보 in-context 예시로부터 구축된 지식 그래프입니다. 이 그래프는 이종 그래프 신경망을 사용하여 구조적 임베딩 표현을 암호화합니다. 둘째, 정책 네트워크(policy network)는 쌍별 엣지 분류기(pairwise edge classifier)와 in-context 매칭 네트워크(in-context matching network)를 포함하여 최적의 예시 순서를 생성하는데 기여합니다. 이 과정에서 임베딩 기반 보상 설계를 통해 RL 훈련 과정을 안정화합니다.

- **Performance Highlights**: 실험 결과, GRL-Prompt는 현재의 최첨단 방법들을 초월하여 ROUGE-1에서 평균 0.10, ROUGE-2에서 0.07, ROUGE-L에서 0.07, BLEU에서 0.05의 성능 향상을 이루었습니다. 여러 개의 데이터셋을 비교하여 GRL-Prompt가 다양한 NLP 작업에서 LLM의 성능을 효과적으로 향상시킨 것을 확인했습니다. 이로써 제안한 방법이 LLM의 잠재력을 극대화할 수 있는 가능성을 보여줍니다.



### StreetviewLLM: Extracting Geographic Information Using a Chain-of-Thought Multimodal Large Language Mod (https://arxiv.org/abs/2411.14476)
- **What's New**: 이번 논문에서는 지리적 예측에 있어 새로운 프레임워크인 StreetViewLLM을 제안합니다. 이 모델은 대규모 언어 모델과 chain-of-thought reasoning을 통합하여 비구조적이거나 다중 양식(multi-modal) 데이터를 효과적으로 처리합니다. StreetViewLLM은 스트리트 뷰 이미지와 지리적 좌표, 텍스트 데이터를 결합하여 지리적 예측의 정확성과 세분성을 향상시킵니다.

- **Technical Details**: StreetViewLLM은 retrieval-augmented generation 기술을 활용하여 도시 환경에 대한 세부 분석 및 지리적 정보 추출을 개선합니다. 이 프레임워크는 홍콩, 도쿄, 싱가포르, 로스앤젤레스, 뉴욕, 런던, 파리 등 7개 글로벌 도시에서 적용되어 효과를 입증했습니다. 이 모델은 인구 밀도, 의료 접근성, 정상화된 식생 지수(normalized difference vegetation index), 건물 높이, 불투수 표면(impervious surface) 등의 도시 지표를 예측하는 데 우수한 성능을 보여줍니다.

- **Performance Highlights**: StreetViewLLM은 기존 기준 모델들에 비해 지속적으로 더 나은 성능을 발휘하며, 예측 정확성이 향상되었습니다. 이 연구는 대규모 언어 모델을 도시 분석(urban analytics), 도시 계획의 의사 결정(decision-making), 인프라 관리(infrastructure management), 환경 모니터링 환경 분야에 통합할 수 있는 새로운 기회를 열어줍니다.



### Large Language Model for Qualitative Research -- A Systematic Mapping Study (https://arxiv.org/abs/2411.14473)
Comments:
          8 pages, includes 1 figures and 3 tables. Submitted to the WSESE 2025 ICSE Workshop

- **What's New**: 이 연구에서는 Large Language Models (LLMs)의 질적 분석에 있어 활용 현황 및 기회를 체계적으로 조사합니다. LLMs는 방대한 텍스트 데이터를 자동화하고 향상시키는 변혁적 도구로, 기존 질적 분석 방법의 한계를 극복할 수 있는 가능성을 보여줍니다. 본 논문은 LLMs의 적용 분야, 설정, 방법론 및 평가 지표를 맵핑하여 질적 연구에서의 최신 동향과 연구 격차를 파악하는 데 중점을 두고 있습니다.

- **Technical Details**: LLMs는 방대한 양의 데이터에서 학습한 패턴을 기반으로 새로운 데이터를 생성하는 Generative AI 모델입니다. 이들 모델은 고급 자연어 처리 기능을 통해 머신 번역, 요약, 텍스트 편집 및 챗봇 지원과 같은 다양한 작업을 수행할 수 있습니다. 특히 질적 분석 분야에서 LLMs는 비구조화된 데이터에서 패턴을 식별하고 이를 체계적으로 분석할 수 있게 해줍니다.

- **Performance Highlights**: LLMs는 기존의 노동 집약적인 질적 분석 과정을 자동화하여 데이터 분석의 속도와 효율성을 drastically 향상시킬 수 있습니다. 예를 들어, 교육 연구에서 ChatGPT는 학생 설문 조사 응답을 신속하게 분석하고 주제를 도출하는 데 사용되었으며, 병원 피드백 콘텐츠 분석에서도 큰 성과를 보였습니다. 그러나 LLMs의 신뢰성을 확보하기 위해서는 Prompt 설계와 같은 여러 가지 시스템적 연구와 개선이 필요합니다.



### Leveraging Gene Expression Data and Explainable Machine Learning for Enhanced Early Detection of Type 2 Diabetes (https://arxiv.org/abs/2411.14471)
Comments:
          8 pages

- **What's New**: 본 연구는 제2형 당뇨병(T2D) 조기 검출을 위한 기계 학습(ML) 기술의 활용에 중점을 두고 있으며, 유전자 발현 데이터의 분석을 통해 질병의 생물학적 메커니즘을 밝히는 새로운 접근 방식을 제안합니다. 기존 연구가 임상 및 인구 통계 데이터에 의존한 반면, 본 논문은 유전자 발현 데이터에서 얻은 분자적 통찰력을 통합하여 T2D의 병리 생리를 이해하는 혁신적인 경로를 제공합니다. 연구에서 여섯 가지 ML 분류기를 사용하여 XGBoost가 97%의 정확도로 가장 높은 성능을 나타냈습니다. 또한, 설명 가능한 인공지능(XAI) 기법을 적용함으로써 모델의 신뢰성을 강화했습니다.

- **Technical Details**: 본 연구는 NCBI의 유전자 발현 데이터베이스에서 수집된 데이터를 바탕으로 T2D와 비당뇨병 환자를 유전자 발현 패턴을 통해 구별하는 것을 목표로 하였습니다. 여섯 종류의 ML 알고리즘, 즉 결정 트리(DT), 랜덤 포레스트(RF), 로지스틱 회귀(LR), 그래디언트 부스팅(GB), 익스트림 그래디언트 부스팅(XGBoost) 및 어댑티브 부스팅(Adaboost)을 사용하여 유전자들의 중요성을 분석했습니다. 특히, XGBoost 모델이 예측 정확도에서 두드러진 성과를 보였으며, 설명 가능한 인공지능(XAI) 기법을 통해 모델의 설명력을 높였습니다.

- **Performance Highlights**: XGBoost 모델은 본 연구의 여섯 개 ML 분류기 중에서 가장 높은 예측 정확도인 97%를 확보하며 두드러진 성과를 보였습니다. 연구에서는 유전자 발현 데이터에서 얻은 분자적 패턴을 통해 T2D의 조기 진단 가능성을 높였습니다. 또한, ML 방법론을 통해 당뇨병의 병인 기전과 관련된 생물학적 표적을 식별할 수 있는 가능성을 열었습니다. 이로 인해 임상 의사 결정에서 더 큰 신뢰를 제공할 수 있습니다.



### Popular LLMs Amplify Race and Gender Disparities in Human Mobility (https://arxiv.org/abs/2411.14469)
- **What's New**: 최근 LLMs의 사용이 사회적 결과에 영향을 미치는 여러 영역으로 확장됨에 따라, 이들 모델이 편견을 지속하고 확대하는 경향을 이해하는 것이 중요해졌습니다. 본 연구에서는 LLMs가 인종과 성별에 따라 인류 이동성(흔히 사람의 행동을 나타내는 중요한 지표)에 대한 예측에서 편견을 드러내는지 조사했습니다. GPT-4, Gemini 및 Claude라는 세 가지 LLM 모델을 사용하여, 인구 통계학적 세부 사항이 포함된 이름과 포함되지 않은 이름을 기반으로 한 관심 지점(POIs)에 대한 방문 예측을 분석하였습니다.

- **Technical Details**: 이 연구에서는 LLM들이 다양한 인구통계적 집단 간 POIs 방문 패턴을 이해하는 방식을 조사하기 위해 일련의 프롬프트를 개발했습니다. POIs는 (1) 직업 관련 및 일상 필수 POIs와 (2) 부유 관련 및 가난 관련 POIs의 두 그룹으로 구분되었습니다. 총 2,675개의 일반적인 이름을 선택하고 인종 및 성별 조합을 분석하며, LLM이 개인의 이름만을 기반으로 POIs를 예측하도록 요청했습니다.

- **Performance Highlights**: 연구 결과, 성별과 인종에 따라 LLM의 예측이 극명한 차이를 보였습니다. 남성은 여성보다 직업 관련 POIs와 강한 연관성을 보이며, 특히 백인 남성이 직업 관련 POIs와 12.2%의 확률로 연결되는 것으로 나타났습니다. 특히, 백인 여성은 직업 관련 POIs와 거의 연관이 없으며, 대부분의 예측이 일상 필수 POIs에 국한되었습니다. 또한, LLM은 인종 정보 없이 예측할 경우 부유 관련 POIs에 96.8%의 비율로 연결된다고 하였습니다.



### A Neural Network Training Method Based on Distributed PID Contro (https://arxiv.org/abs/2411.14468)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문은 대칭 미분 방정식을 기반으로 한 신경망 프레임워크에 대한 새로운 접근 방식을 제안합니다. 이 프레임워크는 완전한 대칭성을 지니고 있어 수학적 특성이 뛰어나며, 기존의 backpropagation 알고리즘의 원칙을 바탕으로 다루지 않았던 네트워크 훈련 방법론을 상세히 설명합니다.

- **Technical Details**: 제안된 방법은 미분 방정식 신호 전파를 활용하여 훈련 효과성을 유지하며 생물학적 해석 가능성을 높입니다. 시스템의 대칭성으로 인해 발생하는 가역성(reversibility)이 이 방법론의 기초가 되며, 이를 통해 PID(비례-적분-미분) 제어 접근법을 추가하여 폐쇄 시스템(closed system) 내에서 구현합니다.

- **Performance Highlights**: 이 방법론을 적용하여 훈련 속도를 향상시키고 정확성을 개선하는 성과를 보였습니다. MNIST 데이터셋을 통해 이 접근법의 실용성을 입증하며, 신경망 훈련에 대한 새로운 통찰력을 제공합니다.



### Learning to Ask: Conversational Product Search via Representation Learning (https://arxiv.org/abs/2411.14466)
Comments:
          Accepted by ACM TOIS

- **What's New**: 본 연구에서는 대화형 제품 검색 모델인 ConvPS를 제안합니다. 기존 연구들은 사용자와 상품 간의 정보 불일치 문제를 제대로 해결하지 못했습니다. 우리의 모델은 사용자, 쿼리, 아이템, 대화를 통합하여 표현 학습을 수행하고, 이를 바탕으로 보다 자연스럽고 적응력 있는 검색 시스템을 구축합니다.

- **Technical Details**: ConvPS 모델은 사용자, 쿼리, 아이템, 대화의 의미를 통합하여 학습하는 통합 생성 프레임워크를 사용합니다. 특히, 슬롯-값쌍(slots-value pairs)을 기반으로 한 질문 풀을 생성하고, 고성능 질문을 사용자에게 순차적으로 질문하는 방법을 학습합니다. 또한, 학습된 표현을 활용하여 상품을 검색합니다.

- **Performance Highlights**: 실험 결과는 ConvPS 모델이 기존 최첨단 문제 해결 모델에 비해 유의미하게 성능을 향상시킴을 보여줍니다. 이를 통해 사용자와의 대화에서 얻는 명확한 피드백을 기반으로 사용자의 동적인 선호를 더 잘 이해하게 됩니다.



### JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data (https://arxiv.org/abs/2411.14464)
Comments:
          7 pages, 5 figures, 2 tables

- **What's New**: 이 논문에서 제안하는 JESTR는 대사체 분석에서 주어진 스펙트럼에 적합한 분자 구조를 찾는 새로운 방식을 제공합니다. JESTR는 기존의 분자 지문(molecular fingerprints)이나 스펙트럼 생성을 피하고, 분자와 그 피드백 스펙트럼을 동일한 데이터의 다른 관점으로 취급하여 그 표현을 공동 공간에 효과적으로 임베드합니다. 이렇게 함으로써 코사인 유사성(cosine similarity)을 기반으로 후보 구조의 랭킹을 수행합니다.

- **Technical Details**: JESTR는 CMC(Contrastive Multiview Coding)를 활용하여 데이터의 다양한 관점 간에 불변 정보를 학습하고 공동 임베딩 공간을 생성합니다. 또한, 훈련 데이터셋에 포함된 분자와 동일한 화학식을 가진 수백만 개의 추가 데이터를 기반으로 정규화(regularization)를 수행하여 목표 분자와 후보 분자를 구별하도록 돕습니다. 이는 JESTR의 반복적 훈련과 일반화(generalization) 성능을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, JESTR는 mol-to-spec 및 spec-to-FP 주석 도구와 비교하여 우수한 성능을 보였습니다. 특히, rank@1 성능은 11.4% 향상되었고, rank@[1-5]의 평균 성능은 다른 도구들보다 23.6%에서 71.6% 향상된 것으로 나타났습니다. JESTR는 정확한 주석을 통해 대사체에 대한 귀중한 통찰을 여는 새로운 방법을 제시하였습니다.



### Leveraging AI and NLP for Bank Marketing: A Systematic Review and Gap Analysis (https://arxiv.org/abs/2411.14463)
- **What's New**: 이 논문은 은행 마케팅에서 AI(인공지능)와 NLP(자연어 처리)의 영향력을 탐구하며, 이들 기술이 마케팅 전략 향상, 고객 참여 개선 및 가치를 창출하는 데 어떻게 기여할 수 있는지를 강조하고 있습니다. 기존의 마케팅에서 AI와 NLP의 연구는 많이 이루어졌으나, 은행 부문에서의 구체적인 활용에 대한 연구는 미비한 상황입니다. 이 연구는 PRISMA 방법론을 적용하여 현재의 AI 및 NLP 활용 현황을 체계적으로 검토하고, 향후 연구 기회를 제시합니다.

- **Technical Details**: 본 연구는 AI 및 NLP의 적용 가능성을 평가하기 위해 PRISMA(체계적인 리뷰를 위한 권장 보고 항목)에 따라 문헌 검토를 수행합니다. 이를 위해 Scopus 데이터베이스를 활용하여 관련 문헌을 광범위하게 스캔하고, 분석 마케팅, 마케팅과 NLP, 마케팅, 은행 및 NLP의 교차점을 탐색하는 세 가지 주요 쿼리를 설정했습니다. 이 과정에서 특정 Boolean 검색 쿼리를 통해 질 높은 연구만을 선별하여 시스템적으로 리뷰합니다.

- **Performance Highlights**: 연구 결과는 NLP가 은행 마케팅 전략을 향상시키는 데 중요한 역할을 하며, 고객 중심의 접근에서 특히 효과적이라는 점을 보여줍니다. 또한, 이 접근법은 고객 참여, 운영 효율성 및 규제 준수와 같은 분야에서 실질적인 통찰을 제공합니다. 이의 실행 가능성을 통해 은행 마케팅에서 NLP 기반의 혁신 성장 프레임워크 개발이 가능함을 제시하고 있습니다.



### Towards Next-Generation Medical Agent: How o1 is Reshaping Decision-Making in Medical Scenarios (https://arxiv.org/abs/2411.14461)
- **What's New**: 인공지능(AI)은 현대 의료에서 필수적이며, 대형 언어 모델(LLM)이 임상 의사결정에 있어 획기적인 발전을 보여주고 있습니다. 전통적인 모델 기반 접근법은 실시간 적응성, 다단계 추론 및 복잡한 의료 작업 처리에서 한계를 보이고 있으며, 이러한 문제를 해결하기 위해 에이전트 기반 AI 시스템이 등장했습니다. 이 시스템은 추론 추적, 상황에 따른 도구 선택, 지식 검색, 단기 및 장기 기억 기능을 포함하여 복잡한 의료 시나리오를 처리할 수 있습니다.

- **Technical Details**: 연구에서는 의료 AI 에이전트의 기초가 되는 LLM 선택의 영향을 조사합니다. 특히 emergent o1 모델이 대안으로 고려되며, Chain-of-Thought (CoT) 추론 프레임워크가 그 훈련 과정에 통합되어 의사결정 능력을 높입니다. o1은 다양한 의료 데이터셋에서 기존 모델들보다 탁월한 성능을 보여주며, 특히 LancetQA 및 NEJMQA 데이터셋과 같은 다단계 추론이 필요한 작업에서 강점을 보입니다.

- **Performance Highlights**: o1 모델을 통합한 세 가지 에이전트(Chain of Diagnosis, MedAgents, AgentClinic)를 다양한 유명 데이터셋에서 테스트한 결과, 진단 정확도 및 일관성이 향상되었습니다. 특히, o1은 DxBench 및 NEJMQA와 같은 임상 벤치마크에서 뛰어난 진단 정확도를 나타내며, 진단의 가변성을 줄이는 데 효과적입니다. 그러나 복잡한 작업에서는 뛰어난 성능을 보이지만, 단순한 작업에서는 효율성이 떨어져 기존 모델이 더 나은 선택이 될 수 있습니다.



### LLaSA: Large Language and Structured Data Assistan (https://arxiv.org/abs/2411.14460)
- **What's New**: 이 논문에서는 LLaSA(Large Language and Structured Data Assistant)라는 새로운 프레임워크를 제안합니다. LLaSA는 다양한 구조화된 데이터를 하이퍼그래프(hypergraph) 형식으로 통합하여 모델의 성능을 향상시키는 방법을 제공합니다. 특히, LLaSA는 self-supervised learning을 통해 하이퍼그래프 인코더와 G-Former를 사전 학습함으로써 다양한 LLM에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: LLaSA는 테이블, 지식 그래프와 같은 구조화된 데이터를 하이퍼그래프로 모델링하여 통합된 GNN을 통해 인코딩을 수행합니다. 이 과정에서 테이블의 셀은 노드로, 행과 열은 하이퍼엣지(hyperedge)로 취급되어 다양한 구조화된 데이터에 대해 일관성 있는 처리가 가능합니다. G-Former는 인코딩된 구조화된 데이터를 LLM이 이해할 수 있는 고정된 수의 soft token으로 변환하여 모듈 간의 격차를 해소합니다.

- **Performance Highlights**: 여러 SKG 태스크에 대한 실험 결과에서 LLaSA는 LLM의 다양한 구조화된 데이터 처리 능력을 현저히 향상시킴을 보여주었습니다. 예를 들어, LLaSA는 frozen LLM인 Llama-7B에서 10개 데이터셋에 대해 평균 12%의 성능 향상을 기록하였고, LoRA 조정된 LLM에서도 0.4%의 평균 향상을 달성했습니다. 또한, LLaSA는 전체 파라미터 조정을 사용하는 이전의 SOTA 방법보다 우수한 성과를 보입니다.



### Unveiling User Preferences: A Knowledge Graph and LLM-Driven Approach for Conversational Recommendation (https://arxiv.org/abs/2411.14459)
- **What's New**: COMPASS라는 새로운 프레임워크가 제안되어 기존의 CRSs(Conversational Recommender Systems)에서 사용자 선호도를 개선하고, 추천 성능과 설명 가능성을 높이고자 합니다. 이 프레임워크는 LLMs(Large Language Models)와 KGs(Knowledge Graphs)를 통합하여 비가시적인 사용자 선호도를 드러내는 새로운 방법을 제시합니다. 특히, COMPASS는 두 단계의 훈련 과정을 통해 비구조적 대화와 구조적 KGs 간의 간극을 좁힙니다.

- **Technical Details**: COMPASS는 그래프 엔티티 설명(pre-training) 메커니즘을 사용하여 KG 구조를 자연어 설명으로 변환하여 LLM이 도메인 특정 정보를 이해할 수 있게 만듭니다. 이어서, COMPASS는 지식 기반(instruction) 세부조정을 통해 사용자 선호에 대한 추론 능력을 높입니다. 이 과정에서 KG로부터 추출된 관련 엔티티 정보와 관계를 포함하여 대화 기록을 보강하는 KG 향상 컨텍스트를 사용합니다.

- **Performance Highlights**: COMPASS는 설명 가능한 사용자 선호 요약 생성능력을 통해 CRSs에서 실제 데이터와 대화 이력을 기반으로 한 지능형 추천 및 이해 가능한 결과를 제공합니다. 이 프레임워크는 기존의 CRS 모델과 통합되는 적응형 게이팅 메커니즘을 포함하여, 성능을 강화하고, 수정 없이도 추천 품질을 높입니다. 이를 통해 COMPASS는 더욱 명확하고 해석 가능한 추천 시스템을 구현할 수 있는 가능성을 제공합니다.



### Improving training time and GPU utilization in geo-distributed language model training (https://arxiv.org/abs/2411.14458)
- **What's New**: 본 연구에서는 여러 데이터 센터(DC)에서의 대규모 언어 모델(LM) 훈련 시 GPU 활용도를 극대화하고 훈련 시간을 단축시키기 위한 두 가지 혁신적 접근법인 ATLAS와 BUBBLETEA를 제안합니다. ATLAS는 지리적으로 분산된 훈련 작업을 효과적으로 처리하기 위해 새로운 설계 아이디어를 도입해 훈련 시간을 최대 17배 향상시키며 GPU 활용도를 94%까지 끌어올립니다. BUBBLETEA는 GPU의 유휴 시간 동안 LM 추론의 일부인 prefill-as-a-service를 실행하여 GPU 활용도를 추가로 개선합니다.

- **Technical Details**: ATLAS는 고속 내-DC 통신을 통해 WAN의 병목 현상을 극복하기 위한 여러 설계를 채택했습니다. 특히, GPU 간의 여러 TCP 연결을 활용하여 WAN 대역폭을 확장하고, 파이프라인 병렬성(PIpeline Parallelism)과 데이터 병렬성(Data Parallelism)의 적절한 조합을 찾았습니다. BUBBLETEA는 훈련 대기 시간 동안 발생하는 GPU 유휴 시간을 최소화하기 위해 prefill 요청을 효율적으로 스케줄링하여 별도의 GPU 자원을 사용하는 대신 훈련과 추론 간의 커뮤니케이션을 최적화합니다.

- **Performance Highlights**: ATLAS와 BUBBLETEA를 결합하여 GPU 훈련 시스템의 전반적인 성능을 크게 향상시켜, 훈련 시간을 최대 17배 단축시키고 GPU 활용도를 94%로 높이는 쾌거를 달성했습니다. 특히, 이 시스템은 복잡한 AI 작업을 처리하면서 효율적인 자원 관리를 통해 사용자에게 높은 가치를 제공합니다. 연구 결과는 향후 AI 산업의 발전에 기여할 것으로 기대됩니다.



### Can Artificial Intelligence Generate Quality Research Topics Reflecting Patient Concerns? (https://arxiv.org/abs/2411.14456)
- **What's New**: 이 연구는 환자 중심의 연구를 위한 자동화된 프레임워크를 제안하며, 이를 통해 연구 및 환자 치료 간의 간극을 좁히려는 노력을 다룹니다. 혁신적인 자연어 처리(NLP)와 인공지능(AI)을 활용하여 환자 포털 메시지를 분석하고, 이를 통해 환자가 중요하게 여기는 문제를 우선시하는 연구 아이디어를 생성합니다. 이러한 접근 방식은 환자의 관점을 연구에 효과적으로 통합할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 연구는 2013년부터 2024년까지의 대규모 학술 병원에서 수집된 614,464개의 환자 메시지를 통해 2단계 비지도 학습 기반의 NLP 주제 모델을 구성하였습니다. AI(GPT-4o, OpenAI Inc, 2024년 4월 버전)를 활용하여 여러 단계의 작업을 수행하고, 지식 해석, 연구 아이디어 생성, 자기 반성 및 수정, 자기 확신의 과정을 통해 환자의 임상 문제를 해결하는 연구 주제를 생성했습니다.

- **Performance Highlights**: AI가 제안한 연구 주제 중 3분의 1이 높은 중요성과 독창성을 갖추었다고 평가되었습니다. 또한, 두 암(유방암과 피부암)에 대해서 AI가 제안한 연구 주제의 3분의 2가 독창적이라고 판단되었습니다. 이러한 결과는 방대한 환자 메시지를 통해 얻어진 AI 생성 연구 주제가 환자 중심의 건강 연구 방향을 의미 있게 안내할 수 있음을 보여줍니다.



### Deferred Backdoor Functionality Attacks on Deep Learning Models (https://arxiv.org/abs/2411.14449)
- **What's New**: 이번 논문에서는 Backdoor 공격의 새로운 패러다임으로 Deferred Backdoor Functionality Activation (DBFA)를 소개합니다. 기존의 공격 방식을 넘어 DBFA는 실제로 트리거가 작동해도 무해한 출력을 생성하여 처음에는 공격의 존재를 숨깁니다. 이를 통해 기존의 탐지 및 방어 메커니즘을 우회할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: DBFA의 핵심은 공격 기능이 처음에 활성화되지 않고 무해한 출력을 생성하는 것입니다. 모델이 이후 업데이트를 받고 난 뒤에만 공격 기능이 활성화되도록 설계되어 있어, 일반적인 모델 업데이트 후에 공격이 작동하게 됩니다. 이를 구현하기 위해 불학습(unlearning)을 취약하게 만들어 백도어를 쉽게 취소하고 다시 활성화할 수 있도록 하였습니다.

- **Performance Highlights**: DBFA에 대한 실험에서는 다양한 fine-tuning 시나리오, 백도어 공격 유형, 데이터셋 및 모델 아키텍처를 통해 그 효과성과 스텔스(stealth) 특성을 입증했습니다. 이 연구는 기존 백도어 공격의 한계를 극복할 수 있는 중요한 가능성을 보여주며, 새로운 공격 방식이 얼마나 효율적으로 작용할 수 있는지를 강조합니다.



### GeMID: Generalizable Models for IoT Device Identification (https://arxiv.org/abs/2411.14441)
Comments:
          8 pages main (9 figures, 2 tables), 19 pages Supplementary Material, 27 pages total

- **What's New**: 본 연구는 IoT 기기 식별(Device Identification, DI)의 보편성을 높이기 위한 새로운 프레임워크를 제안합니다. 기존 방법들이 다양한 네트워크 환경에서 DI 모델의 일반화 문제를 놓치고 있는 반면, 본 연구는 서로 다른 환경에서 수집된 데이터셋을 통해 DI 모델의 일반화 가능성을 평가합니다. 두 단계로 이루어진 접근 방식을 통해 모델 선택과 평가를 세분화하여, IoT 보안 및 기기 식별 분야에 기여합니다.

- **Technical Details**: 본 연구는 유전자 알고리즘(Genetic Algorithm)을 활용하여 다양한 특성 및 모델 선택 방식을 개발합니다. 첫 번째 단계에서는 네트워크 환경에 의존하지 않는 특성을 식별하고, 두 번째 단계에서는 독립적인 데이터셋을 통해 기기 특정 모델 인스턴스를 교육시키고 그 일반화 가능성을 평가합니다. 본 연구는 패킷 특성 기반 접근법이 흐름 또는 창(Window) 통계를 사용하는 방법보다 더 나은 일반화 성능을 나타낸다고 주장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방식이 대안 방법들에 비해 더 뛰어난 일반화 성능을 보임을 입증하였습니다. 기존 방법의 한계를 강조하며, DI 시스템에서 기기 유형에 따른 패킷 특성을 기반으로 한 접근이 어떻게 IoT 네트워크의 위험을 줄이고 모델 효과성을 높일 수 있는지를 보여줍니다. 연구진은 투명성 및 재현성 증진을 위해 코드를 공개하며, 커뮤니티가 연구 결과를 재현하고 발전시킬 수 있는 자원을 제공합니다.



### Transforming Engineering Education Using Generative AI and Digital Twin Technologies (https://arxiv.org/abs/2411.14433)
Comments:
          8 pages, 7 figures

- **What's New**: 디지털 트윈( Digital Twin) 기술이 전통적인 산업 분야를 넘어 교육 경험을 강화하는 데 잠재력을 지니고 있다는 점이 강조됩니다. 이 연구는 다양한 충실도(fidelity)를 가진 디지털 트윈 모델이 블룸(Taxonomy of Bloom) 인지 영역의 다양한 단계를 어떻게 지원할 수 있는지를 탐구하고 있습니다. 특히 다층적인 교육적 필요를 충족시키기 위한 커리큘럼 설계의 새로운 접근법으로 디지털 트윈 모델의 복잡성과 충실도가 교육 단계에 맞추어 조정될 수 있음을 시사합니다.

- **Technical Details**: 디지털 트윈 기술은 항공 우주 분야에서 시작된 개념으로, 온라인 상태 업데이트를 통해 물리적 시스템의 행동을 가상 공간에서 매핑할 수 있도록 합니다. 산업 시스템의 고급 모니터링과 진단, 예측을 위해 IoT 장치와 다양한 센서 데이터를 통합하여 복잡한 실시간 시뮬레이션을 만들어낼 수 있습니다. 블룸의 다양한 인지 단계와 교육 수준에 맞춰 저충실도( low-fidelity)에서 고충실도( high-fidelity) 디지털 트윈 모델이 지원하는 각 단계를 명확히 매칭시켜 교육과정 설계에 도움을 줍니다.

- **Performance Highlights**: 디지털 트윈을 이용한 교육이 직업 교육과 훈련에서 효율적일 뿐만 아니라 학습 효과를 높이는 방법으로 제안됩니다. 저충실도 디지털 트윈은 기초 지식과 기술 습득을 지원하며, 중간 충실도 모델은 복잡한 문제 해결 능력을 향상시키고, 고충실도 모델은 박사 과정 학생들이 혁신적 디자인과 복잡한 실험을 수행할 수 있도록 돕습니다. 이 교육 모델은 Kirkpatrick 모델을 사용하여 각 디지털 트윈 모델의 충실도가 학습 성과에 미치는 영향을 평가하여 교육자들이 DT와 LLM을 효과적으로 통합할 수 있도록 지원합니다.




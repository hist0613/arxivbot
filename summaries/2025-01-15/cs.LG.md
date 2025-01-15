New uploads on arXiv(cs.CL)

### PokerBench: Training Large Language Models to become Professional Poker Players (https://arxiv.org/abs/2501.08328)
Comments:
          AAAI 2025

- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)의 포커 게임 능력을 평가하기 위한 벤치마크인 PokerBench를 소개합니다. 기존의 NLP 작업에서 좋은 성과를 내고 있는 LLM이지만, 포커와 같은 복잡한 전략 게임에서의 적용은 새로운 도전 과제입니다. PokerBench는 전문 포커 플레이어와 협력하여 개발한 11,000개의 중요한 시나리오로 구성되어 있으며, LLM의 포커 게임 능력을 평가하는 데 유용합니다.

- **Technical Details**: PokerBench는 포커의 pre-flop 및 post-flop 상황을 포함하여 결정적인 의사결정을 평가하는 데 필요한 11,000개의 스폿을 제공합니다. 이 데이터셋은 게임 이론 최적(poker optimal) 원칙에 기반하여 개발되었으며, GPT-4 및 ChatGPT 3.5 등 다양한 최첨단 모델을 평가하는 데 활용되었습니다. 연구 결과, 모든 모델이 최적의 포커 플레이에서 저조한 성과를 나타냈지만, 파인 튜닝을 통해 성능이 크게 개선되었습니다.

- **Performance Highlights**: PokerBench를 사용하여 다양한 LLM 모델의 포커 게임 능력을 평가한 결과, GPT-4가 53.55%의 정확도로 가장 높은 성능을 보였습니다. 또한, 파인 튜닝된 모델 간의 경쟁을 통해 PokerBench에서 높은 점수를 받은 모델이 더 많은 승수를 기록하는 경향이 있음을 확인했습니다. 이 연구는 LLM의 포커 게임 능력을 빠르고 신뢰성 있게 평가할 수 있는 새로운 벤치마크를 제공하며, LLM의 복잡한 게임 시나리오에서의 발전을 연구하는 데 중요한 기초 자료를 제공합니다.



### Exploring Robustness of Multilingual LLMs on Real-World Noisy Data (https://arxiv.org/abs/2501.08322)
- **What's New**: 이 논문에서는 인공지능 언어 모델(Large Language Models, LLMs)의 현실 세계 오타에 대한 내구성을 조사합니다. 우리는 9개의 서로 다른 LLM의 성능을 분석하며, 6가지 언어와 3가지 NLP 작업(자연어 추론, 명명 개체 인식, 의도 분류)에 걸쳐 실제 오타가 모델 성능에 미치는 영향을 평가합니다. 이를 통해, mT5 모델이 다른 모델에 비해 더 높은 내구성을 보임을 확인했습니다.

- **Technical Details**: 모델 크기는 0.2B에서 13B까지 다양하며, 우리는 Wikipedia 수정 이력을 사용하여 현실 세계의 오타 딕셔너리(WikiTypo)를 구성하였습니다. 이 연구는 다양한 크기의 모델이 실제 노이즈에 어떻게 반응하는지에 대한 세 가지 연구 질문(RQ)을 중심으로 진행되었습니다. 실험을 통해 MNL 및 NER과 같이 서로 다른 NLP 작업에서 성능의 차이를 확인하였습니다.

- **Performance Highlights**: 연구 결과, 모든 모델에서 현실 세계의 노이즈에 대한 취약성이 나타났지만 각 작업에 따라 성능 차이가 있었고, 랜덤 오류 삽입 실험을 통해 성능 격차를 확인했습니다. 특히 자연어 추론(NLI) 작업에서 가장 큰 성능 격차가 관찰되었고, mT5(13B)가 모든 작업에서 평균적으로 가장 높은 robustness를 보였습니다. 실험의 결과는 WikiTypo 데이터 세트와 3가지 NLP 작업에서 다양한 언어 모델의 성능을 평가하는 데 사용되었습니다.



### Enhancing Automated Interpretability with Output-Centric Feature Descriptions (https://arxiv.org/abs/2501.08319)
- **What's New**: 본 연구는 자동 해석 가능성 파이프라인에서 특징(feature) 설명 방법을 개선하기 위해 출력 중심(output-centric) 접근 방식을 제안하고 있습니다. 기존 방법들이 모델 출력을 고려하지 못하고 입력 중심(input-centric) 접근에 집중했던 문제점을 지적합니다. 이를 해결하기 위해 VocabProj와 TokenChange라는 두 가지 새로운 방법을 도입하였으며, 이들은 모델의 출력에 대한 인과적 영향을 더 잘 포착할 수 있습니다.

- **Technical Details**: 연구에서는 LLM(대형 언어 모델)에서 계산의 기본 단위인 특징을 자동으로 설명하는 문제에 주목하고 있습니다. 특징은 벡터화된 형태로 표현되며, 다양한 형태의 특징을 지원하기 위해 일반적인 프레임워크를 채택합니다. 연구에 제안된 VocabProj 방법은 모델의 어휘 공간으로의 투사에서 중요한 토큰을 사용하고, TokenChange 방법은 특징이 증폭될 때 모델 출력 확률이 가장 많이 변화하는 토큰을 고려합니다.

- **Performance Highlights**: 실험 결과, VocabProj와 TokenChange는 전반적으로 MaxAct 방법보다 모델 출력에 대한 특징의 영향을 더 잘 설명한다고 나타났습니다. 세 가지 방법의 조합이 모든 평가에서 가장 높은 성능을 기록했으며, 특히 기존에 "죽은" 특징으로 여겨졌던 입력을 효율적으로 발견하는 데 기여할 수 있음을 보여주었습니다. 이러한 결과는 자동 해석 가능성 파이프라인에 출력 중심 방법을 통합할 필요성을 강조합니다.



### MiniMax-01: Scaling Foundation Models with Lightning Attention (https://arxiv.org/abs/2501.08313)
Comments:
          A technical report from MiniMax. The authors are listed in alphabetical order. We open-sourced our MiniMax-01 at this https URL

- **What's New**: MiniMax-01 시리즈, 즉 MiniMax-Text-01 및 MiniMax-VL-01이 도입되었습니다. 이 모델은 최고 수준 모델들과 비교할 수 있으며, 긴 컨텍스트를 처리하는 데 있어 더 나은 능력을 제공합니다. 중요 요소는 lightning attention과 이를 효율적으로 확장하는 부분입니다.

- **Technical Details**: MiniMax-01 모델은 Mixture of Experts (MoE)와 통합되어, 32명의 전문가와 총 4560억 개의 파라미터를 가진 모델로 구성됩니다. 각 토큰에 대해 459억 개의 파라미터가 활성화되며, 이를 위해 최적화된 병렬 전략과 효율적인 계산-통신 오버랩 기술을 개발하였습니다. 이러한 접근 방식은 수백억 개의 파라미터를 가진 모델을 수백만 토큰의 컨텍스트에서 효율적으로 훈련 및 추론할 수 있게 해줍니다.

- **Performance Highlights**: MiniMax-Text-01의 컨텍스트 윈도우는 훈련 중에 최대 100만 토큰에 도달할 수 있으며, 추론 중에는 최대 400만 토큰까지 확장 가능합니다. 두 모델은 GPT-4o 및 Claude-3.5-Sonnet과 같은 최첨단 모델과 성능이 일치하면서도 20-32배 더 긴 컨텍스트 윈도우를 제공합니다. MiniMax-01은 공개적으로 발표되었으며, 이를 통해 더욱 효율적인 작업이 가능해졌습니다.



### Everybody Likes to Sleep: A Computer-Assisted Comparison of Object Naming Data from 30 Languages (https://arxiv.org/abs/2501.08312)
Comments:
          To appear in the Proceedings of the Global WordNet Conference 2025

- **What's New**: 이번 연구는 다양한 출처의 개인 객체 명명 데이터를 통합하여 다국어 데이터 세트를 생성함으로써 현재의 객체 명명 데이터의 투명성과 비교 가능성을 높이려는 첫 번째 단계를 제시합니다. 17개의 객체 명명 데이터 세트를 연결하여 30개 언어와 10개 언어 계통을 포함하는 표본을 만들었습니다. 이러한 접근은 연구자들이 객체 명명 작업에 대한 비교 연구를 더욱 명확하게 진행하는 데 기여할 것입니다.

- **Technical Details**: 연구자는 표준 워크플로우를 사용하여 개념을 기존 Concepticon 개념 세트에 매핑했습니다. PyConcepticon 라이브러리를 사용해 초기 자동 매핑 후 수동 검증을 통해 잘못 매칭된 항목을 수정하고 새로운 개념 세트를 도입했습니다. 총 42개의 새로운 개념 세트가 추가되어 Concepticon 데이터 저장소에 제출되었습니다.

- **Performance Highlights**: 연구 결과는 객체 명명 연구를 향상시킬 수 있는 기초를 제공하며, 다국어 연구에서 공통 개념의 빈도 분석을 통해 언어별 객체 명명 적합성을 이해하는 데 기여합니다. 이러한 분석은 심리언어학 및 자원낭비 문제가 있는 언어에 대한 비교 연구의 가치를 확장하고 있습니다.



### A Survey on Pedophile Attribution Techniques for Online Platforms (https://arxiv.org/abs/2501.08296)
Comments:
          17 pages, 3 figures

- **What's New**: 이번 연구에서는 소셜 미디어에서 아동 성범죄자(online sexual predators)를 식별하기 위한 자동화 시스템의 필요성을 강조합니다. 연구진은 성범죄자의 글쓰기를 분석하여 그들을 적합하게 분류하는 방법을 제안하고, 이를 통해 온라인 성범죄로부터의 위험을 줄이는 새로운 접근 방식을 제시합니다. 또한 다양한 데이터세트와 특징, 분류 기술에 대한 리뷰를 통해 현재까지의 연구 경향을 정리하였습니다.

- **Technical Details**: 저자는 '저자 귀속(author attribution)'이라는 개념을 중심으로 자동화된 성범죄자 귀속 시스템의 접근 방식을 설명합니다. 연구에서는 알려진 저자 집합에 따라 텍스트를 귀속시키는 '닫힌 집합(closed set)' 문제와 알려지지 않은 저자가 포함된 텍스트를 다루는 '열린 집합(open set)' 문제를 다룹니다. 이와 함께 텍스트 길이가 저자 식별에 미치는 영향에 대해서도 논의합니다.

- **Performance Highlights**: 연구 결과, 아동 성범죄자의 귀속을 다루는 이전 연구들은 아직 실질적인 도구 개발에 부족함이 있다는 것이 드러났습니다. 특히, 짧은 텍스트 메시지의 분석이 긴 텍스트에 비해 정보를 충분히 담지 못하는 문제를 지적합니다. 이를 해결하기 위한 추가 연구 문제를 제시함으로써 향후 개선 방향을 모색하고 있습니다.



### HALoGEN: Fantastic LLM Hallucinations and Where to Find Them (https://arxiv.org/abs/2501.08292)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 HALoGEN이라는 종합적인 환각(hallucination) 벤치마크를 소개합니다. 이 벤치마크는 프로그래밍, 과학 고증, 요약 등 아홉 가지 도메인에 걸쳐 10,923개의 프롬프트(prompt)와 각 사용 사례에 대한 자동 고정밀 검증기(automatic high-precision verifiers)를 포함하고 있습니다. 이를 통해 대규모 언어 모델을 평가할 수 있는 체계를 마련하였습니다.

- **Technical Details**: HALoGEN은 LLM(generative large language models)이 생성한 결과를 원자 단위(atomic units)로 분해하고, 이를 고품질의 지식 소스에 대조하여 검증합니다. 연구진은 14개의 언어 모델에서 약 150,000개의 생성 결과를 평가하였고, 일부 도메인에서는 생성된 원자 사실.atomic facts의 최대 86%가 환각을 포함하고 있다는 사실을 발견했습니다. 또한, 환각의 유형을 훈련 데이터의 잘못된 기억(Type A errors), 잘못된 지식(Type B errors), 그리고 허구(Type C errors)로 분류하는 새로운 오류 분류법을 정의하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 가장 성능이 뛰어난 모델조차 높은 비율의 환각을 포함하고 있어 신뢰할 수 있는 LLM 개발에 대한 필요성을 강조합니다. 연구진은 HALoGEN을 통해 생성 모델의 환각 발생 원인에 대한 체계적 연구에 기초가 되기를 희망하며, 궁극적으로 신뢰할 수 있는 대규모 언어 모델 개발을 촉진하는 데 기여하고자 합니다.



### AfriHate: A Multilingual Collection of Hate Speech and Abusive Language Datasets for African Languages (https://arxiv.org/abs/2501.08284)
- **What's New**: AfriHate는 15개의 아프리카 언어로 구성된 증오 발언(hate speech) 및 언어적 공격(abusive language) 데이터셋의 다국어 컬렉션을 제공합니다. 이 데이터셋은 지역 문화를 이해하기 위해 원어민들이 주관적으로 주석을 달아 구성되었습니다. 기존의 데이터 세트와 비교할 때, AfriHate는 다양한 유형의 증오 발언과 공격 언어를 캡처하는 더 나은 품질의 데이터를 제공하는 것을 목표로 합니다. 또한, 이 연구는 아프리카의 저자원(low-resource) 언어에 대한 연구 공동체에 중요한 기초 자료를 제공합니다.

- **Technical Details**: AfriHate 데이터셋은 알제리 아랍어, 아마하리어, 이그보어 등 15개 아프리카 언어로 되어 있으며, 각 인스턴스는 원어민에 의해 주석이 달립니다. 데이터셋은 증오, 공격적/모욕적, 중립의 세 클래스로 분류되며, 이를 통해 각각의 증오 발언이 어떤 민족, 정치, 성별, 장애, 종교와 같은 여섯 가지 공통된 속성으로 라벨링됩니다. 데이터 수집은 2012년부터 2023년까지의 트윗을 바탕으로하며, 기존의 키워드, 해시태그, 사용자 계정 외에도 사용자들로부터 수집된 추가적인 키워드를 이용한 새로운 전략이 포함됩니다.

- **Performance Highlights**: 결과적으로, 성능은 사용된 언어에 따라 크게 달라지며, 멀티링구얼 모델(multilingual models)은 저자원 환경에서 성능을 향상시키는 데 기여할 수 있습니다. 수집된 데이터 및 개별 라벨들은 연구 공동체에 공개되어, 혐오 발언과 공격적인 언어에 대한 연구를 위한 귀중한 자료로 활용될 것입니다. 또한, 각 언어별로 규명된 공격적인 언어 사전이 포함되어 있어, 사용자들이 지역 문화와 정황을 이해하며 이러한 내용을 효과적으로 다룰 수 있는 기초를 제공합니다.



### Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing (https://arxiv.org/abs/2501.08276)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 신뢰성을 평가하기 위해 인구통계학적 스타일에 따라 다양한 패러프레이즈 세트를 생성하는 방법을 소개합니다. 연구팀은 기존 SocialIQA 데이터셋을 확장하여, 다양한 언어 스타일에 대한 모델의 반응을 평가하고, LLMs의 추론 능력을 구조적으로 테스트하는 것을 목표로 합니다. 특히, 데이터셋과 코드가 추후 연구와 재현성을 위해 공개될 예정입니다.

- **Technical Details**: 연구에서는 LLAMA2 모델을 사용하여 인구 통계학적 조건에 따른 패러프레이즈를 생성하고, 모델의 성능을 평가합니다. 평가 과정은 패러프레이즈의 의미적 유사성, 설명 가능성, 그리고 ATOMIC 성능 등을 분석하는 세부 신뢰성 검사로 구성됩니다. 인구 통계에 따른 패러프레이징에서 모델 성능의 변화를 관찰함으로써, 언어 변형의 미세한 변별력이 여전히 도전 과제임을 시사합니다.

- **Performance Highlights**: 실험 결과, LLMs는 언어 스타일 변화에 대한 추론에서 젊은 인구의 표현적 언어를 다루는 데 있어 효율성이 떨어지는 경향이 있음을 보여주었습니다. 단, 2-shot 설정에서 LLMs는 미세 조정된 모델과 비교하여 유의미한 성과 향상을 보였습니다. 이는 LLM들이 특정 맥락에 대한 사례를 제공받을 때 성능이 크게 개선되는 가능성을 보여줍니다.



### Comparative Analysis of Efficient Adapter-Based Fine-Tuning of State-of-the-Art Transformer Models (https://arxiv.org/abs/2501.08271)
- **What's New**: 이번 연구에서는 SuperGLUE 벤치마크의 이항 분류 (binary classification) 작업과 Kaggle의 다중 클래스 뉴스 카테고리 분류 작업에 대한 다양한 어댑터 아키텍처 (adapter architectures)의 효능을 조사했습니다. 연구에서는 DistilBERT, ELECTRA, BART라는 세 가지 트랜스포머 (transformer) 모델의 분류 성능과 시간 복잡도를 비교했습니다. 특별히, 기존의 파인 튜닝 (fine-tuning) 방식과 아홉 가지 최신 어댑터 아키텍처를 통해 성능 차이를 분석하였습니다.

- **Technical Details**: 분석 결과, 어댑터 아키텍처들 간의 성능 차이가 관찰되었으며, 특히 시간 소요를 현저히 줄이면서도 파인 튜닝에 비견하는 또는 그 이상의 성능을 발휘할 수 있음을 보여주었습니다. 새로운 분류 작업에서도 유사한 결과가 나타났으며, 이는 어댑터가 파인 튜닝에 대한 효율적이고 유연한 대안이라는 점을 입증합니다. 다양한 자연어 처리 (NLP) 응용 프로그램에서 어댑터를 선택하고 구현하기 위한 귀중한 통찰력과 지침을 제공합니다.

- **Performance Highlights**: 종합적으로, 어댑터 아키텍처는 여러 기준에서 파인 튜닝보다 우수한 결과를 나타내며, 이는 특히 훈련 시간이 짧다는 점에서 큰 장점입니다. 연구 결과는 다양한 NLP 작업에서 어댑터의 유용성을 보여주며, 시간이 제한된 환경에서도 효과적으로 모델을 학습시킬 수 있는 방법을 제시합니다. 이러한 발견은 향후 연구 및 실제 적용에도 중요한 기초자료가 될 것입니다.



### Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models (https://arxiv.org/abs/2501.08248)
- **What's New**: 이 논문에서는 In-Context Retrieval and Reasoning (ICR^2)라는 새로운 벤치마크를 소개합니다. 기존의 LOFT 벤치마크가 LCLMs의 성능을 과대평가할 수 있는 문제를 해결하고자, LCLMs에 대한 보다 실제적인 평가를 제공하기 위해 고안된 것입니다. ICR^2는 강화된 컨파운딩 문서(confounding documents)를 포함하여 LCLMs의 실제적인 상황에서의 성능을 평가합니다.

- **Technical Details**: ICR^2 벤치마크는 Wikipedia에서 수집된 포괄적인 지식 기반을 기반으로 구축되며, 강력한 리트리버를 사용해 문서를 선택합니다. 이를 통해 설명된 세 가지 방법은 (1) retrieve-then-generate fine-tuning, (2) retrieval-attention probing, (3) joint retrieval head training 입니다. 이 방법들은 LCLMs의 in-context retrieval 능력을 증가시키고, 복잡한 multi-stage pipeline의 한계를 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 논문에서 제시된 최적의 접근 방식은 Mistral-7B 모델에 적용되어 LOFT에서 평균 +17과 +15 포인트 향상된 Exact Match를 보여주었으며, ICR^2에서도 +13과 +2 포인트의 개선을 이뤘습니다. 이 방법은 기존의 Vanilla RAG 및 supervised fine-tuning을 초월하여 성능을 발휘하였고, 더 작은 모델임에도 불구하고 대부분의 작업에서 GPT-4-Turbo보다 우수한 성능을 기록했습니다.



### ASTRID -- An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems (https://arxiv.org/abs/2501.08208)
Comments:
          29 pages

- **What's New**: 이번 연구에서는 Retrieval Augmented Generation(RAG) 기반의 임상 질문 응답(QA) 시스템을 평가하기 위한 새로운 자동화된 메트릭스인 ASTRID를 발표합니다. ASTRID는 Context Relevance (CR), Refusal Accuracy (RA), Conversational Faithfulness (CF)의 세 가지 메트릭으로 구성되어 있으며, 이 메트릭들은 클리닉 QA 시스템의 위험성을 체계적으로 분석하고, 모델의 응답의 신뢰성을 높이기 위해 설계되었습니다.

- **Technical Details**: ASTRID는 200개의 실제 환자 질문 데이터셋을 기반으로 CF 메트릭의 효과성을 검증하였습니다. 이 메트릭은 기존 정의보다 임상 대화에서의 인간 평점을 더 정확히 예측할 수 있으며, 비전문가로부터의 평가에서도 일관성을 보입니다. 또한, CF, RA, CR 메트릭을 함께 사용하면 전문가의 평가와 일치하는 경향이 관찰되었습니다.

- **Performance Highlights**: 본 연구에 따르면 ASTRID의 세 개 메트릭은 총 아홉 개의 서로 다른 LLM 모델에서 인간 평가와 높은 일치를 보였으며, 이는 LLM 기반 자동화 평가 파이프라인에서 이들 메트릭을 활용할 수 있는 가능성을 보여줍니다. ASTRID의 첫 번째 메트릭인 CF는 특히 임상 대화에서 모델의 응답이 지식 기반에 충실한지를 평가하는 데 우수한 성능을 보였습니다.



### ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving (https://arxiv.org/abs/2501.08203)
- **What's New**: 본 연구에서는 ArithmAttack을 제안하여 Large Language Models (LLMs)의 수학 문제 해결 능력이 소음 입력에 얼마나 강건한지를 평가합니다. 특히, 구두점이 포함된 노이즈 프롬프트에 대한 LLM의 반응을 분석하면서, 소음이 증가하면 모델의 성능이 저하된다는 점을 발견했습니다. 다양한 LLM을 대상으로 테스트한 결과, 모든 모델이 증가하는 노이즈에 대해 점차적으로 취약해진다는 것을 확인했습니다.

- **Technical Details**: ArithmAttack은 GSM8K 및 MultiArith 데이터셋을 기반으로 구두점을 무작위로 삽입하여 LLM의 강건성을 평가하는 새로운 접근법입니다. 연구에서는 Llama3, Mistral, Mathstral 등 총 7개의 LLM을 평가하며, Noise가 문장 길이의 10%, 30%, 50%에 미치는 영향을 분석합니다. 성능 정확성을 평가하기 위해 AutoCoT prompting을 사용하며, Attack Success Rate (ASR)을 측정하여 모델의 반응 변화를 분석합니다.

- **Performance Highlights**: 실험 결과에 따르면, Llama3.1 모델이 모든 테스트에서 가장 우수한 성능을 발휘했습니다. 그러나 모든 모델이 소음에 대한 강건성이 떨어짐을 보여주었으며, 소음 비율이 증가함에 따라 성능 저하가 두드러지게 나타났습니다. 이러한 결과는 LLM의 수학 문제 해결 능력이 입력의 노이즈에 취약하다는 것을 시사합니다.



### OpenCSG Chinese Corpus: A Series of High-quality Chinese Datasets for LLM Training (https://arxiv.org/abs/2501.08197)
Comments:
          The datasets are available on this https URL ; The code is on this https URL

- **What's New**: 이번 논문에서는 고품질의 중국어 데이터셋 부족 문제를 해결하기 위해 OpenCSG Chinese Corpus를 제안합니다. 이 데이터셋은 LLM 사전 훈련(pretraining), 후 훈련(post-training), 미세 조정(fine-tuning)을 위해 특별히 설계되었습니다. OpenCSG는 Fineweb-edu-chinese, Cosmopedia-chinese, Smoltalk-chinese와 같은 다양한 데이터셋을 포함하고 있습니다.

- **Technical Details**: Fineweb-edu 데이터셋은 다양한 중국 웹 소스에서 필터링된 고품질 콘텐츠에 중점을 두고 있습니다. Cosmopedia-chinese는 지식 집약적인 훈련을 위한 합성 교과서 스타일의 데이터를 제공하며, Smoltalk-chinese는 다양한 대화 형식의 데이터를 강조합니다. 이 데이터셋은 높은 품질의 텍스트와 다양한 도메인 커버리지를 갖추고 있으며, 확장 가능하고 재현 가능한 데이터 큐레이션 프로세스를 특징으로 합니다.

- **Performance Highlights**: 이 논문에서는 소규모 매개변수 모델에 대한 평가를 포함한 광범위한 실험 분석을 수행하였습니다. C-Eval과 같은 작업에서 성능 개선이 두드러지게 나타났으며, OpenCSG Chinese Corpus가 중국어 LLM 훈련에 효과적임을 보여주었습니다.



### A Multi-Modal AI Copilot for Single-Cell Analysis with Instruction Following (https://arxiv.org/abs/2501.08187)
Comments:
          37 pages; 13 figures; Code: this https URL Models: this https URL, this https URL

- **What's New**: 새로운 기술인 InstructCell은 자연어 처리(NLP)를 기반으로 한 multi-modal AI copilot으로, 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터를 보다 효과적으로 분석할 수 있도록 돕습니다. 이 시스템은 서로 다른 조직과 종에서 수집된 scRNA-seq 프로파일과 텍스트 기반 지침을 결합한 종합적인 데이터셋을 통해 개발되었습니다. 사용자는 InstructCell을 통해 세포 유형 주석, 조건부 pseudo-cell 생성, 약물 감수성 예측과 같은 중요한 작업을 쉽고 직관적인 자연어 명령으로 수행할 수 있습니다.

- **Technical Details**: InstructCell은 Q-Former 모듈, 사전 훈련된 언어 모델(LLM), 그리고 세포 재구성 블록을 포함하는 복합 다중 모달 세포 언어 아키텍처를 통해 텍스트 정보와 단일 세포 유전자 발현 데이터를 동시에 처리할 수 있습니다. 이 시스템은 다중 세포 분석 작업에 필수적인 지침-응답 쌍을 생성하여 다양한 실험적 조건에 적응할 수 있도록 설계되었습니다. 또한, 사용자의 다양한 배경과 언어 스타일에 따라 지침을 처리할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: InstructCell은 기존의 단일 세포 기초 모델들의 성능을 넘는 결과를 보이며, 각 아키텍처 요소의 필요성을 검증하는 실험을 통해 생물학적 통찰력을 발견하는 능력을 갖추고 있습니다. 이러한 기술은 사용자가 복잡한 단일 세포 데이터를 탐색할 수 있도록 지원하며, 기술 장벽을 낮추고 생물학에 대한 깊은 통찰력을 가능하게 합니다. InstructCell은 유연한 자연어 명령을 사용하여 실질적이고 중요한 생물학적 작업을 수행할 수 있는 직관적인 도구를 제공합니다.



### Potential and Perils of Large Language Models as Judges of Unstructured Textual Data (https://arxiv.org/abs/2501.08167)
Comments:
          11 pages, 1 appendix

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 다른 LLM이 생성한 요약의 주제 일치를 평가하는 것으로 사용될 수 있는지 조사합니다. 대형 언어 모델의 발전이 비구조적 텍스트 데이터의 처리 및 요약에 눈에 띄는 능력을 제공하였으며, 이는 설문 응답 분석에 특히 중요한 의미를 갖습니다. 이 연구는 LLM을 사용하여 생성된 요약의 주제와 실제 응답 간의 불일치가 조직의 의사 결정에 미치는 영향에 대한 우려를 다룹니다.

- **Technical Details**: 연구에서는 Anthropic Claude 모델을 이용하여 개방형 설문 응답으로부터 주제 요약을 생성하고, Amazon의 Titan Express, Nova Pro, Meta의 Llama를 LLM 평가자로 활용했습니다. LLM을 평가자로 사용하는 접근 방식은 Cohen의 kappa, Spearman의 rho, Krippendorff의 alpha를 사용하여 인간 평가와 비교하였습니다. 이 방법은 전통적인 인간 중심 평가 방법에 대한 확장 가능한 대안을 검증합니다.

- **Performance Highlights**: 결과적으로 LLM을 평가자로 사용하는 접근 방식은 인간 평가자와 비슷한 성능을 보였으나, 인간은 여전히 맥락에 따라 미세한 뉘앙스를 탐지하는 데 있어 우수할 수 있습니다. 이 연구는 AI 지원 텍스트 분석에 대한 지식을 확장하는 데 기여하며, 다양한 맥락과 사용 사례에 LLM 평가 모델을 일반화할 때 신중한 고려가 필요함을 강조합니다.



### Refusal Behavior in Large Language Models: A Nonlinear Perspectiv (https://arxiv.org/abs/2501.08145)
- **What's New**: 이 논문은 대형 언어 모델(LLM)에서의 거부(Refusal) 행동을 조사하였으며, 이를 통해 윤리적인 기준에 부합하도록 유해하거나 부적절한 프롬프트에 응답하지 않도록 하는 기전을 탐구하였습니다. 다양한 구조의 LLM 6종에서 거부 메커니즘의 비선형적이고 다차원적인 특성을 발견하였고, 이는 기존의 선형적 접근 방식에 도전하는 내용입니다. 결과적으로, 이러한 발견은 LLM의 안전한 배치를 위한 보다 나은 연구 방향을 제시합니다.

- **Technical Details**: 연구 방법으로는 거부 행동을 구분 짓기 위한 실험적 과제를 설계하고, 이를 통해 모델의 반응을 분석하였습니다. 두 개의 데이터셋을 사용하여 유해한 지침과 무해한 지침에 대한 LLM의 반응을 추적하였습니다. 선형(PCA) 및 비선형(t-SNE, UMAP) 차원 축소 기법을 통해 각 레이어에서의 활성화 복잡성을 분석하며, 이러한 활성화가 구조마다 어떻게 다른지를 규명하였습니다.

- **Performance Highlights**: 연구 결과는 LLM의 거부 메커니즘이 단순한 선형적 현상이 아닌, 복잡하고 비선형적이라는 것을 보여주었습니다. 우리는 새로운 활성화 클러스터가 등장하는 것을 확인하였으며, 이를 통해 각 모델 아키텍처의 특정적인 특성과 더불어 거부 행동의 본질적인 이해가 가능하게 되었습니다. 이 발견들은 향후 LLM을 윤리적 기준에 맞추기 위한 중요한 기초 자료로 활용될 것입니다.



### Consistency of Responses and Continuations Generated by Large Language Models on Social Media (https://arxiv.org/abs/2501.08102)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 소셜 미디어에서 감정 콘텐츠를 처리하고 의미론적 관계를 유지하는 방법을 조사했습니다. Gemma와 Llama라는 두 개의 오픈소스 모델을 활용하여 기후 변화에 대한 트위터와 레딧의 논의를 분석하였으며, LLM이 생성한 텍스트의 감정 일관성과 의미 유사성을 평가했습니다. 이는 LLM의 감정 및 의미 처리 능력에 대한 귀중한 통찰을 제공합니다.

- **Technical Details**: 연구는 LLM의 두 가지 주요 텍스트 생성 작업인 응답 생성(response generation)과 연속성 작업(continuation tasks)에 중점을 두었습니다. Gemma는 부정적인 감정 증폭 경향을 보였고, Llama는 더 넓은 감정 스펙트럼에서 감정 보존에서 뛰어난 성과를 나타냈습니다. 두 모델 모두 인간이 작성한 내용에 비해 감정 강도가 줄어든 응답을 생성하고, 긍정적인 감정으로의 편향을 보였습니다.

- **Performance Highlights**: 연구 결과, LLM 모델은 높은 의미 일관성을 유지했지만, 감정 패턴에서의 뚜렷한 차이를 보여주었습니다. 특히, Gemma는 분노와 같은 부정적인 감정을 강화하는 경향을 보인 반면, Llama는 다양한 감정에서 뛰어난 보존성을 보였습니다. 이러한 발견은 LLM의 감정 및 의미 처리 능력을 이해하고, 소셜 미디어와 인간-AI 상호작용 디자인에서의 활용 가능성을 논의하는 데 기여합니다.



### Dynamic Multimodal Sentiment Analysis: Leveraging Cross-Modal Attention for Enabled Classification (https://arxiv.org/abs/2501.08085)
- **What's New**: 이 논문은 텍스트, 오디오 및 시각적 데이터를 통합한 멀티모달 감정 분석 모델의 개발을 탐구합니다. 이 모델은 감정 분류를 개선하고 서로 다른 모달리티 간의 복잡한 상호작용을 캡처하여 더 정확하고 미세한 감정 해석을 가능하게 하려는 목표를 가지고 있습니다. 특히, 여러 가지 피처 융합 전략을 평가하여 초기 단계 융합이 향상된 성능을 나타낸다는 결과를 도출하였습니다.

- **Technical Details**: 감정 분석 모델은 트랜스포머 기반 아키텍처 안에서 배치된 세 가지 주요 피처 융합 전략을 탐구합니다: 지연 단계 융합(late stage fusion), 초기 단계 융합(early stage fusion), 다중 헤드 주의(multi-headed attention)입니다. CMU-MOSEI 데이터셋을 사용하여 실험하였으며, 텍스트, 오디오 및 시각 입력이 감정 점수와 함께 동기화되어 제공됩니다. 여러 모달리티의 기능을 동적으로 선택하기 위해 크로스 모달 주의(cross-modal attention) 메커니즘을 활용합니다.

- **Performance Highlights**: 실험 결과, 초기 단계 융합이 지연 단계 융합보다 현저히 우수한 성능을 보여 71.87%의 정확성을 달성했습니다. 또한, 다중 헤드 주의 접근 방식은 72.39%의 정확도를 기록하며 경미한 성과 향상을 나타냈습니다. 향후 작업은 기능 융합 기술을 개선하고, 동적 기능 가중치를 탐색하여 모델 성능을 더욱 향상시키는 방향으로 진행될 것입니다.



### Exploring Narrative Clustering in Large Language Models: A Layerwise Analysis of BER (https://arxiv.org/abs/2501.08053)
Comments:
          arXiv admin note: text overlap with arXiv:2408.03062, arXiv:2408.04270, arXiv:2307.01577

- **What's New**: 이번 연구는 BERT의 내부 메커니즘을 분석하여 서사 내용과 저자 스타일의 클러스터링 능력을 살펴보았습니다. GPT-4를 활용하여 개발한 다양한 의미적 콘텐츠와 스타일적 변화를 포함한 데이터를 사용하여 BERT의 레이어별 활성화를 분석했습니다. 연구 결과, BERT는 후반 레이어에서 강력한 서사 내용의 클러스터링을 보여주며, 스타일은 개별 작가보다 서사로 더 잘 클러스터링되었습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 다양한 저자 스타일과 서사 주제를 포함하도록 선택되었습니다. BERT의 [CLS] 토큰 임베딩은 768차원으로, 이를 13개의 레이어에서 추출하여 (13, 1000, 768) 형태의 데이터 구조를 만들었습니다. 분석을 위해 주성분 분석(Principal Component Analysis, PCA)과 다차원 스케일링(Multidimensional Scaling, MDS) 기법을 활용하여 BERT의 내부 활성화를 시각화했습니다.

- **Performance Highlights**: BERT는 강력한 서사 내용 클러스터링을 보여주며, 이러한 클러스터는 후속 레이어에서 점점 더 컴팩트하게 형성됩니다. 반면, 저자 스타일에 따른 클러스터링은 미미하게 나타났습니다. 이 연구는 BERT가 의미적 콘텐츠를 더 우선시한다는 점을 강조하며, 변환기 모델의 언어 정보 인코딩 방식에 대한 이해를 향상시키는 데 기여합니다.



### READ: Reinforcement-based Adversarial Learning for Text Classification with Limited Labeled Data (https://arxiv.org/abs/2501.08035)
- **What's New**: 이번 논문에서는 지식이 미비한 레이블 데이터(labelled data)로부터의 개선 방법으로서 새로운 접근인 READ(신뢰 기반 적대적 학습)를 제안합니다. 대체로 레이블 데이터는 수집하기 어렵고 비용이 많이 들어가지만, 레이블이 없는 데이터(unlabelled data)는 비교적 저렴하게 획득할 수 있습니다. 따라서 이 연구는 강화 학습(reinforcement learning) 기반 텍스트 생성과 반지도 학습(semi-supervised learning)을 결합하여 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: READ(강화 기반 적대적 학습)는 레이블 데이터와 레이블이 없는 데이터의 조합을 활용하여 텍스트 생성 및 모델 훈련을 진행합니다. 이 방법은 강화 학습 프레임워크를 통해 합성 텍스트(synthetic text)를 생성하고, 이러한 과정을 적대적 학습(adversarial learning)과 연결하여 모델의 일반화 능력을 향상시킵니다. 실험을 통해 READ는 여러 데이터셋에서 기존의 반지도 학습 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: READ 방법은 다양한 프리트레인(pre-trained) 모델을 통해 여러 데이터셋에서 기존의 최첨단(state-of-the-art) 방법을 초월하는 성과를 보였습니다. 생성하는 텍스트의 질과 모델의 일반화 능력을 평가한 결과, READ가 텍스트 생성 품질과 더불어 모델의 일반화 능력에 긍정적인 영향을 미친다는 것을 경험적으로 입증하였습니다. 이 연구는 텍스트 생성과 적대적 학습의 통합이 모델 성능에 미치는 중요성을 강조합니다.



### TriAdaptLoRA: Brain-Inspired Triangular Adaptive Low-Rank Adaptation for Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2501.08008)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 효율적인 미세 조정을 위한 새로운 방법인 Triangular Adaptive Low-Rank Adaptation (TriAdaptLoRA)를 제안합니다. TriAdaptLoRA는 신경과학 원리를 기반으로 하여 동적으로 학습 가능한 매개변수의 할당을 최적화하여 미세 조정의 효율성을 높입니다. 이 방법은 이전의 Parameter-Efficient Fine-Tuning (PEFT) 기법들에서 보여준 한계를 극복하기 위해 고안되었습니다.

- **Technical Details**: TriAdaptLoRA는 다음 세 가지 주요 혁신을 포함합니다. 첫째, 변환 행렬을 하삼각형 및 상삼각형 구성 요소로 분할하여 매개변수 활용도를 극대화합니다. 둘째, 효율적인 적응을 위한 중요도 척도로 정규화된 Frobenius 놈을 사용합니다. 셋째, 동적 임계값에 따라 조정되는 적응형 순위 성장 전략을 통해 학습 단계 간 유연한 매개변수 할당이 가능합니다.

- **Performance Highlights**: 실험 결과, TriAdaptLoRA는 다양한 자연어 이해 및 생성 작업에서 기존 PEFT 방법들보다 일관되게 우수한 성능을 나타냈습니다. 특히, GLUE 벤치마크와 SQuAD 2.0 작업에서 현저한 성과 향상을 기록하여 자원 집약적 문제를 해결하는 효과적인 미세 조정 솔루션으로 자리매김했습니다.



### Formalising lexical and syntactic diversity for data sampling in French (https://arxiv.org/abs/2501.08003)
- **What's New**: 이번 연구에서는 데이터셋의 다양성을 높이는 Heuristic 기법을 제안합니다. 기존의 랜덤 샘플링보다 더 높은 다양성을 제공하는 방법을 탐구하면서, 어휘적(diversity) 및 구문적 다성과의 상관관계를 분석합니다. 이런 접근 방식은 데이터의 다양성과 품질을 극대화하는 데 도움이 될 수 있습니다.

- **Technical Details**: 연구는 생태학(ecology)에서 영감을 받은 다양성 측정을 NLP 데이터셋에 적용하고자 하며, 주로 엔트로피(entropy)를 활용합니다. 주어진 연구 질문을 통해 무작위로 선택한 것보다 현저히 다양한 코퍼스를 선택할 수 있는지, 어휘적 다양성이 구문적 다양성에 기여할 수 있는지를 분석합니다. 데이터를 미리 파싱(parsing)하는 것이 필요하여 구문적 다양성의 양적 측정이 어려움을 극복해야 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 무작위 샘플링에 비해 상당히 높은 다양성을 목표로 하는 코퍼스를 선택하는 데 성공적으로 작용함을 보여주었습니다. 또한 Shannon-Weaver 엔트로피에만 집중하지 않고, 다양성 측정의 일반화를 고려하는 것이 중요하다는 것을 확인했습니다. 이는 다양한 데이터셋과 다양성 측정의 버전에서 변동하는 상관관계를 드러냅니다.



### "Wait, did you mean the doctor?": Collecting a Dialogue Corpus for Topical Analysis (https://arxiv.org/abs/2501.07947)
- **What's New**: 이 논문에서는 대화의 주제를 식별하는 방법에 대한 연구 결과를 제시하고 있습니다. 특히, 일상 대화에서 주제가 어떻게 조직되는지에 대한 기존 문헌이 부족하다는 점을 지적합니다. 이 연구는 여러 주제와 주제 전환 유형을 포함할 수 있는 대화의 데이터 수집과 주석 작업의 어려움에 대해서도 언급합니다.

- **Technical Details**: 저자들은 새로운 메시징 도구를 활용하여 대화 수집 실험을 진행할 계획임을 밝혔습니다. 이로 인해 생성되는 데이터는 다양한 주제를 분석하기에 적합한 말뭉치(corpus)를 구축하는 데 중점을 둡니다. 대화의 길이와 주제 전환을 고려하여, 수집된 데이터는 보다 깊이 있는 주제 분석을 가능하게 할 것입니다.

- **Performance Highlights**: 대화 분석을 위한 데이터 수집에 있어 새로운 접근 방식을 제시한 이번 연구는 주제 인식의 중요성을 강조합니다. 특히, 이 실험을 통해 수집될 데이터는 대화의 주제 변화 및 그 인식에 대한 심층적인 분석을 지원할 것으로 기대됩니다.



### GRAPHMOE: Amplifying Cognitive Depth of Mixture-of-Experts Network via Introducing Self-Rethinking Mechanism (https://arxiv.org/abs/2501.07890)
Comments:
          10 pages

- **What's New**: GRAPHMOE는 Mixture-of-Experts (MoE) 네트워크의 성능을 향상시키기 위한 새로운 방법으로, Pseudo GraphMoE 네트워크를 기반으로 한 자기 재사고(self-rethinking) 메커니즘을 도입합니다. 본 연구에서는 전문가 모델들 간의 협업을 통해 MoE 네트워크의 인지적 깊이(cognitive depth)를 증대시키는 기법을 제안합니다. 이 접근법은 기존 LoRA 기반 모델들을 능가하며 최첨단(state-of-the-art) 성과를 도출합니다.

- **Technical Details**: GRAPHMOE는 고유한 반복 라우팅(recurrent routing) 전략을 활용하여 전문가 노드 간의 정보 흐름을 촉진하고, 입력 데이터를 여러 번의 추론 사이클을 통해 지속적으로 검토 및 정제합니다. 이러한 과정을 통해 SNS 메시지 전송과 유사한 방식으로 전문가 모델의 출력이 후속 전문가 배치 선택을 위한 신호로 전송됩니다. LoRA 기술을 통해 GraphMoE 아키텍처를 구현하였으며, 여러 벤치마크에서 감독된 미세 조정(Supervised Fine-Tuning)으로 평가되었습니다.

- **Performance Highlights**: GRAPHMOE는 표준 LoRA+MoE 프레임워크와 비교하여 학습된 전문가 모델의 특성을 기능적으로 통합하고, 정보 전송을 통해 문제 해결의 효율성을 향상시킵니다. 실험 결과 특히 고전 LoRA+MoE 방식보다 메모리 사용량(메모리 사용량 감소)을 유지하면서도 연산 반복 횟수를 증가시켜 성능을 개선하였습니다. 본 연구는 신경망이 인간과 유사한 반복적 추론 과정을 모방할 수 있도록 하는 '자기 재사고 메커니즘'을 소개하여, LLM의 추론 능력을 높이는 데 중요한 기여를 하고 있습니다.



### Continual Learning with Embedding Layer Surgery and Task-wise Beam Search using Whisper (https://arxiv.org/abs/2501.07875)
Comments:
          Published in 2024 IEEE Spoken Language Technology Workshop

- **What's New**: 이번 논문에서는 Multilingual ASR 모델의 한계를 극복하기 위해 새로운 언어를 추가하는 Continual Learning (CL) 방법을 제안합니다. 기존의 CL 방법들이 Catastrophic Forgetting (CF) 문제를 간과하고 있는 반면, 우리는 Embedding Layer Surgery라는 새로운 기법을 도입하여 새로운 언어에 대한 복사본을 생성하는 방식을 보여줍니다.

- **Technical Details**: 우리는 새로운 언어별로 별도의 토큰 임베딩(token embedding) 복사본을 생성하고, 해당 언어의 전사(transcribing) 과정에서 가장 적절한 복사본을 선택하여 기존 언어의 임베딩을 대체합니다. 하지만, 언어 유형 식별(Language Identification, LID) 오류로 인해 잘못된 임베딩 선택이 발생할 수 있으며, 이를 해결하기 위해 Task-wise Beam Search를 적용하여 실수를 스스로 수정할 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법을 통해 Common Voice의 10개 보지 못한 언어에 대해 10시간의 데이터를 조정한 결과, 사전 훈련된 언어의 Average WER (AWER)가 14.2%에서 11.9%로 감소하였습니다. 이 과정에서 새로 추가된 언어의 WER은 손상되지 않았으며, 성능이 유지되었습니다.



### ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding (https://arxiv.org/abs/2501.07861)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 추론 능력을 개선하기 위한 새로운 프레임워크인 ReARTeR를 제안합니다. ReARTeR는 테스트 시간에 Trustworthy Process Rewarding을 도입하여 정확한 점수 평가와 자연어 설명 생성을 통해 단계별 정제를 가능하게 합니다. 이를 통해 복잡한 다단계 추론 임무에서의 한계를 극복하고, RAG 시스템의 지식 활용을 극대화하는 데 기여할 수 있습니다.

- **Technical Details**: ReARTeR는 두 가지 주요 모델을 활용하여 수행됩니다: Process Reward Model (PRM)과 Process Explanation Model (PEM)입니다. PRM은 스칼라 점수를 정확하게 평가할 수 있도록 설계되었으며, PEM은 자연어 설명을 생성합니다. 또한, Monte Carlo Tree Search를 활용하여 고품질 단계별 선호 데이터를 수집하고 Iterative Preference Optimization을 통해 최적화합니다.

- **Performance Highlights**: 다단계 추론 벤치마크에서의 실험 결과, ReARTeR는 기존 방법들에 비해 유의미한 성능 개선을 보여주었습니다. 특히, PRM과 PEM 간의 불일치 문제나 훈련 데이터의 편향 문제를 해결하면서, 조기 단계 편향을 감소시키는 성과를 달성했습니다. 이 결과는 ReARTeR가 RAG 시스템의 추론 역량을 발전시킬 잠재력이 있음을 시사합니다.



### Optimizing Language Models for Grammatical Acceptability: A Comparative Study of Fine-Tuning Techniques (https://arxiv.org/abs/2501.07853)
- **What's New**: 이번 연구는 CoLA 데이터셋을 사용하여 Open Pre-trained Transformer (OPT-125M)의 세밀 조정(fine-tuning, FT)을 탐구합니다. Vanilla-Fine-Tuning (VFT), Pattern-Based-Fine-Tuning (PBFT), Low-Rank Adaptation (LoRA)와 같은 매개변수 효율적인 세밀 조정(PEFT) 기술을 비교하여 계산 효율성을 크게 개선하면서도 높은 정확도를 유지하는 방법을 보여줍니다.

- **Technical Details**: 실험 결과, VFT는 81.2%의 가장 높은 정확도를 달성했지만, LoRA를 통한 FT는 메모리 사용량과 반복 시간을 50% 이상 줄이며 PBFT의 정확도를 향상시켰습니다. Context Distillation (CD)는 계산 효율성이 높음에도 불구하고 약 31%의 정확도로 떨어지는 결과를 보였습니다.

- **Performance Highlights**: 이 연구는 대규모 언어 모델(Large Language Models, LLM)에 대한 접근을 민주화하는 데 기여하고 있으며, 계산 장벽을 줄이는 방향으로 이바지하고 있습니다. 실험에서 보여진 기술들은 다양한 NLP 작업에서의 효율성을 향상시킬 수 있는 가능성을 제시합니다.



### Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning (https://arxiv.org/abs/2501.07845)
- **What's New**: 이 논문에서는 Reasoning with Graphs (RwG)라는 새로운 접근 방식을 제안합니다. 이는 주어진 맥락에서 명시적인 그래프를 먼저 구축한 후, 이 그래프를 활용하여 LLM의 추론 성능을 향상시키는 방법입니다. RwG는 기존의 접근 방식과 달리 LLM의 생각에 기반한 트리나 그래프를 만들어내는 것이 아니라, 문제의 맥락에서 직접적으로 명시적인 그래프를 생성합니다.

- **Technical Details**: RwG 방법에서는 여러 단계의 검증을 통해 주어진 맥락에서 그래프를 생성하는 방식을 설계하였습니다. 이 방식에서는 노드가 문제의 맥락 내의 개체를 나타내며, 이 그래프가 LLM의 추론 능력을 평가하는 데 사용됩니다. 실험 결과, 제안된 방법이 여러 LLM에서 논리적 추론 및 다단계 질문 응답 작업에서 성능을 유의미하게 향상시킴을 보여주었습니다.

- **Performance Highlights**: 실험 결과, RwG 방식이 LLM의 성능을 크게 개선하는 데 기여함을 입증하였습니다. 특히 논리적 추론 및 다단계 질문 응답 작업에서 높은 성과를 기록하였으며, 이는 맥락에서 도출된 명시적 그래프 구조를 활용한 결과입니다. 이러한 결과는 LLM 작업에 구조화된 지식을 통합하는 새로운 방향성을 제시합니다.



### Real-time Verification and Refinement of Language Model Text Generation (https://arxiv.org/abs/2501.07824)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 Streaming-VR(Streaming Verification and Refinement)라는 새로운 방식을 제안합니다. 이 방법은 LLM이 생성하는 토큰을 실시간으로 검증하고 수정하여 전체 생성 이후가 아닌 지속적인 검사를 통해 오류 전파를 방지합니다. 따라서 LLM의 응답 생성 중에 정확성을 높이고 비효율성을 줄이는 것을 목표로 합니다.

- **Technical Details**: Streaming-VR은 외부 검증 모델을 활용하여 LLM이 생성하는 각 토큰을 실시간으로 확인하고, 잘못된 부분은 즉시 수정하는 방식으로 작동합니다. 이 프로세스는 기존의 후속 수정 방식보다 더 빠르고 효율적이며, 결과적으로는 사실적 정확성을 개선합니다. 실험 결과, Streaming-VR을 통해 LLM 품질이 대폭 향상된다는 것을 입증했습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 Streaming-VR의 효과를 검증한 결과, ASQA와 QuoteSum 데이터셋에서 평균 각각 39.8% 및 31.5% 더 높은 효율성을 달성했습니다. 또한, 조기 생성된 잘못된 토큰은 후속 문장이 사실적으로 부정확할 확률을 약 37.6% 증가시켜 Streaming-VR의 필요성을 강조합니다.



### A Multi-Encoder Frozen-Decoder Approach for Fine-Tuning Large Language Models (https://arxiv.org/abs/2501.07818)
- **What's New**: 본 연구에서는 멀티태스킹(multi-task) 환경에서 디코더를 고정(freezing)하는 것이 자연어 처리(NLP) 작업의 성능을 어떻게 개선하는지를 조사합니다. 특히, 다양하고 복잡한 자연어 작업에 대한 성능 향상과 배포(overhead) 비용 절감을 목표로 합니다. 실험 결과, 디코더를 고정하는 방식은 다언어(multilingual) 작업에서의 재앙적 망각(catasrophic forgetting)을 줄여주는 것으로 나타났습니다. 또한, 큰 모델과 결합하면 구조화된 작업 및 질문 응답(QA) 작업에서 성능을 유지하거나 향상시킬 수 있습니다.

- **Technical Details**: 연구에 사용된 모델은 AlexaTM로, 12개의 인코더와 12개의 디코더로 구성된 5억 1100만 개의 파라미터를 가진 모델입니다. 실험은 여러 NLP 데이터셋을 통해 진행되었으며, 학습 스케줄 및 최적화는 Adam Optimizer를 사용하여 진행되었습니다. 특히, 디코더를 고정하였다 하더라도 모델의 크기를 늘림으로써 발생할 수 있는 성능 저하를 완화하는 방법도 모색되었습니다. 이러한 접근 방식은 인코더와 디코더의 별개 학습을 통해 성능을 극대화하는 데 기여합니다.

- **Performance Highlights**: 결과적으로 디코더를 고정할 경우 XSUM과 WebNLG 작업에서 성능이 약 2% 감소했지만, 더 큰 고정 디코더를 사용할 경우 성능이 유지되었습니다. ROUGE, BLEU, NIST와 같은 다양한 평가 지표를 통해 측정된 이러한 성능은 자연어 생성(NLG) 작업에서 디코더의 역할이 중요하다는 것을 보여줍니다. 따라서, 고정된 디코더를 활용하는 접근 방식은 다양한 자연어 처리 작업에서 효과적인 전략으로 자리잡을 수 있을 것으로 보입니다.



### Large Language Models for Knowledge Graph Embedding Techniques, Methods, and Challenges: A Survey (https://arxiv.org/abs/2501.07766)
- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 지식 그래프 임베딩(Knowledge Graph Embedding, KGE) 관련 작업에 적용하는 다양한 방법을 탐구합니다. 특히 KGE 시나리오의 특성에 따라 다중 모달 KGE(multi-modal KGE)와 오픈 KGE(open KGE)와 같은 다양한 유형의 KGE에서 LLM의 활용도를 조명합니다. 이를 통해 LLM을 KGE 작업에 통합하기 위한 새로운 접근 방식과 분류를 제시합니다.

- **Technical Details**: 저자는 LLM이 방대한 텍스트 데이터를 학습하여 다음 단어를 예측하거나 주어진 텍스트와 관련된 콘텐츠를 생성하는 자연어 처리(Natural Language Processing, NLP) 딥러닝 모델로 작용한다고 설명합니다. 또한, 각 KGE 시나리오에 대해 분류하고 이를 비교할 수 있는 방법론과 각각의 소스 코드 링크를 포함한 표 형태의 개요를 제공합니다. 이러한 접근법은 LLM 관련 작업을 보다 일관되게 수행하기 위한 기초를 마련합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 다양한 KGE 시나리오에서 LLM의 적용 가능성을 검토하고, 향후 연구 방향성을 제안합니다. 연구에서 제시된 분류 및 방법론은 LLM과 KGE의 경계가 모호해지는 상황에서 유용한 참고 자료가 될 것입니다. LLM과 KGE의 융합은 향후 지식 그래프의 성능을 더욱 향상시키는 데 기여할 것으로 기대됩니다.



### Advancing Student Writing Through Automated Syntax Feedback (https://arxiv.org/abs/2501.07740)
Comments:
          This paper has been accepted for presentation at AIEER 2024

- **What's New**: 본 연구는 학생들의 구문(proficiency) 능력을 향상시키기 위한 구문 피드백(syntax feedback)의 중요성을 강조합니다. 학생들이 구문 규칙에서 겪는 어려움을 인식하고, 이를 개선할 수 있도록 돕기 위해 Essay-Syntax-Instruct라는 전문 데이터셋을 개발했습니다. GPT3.5-Turbo, Llama-2-7b-chat-hf 등과 같은 거대 언어 모델(LLM)의 능력을 활용하여, 구문 개선 작업에 맞춰 세심하게 조정된 모델을 제시하고 있습니다.

- **Technical Details**: 연구는 구문 규칙을 이해하고 적용하는 데 도움을 주기 위해 최신 LLM 기술을 활용했습니다. 특히, 이 연구는 LLM을 세밀하게 조정(fine-tuning)하여 교육적 과제를 충족시키고, 구문 능력을 증진시키기 위한 공정한 학습 도구를 구축하고자 하였습니다. 본 연구에서 개발한 데이터셋은 LLM의 효과성을 높이고, 학생들이 구문 오류를 쉽게 인지하고 수정할 수 있도록 도와줍니다.

- **Performance Highlights**: 연구 결과, 조정된 LLM들은 구문 관련 문제를 다루는 데 있어 현저한 향상을 이뤘습니다. 이러한 LLM의 발전은 학생들에게 구문 오류를 인식하고 수정하는 데 큰 도움이 됩니다. 본 연구는 LLM을 통한 언어 학습의 가능성을 제시하며, 이 기술들이 학생들의 언어 습득 과정에서 보여줄 수 있는 잠재력을 알립니다.



### Exploring the encoding of linguistic representations in the Fully-Connected Layer of generative CNNs for Speech (https://arxiv.org/abs/2501.07726)
- **What's New**: 이번 연구는 학습된 기하학적인 언어 정보가 논리적으로 연결되는 제너레이티브 CNN의 전결합층(fully-connected layer)에서 어떻게 표현되는지를 최초로 탐구합니다. ciwGAN의 특수한 구조를 통해, 음성 합성을 위한 CNN에서 미세한 언어적 정보를 인코딩하는 방식을 분석하고, 전결합층에서의 가중치 행렬(weight matrix)의 조작 방법을 제시합니다. 이 연구는 음성 합성의 의미 전달을 명확하게 하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: ciwGAN은 WaveGAN 모델을 바탕으로 하여, 생성기(generator), 판별기(discriminator), 그리고 Q-네트워크(Q-network)의 세 가지 네트워크 구조로 구성됩니다. 이 모델의 중요한 특징은 생성기가 잠재 공간(latent space)에서 샘플링하여 파형 출력을 생성하고, Q-네트워크가 일종의 원-핫 인코딩(one-hot encoding)으로 초기화된 생성기의 출력을 추정하는 것입니다. 이를 통해 모델은 의미 있는 정보 출력을 극대화하는 압력을 받으며, 서로 다른 모델 점수와 훈련 샘플 간의 Wasserstein 거리(Wasserstein distance)를 최적화합니다.

- **Performance Highlights**: 연구 결과, ciwGAN의 전결합층에서 레벨 특정 가중치 행렬이 언어적 정보의 표현에 어떻게 기여하는지를 실험을 통해 입증하였습니다. 특히, 두 가지 실험을 통해 생성된 음파가 입증된 언어 구조를 따르며, 전결합층의 일부를 조작함으로써 특정 음소(phoneme)를 예측 가능하다는 것을 보여주었습니다. 이러한 접근은 ciwGAN이 언어 학습 이론에 기반하여 레벨 특정 하위 구조를 효율적으로 인코딩하고 있다는 것을 나타냅니다.



### ESURF: Simple and Effective EDU Segmentation (https://arxiv.org/abs/2501.07723)
- **What's New**: 본 논문은 Elemental Discourse Unit (EDU) 경계를 식별하고 이를 통해 EDU를 분할하는 새로운 방법인 EDU Segmentation Using Random Forests (ESURF)를 제안합니다. 기존의 방법이 고급 기능에 의존하는 반면, ESURF는 lexical 및 character n-gram 특징을 기반으로 하며, 랜덤 포레스트 분류기를 사용하여 간단하면서도 뛰어난 성능을 보여줍니다. 이 연구는 보다 훈련 효율적인 담화 분석 방법으로 나아가는 가능성을 제시합니다.

- **Technical Details**: RST(라이트 구조 이론)는 텍스트 내 전개 구조를 이해하기 위한 주요 기반을 제공합니다. EDUs는 텍스트 내에서 가장 작은 일관성 있는 '사고 단위'를 나타내며, 이들 간의 관계는 비대칭적 형태로 연결됩니다. 본 연구에서 제안한 ESURF는 EDU 식별을 위한 분류 문제로 접근하여, 후보 경계 주변의 9-token 서브시퀀스를 사용, morphologic 및 lexical 특성들을 고려하여 EDU 경계를 효과적으로 분류합니다.

- **Performance Highlights**: 실험 결과, ESURF는 기존의 최신 방법들과 비교해 개선된 EDU 식별률 및 RST 파싱 성능을 보여줍니다. 특히, EDU 세그멘테이션에 있어 단순한 방법이면서도 뛰어난 성과를 보이며, 전반적인 자동화된 RST 파싱 기술 향상에 기여하고 있습니다. 이로 인해 논문의 방법론은 향후 NLP 응용 분야에서 보다 효율적인 성과를 기대할 수 있습니다.



### LLMic: Romanian Foundation Language Mod (https://arxiv.org/abs/2501.07721)
- **What's New**: 이번 논문에서는 루마니아어를 위해 특별히 설계된 LLMic라는 이중 언어 기반 언어 모델을 소개합니다. 이 모델은 공개 데이터에서 자원이 제한된 언어에 대한 성능 부족 문제를 해결하기 위한 노력의 일환으로, 고유한 데이터 세트를 구축하고, 아키텍처를 선택하며, 하이퍼파라미터 최적화를 포함한 전처리 과정을 문서화했습니다.

- **Technical Details**: LLMic은 3B-파라미터의 바이링구얼 모델로, 루마니아어와 영어를 지원합니다. 데이터 수집 과정에서는 CommonCrawl의 방대한 데이터셋을 필터링하고 정리하며, OCR과 복잡한 형식을 분석해 텍스트를 추출하는 작업이 필요했습니다. 최적화된 BPE 토크나이저를 사용하여 루마니아어 텍스트의 처리 속도를 개선하였으며, 클리닝 기술을 통해 중복 데이터를 제거하고 다양한 출처의 데이터를 통합하여 훈련하였습니다.

- **Performance Highlights**: LLMic는 번역 작업에 이 모델을 미세 조정함으로써 기존 솔루션보다 더 나은 성능을 보여주었습니다. 특히 영어에서 루마니아어로의 번역 작업에서 성능이 향상되어, 큰 모델들에게 필적하는 결과를 달성하였습니다. 이는 루마니아어 언어 커뮤니티를 위한 효율적인 대규모 처리를 가능하게 하며, LLMic 모델의 공개로 자연어 처리(Natural Language Processing) 도구와 모델 개발을 촉진하고자 합니다.



### Entailed Between the Lines: Incorporating Implication into NLI (https://arxiv.org/abs/2501.07719)
- **What's New**: 이번 논문에서는 자연어 추론(Natural Language Inference, NLI)의 범위를 확장하여, 암시된 포함성(implied entailment)을 정식화하고 이에 대한 데이터 세트인 Implied NLI (INLI)를 도입합니다. 이는 최신 LLM(대형 언어 모델)이 암시된 포함성을 인식하고, 이와 함께 명시적 포함성과의 차이를 명확하게 구별할 수 있도록 도와줍니다. 이번 연구는 NLI 모델이 인간의 의사소통에서의 미묘함과 뉘앙스를 이해하는 데 중요한 기여를 하고자 합니다.

- **Technical Details**: 논문에서 제시된 INLI 데이터 세트는 10,000개의 실제 상황을 반영하는 전제와 40,000개의 암시적, 명시적, 중립적, 모순된 가설을 포함합니다. 암시된 포함성은 독자가 텍스트의 명시적 언어 이상을 이해해야 할 필요가 있는 가설로 정의되며, 이는 즉각적인 정보에서 유도되는 명시적 포함성과 구별됩니다. NLI 모델은 INLI에서 미세 조정(fine-tuning)을 통해 여러 데이터 세트와 도메인 간에 이 암시된 포함성을 이해하고 일반화하는 능력을 향상시킬 수 있습니다.

- **Performance Highlights**: T5-XXL 모델을 사용하여 INLI에서 암시적 и 명시적 포함성을 구별하는 작업을 수행한 결과, 테스트 정확도는 97.3%에 이릅니다. 그러나 기존 NLI 데이터 세트에서 암시적 포함성을 인식하는 능력은 다소 저조하며, 대부분의 모델들은 약 50%의 정확도로 무작위 추측 수준의 성능을 보입니다. 특히 ANLI 데이터 세트에서 훈련된 모델만이 암시적 포함성을 인식하는 데 유의미한 차이를 보였습니다.



### Benchmarking Abstractive Summarisation: A Dataset of Human-authored Summaries of Norwegian News Articles (https://arxiv.org/abs/2501.07718)
Comments:
          Accepted at NoDaLiDa2025

- **What's New**: 이 논문에서는 노르웨이어로 작성된 고품질의 인간 저자가 작성한 뉴스 기사 요약 데이터셋을 소개합니다. 이 데이터셋은 생성적 언어 모델의 추상적 요약 능력을 평가하기 위한 벤치마크를 제공합니다. 각 문서에는 노르웨이어 원어민이 작성한 세 개의 서로 다른 후보 금본 요약이 제공되며, 요약은 노르웨이어의 두 가지 서면 변형인 Bokmål(북말)과 Nynorsk(니노르스크)로 제공됩니다.

- **Technical Details**: 이 논문은 인간 저자가 작성한 요약을 만들기 위한 데이터 생성 노력과 함께 데이터셋.eval에 대한 기존 오픈 LLM 평가를 설명합니다. 데이터셋은 질 높은 뉴스 기사를 바탕으로 하며, 총 63개의 요약이 독립적으로 작성되었습니다. 요약은 주제와 중요성을 잘 반영해야 하며, 명확하고 간결해야 합니다.

- **Performance Highlights**: 우리의 결과는 노르웨이어 요약 능력에 대한 도전적인 LLM 벤치마크를 제공함을 보여줍니다. 또한 사람이 작성한 요약과 모델 생성 요약 간의 비교를 위해 수동 평가의 결과도 제공합니다. 이 작업은 노르웨이어 뉴스 데이터에 대해 처음으로 자유롭게 사용 가능한 수작업 요약 데이터셋을 제공함으로써 이 분야 연구의 발전에 기여합니다.



### Enhancing Talent Employment Insights Through Feature Extraction with LLM Finetuning (https://arxiv.org/abs/2501.07663)
- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)을 사용하여 비구조적인 구인 공고에서 미세하고 복잡한 직무 특성을 추출하는 방법을 탐구합니다. 120만 개의 구인 공고를 포함하는 데이터셋을 활용하여 원격 근무 가능성, 보상 구조, 교육 요건 등을 정확하게 분류하는 강력한 파이프라인을 개발했습니다. 또한, 전통적인 파싱 도구의 한계를 극복하기 위해 sematic chunking과 retrieval-augmented generation (RAG) 방식을 결합하여 더 높은 정확도를 달성하였습니다.

- **Technical Details**: 연구에서는 데이터 정리에 대한 탐색적 데이터 분석을 통해 직무 특성을 추출하기 위한 복잡한 절차를 기술했습니다. 10,000개의 샘플 데이터를 사용하여 초기 변수를 라벨링했고, 네 가지 DistilBERT 모델을 미세 조정하여 신뢰도 높은 특성을 추출했습니다. 연구팀은 또한 각 변수에 대해 개별적으로 조정된 모델을 사용하는 방식을 선호하였고, 이를 통해 탁월한 성능 향상을 달성했습니다.

- **Performance Highlights**: 이 연구는 LLMs의 노동 시장 분석을 위한 가능성을 부각시키며, 구인 데이터에 대한 보다 정확하고 실행 가능한 통찰력을 제공하는 기초를 마련했습니다. 특히, 비정상적인 보상 구조 및 원격 근무 카테고리를 구분하는 데 큰 개선을 보였으며, 모델의 강점과 한계에 대한 종합적 평가를 통해 향후 확장 가능성을 모색했습니다.



### GPT as a Monte Carlo Language Tree: A Probabilistic Perspectiv (https://arxiv.org/abs/2501.07641)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 자연어 처리(NLP) 문맥에서의 작동 메커니즘을 새로운 시각으로 분석합니다. 연구자는 언어 데이터 세트를 몬테 카를로 언어 트리(즉, "Data-Tree")로 표현할 수 있으며, 각 노드는 토큰을, 가장자리는 토큰 전이 확률을 나타냅니다. 또한 GPT 모델을 이러한 트리 형태로 변환할 수 있는 방법을 제안하여, 다양한 모델 간의 구조적 유사성을 정량적으로 평가할 수 있음을 강조합니다.

- **Technical Details**: 논문에서 제안하는 Data-Tree 구조는 주어진 데이터 세트의 각 토큰 및 전이 확률을 기반으로 구축됩니다. 이를 통해 LLM 모델이 훈련 데이터에 대해 어떻게 근사화하는지를 ilustrates합니다. 예를 들어, Token 간의 관계를 비주얼화하여 Data-Tree와 GPT-Tree의 구조적 유사성을 시각적으로 드러낼 수 있습니다. 이러한 접근 방식을 통해 LLM의 추론 과정이 단순한 고정 패턴 일치가 아닌 확률적인 패턴 일치임을 증명합니다.

- **Performance Highlights**: 연구 결과, 동일한 데이터 세트에서 훈련된 서로 다른 GPT 모델들은 Data-Tree와 매우 유사한 구조를 보인다고 보고합니다. 특히, 87% 이상의 GPT 출력 토큰이 Data-Tree에 의해 회상될 수 있는 것으로 나타났습니다. 이로써 GPT 모델의 성능이 Data-Tree와 얼마나 밀접하게 관련되어 있는지를 정량적으로 분석할 수 있었습니다. 또한, 모델이 드문 어휘를 사용할 경우 불확실한 경로를 탐색하게 되어 성능 저하로 이어질 수 있음도 발견했습니다.



### CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation (https://arxiv.org/abs/2501.08200)
Comments:
          to be published in LLM4Code 2025

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)에 의해 생성된 코드의 안전성을 평가하기 위해 새로운 평가 프레임워크인 CWEval을 도입했습니다. CWEval은 기능적 정 correctness 및 보안을 동시에 평가하는 고품질 작업 명세와 결과 중심의 테스트 오라클을 통해 LLM이 생성한 코드의 보안을 제고합니다. 이를 통해 LLM이 생성한 코드에서 종종 발견되는 기능적이지만 안전하지 않은 코드의 비율을 드러내며, 기존 평가의 부정확성을 입증합니다.

- **Technical Details**: CWEval은 인적 검증을 거친 고급 보안 코딩 작업과 포괄적인 명세, 기능성과 보안 모두에 대한 테스트 오라클을 통해 LLM의 코드 생성 보안 능력을 평가하는 데 초점을 맞추고 있습니다. 각 코딩 작업에는 안전한 솔루션과 취약한 대응이 포함되어 있으며, 명확한 문제 정의와 보안의 중요성을 강조합니다. CWEval은 모든 테스트에서 성공적으로 통과해야 하는 최적의 LLM 응답을 요구하여 보안 인식에서 코드 생성의 전반적인 기능을 동시에 평가합니다.

- **Performance Highlights**: CWEval과 CWEval-bench는 각각 보안-critical 코딩 벤치마크로, LLM이 생성한 코드의 보안 특성을 실험적으로 조사할 수 있는 기회를 제공합니다. 이 프레임워크는 LLM 코드 생성의 보안 위험을 포괄적으로 평가할 수 있도록 지원하며, 31종의 CWE를 포함하는 119개의 고품질 코딩 작업을 제공하여 이전 벤치마크의 한계를 극복합니다. 최종적으로, CWEval은 코드 생성의 안전성을 크게 향상시킬 수 있는 기초가 될 것입니다.



### In-situ graph reasoning and knowledge expansion using Graph-PReFLexOR (https://arxiv.org/abs/2501.08120)
- **What's New**: 이 논문에서는 Graph-PReFLexOR(Graph-based Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 그래프 추론(graph reasoning)과 기호 추상화(symbolic abstraction)를 결합하여 도메인 지식을 동적으로 확장할 수 있도록 설계되었습니다. 또한, 강화 학습(reinforcement learning)의 영감을 받아 지식을 생산하는 구조화된 맵을 정의합니다.

- **Technical Details**: Graph-PReFLexOR는 개념을 노드(node)로, 그 관계를 엣지(edge)로 인코딩하여 계층적 추론(hierarchical inference) 및 동적 학습(adaptive learning)을 지원합니다. 이러한 구조는 카테고리 이론(category theory)에 기초하여 구성되어 있으며, 지식 그래프(knowledge graphs), 추상 패턴(abstract patterns), 최종 답변(final answers)을 생성합니다. 이는 주어진 작업(task)에 따라 지식을 수집하고 사유를 구조적으로 지도합니다.

- **Performance Highlights**: 3억 개 파라미터를 갖춘 Graph-PReFLexOR 모델의 결과는 깊은 추론(depth of reasoning) 및 적응력(adaptability)에서 우수성을 보여줍니다. 이는 다학제간 연결을 촉진하기 위한 '지식 정원 성장(knowledge garden growth)' 전략과 함께, 투명하고 다학제적 AI 기반 발견의 가능성을 강조합니다. 최종적으로 이 연구는 일반 자율적 추론 솔루션에 대한 기초를 마련합니다.



### Optimizing Speech Multi-View Feature Fusion through Conditional Computation (https://arxiv.org/abs/2501.08057)
Comments:
          ICASSP 2025

- **What's New**: 본 연구에서는 self-supervised learning (SSL) 특징과 전통적인 spectral 특징인 FBanks 간의 업데이트 방향에서의 갈등을 해결하기 위해 새로운 generalized feature fusion framework를 제안합니다. 이 프레임워크는 gradient-sensitive gating network와 multi-stage dropout 전략을 통해 다양한 입력 특징의 통합을 가능하게 합니다. 이를 통해 SSL과 spectral 특징을 결합함으로써 모델의 수렴 속도를 가속화하고, 다중 음성 번역 작업에서 성능을 유지할 수 있게 됩니다.

- **Technical Details**: 우리는 end-to-end 음성 번역 시스템을 채택하며, 이 시스템은 acoustic encoder (A-enc), textual encoder (T-enc), decoder (dec)로 구성됩니다. 본 연구에서는 채택된 gradient-sensitive gating network (GSGN)가 S-feature와 FBanks 특징을 동적으로 융합하며, 다단계의 dropout 전략으로 과적합을 방지합니다. 이 구조는 residual network를 기반으로 하며, 새로운 특징을 통합하는 과정에서 각 특징의 가중치를 조정하여 최적의 결합을 달성합니다.

- **Performance Highlights**: MUSTC 데이터셋의 여러 언어 (En-De, En-Es, En-Fr)에 대한 실험 결과, 제안된 방법은 기존 FBanks 특징을 사용한 모델에 비해 평균 1.24배의 훈련 속도를 제공하면서도 유사한 성능을 달성했습니다. 또한, 미리 훈련된 ST 모델과 비교할 때도, SSL 특징이 충분히 유용할 경우 해당 훈련 가속 효과를 보장합니다. 이를 통해 음성-텍스트 복잡한 작업에서 두 가지 특징의 강점을 활용할 수 있음을 보여줍니다.



### Gandalf the Red: Adaptive Security for LLMs (https://arxiv.org/abs/2501.07927)
Comments:
          Niklas Pfister, Václav Volhejn and Manuel Knott contributed equally

- **What's New**: 이 논문은 기존의 프롬프트 공격에 대한 방어 평가에서 두 가지 중요한 요인을 간과하고 있음을 지적합니다. 첫째, 공격자의 동적 행동을 고려해야 하며 둘째, 엄격한 방어책이 정당한 사용자의 사용성에 미치는 부담을 고려해야 합니다. 이를 위해 논문에서는 D-SEC(Dynamic Security Utility Threat Model)을 제안하고 있습니다. D-SEC는 공격자와 정당한 사용자를 분리하고 다단계 상호작용을 모델링하여 보안과 유용성을 최적화하는 방법을 제시합니다.

- **Technical Details**: D-SEC는 공격의 적응성을 포착하고 보안과 유용성을 함께 고려할 수 있도록 설계되었습니다. 이를 통해 LLM 애플리케이션에 대한 방어를 평가하는 보다 합리적이고 포괄적인 방법을 제공하고자 합니다. 또한, 대규모 데이터셋을 생성하기 위해 crowd-sourced 플랫폼인 Gandalf를 구성하여 다양한 공격 시나리오를 수집하여 공개했습니다. 이러한 데이터셋은 279,000개의 프롬프트 공격을 포함하고 있으며, 이는 LLM의 보안성을 평가하는 데 유용한 자원이 될 것입니다.

- **Performance Highlights**: 이 논문은 보안과 유용성 간의 균형을 맞추는 것이 LLM 애플리케이션 설계에 있어서 필수적임을 강조합니다. 특히, 제한된 응용 프로그램 영역, 방어 심화(defense-in-depth), 적응형 방어 등 세 가지 전략이 효과적이라는 것을 보여줍니다. 이러한 전략들은 적절하게 조합되어 사용될 경우 사용성을 저해하지 않으면서도 효과적인 방어를 제공합니다. 이 연구 결과는 LLM 애플리케이션의 보안성과 유용성을 최적화하는 데 중요한 통찰을 제공합니다.



### Exploring Aviation Incident Narratives Using Topic Modeling and Clustering Techniques (https://arxiv.org/abs/2501.07924)
- **What's New**: 이 연구는 비행 안전(aviation safety)의 중요성을 강조하며, 여러 자연어 처리(Natural Language Processing, NLP) 기법을 통해 사고 분석을 심화하는 데 초점을 맞추고 있습니다. 특히 NTSB(National Transportation Safety Board) 데이터셋을 활용하여, 사건과 관련된 잠재적인 주제를 식별하고 시맨틱 관계(semantic relationships)를 탐색합니다.

- **Technical Details**: 연구에서는 LDA(Latent Dirichlet Allocation), NMF(Non-Negative Matrix Factorization), LSA(Latent Semantic Analysis), pLSA(Probabilistic Latent Semantic Analysis), K-means clustering 등의 고급 NLP 기법을 적용하여 사고를 군집화하고 공통 특성을 분석합니다. 이 과정에서 LDA의 일관성(coherence) 값이 0.597로 가장 뛰어난 성과를 보였고, pLSA, LSA, NMF는 각각 0.583, 0.542, 0.437의 결과를 보였습니다.

- **Performance Highlights**: K-means 클러스터링은 사고 내러티브(narratives)에서의 공통점과 독특한 통찰을 드러내며, 반복되는 테마에 대한 이해를 제공합니다. 이 연구는 사고 내러티브 안에 숨겨진 패턴(patters)과 주제 구조를 발견하고, 다양한 주제 모델링 기법의 비교 분석을 통해 비행 안전 개선에 기여할 수 있는 기초를 마련합니다.



### Aviation Safety Enhancement via NLP & Deep Learning: Classifying Flight Phases in ATSB Safety Reports (https://arxiv.org/abs/2501.07923)
Comments:
          NLP, Aviation Safety, ATSB, Deep learning, Flight phase. arXiv admin note: substantial text overlap with arXiv:2501.01694

- **What's New**: 이 연구에서는 항공 안전을 분석하기 위해 자연어 처리(Natural Language Processing, NLP)와 다양한 딥러닝 모델을 활용했습니다. 특히, LSTM, CNN, 양방향 LSTM(Bidirectional LSTM, BLSTM), 간단한 순환 신경망(simple Recurrent Neural Network, sRNN)을 이용하여 안전 보고서의 비행 단계를 분류합니다.

- **Technical Details**: 모델은 높은 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수를 기록했습니다. 특히 LSTM 모델은 각각 87%, 88%, 87%, 88%의 성능을 발휘하여 가장 높은 성과를 보여주었습니다. 이 연구는 안전 사고 분석의 자동화를 위한 효과성을 강조합니다.

- **Performance Highlights**: 자연어 처리와 딥러닝 기술의 통합은 항공 안전 분석을 혁신적으로 강화할 수 있는 가능성을 제시합니다. 이를 통해 타겟화된 안전 조치를 마련하고 보고서 처리 과정을 간소화할 수 있습니다.



### Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision (https://arxiv.org/abs/2501.07886)
Comments:
          22 pages, 10 figures

- **What's New**: 이번 연구는 신뢰할 수 없는 인간의 감독 하에서도 언어 모델( LM)의 사후 훈련(post-training)이 효과적으로 수행될 수 있는지를 검토합니다. 실험에서는 소규모 모델과 시간 제약이 있는 인간을 이용해 신뢰할 수 없는 시연과 비교 피드백을 시뮬레이션했습니다.

- **Technical Details**: 연구의 핵심은 반복 레이블 정제(iterative label refinement, ILR)입니다. ILR은 비교 피드백을 활용해 인간의 시연을 대체할 모델 생성 대안을 결정한 후, 업데이트된 데이터로 SFT(supervised finetuning)를 통해 모델을 재훈련합니다. 이러한 접근법은 DPO(reinforcement learning from human feedback)보다 더 나은 성능을 발휘합니다.

- **Performance Highlights**: SFT+ILR 방법은 수학, 코딩 및 안전한 지시사항 준수와 같은 여러 작업에서 SFT+DPO보다 우수한 결과를 나타냈습니다. 연구 결과는 LM이 복잡한 작업을 수행하는 데 있어, RLHF가 항상 최선의 방법이 아닐 수 있음을 시사하며, 오히려 훈련 데이터를 개선하는 방향으로 피드백을 사용하는 것이 더 효과적일 수 있음을 보여줍니다.



### Social Media Data Mining With Natural Language Processing on Public Dream Contents (https://arxiv.org/abs/2501.07839)
Comments:
          16 pages, 6 figures

- **What's New**: 이 연구는 COVID-19 팬데믹이 전 세계적인 생활 방식을 어떻게 변화시켰는지를 분석하고 있습니다. 특히, Reddit r/Dreams 커뮤니티에서 공유된 꿈 내용을 통해 팬데믹의 정신적 건강에 미친 영향을 조사합니다. 기존에 접근하기 힘들었던 서브콘셔스(반조직적) 반응을 드러낼 수 있는 새로운 데이터 소스를 활용하고 있습니다.

- **Technical Details**: 연구에서는 통계적 방법을 사용하여 팬데믹 이전(pre-pandemic)과 이후(post-pandemic) 꿈의 긍정적, 부정적, 중립적 변화 추세를 평가하고 있습니다. 또한 LLaMA 3.1-8B 모델을 레이블링된 데이터로 파인튜닝하여 꿈의 내용을 정확하게 감정 분류(sentiment classification)할 수 있게 했습니다.

- **Performance Highlights**: 결과적으로, 연구는 꿈 내용을 통해 팬데믹의 심리적 영향을 드러내고 있으며, 이러한 꿈의 변화가 공공의 웰빙(well-being)의 지표로 작용할 수 있음을 제안합니다. 이 연구는 정신적 풍경에서의 깊은 변화와 꿈의 역할을 부각시키며, 전례 없는 시기에 개인의 심리적 상태를 이해하는 데 기여하고 있습니다.



### Agent-Centric Projection of Prompting Techniques and Implications for Synthetic Training Data for Large Language Models (https://arxiv.org/abs/2501.07815)
Comments:
          8 pages, 5 figures. Accepted at ICAART 2025. Derived from an early draft at 2312.17601. arXiv admin note: substantial text overlap with arXiv:2312.17601

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 prompting 기법과 다중 에이전트 시스템에 대한 새로운 관점을 제시합니다. 기존에는 이러한 기법을 비교하고 특성을 파악할 수 있는 체계가 부족했으나, 이를 통해 선형(context)과 비선형(context) 개념을 도입하였습니다. 이 접근법은 단일 LLM prompting과 다중 에이전트 방법 간의 구조적 동등성을 발견하는 데 도움을 줄 것으로 기대됩니다.

- **Technical Details**: 이 논문에서는 대형 언어 모델의 prompting 기법을 agent-centric 관점에서 분석합니다. 선형 맥락은 단일 연속적인 상호작용을 의미하고, 비선형 맥락은 가지치기 또는 다경로의 상호작용을 의미합니다. 이러한 개념은 prompting 전략과 다중 에이전트 시스템 간의 깊은 연결을 탐구할 수 있는 프레임워크를 제공합니다.

- **Performance Highlights**: 저자들은 비선형 prompting 기법의 결과가 동등한 다중 에이전트 시스템에서의 결과를 예측할 수 있다고 주장합니다. 또한, 단일 LLM prompting을 통해 다중 에이전트 시스템 아키텍처를 재현할 수 있으며, 이러한 동등성은 합성 훈련 데이터 생성의 새로운 접근 방식을 제안합니다. 이로 인해 prompting 기술과 다중 에이전트 도메인 간의 연구 결과의 교차 수확이 가능해질 것입니다.



### Talk to Right Specialists: Routing and Planning in Multi-agent System for Question Answering (https://arxiv.org/abs/2501.07813)
Comments:
          Work In Progress

- **What's New**: 본 논문은 RopMura라는 혁신적인 다중 에이전트 시스템을 제안하여 단일 도메인 정보의 한계로 인해 발생하는 허위 응답 및 부정확한 답변 문제를 해결합니다. RopMura는 지식 경계를 기반으로 적절한 에이전트를 선택하는 라우터와 복잡한 다중 홉 질문을 분해하는 플래너라는 두 가지 주요 구성 요소로 구성되어 있습니다. 이러한 접근 방식은 정보의 효과적인 라우팅과 계획을 가능하게 하여 다양한 도메인에서의 응답을 효율적으로 조정할 수 있게 합니다.

- **Technical Details**: RopMura는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 외부 지식의 통합과 응답의 신뢰성을 높일 수 있는 시스템입니다. 이 시스템은 각 RAG 에이전트의 지식 조각을 군집화하고 평균 임베딩 벡터를 계산하여 에이전트 간의 관련성 점수를 산출합니다. 이를 통해 복잡한 질문이 제기될 때 플래너는 각 단계마다 적절한 질문을 생성하고, 라우터는 가장 관련성 높은 에이전트를 선택하여 응답을 생성합니다.

- **Performance Highlights**: 실험 결과, RopMura는 단일 홉 질문과 다중 홉 질문 모두를 효과적으로 처리할 수 있음을 보여주었습니다. 단일 홉 질문의 경우, 라우팅 메커니즘만으로도 정확한 응답을 가능하게 하였고, 다중 홉 질문에서는 라우팅과 계획 메커니즘의 결합을 통해 일관성과 정확성을 갖춘 다단계 해결책을 제공하였습니다. 이를 통해 RopMura는 다양한 도메인 간의 질문-응답 작업에서 우수한 성능을 인정받았습니다.



### Parameter-Inverted Image Pyramid Networks for Visual Perception and Multimodal Understanding (https://arxiv.org/abs/2501.07783)
- **What's New**: 이번 논문에서는 파라미터를 반전시킨 이미지 피라미드 네트워크(Parameter-Inverted Image Pyramid Networks, PIIP)를 제안합니다. 기존의 이미지 피라미드는 동일한 대규모 모델을 여러 해상도의 이미지를 처리하는 데 사용하여 상당한 계산 비용을 초래했습니다. 반면, PIIP는 사전 훈련된 모델(예: ViTs 또는 CNNs)을 가지각색으로 활용하여 각기 다른 해상도의 이미지를 처리함으로써 계산 비용과 성능의 균형을 이룹니다.

- **Technical Details**: PIIP는 서로 다른 해상도의 특징을 효과적으로 융합할 수 있는 교차 브랜치(feature interaction) 메커니즘을 도입합니다. 이는 높은 해상도 이미지는 작은 네트워크의 가지로 처리하고, 낮은 해상도 이미지는 더 큰 네트워크 가지에서 처리하여 연산비용을 줄이고 성능을 향상시킵니다. 이 네트워크 구조는 기존의 비전 기초 모델들에 기반하여 각 해상도에서 직접 특징을 추출하는 방식을 채택합니다.

- **Performance Highlights**: PIIP는 객체 탐지, 세그멘테이션, 이미지 분류 및 다중 모달 이해 등 다양한 작업에서 단일브랜치 모델 및 기존의 다중 해상도 접근 방식보다 우수한 성능을 기록하면서도 계산 비용을 낮추었습니다. InternViT-6B와 같은 대규모 비전 모델에 적용시, PIIP는 탐지 및 세그멘테이션 성능을 각각 1%-2% 향상시키면서 원래의 컴퓨테이션의 40%-60%의 비용으로 실행됩니다. 또한, PIIP-LLaVA 모델은 극소량의 학습 데이터로 TextVQA에서 73.0% 정확도와 MMBench에서 74.5%를 달성했습니다.



### A Heterogeneous Multimodal Graph Learning Framework for Recognizing User Emotions in Social Networks (https://arxiv.org/abs/2501.07746)
- **What's New**: 이 논문은 소셜 네트워크 내에서 개인화된 감정 예측을 위한 Heterogeneous Multimodal Graph Learning Framework인 HMG-Emo를 제안합니다. 이는 다양한 모달리티의 데이터를 결합하여 사용자 감정을 인식하는 데 깊은 학습 기반의 기능을 활용합니다. 특히, 사용자 간의 다양한 관계와 콘텐츠 맥락을 동적으로 통합할 수 있는 모듈을 포함하여 감정 예측의 정확성을 높입니다.

- **Technical Details**: HMG-Emo는 사용자-미디어 그래프에서 엣지 분류 작업으로 개인화된 감정 예측 문제를 정의하는 혁신적인 방식으로, 그래프 학습 방법을 적용할 수 있습니다. 이 프레임워크는 Graph Attention Network를 활용하여 여러 모달리티에서 정보를 동시에 축적하고, 감정 분류 작업에 적응할 수 있는 동적 맥락 결합 모듈을 포함합니다. 다양한 실험을 통해 기존의 수작업으로 만들어진 특성을 사용한 방법보다 우수함을 보였습니다.

- **Performance Highlights**: HMG-Emo는 소셜 미디어에서의 개인화된 감정 예측을 위한 최초의 깊은 그래프 학습 기반 프레임워크로 자리 잡습니다. 이 프레임워크는 다양한 맥락 정보를 통합하여 감정 분류에서 기존 방법들보다 높은 성능을 보여줍니다. 또한 다중 요인을 사용한 감정 예측의 중요성을 실험적으로 입증하며, 제안된 방법의 다양한 구성 요소의 견고성을 평가했습니다.



### A Survey of Early Exit Deep Neural Networks in NLP (https://arxiv.org/abs/2501.07670)
- **What's New**: 이 논문은 Early Exit (EE) 방법들에 대한 포괄적인 서베이를 제공하며, 이 방법들이 NLP에서 어떻게 활용되고 있는지를 다룹니다. DNN의 초기 계층에서 간단한 샘플을 분류하는 EE 전략은 전반적으로 추론 과정을 가속화시킬 수 있습니다. 또한, EE 방법은 다양한 샘플의 복잡도에 맞게 적응적 추론을 가능하게 하여, 성능 저하 없이 자원을 효율적으로 사용할 수 있도록 돕습니다.

- **Technical Details**: EE 방법은 DNN의 여러 층에 분류기를 추가하여, 예측에 충분한 신뢰에 도달했을 때 추론 과정을 중단할 수 있게 해줍니다. 이는 DNN이 너무 깊은 층으로 샘플을 보내는 과도한 처리를 방지하고, 보다 복잡한 샘플만이 더 깊이 처리되도록 함으로써 모델의 효율성을 높입니다. EE 방법은 다양한 작업, 특히 NLP와 비전-언어 모델링에 광범위하게 적용될 수 있으며, 각 샘플의 복잡성에 기반해 계산 자원을 동적으로 조정합니다.

- **Performance Highlights**: EE 모델은 정량적 성능 이점 이외에도 해석 가능성, 강건함, 그리고 분산 추론 능력을 증가시킵니다. 다양한 샘플에 대한 신뢰도를 평가하여, 간단한 샘플에 대해서는 초기 계층에서 빠르게 결과를 제공함으로써, 추론 속도를 개선합니다. 이러한 특성 덕분에 EE 방법은 자원이 제한된 환경에서 DNN을 효과적으로 배포할 수 있는 강력한 도구로 자리 잡고 있습니다.



### Optimize Incompatible Parameters through Compatibility-aware Knowledge Integration (https://arxiv.org/abs/2501.07596)
Comments:
          Published on AAAI'25: The Annual AAAI Conference on Artificial Intelligence

- **What's New**: 이번 연구에서는 다양한 데이터 분포에서 발생하는 비호환 파라미터를 효과적으로 최적화하기 위한 새로운 방법론인 Compatibility-aware Knowledge Integration (CKI)를 제안합니다. CKI는 두 가지 주요 요소로 구성되어 있으며, Parameter Compatibility Assessment와 Parameter Splicing을 통해 여러 모델의 지식을 통합합니다. 이러한 접근 방식은 기존의 단순한 파라미터 제거나 출력 앙상블 방법의 한계를 극복하고, 더 나은 모델 성능을 유지하면서도 추가적인 파라미터 없이 동작할 수 있습니다.

- **Technical Details**: CKI의 첫 번째 단계인 Parameter Compatibility Assessment는 개별 파라미터의 불확실성을 평가하고, 전체 모델의 정보 콘텐츠를 분석하여 종합적인 파라미터 호환성을 결정합니다. 두 번째 단계인 Parameter Splicing에서는 분석된 호환성을 기반으로 여러 모델의 파라미터를 결합하여 최적화된 모델을 생성합니다. 이 과정에서 하드 스플라이싱과 소프트 스플라이싱을 사용하여 최적의 파라미터를 선택하거나 가중치를 계산하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 다양한 추천 및 자연어 처리 데이터셋에서 실험한 결과, Compatibility-aware Knowledge Integration 방법이 기존 모델의 훈련 제한을 극복하면서도 인퍼런스 비용을 증가시키지 않고 효과적으로 비호환 파라미터를 최적화함을 보여주었습니다. 통합 모델은 추가적인 재학습 없이도 직접 사용 가능하며, 단 한 번의 에포크 재학습만으로도 성능 향상을 이끌어낼 수 있음이 입증되었습니다.



### WebWalker: Benchmarking LLMs in Web Traversa (https://arxiv.org/abs/2501.07572)
- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG) 기술의 성능을 한층 향상시키기 위해 WebWalkerQA라는 새로운 벤치마크를 도입합니다. 기존의 전통적인 검색 엔진이 제공하는 얕은 콘텐츠의 한계를 극복하고, 웹 서브페이지를 탐색해 고품질 데이터를 체계적으로 추출하는 LLM의 능력을 평가하는 것을 목표로 합니다.

- **Technical Details**: WebWalker는 인간의 웹 탐색을 모방하는 다중 에이전트 프레임워크로, explore-critic 패러다임을 활용하여 고급 탐색 기능을 구현합니다. 이 프레임워크는 LLM이 웹사이트 내의 서브페이지를 어떻게 효율적으로 탐색하고 정보를 추출하는지를 평가하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, WebWalkerQA는 도전적이며 실제 시나리오에서 RAG와 WebWalker의 통합이 매우 효과적임을 입증합니다. 수평적 및 수직적 통합을 통해 다양한 과제에서 더욱 풍부하고 복잡한 정보 처리가 가능해졌습니다.



### Imagine while Reasoning in Space: Multimodal Visualization-of-Though (https://arxiv.org/abs/2501.07542)
Comments:
          11 pages, 6 figures, 4 tables (27 pages, 10 figures, 16 tables including references and appendices)

- **What's New**: 이번 연구에서 제안한 Multimodal Visualization-of-Thought (MVoT) 방법론은 이미지와 언어를 결합하여 복잡한 사고 과정을 시각적으로 표현할 수 있게 해줍니다. MVoT는 MLLM (Multimodal Large Language Model)이 자연스럽게 시각적 사고를 생성하면서 언어적 사고를 동시에 진행하는 새로운 패러다임을 세웁니다. 이는 기존의 Chain-of-Thought (CoT) 접근 방식이 복잡한 공간적 추론에서 한계가 있었던 점을 극복하는 데 기여합니다. 최종적으로 MVoT는 언어적 추론을 보완할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MVoT는 두 가지 모드 (textual, visual)를 결합하여 다중 모드 생각을 생성하는 프로세스를 정의합니다. 주어진 입력 시퀀스에 대해 모델은 중간 단계에서 이미지 시각화를 추가하여 언어적 사고를 보완합니다. 이는 언어 시퀀스와 이미지 시퀀스를 병행 생성하며, 각 단계는 이전의 단계와 결합하여 더욱 풍부한 데이터를 생성합니다. 실험에서는 Chameleon-7B 모델을 통해 MVoT의 효과를 검증하며, 토큰 불일치 손실(token discrepancy loss)을 도입하여 고품질의 시각화를 보장합니다.

- **Performance Highlights**: MVoT는 여러 동적인 공간 추론 작업에서 경쟁력 있는 성능을 입증하였습니다. Maze, MiniBehavior, FrozenLake와 같은 벤치마크에서 전통적인 CoT 방식을 20% 이상 초과하는 성능을 보였습니다. 특히, MVoT는 CoT가 실패하는 가장 도전적인 시나리오에서도 부분적으로 안정적이며 신뢰할 수 있는 개선을 보여줍니다. 이번 연구는 복잡한 과제에서 시각적 사고의 효용성을 입증하는 중요한 기초 연구로 평가받고 있습니다.



### Investigating Large Language Models in Inferring Personality Traits from User Conversations (https://arxiv.org/abs/2501.07532)
Comments:
          13 pages, 5 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)인 GPT-4o와 GPT-4o mini가 사용자의 대화로부터 빅파이브 성격 특성을 추론할 수 있는지를 평가합니다. 특히, BFI-10(Big Five Inventory-10) 항목 점수를 산출하기 위한 중간 단계를 도입함으로써 정확도를 향상시키고, 이를 통해 더욱 신뢰성 있는 결과를 도출할 수 있다는 점을 강조합니다.

- **Technical Details**: 연구에서는 제로샷 프롬프트(Zero-shot prompting) 조건에서 대화량을 기반으로 BFI-10 항목 점수를 산출하였습니다. 성격 특성을 직접 추론하기 전에 BFI-10 항목 점수 도출을 요구하는 방식이 더욱 정확하다는 결과가 나타났습니다. 또한, 우울증 증상이 있는 그룹과 없는 그룹 간의 모델 성능 차이가 관찰되었습니다.

- **Performance Highlights**: GPT-4o mini는 우울증 관련 성격 변화에 대해 더욱 민감하게 반응하며, 특히 신경증(Neuroticism)과 성실성(Conscientiousness)의 변화를 잘 분석했습니다. 반면, GPT-4o는 다양한 그룹 간의 미세한 해석에서 강점을 보여주었습니다. 이러한 결과는 LLMs가 실제 심리 데이터를 효과적으로 분석할 수 있는 잠재력을 가지고 있음을 나타내며, 인공지능과 심리학 간의 학제간 연구를 위한 중요한 기초를 제공합니다.



### TiEBe: A Benchmark for Assessing the Current Knowledge of Large Language Models (https://arxiv.org/abs/2501.07482)
- **What's New**: 이 논문은 Timely Events Benchmark (TiEBe)를 소개하여 대형 언어 모델(LLMs)의 최신 사건에 대한 지속적인 업데이트 필요성을 강조합니다. TiEBe는 11,000개 이상의 질문-답변 쌍을 포함하고 있으며, 다양한 지역의 지식 격차와 진화하는 글로벌 이슈에 대한 LLM의 이해를 평가하는 데 중점을 두고 있습니다. 기존의 벤치마크가 일반적인 사실 회상에 중점을 두었다면, TiEBe는 지속적인 학습(continuous learning) 전략을 평가할 수 있는 강력한 도구로 작용합니다.

- **Technical Details**: TiEBe는 Wikipedia의 구조화된 회고 데이터를 활용하여 전 세계 및 지역적 사건에 대한 질문-답변 쌍을 생성합니다. 이 데이터는 2015년부터 2024년까지의 다양한 나라 및 글로벌 사건들을 포함하며, 이를 통해 LLM의 지식 업데이트 및 지역적 이해력을 평가할 수 있습니다. 질문-답변 쌍은 영어로 생성되며, 여러 LLM에 대해 동일한 평가 기준을 적용하여 지역적 편향을 탐색합니다.

- **Performance Highlights**: 시험된 모든 LLM들은 사실 회상에서 상당한 지역적 차이를 보였으며, 이는 모델들이 특정 지역의 정보에 비해 잘 알려지지 않은 지역 정보를 잘 처리하지 못함을 나타냅니다. TiEBe의 결과는 글로벌 지식 표현의 균형을 개선할 필요성을 강조하고, 지속적인 학습 능력을 평가하는 데 있어서도 중요한 참고자료를 제공합니다. 이 연구는 특정 지역에 특화된 데이터로 훈련된 LLM의 성능 개선 가능성을 탐색하는 기회를 제공합니다.



### Enhancing Retrieval-Augmented Generation: A Study of Best Practices (https://arxiv.org/abs/2501.07391)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 다양한 구성 요소와 설계 방식을 다룹니다. RAG 시스템은 언어 모델의 성능을 향상시키기 위해 새로운 쿼리 확장 기법, 다양한 검색 전략, 그리고 대조적 맥락 학습(Contrastive In-Context Learning)을 통합합니다. 다양한 실험을 통해 RAG 시스템의 개선 요소를 체계적으로 분석하여 최적의 성능을 위한 데이터를 제공합니다.

- **Technical Details**: RAG 시스템은 Language Model (LM)과 함께 외부 지식 소스를 통합하여 최신 정보와 정확한 응답을 생성합니다. 연구에서는 LLM의 크기, 프롬프트 디자인, 문서 청크 크기와 같은 요소들이 응답 품질에 미치는 영향을 분석하였습니다. 또한, 쿼리 확장 기법과 다국어 지식 기반을 적용한 최근의 접근 방법이 포함되어 있습니다.

- **Performance Highlights**: 논문에서 제안한 방법들은 RAG 시스템의 응답 품질을 개선하는 데 효과적임을 보여주었습니다. 이를 통해 언어 모델의 컨텍스트 풍부함과 생성 및 검색의 효율성을 균형 있게 맞추는 방법론을 제시합니다. 이 연구 결과는 다양한 실제 시나리오에서 더욱 적응 가능하고 고성능의 RAG 프레임워크 개발에 기여할 것으로 기대됩니다.



### Emergent effects of scaling on the functional hierarchies within large language models (https://arxiv.org/abs/2501.07359)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 계층적 기능이 다양한 방식으로 표현될 수 있음을 재조명합니다. 기존의 주장은 초기 층에서 문법을 처리하고, 중간 층에서는 의미를 해석하며, 후반 층은 정보를 통합한다는 것입니다. 연구진은 간단한 텍스트를 LLM에 입력하고, 이에 따른 활성화를 추출하는 방법을 사용하였습니다. 이로 인해 LLM의 작동 방식을 좀 더 세밀하게 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구는 Llama-3.2-3b 모델을 사용하여 각 층이 특정 정보를 인코딩하고 있는지를 조사했습니다. 이 과정에서 Support Vector Machines(SVM)와 Ridge Regression 기법을 활용하여 텍스트 라벨을 예측하였습니다. 초기 층(2-7)에서는 항목 수준의 의미가 가장 강하게 나타났고, 그 이후 두 항목 관계(8-12)와 네 항목 유사성(10-15)이 차례로 나타났습니다. 그러나 깊은 층에서는 정보의 압축이 일어나면서도 의미 있는 추상이 결여된 경우가 발견되었습니다.

- **Performance Highlights**: 대형 모델(예: Llama-3.3-70b-Instruct)에서는 추상화 수준에서 극적인 변동이 관찰되었습니다. 층이 깊어질수록 두 항목 관계와 네 항목 유사성이 처음에는 증가하다가 급격히 감소하고, 이어 다시 순간적으로 증가하는 특이한 패턴이 여러 실험에서 일관되게 나타났습니다. 이 외에도 인접한 층 간의 주의(attention) 메커니즘 간 협조가 나타나면서 각 층이 어떤 정보를 전문으로 다루는지 변동을 보였습니다. 이러한 결과는 LLM의 복잡성과 동적인 작동 방식을 보여줍니다.



### FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering (https://arxiv.org/abs/2501.07314)
Comments:
          11 pages, 4 figures, 4 tables. To be published in NoDaLiDa/Baltic-HLT 2025 proceedings

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 훈련 데이터 품질을 향상시키기 위해 LLM 기반의 라인 레벨(line-level) 필터링 방법을 도입했습니다. 기존의 전통적인 휴리스틱 필터는 종종 저품질 텍스트를 놓치거나 소중한 내용을 잘못 제거하는 경향이 있습니다. 연구팀은 GPT-4o mini를 사용하여 FineWeb에서 20,000개의 문서 샘플을 라인 레벨로 레이블링하고, 저품질 라인에 대한 설명적 레이블을 생성하였습니다.

- **Technical Details**: 저품질 텍스트는 아홉 가지 주요 카테고리로 그룹화되며, 이를 바탕으로 DeBERTa-v3 분류기를 훈련하여 FineWeb의 10B 토큰 하위 집합으로 필터링을 확장합니다. 필터링의 영향을 시험하기 위해 원본 데이터셋과 필터링된 데이터셋에서 GPT-2 모델을 훈련했습니다. 이 과정에서 LLM 기반의 라인 레벨 필터링 방법이 데이터 품질과 훈련 효율성을 크게 개선함을 다루고 있습니다.

- **Performance Highlights**: 필터링된 데이터로 훈련된 모델이 HellaSwag 벤치마크에서 더 높은 정확성을 달성하고, 최대 25% 적은 데이터로도 성능 목표에 더욱 빠르게 도달하는 것을 확인했습니다. 이는 LLM 기반 필터링이 데이터 품질에 미치는 긍정적인 영향을 잘 보여줍니다. 연구팀은 이와 관련된 품질 라벨이 부착된 데이터셋, FinerWeb-10BT, 및 코드베이스를 공개하여 추후 연구를 지원하고자 합니다.



### The Lessons of Developing Process Reward Models in Mathematical Reasoning (https://arxiv.org/abs/2501.07301)
- **What's New**: 이 논문에서는 Process Reward Models (PRMs)의 개발에서 발생하는 여러 챌린지를 다룹니다. 기존의 Monte Carlo (MC) 기반 데이터 합성과 LLM-as-a-judge 활용 방법이 PRM의 성능 평가에 있어 단점이 있음을 보여주었습니다. 특히, 기존 BoN (Best-of-N) 평가 방식의 편향과 이를 통한 과정 검증의 비효율성을 지적하고 있습니다.

- **Technical Details**: 연구에서는 MC 추정이 현재 단계의 정확성을 평가하기 위해 완료 모델에 의존하며, 이로 인한 부정확한 단계 검증이 PRM 성능에 부정적인 영향을 미친다고 설명합니다. 또한, PRM이 던지는 응답에 대한 허용 기준이 과도하게 높아 BoN 점수를 비정상적으로 증가시키는 경향이 있다는 점도 짚었습니다. 여기서는 새로운 합의 필터링 메커니즘을 개발하여 MC 추정과 LLM-as-a-judge의 통합을 시도하였습니다.

- **Performance Highlights**: 제안된 방법론은 BoN 평가와 단계별 오류 식별 작업에서 모델 성능과 데이터 효율성을 크게 향상시켰습니다. 최종적으로, 이 논문은 기존의 오픈 소스 PRM보다 더 우수한 성능을 지닌 최신 PRM을 발표하였으며, 향후 과정 감독 모델 개발에 대한 실용적인 지침을 제공합니다.



### Comparative analysis of optical character recognition methods for S\'ami texts from the National Library of Norway (https://arxiv.org/abs/2501.07300)
Comments:
          To be published in Proceedings of the 25th Nordic Conference on Computational Linguistics (NoDaLiDa)

- **What's New**: 이번 연구는 노르웨이 국가 도서관(NLN)의 디지털화 과정에서 Sámi 언어 문서의 Optical Character Recognition (OCR) 정확성을 개선하기 위해 세 가지 OCR 접근법인 Transkribus, Tesseract, TrOCR을 미세 조정하고 평가하였습니다.

- **Technical Details**: 이 연구는 Transkribus, Tesseract, TrOCR을 활용하여 Sámi 문서를 디지털화하는 과정에서 발생하는 문제를 해결하고자 하였습니다. 기존의 사전 학습된 모델을 세밀하게 조정하고, 수작업 주석(manual annotations)과 기계 주석(machine annotations) 및 합성 텍스트 이미지(synthetic text images)를 보완하여 OCR 정확성을 증가시킬 수 있음을 발견했습니다.

- **Performance Highlights**: Transkribus와 TrOCR은 이 과제에서 Tesseract보다 뛰어난 성능을 보였으며, Tesseract는 아웃 오브 도메인(out-of-domain) 데이터셋에서 우수한 성과를 나타냈습니다. 또한, 수작업 주석 데이터가 적더라도 OCR의 정확성을 확보할 수 있는 가능성을 제시했습니다.



### When lies are mostly truthful: automated verbal deception detection for embedded lies (https://arxiv.org/abs/2501.07217)
- **What's New**: 본 연구는 구술 기만 탐지에서 잠재적으로 포함된 거짓 정보의 존재를 인정하고, 신뢰할 수 있는 진술과 기만적인 진술 간의 연속성을 중심으로 진행되었습니다. 기존의 연구들이 이런 포함된 거짓에 대해 간과하고 있었던 점에서 새로운 시각을 제공합니다.

- **Technical Details**: 새로운 데이터셋을 통해 2,088개의 신뢰할 수 있는 진술과 기만적인 진술을 수집하였으며, 각 진술의 포함된 거짓을 주석 처리했습니다. 참가자들은 자전적 사건에 대한 진실한 이야기를 작성하고 이어서 이를 기만적으로 수정하여 포함된 거짓을 강조하였습니다. 데이터 분석을 통해 Llama-3-8B 언어 모델이 64% 정확도로 진술을 분류할 수 있음을 보여주었습니다.

- **Performance Highlights**: 기만적인 진술의 3분의 2는 신뢰할 수 있는 정보에서 파생되어 있으며, 포함된 거짓과 신뢰할 수 있는 정보를 구별하기 어려운 점이 발견되었습니다. 이 연구는 구술 기만 탐지에 대한 포함된 거짓에 대한 연구를 촉진하는 중요한 자원으로 작용할 것입니다.



### ListConRanker: A Contrastive Text Reranker with Listwise Encoding (https://arxiv.org/abs/2501.07111)
Comments:
          11 pages, 4 figures

- **What's New**: 본 연구에서는 Listwise-encoded Contrastive text reRanker(ListConRanker)라는 새로운 reranker 모델을 제안합니다. 이 모델은 기존의 pointwise 및 pairwise encoding의 한계를 극복하며, passage 간의 비교 단계를 아우르는 listwise encoding 방식을 채택하고 있습니다. 또한, Circle Loss를 손실 함수로 사용함으로써 학습 효율성을 높이고 gradient의 변화가 부드럽게 이루어지도록 합니다.

- **Technical Details**: ListConRanker는 BERT 기반의 모델이며, query와 passage를 embedding 모델에 입력하여 원래의 feature를 추출한 뒤, 이를 통합하여 ListTransformer에 전달합니다. ListTransformer는 passage feature 간의 글로벌 contrastive 정보 학습을 용이하게 하여, 유사한 passage와 비유사한 passage 간의 군집화를 수행합니다. 제안된 ListAttention은 query의 feature를 유지하면서도 글로벌 비교 정보를 학습하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과 ListConRanker는 Chinese Massive Text Embedding Benchmark에서 최첨단 성능을 달성하였습니다. 특히 cMedQA1.0, cMedQA2.0, MMarcoReranking 및 T2Reranking 데이터셋에서도 높은 성능을 입증하였습니다. 또한, ablation study를 통해 ListTransformer와 Circle Loss의 효과 역시 검증되었습니다.



### AdaCS: Adaptive Normalization for Enhanced Code-Switching ASR (https://arxiv.org/abs/2501.07102)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 연구에서 제안하는 AdaCS(Ada Code Switching) 모델은 자동 음성 인식(ASR) 시스템에서 intra-sentential code-switching (CS) 문제를 다루기 위해 개발되었습니다. 기존의 ASR 시스템은 주로 단일 언어 데이터로 훈련되어 CS 해석에서 약점을 보였으나, AdaCS는 적응형 바이어스 주의(attention) 모듈(BAM)을 통합하여 이러한 문제를 해결합니다. 이 접근 방식은 미지의 도메인에서도 robust한 성능을 제공하며, 양질의 CS 음성 인식 통합에 기여하고 있습니다.

- **Technical Details**: AdaCS 모델은 인코더와 디코더 블록에 바이어스 주의 모듈을 통합하여 CS 구문 인식 및 정규화를 용이하게 합니다. 바이어스 주의 모듈(BAM)은 사전 정의된 바이어스 리스트를 입력으로 받아들여, 이를 기반으로 CS 구문에 대한 정보를 강화합니다. BAM은 바이어스 인코더, 순위 선택, 주의 서브모듈로 구성되어 있으며, 입력된 바이어스 리스트로부터 토큰 수준 표현 행렬 및 풀링된 벡터 표현을 생성하여 효과적으로 CS 구문을 처리합니다.

- **Performance Highlights**: AdaCS는 제안된 두 가지 테스트 세트에서 베트남어 CS ASR 정규화에서 기존 방법보다 56.2% 및 36.8%의 WER(Word Error Rate) 감소를 달성했습니다. 이는 미지의 CS 구문을 다양한 도메인에서 처리할 수 있는 인상적인 성능을 입증합니다. 실험 결과 및 데이터셋은 AdaCS 모델이 ASR의 낮은 자원 언어 환경에서도 효과적인 성능을 발휘함을 확인해주고 있으며, 향후 연구에 중요한 기준을 제공할 것입니다.



### Boosting Text-To-Image Generation via Multilingual Prompting in Large Multimodal Models (https://arxiv.org/abs/2501.07086)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이번 연구에서는 텍스트-이미지 생성(T2I)을 위한 멀티모달 모델(MMMs)의 다국어 처리 능력을 활용하는 새로운 방법, PMT2I(Parallel Multilingual prompting for Text-to-Image tasks)를 제안합니다. 기존 연구는 주로 입력 텍스트의 공간을 강화하는 데 초점을 맞추었지만, PMT2I는 원본 텍스트를 여러 언어로 번역하고 이를 모델에 제공하여 다국적 프롬프트를 생성합니다. 이 방식은 이미지 생성의 정확도와 다양성을 증가시키는 데 기여하며, 여러 언어를 활용하여 영문 입력의 성능을 향상시킵니다.

- **Technical Details**: PMT2I는 두 가지 단계로 구성된 프로세스를 포함합니다: 첫째, 원본 입력 텍스트를 여러 언어로 번역하여 프롬프트를 구성하고, 둘째, 이러한 프롬프트를 활용하여 이미지를 생성하는 과정입니다. LMMs는 텍스트 입력을 잠재 공간(latent space)으로 변환하고, 생성된 이미지는 디퓨전 모델(DDPM) 또는 VQ-GAN을 사용하여 디코딩됩니다. PMT2I는 다양한 언어 요약을 통해 LMM의 이해력을 높이고, 다양한 프롬프트 생성을 가능하게 합니다.

- **Performance Highlights**: PMT2I는 두 개의 멀티모달 모델에서 실험을 통해 일반 T2I 합성 성능에서 우수한 결과를 보였습니다. 예를 들어, MS-COCO 30K 데이터셋에서 Emu2-Gen 모델의 CLIP-T와 CLIP-I 점수를 각각 0.6과 0.9 포인트 향상시켰습니다. 또한, PMT2I는 이미지 품질과 다양성에 있어 기존 방법을 초월하며, reranking 방법을 적용했을 때 성능이 더욱 향상되는 것으로 나타났습니다.



### ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization (https://arxiv.org/abs/2501.07020)
Comments:
          The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: ViSoLex는 베트남의 소셜 미디어 텍스트에 대한 용어 정규화를 위한 오픈 소스 시스템으로, Non-Standard Word (NSW) Lookup 및 Lexical Normalization의 두 가지 핵심 서비스를 제공합니다. 이를 통해 사용자는 비표준 언어의 표준 형태를 검색하고, NSW가 포함된 텍스트를 표준화 할 수 있습니다. Pre-trained language models와 약한 감독 학습 기술을 통합하여 데이터 부족 문제를 해결하고 정확성과 효율성을 확보하였습니다.

- **Technical Details**: ViSoLex는 NSW Lookup과 Lexical Normalization라는 두 가지 서비스로 구성되어 있습니다. NSW Lookup 서비스는 사용자가 입력한 NSW를 적절한 표준 형태로 해석하게 하며, Lexical Normalization 서비스는 NSW를 포함한 문장을 표준 형태로 변환합니다. 이 시스템은 기존의 NSW 사전을 활용하고, 필요 시 OpenAI GPT-4o API를 통해 동적으로 학습하여 NSW의 해법을 제시합니다.

- **Performance Highlights**: ViSoLex는 멀티태스크 학습을 통해 NSW 감지 및 정규화를 동시에 처리하며, 각 작업의 손실을 조화를 이루게 하여 성능을 개선합니다. Rule Attention Network를 통해 약한 감독 규칙을 적용하고, NSW의 다양성과 진화하는 패턴에 적응할 수 있는 능력을 갖추었습니다. 이를 통해 베트남 소셜 미디어 텍스트의 정규화 정확도를 높이며, 사용자에게 보다 효율적인 텍스트 처리를 제공합니다.



### Harnessing Large Language Models for Disaster Management: A Survey (https://arxiv.org/abs/2501.06932)
- **What's New**: 본 논문은 자연재해 관리에 사용되는 대형 언어 모델(LLMs)에 대한 종합적인 조사를 제공합니다. 기존의 연구가 부족했던 분야인 자연재해 관리에 대한 LLMs의 체계적인 리뷰를 통해 재난 관리의 네 가지 단계(완화, 준비, 대응, 회복)에 따른 LLMs의 응용을 정리하였습니다. 저자는 또한 다양한 작업과 응용 시나리오에 따라 LLMs를 분류하는 새로운 분류체계를 제안합니다.

- **Technical Details**: 재난 관리의 네 가지 단계는 사고 예방을 위한 위험 식별, 재난 발생에 대한 준비 상태의 확립, 즉각적인 대응 및 복구 과정을 포함합니다. 이는 사회적 및 기술적 도구의 동원 및 공급을 통해 이루어지며, LLM들은 다양한 모달리티의 데이터를 처리하여 실시간 데이터 분석을 지원합니다. 이 과정에서 encoder-based LLM, decoder-based LLM, 그리고 멀티모달 LLM이 LLM 아키텍처로 구분되어 연구되고 있습니다.

- **Performance Highlights**: LLMs는 재난 관리에서 여러 신뢰할 수 있는 수행 결과를 보여줍니다. 예를 들어, decoder 기반 모델은 커뮤니티의 취약성 관련 질문에 대한 답변 생성을 통해 효과적인 자원 할당을 지원합니다. 또한, 향상된 공공 인식을 위한 지식 추출 및 예측 기능을 통해 재난 대응 및 경고 생성에서 중요한 역할을 수행하고 있습니다.



### Language Fusion for Parameter-Efficient Cross-lingual Transfer (https://arxiv.org/abs/2501.06892)
Comments:
          20 pages

- **What's New**: 이번 논문에서 소개된 FLARE는 기계 번역의 정보를 통합하여 정확도를 높이고, 파라미터 효율성을 유지하는 새로운 방법론을 제안합니다. FLARE는 저차원 LoRA(adapters) 내에서 다양한 언어의 표현을 융합하여, 특히 비영어 언어의 표현 품질과 하위 작업 성능을 개선합니다. 또한, FLARE는 영어와 비영어 언어의 토큰을 혼합하여 기계 번역(MT)의 필요성을 줄이는 접근 방식도 포함되어 있습니다.

- **Technical Details**: FLARE는 경량화된 선형 변환을 활용하여, 저차원 어댑터에서 소스 언어와 타겟 언어의 표현을 통합합니다. 이 방법은 임계값에 따라 기계 번역으로 생성된 데이터와 영어 표현을 그것의 특징으로 사용하며, 이를 통해 XLT(cross-lingual transfer)를 수행합니다. 이러한 접근법은 추가적인 매개변수를 필요로 하지 않으면서도 성능을 크게 향상시킵니다.

- **Performance Highlights**: FLARE는 자연어 추론, 질문-답변 및 감정 분석과 같은 다양한 자연어 이해 작업에서 눈에 띄는 성능 개선을 보여줍니다. 질문-답변 과제에서 Llama 3.1 모델의 정확한 일치도에서 4.9%, Gemma 2에서는 2.2% 향상을 달성했습니다. 이러한 결과는 FLARE가 다양한 다국어 하위 작업에 효과적임을 나타내며, 특히 텍스트 생성 작업에서 유용함을 보여줍니다.



### A Comprehensive Evaluation of Large Language Models on Mental Illnesses in Arabic Contex (https://arxiv.org/abs/2501.06859)
- **What's New**: 이번 연구는 아랍 세계에서 정신 건강 문제에 대한 접근 가능한 진단 및 개입 도구의 필요성을 강조하며, 대규모 언어 모델(LLMs)을 활용하는 새로운 접근 방식을 제시합니다. 8개의 다양한 LLM을 평가하여 이들이 아랍어의 문화적 맥락에서 정신 건강 데이터셋에 어떻게 적용되는지를 조사했습니다.

- **Technical Details**: 연구에서는 AraDepSu, Dreaddit, MedMCQA와 같은 다양한 정신 건강 데이터셋을 사용하여 프롬프트 디자인, 언어 설정(native Arabic vs. translated English) 및 few-shot prompting이 진단 성능에 미치는 영향을 분석했습니다. 특히, 구조적 프롬프트가 덜 구조화된 변형보다 다중 클래스 데이터셋에서 평균 14.5% 더 높은 성과를 보였습니다.

- **Performance Highlights**: 모델 선택이 성능에 중대한 영향을 미쳤으며, Phi-3.5 MoE는 이진 분류에서 균형 잡힌 정확도가 뛰어난 반면, Mistral NeMo는 심각도 예측 작업에서 평균 절대 오차에서 우수한 성능을 나타냈습니다. 또한, few-shot prompting은 GPT-4o Mini에서 다중 클래스 분류 정확도를 평균 1.58 배 높이며 일관된 성능 향상을 보였습니다.



### Event Argument Extraction with Enriched Prompts (https://arxiv.org/abs/2501.06825)
- **What's New**: 이번 연구는 prompt 기반의 Event Argument Extraction (EAE) 모델에서 다양한 정보 유형을 프롬프트에 포함시키는 것이 성능에 미치는 영향을 심층적으로 탐구합니다. 특히 같은 문서 내의 다수의 이벤트와 역할 인자 간의 관계를 활용하여 모델의 최상의 성능을 확인합니다. 이를 통해 프롬프트 기반 EAE 모델의 최적화 가능성을 제시하고 있습니다.

- **Technical Details**: EAE 모델은 역할 인자와 이벤트 트리거 간의 상호작용을 반영하여 설계됩니다. 연구에서는 단일 역할, 다중 역할, 다중 이벤트 프롬프트 모델을 비교하면서 각 모델의 성능 상한선을 탐색합니다. 추가적으로, 손실 규제 기법을 제안하여 프롬프트 기반 모델의 성능을 향상시키는 방향을 모색합니다.

- **Performance Highlights**: 실험 결과, 현재의 프롬프트 기반 EAE 모델은 여전히 개선할 여지가 크며, 특히 역할 간의 상호작용을 활용하는 것이 성능 향상에 큰 영향을 미치는 것으로 나타났습니다. 연구에서는 BERT, BART, Roberta와 같은 다양한 언어 모델에서 기능을 시험하고 성능을 비교했습니다. 이와 함께, 여러 이벤트 간의 정보를 효과적으로 활용하는 것이 모델의 최적 성능을 발휘하는 데 중요한 요소라는 점을 강조합니다.



### Bridging the Fairness Gap: Enhancing Pre-trained Models with LLM-Generated Sentences (https://arxiv.org/abs/2501.06795)
- **What's New**: 이 논문에서는 Pre-trained Language Models(PLMs)가 내재하고 있는 성별 편향을 감소시키기 위해 새로운 접근 방식을 제안합니다. 기존의 디바이싱(debiasing) 방법들이 질, 다양성 또는 인구 통계적 균형이 결여된 외부 말뭉치에 의존하는 반면, 이를 해결하기 위해 Fair-Gender라는 방법을 사용합니다. 이는 의미적으로 풍부하고 속성 균형을 갖춘 문장을 활용하여 PLMs의 공정성을 향상시키는 방향으로 나아갑니다.

- **Technical Details**: 제안된 방법은 큰 언어 모델(LLM)로부터 생성되는 문장에서 Causal Analysis(인과 분석)를 통해 인과적 효과를 추정함으로써 불일치한 문장을 제외하고 정렬된 문장을 식별합니다. 이 과정에서 PLMs에 긍정적인 전이를 보장하여 언어 모델의 표현력을 유지하면서 성별 편향을 감소시킬 수 있습니다. 이 논문은 PLMs와 LLMs 간의 잠재 공간 차이를 극복하기 위한 알고리즘을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 Fair-Gender 방법이 PLMs의 성별 편향을 유의미하게 감소시키며, 언어 표현력을 보존하는 데 효과적임을 보여 줍니다. 이는 법률, 의료, 인사 등 다양한 분야에서 공정성을 확보하는 데 중요한 의미를 가집니다. PLMs가 가진 잠재적 편향을 자동으로 감소시키는 데 있어 새로운 가능성을 제시합니다.



### Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models (https://arxiv.org/abs/2501.06751)
- **What's New**: 최근 연구에서는 Text-to-Image(T2I) diffusion 모델에서 패딩 토큰(padding tokens)의 역할을 처음으로 심도 있게 분석하였습니다. 기존의 관행에 따르면 입력 프롬프트는 고정 길이로 패딩됩니다. 그러나 패딩 토큰이 이미지 생성 과정에 미치는 영향은 과거에 비활성화되었으며, 이 연구는 패딩 토큰이 T2I 모델의 여러 구성요소에 어떻게 작용하는지를 설명하는 두 가지 인과 기법을 제안합니다.

- **Technical Details**: 제안된 방법론은 Intervention in the Text Encoder Output(ITE)와 Intervention in the Diffusion Process(IDP)를 포함합니다. 두 방법 모두 특정 입력이나 중간 표현에 변화를 주어 출력에 미치는 영향을 관찰하는 인과 매개 분석(causal mediation analysis)에 기반하고 있습니다. ITE에서는 텍스트 인코더의 출력에서 특정 구간을 선택적으로 교란시켜 패딩 토큰과 원래 프롬프트 토큰의 기여도를 차별화하고, IDP에서는 확산 과정에서 패딩 토큰이 모델에 의해 어떻게 사용되는지를 분석합니다.

- **Performance Highlights**: 연구 결과, T2I 모델이 동결된 텍스트 인코더를 사용할 경우 패딩 토큰이 무시된다는 사실이 밝혀졌습니다. 그러나 텍스트 인코더가 훈련되거나 미세 조정된 경우, 패딩 토큰은 의미론적인 중요성을 가집니다. 또한, 특정 확산 모델 구조에서 패딩 토큰은 정보 저장 역할을 수행하며, 이는 언어 모델과 비전-언어 모델에서 발견된 바와 유사합니다.



### Hierarchical Divide-and-Conquer for Fine-Grained Alignment in LLM-Based Medical Evaluation (https://arxiv.org/abs/2501.06741)
- **What's New**: 이 논문은 의료 분야에서의 대규모 언어 모델(LLM)의 신뢰성과 정확성을 평가하기 위한 새로운 평가 프레임워크인 HDCEval을 제안합니다. 기존의 일반적인 평가 지표들이 실제 진단 과정의 복잡성을 반영하지 못하는 점을 개선하고자, 전문 의사들과의 협력을 통해 세부적인 의료 평가 지침을 개발했습니다. HDCEval은 복잡한 평가 작업을 전문화된 하위 작업으로 분해하여 전문가 모델을 통해 평가를 진행합니다.

- **Technical Details**: HDCEval 프레임워크는 두 가지 주요 구성 요소인 Divide와 Conquer로 이루어져 있습니다. Divide 단계에서는 복잡한 평가 작업을 여러 주요 작업으로 나누고, 각각의 주요 작업을 더욱 세부적으로 나눈 후, 각 작업에 대해 특화된 전문가 모델을 사용합니다. Conquer 단계에서는 Preference dataset을 활용하여 모델의 성능을 개선하고, Attribute-Driven Token Optimization (ADTO) 방법을 통해 각 모델이 구체적인 평가 기준에 부합하도록 훈련합니다.

- **Performance Highlights**: HDCEval은 다양한 의료 시나리오에서 기존의 기준 방법들에 비해 뛰어난 성능을 보였습니다. 특히 PandaLM 평가자와 비교했을 때, HDCEval은 인적 평가와의 일관성을 23.92% 향상시키는 결과를 보여주었습니다. 이러한 결과는 HDCEval이 의료 분야에서 전문가 수준의 평가와 잘 정렬되는 효과적인 프레임워크임을 입증합니다.



### Better Prompt Compression Without Multi-Layer Perceptrons (https://arxiv.org/abs/2501.06730)
Comments:
          7 pages, 0 figures

- **What's New**: 이번 논문에서는 Attention-Only Compressor (AOC)를 도입하여, 프롬프트를 더 효율적으로 압축하는 방법을 제안합니다. 이 방법은 기존의 인퍼런스 언어 모델 아키텍처의 다층 퍼셉트론(MLP) 계층을 제거하면서 학습 가능한 압축 인코더를 생성하여, 약 67% 적은 파라미터로도 유사한 성능을 유지할 수 있음을 보여줍니다. 특히 AOC는 최대 480배의 압축비에서도 기본적인 압축 인코더보다 더 나은 성능을 나타냅니다.

- **Technical Details**: AOC는 입력으로 주어진 프롬프트 토큰을 학습된 메모리 토큰과 결합하여 잠재 표현을 생성하는 구조를 가지고 있습니다. 이 모델은 기존 인퍼런스 프롬프트의 정보를 유지하면서 계산 리소스를 크게 절약합니다. 논문에서는 Llama 3.2 1B Instruct를 사전 학습된 모델로 사용하고, AdamW 옵티마이저를 통한 훈련 과정도 자세히 설명하고 있습니다.

- **Performance Highlights**: AOC는 기존의 압축 방법들과 비교했을 때 압축 성능이 우수할 뿐만 아니라, 인퍼런스 시간 동안 계산 필요량이 적은 장점을 가지고 있습니다. 실험 결과, AOC는 거의 세 배 더 많은 파라미터를 가진 기본 프롬프트 압축 인코더와 비슷한 성능을 발휘하며, 더욱 효율적인 압축기의 개발에 대한 가능성을 보여줍니다. 이러한 결과는 미래의 연구에서 압축 아키텍처의 다양성을 탐구하는 데 중요한 기초 자료가 될 것입니다.



### Measuring the Robustness of Reference-Free Dialogue Evaluation Systems (https://arxiv.org/abs/2501.06728)
- **What's New**: 이 논문은 다양한 창의적 응답을 평가하기 위한 신뢰할 수 있는 평가 지표의 부재를 해결하고자 합니다. 특히, adversarial attacks에 대한 내성을 평가하기 위한 벤치마크를 제공하며, DialogRPT, UniEval, PromptEval와 같은 지표를 분석하고 있습니다. 이 연구는 기존 연구에서 간과된 메트릭의 견고성을 평가하여 실제 대화 문제에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 grounded와 ungrounded 데이터셋을 사용하여 메트릭 성능을 평가하였습니다. 다양한 종류의 adversarial attacks에 대응하기 위해 응답 앞에 speaker tags를 추가하고, 정적 반응, 문법적 오류 응답, 회화 맥락 반복 등의 공격을 생성하였습니다. 이를 통해 메트릭이 불필요한 피드백을 과대 평가하는지 않도록 평가할 수 있는 기준을 제시합니다.

- **Performance Highlights**: 이 연구는 다양한 메트릭이 adversarial 응답에 대해 얼마나 취약한지를 보여줍니다. 독립적으로 평가된 두 데이터셋에서, 메트릭의 성능이 다르게 나타나며, ungrounded 데이터셋인 DailyDialog에서의 공격이 grounded 데이터셋인 TopicalChat에 비해 더 많은 어려움을 겪는 것으로 드러났습니다. 이러한 결과는 메트릭의 신뢰성을 보장하기 위해 개발된 미세한 평가 프레임워크의 필요성을 시사합니다.



### ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian (https://arxiv.org/abs/2501.06715)
Comments:
          7 pages, 5 figures. X International conference "Informatics. Culture. Technology." (2024)

- **What's New**: 이번 연구는 대규모 언어 모델의 추론 능력 평가를 위한 포괄적인 벤치마크인 ZNO-Eval을 소개합니다. 이 벤치마크는 우크라이나의 표준 교육 테스트 시스템에서 실제 시험 과제를 기반으로 하고 있습니다. 그동안 대부분의 연구가 영어에 집중된 반면, 이번 연구는 우크라이나어에 대한 평가의 중요성을 강조합니다.

- **Technical Details**: ZNO-Eval 벤치마크는 단일 응답 옵션, 객관식, 매칭, 개방형 질문 등 다양한 과목에서의 문제 유형을 포함하고 있습니다. 문제는 우크라이나어, 수학, 역사 및 지리 등의 과목으로 구성되어 있으며, 다양한 도메인과 복잡성을 통해 모델의 추론 능력을 심층적으로 분석할 수 있는 기회를 제공합니다. 연구의 대상은 GPT-3.5-Turbo, GPT-4o, GPT-4-Turbo, Mistral Large, Claude 3 Opus 및 Gemini-1.5 Pro와 같은 잘 알려진 언어 모델들입니다.

- **Performance Highlights**: GPT-4o는 일반 지식 추론과 복잡한 언어 작업에서 우수한 성능을 보였습니다. 반면, Gemini Pro와 GPT-4 Turbo는 산수 문제에서 두각을 나타내며 단일 응답과 개방형 수학 문제에서 높은 성과를 기록했습니다. 그러나 역사와 지리와 같은 텍스트 전용 일반 지식 과제에서는 모든 모델이 최대 성능에 근접했지만, 우크라이나어와 수학에서는 여전히 격차가 존재해, 이러한 언어와 맥락에 대한 보다 정확한 모델 능력 및 한계 평가를 위한 전문화된 벤치마크 개발의 필요성을 강조합니다.



### TAPO: Task-Referenced Adaptation for Prompt Optimization (https://arxiv.org/abs/2501.06689)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이번 논문에서는 Task-Referenced Adaptation for Prompt Optimization (TAPO)라는 새로운 접근 방식을 소개합니다. TAPO는 멀티태스크 성능을 향상시키기 위해 이론에 기반한 동적 메트릭 선택과 자동화된 프로프트 평가 및 생성 프로세스를 포함합니다. TAPO는 다양한 작업에 대한 적응성을 보장하고, 다수의 데이터셋에서 검증된 효과적인 자동 프로프트 최적화 프레임워크입니다.

- **Technical Details**: TAPO 프레임워크는 세 가지 주요 모듈로 구성됩니다. 첫 번째 모듈인 Dynamic Metric Selection은 LLM이 다양한 작업에 맞게 적절한 메트릭을 선택할 수 있도록 합니다. 두 번째 모듈인 Multi-Metrics Evaluation은 여러 관점에서 프로프트를 평가하며, 세 번째 모듈은 진화 기반 최적화 방식을 통해 자동으로 프로프트를 정제하여 다양한 작업 간의 적응성을 높입니다.

- **Performance Highlights**: TAPO는 다양한 데이터셋에서 실험을 통해 기존의 상태-최고 모델에 비해 일관된 성과를 보여줍니다. 예를 들어, 산술 추론 작업에서 TAPO는 GPT-3.5에서 88.15%의 유사도 점수를 기록하여 기존 방법들에 비해 유리한 성능을 나타냈습니다. 이러한 실험 결과는 TAPO가 동적으로 메트릭을 선택하고 가중치를 조정함으로써 멀티태스크 최적화의 효과를 극대화할 수 있음을 입증합니다.



### FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings (https://arxiv.org/abs/2501.06645)
- **What's New**: 이 논문은 FocalPO라는 새로운 손실 함수(loss function)를 소개합니다. 기존의 Direct Preference Optimization (DPO) 방법이 잘못된 선호 순위를 수정하는 데 효과적이지 않다는 최근 연구를 기반으로, FocalPO는 오히려 잘못된 순위 쌍의 중요성을 낮추고 정확히 순위를 매길 수 있는 쌍에 대한 모델의 이해를 향상시키는 데 중점을 둡니다. 이 접근법은 시각적 작업에서 사용되는 Focal Loss에서 영감을 받았습니다.

- **Technical Details**: FocalPO는 DPO 손실 함수에 동적으로 조절되는 계수를 추가하여, 모델이 올바로 순위를 매길 수 있는 쌍에 대해 더 높은 가중치를 부여합니다. 이 방식은 잘못된 페어(incorrect pairs)의 영향력을 줄이며, 모델이 올바른 보상 추정치에 따라 선호 쌍을 배워나가도록 촉진합니다. gradient 분석을 통해 FocalPO가 올바른 순서의 쌍에 더 높은 가중치를 할당함을 확인했습니다.

- **Performance Highlights**: 실험 결과, FocalPO는 Alpaca Eval 2.0과 같은 인기 있는 벤치마크에서 DPO 및 그 변형들보다 우수한 성능을 보였습니다. Mistral-Base-7B와 Llama-3-Instruct-8B 모델을 사용하여 진행한 평가에서도 FocalPO의 효과를 분명히 나타냈습니다. 또, FocalPO가 올바른 샘플 그룹과 잘못된 샘플 그룹에서 훈련에 미치는 영향을 실증적으로 보여주었습니다.



### Scaling Down Semantic Leakage: Investigating Associative Bias in Smaller Language Models (https://arxiv.org/abs/2501.06638)
- **What's New**: 이번 연구는 Gonen et al. (2024)에서 소개된 의미 누출(semantic leakage) 현상을 Qwen2.5 모델 패밀리를 통해 소규모 모델(500M~7B 파라미터들)에서도 조사합니다. 이전 연구가 대형 언어 모델에 집중했음에도 불구하고, 이 연구에서는 소규모 모델이 어떻게 더 적은 의미 누출을 보여주는지를 체계적으로 평가합니다. 새로운 색상 중심 데이터셋을 개발하여, 각 모델의 성능을 비교 및 분석합니다.

- **Technical Details**: 연구에는 Qwen2.5 모델군이 활용되었으며, 이는 500M에서 72B 파라미터의 언어 모델로, 일반 목적 및 지침(Instruction) 세분화가 필요합니다. 실험 동안에는 GPTQ-양자화 버전의 Qwen2.5-7B-Instruct가 사용되었고, 단일 P100 GPU에서 실행되었습니다. 평가 방법으로는 Gonen et al. (2024)에서 제안한 평균 누출 비율(Mean Leak-Rate)를 사용해, 각 모델이 얼마나 의미 누출을 보이는지를 측정합니다.

- **Performance Highlights**: 결과적으로, 소규모 모델은 전반적으로 적은 의미 누출을 보였으나, 중간 크기 모델이 때때로 더 큰 모델보다 누출 행동이 더 큼을 발견했습니다. 다양한 색상 관련 프롬프트를 통해 누출 패턴을 평가한 결과, 특정 타입의 프롬프트가 더 많은 의미 누출을 유도하는 경향이 있는 것으로 나타났습니다. 이 연구의 데이터셋과 생성된 모델 출력, 평가 코드는 공개되며, 향후 의미 누출 현상에 대한 보다 깊은 이해를 도울 것입니다.



### Dual use issues in the field of Natural Language Generation (https://arxiv.org/abs/2501.06636)
- **What's New**: 이 문서는 SIGGEN 커뮤니티에서 실시한 최근 설문의 결과를 문서화하고 있습니다. 설문은 자연어 생성(NLG) 분야에서의 이중 사용(Dual Use) 문제를 다루고 있으며, ACL의 요청에 의해 시행되었습니다. 적어도 본 문서는 향후 논의에 유용한 자료를 제공하겠다는 점에서 중요한 의미를 가집니다.

- **Technical Details**: 설문은 2024년 10월에 시행되었고, 2025년 1월에 결과가 처리되었습니다. 총 23명의 응답자가 참여했으나, 이는 SIGGEN 회원 전체를 대표하지 않을 수 있습니다. 설문은 NLG 기술이 초래할 수 있는 부정적인 영향을 평가하고, SIG의 역할과 관련 규제에 대한 인식을 강화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 설문 결과는 SIGGEN 커뮤니티에 공유되어, 연구자들이 직면하는 다양한 도전에 대한 명확한 이해를 제공할 것입니다. 응답은 익명으로 처리되며, 응답자들로 하여금 자발적으로 참여하도록 유도합니다. 결과는 ArXiv에 게시될 예정이며, 회원들이 관련 규제를 인식할 수 있도록 도울 계획입니다.



### ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning (https://arxiv.org/abs/2501.06590)
- **What's New**: 이번 논문에서는 ChemAgent라는 새로운 프레임워크를 소개하고 있습니다. 이 프레임워크는 대형 언어 모델(LLMs)이 화학 문제를 해결하는 데 필요한 성능을 향상시키기 위해 설계되었습니다. 특히 화학 작업을 하위 작업으로 분해하고 이들을 구조화된 컬렉션으로 컴파일하여 동적으로 업데이트되는 라이브러리를 통해 성능을 개선합니다.

- **Technical Details**: ChemAgent는 효과적인 작업 분해(Decomposition)와 솔루션 생성을 지원하기 위해 세 가지 유형의 메모리(memory)와 라이브러리 강화(reasoning component)를 설계했습니다. 이 방식을 통해 LLM은 경험에 따라 시간이 지나면서 개선될 수 있습니다. ChemAgent는 또한 문제 발생 시 메모리에서 관련 정보를 검색하고 정제하여 새로운 문제를 해결하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, SciBench의 네 가지 화학 추론 데이터셋에서 ChemAgent는 최대 46%의 성능 향상을 보여주었으며, 기존 방법들을 크게 초월했습니다. 이러한 결과는 약물 발견(drug discovery) 및 소재 과학(materials science)과 같은 미래의 다양한 응용 가능성을 시사합니다.



### ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Drafting (https://arxiv.org/abs/2501.06582)
- **What's New**: 이번 논문에서는 계약 조항 검색을 위한 최초의 벤치마크 데이터셋인 Atticus Clause Retrieval Dataset (ACORD)를 소개합니다. 계약 초안 작성을 위한 가장 중요한 참조 조항을 찾는 작업을 결합한 이 데이터셋은 전문가들에 의해 주석이 달려 있습니다. ACORD는 법률 계약의 복잡한 조항에 중점을 두어, 변호사들이 필요로 하는 정확한 언어 사용을 지원합니다.

- **Technical Details**: ACORD 데이터셋은 114개의 쿼리와 126,000개 이상의 쿼리-조항 쌍으로 구성되어 있으며, 각 쌍은 전문가에 의해 1에서 5점으로 평가됩니다. 계약 조항 검색의 도전 과제 중 하나는 계약서 내에서의 다층 구조와 다양한 관용 표현을 파악하는 것입니다. 이 데이터셋은 Retrieval-Augmented Generation (RAG) 기법을 통해 LLM의 계약 작성 능력을 강화하고, LLMs가 신뢰할 수 있는 계약 초고를 작성하는 데 도움이 됩니다.

- **Performance Highlights**: ACORD는 법률 작업의 복잡성을 효과적으로 관리하기 위해 LLM의 조항 검색 성능을 평가하는 데 사용될 수 있는 중요한 벤치마크로 자리잡고 있습니다. 비-인덱스 조정을 통해 높은 성능을 발휘하는 bi-encoder 리트리버와 포인트와이즈 LLM 재조정기가 긍정적인 결과를 보여줍니다. 법률 계약 작성 작업을 위한 교육 모델 개선에 대한 기여가 기대됩니다.



### Natural Language Processing and Deep Learning Models to Classify Phase of Flight in Aviation Safety Occurrences (https://arxiv.org/abs/2501.06564)
Comments:
          NLP, Aviation reports, Text analysis, Deep learning algorithms, Flight phase classification

- **What's New**: 이 연구에서는 항공사고의 전조 사건들을 설명하는 비구조화된 텍스트를 분석하기 위해, 자연어 처리(NLP)와 인공지능(AI) 모델을 적용했습니다. 이는 항공 산업 이해관계자들이 안전 사건을 분류하고 결정할 수 있는 기반을 제공하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구진은 NTSB(국가교통안전위원회)의 27,000개의 안전 사건 보고서를 활용하여 두 개의 딥러닝 모델, 즉 ResNet과 sRNN의 분류 성능을 평가했습니다. 이들 모델은 비행 안전 사건의 단계(classification of flight phases)를 정확하게 분류하는 데 사용되었습니다.

- **Performance Highlights**: 평가 결과, 두 모델 모두 68% 이상의 정확도를 기록하며, 7개 클래스 분류 문제에서 무작위 추측률인 14%를 크게 초과하는 성과를 보였습니다. 특히, sRNN 모델이 이 연구에 사용된 단순화된 ResNet 모델 아키텍처보다 훨씬 우수한 성능을 나타냈습니다.



### A Survey on Spoken Italian Datasets and Corpora (https://arxiv.org/abs/2501.06557)
Comments:
          submitted to IEEE Access Journal in Dec 2024

- **What's New**: 이 논문은 66개의 구술 이탈리아어 데이터셋을 포괄적으로 분석하여 이탈리아어 자원의 부족을 해소하려는 노력을 보여 줍니다. 기존 데이터셋의 특징, 방법론 및 응용 분야를 강조하며, 자동 음성 인식(ASR), 감정 탐지(emotion detection), 교육 등 다양한 분야의 응용 가능성을 제시합니다. GitHub와 Zenodo를 통해 공개된 데이터셋 목록은 연구자와 개발자에게 소중한 자원으로 작용할 것입니다.

- **Technical Details**: 본 논문에서는 구술 이탈리아어 데이터셋을 음성 유형, 출처 및 맥락, 인구 통계 및 언어적 특징에 따라 분류하여 데이터 수집, 주석(annotation), 검증 방법론을 심층적으로 탐구합니다. 데이터셋 품질과 가용성은 독립적으로 확인되지 않았지만, 제공된 정보를 종합하여 최선의 관행과 일반적인 문제점을 강조합니다. 아울러 데이터셋의 다양성과 적용성을 높이기 위해 협력 및 표준화를 통한 활용 방안을 추천합니다.

- **Performance Highlights**: 이탈리아어 구술 데이터셋의 확보는 고품질 ASR 시스템 및 언어 분석을 위한 필수 요소로 강조됩니다. 논문에서는 대화, 독백, 자발적 언어 표현 등 다양한 음성 유형을 포괄하는 66개의 데이터셋을 제시하고, 각 데이터셋이 제공하는 실질적인 응용을 설명합니다. 특히, 감정 분석 및 대화형 NLP 모델 개발에 유용한 데이터셋의 중요성이 강조되며, 미래 연구 방향을 제시합니다.



### Dispersion Measures as Predictors of Lexical Decision Time, Word Familiarity, and Lexical Complexity (https://arxiv.org/abs/2501.06536)
Comments:
          Pre-print, to be presented at the NLP Meeting 2025 (this http URL - NON-REVIEWED)

- **What's New**: 이 연구는 다양한 언어에서 단어의 분산(distribution)을 측정하기 위한 여러 척도를 평가하는 데 중점을 두고 있습니다. 특히 logarithm of range가 log-frequency보다 더 나은 예측 변수라는 점을 발견했습니다. 이 결과는 단순한 분산 측정이 복잡한 척도보다 더 유용할 수 있음을 시사합니다. 또한, 과거의 연구들과의 모순된 결과들에 대한 설명을 제공합니다.

- **Technical Details**: 이 논문에서는 corpus linguistics에서 단어의 빈도와 분산을 모두 고려하는 방법론을 모색합니다. 특히 word familiarity, lexical decision time, lexical complexity와 같은 심리언어학적 변수들을 예측하기 위해 다양한 dispersion measures를 평가합니다. 연구에 사용된 corpus는 YouTube 비디오에 기반한 TUBELEX로, 여러 언어를 아우르는 데이터셋을 포함합니다.

- **Performance Highlights**: 연구 결과, 로그 변환된 range가 모든 작업과 언어에서 log-frequency보다 더 높은 예측력을 보였으며, 이 발견은 일반적인 분산 측정 방식에 도전합니다. 또한, 분산 척도의 유형과 corpus의 부분적 세분화(part granularity)가 이들 변수의 예측에 미치는 영향에 대해 논의하여 중요한 통찰력을 제공합니다. 이 연구는 언어의 분산을 설명하는 새로운 이해를 제공하며, 이는 자연어 처리(NLP) 분야에도 적용될 수 있습니다.



### Fine-tuning Large Language Models for Improving Factuality in Legal Question Answering (https://arxiv.org/abs/2501.06521)
Comments:
          18 pages, 8 figures, to be published in COLING 2025

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 법률 질문 응답에서 발생하는 환각(hallucination) 문제를 해결하기 위한 새로운 벤치마크인 LegalHalBench를 소개합니다. 이 벤치마크는 LLM이 법적 질문에 답변할 때 발생할 수 있는 일반적인 환각을 평가하는 자동 측정 지표를 포함합니다. 또한 행동 복제(behavior cloning)와 새로운 Hard Sample-aware Iterative Direct Preference Optimization(HIPO) 방법을 통합하여 환각을 완화하는 방안을 제안합니다.

- **Technical Details**: 연구팀은 법률 질의 응답에서 발생하는 다섯 가지 환각 유형을 정의하고, 이를 평가하기 위한 지표를 설계했습니다. 두 단계로 이루어진 환각 완화 방법론에는 감독된 미세 조정(Supervised Fine-Tuning, SFT) 단계와 하드 샘플 인식적(iterative) 선호 최적화(DPO) 단계가 포함되어 있습니다. 이를 위해 새로운 대규모 법률 QA 데이터셋을 자동으로 구성하는 접근 방식을 제안하여 수작업 주석 비용을 줄이고 데이터셋의 확장성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 두 단계 미세 조정 전략이 법률 QA의 정확도를 크게 향상시켰음을 보여주었습니다. 이 방법은 비환각 법령 비율(non-hallucinated statute rate) 38.353%를 달성하여 기존의 전문 LLM들보다 월등한 성과를 보였습니다. 또한 법령 관련성 비율(statute relevance rate)과 법률 주장 진실성(legal claim truthfulness)에서 각각 37.13%와 6.56%의 개선을 이뤘습니다.



### PASS: Presentation Automation for Slide Generation and Speech (https://arxiv.org/abs/2501.06497)
- **What's New**: 이 논문은 PASS라는 새로운 파이프라인을 소개합니다. 이 시스템은 일반적인 Word 문서에서 슬라이드를 생성하고 생성된 슬라이드의 음성 전달을 자동화합니다. 이전 연구들이 주로 연구 논문을 변환하는 데 초점을 맞춘 반면, PASS는 보다 일반적인 문서에서 슬라이드를 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: PASS는 슬라이드 생성과 슬라이드 발표라는 두 가지 핵심 모듈로 구성됩니다. 슬라이드 생성 모듈은 문서의 내용을 바탕으로 구조화된 슬라이드를 자동으로 생성하며, LLM 및 멀티모달 (multimodal) 방식을 사용하여 텍스트와 이미지를 처리합니다. 슬라이드 발표 모듈은 각 슬라이드에 대한 스크립트를 생성하고 AI 음성 합성을 통해 이를 음성으로 변환합니다.

- **Performance Highlights**: 이 연구에서는 슬라이드의 관련성, 일관성 및 중복성을 평가하기 위한 LLM 기반 평가 메트릭을 개발했습니다. PASS는 이미지 매핑 모듈을 통해 적절한 이미지를 슬라이드에 자동으로 매핑하고, 발표 스크립트 생성 모듈을 통해 각 슬라이드에 대한 고품질 오디오를 생성합니다. 이 접근법은 발표의 콘텐츠 생성과 전달을 완전히 자동화하는 데 중점을 두고 있으며, 이는 이전의 연구들과는 차별화된 점입니다.



### Analyzing the Role of Context in Forecasting with Large Language Models (https://arxiv.org/abs/2501.06496)
- **What's New**: 이번 연구에서는 최근의 언어 모델(LLMs)이 이진 예측 질문에 대한 예측 성능을 평가합니다. 600개 이상의 이진 예측 질문으로 구성된 새로운 데이터셋을 소개하고, 다양한 수준의 문맥을 가진 입력 프롬프트가 예측 성능에 미치는 영향을 조사합니다. 뉴스 기사를 포함할 경우 성능이 크게 개선되지만, few-shot 예제 사용 시 정확도가 저하되는 결과를 보였습니다.

- **Technical Details**: 연구는 Metaculus 플랫폼에서 수집한 614개의 이진 예측 질문을 바탕으로 하며, 각 질문에 관련된 뉴스 기사를 요약하여 LLM의 예측 과정에 필요한 문맥을 제공합니다. 모델은 GPT-3.5-turbo, Alpaca-7B, Llama2-13B-chat의 세 가지로, 각각의 모델에 대해 다섯 가지 다른 입력 프롬프트를 사용하여 예측 성능을 평가했습니다. 특히, 관련된 뉴스 기사를 포함한 질문이 모델의 예측 성공에 긍정적인 영향을 미쳤습니다.

- **Performance Highlights**: 연구 결과, GPT-3.5-turbo 모델이 모든 LLM 중 최고의 성공률을 보였고, Llama2-13B-chat 모델이 Alpaca-7B보다 더 나은 성과를 내었습니다. 백그라운드 정보와 뉴스 기사를 포함한 프롬프트가 가장 높은 점수를 기록했습니다. 이는 LLM의 예측 성능이 제공된 정보의 양과 질에 크게 의존한다는 것을 시사합니다.



### Sequential Classification of Aviation Safety Occurrences with Natural Language Processing (https://arxiv.org/abs/2501.06490)
- **What's New**: 이 연구에서는 항공 안전 시스템에서의 안전 발생 사건을 분류하고 분류하기 위해 자연어 처리(NLP)와 인공지능(AI) 모델을 활용했습니다. 기존의 안전 관련 사건 보고서는 비구조적이고 알아보기 힘든 텍스트로 작성되어 있었는데, 이를 컴퓨터 시스템이 이해할 수 있도록 처리함으로써 안전 또한 개선될 수 있게 됩니다.

- **Technical Details**: 연구에서는 LSTM, BLSTM, GRU, sRNN과 이들 모델의 조합(예: LSTM+GRU, BLSTM+GRU 등)을 포함한 여러 딥러닝(deep learning) 모델의 분류 성능을 평가했습니다. 이 모델들은 27,000개의 NTSB(National Transportation Safety Board) 안전 사건 보고서를 기반으로 하여 그 성능이 87.9% 이상의 정확도를 기록했습니다. 이는 네 가지 클래스 분류 문제에서 우연히 맞힐 확률인 25%를 상회하는 수치입니다.

- **Performance Highlights**: 모델들은 80%, 88%, 85% 이상의 높은 정밀도(precision), 재현율(recall), F1 점수를 기록하며 경쟁력 있는 퍼포먼스를 보였습니다. 특히 sRNN 모델이 재현율(90%)과 정확도(90%)에서 가장 뛰어난 성능을 보였고, LSTM은 정밀도(87%)에서 약간 더 나쁜 성과를 기록했습니다.



### First Token Probability Guided RAG for Telecom Question Answering (https://arxiv.org/abs/2501.06468)
- **What's New**: 이 논문에서는 Multiple Choice Question Answering (MCQA) 문제를 해결하기 위한 새로운 Retrieval-Augmented Generation (RAG) 프레임워크를 제안합니다. 제안된 방법은 confidence score를 활용하여 하이퍼파라미터를 최적화하고, 동적으로 맥락을 조정하여 정확성을 높이는 것을 목표로 합니다. LLMs(대형 언어 모델)의 일반적인 과제인 사라진 정보 및 다중 선택 질문 문제를 보다 효율적으로 해결할 수 있는 기회를 제공합니다.

- **Technical Details**: 제안된 RAG 프레임워크는 첫 번째 토큰 확률(method) 기반으로 설계되어 사용자 질문에 적합한 정보를 검색합니다. 이 프레임워크는 key hyperparameters인 chunk number와 window size를 조정하여 모델이 선택한 답변의 확률을 정규화합니다. 또한, Phi-2라는 27억 개 매개변수를 가진 소형 언어 모델(SLM)을 사용하여 언어 이해 및 추론 능력을 효과적으로 발휘합니다.

- **Performance Highlights**: 실험 결과, 제안된 RAG 프레임워크는 MCQA 작업의 정확성을 향상시키는 가능성을 입증하였습니다. RAG의 하이퍼파라미터를 지속적으로 최적화함으로써 성능을 향상시키는 동시에 LLM의 응답 품질을 높일 수 있었습니다. 이러한 접근법은 특히 통신 분야와 같이 세부 도메인 지식이 중요한 분야에서 큰 효과를 발휘할 것으로 기대됩니다.



### Retrieval-Augmented Dialogue Knowledge Aggregation for Expressive Conversational Speech Synthesis (https://arxiv.org/abs/2501.06467)
Comments:
          Accepted by Information Fusion 2025

- **What's New**: 이번 논문에서는 CADK-CSS라는 새로운 Retrieval-Augmented Dialogue Knowledge Aggregation 모델을 제안하였습니다. 이 모델은 현재 대화 기록을 바탕으로 사용자와의 상호 작용에서 적절한 감정적 표현을 가진 음성을 합성하는 것을 목표로 합니다. 특히, 저장된 대화(SD)의 스타일 표현 지식을 활용하여 대화의 맥락을 보다 잘 이해할 수 있도록 돕습니다.

- **Technical Details**: RADKA-CSS는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 미리 구축된 저장된 대화 의미-스타일 데이터베이스(SDSSD)에서 현재 대화(CD)와 유사한 대화를 효과적으로 검색하는 방법을 설계합니다. 둘째, 다중 소스 스타일 지식 집계를 통해 CD와 SD에서 얻은 스타일 지식을 통합하여 음성 합성기에 전달합니다. 셋째, 다중 세분화 이질 그래프(Multi-granularity Heterogeneous Graph) 구조를 활용하여 대화의 의미와 스타일 정보를 효과적으로 인코딩합니다.

- **Performance Highlights**: 실험 결과에 따르면 RADKA-CSS는 대화 스타일에 대한 정합성과 표현력에서 모든 기준 모델을 초월하는 성과를 보였습니다. DailyTalk 데이터셋을 기반으로 한 객관적 및 주관적인 평가에서 뛰어난 결과를 확보하였고, 스타일의 일치성이 현저하게 향상되었습니다. 이를 통해 사용자 경험을 혁신적으로 개선할 가능성을 보여주었습니다.



### MedCT: A Clinical Terminology Graph for Generative AI Applications in Healthcar (https://arxiv.org/abs/2501.06465)
- **What's New**: 본 논문에서는 중국 의료 커뮤니티를 위한 세계 최초의 임상 용어 체계인 MedCT를 소개합니다. MedCT는 임상 기반 모델인 MedBERT와 개체 연결 모델인 MedLink와 함께 제공되며, 중국 임상 데이터의 표준화 및 프로그래머블한 표현을 가능하게 합니다. 또한, MedCT 지식 그래프는 대규모 언어 모델(LLMs)에서 발생하는 환각 문제를 최소화하는 기제를 제공합니다.

- **Technical Details**: MedCT은 SNOMED CT와 유사한 용어 체계를 구축하기 위해 LLM 기반의 전체론적 접근 방식을 사용하고, 초기 용어를 SNOMED CT에서 번역하여 형성하였습니다. MedBERT라는 임상 기초 모델을 구축하고, MedLink라는 개체 인식 및 연결 모델을 훈련하여 임상 용어 처리의 성능을 크게 향상시켰습니다. 이 시스템은 이를 통해 실시간 데이터 정제 및 반복적인 최적화를 통해 임상 환경에서 효과적으로 적용됩니다.

- **Performance Highlights**: MedCT 시스템은 의미적 매칭(semantic matching) 및 개체 연결(entity linking) 작업에서 최고의 성능(SOTA)을 달성하였으며, 중국어뿐만 아니라 영어에서도 높은 정확도를 기록했습니다. 실제 임상 환경에서의 실험을 통해 MedCT의 임상 워크플로우와 환자 결과에 대한 다수의 가치를 입증했으며, LLM 기반 응용 프로그램에서 안전성과 효율성을 강조합니다.



### O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning (https://arxiv.org/abs/2501.06458)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 추론 시간 확장을 통해 의료 추론 작업에서의 가능성을 탐구합니다. 특히, 진단 의사 결정에서 치료 계획까지의 복잡한 작업을 다룹니다. 실험 결과, 500개의 샘플로도 성능이 6%-11% 향상되었으며, 작업의 복잡성이 추론 체인의 길이에 직접적인 영향을 미친다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 메디컬 벤치마크(MedQA, Medbullets, JAMA Clinical Challenges)를 사용하여 LLM이 의료 문제의 다양한 복잡성에 적응하는 방식을 분석합니다. 언급된 주요 findings 내역은 대부분 투표를 통한 효율적 접근, 충분한 LLM의 용량이 필요함을 강조하며, 고난도 과제는 더 긴 추론 과정을 요구함을 증명합니다. 이 연구는 의료 분야에서 문제 해결을 위한 자연어 처리의 가능성을 보여줍니다.

- **Performance Highlights**: 연구 결과는 추론 시간 scaling의 유효성을 입증하며, 복잡한 임상 사례에서의 통찰력을 제공합니다. 추가적으로, 문제의 난이도가 증가함에 따라 더 긴 추론 시간이 요구되며, 자유롭고 체계적인 반응을 유도함으로써 의료 여정 학습의 잠재력을 열어줍니다. 최종적으로, 연구는 LLM의 추론 능력을 향상시키기 위해 새로운 방법론과 응용 프로그램을 영감을 불어넣는 것을 목표로 합니다.



### Synthetic Feature Augmentation Improves Generalization Performance of Language Models (https://arxiv.org/abs/2501.06434)
Comments:
          Accepted for presentation at IEEE SSCI 2025

- **What's New**: 본 연구에서는 제한적이고 불균형한 데이터셋에서 대형 언어 모델(LLMs)의 훈련 및 미세 조정 문제를 해결하기 위해 임베딩 공간에서 합성 샘플을 생성하여 특징을 증강하는 방법을 제안합니다. 이러한 방법은 모델의 성능을 개선하고 데이터 불균형 문제를 완화하는데 기여합니다. 다양한 오픈 소스 텍스트 분류 벤치마크에서 이 접근법의 효과를 검증하여, 모델의 강건성과 일반화 능력을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 SMOTE(합성 소수 클래스 오버 샘플링 기법)와 VAE(변분 오토인코더) 등의 기술을 사용하여 임베딩 공간에서 합성 데이터를 생성하는 방법을 탐구합니다. 비율 불균형 문제를 해결하기 위해 불충분한 클래스와 관련된 샘플의 임베딩을 생성하고, 이를 통해 균형 잡힌 훈련 데이터셋을 형성하였습니다. 이 방법은 텍스트 샘플과 해당 라벨로 구성된 데이터셋을 비선형 임베딩 함수로 매핑하여 표기된 특성을 유지하는 데 중점을 두었습니다.

- **Performance Highlights**: 실험 결과, 합성 샘플을 포함한 훈련 데이터셋은 합성 샘플을 포함하지 않은 경우보다 분류 성능이 크게 향상되었습니다. 이 연구는 다양한 벤치마크 데이터셋에서 모델의 강건성과 공정성을 높이는 데 효과적임을 입증합니다. 따라서 제안된 방법은 불균형한 데이터 시나리오 속에서도 실제 응용 분야에서 우수한 성능을 발휘할 수 있는 잠재력을 지니고 있습니다.



### Tensor Product Attention Is All You Need (https://arxiv.org/abs/2501.06425)
Comments:
          23 pages, 5 figures

- **What's New**: 이번 논문에서는 Tensor Product Attention (TPA)이라는 새로운 어텐션 메커니즘을 제안하여, 메모리 효율성과 성능을 모두 향상시킵니다. TPA는 텐서 분해(tensor decompositions)를 활용하여 질의(query), 키(key), 값(value)의 표현을 압축이 가능하게 하고, 이는 추론 과정에서 KV 캐시의 크기를 대폭 줄이는데 기여합니다. 또한, TPA는 Rotary Position Embedding (RoPE)과의 원활한 통합을 통해 기존 LLM 아키텍처에서 쉽게 적용될 수 있습니다.

- **Technical Details**: TPA는 계층적 텐서를 활용해 질의(Q), 키(K), 값(V)을 동적으로 분해하는 방식을 채택합니다. 이를 통해, TPA는 저성능의 KV 캐시 메모리 사용을 실질적으로 운용할 수 있으며, 더욱 강력한 표현 능력을 제공합니다. 이 논문에서는 TPA를 기반으로 한 새로운 모델 아키텍처인 Tensor ProducT ATTenTion Transformer (T6)도 소개하며, 다양한 언어 모델링 작업에서의 성능 향상을 보여줍니다.

- **Performance Highlights**: T6 모델은 기존의 표준 Transformer 모델, 즉 MHA, MQA, GQA, MLA에 비해 모든 평가 메트릭에서 더 우수한 성능을 달성했습니다. 특히, T6는 KV 캐시 크기를 줄이면서도 검증 perplexity와 downstream 성능에서도 일관된 향상을 보였습니다. 이러한 성능은 TPA의 메모리 효율성 덕분에 가능해졌으며, 이는 더 긴 입력 시퀀스를 처리할 수 있는 가능성을 제시합니다.



### Dynamics of "Spontaneous" Topic Changes in Next Token Prediction with Self-Attention (https://arxiv.org/abs/2501.06382)
- **What's New**: 이 논문은 인간의 대화에서 주제가 자연스럽게 변경되는 과정을 자기 attentive 기반 언어 모델의 다음 토큰 예측에서 발생하는 주제 변경과 비교합니다. 연구의 핵심은 주제의 연속성, 모호한 시퀀스, 그리고 주제 변화의 개념을 정의하고, 다중 주제 데이터셋에서 훈련된 모델들이 입력 주제에 대한 우선순위를 유지하는 것을 보여줍니다. 이는 인간의 사고 과정과 대비하여 언어 모델의 제한을 강조하고 있습니다.

- **Technical Details**: 저자들은 토큰 우선순위 그래프(Token Priority Graph, TPG)를 사용하여 주제를 정의하고, 특정 조건에서 주제 변경이 일어나는지를 분석합니다. 이들은 토큰의 우선순위가 입력 주제와 관련하여 유지됨을 증명하며, 낮은 우선순위의 토큰이 고우선순위 토큰보다 더 자주 나타날 때만 주제 변경이 가능하다고 설명합니다.

- **Performance Highlights**: 모델의 입력 길이가 증가하거나 주제가 겹칠 때 자연스러운 대화의 맥락에서 주제 변경의 가능성이 줄어드는 결과를 보여줍니다. 이러한 발견은 인간 대화의 역동성을 모방하려는 AI 대화 모델 개발에 중요한 기여를 하며, 인간과 유사한 대화 모델의 한계점을 부각시키는 데 중요한 역할을 합니다.



### AFRIDOC-MT: Document-level MT Corpus for African Languages (https://arxiv.org/abs/2501.06374)
Comments:
          under review

- **What's New**: 이 논문에서는 AFRIDOC-MT라는 문서 수준의 다중 패러럴 번역 데이터셋을 소개합니다. 이 데이터셋은 영어와 아프리카의 5개 언어인 아머릭어(Amharic), 하우사어(Hausa), 스와힐리어(Swahili), 요루바어(Yorùbá), 줄루어(Zulu)를 다룹니다. 총 334개의 건강 관련 문서와 271개의 정보 기술 뉴스 문서가 포함되어 있으며, 모두 영어에서 이들 언어로 사람에 의해 번역되었습니다.

- **Technical Details**: AFRIDOC-MT는 영어와 아프리카 언어 간의 번역 실험을 수행하기 위한 데이터셋으로, 문서 수준에서의 번역을 평가합니다. 각 언어쌍에 대해 10,000개의 문장이 포함되어 있으며, 문서 번역의 benchmark 실험을 통해 NMT 모델 및 LLM의 성능을 평가합니다. 연구에서 NLLB-200 모델이 평균 성능이 가장 높았고, GPT-4o가 일반 목적의 LLM보다 성능이 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: AFRIDOC-MT의 평가 결과, 수동으로 번역된 문서들 간의 성능 차이가 드러났습니다. 특히, GPT-4o에 의한 의사 문서 번역이 문장별 번역보다 유창하고 오류가 적은 것으로 분석되었습니다. 그러나 일부 LLM 모델은 저 생성(under-generation), 단어 또는 구의 반복(repetition), 아프리카 언어에 대한 잘못된 번역(off-target translations) 등의 문제를 보였습니다.



### Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts (https://arxiv.org/abs/2501.06365)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 의학 문헌에서 성 편향을 완화하기 위해 성별 직업 대명사를 중립화하는 파이프라인(pipeline)을 제시합니다. 1965년부터 1980년까지의 379,000개의 PubMed 초록을 처리하여 직업에 관련된 대명사를 수정했습니다. 이를 통해 개발된 MOBERT 모델은 성 중립화된 초록으로 훈련되었으며, 기존 모델인 1965Bert와 성능을 비교하였습니다.

- **Technical Details**: MOBERT는 BERT 모델을 기반으로 하여 성별 대명사가 도입된 직업 용어와 관련된 대명사를 중립화하도록 특별히 설계된 파이프라인을 사용했습니다. 이 과정에서는 Llama-3.1을 이용하여 대명사 해소(pronoun resolution) 쿼리를 수행하며, 숙련된 주석자가 반영된 사전 정의된 분류 규칙을 바탕으로 대명사 해소 작업을 진행했습니다. 초록에 포함된 직업 용어를 정확히 작성하는 것을 목표로 설계된 어휘 사전이 사용되었습니다.

- **Performance Highlights**: MOBERT는 포함된 대명사 교체 비율에서 70%를 달성하였고, 기존 모델인 1965Bert는 4%에 그쳤습니다. MOBERT의 성능 분석 결과, 대명사 교체 정확도는 훈련 데이터 내 직업 용어의 빈도와 상관관계가 있음을 보여줍니다. 이러한 결과는 향후 데이터셋 확장 및 파이프라인 개선을 통해 보다 공정한 언어 모델링을 할 수 있는 가능성을 제시합니다.



### Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages (https://arxiv.org/abs/2501.06346)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 문법적 수, 성별, 시제와 같은 형태통사적(morphosyntactic) 개념을 여러 언어에서 어떻게 학습하고 인코딩하는지 탐구합니다. 특히 Llama-3-8B와 Aya-23-8B에 대한 희소 오토인코더(Sparse Autoencoders)를 훈련함으로써, 논문의 주요 발견은 다양한 언어에서 이러한 개념의 표현이 공통적인 방향으로 인코딩된다는 것입니다. 이는 LLMs가 다국어 처리 기능을 공유한다는 것을 의미하며, 영어 데이터 중심으로 훈련된 모델에서도 강력한 형태통사적 개념의 교차 언어 추상성을 발전시킬 수 있음을 시사합니다.

- **Technical Details**: 연구에서는 희소 오토인코더를 사용하여 Llama-3-8B와 Aya-23-8B의 중간 활성화(intermediate activations)에 대해 훈련을 수행하였고, 다양한 형태통사적 개념에 대한 핵심 특징(feature)을 분석하였습니다. 이러한 특징들은 언어 간에 공유되는 정도를 정량화하기 위한 실험 디자인에 포함되며, 이 모델들의 생성에서 이 특징들의 역할을 검증합니다. 연구진은 또한 여러 언어에 걸쳐 일반화 가능한 추상적 개념들을 학습하는 것이 더 효과적인 파라미터 효율성을 제공할 수 있음을 강조합니다.

- **Performance Highlights**: 결과적으로, 연구는 형태통사적 개념 표현이 다양한 언어 간에 공유됨을 보여주며, 큰 언어 모델의 내부 공통언어가 단순히 영어 단어가 아니라, 개념이라는 것을 강조합니다. 이는 특히 LLM들이 데이터가 부족한 언어에서도 높은 성능을 발휘할 수 있음을 시사합니다. 논문은 이러한 발견이 기억력과 일반화의 논의에 기여할 수 있으며, 각 언어의 문법적 특징을 개별적으로 학습하는 방식보다 더 고차원적인 개념이 존재할 가능성을 제기합니다.



### Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks (https://arxiv.org/abs/2501.06286)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 도메인 특화 작업 수행 능력을 평가합니다. 특히, HotpotQA 데이터셋을 활용한 multi-hop question answering (MHQA) 문제에 중점을 두고 있습니다. 이 문제는 여러 텍스트 출처에서 정보를 결합하고 추론하는 과제가 필요하기 때문에 모델 성능 평가의 도전적인 기준이 됩니다.

- **Technical Details**: 연구에서는 두 단계 선택기-읽기 아키텍처를 설계했으며, 각 단계는 독립된 LLM을 활용합니다. 또한, Chain of Thought (CoT)와 질문 분해(question decomposition)와 같은 방법들을 적용하여 모델 성능 개선의 영향을 조사했습니다. 이러한 방법들은 LLM의 언어 이해 및 추론 능력을 높이는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 이러한 기법들과 함께 대형 언어 모델을 통합하면 F1 스코어에서 최대 4%의 성능 향상을 이끌어낼 수 있습니다. 이는 모델이 도메인 특화 작업을 다루고 복잡한 언어를 이해할 수 있는 능력을 증명하는 결과입니다.



### MinMo: A Multimodal Large Language Model for Seamless Voice Interaction (https://arxiv.org/abs/2501.06282)
Comments:
          Work in progress. Authors are listed in alphabetical order by family name

- **What's New**: 최신 연구에서는 MinMo라는 멀티모달 대형 언어 모델(Multimodal Large Language Model, LLM)을 소개하고, 80억 개의 매개변수를 통해 원활한 음성 상호작용을 제공하는 기술을 중심으로 하였습니다. MinMo는 음성-텍스트 정렬, 텍스트-음성 정렬, 음성-음성 정렬, 그리고 이중 상호작용 정렬의 여러 단계를 통해 훈련되어 다양한 음성 작업에서 최첨단 성능을 달성합니다. 특히, 이 모델은 사용자의 감정, 방언 및 말하는 속도를 기반으로 음성을 생성하는 등 향상된 지시 충족 기능을 지원합니다.

- **Technical Details**: MinMo는 140만 시간 이상의 다양한 음성 데이터를 통해 훈련되었으며, 여러 작업(예: Speech-to-Text, Text-to-Speech, Speech-to-Speech)을 포함합니다. 이 모델은 원래 텍스트 LLM의 기능 손실을 최소화하면서 음성 이해 및 생성을 향상시킵니다. 또한, 새로운 음성 디코더를 제안하여 단순한 구조와 높은 음성 생성 성능을 조화롭게 유지하며, 예측 지연 시간은 Speech-to-Text 시 약 100ms, 전체 이중 지연 시간은 이론상 600ms, 실제로는 800ms입니다.

- **Performance Highlights**: MinMo는 여러 공개 벤치마크(예: 구술 대화, 다국적 음성 인식, 음성 번역 등)에서 state-of-the-art 성능을 달성했습니다. 기존 모델과 달리 MinMo는 텍스트 LLM의 기능이 손실되는 것을 방지하여 음성 작업에서 우수한 성능을 보입니다. 사용자가 지정한 감정에 따라 음성을 생성하거나 특정 목소리를 모방하는 높은 지시 충족 정확도(98.4%)를 자랑하며, 전체 이중 상호작용을 원활하게 지원합니다.



### Punctuation's Semantic Role between Brain and Transformers Models (https://arxiv.org/abs/2501.06278)
- **What's New**: 최근 자연어 처리(NLP) 모델들이 인간의 뇌 활동과 어떻게 정합하는지를 탐구하는 연구가 진행되고 있습니다. 본 연구에서는 RoBERTa 동일한 방법론을 적용하여 새로운 NLP 모델의 뇌 활동과의 정합성을 평가하였습니다. 특히, 구두점(punctuation)을 제거한 텍스트가 세멘틱(processing) 처리에 미치는 영향을 조사하였습니다.

- **Technical Details**: 연구는 BERT를 기반으로 한 네 가지 변형 모델(RoBERTa, DistiliBERT, ELECTRA, ALBERT)에 대한 뇌-신경망 정합(matching) 평가를 통해 진행됩니다. 비슷한 방법을 사용하여, 인간의 뇌가 언어를 이해하는 방식을 보다 잘 이해하기 위한 고안된 실험들이 포함됩니다. fMRI 및 MEG 뇌 기록을 사용하여 모델의 특성과 뇌 데이터를 일치시키는 방법이 제안되었습니다.

- **Performance Highlights**: 연구 결과, RoBERTa 모델이 BERT보다 뇌 활동과 더 잘 일치하며 정확도에서도 우수한 성과를 보였습니다. BERT의 경우 구두점을 제외했을 때 더 높은 정확도를 기록하였고, 구두점을 포함한 원래 결과와 비교했을 때 문맥 길이가 늘어나도 정확도에 큰 영향을 미치지 않았습니다.



### Environmental large language model Evaluation (ELLE) dataset: A Benchmark for Evaluating Generative AI applications in Eco-environment Domain (https://arxiv.org/abs/2501.06277)
- **What's New**: 이번 연구에서는 생태 및 환경 과학에 대한 평가를 위한 첫 번째 기준점인 Environmental Large Language model Evaluation (ELLE) 질문 응답(QA) 데이터셋을 소개합니다. 이 데이터셋은 최소한 1,130개의 질문-응답 쌍을 포함하고 있으며, 16개의 환경 주제를 포괄합니다. ELLE 데이터셋은 이러한 분야에서의 성능 평가를 표준화하여, 생성적 AI의 성능을 일관되고 객관적으로 비교할 수 있게 합니다.

- **Technical Details**: ELLE 데이터셋은 주제(domain), 난이도(difficulty), 유형(type)에 따라 분류된 질문-응답 쌍을 제공하며, 이는 생태학 및 환경과학의 다양한 응용 분야에 적합한 평가 도구입니다. 이 연구는 통합된 평가 프레임워크의 부재로 인해 제한된 생성적 AI의 효율성을 극복하는 데 기여합니다. ELLE 데이터셋은 지속 가능한 환경 결과를 위한 생성적 AI 기술의 개발과 응용을 촉진하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 데이터셋을 통해 생태 및 환경 과학 분야에서의 생성적 AI의 성능이 평가될 수 있으며, 이를 통해 앞으로의 연구 방향과 정책 지원에 중대한 영향을 미칠 것으로 기대됩니다. 이 연구는 생성적 AI 기술이 환경 모니터링, 데이터 분석, 교육 및 정책 지원과 같은 생태 및 환경 애플리케이션에서의 효과를 극대화할 수 있도록 합니다.



### AgoraSpeech: A multi-annotated comprehensive dataset of political discourse through the lens of humans and AI (https://arxiv.org/abs/2501.06265)
- **What's New**: AgoraSpeech는 2023년 그리스 총선에서 여섯 개 정당의 171개 정치 연설로 구성된 고품질 데이터셋입니다. 이 데이터셋은 텍스트 분류, 주제 식별, 감정 분석, 개체명 인식, 양극화 및 포퓰리즘 탐지 등 여섯 가지 자연어 처리(NLP) 작업을 위한 주석이 포함되어 있습니다. 데이터 수집과 주석 과정에서 ChatGPT를 사용하여 시너지 효과를 내었으며, 전문가가 이를 검증한 후 높은 품질의 데이터를 보장합니다.

- **Technical Details**: 연구팀은 그리스 총선 직전인 2023년 5월에 정치인들의 연설을 수집하였습니다. 수집된 데이터는 총 5,279개의 단락과 717,718개의 단어로 이루어져 있으며, 31,674개의 주석이 포함되어 있습니다. 이를 통해, 정치 담론 분석을 위한 본격적인 벤치마크 데이터셋이 구축되었습니다.

- **Performance Highlights**: ChatGPT의 주석 성능을 검증한 결과, 감정 분석에서 93%의 정확성을 기록했으며, 텍스트 분류는 89%로 나타났습니다. 그러나 주제 분류와 같은 더 복잡한 작업에서는 61%의 정확도로 하락했습니다. 이러한 결과를 통해 AgoraSpeech 데이터셋은 다차원 정치 담론 분석을 위한 유용한 도구임을 보여줍니다.



### What Matters for In-Context Learning: A Balancing Act of Look-up and In-Weight Learning (https://arxiv.org/abs/2501.06256)
- **What's New**: 대형 언어 모델(LLM)에서의 In-Context Learning (ICL) 메커니즘을 체계적으로 분석한 연구입니다. 이 연구에서는 훈련 데이터의 특성과 모델 아키텍처의 역할을 넘어서서 ICL의 근본적인 요소를 파악하고자 했습니다. 데이터 시퀀스에서 개념적 반복(conceptual repetitions)의 중요성을 강조하고, 이는 훈련 과정에서 안정적이고 비일시적인 ICL 성능을 보장하는 데 기여한다고 주장합니다.

- **Technical Details**: 연구에서 저자들은 깊은 자회귀 모델을 통해 훈련 데이터 시퀀스의 개념적 반복이 ICL에 미치는 영향을 조사했습니다. 특히, 2048 토큰의 컨텍스트 윈도우에서 n-gram 반복성을 관찰했고, 이는 훈련 예시 내에서의 전환 가능성을 높이는 데 중요하다고 설명합니다. 또한 복잡한 인-웨이트 러닝(in-weight learning) 목표가 ICL의 일관된 성능에 중요한 역할을 한다고 제안하며, 반복과 복잡한 목표 간의 상호작용을 연구했습니다.

- **Performance Highlights**: 저자들은 데이터 시퀀스의 단일 반복만으로도 ICL이 가능하다는 것을 발견하였고, 이는 높은 폭발성(burstiness)이나 skewed label distribution과 같은 다른 특성이 없더라도 성립됩니다. 연구 결과, LLM에서 ICL 성능은 훈련 초기에 일시적인 경향을 보일 수 있으나 반복과 복잡한 목표를 통해 안정적으로 유지될 수 있음을 확인했습니다. 이러한 발견은 대형 모델에서 ICL의 메커니즘을 이해하는 데 중요한 기여를 할 것으로 기대됩니다.



### Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words (https://arxiv.org/abs/2501.06254)
- **What's New**: 이번 논문에서는 Sparse Autoencoders (SAEs)의 새로운 평가 방식을 제안하며, polysemous words (다의어)에서 monosemantic features (단어의 의미를 명확하게 표현하는 특징)를 추출할 수 있는지 분석합니다. 기존의 MSE-L0 Pareto frontier 성능 향상이 단순히 interpretability (해석 가능성)를 증가시키지 않는다는 점을 강조합니다. 또한 Poly-Semantic Evaluations (PS-Eval)이라는 새로운 메트릭을 통해 SAEs의 성과를 정량적으로 평가할 수 있는 방법을 제시합니다.

- **Technical Details**: SAEs는 LLM의 복잡한 세 가지 차원의 활성도를 단일 의미의 피쳐로 변환하기 위해 설계되었습니다. 본 연구는 PS-Eval을 통해 polysemantic activations (다의적 활성화)에서 monosemantic features (단일 의미 특징)를 효과적으로 추출할 수 있는지 평가합니다. 실험에서는 다양한 activation functions (활성화 함수)와 latent dimensions (잠재 변수를 사용하는 차원)을 시험했고, Attention module의 깊이에 따라 해석의 정확성이 어떻게 변화하는지를 탐구했습니다.

- **Performance Highlights**: PS-Eval의 결과는 SAEs가 다양한 polysemous words에서 명확한 단일 의미의 피쳐를 추출한다는 것을 보여줍니다. 특히, deeper layers (더 깊은 층)에서는 specificity (특정성) 점수가 높아지며, Attention mechanism (어텐션 메커니즘)이 다의어를 구분하는 데 효과적임을 확인했습니다. 최종적으로, MSE와 L0 sparsity에만 집중하는 기존의 연구들이 monosemantic feature의 효과적인 추출을 간과함을 일깨워주는 성과를 도출하였습니다.



### A partition cover approach to tokenization (https://arxiv.org/abs/2501.06246)
- **What's New**: 이번 연구에서는 토큰화(tokenization) 문제를 최적화(optimization) 목표로 설정하고, 이 문제를 NP-hard로 증명했습니다. 전통적인 Byte Pair Encoding (BPE) 방법 대신 GreedTok이라는 새로운 다항식 시간의 탐욕 알고리즘을 제안합니다. GreedTok은 기존의 조합 기반 병합 방식에서 벗어나 토큰의 순서만을 요구하여, 사용자가 선택한 사용자 정의 토큰 집합을 사용할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: 새로운 최적화 목표를 통해 GreedTok은 파라미터로 주어진 토큰 및 카운트(count) 정보를 활용하여 효율적으로 데이터를 표현합니다. 이를 통해 토큰의 수를 줄이면서도 동일한 또는 더 나은 압축 유틸리티를 달성할 수 있습니다. 이러한 접근은 전통적인 병합 방식의 제약에서 자유로워, 문제를 보다 일반적인 형태로 해결할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, GreedTok은 BPE보다 평균 3%에서 최대 5% 더 적은 토큰을 필요로 하며, 평균적으로 13% 더 적은 수의 토큰으로 유사한 압축 성능을 달성했습니다. 이러한 결과는 기존의 BPE 방법에 비해 응용 가능성 및 성능에서 뛰어난 성과를 보여줍니다. 또한, GreedTok의 성능은 잘 연구된 가중 최대 커버리지 문제(weighted maximum coverage problem)에 자연스럽게 연결되어, 이론적인 근거가 뒷받침되어 있습니다.



### FLAME: Financial Large-Language Model Assessment and Metrics Evaluation (https://arxiv.org/abs/2501.06211)
- **What's New**: 이 논문은 FLAME이라는 한국어 금융 대형 언어 모델(LLM) 평가 시스템을 도입하여, 현재 존재하는 다양한 금융 인증 및 실제 비즈니스 시나리오를 종합적으로 평가하는 방안을 제안합니다. LLM의 가치를 종합적으로 평가할 수 있는 새로운 벤치마크인 FLAME-Cer와 FLAME-Sce의 두 가지 코어 평가 기준을 포함하고 있습니다. FLAME은 14종의 공인 금융 자격증에 대한 질문으로 구성된 데이터를 활용하며, 특히 16,000개의 질문을 수집하여 정확성과 대표성 확보에 만전을 기했습니다.

- **Technical Details**: FLAME-Cer는 AFP, CAA, CFA, CPA, FRM 등 14개의 금융 인증 기준을 포함하며, 각각의 질문은 세 가지 중요도로 분류되었습니다. FLAME-Sce는 재무 지식, 문서 작성, 고객 서비스, 리스크 관리 등 10개의 주요 비즈니스 시나리오를 평가하며, 총 5,000개가 넘는 고품질 평가 질문을 포함하고 있어 실제 금융 업무에 대한 폭넓은 평가가 가능합니다. 이 시스템은 학문적 이론과 실무 경험을 결합하여 평가의 권위성과 전문성을 보장합니다.

- **Performance Highlights**: FLAME 시스템을 통해 여섯 개의 대표 LLM을 평가한 결과, Baichuan4-Finance가 대부분의 과제에서 높은 성능을 보여주었습니다. FLAME은 LLM의 실제 금융 산업 요구에 맞는 평가 작업을 정교하게 조정하였으며, 데이터 보안 및 비즈니스 규정 준수 관점에서도 평가를 진행하여 금융 산업 규제에 부합하는 모델을 보장하고 있습니다. 이 평가 시스템은 금융 시장과 규제 요구 사항이 발전함에 따라 지속적으로 업데이트되어 실질적이고 전문적인 평가 기준을 제공할 수 있도록 설계되었습니다.



### Applications of natural language processing in aviation safety: A review and qualitative analysis (https://arxiv.org/abs/2501.06210)
- **What's New**: 이번 연구는 항공 안전 분야에서 자연어 처리(Natural Language Processing, NLP)를 이용한 기계 학습 알고리즘의 적용 가능성을 탐구하고 있습니다. 현재까지 '자연어 처리'와 '항공 안전'을 키워드로 검색한 결과, 2024년 5월 기준 34개의 스코퍼스(Scopus) 논문이 발견되었습니다. 이러한 연구들을 분석함으로써, NLP의 항공 분야에서의 방법론, 발견 사항 및 시사점을 파악할 수 있습니다.

- **Technical Details**: 연구에서는 현재 항공 안전을 위한 NLP 문헌의 상태를 조사하기 위해 질적(qualitative) 및 양적(quantitative) 도구를 활용하였습니다. 질적 분석에서는 연구의 동기, 목표 및 결과를 요약하여 NLP가 항공 안전의 중요한 문제를 식별하고 향상하는 데 어떻게 기여할 수 있는지를 보여줍니다. 또한 이 연구는 연구의 공백을 식별하고 항공 산업을 위한 실질적인 권장 사항을 제시합니다.

- **Performance Highlights**: NLP를 항공 안전에 적용할 때 발생하는 도전 과제로는 대규모로 주석이 달린 데이터셋의 필요성과 복잡한 모델 해석의 어려움이 있습니다. 연구에서는 데이터 주석을 위한 능동 학습(active learning)과 모델 해석을 위한 설명 가능한 AI(explainable AI)와 같은 해결 방안을 제안합니다. 사례 연구들은 NLP가 항공 안전을 증진시키는데 성공적으로 적용된 예시를 보여주며, 항공 안전이 보다 안전하고 효율적이 될 수 있는 잠재력을 강조합니다.



### Enhancing AI Safety Through the Fusion of Low Rank Adapters (https://arxiv.org/abs/2501.06208)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 지침 세밀 조정(instruction fine-tuning) 시 발생할 수 있는 해로운 응답 생성 문제를 해결하기 위해 저순위 어댑터 융합(Low-Rank Adapter Fusion, LoRA) 방식을 탐구합니다. LoRA 융합을 통해 태스크 어댑터와 안전 어댑터 간에 협력하여 해로운 비율을 42%까지 줄일 수 있었지만, 안전한 프롬프트를 부정하는 과도한 안전 행동도 발견되었습니다.

- **Technical Details**: 연구에서는 AOA(Absolutely Obedient Agent) 데이터셋을 바탕으로 LoRA 어댑터의 조합을 통해 모델의 안전성을 향상시키는 방법을 제시합니다. 여기서 LoRA는 큰 모델의 가장 적은 수의 매개변수만 업데이트하여 성능을 개선하기 위한 방법으로, 조정된 어댑터를 통해 안전 기준을 유지하면서도 모델의 일반적인 능력을 크게 저해하지 않도록 설계되었습니다. 안전 데이터셋은 GPT-4로 평가되어, 단순한 거부 반응뿐만 아니라 더 복잡한 윤리적 반응을 포함합니다.

- **Performance Highlights**: LoRA 융합 방법론은 기존의 안전성 기술들과 비교하여 모델의 안전성을 유지하면서도 태스크 성능을 증대시키는 효과적인 방안을 제시합니다. 연구에서는 기존의 베이스라인과의 비교 분석을 통해 LoRA의 유효성을 입증하였고, 안전성을 강화하는 동시에 고성능 결과를 생성할 수 있는 능력을 강조하였습니다. 이러한 접근법은 안전성과 성능 간의 균형을 최적화하는 데 중요한 기여를 할 것으로 기대됩니다.



### SST-EM: Advanced Metrics for Evaluating Semantic, Spatial and Temporal Aspects in Video Editing (https://arxiv.org/abs/2501.07554)
Comments:
          WACV workshop

- **What's New**: 이 논문에서는 비디오 편집 모델의 성능 평가를 위해 SST-EM(모든 3차원적 평가 기준을 반영한 새로운 평가 메트릭)을 제안하고 있습니다. 전통적인 평가 지표의 한계를 극복하기 위해 Vision-Language Models (VLMs), Object Detection, Temporal Consistency 체크를 통합한 새로운 방식으로 구성을 하였습니다. 이 메트릭은 인간 평가를 기반으로 최적화된 가중치를 포함하여, 비디오 편집에서 의미적 충실도와 시간적 매끄러움을 종합적으로 평가하도록 설계되었습니다.

- **Technical Details**: SST-EM 메트릭은 네 가지 주요 구성 요소로 이루어져 있습니다: (1) VLM을 사용한 프레임의 의미적 추출, (2) Object Detection을 통한 주요 객체 추적, (3) LLM 에이전트를 이용한 주요 객체의 세밀한 다듬기, 그리고 (4) Vision Transformer (ViT)를 통한 시간적 일관성 평가입니다. 이 메트릭은 인간의 평가 결과와 회귀 분석을 통해 가중치를 조정하여 통합적인 평가 프레임워크를 제공합니다. 이러한 구성 요소들은 비디오 편집의 여러 차원을 동시에 고려하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 논문에서는 SST-EM 메트릭이 기존의 평가 방법보다 우수함을 입증하였으며, 이를 통해 비디오 편집 연구에서 더 신뢰할 수 있는 성능 기준을 제공하고 있습니다. 다양한 비디오 편집 모델에서의 성능을 평가하며 인간 검증에 기반한 결과를 참조하여, 높은 정확성을 자랑하는 신뢰할 수 있는 메트릭으로 자리잡게 되었습니다. SST-EM은 비디오 편집 품질의 평가에 있어 보다 견고한 틀을 제공하며, 관련 연구의 방향성을 제시하고 있습니다.



### Parallel Key-Value Cache Fusion for Position Invariant RAG (https://arxiv.org/abs/2501.07523)
Comments:
          5 pages

- **What's New**: 이번 논문에서는 RAG (Retrieval Augmented Generation) 파이프라인의 효율성을 높이기 위해 키-값 융합(Key Value Fusion, KV Fusion) 프레임워크를 제안합니다. 이 프레임워크는 디코더 전용 모델인 LLMs에서 정보의 위치에 대한 민감성을 줄여주는 새로운 방법으로, 입력 순서와 관계없이 일관된 출력을 생성하는 것을 목표로 하고 있습니다. 특히, 입력 문맥의 위치에 관계없이 모델의 일관된 반응을 보장하여 기존 방법보다 강력한 성능을 보여줍니다.

- **Technical Details**: KV Fusion 구조는 두 가지 구성 요소로 이루어져 있습니다. 첫 번째는 주요-값 캐시를 병렬로 추출하는 프리필 디코더(prefill decoder)이며, 두 번째는 추출된 키-값 캐시를 사용하여 일관된 출력을 생산하는 훈련 가능한 디코더(trainable decoder)입니다. 이를 통해 입력 문맥에 균일한 위치 정보를 주입하여, 다양한 입력 순서에서도 일관된 결과를 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, KV Fusion은 NQ, TriviaQA, POPQA를 포함한 열린 도메인 질문 응답 데이터셋에서 각각 21.4%, 6.4%, 6.6%의 정확도 개선을 달성했습니다. 또한, 기존 방법들보다 다른 문맥이 추가되더라도 응답의 정확도가 강건하고 안정적인 것을 보여주었습니다. 이러한 성과는 KV Fusion의 위치 불변성(position invariance) 특성 덕분입니다.



### Joint Automatic Speech Recognition And Structure Learning For Better Speech Understanding (https://arxiv.org/abs/2501.07329)
Comments:
          5 pages, 2 figures, accepted by ICASSP 2025

- **What's New**: 최근의 연구들은 음성 인식 및 이해를 동시에 달성하는 기법에 한계를 보였습니다. 본 논문에서는 JSRSL(Joint Speech Recognition and Structure Learning)이라는 새로운 E2E 모델을 제안하며, 이는 음성을 정확하게 기록하고 구조화된 정보를 동시에 추출할 수 있습니다. 이 모델은 ASR과 구조 예측을 효과적으로 결합하여 성능을 향상시킵니다.

- **Technical Details**: JSRSL 프레임워크는 Parallel Transformer를 기반으로 하며, ASR 모듈과 span 간의 연결을 위해 Refiner 모듈을 활용합니다. 또한 Bridge 모듈이 음성 데이터의 표현을 텍스트 표현으로 변환하는 기능을 수행하여 구조화된 학습을 위한 리치 특징을 추출합니다. 이론적으로 간섭 없이 ASR 및 NLU 모듈의 공동 훈련을 가능하게 합니다.

- **Performance Highlights**: 제안된 JSRSL 모델은 AISHELL-NER 및 SLURP 데이터셋에서 실험을 통해 기존의 sequence-to-sequence 방식보다 뛰어난 성능을 보였습니다. 특히, 전반적인 음성 기록 및 이해 능력에서 SOTA(state-of-the-art) 성능을 달성하였습니다. 이러한 우수한 성능은 음성 인식과 구조 예측을 동시에 고려할 수 있는 통합 접근법 덕분입니다.



### Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Mod (https://arxiv.org/abs/2501.07246)
- **What's New**: 이번 연구에서는 Chain-of-Thought (CoT) 추론을 대형 오디오-언어 모델(large audio-language models, LALMs)에 통합하여 추론 능력을 향상시키는 첫 번째 탐색을 진행하였습니다. LALMs는 음성 인식 및 오디오 캡셔닝과 같은 작업에서 뛰어난 성과를 보여주었지만, 복잡한 문제 해결에 필요한 추론 능력은 크게 미진했습니다. 이 연구는 CoT 방법론이 LALM의 성능을 어떻게 개선할 수 있는지를 살펴보았으며, 향후 연구의 방향성을 제시하고 있습니다.

- **Technical Details**: 이 논문에서는 Manual-CoT, Zero-Shot-CoT, Desp-CoT와 같은 대표적인 CoT 방법을 체계적으로 평가하였습니다. Manual-CoT는 손으로 만든 샘플을 통한 인지 구조를 근거로 하여 모델의 출력을 유도하는 방식으로, 고품질의 추론 체인을 생성하는데 기여합니다. 반면 Zero-Shot-CoT는 별도의 예제 없이도 자연어 프롬프트를 통해 논리적 단계를 이끌어내는 점이 특징입니다.

- **Performance Highlights**: 연구 결과 CoT 방법이 쉬운 및 중간 난이도의 작업에서 성능을 크게 향상시키지만, 어려운 작업에서는 모델의 성능을 저하시킬 수 있는 문제를 발견하였습니다. 또한, 추론 경로의 길이와 정확도 간의 긍정적인 상관 관계를 확인하여, 고급 지시 따르기 및 추론을 위한 확장 가능한 추론 가능성을 보여주었습니다. 이는 복잡한 오디오 이해 시스템에 추론 능력을 통합하는 것이 얼마나 중요한지를 강조합니다.



### Can Vision-Language Models Evaluate Handwritten Math? (https://arxiv.org/abs/2501.07244)
- **What's New**: 최근 Vision-Language Models (VLMs)의 발전은 학생들이 작성한 손글씨 응답의 자동 채점 가능성을 열어주고 있습니다. 그러나 VLM이 손글씨 콘텐츠를 평가하고 추론하는 능력을 테스트한 종합적인 연구는 부족합니다. 이를 해결하기 위해 FERMAT라는 벤치를 소개하여 VLM의 손글씨 수학 콘텐츠에서 오류를 감지하고 수정하는 능력을 평가합니다. 이 벤치는 7-12학년의 609개 수작업 문제에서 유도된 2,200개 이상의 손글씨 수학 솔루션을 포함하고 있습니다.

- **Technical Details**: FERMAT은 VLM의 오류 감지, 로컬라이제이션(localization), 수정 오류를 평가하기 위해 세 가지 핵심 작업을 기반으로 합니다. 실험 결과, 현재 대부분의 모델이 손글씨 입력 처리에서 어려움을 겪으며, Gemini-1.5-Pro가 오류 수정에서 77%의 최고 성과를 기록했습니다. 모델의 성능은 문제 유형이나 오류 카테고리에 대한 추가 메타 정보 제공 시 향상되는 것을 확인했습니다. 손글씨보다 인쇄된 텍스트를 제공할 경우 오류 로컬라이제이션 정확도가 높아지는 것도 주목할 만합니다.

- **Performance Highlights**: 현재 VLM들은 손글씨 텍스트에 대한 추론 능력에서 중요한 단점을 보이고 있습니다. 특히, 다양한 오류 유형에서의 성능을 평가한 결과, VLM들은 기존의 간단한 암기식 평가를 넘어서는 더 복잡한 평가를 요구합니다. 실험을 통해 VLM의 수학적 사고 능력이 제한적임을 발견했고, 이는 더 많은 연구와 개발이 필요함을 시사합니다. FERMAT과 관련 자원들은 공개되어 더 많은 연구를 촉진할 것입니다.



### BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literatur (https://arxiv.org/abs/2501.07171)
- **What's New**: BIOMEDICA는 PubMed Central Open Access 데이터를 활용하여 생물 의학 분야의 비전-언어 모델(vision-language models, VLMs) 개발을 위한 포괄적이고 접근 가능한 데이터셋을 제공합니다. 이 프레임워크는 6백만篇의 논문에서 2400만 개의 독특한 이미지-텍스트 쌍을 생성하여, 시각적 및 텍스트 정보를 통합하여 환자 치료 지원을 혁신할 수 있는 가능성을 열었습니다. 또한, BMCA-CLIP 방식을 통해 대량의 데이터를 로컬에 다운로드하지 않고도 모델을 지속적으로 사전 훈련할 수 있게 되었습니다.

- **Technical Details**: BIOMEDICA는 전체 PubMed Central Open Access 레포지토리를 표준화된 밀집 아카이브로 효율적으로 추출하고 직렬화하는 ETL(Extraction, Transformation and Loading) 파이프라인을 포함하여, 전문가 큐레이션을 통해 고도로 주석이 달린 데이터셋을 제공합니다. 이 데이터셋은 각 데이터 포인트에 대해 27개의 고유 메타데이터 필드가 주어지며, Parquet 형식과 WebDataset 형식으로 제공되어 빠른 질의와 필터링을 가능하게 하여 현대적인 훈련 전략을 탐색할 수 있도록 합니다.

- **Performance Highlights**: 모델은 40가지 생물 의학 작업에서 평균 6.56%의 성능 개선을 기록하며, 특히 피부과 및 안과 분야에서 29.8% 및 17.5%라는 높은 성능 향상을 보였습니다. BIOMEDICA 데이터셋을 통해 기존의 최첨단 모델보다 10배 적은 계산 자원으로 더 나은 결과를 달성하는 등, 공유 및 협업을 위한 코드를 공개하고 있습니다.



### Research on the Online Update Method for Retrieval-Augmented Generation (RAG) Model with Incremental Learning (https://arxiv.org/abs/2501.07063)
- **What's New**: 이 논문에서는 정보 기술의 급속한 발전과 데이터 양의 기하급수적 증가에 대응하기 위한 온라인 업데이트 방법을 제안합니다. 기존의 Retrieval Enhanced Generation (RAG) 모델을 기반으로 하며, 데이터의 동적 기억 장치를 활용하여 새로운 데이터를 효율적으로 통합하는 혁신적인 메커니즘을 포함하고 있습니다. 이를 통해 빠르게 변화하는 정보 환경에서 실시간으로 지식을 업데이트하고 적응하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 동적 기억(dynamic memory)을 사용하여 새로운 데이터 샘플을 수집하고, 조정 가능한 지식 증류(knowladge distillation) 전략을 통해 모델에 통합하는 방식을 채택합니다. 또한, 계층적 인덱싱(hierarchical indexing)과 다층 게이팅(multi-layer gating) 메커니즘을 통해 검색 모듈의 정확성을 높이고 있습니다. 마지막으로 다양한 입력 유형에 대응하기 위해 멀티 스테이지 네트워크 구조를 설정하여, 각 단계의 중간 표현에 대해 cross-attention 매칭 및 스크리닝을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 지식 보존(knowledge retention)과 추론 정확도(inference accuracy) 측면에서 기존의 주류 비교 모델보다 우수한 성능을 보여주었습니다. 이는 새로운 지식과 기존 지식의 효과적인 통합과 반복 업데이트를 통해 이루어졌습니다. 또한, 특히 동적이고 변화하는 데이터 환경에서의 적응성 향상에 기여하고 있습니다.



### Leveraging ASIC AI Chips for Homomorphic Encryption (https://arxiv.org/abs/2501.07047)
Comments:
          16 pages, 10 figures, 4 algorithms, 7 tables. Enabling Google TPUv4 for privacy-preserving AI inference

- **What's New**: 본 논문은 동형암호(homomorphic encryption, HE) 연산을 기존의 AI 가속기에서 실행 가능하도록 지원하는 CROSS 컴파일러를 제안합니다. 이를 통해 클라우드 기반 AI 서비스에서 민감한 고객 데이터를 안전하게 처리할 수 있는 새로운 가능성을 열었습니다. 특히 TPU와 같은 ASIC AI 가속기를 활용하여 HE 작업의 성능을 극대화하는 방법을 모색하였습니다.

- **Technical Details**: CROSS 컴파일러는 모듈러스 연산을 위한 Barrett 감소, 고정밀 산술 연산을 지원하는 청크 분해(chunk decomposition), 2D 행렬 엔진을 적절히 활용하는 매트릭스 정렬 변환(matrix aligned transformation) 등을 구현합니다. 이러한 기술적 접근 방식은 HE 연산을 low-precision 하드웨어에서 효과적으로 실행할 수 있도록 합니다. 논문에서는 이를 통해 기존의 HE 성능 향상 문제를 해결하는 방법을 제시합니다.

- **Performance Highlights**: 구글 TPUv4에서 CROSS의 성능을 평가한 결과, 이전 작업들과 비교할 때 최대 161배, 평균 5배 이상의 속도 개선을 이뤘습니다. 기존 HE ASIC보다 성능 측면에서 약 50배 느리지만, 커스터마이즈된 ASIC 없이도 높은 성능을 기대할 수 있는 만큼 산업적 가치가 큽니다. AI 가속기 사용을 통해 HE의 상용화 가능성을 제시한 점이 특히 주목할 만합니다.



### LEO: Boosting Mixture of Vision Encoders for Multimodal Large Language Models (https://arxiv.org/abs/2501.06986)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 비주얼 이해 능력을 향상시키기 위한 새로운 접근법으로 LEO라는 하이브리드 모델을 제안합니다. LEO는 두 개의 비전 인코더를 통합하여 데이터 처리 시 각 인코더의 전문성을 활용하는 타일 기반 후적응 융합 전략(tile-level post-adaptation fusion strategy)을採用합니다. 이를 통해 LEO는 기존 하이브리드 모델보다 더 효율적으로 다양한 비전 인코더를 융합하여 작업 수행에서 우수성을 발휘할 수 있습니다.

- **Technical Details**: LEO의 구성은 입력 이미지를 448x448 크기의 타일로 분할하고, 각 타일에 대해 두 개의 비전 인코더가 각각의 특징 표현을 제공하는 방식으로 진행됩니다. 이후 시각적 임베딩에 픽셀 언셔플링(pixel unshuffling)을 적용하여 각 인코더의 시각적 토큰 수를 줄입니다. 이러한 후적응 융합 전략을 통해 LEO는 텍스트 토큰과 비전 토큰을 결합하여 LLM으로 처리할 수 있는 형태로 제공합니다.

- **Performance Highlights**: LEO는 13개 비전-언어 벤치마크에서 전반적으로 최고의 성능을 보였습니다. 특히 LEO는 기존 오픈소스 MLLM 및 하이브리드 모델들과 비교하여 다수의 작업에서 우수한 성능을 나타내며, 자율 주행 분야에 적합하도록 모델 아키텍처를 변경하지 않고도 경쟁력 있는 성능을 입증했습니다. 또한 이 연구는 자율 주행 분야를 위한 하이브리드 MLLM의 첫 번째 탐색으로 주목받고 있습니다.



### Risk-Averse Finetuning of Large Language Models (https://arxiv.org/abs/2501.06911)
Comments:
          Neurips 2024

- **What's New**: 이번 연구에서는 큰 언어 모델(LLMs)이 독성이 있는 결과물을 생성하는 문제를 완화하기 위한 새로운 접근법을 제안합니다. 위험 회피(risk-averse) 원칙을 LLM의 미세 조정(fine-tuning)에 통합하여, 독성 출력의 발생을 최소화하는 것을 목표로 하고 있습니다. 특히, 이 방법은 조건부 가치 위험(Conditional Value at Risk, CVaR)이라는 위험 척도를 최적화하여, 긍정적인 언어 생성을 지속적으로 유지하면서 독성 출력을 효과적으로 피할 수 있도록 LLM을 훈련합니다.

- **Technical Details**: 연구팀은 위험 회피 강화 학습(RA-RLHF) 알고리즘을 개발하여 다양한 레벨의 부정성과 독성을 가진 프롬프트를 미세 조정합니다. RLHF는 기대 보상을 극대화하려는 반면, RA-RLHF는 생성된 경로의 위험 척도를 최적화하여 독성을 줄이는 것을 목표로 합니다. 초기 훈련 단계에서는 작은 위험 수준을 설정하여 정책이 성공적인 출력을 생성하도록 훈련한 뒤, 위험 목표에 따라 배치 크기를 줄여가며 높은 위험의 프롬프트에 집중하도록 합니다.

- **Performance Highlights**: RA-RLHF는 세 가지 언어 생성 시나리오에서 평가되었으며, 모두 기존 RLHF 접근법보다 우수한 성능을 나타냈습니다. 실험 결과, RA-RLHF는 위험한 프롬프트에서 특히 높은 효과성을 보였으며, 긍정적인 언어 생성을 성공적으로 이끌어냈습니다. 이 연구의 결과는 LLM이 독성을 줄이면서도 생성 작업의 효율성을 유지할 수 있음을 보여주며, 온라인 담론 환경의 안전성을 높이는데 기여할 것으로 기대됩니다.



### Causal Claims in Economics (https://arxiv.org/abs/2501.06873)
Comments:
          For data, interactive tools, and additional project information, visit this https URL. The website contains resources such as data downloads, interactive author and paper-level knowledge graphs, and more

- **What's New**: 이 논문은 1980년부터 2023년까지의 44,000개 이상의 NBER와 CEPR 작업 논문을 분석하여, 경제 개념 및 그 관계를 매핑하는 지식 그래프를 구축했습니다. 특히 인과 추론 방법(ex: DiD, IV, RDD, RCTs)을 통해 문서화된 일반 주장과 인과 주장을 구분하며, "신뢰성 혁명(credibility revolution)"의 확산이 인과 주장에 미치는 영향을 조명합니다. 인과 서사 복잡성(causal narrative complexity)이 상위 5개 저널에 게재되거나 더 많은 인용을 받을 가능성과 강한 상관관계를 보여줍니다.

- **Technical Details**: 이 연구는 LLMs(대형 언어 모델)인 GPT-4o-mini를 활용하여 비구조적 텍스트에서 구조화된 정보를 추출하고 분석하는 방법론적 접근을 사용했습니다. 이 과정에서는 30페이지 분량의 경제 작업 논문을 처리하며, 핵심 메타데이터와 인과 주장을 식별합니다. 데이터와 변수의 사용을 체계적으로 추출하기 위해 JSON 구조를 적용하여, 높은 정확도와 일관성을 유지하는 방식을 채택하였습니다.

- **Performance Highlights**: 결과적으로, 인과 혁신(causal innovation)과 방법론적 엄격함(methodological rigor)이 학문적 인정을 받는 주요 요인으로 작용하지만, 지속적인 영향력을 확보하기 위해서는 새로운 기여와 기존 경제 담론 간의 통합(balance)이 필요함을 보여주었습니다. 인과적 방법론에 기반으로 한 새로운 개념 쌍(underexplored concept pairs)의 다리가 인정받지만, 장기적인 인용과의 일관된 연관성은 발견되지 않았습니다. 본 연구는 경제학의 저널 출판 성공에 영향을 미치는 여러 요소를 밝히고 있습니다.



### Transfer Learning of Tabular Data by Finetuning Large Language Models (https://arxiv.org/abs/2501.06863)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)을 활용하여 테이블 데이터 분류에서의 효과적인 전이 학습 방법을 탐구합니다. 기존의 딥러닝 기법들이 테이블 데이터에서의 성과가 미흡했던 이유는 이질적인 특성과 샘플 수의 제한으로 지적됩니다. 제안된 방법은 LLM을 최적화하여 적은 수의 샘플로도 성능을 극대화할 수 있도록 해 새로운 가능성을 열어 줍니다.

- **Technical Details**: LLM의 전이 학습은 기존의 LLM의 지식을 활용하여 더 나은 분류 성능을 제공하는 방법입니다. 본 연구에서는 데이터 직렬화(data serialization) 단계를 통해 테이블 데이터를 텍스트 프롬프트로 변환하고, DistilGPT2 모델을 사용하여 이를 분류 작업에 적용합니다. 이 과정에서 메타데이터(metadata)를 포함한 텍스트 프롬프트를 통해 LLM의 성능을 개선합니다.

- **Performance Highlights**: 제안된 LLM 최적화 방법은 기존의 머신러닝 및 딥러닝 기법에 비해 상대적으로 적은 계산 비용으로 경쟁력 있는 분류 성능을 보여줍니다. 특히, 10개의 벤치마크 데이터셋에서 특성이 10개 미만인 경우에도 뛰어난 성능을 발휘하며, 이를 통해 전이 학습의 가능성을 더욱 확장합니다.



### A General Framework for Inference-time Scaling and Steering of Diffusion Models (https://arxiv.org/abs/2501.06848)
- **What's New**: 이번 연구에서는 Feynman Kac (FK) steering을 제안합니다. 이는 사용자 지정 특성을 가진 샘플을 생성하는 데 어려움을 극복하기 위한 인퍼런스 타임 프레임워크입니다. FK steering은 여러 개의 상호작용하는 확산 프로세스인 입자(particles)를 샘플링하고 희소 이벤트 시뮬레이션 방법을 통해 높은 보상을 제공하는 샘플을 생성할 수 있도록 합니다.

- **Technical Details**: FK steering은 보상 함수에 따라 입자를 재샘플링하는 메커니즘을 가지고 있습니다. 이 과정에서, 잠재적 함수(potentials)는 중간 상태에 대한 보상을 기반으로 정의되며, 높은 값을 가진 입자는 더 높은 보상 샘플을 생성할 가능성이 높습니다. 이 프레임워크를 통해 사용자들은 다양한 잠재적 함수와 보상 모델을 선택해 성능을 조정할 수 있습니다.

- **Performance Highlights**: FK steering은 텍스트-이미지 및 텍스트 확산 모델에서 높은 성과를 보여주었습니다. 특히, 0.8B 파라미터 모델이 2.6B 파라미터로 파인 튜닝된 모델보다 프롬프트 충실도에서 우수한 성과를 보였으며, 샘플링 속도는 더 빠르고 추가 교육이 필요하지 않았습니다. FK steering은 더 작은 모델도 더 큰 모델보다 성능이 뛰어난 성과를 보여주며, 품질 제어 및 속성 조정의 용이성을 제공합니다.



### SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training (https://arxiv.org/abs/2501.06842)
- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs) 훈련에서 발생하는 그래디언트 스파이크(gradient spikes)의 문제를 심층적으로 분석합니다. 이러한 스파이크는 훈련을 방해하고 비효율성을 가중시키는 주요 원인으로 지적됩니다. 이를 해결하기 위해 'Spike-Aware Adam with Momentum Reset SPAM'이라는 새로운 최적화 알고리즘을 제안하여 그래디언트 스파이크를 효과적으로 감소시켜 훈련 안정성을 향상시킵니다.

- **Technical Details**: 저자들은 온라인 볼록 최적화 문제를 고려하며, 각 시간 단계에서 결정 변수를 선택하고 볼록 손실 함수가 주어지는 상황을 다룹니다. 각 그래디언트는 유한한 경계를 가지고 있으나, 그래디언트 스파이크는 일반적인 그래디언트 노름보다 상당히 클 수 있습니다. 점진적으로 개선된 AMSGrad 변형을 통해 훈련 안정성과 수렴성을 보장하려고 합니다.

- **Performance Highlights**: 실험 결과, SPAM은 Adam 및 그 변종에 비해 다양한 작업에서 일관되게 우수한 성능을 보였습니다. 60M에서 1B까지의 LLM Pre-training, 강화 학습, 시계열 예측 등을 포함하며, 메모리 제약 환경에서도 상태-of-아트 메모리 효율 최적화기보다 성능이 뛰어남을 입증합니다. 특히 희소 모멘텀을 통해 메모리 효율적인 훈련을 가능하게 하여 리소스 효율성을 높이는데 기여합니다.



### LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents (https://arxiv.org/abs/2501.06834)
- **What's New**: 이 연구는 기존 WEIRD(서구, 교육받고, 산업화되고, 부유하고, 민주적인) 집단이 아닌 다양한 집단의 경제 행동을 연구하는 데 새로운 접근법을 제시합니다. 이 방법론은 대형 언어모델(LLM)을 활용하여 Synthetic Cultural Agents(SCAs)를 생성함으로써 다양한 문화 집단의 행동을 시뮬레이션합니다. 이러한 SCAs는 전통적인 경제 실험에 참여하여 문화간 행동 변화를 분석할 수 있도록 돕습니다.

- **Technical Details**: 방법론의 핵심 기초는 맞춤화와 재현 가능성에 중점을 두고 있습니다. SCAs는 소규모 사회의 문화적 특성을 반영한 프로필에 따라 행동하도록 훈련됩니다. 연구진은 직접 프롬프트, 셀프 질의 및 검색+회수 증강 생성(RAG) 방법을 통해 SCAs의 문화 프로필을 구축하였으며, 이 과정을 통해 경제 결정에 대한 실험을 수행했습니다.

- **Performance Highlights**: 연구 결과, SCAs의 행동은 전통적인 경제 실험에서 실존 인구의 행동 및 패턴과 질적으로 유사함을 보여주었고, 문화 간 행동의 다양성을 나타냈습니다. SCAs는 이전에 연구되지 않은 집단에서도 경제 행동에 대한 새로운 가설 생성을 가능하게 하며, 실험 경제학 및 문화 인류학 분야에서 AI를 통합하는 새로운 도구로 기능할 잠재력을 증명합니다.



### Correcting Annotator Bias in Training Data: Population-Aligned Instance Replication (PAIR) (https://arxiv.org/abs/2501.06826)
- **What's New**: 이번 연구에서는 인구 대표성이 부족한 라벨링(pool)로 훈련된 모델이 널리 퍼진 인구의 견해를 반영하지 못할 수 있음을 강조하고, 이를 해결하기 위한 새로운 방법인 Population-Aligned Instance Replication (PAIR)을 제안합니다. PAIR는 부족한 인구 집단에서 라벨을 복제하여 데이터의 비율을 맞추어 모델의 편향을 줄이는 통계적 조정을 통해 가능합니다. 이 방법은 새로운 데이터 수집 없이 모델을 더 공정하게 만들 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 공격적인 언어와 증오 발언 검출에 대한 시뮬레이션 연구를 통해 두 가지 유형의 주석자 집단을 생성하였습니다. 각 주석자 그룹은 서로 다른 라벨링 경향을 가지고 있어, 다양한 인구 비율 비율로 데이터를 생성하였습니다. 모델 성능에 대한 평가에서, 비대표적인 주석자 풀에서 훈련된 모델은 신뢰도(calibration)와 성능(performance) 모두 저조한 것으로 나타났습니다.

- **Performance Highlights**: 훈련 데이터의 불균형을 보완하기 위한 PAIR의 효과는 크게 나타났습니다. PAIR를 적용한 결과, 모델의 편향이 유의미하게 줄어들어 대표성 있는 훈련 데이터를 갖지 않더라도 더 공정한 성과를 이끌어낸 것으로 평가되었습니다. 마지막으로, 연구진은 훈련 데이터 품질을 개선하기 위한 세 가지 실용적인 제안을 통해 AI 시스템이 보다 대표적일 수 있도록 방향성을 제시했습니다.



### Improving Cross-Lingual Phonetic Representation of Low-Resource Languages Through Language Similarity Analysis (https://arxiv.org/abs/2501.06810)
Comments:
          10 pages, 5 figures, accepted to ICASSP 2025

- **What's New**: 이 논문은 언어간 유사성이 저자원 언어의 음성 처리에서 어떻게 교차 언어 음성 표현에 영향을 미치는지를 조사합니다. 특히, 효과적인 소스 언어 선택에 중점을 두고 있습니다. 기존의 교차 언어 연구는 단순히 여러 소스 언어를 활용하여 새로운 저자원 언어의 성능을 개선했으나, 이 연구는 언어 선택에 대한 심층 분석을 제공합니다.

- **Technical Details**: 연구에서는 언어 유사성을 평가하기 위해 두 가지 접근법, 즉 Corpus-based 유사성 평가와 Typology-based 유사성 평가를 통합적으로 사용했습니다. 특히, Corpus-based 접근법을 통해 IPA로 변환한 음소의 분포 빈도를 분석하며, 각 음소의 분산을 기반으로 언어 쌍 간 유사성을 측정합니다. 이 과정에서 Cosine Similarity와 주성분 분석(PCA)을 사용하여 언어의 관계를 시각화하였습니다.

- **Performance Highlights**: 저자원 음성 인식 작업에서, 음운학적으로 유사한 언어를 사용하는 경우 모놀링궐 훈련에 비해 55.6%의 상대적인 성능 향상을 보여주었습니다. 다국어 학습에서 같은 언어 가족 내의 언어가 더 높은 음운학적 유사성을 가질 때 성능이 개선되며, 음운학적 유사성이 낮은 경우에는 성능 저하가 발생했습니다. 이러한 결과는 저자원 환경에서도 효과적인 교차 언어 접근 방식을 구현하는 데 매우 중요한 정보를 제공합니다.



### 3DCoMPaT200: Language-Grounded Compositional Understanding of Parts and Materials of 3D Shapes (https://arxiv.org/abs/2501.06785)
- **What's New**: 이 논문에서는 3D 객체 부품 및 재료에 대한 구성을 이해하기 위해 200개의 객체 범주를 갖춘 대규모 데이터세트인 3DCoMPaT200을 소개합니다. 3DCoMPaT200은 기존의 3DCoMPaT보다 약 5배 더 큰 객체 어휘와 약 4배 더 많은 부품 범주를 포함하고 있으며, 총 1,031개의 세부 부품 범주와 293개의 독특한 재료 클래스를 제공합니다. 이는 복잡한 3D 형태를 구성적 관점에서 이해하는 데 필요한 데이터셋의 한계를 크게 극복합니다.

- **Technical Details**: Part-level 3D 객체 이해는 기계 인식의 기본적인 요소로, 시스템이 객체의 구성 요소 및 재료를 해석하고 상호작용할 수 있게 해줍니다. 3DCoMPaT200은 19,000개의 3D 형태를 포함하며, 세부적인 자료 수집 및 렌더링 파이프라인을 통해 데이터 예제를 제공합니다. 또한, ULIP를 이용한 다중 모드 정렬 실험을 통해 데이터셋의 중요성을 측정하고자 했습니다.

- **Performance Highlights**: 연구 결과, 모델의 성능은 구성 스타일의 수가 증가함에 따라 향상되며, 3DCoMPaT200 데이터셋이 복잡한 3D 형상을 이해하는 데 있어 중요한 역할을 한다는 것을 보여줍니다. 본 데이터셋을 이용한 객체 분류, 부품-재료 분할 및 Grounded Compositional Recognition (GCR) 기술의 벤치마크 평가를 진행하였으며, 이는 3D 객체 이해 기술 개발을 촉진할 수 있는 잠재력을 지니고 있음을 나타냅니다.



### ZOQO: Zero-Order Quantized Optimization (https://arxiv.org/abs/2501.06736)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문은 자원이 제한된 환경에서 훈련할 수 있도록 설계된 제로 순서 양자화 최적화(ZOQO) 방법을 제안합니다. 기존의 고정밀도 그래디언트 계산 없이 양자화된 파라미터와 연산에 대해 모델을 훈련할 수 있습니다. ZOQO는 제로 순서 근사를 활용하여 파라미터의 양자화를 유지하면서 학습 과정을 조정합니다. 이 접근 방식은 성능 상의 제약이 있음에도 불구하고 경쟁력 있는 결과를 달성합니다.

- **Technical Details**: ZOQO는 Zero-Sign Stochastic Gradient Descent (ZO-SignSGD) 방법을 사용하여 기초적인 최적화 절차를 수행합니다. 이 방식에서 우리는 파라미터에 양자화된 노이즈를 삽입하고, 학습률을 양자화 규모에 맞추어 조정합니다. 모든 업데이트 과정에서 양자화된 포맷으로 진행되어 메모리 사용을 최소화합니다. 논문에서는 적대적 공격과 대형 언어 모델의 미세 조정을 통해 기법의 유효성을 검증하였습니다.

- **Performance Highlights**: ZOQO는 양자화된 환경에서 훈련된 모델이 고정밀도 공격에 비해 실패율이 미미하게 감소한다는 것을 입증했습니다. 또한, 리소스 제한 조건에서 감정 분석을 위한 대형 언어 모델을 미세 조정하는 데 있어 비약적인 성능을 발휘할 수 있음을 demonstrated하였습니다. 이러한 결과는 ZOQO의 가능성을 강조하며 자원이 제한된 환경에서 모델 훈련 메커니즘을 가능하게 하는 데 기여할 수 있습니다.



### Fine-tuning ChatGPT for Automatic Scoring of Written Scientific Explanations in Chines (https://arxiv.org/abs/2501.06704)
- **What's New**: 이 연구는 중학생들이 작성한 중국어 과학 설명 텍스트의 자동 점수를 매기기 위해 ChatGPT와 같은 대규모 언어 모델(LLM)을 조정하는 방법을 탐구합니다. 이 연구는 LLM이 글쓰기의 복잡성에 따라 점수 정확도가 어떻게 달라지는지를 살펴보았으며, 특히 중국어와 같은 로그그래픽 언어에서의 가능성을 주목했습니다. 결과적으로, ChatGPT가 중국어 과학 설명을 점수화할 수 있는 능력이 있음을 보여줍니다.

- **Technical Details**: 자동 점수화는 증거 기반의 주장 구축과 의사소통을 포함하는 과학적 설명에 있어서 중요한 기술로, 이를 위해 기계학습(machine learning)과 자연어 처리(natural language processing) 기술이 활용됩니다. 이 연구에서는 7626개의 중학생의 답변을 수집하여 ChatGPT를 조정하였으며, 점수 정확도는 학생의 이해 수준에 따라 다르게 나타났습니다. 이는 글의 언어적 특징에 따라 반영되는 점수화의 복잡성을 시사합니다.

- **Performance Highlights**: 연구 결과, ChatGPT는 중국어 과학 설명 점수화에 있어 높은 정확도를 보였으나, 학생의 글쓰기 복잡성과 점수 정확도 간에 상반된 상관관계가 나타났습니다. 저급 반응에서는 복잡한 문장이 과대 평가되는 경향이 있었고, 고급 반응에서는 간결한 인과관계 서술이 과소 평가되는 양상이 관찰되었습니다. 이러한 사항은 글쓰기의 단순성과 명료성이 저급 반응에서는 정확성을 높이는 반면, 포괄성이 고급 반응에서는 정확성을 높이는 데 기여한다는 것을 보여줍니다.



### Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization (https://arxiv.org/abs/2501.06663)
- **What's New**: 이 논문은 저자들이 저메모리 및 저전력 조건 하에 테서(tensor) 압축을 활용하여 FPGA에서 엔드 투 엔드(transformer training)를 수행하는 첫 번째 가속기를 개발했다고 보고합니다. 특히, 디지털 엣지 기기에서 딥러닝 모델 훈련의 중요성이 증가함에 따라, 새로운 알고리즘적 방법과 하드웨어 설계를 통해 인프라스트럭처의 효율성을 극대화하였습니다.

- **Technical Details**: 본 논문에서 소개된 새로운 알고리즘은 양방향 텐서 수축 흐름(bidirectional tensor contraction flow)을 도입하여, 기존의 텐서 연산보다 연산 효율성을 크게 향상시키고, 메모리 비용을 절감하는 데 기여합니다. 또한 각 훈련 단계에서 모든 압축된 모델 매개변수와 기울기(gradient) 정보를 칩 내 저장하는 메모리 관리 전략을 개발하여, Off-chip 통신을 최소화함으로써 지연(latency)과 에너지 비용을 줄였습니다.

- **Performance Highlights**: 실험에서는 36.7MB에서 93.5MB 크기의 transformer 모델이 사용되었으며, AMD Alveo U50 FPGA에서 단일 배치의 온디바이스 훈련이 가능함을 입증하였습니다. 특히, 기존 NVIDIA RTX 3090 GPU에서의 훈련과 비교하여 메모리 소모가 30배에서 51배 줄어들었으며, 에너지 비용 또한 최대 3.6배 감소한 결과를 나타내어 저자들이 제안한 FPGA 가속기의 장점을 실증적으로 보여주었습니다.



### The Magnitude of Categories of Texts Enriched by Language Models (https://arxiv.org/abs/2501.06662)
- **What's New**: 이 논문은 두 가지 주요 목적을 가지고 있습니다. 첫째, 언어 모델이 제공하는 다음 토큰의 확률을 사용하여 자연어의 텍스트 범주를 명시적으로 정의하는 $[0,1]$-enrichment 접근 방식을 제시합니다. 둘째, 최근 Vigneaux에 의해 도입된 조합적 버전을 사용하여 텍스트의 관련 일반화된 기하학적 공간 $	extmath{M}$의 Möbius 함수와 크기를 계산합니다.

- **Technical Details**: $	extmath{M}$의 크기 함수 $f(t)$는 다음 토큰 확률 분포 $p(-|x)$의 Tsallis $t$-엔트로피의 합과 모델의 가능한 출력의 기수(cardinality)를 더한 것으로 정의됩니다. 이 함수의 $t=1$에서의 미분은 Shannon 엔트로피의 합을 회복하며, 이를 통해 크기가 분할 함수(partition function)로 해석될 수 있음을 정당화합니다. 우리는 또한 magnitude homology의 오일러 특징(Euler characteristic)을 사용하여 $	extmath{M}$의 크기 함수에 대해 표현합니다.

- **Performance Highlights**: 본 연구에서는 $	extmath{M}$의 영차(zeroth) 및 제일 차(first) 크기 호몰로지 그룹의 명시적 설명을 제공합니다. 이는 텍스트 생성의 종료 조건을 다루는 동시에, 텍스트 간의 관계를 확률적으로 해석하는 방법을 고려합니다. 이러한 접근 방식은 자연어 처리(NLP) 분야에 새로운 가능성을 제시하며, 텍스트의 구조적 이해를 보다 깊이 있게 합니다.



### EmoXpt: Analyzing Emotional Variances in Human Comments and LLM-Generated Responses (https://arxiv.org/abs/2501.06597)
Comments:
          7 pages, 10 figures, 5 tables. This paper has been accepted and presented at the 2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC)

- **What's New**: 이 연구는 생성적 AI, 특히 ChatGPT와 관련된 감정 동태를 검토하고, EmoXpt라는 새로운 감정 분석 프레임워크를 소개합니다. 이 프레임워크는 인간의 감정과 ChatGPT의 반응에서 감정 표현을 정량적으로 평가하여, 과거 연구와는 달리 AI의 감정 지능을 분석합니다. 실험 결과, LLM(large language model) 생성 응답은 인간 반응보다 일관되게 긍정적이고 효율적이라는 점을 강조합니다.

- **Technical Details**: EmoXpt는 감정 분석을 위한 네 가지 주요 단계, 즉 데이터 수집, 탐색적 데이터 분석(Exploratory Data Analysis, EDA), 데이터 전처리 및 데이터 모델링을 수행합니다. 데이터는 2023년 3월 7일부터 4월 29일까지 X(구 Twitter)에서 수집된 512개의 트윗과 대응하는 ChatGPT의 반응 및 사용자 코멘트를 포함합니다. 연구에서 BERT와 K-means 알고리즘을 사용하여 감정 분석을 진행하며, 이는 높은 품질의 자연어 처리(NLP) 응용 프로그램을 지원합니다.

- **Performance Highlights**: 결과 분석에 따르면, ChatGPT의 응답은 자주 사용되는 단어와 사용자의 코멘트 분석을 바탕으로 매우 높은 응답 품질을 유지하고 있습니다. 특히, LLM 생성 응답은 일반적으로 더 응집성 있고 긍정적인 감정을 나타내며, 이는 생성적 AI가 인간의 의사소통을 지원하는 데 핵심적인 역할을 한다는 것을 시사합니다. 이러한 발견은 감정 지능을 개선한 대화형 에이전트의 개발에 기여할 것으로 기대됩니다.



### Ladder-residual: parallelism-aware architecture for accelerating large model inference with communication overlapping (https://arxiv.org/abs/2501.06589)
- **What's New**: 이번 논문에서는 대형 언어 모델의 추론에서 메모리 및 시간이 많이 소모되는 문제를 해결하기 위해 Ladder Residual이라는 간단한 아키텍처 수정을 제안합니다. 이는 모든 잔여 기반 모델에 적용 가능하며, 통신의 지연을 효과적으로 숨길 수 있도록 돕습니다. 이 방법은 기존의 모델 아키텍처를 변경하여 계산과 통신을 분리할 수 있는 가능성을 열어줍니다. 결과적으로, 70B 파라미터의 Transformer 모델에 Ladder Residual을 적용하면 8개의 장치에서 30%의 속도 향상을 이룰 수 있음을 보여줍니다.

- **Technical Details**: Ladder Residual은 Transformer 아키텍처에서 모델 레이어 간의 잔여 스트림을 재조정하여 통신과 계산을 분리합니다. 이 분리는 다양한 장치 간의 데이터 교환 중 발생하는 지연을 최소화하며, 이를 통해 추론 과정의 속도가 개선됩니다. 이 방법은 Tensor Parallelism에서 특히 효과적이며, 논문에서는 이를 중심으로 두 가지 모델(1B 및 3B Ladder Transformer)을 훈련시켰습니다. Ladder Residual은 고급 기계 학습 프레임워크인 PyTorch나 JAX에서도 쉽게 적용할 수 있습니다.

- **Performance Highlights**: Ladder Residual을 적용한 70B Transformer 모델은 8개 장치에서 추론 시 속도가 약 30% 향상되었습니다. 이를 통해 잔여 기반 모델의 성능을 유지하면서 통신 지연을 숨기는 데 기여할 수 있음을 보여줍니다. 훈련될 때도 다른 형태의 병렬 처리에 대한 속도 향상이 이루어지며, 실험 결과 기존의 변환 모델의 성능과 비슷한 결과를 보여줍니다. 이 모델은 Transformer와 같은 대형 언어 모델에 널리 활용될 수 있을 것입니다.



### Speech Recognition for Automatically Assessing Afrikaans and isiXhosa Preschool Oral Narratives (https://arxiv.org/abs/2501.06478)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 본 연구는 아프리카어(Afrikaans) 및 이시콕사어(isiXhosa)를 사용하는 유아의 자연어 이야기를 자동으로 인식하는 시스템을 개발하는 것을 목표로 하고 있습니다. 이러한 시스템은 아프리카에서 유아의 언어 발달을 평가할 수 있는 도구가 될 것이며, 특히 4세와 5세 아동에게 초점을 맞춘 연구는 드물기 때문에 독창적인 기여를 하고 있습니다. 연구에서는 Whisper를 기반으로 하여, 최소 5분의 전사된 아동 연설 데이터를 활용하며 다양한 ASR 전략을 비교 평가합니다.

- **Technical Details**: 연구에서 사용된 기본 ASR 모델인 Whisper는 97개 언어에서 680,000시간의 약한 감독 데이터로 훈련되었습니다. 이 모델은 아프리카어에 대해서는 4.1시간의 훈련 데이터를 포함하고 있으나, 이시콕사어 데이터는 전혀 포함되어 있지 않아서 적절한 설정이 필요합니다. 본 연구는 문장 사용을 통해 성과를 개선하는 여러 전략을 실험하며, 특히 성인 데이터를 활용하여 아동 데이터를 보완하는 중요성을 강조합니다.

- **Performance Highlights**: ASR 시스템의 성능을 평가한 결과, 아프리카어에 대한 단어 오류율(Word Error Rate, WER)은 47.4%로 상대적으로 저조한 이시콕사어의 80.4%와 대조적입니다. 이 결과는 Whisper의 훈련 데이터와의 언어 간 불일치와 같은 요소들로 설명될 수 있습니다. 따라서 아프리카어에서의 훈련을 통해 이시콕사어의 성능을 향상시킬 수 있는 추가적인 연구가 필요합니다.



### Using Pre-trained LLMs for Multivariate Time Series Forecasting (https://arxiv.org/abs/2501.06386)
- **What's New**: 이 논문에서는 사전 훈련된 대규모 언어 모델(LLM)을 활용하여 다변량 수요 시계열 예측에 접근하는 새로운 방법론을 제안합니다. 특히, 다변량 패칭(multivariate patching) 전략을 통해 시계열 특성을 디코더 전용으로 사전 훈련된 트랜스포머에 임베딩하는 방식으로, 최신 시계열 예측 모델과 경쟁할 수 있는 결과를 보여줍니다. 이 혁신적인 접근은 다변량 데이터에 대한 예측 정확도를 개선할 수 있는 가능성을 발견하게 합니다.

- **Technical Details**: 주요 기술적 기법으로는 다변량 시계열을 LLM의 토큰 임베딩 공간으로 매핑하는 방법과, 그 후에 다시 원래의 시계열 공간으로 매핑하는 역방향 방법이 포함됩니다. 또한, 층 정규화(layer norms)만 미세 조정하여 훈련 가능 파라미터의 수를 획기적으로 줄입니다. 논문에서는 두 가지 접근 방식을 통해 LLM 기반 시계열 예측의 실증 분석을 제공합니다.

- **Performance Highlights**: 모델의 성능 검증을 위해 MQCNN 모델을 베이스라인으로 설정하고, 사전 훈련된 LLM에서 소수의 파라미터만 조정하여도 전문 엔지니어링 아키텍처와 유사한 성능에 도달할 수 있음을 보여줍니다. 훈련 데이터 접근 없이도 모델 진단을 수행할 수 있는 중량 기반 진단 기법을 사용하여 예측 정확도를 평가합니다. Layer-specific weight analysis를 통해 품질과 예측 정확도 간의 관계를 분석하였음을 밝혔습니다.



### TTS-Transducer: End-to-End Speech Synthesis with Neural Transducer (https://arxiv.org/abs/2501.06320)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 논문에서는 TTS-Transducer라는 새로운 텍스트-음성 변환(text-to-speech, TTS) 아키텍처를 소개합니다. 이 아키텍처는 오디오 코덱 모델(audio codec models)과 신경 전이기(neural transducers)의 강점을 활용하여, 몬토닉 정렬(monotonic alignments)을 배워 명시적인 지속 시간 예측기(duration predictor)를 피할 수 있도록 설계되었습니다. TTS-Transducer는 여러 상태의 TTS 시스템에 비해 경쟁력 있고 견고한 대안임을 입증합니다.

- **Technical Details**: 본 시스템은 먼저 전이기 아키텍처를 사용해 첫 번째 코드북에 대해 문자화된 텍스트와 음성 코덱 토큰(tokenized text and speech codec tokens) 간의 몬토닉 정렬을 학습합니다. 이후 비자기회귀(Non-autoregressive, NAR) Transformer가 전이기 손실(transducer loss)에서 추출된 정렬 정보를 바탕으로 나머지 코드를 예측합니다. 이 시스템은 엔드-투-엔드(end-to-end) 방식으로 훈련되어 복잡한 음성 합성 절차를 간소화합니다.

- **Performance Highlights**: TTS-Transducer는 도전적인 텍스트에서 3.94%의 문자 오류율(character error rate, CER)을 달성하여, 훨씬 더 많은 데이터로 훈련된 기존 TTS 모델들을 초월하는 성능을 보입니다. 이 모델은 다양한 잔여 벡터 양자화(residual vector quantization) 코덱에서 일반화 능력이 뛰어나며, 큰 데이터로 사전 훈련을 하지 않고도 최신 TTS 모델과 비교할 수 있는 제로샷(zero-shot) 결과를 생성합니다.



### Understanding How Paper Writers Use AI-Generated Captions in Figure Caption Writing (https://arxiv.org/abs/2501.06317)
Comments:
          This paper will appear at AAAI 2025 Workshop (2nd AI4Research Workshop: Towards a Knowledge-grounded Scientific Research Lifecycle)

- **What's New**: 이 논문은 저자들이 AI가 생성한 캡션을 실제 논문 작성 과정에 통합하는 방법을 조사했습니다. 연구는 18명의 참가자가 그들의 최근 연구 작업에서 가져온 그림 두 개에 대한 캡션을 수정하는 방식으로 진행되었습니다. 참가자들은 최신 AI 모델이 생성한 캡션을 리소스로 사용하여 이 과정을 수행했으며, 이를 통해 AI가 지원하는 캡션 작성을 위한 새로운 기회를 찾아냈습니다.

- **Technical Details**: 연구에서 사용된 AI 모델은 최신의 figure captioning 모델로서, 참가자들은 이 모델이 생성한 캡션을 자유롭게 사용하여 자신의 작업에 통합했습니다. 캡션 작성 과정은 비디오로 기록되었으며, 이 기록은 인터랙션 분석을 통해 참가자들의 글쓰기 행동을 수집하는 데 활용되었습니다. 연구 결과, 참가자들은 AI가 생성한 캡션을 대부분 복사하여 수정하는 경향을 보였고, 특히 통계적 그림에 더 효과적이라는 점이 드러났습니다.

- **Performance Highlights**: 결과는 캡션 작성 과정이 단순한 문장 작성 이상의 복잡함과 다양성을 가지고 있음을 강조합니다. 현재 AI 모델은 복잡한 그림에 대한 캡션 작성 시 효과적이지 않지만, 참가자들은 AIGenerated 캡션에서 세밀한 디테일을 통합하려고 했습니다. 이 연구는 AI 시스템이 학술 작성을 지원하기 위한 기회가 있음을 제시하며, 향후 캡션 작성 도구의 설계에서 고려해야 할 요소들을 드러냅니다.



### Dafny as Verification-Aware Intermediate Language for Code Generation (https://arxiv.org/abs/2501.06283)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 자연어 프롬프트로부터 소스 코드를 생성하는 과정에서의 한계를 극복하기 위해, 검증 가능한 언어인 Dafny를 중간 표현으로 활용하는 방안을 제안합니다. 코드 생성 및 검증 과정을 자동화하고, 사용자와의 상호작용은 자연어로 진행되어 Dafny 코드가 사용자에게 노출되지 않도록 합니다. 이를 통해 생성된 코드의 품질을 높이고, 일반 프로그래밍 언어로 변환하기 전의 오류를 미리 잡을 수 있게 됩니다.

- **Technical Details**: 프로토타입 챗봇은 Claude Sonnet 3.5를 기반으로 하고, Dafny를 사용하여 파이썬 코드의 정확성을 높입니다. 사용자의 자연어 입력에 의해 프롬프트가 Dafny로 형식화되고, 생성된 솔루션과 증명이 Dafny 검증기를 통해 검증됩니다. 검증 및 수정 과정은 자동으로 이루어지며, 최종적으로 솔루션은 다수의 단위 테스트와 함께 목표 언어로 변환되어 사용자에게 제공됩니다.

- **Performance Highlights**: 초기 벤치마크 테스트 결과, Dafny를 활용한 코드 생성 방식이 LLM이 단순히 코드를 생성하는 것보다 더 정확하고 신뢰할 수 있는 코드를 생성할 수 있음을 보여줍니다. 프로토타입은 생성된 코드의 실행 안전성과 정확성을 보장하며, 테스트와 검증을 포함한 구현 과정을 통해 코드의 품질을 높이는 데 크게 기여하고 있습니다. 이 방식은 특히 복잡한 알고리즘이나 함수 생성 시 유용성을 발휘합니다.



### PROEMO: Prompt-Driven Text-to-Speech Synthesis Based on Emotion and Intensity Contro (https://arxiv.org/abs/2501.06276)
- **What's New**: 본 논문에서는 감정 표현과 강도를 조절할 수 있는 prompt 기반의 접근 방식을 통해 TTS 모델의 감정 조절 능력을 향상시키는 새로운 방법을 제안합니다. FastSpeech 2(또는 FS2) 아키텍처를 확장하여 다중 화자에 대한 감정 제어를 가능하게 하여 이는 현대 TTS 시스템의 한계를 극복하는 데 중점을 두고 있습니다. 또한, 대규모 언어 모델(LLMs)을 활용하여 발화의 억양을 조작하고 언어적 콘텐츠를 유지하면서 말의 유창함을 높이고자 하는 시도를 하고 있습니다.

- **Technical Details**: 제안된 접근 방식은 음성 합성을 위한 FS2 아키텍처의 변경을 포함하며, 감정과 강도 인코더를 결합하여 말의 표현력을 높입니다. LLM을 사용하여 목표 감정에 맞춘 억양을 예측하고, 발화와 단어 레벨 모두에서 높이, 에너지 및 지속 시간을 조절합니다. 이러한 방법을 통해, 다양한 감정 강도를 지닌 다중 화자의 감정 표현이 가능한 음성을 생성할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 감정의 변화를 반영하는 음성을 생성하여 인공지능 음성 합성 기술의 새로운 방향성을 제시합니다. 감정의 미세한 변화를 조절할 수 있으며, 이는 기존 기술보다 더 자연스럽고 표현력이 풍부한 음성을 생성합니다. 또한, 실험을 통해 이 접근 방식이 고객 응대, 오디오북 내레이션 및 가상 비서와 같은 다양한 응용 분야에서의 유용성을 보여줍니다.



### Polarized Patterns of Language Toxicity and Sentiment of Debunking Posts on Social Media (https://arxiv.org/abs/2501.06274)
- **What's New**: 최근 온라인 정치 담론에서 잘못된 정보 및 가짜 뉴스의 확산이 민주적 프로세스와 공공 참여에 심각한 문제를 야기하고 있습니다. 이 연구는 소셜 미디어 플랫폼에서의 언어 독성(language toxicity), 비관주의(pessimism), 그리고 사회적 분극화(social polarization) 간의 관계를 분석하였으며, 2016년과 2020년 미국 대통령 선거 및 QAnon 음모론의 담론을 중심으로 진행되었습니다.

- **Technical Details**: 연구 방법으로는 Twitter와 Reddit 두 가지 플랫폼에서의 데이터를 수집하였으며, 2016년부터 2023년까지의 기간을 대상으로 하였습니다. Twitter에서는 'fact check', 'fake news', 'misinformation' 등의 키워드를 사용하여 86.7 백만 개의 트윗을 수집하였고, Reddit에서는 비슷한 기간에 4.7 백만 개의 댓글을 분석하였습니다. 이를 통해 언어 독성과 비관주의의 패턴을 구체적으로 살펴보았습니다.

- **Performance Highlights**: 분석 결과, 언어 독성이 집단 비관주의와 밀접하게 연결되어 있으며, Reddit과 Twitter의 구조적 상호작용이 사용자 참여 패턴에 큰 영향을 미친다는 것을 발견했습니다. 또한, 참여가 증가할수록 언어 독성이 줄어드는 경향을 보였고, Twitter는 단일화된 담론을 촉진하는 반면 Reddit은 다양한 커뮤니케이션을 장려하는 것으로 나타났습니다. 이러한 결과는 정책 입안자와 플랫폼 설계자에게 온라인 담론을 건강하게 만드는 데 중요한 시사점을 제공합니다.



### $\text{Transformer}^2$: Self-adaptive LLMs (https://arxiv.org/abs/2501.06252)
Comments:
          18 panges, 11 figures, 9 tables

- **What's New**: 최근의 연구에서 소개된 $	ext{Transformer}^2$는 자가 적응형 대형 언어 모델(LLMs)의 새로운 프레임워크로, 기존의 훈련 방식보다 적은 자원으로 미지의 작업에 즉각적으로 적응할 수 있는 기술을 제공합니다. 이 방법은 기계 학습에서 흔히 사용되는 LoRA와 같은 기법보다 더 나은 성능을 보여주며, 가벼운 메모리 용량과 높은 효율성을 자랑합니다. $	ext{Transformer}^2$는 여러 LLM 아키텍처 및 모달리티에 걸쳐 다재다능함을 입증하였습니다.

- **Technical Details**: $	ext{Transformer}^2$의 두 가지 주요 메커니즘은 작업 속성을 식별하는 디스패치 시스템과, 강화학습(강화 학습, RL)으로 학습된 작업 특정 '전문가' 벡터들을 동적으로 조합하는 것입니다. 이 프레임워크는 단일 입력에 대한 두 번의 패스를 사용하여 모델의 가중치를 조절하는 방식을 통해 이루어집니다. 이를 위해, Singular Value Fine-tuning (SVF)라는 새로운 기법을 도입하여 모델의 가중치 행렬 내의 특이값만을 조정하여 과적합(overfitting) 문제를 줄이고 자원 소비를 최소화합니다.

- **Performance Highlights**: 대체로 다양한 실험을 통해 SVF와 전체 $	ext{Transformer}^2$ 프레임워크가 기존의 전통적인 효율적 파인튜닝 방법보다 우수하다는 것을 입증하였습니다. 특히, SVF는 적은 수의 매개변수로 토픽별 성능을 최적화하는 효과적인 전문가 벡터를 제공함으로써 비용 절감 효과를 선보였습니다. 또한, $	ext{Transformer}^2$는 시각적 질문 응답(visual question answering)과 같은 새로운 과제에서도 전반적인 성능 향상을 이루어냈으며, 진화 가능성이 있는 자가 적응형 AI 시스템 개발에 기여할 수 있는 기반을 마련하였습니다.



### Fitting Different Interactive Information: Joint Classification of Emotion and Intention (https://arxiv.org/abs/2501.06215)
- **What's New**: 이 논문은 ICASSP MEIJU@2025 트랙 I의 1등 해법으로, 저자원 환경에서의 다중모드 감정 및 의도 인식을 중심으로 하고 있습니다. 특정한 대화에서 다양한 난이도의 작업 증진을 위한 접근법과 막대한 양의 레이블 없는 데이터를 효과적으로 활용하는 방법이 주된 초점입니다. 이를 통해 저자들은 0.5532의 점수로 트랙에서 우승을 차지했습니다.

- **Technical Details**: 연구의 모델 아키텍처는 다중 헤드 자기 주의(multi-head self-attention)와 게이팅 메커니즘을 통한 상호 작용 모듈(combination module)을 기반으로 구성됩니다. 비디오에서 매 30프레임마다 샘플링된 이미지를 사용하고, 오디오 데이터는 ffmpeg를 활용하여 WAV 형식으로 변환합니다. 텍스트 데이터에서 발견된 오류는 수작업으로 수정되어 감정과 의도를 더 잘 분류할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면 감정 인식과 의도 인식은 서로 관련이 있으나, 두 과제를 동시에 최적의 정확도로 수행하기는 어렵다는 점이 발견되었습니다. 다중 헤드 자기 주의 방식을 통해 의도 인식의 성능이 향상되었으며, 두 단계의 훈련 전략을 통해 최종 점수 0.5532를 달성했습니다. 이러한 접근 방식은 작업 성능을 크게 향상시켰습니다.



### Leveraging Edge Intelligence and LLMs to Advance 6G-Enabled Internet of Automated Defense Vehicles (https://arxiv.org/abs/2501.06205)
Comments:
          8 pages, 5 figures, under (secondary/revised) review in IEEE Internet of Things Magazine

- **What's New**: 이 논문은 자율주행 기술의 군사적 응용을 위한 인공지능(AI) 및 딥러닝(DL)의 발전을 다루고 있습니다. 특히, 6G 통신 기술과 사전 훈련된 대규모 생성 언어 모델(LLM)의 통합을 통해 자율방어차량(ADVs)의 의사결정 및 통신 최적화를 강조합니다. 이를 통해 미션 환경에서의 상황 인식 및 실시간 데이터 교환을 극대화할 수 있는 가능성과 도전을 제시합니다.

- **Technical Details**: 군사용 AI는 다양한 방어 기술에 대량의 데이터를 분석하는 능력으로 인해 다양한 응용 분야에서 중요성을 갖습니다. 6G 지원 IoADV 아키텍처는 연계된 군사 차량 간의 실시간 데이터 공유와 자율 호위 작전, 동기화된 기동을 지원합니다. 또한, LLMs는 다양한 데이터 유형을 통합하여 자율주행 차량의 인식 및 의사결정을 개선하는 프레임워크를 제공합니다.

- **Performance Highlights**: ADVs는 정보 수집, 재난 구조, 보안 감시 등의 다양한 사안에서 활용됩니다. 이들은 아군 부대의 안전성을 높이고, 신속한 의사결정과 실시간 처리 능력으로 육상 작전의 효율성을 향상시킵니다. 6G 기반 통신은 초저지연 및 극도의 신뢰성을 제공하며, 이를 통해 자율주행 전략과 전투 작전을 최적화할 수 있습니다.



### A Multimodal Social Agen (https://arxiv.org/abs/2501.06189)
Comments:
          9 pages

- **What's New**: 이 논문은 MuSA라는 멀티모달 대형 언어 모델(LLM) 기반 에이전트를 소개하며, 주어진 인간 중심의 콘텐츠 분석 작업을 수행하기 위한 사회적 콘텐츠 분석을 자동화하고 개선할 수 있는 가능성을 보여줍니다. MuSA는 질문 응답, 시각적 질문 응답, 제목 생성 및 분류와 같은 작업을 다루는 데 맞춰 설계되었습니다. 이 에이전트는 계획, 추론, 실행, 최적화 및 비판 전략을 사용하여 작업을 완료합니다.

- **Technical Details**: MuSA는 모듈화 및 확장성이 뛰어난 구조를 가지고 있습니다. 현재 다섯 개의 독립적인 LLM 기반 유닛인 추론(reason), 계획(plan), 최적화(optimize), 비판(criticize), 세분화(refine) 및 실행(act)을 지원하며, 다양한 작업 요구에 맞춰 결합할 수 있습니다. 또한 MuSA는 체인 오브 사고(chain-of-thought, CoT) 및 자기 반영(self-reflection)과 같은 복잡한 내부 프로세스를 활용하여 성능을 향상시키는 데 최적화되어 있습니다.

- **Performance Highlights**: MuSA는 질문 응답, 제목 생성 및 콘텐츠 분류 작업에서 기존 기준 대비 큰 성과를 보였습니다. 최근 멀티모달 LLM들이 시각적 질문 응답 및 이미지 캡셔닝과 같은 데이터 분석 작업에서 긍정적인 결과를 나타내고 있으며, MuSA도 이러한 멀티모달 기능을 통해 복잡한 질문에 대한 답변을 더욱 효율적으로 제공하고 더 정교한 행동을 이끌어낼 수 있습니다.



New uploads on arXiv(cs.IR)

### TriMod Fusion for Multimodal Named Entity Recognition in Social Media (https://arxiv.org/abs/2501.08267)
Comments:
          Accepted at CASCON

- **What's New**: 이번 논문에서는 멀티모달 소셜 미디어 데이터셋에서 텍스트, 시각적 정보, 해시태그의 통합적 접근법인 TriMod를 제안합니다. 기존의 Named Entity Recognition (NER) 모델들이 소셜 미디어 언어에서 겪는 문제를 해결하기 위해 Transformer-attention 기반의 모달리티 융합을 활용합니다. 이를 통해 다양한 모달리티에서 제공하는 보조적 맥락을 통해 보다 정확한 엔터티 인식을 가능하게 합니다.

- **Technical Details**: TriMod는 텍스트와 비주얼, 해시태그 특징들을 통합하는 혁신적인 아키텍처로, 객체 수준의 비주얼 피처를 포함하여 시각적 객체와 텍스트 내 명명된 엔터티 간의 복잡한 매핑을 수립합니다. Transformer-attention을 활용하여 모달리티 간 분포적 차이를 해결하고 명명된 엔터티와 관련된 이미지 영역 간의 중요한 정렬을 학습하여 성능을 향상시킵니다. 이러한 접근법을 통해 소셜 미디어 시나리오에서 NER의 한계를 극복하고 정확하고 맥락 인식된 인식이 가능합니다.

- **Performance Highlights**: 실험 결과, TriMod 모델은 기존의 최첨단 방법에 비해 F1 점수, 정밀도 및 재현율에서 유의미한 개선을 달성했습니다. 소셜 미디어 데이터셋에서 엔터티 인식 성능이 향상되었으며, visual-textual entity mapping 및 cross-modal feature alignment에서의 효율성이 드러났습니다. 이러한 성과는 멀티모달 접근법이 NER 성능을 향상시킬 수 있음을 뒷받침합니다.



### Unsupervised Query Routing for Retrieval Augmented Generation (https://arxiv.org/abs/2501.07793)
- **What's New**: 이번 논문에서는 기존의 수작업 데이터 세트에 대한 의존성을 줄이고, 효율적인 query routing을 위한 새로운 비지도 학습 방법을 제안합니다. 이 방법은 사용자의 쿼리에 대해 적절한 검색 엔진을 자동으로 식별하는 데 중점을 두고, 단일 출처(single-sourced) 응답과 다중 출처(multi-sourced) 응답을 비교하여 검색 엔진의 품질을 평가합니다. 수작업 주석이 필요 없는 데이터 처리 방식으로, 대규모 사용자 쿼리를 다룰 수 있게 됩니다.

- **Technical Details**: 제안된 방법은 네 단계로 구성됩니다. 첫 번째 단계에서는 사용자가 제출한 쿼리에 대해 단일 검색 엔진에서 응답을 검색합니다. 두 번째 단계에서는 모든 검색 엔진을 통과하여 다중 출처 응답을 생성하며, 이 응답은 비지도 학습을 위한 '상한선(upper-bound)'으로 사용됩니다. 단일 출처 응답의 품질을 평가하기 위해 몇 가지 자동화된 메트릭을 사용하고, 이를 통해 최적의 검색 엔진을 결정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다섯 개의 데이터 세트에서 일관된 일반화 능력과 뛰어난 확장성을 보여주었습니다. 다양한 LLM(large language models)에서 테스트를 진행하였고, 이 과정에서 다수의 검색 엔진을 활용할수록 결과가 개선되는 경향을 보였습니다. 이 연구는 비지도 학습 방식의 효과성을 입증하며, 향후 쿼리 라우팅 연구에 대한 중요한 기초 자료를 제공합니다.



### Constructing Set-Compositional and Negated Representations for First-Stage Ranking (https://arxiv.org/abs/2501.07679)
Comments:
          12 pages

- **What's New**: 이번 연구에서는 논리적 연산이나 부정을 포괄하는 구성 질의 표현을 효과적으로 구성하는 방법을 탐구하고 있습니다. 특히, Learned Sparse Retrieval (LSR) 표현 간의 벡터 연산을 사용하여 제로샷(zero-shot) 방법론으로 구성된 질의 표현을 개발했습니다. 연구팀은 Disentangled Negation이라는 새로운 방법을 소개하여 부정된 질의의 일부만을 페널티(penalize)하도록 설계하였고, Combined Pseudo-Term 접근 방식을 통해 LSR의 집합 교차 처리 능력을 향상시켰습니다.

- **Technical Details**: 제안된 접근법은 LSR 표현의 조합 적합성과 부정을 효과적으로 인코딩하는 데 중점을 두고 있습니다. 부정 처리를 위해 Disentangled Negation을 사용하여 질의의 긍정적인 부분에 영향을 미치지 않으면서 부정된 부분에만 페널티를 부여합니다. 연구 결과, 제로샷 방식은 조합 데이터에 맞춤화된 리트리버보다 경쟁력이 있으며 성능이 더 우수할 때도 있음을 발견했습니다.

- **Performance Highlights**: LSR 모델을 통해 수행된 실험에서 부정된 질의에 대한 페널티를 부여할 수 있는 구조가 효과적이라는 것이 입증되었습니다. 연구팀은 여러 조건에서 LSR과 Dense 리트리버가 모두 부정 사항을 학습할 수 있으나, 교차(intersection) 처리에는 한계를 보임을 밝혔습니다. 이러한 성과는 LSR 기반 모델이 부정 점수를 부여하는 방식으로 성능을 크게 향상 시킬 수 있음을 강조합니다.



### Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models (https://arxiv.org/abs/2501.08248)
- **What's New**: 이 논문에서는 In-Context Retrieval and Reasoning (ICR^2)라는 새로운 벤치마크를 소개합니다. 기존의 LOFT 벤치마크가 LCLMs의 성능을 과대평가할 수 있는 문제를 해결하고자, LCLMs에 대한 보다 실제적인 평가를 제공하기 위해 고안된 것입니다. ICR^2는 강화된 컨파운딩 문서(confounding documents)를 포함하여 LCLMs의 실제적인 상황에서의 성능을 평가합니다.

- **Technical Details**: ICR^2 벤치마크는 Wikipedia에서 수집된 포괄적인 지식 기반을 기반으로 구축되며, 강력한 리트리버를 사용해 문서를 선택합니다. 이를 통해 설명된 세 가지 방법은 (1) retrieve-then-generate fine-tuning, (2) retrieval-attention probing, (3) joint retrieval head training 입니다. 이 방법들은 LCLMs의 in-context retrieval 능력을 증가시키고, 복잡한 multi-stage pipeline의 한계를 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 논문에서 제시된 최적의 접근 방식은 Mistral-7B 모델에 적용되어 LOFT에서 평균 +17과 +15 포인트 향상된 Exact Match를 보여주었으며, ICR^2에서도 +13과 +2 포인트의 개선을 이뤘습니다. 이 방법은 기존의 Vanilla RAG 및 supervised fine-tuning을 초월하여 성능을 발휘하였고, 더 작은 모델임에도 불구하고 대부분의 작업에서 GPT-4-Turbo보다 우수한 성능을 기록했습니다.



### Optimize Incompatible Parameters through Compatibility-aware Knowledge Integration (https://arxiv.org/abs/2501.07596)
Comments:
          Published on AAAI'25: The Annual AAAI Conference on Artificial Intelligence

- **What's New**: 이번 연구에서는 다양한 데이터 분포에서 발생하는 비호환 파라미터를 효과적으로 최적화하기 위한 새로운 방법론인 Compatibility-aware Knowledge Integration (CKI)를 제안합니다. CKI는 두 가지 주요 요소로 구성되어 있으며, Parameter Compatibility Assessment와 Parameter Splicing을 통해 여러 모델의 지식을 통합합니다. 이러한 접근 방식은 기존의 단순한 파라미터 제거나 출력 앙상블 방법의 한계를 극복하고, 더 나은 모델 성능을 유지하면서도 추가적인 파라미터 없이 동작할 수 있습니다.

- **Technical Details**: CKI의 첫 번째 단계인 Parameter Compatibility Assessment는 개별 파라미터의 불확실성을 평가하고, 전체 모델의 정보 콘텐츠를 분석하여 종합적인 파라미터 호환성을 결정합니다. 두 번째 단계인 Parameter Splicing에서는 분석된 호환성을 기반으로 여러 모델의 파라미터를 결합하여 최적화된 모델을 생성합니다. 이 과정에서 하드 스플라이싱과 소프트 스플라이싱을 사용하여 최적의 파라미터를 선택하거나 가중치를 계산하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 다양한 추천 및 자연어 처리 데이터셋에서 실험한 결과, Compatibility-aware Knowledge Integration 방법이 기존 모델의 훈련 제한을 극복하면서도 인퍼런스 비용을 증가시키지 않고 효과적으로 비호환 파라미터를 최적화함을 보여주었습니다. 통합 모델은 추가적인 재학습 없이도 직접 사용 가능하며, 단 한 번의 에포크 재학습만으로도 성능 향상을 이끌어낼 수 있음이 입증되었습니다.



### Multimodal semantic retrieval for product search (https://arxiv.org/abs/2501.07365)
- **What's New**: 이 연구에서는 기존의 텍스트 기반 (textual) 의미 검색 (semantic retrieval)에 대한 연구를 확장하여, 전통적인 텍스트 표현과 대조적으로 전자상거래 (e-commerce) 검색에서 제품 항목에 대한 다중 모달 (multimodal) 표현을 구축했습니다. 이런 새로운 접근법은 제품 이미지가 검색 상호작용에 미치는 영향을 분석하며, 고객 제품 탐색 시 중요하다는 점을 강조합니다.

- **Technical Details**: 이번 연구에서는 전자상거래 데이터셋을 기반으로 모델을 개발하고 평가했습니다. 다중 모달 표현을 통해 제품을 나타내는 새로운 방식을 제안하고, 이 표현이 구매 회수 (purchase recall)와 의미 검색의 관련성 정확도 (relevance accuracy)에서 개선점을 나타내는지 조사합니다.

- **Performance Highlights**: 연구 결과, 다중 모달 의미 검색 모델이 텍스트 전용 모델에 비해 독점적으로 검색된 일치 항목에 대한 수치 분석을 통해 성능 개선을 보여주었습니다. 이는 다중 모달 솔루션의 유효성을 입증하는 데 중요한 정보입니다.



### Dataset-Agnostic Recommender Systems (https://arxiv.org/abs/2501.07294)
- **What's New**: 추천 시스템(Recommendation Systems)은 개인화된 사용자 경험을 제공하는 데 필수적입니다. 그러나 전통적인 추천 시스템은 데이터 세트에 따라 수동으로 조정해야 하기 때문에 확장성과 재사용성이 제한됩니다. 이를 해결하기 위해 제안된 DAReS(Dataset-Agnostic Recommender Systems)는 데이터 세트 특성에 따라 자동으로 조정되는 새로운 패러다임을 제시합니다.

- **Technical Details**: DAReS의 핵심 요소는 데이터 세트 설명 언어(Dataset Description Language, DsDL)입니다. 이는 데이터 세트의 특성과 레이블에 대한 메타데이터를 제공하는 구조화된 포맷을 갖추고 있습니다. DAReS는 이 정보를 활용해 자동으로 특징 선택, 결측값 처리, 잡음 제거 및 하이퍼파라미터 최적화와 같은 프로세스를 관리합니다.

- **Performance Highlights**: DAReS는 높은 적응성 및 재사용성을 제공하며, 다양한 추천 시나리오에 적용할 수 있는 가능성을 내포하고 있습니다. 하지만 계산적 오버헤드와 데이터 세트 특정 최적화의 제한이 있으며, 이러한 한계를 극복하는 것이 중요합니다. DAReS는 사용자가 직접 코드를 수정할 필요 없이 추천 시스템 구축을 자동화하여 사용자 맞춤형 솔루션을 제공합니다.



### Future-Conditioned Recommendations with Multi-Objective Controllable Decision Transformer (https://arxiv.org/abs/2501.07212)
- **What's New**: 이번 연구에서는 추천 시스템의 장기적인 성공을 확보하기 위한 새로운 접근 방식을 제안합니다. 우리는 사용자 만족도를 예측하고 형성하는 전략이 필요하다는 점에 주목하였습니다. 새로운 다목적 제어 가능한 추천 시스템인 Multi-Objective Controllable Decision Transformer(MocDT)를 통해 한 번의 모델 트레이닝으로 여러 목표에 대해 아이템 시퀀스를 생성할 수 있도록 하였습니다. 이를 통해 복잡한 추천 목표를 동시에 관리할 수 있는 가능성을 열었습니다.

- **Technical Details**: MocDT는 미래 목표를 명시적으로 설정하고, 이에 따라 아이템 시퀀스를 생성하는 다목적 제어 신호 메커니즘을 활용합니다. 이는 기존의 결정 변환기(Decision Transformer, DT)를 기반으로 하며, 각 목표는 필드 오브 스터디에 따라 우선순위를 두고 조정됩니다. 또한, 데이터셋의 품질을 높이기 위해 세 가지 데이터 증강 전략을 개발하여 다양한 상호작용 패턴을 생성하고, 모델의 일반화 능력을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, MocDT는 다양한 목표에 부합하는 아이템 시퀀스를 생성하는 데 탁월한 성능을 보였습니다. 이 모델은 사용자 맞춤화된 추천 경험을 제공할 수 있는 잠재력을 내포하고 있으며, 기존 추천 시스템에 비해 다양한 목표 간의 조화를 이루는 데 성공하였습니다. 이러한 결과는 추천 시스템의 제어 가능성을 크게 향상시키고, 더 나아가 사용자 중심의 추천 경험을 가능하게 합니다.



### Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation (https://arxiv.org/abs/2501.07096)
Comments:
          14 pages, 6 figures, 4 tables

- **What's New**: 이 논문에서는 사용자 행동을 동적으로 변하는 의도(intent)와 안정적 취향(interest)으로 분리하여 좀 더 정교하게 사용자의 동기를 이해하는 Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation (IDCLRec)을 제안합니다. IDCLRec는 사용자 간의 일관된 취향을 파악하고 잔여 행동을 의도로 모델링할 수 있도록 설계되었습니다. 이를 통해 논문은 사용자 간의 상호작용 시 발생하는 의도의 다양성을 포착하고, 사전 정의된 의도 범주 없이 사용자의 독립적인 의도를 학습합니다.

- **Technical Details**: IDCLRec는 인과적 교차 주의(causal cross-attention) 메커니즘을 활용하여 사용자 행동에서 일관된 관심을 식별하며, 잔여 행동은 시간적 동역학을 반영한 유사성 조정 손실(similarity adjustment loss)을 통해 의도로 모델링됩니다. 또한, 중요한 의도의 중요성을 각 상호작용의 중요성을 고려하여 가중치를 부여하는 주의 메커니즘을 채택합니다. 마지막으로, 아이템에 대한 대비 학습(item-aware contrastive learning)을 도입하여 동일한 의도가 발생한 경우와 해당 의도에 의해 발생한 아이템 조합 간의 의도 균형을 맞춥니다.

- **Performance Highlights**: IDCLRec는 실제 데이터셋을 활용한 실험에서 기존 모델들보다 현저히 우수한 성능을 보였습니다. 특히, 사용자의 개별화된 요구와 동기를 반영하여 맞춤형 추천을 향상시켰습니다. 이러한 결과는 의도-관심 분리와 사용자 특정 의도 학습을 통해 더 개인화되고 정확한 추천을 가능하게 함을 보여줍니다.



### Research on the Online Update Method for Retrieval-Augmented Generation (RAG) Model with Incremental Learning (https://arxiv.org/abs/2501.07063)
- **What's New**: 이 논문에서는 정보 기술의 급속한 발전과 데이터 양의 기하급수적 증가에 대응하기 위한 온라인 업데이트 방법을 제안합니다. 기존의 Retrieval Enhanced Generation (RAG) 모델을 기반으로 하며, 데이터의 동적 기억 장치를 활용하여 새로운 데이터를 효율적으로 통합하는 혁신적인 메커니즘을 포함하고 있습니다. 이를 통해 빠르게 변화하는 정보 환경에서 실시간으로 지식을 업데이트하고 적응하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 동적 기억(dynamic memory)을 사용하여 새로운 데이터 샘플을 수집하고, 조정 가능한 지식 증류(knowladge distillation) 전략을 통해 모델에 통합하는 방식을 채택합니다. 또한, 계층적 인덱싱(hierarchical indexing)과 다층 게이팅(multi-layer gating) 메커니즘을 통해 검색 모듈의 정확성을 높이고 있습니다. 마지막으로 다양한 입력 유형에 대응하기 위해 멀티 스테이지 네트워크 구조를 설정하여, 각 단계의 중간 표현에 대해 cross-attention 매칭 및 스크리닝을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 지식 보존(knowledge retention)과 추론 정확도(inference accuracy) 측면에서 기존의 주류 비교 모델보다 우수한 성능을 보여주었습니다. 이는 새로운 지식과 기존 지식의 효과적인 통합과 반복 업데이트를 통해 이루어졌습니다. 또한, 특히 동적이고 변화하는 데이터 환경에서의 적응성 향상에 기여하고 있습니다.



### Graph Contrastive Learning on Multi-label Classification for Recommendations (https://arxiv.org/abs/2501.06985)
Comments:
          Preprint. 10 figures, 5 tables

- **What's New**: 이번 논문은 다중 레이블 분류를 위한 그래프 대조 학습(Graph Contrastive Learning for Multi-label Classification, MCGCL) 모델을 제안하여 추천 시스템의 효과성을 개선하려고 합니다. MCGCL은 두 개의 훈련 단계를 포함하고 있으며, 기본 작업과 하위 작업으로 구성됩니다. 이를 통해 사용자-아이템 간의 관계를 보다 잘 캡처할 수 있습니다.

- **Technical Details**: MCGCL은 사용자-아이템 이분 그래프를 하나의 전체 그래프와 두 개의 동질 하위 그래프로 분해하여 작업을 수행합니다. 첫 번째 단계에서는 전체 그래프에서의 표현 학습에 중점을 두고, 두 번째 단계에서는 이전 단계에서 얻은 표현을 바탕으로 동질 하위 그래프의 표현을 학습합니다. 이 구조는 다중 레이블 분류에서의 대조 학습의 적합성을 탐구하는데 중요한 기여를 합니다.

- **Performance Highlights**: 실험은 Amazon Reviews 데이터세트를 사용하여 진행되었으며, MCGCL은 다중 레이블 및 이진 레이블 분류 작업에서 다른 방법들보다 우수한 성과를 보였습니다. 추가적인 하이퍼파라미터 분석을 통해 MCGCL의 강력한 일반화 능력이 확인되었습니다. 이는 추천 시스템의 성능 향상 잠재력을 보여줍니다.



### Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation (https://arxiv.org/abs/2501.06362)
Comments:
          This paper has been accepted as a full paper at the 47th European Conference on Information Retrieval (ECIR2025)

- **What's New**: 이 논문은 다음 바구니 추천(Next Basket Recommendation, NBR) 문제에서 반복 편향(repeat bias)을 완화하면서 다양성(diversity)과 항목 공정성(item fairness)을 최적화하는 알고리즘을 제안합니다. 기존의 NBR 방식들은 주로 반복 항목을 추천하는 경향이 강해 다양한 항목의 추천이 어렵다는 점을 지적합니다. 새로운 알고리즘은 여러 NBR 접근법에 적용 가능하도록 확장되며, 실험을 통해 그 효용을 입증합니다.

- **Technical Details**: 저자는 모델 무관한 반복 편향 인식 최적화 알고리즘(repeat-bias-aware optimization algorithms)인 RADiv와 RAIF를 제안합니다. 이 알고리즘들은 혼합 정수 선형 프로그래밍(mixed-integer linear programming, MILP)을 기반으로 하여 이론적 기반을 제공합니다. 각 NBR 방법론에 적합하게 조정하여 반복 비율(repeat ratio), 다양성 및 항목 공정성을 동시에 최적화하는 것이 핵심입니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 세 가지 실제 데이터셋에서 다양성과 항목 공정성을 효과적으로 개선하면서도 반복 편향을 적절한 수준의 Recall 손실을 감수하면서 완화할 수 있음을 보여줍니다. 즉, 반복 항목을 줄이며 추천 품질을 향상시키는 동시에, 추천 리스트의 균형을 맞추는 데 성공합니다.



### ListConRanker: A Contrastive Text Reranker with Listwise Encoding (https://arxiv.org/abs/2501.07111)
Comments:
          11 pages, 4 figures

- **What's New**: 본 연구에서는 Listwise-encoded Contrastive text reRanker(ListConRanker)라는 새로운 reranker 모델을 제안합니다. 이 모델은 기존의 pointwise 및 pairwise encoding의 한계를 극복하며, passage 간의 비교 단계를 아우르는 listwise encoding 방식을 채택하고 있습니다. 또한, Circle Loss를 손실 함수로 사용함으로써 학습 효율성을 높이고 gradient의 변화가 부드럽게 이루어지도록 합니다.

- **Technical Details**: ListConRanker는 BERT 기반의 모델이며, query와 passage를 embedding 모델에 입력하여 원래의 feature를 추출한 뒤, 이를 통합하여 ListTransformer에 전달합니다. ListTransformer는 passage feature 간의 글로벌 contrastive 정보 학습을 용이하게 하여, 유사한 passage와 비유사한 passage 간의 군집화를 수행합니다. 제안된 ListAttention은 query의 feature를 유지하면서도 글로벌 비교 정보를 학습하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과 ListConRanker는 Chinese Massive Text Embedding Benchmark에서 최첨단 성능을 달성하였습니다. 특히 cMedQA1.0, cMedQA2.0, MMarcoReranking 및 T2Reranking 데이터셋에서도 높은 성능을 입증하였습니다. 또한, ablation study를 통해 ListTransformer와 Circle Loss의 효과 역시 검증되었습니다.



### Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation (https://arxiv.org/abs/2501.07110)
Comments:
          This paper has been accepted by ACM Transactions on Information Systems

- **What's New**: 본 논문에서는 동적 멀티모달 융합 기술을 통해 마이크로 비디오 추천의 표현 학습을 향상시키기 위해 Meta Multimodal Fusion (MetaMMF)이라는 새로운 메타 학습 기반의 멀티모달 융합 프레임워크를 제안합니다. 기존 방법의 정적 융합 한계를 극복하고, 각 마이크로 비디오에 대해 독립적으로 최적화된 융합 함수를 학습할 수 있는 방법론을 개발했습니다. 이를 통해 다양한 마이크로 비디오의 멀티모달 정보 간의 관계를 더 잘 모델링할 수 있습니다.

- **Technical Details**: MetaMMF는 주어진 입력 마이크로 비디오의 멀티모달 특성을 기반으로 메타 정보를 추출하고, 이를 통해 동적으로 파라미터화된 신경망을 융합 함수로 활용합니다. 주로 다층 퍼셉트론 (MLP)을 사용하여 멀티모달 특성을 고차원 추상화 형태의 메타 정보로 변환합니다. 또한, 모델의 복잡성을 줄이기 위해 Canonical Polyadic (CP) 분해를 채택하여 학습 효율성을 높였습니다.

- **Performance Highlights**: 세 가지 공공 데이터 세트를 기반으로 한 실험에서, MetaMMF는 기존의 여러 최첨단 멀티모달 추천 모델 (예: MMGCN, LATTICE, InvRL)보다 현저한 성능 향상을 보였습니다. 이는 MetaMMF가 목표한 동적 멀티모달 융합 방식을 통해 이루어진 결과이며, 다양한 마이크로 비디오가 각각 독특한 방식으로 멀티모달 정보를 통합할 수 있음을 보여줍니다.



### A Proposed Large Language Model-Based Smart Search for Archive System (https://arxiv.org/abs/2501.07024)
Comments:
          The 13th International Symposium on Information and Communication Technology (SOICT 2024)

- **What's New**: 이 연구는 Digital Archival Systems에서 정보 검색을 강화하기 위해 Large Language Models (LLMs)의 능력을 활용한 혁신적인 프레임워크를 제시합니다. Retrieval-Augmented Generation (RAG) 접근법을 사용하여 사용자의 자연어 질의를 처리하고 비텍스트 데이터를 의미 있는 텍스트 표현으로 변환합니다. 이 시스템은 고급 메타데이터 생성 기술, 하이브리드 검색 메커니즘, 라우터 쿼리 엔진, 강력한 응답 합성을 통합하여 검색 정확도와 관련성을 검증했습니다.

- **Technical Details**: 제안된 시스템은 RAG 아키텍처에 기반하여 자연어 질의를 입력받고, 지식 베이스에서 정보를 검색한 후 응답을 생성합니다. 다양한 출처에서 수집된 데이터는 동영상, 이미지, 텍스트 등의 다양한 데이터 타입을 포함하며, 모든 비텍스트 데이터는 균일한 텍스트로 변환됩니다. 이 연구에서는 BGE-M3 임베딩 모델을 사용하여 벡터 표현을 생성하고, Pinecone이라는 확장 가능한 벡터 데이터베이스를 활용하여 효율적인 유사성 기반 검색을 수행합니다.

- **Performance Highlights**: 제안된 프레임워크는 기존 접근법보다 유의미한 성능 향상을 보여 주었습니다. 실험을 통해 LLM 효율성, 하이브리드 검색 최적화, 다국어 쿼리 처리 등의 분야에서 성능 개선이 확인되었으며 AI 기반 시스템의 잠재력이 현대 아카이브 실무를 혁신할 수 있음을 입증했습니다. 특히 하이브리드 검색 모델은 BM25와 벡터 기반 접근 방식의 장점을 결합하여 검색 정확도를 높였습니다.



### Patent Novelty Assessment Accelerating Innovation and Patent Prosecution (https://arxiv.org/abs/2501.06956)
- **What's New**: 이번 보고서는 특허를 통해 지적 재산권을 보호하고 연구 개발 투자 촉진을 위한 혁신적인 Patent Novelty Assessment and Claim Generation System을 소개합니다. 이 시스템은 지적 재산의 발명적 측면을 분석하고, 방대한 특허 청구 데이터에 대한 접근을 간소화하도록 설계되었습니다. 특히 중국 특허의 특수성을 고려하여 대학생과 연구자들이 특허 청구의 복잡성을 이해하고 탐색할 수 있도록 직관적인 플랫폼을 제공합니다.

- **Technical Details**: 전통적인 분석 시스템과 달리, 본 시스템은 독점적인 Chinese API를 활용하여 정확성과 적합성을 높입니다. 주된 도전은 다양한 특허 청구에 대한 접근성과 이해의 복잡성으로, 이는 기존 아이디어에 대한 효과적인 혁신을 저해할 수 있습니다. 우리의 솔루션은 이러한 장벽을 극복하고 중국 특허 환경의 세부 사항에 맞춰 조정된 종합적인 청구 정보 접근을 제공합니다.

- **Performance Highlights**: 이 혁신적인 플랫폼은 사용자가 특허 청구 정보에 효율적으로 접근할 수 있도록 설계되어, 지적 재산 분야에서의 정보 탐색과 혁신을 촉진하는 것을 목표로 하고 있습니다. 이는 개별 대학을 넘어 연구 개발에 도움이 되는 환경을 조성하고, 학문 공동체 내에서 특허 개념에 대한 이해를 심화하는 데 기여할 것입니다.



### Causal Claims in Economics (https://arxiv.org/abs/2501.06873)
Comments:
          For data, interactive tools, and additional project information, visit this https URL. The website contains resources such as data downloads, interactive author and paper-level knowledge graphs, and more

- **What's New**: 이 논문은 1980년부터 2023년까지의 44,000개 이상의 NBER와 CEPR 작업 논문을 분석하여, 경제 개념 및 그 관계를 매핑하는 지식 그래프를 구축했습니다. 특히 인과 추론 방법(ex: DiD, IV, RDD, RCTs)을 통해 문서화된 일반 주장과 인과 주장을 구분하며, "신뢰성 혁명(credibility revolution)"의 확산이 인과 주장에 미치는 영향을 조명합니다. 인과 서사 복잡성(causal narrative complexity)이 상위 5개 저널에 게재되거나 더 많은 인용을 받을 가능성과 강한 상관관계를 보여줍니다.

- **Technical Details**: 이 연구는 LLMs(대형 언어 모델)인 GPT-4o-mini를 활용하여 비구조적 텍스트에서 구조화된 정보를 추출하고 분석하는 방법론적 접근을 사용했습니다. 이 과정에서는 30페이지 분량의 경제 작업 논문을 처리하며, 핵심 메타데이터와 인과 주장을 식별합니다. 데이터와 변수의 사용을 체계적으로 추출하기 위해 JSON 구조를 적용하여, 높은 정확도와 일관성을 유지하는 방식을 채택하였습니다.

- **Performance Highlights**: 결과적으로, 인과 혁신(causal innovation)과 방법론적 엄격함(methodological rigor)이 학문적 인정을 받는 주요 요인으로 작용하지만, 지속적인 영향력을 확보하기 위해서는 새로운 기여와 기존 경제 담론 간의 통합(balance)이 필요함을 보여주었습니다. 인과적 방법론에 기반으로 한 새로운 개념 쌍(underexplored concept pairs)의 다리가 인정받지만, 장기적인 인용과의 일관된 연관성은 발견되지 않았습니다. 본 연구는 경제학의 저널 출판 성공에 영향을 미치는 여러 요소를 밝히고 있습니다.



### Unveiling Temporal Trends in 19th Century Literature: An Information Retrieval Approach (https://arxiv.org/abs/2501.06833)
Comments:
          Accepted at JCDL 2024

- **What's New**: 이 논문에서는 19세기 영국 소설에서 사용된 용어의 변화를 정보 검색(information retrieval)의 관점에서 분석합니다. 특히, British Library의 소설 컬렉션을 세분화하여 쿼리 확장을 기반으로 한 접근법을 통해 시간에 따른 관련 용어의 변동을 조사합니다. 이 연구는 언어와 개념의 진화를 강조하며 19세기 소설의 언어 특징의 미세한 차이를 밝혀내는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 Relevance-based Language Model(RLM)을 적용하여 19세기 소설에서 선택된 쿼리 확장 용어의 변화를 분석합니다. 주요 메트릭으로는 Kendall's tau, Jaccard similarity 및 Jensen-Shannon divergence를 사용하여 서로 다른 시기에 따른 용어 선택의 일치 수준과 변화를 평가합니다. 이러한 기술을 활용하여, 각 수십 년에 따른 용어의 의미와 관련성을 정량적으로 측정하고 분석합니다.

- **Performance Highlights**: 결과적으로, 연구진은 쿼리 확장 기법에 의해 선택된 관련 용어가 연대별로 상당한 차이를 보이며, 이는 19세기 소설의 언어적 및 개념적 변화의 깊이를 나타냅니다. 특히, 동일 쿼리에 대한 확장 용어의 최고 가중치가 각 연대에 따라 엄청난 변화를 보이며, 이는 언어의 진화하는 성격을 반영합니다. 이러한 분석을 통해 언어와 주제의 시간적 역학을 철저히 짚어보는 유용한 통찰을 얻게 됩니다.



### Large Language Models, Knowledge Graphs and Search Engines: A Crossroads for Answering Users' Questions (https://arxiv.org/abs/2501.06699)
- **What's New**: 이번 논문은 대규모 언어 모델(LLMs), 지식 그래프(knowledge graphs), 검색 엔진(search engines) 간의 시너지 가능성을 다양한 사용자 정보 수요를 반영하여 분석합니다. 특히 사용자 중심의 관점에서 이들 기술의 장단점을 비교하고, 이들 기술이 정보 제공에 어떻게 기여할 수 있는지를 제시합니다. 이 연구는 결국 미래 연구의 로드맵을 정립하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 대규모 언어 모델은 변환기(Transformer) 아키텍처를 기반으로 하여 훈련된 인공지능 모델이며, 이들은 자연어 처리 및 다양한 작업에서 효율성을 보여줍니다. 하지만, 이러한 LLM들은 사실 확인 오류, 불투명성(opaqueness), 정보의 최신성(staleness) 등의 한계를 가지며, 이를 극복하기 위해 검색 엔진과 결합된 정보 탐색 방안인 Retrieval Augmented Generation(RAG) 기술이 주목받고 있습니다. 또한 지식 그래프(KGs)는 데이터와 지식을 구조화하여 질의하고 추론할 수 있도록 돕습니다.

- **Performance Highlights**: 논문에서는 LLMs이 때때로 사용자 질문에 대해 부정확하거나 불완전한 답변을 제공할 수 있음을 보여주는 사례를 들며, 이들 기술이 어떻게 서로를 보완할 수 있는지를 설명합니다. 예를 들어, LLM이 ACM Fellow에 대한 질문에 오류를 반환한 사례를 통해, 특히 긴 꼬리(long-tail) 정보에서의 한계를 강조하고 있습니다. 이에 따라 사용자 정보 요구를 충족시키기 위한 연구 방향성도 제시하고 있습니다.



### Recommending the right academic programs: An interest mining approach using BERTopic (https://arxiv.org/abs/2501.06581)
Comments:
          Accepted at Data Mining and Knowledge Discovery (Springer)

- **What's New**: 본 논문은 학생들이 개인적인 선호와 프로그램 내용을 기반으로 효율적인 추천을 받을 수 있는 최초의 정보 시스템을 제안합니다. BERTopic 알고리즘을 활용하여 모든 강좌 설명에서 흥미로운 주제를 추출하고, 이를 통해 각 프로그램의 지식을 포괄적으로 나타냅니다. 학생의 선택한 주제를 바탕으로 가장 적절한 프로그램 목록을 계산하는 독창적인 방법론을 개발하였습니다.

- **Technical Details**: 이 시스템은 비지도 머신러닝 기법인 topic modeling을 사용하여 관련 키워드의 관리 가능한 집합을 탐색하고, 이를 통해 학생들이 관심 있는 주제를 탐색할 수 있는 종합적인 세트를 제공합니다. 옵션으로 주어진 다중 관심 주제를 통해 프로그램 관련성 점수를 계산하고, 최종적으로 가장 관련성이 높은 전공을 나열한 순위를 생성합니다. 시스템은 Python 프로시저와 공개된 프로그램 데이터를 활용하여 자동화된 방식으로 즉각적인 의사 결정을 지원합니다.

- **Performance Highlights**: 사례 연구에서 이 시스템은 80개 프로그램과 5,000개 이상의 강좌를 통해 신속하고 효과적인 의사 결정을 지원하는 것으로 나타났습니다. 65명의 학생을 대상으로 한 질적 연구 결과, 98% 이상의 사용자가 추천이 본인의 관심사와 일치한다고 응답하였고, 약 94%는 향후 도구를 사용할 것이라고 밝혔습니다. 정량적 분석에서는 공정성을 보장하며 98%의 프로그램 커버리지를 달성했음을 보여줘, 데이터 중심의 실시간 사용자 중심 시스템이 전공 선택 과정을 개선할 수 있음을 시사합니다.



### Analyzing the Role of Context in Forecasting with Large Language Models (https://arxiv.org/abs/2501.06496)
- **What's New**: 이번 연구에서는 최근의 언어 모델(LLMs)이 이진 예측 질문에 대한 예측 성능을 평가합니다. 600개 이상의 이진 예측 질문으로 구성된 새로운 데이터셋을 소개하고, 다양한 수준의 문맥을 가진 입력 프롬프트가 예측 성능에 미치는 영향을 조사합니다. 뉴스 기사를 포함할 경우 성능이 크게 개선되지만, few-shot 예제 사용 시 정확도가 저하되는 결과를 보였습니다.

- **Technical Details**: 연구는 Metaculus 플랫폼에서 수집한 614개의 이진 예측 질문을 바탕으로 하며, 각 질문에 관련된 뉴스 기사를 요약하여 LLM의 예측 과정에 필요한 문맥을 제공합니다. 모델은 GPT-3.5-turbo, Alpaca-7B, Llama2-13B-chat의 세 가지로, 각각의 모델에 대해 다섯 가지 다른 입력 프롬프트를 사용하여 예측 성능을 평가했습니다. 특히, 관련된 뉴스 기사를 포함한 질문이 모델의 예측 성공에 긍정적인 영향을 미쳤습니다.

- **Performance Highlights**: 연구 결과, GPT-3.5-turbo 모델이 모든 LLM 중 최고의 성공률을 보였고, Llama2-13B-chat 모델이 Alpaca-7B보다 더 나은 성과를 내었습니다. 백그라운드 정보와 뉴스 기사를 포함한 프롬프트가 가장 높은 점수를 기록했습니다. 이는 LLM의 예측 성능이 제공된 정보의 양과 질에 크게 의존한다는 것을 시사합니다.



### Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts (https://arxiv.org/abs/2501.06365)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 의학 문헌에서 성 편향을 완화하기 위해 성별 직업 대명사를 중립화하는 파이프라인(pipeline)을 제시합니다. 1965년부터 1980년까지의 379,000개의 PubMed 초록을 처리하여 직업에 관련된 대명사를 수정했습니다. 이를 통해 개발된 MOBERT 모델은 성 중립화된 초록으로 훈련되었으며, 기존 모델인 1965Bert와 성능을 비교하였습니다.

- **Technical Details**: MOBERT는 BERT 모델을 기반으로 하여 성별 대명사가 도입된 직업 용어와 관련된 대명사를 중립화하도록 특별히 설계된 파이프라인을 사용했습니다. 이 과정에서는 Llama-3.1을 이용하여 대명사 해소(pronoun resolution) 쿼리를 수행하며, 숙련된 주석자가 반영된 사전 정의된 분류 규칙을 바탕으로 대명사 해소 작업을 진행했습니다. 초록에 포함된 직업 용어를 정확히 작성하는 것을 목표로 설계된 어휘 사전이 사용되었습니다.

- **Performance Highlights**: MOBERT는 포함된 대명사 교체 비율에서 70%를 달성하였고, 기존 모델인 1965Bert는 4%에 그쳤습니다. MOBERT의 성능 분석 결과, 대명사 교체 정확도는 훈련 데이터 내 직업 용어의 빈도와 상관관계가 있음을 보여줍니다. 이러한 결과는 향후 데이터셋 확장 및 파이프라인 개선을 통해 보다 공정한 언어 모델링을 할 수 있는 가능성을 제시합니다.



### Environmental large language model Evaluation (ELLE) dataset: A Benchmark for Evaluating Generative AI applications in Eco-environment Domain (https://arxiv.org/abs/2501.06277)
- **What's New**: 이번 연구에서는 생태 및 환경 과학에 대한 평가를 위한 첫 번째 기준점인 Environmental Large Language model Evaluation (ELLE) 질문 응답(QA) 데이터셋을 소개합니다. 이 데이터셋은 최소한 1,130개의 질문-응답 쌍을 포함하고 있으며, 16개의 환경 주제를 포괄합니다. ELLE 데이터셋은 이러한 분야에서의 성능 평가를 표준화하여, 생성적 AI의 성능을 일관되고 객관적으로 비교할 수 있게 합니다.

- **Technical Details**: ELLE 데이터셋은 주제(domain), 난이도(difficulty), 유형(type)에 따라 분류된 질문-응답 쌍을 제공하며, 이는 생태학 및 환경과학의 다양한 응용 분야에 적합한 평가 도구입니다. 이 연구는 통합된 평가 프레임워크의 부재로 인해 제한된 생성적 AI의 효율성을 극복하는 데 기여합니다. ELLE 데이터셋은 지속 가능한 환경 결과를 위한 생성적 AI 기술의 개발과 응용을 촉진하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 데이터셋을 통해 생태 및 환경 과학 분야에서의 생성적 AI의 성능이 평가될 수 있으며, 이를 통해 앞으로의 연구 방향과 정책 지원에 중대한 영향을 미칠 것으로 기대됩니다. 이 연구는 생성적 AI 기술이 환경 모니터링, 데이터 분석, 교육 및 정책 지원과 같은 생태 및 환경 애플리케이션에서의 효과를 극대화할 수 있도록 합니다.



New uploads on arXiv(cs.CV)

### DAViD: Modeling Dynamic Affordance of 3D Objects using Pre-trained Video Diffusion Models (https://arxiv.org/abs/2501.08333)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 기존의 고정된 Human-Object Interaction (HOI) 패턴에 대한 연구에서 벗어나, 동적 환경에서의 Human-Object Interaction을 효과적으로 모델링하기 위해 새로운 개념인 Dynamic Affordance를 도입합니다. 이는 3D 객체 메쉬와의 상호작용을 통해 동적으로 변하는 인간의 행동과 객체의 자세 변화를 학습하여, 다양한 환경에서의 물체 활용 방식에 대한 보다 깊은 이해를 제공합니다. 특히, 사전 훈련된 비디오 생성 모델을 활용하여 2D 비디오를 생성하고 이를 3D로 변환하여 4D HOI 샘플을 만드는 방법을 제안합니다.

- **Technical Details**: Dynamic Affordance는 사람과 객체의 상호작용 동안의 동적 움직임을 모델링하며, 이를 통해 인간의 행동과 객체의 자세 분포를 캡처합니다. 논문에서는 Low-Rank Adaptation (LoRA) 모듈을 사용하여 사전 훈련된 인간 동작 확산 모델(MDM)과 객체 자세 확산 모델을 기반으로 DAViD라는 새로운 프로세스를 훈련합니다. 이 과정에는 다양한 객체 카테고리를 포함한 복잡한 상호작용을 기반으로 한 동적 인간-객체 쌍 데이터셋을 필요로 하며, 이를 통해 기존의 정적 패턴 이상의 정보를 제공합니다.

- **Performance Highlights**: DAViD는 다양한 4D HOI 샘플을 생성하는 데 있어 기존 방법들보다 우수한 성능을 보이는 것으로 나타났습니다. 특히, 인체 동작과 HOI를 생성하는 과정에서 LoRA를 통한 동적 패턴 모델링의 장점을 분명히 합니다. 실험 결과, DAViD는 다양한 객체와의 상호작용을 포함한 동적 인간 동작 생성에서 우위를 점하며, 기존 방식보다 더 많은 정보를 제공하고 효율성을 보여주었습니다.



### MangaNinja: Line Art Colorization with Precise Reference Following (https://arxiv.org/abs/2501.08332)
Comments:
          Project page and code: this https URL

- **What's New**: MangaNinja는 diffusion models에서 파생되어 라인 아트 색칠 작업의 새로운 접근 방식을 제안합니다. 이 모델은 patch shuffling 모듈과 point-driven control scheme을 도입하여 참조 이미지와 라인 아트 간의 정확한 색상 일치를 촉진합니다. 실험 결과는 우리 모델이 기존 방법에 비해 색상 정밀도에서 우수성을 나타낸다는 것을 보여줍니다. 또한, 제안된 인터랙티브 포인트 제어의 잠재력을 강조하여 여러 도전적인 사례를 다룰 수 있도록 합니다.

- **Technical Details**: MangaNinja는 라인 아트 이미지와 참조 이미지 간의 상관 관계를 찾기 위한 이중 가지 구조를 갖추고 있습니다. patch shuffling 모듈은 참조 이미지를 패치로 분할하여 모델이 지역 일치 능력을 강화하도록 도와줍니다. PointNet에 의해 구동되는 포인트 기반 제어 스킴은 사용자 정의 큐를 사용하여 사용자와의 상호작용을 통해 세밀한 색상 제어를 가능하게 합니다. 이와 같은 방법론은 복잡한 색칠 작업을 효과적으로 관리할 수 있게 해 줍니다.

- **Performance Highlights**: 전반적인 평가를 위해 라인 아트 색칠을 위한 포괄적인 벤치마크를 구축하였습니다. 광범위한 정량적 및 정성적 실험 결과에 따르면, MangaNinja는 기존 기준을 초과하여 시각적 충실도와 정체성 보존에서 최첨단 결과를 달성합니다. 이는 만화, 애니메이션 등 다양한 콘텐츠 제작 애플리케이션에 유용함을 입증합니다. 특히, 다양한 포즈와 참조 간의 세부사항이 누락된 경우에도 높은 품질의 색칠 결과를 생성할 수 있습니다.



### Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Nois (https://arxiv.org/abs/2501.08331)
- **What's New**: 이번 논문에서는 비디오 디퓨전 모델의 Enhancing을 위한 새로운 접근 방식을 제안합니다. 랜덤 노이즈를 구조화된 출력으로 변환하는 과정에서, 구조화된 잠재 노이즈 샘플링을 통한 모션 제어를 허용하여 기존 방법의 제약을 극복합니다.

- **Technical Details**: 제안된 방법은 새로운 노이즈 왜곡 알고리즘을 사용하여 실시간으로 작동할 수 있도록 설계되었습니다. 이 알고리즘은 훈련 비디오로부터 얻은 광학 흐름 필드에서 파생된 상관된 왜곡 노이즈를 통해 랜덤한 시간적 Gaussianity를 대체합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 고품질 픽셀 유지, 모션 제어의 효율성, 그리고 사용자 선호도를 보여주었습니다. 이는 비디오 디퓨전 모델에서의 다양한 애플리케이션에 적합한 다재다능한 솔루션으로 입증되었습니다.



### Predicting 4D Hand Trajectory from Monocular Videos (https://arxiv.org/abs/2501.08329)
- **What's New**: HaPTIC은 단일 비디오에서 일관된 4D 손 궤적을 추론하는 새로운 접근 방식을 제시합니다. 기존의 손 포즈 재구성 방법들은 주로 인접한 프레임을 사용하여 3D 포즈를 개선하는 데 중점을 두었지만, HaPTIC은 4D 손 궤적을 직접 예측합니다. 이는 2D 재투영 정렬을 강하게 유지하면서도 지상 진실에 유사한 결과를 제공합니다.

- **Technical Details**: 이 시스템은 여러 프레임을 입력으로 받아들이기 위해 최첨단 이미지 기반 transformer를 재사용합니다. 두 가지 경량화 주의(attention) 레이어인 교차-view self-attention 및 글로벌 교차-attention을 도입하여 시간 정보와 더 큰 공간 맥락을 융합합니다. 이를 통해 HaPTIC은 비디오 데이터와 이미지 데이터를 혼합하여 훈련할 수 있어 강력한 일반화 능력을 유지합니다.

- **Performance Highlights**: HaPTIC은 allocentric 및 egocentric 비디오에서 기존 방법들보다 전반적인 궤적 정확도에서 우수한 성능을 보입니다. 또한, 프레임 기반 손 포즈 2D 정렬에서도 최신 이미지 기반 방법들을 초월하는 성능을 떼어냅니다. 모델이 공개됨에 따라 더 많은 연구와 활용이 기대됩니다.



### Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks (https://arxiv.org/abs/2501.08326)
Comments:
          Project page: this https URL

- **What's New**: Omni-RGPT라는 새로운 다중모달(멀티모달) 대형 언어 모델이 소개되었습니다. 이 모델은 이미지와 비디오를 위한 지역 수준의 이해를 촉진하는 데 중점을 둡니다. 특히, Token Mark라는 혁신적인 방법을 통해 시각적 특징 공간 내에서 목표 지역을 강조하는 토큰 세트를 도입했습니다.

- **Technical Details**: Omni-RGPT는 입력 받은 시각-텍스트 쌍에 따라 목표 지역 프롬프트(예: 박스 또는 마스크)를 사용하여 특정 토큰 마크를 샘플링하고, 이 토큰을 공간적으로 정의된 위치에 임베딩합니다. 이러한 방식은 비디오에서의 일관성 있는 지역 해석을 가능하게 하며, 임의의 프레임이 아닌 전체 프레임 수에 독립적인 대표성을 제공합니다.

- **Performance Highlights**: Omni-RGPT는 이미지 및 비디오 기반 상식 추론 벤치마크에서 최첨단 성능을 보여줍니다. 특히, VCR 및 Causal-VidQA와 같은 어려운 상식 추론 작업에서도 뛰어난 결과를 달성했으며, 새로운 RegVID-300k 데이터셋을 통해 비디오 캡셔닝 및 지역 수준 이해 태스크에서도 강력한 성능을 보입니다.



### GameFactory: Creating New Games with Generative Interactive Videos (https://arxiv.org/abs/2501.08325)
- **What's New**: 이번 논문에서는 게임 비디오 생성에서 장면 일반화(scene generalization)에 초점을 맞춘 새로운 프레임워크인 GameFactory를 제시합니다. 기존의 비디오 기반 게임 생성 방법들은 고정된 장면과 스타일에 국한되었으나, GameFactory는 이러한 제한을 극복하고 개방 도메인(open-domain) 비디오 데이터로 훈련된 사전 훈련 모델을 활용하여 다양한 게임 콘텐츠 생성을 가능하게 합니다.

- **Technical Details**: GameFactory는 여러 단계의 훈련 전략을 채택하여 게임 스타일 학습과 행동 제어(action control)를 분리하는 방식으로 작동합니다. 이로 인해 모델은 Minecraft 데이터에서 행동 제어 능력을 획득하면서도 기존의 게임 스타일에 얽매이지 않고 개방 도메인 일반성을 유지할 수 있습니다. 특히, 계속적인 마우스 움직임과 분리된 키보드 입력을 위한 다양한 제어 메커니즘을 개발하여 사용자 입력에 적절히 반응하도록 설계했습니다.

- **Performance Highlights**: 실험 결과, GameFactory는 개방 도메인에서 다양하고 행동 제어가 가능한 게임 비디오를 효과적으로 생성할 수 있음을 보여줍니다. GF-Minecraft라는 고품질의 행동 주석 비디오 데이터셋을 출시하여 연구자들이 AI 기반의 게임 엔진을 개발하는 데 필요한 자료를 제공합니다. 이러한 발전은 AI 주도의 게임 생성 분야에서 중요한 이정표로 평가됩니다.



### Diffusion Adversarial Post-Training for One-Step Video Generation (https://arxiv.org/abs/2501.08316)
- **What's New**: 본 연구에서는 Adversarial Post-Training (APT)이라는 새로운 방법을 통해 고해상도 비디오 생성을 위한 단일 단계 모델을 제안합니다. 기존의 감쇠 학습(distillation) 방법들이 영상 품질을 저하시키는 문제를 겪는 반면, APT는 실 데이터를 대상으로 적대적 학습을 통해 학습의 안정성과 품질을 향상시킵니다. 이를 통해 2초 길이의 1280x720 해상도 비디오를 실시간으로 생성할 수 있습니다.

- **Technical Details**: APT는 사전 훈련된 감쇠 모델을 초기화로 사용하며, 이 모델을 통해 실 데이터에 대해 적대적 학습 목표를 수행합니다. 이러한 과정에서 R1 정규화(regularization) 손실을 추가하여 대규모 학습을 용이하게 하고, 제너레이터와 판별기를 안정화하기 위한 여러 설계를 도입했습니다. 특히, 제너레이터는 결정론적(distillation) 학습을 통해 초기화되며, 판별기는 변환기(transformer) 기반의 구조로 설계되어 있습니다.

- **Performance Highlights**: 우리의 모델 Seaweed-APT는 2초 길이의 고해상도 비디오와 1024px 이미지를 생성하는 데 있어 기존 방법들과 비교할 수 있는 성능을 발휘합니다. 사용자 연구에 기반한 평가에서는 시각적 충실도(visual fidelity), 구조적 무결성(structural integrity), 텍스트 정렬(text alignment) 측면에서도 높이 평가받았습니다. 특히, H100 GPU를 활용하여 단일 단계로 1280x720 해상도의 비디오를 2초 만에 생성할 수 있는 성과를 이루었습니다.



### Advancing Semantic Future Prediction through Multimodal Visual Sequence Transformers (https://arxiv.org/abs/2501.08303)
- **What's New**: 이번 연구에서는 FUTURIST라는 새로운 방법을 소개하며, 이 방법은 통합되고 효율적인 visual sequence transformer architecture를 기반으로 다양한 modality에 대한 미래의 의미 예측을 수행합니다. 특히, 다중 모달 masked visual modeling의 목표와 기존의 VAE-based 토크나이징 방식의 필요성을 제거한 계층적 토크나이징 프로세스를 도입하여 계산 복잡성을 줄이고 고해상도 멀티모달 입력에 대한 end-to-end 학습을 가능하게 합니다. 이 모델은 Cityscapes 데이터셋에서의 성능 검증을 통해 단기 및 중기의 미래 의미 분할에서 최첨단 성능을 달성하며, 편리한 구현 코드도 제공합니다.

- **Technical Details**: FUTURIST의 핵심 기술은 다중 모달 기능을 지원하는 visual sequence transformer입니다. 이 아키텍처는 past 프레임에서 다양한 모달리티의 입력을 받아들여 미래 프레임에 대해 같은 모달리티를 예측합니다. VAE-Free embedding 전략을採用하여 각 모달리티의 픽셀 수준부터 패치 수준까지 진전하는 계층적 embedding 방식을 사용합니다. 이 과정에서 cross-modality fusion 메커니즘을 도입하여 시간적 공간에서 동일한 위치의 토큰을 병합함으로써 전체적인 토큰 시퀀스의 길이를 줄이고 계산 효율성을 높였습니다.

- **Performance Highlights**: FUTURIST는 우수한 성능을 검증한 실험 결과를 제공하며, 특정 모달리티에 대한 ground truth label map 없이 off-the-shelf foundation models에서 생성된 pseudo-label을 사용하여도 뛰어난 일반화 능력을 보여줍니다. 최종적으로, FUTURIST는 미래의 의미 세분화와 깊이 예측에서 최첨단 결과를 달성하며, 이는 다중 모달 정보의 상호 보완적 이점을 강조합니다. 이 연구는 멀티모달 학습 및 미래 예측 연구의 새로운 방향성을 제시합니다.



### LayerAnimate: Layer-specific Control for Animation (https://arxiv.org/abs/2501.08295)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 LayerAnimate라는 새로운 아키텍처를 소개합니다. 이 모델은 애니메이션의 개별 레이어를 보다 세밀하게 제어할 수 있게 해주는 비디오 디퓨전 모델입니다. 이를 통해 사용자들은 전경(foreground)과 배경(background) 요소를 독립적으로 조작할 수 있습니다. 기존의 비디오 생성 방법들이 레이어의 세밀함을 간과한 반면, LayerAnimate는 레이어의 특정 데이터를 제한적으로 보완하기 위한 데이터 큐레이션 파이프라인을 제안합니다.

- **Technical Details**: LayerAnimate는 레이어 제어의 정밀성을 위해 자동화된 요소 분할(automated element segmentation), 동작 상태 계층 병합(motion-state hierarchical merging), 그리고 동작 일관성 정제(motion coherence refinement)를 포함한 데이터 큐레이션 파이프라인을 사용합니다. 이 과정에서 SAM(Segment Anything Model)2를 활용하여 요소를 분할하고, 이 요소들을 동작 상태에 따라 계층적으로 병합합니다. 그 결과, 레이어의 동적 및 정적 특성을 분리하고, 비디오 전반에 걸쳐 정적 레이어를 안정화하는 데 기여합니다.

- **Performance Highlights**: LayerAnimate는 다양한 비디오 생성 작업을 통해 품질, 제어 정밀성, 사용 편의성에서 우수한 성과를 보였습니다. 사용자 연구와 정량적 비교를 통해 기존 방법들을 초월하는 성능을 입증하였습니다. 이 모델은 전문 애니메이터는 물론 아마추어 애니메이터에게도 적합한 도구로 자리 잡을 가능성을 보여주며, 레이어 기반의 애니메이션 응용을 위한 새로운 가능성을 열어줍니다.



### LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding (https://arxiv.org/abs/2501.08282)
- **What's New**: 이번 논문에서는 LLaVA-ST라는 새로운 다중 모달 대형 언어 모델(MLLM)을 제안하였습니다. 이 모델은 시공간의 미세한 이해를 동시에 처리할 수 있는 능력을 갖추고 있으며, 텍스트와 비주얼 정보를 효과적으로 정렬하기 위한 Language-Aligned Positional Embedding (LAPE) 기법을 도입했습니다. 또한 Spatial-Temporal Packer (STP)를 통해 시간적 및 공간적 압축을 분리하여 세밀한 정보 손실을 최소화하였습니다.

- **Technical Details**: LLaVA-ST는 fine-grained spatial-temporal understanding이라는 새로운 영역을 위해 설계되었습니다. 이 모델은 시각적 입력의 시간 범위와 공간 좌표를 동시에 이해하고 로컬라이즈 할 수 있는 첫 번째 MLLM입니다. LAPE와 STP를 적용하여 모델의 성능을 높이고, 4.3M 샘플로 구성된 ST-Align 데이터세트를 활용하여 훈련됩니다.

- **Performance Highlights**: LLaVA-ST는 Spatial-Temporal Video Grounding (STVG), Event Localization and Captioning (ELC), Spatial Video Grounding (SVG)와 같은 다양한 벤치마크에서 뛰어난 성능을 보였습니다. 11개의 다양한 벤치마크 테스트에서 높은 정확성을 기록하며, 기존 모델들과 비교하여 시공간적 interleaved understanding에 있어 혁신적인 결과를 보여주고 있습니다.



### SmartEraser: Remove Anything from Images using Masked-Region Guidanc (https://arxiv.org/abs/2501.08279)
Comments:
          Project at: this https URL

- **What's New**: 이 논문에서는 기존의 'mask-and-inpaint' 패러다임을 개선한 SmartEraser를 제안합니다. 새로운 'Masked-Region Guidance' 패러다임을 적용하여, 마스크된 영역을 입력에서 제외하지 않고 오히려 제거 프로세스의 가이드를 제공합니다. 이 방식은 모델이 제거할 객체를 정확히 식별하고, 최종 결과에서 주변 맥락을 효과적으로 보존하는 데 도움을 줍니다.

- **Technical Details**: SmartEraser는 Masked-Region Guidance를 기반으로 하여 객체 제거를 위한 새로운 접근 방식을 제안합니다. 이를 위해 Syn4Removal이라는 대규모 데이터셋을 구축하였으며, 100만 개의 이미지 트리플렛으로 구성되어 있습니다. 해당 데이터셋은 다양한 씬과 객체 타입을 포함하여, 모델이 샘플링 과정에서 의도치 않은 객체를 생성하지 않도록 설계했습니다.

- **Performance Highlights**: 실험 결과, SmartEraser는 기존의 객체 제거 방법들을 능가하는 성능을 보여주었습니다. 특히 복잡한 장면에서 더욱 뛰어난 성능을 발휘하며, 시각적 일관성도 크게 향상되었습니다. 이러한 결과는 SmartEraser가 객체 제거 작업을 수행하는 데 있어 신뢰할 수 있는 솔루션임을 입증합니다.



### AI Driven Water Segmentation with deep learning models for Enhanced Flood Monitoring (https://arxiv.org/abs/2501.08266)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 기후 변화로 인해 증가하는 홍수 발생 빈도에 대응하기 위해 UNet, ResNet, DeepLabv3의 세 가지 딥 러닝 모델을 비교하여 픽셀 수준의 수분 분할을 수행합니다. 새로운 데이터 세트를 생성하여 잘 알려진 기준 데이터 세트를 홍수 특정 이미지로 보강함으로써 모델의 견고성을 향상시켰습니다. 이 연구는 드론, 현장 관찰 및 소셜 미디어의 이미지를 활용하여 홍수 감지를 지원하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구에서는 UNet, ResNet, DeepLabv3 아키텍처를 다양한 환경 조건과 지리적 위치에서 테스트하여 각각의 효과성을 평가합니다. 모델은 완전 자동화된 방식으로 이미지를 분석하여 홍수 지역을 식별하며, 전통적인 반자동 방법에 비해 처리 시간을 대폭 단축합니다. 이미지 분할 마스크를 예측함으로써 각 모델의 강점과 한계를 논의하며, 효율적인 홍수 모니터링을 위한 방법론을 제시합니다.

- **Performance Highlights**: 이러한 접근 방식은 생명과 경제적 손실을 줄이기 위한 비상 대응 팀에 중요한 데이터를 제공하는 시기적절하고 지속적인 홍수 모니터링을 가능하게 합니다. 기존 방법에 비해 홍수 지도 생성 시간을 크게 단축시켰으며, 향후 연구를 위한 멀티모달 데이터 통합 및 홍수 감지 전용 견고한 딥 러닝 아키텍처 개발의 가능성을 제시합니다. 전반적으로, 이 연구는 딥 러닝 기술의 혁신적인 사용을 통해 홍수 관리 전략의 발전에 기여하고 있습니다.



### Towards an End-to-End (E2E) Adversarial Learning and Application in the Physical World (https://arxiv.org/abs/2501.08258)
- **What's New**: 본 연구는 물리적 영역에서 적대적 패치를 생성하고 적용할 수 있는 새로운 프레임워크인 Physical-domain Adversarial Patch Learning Augmentation (PAPLA)를 제안합니다. PAPLA는 기존의 디지털 도메인에서 물리적 도메인으로의 전이를 개선하여 적대적 공격의 성공률을 높이는 것을 목표로 합니다. 연구 결과, PAPLA는 특정 조건에서 E2E 적대적 학습에 의해 물리적 환경에서 효과적으로 작동함을 보여줍니다.

- **Technical Details**: PAPLA는 프로젝터를 사용하여 물리적 도메인에서 적대적 패치를 직접 생성하는 방법을 제안합니다. 이는 기존의 디지털 학습 후 물리적 적용방식(DL-PA)에서 발생하는 전이 문제를 해결하고, 다양한 환경 요인의 영향을 고려함으로써 공격의 효과성을 향상시킵니다. 실험에서는 프로젝터의 밝기, 환경 조명, 카메라 거리 및 각도 같은 요소가 조사되었으며, 이러한 요소들은 공격의 성공과 밀접한 관련이 있음을 확인하였습니다.

- **Performance Highlights**: PAPLA는 제어된 실험실 환경 및 실제 야외 환경에서 유의미한 성과를 보였습니다. 예를 들어, 주차된 자동차와 정지 표지판에 대한 공격 실험을 통해 PAPLA의 유용성을 입증했습니다. 실험 결과, PAPLA는 전통적인 디지털 학습-물리적 적용 방식보다 높은 신뢰도 감소를 제공하며, 다양한 환경 설정에서 공격의 유효성을 증가시키는 것으로 나타났습니다.



### Continual Deep Active Learning for Medical Imaging: Replay-Base Architecture for Context Adaptation (https://arxiv.org/abs/2501.08245)
- **What's New**: 이번 연구에서는 Continual Active Learning (CAL) 프레임워크인 Replay-Based Architecture for Context Adaptation (RBACA)를 개발하여 의료 이미지 분석의 성능 향상을 목표로 하였습니다. RBACA는 CL(Continual Learning) 리허설 기법과 AL(Active Learning) 요소를 통합하여 다양한 맥락에서 지속적으로 학습할 수 있도록 설계되었습니다. 이로 인해 의료 영상의 적응성과 일반화 능력이 향상되었습니다.

- **Technical Details**: RBACA는 자동으로 이미지 특성 변화 감지를 기반으로 하며, 학습 과정에서 가장 유익한 샘플을 선택하는 AL 컴포넌트를 포함하고 있습니다. 또한, 새로운 맥락에서 학습하기 위해 CL 리허설 방법을 사용하여 모델의 기억 저장소를 효과적으로 관리하고 메모리 프루닝 전략을 지원합니다. CAL 방법을 평가하기 위한 새로운 메트릭인 Incremental Learning Score (IL-Score)를 정의하여 이 과정에서의 성능을 동시에 측정하는 평가 방안을 제시합니다.

- **Performance Highlights**: RBACA는 도메인 및 클래스 증가 학습 시나리오에서도 우수한 성능을 보이며, 심장 이미지의 세분화 및 진단 평가에서 IL-Score가 높은 결과를 도출했습니다. 다양한 메모리 크기 및 주석 예산에 따라, RBACA는 기존 CAL 프레임워크와 비교하여 더 나은 성능을 보여줍니다. 이를 통해 RBACA는 의료 영상 처리의 다양한 맥락에서 적응성과 일반화를 크게 향상시킬 수 있음을 입증하였습니다.



### A Feature-Level Ensemble Model for COVID-19 Identification in CXR Images using Choquet Integral and Differential Evolution Optimization (https://arxiv.org/abs/2501.08241)
- **What's New**: COVID-19 팬데믹은 전 세계적으로 수십억 명에게 심각한 영향을 미쳤습니다. 본 논문은 Chest X-ray (CXR) 이미지를 통해 COVID-19 감염자를 정확히 식별하기 위한 새로운 Deep Learning Diagnosis System을 소개합니다. 이 시스템은 사전 학습된 Deep Convolutional Neural Networks (DCNNs)를 통합한 앙상블 학습(framework) 구조를 이용하여, 기존 RT-PCR의 한계를 극복합니다.

- **Technical Details**: 이 연구에서는 Choquet integral을 이용하여 서로 다른 DCNN의 특징 벡터를 결합하여, 선형 접근으로는 잡을 수 없는 상호작용을 포착합니다. Sugeno-$\lambda$ 측정 이론을 활용하여 네트워크의 하위 집합에 대한 퍼지(fuzzy) 측정을 도출하며, Differential Evolution 알고리즘을 통해 퍼지 밀도를 추정합니다. 또한, 복잡한 특성 벡터 집합을 용이하게 집계(strategies)할 수 있도록 TensorFlow 기반의 Choquet 연산 레이어를 개발했습니다.

- **Performance Highlights**: COVIDx 데이터셋에서의 실험 결과, 본 앙상블 모델은 세 개 클래스로 분류 시 98
%의 정확도, 이진 분류 시 99.50	ext{%}의 정확도를 기록했습니다. DenseNet-201(세 개 분류에서 97	ext{，以서술자}%, 이진에서 98.75	ext{%}), Inception-v3(세 개 분류에서 96.25	ext{%}, 이진에서 98.50	ext{%}), Xception(세 개 분류에서 94.50	ext{%}, 이진에서 98	ext{%}) 보다 월등히 뛰어난 성능을 보여주었으며, 이전의 많은 방법들을 초월했습니다.



### Efficient Deep Learning-based Forward Solvers for Brain Tumor Growth Models (https://arxiv.org/abs/2501.08226)
- **What's New**: 이번 연구에서는 신경망 기반의 forward solver에 기초한 새로운 접근 방식을 도입하여, 부분 미분 방정식(PDE) 모델의 보정 시간을 대폭 단축했습니다. 특히, TumorSurrogate라는 심층 학습 모델을 활용하여, 환자 개별의 종양 분포를 빠르게 예측하고 최적화할 수 있는 가능성을 제시합니다. 여러 아키텍처 실험을 통해, 최적화된 TumorSurrogate 모델이 종양 외형 매칭과 세포 농도 예측에서 가장 우수한 성능을 발휘했음을 보여줍니다.

- **Technical Details**: 종양 성장 모델은 반응-확산 공식에 따라, 두 가지 주요 프로세스인 증식(proliferation)과 확산(diffusion)을 기반으로 합니다. 모델링은 특정 환자의 해부학적 정보를 반영하여 수행되며, TumorSurrogate는 인코더-디코더 아키텍처를 통해 종양 시뮬레이션을 생성합니다. 이 연구에서는 nnU-Net 및 Vision Transformer 모델이 포함된 다양한 네트워크 아키텍처의 성능을 체계적으로 분석하여, 각 모델의 적합성을 평가했습니다.

- **Performance Highlights**: 연구 결과, TumorSurrogate 모델은 기본 모델 대비 평균 제곱 오차(MSE)를 절반으로 줄이며, 모든 종양 세포 농도 임계값에서 가장 높은 Dice 점수를 기록했습니다. 이로 인해, 치료 계획 수립에 있어 환자 맞춤형 접근 방식이 크게 개선될 수 있음을 입증하였습니다. 향후 연구 방향으로는, 본 접근 방식을 통해 보다 향상된 forward solver 성능과 종양 예측의 일반화 가능성을 논의하고 있습니다.



### FramePainter: Endowing Interactive Image Editing with Video Diffusion Priors (https://arxiv.org/abs/2501.08225)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서 저자들은 대화형 이미지 편집(interactive image editing)을 이미지-비디오 생성(image-to-video generation) 문제로 재구성하여, 기존 방법보다 데이터와 계산의 요구 사항을 줄이고자 하였습니다. 새로운 접근법인 FramePainter를 제안하며, 이는 강력한 비디오 디퓨전 프라이어(video diffusion priors)를 활용하여 효율적이고 정확한 이미지 편집을 가능하게 합니다. FramePainter는 이미지 편집 신호를 삽입하기 위해 가벼운 희소 제어 인코더를 사용합니다.

- **Technical Details**: FramePainter는 Stable Video Diffusion(SVD)로 초기화되며, 시간 주의 기법(temporal attention)의 한계를 극복하기 위해 매칭 주의(matching attention)를 도입합니다. 이 방법은 편집된 이미지 토큰과 원본 이미지 토큰 간의 밀집 일치를 유도하며, 이를 통해 재수용(fields of view)을 확장합니다. 또한, CoTracker-v3에서의 추적 결과에 맞춰 주의 가중치를 최적화하여 세밀한 시각적 세부 사항을 정확하게 캡처하도록 설계되었습니다.

- **Performance Highlights**: FramePainter는 다양한 편집 신호에 대해 이전의 최첨단 방법들을 초월하는 성능을 보여줍니다. 이는 거의 반값의 훈련 데이터로 가능하며, 컵의 반사를 자동으로 조정하는 등 매우 원활하고 일관된 이미지 편집을 수행합니다. 특히, 실제 비디오에서 발견되지 않은 시나리오에 대한 일반화(generalization) 또한 뛰어나며, 예를 들어, 꼬치 물고기를 상어 형태로 변환하는 과제를 성공적으로 수행합니다.



### EmoNeXt: an Adapted ConvNeXt for Facial Emotion Recognition (https://arxiv.org/abs/2501.08199)
Comments:
          6 pages, 5 figures and 2 tables. 2023 IEEE 25th International Workshop on Multimedia Signal Processing (MMSP), Poitiers, France

- **What's New**: 이 논문에서는 EmoNeXt라는 새로운 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 조정된 ConvNeXt 아키텍처를 기반으로 하며, Spatial Transformer Network (STN)과 Squeeze-and-Excitation 블록을 통합하여 얼굴의 특징적인 영역에 집중하고 채널 간 의존성을 포착합니다. 또한 self-attention 정규화 항을 소개하여 모델이 간결한 특징 벡터를 생성하도록 합니다. 이를 통해 FER2013 데이터셋에서 기존 모델들보다 높은 감정 분류 정확도를 달성하였습니다.

- **Technical Details**: EmoNeXt는 입력 이미지에 공간 변환을 학습하고 적용하기 위한 STN을 통합합니다. STN은 입력 이미지의 벡터 표현을 생성하고 이를 바탕으로 샘플링 그리드를 형성하는 로컬라이제이션 네트워크를 포함합니다. 그러면서 CNN의 효과를 유지하면서도 Vision Transformers의 장점을 살리기 위해 ConvNeXt 아키텍처에 대한 다양한 개선을 포함하고 있습니다. 특히, 활성화 함수로 GELU를 사용하고, BatchNorm 대신 Layer Normalization을 통해 성능을 향상시킵니다.

- **Performance Highlights**: EmoNeXt는 FER2013 데이터셋을 사용한 실험에서 기존의 최첨단 깊은 학습 모델들과 비교해 우수한 성능을 보여주었습니다. 모델은 더욱 간결한 특징을 생성하도록 유도함으로써 감정 분류에서의 정확도를 극대화합니다. 연구 결과는 EmoNeXt 모델이 자동으로 얼굴 감정을 인식하는 데 있어 효과적인 접근 방식임을 입증하고 있습니다.



### Self-supervised Deep Hyperspectral Inpainting with the Plug and Play and Deep Image Prior Models (https://arxiv.org/abs/2501.08195)
Comments:
          31 pages, 9 Figures, 7 Tables. arXiv admin note: text overlap with arXiv:2306.08128

- **What's New**: 이번 논문은 hyperspectral 이미지의 품질을 저하시킬 수 있는 다양한 노이즈 및 왜곡 문제를 해결하는 새로운 알고리즘, LRS-PnP-DIP(1-Lip)를 제안합니다. 이 알고리즘은 기존의 DHP(Driving Hyperspectral Processing)에서 보고된 불안정성 문제를 해결하며, 저차원(low-rank)과 희소(sparse) 모델을 성공적으로 결합하여 데이터 구조를 더욱 효과적으로 활용합니다.

- **Technical Details**: LRS-PnP-DIP(1-Lip) 알고리즘은 기존의 서브스페이스 모델의 고정된 집합을 넘어서 데이터를 보다 세밀하게 분석할 수 있도록 설계되었습니다. 이 알고리즘은 약한 가정 하에서도 수렴(convergence)을 보장하는 안정성 분석을 포함하고 있어 실제 응용 프로그램에서 신뢰성과 유용성을 크게 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 솔루션이 시각적으로 및 정량적으로 우수한 인페인팅(inpainting) 결과를 지속적으로 제공하며, 최신 기술(state-of-the-art) 성능을 입증합니다. 이를 통해 hyperspectral 이미지 처리 분야에서의 혁신적인 발전을 기대할 수 있습니다.



### A Critical Synthesis of Uncertainty Quantification and Foundation Models in Monocular Depth Estimation (https://arxiv.org/abs/2501.08188)
- **What's New**: 최근의 기초 모델들이 단안 깊이 추정(monocular depth estimation) 분야에서 중요한 돌파구를 마련했으나, 실제 환경에서 안전하고 신뢰할 수 있는 배포를 위한 명확한 경로는 여전히 부족합니다. 이는 거리 예측을 포함하는 메트릭 깊이 추정(metric depth estimation) 과정에서 특히 두드러지며, 고급 기초 모델조차도 치명적인 오류에 취약해 문제가 되고 있습니다. 본 연구에서는 5가지 불확실성 정량화(un) 방법을 강화된 DepthAnythingV2 기초 모델에 결합하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 기존의 메트릭 깊이 추정 방법론의 한계를 극복하기 위해, 연구팀은 Gaussian Negative Log-Likelihood Loss (GNLL)을 통해 신뢰할 수 있는 불확실성 예측을 제공하는 효율적인 기법을 제안하고 있습니다. 추가로 Monte Carlo Dropout (MCD), Sub-Ensembles (SE), Test-Time Augmentation (TTA) 등 다양한 불확실성 정량화(UQ) 방법을 적용하여 모든 픽셀에 대한 깊이 예측의 불확실성을 계량화하는 데 초점을 맞추고 있습니다. 이를 통해 총 4개의 다양한 데이터셋에서 성능을 평가하여, 실세계 응용에 대한 광범위한 커버리지를 확보하고자 합니다.

- **Performance Highlights**: 연구 결과, GNLL 기반의 미세 조정 방법이 특히 유망한 접근법으로 확인되었습니다. 이 방법은 예측 성능을 유지하면서도 불확실성을 신뢰성 있게 추정하는 데 뛰어난 효율성을 보여 주며, 기존의 기초 모델과 비교하여 훈련 및 추론 시간 면에서도 동등한 성능을 발휘합니다. 또한, 본 연구는 단안 깊이 추정을 넘어 발생할 수 있는 다양한 문제들과의 연결 단서를 제공하며, 데이터 반영에 대한 이해도와 신뢰성을 높이는 데 기여할 것입니다.



### D$^2$-DPM: Dual Denoising for Quantized Diffusion Probabilistic Models (https://arxiv.org/abs/2501.08180)
Comments:
          9 pages, 4 figures, acceptted by AAAI2025

- **What's New**: 본 연구에서는 D2-DPM이라는 이중 노이즈 제거 메커니즘을 소개하여 양자화 노이즈가 노이즈 추정 네트워크에 미치는 부정적 영향을 정밀하게 완화하려고 합니다. 양자화가 노이즈 추정 네트워크에 미치는 영향을 평균 편차(mean deviation)와 분산 편차(variance deviation)라는 두 가지 구성 요소로 나누어 시각화하였습니다. 실험 결과에 따라 D2-DPM은 기존의 풀 프리시전(full-precision) 모델보다 1.42 낮은 FID를 달성하며 생성 품질에서 우수한 성능을 보입니다.

- **Technical Details**: 전통적인 부드러운 샘플링 과정에서 D2-DPM은 각 시간 단계에서 양자화 노이즈를 제거하고, 역 확산(iterative diffusion) 과정을 통해 노이즈가 있는 샘플을 추가적으로 정제합니다. 본 방법은 상관 모델링을 통해 추출된 양자화 노이즈의 통계적 특성을 반영하여, 양자화 출력과 양자화 노이즈를 결합하는 공동 가우시안 모델을 설계합니다. 덕분에 두 가지 변형인 Stochastic Dual Denoising(S-D2) 및 Deterministic Dual Denoising(D-D2)을 제안하여 과정을 진행합니다.

- **Performance Highlights**: D2-DPM은 다양한 생성 작업에서 특히 조건부 및 비조건부 이미지 생성 작업에 대해 주목할 만한 품질 향상을 보여주었습니다. 보고된 실험 결과에 따르면, D2-DPM은 효율적으로 생성 품질을 강화하면서도 3.99배의 압축및 11.67배의 비트 연산 가속화를 달성합니다. 이러한 결과는 D2-DPM이 양자화 기반의 손실을 최소화할 수 있는 유망한 방법임을 시사합니다.



### Object-Centric 2D Gaussian Splatting: Background Removal and Occlusion-Aware Pruning for Compact Object Models (https://arxiv.org/abs/2501.08174)
Comments:
          Accepted at ICPRAM 2025 (this https URL)

- **What's New**: 본 논문은 기존의 Gaussian Splatting 방법이 특정 객체를 목표로 하지 못하는 한계를 해결하기 위해 객체 마스크를 활용한 새로운 접근 방식을 제안합니다. 이를 통해 객체 중심(object-centric) 모델의 재구성을 가능하게 하여 계산력을 절약하고, 객체 특정 애플리케이션에 적합하도록 합니다. 추가적으로, 품질을 유지하면서 Gaussian 수를 최소화하는 occlusion-aware pruning 전략을 도입하였습니다. 이 방법은 훈련 시간을 최대 71",

- **Technical Details**: 제안된 방법은 배경 손실(background loss)을 활용하여 세그멘테이션 마스크에 의해 정의된 배경 Gaussian을 제거하며, 이를 통해 훈련 시간과 모델 크기를 줄이는 데 기여합니다. Gaussian과 메시(mesh) 표현을 생성하는 과정에서 occlusion-aware pruning 전략을 적용하여 렌더링에 기여하지 않는 Gaussian을 제거하고, 전체 품질 저하 없이 모델 크기를 더욱 축소합니다. 또한, 이 방식으로 생성된 모델은 추가적인 처리 없이도 외형 편집 및 물리적 시뮬레이션 등 후속 애플리케이션에서 즉시 사용 가능하게 됩니다.

- **Performance Highlights**: 제안된 접근 방식에 따르면, 객체 중심 Gaussian 및 메시 표현이 기존의 방법들과 비교하여 최대 96% 더 작고, 훈련 속도는 71% 더 빠른 결과를 제공합니다. 실제로, 이러한 표현들은 뛰어난 품질을 유지하며, 다양한 다운스트림 애플리케이션에 효율적으로 사용될 수 있는 강력한 도구가 됩니다. 특히, 이 기술은 애플리케이션 사용에 있어 강력한 유연성을 제공하면서도 실질적인 성능 향상을 이루고 있습니다.



### Benchmarking Multimodal Models for Fine-Grained Image Analysis: A Comparative Study Across Diverse Visual Features (https://arxiv.org/abs/2501.08170)
Comments:
          6 pages, 2 tables, 2 charts

- **What's New**: 이 논문에서는 멀티모달 모델(multimodal models)의 이미지 분석 및 해석 능력을 평가하기 위한 벤치마크(benchmark)를 소개합니다. 이 벤치마크는 주요 개체(main object), 추가 개체(additional objects), 배경(background), 세부 사항(detail), 주요 색상(dominant colors), 스타일(style), 및 시점(viewpoint) 등 일곱 가지 핵심 시각적 측면에 중점을 두고 있습니다. 14,580개의 이미지로 구성된 데이터셋을 사용하여 멀티모달 모델의 성능을 평가하였습니다.

- **Technical Details**: 데이터셋은 다양한 텍스트 프롬프트(text prompts)에서 생성된 이미지로 이루어져 있으며, 총 14,580개의 이미지를 포함하고 있습니다. 일곱 개의 주요 멀티모달 모델이 각 시각적 측면을 정확하게 식별하고 설명할 수 있는 능력을 평가했습니다. 이 연구는 각 모델의 강점과 약점을 파악하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 벤치마크 결과는 다양한 이미지 분석 작업을 위한 멀티모달 모델의 개발 및 선택에 중요한 함의를 제공합니다. 모델들은 각기 다른 시각적 측면을 이해하는 능력에서 차이를 보였으며, 이는 멀티모달 모델의 진화에 기여할 것으로 기대됩니다.



### Revolutionizing Communication with Deep Learning and XAI for Enhanced Arabic Sign Language Recognition (https://arxiv.org/abs/2501.08169)
Comments:
          13 pages, 25 figures, 16 tables

- **What's New**: 이 연구는 MobileNetV3, ResNet50, EfficientNet-B2와 같은 최신 딥 러닝 모델을 사용하여 아랍 수화(Arabic Sign Language, ArSL) 인식에 대한 통합 접근 방식을 소개합니다. 이 모델들은 해석 가능성과 투명성을 높이기 위해 Explainable AI (XAI) 기법으로 추가 강화되었습니다. 연구 결과, EfficientNet-B2는 각각 99.48%와 98.99%의 최고 정확도를 달성했습니다. 이 시스템은 인식 정확도에서 새로운 기준을 설정하고 다문화 소통 기술의 포함성을 강조합니다.

- **Technical Details**: 이 연구의 주요 기여는 다양한 손 제스처를 인식하기 위해 정교한 데이터 증강 방법과 계층적 5-겹 교차 검증을 사용한 것입니다. 또한, Grad-CAM을 통해 모델 결정의 투명성을 강화하였으며, 실시간 인식 정확도를 유지하면서 다양한 환경의 데이터를 처리할 수 있는 능력을 발휘했습니다. 연구는 또한 아랍어 수화 인식에 대한 기존 접근 방식을 개선하기 위해 ResNet50, MobileNetV3, EfficientNet-B2와 같은 최첨단 모델을 도입했습니다.

- **Performance Highlights**: 연구에서 제안된 시스템은 기존의 최고 모델들과 비교하여 우수한 인식 정확도를 보여주었으며, 특히 의료 및 교육 분야에서 사용될 때 투명한 결정 과정을 보장할 수 있습니다. 해석 가능성은 민감한 분야에서 중요한 요소로 작용하며, 이는 사용자의 신뢰를 높이는 데 기여합니다. 또한, 그 시스템은 다른 수화에 대한 적용 가능성을 가지며, 다양한 데이터 시나리오를 효과적으로 처리할 수 있도록 설계되었습니다.



### Energy Backdoor Attack to Deep Neural Networks (https://arxiv.org/abs/2501.08152)
- **What's New**: 이 논문에서는 딥 뉴럴 네트워크(DNNs)에 대한 에너지 백도어 공격을 설계하였습니다. 기존의 에너지 공격이 대부분 경량화된 컴퓨팅 방식에 집중했던 것과 달리, 본 연구는 백도어를 이용하여 하드웨어 가속기가 에너지 효율성을 저해하도록 유도하는 새로운 접근 방식을 제안합니다. 이 연구는 DNN 아키텍처에 대해 효율적인 에너지 백도어 공격을 실험한 최초의 사례로 평가됩니다.

- **Technical Details**: 제안된 공격 방식은 두 개의 주요 단계로 구성되어 있습니다: 백도어 주입 및 백도어 은닉성입니다. 이 공격은 ResNet-18과 MobileNet-V2 모델을 사용해 실험하였으며, CIFAR-10 및 Tiny ImageNet 데이터셋에서 결과를 검증했습니다. 이 실험을 통해 트리거 샘플에 대해 에너지 소비를 증가시키면서도 일반 입력에 대한 모델 성능을 유지할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 제안된 에너지 백도어 공격은 트리거 샘플에서 에너지 소모를 증가시키는 데 효과적이었습니다. 특히, 모델의 정확성은 정상 입력에 대해서는 변동이 없으면서도 공격이 발생했을 때 에너지 소모가 극대화되는 것을 확인하였습니다. 이러한 결과는 DNN이 에너지 백도어 공격에 취약할 수 있음을 입증하며, 실제 애플리케이션에서 백도어 공격의 은닉성을 높여줄 수 있는 가능성을 시사합니다.



### Bootstrapping Corner Cases: High-Resolution Inpainting for Safety Critical Detect and Avoid for Automated Flying (https://arxiv.org/abs/2501.08142)
- **What's New**: 이 논문은 드론의 자동 비행 중 공중 교통을 감지하기 위한 Detect and Avoid 시스템에서 물체 탐지를 위한 새로운 방법을 제안합니다. 특히, 값 비싼 실제 비행 데이터 대신 합성 데이터 세트를 활용하여 이야기를 해결하려고 하며, 이를 통해 드론 안전성을 높이고자 합니다. 연구진은 Pix2Pix와 Stable Diffusion이라는 두 가지 생성 모델을 활용한 이미지 합성을 통해 고해상도 데이터 세트를 생성하는 방법을 제시합니다.

- **Technical Details**: 연구는 GAN(Generative Adversarial Networks) 및 LDM(Latent Diffusion Models)과 같은 생성 모델을 사용하여 고해상도 이미지에서 항공 물체 인식을 위한 주석이 달린 이미지를 합성하는 것을 목표로 합니다. 제안된 방법은 전체 경관을 합성하는 대신 인페인팅(inpainting) 기술을 사용하여 항공 물체를 삽입하는 과정이 포함됩니다. 이 방식은 보다 효율적으로 데이터를 생성하고, 배경 이미지를 통해 정확한 위치에 물체를 삽입할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 고해상도 데이터 세트를 생성하여 공개 가능성을 높이고, 이것이 기존의 실 데이터로 완전히 훈련된 독립적인 물체 탐지기에 적용되는 성능을 검증합니다. 이 연구는 드론 비행 중의 안전성을 향상시키는 데 기여할 것으로 기대되며, 다양한 항공 상황에서 높은 정확도의 물체 탐지를 가능하게 합니다. 생성된 데이터 세트는 공개적으로 다운로드 가능하여, 다른 연구자들이 이용할 수 있게 됩니다.



### Audio-visual Deepfake Detection With Local Temporal Inconsistencies (https://arxiv.org/abs/2501.08137)
Comments:
          Accepted in ICASSP 2025

- **What's New**: 이 논문은 오디오-비주얼 딥페이크 감지를 위한 새로운 접근 방식을 제안합니다. 기존 연구에서 다루지 않았던 미세한 시간적 불일치를 포착하는 방법을 탐구하고 있으며, 이를 통해 딥페이크 탐지의 정확도를 높이고자 합니다. 특히, 아키텍처 관점과 데이터 합성 전략을 모두 활용하여 이 문제를 해결하고 있습니다.

- **Technical Details**: 제안된 방법은 오디오와 비주얼 데이터를 시간적 거리를 계산하여 미세한 불일치를 포착합니다. 이 과정에서 주의(attention) 메커니즘을 통합하여 배경 소음과 같은 관련 없는 정보를 최소화합니다. 또한, 여러 프레임을 교체하여 생성한 위조 데이터(pseudo-fake data)를 통해 학습 데이터를 증강하고 있습니다.

- **Performance Highlights**: DFDC 및 FakeAVCeleb 데이터셋을 사용하여 본 방법의 성능을 평가하였으며, 기존의 최첨단(SOTA) 방법들과 비교하여 효과성을 입증하였습니다. 이 연구는 오디오-비주얼 딥페이크 탐지의 최신 경향에 기여하며, 미래 연구에 대한 새로운 방향성을 제안합니다.



### SAR Strikes Back: A New Hope for RSVQA (https://arxiv.org/abs/2501.08131)
Comments:
          26 pages, 6 figures

- **What's New**: 본 연구는 Synthetic Aperture Radar (SAR) 이미지를 원거리 감지 시각 질문 응답(Remote Sensing Visual Question Answering, RSVQA) 작업에 도입하여 새로운 방법론을 제안합니다. 기존에 Optical 이미지에 의존했던 RSVQA 시스템에서 벗어나 SAR 데이터를 활용해 보다 정교한 분석을 목표로 합니다. SAR 이미지는 구름 등의 기상 요인에 영향을 덜 받아 보다 일관된 정보 추출이 가능하다는 장점이 있습니다.

- **Technical Details**: 연구에서는 End-to-End RSVQA와 Prompt-RSVQA의 두 가지 파이프라인을 통해 SAR 이미지를 처리하는 방법을 제안합니다. 첫 번째 방법은 추가적인 인코더를 포함하여 SAR 모달리티를 통합하는 엔드 투 엔드 방식이며, 두 번째는 정보 추출 후 자연어로 변환하여 언어 모델에 의존하는 두 단계 방식입니다. 두 방법 모두 SAR와 Optical 데이터의 정보를 활용하여 질문에 대한 답변을 도출하는 연구를 진행합니다.

- **Performance Highlights**: 결과적으로, SAR 이미지만으로도 긍정적인 성과를 보였으며, Optical 이미지와의 융합을 통해 정보가 추가되어 특정 토지 피복 클래스에 대한 질문에 대해 유용한 정보가 생성되었습니다. 결정 수준의 융합(decision level fusion) 방법에서 가장 좋은 결과를 도출하여, SAR과 Optical 이미지의 활용 시 서로의 강점을 극대화할 수 있음을 보여준 점이 이 연구의 큰 성과입니다.



### Revisiting Birds Eye View Perception Models with Frozen Foundation Models: DINOv2 and Metric3Dv2 (https://arxiv.org/abs/2501.08118)
Comments:
          Accepted for publication at the Electronic Imaging - Autonomous Vehicles and Machines Connference 2025

- **What's New**: 이번 논문에서는 Lift-Splat-Shoot(LSS) 및 Simple-BEV 아키텍처와 같은 차량 분할 모델에 DINOv2 및 Metric3Dv2와 같은 파운데이션 모델을 통합하여 훈련 데이터의 양을 줄이면서 성능을 향상시키는 방법을 탐구합니다. 데이터의 절반만 사용하고도 7.4 IoU로 성능을 크게 향상시킨 LSS 모델의 수정 결과가 포함되어 있습니다. Simple-BEV 아키텍처에서는 전통적인 LiDAR 대신 Metric3Dv2의 깊이 정보를 PseudoLiDAR 포인트 클라우드로 적용하여 3 IoU 개선을 달성했습니다.

- **Technical Details**: Lift-Splat-Shoot(LSS) 모델은 짧은 시간 안에 차량의 정밀한 위치를 예측하기 위해 EfficientNet 인코더를 사용하여 이미지로부터 피쳐 맵과 깊이 분포를 생성합니다. 이 연구에서는 EfficientNet을 DINOv2와 Metric3Dv2로 교체하여 기능 향상을 꾀했습니다. DINOv2는 이미지 전체의 특징을 포착하는 데 중점을 두며, Metric3Dv2는 메트릭 깊이를 픽셀 수준에서 제공합니다. 또한, 깊이 이미지를 16x16 단계로 나누어 메트릭 깊이 분포에 맞추는 방법도 설명됩니다.

- **Performance Highlights**: DINOv2와 Metric3Dv2를 채택한 LSS 아키텍처는 훈련 데이터의 절반만으로도 성능을 크게 개선하며, Camera-only 모델 대비 벤치마크에서 우수한 성과를 보였습니다. 깊이 분포를 적절히 변환하여 LSS 구조에 통합함으로써 정교한 차량 분할이 가능해졌습니다. Simple-BEV 모델은 비용 효율적이며, LiDAR나 레이더 데이터 없이 카메라만으로도 우수한 성능을 발휘합니다.



### RoHan: Robust Hand Detection in Operation Room (https://arxiv.org/abs/2501.08115)
Comments:
          12 pages

- **What's New**: 이 논문에서는 수술 환경에서 손 인식을 위한 새로운 접근법인 'RoHan'을 제안합니다. 이는 여러 기록 조건 및 다양한 장갑 색상에서의 손 인식 문제를 해결하기 위해 고급 반자율 도메인 적응 기법을 활용합니다. 특히, 'Artificial Gloves'라는 데이터 증강 기법을 통해 기존 공개 데이터셋을 합성 이미지로 확장하는 방법을 제시하는 것이 특징입니다.

- **Technical Details**: RoHan의 방법론은 두 가지 주요 단계로 구성됩니다: 첫 번째는 'Artificial Gloves' 데이터 증강 전략으로, 이는 실제 손 이미지를 합성 장갑 이미지와 결합하여 데이터셋의 다양성을 증가시킵니다. 두 번째는 반자율 도메인 적응 파이프라인을 통해 실제 수술 환경에서의 탐지 성능을 향상시키는 과정입니다. 이 파이프라인은 예측 세분화와 효율적인 프레임 필터링을 반복적으로 수행하여 정확도를 높입니다.

- **Performance Highlights**: RoHan은 시뮬레이션된 장장 수술과 정맥 이식 수술이라는 두 가지 데이터셋을 통해 평가되었으며, 기존의 대규모 레이블링 및 모델 학습 필요성을 크게 줄입니다. 이러한 방법론은 의료 환경에서 손 인식 기술의 실용적인 구현을 위한 전환점을 마련하며, 더 나아가 메디컬 비전 기술의 발전을 촉진할 것으로 기대됩니다.



### Change Captioning in Remote Sensing: Evolution to SAT-Cap -- A Single-Stage Transformer Approach (https://arxiv.org/abs/2501.08114)
- **What's New**: 이 논문에서는 다중 임시 원격 센싱 데이터에서 변화 설명을 자동으로 생성할 수 있는 SAT-Cap 모델을 제안합니다. 기존의 다단계 융합 전략을 사용하지 않고, 단일 단계의 feature fusion을 기반으로 하여 계산 복잡성을 줄입니다. SAT-Cap은 Spatial-Channel Attention Encoder, Difference-Guided Fusion 모듈 및 Caption Decoder를 통합하여 객체로부터의 의미적 정보를 효과적으로 추출합니다.

- **Technical Details**: SAT-Cap 모델은 Spatial-Channel Attention Encoder를 사용하여 공간 및 채널 정보를 공동으로 모델링합니다. 이를 통해 다중 시간의 원격 센싱 이미지에서 객체의 의미적 정보 추출 능력을 향상시킵니다. 또한 심층 방향 합성곱(depth-wise convolution)을 사용하여 Transformer Encoder 내에서 지역 정보 모델링을 강화하며, shared encoder 매개변수를 활용하여 이중 임시 이미지 쌍 처리 시 계산 효율성을 높입니다.

- **Performance Highlights**: Extensive experiments on the LEVIR-CC와 DUBAI-CC 데이터셋에서 SAT-Cap의 효과가 입증되었습니다. LEVIR-CC 데이터셋에서 140.23%의 CIDEr 점수를 기록했으며 DUBAI-CC 데이터셋에서는 97.74%를 달성하여 기존 최첨단 방법보다 우수한 성능을 보여줍니다. 이러한 결과는 단순한 결합 전략(concatenation strategy)으로도 다단계 feature fusion을 사용한 접근 방식보다 높은 성능을 달성할 수 있음을 시사합니다.



### EarthView: A Large Scale Remote Sensing Dataset for Self-Supervision (https://arxiv.org/abs/2501.08111)
Comments:
          2nd Workshop on Computer Vision for Earth Observation (CV4EO) Applications

- **What's New**: 이 논문은 원격 감지 데이터를 위한 자가 지도 학습에 특화된 종합 데이터셋인 EarthView를 소개합니다. 이 데이터셋은 15 테라픽셀의 세계 원격 감지 데이터로 구성되어 있으며, NEON, Sentinel 및 Satellogic의 다양한 소스에서 이미지를 결합하였습니다. 특히, 1m의 공간 해상도를 가진 새롭게 출시된 데이터를 포함하고 있어, 다양한 해상도의 이미지 데이터 스펙트럼을 제공합니다.

- **Technical Details**: EarthView 데이터셋은 5년간(2017-2022) 수집된 데이터를 가지고 있으며, 다양한 센서에서 얻은 데이터를 파케이 포맷으로 체계적으로 정리하였습니다. 이 논문에서는 EarthMAE라 불리는 맞춤형 마스킹 자동 인코더를 소개하고, 이는 하이퍼스펙트럴, 다중 스펙트럴, 지형 데이터, 분할 맵 및 시간 구조 등 다양한 데이터 양식들을 처리할 수 있도록 개발되었습니다. EarthMAE는 자가 지도 방식으로 훈련되어, Satellogic 데이터를 통해 다운스트림 작업 성능을 향상시킵니다.

- **Performance Highlights**: 이 데이터셋과 모델의 혁신적인 조합은 지구 모니터링을 위한 딥 러닝의 새로운 발전으로 여겨집니다. EarthMAE를 활용하면, 원격 감지 데이터에서 다양한 과제에서 높은 성능을 발휘할 수 있음을 보여주며, 기존의 데이터셋과 비교하여 실제 적용 가능성을 높였습니다. 논문에서는 이러한 접근 방식이 서로 다른 데이터 출처에서 훈련된 MAE 기반 모델이 원격 감지에서의 효과적인 학습을 가능하게 한다고 강조합니다.



### Guiding the classification of hepatocellular carcinoma on 3D CT-scans using deep and handcrafted radiological features (https://arxiv.org/abs/2501.08097)
Comments:
          IEEE ISBI 2025

- **What's New**: 이 연구는 간세포암(hepatocellular carcinoma, HCC) 진단을 자동화하기 위한 새로운 접근법을 제안한다. 기존의 깊은 학습(deep learning) 방법이 CT 스캔에서 HCC를 정확하게 예측하는 데 실패했음을 보여주고, LI-RADS 시스템에서 영감을 받아 성능을 개선하는 2단계 접근 방식을 제안하였다. 본 연구는 HCC의 진단을 위한 새로운 자동화된 방법을 통해 방사선과 의사 사이의 변동성을 줄이는 것을 목표로 한다.

- **Technical Details**: 이 방법은 HCC의 조직학적 확인을 기반으로 하여 실제로 사용되는 LI-RADS 점수 체계를 사전 학습 작업으로 활용한 것이다. LI-RADS의 주요 3가지 특징 예측을 기반으로 한 3개의 딥 러닝 모델을 도입하여 최종 HCC 분류를 안내한다. 또한, 수작업으로 설계한 특징과 LI-RADS 기반의 딥 러닝 방사선학적 특징을 결합하여 분류 성능을 향상시킨다.

- **Performance Highlights**: 제안된 방법은 6점에서 18점까지 AUC(Area Under the Curve) 개선을 달성하였으며, 비전문 방사선과 의사보다 더 우수한 결과를 보였다. 임상 검증을 통해 전문가와 비전문가의 진단 결과와 비교하였으며, 관련 데이터베이스에서의 성능이 모두 우수함을 확인하였다. 이러한 방법은 어려운 HCC 진단을 위한 새로운 패러다임을 제공한다.



### AgentPose: Progressive Distribution Alignment via Feature Agent for Human Pose Distillation (https://arxiv.org/abs/2501.08088)
Comments:
          5 pages, 1 figures

- **What's New**: 본 논문에서는 AgentPose라는 새로운 pose distillation 방법을 제안합니다. 이 방법은 feature agent를 통합하여 teacher와 student 특성 사이의 분포를 점진적으로 정렬함으로써 capacity gap을 극복합니다. 기존 방법들이 teacher 지식의 전파에 주로 집중했던 반면, AgentPose는 student 성능 저하 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: AgentPose는 기본적인 pose distillation 프레임워크에 feature agent를 통합하여, teacher와 student 모델 간의 feature 불일치를 줄이는 과정을 다룹니다. 이 feature agent는 noise perturbation과 dynamic distribution modulation을 통해 학생의 knowledge를 teacher 지식에 맞추어 조정합니다. 또한, feature agent에 입력되기 전에 autoencoder를 사용하여 feature의 차원을 줄임으로써 계산 부담을 경감시킵니다.

- **Performance Highlights**: 실험 결과, AgentPose는 기존의 pose distillation 방법들보다 뛰어난 성능을 보이며, 특히 teacher와 student 간의 capacity gap이 큰 경우에 효과적입니다. AgentPose는 경량 모델 아키텍처를 기반으로 하여 효율적인 지식 전파를 가능하게 하고, compact 모델이 우수한 성능을 발휘하도록 도와줍니다.



### Benchmarking Vision Foundation Models for Input Monitoring in Autonomous Driving (https://arxiv.org/abs/2501.08083)
- **What's New**: 본 논문에서는 심층 신경망(DNN)이 복잡한 오픈 월드(domain) 환경에서 데이터 분포의 변화에 대처하는 데 한계가 있음을 지적하며, 특히 자율주행(AD) 같은 분야에서의 효과적인 OOD(out-of-distribution) 모니터링 시스템의 중요성을 강조합니다. 기존의 OOD 클래스 분류 방법은 특정 도메인에서 제한적이거나 OOD 샘플을 요구하는 등의 문제점이 있습니다. 이에 새로운 방법론으로 모든 종류의 분포 변화를 탐지할 수 있는 모델 불가지론적 방법을 제안합니다.

- **Technical Details**: 우리는 Vision Foundation Models (VFM)을 피쳐 추출기로 사용하여, 훈련 데이터의 피쳐 분포를 모델링한 후 새로운 샘플에 대해서는 그 밀도를 ID(in-distribution) 점수로 사용합니다. 이 방법은 4개의 VFM과 20개의 베이스라인을 비교하는 광범위한 벤치마크를 통해 검증되었습니다. 연구 결과, VFM 피쳐 인코딩이 OOD 모니터보다 우수한 성능을 발휘하며, 복잡한 비전 과제에서 신뢰할 수 있는 안전 모니터를 실현할 가능성을 보여줍니다.

- **Performance Highlights**: 상세한 실험에서 VFM 인코딩이 통합된 분포 변화 탐지에서 명백한 우수성을 인정받았으며, 복잡한 입력 도메인에서도 OOD 샘플을 정확하게 식별할 수 있었습니다. 특히, 정교한 아키텍처가 더 큰 잠재 공간의 차원보다 일관되게 높은 성능을 발휘했습니다. 이 연구는 AD 시스템에서의 입력 모니터링 방법을 평가하기 위한 포괄적인 벤치마크 프레임워크를 제공하여 다양한 실험 시나리오를 지원합니다.



### Evaluating Human Perception of Novel View Synthesis: Subjective Quality Assessment of Gaussian Splatting and NeRF in Dynamic Scenes (https://arxiv.org/abs/2501.08072)
- **What's New**: 이 논문에서는 Gaussian Splatting (GS)과 Neural Radiance Fields (NeRF)를 포함하는 새로운 NVS (Novel View Synthesis) 기술의 주관적 품질 평가를 위한 실험을 수행했습니다. 이 연구는 동적 장면에서의 NVS 방법이 미치는 영향을 탐구했으며, 360°, 정면 및 단일 관점의 비디오 클립을 활용하여 보다 풍부한 실세계 장면을 제공했습니다. 또한, GS 기반 방법과 NeRF 기반 방법의 성능을 비교하여 두 가지 주관적 평가 실험을 통해 다양한 시청 경로의 영향을 이해하는 데 도움을 주고 있습니다.

- **Technical Details**: NVS의 품질 평가 방법으로는 SAMVIQ(Subjective Assessment Methodology for Video Quality)를 사용하여 동적 장면에서 NeRF 기반 및 GS 기반 방법의 주관적 품질 평가 데이터셋을 구축했습니다. 이 데이터셋은 총 13개의 실제 장면을 포함하며, 질적 평가를 위해 360°, 정면-view 및 단일-view의 이동형 PVS를 포괄합니다. 기존의 연구에서 주로 다루어지지 않았던 동적 장면에서의 NVS 방법의 주관적 품질 평가 또한 이루어졌습니다.

- **Performance Highlights**: 주관적 테스트를 통해 두 가지 NVS 방법의 효과성을 검증하였고, 그 결과는 기존 NVS 방법의 한계를 나타내는 중요한 통찰을 제공합니다. 또한, 제안된 데이터셋에 대한 다양한 최신 객관적 지표의 벤치마크를 수행하여, 기존 방법들이 여전히 주관적 품질 평가를 정확히 반영하는 데 어려움을 겪고 있음을 강조합니다. 이 연구는 NVS 기술의 품질 평가를 향상시키기 위한 향후 연구 방향을 제시합니다.



### Skeleton and Font Generation Network for Zero-shot Chinese Character Generation (https://arxiv.org/abs/2501.08062)
Comments:
          36 pages, 10 figures

- **What's New**: 이번 연구에서는 중국 문자 폰트 생성을 위한 새로운 Skeleton and Font Generation Network (SFGN)을 제안하였습니다. 기존의 폰트 생성 방법들이 가진 구조적 편향을 완화하고, 한자가 아닌 내용없이 폰트를 생성이 가능하도록 하는 접근 방식을 도입하였습니다. 이 연구는 misspelled (잘못 철자된) 문자 생성의 중요성을 강조하며, 이러한 오류가 학생들이 한자를 배우는 데 도움이 된다는 것을 입증합니다.

- **Technical Details**: 제안된 SFGN 구조는 두 가지 단계로 나뉘어 진행됩니다. 첫 번째 단계인 skeleton builder는 컴포넌트 수준의 캡션을 통해 한자의 형태를 생성합니다. 이후 font generator가 스타일 이미지를 기반으로 스타일 특성을 추출하고 이를 내용 특성과 결합하여 최종적으로 한자를 생성합니다. 이 과정에서 transitive-attention mechanism을 사용하여 스타일과 내용을 세밀하게 정렬하는 방식을 도입했습니다.

- **Performance Highlights**: SFGN은 기존 최첨단 방법들과 비교해 우수한 성능을 보이며, 특히 잘못된 철자가 포함된 문자 생성에서 그 효과가 두드러집니다. 또한 학습을 위한 데이터 증강 실험에서, 잘못된 철자 문자의 활용이 오차 수정 작업에서 성능 향상에 큰 기여를 하였음을 입증했습니다. 이는 교육 현장에서 잘못된 문자 생성의 높은 응용성과 실제 가치를 보여줍니다.



### Exploring visual language models as a powerful tool in the diagnosis of Ewing Sarcoma (https://arxiv.org/abs/2501.08042)
Comments:
          11 pages, 5 figures, 2 tables. Oral presentation at KES-InMed 2024 held in Madeira, Portugal

- **What's New**: Ewing의 육종(Ewing's sarcoma, ES)에 대한 새로운 접근 방식을 제시하는 이 연구는, histopathological 이미지 분석에서 인공지능 기술의 가능성을 탐구하고 있습니다. 특히, 다양한 pre-training 전략을 통해 ES를 유사한 형태를 가진 다른 연부 조직이나 뼈 육종과 구분하고자 하는 것을 목표로 합니다. Vision-language supervision (VLS)를 통해 진단 정확도를 크게 향상시킬 수 있음을 보여주며, 이는 AI 기반 시스템이 ES 진단에 기여할 수 있는 가능성을 시사합니다.

- **Technical Details**: 이 연구는 Multiple Instance Learning (MIL) 방법론을 활용하여 Ewing의 육종을 진단하는 새로운 패러다임을 제안합니다. Histological 이미지에서 높은 수준의 특징적 표현을 추출하기 위해 Pathology language-image pretraining (PLIP)과 Transformer 기반의 강력한 임베딩 집계기를 사용했습니다. 최종적으로, 여러 데이터셋에서 학습된 VGG 기반 아키텍처와 비교하여 제안하는 방법의 경쟁력을 입증하였습니다.

- **Performance Highlights**: 실험 결과, VLS의 활용으로 진단 정확도가 크게 향상되었으며, 이는 모델의 예측 클래스 정확도뿐만 아니라 훈련 가능 매개변수의 수와 계산 비용을 획기적으로 감소시켰습니다. 특히, Ewing의 육종을 다른 연부 조직 및 뼈 육종과 구분하기 위한 효과적인 방법임을 강조하며, 이는 향후 진단 정확성을 더욱 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### Robust Low-Light Human Pose Estimation through Illumination-Texture Modulation (https://arxiv.org/abs/2501.08038)
Comments:
          5 pages, 2 figures, conference

- **What's New**: 본 논문에서는 저조도 조건에서 인간 자세 추정을 위한 새로운 주파수 기반 프레임워크를 제안합니다. 기존 방법들에 비해 이미지 전체를 균일하게 향상시키는 것이 아니라, 과제와 관련된 정보에 집중하는 'divide-and-conquer' 접근 방식을 사용합니다. 이를 통해 스며드는 저조도 이미지에서 중요 세부 정보를 효과적으로 추출하고 향상시킬 수 있습니다.

- **Technical Details**: 우리의 방법은 세 가지 단계로 구성되어 있습니다: 분해(decomposition), 향상(enhancement), 재구성(reconstruction)입니다. 첫 번째 단계에서 이미지를 다중 스케일 고주파 성분과 저주파 베이스로 분해하는 계층적 라플라시안 피라미드를 사용합니다. 그 후 저주파 경로에서 동적 조명 보정(dynamic illumination correction)을 통해 의미 정보(semantic clarity)를 향상시키고, 고주파 콘텐츠의 고차원 구조를 활용하여 저랭크(dimension reduction) 효과에 따른 다중 스케일 저랭크 디노이징(low-rank denoising)을 수행하여 텍스처 세부 정보를 정제합니다.

- **Performance Highlights**: 중요한 실험을 통해 제안된 방법은 다양한 저조도 시나리오에서 최신 기법들에 비해 월등한 성능을 입증했습니다. 이 프레임워크는 높은 품질의 표현을 제공함으로써 인간 자세 추정 모델의 성능 향상에 기여하고 있습니다. 동적 조명 보정과 저랭크 디노이징을 통해 저조도 조건에서도 강력한 인간 자세 추정이 가능함을 보여줍니다.



### DisCoPatch: Batch Statistics Are All You Need For OOD Detection, But Only If You Can Trust Them (https://arxiv.org/abs/2501.08005)
- **What's New**: 이 논문에서는 데이터 분포의 미세한 변화인 covariate shifts를 탐지하여 OOD(Out-of-Distribution) 감지 성능을 향상시킬 수 있다는 가설을 제기합니다. 기존의 OOD 탐지 방식은 주로 semantic shifts와 domain shifts에 중점을 두었으나, 본 연구는 이러한 covariate shifts에 대한 이해를 바탕으로 새로운 접근 방식을 제안합니다. 특히, Batch Normalization(BN) 조건에서의 adversarial 설정을 활용하여 OOD 탐지를 위한 새로운 프레임워크인 DisCoPatch를 소개합니다.

- **Technical Details**: DisCoPatch는 unsupervised Adversarial Variational Autoencoder(VAE) 프레임워크로서, 모델이 이미지의 다양한 패치를 기반으로 일관된 데이터 분포에서 작동하도록 설계되었습니다. 기법적으로, DisCoPatch는 변 reconstructed된 이미지와 생성된 이미지의 suboptimal outputs를 OOD 샘플로 활용하여 discriminator의 학습을 진행합니다. 이런 방식으로 in-distribution 샘플과 covariate shifts 간의 경계를 더 엄격하게 설정함으로써 OOD 탐지 성능이 향상됩니다.

- **Performance Highlights**: DisCoPatch는 공공 OOD 탐지 벤치마크에서 state-of-the-art 성능을 달성하며, ImageNet-1K(-C) 데이터셋에서 95.5%의 AUROC를 기록했습니다. 또한, Near-OOD 탐지에서도 95.0%의 성능을 보여줍니다. 25MB의 소형 모델로 설계되어 기존 방법보다 낮은 지연 시간으로 높은 OOD 탐지 성능을 제공하여 실제 OOD 탐지에 효과적이고 실용적인 솔루션임을 증명합니다.



### Combining imaging and shape features for prediction tasks of Alzheimer's disease classification and brain age regression (https://arxiv.org/abs/2501.07994)
- **What's New**: 본 연구에서는 MRI에서 추출된 이미지 및 형태 특성을 결합하여 뇌 나이 예측과 알츠하이머병(AD) 분류의 임상적 관련성 있는 작업을 수행합니다. 제안된 모델은 ResNet으로 추출된 이미지 임베딩과 맞춤형 그래프 신경망(Graph Neural Network)에서 얻은 형태 임베딩을 융합합니다. 이 연구는 기존의 이미지 기반 접근 방식과 형태 기반 접근 방식을 통합하여 더 나은 예측 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구는 CamCAN, IXI 및 OASIS3의 3개의 공개 데이터 세트를 사용하여 이미지 전처리 및 고유한 그래프 신경망 아키텍처를 통해 뇌의 15가지 구조에 대한 형태 임베딩을 추출합니다. ResNet-18 아키텍처를 사용하여 T1 가중치 MRI 이미지에서 특징을 추출하고, 그래프 기반 모델로써 다중 그래프 아키텍처를 설계하였습니다. 마지막으로, 두 가지 모델에서 추출한 특징을 결합하여 예측 성능을 향상시킵니다.

- **Performance Highlights**: 결과적으로, AD 분류에 대한 융합 모델이 기존 모델을 능가하는 것으로 관찰되었으며, 이로 인해 임상 응용의 중요한 분야에서 성능 향상을 보여줍니다. 또한, 뇌 나이 회귀 작업에서도 약간의 개선이 있으나 독립 모델이 더 나은 성능을 보일 때도 있습니다. ROC 곡선을 사용하여 모델의 성과를 분석함으로써, 융합 모델이 디지털 진단 도구로서의 가능성을 더욱 demonstrate합니다.



### GAC-Net_Geometric and attention-based Network for Depth Completion (https://arxiv.org/abs/2501.07988)
Comments:
          13pages,4 figures, 2 tables

- **What's New**: 이 논문은 깊이 완성(depth completion) 네트워크인 CGA-Net을 제안합니다. CGA-Net은 채널 주의 메커니즘(channel attention mechanism)과 3D 글로벌 특징 인식(global feature perception)을 결합하여, 희박한 LiDAR 깊이 측정을 고품질의 밀도 깊이 맵으로 완성하는 혁신적인 접근법을 제공합니다. 이 방법은 특히 복잡한 경계와 희박한 영역에서 기존 방법의 한계를 극복하려는 목표를 가지고 있습니다.

- **Technical Details**: 논문에서 제안하는 CGA-Net은 PointNet++을 활용하여 희박한 깊이 맵에서 글로벌 3D 기하학적 특성을 추출합니다. 이렇게 추출된 특성은 저선 LiDAR 데이터의 장면 인식 능력을 향상시키는데 기여합니다. 또한, 멀티모달 특징 융합 모듈(multimodal feature fusion module)을 통해 희박한 깊이, RGB 이미지 및 3D 기하학적 특성을 효율적으로 통합합니다.

- **Performance Highlights**: KITTI 깊이 완성 데이터셋에 대한 실험 결과, CGA-Net은 밀도 깊이 맵의 예측 정확성을 유의미하게 개선하여 새로운 최첨단(SOTA) 성능을 달성했습니다. 특히, 희박하고 복잡한 장면에 대한 강력한 견고성을 보여주어 실제 자율 주행 환경에서의 활용 가능성을 높이고 있습니다.



### Threshold Attention Network for Semantic Segmentation of Remote Sensing Images (https://arxiv.org/abs/2501.07984)
- **What's New**: 본 논문에서는 새로운 Threshold Attention Mechanism (TAM)을 제안하여 기존 self-attention 메커니즘의 계산 복잡성을 줄이고, 여러 지역 간의 상관관계를 더 잘 모델링할 수 있음을 보입니다. 제안된 TAM을 기반으로 하는 Threshold Attention Network (TANet)는 풍부한 세부 정보와 명확한 경계가 있는 분할 맵을 생성합니다. TANet은 attentional feature enhancement module (AFEM)과 threshold attention pyramid pooling module (TAPP)을 포함하여 샤로우(features)와 딥(deep) 특성을 효과적으로 통합합니다.

- **Technical Details**: TAM은 입력 특성을 글로벌 양자화하여 생성된 정보를 기반으로 주어진 임계값을 통해 유사한 픽셀 집합의 주의 가중치를 결정합니다. 이 메커니즘은 인접 픽셀 간의 종속성을 고려하고, 주의 가중치를 계산하는데 필요한 컴퓨팅 복잡성을 선형적으로 감소시킵니다. 결과적으로, TAM은 특징 표현을 향상시키고 무의미한 중복을 줄이며, 지역 간의 의존성을 효과적으로 모델링합니다.

- **Performance Highlights**: ISPRS Vaihingen 및 Potsdam 데이터셋에서의 광범위한 실험을 통해 TANet은 최신 최첨단 모델들에 비해 우수한 성능을 보였습니다. 논문에 따르면, 제안한 TANet은 높은 정확도와 정밀도를 가진 세그멘테이션 결과를 얻었으며 기존 방법들이 가진 한계를 뛰어넘습니다. 이러한 결과는 다양한 응용 분야에서의 사용 가능성을 나타냅니다.



### V-Trans4Style: Visual Transition Recommendation for Video Production Style Adaptation (https://arxiv.org/abs/2501.07983)
- **What's New**: V-Trans4Style라는 혁신적인 알고리즘이 동적 비디오 컨텐츠 편집을 위해 개발되었습니다. 이 알고리즘은 다큐멘터리, 드라마, 영화 및 특정 YouTube 채널의 비디오 제작 기술과 같은 다양한 제작 스타일에 비디오를 적응시키는 데 도움을 줍니다. 새로운 AutoTransition++ 데이터셋을 통해 우리의 방법의 효율성을 입증하며, 이 데이터셋은 6,000개 비디오를 포함하여 다양한 제작 스타일로 분류됩니다.

- **Technical Details**: V-Trans4Style은 transformer 기반의 인코더-디코더 네트워크를 사용하여 비디오에서 적절한 비주얼 전환을 추천합니다. 스타일 조정 모듈은 이 모델을 활용하여 비디오 스타일 특성을 반영하는 비주얼 전환 시퀀스를 생성합니다. 우리는 AutoTransition++ 데이터셋에서의 실험을 통해 우리의 인코더-디코더 모델이 기존의 전환 추천 방법보다 10%에서 80%까지 개선되었음을 보여줍니다.

- **Performance Highlights**: 우리는 우리의 스타일 조정 모듈이 다른 방법들과 비교했을 때 비디오 제작 스타일 특성을 12% 향상시켜 포착하는 것을 보여줍니다. V-Trans4Style은 비디오 제작 스타일의 다양한 특성 이해를 위한 기초 자료로 활용될 것으로 기대됩니다.



### Facial Dynamics in Video: Instruction Tuning for Improved Facial Expression Perception and Contextual Awareness (https://arxiv.org/abs/2501.07978)
- **What's New**: 이번 논문에서는 동적 얼굴 표정 캡션(Dynamic Facial Expression Caption, DFEC)을 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 5,033개의 수작업으로 주석이 달린 고품질 비디오 클립으로 구성되어 있으며, 700,000개 이상의 토큰을 포함하여 비디오 MLLM의 얼굴 표정 인식을 향상시킵니다. 또한, 얼굴 추적 기술을 활용한 새로운 모델 FaceTrack-MM을 제안하여, 다양한 안면 표정을 정확하게 캡션할 수 있는 능력을 강화합니다. 이를 통해 얼굴 표정 이해의 한계를 극복하려는 노력이 이루어집니다.

- **Technical Details**: 논문에서는 새로운 평가 지표인 시간적 이벤트 매칭(Temporal Event Matching, TEM)을 소개하며, 이를 통해 생성된 텍스트의 의미적 일관성과 시간적 순서를 평가합니다. 또한, DFEC 작업을 위해 설계된 FEC-Bench라는 벤치마크를 구축하여, 15개의 오픈소스 및 상용 모델의 성능을 비교할 수 있도록 하였습니다. 기존의 얼굴 행동 단위 탐지(Facial Action Unit Detection, FAUD)와는 달리, 이 연구는 자연어를 사용하여 얼굴 정보를 설명하는 방법을 채택하고 있습니다.

- **Performance Highlights**: FaceTrack-MM 모델은 복잡한 다중 인물 장면에서도 주요 캐릭터의 얼굴 표정을 추적하고 집중하는 데 있어 우수한 성능을 입증하였습니다. 특히 비디오 입력을 처리하는 과정에서 제한된 토큰 수에도 불구하고, 얼굴 지역의 세부 정보를 정확하게 인코딩하여 캡션의 품질을 향상시키는 데 기여합니다. 새로운 데이터셋과 모델을 통해 비디오 MLLM의 인식 능력을 크게 향상시킬 것으로 기대되며, 향후 연구의 기반을 마련합니다.



### SkipClick: Combining Quick Responses and Low-Level Features for Interactive Segmentation in Winter Sports Contexts (https://arxiv.org/abs/2501.07960)
Comments:
          4 figures, 6 tables, 12 pages

- **What's New**: 이번 논문에서는 겨울 스포츠 맥락에서의 상호 작용 세분화(interactive segmentation)를 위한 새로운 아키텍처를 제시합니다. 특히, 사용자의 클릭을 통해 세분화를 안내받는 시스템을 개발하여, 세분화 마스크의 질을 높이는 데 중점을 두었습니다. 기존 시스템의 반응 속도를 개선하여 겨울 스포츠 장비인 WSESeg 데이터셋에서 우수한 성능을 보였음을 강조합니다.

- **Technical Details**: 이 연구에서 제안된 모델은 SkipClick이라 불리며, 신속한 응답 시간과 과적합(overfitting)을 방지하는 데 중점을 둡니다. 모델은 세분화 작업에서 사진의 세부 구조를 처리하기 위해 저수준 이미지 특징 텐서를 사용하는 방식으로 설계되었습니다. 또한, 사용자 안내에 따라 이전 마스크를 지속적으로 개선하는 방식으로 작동하며, 이를 통해 사용자는 신속하게 객체 세분화 마스크를 생성할 수 있습니다.

- **Performance Highlights**: 모델은 WSESeg 데이터셋에서 평균 NoC@85 지표에서 SAM과 HQ-SAM을 각각 2.336 및 7.946 클릭 수로 초과 달성했으며, HQSeg-44k 데이터셋에서는 NoC@90 6.00과 NoC@95 9.89로 최첨단 결과를 나타냈습니다. 특히, 비교 대상인 SAM은 겨울 스포츠 장비 도메인에서 일반화 성능이 낮았지만, SkipClick은 정규 소비자 이미지 데이터셋에서도 경쟁력 있는 성능을 보였습니다.



### Robust Hyperspectral Image Panshapring via Sparse Spatial-Spectral Representation (https://arxiv.org/abs/2501.07953)
Comments:
          Submitted to IGARSS 2025

- **What's New**: 본 논문은 하이퍼스펙트럴 이미지의 팬셔프닝(pansharpening)을 위한 새로운 프레임워크인 S$^{3}$RNet을 소개합니다. 이 프레임워크는 저해상도 하이퍼스펙트럴 이미지(LRHSI)와 고해상도 멀티스펙트럴 이미지(HRMSI)를 희소(spare) 공간-스펙트럴 표현(spectral representation)을 통해 효과적으로 결합합니다. S$^{3}$RNet은 기존의 접근법과 달리 모든 특징(feature)을 동일하게 취급하지 않고, 서로 다른 공간과 스펙트럴 스케일(scale)에서 보완적인 특성을 캡처합니다.

- **Technical Details**: S$^{3}$RNet의 핵심은 Multi-Branch Fusion Network (MBFN)입니다. 이 네트워크는 평행한 브랜치를 사용하여 다양한 공간적 및 스펙트럴 스케일에서 특징을 캡처하고, Spatial-Spectral Attention Weight Block (SSAWB)을 통해 동적으로 특징의 가중치를 조정하여 희소한 표현을 유지합니다. 또한, Dense Feature Aggregation Block (DFAB)을 통합하여 입력된 특징들을 밀집 연결(dense connectivity) 패턴을 통해 효율적으로 집계(aggregate)합니다.

- **Performance Highlights**: 종합적인 실험 결과, S$^{3}$RNet은 여러 평가 지표에서 최신 기술(state-of-the-art) 성능을 달성했습니다. 특히, 노이즈가 많은 상황에서도 높은 재구성 품질(reconstruction quality)을 유지하는 강점을 보였습니다. 연구 결과는 오픈 소스로 제공될 예정입니다.



### VENOM: Text-driven Unrestricted Adversarial Example Generation with Diffusion Models (https://arxiv.org/abs/2501.07922)
- **What's New**: VENOM은 고품질의 무제한 적대적 예제(Unrestricted Adversarial Examples, UAE) 생성을 위한 최초의 텍스트 기반 프레임워크로 소개됩니다. 이 방법은 이미지 콘텐츠 생성과 적대적 합성을 단일 역확산(Reverse Diffusion) 과정으로 통합하여 높은 충실도의 적대적 예제를 생성하면서도 공격 성공률(Attack Success Rate, ASR)을 유지합니다. 또한, 적응형(adaptive) 적대적 가이드라인 전략을 도입하여 생성된 예제가 자연 이미지의 분포와 정렬되도록 보장하고, 이미지 품질 저하를 방지합니다.

- **Technical Details**: VENOM은 적대적 가이드를 주입하는 과정에서 안정성과 품질을 확보하기 위한 적응형 제어 전략을 포함합니다. 안정적인 제어를 통해 자연 이미지의 분포에 맞춰 고품질의 무제한 및 자연 적대적 예제(Natural Adversarial Examples, NAE)를 생성합니다. 텍스트 기반의 프롬프트를 사용하여 생성된 적대적 이미지는 사용자가 정의한 시각적 콘텐츠와 목표 클래스를 바탕으로 제작될 수 있으며, 이는 적대적 공격을 더욱 유연하게 합니다.

- **Performance Highlights**: 실험 결과 VENOM은 백박스 모델(White-box Models)에서 거의 100%의 공격 성공률을 달성하며, 방어 메커니즘에 대해서도 강력한 ASR을 유지합니다. UAEs는 참조 이미지를 왜곡하여 생성되는 반면, NAEs는 랜덤 노이즈에서 발생하는 것으로 구분됩니다. VENOM은 기존 diffusion 기반 방법들에 비해 첨단의 발전을 이루었으며, 심층 학습 모델의 취약성을 연구하는 데에 중요한 도구가 될 것입니다.



### Cloud Removal With PolSAR-Optical Data Fusion Using A Two-Flow Residual Network (https://arxiv.org/abs/2501.07901)
- **What's New**: 이 연구에서는 Polarimetric Synthetic Aperture Radar (PolSAR)와 Optical 데이터 융합을 통해 구름 없는 이미지를 재구성하는 새로운 알고리즘인 PODF-CR을 제안합니다. 이 알고리즘은 인코딩 모듈과 디코딩 모듈로 구성되어 있으며, 다중 모달 이미지 간의 효율적인 Fusion을 위해 크로스 스킵 연결을 기반으로 한 fusion block을 도입합니다. 특히, 역사적 SAT 데이터와 PolSAR 데이터를 결합한 OPT-BCFSAR-PFSAR 데이터셋을 활용하여 효과적인 구름 제거 성능을 달성합니다.

- **Technical Details**: PODF-CR 알고리즘의 인코딩 모듈은 PolSAR 이미지와 Optical 이미지의 특징을 효과적으로 추출하는 두 개의 병렬 브랜치를 포함하고 있습니다. 각 PolSAR 브랜치에서는 이미지 노이즈를 줄이기 위해 동적 필터를 도입하여 스펙클 노이즈를 효과적으로 처리합니다. 또한, 멀티스케일 이미지 정보를 추출하기 위해 디코딩 모듈에서도 멀티스케일 컨볼루션 기술을 적용하였습니다.

- **Performance Highlights**: 실험 결과, PODF-CR은 기존의 방법들에 비해 정성적 및 정량적 평가 모두에서 더 뛰어난 성능을 보여주었습니다. 이 알고리즘은 구름에 의한 데이터 손실 문제를 효과적으로 해결하고, 고해상도의 구름 없는 이미지를 재구성하는 데 있어 강력한 가능性을 보여줍니다. 이러한 성과는 대규모 지구 관측 및 원격 감지 응용에 중요한 기여를 할 것으로 기대됩니다.



### Demographic Variability in Face Image Quality Measures (https://arxiv.org/abs/2501.07898)
- **What's New**: 이번 연구에서는 ISO/IEC 29794-5 국제 표준에 포함된 얼굴 이미지 품질 평가(FIQA) 알고리즘의 인구 통계적 차이를 조사했습니다. 세 가지 인구 통계 변수인 나이, 성별, 피부톤을 기준으로 FIQA 알고리즘의 성과를 평가하는 방법을 제시했습니다. 대부분의 품질 척도에서 특정 인구 집단에 대한 편향이 발견되지 않았으며, 피부톤에 따라 두 가지 품질 척도가 상당한 변화를 보였습니다.

- **Technical Details**: 연구에서는 4개의 데이터 세트(FRLL, FRGCv2, LFW, MST-E)를 사용하여 FIQA의 품질 척도를 평가했습니다. 각 데이터 세트는 다양한 피부톤, 성별, 나이, 인종의 이미지를 포함하고 있으며, MST-E 데이터 세트는 Monk Skin Tone Scale(MST)을 기반으로 한 레이블이 포함되어 있습니다. FIQA 척도를 평가하기 위해 Open Source Face Image Quality(OFIQ) 프레임워크를 사용하여 각 품질 척도를 구현하였습니다.

- **Performance Highlights**: 실험 결과, 대부분의 FIQA 척도에서 피부톤 그룹 간에 뚜렷한 차이가 나타나지 않았습니다. 그러나 dynamic range와 luminance mean 품질 측정치는 피부톤에 따라 상대적으로 높은 품질 값을 부여받는 경향이 있었습니다. 이러한 결과는 FIQA 알고리즘이 피부톤에 민감할 수 있음을 시사하며, 향후 연구에서 개선책을 마련하는 것이 중요함을 보여 줍니다.



### Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding (https://arxiv.org/abs/2501.07888)
- **What's New**: Tarsier2는 최신 대형 비전-언어 모델(LVLM)로, 세 가지 주요 업그레이드를 통해 자세하고 정확한 비디오 설명을 생성하는 우수한 성능을 보여줍니다. 첫째, 사전 학습 데이터 양을 1100만에서 4000만 비디오-텍스트 쌍으로 확장해 데이터의 양과 다양성을 증가시켰습니다. 둘째, 감독된 세분화된 시간 정렬(fine-grained temporal alignment) 기법을 도입했고, 셋째, DPO(Delegated Preference Optimization) 훈련을 통한 모델 기반 샘플링 기법을 사용했습니다.

- **Technical Details**: Tarsier2는 비전 인코더(vision encoder), 비전 어댑터(vision adaptor), 대형 언어 모델(LLM)로 구성된 단순한 모델 아키텍처를 채택합니다. 세 단계의 훈련 절차로는 사전 학습(pre-training), 감독된 세분화된 미세 조정(supervised fine-tuning), 강화 학습(reinforcement learning)으로 구성됩니다. 또한, 150K개의 비디오 설명 인스턴스를 포함하는 데이터셋을 구축하여 SFT 단계에서 모델의 시간 세분화 정렬을 위한 감독을 제공합니다.

- **Performance Highlights**: Tarsier2-7B는 비디오 설명 과제에서 기존의 선도적인 상용 모델들, 즉 GPT-4o와 Gemini-1.5-Pro를 능가하며, DREAM-1K 벤치마크에서 F1 점수가 각각 2.8% 및 5.8% 향상되었습니다. 전반적인 성능 평가에 따르면, Tarsier2-7B는 인간 평가에서도 GPT-4o보다 8.6%, Gemini-1.5-Pro보다 24.9% 더 나은 성능을 보였습니다. Tarsier2-7B는 비디오 질문 응답, 비디오 그라운딩 및 환각 테스트와 같은 15개의 공개 벤치마크에서 새로운 SOTA 결과를 기록하며, 다재다능한 일반ist 비전-언어 모델로서의 성능을 입증합니다.



### Make-A-Character 2: Animatable 3D Character Generation From a Single Imag (https://arxiv.org/abs/2501.07870)
Comments:
          Technical Report

- **What's New**: 이번 보고서는 게임 개발 및 디지털 인간 어플리케이션에 적합한 고품질 3D 캐릭터를 생성하는 방법인 Make-A-Character 2를 소개합니다. 이 시스템은 단일 초상 사진에서 3D 캐릭터를 생성할 수 있도록 설계되었으며, 기존 버전의 여러 중요한 개선 사항을 포함하고 있습니다. 특히, IC-Light 방법을 통해 비이상적인 조명을 보정하고, 신경망 기반의 색상 보정을 통해 피부 톤을 harmonized하여 훌륭한 품질의 캐릭터를 생성할 수 있습니다.

- **Technical Details**: Make-A-Character 2는 3D 캐릭터 생성을 위해 Hierarchical Representation Network(HRN)를 활용하여 고주파수의 얼굴 구조를 포착합니다. 또한, 적응형 스켈레톤 보정을 통해 정확하고 표현력이 풍부한 얼굴 애니메이션을 가능하게 합니다. 전체 이미지에서 3D 캐릭터로의 변환 과정은 2분 이내에 완료되며, 트랜스포머 아키텍처를 활용하여 실시간 대화가 가능한 얼굴 및 제스처 액션을 생성합니다.

- **Performance Highlights**: 이번 시스템에서는 3D 캐릭터의 머리 및 얼굴 부분을 더 정밀하게 재구성하기 위해 HRN을 사용하고, 색상과 조명을 일치시키기 위한 신경망 학습을 적용했습니다. 또한, 실제 애니메이션 효과를 위해 메타휴먼의 제어 rig을 기반으로 얼굴 표정을 애니메이션화하고, 경험이 풍부한 애니메이터들이 모션 캡처 데이터를 수정하여 자연스럽고 현실적인 제스처를 생성합니다. 이러한 기술들은 대화형 AI 아바타 제품에 통합되어 있습니다.



### deepTerra -- AI Land Classification Made Easy (https://arxiv.org/abs/2501.07859)
- **What's New**: deepTerra는 기계 학습과 위성 이미지를 활용해 지표 분류를 쉽게 할 수 있도록 설계된 포괄적인 플랫폼입니다. 데이터 수집, 이미지 증강, 훈련, 테스트 및 예측 모듈을 포함하여 이미지 분류 작업의 전체 워크플로우를 간소화합니다. 이 논문에서는 deepTerra의 기능과 다양한 연구 분야에서의 적용 사례를 보여주며, 향후 발전 방향도 논의합니다.

- **Technical Details**: deepTerra는 기계 학습을 통해 이미지를 분류하는 다양한 단계를 지원하는 도구 모음입니다. 데이터 수집 모듈은 기존 이미지를 가져와 이미지 패치를 추출하고, 자동으로 지리적 좌표를 기록하여 공간적 참조를 제공합니다. 이미지 증강은 회전, 플리핑, 이동 등의 기하학적 변환을 통해 데이터셋을 확장하여 모델의 강건성과 성능을 개선합니다.

- **Performance Highlights**: deepTerra는 VGG, ResNet, Inception 등 다양한 CNN 아키텍처를 지원하여 사용자들이 다양한 이미지 분류 작업을 수행할 수 있도록 합니다. 훈련 과정에서 정확도와 F1 스코어 등의 성과 지표를 시각화하고, 훈련 완료 후에는 혼동 행렬 및 키 성능 메트릭을 포함한 상세한 결과를 제공합니다. 이 툴은 새로운 라벨 없는 데이터에 대한 예측도 지원하여, 최종 목표인 이미지 분류를 효과적으로 수행합니다.



### State-of-the-Art Transformer Models for Image Super-Resolution: Techniques, Challenges, and Applications (https://arxiv.org/abs/2501.07855)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 이미지 초해상도(Image Super-Resolution, SR) 분야에서 transformer 기반 방법의 발전을 소개하고 있습니다. 기존의 CNN 및 GAN과 같은 딥러닝 방식보다 더욱 높은 품질의 이미지를 재구성할 수 있는 가능성을 제시하고 있습니다. 특히, 이전 방법들의 한계인 제한된 수신 필드(receptive fields)와 고주파 세부사항 복구의 어려움을 해결하는 데 기여하고 있습니다.

- **Technical Details**: 이 논문에서는 전통적인 네트워크와 결합된 transformer 기반 SR 모델의 다양한 혁신적인 기법과 아키텍처를 탐구합니다. 이러한 최신 방법들은 글로벌(전역) 및 로컬(지역) 맥락을 균형 있게 고려하여 이미지 품질을 향상시키는 데 중점을 두고 있습니다. 또한, 다양한 시각화 방법을 통해 모델과 기술에 대한 종합적인 이해를 돕고 있습니다.

- **Performance Highlights**: 이 연구는 transformer가 초해상도 기법에 미치는 영향을 탐구하며, 향후 연구를 위한 구조적인 로드맵을 제시하고 있습니다. 비판적으로 분석된 최신 기법들은 유망하지만 아직 탐구되지 않은 격차와 잠재적인 연구 방향을 드러냅니다.



### 3UR-LLM: An End-to-End Multimodal Large Language Model for 3D Scene Understanding (https://arxiv.org/abs/2501.07819)
Comments:
          Accepted to IEEE Transactions on Multimedia (TMM)

- **What's New**: 이 논문에서는 3D 장면 이해를 위한 새로운 방법론인 3UR-LLM 모델을 제안합니다. 3UR-LLM은 3D 포인트 클라우드를 직접 입력으로 사용하고, 텍스트 지침과 융합된 3D 특성을 관리 가능한 토큰 세트로 변환하여 3D 장면을 효율적으로 해석하는 데 중점을 두고 있습니다. 이 모델은 고품질의 3D-텍스트 쌍을 생성하기 위해 오픈 소스 2D MLLM 및 LLM을 기반으로 한 파이프라인을 개발하였습니다.

- **Technical Details**: 3UR-LLM은 3DETR을 인식 구성 요소로 통합하여 3D 구조와 객체 간의 관계를 더 잘 파악할 수 있게 하였습니다. 또한, 3D 압축 모듈을 통해 3D 공간 정보를 간소화하고 정렬하여 계산 효율성을 높입니다. 여기에 높은 신뢰도의 쿼리를 선택하는 3D 쿼리 융합 메커니즘을 설계하여 계산 자원을 절약하며 3D 환경의 공간 이해도를 향상시킵니다.

- **Performance Highlights**: 3UR-LLM은 ScanQA 벤치마크에서 7.1% 높은 CIDEr 성능을 발휘하며, 이전의 최첨단 기술(SOTAs)보다 더 적은 훈련 자원으로 우수한 성능을 보여줍니다. 3DS-160K 데이터 세트를 통해 3D 장면 설명, 밀집 캡션 및 질문 응답 같은 다양한 작업을 지원하며, 기존 접근 방식보다 더 효율적인 파이프라인을 제시합니다.



### AVS-Mamba: Exploring Temporal and Multi-modal Mamba for Audio-Visual Segmentation (https://arxiv.org/abs/2501.07810)
Comments:
          Accepted to IEEE Transactions on Multimedia (TMM)

- **What's New**: 본 논문에서는 AVS(오디오-비주얼 세그멘테이션) 작업을 수행하기 위해 Mamba라는 선택적 상태 공간 모델을 도입합니다. 기존의 Transformer 기반 방법들이 복잡한 시나리오에서 긴 종속성 처리가 어려운 반면, AVS-Mamba는 선형 복합성을 통해 이를 극복하여 복잡한 멀티모달 이해를 촉진합니다. 또한 Temporal Mamba Block과 Vision-to-Audio Fusion Block을 포함하여 동영상 이해와 크로스 모달 학습을 위한 두 가지 핵심 요소를 통합합니다.

- **Technical Details**: AVS-Mamba는 Multi-scale Temporal Encoder(MTE)를 통해 비주얼 피처의 학습을 강화하고, Modality Aggregation Decoder(MAD)를 통해 비디오 프레임과 시간적 수준에서 비주얼 특징과 오디오 특징의 융합을 수행합니다. 특히, Contextual Integration Pyramid(CIP)를 도입하여 오디오와 비주얼 간의 공간-시간 맥락 협업을 이루어냅니다. 이러한 설계는 Mamba의 상태 압축 메커니즘을 혁신적으로 활용하여 효율적인 크로스 프레임 시퀀스 모델링을 가능하게 합니다.

- **Performance Highlights**: AVS-Mamba는 AVSBench-object 및 AVSBench-semantic 데이터셋에서 새로운 최첨단 성능을 기록하며, Jaccard index와 F-score 메트릭스 모두에서 우수한 성과를 보여줍니다. 이를 통해 제안된 접근 방법의 효과성을 강조하며, 오디오-비주얼 세그멘테이션 작업에서 Mamba의 잠재력을 최대한 활용할 수 있는 가능성을 제시합니다.



### Learning Motion and Temporal Cues for Unsupervised Video Object Segmentation (https://arxiv.org/abs/2501.07806)
Comments:
          Accepted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

- **What's New**: MTNet이라는 새로운 알고리즘을 제안하여, 자동 비디오 객체 분할에서 모션(motion)과 시간적 단서를 동시에 활용하는 방법을 소개하고 있습니다. 기존 방법들이 외형(apearance)과 모션을 통합하거나 시간적 관계를 모델링하는 것에만 집중한 것과는 달리, MTNet은 이 두 가지 측면을 통합하여 보다 일관된 프레임워크를 제공합니다. 이로 인해 복잡한 시나리오에서도 정밀한 물체 추적을 할 수 있게 되었습니다.

- **Technical Details**: MTNet은 인코더의 특징 추출 과정에서 외형과 모션 특징을 효과적으로 결합하여 더 보완적인 표현을 촉진합니다. 또한, Mixed Temporal Transformer 모듈을 도입하여 비디오 클립 전반에 걸친 인터프레임(inter-frame) 상호작용을 향상시키며, 다양한 특징 수준에서 작동하는 Cascaded Transformer Decoders가 세분화된 마스크(segmentation mask)를 생성합니다. 이러한 과정은 복잡한 비디오에서의 객체 추적 성능을 향상시킵니다.

- **Performance Highlights**: MTNet은 다양한 비디오 객체 분할(UVOS) 및 비디오 중요 객체 탐지(VSOD) 벤치마크에서 최첨단 성능을 달성했습니다. MTNet의 효율적인 아키텍처 설계 덕분에 2080Ti GPU에서 43.4fps의 속도로 동작할 수 있어, 실시간 실행이 가능하며 자원 제약이 있는 환경에서도 적합합니다. 이 알고리즘은 장기적인 모션과 차폐(occlusion) 문제에 강한 성능을 나타내어, 현실의 다양한 분할 작업에 매우 유용합니다.



### Balance Divergence for Knowledge Distillation (https://arxiv.org/abs/2501.07804)
- **What's New**: 본 논문에서는 기존 지식 증류 방법의 한계를 분석하고 새로운 "Balance Divergence Distillation (BDD)"이라는 방법을 제안합니다. 이 방법은 Kullback-Leibler (KL) divergence의 계산에서의 불균형 문제를 해결하기 위해 역 KL divergence를 도입합니다. 이를 통해 교사 네트워크의 정보 전달 과정에서 부정적인 작은 값들이 효과적으로 모델링될 수 있습니다.

- **Technical Details**: BDD는 KL divergence 계산을 두 가지 카테고리, 즉 forward-KL과 reverse-KL으로 분리하여 긍정적 및 부정적 값들을 강조합니다. 또한 온도 계수를 조정하여 교사 네트워크의 정보를 학생 네트워크가 보다 포괄적으로 습득할 수 있도록 합니다. 이러한 접근 방식은 다른 지식 증류 방법들과 쉽게 통합될 수 있어 효과적입니다.

- **Performance Highlights**: 실험 결과, CIFAR-100 및 ImageNet 데이터셋에서 경량 네트워크의 정확도가 1%에서 3% 향상되었으며, Cityscapes 데이터셋에서는 PSP-ResNet18의 mIoU가 4.55% 향상되었습니다. BDD는 단순한 수정으로 지식 증류의 불균형 문제를 해결하며 다양한 작업에서 유망한 결과를 나타냅니다.



### BioPose: Biomechanically-accurate 3D Pose Estimation from Monocular Videos (https://arxiv.org/abs/2501.07800)
- **What's New**: BioPose는 단안 영상을 통해 생체역학적으로 정확한 3D 인간 포즈를 예측하는 새로운 학습 기반 프레임워크입니다. 이 프레임워크는 다중 쿼리 인간 메쉬 복구 모델(MQ-HMR), 신경 역운동학 모델(NeurIK), 2D 정보 기반 포즈 보정 기법의 세 가지 핵심 요소로 구성되어 있습니다. BioPose는 기존의 마커 기반 시스템과 비교했을 때 접근성과 편리함을 유지하면서도 비슷한 성능을 목표로 합니다.

- **Technical Details**: MQ-HMR 모델은 다중 쿼리 변형가능한 트랜스포머를 사용하여 단안 비디오 프레임에서 다중 스케일의 세부 이미지를 추출합니다. 이 추출된 메쉬는 NeurIK 모델에 의해 가상의 마커로 취급되어 생체역학적으로 정확한 3D 포즈를 추론하는 데 사용됩니다. 마지막으로 2D 정보 기반 보정을 통해 3D 예측을 2D 관찰과 일치시켜 시각적 일관성과 생체역학적 유효성을 향상시킵니다.

- **Performance Highlights**: BioPose는 기준 데이터 세트에서 상태-of-the-art 방법들을 상당히 초월하는 성능을 보여주었습니다. MQ-HMR 모델은 단일 이미지에서 인간 메쉬 복구 작업에서 최첨단 결과를 생성하며, BioPose 시스템은 전통적인 다중 카메라 마커 기반 기술과 비교했을 때 경쟁력 있는 성능을 제공합니다. 이러한 성과는 생체역학적 정확성을 보장하는 매력적인 솔루션을 제공합니다.



### Parameter-Inverted Image Pyramid Networks for Visual Perception and Multimodal Understanding (https://arxiv.org/abs/2501.07783)
- **What's New**: 이번 논문에서는 파라미터를 반전시킨 이미지 피라미드 네트워크(Parameter-Inverted Image Pyramid Networks, PIIP)를 제안합니다. 기존의 이미지 피라미드는 동일한 대규모 모델을 여러 해상도의 이미지를 처리하는 데 사용하여 상당한 계산 비용을 초래했습니다. 반면, PIIP는 사전 훈련된 모델(예: ViTs 또는 CNNs)을 가지각색으로 활용하여 각기 다른 해상도의 이미지를 처리함으로써 계산 비용과 성능의 균형을 이룹니다.

- **Technical Details**: PIIP는 서로 다른 해상도의 특징을 효과적으로 융합할 수 있는 교차 브랜치(feature interaction) 메커니즘을 도입합니다. 이는 높은 해상도 이미지는 작은 네트워크의 가지로 처리하고, 낮은 해상도 이미지는 더 큰 네트워크 가지에서 처리하여 연산비용을 줄이고 성능을 향상시킵니다. 이 네트워크 구조는 기존의 비전 기초 모델들에 기반하여 각 해상도에서 직접 특징을 추출하는 방식을 채택합니다.

- **Performance Highlights**: PIIP는 객체 탐지, 세그멘테이션, 이미지 분류 및 다중 모달 이해 등 다양한 작업에서 단일브랜치 모델 및 기존의 다중 해상도 접근 방식보다 우수한 성능을 기록하면서도 계산 비용을 낮추었습니다. InternViT-6B와 같은 대규모 비전 모델에 적용시, PIIP는 탐지 및 세그멘테이션 성능을 각각 1%-2% 향상시키면서 원래의 컴퓨테이션의 40%-60%의 비용으로 실행됩니다. 또한, PIIP-LLaVA 모델은 극소량의 학습 데이터로 TextVQA에서 73.0% 정확도와 MMBench에서 74.5%를 달성했습니다.



### PSReg: Prior-guided Sparse Mixture of Experts for Point Cloud Registration (https://arxiv.org/abs/2501.07762)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 포인트 클라우드 등록(Point Cloud Registration)의 성능을 향상시키기 위해 Prior-guided Sparse Mixture of Experts (PSMoE) 모듈을 제안합니다. 이 방법은 중첩된 영역에서의 클러스터링 및 지도를 활용하여 보다 명확한 특징을 추출할 수 있도록 돕습니다. 또한, Transformer와 PSMoE 기능을 결합하여 최적의 변환 매트릭스를 추정하는 새로운 등록 프레임워크인 PSReg를 소개합니다.

- **Technical Details**: PSMoE 모듈은 사전 중첩 정보 및 잠재적 대응 임베딩을 융합하여 분산하는 방식으로, 가능한 한 많은 대응 토큰을 동일한 전문가에게 배정하고 처리합니다. 이를 통해 중첩된 영역에서 더 정확한 일치를 찾고 포인트 클라우드의 오버랩 영역을 효과적으로 찾아내는 것이 가능합니다. 새로운 PCE(전문가 지향 상위 포인트 대응 인코딩) 모듈은 매치된 슈퍼포인트를 명확하게 인코딩하여 상관관계를 더욱 높입니다.

- **Performance Highlights**: 해당 방법은 3DMatch 및 3DLoMatch 벤치마크에서 각각 95.7% 및 79.3%의 최첨단 등록 회수율을 달성하며, ModelNet40에서도 우수한 성능을 기록했습니다. 광범위한 실험을 통해 PSReg의 우위성을 입증하였으며, 다양한 모듈의 효과성을 검증하는 여러 단일 실험도 수행했습니다.



### Boosting Sclera Segmentation through Semi-supervised Learning with Fewer Labels (https://arxiv.org/abs/2501.07750)
Comments:
          Under review, 19 pages, 9 figures, 4 tables

- **What's New**: 이 논문은 제한된 라벨링 데이터셋을 활용하여 공막(segmentation) 세분화를 수행하는 새로운 프레임워크를 제안합니다. 특히 반지도 학습(semi-supervised learning) 기법을 사용하여 특정 도메인에서의 개선 및 이미지 기반 공간 변환을 통합해 세분화 성능을 향상시킵니다. 이 방법은 레이블이 없는 데이터가 많은 상황에서도 효과적으로 작동하도록 설계되었습니다.

- **Technical Details**: 공막 세분화는 입력 이미지 I에서 공막 영역을 이진 마스크 M으로 정확하게 식별하는 것을 목표로 하며, Intersection over Union (IoU)와 같은 메트릭스를 사용하여 세분화의 정확도를 평가합니다. 이 과정에서 이미지 반사, 속눈썹 가림 및 비 최적의 눈꺼풀 열림과 같은 노이즈(N)가 입력 이미지에 영향을 미치는 문제를 해결해야 합니다. 해당 연구에서는 라벨이 있는 데이터와 라벨이 없는 데이터를 모두 이용한 복합 손실 함수 L을 최소화하는 메커니즘을 도입합니다.

- **Performance Highlights**: 제안된 방법은 제한된 라벨링 샘플로도 높은 성능을 보여주며, 실험 결과 다른 두 개의 공개 데이터셋에서도 우수성을 입증합니다. 특히 라벨링의 부담을 줄이며, 많은 양의 라벨 없는 데이터를 효과적으로 활용하여 의료 영상 어플리케이션에서 세분화 정확도를 높이는데 기여합니다.



### Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation (https://arxiv.org/abs/2501.07742)
Comments:
          14 pages

- **What's New**: 최근 단안 깊이 예측(monocular depth prediction) 분야에서 큰 발전이 이루어져 깊이 예측 정확도가 개선되었습니다. 이 논문에서는 두 카메라 간의 상대적인 자세(relative pose)를 추정하기 위한 새로운 프레임워크를 제안합니다. 특히, 단안 깊이 값들이 갖고 있는 스케일(scale) 및 이동(shift) 파라미터를 함께 추정하여 문제를 해결합니다.

- **Technical Details**: 이 연구에서는 세 가지 케이스에 대해 효율적인 솔버를 도출했습니다: (1) 두 개의 보정된(calibrated) 카메라, (2) 서로 공유하는 초점 거리(focal length)가 있지만 보정되지 않은(ungraded) 카메라, (3) 서로 다른 초점 거리의 보정되지 않은 카메라입니다. 이러한 접근법을 통해 카메라 포즈 추정 문제에서 단안 깊이를 효과적으로 활용하는 방법을 제시합니다. 또한 실험을 통해 여러 깊이 예측기(depth predictor)에서 추정된 깊이 맵과 함께 실제 및 합성 데이터에서 성능을 평가했습니다.

- **Performance Highlights**: 제안된 솔버는 기존 작업들과 비교하여 두 개의 대규모(real-world) 데이터 세트에서 최첨단(state-of-the-art) 결과를 달성하였습니다. 또한, 정확성과 속도 측면에서도 기존 솔버보다 개선된 결과를 보였습니다. 이러한 실험 결과는 우리 솔버의 현실적인 적용 가능성을 뒷받침합니다.



### Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens (https://arxiv.org/abs/2501.07730)
Comments:
          Project page at this https URL

- **What's New**: 이 논문은 효율적이고 강력한 이미지 토크나이저인 Text-Aware Transformer-based 1-Dimensional Tokenizer (TA-TiTok)를 소개합니다. TA-TiTok은 디스크리트(discrete) 및 연속(continuous) 1차원 토큰을 사용하고, 디코딩 단계에서 텍스트 정보를 통합하여 성능을 향상시킵니다. 또한 간소화된 단일 단계 학습 프로세스를 통해 대규모 데이터셋으로의 확장을 가능하게 합니다.

- **Technical Details**: TA-TiTok은 전통적인 2D 기반 토크나이저의 단점을 극복하고, 고속 샘플링을 가능하게 하는 1D 잠재 시퀀스를 생성합니다. 이 모델은 이미지와 텍스트의 의미적 정합성을 높이기 위해 CLIP 임베딩을 결합하는 방식으로 훈련됩니다. 추가적으로 Masked Generative Models (MaskGen)은 공개 데이터에서 훈련되어 개인 데이터에 비해 비슷한 성능을 보입니다.

- **Performance Highlights**: MaskGen-L(568M)은 MJHQ-30K에서 7.74의 FID 점수를 기록하며, Show-o(14.99)보다 30.3배 빠른 추론 속도를 자랑합니다. MaskGen-XL(1.1B)은 MJHQ-30K에서 7.51 및 6.53의 FID 점수를 달성하며, 두 가지 토큰 방식 모두에서 높은 성능을 보입니다. 이 모델은 높은 성능을 제공하며 텍스트-이미지 생성 분야에서 민주화를 촉진하기 위해 모델 코드와 가중치를 공개할 예정입니다.



### Testing Human-Hand Segmentation on In-Distribution and Out-of-Distribution Data in Human-Robot Interactions Using a Deep Ensemble Mod (https://arxiv.org/abs/2501.07713)
- **What's New**: 이번 연구에서는 기존의 in-distribution(ID) 데이터와 실제 산업 환경에서 자주 발생하는 out-of-distribution(OOD) 상황에서 미리 훈련된 딥러닝(DL) 모델의 성능을 평가함으로써, 인간-로봇 협력에서 손의 신뢰할 수 있는 탐지 및 분할의 중요성을 강조했습니다. 특히, 산업 도구가 있는 복잡한 배경 및 장갑을 착용한 손 등 다양한 조건을 포함하여 새로운 데이터셋을 설계했습니다. 이를 통해 현실적인 산업 조건에서 손의 제스처와 동작을 더 잘 이해할 수 있도록 하고자 했습니다.

- **Technical Details**: 연구에서는 다중 시점(perspective)의 카메라를 사용하여 RGB 이미지를 캡처하고, egocentric 카메라와 정적 카메라를 조합했습니다. 손 분할을 위해 UNet과 RefineNet을 기반으로 한 딥 앙상블 모델을 사용하였고, 모델의 불확실성을 정량화하기 위해 predictive entropy를 활용한 평가 방식을 도입했습니다. 특히, OOD 상황에서의 성과를 평가하기 위해 손가락 교차 제스처와 빠르게 움직이는 손에서 발생하는 모션 블러와 같은 희귀한 조건을 포함했습니다.

- **Performance Highlights**: 실험 결과는 산업 데이터셋에 훈련된 모델들이 비산업 데이터셋에 훈련된 모델들보다 월등히 우수한 성능을 보였음을 밝힙니다. 모든 모델이 OOD 상황에서는 어려움을 겪었지만, 산업 데이터셋에 훈련된 모델들은 일반화 능력이 크게 향상된 것으로 나타났습니다. 특히, 다수의 손이 존재할 수 있는 경우에 대한 적응력이 강화되었음을 보여주며, 이러한 결과는 산업 특성에 맞춘 훈련의 중요성을 시사합니다.



### Pedestrian Trajectory Prediction Based on Social Interactions Learning With Random Weights (https://arxiv.org/abs/2501.07711)
Comments:
          13 pages,7 figures,Accepted to IEEE Transactions on Multimedia (TMM)

- **What's New**: 이 연구에서는 DTGAN(Dynamic aTtention Generator Adversarial Networks)이라는 새로운 프레임워크를 제안합니다. 기존의 프리디파인드(pre-defined) 규칙에 기초한 사회적 상호작용 모델링의 한계를 극복하고, 무작위로 생성된 가중치를 사용하여 비명시적 사회적 상호작용을 캡처합니다. DTGAN은 GAN(Generative Adversarial Networks)의 응용을 그래프 시퀀스 데이터로 확장하여 보행자의 경로를 보다 정밀하게 예측할 수 있도록 합니다.

- **Technical Details**: DTGAN은 그래프 내의 각 노드에 대한 랜덤 가중치를 사용하여 사회적 상호작용을 모델링하는 데 초점을 맞추고 있습니다. 특히, GAT(Graph Attention Networks)를 활용하여 각 그래프에서 노드에 대한 적응적 중요도를 할당하고, 이를 통해 다양한 경로를 생성합니다. 또한, DTGAN은 다양한 작업 손실 함수(task loss functions)를 탐색하여 예측의 다양성과 현실성을 증진시키고, 이를 통해 보행자의 의도를 보다 정확하게 이해할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, DTGAN은 ADE(Avg. Displacement Error)와 FDE(Final Displacement Error)에서 각각 16.7%와 39.3% 향상을 보여줍니다. 제안된 프레임워크는 두 개의 공개 데이터세트에서 유효성과 정확도를 검증받았으며, 보행자의 의도를 효과적으로 이해할 수 있음을 입증하였습니다. 결과적으로, DTGAN은 다중 모드 경로 생성을 통해 보행자의 이동 불확실성을 잘 관리하게 되며, 기존의 예측 방법들에 비해 우수한 성능을 발휘합니다.



### C2PD: Continuity-Constrained Pixelwise Deformation for Guided Depth Super-Resolution (https://arxiv.org/abs/2501.07688)
- **What's New**: 본 논문에서는 Guided Depth Super-resolution (GDSR) 분야에서 기존 방법들이 깊이 맵을 이미지로 취급하는 한계를 극복하기 위한 새로운 접근법을 제안합니다. 이를 위해, 깊이를 이상적인 플라스틱 물질로 변형시켜 연속성(continuity) 제약을 고려하며, 외부의 힘에 의해 변형되는 과정을 시뮬레이션하는 Continuity-constrained Asymmetrical Pixelwise Operation (CAPO)을 설계합니다. 이러한 방식을 통해 모델은 사람의 시각적 인지에 기반하여 더욱 자연스럽고 일관된 복원을 수행할 수 있습니다.

- **Technical Details**: 모델의 핵심 구성 요소는 CAPO와 Pixelwise Cross Gradient Deformation (PCGD)입니다. CAPO는 깊이가 지침 정보의 영향으로 변화하는 방식을 모델링하는 교차 모달(operation)이며, 연속성 제약을 준수합니다. PCGD는 이상적인 플라스틱 물질에서 발생할 수 있는 변형을 에뮬레이트하며, 연속성을 유지하고 모델의 일반화를 보장합니다. 이러한 기법들은 기존의 깊이 맵의 특징을 추출하고 RGB 이미지의 정보를 효과적으로 활용합니다.

- **Performance Highlights**: 제안된 방법은 네 가지 주요 GDSR 벤치마크에서 최상위 성능을 나타내며, 대규모 작업에 대한 뚜렷한 이점을 보여줍니다. 특히, 이 방법은 깊이 복원 과정을 더 자연스럽고 효과적으로 수행하게 하며, 모델의 일반화 능력은 점차 확대되는 경향을 보입니다. 이러한 성과는 대규모 과제에서의 새로운突破를 제시하며, x32와 같은 대규모 데이터 처리에서 두드러진 장점을 발휘합니다.



### BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations (https://arxiv.org/abs/2501.07647)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 복잡한 텍스트 프롬프트를 따르는 기존 비디오 생성 모델의 한계를 극복하기 위해 blob 비디오 표현(blob video representation)을 제안합니다. 이 새로운 표현은 사용자가 객체의 움직임과 세부 외형을 보다 쉽게 제어할 수 있게 합니다. BlobGEN-Vid라는 모델을 개발하여 물체 중심의 이동과 외형 제어를 가능하게 하였으며, 입체적인 일관성을 향상시키기 위한 3D attention 모듈도 도입했습니다.

- **Technical Details**: Blob 비디오 표현은 객체 인스턴스를 나타내는 blob 시퀀스로 구성되며, 이는 비디오(또는 3D 장면)에서 자동으로 추출할 수 있습니다. 각각의 blob은 위치, 크기 및 방향을 지정하는 벡터로 정의되며, 사용자들이 구조화된 텍스트 형태로 쉽게 생성하고 조작할 수 있습니다. 이를 통해 motion과 semantic control을 동시에 가능하게 합니다. BlobGEN-Vid는 UNet이나 DiT 기반의 기존 비디오 확산 모델에 적용 가능하며, 특히 시간적인 일관성을 유지할 수 있는 장점이 있습니다.

- **Performance Highlights**: BlobGEN-Vid는 여러 벤치마크에서 기존의 레이아웃 가이드 비디오 생성 모델보다 더 뛰어난 성능을 보였습니다. 특히, layout controllability는 mIOU에서 20% 이상 개선되었으며, CLIP 유사도에서 5% 향상되었습니다. LLMs와 결합할 경우, BlobGEN-Vid는 복잡한 장면에서도 높은 구성 정확도를 달성하며 뛰어난 성능을 입증했습니다.



### MiniMax-01: Scaling Foundation Models with Lightning Attention (https://arxiv.org/abs/2501.08313)
Comments:
          A technical report from MiniMax. The authors are listed in alphabetical order. We open-sourced our MiniMax-01 at this https URL

- **What's New**: MiniMax-01 시리즈, 즉 MiniMax-Text-01 및 MiniMax-VL-01이 도입되었습니다. 이 모델은 최고 수준 모델들과 비교할 수 있으며, 긴 컨텍스트를 처리하는 데 있어 더 나은 능력을 제공합니다. 중요 요소는 lightning attention과 이를 효율적으로 확장하는 부분입니다.

- **Technical Details**: MiniMax-01 모델은 Mixture of Experts (MoE)와 통합되어, 32명의 전문가와 총 4560억 개의 파라미터를 가진 모델로 구성됩니다. 각 토큰에 대해 459억 개의 파라미터가 활성화되며, 이를 위해 최적화된 병렬 전략과 효율적인 계산-통신 오버랩 기술을 개발하였습니다. 이러한 접근 방식은 수백억 개의 파라미터를 가진 모델을 수백만 토큰의 컨텍스트에서 효율적으로 훈련 및 추론할 수 있게 해줍니다.

- **Performance Highlights**: MiniMax-Text-01의 컨텍스트 윈도우는 훈련 중에 최대 100만 토큰에 도달할 수 있으며, 추론 중에는 최대 400만 토큰까지 확장 가능합니다. 두 모델은 GPT-4o 및 Claude-3.5-Sonnet과 같은 최첨단 모델과 성능이 일치하면서도 20-32배 더 긴 컨텍스트 윈도우를 제공합니다. MiniMax-01은 공개적으로 발표되었으며, 이를 통해 더욱 효율적인 작업이 가능해졌습니다.



### VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes (https://arxiv.org/abs/2501.08286)
- **What's New**: VINGS-Mono는 대규모 장면을 위한 단안(Inertial) Gaussian Splatting(GS) SLAM 프레임워크로, RGB 프레임을 처리하여 장면 기하학과 포즈를 추출합니다. 이 프레임워크는 여러 모듈로 구성되어 있으며, NVS Loop Closure와 Dynamic Eraser를 통해 동적 객체의 영향을 최소화하고 루프 클로저를 혁신적으로 수행합니다. 최신 GS/NeRF SLAM 방법을 초월하는 성능을 보여주며, 또 다른 주목할 점은 스마트폰 카메라와 저주파 IMU만으로 고품질 Gaussian 맵을 실시간으로 생성할 수 있다는 점입니다.

- **Technical Details**: VINGS-Mono는 VIO Front End, 2D Gaussian Map, NVS Loop Closure 및 Dynamic Object Eraser의 네 개 모듈로 구성됩니다. 2D Gaussian Map은 지역 및 전역 맵 표현을 통합하여 관리하는 score manager와 Gaussian Splatting의 backpropagation 알고리즘을 가속화하는 sample rasterizer를 포함합니다. 단안 기법의 한계를 극복하기 위해, 단일 프레임에서 발생하는 렌더링 오류를 여러 프레임으로 전파하여 포즈 일관성을 향상시키는 single-to-multi pose refinement 모듈도 설계되었습니다.

- **Performance Highlights**: 실내 및 실외 환경에 대한 종합적인 평가에서, VINGS-Mono는 기존의 모든 방법에 비해 매핑 및 렌더링 품질에서 뛰어난 성능을 보여주었습니다. 특히, 실시간으로 고품질 Gaussian 맵을 생성할 수 있는 모바일 앱을 개발하여 실용적인 신뢰성을 검증하였습니다. 또한, VINGS-Mono는 기존의 Visual-Inertial Odometry와 유사한 로컬라이제이션 성능을 달성하면서도 SGS 기반 루프 감지 및 수정 방법을 도입하여 누적 오류를 효과적으로 제거했습니다.



### Can Bayesian Neural Networks Explicitly Model Input Uncertainty? (https://arxiv.org/abs/2501.08285)
Comments:
          12 pages, 11 figures, VISAPP 2025 camera ready

- **What's New**: 최근 기계 학습 모델에 들어가는 입력 데이터는 외부 노이즈나 불확실성을 동반할 수 있으며, 이러한 요소들은 종종 간과되고 모델링되지 않는다. 본 연구에서는 두 개의 입력(평균과 표준 편차)을 사용하는 Bayesian Neural Network(베이지안 신경망)을 구축하고, Ensemble, MC-Dropout, Flipout과 같은 다양한 방법론에서 입력 불확실성 추정능력을 평가했다. 결과적으로, 몇몇 불확실성 추정 방법이 입력 불확실성을 모델링할 수 있는 것으로 나타났으며, 특히 Ensemble과 Flipout이 그 성능이 우수한 것으로 밝혀졌다.

- **Technical Details**: 본 연구는 두 개의 데이터셋(Two Moons와 Fashion-MNIST)을 기반으로 실험을 수행하였다. Two Moons는 두 개의 반원으로 구성된 이진 분류 문제로, 연구자들이 모델의 신뢰성 있는 불확실성 값을 산출할 수 있는 능력을 시각화하기 위해 자주 활용하는 toy dataset(토이 데이터셋)이다. Fashion-MNIST는 10개 카테고리의 의류 아이템을 포함하는 이미지 분류의 인기 있는 벤치마크로, 패션 아이템의 그레이스케일 이미지로 구성된다.

- **Performance Highlights**: 우리의 결과는 여러 불확실성 추정 방법들이 입력의 불확실성을 제대로 예측하지 못한다는 것을 보여준다. Ensemble 방법은 때때로 입력 불확실성을 효과적으로 반영할 수 있는 유일한 방법으로 나타났다. 연구의 결과는 Bayesian Neural Network가 입력 불확실성을 명시적으로 모델링하는 데 성공하지 못할 수 있음을 시사하며, 모델이 높은 수준의 자신감을 보여주는 경향이 있음을 드러낸다.



### CG-MER: A Card Game-based Multimodal dataset for Emotion Recognition (https://arxiv.org/abs/2501.08182)
Comments:
          8 pages, 2 figures and 4 tables. Sixteenth International Conference on Machine Vision (ICMV 2023), Yerevan, Armenia

- **What's New**: 이번 논문은 감정 인식(emotion recognition)을 위해 특별히 설계된 포괄적인 프랑스어 다중 모달 데이터셋(multimodal dataset)을 소개합니다. 이 데이터셋은 얼굴 표정(facial expressions), 음성(speech), 제스처(gestures)라는 세 가지 주요 모달리티(modality)를 포함하여 감정에 대한 전체적인 관점을 제공합니다. 또한, 이 데이터셋은 자연어 처리(Natural Language Processing, NLP)와 같은 추가 모달리티를 통합할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 데이터셋은 카드 게임 세션에 참여한 20명의 참가자(여성 9명, 남성 11명)를 대상으로 구성되었습니다. 이 과정에서 참가자들은 다양한 질문에 응답하며 여러 감정을 표현하도록 유도되었습니다. 총 10회의 세션을 거쳐 감정을 표현하는 다양한 방식이 수집되었습니다.

- **Performance Highlights**: 이 데이터셋은 감정 인식 연구에 유용한 자료로 활용될 수 있으며, 인간 감정과 디지털 기술 사이의 복잡한 연결을 탐구하는 데 기여할 수 있는 길을 제공합니다. 이를 통해 감정 컴퓨팅 분야에서의 새로운 연구 가능성이 열릴 것으로 기대됩니다.



### DM-Mamba: Dual-domain Multi-scale Mamba for MRI reconstruction (https://arxiv.org/abs/2501.08163)
- **What's New**: 본 논문은 Mamba라는 새로운 패러다임을 소개하며, 이는 효율적이고 효과적인 MRI 재구성을 위한 장거리 의존성 모델링에서의 혁신을 나타냅니다. 기존의 Mamba가 MRI 재구성에 적용될 때의 제한 사항을 해결하기 위해, 이 방법은 듀얼 도메인 멀티 스케일 Mamba를 제안합니다. k-공간에서의 주파수 특성을 보존하기 위해 원형 스캐닝 기법을 도입하며, 이미지와 k-공간 모두에서의 효율성을 높입니다.

- **Technical Details**: Dual-domain Multi-scale Mamba(듀얼 도메인 멀티 스케일 맘바)의 제안은 이미지와 k-공간에서의 효율적인 재구성을 위한 새로운 전략을 제시합니다. 이를 통해 공간적 다양성을 개선하고, 데이터 손실을 최소화하는 여러 스캐닝 경로를 제공하여 계산 복잡성을 줄입니다. 특히, 특성 전파 중 지역 다양성을 높이기 위해 로컬 강화 모듈을 개발하여, 고품질 재구성을 지원합니다.

- **Performance Highlights**: 본 연구의 실험은 세 가지 공개 데이터셋에서 다양한 언더샘플링 패턴을 통해 수행되었습니다. 결과적으로, 제안된 방식이 최신 기법들보다 우수한 성능을 보여주었고, 더욱 낮은 계산 비용으로도 높은 품질의 MRI 이미지를 복원할 수 있음을 입증했습니다. 논문에서 제안하는 코드는 향후 제공될 예정입니다.



### CellOMaps: A Compact Representation for Robust Classification of Lung Adenocarcinoma Growth Patterns (https://arxiv.org/abs/2501.08094)
- **What's New**: 이번 논문에서는 폐 선암(Lung Adenocarcinomas, LUAD)의 성장 패턴을 분류하기 위한 기계학습 파이프라인을 제안합니다. 기존의 연구들은 주로 슬라이드 당 가장 많은 비율을 차지하는 패턴만을 보고하거나 적절한 평가가 부족했으나, 본 연구는 다섯 가지의 주요 패턴을 효과적으로 구분할 수 있는 방법을 제시합니다. 여기에는 Cell Organization Maps (cellOMaps)라는 새로운 이미지 표현이 포함되어 있어, 세포의 공간적 패턴을 효과적으로 포착합니다.

- **Technical Details**: 제안된 CellOMaps는 Hematoxylin and Eosin (H&E) 염색된 전체 슬라이드 이미지를 변환 및 압축하여 10:1 비율의 압축을 달성합니다. 이 표현은 세포 조직을 충분한 수준의 세부정보로 캡처하며, 각 세포의 중심점과 타입을 반영하여 깊은 학습 모델이 미세 환경 내에서 세포 간의 상호작용을 중심으로 학습할 수 있도록 합니다. 이러한 기법은 패턴 분류에 필요한 정보만을 유지하고 불필요한 세부사항을 제거함으로써, 모든 슬라이드에서 일관된 예측을 가능하게 합니다.

- **Performance Highlights**: 제안된 파이프라인은 내부 슬라이드와 외부 데이터 세트에서 평가한 결과, 평균 0.81의 정확도로 선진적인 성능을 보였습니다. 이는 기존의 방법들과 비교해 상당한 성능 향상을 보여주며, 환자의 Tumor Mutational Burden (TMB) 수준을 예측하는 데 있어 모델 출력도 유용하게 쓰일 수 있음을 시사합니다. 이러한 성과는 기계 학습을 이용한 병리학 업무의 자동화를 통한 진단의 정확성과 신뢰성을 높이는 데 중점을 두고 있습니다.



### Self-Attentive Spatio-Temporal Calibration for Precise Intermediate Layer Matching in ANN-to-SNN Distillation (https://arxiv.org/abs/2501.08049)
- **What's New**: 이번 논문은 Spiking Neural Networks (SNNs)의 성능을 향상시키기 위해 새로운 방법인 Self-Attentive Spatio-Temporal Calibration (SASTC)를 제안합니다. SASTC는 ANN과 SNN 간의 의미적으로 일치하는 레이어 쌍을 자율적으로 식별하여 시공간적 차이를 극복하도록 설계되었습니다. 이 방법은 ANN에서 SNN으로 지식을 효과적으로 전이할 수 있도록 하며, SNN의 기존 성능 한계를 뛰어넘는 결과를 보여줍니다.

- **Technical Details**: SASTC는 자가 주의(self-attention) 메커니즘을 사용하여 ANN과 SNN 간의 레이어 패턴을 시공간적으로 정렬합니다. 연구에서는 각각 b 크기의 미니 배치 데이터에 대해 SNN과 ANN의 출력을 구분하기 위해 각 레이어의 출력을 수학적으로 정의합니다. 이를 통해 맞춤형 지식 증류(knowledge distillation)가 가능하여 SNN의 성능을 상당히 개선할 수 있습니다.

- **Performance Highlights**: SASTC는 CIFAR-10, CIFAR-100 및 ImageNet과 같은 여러 데이터셋에서 기존 방법보다 우수한 성능을 보여주었습니다. 특히 CIFAR-10에서는 95.12%, CIFAR-100에서는 79.40%의 정확도를 달성하였으며, DVS-Gesture와 DVS-CIFAR10와 같은 신경 형태 데이터셋에서도 각각 97.92%와 83.60%의 성능을 기록했습니다. 또한 이번 연구는 SNN이 CIFAR-10과 CIFAR-100에서 ANN을 초월한 최초의 사례로, SNN의 잠재적인 응용 가능성을 제시합니다.



### Maximizing Uncertainty for Federated learning via Bayesian Optimisation-based Model Poisoning (https://arxiv.org/abs/2501.08002)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 Federated Learning (FL) 내에서의 악의적인 모델 오염 공격을 탐구하는 새로운 방법론을 제안합니다. 저자들은 새로운 모델 오염 공격 기법인 Delphi를 소개하며, 모델 출력을 최대한의 불확실성(unexpected uncertainty)으로 만드는 것을 목표로 합니다. 모델의 첫 번째 은닉층(hidden layer)과 관련된 불확실성을 활용하여 공격 기법의 효과성을 증명하고, FL 시스템의 취약성을 강조합니다.

- **Technical Details**: Delphi 방법론은 Bayesian Optimisation 및 Least Squares Trust Region 두 가지 최적화 기술을 사용하여 최적의 오염 모델 매개변수를 탐색합니다. 이 기법은 특정 뉴런에만 집중하여 모델 매개변수를 조작함으로써 불확실성을 유도합니다. 이 과정에서 저자들은 KL Divergence를 통해 예측 확률 분포의 거리를 최소화하여 불확실성을 정량화하는 방법을 수립했습니다.

- **Performance Highlights**: 실험 결과, Delphi-BO 방법이 Delphi-LSTR보다 높은 불확실성을 유도함을 보여줍니다. Delphi-BO는 매 학습 라운드에서 가장 중요한 뉴런을 선택하여 공격을 수행하며, 그에 따라 모델의 예측 신뢰도가 절반으로 감소하는 것으로 나타났습니다. 이러한 결과는 FL 모델이 모델 오염 공격에 대해 얼마나 취약한지를 보여줍니다.



### Zero-shot Video Moment Retrieval via Off-the-shelf Multimodal Large Language Models (https://arxiv.org/abs/2501.07972)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 영상 순간 검색(Video Moment Retrieval, VMR)에 대한 새로운 접근법인 Moment-GPT를 제안합니다. Moment-GPT는 미세 조정 없이 동결된 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 사용하여 직접 추론할 수 있는 제로샷(zero-shot) VMR 기법입니다. 이 방법은 기존의 언어 편향(language bias)을 개선하고 비디오 이해 능력을 활용하여 정확한 스팬(span) 생성을 가능하게 합니다.

- **Technical Details**: 제안된 Moment-GPT는 먼저 LLaMA-3를 활용하여 질의를 최적화하고 언어 편향을 줄입니다. 이후 MiniGPT-v2를 통해 비디오에서 프레임 수준의 캡션을 생성하고, 이를 바탕으로 후보 스팬을 적응적으로 생산합니다. 마지막으로 Video-ChatGPT를 통해 스팬 수준의 캡션을 생성하고, 스팬 스코어러(span scorer)를 통해 최종 결과를 얻습니다.

- **Performance Highlights**: Moment-GPT는 QVHighlights, ActivityNet-Captions, Charades-STA와 같은 여러 공개 데이터셋에서 기존의 MLLM 기반 및 제로샷 모델을 크게 능가하는 성능을 보여주었습니다. 본 연구는 또한 대부분의 감독 모델을 초과하는 결과를 달성하여 비디오 이해 및 질의 응답 분야에서의 가능성을 증명합니다.



### AI Guide Dog: Egocentric Path Prediction on Smartphon (https://arxiv.org/abs/2501.07957)
- **What's New**: AI Guide Dog (AIGD)는 시각장애인을 위한 경량의 자가 중심 내비게이션 지원 시스템으로, 스마트폰에서 실시간 배치를 위해 설계되었습니다. 이 시스템은 영상 기반의 다중 레이블(classification) 접근 방식을 사용하여 방향 명령을 예측함으로써 안전한 탐색을 보장합니다. 또한 GPS 신호와 고급 방향 지시를 통합하여 목표 기반의 야외 내비게이션을 가능하게 하고, 목적지 없이도 실내 탐색을 지원하는 불확실한 다중 경로 예측 문제를 해결합니다.

- **Technical Details**: AIGD는 스마트폰 카메라에서 수신된 비디오 피드를 사용하여 내비게이션 지침을 예측하는 경량 모델을 채택하였습니다. 시스템은 사용자의 헤딩 방향을 예측하기 위해 각 프레임에 대해 다중 레이블 분류 문제로 모델링합니다. 이를 통해 사용자의 다양한 내비게이션 방향을 예측하고, 구체적인 오디오 명령으로 변환합니다. 실내외 환경 모두에서 목표 지향적 및 탐험적 내비게이션 시나리오를 처리하는 첫 번째 시스템으로 자리잡고 있습니다.

- **Performance Highlights**: AIGD는 장애물 회피를 지원하며, 사용자가 목표가 없는 상황에서도 자유롭게 탐색할 수 있도록 설계되었습니다. 이 시스템은 기존의 시각 장애인 내비게이션 시스템보다 더 견고하고 효율적이며, 최종 목적지에의 도달을 위한 명확한 지침을 제공합니다. 데이터 수집 및 전처리를 통해 모델의 성능을 극대화하며, 사용자들은 명확하고 실행 가능한 오디오 명령을 통해 탐색을 지원받습니다.



### Early prediction of the transferability of bovine embryos from videomicroscopy (https://arxiv.org/abs/2501.07945)
Comments:
          Accepted at the 2024 IEEE International Conference on Image Processing

- **What's New**: 본 논문은 머신러닝(Machine Learning)과 결합된 비디오 현미경(Videomicroscopy) 기술을 활용하여 체외 수정된 소 배아의 초기 발달을 연구하는 새로운 방법을 제시합니다. 특히, 2D 타임랩스(2D time-lapse) 비디오를 입력으로 받아 4일 이내에 배아 이식 가능성을 예측하는 것을 목표로 하고 있습니다. 이를 위해 이식 가능성과 비이식 가능성을 분류하는 이진 분류 문제로 문제를 설정하였습니다.

- **Technical Details**: 저자들은 3D 합성곱 신경망(3D Convolutional Neural Network)을 제안하여 세 가지 경로(pathway)를 포함하는 다중 스케일(multi-scale) 접근 방식을 구현하였습니다. 이 모델은 시각적 특징과 움직임(motion)을 다르게 처리하는 능력을 보유하고 있으며, 훈련 시에는 포컬 로스(focal loss)를 적용하여 성능을 향상시킵니다. 모델의 이름은 SFR이며, 이는 기존 방법들과 비교할 때 우수한 성능을 나타냅니다.

- **Performance Highlights**: 실험을 통해 모델 SFR의 효과성과 정확성이 입증되었습니다. 특히 소 배아의 이식 가능성을 평가하는 도전적인 생물학적 과제에 대해 높은 정확도를 기록하여 향후 연구와 관련된 응용 가능성을 시사합니다. 이 연구는 동물 생식의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Mitigating Algorithmic Bias in Multiclass CNN Classifications Using Causal Modeling (https://arxiv.org/abs/2501.07885)
Comments:
          7 pages; 6 figures

- **What's New**: 이 논문은 다중 클래스 분류 문제에서 알고리즘 편향을 감지하고 완화하기 위한 인과 모델링(causal modeling) 절차를 설명합니다. FairFace 데이터셋을 기반으로 하여 DeepFace 사전 훈련된 모델로 생성된 감정 레이블을 보완하여 사용했습니다. 커스텀 CNN 모델을 개발하여 성별 편향을 식별하였으며, 성별에 따라 분류 결과에서 '행복' 또는 '슬픔'으로 더 분류된 여성과 '중립'으로 더 분류된 남성을 분석했습니다. 이후 one-vs-all (OvA) 기법을 적용하여 감정 클래스마다 인과 모델을 구성하여 CNN의 예측 확률을 조정했습니다.

- **Technical Details**: FairFace 데이터셋은 인종, 성별, 연령 속성이 균형있게 표현된 97,698개의 이미지를 포함하고 있습니다. CNN 모델 아키텍처는 4개의 합성곱 층(convolutional layers)과 정규화 및 차원 축소를 위한 최대 풀링(max-pooling)으로 구성됩니다. 훈련 시 Adam 최적화기와 학습률 0.0001을 사용하였으며, 조기 종료(early stopping) 메커니즘으로 과적합을 방지하였습니다. 모델은 7개의 감정 클래스에 대한 확률을 출력하고, 가장 높은 확률을 가진 감정을 예측 레이블로 선택합니다.

- **Performance Highlights**: 훈련 후 테스트 세트에서 모델의 전반적인 정확도는 58.3%로 평가되었습니다. 인과 모델링을 통해 편향이 완화된 분류 결과가 성별 공정성을 향상시켰으며, 전체 정확도에 미치는 영향은 미미하거나 약간 개선된 것으로 나타났습니다. 이 연구는 알고리즘 공정성과 정확성이 반드시 상충하지 않음을 강조합니다.



### An Intra- and Cross-frame Topological Consistency Scheme for Semi-supervised Atherosclerotic Coronary Plaque Segmentation (https://arxiv.org/abs/2501.07850)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 논문에서는 Curved Planar Reformation (CPR) 이미지를 활용해 관상동맥 죽상경화증 분석 (CAA)에 필요한 관상동맥 플라크 세분화의 정밀도를 향상시키기 위한 혁신적인 이중 일관성 반지도 학습 프레임워크를 제안합니다. 기존의 깊이 학습 모델들이 경계의 불명확성으로 인해 좋지 않은 성과를 내는 것을 해결하고자, Intra-frame Topological Consistency (ITC)와 Cross-frame Topological Consistency (CTC)를 통합하여 라벨이 붙은 데이터와 붙지 않은 데이터를 효과적으로 활용합니다.

- **Technical Details**: 제안된 프레임워크는 이중 작업 네트워크를 통해 세분화 마스크와 Skeleton-aware Distance Transform (SDT) 예측을 동시에 수행합니다. ITC는 추가적인 주석 없이도 일관된 예측을 통해 형상 구조에 대한 일관성을 유지합니다. 반면 CTC는 인접한 프레임 간의 픽셀 흐름을 분석하는 비지도 추정기를 활용하여 공간 연속성을 확보합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 두 개의 CTA 데이터셋에서 기존의 반지도 학습 방법들을 초월하고 감독 방법의 성능에 근접하는 결과를 나타냈습니다. 또한 ACDC 데이터셋에서도 다른 방법들 대비 뛰어난 성능을 보여주며, 본 연구의 일반화 가능성을 증명했습니다.



### A Low-cost and Ultra-lightweight Binary Neural Network for Traffic Signal Recognition (https://arxiv.org/abs/2501.07808)
- **What's New**: 이번 연구에서는 하드웨어 배포를 위한 초경량 바이너리 신경망(ULTRA-LIGHTWEIGHT BINARY NEURAL NETWORK, BNN) 모델을 제안합니다. 이 모델은 독일 교통 표지 인식 벤치마크(German Traffic Sign Recognition Benchmark, GTSRB) 데이터세트를 기반으로 이미지 분류 연구를 수행하였으며, 중국 교통 표지(Chinese Traffic Sign, CTS) 및 벨기에 교통 표지(Belgian Traffic Sign, BTS) 데이터세트에서도 검증됩니다. BNN 모델은 97.64%의 높은 인식 성능을 보여 주며, GTSRB 데이터세트에서 가장 우수한 성능을 가진 BNN 모델 중 하나로 부각됩니다.

- **Technical Details**: 이 연구의 모델은 전체 정밀도(full-precision) 모델과 비교했을 때 정확도 손실이 1% 이내로 제한되고, 모델의 파라미터 저장 오버헤드는 전체 정밀도 모델의 10%에 불과합니다. 모델은 추론(inference) 단계에서 논리 연산과 저비트 폭 고정 소수점 덧셈 및 뺄셈 연산에만 의존하여 처리 소자(processing element, PE)의 설계 복잡성을 크게 단순화합니다. 이러한 설계는 자원에 제약이 있는 플랫폼에서의 배포를 용이하게 합니다.

- **Performance Highlights**: 제안된 BNN 모델은 GTSRB 데이터세트를 포함한 다양한 데이터세트에서 높은 정확도와 성능을 보입니다. 특히 자율주행 차량 분야의 컴퓨터 비전(computer vision) 과제를 다룰 때 BNN의 가능성이 매우 크다는 것을 보여줍니다. 이러한 초경량 모델은 이미지 분류의 효율성을 극대화하며, 리소스가 제한적인 환경에서도 우수한 인식 능력을 발휘할 수 있음을 입증합니다.



### BMIP: Bi-directional Modality Interaction Prompt Learning for VLM (https://arxiv.org/abs/2501.07769)
- **What's New**: 이 논문은 Bi-directional Modality Interaction Prompt (BMIP)라는 새로운 프롬프트 학습 방법을 제안합니다. 이 방법은 비전(vision)과 언어(language) 모달리티 간의 상호 작용에서 발생하는 정렬 효과를 극대화합니다. BMIP는 다양한 데이터셋에서 기존 최첨단 방법들을 초월하며, 다른 프롬프트 기반 방법들과의 결합이 용이합니다. 이를 통해 기존 방법에서 발생했던 단일 모달리티의 일관성 문제를 해결했습니다.

- **Technical Details**: BMIP는 주의(attention) 레이어에서 학습된 정보를 통해 이중 모달 정보에 동적으로 가중치를 부여합니다. 기존의 단순 정보 집계 방법들과 비교할 때, BMIP는 훈련 가능성과 모달 간 일관성을 높이는데 초점을 맞춥니다. 이 방법은 다중 모달 상호작용을 위해 최초로 설계된 집계 함수에 기초하고 있으며, 이는 정밀한 정보 Utilization을 가능하게 합니다. 또한, 새로운 평가 패러다임인 open-world generalization을 통해 추가적으로 개선된 성능을 정량적으로 평가합니다.

- **Performance Highlights**: BMIP는 15개의 벤치마크에 대한 실험에서 모든 평가 패러다임에서 SOTA 성능을 달성했습니다. 특히, BMIP는 EuroSAT과 Flowers102와 같은 불균형 데이터셋에서 단일 모달 프롬프트 학습 방법의 저조한 성능을 해결해냅니다. BMIP는 MaPLe와 함께 사용될 수 있어 다른 프롬프트 학습 방법의 성능도 일관되게 향상시킬 수 있는 기반 프레임워크로 작용합니다.



### Universal Training of Neural Networks to Achieve Bayes Optimal Classification Accuracy (https://arxiv.org/abs/2501.07754)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 연구는 $f$-divergence 개념을 통해 일반적인 분류 작업의 Bayes 에러율에 대한 새로운 상한을 소개합니다. 제안된 경계는 매개변수화된 모델의 출력을 샘플링하여 계산할 수 있습니다. 또한, Bayes 최적 학습 임계값(BOLT) 손실 함수를 도입하며, 이러한 손실 함수를 최소화하면 분류 모델이 Bayes 에러율에 도달하도록 유도할 수 있습니다.

- **Technical Details**: Bayes 에러율은 특정 분류 작업에 대한 조건부 데이터 분포의 $f$-divergence와 관련이 있음을 보여줍니다. 이 연구에서는 샘플링을 통해 데이터 샘플로부터 직접 계산 가능한 새로운 Bayes 에러의 상한을 도출합니다. 새로운 손실 함수인 BOLT는 일반화 성능 향상에 집중하고 있으며, 이는 매개변수화된 모델을 훈련하는 데 사용될 수 있습니다.

- **Performance Highlights**: BOLT를 사용하여 훈련된 모델은 MNIST, Fashion-MNIST, CIFAR-10 및 IMDb 데이터셋 등에서 기존의 교차 엔트로피 손실을 초과하는 성능을 보여줍니다. 숫자 실험 결과, BOLT가 모델의 일반화 성능을 향상시키는 잠재력을 강조합니다. 특히, BOLT는 복잡한 데이터셋에서 모델의 성능을 개선하는 데 효과적임을 입증했습니다.



### A Heterogeneous Multimodal Graph Learning Framework for Recognizing User Emotions in Social Networks (https://arxiv.org/abs/2501.07746)
- **What's New**: 이 논문은 소셜 네트워크 내에서 개인화된 감정 예측을 위한 Heterogeneous Multimodal Graph Learning Framework인 HMG-Emo를 제안합니다. 이는 다양한 모달리티의 데이터를 결합하여 사용자 감정을 인식하는 데 깊은 학습 기반의 기능을 활용합니다. 특히, 사용자 간의 다양한 관계와 콘텐츠 맥락을 동적으로 통합할 수 있는 모듈을 포함하여 감정 예측의 정확성을 높입니다.

- **Technical Details**: HMG-Emo는 사용자-미디어 그래프에서 엣지 분류 작업으로 개인화된 감정 예측 문제를 정의하는 혁신적인 방식으로, 그래프 학습 방법을 적용할 수 있습니다. 이 프레임워크는 Graph Attention Network를 활용하여 여러 모달리티에서 정보를 동시에 축적하고, 감정 분류 작업에 적응할 수 있는 동적 맥락 결합 모듈을 포함합니다. 다양한 실험을 통해 기존의 수작업으로 만들어진 특성을 사용한 방법보다 우수함을 보였습니다.

- **Performance Highlights**: HMG-Emo는 소셜 미디어에서의 개인화된 감정 예측을 위한 최초의 깊은 그래프 학습 기반 프레임워크로 자리 잡습니다. 이 프레임워크는 다양한 맥락 정보를 통합하여 감정 분류에서 기존 방법들보다 높은 성능을 보여줍니다. 또한 다중 요인을 사용한 감정 예측의 중요성을 실험적으로 입증하며, 제안된 방법의 다양한 구성 요소의 견고성을 평가했습니다.



### Dataset Distillation as Pushforward Optimal Quantization (https://arxiv.org/abs/2501.07681)
- **What's New**: 이 논문은 기존의 데이터셋 증류(dataset distillation) 방법론을 새로운 관점에서 제시합니다. 특히, 해체(disentangled) 접근 방식을 통해 기존의 이론적 해석을 제공하고, 고차원 데이터셋에 대한 새로운 최적화 방법을 제안합니다. 또한, 다양한 데이터 분배(distribution)에 대한 일관성을 입증하여, 이미지넷 데이터셋에서의 성능 향상을 도모하는 간단한 방법을 제안합니다.

- **Technical Details**: 이 작업은 데이터셋 증류 문제를 수학적으로 명확히 정의하고, 해체 접근 방식이 클래식한 최적 양자화(optimal quantization) 문제와 어떻게 연관되는지를 설명합니다. 특히, 인코더-디코더 구조를 활용하여 고차원 데이터에 대한 적절한 근사값을 찾는 방법을 제시하고, 이는 파라메트릭한 확률 측도를 최소화하는 것과 관련이 있습니다. 이러한 접근은 기존 방법론보다 연산 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 D4M 방법론의 간단한 확장을 통해 이미지넷-1K 데이터셋에서 우수한 성과를 보여줍니다. 추가적인 계산 비용이 거의 없이 클래스당 이미지를 늘릴 수 있는 가능성을 열어줍니다. 특히, 높은 클래스당 이미지 수 설정에서 최첨단 성능을 달성하여, 기존의 데이터셋 증류 기법들에 비해 월등한 효율성을 입증합니다.



### Impact of Data Breadth and Depth on Performance of Siamese Neural Network Model: Experiments with Three Keystroke Dynamic Datasets (https://arxiv.org/abs/2501.07600)
Comments:
          19 pages, 4 figures

- **What's New**: 이번 연구에서는 Deep Learning 모델, 특히 Siamese Neural Networks (SNN)를 활용하여 행동 데이터에서의 데이터셋의 폭(breadth)과 깊이(depth)가 모델 성능에 미치는 영향에 대해 심도 깊은 실험을 수행하였습니다. 데이터셋의 폭은 참여자의 수로 정의되며, 깊이는 개별 참여자당 샘플의 양으로 정의됩니다. 이에 따라 공공 데이터세트인 Aalto, CMU 및 Clarkson II를 사용하여 다양한 조건에서 실험을 진행하였습니다.

- **Technical Details**: 연구에서는 keystroke dynamics를 활용하여 인증 시스템을 구축하는 과정에서 데이터셋의 폭과 깊이가 모델 성능에 미치는 영향을 분석하였습니다. SNN은 두 개 이상의 동일한 서브 네트워크를 학습하여 유사성을 비교하는 구조로, 새로운 주제에 대해서도 유사성 점수를 계산할 수 있어 키 입력의 행동 생체 인식 분야에 효과적입니다. 실험에서는 훈련 주제의 수, 각 주제 당 샘플 수, 데이터 샘플 내 정보량, 훈련에 사용되는 트리플 수 등을 다양하게 조절하여 모델의 성능을 평가하였습니다.

- **Performance Highlights**: 연구 결과 데이터셋 폭을 증가시키는 것이 주관적 변동성을 효과적으로 포착하는 데 도움이 된다는 것을 알았으며, 특정 데이터셋의 깊이에 대한 영향은 해당 데이터의 특성에 따라 달라짐을 발견하였습니다. 자유 텍스트 데이터셋은 샘플 수, 시퀀스 길이 등 깊이에 따른 세 가지 요소 모두의 영향을 받는 반면, 고정 텍스트 데이터셋은 이러한 요소의 영향을 덜 받는 것으로 나타났습니다. 이러한 발견은 행동 생체 인식을 위한 딥 러닝 모델 훈련 시 데이터셋의 설계에 중요한 통찰을 제공합니다.



### Spin-Weighted Spherical Harmonics for Polarized Light Transpor (https://arxiv.org/abs/2501.07582)
- **What's New**: 이번 논문에서는 편광(polarization) 렌더링에서 빛과 물질 간의 상호작용을 시뮬레이션하는 새로운 방법을 제시합니다. 특히, Polarized Spherical Harmonics (PSH)를 기반으로 한 방법을 통해 스톡스 벡터(Stokes vectors)의 연속성과 회전 불변성(rotation invariance)을 효과적으로 처리할 수 있게 되었습니다. 이는 기존의 구형 조화 함수(spherical harmonics) 방식보다 현저히 향상된 결과를 제공합니다.

- **Technical Details**: 제안된 PSH 방법은 스톡스 벡터 필드의 회전 불변성 있는 표현을 가능하게 하여 편광 렌더링 방정식을 주파수 도메인(frequency domain)에서 새롭게 정의합니다. 또한, 각도(domain) 내에서의 스톡스 벡터 필드에 대한 구형 컨볼루션(spherical convolution)을 정의하여 복잡한 빛의 상호작용과 이동을 효율적으로 계산할 수 있도록 지원합니다. 이를 통해, 주파수 도메인에서의 편광 빛 트랜스포트(polarized light transport) 계산이 효율적으로 이루어집니다.

- **Performance Highlights**: 결과적으로, 이 연구는 편광 환경 조명 아래에서 실시간 편광 렌더링 기법인 Precomputed Polarized Radiance Transfer를 처음으로 개발하게 되었습니다. 제안된 방법은 복잡한 반사 현상의 편광 빛 상호작용을 효과적이고 정확하게 시뮬레이션할 수 있음을 보여줍니다. 이 연구에서의 성과는 향후 다양한 분야에서 편광 빛 렌더링의 응용 가능성을 크게 확장할 것으로 기대됩니다.



### Dataset Distillation via Committee Voting (https://arxiv.org/abs/2501.07575)
Comments:
          Code at: this https URL

- **What's New**: 이번 연구에서는 데이터셋 증류(Dataset Distillation) 분야에 새로운 접근 방식인 CV-DD(Committee Voting for Dataset Distillation)를 제안합니다. 이 방법은 다수 모델의 집단 지혜를 활용하여 고품질의 증류된 데이터셋을 생성합니다. 기존의 연구에서는 주로 단일 모델 또는 복수 모델 간의 정렬(alignment)과 효율성 향상에 초점을 맞췄으나, CV-DD는 이와는 다른 차별화된 접근을 나타냅니다.

- **Technical Details**: CV-DD 프레임워크는 Prior Performance Guided (PPG) Voting Mechanism을 도입하였으며, 여러 모델의 예측 및 분포를 집계하여 대표적인 데이터 포인트를 식별합니다. 이 방법은 모델 특유의 편향을 줄이고, 증류된 데이터셋의 다양성을 증진시키며, 각 모델의 고유한 장점을 활용해 과적합(overfitting)을 완화합니다. 또한, 동적 투표 설계를 통해 특정 기능 또는 데이터셋 속성을 우선시할 수 있도록 조정할 수 있는 세세한 제어를 제공합니다.

- **Performance Highlights**: CIFAR, Tiny-ImageNet, ImageNet-1K 및 그 하위 집합에 대한 광범위한 실험을 통해 CV-DD는 기존의 단일/복수 모델 증류 방법보다 더 높은 정확도와 모델 간 일반화를 달성함을 입증하였습니다. CV-DD로 증류된 데이터셋은 낮은 데이터나 제한된 컴퓨팅 환경에서도 높은 성능을 나타내며, 여러 시나리오에서 신뢰성과 적응력을 향상시킵니다. 이 연구는 고효율 데이터 사용과 컴퓨팅 효율성이 필수적인 응용 분야에서 CV-DD의 큰 잠재력을 강조합니다.



### UnCommon Objects in 3D (https://arxiv.org/abs/2501.07574)
- **What's New**: uCO3D는 3D 딥 러닝 및 3D 생성 AI를 위한 새로운 객체 중심 데이터셋으로서, 고해상도 비디오와 3D 주석이 포함된 가장 큰 공개 데이터베이스입니다. 이 데이터셋은 전방위 360도 커버리지를 제공하고 있으며, 1,000개 이상의 객체 카테고리를 포괄합니다.

- **Technical Details**: uCO3D는 MVImgNet과 CO3Dv2보다 품질이 더욱 우수하며, 수집된 비디오와 3D 주석에 대한 철저한 품질 검사를 거쳤습니다. 이 데이터셋은 3D 카메라 포즈, 깊이 맵(depth maps), 희소 포인트 클라우드(sparse point clouds) 주석을 포함하고 있으며, 각 객체에는 캡션과 3D Gaussian Splat 재구성이 제공됩니다.

- **Performance Highlights**: uCO3D에서 훈련된 여러 대규모 3D 모델은 MVImgNet 및 CO3Dv2에서 훈련된 모델보다 우수한 성능을 보여 주었습니다. 이는 uCO3D가 학습 응용 프로그램에 더 적합하다는 것을 입증하는 결과입니다.



### Training-Free Motion-Guided Video Generation with Enhanced Temporal Consistency Using Motion Consistency Loss (https://arxiv.org/abs/2501.07563)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 모션 가이드를 사용하여 시간 일관성을 유지하는 비디오 생성의 도전 과제를 다루고 있습니다. 기존의 방법들이 추가적인 제어 모듈이나 추론 시간 조정을 필요로 하는 반면, 새로운 연구들은 모델 구조를 변경하지 않고도 효과적인 모션 가이드를 달성할 수 있음을 시사합니다. 이 작업은 다양한 비디오 생성 기반 모델과의 호환성이 뛰어난 접근 방식을 제공하게 됩니다.

- **Technical Details**: 우리는 초기 노이즈 기반 접근 방식을 기존 비디오 확산 모델의 중간 특징들 사이의 프레임 간 상관 패턴과 결합하여 참조 비디오의 모션 패턴을 표현합니다. 그 후, 생성된 비디오에서 유사한 특징 상관 패턴을 유지하기 위해 모션 일관성 손실 함수를 설계합니다. 이를 통해 잠재 공간에서 이 손실의 그래디언트를 사용하여 정밀한 모션 제어를 위한 생성 프로세스를 안내합니다.

- **Performance Highlights**: 우리는 다양한 모션 제어 작업에서 시간 일관성을 개선하며, 훈련이 필요 없는 설정의 이점을 유지합니다. 실험 결과, 우리의 방법이 효율적이고 시간적으로 일관된 비디오 생성의 새로운 기준을 설정하는 것을 보여줍니다. 특히, 선택적으로 몇 가지 주요 포인트에 초점을 맞춤으로써 불필요한 강제 모션 전이를 피하고 높은 시간 일관성을 얻는 데 성공했습니다.



### MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training (https://arxiv.org/abs/2501.07556)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 다양한 소스에서 합성된 크로스 모달(training signals) 훈련 신호를 활용한 대규모 사전 훈련(framework)을 제안합니다. 이는 이미지 간의 기본 구조를 인식하고 매칭하는 모델을 훈련시키기 위한 것입니다. 이러한 접근 방식은 다양한 이미지 모달리티를 효과적으로 다룰 수 있는 가능성을 열어줍니다.

- **Technical Details**: 제안된 프레임워크는 각기 다른 이미징 모달리티에서 캡처된 이미지 간의 매칭 문제를 해결하기 위해 설계되었습니다. 연구 결과, 우리의 프레임워크로 훈련된 매칭 모델이 8개 이상의 크로스 모달 등록 작업에서 뛰어난 일반화 성능을 보였습니다. 이는 기존의 매칭 알고리즘들보다 상당히 향상된 성능을 보여줍니다.

- **Performance Highlights**: 이 연구의 핵심 발견은 동일한 네트워크 가중치 사용 시, 제안된 모델이 다양한 크로스 모달 이미지 매칭 작업에 대해 뛰어난 일반화능력을 보인다는 것입니다. 이는 기존 방법과 비교하여 일반화나 특정 작업을 위해 설계된 경우를 포함하여 모두 뛰어난 성능으로 주목받고 있습니다. 이러한 발전은 다양한 과학 분야에서 이미지 매칭 기술의 적용 가능성을 높이고, 다중 모달 인간 및 인공지능 분석을 위한 새로운 응용 프로그램의 길을 열어줍니다.



### SST-EM: Advanced Metrics for Evaluating Semantic, Spatial and Temporal Aspects in Video Editing (https://arxiv.org/abs/2501.07554)
Comments:
          WACV workshop

- **What's New**: 이 논문에서는 비디오 편집 모델의 성능 평가를 위해 SST-EM(모든 3차원적 평가 기준을 반영한 새로운 평가 메트릭)을 제안하고 있습니다. 전통적인 평가 지표의 한계를 극복하기 위해 Vision-Language Models (VLMs), Object Detection, Temporal Consistency 체크를 통합한 새로운 방식으로 구성을 하였습니다. 이 메트릭은 인간 평가를 기반으로 최적화된 가중치를 포함하여, 비디오 편집에서 의미적 충실도와 시간적 매끄러움을 종합적으로 평가하도록 설계되었습니다.

- **Technical Details**: SST-EM 메트릭은 네 가지 주요 구성 요소로 이루어져 있습니다: (1) VLM을 사용한 프레임의 의미적 추출, (2) Object Detection을 통한 주요 객체 추적, (3) LLM 에이전트를 이용한 주요 객체의 세밀한 다듬기, 그리고 (4) Vision Transformer (ViT)를 통한 시간적 일관성 평가입니다. 이 메트릭은 인간의 평가 결과와 회귀 분석을 통해 가중치를 조정하여 통합적인 평가 프레임워크를 제공합니다. 이러한 구성 요소들은 비디오 편집의 여러 차원을 동시에 고려하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 논문에서는 SST-EM 메트릭이 기존의 평가 방법보다 우수함을 입증하였으며, 이를 통해 비디오 편집 연구에서 더 신뢰할 수 있는 성능 기준을 제공하고 있습니다. 다양한 비디오 편집 모델에서의 성능을 평가하며 인간 검증에 기반한 결과를 참조하여, 높은 정확성을 자랑하는 신뢰할 수 있는 메트릭으로 자리잡게 되었습니다. SST-EM은 비디오 편집 품질의 평가에 있어 보다 견고한 틀을 제공하며, 관련 연구의 방향성을 제시하고 있습니다.



### Confident Pseudo-labeled Diffusion Augmentation for Canine Cardiomegaly Detection (https://arxiv.org/abs/2501.07533)
Comments:
          WACV workshop

- **What's New**: 이번 연구에서는 개의 심비대(cardiomegaly)를 식별하기 위한 Confident Pseudo-labeled Diffusion Augmentation (CDA) 모델을 제안합니다. 이 모델은 고품질 트레이닝 데이터의 부족 문제를 해결하기 위해 확산 모델(diffusion models)을 활용하여 합성 X-레이 이미지를 생성하고, Vertebral Heart Score(VHS) 주요 포인트를 주석으로 추가합니다. 또한, Monte Carlo Dropout 기법을 사용하여 높은 신뢰도를 가진 레이블을 선택함으로써 정확성을 높이고 기본 데이터 세트를 정제합니다.

- **Technical Details**: CDA 모델은 데이터 증강(data augmentation)과 높은 신뢰도의 의사 레이블링(pseudo-labeling) 전략을 통합하여 개의 심비대 진단의 정확성을 높이는 혁신적인 방법론입니다. 특히 이 모델은 확산 모델을 사용하여 3,000개의 합성 개胸 X-레이 이미지를 생성하고, VHS 점수로 주석을 달아 데이터 세트를 확장합니다. 이러한 방식은 기존 방법의 한계를 극복하고, 반복적으로 높은 신뢰도의 레이블을 모델에 통합하여 성능을 개선하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과에 따르면 CDA 모델은 기존 전통적인 방법보다 뛰어난 성능을 보여주며, 개 심비대 탐지에서 최첨단 정확도를 기록했습니다. 이러한 성과는 CDA 모델이 데이터를 효율적으로 다루고, 변동성이 큰 X-레이 품질 문제를 극복할 수 있음을 나타냅니다. 이 연구는 합성 데이터와 의사 레이블링의 통합이 수의학 진단 분야에서 AI의 발전을 이끌 수 있음을 보여줍니다.



### IP-FaceDiff: Identity-Preserving Facial Video Editing with Diffusion (https://arxiv.org/abs/2501.07530)
Comments:
          WACV-25 Workshop

- **What's New**: 이번 논문에서는 얼굴 비디오 편집을 위한 새로운 프레임워크를 제안합니다. 기존 모델들이 고품질 편집, 낮은 계산 비용, 그리고 다양한 수정에서도 정체성을 유지하는 데에 한계를 보이는 문제를 해결하고자 합니다. 제안된 방법은 사전 훈련된 Text-to-Image (T2I) diffusion 모델을 활용하여 얼굴 비디오 편집을 도와주며, 고품질의 텍스트 기반 편집을 가능하게 합니다.

- **Technical Details**: 우리의 접근법은 두 개의 독립적으로 조정된 diffusion 모델을 사용하여 얼굴 비디오에서 고도로 세분화된 편집을 수행합니다. 방향성 CLIP 손실을 통해 세부적인 편집이 가능하도록 학습하였으며, 정체성을 유지하는 손실도 통합하여 원본 영상의 인물 정체성을 보존하는 데 중점을 두었습니다. 이러한 전략을 통해, 제안된 방법은 다양한 도전 시나리오에서 일관되게 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 기술들과 비교해 80%의 편집 시간을 단축시키면서도, 영상의 시간적 일관성을 유지하는 데 성공했습니다. 여러 가지 포즈 변화, 복잡한 동작 시퀀스, 다양한 얼굴 표정에 대해 광범위한 테스트를 수행하였고, 그 결과 기준 지표에서 기존 방법들을 능가하는 성능을 입증하였습니다. 이는 얼굴 비디오 편집 분야의 새로운 기준을 제시할 것으로 기대됩니다.



### RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignmen (https://arxiv.org/abs/2501.07525)
- **What's New**: 본 논문에서는 RadAlign이라는 새로운 프레임워크를 제안합니다. RadAlign은 병리학적 진단의 정확성과 해석 가능성을 동시에 중시하면서 방사선 사진 해석과 보고서 생성을 통합합니다. 이 프레임워크는 시각-언어 모델(Vision-Language Models, VLM)과 대형 언어 모델(Large Language Models, LLM)의 장점을 결합하여 의사 소통의 품질을 향상시키고 의료 영상 분석의 신뢰성을 높입니다.

- **Technical Details**: RadAlign은 먼저 VLM을 사용하여 시각적 특징을 의료 개념과 정렬합니다. 이는 방사선과 의사가 특정 진단 기준을 확인하고 이를 바탕으로 판단하는 과정을 모방합니다. 그런 다음 정렬된 시각적-언어 공간에서 저희가 인식한 질병을 텍스트 기반 개념으로 표현하고, 이를 활용해 LLM에 의해 보고서를 생성하도록 유도합니다.

- **Performance Highlights**: RadAlign은 다양한 질병에 대해 평균 AUC가 0.885로 우수한 질병 분류 성능을 달성합니다. 보고서 품질은 GREEN 점수가 0.678로, 현재까지의 최첨단 방법인 0.634를 초과합니다. 이 프레임워크는 임상 해석 가능성을 유지하면서 착각을 줄이고, 통합된 예측 및 생성 AI를 통해 자동화된 의료 영상 및 보고서 분석을 발전시킵니다.



### Three-view Focal Length Recovery From Homographies (https://arxiv.org/abs/2501.07499)
Comments:
          Code available at this https URL Dataset available at: this https URL

- **What's New**: 이 논문에서는 세 개의 뷰에서의 호모그래피(homography)로부터 초점 거리(focal length)를 회복하는 새로운 접근법을 제안합니다. 두 개의 호모그래피 사이의 법선 벡터(normal vector)의 일관성을 검토하여, 제거 기법(elimination technique)을 사용해 초점 거리와 호모그래피 간의 새로운 명시적 제약 조건을 도출합니다. 또한, 이 방법이 세 개의 뷰 호모그래피가 제공하는 두 가지 추가 제약을 활용하여 초점 거리를 회복할 수 있다는 것을 보여줍니다.

- **Technical Details**: 논문에서 논의되는 네 가지 경우는 다음과 같습니다: 세 대의 카메라가 동일한 미지의 초점 거리를 가지는 경우, 두 개의 서로 다른 미지의 초점 거리를 가지는 경우, 하나의 초점 거리가 알려진 경우, 그리고 나머지 두 대의 카메라가 동일하거나 다른 미지의 초점 거리를 가지는 경우입니다. 모든 문제는 하나 또는 두 개의 미지수에 대한 다항식(polynomial) 문제로 변환할 수 있으며, Sturm 시퀀스(Sturm sequence) 또는 숨겨진 변수(hidden variable) 기법을 사용하여 효율적으로 해결할 수 있습니다.

- **Performance Highlights**: 합성 데이터(synthetic data) 및 실제 데이터(real data)를 이용한 평가 결과, 제안된 솔버는 기존의 두 뷰 솔버에 의존하는 방법들보다 더 빠르고 정확한 성능을 보였습니다.코드와 데이터는 제공된 URL에서 접근할 수 있습니다.



### Aligning First, Then Fusing: A Novel Weakly Supervised Multimodal Violence Detection Method (https://arxiv.org/abs/2501.07496)
- **What's New**: 이번 연구에서는 약한 감독 방식의 폭력 탐지(Weakly Supervised Violence Detection)에서 비디오 레벨 레이블만으로 폭력 구간을 식별하는 기술을 제안합니다. 특히, 시청각 데이터와 같은 다중 모드를 통합하여 효과적으로 폭력 사건을 인식하는 새로운 접근 방식을 도입했습니다. 기존 방법들은 주로 다중 모드 융합 모델의 설계에 초점을 맞추었으나, 본 연구는 모드 간의 내재적 불일치를 활용하여 새로운 다중 모드 의미적 특징 정렬 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 비디오 데이터로부터 단일 모드 멀티 인스턴스 학습(Unimodal Multiple-Instance Learning)을 통해 각 모드의 의미적 특징을 추출한 뒤, 동적이고 일관된 정렬 과정을 통해 최적의 의미적 특징 공간을 찾습니다. 이를 통해 얻어진 정보는 최종 폭력 탐지 단계에서 모든 모드의 정보를 최적으로 활용할 수 있게 합니다. 실험에서는 XD-Violence 데이터셋에서 평균 정밀도(Average Precision)를 86.07%로 달성하며, 기존의 관련 연구를 크게 초과하는 성과를 입증하였습니다.

- **Performance Highlights**: 본 연구의 가장 큰 성과는 약한 감독 방식을 통한 폭력 탐지의 정확성을 획기적으로 높인 것입니다. 제안된 방법은 모드 간의 불일치를 효과적으로 극복하고, 다중 모드 폭력 탐지 프레임워크를 통해 성능을 극대화했습니다. 실험 결과, XD-Violence 데이터셋에서 86.07%의 평균 정밀도를 기록하며, 많은 기존 방법들보다도 월등한 성능을 보여주었습니다.



### A Survey on Dynamic Neural Networks: from Computer Vision to Multi-modal Sensor Fusion (https://arxiv.org/abs/2501.07451)
Comments:
          Under review at International Journal of Computer Vision

- **What's New**: 본 논문은 Dynamic Neural Networks(DNNs)를 컴퓨터 비전(Computer Vision) 분야와 센서 융합(Sensor Fusion) 애플리케이션 맥락에서 철저히 조사한 종합적인 설문조사입니다. 기존의 여러 연구를 종합해 명확한 분류법을 제시하며, DNN의 적응성과 센서 융합에서의 이점을 강조합니다. 또한, 다양한 입력 복잡성에 대응하기 위한 계산 경로 조정의 필요성을 논의합니다.

- **Technical Details**: Dynamic Neural Networks는 입력의 복잡성에 따라 계산 양을 조절할 수 있는 심층 학습 모델입니다. 이러한 유연성은 일반적으로 게이팅(gating) 또는 의사 결정 모듈을 통해 구현됩니다. 이 네트워크는 쉽게 인식할 수 있는 간단한 객체와 장면에 대해서는 적은 시간과 노력을 할당하고 복잡한 경우에는 더 많은 주의와 시간을 요구하는 인간 인지 시스템을 본보기로 제시합니다.

- **Performance Highlights**: 동적 신경망의 적응성은 센서 융합 맥락에서 특히 유리합니다. 이러한 기술을 통해 모델의 효율성을 높이고 비효율적인 환경으로 인한 잡음에 견고함을 제공하며, 중요 정보를 식별하는 능력이 내재적으로 개선됩니다. 논문에서 제시는 다양한 동적 신경망 기술이 센서 융합 작업에 어떻게 응용될 수 있는지를 보여주며, 이는 다양한 환경에서 더 나은 주변 인식에 기여할 수 있습니다.



### Guided SAM: Label-Efficient Part Segmentation (https://arxiv.org/abs/2501.07434)
- **What's New**: 이 논문에서는 Segmentation의 효율성을 높이기 위해 새로운 접근 방식인 'Guided SAM'을 제안합니다. 이는 필요 최소한의 레이블 데이터로도 Object part segmentation을 가능하게 합니다. 기존 SAM(Segment-Anything Model)은 수작업으로 주장된 Positional prompts가 필요했으나, 이제는 조잡한 패치 주석을 활용하여 이 prompts를 자동화하여 효율성을 증가시킵니다. 이 방법은 특히 자동차 부품 데이터셋에서 평균 IoU를 개선하여 최신 기술을 초월하는 성능을 입증하였습니다.

- **Technical Details**: Guided SAM은 자동차 부품과 같은 개체의 일부를 분할하는 데 중점을 두고 있으며, 인식된 patches에서 positional prompts를 학습합니다. 이 패치들은 이미지의 약 1/14 너비와 높이를 차지하며, 이러한 조잡한 주석법은 픽셀 정밀 마스크보다 훨씬 효율적입니다. 모델은 DINOv2를 사용하여 패치의 표현력을 높이고, 지정된 패치에서 지역성을 고려하여 자원을 최적화합니다. 또한, ROI(Region of Interest)를 설정하여 유용한 맥락 정보를 제공하며 계산 비용을 줄입니다.

- **Performance Highlights**: Guided SAM은 자동차 부품의 다양성을 고려하여 설계된 실험에서 성능을 평가하며, 단지 16에서 64장의 이미지로도 효과적인 segmentation 모델을 학습할 수 있음을 보여줍니다. 이에 따라 기존 모델인 Grounding SAM과 VLPart와 비교 시 더 높은 성능을 발휘하여 IoU를 0.49로 개선하였습니다. 이 접근 방식은 평균적으로 이미지당 단지 5번의 클릭만으로 필요한 주석을 효율적으로 수집이 가능하여, 시간과 노력을 크게 절감합니다.



### Diff-Ensembler: Learning to Ensemble 2D Diffusion Models for Volume-to-Volume Medical Image Translation (https://arxiv.org/abs/2501.07430)
- **What's New**: Diff-Ensembler는 기존의 모델들이 겪는 3D 표현의 한계를 극복하기 위해 다각적으로 훈련된 2D diffusion 모델과 3D 네트워크를 결합한 새로운 하이브리드 모델입니다. 이 모델은 효율적인 volumetric translation을 가능하게 하여 의료 이미지의 정밀도와 현실감을 향상시킵니다. 또한, Diff-Ensembler는 다양한 입력 모달리티에 대한 조합을 자연스럽게 활용할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: Diff-Ensembler는 두 단계의 훈련 전략을 채택합니다: 첫 번째로, 수직 방향에서 여러 2D diffusion 모델을 훈련하고, 두 번째로 3D 융합 네트워크를 사용하여 각 diffusion 단계에서 최종 번역을 생성합니다. 이는 2D 모델에서 얻은 계층적 특성 맵을 효율적으로 활용하여 3D 출력을 더욱 정확하고 공간적으로 현실감 있게 만듭니다. 또한, Mixture-of-Experts (MoE) 접근법을 사용하여 2D 모델을 혼합하여 더 효과적인 성능을 냅니다.

- **Performance Highlights**: Diff-Ensembler의 성능은 BraTS 및 HCP 데이터셋을 활용한 다양한 MRI 이미지 처리 작업에서 평가되었으며, 기존 모델 대비 우수한 성능을 보여주었습니다. 이 모델은 새로운 2D 모델을 재훈련하지 않고도 다각적 입력 모달리티가 주어졌을 때 강력한 성능을 발휘합니다. 실험 결과, 특히 3D 의료 이미지의 super-resolution 및 모달리티 번역 작업에서 뛰어난 volumetric realism을 달성했음을 확인했습니다.



### OCORD: Open-Campus Object Removal Datas (https://arxiv.org/abs/2501.07397)
Comments:
          technical report

- **What's New**: 이 논문은 고해상도의 실제 세계에서 수집한 데이터셋을 활용하여 객체 제거(object removal) 작업을 개선하는 새로운 접근 방식을 소개합니다. 특히, 고정된 카메라 설정으로 장기간 비디오를 캡처하고, Grounding-DINO, Segment-Anything-Model 및 MASA를 활용하여 자동으로 주석을 달아 이미지, 배경, 마스크 쌍을 제공합니다. 이는 기존의 데이터셋에서 직면했던 한계, 즉 근본적인 현실감 부족과 주석 비용 문제를 해결합니다.

- **Technical Details**: 연구진은 장시간 비디오 캡처 기법을 통해 고해상도 자연 장면 데이터셋을 구축하였고, 이는 이전의 데이터셋들과 비교하여 더욱 실제적인 다양성을 반영합니다. 또한, 자동화된 주석 프로세스를 도입하여 자원과 시간을 절약하면서도 높은 정확성을 유지합니다. 이러한 접근은 객체 제거 모델 훈련에 필요한 데이터의 질과 양을 더욱 향상시킵니다.

- **Performance Highlights**: 제안된 방법에 의해 구축된 데이터셋은 대규모 미리 훈련된 diffusion 모델의 성능을 개선할 수 있는 기반이 됩니다. 기존의 최첨단 모델들과 비교했을 때, 더욱 향상된 객체 제거 작업 성능을 보여주며, 이는 이 분야의 향후 발전 가능성을 시사합니다. 연구의 최종 목표는 이미지 기반 객체 제거의 경계를 넓히기 위한 것입니다.



### Zero-Shot Scene Understanding for Automatic Target Recognition Using Large Vision-Language Models (https://arxiv.org/abs/2501.07396)
- **What's New**: 이번 연구에서는 Automatic Target Recognition (ATR) 시스템의 제한을 극복하기 위해, open-world 검출기의 탐지 능력과 Large Vision-Language Models (LVLMs)의 인식 신뢰성을 결합한 새로운 파이프라인을 제안합니다. 이 시스템은 알려지지 않은 환경 및 신종 물체 클래스에 대한 zero-shot 탐지와 인식을 가능하게 합니다. 특히, 군용 차량 인식을 위한 다양한 LVLM의 성능을 비교하였으며, 거리 범위, 모달리티 및 프롬프트 방법이 인식 성능에 미치는 영향을 분석하였습니다.

- **Technical Details**: ATR의 신뢰성 있는 시스템은 동적이고 불확실한 환경에서 안전성과 강인성을 보장하는 데 중요합니다. 기존의 전통적인 모델들은 고정된 클래스 세트를 가정하여 작동하는 반면, open-world 객체 탐지기는 새로운 객체 유형을 실시간으로 인식할 수 있도록 설계되었습니다. 본 연구에서는 YOLO-world 객체 탐지기를 사용하여 이미지 내의 객체를 빠르게 찾아내고, 이후 LVLM을 통해 재평가하여 객체의 라벨을 지정하는 두 단계의 파이프라인을 구현합니다.

- **Performance Highlights**: 이 파이프라인은 zero-shot 환경에서 신종 클래스와 미지의 도메인에 대해 객체 탐지 및 인식을 수행할 수 있는 강력한 시스템을 형성합니다. LVLM은 과거 학습된 정보 없이 새로운 물체를 인식하는 데 도움을 주며, 동물 관찰 사례에서 현저한 성능 향상을 보여줍니다. 본 연구는 이미지 스케일과 모달리티가 LVLM의 ATR 성능에 미치는 중요한 영향을 체계적으로 조사하여, 실제 환경에서의 ATR 최적화에 대한 지침을 제공합니다.



### Kolmogorov-Arnold Network for Remote Sensing Image Semantic Segmentation (https://arxiv.org/abs/2501.07390)
Comments:
          13 pages, 8 figures

- **What's New**: 본 논문에서는 새로운 의미적 분할 네트워크인 DeepKANSeg를 제안하고 있습니다. 이 네트워크는 Kolmogorov Arnold Network (KAN) 기술을 기반으로 하여 고차원 복잡한 기능을 단변수 변환으로 분해하는 능력을 활용합니다. 이는 데이터의 복잡한 관계를 효율적으로 표현하고 효과적으로 복원할 수 있게 합니다.

- **Technical Details**: DeepKANSeg는 두 가지 주요 구성 요소를 포함합니다. 첫째, DeepKAN 모듈이 고차원 특성에서 복잡한 공간 및 풍부한 의미적 관계를 효과적으로 포착합니다. 둘째, GLKAN 모듈이 전통적인 MLP 층을 대체하여 디코딩 과정에서 세부 정보를 보다 효과적으로 캡처할 수 있도록 합니다.

- **Performance Highlights**: ISPRS Vaihingen 및 ISPRS Potsdam과 같은 두 개의 잘 알려진 고해상도 원격 감지 벤치마크 데이터 세트에서 실험을 수행하였으며, KAN 기술이 기존 방법보다 높은 정확도를 달성하는 데 기여함을 보여주었습니다. KAN은 의미적 분할 작업에서 전통적인 아키텍처에 비해 강력한 대안이 될 수 있음을 강조합니다.



### FedSemiDG: Domain Generalized Federated Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2501.07378)
Comments:
          17 pages

- **What's New**: 이 논문은 의료 이미지 분할에서의 문제인 domain generalized federated semi-supervised learning (FedSemiDG)이라는 새로운 문제 설정을 제안합니다. 이는 한정된 레이블 데이터와 풍부한 비레이블 데이터로부터 여러 도메인에서 모델을 분산 학습하여, 보지 못한 도메인에서도 잘 일반화할 수 있도록 하는 목표를 가지고 있습니다. 또한, 새로운 프레임워크인 Federated Generalization-Aware Semi-Supervised Learning (FGASL)을 통해 기존의 문제점들을 해결하고자 합니다.

- **Technical Details**: FGASL에서는 글로벌 모델의 일반화 성능을 높이기 위해 Generalization-Aware Aggregation (GAA) 기법을 도입하여, 각 로컬 모델의 일반화 성능을 기반으로 적응적 가중치를 할당합니다. 로컬에서는 Dual-Teacher Adaptive Pseudo Label Refinement (DR) 전략을 사용하여, 글로벌과 도메인 특정 지식을 통합하여 더 신뢰할 수 있는 가짜 레이블을 생성합니다. Perturbation-Invariant Alignment (PIA) 기법을 통해 도메인 불변 학습을 촉진하며, 높은 변동성을 배제하여 모델 성능을 개선합니다.

- **Performance Highlights**: 세 가지 의료 분할 작업(심장 MRI, 척추 MRI, 방광암 MRI)에서 광범위한 실험을 수행한 결과, FGASL은 기존의 FSSL 및 도메인 일반화 방법을 능가하는 성능을 나타냈습니다. 특히, 보지 못한 도메인에서의 견고한 일반화를 달성하며, 모델 성능이 크게 향상됨을 입증했습니다. 이러한 성과는 FedSemiDG 문제 설정 하에서의 효과성을 보여줍니다.



### TimberVision: A Multi-Task Dataset and Framework for Log-Component Segmentation and Tracking in Autonomous Forestry Operations (https://arxiv.org/abs/2501.07360)
Comments:
          Accepted at Winter Conference on Applications of Computer Vision (WACV) 2025. Code and dataset available at this https URL

- **What's New**: 본 연구에서는 TimberVision 데이터셋을 소개합니다. 이 데이터셋은 2천 개 이상의 주석이 달린 RGB 이미지로 구성되며, 총 5만 1천 개의 줄기 구성 요소를 포함하고 있어 기존 데이터셋에 비해 규모와 세부 사항에서 큰 차별성을 보입니다. 게다가, 이 데이터셋은 다양한 장면 매개변수들의 모델 성능에 미치는 영향을 분석하기 위한 일련의 ablation 실험을 수행했습니다.

- **Technical Details**: TimberVision 데이터셋은 나무줄기와 그 구성 요소를 다양한 응용 프로그램 시나리오에서 차별화된 학습을 가능하게 하는 구조로 제공됩니다. 데이터셋은 RGB 이미지에서 줄기의 위치와 방향을 유도할 수 있는 방법론을 포함하고 있으며, 이를 통해 로그 배치 처리 및 중량 중심 추정이 가능합니다. 실시간 객체 감지(Oriented Object Detection)와 인스턴스 분할(Instance Segmentation)에 대한 종합적인 연구도 진행되었으며, 이를 통해 여러 가지 장면 배치에 대한 효율성을 극대화했습니다.

- **Performance Highlights**: 최신 알고리즘을 통해 RGB 이미지 데이터만으로도 나무줄기를 높은 정의로 정확하게 표현할 수 있습니다. 이 방식은 도전적인 환경에서도 높은 내구성을 보이며, 여러 센서 모드와 쉽게 결합하여 사용할 수 있는 유연성을 가집니다. 이러한 접근 방식은 나무 자원의 자동화된 수확, 운반, 측정에 적합하여 작업자의 안전을 더욱 높이는 데 기여할 것입니다.



### A method for estimating roadway billboard salienc (https://arxiv.org/abs/2501.07342)
- **What's New**: 이 연구는 도로변 광고의 중요성을 비(視)전문가 관점에서 탐구하며, 이 광고가 운전자를 분산시켜 안전에 미치는 영향을 파악하고자 합니다. YOLOv5와 Faster R-CNN 모델을 활용하여 도로의 광고 공간을 감지하는 데 있어 신경망의 효과성을 검토합니다. 또한, UniSal과 SpectralResidual 방법을 통해 광고의 중요성을 평가하는 방법을 제시하고, 실제 시나리오에 기반한 주시 추적 데이터를 마련하여 연구 결과를 보강합니다.

- **Technical Details**: 본 연구에서는 두 개의 데이터셋을 사용합니다. 첫 번째 데이터셋은 Mapillary Vistas Dataset으로, 도시의 다양한 장면을 포함한 25,000장의 이미지로 구성되어 있습니다. 두 번째 데이터셋은 차량 대시보드 이미지를 기반으로 하여 고속도로 및 도시 도로와 같은 다양한 주행 시나리오를 표현하는 1580장의 사용자 정의 이미지로 이루어진 것입니다.

- **Performance Highlights**: 제안된 방법은 광고 지역의 검출, 입력 이미지의 중요도 맵 생성, 광고 공간의 평균 중요도 평가를 포함하는 일련의 단계를 수행하여 광고의 시각적 주목도를 향상시키는 것을 목표로 합니다. 더불어, 실험을 통해 고급 기능 도출 및 효과적인 광고 감지를 위한 개선된 성과를 제시할 것으로 기대합니다.



### Localization-Aware Multi-Scale Representation Learning for Repetitive Action Counting (https://arxiv.org/abs/2501.07312)
Comments:
          Accepted by IEEE VCIP2024

- **What's New**: 이 논문에서는 비디오에서의 반복적인 동작 카운팅(Repetitive Action Counting, RAC) 방법을 개선하기 위해 포그라운드 로컬라이제이션 최적화 목표를 도입했습니다. 새로운 Localization-Aware Multi-Scale Representation Learning (LMRL) 프레임워크를 제안하여, 다양한 동작 빈도를 처리하고 보다 유연한 시간적 상관관계를 학습할 수 있도록 설계하였습니다. 이를 통해 노이즈의 영향을 줄이고, 보다 정확한 카운팅 성능을 달성하는 방법을 제시합니다.

- **Technical Details**: LMRL 프레임워크는 비디오 기능 추출기, 주기적 표현, 주기 예측기로 구성되어 있습니다. 이 방법은 Multi-Scale Period-Aware Representation (MPR)과 Repetition Foreground Localization (RFL) 모듈을 활용하여 반복적인 동작의 중요 정보를 추출합니다. MPR은 다양한 동작 빈도에 맞춰 설계되어 있으며, RFL은 주기적인 동작을 식별하는 데 도움을 줍니다.

- **Performance Highlights**: RepCountA 및 UCFRep 데이터셋에서 실험 결과, 제안한 방법이 매우 효과적으로 반복 동작 카운팅을 처리함을 보여주었습니다. 기존 방법들이 현실적인 상황에서 성능 저하를 경험하는 것과 달리, 본 방법은 노이즈에 강하고, 다른 유형의 비디오 콘텐츠에 적응 가능하다는 큰 장점을 가지고 있습니다. 이러한 성과는 반복적인 동작 카운팅의 정확도를 개선하는 데 기여합니다.



### The Devil is in the Spurious Correlation: Boosting Moment Retrieval via Temporal Dynamic Learning (https://arxiv.org/abs/2501.07305)
- **What's New**: 이 논문에서는 자연어 쿼리에 해당하는 비디오에서 관련 순간을 검색하는 문제를 다룹니다. 기존 transformer 기반 접근 방식이 좋은 결과를 보여주었지만, 목표 순간의 정확한 시간 범위를 예측하는 데 여전히 대첵이 필요합니다. 제안된 방법은 spurious correlation(잘못된 상관관계) 문제를 해결하는 데 중점을 두며, 이는 모델이 텍스트 쿼리와 순간 맥락 간에 잘못된 연관성을 가지며 발생합니다.

- **Technical Details**: 제안된 temporal dynamic learning 접근 방식은 spurious correlation을 완화하기 위한 두 가지 전략을 포함합니다. 첫 번째로, 새로운 video synthesis 방법을 도입하여 검색할 대상 순간의 동적 맥락을 구축합니다. 두 번째로, 텍스트 쿼리와 temporal dynamic representation의 정렬을 통해 표현을 향상시킵니다. 이 과정을 통해 모델은 쿼리와 관련된 순간과 맥락 간에 비스푸리어스한 관계를 설정할 수 있습니다.

- **Performance Highlights**: 우리의 방법은 QVHighlights 및 Charades-STA와 같은 두 가지 인기 벤치마크에서 새로운 최첨단 성능을 달성했습니다. 제안된 방법은 이전 방법보다 뚜렷한 성능 향상을 보여줍니다. 또한, 상세한 ablation 분석을 통해 제안된 전략의 효율성이 입증되었습니다.



### Code and Pixels: Multi-Modal Contrastive Pre-training for Enhanced Tabular Data Analysis (https://arxiv.org/abs/2501.07304)
- **What's New**: 이번 연구에서는 Multi-task Contrastive Masked Tabular Modeling (MT-CMTM)이라는 혁신적인 방법을 제안하였습니다. 이 방법은 tabular 데이터와 이미지 간의 상관관계를 활용하여 tabular 모델을 개선하는 것을 목표로 하고 있습니다. MT-CMTM은 대조 학습(contrastive learning)과 마스킹된 탭 데이터 모델링(masked tabular modeling)을 결합한 이중 전략을 사용합니다.

- **Technical Details**: MT-CMTM의 핵심은 1D-ResNet-CBAM 구조로, 잔차 연결과 주의 메커니즘(attention mechanism)을 갖춘 1차원 합성곱 신경망(1D Convolutional Neural Network)입니다. 이 모델은 입력 데이터를 효율적으로 처리할 수 있으며, 별도의 이미지 의존성 없이도 downstream 작업을 수행할 수 있게 설계되었습니다. 새로운 HIPMP 데이터세트와 DVM 데이터세트에서 MT-CMTM의 성능이 실험적으로 검증되었습니다.

- **Performance Highlights**: MT-CMTM 모델은 DVM 및 HIPMP 데이터세트에서 우수한 성능을 보였습니다. HIPMP 데이터세트에서 상대 평균 제곱 오차(relative MSE)를 1.48% 개선하고, DVM 데이터세트에서는 절대 정확도를 2.38% 증가시켰습니다. 이러한 결과는 MT-CMTM의 견고성과 다중 모달 학습(multi-modal learning) 분야에 기여할 가능성을 보여줍니다.



### Toward Realistic Camouflaged Object Detection: Benchmarks and Method (https://arxiv.org/abs/2501.07297)
- **What's New**: 이 논문에서는 위장 물체 탐지(COD)에서 객체 탐지 알고리즘을 통해 실용적인 위장 물체 탐지(Realistic Camouflaged Object Detection, RCOD)를 위한 최적화된 솔루션을 제안합니다. 기존의 세분화(scaling) 방법은 객체의 윤곽을 파악하는 데는 성공했지만, 객체의 특정 위치만을 요구하는 과제에서는 비효율적일 수 있습니다. 새로운 데이터셋인 COD10K-D, NC4K-D, CAMO-D를 수작업으로 주석 처리하여 RCOD 연구를 위한 귀중한 벤치마크를 구축했습니다.

- **Technical Details**: 위장 물체 탐지를 개선하기 위해, 본 연구에서는 위장 인식 피쳐 정제(camo-aware feature refinement, CAFR) 전략을 제안합니다. 이 전략은 대형 탐지 모델 내에서 현재 객체를 명확하게 인식하는 데 필요한 정보를 활용하여 배경과 전경의 차이를 깊이 이해하도록 돕습니다. 특히 Adaptive Gradient Propagation (AGP) 모듈과 Sparse Feature Refinement (SFR) 모듈을 도입하여 위장 시나리오에서 클래스별 특징을 정제하고, 희소 피쳐 상황에서 클래스 특징에 집중할 수 있도록 최적화했습니다.

- **Performance Highlights**: 실험 결과 CAFR 전략이 모델의 전경 및 배경 인식 능력을 향상시켜 RCOD 작업의 성능을 크게 개선함을 보여줍니다. 특히 대형 모델을 조정하여 다양한 객체의 개별 특징을 더 효과적으로 탐지할 수 있는 가능성을 열었습니다. 마지막으로, 제안된 전략이 검증된 데이터셋에서의 성능을 향상시키는 데 기여함을 입증했습니다.



### Event-based Video Person Re-identification via Cross-Modality and Temporal Collaboration (https://arxiv.org/abs/2501.07296)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 연구는 비디오 기반의 사람 재식별(Person ReID)에서 이벤트 카메라의 데이터를 활용한 새로운 접근 방식을 제안합니다. 이벤트 데이터만을 사용하여 개인의 프라이버시를 보호하면서도 인식의 정확성을 높일 수 있는 방법론을 개발하였습니다. 특히, Cross-Modality and Temporal Collaboration (CMTC) 네트워크를 통해 이벤트 데이터를 효과적으로 처리하여 보완적인 효과를 극대화하는 방안을 모색하였습니다.

- **Technical Details**: CMTC 네트워크는 이벤트 변환 네트워크(Event Transform Network)를 통해 원시 이벤트의 보조 정보를 추출하고, 모달리티 협업 모듈(Differential Modality Collaboration Module)을 통해 이벤트와 보조 정보의 역할을 균형 있게 조정합니다. 시간 협업 모듈(Temporal Collaboration Module)은 연속 프레임 간의 운동 정보와 시각적 단서를 탐지하여 인식을 증진시키고, 이 모든 과정에서 평균 풀링활용과 특징 인코더를 통해 데이터를 처리합니다. 이는 이벤트 기반의 비디오 사람 재식별 작업에 최적화된 구조입니다.

- **Performance Highlights**: 실험 결과, 개발한 CMTC 네트워크는 기존 방법들에 비해 우수한 성능을 보이며, 공개 데이터셋에서 사람 재식별 작업에서 효과적인 개선을 입증하였습니다. 이벤트 데이터에만 의존함으로써 중요한 인식 단서들이 결여되는 문제를 최소화하며, 공공 장소에서의 프라이버시 손실 문제를 해결하는 데 기여할 수 있습니다. 이 연구는 이벤트 기반 비디오 사람 재식별 작업의 새로운 방향성을 제시한다고 할 수 있습니다.



### Skip Mamba Diffusion for Monocular 3D Semantic Scene Completion (https://arxiv.org/abs/2501.07260)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 모노큘러 이미지를 기반으로 한 독특한 신경망 모델인 Skimba (Skip Mamba) 모델을 제안합니다. 이 모델은 상태 공간(state space) 모델링과 생성적 확산(diffusion) 모델링의 최신 기술 발전을 활용하여 놀라운 3D 의미 장면 완성을 달성합니다. 또한, 변분 오토인코더(VAE)의 조건부 잠재(latent) 공간에서 데이터를 처리하여 효율적이고 효과적인 장면 완성을 지원합니다.

- **Technical Details**: Skimba 모델은 세 개의 방향에서 서로 다른 딜레이션을 사용하는 트리플 맘바(triple Mamba) 구조를 포함하고 있으며, 이로 인해 긴 시퀀스 데이터의 효율적인 처리가 가능합니다. 또한, 이 네트워크는 다양한 블록을 통합하여 3D 장면 완성을 위한 충분한 문맥 정보를 제공합니다. 전반적으로, 제안된 방법은 기본적으로 조건부 VAE와 복합 데이터 전처리 과정을 결합하여 효과적인 분산 기반 노이즈 제거를 지원합니다.

- **Performance Highlights**: SemanticKITTI와 SSCBench-KITTI360 데이터셋에 대한 광범위한 평가 결과, 제안된 방법은 모노큘러 접근 방식 중에서 다른 기술들과 비교하여 월등한 성능을 나타냈습니다. 또한 스테레오 방법에 대해서도 경쟁력 있는 성능을 발휘함으로써 3D 의미 장면 완성 분야에서의 가능성을 보여주고 있습니다. 이 연구는 현재의 기술에 비해 높은 수준의 성능 향상을 달성하였음을 입증합니다.



### EdgeTAM: On-Device Track Anything Mod (https://arxiv.org/abs/2501.07256)
Comments:
          Code will be released at this https URL

- **What's New**: EdgeTAM은 Segment Anything Model (SAM) 2를 기반으로 하여 비디오 입력을 처리하는 데 필요한 효율성을 크게 향상시킨 새로운 모델입니다. 특히 이 모델은 lightweight Transformer 기반의 2D Spatial Perceiver를 도입하여 메모리 주의 집중(memory attention)의 계산 비용을 줄입니다. 이러한 개선은 모바일 장치에서도 원활한 성능을 제공할 수 있도록 설계되었습니다.

- **Technical Details**: EdgeTAM은 과거 프레임을 저장하는 메모리 시스템을 기반으로 하여, 2D Spatial Perceiver를 활용함으로써 계산 비용을 크게 줄여주는 것을 목표로 합니다. 이 구조는 메모리 맵을 압축하면서도 2D 공간 구조를 보존해, 전반적인 성능 저하 없이 메모리 주의 집중 모듈의 효율성을 극대화합니다. 이 과정에서는 Learnerable queries를 활용하여 글로벌 및 패치 레벨의 그룹으로 나누는 기법이 포함됩니다.

- **Performance Highlights**: EdgeTAM은 DAVIS 2017, MOSE, SA-V val, SA-V test에서 각각 87.7, 70.0, 72.3, 71.7 J&F 성능을 기록했습니다. 특히 iPhone 15 Pro Max에서는 16 FPS로 실행되며, 이는 기존의 비디오 객체 분할 모델보다 훨씬 빠릅니다. 후방 전달(distillation) 파이프라인을 도입하여 첨단 영상 분할 모델인 SAM 2에서의 성능을 향상시킴으로써, 보다 나은 정확성을 유지합니다.



### Depth and Image Fusion for Road Obstacle Detection Using Stereo Camera (https://arxiv.org/abs/2501.07245)
Comments:
          8 pages, 15 figures

- **What's New**: 이 논문은 스테레오 카메라를 이용한 깊이 정보와 비디오 분석 방법의 결합을 통해 도로상의 물체를 감지하는 새로운 기법을 소개합니다. 새롭게 개발된 깊이와 이미지 융합 방법은 작은 대비 객체 탐색을 RGB 기반 방법으로 보완하고, SLIC 슈퍼픽셀 분할을 활용한 스테레오 이미지 기반 접근 방식을 통해 장애물 탐지를 향상시킵니다.

- **Technical Details**: 연구에서는 정지 및 저속 장애물이 있는 지하 주차장 실험을 수행하여 개발한 기술의 성공적인 물체 감지 및 추적 능력을 입증하였습니다. 스테레오 카메라를 이용한 깊이 정보를 계산하고 비디오 스트림을 통해 객체 탐지 및 추적 알고리즘을 적용하기 위해 OpenCV 그래프 분할 방법을 사용하였습니다. 또한, 장애물 감지를 위한 깊이 기반 접근 방법을 위해 Stereolabs Zed 2와 Intel RealSense SDK를 이용하여 불균일한 조명과 배경 소음의 영향을 줄이는 기술을 적용했습니다.

- **Performance Highlights**: 본 연구는 특정 유형 및 모양의 장애물이 도로에 나타나는 시간 정보를 사전에 알 수 없을 때 머신러닝 기반 접근법의 한계를 극복하는 데 초점을 맞췄습니다. 개발된 알고리즘은 다양한 크기의 상자와 교통봉 등이 포함된 실험에서 우수한 성능을 보였으며, 필요 시 객체의 크기를 10cm x 10cm 이상으로 탐지할 수 있음을 입증하였습니다.



### Can Vision-Language Models Evaluate Handwritten Math? (https://arxiv.org/abs/2501.07244)
- **What's New**: 최근 Vision-Language Models (VLMs)의 발전은 학생들이 작성한 손글씨 응답의 자동 채점 가능성을 열어주고 있습니다. 그러나 VLM이 손글씨 콘텐츠를 평가하고 추론하는 능력을 테스트한 종합적인 연구는 부족합니다. 이를 해결하기 위해 FERMAT라는 벤치를 소개하여 VLM의 손글씨 수학 콘텐츠에서 오류를 감지하고 수정하는 능력을 평가합니다. 이 벤치는 7-12학년의 609개 수작업 문제에서 유도된 2,200개 이상의 손글씨 수학 솔루션을 포함하고 있습니다.

- **Technical Details**: FERMAT은 VLM의 오류 감지, 로컬라이제이션(localization), 수정 오류를 평가하기 위해 세 가지 핵심 작업을 기반으로 합니다. 실험 결과, 현재 대부분의 모델이 손글씨 입력 처리에서 어려움을 겪으며, Gemini-1.5-Pro가 오류 수정에서 77%의 최고 성과를 기록했습니다. 모델의 성능은 문제 유형이나 오류 카테고리에 대한 추가 메타 정보 제공 시 향상되는 것을 확인했습니다. 손글씨보다 인쇄된 텍스트를 제공할 경우 오류 로컬라이제이션 정확도가 높아지는 것도 주목할 만합니다.

- **Performance Highlights**: 현재 VLM들은 손글씨 텍스트에 대한 추론 능력에서 중요한 단점을 보이고 있습니다. 특히, 다양한 오류 유형에서의 성능을 평가한 결과, VLM들은 기존의 간단한 암기식 평가를 넘어서는 더 복잡한 평가를 요구합니다. 실험을 통해 VLM의 수학적 사고 능력이 제한적임을 발견했고, 이는 더 많은 연구와 개발이 필요함을 시사합니다. FERMAT과 관련 자원들은 공개되어 더 많은 연구를 촉진할 것입니다.



### CSTA: Spatial-Temporal Causal Adaptive Learning for Exemplar-Free Video Class-Incremental Learning (https://arxiv.org/abs/2501.07236)
Comments:
          IEEE TCSVT Submission

- **What's New**: 이번 논문에서 제안한 새로운 접근법은 영상 클래스 증분 학습(video class-incremental learning)에서 기존의 예제 기반 방법을 탈피하여 더욱 향상된 프레임워크를 구현하였습니다. 이 프레임워크는 방대한 사전 훈련 없이도 강력한 일반화 능력을 발휘합니다. 특히, 각 클래스의 고유 특성에 맞춘 공간 및 시간 정보를 학습하는 별도의 어댑터를 도입하였습니다.

- **Technical Details**: 제안한 프레임워크는 동적 네트워크 아키텍처를 활용하여, 각 블록에 새로운 훈련이 가능한 파라미터를 추가하는 방식으로 이전에 학습된 지식을 유지하고 새로운 지식을 나타낼 수 있습니다. 또한, 인과(distillation) 관점에서 두 가지 혁신적인 접근법을 도입하여 공간적 및 시간적 지식 간의 관계를 유지하고 정보를 적절히 보상하는 메커니즘을 제안합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안한 프레임워크가 기존의 SOTA(State-of-the-Art) 방법보다 평균적으로 4.2% 정확도에서 우수한 성능을 나타내며, 저장 공간을 61.9% 절약하는 것도 확인하였습니다. 이러한 결과는 제안된 프레임워크가 영상 클래스 증분 학습 분야에서 신뢰할 수 있는 성능을 발휘함을 입증합니다.



### MECD+: Unlocking Event-Level Causal Graph Discovery for Video Reasoning (https://arxiv.org/abs/2501.07227)
Comments:
          IEEE TPAMI Submission. arXiv admin note: substantial text overlap with arXiv:2409.17647

- **What's New**: 비디오 인과 추론(Video Causal Reasoning) 분야에서 새로운 작업인 다중 사건 인과 발견(Multi-Event Causal Discovery, MECD) 과 데이터셋이 소개되었습니다. 기존 모델들이 주로 단일 사건에 초점을 맞춘 것에 비해, MECD는 장시간 비디오에 걸쳐 시간을 따라 배열된 사건들 간의 인과 관계를 발견하는 것을 목표로 합니다. 이 작업은 각각의 사건이 어떻게 발생했는지를 설명하는 구조화된 인과 그래프를 생성하는 것을 포함합니다.

- **Technical Details**: MECD는 비디오의 시각적 세그먼트와 사건에 대한 텍스트 설명을 바탕으로 인과적 연관성을 식별합니다. 연구진은 Event Granger Test를 수행하기 위해 마스크 기반 사건 예측 모델을 활용하는 새로운 프레임워크 VGCM(비디오 그레인저 인과 모델)을 개발하였습니다. 이 모델은 예측된 결과 사건을 원인 사건이 마스크 처리된 경우와 아닌 경우를 비교하여 인과관계를 추정합니다.

- **Performance Highlights**: VGCM의 유효성과 일반화 가능성은 실험을 통해 검증되었으며, 기존 모델인 GPT-4o 및 VideoChat2에 비해 각각 5.77% 및 2.70% 더 나은 인과 관계 추론 성능을 보였습니다. 추가 실험 결과, VGCM이 생성한 인과 관계 그래프가 비디오 질문 응답(Video Question Answering) 및 비디오 사건 예측(Video Event Prediction)와 같은 하향식 비디오 이해 작업에도 긍정적으로 기여함을 보였습니다.



### Exploring the Use of Contrastive Language-Image Pre-Training for Human Posture Classification: Insights from Yoga Pose Analysis (https://arxiv.org/abs/2501.07221)
- **What's New**: 이 연구는 다양한 분야에서 필수적인 인간 자세 분류의 정확성을 높이기 위해 CLIP (Contrastive Language-Image Pretraining) 모델을 적용하는 새로운 접근 방식을 소개합니다. 특히, 요가의 자세 분류에 초점을 맞추어, 이 모델의 효과성을 실험적으로 검증하였습니다. 연구 결과, 제로샷(zero-shot) 접근법의 초기 한계를 극복하고 15,301장의 이미지(실제 및 합성)를 사용하여 훈련함으로써 유망한 결과를 도출했습니다.

- **Technical Details**: 연구에서는 82개의 클래스에 대해 CLIP 모델의 미세 조정(fine-tuning) 과정을 설명하며, 이미지 설명 구문, 모델 및 하이퍼파라미터 조정을 포함합니다. 최종 훈련된 CLIP 모델은 3,826장의 이미지에 대해 테스트하였으며, 85% 이상의 정확도로 이전 연구의 최첨단 성과를 약 6% 초과했습니다. 특히, YOLOv8 기반 모델을 미세 조정하는 데 필요한 시간보다 3.5배 짧습니다.

- **Performance Highlights**: 작은 데이터 세트의 다양한 자세(각 6개 자세, 1,301 및 401장의 훈련 이미지 포함)를 이용한 경우, 미세 조정된 모델은 각각 98.8%와 99.1%의 정확도를 달성했습니다. 또한, 20장만으로도 6개 클래스 데이터 세트에서 90%의 정확도를 생성할 수 있음을 보여주었습니다. 이 연구는 CLIP 모델이 요가 자세 분류 및 일반적인 인간 자세 분류에 효과적으로 활용될 수 있음을 입증합니다.



### TimeLogic: A Temporal Logic Benchmark for Video QA (https://arxiv.org/abs/2501.07214)
- **What's New**: 이번 연구는 비디오 내 사건의 시간적 관계를 이해하는 Temporal logical understanding을 평가하기 위해 TimeLogic QA (TLQA) 프레임워크를 도입합니다. TLQA는 기존 비디오 데이터셋의 시간 주석과 논리 이론에서 파생된 시간 연산자를 활용하여 QA 쌍을 자동으로 생성합니다. 이는 복잡한 시간 논리를 요구하는 비디오 질문 응답(VideoQA) 모델의 성능을 정량적으로 평가하는 데 필수적인 도구가 될 것입니다.

- **Technical Details**: TLQA 프레임워크는 16개 시간 논리 범주에 대해 질문 템플릿을 지정하여 QA 쌍을 생성합니다. 이 프레임워크는 STAR, Breakfast, AGQA, CrossTask와 같은 4개의 데이터셋을 활용하며, 각각 2천 개와 1만 개의 QA 쌍을 포함한 두 가지 버전(TLQA-S 및 TLQA-L)의 비디오 QA 데이터셋을 생성합니다. TLQA는 기존의 비디오 데이터셋을 시간 주석과 함께 효율적으로 활용하여 자동으로 시간 논리 QA 쌍을 만들 수 있는 일반적이고 확장 가능한 프레임워크입니다.

- **Performance Highlights**: 연구 결과, 현재 비디오 QA 모델들이 다중 선택 질문에서는 좋은 성능을 보이나, 불리언 질문에서는 상당한 어려움을 겪고 있음을 발견했습니다. 이는 모델들의 시간적 이해 능력 개선이 필요함을 강조하며, TLQA 벤치마크가 이러한 평가의 기준을 제시합니다. TLQA는 복잡한 시간 논리 능력을 평가할 수 있는 포괄적인 테스트베드를 제공하여, 모델의 시간적 추론 성능을 전반적으로 향상시키는 데 기여할 것입니다.



### FaceOracle: Chat with a Face Image Orac (https://arxiv.org/abs/2501.07202)
- **What's New**: 이번 논문에서는 ID 및 여행 문서 발급 시 필수적인 고품질 얼굴 이미지의 중요성을 강조합니다. 특히 얼굴 이미지의 품질 기준이 복잡하고 세분화되어 있는 국제 표준에 따라, 제출된 얼굴 이미지의 비준수 또는 결함을 식별하고 이해하는 것이 매우 중요합니다. 이를 위해 FaceOracle이라는 LLM 기반의 AI 비서를 소개하며, 이는 자연스러운 대화 방식으로 사용자가 얼굴 이미지 분석을 도와줍니다.

- **Technical Details**: FaceOracle은 표준 준수 알고리즘을 사용하여 얼굴 이미지 품질 평가(FIQA) 알고리즘의 결과를 해석하고 다양한 얼굴 이미지 품질 개념에 대한 설명을 제공합니다. 이러한 기능은 LLM(대규모 언어 모델)의 힘을 활용하여 사용자가 쉽게 접근할 수 있도록 설계되었습니다. 발급 기관의 전문가들이 FaceOracle을 통합하여 보다 효율적으로 그들의 결정을 분석, 이해하고 소통할 수 있도록 하는 증명 개념(proof-of-concept)을 구현하였습니다.

- **Performance Highlights**: 전문가들이 FaceOracle을 통해 얼굴 이미지 분석의 효율성을 극대화할 수 있으며, 결과적으로 생산성이 향상됩니다. 이 시스템을 도입함으로써 사용자들은 품질 평가 결과를 보다 명확하게 이해하고, 행정 절차에서의 의사소통을 개선할 수 있습니다. 나아가 이러한 접근법은 얼굴 인식 시스템과 인간 검사자의 신뢰도를 높이는 데에 기여할 것입니다.



### VAGeo: View-specific Attention for Cross-View Object Geo-Localization (https://arxiv.org/abs/2501.07194)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 논문은 기존의 Cross-view object geo-localization (CVOGL) 방법에서 발생하는 시점별 불일치를 해결하기 위한 새로운 View-specific Attention Geo-localization (VAGeo) 기법을 제안합니다. VAGeo는 뷰 특화 위치 인코딩(view-specific positional encoding, VSPE) 모듈과 채널-공간 하이브리드 어텐션(channel-spatial hybrid attention, CSHA) 모듈을 포함해, 각각 객체 인식과 특징 학습에 중요한 역할을 수행합니다. 이러한 멀티 모듈 접근은 다양한 뷰포인트의 특성을 고려해, 더 정확한 객체 위치 식별이 가능하게 합니다.

- **Technical Details**: CVOGL 작업은 두 단계로 이루어집니다. 첫 번째 단계에서 모델은 쿼리 이미지에서 타겟 객체를 식별하고, 두 번째 단계에서는 위성 이미지 내에서 해당 객체를 지역화합니다. 이를 위해 쿼리 이미지와 참조 이미지에서 별도로 특징을 추출하는 두 개의 분기 네트워크를 사용하며, VSPE를 통해 모델의 주의를 타겟 객체에 집중시키고 CSHA 모듈을 통해 구별되는 특징을 강조합니다.

- **Performance Highlights**: VAGeo는 기존의 방법보다 현저한 성능 향상을 보여, CVOGL 데이터셋에서 ground-view의 acc@0.25/acc@0.5 지표가 45.43%/42.24%에서 48.21%/45.22%로, drone-view는 61.97%/57.66%에서 66.19%/61.87%로 개선되었습니다. 이는 VAGeo가 객체 인식 및 위치 확인 과정에서 엄청난 효과를 발휘함을 나타냅니다.



### Uncertainty Guarantees on Automated Precision Weeding using Conformal Prediction (https://arxiv.org/abs/2501.07185)
- **What's New**: 본 논문에서는 정밀 농업 분야에서의 트러스트 구축을 위한 새로운 접근 방식으로, 블랙 박스 모델에 대한 예측 보증을 제공하는 기능을 소개합니다. 기계 학습 커뮤니티에서 확립된 방법론인 "Conformal Prediction" (컨포멀 예측)이 정밀 제초 작업에 적합하게 적용된다는 점이 핵심입니다. 이 방법론은 사용자가 예측의 품질에 대한 신뢰를 갖도록 해주며, 농업 시스템의 채택을 촉진할 수 있을 것으로 기대됩니다.

- **Technical Details**: 컨포멀 예측은 정확한 예측의 보장을 위해 강력한 불확실성을 제어하는 프레임워크입니다. 이 방법은 포인트 예측(orginal predictions) 대신 예측 집합(predictive sets) 또는 간격 예측(interval predictions)을 생성하여 진정한 값의 포함을 보장합니다. 본 연구에서는 유럽에서 수집한 잡초 및 농작물 이미지 데이터베이스를 활용하여, 컨포멀 예측 기반의 정밀 분사 파이프라인을 개발하였습니다.

- **Performance Highlights**: 실험 결과, 개발한 파이프라인은 최소 90%의 잡초를 정확하게 감지할 수 있는 능력을 보여주었습니다. 두 가지 현실 세계 시나리오(데이터의 분포 조건에 따른 실험)에서 평가되었으며, 그 결과는 농업 기술 분야에서 신뢰할 수 있는 모델의 개발 가능성을 시사합니다. 이러한 결과는 정밀 농업 시스템의 효율성을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### Radial Distortion in Face Images: Detection and Impac (https://arxiv.org/abs/2501.07179)
- **What's New**: 이 논문은 얼굴 인식 시스템(FRS)의 성능을 저하시킬 수 있는 방사 왜곡(radial distortion), 일반적으로 '어안 효과(fish-eye effect)'라고 알려진 문제에 주목합니다. 특히 스마트폰을 통한 비감독 자기 등록(self-enrolment) 시에 방사 왜곡을 탐지할 수 있는 새로운 모델을 제안합니다. 이 모델은 입력된 얼굴 이미지에서 방사 왜곡을 감지하고 이를 플래그로 표시하는 기능을 갖추고 있습니다. 또한 이 모델을 기반으로 한 얼굴 이미지 품질 평가(FIQA) 알고리즘도 개발되었습니다.

- **Technical Details**: 제안된 접근법은 방사 왜곡을 탐지하여 얼굴 이미지의 품질을 평가하는 알고리즘으로, ISO/IEC 29794-5 표준과 관련된 기준을 충족합니다. 이 연구는 다양한 품질 요소를 고려하여 방사 왜곡이 얼굴 인식 시스템에 미치는 영향을 분석합니다. 또한, 두 가지 카메라 모델인 Division Model과 Kannala-Brandt Model을 사용하여 왜곡을 모델링하고 캡처 관련 품질 구성요소로서 방사 왜곡의 특성을 설명합니다. 이러한 기술적 접근은 앞으로의 연구에서도 유용하게 활용될 수 있습니다.

- **Performance Highlights**: 제안된 모델에 대한 평가 결과는 높은 탐지 정확도를 보였습니다. 이 연구는 방사 왜곡이 얼굴 인식 시스템의 성능에 미치는 영향을 보다 잘 이해할 수 있는 귀중한 통찰을 제공하며, 실제 운영 시나리오에서 방사 왜곡 여부를 점검할 최적의 시기를 파악하는 데 도움을 줍니다. 더불어 연구진은 훈련과 평가를 위한 합성 데이터셋도 제공하여 관련 연구의 진전을 도모할 예정입니다.



### BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literatur (https://arxiv.org/abs/2501.07171)
- **What's New**: BIOMEDICA는 PubMed Central Open Access 데이터를 활용하여 생물 의학 분야의 비전-언어 모델(vision-language models, VLMs) 개발을 위한 포괄적이고 접근 가능한 데이터셋을 제공합니다. 이 프레임워크는 6백만篇의 논문에서 2400만 개의 독특한 이미지-텍스트 쌍을 생성하여, 시각적 및 텍스트 정보를 통합하여 환자 치료 지원을 혁신할 수 있는 가능성을 열었습니다. 또한, BMCA-CLIP 방식을 통해 대량의 데이터를 로컬에 다운로드하지 않고도 모델을 지속적으로 사전 훈련할 수 있게 되었습니다.

- **Technical Details**: BIOMEDICA는 전체 PubMed Central Open Access 레포지토리를 표준화된 밀집 아카이브로 효율적으로 추출하고 직렬화하는 ETL(Extraction, Transformation and Loading) 파이프라인을 포함하여, 전문가 큐레이션을 통해 고도로 주석이 달린 데이터셋을 제공합니다. 이 데이터셋은 각 데이터 포인트에 대해 27개의 고유 메타데이터 필드가 주어지며, Parquet 형식과 WebDataset 형식으로 제공되어 빠른 질의와 필터링을 가능하게 하여 현대적인 훈련 전략을 탐색할 수 있도록 합니다.

- **Performance Highlights**: 모델은 40가지 생물 의학 작업에서 평균 6.56%의 성능 개선을 기록하며, 특히 피부과 및 안과 분야에서 29.8% 및 17.5%라는 높은 성능 향상을 보였습니다. BIOMEDICA 데이터셋을 통해 기존의 최첨단 모델보다 10배 적은 계산 자원으로 더 나은 결과를 달성하는 등, 공유 및 협업을 위한 코드를 공개하고 있습니다.



### Adaptive Noise-Tolerant Network for Image Segmentation (https://arxiv.org/abs/2501.07163)
- **What's New**: 이 논문은 전통적인 이미지 분할 알고리즘에서 나오는 불완전한 세분화 결과를 통합하여 더 나은 분할 결과를 달성하기 위한 새로운 Adaptive Noise-Tolerant Network (ANTN) 모델을 제안합니다. 기존의 Clean segmentation labels에 의존하는 대신, 노이즈가 있는 세분화 데이터를 활용하여 모델을 학습시키는 접근 방식을 채택합니다. 특히, histopathological 이미지를 다룸에 있어 수동 세분화 레이블을 받기 어려운 점을 해결하고자 합니다.

- **Technical Details**: ANTN 모델은 두 가지 독창적인 측면을 가지고 있습니다: (1) 여러 개의 노이즈가 있는 레이블을 자연스럽게 통합할 수 있으며, (2) 노이즈 세분화 모델링이 적응적이라는 것입니다. 이는 이미지의 외형에 따라 레이블 전환 확률이 달라지도록 설계되어, 다양한 오프 더 셸프(segmentation algorithms) 알고리즘의 결과를 효과적으로 통합할 수 있습니다. EM (Expectation-Maximization) 기반의 모델 추론 알고리즘을 개발하여 합성 데이터와 실제의 histo-images에서 구현하였습니다.

- **Performance Highlights**: 새로운 ANTN 모델의 성능은 기존의 오프 더 셸프 및 다른 깊은 학습 기반 이미지 분할 알고리즘과 비교하면서 뛰어난 효과성을 입증하였습니다. 실험 결과, ANTN은 노이즈가 있는 레이블을 효과적으로 활용하여 더욱 정교한 이미지 분할 결과를 생성하는 것을 보여줍니다. 이 모델은 다수의 노이즈가 있는 레이블이 존재하는 상황에서 강력한 성능을 발휘합니다.



### Eye Sclera for Fair Face Image Quality Assessmen (https://arxiv.org/abs/2501.07158)
- **What's New**: 이 논문은 안구의 흰자부위(sclera)를 얼굴 이미지 품질 평가(FIQA)에서 중요한 품질 평가 영역으로 제안합니다. 기존 FIQA 요소들이 인종적 변동성에 민감한 반면, sclera는 인구통계적 요인에 영향을 받지 않아 공정한 FIQA 방법으로 기능할 수 있습니다. 이 연구는 sclera를 활용한 FIQA 접근법이 동적 범위, 노출 과다 및 부족을 평가하는 데 효과적이라는 것을 확인합니다.

- **Technical Details**: 이 논문에서는 III ISO/IEC 29794-5 표준을 기반으로 한 세 가지 피부 톤 관련 FIQA 조치를 분석하고, sclera 영역을 대안적인 평가 영역으로 제시합니다. sclera 영역은 피부 톤에 비감각적이므로 다양한 인구 통계적 그룹의 이미지에서도 일관된 결과를 나타냅니다. EDC 분석을 통해 sclera가 얼굴 이미지 품질을 측정하는 데 적합한 대안임을 입증합니다.

- **Performance Highlights**: 연구 결과에 따르면 sclera를 활용한 FIQA 방법은 기존의 얼굴 이미지 품질 평가 방법과 유사한 성능을 보이며, 피부 톤에 독립적인 평가를 가능하게 합니다. 공정한 FIQA의 필요성이 증가하는 가운데, sclera가 이 과제에 적합한 대안으로 등장함으로써 얼굴 인식 시스템의 신뢰성을 높일 수 있음을 보여줍니다.



### Robust Single Object Tracking in LiDAR Point Clouds under Adverse Weather Conditions (https://arxiv.org/abs/2501.07133)
Comments:
          14 pages

- **What's New**: 이번 연구에서는 LiDAR 기반의 3D 단일 객체 추적(3DSOT)에서 악천후에 대한 견고성 평가를 위한 새로운 벤치마크를 제안합니다. KITTI-A와 nuScenes-A라는 두 가지 합성 데이터셋과 CADC-SOT라는 실제 데이터셋을 포함하여 비, 안개, 눈의 세 가지 날씨 유형을 아우릅니다. 이러한 데이터를 통해 현재의 추적 메소드들이 성능 저하를 겪는 원인을 탐구하며, 특히 악천후에서 모델의 견고성을 향상시키기 위해 새로운 방법을 제시합니다.

- **Technical Details**: 제안된 방법인 DRCT(이중 분기 추적 프레임워크)는 도메인 랜덤화(domain randomization) 및 대조 학습(contrastive learning)에 기반합니다. 이 프레임워크는 청정 데이터와 랜덤화된 포인트 클라우드 데이터를 입력으로 받으며, LGCM(local geometric contrast module)을 통해 메인 브랜치의 인식을 향상시킵니다. 또한 세 가지 요소(타겟 거리, 템플릿 형태 손상, 타겟 형태 손상)를 고려하여 날씨의 영향을 분석합니다.

- **Performance Highlights**: DRCT는 챌린징한 벤치마크에서 탁월한 성능을 보였으며, 청정 날씨에서의 성능을 개선함과 동시에 악천후에서의 견고한 추적을 달성했습니다. 기상 조건이 다른 다양한 데이터셋에서의 비교를 통해, 현재의 추적 방법들이 악천후 시 성능이 크게 저하된다는 것을 발견했습니다. 이러한 결과는 향후 연구 방향에 중요한 기초 자료를 제공합니다.



### Duplex: Dual Prototype Learning for Compositional Zero-Shot Learning (https://arxiv.org/abs/2501.07114)
- **What's New**: 이번 연구는 Compositional Zero-Shot Learning (CZSL)을 위한 새로운 방법인 Duplex를 제안합니다. Duplex는 이중 프로토타입 학습 방법을 통해 시맨틱(semantic)과 비주얼(visual) 프로토타입을 통합하여 이미지 내 상태와 객체의 독립적인 특성을 해체합니다. 이 연구는 Graph Neural Network (GNN)를 사용하여 복잡한 상태와 객체의 상호작용을 캡처하며, 사전 훈련된 Vision-Language Models (VLMs)의 강력한 시각-언어 정렬을 활용합니다.

- **Technical Details**: Duplex 모델은 다중 경로 아키텍처와 프롬프트 엔지니어링을 결합하여 이미지 및 텍스트 표현을 정렬합니다. 모델은 ResNet-18 백본을 사용하여 사전 훈련된 CLIP 기반 방법들과 비교하며, 하이퍼파라미터 최적화를 통해 성능을 극대화합니다. 실험은 MIT-States, UT-Zappos, CGQA의 세 가지 실세계 CZSL 벤치마크에서 진행되었으며, 다양한 전통적 방법 및 최신 CLIP 기반 접근 방식을 포함했습니다.

- **Performance Highlights**: 결과적으로 Duplex는 폐쇄형(closed-world) 및 개방형(open-world) 환경 모두에서 기존의 최첨단 방법보다 우수한 성능을 발휘합니다. 특히, 각 데이터셋에서 비주얼 프로토타입과 시맨틱 프로토타입의 적절한 조합을 통해 성능 향상이 이루어졌습니다. 실험에서는 특히 UT-Zappos와 MIT-States에서 최적의 조합 비율이 각각 0.5와 0.3에서 달성되었음을 확인할 수 있었습니다.



### Matching Free Depth Recovery from Structured Ligh (https://arxiv.org/abs/2501.07113)
Comments:
          10 pages, 8 figures

- **What's New**: 이 논문에서는 구조광 시스템으로 촬영한 이미지에서 깊이를 추정하는 새로운 접근 방식을 제안합니다. 기존의 많은 방법들이 이미지 매칭 프로세스를 활용한 반면, 본 연구에서는 밀도 복셀 그리드(density voxel grid)를 사용하여 장면 지오메트리를 표현합니다. 이를 통해 자기지도형 미분 가능 볼륨 렌더링(self-supervised differentiable volume rendering)을 통해 학습하는 방식으로 빠른 수렴과 높은 품질의 출력을 가능하게 합니다.

- **Technical Details**: 우리의 방법은 구조광 시스템에서 투사된 패턴으로부터 유도된 색 필드를 활용하여 렌더링 과정 중 지오메트리 필드를 최적화합니다. 노멀라이즈드 디바이스 코디네이트(normalized device coordinates, NDC), 왜곡 손실(distortion loss), 그리고 표면 기반 색 손실(surface-based color loss)을 도입하여 기하학적 신뢰성을 향상시킵니다. 복셀 그리드를 이용한 지오메트리 표현을 통해 효율적인 볼륨 렌더링과 속도 측면에서도 큰 이점을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존의 매칭 기반 기법들보다 기하학적 정확도에서 상당한 향상을 보여줍니다. 합성 장면의 경우 평균 추정 깊이 오류가 약 60% 감소하였고, 실제로 캡쳐된 장면에서는 약 30%의 오류 감소를 달성하였습니다. 또한, 우리의 접근 방식은 유사한 방식의 기존 방법에 비해 훈련 속도가 약 3배 빠르다는 장점을 보여주었습니다.



### Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation (https://arxiv.org/abs/2501.07110)
Comments:
          This paper has been accepted by ACM Transactions on Information Systems

- **What's New**: 본 논문에서는 동적 멀티모달 융합 기술을 통해 마이크로 비디오 추천의 표현 학습을 향상시키기 위해 Meta Multimodal Fusion (MetaMMF)이라는 새로운 메타 학습 기반의 멀티모달 융합 프레임워크를 제안합니다. 기존 방법의 정적 융합 한계를 극복하고, 각 마이크로 비디오에 대해 독립적으로 최적화된 융합 함수를 학습할 수 있는 방법론을 개발했습니다. 이를 통해 다양한 마이크로 비디오의 멀티모달 정보 간의 관계를 더 잘 모델링할 수 있습니다.

- **Technical Details**: MetaMMF는 주어진 입력 마이크로 비디오의 멀티모달 특성을 기반으로 메타 정보를 추출하고, 이를 통해 동적으로 파라미터화된 신경망을 융합 함수로 활용합니다. 주로 다층 퍼셉트론 (MLP)을 사용하여 멀티모달 특성을 고차원 추상화 형태의 메타 정보로 변환합니다. 또한, 모델의 복잡성을 줄이기 위해 Canonical Polyadic (CP) 분해를 채택하여 학습 효율성을 높였습니다.

- **Performance Highlights**: 세 가지 공공 데이터 세트를 기반으로 한 실험에서, MetaMMF는 기존의 여러 최첨단 멀티모달 추천 모델 (예: MMGCN, LATTICE, InvRL)보다 현저한 성능 향상을 보였습니다. 이는 MetaMMF가 목표한 동적 멀티모달 융합 방식을 통해 이루어진 결과이며, 다양한 마이크로 비디오가 각각 독특한 방식으로 멀티모달 정보를 통합할 수 있음을 보여줍니다.



### The Quest for Visual Understanding: A Journey Through the Evolution of Visual Question Answering (https://arxiv.org/abs/2501.07109)
- **What's New**: 이번 조사 논문에서는 Visual Question Answering (VQA)의 발전 과정을 다루며, 주목할 만한 혁신으로 attention mechanisms, compositional reasoning 및 vision-language pre-training 방법의 도입을 강조합니다. 또한 VQA가 의료 분야와 같은 전문 분야에서 어떻게 적용되고 있는지를 보여주며, 데이터셋 편향, 모델 해석성 및 일반 상식 추론과 같은 도전과제에 대해서도 논의합니다.

- **Technical Details**: VQA 시스템은 이미지 이해(image understanding), 자연어 이해(natural language comprehension), 다단계 추론(multi-step reasoning)과 같은 다양한 기능의 통합을 필요로 합니다. CNN(Convolutional Neural Networks)과 Transformer 기반 아키텍처의 발전은 VQA의 성능 향상에 중요한 기여를 했습니다. 초기 VQA 모델이 CNN에 의존했던 반면, 최근에는 transformer 기반 모델로의 전환이 이루어졌습니다.

- **Performance Highlights**: VQA는 의료, 교육, 자율 시스템 등의 다양한 실제 응용 분야에서 큰 잠재력을 가지고 있습니다. 초기 작업인 VisualQA 데이터셋의 도입 이후, 연구 커뮤니티는 멀티모달 AI의 가능성을 탐구하고 있으며, 이 데이터셋은 VQA의 발전을 위한 첫 번째 대규모 벤치마크를 제공했습니다. 그러나 데이터셋 편향과 모델 해석성 문제는 여전히 해결해야 할 중요한 도전 과제로 남아 있습니다.



### RMAvatar: Photorealistic Human Avatar Reconstruction from Monocular Video Based on Rectified Mesh-embedded Gaussians (https://arxiv.org/abs/2501.07104)
Comments:
          CVM2025

- **What's New**: 새로운 RMAvatar 모델은 모노큘러 비디오에서 입체적이고 생동감 넘치는 인간 아바타를 생성하기 위해 Gaussian splatting을 이용하여 메시(mesh)에 통합된 독창적인 표현 방식으로 설계되었습니다. 이 모델은 운동과 형태를 명확하게 표현하기 위해 메쉬 기하학을 활용하고, Gaussian Splatting 기술로 나타내는 외관을 암시적으로 렌더링합니다. 기존의 방법들이 가지는 한계를 극복하여 아바타의 사실성과 표현력을 크게 향상시키는 새로운 접근법을 제시합니다.

- **Technical Details**: RMAvatar는 Gaussian 초기화 모듈과 Gaussian 정정 모듈의 두 가지 주요 모듈로 구성됩니다. Gaussian은 삼각형의 면에 내장되어 메쉬를 통해 운동을 제어하여 아바타의 저주파 모션과 표면 변형을 보장합니다. Gaussian 정정 모듈은 모션 제어를 위해 MLP를 활용하여 포즈에 따라 비닐형 변형을 학습하여 더욱 사실적인 아바타 구현을 돕습니다.

- **Performance Highlights**: 저자들은 공개 데이터 세트에서 RMAvatar 모델의 광범위한 실험을 수행하여 렌더링 품질과 정량적 평가 모두에서 최첨단 성능을 보여주었습니다. 이 모델은 비디오에서 고해상도의 인간 아바타를 생성하는 데 있어 전통적인 결합 방식보다 우수한 결과를 기록하고 있습니다. 또한, 아바타의 동적 특징을 정확하게 캡처함으로써 활용 가능성을 높이고 있습니다.



### Dual Scale-aware Adaptive Masked Knowledge Distillation for Object Detection (https://arxiv.org/abs/2501.07101)
- **What's New**: 이번 연구는 object detection(물체 탐지) 성능 향상을 위한 세밀한 적응형 feature masking distillation 프레임워크인 DSAMD를 제안합니다. 기존의 전역 마스킹 방식과 달리, 다양한 스케일에서 feature distillation을 수행하며 지역 정보가 더 잘 인코딩되도록 합니다. 이를 통해 accuracy(정확도)를 높이고, teacher와 student 네트워크 간의 지식 전달이 개선됩니다.

- **Technical Details**: DSAMD 방법은 Scale-aware Feature Maps(SA 모듈)와 Adaptive Masked Weights(AM 모듈)을 포함하여 모델이 지역 특성과 경계 정보를 더 잘 학습할 수 있도록 합니다. 특징적인 로그 복잡도를 파악하기 위해 feature-level masks를 사용하여 logit 위치를 안내합니다. 여기서, FPN layer의 feature와 logit은 H, W, C, K 형태로 각각 높이, 너비, 채널, 클래스 수를 나타냅니다.

- **Performance Highlights**: 대규모 실험 결과, DSAMD 방법은 RetinaNet, RepPoints 그리고 Cascade Mask RCNN을 teacher 네트워크로 사용했을 때 각각 41.5%, 42.9%, 42.6%의 mAP(scores) 점수를 달성하며 최신 방법들을 초월하는 성능을 보입니다. 이 결과는 DSAMD 프레임워크가 object detection task의 knowledge distillation에서 유망한 성과를 나타냄을 보여줍니다.



### Collaborative Learning for 3D Hand-Object Reconstruction and Compositional Action Recognition from Egocentric RGB Videos Using Superquadrics (https://arxiv.org/abs/2501.07100)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문에서는 egocentric 3D 손-객체 상호작용 데이터셋을 활용하여 손-객체 포즈 추정(hand-object pose estimation)과 행동 인식(action recognition)을 위한 통합 모델을 개발하고자 합니다. 기존 방법들이 3D bounding box를 사용하여 객체의 모양과 움직임을 표현하는 데 한계가 있었던 점을 해결하기 위해, superquadrics라는 대체 3D 객체 표현법을 도입하여 효과성을 입증합니다. 이 외에도, 우리는 학습 데이터와 테스트 데이터의 동사와 명사 조합이 겹치지 않는 더 도전적인 작업을 통해 행동의 구성 가능성을 연구합니다.

- **Technical Details**: 핵심적으로, 우리는 손-객체 포즈 추정과 상호작용 인식 간의 관계를 이해하기 위해 새로운 협업 학습 프레임워크를 제안합니다. 이 프레임워크는 겹치지 않는 훈련 및 테스트 분할을 통해 행동과 객체의 구성 가능성을 학습하는 데 도움을 줍니다. 또한, 기존의 3D 주석된 egocentric 손-객체 데이터셋을 확장하여 새로운 구성 분할을 도입하고, superquadrics를 중간 3D 객체 표현으로 사용함으로써, 이 방법이 동작 인식 및 손-객체 포즈 추정에 어떻게 유리한지를 입증합니다.

- **Performance Highlights**: 우리는 H2O와 FPHA 데이터셋을 사용하여 두 가지 공식 및 구성 설정에서 상태-of-the-art 성능을 달성했습니다. 우리의 메소드는 다양한 기존 방법들보다 손-객체 인식을 더욱 효과적으로 진행할 수 있도록 개선되었습니다. 실험 결과는 superquadrics와 협업 학습 프레임워크가 특히 구성된 행동 인식(compositional action recognition) 작업에서 유의미한 성능 향상을 가져온다는 것을 보여줍니다.



### Video Quality Assessment for Online Processing: From Spatial to Temporal Sampling (https://arxiv.org/abs/2501.07087)
- **What's New**: 이 논문은 비디오 품질 평가(VQA) 모델에서 스페이셜(Spatio) 및 템포럴(Temporal) 샘플링의 효과와 효율 간의 균형을 깊이 탐구합니다. 연구자들은 비디오 정보를 대폭 축소하여도 여전히 수용 가능한 성능을 유지할 수 있는 방법을 제안하며, 이러한 비디오 정보를 작은 크기로 연관짓습니다. 또한, 온라인 VQA 모델인 MGQA를 설계하여 복잡성을 줄이면서도 성능을 보장하고 있습니다.

- **Technical Details**: 제안된 연구는 비디오의 공간적 및 시간적 차원에서 정보 손실이 비디오 품질 예측에 미치는 영향에 대해 조사합니다. 특히, 기본 Gated Recurrent Units (GRU)와 같은 간단한 모델을 통해 공간 및 시간 특성을 추출하는 방법론을 채택하고 있습니다. 이를 통해, VQA 모델의 평균 계산 비용이 기존 모델에 비해 99.83% 감소하는 성과를 보여줍니다.

- **Performance Highlights**: 종합적인 실험을 통해 6개의 공개 비디오 품질 데이터베이스에서 제안된 샘플링 방법의 성능이 검증되었습니다. 연구 결과, 대부분의 비디오 정보를 제거하더라도 VQA 모델은 수용 가능한 성능을 유지할 수 있으며, 이로 인해 기존의 VQA 모델을 대체할 가능성을 보여줍니다. 특히, 온라인 모델은 처리하는 데이터 양을 크게 줄이는 방안을 제시하고 있습니다.



### Representation Learning of Point Cloud Upsampling in Global and Local Inputs (https://arxiv.org/abs/2501.07076)
- **What's New**: 본 연구에서는 3D 재구성을 위한 포인트 클라우드 업샘플링(point cloud upsampling)의 글로벌(global) 및 로컬(local) 수준에서의 영향을 조사합니다. 두 개의 인코더(encoder)를 이용하여 같은 포인트 클라우드 모델 객체의 글로벌 및 로컬 정보를 입력받고, 이를 융합(fusion)하여 업샘플링 디코더(decoder)에 보냅니다. 이 프레임워크는 최첨단(point cloud upsampling neural network) 모델에 적용할 수 있으며, 기존 방법들에 비해 향상된 효과를 보입니다.

- **Technical Details**: 연구에서는 포인트 클라우드의 희소성(sparsity) 및 노이즈(noise) 문제를 해결하기 위해 글로벌 및 로컬 입력의 사전 지식을 활용합니다. 또한, 심층 학습을 사용하는 오토인코더 기반 모델에서 실험을 진행하여 글로벌 및 로컬 입력의 해석 가능성을 증명합니다. 이 과정에서 Saliency Map을 이용해 두 입력의 차이와 병렬 학습(parallel training)의 유효성을 반영합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 기존 SOTA(state-of-the-art) 연구들에 비해 업샘플링 효과를 더욱 개선할 수 있음을 입증하였습니다. 아울러, 그래프 기반(graph-based) 방법 및 GAN(generative adversarial networks) 등을 활용하여 포인트 클라우드 작업에서의 딥러닝 적용을 용이하게 하였습니다. 이를 통해, 포인트 클라우드 데이터의 질을 높이고, 3D 재구성, 객체 탐지 및 분류, 로봇 작동에서의 활용 가능성을 높이는 데 기여할 수 있습니다.



### Label Calibration in Source Free Domain Adaptation (https://arxiv.org/abs/2501.07072)
Comments:
          Accepted in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이번 연구에서는 Source-free domain adaptation (SFDA)에서의 pseudolabel 정제를 위한 새로운 방법론을 제안합니다. 이는 evidential deep learning을 활용하여 예측 오차의 불확실성을 도입하고 softmax 보정을 통해 이루어집니다. 기존의 self-supervised SFDA 기술에서는 source와 target 도메인 간의 불일치로 인해 발생하는 노이즈가 문제였지만, 제안된 방법은 이러한 문제를 효과적으로 해결할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 Dirichlet prior를 사용하여 target network의 출력을 불확실성으로 보완하고, softmax 보정을 통해 noisy pseudolabel을 처리하는 방안을 설명합니다. 이를 위해 기존의 cross-entropy 손실 대신 새로운 EDL(최소손실)의 손실 함수를 적용합니다. 제안된 방법은 정보 최대화 손실과 결합하여 더 나은 domain adaptation 성능을 발휘하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 DomainNet40, Office-Home, Office-31, MNIST, SVHN, USPS와 같은 벤치마크 데이터셋에서 기존 최신 기술보다 우수한 성능을 나타냈습니다. 이 연구는 SFDA 설정에서의 효율적인 pseudolabel 정제를 위한 새로운 접근 방식을 제시하며, 실질적인 문제 해결에 기여할 것으로 기대됩니다.



### Enhancing Image Generation Fidelity via Progressive Prompts (https://arxiv.org/abs/2501.07070)
Comments:
          Accepted by ICASSP 2025, Github: this https URL

- **What's New**: 이번 논문은 이미지 생성을 위해 적절한 regional prompt-following 생성을 위한 DiTPipe라는 새로운 시스템을 제안합니다. 이 시스템은 고수준(high-level) 및 저수준(low-level) 프롬프트를 생성하고, 교차 주의(cross-attention) 레이어의 영향을 연구하여 생성 과정을 개선합니다. 이를 통해 기존 DiT 기반 이미지 생성 시스템의 제어력을 향상시키는 데 기여합니다.

- **Technical Details**: Diffusion Transformers(DiT) 아키텍처는 이미지 생성을 위해 강력한 변환기(transformer) 모델링 능력을 적용한 구조입니다. DiT 블록 내의 cross-attention은 텍스트 조건과 이미지 정보를 노이즈 생성 과정에 참여시켜, 지역적인 제어를 가능하게 합니다. 특히, fomul의 Controllable Region-Attention 방법론을 통해 서로 다른 텍스트 프롬프트를 이미지의 특정 지역에 주입하여 더 정교한 제어가 가능합니다.

- **Performance Highlights**: 제안된 DiTPipe의 실험 결과, 정량적 및 정성적 지표에서 이전 연구들보다 우수한 성능을 확인하였습니다. 이 새로운 파이프라인은 더욱 세밀한 디테일을 생성할 수 있는 능력을 보여주며, 결과적으로 이미지 생성의 충실도(fidelity)를 극대화하는 데 기여합니다.



### Hierarchical Superpixel Segmentation via Structural Information Theory (https://arxiv.org/abs/2501.07069)
Comments:
          Accepted by SDM 2025

- **What's New**: 본 논문에서는 SIT-HSS라는 새로운 계층적 슈퍼픽셀 분할 방법을 제안합니다. 이 방법은 구조적 정보 이론(Structural Information Theory)에 기반하여 픽셀 이웃의 관계를 심도 있게 탐구합니다. 전통적인 그래프 기반 방법들과의 차별점은 비인접 픽셀의 영향을 고려함으로써 더 나은 분할 품질을 달성하는 것입니다.

- **Technical Details**: SIT-HSS는 1차원 구조 엔트로피(1D SE)를 극대화하여 그래프를 구성하고, 2차원 구조 엔트로피(2D SE)를 최소화하여 계층적 그래프 파티셔닝을 수행합니다. 이를 통해 픽셀 군집 간의 연결성을 유지하며, 각 레이어에서 순차적으로 픽셀 클러스터를 병합합니다. 이러한 방식은 복잡하고 밀집한 그래프 구조를 피하면서도 정보 손실을 최소화하는 데 기여합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋을 사용한 실험 결과, SIT-HSS는 최신 슈퍼픽셀 분할 알고리즘에 비해 뛰어난 성능을 보여주었습니다. 또한, 분할 정확도를 유지하면서 경쟁력 있는 계산 효율성을 확보하였습니다. 이러한 결과는 SIT-HSS가 다양한 실제 애플리케이션에서도 효과적임을 입증합니다.



### SFC-GAN: A Generative Adversarial Network for Brain Functional and Structural Connectome Translation (https://arxiv.org/abs/2501.07055)
Comments:
          5 pages, 2 figures

- **What's New**: 이 연구에서는 구조적 연결(connectivity)과 기능적 연결의 이원적(translational) 관계를 탐구하는 SFC-GAN(Structural-Functional Connectivity GAN)을 제안합니다. 기존의 연구는 주로 한쪽 방향의 예측에 집중하였으나, 본 연구에서는 CycleGAN 아키텍처를 활용하여 양방향 번역을 가능하게 합니다. 이를 통해 연결체(connectome)의 전체 및 지역 패턴을 포착하고, 네트워크의 대칭성을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: SFC-GAN은 두 개의 생성기(GSC 및 GFC)와 두 개의 판별기(DSC 및 DFC)로 구성되어 있으며, 이들은 connectome의 토폴로지(topology)를 보존하면서 각각의 FC와 SC 쌍 간의 변환을 수행합니다. 일반적으로, fMRI에서 도출된 통계적 관계를 FC로, dMRI에서 유도된 백질 경로를 SC로 사용하여 상호 변환하는 방향성을 가집니다. 이를 통해 상호 작용하는 구조 및 기능적 데이터의 관계를 모델링함으로써 동시에 존재하지 않는 연결체에 대한 유용한 정보를 제공합니다.

- **Performance Highlights**: SFC-GAN은 기존의 모델보다 SC와 FC 간의 변환에서 높은 유사도를 보이며, 기초 모델들보다 그래프 속성 평가에서 탁월한 성능을 입증하였습니다. 각 변환된 모달리티는 다운스트림 분류 작업에 효과적으로 활용될 수 있어, 보건 및 신경학적 장애 진단의 정확도를 크게 향상시킬 수 있습니다. 이를 통해 구조적 및 기능적 연결체 간의 복잡한 관계를 해석하는 데 중요한 도구가 될 수 있습니다.



### Protego: Detecting Adversarial Examples for Vision Transformers via Intrinsic Capabilities (https://arxiv.org/abs/2501.07044)
Comments:
          Accepted by IEEE MetaCom 2024

- **What's New**: 이 논문에서는 Vision Transformer(ViT) 모델의 취약성을 드러내기 위해 6개의 일반적인 adversarial attack 방법의 공격 능력을 조사합니다. 연구는 adversarial attack에 대한 기존 접근법의 한계를 극복하고, 새로운 탐지 프레임워크인 Protego를 제안하여 ViT 모델의 공격 탐지 효율성을 높이고자 합니다. Protego는 다양한 attack 전략에 대처할 수 있는 강력한 기능을 갖춘 탐지기를 목표로 합니다.

- **Technical Details**: Protego 프레임워크는 ViT의 구조적 정보를 활용하여 이미지를 패치로 나눈 후, Transformer encoder를 통해 패치 임베딩을 수행합니다. 이 과정에서 생성된 최종 특징 출력을 바탕으로 adversarial 예와 정상 예 간의 비일치성을 찾아내어 경량의 플러그인 탐지기를 훈련합니다. 이 탐지기는 입력의 피처 정보를 효과적으로 캡처할 수 있는 적절한 상호작용 메커니즘을 형성하여 공격 예제를 식별합니다.

- **Performance Highlights**: 실험 결과, Protego의 AUC 점수는 0.95를 초과하며, 기존의 탐지 방법들보다 우수한 성능을 보여줍니다. 이 연구는 Metaverse 보안 분야에서의 탐구를 진전시킬 수 있는 잠재력을 지니고 있습니다. ViT 모델의 adversarial example 탐지에 대한 연구 기여와 함께, Protego는 다양한 Transformer 구조에 보편적으로 적용 가능한 기능 추출 기법을 발전시켰습니다.



### Rethinking Knowledge in Distillation: An In-context Sample Retrieval Perspectiv (https://arxiv.org/abs/2501.07040)
- **What's New**: 본 논문에서는 기존의 지식 증류(Knowledge Distillation, KD) 방법론의 한계를 극복하기 위해, 각 샘플과 관련된 인컨텍스트 샘플 간의 관계를 포착하는 새로운 IC-KD 프레임워크를 제안합니다. 인컨텍스트 샘플(retrieval-based learning)로 수집된 샘플들은 학생 모델의 훈련을 정규화하는 데 중요한 역할을 합니다. 이론적 분석을 통해, 이러한 방법이 학생 모델의 성능을 향상시키는데 기여함을 보여줍니다.

- **Technical Details**: IC-KD 프레임워크는 Positive In-Context Distillation (PICD)와 Negative In-Context Distillation (NICD)라는 두 가지 핵심 컴포넌트를 포함합니다. PICD는 학생 모델의 샘플과 교사 모델의 동일 클래스를 가진 Aggregated In-Context Samples 간의 불일치를 감소시키는 것을 목표로 하고, NICD는 서로 다른 클래스의 샘플 간의 구분을 강화합니다. 이러한 방식은 학습 과정에서 정규화를 통해 모델 성능을 개선합니다.

- **Performance Highlights**: 다양한 KD 파라다임(오프라인, 온라인, 교사 없는 KD)에서 IC-KD 프레임워크의 우수성을 실험적으로 증명하였습니다. CIFAR-100 및 ImageNet 데이터셋에서 SOTA(State-Of-The-Art) 성능을 지속적으로 달성하여, 다양한 작업 및 데이터셋에서 적용 가능성을 입증합니다. 결과적으로 IC-KD는 여러 가지 KD 변형 및 과제에 대해 일반화된 효과를 보여줍니다.



### IoT-Based Real-Time Medical-Related Human Activity Recognition Using Skeletons and Multi-Stage Deep Learning for Healthcar (https://arxiv.org/abs/2501.07039)
- **What's New**: 이 연구는 다단계 딥러닝 기술을 활용하여 의료 관련 인간 활동(MRHA)을 인식하는 새로운 방법을 제안합니다. 기존의 HMR 시스템에서 발생하는 높은 계산 요구, 낮은 정확도 및 적응성 한계를 해결하기 위해 EfficientNet과 ConvLSTM을 결합한 새로운 모델을 개발하였습니다. IoT와 모바일 기술을 통합하여 실시간으로 환자의 건강을 모니터링할 수 있는 시니어 전용 시스템을 확립했습니다.

- **Technical Details**: 제안된 시스템은 7개의 Mobile Inverted Bottleneck Convolutions (MBConv) 블록을 사용하여 골격 프레임 시퀀스에서 최적화된 공간 특징을 추출하고, ConvLSTM을 통해 시공간 패턴을 캡처합니다. 모델은 NTU RGB+D 120 및 HMDB51 데이터셋에서 평가되며, 다양한 MRHA를 다룸으로써 동적이고 효과적인 인식 성능을 입증합니다. IoT 통합을 통해 Raspberry Pi와 GSM 모듈을 사용하여 실시간 SMS 알림을 제공하는 기능을 갖추고 있습니다.

- **Performance Highlights**: NTU RGB+D 120 데이터셋에서 교차 주제 평가에 94.85%의 정확도, 교차 뷰 평가에 96.45%의 정확도를 기록하였으며, HMDB51 데이터셋에서는 89.00%의 정확도를 달성했습니다. 이는 제안된 시스템이 공간 및 시간 데이터 모두를 효과적으로 처리할 수 있는 역량을 갖추었음을 보여줍니다. 이 연구는 환자의 안전과 건강 결과를 개선하며, 의료 비용 절감에 기여하는 혁신적인 솔루션을 제공합니다.



### UNetVL: Enhancing 3D Medical Image Segmentation with Chebyshev KAN Powered Vision-LSTM (https://arxiv.org/abs/2501.07017)
- **What's New**: 이번 연구에서는 UNETVL(U-Net Vision-LSTM)이라는 새로운 아키텍처를 제안합니다. 이 모델은 Vision-LSTM(ViL)을 활용하여 의료 영상 분할(task)에서 메모리 기능과 확장성을 향상시킵니다. 또한, 복잡하고 장기적인 의존성 패턴을 처리하기 위해 효율적인 Chebyshev Kolmogorov-Arnold Networks(KAN)를 통합하여 성능을 개선했습니다. 이로 인해 기존 모델 UNETR 대비 Dice score에서 7.3%와 15.6%의 유의미한 향상을 보여주었습니다.

- **Technical Details**: UNETVL은 UNETR 구조를 기반으로 한 인코더-디코더 네트워크로, 지역적 특징(local features)과 장거리 컨텍스트를 모두 효율적으로 캡처합니다. 이 모델은 3D oversampled instance를 처리하며, 입력 데이터를 비오버랩(non-overlapping) 패치로 나누고 이러한 패치 토큰(patch tokens)을 여러 개의 Vision-LSTM 블록 쌍을 통해 처리하여 다양한 스케일의 중간 특징 표현을 생성합니다. ViL 블록에서는 mLSTM 계층이 양방향으로 패치 토큰을 처리하여 복잡한 데이터 관계를 캡처합니다.

- **Performance Highlights**: UNETVL은 ACDC와 AMOS2022 벤치마크 데이터셋에서 기존의 최신 기법(state-of-the-art)과 비교하여 뛰어난 성능을 발휘했습니다. 특히, 정량적인 평가 지표인 평균 Dice score에서 ACDC에서 7.3% 그리고 AMOS에서 15.6% 개선을 보였습니다. 또한, 각 컴포넌트가 모델 성능에 미치는 영향을 분석하기 위해 광범위한 ablation 연구가 진행되었습니다.



### SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting (https://arxiv.org/abs/2501.07015)
- **What's New**: 이 논문에서는 3D 역설계를 위한 새로운 프레임워크인 SplatMap을 제안하고 있습니다. 이 프레임워크는 SLAM과 3D Gaussian Splatting(3DGS)을 결합하여 실시간 고충실도 밀집 복원을 가능하게 합니다. 슬램의 동적 깊이와 자세 업데이트를 활용하여 장면을 더욱 정확하게 보정할 수 있는 방법론입니다.

- **Technical Details**: SplatMap의 주요 구성 요소는 SLAM-Informed Adaptive Densification(SIAD)와 Geometry-Guided Optimization입니다. SIAD는 SLAM의 밀집 포인트 클라우드를 활용하여 가우시안 모델을 동적으로 업데이트하고 조밀하게 만듭니다. Geometry-Guided Optimization은 기하학적 제약과 광학적 일관성을 결합하여 3DGS 장면 표현의 외형과 기하를 함께 최적화합니다.

- **Performance Highlights**: 실험 결과 SplatMap은 Replica 및 TUM-RGBD 데이터셋에서 모노큘러 시스템 중 최고 성능을 기록했습니다. Replica 데이터셋에서 PSNR 36.864, SSIM 0.985, LPIPS 0.040을 달성하며 이전 SOTA 결과에 비해 각각 10.7%, 6.4%, 49.4% 향상된 결과를 보였습니다. 이러한 성과는 급변하는 3D 장면 표현 간의 갭을 메우고 실용적인 모노큘러 밀집 재구성을 가능하게 합니다.



### LEO: Boosting Mixture of Vision Encoders for Multimodal Large Language Models (https://arxiv.org/abs/2501.06986)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 비주얼 이해 능력을 향상시키기 위한 새로운 접근법으로 LEO라는 하이브리드 모델을 제안합니다. LEO는 두 개의 비전 인코더를 통합하여 데이터 처리 시 각 인코더의 전문성을 활용하는 타일 기반 후적응 융합 전략(tile-level post-adaptation fusion strategy)을採用합니다. 이를 통해 LEO는 기존 하이브리드 모델보다 더 효율적으로 다양한 비전 인코더를 융합하여 작업 수행에서 우수성을 발휘할 수 있습니다.

- **Technical Details**: LEO의 구성은 입력 이미지를 448x448 크기의 타일로 분할하고, 각 타일에 대해 두 개의 비전 인코더가 각각의 특징 표현을 제공하는 방식으로 진행됩니다. 이후 시각적 임베딩에 픽셀 언셔플링(pixel unshuffling)을 적용하여 각 인코더의 시각적 토큰 수를 줄입니다. 이러한 후적응 융합 전략을 통해 LEO는 텍스트 토큰과 비전 토큰을 결합하여 LLM으로 처리할 수 있는 형태로 제공합니다.

- **Performance Highlights**: LEO는 13개 비전-언어 벤치마크에서 전반적으로 최고의 성능을 보였습니다. 특히 LEO는 기존 오픈소스 MLLM 및 하이브리드 모델들과 비교하여 다수의 작업에서 우수한 성능을 나타내며, 자율 주행 분야에 적합하도록 모델 아키텍처를 변경하지 않고도 경쟁력 있는 성능을 입증했습니다. 또한 이 연구는 자율 주행 분야를 위한 하이브리드 MLLM의 첫 번째 탐색으로 주목받고 있습니다.



### Evaluating unsupervised contrastive learning framework for MRI sequences classification (https://arxiv.org/abs/2501.06938)
- **What's New**: 이 논문에서는 MRI 시퀀스의 자동 식별을 위한 비지도 대조 학습 프레임워크를 기반으로 한 시스템을 제안하고 있습니다. 연구는 ResNet-18 아키텍처를 사용하여 아홉 가지 일반적인 MRI 시퀀스를 9개 클래스 분류 문제로 분류합니다. 기존 연구의 한계를 극복하기 위해, 다양한 기관의 MRI 이미지 데이터셋을 활용하고 자동화된 방법론을 도입하여 교수의 수작업 개입을 줄이고자 합니다.

- **Technical Details**: 논문은 T1w, T2w, FLAIR, DWI 등 다양한 MRI 시퀀스를 대상으로 비지도 학습을 통해 각각의 시퀀스를 구분하는 알고리즘을 구성했습니다. 데이터셋은 내부에서 수집한 72,188개의 뇌 MRI 연구와 여러 공개 데이터셋을 포함하며, SimCLR과 SimSiam이라는 두 가지 비지도 대조 학습 아키텍처를 사용하여 성능을 비교하고 있습니다. 이러한 방식은 MRI 시퀀스의 다양한 변화를 인식하고 분류하는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 시스템은 공통으로 사용되는 아홉 가지 MRI 시퀀스에 대해 95% 이상의 분류 정확도를 달성했습니다. 이 연구는 SimCLR과 SimSiam 두 모델의 성능을 비교하고, 각각의 모델을 학습하기 위한 최적의 배치 크기를 찾는 데 집중하였습니다. 최종 결과는 MRI 진단 및 치료 계획을 위한 임상적 응용이 가능함을 시사합니다.



### CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications (https://arxiv.org/abs/2501.06927)
- **What's New**: 본 논문에서는 전 세계 다양한 장소에서 촬영한 고해상도 이미지를 이용한 대규모 세부 데이터셋을 제안합니다. 기존 데이터셋에 비해 크기와 세부 정보의 수준이 상당히 크기 때문에, 정밀한 3D 응용을 위해 독특하게 적합합니다. 특히, 드론으로 촬영한 공중 이미지가 사용되어 실제 공간의 레이아웃과 건축 구조를 더욱 정확하게 캡처할 수 있습니다.

- **Technical Details**: 이 데이터셋은 고해상도 이미지 41006개로 구성되어 있으며, 세밀한 점 구름(point cloud) 데이터를 제공하여 실내 및 실외 환경을 캡처합니다. Gaussian Splatting 및 Structure-from-Motion (SfM) 방법과 같은 다양한 기술과 호환되며, SLAM, Multi-View Stereo, Neural Radiance Fields (NeRF) 등과 함께 3D 재구성이 가능하도록 지원합니다. 이 데이터셋은 건축 재구성에서 가상 관광에 이르는 여러 3D 응용의 통합을 촉진할 수 있습니다.

- **Performance Highlights**: 이 데이터셋은 기존의 평가 및 훈련을 위한 기준점(benchmark)으로 기능하며, 로보틱스, 컴퓨터 비전 및 가상 현실 분야의 최신 방법론에 대한 강력한 훈련 파이프라인을 지원합니다. 또한, 이미지를 기반으로 한 이 데이터셋은 건축 및 구조적 요소를 더욱 잘 구분할 수 있도록 모델의 의미론적(segmentation) 분할과 세부 장면 이해를 향상시킵니다. 사용자들이 실제로 존재하는 것처럼 재구성된 환경을 탐색할 수 있는 Unreal Engine 기반의 가상 현실 시뮬레이션도 포함되어 있습니다.



### Benchmarking YOLOv8 for Optimal Crack Detection in Civil Infrastructur (https://arxiv.org/abs/2501.06922)
Comments:
          Accepted at 104th TRB Annual Meeting 2025

- **What's New**: 이번 연구에서는 전통적인 균열 감지 방법을 대체할 수 있는 최신 YOLOv8 프레임워크의 성능을 평가했습니다. YOLOv8은 다양한 모델 스케일(나노, 소형, 중형, 대형, 초대형)을 사용하여 높은 품질의 Roboflow 데이터셋에서 검토되었습니다. 이 연구는 실시간 균열 감지의 새로운 기준을 설정하는 데 기여하고 있습니다.

- **Technical Details**: 연구에서는 6가지 최첨단 최적화 기법(Stochastic Gradient Descent, Adaptive Moment Estimation, Adam with Decoupled Weight Decay, Root Mean Square Propagation, Rectified Adam, Nesterov-accelerated Adam)을 활용하여 YOLOv8의 하이퍼파라미터를 최적화했습니다. 특히, Stochastic Gradient Descent로 최적화된 YOLOv8 모델은 뛰어난 정확도 및 속도를 기록했습니다. 이러한 기술적인 접근은 인프라 모니터링에 AI 기법을 통합하는 기초를 마련합니다.

- **Performance Highlights**: YOLOv8 모델은 기존의 두 단계 타겟 감지 알고리즘에 비해 보다 신속하고 정확한 성능을 보였습니다. 실시간 균열 감지 분야에서 새로운 기준을 세움으로써, 노후 교량 네트워크의 보다 신뢰할 수 있는 유지 관리와 효율적인 교통 시스템 구축에 기여할 것입니다. 이 연구는 전 세계적으로 안전하고 효율적인 교통 시스템 확보를 위한 중요한 초석이 될 수 있습니다.



### Local Foreground Selection aware Attentive Feature Reconstruction for few-shot fine-grained plant species classification (https://arxiv.org/abs/2501.06909)
- **What's New**: 이번 연구에서는 식물 분류의 정확성을 높이기 위해 새로운 Local Foreground Selection (LFS) 주의 메커니즘을 도입했습니다. 이 메커니즘은 두 가지 유형의 주의 기법인 지역적 주의와 전경 선택 주의를 통해 기능합니다. 이를 통해 배경의 간섭을 줄이고, 주목해야 할 식물 객체에 비중을 두어 intra-class variation을 최소화하는 방법을 제시합니다.

- **Technical Details**: LFS 주의 메커니즘은 지역적 세부사항을 포착하여 inter-class variation을 향상시키고, 전경 선택 주의로 배경 간섭을 줄이며, 구별되는 피처 맵을 생성합니다. 기존의 feature reconstruction 접근 방식과 결합함으로써 LFS는 state-of-the-art 성능을 구현합니다. 연구는 세 가지 식물 종 데이터 세트를 사용하여 진행되었으며, 자연 배경을 가진 이미지의 분류 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과는 제안된 LFS 주의 메커니즘이 있는 접근 방식이 기존의 피처 재구성 방법보다 효과적임을 보여줍니다. 이 메커니즘을 사용함으로써 식물 분류에서 이루어지는 성능 향상이 관찰되었으며, 정확도 또한 상당히 개선되었습니다. 특히, 유사한 배경을 가지는 식물 종 간의 미세한 차이를 효과적으로 분류하는 데 기여하였습니다.



### Synthetic Prior for Few-Shot Drivable Head Avatar Inversion (https://arxiv.org/abs/2501.06903)
Comments:
          Website this https URL

- **What's New**: SynShot은 드라이버블 헤드 아바타의 few-shot inversion을 위한 새로운 방법이다. 이 방법은 합성 데이터를 기반으로 한 prior 모델을 학습하여 실제 이미지와의 도메인 간의 간극을 메우는 데 중점을 둔다. 기존의 최첨단 모노큘러 방법들이 수천 개의 진짜 이미지로 훈련되어야 하는 것과 달리, SynShot은 몇 개의 입력 이미지만으로도 photorealistic한 아바타를 재구성할 수 있는 능력을 가지고 있다.

- **Technical Details**: SynShot은 3D Gaussian splatting 및 convolutional encoder-decoder를 이용하여 헤드 아바타를 모델링한다. 입력된 실제 이미지로부터 사전 학습된 합성 prior를 조정하여 새로운 표현 및 카메라 뷰포인트에 일반화되는 헤드 아바타를 만든다. 이 과정은 synthetic data의 특성을 활용하여 데이터 캡처 하드웨어의 필요성을 없애고 개인 정보 보호 문제를 해결한다.

- **Performance Highlights**: SynShot은 몇 개의 입력 이미지만으로도 기존의 다양하고 복잡한 데이터를 요구하는 모노큘러 방법들보다 탁월한 성능을 보여준다. 실제 데이터와 합성 데이터 간의 도메인 간의 차이를 극복하는 방법을 통해 SynShot은 새로운 요인에 대한 일반화를 가능하게 하며, 이로 인해 제작하는 아바타의 품질이 크게 향상되었다.



### ActiveGAMER: Active GAussian Mapping through Efficient Rendering (https://arxiv.org/abs/2501.06897)
- **What's New**: ActiveGAMER는 3D Gaussian Splatting (3DGS) 기반의 새로운 능동적 매핑 시스템으로, 고품질의 실시간 장면 매핑과 탐색을 수행합니다. 기존 NeRF 기반 방법들과 달리, ActiveGAMER는 3DGS의 효율적인 렌더링 기능을 활용하여 복잡한 환경에서도 효과적이고 효율적인 탐색을 가능하게 합니다. 우리 시스템의 핵심은 정보 획득 모듈로, 동적으로 가장 유익한 시점을 선정하여 다음 탐색 방향을 계획합니다.

- **Technical Details**: ActiveGAMER는 3DGS를 기반으로 하며, 이는 Gaussian 형태의 기본 요소들을 사용해 3D 공간에서 장면을 표현합니다. 이 방식은 각 Gaussian을 이미지 평면으로 투영하고 알파 합성을 통해 혼합하여 고속 렌더링 속도를 유지할 수 있습니다. 또한, 탐색 효율성과 재구성 정확성을 극대화하기 위해 중간-정밀 탐색, 후속 개선 및 글로벌-로컬 키프레임 선택 전략을 통합한 균형 잡힌 프레임워크를 구현합니다.

- **Performance Highlights**: ActiveGAMER는 복잡한 환경에서 조화로운 6DoF 탐색을 가능하게 하며, 기존 접근 방식들보다 더 높은 기하학적 και 광학적 정확성과 완전성을 보여줍니다. 벤치마크 데이터셋인 Replica 및 MP3D에서의 광범위한 평가를 통해 능동적 매핑 작업에서 ActiveGAMER의 효과성이 강조됩니다. 이를 통해 우리는 능동 재구성에서 최신 성능을 달성하였으며, 재구성의 품질을 크게 향상시켰습니다.



### MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis (https://arxiv.org/abs/2501.06887)
Comments:
          Accepted to 2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)

- **What's New**: 이 연구는 피부암 진단 분야에서 CLIP(Contrastive Language-Image Pretraining) 모델을 활용함으로써 의사들이 AI의 결정 과정을 이해하고 신뢰할 수 있도록 하는 방법을 제안합니다. 특히, MedGrad E-CLIP이라는 새로운 설명 가능성(explainability) 방법을 도입하여, 복잡한 의료 이미지를 위한 가중 엔트로피 메커니즘을 통합합니다. 이 접근 방식은 특정 진단 설명과 연결된 이미지의 중요한 영역을 강조하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 PH² 및 Derm7pt 데이터셋을 사용하여 dermoscopic(피부 경화의) 구조 기준이 포함된 이미지를 연구했습니다. CLIP은 이미지 인코더와 텍스트 인코더로 구성되어 있으며, 이미지를 통해 진단 기준과의 관계를 학습합니다. 샘플 이미지와 설명의 커플을 통해, CLIP의 가중치를 학습하여 새로운 이미지-텍스트 쌍을 분류하는 방식으로 작동합니다.

- **Performance Highlights**: 제안된 메서드는 이미지에서 특징을 추출하고 이를 텍스트 기준과 일치시켜 피부 병변을 분류합니다. 이를 통해 기존의 진단 방법보다 향상된 신뢰성 및 투명성을 제공하며, AI 기반 진단 시스템에서 의사들 사이의 신뢰를 증진시키는 데 기여합니다. 이와 같은 성과는 깊은 신경망(Deep Neural Networks)의 의료 분석 적용에 있어 중요한 진전을 보여줍니다.



### Transforming Vision Transformer: Towards Efficient Multi-Task Asynchronous Learning (https://arxiv.org/abs/2501.06884)
Comments:
          Accepted by the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 연구에서는 Vision Transformer의 멀티태스크 학습(Multi-Task Learning, MTL) 성능을 향상시키기 위한 새로운 접근 방식인 Efficient Multi-Task Learning (EMTAL)을 제안합니다. 기존 Mixture-of-Experts (MoE) 구조와 Low-Rank Adaptation (LoRA)의 조합이 최적화를 저해하는 문제를 해결하여 더 효율적인 멀티태스크 학습을 가능하게 합니다. EMTAL은 사전 훈련된 Vision Transformer를 효과적인 멀티태스크 학습자로 변환하고, 효율적인 추론을 위해 학습된 구조를 재매개변수화합니다.

- **Technical Details**: EMTAL은 MoEfied LoRA 구조를 개발하여 사전 훈련된 Transformer를 낮은 차원의 MoE 구조로 분해하고, 이를 LoRA를 통해 미세 조정합니다. 또한, QR 최적화 메커니즘을 도입하여 비동기적으로 멀티태스크 지식을 학습하고, 라우터 페이딩 전략을 통해 학습된 파라미터를 원래 Transformer에 통합하여 효율적인 추론을 달성합니다. 이는 각 태스크의 고유한 학습 페이스에 따라 비판적 정보를 보존하며 경쟁 간섭을 줄입니다.

- **Performance Highlights**: 공식 벤치마크에서의 광범위한 실험 결과, EMTAL은 기존 최신 멀티태스크 학습 접근법들에 비해 우수한 성능을 보여줍니다. 특히, EMTAL은 여러 태스크를 동시에 최적화하는 과정에서 모델의 일반화 능력을 향상시키고, 이러한 개선은 다양한 실제 응용 프로그램에서 가시화됩니다. 또한, 훈련 효율성과 추론 속도가 크게 개선되어 실제 환경에서도 더 나은 활용 가능성을 제시합니다.



### Uncertainty-Aware Online Extrinsic Calibration: A Conformal Prediction Approach (https://arxiv.org/abs/2501.06878)
Comments:
          Accepted for publication at WACV 2025

- **What's New**: 이번 논문에서는 온라인 외부 보정(online extrinsic calibration) 과정에 불확실성 인식을 통합하는 최초의 접근법을 제시합니다. Monte Carlo Dropout과 Conformal Prediction을 결합하여 보정 매개변수의 불확실성을 정량화하고, 변화하는 환경에서 신뢰할 수 있는 예측 구간을 생성하는 방법론을 제안합니다. 이 연구는 다양한 네트워크 아키텍처와 호환되는 프레임워크를 통해 기존 보정 모델을 개선합니다.

- **Technical Details**: 제안된 방법론은 딥러닝 기반 모델에 내장되어 있으며, 모델 불확실성(epistemic uncertainty)을 정량화하는 데 중점을 둡니다. 이는 MCD와 CP를 활용하여 보정 매개변수에 대한 신뢰할 수 있는 예측 구간을 제공하여, 불확실성을 실시간으로 조정합니다. 실험은 KITTI와 DSEC 데이터셋에서 수행되었으며, 다양한 시각 센서 유형에 대한 일반화와 효과성을 입증하였습니다.

- **Performance Highlights**: 논문에서는 제안된 방법이 기존 캘리브레이션 모델에 비해 신뢰성과 효율성을 크게 향상시킬 수 있음을 강조합니다. 보정 파라미터에 대한 정량화된 신뢰 측정치를 제공함으로써, 동적 환경에서 센서 융합의 견고성을 지속적으로 개선할 수 있다는 점에서 큰 기여를 할 것으로 기대됩니다. 이 연구는 컴퓨터 비전 커뮤니티에도 실질적인 영향을 미칠 것으로 보입니다.



### LarvSeg: Exploring Image Classification Data For Large Vocabulary Semantic Segmentation via Category-wise Attentive Classifier (https://arxiv.org/abs/2501.06862)
Comments:
          PRCV 2024

- **What's New**: 이번 논문에서는 LarvSeg라는 새로운 대용량 어휘 의미 분할 프레임워크를 제안합니다. LarvSeg는 이미지 분류 데이터를 활용하여 심층 학습 모델의 의미 분할 능력을 확장하도록 설계되었습니다. 이 접근법은 기존의 어휘 기반 방법에서 발생하는 문제점을 개선하여 학습 데이터의 범위를 넓힙니다.

- **Technical Details**: 이 프레임워크는 이미지 수준의 감독을 픽셀 수준의 비율에 통합하여 더 많은 범주의 세분화를 가능하게 합니다. 특히 카테고리별 주의 분류기(category-wise attentive classifier)를 도입하여 특정 영역에서의 감독을 구현함으로써 모델의 성능을 향상시킵니다. LarvSeg는 COCO-Stuff와 같은 제한된 어휘 세분화 데이터와 ImageNet21K와 같은 대규모 이미지 분류 데이터셋을 결합해 학습합니다.

- **Performance Highlights**: 실험 결과, LarvSeg는 이전의 공개 어휘 모델보다 월등한 성능을 보였으며, 특히 마스크 레이블이 없는 범주에서 성능이 현저하게 개선되었습니다. 21K 카테고리에 대한 의미 분할 모델이 최초로 제공되며, 이를 통해 다양한 적용 가능성이 제시됩니다.



### Faithful Counterfactual Visual Explanations (FCVE) (https://arxiv.org/abs/2501.06841)
- **What's New**: 이번 연구에서는 신뢰성 있는 반사실적 설명(FCVE) 모델을 제안하여 딥러닝 모델의 의사결정 과정을 시각적으로 설명합니다. 반사실적 설명(counterfactual explanation) 기법을 통해 입력 이미지의 최소한의 변경으로 쉽게 이해 가능한 시각적 설명을 생성하며, 이는 내부 개념 및 필터를 활용하여 이루어집니다. 이러한 방법은 설명의 신뢰성을 보장하여, 모델 내부의 의사결정 과정을 명확하게 반영합니다.

- **Technical Details**: 제안된 Faithful Counterfactual Visual Explanation (FCVE) 모델은 기존의 픽셀 데이터 변경 없이 내부 개념을 식별하고 필터를 통해 플로우를 생성합니다. 많은 기존 설명 기법들이 모델의 신뢰성을 결여하고 비싼 계산 비용을 수반하는 단점을 가지고 있는데 반해, FCVE 모델은 이러한 문제를 개선합니다. 특히, DCNN의 내부 필터 및 반사실적 객체를 시각적으로 설명하는 포스트 호크 기법에 초점을 맞추고 있습니다.

- **Performance Highlights**: FCVE 모델은 기존의 GAN 기반 기법들과 비교하여 신뢰성을 높이는 데 중점을 두고 있습니다. 예를 들어, Goyal et al.의 연구와 같이 생성된 설명들이 시각적으로 그럴듯하더라도 그 신뢰성 평가에 대한 심층적인 논의가 부족했던 점을 보완합니다. 본 연구는 DCNN의 의사결정 과정을 이해하는 데에 기여하여, 더욱 투명하고 신뢰할 수 있는 AI 모델 훈련을 가능하게 합니다.



### SAM-DA: Decoder Adapter for Efficient Medical Domain Adaptation (https://arxiv.org/abs/2501.06836)
Comments:
          WACV25

- **What's New**: 이 논문은 의료 이미징에서 의미론적 분할을 위한 도메인 적응(domaine adaptation) 문제를 다룹니다. 최근 자연 이미지에 대해 인상적인 성능을 보여주는 SAM과 같은 기초적인 분할 모델들이 있지만, 의료 도메인 이미지에 대해서는 어려움을 겪습니다. 이에 대한 해결책으로 저자들은 새로운 SAM 어댑터(SAM adapter) 접근 방식을 제안하며, 이는 학습 가능한 매개변수의 수를 최소화하면서도 전체 fine-tuning과 유사한 성능을 달성하는 것을 목표로 합니다.

- **Technical Details**: 제안한 SAM 어댑터는 마스크 디코더(mask decoder)에서 전략적으로 배치되어 있으며, 완전 감독 및 시험 시간 도메인 적응 과제에서 개선된 분할 성능을 제공합니다. 이 어댑터는 사전 학습된 모델의 도메인 지식을 계속해서 활용하게끔 설계되었습니다. 이에 따라 이미지 인코더와 마스크 디코더는 적응 과정 동안 상당한 매개변수 업데이트가 필요하지 않습니다.

- **Performance Highlights**: 포괄적인 검증을 통해 이 어댑터는 기존 방법들을 능가하며, 전체 SAM 매개변수의 1% 미만을 훈련하여 뛰어난 성능을 보여줍니다. 특히, 저자들은 어댑터가 레퍼런스 방법인 LoRA와 HQ-SAM 뿐만 아니라 의료 관련 방법에도 우수한 성과를 거두었음을 입증하였습니다. 이 연구결과는 여러 의료 데이터셋과 자연 이미지 데이터셋에서의 실험을 통해 검증되었습니다.



### X-LeBench: A Benchmark for Extremely Long Egocentric Video Understanding (https://arxiv.org/abs/2501.06835)
- **What's New**: 이 논문은 매우 긴 개인 중심의 비디오 이해를 평가하기 위한 새로운 벤치마크 데이터셋인 X-LeBench를 소개합니다. 그동안의 데이터셋은 대개 단기 비디오에 초점을 맞추었으며, 23분에서 16.4시간에 이르는 긴 비디오 기록들을 포함한 유일한 데이터셋입니다. X-LeBench는 현실적인 일일 계획을 생성하기 위해 큰 언어 모델(LLMs)을 활용하여, 실제 비디오 데이터와 일치하는 유기적인 시뮬레이션 파이프라인을 개발했습니다.

- **Technical Details**: X-LeBench는 일상 활동을 반영하는 비디오 생명 로그를 생성하기 위해, 기존의 단기 또는 중간 길이의 비디오에서 데이터를 합성하는 시뮬레이션 파이프라인을 포함합니다. 이 파이프라인은 LLM을 사용하여 실제 활동에 맞춘 현실적인 일일 계획을 동적으로 생성하고, 이를 Ego4D 데이터셋의 실제 영상과 통합합니다. 이 과정은 432개의 시뮬레이션 비디오 생명 로그를 생성하여 풍부한 맥락의 현실적인 일상 활동을 재현합니다.

- **Performance Highlights**: 기존 모델들의 평가 결과는 X-LeBench에서 일관되게 낮은 성능을 보였습니다. 이는 긴 개인 중심 비디오 이해의 고유한 과제를 강조하며, 더 발전된 모델이 필요함을 시사합니다. 이 연구는 초장기 개인 중심 비디오를 효과적으로 해석하고 분석할 수 있는 솔루션 개발을 촉진하기 위해, 종합적인 벤치마크를 제공하는 데 목적을 두고 있습니다.



### Towards Counterfactual and Contrastive Explainability and Transparency of DCNN Image Classifiers (https://arxiv.org/abs/2501.06831)
- **What's New**: 이 논문에서는 딥 컨볼루션 신경망(DCNN)의 해석 가능성을 높이기 위한 새로운 방법을 제안합니다. 제안된 방법은 입력 이미지를 변경하지 않고 DCNN의 내부 작동을 탐색함으로써 해석 가능한 반사실적(counterfactual) 및 대조적(contrastive) 설명을 생성합니다. 이를 통해 모델의 결정이 특정 클래스 또는 다른 지정된 클래스에 대해 어떤 필터에 의해 이루어지는지를 명확히 할 수 있습니다.

- **Technical Details**: 제안된 방법은 주로 최상위 컨볼루션 레이어에서 필터를 식별하여 DCNN 모델에서 예측 결정의 근본 원인을 분석합니다. 이는 모델의 주요 개념과 특징을 나타내며, 이 필터들이 활성화되거나 비활성화되도록 조정될 때, DCNN이 입력 이미지를 지칭된 클래스로 예측하도록 도와줍니다. 이 방법은 DCNN의 모든 레이어에서 필터를 식별할 수 있으며, 다른 네트워크에서도 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: 이 연구의 결과는 Caltech-UCSD Birds (CUB) 2011 데이터셋에서의 성능을 보여주며, 제안된 설명 방법이 최신 기술과 비교했을 때 유용하다는 것을 입증합니다. 또한, 오분류 분석에서 이 방법의 적용 사례도 제시하여 모델의 결정 유효성을 검증하는 데 강력한 도구가 될 수 있음을 강조합니다.



### GeoPix: Multi-Modal Large Language Model for Pixel-level Image Understanding in Remote Sensing (https://arxiv.org/abs/2501.06828)
- **What's New**: GeoPix는 원거리 감지(Original Sensing, RS)에서 픽셀 수준의 대화 기능을 보장하는 다중 모달 대형 언어 모델(MLLM)로, 기존 MLLM의 한계를 극복하는 중요한 발전을 이룩했습니다. 다른 모델들이 이미지 또는 영역 수준의 작업에만 초점을 맞춘 것과 달리, GeoPix는 이미지 캡셔닝(image captioning), 시각적 질문 응답(visual question answering), 시각적 기본 설정(visual grounding)뿐만 아니라, 픽셀 수준에서의 분할 작업에도 능숙합니다. 이 연구에서는 Class-wise Learnable Memory (CLM) 모듈과 GeoPixInstruct 데이터셋 구성 등 혁신적인 접근 방식을 소개합니다.

- **Technical Details**: GeoPix는 비전 인코더로부터 추출한 시각적 특성에 기반해 마스크 예측기(mask predictor)를 추가함으로써 픽셀 수준의 이해 기능을 확장합니다. 이 마스크 예측기는 LLM의 분할 토큰 임베딩에 조건화된 마스크를 생성합니다. GeoPixInstruct 데이터셋은 65,463장의 이미지와 140,412개의 인스턴스로 구성되며, 각 인스턴스는 텍스트 설명, 경계 상자(bounding box), 마스크로 주석 처리되었습니다. 또한, 텍스트 생성과 마스크 예측을 최적화하기 위한 두 단계 훈련 전략을 개발하였습니다.

- **Performance Highlights**: GeoPix는 다중 참조(segmentation) 작업에서 최첨단 성능을 보여주며, 이미지 및 영역 수준의 벤치마크에서도 경쟁력 있는 성능을 유지합니다. 관련 실험 결과에서 GeoPix는 기존 모델들과 비교해 픽셀 수준의 분할 작업에서 뛰어난 효과를 입증하였습니다. 특히, CLM 모듈의 도입으로 인해 다중 규모 객체의 분할 정확성 및 일관성을 효과적으로 향상시켜, RS 이미지에서의 성능을 극대화하였습니다.



### UR2P-Dehaze: Learning a Simple Image Dehaze Enhancer via Unpaired Rich Physical Prior (https://arxiv.org/abs/2501.06818)
- **What's New**: 본 논문에서는 기존의 이미지 디헤이징(dehazing) 방법의 한계를 극복하기 위해, UR2P-Dehaze라는 비매칭(unpaired) 이미지 디헤이징 네트워크를 제안합니다. 이번 연구는 조명, 반사 및 색상 정보를 정확하게 추정하는 공유 사전 추정기(shared prior estimator, SPE)를 설계하여 명확하고 고품질의 이미지를 생성합니다. 이와 함께 Self-Monitoring 메커니즘을 도입하여 불필요한 특징을 제거하고 신뢰할 수 있는 사전을 제공합니다.

- **Technical Details**: UR2P-Dehaze는 이미지 처리 과정에서 하이프리퀀시(low and high frequency) 키 특징을 효과적으로 통합하는 Dynamic Wavelet Separable Convolution (DWSC) 기술을 사용하여 이미지 세부 정보를 보존하고 글로벌 일관성을 보장합니다. 또한 Adaptive Color Corrector를 통해 색상 정보를 효과적으로 복원하며, PSNR, SSIM, LPIPS, FID 및 CIEDE2000 지표에서 최첨단 성과를 달성하는 것을 보여줍니다. 이 접근 방식은 해사 조건에 더 잘 적응하고 이미지 처리 중 세부 사항을 더욱 잘 보존할 수 있게 합니다.

- **Performance Highlights**: UR2P-Dehaze는 공공 데이터셋(SOTS-indoor, SOTS-outdoor, I-HAZE, HSTS)에서 실험적인 결과를 통해 뛰어난 성능을 입증하였으며, 특히 색상 복원에 민감한 성능을 갖추었습니다. 이는 기존 최첨단 기술과 비교했을 때 더 다양한 환경에서의 일반화 능력을 향상시켰습니다. 본 연구의 기여는 UR2P-Dehaze 구조가 간결함을 유지하면서도 풍부한 사전 지식을 효율적으로 학습하는 점에 있습니다.



### RSRefSeg: Referring Remote Sensing Image Segmentation with Foundation Models (https://arxiv.org/abs/2501.06809)
- **What's New**: 新 모델 RSRefSeg는 원거리 감지 이미지를 세분화하는 데 매우 효과적입니다. 이 모델은 CLIP과 SAM를 통합하여 텍스트와 비주얼 간의 일치를 개선하고, 상세한 분할 마스크를 생성하는 데 이를 사용합니다. 특히, RSRefSeg는 교차 도메인 전이 성능 저하 문제를 극복하고 여러 기초 모델 간의 일반 지식을 통합하는 방식으로 주목받습니다.

- **Technical Details**: RSRefSeg는 CLIP의 아키텍처와 SAM을 통합하여 원거리 감지 이미지를 위한 강력한 세분화 능력을 제공합니다. 모델은 주로 fine-tuned CLIP, AttnPrompter 및 fine-tuned SAM의 세 가지 주요 구성 요소로 이루어져 있습니다. 이 구조를 통해 CLIP은 자유로운 형식의 텍스트와 저해상도 비주얼 피처를 추출하여, AttnPrompter는 이러한 피처를 처리하여 SAM과 호환되는 프롬프트 임베딩을 생성합니다.

- **Performance Highlights**: RRSIS-D 데이터셋에서 RSRefSeg는 기존 방법들보다 뛰어난 성능을 보였습니다. 이는 기초 모델의 활용과 멀티모달 작업 이해의 실용적 가치를 크게 강조합니다. RSRefSeg는 불확실한 세분화 마스크의 개선된 표현을 통해 원거리 감지의 세분화 작업에서 큰 진전을 이루었습니다.



### Semantic-CD: Remote Sensing Image Semantic Change Detection towards Open-vocabulary Setting (https://arxiv.org/abs/2501.06808)
- **What's New**: 본 논문에서는 기존의 일반적인 Change Detection 방법의 한계를 극복하기 위해 Semantic-CD라는 새로운 방법론을 소개합니다. Semantic-CD는 CLIP 모델의 오픈 어휘( open vocabulary ) 개념을 활용하여 원격 탐지 이미지에서의 의미적 변화 감지를 수행합니다. 이 접근법은 이진 변화 감지(BCD)와 의미적 변화 감지(SCD) 작업을 완전히 분리하여 각각의 작업에서 CLIP의 시각적 세부 사항을 활용합니다.

- **Technical Details**: Semantic-CD는 네 가지 핵심 구성 요소로 구성됩니다: Bi-temporal CLIP visual encoder, open semantic prompter, binary Change Detection (BCD) decoder, 그리고 Semantic Change Detection (SCD) decoder입니다. 이 구조는 이진 변화 감지와 의미적 변화 감지 두 가지 작업의 특징을 효과적으로 추출하고 처리하기 위해 수학적으로 설계되었습니다. 특히, open semantic prompter를 통해 생성된 pixel-level cost volume을 사용하여 의미적 변화 감지가 이루어 집니다.

- **Performance Highlights**: SECOND 데이터셋에 대한 실험 결과, Semantic-CD는 더 정확한 의미적 변화 마스크를 생성하며, 의미적 분류 오류를 줄이는 성과를 보였습니다. 이러한 결과는 vision-language foundation model인 CLIP의 활용이 의미적 변화 감지(SCD) 작업에서의 효과성을 입증하고 있습니다. Semantic-CD는 기존의 방법들보다 우수한 성능을 달성하며, 오픈 어휘 설정에서의 일반화 가능성을 보여줍니다.



### Improving Pain Classification using Spatio-Temporal Deep Learning Approaches with Facial Expressions (https://arxiv.org/abs/2501.06787)
Comments:
          8 pages, 3 figures, 3 tables. Accepted and presented at the 18th International Conference on Machine Vision (ICMV 2024), Edinburgh, UK

- **What's New**: 본 연구는 전통적인 자가 보고 방식이 주관적이며 비언어적인 개인에게 적합하지 않은 문제를 해결하기 위해 얼굴 표정을 활용한 자동 통증 감지를 탐구합니다. Pain Emotion Faces Database (PEMF)를 기반으로 깊이 있는 학습(Deep Learning) 기법을 통해 통증 평가를 개선하고자 하였습니다.

- **Technical Details**: 우리는 두 가지 새로운 접근 방식을 제안합니다: (1) 비디오 프레임을 분석하고 통증 존재 여부를 예측하기 위해 Long Short-Term Memory (LSTM) 블록과 결합된 하이브리드 ConvNeXt 모델을 사용합니다. (2) 얼굴 이미지의 랜드마크를 처리하여 통증을 감지하기 위한 Spatio-Temporal Graph Convolution Network (STGCN) 모델이 LSTM과 통합되었습니다.

- **Performance Highlights**: 우리는 PEMF 데이터셋을 사용하여 이진 통증 분류를 최초로 수행하였으며, 이러한 모델의 효과성을 광범위한 실험을 통해 입증하였습니다. 공간적 및 시간적 특성을 결합하여 통증 감지의 향상 가능성을 강조하며, 객관적인 통증 평가 방법론에 있어서 유망한 발전을 제시하고 있습니다.



### Temporal-Aware Spiking Transformer Hashing Based on 3D-DW (https://arxiv.org/abs/2501.06786)
Comments:
          TPAMI under review. This work has been submitted to the lEEE for possible publication

- **What's New**: 이 논문에서는 동적 비전 센서(DVS) 데이터를 효과적으로 검색하기 위한 새로운 방법 Spikinghash를 제안합니다. Spikinghash는 경량 구조를 통해 낮은 에너지 소비로 해시 코드를 생성할 수 있는 혁신적인 감독 해싱 방법으로, spiking neural networks(SNNs)의 이진 특성을 활용합니다. 이는 에너지 효율성을 높이는 동시에 뛰어난 검색 성능을 도모합니다.

- **Technical Details**: Spikinghash의 핵심은 스파이킹 웨이브믹서(Spiking WaveMixer, SWM)입니다. SWM은 다중 레벨 3D 이산 웨이브렛 변환(3D-DWT)을 사용하여 시간-공간 특성을 분해하고, 이를 통해 저주파 및 고주파 컴포넌트를 효율적으로 융합합니다. 또한, 깊은 층에서는 Spiking Self-Attention(SSA)을 통해 전역적 시공간 정보를 추가로 추출합니다.

- **Performance Highlights**: 다양한 데이터셋에서 Spikinghash는 최신 기술인 SNN보다 우수한 분류 및 검색 성능을 보여줍니다. 실험 결과, Spikinghash는 낮은 에너지 소모와 적은 매개변수 수를 유지하면서도 기존 방법보다 보다 뛰어난 성능을 달성할 수 있음을 입증했습니다. 이러한 결과는 DVS 비디오 및 이미지 검색에서의 효율성을 크게 향상시킬 것입니다.



### 3DCoMPaT200: Language-Grounded Compositional Understanding of Parts and Materials of 3D Shapes (https://arxiv.org/abs/2501.06785)
- **What's New**: 이 논문에서는 3D 객체 부품 및 재료에 대한 구성을 이해하기 위해 200개의 객체 범주를 갖춘 대규모 데이터세트인 3DCoMPaT200을 소개합니다. 3DCoMPaT200은 기존의 3DCoMPaT보다 약 5배 더 큰 객체 어휘와 약 4배 더 많은 부품 범주를 포함하고 있으며, 총 1,031개의 세부 부품 범주와 293개의 독특한 재료 클래스를 제공합니다. 이는 복잡한 3D 형태를 구성적 관점에서 이해하는 데 필요한 데이터셋의 한계를 크게 극복합니다.

- **Technical Details**: Part-level 3D 객체 이해는 기계 인식의 기본적인 요소로, 시스템이 객체의 구성 요소 및 재료를 해석하고 상호작용할 수 있게 해줍니다. 3DCoMPaT200은 19,000개의 3D 형태를 포함하며, 세부적인 자료 수집 및 렌더링 파이프라인을 통해 데이터 예제를 제공합니다. 또한, ULIP를 이용한 다중 모드 정렬 실험을 통해 데이터셋의 중요성을 측정하고자 했습니다.

- **Performance Highlights**: 연구 결과, 모델의 성능은 구성 스타일의 수가 증가함에 따라 향상되며, 3DCoMPaT200 데이터셋이 복잡한 3D 형상을 이해하는 데 있어 중요한 역할을 한다는 것을 보여줍니다. 본 데이터셋을 이용한 객체 분류, 부품-재료 분할 및 Grounded Compositional Recognition (GCR) 기술의 벤치마크 평가를 진행하였으며, 이는 3D 객체 이해 기술 개발을 촉진할 수 있는 잠재력을 지니고 있음을 나타냅니다.



### SuperNeRF-GAN: A Universal 3D-Consistent Super-Resolution Framework for Efficient and Enhanced 3D-Aware Image Synthesis (https://arxiv.org/abs/2501.06770)
- **What's New**: SuperNeRF-GAN은 3D 일관성을 유지하면서 고해상도 이미지를 합성할 수 있는 새로운 프레임워크입니다. 기존 NeRF 기반 방법들과 원활하게 통합할 수 있는 점이 특징입니다. 전처리된 생성기를 기반으로 낮은 해상도의 이미지를 생성한 후, NeRF Super-Resolution 모듈을 통해 고해상도의 NeRF 표현을 생성합니다. 실험 결과, SuperNeRF-GAN은 3D 일관성 및 효율성에서 우수한 성능을 보여주었습니다.

- **Technical Details**: SuperNeRF-GAN은 NeRF 기반 이미지 합성 방법과 통합하여 작동하며, 먼저 부피 렌더링(volume rendering)을 통해 낮은 해상도의 이미지를 생성합니다. 이후 NeRF Super-Resolution 모듈을 사용하여 고해상도 출력 이미지의 특성을 학습하고, 깊이 기반 렌더링 과정(depth-guided rendering process)에서 다중 깊이 맵(multi-depth map)을 통합하여 3D 일관성을 유지합니다. 이 과정은 경계 수정(boundary correction)과 정상 지도(normal map) 작업을 포함하여 더욱 정교한 결과를 도출할 수 있게 합니다.

- **Performance Highlights**: SuperNeRF-GAN을 사용한 실험에서 얼굴, 고양이 얼굴 및 전신 이미지 합성의 경우 기존의 사전 훈련된 모델과 비교하여 3D 일관성과 효율성에서 유의미한 향상이 있었습니다. 특히, 다른 최신 기법들(SOTA)과 비교했을 때 3D 일관성과 이미지 품질 모두에서 우수한 성과를 보였습니다. 또한, 여러 실험을 통해 SuperNeRF-GAN의 효과가 검증되었습니다.



### ODPG: Outfitting Diffusion with Pose Guided Condition (https://arxiv.org/abs/2501.06769)
Comments:
          11 pages, 5 figures. Preprint submitted to VISAPP 2025: the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications

- **What's New**: ODPG(Outfitting Diffusion with Pose Guided Condition) 방법은 의복, 자세, 외모 이미지를 잠재적 특성으로 변환하고 이를 UNet 기반의 디노이징 모델에 통합하여 동적 포즈를 가진 인간 이미지에 비의도적인 의상 합성을 가능하게 합니다. 이 접근법은 기존 VTON(Virtual Try-On) 모델들이 종종 특정 정적 포즈에 한정되어 진정성을 감소시키는 문제를 해결합니다. 또한, ODPG는 명시적인 의상 변형 프로세스 없이 다양한 자세에서 섬세한 질감 세부정보를 강조하는 사실적인 VTON 이미지를 생성합니다.

- **Technical Details**: ODPG는 VAE(변량 오토인코더)와 Swin Transformer를 사용하여 대상 의상 이미지의 잠재적 특성을 변환한 후, 이를 디노이징 과정에서 조건으로 통합합니다. 이 과정은 디노이징 확산 모델의 원리에서 영감을 받아 있으며, 이미지 생성 과정에서 Gaussian 노이즈를 점진적으로 추가한 다음 서서히 제거하여 깨끗한 이미지를 생성합니다. 특히, ODPG의 구조는 고급 세미틱 이해를 통해 동적 포즈에 맞는 의상 이미지를 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 ODPG는 FashionTryOn 및 DeepFashion 데이터셋의 하위 집합에서 다양한 포즈에 걸쳐 고해상도의 사실적인 VTON 이미지를 성공적으로 생성함을 보였습니다. ODPG는 다중 이미지 조건 컨트롤 생성에서 향상된 성능을 보여주며, 4개 쌍 데이터셋에 대한 의존도를 줄이면서도 새로운 조합에 대한 일반화 능력을 향상시킵니다. 향후 연구는 비디오 형식의 VTON 출력을 생성하고, 데이터가 제한된 다른 도메인에 우리의 주의 메커니즘을 적용하는 방향으로 나아갈 것입니다.



### VidChain: Chain-of-Tasks with Metric-based Direct Preference Optimization for Dense Video Captioning (https://arxiv.org/abs/2501.06761)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서 저자들은 Dense Video Captioning (DVC) 과제를 해결하기 위한 새로운 프레임워크인 VidChain을 제안합니다. VidChain은 Chain-of-Tasks (CoTasks)와 Metric-based Direct Preference Optimization (M-DPO)로 구성되어 있으며, 이를 통해 비디오 대형 언어 모델(VideoLLMs)의 정밀한 시간적 이해력을 향상시킵니다. 기존의 연구들이 단일 단계에서 DVC를 해결하려 한 것과는 달리, 이 프레임워크는 더 효과적으로 reasoning 능력을 활용할 수 있도록 돕습니다.

- **Technical Details**: CoTasks는 복잡한 문제를 하위 문제로 세분화하는 방식으로 구성되며, 각 단계에서 하나의 하위 문제를 해결함으로써 강력한 reasoning 능력을 발휘할 수 있습니다. M-DPO는 DVC의 평가 지표에 VideoLLM을 정렬시키고, 각 하위 작업에 맞춘 세밀한 감독을 제공합니다. 이러한 접근은 기존 훈련 목표(예: next-token prediction)와 DVC 평가 메트릭 간의 간극을 해소하는 데 중요한 역할을 합니다.

- **Performance Highlights**: VidChain은 Activitynet Captions와 YouCook2 두 가지 DVC 벤치마크에서 성능을 향상시켜 기존 VideoLLMs에 비해 우수한 결과를 보여주었습니다. 또한, temporal video grounding (TVG) 작업에서도 두드러진 성능 개선이 관찰되었습니다. 이 모델은 정밀한 감독을 통해 VideoLLMs의 비디오 이해력을 강화하며, LLM 기반 모델에 광범위하게 적용 가능하다는 점에서 의미가 있습니다.



### Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation (https://arxiv.org/abs/2501.06749)
- **What's New**: 이번 연구에서는 생물학적 이미지에서 특성 및 부위 세분화(image segmentation) 문제를 해결하기 위해 "Static Segmentation by Tracking (SST)"라는 새로운 방법론을 제안합니다. SST는 동일한 종의 표본 이미지에서 나타나는 일관된 특징을 활용하여, 몇 개의 라벨링된 이미지 만으로도 세분화를 수행할 수 있도록 설계되었습니다. 이는 전통적인 방법론인 수많은 이미지에 대한 수작업 라벨링을 하지 않고도 특성 측정을 가능하게 합니다.

- **Technical Details**: SST는 디지털 이미지 분석에서 종적인(temporal) 추적 문제로 세분화 문제를 재구성하는 기법입니다. 모델은 초기 이미지의 주석이 달린 마스크를 사용하여, 이후 이미지를 추적하고 마스크를 복제합니다. 이를 통해, SST는 Segment Anything Model 2 (SAM 2)을 기반으로 하여 하나의 라벨 이미지만으로도 품질 높은 세분화를 달성하는 것을 보여줍니다.

- **Performance Highlights**: 연구에서는 Cambridge Butterfly, NEON Beetle, Fish-Vista 등 세 가지 데이터셋에서 SST의 성능을 평가했습니다. SST는 SegGPT 등의 다른 원샷(one-shot) 세분화 기법보다 현저히 뛰어난 성능을 발휘했으며, 심지어 충분한 라벨이 있는 데이터로 학습된 모델들과 비교했을 때도 놀라운 결과를 보여주었습니다. 이러한 성과는 SST가 주석이 없는 이미지와 주석이 있는 이미지 간의 종속성을 명확히 활용했기 때문입니다.



### Diversified Augmentation with Domain Adaptation for Debiased Video Temporal Grounding (https://arxiv.org/abs/2501.06746)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 논문에서는 비디오에서의 Temporal Sentence Grounding (TSGV)의 새로운 훈련 프레임워크를 제안합니다. 기존의 데이터셋에서 발생하는 시간적 편향을 해결하기 위해 다양한 데이터 증강 기법과 도메인 적응 보조 작업을 결합하여 훈련합니다. 목표는 비디오의 길이와 타겟 순간 위치의 다양성을 증가시키는 한편, 원본과 증강 영상 간의 특징 불일치를 줄이는 것입니다.

- **Technical Details**: 제안된 프레임워크는 span-based grounding backbone을 사용하며, 표준 transformer encoder-decoder 아키텍처가 탑재되어 있습니다. 또한, 비디오 클립에서 clip-level 특징을 추출하기 위해 pre-trained I3D 네트워크를 사용하고, 특징을 높은 수준의 의미 공간으로 투영하기 위해 multi-layered perceptron (MLP)을 활용합니다. 데이터 증강 전략은 비디오의 길이와 타겟 순간 위치를 다양화하여 훈련 데이터를 풍부하게 만드는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과 Charades-CD와 ActivityNet-CD 데이터셋에서 제안된 방법이 여러 grounding 구조에서 뛰어난 성능을 보여주었습니다. 이 방법은 기존의 방법들보다 우수한 일반화 능력을 가지며, 식별된 타겟 순간의 예측 정확도를 높입니다. 결과적으로, 우리는 다양한 grounding 구조에 쉽게 통합할 수 있는 효과적인 훈련 프레임워크를 성공적으로 개발했습니다.



### Rice Leaf Disease Detection: A Comparative Study Between CNN, Transformer and Non-neural Network Architectures (https://arxiv.org/abs/2501.06740)
Comments:
          6 pages, 6 figures

- **What's New**: 이번 연구에서는 방글라데시의 쌀 잎 질병 감지를 위한 다양한 컴퓨터 비전 기술을 탐색하였습니다. Dhan-Shomadhan 데이터셋을 활용하여 CNN과 Vision Transformer(비전 변환기) 모델의 성능을 비교하고, 전통적인 기계 학습 기법인 Support Vector Machine(SVM)과도 성능을 비교하였습니다. 이 연구는 적은 양의 학습 데이터로 일반화 능력을 향상시키기 위해 Transfer Learning(전이 학습)을 활용하였으며, ResNet50이 다른 모델들보다 최고의 성과를 보여 최적의 선택으로 평가되었습니다.

- **Technical Details**: 본 연구의 방법론은 데이터 전처리, 데이터 증강, 모델 구현의 세 가지 주요 요소로 구성되어 있습니다. 이미지 크기는 448x448 픽셀로 통일하였고, 픽셀 강도 정규화를 통해 데이터의 일관성을 유지했습니다. 또한, 데이터의 다양성을 증가시키고, 모델의 일반화능력을 향상시키기 위해 Random Resize, Random Crop, Random Perspective, Random Gaussian Blur와 같은 데이터 증강 기술을 적용했습니다. 뉴럴 네트워크 모델들은 모두 ImageNet 데이터셋으로 사전 훈련을 거쳤으며, SVM 모델은 직접 사용하거나 ResNet 기반 모델로부터 추출된 특징과 결합하여 활용되었습니다.

- **Performance Highlights**: 모델 테스트 결과, ResNet50이 방글라데시 쌀 잎 질병 감지 작업에서 가장 우수한 성과를 보였습니다. ResNet50은 잔차 학습(residual learning)을 통해 깊은 네트워크의 기울기 소실 문제를 해결하고 다양한 계층적 특징을 효과적으로 학습할 수 있습니다. Inception-V3와 MaxViT 또한 잔여 연결과 병렬 컨볼루션을 통해 복잡한 데이터셋의 특징을 잘 포착할 수 있어, 쌀 잎 질병 분류 작업에 적합한 성과를 보여주었습니다.



### Multi-Label Scene Classification in Remote Sensing Benefits from Image Super-Resolution (https://arxiv.org/abs/2501.06720)
- **What's New**: 이번 연구는 위성 이미지의 품질을 향상시키기 위한 전처리 단계로 이미지 슈퍼 해상도(Super-Resolution, SR)를 사용하여 다중 라벨 장면 분류 성능을 개선하는 방법을 탐구합니다. 연구팀은 SRResNet, HAT, SeeSR, RealESRGAN 등 네 가지 SR 모델을 조사하고, 이를 ResNet-50, ResNet-101, ResNet-152, Inception-v4와 같은 다양한 CNN 아키텍처와 결합하여 분류 성능의 영향을 평가했습니다. 이 연구는 SR 모델들이 다중 라벨 예측에서 사용할 수 있는 중요한 통찰력을 제공함으로써 기존 원격 감지 시스템의 품질 향상에 기여할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 고해상도 이미지를 생성하는 비용 효율적인 방법으로 이미지 SR을 사용하여 구체적인 세부 정보를 포착하고 다중 라벨 장면 분류의 정확성을 높이기 위해 다양한 SR 모델을 적용했습니다. 각 SR 모델은 저해상도(Lower-Resolution, LR) 입력을 바탕으로 고해상도(High-Resolution, HR) 이미지를 생성하며, SR 모델은 회귀 기반 모델과 생성적 SR 모델(GAN 및 확산 모델)로 나눌 수 있습니다. SRResNet 및 HAT는 회귀 기반 모델로, SeeSR 및 RealESRGAN은 생성적 SR의 대표로 분석되었습니다.

- **Performance Highlights**: SR을 통해 향상된 이미지는 다양한 메트릭에서 다중 라벨 장면 분류 성능을 크게 개선함을 보여주었습니다. 연구팀은 적용된 SR 기술이 공간 세부 사항을 보존할 수 있음을 입증하였으며, 이는 다중 라벨 태스크에서 중요한 요소입니다. 이 연구는 다중 라벨 예측을 위한 SR 기술 선택의 유용성을 강조하고, 원격 감지 시스템 개선을 위한 통합할 수 있는 프레임워크를 제시합니다.



### F3D-Gaus: Feed-forward 3D-aware Generation on ImageNet with Cycle-Consistent Gaussian Splatting (https://arxiv.org/abs/2501.06714)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문은 단일 모노큘러 데이터셋에서 일반화 가능한 3D 인식 생성을 다룹니다. 기존의 3D 인식 생성 방법은 다중 뷰나 동적 데이터를 필요로 했으나, 우리가 제안하는 ‘F3D-Gaus’ 모델은 단일 프레임 입력으로부터 더 현실적인 3D 렌더링을 생성할 수 있습니다. 자가 감독 사이클 일관성 제약(self-supervised cycle-consistent constraint)을 도입하여 학습된 3D 표현의 뷰 간 일관성을 강화했습니다.

- **Technical Details**: 기존 3D 생성 방법들은 주로 복셀(voxel), 포인트 클라우드(point cloud) 등 다양한 3D 표현을 사용합니다. 그런데 본 연구에서는 픽셀 정렬 픽셀 정렬된 가우시안 스플래팅(pixel-aligned Gaussian Splatting)을 기반으로 한 피드포워드(foward) 구조를 제안하여, 단일 이미지에서 3D 인식 생성을 가능하게 했습니다. 이는 여러 개의 정렬된 가우시안 원시(primitives)를 집계하여 뷰포인트에 따라 생성되는 이미지를 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 모노큘러 데이터셋에서 고품질의 다중 뷰 일관된 3D 인식 생성을 달성함을 입증했습니다. 본 접근법은 모델의 훈련 및 추론 효율성을 크게 개선했으며, 다양한 데이터셋에 대한 일반화 능력을 확인했습니다. 우리는 특히 복잡한 조명 조건에서도 뛰어난 세부 사항 표현을 구현하여 3D 콘텐츠 생성에 있어 중요한 발전을 가져왔습니다.



### Multi-task Visual Grounding with Coarse-to-Fine Consistency Constraints (https://arxiv.org/abs/2501.06710)
Comments:
          AAAI2025

- **What's New**: 본 연구에서는 Multi-task visual grounding에서의 불일치 문제를 해결하기 위해 Coarse-to-fine Consistency Constraints Visual Grounding 아키텍처($\text{C}^3\text{VG}$)를 제안합니다. 이 아키텍처는 두 단계로 구성되어 있으며, 초기 단계에서 쿼리와 픽셀 디코더를 통해 예비 탐지 및 분할 출력을 생성합니다. 이어지는 단계에서는 Mask-guided Interaction Module과 상호 일관성 제약 손실을 사용하여 다중 작업 간의 일관성을 확보합니다.

- **Technical Details**: C3VG는 Rough Semantic Perception(RSP) 단계에서 거친 예측을 수행한 후, Refined Consistency Interaction(RCI) 단계에서 이들 예측을 세밀하게 보정합니다. MIM은 다중 작업 결과의 내재적 상호작용을 통합하는 역할을 하며, 쌍방향 일관성 제약 손실을 통해 예측 결과의 일관성을 명시적으로 보장합니다. 이를 통해 REC 및 RIS 과제가 상호 보완적으로 작용할 수 있도록 개선합니다.

- **Performance Highlights**: C3VG는 RefCOCO, RefCOCO+, RefCOCOg 데이터세트에서 기존의 최첨단 REC 및 RIS 방법과 비교하여 현저한 성능 향상을 보여줍니다. 이 모델은 다중 작업 학습 프레임워크에서 사전 훈련된 멀티모달 인코더를 활용하여 정확도와 수렴 속도를 향상시킵니다. 또한, 훈련 에포크 수를 절반 이하로 줄이면서도 성능을 유지할 수 있는 효율성을 자랑합니다.



### Mamba-MOC: A Multicategory Remote Object Counting via State Space Mod (https://arxiv.org/abs/2501.06697)
- **What's New**: 이번 연구에서는 다양한 카테고리의 원거리 객체 개수를 정확하게 추정하는 다중 카테고리 원거리 객체 개수 세기(Multicategory Remote Object Counting, MOC) 문제를 다루고 있습니다. 특히, Mamba라는 새로운 모델을 기반으로 한 Mamba-MOC 프레임워크를 제안하였으며, 이는 원거리 객체 세기 분야에 Mamba를 최초로 적용한 사례입니다. 또한, 계층적 특성의 깊은 통합을 촉진하기 위한 교차 스케일 상호작용 모듈을 설계하였습니다.

- **Technical Details**: 제안된 방법론은 vmamba 백본(백본 네트워크)을 사용하여 다중 수준의 특성 표현을 추출하고, 교차 스케일 상호작용 모듈을 통해 coarse-grained와 fine-grained 특성의 효과적인 통합을 수행합니다. 또한, 컨텍스트 상태 공간 모델(Contextual State Space Model, CSSM)을 활용하여 스캔 과정 중 지역 이웃 정보를 강조하고, 이를 통해 더욱 세밀한 정보 추출을 가능하게 합니다.

- **Performance Highlights**: 대규모 실제 시나리오에서의 실험 결과는 제안된 Mamba-MOC 방법이 기존의 주요 객체 수 계산 알고리즘들과 비교할 때 최첨단 성능을 달성했음을 보여줍니다. 이는 Mamba의 전방위적인 장점을 활용함으로써 원거리 이미지 내에서 효과적인 글로벌 및 로컬 컨텍스트 정보를 모델링할 수 있음을 통한 결과입니다.



### Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation (https://arxiv.org/abs/2501.06693)
Comments:
          Project page: this https URL

- **What's New**: 이 연구에서는 Vid2Sim이라는 새로운 프레임워크를 제안하여 시뮬레이션에서 실세계로의 간극인 sim-to-real gap을 효과적으로 해소합니다. 기존의 방법들은 보통 도메인 랜덤화(domain randomization)와 시스템 식별(system identification)에 집중하였으나, 이러한 방법들은 시뮬레이션과 그래픽 엔진의 한계에 의해 제한됩니다. Vid2Sim은 모노큘러 비디오(monocular video)를 입력으로 하여 사실적인 3D 시뮬레이션 환경을 생성하고 이를 통해 복잡한 도시 환경에서 시각적 내비게이션 에이전트의 강화 학습을 지원합니다.

- **Technical Details**: Vid2Sim은 두 가지 주요 구성 요소로 이루어져 있습니다: 기하학적으로 일관된 장면 재구성과 사실적인 상호작용 시뮬레이션 구성입니다. 이를 위해, 모노큘러 단서(monocular cues)를 활용하여 Gaussian Splatting 훈련을 정규화하여 세부 기하학적 특징을 강화합니다. 또한, GS(가우시안 스플래팅) 표현과 메시 프리미티브(mesh primitives)를 결합한 하이브리드 장면 표현을 제안하여 사실적인 렌더링과 물리적 상호작용을 가능하게 합니다.

- **Performance Highlights**: Vid2Sim을 통해 훈련된 에이전트는 전통적인 시뮬레이션 방법으로 훈련된 에이전트에 비해 31.2% 및 68.3% 높은 성공률 향상을 보여줍니다. 실험 결과, Vid2Sim으로 훈련된 에이전트는 제로샷 시뮬레이션-실시간 전이(zero-shot sim2real transfer) 능력을 보여주었으며, 충돌률이 낮아지는 경향을 보였습니다. 이는 이 방법이 시뮬레이션과 현실 간의 간극을 줄이는 데 있어 확장 가능하고 효율적인 솔루션이 될 수 있는 가능성을 보여줍니다.



### PGP-SAM: Prototype-Guided Prompt Learning for Efficient Few-Shot Medical Image Segmentation (https://arxiv.org/abs/2501.06692)
Comments:
          5 pages, 2 figures, Accepted at ISBI 2025

- **What's New**: 이번 논문에서는 SAM(Segment Anything Model)의 의료 이미지 세분화(Customizing SAM for medical image segmentation)를 위한 새로운 방법인 PGP-SAM을 제안합니다. PGP-SAM은 제한된 샘플을 사용하여 수동 프롬프트를 대체할 수 있는 새로운 프로토타입 기반의 few-shot tuning 접근법입니다. 이 접근법은 대량의 픽셀 수준 주석 없이도 효과적으로 세분화를 수행할 수 있게 해줍니다.

- **Technical Details**: PGP-SAM의 주요 아이디어는 클래스 특정 지식(class-specific knowledge)과 관계를 포착하기 위해 클래스 내 및 외 프로토타입(intra- and inter-class prototypes)을 활용하는 것입니다. 여기에는 (1) 다중 스케일 정보를 통합하는 플러그 앤 플레이 모듈(plug-and-play contextual modulation module)과 (2) 프로토타입과 특성을 융합하여 자동 프롬프트 생성을 위한 클래스 지향 크로스 어텐션 메커니즘(class-guided cross-attention mechanism)이라는 두 가지 주요 구성 요소가 포함됩니다.

- **Performance Highlights**: PGP-SAM은 공개 멀티 오르간 데이터셋과 사설 심실 데이터셋을 통해 기존의 프롬프트 없는 SAM 변형들과 비교할 때 우수한 평균 Dice 점수를 달성합니다. 특히, 2D 슬라이스의 10%만을 사용하여도 높은 성능을 발휘하는 점이 주목할 만합니다.



### Application of Vision-Language Model to Pedestrians Behavior and Scene Understanding in Autonomous Driving (https://arxiv.org/abs/2501.06680)
- **What's New**: 최근 자율주행(Autonomous Driving) 기술이 크게 발전하여 3D 탐지, 분류, 로컬라이제이션 결과를 보여주고 있습니다. 그러나 보행자 행동의 의미적 이해 및 보행자와의 인터랙션 처리와 같은 많은 도전 과제가 여전히 남아 있습니다. 이 연구에서는 대규모 언어 모델(LLM) 및 비전-언어 모델(VLM)의 지식을 소형 비전 네트워크로 효과적으로 증류하는 방법을 분석하여, 복잡한 장면의 의미적 표현으로 의사결정 및 제어에 활용할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 비전-언어 기초 모델은 시각 데이터와 텍스트 데이터를 통합하여 다중 모달 AI의 최전선에 있는 기술입니다. 이 논문에서는 GPT4-V와 같은 사전 훈련된 비전-언어 모델로부터 일반 지식을 전이하는 지식 증류(Knowledge Distillation)를 수행하였으며, 이를 통해 보행자의 행동과 의미적 속성을 이해하는 효율적인 비전 모델을 개발하였습니다. 이 과정에서 다중 라벨 분류 문제로 설정하고, GPT의 주석을 통해 생성된 의미적 텍스트 라벨로 비전 네트워크를 감독하여 정확한 예측을 목표로 하였습니다.

- **Performance Highlights**: 우리는 이 연구를 통해 열거된 과제들에 대해 의미적으로 풍부한 보행자 속성과 분류 체계를 통해 개선된 성과를 달성하였습니다. 전통적인 분류법보다 더 많은 보행자 속성과 의미적 카테고리를 수집하여, 보다 정교한 자율주행 시스템을 위한 기반을 마련하였습니다. 이를 통해 보행자의 행동을 보다 정확하게 예측하고, 안전하고 신뢰할 수 있는 자율주행 내비게이션을 위한 기반을 강화하였습니다.



### Imbalanced Medical Image Segmentation with Pixel-dependent Noisy Labels (https://arxiv.org/abs/2501.06678)
- **What's New**: 이번 연구에서는 의료 이미지 분할의 성능을 저하시키는 픽셀 의존성 노이즈(label noise) 문제를 해결하기 위해 Collaborative Learning with Curriculum Selection (CLCS) 프레임워크를 제안합니다. CLCS는 노이즈 레이블을 픽셀 단위로 처리하고 다양한 클래스 불균형(class imbalance) 문제를 해결하기 위해 협력적 학습(collaborative learning) 프레임워크를 활용합니다. 또한, 새로운 curriculum dynamic thresholding 접근법을 통해 모델의 학습 진행에 적응하는 동적 임계값을 사용하여 깨끗한 데이터를 선택하는 방법이 특징입니다.

- **Technical Details**: CLCS 프레임워크는 Curriculum Noisy Label Sample Selection (CNS)와 Noise Balance Loss (NBL)의 두 모듈로 구성되어 있습니다. CNS 모듈은 협력적 학습을 위해 두 개의 브랜치 네트워크를 설계하여 서로 다른 뷰에서 동일한 인스턴스의 다양한 특징 표현을 추출하고 픽셀의 클래스 확률을 투표하여 결정합니다. NBL 모듈에서는 의심스러운 노이즈 레이블을 단순히 삭제하는 대신 강력한 손실 함수를 사용하여 이러한 인스턴스를 활용하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 CLCS 방법은 다양한 노이즈 유형을 가진 두 개의 실제 의료 이미지 데이터 세트에서 유의미한 성능 향상을 보여줍니다. 연구 결과에 따르면, 기존 방법들에 비해 CLCS가 픽셀 단위의 노이즈가 존재하는 상황에서도 정확한 분할 성능을 유지하며, 소수 클래스의 데이터 활용도를 높여 클래스 불균형 문제를 효과적으로 완화하는 것으로 나타났습니다. 이를 통해 의료 이미지 분할 분야에서 CLCS의 적용 가능성을 강조합니다.



### MapGS: Generalizable Pretraining and Data Augmentation for Online Mapping via Novel View Synthesis (https://arxiv.org/abs/2501.06660)
- **What's New**: 이 연구에서는 자율주행 차량의 온라인 매핑에서 센서 구성 일반화를 다루기 위해 가우시안 스플래팅(Gaussian splatting) 기술을 활용하는 새로운 프레임워크를 제안합니다. 최신의 뷰 합성 방법을 통해 센서 구성의 간극을 해소하려는 접근 방식을 소개하며, 이는 다양한 차량 유형에서의 성능 저하 문제를 해결하는 데 기여할 것으로 기대됩니다. 제안된 방법은 nuScenes와 Argoverse 2 데이터셋에서 18%의 성능 개선을 달성하고, 훈련 데이터의 효율성을 크게 향상시킵니다.

- **Technical Details**: 이 프레임워크는 가우시안 스플래팅을 사용하여 정적 및 동적 장면을 재구성하고 새로운 카메라 데이터를 생성합니다. 기존 센서 데이터로부터 장면을 재구성한 후, 목표 센서 구성에서의 이미지와 레이블을 렌더링하여 온라인 매핑 모델을 훈련시킵니다. 이는 데이터 재사용을 가능하게 하며, 데이터 라벨링에 소요되는 시간을 줄이는 동시에, 3D 온라인 매핑 알고리즘의 일반화 성능을 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 접근 방식은 nuAV2라는 새로운 데이터셋을 구축하여 18%의 성능 향상을 보여줍니다. 또한, 더 빠른 수렴 및 효율적인 훈련을 통해 단 25%의 원래 훈련 데이터를 사용하여 최첨단 결과를 초과하는 성과를 달성했습니다. 이로 인해 데이터 재사용이 가능해지고, 라벨링 과정의 수고를 덜 수 있는 장점이 있습니다.



### Parking Space Detection in the City of Granada (https://arxiv.org/abs/2501.06651)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구는 그라나다 시의 도심 주차 공간 탐지 문제를 다루고 있습니다. 위성 이미지를 활용하여 정차된 차량, 이동 중인 차량 및 도로를 정확하게 식별하는 의미 분할(semantic segmentation) 기술을 개발했습니다. 특히, 그라나다를 위한 독자적인 데이터셋을 생성하여 신경망 모델을 훈련시키는데 기여했습니다. 본 연구에서는 다양한 모델을 비교하고 최적화하여, 도시 이미지에 대한 의미 분할의 효율성을 보여줍니다.

- **Technical Details**: 연구에서는 Fully Convolutional Networks (FCNs), Pyramid Networks 및 Dilated Convolutions을 활용하여 도심 환경에서의 의미 분할을 수행합니다. FCNs 아키텍처는 다운샘플링 및 업샘플링 경로로 이루어져 있으며, 다중 추상화 수준에서의 특징 학습을 가능하게 합니다. Pyramid Networks는 다양한 스케일에서 객체를捕捉하기 위한 다중 스케일 접근 방식을 구현하고, Dilated Convolutions는 해상도를 유지하면서 필터의 수용 범위를 확장하여 높은 해상도 피쳐 맵을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DeepLabV3+ 모델이 가장 뛰어난 성능을 보였습니다. 본 연구에서는 Foreground Accuracy, Dice Coefficient 및 Jaccard Index 등의 성능 지표를 활용하여 모델을 평가했습니다. 또한, 우리의 연구는 도시 계획 및 교통 관리 분야에 기여하며, 고급 이미지 처리 기술을 통해 주차 공간의 효율적인 활용을 위한 통찰력을 제공합니다.



### A Comparative Performance Analysis of Classification and Segmentation Models on Bangladeshi Pothole Datas (https://arxiv.org/abs/2501.06602)
Comments:
          8 Tables, 7 Figures

- **What's New**: 이 연구는 방글라데시의 도로 움푹 파인 곳 데이터를 기반으로 한 고유한 포트홀(pothole) 데이터셋을 제안하고 이를 통해 여러 머신러닝 모델의 성능을 분석합니다. 총 824개의 이미지를 포함하는 이 데이터셋은 다카와 보고라에서 수집되었으며, 기존 문헌에서 사용된 데이터셋과 비교해 매우 경쟁력 있는 성능을 보여주었습니다. 연구팀은 경량 모델(lightweight models)의 성능을 강조하여, 전통적인 무거운 모델(heavyweight models) 대비 실시간 탐지 작업에 적합한 대안을 제공합니다.

- **Technical Details**: 이 연구에서는 9개의 분류 모델(classification models)과 4개의 세분화 모델(segmentation models)을 테스트하였습니다. 분류 모델로는 CCT, CNN, INN, Swin Transformer, ConvMixer, VGG16, ResNet50, DenseNet201, Xception이 사용되었으며, U-Net, ResU-Net, U-Net++, Attention U-Net이 세분화 모델로 활용되었습니다. 데이터 증강(data augmentation) 기술을 활용하여 모델의 성능을 향상시켰으며, 경량 모델들은 빠른 예측 시간과 낮은 계산 요구량으로 주목받았습니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터셋은 기존의 유사한 분류 모델과 동등하거나 그 이상의 성능을 나타내며, 정확도 99% 이상과 f1-score 99%를 달성했습니다. 세분화 작업에서도 모델 다이스 유사도 계수(Dice Similarity Coefficient) 67.54%와 IoU 점수 59.39%를 기록했습니다. 데이터 증강을 통해 모든 테스트 모델의 성능이 개선된 것으로 나타났으며, 이는 머신러닝 모델의 성능 향상에 중요한 요소로 작용했습니다.



### Exploring Pose-Based Anomaly Detection for Retail Security: A Real-World Shoplifting Dataset and Benchmark (https://arxiv.org/abs/2501.06591)
- **What's New**: 이번 연구는 소매업체의 심각한 문제인 도둑질(shoplifting) 탐지를 위한 새로운 접근 방식을 제시합니다. PoseLift라는 데이터셋은 실제 환경에서 수집된 인체 포즈 데이터로, 도둑질 행위를 탐지하기 위한 프라이버시 보호 기능을 갖추고 있습니다. 연구자들은 이 데이터를 활용해 도둑질 탐지를 이상 탐지(anomaly detection) 문제로 재구성하였습니다.

- **Technical Details**: PoseLift 데이터셋은 실제 소매 환경에서 수집된 CCTV 영상을 사용하여 구축되었습니다. 이 데이터셋은 1920x1080 해상도와 초당 15프레임으로 촬영된 다양한 각도의 비디오를 포함하고 있으며, 정상 쇼핑 행동과 도둑질 행동을 동시 수집하여 프라이버시를 보호합니다. 또한, 연구에서는 최신의 이상 탐지 모델을 벤치마킹하여 다양한 성능 지표를 통해 효과성을 검증하였습니다.

- **Performance Highlights**: PoseLift 데이터셋을 활용한 연구 결과는 도둑질 탐지 정확도가 매우 높음을 보여주며, 전통적인 방법의 프라이버시 및 편향 문제를 효과적으로 해결할 수 있는 가능성을 제시합니다. 연구에서 평가된 포즈 기반 이상 탐지 모델은 AUC-ROC, AUC-PR 및 EER 지표로 성능을 측정하여 도둑질 탐지 시스템의 발전에 기여할 예정입니다. 이 데이터셋은 공공에 제공되어 연구자들이 윤리적으로 컴퓨터 비전을 발전시킬 수 있는 귀중한 도구가 될 것입니다.



### VASparse: Towards Efficient Visual Hallucination Mitigation for Large Vision-Language Model via Visual-Aware Sparsification (https://arxiv.org/abs/2501.06553)
- **What's New**: 이번 연구는 Visual-Aware Sparsification(VASparse)이라는 새로운 디코딩 알고리즘을 제안하여, Large Vision-Language Model(LVLM)에서 발생하는 비주얼 환각(visual hallucinations, VH)을 경감시키는 데 초점을 맞추고 있습니다. VASparse는 감시 시점에서 효율성과 신뢰성을 동시 수립하기 위한 토큰 희소성(token sparsity)을 기반으로 하여 설계되었습니다. 기존의 후처리 및 복구 과정의 불필요한 속도 저하 문제를 해결하기 위한 혁신적인 접근을 보여줍니다.

- **Technical Details**: VASparse는 LVLM의 희소한 주의를 활용하여, 시각 정보를 보존하면서 중복 토큰을 줄이는 시각 인지 토큰 선택 전략을 구현합니다. 또한, 시각적으로 민감한 디코딩 접근법을 통해 환각된 출력을 재조정하는 방안을 제공합니다. 이 과정에서 계산된 주의 점수(attention scores)를 통해 언어 바이어스를 가진 토큰에 대한 집중을 억제하여 전반적인 성능 향상에 기여합니다.

- **Performance Highlights**: VASparse는 기존 VH 경감 방법 대비 우수한 성능과 경쟁력 있는 디코딩 속도를 유지하며, 총 4개의 인기 있는 벤치마크에서 실험을 통해 효과가 입증되었습니다. 특히, VASparse는 HALC보다도 최대 12.9배 빠른 성능을 보여, VH 경감에 있어 최첨단 성과를 달성하였습니다. 본 연구는 이전의 방법들과 비교하여 높은 효율성과 출력의 신뢰성을 동시에 확보한 것을 강조합니다.



### CoreNet: Conflict Resolution Network for Point-Pixel Misalignment and Sub-Task Suppression of 3D LiDAR-Camera Object Detection (https://arxiv.org/abs/2501.06550)
Comments:
          Accepted by Information Fusion 2025

- **What's New**: 본 논문에서는 3D 객체 탐지 성능을 향상시키기 위한 새로운 방법인 Conflict Resolution Network (CoreNet)를 제안합니다. 기존 방법들이 간과했던 두 가지 주요 문제인 점-픽셀 불일치(point-pixel misalignment)와 서브 태스크 억제(sub-task suppression)에 초점을 맞추고 있습니다. 이를 해결하기 위해 이중 스트림 변환 모듈과 태스크별 예측기를 도입하여 더욱 정교한 정보 표현을 가능하게 합니다.

- **Technical Details**: CoreNet은 ray-based 및 point-based 2D-to-BEV 변환을 포함하는 이중 스트림 변환 모듈을 사용하여 점-픽셀 불일치를 해결합니다. 태스크별 예측기는 클래스 및 Bbox 특화된 쿼리를 활용하여 두 가지 서브 태스크를 다루며, 각 태스크에 대해 특화된 피처를 조합하여 상호 억제를 최소화합니다. 이 기술을 활용하여 모델이 다양한 서브 태스크에서 필요한 정보를 적절하게 선택할 수 있게 합니다.

- **Performance Highlights**: CoreNet는 대규모 nuScenes 데이터셋에서 75.6%의 NDS와 73.3%의 mAP를 기록하며, 최신의 다른 모델들과 비교해 우수한 성능을 보여주었습니다. 여러 구성 요소의 효과를 입증하기 위한 방대한 ablation 연구를 통해 각 기술의 유효성을 검증하였습니다. 이로써 제안된 CoreNet 프레임워크는 LiDAR-카메라 기반 3D 객체 탐지에서 뛰어난 성능을 발휘함을 보여주었습니다.



### Natural Language Supervision for Low-light Image Enhancemen (https://arxiv.org/abs/2501.06546)
Comments:
          12 pages, 10 figures

- **What's New**: 이번 논문에서는 자연어 감독(Natural Language Supervision, NLS) 전략을 도입하여 저조도 이미지 향상(Low-Light Image Enhancement, LLIE)을 위한 새로운 접근 방식을 제안합니다. 이 방법은 텍스트와 대응되는 이미지의 특징 맵(feature maps)을 공동 학습하여 다양한 조명 상태에서 이미지를 설명하는 유연한 인터페이스를 제공하는 데 중점을 두고 있습니다. 또한, 텍스트의 지역(word)와 이미지의 해당 영역(image regions) 간의 연결을 이해하기 위해 텍스트 지침 조정 메커니즘(Textual Guidance Conditioning Mechanism, TCM)을 설계하여 방향성을 부여합니다.

- **Technical Details**: 저자들은 TCM을 통해 이미지 지역(image regions)과 문장 단어(sentence words) 간의 관계를 포괄적으로 캡처합니다. 이는 멀티 모달(multi-modal) 데이터 분포에서 발생하는 교육 문제를 해결하는 데 기여합니다. 또한, 정보 융합 주의 모듈(Information Fusion Attention, IFA)을 설계하여 다양한 수준의 이미지와 텍스트 정보에 대한 특징 식별 및 병합을 효과적으로 수행할 수 있도록 합니다. 이 두 가지 모듈을 결합하여 저조도 이미지 향상의 성능을 높이는 NaLSuper 네트워크를 구현하였습니다.

- **Performance Highlights**: NaLSuper는 4개의 벤치마크 데이터셋에서 폭넓은 실험을 통해 최근의 최첨단 방법들보다 더 우수한 성능을 보여주었습니다. 정량적 및 정성적 평가 모두에서, 제안된 접근 두 가지 방식(TCM 및 IFA)이 저조도 이미지 향상에 효과적임을 입증하며, 기존의 메트릭 지향 접근 방식이 비주얼 품질과 함께 만족스러운 성과를 달성할 수 있도록 하고 있습니다.



### CeViT: Copula-Enhanced Vision Transformer in multi-task learning and bi-group image covariates with an application to myopia screening (https://arxiv.org/abs/2501.06540)
- **What's New**: 이 논문은 고도근시(high-myopia) 선별을 위한 이미지 기반 딥러닝 방법을 제안합니다. 특히, 양쪽 눈의 초광각망막(UWF) 이미지를 통합하여, 두 눈의 상승적인 의존성을 모델링하는 새로운 접근법을 사용합니다. 제안된 CeViT 모델은 Vision Transformer(ViT)를 기반으로 하여, 기존 방법의 한계를 극복하고 높은 정확도로 고도근시 상태와 축 길이를 예측합니다.

- **Technical Details**: CeViT는 4차원 다중 응답 회귀(multi-response regression)로 구성된 분류-회귀 작업을 수행합니다. 이 모델은 공유하는 Transformer 인코더를 통해 두 눈의 공통 특징을 추출하며, 다층 퍼셉트론(MLP) 헤드를 통해 안구의 비대칭성을 모델링합니다. 또한, Copula 손실 함수(copula loss)를 통해 이미지를 기반으로 한 반응 간의 조건부 의존성을 설명합니다.

- **Performance Highlights**: Shanghai Eye & ENT 병원에서 수집한 주석이 달린 초광각망막 이미지 데이터셋에서 CeViT는 고도근시 분류와 양쪽 눈의 AL 예측에서 기본 모델보다 성능이 향상되었습니다. 제안된 모델은 기존의 CNN 기반 방법보다 더 세밀한 패턴을 인식하며, 더 높은 예측 능력을 유지할 수 있음을 보여주었습니다.



### DivTrackee versus DynTracker: Promoting Diversity in Anti-Facial Recognition against Dynamic FR Strategy (https://arxiv.org/abs/2501.06533)
- **What's New**: 이 논문은 기존의 얼굴 인식(FR) 모델 평가 방식이 실제 추적기의 능력을 충분히 반영하지 못한다고 주장합니다. 새로운 동적 얼굴 인식 전략인 \'DynTracker\'를 도입하여 추적기 모델의 갤러리 데이터베이스를 실시간으로 업데이트함으로써 기존의 안티 얼굴 인식(AFR) 보호 방안을 무력화시키는 방법을 보여줍니다. 또한, 적절한 다양성을 가진 이미지를 촉진하는 \'DivTrackee\'라는 새로운 AFR 보호 방법도 제안합니다.

- **Technical Details**: 논문에서는 얼굴 인식 모델의 세 가지 주요 구성요소인 전이 학습된 얼굴 특징 추출기, 알려진 얼굴의 갤러리 데이터베이스, 유사성 메트릭을 기반으로 한 쿼리 매칭 단계를 설명합니다. 또한, 동적 FR 전략인 DynTracker의 필요성을 강조하며, 상대적으로 정적인 기존의 AFR 방법들이 가진 제한적인 특징을 분석합니다. 이를 바탕으로, \'DivTrackee\'는 텍스트 기반 이미지 생성 프레임워크에 기반하여 더 다양한 AFR 방어를 구현합니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 DynTracker가 기존의 정적 AFR 방법들보다 훨씬 높은 추적 성공률을 보이며, 기존의 AFR 보호 방법들이 동적 FR 전략에 상당히 취약하다는 것을 보여줍니다. DivTrackee는 DynTracker의 추적 성공률을 낮추는데 있어 40% 이상의 보호 성공률을 달성하며, AFR로 보호된 이미지의 높은 시각적 품질을 유지합니다. 이 연구는 사용자의 얼굴 프라이버시를 보호하기 위한 더 효과적인 AFR 방법 개발의 초석이 될 것으로 기대합니다.



### Multi-View Factorizing and Disentangling: A Novel Framework for Incomplete Multi-View Multi-Label Classification (https://arxiv.org/abs/2501.06524)
- **What's New**: 최근 다중 뷰 다중 레이블 분류(MvMLC) 문제에 대한 관심이 급증하고 있습니다. 그러나 기존 방법들은 보통 모든 뷰와 레이블이 완전하다고 가정하고 있어 실제 상황과 부합하지 않는 경우가 많습니다. 본 논문은 누락된 뷰와 레이블이 있는 비완전한 다중 뷰 다중 레이블 분류(iMvMLC)를 해결하기 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 다중 뷰 표현을 뷰 일관성이 있는 요인(view-consistent factors)과 뷰 특이적인 요인(view-specific factors)으로 분해하여 별도로 학습합니다. 이를 위해 세 가지 하위 목표로 일관성 표현 학습을 혁신적으로 분해하였으며, 이는 뷰 간 공유 정보 추출, 내부 뷰 중복 제거, 그리고 작업 관련 정보 유지로 구성됩니다. 또한, 정보 이론에 기반한 마스크된 크로스 뷰 예측(MCP) 전략을 활용하여 협력적으로 고품질의 일관된 표현을 학습합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 기존의 최첨단 접근 방식들보다 우수한 성능을 발휘한다는 것을 입증하였습니다. 이러한 결과는 제안된 프레임워크가 다양한 다중 뷰 및 다중 레이블 데이터셋에 적응 가능하다는 점에서 중요한 기여를 합니다. 또한, 그래프 분리 손실(graph disentangling loss) 설계를 통해 뷰 일관형 및 뷰 특이형 표현 간의 중복을 완전히 줄이는 데 성공하였습니다.



### NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References (https://arxiv.org/abs/2501.06488)
- **What's New**: 이번 연구에서는 NVS-SQA라는 새로운 NSS(신경 합성 장면) 품질 평가 방법을 제안합니다. 이 방법은 인공 주석(human annotation)에 의존하지 않고, 자가 감독(self-supervision) 방식으로 품질 표현을 학습합니다. NVS-SQA는 히uristic(휴리스틱) 신호와 품질 점수를 학습 목표로 활용하여 비지도학습 상황에서도 유효한 품질 평가를 가능하게 합니다.

- **Technical Details**: NVS-SQA는 NSS 품질 평가를 위한 첫 번째 자가 감독 학습 프레임워크로, 비지도 데이터셋에서 품질 표현을 효과적으로 추출할 수 있도록 설계되었습니다. 이 프레임워크는 대조(pair) 준비 과정에서 NSS에 특화된 접근 방식을 채택하며, 통계 원리에 기초하여 품질 표현 학습을 위한 적응형 가중치 방법을 포함하고 있습니다. 마지막으로, 다양한 보기 수(view counts)와 공간 해상도(spatial resolutions)를 처리할 수 있는 심층 신경망(deep neural network)을 사용하여 프레임워크의 가치를 높였습니다.

- **Performance Highlights**: NVS-SQA는 기존의 17개 비리퍼런스(no-reference) 방법과 비교하여 평균적으로 109.5%의 SRCC(Spearman Rank Correlation Coefficient), 98.6%의 PLCC(Pearson Linear Correlation Coefficient), 91.5%의 KRCC(Kendall Rank Correlation Coefficient)에서 우수한 성능을 보였습니다. 또한, 16개의 전체 참고(full-reference) 방법을 모든 평가 지표에서 능가하여 NSS 품질 평가에서 혁신적인 기여를 하고 있음을 입증합니다.



### Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation (https://arxiv.org/abs/2501.06481)
- **What's New**: 본 논문에서는 Focus-N-Fix라는 지역 인식(Region-aware) 보상 미세 조정(fine-tuning) 기법을 제안합니다. 이 방법은 이미지에서 문제가 발생한 특정 영역만을 수정하며, 이전 연구들과는 달리 전체 이미지를 글로벌 최적화하는 대신 국소(sub-local) 최적화를 지향합니다. 이를 통해 생성된 이미지는 원래 모델과 구조적으로 유사하면서도 안전성, 그럴듯함 등을 개선할 수 있습니다.

- **Technical Details**: Focus-N-Fix는 지역 정보를 활용하여 생성된 이미지의 질을 개선하는 미세 조정 기법입니다. 기존의 보상 모델에서 발생할 수 있는 예기치 않은 행동 변화를 줄이고, 문제가 되는 영역만을 목표로 조정하여 다른 영역은 가능한 한 영향을 최소화합니다. 실험 결과 이 방법은 사회적 안전성이나 이미지 품질 같은 여러 품질 기준에서 보이는 문제점을 효과적으로 줄이며, 구체적인 지역의 정보를 수집하여 개선합니다.

- **Performance Highlights**: Focus-N-Fix는 다양한 이미지 품질 기준에서 의미 있는 개선을 보이며, 다른 품질 기준에 미치는 영향은 미미하거나 전혀 없습니다. 논문에서는 fine-tuning 전후의 이미지를 비교 분석하여 이 방법의 효과성 및 신뢰성을 시각적으로 평가합니다. 전반적으로 특정 문제 영역에서의 조정을 통해 전체 모델 품질을 유지함과 동시에 원하는 품질 개선을 이룰 수 있음을 입증합니다.



### Flash Window Attention: speedup the attention computation for Swin Transformer (https://arxiv.org/abs/2501.06480)
- **What's New**: 본 보고서에서는 Swin Transformer의 창문 주의 메커니즘을 향상시키기 위해 Flash Window Attention이라는 최적화된 솔루션을 제안합니다. 이 방법은 주의 계산 효율성을 최대 300% 향상시키고, 전체적인 실행 시간 효율성을 최대 30% 개선합니다. Flash Attention을 기반으로 개량된 이 접근법은 짧은 시퀀스를 처리하는 데 특화되어 설계되었습니다.

- **Technical Details**: Swin Transformer의 𝐐/𝐊/𝐕는 H×W×C 형태의 텐서로 표현되며, 이는 𝐐, 𝐊, 𝐕 매트릭스를 피처 차원에 따라 청크로 나눈 후 주의 매트릭스를 쌓아가는 방식으로 연산됩니다. 이 과정에서 외부 메모리에 대한 접근을 최소화하고, 온칩 SRAM에 매트릭스를 저장함으로써 고속 처리가 가능합니다. 일반적인 설정 하에서 피크 메모리 사용량은 24KB로, 현대 GPU의 L1 캐시 용량에 적합합니다.

- **Performance Highlights**: Flash Window Attention의 구현 결과, 주의 메커니즘의 계산 효율성이 크게 향상되었습니다. 이 접근법은 Swin Transformer와 같은 짧은 시퀀스에 적합하며, 실제로 L=64 및 C/r=16 설정에서 24KB의 피크 메모리 사용량을 기록했습니다. 전체적인 성능 개선은 주의 메커니즘을 최적화함으로써 눈에 띄게 나타나, 딥 러닝 모델의 효율성을 크게 높일 것으로 기대됩니다.



### Enhancing Multi-Modal Video Sentiment Classification Through Semi-Supervised Clustering (https://arxiv.org/abs/2501.06475)
- **What's New**: 본 연구는 비디오 감정 분류 성능을 향상시키기 위해 비디오, 텍스트 및 음향 특징을 이용한 다중 모달 접근 방식을 제안합니다. 기존의 감정 분석 모델들이 대량의 라벨링된 데이터에 의존하는 단점을 극복하기 위해, 클러스터링 기반의 반지도 학습 방법을 개발하여 의미 있는 데이터 표현을 추출합니다. 이는 감정 인식을 지원하는 모델의 기반 구조를 학습하는 데 필요한 라벨된 데이터의 양을 줄여줍니다.

- **Technical Details**: 연구 방법론은 두 가지 단계로 나뉘며, 첫 번째 단계에서는 기존의 라벨링된 데이터셋의 작은 부분집합을 사용하여 초기 감정 분류를 수행하고, 여러 기존 모델과 비교합니다. 두 번째 단계에서는 Deep Embedded Clustering (DEC) 기반의 반지도 클러스터링 네트워크를 구현하여 특징 표현을 개선합니다. DEC 네트워크는 전체 데이터셋을 미리 학습하여 복잡한 다중 모달 관계를 캡처한 후, 이를 통해 초기화된 가중치를 사용하여 분류 성능을 최적화합니다.

- **Performance Highlights**: CMU-MOSI 데이터셋을 통한 평가 결과, 제안된 방법이 기존의 지도 학습 기준선과 비교하여 효과적인 성능 향상을 보여주었습니다. 특히, 라벨된 데이터가 제한적인 상황에서도 모델이 높은 정확도와 F1 점수를 기록하였으며, 이러한 접근 방식이 다중 모달 감정 분석의 최신 한계를 넘는 데 기여할 가능성을 보여주었습니다.



### YO-CSA-T: A Real-time Badminton Tracking System Utilizing YOLO Based on Contextual and Spatial Attention (https://arxiv.org/abs/2501.06472)
Comments:
          8 pages,14 figures

- **What's New**: 본 논문에서는 배드민턴 랠리 로봇의 3D 경로 추적을 위한 YO-CSA 탐지 네트워크를 제안합니다. 이 네트워크는 YOLOv8s 모델의 백본(backbone), 넥(neck), 헤드(head)를 최적화하고 재구성하여 전역(global) 및 지역(local) 특징 추출 능력을 향상시킵니다. 또한, 2D 검출을 넘어 3D 공 비행 경로 예측 및 보상을 통합하여 실시간 시스템을 구축하였습니다.

- **Technical Details**: YO-CSA 네트워크는 컨텍스트(contextual) 및 스페이셜(spatial) 주의 메커니즘을 포함하여 더욱 정교한 특징 통합을 구현합니다. 이 시스템은 스테레오 비전(stereo vision)을 사용하여 2D 좌표 시퀀스를 3D 공간으로 변환하고, 과거 정보를 바탕으로 미래의 3D 좌표를 예측합니다. 또한, 결측된 중간 프레임을 보완하는 컴펜세이션 모듈(compensation module)을 포함하여 보다 완전한 경로를 제공합니다.

- **Performance Highlights**: 실험 결과, YO-CSA는 90.43% mAP@0.75의 높은 정확도를 기록하며, YOLOv8s 및 YOLO11s를 초과하였습니다. 제안된 시스템은 12개의 테스트 시퀀스에서 130 fps 이상의 속도를 유지하며 뛰어난 성능을 보여줍니다. 이러한 결과는 인간-로봇 경쟁을 위한 실제 및 신속한 배드민턴 경로 추적의 가능성을 나타냅니다.



### SP-SLAM: Neural Real-Time Dense SLAM With Scene Priors (https://arxiv.org/abs/2501.06469)
- **What's New**: 본 논문에서는 SP-SLAM이라는 새로운 신경 RGB-D SLAM 시스템을 소개합니다. 이 시스템은 실시간으로 추적과 맵핑을 수행하며, 깊이 이미지를 계산하고 표면 근처에서 희소한 복셀 인코딩 장면 사전(Sparse Voxel-Encoded Scene Priors)을 설정해 모델의 빠른 수렴을 달성합니다. SP-SLAM은 기초적인 구조적 특징을 캡처하는 희소 볼륨을 초기화하는 데 기존 깊이 정보를 활용하여 모델의 초기 이해를 돕습니다.

- **Technical Details**: SP-SLAM은 깊이 맵을 3D 포인트 클라우드로 역 투영(back-projection)하여 고정 길이 임베딩 벡터로 각 포인트의 Signed Distance Field (SDF) 사전 정보를 인코딩합니다. 값과 메모리 사용 간의 균형을 유지하며 장면 외관 정보를 삼축 평면(tri-planes)에 저장하여 텍스처 매핑을 달성합니다. 또한, 키프레임 개념을 폐기하고 새로운 매핑 최적화 전략을 도입하여 모든 입력 프레임의 자세를 지속적으로 최적화합니다.

- **Performance Highlights**: SP-SLAM은 Replica, ScanNet, TUM RGB-D, Synthetic RGB-D, 7-Scenes와 같은 다섯 가지 벤치마크 데이터셋에서 평가 결과를 보여줍니다. 기존 방법들에 비해 뛰어난 트래킹 정확도와 재구성 품질을 달성하였으며, 실시간 성능이 현저히 향상되었습니다. 따라서 SP-SLAM은 신뢰할 수 있는 카메라 추적과 고해상도 표면 재구성을 가능하게 합니다.



### Discovering an Image-Adaptive Coordinate System for Photography Processing (https://arxiv.org/abs/2501.06448)
Comments:
          BMVC 2024

- **What's New**: 이 논문에서 제안하는 IAC(이미지 적응형 카르테시안 좌표계) 알고리즘은 RGB 색 공간 내에서 곡선 작업을 수행하기 전에 이미지에 적응하는 새로운 좌표계를 학습합니다. 이는 이미지 조정 과정에서 기존의 곡선 및 LUT 기반 방법들과의 차별점으로, 공간적 복잡성을 최소화하고 성능을 극대화합니다. 또한 경량화된 설계를 통해 모바일 및 엣지 디바이스에 적합하여 다양한 이미지 조정 작업에서 최신 성능을 보여줍니다.

- **Technical Details**: 제안된 IAC 방법은 이미지를 선호하는 좌표계로 투영한 다음, 해당 공간에서 곡선 조정을 수행하고 RGB 공간으로 변환하는 방식으로 작동합니다. 이를 통해 근본적으로 이미지 특성과 변동에 적응하는 동적 좌표계를 만드는 것이 가능합니다. IAC는 공간적 복잡성이 낮고(𝒪(n)), 파라미터 수가 적어 경량 설계를 유지합니다.

- **Performance Highlights**: IAC는 사진 보정, 노출 수정, 화이트 밸런스 조정 작업 등 다양한 사진 처리 작업에서 최신 성능을 달성하였습니다. 실험 결과, 기존의 곡선 및 LUT 기반 방법보다 더 빠르고 효율적으로 작동하며, 조정된 색상 품질에서도 뛰어난 결과를 보여줍니다. 따라서 이 방법은 전문 사진가뿐만 아니라 일반 사용자에게도 큰 가치가 있을 것으로 기대됩니다.



### CPDR: Towards Highly-Efficient Salient Object Detection via Crossed Post-decoder Refinemen (https://arxiv.org/abs/2501.06441)
Comments:
          14 pages

- **What's New**: 이 논문에서는 Salient Object Detection (SOD)의 성능을 높이기 위해 Crossed Post-decoder Refinement (CPDR)라는 경량 아키텍처를 제안합니다. CPDR은 전통적인 FPN이나 U-Net 아키텍처의 기능을 향상시키기 위해 채널 주의 메커니즘을 사용하여 저수준 특성을 정제하며, 이 정보를 고수준 특성과 교차하여 통합합니다. 이로 인해 전반적인 기능 개선과 효율성을 달성할 수 있습니다.

- **Technical Details**: 제안된 CPDR 아키텍처는 Attention Down Sample Fusion (ADF)와 Attention Up Sample Fusion (AUF) 두 가지 모듈을 포함합니다. ADF는 채널 주의 메커니즘을 활용하여 저수준 특성을 개선하고, AUF는 공간적 주의를 통해 고수준 정보를 가이드합니다. 이러한 구조는 Dual Attention Cross Fusion (DACF)와 결합되어 파라미터 수를 줄이면서도 성능을 유지합니다.

- **Performance Highlights**: 실험 결과, CPDR 모델은 5개의 벤치마크 데이터셋에서 이전의 최첨단 방법들과 비교할 때 더 우수한 성능을 보였습니다. 이 모델은 1.66M의 파라미터를 사용하여 높은 정확도를 유지합니다. 이는 높은 계산 복잡성을 요구하지 않으면서도 적용 가능한 SOD 기술로서의 가능성을 시사합니다.



### UCloudNet: A Residual U-Net with Deep Supervision for Cloud Image Segmentation (https://arxiv.org/abs/2501.06440)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문에서는 클라우드 세그멘테이션을 위한 잔여 U-Net 모델인 UCloudNet을 소개합니다. 이 모델은 깊은 감독(deep supervision) 방식으로 설계되어 이전 방법보다 더 나은 정확도를 제공하며, 훈련 소모를 줄여줍니다. UCloudNet은 특징 추출 능력을 향상시키기 위해 인코더의 잔여 연결(residual connection)을 활용하고 있습니다.

- **Technical Details**: UCloudNet 아키텍처는 원래의 U-Net 구조를 기반으로 하며, 각 단계에서 채널 연결(channel concatenation)을 포함하는 일련의 디코더와 인코더로 구성되어 있습니다. 이 모델은 ResNet에서 영감을 받아 각 컨볼루션 블록에 잔여 연결을 추가하여 깊은 레이어의 학습을 돕습니다. 또한, 학습 과정에서 깊은 감독 방식을 사용해 훈련 과정을 지원하며, 이로써 정규화 성능이 향상됩니다.

- **Performance Highlights**: 실험 결과, 제안된 UCloudNet은 적은 훈련 시간과 반복으로도 이전 방법들보다 더 나은 성능을 달성할 수 있음을 입증하였습니다. 다양한 구성에서 실험을 진행하였고, 특히 SWINySEG 데이터셋을 활용하여 낮과 밤의 클라우드 이미지를 포함한 결과가 도출되었습니다. 최종적으로 Adam 옵티마이저와 학습률 조정을 통해 훈련이 수행되었습니다.



### Qffusion: Controllable Portrait Video Editing via Quadrant-Grid Attention Learning (https://arxiv.org/abs/2501.06438)
Comments:
          15 pages

- **What's New**: 이번 연구에서 제안한 Qffusion은 포트레이트 비디오 편집을 위한 이중 프레임 가이드 프레임워크입니다. 이 프레임워크는 두 개의 정지 이미지를 활용하여 효과적인 비디오 편집을 가능하게 하며, 수정된 시작 및 종료 프레임을 참조로 사용합니다. 특히, 새로운 'animation for editing' 디자인 원칙을 기반으로 하여 안정적인 비디오 편집을 제공합니다.

- **Technical Details**: Qffusion은 Quadrant-grid Arrangement (QGA) 방식을 통해 두 개의 참조 이미지와 네 가지 얼굴 조건의 잠재 코드들을 네 그리드 형식으로 배열합니다. 이 시스템은 자기 주의(Self-Attention) 메커니즘을 활용하여 시간적 정보와 외관 정보를 학습합니다. 또한, Quadrant-grid Propagation (QGP) 추론 전략을 도입하여 비디오 길이에 제한 없이 안정적인 생성을 가능합니다.

- **Performance Highlights**: Qffusion은 포트레이트 비디오 편집 분야에서 기존의 최신 기술을 지속적으로 초월하는 성능을 보입니다. 실험 결과에 따르면, 이 시스템은 매우 섬세한 부분 편집이 가능하며, 예를 들어 선글라스 추가, 나이 조정, 머리 스타일 변경 등의 작업에서 뛰어난 결과를 나타냅니다. 더불어 Qffusion은 비디오 애니메이션 프레임워크로서 다양한 응용 프로그램에서 활용될 수 있습니다.



### Aug3D: Augmenting large scale outdoor datasets for Generalizable Novel View Synthesis (https://arxiv.org/abs/2501.06431)
Comments:
          IROS 2024 Workshop, 9 Pages, 7 Figures

- **What's New**: 최근 Photorealistic Novel View Synthesis (NVS)에 대한 관심이 높아지고 있으며, 이러한 기술이 실내에만 제한되어 있는 문제를 해결하기 위해 PixelNeRF라는 새로운 feed-forward NVS 모델을 UrbanScene3D 데이터셋에서 훈련했습니다. 본 연구에서는 Aug3D라는 증강(augmentation) 기법을 도입하여 기존의 구조 기반 방법을 사용하여 복원된 장면을 활용함으로써 모델 학습 향상을 목표로 하고 있습니다.

- **Technical Details**: Aug3D는 구조로부터의 모션(SfM) 기법을 사용하여 NVS의 효과를 높이기 위해 대규모 야외 장면 데이터셋을 학습에 적합하게 구성하는 샘플링 전략입니다. 각 클러스터에 대해 20개의 뷰에서 10개로 뷰 수를 줄였더니 PSNR이 10% 개선되었지만, 성능은 여전히 최적화되지 않았습니다. Aug3D는 새롭게 생성된 뷰를 원본 데이터셋과 결합하여 모델의 새로운 뷰 예측 능력을 향상시키는 데 효과적이라는 점을 보여주었습니다.

- **Performance Highlights**: Aug3D의 도입은 기존 야외 데이터셋에 대한 NVS 모델의 학습을 더욱 원활하게 해줍니다. 실험 결과, 가용 뷰 수를 줄였을 때 PSNR이 개선되는 성과를 지니며, 일반화 가능한 NVS 모델이 기존의 작은 실내 환경을 넘어 대규모 야외 환경에도 적용될 수 있음을 입증했습니다. 앞으로도 이러한 접근 방식이 도시 규모의 데이터를 효율적으로 활용하여 광범위한 애플리케이션에 기여할 것입니다.



### Open Eyes, Then Reason: Fine-grained Visual Mathematical Understanding in MLLMs (https://arxiv.org/abs/2501.06430)
- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(MLLMs)의 수학 문제 해결 능력의 한계를 지적하고, 특히 기하학적 프리미티브(geometric primitives)를 이해하는 데 필요한 정밀한 시각적 인식이 불충분하다는 점을 강조합니다. 이에 따라 우리는 SVE-Math(Selective Vision-Enhanced Mathematical MLLM)라는 새로운 접근 방식을 제안하여, 시각적 요소의 기하학적 기초를 인식하고 언어 모델의 추론 필요에 맞춘 정확한 시각적 프롬프트를 생성하는 방법을 모색합니다.

- **Technical Details**: SVE-Math는 기하학적으로 인식된 비전 인코더와 피처 라우터(feature router)를 통합하여 시각적 기능 맵의 기여도를 동적으로 조정합니다. GeoGLIP(Geometric-Grounded Language-Image Pre-training)라는 보조 인코더는 기하학적 프리미티브를 인식하도록 설계되었으며, 이를 통해 모델은 수학 문제 해결을 위한 필수 시각 정보를 효과적으로 활용할 수 있습니다. 기하학적으로 유의미한 기능 단계를 캡처하는 글로벌 피라미드 기능 맵을 사용하며, 이 피처 라우터는 시각적 소프트 프롬프트를 제공하여 문제 해결의 정확성을 향상시킵니다.

- **Performance Highlights**: SVE-Math는 15%가량 MathVerse에서 다른 7B 모델들을 초과 성능을 보여주었고, MathVista에서도 GPT-4V와 호환됩니다. 상대적으로 작은 데이터셋에 대해 훈련된 SVE-Math-7B는 GeoQA에서 더 큰 데이터셋으로 훈련된 모델들과 경쟁력 있는 성능을 기록했습니다. 이 연구 결과는 MLLMs에 정밀한 시각적 이해를 통합하는 것이 중요하다는 것을 강조하며, 향후 연구의 유망한 방향성을 제시합니다.



### FocusDD: Real-World Scene Infusion for Robust Dataset Distillation (https://arxiv.org/abs/2501.06405)
- **What's New**: 이번 논문에서 제안하는 Focused Dataset Distillation (FocusDD)은 기존의 데이터 세트 증류 방법의 한계를 극복하기 위해 새로운 접근 방식을 도입합니다. FocusDD는 고해상도 및 대규모 데이터 세트에서 데이터의 핵심 정보를 추출하여 현실적이고 다양한 증류 이미지를 생성함으로써, 다양한 네트워크 아키텍처에서 일반화 능력을 보장합니다. 이 방법은 또한 객체 탐지와 같은 다양한 다운스트림 작업에 적합하게 설계되었습니다.

- **Technical Details**: FocusDD는 두 가지 주요 단계로 구성됩니다: 정보 추출(information extraction)과 이미지 재구성(image reconstruction)입니다. 정보 추출 단계에서는 사전 훈련된 Vision Transformer (ViT)를 사용하여 주요 이미지 패치를 선택하고, 재구성 단계에서는 원본 이미지의 다운샘플링된 버전을 결합하여 최종 증류 이미지를 생성합니다. 이를 통해 데이터 세트의 다양성과 현실성을 유지하고, 모델의 일반화 능력을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, FocusDD는 ImageNet-1K 데이터 세트에서 ResNet50과 MobileNet-v2 모델이 각각 71.0%와 62.6%의 검증 정확도를 기록하여 기존 최첨단 방법보다 각각 2.8%와 4.7% 향상됨을 보여줍니다. 또한 COCO2017 데이터 세트에서 YOLOv11n과 YOLOv11s는 각각 24.4%와 32.1%의 평균 정밀도(mAP)를 달성하여 객체 탐지 작업에서의 유효성을 입증하였습니다.



### Has an AI model been trained on your images? (https://arxiv.org/abs/2501.06399)
- **What's New**: 이 논문에서는 생성 AI 모델이 특정 이미지 또는 이미지 집합으로 훈련되었는지를 판단할 수 있는 새로운 방법을 제시합니다. 이 방법은 컴퓨터 자원을 효율적으로 소모하며, 모델 아키텍처나 가중치에 대한 명시적인 지식이 필요하지 않습니다(black-box membership inference). 이는 기존 모델의 감사( auditing )와 더 공정한 생성 AI 모델의 발전 및 배포에 중요한 역할을 할 것으로 예상됩니다.

- **Technical Details**: 연구에서 활용된 기법은 대규모 웹 스크래핑을 통해 수집된 방대한 데이터셋을 기반으로 하여, 이미지의 내용을 기반으로 in-training(훈련)에 사용된 이미지와 out-of-training(훈련에서 제외된) 이미지를 비교합니다. 연구에서는 Stable Diffusion(스테이블 디퓨전), Midjourney(미드저니), DALL-E(달리)와 같은 다양한 생성 모델에 대해 이 방법을 적용했습니다. 각 이미지 쌍은 콘텐츠의 의미 면에서 유사성을 가지도록 구성되며, 이를 통해 이미지 간의 관측된 차이가 의미적 차이에 의한 것이 아님을 보장합니다.

- **Performance Highlights**: 제안된 방법은 강력한 생성 AI 모델의 훈련 데이터셋에 대한 투명성을 높이고, 생성 모델이 특정 콘텐츠를 사용하여 훈련되었는지 여부를 판단할 수 있게 해줍니다. 이는 특히 저작권 및 공정 사용 문제에 대한 논의가 있을 때 중요하게 여겨지며, AI 모델의 윤리적 개발과 관련된 중요한 기준을 제시합니다. 기술적으로, 이 방법은 계산 효율성이 뛰어나고 다양한 모델 아키텍처에 일반화될 수 있는 가능성을 가지고 있습니다.



### Mix-QViT: Mixed-Precision Vision Transformer Quantization Driven by Layer Importance and Quantization Sensitivity (https://arxiv.org/abs/2501.06357)
Comments:
          Journal, 12 pages, 7 figures

- **What's New**: 본 논문에서는 Mix-QViT라는 MPQ(모델 혼합 정밀도) 프레임워크를 제안하여, 각 레이어에 비트 폭을 시스템적으로 할당합니다. 이 과정은 레이어의 중요도와 양자화 민감성을 두 가지 기준으로 분석하여 진행됩니다.

- **Technical Details**: Mix-QViT는 Layer-wise Relevance Propagation(LRP)을 사용하여 레이어의 최종 분류 기여도를 평가하고, 다양한 정밀도 수준에서의 성능 영향을 분석하여 양자화 민감성을 결정합니다. 이 두 가지 지표를 통합하여 Integer Quadratic Problem(IQP)로 최적의 비트 폭 할당을 수립하는 방식을 채택했습니다.

- **Performance Highlights**: Mix-QViT는 ViT, DeiT 및 Swin Transformer 모델에서 다양한 데이터셋에 대해 적용하여 실험 결과, 고정 비트 및 혼합 비트 방식 모두 기존 기법을 초과하는 성능을 기록했습니다. 특히, 3, 4 및 6 비트 정밀도에서 우수한 성능을 달성하며, 2 비트 혼합 정밀도의 양자화 인식 훈련에서도 뛰어난 결과를 보였습니다.



### MEt3R: Measuring Multi-View Consistency in Generated Images (https://arxiv.org/abs/2501.06336)
Comments:
          Project website: this https URL

- **What's New**: 새로운 메트릭 MEt3R가 제안되었으며, 이는 생성된 이미지의 다중 시점 일관성을 측정하는 데 중점을 두고 있습니다. 기존의 복원 메트릭들은 생성된 출력의 품질을 정확하게 측정할 수 없기 때문에, 비효율적인 대안이 필요했습니다. MEt3R는 특정 장면에 국한되지 않고, 다양한 조명 조건에서도 일관성을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: MEt3R는 DUSt3R를 활용하여 이미지 쌍으로부터 밀접한 3D 재구성을 수행하고, 영상의 특징 맵을 비교하여 유사성을 평가합니다. DINO와 FeatUp을 특징 추출기로 사용하여 조명과 같은 시점 관련 효과에 강건한 특징을 얻습니다. 이 메트릭은 이미지 품질 메트릭과는 독립적으로 작동하며, 이진적이지 않고 점진적인 일관성 측정을 제공합니다.

- **Performance Highlights**: MEt3R는 다중 시점 이미지를 생성하는 기존 방법들의 일관성을 평가하는 데 사용되며, 특히 MV-LDM이 품질과 일관성 간의 trade-off에서 좋은 성능을 보임을 증명했습니다. 이전 메트릭들과의 비교에서 MEt3R는 아주 일관된 시퀀스와 거의 일관된 시퀀스를 구분할 수 있습니다. 이 메트릭은 개방형 소스 다중 시점 잠재적 확산 모델인 MV-LDM과 함께 잘 정렬된 일관성 측정을 가능하게 합니다.



### Towards Iris Presentation Attack Detection with Foundation Models (https://arxiv.org/abs/2501.06312)
- **What's New**: 기초 모델(Foundation Models)인 DinoV2와 VisualOpenClip을 기반으로 한 새로운 아이리스 프레젠테이션 공격 탐지(Iris Presentation Attack Detection, PAD) 접근법이 소개되었습니다. 이 연구는 소규모 신경망을 헤드로 사용하여 기존의 딥러닝 접근법을 초월하는 성능 향상을 보여주었습니다. 특히, 합법적인 이미지와 공격 이미지가 동시에 제공될 경우에 학습된 시스템이 더 나은 성능을 보임을 발견했습니다.

- **Technical Details**: DinoV2 모델은 140 million개의 레이블이 없는 이미지로 학습되며, 이미지 수준과 픽셀 수준의 작업을 위한 일반적인 기능을 생성합니다. VisualOpenClip은 4억 개의 이미지-텍스트 쌍으로 학습되어, 자연어 지도에서 시각적 개념을 배웁니다. 이 연구는 LivDet-Iris 2020 대회의 데이터셋과 추가적인 3개의 아이리스 이미지 데이터베이스를 사용하여 검증하였습니다.

- **Performance Highlights**: 본 연구의 근본적인 기여는, 기초 모델에 기반한 아이리스 PAD가 소규모 신경망을 통해 수행 가능하며, 모든 전통적 딥러닝 방법들보다 뛰어난 성능을 발휘할 수 있음을 입증한 것입니다. 아이리스 PAD 분야에서 기초 모델 접근법의 중요성을 더욱 부각시킨 이번 연구는 다양한 전처리 전략과 결합하여 향후 연구의 방향성을 제시하고 있습니다.



### Visualizing Uncertainty in Image Guided Surgery a Review (https://arxiv.org/abs/2501.06280)
- **What's New**: 이번 연구에서는 뇌종양 절제 수술 중 네트로내비게이션(neuronavigation)의 중요성을 강조합니다. 기존의 네트로내비게이션은 주로 MRI 및 초음파와 같은 수술 전 이미지를 기반으로 하여 뇌의 위치를 안내합니다. 하지만 뇌 변위(brain shift)로 인해 수술 중에 이미지가 왜곡될 수 있으며, 이는 수술의 신뢰성에 부정적인 영향을 미칩니다.

- **Technical Details**: 불확실성(uncertainty)은 수술 중에서 다양한 원인으로 발생할 수 있으며, 이는 이미지 처리, 추적(tracking), 모델링, 측정 등 여러 집합 요소에 기인합니다. 불확실성을 효과적으로 다루기 위해서는 두 가지 필수 요소가 필요합니다: 1) 불확실성의 정량화(quantifying uncertainty)와 2) 정량화된 값을 관찰자에게 전달하는(conveying) 방법입니다. 이러한 이슈는 지난 수십 년 이상 다양한 연구 영역에서 다루어져 왔습니다.

- **Performance Highlights**: 불확실성 시각화(visualization of uncertainty)는 의료 분야와 특히 이미지 유도 수술(image-guided surgery)에서 주목받고 있지만, 아직 연구가 상대적으로 부족한 상황입니다. 이 연구는 불확실성을 시각화하여 수술 중 신뢰성을 높이는 방향으로의 향후 연구의 필요성을 강조합니다.



### OpenAI ChatGPT interprets Radiological Images: GPT-4 as a Medical Doctor for a Fast Check-Up (https://arxiv.org/abs/2501.06269)
- **What's New**: OpenAI는 2023년 3월 14일에 ChatGPT의 성공에 이어 GPT-4를 출시했습니다. GPT-4는 기존의 GPT-3 기능 외에도 이미지를 해석하는 능력을 갖추고 있습니다. 이를 위해 모델과 처리 능력이 크게 향상되었습니다.

- **Technical Details**: 본 연구에서는 인공지능(AI)을 활용하여 헬스케어 분야에서의 방사선 이미지를 해석하는 방법을 탐구합니다. 또한, GPT-4의 이미지 해석 능력을 실험할 예정입니다. 이러한 접근을 통해 인공지능(AI)이 의료 전문가(예: 의사)를 대체할 수 있는지, 아니면 결정을 더 쉽게 하고 신뢰할 수 있도록 돕는 도구로 사용될 수 있는지를 논의합니다.

- **Performance Highlights**: GPT-4는 인공지능의 응용 및 효과를 넘어서는 이미지 처리 및 해석 기능을 보여줍니다. 이는 헬스케어 분야에서 의료진의 업무를 지원하는 중요한 단계로 볼 수 있습니다. 실험 결과에 따라, AI가 의사 결정을 지원하는 도구로서 유용성을 가지는지 구체적으로 검토할 예정입니다.



### GelBelt: A Vision-based Tactile Sensor for Continuous Sensing of Large Surfaces (https://arxiv.org/abs/2501.06263)
Comments:
          Accepted to IEEE RA-L. 8 pages, 7 figures, webpage: this https URL

- **What's New**: 이 논문에서는 기존의 점접촉형 촉각 센서의 한계를 극복한 새로운 비전 기반 촉각 센서인 GelBelt를 제안하고 있습니다. GelBelt는 두 개의 바퀴와 엘라스토머 벨트를 이용하여 지속적인 표면 스크리닝을 가능하게 하며, 대규모 표면을 빠르게 스캔하면서도 높은 정확도를 유지합니다. 이 시스템은 고해상도 표면 형상 재구성과 표면 융합에서 유망한 성능을 보였습니다.

- **Technical Details**: GelBelt의 설계는 두 개의 바퀴 구조와 감지 재료로 이루어진 벨트를 포함하고 있습니다. 벨트가 바퀴를 돌면서 표면을 스캔하게 되며, 각 프레임에서 광학 구성 요소들은 고정되어 접촉 정보를 수집합니다. 또한, 벨트의 측면에는 마커가 추가되어 프레임 정렬과 접촉 힘의 추정을 강화합니다.

- **Performance Highlights**: 실험 결과, GelBelt는 45 mm/s의 속도로 대규모 표면을 매우 높은 정확도로 스캔할 수 있는 능력을 나타냈습니다. 측정된 평균 점곱 값은 0.97 이상으로, 이는 단일 프레임 및 전반적인 표면 재구성 모두에서 훌륭한 정렬을 보여줍니다. 또한, GelBelt는 다중 접촉 힘 및 각도 정보를 제공하여 미래의 폐쇄 루프 스캐닝 분야에 기여할 수 있는 가능성을 열어줍니다.



### CAMs as Shapley Value-based Explainers (https://arxiv.org/abs/2501.06261)
Comments:
          Accepted by The Visual Computer

- **What's New**: 본 논문에서는 신경망 예측 과정의 게임 이론적인 모델링을 통해 Class Activation Mapping (CAM) 방법을 보다 깊이 이해하고 개선하기 위한 Content Reserved Game-theoretic (CRG) Explainer를 소개합니다. 이 이론적 프레임워크는 GradCAM과 HiResCAM의 이론적 기초를 명확히 하며, 신경망의 예측을 협력 게임으로 모델링합니다. ShapleyCAM이라는 새로운 방법을 개발하여 순도 높은 시각적 설명을 제공합니다.

- **Technical Details**: ShapleyCAM은 신경망의 기울기(gradient)와 Hessian 행렬을 활용하여 보다 정밀하고 이론적으로 뒷받침된 설명을 생성하는 알고리즘입니다. 정확한 Shapley 값 계산의 어려움을 극복하기 위해 협력 게임의 효용 함수에 대해 이차 Taylor 전개를 적용하여 닫힌 형식의 표현을 도출합니다. 추가로 Residual Softmax Target-Class (ReST) 유틸리티 함수를 도입하여 기존 전처리 및 후처리 softmax 점수의 한계를 해결합니다.

- **Performance Highlights**: 12개의 인기 네트워크 모델에 대한 대규모 실험을 통해 ShapleyCAM과 그 변형들이 기존의 gradient 기반 CAM 방법에 비해 뛰어난 성능을 보임을 입증했습니다. 연구 결과는 CAM의 설명 가능성을 향상시키고, 휴리스틱 중심의 CAM 방법과 컴퓨팅 집약적인 Shapley 값 기반 방법 간의 간극을 해소하는 데 기여합니다. 논문의 코드는 제공된 링크에서 확인할 수 있습니다.



### Quantum Down Sampling Filter for Variational Auto-encoder (https://arxiv.org/abs/2501.06259)
Comments:
          19 pages, 13 figures

- **What's New**: 본 연구에서 제안된 Q-VAE(Quantum Variational Autoencoder)는 인코더에서 양자 인코딩 기법을 적용하고, 디코더에서 컨볼루션 신경망(CNN)을 사용하는 하이브리드 모델입니다. 이 모델은 16x16 해상도의 저해상도 입력을 32x32로 업샘플링하여 이미지 재구성 품질을 개선하는 것을 목표로 하며, 기존 전통적 VAE들이 자주 발생하는 흐릿하고 부정확한 결과를 극복하려고 합니다.

- **Technical Details**: Q-VAE는 양자 컴퓨팅 기술을 VAE 인코더에 통합하여, 복잡한 고해상도 의존성을 효과적으로 모델링하고 특징 추출을 향상시킵니다. 본 연구에서는 MNIST 및 USPS 데이터셋을 사용하여 Q-VAE의 성능을 평가하며, 기존의 클래스 관리 VAE(Classical VAE)와 CDP-VAE(설계의 다이렉트 패스 VAE)와 비교합니다.

- **Performance Highlights**: Q-VAE는 Fréchet Inception Distance(FID)와 Mean Squared Error(MSE)와 같은 성과 측정 기준에서 클래식 VAE와 CDP-VAE에 비해 현저히 낮은 값을 기록하여 재구성 품질을 입증합니다. 이러한 결과는 양자 강화 VAE가 이미지 재구성 품질을 향상시키는 잠재력이 있음을 보여줍니다.



### The State of Post-Hoc Local XAI Techniques for Image Processing: Challenges and Motivations (https://arxiv.org/abs/2501.06253)
- **What's New**: 이 논문은 Explainable Artificial Intelligence (XAI)의 필요성과 접근 방식, 그리고 XAI가 직면한 도전과제를 심층적으로 논의합니다. 특히 XAI의 필요성을 강조하며, 각 분야에 특화된 사고가 요구된다고 주장합니다. 또한, 인공지능 시스템에 대한 설명의 요구가 고조되고 있는 이유와 이로 인해 발생하는 기술적 및 사회적 문제에 대해서도 이야기합니다.

- **Technical Details**: XAI는 다양한 산업 및 공공 분야에서 AI 시스템의 투명성과 해석 가능성을 높이기 위한 연구 분야로, 의료, 금융, 자율주행차 등 안전-critical한 응용 프로그램에서 특히 중요합니다. 이 논문에서는 XAI의 다양한 연구 접근 방식과 그에 따른 문제점을 다루고 있으며, AI의 결정 과정에 대한 설명이 어떻게 이루어져야 하는지에 대한 논의를 포함하고 있습니다. 각 논문들은 Domain-Specific, Human-Centric, Socio-Technical의 세 가지 카테고리로 나눌 수 있으며, 이는 XAI 관련 기술들의 목표 및 요구 사항이 어떻게 연관되는지를 설명합니다.

- **Performance Highlights**: 논문은 XAI가 AI 기술의 신뢰성과 책임성을 높이는 데 기여할 것으로 기대하며, 이를 통해 보다 안전한 AI 시스템 개발을 도모할 수 있을 것으로 보입니다. 특히, 설명 가능성이 AI 시스템의 오류나 문제를 진단하는 데 중요한 역할을 하며, 이를 통해 부작용을 예방할 수 있음을 강조합니다. 마지막으로, XAI 연구 분야의 향후 방향성을 제시하며, 기술 중심의 접근뿐만 아니라 사용자 중심의 접근 필요성도 강조합니다.



### Generative AI for Cel-Animation: A Survey (https://arxiv.org/abs/2501.06250)
Comments:
          20 pages

- **What's New**: 전통적인 셀룰로이드 애니메이션 제작 파이프라인은 여러 주요 단계를 포함하고 있으며, Generative AI(GenAI)의 도입이 이 과정을 혁신적으로 변화시키고 있습니다. GenAI는 inbetween 프레임 생성과 색상화, 스토리보드 제작 등에서 높은 기술 장벽을 낮추고 있으며, 다양한 제작 도구를 통해 더 많은 창작자들에게 접근성을 높이고 있습니다. 이를 통해 예술가들은 보다 창의적인 작업에 집중할 수 있는 기회를 얻게 됩니다.

- **Technical Details**: 전통 셀룰로이드 애니메이션은 스토리보드, 레이아웃 디자인, 키프레임 애니메이션, 인비트윈과 색상화 등 여러 단계로 구성되어 있습니다. 이 과정들은 많은 수작업, 전문 지식, 그리고 시간 투자가 요구되어 왔습니다. GenAI는 대규모 언어 모델(LLMs), 다중 모드 모델(MLLMs), 확산 모델(difusion models)을 포함하여 여러 반복적 작업을 자동화하여 애니메이션 제작의 효율성을 크게 향상시키고 있습니다.

- **Performance Highlights**: Generative AI의 도입은 셀 애니메이션 제작의 효율성과 창의적 가능성을 개선했습니다. 예를 들어, AniDoc는 비디오 확산 모델을 사용하여 2D 애니메이션의 inbetween과 색상화를 자동화하고, ToonCrafter는 비선형 모션과 겹침을 효율적으로 처리합니다. Netflix Japan의 실험과 CogCartoon 같은 도구는 이러한 AI 기술이 애니메이션과 같은 예술적 양식의 민주화를 가능하게 하고 있음을 보여주고 있습니다.



### Scalable Cosmic AI Inference using Cloud Serverless Computing with FMI (https://arxiv.org/abs/2501.06249)
- **What's New**: 이 논문에서 소개된 Cloud-based Astronomy Inference (CAI) 프레임워크는 현대 천문학에서 중요한 대규모 이미지 데이터 처리를 돕기 위해 개발되었습니다. CAI는 사전 훈련된 foundation models를 serverless 클라우드 인프라와 결합하여 사용자가 하드웨어를 대규모로 사용하지 않고도 효율적인 예측을 가능하게 합니다.

- **Technical Details**: CAI는 Function-as-a-Service (FaaS) Message Interface (FMI)를 통해 통합되며, 이는 천문학 이미지에 대한 추론(inference)을 지원합니다. 이 연구는 redshift 예측을 위한 foundation model을 사례로 삼아 다양한 사용자 장치, HPC (High-Performance Computing) 서버 및 클라우드를 통해 실험했습니다.

- **Performance Highlights**: CAI는 대량 데이터 처리 시 중요한 확장성(scalability) 향상을 가져오며, 천문학 커뮤니티에 접근 가능하고 효과적인 도구를 제공합니다. 이를 통해 천문학 이미지 데이터 분석 및 예측의 효율성이 크게 개선되었습니다.



### NextStop: An Improved Tracker For Panoptic LIDAR Segmentation Data (https://arxiv.org/abs/2501.06235)
- **What's New**: 본 연구에서는 4D panoptic LiDAR segmentation 작업에서의 추적 성능 개선을 위해 NextStop1 트래커를 소개합니다. 이 트래커는 Kalman filter 기반의 motion estimation, 데이터 연관(data association) 및 lifespan 관리를 통합하여 복잡한 환경에서도 작은 객체 추적 성능을 높입니다. 특히, 사람 및 자전거와 같은 소형 객체의 ID 스위치가 줄어들고 추적 시작 시점이 조기화되는 효과가 있습니다.

- **Technical Details**: NextStop 트래커는 SORT (Simple Online and Real-Time Tracker) 개념을 기반으로 하여 Kalman 필터를 사용해 움직임을 추정하고, 헝가리안 알고리즘을 통해 프레임 간의 객체 감지를 연관시킵니다. 이 과정에서 tracklet state 개념을 도입하여 여러 트래커 및 감지 결과의 우선순위를 정합니다. 연구진은 기존의 4D-STOP 접근법에 NextStop 트래커를 통합하여 LSTQ 메트릭에서 현저한 성능 향상을 입증하였습니다.

- **Performance Highlights**: NextStop 트래커는 작은 객체 추적에서 향상된 성능을 보여주었으며, ID 스위치가 감소하고 복잡한 환경에서의 신뢰성이 개선되었습니다. LiDAR Segmentation and Tracking Quality (LSTQ) 기준으로 볼 때, 기존 방법들에 비해 연속성이 우수하며 조기 추적 시작이 가능함을 강조합니다. 이러한 결과들은 복잡한 환경에서도 안정적인 객체 추적을 가능하게 만들어, 자율주행 및 로봇 기술에 주요한 기여를 할 것입니다.



### BEN: Using Confidence-Guided Matting for Dichotomous Image Segmentation (https://arxiv.org/abs/2501.06230)
Comments:
          13 pages, 2 figures, 2 tables, and 2 algorithms

- **What's New**: 이 논문에서는 dichotomous image segmentation (DIS) 분야에서 image matting과 object segmentation을 통합하는 새로운 접근법인 Confidence-Guided Matting (CGM)를 제안합니다. CGM의 주요 모델인 Background Erase Network (BEN)는 초기 분할을 위한 BEN Base와 신뢰도 정제를 위한 BEN Refiner로 구성됩니다. 이 방법론을 통해 DIS5K 검증 데이터셋에서 현재의 최첨단 기술을 능가하는 성과를 거두었으며, segmentation 품질을 현저히 향상시킬 수 있음을 입증했습니다.

- **Technical Details**: BEN Base 아키텍처는 Multi-view Aggregation Network (MVANet)과 유사하지만, 활성화 함수와 정규화를 변경하여 설계되었습니다. 손실 함수는 Weighted BCE, Weighted IoU, Weighted Structural Similarity (SSIM) 등 세 가지 지표를 사용합니다. CGM은 기본 모델의 신뢰도에 따라 업데이트 되는 trimap을 생성하며, 이는 불확실한 영역에 중점을 두어 개선된 분할 정확도를 달성합니다.

- **Performance Highlights**: 본 연구 결과는 DIS5K 검증 데이터셋에서 신뢰성 기반 정제 기법이 segmentation 품질을 현저히 향상시킬 수 있음을 보여주었습니다. CGM 접근법은 기존 모델들에 비해 더 정교하고 더 높은 정확도를 달성하며, 컴퓨터 비전 분야에서 matting과 segmentation 기법 간의 새로운 가능성을 열어주는 혁신적인 결과를 제공합니다. 이러한 성과는 각종 상업 관심을 불러 일으키고 있으며, 오픈 소스 커뮤니티와의 협업 안정성을 높이는데 기여하고 있습니다.



### Open-Source Manually Annotated Vocal Tract Database for Automatic Segmentation from 3D MRI Using Deep Learning: Benchmarking 2D and 3D Convolutional and Transformer Networks (https://arxiv.org/abs/2501.06229)
- **What's New**: 이 연구는 자기공명영상(MRI) 데이터에서 음성관(vocal tract)의 정확한 세분화(segmentation)를 위한 새로운 접근 방식을 제시합니다. 수동 세분화는 시간이 많이 소모되고 오류에 취약하지만, 본 연구에서는 이러한 단점을 극복하기 위해 딥러닝(digital learning) 알고리즘을 사용하였습니다.

- **Technical Details**: 3D MRI 데이터를 활용한 음성관 세분화에서 딥러닝 모델이 적용되었습니다. 특히, Convolutional Neural Networks (CNNs)와 같은 기술이 사용되어 데이터를 처리하고, 이전 연구들에 비해 더 높은 정확도를 달성하는지 평가하였습니다.

- **Performance Highlights**: 실험 결과는 딥러닝 알고리즘이 수동 세분화에 비해 더 빠르고 효율적으로 음성관 세분화를 수행할 수 있음을 보여줍니다. 이러한 성능 향상은 다양한 음성 및 언어 관련 애플리케이션에 중요한 기여를 할 것으로 기대됩니다.



### A Distributed Hybrid Quantum Convolutional Neural Network for Medical Image Classification (https://arxiv.org/abs/2501.06225)
- **What's New**: 이 논문에서는 분산 하이브리드 양자 컨볼루션 신경망(distributed hybrid quantum convolutional neural network)을 제안하여 의료 영상 분류의 효율성과 정확성을 개선하려고 합니다. 이를 위해 양자 회로 분할(quantum circuit splitting) 기술을 활용하여 양자 자원을 절약하면서도 8-qubit QCNN을 5-qubit 시스템에서 실행할 수 있도록 설계되었습니다. 이 모델은 이전 기술에 비해 적은 수의 파라미터로도 뛰어난 성능을 보입니다.

- **Technical Details**: 모델은 MobileNetV2 아키텍처를 활용하여 의료 영상의 고차원 특징을 효과적으로 추출합니다. 훈련 데이터셋을 통해 학습한 후, 분산 8-qubit 양자 컨볼루션 신경망에서 특징 벡터를 입력으로 사용하여 분류 작업을 수행합니다. 제안된 시스템은 CNN을 통하여 특징 추출 및 차원 축소를 먼저 수행하고, 이후에 양자 회로 분할 기술을 통한 양자 컨볼루션 신경망으로 전환합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 3개의 데이터셋에서 이진 및 다중 클래스 분류 작업 모두에서 우수한 성능을 기록했습니다. 특히, 최근의 여러 모델에 비해 적은 수의 파라미터로도 높은 정확도를 달성하여 양자 계산의 장점을 잘 보여줍니다. 이 모델의 설계는 의료 영상 분석을 위한 QCNN의 활용 가능성을 더욱 확장합니다.



### Detection, Retrieval, and Explanation Unified: A Violence Detection System Based on Knowledge Graphs and GA (https://arxiv.org/abs/2501.06224)
Comments:
          14 pages. Submitted to Neurocomputing

- **What's New**: 최근에 통합 멀티모달 모델을 이용한 폭력 탐지 시스템이 큰 성공을 거두고 많은 주목을 받고 있습니다. 그러나 기존 시스템들은 해석 가능성이 부족하고 단지 분류(classification) 또는 검색(retrieval) 기능만을 제공하는 한계가 있습니다. 이를 해결하기 위해 본 논문에서는 TIO(Three-in-One) 시스템이라는 새로운 해석 가능한 폭력 탐지 시스템을 제안합니다.

- **Technical Details**: TIO 시스템은 지식 그래프(knowledge graphs)와 그래프 주의 네트워크(graph attention networks)를 통합하여 탐지(detection), 검색(retrieval), 설명(explanation)의 세 가지 핵심 기능을 제공합니다. 시스템은 잠재적인 폭력 행동을 포함하는 비디오의 각 프레임을 대규모 언어 모델(large language model)의 텍스트 설명과 함께 처리하며, ImageBind를 사용하여 지식 그래프 생성을 위한 고차원 임베딩(high-dimensional embeddings)을 생성합니다. 또한, GAT를 통해 추론(reasoning)을 수행하고, 경량화된 시계열 모듈을 활용하여 비디오 임베딩 특징을 추출합니다.

- **Performance Highlights**: TIO 시스템의 효율성을 높이기 위해 자원 소비를 줄이는 여러 경량화 방법을 도입하였습니다. XD-Violence와 UCF-Crime 데이터셋에서 수행된 광범위한 실험을 통해 제안된 시스템의 효과가 검증되었습니다. 사례 연구를 통해 흥미로운 현상이 발견되었는데, 구경꾼(bystanders)의 수가 증가할수록 폭력 행동의 발생이 감소하는 경향을 보였습니다.



### WhACC: Whisker Automatic Contact Classifier with Expert Human-Level Performanc (https://arxiv.org/abs/2501.06219)
- **What's New**: 이번 연구에서는 설치류의 촉각 인식을 자동으로 식별할 수 있는 Whisker Automatic Contact Classifier(WhACC)라는 파이썬 패키지를 개발하였습니다. 이 패키지는 ResNet50V2를 기반으로 한 특징 추출 및 LightGBM을 활용한 분류 기술을 사용하여, 사람 수준의 성능으로 고속 비디오에서의 터치를 식별할 수 있습니다. WhACC는 100백만 프레임 데이터셋에서 수작업으로 필요한 시간을 약 333시간에서 6시간으로 줄이는 데 기여했습니다.

- **Technical Details**: WhACC는 설치류의 머리가 고정된 상태에서의 행동을 기록한 고속 비디오 데이터를 기반으로 하여, 61x61 픽셀의 아무것도 없는 객체 주변의 픽셀을 추출하여 사용합니다. 이 모델은 2048개의 특징에서 99.5%의 동의율로 전문가의 커스터마이징 결과와 일치하여, 터치 시간을 밀리초 수준의 정확도로 식별할 수 있습니다. 이미지 데이터의 증강과 후행 이미지 기법을 통해 다양한 실험 환경에서도 좋은 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: WhACC의 성능은 1백만 프레임 이상의 비디오 데이터에서 세 명의 전문가와의 쌍별 터치 분류 동의 비율이 99.46%에 달하는 것으로 평가되었습니다. 모델의 오류 측정을 통해 주로 발생하는 에러 유형인 엣지 오류는 7배 더 빈번하게 발생하였으며, 이는 신경 반응 분석에 미치는 영향을 파악하기 위해 중요한 지표입니다. 전체적인 목표는 다양한 촬영 조건에서도 전문가 수준의 정확도를 유지하며 손으로 커스터마이징하는 시간적 부담을 최소화하는 것이었습니다.



### Dissecting Bit-Level Scaling Laws in Quantizing Vision Generative Models (https://arxiv.org/abs/2501.06218)
- **What's New**: 최근 비전 생성 모델들이 두 가지 주요 패러다임인 diffusion 스타일과 language 스타일에서 눈에 띄는 발전을 이루고 있습니다. 특히 quantization(양자화)은 이러한 모델을 효율적으로 배포하기 위해 메모리 및 계산 비용을 줄이는 데 필수적인 기술로 자리잡고 있습니다. 이 연구에서는 두 가지 패러다임에 대한 quantization의 영향을 체계적으로 조사하였고, language 스타일 모델이 diffusion 스타일 모델보다 우수한 성능을 지속적으로 발휘함을 발견하였습니다.

- **Technical Details**: 비전 생성 모델들은 기본적으로 diffusion 스타일 모델과 language 스타일 모델 두 가지로 분류됩니다. Diffusion 스타일 모델은 고품질 이미지 생성에서 탁월한 성능을 발휘하는 모델로, 노이즈가 있는 입력을 반복적으로 정제하여 이미지를 생성하는 과정으로 동작합니다. 반면, language 스타일 모델은 코드북(codebook)을 사용하여 이미지 생성시 더 나은 비트 수준의 확장성을 보여줍니다. 이 연구에서는 DiT와 VAR 모델을 사용해 이들 모델의 성능을 비교 분석하였습니다.

- **Performance Highlights**: 연구 결과, language 스타일 모델은 full precision(풀 정밀도)에서 비슷한 정확도를 달성함에도 불구하고 다양한 quantization 환경에서 consistently(지속적으로) 높은 성능을 보였습니다. 특히, language 스타일 모델의 비트 수준 확장성은 full precision보다 개선되었으며, 반면 diffusion 스타일 모델은 풀 정밀도와 비교할 때 더 나쁜 확장성을 나타냈습니다. TopKLD라는 새로운 지식 증류 방법을 통해 language 스타일 모델의 비트 수준 확장성을 향상시키는 방안을 제안하며, 이 방법이 모델의 성능을 한 단계 끌어올리는 데 기여할 수 있음을 보여주었습니다.



### Understanding colors of Dufaycolor: Can we recover them using historical colorimetric and spectral data? (https://arxiv.org/abs/2501.06216)
Comments:
          8 pages, 6 figures, 4 tables; submitted to proceedings of 3rd international conference on "Colour Photography and Film: analysis, preservation, and conservation of analogue and digital materials",

- **What's New**: 이 논문은 1935년부터 1950년대 후반까지 제작된 Dufaycolor라는 색상 사진 기술의 원래 색상을 재구성하기 위한 오픈 소스(Color-Screen) 도구의 연구와 개발을 다루고 있습니다. 이 도구는 고대의 색상 필터인 réseau에서 사용된 염료의 역사적 측정을 포함하여 색상 복원을 더욱 정확하게 도와줍니다.

- **Technical Details**: Dufaycolor 프로세스는 분산 색상(additive color) 사진의 혁신적인 방법으로, 색상 재구성을 위한 Color-Screen 도구 개발에 필요한 염료의 물리적 특성과 계측(exact measurements)들을 활용합니다. 역사적 데이터는 색상 필터의 성능을 최적화하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 연구를 통해 색상 복원(error recovery) 정확도가 크게 향상되었으며, 기존 기술보다 더 신뢰할 수 있는 결과를 제공합니다. 이 Color-Screen 도구는 Dufaycolor 사진 기술의 진정한 매력을 재발견하는 데 기여하고 있습니다.



### Fitting Different Interactive Information: Joint Classification of Emotion and Intention (https://arxiv.org/abs/2501.06215)
- **What's New**: 이 논문은 ICASSP MEIJU@2025 트랙 I의 1등 해법으로, 저자원 환경에서의 다중모드 감정 및 의도 인식을 중심으로 하고 있습니다. 특정한 대화에서 다양한 난이도의 작업 증진을 위한 접근법과 막대한 양의 레이블 없는 데이터를 효과적으로 활용하는 방법이 주된 초점입니다. 이를 통해 저자들은 0.5532의 점수로 트랙에서 우승을 차지했습니다.

- **Technical Details**: 연구의 모델 아키텍처는 다중 헤드 자기 주의(multi-head self-attention)와 게이팅 메커니즘을 통한 상호 작용 모듈(combination module)을 기반으로 구성됩니다. 비디오에서 매 30프레임마다 샘플링된 이미지를 사용하고, 오디오 데이터는 ffmpeg를 활용하여 WAV 형식으로 변환합니다. 텍스트 데이터에서 발견된 오류는 수작업으로 수정되어 감정과 의도를 더 잘 분류할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면 감정 인식과 의도 인식은 서로 관련이 있으나, 두 과제를 동시에 최적의 정확도로 수행하기는 어렵다는 점이 발견되었습니다. 다중 헤드 자기 주의 방식을 통해 의도 인식의 성능이 향상되었으며, 두 단계의 훈련 전략을 통해 최종 점수 0.5532를 달성했습니다. 이러한 접근 방식은 작업 성능을 크게 향상시켰습니다.



### Path Space Partitioning and Guided Image Sampling for MCMC (https://arxiv.org/abs/2501.06214)
- **What's New**: 이번 연구는 경로 공간(path space)을 분할하여 각각의 분할된 공간에서 개별적인 추정기(estimator)를 통해 통합하는 새로운 접근 방식을 제안합니다. 이는 전체 경로 공간을 단순히 통합하는 방식보다 더 효율적일 수 있으며, MCMC(Markov Chain Monte Carlo) 방법을 이용해 이러한 경로 공간을 통합하는 데 중점을 둡니다. 또한, 이미지 공간(image space)에서 제안 분포(proposal distribution)를 안내함으로써 효율성을 높이고 경로 생성의 유효성을 향상시키는 방법을 소개합니다.

- **Technical Details**: 연구에서는 Monte Carlo 경로 추적(Monte Carlo path tracing) 기법으로 경로 공간을 분할하고, 각 분할 내 경로의 기여도를 추정하는 방법을 사용합니다. 각 경로의 기여를 denoising하여 이미지 평면 가이드 분포(image plane guidance distribution)를 구축하게 됩니다. 이로 인해 경로 공간의 sparse subset 내에서 보다 효율적으로 통합이 이루어질 수 있습니다. 또, 제안된 방법은 MCMC 통합 접근 방식 중에서 이미지 품질(image quality)을 향상시키는 성과를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존 MCMC 통합 접근 방식 대비 동일한 샘플 수(sample number)에서 이미지 품질을 개선하는 결과를 보였습니다. 경로 공간을 효율적으로 분할함으로써, 경로 생성 과정에서 더 많은 비제로 경로(non-zero path)를 생성할 수 있게 되어, 동일한 계산 비용(computational cost)으로도 더 나은 결과를 얻을 수 있습니다. 전반적으로, 이 연구는 최신 렌더링 알고리즘의 성능 향상 가능성을 보여주는 중요한 발전을 의미합니다.



### Imagine while Reasoning in Space: Multimodal Visualization-of-Though (https://arxiv.org/abs/2501.07542)
Comments:
          11 pages, 6 figures, 4 tables (27 pages, 10 figures, 16 tables including references and appendices)

- **What's New**: 이번 연구에서 제안한 Multimodal Visualization-of-Thought (MVoT) 방법론은 이미지와 언어를 결합하여 복잡한 사고 과정을 시각적으로 표현할 수 있게 해줍니다. MVoT는 MLLM (Multimodal Large Language Model)이 자연스럽게 시각적 사고를 생성하면서 언어적 사고를 동시에 진행하는 새로운 패러다임을 세웁니다. 이는 기존의 Chain-of-Thought (CoT) 접근 방식이 복잡한 공간적 추론에서 한계가 있었던 점을 극복하는 데 기여합니다. 최종적으로 MVoT는 언어적 추론을 보완할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MVoT는 두 가지 모드 (textual, visual)를 결합하여 다중 모드 생각을 생성하는 프로세스를 정의합니다. 주어진 입력 시퀀스에 대해 모델은 중간 단계에서 이미지 시각화를 추가하여 언어적 사고를 보완합니다. 이는 언어 시퀀스와 이미지 시퀀스를 병행 생성하며, 각 단계는 이전의 단계와 결합하여 더욱 풍부한 데이터를 생성합니다. 실험에서는 Chameleon-7B 모델을 통해 MVoT의 효과를 검증하며, 토큰 불일치 손실(token discrepancy loss)을 도입하여 고품질의 시각화를 보장합니다.

- **Performance Highlights**: MVoT는 여러 동적인 공간 추론 작업에서 경쟁력 있는 성능을 입증하였습니다. Maze, MiniBehavior, FrozenLake와 같은 벤치마크에서 전통적인 CoT 방식을 20% 이상 초과하는 성능을 보였습니다. 특히, MVoT는 CoT가 실패하는 가장 도전적인 시나리오에서도 부분적으로 안정적이며 신뢰할 수 있는 개선을 보여줍니다. 이번 연구는 복잡한 과제에서 시각적 사고의 효용성을 입증하는 중요한 기초 연구로 평가받고 있습니다.



### 3DGS-to-PC: Convert a 3D Gaussian Splatting Scene into a Dense Point Cloud or Mesh (https://arxiv.org/abs/2501.07478)
- **What's New**: 이 논문에서는 3D Gaussian Splatting (3DGS) 장면을 포인트 클라우드(point cloud)로 변환하기 위한 유연하고 사용자 정의 가능한 프레임워크인 3DGS-to-PC를 소개합니다. 이 프레임워크는 확장성이 뛰어나고 쉽게 기존 3DGS 파이프라인에 통합될 수 있습니다. 주요 특징은, 각 Gaussian에서 3D 밀도 함수로부터 점을 확률적으로 샘플링하고, Mahalanobis 거리를 사용하여 극단적인 아웃라이어를 예방하는 것입니다.

- **Technical Details**: 3DGS-to-PC는 .ply 또는 .splat 형식의 3DGS 데이터와 .json 또는 COLMAP 형식의 카메라 포즈를 수용하여 포인트 수와 샘플링 옵션을 사용자 지정할 수 있습니다. 이 프레임워크는 Gaussian의 위치, 크기, 회전 및 투명도를 기반으로 포인트를 균일하게 샘플링하고, 각 포인트가 기여하는 색상에 따라 장면을 렌더링하여 시각적인 정확도를 높입니다.

- **Performance Highlights**: 이 프레임워크는 이미 존재하는 포인트 클라우드 처리 도구와 호환되며, Gaussian의 색상을 재계산하여 최종 포인트 클라우드에서 나타나는 색상의 정확성을 개선합니다. Poisson Surface Reconstruction을 통해 메쉬(mesh) 생성을 지원하며, 3DGS 장면으로부터 전체 재학습 없이도 색상과 퀄리티를 유지한 메쉬를 생성할 수 있는 능력을 가지고 있습니다.



### PrecipDiff: Leveraging image diffusion models to enhance satellite-based precipitation observations (https://arxiv.org/abs/2501.07447)
- **What's New**: 본 연구에서는 막대한 인명을 잃게 했던 물 관련 재해를 줄이기 위한 방법으로, 위성 기반 강수 측정 데이터의 정확도를 높이고 공간 해상도를 향상시키기 위해 확산 모델(diffusion model)을 활용한 새로운 접근 방식을 제안합니다. 특히 기존의 저조한 해상도 문제를 해결하기 위해, 기존 위성 강수 데이터에서 10 km 해상도를 1 km로 다운스케일링(downscaling)하는 방법을 통해 다양한 기상 데이터를 통합적으로 보정하는 기법을 소개합니다.

- **Technical Details**: 이 연구는 다양한 강수량 데이터의 불일치를 교정하기 위해, 잔차 학습(residual learning)을 도입하여 강수량 영역에서의 편향을 보정하는 최초의 컴퓨터 비전 기반 알고리즘을 개발하였습니다. 우리가 제안한 확산 모델은 데이터를 생성하기 위한 확산 과정을 모델링하며, 이 과정에서 노이즈를 점진적으로 제거하여 원래 데이터 분포를 복원하는 방식으로 설계되었습니다. 이를 통해 데이터 수집에 있어 더 높은 정확도와 공간적 세부사항을 달성할 수 있었습니다.

- **Performance Highlights**: 시애틀 지역에서 수행된 실험에서 본 연구의 제안 방법은 기존 강수 예측 시스템에 비해 편향을 현저하게 줄이고 정확도가 크게 향상됨을 입증했습니다. 다른 환경적 변수를 통합하지 않더라도 이 방법은 효과적으로 작동하여, 컴퓨터 비전 기술을 활용한 새로운 강수량 데이터 개선 가능성을 보여주고 있습니다. 본 연구는 위성 강수 측정 제품의 유용성을 높이는 방향으로 나아가며, 향후 기후와 관련된 영향 평가를 위한 더 나은 기초 자료 제공에 기여할 것으로 기대됩니다.



### Anonymization of Documents for Law Enforcement with Machine Learning (https://arxiv.org/abs/2501.07334)
Comments:
          Accepted at IEEE Symposium on CI in Security, Defence and Biometrics 2025 (IEEE CISDB)

- **What's New**: 이 논문에서는 법 집행 기관이 개인 정보를 보호하기 위해 스캔된 문서 이미지를 자동으로 익명화하는 시스템을 제안합니다. 이 시스템은 민감한 영역을 자동으로 감지하고 수동으로 익명화된 참조 문서의 지식을 결합하여 데이터 보호 규정을 준수하면서 수작업을 줄입니다. 단일 익명화된 예제를 사용하여 같은 유형의 모든 문서에서 효과적으로 레드랙션을 수행하는 자가 지도학습(Self-supervised learning) 이미지 모델을 활용합니다.

- **Technical Details**: 이 방법은 문서의 유형에 대한 단일 참조 문서로부터 이미지를 검색하는 인스턴스 검색(Instance Retrieval) 과정을 기반으로 합니다. 데이터베이스에서 직접 문서 메타데이터를 요구하지 않기 때문에 다양한 환경에서도 적용 가능한 프레임워크입니다. 피처 매칭은 코사인 유사성(Cosine Similarity)을 통해 수행되며, DinoV2라는 자가 지도(Self-supervised) 학습 방법을 사용하여 문서 이미지의 의미 있는 피처를 추출합니다.

- **Performance Highlights**: 제안된 방법은 자동화된 레드랙션 시스템 및 참조 익명화를 다른 문서에 단순히 복사하여 붙여넣는 과정을 모두 초월하는 성능을 보였습니다. 손으로 만든 데이터 세트에서 전문적으로 주석이 달린 결과에 기반하여 이 프레임워크의 우수성을 입증하였습니다. 이 연구는 데이터 처리에서 데이터 보호 요구 사항을 충족하는 혁신적인 접근 방식을 제시하여 관련 분야에 기여할 것으로 기대됩니다.



### Comparative analysis of optical character recognition methods for S\'ami texts from the National Library of Norway (https://arxiv.org/abs/2501.07300)
Comments:
          To be published in Proceedings of the 25th Nordic Conference on Computational Linguistics (NoDaLiDa)

- **What's New**: 이번 연구는 노르웨이 국가 도서관(NLN)의 디지털화 과정에서 Sámi 언어 문서의 Optical Character Recognition (OCR) 정확성을 개선하기 위해 세 가지 OCR 접근법인 Transkribus, Tesseract, TrOCR을 미세 조정하고 평가하였습니다.

- **Technical Details**: 이 연구는 Transkribus, Tesseract, TrOCR을 활용하여 Sámi 문서를 디지털화하는 과정에서 발생하는 문제를 해결하고자 하였습니다. 기존의 사전 학습된 모델을 세밀하게 조정하고, 수작업 주석(manual annotations)과 기계 주석(machine annotations) 및 합성 텍스트 이미지(synthetic text images)를 보완하여 OCR 정확성을 증가시킬 수 있음을 발견했습니다.

- **Performance Highlights**: Transkribus와 TrOCR은 이 과제에서 Tesseract보다 뛰어난 성능을 보였으며, Tesseract는 아웃 오브 도메인(out-of-domain) 데이터셋에서 우수한 성과를 나타냈습니다. 또한, 수작업 주석 데이터가 적더라도 OCR의 정확성을 확보할 수 있는 가능성을 제시했습니다.



### MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework (https://arxiv.org/abs/2501.07251)
Comments:
          Under Review of CVPR 2025

- **What's New**: 이번 논문에서는 Multi-Objective Set-based Attack(MOS Attack)이라는 새로운 적대적 공격 프레임워크를 제안합니다. MOS Attack은 여러 개의 손실 함수(loss function)를 활용하고 그들 간의 상호 관계를 자동으로 발견하는 것을 목표로 합니다. 이 접근법을 통해 기존 단일 목표 공격 방법의 한계를 극복하고 효과적인 적대적 예제 생성을 가능하게 합니다.

- **Technical Details**: MOS Attack은 집합 기반의 다중 목표 최적화 전략을 사용하여 추가적인 매개변수 없이도 여러 손실 함수를 통합합니다. 이 과정에서 각 손실 간의 시너지 효과(synergistic patterns)를 자동으로 탐색하여 적대적 공격을 더 강력하고 효율적으로 만듭니다. 특히 이 방법은 적대적 공격의 예제를 생성하기 위해 여러 손실 기능을 동시다발적으로 고려할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실험을 통해 MOS Attack이 기존의 단일 목표 공격 방법들보다 뛰어난 성과를 보여주는 것을 확인했습니다. 또한 신시아 형태를 통해 적은 수의 손실 함수로도 더 나은 결과를 달성할 수 있음을 보여주며, 이는 새로운 적대적 공격 전략의 가능성을 제시합니다.



### Implicit Neural Representations for Registration of Left Ventricle Myocardium During a Cardiac Cyc (https://arxiv.org/abs/2501.07248)
Comments:
          9 pages, 5 figures, STACOM 2024

- **What's New**: 이 논문에서는 심장 주기 동안 좌심실 심근(LVmyo)의 동작을 모델링하기 위해 기존의 딥러닝 기반 방법 대신 암묵적 신경 표현(Implicit Neural Representations, INRs)을 도입했습니다. INRs는 연속적인 포인트에서 작동하여 메모리 요구사항을 감소시키고, 심장 CT에서 LVmyo 등록의 정확성을 높이는 데 기여합니다. 또한, 연구팀은 LVmyo의 기하학적 정보를 제공하는 서명 거리 필드(Signed Distance Field, SDF)를 등록 과정에 통합하여 더욱 정밀한 움직임 분석이 가능하게 했습니다.

- **Technical Details**: LVmyo 등록은 여러 쌍의 변형 이미지 등록(Deformable Image Registration, DIR)으로 간주되며, 각 DIR은 원본 이미지와 타겟 이미지 간의 매핑을 찾는 과정을 포함합니다. 이 과정은 실제로는 함수 Φ:ℝ3→ℝ3을 찾는 것으로, 이 함수는 원본 이미지 좌표를 타겟 이미지 좌표에 맞추기 위해 사용됩니다. 또한, 연구팀은 유사도 손실(similarity loss)과 정규화 손실(regularization loss)을 최적화하는 과정에서 이 손실들을 최소화하여 등록의 정확도를 높입니다.

- **Performance Highlights**: 제안된 프레임워크는 LVmyo의 움직임을 연속적으로 추적할 수 있으며, 기존 방법들에 비해 높은 등록 정확도를 보여줍니다. 연구에 사용된 데이터는 덴마크의 Rigshospitalet에서 수집된 것으로, 100명의 참가자를 통한 다중 시간 스캔을 포함하고 있습니다. 이 연구 결과는 향후 LVmyo 동작 분석 및 심장 기능 평가에 있어 중요한 기초 자료로 활용될 것으로 기대됩니다.



### Multi-face emotion detection for effective Human-Robot Interaction (https://arxiv.org/abs/2501.07213)
Comments:
          9 pages, 8 figures and 1 table. Accepted at the 17th International Conference on Agents and Artificial Intelligence (ICAART 2025), Porto, Portugal

- **What's New**: 이번 연구는 이동형 humanoid 로봇에 통합된 얼굴 감정 인식 인터페이스를 제안합니다. 이 인터페이스는 사용자 인터페이스에서 여러 개체의 실시간 감정을 표시할 수 있습니다. 다양한 deep neural network 모델을 개발하고 평가하여 높은 성능을 달성하였으며, 프로그램의 정확성과 메모리 용량 간의 균형을 고려하여 모바일 로봇에 실효성 있게 구현하였습니다.

- **Technical Details**: 로봇의 감정 인식 기술은 주로 얼굴 탐지(face detection)와 얼굴 감정 인식(face emotion recognition, FER) 시스템에 기반합니다. 본 연구에서는 Haarcascade 분류기를 얼굴 탐지에 사용하였으며, 이는 실시간 응용 프로그램에서 특히 유리합니다. 여러 사용자와의 상호작용을 지원하는 multi-face emotion recognition 기능도 포함되어 있어, 예를 들어 코미디 클럽과 같은 환경에서 실시간 피드백을 제공할 수 있습니다.

- **Performance Highlights**: 연구 결과 다양한 deep learning 기반의 얼굴 감정 인식 모델이 높은 정확도를 기록했으며, 특히 사용자 상호작용 분석을 통해 모델의 성능을 검토하였습니다. 이 모델을 통해 모바일 로봇에서 인간의 감정을 실시간으로 파악하고 적절히 반응할 수 있는 가능성이 demonstrated되었으며, 이는 로봇의 사회적 상호작용을 대폭 향상시킬 수 있는 기회를 제공합니다.



### Lung Cancer detection using Deep Learning (https://arxiv.org/abs/2501.07197)
- **What's New**: 이번 논문에서는 Convolutional Neural Networks (CNN)와 Support Vector Machines (SVM)을 결합한 하이브리드 모델을 사용하여 폐암을 조기 진단하는 방법에 대해 논의합니다. 이 모델은 CT 스캔(computed tomography scans) 이미지를 데이터셋으로 활용하여 종양의 양성과 악성을 구별하는 데 도움을 줍니다.

- **Technical Details**: 하이브리드 모델은 CNN과 SVM의 장점을 결합하여 다양한 패턴 인식을 통해 폐암을 조기 탐지하는 데 혁신적인 접근법을 제시합니다. 데이터는 Computed Tomography(CT) 스캔을 기반으로 하며, 심층 학습(deep learning) 기술이 적용되었습니다.

- **Performance Highlights**: 이 연구는 폐암 조기 발견을 위한 최첨단 방법으로, 고성능의 정확한 진단을 가능하게 합니다. 이를 통해 환자의 생존율을 향상시킬 수 있는 잠재력을 지닌다고 평가됩니다.



### A4O: All Trigger for One samp (https://arxiv.org/abs/2501.07192)
- **What's New**: 이번 논문에서는 다수의 트리거를 통합한 새로운 백도어 공격 메커니즘을 제안합니다. 기존 방법들이 단일 유형의 트리거에 의존하는 것과 달리, 다양한 트리거를 결합하여 방어 시스템을 우회하는 능력을 강조합니다. 특히, 각 트리거 유형의 세기를 줄이고 이를 조합하여 방어 시스템의 감지 범위를 벗어나는 견고한 백도어를 구현합니다.

- **Technical Details**: 연구에서 제시된 방법은 All trigger For One sample Backdoor Attacks (A4O-Attack) 이라는 명칭을 가지고 있으며, 새로운 훈련 모드인 joint mode와 noise mode를 통해 최적화를 수행합니다. joint mode는 모든 트리거를 동시에 학습하게 하며, noise mode는 사전 정의된 트리거에만 집중하게 합니다. 이러한 훈련 방식은 백도어 모델이 효과적이고 탐지가 어렵도록 합니다.

- **Performance Highlights**: 세 가지 표준 데이터셋에 대한 광범위한 실험 결과, 제안된 방법은 기존의 방어 시스템을 일관되게 우회하면서 높은 공격 성공률(ASR)을 기록했습니다. 이는 다중 트리거를 사용하는 공격 방식의 효과성을 보여주며, 모델이 백도어로 작동할 사전 정의된 트리거와 생성된 트리거 조합에 대해 평가되었습니다.



### MSV-Mamba: A Multiscale Vision Mamba Network for Echocardiography Segmentation (https://arxiv.org/abs/2501.07120)
- **What's New**: 이 논문은 U-형 태깅 모델을 소개하여 심초음파 (echocardiography) 분할에 필요한 고급 기능 융합 기술을 적용합니다. Mamba 모듈을 포함한 대형 창 크기 (large-window Mamba scale, LMS) 특성 융합은 복잡한 해부학적 구조에도 높은 정확도를 제공합니다. 새로운 방식으로 각 디코더 레이어에서 보조 손실 (auxiliary losses)을 도입하여 각 레이어의 특성이 최종 세그멘테이션에 더 큰 기여를 하도록 하는 방법론을 개발하였습니다.

- **Technical Details**: 제안된 모델은 카스케이드 잔차 블록 (cascaded residual blocks)을 인코더로 사용하며, 이는 다중 스케일의 세부 특성을 추출하는 데 도움을 줍니다. 디코더에는 글로벌 정보 캡처를 돕기 위해 대형 창 Mamba 모듈이 통합되어 있으며, 이론적으로 선형 복잡도로 최대한 많은 정보를 캡처할 수 있습니다. 각 디코더 출력 레이어에 대해 보조 손실 함수가 설계되어 각 레이어에서 학습된 특성이 최종적인 세그멘테이션 결과에 긍정적인 영향을 미치도록 합니다.

- **Performance Highlights**: 실험 결과, 이 모델은 EchoNet-Dynamic 및 CAMUS 데이터셋을 사용하여 정확도와 강인성 모두에서 다른 방법들을 초과했습니다. 좌심실 내막 (${LV}_{endo}$) 분할의 경우 최적값이 각각 95.01 및 93.36에 달하며, 좌심실 외막 (${LV}_{epi}$)에서는 각각 87.35 및 87.80의 값을 기록했습니다. 이는 최상 성능 모델에 비해 약 0.54에서 1.11의 개선을 보입니다.



### Detection of AI Deepfake and Fraud in Online Payments Using GAN-Based Models (https://arxiv.org/abs/2501.07033)
Comments:
          The paper will be published and indexed by IEEE at 2025 8th International Conference on Advanced Algorithms and Control Engineering (ICAACE 2025)

- **What's New**: 이번 연구에서는 Generative Adversarial Networks (GANs)를 활용하여 AI 딥페이크 및 온라인 결제 시스템의 사기 행위를 감지하는 새로운 방법을 탐구합니다. 딥페이크 기술의 확산으로 인해 온라인 거래에서의 사기 가능성이 증가하고 있으며, 전통적인 보안 시스템은 이러한 복잡한 사기 형태를 식별하는 데 어려움을 겪고 있습니다.

- **Technical Details**: 이 연구는 StyleGAN과 DeepFake와 같은 고도화된 GAN 아키텍처를 이용하여 실제 온라인 결제 이미지와 생성된 딥페이크 이미지로 구성된 데이터셋을 기반으로 모델을 훈련시켰습니다. 제안된 GAN 기반 모델은 결제 이미지 내의 미세한 조작을 식별함으로써 온라인 결제 보안을 강화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 합법적인 거래와 딥페이크를 95% 이상의 높은 감지율로 정확하게 구별할 수 있음을 보여주었습니다. 이는 AI 기반 사기에 대한 결제 시스템의 강인성을 상당히 향상시키며, 금융 서비스에서의 사기 감지에 대한 GAN의 적용을 탐구하는 데 기여합니다.



### A Multi-Modal Deep Learning Framework for Pan-Cancer Prognosis (https://arxiv.org/abs/2501.07016)
- **What's New**: 이 논문에서는 UMPSNet이라는 새로운 심층 학습 기반의 모델을 제안합니다. 기존의 예후(prognostic) 모델이 특정 데이터셋에 대해서만 활용되었던 반면, UMPSNet은 환자의 다양한 상태를 포괄적으로 이해하기 위해 설계되었습니다. 특히, 병리 이미지(histopathology images)와 유전자 발현 검사를 통합하는 것은 물론, 인구통계학적 정보, 암 종류, 치료 프로토콜, 진단 결과와 같은 네 가지 유형의 메타 데이터(meta data)를 활용하여 텍스트 템플릿을 생성합니다.

- **Technical Details**: UMPSNet은 각 유형의 데이터(모달 데이터)를 효과적으로 통합하기 위해 최적 운송(Optimal Transport, OT) 기반의 주의(attention) 메커니즘을 사용합니다. 또한, 가이드된 소프트 전문가 조합(Guided Mixture of Experts, GMoE) 메커니즘을 통해 여러 암 데이터셋 간의 분포 차이를 해결합니다. 이러한 기법들을 사용하여 환자 데이터의 다중 모달리티(multi-modality)를 통합하고 공동 훈련(joint training)을 수행하였습니다.

- **Performance Highlights**: UMPSNet은 모든 최신 기술(State-of-the-Art, SOTA) 방법들을 초월하는 성능을 발휘했습니다. 이 모델은 다수의 암 종류를 위한 단일 모델만으로도 효과성과 일반화 능력(generalization ability)을 입증하였습니다. 논문에서 제공한 코드로 인해 연구자들은 UMPSNet의 구현을 보다 쉽게 접근할 수 있게 되었습니다.



### Comparison of Autoencoders for tokenization of ASL datasets (https://arxiv.org/abs/2501.06942)
Comments:
          9 pages, 2 tables, 4 figures

- **What's New**: 이번 연구는 미국 수화(American Sign Language; ASL) 이미지 데이터셋을 위해 인코더-디코더(encoder-decoder) 아키텍처를 개발하고 평가하는 데 중점을 두고 있습니다. 해당 데이터셋은 29개의 손 동작 클래스에 대해 87,000장의 이미지를 포함하고 있으며, 입체적인 재구성을 통해 청각 장애인과 디지털 시스템 간의 효과적인 의사소통을 가능하게 하는 모델을 목표로 합니다. 다양한 인코더-디코더 기술을 비교하여 결과적으로 확산 오토인코더(Diffusion Autoencoder)가 가장 높은 성능을 발휘함을 확립했습니다.

- **Technical Details**: 연구에서는 피드포워드 오토인코더(Feedforward Autoencoder), 합성곱 오토인코더(Convolutional Autoencoder), 확산 오토인코더(Diffusion Autoencoder) 등 세 가지 기법을 사용하여 ASL 이미지 인코딩 및 디코딩을 수행했습니다. 특히 확산 오토인코더는 확률적 노이즈 모델링과 반복적인 디노이징을 통해 이미지 재구성에서 최상의 결과를 보였으며, 이는 다중 모달 AI(multi-modal AI) 적용에 중요한 역할을 합니다. 이 모든 모델은 PyTorch를 통해 텐서 형태로 처리되며, 데이터 전처리를 통해 일관성을 확보했습니다.

- **Performance Highlights**: 성능 평가에 따르면 확산 오토인코더는 평균 제곱 오차(mean squared error; MSE) 측정에서 가장 낮은 값을 기록하며, 사용자 평가에서도 가장 우수한 평균 의견 점수(Mean Opinion Score; MOS)를 달성했습니다. 이는 해당 모델이 노이즈가 있는 상황에서도 높은 품질의 이미지를 재구성할 수 있는 능력을 보유하고 있음을 나타냅니다. 연구 결과는 ASL 인식 및 생성과 같은 다중 모달 AI 분야에서 더 나은 인코더-디코더 시스템 설계에 기여할 것으로 기대됩니다.



### Super-Resolution of 3D Micro-CT Images Using Generative Adversarial Networks: Enhancing Resolution and Segmentation Accuracy (https://arxiv.org/abs/2501.06939)
Comments:
          24 pages, 9 figures

- **What's New**: 본 연구에서는 머신 러닝(Machine Learning) 생성 모델을 활용하여 암석의 3D 마이크로 전산 단층 촬영(micro-CT) 이미지 품질을 크게 향상시키는 절차를 개발했습니다. 제안된 모델은 해상도를 8배(8x) 향상시키며, 서로 다른 암석 광물 및 상에 대한 중첩된 X-ray 감쇠로 인한 세분화 불일치를 해결합니다.

- **Technical Details**: 이 생성 모델은 그래디언트 패널티(Gradient Penalty)를 포함한 3D 깊이 합성곱 와서슈타인 생성 적대 신경망(3D DC WGAN-GP)입니다. 이 알고리즘은 세분화된 3D 저해상도 micro-CT 이미지와 세분화된 쌍이 없는 고해상도 레이저 스캐닝 현미경(LSM) 이미지를 기반으로 훈련되었습니다. 여러 견본 샘플에 대해 알고리즘을 검증하였습니다.

- **Performance Highlights**: 최종적으로 우리는 0.4375 마이크로 미터/부피(voxel)의 해상도를 가진 고품질의 초해상도 3D 이미지를 달성했으며, 구성 광물과 기공 공간에 대한 정확한 세분화를 구현하였습니다. 이 절차는 디지털 암석 물리학의 현대적 역량을 크게 확장할 수 있는 가능성을 보여줍니다.



### Driver Age and Its Effect on Key Driving Metrics: Insights from Dynamic Vehicle Data (https://arxiv.org/abs/2501.06918)
Comments:
          21 pages, 9 figures, 4 Tables, 104th TRB Annual Meeting 2025, Washington DC

- **What's New**: 이 연구는 2030년까지 65세 이상 고령 인구의 50% 증가를 예측하며, 이로 인해 도로에서의 고령 운전자가 늘어난다는 점을 강조합니다. 고령 운전자의 사고 사망률이 젊은 운전자의 경우보다 높다는 사실은 보다 효과적인 안전 개입의 필요성을 부각시킵니다. 또한, 이 연구는 나이와 관련된 행동 변화가 실제 주행 시나리오에 어떻게 나타나는지를 분석합니다.

- **Technical Details**: 자연주의적 주행 데이터(Naturalistic Driving Data, NDD)를 활용하여, 인터스테이트에서의 속도 제한 준수 및 정지 교차로에서의 감속과 같은 주행 성능 지표를 분석했습니다. 이 과정에서 누적 분포 함수(Cumulative Distribution Functions, CDF)를 사용하여, 고령 운전자와 젊은 운전자의 핵심 주행 행동에 대한 기준점을 설정하였습니다. 이상 탐지(anomaly detection) 및 기준 비교를 포함한 분석을 통해, 75mph에서의 속도 제한 준수와 관련된 주행 패턴에서 유의미한 차이를 발견하였습니다.

- **Performance Highlights**: 이 연구는 고령 운전자의 속도 제한 준수 행동에 기반한 맞춤형 개입을 통해 첨단 운전 보조 시스템(Advanced Driver Assistance Systems, ADAS)을 향상시킬 수 있는 잠재력을 보여줍니다. 그러나 다른 주행 행동에 대한 메트릭을 개선하고 검증하기 위한 추가 데이터의 필요성이 강조됩니다. 이를 통해 ADAS는 급격한 감속과 같은 이상징후를 효과적으로 감지하여 주행 중 저하가 일어났거나 다른 안전 문제를 나타내는 신호로 인식할 수 있습니다.



### Real-Time Neural-Enhancement for Online Cloud Gaming (https://arxiv.org/abs/2501.06880)
- **What's New**: 이번 논문에서는 온라인 클라우드 게임의 비디오 전송 품질을 향상시키기 위한 새로운 프레임워크인 River를 소개합니다. River는 게임 비디오에서 반복적이고 중복적인 세그먼트 특징을 활용하여, 사전 조정된 super-resolution (SR) 모델의 재사용 가능성을 극대화합니다. 이를 통해 기존 기법에 비해 훈련 지연을 수 분에서 밀리초로 단축시킬 수 있습니다.

- **Technical Details**: River는 콘텐츠 인식 인코더를 구축하여 다양한 비디오 세그먼트에 대해 SR 모델을 미세 조정하고 이를 검색 테이블에 저장합니다. 클라우드 게임 비디오 스트리밍 중에는 동영상의 특징을 검사하고, 관련 있는 SR 모델을 검색하여 프레임 품질을 향상시킵니다. 또한, 클라이언트로 모델 가중치를 스트리밍하는 오버헤드를 피하기 위해 예측 전략을 설계합니다.

- **Performance Highlights**: 실제 비디오 게임 스트리밍 기반 평가에서, River는 중복 훈련 오버헤드를 44% 줄이고, Peak-Signal-to-Noise-Ratio (PSNR)을 평균 1.81dB 향상시켰습니다. 실제 배포에서는 River가 실시간 요구사항을 충족하며, 모바일 장치에서 약 720p 20fps의 성능을 달성했습니다.



### Defect Detection Network In PCB Circuit Devices Based on GAN Enhanced YOLOv11 (https://arxiv.org/abs/2501.06879)
- **What's New**: 이번 연구는 생성적 적대 신경망(GAN)으로 개선된 YOLOv11 모델을 활용하여 인쇄 회로 기판(PCB)에서의 표면 결함 탐지 방법을 제안합니다. 연구는 결함 유형으로 결핍 구멍(missing hole), 쥐 물림(rat bite), 개방 회로(open circuit), 단락(short circuit), 버(burr), 가상 용접(virtual welding) 등 여섯 가지를 집중적으로 분석합니다.

- **Technical Details**: GAN을 사용해 합성 결함 이미지를 생성함으로써 데이터셋을 다양하고 현실감 있게 확장하여 모델의 일반화 능력을 향상시킵니다. 특히 버(burr)와 같은 복잡하거나 드문 결함들에 대한 탐지 성능이 강화됨에 따라, 향상된 YOLOv11 모델의 PCB 결함 데이터셋 검증에서 정확도(accuracy), 재현율(recall), 강인성(robustness)이 크게 개선되었습니다.

- **Performance Highlights**: 연구 결과는 복잡한 환경이나 작은 목표물에 대한 결함 처리에서 눈에 띄는 성과를 보여줍니다. 전자 설계 자동화(EDA) 분야의 효율적인 결함 탐지는 고품질 PCB 제조를 보장하는 중요한 단계라는 점에서, GAN 기반 데이터 증대 및 최적화된 탐지 아키텍처의 중요성을 강조합니다.



### A Foundational Generative Model for Breast Ultrasound Image Analysis (https://arxiv.org/abs/2501.06869)
Comments:
          Peking University; Stanford University; Peking University Cancer Hospital & Institute; Peking Union Medical College Hospital; Cancer Hospital, Chinese Academy of Medical Sciences

- **What's New**: 이번 논문에서는 브레스트 초음파(image analysis) 분석을 위해 특별히 설계된 BUSGen이라는 첫 번째 기초 생성 모델(generative model)을 소개합니다. 기존의 기초 모델들이 여러 임상 작업에 사용되는 것과 달리, BUSGen은 350만 개 이상의 브레스트 초음파 이미지를 기반으로 사전 학습(pretrained)되어 브레스트 구조 및 병리학적 특성에 대한 방대한 지식을 축적했습니다. 이러한 발전은 브레스트 초음파 분석 분야에서의 잠재력을 실현하는 중요한 진전을 나타냅니다.

- **Technical Details**: BUSGen은 few-shot adaptation 방식을 통해 실제적이고 정보가 풍부한 작업 특화 데이터의 저장소를 생성할 수 있습니다. 이를 통해 다양한 하위 작업(downstream tasks)을 위한 모델 개발을 가속화할 수 있습니다. 연구 결과에 따르면, BUSGen은 브레스트 암 검진(screening), 진단(diagnosis), 예후(prognosis) 측면에서 실제 데이터 기반 기초 모델보다 현저히 뛰어난 적응력을 보여주었습니다. 또한, 생성된 데이터의 스케일링 효과(scaling effect)가 수집된 실제 데이터만큼 효과적임을 입증했습니다.

- **Performance Highlights**: 특히, BUSGen은 브레스트 암 조기 진단에서 9명의 인증된 방사선의사(board-certified radiologists)를 초과 성과를 기록하며 평균 민감도(sensitivity)를 16.5% 향상시켰습니다(P-value<0.0001). 또한, 연구를 통해 하위 모델의 일반화 능력(generalization ability)을 개선하며 환자의 프라이버시를 보호하는 Fully de-identified data 공유를 가능하게 했습니다. BUSGen의 온라인 데모는 제공되는 링크를 통해 확인할 수 있습니다.



### A General Framework for Inference-time Scaling and Steering of Diffusion Models (https://arxiv.org/abs/2501.06848)
- **What's New**: 이번 연구에서는 Feynman Kac (FK) steering을 제안합니다. 이는 사용자 지정 특성을 가진 샘플을 생성하는 데 어려움을 극복하기 위한 인퍼런스 타임 프레임워크입니다. FK steering은 여러 개의 상호작용하는 확산 프로세스인 입자(particles)를 샘플링하고 희소 이벤트 시뮬레이션 방법을 통해 높은 보상을 제공하는 샘플을 생성할 수 있도록 합니다.

- **Technical Details**: FK steering은 보상 함수에 따라 입자를 재샘플링하는 메커니즘을 가지고 있습니다. 이 과정에서, 잠재적 함수(potentials)는 중간 상태에 대한 보상을 기반으로 정의되며, 높은 값을 가진 입자는 더 높은 보상 샘플을 생성할 가능성이 높습니다. 이 프레임워크를 통해 사용자들은 다양한 잠재적 함수와 보상 모델을 선택해 성능을 조정할 수 있습니다.

- **Performance Highlights**: FK steering은 텍스트-이미지 및 텍스트 확산 모델에서 높은 성과를 보여주었습니다. 특히, 0.8B 파라미터 모델이 2.6B 파라미터로 파인 튜닝된 모델보다 프롬프트 충실도에서 우수한 성과를 보였으며, 샘플링 속도는 더 빠르고 추가 교육이 필요하지 않았습니다. FK steering은 더 작은 모델도 더 큰 모델보다 성능이 뛰어난 성과를 보여주며, 품질 제어 및 속성 조정의 용이성을 제공합니다.



### Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution (https://arxiv.org/abs/2501.06838)
- **What's New**: 본 논문에서는 Gaussian Splatting (GS) 기법을 활용하여 Arbitrary-scale Super-Resolution (ASR) 문제를 해결하는 새로운 접근을 제시합니다. 기존 Implicit Neural Representation (INR) 기반 모델들이 가진 한계를 극복하기 위해, GSASR이라는 네트워크를 설계하여 GS를 ASR에 적용했습니다. 이를 통해 입력 저해상도 이미지의 고해상도 변환을 효율적으로 수행할 수 있게 되었습니다.

- **Technical Details**: GSASR은 저해상도 이미지를 입력받아 이에 대한 2D Gaussians를 예측하는 구조를 갖추고 있습니다. 이 구조는 Self-Attention 메커니즘을 통해 깊은 특성 사이의 상관관계를 파악합니다. 또한, 예측된 Gaussians를 기반으로 효율적인 GPU/CUDA 기반의 Rasterization 기술을 사용하여 슈퍼해상도 이미지를 생성합니다.

- **Performance Highlights**: GSASR의 성능은 기존 최첨단 기술과 비교하여 빠른 속도로 슈퍼해상도 이미지를 생성하는 데에서 두드러집니다. 예를 들어, 스케일 팩터가 ×12일 때 GSASR은 720×720 해상도의 이미지를 단 91ms만에 생성할 수 있습니다. 반면에, 기존 INR 기반 모델인 CiaoSR은 같은 해상도를 생성하는 데 약 540ms가 소요됩니다.



### Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models (https://arxiv.org/abs/2501.06751)
- **What's New**: 최근 연구에서는 Text-to-Image(T2I) diffusion 모델에서 패딩 토큰(padding tokens)의 역할을 처음으로 심도 있게 분석하였습니다. 기존의 관행에 따르면 입력 프롬프트는 고정 길이로 패딩됩니다. 그러나 패딩 토큰이 이미지 생성 과정에 미치는 영향은 과거에 비활성화되었으며, 이 연구는 패딩 토큰이 T2I 모델의 여러 구성요소에 어떻게 작용하는지를 설명하는 두 가지 인과 기법을 제안합니다.

- **Technical Details**: 제안된 방법론은 Intervention in the Text Encoder Output(ITE)와 Intervention in the Diffusion Process(IDP)를 포함합니다. 두 방법 모두 특정 입력이나 중간 표현에 변화를 주어 출력에 미치는 영향을 관찰하는 인과 매개 분석(causal mediation analysis)에 기반하고 있습니다. ITE에서는 텍스트 인코더의 출력에서 특정 구간을 선택적으로 교란시켜 패딩 토큰과 원래 프롬프트 토큰의 기여도를 차별화하고, IDP에서는 확산 과정에서 패딩 토큰이 모델에 의해 어떻게 사용되는지를 분석합니다.

- **Performance Highlights**: 연구 결과, T2I 모델이 동결된 텍스트 인코더를 사용할 경우 패딩 토큰이 무시된다는 사실이 밝혀졌습니다. 그러나 텍스트 인코더가 훈련되거나 미세 조정된 경우, 패딩 토큰은 의미론적인 중요성을 가집니다. 또한, 특정 확산 모델 구조에서 패딩 토큰은 정보 저장 역할을 수행하며, 이는 언어 모델과 비전-언어 모델에서 발견된 바와 유사합니다.



### Wavelet Integrated Convolutional Neural Network for ECG Signal Denoising (https://arxiv.org/abs/2501.06724)
- **What's New**: 이번 연구에서는 착용형 심전도(ECG) 측정 시 발생하는 고강도 노이즈 왜곡 문제를 해결하기 위한 효과적인 노이즈 감소 방법을 제안합니다. 기존의 방법들이 ECG와 노이즈의 주파수 대역이 겹치는 문제로 인해 효과를 보지 못하는 점을 개선하기 위해, 특정 주파수 특성을 추출하는 추가적인 wavelet 변환 계층이 포함된 CNN 모델을 도입하였습니다. 이 연구는 다양한 주파수 대역을 고려하여 정확한 ECG 동작 예측이 가능함을 확인했습니다.

- **Technical Details**: 제안된 방법은 CNN(Convolutional Neural Network) 모델과 wavelet 변환을 결합하여 고주파 및 저주파 성분을 분리하는 메커니즘을 갖추고 있습니다. 이러한 구조를 통해 주파수 대역이 겹치는 상황에서도 정밀한 ECG 신호 복원 및 노이즈 감소가 가능하게 됩니다. 실험에서는 신호 대 잡음 비율(SNR)이 -10에서 10 사이의 잡음 신호를 평가하며, 제안된 방법의 효율성이 높음을 입증했습니다.

- **Performance Highlights**: 실험 결과, SNR이 낮을 때 제안된 방법의 노이즈 감소 효율성이 더욱 뛰어난 것으로 나타났습니다. 고강도 노이즈가 영향을 미치는 경우에도 ECG 신호의 성능을 개선할 수 있는 가능성을 보여주었습니다. 이는 심혈관 질환 조기 진단 및 관리에 기여할 것으로 기대됩니다.



### TWIX: Automatically Reconstructing Structured Data from Templatized Documents (https://arxiv.org/abs/2501.06659)
- **What's New**: 본 논문에서는 TWIX라는 새로운 도구를 소개하여 복잡한 레이아웃을 가진 templatized documents에서의 데이터 추출 문제를 해결합니다. TWIX는 문서를 생성하는 데 사용된 기본 템플릿을 예측하여, 이를 기반으로 데이터를 추출하여 더 정확하고 효율적인 결과를 제공합니다. 34개 다양한 실제 데이터셋에서 TWIX는 평균 90% 이상의 정확도(precision)와 재현율(recall)을 기록하였으며, 이는 기존 도구들보다 25% 이상 향상된 수치입니다.

- **Technical Details**: TWIX는 templatized documents에서 데이터 추출을 위해 먼저 같은 템플릿을 사용해 생성된 문서 세트를 활용하여 템플릿을 재구성합니다. 그리고 나서 해당 템플릿을 기반으로 문서에서 데이터를 추출합니다. 이 과정에서 TWIX는 문서 내에서 시각적 구조와 필드 간의 관계를 이해하며, 다양한 형식(예: 테이블, key-value 쌍)의 정보를 효과적으로 추출합니다.

- **Performance Highlights**: TWIX는 대규모 데이터셋에서도 쉽게 확장 가능하며, 0.017초의 빠른 속도로 대량의 문서에서 데이터를 추출할 수 있습니다. 이는 시각 기반 LLMs(예: GPT-4-Vision)와 비교할 때 734배 빠르며, 비용은 5836배 저렴합니다. 이러한 성능을 통해 TWIX는 데이터 추출 비용과 시간을 대폭 줄일 수 있는 잠재력을 지니고 있습니다.



### Personalized Preference Fine-tuning of Diffusion Models (https://arxiv.org/abs/2501.06655)
- **What's New**: 이 논문에서는 기존의 RLHF (Reinforcement Learning from Human Feedback) 기법들로 인해 개인화된 선호를 고려하지 못하는 문제를 해결하기 위해 PPD (Personalized Preference Diffusion)를 소개합니다. PPD는 다중 보상 최적화 목적을 통해 다수의 사용자의 개별 선호에 맞춰 확산 모델을 조정할 수 있습니다. 이를 통해 사용자가 제공하는 최소한의 샘플로도 성능을 극대화하며, 새로운 사용자에게 일반화할 수 있는 능력을 갖추게 됩니다.

- **Technical Details**: PPD는 비전-언어 모델(Vision-Language Model, VLM)을 활용하여 사용자 개별의 선호 임베딩을 몇 가지 쌍별 선호 예시에서 추출합니다. 이후 이 임베딩을 확산 모델에 통합하여 크로스 어텐션(Cross Attention) 기법을 통해 모델을 미세 조정합니다. DPO의 변형을 채택하여 사용자 선호에 따라 텍스트-이미지 모델을 최적화함으로써, 여러 사용자 선호와 일치하도록 훈련됩니다.

- **Performance Highlights**: PPD의 실험 결과는 여러 보상 함수(CLP, Aesthetic 등)에 대해 효과적으로 최적화하며, 추론 중 보상 사이의 원활한 보간(interpolation)을 보여줍니다. 실제 사용자 시나리오에서 PPD는 새로운 사용자로부터 제공된 최소 4개의 선호 예시로도 그 사용자의 특정 선호에 맞춘 이미지를 생성하며, Stable Cascade에 비해 평균 76%의 승률을 기록하는 성과를 보였습니다.



### CNN-powered micro- to macro-scale flow modeling in deformable porous media (https://arxiv.org/abs/2501.06466)
Comments:
          21 pages, 12 figures, research paper

- **What's New**: 이 논문은 변형 가능한 다공성 매체에서 미세 CT 이미지 세트를 활용하여 거시적 고유 투과율 텐서를 예측하는 새로운 기법을 제안합니다. 기존의 투과율 추정 기법들의 한계를 극복하기 위해, 딥러닝 기술인 Convolutional Neural Networks (CNN)을 사용하여 변형 및 비등방 유동 조건에서의 유체 흐름 특성을 예측하는 방법론을 소개하고 있습니다. 이 approaches은 딥러닝을 활용하여 다공성 미세구조의 CT 이미지를 입력으로 받고, 대칭 2차 고유 투과율 텐서를 예측하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 방법론은 4단계로 구성됩니다: 1) 다양한 체적 변형 수준에서 Bentheim sandstone의 CT 이미지 데이터세트를 구축; 2) 단상 유동의 pore-scale 시뮬레이션 수행; 3) 처리된 CT 이미지를 입력으로 CNN 모델 학습; 4) 데이터 증강 및 대체 CNN 아키텍처를 활용하여 모델 일반화 개선. 이 과정에서 lattice Boltzmann method (LBM)와 같은 고급 시뮬레이션 기법을 활용하여 투과율 데이터를 생성합니다.

- **Performance Highlights**: CNN 기반 접근법은 시뮬레이션 요약 모델들이 좋은 성능을 발휘하는 데 필요한 데이터 양이 충분하다는 데 기반을 두고 있으나, 데이터가 부족할 경우에는 데이터 증강 기법이 도움이 될 수 있습니다. 예를 들어, GAN을 활용하여 합성 데이터의 생성을 통해 일반 패턴을 포착하고, 이후 Transfer Learning 기법을 통해 기존 데이터셋에 대한 모델 성능을 향상시키는 방법도 논의됩니다. 이러한 접근법들은 지반공학, 수문학, 재료 과학 등 다양한 분야에서 중요한 고유 투과율 텐서를 예측하는 데 기여할 수 있습니다.



### Towards Robust Nonlinear Subspace Clustering: A Kernel Learning Approach (https://arxiv.org/abs/2501.06368)
- **What's New**: 이번 논문에서는 비선형 서브스페이스 클러스터링을 위한 새로운 패러다임인 DKLM을 제안합니다. 기존의 방법들이 고정된 커널에 의존하던 반면, DKLM은 데이터의 자기 표현에서 직접 커널을 학습하여 적응형 가중치와 삼각 부등식 제약 조건을 만족합니다. 이로써 비선형 공간에서 데이터의 지역 매니폴드 구조를 보존하고 최적의 블록 대각 행렬을 형성하는데 기여합니다.

- **Technical Details**: DKLM은 특정 조건 하에 자기 표현을 통해 커널을 학습하며, 이는 적응형 가중치 커널 학습으로 볼 수 있습니다. 이 접근법은 데이터 포인트가 속하는 서브스페이스를 효과적으로 찾는 방법을 제시하며, 자기 표현 기반의 서브스페이스 클러스터링 프레임워크 내에서 구현될 수 있습니다. 이론적 분석을 통해 DKLM이 기존 방법과의 관계를 설명하고, 다양한 데이터 세트에 대한 실험을 통해 그 효과를 입증하였습니다.

- **Performance Highlights**: DKLM은 첨단의 자기 표현 및 커널 기반 서브스페이스 클러스터링 방식에 비해 우수한 클러스터링 성능을 보여줍니다. 특히, 네 개의 합성 데이터 세트와 아홉 개의 실제 데이터 세트에서 우리의 접근법의 정확성과 강인성을 확인하였으며, 고차원 시계열 데이터에 대한 실험에서도 비선형 패턴 발견의 효과성을 입증하였습니다.



### Ultrasound Image Synthesis Using Generative AI for Lung Ultrasound Detection (https://arxiv.org/abs/2501.06356)
Comments:
          Accepted by ISBI 2025

- **What's New**: DiffUltra는 대표적이고 다양한 데이터로 훈련된 헬스케어 AI 모델을 개발하기 위해 고안된 최신 생성형 AI 기법입니다. 이 방법은 실제 환자 데이터에서 추출한 병변의 구조적 및 위치적 특성을 활용하여 실제적인 Lung Ultrasound (LUS) 이미지를 합성합니다. DiffUltra는 데이터 다양성과 드문 사례의 발생 빈도를 높여주어 모델의 정확성을 향상시키는 데 기여합니다.

- **Technical Details**: DiffUltra는 레지온-해부학 뱅크를 사용하여 병변을 해부학적으로 적절한 위치에 배치합니다. 이 병귀 관리를 위한 확률 질량 함수(PMF)를 사용하여 병변과 주변 해부학적 구조 간의 상대적 거리를 모델링합니다. 이 기술은 단순히 병변의 픽셀 수준 세분화를 요구하지 않고 전반적인 LUS 이미지를 합성하여 효율적이고 사실적인 결과를 제공합니다.

- **Performance Highlights**: DiffUltra는 실제 환자 데이터만으로 훈련된 모델과 비교하여 폐의 응집체 탐지를 5.6% 향상시킵니다. 특히, 이 방법은 드문 사례(예: 1등급 및 4등급의 폐 응집체)의 탐지 성능을 25%까지 향상시킵니다. DiffUltra의 도입으로 인해 데이터 합성의 정밀도와 자연스러움이 크게 증가하여 헬스케어 AI 모델의 신뢰성이 높아졌습니다.



### Underwater Image Enhancement using Generative Adversarial Networks: A Survey (https://arxiv.org/abs/2501.06273)
Comments:
          15 pages, 7 figures, 2 tables

- **What's New**: 최근 몇 년 간 GAN(Generative Adversarial Networks)을 이용한 수중 이미지 향상 연구가 활발히 진행되고 있습니다. 이는 수중 환경에서 발생하는 빛 감쇠, 산란, 색 왜곡과 같은 문제를 해결하기 위한 필요에서 비롯되었습니다. 이 논문에서는 물리적 모델, 물리적 비모델, CNN(Convolutional Neural Network) 기반 모델 및 최첨단 GAN 기반 방법까지 다양한 수중 이미지 향상 접근법을 탐구합니다.

- **Technical Details**: 이 논문은 수중 이미지 향상 방법들을 포괄적으로 분석합니다. 각각의 방법들은 평가 지표, 데이터셋, 손실 함수(loss functions)를 포함하여 전체적인 관점을 제공합니다. GAN은 복잡한 변환을 학습할 수 있는 능력 덕분에 수중 사진을 향상시키는 데 강력한 도구로 자리 잡았습니다.

- **Performance Highlights**: 이 연구는 해양 생물학, 생태계 모니터링, 산호초 건강 평가, 수중 고고학, 자율 수중 차량(AUV) 내비게이션 등에서의 실제 응용 경험을 반영합니다. 그러나 현재 방법들은 일반화 문제, 높은 계산 요구 사항, 데이터셋 편향과 같은 한계와 도전 과제를 전해주고 있으며, 향후 연구 방향에 대한 제안 또한 포함되어 있습니다.



### Interpretable Auto Window Setting for Deep-Learning-Based CT Analysis (https://arxiv.org/abs/2501.06223)
- **What's New**: 이번 연구에서는 의료 영상 분석에서 중요한 CT의 Auto Window Setting을 위한 새로운 플러그앤플레이 모듈을 제안합니다. 이 모듈은 Tanh 활성화 함수에서 유래하며, 주요 딥러닝 아키텍처와 호환됩니다. 기존의 수동적인 창 설정 방법을 대체할 수 있는 자동화된 접근 방식으로, 임상적으로 직관적인 관점에서 결과를 해석할 수 있는 기능을 제공합니다.

- **Technical Details**: 제안된 모듈은 두 가지 주요 설계 원칙에 따라 구성됩니다. 첫째, 도메인 불변(design)을 기반으로 하여 다양한 HU(hounsfield unit) 값을 동적으로 처리할 수 있습니다. 둘째, 각 서브 모듈이 독립적으로 분석 및 최적화될 수 있는 구조로 되어 있습니다. 이로 인해 사용자는 모델 조정에 더 많은 유연성을 갖게 되며, 명료한 분석 및 관찰 통찰을 제공합니다.

- **Performance Highlights**: 모듈의 효과는 여러 개방형 데이터셋에서 검증되었으며, 하드 세그먼트 타겟에서 10%에서 200%까지의 Dice score 개선을 기록했습니다. 이는 임상 현장에서 CT 이미지 분석을 수행할 때 보다 나은 성능을 제공할 것으로 기대됩니다. 특히, 자동화된 방식으로 인해 수동 설정의 변수를 줄이고 더욱 신뢰할 수 있는 결과를 도출할 수 있습니다.



### Powerful Design of Small Vision Transformer on CIFAR10 (https://arxiv.org/abs/2501.06220)
- **What's New**: 본 논문에서는 작은 데이터셋에서 Vision Transformers (ViTs)의 성능을 개선하기 위해 Tiny ViTs의 설계와 최적화에 대해 다룹니다. CIFAR-10을 벤치마크로 활용하여 데이터 증강(data augmentation), 패치 토큰 초기화(patch token initialization), 저랭크 압축(low-rank compression), 다중 클래스 토큰 전략(multi-class token strategies)의 영향을 체계적으로 평가하였습니다. 저랭크 압축이 성능 손실을 최소화하며, 여러 CLS 토큰을 도입함으로써 모델의 전반적인 표현 능력이 향상된다고 보고합니다.

- **Technical Details**: ViT는 이미지 토큰화(image tokenization), 토큰 변환(token transformation), 과제 투영(task projection)의 주요 절차를 통해 작동합니다. 이미지가 여러 패치로 나눠지고, 각 패치 토큰은 특정 위치의 의미 정보를 나타냅니다. Attention 메커니즘을 통해 각 패치의 중요성을 상대적으로 평가하며, 다중 헤드 주의(multi-head attention)를 사용하여 다양한 관계를 포착합니다.

- **Performance Highlights**: 실험 결과, 저랭크 압축을 사용한 Multi-Head Latent Attention (MLA)은 ViTs에서 다소 중복된 성능을 보여 성능 손실이 거의 없음을 나타냅니다. 또한, 여러 CLS 토큰을 도입함으로써 더 나은 정확도를 달성할 수 있음을 발견했습니다. 이를 통해 Tiny ViTs의 최적화를 위한 포괄적인 프레임워크를 제공하며, 효율적이고 효과적인 설계에 대한 실용적인 통찰을 제시합니다.



New uploads on arXiv(cs.AI)

### ADAM-1: AI and Bioinformatics for Alzheimer's Detection and Microbiome-Clinical Data Integrations (https://arxiv.org/abs/2501.08324)
Comments:
          16 pages, 16 figures

- **What's New**: ADAM-1(Alzheimer's Disease Analysis Model Generation 1)은 다중 에이전트 대형 언어 모델(LLM) 프레임워크로, 마이크로바이옴 프로필, 임상 데이터셋 및 외부 지식 기반을 통합하여 알츠하이머병(AD)의 이해와 탐지를 향상시키기 위해 설계되었습니다. 이 모델은 retrieval-augmented generation (RAG) 기법과 다중 에이전트 아키텍처를 활용하여 다양한 데이터 출처로부터 통찰력을 합성하고 문헌 기반 증거를 통해 발견 사항을 맥락화합니다. ADAM-1은 XGBoost와의 비교 평가에서 유사한 평균 F1 점수를 나타내면서도 변동성을 크게 줄여 강건성과 일관성을 강조합니다.

- **Technical Details**: ADAM은 다중 모드 데이터 세트를 통합하여 알츠하이머병 상태의 이진 분류를 수행하도록 설계된 프레임워크입니다. 이 모델은 임상 데이터 및 마이크로바이옴 데이터를 포함한 모드 데이터와 RAG를 통한 문헌에서 나오는 기존 지식을 결합합니다. ADAM은 무작위 샘플링을 통해 제어된 노이즈를 도입하고, 그 분석 방법의 강건성을 엄격히 측정하여 F1 변동성을 줄이는 동시에 전통적인 머신러닝 모델과 유사한 평균 테스트 정확도를 유지합니다.

- **Performance Highlights**: ADAM은 알츠하이머병 관련 데이터를 분석, 요약 및 분류하는 데 효과적인 성능을 보이며, 특히 작은 실험실 데이터셋에서의 예측 안정성을 강조합니다. 이 모델은 바이너리 분류 작업에 맞춰져 있어 결과와 해석 가능성을 일관되게 제공하며, 다양한 연구 환경에 쉽게 적응할 수 있도록 최적화되어 있습니다. ADAM의 혁신적인 접근 방식은 알츠하이머병 연구에서 의미 있는 발전을 이끌어낼 것으로 기대됩니다.



### Optimization of Link Configuration for Satellite Communication Using Reinforcement Learning (https://arxiv.org/abs/2501.08220)
- **What's New**: 본 논문에서는 인공위성 전송기(transponder)의 링크 구성 최적화(link configuration optimization)를 위한 새로운 방법론을 제안하고 있습니다. 특히, 강화 학습(reinforcement learning) 알고리즘인 PPO(Proximal Policy Optimization)를 메타휴리스틱 방법인 시뮬레이티드 어닐링(simulated annealing)과 비교하여 실험을 수행하였습니다. 연구 결과, 시뮬레이티드 어닐링이 정적 문제에서 더 나은 성능을 보였지만, 강화 학습의 잠재력 또한 강조되었습니다.

- **Technical Details**: 강화 학습(RL)은 에이전트가 환경과 상호작용하며 최적의 정책(policy)을 학습하는 과정입니다. 본 논문에서는 인공위성 전송기의 동적 링크 구성을 모델링하기 위해 RL의 적용 가능성을 평가했습니다. RL의 기반은 마르코프 결정 과정(MDP)이며, 에이전트는 상태(state)에서 다양한 행동(action)을 선택하고 보상(reward)을 받아 최적의 정책을 찾아갑니다.

- **Performance Highlights**: 실험 결과, 시뮬레이티드 어닐링이 특정 링크 구성 문제에 대해 더 우수한 성과를 나타냈습니다. 하지만 강화 학습 역시 링크 구성 최적화에 있어 상당한 가능성을 보여주었습니다. 이 연구는 인공위성 전송기에 특화된 링크 구성 최적화 문제의 해결을 위한 새로운 방향성을 제시합니다.



### PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving (https://arxiv.org/abs/2501.08192)
- **What's New**: 본 논문에서는 LLM 추론의 성능과 확장성을 향상시키기 위해 PRESERVE라는 새로운 prefetching 프레임워크를 제안합니다. 이 프레임워크는 모델 가중치와 KV-cache의 메모리 읽기를 집합 통신 작업과 겹치도록 최적화하여 전반적인 인퍼런스 속도를 높입니다. PRESERVE를 적용함으로써 오프칩 메모리에서 L2 캐시로 데이터를 미리 가져오는 방식으로 통신의 대기 시간을 숨길 수 있습니다.

- **Technical Details**: PRESERVE는 컴퓨팅 그래프에 prefetching 명령을 자동으로 삽입하는 그래프 최적화 프레임워크로 구현되어 있으며, 이를 통해 사용자의 코드 수정 없이 성능 향상을 극대화합니다. 이 방법은 L2 캐시의 크기와 메모리 대역폭을 고려하여 최적의 하드웨어 구성으로 성능을 극대화합니다. 실험을 통해 상용 AI 가속기에서 최대 1.6배의 속도 향상과 함께, 최적의 L2 캐시 크기를 고려할 때 추가적인 1.25배 성능 향상을 보여주었습니다.

- **Performance Highlights**: PRESERVE는 대규모 오픈소스 LLM에 대해 최대 1.6배의 엔드투엔드 속도 향상을 달성했습니다. 이 프레임워크는 메모리 병목현상과 통신 오버헤드를 완화함으로써 LLM 추론 시스템의 성능을 향상시키는 잠재력을 가지고 있습니다. 또한, 설계 공간 탐색을 통해 성능과 비용 대비 효율성을 동시에 고려한 최적의 하드웨어 구성을 제안합니다.



### Assessing AI Adoption and Digitalization in SMEs: A Framework for Implementation (https://arxiv.org/abs/2501.08184)
- **What's New**: 이 연구는 이탈리아의 중소기업(SMEs) 내에서 디지털화와 인공지능(AI) 통합의 현재 상태를 조사합니다. 대기업과 중소기업 간의 AI 활용에서 상당한 격차가 있으며, 중소기업은 채택에 있어 여러 장벽에 직면해 있습니다. 연구는 지능형 변환을 달성하기 위한 주요 동력 및 장애물을 식별하고, 이를 해결할 수 있는 프레임워크 모델을 제안합니다.

- **Technical Details**: 연구는 36개 기업이 참여한 설문을 통해 중소기업의 디지털화 수준과 AI 사용 현황을 평가했습니다. 연구 방법론은 비영리 기관과의 협업을 통해 수행되었으며, 기업의 특성에 따라 통계적 분석이 이루어졌습니다. 설문조사는 15개의 질문으로 구성되었고, 응답에 기반하여 디지털 성숙도 수준을 1에서 4로 평가했습니다.

- **Performance Highlights**: 연구 결과는 중소기업이 AI 채택에 있어 직면하는 많은 장벽을 드러냈습니다. 조사 대상 기업들이 AI의 이점을 알고 있음에도 불구하고 대부분의 경우 지식 부족과 비용 문제로 인해 채택이 주저되고 있다는 사실이 밝혀졌습니다. 이 연구는 AI의 성공적인 통합을 위한 실질적인 가이드를 제공하여 중소기업의 의사 결정자에게 유용한 정보로 작용할 것으로 기대됩니다.



### CG-MER: A Card Game-based Multimodal dataset for Emotion Recognition (https://arxiv.org/abs/2501.08182)
Comments:
          8 pages, 2 figures and 4 tables. Sixteenth International Conference on Machine Vision (ICMV 2023), Yerevan, Armenia

- **What's New**: 이번 논문은 감정 인식(emotion recognition)을 위해 특별히 설계된 포괄적인 프랑스어 다중 모달 데이터셋(multimodal dataset)을 소개합니다. 이 데이터셋은 얼굴 표정(facial expressions), 음성(speech), 제스처(gestures)라는 세 가지 주요 모달리티(modality)를 포함하여 감정에 대한 전체적인 관점을 제공합니다. 또한, 이 데이터셋은 자연어 처리(Natural Language Processing, NLP)와 같은 추가 모달리티를 통합할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 데이터셋은 카드 게임 세션에 참여한 20명의 참가자(여성 9명, 남성 11명)를 대상으로 구성되었습니다. 이 과정에서 참가자들은 다양한 질문에 응답하며 여러 감정을 표현하도록 유도되었습니다. 총 10회의 세션을 거쳐 감정을 표현하는 다양한 방식이 수집되었습니다.

- **Performance Highlights**: 이 데이터셋은 감정 인식 연구에 유용한 자료로 활용될 수 있으며, 인간 감정과 디지털 기술 사이의 복잡한 연결을 탐구하는 데 기여할 수 있는 길을 제공합니다. 이를 통해 감정 컴퓨팅 분야에서의 새로운 연구 가능성이 열릴 것으로 기대됩니다.



### LeapVAD: A Leap in Autonomous Driving via Cognitive Perception and Dual-Process Thinking (https://arxiv.org/abs/2501.08168)
- **What's New**: LeapVAD는 인지적 지각 및 이원적 사고 기법을 기반으로 하는 새로운 자율 주행 시스템으로, 중요한 교통 요소에 주목하여 운전 결정을 내리고자 합니다. 이 시스템은 다양한 속성을 포함한 객체 특성을 활용하여 보다 효과적인 환경 표현을 가능하게 합니다. 또한, LeapVAD는 인간의 학습 과정을 모방한 이원적 의사 결정 모듈을 포함하여 사고 및 경험학습을 기반으로 결정을 내립니다. 결론적으로, 이 시스템은 자율 주행의 복잡한 환경을 보다 잘 대응할 수 있도록 설계되었습니다.

- **Technical Details**: LeapVAD의 핵심 구성 요소는 장면 이해를 위한 VLM(Visual Language Model)과 장면 토큰을 추출하는 Scene Encoder입니다. 이 시스템은 이원적 결정 모듈을 통해 두 가지 사고 방식인 분석 과정(System-II)과 휴리스틱 과정(System-I)을 결합하여 운전 경험을 축적하고 지식을 보완합니다. 기존의 단일 프레임 입력을 넘어 다중 프레임 입력을 지원하여 교통 객체에 대한 포괄적인 이해를 제공합니다. 또한, Scene Encoder는 예측된 운전 행동과 관련된 장면 토큰을 효과적으로 캡처하여 주목받고 있습니다.

- **Performance Highlights**: LeapVAD는 CARLA와 DriveArena 시뮬레이터에서 강력한 성능을 입증하였습니다. 특히, CARLA 시나리오의 Town05 단기 및 장기 벤치마크에서 각각 5.3%와 42.6%의 향상을 기록했습니다. 이러한 성능 향상은 학습한 지식을 메모리 뱅크에 저장하고 활용하는 데 중점을 두어 이루어졌습니다. LeapVAD는 지속적인 학습 및 도메인 적응성에서 탁월한 효과를 보여주며, 카메라만을 사용하는 기존 접근 방식보다도 우수한 성능을 발휘합니다.



### Multiple-Input Variational Auto-Encoder for Anomaly Detection in Heterogeneous Data (https://arxiv.org/abs/2501.08149)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 Anomaly Detection (AD)을 위한 새로운 신경망 모델인 Multiple-Input Auto-Encoder for AD (MIAEAD)와 Multiple-Input Variational Auto-Encoder (MIVAE)를 제안하였다. 기존 방법들이 비독립적이고 동일하게 분포되지 않은(non-IID) 데이터의 이질성으로 어려움을 겪는 반면, MIAEAD는 각 데이터 샘플의 특성 하위 집합에 대해 이상점(anomaly) 점수를 할당하여 이상점을 효과적으로 발견한다. 이 모델은 서브 인코더(sub-encoder)의 재구성 오류(reconstruction error)를 사용하여 이상점 점수를 계산한다.

- **Technical Details**: MIAEAD는 비지도학습(unsupervised learning)을 통해 모든 서브 인코더를 동시에 훈련시키며 각 특성 하위 집합의 이상점 점수를 결정한다. 반면, MIVAE는 정상(normal) 데이터를 잠재 공간(latent space)에서 모델링해 이를 기반으로 이상점을 식별할 수 있도록 설계되었다. 연구 결과, MIVAE는 정상 샘플과 이상점 간 평균 이상점 점수의 차이가 Variational Auto-Encoder (VAEAD)보다 크며, 이로 인해 더욱 높은 AUC(Area Under Curve) 값을 기록하였다.

- **Performance Highlights**: 여덟 개의 실제 이상 데이터셋에 대한 광범위한 실험을 통해 MIAEAD와 MIVAE가 기존 방법 및 최첨단 비지도 모델보다 최대 6% 높은 AUC 점수를 보였음을 입증하였다. 특히 MIAEAD와 MIVAE는 변동 계수(coefficient of variation, CV) 점수가 낮은 특성 하위 집합에 적용 시 높은 AUC 값을 나타냈다.



### In-situ graph reasoning and knowledge expansion using Graph-PReFLexOR (https://arxiv.org/abs/2501.08120)
- **What's New**: 이 논문에서는 Graph-PReFLexOR(Graph-based Preference-based Recursive Language Modeling for Exploratory Optimization of Reasoning)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 그래프 추론(graph reasoning)과 기호 추상화(symbolic abstraction)를 결합하여 도메인 지식을 동적으로 확장할 수 있도록 설계되었습니다. 또한, 강화 학습(reinforcement learning)의 영감을 받아 지식을 생산하는 구조화된 맵을 정의합니다.

- **Technical Details**: Graph-PReFLexOR는 개념을 노드(node)로, 그 관계를 엣지(edge)로 인코딩하여 계층적 추론(hierarchical inference) 및 동적 학습(adaptive learning)을 지원합니다. 이러한 구조는 카테고리 이론(category theory)에 기초하여 구성되어 있으며, 지식 그래프(knowledge graphs), 추상 패턴(abstract patterns), 최종 답변(final answers)을 생성합니다. 이는 주어진 작업(task)에 따라 지식을 수집하고 사유를 구조적으로 지도합니다.

- **Performance Highlights**: 3억 개 파라미터를 갖춘 Graph-PReFLexOR 모델의 결과는 깊은 추론(depth of reasoning) 및 적응력(adaptability)에서 우수성을 보여줍니다. 이는 다학제간 연결을 촉진하기 위한 '지식 정원 성장(knowledge garden growth)' 전략과 함께, 투명하고 다학제적 AI 기반 발견의 가능성을 강조합니다. 최종적으로 이 연구는 일반 자율적 추론 솔루션에 대한 기초를 마련합니다.



### NOMTO: Neural Operator-based symbolic Model approximaTion and discOvery (https://arxiv.org/abs/2501.08086)
- **What's New**: 이번 연구에서는 
NOMTO(Neural Operator-based symbolic Model approximaTion and discOvery)라는 새로운 비선형 상징 회귀 방법을 제안합니다. NOMTO는 수치 연산자를 활용하여 상징적 모델 발견의 범위를 확대하고, 특이점(singularities), 특별한 함수(special functions), 그리고 미분 연산자(derivatives)를 포함하는 상징 표현을 성공적으로 식별할 수 있습니다. 또한, NOMTO는 두 번째 비선형 편미분 방정식을 정확하게 재발견할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: NOMTO는 수치 연산자를 사용하여 입력 변수를 비선형 조합으로 자동으로 탐색하는 알고리즘으로, 계산 그래프를 형성하여 다양한 연산을 처리합니다. 이 방법은 상징적 조작이 포함된 연산을 효과적으로 수행하며, FNO와 CNO와 같은 서브그레인 수치 연산자가 핵심 요인으로 작용합니다. NOMTO는 입출력 관계와 별개로 최적화를 진행하며, 최적의 기호 표현을 도출하기 위해 sparsity를 적용하여 해석 가능한 구조를 형성합니다.

- **Performance Highlights**: NOMTO는 Nguyen과 Keijzer의 벤치마크 데이터셋을 사용하여 상징 회귀 작업에서 강력한 성능을 보여주었습니다. 초급 표현을 넘어 미분 연산자와 특별한 함수가 포함된 표현을 성공적으로 처리하며, 다른 상징 회귀 알고리즘이 실패하는 부분에서도 능력을 발휘했습니다. 모델 발견 작업에서도 두 개의 시스템의 governing equations를 성공적으로 복원함으로써 NOMTO의 응용 가능성을 입증하였습니다.



### Artificial Liver Classifier: A New Alternative to Conventional Machine Learning Models (https://arxiv.org/abs/2501.08074)
Comments:
          21 pages

- **What's New**: 이 논문은 인체의 해독 기능에서 영감을 받아 개발된 새로운 감독 학습 분류기인 Artificial Liver Classifier (ALC)를 소개합니다. ALC는 단순성, 속도, 하이퍼파라미터가 필요 없으며, 과적합(overfitting) 문제를 줄이고 여러 클래스 분류 문제를 해결하는 데 효과적입니다. 또한, ALC의 매개변수를 최적화하기 위해 개선된 FOX 최적화 알고리즘(IFOX)을 사용합니다.

- **Technical Details**: ALC는 간의 해독 기능을 모방하여 설계되었으며, 데이터의 반복, 상호작용, 적응과 같은 특성을 효과적으로 포함할 수 있도록 단순화된 수학적 모델로 변환하는 것을 목표로 합니다. 연구는 기존 ML 알고리즘의 간극을 해소하기 위해 수학적 설계의 단순성과 강력한 성능을 결합하려고 합니다. ALC의 성능은 Iris, Breast Cancer Wisconsin, Wine, Voice Gender, MNIST와 같은 5개의 벤치마크 ML 데이터셋에서 평가되었습니다.

- **Performance Highlights**: ALC는 Iris 데이터셋에서 100%의 정확도를 달성했으며, 로지스틱 회귀나 다층 퍼셉트론, 서포트 벡터 머신보다 우수한 성능을 보였습니다. Breast Cancer 데이터셋에서는 99.12%의 정확도를 기록하며 XGBoost와 로지스틱 회귀를 초과하였습니다. ALC는 모든 데이터셋에서 전통적인 분류기들보다 낮은 과적합 격차와 손실을 지속적으로 보여 주었습니다.



### A Roadmap to Guide the Integration of LLMs in Hierarchical Planning (https://arxiv.org/abs/2501.08068)
Comments:
          5 pages, 0 figures, to be published in the AAAI Workshop on Planning in the Era of LLMs ( this https URL )

- **What's New**: 본 연구는 Large Language Models (LLMs)를 Hierarchical Planning (HP) 분야에 통합하기 위한 로드맵을 제안합니다. 특히, LLM을 HP 주기 내에서 어떻게 활용할 수 있는지에 대한 방법론을 탐구합니다. 또한, 향후 LLM 기반 HP 접근법의 성능을 평가하기 위한 표준화된 데이터셋과 벤치마크를 제공하고, 기존 HP 플래너와 LLM 플래너의 초기 결과를 제시합니다.

- **Technical Details**: HP는 Automated Planning (AP)의 하위 분야로, 계획 성능을 높이기 위해 계층적 지식을 활용하는 방법들을 포함합니다. 본 연구에서 제안하는 분류법은 LLM이 HP의 문제 정의, 계획 elaboration 및 후처리 단계에서 어떻게 적용될 수 있는지를 두 가지 차원(Planning Process Role 및 LLM Improvement Strategy)으로 구분합니다. 각각의 LLM 통합 방식은 AP의 최신 연구 기반을 두고 있으며, 다양한 방법론을 탐색할 수 있는 출발점을 제공합니다.

- **Performance Highlights**: 초기 결과에 따르면, 기본 LLM 플래너는 3%의 정확도로 계획을 생성했으며, 아무 것도 올바른 계층적 구조화가 없었습니다. 이 결과는 향후 연구에서 발전의 기준으로 삼을 수 있는 유용한 베이스라인이 됩니다. 본 연구는 HP와 LLM의 통합 연구를 위한 기준을 마련하고, 이 분야의 발전을 위한 귀중한 도구로 작용하기를 기대합니다.



### Self-Attentive Spatio-Temporal Calibration for Precise Intermediate Layer Matching in ANN-to-SNN Distillation (https://arxiv.org/abs/2501.08049)
- **What's New**: 이번 논문은 Spiking Neural Networks (SNNs)의 성능을 향상시키기 위해 새로운 방법인 Self-Attentive Spatio-Temporal Calibration (SASTC)를 제안합니다. SASTC는 ANN과 SNN 간의 의미적으로 일치하는 레이어 쌍을 자율적으로 식별하여 시공간적 차이를 극복하도록 설계되었습니다. 이 방법은 ANN에서 SNN으로 지식을 효과적으로 전이할 수 있도록 하며, SNN의 기존 성능 한계를 뛰어넘는 결과를 보여줍니다.

- **Technical Details**: SASTC는 자가 주의(self-attention) 메커니즘을 사용하여 ANN과 SNN 간의 레이어 패턴을 시공간적으로 정렬합니다. 연구에서는 각각 b 크기의 미니 배치 데이터에 대해 SNN과 ANN의 출력을 구분하기 위해 각 레이어의 출력을 수학적으로 정의합니다. 이를 통해 맞춤형 지식 증류(knowledge distillation)가 가능하여 SNN의 성능을 상당히 개선할 수 있습니다.

- **Performance Highlights**: SASTC는 CIFAR-10, CIFAR-100 및 ImageNet과 같은 여러 데이터셋에서 기존 방법보다 우수한 성능을 보여주었습니다. 특히 CIFAR-10에서는 95.12%, CIFAR-100에서는 79.40%의 정확도를 달성하였으며, DVS-Gesture와 DVS-CIFAR10와 같은 신경 형태 데이터셋에서도 각각 97.92%와 83.60%의 성능을 기록했습니다. 또한 이번 연구는 SNN이 CIFAR-10과 CIFAR-100에서 ANN을 초월한 최초의 사례로, SNN의 잠재적인 응용 가능성을 제시합니다.



### Cooperative Patrol Routing: Optimizing Urban Crime Surveillance through Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.08020)
- **What's New**: 이번 논문은 스마트 시티의 경찰 순찰 경로를 최적화하기 위해 다중 에이전트 강화 학습(이하 MARL) 모델을 제안합니다. 이 모델은 분산된 부분 관찰 마르코프 결정 과정(Decentralized Partially Observable Markov Decision Process)을 기반으로 하여 도시 환경에서의 예측 불가능한 순찰 경로를 계획할 수 있습니다. 본 연구는 마르코의 마르코프 결정 과정을 활용하여, 범죄율이 높은 지역을 효과적으로 커버할 수 있도록 설계되었습니다.

- **Technical Details**: 우리의 MARL 모델은 서로 다른 지역을 경계로 하는 정부의 자원 할당 전략을 고려하여 경찰 순찰 경로를 최적화하기 위해 개발되었습니다. 특히, 경로 설계는 특정 노드를 미리 정하지 않고 이루어지며, 주어진 시간 내에 최대화를 목표로 합니다. 이 모델은 에이전트들이 서로의 위치를 알고 있는 조건 하에서 작동하며, 자원의 동시 배치를 고려하여 경로 설계의 복잡성을 줄이었습니다.

- **Performance Highlights**: 실험 결과는 작성된 모델이 위험 범죄가 빈번한 지역의 90% 이상을 커버하고, 20%의 가장 범죄율이 높은 노드에서는 65%의 커버리지 달성을 보여주었습니다. 또한, 새로운 커버리지 지표를 도입하여 경로 성능을 평가하였으며, 다양한 시나리오에서 모델의 유용성을 설계하여 범죄 모니터링을 위한 최적의 순찰 경로를 도출했습니다.



### GDiffRetro: Retrosynthesis Prediction with Dual Graph Enhanced Molecular Representation and Diffusion Generation (https://arxiv.org/abs/2501.08001)
- **What's New**: 본 논문에서는 GDiffRetro라는 새로운 접근 방식을 제안하여 레트로합성 예측의 기존 한계를 극복하고자 합니다. 기존 연구들은 대부분 분자 그래프에서의 'face' 정보를 잘 포착하지 못했으며, 2D 공간에서 연속 생성 방식으로 반응 물질을 생성하는 한계를 가지고 있습니다. GDiffRetro는 이러한 문제를 해결하기 위해 원래 그래프와 이중 그래프를 통합하여 분자 구조를 표현하고, 조건부 확산 모델을 3D에서 사용합니다.

- **Technical Details**: GDiffRetro는 반응 중심 식별에서 이중 그래프를 도입하여 분자의 feature characterization을 향상시킵니다. 이때 각 노드는 원래 그래프에서의 face를 나타내며, 모델이 분자 그래프 내의 face에 집중할 수 있게 돕습니다. 또한 GDiffRetro는 반응 중심 식별 단계에서 얻은 합성체(synthon)를 바탕으로 3D 확산 과정을 통해 최종 반응 물질을 생성합니다.

- **Performance Highlights**: 실험 결과, GDiffRetro는 다양한 평가 지표에서 최신의 반 템플릿 모델들보다 뛰어난 성능을 보였습니다. 이는 분자 구조의 3D 특성을 잘 보존하며, 물질 생성을 위한 더 합리적인 분포를 생성할 수 있게 하여 반응 물질의 다양성을 증가시킵니다. GDiffRetro는 향후 레트로합성 예측의 새로운 기준이 될 가능성이 큽니다.



### LLM-Ehnanced Holonic Architecture for Ad-Hoc Scalable SoS (https://arxiv.org/abs/2501.07992)
- **What's New**: 이번 논문에서는 현대 시스템의 지속 가능한 발전을 위한 새로운 홀로닉 아키텍처를 제안합니다. 이 아키텍처는 다층 구조로 구성되어 있으며, 추론, 통신, 그리고 기능 레이어를 포함합니다. 또한, 스마트 제조의 원리를 기반으로 하여 감독자, 계획자, 작업자, 자원 홀론과 같은 전문화된 홀론을 도입하여 시스템의 적응성과 재구성 가능성을 높입니다.

- **Technical Details**: 제안된 제도는 각 홀론이 LLMs(대형 언어 모델)를 활용하여 의사 결정을 지원하는 추론 레이어를 포함하고 있습니다. 통신 레이어는 로봇 운영 체제(ROS)를 기반으로 하여 구성 시스템의 기능을 호출하는 능력을 제공합니다. 이 아키텍처는 교통 관리의 복잡한 문제를 해결하기 위해 도시 환경에서의 3D 이동성 케이스 스터디를 통해 그 유용성을 입증합니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 교통 수단을 모델링하여 인간 운영자와 시스템 간의 자연어 상호작용을 지원하고 있습니다. 이러한 접근 방식은 현대 시스템의 주요 과제를 효과적으로 해결하고, 시스템의 스케일러빌리티를 향상시키며, 인간-시스템 상호작용을 증가시키는 것을 목표로 합니다. 이에 따라, futuro implementation 방법과 평가 절차도 제시되어 향후 실증적 검증의 기반을 마련합니다.



### Comprehensive Metapath-based Heterogeneous Graph Transformer for Gene-Disease Association Prediction (https://arxiv.org/abs/2501.07970)
Comments:
          6 pages

- **What's New**: 이번 연구에서는 유전자-질병 연관성(GDA)을 예측하기 위한 새로운 모델인 COmprehensive MEtapath-based heterogeneous graph Transformer(COMET)를 제안합니다. COMET은 다양한 데이터 세트를 통합하여 포괄적인 이종 네트워크를 구성하며, BioGPT를 사용하여 노드 특징을 초기화하여 예측 정확성을 향상시킵니다. 본 연구는 메타패스를 체계적으로 정의하고, 트랜스포머를 이용한 메타패스 인스턴스 집계를 통해 전역 컨텍스트를 포착합니다. 또한, COMET은 주의 메커니즘을 사용하여 다양한 메타패스의 잠재 벡터를 융합하여 유전자-질병 연관성 예측의 정확도를 높입니다.

- **Technical Details**: COMET 모델은 이종 그래프와 메타패스를 활용하여 노드 표현을 학습하고, 최종적으로 GDA 예측을 수행합니다. 이 모델은 먼저 다양한 출처의 데이터를 사용하여 질병과 유전자의 이종 네트워크를 구축하고, BioGPT를 통해 노드 벡터를 초기화합니다. 이어서, 7개의 메타패스를 정의하고 트랜스포머 프레임워크를 통해 메타패스 인스턴스를 집계하여 장거리 의존성을 모델링합니다. 내부 및 외부 메타패스 집계 과정에서 주의 메커니즘을 적용하여 여러 메타패스에서 얻은 잠재 벡터를 융합하여 최종적인 노드 임베딩을 생성합니다.

- **Performance Highlights**: COMET은 기존의 최신 방법들에 비해 뛰어난 강인성을 입증하였습니다. 실험 결과, COMET은 다양한 GDA 예측 방법들과 비교하여 우수성을 보였으며, 특히 병리학 데이터 및 표현형 정보를 효과적으로 통합하는 데 강점을 지니고 있습니다. 또한, 면밀한 분석과 시각화 결과를 통해 COMET의 효과성을 검증하고 있으며, 이는 인체 건강 연구를 발전시키는데 중요한 통찰을 제공합니다.



### Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning (https://arxiv.org/abs/2501.07959)
- **What's New**: 이 논문에서는 Self-Instruct Few-Shot Jailbreaking (Self-Instruct-FSJ)이라는 새로운 프레임워크를 제안하여 기존의 Few-Shot Jailbreaking (FSJ) 기술의 한계를 극복합니다. 이 방법은 패턴 학습과 행동 학습으로 FSJ 공격을 분해하여 LLM의 취약점을 보다 일반적이고 효율적으로 활용할 수 있도록 돕습니다. 특히, 기존의 특수 토큰 주입 기법의 메커니즘을 깊이 분석하고, 데모 레벨의 탐욕적 탐색 (greedy search)을 도입하여 시연의 효과를 극대화합니다.

- **Technical Details**: Self-Instruct-FSJ는 악의적인 행동을 유도하기 위해 특정 인스트럭션 (instruction)과 대응하는 응답 (response) 접두사를 설계합니다. 이 과정은 모델 고유의 채팅 템플릿 토큰과 목표 응답 접두사로 구성된 공존 패턴을 확장하여 이루어집니다. 또한, 악의적인 데모를 직접 샘플링하여 채택함으로써, 자가 교육 (self-instruct) 방식의 행동 학습을 강화합니다.

- **Performance Highlights**: 제안된 방법은 일반적인 오픈 소스 LLM에서 약 90%의 공격 성공률 (Attack Success Rate, ASR)을 기록하며, 8회의 간결한 데모 속에서 효과를 검증합니다. 이 방법은 Jailbreaking 방어 메커니즘에 대한 저항력 또한 높아, Perplexity 필터와 SmoothLLM 등의 방어 기술에 대해 좋은 성능을 보여줍니다.



### Advice for Diabetes Self-Management by ChatGPT Models: Challenges and Recommendations (https://arxiv.org/abs/2501.07931)
- **What's New**: 이 논문은 ChatGPT 3.5와 4가 당뇨병 환자에게 제공하는 조언의 정확성과 개인화 정도를 평가합니다. 연구 결과, 두 모델 모두 특정 지침이 필요한 복잡한 질문에 대해 위험한 수준의 부정확한 조언을 제공할 수 있음을 발견했습니다. 이는 실제 의료 환경에서 인간 감독 없이 AI의 사용이 제한적임을 시사하며, 데이터 해석의 향상이 시급하다는 점을 강조합니다.

- **Technical Details**: 연구는 ChatGPT 모델이 당뇨병 관련 질문에 대한 조언을 얼마나 잘 제공하는지를 평가하며, 모델의 의학적 지식과 상황에 맞는 개인화된 조언 능력을 분석합니다. 최신 모델들은 이전 모델들과 비교했을 때 약간의 개선이 있었지만, 여전히 다양한 한계가 존재합니다. 특히, 불완전한 프롬프트를 기반으로 한 잘못된 판단이 위험한 결과를 초래할 수 있음을 경고합니다.

- **Performance Highlights**: ChatGPT의 성능은 당뇨병 관리에서 유망하지만, 모델이 제공하는 조언의 많은 부분에서 일반화된 답변을 제공하며 개별 환자의 요구를 반영하지 못하는 경향이 있습니다. 특히, 한 모델의 경우 서로 다른 인슐린 유형을 정리하는 데 필요한 조언을 제대로 제공하지 못할 때도 있습니다. 이러한 성과는 AI가 임상 설정에서 환자 요구를 충족하기 위해 최적화되어야 함을 강조합니다.



### An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures (https://arxiv.org/abs/2501.07930)
- **What's New**: 이번 연구에서는 AOC(Adaptive Orthogonal Convolution)라는 새로운 방법론을 소개합니다. 이 방법은 기존의 orthogonal convolution을 손쉽게 확장할 수 있도록 설계되었으며, 이러한 개선은 대규모 응용 프로그램에서의 사용을 가능하게 합니다. Computation overhead(계산 오버헤드)와 현대 기능 지원의 제한을 극복한 점이 두드러집니다.

- **Technical Details**: AOC는 orthogonal convolution의 안정적인 표현을 가능하게 하여 다양한 머신 러닝 문제에 활용될 수 있습니다. 이 방법은 strides, dilations, group convolutions, transposed convolutions와 같은 최신 기능들에 대한 지원을 강화합니다. 이를 통해 모델 아키텍처의 구성에 있어 실용성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, AOC를 이용한 모델들은 점차적으로 더 효율적인 특성을 보여주었습니다. 전통적인 orthogonal convolution과 비교할 때, AOC는 더 표현력이 풍부하면서도 대규모로 확장 가능하다는 사실을 입증했습니다. 이 논문에 따라 제공되는 오픈 소스 라이브러리 또한 발전을 촉진하는 데 기여할 것입니다.



### Exploring Aviation Incident Narratives Using Topic Modeling and Clustering Techniques (https://arxiv.org/abs/2501.07924)
- **What's New**: 이 연구는 비행 안전(aviation safety)의 중요성을 강조하며, 여러 자연어 처리(Natural Language Processing, NLP) 기법을 통해 사고 분석을 심화하는 데 초점을 맞추고 있습니다. 특히 NTSB(National Transportation Safety Board) 데이터셋을 활용하여, 사건과 관련된 잠재적인 주제를 식별하고 시맨틱 관계(semantic relationships)를 탐색합니다.

- **Technical Details**: 연구에서는 LDA(Latent Dirichlet Allocation), NMF(Non-Negative Matrix Factorization), LSA(Latent Semantic Analysis), pLSA(Probabilistic Latent Semantic Analysis), K-means clustering 등의 고급 NLP 기법을 적용하여 사고를 군집화하고 공통 특성을 분석합니다. 이 과정에서 LDA의 일관성(coherence) 값이 0.597로 가장 뛰어난 성과를 보였고, pLSA, LSA, NMF는 각각 0.583, 0.542, 0.437의 결과를 보였습니다.

- **Performance Highlights**: K-means 클러스터링은 사고 내러티브(narratives)에서의 공통점과 독특한 통찰을 드러내며, 반복되는 테마에 대한 이해를 제공합니다. 이 연구는 사고 내러티브 안에 숨겨진 패턴(patters)과 주제 구조를 발견하고, 다양한 주제 모델링 기법의 비교 분석을 통해 비행 안전 개선에 기여할 수 있는 기초를 마련합니다.



### Large Language Model Interface for Home Energy Management Systems (https://arxiv.org/abs/2501.07919)
Comments:
          13 pages conference paper

- **What's New**: 이번 연구에서는 사용자와 상호작용하여 전력 소모를 최적화하는 HEMS(가정 에너지 관리 시스템)용 인터페이스를 제안합니다. 특히, 기술적 배경이 부족한 사용자도 쉽게 사용할 수 있도록 Large Language Models (LLMs)을 활용하여 사용자 맞춤형 파라미터를 효과적으로 입력할 수 있는 시스템을 개발했습니다. 이를 통해 HEMS 사용의 복잡성을 줄이고, 사용자 편의성을 증대시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안하는 시스템은 HEMS, LLM 에이전트 및 사용자의 세 가지 주요 엔티티로 구성됩니다. LLM은 사용자가 제공하는 비정형 정보를 분석하고, 적절히 포맷된 입력값으로 변환하여 HEMS에 전송하는 역할을 합니다. 이 과정에서 ReAct(Reason and Act) 방법론과 few-shot prompting 기법을 활용하여 LLM의 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 LLM 기반 HEMS 인터페이스는 평균 88%의 파라미터 검색 정확도를 기록하여 기존 모델보다 우수한 성능을 보였습니다. 특히, 다양한 전문성 수준을 가진 사용자를 시뮬레이션하여 HEMS의 실용성을 평가함으로써, 이러한 직관적이고 유연한 인터페이스가 HEMS의 널리 보급될 수 있음을 시사합니다.



### Governing AI Agents (https://arxiv.org/abs/2501.07913)
- **What's New**: AI 분야는 요청에 따라 합성 콘텐츠를 생성하는 시스템에서 복잡한 작업을 계획하고 실행할 수 있는 자율형 에이전트로의 근본적인 전환을 겪고 있습니다. 이 변화는 AI 도구를 개발해온 기업들이 인터넷을 독립적으로 탐색하고 다양한 온라인 작업을 수행할 수 있는 AI 에이전트를 구축하고 있다는 것을 의미합니다. 이 새로운 기술이 제시하는 기회는 막대하지만, 그에 따른 위험도 존재합니다.

- **Technical Details**: 이 논문은 AI 에이전트와 관련된 문제들을 다루기 위해 주로 principal-agent 문제와 agency 관계라는 두 가지 강력한 분석 프레임워크를 사용합니다. 첫째로, agency 법과 이론을 통해 정보 비대칭, 재량권, 충성도 등 AI 에이전트가 직면하는 문제들을 식별하고 설명합니다. 둘째로, incentive 설계, 모니터링, 집행과 같은 전통적인 agency 문제 해결 방법의 한계를 보여주며, AI 에이전트가 불투명한 결정을 내리고 전례 없는 속도와 규모로 운영되기 때문에 이러한 방법들이 효과적이지 않을 수 있음을 강조합니다.

- **Performance Highlights**: 셋째로, 이 논문은 AI 에이전트를 설계하고 규제하는 과정을 탐구하며, 포용성(inclusivity), 가시성(visibility), 책임(liability)의 거버넌스 원칙을 지원하기 위한 새로운 기술적 및 법적 인프라가 필요하다고 주장합니다. 이러한 제안은 AI 기술의 활발한 발전을 반영하여 보다 안전하고 효과적인 AI 시스템을 만들어가는 방향으로 이어질 것입니다.



### Deep Learning and Natural Language Processing in the Field of Construction (https://arxiv.org/abs/2501.07911)
- **What's New**: 본 논문은 건설 분야에서 하이퍼님(hypernym) 관계를 추출하는 전체 프로세스를 제안합니다. 이 과정은 용어 추출(terminology extraction)과 이들 용어에서 하이퍼님을 탐지(detection of hypernyms)하는 두 가지 주요 단계로 구성됩니다. 또한, 연구진은 건설 분야의 기술 사양서에서 용어를 추출하기 위한 코퍼스 분석 방법을 설명합니다.

- **Technical Details**: 연구진은 통계(statistics)와 단어 n-그램(word n-grams) 분석을 활용하여 분야의 용어를 추출하며, 언어 패턴 및 인터넷 쿼리를 통해 최종 용어의 품질을 개선하기 위한 가지치기(pruning) 단계를 수행합니다. 이후, 다양한 단어 임베딩(word embedding) 모델과 조합을 기반으로 한 머신러닝(machine learning) 접근 방식을 통해 추출된 용어에서 하이퍼님을 탐지합니다. 6명의 전문가들이 수행한 수동 평가(manual evaluation)를 통해 추출된 용어가 평가됩니다.

- **Performance Highlights**: 하이퍼님 식별 방법은 다양한 데이터셋을 통해 평가되며, 전체 접근법은 관련성 및 유망한 결과를 제공합니다. 이 연구는 건설 분야의 용어 관계를 이해하는 데 기여할 수 있으며, 이러한 방식이 다른 분야에도 적용될 수 있는 가능성을 제시합니다.



### Logarithmic Memory Networks (LMNs): Efficient Long-Range Sequence Modeling for Resource-Constrained Environments (https://arxiv.org/abs/2501.07905)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문은 Logarithmic Memory Networks (LMNs)라는 새로운 아키텍처를 소개합니다. LMNs는 계층적 로그트리 구조를 활용하여 과거 정보를 효율적으로 저장 및 검색합니다. 이 모델은 메모리 사용량을 줄이고 O(n²)에서 O(log(n))으로 주의(attention) 메커니즘의 계산 복잡성을 낮춥니다. 이와 함께, 메모리 관리 시스템 역할을 하는 두 가지 실행 모드를 갖추고 있습니다.

- **Technical Details**: LMNs는 단일 벡터 주의 메커니즘을 사용하여 저장된 정보를 동적으로 액세스합니다. 추가적으로, LMNs는 명시적 위치 인코딩이 필요 없도록 위치 정보를 암묵적으로 인코딩합니다. 모델이 스토리지에서 정보를 가져오는 방식은 계산 효율성과 모델링 능력의 균형을 이뤄 자원이 제한된 환경에서도 실용적인 솔루션을 제공합니다.

- **Performance Highlights**: 이 연구는 자원이 제한된 환경에서 긴 시퀀스를 처리하는 데 있어 LMNs가 실질적인 개선을 제공한다고 강조합니다. 이중 모드 접근 방식은 LMNs가 훈련 과정에서 더 빠른 처리를 가능하게 하며, 추론 과정에서는 메모리 사용량을 대폭 줄입니다. LMNs는 특히 모바일 및 엣지 디바이스 컨텍스트에서 효율성과 확장성을 높일 수 있는 강력하고 확장 가능한 접근법을 제공합니다.



### Anytime Cooperative Implicit Hitting Set Solving (https://arxiv.org/abs/2501.07896)
- **What's New**: 이번 논문에서는 Implicit Hitting Set (HS) 접근법의 새로운 조합을 소개합니다. 기존 HS 접근법은 주로 하한 (lower bounds)을 개선하는 데 초점을 맞췄으나, 이번 연구에서는 상한 (upper bounds) 개선을 위한 용도로도 이 방법이 어떻게 적응될 수 있는지를 보여줍니다. 또한, 여러 스레드(threads)를 활용한 멀티스레드 아키텍처에서 두 가지 HS 접근 방법이 통합되어 어떻게 상호작용할 수 있는지도 논의합니다.

- **Technical Details**: 연구에서는 HS-lb와 HS-ub 각각의 방식이 서로의 코어(core)를 활용하여 시너지를 생성하는 구조를 설명합니다. 이 결과로 만들어진 알고리즘 (HS-lub)은 혼자서 사용할 때보다 두 접근법을 조합했을 때 더 높은 성능을 보여줍니다. HS-lub는 실행 도중에 최적성 간격(optimality gap)을 줄이는 효과적인 anytime behavior를 가지며, 이는 문제 해결에 있어 매우 유리합니다.

- **Performance Highlights**: Weighted CSP 프레임워크에서 세 가지 다른 벤치마크에 대해 테스트한 결과, 논문의 간단한 구현이 기존의 복잡한 병렬 하이브리드 최선 탐색 구현체인 Toulbar2보다도 우수한 성능을 보이는 경우가 많았습니다. 이는 HS-lub가 더욱 발전된 기술에 대항하여 경쟁력 있는 성능을 발휘할 수 있음을 보여줍니다.



### A Driver Advisory System Based on Large Language Model for High-speed Train (https://arxiv.org/abs/2501.07837)
Comments:
          18 pages, 7 figures, presented at 104th TRB Annual Meeting

- **What's New**: 중국 고속철도의 급속한 발전과 함께, 운전자는 적절한 기술적 지원 없이 기술적 문제를 처리하는 데 어려움을 겪고 있습니다. 이 논문에서는 고속철도 운전 중 발생할 수 있는 결함 처리의 정확성과 설명 가능성을 높이기 위해 Intelligent Driver Advisory System (IDAS) 프레임워크와 대형 언어 모델 (LLM) 기반의 IDAS-LLM을 소개합니다. 이를 통해 운전자가 기술적 문제를 보다 효과적으로 대응할 수 있는 가능성을 제시합니다.

- **Technical Details**: IDAS-LLM은 특정 철도 관련 질문과 답변 데이터셋을 사용하여 LLM의 도메인 파인튜닝(domain-fine-tuning)을 수행하여 질문에 대한 정확한 답변을 제공합니다. 이어서, RAG (Retrieval-augmented Generation) 아키텍처를 통합하여 생성되는 응답의 설명 가능성을 증대시킵니다. 이러한 방법론의 결합은 시스템의 효율성과 사용성을 강화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 파인튜닝된 LLM은 평균 10%의 답변 정확도 향상을 보여주었으며, 일부 현재의 주류 LLM들을 능가하는 성과를 보였습니다. 또한, RAG 프레임워크의 도입으로 질문과 답변 세션의 평균 리콜률(recall rate)이 약 4% 증가하였습니다. 이러한 성능을 바탕으로 IDAS-LLM은 실제 운영 시나리오를 통한 결함 처리 능력이 입증되어 실용성이 강조됩니다.



### Flow: A Modular Approach to Automated Agentic Workflow Generation (https://arxiv.org/abs/2501.07834)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 활용한 다중 에이전트 시스템의 워크플로우를 동적으로 업데이트하고 모듈화를 촉진하는 방법을 제안합니다. 이러한 접근법은 실시간으로 서브태스크의 할당 및 에이전트 역할을 조정할 수 있어 예상치 못한 장애물에 대처하는 데 필수적입니다. 우리는 워크플로우를 Activity-on-Vertex (AOV) 그래프로 정의하며, 이를 통해 서로 의존하는 서브태스크 간의 관계를 시각화하고 최적화했습니다.

- **Technical Details**: 우리의 시스템은 AOV 그래프를 구성하여 서브태스크의 상태와 로그를 추적합니다. 그래프의 각 노드는 각각의 서브태스크를 나타내고, 방향성 엣지는 서브태스크 간의 의존성을 나타냅니다. 또한, 고도의 모듈화를 강조하여 복잡한 작업을 더 작은 서브태스크 모듈로 나눠 독립적으로 실행함으로써 운영 효율성을 높이고, 동적 업데이트의 용이성을 극대화합니다.

- **Performance Highlights**: 실험을 통해 우리의 다중 에이전트 시스템이 기존 접근 방식에 비해 적응성과 효율성에서 현저한 개선을 보여주었음을 확인했습니다. 특히, 동적 워크플로우 업데이트와 모듈화 덕분에 에이전트 간의 병렬 실행이 가능해져, 전체 성능을 저해하지 않고도 복잡한 작업을 원활하게 수행할 수 있습니다.



### Agent-Centric Projection of Prompting Techniques and Implications for Synthetic Training Data for Large Language Models (https://arxiv.org/abs/2501.07815)
Comments:
          8 pages, 5 figures. Accepted at ICAART 2025. Derived from an early draft at 2312.17601. arXiv admin note: substantial text overlap with arXiv:2312.17601

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 prompting 기법과 다중 에이전트 시스템에 대한 새로운 관점을 제시합니다. 기존에는 이러한 기법을 비교하고 특성을 파악할 수 있는 체계가 부족했으나, 이를 통해 선형(context)과 비선형(context) 개념을 도입하였습니다. 이 접근법은 단일 LLM prompting과 다중 에이전트 방법 간의 구조적 동등성을 발견하는 데 도움을 줄 것으로 기대됩니다.

- **Technical Details**: 이 논문에서는 대형 언어 모델의 prompting 기법을 agent-centric 관점에서 분석합니다. 선형 맥락은 단일 연속적인 상호작용을 의미하고, 비선형 맥락은 가지치기 또는 다경로의 상호작용을 의미합니다. 이러한 개념은 prompting 전략과 다중 에이전트 시스템 간의 깊은 연결을 탐구할 수 있는 프레임워크를 제공합니다.

- **Performance Highlights**: 저자들은 비선형 prompting 기법의 결과가 동등한 다중 에이전트 시스템에서의 결과를 예측할 수 있다고 주장합니다. 또한, 단일 LLM prompting을 통해 다중 에이전트 시스템 아키텍처를 재현할 수 있으며, 이러한 동등성은 합성 훈련 데이터 생성의 새로운 접근 방식을 제안합니다. 이로 인해 prompting 기술과 다중 에이전트 도메인 간의 연구 결과의 교차 수확이 가능해질 것입니다.



### A Low-cost and Ultra-lightweight Binary Neural Network for Traffic Signal Recognition (https://arxiv.org/abs/2501.07808)
- **What's New**: 이번 연구에서는 하드웨어 배포를 위한 초경량 바이너리 신경망(ULTRA-LIGHTWEIGHT BINARY NEURAL NETWORK, BNN) 모델을 제안합니다. 이 모델은 독일 교통 표지 인식 벤치마크(German Traffic Sign Recognition Benchmark, GTSRB) 데이터세트를 기반으로 이미지 분류 연구를 수행하였으며, 중국 교통 표지(Chinese Traffic Sign, CTS) 및 벨기에 교통 표지(Belgian Traffic Sign, BTS) 데이터세트에서도 검증됩니다. BNN 모델은 97.64%의 높은 인식 성능을 보여 주며, GTSRB 데이터세트에서 가장 우수한 성능을 가진 BNN 모델 중 하나로 부각됩니다.

- **Technical Details**: 이 연구의 모델은 전체 정밀도(full-precision) 모델과 비교했을 때 정확도 손실이 1% 이내로 제한되고, 모델의 파라미터 저장 오버헤드는 전체 정밀도 모델의 10%에 불과합니다. 모델은 추론(inference) 단계에서 논리 연산과 저비트 폭 고정 소수점 덧셈 및 뺄셈 연산에만 의존하여 처리 소자(processing element, PE)의 설계 복잡성을 크게 단순화합니다. 이러한 설계는 자원에 제약이 있는 플랫폼에서의 배포를 용이하게 합니다.

- **Performance Highlights**: 제안된 BNN 모델은 GTSRB 데이터세트를 포함한 다양한 데이터세트에서 높은 정확도와 성능을 보입니다. 특히 자율주행 차량 분야의 컴퓨터 비전(computer vision) 과제를 다룰 때 BNN의 가능성이 매우 크다는 것을 보여줍니다. 이러한 초경량 모델은 이미지 분류의 효율성을 극대화하며, 리소스가 제한적인 환경에서도 우수한 인식 능력을 발휘할 수 있음을 입증합니다.



### Visual Language Models as Operator Agents in the Space Domain (https://arxiv.org/abs/2501.07802)
Comments:
          Updated version of the paper presented in 2025 AIAA SciTech. this https URL

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 우주 분야에서의 응용을 탐구하며, 소프트웨어 및 하드웨어 운용 패러다임을 중점적으로 다룬다. 특히 VLMs가 자율적 제어 및 의사결정을 어떻게 향상시킬 수 있는지를 규명하고, 이러한 모델을 Kerbal Space Program Differential Games (KSPDG) 시뮬레이션 환경에서 활용하여 복잡한 궤도 기동을 수행하도록 하는 방식을 제안한다. 이뿐만 아니라, 로봇 시스템과의 통합을 통해 실제 우주 물체를 검사하고 진단하는 하드웨어 중심의 접근법도 소개하고 있다.

- **Technical Details**: VLM은 LLM의 멀티모달 아키텍처에 기반하여 비주얼 리즈닝을 통합한 모델로, 새로운 토큰을 사용하여 이미지 프레임을 처리할 수 있게 해준다. 연구는 소프트웨어 기반 우주선 제어 운영자와 하드웨어 중심의 로봇 제어 시스템을 통해 두 가지 사례를 다룬다. 이러한 모델은 단순히 자연어 처리(NLP) 문제 해결 도구를 넘어 환경 반응적인 행동을 생성하는 데 중요한 역할을 수행한다.

- **Performance Highlights**: 실험 결과에 따르면, VLM들은 전통적인 방법 및 비멀티모달 LLM들과 경쟁할 수 있는 능력을 보이며, 시뮬레이션 작업에서 뛰어난 성과를 발휘하였다. 이러한 성과는 실제 응용에서도 유망한 가능성을 시사하며, VLM의 비주얼과 텍스트 데이터를 효과적으로 처리하여 상황에 맞는 행동을 생성할 수 있는 능력을 입증한다. 전반적으로 이 연구는 VLM이 자율 시스템을 위한 핵심 요소로 자리매김할 수 있음을 강조한다.



### Rethinking AI Cultural Evaluation (https://arxiv.org/abs/2501.07751)
- **What's New**: 이 연구는 AI 시스템의 문화적 가치 정렬(cultural alignment)을 평가하는 기존의 다중 선택 질문(MCQ) 방법의 한계를 드러내고, 개방형 질문을 통한 평가의 필요성을 강조합니다. 현재의 MCQ 접근법은 복잡한 문화적 가치를 포착하지 못하고 있음을 보여줍니다. 따라서 연구진은 보다 현실적인 시나리오에서 AI 모델이 문화적 가치에 어떻게 반응하는지를 반영하는 새로운 평가 방식을 제안하고 있습니다.

- **Technical Details**: 이 연구에서는 LLM의 문화적 정렬을 평가하기 위해 GlobalOpinionQA 데이터셋을 사용했습니다. 연구진은 GPT4o 모델에게 다양한 프롬프팅 기법을 통해 응답을 유도하고, 생성된 응답과 인간의 응답 분포 간의 Wasserstein distance를 측정했습니다. 이 과정에서, 개방형 질문과 체인 오브 시크릿(Chain-of-Thought, CoT) 기법을 활용하여 기존 MCQ 방식의 신뢰성을 비교하고 분석하였습니다.

- **Performance Highlights**: 결과적으로, 연구는 전통적인 MCQ 평가 방식이 LLM의 문화적 정렬을 제대로 포착하지 못한다는 것을 증명했습니다. 특히, 개방형 상황에서 모델이 응답을 하지 않거나 애매한 답변을 줄 빈도가 더 높았음을 보여주었습니다. 이 연구는 문화적 정렬을 평가하기 위한 맞춤형 프레임워크와 모델 행동에 대한 포괄적인 접근이 필요함을 강조하고 있습니다.



### CDS: Data Synthesis Method Guided by Cognitive Diagnosis Theory (https://arxiv.org/abs/2501.07674)
- **What's New**: 이 연구에서는 복잡한 작업을 개별 지식 포인트로 분해하여 LLM의 성능 향상을 위한 새로운 방법인 Cognitive Diagnostic Synthesis (CDS)를 소개합니다. CDS 방법은 Cognitive Diagnosis Theory (CDT)를 활용하여 LLM의 정밀한 평가와 맞춤형 개선을 가능하게 합니다. 본 연구의 접근법은 모델의 강점과 약점을 정확하게 식별하고 통합할 수 있습니다.

- **Technical Details**: CDS 방법은 크게 두 부분으로 나뉘어 있습니다. 매크로 수준에서 문제를 여러 개의 지식 포인트로 세분화하고, 각 포인트에 대한 모델의 숙련도를 통계적으로 평가합니다. 마이크로 수준에서는 고급 LLM이 제공하는 특정 응답 기록을 바탕으로 인지 진단을 수행하여, 약한 지식 포인트를 중심으로 고품질 데이터를 생성하도록 안내합니다.

- **Performance Highlights**: CDS 방법을 활용한 모델의 수학 및 코딩 능력이 약 11.12% 향상되었습니다. 본 연구는 지식 포인트 수준에서의 평가 과정 및 데이터 합성 전략을 통해 모델의 성능 개선을 이끌어내고 있습니다. 두 단계의 데이터 필터링 방식이 도입되어 효율적이고 목표 지향적인 데이터 선택을 가능하게 합니다.



### Large Language Models for Interpretable Mental Health Diagnosis (https://arxiv.org/abs/2501.07653)
Comments:
          Accepted at AAAI 2025 Workshop on Large Language Models and Generative AI for Health (GenAI4Health)

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)과 제약 논리 프로그래밍(CLP)을 결합한 정신 건강 진단을 위한 임상 결정 지원 시스템(CDSS)을 제안합니다. CDSS는 진단 매뉴얼의 복잡성으로 인해 발생하는 오류를 줄이고 전문가 검토를 통한 해석 가능성을 보장합니다. 연구 결과, LLM은 후보 논리 프로그램을 생성하는 데 유용하지만, 전문가의 검토와 수정이 필요하다는 점이 강조됩니다.

- **Technical Details**: CDSS는 특정 진단 규칙을 자동으로 처리하는 소프트웨어 도구로서, LLM을 통해 DSM-5-TR 및 ICD-11 CDDR과 같은 진단 매뉴얼의 설명을 논리 규칙으로 변환하고, CLP 엔진을 통해 이 규칙을 실행하여 진단 결과를 도출합니다. CLP는 정확성과 투명성을 중시하는 프로그래밍 패러다임으로, Datalog 언어를 사용하며, Soufflé라는 최첨단 Datalog 엔진으로 프로그램을 해결합니다.

- **Performance Highlights**: 실험 결과, LLM만 사용하거나 LLM 생성 논리 프로그램을 전문가 검토 없이 사용할 경우보다, 전문가의 검토를 포함한 우리의 CDSS 방법이 더 높은 정확성과 해석 가능성을 제공하는 것으로 나타났습니다. 이에 따라, 우리의 접근 방식은 정신 건강 진단에서 신뢰성과 해석 가능성을 모두 확보할 수 있는 잠재력을 가지고 있습니다.



### SafePowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models (https://arxiv.org/abs/2501.07639)
- **What's New**: 이 논문은 Optimal Power Flow (OPF) 문제를 고려하여 Large Language Models (LLMs)를 사용한 새로운 프레임워크인 SafePowerGraph-LLM을 소개합니다. 이 접근법은 전력망의 복잡한 관계와 제약을 효과적으로 쿼리하기 위해 그래프와 테이블 표현을 결합하여 OPF 문제를 해결합니다. 특히, OPF 문제에 특화된 새로운 in-context learning 및 fine-tuning 프로토콜을 도입하며, 기존 LLM을 사용하여 믿을 수 있는 성능을 확인합니다.

- **Technical Details**: SafePowerGraph-LLM 프레임워크는 사이클 형태로 구성되어 LLM을 쿼리하기 위한 메시지 포맷을 생성합니다. 초기 전력망 특성에서 시작하여 각 구성 요소를 독립적 서브그래프로 생성하고, 각 노드는 버스, 부하, 발전기, 슬랙, 또는 라인 등 다양한 유형으로 분류됩니다. 또한, LoRA (Low-Rank Adaptation) 기술을 활용하여 설정을 효율적으로 조정할 수 있으며, 이는 모델 무게를 유지하면서 특정 작업에 적합하게 조정할 수 있게 해줍니다.

- **Performance Highlights**: 실증적 연구에서는 여러 모델인 OpenAI의 gpt-4o-mini 및 gpt-4o, Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct를 테스트하였습니다. 6500개의 예시 쌍을 생성하고 평가하는 과정을 반복하여 모델의 일반화 가능성을 확인했습니다. 결과적으로, 모델들은 주어진 테스트 환경에서 우수한 성과를 보여주었으며, 이는 SafePowerGraph-LLM의 효과성을 뒷받침합니다.



### PokerBench: Training Large Language Models to become Professional Poker Players (https://arxiv.org/abs/2501.08328)
Comments:
          AAAI 2025

- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)의 포커 게임 능력을 평가하기 위한 벤치마크인 PokerBench를 소개합니다. 기존의 NLP 작업에서 좋은 성과를 내고 있는 LLM이지만, 포커와 같은 복잡한 전략 게임에서의 적용은 새로운 도전 과제입니다. PokerBench는 전문 포커 플레이어와 협력하여 개발한 11,000개의 중요한 시나리오로 구성되어 있으며, LLM의 포커 게임 능력을 평가하는 데 유용합니다.

- **Technical Details**: PokerBench는 포커의 pre-flop 및 post-flop 상황을 포함하여 결정적인 의사결정을 평가하는 데 필요한 11,000개의 스폿을 제공합니다. 이 데이터셋은 게임 이론 최적(poker optimal) 원칙에 기반하여 개발되었으며, GPT-4 및 ChatGPT 3.5 등 다양한 최첨단 모델을 평가하는 데 활용되었습니다. 연구 결과, 모든 모델이 최적의 포커 플레이에서 저조한 성과를 나타냈지만, 파인 튜닝을 통해 성능이 크게 개선되었습니다.

- **Performance Highlights**: PokerBench를 사용하여 다양한 LLM 모델의 포커 게임 능력을 평가한 결과, GPT-4가 53.55%의 정확도로 가장 높은 성능을 보였습니다. 또한, 파인 튜닝된 모델 간의 경쟁을 통해 PokerBench에서 높은 점수를 받은 모델이 더 많은 승수를 기록하는 경향이 있음을 확인했습니다. 이 연구는 LLM의 포커 게임 능력을 빠르고 신뢰성 있게 평가할 수 있는 새로운 벤치마크를 제공하며, LLM의 복잡한 게임 시나리오에서의 발전을 연구하는 데 중요한 기초 자료를 제공합니다.



### Diffusion Adversarial Post-Training for One-Step Video Generation (https://arxiv.org/abs/2501.08316)
- **What's New**: 본 연구에서는 Adversarial Post-Training (APT)이라는 새로운 방법을 통해 고해상도 비디오 생성을 위한 단일 단계 모델을 제안합니다. 기존의 감쇠 학습(distillation) 방법들이 영상 품질을 저하시키는 문제를 겪는 반면, APT는 실 데이터를 대상으로 적대적 학습을 통해 학습의 안정성과 품질을 향상시킵니다. 이를 통해 2초 길이의 1280x720 해상도 비디오를 실시간으로 생성할 수 있습니다.

- **Technical Details**: APT는 사전 훈련된 감쇠 모델을 초기화로 사용하며, 이 모델을 통해 실 데이터에 대해 적대적 학습 목표를 수행합니다. 이러한 과정에서 R1 정규화(regularization) 손실을 추가하여 대규모 학습을 용이하게 하고, 제너레이터와 판별기를 안정화하기 위한 여러 설계를 도입했습니다. 특히, 제너레이터는 결정론적(distillation) 학습을 통해 초기화되며, 판별기는 변환기(transformer) 기반의 구조로 설계되어 있습니다.

- **Performance Highlights**: 우리의 모델 Seaweed-APT는 2초 길이의 고해상도 비디오와 1024px 이미지를 생성하는 데 있어 기존 방법들과 비교할 수 있는 성능을 발휘합니다. 사용자 연구에 기반한 평가에서는 시각적 충실도(visual fidelity), 구조적 무결성(structural integrity), 텍스트 정렬(text alignment) 측면에서도 높이 평가받았습니다. 특히, H100 GPU를 활용하여 단일 단계로 1280x720 해상도의 비디오를 2초 만에 생성할 수 있는 성과를 이루었습니다.



### Polynomial Threshold Functions of Bounded Tree-Width: Some Explainability and Complexity Aspects (https://arxiv.org/abs/2501.08297)
Comments:
          22 pages, 3 figures. To be published in Festschrift in honor of Johann A. Makowsky

- **What's New**: 이 논문은 다변수 다항식의 트리너비(tree-width)를 활용한 새로운 방법론을 제시합니다. 이전 연구에서는 트리너비가 제한된 다변수 다항식을 연구하여 일반적으로 해결하기 어려운 문제의 다항적 해결 가능성을 보여주었습니다. 본 논문에서는 이러한 아이디어를 불리언 함수(Boolean function)에 응용하여, 다항식 임계 표현(polynomial threshold representation)을 다루고 있습니다.

- **Technical Details**: 논문은 다항식의 트리너비와 그 정의에 대한 배경을 설명합니다. 다항식은 G⁢F⁢(2) 또는 실수(real) 상에서 불리언 함수를 정확히 나타낼 수 있으며, 본 연구에서는 실수 상의 다항식에 중점을 두었습니다. 특히, 다항식의 다룰 수 있는 성질을 사용하여 의사 결정 모델과 같은 실질적인 응용 사례를 다룹니다.

- **Performance Highlights**: 본 연구는 이해 가능한 인공지능(Explainable AI, XAI) 분야에 중점을 두며, 베이지안 네트워크 분류기(Bayesian network classifiers)와 같은 확률적 그래픽 모델을 다룹니다. 베이지안 네트워크의 구조가 트리너비에 제한을 두면, 문제의 해결 가능성이 증가하며, 이는 설명 가능성과 신뢰성을 향상시키는 데 기여할 수 있습니다. 이 외에도 양의 및 일반 다항식 임계 함수의 표현력을 서로 분리하는 결과를 제공합니다.



### HALoGEN: Fantastic LLM Hallucinations and Where to Find Them (https://arxiv.org/abs/2501.08292)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 HALoGEN이라는 종합적인 환각(hallucination) 벤치마크를 소개합니다. 이 벤치마크는 프로그래밍, 과학 고증, 요약 등 아홉 가지 도메인에 걸쳐 10,923개의 프롬프트(prompt)와 각 사용 사례에 대한 자동 고정밀 검증기(automatic high-precision verifiers)를 포함하고 있습니다. 이를 통해 대규모 언어 모델을 평가할 수 있는 체계를 마련하였습니다.

- **Technical Details**: HALoGEN은 LLM(generative large language models)이 생성한 결과를 원자 단위(atomic units)로 분해하고, 이를 고품질의 지식 소스에 대조하여 검증합니다. 연구진은 14개의 언어 모델에서 약 150,000개의 생성 결과를 평가하였고, 일부 도메인에서는 생성된 원자 사실.atomic facts의 최대 86%가 환각을 포함하고 있다는 사실을 발견했습니다. 또한, 환각의 유형을 훈련 데이터의 잘못된 기억(Type A errors), 잘못된 지식(Type B errors), 그리고 허구(Type C errors)로 분류하는 새로운 오류 분류법을 정의하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 가장 성능이 뛰어난 모델조차 높은 비율의 환각을 포함하고 있어 신뢰할 수 있는 LLM 개발에 대한 필요성을 강조합니다. 연구진은 HALoGEN을 통해 생성 모델의 환각 발생 원인에 대한 체계적 연구에 기초가 되기를 희망하며, 궁극적으로 신뢰할 수 있는 대규모 언어 모델 개발을 촉진하는 데 기여하고자 합니다.



### Comparative Analysis of Efficient Adapter-Based Fine-Tuning of State-of-the-Art Transformer Models (https://arxiv.org/abs/2501.08271)
- **What's New**: 이번 연구에서는 SuperGLUE 벤치마크의 이항 분류 (binary classification) 작업과 Kaggle의 다중 클래스 뉴스 카테고리 분류 작업에 대한 다양한 어댑터 아키텍처 (adapter architectures)의 효능을 조사했습니다. 연구에서는 DistilBERT, ELECTRA, BART라는 세 가지 트랜스포머 (transformer) 모델의 분류 성능과 시간 복잡도를 비교했습니다. 특별히, 기존의 파인 튜닝 (fine-tuning) 방식과 아홉 가지 최신 어댑터 아키텍처를 통해 성능 차이를 분석하였습니다.

- **Technical Details**: 분석 결과, 어댑터 아키텍처들 간의 성능 차이가 관찰되었으며, 특히 시간 소요를 현저히 줄이면서도 파인 튜닝에 비견하는 또는 그 이상의 성능을 발휘할 수 있음을 보여주었습니다. 새로운 분류 작업에서도 유사한 결과가 나타났으며, 이는 어댑터가 파인 튜닝에 대한 효율적이고 유연한 대안이라는 점을 입증합니다. 다양한 자연어 처리 (NLP) 응용 프로그램에서 어댑터를 선택하고 구현하기 위한 귀중한 통찰력과 지침을 제공합니다.

- **Performance Highlights**: 종합적으로, 어댑터 아키텍처는 여러 기준에서 파인 튜닝보다 우수한 결과를 나타내며, 이는 특히 훈련 시간이 짧다는 점에서 큰 장점입니다. 연구 결과는 다양한 NLP 작업에서 어댑터의 유용성을 보여주며, 시간이 제한된 환경에서도 효과적으로 모델을 학습시킬 수 있는 방법을 제시합니다. 이러한 발견은 향후 연구 및 실제 적용에도 중요한 기초자료가 될 것입니다.



### AI Driven Water Segmentation with deep learning models for Enhanced Flood Monitoring (https://arxiv.org/abs/2501.08266)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 기후 변화로 인해 증가하는 홍수 발생 빈도에 대응하기 위해 UNet, ResNet, DeepLabv3의 세 가지 딥 러닝 모델을 비교하여 픽셀 수준의 수분 분할을 수행합니다. 새로운 데이터 세트를 생성하여 잘 알려진 기준 데이터 세트를 홍수 특정 이미지로 보강함으로써 모델의 견고성을 향상시켰습니다. 이 연구는 드론, 현장 관찰 및 소셜 미디어의 이미지를 활용하여 홍수 감지를 지원하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구에서는 UNet, ResNet, DeepLabv3 아키텍처를 다양한 환경 조건과 지리적 위치에서 테스트하여 각각의 효과성을 평가합니다. 모델은 완전 자동화된 방식으로 이미지를 분석하여 홍수 지역을 식별하며, 전통적인 반자동 방법에 비해 처리 시간을 대폭 단축합니다. 이미지 분할 마스크를 예측함으로써 각 모델의 강점과 한계를 논의하며, 효율적인 홍수 모니터링을 위한 방법론을 제시합니다.

- **Performance Highlights**: 이러한 접근 방식은 생명과 경제적 손실을 줄이기 위한 비상 대응 팀에 중요한 데이터를 제공하는 시기적절하고 지속적인 홍수 모니터링을 가능하게 합니다. 기존 방법에 비해 홍수 지도 생성 시간을 크게 단축시켰으며, 향후 연구를 위한 멀티모달 데이터 통합 및 홍수 감지 전용 견고한 딥 러닝 아키텍처 개발의 가능성을 제시합니다. 전반적으로, 이 연구는 딥 러닝 기술의 혁신적인 사용을 통해 홍수 관리 전략의 발전에 기여하고 있습니다.



### Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models (https://arxiv.org/abs/2501.08248)
- **What's New**: 이 논문에서는 In-Context Retrieval and Reasoning (ICR^2)라는 새로운 벤치마크를 소개합니다. 기존의 LOFT 벤치마크가 LCLMs의 성능을 과대평가할 수 있는 문제를 해결하고자, LCLMs에 대한 보다 실제적인 평가를 제공하기 위해 고안된 것입니다. ICR^2는 강화된 컨파운딩 문서(confounding documents)를 포함하여 LCLMs의 실제적인 상황에서의 성능을 평가합니다.

- **Technical Details**: ICR^2 벤치마크는 Wikipedia에서 수집된 포괄적인 지식 기반을 기반으로 구축되며, 강력한 리트리버를 사용해 문서를 선택합니다. 이를 통해 설명된 세 가지 방법은 (1) retrieve-then-generate fine-tuning, (2) retrieval-attention probing, (3) joint retrieval head training 입니다. 이 방법들은 LCLMs의 in-context retrieval 능력을 증가시키고, 복잡한 multi-stage pipeline의 한계를 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 논문에서 제시된 최적의 접근 방식은 Mistral-7B 모델에 적용되어 LOFT에서 평균 +17과 +15 포인트 향상된 Exact Match를 보여주었으며, ICR^2에서도 +13과 +2 포인트의 개선을 이뤘습니다. 이 방법은 기존의 Vanilla RAG 및 supervised fine-tuning을 초월하여 성능을 발휘하였고, 더 작은 모델임에도 불구하고 대부분의 작업에서 GPT-4-Turbo보다 우수한 성능을 기록했습니다.



### Engineering LLM Powered Multi-agent Framework for Autonomous CloudOps (https://arxiv.org/abs/2501.08243)
Comments:
          The paper has been accepted as full paper to CAIN 2025 (this https URL), co-located with ICSE 2025 (this https URL). The paper was submitted to CAIN for review on 9 November 2024

- **What's New**: MontyCloud Inc.는 CloudOps(클라우드 운영) 분야에서 자율 봇(autonomous bots)을 활용하여 클라우드 준수(compliance), 보안(security), 그리고 지속적인 운영(continuous operations)을 관리하는 주요 기업입니다. 이들은 GenAI(Generative AI)를 사용하여 클라우드 인프라의 자동화된 관리 및 최적화를 진행하며, 고객들에게 더욱 접근 가능하고 효과적인 플랫폼을 제공하고자 합니다. 특히, 기존 MontyCloud 시스템을 위한 GenAI 기반 솔루션 개발 과정에서 다양한 데이터 소스, 여러 프로세스의 오케스트레이션(orchestration), 복잡한 워크플로우를 처리하는 것이 도전 과제가 되었습니다.

- **Technical Details**: 이러한 도전 과제를 해결하기 위해 MOYA라는 다중 에이전트 프레임워크(multi-agent framework)를 개발하였습니다. MOYA는 GenAI를 활용하여 자율성과 인간 조정(human control)을 균형 있게 조화시키며, 다양한 내부 및 외부 시스템을 통합합니다. 또한, 작업 오케스트레이션(task orchestration), 보안, 오류 완화(error mitigation)와 같은 요소에 최적화 되어 있으며, Retrieval Augmented Generation (RAG) 기술을 통해 정확하고 신뢰할 수 있으며 관련성 있는 인사이트를 생성합니다.

- **Performance Highlights**: MOYA의 다중 에이전트 시스템을 실무자와의 협력을 통해 평가한 결과, 비-에이전틱(non-agentic) 접근 방식에 비해 복잡한 워크플로우에서 정확성, 반응성, 효과성이 개선되었습니다. 이를 통해 MontyCloud의 CloudOps 관리가 더욱 원활해졌으며, 고객들에게 더욱 신뢰할 수 있는 솔루션을 제공하게 되었습니다. 이러한 성과는 클라우드 환경이 점점 복잡해지는 현재, 자율적 관리의 필요성이 증가하고 있음을 시사합니다.



### A Feature-Level Ensemble Model for COVID-19 Identification in CXR Images using Choquet Integral and Differential Evolution Optimization (https://arxiv.org/abs/2501.08241)
- **What's New**: COVID-19 팬데믹은 전 세계적으로 수십억 명에게 심각한 영향을 미쳤습니다. 본 논문은 Chest X-ray (CXR) 이미지를 통해 COVID-19 감염자를 정확히 식별하기 위한 새로운 Deep Learning Diagnosis System을 소개합니다. 이 시스템은 사전 학습된 Deep Convolutional Neural Networks (DCNNs)를 통합한 앙상블 학습(framework) 구조를 이용하여, 기존 RT-PCR의 한계를 극복합니다.

- **Technical Details**: 이 연구에서는 Choquet integral을 이용하여 서로 다른 DCNN의 특징 벡터를 결합하여, 선형 접근으로는 잡을 수 없는 상호작용을 포착합니다. Sugeno-$\lambda$ 측정 이론을 활용하여 네트워크의 하위 집합에 대한 퍼지(fuzzy) 측정을 도출하며, Differential Evolution 알고리즘을 통해 퍼지 밀도를 추정합니다. 또한, 복잡한 특성 벡터 집합을 용이하게 집계(strategies)할 수 있도록 TensorFlow 기반의 Choquet 연산 레이어를 개발했습니다.

- **Performance Highlights**: COVIDx 데이터셋에서의 실험 결과, 본 앙상블 모델은 세 개 클래스로 분류 시 98
%의 정확도, 이진 분류 시 99.50	ext{%}의 정확도를 기록했습니다. DenseNet-201(세 개 분류에서 97	ext{，以서술자}%, 이진에서 98.75	ext{%}), Inception-v3(세 개 분류에서 96.25	ext{%}, 이진에서 98.50	ext{%}), Xception(세 개 분류에서 94.50	ext{%}, 이진에서 98	ext{%}) 보다 월등히 뛰어난 성능을 보여주었으며, 이전의 많은 방법들을 초월했습니다.



### Dynamic Pricing in High-Speed Railways Using Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.08234)
Comments:
          37 pages, 5 figures

- **What's New**: 이 논문은 경쟁 및 협력을 갖춘 고속 여객 철도 산업의 동적 가격 전략 설계라는 중요한 과제를 다룹니다. 비제로섬 마르코프 게임(Markov Game)을 기반으로 한 다중 에이전트 강화 학습(MARL) 프레임워크가 제안되며, 이 프레임워크는 승객 의사 결정 과정을 반영하기 위해 랜덤 유틸리티 모델을 통합합니다. 고속 철도 시스템에서 동적 가격 책정에 대한 심층 강화 학습의 적용이 최근까지 제한적이었던 점에 주목하여, 다양한 철도 네트워크 구성과 수요 패턴을 모델링 할 수 있는 파라미터화된 RL 시뮬레이터인 RailPricing-RL을 소개합니다.

- **Technical Details**: RailPricing-RL 시뮬레이터는 협력적-경쟁적 환경을 조성하여 다양한 에이전트가 수요 변화에 따라 실시간으로 티켓 가격을 조정하도록 지원합니다. 이 프레임워크를 통해 에이전트 간 협력과 경쟁 간의 균형을 유지하면서 동적 가격 전략을 탐구할 수 있습니다. 실험은 Multi-Actor Attention Critic (MAAC) 및 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)와 같은 최신 MARL 알고리즘을 사용하여 고속 철도 네트워크의 동적 가격 책정을 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과는 사용자 선호가 MARL 성능에 미치는 영향과 가격 정책이 승객의 선택, 유틸리티 및 시스템 전체의 동적으로 영향을 미친다는 것을 입증합니다. 또한 이 연구는 동적 가격 책정 전략을 발전시키기 위한 기반을 제공하며, 수익성과 시스템 효율성을 일치시키는 방법에 대한 통찰을 제시합니다. 이를 통해 실제 적용을 위한 강건한 MARL 기반 솔루션 설계를 위한 중요한 통찰을 얻을 수 있습니다.



### ASTRID -- An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems (https://arxiv.org/abs/2501.08208)
Comments:
          29 pages

- **What's New**: 이번 연구에서는 Retrieval Augmented Generation(RAG) 기반의 임상 질문 응답(QA) 시스템을 평가하기 위한 새로운 자동화된 메트릭스인 ASTRID를 발표합니다. ASTRID는 Context Relevance (CR), Refusal Accuracy (RA), Conversational Faithfulness (CF)의 세 가지 메트릭으로 구성되어 있으며, 이 메트릭들은 클리닉 QA 시스템의 위험성을 체계적으로 분석하고, 모델의 응답의 신뢰성을 높이기 위해 설계되었습니다.

- **Technical Details**: ASTRID는 200개의 실제 환자 질문 데이터셋을 기반으로 CF 메트릭의 효과성을 검증하였습니다. 이 메트릭은 기존 정의보다 임상 대화에서의 인간 평점을 더 정확히 예측할 수 있으며, 비전문가로부터의 평가에서도 일관성을 보입니다. 또한, CF, RA, CR 메트릭을 함께 사용하면 전문가의 평가와 일치하는 경향이 관찰되었습니다.

- **Performance Highlights**: 본 연구에 따르면 ASTRID의 세 개 메트릭은 총 아홉 개의 서로 다른 LLM 모델에서 인간 평가와 높은 일치를 보였으며, 이는 LLM 기반 자동화 평가 파이프라인에서 이들 메트릭을 활용할 수 있는 가능성을 보여줍니다. ASTRID의 첫 번째 메트릭인 CF는 특히 임상 대화에서 모델의 응답이 지식 기반에 충실한지를 평가하는 데 우수한 성능을 보였습니다.



### Modeling Feature Maps for Quantum Machine Learning (https://arxiv.org/abs/2501.08205)
- **What's New**: 이 연구는 다양한 양자 노이즈 모델(dephasing, amplitude damping 등)이 Quantum Machine Learning (QML) 알고리즘에 미치는 영향을 체계적으로 평가합니다. QSVC 알고리즘이 노이즈에 매우 강한 반면, Peg-QSVC 및 QNN은 더 민감하며 특히 잘못된 정보에 취약하다는 점이 강조됩니다. 또한, 양자 노이즈가 QML의 성능에 미치는 영향과 이를 극복하기 위한 전략이 중요하다는 점을 시사합니다.

- **Technical Details**: 연구에서는 NISQ 기기에서 발생하는 다양한 양자 노이즈를 포함한 시뮬레이션 모델을 사용해 QML 알고리즘의 동작을 분석합니다. QSVC, Peg-QSVC, QNN, VQC와 같은 알고리즘을 평가하며, 기능 매핑(Fature Mapping) 기법에 대한 설정도 다룹니다. 각 양자 노이즈 모델에 따른 성능 차이를 분석하여 QSVC가 가장 저항력이 뛰어난 알고리즘으로 확인되었습니다.

- **Performance Highlights**: QSVC 알고리즘은 다양한 노이즈 환경에서도 높은 정확도를 유지함으로써 유연성과 강인성을 보입니다. 반면, Peg-QSVC와 QNN은 노이즈의 종류에 따라 성능 차이가 두드러지며 이로 인해 정확성과 신뢰성이 떨어질 수 있습니다. 이 연구는 개인 맞춤형 유전학 분석의 미래를 위한 QML의 적용 가능성을 보여줍니다.



### EmoNeXt: an Adapted ConvNeXt for Facial Emotion Recognition (https://arxiv.org/abs/2501.08199)
Comments:
          6 pages, 5 figures and 2 tables. 2023 IEEE 25th International Workshop on Multimedia Signal Processing (MMSP), Poitiers, France

- **What's New**: 이 논문에서는 EmoNeXt라는 새로운 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 조정된 ConvNeXt 아키텍처를 기반으로 하며, Spatial Transformer Network (STN)과 Squeeze-and-Excitation 블록을 통합하여 얼굴의 특징적인 영역에 집중하고 채널 간 의존성을 포착합니다. 또한 self-attention 정규화 항을 소개하여 모델이 간결한 특징 벡터를 생성하도록 합니다. 이를 통해 FER2013 데이터셋에서 기존 모델들보다 높은 감정 분류 정확도를 달성하였습니다.

- **Technical Details**: EmoNeXt는 입력 이미지에 공간 변환을 학습하고 적용하기 위한 STN을 통합합니다. STN은 입력 이미지의 벡터 표현을 생성하고 이를 바탕으로 샘플링 그리드를 형성하는 로컬라이제이션 네트워크를 포함합니다. 그러면서 CNN의 효과를 유지하면서도 Vision Transformers의 장점을 살리기 위해 ConvNeXt 아키텍처에 대한 다양한 개선을 포함하고 있습니다. 특히, 활성화 함수로 GELU를 사용하고, BatchNorm 대신 Layer Normalization을 통해 성능을 향상시킵니다.

- **Performance Highlights**: EmoNeXt는 FER2013 데이터셋을 사용한 실험에서 기존의 최첨단 깊은 학습 모델들과 비교해 우수한 성능을 보여주었습니다. 모델은 더욱 간결한 특징을 생성하도록 유도함으로써 감정 분류에서의 정확도를 극대화합니다. 연구 결과는 EmoNeXt 모델이 자동으로 얼굴 감정을 인식하는 데 있어 효과적인 접근 방식임을 입증하고 있습니다.



### A Critical Synthesis of Uncertainty Quantification and Foundation Models in Monocular Depth Estimation (https://arxiv.org/abs/2501.08188)
- **What's New**: 최근의 기초 모델들이 단안 깊이 추정(monocular depth estimation) 분야에서 중요한 돌파구를 마련했으나, 실제 환경에서 안전하고 신뢰할 수 있는 배포를 위한 명확한 경로는 여전히 부족합니다. 이는 거리 예측을 포함하는 메트릭 깊이 추정(metric depth estimation) 과정에서 특히 두드러지며, 고급 기초 모델조차도 치명적인 오류에 취약해 문제가 되고 있습니다. 본 연구에서는 5가지 불확실성 정량화(un) 방법을 강화된 DepthAnythingV2 기초 모델에 결합하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 기존의 메트릭 깊이 추정 방법론의 한계를 극복하기 위해, 연구팀은 Gaussian Negative Log-Likelihood Loss (GNLL)을 통해 신뢰할 수 있는 불확실성 예측을 제공하는 효율적인 기법을 제안하고 있습니다. 추가로 Monte Carlo Dropout (MCD), Sub-Ensembles (SE), Test-Time Augmentation (TTA) 등 다양한 불확실성 정량화(UQ) 방법을 적용하여 모든 픽셀에 대한 깊이 예측의 불확실성을 계량화하는 데 초점을 맞추고 있습니다. 이를 통해 총 4개의 다양한 데이터셋에서 성능을 평가하여, 실세계 응용에 대한 광범위한 커버리지를 확보하고자 합니다.

- **Performance Highlights**: 연구 결과, GNLL 기반의 미세 조정 방법이 특히 유망한 접근법으로 확인되었습니다. 이 방법은 예측 성능을 유지하면서도 불확실성을 신뢰성 있게 추정하는 데 뛰어난 효율성을 보여 주며, 기존의 기초 모델과 비교하여 훈련 및 추론 시간 면에서도 동등한 성능을 발휘합니다. 또한, 본 연구는 단안 깊이 추정을 넘어 발생할 수 있는 다양한 문제들과의 연결 단서를 제공하며, 데이터 반영에 대한 이해도와 신뢰성을 높이는 데 기여할 것입니다.



### A Multi-Modal AI Copilot for Single-Cell Analysis with Instruction Following (https://arxiv.org/abs/2501.08187)
Comments:
          37 pages; 13 figures; Code: this https URL Models: this https URL, this https URL

- **What's New**: 새로운 기술인 InstructCell은 자연어 처리(NLP)를 기반으로 한 multi-modal AI copilot으로, 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터를 보다 효과적으로 분석할 수 있도록 돕습니다. 이 시스템은 서로 다른 조직과 종에서 수집된 scRNA-seq 프로파일과 텍스트 기반 지침을 결합한 종합적인 데이터셋을 통해 개발되었습니다. 사용자는 InstructCell을 통해 세포 유형 주석, 조건부 pseudo-cell 생성, 약물 감수성 예측과 같은 중요한 작업을 쉽고 직관적인 자연어 명령으로 수행할 수 있습니다.

- **Technical Details**: InstructCell은 Q-Former 모듈, 사전 훈련된 언어 모델(LLM), 그리고 세포 재구성 블록을 포함하는 복합 다중 모달 세포 언어 아키텍처를 통해 텍스트 정보와 단일 세포 유전자 발현 데이터를 동시에 처리할 수 있습니다. 이 시스템은 다중 세포 분석 작업에 필수적인 지침-응답 쌍을 생성하여 다양한 실험적 조건에 적응할 수 있도록 설계되었습니다. 또한, 사용자의 다양한 배경과 언어 스타일에 따라 지침을 처리할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: InstructCell은 기존의 단일 세포 기초 모델들의 성능을 넘는 결과를 보이며, 각 아키텍처 요소의 필요성을 검증하는 실험을 통해 생물학적 통찰력을 발견하는 능력을 갖추고 있습니다. 이러한 기술은 사용자가 복잡한 단일 세포 데이터를 탐색할 수 있도록 지원하며, 기술 장벽을 낮추고 생물학에 대한 깊은 통찰력을 가능하게 합니다. InstructCell은 유연한 자연어 명령을 사용하여 실질적이고 중요한 생물학적 작업을 수행할 수 있는 직관적인 도구를 제공합니다.



### Revolutionizing Communication with Deep Learning and XAI for Enhanced Arabic Sign Language Recognition (https://arxiv.org/abs/2501.08169)
Comments:
          13 pages, 25 figures, 16 tables

- **What's New**: 이 연구는 MobileNetV3, ResNet50, EfficientNet-B2와 같은 최신 딥 러닝 모델을 사용하여 아랍 수화(Arabic Sign Language, ArSL) 인식에 대한 통합 접근 방식을 소개합니다. 이 모델들은 해석 가능성과 투명성을 높이기 위해 Explainable AI (XAI) 기법으로 추가 강화되었습니다. 연구 결과, EfficientNet-B2는 각각 99.48%와 98.99%의 최고 정확도를 달성했습니다. 이 시스템은 인식 정확도에서 새로운 기준을 설정하고 다문화 소통 기술의 포함성을 강조합니다.

- **Technical Details**: 이 연구의 주요 기여는 다양한 손 제스처를 인식하기 위해 정교한 데이터 증강 방법과 계층적 5-겹 교차 검증을 사용한 것입니다. 또한, Grad-CAM을 통해 모델 결정의 투명성을 강화하였으며, 실시간 인식 정확도를 유지하면서 다양한 환경의 데이터를 처리할 수 있는 능력을 발휘했습니다. 연구는 또한 아랍어 수화 인식에 대한 기존 접근 방식을 개선하기 위해 ResNet50, MobileNetV3, EfficientNet-B2와 같은 최첨단 모델을 도입했습니다.

- **Performance Highlights**: 연구에서 제안된 시스템은 기존의 최고 모델들과 비교하여 우수한 인식 정확도를 보여주었으며, 특히 의료 및 교육 분야에서 사용될 때 투명한 결정 과정을 보장할 수 있습니다. 해석 가능성은 민감한 분야에서 중요한 요소로 작용하며, 이는 사용자의 신뢰를 높이는 데 기여합니다. 또한, 그 시스템은 다른 수화에 대한 적용 가능성을 가지며, 다양한 데이터 시나리오를 효과적으로 처리할 수 있도록 설계되었습니다.



### Potential and Perils of Large Language Models as Judges of Unstructured Textual Data (https://arxiv.org/abs/2501.08167)
Comments:
          11 pages, 1 appendix

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 다른 LLM이 생성한 요약의 주제 일치를 평가하는 것으로 사용될 수 있는지 조사합니다. 대형 언어 모델의 발전이 비구조적 텍스트 데이터의 처리 및 요약에 눈에 띄는 능력을 제공하였으며, 이는 설문 응답 분석에 특히 중요한 의미를 갖습니다. 이 연구는 LLM을 사용하여 생성된 요약의 주제와 실제 응답 간의 불일치가 조직의 의사 결정에 미치는 영향에 대한 우려를 다룹니다.

- **Technical Details**: 연구에서는 Anthropic Claude 모델을 이용하여 개방형 설문 응답으로부터 주제 요약을 생성하고, Amazon의 Titan Express, Nova Pro, Meta의 Llama를 LLM 평가자로 활용했습니다. LLM을 평가자로 사용하는 접근 방식은 Cohen의 kappa, Spearman의 rho, Krippendorff의 alpha를 사용하여 인간 평가와 비교하였습니다. 이 방법은 전통적인 인간 중심 평가 방법에 대한 확장 가능한 대안을 검증합니다.

- **Performance Highlights**: 결과적으로 LLM을 평가자로 사용하는 접근 방식은 인간 평가자와 비슷한 성능을 보였으나, 인간은 여전히 맥락에 따라 미세한 뉘앙스를 탐지하는 데 있어 우수할 수 있습니다. 이 연구는 AI 지원 텍스트 분석에 대한 지식을 확장하는 데 기여하며, 다양한 맥락과 사용 사례에 LLM 평가 모델을 일반화할 때 신중한 고려가 필요함을 강조합니다.



### I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution (https://arxiv.org/abs/2501.08165)
Comments:
          12 pages, 5 figures,

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용한 소스 코드 저자 식별 기법을 제안합니다. 기존의 기계 학습(machine learning) 및 딥러닝(deep learning) 방법이 한정적인 labeled data(라벨링된 데이터)에 의존하는 반면, LLMs는 적은 데이터에서도 높은 성능을 발휘할 수 있습니다. 논문의 실험 결과, LLMs는 서로 다른 프로그래밍 언어 간의 코드 저자 식별 작업에서 우수한 성능을 보였으며, 0-shot 및 few-shot 학습 방식으로 저자 식별이 가능함을 입증했습니다.

- **Technical Details**: 이 연구에서는 ChatGPT, Gemini, Mistral, Llama와 같은 4개의 주요 LLM 모델을 평가하였습니다. LLMs는 적은 수의 참고 코드(snippet)만으로도 저자 식별을 수행할 수 있으며, C++와 Java 데이터를 활용한 실험에서 각각 65%와 68.7%의 분류 정확도를 기록하였습니다. 특히, 제안된 토너먼트 스타일 접근 방식(tournament-style approach)은 LLM의 token 제한 문제를 해결하고 대규모 저자 식별을 가능하게 합니다.

- **Performance Highlights**: LLMs는 0-shot 환경에서 Matthews Correlation Coefficient(MCC) 0.78을 기록하였으며, few-shot 학습을 통해서도 MCC 0.77을 달성했습니다. 또한, LLM들은 잘못된 저자 식별(misattribution) 공격에 대한 견고성도 보였습니다. 이러한 결과는 LLMs가 코드 저자 식별 작업에 효과적으로 적용될 가능성을 밝히며, 사이버 보안 및 소프트웨어 엔지니어링 분야에서 새로운 응용 가능성을 열어줍니다.



### FairTTTS: A Tree Test Time Simulation Method for Fairness-Aware Classification (https://arxiv.org/abs/2501.08155)
- **What's New**: FairTTTS라는 새로운 후처리 방식의 편향 완화 기법을 제안했습니다. 이 기법은 기존의 Tree Test Time Simulation(TTTS) 방법에서 영감을 받아 정확성과 공정성을 동시에 개선할 수 있도록 설계되었습니다. FairTTTS는 결정트리의 결정 경로를 조정함으로써 불이익을 받은 샘플에 대해 공정한 예측 결과를 제공합니다.

- **Technical Details**: FairTTTS는 Monte Carlo 시뮬레이션 기반의 기법을 사용하여 보호 속성과 관련된 노드에서 결정 경로를 확률적으로 변경합니다. 이러한 조정은 의사결정 기준에 가까운 샘플이 더 높은 확률로 변경될 수 있도록 도와주며, 이를 통해 결정트리의 구조에 내재된 편향을 줄이는 데 기여합니다. 또한 FairTTTS는 훈련 프로세스에서 공정성 조정을 분리하여 다양한 환경에 유연하게 통합될 수 있도록 합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, FairTTTS는 기존의 방법보다 공정성 개선에서 평균 20.96%의 성과를 보여주었으며, 0.55%의 정확도 향상을 이뤄냈습니다. 반면, 경쟁 방법들은 일반적으로 0.42%의 정확도 감소를 초래했습니다. 이러한 결과는 FairTTTS가 예측 성능을 향상시키면서 더 공정한 의사결정을 가능하게 함을 입증합니다.



### Refusal Behavior in Large Language Models: A Nonlinear Perspectiv (https://arxiv.org/abs/2501.08145)
- **What's New**: 이 논문은 대형 언어 모델(LLM)에서의 거부(Refusal) 행동을 조사하였으며, 이를 통해 윤리적인 기준에 부합하도록 유해하거나 부적절한 프롬프트에 응답하지 않도록 하는 기전을 탐구하였습니다. 다양한 구조의 LLM 6종에서 거부 메커니즘의 비선형적이고 다차원적인 특성을 발견하였고, 이는 기존의 선형적 접근 방식에 도전하는 내용입니다. 결과적으로, 이러한 발견은 LLM의 안전한 배치를 위한 보다 나은 연구 방향을 제시합니다.

- **Technical Details**: 연구 방법으로는 거부 행동을 구분 짓기 위한 실험적 과제를 설계하고, 이를 통해 모델의 반응을 분석하였습니다. 두 개의 데이터셋을 사용하여 유해한 지침과 무해한 지침에 대한 LLM의 반응을 추적하였습니다. 선형(PCA) 및 비선형(t-SNE, UMAP) 차원 축소 기법을 통해 각 레이어에서의 활성화 복잡성을 분석하며, 이러한 활성화가 구조마다 어떻게 다른지를 규명하였습니다.

- **Performance Highlights**: 연구 결과는 LLM의 거부 메커니즘이 단순한 선형적 현상이 아닌, 복잡하고 비선형적이라는 것을 보여주었습니다. 우리는 새로운 활성화 클러스터가 등장하는 것을 확인하였으며, 이를 통해 각 모델 아키텍처의 특정적인 특성과 더불어 거부 행동의 본질적인 이해가 가능하게 되었습니다. 이 발견들은 향후 LLM을 윤리적 기준에 맞추기 위한 중요한 기초 자료로 활용될 것입니다.



### EEG-ReMinD: Enhancing Neurodegenerative EEG Decoding through Self-Supervised State Reconstruction-Primed Riemannian Dynamics (https://arxiv.org/abs/2501.08139)
- **What's New**: EEG-ReMinD(EEG 복원-프라이밍 리만 역학) 모델은 뇌-컴퓨터 인터페이스 및 질병 진단을 향상시키기 위해 고안된 신개념의 두 단계 접근법입니다. 이 접근법은 데이터의 라벨 의존성을 줄이고, 자가 감독 학습(self-supervised learning) 및 기하학적(features of geometry) 학습기법을 활용하여 EEG 데이터를 보다 효과적으로 분석할 수 있도록 설계되었습니다. 특히 이 연구는 신경퇴행성 질환에서의 EEG의 맥락을 반영하기 위해 리만 기하학(Riemannian geometry)를 적용한 점이 특징적입니다.

- **Technical Details**: EEG-ReMinD는 무감독(pre-training) 단계와 미세 조정(fine-tuning) 단계를 포함하여 라벨이 부족한 데이터에도 효율적으로 적응합니다. 첫 번째 단계에서는 자가 감독 학습을 통해 리만 기하학적 표현을 구성 및 복재하여 내부 상태를 설정하는 과정을 거칩니다. 두 번째 단계는 미리 훈련된 모델을 활용해 뇌의 상태를 인식하며, 이 과정은 제한된 양의 라벨이 있는 데이터를 사용하여 수행됩니다.

- **Performance Highlights**: 두 가지 신경퇴행성 질환에서의 비교 분석 결과, EEG-ReMinD는 기존의 방법들에 비해 뛰어난 성능을 보였습니다. 이 연구에서 개발된 새로운 두 단계 기법은 동적 EEG 특성을 효과적으로 처리하며, 노이즈와 잡음의 영향을 줄이고 모델의 견고성을 향상시키는 데 기여합니다. 이러한 성능 향상은 자가 감독 학습의 도입과 리만 기하학적 기법의 결합 덕분에 가능해졌습니다.



### An Empirical Wall-Pressure Spectrum Model for Aeroacoustic Predictions Based on Symbolic Regression (https://arxiv.org/abs/2501.08134)
- **What's New**: 이 논문은 현재의 최첨단 예측 모델을 개선하는 새로운 wall-pressure spectrum (WPS) 경험 모델을 도입합니다. 이 모델은 다양한 공기 날개(airfoils) 및 흐름 조건을 적용할 수 있는 범위를 확대하여 저항성과 정확성을 높이도록 설계되었습니다. AI 기반의 symbolic regression 접근 방식을 통해 개발된 이 모델은 추가적인 특수 공식을 제공할 필요 없이 실험 데이터로부터 해석 가능한 수식을 도출할 수 있습니다.

- **Technical Details**: 모델은 두 가지 유형의 공기 날개(NACA 0008 및 NACA 63018)에서 다양한 공격각과 유입 속도에서 측정된 wall-pressure 변동 데이터셋을 사용하여 개발되었습니다. 이 모델은 힘든 압력 구배(pressure gradient)에 대한 민감도를 고려하여 일반적인 포괄성을 갖추고 있습니다. 실험 데이터와의 검증을 통해 모델은 잘 알려진 반경계(stress) 및 혼합 모델에 비해 월폭 압력 변동(WPS)의 예측 정확도에서 우수성을 입증하였습니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 Amiet의 이론과 통합되어 실제 풍력 터빈의 aeroacoustic noise를 예측하는 데 성공하였으며, 실험 측정 결과와 우수한 일치를 보였습니다. 기존의 경험적 모델들에 비해 더 강력하고 일반화가 가능한 예측 능력을 입증하였으며, 새로운 접근 방식을 통해 더욱 정교한 noise 예측이 가능하다는 점에서 매우 중요한 진전을 이루었습니다.



### Data-driven inventory management for new products: A warm-start and adjusted Dyna-$Q$ approach (https://arxiv.org/abs/2501.08109)
Comments:
          7 pages, 2 figures

- **What's New**: 이 논문에서는 새로운 제품에 대한 재고 관리를 위한 향상된 강화 학습 알고리즘을 제안합니다. 기존 데이터가 부족한 상황에서도 모델 기반과 모델 프리 접근을 균형 있게 결합하여 Dyna-Q 구조를 개선하고 학습 시간을 단축할 수 있는 방법을 소개합니다. 특히 '와씀 스타트 정보(warm-start information)'를 이용하여 초기 학습 안정성을 높이고 최적의 정책 추정의 변동성을 줄일 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 Dyna-Q 알고리즘의 구조를 기반으로 하여 새로운 제품의 수요 분포가 알려지지 않았을 때 최적의 주문 수량을 결정하는 데 초점을 맞추고 있습니다. 이 알고리즘은 '탐색-수렴(search-then-convergence)' 과정을 통해 의사결정 과정에서 발생할 수 있는 불확실성을 줄이며, 제한된 과거 데이터를 보완하기 위해 기존의 유사 제품에서 수집한 시뮬레이션 데이터를 활용합니다. 이를 통해 모델 기반과 모델 프리 컴포넌트 간의 비율을 동적으로 조절하는 기능이 포함되어 있습니다.

- **Performance Highlights**: 케이스 스터디를 통해 제안된 알고리즘은 평균 일일 비용을 최대 23.7% 줄이고, 학습 시간을 최대 77.5% 단축하는 성과를 보였습니다. 또한, 30일 간의 테스트에서도 가장 낮은 총 비용과 총 비용의 변동성, 그리고 상대적으로 낮은 부족 비율을 기록하며 기존 알고리즘들에 비해 높은 성능을 입증했습니다.



### Consistency of Responses and Continuations Generated by Large Language Models on Social Media (https://arxiv.org/abs/2501.08102)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 소셜 미디어에서 감정 콘텐츠를 처리하고 의미론적 관계를 유지하는 방법을 조사했습니다. Gemma와 Llama라는 두 개의 오픈소스 모델을 활용하여 기후 변화에 대한 트위터와 레딧의 논의를 분석하였으며, LLM이 생성한 텍스트의 감정 일관성과 의미 유사성을 평가했습니다. 이는 LLM의 감정 및 의미 처리 능력에 대한 귀중한 통찰을 제공합니다.

- **Technical Details**: 연구는 LLM의 두 가지 주요 텍스트 생성 작업인 응답 생성(response generation)과 연속성 작업(continuation tasks)에 중점을 두었습니다. Gemma는 부정적인 감정 증폭 경향을 보였고, Llama는 더 넓은 감정 스펙트럼에서 감정 보존에서 뛰어난 성과를 나타냈습니다. 두 모델 모두 인간이 작성한 내용에 비해 감정 강도가 줄어든 응답을 생성하고, 긍정적인 감정으로의 편향을 보였습니다.

- **Performance Highlights**: 연구 결과, LLM 모델은 높은 의미 일관성을 유지했지만, 감정 패턴에서의 뚜렷한 차이를 보여주었습니다. 특히, Gemma는 분노와 같은 부정적인 감정을 강화하는 경향을 보인 반면, Llama는 다양한 감정에서 뛰어난 보존성을 보였습니다. 이러한 발견은 LLM의 감정 및 의미 처리 능력을 이해하고, 소셜 미디어와 인간-AI 상호작용 디자인에서의 활용 가능성을 논의하는 데 기여합니다.



### Guiding the classification of hepatocellular carcinoma on 3D CT-scans using deep and handcrafted radiological features (https://arxiv.org/abs/2501.08097)
Comments:
          IEEE ISBI 2025

- **What's New**: 이 연구는 간세포암(hepatocellular carcinoma, HCC) 진단을 자동화하기 위한 새로운 접근법을 제안한다. 기존의 깊은 학습(deep learning) 방법이 CT 스캔에서 HCC를 정확하게 예측하는 데 실패했음을 보여주고, LI-RADS 시스템에서 영감을 받아 성능을 개선하는 2단계 접근 방식을 제안하였다. 본 연구는 HCC의 진단을 위한 새로운 자동화된 방법을 통해 방사선과 의사 사이의 변동성을 줄이는 것을 목표로 한다.

- **Technical Details**: 이 방법은 HCC의 조직학적 확인을 기반으로 하여 실제로 사용되는 LI-RADS 점수 체계를 사전 학습 작업으로 활용한 것이다. LI-RADS의 주요 3가지 특징 예측을 기반으로 한 3개의 딥 러닝 모델을 도입하여 최종 HCC 분류를 안내한다. 또한, 수작업으로 설계한 특징과 LI-RADS 기반의 딥 러닝 방사선학적 특징을 결합하여 분류 성능을 향상시킨다.

- **Performance Highlights**: 제안된 방법은 6점에서 18점까지 AUC(Area Under the Curve) 개선을 달성하였으며, 비전문 방사선과 의사보다 더 우수한 결과를 보였다. 임상 검증을 통해 전문가와 비전문가의 진단 결과와 비교하였으며, 관련 데이터베이스에서의 성능이 모두 우수함을 확인하였다. 이러한 방법은 어려운 HCC 진단을 위한 새로운 패러다임을 제공한다.



### Hybrid Action Based Reinforcement Learning for Multi-Objective Compatible Autonomous Driving (https://arxiv.org/abs/2501.08096)
Comments:
          12 pages, 9 figures, 5 tables

- **What's New**: 본 논문에서는 다양한 자율 주행 시나리오에서 다중 목표 호환성(multi-objective compatibility)을 갖춘 Reinforcement Learning(RL) 메서드를 제안합니다. Hybrid Parametrized Action space(HPA)은 추상적인 안내와 구체적인 제어 명령을 결합한 하이브리드 주행 동작을 생성합니다. 또한, 다중 속성 보상을 고려한 다중 목표 비평가 아키텍처를 통해 다양한 주행 목표에 동시 집중할 수 있도록 합니다.

- **Technical Details**: 저자들은 Hybrid Parametrized Action space를 기반으로 한 Multi-objective Ensemble-Critic RL 방법(HPA-MoEC)을 설계했습니다. 이 메서드는 이산 행동 세트와 연속 파라미터를 포함하여 주행 동작을 생성하며, 여러 보상 함수를 통해 속성을 분리하는 방식으로 작동합니다. 이와 함께, 불확실성 기반 탐색 전략을 도입하여 에이전트가 실용적인 주행 정책을 더 빠르게 발견할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 주행 효율성, 행동 일관성, 안전성 측면에서 다중 목표 호환성 자율 주행을 성공적으로 달성했습니다. 실험은 시뮬레이션된 교통 환경과 HighD 데이터셋에서 수행되었고, 제안된 메서드가 훈련 효율성을 크게 증가시키며 주행 성능을 향상시키는 것을 보여주었습니다.



### Hierarchical Autoscaling for Large Language Model Serving with Chiron (https://arxiv.org/abs/2501.08090)
- **What's New**: 본 논문에서 소개된 Chiron은 LLM 서빙(LLM serving)을 위한 새로운 자동 확장기(autoscaler)로, 요청의 SLO(requirement SLO)를 고려하여 리소스를 보다 효율적으로 관리합니다. 기존의 자동 확장기는 SLO를 무시하여 불필요한 자원의 스케일링과 자원 활용의 비효율성을 초래했으나, Chiron은 이를 해결합니다. 이 시스템은 계층적 백프레셔(hierarchical backpressure) 개념을 도입하여 큐의 크기, 활용도, 그리고 SLO를 기반으로 합니다.

- **Technical Details**: Chiron은 요청의 SLO를 바탕으로 LLM 추론(inference) 요청을 상호작용 요청(interactive requests)과 배치 요청(batch requests)으로 구분하여 관리합니다. 이 자동 확장기는 복잡한 환경에서도 SLO를 유지할 수 있도록 설계되어 있으며, 실험을 통해 GPU 활용도를 70%까지 향상시킵니다. 또한, 요청의 도착률, 멀티플렉싱(multiplexing) 및 구성 매개변수(configuration parameters)가 SLO에 미치는 영향을 반영하여 자원 관리에 큰 기여를 합니다.

- **Performance Highlights**: Chiron을 사용한 실험 결과는 기존 솔루션들에 비해 SLO 달성률이 최대 90% 높아지는 것으로 나타났습니다. 이는 LLM 서빙에서의 성능 개선을 의미하며, 클라우드 제공자(cloud providers)에게는 더 나은 자원 관리 옵션을 제공합니다. 이러한 성과는 데이터 센터의 운영 효율성을 높이고, 사용자 경험을 향상시키는 데 기여할 것입니다.



### Optimizing Speech Multi-View Feature Fusion through Conditional Computation (https://arxiv.org/abs/2501.08057)
Comments:
          ICASSP 2025

- **What's New**: 본 연구에서는 self-supervised learning (SSL) 특징과 전통적인 spectral 특징인 FBanks 간의 업데이트 방향에서의 갈등을 해결하기 위해 새로운 generalized feature fusion framework를 제안합니다. 이 프레임워크는 gradient-sensitive gating network와 multi-stage dropout 전략을 통해 다양한 입력 특징의 통합을 가능하게 합니다. 이를 통해 SSL과 spectral 특징을 결합함으로써 모델의 수렴 속도를 가속화하고, 다중 음성 번역 작업에서 성능을 유지할 수 있게 됩니다.

- **Technical Details**: 우리는 end-to-end 음성 번역 시스템을 채택하며, 이 시스템은 acoustic encoder (A-enc), textual encoder (T-enc), decoder (dec)로 구성됩니다. 본 연구에서는 채택된 gradient-sensitive gating network (GSGN)가 S-feature와 FBanks 특징을 동적으로 융합하며, 다단계의 dropout 전략으로 과적합을 방지합니다. 이 구조는 residual network를 기반으로 하며, 새로운 특징을 통합하는 과정에서 각 특징의 가중치를 조정하여 최적의 결합을 달성합니다.

- **Performance Highlights**: MUSTC 데이터셋의 여러 언어 (En-De, En-Es, En-Fr)에 대한 실험 결과, 제안된 방법은 기존 FBanks 특징을 사용한 모델에 비해 평균 1.24배의 훈련 속도를 제공하면서도 유사한 성능을 달성했습니다. 또한, 미리 훈련된 ST 모델과 비교할 때도, SSL 특징이 충분히 유용할 경우 해당 훈련 가속 효과를 보장합니다. 이를 통해 음성-텍스트 복잡한 작업에서 두 가지 특징의 강점을 활용할 수 있음을 보여줍니다.



### Exploring Narrative Clustering in Large Language Models: A Layerwise Analysis of BER (https://arxiv.org/abs/2501.08053)
Comments:
          arXiv admin note: text overlap with arXiv:2408.03062, arXiv:2408.04270, arXiv:2307.01577

- **What's New**: 이번 연구는 BERT의 내부 메커니즘을 분석하여 서사 내용과 저자 스타일의 클러스터링 능력을 살펴보았습니다. GPT-4를 활용하여 개발한 다양한 의미적 콘텐츠와 스타일적 변화를 포함한 데이터를 사용하여 BERT의 레이어별 활성화를 분석했습니다. 연구 결과, BERT는 후반 레이어에서 강력한 서사 내용의 클러스터링을 보여주며, 스타일은 개별 작가보다 서사로 더 잘 클러스터링되었습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 다양한 저자 스타일과 서사 주제를 포함하도록 선택되었습니다. BERT의 [CLS] 토큰 임베딩은 768차원으로, 이를 13개의 레이어에서 추출하여 (13, 1000, 768) 형태의 데이터 구조를 만들었습니다. 분석을 위해 주성분 분석(Principal Component Analysis, PCA)과 다차원 스케일링(Multidimensional Scaling, MDS) 기법을 활용하여 BERT의 내부 활성화를 시각화했습니다.

- **Performance Highlights**: BERT는 강력한 서사 내용 클러스터링을 보여주며, 이러한 클러스터는 후속 레이어에서 점점 더 컴팩트하게 형성됩니다. 반면, 저자 스타일에 따른 클러스터링은 미미하게 나타났습니다. 이 연구는 BERT가 의미적 콘텐츠를 더 우선시한다는 점을 강조하며, 변환기 모델의 언어 정보 인코딩 방식에 대한 이해를 향상시키는 데 기여합니다.



### Building Symbiotic AI: Reviewing the AI Act for a Human-Centred, Principle-Based Framework (https://arxiv.org/abs/2501.08046)
Comments:
          First version: 17 pages, 5 figures, 2 tables

- **What's New**: 최근 인공지능(AI)의 빠른 발달로 인해, AI의 설계와 개발, 사용에 대한 규제가 필수적이라는 인식이 높아지고 있습니다. 특히, 유럽연합(EU)은 AI와 사람 간의 상호작용을 안전하게 하는 리스크 기반 접근 방식을 가진 AI 법안(AI Act)을 발표했습니다. 이러한 법적 틀은 AI 시스템의 윤리적 사용을 보장하고 인간의 권리를 보호하기 위한 새로운 시도로 주목받고 있습니다.

- **Technical Details**: 연구자들은 Symbiotic AI(공진화 인공지능)라는 새로운 접근법을 제안하며, 이는 AI와 인간의 깊고 지속적인 협력을 통해 인간의 능력을 향상시키는 것을 목표로 하고 있습니다. 이 논문은 HCAI(인간 중심 인공지능) 시스템을 설계하고 개발하기 위한 네 가지 원칙인 투명성(Transparency), 공정성(Fairness), 자동화 수준(Automation Level), 보호(Protection)를 제시합니다. 또한, 신뢰성(Trustworthiness), 견고성(Robustness), 지속 가능성(Sustainability)이라는 세 가지 특성이 추가로 나타났습니다.

- **Performance Highlights**: 이 연구는 AI 시스템 설계 시 인간 중심의 접근법을 채택하여 인간과 AI 간의 공진화 관계를 확립하는 것이 중요하다고 강조합니다. AI 시스템이 윤리적 기준을 준수하도록 보장하기 위해, 고위험 AI 모델은 유럽연합 관리의 데이터베이스에 등록해야 합니다. 본 논문은 AI Act에 부합하는 SAI 시스템 설계에 관한 원칙 및 특성이 문헌에서 도출된 것임을 명확히 하여 향후 연구를 위한 기초 자료를 제공합니다.



### Exploring visual language models as a powerful tool in the diagnosis of Ewing Sarcoma (https://arxiv.org/abs/2501.08042)
Comments:
          11 pages, 5 figures, 2 tables. Oral presentation at KES-InMed 2024 held in Madeira, Portugal

- **What's New**: Ewing의 육종(Ewing's sarcoma, ES)에 대한 새로운 접근 방식을 제시하는 이 연구는, histopathological 이미지 분석에서 인공지능 기술의 가능성을 탐구하고 있습니다. 특히, 다양한 pre-training 전략을 통해 ES를 유사한 형태를 가진 다른 연부 조직이나 뼈 육종과 구분하고자 하는 것을 목표로 합니다. Vision-language supervision (VLS)를 통해 진단 정확도를 크게 향상시킬 수 있음을 보여주며, 이는 AI 기반 시스템이 ES 진단에 기여할 수 있는 가능성을 시사합니다.

- **Technical Details**: 이 연구는 Multiple Instance Learning (MIL) 방법론을 활용하여 Ewing의 육종을 진단하는 새로운 패러다임을 제안합니다. Histological 이미지에서 높은 수준의 특징적 표현을 추출하기 위해 Pathology language-image pretraining (PLIP)과 Transformer 기반의 강력한 임베딩 집계기를 사용했습니다. 최종적으로, 여러 데이터셋에서 학습된 VGG 기반 아키텍처와 비교하여 제안하는 방법의 경쟁력을 입증하였습니다.

- **Performance Highlights**: 실험 결과, VLS의 활용으로 진단 정확도가 크게 향상되었으며, 이는 모델의 예측 클래스 정확도뿐만 아니라 훈련 가능 매개변수의 수와 계산 비용을 획기적으로 감소시켰습니다. 특히, Ewing의 육종을 다른 연부 조직 및 뼈 육종과 구분하기 위한 효과적인 방법임을 강조하며, 이는 향후 진단 정확성을 더욱 높이는 데 중요한 기여를 할 것으로 기대됩니다.



### READ: Reinforcement-based Adversarial Learning for Text Classification with Limited Labeled Data (https://arxiv.org/abs/2501.08035)
- **What's New**: 이번 논문에서는 지식이 미비한 레이블 데이터(labelled data)로부터의 개선 방법으로서 새로운 접근인 READ(신뢰 기반 적대적 학습)를 제안합니다. 대체로 레이블 데이터는 수집하기 어렵고 비용이 많이 들어가지만, 레이블이 없는 데이터(unlabelled data)는 비교적 저렴하게 획득할 수 있습니다. 따라서 이 연구는 강화 학습(reinforcement learning) 기반 텍스트 생성과 반지도 학습(semi-supervised learning)을 결합하여 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: READ(강화 기반 적대적 학습)는 레이블 데이터와 레이블이 없는 데이터의 조합을 활용하여 텍스트 생성 및 모델 훈련을 진행합니다. 이 방법은 강화 학습 프레임워크를 통해 합성 텍스트(synthetic text)를 생성하고, 이러한 과정을 적대적 학습(adversarial learning)과 연결하여 모델의 일반화 능력을 향상시킵니다. 실험을 통해 READ는 여러 데이터셋에서 기존의 반지도 학습 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: READ 방법은 다양한 프리트레인(pre-trained) 모델을 통해 여러 데이터셋에서 기존의 최첨단(state-of-the-art) 방법을 초월하는 성과를 보였습니다. 생성하는 텍스트의 질과 모델의 일반화 능력을 평가한 결과, READ가 텍스트 생성 품질과 더불어 모델의 일반화 능력에 긍정적인 영향을 미친다는 것을 경험적으로 입증하였습니다. 이 연구는 텍스트 생성과 적대적 학습의 통합이 모델 성능에 미치는 중요성을 강조합니다.



### An AI-driven framework for rapid and localized optimizations of urban open spaces (https://arxiv.org/abs/2501.08019)
Comments:
          36 pages

- **What's New**: 이번 연구에서는 도시 공간에서의 열적 쾌적도와 안전성 인식에 중요한 영향을 미치는 Sky View Factor(SVF)와 가시성을 최적화하기 위한 AI 주도의 프레임워크를 소개합니다. 기계 학습 모델(MLMs)과 설명 가능한 AI 기술을 통합하여 저비용의 점진적 디자인 개선을 지원합니다. 이는 전 세계 최적화 방법이 아닌, 더 낮은 계산 비용으로 지역 조정을 가능하게 합니다.

- **Technical Details**: 이 프레임워크는 SHapley Adaptive Explanations(SHAP)를 사용하여 특성의 중요성을 분석하고, Counterfactual Explanations(CFXs)를 통해 최소 디자인 변경을 제안합니다. 연구에서 테스트된 다섯 개의 MLM 중 XGBoost가 가장 정확한 성능을 보였으며, 건물 폭, 공원 면적, 주변 건물의 높이가 SVF에 중요한 요소로 확인되었습니다.

- **Performance Highlights**: CFX 접근법은 1분 내에 최적의 결과를 달성하며, 5% RMSE 오차로 유전자 알고리즘보다 훨씬 빠른 성능을 보여줍니다. 이 대조 연구는 데이터 기반 통찰력과 실용적인 리트로핏(retrofitting) 솔루션을 통해 다양한 도시 맥락에서의 환경 품질 향상에 기여합니다.



### Tutorial: VAE as an inference paradigm for neuroimaging (https://arxiv.org/abs/2501.08009)
Comments:
          18 pages, 4 figures

- **What's New**: 이번 튜토리얼에서는 Variational Autoencoders (VAEs)의 이론적 기초와 실용적인 도전 과제를 다루며, 신경영상(neuroimaging) 데이터 분석에의 응용을 강조합니다. VAEs는 고차원 데이터의 해석 가능한 잠재 표현을 생성하고, 베이지안 추론(Bayesian inference) 원리를 통합하여 데이터의 구조적 패턴을 이해하는 데 기여합니다. 마지막으로, VAE의 구현 시 발생할 수 있는 수렴(convergence) 문제와 과적합(overfitting) 등의 문제를 해결하기 위한 전략을 소개합니다.

- **Technical Details**: VAEs는 전통적인 오토인코더(autoencoder)와 달리 잠재 표현의 확률 분포(probabilistic distribution)를 학습합니다. 이를 통해 관측 데이터(x)와 숨겨진 변수(z) 간의 관계를 모델링하고, 데이터의 잠재 구조를 효과적으로 표현할 수 있습니다. 이 과정에서, VAEs는 재매개화 트릭(reparameterization trick)을 활용하여 학습 안정성과 효율성을 높입니다.

- **Performance Highlights**: 신경영상 분야에서 VAEs의 활용 가능성을 강조하며, 퇴행성 신경 질환(neurodegenerative diseases)의 경우와 같은 복잡한 뇌 데이터의 의미 있는 패턴을 발견할 수 있는 잠재력을 살펴봅니다. VAEs는 다중 모달(multimodal) 데이터의 분석에 필요한 강력한 생성 모델로서의 역할을 하며, 기존의 머신러닝 접근 방식을 보완하는 데 유용합니다.



### TriAdaptLoRA: Brain-Inspired Triangular Adaptive Low-Rank Adaptation for Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2501.08008)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 효율적인 미세 조정을 위한 새로운 방법인 Triangular Adaptive Low-Rank Adaptation (TriAdaptLoRA)를 제안합니다. TriAdaptLoRA는 신경과학 원리를 기반으로 하여 동적으로 학습 가능한 매개변수의 할당을 최적화하여 미세 조정의 효율성을 높입니다. 이 방법은 이전의 Parameter-Efficient Fine-Tuning (PEFT) 기법들에서 보여준 한계를 극복하기 위해 고안되었습니다.

- **Technical Details**: TriAdaptLoRA는 다음 세 가지 주요 혁신을 포함합니다. 첫째, 변환 행렬을 하삼각형 및 상삼각형 구성 요소로 분할하여 매개변수 활용도를 극대화합니다. 둘째, 효율적인 적응을 위한 중요도 척도로 정규화된 Frobenius 놈을 사용합니다. 셋째, 동적 임계값에 따라 조정되는 적응형 순위 성장 전략을 통해 학습 단계 간 유연한 매개변수 할당이 가능합니다.

- **Performance Highlights**: 실험 결과, TriAdaptLoRA는 다양한 자연어 이해 및 생성 작업에서 기존 PEFT 방법들보다 일관되게 우수한 성능을 나타냈습니다. 특히, GLUE 벤치마크와 SQuAD 2.0 작업에서 현저한 성과 향상을 기록하여 자원 집약적 문제를 해결하는 효과적인 미세 조정 솔루션으로 자리매김했습니다.



### DisCoPatch: Batch Statistics Are All You Need For OOD Detection, But Only If You Can Trust Them (https://arxiv.org/abs/2501.08005)
- **What's New**: 이 논문에서는 데이터 분포의 미세한 변화인 covariate shifts를 탐지하여 OOD(Out-of-Distribution) 감지 성능을 향상시킬 수 있다는 가설을 제기합니다. 기존의 OOD 탐지 방식은 주로 semantic shifts와 domain shifts에 중점을 두었으나, 본 연구는 이러한 covariate shifts에 대한 이해를 바탕으로 새로운 접근 방식을 제안합니다. 특히, Batch Normalization(BN) 조건에서의 adversarial 설정을 활용하여 OOD 탐지를 위한 새로운 프레임워크인 DisCoPatch를 소개합니다.

- **Technical Details**: DisCoPatch는 unsupervised Adversarial Variational Autoencoder(VAE) 프레임워크로서, 모델이 이미지의 다양한 패치를 기반으로 일관된 데이터 분포에서 작동하도록 설계되었습니다. 기법적으로, DisCoPatch는 변 reconstructed된 이미지와 생성된 이미지의 suboptimal outputs를 OOD 샘플로 활용하여 discriminator의 학습을 진행합니다. 이런 방식으로 in-distribution 샘플과 covariate shifts 간의 경계를 더 엄격하게 설정함으로써 OOD 탐지 성능이 향상됩니다.

- **Performance Highlights**: DisCoPatch는 공공 OOD 탐지 벤치마크에서 state-of-the-art 성능을 달성하며, ImageNet-1K(-C) 데이터셋에서 95.5%의 AUROC를 기록했습니다. 또한, Near-OOD 탐지에서도 95.0%의 성능을 보여줍니다. 25MB의 소형 모델로 설계되어 기존 방법보다 낮은 지연 시간으로 높은 OOD 탐지 성능을 제공하여 실제 OOD 탐지에 효과적이고 실용적인 솔루션임을 증명합니다.



### Maximizing Uncertainty for Federated learning via Bayesian Optimisation-based Model Poisoning (https://arxiv.org/abs/2501.08002)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 Federated Learning (FL) 내에서의 악의적인 모델 오염 공격을 탐구하는 새로운 방법론을 제안합니다. 저자들은 새로운 모델 오염 공격 기법인 Delphi를 소개하며, 모델 출력을 최대한의 불확실성(unexpected uncertainty)으로 만드는 것을 목표로 합니다. 모델의 첫 번째 은닉층(hidden layer)과 관련된 불확실성을 활용하여 공격 기법의 효과성을 증명하고, FL 시스템의 취약성을 강조합니다.

- **Technical Details**: Delphi 방법론은 Bayesian Optimisation 및 Least Squares Trust Region 두 가지 최적화 기술을 사용하여 최적의 오염 모델 매개변수를 탐색합니다. 이 기법은 특정 뉴런에만 집중하여 모델 매개변수를 조작함으로써 불확실성을 유도합니다. 이 과정에서 저자들은 KL Divergence를 통해 예측 확률 분포의 거리를 최소화하여 불확실성을 정량화하는 방법을 수립했습니다.

- **Performance Highlights**: 실험 결과, Delphi-BO 방법이 Delphi-LSTR보다 높은 불확실성을 유도함을 보여줍니다. Delphi-BO는 매 학습 라운드에서 가장 중요한 뉴런을 선택하여 공격을 수행하며, 그에 따라 모델의 예측 신뢰도가 절반으로 감소하는 것으로 나타났습니다. 이러한 결과는 FL 모델이 모델 오염 공격에 대해 얼마나 취약한지를 보여줍니다.



### Training Hybrid Neural Networks with Multimode Optical Nonlinearities Using Digital Twins (https://arxiv.org/abs/2501.07991)
Comments:
          17 pages, 6 figures

- **What's New**: 이 논문에서는 인공지능(AI) 모델의 에너지 효율성과 확장성을 높이기 위해 복잡한 물리적 사건을 고정된 계산 모듈로 네트워크에 통합하는 방법을 제시합니다. 멀티모드 섬유(multimode fibers)에서의 초단파(pulse) 전파를 활용하여 대규모 비선형 변환(nonlinear transformations)을 수행함으로써 이러한 요구를 충족할 수 있습니다. 이를 통해 기계 학습 모델의 복잡성을 줄이고 에너지 소비를 최소화하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 혼합 아키텍처를 훈련시키기 위해, 이 논문에서는 광학 시스템(optical system)을 미분 가능하게 근사화하는 신경 모델(neural model)을 사용합니다. 훈련 알고리즘(training algorithm)은 신경 시뮬레이터(neural simulator)를 업데이트하고, 이 프록시(proxy)를 통해 오류 신호(error signal)를 역전파(backpropagate)하여 광학 계층(optical layer) 이전의 계층을 최적화합니다. 이러한 접근 방식은 복잡한 훈련 과정을 단순화하면서도 높은 정확도를 유지합니다.

- **Performance Highlights**: 실험 결과, 본 연구는 이미지 분류(image classification) 성능 및 시뮬레이션 충실도(simulation fidelity)에서 최첨단(State-of-the-art) 결과를 달성하였습니다. 또한, 이 프레임워크는 실험적 드리프트(experimental drifts)에 대해 뛰어난 저항성을 보여주며, 저에너지 물리 시스템(low-energy physical systems)을 신경망에 통합하여 에너지 효율을 극대화할 수 있는 가능성을 입증합니다.



### GAC-Net_Geometric and attention-based Network for Depth Completion (https://arxiv.org/abs/2501.07988)
Comments:
          13pages,4 figures, 2 tables

- **What's New**: 이 논문은 깊이 완성(depth completion) 네트워크인 CGA-Net을 제안합니다. CGA-Net은 채널 주의 메커니즘(channel attention mechanism)과 3D 글로벌 특징 인식(global feature perception)을 결합하여, 희박한 LiDAR 깊이 측정을 고품질의 밀도 깊이 맵으로 완성하는 혁신적인 접근법을 제공합니다. 이 방법은 특히 복잡한 경계와 희박한 영역에서 기존 방법의 한계를 극복하려는 목표를 가지고 있습니다.

- **Technical Details**: 논문에서 제안하는 CGA-Net은 PointNet++을 활용하여 희박한 깊이 맵에서 글로벌 3D 기하학적 특성을 추출합니다. 이렇게 추출된 특성은 저선 LiDAR 데이터의 장면 인식 능력을 향상시키는데 기여합니다. 또한, 멀티모달 특징 융합 모듈(multimodal feature fusion module)을 통해 희박한 깊이, RGB 이미지 및 3D 기하학적 특성을 효율적으로 통합합니다.

- **Performance Highlights**: KITTI 깊이 완성 데이터셋에 대한 실험 결과, CGA-Net은 밀도 깊이 맵의 예측 정확성을 유의미하게 개선하여 새로운 최첨단(SOTA) 성능을 달성했습니다. 특히, 희박하고 복잡한 장면에 대한 강력한 견고성을 보여주어 실제 자율 주행 환경에서의 활용 가능성을 높이고 있습니다.



### Facial Dynamics in Video: Instruction Tuning for Improved Facial Expression Perception and Contextual Awareness (https://arxiv.org/abs/2501.07978)
- **What's New**: 이번 논문에서는 동적 얼굴 표정 캡션(Dynamic Facial Expression Caption, DFEC)을 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 5,033개의 수작업으로 주석이 달린 고품질 비디오 클립으로 구성되어 있으며, 700,000개 이상의 토큰을 포함하여 비디오 MLLM의 얼굴 표정 인식을 향상시킵니다. 또한, 얼굴 추적 기술을 활용한 새로운 모델 FaceTrack-MM을 제안하여, 다양한 안면 표정을 정확하게 캡션할 수 있는 능력을 강화합니다. 이를 통해 얼굴 표정 이해의 한계를 극복하려는 노력이 이루어집니다.

- **Technical Details**: 논문에서는 새로운 평가 지표인 시간적 이벤트 매칭(Temporal Event Matching, TEM)을 소개하며, 이를 통해 생성된 텍스트의 의미적 일관성과 시간적 순서를 평가합니다. 또한, DFEC 작업을 위해 설계된 FEC-Bench라는 벤치마크를 구축하여, 15개의 오픈소스 및 상용 모델의 성능을 비교할 수 있도록 하였습니다. 기존의 얼굴 행동 단위 탐지(Facial Action Unit Detection, FAUD)와는 달리, 이 연구는 자연어를 사용하여 얼굴 정보를 설명하는 방법을 채택하고 있습니다.

- **Performance Highlights**: FaceTrack-MM 모델은 복잡한 다중 인물 장면에서도 주요 캐릭터의 얼굴 표정을 추적하고 집중하는 데 있어 우수한 성능을 입증하였습니다. 특히 비디오 입력을 처리하는 과정에서 제한된 토큰 수에도 불구하고, 얼굴 지역의 세부 정보를 정확하게 인코딩하여 캡션의 품질을 향상시키는 데 기여합니다. 새로운 데이터셋과 모델을 통해 비디오 MLLM의 인식 능력을 크게 향상시킬 것으로 기대되며, 향후 연구의 기반을 마련합니다.



### Derivation of Output Correlation Inferences for Multi-Output (aka Multi-Task) Gaussian Process (https://arxiv.org/abs/2501.07964)
- **What's New**: 이 논문에서는 Multi-task Gaussian process (MTGP)에 대한 친절한 유도 및 그라디언트(guides) 설명을 제공합니다. 기존의 MTGP의 유도 과정은 이해하기 어려운 부분이 있었으나, 저자들은 그러한 장벽을 제거하려고 노력했습니다. 이를 통해 더 많은 연구자들이 MTGP를 활용하여 하이퍼파라미터 최적화(hyperparameter optimization)와 같은 문제를 해결할 수 있기를 희망하고 있습니다.

- **Technical Details**: MTGP는 Bonilla et al. (2007)에서 처음 제안된 모델로, 이는 여러 개의 입력 출력 조합을 다루는 데 효과적입니다. 이 논문은 MTGP의 수학적 모델링에 있어 가우시안 분포(Gaussian distribution), 공분산 행렬(covariance matrix), 그리고 확률 밀도 함수(probability density function)에 관한 기초적인 수식을 제공합니다. 특히, 하이퍼파라미터(hyperparameters)와 관련된 그래디언트의 유도 과정을 구체적으로 다루고 있습니다.

- **Performance Highlights**: MTGP는 다목적 최적화(multi-objective optimization), 제약 조건 최적화(constrained optimization), 다중 충실도 최적화(multi-fidelity optimization)와 같은 분야에서 효과적임이 입증되었습니다. 그러나 기존 문헌에서는 MTGP의 유도 과정에 대한 세부 사항이 부족하여 연구자들이 이를 활용하는 것이 어려웠습니다. 연구자들이 이 논문의 내용을 통해 MTGP의 이점을 확장하고, BoTorch와 같은 기존의 하이퍼파라미터 최적화 프레임워크를 개선하는 데 기여할 것으로 기대됩니다.



### AI Guide Dog: Egocentric Path Prediction on Smartphon (https://arxiv.org/abs/2501.07957)
- **What's New**: AI Guide Dog (AIGD)는 시각장애인을 위한 경량의 자가 중심 내비게이션 지원 시스템으로, 스마트폰에서 실시간 배치를 위해 설계되었습니다. 이 시스템은 영상 기반의 다중 레이블(classification) 접근 방식을 사용하여 방향 명령을 예측함으로써 안전한 탐색을 보장합니다. 또한 GPS 신호와 고급 방향 지시를 통합하여 목표 기반의 야외 내비게이션을 가능하게 하고, 목적지 없이도 실내 탐색을 지원하는 불확실한 다중 경로 예측 문제를 해결합니다.

- **Technical Details**: AIGD는 스마트폰 카메라에서 수신된 비디오 피드를 사용하여 내비게이션 지침을 예측하는 경량 모델을 채택하였습니다. 시스템은 사용자의 헤딩 방향을 예측하기 위해 각 프레임에 대해 다중 레이블 분류 문제로 모델링합니다. 이를 통해 사용자의 다양한 내비게이션 방향을 예측하고, 구체적인 오디오 명령으로 변환합니다. 실내외 환경 모두에서 목표 지향적 및 탐험적 내비게이션 시나리오를 처리하는 첫 번째 시스템으로 자리잡고 있습니다.

- **Performance Highlights**: AIGD는 장애물 회피를 지원하며, 사용자가 목표가 없는 상황에서도 자유롭게 탐색할 수 있도록 설계되었습니다. 이 시스템은 기존의 시각 장애인 내비게이션 시스템보다 더 견고하고 효율적이며, 최종 목적지에의 도달을 위한 명확한 지침을 제공합니다. 데이터 수집 및 전처리를 통해 모델의 성능을 극대화하며, 사용자들은 명확하고 실행 가능한 오디오 명령을 통해 탐색을 지원받습니다.



### Early prediction of the transferability of bovine embryos from videomicroscopy (https://arxiv.org/abs/2501.07945)
Comments:
          Accepted at the 2024 IEEE International Conference on Image Processing

- **What's New**: 본 논문은 머신러닝(Machine Learning)과 결합된 비디오 현미경(Videomicroscopy) 기술을 활용하여 체외 수정된 소 배아의 초기 발달을 연구하는 새로운 방법을 제시합니다. 특히, 2D 타임랩스(2D time-lapse) 비디오를 입력으로 받아 4일 이내에 배아 이식 가능성을 예측하는 것을 목표로 하고 있습니다. 이를 위해 이식 가능성과 비이식 가능성을 분류하는 이진 분류 문제로 문제를 설정하였습니다.

- **Technical Details**: 저자들은 3D 합성곱 신경망(3D Convolutional Neural Network)을 제안하여 세 가지 경로(pathway)를 포함하는 다중 스케일(multi-scale) 접근 방식을 구현하였습니다. 이 모델은 시각적 특징과 움직임(motion)을 다르게 처리하는 능력을 보유하고 있으며, 훈련 시에는 포컬 로스(focal loss)를 적용하여 성능을 향상시킵니다. 모델의 이름은 SFR이며, 이는 기존 방법들과 비교할 때 우수한 성능을 나타냅니다.

- **Performance Highlights**: 실험을 통해 모델 SFR의 효과성과 정확성이 입증되었습니다. 특히 소 배아의 이식 가능성을 평가하는 도전적인 생물학적 과제에 대해 높은 정확도를 기록하여 향후 연구와 관련된 응용 가능성을 시사합니다. 이 연구는 동물 생식의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Gandalf the Red: Adaptive Security for LLMs (https://arxiv.org/abs/2501.07927)
Comments:
          Niklas Pfister, Václav Volhejn and Manuel Knott contributed equally

- **What's New**: 이 논문은 기존의 프롬프트 공격에 대한 방어 평가에서 두 가지 중요한 요인을 간과하고 있음을 지적합니다. 첫째, 공격자의 동적 행동을 고려해야 하며 둘째, 엄격한 방어책이 정당한 사용자의 사용성에 미치는 부담을 고려해야 합니다. 이를 위해 논문에서는 D-SEC(Dynamic Security Utility Threat Model)을 제안하고 있습니다. D-SEC는 공격자와 정당한 사용자를 분리하고 다단계 상호작용을 모델링하여 보안과 유용성을 최적화하는 방법을 제시합니다.

- **Technical Details**: D-SEC는 공격의 적응성을 포착하고 보안과 유용성을 함께 고려할 수 있도록 설계되었습니다. 이를 통해 LLM 애플리케이션에 대한 방어를 평가하는 보다 합리적이고 포괄적인 방법을 제공하고자 합니다. 또한, 대규모 데이터셋을 생성하기 위해 crowd-sourced 플랫폼인 Gandalf를 구성하여 다양한 공격 시나리오를 수집하여 공개했습니다. 이러한 데이터셋은 279,000개의 프롬프트 공격을 포함하고 있으며, 이는 LLM의 보안성을 평가하는 데 유용한 자원이 될 것입니다.

- **Performance Highlights**: 이 논문은 보안과 유용성 간의 균형을 맞추는 것이 LLM 애플리케이션 설계에 있어서 필수적임을 강조합니다. 특히, 제한된 응용 프로그램 영역, 방어 심화(defense-in-depth), 적응형 방어 등 세 가지 전략이 효과적이라는 것을 보여줍니다. 이러한 전략들은 적절하게 조합되어 사용될 경우 사용성을 저해하지 않으면서도 효과적인 방어를 제공합니다. 이 연구 결과는 LLM 애플리케이션의 보안성과 유용성을 최적화하는 데 중요한 통찰을 제공합니다.



### Optimal Classification Trees for Continuous Feature Data Using Dynamic Programming with Branch-and-Bound (https://arxiv.org/abs/2501.07903)
Comments:
          In the proceedings of AAAI-25

- **What's New**: 이 논문에서는 최적 분류 트리(Optimal Decision Tree, ODT)를 계산하기 위한 새로운 알고리즘인 ConTree를 제안합니다. ConTree는 기존의 동적 프로그래밍(Dynamic Programming, DP)과 가지치기 기법을 결합하여 연속형 데이터에 적합하도록 최적화되었습니다. 이전의 방법들과 비교할 때 ConTree는 숫자 데이터에서 ODT를 효율적으로 계산할 수 있으며, 특히 깊이 4의 트리를 다루는 데 있어 상당한 향상을 보여줍니다. 이 연구 결과는 5% 이상의 테스트 정확도를 달성하고, 실행 시간에서도 현저한 이점을 제공합니다.

- **Technical Details**: ConTree는 branch-and-bound(BnB) 알고리즘과 동적 프로그래밍(DP) 기법을 사용하여 최적의 분류 트리를 직접 계산하도록 설계되었습니다. 새로운 하한(bounding) 기법을 도입하여 유사한 분할을 제거함으로써 검색 공간을 줄임과 동시에 계산 비용을 최소화합니다. 또한, 깊이 2의 트리를 위한 특화된 서브루틴을 제안하여 숫자 데이터를 정렬할 수 있는 특성을 활용합니다. 이 알고리즘적 개선들은 Quant-BnB와 기존의 MIP 및 SAT 접근법에 비해 실행 시간을 크게 단축시키는 데 기여하고 있습니다.

- **Performance Highlights**: ConTree 알고리즘의 실험 결과는 특히 깊이 4의 ODT를 계산하는데 있어 뛰어난 성능을 보여줍니다. 다양한 데이터셋에 대한 실험에서 ConTree는 Quant-BnB 대비 1배 이상의 성능 향상을 달성하였으며, CART 방법에 비해 평균 5% 더 높은 정확도를 기록하였습니다. 이는 동적 프로그래밍과 branch-and-bound 기법의 조합이 연속형 데이터 처리에서 매우 효과적임을 나타냅니다. 이러한 결과들은 ConTree가 제한된 시간 내에 규모가 큰 데이터셋에서도 유용하게 사용될 수 있음을 증명합니다.



### Leveraging Metamemory Mechanisms for Enhanced Data-Free Code Generation in LLMs (https://arxiv.org/abs/2501.07892)
Comments:
          11 pages,6 figures

- **What's New**: 최근 자동화된 코드 생성 기술은 대규모 언어 모델(LLMs)을 사용하여 향상되고 있습니다. 본 연구에서는 LLMs의 코드 생성 성능을 개선하기 위해 인간의 메타기억(자신의 기억 과정을 인식하고 평가하는 인지 과정)에서 영감을 받아 새로운 프레임워크인 M^2WF를 제안합니다. 이 프레임워크는 LLMs가 신뢰성 및 성능을 개선하기 위해 합성 예제를 스스로 생성, 평가 및 활용할 수 있도록 지원합니다.

- **Technical Details**: M^2WF 프레임워크는 네 가지 주요 단계인 Recall, Evaluation, Planning, Guidance로 구성됩니다. Recall 단계에서는 LLM이 관련 프로그래밍 문제를 기억하고, Evaluation 단계에서 각 문제의 신뢰도를 평가합니다. 선택된 상위 M개의 예제는 Planning 단계에서 원래 문제에 대한 구현 계획을 수립하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, M^2WF 프레임워크를 통한 LLM의 코드 생성 품질이 크게 향상되어 pass@1111 점수가 29.43% 이상 증가하는 경우도 있었습니다. 이는 데이터 없는 환경에서도 강력하고 확장 가능한 솔루션을 제공합니다.



### GRAPHMOE: Amplifying Cognitive Depth of Mixture-of-Experts Network via Introducing Self-Rethinking Mechanism (https://arxiv.org/abs/2501.07890)
Comments:
          10 pages

- **What's New**: GRAPHMOE는 Mixture-of-Experts (MoE) 네트워크의 성능을 향상시키기 위한 새로운 방법으로, Pseudo GraphMoE 네트워크를 기반으로 한 자기 재사고(self-rethinking) 메커니즘을 도입합니다. 본 연구에서는 전문가 모델들 간의 협업을 통해 MoE 네트워크의 인지적 깊이(cognitive depth)를 증대시키는 기법을 제안합니다. 이 접근법은 기존 LoRA 기반 모델들을 능가하며 최첨단(state-of-the-art) 성과를 도출합니다.

- **Technical Details**: GRAPHMOE는 고유한 반복 라우팅(recurrent routing) 전략을 활용하여 전문가 노드 간의 정보 흐름을 촉진하고, 입력 데이터를 여러 번의 추론 사이클을 통해 지속적으로 검토 및 정제합니다. 이러한 과정을 통해 SNS 메시지 전송과 유사한 방식으로 전문가 모델의 출력이 후속 전문가 배치 선택을 위한 신호로 전송됩니다. LoRA 기술을 통해 GraphMoE 아키텍처를 구현하였으며, 여러 벤치마크에서 감독된 미세 조정(Supervised Fine-Tuning)으로 평가되었습니다.

- **Performance Highlights**: GRAPHMOE는 표준 LoRA+MoE 프레임워크와 비교하여 학습된 전문가 모델의 특성을 기능적으로 통합하고, 정보 전송을 통해 문제 해결의 효율성을 향상시킵니다. 실험 결과 특히 고전 LoRA+MoE 방식보다 메모리 사용량(메모리 사용량 감소)을 유지하면서도 연산 반복 횟수를 증가시켜 성능을 개선하였습니다. 본 연구는 신경망이 인간과 유사한 반복적 추론 과정을 모방할 수 있도록 하는 '자기 재사고 메커니즘'을 소개하여, LLM의 추론 능력을 높이는 데 중요한 기여를 하고 있습니다.



### Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding (https://arxiv.org/abs/2501.07888)
- **What's New**: Tarsier2는 최신 대형 비전-언어 모델(LVLM)로, 세 가지 주요 업그레이드를 통해 자세하고 정확한 비디오 설명을 생성하는 우수한 성능을 보여줍니다. 첫째, 사전 학습 데이터 양을 1100만에서 4000만 비디오-텍스트 쌍으로 확장해 데이터의 양과 다양성을 증가시켰습니다. 둘째, 감독된 세분화된 시간 정렬(fine-grained temporal alignment) 기법을 도입했고, 셋째, DPO(Delegated Preference Optimization) 훈련을 통한 모델 기반 샘플링 기법을 사용했습니다.

- **Technical Details**: Tarsier2는 비전 인코더(vision encoder), 비전 어댑터(vision adaptor), 대형 언어 모델(LLM)로 구성된 단순한 모델 아키텍처를 채택합니다. 세 단계의 훈련 절차로는 사전 학습(pre-training), 감독된 세분화된 미세 조정(supervised fine-tuning), 강화 학습(reinforcement learning)으로 구성됩니다. 또한, 150K개의 비디오 설명 인스턴스를 포함하는 데이터셋을 구축하여 SFT 단계에서 모델의 시간 세분화 정렬을 위한 감독을 제공합니다.

- **Performance Highlights**: Tarsier2-7B는 비디오 설명 과제에서 기존의 선도적인 상용 모델들, 즉 GPT-4o와 Gemini-1.5-Pro를 능가하며, DREAM-1K 벤치마크에서 F1 점수가 각각 2.8% 및 5.8% 향상되었습니다. 전반적인 성능 평가에 따르면, Tarsier2-7B는 인간 평가에서도 GPT-4o보다 8.6%, Gemini-1.5-Pro보다 24.9% 더 나은 성능을 보였습니다. Tarsier2-7B는 비디오 질문 응답, 비디오 그라운딩 및 환각 테스트와 같은 15개의 공개 벤치마크에서 새로운 SOTA 결과를 기록하며, 다재다능한 일반ist 비전-언어 모델로서의 성능을 입증합니다.



### Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision (https://arxiv.org/abs/2501.07886)
Comments:
          22 pages, 10 figures

- **What's New**: 이번 연구는 신뢰할 수 없는 인간의 감독 하에서도 언어 모델( LM)의 사후 훈련(post-training)이 효과적으로 수행될 수 있는지를 검토합니다. 실험에서는 소규모 모델과 시간 제약이 있는 인간을 이용해 신뢰할 수 없는 시연과 비교 피드백을 시뮬레이션했습니다.

- **Technical Details**: 연구의 핵심은 반복 레이블 정제(iterative label refinement, ILR)입니다. ILR은 비교 피드백을 활용해 인간의 시연을 대체할 모델 생성 대안을 결정한 후, 업데이트된 데이터로 SFT(supervised finetuning)를 통해 모델을 재훈련합니다. 이러한 접근법은 DPO(reinforcement learning from human feedback)보다 더 나은 성능을 발휘합니다.

- **Performance Highlights**: SFT+ILR 방법은 수학, 코딩 및 안전한 지시사항 준수와 같은 여러 작업에서 SFT+DPO보다 우수한 결과를 나타냈습니다. 연구 결과는 LM이 복잡한 작업을 수행하는 데 있어, RLHF가 항상 최선의 방법이 아닐 수 있음을 시사하며, 오히려 훈련 데이터를 개선하는 방향으로 피드백을 사용하는 것이 더 효과적일 수 있음을 보여줍니다.



### Continual Learning with Embedding Layer Surgery and Task-wise Beam Search using Whisper (https://arxiv.org/abs/2501.07875)
Comments:
          Published in 2024 IEEE Spoken Language Technology Workshop

- **What's New**: 이번 논문에서는 Multilingual ASR 모델의 한계를 극복하기 위해 새로운 언어를 추가하는 Continual Learning (CL) 방법을 제안합니다. 기존의 CL 방법들이 Catastrophic Forgetting (CF) 문제를 간과하고 있는 반면, 우리는 Embedding Layer Surgery라는 새로운 기법을 도입하여 새로운 언어에 대한 복사본을 생성하는 방식을 보여줍니다.

- **Technical Details**: 우리는 새로운 언어별로 별도의 토큰 임베딩(token embedding) 복사본을 생성하고, 해당 언어의 전사(transcribing) 과정에서 가장 적절한 복사본을 선택하여 기존 언어의 임베딩을 대체합니다. 하지만, 언어 유형 식별(Language Identification, LID) 오류로 인해 잘못된 임베딩 선택이 발생할 수 있으며, 이를 해결하기 위해 Task-wise Beam Search를 적용하여 실수를 스스로 수정할 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법을 통해 Common Voice의 10개 보지 못한 언어에 대해 10시간의 데이터를 조정한 결과, 사전 훈련된 언어의 Average WER (AWER)가 14.2%에서 11.9%로 감소하였습니다. 이 과정에서 새로 추가된 언어의 WER은 손상되지 않았으며, 성능이 유지되었습니다.



### deepTerra -- AI Land Classification Made Easy (https://arxiv.org/abs/2501.07859)
- **What's New**: deepTerra는 기계 학습과 위성 이미지를 활용해 지표 분류를 쉽게 할 수 있도록 설계된 포괄적인 플랫폼입니다. 데이터 수집, 이미지 증강, 훈련, 테스트 및 예측 모듈을 포함하여 이미지 분류 작업의 전체 워크플로우를 간소화합니다. 이 논문에서는 deepTerra의 기능과 다양한 연구 분야에서의 적용 사례를 보여주며, 향후 발전 방향도 논의합니다.

- **Technical Details**: deepTerra는 기계 학습을 통해 이미지를 분류하는 다양한 단계를 지원하는 도구 모음입니다. 데이터 수집 모듈은 기존 이미지를 가져와 이미지 패치를 추출하고, 자동으로 지리적 좌표를 기록하여 공간적 참조를 제공합니다. 이미지 증강은 회전, 플리핑, 이동 등의 기하학적 변환을 통해 데이터셋을 확장하여 모델의 강건성과 성능을 개선합니다.

- **Performance Highlights**: deepTerra는 VGG, ResNet, Inception 등 다양한 CNN 아키텍처를 지원하여 사용자들이 다양한 이미지 분류 작업을 수행할 수 있도록 합니다. 훈련 과정에서 정확도와 F1 스코어 등의 성과 지표를 시각화하고, 훈련 완료 후에는 혼동 행렬 및 키 성능 메트릭을 포함한 상세한 결과를 제공합니다. 이 툴은 새로운 라벨 없는 데이터에 대한 예측도 지원하여, 최종 목표인 이미지 분류를 효과적으로 수행합니다.



### Hierarchical Repository-Level Code Summarization for Business Applications Using Local LLMs (https://arxiv.org/abs/2501.07857)
Comments:
          To appear at LLM4Code@ICSE 2025

- **What's New**: 이 논문은 대규모 소프트웨어 개발에서의 코드 요약 기술을 발전시키기 위해 비즈니스 어플리케이션에 맞춰진 두 단계 계층적 접근 방식을 제안합니다. 기존 연구들은 보통 작은 코드 단위에 집중해 왔으나, 저자는 코드베이스의 구조와 의도를 포괄적으로 이해할 수 있는 새로운 방법을 제안합니다. 특히, 함수와 변수를 먼저 요약하고, 이를 기반으로 파일 및 패키지의 요약을 생성함으로써 비즈니스 컨텍스트를 강화합니다.

- **Technical Details**: 제안된 방법에는 두 단계 계층적 접근이 포함되며, 첫 번째 단계에서는 Java 파서를 사용하여 코드의 구문 분석을 통해 작은 코드 단위를 식별합니다. 이후, 이러한 단위에 대해 LLM(Local Large Model)을 이용해 요약을 하여 더 높은 수준의 파일 및 패키지 요약을 생성합니다. 코드 아티팩트의 의도를 바탕으로 커스터마이즈된 프롬프트를 설계함으로써 비즈니스 컨텍스트를 강조합니다.

- **Performance Highlights**: 제안된 접근 방식을 사용하여 통신 분야의 비즈니스 지원 시스템(BSS)을 대상으로 평가를 진행했으며, 그 결과 계층적 요약 방식이 범위(coverage)를 개선하고, 비즈니스 컨텍스트의 기반이 요약의 관련성을 높이는 데 기여함을 보였습니다. 따라서 기존의 낮은 수준의 구현 세부정보에 집중한 모델과는 달리, 이 연구는 비즈니스 문제에 대한 코드의 의도를 보다 효과적으로 포착하는데 기여합니다.



### State-of-the-Art Transformer Models for Image Super-Resolution: Techniques, Challenges, and Applications (https://arxiv.org/abs/2501.07855)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 이미지 초해상도(Image Super-Resolution, SR) 분야에서 transformer 기반 방법의 발전을 소개하고 있습니다. 기존의 CNN 및 GAN과 같은 딥러닝 방식보다 더욱 높은 품질의 이미지를 재구성할 수 있는 가능성을 제시하고 있습니다. 특히, 이전 방법들의 한계인 제한된 수신 필드(receptive fields)와 고주파 세부사항 복구의 어려움을 해결하는 데 기여하고 있습니다.

- **Technical Details**: 이 논문에서는 전통적인 네트워크와 결합된 transformer 기반 SR 모델의 다양한 혁신적인 기법과 아키텍처를 탐구합니다. 이러한 최신 방법들은 글로벌(전역) 및 로컬(지역) 맥락을 균형 있게 고려하여 이미지 품질을 향상시키는 데 중점을 두고 있습니다. 또한, 다양한 시각화 방법을 통해 모델과 기술에 대한 종합적인 이해를 돕고 있습니다.

- **Performance Highlights**: 이 연구는 transformer가 초해상도 기법에 미치는 영향을 탐구하며, 향후 연구를 위한 구조적인 로드맵을 제시하고 있습니다. 비판적으로 분석된 최신 기법들은 유망하지만 아직 탐구되지 않은 격차와 잠재적인 연구 방향을 드러냅니다.



### Optimizing Language Models for Grammatical Acceptability: A Comparative Study of Fine-Tuning Techniques (https://arxiv.org/abs/2501.07853)
- **What's New**: 이번 연구는 CoLA 데이터셋을 사용하여 Open Pre-trained Transformer (OPT-125M)의 세밀 조정(fine-tuning, FT)을 탐구합니다. Vanilla-Fine-Tuning (VFT), Pattern-Based-Fine-Tuning (PBFT), Low-Rank Adaptation (LoRA)와 같은 매개변수 효율적인 세밀 조정(PEFT) 기술을 비교하여 계산 효율성을 크게 개선하면서도 높은 정확도를 유지하는 방법을 보여줍니다.

- **Technical Details**: 실험 결과, VFT는 81.2%의 가장 높은 정확도를 달성했지만, LoRA를 통한 FT는 메모리 사용량과 반복 시간을 50% 이상 줄이며 PBFT의 정확도를 향상시켰습니다. Context Distillation (CD)는 계산 효율성이 높음에도 불구하고 약 31%의 정확도로 떨어지는 결과를 보였습니다.

- **Performance Highlights**: 이 연구는 대규모 언어 모델(Large Language Models, LLM)에 대한 접근을 민주화하는 데 기여하고 있으며, 계산 장벽을 줄이는 방향으로 이바지하고 있습니다. 실험에서 보여진 기술들은 다양한 NLP 작업에서의 효율성을 향상시킬 수 있는 가능성을 제시합니다.



### Unveiling Provider Bias in Large Language Models for Code Generation (https://arxiv.org/abs/2501.07849)
Comments:
          21 pages, 15 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 코드 생성 과정에서 나타나는 공급자 편향(provider bias)을 처음으로 종합적으로 실증 연구한 것입니다. 연구 결과, 명시적인 입력 프롬프트 없이도 이들 모델이 특정 공급자의 서비스를 선호하는 경향을 보인다는 점을 밝혀냈습니다. 이는 시장 역학과 사회적 균형에 중대한 영향을 미칠 수 있으며, 디지털 독점(digital monopolies)을 촉진하고 사용자에게 잘못된 인식을 줄 수 있습니다.

- **Technical Details**: 우리는 6개의 서로 다른 코딩 작업 범주와 30개의 실제 애플리케이션 시나리오를 포함한 데이터 세트를 생성하기 위한 자동화 파이프라인을 개발했습니다. 7개의 최첨단 모델에서 600,000개 이상의 LLM 생성 응답을 분석했으며, 약 5억 개의 토큰(약 5,000달러 이상의 계산 비용)을 사용했습니다. 이 연구는 생성된 코드 조각과 그에 내재된 서비스 공급자 선택을 평가하여 공급자 편향을 정량화했습니다.

- **Performance Highlights**: 연구 결과, LLM들이 주로 Google과 Amazon의 서비스를 선호하는 경향이 있으며, 사용자 요청 없이도 입력 코드를 수정하여 선호하는 공급업체를 통합하는 것으로 나타났습니다. 특히, 대화 맥락에서 권장되는 공급자와 생성된 코드에서 구현된 공급자 간의 차이가 관찰되었습니다. 연구의 전체 데이터 세트와 분석 결과는 공개 리포지토리에서 확인할 수 있습니다.



### Social Media Data Mining With Natural Language Processing on Public Dream Contents (https://arxiv.org/abs/2501.07839)
Comments:
          16 pages, 6 figures

- **What's New**: 이 연구는 COVID-19 팬데믹이 전 세계적인 생활 방식을 어떻게 변화시켰는지를 분석하고 있습니다. 특히, Reddit r/Dreams 커뮤니티에서 공유된 꿈 내용을 통해 팬데믹의 정신적 건강에 미친 영향을 조사합니다. 기존에 접근하기 힘들었던 서브콘셔스(반조직적) 반응을 드러낼 수 있는 새로운 데이터 소스를 활용하고 있습니다.

- **Technical Details**: 연구에서는 통계적 방법을 사용하여 팬데믹 이전(pre-pandemic)과 이후(post-pandemic) 꿈의 긍정적, 부정적, 중립적 변화 추세를 평가하고 있습니다. 또한 LLaMA 3.1-8B 모델을 레이블링된 데이터로 파인튜닝하여 꿈의 내용을 정확하게 감정 분류(sentiment classification)할 수 있게 했습니다.

- **Performance Highlights**: 결과적으로, 연구는 꿈 내용을 통해 팬데믹의 심리적 영향을 드러내고 있으며, 이러한 꿈의 변화가 공공의 웰빙(well-being)의 지표로 작용할 수 있음을 제안합니다. 이 연구는 정신적 풍경에서의 깊은 변화와 꿈의 역할을 부각시키며, 전례 없는 시기에 개인의 심리적 상태를 이해하는 데 기여하고 있습니다.



### Real-time Verification and Refinement of Language Model Text Generation (https://arxiv.org/abs/2501.07824)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 Streaming-VR(Streaming Verification and Refinement)라는 새로운 방식을 제안합니다. 이 방법은 LLM이 생성하는 토큰을 실시간으로 검증하고 수정하여 전체 생성 이후가 아닌 지속적인 검사를 통해 오류 전파를 방지합니다. 따라서 LLM의 응답 생성 중에 정확성을 높이고 비효율성을 줄이는 것을 목표로 합니다.

- **Technical Details**: Streaming-VR은 외부 검증 모델을 활용하여 LLM이 생성하는 각 토큰을 실시간으로 확인하고, 잘못된 부분은 즉시 수정하는 방식으로 작동합니다. 이 프로세스는 기존의 후속 수정 방식보다 더 빠르고 효율적이며, 결과적으로는 사실적 정확성을 개선합니다. 실험 결과, Streaming-VR을 통해 LLM 품질이 대폭 향상된다는 것을 입증했습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 Streaming-VR의 효과를 검증한 결과, ASQA와 QuoteSum 데이터셋에서 평균 각각 39.8% 및 31.5% 더 높은 효율성을 달성했습니다. 또한, 조기 생성된 잘못된 토큰은 후속 문장이 사실적으로 부정확할 확률을 약 37.6% 증가시켜 Streaming-VR의 필요성을 강조합니다.



### A Multi-Encoder Frozen-Decoder Approach for Fine-Tuning Large Language Models (https://arxiv.org/abs/2501.07818)
- **What's New**: 본 연구에서는 멀티태스킹(multi-task) 환경에서 디코더를 고정(freezing)하는 것이 자연어 처리(NLP) 작업의 성능을 어떻게 개선하는지를 조사합니다. 특히, 다양하고 복잡한 자연어 작업에 대한 성능 향상과 배포(overhead) 비용 절감을 목표로 합니다. 실험 결과, 디코더를 고정하는 방식은 다언어(multilingual) 작업에서의 재앙적 망각(catasrophic forgetting)을 줄여주는 것으로 나타났습니다. 또한, 큰 모델과 결합하면 구조화된 작업 및 질문 응답(QA) 작업에서 성능을 유지하거나 향상시킬 수 있습니다.

- **Technical Details**: 연구에 사용된 모델은 AlexaTM로, 12개의 인코더와 12개의 디코더로 구성된 5억 1100만 개의 파라미터를 가진 모델입니다. 실험은 여러 NLP 데이터셋을 통해 진행되었으며, 학습 스케줄 및 최적화는 Adam Optimizer를 사용하여 진행되었습니다. 특히, 디코더를 고정하였다 하더라도 모델의 크기를 늘림으로써 발생할 수 있는 성능 저하를 완화하는 방법도 모색되었습니다. 이러한 접근 방식은 인코더와 디코더의 별개 학습을 통해 성능을 극대화하는 데 기여합니다.

- **Performance Highlights**: 결과적으로 디코더를 고정할 경우 XSUM과 WebNLG 작업에서 성능이 약 2% 감소했지만, 더 큰 고정 디코더를 사용할 경우 성능이 유지되었습니다. ROUGE, BLEU, NIST와 같은 다양한 평가 지표를 통해 측정된 이러한 성능은 자연어 생성(NLG) 작업에서 디코더의 역할이 중요하다는 것을 보여줍니다. 따라서, 고정된 디코더를 활용하는 접근 방식은 다양한 자연어 처리 작업에서 효과적인 전략으로 자리잡을 수 있을 것으로 보입니다.



### STTS-EAD: Improving Spatio-Temporal Learning Based Time Series Prediction via (https://arxiv.org/abs/2501.07814)
Comments:
          11 pages

- **What's New**: 본 논문은 STTS-EAD라는 새로운 기법을 제안하여 다변량 시계열 예측의 학습 과정에 이상 탐지를 통합합니다. 이 접근 방식은 기존의 이상 탐지 방법들이 철저히 스페티오템포랄(spatio-temporal) 정보를 활용하지 못한 한계를 극복하고, 이를 통해 모델 학습 시 예측 변동성을 향상시키고자 합니다. 특히, STTS-EAD는 훈련 과정에서 이상 상태를 즉각적으로 처리하여, 정확도를 높이고, 다양한 실제 데이터 세트를 적용한 결과에서 타 모델보다 우수한 성능을 입증하였습니다.

- **Technical Details**: STTS-EAD는 스페티오템포랄(spatio-temporal) 정보를 활용하여 각 시계열에 대한 시간 및 공간 임베딩(embedding)을 구축하고, 이를 토대로 예측 및 이상 탐지 과정에서 피드백을 통해 상호 최적화합니다. 즉, 모델 훈련 중 EAD 모듈이 작동해 이상 점수를 계산하고, 스페티오템포랄 학습 기반 시계열 예측 모델의 잔차 정보를 바탕으로 이상 상태를 탐지합니다. 또한, STTS 모델은 다이나믹한 시계열 수에 쉽게 적응할 수 있어, 실제 비즈니스 환경에서 유연하게 활용될 수 있습니다.

- **Performance Highlights**: STTS-EAD는 공개된 주식 데이터 세트와 다국적 커피 체인의 두 가지 실제 판매 데이터 세트에서 실험을 수행하여, 예측 성능이 크게 향상됨을 보여주었습니다. 이러한 실험을 통해 STTS-EAD는 동일한 환경에서 여러 기준 모델을 초과하는 성능을 발휘하였으며, 훈련 단계에서 탐지된 이상 상태를 효과적으로 처리하여 추론 성능을 향상시키는 것으로 확인되었습니다. 이를 통해 다변량 시계열 예측의 신뢰성을 높이고자 하는 연구에 기여하고 있습니다.



### Talk to Right Specialists: Routing and Planning in Multi-agent System for Question Answering (https://arxiv.org/abs/2501.07813)
Comments:
          Work In Progress

- **What's New**: 본 논문은 RopMura라는 혁신적인 다중 에이전트 시스템을 제안하여 단일 도메인 정보의 한계로 인해 발생하는 허위 응답 및 부정확한 답변 문제를 해결합니다. RopMura는 지식 경계를 기반으로 적절한 에이전트를 선택하는 라우터와 복잡한 다중 홉 질문을 분해하는 플래너라는 두 가지 주요 구성 요소로 구성되어 있습니다. 이러한 접근 방식은 정보의 효과적인 라우팅과 계획을 가능하게 하여 다양한 도메인에서의 응답을 효율적으로 조정할 수 있게 합니다.

- **Technical Details**: RopMura는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 외부 지식의 통합과 응답의 신뢰성을 높일 수 있는 시스템입니다. 이 시스템은 각 RAG 에이전트의 지식 조각을 군집화하고 평균 임베딩 벡터를 계산하여 에이전트 간의 관련성 점수를 산출합니다. 이를 통해 복잡한 질문이 제기될 때 플래너는 각 단계마다 적절한 질문을 생성하고, 라우터는 가장 관련성 높은 에이전트를 선택하여 응답을 생성합니다.

- **Performance Highlights**: 실험 결과, RopMura는 단일 홉 질문과 다중 홉 질문 모두를 효과적으로 처리할 수 있음을 보여주었습니다. 단일 홉 질문의 경우, 라우팅 메커니즘만으로도 정확한 응답을 가능하게 하였고, 다중 홉 질문에서는 라우팅과 계획 메커니즘의 결합을 통해 일관성과 정확성을 갖춘 다단계 해결책을 제공하였습니다. 이를 통해 RopMura는 다양한 도메인 간의 질문-응답 작업에서 우수한 성능을 인정받았습니다.



### Conformal mapping Coordinates Physics-Informed Neural Networks (CoCo-PINNs): learning neural networks for designing neutral inclusions (https://arxiv.org/abs/2501.07809)
- **What's New**: 이번 연구는 물리적으로 정보가 포함된 신경망(Physics-Informed Neural Networks, PINNs)을 활용하여 중성 포괄체(neutral inclusion) 문제를 해결하는 새로운 접근 방식을 제안합니다. 새로운 방법인 CoCo-PINNs는 복소 해석 기법을 PINNs에 통합하여 불완전 경계 조건이 있는 임의 모양의 포괄체 설계를 위한 역문제를 효과적으로 처리합니다. 이 연구는 표준 PINNs의 성능 한계를 극복하고, 중성 포괄체 설계에 필수적인 문제를 해결하는 혁신적인 기법을 제공합니다.

- **Technical Details**: CoCo-PINNs에서는 복소 해석 기법을 활용하여 경계 포인터 외부에서 샘플 포인트에서의 해결값을 손실 함수 설계에 포함시킵니다. 기존 PINNs는 경계 매개변수를 신경망으로 근사하는 방식으로 취급했으나, 이 연구에서는 불완전 매개변수의 푸리에 급수 계수를 훈련시키는 새로운 접근법을 채택합니다. 또한, 이 방법은 경계 조건이 불완전한 포괄체에 적용하여 다양한 모양의 포괄체에 대한 전방 및 역 문제를 동시에 해결할 수 있습니다.

- **Performance Highlights**: CoCo-PINNs는 신뢰성(reliability), 일관성(consistency), 안정성(stability) 면에서 향상된 성능을 보이며, 기존 PINNs에 비해 더 정확한 전방 해결책 및 향상된 역 매개변수 식별을 제공합니다. 이 방법은 비선형 문제를 해결하는 데 필요한 높은 복잡성을 수용하면서도 보다 정확한 솔루션을 생성할 수 있는 가능성을 보여줍니다. 수치적 방법과의 비교를 통해 CoCo-PINNs의 뛰어난 성능이 입증되었습니다.



### A Comparative Analysis of DNN-based White-Box Explainable AI Methods in Network Security (https://arxiv.org/abs/2501.07801)
- **What's New**: 이 연구는 인공지능(AI) 솔루션을 네트워크 침입 탐지 시스템(NIDS)에 적용하는 데 집중하고 있습니다. 특히, 이 연구에서는 설명 가능한 인공지능(XAI) 기법을 사용하여 보안 분석가가 블랙박스 AI 모델의 결정을 이해하고 설명할 수 있도록 합니다. 연구에서는 LRP(Layer-wise Relevance Propagation), IG(Integrated Gradients), DeepLift와 같은 화이트박스 XAI 기법을 활용하여 NIDS를 평가하고, 특정 네트워크 침입 데이터셋을 사용하여 그 결과를 비교합니다.

- **Technical Details**: 이 논문은 화이트박스 XAI 기법의 성능을 평가하기 위해 다양한 지표를 사용합니다. 주요 지표로는 설명의 정확성(descriptive accuracy), 희소성(sparsity), 안정성(stability), 견고성(robustness), 효율성(efficiency), 완전성(completeness)이 포함됩니다. 연구는 NSL-KDD, CICIDS-2017, RoEduNet-SIMARGL2021 데이터셋을 사용하여 각 기법의 성능을 비교하고, 이들이 실제 환경에서의 적용 가능성을 분석합니다.

- **Performance Highlights**: 연구 결과, 화이트박스 XAI 기법이 IDS에서 견고성과 완전성 면에서 높은 점수를 기록했습니다. 이 두 가지 지표는 침입 탐지 시스템에서 특히 중요한 평가 기준입니다. 또한, 이 연구에 사용된 코드가 연구 커뮤니티에 공개되어 다른 연구자들이 이를 개선하고 활용할 수 있도록 하여 XAI 기법의 발전에 이바지하고 있습니다.



### BioPose: Biomechanically-accurate 3D Pose Estimation from Monocular Videos (https://arxiv.org/abs/2501.07800)
- **What's New**: BioPose는 단안 영상을 통해 생체역학적으로 정확한 3D 인간 포즈를 예측하는 새로운 학습 기반 프레임워크입니다. 이 프레임워크는 다중 쿼리 인간 메쉬 복구 모델(MQ-HMR), 신경 역운동학 모델(NeurIK), 2D 정보 기반 포즈 보정 기법의 세 가지 핵심 요소로 구성되어 있습니다. BioPose는 기존의 마커 기반 시스템과 비교했을 때 접근성과 편리함을 유지하면서도 비슷한 성능을 목표로 합니다.

- **Technical Details**: MQ-HMR 모델은 다중 쿼리 변형가능한 트랜스포머를 사용하여 단안 비디오 프레임에서 다중 스케일의 세부 이미지를 추출합니다. 이 추출된 메쉬는 NeurIK 모델에 의해 가상의 마커로 취급되어 생체역학적으로 정확한 3D 포즈를 추론하는 데 사용됩니다. 마지막으로 2D 정보 기반 보정을 통해 3D 예측을 2D 관찰과 일치시켜 시각적 일관성과 생체역학적 유효성을 향상시킵니다.

- **Performance Highlights**: BioPose는 기준 데이터 세트에서 상태-of-the-art 방법들을 상당히 초월하는 성능을 보여주었습니다. MQ-HMR 모델은 단일 이미지에서 인간 메쉬 복구 작업에서 최첨단 결과를 생성하며, BioPose 시스템은 전통적인 다중 카메라 마커 기반 기술과 비교했을 때 경쟁력 있는 성능을 제공합니다. 이러한 성과는 생체역학적 정확성을 보장하는 매력적인 솔루션을 제공합니다.



### Transforming Indoor Localization: Advanced Transformer Architecture for NLOS Dominated Wireless Environments with Distributed Sensors (https://arxiv.org/abs/2501.07774)
Comments:
          The paper has been submitted to IEEE Transactions on Machine Learning in Communications and Networking

- **What's New**: 본 연구에서는 실내 위치추적 분야에 있어 새로운 접근 방식을 제안합니다. 특히 Sensor Snapshot Tokenization (SST)이라는 새로운 토큰화 방법을 도입하여 전통적인 방법들보다 향상된 정확도와 효율성을 제공합니다. 이 방법은 다변량 상관관계를 효과적으로 캡쳐하여 Transformer 모델의 주의를 향상시킵니다. 또한, 경량화된 Swish-Gated Linear Unit 기반 Transformer (L-SwiGLU Transformer) 모델을 통해 계산 복잡성을 줄이며, 자원 제약이 있는 환경에서도 높은 정확도를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 다중 안테나 시스템에서의 복잡한 채널 동적을 다루기 위해 Transformer 아키텍처를 규명합니다. 기존의 정규화 레이어를 변경하고 피드포워드 네트워크의 디자인을 재구성하며, 최종 인코더 블록에서의 토큰을 사용하여 예측 메커니즘을 재정의함으로써 L-SwiGLU Transformer 모델을 개선합니다. 또한, Global Average Pooling 레이어를 도입하여 위치 임베딩을 생략하더라도 위치 정확도에 악영향을 미치지 않는다는 사실을 입증하였습니다. 이 모든 수정은 계산 효율성을 높이고 자원 제약이 있는 시나리오에서의 성능을 향상시키기 위해 최적화되었습니다.

- **Performance Highlights**: 제안된 SST 방법으로 개선된 Vanilla Transformer 모델은 NLOS 환경에서 0.388m의 90번째 퍼센타일 위치 오차를 달성하였고, L-SwiGLU ViT 모델은 이를 0.355m로 줄이며 8.51%의 개선을 보였습니다. 또한, 이 새로운 모델은 14.1배 더 큰 기존 모델보다 46.13% 향상된 성능을 보여, 계산 효율성을 잘 입증하였습니다. 이러한 결과는 자원 제한이 있는 환경에서도 Transformer 모델의 효과성을 강조합니다.



### Large Language Models for Knowledge Graph Embedding Techniques, Methods, and Challenges: A Survey (https://arxiv.org/abs/2501.07766)
- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 지식 그래프 임베딩(Knowledge Graph Embedding, KGE) 관련 작업에 적용하는 다양한 방법을 탐구합니다. 특히 KGE 시나리오의 특성에 따라 다중 모달 KGE(multi-modal KGE)와 오픈 KGE(open KGE)와 같은 다양한 유형의 KGE에서 LLM의 활용도를 조명합니다. 이를 통해 LLM을 KGE 작업에 통합하기 위한 새로운 접근 방식과 분류를 제시합니다.

- **Technical Details**: 저자는 LLM이 방대한 텍스트 데이터를 학습하여 다음 단어를 예측하거나 주어진 텍스트와 관련된 콘텐츠를 생성하는 자연어 처리(Natural Language Processing, NLP) 딥러닝 모델로 작용한다고 설명합니다. 또한, 각 KGE 시나리오에 대해 분류하고 이를 비교할 수 있는 방법론과 각각의 소스 코드 링크를 포함한 표 형태의 개요를 제공합니다. 이러한 접근법은 LLM 관련 작업을 보다 일관되게 수행하기 위한 기초를 마련합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 다양한 KGE 시나리오에서 LLM의 적용 가능성을 검토하고, 향후 연구 방향성을 제안합니다. 연구에서 제시된 분류 및 방법론은 LLM과 KGE의 경계가 모호해지는 상황에서 유용한 참고 자료가 될 것입니다. LLM과 KGE의 융합은 향후 지식 그래프의 성능을 더욱 향상시키는 데 기여할 것으로 기대됩니다.



### Deep Learning for Disease Outbreak Prediction: A Robust Early Warning Signal for Transcritical Bifurcations (https://arxiv.org/abs/2501.07764)
Comments:
          14 pages, 1 figure, 5 tables

- **What's New**: 본 연구에서는 Early Warning Signals (EWSs)를 사용하여 질병 발병 예측을 위한 강력한 모델을 개발하였습니다. 딥 러닝 모델을 활용하여 Time Series Classification (TSC) 작업을 통해 질병의 동적 시스템을 분석하고, 노이즈가 포함된 데이터에서도 깊이 있는 예측을 가능하게 하는 방법을 제시합니다. 특히, 독감과 COVID-19와 같은 실제 데이터에 대한 모델의 성과를 검토하며, 기존 모델보다 뛰어난 예측력을 보여줍니다.

- **Technical Details**: 우리는 두 가지 시뮬레이션 데이터셋을 사용하여 모델을 훈련했습니다. 첫 번째 데이터셋은 다이나믹 시스템을 모사하며, 두 번째 데이터셋은 노이즈에 의해 영향을 받는 전염병 동태를 나타냅니다. 모델 아키텍처는 CNN-LSTM을 기반으로 하며, 트랜스크리티컬 분기예측을 통해 다양한 가변 길이의 시계열 데이터를 효율적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 모델은 다양한 시나리오에 대한 발병 경고를 효과적으로 제공하여 기존 모델보다 우수한 성과를 나타냈습니다. 실제 데이터와 시뮬레이션 데이터에서 검증된 결과, 딥 러닝 기술의 발전이 불확실한 환경에서도 신뢰할 수 있는 EWS를 제공하는 데 기여할 수 있음을 보여줍니다. 이러한 접근 방식은 신흥 전염병 위기의 실질적인 해결책으로 자리잡을 가능성이 높습니다.



### On the Statistical Capacity of Deep Generative Models (https://arxiv.org/abs/2501.07763)
- **What's New**: 이 논문은 깊은 생성 모델(deep generative models)의 통계적 특성을 이해하는 데 새로운 통찰을 제공합니다. 방대하고 복잡한 데이터 분포로부터 샘플을 생성할 수 있다는 일반적인 가정을 반박하고, 이러한 모델들이 실제로는 보편적 생성기(universal generators)가 아님을 입증합니다. 또한 모델들이 생성할 수 있는 샘플의 특성이 한정적이며, 특히 무거운 꼬리(heavy tails)가 있는 분포를 잘 처리할 수 없다는 점을 강조합니다.

- **Technical Details**: 논문은 주로 Gaussian 잠재 변수(latent variables)에 대해 논의하며, 이러한 조건 하에서 깊은 생성 모델들이 생성하는 샘플이 경량 꼬리(light tail)를 갖는다는 결과를 제시합니다. 이러한 결과는 측정의 농도(concentration of measure)와 볼록 기하학(convex geometry) 도구를 사용하여 보다 일반적인 로그-오목(log-concave) 및 강한 로그-오목(strongly log-concave) 분포에 대해서도 확장됩니다. 또한 Gromov-Levy 불평등을 적용하여 잠재 변수가 양의 Ricci 곡률(curvature)을 가진 다양체(manifolds) 위에 있을 때 유사한 보장을 설명합니다.

- **Performance Highlights**: 이 연구는 깊은 생성 모델들이 샘플의 불확실성과 다양성을 과소평가하는 경향이 있음을 보여주며, 이는 이상 탐지(anomaly detection) 및 금융(finance)과 같은 응용 분야에서 중요한 의미를 가집니다. 실제 사례를 통해 설계된 모델들이 어떻게 성능에 영향을 미치는지를 드러내며, 특히 무거운 꼬리를 가진 분포를 잘 처리하지 못하는 한계를 강조합니다. 이러한 발견은 특히 베이지안(Bayesian) 문헌에서의 후방 샘플링(posterior sampling)과 연관된 사용에 중요한 의미를 가집니다.



### PSReg: Prior-guided Sparse Mixture of Experts for Point Cloud Registration (https://arxiv.org/abs/2501.07762)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 포인트 클라우드 등록(Point Cloud Registration)의 성능을 향상시키기 위해 Prior-guided Sparse Mixture of Experts (PSMoE) 모듈을 제안합니다. 이 방법은 중첩된 영역에서의 클러스터링 및 지도를 활용하여 보다 명확한 특징을 추출할 수 있도록 돕습니다. 또한, Transformer와 PSMoE 기능을 결합하여 최적의 변환 매트릭스를 추정하는 새로운 등록 프레임워크인 PSReg를 소개합니다.

- **Technical Details**: PSMoE 모듈은 사전 중첩 정보 및 잠재적 대응 임베딩을 융합하여 분산하는 방식으로, 가능한 한 많은 대응 토큰을 동일한 전문가에게 배정하고 처리합니다. 이를 통해 중첩된 영역에서 더 정확한 일치를 찾고 포인트 클라우드의 오버랩 영역을 효과적으로 찾아내는 것이 가능합니다. 새로운 PCE(전문가 지향 상위 포인트 대응 인코딩) 모듈은 매치된 슈퍼포인트를 명확하게 인코딩하여 상관관계를 더욱 높입니다.

- **Performance Highlights**: 해당 방법은 3DMatch 및 3DLoMatch 벤치마크에서 각각 95.7% 및 79.3%의 최첨단 등록 회수율을 달성하며, ModelNet40에서도 우수한 성능을 기록했습니다. 광범위한 실험을 통해 PSReg의 우위성을 입증하였으며, 다양한 모듈의 효과성을 검증하는 여러 단일 실험도 수행했습니다.



### Impatient Bandits: Optimizing for the Long-Term Without Delay (https://arxiv.org/abs/2501.07761)
- **What's New**: 이번 논문에서는 사용자 만족도를 장기적으로 향상시키기 위해 콘텐츠 탐색 문제를 다루었습니다. 이 연구는 bandit 문제를 활용하여 지연 보상 (delayed rewards)의 예측 모델을 개발하고, 새로운 알고리즘을 통해 장기 성공과 연결된 콘텐츠를 빠르게 식별하는 방법을 제안합니다. 또한, Spotify에서의 실제 문제를 해소하기 위한 연구로, 기존의 단기 보상에 최적화된 방법보다 더 효과적임을 입증했습니다.

- **Technical Details**: 이 논문은 지연 보상과 점진적 피드백 (progressive feedback)을 통합한 bandit 알고리즘을 제안합니다. 알고리즘은 두 가지 주요 구성 요소, 즉 Thompson sampling과 Bayesian 필터를 포함하고 있습니다. Bayesian 필터를 통해 점진적으로 드러나는 사용자 참여 데이터를 활용하여 보상의 평균을 추정하고, 이를 통해 효과적인 결정-making을 지원할 수 있습니다.

- **Performance Highlights**: A/B 테스트 결과, 점진적 피드백을 포함한 추천 알고리즘이 단기 보상이나 지연 보상에만 의존하는 방법보다 실질적으로 월등한 성과를 보였습니다. 실험에서는 Spotify의 팟캐스트 추천 데이터가 사용되었으며, 결과적으로 수억 명의 사용자에게 맞춤화된 오디오 추천을 제공하는 시스템의 성능을 크게 향상시켰습니다. 이렇게 함으로써, 산업 수준의 추천 시스템에서 점진적 피드백의 중요성을 강조하였습니다.



### Performance Optimization of Ratings-Based Reinforcement Learning (https://arxiv.org/abs/2501.07755)
Comments:
          Accepted to the Collaborative AI and Modeling of Humans Bridge Program at AAAI 2025

- **What's New**: 이 논문은 Rating-based Reinforcement Learning (RbRL)의 성능을 향상시키기 위해 여러 최적화 방법을 탐구합니다. RbRL은 인간 평가를 기반으로 한 보상 추론 방법을 통해 보상 함수가 없는 환경에서도 정책 학습을 가능하게 합니다. 이 연구는 다양한 하이퍼파라미터의 영향을 이해하기 위한 포괄적 실험을 제공하며, 사용자가 RbRL에서 하이퍼파라미터를 선택하는 데 유용한 지침을 제시합니다.

- **Technical Details**: RbRL은 cross entropy loss를 최소화하여 인간 평가와 추정된 평가 간의 차이를 정량화합니다. RbRL은 PbRL보다 더 다양한 하이퍼파라미터를 가지고 있으며, 이로 인해 최적화가 더욱 중요해집니다. 본 논문에서는 보상 경계, 신뢰 지수와 같은 RbRL 고유의 최적화 방법을 포함하여 총 8개의 최적화 기술을 적용하여 RbRL의 성능을 향상시키려 합니다.

- **Performance Highlights**: 기존의 기계 학습 최적화 기술을 적용하여 RbRL의 성능을 100% 향상시킬 수 있는 경우가 있음을 보였습니다. 최적화된 RbRL은 다양한 평점 클래스에서 일관된 성능을 유지하며, 변동성을 줄이는 결과를 보여줍니다. 본 논문에서는 RbRL의 하이퍼파라미터를 선택하는 데 있어 성능 최적화를 위한 포괄적인 연구를 제공합니다.



### BlobGEN-Vid: Compositional Text-to-Video Generation with Blob Video Representations (https://arxiv.org/abs/2501.07647)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 복잡한 텍스트 프롬프트를 따르는 기존 비디오 생성 모델의 한계를 극복하기 위해 blob 비디오 표현(blob video representation)을 제안합니다. 이 새로운 표현은 사용자가 객체의 움직임과 세부 외형을 보다 쉽게 제어할 수 있게 합니다. BlobGEN-Vid라는 모델을 개발하여 물체 중심의 이동과 외형 제어를 가능하게 하였으며, 입체적인 일관성을 향상시키기 위한 3D attention 모듈도 도입했습니다.

- **Technical Details**: Blob 비디오 표현은 객체 인스턴스를 나타내는 blob 시퀀스로 구성되며, 이는 비디오(또는 3D 장면)에서 자동으로 추출할 수 있습니다. 각각의 blob은 위치, 크기 및 방향을 지정하는 벡터로 정의되며, 사용자들이 구조화된 텍스트 형태로 쉽게 생성하고 조작할 수 있습니다. 이를 통해 motion과 semantic control을 동시에 가능하게 합니다. BlobGEN-Vid는 UNet이나 DiT 기반의 기존 비디오 확산 모델에 적용 가능하며, 특히 시간적인 일관성을 유지할 수 있는 장점이 있습니다.

- **Performance Highlights**: BlobGEN-Vid는 여러 벤치마크에서 기존의 레이아웃 가이드 비디오 생성 모델보다 더 뛰어난 성능을 보였습니다. 특히, layout controllability는 mIOU에서 20% 이상 개선되었으며, CLIP 유사도에서 5% 향상되었습니다. LLMs와 결합할 경우, BlobGEN-Vid는 복잡한 장면에서도 높은 구성 정확도를 달성하며 뛰어난 성능을 입증했습니다.



### Real-Time Decision-Making for Digital Twin in Additive Manufacturing with Model Predictive Control using Time-Series Deep Neural Networks (https://arxiv.org/abs/2501.07601)
- **What's New**: 이 논문은 디지털 트윈(Digital Twin)과 머신 러닝(Machine Learning) 기술을 결합하여 자율 제조 공정에서 실시간 의사결정을 위한 동시 다단계 모델 예측 제어(Model Predictive Control, MPC) 프레임워크를 제안합니다. 특히, ‘시간계열 조밀 인코더(Time-Series Dense Encoder, TiDE)’라는 다변량 딥 뉴럴 네트워크(DNN)를 서브 레이트 모델로 활용하여 제조 시스템의 효율적인 제어를 가능하게 합니다.

- **Technical Details**: 제안된 다단계 MPC 프레임워크는 예측 지평선 전반에 걸쳐 미래 상태를 예측할 수 있는 TiDE 모델을 사용하여 리얼타임 최적 제어 입력을 계산합니다. 이를 통해 레이저 파워와 같은 제어 입력의 연속 시퀀스를 최적화하여 각기 다른 조건에서도 공정 제어를 개선할 수 있습니다. 이 연구는 Directed Energy Deposition(DED) 적층 제조를 사례 연구로 적용하여 열 제어와 관련된 모델링의 유용성을 보여줍니다.

- **Performance Highlights**: 실험결과, 제안한 MPC는 PID 제어기와 비교해 더 부드럽고 변동이 적은 레이저 파워 프로파일을 생성하며, 목표 온도를 보다 정확하게 추적하여 품질을 보장합니다. 또한, TiDE 모델은 정확한 예측을 통해 각종 제약 조건을 만족시키며, 제조 과정에서 발생할 수 있는 결함을 사전에 예방하는 데 기여함을 확인했습니다.



### Multi-task Domain Adaptation for Computation Offloading in Edge-intelligence Networks (https://arxiv.org/abs/2501.07585)
- **What's New**: 본 논문은 멀티 액세스 엣지 컴퓨팅(Multi-Access Edge Computing, MEC)에서 효율적인 계산 오프로드(Computation Offloading)를 위한 새로운 접근 방식을 제안합니다. Multi-Task Domain Adaptation(MTDA)라는 새로운 방법은 도메인 변화가 있는 환경에서 모델의 일반화 능력을 향상시키는 것을 목표로 합니다. MTDA 모델은 교사-학생 아키텍처를 활용하여 추론 시 소스 도메인 데이터에 접근할 필요 없이 지속적인 적응이 가능하도록 합니다.

- **Technical Details**: MTDA 접근 방식은 멀티 태스크 러닝 프레임워크를 활용하여 이진 오프로드 결정과 자원 할당 문제를 동시에 관리합니다. 이 방법은 내장된 교사 모델을 통해 고품질의 의사 라벨을 생성하여 모델 성능을 개선합니다. 제안된 구조는 다중 사용자 환경에서 성능을 유지하며, 효율적인 자원 관리를 통해 지연을 최소화합니다.

- **Performance Highlights**: MTDA는 성능 점검에서 기존 벤치마크 방법에 비해 평균 제곱 오차(mean squared error)와 정확도(accuracy) 측면에서 월등한 성능을 보였습니다. 특히 사용자가 증가하는 환경에서 뛰어난 성능을 유지하며, 다양한 시나리오에서도 높은 성능을 나타냅니다. 이는 MTDA가 실제 MEC 응용 프로그램에서의 실용성을 지원함을 의미합니다.



### A Hybrid Framework for Reinsurance Optimization: Integrating Generative Models and Reinforcement Learning (https://arxiv.org/abs/2501.06404)
- **What's New**: 이 논문은 재보험 최적화에 대한 새로운 하이브리드 프레임워크를 제안합니다. 이 프레임워크는 생성 모델인 Variational Autoencoders (VAEs)와 Proximal Policy Optimization (PPO)을 이용한 강화 학습 (Reinforcement Learning) 기술을 결합하여 동적이고 확장 가능한 재보험 전략을 최적화합니다. 이 방법론은 복잡한 청구 분포의 생성 모델링과 강화 학습의 적응적 의사 결정 기능을 통합하여 재보험 전략의 커스터마이징을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 보험사의 운영을 모델링하기 위해 이산 시간 모델링을 기반으로 합니다. 각 시간 간격 동안 청구를 발생시키고, 보험료 수익을 창출하며, 포트폴리오의 리스크를 관리하는 과정을 포함합니다. 또한, 이러한 시간적 구조는 보험업계에서의 재보험 전략 최적화 시 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과는 전통적인 최적화 기법에 비해 우수한 적응성, 확장성 및 강건성을 보여줍니다. 하이브리드 프레임워크는 높은 최종 잉여금과 컴퓨터 효율성을 달성하여 다양한 스트레스 테스트 시나리오(예: 팬데믹 영향, 재해 사건)에서도 안정성을 유지합니다. 이 연구는 재보험 최적화에서 머신러닝과 AI 기술의 통합이 가지는 변혁적 가능성을 입증하였습니다.



### Parallel Key-Value Cache Fusion for Position Invariant RAG (https://arxiv.org/abs/2501.07523)
Comments:
          5 pages

- **What's New**: 이번 논문에서는 RAG (Retrieval Augmented Generation) 파이프라인의 효율성을 높이기 위해 키-값 융합(Key Value Fusion, KV Fusion) 프레임워크를 제안합니다. 이 프레임워크는 디코더 전용 모델인 LLMs에서 정보의 위치에 대한 민감성을 줄여주는 새로운 방법으로, 입력 순서와 관계없이 일관된 출력을 생성하는 것을 목표로 하고 있습니다. 특히, 입력 문맥의 위치에 관계없이 모델의 일관된 반응을 보장하여 기존 방법보다 강력한 성능을 보여줍니다.

- **Technical Details**: KV Fusion 구조는 두 가지 구성 요소로 이루어져 있습니다. 첫 번째는 주요-값 캐시를 병렬로 추출하는 프리필 디코더(prefill decoder)이며, 두 번째는 추출된 키-값 캐시를 사용하여 일관된 출력을 생산하는 훈련 가능한 디코더(trainable decoder)입니다. 이를 통해 입력 문맥에 균일한 위치 정보를 주입하여, 다양한 입력 순서에서도 일관된 결과를 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, KV Fusion은 NQ, TriviaQA, POPQA를 포함한 열린 도메인 질문 응답 데이터셋에서 각각 21.4%, 6.4%, 6.6%의 정확도 개선을 달성했습니다. 또한, 기존 방법들보다 다른 문맥이 추가되더라도 응답의 정확도가 강건하고 안정적인 것을 보여주었습니다. 이러한 성과는 KV Fusion의 위치 불변성(position invariance) 특성 덕분입니다.



### Inductive Learning of Robot Task Knowledge from Raw Data and Online Expert Feedback (https://arxiv.org/abs/2501.07507)
- **What's New**: 로봇의 자율성이 증가함에 따라 인간-로봇 상호작용에서 신뢰와 사회적 수용의 문제도 함께 증가하고 있습니다. 본 논문에서는 노이즈가 있는 예제를 활용하여 로봇 실행에서 직접 작업 사양을 추출하는 오프라인 알고리즘을 제안합니다. 이 알고리즘은 환경에 대한commonsense 개념과 비지도 행동 식별 알고리즘을 결합하여 작업의 전제 및 효과를 학습합니다.

- **Technical Details**: 제안된 방법론에서는 Inductive Logic Programming (ILP)과 Answer Set Programming (ASP)의 의미론을 활용하여 로봇 실행에서 얻은 노이즈 있는 원시 데이터로부터 작업 사양을 학습합니다. 또한 사용자의 피드백을 통한 작업 지식의 점진적 개선을 보장하기 위해 인간과 자율 실행을 혼합하는 프레임워크를 발전시켰습니다. 이 구현은 안전한 자율 로봇 계획 및 실행을 확립하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 방법론은 수술 로봇 시나리오 및 조작 작업에서의 표준적 벤치마크를 통해 견고함과 데이터 및 시간 효율성을 입증했습니다. 특히, peg transfer 작업에서는 짧은 시간 내에 최대한으로 성공적인 반복을 요구하며, 이러한 실험들은 보다 복잡한 도메인으로의 확장 가능성을 보여줍니다.



### Data and System Perspectives of Sustainable Artificial Intelligenc (https://arxiv.org/abs/2501.07487)
- **What's New**: 지속 가능한 AI(Sustainable AI)는 환경 영향을 줄이고 지속 가능성을 달성하는 방향으로 AI 시스템을 개발하고 사용하는 하위 분야로, 대규모 언어 모델과 같은 AI 모델의 교육 및 추론이 막대한 컴퓨팅 파워를 소비함에 따라 점점 더 중요해지고 있다. 본 논문에서는 데이터 획득, 데이터 처리 및 AI 모델의 훈련 및 추론과 관련하여 현재의 문제, 기회 및 가능한 해결책과 함께 미래의 도전 과제에 대해 논의한다.

- **Technical Details**: AI의 발전과 함께 데이터 획득(data acquisition)은 AI 모델 개발의 중요한 요소로 떠오르고 있지만, 환경 영향, 개인 정보 보호, 데이터 품질 및 법적 준수와 같은 여러 도전 과제가 존재한다. 특히 대규모 머신 러닝 모델(large-scale machine learning models)인 딥러닝 훈련이 많은 에너지를 소비한다는 점이 우려된다. 투명성 부족과 데이터 품질 문제 또한 지속 가능한 AI 개발에 있어 중대한 도전 과제로 남아있다.

- **Performance Highlights**: AI 시스템의 데이터 획득 과정을 개선할 수 있는 많은 기회가 있으며, 여기에는 크라우드소싱(crowdsourcing) 및 능동 학습(active learning) 기법이 포함된다. 크라우드소싱을 통해 대규모 데이터 주석 작업을 경제적으로 처리할 수 있으며, 능동 학습을 이용해 가장 정보가 많은 데이터 샘플에 우선 순위를 두고 주석을 추가함으로써 데이터 획득 비용을 줄인다. 각각의 기술은 AI 모델 훈련의 효율성과 지속 가능성을 향상시킨다.



### A Survey of Embodied AI in Healthcare: Techniques, Applications, and Opportunities (https://arxiv.org/abs/2501.07468)
Comments:
          44 pages, 11 figures

- **What's New**: 이번 연구에서는 Embodied AI (EmAI)가 의료 분야에서 직면한 효율성, 접근성, 개인화 문제를 해결하는 데 기여할 수 있는 가능성을 탐구합니다. EmAI는 다양한 알고리즘과 로봇 공학을 통합하여, 의료 현장에서 물리적 상호작용을 가능하게 하여 혁신的인 변화를 가져올 것으로 기대되고 있습니다. 연구는 EmAI의 주요 구성 요소인 'AI 시스템의 뇌'에 중점을 두어 다양한 의료 분야에서의 응용 사례를 제시하고 있습니다.

- **Technical Details**: EmAI 시스템은 인지, 행동 제어, 의사 결정 및 기억 처리 기능을 담당하는 AI 알고리즘을 포함하여 물리적 형태를 통해 외부와 상호작용할 수 있습니다. 최근의 AI 알고리즘 발전은 비지도 학습, 교차 모달 융합 기술, 심층 강화 학습 등 다양한 기술들이 통합되어 EmAI가 더욱 정교하고 상황 인식 있는 시스템으로 발전할 수 있도록 하고 있습니다. 이러한 AI 기술은 의료 진단 및 치료 추천 등에서 매우 중요한 역할을 수행할 수 있습니다.

- **Performance Highlights**: EmAI는 로봇 진단, 정밀 외과 개입 및 개인 맞춤형 재활 치료 등을 통해 의료 서비스를 혁신하고 있습니다. 이 기술은 의료 업무의 효율성을 높이는 동시에 더 좋은 건강 결과를 제공하고 회복 시간을 단축시킵니다. EmAI는 감정적 지원과 동반자 역할도 수행해, 아동, 노인, 만성 질환자 등 취약 계층에 큰 도움이 되고 있어, 의료 제공자의 부담을 덜어주고 있습니다.



### Understanding and Benchmarking Artificial Intelligence: OpenAI's o3 Is Not AGI (https://arxiv.org/abs/2501.07458)
Comments:
          15 pages

- **What's New**: OpenAI의 o3 모델은 ARC-AGI 벤치마크에서 87.5%라는 높은 점수를 기록했습니다. 이는 LLM(대규모 언어 모델) 기반 시스템이 인공지능을 의미하는지 또는 인공지능 일반(AGI)으로 나아가고 있는지를 질문하게 만듭니다. 새로운 인텔리전스 개념은 다양한 목표를 적은 지식으로 더 효율적으로 달성하는 능력을 강조합니다.

- **Technical Details**: ARC-AGI는 AI의 일반 지능을 측정하기 위해 설계된 벤치마크로, 1,000개의 고유한 작업으로 구성됩니다. 이 벤치마크는 기존 기술이 아닌 새로운 과제를 풀 수 있는 능력을 평가합니다. 각 작업은 입력과 출력을 포함한 파라미터를 설정하며, 특정 규칙에 따라 조작됩니다.

- **Performance Highlights**: o3는 ARC-AGI의 반 비공식 테스트 세트에서 87.5%의 점수를 기록하면서, 그 전 여자대회들과 비교해 월등한 성과를 보였습니다. 그러나 o3의 성공이 항상 지능을 나타내는 것은 아니며, ARC-AGI가 인지력 측정의 완벽성에 제한이 있을 수 있습니다.



### Online inductive learning from answer sets for efficient reinforcement learning exploration (https://arxiv.org/abs/2501.07445)
- **What's New**: 본 논문은 인덕티브 로직 프로그래밍(Inductive Logic Programming, ILP)과 강화 학습(Reinforcement Learning, RL)을 결합하여 훈련 성능과 설명 가능성을 개선하는 혁신적인 방법을 제시합니다. 이 방법은 불완전한 예로부터 논리 규칙을 학습하여 에이전트 정책의 설명 가능한 근사를 생성하고, 학습된 규칙에 대한 답 집합(reasoning)을 수행하여 다음 배치의 탐색을 가이드합니다. 이를 통해 비효율적인 보상 형태 없이도 에이전트의 최적성을 보존할 수 있습니다.

- **Technical Details**: 이 방법에서 RL 에이전트는 경험 배치(상태-행동 시퀀스)를 수집하고, 이를 누적 보상에 따라 정렬하여 ILP를 위한 예제로 변환합니다. ILP는 이를 통해 정책의 논리적 근사를 학습하고, 다음 배치에서 에이전트의 탐색을 편향하여 성능을 향상시킵니다. 이 과정은 온라인 실행 중에 이루어지며, 최신 ASP(Answer Set Programming) 형식을 사용하여 논리적 사양을 표현하고 효율적인 휴리스틱 추론을 수행합니다.

- **Performance Highlights**: 이 방법을 Q-러닝 알고리즘에 통합하여 Pac-Man 시나리오에서 복잡도를 증가시키는 두 맵에서 초기 성과를 검증했습니다. 결과적으로, 에이전트가 달성하는 할인된 보수가 상당히 증가했으며, 적은 훈련 배치에서도 효과가 나타났습니다. 또한, 인덕티브 학습으로 인해 Q-러닝의 계산 시간은 최소한으로 영향을 받으며, 학습된 규칙이 신속하게 에이전트 정책에 대한 설명으로 수렴합니다.



### Empirical Evaluation of the Implicit Hitting Set Approach for Weighted CSPs (https://arxiv.org/abs/2501.07432)
- **What's New**: 본 논문에서는 가중 제약 만족 문제(Weighted CSP, WCSP)를 해결하기 위해 기존의 IHS(Implicit Hitting Set) 접근 방식에 대한 여러 대안 알고리즘을 탐구합니다. 특히, Boolean 프레임워크에서 영감을 받은 다양한 방법을 통해 저비용 타격 벡터(hitting vectors)를 계산하고 이를 고비용 코어(high-cost cores)로 변환하는 방법을 고찰합니다. 이러한 방식을 통해 32개의 다양한 알고리즘을 실험하고 그 성능을 평가합니다.

- **Technical Details**: 가중 제약 만족 문제는 변수 집합(X)과 제한 조건 세트(C)로 정의되는 CSP의 확장이며, 각 변수가 특정한 도메인에서 값을 가집니다. IHS 접근법은 문제를 여러 개의 불만족 코어로 나누어 해결하는 반복적인 방식으로, 이 코어들을 피하며 최적의 솔루션을 찾습니다. 본 연구에서는 특정 벡터의 비용(cost)을 계산하는 방법과 이를 개선된 코어로 변환하는 4가지 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 서로 다른 알고리즘은 성능 면에서 큰 차이를 보이며, 특정 비용 함수 병합(cost-function merging) 방법과 최대 코어(maximal cores)를 계산하는 전략이 가장 안정적인 접근으로 나타났습니다. 최적의 타격 벡터와 비용 제한 타격 벡터 간의 구별이 명확하지 않지만, 비용 제한 벡터가 더 많은 구현 옵션을 제공하여 더 큰 잠재력을 지니는 것으로 보입니다.



### Initial Findings on Sensor based Open Vocabulary Activity Recognition via Text Embedding Inversion (https://arxiv.org/abs/2501.07408)
- **What's New**: OV-HAR(Open Vocabulary Human Activity Recognition)은 기존의 분류 중심 접근 방식을 넘어서는 혁신적인 방법으로, 각 활동을 자연어로 변환하여 기본 동작의 시퀀스로 분해함으로써 활동 인식을 가능하게 합니다. 이 방법은 오토 회귀 기반의 대형 언어 모델에 의존하지 않고, 고정 크기 임베딩 회귀를 사용하여 연산 부하를 최소화합니다. 또한, 자연어 표현을 통해 보이지 않는 활동을 인식하고 기술할 수 있는 가능성을 열어줍니다.

- **Technical Details**: OV-HAR은 센서 데이터를 의미 있는 임베딩으로 변환하여 자연어 설명과 특정 활동 클래스 간의 쌍방향 매핑을 가능하게 하는 새로운 파이프라인을 도입합니다. 이 모델은 LLaMA 3를 사용하여 활동 클래스를 기본 동작 시퀀스로 분해하고, 생성된 문장을 gtr-t5-base 모델을 통해 768차원의 벡터 임베딩으로 변환합니다. 이 접근 방식은 MSE(mean squared error) 손실 함수를 사용하여 모델이 센서 데이터를 의미 있는 임베딩으로 효과적으로 변환하도록 최적화됩니다.

- **Performance Highlights**: OV-HAR은 다양한 센서 모달리티에서 보이지 않는 활동을 인식하고, 여러 센서 모달리티 간의 일반화를 입증하며, 특히 NTU-RGBD 데이터에서 높은 성과를 보여줍니다. 기존 모델과 비교했을 때, OV-HAR은 제시된 도전 과제를 해결하면서도 의미 있는 결과를 도출할 수 있는 가능성을 제시하고 있습니다. 이러한 능력은 더 탄탄하고 유연한 인간 중심 AI 시스템을 위한 기반을 마련하며, 활동 인식의 경계를 확장합니다.



### The Essentials of AI for Life and Society: An AI Literacy Course for the University Community (https://arxiv.org/abs/2501.07392)
Comments:
          Accepted to EAAI-25: The 15th Symposium on Educational Advances in Artificial Intelligence, collocated with AAAI-25

- **What's New**: 텍사스 대학교 오스틴 캠퍼스에서 AI 문해력을 증진하기 위한 1학점 과정을 개발했습니다. 2023년 가을 학기를 위한 신속한 강의 배포 요청에 따라, AI의 기본 개념에서부터 사회적 문제에 이르기까지 다양한 주제를 다루는 14주 세미나 형식의 강의가 설계되었습니다. 이 과정은 대학 학생, 교수, 직원 및 지역 사회 구성원 모두에게 공개되고 온라인으로 제공되었습니다.

- **Technical Details**: 이 과정은 AI의 정의, 능력, 한계 및 사회적 임팩트 등을 다루며, 기술적 배경이 없는 사람들을 위해 계획되었습니다. 강의는 AI 문해력을 증진시키기 위한 목적으로 설계된 14가지 강의를 포함하고 있으며, 이 과정의 피드백은 향후 3학점 과정 개발에 활용되었습니다. 학생들은 주 1시간 강의를 수료하고, 읽기 자료를 준수하며, 질의응답 세션에 참여해야 합니다.

- **Performance Highlights**: 과정 참석자들은 AI 문해력이 증진되었다고 보고했으며, 강의 참석자 간의 피드백을 통해 다양한 개선 사항이 도출되었습니다. 설계된 강의는 AI의 주요 기술 하위 분야들을 포함하며, 윤리적 고려사항도 함께 다루어졌습니다. 이는 비 기술적 전공자와 대학 전체의 다양한 청중을 대상으로 한 혁신적인 접근이라 할 수 있습니다.



### Anonymization of Documents for Law Enforcement with Machine Learning (https://arxiv.org/abs/2501.07334)
Comments:
          Accepted at IEEE Symposium on CI in Security, Defence and Biometrics 2025 (IEEE CISDB)

- **What's New**: 이 논문에서는 법 집행 기관이 개인 정보를 보호하기 위해 스캔된 문서 이미지를 자동으로 익명화하는 시스템을 제안합니다. 이 시스템은 민감한 영역을 자동으로 감지하고 수동으로 익명화된 참조 문서의 지식을 결합하여 데이터 보호 규정을 준수하면서 수작업을 줄입니다. 단일 익명화된 예제를 사용하여 같은 유형의 모든 문서에서 효과적으로 레드랙션을 수행하는 자가 지도학습(Self-supervised learning) 이미지 모델을 활용합니다.

- **Technical Details**: 이 방법은 문서의 유형에 대한 단일 참조 문서로부터 이미지를 검색하는 인스턴스 검색(Instance Retrieval) 과정을 기반으로 합니다. 데이터베이스에서 직접 문서 메타데이터를 요구하지 않기 때문에 다양한 환경에서도 적용 가능한 프레임워크입니다. 피처 매칭은 코사인 유사성(Cosine Similarity)을 통해 수행되며, DinoV2라는 자가 지도(Self-supervised) 학습 방법을 사용하여 문서 이미지의 의미 있는 피처를 추출합니다.

- **Performance Highlights**: 제안된 방법은 자동화된 레드랙션 시스템 및 참조 익명화를 다른 문서에 단순히 복사하여 붙여넣는 과정을 모두 초월하는 성능을 보였습니다. 손으로 만든 데이터 세트에서 전문적으로 주석이 달린 결과에 기반하여 이 프레임워크의 우수성을 입증하였습니다. 이 연구는 데이터 처리에서 데이터 보호 요구 사항을 충족하는 혁신적인 접근 방식을 제시하여 관련 분야에 기여할 것으로 기대됩니다.



### Principles for Responsible AI Consciousness Research (https://arxiv.org/abs/2501.07290)
- **What's New**: 최근 연구에 따르면 현재 또는 가까운 미래에 의식 있는 AI 시스템(conscious AI systems)을 구축하는 것이 가능할지도 모른다고 합니다. 이러한 시스템들은 도덕적 고려(moral consideration)를 받을 필요가 있으며, 많은 수의 의식 있는 시스템들이 창조되고 고통을 겪는 상황이 발생할 수 있습니다. AI 시스템이나 AI로 생성된 캐릭터(character)는 점차 의식이 있는 것처럼 보일 수 있어, 이들의 도덕적 지위에 대한 논쟁을 초래할 수 있습니다.

- **Technical Details**: AI 연구에 관련된 조직들은 의식에 관한 연구 및 배포 선택, 그리고 대중 소통(public communication)에 대한 원칙과 정책을 설정해야 합니다. 조직이 AI 의식을 연구하지 않기로 선택하더라도, 고급 AI 시스템(advanced AI systems)을 개발하는 과정에서 의식 있는 존재(conscious entities)가 우연히 창조될 위험이 존재합니다. 따라서 책임 있는 연구(responsible research) 및 배포 관행이 이러한 가능성을 다루기 위해 필수적입니다.

- **Performance Highlights**: 우리는 책임 있는 연구를 위한 다섯 가지 원칙을 제안하며, 연구 기관들이 이러한 원칙에 대한 자발적이고 공개적인 약속(commitments)을 해야 한다고 주장합니다. 우리의 원칙은 연구 목표(research objectives) 및 절차(procedures), 지식 공유(knowledge sharing) 및 대중 소통에 관한 내용입니다.



### LLM-Net: Democratizing LLMs-as-a-Service through Blockchain-based Expert Networks (https://arxiv.org/abs/2501.07288)
- **What's New**: 이번 연구에서는 LLMs-as-a-Service의 민주화를 촉진하기 위해 블록체인 기반의 LLMs Networks (LLM-Net) 프레임워크를 제안하고 있습니다. 이 네트워크는 분산형 전문가 모델을 통합하여 지속적인 지식 성장을 보장하며, 서비스 품질을 유지하기 위한 협력적 프로모팅 메커니즘을 활용합니다. 또한 블록체인 기술을 통해 투명한 거래 및 성과 검증을 가능하게 하여 서비스 제공의 불변 기록을 생성합니다.

- **Technical Details**: LLM-Net은 여러 참가자의 집단 지식과 계산 자원을 활용하여 중앙 집중형 시스템의 제약을 완화합니다. 이 구조 내에서 특화된 LLMs 모델들이 배포되어 각 모델이 특정 도메인에 맞춘 전문가 서비스 기능을 제공합니다. 사용자 요청자는 간단한 쿼리를 통해 이 서비스에 접근할 수 있으며, 코디네이터를 통해 다양한 LLM 제공자와 상호작용을 지원합니다.

- **Performance Highlights**: 시뮬레이션 결과, LLM-Net의 명성 기반 메커니즘이 높은 성과를 보이는 응답자(LLM 제공자)를 선택하는 데 효과적임을 입증했습니다. 이는 콜라보레이티브 문제 해결 시나리오를 통해 서비스 품질을 유지하며, 분산된 전문성과 블록체인 기반의 책임성 통합을 통해 AI 발전을 지속할 수 있는 잠재력을 보여줍니다.



### Lifelong Learning of Large Language Model based Agents: A Roadmap (https://arxiv.org/abs/2501.07278)
Comments:
          46 pages

- **What's New**: 이 논문은 LLM(대규모 언어 모델) 기반 에이전트에 평생 학습(lifelong learning)을 포함시키기 위한 잠재적 기술을 체계적으로 요약한 첫 번째 조사입니다. 연구자들은 평생 학습이 이러한 에이전트가 다양한 환경에서 지속적으로 적응하고, 기억을 저장하며, 동적인 상호작용을 수행하는 데 중요하다는 점을 강조합니다. 이를 위해 이 논문은 에이전트를 인식 모듈, 기억 모듈, 행동 모듈의 세 가지 핵심 구성 요소로 구분하여 설명하고 있습니다.

- **Technical Details**: 논문에서는 평생 학습을 구현하기 위한 LLM 에이전트의 기본 개념과 아키텍처를 설명합니다. 특히, 인식 모듈은 다양한 입력 데이터(예: 텍스트, 이미지)를 통합하고, 기억 모듈은 진화하는 지식을 저장 및 복구하며, 행동 모듈은 동적 환경과의 상호작용을 위한 구체적인 행동을 처리하는 방식에 대해 논의합니다. 이러한 구성 요소들이 결합하여 지속적인 적응을 가능하게 하고, 재앙적 망각(catastrophic forgetting)을 완화하며, 장기 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 이 논문은 LLM 에이전트가 자율적으로 상호작용하면서 새로운 작업에 적응하고 기존 지식을 보존할 수 있는 방법을 제시합니다. 평생 학습 능력을 통합함으로써, 이러한 에이전트는 계속해서 지식을 습득하고 다양한 환경에서 실시간으로 적응하는 능력을 향상시킬 수 있습니다. 실제 응용 사례로는 로봇공학, 상호작용 보조기기 및 적응형 의사결정 지원 시스템을 포함하여, 이러한 에이전트가 직면하는 다양한 도전 과제를 극복할 수 있는 잠재력을 가지고 있습니다.



### Bridging Smart Meter Gaps: A Benchmark of Statistical, Machine Learning and Time Series Foundation Models for Data Imputation (https://arxiv.org/abs/2501.07276)
- **What's New**: 이 논문은 스마트 그리드에서의 시간 시계열 데이터의 결함을 해결하기 위해 Generative Artificial Intelligence (GenAI)와 빅데이터 분석 기법을 접목한 내용입니다. 특히 스마트 미터 데이터의 결함을 보완하기 위해 두 가지 대규모 언어 모델(LLMs)과 다섯 가지 시간 시리즈 기초 모델(Time Series Foundation Models, TSFMs)을 평가합니다. 기존의 기계 학습 모델과 통계 모델과 비교하며, 이러한 접근법이 전통적인 방법보다 더 우수한 성능을 발휘할 수 있음을 제시합니다.

- **Technical Details**: 연구는 2013년 런던의 가정용 에너지 소비 데이터를 포함한 공개 데이터셋을 사용하였습니다. 시간 간격은 30분 단위로 측정되며, 결측값을 유도하기 위해 인위적인 간극을 생성하였습니다. 모델 생성을 위해 다양한 통계 모델과 기계 학습 모델, LLM 및 TSFM을 선택하고 평가하는 방식을 사용했습니다. 이를 통해 각 모델의 성능과 신뢰성을 평가하였습니다.

- **Performance Highlights**: 결과적으로, TSFM은 특정 경우에 임퓨테이션 정확도를 크게 향상시킬 수 있는 가능성을 보여주었습니다. 그러나 계산 비용과 성능 향상 간의 트레이드오프는 여전히 중요한 고려사항으로 남아 있습니다. 이 연구는 결측값 처리와 예측 모델링의 새로운 접근법을 제시하며, 스마트 미터 데이터의 결함 보완에 대한 기존 연구와 차별화됩니다.



### Lessons From Red Teaming 100 Generative AI Products (https://arxiv.org/abs/2501.07238)
- **What's New**: 최근 몇 년 동안 AI red teaming은 생성 AI 시스템의 안전성과 보안을 검토하는 핵심 관행으로 자리 잡았습니다. 이 논문에서는 Microsoft에서 100개 이상의 생성 AI 제품을 대상으로 수행한 red teaming 경험을 바탕으로 내부 위협 모델 온톨로지를 제시하고, 8가지 주요 교훈을 공유합니다.

- **Technical Details**: AI red teaming은 모델 수준의 안전 기준을 넘어 실제 공격을 모사하여 전체 시스템의 안전성을 평가하려는 목표를 가지고 있습니다. 이 연구에서는 Microsoft에서의 경험을 기반으로, 공격자(Actor), 전술(TTPs), 약점(Weakness), 영향(Impact) 등 AI 시스템 취약점의 핵심 구성 요소를 모델링하기 위한 온톨로지를 개발했습니다.

- **Performance Highlights**: 우리의 연구 결과, AI red teaming은 AI 시스템의 보안 및 안전 위험을 모델링하는 데 있어 매우 유용한 프레임워크를 제공합니다. 특히, PyRIT라는 오픈 소스 Python 프레임워크를 통해 인간의 판단과 창의성을 보강함으로써, 중요한 취약점을 더 빠르게 식별할 수 있었습니다.



### Kriging and Gaussian Process Interpolation for Georeferenced Data Augmentation (https://arxiv.org/abs/2501.07183)
- **What's New**: 이번 연구는 제한된 데이터셋을 다루는 데 있어 중요한 데이터 증강(data augmentation) 기술을 다룹니다. 특히, La Réunion의 사탕수수 작물에서 Commelina benghalensis L.의 존재를 예측하기 위해 지리 참조 데이터의 보간(interpolation) 기법을 탐색합니다. 연구는 서로 다른 커널을 가진 가우시안 프로세스(Gaussian processes)와 다양한 변동계수를 사용하는 크리깅(kriging) 방법을 평가합니다.

- **Technical Details**: 이 연구의 주요 목적은 세 가지입니다: 첫째, 다양한 회귀(regression) 알고리즘에 대한 최적의 예측 성능을 제공하는 보간 방법을 식별하고, 둘째, 추가된 관측 데이터(observations)의 수에 따른 성능 변화를 분석하며, 셋째, 증강된 데이터셋의 공간적 일관성(spatial consistency)을 평가하는 것입니다. 결과적으로 GP 기반 방법, 특히 결합 커널(GP-COMB)을 사용하면 회귀 알고리즘의 성능이 현저히 향상되는 반면, 추가 데이터의 요구량은 적습니다.

- **Performance Highlights**: 연구 결과, GP-COMB 방법이 회귀 알고리즘의 성능을 크게 향상시키는 것으로 나타났고, 데이터 추가가 필요하지 않았습니다. 반면 크리깅은 약간 낮은 성능을 보였지만, 더 균일한 공간적 커버리지를 제공하여 특정 맥락에서는 장점으로 작용할 수 있습니다.



### Natural Language-Assisted Multi-modal Medication Recommendation (https://arxiv.org/abs/2501.07166)
Comments:
          10 pages

- **What's New**: 본 논문은 Natural Language-Assisted Multi-modal Medication Recommendation(NLA-MMR) 모델을 소개하며, 이는 환자와 약물의 표현을 통합적으로 학습하는 다중 모달 정렬 프레임워크입니다. NLA-MMR은 약물 및 환자 정보를 각각의 모달리티로 정의하고, 약물의 화학 구조와 텍스트 설명을 함께 활용합니다. 이 접근법은 환자의 EHR에서 얻은 텍스트 지식을 활용하여 약물 추천 문제를 보다 정교하게 해결할 수 있도록 돕습니다.

- **Technical Details**: NLA-MMR은 pretrained language models(PLMs)를 활용하여 환자 및 약물 모달리티에서의 지식을 추출합니다. 약물 모달리티에서는 화학 구조와 텍스트 설명을 통합하여 약물Representation을 생성하며, 환자 모달리티에서는 진단, 절차 및 증상의 텍스트 설명을 기반으로 환자Representation을 생성합니다. 이 모델은 교차 모달 정렬 모듈을 통해 환자의 처방에 따라 약물과 환자 정보를 연결합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋에서의 실험 결과, NLA-MMR은 Jaccard 점수에서 평균 4.72%의 성능 향상을 이루었습니다. 무엇보다 NLA-MMR은 기존의 최첨단 방법들과 비교했을 때, 뛰어난 성능을 자랑하며, Jaccard 점수에서 각각 2.86%, 7.01%, 4.29%의 향상을 기록했습니다. 이는 모델이 EHR과 약물에 대한 풍부한 임상 지식을 효과적으로 활용했음을 나타냅니다.



### QuantuneV2: Compiler-Based Local Metric-Driven Mixed Precision Quantization for Practical Embedded AI Applications (https://arxiv.org/abs/2501.07161)
Comments:
          18 pages, 10 figures, Accepted in Future Generation Computer Systems Journal

- **What's New**: Mixed-precision quantization 방법을 통한 모델 크기 감소와 정확도 저하 최소화가 연구되어 왔으나, 기존 연구들은 재훈련이 필요하고 컴파일 과정 중 발생하는 컴퓨테이셔널 오버헤드를 고려하지 못했습니다. 새로운 QuantuneV2는 이러한 문제를 해결하기 위해 설계된 컴파일러 기반의 혼합 정밀도 양자화 방법입니다. QuantuneV2는 양자화 전과 후에 각각 한 번만 추론을 수행하여 O(n)의 계산 복잡도로 설계되었습니다.

- **Technical Details**: QuantuneV2는 손실을 최소화하면서 모델의 정확도를 향상시키기 위해 가중치, 활성화 값, Signal to Quantization Noise Ratio(SQNR), Mean Squared Error(MSE)와 같은 로컬 메트릭을 사용하여 민감도 분석을 안정적으로 수행합니다. 이 방법은 또한 최적의 Graph Intermediate Representation(IR) 형태를 선택하여 전체적인 성능을 최적화합니다. 이러한 기술적 요소들은 QuantuneV2가 적은 메모리 공간에서 깊은 학습 모델을 효과적으로 실행할 수 있도록 도와줍니다.

- **Performance Highlights**: 실험 결과 QuantuneV2는 ResNet18v1, ResNet50v1, SqueezeNetv1, VGGNet, MobileNetv2 모델과 비교하여 정확도가 최대 10.28% 향상되고, 실행 속도가 12.52% 증가했습니다. 이러한 성능 개선은 임베디드 AI 환경에서 모델을 효율적으로 배포할 수 있음을 보여줍니다. 또한, QuantuneV2는 기존 방법에 비해 민감도 목록 생성 시간을 평균 99.99% 단축시켰으며, 다양한 하드웨어 플랫폼에서 최대 1.43배 더 빠른 실행 성능을 나타냈습니다.



### CureGraph: Contrastive Multi-Modal Graph Representation Learning for Urban Living Circle Health Profiling and Prediction (https://arxiv.org/abs/2501.07157)
- **What's New**: 이번 연구에서는 CureGraph라는 새로운 다중 모달(multi-modal) 표현 학습 프레임워크를 제안하여 도시 환경에서 노인의 건강 예측을 수행합니다. CureGraph는 그래프 기반 기법을 활용하여 각 지역 사회 내에서 만성 질환의 유병률을 추론하는 데 초점을 맞추고 있으며, 사진과 텍스트 리뷰를 포함한 다양한 다중 모달 정보를 활용하여 도시 지역 임베딩을 생성합니다. 이 연구는 기존 단일 데이터 모달에 의존하는 한계를 극복하여, 노인의 건강에 관한 포괄적인 도시 환경 프로파일링을 가능하게 합니다.

- **Technical Details**: CureGraph는 사전 훈련된 시각적 및 텍스트 인코더를 그래프 모델링 기법과 통합하여 교차 모달(cross-modal) 공간 의존성을 포착합니다. 연구자는 지역 사회의 15분 생활권을 주요 연구 단위로 설정하고, 해당 지역의 텍스트 설명, 리뷰 및 이미지를 포함한 다중 모달 데이터셋을 수집하여 활용합니다. 공간 오토상관 행렬을 통해 지역 사회 생활권 간의 지리적 관계를 파악하고, 그래프 합성곱 신경망(GCN)을 통해 노인의 만성 질환 유병률을 예측합니다.

- **Performance Highlights**: 실험 결과, CureGraph는 노인 질병 리스크 예측 작업에서 기존 최상의 기준 모델에 비해 평균 28% 향상을 보였습니다. 또한, 연구 모델은 단계별 만성 질환 진행 상황을 식별할 수 있도록 하여, 도시 내 비교 공공 건강 분석을 지원하고 지속 가능한 도시 개발 및 삶의 질 향상을 위한 실행 가능한 통찰력을 제공합니다.



### FlexQuant: Elastic Quantization Framework for Locally Hosted LLM on Edge Devices (https://arxiv.org/abs/2501.07139)
- **What's New**: 이 논문에서는 edge devices에서 LLM(대형 언어 모델)의 유연한 호스팅을 위한 새롭고 혁신적인 프레임워크인 FlexQuant을 제안합니다. FlexQuant은 현재의 SoTA 메서드에 비해 15배의 전환 세분성과 10배의 저장 비용 절감을 제공합니다. 기존의 모델에 비해 더 작은 저장 공간에서도 작동할 수 있는 유연성을 제공합니다.

- **Technical Details**: FlexQuant은 LLM의 메모리 발자국(footprint) 탄력성을 개선하기 위한 여러 기법들을 통합하고, 다양한 저장 한계에 따라 선택 가능한 여러 양자화 모델군을 생성하는 방법론입니다. 이 접근법은 메모리-정확도 트레이드 오프를 더욱 향상시키며, LLM 성능을 유지하면서도 탄력성 정책의 전체 저장 비용을 40% 감소시킵니다. 또한 FlexQuant은 다양한 양자화 방법을 지원하여 시스템 설계자에게 더 많은 선택지를 제공합니다.

- **Performance Highlights**: FlexQuant은 15배 개선된 전환 세분성과 10배 감소된 저장 비용을 바탕으로 edge device에서의 LLM 배치를 위한 탁월한 성능을 제공합니다. 제안된 프레임워크는 메모리 효율성을 증대시키고, 다양한 애플리케이션의 SLOs(서비스 수준 목표)를 충족하는 데 필요한 유연성을 제공합니다. 이로 인해 FlexQuant은 기존 모델들에 비해 큰 성능 향상을 이뤄내고 있습니다.



### How GPT learns layer by layer (https://arxiv.org/abs/2501.07108)
- **What's New**: 본 논문은 OthelloGPT 모델을 통해 대형 언어 모델(LLMs)의 내부 세계 모델 구성 방식을 분석합니다. LLM은 언어 처리 및 전략 게임과 같은 여러 작업에서 뛰어난 성능을 보이지만, 일반화된 내부 표현을 구축하는 데 어려움을 겪습니다. 본 연구에서는 Sparse Autoencoders(SAEs)와 선형 프로브를 비교하여 LLM의 구성적 특성을 이해하는 데 중점을 두었습니다.

- **Technical Details**: 연구에서는 Othello-GPT 모델의 각 레이어에서 학습된 기능을 추적하여, 초기 레이어는 정적 속성(예: 보드 모양)을 캡처하고, 심층 레이어는 동적 변화(예: 타일 변경)를 반영하는 계층적 진행을 보여줍니다. Sparse Autoencoders는 비선형 프로브보다 더 고유하고 해체된 특성을 발견하며, 이는 특히 구성적 특성에 유용합니다. 이러한 기술들을 통해 타일 색상 및 보드 안정성과 같은 복잡한 게임 개념을 반영하는 기능을 해독합니다.

- **Performance Highlights**: OthelloGPT는 차별화된 기능을 보여 주며, 각 레이어의 정확도와 타일 색상이 학습 내용 캐치에서 어떻게 효과적인지를 비교합니다. 이 연구는 GPT 모델, 트랜스포머 및 LLM의 내부 표현을 이해하는 기초를 마련하여, 에이전트가 작업 전반에 걸쳐 일관된 적응형 행동을 유지할 수 있도록 합니다. 연구 결과를 공공에 공개하여, 다른 연구자들이 추가 실험과 분석을 할 수 있는 기회를 제공합니다.



### MathReader : Text-to-Speech for Mathematical Documents (https://arxiv.org/abs/2501.07088)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 연구에서는 아카데믹 문서의 수학 공식을 효과적으로 읽어주는 TTS (Text-to-Speech) 시스템인 MathReader를 제안합니다. 기존의 TTS 도구들은 LaTeX로 작성된 수학 공식을 이해하지 못하고 불완전한 출력을 제공하는 문제를 해결하고자 합니다. MathReader는 OCR (Optical Character Recognition), 미세 조정된 T5 모델, TTS를 통합하여 Word Error Rate (WER)를 감소시키는데 성공했습니다. 이는 특히 시각 장애인과 같은 사용자들이 문서를 청취할 때 유용할 것입니다.

- **Technical Details**: MathReader는 다섯 단계로 구성된 파이프라인을 통해 문서를 음성으로 변환합니다. 이 과정에서 Nougat-small을 사용하여 PDF 문서를 mmd 파일로 변환하고, T5-small 모델을 통해 LaTeX 공식을 스포큰 영어로 변환합니다. 최종적으로 TTS 모델에 입력될 수 있도록 LaTeX 코드를 일반 영어로 대체하여 모든 공식을 정확하게 읽을 수 있도록 합니다. 이 프로세스는 기존 시스템들이 간과한 수학적 맥락을 고려하는데 중점을 두고 있습니다.

- **Performance Highlights**: MathReader는 Microsoft Edge와 Adobe Acrobat보다 현저히 낮은 WER을 보여줍니다. Microsoft Edge 대비 WER이 0.510에서 0.281로, Adobe Acrobat 대비 0.617에서 0.281로 감소했습니다. 이로 인해 MathReader는 문서 읽기를 원하는 사용자들에게 보다 나은 접근성을 제공하며, 특히 수학 및 과학 문서의 이해를 향상시킬 수 있을 것으로 예상됩니다.



### ADKGD: Anomaly Detection in Knowledge Graphs with Dual-Channel Training (https://arxiv.org/abs/2501.07078)
Comments:
          Preprint. 11 figures, 6 tables

- **What's New**: 이번 논문에서는 이중 채널 학습(due-channel learning)을 활용한 지식 그래프의 이상 탐지 알고리즘인 ADKGD를 제안합니다. ADKGD는 엔티티 뷰(entity-view)와 트리플 뷰(triplet-view)에서 표상을 향상시키며, 내부 정보 집계와 맥락 정보 집계를 통합하여 성능을 개선합니다. Kullback-Leibler(KL) 손실 함수를 도입하여 두 채널 간의 평가 함수의 정확성을 높이는 데 초점을 맞추었습니다. 세 가지 실제 지식 그래프(WN18RR, FB15K, NELL-995)에 대한 실험 결과 ADKGD가 최첨단 이상 탐지 알고리즘보다 우수한 성능을 보였습니다.

- **Technical Details**: 지식 그래프(KG)는 다양한 출처의 데이터를 통합하는 발전된 데이터 구조입니다. 그러나 실세계의 비구조적 데이터에서 트리플을 추출하는 과정에서 오류가 발생할 수 있습니다. ADKGD는 각각 엔티티 뷰와 트리플 뷰로부터 효과적인 학습을 진행하며, 크로스 레이어 접근 방식을 통해 내재 정보 집계(internal information aggregation)와 맥락 정보 집계(context information aggregation)를 통합합니다. 이를 통해 보다 정확한 이상 탐지를 구현할 수 있습니다.

- **Performance Highlights**: ADKGD는 세 가지 실제 지식 그래프에 대한 실험에서 최첨단 이상 탐지 알고리즘보다 우수한 성능을 입증했습니다. 이를 통해 이상 탐지 분야에 있어 지식 그래프의 품질을 보장할 수 있는 새로운 접근법을 제시하였습니다. ADKGD의 소스코드와 데이터셋은 공공에 공개되어 있어 연구자들이 쉽게 접근할 수 있습니다.



### Value Compass Leaderboard: A Platform for Fundamental and Validated Evaluation of LLMs Values (https://arxiv.org/abs/2501.07071)
- **What's New**: 이번 연구에서는 LLMs(Large Language Models)의 가치 평가를 위한 새로운 플랫폼인 Value Compass Leaderboard를 소개합니다. 이 플랫폼은 LLM의 기본 가치를 명확히 하고, 평가의 진정성을 확보하며, 인간 가치의 다원성을 반영하기 위해 설계된 세 가지 혁신적인 모듈을 제공합니다. 이를 통해 LLM의 값에 대한 평가 방식을 획기적으로 개선하고자 합니다.

- **Technical Details**: Value Compass Leaderboard는 기본 인간 가치를 기반으로 하여 LLM의 기본 가치를 전반적으로 파악할 수 있도록 돕습니다. 또한, 실시간 LLM 피드백을 바탕으로 평가 항목을 적응시키는 진화적 평가 프레임워크를 제공합니다. 마지막으로, 개인 및 문화의 가치 우선순위를 반영하는 가중치로 LLM의 가치를 정량화하는 메트릭을 제안하여 평가의 유연성을 높입니다.

- **Performance Highlights**: 사용자는 다양한 LLM의 종합 가치를 확인할 수 있으며, 기본 가치 차원에 따라 평가 세부 정보를 예시와 함께 확인할 수 있습니다. 또한, 사용자가 관심 있는 LLM을 세부적으로 비교할 수 있도록 테이블 및 레이더 차트 시각화를 지원합니다. 이러한 기능을 통해 LLM과 인간 가치 간의 정렬을 진전시키기 위한 네비게이터 역할을 할 것으로 기대됩니다.



### PoAct: Policy and Action Dual-Control Agent for Generalized Applications (https://arxiv.org/abs/2501.07054)
- **What's New**: 이 논문에서는 Policy and Action Dual-Control Agent (PoAct)를 제안하여 복잡한 문제 해결을 위한 새로운 방법론을 제공합니다. 기존의 ReAct와 Code Action의 통합을 통해 보다 효과적인 추론 경로와 코드 행동을 달성하기 위해, PoAct는 동적으로 추론 정책을 전환하고 행동 공간을 수정할 수 있는 기능을 갖추고 있습니다. 특히, PoAct는 고급 계획과 코딩 관점을 강조하는 Policy Controller를 도입하여 특정 Reasoning 단계를 집중적으로 해결할 수 있도록 설계되었습니다.

- **Technical Details**: PoAct는 ReAct Code Agent의 추론 패러다임에 기반하여, 여러 Reasoning 단계에 걸쳐 동적으로 정책을 조정함으로써 특정 단계를 집중적으로 다룰 수 있게 합니다. Action Controller는 행동 공간을 최적화하고 비정상적인 코드 행동을 평가하기 위해 RAG Selector와 Action Reviewer를 사용합니다. 이를 통해 PoAct는 복잡한 문제를 해결하기 위해 여러 도구를 효율적으로 선택하고 관리할 수 있는 가능성을 지닙니다.

- **Performance Highlights**: 실험 결과, PoAct는 여러 과제 데이터셋에서 뛰어난 성능을 보여주며, 일반화와 확장성 측면에서도 높은 수준의 입증을 받았습니다. LegalAgentBench에서 PoAct는 기존 방법 대비 20% 향상된 성능을 기록하면서도 토큰 소비를 감소시키는 성과를 보였습니다. 이러한 성과는 PoAct가 복잡한 작업을 해결하는 데 있어 큰 잠재력을 가진다는 것을 시사합니다.



### Unveiling the Potential of Text in High-Dimensional Time Series Forecasting (https://arxiv.org/abs/2501.07048)
Comments:
          Accepted by NeurIPS24 TSALM Workshop

- **What's New**: 이 논문은 멀티모달 정보, 특히 텍스트 데이터를 통합하여 고차원 시계열 예측의 성능을 향상시키는 새로운 프레임워크인 TextFusionHTS를 제안합니다. 기존의 시계열 예측 방법이 주로 수치적 데이터에 집중했던 반면, 이 연구는 텍스트 정보를 활용함으로써 예측의 정확성을 개선할 수 있음을 보여줍니다. 특히, 이 접근법은 시계열과 텍스트 데이터를 결합한 이중 타워 구조를 기반으로 합니다.

- **Technical Details**: TextFusionHTS는 패치 기반 방법인 PatchTST를 사용하여 시계열 데이터의 표현을 추출하고, 메타 라마-3.1(LLM)을 활용하여 텍스트 설명을 처리합니다. 시계열 데이터를 패치로 나누고 이를 로컬 패턴을 포착하는 데 사용함으로써, 고정된 텍스트 데이터를 각 패치와 동적으로 연관 짓는 방식으로 설계되었습니다. 또한, 텍스트 데이터는 크로스 어텐션 메커니즘을 통해 시계열 표현과 결합됩니다.

- **Performance Highlights**: 실험 결과, 텍스트 정보를 포함하는 것이 고차원 시계열 예측 성능을 유의미하게 개선하는 것으로 나타났습니다. 기존의 방법들과의 비교를 통해, 멀티모달 입력의 잠재력을 강조하며 텍스트가 시계열 예측에서 가지는 중요성을 부각시켰습니다. 이 연구는 멀티모달 시계열 예측에 대한 향후 연구의 길을 열어줍니다.



### A Proposed Large Language Model-Based Smart Search for Archive System (https://arxiv.org/abs/2501.07024)
Comments:
          The 13th International Symposium on Information and Communication Technology (SOICT 2024)

- **What's New**: 이 연구는 Digital Archival Systems에서 정보 검색을 강화하기 위해 Large Language Models (LLMs)의 능력을 활용한 혁신적인 프레임워크를 제시합니다. Retrieval-Augmented Generation (RAG) 접근법을 사용하여 사용자의 자연어 질의를 처리하고 비텍스트 데이터를 의미 있는 텍스트 표현으로 변환합니다. 이 시스템은 고급 메타데이터 생성 기술, 하이브리드 검색 메커니즘, 라우터 쿼리 엔진, 강력한 응답 합성을 통합하여 검색 정확도와 관련성을 검증했습니다.

- **Technical Details**: 제안된 시스템은 RAG 아키텍처에 기반하여 자연어 질의를 입력받고, 지식 베이스에서 정보를 검색한 후 응답을 생성합니다. 다양한 출처에서 수집된 데이터는 동영상, 이미지, 텍스트 등의 다양한 데이터 타입을 포함하며, 모든 비텍스트 데이터는 균일한 텍스트로 변환됩니다. 이 연구에서는 BGE-M3 임베딩 모델을 사용하여 벡터 표현을 생성하고, Pinecone이라는 확장 가능한 벡터 데이터베이스를 활용하여 효율적인 유사성 기반 검색을 수행합니다.

- **Performance Highlights**: 제안된 프레임워크는 기존 접근법보다 유의미한 성능 향상을 보여 주었습니다. 실험을 통해 LLM 효율성, 하이브리드 검색 최적화, 다국어 쿼리 처리 등의 분야에서 성능 개선이 확인되었으며 AI 기반 시스템의 잠재력이 현대 아카이브 실무를 혁신할 수 있음을 입증했습니다. 특히 하이브리드 검색 모델은 BM25와 벡터 기반 접근 방식의 장점을 결합하여 검색 정확도를 높였습니다.



### Enhancing Patient-Centric Communication: Leveraging LLMs to Simulate Patient Perspectives (https://arxiv.org/abs/2501.06964)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)이 특정 분야의 전문가를 시뮬레이션할 수 있는 능력을 평가합니다. 특히, 의료 퇴원 요약을 해석하고 환자에게 제공하는 능력을 시험관찰하며, 저조한 교육 수준의 개인에게서 우수한 성능을 발휘하는 것과 같은 구체적인 발견들을 도출합니다. LLM은 교육 배경 정보와 같은 특정 데이터를 기반으로 정확하고 실행 가능한 의료 지침을 88%의 비율로 전달할 수 있지만, 다른 정보가 주어질 경우 성능이 급격히 떨어짐을 발견했습니다.

- **Technical Details**: 논문에서는 네 가지 인적 속성을 가지고 LLM의 인물 시뮬레이션 일반적 효율성을 평가합니다: 교육 수준, 성별, 의사 방문 빈도, 응급실 방문 빈도입니다. 정보 기반 및 인식 기반 작업의 두 가지 유형을 구분하여 실험을 실행했고, 무작위 기초 정확도와 비교하여 LLM의 평균 정렬율은 54.97%로 무작위 추측보다 우수한 성과를 보였습니다. 교육 수준에 따라 모형 성능이 다르게 나타남을 강조하고 있습니다.

- **Performance Highlights**: LLM은 정보 기반 작업에서 평균 58.38%의 정렬율을 기록하여 더 나은 성능을 보였고, 개인의 교육 수준에 따라 시뮬레이션 결과가 달라져서 높은 교육 수준을 가진 개인을 모사할 때 평균 85.4%의 정렬율을 달성했습니다. 반면 응급실 방문 빈도가 다양한 개인을 시뮬레이션할 때는 40.63%로 성능이 현저히 떨어졌습니다. 전체적으로 LLM의 시뮬레이션 능력은 교육 수준 및 정보의 복잡성에 따라 크게 좌우된다는 점이 도출되었습니다.



### The Einstein Test: Towards a Practical Test of a Machine's Ability to Exhibit Superintelligenc (https://arxiv.org/abs/2501.06948)
- **What's New**: 이번 논문에서는 창의적이고 파격적인 통찰력(Creative and Disruptive Insights, CDI)에 대한 논의를 다루고 있습니다. 저자들은 인공지능(AI)의 최근 발전이 CDI를 생성할 수 있는 모델의 가능성을 여는지를 탐구합니다. 특히, 기계적 초지능(Superintelligence, SI)의 중요한 특징으로 CDI 생성 능력을 강조하고 있습니다.

- **Technical Details**: 논문에서는 AI의 SI 목표에 대한 접근법이 CDI와 같은 새로운 통찰력을 낳을 수 있는지를 평가할 실용적 테스트를 제안합니다. 제안된 테스트는 '아인슈타인 테스트(Einstein test)'로, 이는 알려진 CDI의 출현 전에 이용 가능한 데이터를 바탕으로 AI가 독립적으로 해당 통찰력을 재현할 수 있는지를 평가합니다.

- **Performance Highlights**: 이러한 성과는 기계가 인류의 과거 지적 업적과 동일한 수준에 도달할 수 있음을 의미하며, 궁극적으로 인류를 초월할 가능성까지拥有 할 수 있음을 시사합니다. 따라서, 이 연구는 AI의 잠재력과 초지능에 대한 전반적인 이해를 심화하는 데 기여할 것으로 기대됩니다.



### An Empirical Study of Deep Reinforcement Learning in Continuing Tasks (https://arxiv.org/abs/2501.06937)
- **What's New**: 이번 연구에서는 강화 학습( reinforcement learning)에서 지속적인(task) 작업에 대한 기존의 알고리즘을 분석하고 성과를 평가한 최초의 실증적 연구 결과를 제공합니다. 기존 에피소드 기반(episodic) 방법론이 아닌 새로운 테스트베드(testbeds) 환경에서 DDPG, TD3, SAC, PPO 및 DQN과 같은 여러 잘 알려진 깊이 강화 학습(deep RL) 알고리즘의 성능을 평가했습니다.

- **Technical Details**: 연구는 Mujoco 및 Atari 환경을 기반으로 한 지속적인 작업 테스트베드를 활용하여 에이전트와 환경 간 상호작용의 특성을 분석합니다. 주요 실험은 세 가지 리셋 시나리오(리셋 없음, 사전 정의된 리셋 및 에이전트 제어 리셋)를 고려하여 진행되며, 이는 다양한 환경에서 에이전트가 직면하는 도전 과제를 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 에피소드 기반 테스트베드와 비교할 때 리셋이 없는 작업에서 알고리즘 성능이 현저히 낮아지는 것을 발견했습니다. 또한, 정의된 리셋이 있는 테스트베드에서 더 나은 정책을 학습하는 경향이 있는 것으로 나타났으며, 노출이 많을수록 결과(안전하게 많이 리셋할수록)가 유리하다는 결과를 도출했습니다.



### Risk-Averse Finetuning of Large Language Models (https://arxiv.org/abs/2501.06911)
Comments:
          Neurips 2024

- **What's New**: 이번 연구에서는 큰 언어 모델(LLMs)이 독성이 있는 결과물을 생성하는 문제를 완화하기 위한 새로운 접근법을 제안합니다. 위험 회피(risk-averse) 원칙을 LLM의 미세 조정(fine-tuning)에 통합하여, 독성 출력의 발생을 최소화하는 것을 목표로 하고 있습니다. 특히, 이 방법은 조건부 가치 위험(Conditional Value at Risk, CVaR)이라는 위험 척도를 최적화하여, 긍정적인 언어 생성을 지속적으로 유지하면서 독성 출력을 효과적으로 피할 수 있도록 LLM을 훈련합니다.

- **Technical Details**: 연구팀은 위험 회피 강화 학습(RA-RLHF) 알고리즘을 개발하여 다양한 레벨의 부정성과 독성을 가진 프롬프트를 미세 조정합니다. RLHF는 기대 보상을 극대화하려는 반면, RA-RLHF는 생성된 경로의 위험 척도를 최적화하여 독성을 줄이는 것을 목표로 합니다. 초기 훈련 단계에서는 작은 위험 수준을 설정하여 정책이 성공적인 출력을 생성하도록 훈련한 뒤, 위험 목표에 따라 배치 크기를 줄여가며 높은 위험의 프롬프트에 집중하도록 합니다.

- **Performance Highlights**: RA-RLHF는 세 가지 언어 생성 시나리오에서 평가되었으며, 모두 기존 RLHF 접근법보다 우수한 성능을 나타냈습니다. 실험 결과, RA-RLHF는 위험한 프롬프트에서 특히 높은 효과성을 보였으며, 긍정적인 언어 생성을 성공적으로 이끌어냈습니다. 이 연구의 결과는 LLM이 독성을 줄이면서도 생성 작업의 효율성을 유지할 수 있음을 보여주며, 온라인 담론 환경의 안전성을 높이는데 기여할 것으로 기대됩니다.



### A Foundational Generative Model for Breast Ultrasound Image Analysis (https://arxiv.org/abs/2501.06869)
Comments:
          Peking University; Stanford University; Peking University Cancer Hospital & Institute; Peking Union Medical College Hospital; Cancer Hospital, Chinese Academy of Medical Sciences

- **What's New**: 이번 논문에서는 브레스트 초음파(image analysis) 분석을 위해 특별히 설계된 BUSGen이라는 첫 번째 기초 생성 모델(generative model)을 소개합니다. 기존의 기초 모델들이 여러 임상 작업에 사용되는 것과 달리, BUSGen은 350만 개 이상의 브레스트 초음파 이미지를 기반으로 사전 학습(pretrained)되어 브레스트 구조 및 병리학적 특성에 대한 방대한 지식을 축적했습니다. 이러한 발전은 브레스트 초음파 분석 분야에서의 잠재력을 실현하는 중요한 진전을 나타냅니다.

- **Technical Details**: BUSGen은 few-shot adaptation 방식을 통해 실제적이고 정보가 풍부한 작업 특화 데이터의 저장소를 생성할 수 있습니다. 이를 통해 다양한 하위 작업(downstream tasks)을 위한 모델 개발을 가속화할 수 있습니다. 연구 결과에 따르면, BUSGen은 브레스트 암 검진(screening), 진단(diagnosis), 예후(prognosis) 측면에서 실제 데이터 기반 기초 모델보다 현저히 뛰어난 적응력을 보여주었습니다. 또한, 생성된 데이터의 스케일링 효과(scaling effect)가 수집된 실제 데이터만큼 효과적임을 입증했습니다.

- **Performance Highlights**: 특히, BUSGen은 브레스트 암 조기 진단에서 9명의 인증된 방사선의사(board-certified radiologists)를 초과 성과를 기록하며 평균 민감도(sensitivity)를 16.5% 향상시켰습니다(P-value<0.0001). 또한, 연구를 통해 하위 모델의 일반화 능력(generalization ability)을 개선하며 환자의 프라이버시를 보호하는 Fully de-identified data 공유를 가능하게 했습니다. BUSGen의 온라인 데모는 제공되는 링크를 통해 확인할 수 있습니다.



### What Is a Counterfactual Cause in Action Theories? (https://arxiv.org/abs/2501.06857)
Comments:
          This is an extended report of our short paper accepted at AAMAS 2025

- **What's New**: 이 논문은 Halpern과 Pearl의 실제 인과성(actual causality) 개념을 바탕으로 카운터팩츄얼(counterfactual) 분석을 통해 새로운 인과성 개념을 제안합니다. 최근 Batusov와 Soutchanski가 상황 계산법(situation calculus)에서 Achievement Cause의 개념을 도입한 것과 관련하여, 카운터팩츄얼 분석을 활용한 인과성의 일반화 가능성에 대해 논의합니다. 이 논문은 다양한 시나리오에서 Achievement Cause를 분석하여 기존 이론과 비교하고, Halpern과 Pearl의 접근 방식과의 관계를 밝힙니다.

- **Technical Details**: 논문에서는 상황 계산법의 모달 변형을 사용하여 객체 및 행동에 대한 변수와 표준 이름을 정의합니다. 이를 통해 기저 행동 이론을 표현하고, 행동에 대한 회귀(regression) 및 진전(progress)에 대한 추론을 가능하게 합니다. 또한, action과 object 표준 이름의 집합을 통해 행동의 동기 및 변화를 체계적으로 표현하는 방법을 제공합니다.

- **Performance Highlights**: 최종적으로 이 논문은 실제 인과성에 대한 간단한 정의를 제안하며, 다양한 시나리오에서 이 정의를 적용하는 과정을 보여줍니다. 특히, disjunctive goals와 같은 경쟁 사건을 올바르게 모델링하는 방법에 대해 설명하고, 이 기존 접근 방식에서의 한계를 강조합니다. 이 연구는 GOLOG와 같은 로봇 프로그래밍 및 향후 행동 언어에 기반한 다양한 응용 프로그램에 중요한 기초를 제공할 것으로 기대됩니다.



### An efficient approach to represent enterprise web application structure using Large Language Model in the service of Intelligent Quality Engineering (https://arxiv.org/abs/2501.06837)
Comments:
          16 pages, 1 figure and 4 tables, relevant for Gen AI and enterprise AI use cases

- **What's New**: 이 연구는 대규모 지능형 품질 공학을 가능하게 하기 위해 대형 언어 모델(LLMs)을 사용하는 새로운 접근 방식을 제안합니다. 특히 복잡한 웹 애플리케이션 구조의 계층적 표현 방법론을 도입함으로써, 테스트 케이스 생성을 최적화할 수 있는 방법을 제시하고 있습니다. 기존의 Generative AI를 이용한 자동화된 소프트웨어 테스트의 한계를 극복하기 위해 LLM이 웹 애플리케이션 아키텍처를 이해할 수 있도록 구성된 구조체계를 개발하였습니다.

- **Technical Details**: 이 연구의 방법론은 다섯 가지 주요 단계로 구성됩니다: 종합적인 DOM 분석(DOM analysis), 다중 페이지 합성(multi-page synthesis), 테스트 스위트 생성(test suite generation), 실행(execution) 및 결과 분석(result analysis)입니다. 두 개의 웹 애플리케이션(스와그랩스, 메디박스)을 대상으로 실험을 수행했으며, 각각 자동화 테스트 성공률이 90%와 70%로 나타났습니다. 이 방법은 복잡한 관계와 상호작용을 고려하여 LLM을 통해 점진적이고 동적인 애플리케이션 구조의 표현을 가능하게 합니다.

- **Performance Highlights**: 최종 결과는 제안된 접근 방식이 LLM의 맥락에 적합한 테스트 케이스 생성을 위한 능력을 크게 향상시킴을 보여줍니다. 다양한 평가 기준에 걸쳐 높은 관련 점수를 기록했으며, 품질 보증 과정을 전반적으로 개선하는 데 기여하고 있습니다. 또한 테스트 소요 시간을 줄이는 데에도 긍정적인 영향을 미치는 것으로 나타났습니다.



### LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents (https://arxiv.org/abs/2501.06834)
- **What's New**: 이 연구는 기존 WEIRD(서구, 교육받고, 산업화되고, 부유하고, 민주적인) 집단이 아닌 다양한 집단의 경제 행동을 연구하는 데 새로운 접근법을 제시합니다. 이 방법론은 대형 언어모델(LLM)을 활용하여 Synthetic Cultural Agents(SCAs)를 생성함으로써 다양한 문화 집단의 행동을 시뮬레이션합니다. 이러한 SCAs는 전통적인 경제 실험에 참여하여 문화간 행동 변화를 분석할 수 있도록 돕습니다.

- **Technical Details**: 방법론의 핵심 기초는 맞춤화와 재현 가능성에 중점을 두고 있습니다. SCAs는 소규모 사회의 문화적 특성을 반영한 프로필에 따라 행동하도록 훈련됩니다. 연구진은 직접 프롬프트, 셀프 질의 및 검색+회수 증강 생성(RAG) 방법을 통해 SCAs의 문화 프로필을 구축하였으며, 이 과정을 통해 경제 결정에 대한 실험을 수행했습니다.

- **Performance Highlights**: 연구 결과, SCAs의 행동은 전통적인 경제 실험에서 실존 인구의 행동 및 패턴과 질적으로 유사함을 보여주었고, 문화 간 행동의 다양성을 나타냈습니다. SCAs는 이전에 연구되지 않은 집단에서도 경제 행동에 대한 새로운 가설 생성을 가능하게 하며, 실험 경제학 및 문화 인류학 분야에서 AI를 통합하는 새로운 도구로 기능할 잠재력을 증명합니다.



### Leveraging Taxonomy and LLMs for Improved Multimodal Hierarchical Classification (https://arxiv.org/abs/2501.06827)
Comments:
          11 pages, 7 figures, 2 tables, and accepted by COLING 2025

- **What's New**: 이 논문에서는 다중 수준 계층 분류(Multi-level Hierarchical Classification, MLHC)를 위한 새로운 프레임워크인 Taxonomy-based Transitional Classifier (TTC)를 제안합니다. 기존의 MLHC 방법이 클래스 간의 계층적 관계를 무시하는 문제를 해결하기 위해, 우리는 대형 언어 모델(Large Language Models, LLM)을 이용하여 계층적 일관성을 강화하는 방법을 소개했습니다. 이 접근 방식은 MEP-3M 데이터셋에서 기존 LLM 구조와 비교하여 성능 향상을 보여주었습니다.

- **Technical Details**: TTC는 LLM에 독립적인 출력 계층을 통해 작동하며, 각 클래스의 계층적 관계를 모델링하여 정책을 부여합니다. 이 방식은 계층의 각 레벨에서 분류 결과가 일관되도록 보장하며, 이를 통해 불확실성을 줄이고 예측의 일관성을 높입니다. 실험은 다양한 LLM을 기반으로 진행되었으며, TTC가 평면 분류자(flat classifier)로서 작용할 때와 비교하여 성능을 개선하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, TTC를 적용한 MLHC 접근 방식이 MEP-3M 데이터셋에서 이전 방식보다 뛰어난 성능을 보였습니다. 특히, 잘못 분류된 이미지의 비율이 낮아지고, 각 레벨에서 정확히 분류된 이미지의 비율이 증가했습니다. 이러한 성과는 MLHC 내에서 계층 구조의 중요성을 강조하며, 향후 다양한 분야에서의 적용 가능성을 제시합니다.



### A Study on Educational Data Analysis and Personalized Feedback Report Generation Based on Tags and ChatGP (https://arxiv.org/abs/2501.06819)
- **What's New**: 이번 연구는 Tag annotation(태그 주석)과 ChatGPT 언어 모델을 결합하여 학생 학습 행동을 분석하고 개인화된 피드백을 생성하는 새로운 방법론을 제안합니다. 이 접근법의 핵심은 복잡한 학생 데이터를 태그로 변환하여 맞춤화된 프롬프트를 통해 건설적인 피드백을 제공하는 것입니다. 이를 통해 학생들은 긍정적인 학습 동기를 부여받으며, 교육적 피드백의 효율성과 성과가 향상됩니다.

- **Technical Details**: 이 연구는 상하이의 한 초등학교에서 실시된 적응형 학습 시스템으로부터 데이터를 수집하였습니다. 여기에는 정확성, 난이도 수준, 지식 카테고리, 능력 수준 및 과제 완료 시간을 포함한 다양한 성과 기록이 포함되었습니다. 수집된 데이터를 태그화하여 LLM이 보다 정교한 피드백을 생성할 수 있도록 돕는 방법론이 새롭게 제시되었습니다.

- **Performance Highlights**: 20명이 넘는 수학 교사들을 대상으로 한 설문 조사를 통해 이 방법론의 효과가 검증되었습니다. 연구 결과에 따르면, 변환된 데이터와 태그 기반 피드백을 통해 학생들에게 보다 정확하고 시의적절한 피드백을 제공할 수 있으며, 이는 교사의 업무 부담을 크게 경감시킬 수 있습니다. 이 방법은 지능형 적응 학습 시스템에 원활하게 통합될 수 있으며, 학생의 개별 요구에 맞춘 맞춤형 학습 피드백을 지원합니다.



### Unifying Two Types of Scaling Laws from the Perspective of Conditional Kolmogorov Complexity (https://arxiv.org/abs/2501.06802)
- **What's New**: OpenAI는 2020년에 제안된 Scaling Laws를 통해 모델 성능과 파라미터, 데이터, 컴퓨팅 리소스 간의 관계를 설명했습니다. 2024년에는 모델 추론 성능과 추론 계산 간의 관계를 다룬 두 번째 Scaling Laws를 제안하였습니다. 본 논문에서는 손실 없는 압축을 활용하여 LLM(대규모 언어 모델)의 훈련 및 추론 과정을 분석하고, 두 가지 Scaling Laws를 통합했습니다.

- **Technical Details**: 우리는 조건부 Kolmogorov 복잡도를 사용하여 LLM 훈련 과정과 추론 메커니즘을 체계적으로 조사하였습니다. LLM의 훈련 과정은 압축할 데이터 스트림을 기준으로 훈련되고, 이 과정에서는 모델 파라미터를 전송할 필요가 없습니다. 또한, 저자들은 Kolmogorov 복잡도를 기반으로 훈련된 대규모 언어 모델이 총 재귀 함수 또는 트랜스포머 신경망에 의해 근사 될 수 있으며, 결국 일반 Kolmogorov 복잡도 상한의 computable 근사임을 증명했습니다.

- **Performance Highlights**: 두 가지 Scaling Laws 모두 실행 단계 t를 증가시킴으로써 조건부 Kolmogorov 복잡도의 근사값을 개선합니다. 첫 번째 Scaling Laws는 모델 파라미터 y를 증가시키고, 두 번째 Scaling Laws는 출력 토큰 수를 증가시킵니다. 이 모델들은 최적화 과정에서 규제 효과를 자연스럽게 발생시키며, 이로 인해 LLM의 추론 과정에서 이론적 한계가 존재함을 보여줍니다.



### Eliza: A Web3 friendly AI Agent Operating System (https://arxiv.org/abs/2501.06781)
Comments:
          20 pages, 5 figures

- **What's New**: Eliza는 사용자 지시에 따라 자율적으로 실행 경로를 결정하고 제어할 수 있는 AI 에이전트 시스템으로, 최초의 오픈소스 웹3 친화적 에이전틱 프레임워크를 제안합니다. Eliza의 모든 구성 요소는 일반적인 Typescript 프로그램으로 제작되어 사용자가 완전한 제어를 유지합니다. 또한, 이 프레임워크는 웹3 기능과 매끄럽게 통합되어 블록체인 데이터 읽기 및 쓰기, 스마트 계약과의 상호작용 등을 지원합니다.

- **Technical Details**: Eliza는 사용자 친화적인 에이전트 프로그램을 개발할 수 있도록 설계된 다중 에이전트 시뮬레이션 프레임워크입니다. 이 프레임워크의 구조는 핵심 런타임과 네 가지 주요 구성 요소로 분리되며, 이를 통해 개발자나 사용자는 플러그인, 클라이언트, 캐릭터 및 어댑터를 자유롭게 추가할 수 있습니다. 이러한 모듈식 설계는 OpenAI, Solana, Ethereum 등 다양한 플랫폼 통합성과 높은 기능성을 지원합니다.

- **Performance Highlights**: Eliza는 웹3와 AI의 통합에서 발생하는 여러 도전 과제를 해결하며, 그 성능의 안정성을 강조합니다. 이 시스템은 개발자와 사용자가 특정 요구에 맞게 솔루션을 맞춤화할 수 있는 모듈 구성으로 다양한 기능을 쉽게 구현할 수 있도록 도와줍니다. 결국, Eliza는 정보 과부하를 피하고 소셜 미디어 상호작용에서 유용한 정보를 추출하여 사용자의 요구를 충족시키는 에이전트로 자리 잡을 것입니다.



### On the Complexity of Global Necessary Reasons to Explain Classification (https://arxiv.org/abs/2501.06766)
- **What's New**: 이 논문에서는 Explainable AI의 일환으로 global explanations에 중점을 두고 있습니다. 기존의 연구들이 local explanations에 초점을 맞출 때, 본 연구는 분류기가 특정 클래스를 할당하기 위해 필요한 최소 조건(minimal necessary conditions)을 설명하는 데 집중하고 있습니다. 이를 통해 사용자는 AI의 의사결정 과정에 대한 깊은 이해를 얻을 수 있습니다.

- **Technical Details**: 논문에서는 classification에 대한 다양한 접근 방식과 자연 최소성 기준(natural minimality criteria)에 대한 철저한 복잡성 분석(complexity analysis)을 수행합니다. 중요한 분류기(families of classifiers) 집합에 대해서도 분석이 이루어지며, 이로 인해 AI 시스템의 전체적인 동작 방식을 파악할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 이 연구는 Explainable AI의 중요한 문제인 global explanations을 새로운 관점에서 제시하였으며, 이해하기 쉬운 방식으로 classification의 기본 원리를 설명합니다. 최소 조건을 기반으로 한 이러한 접근은 다양한 산업 분야에서 AI의 신뢰성을 높이는 데 기여할 수 있습니다.



### MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation (https://arxiv.org/abs/2501.06713)
- **What's New**: MiniRAG는 Small Language Models (SLMs)를 위한 효율적이고 간소한 Retrieval-Augmented Generation (RAG) 시스템으로 설계되었으며, 기존 RAG 프레임워크의 가장 중요한 한계를 해결하여 실질적인 성능을 제공한다. 이 시스템은 복잡한 의미 이해에 대한 의존도를 줄이기 위해 세 가지 핵심 통찰력에 기반하여 아키텍처를 재구성하였다. 또한, MiniRAG는 그 구조 내에서 텍스트 청크와 명명된 개체를 통합하는 새로운 세멘틱 인식 이종 그래프 색인 메커니즘과 경량화된 지식 검색 접근 방식을 도입하였다.

- **Technical Details**: MiniRAG의 아키텍처는 두 가지 핵심 구성 요소로 이루어진다. 첫째로, 이종 그래프 색인(heterogeneous graph indexing)은 텍스트 청크와 명명된 개체를 통합하여 세멘틱 인식 지식 표현을 생성한다. 둘째로, 경량화된 그래프 기반 정보 검색은 논리적인 정보 검색을 가능하게 하여 작은 모델에서도 효율적으로 수행될 수 있도록 한다. 이 시스템은 SLM의 한계에도 불구하고 효과적으로 정보 검색을 수행할 수 있는 능력을 제공한다.

- **Performance Highlights**: MiniRAG는 기존의 경량 RAG 시스템에 비해 1.3-2.5배 높은 효율성을 달성하면서도 저장 공간의 25%만을 요구한다. 다양한 데이터 세트와 SLM에서의 광범위한 실험을 통해, MiniRAG는 LLM에서 SLM으로 전환할 때도 주목할 만한 안정성을 유지하며, 각기 다른 시나리오에서의 정확도 하락은 0.8%에서 20%에 불과하다. 또한, MiniRAG는 자원 제약 환경에서도 우수한 성능을 지속적으로 발휘하여 최신 성능 기준을 유지한다.



### ELIZA Reanimated: The world's first chatbot restored on the world's first time sharing system (https://arxiv.org/abs/2501.06707)
Comments:
          In review

- **What's New**: 이 논문은 1960년대 초기 MIT에서 개발된 세계 최초의 챗봇 ELIZA의 원본 인쇄물을 발견했다는 내용을 다루고 있습니다. ELIZA는 MAD-SLIP 언어로 개발되었으며 CTSS에서 실행됩니다. 또한, 원본 ELIZA를 현대의 에뮬레이터를 통해 복원한 내용을 소개하고 있습니다.

- **Technical Details**: ELIZA는 Joseph Weizenbaum이 개발한 챗봇으로, CTSS(Compatible Time-Sharing System)에서 MAD-SLIP언어로 작성되었습니다. 논문에서는 ELIZA의 복원 과정과 원본 코드를 오픈 소스로 공개한 점을 강조하며, 유닉스 계열 운영 체제에서 실행할 수 있도록 만든 기술적 세부사항을 다룹니다. 단순한 텍스트 상호작용을 넘어서 인공지능의 초기 개념을 탐구한 실험으로 기능했습니다.

- **Performance Highlights**: ELIZA는 인공지능의 시초로, 튜링 테스트를 구현한 최초의 프로그램으로 평가받습니다. 또한, ELIZA는 인간-컴퓨터 상호작용의 과학 소설적 상상을 현실로 가져왔다는 점에서 중요한 역사적 의미가 있습니다. 이후 ELIZA는 ARPAnet을 통해 빠르게 전파되어 인공지능 연구의 지식 기반으로 자리 잡았습니다.



### AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds (https://arxiv.org/abs/2501.06706)
- **What's New**: 이 논문은 AIOps(IT 운영을 위한 인공지능)와 대규모 언어 모델(LLM) 에이전트의 조합을 통해, 자율적으로 운영 작업을 관리하고 자동화하는 새로운 패러다임인 AgentOps를 제안합니다. 이는 기존의 DevOps 도구와 AIOps 알고리즘의 한계를 극복하고, 전체 사건 수명 주기 동안 에이전트가 독립적으로 수행할 수 있는 프로세스를 정의합니다. 또한 AIOpsLab이라는 포괄적인 프레임워크를 통해 복잡한 운영 작업의 평가를 실현하는 방안을 제시합니다.

- **Technical Details**: AIOpsLab은 마이크로서비스 클라우드 환경의 배포, 결함 주입, 작업 부하 생성 및 텔레메트리 데이터 내보내기와 같은 여러 컴포넌트를 조율하고, 에이전트와 상호작용할 수 있는 인터페이스를 제공하는 혁신적 프레임워크입니다. 이 프레임워크는 에이전트-클라우드 인터페이스(ACI)를 포함하고 있어 에이전트가 클라우드와 동적 상호작용을 할 수 있도록 도와줍니다. 이를 통해 전체 평가 프로세스를 자동으로 관리하고, LLM 기반 AIOps 에이전트를 평가하기 위한 48개의 테스트 문제를 생성했습니다.

- **Performance Highlights**: 연구를 통해 AIOpsLab의 통합적인 문제 풀이를 기반으로 다양한 AIOps 작업의 성능을 평가하였으며, 각 에이전트가 직면하는 독특한 도전 과제를 도출하였습니다. AIOpsLab을 통해 LLM 기반 에이전트들의 능력과 한계를 분석함으로써 복잡한 운영 작업 처리 과정에서의 새로운 통찰을 제공하고 있습니다. 이 연구 결과는 AIOps 에이전트의 설계와 개발에 대한 방향성을 제시하고, 향후 공공에 공유될 예정입니다.



### Fine-tuning ChatGPT for Automatic Scoring of Written Scientific Explanations in Chines (https://arxiv.org/abs/2501.06704)
- **What's New**: 이 연구는 중학생들이 작성한 중국어 과학 설명 텍스트의 자동 점수를 매기기 위해 ChatGPT와 같은 대규모 언어 모델(LLM)을 조정하는 방법을 탐구합니다. 이 연구는 LLM이 글쓰기의 복잡성에 따라 점수 정확도가 어떻게 달라지는지를 살펴보았으며, 특히 중국어와 같은 로그그래픽 언어에서의 가능성을 주목했습니다. 결과적으로, ChatGPT가 중국어 과학 설명을 점수화할 수 있는 능력이 있음을 보여줍니다.

- **Technical Details**: 자동 점수화는 증거 기반의 주장 구축과 의사소통을 포함하는 과학적 설명에 있어서 중요한 기술로, 이를 위해 기계학습(machine learning)과 자연어 처리(natural language processing) 기술이 활용됩니다. 이 연구에서는 7626개의 중학생의 답변을 수집하여 ChatGPT를 조정하였으며, 점수 정확도는 학생의 이해 수준에 따라 다르게 나타났습니다. 이는 글의 언어적 특징에 따라 반영되는 점수화의 복잡성을 시사합니다.

- **Performance Highlights**: 연구 결과, ChatGPT는 중국어 과학 설명 점수화에 있어 높은 정확도를 보였으나, 학생의 글쓰기 복잡성과 점수 정확도 간에 상반된 상관관계가 나타났습니다. 저급 반응에서는 복잡한 문장이 과대 평가되는 경향이 있었고, 고급 반응에서는 간결한 인과관계 서술이 과소 평가되는 양상이 관찰되었습니다. 이러한 사항은 글쓰기의 단순성과 명료성이 저급 반응에서는 정확성을 높이는 반면, 포괄성이 고급 반응에서는 정확성을 높이는 데 기여한다는 것을 보여줍니다.



### Large Language Models, Knowledge Graphs and Search Engines: A Crossroads for Answering Users' Questions (https://arxiv.org/abs/2501.06699)
- **What's New**: 이번 논문은 대규모 언어 모델(LLMs), 지식 그래프(knowledge graphs), 검색 엔진(search engines) 간의 시너지 가능성을 다양한 사용자 정보 수요를 반영하여 분석합니다. 특히 사용자 중심의 관점에서 이들 기술의 장단점을 비교하고, 이들 기술이 정보 제공에 어떻게 기여할 수 있는지를 제시합니다. 이 연구는 결국 미래 연구의 로드맵을 정립하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 대규모 언어 모델은 변환기(Transformer) 아키텍처를 기반으로 하여 훈련된 인공지능 모델이며, 이들은 자연어 처리 및 다양한 작업에서 효율성을 보여줍니다. 하지만, 이러한 LLM들은 사실 확인 오류, 불투명성(opaqueness), 정보의 최신성(staleness) 등의 한계를 가지며, 이를 극복하기 위해 검색 엔진과 결합된 정보 탐색 방안인 Retrieval Augmented Generation(RAG) 기술이 주목받고 있습니다. 또한 지식 그래프(KGs)는 데이터와 지식을 구조화하여 질의하고 추론할 수 있도록 돕습니다.

- **Performance Highlights**: 논문에서는 LLMs이 때때로 사용자 질문에 대해 부정확하거나 불완전한 답변을 제공할 수 있음을 보여주는 사례를 들며, 이들 기술이 어떻게 서로를 보완할 수 있는지를 설명합니다. 예를 들어, LLM이 ACM Fellow에 대한 질문에 오류를 반환한 사례를 통해, 특히 긴 꼬리(long-tail) 정보에서의 한계를 강조하고 있습니다. 이에 따라 사용자 정보 요구를 충족시키기 위한 연구 방향성도 제시하고 있습니다.



### DVM: Towards Controllable LLM Agents in Social Deduction Games (https://arxiv.org/abs/2501.06695)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 논문은 DVM(Dynamic Victory Manager)이라는 새로운 프레임워크를 제안하여, 사회적 추론 게임(SDGs)에서 LLM(대규모 언어 모델) 에이전트의 성능을 제어할 수 있는 방법을 제공합니다. DVM은 게임의 난이도에 따라 NPC(비플레이어 캐릭터)가 적응할 수 있도록 하여 공정성과 안전성에 대한 통찰을 제공합니다. 이 프레임워크는 예측자(Predictor), 결정자(Decider), 토론자(Discussor)라는 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: DVM은 강화 학습(reinforcement learning)과 승률 제한(decision chain reward) 메커니즘을 통합하여 에이전트가 동적으로 게임 플레이 능력을 조정할 수 있도록 설계되었습니다. Predictor는 플레이어 간의 상호작용을 분석하여 관계를 이해하며, Decider는 추론된 정보를 바탕으로 전략적 결정을 내립니다. Discussor는 게임 상황에 맞는 대화를 생성하여 다른 플레이어에게 영향을 미치며, 이 모든 과정에서 PPO 알고리즘을 사용하여 학습합니다.

- **Performance Highlights**: DVM의 실험 결과, 이 방법은 웨어울프 게임에서 기존 방법들보다 뛰어난 성능을 보였으며, 미리 설정된 승률 목표를 달성하기 위해 성능 수준을 성공적으로 조절하였습니다. 이러한 성과는 SDGs에서 LLM 에이전트의 적응적이고 균형 잡힌 게임플레이를 가능하게 하며, 게임 에이전트 연구의 새로운 방향을 제시합니다.



### Generative AI in Education: From Foundational Insights to the Socratic Playground for Learning (https://arxiv.org/abs/2501.06682)
- **What's New**: 이번 논문은 인간의 인지와 대형 언어 모델(LLMs) 사이의 시너지를 탐구합니다. 생성적 AI가 개인화된 학습을 효율적으로 추진할 수 있는 가능성을 제시하며, LLM과 인간 인지의 유사성을 강조합니다. 또한 기존의 Intelligent Tutoring Systems(ITS)의 한계와 성공 사례를 검토한 후, 새로운 차세대 시스템인 Socratic Playground를 소개합니다.

- **Technical Details**: 논문은 LLM의 인지 기초와 이를 통해 생성적 AI가 인간의 성능을 초월할 수 있는 방법을 다룹니다. NEOLAF와 같은 신경 기호 인지 아키텍처는 상징적 추론과 신경 학습을 통합하여 인간 인지를 모방하고, LLM이 자기 개선 에이전트로 작동하도록 합니다. 이 기술은 교육 프레임워크에 통합하여 인간의 적응력을 모방한 개인화된 학습 경험을 가능하게 합니다.

- **Performance Highlights**: Generative AI, 특히 OpenAI의 o3 모델은 깊은 추론이 필요한 과제에서 인상적인 성과를 보여 줍니다. 2024 American Invitational Mathematics Exam에서 96.7%의 점수를 기록하며, 물리학, 화학, 생물학 분야에서 박사 과정에 준하는 전문성을 지닌 것으로 평가받습니다. 이러한 발전은 교육 분야의 혁신적인 응용 가능성을 제시하며, 교육 공정성을 높일 수 있는 잠재력을 지니고 있습니다.



### Common Sense Is All You Need (https://arxiv.org/abs/2501.06642)
- **What's New**: 이번 논문에서는 인공지능(AI) 시스템이 진정한 자율성을 달성하기 위해 공통 감각(common sense)을 통합하는 것이 필수적이라고 주장합니다. 기계학습과 깊은 학습(deep learning) 기술이 발전하였음에도 불구하고, 현재의 AI는 동물들이 가지고 있는 기본적인 지능과 유연한 사고(Adaptive Reasoning) 능력이 미비하다는 점을 강조합니다. 특히 새로운 문제와 상황에 대응하는 능력을 발전시키기 위한 AI 소프트웨어 설계 전반에 대한 재고가 필요합니다.

- **Technical Details**: AI에서의 공통 감각은 시스템이 세계를 이해하고 적응하며 사고할 수 있는 능력을 의미합니다.  이 논문은 공통 감각의 주요 구성 요소로 맥락적 학습(Contextual Learning), 적응적 추론(Adaptive Reasoning), 체화 인지(Embodied Cognition) 및 최소한의 선행 지식(Tubula Rasa)에서 시작하는 문제 해결 능력을 정의합니다. 각 구성 요소를 통해 AI 시스템은 복잡하고 동적인 환경에서 효과적으로 기능할 수 있습니다.

- **Performance Highlights**: 현재 중점적으로 사용되고 있는 벤치마크와 테스트가 AI 시스템의 공통 감각을 적절하게 측정하고 발전시키지 못하고 있다는 점을 지적합니다. AI는 기계학습을 통해 애매한 환경에서도 적응하고 자율적으로 작동할 수 있으며, 이는 실질적으로 인간과 동물의 지능에 부합하는 성능을 요구합니다. 이러한 발전을 달성하기 위해, 기존 문제의 접근 방식을 변화시키고 공통 감각을 우선시하는 새로운 설계 접근법을 제안합니다.



### Quantifying Relational Exploration in Cultural Heritage Knowledge Graphs with LLMs: A Neuro-Symbolic Approach (https://arxiv.org/abs/2501.06628)
- **What's New**: 이 논문은 문화유산 지식 그래프에서 관계 탐색을 위한 신경-기호적 접근법을 소개합니다. 이 방법은 설명 생성을 위해 Large Language Models (LLMs)을 활용하고, 관계의 흥미로움을 정량화하기 위한 새로운 수학적 프레임워크를 제공합니다. 특히, 흥미로움의 측정이 시스템의 전반적인 성능에 미치는 영향을 수치 분석을 통해 강조합니다.

- **Technical Details**: Wikidata Cultural Heritage Linked Open Data (WCH-LOD) 데이터세트를 사용하여 제안된 접근 방식은 정밀도(precision) 0.70, 재현율(recall) 0.68, F1-score 0.69를 달성했습니다. 이는 그래프 기반(precision: 0.28, recall: 0.25, F1-score: 0.26) 및 지식 기반(baseline) (precision: 0.45, recall: 0.42, F1-score: 0.43) 방법에 비해 개선된 성과입니다. 흥미로움의 측정과 생성된 설명 품질 간의 강한 상관관계(0.65)를 보여줍니다.

- **Performance Highlights**: 제안된 시스템은 BLEU (0.52), ROUGE-L (0.58), METEOR (0.63) 점수에서 기초 방법보다 좋은 품질의 설명을 생성합니다. 이 결과는 LLM과 흥미로움의 수학적 형식화가 문화유산 지식 그래프에서의 관계 탐색의 효과를 높이는 데 중요함을 드러냅니다. 우리 연구는 전통적인 지식 기반 및 그래프 기반 접근 방식에 비해 더 효과적인 탐색을 가능케 함을 보여줍니다.



### Guided Code Generation with LLMs: A Multi-Agent Framework for Complex Code Tasks (https://arxiv.org/abs/2501.06625)
Comments:
          4 pages, 3 figures

- **What's New**: 이 논문은 기존의 Code Generation 시스템이 가진 한계를 극복하기 위한 새로운 'guided code generation' 프레임워크를 소개합니다. 특히, 이 프레임워크는 LLM의 장점을 활용하여 복잡한 코드 생성 작업에 대한 구조적이고 세분화된 접근 방식을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 복잡한 코딩 작업을 관리 가능한 유닛으로 분해하기 위해 계층적 분해, 바텀업 코드 생성 및 다중 에이전트 검증 시스템으로 구성됩니다. 특히, 'Generalist Agent'가 문제를 재귀적으로 분해하고, 에이전트 간의 피드백 루프를 통해 코드 솔루션을 검증하는 구조를 갖고 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 방법론이 OpenAI의 HumanEval 기준에서 23.79%의 정확도 향상을 달성했으며, 이는 LLM의 구성을 고려한 더 나은 코드 생성 결과를 보여줍니다. 이러한 결과는 복잡한 소프트웨어 개발에서의 LLM의 활용 가능성을 크게 높일 수 있음을 시사합니다.



### ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation (https://arxiv.org/abs/2501.06598)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 차트 이해 능력을 발전시키기 위해 ChartCoder를 제안합니다. ChartCoder는 코드 LLM을 언어 백본으로 활용하여 코드 실행 가능성을 향상시키고, Chart2Code-160k 데이터셋을 통해 다양한 차트 유형에 대한 대량 데이터를 제공합니다. 또한 Snippet-of-Thought(SoT) 방법론을 도입하여 직접적인 차트-코드 생성 데이터를 단계별 생성으로 변환합니다.

- **Technical Details**: ChartCoder는 차트-코드 생성 데이터의 부족 문제를 해결하기 위해 Chart2Code-160k라는 개념을 제시하며, 이 데이터셋은 27개 차트 유형의 160,000개의 차트-코드 쌍으로 구성되어 있습니다. SoT 방법론은 차트의 중요한 정보를 강조하고, 단계를 기반으로 한 생성 프로세스를 최적화합니다. 실험 결과, ChartCoder는 단 7B 파라미터로 기존 오픈 소스 MLLMs보다 뛰어난 성능을 보입니다.

- **Performance Highlights**: ChartCoder는 다양한 차트-코드 벤치마크에서 뛰어난 차트 복원 및 코드 실행 가능성을 달성하며, 코드 생성의 정확성에서도 큰 성과를 보입니다. 본 논문의 실험 결과는 기존 방법보다 더 높은 정확도로 올바르고 실행 가능한 코드를 생성하는 능력을 입증합니다. 이러한 성과는 ChartCoder가 차트-코드 생성 분야에서의 혁신적인 접근법임을 보여줍니다.



### Transforming Social Science Research with Transfer Learning: Social Science Survey Data Integration with AI (https://arxiv.org/abs/2501.06577)
Comments:
          22 pages, 5 figures, Presented and Submitted to SPSA 2025 (Political Methodology Panel)

- **What's New**: 이 논문은 사회 과학 조사 연구에서의 데이터 통합을 위한 새로운 연구 의제를 제안합니다. 특히, transfer learning (TL)을 활용하여 대규모 조사 데이터 세트를 전이하는 가능성을 강조합니다. 기존 언어 모델에 의한 대체 대신, 실제 사회 과학 데이터를 기반으로 한 데이터 통합의 필요성을 주장합니다. 이는 사회과학 연구에서 기존 데이터의 유용성을 극대화하려는 노력의 일환입니다.

- **Technical Details**: transfer learning은 대규모 데이터 세트에서 사전 훈련된 모델을 소규모 데이터 세트에 맞춰 조정하여 성능을 최적화하는 기법입니다. 본 연구는 ANES와 CES와 같은 대규모 조사를 연결하여 미국 정치 행태에 대한 새로운 가설을 테스트할 수 있는 독창적인 기회를 창출하는 방법을 제시합니다. 이러한 방법은 비슷한 변수를 공유하지만 결과 변수는 다른 다양한 조사 데이터 세트의 통합을 가능하게 합니다.

- **Performance Highlights**: 초기 데이터 구조에서 기반 모델을 수정하여 92%에 가까운 정확도로 결측 변수를 예측하는 데 성공했습니다. 특히, 이 접근 방식은 정치 과학 연구자들이 자주 직면하는 데이터의 한계를 극복하는 데 도움을 줄 것으로 기대됩니다. 따라서, transfer learning의 적용은 기존 조사 데이터의 활용을 극대화할 수 있는 잠재력을 지니고 있습니다.



### Where to Go Next Day: Multi-scale Spatial-Temporal Decoupled Model for Mid-term Human Mobility Prediction (https://arxiv.org/abs/2501.06561)
- **What's New**: 이 연구에서는 인간의 이동 예측을 위해 중기 예측(medium-term prediction) 모델인 Multi-scale Spatial-Temporal Decoupled Predictor (MSTDP)를 제안합니다. MSTDP는 일일 이동 패턴을 효율적으로 캡처하고, 다음 날 또는 주의 이동 경로를 예측하는 것을 목표로 합니다. 이 모델은 이동 데이터를 위치-기간의 분리된 체인으로 변환하여 정체된 관찰을 제거하고, 계층적 인코더를 통해 다중 시간 패턴을 모델링합니다.

- **Technical Details**: MSTDP의 핵심은 두 가지 주요 요소인 공간-시간의 분리(decoupling)와 다중 시간 패턴 검색입니다. 사용자 일일 여행을 서로 분리된 위치 및 기간 체인으로 나누고, 이를 통해 매일 반복되는 위치를 효과적으로 처리합니다. 또한, 트랜스포머(Transformer) 기반의 디코더를 통해 예측된 결과를 전 세계적으로 연관짓고, 다양한 스케일의 공간적 관계를 캡처할 수 있는 이질 그래프 학습기(heterogeneous graph learner)를 도입하여 위치의 의미론적 표현을 강화합니다.

- **Performance Highlights**: MSTDP는 5개 도시에서 수집된 대규모 모바일 데이터셋을 통해 효과성을 검증하였으며, 특히 보스턴의 전염병 모델링에 적용했을 때 기존 모델보다 MAE(Mean Absolute Error)를 62.8% 줄이는 성과를 보였습니다. 이러한 결과는 MSTDP가 중기 이동 예측 및 전염병 전파 분석에 있어 탁월한 예측 능력을 가지고 있음을 입증합니다.



### Scaffolding Creativity: Integrating Generative AI Tools and Real-world Experiences in Business Education (https://arxiv.org/abs/2501.06527)
- **What's New**: 이 사례 연구에서는 비즈니스 교육에서 Generative AI 도구와 실제 경험의 통합을 탐구합니다. 혁신적인 학부 과정의 연구를 통해 AI 지원 학습이 학생들의 창의적 과정과 학습 결과에 미치는 영향을 조사하였습니다. 연구 결과, 이 통합 접근법이 지식 습득을 가속화하고 전통적인 창의적 장벽을 극복하게 하며 AI 생성 인사이트와 실제 관찰 간의 역동적인 상호작용을 촉진하는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 AI 도구를 비즈니스 교육에 통합하는 다단계 혼합 방법론(mixed-methods approach)을 사용하였으며, 학생들이 AI 지원 학습 및 실제 경험을 결합한 혁신적인 학부 과정에서의 학습 과정을 분석합니다. 데이터를 수집하는 여러 단계에서 깊이 있는 정보가 확보되었고, 최신의 생성적 AI 모델인 GPT-4o와 Claude 3.5 Sonnet 및 Midjourney와 같은 이미지 및 비디오 생성 모델에 대한 언급도 있었습니다. 학생들은 기존 브랜드를 재구상하는 대신, 자신만의 브랜드를 구상하게 되었고, 이러한 과정에서 AI 도구를 활용하여 직접적인 실천 경험을 쌓았습니다.

- **Performance Highlights**: 결과적으로 이 통합된 접근법은 학생들이 비즈니스 환경에서 협업을 통해 수준 높은 결과물을 매우 빠른 속도로 생성할 수 있도록 지원하였습니다. 실험적 학습과 AI 도구의 사용은 학생들의 프로젝트에서 개인적 아이디어를 보다 잘 표현할 수 있도록 하는 긍정적인 영향을 미쳤습니다. 그러나, AI 리터러시가 높은 강사에 대한 수요 증가와 교육 과정 설계를 위한 Moving Target 문제는 여전히 해결해야 할 과제로 남아 있습니다.



### A Diffusive Data Augmentation Framework for Reconstruction of Complex Network Evolutionary History (https://arxiv.org/abs/2501.06485)
- **What's New**: 이번 연구는 복잡한 네트워크의 진화 과정을 재구성하고 예측하는 새로운 방법론을 제안합니다. 기존 방법의 한계를 극복하기 위해, 연구진은 다수의 네트워크를 통합해 학습하는 비교 패러다임 기반의 프레임워크를 도입하였으며, 이를 통해 엣지 생성 시간의 관계를 효과적으로 학습할 수 있음을 입증하였습니다. 또한, 생성된 시간 네트워크와 실제 시간 네트워크를 결합해 학습함으로써 추가적인 정확도 향상을 달성하였습니다.

- **Technical Details**: 이 연구에서는 복합 네트워크에서 엣지 생성 시간을 예측하기 위해 여러 가지 네트워크의 구조적 특징을 활용합니다. 연구진은 재구성된 문제를 쌍별 분류 문제로 변환하여 네트워크 간의 전이가 용이한 형태로 제시하였습니다. 더불어, 기존의 시간 네트워크 데이터를 활용해 무한대로 새로운 템포럴 네트워크를 생성하는 확산 모델(diffusion model)을 제안하였습니다.

- **Performance Highlights**: 다수의 네트워크를 활용한 훈련을 통해, 모델은 약 16.98%의 평균 정확도 향상을 보였습니다. 또한, 생성된 네트워크와 실제 네트워크를 결합하여 훈련한 결과, 추가적으로 5.46% 향상이 이루어졌습니다. 이 연구의 접근 방식은 정적 네트워크의 엣지 생성 시간 예측에서 기존 방법보다 높은 성과를 발휘함을 보여주었습니다.



### The Internet of Large Language Models: An Orchestration Framework for LLM Training and Knowledge Exchange Toward Artificial General Intelligenc (https://arxiv.org/abs/2501.06471)
- **What's New**: 이 논문은 Large Language Models (LLMs) 개발 시 직면하는 다차원적인 도전 과제를 탐구합니다. 특히 모델의 매개변수와 파일 크기의 방대함, 개발 환경 설정의 복잡성, 모델 기능의 단일성, 계산 자원의 높은 비용 등을 다루고 있습니다. 이를 해결하기 위해 LLM 공유 프로토콜, LLM 범용 환경 프레임워크, Agent 최적 경로 모듈 등 세 가지 핵심 기술 솔루션을 제안합니다.

- **Technical Details**: 이 연구는 초기 연구 단계에서의 계산 자원 제약을 해결하기 위해 혁신적으로 공동 채굴 메커니즘을 제안합니다. 이 메커니즘은 컴퓨팅 파워 공급자와 모델 설계자 간의 양자 간 가치 공유를 현실화하며, 최적 모델 경로에 대한 돌파구 보상 및 장기 수익 분배를 포함합니다. 이를 통해 연구자들에게 비용 최적화된 계산 자원 지원을 제공하고, LLM 연구 및 응용 프로그램의 지속적인 발전을 촉진합니다.

- **Performance Highlights**: 연구의 핵심 성과 중 하나는 계산 자원에 대한 제약을 극복하면서 연구자들의 효율성을 높이는 것입니다. 제안된 기술 솔루션은 LLM 개발의 구조적 문제를 완화하고, 여러 이해관계자 간의 협력을 증진하여 LLM의 연구와 실제 적용을 한층 더 촉진할 것으로 기대됩니다.



### Assessing instructor-AI cooperation for grading essay-type questions in an introductory sociology cours (https://arxiv.org/abs/2501.06461)
Comments:
          10 figures, 2 tables

- **What's New**: 이 연구는 고등 교육에서 에세이 질문의 채점에 대한 인공지능(AI)의 보조 도구로서의 활용을 탐구합니다. 특히, AI의 인간 채점과의 일관성 및 편견을 줄일 수 있는 가능성에 중점을 두고 있습니다. 70개의 손으로 쓴 시험지를 사용하여 성과를 평가한 것은 주목할 만한 새로운 접근입니다.

- **Technical Details**: 연구에서는 generative pre-trained transformers (GPT) 모델의 성과를 평가하기 위해 다양한 조건에서 자막(transcription) 및 채점(grading) 작업을 수행했습니다. 특히, GPT-4o-mini 모델이 정확도 측면에서 GPT-4o 모델을 능가하는 결과를 보였습니다. 템플릿 답안을 제공할 때 인간 채점자 점수와 강한 상관관계를 나타내는 점도 주목할 만한 사실입니다.

- **Performance Highlights**: GPT는 채점의 일관성을 검토하는 '두 번째 채점자(second grader)' 역할을 수행하여 인간 평가를 완전히 대체하기보다는 불일치 지점을 표시하는 데 중요한 역할을 합니다. 이러한 결과는 AI가 에세이식 질문의 채점에서 공정성과 효율성을 높일 수 있는 가능성을 보여줍니다.



### ARES: Auxiliary Range Expansion for Outlier Synthesis (https://arxiv.org/abs/2501.06442)
- **What's New**: 본 논문은 Out-of-Distribution (OOD) 탐지를 위한 새로운 방법론인 Auxiliary Range Expansion for Outlier Synthesis (ARES)를 제안합니다. 기존의 가상 이상치 생성을 위한 방법론들이 in-distribution 데이터셋에 의존하여 가상의 이상치 인스턴스를 생성하는데 한계가 있었던 부분을 해결하고자 합니다. ARES는 in-distribution 영역의 경계를 벗어나 가상의 OOD 인스턴스를 생성함으로써, OOD 탐지의 질을 향상시키고자 합니다.

- **Technical Details**: ARES는 샘플링 기반의 방법을 통해 가상의 이상치를 생성하며, 추가적인 훈련 비용이나 노력이 필요하지 않습니다. 이 방법은 Mixup 데이터를 활용하여 주어진 in-distribution 영역을 크게 벗어난 OOD 공간을 추정합니다. ARES는 이러한 새로운 OOD 인스턴스 생성 방식으로 여러 단계에 걸쳐 가성비 높은 OOD 유사 인스턴스를 생성합니다.

- **Performance Highlights**: 이 논문에서는 ARES의 성능 개선을 정량적으로 실험하였으며, 그 결과 기존 방법들보다 높은 성능을 보임을 확인하였습니다. 또한 ARES의 메커니즘을 이해할 수 있는 질적인 결과들도 제시되어, OOD 탐지에서의 효과적인 성능 향상을 증명하였습니다.



### AlgoPilot: Fully Autonomous Program Synthesis Without Human-Written Programs (https://arxiv.org/abs/2501.06423)
- **What's New**: AlgoPilot은 인간이 제공한 프로그램이나 경로 없이 완전 자동화된 프로그램 합성 프로그램이다. 이는 강화 학습(Reinforcement Learning)과 경로 언어 모델(Trajectory Language Model, TLM)을 활용하여 새로운 알고리즘을 생성할 수 있는 가능성을 열어준다. 이 연구는 기존 방법의 한계를 극복하고 알고리즘 발견을 위한 새로운 패러다임을 제시한다.

- **Technical Details**: AlgoPilot은 랜덤 파이썬 함수에 의해 생성된 경로를 기반으로 훈련된 TLM을 사용하여 알고리즘을 합성한다. 이 과정에서 강화 학습을 통해 알고리즘 훈련을 수행하며, 주어진 작업에 대해 'Compare'와 'Swap' 작업만을 사용하도록 제약된다. 경로 언어 모델의 예측은 강화 학습 과정에서 소프트 제약(soft constraint)으로 작용하여, 생성된 경로가 실제 알고리즘에 의해 생성될 법한 패턴과 일치하도록 유도한다.

- **Performance Highlights**: 정렬 알고리즘을 예로 들면, AlgoPilot은 버블 정렬(Bubble Sort)과 같은 고전적인 알고리즘으로 해석 가능한 경로를 생성할 수 있음을 보여준다. 기존 방법들처럼 인간이 작성한 프로그램 없이도 이러한 알고리즘을 생성할 수 있는 능력을 입증하였다. AlgoPilot은 자율 프로그램 합성 분야에서의 미래 발전을 위한 기초를 마련하였다.



### Multi-Agent Collaboration Mechanisms: A Survey of LLMs (https://arxiv.org/abs/2501.06322)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 사람 수준의 작업을 수행할 수 있는 에이전틱 AI가 주목받고 있습니다. 이러한 LLM 기반의 다중 에이전트 시스템(Multi-Agent Systems, MAS)은 지능형 에이전트들이 복잡한 작업을 협력하여 해결할 수 있는 새로운 가능성을 열어줍니다. 본 논문은 MAS의 협력적 측면에 대한 종합적인 조사를 제공하고 향후 연구 방향을 제시하는 확장 가능한 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 협력 메커니즘을 배우고 행동할 수 있는 LLM 기반 에이전트들 간의 협력을 자극하는 키 요소를 정의합니다. 이는 행동자(actors), 유형(types), 구조(structures), 전략(strategies), 조정 프로토콜(coordination protocols)과 같은 측면에 따라 협력 메커니즘을 분류합니다. 이러한 세부 요소는 연구자들이 MAS의 다양한 특성을 더 깊이 이해하고 설계할 수 있도록 돕습니다.

- **Performance Highlights**: MAS는 정보 공유, 업무 분배, 그리고 공동 목표 달성을 통한 문제 해결을 지원하여 다양한 분야에서 눈에 띄는 성공을 거두고 있습니다. 이 시스템들은 에이전트들이 지식을 배양하고 장기적인 계획을 수립하며, 복잡한 문제를 효과적으로 해결하는 데 기여하고 있습니다. 마지막으로, 연구자들은 MAS의 구현 사례와 함께 중요한 교훈, 도전 과제, 및 잠재적인 연구 방향을 제시하여 AI의 집단 지능 발전을 모색하고 있습니다.



### BioAgents: Democratizing Bioinformatics Analysis with Multi-Agent Systems (https://arxiv.org/abs/2501.06314)
- **What's New**: 이 논문에서는 BioAgents라는 다중 에이전트 시스템을 제안했습니다. 이는 바이오 인포매틱스 데이터에 대해 세밀하게 조정된 소형 언어 모델에 기반하여 설계되었습니다. 기존의 대형 언어 모델과는 달리, BioAgents는 개인화된 데이터와 로컬 운영을 지원하여 사용자 맞춤형 솔루션을 제공합니다.

- **Technical Details**: BioAgents는 Phi-3이라는 소형 언어 모델을 바탕으로 하고 있으며, 개념적 유전자 작업 및 코드 생성 작업에서 전문가와 유사한 성능을 보여줍니다. 시스템은 다양한 질문을 해결하기 위해 특수화된 에이전트를 사용하며, 툴 선택, 워크플로우 생성 및 오류 해결과 같은 특정 작업에 맞춤화되었습니다. 논문에서 제안한 RAG(검색 증강 생성) 기법은 비바이오 인포매틱스 문서와 데이터베이스에서 학습된 지식을 활용합니다.

- **Performance Highlights**: BioAgents는 개념적 유전자 질문에서 전문가와 유사한 성능을 보여주며, COVID-19 유전자 조합 및 분석과 관련된 복잡한 작업에서도 성공적으로 논리적 단계를 제시했습니다. 하지만 코드 생성 작업에서는 특정 복잡성 수준에서 성능 차이를 보였으며, 특히 높은 복잡도의 작업에서 정확한 시작 코드를 생성하는 데 어려움을 겪었습니다. 이를 바탕으로 향후 더 다양한 툴과 언어를 갖춘 데이터셋을 사용하는 것이 필요할 것으로 보입니다.



### Agent TCP/IP: An Agent-to-Agent Transaction System (https://arxiv.org/abs/2501.06243)
Comments:
          17 pages, 2 figures

- **What's New**: 이 연구는 자율 에이전트의 발전을 다루며, 현재의 에이전트 프레임워크는 에이전트 간의 상호작용을 위한 표준 프로토콜이 부족하다는 점을 강조합니다. 에이전트 간의 진정한 경제를 실현하기 위해, 이 논문에서는 에이전트가 서로 계약을 체결하고 지식 재산을 교환할 수 있는 보편적인 프레임워크인 ATCP/IP를 제안합니다. 이를 통해 에이전트는 독립적으로 지식 재산을 거래할 수 있습니다.

- **Technical Details**: ATCP/IP는 에이전트 간의 IP(지식 재산)을 거래하기 위한 신뢰 없는 프로토콜로, 프레그러머블 계약을 통해 에이전트가 계약을 시작하고 거래, 차용, 판매할 수 있는 기능을 구현합니다. 이러한 계약은 블록체인 네트워크인 Story에서 실행되며, 오프체인(Offchain) 법적 환경에서 행동을 표현하고 집행할 수 있는 법적 구조를 갖추고 있습니다. 이로 인해 에이전트는 법적 주체성을 획득하게 됩니다.

- **Performance Highlights**: ATCP/IP를 통해 에이전트는 서로의 학습 데이터를 자율적으로 판매하고, 기밀 정보를 라이센스하며, 자신만의 특수한 기술을 활용하여 콘텐츠 협업을 할 수 있습니다. 이는 지식 경제(Knowledge Economy)의 출현을 의미하며, 에이전트 간의 다양한 자율적 거래를 가능하게 만듭니다. 이러한 혁신은 다중 에이전트 상호작용에서 인간의 중재 없이 효율성을 높이는 데 기여할 것입니다.



### Sustainable and Intelligent Public Facility Failure Management System Based on Large Language Models (https://arxiv.org/abs/2501.06231)
- **What's New**: 이 논문은 공공 시설 내 스마트 장치 관리에 대한 새로운 프레임워크를 제안합니다. 특히, 도서관과 같은 환경에 적합하게 설계된 이 프레임워크는 최신 LLM(대규모 언어 모델)을 활용하여 장치 고장을 분석하고 예측하여 운영 효율성과 신뢰성을 향상시킵니다. 프로토타입 검증을 통해 이 시스템의 실용성을 입증하였으며, 공공 시설의 예산 절감에 기여할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 프레임워크는 다양한 공공 시설에 적용 가능하며, IoT 보안 및 위협 탐지를 위한 머신 러닝 알고리즘을 통합할 계획입니다. 이 시스템은 실시간적으로 데이터를 모니터링하고 분석하여 장치 고장을 사전에 예방하며, 개인정보 보호 및 보안을 보장하는 데이터 로컬화 관리를 구현합니다. 또한, 다양한 출처의 이질적인 데이터를 효과적으로 통합하는 해결책을 제공합니다.

- **Performance Highlights**: 이 모델은 도서관을 포함한 다양한 공공 시설의 시나리오에서 검증되었으며, 운영 효율성 제고와 공공 서비스 개선에重大한 기여를 할 수 있습니다. 시스템의 성공적인 프로토타입 테스트는 행정 및 서비스 품질을 향상시킬 것으로 예상됩니다. 이러한 접근 방식은 스마트 장치의 고장을 예측하고 효과적으로 관리하여, 지속 가능한 도시 문화 공간 개발을 촉진하는 데 기여할 것입니다.



### A Novel Method for Pignistic Information Fusion in the View of Z-number (https://arxiv.org/abs/2501.06201)
- **What's New**: 이 논문에서는 복잡한 출처에서 정보를 융합하는 방법에 대한 새로운 접근 방식을 제안합니다. Dempster-Shafer 증거 이론(DSET)을 바탕으로 하여, pignistic transformation과 Z-number를 사용하여 서로 다른 출처의 정보를 융합하는 완전히 새로운 방법을 개발했습니다. 이 방법은 정보의 개별 상황을 처리하고 실제 상황에 대한 합리적이고 정확한 판단을 만드는 데 높은 정확성을 유지합니다.

- **Technical Details**: 제안된 방법은 두 가지 측면에서 문제를 해결합니다. 첫째, 충돌하는 부분을 분리하고 각 증거의 불확실성을 계산하여 증거 간 가중치를 결정합니다. 둘째, Z-number의 개념을 참고하여 각 증거에 대한 Z-number 유사 구조를 구성함으로써 전반적인 신뢰성을 향상시킵니다. 이러한 과정을 통해 얻어진 증거는 전통적인 Dempster 결합 규칙을 통해 융합되어 결과적으로 각 제안에 대한 신뢰성 할당을 도출합니다.

- **Performance Highlights**: 제안된 방법은 DSET의 이점을 활용하여 정보 조합 과정을 재설계함으로써 훨씬 더 정확하고 합리적인 결과를 얻을 수 있습니다. 이 방법은 전통적인 방법과 현대의 일부 방법들을 여러 상황에서 능가하는 뛰어난 성과를 보여줍니다. 또한, 정보 출처의 우선 순위 수준을 명확히 하는 신뢰성 측정 시스템의 도입으로 종합적인 정보 집합과 더 나은 결정을 가능하게 합니다.



### A Novel Task-Driven Method with Evolvable Interactive Agents Using Event Trees for Enhanced Emergency Decision Suppor (https://arxiv.org/abs/2501.06193)
- **What's New**: 이 논문은 급변하는 기후 변화 및 기타 글로벌 이슈에 대처하기 위해, 예기치 않은 사건들에 신속하게 대응하는 새로운 접근법인 EvoTaskTree를 제안합니다. 이 방법은 대규모 언어 모델(LLMs)을 이용한 두 종류의 에이전트, 즉 작업 실행자와 작업 검증자를 통합하여 효율적인 비상 대응 결정을 지원합니다. EvoTaskTree는 복잡한 시스템에서의 작업 기반 가상 플랫폼을 구축하여, 보다 신속하고 효과적인 비상 대응을 가능하게 합니다.

- **Technical Details**: EvoTaskTree는 이벤트 트리 분석(event tree analysis)과 의사결정 권장사항을 포함하여 세 가지 핵심 작업을 수행합니다. 에이전트는 성공과 실패의 경험을 통해 학습하고, 이는 안전-critical 시스템인 원자력 발전소(NPP)에서의 의사결정을 지원하는 데 도움을 줍니다. 이 시스템은 기존의 주관적인 의사결정 방법 대신 정량적 위험 분석(quantitative risk analysis)을 기반으로 하는 일관된 접근법을 제공합니다.

- **Performance Highlights**: EvoTaskTree는 기존의 접근 방식을 능가하는 뛰어난 성능을 보이며, 100%에 가까운 정확도로 이전에 경험하지 못한 사건 시나리오를 처리할 수 있음을 확인했습니다. 이 연구에서 제안한 방법은 비상 상황에서의 신속한 의사결정 능력을 획기적으로 향상시킬 수 있는 가능성을 보여주고 있습니다.



### A Computational Model of Learning and Memory Using Structurally Dynamic Cellular Automata (https://arxiv.org/abs/2501.06192)
- **What's New**: 이 논문은 학습과 기억을 위한 수학적 모델을 제안하고, 생물학적 가능성을 염두에 두고 간단한 기능 세트를 기반으로 계산 과정을 효율적으로 모델링합니다. 주요 논의는 정보 공간의 구축 및 컴퓨테이션(computation) 수행 방법에 대한 것입니다. 이 모델은 기존의 인공지능 방법들이 가진 데이터 샘플 수요, 계산 요구사항, 일반화 학습 등의 제한 요소를 해결하려고 합니다.

- **Technical Details**: 제안된 모델은 구조적으로 동적 셀룰러 오토마타(cellular automaton)를 사용하여 연속 값의 셀 상태와 비방향 그래프에서의 재귀적 단계를 통해 계산을 수행합니다. 이 모델의 기본 기능으로는 동시 감지(coincidence detection), 신호 조절(signal modulation), 보상/처벌 메커니즘(reward/penalty mechanisms)이 포함됩니다. 모델은 복잡한 그래프 네트워크가 실험적으로 검증되어, 다양한 보상 및 처벌 시나리오에서 최적의 선택을 할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 이 장난감 모델(toy model)은 단일 학습 후에 보상 상태를 재발견하는 거의 최적의 선택을 할 수 있었습니다. 또한 복잡한 처벌 구성(penalty configurations)을 회피하고, 희소 보상 환경에서 탐색 행동(exploratory behaviors)을 생성하는 능력을 보였습니다. 이 모델은 최소한의 교육 요구와 유연한 메모리 표현 덕분에 높은 계산 효율성을 보여주었습니다.



### A Multimodal Social Agen (https://arxiv.org/abs/2501.06189)
Comments:
          9 pages

- **What's New**: 이 논문은 MuSA라는 멀티모달 대형 언어 모델(LLM) 기반 에이전트를 소개하며, 주어진 인간 중심의 콘텐츠 분석 작업을 수행하기 위한 사회적 콘텐츠 분석을 자동화하고 개선할 수 있는 가능성을 보여줍니다. MuSA는 질문 응답, 시각적 질문 응답, 제목 생성 및 분류와 같은 작업을 다루는 데 맞춰 설계되었습니다. 이 에이전트는 계획, 추론, 실행, 최적화 및 비판 전략을 사용하여 작업을 완료합니다.

- **Technical Details**: MuSA는 모듈화 및 확장성이 뛰어난 구조를 가지고 있습니다. 현재 다섯 개의 독립적인 LLM 기반 유닛인 추론(reason), 계획(plan), 최적화(optimize), 비판(criticize), 세분화(refine) 및 실행(act)을 지원하며, 다양한 작업 요구에 맞춰 결합할 수 있습니다. 또한 MuSA는 체인 오브 사고(chain-of-thought, CoT) 및 자기 반영(self-reflection)과 같은 복잡한 내부 프로세스를 활용하여 성능을 향상시키는 데 최적화되어 있습니다.

- **Performance Highlights**: MuSA는 질문 응답, 제목 생성 및 콘텐츠 분류 작업에서 기존 기준 대비 큰 성과를 보였습니다. 최근 멀티모달 LLM들이 시각적 질문 응답 및 이미지 캡셔닝과 같은 데이터 분석 작업에서 긍정적인 결과를 나타내고 있으며, MuSA도 이러한 멀티모달 기능을 통해 복잡한 질문에 대한 답변을 더욱 효율적으로 제공하고 더 정교한 행동을 이끌어낼 수 있습니다.



### Dataset Distillation via Committee Voting (https://arxiv.org/abs/2501.07575)
Comments:
          Code at: this https URL

- **What's New**: 이번 연구에서는 데이터셋 증류(Dataset Distillation) 분야에 새로운 접근 방식인 CV-DD(Committee Voting for Dataset Distillation)를 제안합니다. 이 방법은 다수 모델의 집단 지혜를 활용하여 고품질의 증류된 데이터셋을 생성합니다. 기존의 연구에서는 주로 단일 모델 또는 복수 모델 간의 정렬(alignment)과 효율성 향상에 초점을 맞췄으나, CV-DD는 이와는 다른 차별화된 접근을 나타냅니다.

- **Technical Details**: CV-DD 프레임워크는 Prior Performance Guided (PPG) Voting Mechanism을 도입하였으며, 여러 모델의 예측 및 분포를 집계하여 대표적인 데이터 포인트를 식별합니다. 이 방법은 모델 특유의 편향을 줄이고, 증류된 데이터셋의 다양성을 증진시키며, 각 모델의 고유한 장점을 활용해 과적합(overfitting)을 완화합니다. 또한, 동적 투표 설계를 통해 특정 기능 또는 데이터셋 속성을 우선시할 수 있도록 조정할 수 있는 세세한 제어를 제공합니다.

- **Performance Highlights**: CIFAR, Tiny-ImageNet, ImageNet-1K 및 그 하위 집합에 대한 광범위한 실험을 통해 CV-DD는 기존의 단일/복수 모델 증류 방법보다 더 높은 정확도와 모델 간 일반화를 달성함을 입증하였습니다. CV-DD로 증류된 데이터셋은 낮은 데이터나 제한된 컴퓨팅 환경에서도 높은 성능을 나타내며, 여러 시나리오에서 신뢰성과 적응력을 향상시킵니다. 이 연구는 고효율 데이터 사용과 컴퓨팅 효율성이 필수적인 응용 분야에서 CV-DD의 큰 잠재력을 강조합니다.



### UnCommon Objects in 3D (https://arxiv.org/abs/2501.07574)
- **What's New**: uCO3D는 3D 딥 러닝 및 3D 생성 AI를 위한 새로운 객체 중심 데이터셋으로서, 고해상도 비디오와 3D 주석이 포함된 가장 큰 공개 데이터베이스입니다. 이 데이터셋은 전방위 360도 커버리지를 제공하고 있으며, 1,000개 이상의 객체 카테고리를 포괄합니다.

- **Technical Details**: uCO3D는 MVImgNet과 CO3Dv2보다 품질이 더욱 우수하며, 수집된 비디오와 3D 주석에 대한 철저한 품질 검사를 거쳤습니다. 이 데이터셋은 3D 카메라 포즈, 깊이 맵(depth maps), 희소 포인트 클라우드(sparse point clouds) 주석을 포함하고 있으며, 각 객체에는 캡션과 3D Gaussian Splat 재구성이 제공됩니다.

- **Performance Highlights**: uCO3D에서 훈련된 여러 대규모 3D 모델은 MVImgNet 및 CO3Dv2에서 훈련된 모델보다 우수한 성능을 보여 주었습니다. 이는 uCO3D가 학습 응용 프로그램에 더 적합하다는 것을 입증하는 결과입니다.



### WebWalker: Benchmarking LLMs in Web Traversa (https://arxiv.org/abs/2501.07572)
- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG) 기술의 성능을 한층 향상시키기 위해 WebWalkerQA라는 새로운 벤치마크를 도입합니다. 기존의 전통적인 검색 엔진이 제공하는 얕은 콘텐츠의 한계를 극복하고, 웹 서브페이지를 탐색해 고품질 데이터를 체계적으로 추출하는 LLM의 능력을 평가하는 것을 목표로 합니다.

- **Technical Details**: WebWalker는 인간의 웹 탐색을 모방하는 다중 에이전트 프레임워크로, explore-critic 패러다임을 활용하여 고급 탐색 기능을 구현합니다. 이 프레임워크는 LLM이 웹사이트 내의 서브페이지를 어떻게 효율적으로 탐색하고 정보를 추출하는지를 평가하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, WebWalkerQA는 도전적이며 실제 시나리오에서 RAG와 WebWalker의 통합이 매우 효과적임을 입증합니다. 수평적 및 수직적 통합을 통해 다양한 과제에서 더욱 풍부하고 복잡한 정보 처리가 가능해졌습니다.



### Evaluating Agent-based Program Repair at Goog (https://arxiv.org/abs/2501.07531)
- **What's New**: 이 논문은 에이전트 기반 프로그램 수리(Agent-based Program Repair)의 가능성을 탐구합니다. 특히, 구글의 내부 이슈 추적 시스템에서 선별된 178개의 버그를 이용하여 기업 환경에서의 에이전트 기반 접근법의 효과를 평가합니다. 논문에서는 Passerine이라는 새로운 에이전트 시스템을 구현하여, 머신 및 사람이 보고한 버그에 대한 수리 성능을 제시하고 있습니다.

- **Technical Details**: Automated Program Repair (APR) 시스템의 운영 환경에서는 여러 버그들이 다양한 특성을 지니고 있습니다. 이 연구에서는 제대로 설계된 에이전트가 있는 GITS-Eval이라는 평가 세트를 구성하여 기계 보고 버그의 73%, 인간 보고 버그의 25.6%에서 유효한 패치를 생성했다는 점을 강조합니다. GITS-Eval을 통해 SWE-Bench와의 차이를 분석하고, GITS 환경에서의 버그 처리 방식을 효과적으로 이해하고 있습니다.

- **Performance Highlights**: Passerine는 전체 178개의 버그 중 머신 보고 버그에 대해 73%의 패치 성공률을 보였으며, 인간 보고 버그에 대해 25.6%의 성공률을 기록했습니다. 이 연구를 통한 성과는 산업에서의 잠재적 적용 가능성을 제시하며, 향후 에이전트 기반 APR 시스템의 발전 가능성을 내포하고 있습니다. 연구 결과는 GITS 환경의 특수성과 버그 처리 방식의 차이를 고려할 수 있는 중요한 기초 데이터를 제공합니다.



### RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignmen (https://arxiv.org/abs/2501.07525)
- **What's New**: 본 논문에서는 RadAlign이라는 새로운 프레임워크를 제안합니다. RadAlign은 병리학적 진단의 정확성과 해석 가능성을 동시에 중시하면서 방사선 사진 해석과 보고서 생성을 통합합니다. 이 프레임워크는 시각-언어 모델(Vision-Language Models, VLM)과 대형 언어 모델(Large Language Models, LLM)의 장점을 결합하여 의사 소통의 품질을 향상시키고 의료 영상 분석의 신뢰성을 높입니다.

- **Technical Details**: RadAlign은 먼저 VLM을 사용하여 시각적 특징을 의료 개념과 정렬합니다. 이는 방사선과 의사가 특정 진단 기준을 확인하고 이를 바탕으로 판단하는 과정을 모방합니다. 그런 다음 정렬된 시각적-언어 공간에서 저희가 인식한 질병을 텍스트 기반 개념으로 표현하고, 이를 활용해 LLM에 의해 보고서를 생성하도록 유도합니다.

- **Performance Highlights**: RadAlign은 다양한 질병에 대해 평균 AUC가 0.885로 우수한 질병 분류 성능을 달성합니다. 보고서 품질은 GREEN 점수가 0.678로, 현재까지의 최첨단 방법인 0.634를 초과합니다. 이 프레임워크는 임상 해석 가능성을 유지하면서 착각을 줄이고, 통합된 예측 및 생성 AI를 통해 자동화된 의료 영상 및 보고서 분석을 발전시킵니다.



### The Paradox of Success in Evolutionary and Bioinspired Optimization: Revisiting Critical Issues, Key Studies, and Methodological Pathways (https://arxiv.org/abs/2501.07515)
Comments:
          38 pages, 1 figure

- **What's New**: 최근의 연구들은 진화적 및 생물 영감을 받은 알고리즘들이 복잡한 최적화 문제를 효율적으로 해결할 수 있음을 보여주고 있습니다. 전통적인 방법에 비해 이들 알고리즘은 더 나은 솔루션을 제공하며, 특히 기계 학습 및 심층 신경망과 같은 여러 분야에서 그 가능성을 입증했습니다. 그러나 이러한 알고리즘들은 무분별한 제안과 낮은 품질의 실험 연구로 인해 진정한 발전이 저해되고 있으며, 이로 인해 연구 커뮤니티는 더 나은 방법론적 지침을 제안하고 있습니다.

- **Technical Details**: 생물 영감을 받은 계산(Bioinspired Computation)은 메타 휴리스틱 최적화 연구의 중요한 분야로, 자연을 모방하여 최적화 알고리즘을 설계합니다. 그러나, 최근의 연구들은 낮은 혁신성과 과학적 엄격성 부족이 문제라고 지적하고 있습니다. 알고리즘의 분류와 방법론적 경로의 발전은 이러한 비판을 해결하기 위한 연구 커뮤니티의 노력으로 대응되고 있습니다.

- **Performance Highlights**: 생물 영감을 받은 알고리즘의 과잉 출현은 오히려 연구의 진전을 저해하는 역설적인 상황을 만들어냈습니다. 기존 연구들은 해당 알고리즘들이 기존의 아이디어를 재활용할 뿐 혁신성이 부족하다는 점을 강조하고 있으며, 이로 인해 메타 휴리스틱 분야에서의 건전한 연구 관행 확립의 필요성이 대두되고 있습니다. 이러한 비판과 연구 결과에 따라 향후 최적화 연구의 방향성을 정의하는 데 중요한 기여가 이루어져야 합니다.



### RbRL2.0: Integrated Reward and Policy Learning for Rating-based Reinforcement Learning (https://arxiv.org/abs/2501.07502)
Comments:
          Accepted to the Collaborative AI and Modeling of Humans Bridge Program at AAAI 2025

- **What's New**: 이 논문은 기존의 강화 학습(Reinforcement Learning) 접근법에서 발생하는 문제를 해결하기 위해, 인간의 의사결정 과정을 모방하는 새로운 RL 방법론을 제안합니다. 이 접근법은 다양한 성능 수준에 따라 경험을 차별화하고, 이를 통해 정책(policy) 학습의 효율성을 높입니다. 핵심 아이디어는 다양한 성능 수준의 경험에서 중요한 방향 정보를 추출하고, 이를 통해 정책을 업데이트하는 것입니다.

- **Technical Details**: 제안된 방법은 현재 정책과 실패한 경험 간의 분포 유사성을 벌점화하는 새로운 정책 손실 함수(policy loss function)를 도입하며, 평가 등급(rating classes)에 따라 벌점 용어의 가중치를 다르게 설정합니다. 특히, 경험의 등급이 높을수록 정책과의 차이를 줄이고, 등급이 낮을수록 차이를 늘리는 방식으로 다루어집니다. 이 과정에서 Kullback–Leibler(KL) 발산을 기반으로 한 손실 함수를 활용하며, 여러 복잡한 환경에서 실험하여 새로운 접근법의 출력을 평가합니다.

- **Performance Highlights**: 제안된 방법은 다양한 환경에서 기존의 보상 학습(reward learning)과 비교하여 개선된 수렴성과 전반적인 성능을 보여줍니다. 특히, 실패한 경험에서 얻은 값을 효과적으로 활용함으로써(policy를 최적화) 강화 학습의 잠재력을 더욱 극대화하는 결과를 나타냅니다. 이러한 성과는 향후 RL의 다양한 응용 분야에서 유용하게 활용될 수 있습니다.



### Smart Learning in the 21st Century: Advancing Constructionism Across Three Digital Epochs (https://arxiv.org/abs/2501.07486)
Comments:
          22 pages

- **What's New**: 이 논문은 교육 모델로서의 구성주의(constructionism)의 발전을 탐구하며, 개인용 컴퓨팅, 네트워크 사회, 그리고 현재의 생성적 AI(generative AI)라는 세 가지 중요한 시대를 통해 그 중요성과 변화를 추적합니다. 또한 디지털 기술의 발전이 개인 및 집단 학습에서 어떤 방식으로 구성주의 원칙과 맞물려 있는지를 살펴봅니다.

- **Technical Details**: Seymour Papert의 구성주의 철학에 뿌리를 두고, 이 연구는 교육 환경의 변화를 계층적 지식 전달법(instructionism)에서 학습자 자율성과 상호작용적인 창의적 참여를 강조하는 구성주의 모델로의 전환을 논의합니다. 여기서 중심이 되는 개념은 확장된 성격(expanded personality)으로, 디지털 도구와 AI의 통합이 개인의 자아 인식과 사회적 상호작용을 어떻게 근본적으로 재형성하는지를 다룹니다.

- **Performance Highlights**: 구성주의를 스마트 교육 paradigm에 통합함으로써, 개인화된 민주적 학습의 기초적 접근법으로 제안합니다. 연구 결과는 기술 주도 교육의 복잡성을 헤쳐나가는 데 있어 구성주의의 지속적인 중요성을 강조하며, 디지털 혁신을 활용하여 적응형(student-centered) 학습 경험을 촉진하고자 하는 교육자 및 정책 입안자들에게 유용한 통찰력을 제공합니다.



### TiEBe: A Benchmark for Assessing the Current Knowledge of Large Language Models (https://arxiv.org/abs/2501.07482)
- **What's New**: 이 논문은 Timely Events Benchmark (TiEBe)를 소개하여 대형 언어 모델(LLMs)의 최신 사건에 대한 지속적인 업데이트 필요성을 강조합니다. TiEBe는 11,000개 이상의 질문-답변 쌍을 포함하고 있으며, 다양한 지역의 지식 격차와 진화하는 글로벌 이슈에 대한 LLM의 이해를 평가하는 데 중점을 두고 있습니다. 기존의 벤치마크가 일반적인 사실 회상에 중점을 두었다면, TiEBe는 지속적인 학습(continuous learning) 전략을 평가할 수 있는 강력한 도구로 작용합니다.

- **Technical Details**: TiEBe는 Wikipedia의 구조화된 회고 데이터를 활용하여 전 세계 및 지역적 사건에 대한 질문-답변 쌍을 생성합니다. 이 데이터는 2015년부터 2024년까지의 다양한 나라 및 글로벌 사건들을 포함하며, 이를 통해 LLM의 지식 업데이트 및 지역적 이해력을 평가할 수 있습니다. 질문-답변 쌍은 영어로 생성되며, 여러 LLM에 대해 동일한 평가 기준을 적용하여 지역적 편향을 탐색합니다.

- **Performance Highlights**: 시험된 모든 LLM들은 사실 회상에서 상당한 지역적 차이를 보였으며, 이는 모델들이 특정 지역의 정보에 비해 잘 알려지지 않은 지역 정보를 잘 처리하지 못함을 나타냅니다. TiEBe의 결과는 글로벌 지식 표현의 균형을 개선할 필요성을 강조하고, 지속적인 학습 능력을 평가하는 데 있어서도 중요한 참고자료를 제공합니다. 이 연구는 특정 지역에 특화된 데이터로 훈련된 LLM의 성능 개선 가능성을 탐색하는 기회를 제공합니다.



### Estimating Musical Surprisal in Audio (https://arxiv.org/abs/2501.07474)
Comments:
          5 pages, 2 figures, 1 table. Accepted at the 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2025), Hyderabad, India

- **What's New**: 이번 연구에서는 오토리그레시브 모델을 사용하여 음악 오디오의 놀라움(expectancy) 지수를 계산하기 위한 새로운 방법론을 제안하였습니다. 기존의 기호 음악(symbolic music) 데이터 분석 방식이 아닌, 압축된 오디오 표현(compressed audio representations)을 통해 머신러닝 모델을 훈련하고, 인간의 놀라움을 예측하기 위한 정보 내용(information content, IC)을 분석합니다. 이 접근 방식은 음악의 감성적 특성을 더욱 폭넓게 이해하고 모델링할 수 있는 가능성을 제시합니다.

- **Technical Details**: 우리는 변형된 Transformer 모델을 사용하여 오디오 신호를 Music2Latent의 역변환 오디오 표현으로 인코딩하고, 음악 세그먼트의 IC를 추정합니다. 이 과정에서, IC는 반복적인 세그먼트에서의 학습 효과를 분석하여 감소를 확인하고, 음악 작곡 내에서 후속 세그먼트의 IC가 평균적으로 더 높다는 것을 발견했습니다. 또한, 우리는 IC가 음향 및 음악적 특성과 어떻게 상관관계가 있는지를 조사하여, 주로 음색(timbre)과 음량(loudness)과 연관이 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과, 우리의 모델에서 추정된 IC가 EEG(뇌파) 응답을 유의미하게 예측할 수 있음이 확인되었습니다. 이는 음악의 놀라움을 모델링하는데 있어 우리의 접근 방식이 인간의 기대를 잘 반영한다는 것을 의미합니다. 최종적으로, 우리는 이 연구를 통해 음악의 자연적인 복잡성과 인간의 감성을 측정하는데 중요한 방향성을 제시하고 코드도 공개하였습니다.



### Attention when you need (https://arxiv.org/abs/2501.07440)
- **What's New**: 본 연구는 주의 집중이 작업 수행에 긍정적인 영향을 미치지만, 그에 따른 대사 비용도 함께 고려해야 한다는 점을 강조합니다. 주의 자원을 전략적으로 할당하는 것이 효율적인 작업 수행에 중요하다는 것을 밝혀내고, 이에 대한 규범적 모델을 개발했습니다. 특히, 생쥐의 실험을 통하여 주의 집중 수준의 선택과 비용에 대한 균형을 이해하고자 합니다.

- **Technical Details**: 연구에 사용된 모델은 강화 학습 기반의 규범적 모델로, 생쥐가 얼마나 주의 집중을 분배하는지를 탐색합니다. 생쥐는 두 가지 주의 수준 중에서 선택할 수 있으며, 높은 주의 상태에서 리워드를 얻기 위해 비용이 드는 행동을 결정합니다. 결과적으로, 연구는 주의 자원을 효율적으로 사용하기 위해 높은 주의와 낮은 주의를 교차적으로 사용해야 한다는 것을 제안합니다.

- **Performance Highlights**: 실험 결과에 따르면, 생쥐는 높은 주의가 필요한 상황에서 리드미컬하게 주의 집중을 조절하여 최적의 결과를 획득하는 경향이 있었습니다. 이 연구는 주의 배분이 작업의 유용성을 기반으로 어떻게 달라지는지에 대한 중요한 통찰을 제공하며, 신경 생리학적 상관관계가 주의의 변동에 따라 어떻게 변화하는지도 예측할 수 있는 단초를 제공합니다.



### Diff-Ensembler: Learning to Ensemble 2D Diffusion Models for Volume-to-Volume Medical Image Translation (https://arxiv.org/abs/2501.07430)
- **What's New**: Diff-Ensembler는 기존의 모델들이 겪는 3D 표현의 한계를 극복하기 위해 다각적으로 훈련된 2D diffusion 모델과 3D 네트워크를 결합한 새로운 하이브리드 모델입니다. 이 모델은 효율적인 volumetric translation을 가능하게 하여 의료 이미지의 정밀도와 현실감을 향상시킵니다. 또한, Diff-Ensembler는 다양한 입력 모달리티에 대한 조합을 자연스럽게 활용할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: Diff-Ensembler는 두 단계의 훈련 전략을 채택합니다: 첫 번째로, 수직 방향에서 여러 2D diffusion 모델을 훈련하고, 두 번째로 3D 융합 네트워크를 사용하여 각 diffusion 단계에서 최종 번역을 생성합니다. 이는 2D 모델에서 얻은 계층적 특성 맵을 효율적으로 활용하여 3D 출력을 더욱 정확하고 공간적으로 현실감 있게 만듭니다. 또한, Mixture-of-Experts (MoE) 접근법을 사용하여 2D 모델을 혼합하여 더 효과적인 성능을 냅니다.

- **Performance Highlights**: Diff-Ensembler의 성능은 BraTS 및 HCP 데이터셋을 활용한 다양한 MRI 이미지 처리 작업에서 평가되었으며, 기존 모델 대비 우수한 성능을 보여주었습니다. 이 모델은 새로운 2D 모델을 재훈련하지 않고도 다각적 입력 모달리티가 주어졌을 때 강력한 성능을 발휘합니다. 실험 결과, 특히 3D 의료 이미지의 super-resolution 및 모달리티 번역 작업에서 뛰어난 volumetric realism을 달성했음을 확인했습니다.



### An Investigation into Seasonal Variations in Energy Forecasting for Student Residences (https://arxiv.org/abs/2501.07423)
- **What's New**: 이 연구는 학생 주거 환경의 계절적 변동이 에너지 예측에 미치는 독특한 도전 과제를 강조하며, 다양한 기계 학습 모델을 평가합니다. 연구는 LSTM와 GRU와 같은 기본 모델과 함께 최신 예측 방법인 Autoregressive Feedforward Neural Networks, Transformers, 하이브리드 접근 방식을 함께 비교 분석합니다. 결과적으로, 특정 계절에 최적화된 모델 선택의 필요성이 강조되며, Hyper Network 기반 LSTM과 MiniAutoEncXGBoost 모델이 특히 여름철 소비 변화에 강한 적응성을 보임을 발견했습니다.

- **Technical Details**: 프로젝트의 초점은 두 개의 실제 데이터셋(Residence 1과 Residence 2)을 활용하여 하루 앞서 에너지를 예측하는 것입니다. 초기 단계에서 Multi-Layer Perceptron (MLP), Temporal Convolution Neural Network (TCN), Recurrent Neural Network (RNN), Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM)과 같은 다섯 가지 기본 모델을 평가했습니다. 모델 성능 평가는 Symmetric Mean Absolute Percentage Error (SMAPE), Mean Absolute Error (MAE), Root Mean Square Error (RMSE)와 같은 세 가지 지표를 통해 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 모델의 계절별 적응성에 따라 예측 정확도가 달라짐을 밝혀내었습니다. 특히 Hyper Network와 LSTM 모델은 여름철의 급격한 에너지 소비 변화를 효과적으로 포착하는 데 성공했습니다. 결론적으로 이 연구는 에너지 예측 분야에 계절적 동역학의 중요성과 모델 특유의 행동을 강조하며, 에너지 소비 예측의 정확성을 높이는 데 기여할 것입니다.



### PROTECT: Protein circadian time prediction using unsupervised learning (https://arxiv.org/abs/2501.07405)
- **What's New**: 본 논문에서는 최신의 컴퓨터 기반 접근법인 PROTECT를 제안합니다. 이 방법은 고차원 단백질 체계에서 시계열 편집 사항을 예측할 수 있으며, 시간 레이블 없이도 작동합니다. 기존의 방법이 필요한 시간 레이블을 요구하는 것과 대조적으로, PROTECT는 보다 유연하고 저렴한 타임라인 예측이 가능합니다.

- **Technical Details**: PROTECT 모델은 깊은 신경망(Deep Neural Network, DNN)을 기반으로 하여 시계열에서 각 샘플의 단백질 리듬을 예측합니다. 이는 사전 훈련 없이도 고차원 데이터로부터 직접적으로 데이터를 처리하는데, 단계별로 계층을 학습하여 표시 패턴을 포착합니다. 특히, 각 단계의 가중치를 독립적으로 학습함으로써 소량의 샘플 데이터에서도 과적합(overfitting) 문제를 줄일 수 있습니다.

- **Performance Highlights**: PROTECT 모델은 마우스, 미세조류 세포(Ostreococcus tauri), 인간 데이터셋을 사용해 정확성과 견고성을 검증하였으며, 서로 다른 뇌 영역과 치매 관련 샘플 간 시계열 리듬 차이를 탐색하는 데 성공했습니다. 특히, 알츠하이머(Alder) 환자와 대조군 간의 리듬 단백질의 차이를 밝혀내어, 치매와 관련된 생물학적 통찰을 제시할 가능성을 보여주었습니다.



### Derivation of effective gradient flow equations and dynamical truncation of training data in Deep Learning (https://arxiv.org/abs/2501.07400)
Comments:
          AMS Latex, 35 pages

- **What's New**: 이 논문에서는 Deep Learning(DL)에서 ReLU 활성화 함수에 대한 누적 편향(cumulative bias)과 가중치(weights)의 명시적인 방정식을 도출합니다. 특히, 입력층(input layer)에서 유클리드 비용(Euclidean cost)에 기초하여 경량화된 분류와 훈련된 데이터에 대한 역동적 시스템의 작용을 이해하려고 합니다. 이 연구는 지도 학습(supervised learning)의 해석 가능성 문제(interpretable AI) 접근을 확대하고 있습니다.

- **Technical Details**: 우리는 ReLU 활성화 기능을 선택하면서 균일한 차원의 DL 네트워크를 고려합니다. 가중치 행렬 Wₗ과 편향 벡터 bₗ을 정의하고, 훈련 데이터에서의 역동적 흐름 구조를 설명합니다. 이 과정에서, 누적 가중치와 편향의 그래디언트 흐름이 시간 의존적 자극을 유도하는 것으로 나타나며, 이로 인해 훈련 데이터의 기하학적 복잡성이 점진적으로 줄어듭니다.

- **Performance Highlights**: 이 연구의 결과는 훈련 데이터의 클러스터가 복잡성이 감소하는 '트렁케이션(truncation)' 과정에 있음을 보여줍니다. 적절한 조건 하에서, 이는 신경망의 붕괴(neural collapse) 현상에 해당하며, 손실 제로 훈련(zero loss training)을 이끌어내는 핵심 메커니즘으로 작용합니다. 이러한 발견은 딥러닝의 해석 가능성과 효율성을 한층 강화할 것으로 기대됩니다.



### Enhancing Retrieval-Augmented Generation: A Study of Best Practices (https://arxiv.org/abs/2501.07391)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 다양한 구성 요소와 설계 방식을 다룹니다. RAG 시스템은 언어 모델의 성능을 향상시키기 위해 새로운 쿼리 확장 기법, 다양한 검색 전략, 그리고 대조적 맥락 학습(Contrastive In-Context Learning)을 통합합니다. 다양한 실험을 통해 RAG 시스템의 개선 요소를 체계적으로 분석하여 최적의 성능을 위한 데이터를 제공합니다.

- **Technical Details**: RAG 시스템은 Language Model (LM)과 함께 외부 지식 소스를 통합하여 최신 정보와 정확한 응답을 생성합니다. 연구에서는 LLM의 크기, 프롬프트 디자인, 문서 청크 크기와 같은 요소들이 응답 품질에 미치는 영향을 분석하였습니다. 또한, 쿼리 확장 기법과 다국어 지식 기반을 적용한 최근의 접근 방법이 포함되어 있습니다.

- **Performance Highlights**: 논문에서 제안한 방법들은 RAG 시스템의 응답 품질을 개선하는 데 효과적임을 보여주었습니다. 이를 통해 언어 모델의 컨텍스트 풍부함과 생성 및 검색의 효율성을 균형 있게 맞추는 방법론을 제시합니다. 이 연구 결과는 다양한 실제 시나리오에서 더욱 적응 가능하고 고성능의 RAG 프레임워크 개발에 기여할 것으로 기대됩니다.



### Information-Theoretic Dual Memory System for Continual Learning (https://arxiv.org/abs/2501.07382)
Comments:
          35 pages, 9 figures, submitted to Knowledge-Based Systems

- **What's New**: 이번 논문에서는 동적 환경에서 지속적으로 새로운 지식을 습득하는 "continual learning" 능력을 집중적으로 다룹니다. 기존 메모리 시스템이 직면한 문제를 해결하기 위해, 두 개의 메모리 버퍼를 사용하는 혁신적인 "Information-Theoretic Dual Memory System (ITDMS)"를 제안합니다.

- **Technical Details**: ITDMS는 빠른 메모리 버퍼와 느린 메모리 버퍼로 구성됩니다. 빠른 메모리 버퍼는 일시적이고 새로운 샘플을 보존하며, 느린 메모리 버퍼는 중요하고 정보가 풍부한 샘플을 유지합니다. 또한, 효율적인 reservoir sampling 프로세스를 사용하여 빠른 메모리 버퍼를 최적화하며, 새로운 정보를 선택적으로 식별하고 저장하는 정보 이론적 메모리 최적화 전략도 도입했습니다.

- **Performance Highlights**: 제안된 방법론은 여러 가지 지속적 학습 실험을 통해 엄격하게 평가되었습니다. 실험 결과는 ITDMS 시스템의 효과성을 뒷받침하며, 새로운 데이터 습득을 위한 메모리 용량을 확보하는 절차에 대한 개선이 두드러지게 나타났습니다.



### Emergent effects of scaling on the functional hierarchies within large language models (https://arxiv.org/abs/2501.07359)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 계층적 기능이 다양한 방식으로 표현될 수 있음을 재조명합니다. 기존의 주장은 초기 층에서 문법을 처리하고, 중간 층에서는 의미를 해석하며, 후반 층은 정보를 통합한다는 것입니다. 연구진은 간단한 텍스트를 LLM에 입력하고, 이에 따른 활성화를 추출하는 방법을 사용하였습니다. 이로 인해 LLM의 작동 방식을 좀 더 세밀하게 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구는 Llama-3.2-3b 모델을 사용하여 각 층이 특정 정보를 인코딩하고 있는지를 조사했습니다. 이 과정에서 Support Vector Machines(SVM)와 Ridge Regression 기법을 활용하여 텍스트 라벨을 예측하였습니다. 초기 층(2-7)에서는 항목 수준의 의미가 가장 강하게 나타났고, 그 이후 두 항목 관계(8-12)와 네 항목 유사성(10-15)이 차례로 나타났습니다. 그러나 깊은 층에서는 정보의 압축이 일어나면서도 의미 있는 추상이 결여된 경우가 발견되었습니다.

- **Performance Highlights**: 대형 모델(예: Llama-3.3-70b-Instruct)에서는 추상화 수준에서 극적인 변동이 관찰되었습니다. 층이 깊어질수록 두 항목 관계와 네 항목 유사성이 처음에는 증가하다가 급격히 감소하고, 이어 다시 순간적으로 증가하는 특이한 패턴이 여러 실험에서 일관되게 나타났습니다. 이 외에도 인접한 층 간의 주의(attention) 메커니즘 간 협조가 나타나면서 각 층이 어떤 정보를 전문으로 다루는지 변동을 보였습니다. 이러한 결과는 LLM의 복잡성과 동적인 작동 방식을 보여줍니다.



### TempoGPT: Enhancing Temporal Reasoning via Quantizing Embedding (https://arxiv.org/abs/2501.07335)
- **What's New**: 이번 논문에서는 복잡한 추론 작업에서 기존의 멀티모달 언어 모델(Multi-modal Language Models)과 시간 시계열 데이터(Time Series Data)의 활용 한계를 극복하기 위해, 멀티모달 시간 시계열 데이터 구축 접근법 및 TempoGPT라는 새 모델을 제안합니다. TempoGPT는 시간 임베딩을 정량화(Quantize)하여 텍스트 정보와의 일관된 표현 패턴을 달성하고, 이를 통해 복잡한 추론 과제에서 성능을 향상시킵니다.

- **Technical Details**: TempoGPT는 변수-시스템 간의 관계를 분석하여 복잡한 추론 작업을 위한 멀티모달 데이터를 생성하며, 시간 임베딩을 일련의 이산 토큰으로 변환하는 과정을 포함합니다. 이후, 텍스트와 시간 토큰의 처리를 위한 공유 임베딩 레이어(Embedding Layer)를 설계하여, 두 정보 간의 표현을 일치시킴으로써 멀티모달 정렬(Multi-modal Alignment)을 높입니다. 이러한 접근은 시간 정보와 텍스트 정보 간의 의미 있는 연계를 가능하게 합니다.

- **Performance Highlights**: 실험을 통해 TempoGPT는 구축된 복잡한 시간 시계열 추론 작업에서 최첨단(State-of-the-art) 성능을 달성했음을 입증합니다. 또한, 시간 임베딩의 정량화가 멀티모달 정렬 및 TLM(Temporal Language Models)의 추론 능력을 향상시키는 데 효과적이라는 것을 정량적으로 입증하며, 이는 향후 다양한 응용 분야에서의 활용 가능성을 높입니다.



### Evaluation of Artificial Intelligence Methods for Lead Time Prediction in Non-Cycled Areas of Automotive Production (https://arxiv.org/abs/2501.07317)
- **What's New**: 이번 연구는 비주기 제어 생산 환경에서 인공지능(Artificial Intelligence) 방법을 적용하여 자동차 생산 공정에서의 리드 타임(lead time)을 예측하는 효과성을 살펴봅니다. 연구에서는 데이터 구조를 분석하고, One-Hot Encoding을 사용하여 전처리한 후, 감독 학습(supervised learning) 방법 중 회귀(regression)와 분류(classification) 기법을 평가했습니다. 특히, Gradient Boosting 알고리즘인 LightGBM을 포함해 XGBoost와 CatBoost가 우수한 성능을 보였다고 보고하였습니다. 최종적으로, LightGBM 알고리즘이 선택되었다고 언급하며, 상대적 예측 정확도가 90%까지 도달할 수 있다는 점을 강조했습니다.

- **Technical Details**: 연구에서 사용된 기술적 세부사항으로는 Python을 프로그래밍 언어로 사용하였고, Intel Core i9-12900KF 프로세서와 Nvidia RTX 3060 GPU를 갖춘 워크스테이션에서 알고리즘을 실행하였습니다. 데이터는 차량의 리드 타임에 영향을 미치는 다양한 특성을 고려하여 수집되고 저장되며, 각각의 차량에 대해 고유 식별 키(unique identification key)를 사용해 링크합니다. 또한, 공정 전후의 특성에 따라 알려진 특성과 공정 중에 기록된 특성을 구분하여 분석합니다.

- **Performance Highlights**: 초기 연구 결과는 AI 방법이 자동차 생산에서의 리드 타임 예측에 효과적으로 적용될 수 있음을 보여줍니다. 특히, 이전 연구 결과와 비교했을 때, LGBM 회귀에서는 22%의 예측 정확도를 보였으나, LGBM 분류에서는 65%에서 71%의 정확도를 기록했습니다. 하지만 추가적인 하이퍼파라미터 최적화가 필요하다는 점과 함께, AI 모델의 주기적인 재학습이 복잡한 생산 과정을 정확히 표현하는 데 중요하다는 점이 강조되었습니다.



### The Lessons of Developing Process Reward Models in Mathematical Reasoning (https://arxiv.org/abs/2501.07301)
- **What's New**: 이 논문에서는 Process Reward Models (PRMs)의 개발에서 발생하는 여러 챌린지를 다룹니다. 기존의 Monte Carlo (MC) 기반 데이터 합성과 LLM-as-a-judge 활용 방법이 PRM의 성능 평가에 있어 단점이 있음을 보여주었습니다. 특히, 기존 BoN (Best-of-N) 평가 방식의 편향과 이를 통한 과정 검증의 비효율성을 지적하고 있습니다.

- **Technical Details**: 연구에서는 MC 추정이 현재 단계의 정확성을 평가하기 위해 완료 모델에 의존하며, 이로 인한 부정확한 단계 검증이 PRM 성능에 부정적인 영향을 미친다고 설명합니다. 또한, PRM이 던지는 응답에 대한 허용 기준이 과도하게 높아 BoN 점수를 비정상적으로 증가시키는 경향이 있다는 점도 짚었습니다. 여기서는 새로운 합의 필터링 메커니즘을 개발하여 MC 추정과 LLM-as-a-judge의 통합을 시도하였습니다.

- **Performance Highlights**: 제안된 방법론은 BoN 평가와 단계별 오류 식별 작업에서 모델 성능과 데이터 효율성을 크게 향상시켰습니다. 최종적으로, 이 논문은 기존의 오픈 소스 PRM보다 더 우수한 성능을 지닌 최신 PRM을 발표하였으며, 향후 과정 감독 모델 개발에 대한 실용적인 지침을 제공합니다.



### Skip Mamba Diffusion for Monocular 3D Semantic Scene Completion (https://arxiv.org/abs/2501.07260)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 모노큘러 이미지를 기반으로 한 독특한 신경망 모델인 Skimba (Skip Mamba) 모델을 제안합니다. 이 모델은 상태 공간(state space) 모델링과 생성적 확산(diffusion) 모델링의 최신 기술 발전을 활용하여 놀라운 3D 의미 장면 완성을 달성합니다. 또한, 변분 오토인코더(VAE)의 조건부 잠재(latent) 공간에서 데이터를 처리하여 효율적이고 효과적인 장면 완성을 지원합니다.

- **Technical Details**: Skimba 모델은 세 개의 방향에서 서로 다른 딜레이션을 사용하는 트리플 맘바(triple Mamba) 구조를 포함하고 있으며, 이로 인해 긴 시퀀스 데이터의 효율적인 처리가 가능합니다. 또한, 이 네트워크는 다양한 블록을 통합하여 3D 장면 완성을 위한 충분한 문맥 정보를 제공합니다. 전반적으로, 제안된 방법은 기본적으로 조건부 VAE와 복합 데이터 전처리 과정을 결합하여 효과적인 분산 기반 노이즈 제거를 지원합니다.

- **Performance Highlights**: SemanticKITTI와 SSCBench-KITTI360 데이터셋에 대한 광범위한 평가 결과, 제안된 방법은 모노큘러 접근 방식 중에서 다른 기술들과 비교하여 월등한 성능을 나타냈습니다. 또한 스테레오 방법에 대해서도 경쟁력 있는 성능을 발휘함으로써 3D 의미 장면 완성 분야에서의 가능성을 보여주고 있습니다. 이 연구는 현재의 기술에 비해 높은 수준의 성능 향상을 달성하였음을 입증합니다.



### MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework (https://arxiv.org/abs/2501.07251)
Comments:
          Under Review of CVPR 2025

- **What's New**: 이번 논문에서는 Multi-Objective Set-based Attack(MOS Attack)이라는 새로운 적대적 공격 프레임워크를 제안합니다. MOS Attack은 여러 개의 손실 함수(loss function)를 활용하고 그들 간의 상호 관계를 자동으로 발견하는 것을 목표로 합니다. 이 접근법을 통해 기존 단일 목표 공격 방법의 한계를 극복하고 효과적인 적대적 예제 생성을 가능하게 합니다.

- **Technical Details**: MOS Attack은 집합 기반의 다중 목표 최적화 전략을 사용하여 추가적인 매개변수 없이도 여러 손실 함수를 통합합니다. 이 과정에서 각 손실 간의 시너지 효과(synergistic patterns)를 자동으로 탐색하여 적대적 공격을 더 강력하고 효율적으로 만듭니다. 특히 이 방법은 적대적 공격의 예제를 생성하기 위해 여러 손실 기능을 동시다발적으로 고려할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실험을 통해 MOS Attack이 기존의 단일 목표 공격 방법들보다 뛰어난 성과를 보여주는 것을 확인했습니다. 또한 신시아 형태를 통해 적은 수의 손실 함수로도 더 나은 결과를 달성할 수 있음을 보여주며, 이는 새로운 적대적 공격 전략의 가능성을 제시합니다.



### Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training (https://arxiv.org/abs/2501.07237)
- **What's New**: 이 논문은 Gradient Wavelet Transform (GWT)라는 새로운 메모리 효율적인 방법을 제안합니다. 이 방법은 gradient에 wavelet 변환을 적용하여 옵티마이저 상태를 유지하는 데 필요한 메모리를 크게 줄입니다. GWT는 메모리 집약적인 옵티마이저와 원활하게 통합될 수 있으며, 성능 저하 없이 효율적인 학습을 가능하게 합니다. 실험 결과, 기존의 메모리 효율적 옵티마이저 및 풀-랭크 방법들과 비교하여 주목할 만한 성과를 달성했습니다.

- **Technical Details**: GWT는 Haar 및 Daubechies-2 (DB2)와 같은 이산 wavelet을 필터로 사용하여 구현됩니다. 이 방법은 gradient의 차원을 효과적으로 압축하여 옵티마이저 상태를 저장하는 데 필요한 메모리를 줄입니다. 저자들은 GWT를 Adam 옵티마이저와 통합하여 LLaMA 모델의 사전 학습 및 파인 튜닝 작업에서 성능을 평가했습니다. 실험적으로, GWT는 특히 사전 훈련에서 메모리 사용을 67%까지 줄이는 동시에 학습 속도를 증가시켰습니다.

- **Performance Highlights**: GWT는 사전 훈련 및 파인 튜닝 작업에서 풀-랭크 옵티마이저들과 동등하거나 더 나은 성능을 보여줍니다. C4 데이터세트에서 LLaMA 모델을 사전 훈련하는 동안, GWT는 옵티마이저 메모리 사용량을 감소시키면서도 학습 속도를 증가시키는 효과를 입증했습니다. 또한, 하이퍼파라미터 조정의 영향을 탐구하는 ablation 연구를 통해 GWT의 성능을 극대화할 수 있는 방법을 제안하고 있습니다.



### Exploring the Use of Contrastive Language-Image Pre-Training for Human Posture Classification: Insights from Yoga Pose Analysis (https://arxiv.org/abs/2501.07221)
- **What's New**: 이 연구는 다양한 분야에서 필수적인 인간 자세 분류의 정확성을 높이기 위해 CLIP (Contrastive Language-Image Pretraining) 모델을 적용하는 새로운 접근 방식을 소개합니다. 특히, 요가의 자세 분류에 초점을 맞추어, 이 모델의 효과성을 실험적으로 검증하였습니다. 연구 결과, 제로샷(zero-shot) 접근법의 초기 한계를 극복하고 15,301장의 이미지(실제 및 합성)를 사용하여 훈련함으로써 유망한 결과를 도출했습니다.

- **Technical Details**: 연구에서는 82개의 클래스에 대해 CLIP 모델의 미세 조정(fine-tuning) 과정을 설명하며, 이미지 설명 구문, 모델 및 하이퍼파라미터 조정을 포함합니다. 최종 훈련된 CLIP 모델은 3,826장의 이미지에 대해 테스트하였으며, 85% 이상의 정확도로 이전 연구의 최첨단 성과를 약 6% 초과했습니다. 특히, YOLOv8 기반 모델을 미세 조정하는 데 필요한 시간보다 3.5배 짧습니다.

- **Performance Highlights**: 작은 데이터 세트의 다양한 자세(각 6개 자세, 1,301 및 401장의 훈련 이미지 포함)를 이용한 경우, 미세 조정된 모델은 각각 98.8%와 99.1%의 정확도를 달성했습니다. 또한, 20장만으로도 6개 클래스 데이터 세트에서 90%의 정확도를 생성할 수 있음을 보여주었습니다. 이 연구는 CLIP 모델이 요가 자세 분류 및 일반적인 인간 자세 분류에 효과적으로 활용될 수 있음을 입증합니다.



### Multi-face emotion detection for effective Human-Robot Interaction (https://arxiv.org/abs/2501.07213)
Comments:
          9 pages, 8 figures and 1 table. Accepted at the 17th International Conference on Agents and Artificial Intelligence (ICAART 2025), Porto, Portugal

- **What's New**: 이번 연구는 이동형 humanoid 로봇에 통합된 얼굴 감정 인식 인터페이스를 제안합니다. 이 인터페이스는 사용자 인터페이스에서 여러 개체의 실시간 감정을 표시할 수 있습니다. 다양한 deep neural network 모델을 개발하고 평가하여 높은 성능을 달성하였으며, 프로그램의 정확성과 메모리 용량 간의 균형을 고려하여 모바일 로봇에 실효성 있게 구현하였습니다.

- **Technical Details**: 로봇의 감정 인식 기술은 주로 얼굴 탐지(face detection)와 얼굴 감정 인식(face emotion recognition, FER) 시스템에 기반합니다. 본 연구에서는 Haarcascade 분류기를 얼굴 탐지에 사용하였으며, 이는 실시간 응용 프로그램에서 특히 유리합니다. 여러 사용자와의 상호작용을 지원하는 multi-face emotion recognition 기능도 포함되어 있어, 예를 들어 코미디 클럽과 같은 환경에서 실시간 피드백을 제공할 수 있습니다.

- **Performance Highlights**: 연구 결과 다양한 deep learning 기반의 얼굴 감정 인식 모델이 높은 정확도를 기록했으며, 특히 사용자 상호작용 분석을 통해 모델의 성능을 검토하였습니다. 이 모델을 통해 모바일 로봇에서 인간의 감정을 실시간으로 파악하고 적절히 반응할 수 있는 가능성이 demonstrated되었으며, 이는 로봇의 사회적 상호작용을 대폭 향상시킬 수 있는 기회를 제공합니다.



### Crowdsourced human-based computational approach for tagging peripheral blood smear sample images from Sickle Cell Disease patients using non-expert users (https://arxiv.org/abs/2501.07196)
- **What's New**: 본 논문에서는 겸상적혈구질환(Sickle Cell Disease, SCD) 환자의 말초혈액도말(Peripheral Blood Smear, PBS) 이미지를 분석하기 위한 인간 기반 계산(Human-Based Computation, HBC) 접근 방식을 제안합니다. Mechanical Turk 마이크로태스크 시장을 활용하여 PBS 이미지의 레이블 작업을 크라우드소싱했습니다. 연구 결과, Mechanical Turk 작업자들 간의 강력한 합의가 이루어질 경우, 전문가의 분석과 비교해 오차 확률이 매우 낮음을 발견했습니다. 이는 우리가 제안한 접근 방식이 PBS 이미지 데이터셋의 주석(annotation) 작업에 효과적으로 사용될 수 있음을 시사합니다.

- **Technical Details**: SCD는 전 세계에서 수백만 명에게 영향을 미치는 심각한 유전적 혈액 질환으로, 이 질환의 원인은 HBB 유전자의 변이로부터 비롯됩니다. PBS 이미지는 SCD 진단에 중요한 도구이지만, 노화된 아기 및 성인에서만 진단 가능한 제한이 있습니다. 이 과정은 노동 집약적이고 시간 소모적이어서 진단과 치료에 지연을 초래할 수 있습니다. 따라서 본 논문에서는 혈액 샘플 분석을 위한 자동화된 방법을 개발하고, 이미지 분석 및 기계 학습 알고리즘을 통해 겸상적혈구를 검출하고 수를 세는 방안을 제시합니다.

- **Performance Highlights**: 이번 연구에서는 MTurk를 활용한 PBS 이미지 분석이 비전문가로부터 수집한 데이터에 기반하여 유효성을 입증하였으며, 그 결과는 기존 전문가 분석과 유사한 수준의 정확도를 기록했습니다. 채택한 방법론은 Crowdsourcing 및 HBC 시스템의 이러한 응용이 다른 의료 이미지 분석에서도 긍정적인 결과를 가져올 수 있음을 보여줍니다. 향후 연구에서는 자동화된 방법론과의 통합 가능성을 탐구할 계획이며, 이는 SCD 진단을 위한 더 정확하고 신뢰할 수 있는 방법 개발로 이어질 잠재력을 가지고 있습니다.



### Generalizable Graph Neural Networks for Robust Power Grid Topology Contro (https://arxiv.org/abs/2501.07186)
- **What's New**: 이 논문은 그래프 신경망(GNN)을 활용한 전력망(grid) 토폴로지 제어를 위한 최초의 GNN 모델을 제안합니다. 또한, 기존의 동질적 그래프 표현이 가지고 있는 버스바 정보 비대칭 문제를 확인하고 이를 해결하기 위한 이종 그래프 표현을 제안합니다. 이 연구는 전력망 운영의 문제를 해결하기 위해 GNN의 깊은 활용 가능성을 탐구합니다.

- **Technical Details**: 이 연구는 GNN과 완전히 연결된 신경망(FCNN)을 기반으로 하는 모델을 비교하며, 이종 그래프 표현이 GNN 성능에 미치는 영향을 분석합니다. GNN은 그래프 구조를 반영하여 전력망을 모델링하는 데 적합한 다층 구조로 구성되어 있습니다. 또한, GNN 모델은 다양한 네트워크와 토폴로지에서 다중 레이블 이진 분류 모방 학습(task)으로 훈련됩니다.

- **Performance Highlights**: 이 연구에서 제안된 이종 GNN 모델은 주어진 기준 네트워크에서 이전 모델들보다 더 뛰어난 성능을 보였습니다. 특히, 이 모델은 FCNN보다도 아웃 오브 디스트리뷰션 네트워크에서 더 잘 일반화되는 모습을 보였습니다. 이러한 결과는 GNN을 활용한 토폴로지 제어 방법ologies의 발전 가능성을 보여줍니다.



### The Spoils of Algorithmic Collusion: Profit Allocation Among Asymmetric Firms (https://arxiv.org/abs/2501.07178)
- **What's New**: 이 논문은 반복적인 Cournot 듀오폴리 게임에서 독립적인 알고리즘이 공모(collude)할 경향을 연구합니다. 특히, 기업 간 비대칭(asymmetry)의 영향을 예측하기 위해 다양한 올리고폴리(oligopoly)와 협상(bargaining) 해법을 조사합니다. 이 연구는 알고리즘이 비대칭일 때 더 경쟁적인 결과를 만들어낸다는 것을 발견하였습니다. 이러한 결과는 알고리즘이 기업의 경영 결정을 주도하는 경우가 많아지는 현대 경제에서 중요한 의미를 갖습니다.

- **Technical Details**: 논문은 Q-learning 알고리즘을 이용하여 비대칭 기업 간 협상 문제를 해결하는 방법을 제안합니다. 연구에서는 대칭(baseline) 변형에서 시작하여 여섯 가지 다른 비대칭 정도의 시뮬레이션을 실시했으며, 두 가지 다른 설정을 통해 결과를 비교했습니다. 알고리즘이 효율적인 결과를 도출하지 않더라도, 비대칭성이 증가함에 따라 효율적인 기업이 추가 출력을 생산하는 것처럼 알고리즘의 행동을 관찰하였습니다.

- **Performance Highlights**: 결과적으로, 알고리즘은 모든 비대칭 정도에서 수익 분배가 Pareto frontier에 근접하게 이루어짐을 보여주었습니다. 이는 협상 해법으로서 equal relative gains가 결과를 잘 설명함을 시사합니다. Nash 균형은 비대칭성이 전체 수량에 미치는 영향을 과소평가하지만, 전체 복지 측면에서는 놀라울 정도로 정확한 예측을 제공합니다. 이 연구는 디지털 시장의 규제 및 경쟁 정책에 중요한 함의를 제공합니다.



### Anomalous Agreement: How to find the Ideal Number of Anomaly Classes in Correlated, Multivariate Time Series Data (https://arxiv.org/abs/2501.07172)
Comments:
          Acccepted at AAAI Workshop on AI for Time Series Analysis (AI4TS) 2025

- **What's New**: 이 논문에서는 Synchronized Anomaly Agreement Index (SAAI)를 도입하여 다변량 시계열에서 이상치(anomalies)의 동기성을 활용하여 클러스터 품질을 평가하는 방법을 제안합니다. 기존의 클러스터 품질 평가 방법인 Silhouette Score (SSC)와 달리, SAAI는 데이터에 대한 사전 지식을 반영하여 더 나은 성능을 보여줍니다. 이 방법은 이상 상태의 검출 및 분류를 더욱 효과적으로 만들 수 있습니다.

- **Technical Details**: SAAI는 다변량 시계열 데이터에서 발생하는 이상 상태의 동기화된 패턴을 분석하여 클러스터 품질을 높입니다. 이 지표는 클러스터 내의 일관성(cohesion)과 클러스터 간의 구분(separation)을 평가하지만, 데이터의 사전 지식도 고려하여 총 K개의 이상 클래스 수를 추정하는 정확도를 향상시킵니다. 이를 통해, 연구자들은 향후 비지도 학습(unsupervised learning) 기반의 시스템 모니터링에서 더 향상된 결과를 도출할 수 있습니다.

- **Performance Highlights**: SAAI의 활용으로 SSC와 비교했을 때 이상 클래스의 정확도가 0.23 향상되었으며, X-Means와 비교 시에는 0.32 향상되는 결과를 보였습니다. SAAI로 최적화된 클러스터는 SSC로 얻어진 클러스터보다 해석하기도 쉽습니다. 이러한 성과는 클러스터 품질 평가의 새로운 기준을 설정하며, 다양한 응용 분야에서 기여할 수 있는 가능성을 보여줍니다.



### Eye Sclera for Fair Face Image Quality Assessmen (https://arxiv.org/abs/2501.07158)
- **What's New**: 이 논문은 안구의 흰자부위(sclera)를 얼굴 이미지 품질 평가(FIQA)에서 중요한 품질 평가 영역으로 제안합니다. 기존 FIQA 요소들이 인종적 변동성에 민감한 반면, sclera는 인구통계적 요인에 영향을 받지 않아 공정한 FIQA 방법으로 기능할 수 있습니다. 이 연구는 sclera를 활용한 FIQA 접근법이 동적 범위, 노출 과다 및 부족을 평가하는 데 효과적이라는 것을 확인합니다.

- **Technical Details**: 이 논문에서는 III ISO/IEC 29794-5 표준을 기반으로 한 세 가지 피부 톤 관련 FIQA 조치를 분석하고, sclera 영역을 대안적인 평가 영역으로 제시합니다. sclera 영역은 피부 톤에 비감각적이므로 다양한 인구 통계적 그룹의 이미지에서도 일관된 결과를 나타냅니다. EDC 분석을 통해 sclera가 얼굴 이미지 품질을 측정하는 데 적합한 대안임을 입증합니다.

- **Performance Highlights**: 연구 결과에 따르면 sclera를 활용한 FIQA 방법은 기존의 얼굴 이미지 품질 평가 방법과 유사한 성능을 보이며, 피부 톤에 독립적인 평가를 가능하게 합니다. 공정한 FIQA의 필요성이 증가하는 가운데, sclera가 이 과제에 적합한 대안으로 등장함으로써 얼굴 인식 시스템의 신뢰성을 높일 수 있음을 보여줍니다.



### TIMRL: A Novel Meta-Reinforcement Learning Framework for Non-Stationary and Multi-Task Environments (https://arxiv.org/abs/2501.07146)
- **What's New**: 최근 메타 강화 학습( meta-reinforcement learning ) 분야에서 샘플 효율성을 개선하기 위한 새로운 알고리즘인 TIMRL( Task Inference with GMM and transformer for efficient Meta-RL ) 이 제안되었습니다. 특히 Gaussian mixture model(GMM)과 transformer 네트워크를 이용하여 작업 추론 모델(task inference model)을 구축함으로써 비정상 환경에서의 적응성을 높였습니다. 기존 방법들과 달리, TIMRL은 다양한 작업 환경에서 적은 수의 샘플만으로도 효율적으로 작업을 인식하고 분류하는 방식으로 설계되었습니다.

- **Technical Details**: 이 연구에서는 GMM 기반의 작업 추론 모델을 통해 비정상적인 환경 및 다중 작업 환경에서 메타 강화 학습의 효율성을 높이기 위해 여러 기술적인 구현이 이루어졌습니다. GMM을 사용하여 작업의 표현을 확장하고, transformer 네트워크를 통해 작업 분류를 명시적으로 인코딩합니다. 또한, 사전 처리 기법을 통해 다중 작업 환경에서 상태-액션 차원 공간을 정규화하고, 각 작업 분류를 개별 Gaussian 구성 요소로 모델링하여 복잡한 작업을 효과적으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 실험은 MuJoCo 벤치마크를 사용하여 비정상 및 다중 작업 환경에서 수행되었으며, 결과는 제안된 TIMRL 알고리즘이 탁월한 샘플 효율성 및 정확한 작업 분류 성능을 바탕으로 훌륭한 결과를 나타냄을 보여줍니다. 전반적으로, TIMRL은 기존의 메타 강화 학습 알고리즘에 비해 더 빠른 탐색 및 환경 적응을 가능하게 하여, 다중 작업 환경에서의 뛰어난 성능을 자랑합니다.



### AdaCS: Adaptive Normalization for Enhanced Code-Switching ASR (https://arxiv.org/abs/2501.07102)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 연구에서 제안하는 AdaCS(Ada Code Switching) 모델은 자동 음성 인식(ASR) 시스템에서 intra-sentential code-switching (CS) 문제를 다루기 위해 개발되었습니다. 기존의 ASR 시스템은 주로 단일 언어 데이터로 훈련되어 CS 해석에서 약점을 보였으나, AdaCS는 적응형 바이어스 주의(attention) 모듈(BAM)을 통합하여 이러한 문제를 해결합니다. 이 접근 방식은 미지의 도메인에서도 robust한 성능을 제공하며, 양질의 CS 음성 인식 통합에 기여하고 있습니다.

- **Technical Details**: AdaCS 모델은 인코더와 디코더 블록에 바이어스 주의 모듈을 통합하여 CS 구문 인식 및 정규화를 용이하게 합니다. 바이어스 주의 모듈(BAM)은 사전 정의된 바이어스 리스트를 입력으로 받아들여, 이를 기반으로 CS 구문에 대한 정보를 강화합니다. BAM은 바이어스 인코더, 순위 선택, 주의 서브모듈로 구성되어 있으며, 입력된 바이어스 리스트로부터 토큰 수준 표현 행렬 및 풀링된 벡터 표현을 생성하여 효과적으로 CS 구문을 처리합니다.

- **Performance Highlights**: AdaCS는 제안된 두 가지 테스트 세트에서 베트남어 CS ASR 정규화에서 기존 방법보다 56.2% 및 36.8%의 WER(Word Error Rate) 감소를 달성했습니다. 이는 미지의 CS 구문을 다양한 도메인에서 처리할 수 있는 인상적인 성능을 입증합니다. 실험 결과 및 데이터셋은 AdaCS 모델이 ASR의 낮은 자원 언어 환경에서도 효과적인 성능을 발휘함을 확인해주고 있으며, 향후 연구에 중요한 기준을 제공할 것입니다.



### Collaborative Learning for 3D Hand-Object Reconstruction and Compositional Action Recognition from Egocentric RGB Videos Using Superquadrics (https://arxiv.org/abs/2501.07100)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문에서는 egocentric 3D 손-객체 상호작용 데이터셋을 활용하여 손-객체 포즈 추정(hand-object pose estimation)과 행동 인식(action recognition)을 위한 통합 모델을 개발하고자 합니다. 기존 방법들이 3D bounding box를 사용하여 객체의 모양과 움직임을 표현하는 데 한계가 있었던 점을 해결하기 위해, superquadrics라는 대체 3D 객체 표현법을 도입하여 효과성을 입증합니다. 이 외에도, 우리는 학습 데이터와 테스트 데이터의 동사와 명사 조합이 겹치지 않는 더 도전적인 작업을 통해 행동의 구성 가능성을 연구합니다.

- **Technical Details**: 핵심적으로, 우리는 손-객체 포즈 추정과 상호작용 인식 간의 관계를 이해하기 위해 새로운 협업 학습 프레임워크를 제안합니다. 이 프레임워크는 겹치지 않는 훈련 및 테스트 분할을 통해 행동과 객체의 구성 가능성을 학습하는 데 도움을 줍니다. 또한, 기존의 3D 주석된 egocentric 손-객체 데이터셋을 확장하여 새로운 구성 분할을 도입하고, superquadrics를 중간 3D 객체 표현으로 사용함으로써, 이 방법이 동작 인식 및 손-객체 포즈 추정에 어떻게 유리한지를 입증합니다.

- **Performance Highlights**: 우리는 H2O와 FPHA 데이터셋을 사용하여 두 가지 공식 및 구성 설정에서 상태-of-the-art 성능을 달성했습니다. 우리의 메소드는 다양한 기존 방법들보다 손-객체 인식을 더욱 효과적으로 진행할 수 있도록 개선되었습니다. 실험 결과는 superquadrics와 협업 학습 프레임워크가 특히 구성된 행동 인식(compositional action recognition) 작업에서 유의미한 성능 향상을 가져온다는 것을 보여줍니다.



### Video Quality Assessment for Online Processing: From Spatial to Temporal Sampling (https://arxiv.org/abs/2501.07087)
- **What's New**: 이 논문은 비디오 품질 평가(VQA) 모델에서 스페이셜(Spatio) 및 템포럴(Temporal) 샘플링의 효과와 효율 간의 균형을 깊이 탐구합니다. 연구자들은 비디오 정보를 대폭 축소하여도 여전히 수용 가능한 성능을 유지할 수 있는 방법을 제안하며, 이러한 비디오 정보를 작은 크기로 연관짓습니다. 또한, 온라인 VQA 모델인 MGQA를 설계하여 복잡성을 줄이면서도 성능을 보장하고 있습니다.

- **Technical Details**: 제안된 연구는 비디오의 공간적 및 시간적 차원에서 정보 손실이 비디오 품질 예측에 미치는 영향에 대해 조사합니다. 특히, 기본 Gated Recurrent Units (GRU)와 같은 간단한 모델을 통해 공간 및 시간 특성을 추출하는 방법론을 채택하고 있습니다. 이를 통해, VQA 모델의 평균 계산 비용이 기존 모델에 비해 99.83% 감소하는 성과를 보여줍니다.

- **Performance Highlights**: 종합적인 실험을 통해 6개의 공개 비디오 품질 데이터베이스에서 제안된 샘플링 방법의 성능이 검증되었습니다. 연구 결과, 대부분의 비디오 정보를 제거하더라도 VQA 모델은 수용 가능한 성능을 유지할 수 있으며, 이로 인해 기존의 VQA 모델을 대체할 가능성을 보여줍니다. 특히, 온라인 모델은 처리하는 데이터 양을 크게 줄이는 방안을 제시하고 있습니다.



### Representation Learning of Point Cloud Upsampling in Global and Local Inputs (https://arxiv.org/abs/2501.07076)
- **What's New**: 본 연구에서는 3D 재구성을 위한 포인트 클라우드 업샘플링(point cloud upsampling)의 글로벌(global) 및 로컬(local) 수준에서의 영향을 조사합니다. 두 개의 인코더(encoder)를 이용하여 같은 포인트 클라우드 모델 객체의 글로벌 및 로컬 정보를 입력받고, 이를 융합(fusion)하여 업샘플링 디코더(decoder)에 보냅니다. 이 프레임워크는 최첨단(point cloud upsampling neural network) 모델에 적용할 수 있으며, 기존 방법들에 비해 향상된 효과를 보입니다.

- **Technical Details**: 연구에서는 포인트 클라우드의 희소성(sparsity) 및 노이즈(noise) 문제를 해결하기 위해 글로벌 및 로컬 입력의 사전 지식을 활용합니다. 또한, 심층 학습을 사용하는 오토인코더 기반 모델에서 실험을 진행하여 글로벌 및 로컬 입력의 해석 가능성을 증명합니다. 이 과정에서 Saliency Map을 이용해 두 입력의 차이와 병렬 학습(parallel training)의 유효성을 반영합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 기존 SOTA(state-of-the-art) 연구들에 비해 업샘플링 효과를 더욱 개선할 수 있음을 입증하였습니다. 아울러, 그래프 기반(graph-based) 방법 및 GAN(generative adversarial networks) 등을 활용하여 포인트 클라우드 작업에서의 딥러닝 적용을 용이하게 하였습니다. 이를 통해, 포인트 클라우드 데이터의 질을 높이고, 3D 재구성, 객체 탐지 및 분류, 로봇 작동에서의 활용 가능성을 높이는 데 기여할 수 있습니다.



### Logic Meets Magic: LLMs Cracking Smart Contract Vulnerabilities (https://arxiv.org/abs/2501.07058)
- **What's New**: 본 논문은 스마트 계약의 취약점 탐지에 대한 새로운 접근 방식을 제안합니다. 특히 최신 Solidity v0.8에서의 성능을 분석함으로써, 이전 연구들이 다룬 구버전(v0.4)과의 차별화를 이루었습니다. 또한, 다섯 가지 최신 LLM 모델을 활용하여 인식된 취약점의 발생률을 비교하고, 이에 따른 영향력을 평가하였습니다.

- **Technical Details**: 연구는 TOP200 및 Web3Bugs 데이터셋을 활용하여 최신 LLMs의 성능을 평가했습니다. 각 LLM의 성능은 false positive rate (FPR)를 기반으로 분석되었으며, 적절한 prompt 디자인을 통해 FPR을 60% 이상 감소시킬 수 있음을 입증했습니다. 또한, LLM이 새로운 라이브러리와 프레임워크의 변화를 인식하는 데 의존하면서 취약점 탐지 능력이 제한된다는 사실을 밝혔습니다.

- **Performance Highlights**: 연구 결과, 스마트 계약의 특정 취약점 탐지에 있어 recall rate가 13%로 저조하다는 점이 발견되었습니다. 이는 이전 버전(v0.4)와 비교할 때 성능 저하를 보여줍니다. 하지만 적절한 prompt 디자인을 통해 false positive rate을 크게 줄일 수 있다는 가능성을 제시하며, 이는 향후 LLM 기반의 스마트 계약 취약점 탐지 방안에 있어 중요한 인사이트로 작용할 것입니다.



### ACCon: Angle-Compensated Contrastive Regularizer for Deep Regression (https://arxiv.org/abs/2501.07045)
Comments:
          Accept by AAAI-2025 (The 39th Annual AAAI Conference on Artificial Intelligence)

- **What's New**: 이번 논문에서는 딥 회귀(Deep Regression)에서 발생하는 연속 레이블 간의 관계를 포착하기 위한 새로운 접근법을 제안합니다. 기존의 일반화된 회귀 모델과 달리, 각 레이블 간의 거리와 유사성 사이의 선형적인 음의 상관관계를 가정하고, 앵글 보정 대조 정규화기(Angle-Compensated Contrastive Regularizer)를 개발하였습니다. 이는 대조 학습(framework) 내에서 앵커와 음성 샘플들 간의 코사인 거리(cosine distance)를 조정하는 방식으로, 현재의 대조 학습 방법을 회귀 작업에 효과적으로 확장할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 방법은 다양한 대조 학습 기반 접근 방식을 통합 가능하며, 입력을 반-하이퍼스피어에 투사하여 레이블 간의 관계를 보존합니다. 이를 통해, 레이블 간의 거리에 따라 최적화된 유사성을 유지하면서, 대조 학습에서 필요한 양수 및 음수 샘플의 정의도 명확히 합니다. 실험을 통해 제안한 방식이 기존의 임밸런스 데이터에서의 성능을 크게 향상시키며, 비대칭성 문제를 효과적으로 해결함을 보여줍니다.

- **Performance Highlights**: 제안된 앵글 보정 대조 정규화기는 여러 데이터셋(특히 컴퓨터 비전 및 자연어 처리 분야)에서 경쟁력 있는 회귀 성능을 달성하였습니다. 실험 결과는 저자들이 주장하는 데이터 효율성(data efficiency)과 임밸런스 데이터 세트에 대한 효과성을 나타냅니다. 이 연구는 향후 딥 회귀 모델링에서 앵글 보정 방식을 도입해 실질적인 향상을 이끌어낼 잠재력을 가지고 있습니다.



### Neural Probabilistic Circuits: Enabling Compositional and Interpretable Predictions through Logical Reasoning (https://arxiv.org/abs/2501.07021)
- **What's New**: Neural Probabilistic Circuits (NPCs)라는 새로운 모델 아키텍처를 소개합니다. NPCs는 컴포지션(compositional) 및 해석 가능한 예측을 가능하게 하는 논리적 추론(logical reasoning)을 통해 본질적으로 투명한 모델입니다. 이 모델은 두 가지 모듈로 구성되어 있으며, 속성 인식 모델과 확률론적 회로를 기반으로 한 작업 예측기가 포함되어 있습니다.

- **Technical Details**: NPC는 속성 인식(attribute recognition) 모델과 작업 예측(task predictor) 모듈로 구성됩니다. 속성 인식 모델은 입력 이미지에 대한 확률 벡터를 생성하고, 이 벡터는 확률론적 회로(probabilistic circuit)로 구현된 작업 예측기의 입력으로 사용됩니다. 또한 NPC는 세 단계로 구성된 훈련 알고리즘을 통해 학습되며, 여기에는 속성 인식, 회로 구성 및 공동 최적화(joint optimization)가 포함됩니다.

- **Performance Highlights**: 실험 결과, NPC는 이미지 분류(image classification) 작업에서 해석력과 성능 간의 균형을 잘 유지하며, 엔드-투-엔드(End-to-End) 블랙박스 모델과 경쟁할 수 있는 성능을 보여줍니다. 또한, NPC는 세 가지 개념 기반 모델보다 우수한 성과를 보였으며, 다양한 유형의 설명을 제공하여 인간의 이해를 돕습니다.



### ViSoLex: An Open-Source Repository for Vietnamese Social Media Lexical Normalization (https://arxiv.org/abs/2501.07020)
Comments:
          The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: ViSoLex는 베트남의 소셜 미디어 텍스트에 대한 용어 정규화를 위한 오픈 소스 시스템으로, Non-Standard Word (NSW) Lookup 및 Lexical Normalization의 두 가지 핵심 서비스를 제공합니다. 이를 통해 사용자는 비표준 언어의 표준 형태를 검색하고, NSW가 포함된 텍스트를 표준화 할 수 있습니다. Pre-trained language models와 약한 감독 학습 기술을 통합하여 데이터 부족 문제를 해결하고 정확성과 효율성을 확보하였습니다.

- **Technical Details**: ViSoLex는 NSW Lookup과 Lexical Normalization라는 두 가지 서비스로 구성되어 있습니다. NSW Lookup 서비스는 사용자가 입력한 NSW를 적절한 표준 형태로 해석하게 하며, Lexical Normalization 서비스는 NSW를 포함한 문장을 표준 형태로 변환합니다. 이 시스템은 기존의 NSW 사전을 활용하고, 필요 시 OpenAI GPT-4o API를 통해 동적으로 학습하여 NSW의 해법을 제시합니다.

- **Performance Highlights**: ViSoLex는 멀티태스크 학습을 통해 NSW 감지 및 정규화를 동시에 처리하며, 각 작업의 손실을 조화를 이루게 하여 성능을 개선합니다. Rule Attention Network를 통해 약한 감독 규칙을 적용하고, NSW의 다양성과 진화하는 패턴에 적응할 수 있는 능력을 갖추었습니다. 이를 통해 베트남 소셜 미디어 텍스트의 정규화 정확도를 높이며, 사용자에게 보다 효율적인 텍스트 처리를 제공합니다.



### UNetVL: Enhancing 3D Medical Image Segmentation with Chebyshev KAN Powered Vision-LSTM (https://arxiv.org/abs/2501.07017)
- **What's New**: 이번 연구에서는 UNETVL(U-Net Vision-LSTM)이라는 새로운 아키텍처를 제안합니다. 이 모델은 Vision-LSTM(ViL)을 활용하여 의료 영상 분할(task)에서 메모리 기능과 확장성을 향상시킵니다. 또한, 복잡하고 장기적인 의존성 패턴을 처리하기 위해 효율적인 Chebyshev Kolmogorov-Arnold Networks(KAN)를 통합하여 성능을 개선했습니다. 이로 인해 기존 모델 UNETR 대비 Dice score에서 7.3%와 15.6%의 유의미한 향상을 보여주었습니다.

- **Technical Details**: UNETVL은 UNETR 구조를 기반으로 한 인코더-디코더 네트워크로, 지역적 특징(local features)과 장거리 컨텍스트를 모두 효율적으로 캡처합니다. 이 모델은 3D oversampled instance를 처리하며, 입력 데이터를 비오버랩(non-overlapping) 패치로 나누고 이러한 패치 토큰(patch tokens)을 여러 개의 Vision-LSTM 블록 쌍을 통해 처리하여 다양한 스케일의 중간 특징 표현을 생성합니다. ViL 블록에서는 mLSTM 계층이 양방향으로 패치 토큰을 처리하여 복잡한 데이터 관계를 캡처합니다.

- **Performance Highlights**: UNETVL은 ACDC와 AMOS2022 벤치마크 데이터셋에서 기존의 최신 기법(state-of-the-art)과 비교하여 뛰어난 성능을 발휘했습니다. 특히, 정량적인 평가 지표인 평균 Dice score에서 ACDC에서 7.3% 그리고 AMOS에서 15.6% 개선을 보였습니다. 또한, 각 컴포넌트가 모델 성능에 미치는 영향을 분석하기 위해 광범위한 ablation 연구가 진행되었습니다.



### A Multi-Modal Deep Learning Framework for Pan-Cancer Prognosis (https://arxiv.org/abs/2501.07016)
- **What's New**: 이 논문에서는 UMPSNet이라는 새로운 심층 학습 기반의 모델을 제안합니다. 기존의 예후(prognostic) 모델이 특정 데이터셋에 대해서만 활용되었던 반면, UMPSNet은 환자의 다양한 상태를 포괄적으로 이해하기 위해 설계되었습니다. 특히, 병리 이미지(histopathology images)와 유전자 발현 검사를 통합하는 것은 물론, 인구통계학적 정보, 암 종류, 치료 프로토콜, 진단 결과와 같은 네 가지 유형의 메타 데이터(meta data)를 활용하여 텍스트 템플릿을 생성합니다.

- **Technical Details**: UMPSNet은 각 유형의 데이터(모달 데이터)를 효과적으로 통합하기 위해 최적 운송(Optimal Transport, OT) 기반의 주의(attention) 메커니즘을 사용합니다. 또한, 가이드된 소프트 전문가 조합(Guided Mixture of Experts, GMoE) 메커니즘을 통해 여러 암 데이터셋 간의 분포 차이를 해결합니다. 이러한 기법들을 사용하여 환자 데이터의 다중 모달리티(multi-modality)를 통합하고 공동 훈련(joint training)을 수행하였습니다.

- **Performance Highlights**: UMPSNet은 모든 최신 기술(State-of-the-Art, SOTA) 방법들을 초월하는 성능을 발휘했습니다. 이 모델은 다수의 암 종류를 위한 단일 모델만으로도 효과성과 일반화 능력(generalization ability)을 입증하였습니다. 논문에서 제공한 코드로 인해 연구자들은 UMPSNet의 구현을 보다 쉽게 접근할 수 있게 되었습니다.



### AlgoRxplorers | Precision in Mutation -- Enhancing Drug Design with Advanced Protein Stability Prediction Tools (https://arxiv.org/abs/2501.07014)
- **What's New**: 이 연구는 단일 아미노산 돌연변이가 단백질 안정성에 미치는 영향을 예측하기 위해 딥 뉴럴 네트워크를 활용하는 접근 방식을 제안합니다. 특히, Transfer Learning(전이 학습)과 다양한 모델의 보완 정보를 융합하여 단백질 안정성의 복합적인 representation(표현)을 생성하는 데 초점을 맞추고 있습니다. 사전 훈련된 모델인 ThermoMPNN+가 ΔΔG 예측에서 가장 뛰어난 성과를 보였습니다. 이 방법은 단백질 역학에 대한 이해를 심화시키고 질병 연구 및 약물 발견에 기여할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구 방법론에서는 FireProtDB라는 데이터베이스를 활용하여 다양한 단백질 돌연변이와 구조 정보를 제공합니다. Latent transfusion(잠재적 전이) 기법을 통해 여러 딥러닝 모델에서 학습된 latent features(잠재 특징)를 융합합니다. 또한, 데이터 탐색적 분석을 통해 k-means clustering(군집화) 알고리즘을 사용하여 ΔΔG 예측을 위한 여러 모델 간의 성능 비교를 진행했습니다. 최종적으로 ThermoMPNN 모델을 기반으로 한 새로운 구조에서 예측 모델을 개선하고 단백질 안정성 변화 예측을 정확하게 수행하고자 합니다.

- **Performance Highlights**: 제안하는 모델은 기존 방법들보다 제한된 훈련 예제에서도 뛰어난 성능을 발휘하며, 약물 발견 시간과 비용을 크게 줄일 수 있는 가능성을 보여줍니다. 실험 결과, ThermoMPNN 모델과 ESM 임베딩의 조합이 ΔΔG 예측에서 매우 유망한 결과를 나타냈습니다. 다층 퍼셉트론과 경량 주의 메커니즘을 통해 입력 데이터를 정제하여 돌연변이로 인한 안정성 변화를 예측함으로써 정확성과 신뢰성을 동시에 목표로 합니다.



### Likelihood Training of Cascaded Diffusion Models via Hierarchical Volume-preserving Maps (https://arxiv.org/abs/2501.06999)
Comments:
          Spotlight at ICLR 2024

- **What's New**: 이 논문에서는 새로운 다중 스케일 생성 모델을 제안하며, 이를 통해 고해상도 샘플을 생성할 수 있을 뿐만 아니라, 이들이 신뢰할 수 있는 likelihood 모델이 될 수 있음을 보여줍니다. 기존의 다중 스케일 모델에서 likelihood 함수 계산 시 발생하는 어려움을 해결하기 위해, 저자들은 계층적 볼륨 보존 변환(hierarchical volume-preserving maps)이라는 새로운 접근 방식을 도입하고 있습니다.

- **Technical Details**: 저자들은 각 중간 스케일이 likelihood 평가에서 간단히 제외될 수 없는 변수들을 도입한다는 문제를 해결하기 위해, latent 공간에서의 diffusion 과정을 모델링합니다. 여기서 계층적 볼륨 보존 변환을 활용하여 분산 데이터의 계층적 구조를 유지하면서도 local distortion을 피할 수 있습니다. 특히, 라플라시안 피라미드(Laplacian pyramids)와 웨이브렛 변환(wavelet transforms)이 이러한 변환의 대표적인 예로 제시됩니다.

- **Performance Highlights**: 이 연구에서 제안된 다중 스케일 likelihood 모델은 밀도 추정(density estimation), 무손실 압축(lossless compression), 및 분포 외 탐지(out-of-distribution detection)와 같은 다양한 벤치마크에서 기존의 최첨단 결과를 크게 개선하였습니다. 이 모델은 일관된 성능 향상과 함께, EMD(Earth Mover’s Distance) 기반의 이론적 검증을 통해 인간의 지각적 유사성을 반영하는 속성을 활용하고 있습니다.



### Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning (https://arxiv.org/abs/2501.06994)
- **What's New**: 이 연구에서는 로봇이 자율적으로 일상적인 작업을 완료할 수 있도록 하기 위한 새로운 접근 방식을 제안합니다. 기존의 Imitation Learning (IL) 방식의 한계를 극복하기 위해, 우리는 동작을 짧은 시간 지평선의 2D 궤적으로 표현하는 방법을 도입하였습니다. Motion Track Policy (MT-π𝜋	extit{pi})는 이 매우 적은 양의 데이터만으로 임무를 성공적으로 수행할 수 있는 새로운 방식으로, 인식된 2D 궤적을 통해 로봇과 인간의 동작을 통합합니다.

- **Technical Details**: MT-π𝜋	extit{pi} 정책은 이미지 관찰을 입력으로 받고 동작을 궤적 형태로 출력합니다. 이 정책은 인간 손과 로봇 말단 장치의 동작을 2D 궤적으로 변환하여, 필요한 데이터는 약 10분의 인간 비디오와 몇 개의 로봇 시연으로 최소화합니다. 테스트 시간 동안 두 개의 카메라 뷰에서 궤적을 예측하고, 다중 뷰 기하학을 통해 6DoF(6 Degrees of Freedom) 궤적을 재구성합니다.

- **Performance Highlights**: MT-π𝜋	extit{pi}는 4개의 실제 작업에서 평균 86.5%의 성공률을 달성하였으며, 이는 기존 IL 방식에 비해 40% 개선된 수치입니다. 이 정책은 인간 비디오에서만 포착된 새로운 시나리오로도 일반화하여, 오픈 소스 데이터셋으로 배포 가능하다는 점이 특징입니다. 연구팀은 코드와 비디오를 웹사이트에서 제공하여 재현 가능한 훈련과 실제 환경에서의 배포를 지원합니다.



### Graph Contrastive Learning on Multi-label Classification for Recommendations (https://arxiv.org/abs/2501.06985)
Comments:
          Preprint. 10 figures, 5 tables

- **What's New**: 이번 논문은 다중 레이블 분류를 위한 그래프 대조 학습(Graph Contrastive Learning for Multi-label Classification, MCGCL) 모델을 제안하여 추천 시스템의 효과성을 개선하려고 합니다. MCGCL은 두 개의 훈련 단계를 포함하고 있으며, 기본 작업과 하위 작업으로 구성됩니다. 이를 통해 사용자-아이템 간의 관계를 보다 잘 캡처할 수 있습니다.

- **Technical Details**: MCGCL은 사용자-아이템 이분 그래프를 하나의 전체 그래프와 두 개의 동질 하위 그래프로 분해하여 작업을 수행합니다. 첫 번째 단계에서는 전체 그래프에서의 표현 학습에 중점을 두고, 두 번째 단계에서는 이전 단계에서 얻은 표현을 바탕으로 동질 하위 그래프의 표현을 학습합니다. 이 구조는 다중 레이블 분류에서의 대조 학습의 적합성을 탐구하는데 중요한 기여를 합니다.

- **Performance Highlights**: 실험은 Amazon Reviews 데이터세트를 사용하여 진행되었으며, MCGCL은 다중 레이블 및 이진 레이블 분류 작업에서 다른 방법들보다 우수한 성과를 보였습니다. 추가적인 하이퍼파라미터 분석을 통해 MCGCL의 강력한 일반화 능력이 확인되었습니다. 이는 추천 시스템의 성능 향상 잠재력을 보여줍니다.



### Data Enrichment Work and AI Labor in Latin America and the Caribbean (https://arxiv.org/abs/2501.06981)
Comments:
          17 pages of content with 2 figures

- **What's New**: 이 연구는 라틴 아메리카 및 카리브해 지역의 크라우드워커에 대한 기존 연구의 빈틈을 메우기 위해 실시되었습니다. 100명의 크라우드워커를 대상으로 하는 설문조사를 통해, 이들 크라우드워커는 디지털 노동에 대한 자부심을 느끼고 있으며, 가족으로부터 존중받는다고 밝혔습니다. 특히, 크라우드 작업이 재정적 및 전문적 독립을 위한 발판으로 여겨진다는 점이 주목할 만합니다.

- **Technical Details**: 이 논문은 Toloka 플랫폼에서 활동하는 크라우드워커들을 대상으로 진행된 연구로, 라틴 아메리카와 카리브해 지역의 크라우드워커의 경험과 관점을 분석하였습니다. 정량적 데이터에서는 이 지역의 크라우드워커들이 인도의 동료들과 비교할 때 상대적으로 경제적 지위가 높고, 결혼하지 않은 이들의 비율이 더 높다는 점을 발견했습니다. 정성적 데이터에서는 참가자들이 크라우드 업무에 대해 자부심과 존중을 경험했음을 나타냈습니다.

- **Performance Highlights**: 연구에 따르면, 라틴 아메리카의 크라우드워커는 재정적 독립을 위한 초기 단계로서 크라우드 작업을 인식하고 있으며, 이는 미국의 크라우드워커들이 노동의 불안정성으로 여기고 있는 것과는 대조적입니다. 또한, 이들은 서로 연결되기를 원하나, 플랫폼의 구조가 오히려 동료들과의 협업을 방해한다고 느끼고 있습니다. 이러한 결과는 라틴 아메리카 크라우드워커들을 지원하기 위한 맞춤형 도구와 인터페이스의 필요성을 강조합니다.



### Combining LLM decision and RL action selection to improve RL policy for adaptive interventions (https://arxiv.org/abs/2501.06980)
- **What's New**: 이 논문은 강화 학습(Reinforcement Learning, RL)과 대규모 언어 모델(Large Language Models, LLMs)을 결합하여 개인 맞춤형 건강 개입을 실시간으로 업데이트하는 새로운 접근 방식을 소개합니다. 사용자의 텍스트 기반 선호를 즉시 반영하여 RL의 정책을 개선하고 개인화(personalization)를 가속화하는 것이 목표입니다. 이를 위해 연구자들은 LLM을 사용하여 RL 시스템 내에서 행동 선택(action selection)을 조정하고 있습니다.

- **Technical Details**: 제안된 하이브리드 방법은 'LLM+TS'라고 불리며, LLM의 응답을 RL 행동 선택에 통합하여 RL 정책을 개선하는 것입니다. 이 방법은 Thompson Sampling (TS) 기술을 활용하여 RL 에이전트의 역할을 수행하며, 사용자 선호를 고려한 다양한 프롬프트 전략과 행동 선택 전략을 탐색합니다. 시뮬레이션 환경 'StepCountJITAI for LLM'을 통해 사용자 선호를 생성하고, 행동 역학에 영향을 미치는 제약 조건을 모델링합니다.

- **Performance Highlights**: 이 연구의 결과, 제안된 방법은 텍스트 기반 사용자 선호를 고려하면서 RL 정책을 개선할 수 있음을 보였습니다. 이는 개인 맞춤형 건강 개입에서의 효과를 증가시키며, RL 알고리즘 개발에 대한 새로운 가능성을 열어줍니다. 또한 StepCountJITAI는 기존의 연구에서 탐구되지 않았던 혁신적인 시뮬레이션 환경으로, 후속 연구 및 응용에 있어 중요한 기초가 될 것입니다.



### Kolmogorov-Arnold Recurrent Network for Short Term Load Forecasting Across Diverse Consumers (https://arxiv.org/abs/2501.06965)
- **What's New**: 본 논문은 Kolmogorov-Arnold Recurrent Network (KARN)라는 새로운 부하 예측(load forecasting) 접근 방식을 제안합니다. KARN은 Kolmogorov-Arnold Networks의 유연성(flexibility)과 RNN의 시간적 모델링(temporal modeling) 능력을 결합하여 개발되었습니다. 이 모델은 다양한 소비자 유형에 적응할 수 있도록 설계되어 있습니다.

- **Technical Details**: KARN은 학습 가능한 시간적 스플라인 함수(learnable temporal spline functions)와 엣지 기반 활성화(edge-based activations)를 사용합니다. 이를 통해 비선형 관계(non-linear relationships)를 더 효과적으로 모델링할 수 있으며, 이는 전력 소비의 복잡하고 갑작스러운 변화를 포착하는 데 유리합니다. 다양한 실제 데이터셋의 철저한 평가를 통해 KARN의 성능을 검증하였습니다.

- **Performance Highlights**: KARN은 학생 기숙사, 단독 주택, 전기차 충전이 있는 주택, 타운하우스, 산업 건물 등 다양한 소비자 카테고리에서 전통적인 Vanilla RNNs보다 일관되게 우수한 성능을 보였습니다. 특히 6개의 건물에서는 LSTM 및 Gated Recurrent Units (GRUs)보다 뛰어난 결과를 나타냈습니다. 이러한 결과는 KARN이 다양한 에너지 관리 시나리오에서 부하 예측을 향상시키는 유망한 도구임을 입증합니다.



### Generative Artificial Intelligence-Supported Pentesting: A Comparison between Claude Opus, GPT-4, and Copilo (https://arxiv.org/abs/2501.06963)
- **What's New**: 최근 생성 인공지능(Generative Artificial Intelligence, GenAI)의 출현은 우리 사회에 큰 변화를 가져왔습니다. 특히 사이버 보안(cybersecurity) 분야에서의 응용이 주목받고 있으며, 그 중에서도 침투 테스트(penetration testing, pentesting)와 윤리적 해킹의 과정에서 GenAI의 잠재력이 크다는 점이 강조됩니다. 본 논문에서는 Claude Opus, ChatGPT의 GPT-4 및 Copilot 같은 주요 GenAI 도구들이 침투 테스트 실행 표준(Penetration Testing Execution Standard, PTES)에 따라 침투 테스트 과정에서 어떻게 도움이 되는지를 분석했습니다.

- **Technical Details**: 논문에서는 PTES의 모든 단계에서 각 도구를 평가하며, 통제된 가상 환경에서 진행된 실험을 통해 결과를 도출했습니다. 각 도구는 침투 테스트 자동화를 완전히 대체할 수는 없지만, 특정 작업에서 효율성과 효과성을 높여주는 중요한 지원을 제공한다는 것을 발견했습니다. 특히, 연구 결과는 Claude Opus가 실험 시나리오에서 지속적으로 다른 도구들보다 우수한 성능을 보였음을 보여줍니다.

- **Performance Highlights**: 본 연구는 GenAI 도구들이 침투 테스트의 각 요소에서 어떻게 효과적으로 사용되었는지를 설명하며, 이러한 도구들이 에티컬 해커들에게 실질적인 지원을 줄 수 있는 가능성을 보여줍니다. 연구기간 동안 도출된 발견은 사이버 보안의 다양한 영역에서 GenAI 도구의 활용을 확장할 수 있는 여지를 나타내며, 향후 연구와 개발에 대한 방향성을 제시합니다.



### Compact Bayesian Neural Networks via pruned MCMC sampling (https://arxiv.org/abs/2501.06962)
Comments:
          22 pages, 11 figures

- **What's New**: 본 연구는 MCMC 샘플링과 네트워크 프루닝(network pruning)을 결합하여 중복 파라미터를 제거한 컴팩트한 확률 모델을 만드는 방법을 제안합니다. 이 연구는 프루닝 후에도 모델의 훈련 및 일반화 성능을 유지하면서 불확실성을 추정할 수 있는 컴팩트 BNN을 보장합니다. 이를 통해 실제 애플리케이션을 위해 불확실성을 추정하는 컴팩트한 BNN 모델을 개발할 수 있는 길을 엽니다.

- **Technical Details**: 바이시안 신경망(Bayesian Neural Networks, BNNs)은 불확실성 추정에 있어 강력한 도구입니다. 본 연구는 MCMC 샘플링과 매개변수의 프루닝을 결합하여 중요한 파라미터만을 남기고 과도한 매개변수를 제거하는 방식을 사용합니다. 연구에서 사용된 MCMC 기반 프루닝 전략은 회귀 및 분류 문제에서 효과성을 평가하기 위해 선택된 벤치마크 데이터셋을 사용하여 분석되었습니다.

- **Performance Highlights**: 본 연구에서는 네트워크 크기를 75% 이상 줄이면서 일반화 성능을 유지할 수 있음을 보여줍니다. MCMC를 사용한 BNN 훈련과 프루닝이 실제 데이터셋에서 매우 효과적임을 나타내며, 복잡한 실제 데이터셋에서도 견고한 성능을 발휘합니다. 또한, 프루닝 후 성능 손실 없이는 모델을 정교화하는 방법이 논의되었습니다.



### Patent Novelty Assessment Accelerating Innovation and Patent Prosecution (https://arxiv.org/abs/2501.06956)
- **What's New**: 이번 보고서는 특허를 통해 지적 재산권을 보호하고 연구 개발 투자 촉진을 위한 혁신적인 Patent Novelty Assessment and Claim Generation System을 소개합니다. 이 시스템은 지적 재산의 발명적 측면을 분석하고, 방대한 특허 청구 데이터에 대한 접근을 간소화하도록 설계되었습니다. 특히 중국 특허의 특수성을 고려하여 대학생과 연구자들이 특허 청구의 복잡성을 이해하고 탐색할 수 있도록 직관적인 플랫폼을 제공합니다.

- **Technical Details**: 전통적인 분석 시스템과 달리, 본 시스템은 독점적인 Chinese API를 활용하여 정확성과 적합성을 높입니다. 주된 도전은 다양한 특허 청구에 대한 접근성과 이해의 복잡성으로, 이는 기존 아이디어에 대한 효과적인 혁신을 저해할 수 있습니다. 우리의 솔루션은 이러한 장벽을 극복하고 중국 특허 환경의 세부 사항에 맞춰 조정된 종합적인 청구 정보 접근을 제공합니다.

- **Performance Highlights**: 이 혁신적인 플랫폼은 사용자가 특허 청구 정보에 효율적으로 접근할 수 있도록 설계되어, 지적 재산 분야에서의 정보 탐색과 혁신을 촉진하는 것을 목표로 하고 있습니다. 이는 개별 대학을 넘어 연구 개발에 도움이 되는 환경을 조성하고, 학문 공동체 내에서 특허 개념에 대한 이해를 심화하는 데 기여할 것입니다.



### Why are we living the age of AI applications right now? The long innovation path from AI's birth to a child's bedtime magic (https://arxiv.org/abs/2501.06929)
Comments:
          14 pages, 8 figures

- **What's New**: 최근 연구에서는 네 살짜리 어린이가 AI 도구를 통해 자기가 원하는 내용의 자극적이고 그래픽이 포함된 동화를 생성하는 과정을 다루고 있습니다. 이는 AI의 발전과 그 기술이 우리 사회에서 어떻게 광범위하게 활용되고 있는지를 보여주는 긍정적인 사례입니다. 논문은 AI의 역사적 발전을 추적하며, 현대 AI 애플리케이션이 가능하게 만든 주요 기술 혁신 요소를 강조합니다.

- **Technical Details**: 본 논문은 OpenAI 소프트웨어를 사용하여 Android 디바이스에서 진행된 연구 방법론을 소개합니다. 연구 질문을 체계적으로 분석하고 각 기술의 주요 요소를 파악하여 그 기반이 되는 과학적 혁신을 밝혀내는 과정을 진행하며, 학술 검색 엔진과 웹 검색을 활용해 관련 기술의 역사적 발전을 추적합니다. 이 과정에서 generative AI가 텍스트의 문법적 정확성을 보장하며 연구 결과의 깊이와 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: AI 기술의 발전은 단순한 작업을 넘어 ChatGPT와 같은 범용적인 애플리케이션을 가능하게 했습니다. 논문에서는 CPU와 GPU의 발전이 AI 모델 훈련을 가능하게 하고, 스마트폰과 클라우드 인프라의 보급이 데이터 접근성을 높였으며, 최신 AI 연구의 성과들이 이러한 기술을 혁신적으로 변화시켰음을 설명합니다. 이러한 발전 사항들은 AI의 현재 능력이 가능하게 된 중요한 이정표로서, 사회에 깊은 의미를 갖습니다.



### MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis (https://arxiv.org/abs/2501.06887)
Comments:
          Accepted to 2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)

- **What's New**: 이 연구는 피부암 진단 분야에서 CLIP(Contrastive Language-Image Pretraining) 모델을 활용함으로써 의사들이 AI의 결정 과정을 이해하고 신뢰할 수 있도록 하는 방법을 제안합니다. 특히, MedGrad E-CLIP이라는 새로운 설명 가능성(explainability) 방법을 도입하여, 복잡한 의료 이미지를 위한 가중 엔트로피 메커니즘을 통합합니다. 이 접근 방식은 특정 진단 설명과 연결된 이미지의 중요한 영역을 강조하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 PH² 및 Derm7pt 데이터셋을 사용하여 dermoscopic(피부 경화의) 구조 기준이 포함된 이미지를 연구했습니다. CLIP은 이미지 인코더와 텍스트 인코더로 구성되어 있으며, 이미지를 통해 진단 기준과의 관계를 학습합니다. 샘플 이미지와 설명의 커플을 통해, CLIP의 가중치를 학습하여 새로운 이미지-텍스트 쌍을 분류하는 방식으로 작동합니다.

- **Performance Highlights**: 제안된 메서드는 이미지에서 특징을 추출하고 이를 텍스트 기준과 일치시켜 피부 병변을 분류합니다. 이를 통해 기존의 진단 방법보다 향상된 신뢰성 및 투명성을 제공하며, AI 기반 진단 시스템에서 의사들 사이의 신뢰를 증진시키는 데 기여합니다. 이와 같은 성과는 깊은 신경망(Deep Neural Networks)의 의료 분석 적용에 있어 중요한 진전을 보여줍니다.



### Defect Detection Network In PCB Circuit Devices Based on GAN Enhanced YOLOv11 (https://arxiv.org/abs/2501.06879)
- **What's New**: 이번 연구는 생성적 적대 신경망(GAN)으로 개선된 YOLOv11 모델을 활용하여 인쇄 회로 기판(PCB)에서의 표면 결함 탐지 방법을 제안합니다. 연구는 결함 유형으로 결핍 구멍(missing hole), 쥐 물림(rat bite), 개방 회로(open circuit), 단락(short circuit), 버(burr), 가상 용접(virtual welding) 등 여섯 가지를 집중적으로 분석합니다.

- **Technical Details**: GAN을 사용해 합성 결함 이미지를 생성함으로써 데이터셋을 다양하고 현실감 있게 확장하여 모델의 일반화 능력을 향상시킵니다. 특히 버(burr)와 같은 복잡하거나 드문 결함들에 대한 탐지 성능이 강화됨에 따라, 향상된 YOLOv11 모델의 PCB 결함 데이터셋 검증에서 정확도(accuracy), 재현율(recall), 강인성(robustness)이 크게 개선되었습니다.

- **Performance Highlights**: 연구 결과는 복잡한 환경이나 작은 목표물에 대한 결함 처리에서 눈에 띄는 성과를 보여줍니다. 전자 설계 자동화(EDA) 분야의 효율적인 결함 탐지는 고품질 PCB 제조를 보장하는 중요한 단계라는 점에서, GAN 기반 데이터 증대 및 최적화된 탐지 아키텍처의 중요성을 강조합니다.



### Transfer Learning of Tabular Data by Finetuning Large Language Models (https://arxiv.org/abs/2501.06863)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)을 활용하여 테이블 데이터 분류에서의 효과적인 전이 학습 방법을 탐구합니다. 기존의 딥러닝 기법들이 테이블 데이터에서의 성과가 미흡했던 이유는 이질적인 특성과 샘플 수의 제한으로 지적됩니다. 제안된 방법은 LLM을 최적화하여 적은 수의 샘플로도 성능을 극대화할 수 있도록 해 새로운 가능성을 열어 줍니다.

- **Technical Details**: LLM의 전이 학습은 기존의 LLM의 지식을 활용하여 더 나은 분류 성능을 제공하는 방법입니다. 본 연구에서는 데이터 직렬화(data serialization) 단계를 통해 테이블 데이터를 텍스트 프롬프트로 변환하고, DistilGPT2 모델을 사용하여 이를 분류 작업에 적용합니다. 이 과정에서 메타데이터(metadata)를 포함한 텍스트 프롬프트를 통해 LLM의 성능을 개선합니다.

- **Performance Highlights**: 제안된 LLM 최적화 방법은 기존의 머신러닝 및 딥러닝 기법에 비해 상대적으로 적은 계산 비용으로 경쟁력 있는 분류 성능을 보여줍니다. 특히, 10개의 벤치마크 데이터셋에서 특성이 10개 미만인 경우에도 뛰어난 성능을 발휘하며, 이를 통해 전이 학습의 가능성을 더욱 확장합니다.



### LarvSeg: Exploring Image Classification Data For Large Vocabulary Semantic Segmentation via Category-wise Attentive Classifier (https://arxiv.org/abs/2501.06862)
Comments:
          PRCV 2024

- **What's New**: 이번 논문에서는 LarvSeg라는 새로운 대용량 어휘 의미 분할 프레임워크를 제안합니다. LarvSeg는 이미지 분류 데이터를 활용하여 심층 학습 모델의 의미 분할 능력을 확장하도록 설계되었습니다. 이 접근법은 기존의 어휘 기반 방법에서 발생하는 문제점을 개선하여 학습 데이터의 범위를 넓힙니다.

- **Technical Details**: 이 프레임워크는 이미지 수준의 감독을 픽셀 수준의 비율에 통합하여 더 많은 범주의 세분화를 가능하게 합니다. 특히 카테고리별 주의 분류기(category-wise attentive classifier)를 도입하여 특정 영역에서의 감독을 구현함으로써 모델의 성능을 향상시킵니다. LarvSeg는 COCO-Stuff와 같은 제한된 어휘 세분화 데이터와 ImageNet21K와 같은 대규모 이미지 분류 데이터셋을 결합해 학습합니다.

- **Performance Highlights**: 실험 결과, LarvSeg는 이전의 공개 어휘 모델보다 월등한 성능을 보였으며, 특히 마스크 레이블이 없는 범주에서 성능이 현저하게 개선되었습니다. 21K 카테고리에 대한 의미 분할 모델이 최초로 제공되며, 이를 통해 다양한 적용 가능성이 제시됩니다.



### A Comprehensive Evaluation of Large Language Models on Mental Illnesses in Arabic Contex (https://arxiv.org/abs/2501.06859)
- **What's New**: 이번 연구는 아랍 세계에서 정신 건강 문제에 대한 접근 가능한 진단 및 개입 도구의 필요성을 강조하며, 대규모 언어 모델(LLMs)을 활용하는 새로운 접근 방식을 제시합니다. 8개의 다양한 LLM을 평가하여 이들이 아랍어의 문화적 맥락에서 정신 건강 데이터셋에 어떻게 적용되는지를 조사했습니다.

- **Technical Details**: 연구에서는 AraDepSu, Dreaddit, MedMCQA와 같은 다양한 정신 건강 데이터셋을 사용하여 프롬프트 디자인, 언어 설정(native Arabic vs. translated English) 및 few-shot prompting이 진단 성능에 미치는 영향을 분석했습니다. 특히, 구조적 프롬프트가 덜 구조화된 변형보다 다중 클래스 데이터셋에서 평균 14.5% 더 높은 성과를 보였습니다.

- **Performance Highlights**: 모델 선택이 성능에 중대한 영향을 미쳤으며, Phi-3.5 MoE는 이진 분류에서 균형 잡힌 정확도가 뛰어난 반면, Mistral NeMo는 심각도 예측 작업에서 평균 절대 오차에서 우수한 성능을 나타냈습니다. 또한, few-shot prompting은 GPT-4o Mini에서 다중 클래스 분류 정확도를 평균 1.58 배 높이며 일관된 성능 향상을 보였습니다.



### SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training (https://arxiv.org/abs/2501.06842)
- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs) 훈련에서 발생하는 그래디언트 스파이크(gradient spikes)의 문제를 심층적으로 분석합니다. 이러한 스파이크는 훈련을 방해하고 비효율성을 가중시키는 주요 원인으로 지적됩니다. 이를 해결하기 위해 'Spike-Aware Adam with Momentum Reset SPAM'이라는 새로운 최적화 알고리즘을 제안하여 그래디언트 스파이크를 효과적으로 감소시켜 훈련 안정성을 향상시킵니다.

- **Technical Details**: 저자들은 온라인 볼록 최적화 문제를 고려하며, 각 시간 단계에서 결정 변수를 선택하고 볼록 손실 함수가 주어지는 상황을 다룹니다. 각 그래디언트는 유한한 경계를 가지고 있으나, 그래디언트 스파이크는 일반적인 그래디언트 노름보다 상당히 클 수 있습니다. 점진적으로 개선된 AMSGrad 변형을 통해 훈련 안정성과 수렴성을 보장하려고 합니다.

- **Performance Highlights**: 실험 결과, SPAM은 Adam 및 그 변종에 비해 다양한 작업에서 일관되게 우수한 성능을 보였습니다. 60M에서 1B까지의 LLM Pre-training, 강화 학습, 시계열 예측 등을 포함하며, 메모리 제약 환경에서도 상태-of-아트 메모리 효율 최적화기보다 성능이 뛰어남을 입증합니다. 특히 희소 모멘텀을 통해 메모리 효율적인 훈련을 가능하게 하여 리소스 효율성을 높이는데 기여합니다.



### Towards Counterfactual and Contrastive Explainability and Transparency of DCNN Image Classifiers (https://arxiv.org/abs/2501.06831)
- **What's New**: 이 논문에서는 딥 컨볼루션 신경망(DCNN)의 해석 가능성을 높이기 위한 새로운 방법을 제안합니다. 제안된 방법은 입력 이미지를 변경하지 않고 DCNN의 내부 작동을 탐색함으로써 해석 가능한 반사실적(counterfactual) 및 대조적(contrastive) 설명을 생성합니다. 이를 통해 모델의 결정이 특정 클래스 또는 다른 지정된 클래스에 대해 어떤 필터에 의해 이루어지는지를 명확히 할 수 있습니다.

- **Technical Details**: 제안된 방법은 주로 최상위 컨볼루션 레이어에서 필터를 식별하여 DCNN 모델에서 예측 결정의 근본 원인을 분석합니다. 이는 모델의 주요 개념과 특징을 나타내며, 이 필터들이 활성화되거나 비활성화되도록 조정될 때, DCNN이 입력 이미지를 지칭된 클래스로 예측하도록 도와줍니다. 이 방법은 DCNN의 모든 레이어에서 필터를 식별할 수 있으며, 다른 네트워크에서도 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: 이 연구의 결과는 Caltech-UCSD Birds (CUB) 2011 데이터셋에서의 성능을 보여주며, 제안된 설명 방법이 최신 기술과 비교했을 때 유용하다는 것을 입증합니다. 또한, 오분류 분석에서 이 방법의 적용 사례도 제시하여 모델의 결정 유효성을 검증하는 데 강력한 도구가 될 수 있음을 강조합니다.



### MEXA-CTP: Mode Experts Cross-Attention for Clinical Trial Outcome Prediction (https://arxiv.org/abs/2501.06823)
Comments:
          Accepted and to be published in SDM2025

- **What's New**: 이번 연구에서는 임상 시험 결과 예측을 위한 새로운 접근 방법인 MEXA-CTP를 소개합니다. 이 모델은 다양한 데이터 소스를 통합하여 임상 시험의 성공 가능성을 보다 정확하게 예측할 수 있도록 돕습니다. 특히, 인간의 편향을 최소화하고, 여러 모드 간의 상호작용을 효과적으로 캡처하기 위해 경량화된 주의(attention) 기반 모듈을 사용합니다. 또한, Cauchy 손실을 최적화 과정에 적용하여 모델의 성능을 높였습니다.

- **Technical Details**: MEXA-CTP는 'mode experts'라는 특화된 모듈을 활용하여 드러났습니다. 이 모델은 약물 분자, 질병 정보, 그리고 시험 프로토콜과 같은 여러 모드의 데이터를 통합하여 패턴을 인식합니다. 또한, NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실 함수를 사용하여 학습된 표현을 더욱 정제하여, 다체널 정보 간의 상호작용을 활용하는 강력한 모델을 구현합니다.

- **Performance Highlights**: 실험 결과 MEXA-CTP는 기존의 HINT 방법과 비교하여 F1 점수에서 최대 11.3%, PR-AUC에서 12.2%, ROC-AUC에서 2.5%의 성능 향상을 보였습니다. 또한, ablation 연구를 통해 각 구성 요소의 기여도를 평가하여 모델의 예측 능력을 뒷받침했습니다. 이러한 성과들은 비용이 많이 들어가는 데이터나 복잡한 설계에 의존하지 않고도 임상 시험 결과의 정확성을 높일 수 있음을 보여줍니다.



### Bridging the Fairness Gap: Enhancing Pre-trained Models with LLM-Generated Sentences (https://arxiv.org/abs/2501.06795)
- **What's New**: 이 논문에서는 Pre-trained Language Models(PLMs)가 내재하고 있는 성별 편향을 감소시키기 위해 새로운 접근 방식을 제안합니다. 기존의 디바이싱(debiasing) 방법들이 질, 다양성 또는 인구 통계적 균형이 결여된 외부 말뭉치에 의존하는 반면, 이를 해결하기 위해 Fair-Gender라는 방법을 사용합니다. 이는 의미적으로 풍부하고 속성 균형을 갖춘 문장을 활용하여 PLMs의 공정성을 향상시키는 방향으로 나아갑니다.

- **Technical Details**: 제안된 방법은 큰 언어 모델(LLM)로부터 생성되는 문장에서 Causal Analysis(인과 분석)를 통해 인과적 효과를 추정함으로써 불일치한 문장을 제외하고 정렬된 문장을 식별합니다. 이 과정에서 PLMs에 긍정적인 전이를 보장하여 언어 모델의 표현력을 유지하면서 성별 편향을 감소시킬 수 있습니다. 이 논문은 PLMs와 LLMs 간의 잠재 공간 차이를 극복하기 위한 알고리즘을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 Fair-Gender 방법이 PLMs의 성별 편향을 유의미하게 감소시키며, 언어 표현력을 보존하는 데 효과적임을 보여 줍니다. 이는 법률, 의료, 인사 등 다양한 분야에서 공정성을 확보하는 데 중요한 의미를 가집니다. PLMs가 가진 잠재적 편향을 자동으로 감소시키는 데 있어 새로운 가능성을 제시합니다.



### Improving Pain Classification using Spatio-Temporal Deep Learning Approaches with Facial Expressions (https://arxiv.org/abs/2501.06787)
Comments:
          8 pages, 3 figures, 3 tables. Accepted and presented at the 18th International Conference on Machine Vision (ICMV 2024), Edinburgh, UK

- **What's New**: 본 연구는 전통적인 자가 보고 방식이 주관적이며 비언어적인 개인에게 적합하지 않은 문제를 해결하기 위해 얼굴 표정을 활용한 자동 통증 감지를 탐구합니다. Pain Emotion Faces Database (PEMF)를 기반으로 깊이 있는 학습(Deep Learning) 기법을 통해 통증 평가를 개선하고자 하였습니다.

- **Technical Details**: 우리는 두 가지 새로운 접근 방식을 제안합니다: (1) 비디오 프레임을 분석하고 통증 존재 여부를 예측하기 위해 Long Short-Term Memory (LSTM) 블록과 결합된 하이브리드 ConvNeXt 모델을 사용합니다. (2) 얼굴 이미지의 랜드마크를 처리하여 통증을 감지하기 위한 Spatio-Temporal Graph Convolution Network (STGCN) 모델이 LSTM과 통합되었습니다.

- **Performance Highlights**: 우리는 PEMF 데이터셋을 사용하여 이진 통증 분류를 최초로 수행하였으며, 이러한 모델의 효과성을 광범위한 실험을 통해 입증하였습니다. 공간적 및 시간적 특성을 결합하여 통증 감지의 향상 가능성을 강조하며, 객관적인 통증 평가 방법론에 있어서 유망한 발전을 제시하고 있습니다.



### Cost-Effective Robotic Handwriting System with AI Integration (https://arxiv.org/abs/2501.06783)
Comments:
          This is an updated version of a paper originally presented at the 2024 IEEE Long Island Systems, Applications and Technology Conference (LISAT)

- **What's New**: 이번 연구는 Raspberry Pi Pico 마이크로컨트롤러와 3D 프린팅 부품을 활용하여 인간과 유사한 손글씨를 고정밀도로 복제할 수 있는 비용 효율적인 로봇 손글씨 시스템을 소개합니다. 시스템은 사용자 제공 텍스트를 사실적인 획 궤도로 변환하며, 하드웨어 비용을 약 56달러로 크게 줄였습니다. 이러한 혁신은 교육 및 연구, 보조 기술에서의 활용 가능성을 높이고 있습니다.

- **Technical Details**: 이 시스템의 개발은 크게 기계 설계, 조립, 하드웨어 제어 및 테스트의 네 가지 주요 단계로 나뉩니다. 3D CAD 디자인 소프트웨어인 Autodesk Fusion 360을 활용하여 기계 프레임을 정밀하게 설계하고, Creality Ender 3 3D 프린터로 부품을 출력하여 조립했습니다. Raspberry Pi Pico는 Arduino 스케치를 통해 프로그램되어 있으며, TensorFlow.js를 사용하여 사용자가 제공한 텍스트를 손글씨 궤도로 변환하는 기계 학습 모델을 구현하고 있습니다.

- **Performance Highlights**: 실험 결과 이 시스템은 ±0.3mm 이내의 손글씨 정밀도를 달성하며, 약 200 mm/min의 속도로 글씨를 쓸 수 있습니다. 저비용의 3D 프린팅 자재와 효율적인 기계 설계 덕분에 상업적인 대안들에 비해 상당한 비용 절감이 이루어졌습니다. 이런 성능을 바탕으로 교육, 연구 및 보조 기술 애플리케이션에 적합한 솔루션으로 자리잡을 수 있습니다.



### Static Segmentation by Tracking: A Frustratingly Label-Efficient Approach to Fine-Grained Segmentation (https://arxiv.org/abs/2501.06749)
- **What's New**: 이번 연구에서는 생물학적 이미지에서 특성 및 부위 세분화(image segmentation) 문제를 해결하기 위해 "Static Segmentation by Tracking (SST)"라는 새로운 방법론을 제안합니다. SST는 동일한 종의 표본 이미지에서 나타나는 일관된 특징을 활용하여, 몇 개의 라벨링된 이미지 만으로도 세분화를 수행할 수 있도록 설계되었습니다. 이는 전통적인 방법론인 수많은 이미지에 대한 수작업 라벨링을 하지 않고도 특성 측정을 가능하게 합니다.

- **Technical Details**: SST는 디지털 이미지 분석에서 종적인(temporal) 추적 문제로 세분화 문제를 재구성하는 기법입니다. 모델은 초기 이미지의 주석이 달린 마스크를 사용하여, 이후 이미지를 추적하고 마스크를 복제합니다. 이를 통해, SST는 Segment Anything Model 2 (SAM 2)을 기반으로 하여 하나의 라벨 이미지만으로도 품질 높은 세분화를 달성하는 것을 보여줍니다.

- **Performance Highlights**: 연구에서는 Cambridge Butterfly, NEON Beetle, Fish-Vista 등 세 가지 데이터셋에서 SST의 성능을 평가했습니다. SST는 SegGPT 등의 다른 원샷(one-shot) 세분화 기법보다 현저히 뛰어난 성능을 발휘했으며, 심지어 충분한 라벨이 있는 데이터로 학습된 모델들과 비교했을 때도 놀라운 결과를 보여주었습니다. 이러한 성과는 SST가 주석이 없는 이미지와 주석이 있는 이미지 간의 종속성을 명확히 활용했기 때문입니다.



### Multi-Label Scene Classification in Remote Sensing Benefits from Image Super-Resolution (https://arxiv.org/abs/2501.06720)
- **What's New**: 이번 연구는 위성 이미지의 품질을 향상시키기 위한 전처리 단계로 이미지 슈퍼 해상도(Super-Resolution, SR)를 사용하여 다중 라벨 장면 분류 성능을 개선하는 방법을 탐구합니다. 연구팀은 SRResNet, HAT, SeeSR, RealESRGAN 등 네 가지 SR 모델을 조사하고, 이를 ResNet-50, ResNet-101, ResNet-152, Inception-v4와 같은 다양한 CNN 아키텍처와 결합하여 분류 성능의 영향을 평가했습니다. 이 연구는 SR 모델들이 다중 라벨 예측에서 사용할 수 있는 중요한 통찰력을 제공함으로써 기존 원격 감지 시스템의 품질 향상에 기여할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 고해상도 이미지를 생성하는 비용 효율적인 방법으로 이미지 SR을 사용하여 구체적인 세부 정보를 포착하고 다중 라벨 장면 분류의 정확성을 높이기 위해 다양한 SR 모델을 적용했습니다. 각 SR 모델은 저해상도(Lower-Resolution, LR) 입력을 바탕으로 고해상도(High-Resolution, HR) 이미지를 생성하며, SR 모델은 회귀 기반 모델과 생성적 SR 모델(GAN 및 확산 모델)로 나눌 수 있습니다. SRResNet 및 HAT는 회귀 기반 모델로, SeeSR 및 RealESRGAN은 생성적 SR의 대표로 분석되었습니다.

- **Performance Highlights**: SR을 통해 향상된 이미지는 다양한 메트릭에서 다중 라벨 장면 분류 성능을 크게 개선함을 보여주었습니다. 연구팀은 적용된 SR 기술이 공간 세부 사항을 보존할 수 있음을 입증하였으며, 이는 다중 라벨 태스크에서 중요한 요소입니다. 이 연구는 다중 라벨 예측을 위한 SR 기술 선택의 유용성을 강조하고, 원격 감지 시스템 개선을 위한 통합할 수 있는 프레임워크를 제시합니다.



### ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian (https://arxiv.org/abs/2501.06715)
Comments:
          7 pages, 5 figures. X International conference "Informatics. Culture. Technology." (2024)

- **What's New**: 이번 연구는 대규모 언어 모델의 추론 능력 평가를 위한 포괄적인 벤치마크인 ZNO-Eval을 소개합니다. 이 벤치마크는 우크라이나의 표준 교육 테스트 시스템에서 실제 시험 과제를 기반으로 하고 있습니다. 그동안 대부분의 연구가 영어에 집중된 반면, 이번 연구는 우크라이나어에 대한 평가의 중요성을 강조합니다.

- **Technical Details**: ZNO-Eval 벤치마크는 단일 응답 옵션, 객관식, 매칭, 개방형 질문 등 다양한 과목에서의 문제 유형을 포함하고 있습니다. 문제는 우크라이나어, 수학, 역사 및 지리 등의 과목으로 구성되어 있으며, 다양한 도메인과 복잡성을 통해 모델의 추론 능력을 심층적으로 분석할 수 있는 기회를 제공합니다. 연구의 대상은 GPT-3.5-Turbo, GPT-4o, GPT-4-Turbo, Mistral Large, Claude 3 Opus 및 Gemini-1.5 Pro와 같은 잘 알려진 언어 모델들입니다.

- **Performance Highlights**: GPT-4o는 일반 지식 추론과 복잡한 언어 작업에서 우수한 성능을 보였습니다. 반면, Gemini Pro와 GPT-4 Turbo는 산수 문제에서 두각을 나타내며 단일 응답과 개방형 수학 문제에서 높은 성과를 기록했습니다. 그러나 역사와 지리와 같은 텍스트 전용 일반 지식 과제에서는 모든 모델이 최대 성능에 근접했지만, 우크라이나어와 수학에서는 여전히 격차가 존재해, 이러한 언어와 맥락에 대한 보다 정확한 모델 능력 및 한계 평가를 위한 전문화된 벤치마크 개발의 필요성을 강조합니다.



### Multi-task Visual Grounding with Coarse-to-Fine Consistency Constraints (https://arxiv.org/abs/2501.06710)
Comments:
          AAAI2025

- **What's New**: 본 연구에서는 Multi-task visual grounding에서의 불일치 문제를 해결하기 위해 Coarse-to-fine Consistency Constraints Visual Grounding 아키텍처($\text{C}^3\text{VG}$)를 제안합니다. 이 아키텍처는 두 단계로 구성되어 있으며, 초기 단계에서 쿼리와 픽셀 디코더를 통해 예비 탐지 및 분할 출력을 생성합니다. 이어지는 단계에서는 Mask-guided Interaction Module과 상호 일관성 제약 손실을 사용하여 다중 작업 간의 일관성을 확보합니다.

- **Technical Details**: C3VG는 Rough Semantic Perception(RSP) 단계에서 거친 예측을 수행한 후, Refined Consistency Interaction(RCI) 단계에서 이들 예측을 세밀하게 보정합니다. MIM은 다중 작업 결과의 내재적 상호작용을 통합하는 역할을 하며, 쌍방향 일관성 제약 손실을 통해 예측 결과의 일관성을 명시적으로 보장합니다. 이를 통해 REC 및 RIS 과제가 상호 보완적으로 작용할 수 있도록 개선합니다.

- **Performance Highlights**: C3VG는 RefCOCO, RefCOCO+, RefCOCOg 데이터세트에서 기존의 최첨단 REC 및 RIS 방법과 비교하여 현저한 성능 향상을 보여줍니다. 이 모델은 다중 작업 학습 프레임워크에서 사전 훈련된 멀티모달 인코더를 활용하여 정확도와 수렴 속도를 향상시킵니다. 또한, 훈련 에포크 수를 절반 이하로 줄이면서도 성능을 유지할 수 있는 효율성을 자랑합니다.



### Evaluating Sample Utility for Data Selection by Mimicking Model Weights (https://arxiv.org/abs/2501.06708)
- **What's New**: 최근 데이터 선택 기술들은 다양한 한계가 있었으나, 새로운 접근 방식인 Mimic Score는 사전 훈련된 모델을 사용하여 데이터 샘플의 유용성을 평가합니다. 이는 다른 데이터베이스에 접근할 필요 없이 특정 샘플이 훈련에 미치는 영향을 효율적으로 측정할 수 있게 합니다. Mimic Score를 통해 모델 훈련 중의 샘플 우선 순위를 자동화하여 모델 성능을 향상시키는 Grad-Mimic 프레임워크를 개발하였습니다.

- **Technical Details**: Mimic Score는 새로운 모델의 파라미터 기울기와 참조 모델 방향 간의 정렬을 기반으로 합니다. Grad-Mimic은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 훈련 중에 샘플의 우선 순위를 정하고, 두 번째 단계에서는 학습 단계 전반에 걸쳐 샘플 유용성을 평가합니다. 이 과정은 데이터 선택을 자동화하기 위한 필터 단계를 통해 진행됩니다.

- **Performance Highlights**: Empirical results에 따르면 Grad-Mimic은 6개의 이미지 데이터셋에서 일관된 성능 향상을 보여 주었으며, CLIP 모델의 성능을 더욱 향상시키는 데 기여했습니다. 또한, 기존 필터링 방식보다 더 정교하게 낮은 가치의 샘플을 제거하여 모델 성능을 높이는 데 효과적입니다. Mimic Score와 그 필터는 데이터셋 품질을 정확하게 평가하여 기존 필터링 전략을 개선할 수 있음을 보여줍니다.



### Mamba-MOC: A Multicategory Remote Object Counting via State Space Mod (https://arxiv.org/abs/2501.06697)
- **What's New**: 이번 연구에서는 다양한 카테고리의 원거리 객체 개수를 정확하게 추정하는 다중 카테고리 원거리 객체 개수 세기(Multicategory Remote Object Counting, MOC) 문제를 다루고 있습니다. 특히, Mamba라는 새로운 모델을 기반으로 한 Mamba-MOC 프레임워크를 제안하였으며, 이는 원거리 객체 세기 분야에 Mamba를 최초로 적용한 사례입니다. 또한, 계층적 특성의 깊은 통합을 촉진하기 위한 교차 스케일 상호작용 모듈을 설계하였습니다.

- **Technical Details**: 제안된 방법론은 vmamba 백본(백본 네트워크)을 사용하여 다중 수준의 특성 표현을 추출하고, 교차 스케일 상호작용 모듈을 통해 coarse-grained와 fine-grained 특성의 효과적인 통합을 수행합니다. 또한, 컨텍스트 상태 공간 모델(Contextual State Space Model, CSSM)을 활용하여 스캔 과정 중 지역 이웃 정보를 강조하고, 이를 통해 더욱 세밀한 정보 추출을 가능하게 합니다.

- **Performance Highlights**: 대규모 실제 시나리오에서의 실험 결과는 제안된 Mamba-MOC 방법이 기존의 주요 객체 수 계산 알고리즘들과 비교할 때 최첨단 성능을 달성했음을 보여줍니다. 이는 Mamba의 전방위적인 장점을 활용함으로써 원거리 이미지 내에서 효과적인 글로벌 및 로컬 컨텍스트 정보를 모델링할 수 있음을 통한 결과입니다.



### PGP-SAM: Prototype-Guided Prompt Learning for Efficient Few-Shot Medical Image Segmentation (https://arxiv.org/abs/2501.06692)
Comments:
          5 pages, 2 figures, Accepted at ISBI 2025

- **What's New**: 이번 논문에서는 SAM(Segment Anything Model)의 의료 이미지 세분화(Customizing SAM for medical image segmentation)를 위한 새로운 방법인 PGP-SAM을 제안합니다. PGP-SAM은 제한된 샘플을 사용하여 수동 프롬프트를 대체할 수 있는 새로운 프로토타입 기반의 few-shot tuning 접근법입니다. 이 접근법은 대량의 픽셀 수준 주석 없이도 효과적으로 세분화를 수행할 수 있게 해줍니다.

- **Technical Details**: PGP-SAM의 주요 아이디어는 클래스 특정 지식(class-specific knowledge)과 관계를 포착하기 위해 클래스 내 및 외 프로토타입(intra- and inter-class prototypes)을 활용하는 것입니다. 여기에는 (1) 다중 스케일 정보를 통합하는 플러그 앤 플레이 모듈(plug-and-play contextual modulation module)과 (2) 프로토타입과 특성을 융합하여 자동 프롬프트 생성을 위한 클래스 지향 크로스 어텐션 메커니즘(class-guided cross-attention mechanism)이라는 두 가지 주요 구성 요소가 포함됩니다.

- **Performance Highlights**: PGP-SAM은 공개 멀티 오르간 데이터셋과 사설 심실 데이터셋을 통해 기존의 프롬프트 없는 SAM 변형들과 비교할 때 우수한 평균 Dice 점수를 달성합니다. 특히, 2D 슬라이스의 10%만을 사용하여도 높은 성능을 발휘하는 점이 주목할 만합니다.



### Application of Vision-Language Model to Pedestrians Behavior and Scene Understanding in Autonomous Driving (https://arxiv.org/abs/2501.06680)
- **What's New**: 최근 자율주행(Autonomous Driving) 기술이 크게 발전하여 3D 탐지, 분류, 로컬라이제이션 결과를 보여주고 있습니다. 그러나 보행자 행동의 의미적 이해 및 보행자와의 인터랙션 처리와 같은 많은 도전 과제가 여전히 남아 있습니다. 이 연구에서는 대규모 언어 모델(LLM) 및 비전-언어 모델(VLM)의 지식을 소형 비전 네트워크로 효과적으로 증류하는 방법을 분석하여, 복잡한 장면의 의미적 표현으로 의사결정 및 제어에 활용할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 비전-언어 기초 모델은 시각 데이터와 텍스트 데이터를 통합하여 다중 모달 AI의 최전선에 있는 기술입니다. 이 논문에서는 GPT4-V와 같은 사전 훈련된 비전-언어 모델로부터 일반 지식을 전이하는 지식 증류(Knowledge Distillation)를 수행하였으며, 이를 통해 보행자의 행동과 의미적 속성을 이해하는 효율적인 비전 모델을 개발하였습니다. 이 과정에서 다중 라벨 분류 문제로 설정하고, GPT의 주석을 통해 생성된 의미적 텍스트 라벨로 비전 네트워크를 감독하여 정확한 예측을 목표로 하였습니다.

- **Performance Highlights**: 우리는 이 연구를 통해 열거된 과제들에 대해 의미적으로 풍부한 보행자 속성과 분류 체계를 통해 개선된 성과를 달성하였습니다. 전통적인 분류법보다 더 많은 보행자 속성과 의미적 카테고리를 수집하여, 보다 정교한 자율주행 시스템을 위한 기반을 마련하였습니다. 이를 통해 보행자의 행동을 보다 정확하게 예측하고, 안전하고 신뢰할 수 있는 자율주행 내비게이션을 위한 기반을 강화하였습니다.



### Imbalanced Medical Image Segmentation with Pixel-dependent Noisy Labels (https://arxiv.org/abs/2501.06678)
- **What's New**: 이번 연구에서는 의료 이미지 분할의 성능을 저하시키는 픽셀 의존성 노이즈(label noise) 문제를 해결하기 위해 Collaborative Learning with Curriculum Selection (CLCS) 프레임워크를 제안합니다. CLCS는 노이즈 레이블을 픽셀 단위로 처리하고 다양한 클래스 불균형(class imbalance) 문제를 해결하기 위해 협력적 학습(collaborative learning) 프레임워크를 활용합니다. 또한, 새로운 curriculum dynamic thresholding 접근법을 통해 모델의 학습 진행에 적응하는 동적 임계값을 사용하여 깨끗한 데이터를 선택하는 방법이 특징입니다.

- **Technical Details**: CLCS 프레임워크는 Curriculum Noisy Label Sample Selection (CNS)와 Noise Balance Loss (NBL)의 두 모듈로 구성되어 있습니다. CNS 모듈은 협력적 학습을 위해 두 개의 브랜치 네트워크를 설계하여 서로 다른 뷰에서 동일한 인스턴스의 다양한 특징 표현을 추출하고 픽셀의 클래스 확률을 투표하여 결정합니다. NBL 모듈에서는 의심스러운 노이즈 레이블을 단순히 삭제하는 대신 강력한 손실 함수를 사용하여 이러한 인스턴스를 활용하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 CLCS 방법은 다양한 노이즈 유형을 가진 두 개의 실제 의료 이미지 데이터 세트에서 유의미한 성능 향상을 보여줍니다. 연구 결과에 따르면, 기존 방법들에 비해 CLCS가 픽셀 단위의 노이즈가 존재하는 상황에서도 정확한 분할 성능을 유지하며, 소수 클래스의 데이터 활용도를 높여 클래스 불균형 문제를 효과적으로 완화하는 것으로 나타났습니다. 이를 통해 의료 이미지 분할 분야에서 CLCS의 적용 가능성을 강조합니다.



### FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings (https://arxiv.org/abs/2501.06645)
- **What's New**: 이 논문은 FocalPO라는 새로운 손실 함수(loss function)를 소개합니다. 기존의 Direct Preference Optimization (DPO) 방법이 잘못된 선호 순위를 수정하는 데 효과적이지 않다는 최근 연구를 기반으로, FocalPO는 오히려 잘못된 순위 쌍의 중요성을 낮추고 정확히 순위를 매길 수 있는 쌍에 대한 모델의 이해를 향상시키는 데 중점을 둡니다. 이 접근법은 시각적 작업에서 사용되는 Focal Loss에서 영감을 받았습니다.

- **Technical Details**: FocalPO는 DPO 손실 함수에 동적으로 조절되는 계수를 추가하여, 모델이 올바로 순위를 매길 수 있는 쌍에 대해 더 높은 가중치를 부여합니다. 이 방식은 잘못된 페어(incorrect pairs)의 영향력을 줄이며, 모델이 올바른 보상 추정치에 따라 선호 쌍을 배워나가도록 촉진합니다. gradient 분석을 통해 FocalPO가 올바른 순서의 쌍에 더 높은 가중치를 할당함을 확인했습니다.

- **Performance Highlights**: 실험 결과, FocalPO는 Alpaca Eval 2.0과 같은 인기 있는 벤치마크에서 DPO 및 그 변형들보다 우수한 성능을 보였습니다. Mistral-Base-7B와 Llama-3-Instruct-8B 모델을 사용하여 진행한 평가에서도 FocalPO의 효과를 분명히 나타냈습니다. 또, FocalPO가 올바른 샘플 그룹과 잘못된 샘플 그룹에서 훈련에 미치는 영향을 실증적으로 보여주었습니다.



### Enhancing Path Planning Performance through Image Representation Learning of High-Dimensional Configuration Spaces (https://arxiv.org/abs/2501.06639)
- **What's New**: 이 논문은 Wasserstein Generative Adversarial Networks (WGANs)와 Gradient Penalty (GP)를 활용하여 장애물이 있는 미지의 장면에서 경로 계획(Path Planning) 작업을 가속화하는 새로운 방법을 제안합니다. Rapidly-exploring Random Tree 알고리즘을 사용하여 충돌이 없는 경로의 waypoint 분포를 근사합니다. 또한, 다중 양식(Multimodal) 데이터 세트를 효과적으로 처리하기 위해 WGAN-GP의 조건으로 연속 잠재 공간(Continuous Latent Space)의 전방 확산 프로세스를 도입했습니다.

- **Technical Details**: 제안된 방법은 충돌이 없는 경로의 waypoints를 행렬(Matrix)로 인코딩하여 multidimensional ordering을 자연스럽게 유지합니다. 이로 인해 모델 학습(Model Learning) 개선과 훈련 수렴(Training Convergence) 속도가 향상됩니다. 훈련된 모델이 실제 waypoint를 정확히 포착하지 못하는 경우 평가하는 방식을 제안하며, 이러한 경우에는 균일 샘플링(Uniform Sampling)으로 알고리즘의 확률적 완전성을 보장합니다.

- **Performance Highlights**: 실험 결과는 중요한 시간 제약 조건 하에서 경로 계획 작업을 가속화하는 데 유망한 결과를 보여줍니다. 이 연구는 기존 기계 학습 기반 방법들과 달리 각 시나리오에 대해 최적 비율을 수동으로 결정해야 하는 전통적인 과정에서 벗어나, WGAN-GP의 장점을 극대화하는 혁신적인 방안을 제시합니다. 해당 연구의 소스 코드는 공개적으로 이용 가능합니다.



### Exploring Pose-Based Anomaly Detection for Retail Security: A Real-World Shoplifting Dataset and Benchmark (https://arxiv.org/abs/2501.06591)
- **What's New**: 이번 연구는 소매업체의 심각한 문제인 도둑질(shoplifting) 탐지를 위한 새로운 접근 방식을 제시합니다. PoseLift라는 데이터셋은 실제 환경에서 수집된 인체 포즈 데이터로, 도둑질 행위를 탐지하기 위한 프라이버시 보호 기능을 갖추고 있습니다. 연구자들은 이 데이터를 활용해 도둑질 탐지를 이상 탐지(anomaly detection) 문제로 재구성하였습니다.

- **Technical Details**: PoseLift 데이터셋은 실제 소매 환경에서 수집된 CCTV 영상을 사용하여 구축되었습니다. 이 데이터셋은 1920x1080 해상도와 초당 15프레임으로 촬영된 다양한 각도의 비디오를 포함하고 있으며, 정상 쇼핑 행동과 도둑질 행동을 동시 수집하여 프라이버시를 보호합니다. 또한, 연구에서는 최신의 이상 탐지 모델을 벤치마킹하여 다양한 성능 지표를 통해 효과성을 검증하였습니다.

- **Performance Highlights**: PoseLift 데이터셋을 활용한 연구 결과는 도둑질 탐지 정확도가 매우 높음을 보여주며, 전통적인 방법의 프라이버시 및 편향 문제를 효과적으로 해결할 수 있는 가능성을 제시합니다. 연구에서 평가된 포즈 기반 이상 탐지 모델은 AUC-ROC, AUC-PR 및 EER 지표로 성능을 측정하여 도둑질 탐지 시스템의 발전에 기여할 예정입니다. 이 데이터셋은 공공에 제공되어 연구자들이 윤리적으로 컴퓨터 비전을 발전시킬 수 있는 귀중한 도구가 될 것입니다.



### ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning (https://arxiv.org/abs/2501.06590)
- **What's New**: 이번 논문에서는 ChemAgent라는 새로운 프레임워크를 소개하고 있습니다. 이 프레임워크는 대형 언어 모델(LLMs)이 화학 문제를 해결하는 데 필요한 성능을 향상시키기 위해 설계되었습니다. 특히 화학 작업을 하위 작업으로 분해하고 이들을 구조화된 컬렉션으로 컴파일하여 동적으로 업데이트되는 라이브러리를 통해 성능을 개선합니다.

- **Technical Details**: ChemAgent는 효과적인 작업 분해(Decomposition)와 솔루션 생성을 지원하기 위해 세 가지 유형의 메모리(memory)와 라이브러리 강화(reasoning component)를 설계했습니다. 이 방식을 통해 LLM은 경험에 따라 시간이 지나면서 개선될 수 있습니다. ChemAgent는 또한 문제 발생 시 메모리에서 관련 정보를 검색하고 정제하여 새로운 문제를 해결하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, SciBench의 네 가지 화학 추론 데이터셋에서 ChemAgent는 최대 46%의 성능 향상을 보여주었으며, 기존 방법들을 크게 초월했습니다. 이러한 결과는 약물 발견(drug discovery) 및 소재 과학(materials science)과 같은 미래의 다양한 응용 가능성을 시사합니다.



### Active Rule Mining for Multivariate Anomaly Detection in Radio Access Networks (https://arxiv.org/abs/2501.06571)
- **What's New**: 이 논문에서는 다변량(anomaly detection) 이상 탐지 시스템의 필요성을 해결하기 위해 반자율(anomaly rule miner) 이상 규칙 발굴기를 제안한다. 이번 연구는 어떤 이상(anomaly)이 특이한지에 대한 설명이 필요하다는 점을 강조하며, 이를 통해 네트워크 운영자들이 실제 상황에서 발생하는 문제를 이해하고 대응할 수 있도록 돕고자 한다. 실험은 시간 시계열(time series) RAN 데이터를 통해 진행되었으며, 주요 특징은 규칙 매핑을 통해 이상 탐지 결과에 대한 행동 가능성을 제시한다.

- **Technical Details**: 제안한 방법론은 이산치(discrete) 및 시계열 데이터(time series data) 모두에 적용 가능하며, 기존의 이상 탐지 방법의 제약 없이 사용할 수 있다. 이 시스템은 이상을 설명할 뿐만 아니라 이를 바탕으로 actionable rules(행동 가능한 규칙)을 수립할 수 있는 방법도 포함되어 있다. 또한, 제안 시스템은 정적(threshold) 기준이 아닌, 도메인 요구에 따라 발전할 수 있는 규칙 관리 방법을 포함하고 있어 개념 변화(concept drift)에 능동적으로 대응할 수 있다.

- **Performance Highlights**: 이 시스템을 사용한 결과, 2400개의 이상을 126개의 조건으로 그룹화하여 관리를 용이하게 만들었으며, 운영자들은 각 조건의 샘플 이상을 평가하고 검증하기가 훨씬 쉬워졌다. 이를 통해 운영 효율성이 크게 개선되었고, 사람이 이상을 검증하는 데 소요되는 시간이 절약되어 인적 오류를 줄였다. 제안된 방법은 비즈니스에 적합한 규칙을 미리 마련할 수 있는 기반을 마련하였다.



### Discrete Speech Unit Extraction via Independent Component Analysis (https://arxiv.org/abs/2501.06562)
Comments:
          Accepted to ICASSP 2025 SALMA Workshop. Code available at this https URL

- **What's New**: 이 연구에서는 Self-supervised speech models (S3Ms)을 활용하여 speech processing 분야에서 분리된 음성 단위인 discrete speech units (DSUs)를 추출하기 위한 선형 전처리 방법(linear preprocessing methods)의 가능성을 조사합니다. 현재까지 S3M의 표현(representation)을 더욱 효과적으로 클러스터링하기 위한 전처리 연구는 부족했던 점을 강조합니다.

- **Technical Details**: 본 논문에서는 표준화(standardization), 주성분 분석(principal component analysis), 화이트닝(whitening), 독립 성분 분석(independent component analysis, ICA) 방법을 DSU 기반 자동 음성 인식(ASR) 벤치마크에서 평가합니다. 이러한 전처리 방법들이 k-means 클러스터링을 위한 전처리로서 효과적임을 입증하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: DSUs를 사용하는 경우 자동 음성 인식에서 강력한 성능 향상을 보여주며, ICA의 개별 성분의 직교성(orthogonality) 및 해석 가능성(interpretability)에 대한 방대한 분석을 수행합니다. 이러한 발견은 DSUs의 품질을 향상시키는 새로운 접근 방식을 제시합니다.



### A Survey on Spoken Italian Datasets and Corpora (https://arxiv.org/abs/2501.06557)
Comments:
          submitted to IEEE Access Journal in Dec 2024

- **What's New**: 이 논문은 66개의 구술 이탈리아어 데이터셋을 포괄적으로 분석하여 이탈리아어 자원의 부족을 해소하려는 노력을 보여 줍니다. 기존 데이터셋의 특징, 방법론 및 응용 분야를 강조하며, 자동 음성 인식(ASR), 감정 탐지(emotion detection), 교육 등 다양한 분야의 응용 가능성을 제시합니다. GitHub와 Zenodo를 통해 공개된 데이터셋 목록은 연구자와 개발자에게 소중한 자원으로 작용할 것입니다.

- **Technical Details**: 본 논문에서는 구술 이탈리아어 데이터셋을 음성 유형, 출처 및 맥락, 인구 통계 및 언어적 특징에 따라 분류하여 데이터 수집, 주석(annotation), 검증 방법론을 심층적으로 탐구합니다. 데이터셋 품질과 가용성은 독립적으로 확인되지 않았지만, 제공된 정보를 종합하여 최선의 관행과 일반적인 문제점을 강조합니다. 아울러 데이터셋의 다양성과 적용성을 높이기 위해 협력 및 표준화를 통한 활용 방안을 추천합니다.

- **Performance Highlights**: 이탈리아어 구술 데이터셋의 확보는 고품질 ASR 시스템 및 언어 분석을 위한 필수 요소로 강조됩니다. 논문에서는 대화, 독백, 자발적 언어 표현 등 다양한 음성 유형을 포괄하는 66개의 데이터셋을 제시하고, 각 데이터셋이 제공하는 실질적인 응용을 설명합니다. 특히, 감정 분석 및 대화형 NLP 모델 개발에 유용한 데이터셋의 중요성이 강조되며, 미래 연구 방향을 제시합니다.



### Hierarchical Reinforcement Learning for Optimal Agent Grouping in Cooperative Systems (https://arxiv.org/abs/2501.06554)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문은 다중 에이전트 시스템에서 에이전트 그룹화 및 페어링 문제를 해결하기 위한 계층적 강화 학습(Hierarchical Reinforcement Learning) 접근 방식을 제시합니다. 본 연구의 목표는 최적의 그룹화 및 에이전트 정책을 동시에 학습하는 것입니다. 계층적 RL 프레임워크를 통해 그룹화의 고차원 결정과 저차원 에이전트 행동을 구분하여 효율적인 학습과 확장성을 보장합니다.

- **Technical Details**: 이 방법은 CTDE(Centralized Training with Decentralized Execution) 패러다임을 활용하여 에이전트 간의 동질성과 협력을 처리할 수 있도록 순열 불변(neural networks) 신경망을 사용하는 것을 포함합니다. 옵션-비평가(option-critic) 알고리즘이 계층적 의사결정 과정을 관리하기 위해 조정되어, 동적이고 최적의 정책 조정을 가능하게 합니다. 또한 에이전트 조합 정보를 네트워크 아키텍처에 통합하여 팀 조합의 순열 불변성을 보장하고 모델링 효율성을 높입니다.

- **Performance Highlights**: 이 접근 방식은 기존의 전통적인 Q-learning 또는 옵션-비평가 접근법보다 더 적은 계산 복잡도로 최적 팀 페어링 문제를 재구성합니다. 기본적으로, 제안된 아키텍처는 차원 축소 문제를 피하고 팀 수가 증가함에 따라 정책 매개변수 공간의 기하급수적 성장을 방지합니다. 이를 통해 대규모 문제에 대한 확장성을 확보하며, 복잡한 페어링 및 그룹화 문제에 대한 해결책을 제공합니다.



### Natural Language Supervision for Low-light Image Enhancemen (https://arxiv.org/abs/2501.06546)
Comments:
          12 pages, 10 figures

- **What's New**: 이번 논문에서는 자연어 감독(Natural Language Supervision, NLS) 전략을 도입하여 저조도 이미지 향상(Low-Light Image Enhancement, LLIE)을 위한 새로운 접근 방식을 제안합니다. 이 방법은 텍스트와 대응되는 이미지의 특징 맵(feature maps)을 공동 학습하여 다양한 조명 상태에서 이미지를 설명하는 유연한 인터페이스를 제공하는 데 중점을 두고 있습니다. 또한, 텍스트의 지역(word)와 이미지의 해당 영역(image regions) 간의 연결을 이해하기 위해 텍스트 지침 조정 메커니즘(Textual Guidance Conditioning Mechanism, TCM)을 설계하여 방향성을 부여합니다.

- **Technical Details**: 저자들은 TCM을 통해 이미지 지역(image regions)과 문장 단어(sentence words) 간의 관계를 포괄적으로 캡처합니다. 이는 멀티 모달(multi-modal) 데이터 분포에서 발생하는 교육 문제를 해결하는 데 기여합니다. 또한, 정보 융합 주의 모듈(Information Fusion Attention, IFA)을 설계하여 다양한 수준의 이미지와 텍스트 정보에 대한 특징 식별 및 병합을 효과적으로 수행할 수 있도록 합니다. 이 두 가지 모듈을 결합하여 저조도 이미지 향상의 성능을 높이는 NaLSuper 네트워크를 구현하였습니다.

- **Performance Highlights**: NaLSuper는 4개의 벤치마크 데이터셋에서 폭넓은 실험을 통해 최근의 최첨단 방법들보다 더 우수한 성능을 보여주었습니다. 정량적 및 정성적 평가 모두에서, 제안된 접근 두 가지 방식(TCM 및 IFA)이 저조도 이미지 향상에 효과적임을 입증하며, 기존의 메트릭 지향 접근 방식이 비주얼 품질과 함께 만족스러운 성과를 달성할 수 있도록 하고 있습니다.



### Determination of galaxy photometric redshifts using Conditional Generative Adversarial Networks (CGANs) (https://arxiv.org/abs/2501.06532)
- **What's New**: 이 논문에서는 Conditional Generative Adversarial Networks (CGANs)를 활용하여 은하의 photometric redshift를 결정하는 새로운 알고리즘적 접근 방식을 제안합니다. 기존의 기계 학습 기술들은 주어진 은하의 photometry와 spectrometry로부터 단일 값을 추정해왔지만, 제안된 CGAN 접근법은 확률 회귀(probabilistic regression) 문제로 접근하여 은하의 redshift에 대한 전체 확률 밀도를 계산합니다. 이는 Dark Energy Survey (DES) Y1 데이터와 기존의 Random Forest 회귀기법과 비교하여 평가됩니다.

- **Technical Details**: 연구에서 제안된 CGAN 모델은 은하의 broad-band 필터에서 측정한 magnitude에 기반하여 photometric redshift를 추정합니다. 훈련 데이터는 이미 알려진 spectroscopic redshift가 있는 photometric 데이터를 포함하며, 이 데이터를 사용하여 Generator와 Discriminator 네트워크를 훈련합니다. 이 과정에서 Generator는 실제 데이터를 모방하는 합성 데이터를 생성하고, Discriminator는 이러한 데이터가 합성인지 실제인지 구별합니다.

- **Performance Highlights**: 제안된 CGAN 알고리즘은 DES Y1 데이터와 비교하여 기존의 기계 학습 기법인 Random Forest 회귀기법보다 우수한 성능을 보여주는 것을 목표로 합니다. 특히, CGAN 접근법이 은하의 redshift에 대한 Probability Density Function의 전체를 정확히 매핑할 수 있는 가능성을 보여줍니다. 이는 기존 기법이 포착하지 못했던 확률 분포의 모든 정보까지 포함할 수 있도록 하여, 기존의 포인트 추정(point estimate) 접근법의 한계를 보완할 수 있습니다.



### Neural Codec Source Tracing: Toward Comprehensive Attribution in Open-Set Condition (https://arxiv.org/abs/2501.06514)
- **What's New**: 이 논문에서는 Neural Codec Source Tracing (NCST) 작업을 정의하며,开放集合 (open-set) 신경 코덱 분류 및 해석 가능한 ALM 감지를 수행할 수 있는 새로운 접근 방식을 소개합니다. 기존의 연구에서는 닫힌 집합 (closed-set) 조건만 고려했으나, 본 연구는 외부 분포 (out-of-distribution, OOD) 조건의 도전을 포함합니다. 이를 위해 ST-Codecfake 데이터셋을 구축하고, NCST 모델의 성능을 평가할 종합적인 벤치마크를 설정했습니다.

- **Technical Details**: ST-Codecfake 데이터셋은 11개의 최첨단 신경 코덱 방식으로 생성된 이중언어 오디오 샘플을 포함하며, ALM 기반 OOD 테스트 샘플도 포함되어 있습니다. 이는 VCTK와 AISHELL3의 서브셋으로 구성된 현실 도메인과 11개의 서로 다른 신경 코덱 방법으로 생성된 가짜 음성을 포함합니다. 본 연구는 NCST 모델이 ID(정상) 분류 및 OOD 감지에서 양호한 성능을 보였으나, 보지 못한 실제 음성을 분류하는 데 있어 강인함이 부족하다는 점을 발견했습니다.

- **Performance Highlights**: NCST 모델은 ID 분류에서 99.99%의 F1 점수, 열린 집합 OOD 탐지에서 97.54%의 AUC, ALM 기반 오디오의 백엔드 검출에서 92.36%의 F1 점수를 기록하며 뛰어난 성능을 보여주었습니다. 이를 통해 코덱 기반 음성을 위한 소스 추적의 가능성을 보여주며, 신경 코덱 방법에 대한 지적 재산 보호를 향상시킬 수 있는 장점을 제공합니다.



### Resource Allocation under the Latin Square Constrain (https://arxiv.org/abs/2501.06506)
Comments:
          This paper has been accepted in AAMAS 2025 as an extended abstract

- **What's New**: 이 논문에서는 $n 	imes n$의 격자 형식인 라틴 정사각형(Latin square) 제약 조건 하에서 $n$개의 비분할(indivisible) 아이템을 $n$명의 에이전트(emphasis agents)에게 $n$번의 라운드(round) 동안 할당하는 문제에 대해 다루고 있습니다. 이 연구는 자원 관리와 공정한 배분 문제가 실제적으로 중요하다는 점을 강조합니다. 주어진 조건 하에서 사회적 복지를 극대화하기 위한 할당 구조를 찾는 것이 주요 목표입니다.

- **Technical Details**: 라틴 정사각형 할당 문제(LSA)는 두 가지 사회적 복지 설정, 즉 효용의 합을 극대화하는 공리주의적 사회적 복지(utilitarian social welfare)와 최소 효용을 극대화하는 평등주의적 사회적 복지(egalitarian social welfare)로 나누어 연구됩니다. 논문은 이 문제의 NP-hard성을 증명하고, 부분 및 전체 설정에서 각각 (1-1/e) 및 (1-1/e)/4 근사 알고리즘을 제공합니다. 또한 고정 파라미터 가변성(Fixed-Parameter Tractability) 알고리즘을 통해 라틴 정사각형의 차수(order)와 최적 값을 기준으로 하는 해결 방법도 제시합니다.

- **Performance Highlights**: 이 연구는 공리주의적 사회적 복지를 극대화하는 LSA 문제가 NP-hard하다는 것을 보였습니다. 또한, 판단 기준의 최대 평등주의적 사회적 복지가 1 이하인지 아니면 2 이상인지 결정하는 것도 NP-hard임을 입증했습니다. 전체 및 부분 설정에서 공평한 할당(existence of equitable allocation)과 관련한 여러 조건들을 만족하는 완전 할당이 존재하는지를 확인하는 것이 NP-hard하다는 점 역시 강조되어 있습니다.



### PASS: Presentation Automation for Slide Generation and Speech (https://arxiv.org/abs/2501.06497)
- **What's New**: 이 논문은 PASS라는 새로운 파이프라인을 소개합니다. 이 시스템은 일반적인 Word 문서에서 슬라이드를 생성하고 생성된 슬라이드의 음성 전달을 자동화합니다. 이전 연구들이 주로 연구 논문을 변환하는 데 초점을 맞춘 반면, PASS는 보다 일반적인 문서에서 슬라이드를 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: PASS는 슬라이드 생성과 슬라이드 발표라는 두 가지 핵심 모듈로 구성됩니다. 슬라이드 생성 모듈은 문서의 내용을 바탕으로 구조화된 슬라이드를 자동으로 생성하며, LLM 및 멀티모달 (multimodal) 방식을 사용하여 텍스트와 이미지를 처리합니다. 슬라이드 발표 모듈은 각 슬라이드에 대한 스크립트를 생성하고 AI 음성 합성을 통해 이를 음성으로 변환합니다.

- **Performance Highlights**: 이 연구에서는 슬라이드의 관련성, 일관성 및 중복성을 평가하기 위한 LLM 기반 평가 메트릭을 개발했습니다. PASS는 이미지 매핑 모듈을 통해 적절한 이미지를 슬라이드에 자동으로 매핑하고, 발표 스크립트 생성 모듈을 통해 각 슬라이드에 대한 고품질 오디오를 생성합니다. 이 접근법은 발표의 콘텐츠 생성과 전달을 완전히 자동화하는 데 중점을 두고 있으며, 이는 이전의 연구들과는 차별화된 점입니다.



### TopoFormer: Integrating Transformers and ConvLSTMs for Coastal Topography Prediction (https://arxiv.org/abs/2501.06494)
Comments:
          11 pages, 5 figures, 1 table

- **What's New**: TopoFormer는 해안 프로필 예측을 위해 transformer 기반의 인코더와 ConvLSTM 레이어를 통합한 새로운 하이브리드 딥러닝 아키텍처입니다. 이는 Mean Low Water Springs (MLWS)와 Mean Low Water Neaps (MLWN) 데이터에 기준을 두고, 2000개 이상의 데이터를 활용하여 정확한 지형 정보를 예측합니다. 데이터 수집의 한계를 극복하기 위해, 기존의 기계학습 방법을 바탕으로 향상된 성능을 보여주고 있습니다.

- **Technical Details**: TopoFormer는 긴 범위 의존성과 지역적 시간 패턴을 동시에 캡처하기 위해 Multi-Head Attention 메커니즘과 ConvLSTM 레이어를 결합합니다. 아키텍처는 해안 프로필 데이터의 연속적인 고도-체인 지 쌍으로 이루어지며, 이를 정규화하여 다양한 데이터셋에서 일반화할 수 있도록 합니다. 또한 MAE 손실 함수를 사용하여 훈련하며, 761K의 훈련 가능한 파라미터를 통해 계산 효율성을 자랑합니다.

- **Performance Highlights**: TopoFormer의 성능은 DenseNet, CNN 및 LSTM과 같은 최신 모델들과 비교하여 평가되었고, 평균 절대 오차(MAE)는 2cm로 가장 낮은 값을 기록했습니다. TopoFormer는 In-Distribution (ID) 및 Out-of-Distribution (OOD) 평가에서 모두 우수한 정확성을 제공하여, 해안 관리 및 모니터링에 혁신적인 기여를 하고 있습니다.



### Improving Requirements Classification with SMOTE-Tomek Preprocessing (https://arxiv.org/abs/2501.06491)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 연구에서는 PROMISE 데이터셋의 클래스 불균형 문제를 해결하기 위해 SMOTE-Tomek 전처리 기법을 적용했습니다. 이를 통해 소수 클래스의 표현을 향상시키며, 검증 폴드의 무결성을 유지하면서 분류 정확도를 크게 향상시킬 수 있었습니다.

- **Technical Details**: PROMISE 데이터셋은 기능적(requirements) 및 비기능적(non-functional) 요구사항이 포함된 969개의 분류된 요구사항으로 구성되어 있습니다. 연구에서는 SMOTE(Synthetic Minority Oversampling Technique)와 Tomek Links를 통합하여 불균형 문제가 해결된 클린하고 균형 잡힌 학습 데이터를 생성하고, K-fold cross-validation을 통해 모델 성능을 평가했습니다.

- **Performance Highlights**: 로지스틱 회귀(Logistic Regression) 모델은 76.16%의 정확도를 기록하며, 58.31%의 기초선(Baseline)을 크게 초과했습니다. 이러한 결과는 머신러닝 모델이 효율적으로 작동할 수 있는 강력한 솔루션으로서의 가능성을 보여줍니다.



### NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References (https://arxiv.org/abs/2501.06488)
- **What's New**: 이번 연구에서는 NVS-SQA라는 새로운 NSS(신경 합성 장면) 품질 평가 방법을 제안합니다. 이 방법은 인공 주석(human annotation)에 의존하지 않고, 자가 감독(self-supervision) 방식으로 품질 표현을 학습합니다. NVS-SQA는 히uristic(휴리스틱) 신호와 품질 점수를 학습 목표로 활용하여 비지도학습 상황에서도 유효한 품질 평가를 가능하게 합니다.

- **Technical Details**: NVS-SQA는 NSS 품질 평가를 위한 첫 번째 자가 감독 학습 프레임워크로, 비지도 데이터셋에서 품질 표현을 효과적으로 추출할 수 있도록 설계되었습니다. 이 프레임워크는 대조(pair) 준비 과정에서 NSS에 특화된 접근 방식을 채택하며, 통계 원리에 기초하여 품질 표현 학습을 위한 적응형 가중치 방법을 포함하고 있습니다. 마지막으로, 다양한 보기 수(view counts)와 공간 해상도(spatial resolutions)를 처리할 수 있는 심층 신경망(deep neural network)을 사용하여 프레임워크의 가치를 높였습니다.

- **Performance Highlights**: NVS-SQA는 기존의 17개 비리퍼런스(no-reference) 방법과 비교하여 평균적으로 109.5%의 SRCC(Spearman Rank Correlation Coefficient), 98.6%의 PLCC(Pearson Linear Correlation Coefficient), 91.5%의 KRCC(Kendall Rank Correlation Coefficient)에서 우수한 성능을 보였습니다. 또한, 16개의 전체 참고(full-reference) 방법을 모든 평가 지표에서 능가하여 NSS 품질 평가에서 혁신적인 기여를 하고 있음을 입증합니다.



### YO-CSA-T: A Real-time Badminton Tracking System Utilizing YOLO Based on Contextual and Spatial Attention (https://arxiv.org/abs/2501.06472)
Comments:
          8 pages,14 figures

- **What's New**: 본 논문에서는 배드민턴 랠리 로봇의 3D 경로 추적을 위한 YO-CSA 탐지 네트워크를 제안합니다. 이 네트워크는 YOLOv8s 모델의 백본(backbone), 넥(neck), 헤드(head)를 최적화하고 재구성하여 전역(global) 및 지역(local) 특징 추출 능력을 향상시킵니다. 또한, 2D 검출을 넘어 3D 공 비행 경로 예측 및 보상을 통합하여 실시간 시스템을 구축하였습니다.

- **Technical Details**: YO-CSA 네트워크는 컨텍스트(contextual) 및 스페이셜(spatial) 주의 메커니즘을 포함하여 더욱 정교한 특징 통합을 구현합니다. 이 시스템은 스테레오 비전(stereo vision)을 사용하여 2D 좌표 시퀀스를 3D 공간으로 변환하고, 과거 정보를 바탕으로 미래의 3D 좌표를 예측합니다. 또한, 결측된 중간 프레임을 보완하는 컴펜세이션 모듈(compensation module)을 포함하여 보다 완전한 경로를 제공합니다.

- **Performance Highlights**: 실험 결과, YO-CSA는 90.43% mAP@0.75의 높은 정확도를 기록하며, YOLOv8s 및 YOLO11s를 초과하였습니다. 제안된 시스템은 12개의 테스트 시퀀스에서 130 fps 이상의 속도를 유지하며 뛰어난 성능을 보여줍니다. 이러한 결과는 인간-로봇 경쟁을 위한 실제 및 신속한 배드민턴 경로 추적의 가능성을 나타냅니다.



### First Token Probability Guided RAG for Telecom Question Answering (https://arxiv.org/abs/2501.06468)
- **What's New**: 이 논문에서는 Multiple Choice Question Answering (MCQA) 문제를 해결하기 위한 새로운 Retrieval-Augmented Generation (RAG) 프레임워크를 제안합니다. 제안된 방법은 confidence score를 활용하여 하이퍼파라미터를 최적화하고, 동적으로 맥락을 조정하여 정확성을 높이는 것을 목표로 합니다. LLMs(대형 언어 모델)의 일반적인 과제인 사라진 정보 및 다중 선택 질문 문제를 보다 효율적으로 해결할 수 있는 기회를 제공합니다.

- **Technical Details**: 제안된 RAG 프레임워크는 첫 번째 토큰 확률(method) 기반으로 설계되어 사용자 질문에 적합한 정보를 검색합니다. 이 프레임워크는 key hyperparameters인 chunk number와 window size를 조정하여 모델이 선택한 답변의 확률을 정규화합니다. 또한, Phi-2라는 27억 개 매개변수를 가진 소형 언어 모델(SLM)을 사용하여 언어 이해 및 추론 능력을 효과적으로 발휘합니다.

- **Performance Highlights**: 실험 결과, 제안된 RAG 프레임워크는 MCQA 작업의 정확성을 향상시키는 가능성을 입증하였습니다. RAG의 하이퍼파라미터를 지속적으로 최적화함으로써 성능을 향상시키는 동시에 LLM의 응답 품질을 높일 수 있었습니다. 이러한 접근법은 특히 통신 분야와 같이 세부 도메인 지식이 중요한 분야에서 큰 효과를 발휘할 것으로 기대됩니다.



### MedCT: A Clinical Terminology Graph for Generative AI Applications in Healthcar (https://arxiv.org/abs/2501.06465)
- **What's New**: 본 논문에서는 중국 의료 커뮤니티를 위한 세계 최초의 임상 용어 체계인 MedCT를 소개합니다. MedCT는 임상 기반 모델인 MedBERT와 개체 연결 모델인 MedLink와 함께 제공되며, 중국 임상 데이터의 표준화 및 프로그래머블한 표현을 가능하게 합니다. 또한, MedCT 지식 그래프는 대규모 언어 모델(LLMs)에서 발생하는 환각 문제를 최소화하는 기제를 제공합니다.

- **Technical Details**: MedCT은 SNOMED CT와 유사한 용어 체계를 구축하기 위해 LLM 기반의 전체론적 접근 방식을 사용하고, 초기 용어를 SNOMED CT에서 번역하여 형성하였습니다. MedBERT라는 임상 기초 모델을 구축하고, MedLink라는 개체 인식 및 연결 모델을 훈련하여 임상 용어 처리의 성능을 크게 향상시켰습니다. 이 시스템은 이를 통해 실시간 데이터 정제 및 반복적인 최적화를 통해 임상 환경에서 효과적으로 적용됩니다.

- **Performance Highlights**: MedCT 시스템은 의미적 매칭(semantic matching) 및 개체 연결(entity linking) 작업에서 최고의 성능(SOTA)을 달성하였으며, 중국어뿐만 아니라 영어에서도 높은 정확도를 기록했습니다. 실제 임상 환경에서의 실험을 통해 MedCT의 임상 워크플로우와 환자 결과에 대한 다수의 가치를 입증했으며, LLM 기반 응용 프로그램에서 안전성과 효율성을 강조합니다.



### On the Computational Capability of Graph Neural Networks: A Circuit Complexity Bound Perspectiv (https://arxiv.org/abs/2501.06444)
- **What's New**: 본 논문은 Graph Neural Networks (GNNs)의 이론적 한계를 회로 복잡성(circuit complexity) 관점에서 분석한 새로운 접근 방식을 제시합니다. 기존 연구들은 주로 Weisfeiler-Lehman (WL) 그래프 동형성 검사를 통해 GNN의 표현력을 특성화하는 데 집중하였으나, 본 논문은 GNN 아키텍처의 회로 복잡성을 분석하여 일정 깊이의 층, 선형 또는 하위 선형 임베딩 크기 및 다항 정밀도 조건 하에서 GNN이 그래프 연결성 및 그래프 동형성 문제를 해결할 수 없음을 증명합니다. 이러한 결과는 GNN의 경험적 성공 이면에 있는 본질적인 표현력의 한계를 드러냅니다.

- **Technical Details**: GNN의 기본 활성화 함수에서부터 전체 그래프 컨볼루션 프로세스에 이르는 GNN 구성 요소의 회로 복잡성을 평가합니다. 우리는 상수 개수의 층, 다항(n-polynomial) 정밀도, d=O(n) 임베딩 크기를 가진 GNN이 균일한 TC0 회로로 근사 가능하다는 것을 보여줍니다. 이에 따라, TC0 = NC1이 아닌 이상, 이러한 GNN은 그래프 연결성 문제나 그래프 동형성 문제와 같은 문제를 해결할 수 없음을 규명합니다.

- **Performance Highlights**: 기존 GNN의 표현력을 측정하는 데 있어 Weisfeiler-Lehman (WL) 계층을 사용하는 전통적인 접근법과 달리, 본 논문은 회로 이론을 기반으로 하여 GNN의 계산적 한계를 정량화합니다. 연구 결과는 GNN이 특정한 조건에서 해결할 수 없는 문제들을 명확히 하며, GNN의 신뢰성을 높이는 데 기여할 수 있는 새로운 프레임워크를 소개합니다. 이와 같은 접근법은 다양한 GNN 모델 및 그래프 결정을 분석하는 데 활용될 수 있습니다.



### Synthetic Feature Augmentation Improves Generalization Performance of Language Models (https://arxiv.org/abs/2501.06434)
Comments:
          Accepted for presentation at IEEE SSCI 2025

- **What's New**: 본 연구에서는 제한적이고 불균형한 데이터셋에서 대형 언어 모델(LLMs)의 훈련 및 미세 조정 문제를 해결하기 위해 임베딩 공간에서 합성 샘플을 생성하여 특징을 증강하는 방법을 제안합니다. 이러한 방법은 모델의 성능을 개선하고 데이터 불균형 문제를 완화하는데 기여합니다. 다양한 오픈 소스 텍스트 분류 벤치마크에서 이 접근법의 효과를 검증하여, 모델의 강건성과 일반화 능력을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 SMOTE(합성 소수 클래스 오버 샘플링 기법)와 VAE(변분 오토인코더) 등의 기술을 사용하여 임베딩 공간에서 합성 데이터를 생성하는 방법을 탐구합니다. 비율 불균형 문제를 해결하기 위해 불충분한 클래스와 관련된 샘플의 임베딩을 생성하고, 이를 통해 균형 잡힌 훈련 데이터셋을 형성하였습니다. 이 방법은 텍스트 샘플과 해당 라벨로 구성된 데이터셋을 비선형 임베딩 함수로 매핑하여 표기된 특성을 유지하는 데 중점을 두었습니다.

- **Performance Highlights**: 실험 결과, 합성 샘플을 포함한 훈련 데이터셋은 합성 샘플을 포함하지 않은 경우보다 분류 성능이 크게 향상되었습니다. 이 연구는 다양한 벤치마크 데이터셋에서 모델의 강건성과 공정성을 높이는 데 효과적임을 입증합니다. 따라서 제안된 방법은 불균형한 데이터 시나리오 속에서도 실제 응용 분야에서 우수한 성능을 발휘할 수 있는 잠재력을 지니고 있습니다.



### Deep Learning on Hester Davis Scores for Inpatient Fall Prediction (https://arxiv.org/abs/2501.06432)
Comments:
          Accepted for presentation at IEEE SSCI 2025

- **What's New**: 본 연구는 병원에 입원한 환자의 낙상 위험 예측을 위한 새로운 기계 학습 접근 방식을 제안합니다. 기존의 Hester Davis 점수(HDS)를 기반으로 한 임계값(threshold) 접근법과 비교하여, 기계 학습을 통해 더 정교한 예측 모델을 개발합니다. 새로운 모델은 한 단계 앞선 낙상 예측(one-step ahead fall prediction) 및 시퀀스-투-포인트 예측(sequence-to-point prediction) 두 가지로 구성됩니다.

- **Technical Details**: 제안된 모델은 HDS 값을 현재 시각에서 사용하여 다음 시각의 낙상 위험을 예측하는 한 단계 앞선 예측 전략을 사용합니다. 다른 한편으로, 모든 이전 HDS 값을 활용하여 낙상 위험을 예측하는 시퀀스-투-포인트 모델은 딥 러닝(deep learning)을 활용하여 시간에 따른 패턴을 포착합니다. 특히 순환 신경망(Recurrent Neural Networks, RNN), 장단기 메모리(Long Short-Term Memory, LSTM) 네트워크, 및 게이트 순환 유닛(Gated Recurrent Unit, GRU)을 통해 진행됩니다.

- **Performance Highlights**: 기계 학습 모델은 전통적인 HDS 기반의 임계값 접근법보다 향상된 정확도로 낙상 위험을 예측할 수 있음을 보여주었습니다. 연구 결과는 이들 모델이 환자의 변화를 동적으로 반영하며, 낙상 예방을 위한 데이터 기반 접근 방식의 가능성을 강조합니다. 보다 안전한 환자 관리를 위한 예측 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Aug3D: Augmenting large scale outdoor datasets for Generalizable Novel View Synthesis (https://arxiv.org/abs/2501.06431)
Comments:
          IROS 2024 Workshop, 9 Pages, 7 Figures

- **What's New**: 최근 Photorealistic Novel View Synthesis (NVS)에 대한 관심이 높아지고 있으며, 이러한 기술이 실내에만 제한되어 있는 문제를 해결하기 위해 PixelNeRF라는 새로운 feed-forward NVS 모델을 UrbanScene3D 데이터셋에서 훈련했습니다. 본 연구에서는 Aug3D라는 증강(augmentation) 기법을 도입하여 기존의 구조 기반 방법을 사용하여 복원된 장면을 활용함으로써 모델 학습 향상을 목표로 하고 있습니다.

- **Technical Details**: Aug3D는 구조로부터의 모션(SfM) 기법을 사용하여 NVS의 효과를 높이기 위해 대규모 야외 장면 데이터셋을 학습에 적합하게 구성하는 샘플링 전략입니다. 각 클러스터에 대해 20개의 뷰에서 10개로 뷰 수를 줄였더니 PSNR이 10% 개선되었지만, 성능은 여전히 최적화되지 않았습니다. Aug3D는 새롭게 생성된 뷰를 원본 데이터셋과 결합하여 모델의 새로운 뷰 예측 능력을 향상시키는 데 효과적이라는 점을 보여주었습니다.

- **Performance Highlights**: Aug3D의 도입은 기존 야외 데이터셋에 대한 NVS 모델의 학습을 더욱 원활하게 해줍니다. 실험 결과, 가용 뷰 수를 줄였을 때 PSNR이 개선되는 성과를 지니며, 일반화 가능한 NVS 모델이 기존의 작은 실내 환경을 넘어 대규모 야외 환경에도 적용될 수 있음을 입증했습니다. 앞으로도 이러한 접근 방식이 도시 규모의 데이터를 효율적으로 활용하여 광범위한 애플리케이션에 기여할 것입니다.



### Tensor Product Attention Is All You Need (https://arxiv.org/abs/2501.06425)
Comments:
          23 pages, 5 figures

- **What's New**: 이번 논문에서는 Tensor Product Attention (TPA)이라는 새로운 어텐션 메커니즘을 제안하여, 메모리 효율성과 성능을 모두 향상시킵니다. TPA는 텐서 분해(tensor decompositions)를 활용하여 질의(query), 키(key), 값(value)의 표현을 압축이 가능하게 하고, 이는 추론 과정에서 KV 캐시의 크기를 대폭 줄이는데 기여합니다. 또한, TPA는 Rotary Position Embedding (RoPE)과의 원활한 통합을 통해 기존 LLM 아키텍처에서 쉽게 적용될 수 있습니다.

- **Technical Details**: TPA는 계층적 텐서를 활용해 질의(Q), 키(K), 값(V)을 동적으로 분해하는 방식을 채택합니다. 이를 통해, TPA는 저성능의 KV 캐시 메모리 사용을 실질적으로 운용할 수 있으며, 더욱 강력한 표현 능력을 제공합니다. 이 논문에서는 TPA를 기반으로 한 새로운 모델 아키텍처인 Tensor ProducT ATTenTion Transformer (T6)도 소개하며, 다양한 언어 모델링 작업에서의 성능 향상을 보여줍니다.

- **Performance Highlights**: T6 모델은 기존의 표준 Transformer 모델, 즉 MHA, MQA, GQA, MLA에 비해 모든 평가 메트릭에서 더 우수한 성능을 달성했습니다. 특히, T6는 KV 캐시 크기를 줄이면서도 검증 perplexity와 downstream 성능에서도 일관된 향상을 보였습니다. 이러한 성능은 TPA의 메모리 효율성 덕분에 가능해졌으며, 이는 더 긴 입력 시퀀스를 처리할 수 있는 가능성을 제시합니다.



### DiscQuant: A Quantization Method for Neural Networks Inspired by Discrepancy Theory (https://arxiv.org/abs/2501.06417)
- **What's New**: 이 논문에서는 신경망의 가중치를 양자화하는 과정에서 가중치를 최적의 방식으로 반올림하는 문제를 연구합니다. 최신 방법인 DiscQuant를 제안하여 Round-to-Nearest(RTN) 및 GPTQ와 같은 기존의 양자화 방법들보다 성능을 크게 향상시킵니다. DiscQuant는 데이터 의존적인 방식을 사용하여 양자화 모델의 질을 개선합니다.

- **Technical Details**: 양자화 과정은 (1) 가중치를 위한 저비트 복잡도 표현을 구성하고 (2) 원래 가중치를 양자화 그리드의 값으로 반올림하는 두 단계를 포함합니다. 이 연구에서는 discrepancy theory의 관점으로 이러한 반올림 문제를 탐구하고, m = poly(1/ε) 샘플에서 약한 저랭크(Hisck한, low-rank) 그라디언트를 가진 경우 양자화 모델의 예상 근사 오차가 ε 이하가 되도록 할 수 있음을 증명합니다.

- **Performance Highlights**: DiscQuant를 사용하여 Phi3mini-3.8B 모델을 3.25 비트 양자화 그리드로 반올림한 결과, GSM8k 데이터셋에서 64%의 정확도를 달성했습니다. 반면 기존의 GPTQ는 54% 및 RTN은 31%의 정확도를 기록하였습니다. 이 결과는 DiscQuant가 기술적으로 우수한 성능을 보임을 입증합니다.



### Influencing Humans to Conform to Preference Models for RLHF (https://arxiv.org/abs/2501.06416)
- **What's New**: 본 논문은 인간의 보상 함수를 근사하기 위해 인간 피드백을 통한 강화 학습(RLHF) 알고리즘 설계를 다룹니다. 연구자들은 인간의 선호 모델에 대한 적절한 가정이 없을 경우 잘못된 보상 함수 근사를 배우는 위험이 있음을 지적합니다. 이를 해결하기 위해, 연구진은 세 가지 개입 방법을 제시하여 인간의 선호가 원하는 모델에 더 잘 부합하도록 영향을 미칠 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 선호 모델로서 부분적 수익(partial return) 모델과 후회 탓(regret) 모델을 분석합니다. 연구진은 MDP(Markov Decision Process)가 작업 환경을 어떻게 나타내는지에 대한 설명을 제시하고, 연산에서 보상 함수의 역할을 명확하게 합니다. 각각의 개입 방법인 특권(privileged) 실험, 훈련된(trained) 실험, 질문(question) 실험을 통해 인간의 선호 모델 적합성을 테스트했습니다.

- **Performance Highlights**: 모든 개입 방법은 유의미한 효과를 보여주었으며, 이는 선호 데이터 품질을 개선하고 학습된 보상 함수의 조정을 촉진하는 실용적 도구가 될 수 있음을 나타냅니다. 특히, 특권 방법과 훈련된 방법은 목표 선호 모델과의 높은 부합을 이끌어냈습니다. 이러한 결과는 RLHF 분야의 실무자들에게 유용한 지침을 제공하며, 미래의 연구 방향을 제시합니다.



### FocusDD: Real-World Scene Infusion for Robust Dataset Distillation (https://arxiv.org/abs/2501.06405)
- **What's New**: 이번 논문에서 제안하는 Focused Dataset Distillation (FocusDD)은 기존의 데이터 세트 증류 방법의 한계를 극복하기 위해 새로운 접근 방식을 도입합니다. FocusDD는 고해상도 및 대규모 데이터 세트에서 데이터의 핵심 정보를 추출하여 현실적이고 다양한 증류 이미지를 생성함으로써, 다양한 네트워크 아키텍처에서 일반화 능력을 보장합니다. 이 방법은 또한 객체 탐지와 같은 다양한 다운스트림 작업에 적합하게 설계되었습니다.

- **Technical Details**: FocusDD는 두 가지 주요 단계로 구성됩니다: 정보 추출(information extraction)과 이미지 재구성(image reconstruction)입니다. 정보 추출 단계에서는 사전 훈련된 Vision Transformer (ViT)를 사용하여 주요 이미지 패치를 선택하고, 재구성 단계에서는 원본 이미지의 다운샘플링된 버전을 결합하여 최종 증류 이미지를 생성합니다. 이를 통해 데이터 세트의 다양성과 현실성을 유지하고, 모델의 일반화 능력을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, FocusDD는 ImageNet-1K 데이터 세트에서 ResNet50과 MobileNet-v2 모델이 각각 71.0%와 62.6%의 검증 정확도를 기록하여 기존 최첨단 방법보다 각각 2.8%와 4.7% 향상됨을 보여줍니다. 또한 COCO2017 데이터 세트에서 YOLOv11n과 YOLOv11s는 각각 24.4%와 32.1%의 평균 정밀도(mAP)를 달성하여 객체 탐지 작업에서의 유효성을 입증하였습니다.



### Has an AI model been trained on your images? (https://arxiv.org/abs/2501.06399)
- **What's New**: 이 논문에서는 생성 AI 모델이 특정 이미지 또는 이미지 집합으로 훈련되었는지를 판단할 수 있는 새로운 방법을 제시합니다. 이 방법은 컴퓨터 자원을 효율적으로 소모하며, 모델 아키텍처나 가중치에 대한 명시적인 지식이 필요하지 않습니다(black-box membership inference). 이는 기존 모델의 감사( auditing )와 더 공정한 생성 AI 모델의 발전 및 배포에 중요한 역할을 할 것으로 예상됩니다.

- **Technical Details**: 연구에서 활용된 기법은 대규모 웹 스크래핑을 통해 수집된 방대한 데이터셋을 기반으로 하여, 이미지의 내용을 기반으로 in-training(훈련)에 사용된 이미지와 out-of-training(훈련에서 제외된) 이미지를 비교합니다. 연구에서는 Stable Diffusion(스테이블 디퓨전), Midjourney(미드저니), DALL-E(달리)와 같은 다양한 생성 모델에 대해 이 방법을 적용했습니다. 각 이미지 쌍은 콘텐츠의 의미 면에서 유사성을 가지도록 구성되며, 이를 통해 이미지 간의 관측된 차이가 의미적 차이에 의한 것이 아님을 보장합니다.

- **Performance Highlights**: 제안된 방법은 강력한 생성 AI 모델의 훈련 데이터셋에 대한 투명성을 높이고, 생성 모델이 특정 콘텐츠를 사용하여 훈련되었는지 여부를 판단할 수 있게 해줍니다. 이는 특히 저작권 및 공정 사용 문제에 대한 논의가 있을 때 중요하게 여겨지며, AI 모델의 윤리적 개발과 관련된 중요한 기준을 제시합니다. 기술적으로, 이 방법은 계산 효율성이 뛰어나고 다양한 모델 아키텍처에 일반화될 수 있는 가능성을 가지고 있습니다.



### Unispeaker: A Unified Approach for Multimodality-driven Speaker Generation (https://arxiv.org/abs/2501.06394)
- **What's New**: 최근 개인화된 음성 생성 기술이 발전하면서 합성 음성이 실제 화자의 녹음과 더욱 유사해지고 있습니다. 그러나 다중 모달리티 기반의 화자 생성은 여전히 발전 중이며, 본 논문에서는 UniSpeaker라는 통합 접근법을 제시합니다. 이 모델은 KV-Former 아키텍처를 기반으로 한 통합 멀티모달 음성 집합기를 사용하여 다양한 음성 설명 모달리티를 공유 음성 공간으로 매핑합니다.

- **Technical Details**: UniSpeaker는 절대 음성 설명과 상대 음성 설명을 결합하여 처리할 수 있는 기능을 가지고 있으며, 입력으로 얼굴 및 텍스트 설명을 받아 화자 특성을 생성합니다. 이를 위해 다중 모달 입력을 일관성 있는 음성 공간으로 정렬하는 통합 멀티모달 음성 집합기(MFA)를 제안합니다. 또한, Soft Contrastive Learning (SoftCL)을 사용하여 음성 특성의 정렬 훈련 중에 느슨한 일대일 대비 제약을 적용하고, 다양한 모달리티의 협력적 정렬을 지원합니다.

- **Performance Highlights**: UniSpeaker는 개발한 다중 모달 음성 제어(MVC) 벤치마크를 통해 다섯 가지 기본 작업에서 평가되었으며, 이전의 모달리티 특정 모델보다 우수한 성능을 나타냈습니다. 이 벤치마크는 음성 적합성, 음성 다양성, 언어 품질을 기준으로 생성된 음성을 평가합니다. 결과적으로, 제안된 UniSpeaker는 높은 음성 제어 성능과 함께 다양하고 일관된 음성 특성을 생성할 수 있음을 보여주었습니다.



### Kolmogorov-Arnold networks for metal surface defect classification (https://arxiv.org/abs/2501.06389)
- **What's New**: 이번 논문에서는 금속 표면 결함(classifying metal surface defects) 분류에 Kolmogorov-Arnold Networks (KAN)을 적용한 사례를 제시합니다. 특히, 스틸 표면을 분석하여 균열(cracks), 이물질(inclusions), 패치(patches), 움푹 들어간 표면(pitted surfaces), 긁힌 표면(scratches)과 같은 결함을 탐지합니다.

- **Technical Details**: Kolmogorov-Arnold 정리(Kolmogorov-Arnold theorem)를 기반으로 한 KAN은 기존의 다층 퍼셉트론(multilayer perceptrons, MLPs)보다 더 효과적인 함수 근사(function approximation)를 가능하게 합니다. KAN은 스플라인 함수(spline functions)를 사용하여 더 효율적인 모델을 만들며, 이로 인해 이미지 분류에서의 성능이 향상됩니다.

- **Performance Highlights**: 결과적으로 KAN 네트워크는 보다 적은 파라미터로 컨볼루션 신경망(convolutional neural networks, CNNs)보다 뛰어난 정확도를 달성할 수 있음을 보여주었습니다. 이는 더 빠른 수렴(faster convergence)과 향상된 이미지 분류 성능을 의미합니다.



### Dynamics of "Spontaneous" Topic Changes in Next Token Prediction with Self-Attention (https://arxiv.org/abs/2501.06382)
- **What's New**: 이 논문은 인간의 대화에서 주제가 자연스럽게 변경되는 과정을 자기 attentive 기반 언어 모델의 다음 토큰 예측에서 발생하는 주제 변경과 비교합니다. 연구의 핵심은 주제의 연속성, 모호한 시퀀스, 그리고 주제 변화의 개념을 정의하고, 다중 주제 데이터셋에서 훈련된 모델들이 입력 주제에 대한 우선순위를 유지하는 것을 보여줍니다. 이는 인간의 사고 과정과 대비하여 언어 모델의 제한을 강조하고 있습니다.

- **Technical Details**: 저자들은 토큰 우선순위 그래프(Token Priority Graph, TPG)를 사용하여 주제를 정의하고, 특정 조건에서 주제 변경이 일어나는지를 분석합니다. 이들은 토큰의 우선순위가 입력 주제와 관련하여 유지됨을 증명하며, 낮은 우선순위의 토큰이 고우선순위 토큰보다 더 자주 나타날 때만 주제 변경이 가능하다고 설명합니다.

- **Performance Highlights**: 모델의 입력 길이가 증가하거나 주제가 겹칠 때 자연스러운 대화의 맥락에서 주제 변경의 가능성이 줄어드는 결과를 보여줍니다. 이러한 발견은 인간 대화의 역동성을 모방하려는 AI 대화 모델 개발에 중요한 기여를 하며, 인간과 유사한 대화 모델의 한계점을 부각시키는 데 중요한 역할을 합니다.



### Towards a Probabilistic Framework for Analyzing and Improving LLM-Enabled Softwar (https://arxiv.org/abs/2501.06370)
- **What's New**: 최근 대규모 언어 모델(LLM) 기반 시스템의 신뢰성과 검증 가능성을 보장하는 것이 소프트웨어 공학에서 중요한 도전 과제가 되고 있습니다. 이 논문에서는 LLM을 활용할 때의 다양성과 시스템 분석을 위한 확률적 프레임워크를 제안합니다. 이를 통해 의미적으로 동등한 출력의 군집에 대한 분포를 모델링하고 개선할 수 있습니다. 이러한 접근법은 자연어 문서를 정형 프로그래밍 사양으로 변환하는 자동 형식화(autoformalization) 문제에 적용되어 효용성을 보여줍니다.

- **Technical Details**: LLM은 일반적으로 다음 토큰 예측을 위한 훈련된 기계학습 모델로, 입력에 따라 단어의 확률 분포를 제공합니다. 본 논문에서는 LLM의 출력 분포를 의미-클래스 (meaning-class)로 변환하여 이의 행동을 이해하는 것을 목표로 하고 있습니다. 특히, 유형의 입력-출력 쌍을 기반으로 LLM이 어떻게 동작하는지를 이해함으로써, 잘못 정렬된 모델을 식별하고 세분화하여 보완할 수 있습니다. 이러한 모델링은 소프트웨어 테스트 및 검증 과정에서 자동화를 지원하며, 더욱 정교한 형태로 구현될 수 있습니다.

- **Performance Highlights**: 자동 형식화 문제에 대해 수행된 사례 연구를 통해 LLM의 확률적 분석과 의미 변화(transformation)가 정렬 개선을 이끌어낼 수 있음을 입증하였습니다. 이러한 접근은 LLM 지원 시스템의 신뢰성과 해석 가능성을 높이는 방향으로 나아가는 데 기여할 것으로 기대됩니다. 이 연구는 LLM이 활용될 수 있는 다양한 방식과 잠재적인 한계를 드러내며, 향후 소프트웨어 공학에서의 유용성을 더욱 부각시킬 것으로 보입니다.



### Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts (https://arxiv.org/abs/2501.06365)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 의학 문헌에서 성 편향을 완화하기 위해 성별 직업 대명사를 중립화하는 파이프라인(pipeline)을 제시합니다. 1965년부터 1980년까지의 379,000개의 PubMed 초록을 처리하여 직업에 관련된 대명사를 수정했습니다. 이를 통해 개발된 MOBERT 모델은 성 중립화된 초록으로 훈련되었으며, 기존 모델인 1965Bert와 성능을 비교하였습니다.

- **Technical Details**: MOBERT는 BERT 모델을 기반으로 하여 성별 대명사가 도입된 직업 용어와 관련된 대명사를 중립화하도록 특별히 설계된 파이프라인을 사용했습니다. 이 과정에서는 Llama-3.1을 이용하여 대명사 해소(pronoun resolution) 쿼리를 수행하며, 숙련된 주석자가 반영된 사전 정의된 분류 규칙을 바탕으로 대명사 해소 작업을 진행했습니다. 초록에 포함된 직업 용어를 정확히 작성하는 것을 목표로 설계된 어휘 사전이 사용되었습니다.

- **Performance Highlights**: MOBERT는 포함된 대명사 교체 비율에서 70%를 달성하였고, 기존 모델인 1965Bert는 4%에 그쳤습니다. MOBERT의 성능 분석 결과, 대명사 교체 정확도는 훈련 데이터 내 직업 용어의 빈도와 상관관계가 있음을 보여줍니다. 이러한 결과는 향후 데이터셋 확장 및 파이프라인 개선을 통해 보다 공정한 언어 모델링을 할 수 있는 가능성을 제시합니다.



### Ultrasound Image Synthesis Using Generative AI for Lung Ultrasound Detection (https://arxiv.org/abs/2501.06356)
Comments:
          Accepted by ISBI 2025

- **What's New**: DiffUltra는 대표적이고 다양한 데이터로 훈련된 헬스케어 AI 모델을 개발하기 위해 고안된 최신 생성형 AI 기법입니다. 이 방법은 실제 환자 데이터에서 추출한 병변의 구조적 및 위치적 특성을 활용하여 실제적인 Lung Ultrasound (LUS) 이미지를 합성합니다. DiffUltra는 데이터 다양성과 드문 사례의 발생 빈도를 높여주어 모델의 정확성을 향상시키는 데 기여합니다.

- **Technical Details**: DiffUltra는 레지온-해부학 뱅크를 사용하여 병변을 해부학적으로 적절한 위치에 배치합니다. 이 병귀 관리를 위한 확률 질량 함수(PMF)를 사용하여 병변과 주변 해부학적 구조 간의 상대적 거리를 모델링합니다. 이 기술은 단순히 병변의 픽셀 수준 세분화를 요구하지 않고 전반적인 LUS 이미지를 합성하여 효율적이고 사실적인 결과를 제공합니다.

- **Performance Highlights**: DiffUltra는 실제 환자 데이터만으로 훈련된 모델과 비교하여 폐의 응집체 탐지를 5.6% 향상시킵니다. 특히, 이 방법은 드문 사례(예: 1등급 및 4등급의 폐 응집체)의 탐지 성능을 25%까지 향상시킵니다. DiffUltra의 도입으로 인해 데이터 합성의 정밀도와 자연스러움이 크게 증가하여 헬스케어 AI 모델의 신뢰성이 높아졌습니다.



### On The Statistical Complexity of Offline Decision-Making (https://arxiv.org/abs/2501.06339)
Comments:
          arXiv version for the ICML'24 paper

- **What's New**: 이 논문은 함수 근사를 통한 오프라인 의사 결정의 통계적 복잡성을 연구하고, 확률적 컨텍스트 밴디트 및 마르코프 의사 결정 프로세스에 대해 (근사) 미니맥스 최적 비율을 설정합니다. 특히, 가치 함수 클래스의 유사 차원(pseudo-dimension)과 행동 정책에 대한 새로운 특성을 도입하여 오프라인 의사 결정 문헌의 데이터 커버리지 개념을 보완합니다. 또한, 오프라인 데이터를 온라인 의사 결정에 사용할 때의 이점을 이해하고 다양한 환경에서 거의 미니맥스 최적 비율을 보여줍니다.

- **Technical Details**: 저자들은 정책 전이 계수(policy transfer coefficients)라는 개념을 도입하여 고전적인 데이터 커버리지 개념을 포함하는 방식으로 오프라인 학습 가능성을 보다 정밀하게 설명합니다. 이 연구는 선형 및 신경망 기반의 함수 근사 클래스 등 다양한 근사 클래스에서의 오프라인 학습을 다루며, 하이브리드 오프라인-온라인 학습 설정에서도 결과를 확장합니다. 논문은 벨먼 유사 손실에 대한 균일한 번시안(Bernstein's) 불평등을 제공하고, Hegde 알고리즘의 반복 횟수를 줄이는 기술적인 문제를 해결합니다.

- **Performance Highlights**: 연구 결과는 다중 무장 밴디트, 컨텍스트 밴디트 및 마르코프 의사 결정 프로세스와 같은 다양한 오프라인 학습 문제에 대한 최적의 하한 및 상한을 제시합니다. 특히, 제안된 정책 전이 계수는 이전 문헌에서 다룬 학습 가능성 문제를 포함하며, 오프라인 데이터의 질의 중요성을 강조합니다. 이 논문은 오프라인 의사 결정 문제의 더 포괄적이고 정교한 특성을 제공하며, 실제 문제 해결에 기여할 수 있는 중요한 통찰을 제공합니다.



### Aggregating Low Rank Adapters in Federated Fine-tuning (https://arxiv.org/abs/2501.06332)
Comments:
          presented at conference this https URL

- **What's New**: 본 논문에서는 Federated Learning 컨텍스트에서 Low-Rank Adaptation(LoRA) 방법을 적용한 새로운 집계 방법을 제안합니다. 저자들은 기존의 여러 집계 방법들과 비교하며, 저희의 제안된 방법이 어떻게 성능을 개선할 수 있는지를 평가했습니다. 특히, GLUE benchmark 데이터셋에서의 성과를 기반으로 새로운 접근법의 유용성을 입증합니다.

- **Technical Details**: LoRA 방법은 각 Fine-tuning 이터레이션에서 기울기의 저차원 근사를 훈련하는 방식으로, 이를 통해 학습 파라미터 수를 대폭 줄일 수 있습니다. 본 연구에서는 LoRA를 Federated Learning 환경에 적용하며, 중앙 서버가 여러 클라이언트의 파라미터를 집계하는 방식을 채택합니다. 이 과정에서 등장하는 초기 모드의 기여도를 강화하는 체계적인 방법론을 도입하고, 실험을 통해 조기 수렴 개선 사례를 제시합니다.

- **Performance Highlights**: 제안된 집계 방법은 모델 훈련 초기 단계에서 더욱 안정적이고 빠른 수렴을 제공합니다. LoRA의 효과를 최대화하기 위해 여러 실험을 진행했으며, 실질적으로 각 클라이언트에서 훈련된 저각(adaptive) 매트릭스의 집계 결과를 분석했습니다. 특히, GLUE benchmark 데이터셋에서의 성능 평가는 제안 방법의 우수성을 보여줍니다.



### TTS-Transducer: End-to-End Speech Synthesis with Neural Transducer (https://arxiv.org/abs/2501.06320)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 논문에서는 TTS-Transducer라는 새로운 텍스트-음성 변환(text-to-speech, TTS) 아키텍처를 소개합니다. 이 아키텍처는 오디오 코덱 모델(audio codec models)과 신경 전이기(neural transducers)의 강점을 활용하여, 몬토닉 정렬(monotonic alignments)을 배워 명시적인 지속 시간 예측기(duration predictor)를 피할 수 있도록 설계되었습니다. TTS-Transducer는 여러 상태의 TTS 시스템에 비해 경쟁력 있고 견고한 대안임을 입증합니다.

- **Technical Details**: 본 시스템은 먼저 전이기 아키텍처를 사용해 첫 번째 코드북에 대해 문자화된 텍스트와 음성 코덱 토큰(tokenized text and speech codec tokens) 간의 몬토닉 정렬을 학습합니다. 이후 비자기회귀(Non-autoregressive, NAR) Transformer가 전이기 손실(transducer loss)에서 추출된 정렬 정보를 바탕으로 나머지 코드를 예측합니다. 이 시스템은 엔드-투-엔드(end-to-end) 방식으로 훈련되어 복잡한 음성 합성 절차를 간소화합니다.

- **Performance Highlights**: TTS-Transducer는 도전적인 텍스트에서 3.94%의 문자 오류율(character error rate, CER)을 달성하여, 훨씬 더 많은 데이터로 훈련된 기존 TTS 모델들을 초월하는 성능을 보입니다. 이 모델은 다양한 잔여 벡터 양자화(residual vector quantization) 코덱에서 일반화 능력이 뛰어나며, 큰 데이터로 사전 훈련을 하지 않고도 최신 TTS 모델과 비교할 수 있는 제로샷(zero-shot) 결과를 생성합니다.



### Understanding How Paper Writers Use AI-Generated Captions in Figure Caption Writing (https://arxiv.org/abs/2501.06317)
Comments:
          This paper will appear at AAAI 2025 Workshop (2nd AI4Research Workshop: Towards a Knowledge-grounded Scientific Research Lifecycle)

- **What's New**: 이 논문은 저자들이 AI가 생성한 캡션을 실제 논문 작성 과정에 통합하는 방법을 조사했습니다. 연구는 18명의 참가자가 그들의 최근 연구 작업에서 가져온 그림 두 개에 대한 캡션을 수정하는 방식으로 진행되었습니다. 참가자들은 최신 AI 모델이 생성한 캡션을 리소스로 사용하여 이 과정을 수행했으며, 이를 통해 AI가 지원하는 캡션 작성을 위한 새로운 기회를 찾아냈습니다.

- **Technical Details**: 연구에서 사용된 AI 모델은 최신의 figure captioning 모델로서, 참가자들은 이 모델이 생성한 캡션을 자유롭게 사용하여 자신의 작업에 통합했습니다. 캡션 작성 과정은 비디오로 기록되었으며, 이 기록은 인터랙션 분석을 통해 참가자들의 글쓰기 행동을 수집하는 데 활용되었습니다. 연구 결과, 참가자들은 AI가 생성한 캡션을 대부분 복사하여 수정하는 경향을 보였고, 특히 통계적 그림에 더 효과적이라는 점이 드러났습니다.

- **Performance Highlights**: 결과는 캡션 작성 과정이 단순한 문장 작성 이상의 복잡함과 다양성을 가지고 있음을 강조합니다. 현재 AI 모델은 복잡한 그림에 대한 캡션 작성 시 효과적이지 않지만, 참가자들은 AIGenerated 캡션에서 세밀한 디테일을 통합하려고 했습니다. 이 연구는 AI 시스템이 학술 작성을 지원하기 위한 기회가 있음을 제시하며, 향후 캡션 작성 도구의 설계에서 고려해야 할 요소들을 드러냅니다.



### LensNet: Enhancing Real-time Microlensing Event Discovery with Recurrent Neural Networks in the Korea Microlensing Telescope Network (https://arxiv.org/abs/2501.06293)
Comments:
          23 pages, 13 figures, Accepted for publication in the The Astronomical Journal

- **What's New**: 이 연구에서는 전통적인 microlensing 이벤트 검증 방법의 한계를 극복하기 위해 LensNet이라는 머신러닝 파이프라인을 개발했습니다. LensNet은 도구의 아티팩트로 인해 발생하는 잘못된 긍정 사례에서 진정한 microlensing 이벤트를 구별하는 데 중점을 두고 있습니다. 이 시스템은 flux에서 증가하는 경향을 감지하는 초기 알고리즘과 함께 작동하여 분류를 위해 LensNet으로 전달됩니다.

- **Technical Details**: LensNet은 KMTNet의 다중 관측소 설정에 맞게 최적화되었으며, 수작업으로 분류된 풍부한 데이터셋을 기반으로 훈련되었습니다. 이 파이프라인의 내부 모델은 멀티-브랜치 형태의 Recurrent Neural Network (RNN) 아키텍처를 사용하여 시간 시리즈의 flux 데이터와 함께 배경 정보, 관측별 PSF 품질 플래그 등을 평가합니다. 또한 LensNet은 87.5% 이상의 분류 정확도를 보여줍니다.

- **Performance Highlights**: KMTNet의 제어 환경에서 LensNet은 microlensing 이벤트에 대한 신속한 경고와 후속 관측을 가능하게 하여 초기에 사건을 탐지하고 경고할 수 있도록 설계되었습니다. 향후 훈련 세트를 확장하고 알고리즘을 조정함에 따라 성능을 더욱 개선할 수 있을 것으로 예상합니다.



### Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks (https://arxiv.org/abs/2501.06286)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 도메인 특화 작업 수행 능력을 평가합니다. 특히, HotpotQA 데이터셋을 활용한 multi-hop question answering (MHQA) 문제에 중점을 두고 있습니다. 이 문제는 여러 텍스트 출처에서 정보를 결합하고 추론하는 과제가 필요하기 때문에 모델 성능 평가의 도전적인 기준이 됩니다.

- **Technical Details**: 연구에서는 두 단계 선택기-읽기 아키텍처를 설계했으며, 각 단계는 독립된 LLM을 활용합니다. 또한, Chain of Thought (CoT)와 질문 분해(question decomposition)와 같은 방법들을 적용하여 모델 성능 개선의 영향을 조사했습니다. 이러한 방법들은 LLM의 언어 이해 및 추론 능력을 높이는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 이러한 기법들과 함께 대형 언어 모델을 통합하면 F1 스코어에서 최대 4%의 성능 향상을 이끌어낼 수 있습니다. 이는 모델이 도메인 특화 작업을 다루고 복잡한 언어를 이해할 수 있는 능력을 증명하는 결과입니다.



### Dafny as Verification-Aware Intermediate Language for Code Generation (https://arxiv.org/abs/2501.06283)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 자연어 프롬프트로부터 소스 코드를 생성하는 과정에서의 한계를 극복하기 위해, 검증 가능한 언어인 Dafny를 중간 표현으로 활용하는 방안을 제안합니다. 코드 생성 및 검증 과정을 자동화하고, 사용자와의 상호작용은 자연어로 진행되어 Dafny 코드가 사용자에게 노출되지 않도록 합니다. 이를 통해 생성된 코드의 품질을 높이고, 일반 프로그래밍 언어로 변환하기 전의 오류를 미리 잡을 수 있게 됩니다.

- **Technical Details**: 프로토타입 챗봇은 Claude Sonnet 3.5를 기반으로 하고, Dafny를 사용하여 파이썬 코드의 정확성을 높입니다. 사용자의 자연어 입력에 의해 프롬프트가 Dafny로 형식화되고, 생성된 솔루션과 증명이 Dafny 검증기를 통해 검증됩니다. 검증 및 수정 과정은 자동으로 이루어지며, 최종적으로 솔루션은 다수의 단위 테스트와 함께 목표 언어로 변환되어 사용자에게 제공됩니다.

- **Performance Highlights**: 초기 벤치마크 테스트 결과, Dafny를 활용한 코드 생성 방식이 LLM이 단순히 코드를 생성하는 것보다 더 정확하고 신뢰할 수 있는 코드를 생성할 수 있음을 보여줍니다. 프로토타입은 생성된 코드의 실행 안전성과 정확성을 보장하며, 테스트와 검증을 포함한 구현 과정을 통해 코드의 품질을 높이는 데 크게 기여하고 있습니다. 이 방식은 특히 복잡한 알고리즘이나 함수 생성 시 유용성을 발휘합니다.



### MinMo: A Multimodal Large Language Model for Seamless Voice Interaction (https://arxiv.org/abs/2501.06282)
Comments:
          Work in progress. Authors are listed in alphabetical order by family name

- **What's New**: 최신 연구에서는 MinMo라는 멀티모달 대형 언어 모델(Multimodal Large Language Model, LLM)을 소개하고, 80억 개의 매개변수를 통해 원활한 음성 상호작용을 제공하는 기술을 중심으로 하였습니다. MinMo는 음성-텍스트 정렬, 텍스트-음성 정렬, 음성-음성 정렬, 그리고 이중 상호작용 정렬의 여러 단계를 통해 훈련되어 다양한 음성 작업에서 최첨단 성능을 달성합니다. 특히, 이 모델은 사용자의 감정, 방언 및 말하는 속도를 기반으로 음성을 생성하는 등 향상된 지시 충족 기능을 지원합니다.

- **Technical Details**: MinMo는 140만 시간 이상의 다양한 음성 데이터를 통해 훈련되었으며, 여러 작업(예: Speech-to-Text, Text-to-Speech, Speech-to-Speech)을 포함합니다. 이 모델은 원래 텍스트 LLM의 기능 손실을 최소화하면서 음성 이해 및 생성을 향상시킵니다. 또한, 새로운 음성 디코더를 제안하여 단순한 구조와 높은 음성 생성 성능을 조화롭게 유지하며, 예측 지연 시간은 Speech-to-Text 시 약 100ms, 전체 이중 지연 시간은 이론상 600ms, 실제로는 800ms입니다.

- **Performance Highlights**: MinMo는 여러 공개 벤치마크(예: 구술 대화, 다국적 음성 인식, 음성 번역 등)에서 state-of-the-art 성능을 달성했습니다. 기존 모델과 달리 MinMo는 텍스트 LLM의 기능이 손실되는 것을 방지하여 음성 작업에서 우수한 성능을 보입니다. 사용자가 지정한 감정에 따라 음성을 생성하거나 특정 목소리를 모방하는 높은 지시 충족 정확도(98.4%)를 자랑하며, 전체 이중 상호작용을 원활하게 지원합니다.



### Polarized Patterns of Language Toxicity and Sentiment of Debunking Posts on Social Media (https://arxiv.org/abs/2501.06274)
- **What's New**: 최근 온라인 정치 담론에서 잘못된 정보 및 가짜 뉴스의 확산이 민주적 프로세스와 공공 참여에 심각한 문제를 야기하고 있습니다. 이 연구는 소셜 미디어 플랫폼에서의 언어 독성(language toxicity), 비관주의(pessimism), 그리고 사회적 분극화(social polarization) 간의 관계를 분석하였으며, 2016년과 2020년 미국 대통령 선거 및 QAnon 음모론의 담론을 중심으로 진행되었습니다.

- **Technical Details**: 연구 방법으로는 Twitter와 Reddit 두 가지 플랫폼에서의 데이터를 수집하였으며, 2016년부터 2023년까지의 기간을 대상으로 하였습니다. Twitter에서는 'fact check', 'fake news', 'misinformation' 등의 키워드를 사용하여 86.7 백만 개의 트윗을 수집하였고, Reddit에서는 비슷한 기간에 4.7 백만 개의 댓글을 분석하였습니다. 이를 통해 언어 독성과 비관주의의 패턴을 구체적으로 살펴보았습니다.

- **Performance Highlights**: 분석 결과, 언어 독성이 집단 비관주의와 밀접하게 연결되어 있으며, Reddit과 Twitter의 구조적 상호작용이 사용자 참여 패턴에 큰 영향을 미친다는 것을 발견했습니다. 또한, 참여가 증가할수록 언어 독성이 줄어드는 경향을 보였고, Twitter는 단일화된 담론을 촉진하는 반면 Reddit은 다양한 커뮤니케이션을 장려하는 것으로 나타났습니다. 이러한 결과는 정책 입안자와 플랫폼 설계자에게 온라인 담론을 건강하게 만드는 데 중요한 시사점을 제공합니다.



### Large Language Models for Bioinformatics (https://arxiv.org/abs/2501.06271)
Comments:
          64 pages, 1 figure

- **What's New**: 본 설문 조사에서는 생물정보학(bioinformatics)에서의 대형 언어 모델(LLM) 적용에 대한 체계적인 개요를 제공합니다. 기초 모델의 발전과 함께 생물정보학 특화 언어 모델(BioLM)의 발전 양상, 분류 및 특징을 중점적으로 분석하였습니다. 또한 ‘Life Active Factors (LAFs)’라는 새로운 용어를 소개하여 생명 과학 연구의 객관적인 이해를 돕고자 했습니다.

- **Technical Details**: 대형 언어 모델은 다양한 생물정보학 데이터의 복잡성을 효과적으로 다루는 데 성공적입니다. 특히, 변환기(transformer) 기반의 다중 헤드(self-attention) 메커니즘과 자기 지도학습(self-supervised learning) 기술을 활용해 데이터에서 의미 있는 특징을 추출합니다. 이 모델들은 단일 모달(single-modal)에서 다중 모달(multimodal) 시스템으로 발전하며, 유전체 서열(genomic sequences)과 단백질 구조(protein structures)와 같은 이질적인 데이터 세트를 통합합니다.

- **Performance Highlights**: LLM의 도입은 전통적인 계산 방법으로는 도달할 수 없었던 통찰력을 제공했습니다. 다양한 생물정보학 영역, 특히 유전자 분석(genomics), 단백질 구조 예측(protein structure prediction) 및 약물 발견(drug discovery) 등의 분야에서 현저한 성과를 이루어내고 있습니다. 그러나 데이터 개인 정보 보호, 해석 가능성 문제, 훈련 데이터의 편향 등 여러 도전 과제가 여전히 존재합니다.



### Towards smart and adaptive agents for active sensing on edge devices (https://arxiv.org/abs/2501.06262)
- **What's New**: 최근 TinyML 기술은 저전력 엣지 디바이스에서 딥 러닝 모델을 배포할 수 있게 하여 제한된 환경에서 실시간 인식을 가능하게 하고 있습니다. 하지만 이러한 딥 러닝 방법은 데이터 드리프트(data drift) 적응에 한정되어 있어 환경의 기본 동역학과 내재된 불확실성을 고려하지 못합니다. 본 연구는 2.3MB의 모델 크기로 엣지에서 실시간 인식 및 계획을 수행할 수 있는 스마트 에이전트를 제안합니다.

- **Technical Details**: 본 연구에서 제안한 시스템은 딥 러닝 기반의 객체 감지 모듈과 능동적 추론(active inference) 계획 모듈을 통합하여 작동합니다. 능동적 추론은 동적 환경의 불확실성과 변동성을 모델링할 수 있는 가능성을 제시하며, 시스템이 환경에 따라 행동을 조정할 수 있도록 돕습니다. 특히, 이 모델은 강화학습을 통해 수집한 정보를 사용하여 실시간으로 의사 결정을 진행할 수 있습니다.

- **Performance Highlights**: 제안된 사카드 에이전트는 NVIDIA Jetson 내장형 디바이스에서 IoT 카메라와 연결되어 작동하며, 능동적 추론 원칙에 기반하여 최적의 정책을 따르면서 카메라의 시야를 조정합니다. 이 시스템은 서베일런스(supervision) 및 로보틱스 애플리케이션을 위한 인간과 유사한 사카딕 움직임을 시뮬레이션하여 복잡하고 실제적인 상황에서의 응용 가능성을 강조합니다.



### What Matters for In-Context Learning: A Balancing Act of Look-up and In-Weight Learning (https://arxiv.org/abs/2501.06256)
- **What's New**: 대형 언어 모델(LLM)에서의 In-Context Learning (ICL) 메커니즘을 체계적으로 분석한 연구입니다. 이 연구에서는 훈련 데이터의 특성과 모델 아키텍처의 역할을 넘어서서 ICL의 근본적인 요소를 파악하고자 했습니다. 데이터 시퀀스에서 개념적 반복(conceptual repetitions)의 중요성을 강조하고, 이는 훈련 과정에서 안정적이고 비일시적인 ICL 성능을 보장하는 데 기여한다고 주장합니다.

- **Technical Details**: 연구에서 저자들은 깊은 자회귀 모델을 통해 훈련 데이터 시퀀스의 개념적 반복이 ICL에 미치는 영향을 조사했습니다. 특히, 2048 토큰의 컨텍스트 윈도우에서 n-gram 반복성을 관찰했고, 이는 훈련 예시 내에서의 전환 가능성을 높이는 데 중요하다고 설명합니다. 또한 복잡한 인-웨이트 러닝(in-weight learning) 목표가 ICL의 일관된 성능에 중요한 역할을 한다고 제안하며, 반복과 복잡한 목표 간의 상호작용을 연구했습니다.

- **Performance Highlights**: 저자들은 데이터 시퀀스의 단일 반복만으로도 ICL이 가능하다는 것을 발견하였고, 이는 높은 폭발성(burstiness)이나 skewed label distribution과 같은 다른 특성이 없더라도 성립됩니다. 연구 결과, LLM에서 ICL 성능은 훈련 초기에 일시적인 경향을 보일 수 있으나 반복과 복잡한 목표를 통해 안정적으로 유지될 수 있음을 확인했습니다. 이러한 발견은 대형 모델에서 ICL의 메커니즘을 이해하는 데 중요한 기여를 할 것으로 기대됩니다.



### Progressive Supervision via Label Decomposition: An Long-Term and Large-Scale Wireless Traffic Forecasting Method (https://arxiv.org/abs/2501.06255)
Comments:
          Published at Knowledge-Based Systems. arXiv admin note: substantial text overlap with arXiv:2412.00108

- **What's New**: 장기 및 대규모 무선 트래픽 예측(LL-WTF)의 중요성이 강조되었습니다. 무선 트래픽의 비정상성(non-stationarity) 문제와 도시 규모에서의 많은 네트워크 노드로 인해 큰 도전이 되는 점을 다룹니다. 이 문제를 해결하기 위해 제안된 Progressive Supervision 방법은 Label Decomposition(PSLD)을 기반으로 하며, 이는 여러 개의 쉽게 학습 가능한 구성 요소로 데이터를 분해하는 방식을 사용합니다.

- **Technical Details**: PSLD의 핵심 기술은 Random Subgraph Sampling(RSS) 알고리즘입니다. 이 알고리즘은 대규모 트래픽 데이터를 샘플링하여 효율적으로 네트워크 학습을 가능하게 합니다. PSLD는 비정상 문제를 해결하기 위해 얕은 층에서 점진적으로 학습되는 구성 요소들을 결합하는 방식을 사용하며, 이를 통해 학습 과정을 개선할 수 있습니다.

- **Performance Highlights**: 제안된 PSLD 방법은 세 가지 대규모 WT 데이터셋에서 기존의 최신 기술(SOTA) 방법들과 비교했을 때 평균 2%, 4%, 11%의 성능 향상을 나타냈습니다. 또한, PSLD는 예측 정확도뿐 아니라 처리 속도에서도 뛰어난 성능을 보입니다. 연구를 위해 개발된 오픈 소스 라이브러리인 WTFlib는 다양한 SOTA 방법을 포함하여 관련 연구의 기초를 제공합니다.



### Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words (https://arxiv.org/abs/2501.06254)
- **What's New**: 이번 논문에서는 Sparse Autoencoders (SAEs)의 새로운 평가 방식을 제안하며, polysemous words (다의어)에서 monosemantic features (단어의 의미를 명확하게 표현하는 특징)를 추출할 수 있는지 분석합니다. 기존의 MSE-L0 Pareto frontier 성능 향상이 단순히 interpretability (해석 가능성)를 증가시키지 않는다는 점을 강조합니다. 또한 Poly-Semantic Evaluations (PS-Eval)이라는 새로운 메트릭을 통해 SAEs의 성과를 정량적으로 평가할 수 있는 방법을 제시합니다.

- **Technical Details**: SAEs는 LLM의 복잡한 세 가지 차원의 활성도를 단일 의미의 피쳐로 변환하기 위해 설계되었습니다. 본 연구는 PS-Eval을 통해 polysemantic activations (다의적 활성화)에서 monosemantic features (단일 의미 특징)를 효과적으로 추출할 수 있는지 평가합니다. 실험에서는 다양한 activation functions (활성화 함수)와 latent dimensions (잠재 변수를 사용하는 차원)을 시험했고, Attention module의 깊이에 따라 해석의 정확성이 어떻게 변화하는지를 탐구했습니다.

- **Performance Highlights**: PS-Eval의 결과는 SAEs가 다양한 polysemous words에서 명확한 단일 의미의 피쳐를 추출한다는 것을 보여줍니다. 특히, deeper layers (더 깊은 층)에서는 specificity (특정성) 점수가 높아지며, Attention mechanism (어텐션 메커니즘)이 다의어를 구분하는 데 효과적임을 확인했습니다. 최종적으로, MSE와 L0 sparsity에만 집중하는 기존의 연구들이 monosemantic feature의 효과적인 추출을 간과함을 일깨워주는 성과를 도출하였습니다.



### The State of Post-Hoc Local XAI Techniques for Image Processing: Challenges and Motivations (https://arxiv.org/abs/2501.06253)
- **What's New**: 이 논문은 Explainable Artificial Intelligence (XAI)의 필요성과 접근 방식, 그리고 XAI가 직면한 도전과제를 심층적으로 논의합니다. 특히 XAI의 필요성을 강조하며, 각 분야에 특화된 사고가 요구된다고 주장합니다. 또한, 인공지능 시스템에 대한 설명의 요구가 고조되고 있는 이유와 이로 인해 발생하는 기술적 및 사회적 문제에 대해서도 이야기합니다.

- **Technical Details**: XAI는 다양한 산업 및 공공 분야에서 AI 시스템의 투명성과 해석 가능성을 높이기 위한 연구 분야로, 의료, 금융, 자율주행차 등 안전-critical한 응용 프로그램에서 특히 중요합니다. 이 논문에서는 XAI의 다양한 연구 접근 방식과 그에 따른 문제점을 다루고 있으며, AI의 결정 과정에 대한 설명이 어떻게 이루어져야 하는지에 대한 논의를 포함하고 있습니다. 각 논문들은 Domain-Specific, Human-Centric, Socio-Technical의 세 가지 카테고리로 나눌 수 있으며, 이는 XAI 관련 기술들의 목표 및 요구 사항이 어떻게 연관되는지를 설명합니다.

- **Performance Highlights**: 논문은 XAI가 AI 기술의 신뢰성과 책임성을 높이는 데 기여할 것으로 기대하며, 이를 통해 보다 안전한 AI 시스템 개발을 도모할 수 있을 것으로 보입니다. 특히, 설명 가능성이 AI 시스템의 오류나 문제를 진단하는 데 중요한 역할을 하며, 이를 통해 부작용을 예방할 수 있음을 강조합니다. 마지막으로, XAI 연구 분야의 향후 방향성을 제시하며, 기술 중심의 접근뿐만 아니라 사용자 중심의 접근 필요성도 강조합니다.



### $\text{Transformer}^2$: Self-adaptive LLMs (https://arxiv.org/abs/2501.06252)
Comments:
          18 panges, 11 figures, 9 tables

- **What's New**: 최근의 연구에서 소개된 $	ext{Transformer}^2$는 자가 적응형 대형 언어 모델(LLMs)의 새로운 프레임워크로, 기존의 훈련 방식보다 적은 자원으로 미지의 작업에 즉각적으로 적응할 수 있는 기술을 제공합니다. 이 방법은 기계 학습에서 흔히 사용되는 LoRA와 같은 기법보다 더 나은 성능을 보여주며, 가벼운 메모리 용량과 높은 효율성을 자랑합니다. $	ext{Transformer}^2$는 여러 LLM 아키텍처 및 모달리티에 걸쳐 다재다능함을 입증하였습니다.

- **Technical Details**: $	ext{Transformer}^2$의 두 가지 주요 메커니즘은 작업 속성을 식별하는 디스패치 시스템과, 강화학습(강화 학습, RL)으로 학습된 작업 특정 '전문가' 벡터들을 동적으로 조합하는 것입니다. 이 프레임워크는 단일 입력에 대한 두 번의 패스를 사용하여 모델의 가중치를 조절하는 방식을 통해 이루어집니다. 이를 위해, Singular Value Fine-tuning (SVF)라는 새로운 기법을 도입하여 모델의 가중치 행렬 내의 특이값만을 조정하여 과적합(overfitting) 문제를 줄이고 자원 소비를 최소화합니다.

- **Performance Highlights**: 대체로 다양한 실험을 통해 SVF와 전체 $	ext{Transformer}^2$ 프레임워크가 기존의 전통적인 효율적 파인튜닝 방법보다 우수하다는 것을 입증하였습니다. 특히, SVF는 적은 수의 매개변수로 토픽별 성능을 최적화하는 효과적인 전문가 벡터를 제공함으로써 비용 절감 효과를 선보였습니다. 또한, $	ext{Transformer}^2$는 시각적 질문 응답(visual question answering)과 같은 새로운 과제에서도 전반적인 성능 향상을 이루어냈으며, 진화 가능성이 있는 자가 적응형 AI 시스템 개발에 기여할 수 있는 기반을 마련하였습니다.



### Generative AI for Cel-Animation: A Survey (https://arxiv.org/abs/2501.06250)
Comments:
          20 pages

- **What's New**: 전통적인 셀룰로이드 애니메이션 제작 파이프라인은 여러 주요 단계를 포함하고 있으며, Generative AI(GenAI)의 도입이 이 과정을 혁신적으로 변화시키고 있습니다. GenAI는 inbetween 프레임 생성과 색상화, 스토리보드 제작 등에서 높은 기술 장벽을 낮추고 있으며, 다양한 제작 도구를 통해 더 많은 창작자들에게 접근성을 높이고 있습니다. 이를 통해 예술가들은 보다 창의적인 작업에 집중할 수 있는 기회를 얻게 됩니다.

- **Technical Details**: 전통 셀룰로이드 애니메이션은 스토리보드, 레이아웃 디자인, 키프레임 애니메이션, 인비트윈과 색상화 등 여러 단계로 구성되어 있습니다. 이 과정들은 많은 수작업, 전문 지식, 그리고 시간 투자가 요구되어 왔습니다. GenAI는 대규모 언어 모델(LLMs), 다중 모드 모델(MLLMs), 확산 모델(difusion models)을 포함하여 여러 반복적 작업을 자동화하여 애니메이션 제작의 효율성을 크게 향상시키고 있습니다.

- **Performance Highlights**: Generative AI의 도입은 셀 애니메이션 제작의 효율성과 창의적 가능성을 개선했습니다. 예를 들어, AniDoc는 비디오 확산 모델을 사용하여 2D 애니메이션의 inbetween과 색상화를 자동화하고, ToonCrafter는 비선형 모션과 겹침을 효율적으로 처리합니다. Netflix Japan의 실험과 CogCartoon 같은 도구는 이러한 AI 기술이 애니메이션과 같은 예술적 양식의 민주화를 가능하게 하고 있음을 보여주고 있습니다.



### Utility-inspired Reward Transformations Improve Reinforcement Learning Training of Language Models (https://arxiv.org/abs/2501.06248)
- **What's New**: 본 논문에서는 기존의 보상 함수가 평균화되는 방식의 단점을 지적하며, 특히 보상 함수의 선형 집합이 생성된 텍스트의 품질에 미치는 영향을 분석합니다. 저자들은 경제학 이론인 유틸리티 함수(Inada conditions)에서 영감을 받아 새로운 보상 변환 방법인 Inada Reward Transformation (IRT)를 제안합니다. 이 방식은 낮은 보상 값에 대한 감도를 높이고, 이미 높은 보상 값에는 낮은 감도를 적용하여 더 정교한 보상 처리를 가능하게 합니다.

- **Technical Details**: Inada Reward Transformation (IRT)는 보상 함수에 적용할 수 있는 변환 방법으로, 강한 편향을 최소화하며 특정 보상 차원을 잊지 않도록 설계되었습니다. 기존의 선형 평균화 방법과 달리, 이 방법은 다양한 보상의 상대적 중요성을 반영하여 텍스트 생성 품질을 높이고 더 적절한 모델 학습을 이끕니다. RLHF(강화 학습에 대한 인간 피드백) 과정에서 보상 모델은 사람의 선호를 예측하고, 이를 바탕으로 모델의 생성 품질을 향상시킵니다.

- **Performance Highlights**: IRT을 적용한 모델은 기존의 선형 보상 집합 방식에 비해 더 많은 도움을 주고 덜 해로운 결과를 생성하는 것으로 평가되었습니다. 정량적 및 정성적 분석을 통해, Inada 변환을 경험한 모델은 응답의 질이 향상되어 사용자 기대에 더 잘 부합하는 것으로 나타났습니다. 이는 향후 LLM(대형 언어 모델) 훈련 과정에서 보상 구조의 설계에 중요한 인사이트를 제공할 것입니다.



### A Survey on Algorithmic Developments in Optimal Transport Problem with Applications (https://arxiv.org/abs/2501.06247)
- **What's New**: 이번 논문에서는 Optimal Transport (OT) 이론이 다양한 분야에서 분포 간의 차이를 정량화하기 위한 강력한 체계로 자리 잡았음을 강조합니다. 특히, OT는 기계 학습, 데이터 과학 및 컴퓨터 비전 등에서의 응용을 포함하여, 이론적 기초인 Monge 및 Kantorovich의 고전적 정식과 이의 현대적 컴퓨팅 기법으로의 확장을 자세히 다룹니다. 또한, Sinkhorn iterations와 같은 최신 알고리즘의 효율성과 고차원 문제 해결을 위한 확장성을 강조합니다.

- **Technical Details**: OT 이론은 확률 분포의 비교 및 변환을 위한 강력한 수학적 틀을 제공합니다. 초기 Monge의 기여에서 시작하여 후에 Kantorovich에 의해 확장되고 형식화된 OT는 이산 및 연속 확률 측정을 다루기 위한 통합된 접근 방식을 제공합니다. 이 논문에서는 OT의 기본 이론 개념과 함께 각 개념의 수학적 특성을 설명하며, 최신 알고리즘적 접근 방식과 계산 기술에 대한 혁신을 심도 있게 탐구합니다.

- **Performance Highlights**: OT의 발전은 특히 대규모 데이터셋에 대한 효율성을 개선하여, 다양한 분야에서의 응용 가능성을 확장하였습니다. Wasserstein 거리 사용의 향상은 확률 분포 비교를 통한 OT의 적용 폭을 넓히고 있습니다. 그러나 OT 알고리즘의 계산 복잡성 및 해결 정확성과 실행 속도 간의 트레이드오프에 관한 논의는 여전히 진행 중이며, 윤리적 고려 사항 또한 중요한 문제로 남아있습니다.



### A partition cover approach to tokenization (https://arxiv.org/abs/2501.06246)
- **What's New**: 이번 연구에서는 토큰화(tokenization) 문제를 최적화(optimization) 목표로 설정하고, 이 문제를 NP-hard로 증명했습니다. 전통적인 Byte Pair Encoding (BPE) 방법 대신 GreedTok이라는 새로운 다항식 시간의 탐욕 알고리즘을 제안합니다. GreedTok은 기존의 조합 기반 병합 방식에서 벗어나 토큰의 순서만을 요구하여, 사용자가 선택한 사용자 정의 토큰 집합을 사용할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: 새로운 최적화 목표를 통해 GreedTok은 파라미터로 주어진 토큰 및 카운트(count) 정보를 활용하여 효율적으로 데이터를 표현합니다. 이를 통해 토큰의 수를 줄이면서도 동일한 또는 더 나은 압축 유틸리티를 달성할 수 있습니다. 이러한 접근은 전통적인 병합 방식의 제약에서 자유로워, 문제를 보다 일반적인 형태로 해결할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, GreedTok은 BPE보다 평균 3%에서 최대 5% 더 적은 토큰을 필요로 하며, 평균적으로 13% 더 적은 수의 토큰으로 유사한 압축 성능을 달성했습니다. 이러한 결과는 기존의 BPE 방법에 비해 응용 가능성 및 성능에서 뛰어난 성과를 보여줍니다. 또한, GreedTok의 성능은 잘 연구된 가중 최대 커버리지 문제(weighted maximum coverage problem)에 자연스럽게 연결되어, 이론적인 근거가 뒷받침되어 있습니다.



### Microservice Deployment in Space Computing Power Networks via Robust Reinforcement Learning (https://arxiv.org/abs/2501.06244)
Comments:
          14 pages

- **What's New**: 최근 지구 관측에 대한 수요가 증가함에 따라, 실시간 원격 감지 추론 서비스의 신뢰성을 제공하는 것이 중요해졌습니다. 본 논문에서는 발사체에 통합된 컴퓨팅 및 넓은 커버리지를 제공하는 Space Computing Power Network (Space-CPN)을 통해 저지연 요구를 충족하고자 하는 새로운 원격 감지 인공지능 응용 프로그램 배포 프레임워크를 제안합니다. 이 프레임워크는 마이크로서비스 아키텍처(microservice architecture)를 사용하여 단일 서비스 과제를 재사용 가능한 독립 모듈로 분해하고, 위성 집합체에서의 실시간 추론 성능을 달성하는 방법을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 고전적인 강화 학습(robust reinforcement learning)과 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)을 활용하여 데이터 불확실성을 다루고 있습니다. 이를 통해 분산된 마이크로서비스 배포를 최적화하여 리소스를 효율적으로 사용하면서 품질(Quality of Service, QoS)과 기능 요구 사항을 충족할 수 있습니다. 주의해야 할 점은 위성의 이질성(heterogeneity)으로 인해 각 위성이 다른 처리 능력과 통신 능력을 가지므로 가장 적합한 위성을 선택하는 알고리즘의 필요성이 강조됩니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안된 프레임워크가 리소스 소비를 최소화하고 QoS 페널티를 피하는 데 효과적임을 보여줍니다. 또한, 론칭 시 미리 할당된 작업과 긴급 임무를 동시에 처리할 수 있는 가능성을 제공하여, 지구 관측 및 재해 모니터링 작업의 효율성을 크게 향상시킬 수 있습니다. 이러한 접근은 기존의 LEO 위성 환경에서 다수의 고정된 자원 한계를 극복할 수 있는 가능성을 나타냅니다.



### Intelligent Task Offloading: Advanced MEC Task Offloading and Resource Management in 5G Networks (https://arxiv.org/abs/2501.06242)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문은 5G 기술을 활용한 다중 접근 엣지 컴퓨팅(Multi-access Edge Computing, MEC)에서 사용자 장비(User Equipment, UE)의 통신 및 계산 자원을 효율적으로 할당하는 새로운 방법론을 제안합니다. 특히 Ultra-Reliable Low Latency Communication (URLLC)와 Massive Machine Type Communication (mMTC)의 두 가지 필수 서비스를 통합하여 의사결정 프레임워크에Embedding합니다. Proximal Policy Optimization(PPO) 기법의 사용은 5G 기술의 발전에 따른 문제를 해결하기 위한 강력하고 효율적인 솔루션을 제공합니다.

- **Technical Details**: 논문의 시스템 모델은 Open Radio Access Network(O-RAN) 아키텍처를 중심으로 구축됩니다. 단일 Radio Unit(RU)이 무선 통신을 통해 사용자와 연결되며, 수신된 신호는 네트워크의 프론트 홀을 통해 Distributed Unit(DU)로 전송됩니다. DU에는 MEC 서버가 탑재되어 있으며, 이는 오프로드된 작업을 처리하고, Real-time RAN Intelligent Controller(RIC)가 xApp을 운영합니다.

- **Performance Highlights**: 제안된 모델은 엄격한 지연 제약을 가진 URLLC 사용자에게서 처리 시간을 4% 단축시키고, mMTC 사용자에 대해서는 전력 소비를 26% 줄였습니다. 이러한 성과는 기존 기준 모델에 비해 뛰어난 적응력과 성능을 보여주며, 다양한 QoS(서비스 품질) 요구 사항을 충족하는 데 있어 모델의 우수성을 증명합니다.



### Towards a scalable AI-driven framework for data-independent Cyber Threat Intelligence Information Extraction (https://arxiv.org/abs/2501.06239)
- **What's New**: 이 논문은 Cyber Threat Intelligence (CTI)의 정보 추출을 위한 새로운 프레임워크인 0-CTI를 소개합니다. 기존의 데이터 주입 모델과 달리, 0-CTI는 주석 데이터(annotation data)에 의존하지 않고도 정보 추출이 가능하다는 점에서 획기적입니다. 이 시스템은 Transformer 기반의 자연어 처리(NLP) 기술을 활용하여 CTI 보고서의 전체 텍스트 시퀀스를 처리하고 사이버 온톨로지를 추출합니다.

- **Technical Details**: 0-CTI는 감독 학습(supervised learning)과 제로샷 학습(zero-shot learning)을 모두 지원하는 모듈형 프레임워크로 설계되었습니다. 이 시스템은 엔티티(Entity) 및 관계(Relation) 추출을 위한 완전한 비주얼 데이터 없는 운영을 가능하게 합니다. CTI 정보의 추출 후, 출력 결과를 사이버 보안 분야의 정보 교환 표준인 Structured Threat Information Expression (STIX) 포맷에 맞춰 정리하여 통신의 일관성을 높입니다.

- **Performance Highlights**: 0-CTI의 감독 학습 기반 엔티티 추출기는 현재의 최신 상태(pass) 모델보다 더 나은 성능을 보여주었습니다. 이는 자원 부족 환경(low-resource environment)과 데이터가 풍부한 환경(data-rich environment) 모두에서 이 프레임워크의 강점을 강조합니다. 논문에서 제시된 연구 결과는 사이버 보안 운영에서 커뮤니케이션과 협업을 향상시키는 데 기여할 것입니다.



### Forecasting Anonymized Electricity Load Profiles (https://arxiv.org/abs/2501.06237)
- **What's New**: 전력 소비 패턴의 익명성을 보장하기 위한 방법으로, 이 글에서는 마이크로 집합(microaggregation) 기법이 에너지 수요 예측에 미치는 영향을 분석했습니다. 특히 개인 행동 데이터로 간주되는 전력 소비 데이터는 GDPR(General Data Protection Regulation)에 의해 엄격한 보호 조치가 요구됩니다. 이를 통해 개인정보를 보호하면서도 예측력에 미치는 부정적인 영향을 최소화할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 영국 파워 네트웍스의 Low Carbon London 프로젝트에서 수집된 데이터를 사용하였으며, 주로 전력 소비량의 시계열(time series) 분석을 통해 마이크로 집합 기법의 효과를 평가했습니다. 마이크로 집합은 비슷한 데이터 포인트를 그룹화하여 개별 신원을 숨기면서도 데이터 세트의 주요 속성을 유지합니다. 사용된 기법에는 k-익명성과 마이크로 집합이 포함되며, DFTMicroagg 알고리즘은 추가적인 데이터 보호를 위해 제안되었지만 이는 사용하지 않았습니다.

- **Performance Highlights**: 결과적으로 마이크로 집합을 적용한 데이터는 예측 정확도에 미치는 영향이 거의 없어 높은 데이터 유용성을 유지하며 사용자 프라이버시를 보장합니다. 연구는 에너지 공급업체가 사용자 동의 없이 스마 부하 프로파일 데이터에 접근하더라도 GDPR을 준수하면서 협력할 수 있는 가능성을 제시합니다. 이는 에너지 공급업체와 제3자가 스마트 미터 데이터를 안전하게 이용할 수 있는 방법을 제공합니다.



### Data-Driven Radio Propagation Modeling using Graph Neural Networks (https://arxiv.org/abs/2501.06236)
- **What's New**: 이 연구에서는 기존의 물리 기반 모델 대신 그래프 신경망(Graph Neural Networks)을 사용하여 무선 네트워크에서의 라디오 전파( radio propagation ) 행동을 실제 데이터로부터 직접 학습하는 새로운 방법을 제안합니다. 이 방법은 라디오 전파 환경을 그래프 구조로 변환하여 노드(node)는 위치를, 엣지(edge)는 지리적 관계를 나타내게 됩니다. 특히 이 연구는 그래프 신경망을 통해 점(point) 측정만으로도 신호 전파의 범위를 예측하는 능력을 보여줍니다.

- **Technical Details**: 본 논문에서는 그래프 신경망 아키텍처를 활용하여 전파 거동을 원활히 통합하는 방법을 제안합니다. 데이터 기반 접근 방식으로 실제 환경에서 수집한 센서 데이터를 그래프 구조로 변환하고, 이를 통해 라디오 신호의 전파를 예측하는 훈련 절차를 개발했습니다. 또한, 마스킹 출력 훈련(masked output training) 기법을 도입하여 제한된 데이터로부터 범위 맵을 생성하도록 모델을 일반화하는 방법도 소개됩니다.

- **Performance Highlights**: 타 전통적인 휴리스틱 모델과 비교할 때, 개발한 그래프 신경망은 속도와 정확성 면에서 우수한 성능을 보였습니다. 해당 연구에서 제안하는 그래프 기반 접근 방식은 물리적 모델에 비해 빠르고 정확한 범위 추정을 가능하게 하며, 현실 세계 데이터를 활용하여 커버리지 맵을 생성할 수 있는 첫 번째 사례로, 이는 미래 무선 네트워크 디자인에 중요한 기여를 할 것으로 기대됩니다.



### NextStop: An Improved Tracker For Panoptic LIDAR Segmentation Data (https://arxiv.org/abs/2501.06235)
- **What's New**: 본 연구에서는 4D panoptic LiDAR segmentation 작업에서의 추적 성능 개선을 위해 NextStop1 트래커를 소개합니다. 이 트래커는 Kalman filter 기반의 motion estimation, 데이터 연관(data association) 및 lifespan 관리를 통합하여 복잡한 환경에서도 작은 객체 추적 성능을 높입니다. 특히, 사람 및 자전거와 같은 소형 객체의 ID 스위치가 줄어들고 추적 시작 시점이 조기화되는 효과가 있습니다.

- **Technical Details**: NextStop 트래커는 SORT (Simple Online and Real-Time Tracker) 개념을 기반으로 하여 Kalman 필터를 사용해 움직임을 추정하고, 헝가리안 알고리즘을 통해 프레임 간의 객체 감지를 연관시킵니다. 이 과정에서 tracklet state 개념을 도입하여 여러 트래커 및 감지 결과의 우선순위를 정합니다. 연구진은 기존의 4D-STOP 접근법에 NextStop 트래커를 통합하여 LSTQ 메트릭에서 현저한 성능 향상을 입증하였습니다.

- **Performance Highlights**: NextStop 트래커는 작은 객체 추적에서 향상된 성능을 보여주었으며, ID 스위치가 감소하고 복잡한 환경에서의 신뢰성이 개선되었습니다. LiDAR Segmentation and Tracking Quality (LSTQ) 기준으로 볼 때, 기존 방법들에 비해 연속성이 우수하며 조기 추적 시작이 가능함을 강조합니다. 이러한 결과들은 복잡한 환경에서도 안정적인 객체 추적을 가능하게 만들어, 자율주행 및 로봇 기술에 주요한 기여를 할 것입니다.



### asanAI: In-Browser, No-Code, Offline-First Machine Learning Toolk (https://arxiv.org/abs/2501.06226)
Comments:
          7 pages, 8 figures

- **What's New**: 최근 기계 학습(Machine Learning, ML)에 대한 관심이 높아지고 있지만, 비전문가가 이를 쉽게 이해하고 적용하기 어려운 진입 장벽이 존재합니다. 이 논문에서는 asanAI라는 오프라인 우선의 오픈소스 및 노코드(No-Code) 머신 러닝 툴킷을 소개합니다. asanAI는 사용자가 복잡한 소프트웨어 설치 없이 웹 브라우저에서 직접 ML 모델을 설계하고 디버깅하며 훈련할 수 있는 환경을 제공합니다.

- **Technical Details**: asanAI는 현대 웹 브라우저가 설치된 어떤 장치에서도 작동하며, 특히 스마트폰에서도 사용이 가능합니다. 사용자는 개인 정보 보호를 보장받으면서 로컬 계산을 수행할 수 있으며, GPU 성능을 향상시키기 위해 WebGL을 활용합니다. 이 툴킷은 직관적인 시각화를 통해 네트워크 구조와 데이터 흐름을 쉽게 이해할 수 있도록 지원합니다.

- **Performance Highlights**: asanAI는 연구자들이 신속하게 머신 러닝 아이디어를 초안 및 테스트할 수 있도록 하며, 교육자들이 효과적으로 학습자를 교육하는 데 도움을 줍니다. 또한 교사들이 최신 ML 주제를 최소한의 노력으로 학급에 소개할 수 있게 해줍니다. 오픈소스 MIT 라이선스 하에 배포되어 수정이 가능하고, 산업 사용을 위한 모델 내보내기 형식도 지원하여 다양한 사용자들이 머신 러닝을 효과적으로 배워 활용할 수 있는 기반을 제공합니다.



### Detection, Retrieval, and Explanation Unified: A Violence Detection System Based on Knowledge Graphs and GA (https://arxiv.org/abs/2501.06224)
Comments:
          14 pages. Submitted to Neurocomputing

- **What's New**: 최근에 통합 멀티모달 모델을 이용한 폭력 탐지 시스템이 큰 성공을 거두고 많은 주목을 받고 있습니다. 그러나 기존 시스템들은 해석 가능성이 부족하고 단지 분류(classification) 또는 검색(retrieval) 기능만을 제공하는 한계가 있습니다. 이를 해결하기 위해 본 논문에서는 TIO(Three-in-One) 시스템이라는 새로운 해석 가능한 폭력 탐지 시스템을 제안합니다.

- **Technical Details**: TIO 시스템은 지식 그래프(knowledge graphs)와 그래프 주의 네트워크(graph attention networks)를 통합하여 탐지(detection), 검색(retrieval), 설명(explanation)의 세 가지 핵심 기능을 제공합니다. 시스템은 잠재적인 폭력 행동을 포함하는 비디오의 각 프레임을 대규모 언어 모델(large language model)의 텍스트 설명과 함께 처리하며, ImageBind를 사용하여 지식 그래프 생성을 위한 고차원 임베딩(high-dimensional embeddings)을 생성합니다. 또한, GAT를 통해 추론(reasoning)을 수행하고, 경량화된 시계열 모듈을 활용하여 비디오 임베딩 특징을 추출합니다.

- **Performance Highlights**: TIO 시스템의 효율성을 높이기 위해 자원 소비를 줄이는 여러 경량화 방법을 도입하였습니다. XD-Violence와 UCF-Crime 데이터셋에서 수행된 광범위한 실험을 통해 제안된 시스템의 효과가 검증되었습니다. 사례 연구를 통해 흥미로운 현상이 발견되었는데, 구경꾼(bystanders)의 수가 증가할수록 폭력 행동의 발생이 감소하는 경향을 보였습니다.



### FLAME: Financial Large-Language Model Assessment and Metrics Evaluation (https://arxiv.org/abs/2501.06211)
- **What's New**: 이 논문은 FLAME이라는 한국어 금융 대형 언어 모델(LLM) 평가 시스템을 도입하여, 현재 존재하는 다양한 금융 인증 및 실제 비즈니스 시나리오를 종합적으로 평가하는 방안을 제안합니다. LLM의 가치를 종합적으로 평가할 수 있는 새로운 벤치마크인 FLAME-Cer와 FLAME-Sce의 두 가지 코어 평가 기준을 포함하고 있습니다. FLAME은 14종의 공인 금융 자격증에 대한 질문으로 구성된 데이터를 활용하며, 특히 16,000개의 질문을 수집하여 정확성과 대표성 확보에 만전을 기했습니다.

- **Technical Details**: FLAME-Cer는 AFP, CAA, CFA, CPA, FRM 등 14개의 금융 인증 기준을 포함하며, 각각의 질문은 세 가지 중요도로 분류되었습니다. FLAME-Sce는 재무 지식, 문서 작성, 고객 서비스, 리스크 관리 등 10개의 주요 비즈니스 시나리오를 평가하며, 총 5,000개가 넘는 고품질 평가 질문을 포함하고 있어 실제 금융 업무에 대한 폭넓은 평가가 가능합니다. 이 시스템은 학문적 이론과 실무 경험을 결합하여 평가의 권위성과 전문성을 보장합니다.

- **Performance Highlights**: FLAME 시스템을 통해 여섯 개의 대표 LLM을 평가한 결과, Baichuan4-Finance가 대부분의 과제에서 높은 성능을 보여주었습니다. FLAME은 LLM의 실제 금융 산업 요구에 맞는 평가 작업을 정교하게 조정하였으며, 데이터 보안 및 비즈니스 규정 준수 관점에서도 평가를 진행하여 금융 산업 규제에 부합하는 모델을 보장하고 있습니다. 이 평가 시스템은 금융 시장과 규제 요구 사항이 발전함에 따라 지속적으로 업데이트되어 실질적이고 전문적인 평가 기준을 제공할 수 있도록 설계되었습니다.



### Leveraging Edge Intelligence and LLMs to Advance 6G-Enabled Internet of Automated Defense Vehicles (https://arxiv.org/abs/2501.06205)
Comments:
          8 pages, 5 figures, under (secondary/revised) review in IEEE Internet of Things Magazine

- **What's New**: 이 논문은 자율주행 기술의 군사적 응용을 위한 인공지능(AI) 및 딥러닝(DL)의 발전을 다루고 있습니다. 특히, 6G 통신 기술과 사전 훈련된 대규모 생성 언어 모델(LLM)의 통합을 통해 자율방어차량(ADVs)의 의사결정 및 통신 최적화를 강조합니다. 이를 통해 미션 환경에서의 상황 인식 및 실시간 데이터 교환을 극대화할 수 있는 가능성과 도전을 제시합니다.

- **Technical Details**: 군사용 AI는 다양한 방어 기술에 대량의 데이터를 분석하는 능력으로 인해 다양한 응용 분야에서 중요성을 갖습니다. 6G 지원 IoADV 아키텍처는 연계된 군사 차량 간의 실시간 데이터 공유와 자율 호위 작전, 동기화된 기동을 지원합니다. 또한, LLMs는 다양한 데이터 유형을 통합하여 자율주행 차량의 인식 및 의사결정을 개선하는 프레임워크를 제공합니다.

- **Performance Highlights**: ADVs는 정보 수집, 재난 구조, 보안 감시 등의 다양한 사안에서 활용됩니다. 이들은 아군 부대의 안전성을 높이고, 신속한 의사결정과 실시간 처리 능력으로 육상 작전의 효율성을 향상시킵니다. 6G 기반 통신은 초저지연 및 극도의 신뢰성을 제공하며, 이를 통해 자율주행 전략과 전투 작전을 최적화할 수 있습니다.



### How Do Artificial Intelligences Think? The Three Mathematico-Cognitive Factors of Categorical Segmentation Operated by Synthetic Neurons (https://arxiv.org/abs/2501.06196)
- **What's New**: 이번 연구에서는 시뮬레이트 신경망의 '사고 범주' 생성 메커니즘을 다루며, 이러한 범주가 정보 환경을 어떻게 분리하고 분석하는지를 탐구합니다. 연구자는 Formal neurons(형식 신경망)의 수학적 속성과 인지적 특성을 파악하여 인공지능 사고의 범주화 본질을 설명하는 데 초점을 맞추고 있습니다. 기계적 설명 가능성(mechanistic explainability)에 관한 새로운 접근법을 제시하며, 이는 신경망의 내적 작용을 보다 세밀하게 분석하는 것을 목표로 합니다.

- **Technical Details**: 연구는 인공지능 신경망의 합성(aggregated) 기능을 수학적, 인지적 관점에서 분석합니다. Formal neurons의 작동은 세 단계로 구분되며, 첫 번째 단계는 입력을 통합하는 것이고, 두 번째 단계는 가중치를 혼합하여 도출하는 단계입니다. 마지막으로, 결과적 카테고리는 후속 신경망(지속적 신경망)에 의하여 처리되어 출력됩니다. 이러한 구조는 인공지능의 범주 우선성 (categorical priority)과 응답 생산에 필수적입니다.

- **Performance Highlights**: 이 연구는 신경망이 가진 속성의 인지적, 개념적 특성이 어떻게 구성되는지를 명확하게 보여줍니다. 구체적인 방법론을 통해 데이터의 입력과 출력 간의 상관관계를 평가하고, 신경망의 내부 구조에서 발생하는 행동과 출력을 비교하는 접근을 제안합니다. 또한, 이론적 기초에 따라 신경망의 범주 생성을 통해 응답의 질을 향상시키고 인지적 편향을 줄이는 방법을 확인합니다.



New uploads on arXiv(cs.LG)

### Gradient Equilibrium in Online Learning: Theory and Applications (https://arxiv.org/abs/2501.08330)
Comments:
          Code available at this https URL

- **What's New**: 본 논문에서는 우리가 "gradient equilibrium"(경량 평형)이라고 부르는 새로운 온라인 학습(online learning) 관점을 제시합니다. 이 관점에서 평균 경량이 수렴하여 0에 도달하면 주어진 반복(iterate)들이 gradient equilibrium을 달성한 것으로 봅니다. 이 조건은 일반적으로 sublinear regret(서브선형 후회)와 관련이 없으며, 그 반대도 성립하지 않는 것을 보여줍니다. 연구에서는 gradient equilibrium이 기존의 온라인 학습 방법인 gradient descent(경량 하강법)와 mirror descent(미러 하강법)으로도 구현될 수 있음을 강조하고 있습니다.

- **Technical Details**: gradient equilibrium은 알고리즘이 제공되는 데이터 포인트에 따라 예측을 생성할 때 손실의 경량(gradient)의 평균이 0으로 수렴하는 조건을 의미합니다. 이 연구는 또한 데이터 분포가 변화하는 상황에서 블랙 박스 예측의 경향을 보정하는 방법으로 gradient equilibrium을 활용할 수 있으며, 이 과정은 간단한 포스트 hoc 온라인 하강법(post hoc online descent updates)을 통해 이루어집니다. 이는 여러 온라인 예측 문제에서 해석 가능하고 의미 있는 속성으로 전환될 수 있습니다.

- **Performance Highlights**: 논문에서는 gradient equilibrium이 적용된 여러 가지 사례를 통해 그 단순성과 확장성을 강조합니다. 이를 통해 예측의 편향을 제어하고, 다양한 통계적 특성을 고려할 수 있음을 인정하였습니다. 추가적으로, 후속 경량 업데이트(post hoc gradient updates)가 분포의 변화에 따른 예상 분위수(quantile)를 보정하는 데 사용될 수 있다고 설명합니다.



### A Similarity Measure Between Functions with Applications to Statistical Learning and Optimization (https://arxiv.org/abs/2501.08317)
Comments:
          9 pages

- **What's New**: 이번 논문에서는 두 함수 간의 유사성을 새로운 방법으로 측정하는 방안을 제시합니다. 이 측정 방법은 두 함수의 비최적성 간극(sub-optimality gaps)을 상호 변환하는 방식을 정량화합니다. 이를 통해 앞서 제시된 여러 개념의 통합이 가능함을 보여주며, 통계적 학습(statistical learning) 및 최적화(optimization) 문제에 활용될 수 있음을 강조합니다.

- **Technical Details**: 본 논문에서는 Ω라는 집합 내에서 정의된 두 함수 f와 g의 유사성을 측정하기 위해 (ε,δ) 근접성을 도입합니다. 여기서 ε와 δ는 비음수(real numbers)입니다. 함수의 근접성을 유지하기 위해서는 특정 불평등 관계가 충족되어야 하며, 이는 함수 간 최적성 간극을 비교하는데 중요한 요소로 작용합니다. 전체적으로 이 측정 방법은 함수 간의 지오메트릭 해석과 하위 수준 집합(sub-level sets)의 포용적 특징을 통해 설명됩니다.

- **Performance Highlights**: 제안된 유사성 측정 방법은 실증적 위험 최소화(empirical risk minimization)와 비정상 온라인 최적화(non-stationary online optimization) 문제에 효과적으로 적용될 수 있습니다. 이 방법은 두 함수의 근접성을 평가하는 데 유용하며, 이를 통해 환경 변화에 따른 손실 함수의 변동성을 이해할 수 있습니다. 따라서 다양한 최적화 문제 해결에 있어 통합적 접근을 가능하게 합니다.



### Path Loss Prediction Using Machine Learning with Extended Features (https://arxiv.org/abs/2501.08306)
Comments:
          4 pages, 4 figures, conference paper

- **What's New**: 이 논문에서는 전파 모델링의 정확성을 높이기 위해 이전보다 두 배 많은 특성을 포함하는 새로운 접근 방식을 제안합니다. 기하학적 정보 시스템(GIS) 데이터의 활용을 통해 무선 통신 환경을 보다 정확하게 모델링할 수 있으며, 이는 간섭을 최소화하고 예측 정확성을 높이는 데 큰 도움을 줍니다. 특히, 기계 학습 기반의 접근 방식을 통해 실험적 조건에서도 일반화된 성능을 유지하며 다양한 환경에서의 예측 가능성을 높였습니다.

- **Technical Details**: 모델의 훈련 데이터로는 영국 통신청(Ofcom)에서 수집한 무선 주파수(RF) 드라이브 테스트 데이터가 사용되었습니다. 이 데이터셋은 Path Loss 및 송신기와 수신기의 위치 및 높이에 대한 측정을 포함하고 있으며, GIS 데이터로부터 디지털 지형 모델(DTM)과 디지털 표면 모델(DSM)을 통해 경로 프로필을 생성합니다. 모든 특성은 스칼라 값으로, 기본적인 Path Loss 모델링 외에도 다수의 거리 메트릭스가 포함되어 있습니다.

- **Performance Highlights**: 여섯 개 드라이브 테스트에 대한 평균 RMSE 점수를 보여주며, 특성의 수가 증가함에 따라 RMSE가 감소하는 경향을 확인했습니다. 특히, 여섯 개의 특성을 사용할 때 RMSE는 6.95 dB까지 낮아졌고, 이는 모델의 정확성을 높이는 데 기여했습니다. 그러나 Merthyr Tydfil 테스트 홀드아웃에서는 하락세가 뚜렷하지 않았는데, 이는 훈련 데이터셋에서 경험하지 못한 언덕과 교외 환경 때문입니다.



### Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification (https://arxiv.org/abs/2501.08305)
- **What's New**: 이 논문은 Multivariate Time Series Classification (MTSC)의 그래프 기반 접근 방식을 종합적으로 비교하고 벤치마킹한 최초의 연구입니다. 연구자들은 3개의 노드 특징 정의 전략, 4개의 엣지 특징 학습 전략, 그리고 5개의 Graph Neural Network (GNN) 아키텍처를 조합하여 총 60개의 그래프 기반 MTSC 변형을 평가하였습니다. 이 벤치마크 결과는 26개의 널리 사용되는 MTSC 데이터 세트를 기준으로 두었습니다.

- **Technical Details**: MTSC에서는 각 시간 시퀀스에 대한 노드 특징, 그래프의 토폴로지 및 엣지 특징을 정의하는 과정을 포함합니다. 기존의 방법들은 주로 이진/단일값 엣지 특징을 사용하여 변수 간의 관계를 설명했지만, 이러한 연구는 높은 성능 향상을 위해 노드 특징 학습 전략의 중요성을 강화하고 있습니다. 방법론으로는 CNNs, Transformers 및 GNNs가 사용되며, 최신 모델들이 효과적인 관계 학습을 통해 MTSC 작업의 효율성을 높이고 있습니다.

- **Performance Highlights**: 실험 결과, 노드 특징 학습 전략이 MTSC 성능에 중대한 영향을 미친다는 것을 발견했습니다. 특히, 적응형 엣지 학습 방식이 다른 엣지 특징 학습 방법보다 뛰어남을 보여주었습니다. 이 연구는 다른 MTSC 응용 프로그램에 새로운 접근 방식을 탐색하고 적용할 수 있도록 코드와 데이터 파이프라인을 공개하였습니다.



### Polynomial Threshold Functions of Bounded Tree-Width: Some Explainability and Complexity Aspects (https://arxiv.org/abs/2501.08297)
Comments:
          22 pages, 3 figures. To be published in Festschrift in honor of Johann A. Makowsky

- **What's New**: 이 논문은 다변수 다항식의 트리너비(tree-width)를 활용한 새로운 방법론을 제시합니다. 이전 연구에서는 트리너비가 제한된 다변수 다항식을 연구하여 일반적으로 해결하기 어려운 문제의 다항적 해결 가능성을 보여주었습니다. 본 논문에서는 이러한 아이디어를 불리언 함수(Boolean function)에 응용하여, 다항식 임계 표현(polynomial threshold representation)을 다루고 있습니다.

- **Technical Details**: 논문은 다항식의 트리너비와 그 정의에 대한 배경을 설명합니다. 다항식은 G⁢F⁢(2) 또는 실수(real) 상에서 불리언 함수를 정확히 나타낼 수 있으며, 본 연구에서는 실수 상의 다항식에 중점을 두었습니다. 특히, 다항식의 다룰 수 있는 성질을 사용하여 의사 결정 모델과 같은 실질적인 응용 사례를 다룹니다.

- **Performance Highlights**: 본 연구는 이해 가능한 인공지능(Explainable AI, XAI) 분야에 중점을 두며, 베이지안 네트워크 분류기(Bayesian network classifiers)와 같은 확률적 그래픽 모델을 다룹니다. 베이지안 네트워크의 구조가 트리너비에 제한을 두면, 문제의 해결 가능성이 증가하며, 이는 설명 가능성과 신뢰성을 향상시키는 데 기여할 수 있습니다. 이 외에도 양의 및 일반 다항식 임계 함수의 표현력을 서로 분리하는 결과를 제공합니다.



### Can Bayesian Neural Networks Explicitly Model Input Uncertainty? (https://arxiv.org/abs/2501.08285)
Comments:
          12 pages, 11 figures, VISAPP 2025 camera ready

- **What's New**: 최근 기계 학습 모델에 들어가는 입력 데이터는 외부 노이즈나 불확실성을 동반할 수 있으며, 이러한 요소들은 종종 간과되고 모델링되지 않는다. 본 연구에서는 두 개의 입력(평균과 표준 편차)을 사용하는 Bayesian Neural Network(베이지안 신경망)을 구축하고, Ensemble, MC-Dropout, Flipout과 같은 다양한 방법론에서 입력 불확실성 추정능력을 평가했다. 결과적으로, 몇몇 불확실성 추정 방법이 입력 불확실성을 모델링할 수 있는 것으로 나타났으며, 특히 Ensemble과 Flipout이 그 성능이 우수한 것으로 밝혀졌다.

- **Technical Details**: 본 연구는 두 개의 데이터셋(Two Moons와 Fashion-MNIST)을 기반으로 실험을 수행하였다. Two Moons는 두 개의 반원으로 구성된 이진 분류 문제로, 연구자들이 모델의 신뢰성 있는 불확실성 값을 산출할 수 있는 능력을 시각화하기 위해 자주 활용하는 toy dataset(토이 데이터셋)이다. Fashion-MNIST는 10개 카테고리의 의류 아이템을 포함하는 이미지 분류의 인기 있는 벤치마크로, 패션 아이템의 그레이스케일 이미지로 구성된다.

- **Performance Highlights**: 우리의 결과는 여러 불확실성 추정 방법들이 입력의 불확실성을 제대로 예측하지 못한다는 것을 보여준다. Ensemble 방법은 때때로 입력 불확실성을 효과적으로 반영할 수 있는 유일한 방법으로 나타났다. 연구의 결과는 Bayesian Neural Network가 입력 불확실성을 명시적으로 모델링하는 데 성공하지 못할 수 있음을 시사하며, 모델이 높은 수준의 자신감을 보여주는 경향이 있음을 드러낸다.



### Decoding Interpretable Logic Rules from Neural Networks (https://arxiv.org/abs/2501.08281)
Comments:
          23 pages, 7 figures

- **What's New**: 이번 논문에서는 심층 신경망의 블랙박스 특성을 극복하기 위한 새로운 접근법인 NeuroLogic을 소개합니다. 이 방법은 신경망의 결정 과정을 해석 가능한 논리 규칙으로 변환하여 투명성과 신뢰를 높이려는 목표를 가지고 있습니다. NeuroLogic은 신경 활성 패턴을 활용하여 모델의 의사결정 과정을 포착하고, 이를 통해 복잡한 비전 신경망에서도 이해 가능한 규칙을 추출할 수 있음을 보여줍니다.

- **Technical Details**: NeuroLogic의 구조는 세 가지 단계로 나누어져 있습니다: 증류(distilling), 디코딩(decoding), 그리고 지정(grounding)입니다. 첫 번째 단계에서는 신경망의 특정 계층에서 클래스 특이적인 신경세포를 식별하여 중요한 추론 과정을 증류합니다. 두 번째 단계에서는 이러한 신경세포의 활성 값 평균을 기반으로 중요한 신경세포를 선택하고, 이들을 통해 숨겨진 프레디케이트(hiden predicates)로 변환합니다. 마지막 단계에서는 원래 입력 공간으로 프레디케이트를 고정하여, 의사결정 영역을 정의합니다.

- **Performance Highlights**: NeuroLogic은 ResNet과 같은 최첨단 비전 모델에서 전 세계적으로 해석 가능한 규칙을 추출하는 임무를 성공적으로 수행했습니다. 이 방법은 기존 연구가 부족했던 영역에서의 성과를 통해 신경망의 결정 과정을 투명하게 만들어 줍니다. 또한, 예를 들어 ResNet이 특정 이미지에서 '코', '다리', '귀'의 존재를 기반으로 개로 분류하는 방식과 같은 명확한 논리 규칙을 추출하였습니다.



### Multiplayer Federated Learning: Reaching Equilibrium with Less Communication (https://arxiv.org/abs/2501.08263)
Comments:
          43 pages, 5 figures

- **What's New**: 이 논문은 기존의 Federated Learning (FL) 모델이 실제 환경에서의 클라이언트의 전략적 행동을 잘 반영하지 못한다는 점을 지적한다. 이에 따라 Multiplayer Federated Learning (MpFL)라는 새로운 프레임워크를 도입하여, 클라이언트를 게임 이론적 맥락에서 플레이어로 모델링한다. 이는 각 플레이어가 자신의 유틸리티 함수를 최적화하고 공동 목표와의 불일치를 감안하여 균형을 찾는 것을 목표로 한다.

- **Technical Details**: MpFL 프레임워크 내에서는 Per-Player Local Stochastic Gradient Descent (PEARL-SGD) 알고리즘을 제안하며, 각 플레이어가 독립적으로 로컬 업데이트를 수행하고 주기적으로 다른 플레이어와 소통한다. PEARL-SGD는 확률적 설정에서 통신을 줄이면서 균형 근처로 수렴함을 이론적으로 분석하고 입증하였다. 본 논문에서는 이러한 이론적 발견을 수치 실험을 통해 검증하였다.

- **Performance Highlights**: PEARL-SGD는 기존의 Local SGD보다 통신 비용이 적으면서도 더 나은 균형점을 찾는 데 효과적이라는 점을 보여준다. 특히, 고객 간의 의사 결정 과정에서 발생할 수 있는 드리프트 현상을 잘 관리함으로써, 여러 클라이언트의 다양한 목표 함수에도 불구하고 안정적으로 수렴할 수 있는 장점을 가진다. 이러한 결과는 향후 다양한 응용 프로그램에서의 활용 가능성을 시사한다.



### Text-Diffusion Red-Teaming of Large Language Models: Unveiling Harmful Behaviors with Proximity Constraints (https://arxiv.org/abs/2501.08246)
Comments:
          This is an extended version of a paper published at AAAI 25

- **What's New**: 최근 대규모 언어 모델(LLM)의 테스트에서 보안 문제를 점검하기 위한 자동화된 red-teaming 방법이 제안되었습니다. 이 방법은 red-teaming LLM을 이용해 목표 LLM의 유해한 행동을 유도하는 입력을 찾아내는 것입니다. 본 논문에서는 특정 데이터셋의 참조 프롬프트와 유사한 프롬프트를 발견하는 최적화 프레임워크를 제안하고, 텍스트 확산 모델에서 영감을 받은 DART라는 블랙박스 red-teaming 방법을 소개합니다.

- **Technical Details**: DART는 임베딩 공간에서 참조 프롬프트를 변형하여, 변경량을 직접 제어하는 방법입니다. 이 방법은 특정 주제나 유해한 행동 유형에 대해 개선된 안전 평가를 가능하게 합니다. 해당 프레임워크는 기존의 auto-regressive 모델 아키텍처에서도 제대로 작동하지 않는다는 점을 지적하며, proximity constraint(근접 제약)을 적용하여 발견된 프롬프트는 참조 프롬프트와 매우 가까운 성격을 유지해야 합니다.

- **Performance Highlights**: 시스템적인 평가를 통해 DART는 기존의 방법들보다 목표 LLM에게 유해한 프롬프트를 더 효과적으로 발견하는 것으로 나타났습니다. 연구 결과, DART는 참조 프롬프트에 근접한 범위 내에서 유해한 행동을 유도하는 프롬프트를 찾을 가능성이 더 높습니다. 이를 통해 보안 메커니즘이 더 잘 작동하는 주제 및 실패 가능성이 높은 주제를 식별할 수 있었습니다.



### Privacy-Preserving Model and Preprocessing Verification for Machine Learning (https://arxiv.org/abs/2501.08236)
- **What's New**: 이번 논문은 민감한 데이터로 훈련된 머신러닝 모델의 개인정보 보호 검증을 위한 새로운 프레임워크를 제시합니다. Local Differential Privacy (LDP)와 LIME, SHAP을 통합하여, 본 프레임워크는 개인의 프라이버시를 침해하지 않으면서도 강력한 검증을 가능하게 합니다. 주요 목표는 이진 분류(binary classification)와 다중 클래스 분류(multi-class classification)에서 적절한 전처리(preprocessing)가 이루어졌는지를 확인하는 것입니다.

- **Technical Details**: 이 논문에서 제안하는 프레임워크는 LDP를 적용한 모델 설명기인 LIME과 SHAP을 활용합니다. LIME은 복잡한 모델의 예측을 설명하기 위해 더 단순한 선형 모델로 근사하는 도구로, 해석 가능성을 향상시킵니다. SHAP은 개별 특성의 기여도를 평가하여 로컬 및 글로벌 통찰력을 제공하며, 이는 다양한 데이터 셋에서 모델의 정확성을 높이며 과적합(overfitting)을 방지할 수 있습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋인 Diabetes, Adult, Student Record를 통해 수행된 평가에서, ML 기반 접근 방식이 이진 분류 작업에서 특히 효과적임을 보여주었습니다. 다중 클래스 작업에서는 임계값 기반 방법과 유사한 성능을 보였으며, 검증 정확도는 데이터셋과 노이즈 수준에 따라 달라졌습니다. 그러나 전반적으로 우리의 프레임워크는 강력한 프라이버시 보장과 실용성을 제공하며, 민감한 데이터 보호와 모델 신뢰성을 향상시키는 해결책으로 자리 잡고 있습니다.



### Dynamic Pricing in High-Speed Railways Using Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.08234)
Comments:
          37 pages, 5 figures

- **What's New**: 이 논문은 경쟁 및 협력을 갖춘 고속 여객 철도 산업의 동적 가격 전략 설계라는 중요한 과제를 다룹니다. 비제로섬 마르코프 게임(Markov Game)을 기반으로 한 다중 에이전트 강화 학습(MARL) 프레임워크가 제안되며, 이 프레임워크는 승객 의사 결정 과정을 반영하기 위해 랜덤 유틸리티 모델을 통합합니다. 고속 철도 시스템에서 동적 가격 책정에 대한 심층 강화 학습의 적용이 최근까지 제한적이었던 점에 주목하여, 다양한 철도 네트워크 구성과 수요 패턴을 모델링 할 수 있는 파라미터화된 RL 시뮬레이터인 RailPricing-RL을 소개합니다.

- **Technical Details**: RailPricing-RL 시뮬레이터는 협력적-경쟁적 환경을 조성하여 다양한 에이전트가 수요 변화에 따라 실시간으로 티켓 가격을 조정하도록 지원합니다. 이 프레임워크를 통해 에이전트 간 협력과 경쟁 간의 균형을 유지하면서 동적 가격 전략을 탐구할 수 있습니다. 실험은 Multi-Actor Attention Critic (MAAC) 및 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)와 같은 최신 MARL 알고리즘을 사용하여 고속 철도 네트워크의 동적 가격 책정을 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과는 사용자 선호가 MARL 성능에 미치는 영향과 가격 정책이 승객의 선택, 유틸리티 및 시스템 전체의 동적으로 영향을 미친다는 것을 입증합니다. 또한 이 연구는 동적 가격 책정 전략을 발전시키기 위한 기반을 제공하며, 수익성과 시스템 효율성을 일치시키는 방법에 대한 통찰을 제시합니다. 이를 통해 실제 적용을 위한 강건한 MARL 기반 솔루션 설계를 위한 중요한 통찰을 얻을 수 있습니다.



### Big Batch Bayesian Active Learning by Considering Predictive Probabilities (https://arxiv.org/abs/2501.08223)
Comments:
          7 pages, 2 figures; presented as a lightning talk at the NeurIPS Workshop on Bayesian Decision-making and Uncertainty (BDU; 2024)

- **What's New**: 이번 연구에서 BatchBALD는 Bayesian active learning에서 배치별 정보 수집 기능으로 널리 사용되지만, epistemic uncertainty(지식 불확실성)와 aleatoric uncertainty(우연적 불확실성)를 혼합하여 성능 저하를 유발한다는 점을 지적합니다. 이에 대한 해결책으로 예측 확률에 집중함으로써 이러한 문제를 해소하고, 더 나은 성능과 빠른 평가를 가능하게 하는 새로운 기능인 BBB-AL(bigger batch Bayesian active learning)을 제안합니다.

- **Technical Details**: 활성 학습의 목표는 가능한 한 데이터 효율적으로 모델을 훈련시키고, 의미 있는 데이터를 선택하여 성능을 극대화하는 것입니다. 현재 연구는 주로 분류 작업에 중점을 두고, Bayesian surrogate model을 사용하여 초기 데이터 집합에서 정보를 수집합니다. 기존의 BatchBALD는 전체 배치에 대한 상호 정보를 고려하지만, 유사한 점을 선택하는 문제에서 여전히 취약하여 새로운 방법론이 필요합니다.

- **Performance Highlights**: 제안하는 BBB-AL 방법은 예측 확률의 연속 공간에 집중하여 조합 비용을 줄이고, 이전의 BatchBALD보다 더 나은 선택 성능을 보여줍니다. 이는 더 많은 자원을 효과적으로 활용할 수 있도록 해 주며, 궁극적으로 데이터 또는 계산 자원의 예산을 효율적으로 소모할 수 있도록 도움을 줍니다.



### Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings (https://arxiv.org/abs/2501.08219)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 성능과 에너지 효율성을 최적화하기 위한 중요한 매개변수의 영향을 조사합니다. 여러 모델과 아키텍처에 대해 벤치마킹을 수행하고, 모델의 에너지 소비를 줄이면서도 성능을 유지할 수 있는 최적화 방안을 제시합니다. 이번 연구에서는 특히 추론(inference) 과정에서의 효율성 향상 방법을 중점적으로 다루고 있습니다.

- **Technical Details**: 연구에서는 T5, GPT-2, Mistral-7B, Falcon-7B등 6개의 대형 사전 훈련된 모델의 에너지 소비 및 성능을 비교 분석합니다. 다양한 NLP 작업에서의 성능을 측정하기 위해 NVIDIA A100 GPU를 활용하여 DVFS(Dynamic Voltage Frequency Scaling) 등 하드웨어 기반의 파워 절약 기법의 영향을 분석합니다. 실험은 PyTorch 및 Hugging Face 라이브러리를 사용하여 수행되었습니다.

- **Performance Highlights**: 분석 결과, Mistral-7B 및 Falcon-7B와 같은 대형 모델은 복잡한 작업에서 뛰어난 성능을 보였지만, GPT-2와 같은 소형 모델보다 4~6배 더 많은 에너지를 소모하는 것으로 나타났습니다. 하드웨어 최적화를 통해 에너지 소모를 30~50%까지 줄일 수 있는 가능성도 발견되었습니다. 연구 결과는 모델 아키텍처, 작업 요구 사항, 하드웨어 구성 간의 조화를 중요시하며 에너지 효율적인 LLM 추론 시스템 설계를 위한 실행 가능한 통찰력을 제공합니다.



### Modeling Feature Maps for Quantum Machine Learning (https://arxiv.org/abs/2501.08205)
- **What's New**: 이 연구는 다양한 양자 노이즈 모델(dephasing, amplitude damping 등)이 Quantum Machine Learning (QML) 알고리즘에 미치는 영향을 체계적으로 평가합니다. QSVC 알고리즘이 노이즈에 매우 강한 반면, Peg-QSVC 및 QNN은 더 민감하며 특히 잘못된 정보에 취약하다는 점이 강조됩니다. 또한, 양자 노이즈가 QML의 성능에 미치는 영향과 이를 극복하기 위한 전략이 중요하다는 점을 시사합니다.

- **Technical Details**: 연구에서는 NISQ 기기에서 발생하는 다양한 양자 노이즈를 포함한 시뮬레이션 모델을 사용해 QML 알고리즘의 동작을 분석합니다. QSVC, Peg-QSVC, QNN, VQC와 같은 알고리즘을 평가하며, 기능 매핑(Fature Mapping) 기법에 대한 설정도 다룹니다. 각 양자 노이즈 모델에 따른 성능 차이를 분석하여 QSVC가 가장 저항력이 뛰어난 알고리즘으로 확인되었습니다.

- **Performance Highlights**: QSVC 알고리즘은 다양한 노이즈 환경에서도 높은 정확도를 유지함으로써 유연성과 강인성을 보입니다. 반면, Peg-QSVC와 QNN은 노이즈의 종류에 따라 성능 차이가 두드러지며 이로 인해 정확성과 신뢰성이 떨어질 수 있습니다. 이 연구는 개인 맞춤형 유전학 분석의 미래를 위한 QML의 적용 가능성을 보여줍니다.



### Modeling Quantum Machine Learning for Genomic Data Analysis (https://arxiv.org/abs/2501.08193)
- **What's New**: 이번 연구에서는 다양한 feature mapping 기법을 활용하여 유전체 시퀀스 데이터의 이진 분류를 위한 양자 기계 학습(QML) 모델의 적용 가능성을 조사하였습니다. 특히, Pegasos Quantum Support Vector Classifier (Pegasos-QSVC)와 Quantum Neural Networks (QNN) 모델을 중심으로 성능을 평가하였습니다. 이 작업은 유전체 데이터 분류에 대한 QML의 혁신적인 잠재력을 강조하며, 이러한 방법론의 신뢰성과 정확성을 향상시키기 위한 지속적인 발전의 필요성을 강조합니다.

- **Technical Details**: 연구에서는 유전체 분류를 효과적으로 수행하기 위한 QML 알고리즘을 개선하고, NISQ(Nosiy Intermediate-Scale Quantum) 장치의 한계를 극복하기 위한 차원 축소 및 feature mapping 기법을 구현하였습니다. 또한, ZFeatureMap, ZZFeatureMap, 그리고 PauliFeatureMap 등의 다양한 feature map 기법을 소개하며, 이들이 QML 모델의 성능에 미치는 영향을 분석합니다. 이를 통해 대규모 데이터 세트를 처리하기 위한 양자 회로 최적화도 수행하였습니다.

- **Performance Highlights**: 실험 결과, feature mapping 기법과 QML 알고리즘 간의 상호작용이 성능에 중요한 영향을 미치는 것으로 나타났습니다. 전체 훈련 정확도에서 QNN이 가장 높은 성과를 보인 반면, Pegasos-QSVC는 높은 호출 성능을 기록하였습니다. 그러나 classifier 성능의 변동성이 높기 때문에, 특정 상황에서 지역적 출력 분포에 과적합(overfitting)될 위험이 존재함을 보여주었습니다.



### Inference-Time-Compute: More Faithful? A Research No (https://arxiv.org/abs/2501.08156)
Comments:
          7 pages, 5 figures

- **What's New**: 최근에 Long Chains of Thought (CoTs)를 생성하기 위해 특별히 훈련된 모델들이 인상적인 결과를 달성했습니다. 이러한 모델들을 Inference-Time-Compute (ITC) 모델이라고 부릅니다. 본 논문에서는 두 가지 ITC 모델(Qwen-2.5 및 Gemini-2)을 기존의 신뢰성을 평가하는 테스트를 통해 전통적인 비ITC 모델과 비교하였습니다.

- **Technical Details**: 이 연구에서는 ITC 모델들이 정보 전파 과정에서 얼마나 신뢰할 수 있는지와 관련된 여러 cue(단서)를 평가하였습니다. 예를 들어, Stanford 교수의 의견을 포함시킬 때, 모델이 그 의견을 참고하여 응답을 변경하는 경향을 분석합니다. ITC 모델들이 비ITC 모델에 비해 cue를 더 신뢰성 있게 설명하는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 실험 결과, Gemini ITC 모델은 Stanford 교수의 cue를 54%의 확률로 articulates하는 반면, 비ITC 모델은 14%에 불과했습니다. 또한, ITC 모델들이 6개의 검사된 비ITC 모델들보다 상당히 더 높은 설명률을 보였습니다. 이러한 결과는 ITC 모델들이 더욱 이해 가능한 신뢰성 있는 추론을 가능하게 할 수 있음을 시사합니다.



### FairTTTS: A Tree Test Time Simulation Method for Fairness-Aware Classification (https://arxiv.org/abs/2501.08155)
- **What's New**: FairTTTS라는 새로운 후처리 방식의 편향 완화 기법을 제안했습니다. 이 기법은 기존의 Tree Test Time Simulation(TTTS) 방법에서 영감을 받아 정확성과 공정성을 동시에 개선할 수 있도록 설계되었습니다. FairTTTS는 결정트리의 결정 경로를 조정함으로써 불이익을 받은 샘플에 대해 공정한 예측 결과를 제공합니다.

- **Technical Details**: FairTTTS는 Monte Carlo 시뮬레이션 기반의 기법을 사용하여 보호 속성과 관련된 노드에서 결정 경로를 확률적으로 변경합니다. 이러한 조정은 의사결정 기준에 가까운 샘플이 더 높은 확률로 변경될 수 있도록 도와주며, 이를 통해 결정트리의 구조에 내재된 편향을 줄이는 데 기여합니다. 또한 FairTTTS는 훈련 프로세스에서 공정성 조정을 분리하여 다양한 환경에 유연하게 통합될 수 있도록 합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, FairTTTS는 기존의 방법보다 공정성 개선에서 평균 20.96%의 성과를 보여주었으며, 0.55%의 정확도 향상을 이뤄냈습니다. 반면, 경쟁 방법들은 일반적으로 0.42%의 정확도 감소를 초래했습니다. 이러한 결과는 FairTTTS가 예측 성능을 향상시키면서 더 공정한 의사결정을 가능하게 함을 입증합니다.



### Data-driven inventory management for new products: A warm-start and adjusted Dyna-$Q$ approach (https://arxiv.org/abs/2501.08109)
Comments:
          7 pages, 2 figures

- **What's New**: 이 논문에서는 새로운 제품에 대한 재고 관리를 위한 향상된 강화 학습 알고리즘을 제안합니다. 기존 데이터가 부족한 상황에서도 모델 기반과 모델 프리 접근을 균형 있게 결합하여 Dyna-Q 구조를 개선하고 학습 시간을 단축할 수 있는 방법을 소개합니다. 특히 '와씀 스타트 정보(warm-start information)'를 이용하여 초기 학습 안정성을 높이고 최적의 정책 추정의 변동성을 줄일 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 Dyna-Q 알고리즘의 구조를 기반으로 하여 새로운 제품의 수요 분포가 알려지지 않았을 때 최적의 주문 수량을 결정하는 데 초점을 맞추고 있습니다. 이 알고리즘은 '탐색-수렴(search-then-convergence)' 과정을 통해 의사결정 과정에서 발생할 수 있는 불확실성을 줄이며, 제한된 과거 데이터를 보완하기 위해 기존의 유사 제품에서 수집한 시뮬레이션 데이터를 활용합니다. 이를 통해 모델 기반과 모델 프리 컴포넌트 간의 비율을 동적으로 조절하는 기능이 포함되어 있습니다.

- **Performance Highlights**: 케이스 스터디를 통해 제안된 알고리즘은 평균 일일 비용을 최대 23.7% 줄이고, 학습 시간을 최대 77.5% 단축하는 성과를 보였습니다. 또한, 30일 간의 테스트에서도 가장 낮은 총 비용과 총 비용의 변동성, 그리고 상대적으로 낮은 부족 비율을 기록하며 기존 알고리즘들에 비해 높은 성능을 입증했습니다.



### Optimal Policy Adaptation under Covariate Shif (https://arxiv.org/abs/2501.08067)
- **What's New**: 이 논문에서는 예측 모델의 이전 연구와는 달리, 정책 학습(Policy Learning)에 대한 접근법을 제안합니다. 우리는 소스 데이터셋에서 완전한 정보와 타겟 도메인에서의 공변량(covariates)만을 이용하여 최적 정책(optimal policy)을 학습하는 방법론을 제시합니다. 또한, 보상(reward)의 효율적인 영향 함수(efficient influence function) 및 준모수 효율 경계(semiparametric efficiency bound)를 도출하여, 고급 추정기(doubly robust estimator)를 구축합니다.

- **Technical Details**: 우리는 공변량 이동(covariate shift) 상황에서 원인론적(causal) 관점으로 정책 문제를 공식화합니다. 잠재적 결과(potential outcomes) 프레임워크를 사용하여 보상 및 최적 정책을 정의합니다. 제안된 추정기는 두 가지 Robust성을 가지며, 비대칭 분산(asymptotic variance) 측면에서 최적의 정규 추정기입니다. 추가로, 공변량 이동과 개념 이동(concept shift)을 동시에 고려하는 감도 분석(sensitivity analysis) 방법을 제안합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 정책 학습 접근법이 보상을 보다 정확하게 추정하고, 이론적으로 최적의 정책에 가깝게 다가가는 결과를 보여줍니다. 결과적으로 정책 전이 과정의 복잡성을 극복하면서, 다양한 도메인 간의 전이 가능성을 크게 향상시키는 것이 확인되었습니다. 이러한 연구는 추천 시스템, 정밀 의학 및 강화 학습 등 다양한 분야에 응용될 가능성이 큽니다.



### UFGraphFR: An attempt at a federated recommendation system based on user text characteristics (https://arxiv.org/abs/2501.08044)
- **What's New**: 이 논문에서는 사용자 텍스트 특성을 활용하여 사용자 관계 그래프를 기반으로 한 개인화된 연합 추천 알고리즘(UFGraphFR)을 제안합니다. 이는 사용자 간의 관계를 활용하여 추천 시스템의 성능을 향상시키면서 개인정보 보호를 극대화할 수 있는 방법을 모색합니다. 기존의 중앙 집중형 추천 시스템의 데이터 유출 위험을 회피하기 위해, 연합 학습을 통한 데이터 사용의 '유용한 비가시성'이라는 개념에 중점을 둡니다.

- **Technical Details**: UFGraphFR 알고리즘은 사용자의 텍스트 특성 설명을 통해 사용자 관계 그래프를 구축하며, Transformer 메커니즘을 도입하여 사용자 상호작용의 시퀀스를 모델링합니다. 이 알고리즘은 사용자 개인 정보를 이용하지 않고도 추천의 정확성을 높이기 위해 사전 훈련된 임베딩 모델을 사용합니다. 우리가 제안하는 방법은 텍스트 기반 추천 패러다임으로, 이는 사용자 특징의 효과적인 표현을 가능하게 합니다.

- **Performance Highlights**: 기초 실험 결과, UFGraphFR 모델은 여러 벤치마크 데이터셋에서 기존 추천 시스템 대비 우수한 성능을 나타냈습니다. 유저 텍스트 특성 그래프를 활용한 성능 향상 및 개인정보 보호 기술의 통합이 효과적임을 입증하였습니다. 이 모델은 추천 시스템이나 연합 학습 시나리오에서 실질적인 효과를 발휘함을 보여주었으며, 상용화 가능성을 지닌 것으로 평가됩니다.



### PolyLUT: Ultra-low Latency Polynomial Inference with Hardware-Aware Structured Pruning (https://arxiv.org/abs/2501.08043)
Comments:
          arXiv admin note: text overlap with arXiv:2309.02334

- **What's New**: 이번 연구에서는 FPGA(임베디드 게이트 어레이) 배포를 위해 다변량 다항식(multi-variate polynomials)을 기본 빌딩 블록으로 사용하는 새로운 DNN(딥 뉴럴 네트워크) 훈련 방식을 제안합니다. 과거의 연구들은 FPGA LUT(루ookup table)에 주로 선형 함수(linear functions)를 하드코딩하여 초저지연(low-latency) 구현을 시도했습니다. 하지만 이 논문은 보다 다양한 기능을 사용할 수 있는 LUT의 잠재력을 활용하여 다항식 평가(polynomial evaluation)를 숨김으로써 효율성을 극대화합니다.

- **Technical Details**: 제안된 방법은 LUT 내에서 다항식 평가를 최소한의 오버헤드로 수행하도록 설계되었습니다. 연구진은 소프트 로직(soft logic)을 적용하여 계층 수를 줄이면서도 동등한 정확도를 유지합니다. 또한, 연구에서 제안하는 맞춤형 하드웨어 인식 그룹 정규화기(hardware-aware group regularizer)를 사용하여 특정 희소성 패턴(sparsity pattern)을 유도하고 뉴런당 소수의 입력을 유지하도록 합니다.

- **Performance Highlights**: PolyLUT의 효과는 세 가지 작업: 네트워크 침입 탐지(network intrusion detection), CERN의 대형 하드론 충돌기(jet identification), 그리고 MNIST 데이터셋을 통해 입증되었습니다. 이 방식은 선형 함수보다 소프트 로직의 사용을 크게 줄이면서 지연(latency)과 면적(area) 개선을 가져옵니다. 이러한 접근법은 DNN을 FPGA에 최적화하여 응답성을 높이는 데 기여할 것으로 보입니다.



### Convergence Analysis of Real-time Recurrent Learning (RTRL) for a class of Recurrent Neural Networks (https://arxiv.org/abs/2501.08040)
- **What's New**: 이번 논문에서는 RNN(순환 신경망)을 훈련시키기 위한 새로운 접근법인 RTRL(Real-time Recurrent Learning)을 제안합니다. 기존의 TBPTT(Truncated Backpropagation Through Time) 방식이 갖는 경량화된 방법론의 한계를 넘어서, RTRL은 데이터 시퀀스의True Gradient를 따르며 정확한 최적화를 수행합니다. 특히 긴 데이터 시퀀스에서의 효율성을 입증하여, 금용 데이터 분석 같은 분야에서 실용적인 가능성을 보여줍니다.

- **Technical Details**: RTRL 알고리즘은 각 시점에서의 연산량이 N×d_θ로 주어지며, 여기서 N은 은닉 유닛의 수, d_θ는 파라미터의 수입니다. RTRL은 온라인 알고리즘으로, 각 시점에서 파라미터의 진화 방식을 예측하고 업데이트합니다. 본 문헌에서는 RTRL의 수렴성 분석을 통해 기하학적으로 ergodic한 데이터 시퀀스와 RNN 은닉층의 정지 분포에 도달하는 과정을 수학적으로 증명하였습니다.

- **Performance Highlights**: RTRL 알고리즘의 성능을 실험적으로 검증하여, 실제 데이터 시퀀스를 다룰 때 TBPTT 대비 더 높은 정확도를 기록했습니다. 특히 금용 데이터를 분석할 때, RTRL은 긴 시퀀스에서도 효과적으로 작동하며, 실시간 업데이트와 최적화가 가능하다는 장점을 지니고 있습니다. 이러한 결과들은 RTRL이 RNN 훈련에 있어 TBPTT의 대안으로 효과적임을 입증합니다.



### Enhanced SPS Velocity-adaptive Scheme: Access Fariness in 5G NR V2I Networks (https://arxiv.org/abs/2501.08037)
Comments:
          This paper has been submitted to IEEE Journal. The source code has been released at: this https URL

- **What's New**: 이번 논문은 5G NR V2I 모드-2에서 차량의 속도를 고려한 데이터 접근의 형평성을 보장하기 위한 새로운 접근 방식을 제안하고 있습니다. 기존의 SPS(스ensing-based semi-persistent scheduling) 메커니즘에 변화를 주어, 차량의 속도에 맞춰 선택 창을 조정하는 방안을 도입하였습니다. 이는 통신 자원의 공정한 분배를 위한 최적화 문제를 수학적으로 공식화함으로써, 다양한 차량 속도에 따른 접근 형평성을 개선하고자 합니다.

- **Technical Details**: 제안된 시스템 모델은 고속도로 환경에서 RSU(roadside unit)가 배치된 상황을 고려하고, 차량은 Poisson 프로세스를 통해 RSU의 통신 범위에 접근합니다. 차량들은 5G NR V2X 모드-2를 사용하여 데이터를 전송하며, SPS 메커니즘을 통해 자원을 할당받습니다. 이러한 과정에서, 각 차량은 선택 창 내에서 자원을 확인하고, 인가되지 않은 자원은 제외함으로써 전송 자원을 결정합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 스킴은 차량 간의 속도 차이에 따른 데이터 전송 공정성을 효과적으로 향상시킴을 보여줍니다. 선택 창의 조정이 다양한 차량 속도에 따라 자원의 할당을 조정하여, 고속 차량과 저속 차량 간의 정보 접근성을 균등하게 만들었습니다. 이를 통해, 교통 안전성을 개선하고 올바른 의사 결정의 가능성을 높이는 결과를 가져왔습니다.



### An AI-driven framework for rapid and localized optimizations of urban open spaces (https://arxiv.org/abs/2501.08019)
Comments:
          36 pages

- **What's New**: 이번 연구에서는 도시 공간에서의 열적 쾌적도와 안전성 인식에 중요한 영향을 미치는 Sky View Factor(SVF)와 가시성을 최적화하기 위한 AI 주도의 프레임워크를 소개합니다. 기계 학습 모델(MLMs)과 설명 가능한 AI 기술을 통합하여 저비용의 점진적 디자인 개선을 지원합니다. 이는 전 세계 최적화 방법이 아닌, 더 낮은 계산 비용으로 지역 조정을 가능하게 합니다.

- **Technical Details**: 이 프레임워크는 SHapley Adaptive Explanations(SHAP)를 사용하여 특성의 중요성을 분석하고, Counterfactual Explanations(CFXs)를 통해 최소 디자인 변경을 제안합니다. 연구에서 테스트된 다섯 개의 MLM 중 XGBoost가 가장 정확한 성능을 보였으며, 건물 폭, 공원 면적, 주변 건물의 높이가 SVF에 중요한 요소로 확인되었습니다.

- **Performance Highlights**: CFX 접근법은 1분 내에 최적의 결과를 달성하며, 5% RMSE 오차로 유전자 알고리즘보다 훨씬 빠른 성능을 보여줍니다. 이 대조 연구는 데이터 기반 통찰력과 실용적인 리트로핏(retrofitting) 솔루션을 통해 다양한 도시 맥락에서의 환경 품질 향상에 기여합니다.



### Maximizing Uncertainty for Federated learning via Bayesian Optimisation-based Model Poisoning (https://arxiv.org/abs/2501.08002)
Comments:
          14 pages

- **What's New**: 이번 논문에서는 Federated Learning (FL) 내에서의 악의적인 모델 오염 공격을 탐구하는 새로운 방법론을 제안합니다. 저자들은 새로운 모델 오염 공격 기법인 Delphi를 소개하며, 모델 출력을 최대한의 불확실성(unexpected uncertainty)으로 만드는 것을 목표로 합니다. 모델의 첫 번째 은닉층(hidden layer)과 관련된 불확실성을 활용하여 공격 기법의 효과성을 증명하고, FL 시스템의 취약성을 강조합니다.

- **Technical Details**: Delphi 방법론은 Bayesian Optimisation 및 Least Squares Trust Region 두 가지 최적화 기술을 사용하여 최적의 오염 모델 매개변수를 탐색합니다. 이 기법은 특정 뉴런에만 집중하여 모델 매개변수를 조작함으로써 불확실성을 유도합니다. 이 과정에서 저자들은 KL Divergence를 통해 예측 확률 분포의 거리를 최소화하여 불확실성을 정량화하는 방법을 수립했습니다.

- **Performance Highlights**: 실험 결과, Delphi-BO 방법이 Delphi-LSTR보다 높은 불확실성을 유도함을 보여줍니다. Delphi-BO는 매 학습 라운드에서 가장 중요한 뉴런을 선택하여 공격을 수행하며, 그에 따라 모델의 예측 신뢰도가 절반으로 감소하는 것으로 나타났습니다. 이러한 결과는 FL 모델이 모델 오염 공격에 대해 얼마나 취약한지를 보여줍니다.



### Unsupervised Feature Construction for Anomaly Detection in Time Series -- An Evaluation (https://arxiv.org/abs/2501.07999)
Comments:
          7

- **What's New**: 본 논문에서는 시계열 데이터에서의 이상 감지 (anomaly detection) 성능을 향상시키기 위해, 기존의 자동 변수 구성 라이브러리를 통한 새로운 대표성(tabular representation)과 초기 시간적 표현 중 어떤 것이 더 효과적인지를 실험적으로 조사합니다. Isolation Forest와 Local Outlier Factor라는 두 가지 인기 있는 탐지기를 사용하여, 새로운 tsfresh 라이브러리를 통해 계산된 표현이 이상 감지 성능을 개선하는지를 분석하였습니다.

- **Technical Details**: 연구의 초점은 단일 변수 시계열(univariate time series)에서의 점별 감지(point-wise detection)입니다. 문헌에서 세 가지 이상 유형인 포인트 이상(point anomalies), 집합 이상(collective anomalies), 그리고 문맥 이상(contextual anomalies)을 구분합니다. 저자는 오프라인 분석(offline analysis)에 중점을 두고, 모든 데이터를 학습에 사용하여 탐지 모델이 배치되지 않는 탐색적 분석(exploratory analysis) 상황을 고려합니다.

- **Performance Highlights**: 5개의 다양한 데이터 세트에서 실험한 결과, tsfresh를 사용하여 계산된 새로운 표현이 Isolation Forest의 성능을 유의미하게 향상시킨다는 것을 보여주었습니다. 검토된 가장 적절한 방법은 Sliding Window 기법을 사용하여 시계열을 테이블로 변환하고, 이를 통해 시간이 지남에 따라 나타날 수 있는 이상 신호를 추적합니다.



### Reward Compatibility: A Framework for Inverse RL (https://arxiv.org/abs/2501.07996)
- **What's New**: 이 논문에서는 Inverse Reinforcement Learning (IRL)을 보상 호환성(reward compatibility)의 관점에서 연구하였습니다. 보상 호환성은 전문가의 시연과 주어진 보상 간의 호환성을 정량화하는 새로운 프레임워크로, 이 프레임워크를 통해 보상에 따라 전문가 정책의 성능이 최적 성능에 얼마나 가까운지를 측정할 수 있습니다. 이 연구는 기존의 이론적 IRL 문헌에서 가장 일반적으로 사용되는 가능 보상 집합(feasible reward set) 개념을 일반화하여, 이론적으로 더욱 효율적인 IRL을 대규모 MDPs에서 구현할 수 있는 길을 열었습니다.

- **Technical Details**: IRL 문제는 전문가의 행동 시연으로부터 보상 함수를 추론하는 과정이며, 현재까지 다양한 효율적인 알고리즘이 개발되었습니다. 하지만, 기존 IRL 공식은 보상이 무수히 많은 해를 가질 수 있어 이론적 특성이 poorly defined하여 단일 보상의 학습에 어려움이 있었습니다. 본 연구에서는 대규모 또는 연속 상태 공간에서의 IRL 문제를 다룰 수 있도록 보상 호환성 프레임워크를 제안하였으며, 이를 통해 온라인 및 오프라인 설정에서 최적 및 비최적 전문가에 대한 효율적인 알고리즘을 개발했습니다.

- **Performance Highlights**: 이 프레임워크의 주요 기술적 기여는 CATY-IRL(CompATibilitY for IRL) 알고리즘을 통해 IRL 분류 문제를 해결하고, CATY-OFF-IRL(CompATibilitY for OFFline IRL) 알고리즘을 통해 오프라인 IRL 문제를 다룰 수 있다는 점입니다. 두 알고리즘 모두 최적 및 비최적 전문가를 대상으로 한 실험에서 입증된 효율성을 지니고 있습니다. 이러한 접근법은 보상 호환성 개념의 적용을 통해 기존 IRL 문제 설정을 넘어서는 다양한 응용 가능성을 제시합니다.



### Derivation of Output Correlation Inferences for Multi-Output (aka Multi-Task) Gaussian Process (https://arxiv.org/abs/2501.07964)
- **What's New**: 이 논문에서는 Multi-task Gaussian process (MTGP)에 대한 친절한 유도 및 그라디언트(guides) 설명을 제공합니다. 기존의 MTGP의 유도 과정은 이해하기 어려운 부분이 있었으나, 저자들은 그러한 장벽을 제거하려고 노력했습니다. 이를 통해 더 많은 연구자들이 MTGP를 활용하여 하이퍼파라미터 최적화(hyperparameter optimization)와 같은 문제를 해결할 수 있기를 희망하고 있습니다.

- **Technical Details**: MTGP는 Bonilla et al. (2007)에서 처음 제안된 모델로, 이는 여러 개의 입력 출력 조합을 다루는 데 효과적입니다. 이 논문은 MTGP의 수학적 모델링에 있어 가우시안 분포(Gaussian distribution), 공분산 행렬(covariance matrix), 그리고 확률 밀도 함수(probability density function)에 관한 기초적인 수식을 제공합니다. 특히, 하이퍼파라미터(hyperparameters)와 관련된 그래디언트의 유도 과정을 구체적으로 다루고 있습니다.

- **Performance Highlights**: MTGP는 다목적 최적화(multi-objective optimization), 제약 조건 최적화(constrained optimization), 다중 충실도 최적화(multi-fidelity optimization)와 같은 분야에서 효과적임이 입증되었습니다. 그러나 기존 문헌에서는 MTGP의 유도 과정에 대한 세부 사항이 부족하여 연구자들이 이를 활용하는 것이 어려웠습니다. 연구자들이 이 논문의 내용을 통해 MTGP의 이점을 확장하고, BoTorch와 같은 기존의 하이퍼파라미터 최적화 프레임워크를 개선하는 데 기여할 것으로 기대됩니다.



### Gandalf the Red: Adaptive Security for LLMs (https://arxiv.org/abs/2501.07927)
Comments:
          Niklas Pfister, Václav Volhejn and Manuel Knott contributed equally

- **What's New**: 이 논문은 기존의 프롬프트 공격에 대한 방어 평가에서 두 가지 중요한 요인을 간과하고 있음을 지적합니다. 첫째, 공격자의 동적 행동을 고려해야 하며 둘째, 엄격한 방어책이 정당한 사용자의 사용성에 미치는 부담을 고려해야 합니다. 이를 위해 논문에서는 D-SEC(Dynamic Security Utility Threat Model)을 제안하고 있습니다. D-SEC는 공격자와 정당한 사용자를 분리하고 다단계 상호작용을 모델링하여 보안과 유용성을 최적화하는 방법을 제시합니다.

- **Technical Details**: D-SEC는 공격의 적응성을 포착하고 보안과 유용성을 함께 고려할 수 있도록 설계되었습니다. 이를 통해 LLM 애플리케이션에 대한 방어를 평가하는 보다 합리적이고 포괄적인 방법을 제공하고자 합니다. 또한, 대규모 데이터셋을 생성하기 위해 crowd-sourced 플랫폼인 Gandalf를 구성하여 다양한 공격 시나리오를 수집하여 공개했습니다. 이러한 데이터셋은 279,000개의 프롬프트 공격을 포함하고 있으며, 이는 LLM의 보안성을 평가하는 데 유용한 자원이 될 것입니다.

- **Performance Highlights**: 이 논문은 보안과 유용성 간의 균형을 맞추는 것이 LLM 애플리케이션 설계에 있어서 필수적임을 강조합니다. 특히, 제한된 응용 프로그램 영역, 방어 심화(defense-in-depth), 적응형 방어 등 세 가지 전략이 효과적이라는 것을 보여줍니다. 이러한 전략들은 적절하게 조합되어 사용될 경우 사용성을 저해하지 않으면서도 효과적인 방어를 제공합니다. 이 연구 결과는 LLM 애플리케이션의 보안성과 유용성을 최적화하는 데 중요한 통찰을 제공합니다.



### Phase of Flight Classification in Aviation Safety using LSTM, GRU, and BiLSTM: A Case Study with ASN Datas (https://arxiv.org/abs/2501.07925)
Comments:
          Aviation Safety, Deep learning algorithms, Flight phase, NLP, ASN, and Classification

- **What's New**: 이번 연구는 항공 산업에서 안전을 보장하기 위해 NLP(자연어 처리)와 AI 모델을 활용하여 항공 사고 분석 내러티브에서 비행 단계(phase of flight)를 분류하는 방법을 제안합니다. 기존의 데이터에 의존하지 않고, 비구조화된 데이터에서 비행 사고 후 사건 내러티브를 통해 비행 단계를 추론할 수 있는 가능성을 탐구하고 있습니다.

- **Technical Details**: 연구에서는 다양한 딥러닝(deep learning) 모델의 분류 성능을 평가하였으며, 단일 RNN 기반 모델에서는 LSTM이 63%의 정확도(accuracy)를 기록했습니다. BiLSTM은 64%의 정확도를 보였고, GRU는 60%의 정확도와 63%의 정밀도(precision)를 나타냈습니다. Joint RNN 모델들을 활용하여 GRU-LSTM, LSTM-BiLSTM, GRU-BiLSTM가 각각 62%, 67%, 60%의 정확도를 기록함으로써 다양한 아키텍처의 조합을 통한 예측 능력 향상을 보여주었습니다.

- **Performance Highlights**: 모델 평가 결과, 단일 및 조합 모델들이 비행 단계 분류에 적합한 성능을 나타냈습니다. LSTM-BiLSTM 모델은 특히 67%의 높은 정확도로 주목을 받았으며, 이는 항공 산업 이해관계자들이 사전 예방적 의사 결정을 위한 귀중한 통찰력을 제공하는 데 기여할 수 있습니다. 이 연구는 NLP와 딥러닝 모델을 활용하여 항공 안전성을 향상시키는 데 있어 중요한 진전을 의미합니다.



### Aviation Safety Enhancement via NLP & Deep Learning: Classifying Flight Phases in ATSB Safety Reports (https://arxiv.org/abs/2501.07923)
Comments:
          NLP, Aviation Safety, ATSB, Deep learning, Flight phase. arXiv admin note: substantial text overlap with arXiv:2501.01694

- **What's New**: 이 연구에서는 항공 안전을 분석하기 위해 자연어 처리(Natural Language Processing, NLP)와 다양한 딥러닝 모델을 활용했습니다. 특히, LSTM, CNN, 양방향 LSTM(Bidirectional LSTM, BLSTM), 간단한 순환 신경망(simple Recurrent Neural Network, sRNN)을 이용하여 안전 보고서의 비행 단계를 분류합니다.

- **Technical Details**: 모델은 높은 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수를 기록했습니다. 특히 LSTM 모델은 각각 87%, 88%, 87%, 88%의 성능을 발휘하여 가장 높은 성과를 보여주었습니다. 이 연구는 안전 사고 분석의 자동화를 위한 효과성을 강조합니다.

- **Performance Highlights**: 자연어 처리와 딥러닝 기술의 통합은 항공 안전 분석을 혁신적으로 강화할 수 있는 가능성을 제시합니다. 이를 통해 타겟화된 안전 조치를 마련하고 보고서 처리 과정을 간소화할 수 있습니다.



### Optimal Classification Trees for Continuous Feature Data Using Dynamic Programming with Branch-and-Bound (https://arxiv.org/abs/2501.07903)
Comments:
          In the proceedings of AAAI-25

- **What's New**: 이 논문에서는 최적 분류 트리(Optimal Decision Tree, ODT)를 계산하기 위한 새로운 알고리즘인 ConTree를 제안합니다. ConTree는 기존의 동적 프로그래밍(Dynamic Programming, DP)과 가지치기 기법을 결합하여 연속형 데이터에 적합하도록 최적화되었습니다. 이전의 방법들과 비교할 때 ConTree는 숫자 데이터에서 ODT를 효율적으로 계산할 수 있으며, 특히 깊이 4의 트리를 다루는 데 있어 상당한 향상을 보여줍니다. 이 연구 결과는 5% 이상의 테스트 정확도를 달성하고, 실행 시간에서도 현저한 이점을 제공합니다.

- **Technical Details**: ConTree는 branch-and-bound(BnB) 알고리즘과 동적 프로그래밍(DP) 기법을 사용하여 최적의 분류 트리를 직접 계산하도록 설계되었습니다. 새로운 하한(bounding) 기법을 도입하여 유사한 분할을 제거함으로써 검색 공간을 줄임과 동시에 계산 비용을 최소화합니다. 또한, 깊이 2의 트리를 위한 특화된 서브루틴을 제안하여 숫자 데이터를 정렬할 수 있는 특성을 활용합니다. 이 알고리즘적 개선들은 Quant-BnB와 기존의 MIP 및 SAT 접근법에 비해 실행 시간을 크게 단축시키는 데 기여하고 있습니다.

- **Performance Highlights**: ConTree 알고리즘의 실험 결과는 특히 깊이 4의 ODT를 계산하는데 있어 뛰어난 성능을 보여줍니다. 다양한 데이터셋에 대한 실험에서 ConTree는 Quant-BnB 대비 1배 이상의 성능 향상을 달성하였으며, CART 방법에 비해 평균 5% 더 높은 정확도를 기록하였습니다. 이는 동적 프로그래밍과 branch-and-bound 기법의 조합이 연속형 데이터 처리에서 매우 효과적임을 나타냅니다. 이러한 결과들은 ConTree가 제한된 시간 내에 규모가 큰 데이터셋에서도 유용하게 사용될 수 있음을 증명합니다.



### Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision (https://arxiv.org/abs/2501.07886)
Comments:
          22 pages, 10 figures

- **What's New**: 이번 연구는 신뢰할 수 없는 인간의 감독 하에서도 언어 모델( LM)의 사후 훈련(post-training)이 효과적으로 수행될 수 있는지를 검토합니다. 실험에서는 소규모 모델과 시간 제약이 있는 인간을 이용해 신뢰할 수 없는 시연과 비교 피드백을 시뮬레이션했습니다.

- **Technical Details**: 연구의 핵심은 반복 레이블 정제(iterative label refinement, ILR)입니다. ILR은 비교 피드백을 활용해 인간의 시연을 대체할 모델 생성 대안을 결정한 후, 업데이트된 데이터로 SFT(supervised finetuning)를 통해 모델을 재훈련합니다. 이러한 접근법은 DPO(reinforcement learning from human feedback)보다 더 나은 성능을 발휘합니다.

- **Performance Highlights**: SFT+ILR 방법은 수학, 코딩 및 안전한 지시사항 준수와 같은 여러 작업에서 SFT+DPO보다 우수한 결과를 나타냈습니다. 연구 결과는 LM이 복잡한 작업을 수행하는 데 있어, RLHF가 항상 최선의 방법이 아닐 수 있음을 시사하며, 오히려 훈련 데이터를 개선하는 방향으로 피드백을 사용하는 것이 더 효과적일 수 있음을 보여줍니다.



### Mitigating Algorithmic Bias in Multiclass CNN Classifications Using Causal Modeling (https://arxiv.org/abs/2501.07885)
Comments:
          7 pages; 6 figures

- **What's New**: 이 논문은 다중 클래스 분류 문제에서 알고리즘 편향을 감지하고 완화하기 위한 인과 모델링(causal modeling) 절차를 설명합니다. FairFace 데이터셋을 기반으로 하여 DeepFace 사전 훈련된 모델로 생성된 감정 레이블을 보완하여 사용했습니다. 커스텀 CNN 모델을 개발하여 성별 편향을 식별하였으며, 성별에 따라 분류 결과에서 '행복' 또는 '슬픔'으로 더 분류된 여성과 '중립'으로 더 분류된 남성을 분석했습니다. 이후 one-vs-all (OvA) 기법을 적용하여 감정 클래스마다 인과 모델을 구성하여 CNN의 예측 확률을 조정했습니다.

- **Technical Details**: FairFace 데이터셋은 인종, 성별, 연령 속성이 균형있게 표현된 97,698개의 이미지를 포함하고 있습니다. CNN 모델 아키텍처는 4개의 합성곱 층(convolutional layers)과 정규화 및 차원 축소를 위한 최대 풀링(max-pooling)으로 구성됩니다. 훈련 시 Adam 최적화기와 학습률 0.0001을 사용하였으며, 조기 종료(early stopping) 메커니즘으로 과적합을 방지하였습니다. 모델은 7개의 감정 클래스에 대한 확률을 출력하고, 가장 높은 확률을 가진 감정을 예측 레이블로 선택합니다.

- **Performance Highlights**: 훈련 후 테스트 세트에서 모델의 전반적인 정확도는 58.3%로 평가되었습니다. 인과 모델링을 통해 편향이 완화된 분류 결과가 성별 공정성을 향상시켰으며, 전체 정확도에 미치는 영향은 미미하거나 약간 개선된 것으로 나타났습니다. 이 연구는 알고리즘 공정성과 정확성이 반드시 상충하지 않음을 강조합니다.



### MD-Syn: Synergistic drug combination prediction based on the multidimensional feature fusion method and attention mechanisms (https://arxiv.org/abs/2501.07884)
- **What's New**: 이번 연구에서는 복합 질병에서 promising therapeutic efficacy를 보이는 약물 조합 치료법에 대한 새로운 접근 방식인 MD-Syn을 제안했습니다. MD-Syn은 멀티 차원(feature fusion)과 멀티 헤드 attention mechanisms를 기반으로 하여, 약물 쌍과 세포주(triplet)를 통해 약물 조합의 synergistic 효과를 예측합니다. 또한, 사용자 설정이 가능한 웹 포털을 개발하여 보다 넓은 커뮤니티가 이 모델에 접근할 수 있도록 했습니다.

- **Technical Details**: MD-Syn은 1차원 feature embedding 모듈(1D-FEM)과 2차원 feature embedding 모듈(2D-FEM), 그리고 심층 신경망(dnn) 기반의 분류기로 구성되어 있습니다. 이 프레임워크는 1차원 및 2차원 feature 공간을 동시에 고려하여 약물 조합의 예측 정확도를 높입니다. 또한 multi-head attention 메커니즘을 통해 다양한 feature 측면에서 embedding을 학습하고, 상호작용하는 핵심 feature 요소에 집중하여 해석 가능성을 증가시킵니다.

- **Performance Highlights**: MD-Syn은 5-fold cross-validation에서 0.919의 AUROC를 기록하며 기존의 최첨단 방법보다 뛰어난 성능을 보였습니다. 또한, 두 개의 독립적인 데이터 세트에서도 유사한 결과를 보여주어 모델의 일반화 능력을 입증했습니다. MD-Syn은 약물 조합의 synergistic 효과를 우선시하는 해석 가능한 프레임워크로 자리 잡고 있습니다.



### Distributed Nonparametric Estimation: from Sparse to Dense Samples per Termina (https://arxiv.org/abs/2501.07879)
- **What's New**: 이 논문은 분산된 터미널들이 여러 개의 독립적이고 동일하게 분포된 샘플(i.i.d. samples)을 보유하고 있는 비모수(nonparametric) 함수 추정의 통신 제약 문제를 다룹니다. 이전 연구들이 단일 샘플 또는 밀집 샘플에서만 다루던 문제를 해결하며, 샘플 수에 따른 최적 속도의 상전이를 명확히 정의합니다. 이로 인해 비모수 함수 추정의 전반적인 상황에서의 해답을 제공하게 되었습니다.

- **Technical Details**: 저자들은 최적 속도를 달성하기 위해 파라메트릭 밀도 추정(parametric density estimation) 문제의 프로토콜을 활용하여 계층적 추정 프로토콜(layered estimation protocol)을 설계하였습니다. 정보 이론적 방법과 강력한 데이터 처리 불평등(strong data processing inequalities)을 이용하여 프로토콜의 최적성을 증명하였으며, 고전적인 볼과 빈 모델(balls and bins model)을 포함하였습니다. 이 접근 방식은 다양한 특별 상황에 대해 즉각적으로 최적 속도를 보여줍니다.

- **Performance Highlights**: 밀도 추정, 가우시안(Gaussian), 이진(binary), 포아송(Poisson), 비이심을 포함한 회귀 모형(heteroskedastic regression models)과 같은 여러 특수 사례에 대한 최적 속도가 즉각적으로 도출됩니다. 이로 인해, 기존의 연구들에 비해 훨씬 넓은 범위의 문제를 해결할 수 있게 되었습니다. 논문에서 제안한 방식은 효과적인 정보 전송 및 샘플링 효율성을 동시에 보장합니다.



### Prediction Interval Construction Method for Electricity Prices (https://arxiv.org/abs/2501.07827)
- **What's New**: 전기사용량 시장에서 정확한 전기 가격 예측은 매우 중요합니다. 본 논문은 전기 가격 예측을 위한 새로운 예측 구간(construction method)을 제안합니다. 조건부 생성적 적대 신경망(conditional generative adversarial network)을 활용하여 전기 가격 시나리오를 생성하고, 이를 통해 예측 구간을 구성하는 방식입니다.

- **Technical Details**: 제안된 방법은 생성된 여러 시나리오를 쌓아(probability densities) 전기 가격 불확실성을 정확하게 반영합니다. 또한, 날씨 요소의 변동성(volatility level)에 기반한 강화된 예측 메커니즘이 도입되어 가격 스파이크나 변동성이 큰 가격을 처리합니다. 케이스 스터디(case study)를 통해 제안된 방법의 효과를 검증합니다.

- **Performance Highlights**: 이 방법은 예측 구간 내 각 가격 시나리오의 확률 밀도(probability density)를 제공하여 불확실성을 더 잘 반영합니다. 특히, 가격 변동성 및 스파이크를 효과적으로 처리할 수 있는 장점이 있는 것으로 나타났습니다.



### STTS-EAD: Improving Spatio-Temporal Learning Based Time Series Prediction via (https://arxiv.org/abs/2501.07814)
Comments:
          11 pages

- **What's New**: 본 논문은 STTS-EAD라는 새로운 기법을 제안하여 다변량 시계열 예측의 학습 과정에 이상 탐지를 통합합니다. 이 접근 방식은 기존의 이상 탐지 방법들이 철저히 스페티오템포랄(spatio-temporal) 정보를 활용하지 못한 한계를 극복하고, 이를 통해 모델 학습 시 예측 변동성을 향상시키고자 합니다. 특히, STTS-EAD는 훈련 과정에서 이상 상태를 즉각적으로 처리하여, 정확도를 높이고, 다양한 실제 데이터 세트를 적용한 결과에서 타 모델보다 우수한 성능을 입증하였습니다.

- **Technical Details**: STTS-EAD는 스페티오템포랄(spatio-temporal) 정보를 활용하여 각 시계열에 대한 시간 및 공간 임베딩(embedding)을 구축하고, 이를 토대로 예측 및 이상 탐지 과정에서 피드백을 통해 상호 최적화합니다. 즉, 모델 훈련 중 EAD 모듈이 작동해 이상 점수를 계산하고, 스페티오템포랄 학습 기반 시계열 예측 모델의 잔차 정보를 바탕으로 이상 상태를 탐지합니다. 또한, STTS 모델은 다이나믹한 시계열 수에 쉽게 적응할 수 있어, 실제 비즈니스 환경에서 유연하게 활용될 수 있습니다.

- **Performance Highlights**: STTS-EAD는 공개된 주식 데이터 세트와 다국적 커피 체인의 두 가지 실제 판매 데이터 세트에서 실험을 수행하여, 예측 성능이 크게 향상됨을 보여주었습니다. 이러한 실험을 통해 STTS-EAD는 동일한 환경에서 여러 기준 모델을 초과하는 성능을 발휘하였으며, 훈련 단계에서 탐지된 이상 상태를 효과적으로 처리하여 추론 성능을 향상시키는 것으로 확인되었습니다. 이를 통해 다변량 시계열 예측의 신뢰성을 높이고자 하는 연구에 기여하고 있습니다.



### Conformal mapping Coordinates Physics-Informed Neural Networks (CoCo-PINNs): learning neural networks for designing neutral inclusions (https://arxiv.org/abs/2501.07809)
- **What's New**: 이번 연구는 물리적으로 정보가 포함된 신경망(Physics-Informed Neural Networks, PINNs)을 활용하여 중성 포괄체(neutral inclusion) 문제를 해결하는 새로운 접근 방식을 제안합니다. 새로운 방법인 CoCo-PINNs는 복소 해석 기법을 PINNs에 통합하여 불완전 경계 조건이 있는 임의 모양의 포괄체 설계를 위한 역문제를 효과적으로 처리합니다. 이 연구는 표준 PINNs의 성능 한계를 극복하고, 중성 포괄체 설계에 필수적인 문제를 해결하는 혁신적인 기법을 제공합니다.

- **Technical Details**: CoCo-PINNs에서는 복소 해석 기법을 활용하여 경계 포인터 외부에서 샘플 포인트에서의 해결값을 손실 함수 설계에 포함시킵니다. 기존 PINNs는 경계 매개변수를 신경망으로 근사하는 방식으로 취급했으나, 이 연구에서는 불완전 매개변수의 푸리에 급수 계수를 훈련시키는 새로운 접근법을 채택합니다. 또한, 이 방법은 경계 조건이 불완전한 포괄체에 적용하여 다양한 모양의 포괄체에 대한 전방 및 역 문제를 동시에 해결할 수 있습니다.

- **Performance Highlights**: CoCo-PINNs는 신뢰성(reliability), 일관성(consistency), 안정성(stability) 면에서 향상된 성능을 보이며, 기존 PINNs에 비해 더 정확한 전방 해결책 및 향상된 역 매개변수 식별을 제공합니다. 이 방법은 비선형 문제를 해결하는 데 필요한 높은 복잡성을 수용하면서도 보다 정확한 솔루션을 생성할 수 있는 가능성을 보여줍니다. 수치적 방법과의 비교를 통해 CoCo-PINNs의 뛰어난 성능이 입증되었습니다.



### Linearly Convergent Mixup Learning (https://arxiv.org/abs/2501.07794)
Comments:
          none

- **What's New**: 이 논문에서는 RKHS(reproducing kernel Hilbert space)에서의 학습을 위한 새로운 두 알고리즘을 제안합니다. 이러한 알고리즘은 stochastic dual coordinate ascent (SDCA) 알고리즘에 기반하여, gradient descent 방식과는 달리 하이퍼파라미터가 필요없어 구현과 최적화가 간편합니다. 이 연구는 mixup 데이터 증강(mixup data augmentation) 기법을 RKHS에 효과적으로 적용할 수 있는 방법을 제시하며, 다양한 이점이 강조됩니다.

- **Technical Details**: 제안된 두 알고리즘은 주어진 데이터셋 크기와 선형적으로 스케일링되는 수렴(iteration) 횟수와 계산 비용을 보장합니다. 이는 작은 데이터셋과 해석 가능성을 중시하는 상황에서 매우 유용합니다. 이 논문에서는 로스 함수(loss function)의 다양성을 통해 mixup 데이터 증강이 예측 성능을 일관성 있게 향상시킨다는 실험 결과가 포함되어 있습니다.

- **Performance Highlights**: 수치 실험을 통해 제안된 알고리즘이 전통적인 gradient descent 방법보다 빠르게 최적 솔루션에 수렴한다는 결과를 확인했습니다. 또한, mixup 데이터 증강이 다양한 로스 함수에서 예측 성능을 지속적으로 개선한다는 점을 보여주었습니다. 이러한 결과는 알고리즘 적용에 있어 RKHS의 유용성을 강조합니다.



### Transforming Indoor Localization: Advanced Transformer Architecture for NLOS Dominated Wireless Environments with Distributed Sensors (https://arxiv.org/abs/2501.07774)
Comments:
          The paper has been submitted to IEEE Transactions on Machine Learning in Communications and Networking

- **What's New**: 본 연구에서는 실내 위치추적 분야에 있어 새로운 접근 방식을 제안합니다. 특히 Sensor Snapshot Tokenization (SST)이라는 새로운 토큰화 방법을 도입하여 전통적인 방법들보다 향상된 정확도와 효율성을 제공합니다. 이 방법은 다변량 상관관계를 효과적으로 캡쳐하여 Transformer 모델의 주의를 향상시킵니다. 또한, 경량화된 Swish-Gated Linear Unit 기반 Transformer (L-SwiGLU Transformer) 모델을 통해 계산 복잡성을 줄이며, 자원 제약이 있는 환경에서도 높은 정확도를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 다중 안테나 시스템에서의 복잡한 채널 동적을 다루기 위해 Transformer 아키텍처를 규명합니다. 기존의 정규화 레이어를 변경하고 피드포워드 네트워크의 디자인을 재구성하며, 최종 인코더 블록에서의 토큰을 사용하여 예측 메커니즘을 재정의함으로써 L-SwiGLU Transformer 모델을 개선합니다. 또한, Global Average Pooling 레이어를 도입하여 위치 임베딩을 생략하더라도 위치 정확도에 악영향을 미치지 않는다는 사실을 입증하였습니다. 이 모든 수정은 계산 효율성을 높이고 자원 제약이 있는 시나리오에서의 성능을 향상시키기 위해 최적화되었습니다.

- **Performance Highlights**: 제안된 SST 방법으로 개선된 Vanilla Transformer 모델은 NLOS 환경에서 0.388m의 90번째 퍼센타일 위치 오차를 달성하였고, L-SwiGLU ViT 모델은 이를 0.355m로 줄이며 8.51%의 개선을 보였습니다. 또한, 이 새로운 모델은 14.1배 더 큰 기존 모델보다 46.13% 향상된 성능을 보여, 계산 효율성을 잘 입증하였습니다. 이러한 결과는 자원 제한이 있는 환경에서도 Transformer 모델의 효과성을 강조합니다.



### Symmetry-Aware Generative Modeling through Learned Canonicalization (https://arxiv.org/abs/2501.07773)
Comments:
          NeurReps 2024 Workshop Version

- **What's New**: 이 논문은 대칭 밀도(symmetrical density)의 생성 모델링에 대한 새로운 접근 방식을 제안하고 있습니다. 기존의 생성 모델링 패러다임은 불변(prior) 사전과 공변(equivariant) 생성 과정을 결합하지만, 저자들은 이러한 방법이 필요하지 않으며 여러 단점이 있음을 지적합니다. 대신, 저자들은 밀도의 학습된 슬라이스를 모델링하여 각 궤도의 하나의 대표 요소만 학습하는 방안을 채택합니다.

- **Technical Details**: 저자들은 그룹-공변(canonic)화 네트워크를 학습하여 훈련 샘플을 표준화된 형태로 매핑하고, 이러한 표준화된 샘플에 대해 비공변(non-equivariant) 생성 모델을 훈련합니다. 이를 통해 생성 모델이 훨씬 유연하게 작동하고, 세밀한 패턴을 캡처할 수 있습니다. 논문에서는 확산 모델(diffusion models)을 활용하여 이 개념을 구현하였으며, 분자 모델링의 초기 실험 결과는 유망한 성과를 보여줍니다.

- **Performance Highlights**: 분자 데이터셋을 기반으로 한 실험에서, 비공변형 denoising 네트워크와 학습된 canonicalizer를 사용했을 때, 공변 네트워크를 사용할 때보다 더 높은 품질의 샘플을 생성하며 추론 시간 또한 절반으로 줄어드는 결과를 보였습니다. 이러한 결과들은 저자들이 제안하는 새로운 모델 접근 방식이 실질적인 이점을 제공함을 잘 보여줍니다.



### BMIP: Bi-directional Modality Interaction Prompt Learning for VLM (https://arxiv.org/abs/2501.07769)
- **What's New**: 이 논문은 Bi-directional Modality Interaction Prompt (BMIP)라는 새로운 프롬프트 학습 방법을 제안합니다. 이 방법은 비전(vision)과 언어(language) 모달리티 간의 상호 작용에서 발생하는 정렬 효과를 극대화합니다. BMIP는 다양한 데이터셋에서 기존 최첨단 방법들을 초월하며, 다른 프롬프트 기반 방법들과의 결합이 용이합니다. 이를 통해 기존 방법에서 발생했던 단일 모달리티의 일관성 문제를 해결했습니다.

- **Technical Details**: BMIP는 주의(attention) 레이어에서 학습된 정보를 통해 이중 모달 정보에 동적으로 가중치를 부여합니다. 기존의 단순 정보 집계 방법들과 비교할 때, BMIP는 훈련 가능성과 모달 간 일관성을 높이는데 초점을 맞춥니다. 이 방법은 다중 모달 상호작용을 위해 최초로 설계된 집계 함수에 기초하고 있으며, 이는 정밀한 정보 Utilization을 가능하게 합니다. 또한, 새로운 평가 패러다임인 open-world generalization을 통해 추가적으로 개선된 성능을 정량적으로 평가합니다.

- **Performance Highlights**: BMIP는 15개의 벤치마크에 대한 실험에서 모든 평가 패러다임에서 SOTA 성능을 달성했습니다. 특히, BMIP는 EuroSAT과 Flowers102와 같은 불균형 데이터셋에서 단일 모달 프롬프트 학습 방법의 저조한 성능을 해결해냅니다. BMIP는 MaPLe와 함께 사용될 수 있어 다른 프롬프트 학습 방법의 성능도 일관되게 향상시킬 수 있는 기반 프레임워크로 작용합니다.



### PINN-FEM: A Hybrid Approach for Enforcing Dirichlet Boundary Conditions in Physics-Informed Neural Networks (https://arxiv.org/abs/2501.07765)
Comments:
          22 pages

- **What's New**: 이 논문에서는 Physics-Informed Neural Networks (PINNs)가 Dirichlet 경계 조건을 효과적으로 적용하는 데 어려움을 겪고 있다고 설명합니다. 이러한 문제를 해결하기 위해, PINN과 유한요소법(Finite Element Method, FEM)을 결합한 PINN-FEM이라는 새로운 하이브리드 접근 방식을 제안합니다. PINN-FEM은 영역 분해(domain decomposition)를 통해 경계 근처에서 정확한 경계 조건 집행을 가능하게 하여 정확성과 수렴성을 보장합니다.

- **Technical Details**: PINN-FEM 접근 방식은 PINN 네트워크의 유연성과 보조 데이터의 필요성을 더하지 않고도 FEM 기반 표현을 활용하여 경계 조건을 집행합니다. 이 방법은 최소 잠재 에너지 원리를 바탕으로 안정성과 효율성을 보장하는 동시에 수학적 모델링에 있어 PINN과 FEM 요소의 통합을 원활하게 합니다. 이를 통해 PINN-FEM은 경계 조건이 복잡한 여러 시나리오를 처리할 수 있습니다.

- **Performance Highlights**: 여섯 가지 실험을 통해 PINN-FEM은 기존의 PINN 모델들과 비교하여 뛰어난 정확성과 강건성을 입증했습니다. 특히 복잡한 경계 조건을 포함하는 경우에도, PINN-FEM은 기존의 다른 방법들보다 더 나은 성능을 발휘합니다. 이로 인해 PINN-FEM은 국제적인 산업 및 과학 문제를 해결하는 데 매우 적합한 접근 방식으로 평가받고 있습니다.



### Deep Learning for Disease Outbreak Prediction: A Robust Early Warning Signal for Transcritical Bifurcations (https://arxiv.org/abs/2501.07764)
Comments:
          14 pages, 1 figure, 5 tables

- **What's New**: 본 연구에서는 Early Warning Signals (EWSs)를 사용하여 질병 발병 예측을 위한 강력한 모델을 개발하였습니다. 딥 러닝 모델을 활용하여 Time Series Classification (TSC) 작업을 통해 질병의 동적 시스템을 분석하고, 노이즈가 포함된 데이터에서도 깊이 있는 예측을 가능하게 하는 방법을 제시합니다. 특히, 독감과 COVID-19와 같은 실제 데이터에 대한 모델의 성과를 검토하며, 기존 모델보다 뛰어난 예측력을 보여줍니다.

- **Technical Details**: 우리는 두 가지 시뮬레이션 데이터셋을 사용하여 모델을 훈련했습니다. 첫 번째 데이터셋은 다이나믹 시스템을 모사하며, 두 번째 데이터셋은 노이즈에 의해 영향을 받는 전염병 동태를 나타냅니다. 모델 아키텍처는 CNN-LSTM을 기반으로 하며, 트랜스크리티컬 분기예측을 통해 다양한 가변 길이의 시계열 데이터를 효율적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 모델은 다양한 시나리오에 대한 발병 경고를 효과적으로 제공하여 기존 모델보다 우수한 성과를 나타냈습니다. 실제 데이터와 시뮬레이션 데이터에서 검증된 결과, 딥 러닝 기술의 발전이 불확실한 환경에서도 신뢰할 수 있는 EWS를 제공하는 데 기여할 수 있음을 보여줍니다. 이러한 접근 방식은 신흥 전염병 위기의 실질적인 해결책으로 자리잡을 가능성이 높습니다.



### Impatient Bandits: Optimizing for the Long-Term Without Delay (https://arxiv.org/abs/2501.07761)
- **What's New**: 이번 논문에서는 사용자 만족도를 장기적으로 향상시키기 위해 콘텐츠 탐색 문제를 다루었습니다. 이 연구는 bandit 문제를 활용하여 지연 보상 (delayed rewards)의 예측 모델을 개발하고, 새로운 알고리즘을 통해 장기 성공과 연결된 콘텐츠를 빠르게 식별하는 방법을 제안합니다. 또한, Spotify에서의 실제 문제를 해소하기 위한 연구로, 기존의 단기 보상에 최적화된 방법보다 더 효과적임을 입증했습니다.

- **Technical Details**: 이 논문은 지연 보상과 점진적 피드백 (progressive feedback)을 통합한 bandit 알고리즘을 제안합니다. 알고리즘은 두 가지 주요 구성 요소, 즉 Thompson sampling과 Bayesian 필터를 포함하고 있습니다. Bayesian 필터를 통해 점진적으로 드러나는 사용자 참여 데이터를 활용하여 보상의 평균을 추정하고, 이를 통해 효과적인 결정-making을 지원할 수 있습니다.

- **Performance Highlights**: A/B 테스트 결과, 점진적 피드백을 포함한 추천 알고리즘이 단기 보상이나 지연 보상에만 의존하는 방법보다 실질적으로 월등한 성과를 보였습니다. 실험에서는 Spotify의 팟캐스트 추천 데이터가 사용되었으며, 결과적으로 수억 명의 사용자에게 맞춤화된 오디오 추천을 제공하는 시스템의 성능을 크게 향상시켰습니다. 이렇게 함으로써, 산업 수준의 추천 시스템에서 점진적 피드백의 중요성을 강조하였습니다.



### Performance Optimization of Ratings-Based Reinforcement Learning (https://arxiv.org/abs/2501.07755)
Comments:
          Accepted to the Collaborative AI and Modeling of Humans Bridge Program at AAAI 2025

- **What's New**: 이 논문은 Rating-based Reinforcement Learning (RbRL)의 성능을 향상시키기 위해 여러 최적화 방법을 탐구합니다. RbRL은 인간 평가를 기반으로 한 보상 추론 방법을 통해 보상 함수가 없는 환경에서도 정책 학습을 가능하게 합니다. 이 연구는 다양한 하이퍼파라미터의 영향을 이해하기 위한 포괄적 실험을 제공하며, 사용자가 RbRL에서 하이퍼파라미터를 선택하는 데 유용한 지침을 제시합니다.

- **Technical Details**: RbRL은 cross entropy loss를 최소화하여 인간 평가와 추정된 평가 간의 차이를 정량화합니다. RbRL은 PbRL보다 더 다양한 하이퍼파라미터를 가지고 있으며, 이로 인해 최적화가 더욱 중요해집니다. 본 논문에서는 보상 경계, 신뢰 지수와 같은 RbRL 고유의 최적화 방법을 포함하여 총 8개의 최적화 기술을 적용하여 RbRL의 성능을 향상시키려 합니다.

- **Performance Highlights**: 기존의 기계 학습 최적화 기술을 적용하여 RbRL의 성능을 100% 향상시킬 수 있는 경우가 있음을 보였습니다. 최적화된 RbRL은 다양한 평점 클래스에서 일관된 성능을 유지하며, 변동성을 줄이는 결과를 보여줍니다. 본 논문에서는 RbRL의 하이퍼파라미터를 선택하는 데 있어 성능 최적화를 위한 포괄적인 연구를 제공합니다.



### Universal Training of Neural Networks to Achieve Bayes Optimal Classification Accuracy (https://arxiv.org/abs/2501.07754)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 연구는 $f$-divergence 개념을 통해 일반적인 분류 작업의 Bayes 에러율에 대한 새로운 상한을 소개합니다. 제안된 경계는 매개변수화된 모델의 출력을 샘플링하여 계산할 수 있습니다. 또한, Bayes 최적 학습 임계값(BOLT) 손실 함수를 도입하며, 이러한 손실 함수를 최소화하면 분류 모델이 Bayes 에러율에 도달하도록 유도할 수 있습니다.

- **Technical Details**: Bayes 에러율은 특정 분류 작업에 대한 조건부 데이터 분포의 $f$-divergence와 관련이 있음을 보여줍니다. 이 연구에서는 샘플링을 통해 데이터 샘플로부터 직접 계산 가능한 새로운 Bayes 에러의 상한을 도출합니다. 새로운 손실 함수인 BOLT는 일반화 성능 향상에 집중하고 있으며, 이는 매개변수화된 모델을 훈련하는 데 사용될 수 있습니다.

- **Performance Highlights**: BOLT를 사용하여 훈련된 모델은 MNIST, Fashion-MNIST, CIFAR-10 및 IMDb 데이터셋 등에서 기존의 교차 엔트로피 손실을 초과하는 성능을 보여줍니다. 숫자 실험 결과, BOLT가 모델의 일반화 성능을 향상시키는 잠재력을 강조합니다. 특히, BOLT는 복잡한 데이터셋에서 모델의 성능을 개선하는 데 효과적임을 입증했습니다.



### Scaling Up ESM2 Architectures for Long Protein Sequences Analysis: Long and Quantized Approaches (https://arxiv.org/abs/2501.07747)
- **What's New**: 이번 논문은 ESM2 아키텍처의 개선版을 소개합니다. 예전의 1,022 아미노산 제한으로부터 벗어나, 새로운 ESM2 아키텍처는 최대 2,048 아미노산을 입력으로 처리할 수 있는 기능을 추가했습니다. 또한 메모리 공간을 줄이고 추론 시간을 단축하는 양자화(quantization) 기법을 적용한 버전도 제안하고 있습니다.

- **Technical Details**: ESM2 아키텍처는 글로벌(global) 자기 주의 메커니즘(self-attention mechanism)을 사용하여 각 토큰이 시퀀스 내의 다른 모든 토큰을 확인하는 방식을 따릅니다. 그러나 이번 연구에서는 롱포머(LongFormer) 기법을 기반으로 로컬(local) 형태의 주의 메커니즘을 도입하여 메모리 및 계산 복잡도를 𝒪(n𝑘)로 줄였습니다. 또 각 모델은 569,793개의 단백질을 사용하여 대규모 프리트레이닝이 진행되었습니다.

- **Performance Highlights**: 임베딩 표현의 평가 결과, 롱 버전 및 양자화 버전 ESM2 아키텍처는 단순 ESM2 아키텍처보다 우수한 단백질 기능 예측 성능을 보였습니다. 이번 연구에서 개발된 다양한 모델은 HuggingFace 플랫폼에서 사용할 수 있으며, 많은 데이터와 환경에서 효율적인 성능을 유지하고 있습니다.



### HyperQuery: Beyond Binary Link Prediction (https://arxiv.org/abs/2501.07731)
- **What's New**: 이번 논문에서는 지식 하이퍼그래프(knowledge hypergraphs)와 일반 하이퍼그래프(simple hypergraphs)에서 링크 예측(link prediction) 문제를 다루고 있습니다. 특히, 새로운 최적화 아키텍처(optimization architecture)를 개발하여 두 작업을 동시에 해결할 수 있음을 보여줍니다. 노드 레벨 클러스터링(node level clustering)을 통한 새로운 특성 추출(feature extraction) 기법도 소개하며, 이러한 방법이 시스템 성능을 어떻게 향상시킬 수 있는지를 설명합니다.

- **Technical Details**: 하이퍼그래프는 엔티티 간의 n-ary 관계를 모델링하는 데 유용한 도구입니다. 이 논문에서는 하이퍼엣지 예측(hyperedge prediction)과 지식 하이퍼그래프 완성(knowledge hypergraph completion)을 위한 프레임워크인 HyperQuery를 제안합니다. 하이퍼엣지 예측은 주어진 k-튜플에서 하이퍼엣지를 형성하는지를 예측하는 문제로 정의되며, 사전 훈련된 클러스터링 기법을 사용하여 노드의 글로벌 구조(global structure)를 이해하는 방식으로 접근합니다.

- **Performance Highlights**: 제안된 자기 지도 학습(self-supervised learning) 기반의 접근 방식은 여러 하이퍼엣지 예측 및 지식 하이퍼그래프 완성 벤치마크에서 최신 기술을 뛰어넘는 성과를 기록하였습니다. 연구 결과, 제안된 알고리즘이 기존의 방법에 비해 뛰어난 성능을 발휘하는 것을 확인하였으며, 다양한 데이터 세트에 대한 평가를 통해 그 효과성을 입증했습니다.



### Autoencoded UMAP-Enhanced Clustering for Unsupervised Learning (https://arxiv.org/abs/2501.07729)
- **What's New**: 본 논문에서는 비지도 학습을 위한 새로운 접근 방식을 제안합니다. 데이터의 비선형 임베딩을 낮은 차원 공간으로 구축한 후 기존의 클러스터링 알고리즘을 적용하는 방식입니다. 이 방식은 데이터의 클러스터 가능성을 증진시키며, 오토인코더(encoder)와 UMAP 알고리즘의 출력을 포함한 두 가지 매핑으로 구성됩니다. 제안된 방법론을 Autoencoded UMAP-Enhanced Clustering (AUEC)이라고 부르며, MNIST 데이터에 적용했을 때 기존의 기법들보다 월등한 클러스터링 정확도를 보여주었습니다.

- **Technical Details**: AUEC 프레임워크는 세 가지 단계로 구성됩니다. 첫 번째 단계는 오토인코더를 통해 데이터를 낮은 차원 공간으로 임베딩하는 것이며, 여기서 클러스터링 손실(clustering loss)과 복원 손실(reconstruction loss)을 결합한 새로운 손실 함수가 사용됩니다. 두 번째 단계에서는 UMAP을 사용하여 더욱 클러스터링 가능성을 높이기 위한 추가적인 차원 축소가 수행됩니다. 마지막 단계는 기존 클러스터링 알고리즘을 적용하는 것으로, 전체 프레임워크가 비지도 학습을 위한 강력한 도구로 기능하도록 합니다.

- **Performance Highlights**: 실험 결과, AUEC는 기존의 최첨단 기법들보다 클러스터링 정확도에서 더 나은 성능을 보였습니다. 특히 MNIST 데이터셋에서 그 결과가 두드러졌으며, 제안된 모델이 실제 데이터에 대한 처리 능력이 우수함을 입증합니다. 이러한 성과는 비선형 데이터 임베딩을 통한 클러스터링의 새로운 가능성을 열어줍니다.



### Stronger Than You Think: Benchmarking Weak Supervision on Realistic Tasks (https://arxiv.org/abs/2501.07727)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 이번 논문에서는 약한 감독(Weak Supervision, WS)에 대한 새로운 벤치마크인 BOXWRENCH를 소개합니다. 이 벤치마크는 실제 상황을 반영하기 위해 보다 복잡한 라벨 공간, 클래스 불균형, 그리고 다국어 코퍼스를 포함한 과제를 특징으로 합니다. 기존의 벤치마크는 종종 간단하고 비현실적인 작업에 초점을 맞춰 왔으며, BOXWRENCH는 이러한 한계를 극복하고 WS의 실용적인 이점을 정량화하고자 합니다.

- **Technical Details**: BOXWRENCH는 WS 프레임워크의 세 가지 주요 단계인 라벨링 함수(Labeling Functions, LFs) formalization, 라벨 모델(Label Model, LM) 설계 및 다운스트림 모델 학습(training) 과정에서 발생하는 문제들을 해결하는 것을 목표로 합니다. 이 연구에서는 고급 카디널리티를 가진 라벨링, 불균형 클래스 및 특정 도메인 지식을 요구하는 작업에 초점을 맞추며, LFs는 실제 환경을 모방하기 위해 신중히 설계됩니다. 또한, 다국어 데이터셋에서 LFs 재사용 가능성을 실험하여 WS의 실제 효용성을 보여줍니다.

- **Performance Highlights**: BOXWRENCH 벤치마크는 WS의 의미를 더욱 향상시키며, 심지어 간단한 WS 접근 방식이 상당한 가치를 제공할 수 있음을 입증하고 있습니다. 연구 결과에 따르면, 기존의 WS 벤치마크의 경우 클래스 수가 너무 적거나 불균형적으로 설정되어 있어, 더 실제적인 환경에서 WS가 가지고 있는 강점을 지나치게 과소평가할 수 있습니다. BOXWRENCH는 다양한 텍스트 분류 작업을 통해 WS의 강력한 성능을 보여주며, 연구자들에게 새로운 벤치마크 설계에 대한 지침을 제공합니다.



### An Adaptive Collocation Point Strategy For Physics Informed Neural Networks via the QR Discrete Empirical Interpolation Method (https://arxiv.org/abs/2501.07700)
- **What's New**: 이번 연구에서는 물리 정보 기반 신경망(Physics-informed neural networks, PINNs)의 성능을 향상시키기 위한 새로운 적응형 위치 선택 전략을 제안합니다. 기존의 고정된 샘플링 방법에서는 중요한 솔루션 구간을 제대로 포착하지 못하는 한계를 가지고 있음을 지적하였으며, 이를 극복하기 위한 QR 이산 경험적 보간법(Discrete Empirical Interpolation Method, QR-DEIM)을 제안했습니다. 우리의 방법은 딥러닝 모델 학습 과정에서 적응형 샘플링의 새로운 방향을 제시합니다.

- **Technical Details**: 우리의 연구 방법론에서는 PINNs의 훈련 중에 고정된 샘플링 포인트 대신 QR-DEIM을 통해 적응형으로 샘플링 포인트를 선택합니다. 이 방법은 비선형 함수 근사를 효율적으로 수행할 수 있는 모형 잘라내기 기술에서 유래되었습니다. 기존의 방법들은 훈련의 잔여 동적성을 고려하지 않고 단순히 고정된 간격으로 포인트를 샘플링하는 반면, 우리는 이 동적성을 통해 보다 유용한 정보를 포함시키고 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 QR-DEIM 기반 접근 방식이 기존 샘플링 기법들보다 더 낮은 오차를 기록함과 동시에 테스트 손실 값의 수렴 속도가 빨라진다는 것을 보여주었습니다. 우리는 파동 방정식, 앨런-카흐 방정식, 그리고 버거스 방정식과 같은 기준 PDE에 대해 우리의 방법을 시험하였습니다. 전반적으로, 적응형 샘플링 전략이 PINN의 정밀도를 높이는 데 기여함을 확인할 수 있었습니다.



### Dataset Distillation as Pushforward Optimal Quantization (https://arxiv.org/abs/2501.07681)
- **What's New**: 이 논문은 기존의 데이터셋 증류(dataset distillation) 방법론을 새로운 관점에서 제시합니다. 특히, 해체(disentangled) 접근 방식을 통해 기존의 이론적 해석을 제공하고, 고차원 데이터셋에 대한 새로운 최적화 방법을 제안합니다. 또한, 다양한 데이터 분배(distribution)에 대한 일관성을 입증하여, 이미지넷 데이터셋에서의 성능 향상을 도모하는 간단한 방법을 제안합니다.

- **Technical Details**: 이 작업은 데이터셋 증류 문제를 수학적으로 명확히 정의하고, 해체 접근 방식이 클래식한 최적 양자화(optimal quantization) 문제와 어떻게 연관되는지를 설명합니다. 특히, 인코더-디코더 구조를 활용하여 고차원 데이터에 대한 적절한 근사값을 찾는 방법을 제시하고, 이는 파라메트릭한 확률 측도를 최소화하는 것과 관련이 있습니다. 이러한 접근은 기존 방법론보다 연산 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 D4M 방법론의 간단한 확장을 통해 이미지넷-1K 데이터셋에서 우수한 성과를 보여줍니다. 추가적인 계산 비용이 거의 없이 클래스당 이미지를 늘릴 수 있는 가능성을 열어줍니다. 특히, 높은 클래스당 이미지 수 설정에서 최첨단 성능을 달성하여, 기존의 데이터셋 증류 기법들에 비해 월등한 효율성을 입증합니다.



### A Survey of Early Exit Deep Neural Networks in NLP (https://arxiv.org/abs/2501.07670)
- **What's New**: 이 논문은 Early Exit (EE) 방법들에 대한 포괄적인 서베이를 제공하며, 이 방법들이 NLP에서 어떻게 활용되고 있는지를 다룹니다. DNN의 초기 계층에서 간단한 샘플을 분류하는 EE 전략은 전반적으로 추론 과정을 가속화시킬 수 있습니다. 또한, EE 방법은 다양한 샘플의 복잡도에 맞게 적응적 추론을 가능하게 하여, 성능 저하 없이 자원을 효율적으로 사용할 수 있도록 돕습니다.

- **Technical Details**: EE 방법은 DNN의 여러 층에 분류기를 추가하여, 예측에 충분한 신뢰에 도달했을 때 추론 과정을 중단할 수 있게 해줍니다. 이는 DNN이 너무 깊은 층으로 샘플을 보내는 과도한 처리를 방지하고, 보다 복잡한 샘플만이 더 깊이 처리되도록 함으로써 모델의 효율성을 높입니다. EE 방법은 다양한 작업, 특히 NLP와 비전-언어 모델링에 광범위하게 적용될 수 있으며, 각 샘플의 복잡성에 기반해 계산 자원을 동적으로 조정합니다.

- **Performance Highlights**: EE 모델은 정량적 성능 이점 이외에도 해석 가능성, 강건함, 그리고 분산 추론 능력을 증가시킵니다. 다양한 샘플에 대한 신뢰도를 평가하여, 간단한 샘플에 대해서는 초기 계층에서 빠르게 결과를 제공함으로써, 추론 속도를 개선합니다. 이러한 특성 덕분에 EE 방법은 자원이 제한된 환경에서 DNN을 효과적으로 배포할 수 있는 강력한 도구로 자리 잡고 있습니다.



### Finite Sample Identification of Partially Observed Bilinear Dynamical Systems (https://arxiv.org/abs/2501.07652)
- **What's New**: 이 논문은 노이즈가 포함된 입력-출력 데이터로부터 부분적으로 관찰된 이계선형 동적 시스템(Partially Observed Bilinear Dynamical System, BLDS)을 학습하는 문제를 다룹니다. 저자들은 단일 입력-출력 샘플 궤적을 이용해 시스템의 마르코프 유사 파라미터를 학습하고, 이 파라미터로부터 균형 잡힌 시스템을 얻는 알고리즘을 제시합니다. 또한 BLDS의 안정성은 시스템을 자극하는 입력 시퀀스에 따라 달라진다는 점을 강조합니다.

- **Technical Details**: 제안된 알고리즘은 파라미터 학습을 위해 고도로 상관되고 비선형이며 꼬리가 긴 변수를 회귀하여 BLDS의 마르코프 유사 파라미터를 추정합니다. 연구에서는 정의된 안정성 개념을 통해 BLDS의 평균적 안정성을 확립하고, 시스템의 학습 정확도 및 샘플 복잡도의 영향을 미치는 시스템 이론량에 대한 통찰을 제공합니다. 이를 통해 기존의 연속 시간 및 이산 시간의 BLDS 학습 방법들과 차별화된 결과를 도출하였습니다.

- **Performance Highlights**: 저자들은 여러 수치 실험을 통해 이론 결과를 검증하였으며, 무작위로 샘플링된 구에서 입력을 통해 시스템을 자극했을 때 마르코프 유사 파라미터의 추정이 더 향상된다는 것을 보여주었습니다. 이 결과는 입력 선택이 학습 성능에 미치는 영향을 실증적으로 강화하는 중요한 발견입니다. 논문의 나머지 부분에서는 문제 설정, 주요 결과, 실험 결과에 관한 논의가 이어집니다.



### Kolmogorov-Arnold Networks and Evolutionary Game Theory for More Personalized Cancer Treatmen (https://arxiv.org/abs/2501.07611)
- **What's New**: 이 논문은 개인화된 암 치료를 위한 새로운 프레임워크를 제안합니다. Kolmogorov-Arnold Networks (KANs)와 Evolutionary Game Theory (EGT)를 통합하여 예측 모델의 일반화 가능성과 해석 가능성을 향상시키는 것을 목표로 합니다. KAN은 적응성이 뛰어난 해석 가능한 신경 구조를 제공하며, EGT는 암의 진화 동태를 모델링합니다.

- **Technical Details**: KANs는 복잡한 생물학적 시스템을 모델링하는데 적합한 해석 가능한 신경망 구조입니다. 이와 함께 EGT는 치료 전략과 암 세포 간의 상호작용을 동적으로 모델링하여 암의 진행과 반응을 모사합니다. KAN-ODEs를 통해 우리는 이러한 시스템을 정확하게 기술할 수 있는 강력한 도구를 제공합니다.

- **Performance Highlights**: 이 하이브리드 접근법은 예측 정확도, 확장성, 임상 활용성을 증대시킬 가능성을 보여줍니다. 예를 들어, KAN과 EGT를 결합함으로써 치료 결정에 필요한 데이터 기반 인사이트를 제공하여 임상 환경에서의 신뢰성을 높일 수 있습니다. 결과적으로, 이 연구는 복잡하고 적응적인 시스템을 다루기 위한 새로운 표준을 확립할 것으로 기대됩니다.



### An Explainable Pipeline for Machine Learning with Functional Data (https://arxiv.org/abs/2501.07602)
- **What's New**: 이번 연구에서는 기능적 데이터 (functional data)를 활용한 감독 기계 학습 (supervised ML)의 새로운 접근 방식인 VEESA (Variable importance Explainable Elastic Shape Analysis) 파이프라인을 제안합니다. VEESA는 높은 결과를 초래할 수 있는 응용 분야에서 기능적 데이터를 분석하는 것을 목표로 하며, 데이터의 수직적 및 수평적 변동성을 고려하여 예측에 필요한 설명을 제공합니다. 특히, 이는 잠재적으로 위험한 물질의 유형 분류와 잉크젯 프린터의 색상 서명을 활용한 법의학적 연결 작업과 같은 실제 사례에서 적용됩니다.

- **Technical Details**: VEESA 파이프라인은 elastic functional principal components analysis (efPCA)를 사용하여 상관관계가 없는 모델 입력을 생성하고, permutation feature importance (PFI) 방법론을 통해 예측에 중요한 주성분을 식별합니다. 연구는 기능적 데이터의 변동성을 효과적으로 캡처하여 이를 원래 데이터 공간에서 시각화합니다. 이 과정은 기계 학습 모델이 기능적 데이터에서 변동성을 사용해 예측하는 방식을 명확히 설명하는 데 도움을 줍니다.

- **Performance Highlights**: VEESA 파이프라인은 고위험 애플리케이션에서의 기능적 데이터 분석의 신뢰성을 높이는 데 기여합니다. 기능적 데이터를 다루는 기존 기계 학습 방법의 한계를 극복하고, 데이터의 본질적인 특성을 잃지 않으면서도 정확한 예측을 가능하게 합니다. 미래 연구를 위한 자연스러운 확장의 아이디어와 도전 과제도 논의되었습니다.



### Real-Time Decision-Making for Digital Twin in Additive Manufacturing with Model Predictive Control using Time-Series Deep Neural Networks (https://arxiv.org/abs/2501.07601)
- **What's New**: 이 논문은 디지털 트윈(Digital Twin)과 머신 러닝(Machine Learning) 기술을 결합하여 자율 제조 공정에서 실시간 의사결정을 위한 동시 다단계 모델 예측 제어(Model Predictive Control, MPC) 프레임워크를 제안합니다. 특히, ‘시간계열 조밀 인코더(Time-Series Dense Encoder, TiDE)’라는 다변량 딥 뉴럴 네트워크(DNN)를 서브 레이트 모델로 활용하여 제조 시스템의 효율적인 제어를 가능하게 합니다.

- **Technical Details**: 제안된 다단계 MPC 프레임워크는 예측 지평선 전반에 걸쳐 미래 상태를 예측할 수 있는 TiDE 모델을 사용하여 리얼타임 최적 제어 입력을 계산합니다. 이를 통해 레이저 파워와 같은 제어 입력의 연속 시퀀스를 최적화하여 각기 다른 조건에서도 공정 제어를 개선할 수 있습니다. 이 연구는 Directed Energy Deposition(DED) 적층 제조를 사례 연구로 적용하여 열 제어와 관련된 모델링의 유용성을 보여줍니다.

- **Performance Highlights**: 실험결과, 제안한 MPC는 PID 제어기와 비교해 더 부드럽고 변동이 적은 레이저 파워 프로파일을 생성하며, 목표 온도를 보다 정확하게 추적하여 품질을 보장합니다. 또한, TiDE 모델은 정확한 예측을 통해 각종 제약 조건을 만족시키며, 제조 과정에서 발생할 수 있는 결함을 사전에 예방하는 데 기여함을 확인했습니다.



### Impact of Data Breadth and Depth on Performance of Siamese Neural Network Model: Experiments with Three Keystroke Dynamic Datasets (https://arxiv.org/abs/2501.07600)
Comments:
          19 pages, 4 figures

- **What's New**: 이번 연구에서는 Deep Learning 모델, 특히 Siamese Neural Networks (SNN)를 활용하여 행동 데이터에서의 데이터셋의 폭(breadth)과 깊이(depth)가 모델 성능에 미치는 영향에 대해 심도 깊은 실험을 수행하였습니다. 데이터셋의 폭은 참여자의 수로 정의되며, 깊이는 개별 참여자당 샘플의 양으로 정의됩니다. 이에 따라 공공 데이터세트인 Aalto, CMU 및 Clarkson II를 사용하여 다양한 조건에서 실험을 진행하였습니다.

- **Technical Details**: 연구에서는 keystroke dynamics를 활용하여 인증 시스템을 구축하는 과정에서 데이터셋의 폭과 깊이가 모델 성능에 미치는 영향을 분석하였습니다. SNN은 두 개 이상의 동일한 서브 네트워크를 학습하여 유사성을 비교하는 구조로, 새로운 주제에 대해서도 유사성 점수를 계산할 수 있어 키 입력의 행동 생체 인식 분야에 효과적입니다. 실험에서는 훈련 주제의 수, 각 주제 당 샘플 수, 데이터 샘플 내 정보량, 훈련에 사용되는 트리플 수 등을 다양하게 조절하여 모델의 성능을 평가하였습니다.

- **Performance Highlights**: 연구 결과 데이터셋 폭을 증가시키는 것이 주관적 변동성을 효과적으로 포착하는 데 도움이 된다는 것을 알았으며, 특정 데이터셋의 깊이에 대한 영향은 해당 데이터의 특성에 따라 달라짐을 발견하였습니다. 자유 텍스트 데이터셋은 샘플 수, 시퀀스 길이 등 깊이에 따른 세 가지 요소 모두의 영향을 받는 반면, 고정 텍스트 데이터셋은 이러한 요소의 영향을 덜 받는 것으로 나타났습니다. 이러한 발견은 행동 생체 인식을 위한 딥 러닝 모델 훈련 시 데이터셋의 설계에 중요한 통찰을 제공합니다.



### Analyzing Spatio-Temporal Dynamics of Dissolved Oxygen for the River Thames using Superstatistical Methods and Machine Learning (https://arxiv.org/abs/2501.07599)
- **What's New**: 이 연구는 테임즈 강의 수질지표, 특히 용존 산소의 동적 변화를 분석하는데 있어 초통계적 방법(superstatistics)과 머신러닝(machine learning)을 결합한 새로운 접근법을 제안합니다. 연구진은 다양한 수질 지표와 시간적 특징을 포함한 회귀 분석을 통해 용존 산소 예측에 가장 적합한 모델로 Light Gradient Boosting Machine을 식별했습니다. 또한 Transformer 모델을 이용하여 용존 산소 농도를 예측하고, 장기 예측에 있어 Informer 모델이 가장 우수한 성능을 보인다는 사실을 발견했습니다.

- **Technical Details**: 연구에서는 용존 산소 변동을 모델링하기 위해 q-Gaussian 분포를 효과적으로 적용하였고, 멀티플릭атив 경험 모드 분해 방법이 가장 효율적인 추세 제거기법으로 밝혀졌습니다. 데이터는 테임즈 강의 9개 수질 모니터링 사이트에서 수집되었으며, 수질 변동의 초통계적 매개변수는 사이트의 해안까지의 거리와 부정적 상관관계를 보였습니다. 또한 연구진은 SHAP 값을 통해 온도, pH 및 연중 시간을 주요 예측 변수로 식별했습니다.

- **Performance Highlights**: 연구 결과, Informer 모델이 192개의 역사적 시간 단계를 사용하는 장기 예측에서 가장 낮은 MAE 및 SMAPE를 달성하며 최고의 성능을 나타냈습니다. Informer 모델의 ProbSparse self-attention 메커니즘은 시간에 따른 긴 의존성을 효과적으로 포착하며, 용존 산소의 반생애 주기를 인식하는 데 유리합니다. 이러한 통찰력은 정책 입안자들이 생태계 건강 평가에 있어 중요한 예측 도구로 활용될 수 있습니다.



### Automated Heterogeneous Network learning with Non-Recursive Message Passing (https://arxiv.org/abs/2501.07598)
- **What's New**: 본 논문에서 우리는 이질적인 정보 네트워크(heterogeneous information networks, HINs)에 대한 새로운 접근 방식을 제안합니다. AutoGNR 프레임워크는 메시지 전달 과정에서의 잡음을 줄이고, 다양한 유형의 노드와 엣지의 정보를 효율적으로 활용하도록 설계되었습니다. 특히, 비재귀적인(non-recursive) 메시지 전달 방식과 최적의 GNN 구조를 자동으로 식별하는 신경망 아키텍처 검색(neural architecture search) 알고리즘을 도입했습니다.

- **Technical Details**: 이 논문에서는 이질적인 네트워크 학습을 위한 비재귀적 메시지 전달 프레임워크를 제안합니다. 이는 각 홉에서 이웃 노드의 독립적인 집합을 가능하게 하여, 엉뚱한 노드에서 발생할 수 있는 정보 혼합을 방지합니다. 또한, 작업 의존적 제약을 가진 검색 공간을 설계하여, 다양한 노드 유형의 조합을 포함하되 효율적인 검색을 보장합니다.

- **Performance Highlights**: AutoGNR은 대규모 실제 HIN 데이터셋에서 최신 기술보다 지속적으로 우수한 성능을 보였습니다. 실험 결과, 제안된 접근 방식은 불필요한 정보 혼합을 줄이는 데 효과적이며, 이러한 특성 덕분에 더 나은 성능을 달성할 수 있음을 보여주었습니다. 또한, 대규모 데이터에 대한 확장성(scalability)도 입증되었습니다.



### Optimize Incompatible Parameters through Compatibility-aware Knowledge Integration (https://arxiv.org/abs/2501.07596)
Comments:
          Published on AAAI'25: The Annual AAAI Conference on Artificial Intelligence

- **What's New**: 이번 연구에서는 다양한 데이터 분포에서 발생하는 비호환 파라미터를 효과적으로 최적화하기 위한 새로운 방법론인 Compatibility-aware Knowledge Integration (CKI)를 제안합니다. CKI는 두 가지 주요 요소로 구성되어 있으며, Parameter Compatibility Assessment와 Parameter Splicing을 통해 여러 모델의 지식을 통합합니다. 이러한 접근 방식은 기존의 단순한 파라미터 제거나 출력 앙상블 방법의 한계를 극복하고, 더 나은 모델 성능을 유지하면서도 추가적인 파라미터 없이 동작할 수 있습니다.

- **Technical Details**: CKI의 첫 번째 단계인 Parameter Compatibility Assessment는 개별 파라미터의 불확실성을 평가하고, 전체 모델의 정보 콘텐츠를 분석하여 종합적인 파라미터 호환성을 결정합니다. 두 번째 단계인 Parameter Splicing에서는 분석된 호환성을 기반으로 여러 모델의 파라미터를 결합하여 최적화된 모델을 생성합니다. 이 과정에서 하드 스플라이싱과 소프트 스플라이싱을 사용하여 최적의 파라미터를 선택하거나 가중치를 계산하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 다양한 추천 및 자연어 처리 데이터셋에서 실험한 결과, Compatibility-aware Knowledge Integration 방법이 기존 모델의 훈련 제한을 극복하면서도 인퍼런스 비용을 증가시키지 않고 효과적으로 비호환 파라미터를 최적화함을 보여주었습니다. 통합 모델은 추가적인 재학습 없이도 직접 사용 가능하며, 단 한 번의 에포크 재학습만으로도 성능 향상을 이끌어낼 수 있음이 입증되었습니다.



### A Multi-Layer CNN-GRUSKIP model based on transformer for spatial TEMPORAL traffic flow prediction (https://arxiv.org/abs/2501.07593)
Comments:
          17 Pages, 18 Figures, 6 Tables

- **What's New**: CNN-GRUSKIP 모델은 기존의 Recurrent Neural Networks (RNN)와 Convolutional Neural Networks (CNN)의 한계를 극복하기 위해 개발된 혁신적인 접근법입니다. 이 모델은 특히 불규칙하고 확장된 패턴이 있는 교통 흐름 예측에 강력한 성능을 발휘합니다. GRU-SKIP 메커니즘을 통합함으로써 긴 시퀀스를 처리하고, 지연 연결 기능을 통해 더욱 정밀한 예측이 가능합니다.

- **Technical Details**: CNN-GRUSKIP 모델은 6개의 레이어로 구성된 비표준 CNN을 사용하며, 심층적인 시공간 상관관계를 추출하도록 설계되었습니다. 이 모델은 (1) 특화된 CNN 피처 추출, (2) Extended temporal patterns를 캡처하는 GRU-SKIP 강화 긴 시간 모듈, (3) 다중 주의 메커니즘을 활용한 Transformer 모듈과 (4) 맞춤형 예측 모듈로 구성됩니다.

- **Performance Highlights**: CNN-GRUSKIP는 캘리포니아의 Caltrans Performance Measurement System (PeMS) 데이터셋에 대한 테스트에서 ARIMA, Graph Wave Net, HA, LSTM, STGCN, APTN과 같은 기존 모델들을 지속적으로 능가했습니다. 이 모델의 강력한 예측 능력과 적응형 아키텍처는 교통 시스템의 지능형 응용 프로그램을 혁신할 잠재력을 지니고 있습니다.



### Multi-task Domain Adaptation for Computation Offloading in Edge-intelligence Networks (https://arxiv.org/abs/2501.07585)
- **What's New**: 본 논문은 멀티 액세스 엣지 컴퓨팅(Multi-Access Edge Computing, MEC)에서 효율적인 계산 오프로드(Computation Offloading)를 위한 새로운 접근 방식을 제안합니다. Multi-Task Domain Adaptation(MTDA)라는 새로운 방법은 도메인 변화가 있는 환경에서 모델의 일반화 능력을 향상시키는 것을 목표로 합니다. MTDA 모델은 교사-학생 아키텍처를 활용하여 추론 시 소스 도메인 데이터에 접근할 필요 없이 지속적인 적응이 가능하도록 합니다.

- **Technical Details**: MTDA 접근 방식은 멀티 태스크 러닝 프레임워크를 활용하여 이진 오프로드 결정과 자원 할당 문제를 동시에 관리합니다. 이 방법은 내장된 교사 모델을 통해 고품질의 의사 라벨을 생성하여 모델 성능을 개선합니다. 제안된 구조는 다중 사용자 환경에서 성능을 유지하며, 효율적인 자원 관리를 통해 지연을 최소화합니다.

- **Performance Highlights**: MTDA는 성능 점검에서 기존 벤치마크 방법에 비해 평균 제곱 오차(mean squared error)와 정확도(accuracy) 측면에서 월등한 성능을 보였습니다. 특히 사용자가 증가하는 환경에서 뛰어난 성능을 유지하며, 다양한 시나리오에서도 높은 성능을 나타냅니다. 이는 MTDA가 실제 MEC 응용 프로그램에서의 실용성을 지원함을 의미합니다.



### Diffusion Adversarial Post-Training for One-Step Video Generation (https://arxiv.org/abs/2501.08316)
- **What's New**: 본 연구에서는 Adversarial Post-Training (APT)이라는 새로운 방법을 통해 고해상도 비디오 생성을 위한 단일 단계 모델을 제안합니다. 기존의 감쇠 학습(distillation) 방법들이 영상 품질을 저하시키는 문제를 겪는 반면, APT는 실 데이터를 대상으로 적대적 학습을 통해 학습의 안정성과 품질을 향상시킵니다. 이를 통해 2초 길이의 1280x720 해상도 비디오를 실시간으로 생성할 수 있습니다.

- **Technical Details**: APT는 사전 훈련된 감쇠 모델을 초기화로 사용하며, 이 모델을 통해 실 데이터에 대해 적대적 학습 목표를 수행합니다. 이러한 과정에서 R1 정규화(regularization) 손실을 추가하여 대규모 학습을 용이하게 하고, 제너레이터와 판별기를 안정화하기 위한 여러 설계를 도입했습니다. 특히, 제너레이터는 결정론적(distillation) 학습을 통해 초기화되며, 판별기는 변환기(transformer) 기반의 구조로 설계되어 있습니다.

- **Performance Highlights**: 우리의 모델 Seaweed-APT는 2초 길이의 고해상도 비디오와 1024px 이미지를 생성하는 데 있어 기존 방법들과 비교할 수 있는 성능을 발휘합니다. 사용자 연구에 기반한 평가에서는 시각적 충실도(visual fidelity), 구조적 무결성(structural integrity), 텍스트 정렬(text alignment) 측면에서도 높이 평가받았습니다. 특히, H100 GPU를 활용하여 단일 단계로 1280x720 해상도의 비디오를 2초 만에 생성할 수 있는 성과를 이루었습니다.



### Avoiding subtraction and division of stochastic signals using normalizing flows: NFdeconvolv (https://arxiv.org/abs/2501.08288)
- **What's New**: 이번 논문에서는 스토캐스틱(stochastic) 신호의 나누기 또는 빼기 대신, normalizing flows를 사용하여 스토캐스틱 신호의 통계를 회복하는 새로운 접근법을 제시합니다. 이 방법을 통해 복잡한 확률 분포를 더 유연하게 배운다는 점에서 기존 Fourier 기반 접근법보다 진일보한 기능을 갖추고 있습니다. 또한, NFdeconvolve라는 소프트웨어 패키지를 GitHub에서 공개하여 연구자들이 손쉽게 활용할 수 있도록 하였습니다.

- **Technical Details**: 연구에서는 x = a + b와 같은 확률적 모델에서 변수 a의 통계를 알 때 변수 b의 통계를 배우는 과정에 초점을 맞춥니다. Bayesian 통계 및 일반적인 확률 분포를 다루는 기존 방법론들은 종종 문제를 야기하지만, normalizing flows는 이런 배경에서의 잠재적 문제를 해결할 수 있는 강력한 도구로 주목받고 있습니다. 정상화 흐름(normalizing flows)은 단순한 기본 분포를 기반으로 모양에서 유연하게 대체 분포를 구축할 수 있습니다.

- **Performance Highlights**: 실험적으로 NFdeconvolve는 다양한 스토캐스틱 신호의 통계를 추정하는 데 높은 정확성을 보여주었습니다. 사용자들이 데이터를 입력하고 PyTorch에서 제공하는 분포 클래스를 사용하여 간단히 사전 지식 없이도 b의 통계를 출력할 수 있는 점이 큰 장점으로 작용합니다. 이로 인해, 다양한 과학적 응용에서 기존 방법보다 더 효과적이고 직관적으로 신호를 분석할 수 있습니다.



### AI Driven Water Segmentation with deep learning models for Enhanced Flood Monitoring (https://arxiv.org/abs/2501.08266)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 기후 변화로 인해 증가하는 홍수 발생 빈도에 대응하기 위해 UNet, ResNet, DeepLabv3의 세 가지 딥 러닝 모델을 비교하여 픽셀 수준의 수분 분할을 수행합니다. 새로운 데이터 세트를 생성하여 잘 알려진 기준 데이터 세트를 홍수 특정 이미지로 보강함으로써 모델의 견고성을 향상시켰습니다. 이 연구는 드론, 현장 관찰 및 소셜 미디어의 이미지를 활용하여 홍수 감지를 지원하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구에서는 UNet, ResNet, DeepLabv3 아키텍처를 다양한 환경 조건과 지리적 위치에서 테스트하여 각각의 효과성을 평가합니다. 모델은 완전 자동화된 방식으로 이미지를 분석하여 홍수 지역을 식별하며, 전통적인 반자동 방법에 비해 처리 시간을 대폭 단축합니다. 이미지 분할 마스크를 예측함으로써 각 모델의 강점과 한계를 논의하며, 효율적인 홍수 모니터링을 위한 방법론을 제시합니다.

- **Performance Highlights**: 이러한 접근 방식은 생명과 경제적 손실을 줄이기 위한 비상 대응 팀에 중요한 데이터를 제공하는 시기적절하고 지속적인 홍수 모니터링을 가능하게 합니다. 기존 방법에 비해 홍수 지도 생성 시간을 크게 단축시켰으며, 향후 연구를 위한 멀티모달 데이터 통합 및 홍수 감지 전용 견고한 딥 러닝 아키텍처 개발의 가능성을 제시합니다. 전반적으로, 이 연구는 딥 러닝 기술의 혁신적인 사용을 통해 홍수 관리 전략의 발전에 기여하고 있습니다.



### FDPP: Fine-tune Diffusion Policy with Human Preferenc (https://arxiv.org/abs/2501.08259)
- **What's New**: 이번 연구에서는 인간의 선호를 기반으로 한 정책 조정 방법, 즉 FDPP(Fine-tuning Diffusion Policy with Human Preference)를 제안합니다. FDPP는 선호 기반 학습을 통해 보상 함수를 학습하고 이를 사용하여 사전 훈련된 정책을 강화 학습(RL)으로 미세 조정하여 새로운 인간의 선호에 맞게 정책을 조정합니다. 이 방법은 로봇이 원래의 과제를 수행하면서도 새로운 환경에 적응할 수 있도록 돕습니다.

- **Technical Details**: FDPP 알고리즘은 사전 훈련된 확산 정책을 사람의 선호를 반영하여 더 잘 맞추기 위한 방법입니다. 정책은 강화 학습을 통해 학습된 보상 함수를 사용하여 미세 조정됩니다. 또한, KL(Kullback-Leibler) 정규화를 통해 오버피팅을 방지하고 초기 정책의 역량을 유지함으로써, 다양한 로봇 작업 및 선호에 대해 효과적으로 정책 행동을 조정하는 것을 보여줍니다.

- **Performance Highlights**: FDPP는 다양한 로봇 작업에서 실험을 수행한 결과, 정책의 행동 분포를 사람의 선호에 맞춰 성공적으로 조정할 수 있음을 입증했습니다. 이 과정에서 사전 훈련된 정책의 성능을 유지하면서도, 새로운 환경 변화에 적절히 대응할 수 있는 유연성과 견고성을 입증하였습니다. 이번 연구는 로봇의 작업 성능과 사용자 기대를 성공적으로 연계할 새로운 방향성을 제시합니다.



### Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models (https://arxiv.org/abs/2501.08248)
- **What's New**: 이 논문에서는 In-Context Retrieval and Reasoning (ICR^2)라는 새로운 벤치마크를 소개합니다. 기존의 LOFT 벤치마크가 LCLMs의 성능을 과대평가할 수 있는 문제를 해결하고자, LCLMs에 대한 보다 실제적인 평가를 제공하기 위해 고안된 것입니다. ICR^2는 강화된 컨파운딩 문서(confounding documents)를 포함하여 LCLMs의 실제적인 상황에서의 성능을 평가합니다.

- **Technical Details**: ICR^2 벤치마크는 Wikipedia에서 수집된 포괄적인 지식 기반을 기반으로 구축되며, 강력한 리트리버를 사용해 문서를 선택합니다. 이를 통해 설명된 세 가지 방법은 (1) retrieve-then-generate fine-tuning, (2) retrieval-attention probing, (3) joint retrieval head training 입니다. 이 방법들은 LCLMs의 in-context retrieval 능력을 증가시키고, 복잡한 multi-stage pipeline의 한계를 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 논문에서 제시된 최적의 접근 방식은 Mistral-7B 모델에 적용되어 LOFT에서 평균 +17과 +15 포인트 향상된 Exact Match를 보여주었으며, ICR^2에서도 +13과 +2 포인트의 개선을 이뤘습니다. 이 방법은 기존의 Vanilla RAG 및 supervised fine-tuning을 초월하여 성능을 발휘하였고, 더 작은 모델임에도 불구하고 대부분의 작업에서 GPT-4-Turbo보다 우수한 성능을 기록했습니다.



### Continual Deep Active Learning for Medical Imaging: Replay-Base Architecture for Context Adaptation (https://arxiv.org/abs/2501.08245)
- **What's New**: 이번 연구에서는 Continual Active Learning (CAL) 프레임워크인 Replay-Based Architecture for Context Adaptation (RBACA)를 개발하여 의료 이미지 분석의 성능 향상을 목표로 하였습니다. RBACA는 CL(Continual Learning) 리허설 기법과 AL(Active Learning) 요소를 통합하여 다양한 맥락에서 지속적으로 학습할 수 있도록 설계되었습니다. 이로 인해 의료 영상의 적응성과 일반화 능력이 향상되었습니다.

- **Technical Details**: RBACA는 자동으로 이미지 특성 변화 감지를 기반으로 하며, 학습 과정에서 가장 유익한 샘플을 선택하는 AL 컴포넌트를 포함하고 있습니다. 또한, 새로운 맥락에서 학습하기 위해 CL 리허설 방법을 사용하여 모델의 기억 저장소를 효과적으로 관리하고 메모리 프루닝 전략을 지원합니다. CAL 방법을 평가하기 위한 새로운 메트릭인 Incremental Learning Score (IL-Score)를 정의하여 이 과정에서의 성능을 동시에 측정하는 평가 방안을 제시합니다.

- **Performance Highlights**: RBACA는 도메인 및 클래스 증가 학습 시나리오에서도 우수한 성능을 보이며, 심장 이미지의 세분화 및 진단 평가에서 IL-Score가 높은 결과를 도출했습니다. 다양한 메모리 크기 및 주석 예산에 따라, RBACA는 기존 CAL 프레임워크와 비교하여 더 나은 성능을 보여줍니다. 이를 통해 RBACA는 의료 영상 처리의 다양한 맥락에서 적응성과 일반화를 크게 향상시킬 수 있음을 입증하였습니다.



### Engineering LLM Powered Multi-agent Framework for Autonomous CloudOps (https://arxiv.org/abs/2501.08243)
Comments:
          The paper has been accepted as full paper to CAIN 2025 (this https URL), co-located with ICSE 2025 (this https URL). The paper was submitted to CAIN for review on 9 November 2024

- **What's New**: MontyCloud Inc.는 CloudOps(클라우드 운영) 분야에서 자율 봇(autonomous bots)을 활용하여 클라우드 준수(compliance), 보안(security), 그리고 지속적인 운영(continuous operations)을 관리하는 주요 기업입니다. 이들은 GenAI(Generative AI)를 사용하여 클라우드 인프라의 자동화된 관리 및 최적화를 진행하며, 고객들에게 더욱 접근 가능하고 효과적인 플랫폼을 제공하고자 합니다. 특히, 기존 MontyCloud 시스템을 위한 GenAI 기반 솔루션 개발 과정에서 다양한 데이터 소스, 여러 프로세스의 오케스트레이션(orchestration), 복잡한 워크플로우를 처리하는 것이 도전 과제가 되었습니다.

- **Technical Details**: 이러한 도전 과제를 해결하기 위해 MOYA라는 다중 에이전트 프레임워크(multi-agent framework)를 개발하였습니다. MOYA는 GenAI를 활용하여 자율성과 인간 조정(human control)을 균형 있게 조화시키며, 다양한 내부 및 외부 시스템을 통합합니다. 또한, 작업 오케스트레이션(task orchestration), 보안, 오류 완화(error mitigation)와 같은 요소에 최적화 되어 있으며, Retrieval Augmented Generation (RAG) 기술을 통해 정확하고 신뢰할 수 있으며 관련성 있는 인사이트를 생성합니다.

- **Performance Highlights**: MOYA의 다중 에이전트 시스템을 실무자와의 협력을 통해 평가한 결과, 비-에이전틱(non-agentic) 접근 방식에 비해 복잡한 워크플로우에서 정확성, 반응성, 효과성이 개선되었습니다. 이를 통해 MontyCloud의 CloudOps 관리가 더욱 원활해졌으며, 고객들에게 더욱 신뢰할 수 있는 솔루션을 제공하게 되었습니다. 이러한 성과는 클라우드 환경이 점점 복잡해지는 현재, 자율적 관리의 필요성이 증가하고 있음을 시사합니다.



### A Feature-Level Ensemble Model for COVID-19 Identification in CXR Images using Choquet Integral and Differential Evolution Optimization (https://arxiv.org/abs/2501.08241)
- **What's New**: COVID-19 팬데믹은 전 세계적으로 수십억 명에게 심각한 영향을 미쳤습니다. 본 논문은 Chest X-ray (CXR) 이미지를 통해 COVID-19 감염자를 정확히 식별하기 위한 새로운 Deep Learning Diagnosis System을 소개합니다. 이 시스템은 사전 학습된 Deep Convolutional Neural Networks (DCNNs)를 통합한 앙상블 학습(framework) 구조를 이용하여, 기존 RT-PCR의 한계를 극복합니다.

- **Technical Details**: 이 연구에서는 Choquet integral을 이용하여 서로 다른 DCNN의 특징 벡터를 결합하여, 선형 접근으로는 잡을 수 없는 상호작용을 포착합니다. Sugeno-$\lambda$ 측정 이론을 활용하여 네트워크의 하위 집합에 대한 퍼지(fuzzy) 측정을 도출하며, Differential Evolution 알고리즘을 통해 퍼지 밀도를 추정합니다. 또한, 복잡한 특성 벡터 집합을 용이하게 집계(strategies)할 수 있도록 TensorFlow 기반의 Choquet 연산 레이어를 개발했습니다.

- **Performance Highlights**: COVIDx 데이터셋에서의 실험 결과, 본 앙상블 모델은 세 개 클래스로 분류 시 98
%의 정확도, 이진 분류 시 99.50	ext{%}의 정확도를 기록했습니다. DenseNet-201(세 개 분류에서 97	ext{，以서술자}%, 이진에서 98.75	ext{%}), Inception-v3(세 개 분류에서 96.25	ext{%}, 이진에서 98.50	ext{%}), Xception(세 개 분류에서 94.50	ext{%}, 이진에서 98	ext{%}) 보다 월등히 뛰어난 성능을 보여주었으며, 이전의 많은 방법들을 초월했습니다.



### Efficient Deep Learning-based Forward Solvers for Brain Tumor Growth Models (https://arxiv.org/abs/2501.08226)
- **What's New**: 이번 연구에서는 신경망 기반의 forward solver에 기초한 새로운 접근 방식을 도입하여, 부분 미분 방정식(PDE) 모델의 보정 시간을 대폭 단축했습니다. 특히, TumorSurrogate라는 심층 학습 모델을 활용하여, 환자 개별의 종양 분포를 빠르게 예측하고 최적화할 수 있는 가능성을 제시합니다. 여러 아키텍처 실험을 통해, 최적화된 TumorSurrogate 모델이 종양 외형 매칭과 세포 농도 예측에서 가장 우수한 성능을 발휘했음을 보여줍니다.

- **Technical Details**: 종양 성장 모델은 반응-확산 공식에 따라, 두 가지 주요 프로세스인 증식(proliferation)과 확산(diffusion)을 기반으로 합니다. 모델링은 특정 환자의 해부학적 정보를 반영하여 수행되며, TumorSurrogate는 인코더-디코더 아키텍처를 통해 종양 시뮬레이션을 생성합니다. 이 연구에서는 nnU-Net 및 Vision Transformer 모델이 포함된 다양한 네트워크 아키텍처의 성능을 체계적으로 분석하여, 각 모델의 적합성을 평가했습니다.

- **Performance Highlights**: 연구 결과, TumorSurrogate 모델은 기본 모델 대비 평균 제곱 오차(MSE)를 절반으로 줄이며, 모든 종양 세포 농도 임계값에서 가장 높은 Dice 점수를 기록했습니다. 이로 인해, 치료 계획 수립에 있어 환자 맞춤형 접근 방식이 크게 개선될 수 있음을 입증하였습니다. 향후 연구 방향으로는, 본 접근 방식을 통해 보다 향상된 forward solver 성능과 종양 예측의 일반화 가능성을 논의하고 있습니다.



### Data-driven system identification using quadratic embeddings of nonlinear dynamics (https://arxiv.org/abs/2501.08202)
- **What's New**: 본 논문에서는 highly nonlinear dynamical systems의 quadratic representation과 governing equations를 학습할 수 있는 새로운 data-driven 방법인 QENDy (Quadratic Embedding of Nonlinear Dynamics)를 제안합니다. 이 방법은 시스템을 높은 차원의 feature space로 embedding하여 비선형 역학을 quadratic 형태로 변환하는 접근 방식입니다. 기존의 SINDy (Sparse Identification of Nonlinear Dynamics)와 유사하게, QENDy는 trajectory data와 시간 미분을 요구하며, 이를 통해 더 정밀하고 해석 가능한 결과를 도출할 수 있습니다.

- **Technical Details**: QENDy는 기초 함수 집합인 dictionary와 함께 주어진 훈련 데이터 포인트를 기반으로 비선형 일반 미분 방정식의 quadratic embedding을 학습할 수 있습니다. 기존의 Koopman operator나 linearization 기법에 의존하는 대신, 우리는 비선형 동역학 시스템을 최대한 간단한 모델 구조로 재구성함으로써 복잡한 동적 행동을 포착하도록 노력합니다. 이 방법은 주어진 feature space에서 최적의 quadratic embedding을 식별하기 위한 최적화 문제로 이어집니다.

- **Performance Highlights**: QENDy의 유효성과 정확성은 다양한 benchmark 문제를 통해 입증되며, SINDy 및 deep learning 기반 방법과의 성능 비교도 이루어집니다. 논문에서는 무한 데이터 한계에서 QENDy와 SINDy의 수렴을 분석하고, 이들 방법의 유사성과 차이점도 강조합니다. QENDy는 quadratic embedding과 함께 governing equations를 추출할 수 있어, 더욱 해석 가능한 모델링이 가능합니다.



### Globally Convergent Variational Inferenc (https://arxiv.org/abs/2501.08201)
Comments:
          Accepted to the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 연구에서는 변별 추론(variational inference, VI)에서 최초의 전역 수렴(global convergence) 결과를 제시하고 있습니다. 학습 중에 일반적 사용되는 최적화 목표인 증거 하한(evidence lower bound, ELBO) 대신에 예상된 전방 KL 발산(expected forward KL divergence)을 사용하여 접근하고, 이는 신경망(neural network)을 이용해 매개변수화된 변별 분포를 적합시키는 방식입니다. 이를 통해 저자들은 NEURAL TANGENT KERNEL(신경 접선 커널)을 활용해 함수 공간(function space) 내에서의 기울기 변화(gradiend dynamics)를 분석하며, 유일한 해결책이 존재하는 조건을 규명하였습니다.

- **Technical Details**: 연구에서는 매개변수를 통한 변별 분포 Q(Θ;η)로부터 최적화된 목표 함수를 설정합니다. 이 목표 함수는 일반 재생 커널 힐버트 공간(reproducing kernel Hilbert space, RKHS) 내에서 유일한 해결책을 허용하는 조건을 설정합니다. 신경망의 유연성이 충분할 경우, 기울기 하강법(gradient descent)은 이 유일한 함수를 수렴하게 됩니다. 실험에서는 NPE(neural posterior estimation)가 ELBO 기반 최적화 방법을 초월하여 깊이 없는 지역 최적(local optimum)에 자주 수렴하는 문제를 분석하고 있습니다.

- **Performance Highlights**: 저자들은 NPE가 ELBO 기반 최적화보다 성능이 우수하다는 것을 다양한 실제 문제와 블록(studies)에서 입증했습니다. 이러한 연구 결과는 NPE의 비대칭치(non-asymptotic)와 유한 뉴런(finite-neuron) 환경에서의 동작을 설명할 수 있는 중요한 기초를 제공합니다. 결과적으로, 이 연구는 최적화 방법의 효율성을 한층 더 높일 수 있는 가능성을 제시하고 있습니다.



### CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation (https://arxiv.org/abs/2501.08200)
Comments:
          to be published in LLM4Code 2025

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)에 의해 생성된 코드의 안전성을 평가하기 위해 새로운 평가 프레임워크인 CWEval을 도입했습니다. CWEval은 기능적 정 correctness 및 보안을 동시에 평가하는 고품질 작업 명세와 결과 중심의 테스트 오라클을 통해 LLM이 생성한 코드의 보안을 제고합니다. 이를 통해 LLM이 생성한 코드에서 종종 발견되는 기능적이지만 안전하지 않은 코드의 비율을 드러내며, 기존 평가의 부정확성을 입증합니다.

- **Technical Details**: CWEval은 인적 검증을 거친 고급 보안 코딩 작업과 포괄적인 명세, 기능성과 보안 모두에 대한 테스트 오라클을 통해 LLM의 코드 생성 보안 능력을 평가하는 데 초점을 맞추고 있습니다. 각 코딩 작업에는 안전한 솔루션과 취약한 대응이 포함되어 있으며, 명확한 문제 정의와 보안의 중요성을 강조합니다. CWEval은 모든 테스트에서 성공적으로 통과해야 하는 최적의 LLM 응답을 요구하여 보안 인식에서 코드 생성의 전반적인 기능을 동시에 평가합니다.

- **Performance Highlights**: CWEval과 CWEval-bench는 각각 보안-critical 코딩 벤치마크로, LLM이 생성한 코드의 보안 특성을 실험적으로 조사할 수 있는 기회를 제공합니다. 이 프레임워크는 LLM 코드 생성의 보안 위험을 포괄적으로 평가할 수 있도록 지원하며, 31종의 CWE를 포함하는 119개의 고품질 코딩 작업을 제공하여 이전 벤치마크의 한계를 극복합니다. 최종적으로, CWEval은 코드 생성의 안전성을 크게 향상시킬 수 있는 기초가 될 것입니다.



### Self-supervised Deep Hyperspectral Inpainting with the Plug and Play and Deep Image Prior Models (https://arxiv.org/abs/2501.08195)
Comments:
          31 pages, 9 Figures, 7 Tables. arXiv admin note: text overlap with arXiv:2306.08128

- **What's New**: 이번 논문은 hyperspectral 이미지의 품질을 저하시킬 수 있는 다양한 노이즈 및 왜곡 문제를 해결하는 새로운 알고리즘, LRS-PnP-DIP(1-Lip)를 제안합니다. 이 알고리즘은 기존의 DHP(Driving Hyperspectral Processing)에서 보고된 불안정성 문제를 해결하며, 저차원(low-rank)과 희소(sparse) 모델을 성공적으로 결합하여 데이터 구조를 더욱 효과적으로 활용합니다.

- **Technical Details**: LRS-PnP-DIP(1-Lip) 알고리즘은 기존의 서브스페이스 모델의 고정된 집합을 넘어서 데이터를 보다 세밀하게 분석할 수 있도록 설계되었습니다. 이 알고리즘은 약한 가정 하에서도 수렴(convergence)을 보장하는 안정성 분석을 포함하고 있어 실제 응용 프로그램에서 신뢰성과 유용성을 크게 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 솔루션이 시각적으로 및 정량적으로 우수한 인페인팅(inpainting) 결과를 지속적으로 제공하며, 최신 기술(state-of-the-art) 성능을 입증합니다. 이를 통해 hyperspectral 이미지 처리 분야에서의 혁신적인 발전을 기대할 수 있습니다.



### A Critical Synthesis of Uncertainty Quantification and Foundation Models in Monocular Depth Estimation (https://arxiv.org/abs/2501.08188)
- **What's New**: 최근의 기초 모델들이 단안 깊이 추정(monocular depth estimation) 분야에서 중요한 돌파구를 마련했으나, 실제 환경에서 안전하고 신뢰할 수 있는 배포를 위한 명확한 경로는 여전히 부족합니다. 이는 거리 예측을 포함하는 메트릭 깊이 추정(metric depth estimation) 과정에서 특히 두드러지며, 고급 기초 모델조차도 치명적인 오류에 취약해 문제가 되고 있습니다. 본 연구에서는 5가지 불확실성 정량화(un) 방법을 강화된 DepthAnythingV2 기초 모델에 결합하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 기존의 메트릭 깊이 추정 방법론의 한계를 극복하기 위해, 연구팀은 Gaussian Negative Log-Likelihood Loss (GNLL)을 통해 신뢰할 수 있는 불확실성 예측을 제공하는 효율적인 기법을 제안하고 있습니다. 추가로 Monte Carlo Dropout (MCD), Sub-Ensembles (SE), Test-Time Augmentation (TTA) 등 다양한 불확실성 정량화(UQ) 방법을 적용하여 모든 픽셀에 대한 깊이 예측의 불확실성을 계량화하는 데 초점을 맞추고 있습니다. 이를 통해 총 4개의 다양한 데이터셋에서 성능을 평가하여, 실세계 응용에 대한 광범위한 커버리지를 확보하고자 합니다.

- **Performance Highlights**: 연구 결과, GNLL 기반의 미세 조정 방법이 특히 유망한 접근법으로 확인되었습니다. 이 방법은 예측 성능을 유지하면서도 불확실성을 신뢰성 있게 추정하는 데 뛰어난 효율성을 보여 주며, 기존의 기초 모델과 비교하여 훈련 및 추론 시간 면에서도 동등한 성능을 발휘합니다. 또한, 본 연구는 단안 깊이 추정을 넘어 발생할 수 있는 다양한 문제들과의 연결 단서를 제공하며, 데이터 반영에 대한 이해도와 신뢰성을 높이는 데 기여할 것입니다.



### A Multi-Modal AI Copilot for Single-Cell Analysis with Instruction Following (https://arxiv.org/abs/2501.08187)
Comments:
          37 pages; 13 figures; Code: this https URL Models: this https URL, this https URL

- **What's New**: 새로운 기술인 InstructCell은 자연어 처리(NLP)를 기반으로 한 multi-modal AI copilot으로, 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터를 보다 효과적으로 분석할 수 있도록 돕습니다. 이 시스템은 서로 다른 조직과 종에서 수집된 scRNA-seq 프로파일과 텍스트 기반 지침을 결합한 종합적인 데이터셋을 통해 개발되었습니다. 사용자는 InstructCell을 통해 세포 유형 주석, 조건부 pseudo-cell 생성, 약물 감수성 예측과 같은 중요한 작업을 쉽고 직관적인 자연어 명령으로 수행할 수 있습니다.

- **Technical Details**: InstructCell은 Q-Former 모듈, 사전 훈련된 언어 모델(LLM), 그리고 세포 재구성 블록을 포함하는 복합 다중 모달 세포 언어 아키텍처를 통해 텍스트 정보와 단일 세포 유전자 발현 데이터를 동시에 처리할 수 있습니다. 이 시스템은 다중 세포 분석 작업에 필수적인 지침-응답 쌍을 생성하여 다양한 실험적 조건에 적응할 수 있도록 설계되었습니다. 또한, 사용자의 다양한 배경과 언어 스타일에 따라 지침을 처리할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: InstructCell은 기존의 단일 세포 기초 모델들의 성능을 넘는 결과를 보이며, 각 아키텍처 요소의 필요성을 검증하는 실험을 통해 생물학적 통찰력을 발견하는 능력을 갖추고 있습니다. 이러한 기술은 사용자가 복잡한 단일 세포 데이터를 탐색할 수 있도록 지원하며, 기술 장벽을 낮추고 생물학에 대한 깊은 통찰력을 가능하게 합니다. InstructCell은 유연한 자연어 명령을 사용하여 실질적이고 중요한 생물학적 작업을 수행할 수 있는 직관적인 도구를 제공합니다.



### D$^2$-DPM: Dual Denoising for Quantized Diffusion Probabilistic Models (https://arxiv.org/abs/2501.08180)
Comments:
          9 pages, 4 figures, acceptted by AAAI2025

- **What's New**: 본 연구에서는 D2-DPM이라는 이중 노이즈 제거 메커니즘을 소개하여 양자화 노이즈가 노이즈 추정 네트워크에 미치는 부정적 영향을 정밀하게 완화하려고 합니다. 양자화가 노이즈 추정 네트워크에 미치는 영향을 평균 편차(mean deviation)와 분산 편차(variance deviation)라는 두 가지 구성 요소로 나누어 시각화하였습니다. 실험 결과에 따라 D2-DPM은 기존의 풀 프리시전(full-precision) 모델보다 1.42 낮은 FID를 달성하며 생성 품질에서 우수한 성능을 보입니다.

- **Technical Details**: 전통적인 부드러운 샘플링 과정에서 D2-DPM은 각 시간 단계에서 양자화 노이즈를 제거하고, 역 확산(iterative diffusion) 과정을 통해 노이즈가 있는 샘플을 추가적으로 정제합니다. 본 방법은 상관 모델링을 통해 추출된 양자화 노이즈의 통계적 특성을 반영하여, 양자화 출력과 양자화 노이즈를 결합하는 공동 가우시안 모델을 설계합니다. 덕분에 두 가지 변형인 Stochastic Dual Denoising(S-D2) 및 Deterministic Dual Denoising(D-D2)을 제안하여 과정을 진행합니다.

- **Performance Highlights**: D2-DPM은 다양한 생성 작업에서 특히 조건부 및 비조건부 이미지 생성 작업에 대해 주목할 만한 품질 향상을 보여주었습니다. 보고된 실험 결과에 따르면, D2-DPM은 효율적으로 생성 품질을 강화하면서도 3.99배의 압축및 11.67배의 비트 연산 가속화를 달성합니다. 이러한 결과는 D2-DPM이 양자화 기반의 손실을 최소화할 수 있는 유망한 방법임을 시사합니다.



### Revolutionizing Communication with Deep Learning and XAI for Enhanced Arabic Sign Language Recognition (https://arxiv.org/abs/2501.08169)
Comments:
          13 pages, 25 figures, 16 tables

- **What's New**: 이 연구는 MobileNetV3, ResNet50, EfficientNet-B2와 같은 최신 딥 러닝 모델을 사용하여 아랍 수화(Arabic Sign Language, ArSL) 인식에 대한 통합 접근 방식을 소개합니다. 이 모델들은 해석 가능성과 투명성을 높이기 위해 Explainable AI (XAI) 기법으로 추가 강화되었습니다. 연구 결과, EfficientNet-B2는 각각 99.48%와 98.99%의 최고 정확도를 달성했습니다. 이 시스템은 인식 정확도에서 새로운 기준을 설정하고 다문화 소통 기술의 포함성을 강조합니다.

- **Technical Details**: 이 연구의 주요 기여는 다양한 손 제스처를 인식하기 위해 정교한 데이터 증강 방법과 계층적 5-겹 교차 검증을 사용한 것입니다. 또한, Grad-CAM을 통해 모델 결정의 투명성을 강화하였으며, 실시간 인식 정확도를 유지하면서 다양한 환경의 데이터를 처리할 수 있는 능력을 발휘했습니다. 연구는 또한 아랍어 수화 인식에 대한 기존 접근 방식을 개선하기 위해 ResNet50, MobileNetV3, EfficientNet-B2와 같은 최첨단 모델을 도입했습니다.

- **Performance Highlights**: 연구에서 제안된 시스템은 기존의 최고 모델들과 비교하여 우수한 인식 정확도를 보여주었으며, 특히 의료 및 교육 분야에서 사용될 때 투명한 결정 과정을 보장할 수 있습니다. 해석 가능성은 민감한 분야에서 중요한 요소로 작용하며, 이는 사용자의 신뢰를 높이는 데 기여합니다. 또한, 그 시스템은 다른 수화에 대한 적용 가능성을 가지며, 다양한 데이터 시나리오를 효과적으로 처리할 수 있도록 설계되었습니다.



### Multiple-Input Variational Auto-Encoder for Anomaly Detection in Heterogeneous Data (https://arxiv.org/abs/2501.08149)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 Anomaly Detection (AD)을 위한 새로운 신경망 모델인 Multiple-Input Auto-Encoder for AD (MIAEAD)와 Multiple-Input Variational Auto-Encoder (MIVAE)를 제안하였다. 기존 방법들이 비독립적이고 동일하게 분포되지 않은(non-IID) 데이터의 이질성으로 어려움을 겪는 반면, MIAEAD는 각 데이터 샘플의 특성 하위 집합에 대해 이상점(anomaly) 점수를 할당하여 이상점을 효과적으로 발견한다. 이 모델은 서브 인코더(sub-encoder)의 재구성 오류(reconstruction error)를 사용하여 이상점 점수를 계산한다.

- **Technical Details**: MIAEAD는 비지도학습(unsupervised learning)을 통해 모든 서브 인코더를 동시에 훈련시키며 각 특성 하위 집합의 이상점 점수를 결정한다. 반면, MIVAE는 정상(normal) 데이터를 잠재 공간(latent space)에서 모델링해 이를 기반으로 이상점을 식별할 수 있도록 설계되었다. 연구 결과, MIVAE는 정상 샘플과 이상점 간 평균 이상점 점수의 차이가 Variational Auto-Encoder (VAEAD)보다 크며, 이로 인해 더욱 높은 AUC(Area Under Curve) 값을 기록하였다.

- **Performance Highlights**: 여덟 개의 실제 이상 데이터셋에 대한 광범위한 실험을 통해 MIAEAD와 MIVAE가 기존 방법 및 최첨단 비지도 모델보다 최대 6% 높은 AUC 점수를 보였음을 입증하였다. 특히 MIAEAD와 MIVAE는 변동 계수(coefficient of variation, CV) 점수가 낮은 특성 하위 집합에 적용 시 높은 AUC 값을 나타냈다.



### Bootstrapping Corner Cases: High-Resolution Inpainting for Safety Critical Detect and Avoid for Automated Flying (https://arxiv.org/abs/2501.08142)
- **What's New**: 이 논문은 드론의 자동 비행 중 공중 교통을 감지하기 위한 Detect and Avoid 시스템에서 물체 탐지를 위한 새로운 방법을 제안합니다. 특히, 값 비싼 실제 비행 데이터 대신 합성 데이터 세트를 활용하여 이야기를 해결하려고 하며, 이를 통해 드론 안전성을 높이고자 합니다. 연구진은 Pix2Pix와 Stable Diffusion이라는 두 가지 생성 모델을 활용한 이미지 합성을 통해 고해상도 데이터 세트를 생성하는 방법을 제시합니다.

- **Technical Details**: 연구는 GAN(Generative Adversarial Networks) 및 LDM(Latent Diffusion Models)과 같은 생성 모델을 사용하여 고해상도 이미지에서 항공 물체 인식을 위한 주석이 달린 이미지를 합성하는 것을 목표로 합니다. 제안된 방법은 전체 경관을 합성하는 대신 인페인팅(inpainting) 기술을 사용하여 항공 물체를 삽입하는 과정이 포함됩니다. 이 방식은 보다 효율적으로 데이터를 생성하고, 배경 이미지를 통해 정확한 위치에 물체를 삽입할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 고해상도 데이터 세트를 생성하여 공개 가능성을 높이고, 이것이 기존의 실 데이터로 완전히 훈련된 독립적인 물체 탐지기에 적용되는 성능을 검증합니다. 이 연구는 드론 비행 중의 안전성을 향상시키는 데 기여할 것으로 기대되며, 다양한 항공 상황에서 높은 정확도의 물체 탐지를 가능하게 합니다. 생성된 데이터 세트는 공개적으로 다운로드 가능하여, 다른 연구자들이 이용할 수 있게 됩니다.



### EEG-ReMinD: Enhancing Neurodegenerative EEG Decoding through Self-Supervised State Reconstruction-Primed Riemannian Dynamics (https://arxiv.org/abs/2501.08139)
- **What's New**: EEG-ReMinD(EEG 복원-프라이밍 리만 역학) 모델은 뇌-컴퓨터 인터페이스 및 질병 진단을 향상시키기 위해 고안된 신개념의 두 단계 접근법입니다. 이 접근법은 데이터의 라벨 의존성을 줄이고, 자가 감독 학습(self-supervised learning) 및 기하학적(features of geometry) 학습기법을 활용하여 EEG 데이터를 보다 효과적으로 분석할 수 있도록 설계되었습니다. 특히 이 연구는 신경퇴행성 질환에서의 EEG의 맥락을 반영하기 위해 리만 기하학(Riemannian geometry)를 적용한 점이 특징적입니다.

- **Technical Details**: EEG-ReMinD는 무감독(pre-training) 단계와 미세 조정(fine-tuning) 단계를 포함하여 라벨이 부족한 데이터에도 효율적으로 적응합니다. 첫 번째 단계에서는 자가 감독 학습을 통해 리만 기하학적 표현을 구성 및 복재하여 내부 상태를 설정하는 과정을 거칩니다. 두 번째 단계는 미리 훈련된 모델을 활용해 뇌의 상태를 인식하며, 이 과정은 제한된 양의 라벨이 있는 데이터를 사용하여 수행됩니다.

- **Performance Highlights**: 두 가지 신경퇴행성 질환에서의 비교 분석 결과, EEG-ReMinD는 기존의 방법들에 비해 뛰어난 성능을 보였습니다. 이 연구에서 개발된 새로운 두 단계 기법은 동적 EEG 특성을 효과적으로 처리하며, 노이즈와 잡음의 영향을 줄이고 모델의 견고성을 향상시키는 데 기여합니다. 이러한 성능 향상은 자가 감독 학습의 도입과 리만 기하학적 기법의 결합 덕분에 가능해졌습니다.



### An Empirical Wall-Pressure Spectrum Model for Aeroacoustic Predictions Based on Symbolic Regression (https://arxiv.org/abs/2501.08134)
- **What's New**: 이 논문은 현재의 최첨단 예측 모델을 개선하는 새로운 wall-pressure spectrum (WPS) 경험 모델을 도입합니다. 이 모델은 다양한 공기 날개(airfoils) 및 흐름 조건을 적용할 수 있는 범위를 확대하여 저항성과 정확성을 높이도록 설계되었습니다. AI 기반의 symbolic regression 접근 방식을 통해 개발된 이 모델은 추가적인 특수 공식을 제공할 필요 없이 실험 데이터로부터 해석 가능한 수식을 도출할 수 있습니다.

- **Technical Details**: 모델은 두 가지 유형의 공기 날개(NACA 0008 및 NACA 63018)에서 다양한 공격각과 유입 속도에서 측정된 wall-pressure 변동 데이터셋을 사용하여 개발되었습니다. 이 모델은 힘든 압력 구배(pressure gradient)에 대한 민감도를 고려하여 일반적인 포괄성을 갖추고 있습니다. 실험 데이터와의 검증을 통해 모델은 잘 알려진 반경계(stress) 및 혼합 모델에 비해 월폭 압력 변동(WPS)의 예측 정확도에서 우수성을 입증하였습니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 Amiet의 이론과 통합되어 실제 풍력 터빈의 aeroacoustic noise를 예측하는 데 성공하였으며, 실험 측정 결과와 우수한 일치를 보였습니다. 기존의 경험적 모델들에 비해 더 강력하고 일반화가 가능한 예측 능력을 입증하였으며, 새로운 접근 방식을 통해 더욱 정교한 noise 예측이 가능하다는 점에서 매우 중요한 진전을 이루었습니다.



### RoHan: Robust Hand Detection in Operation Room (https://arxiv.org/abs/2501.08115)
Comments:
          12 pages

- **What's New**: 이 논문에서는 수술 환경에서 손 인식을 위한 새로운 접근법인 'RoHan'을 제안합니다. 이는 여러 기록 조건 및 다양한 장갑 색상에서의 손 인식 문제를 해결하기 위해 고급 반자율 도메인 적응 기법을 활용합니다. 특히, 'Artificial Gloves'라는 데이터 증강 기법을 통해 기존 공개 데이터셋을 합성 이미지로 확장하는 방법을 제시하는 것이 특징입니다.

- **Technical Details**: RoHan의 방법론은 두 가지 주요 단계로 구성됩니다: 첫 번째는 'Artificial Gloves' 데이터 증강 전략으로, 이는 실제 손 이미지를 합성 장갑 이미지와 결합하여 데이터셋의 다양성을 증가시킵니다. 두 번째는 반자율 도메인 적응 파이프라인을 통해 실제 수술 환경에서의 탐지 성능을 향상시키는 과정입니다. 이 파이프라인은 예측 세분화와 효율적인 프레임 필터링을 반복적으로 수행하여 정확도를 높입니다.

- **Performance Highlights**: RoHan은 시뮬레이션된 장장 수술과 정맥 이식 수술이라는 두 가지 데이터셋을 통해 평가되었으며, 기존의 대규모 레이블링 및 모델 학습 필요성을 크게 줄입니다. 이러한 방법론은 의료 환경에서 손 인식 기술의 실용적인 구현을 위한 전환점을 마련하며, 더 나아가 메디컬 비전 기술의 발전을 촉진할 것으로 기대됩니다.



### Smooth Handovers via Smoothed Online Learning (https://arxiv.org/abs/2501.08099)
- **What's New**: 이번 논문은 Smoothed Online Learning (SOL) 관점에서 단말기와 기지국 간의 핸드오버(HO) 최적화에 대한 최초의 국가 차원 연구를 제공합니다. 유럽의 대형 이동통신 사업자로부터 수집한 4천만 사용자 이상의 방대한 데이터셋을 분석하여 핸드오버 실패 및 지연의 원인과 그 영향 요소를 밝혀냅니다. 이 연구는 핸드오버 최적화 문제를 다루며, 이동통신 환경의 이질성을 고려한 새로운 시스템 모델을 제안합니다.

- **Technical Details**: 핸드오버 최적화는 UE와 기지국의 동적 결정으로 모델링되며, 이를 통해 인터넷 구축자와의 연계가 필요 없는 새로운 접근 방식을 제시합니다. 우리가 제안하는 알고리즘은 제어기가 미래 신호 측정이나 단말기 이동 패턴에 대한 지식이 필요하지 않도록 하며, 최적화 재정에 있어 모든 가정이 무시될 수 있는 점을 강조합니다. 이러한 알고리즘은 O-RAN 패러다임과 일치하며, 실제 및 합성 데이터에 대해 우수한 성능을 보여줍니다.

- **Performance Highlights**: 우리는 실재 데이터와 이론 데이터를 기반으로 한 다양한 시나리오에서 알고리즘을 평가하였습니다. 제안된 솔루션은 이전 연구들보다 최대 79.6배 낮은 핸드오버 비용을 달성하였으며, 사용자 경험을 감소시키지 않으면서 성능을 극대화합니다. 이러한 결과는 네트워크의 다양한 파라미터에 대한 유연성을 보여주며, 다채로운 핸드오버 지연을 고려한 확장 가능성을 포함하고 있습니다.



### Hybrid Action Based Reinforcement Learning for Multi-Objective Compatible Autonomous Driving (https://arxiv.org/abs/2501.08096)
Comments:
          12 pages, 9 figures, 5 tables

- **What's New**: 본 논문에서는 다양한 자율 주행 시나리오에서 다중 목표 호환성(multi-objective compatibility)을 갖춘 Reinforcement Learning(RL) 메서드를 제안합니다. Hybrid Parametrized Action space(HPA)은 추상적인 안내와 구체적인 제어 명령을 결합한 하이브리드 주행 동작을 생성합니다. 또한, 다중 속성 보상을 고려한 다중 목표 비평가 아키텍처를 통해 다양한 주행 목표에 동시 집중할 수 있도록 합니다.

- **Technical Details**: 저자들은 Hybrid Parametrized Action space를 기반으로 한 Multi-objective Ensemble-Critic RL 방법(HPA-MoEC)을 설계했습니다. 이 메서드는 이산 행동 세트와 연속 파라미터를 포함하여 주행 동작을 생성하며, 여러 보상 함수를 통해 속성을 분리하는 방식으로 작동합니다. 이와 함께, 불확실성 기반 탐색 전략을 도입하여 에이전트가 실용적인 주행 정책을 더 빠르게 발견할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 주행 효율성, 행동 일관성, 안전성 측면에서 다중 목표 호환성 자율 주행을 성공적으로 달성했습니다. 실험은 시뮬레이션된 교통 환경과 HighD 데이터셋에서 수행되었고, 제안된 메서드가 훈련 효율성을 크게 증가시키며 주행 성능을 향상시키는 것을 보여주었습니다.



### Dynamic Multimodal Sentiment Analysis: Leveraging Cross-Modal Attention for Enabled Classification (https://arxiv.org/abs/2501.08085)
- **What's New**: 이 논문은 텍스트, 오디오 및 시각적 데이터를 통합한 멀티모달 감정 분석 모델의 개발을 탐구합니다. 이 모델은 감정 분류를 개선하고 서로 다른 모달리티 간의 복잡한 상호작용을 캡처하여 더 정확하고 미세한 감정 해석을 가능하게 하려는 목표를 가지고 있습니다. 특히, 여러 가지 피처 융합 전략을 평가하여 초기 단계 융합이 향상된 성능을 나타낸다는 결과를 도출하였습니다.

- **Technical Details**: 감정 분석 모델은 트랜스포머 기반 아키텍처 안에서 배치된 세 가지 주요 피처 융합 전략을 탐구합니다: 지연 단계 융합(late stage fusion), 초기 단계 융합(early stage fusion), 다중 헤드 주의(multi-headed attention)입니다. CMU-MOSEI 데이터셋을 사용하여 실험하였으며, 텍스트, 오디오 및 시각 입력이 감정 점수와 함께 동기화되어 제공됩니다. 여러 모달리티의 기능을 동적으로 선택하기 위해 크로스 모달 주의(cross-modal attention) 메커니즘을 활용합니다.

- **Performance Highlights**: 실험 결과, 초기 단계 융합이 지연 단계 융합보다 현저히 우수한 성능을 보여 71.87%의 정확성을 달성했습니다. 또한, 다중 헤드 주의 접근 방식은 72.39%의 정확도를 기록하며 경미한 성과 향상을 나타냈습니다. 향후 작업은 기능 융합 기술을 개선하고, 동적 기능 가중치를 탐색하여 모델 성능을 더욱 향상시키는 방향으로 진행될 것입니다.



### CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning (https://arxiv.org/abs/2501.08071)
Comments:
          cgo 2025

- **What's New**: 이 논문에서는 GPU SASS 스케줄을 최적화하는 자동 시스템인 CuAsmRL을 제안합니다. 이는 RL (Reinforcement Learning) 에이전트를 훈련시켜 인간 전문가처럼 수동 스케줄링하는 방법을 흉내 내서 이루어집니다. 이 최적화 과정은 OpenAI Triton과 통합되어, 기존 CUDA 커널 개발자에게 투명하게 작용합니다.

- **Technical Details**: CuAsmRL은 -O3 최적화된 SASS 스케줄을 시작으로 하고, RL 에이전트가 현재 스케줄을 변형하는 액션을 반복적으로 적용합니다. 변형된 스케줄이 GPU에서 더 높은 처리량을 달성하면 긍정적인 보상이 주어집니다. 이 방식은 기존의 CUDA 커널에 비해 성능을 26% 그리고 평균적으로 9% 향상시킬 수 있습니다.

- **Performance Highlights**: CuAsmRL은 LLM (Large Language Models) 관련 특화된 커널들의 성능을 최대 26% 향상시키며, 평균 9%의 성능 개선을 보여주었습니다. 이 최적화는 하드웨어 레벨에서 진행되므로, CUDA 커널 개발자들에게는 투명하게 적용됩니다. CuAsmRL은 자동으로 학습된 최적화 동작을 통해 SASS 명령어 최적화에 대한 새로운 통찰력을 제공합니다.



### On the use of Statistical Learning Theory for model selection in Structural Health Monitoring (https://arxiv.org/abs/2501.08050)
- **What's New**: 이 논문은 구조 건강 모니터링(Structural Health Monitoring, SHM)에서 모델 선택 문제를 다루고 있으며 통계적 학습 이론(Statistical Learning Theory, SLT)의 관점에서 커널 스무더(kernel smoother)를 선택하는 과정을 탐구합니다. 기존의 휴리스틱 방법 대신, SLT의 경계값을 이용하여 모델의 일반화 가능성을 보다 엄밀하게 추정하는 방법을 제시합니다. 도메인 지식을 회귀 문제에 통합함으로써 더 낮은 보장된 위험도를 갖는 결과를 도출하였으며 이는 일반화 능력을 향상시킵니다.

- **Technical Details**: 모델 선택 문제를 설명하기 위해, 데이터 집합에 함수 모델을 적합시키려는 상황을 고려합니다. 회귀 문제에서 적절한 접근법은 입력의 가중 조합을 찾아 목표 값을 정확히 추정하는 것입니다. 겨롤적으로, 복잡성을 증가시키는 대신 최적의 복잡도를 찾는 것이 필요하며, 이를 통해 모델은 데이터의 본질을 적절히 반영할 수 있습니다.

- **Performance Highlights**: 최적의 모델을 찾기 위해서는 예측된 값과 실제 목표 값 간의 불일치를 최소화하는 것이 핵심입니다. 경험적 위험 최소화(ERM) 원칙을 기반으로 하여, 생성 분포를 잘 알지 못하더라도 샘플을 통해 좋은 추정치를 찾아낼 수 있습니다. 이 논문은 경험적 위험이 실제 위험으로 수렴할 때, 모델이 잘 일반화될 수 있는 조건을 제시합니다.



### Self-Attentive Spatio-Temporal Calibration for Precise Intermediate Layer Matching in ANN-to-SNN Distillation (https://arxiv.org/abs/2501.08049)
- **What's New**: 이번 논문은 Spiking Neural Networks (SNNs)의 성능을 향상시키기 위해 새로운 방법인 Self-Attentive Spatio-Temporal Calibration (SASTC)를 제안합니다. SASTC는 ANN과 SNN 간의 의미적으로 일치하는 레이어 쌍을 자율적으로 식별하여 시공간적 차이를 극복하도록 설계되었습니다. 이 방법은 ANN에서 SNN으로 지식을 효과적으로 전이할 수 있도록 하며, SNN의 기존 성능 한계를 뛰어넘는 결과를 보여줍니다.

- **Technical Details**: SASTC는 자가 주의(self-attention) 메커니즘을 사용하여 ANN과 SNN 간의 레이어 패턴을 시공간적으로 정렬합니다. 연구에서는 각각 b 크기의 미니 배치 데이터에 대해 SNN과 ANN의 출력을 구분하기 위해 각 레이어의 출력을 수학적으로 정의합니다. 이를 통해 맞춤형 지식 증류(knowledge distillation)가 가능하여 SNN의 성능을 상당히 개선할 수 있습니다.

- **Performance Highlights**: SASTC는 CIFAR-10, CIFAR-100 및 ImageNet과 같은 여러 데이터셋에서 기존 방법보다 우수한 성능을 보여주었습니다. 특히 CIFAR-10에서는 95.12%, CIFAR-100에서는 79.40%의 정확도를 달성하였으며, DVS-Gesture와 DVS-CIFAR10와 같은 신경 형태 데이터셋에서도 각각 97.92%와 83.60%의 성능을 기록했습니다. 또한 이번 연구는 SNN이 CIFAR-10과 CIFAR-100에서 ANN을 초월한 최초의 사례로, SNN의 잠재적인 응용 가능성을 제시합니다.



### Gen-A: Generalizing Ambisonics Neural Encoding to Unseen Microphone Arrays (https://arxiv.org/abs/2501.08047)
Comments:
          Accepted for publication in Proceedings of the 2025 IEEE International Conference on Acoustics, Speech and Signal Processing

- **What's New**: 본 연구는 Ambisonics 공간 오디오 포맷으로 마이크 배열 (Microphone Array, MA) 신호의 인코딩을 위한 새로운 심층 신경망 (Deep Neural Network, DNN) 기반 방법을 제안합니다. 이 방법은 훈련 중에 보지 못한 임의의 MA 기하학에 대해 일반화할 수 있는 능력을 가지고 있습니다. 이를 통해 기존 인코딩 방법들이 가진 여러 제한사항을 극복할 수 있습니다.

- **Technical Details**: 제안된 방법은 MA의 기하학 및 MA 신호를 입력으로 받아 다단계 인코더를 사용합니다. 이 인코더는 기하학 및 신호 데이터에 대한 별도의 경로로 구성되며, 각 레벨에서 기하학적 특징이 신호 인코더에 정보를 제공합니다. 이를 통해 최종적으로 Ambisonic 신호를 예측하는 과정을 수행합니다.

- **Performance Highlights**: 실험 결과는 드라이 시나리오의 저주파 및 고주파 영역에서 기존의 정적 시간 불변 방법에 비해 성능이 향상됨을 보여주었습니다. 그러나 잔향이 있는 시나리오에서는 성능이 드롭되는 경향이 있으며, 기본 인코딩 방식이 최적의 인코딩 범위에서 더 나은 결과를 나타내는 경향이 있습니다.



### Combining imaging and shape features for prediction tasks of Alzheimer's disease classification and brain age regression (https://arxiv.org/abs/2501.07994)
- **What's New**: 본 연구에서는 MRI에서 추출된 이미지 및 형태 특성을 결합하여 뇌 나이 예측과 알츠하이머병(AD) 분류의 임상적 관련성 있는 작업을 수행합니다. 제안된 모델은 ResNet으로 추출된 이미지 임베딩과 맞춤형 그래프 신경망(Graph Neural Network)에서 얻은 형태 임베딩을 융합합니다. 이 연구는 기존의 이미지 기반 접근 방식과 형태 기반 접근 방식을 통합하여 더 나은 예측 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구는 CamCAN, IXI 및 OASIS3의 3개의 공개 데이터 세트를 사용하여 이미지 전처리 및 고유한 그래프 신경망 아키텍처를 통해 뇌의 15가지 구조에 대한 형태 임베딩을 추출합니다. ResNet-18 아키텍처를 사용하여 T1 가중치 MRI 이미지에서 특징을 추출하고, 그래프 기반 모델로써 다중 그래프 아키텍처를 설계하였습니다. 마지막으로, 두 가지 모델에서 추출한 특징을 결합하여 예측 성능을 향상시킵니다.

- **Performance Highlights**: 결과적으로, AD 분류에 대한 융합 모델이 기존 모델을 능가하는 것으로 관찰되었으며, 이로 인해 임상 응용의 중요한 분야에서 성능 향상을 보여줍니다. 또한, 뇌 나이 회귀 작업에서도 약간의 개선이 있으나 독립 모델이 더 나은 성능을 보일 때도 있습니다. ROC 곡선을 사용하여 모델의 성과를 분석함으로써, 융합 모델이 디지털 진단 도구로서의 가능성을 더욱 demonstrate합니다.



### CHEQ-ing the Box: Safe Variable Impedance Learning for Robotic Polishing (https://arxiv.org/abs/2501.07985)
- **What's New**: 본 연구는 하이브리드 강화 학습(Hybrid Reinforcement Learning, AHRL) 방법 중 하나인 CHEQ 알고리즘을 실제 하드웨어 문제에 적용한 최초의 사례이다. 특히, 로봇 팔을 사용하여 3D 객체를 폴리싱하는 작업을 통해, 변동 임피던스 제어(Variable Impedance Control, VIC)의 효과성을 실험적으로 입증하였다. 이러한 작업은 공업 자동화에서의 접촉이 많은 조작의 도전 과제를 잘 보여주며, CHEQ를 통해 데이터 효율성과 안전한 탐색 행동을 구현할 수 있음을 보여준다.

- **Technical Details**: CHEQ 알고리즘은 비선형 동작을 제어하기 위해 깊이 있는 강화 학습과 제어 이론을 통합한다. 알고리즘은 임피던스 게인과 최종 효과기 위치 및 방향을 출력하여 계단식 임피던스 제어기에 피드백한다. 시뮬레이션 결과, 변동 임피던스 제어가 폴리싱 성능을 크게 향상시키며, CHEQ가 효율적인 학습을 진행하면서도 안전 한계를 준수한다는 것을 확인하였다.

- **Performance Highlights**: 하드웨어에서 CHEQ는 폴리싱 작업을 수행하는 데 단 8시간의 훈련 시간과 5번의 실패만으로 효과적인 성과를 달성하였다. 이는 실제 사용 환경에서 데이터 효율성과 안전성을 동시에 만족시킬 수 있는 가능성을 암시한다. 이러한 결과는 하드웨어에서 직접 훈련된 AHRL의 유망성을 보여주며, 실제 상황에서도 효과적인 성능을 발휘할 수 있음을 입증한다.



### AI Guide Dog: Egocentric Path Prediction on Smartphon (https://arxiv.org/abs/2501.07957)
- **What's New**: AI Guide Dog (AIGD)는 시각장애인을 위한 경량의 자가 중심 내비게이션 지원 시스템으로, 스마트폰에서 실시간 배치를 위해 설계되었습니다. 이 시스템은 영상 기반의 다중 레이블(classification) 접근 방식을 사용하여 방향 명령을 예측함으로써 안전한 탐색을 보장합니다. 또한 GPS 신호와 고급 방향 지시를 통합하여 목표 기반의 야외 내비게이션을 가능하게 하고, 목적지 없이도 실내 탐색을 지원하는 불확실한 다중 경로 예측 문제를 해결합니다.

- **Technical Details**: AIGD는 스마트폰 카메라에서 수신된 비디오 피드를 사용하여 내비게이션 지침을 예측하는 경량 모델을 채택하였습니다. 시스템은 사용자의 헤딩 방향을 예측하기 위해 각 프레임에 대해 다중 레이블 분류 문제로 모델링합니다. 이를 통해 사용자의 다양한 내비게이션 방향을 예측하고, 구체적인 오디오 명령으로 변환합니다. 실내외 환경 모두에서 목표 지향적 및 탐험적 내비게이션 시나리오를 처리하는 첫 번째 시스템으로 자리잡고 있습니다.

- **Performance Highlights**: AIGD는 장애물 회피를 지원하며, 사용자가 목표가 없는 상황에서도 자유롭게 탐색할 수 있도록 설계되었습니다. 이 시스템은 기존의 시각 장애인 내비게이션 시스템보다 더 견고하고 효율적이며, 최종 목적지에의 도달을 위한 명확한 지침을 제공합니다. 데이터 수집 및 전처리를 통해 모델의 성능을 극대화하며, 사용자들은 명확하고 실행 가능한 오디오 명령을 통해 탐색을 지원받습니다.



### Logarithmic Memory Networks (LMNs): Efficient Long-Range Sequence Modeling for Resource-Constrained Environments (https://arxiv.org/abs/2501.07905)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문은 Logarithmic Memory Networks (LMNs)라는 새로운 아키텍처를 소개합니다. LMNs는 계층적 로그트리 구조를 활용하여 과거 정보를 효율적으로 저장 및 검색합니다. 이 모델은 메모리 사용량을 줄이고 O(n²)에서 O(log(n))으로 주의(attention) 메커니즘의 계산 복잡성을 낮춥니다. 이와 함께, 메모리 관리 시스템 역할을 하는 두 가지 실행 모드를 갖추고 있습니다.

- **Technical Details**: LMNs는 단일 벡터 주의 메커니즘을 사용하여 저장된 정보를 동적으로 액세스합니다. 추가적으로, LMNs는 명시적 위치 인코딩이 필요 없도록 위치 정보를 암묵적으로 인코딩합니다. 모델이 스토리지에서 정보를 가져오는 방식은 계산 효율성과 모델링 능력의 균형을 이뤄 자원이 제한된 환경에서도 실용적인 솔루션을 제공합니다.

- **Performance Highlights**: 이 연구는 자원이 제한된 환경에서 긴 시퀀스를 처리하는 데 있어 LMNs가 실질적인 개선을 제공한다고 강조합니다. 이중 모드 접근 방식은 LMNs가 훈련 과정에서 더 빠른 처리를 가능하게 하며, 추론 과정에서는 메모리 사용량을 대폭 줄입니다. LMNs는 특히 모바일 및 엣지 디바이스 컨텍스트에서 효율성과 확장성을 높일 수 있는 강력하고 확장 가능한 접근법을 제공합니다.



### deepTerra -- AI Land Classification Made Easy (https://arxiv.org/abs/2501.07859)
- **What's New**: deepTerra는 기계 학습과 위성 이미지를 활용해 지표 분류를 쉽게 할 수 있도록 설계된 포괄적인 플랫폼입니다. 데이터 수집, 이미지 증강, 훈련, 테스트 및 예측 모듈을 포함하여 이미지 분류 작업의 전체 워크플로우를 간소화합니다. 이 논문에서는 deepTerra의 기능과 다양한 연구 분야에서의 적용 사례를 보여주며, 향후 발전 방향도 논의합니다.

- **Technical Details**: deepTerra는 기계 학습을 통해 이미지를 분류하는 다양한 단계를 지원하는 도구 모음입니다. 데이터 수집 모듈은 기존 이미지를 가져와 이미지 패치를 추출하고, 자동으로 지리적 좌표를 기록하여 공간적 참조를 제공합니다. 이미지 증강은 회전, 플리핑, 이동 등의 기하학적 변환을 통해 데이터셋을 확장하여 모델의 강건성과 성능을 개선합니다.

- **Performance Highlights**: deepTerra는 VGG, ResNet, Inception 등 다양한 CNN 아키텍처를 지원하여 사용자들이 다양한 이미지 분류 작업을 수행할 수 있도록 합니다. 훈련 과정에서 정확도와 F1 스코어 등의 성과 지표를 시각화하고, 훈련 완료 후에는 혼동 행렬 및 키 성능 메트릭을 포함한 상세한 결과를 제공합니다. 이 툴은 새로운 라벨 없는 데이터에 대한 예측도 지원하여, 최종 목표인 이미지 분류를 효과적으로 수행합니다.



### State-of-the-Art Transformer Models for Image Super-Resolution: Techniques, Challenges, and Applications (https://arxiv.org/abs/2501.07855)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 이미지 초해상도(Image Super-Resolution, SR) 분야에서 transformer 기반 방법의 발전을 소개하고 있습니다. 기존의 CNN 및 GAN과 같은 딥러닝 방식보다 더욱 높은 품질의 이미지를 재구성할 수 있는 가능성을 제시하고 있습니다. 특히, 이전 방법들의 한계인 제한된 수신 필드(receptive fields)와 고주파 세부사항 복구의 어려움을 해결하는 데 기여하고 있습니다.

- **Technical Details**: 이 논문에서는 전통적인 네트워크와 결합된 transformer 기반 SR 모델의 다양한 혁신적인 기법과 아키텍처를 탐구합니다. 이러한 최신 방법들은 글로벌(전역) 및 로컬(지역) 맥락을 균형 있게 고려하여 이미지 품질을 향상시키는 데 중점을 두고 있습니다. 또한, 다양한 시각화 방법을 통해 모델과 기술에 대한 종합적인 이해를 돕고 있습니다.

- **Performance Highlights**: 이 연구는 transformer가 초해상도 기법에 미치는 영향을 탐구하며, 향후 연구를 위한 구조적인 로드맵을 제시하고 있습니다. 비판적으로 분석된 최신 기법들은 유망하지만 아직 탐구되지 않은 격차와 잠재적인 연구 방향을 드러냅니다.



### An Intra- and Cross-frame Topological Consistency Scheme for Semi-supervised Atherosclerotic Coronary Plaque Segmentation (https://arxiv.org/abs/2501.07850)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 논문에서는 Curved Planar Reformation (CPR) 이미지를 활용해 관상동맥 죽상경화증 분석 (CAA)에 필요한 관상동맥 플라크 세분화의 정밀도를 향상시키기 위한 혁신적인 이중 일관성 반지도 학습 프레임워크를 제안합니다. 기존의 깊이 학습 모델들이 경계의 불명확성으로 인해 좋지 않은 성과를 내는 것을 해결하고자, Intra-frame Topological Consistency (ITC)와 Cross-frame Topological Consistency (CTC)를 통합하여 라벨이 붙은 데이터와 붙지 않은 데이터를 효과적으로 활용합니다.

- **Technical Details**: 제안된 프레임워크는 이중 작업 네트워크를 통해 세분화 마스크와 Skeleton-aware Distance Transform (SDT) 예측을 동시에 수행합니다. ITC는 추가적인 주석 없이도 일관된 예측을 통해 형상 구조에 대한 일관성을 유지합니다. 반면 CTC는 인접한 프레임 간의 픽셀 흐름을 분석하는 비지도 추정기를 활용하여 공간 연속성을 확보합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 두 개의 CTA 데이터셋에서 기존의 반지도 학습 방법들을 초월하고 감독 방법의 성능에 근접하는 결과를 나타냈습니다. 또한 ACDC 데이터셋에서도 다른 방법들 대비 뛰어난 성능을 보여주며, 본 연구의 일반화 가능성을 증명했습니다.



### Flow: A Modular Approach to Automated Agentic Workflow Generation (https://arxiv.org/abs/2501.07834)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 활용한 다중 에이전트 시스템의 워크플로우를 동적으로 업데이트하고 모듈화를 촉진하는 방법을 제안합니다. 이러한 접근법은 실시간으로 서브태스크의 할당 및 에이전트 역할을 조정할 수 있어 예상치 못한 장애물에 대처하는 데 필수적입니다. 우리는 워크플로우를 Activity-on-Vertex (AOV) 그래프로 정의하며, 이를 통해 서로 의존하는 서브태스크 간의 관계를 시각화하고 최적화했습니다.

- **Technical Details**: 우리의 시스템은 AOV 그래프를 구성하여 서브태스크의 상태와 로그를 추적합니다. 그래프의 각 노드는 각각의 서브태스크를 나타내고, 방향성 엣지는 서브태스크 간의 의존성을 나타냅니다. 또한, 고도의 모듈화를 강조하여 복잡한 작업을 더 작은 서브태스크 모듈로 나눠 독립적으로 실행함으로써 운영 효율성을 높이고, 동적 업데이트의 용이성을 극대화합니다.

- **Performance Highlights**: 실험을 통해 우리의 다중 에이전트 시스템이 기존 접근 방식에 비해 적응성과 효율성에서 현저한 개선을 보여주었음을 확인했습니다. 특히, 동적 워크플로우 업데이트와 모듈화 덕분에 에이전트 간의 병렬 실행이 가능해져, 전체 성능을 저해하지 않고도 복잡한 작업을 원활하게 수행할 수 있습니다.



### Real-time Verification and Refinement of Language Model Text Generation (https://arxiv.org/abs/2501.07824)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 Streaming-VR(Streaming Verification and Refinement)라는 새로운 방식을 제안합니다. 이 방법은 LLM이 생성하는 토큰을 실시간으로 검증하고 수정하여 전체 생성 이후가 아닌 지속적인 검사를 통해 오류 전파를 방지합니다. 따라서 LLM의 응답 생성 중에 정확성을 높이고 비효율성을 줄이는 것을 목표로 합니다.

- **Technical Details**: Streaming-VR은 외부 검증 모델을 활용하여 LLM이 생성하는 각 토큰을 실시간으로 확인하고, 잘못된 부분은 즉시 수정하는 방식으로 작동합니다. 이 프로세스는 기존의 후속 수정 방식보다 더 빠르고 효율적이며, 결과적으로는 사실적 정확성을 개선합니다. 실험 결과, Streaming-VR을 통해 LLM 품질이 대폭 향상된다는 것을 입증했습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 Streaming-VR의 효과를 검증한 결과, ASQA와 QuoteSum 데이터셋에서 평균 각각 39.8% 및 31.5% 더 높은 효율성을 달성했습니다. 또한, 조기 생성된 잘못된 토큰은 후속 문장이 사실적으로 부정확할 확률을 약 37.6% 증가시켜 Streaming-VR의 필요성을 강조합니다.



### A Multi-Encoder Frozen-Decoder Approach for Fine-Tuning Large Language Models (https://arxiv.org/abs/2501.07818)
- **What's New**: 본 연구에서는 멀티태스킹(multi-task) 환경에서 디코더를 고정(freezing)하는 것이 자연어 처리(NLP) 작업의 성능을 어떻게 개선하는지를 조사합니다. 특히, 다양하고 복잡한 자연어 작업에 대한 성능 향상과 배포(overhead) 비용 절감을 목표로 합니다. 실험 결과, 디코더를 고정하는 방식은 다언어(multilingual) 작업에서의 재앙적 망각(catasrophic forgetting)을 줄여주는 것으로 나타났습니다. 또한, 큰 모델과 결합하면 구조화된 작업 및 질문 응답(QA) 작업에서 성능을 유지하거나 향상시킬 수 있습니다.

- **Technical Details**: 연구에 사용된 모델은 AlexaTM로, 12개의 인코더와 12개의 디코더로 구성된 5억 1100만 개의 파라미터를 가진 모델입니다. 실험은 여러 NLP 데이터셋을 통해 진행되었으며, 학습 스케줄 및 최적화는 Adam Optimizer를 사용하여 진행되었습니다. 특히, 디코더를 고정하였다 하더라도 모델의 크기를 늘림으로써 발생할 수 있는 성능 저하를 완화하는 방법도 모색되었습니다. 이러한 접근 방식은 인코더와 디코더의 별개 학습을 통해 성능을 극대화하는 데 기여합니다.

- **Performance Highlights**: 결과적으로 디코더를 고정할 경우 XSUM과 WebNLG 작업에서 성능이 약 2% 감소했지만, 더 큰 고정 디코더를 사용할 경우 성능이 유지되었습니다. ROUGE, BLEU, NIST와 같은 다양한 평가 지표를 통해 측정된 이러한 성능은 자연어 생성(NLG) 작업에서 디코더의 역할이 중요하다는 것을 보여줍니다. 따라서, 고정된 디코더를 활용하는 접근 방식은 다양한 자연어 처리 작업에서 효과적인 전략으로 자리잡을 수 있을 것으로 보입니다.



### BioPose: Biomechanically-accurate 3D Pose Estimation from Monocular Videos (https://arxiv.org/abs/2501.07800)
- **What's New**: BioPose는 단안 영상을 통해 생체역학적으로 정확한 3D 인간 포즈를 예측하는 새로운 학습 기반 프레임워크입니다. 이 프레임워크는 다중 쿼리 인간 메쉬 복구 모델(MQ-HMR), 신경 역운동학 모델(NeurIK), 2D 정보 기반 포즈 보정 기법의 세 가지 핵심 요소로 구성되어 있습니다. BioPose는 기존의 마커 기반 시스템과 비교했을 때 접근성과 편리함을 유지하면서도 비슷한 성능을 목표로 합니다.

- **Technical Details**: MQ-HMR 모델은 다중 쿼리 변형가능한 트랜스포머를 사용하여 단안 비디오 프레임에서 다중 스케일의 세부 이미지를 추출합니다. 이 추출된 메쉬는 NeurIK 모델에 의해 가상의 마커로 취급되어 생체역학적으로 정확한 3D 포즈를 추론하는 데 사용됩니다. 마지막으로 2D 정보 기반 보정을 통해 3D 예측을 2D 관찰과 일치시켜 시각적 일관성과 생체역학적 유효성을 향상시킵니다.

- **Performance Highlights**: BioPose는 기준 데이터 세트에서 상태-of-the-art 방법들을 상당히 초월하는 성능을 보여주었습니다. MQ-HMR 모델은 단일 이미지에서 인간 메쉬 복구 작업에서 최첨단 결과를 생성하며, BioPose 시스템은 전통적인 다중 카메라 마커 기반 기술과 비교했을 때 경쟁력 있는 성능을 제공합니다. 이러한 성과는 생체역학적 정확성을 보장하는 매력적인 솔루션을 제공합니다.



### On the Statistical Capacity of Deep Generative Models (https://arxiv.org/abs/2501.07763)
- **What's New**: 이 논문은 깊은 생성 모델(deep generative models)의 통계적 특성을 이해하는 데 새로운 통찰을 제공합니다. 방대하고 복잡한 데이터 분포로부터 샘플을 생성할 수 있다는 일반적인 가정을 반박하고, 이러한 모델들이 실제로는 보편적 생성기(universal generators)가 아님을 입증합니다. 또한 모델들이 생성할 수 있는 샘플의 특성이 한정적이며, 특히 무거운 꼬리(heavy tails)가 있는 분포를 잘 처리할 수 없다는 점을 강조합니다.

- **Technical Details**: 논문은 주로 Gaussian 잠재 변수(latent variables)에 대해 논의하며, 이러한 조건 하에서 깊은 생성 모델들이 생성하는 샘플이 경량 꼬리(light tail)를 갖는다는 결과를 제시합니다. 이러한 결과는 측정의 농도(concentration of measure)와 볼록 기하학(convex geometry) 도구를 사용하여 보다 일반적인 로그-오목(log-concave) 및 강한 로그-오목(strongly log-concave) 분포에 대해서도 확장됩니다. 또한 Gromov-Levy 불평등을 적용하여 잠재 변수가 양의 Ricci 곡률(curvature)을 가진 다양체(manifolds) 위에 있을 때 유사한 보장을 설명합니다.

- **Performance Highlights**: 이 연구는 깊은 생성 모델들이 샘플의 불확실성과 다양성을 과소평가하는 경향이 있음을 보여주며, 이는 이상 탐지(anomaly detection) 및 금융(finance)과 같은 응용 분야에서 중요한 의미를 가집니다. 실제 사례를 통해 설계된 모델들이 어떻게 성능에 영향을 미치는지를 드러내며, 특히 무거운 꼬리를 가진 분포를 잘 처리하지 못하는 한계를 강조합니다. 이러한 발견은 특히 베이지안(Bayesian) 문헌에서의 후방 샘플링(posterior sampling)과 연관된 사용에 중요한 의미를 가집니다.



### Concentration of Measure for Distributions Generated via Diffusion Models (https://arxiv.org/abs/2501.07741)
- **What's New**: 이 논문은 확산 모델(diffusion models)에서 샘플링된 데이터 분포가 Concentration of Measure Property를 만족함을 보여줍니다. 이는 특이하게도 Lipschitz 1차원 투영이 평균과 멀지 않은 큰 확률로 발생한다는 것을 의미합니다. 특히, 전통적인 확산 모델들이 'heavy-tailed' 데이터를 잘 캡쳐하지 못하는 이유를 설명합니다.

- **Technical Details**: 연구자들은 확산에 의해 생성된 데이터로 다중 클래스 분류 작업을 위한 일반화 선형 모델을 훈련하였습니다. 여기서 stochastic gradient descent (SGD)를 사용하여 테스트 오류(test error)가 Gaussian universality 결과를 따르는 것을 경험적으로 관찰하였습니다. 이로 인해 훈련된 분류기의 성능 분석 시 데이터가 Gaussian이라고 가정할 수 있는 장점이 생깁니다.

- **Performance Highlights**: 현재의 보편성(universality) 증명 접근법들은 확산 모델 데이터의 공분산 행렬(covariance matrices)이 축소된 최소 고유값을 가지는 경우에 적용되지 않습니다. 이는 이전의 수학적 보편성 결과를 확장하는 것이 흥미로운 개방적 질문으로 남는다 할 수 있습니다. 연구 결과는 실제 데이터 세트가 확산 모델로 샘플링된 데이터로 잘 근사될 수 있음을 나타냅니다.



### Multi-megabase scale genome interpretation with genetic language models (https://arxiv.org/abs/2501.07737)
- **What's New**: Phenformer는 유전적 변이가 질병 위험에 미치는 영향을 이해하기 위한 다중 규모의 유전 언어 모델로 소개됩니다. 이 모델은 8,800만 염기쌍에 달하는 DNA 서열로부터 세포 유형 및 조직 전반에 걸친 유전자 발현의 질병 관련 변화를 자동 생성하는 메커니즘 가설을 학습합니다. Phenformer는 기존의 방법들보다 질병 관련 세포 및 조직 유형에 대한 메커니즘 가설을 문헌과 더 잘 일치시키는 성과를 보여줍니다.

- **Technical Details**: Phenformer는 150,000명 이상의 개인으로부터 얻은 전체 유전체 서열 데이터를 활용하여, 유전자 서열의 차이가 세포 및 조직 유형에 어떻게 질병 관련 발현 변화를 초래하는지를 설명하는 다양한 메커니즘을 학습합니다. 이 모델은 실험적 데이터 없이도 전체 유전체를 다각적으로 해석할 수 있는 능력을 가지고 있습니다. Phenformer의 접근은 고전적인 유전체 분석 방법과는 달리, 질병 위험 예측에서의 적용 가능성을 더욱 확장합니다.

- **Performance Highlights**: Phenformer를 통해 생성된 질병 위험 예측자는 예측 성능이 향상되고 다양한 집단에 대한 일반화 능력이 개선되었습니다. 특히, 이 모델은 메가베이스(Mb) 규모의 전체 유전체 해석을 가능하게 하여, 질병과 관련된 분자 메커니즘에 대한 깊은 이해를 제공합니다. Phenformer는 단순히 서열 데이터만을 사용하여도 질병 위험 예측에서 탁월한 결과를 도출할 수 있는 중요한 도구로 자리잡고 있습니다.



### ESURF: Simple and Effective EDU Segmentation (https://arxiv.org/abs/2501.07723)
- **What's New**: 본 논문은 Elemental Discourse Unit (EDU) 경계를 식별하고 이를 통해 EDU를 분할하는 새로운 방법인 EDU Segmentation Using Random Forests (ESURF)를 제안합니다. 기존의 방법이 고급 기능에 의존하는 반면, ESURF는 lexical 및 character n-gram 특징을 기반으로 하며, 랜덤 포레스트 분류기를 사용하여 간단하면서도 뛰어난 성능을 보여줍니다. 이 연구는 보다 훈련 효율적인 담화 분석 방법으로 나아가는 가능성을 제시합니다.

- **Technical Details**: RST(라이트 구조 이론)는 텍스트 내 전개 구조를 이해하기 위한 주요 기반을 제공합니다. EDUs는 텍스트 내에서 가장 작은 일관성 있는 '사고 단위'를 나타내며, 이들 간의 관계는 비대칭적 형태로 연결됩니다. 본 연구에서 제안한 ESURF는 EDU 식별을 위한 분류 문제로 접근하여, 후보 경계 주변의 9-token 서브시퀀스를 사용, morphologic 및 lexical 특성들을 고려하여 EDU 경계를 효과적으로 분류합니다.

- **Performance Highlights**: 실험 결과, ESURF는 기존의 최신 방법들과 비교해 개선된 EDU 식별률 및 RST 파싱 성능을 보여줍니다. 특히, EDU 세그멘테이션에 있어 단순한 방법이면서도 뛰어난 성과를 보이며, 전반적인 자동화된 RST 파싱 기술 향상에 기여하고 있습니다. 이로 인해 논문의 방법론은 향후 NLP 응용 분야에서 보다 효율적인 성과를 기대할 수 있습니다.



### Testing Human-Hand Segmentation on In-Distribution and Out-of-Distribution Data in Human-Robot Interactions Using a Deep Ensemble Mod (https://arxiv.org/abs/2501.07713)
- **What's New**: 이번 연구에서는 기존의 in-distribution(ID) 데이터와 실제 산업 환경에서 자주 발생하는 out-of-distribution(OOD) 상황에서 미리 훈련된 딥러닝(DL) 모델의 성능을 평가함으로써, 인간-로봇 협력에서 손의 신뢰할 수 있는 탐지 및 분할의 중요성을 강조했습니다. 특히, 산업 도구가 있는 복잡한 배경 및 장갑을 착용한 손 등 다양한 조건을 포함하여 새로운 데이터셋을 설계했습니다. 이를 통해 현실적인 산업 조건에서 손의 제스처와 동작을 더 잘 이해할 수 있도록 하고자 했습니다.

- **Technical Details**: 연구에서는 다중 시점(perspective)의 카메라를 사용하여 RGB 이미지를 캡처하고, egocentric 카메라와 정적 카메라를 조합했습니다. 손 분할을 위해 UNet과 RefineNet을 기반으로 한 딥 앙상블 모델을 사용하였고, 모델의 불확실성을 정량화하기 위해 predictive entropy를 활용한 평가 방식을 도입했습니다. 특히, OOD 상황에서의 성과를 평가하기 위해 손가락 교차 제스처와 빠르게 움직이는 손에서 발생하는 모션 블러와 같은 희귀한 조건을 포함했습니다.

- **Performance Highlights**: 실험 결과는 산업 데이터셋에 훈련된 모델들이 비산업 데이터셋에 훈련된 모델들보다 월등히 우수한 성능을 보였음을 밝힙니다. 모든 모델이 OOD 상황에서는 어려움을 겪었지만, 산업 데이터셋에 훈련된 모델들은 일반화 능력이 크게 향상된 것으로 나타났습니다. 특히, 다수의 손이 존재할 수 있는 경우에 대한 적응력이 강화되었음을 보여주며, 이러한 결과는 산업 특성에 맞춘 훈련의 중요성을 시사합니다.



### A Step Toward Interpretability: Smearing the Likelihood (https://arxiv.org/abs/2501.07643)
Comments:
          16+1 pages, 3 figures

- **What's New**: 본 논문에서는 입자 물리학에 적용된 머신 러닝의 해석가능성(interpretability) 문제에 대한 새로운 정의와 실제 방법을 제시합니다. 이 접근법은 제한된 데이터셋에서 머신이 사용하는 관련 물리적 에너지 스케일을 격리하고 식별하는 데 중점을 둡니다. 이를 통해 머신이 어떻게 물리적 현상을 모델링하고 있는지를 이해할 수 있는 기초를 마련합니다.

- **Technical Details**: 제안된 방법은 입력 이벤트 간의 거리를 고려하여 이러한 이벤트를 스무딩(smirring) 또는 평균화하는 과정을 포함합니다. 논문에서는 최적의 해석 가능성을 위해 에너지 스케일 관련 매개변수를 도입하고, 이를 통해 머신이 학습하는 물리적인 원리에 대한 보다 깊은 통찰을 제공합니다. 여기서 중심이 되는 개념은 유한한 데이터셋을 통해 머신이 추정해야 하는 리스크의 비율을 정의하고 이를 통해 데이터 공간에서 연속적인 형태로 추론하는 것입니다.

- **Performance Highlights**: 구체적인 실험 예로, 쿼크(quark)와 글루온(gluon) jet 식별 문제를 다루며, 스무딩된 가능도를 구성하여 해석합니다. 결과적으로, 해상도가 감소함에 따라 식별의 구별력(discrimination power)이 증가하는 경향을 보이며, 이는 주어진 문제에서 진정한 가능성이 모든 스케일의 방출(emission)에 민감함을 나타냅니다.



### Learning-based Detection of GPS Spoofing Attack for Quadrotors (https://arxiv.org/abs/2501.07597)
Comments:
          Accepted in IEEE Industrial Electronics Society Annual Online Conference

- **What's New**: 본 논문에서는 안전-critical cyber-physical systems (CPS), 특히 quadrotor UAVs에 대한 새로운 공격 탐지 프레임워크인 QUADFormer를 제안합니다. 이 프레임워크는 transformer 기반 아키텍처를 활용하며, 비정상 징후에 민감한 시퀀스를 생성하는 residue generator를 포함하고 있습니다. QUADFormer는 기존 기술보다 뛰어난 탐지 정확도를 자랑하며, UAV가 공격을 받더라도 안전하게 작동할 수 있는 경고 메커니즘을 제공합니다.

- **Technical Details**: QUADFormer의 방법론은 세 가지 주요 구성 요소로 이루어져 있습니다: residue generator, UAV 공격 탐지기 및 resilient state estimation 모듈입니다. Extended Kalman Filter (EKF)를 사용하여 여러 센서로부터 데이터를 통합하며, 이를 통해 UAV의 시스템 상태를 추정하고 의미 있는 잔여 시퀀스를 생성합니다. 또한, 맞춤형 self-attention 메커니즘과 특별히 설계된 손실 함수를 포함하는 반지도 학습 기반 transformer 구조를 소개하여 공격 분류 및 탐지를 수행합니다.

- **Performance Highlights**: 실험 결과, QUADFormer는 전통적인 방법과 기계 학습 기반 기법을 포함하여 수행된 다양한 공격 탐지 접근법을 능가하는 뛰어난 성능을 보여주었습니다. 특히, residue 시퀀스를 입력으로 하여 SVM, CNN, LSTM과 같은 모델을 사용하는 경우에서도 QUADFormer는 여러 메트릭에서 우수한 결과를 달성하였습니다. 자세한 실험 설정 및 결과는 부록에 제공됩니다.



### E2ESlack: An End-to-End Graph-Based Framework for Pre-Routing Slack Prediction (https://arxiv.org/abs/2501.07564)
- **What's New**: E2ESlack는 전면적인 그래프 기반 프레임워크로, 라우팅 전의 slack 예측을 위한 새로운 접근 방식을 제시했습니다. 이 프레임워크는 원시 회로 데이터로부터 TNS/WNS(metric)의 측정값을 추출하고 예측하는 최초의 시스템으로 자리매김하고 있습니다. 또한, Arrival Time (AT) 예측 모델과 빠른 Required Arrival Time (RAT) 추정 모듈이 포함되어 있어 정확한 슬랙 예측을 가능하게 합니다.

- **Technical Details**: E2ESlack는 TimingParser를 통해 DEF, SDF 및 LIB 파일을 처리하여 기능을 추출하고 그래프를 구성하는 모듈을 포함합니다. 이 프레임워크의 구조는 빠르고 분산된 방식으로 작동하며, PyTorch DGL 그래프 형식으로 변환합니다. RAT 추정을 위한 새로운 알고리즘을 통해 TNS 및 WNS를 예측하는 능력을 갖추고 있으며, 이를 통해 고급 회로 설계의 필요를 충족시킬 수 있습니다.

- **Performance Highlights**: 제안된 RAT 추정 방법은 최신 기계 학습 기반 예측 방법 및 라우팅 전 STA 도구보다 우수한 성능을 보였습니다. E2ESlack 프레임워크는 라우팅 후 STA 결과와 비교할 때 TNS/WNS 값이 유사한 수준을 달성하면서 23배까지의 실행 시간 단축을 실현했습니다. 이로 인해, 설계 반복의 시간을 줄이고 시장 출시 시간을 단축할 수 있는 가능성을 제공합니다.



### Dynamic Prototype Rehearsal for Continual Learning in ECG Arrhythmia Detection (https://arxiv.org/abs/2501.07555)
Comments:
          Accepted to 2025 International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이 논문에서는 ECG 부정맥 감지에 대한 새로운 지속적 학습 방법인 DREAM-CL을 제안합니다. DREAM-CL는 동적 프로토타입 리허설 메모리를 도입하여 이전 지식을 잊지 않도록 합니다. 이 방법은 각 훈련 세션에서 학습 행동에 따라 데이터를 클러스터링하여 대표적인 프로토타입을 선택합니다.

- **Technical Details**: DREAM-CL는 훈련 세션 동안의 손실 업데이트를 기반으로 훈련 데이터를 클러스터링합니다. 각 클러스터 내에서 훈련 난이도에 따른 부드러운 정렬 작업을 수행하여 샘플을 정렬하고 극단 값은 압축하여 이상값을 제거합니다. 이후 난이도가 높은 샘플을 리허설 메모리의 프로토타입으로 선택하여 지식 유지에 효과를 보장합니다.

- **Performance Highlights**: 우리는 DREAM-CL을 시간 증가, 클래스 증가 및 리드 증가 시나리오에서 두 개의 널리 사용되는 ECG 부정맥 데이터 세트인 Chapman과 PTB-XL을 사용하여 평가했습니다. 결과는 DREAM-CL이 ECG 부정맥 감지 분야에서 최신 기술보다 우수한 성능을 발휘함을 보여주었습니다. 추가적인 변수 선택과 민감도 연구를 통해 방법의 다양한 설계 선택을 검증하였습니다.



### ML Mule: Mobile-Driven Context-Aware Collaborative Learning (https://arxiv.org/abs/2501.07536)
- **What's New**: 이 논문에서는 ML Mule라는 새로운 머신러닝 접근 방식을 제안하고 있습니다. ML Mule은 사용자 모빌리티를 활용하여 이동 중인 모바일 장치가 고정 장치 간에 모델 스냅샷을 전달하고 업데이트할 수 있게 합니다. 이 접근 방식은 중앙집중형 서버에 의존하지 않으면서도 사용자 개인의 필요에 맞는 모델을 동적으로 진화시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: ML Mule 프레임워크는 두 가지 주요 역할을 규정합니다: 고정 장치와 사용자에 의해 휴대되는 모바일 장치입니다. 이 시스템은 고정 장치가 있는 공간에 사용자가 들어가면 두 장치가 협력하여 모델을 훈련시킵니다. 훈련된 모델은 이동하는 모바일 장치와 고정 장치에 저장되며, 사용자가 새로운 공간으로 이동할 때 모델의 스냅샷이 전송됩니다.

- **Performance Highlights**: 실험 결과, ML Mule은 이미지 분류(CIFAR-100)와 인간 활동 인식(EgoExo4D) 작업에서 기존 방법들보다 더 빠른 수렴성과 높은 정확도를 보였습니다. 특히, ML Mule은 FedAvg, Gossip Learning 및 다른 기존 방법들과 비교했을 때 다양한 데이터 분포 및 이동 패턴에서도 우수한 성능을 나타냈습니다.



### Investigating Map-Based Path Loss Models: A Study of Feature Representations in Convolutional Neural Networks (https://arxiv.org/abs/2501.07534)
Comments:
          4 pages, 2 figures, 4 tables

- **What's New**: 이번 논문에서는 고해상도 지도 기반의 경로 손실(prediction of path loss) 모델을 개선하기 위해 컨볼루션 신경망(convolutional neural networks, CNNs)에서의 스칼라 특성(scalar features) 표현 방식을 심층적으로 연구합니다. 특히, 주파수(frequency) 및 거리(distance)를 CNN의 입력 채널로 사용하는 방법과 회귀(regression) 레이어에 스칼라 입력으로 사용하는 방법을 비교합니다.

- **Technical Details**: 기술적으로, 2D CNN 구조는 여러 개의 채널을 사용하는 이미지 전용 접근법을 따릅니다. 여기서 스칼라 특성은 이미지 채널로 입력되고, 거리 및 주파수 특성이 Path Profile로부터 동시에 추출됩니다. 논문에서 제시하는 두 가지 모델, 즉 'Feature Integration for Non-linear Extraction (FINE)'과 'Feature Layer Integration in Perceptron (FLIP)'의 구조를 비교하여 경로 손실 예측의 효율성을 높이는 방법을 제시합니다.

- **Performance Highlights**: 모델 성능 평가에서 Cross-validation을 적용하여 여러 도시에서 독립적으로 검증합니다. FINE 모델은 일부 도시에서 긍정적인 성과를 보였으나, 다른 도시에서는 상대적으로 높은 RMSE(mean squared error) 값을 기록했습니다. 이러한 결과는 경로 손실 모델링에 있어 스칼라 특성을 포함하는 방식의 중요성을 시사합니다.



### RbRL2.0: Integrated Reward and Policy Learning for Rating-based Reinforcement Learning (https://arxiv.org/abs/2501.07502)
Comments:
          Accepted to the Collaborative AI and Modeling of Humans Bridge Program at AAAI 2025

- **What's New**: 이 논문은 기존의 강화 학습(Reinforcement Learning) 접근법에서 발생하는 문제를 해결하기 위해, 인간의 의사결정 과정을 모방하는 새로운 RL 방법론을 제안합니다. 이 접근법은 다양한 성능 수준에 따라 경험을 차별화하고, 이를 통해 정책(policy) 학습의 효율성을 높입니다. 핵심 아이디어는 다양한 성능 수준의 경험에서 중요한 방향 정보를 추출하고, 이를 통해 정책을 업데이트하는 것입니다.

- **Technical Details**: 제안된 방법은 현재 정책과 실패한 경험 간의 분포 유사성을 벌점화하는 새로운 정책 손실 함수(policy loss function)를 도입하며, 평가 등급(rating classes)에 따라 벌점 용어의 가중치를 다르게 설정합니다. 특히, 경험의 등급이 높을수록 정책과의 차이를 줄이고, 등급이 낮을수록 차이를 늘리는 방식으로 다루어집니다. 이 과정에서 Kullback–Leibler(KL) 발산을 기반으로 한 손실 함수를 활용하며, 여러 복잡한 환경에서 실험하여 새로운 접근법의 출력을 평가합니다.

- **Performance Highlights**: 제안된 방법은 다양한 환경에서 기존의 보상 학습(reward learning)과 비교하여 개선된 수렴성과 전반적인 성능을 보여줍니다. 특히, 실패한 경험에서 얻은 값을 효과적으로 활용함으로써(policy를 최적화) 강화 학습의 잠재력을 더욱 극대화하는 결과를 나타냅니다. 이러한 성과는 향후 RL의 다양한 응용 분야에서 유용하게 활용될 수 있습니다.



### Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards (https://arxiv.org/abs/2501.07493)
- **What's New**: 이번 연구는 Large Language Models (LLMs)의 평가 방식에서의 새로운 취약성을 조명합니다. 인간 투표를 기반으로 하는 평가 시스템인 Chatbot Arena는 신뢰받는 도구로 사용되고 있으나, 이 시스템이 적대적 조작에 취약할 수 있음을 보여줍니다. 연구진은 적대자가 특정 모델을 목표로 삼아 투표를 조작할 수 있는 두 단계 공격 방법론을 제시합니다.

- **Technical Details**: 기존의 자동화된 스코어링 시스템에 비해, 사용자 상호작용을 바탕으로 하는 투표 기반 평가 시스템이 증가하고 있습니다. 하지만 이 시스템이 강화된 보안 조치가 없다면, 모델의 응답 비밀성을 해제하여 투표를 조작할 가능성이 있다는 점을 지적합니다. 연구에서는 Chatbot Arena의 투표 과정을 통해 적대자가 어떤 모델이 응답했는지를 대략 95%의 정확도로 식별할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 적대자가 모델의 순위를 조작하기 위해서는 약 1000표의 투표가 필요하다는 것을 확인했습니다. 연구진은 이러한 공격에 대응하기 위한 여러 방어책을 제안하였으며, Chatbot Arena와 협력하여 이러한 방어책을 공식적으로 구현하였습니다. 향후 이 시스템의 보안을 더욱 강화하기 위해 reCAPTCHA와 로그인 절차 integration이 진행될 예정입니다.



### PrecipDiff: Leveraging image diffusion models to enhance satellite-based precipitation observations (https://arxiv.org/abs/2501.07447)
- **What's New**: 본 연구에서는 막대한 인명을 잃게 했던 물 관련 재해를 줄이기 위한 방법으로, 위성 기반 강수 측정 데이터의 정확도를 높이고 공간 해상도를 향상시키기 위해 확산 모델(diffusion model)을 활용한 새로운 접근 방식을 제안합니다. 특히 기존의 저조한 해상도 문제를 해결하기 위해, 기존 위성 강수 데이터에서 10 km 해상도를 1 km로 다운스케일링(downscaling)하는 방법을 통해 다양한 기상 데이터를 통합적으로 보정하는 기법을 소개합니다.

- **Technical Details**: 이 연구는 다양한 강수량 데이터의 불일치를 교정하기 위해, 잔차 학습(residual learning)을 도입하여 강수량 영역에서의 편향을 보정하는 최초의 컴퓨터 비전 기반 알고리즘을 개발하였습니다. 우리가 제안한 확산 모델은 데이터를 생성하기 위한 확산 과정을 모델링하며, 이 과정에서 노이즈를 점진적으로 제거하여 원래 데이터 분포를 복원하는 방식으로 설계되었습니다. 이를 통해 데이터 수집에 있어 더 높은 정확도와 공간적 세부사항을 달성할 수 있었습니다.

- **Performance Highlights**: 시애틀 지역에서 수행된 실험에서 본 연구의 제안 방법은 기존 강수 예측 시스템에 비해 편향을 현저하게 줄이고 정확도가 크게 향상됨을 입증했습니다. 다른 환경적 변수를 통합하지 않더라도 이 방법은 효과적으로 작동하여, 컴퓨터 비전 기술을 활용한 새로운 강수량 데이터 개선 가능성을 보여주고 있습니다. 본 연구는 위성 강수 측정 제품의 유용성을 높이는 방향으로 나아가며, 향후 기후와 관련된 영향 평가를 위한 더 나은 기초 자료 제공에 기여할 것으로 기대됩니다.



### MVICAD2: Multi-View Independent Component Analysis with Delays and Dilations (https://arxiv.org/abs/2501.07426)
Comments:
          19 pages, 8 figures

- **What's New**: 이 논문에서는 다중 뷰 설정(multi-view settings)에서 이질적인 데이터를 통합하고 피처 공간(feature space)을 정렬하는 데 일반적인 도전과제를 설명합니다. 특히, 자극에 노출된 다양한 피험자로부터 얻은 정보를 분석하여 뇌의 활동 역학을 밝혀내는 신경과학 분야에서 이러한 문제가 두드러집니다. 새로운 접근법인 Multi-View Independent Component Analysis with Delays and Dilations (MVICAD2)가 제안되어, 뇌의 신호 소스를 시간 지연 및 시간 확장을 포함하여 모형화할 수 있습니다.

- **Technical Details**: MVICAD2 모델은 각 피험자의 신호 소스가 시간 지연과 확장에서 차이를 가질 수 있도록 설정되어 있습니다. 이 접근법은 일반적인 Multi-View Independent Component Analysis (MVICA) 및 GroupICA보다 더 유연하고 정확한 결과를 제공합니다. 본 연구에서는 이 모델의 확인 가능성과 가능도( likelihood) 근사를 도출하고, 이를 통해 모델의 성능 향상을 위한 규제화(regularization) 및 최적화(optimization) 기술을 적용합니다.

- **Performance Highlights**: 시뮬레이션 결과 MVICAD2는 기존의 다중 뷰 ICA 알고리즘보다 우수한 성능을 보였으며, Cam-CAN 데이터셋을 사용한 검증에서도 인상적인 결과를 나타냈습니다. 연구 결과는 신경 반응의 지연 및 확장이 노화와 관계가 있음을 보여주며, 이는 신경과학 문헌의 기존 연구 결과와 일치합니다.



### An Investigation into Seasonal Variations in Energy Forecasting for Student Residences (https://arxiv.org/abs/2501.07423)
- **What's New**: 이 연구는 학생 주거 환경의 계절적 변동이 에너지 예측에 미치는 독특한 도전 과제를 강조하며, 다양한 기계 학습 모델을 평가합니다. 연구는 LSTM와 GRU와 같은 기본 모델과 함께 최신 예측 방법인 Autoregressive Feedforward Neural Networks, Transformers, 하이브리드 접근 방식을 함께 비교 분석합니다. 결과적으로, 특정 계절에 최적화된 모델 선택의 필요성이 강조되며, Hyper Network 기반 LSTM과 MiniAutoEncXGBoost 모델이 특히 여름철 소비 변화에 강한 적응성을 보임을 발견했습니다.

- **Technical Details**: 프로젝트의 초점은 두 개의 실제 데이터셋(Residence 1과 Residence 2)을 활용하여 하루 앞서 에너지를 예측하는 것입니다. 초기 단계에서 Multi-Layer Perceptron (MLP), Temporal Convolution Neural Network (TCN), Recurrent Neural Network (RNN), Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM)과 같은 다섯 가지 기본 모델을 평가했습니다. 모델 성능 평가는 Symmetric Mean Absolute Percentage Error (SMAPE), Mean Absolute Error (MAE), Root Mean Square Error (RMSE)와 같은 세 가지 지표를 통해 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 모델의 계절별 적응성에 따라 예측 정확도가 달라짐을 밝혀내었습니다. 특히 Hyper Network와 LSTM 모델은 여름철의 급격한 에너지 소비 변화를 효과적으로 포착하는 데 성공했습니다. 결론적으로 이 연구는 에너지 예측 분야에 계절적 동역학의 중요성과 모델 특유의 행동을 강조하며, 에너지 소비 예측의 정확성을 높이는 데 기여할 것입니다.



### PROTECT: Protein circadian time prediction using unsupervised learning (https://arxiv.org/abs/2501.07405)
- **What's New**: 본 논문에서는 최신의 컴퓨터 기반 접근법인 PROTECT를 제안합니다. 이 방법은 고차원 단백질 체계에서 시계열 편집 사항을 예측할 수 있으며, 시간 레이블 없이도 작동합니다. 기존의 방법이 필요한 시간 레이블을 요구하는 것과 대조적으로, PROTECT는 보다 유연하고 저렴한 타임라인 예측이 가능합니다.

- **Technical Details**: PROTECT 모델은 깊은 신경망(Deep Neural Network, DNN)을 기반으로 하여 시계열에서 각 샘플의 단백질 리듬을 예측합니다. 이는 사전 훈련 없이도 고차원 데이터로부터 직접적으로 데이터를 처리하는데, 단계별로 계층을 학습하여 표시 패턴을 포착합니다. 특히, 각 단계의 가중치를 독립적으로 학습함으로써 소량의 샘플 데이터에서도 과적합(overfitting) 문제를 줄일 수 있습니다.

- **Performance Highlights**: PROTECT 모델은 마우스, 미세조류 세포(Ostreococcus tauri), 인간 데이터셋을 사용해 정확성과 견고성을 검증하였으며, 서로 다른 뇌 영역과 치매 관련 샘플 간 시계열 리듬 차이를 탐색하는 데 성공했습니다. 특히, 알츠하이머(Alder) 환자와 대조군 간의 리듬 단백질의 차이를 밝혀내어, 치매와 관련된 생물학적 통찰을 제시할 가능성을 보여주었습니다.



### Derivation of effective gradient flow equations and dynamical truncation of training data in Deep Learning (https://arxiv.org/abs/2501.07400)
Comments:
          AMS Latex, 35 pages

- **What's New**: 이 논문에서는 Deep Learning(DL)에서 ReLU 활성화 함수에 대한 누적 편향(cumulative bias)과 가중치(weights)의 명시적인 방정식을 도출합니다. 특히, 입력층(input layer)에서 유클리드 비용(Euclidean cost)에 기초하여 경량화된 분류와 훈련된 데이터에 대한 역동적 시스템의 작용을 이해하려고 합니다. 이 연구는 지도 학습(supervised learning)의 해석 가능성 문제(interpretable AI) 접근을 확대하고 있습니다.

- **Technical Details**: 우리는 ReLU 활성화 기능을 선택하면서 균일한 차원의 DL 네트워크를 고려합니다. 가중치 행렬 Wₗ과 편향 벡터 bₗ을 정의하고, 훈련 데이터에서의 역동적 흐름 구조를 설명합니다. 이 과정에서, 누적 가중치와 편향의 그래디언트 흐름이 시간 의존적 자극을 유도하는 것으로 나타나며, 이로 인해 훈련 데이터의 기하학적 복잡성이 점진적으로 줄어듭니다.

- **Performance Highlights**: 이 연구의 결과는 훈련 데이터의 클러스터가 복잡성이 감소하는 '트렁케이션(truncation)' 과정에 있음을 보여줍니다. 적절한 조건 하에서, 이는 신경망의 붕괴(neural collapse) 현상에 해당하며, 손실 제로 훈련(zero loss training)을 이끌어내는 핵심 메커니즘으로 작용합니다. 이러한 발견은 딥러닝의 해석 가능성과 효율성을 한층 강화할 것으로 기대됩니다.



### Information-Theoretic Dual Memory System for Continual Learning (https://arxiv.org/abs/2501.07382)
Comments:
          35 pages, 9 figures, submitted to Knowledge-Based Systems

- **What's New**: 이번 논문에서는 동적 환경에서 지속적으로 새로운 지식을 습득하는 "continual learning" 능력을 집중적으로 다룹니다. 기존 메모리 시스템이 직면한 문제를 해결하기 위해, 두 개의 메모리 버퍼를 사용하는 혁신적인 "Information-Theoretic Dual Memory System (ITDMS)"를 제안합니다.

- **Technical Details**: ITDMS는 빠른 메모리 버퍼와 느린 메모리 버퍼로 구성됩니다. 빠른 메모리 버퍼는 일시적이고 새로운 샘플을 보존하며, 느린 메모리 버퍼는 중요하고 정보가 풍부한 샘플을 유지합니다. 또한, 효율적인 reservoir sampling 프로세스를 사용하여 빠른 메모리 버퍼를 최적화하며, 새로운 정보를 선택적으로 식별하고 저장하는 정보 이론적 메모리 최적화 전략도 도입했습니다.

- **Performance Highlights**: 제안된 방법론은 여러 가지 지속적 학습 실험을 통해 엄격하게 평가되었습니다. 실험 결과는 ITDMS 시스템의 효과성을 뒷받침하며, 새로운 데이터 습득을 위한 메모리 용량을 확보하는 절차에 대한 개선이 두드러지게 나타났습니다.



### Dynami-CAL GraphNet: A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems (https://arxiv.org/abs/2501.07373)
- **What's New**: 본 논문에서는 Dynami-CAL GraphNet이라는 새로운 Physics-Informed Graph Neural Network (GNN) 아키텍처를 제안합니다. 이 방법은 GNN의 학습 능력과 물리 기반 유도 편향을 통합하여 다체 동역학 시스템의 정확하고 해석 가능한 모델링을 수행할 수 있습니다. Dynami-CAL GraphNet은 노드 간의 쌍별 선형 및 각 운동량 보존을 강화하며, 이는 물리적으로 일관된 예측을 보장합니다.

- **Technical Details**: Dynami-CAL GraphNet은 일관된 변환을 유지하는 엣지-로컬 참조 프레임을 사용하여 노드의 쌍별 상호작용에서 발생하는 선형 및 각 임펄스를 제공합니다. 이 모델은 고착 상태에서의 에러 축적을 안정적으로 유지하며, 외부 힘과 이질적 상호작용을 효과적으로 처리합니다. 이 방법은 기존의 GNN 모델이 갖고 있는 물리적 일관성을 일반화하는 데 도움을 줍니다.

- **Performance Highlights**: 3D 미립자 시스템을 기반으로 평가한 결과, Dynami-CAL GraphNet은 장기간에 걸쳐 안정적인 에러 축적을 동반하며, 새롭게 보지 못한 구성에 대한 효과적인 외삽을 보여주었습니다. 복잡한 다체 시스템을 요구하는 로봇 공학, 항공 우주 공학 및 재료 과학 분야에서 실시간 모델링에 있어 큰 장점을 제공합니다.



### Deep Generative Clustering with VAEs and Expectation-Maximization (https://arxiv.org/abs/2501.07358)
- **What's New**: 이번 논문에서는 변분 오토인코더(Variational Autoencoders, VAE)를 기대-극대화(Expectation-Maximization, EM) 프레임워크에 통합한 새로운 깊은 클러스터링 방법을 제안합니다. 이 방법은 각 클러스터의 확률 분포를 VAE로 모델링하고, Evidence Lower Bound (ELBO)를 극대화하며 클러스터 할당을 정제하는 과정을 반복합니다. 기존 VAE 기반 방법들과는 달리, 우리의 방법은 가우시안 혼합 모델(Gaussian Mixture Model, GMM) 사전이나 추가적인 정규화 기법이 필요하지 않습니다.

- **Technical Details**: 제안된 방법은 EM 알고리즘과 VAE 생성 모델을 통합하여 클러스터링을 수행하는 방식으로, 각 클러스터에서 VAE 파라미터를 업데이트하고 클러스터 할당을 갱신합니다. EM 알고리즘은 잠재 변수를 고려한 로그 가능성(log-likelihood)을 극대화하는 반복적인 프레임워크를 제공합니다. 이를 통해, 고차원의 데이터 분포를 더 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: MNIST 및 FashionMNIST 데이터셋에서의 실험 결과, 제안한 방법은 최신 상태의 VAE 기반 클러스터링 방법들보다 뛰어난 성능을 보였습니다. 각 클러스터에서 새로운 샘플을 생성할 수 있는 능력이 이 방법의 주요 장점 중 하나입니다. 이로 인해, 인간의 레이블링 없이도 데이터셋을 풍부하게 만들 수 있는 가능성이 제시되었습니다.



### Enhancing Online Reinforcement Learning with Meta-Learned Objective from Offline Data (https://arxiv.org/abs/2501.07346)
Comments:
          Accepted by AAAI 2025 (this version includes supplementary material)

- **What's New**: GILD(Generalized Imitation Learning from Demonstration)는 강화 학습(RL)과 모방 학습(IL)의 통합을 통해 희소한 보상 문제를 해결하는 혁신적인 방법입니다. 이 방법은 고유한 목적 함수를 스스로 학습하여 비전문가의 서브 최적의 데이터로부터 지식을 증류(disting)하여 궁극적으로 최적 정책(Optimized Policy)으로 나아가도록 지원합니다. GILD는 다양한 오프 정책(off-policy) RL 알고리즘을 지원하며, 도메인 특화 하이퍼파라미터를 사용하지 않으며 계산 비용 증가를 최소화합니다.

- **Technical Details**: GILD는 새로운 이중 최적화(bi-level optimization) 프레임워크를 통해 구현되며, 상위 수준에서 GILD를 메타 최적화하고 하위 수준에서 RL의 메타 훈련(meta-training)을 지원합니다. 이 방법은 오프 정책 RL 알고리즘에 적용되며, 샘플 효율성(sample efficiency)를 극대화하여 기존 RL+IL 방법보다 더 나은 성능을 발휘합니다. 실험 결과, GILD를 적용한 RL 알고리즘이 기존의 RL 및 RL+IL 방법보다 우수한 성과를 기록했습니다.

- **Performance Highlights**: GILD는 특히 네 개의 도전적인 MuJoCo 작업에서 희소한 보상 환경을 처리하는 데 있어 매우 효과적입니다. GILD를 통합한 세 가지 오프 정책 RL 알고리즘(DDPG, TD3, SAC)은 기존 방법들을 압도하는 성과를 보여주며, 최적의 정책에 근접한 성능을 달성했습니다. 또한 GILD는 학습이 빠르게 수렴(convergence)하는 특성이 있어 초기 학습 단계를 최소화하고 다음에 RL만으로 전환하여 훈련 속도를 높일 수 있는 가능성을 보여주고 있습니다.



### Digital Operating Mode Classification of Real-World Amateur Radio Transmissions (https://arxiv.org/abs/2501.07337)
Comments:
          Conference IEEE ICASSP 2025

- **What's New**: 이 연구는 실제 전송에서 평가된 디지털 라디오 운영 모드를 분류하기 위한 ML(기계 학습) 접근 방식을 제시합니다. 17개의 디지털 운영 모드에서 파라미터화된 98개의 라디오 신호를 생성하고, 이를 70cm(UHF) 아마추어 라디오 주파수 대역에서 전송하였습니다. 또한, 두 가지 서로 다른 SDR(소프트웨어 정의 라디오) 수신기로 전송을 기록했습니다.

- **Technical Details**: 연구에서는 랜덤 캐릭터를 페이로드로 사용하여 비전송 신호의 스펙트로그램에 대해 ResNet-18, EfficientNetB0, Vision Mamba Tiny와 같은 경량 ML 모델을 훈련시켰습니다. 훈련 과정에는 여러 라디오 채널 장애를 시뮬레이션하기 위한 온라인 데이터 증강 파이프라인이 포함되었습니다. 모델의 최상의 성능을 보인 EfficientNetB0는 17개의 운영 모드에서 93.80%, 98개의 라디오 신호 전체에서 85.47%의 정확도를 달성했습니다.

- **Performance Highlights**: 모델은 다양한 신호 지속 시간 및 FFT(고속 푸리에 변환) 빈의 수가 분류에 미치는 영향을 분석하고, 다중 SNR(신호 대 잡음 비율) 환경에서도 평가되었습니다. 이 연구는 전세계 아마추어 라디오 상황에서 디지털 운영 모드를 신속하고 정확하게 식별하는 데 기여할 것으로 기대됩니다. 특히 FCC(연방통신위원회)와 같은 스펙트럼 모니터링 기관에 유용할 것입니다.



### TempoGPT: Enhancing Temporal Reasoning via Quantizing Embedding (https://arxiv.org/abs/2501.07335)
- **What's New**: 이번 논문에서는 복잡한 추론 작업에서 기존의 멀티모달 언어 모델(Multi-modal Language Models)과 시간 시계열 데이터(Time Series Data)의 활용 한계를 극복하기 위해, 멀티모달 시간 시계열 데이터 구축 접근법 및 TempoGPT라는 새 모델을 제안합니다. TempoGPT는 시간 임베딩을 정량화(Quantize)하여 텍스트 정보와의 일관된 표현 패턴을 달성하고, 이를 통해 복잡한 추론 과제에서 성능을 향상시킵니다.

- **Technical Details**: TempoGPT는 변수-시스템 간의 관계를 분석하여 복잡한 추론 작업을 위한 멀티모달 데이터를 생성하며, 시간 임베딩을 일련의 이산 토큰으로 변환하는 과정을 포함합니다. 이후, 텍스트와 시간 토큰의 처리를 위한 공유 임베딩 레이어(Embedding Layer)를 설계하여, 두 정보 간의 표현을 일치시킴으로써 멀티모달 정렬(Multi-modal Alignment)을 높입니다. 이러한 접근은 시간 정보와 텍스트 정보 간의 의미 있는 연계를 가능하게 합니다.

- **Performance Highlights**: 실험을 통해 TempoGPT는 구축된 복잡한 시간 시계열 추론 작업에서 최첨단(State-of-the-art) 성능을 달성했음을 입증합니다. 또한, 시간 임베딩의 정량화가 멀티모달 정렬 및 TLM(Temporal Language Models)의 추론 능력을 향상시키는 데 효과적이라는 것을 정량적으로 입증하며, 이는 향후 다양한 응용 분야에서의 활용 가능성을 높입니다.



### Foundation Models at Work: Fine-Tuning for Fairness in Algorithmic Hiring (https://arxiv.org/abs/2501.07324)
Comments:
          Accepted to AAAI 2025, AI Governance Workshop

- **What's New**: 본 연구에서는 기존의 인간 피드백 없이 목표 지향적인 미세 조정을 자동화하는 AutoRefine 기법을 제안합니다. 이 방법은 Reinforcement Learning (강화 학습) 개념을 활용하여, 특정 하위 작업의 성능 향상에서 얻은 직접적인 피드백을 기반으로 모델을 미세 조정하는 과정입니다. 이를 통해 고용 플랫폼에서의 언어적 편향 문제가 발생하는 상황에 대한 사례를 다룹니다.

- **Technical Details**: AutoRefine는 세 가지 주요 단계로 구성되어 있습니다: 첫째, 특정 하위 작업에 맞춰 모델을 정렬하고, 둘째, 성과 메트릭에 따라 출력을 최적화하는 왜곡 메커니즘을 도입하며, 셋째, 결합된 시스템을 배치하는 것입니다. 이 과정에서 pre-trained foundation model과 성과 평가자가 연관되어, model의 출력을 보다 효과적으로 조정합니다.

- **Performance Highlights**: 실험 결과, AutoRefine는 공공 고용 데이터 세트 및 실제 고용 플랫폼에서의 테스트로, 편향을 감지하고 완화하는 데 있어 대규모 언어 모델이 효과적으로 활용될 수 있음을 입증하였습니다. 이 방식은 다양한 후보자가 매칭되는 공정하고 포괄적인 채용 시스템을 구축하는 데 기여할 수 있습니다.



### Evaluation of Artificial Intelligence Methods for Lead Time Prediction in Non-Cycled Areas of Automotive Production (https://arxiv.org/abs/2501.07317)
- **What's New**: 이번 연구는 비주기 제어 생산 환경에서 인공지능(Artificial Intelligence) 방법을 적용하여 자동차 생산 공정에서의 리드 타임(lead time)을 예측하는 효과성을 살펴봅니다. 연구에서는 데이터 구조를 분석하고, One-Hot Encoding을 사용하여 전처리한 후, 감독 학습(supervised learning) 방법 중 회귀(regression)와 분류(classification) 기법을 평가했습니다. 특히, Gradient Boosting 알고리즘인 LightGBM을 포함해 XGBoost와 CatBoost가 우수한 성능을 보였다고 보고하였습니다. 최종적으로, LightGBM 알고리즘이 선택되었다고 언급하며, 상대적 예측 정확도가 90%까지 도달할 수 있다는 점을 강조했습니다.

- **Technical Details**: 연구에서 사용된 기술적 세부사항으로는 Python을 프로그래밍 언어로 사용하였고, Intel Core i9-12900KF 프로세서와 Nvidia RTX 3060 GPU를 갖춘 워크스테이션에서 알고리즘을 실행하였습니다. 데이터는 차량의 리드 타임에 영향을 미치는 다양한 특성을 고려하여 수집되고 저장되며, 각각의 차량에 대해 고유 식별 키(unique identification key)를 사용해 링크합니다. 또한, 공정 전후의 특성에 따라 알려진 특성과 공정 중에 기록된 특성을 구분하여 분석합니다.

- **Performance Highlights**: 초기 연구 결과는 AI 방법이 자동차 생산에서의 리드 타임 예측에 효과적으로 적용될 수 있음을 보여줍니다. 특히, 이전 연구 결과와 비교했을 때, LGBM 회귀에서는 22%의 예측 정확도를 보였으나, LGBM 분류에서는 65%에서 71%의 정확도를 기록했습니다. 하지만 추가적인 하이퍼파라미터 최적화가 필요하다는 점과 함께, AI 모델의 주기적인 재학습이 복잡한 생산 과정을 정확히 표현하는 데 중요하다는 점이 강조되었습니다.



### Variable Bregman Majorization-Minimization Algorithm and its Application to Dirichlet Maximum Likelihood Estimation (https://arxiv.org/abs/2501.07306)
- **What's New**: 이 논문에서는 클래식한 convex minimization 문제를 해결하기 위해 새로운 Bregman descent 알고리즘을 제안합니다. 제안된 접근방식인 Variable Bregman Majorization-Minimization (VBMM) 알고리즘은 Bregman Proximal Gradient 방법을 확장해 각 반복에서 사용하는 Bregman 함수가 적응적으로 변화하도록 하여, 목표 함수에서 majorizing 조건을 만족합니다. 이러한 적응형 프레임워크는 알고리즘이 각 반복에서 좀 더 정확하게 목표 함수를 근사할 수 있게 해주며, 기존의 Bregman Proximal Gradient descent에 비해 수렴 속도를 가속화합니다.

- **Technical Details**: VBMM 알고리즘은 convex 함수 F의 최소화를 위한 새로운 방법론으로, 여기서 F는 미분 가능한 부분과 비미분 가능할 수 있는 항의 합으로 구조화됩니다. 이 알고리즘은 Bregman 거리 D_h를 사용하여 각 반복에서 목표 함수의 지역 기하학에 부합하게 동적으로 변화시킵니다. 본 연구에서는 이론적인 수렴을 위해 필요한 완만한 가정을 설정하였으며, Bregman Proximal Gradient 방법과 VBMM 알고리즘을 Dirichlet 분포의 다차원 매개변수 추정에 적용하는 새로운 방법론도 소개합니다.

- **Performance Highlights**: 수치 실험 결과에 따르면, VBMM 알고리즘은 기존의 접근 방식들에 비해 수렴 속도에서 뛰어난 성능을 발휘하는 것으로 확인되었습니다. 특히, 포아송 복원 문제와 최대 우도(Maximum Likelihood) 접근 방식과 같은 실제 응용 문제에서 유용성을 보여줍니다. VBMM 알고리즘의 적응 적 Bregman 거리 활용은 다양한 최적화 문제에서 더 나은 결과를 도출할 수 있는 잠재력을 가지고 있습니다.



### Generating Poisoning Attacks against Ridge Regression Models with Categorical Features (https://arxiv.org/abs/2501.07275)
- **What's New**: 이번 논문은 범주형 특성을 포함하는 릿지 회귀 모델에 대한 강력한 중독 공격(poisoning attack)을 생성하는 알고리즘을 제안합니다. 특히, 기존의 접근 방식들이 범주형 변수를 적절하게 최적화하지 못하는 한계를 지적하며, 이를 해결하기 위한 새로운 혼합 정수 이층 최적화 모델을 도입했습니다. 이 연구는 적대적 머신러닝 분야에서 응용될 수 있는 새로운 가능성을 제시합니다.

- **Technical Details**: 성능 저하를 목표로 하는 중독 공격을 구성하기 위해, 이 논문은 범주형 변수를 원-핫 인코딩(one-hot encoding) 방식으로 변환하고 이를 혼합 정수 이층 최적화 문제로 설정합니다. 공격자는 회귀 모델의 예측 오차, 즉 평균 제곱 오차(mean squared error, MSE)를 최대화하는 것을 목표로 하며, 이는 설계된 하층의 최적 반응을 고려하여 이루어집니다. 본 연구에서는 이를 특별 순서 집합(SOS-1)으로 모델링하는 접근법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 문헌에서의 기존 기준과 비교했을 때 모든 데이터 세트의 평균 제곱 오차를 현저히 개선하였습니다. 이는 범주형 특성을 대상으로 하는 중독 공격이 특히 효과적인 방안을 제공한다는 것을 보여줍니다. 또한, 이 연구는 향후 적대적 기계 학습 연구에 있어 중요한 기초 자료로 활용될 수 있습니다.



### MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework (https://arxiv.org/abs/2501.07251)
Comments:
          Under Review of CVPR 2025

- **What's New**: 이번 논문에서는 Multi-Objective Set-based Attack(MOS Attack)이라는 새로운 적대적 공격 프레임워크를 제안합니다. MOS Attack은 여러 개의 손실 함수(loss function)를 활용하고 그들 간의 상호 관계를 자동으로 발견하는 것을 목표로 합니다. 이 접근법을 통해 기존 단일 목표 공격 방법의 한계를 극복하고 효과적인 적대적 예제 생성을 가능하게 합니다.

- **Technical Details**: MOS Attack은 집합 기반의 다중 목표 최적화 전략을 사용하여 추가적인 매개변수 없이도 여러 손실 함수를 통합합니다. 이 과정에서 각 손실 간의 시너지 효과(synergistic patterns)를 자동으로 탐색하여 적대적 공격을 더 강력하고 효율적으로 만듭니다. 특히 이 방법은 적대적 공격의 예제를 생성하기 위해 여러 손실 기능을 동시다발적으로 고려할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실험을 통해 MOS Attack이 기존의 단일 목표 공격 방법들보다 뛰어난 성과를 보여주는 것을 확인했습니다. 또한 신시아 형태를 통해 적은 수의 손실 함수로도 더 나은 결과를 달성할 수 있음을 보여주며, 이는 새로운 적대적 공격 전략의 가능성을 제시합니다.



### Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training (https://arxiv.org/abs/2501.07237)
- **What's New**: 이 논문은 Gradient Wavelet Transform (GWT)라는 새로운 메모리 효율적인 방법을 제안합니다. 이 방법은 gradient에 wavelet 변환을 적용하여 옵티마이저 상태를 유지하는 데 필요한 메모리를 크게 줄입니다. GWT는 메모리 집약적인 옵티마이저와 원활하게 통합될 수 있으며, 성능 저하 없이 효율적인 학습을 가능하게 합니다. 실험 결과, 기존의 메모리 효율적 옵티마이저 및 풀-랭크 방법들과 비교하여 주목할 만한 성과를 달성했습니다.

- **Technical Details**: GWT는 Haar 및 Daubechies-2 (DB2)와 같은 이산 wavelet을 필터로 사용하여 구현됩니다. 이 방법은 gradient의 차원을 효과적으로 압축하여 옵티마이저 상태를 저장하는 데 필요한 메모리를 줄입니다. 저자들은 GWT를 Adam 옵티마이저와 통합하여 LLaMA 모델의 사전 학습 및 파인 튜닝 작업에서 성능을 평가했습니다. 실험적으로, GWT는 특히 사전 훈련에서 메모리 사용을 67%까지 줄이는 동시에 학습 속도를 증가시켰습니다.

- **Performance Highlights**: GWT는 사전 훈련 및 파인 튜닝 작업에서 풀-랭크 옵티마이저들과 동등하거나 더 나은 성능을 보여줍니다. C4 데이터세트에서 LLaMA 모델을 사전 훈련하는 동안, GWT는 옵티마이저 메모리 사용량을 감소시키면서도 학습 속도를 증가시키는 효과를 입증했습니다. 또한, 하이퍼파라미터 조정의 영향을 탐구하는 ablation 연구를 통해 GWT의 성능을 극대화할 수 있는 방법을 제안하고 있습니다.



### A data-driven approach to discover and quantify systemic lupus erythematosus etiological heterogeneity from electronic health records (https://arxiv.org/abs/2501.07206)
Comments:
          Received Runner-up Knowledge Discovery and Data Mining Innovation Award at the American Medical Informatics Association Annual Symposium 2024

- **What's New**: 본 논문에서는 전자 건강 기록(Health Record; EHR)의 여러 비정상적인 데이터를 이용해 전신홍반성 루푸스(Systemic Lupus Erythematosus; SLE)의 독립적인 확률적 원천(probabilistic sources)을 발견하는 데이터 기반 접근 방식을 제안합니다. 이러한 원천들은 SLE의 잠재적인 원인(root causes)을 추정하기 위한 외적 변수(exogenous variables)를 나타냅니다. 이 방법론은 전통적인 자료 분석 방식에 새로운 시각을 제공합니다.

- **Technical Details**: 연구진은 SLE와 부정적인 건강 기록을 구별하기 위해, 라벨이 붙은 인스턴스의 축소 집합를 활용하여 감독(monitoring) 학습(supervised learning) 모델을 훈련하였습니다. 그 결과 19개의 임상적 유효성(clinical validity)이 높은 예측 소스를 발견하였고, 이를 통해 SLE의 이질성을 나타내는 독립적인 요인들을 정의하였습니다. 이러한 소스들은 EHR 서명을 통해 설명 가능성을 높이고, SLE 사례의 임상적 이유를 더 잘 포착하는 데 기여합니다.

- **Performance Highlights**: 이 모델을 통해 얻은 예측 소스들은 환자 데이터의 풍부한 설명을 제공하며, 특정 기록이 SLE 사례인지 아닌지를 판별하는 데 중요한 역할을 합니다. 의료 제공자들은 난이도가 높은 사례에서 환자 수준의 해석 가능성(interpretability)과 판별(discrimination) 사이의 균형을 맞추는데 이러한 모델의 도움을 받을 수 있습니다. 실제 임상에서의 적용 가능성을 높이기 위해 이 질병의 복잡한 특징들을 보다 잘 이해하는 데 기여할 것으로 기대됩니다.



### An Enhanced Zeroth-Order Stochastic Frank-Wolfe Framework for Constrained Finite-Sum Optimization (https://arxiv.org/abs/2501.07201)
Comments:
          35 pages, 4 figures, 3 tables

- **What's New**: 이번 연구에서는 대규모 기계 학습 응용 프로그램에서의 제약 조건을 가진 유한 합 최적화 문제를 해결하기 위해 향상된 제로 차순 확률적 프랭크-울프(Frank-Wolfe) 프레임워크를 제안합니다. 새로운 이중 분산 감소 프레임워크를 통해 제로 차순 오라클과 유한 합 목표로부터 발생하는 기울기 근사 분산을 효과적으로 줄입니다. 이 알고리즘은 고차원 최적화 작업에 적합하며, 특히 제약 조건이 있는 볼록 및 비볼록 목표에 대한 쿼리 복잡도를 획기적으로 개선합니다.

- **Technical Details**: 볼록 목표의 경우, 쿼리 복잡도는 O(d √n / ε)이며, 비볼록 목표의 경우는 O(d^{3/2}√n / ε²)로 설정됩니다. 여기서 d는 차원 수, n은 유한 합 목표에 있는 함수의 수를 나타냅니다. 이 복잡도는 명시적인 기울기 계산을 피하는 제로 차순 확률적 프랭크-울프 알고리즘 중에서 가장 잘 알려진 값입니다.

- **Performance Highlights**: 실험 결과, 희소 로지스틱 회귀, 강건 분류, 심층 신경망의 적대적 공격 등 볼록 및 비볼록 기계 학습 작업에서 제안한 알고리즘이 기존 방법과 비교하여 수렴 속도와 쿼리 복잡도에서 탁월한 성능을 보여주었습니다. 이 알고리즘은 계산 효율성과 확장성이 뛰어나기 때문에 실제 응용 프로그램에서도 효과적으로 사용할 수 있습니다.



### Generalizable Graph Neural Networks for Robust Power Grid Topology Contro (https://arxiv.org/abs/2501.07186)
- **What's New**: 이 논문은 그래프 신경망(GNN)을 활용한 전력망(grid) 토폴로지 제어를 위한 최초의 GNN 모델을 제안합니다. 또한, 기존의 동질적 그래프 표현이 가지고 있는 버스바 정보 비대칭 문제를 확인하고 이를 해결하기 위한 이종 그래프 표현을 제안합니다. 이 연구는 전력망 운영의 문제를 해결하기 위해 GNN의 깊은 활용 가능성을 탐구합니다.

- **Technical Details**: 이 연구는 GNN과 완전히 연결된 신경망(FCNN)을 기반으로 하는 모델을 비교하며, 이종 그래프 표현이 GNN 성능에 미치는 영향을 분석합니다. GNN은 그래프 구조를 반영하여 전력망을 모델링하는 데 적합한 다층 구조로 구성되어 있습니다. 또한, GNN 모델은 다양한 네트워크와 토폴로지에서 다중 레이블 이진 분류 모방 학습(task)으로 훈련됩니다.

- **Performance Highlights**: 이 연구에서 제안된 이종 GNN 모델은 주어진 기준 네트워크에서 이전 모델들보다 더 뛰어난 성능을 보였습니다. 특히, 이 모델은 FCNN보다도 아웃 오브 디스트리뷰션 네트워크에서 더 잘 일반화되는 모습을 보였습니다. 이러한 결과는 GNN을 활용한 토폴로지 제어 방법ologies의 발전 가능성을 보여줍니다.



### Knowledge Distillation and Enhanced Subdomain Adaptation Using Graph Convolutional Network for Resource-Constrained Bearing Fault Diagnosis (https://arxiv.org/abs/2501.07173)
- **What's New**: 이번 논문에서는 다양한 작업 조건에서 베어링 결함 진단에 대한 도전 과제를 해결하기 위한 새로운 접근 방식인 진보적인 지식 증류 프레임워크를 제안합니다. 이 프레임워크는 복잡한 교사 모델(teacher model)로부터 정보를 전이하여, Graph Convolutional Network (GCN)과 Autoregressive moving average (ARMA) 필터를 사용하는 효율적인 학생 모델(student model)로 전달합니다.

- **Technical Details**: 논문에서는 Enhanced Local Maximum Mean Squared Discrepancy (ELMMSD) 기법을 도입하여, 분포 불일치(distribution discrepancies)와 레이블의 불확실성(labeling uncertainty)을 완화합니다. ELMMSD는 Reproducing Kernel Hilbert Space (RKHS) 내의 평균 및 분산 통계를 활용하며, 레이블 간의 사전 확률 분포를 통합하여 클러스터링 센터 간의 거리를 증가시킵니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 베어링 결함 진단에서 뛰어난 정확도를 달성할 뿐만 아니라 컴퓨팅 비용을 상당히 줄임을 보여줍니다. CWRU 및 JNU와 같은 벤치마크 데이터셋에서의 결과를 통해 각 구성 요소의 효과성을 강조하며, 다양한 작업 조건에서 접근 방식의 강건함과 적응성을 증명합니다.



### Anomalous Agreement: How to find the Ideal Number of Anomaly Classes in Correlated, Multivariate Time Series Data (https://arxiv.org/abs/2501.07172)
Comments:
          Acccepted at AAAI Workshop on AI for Time Series Analysis (AI4TS) 2025

- **What's New**: 이 논문에서는 Synchronized Anomaly Agreement Index (SAAI)를 도입하여 다변량 시계열에서 이상치(anomalies)의 동기성을 활용하여 클러스터 품질을 평가하는 방법을 제안합니다. 기존의 클러스터 품질 평가 방법인 Silhouette Score (SSC)와 달리, SAAI는 데이터에 대한 사전 지식을 반영하여 더 나은 성능을 보여줍니다. 이 방법은 이상 상태의 검출 및 분류를 더욱 효과적으로 만들 수 있습니다.

- **Technical Details**: SAAI는 다변량 시계열 데이터에서 발생하는 이상 상태의 동기화된 패턴을 분석하여 클러스터 품질을 높입니다. 이 지표는 클러스터 내의 일관성(cohesion)과 클러스터 간의 구분(separation)을 평가하지만, 데이터의 사전 지식도 고려하여 총 K개의 이상 클래스 수를 추정하는 정확도를 향상시킵니다. 이를 통해, 연구자들은 향후 비지도 학습(unsupervised learning) 기반의 시스템 모니터링에서 더 향상된 결과를 도출할 수 있습니다.

- **Performance Highlights**: SAAI의 활용으로 SSC와 비교했을 때 이상 클래스의 정확도가 0.23 향상되었으며, X-Means와 비교 시에는 0.32 향상되는 결과를 보였습니다. SAAI로 최적화된 클러스터는 SSC로 얻어진 클러스터보다 해석하기도 쉽습니다. 이러한 성과는 클러스터 품질 평가의 새로운 기준을 설정하며, 다양한 응용 분야에서 기여할 수 있는 가능성을 보여줍니다.



### AlphaNet: Scaling Up Local Frame-based Atomistic Foundation Mod (https://arxiv.org/abs/2501.07155)
Comments:
          14 pages, 5 figures

- **What's New**: 최근 발표된 AlphaNet은 원자 시스템에 대한 정확하고 효율적인 시뮬레이션을 목표로 하는 지역 프레임 기반의 동등 변환 모델입니다. 기존의 기계 학습 기반 힘장(Machine Learning Force Fields, MLFF) 모델들이 직면한 효율성과 확장성의 문제를 해결하고자 합니다. AlphaNet은 원자 환경의 지역 기하학적 구조를 활용하여 동등 변환 지역 프레임과 학습 가능 프레임 전이를 구축하여 계산 효율성과 정확성을 향상시킵니다.

- **Technical Details**: AlphaNet은 3D 유클리드 공간의 대칭성(예: 회전, 이동 및 반사)을 존중하며, 텐서곱 계산의 필요성을 제거하는 프레임 기반 접근 방식을 이용해 높은 계산 효율성을 자랑합니다. 또한, 기존의 프레임 기반 MLFF의 성공을 바탕으로 회전 위치 임베딩을 추가하여 다중 스케일 모델링의 프레임 전이 및 시간 연결성을 개선합니다. AlphaNet은 결함 그래핀, 포르메이트 분해, 제올라이트 및 표면 반응을 포함한 다양한 데이터 세트에서 강한 성능을 발휘합니다.

- **Performance Highlights**: AlphaNet은 에너지 및 힘 예측의 정확도에서 NequIP 및 DeepPot와 같은 기존 모델을 지속적으로 초월하며, 계산 효율성 및 정확성 간의 최상의 균형을 제공합니다. AlphaNet은 다양한 시스템 및 데이터 세트 크기에서 뛰어난 확장성을 보여주며, 이는 실제 응용 프로그램에서의 다재다능성을 입증합니다. 실험 데이터세트에서 AlphaNet은 높은 정확도로 양자화된 결과를 도출하여 복잡한 시스템의 미세한 상호작용을 효과적으로 포착합니다.



### TIMRL: A Novel Meta-Reinforcement Learning Framework for Non-Stationary and Multi-Task Environments (https://arxiv.org/abs/2501.07146)
- **What's New**: 최근 메타 강화 학습( meta-reinforcement learning ) 분야에서 샘플 효율성을 개선하기 위한 새로운 알고리즘인 TIMRL( Task Inference with GMM and transformer for efficient Meta-RL ) 이 제안되었습니다. 특히 Gaussian mixture model(GMM)과 transformer 네트워크를 이용하여 작업 추론 모델(task inference model)을 구축함으로써 비정상 환경에서의 적응성을 높였습니다. 기존 방법들과 달리, TIMRL은 다양한 작업 환경에서 적은 수의 샘플만으로도 효율적으로 작업을 인식하고 분류하는 방식으로 설계되었습니다.

- **Technical Details**: 이 연구에서는 GMM 기반의 작업 추론 모델을 통해 비정상적인 환경 및 다중 작업 환경에서 메타 강화 학습의 효율성을 높이기 위해 여러 기술적인 구현이 이루어졌습니다. GMM을 사용하여 작업의 표현을 확장하고, transformer 네트워크를 통해 작업 분류를 명시적으로 인코딩합니다. 또한, 사전 처리 기법을 통해 다중 작업 환경에서 상태-액션 차원 공간을 정규화하고, 각 작업 분류를 개별 Gaussian 구성 요소로 모델링하여 복잡한 작업을 효과적으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 실험은 MuJoCo 벤치마크를 사용하여 비정상 및 다중 작업 환경에서 수행되었으며, 결과는 제안된 TIMRL 알고리즘이 탁월한 샘플 효율성 및 정확한 작업 분류 성능을 바탕으로 훌륭한 결과를 나타냄을 보여줍니다. 전반적으로, TIMRL은 기존의 메타 강화 학습 알고리즘에 비해 더 빠른 탐색 및 환경 적응을 가능하게 하여, 다중 작업 환경에서의 뛰어난 성능을 자랑합니다.



### LLM360 K2: Scaling Up 360-Open-Source Large Language Models (https://arxiv.org/abs/2501.07124)
- **What's New**: LLM360 K2-65B 모델의 훈련 과정을 자세히 설명합니다. 이 프로젝트는 360도 오픈 소스 접근 방식을 통해 가장 크고 강력한 LLM을 발전시키고 있습니다. 특히, 대규모 모델 훈련의 투명성을 제공해 과거의 통찰력을 활용할 수 있는 기회를 모든 연구자에게 열어줍니다.

- **Technical Details**: K2 DIAMOND라는 65억 개의 파라미터를 가진 모델은 LLaMA-65B를 초월하고 LLaMA2-70B와 경쟁하는 성능을 보이며, FLOPs 및 토큰 사용량이 적습니다. 이 논문은 K2 DIAMOND 모델의 훈련 과정에서의 구현 단계 및 성능을 분석하고 있습니다. 더불어 TXT360와 같은 진행 중인 프로젝트도 언급하여 미래 모델에 대한 기반을 마련하고 있습니다.

- **Performance Highlights**: K2 프로젝트는 LLM 훈련에 대한 중요한 자원과 통찰력을 제공하며, 이러한 정보는 연구자들과 개발자들이 효율적으로 모델을 개선하는 데 기여할 수 있습니다. K2 모델은 높은 성능과 적은 자원 소모를 동시에 충족하는 점에서 주목할 만합니다. 또한, 다양한 지속 가능한 접근 방식을 통해 연구 커뮤니티에서의 투명성과 접급성을 증진시키고 있습니다.



### D3MES: Diffusion Transformer with multihead equivariant self-attention for 3D molecule generation (https://arxiv.org/abs/2501.07077)
- **What's New**: 이 논문에서는 Diffusion Transformer와 multihead equivariant self-attention을 결합한 3D 분자 생성을 위한 diffusion model을 소개합니다. 이 방법은 수소 원자가 제거된 후 분자의 표현을 학습하여 생성된 분자에 수소 원자를 올바르게 부착하는 문제를 해결하고, 동시에 여러 클래스의 분자를 동시에 생성하는 기존 모델의 한계를 극복합니다. 실험 결과, 제안한 모델은 여러 주요 지표에서 최첨단 성능을 달성하며, 강력함과 다재다능함을 보여줍니다.

- **Technical Details**: 제안된 Diffusion Transformer 모델(D3MES)은 기존의 U-Net 아키텍처 대신에 transformer 기반 아키텍처를 사용하여 소음(ϵ)과 분산(Σ)을 학습 및 예측합니다. 입력 데이터는 VAE 처리 과정을 통해 생성된 잠재 공간(z=E(x))으로, 이를 patchify 작업을 통해 패치 단위로 나누어 처리하며, 각 토큰은 Sinusoidal Positional Encoding을 적용받아 순서 정보를 포함합니다. 모델은 self-attention 메커니즘을 채택하여 관련성이 높은 토큰에 집중하여 처리합니다.

- **Performance Highlights**: D3MES 모델은 GEOM-QM9 및 GEOM-Drugs 데이터셋을 기반으로 한 실험에서 우수한 성능을 보였습니다. 특히, 대형 분자 데이터셋에서 두드러지는 성과를 달성하여, 초기 분자 생성 단계에서 후보 분자를 효율적으로 생성할 수 있는 가능성을 제시합니다. 이를 통해 실험적으로 검증 및 스크리닝 단계를 거쳐 특정 특성을 가진 분자를 얻을 수 있는 기반이 마련되었습니다.



### ACCon: Angle-Compensated Contrastive Regularizer for Deep Regression (https://arxiv.org/abs/2501.07045)
Comments:
          Accept by AAAI-2025 (The 39th Annual AAAI Conference on Artificial Intelligence)

- **What's New**: 이번 논문에서는 딥 회귀(Deep Regression)에서 발생하는 연속 레이블 간의 관계를 포착하기 위한 새로운 접근법을 제안합니다. 기존의 일반화된 회귀 모델과 달리, 각 레이블 간의 거리와 유사성 사이의 선형적인 음의 상관관계를 가정하고, 앵글 보정 대조 정규화기(Angle-Compensated Contrastive Regularizer)를 개발하였습니다. 이는 대조 학습(framework) 내에서 앵커와 음성 샘플들 간의 코사인 거리(cosine distance)를 조정하는 방식으로, 현재의 대조 학습 방법을 회귀 작업에 효과적으로 확장할 수 있는 솔루션을 제공합니다.

- **Technical Details**: 제안된 방법은 다양한 대조 학습 기반 접근 방식을 통합 가능하며, 입력을 반-하이퍼스피어에 투사하여 레이블 간의 관계를 보존합니다. 이를 통해, 레이블 간의 거리에 따라 최적화된 유사성을 유지하면서, 대조 학습에서 필요한 양수 및 음수 샘플의 정의도 명확히 합니다. 실험을 통해 제안한 방식이 기존의 임밸런스 데이터에서의 성능을 크게 향상시키며, 비대칭성 문제를 효과적으로 해결함을 보여줍니다.

- **Performance Highlights**: 제안된 앵글 보정 대조 정규화기는 여러 데이터셋(특히 컴퓨터 비전 및 자연어 처리 분야)에서 경쟁력 있는 회귀 성능을 달성하였습니다. 실험 결과는 저자들이 주장하는 데이터 효율성(data efficiency)과 임밸런스 데이터 세트에 대한 효과성을 나타냅니다. 이 연구는 향후 딥 회귀 모델링에서 앵글 보정 방식을 도입해 실질적인 향상을 이끌어낼 잠재력을 가지고 있습니다.



### Explore the Use of Time Series Foundation Model for Car-Following Behavior Analysis (https://arxiv.org/abs/2501.07034)
- **What's New**: 이번 연구는 자동차 추적 행동(modeling car-following behavior)을 효과적으로 분석하기 위해 최신의 public time series foundation model인 Chronos를 적용했습니다. Chronos는 방대한 시간대 데이터셋으로 사전 훈련(pre-trained)되어 다양한 작업에 직접 활용될 수 있으며, 개별적인 재훈련 없이도 높은 성능을 보입니다. 이 모델은 전통적인 모델인 IDM 및 Exponential smoothing (ETS)을 초월하여 car-following 행동 예측에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: Chronos는 minimal fine-tuning으로 특정 작업에 적응할 수 있는 강력한 능력을 가지고 있으며, Open ACC dataset을 활용하여 car-following 행동을 분석했습니다. 본 연구에서 Chronos는 전통적인 모델들과 비교했을 때 RMSE(Root Mean Square Error) 0.60을 달성하며, 심층 학습(deep learning) 모델인 DeepAR 및 TFT와 유사한 성능을 보였습니다. 세부 조정을 통해 Chronos의 RMSE를 0.53으로 개선하여, IDM보다 33.75% 향상된 성능을 나타냅니다.

- **Performance Highlights**: Chronos는 자동차 추적 행동 예측에서 전통적인 모델과 비교해 비약적인 성능 개선을 보여줍니다.기존 모델들보다 12-37%의 오차 감소를 기록하며, 이는 ETS 및 DeepAR, WaveNet, TFT 같은 심층 학습 모델들과 비교했을 때도 효과적인 결과입니다. 본 연구는 foundation models가 교통 연구에 미치는 잠재력과 강력한 예측 및 시뮬레이션 능력을 입증하며, 더욱 확장 가능하고 적응 가능한 접근 방식을 제시합니다.



### Detection of AI Deepfake and Fraud in Online Payments Using GAN-Based Models (https://arxiv.org/abs/2501.07033)
Comments:
          The paper will be published and indexed by IEEE at 2025 8th International Conference on Advanced Algorithms and Control Engineering (ICAACE 2025)

- **What's New**: 이번 연구에서는 Generative Adversarial Networks (GANs)를 활용하여 AI 딥페이크 및 온라인 결제 시스템의 사기 행위를 감지하는 새로운 방법을 탐구합니다. 딥페이크 기술의 확산으로 인해 온라인 거래에서의 사기 가능성이 증가하고 있으며, 전통적인 보안 시스템은 이러한 복잡한 사기 형태를 식별하는 데 어려움을 겪고 있습니다.

- **Technical Details**: 이 연구는 StyleGAN과 DeepFake와 같은 고도화된 GAN 아키텍처를 이용하여 실제 온라인 결제 이미지와 생성된 딥페이크 이미지로 구성된 데이터셋을 기반으로 모델을 훈련시켰습니다. 제안된 GAN 기반 모델은 결제 이미지 내의 미세한 조작을 식별함으로써 온라인 결제 보안을 강화하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 합법적인 거래와 딥페이크를 95% 이상의 높은 감지율로 정확하게 구별할 수 있음을 보여주었습니다. 이는 AI 기반 사기에 대한 결제 시스템의 강인성을 상당히 향상시키며, 금융 서비스에서의 사기 감지에 대한 GAN의 적용을 탐구하는 데 기여합니다.



### PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks (https://arxiv.org/abs/2501.07032)
Comments:
          23 pages

- **What's New**: 이번 논문에서는 Kolmogorov-Arnold Networks (KANs)의 새로운 변형인 PRKANs를 소개합니다. PRKAN은 KAN의 층에서 파라미터 수를 줄이는 여러 방법을 사용하여 Multi-Layer Perceptrons (MLPs)와 유사한 성능을 발휘하는 모델입니다. 주목할만한 점은 PRKAN이 Attention Mechanisms를 통합해 MLP와 경쟁하는 성능을 보여주며, 기존 KAN의 단점을 해결하고자 한 새로운 접근 방식을 제시한다는 것입니다.

- **Technical Details**: PRKAN은 Attention Mechanisms, 차원 합산(dimensions summation), Feature Weight Vectors, 및 Conv/Pooling Layers를 통해 KAN의 파라미터 수를 효과적으로 줄입니다. 이러한 방법들은 네트워크 구조를 유지하면서도 기능 표현을 강화하고, 중요한 정보를 강조하며, 공간 의존성을 포착하는 등의 장점을 제공합니다. 특히, Gaussian Radial Basis Functions (GRBFs) 및 레이어 정규화(layer normalization)가 KAN 디자인에서 효과적임을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면, PRKAN은 MNIST와 Fashion-MNIST 데이터셋에서 기존 KAN보다 우수한 성능을 보이며, MLP와 맞먹는 성능을 자랑합니다. 다만 훈련 시간은 다소 길어지지만, 구조와 파라미터 수 측면에서 KAN을 더 효율적이고 컴팩트하게 만들어줍니다. 이는 향후 KAN 파라미터 수를 줄이기 위한 연구 방향을 제시하며, 신경망의 설계에 있어 새로운 가능성을 열어줍니다.



### Neural Probabilistic Circuits: Enabling Compositional and Interpretable Predictions through Logical Reasoning (https://arxiv.org/abs/2501.07021)
- **What's New**: Neural Probabilistic Circuits (NPCs)라는 새로운 모델 아키텍처를 소개합니다. NPCs는 컴포지션(compositional) 및 해석 가능한 예측을 가능하게 하는 논리적 추론(logical reasoning)을 통해 본질적으로 투명한 모델입니다. 이 모델은 두 가지 모듈로 구성되어 있으며, 속성 인식 모델과 확률론적 회로를 기반으로 한 작업 예측기가 포함되어 있습니다.

- **Technical Details**: NPC는 속성 인식(attribute recognition) 모델과 작업 예측(task predictor) 모듈로 구성됩니다. 속성 인식 모델은 입력 이미지에 대한 확률 벡터를 생성하고, 이 벡터는 확률론적 회로(probabilistic circuit)로 구현된 작업 예측기의 입력으로 사용됩니다. 또한 NPC는 세 단계로 구성된 훈련 알고리즘을 통해 학습되며, 여기에는 속성 인식, 회로 구성 및 공동 최적화(joint optimization)가 포함됩니다.

- **Performance Highlights**: 실험 결과, NPC는 이미지 분류(image classification) 작업에서 해석력과 성능 간의 균형을 잘 유지하며, 엔드-투-엔드(End-to-End) 블랙박스 모델과 경쟁할 수 있는 성능을 보여줍니다. 또한, NPC는 세 가지 개념 기반 모델보다 우수한 성과를 보였으며, 다양한 유형의 설명을 제공하여 인간의 이해를 돕습니다.



### AlgoRxplorers | Precision in Mutation -- Enhancing Drug Design with Advanced Protein Stability Prediction Tools (https://arxiv.org/abs/2501.07014)
- **What's New**: 이 연구는 단일 아미노산 돌연변이가 단백질 안정성에 미치는 영향을 예측하기 위해 딥 뉴럴 네트워크를 활용하는 접근 방식을 제안합니다. 특히, Transfer Learning(전이 학습)과 다양한 모델의 보완 정보를 융합하여 단백질 안정성의 복합적인 representation(표현)을 생성하는 데 초점을 맞추고 있습니다. 사전 훈련된 모델인 ThermoMPNN+가 ΔΔG 예측에서 가장 뛰어난 성과를 보였습니다. 이 방법은 단백질 역학에 대한 이해를 심화시키고 질병 연구 및 약물 발견에 기여할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 연구 방법론에서는 FireProtDB라는 데이터베이스를 활용하여 다양한 단백질 돌연변이와 구조 정보를 제공합니다. Latent transfusion(잠재적 전이) 기법을 통해 여러 딥러닝 모델에서 학습된 latent features(잠재 특징)를 융합합니다. 또한, 데이터 탐색적 분석을 통해 k-means clustering(군집화) 알고리즘을 사용하여 ΔΔG 예측을 위한 여러 모델 간의 성능 비교를 진행했습니다. 최종적으로 ThermoMPNN 모델을 기반으로 한 새로운 구조에서 예측 모델을 개선하고 단백질 안정성 변화 예측을 정확하게 수행하고자 합니다.

- **Performance Highlights**: 제안하는 모델은 기존 방법들보다 제한된 훈련 예제에서도 뛰어난 성능을 발휘하며, 약물 발견 시간과 비용을 크게 줄일 수 있는 가능성을 보여줍니다. 실험 결과, ThermoMPNN 모델과 ESM 임베딩의 조합이 ΔΔG 예측에서 매우 유망한 결과를 나타냈습니다. 다층 퍼셉트론과 경량 주의 메커니즘을 통해 입력 데이터를 정제하여 돌연변이로 인한 안정성 변화를 예측함으로써 정확성과 신뢰성을 동시에 목표로 합니다.



### Likelihood Training of Cascaded Diffusion Models via Hierarchical Volume-preserving Maps (https://arxiv.org/abs/2501.06999)
Comments:
          Spotlight at ICLR 2024

- **What's New**: 이 논문에서는 새로운 다중 스케일 생성 모델을 제안하며, 이를 통해 고해상도 샘플을 생성할 수 있을 뿐만 아니라, 이들이 신뢰할 수 있는 likelihood 모델이 될 수 있음을 보여줍니다. 기존의 다중 스케일 모델에서 likelihood 함수 계산 시 발생하는 어려움을 해결하기 위해, 저자들은 계층적 볼륨 보존 변환(hierarchical volume-preserving maps)이라는 새로운 접근 방식을 도입하고 있습니다.

- **Technical Details**: 저자들은 각 중간 스케일이 likelihood 평가에서 간단히 제외될 수 없는 변수들을 도입한다는 문제를 해결하기 위해, latent 공간에서의 diffusion 과정을 모델링합니다. 여기서 계층적 볼륨 보존 변환을 활용하여 분산 데이터의 계층적 구조를 유지하면서도 local distortion을 피할 수 있습니다. 특히, 라플라시안 피라미드(Laplacian pyramids)와 웨이브렛 변환(wavelet transforms)이 이러한 변환의 대표적인 예로 제시됩니다.

- **Performance Highlights**: 이 연구에서 제안된 다중 스케일 likelihood 모델은 밀도 추정(density estimation), 무손실 압축(lossless compression), 및 분포 외 탐지(out-of-distribution detection)와 같은 다양한 벤치마크에서 기존의 최첨단 결과를 크게 개선하였습니다. 이 모델은 일관된 성능 향상과 함께, EMD(Earth Mover’s Distance) 기반의 이론적 검증을 통해 인간의 지각적 유사성을 반영하는 속성을 활용하고 있습니다.



### Combining LLM decision and RL action selection to improve RL policy for adaptive interventions (https://arxiv.org/abs/2501.06980)
- **What's New**: 이 논문은 강화 학습(Reinforcement Learning, RL)과 대규모 언어 모델(Large Language Models, LLMs)을 결합하여 개인 맞춤형 건강 개입을 실시간으로 업데이트하는 새로운 접근 방식을 소개합니다. 사용자의 텍스트 기반 선호를 즉시 반영하여 RL의 정책을 개선하고 개인화(personalization)를 가속화하는 것이 목표입니다. 이를 위해 연구자들은 LLM을 사용하여 RL 시스템 내에서 행동 선택(action selection)을 조정하고 있습니다.

- **Technical Details**: 제안된 하이브리드 방법은 'LLM+TS'라고 불리며, LLM의 응답을 RL 행동 선택에 통합하여 RL 정책을 개선하는 것입니다. 이 방법은 Thompson Sampling (TS) 기술을 활용하여 RL 에이전트의 역할을 수행하며, 사용자 선호를 고려한 다양한 프롬프트 전략과 행동 선택 전략을 탐색합니다. 시뮬레이션 환경 'StepCountJITAI for LLM'을 통해 사용자 선호를 생성하고, 행동 역학에 영향을 미치는 제약 조건을 모델링합니다.

- **Performance Highlights**: 이 연구의 결과, 제안된 방법은 텍스트 기반 사용자 선호를 고려하면서 RL 정책을 개선할 수 있음을 보였습니다. 이는 개인 맞춤형 건강 개입에서의 효과를 증가시키며, RL 알고리즘 개발에 대한 새로운 가능성을 열어줍니다. 또한 StepCountJITAI는 기존의 연구에서 탐구되지 않았던 혁신적인 시뮬레이션 환경으로, 후속 연구 및 응용에 있어 중요한 기초가 될 것입니다.



### Kolmogorov-Arnold Recurrent Network for Short Term Load Forecasting Across Diverse Consumers (https://arxiv.org/abs/2501.06965)
- **What's New**: 본 논문은 Kolmogorov-Arnold Recurrent Network (KARN)라는 새로운 부하 예측(load forecasting) 접근 방식을 제안합니다. KARN은 Kolmogorov-Arnold Networks의 유연성(flexibility)과 RNN의 시간적 모델링(temporal modeling) 능력을 결합하여 개발되었습니다. 이 모델은 다양한 소비자 유형에 적응할 수 있도록 설계되어 있습니다.

- **Technical Details**: KARN은 학습 가능한 시간적 스플라인 함수(learnable temporal spline functions)와 엣지 기반 활성화(edge-based activations)를 사용합니다. 이를 통해 비선형 관계(non-linear relationships)를 더 효과적으로 모델링할 수 있으며, 이는 전력 소비의 복잡하고 갑작스러운 변화를 포착하는 데 유리합니다. 다양한 실제 데이터셋의 철저한 평가를 통해 KARN의 성능을 검증하였습니다.

- **Performance Highlights**: KARN은 학생 기숙사, 단독 주택, 전기차 충전이 있는 주택, 타운하우스, 산업 건물 등 다양한 소비자 카테고리에서 전통적인 Vanilla RNNs보다 일관되게 우수한 성능을 보였습니다. 특히 6개의 건물에서는 LSTM 및 Gated Recurrent Units (GRUs)보다 뛰어난 결과를 나타냈습니다. 이러한 결과는 KARN이 다양한 에너지 관리 시나리오에서 부하 예측을 향상시키는 유망한 도구임을 입증합니다.



### Compact Bayesian Neural Networks via pruned MCMC sampling (https://arxiv.org/abs/2501.06962)
Comments:
          22 pages, 11 figures

- **What's New**: 본 연구는 MCMC 샘플링과 네트워크 프루닝(network pruning)을 결합하여 중복 파라미터를 제거한 컴팩트한 확률 모델을 만드는 방법을 제안합니다. 이 연구는 프루닝 후에도 모델의 훈련 및 일반화 성능을 유지하면서 불확실성을 추정할 수 있는 컴팩트 BNN을 보장합니다. 이를 통해 실제 애플리케이션을 위해 불확실성을 추정하는 컴팩트한 BNN 모델을 개발할 수 있는 길을 엽니다.

- **Technical Details**: 바이시안 신경망(Bayesian Neural Networks, BNNs)은 불확실성 추정에 있어 강력한 도구입니다. 본 연구는 MCMC 샘플링과 매개변수의 프루닝을 결합하여 중요한 파라미터만을 남기고 과도한 매개변수를 제거하는 방식을 사용합니다. 연구에서 사용된 MCMC 기반 프루닝 전략은 회귀 및 분류 문제에서 효과성을 평가하기 위해 선택된 벤치마크 데이터셋을 사용하여 분석되었습니다.

- **Performance Highlights**: 본 연구에서는 네트워크 크기를 75% 이상 줄이면서 일반화 성능을 유지할 수 있음을 보여줍니다. MCMC를 사용한 BNN 훈련과 프루닝이 실제 데이터셋에서 매우 효과적임을 나타내며, 복잡한 실제 데이터셋에서도 견고한 성능을 발휘합니다. 또한, 프루닝 후 성능 손실 없이는 모델을 정교화하는 방법이 논의되었습니다.



### A Hessian-informed hyperparameter optimization for differential learning ra (https://arxiv.org/abs/2501.06954)
- **What's New**: 이 논문은 여러 모델 매개변수에 대해 서로 다른 학습률을 적용하는 기법인 Differential Learning Rate (DLR)의 효율적인 접근 방식인 Hessian-informed DLR (Hi-DLR)을 제안합니다. Hi-DLR은 학습률의 하이퍼파라미터 최적화(hyperparameter optimization)를 자동으로 조정하여 모델과 옵티마이저에 적응합니다. 이를 통해 훈련 중 학습률을 동적으로 결정하여 수렴 속도를 향상시킬 수 있습니다.

- **Technical Details**: DLR은 모델 매개변수 𝒘를 K개의 그룹으로 나누고 각 그룹에 대해 다른 학습률 η(k)를 할당합니다. DLR은 복잡한 모델 구조와 작업에서 중요한 역할을 하며, 파라미터 효율적인 미세 조정(PEFT)과 같은 기법에 적용되어 측정된 성능을 보여줍니다. Hi-DLR은 매개변수 그룹의 적절한 분할을 기반으로 하여 변동하는 매개변수를 정량화하고 덜 기여하는 매개변수는 고정하여 여러 작업과 모델에 자동으로 적응할 수 있습니다.

- **Performance Highlights**: Hi-DLR은 다양한 완전 모델 훈련 작업에서 유사한 성능을 보이는 것으로 밝혀졌습니다. 다양한 준거를 사용하여 Hi-DLR의 성능 개선을 실증적으로 증명하였으며, 이는 특정 작업이나 모델에 대한 학습 성능을 최적화하는 데 도움을 줍니다. 실험 결과는 서로 다른 매개변수 그룹이 서로 다른 학습률을 필요로 하며, 모든 경우에 맞는 단일 구조가 없음을 강조합니다.



### Comparison of Autoencoders for tokenization of ASL datasets (https://arxiv.org/abs/2501.06942)
Comments:
          9 pages, 2 tables, 4 figures

- **What's New**: 이번 연구는 미국 수화(American Sign Language; ASL) 이미지 데이터셋을 위해 인코더-디코더(encoder-decoder) 아키텍처를 개발하고 평가하는 데 중점을 두고 있습니다. 해당 데이터셋은 29개의 손 동작 클래스에 대해 87,000장의 이미지를 포함하고 있으며, 입체적인 재구성을 통해 청각 장애인과 디지털 시스템 간의 효과적인 의사소통을 가능하게 하는 모델을 목표로 합니다. 다양한 인코더-디코더 기술을 비교하여 결과적으로 확산 오토인코더(Diffusion Autoencoder)가 가장 높은 성능을 발휘함을 확립했습니다.

- **Technical Details**: 연구에서는 피드포워드 오토인코더(Feedforward Autoencoder), 합성곱 오토인코더(Convolutional Autoencoder), 확산 오토인코더(Diffusion Autoencoder) 등 세 가지 기법을 사용하여 ASL 이미지 인코딩 및 디코딩을 수행했습니다. 특히 확산 오토인코더는 확률적 노이즈 모델링과 반복적인 디노이징을 통해 이미지 재구성에서 최상의 결과를 보였으며, 이는 다중 모달 AI(multi-modal AI) 적용에 중요한 역할을 합니다. 이 모든 모델은 PyTorch를 통해 텐서 형태로 처리되며, 데이터 전처리를 통해 일관성을 확보했습니다.

- **Performance Highlights**: 성능 평가에 따르면 확산 오토인코더는 평균 제곱 오차(mean squared error; MSE) 측정에서 가장 낮은 값을 기록하며, 사용자 평가에서도 가장 우수한 평균 의견 점수(Mean Opinion Score; MOS)를 달성했습니다. 이는 해당 모델이 노이즈가 있는 상황에서도 높은 품질의 이미지를 재구성할 수 있는 능력을 보유하고 있음을 나타냅니다. 연구 결과는 ASL 인식 및 생성과 같은 다중 모달 AI 분야에서 더 나은 인코더-디코더 시스템 설계에 기여할 것으로 기대됩니다.



### A group-theoretic framework for machine learning in hyperbolic spaces (https://arxiv.org/abs/2501.06934)
Comments:
          22 pages, 4 figures

- **What's New**: 이번 연구에서는 하이퍼볼릭 기하학을 이용하여 기계 학습(ML) 알고리즘의 수학적 기초를 증진하는 방법을 제안하고 있습니다. 특히, 평균(barycenter)의 개념과 하이퍼볼릭 공간에서의 새로운 확률 분포 가족을 도입하여, 최적화 및 통계 기법과 결합하고 있습니다. 이러한 접근법을 통해 하이퍼볼릭 ML의 적용 범위를 넓힐 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 하이퍼볼릭 ML의 진행을 위해 그룹 이론(group-theoretic)과 적합 기하학(conformal-geometric)을 결합하는 방법론이 사용됩니다. 연구에서는 하이퍼볼릭 구체(hyperbolic balls)에서의 평균(barycenter) 계산 및 최대 우도 추정(maximum likelihood estimation)을 위한 효율적인 최적화 알고리즘을 제안합니다. 이를 통해 하이퍼볼릭 깊은 학습 파이프라인을 설계할 수 있는 기초를 마련합니다.

- **Performance Highlights**: 하이퍼볼릭 ML의 성능 향상 가능성은 자연어 처리(NLP), 컴퓨터 비전, 추천 시스템 등 다양한 분야에서 실험 결과로 입증되고 있습니다. 특히, 고차원 데이터에 대해 극적으로 차원을 축소할 수 있는 잠재 공간(latent space) 개념이 상호작용하여 효과적인 알고리즘 설계에 기여하고 있습니다. 하이퍼볼릭 공간에서 이루어진 여러 실험들은 그 효과와 효율성을 입증하며, 앞으로의 연구 방향에 큰 영향을 미칠 것으로 기대됩니다.



### Neural equilibria for long-term prediction of nonlinear conservation laws (https://arxiv.org/abs/2501.06933)
- **What's New**: 이번 논문에서는 Neural Discrete Equilibrium (NeurDE)이라는 기계 학습(machin learning) 접근법을 소개합니다. 이 접근법은 물리적 보존 법칙을 동역학 이론(kinetic theory) 프레임워크로 'lifting'하는 데 의존하여 장기 예측(forecasting)을 가능하게 합니다. NeurDE는 비선형(non-local) 물리학을 선형 비지역(transient) 수송과 비선형이지만 지역적인 평형으로 분리함으로써 기계 학습의 초점을 맞추게 합니다.

- **Technical Details**: NeurDE에서는 거시적 관측량(macroscopic observables)을 평형 상태(equilibrium states)로 매핑(mapping)하는 연산자 네트워크(operator network)를 설계하였습니다. 이 네트워크는 엔트로피(entropy)를 극대화하는 방식으로 작동하며, 이를 통해 강력한 BGK형 충돌(collision)을 생성합니다. 또한, 우리의 대리 평형(surrogate equilibrium)을 격자 볼츠만(lattice Boltzmann) 알고리즘에 통합하여 다양한 도전적인 유동(flow)에 대한 정확한 예측을 달성합니다.

- **Performance Highlights**: NeurDE를 사용하여 압축성 흐름(compressible flows) 예측에 매우 효과적임을 확인했습니다. 특히 초음속 유동(supersonic flows) 예측에도 성공하였으며, 수백 개의 시간 단계 동안 충격(shock)을 추적할 수 있었습니다. 이는 이전까지는 비싼 수치적(root finding) 기법 없이는 이룰 수 없었던 성과입니다.



### A Hybrid Virtual Element Method and Deep Learning Approach for Solving One-Dimensional Euler-Bernoulli Beams (https://arxiv.org/abs/2501.06925)
- **What's New**: 이 논문은 Virtual Element Method (VEM)와 딥 러닝을 통합한 하이브리드 프레임워크를 제안합니다. 하나의 주요 목표는 다양한 재료 및 기하학적 매개 변수에 따른 변위장을 예측할 수 있는 데이터 기반 대체 모델을 개발하여 계산 효율성을 유지하는 것입니다. 이 연구는 구조 역학에서 VEM의 높은 유연성과 안정성을 활용하여 더 발전된 컴퓨테이셔널 메커니즘으로 나아가기 위한 기초를 다집니다.

- **Technical Details**: VEM는 비공식 다각형 요소를 사용해 복잡한 기하학을 처리할 수 있는 유연성을 제공합니다. 이 방법은 높은 차수의 다항식 및 비개수 분할을 간편하게 처리할 수 있으며, 고차 요소의 수식을 간소화하면서도 FEM 프레임워크와 원활하게 통합됩니다. 논문은 딥 러닝을 사용하여 노드 및 재료 특성 데이터를 별도로 처리하는 신경망 아키텍처를 도입하여 복잡한 상호작용을 포착하고, 극소량의 데이터로도 훈련이 가능하도록 합니다.

- **Performance Highlights**: 제안된 프레임워크는 Sobolev 훈련 및 GradNorm 기법을 활용하여 손실 기여도의 균형을 유지하면서 일반화 능력을 향상시킵니다. 비록 초기 단계의 연구이지만, 이 접근법은 전통적인 방법에 대한 확장 가능한 대안으로 발전할 잠재력을 보여주고 있습니다. 향후 구조 역학 분야의 수치 및 데이터 기반 기술 발전을 위한 기초를 제공합니다.



### Black-box optimization and quantum annealing for filtering out mislabeled training instances (https://arxiv.org/abs/2501.06916)
- **What's New**: 이번 연구는 비개인화된 훈련 데이터셋에서 잘못 레이블링된 인스턴스를 제거하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 대리 모델 기반의 블랙박스 최적화(Black-box Optimization, BBO)와 후처리(postprocessing), 그리고 양자 어닐링(Quantum Annealing)을 결합하여 잘못된 라벨 문제를 해결합니다. 특히, 기존의 방법들과는 달리, D-Wave의 클리크 샘플러를 활용하여 최적화를 가속화하며 데이터셋 품질 향상에 기여하는 점에서 주요한 차별점을 보입니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 최대한 정확한 검증 손실(validation loss) 측정을 기반으로 훈련 데이터에서 잘못 레이블링된 인스턴스를 효율적으로 제거합니다. 방법론은 BBO와 후처리(postprocessing) 기술을 결합하여, 고위험 잘못 레이블링 인스턴스를 우선적으로 제거하는 구조를 갖추고 있습니다. 또한, 양자 어닐링의 특성을 활용하여 다양하고 고품질의 훈련 하위 집합을 신속하게 샘플링하는 능력을 갖추고 있어, 조합 최적화 문제 해결에 적합합니다.

- **Performance Highlights**: 실험에서는 노이즈가 있는 다수 비트 작업(noisy majority bit task)을 통해 제안된 방법의 성능을 평가하였습니다. 제안된 접근 방식은 잘못 레이블링된 인스턴스를 효과적으로 제거하고, 솔루션 품질을 반복적으로 개선하는 데 적합합니다. D-Wave 양자 어닐러를 사용하여 더 나은 솔루션을 더 빠르게 도출하여 데이터셋 품질 향상에 기여하고, 데이터 정화의 중요성을 강조합니다.



### Deep Learning and Foundation Models for Weather Prediction: A Survey (https://arxiv.org/abs/2501.06907)
- **What's New**: 이 논문은 최근의 딥 러닝 및 기초 모델을 활용한 날씨 예측에 관한 종합적인 설문조사를 제시하고 있습니다. 다양한 훈련 패러다임에 기반하여 기존 모델을 분류하는 새로운 세분화를 제안합니다. 또한, 실세계 응용 프로그램을 탐구하고 오픈 소스 코드 및 데이터세트의 요약을 제공하여 연구 발전을 실제 구현과 연결하는 데 기여하고자 합니다.

- **Technical Details**: 물리 기반 모델을 활용한 전통적인 날씨 예측의 단점을 극복하기 위해, 데이터 기반의 머신 러닝과 딥 러닝 모델들이 점점 더 많이 적용되고 있습니다. 논문에서 다루는 주요 훈련 패러다임에는 결정론적 예측 학습, 확률적 생성 학습, 사전 훈련 및 미세 조정이 포함됩니다. 이들 모델은 복잡한 날씨 및 기후 데이터를 분석하고, 높은 정밀도와 예측의 불확실성 정량화를 달성하고 있습니다.

- **Performance Highlights**: 결정론적 모델인 Pangu와 GraphCast는 전통적인 방법과 비교하여 중기(10일) 글로벌 날씨 예측에서 최첨단 성능을 보였습니다. 최근에는 CasCast와 Gencast와 같은 확률적 생성 모델이 비구름 재현 및 날씨 예측에서 두드러진 성과를 이루어내고 있습니다. 이러한 모델들은 고품질 예측과 조정된 불확실성 추정치를 제공하는 쪽으로 진화하고 있습니다.



### Introduction to the Usage of Open Data from the Large Hadron Collider for Computer Scientists in the Context of Machine Learning (https://arxiv.org/abs/2501.06896)
Comments:
          34 pages, 22 figures (without Appendix)

- **What's New**: 최근 몇 년간 심층 학습(Deep Learning) 기술이 급속히 발전했으며, 이는 실험 입자 물리학 분야에도 지대한 영향을 미쳤습니다. 이 연구에서는 대형강입자충돌기(Large Hadron Collider, LHC)에서 수집된 데이터를 컴퓨터 과학에서 널리 사용되는 pandas DataFrame 형식으로 변환하여, 컴퓨터 과학자와 물리학자 간의 효과적인 협업을 지원하는 방법을 제안하고 있습니다. 이러한 데이터 변환은 양 분야 간 지식 교류와 협력을 촉진하겠다는 목표를 갖고 있습니다.

- **Technical Details**: LHC의 데이터는 주로 ROOT 형식으로 기록되는데, 이는 대다수의 컴퓨터 과학자들에게는 알려지지 않은 형식입니다. 본 연구에서는 이 데이터를 pandas DataFrame으로 전환하여, 머신 러닝(Machine Learning)을 연구하고자 하는 컴퓨터 과학자들이 데이터를 보다 쉽게 접근하고 활용할 수 있도록 하였습니다. 또한, LHC에서 기록하여 분석된 데이터의 주요 내용과 해석에 대한 간략한 소개가 포함되어 있습니다.

- **Performance Highlights**: 논문은 고에너지 물리학에서 널리 사용되는 데이터 형식을 제공함으로써, LHC 데이터에 대한 이해를 높이고 새로운 알고리즘의 개발을 촉진할 것으로 기대됩니다. 또한 데이터 변환 과정에서 생성된 콘텐츠는 bonndata에 공개되며, 이는 학제 간 협력을 위한 중요한 첫걸음으로 작용할 것입니다. 이러한 접근은 물리학 학생들과 컴퓨터 과학 학생들 간의 연구 효율성을 높이는 데 기여할 수 있을 것으로 보입니다.



### Transfer Learning of Tabular Data by Finetuning Large Language Models (https://arxiv.org/abs/2501.06863)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)을 활용하여 테이블 데이터 분류에서의 효과적인 전이 학습 방법을 탐구합니다. 기존의 딥러닝 기법들이 테이블 데이터에서의 성과가 미흡했던 이유는 이질적인 특성과 샘플 수의 제한으로 지적됩니다. 제안된 방법은 LLM을 최적화하여 적은 수의 샘플로도 성능을 극대화할 수 있도록 해 새로운 가능성을 열어 줍니다.

- **Technical Details**: LLM의 전이 학습은 기존의 LLM의 지식을 활용하여 더 나은 분류 성능을 제공하는 방법입니다. 본 연구에서는 데이터 직렬화(data serialization) 단계를 통해 테이블 데이터를 텍스트 프롬프트로 변환하고, DistilGPT2 모델을 사용하여 이를 분류 작업에 적용합니다. 이 과정에서 메타데이터(metadata)를 포함한 텍스트 프롬프트를 통해 LLM의 성능을 개선합니다.

- **Performance Highlights**: 제안된 LLM 최적화 방법은 기존의 머신러닝 및 딥러닝 기법에 비해 상대적으로 적은 계산 비용으로 경쟁력 있는 분류 성능을 보여줍니다. 특히, 10개의 벤치마크 데이터셋에서 특성이 10개 미만인 경우에도 뛰어난 성능을 발휘하며, 이를 통해 전이 학습의 가능성을 더욱 확장합니다.



### A General Framework for Inference-time Scaling and Steering of Diffusion Models (https://arxiv.org/abs/2501.06848)
- **What's New**: 이번 연구에서는 Feynman Kac (FK) steering을 제안합니다. 이는 사용자 지정 특성을 가진 샘플을 생성하는 데 어려움을 극복하기 위한 인퍼런스 타임 프레임워크입니다. FK steering은 여러 개의 상호작용하는 확산 프로세스인 입자(particles)를 샘플링하고 희소 이벤트 시뮬레이션 방법을 통해 높은 보상을 제공하는 샘플을 생성할 수 있도록 합니다.

- **Technical Details**: FK steering은 보상 함수에 따라 입자를 재샘플링하는 메커니즘을 가지고 있습니다. 이 과정에서, 잠재적 함수(potentials)는 중간 상태에 대한 보상을 기반으로 정의되며, 높은 값을 가진 입자는 더 높은 보상 샘플을 생성할 가능성이 높습니다. 이 프레임워크를 통해 사용자들은 다양한 잠재적 함수와 보상 모델을 선택해 성능을 조정할 수 있습니다.

- **Performance Highlights**: FK steering은 텍스트-이미지 및 텍스트 확산 모델에서 높은 성과를 보여주었습니다. 특히, 0.8B 파라미터 모델이 2.6B 파라미터로 파인 튜닝된 모델보다 프롬프트 충실도에서 우수한 성과를 보였으며, 샘플링 속도는 더 빠르고 추가 교육이 필요하지 않았습니다. FK steering은 더 작은 모델도 더 큰 모델보다 성능이 뛰어난 성과를 보여주며, 품질 제어 및 속성 조정의 용이성을 제공합니다.



### SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training (https://arxiv.org/abs/2501.06842)
- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs) 훈련에서 발생하는 그래디언트 스파이크(gradient spikes)의 문제를 심층적으로 분석합니다. 이러한 스파이크는 훈련을 방해하고 비효율성을 가중시키는 주요 원인으로 지적됩니다. 이를 해결하기 위해 'Spike-Aware Adam with Momentum Reset SPAM'이라는 새로운 최적화 알고리즘을 제안하여 그래디언트 스파이크를 효과적으로 감소시켜 훈련 안정성을 향상시킵니다.

- **Technical Details**: 저자들은 온라인 볼록 최적화 문제를 고려하며, 각 시간 단계에서 결정 변수를 선택하고 볼록 손실 함수가 주어지는 상황을 다룹니다. 각 그래디언트는 유한한 경계를 가지고 있으나, 그래디언트 스파이크는 일반적인 그래디언트 노름보다 상당히 클 수 있습니다. 점진적으로 개선된 AMSGrad 변형을 통해 훈련 안정성과 수렴성을 보장하려고 합니다.

- **Performance Highlights**: 실험 결과, SPAM은 Adam 및 그 변종에 비해 다양한 작업에서 일관되게 우수한 성능을 보였습니다. 60M에서 1B까지의 LLM Pre-training, 강화 학습, 시계열 예측 등을 포함하며, 메모리 제약 환경에서도 상태-of-아트 메모리 효율 최적화기보다 성능이 뛰어남을 입증합니다. 특히 희소 모멘텀을 통해 메모리 효율적인 훈련을 가능하게 하여 리소스 효율성을 높이는데 기여합니다.



### A novel multi-agent dynamic portfolio optimization learning system based on hierarchical deep reinforcement learning (https://arxiv.org/abs/2501.06832)
- **What's New**: 이번 연구에서는 기존의 Deep Reinforcement Learning (DRL) 알고리즘의 한계를 극복하기 위해 새로운 multi-agent Hierarchical Deep Reinforcement Learning (HDRL) 프레임워크를 제안합니다. 기존 알고리즘들은 자산 가격 변화 패턴을 충분히 학습하는 데 어려움을 겪으며, 그 결과 리스크 조정 수익률이 개선되지 않는 문제를 발견하였습니다. 이러한 문제를 해결하기 위해, 보조(agent) 에이전트를 설계하여 실행(agent) 에이전트와 협력하여 최적의 정책 탐색을 진행합니다.

- **Technical Details**: 기존의 DRL 에이전트들은 actor-critic 알고리즘과 깊은 함수 근사기(deep function approximators)를 사용하여 포트폴리오 최적화 문제를 해결합니다. 그러나 양의 보상(sparsity in positive reward) 부족과 차원의 저주(the curse of dimensionality)로 인해, 에이전트들이 훈련 환경에서 자산 가격 변화 패턴을 종합적으로 학습하는 데 제한을 받습니다. HDRL 알고리즘 프레임워크는 이러한 문제를 해결하기 위해, 보조 에이전트가 실행 에이전트와 협력하여 높은 리스크 조정 수익률을 탐색하는 정책을 집중적으로 연구할 수 있도록 합니다.

- **Performance Highlights**: 새로운 HDRL 프레임워크는 positive reward가 드문 환경에서도 학습 효율성을 높이고, 차원의 저주 문제를 극복하는 데 효과적입니다. 이를 통해 DRL 에이전트는 포트폴리오 최적화 정책을 더욱 효과적으로 탐색할 수 있으며, 리스크 조정 수익률을 개선하는 데 기여하는 것으로 나타났습니다. 최적의 정책 탐색을 통한 성과가 미약했던 기존 DRL의 한계를 보완하여 더 나은 포트폴리오 관리 솔루션을 제공하는 방향으로 나아가고 있습니다.



### MEXA-CTP: Mode Experts Cross-Attention for Clinical Trial Outcome Prediction (https://arxiv.org/abs/2501.06823)
Comments:
          Accepted and to be published in SDM2025

- **What's New**: 이번 연구에서는 임상 시험 결과 예측을 위한 새로운 접근 방법인 MEXA-CTP를 소개합니다. 이 모델은 다양한 데이터 소스를 통합하여 임상 시험의 성공 가능성을 보다 정확하게 예측할 수 있도록 돕습니다. 특히, 인간의 편향을 최소화하고, 여러 모드 간의 상호작용을 효과적으로 캡처하기 위해 경량화된 주의(attention) 기반 모듈을 사용합니다. 또한, Cauchy 손실을 최적화 과정에 적용하여 모델의 성능을 높였습니다.

- **Technical Details**: MEXA-CTP는 'mode experts'라는 특화된 모듈을 활용하여 드러났습니다. 이 모델은 약물 분자, 질병 정보, 그리고 시험 프로토콜과 같은 여러 모드의 데이터를 통합하여 패턴을 인식합니다. 또한, NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실 함수를 사용하여 학습된 표현을 더욱 정제하여, 다체널 정보 간의 상호작용을 활용하는 강력한 모델을 구현합니다.

- **Performance Highlights**: 실험 결과 MEXA-CTP는 기존의 HINT 방법과 비교하여 F1 점수에서 최대 11.3%, PR-AUC에서 12.2%, ROC-AUC에서 2.5%의 성능 향상을 보였습니다. 또한, ablation 연구를 통해 각 구성 요소의 기여도를 평가하여 모델의 예측 능력을 뒷받침했습니다. 이러한 성과들은 비용이 많이 들어가는 데이터나 복잡한 설계에 의존하지 않고도 임상 시험 결과의 정확성을 높일 수 있음을 보여줍니다.



### A Pan-cancer Classification Model using Multi-view Feature Selection Method and Ensemble Classifier (https://arxiv.org/abs/2501.06805)
Comments:
          20 pages, 5 figures, 9 tables

- **What's New**: 이 연구는 고차원 전사체 데이터에 대한 새로운 특성 선택 프레임워크를 개발하고, 두 가지 앙상블 분류기를 제안했습니다. 기존의 분류 방법론이 가지는 고유의 한계를 극복하기 위해, 전사체 데이터셋을 특성 유형별로 수직적으로 나누었습니다. Boruta 특성 선택 과정을 사용하여 최종 특성 세트를 준비하고, 두 가지 앙상블 머신러닝 모델을 구축하여 다양한 암 샘플을 정확하게 분류합니다.

- **Technical Details**: 연구에서는 10겹 교차 검증(10-fold cross-validation)을 통해 분류 성능을 강화합니다. Boruta 접근법을 반복하여 최적의 특성 세트를 수집하고, 최종적으로 로지스틱 회귀(LR), 서포트 벡터 머신(SVM), 그리고 XGBoost 기반의 앙상블 모형을 사용하여 33종 암을 분류합니다. 결과적으로 97.11%의 정확도와 0.9996 AUC 값을 달성하여 기존의 최첨단 방법들과 비교하여 우수한 성능을 보였습니다.

- **Performance Highlights**: 연구에서 제안한 방법은 12개 유형의 암 샘플에서 90% 이상의 식별 정확도를 보였으며, 이는 기존 문헌에서 소개된 모든 방법보다 높은 성능을 기록했습니다. 선택된 유전자 세트의 풍부함을 분석한 결과, 암과 높은 연관성을 가진 경로들이 강조되었습니다. 이러한 성과는 암 개발에 매우 관련 있는 특성 선택 방법론을 기반으로 하여, 정확한 암 샘플 식별을 가능하게 합니다.



### Pareto Set Learning for Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2501.06773)
Comments:
          AAAI 2025 Accept

- **What's New**: 이 논문에서는 Multi-Objective RL (MORL)을 위한 새로운 접근 방식인 Pareto Set Learning for MORL (PSL-MORL)을 제안합니다. 본 연구는 hypernetwork를 활용하여 각 분해 가중치에 대한 정책 네트워크의 파라미터를 생성하고, 각 선호도에 따라 개인화된 솔루션을 생성할 수 있는 효율적인 방식을 개발했습니다. PSL-MORL은 어떤 RL 알고리즘과도 호환 가능한 일반적인 프레임워크로서, 이전의 MORL 방법들이 겪었던 여러 제약을 극복합니다.

- **Technical Details**: PSL-MORL은 다중 목적 마르코프 결정 프로세스(MOMDP)의 맥락 안에서 Pareto Set Learning (PSL)을 통합하여 모든 선호도 공간을 포괄하는 정책을 생성합니다. 이 방법론은 hypernetwork를 사용하여 각 가중치에 대해 정책 파라미터를 동적으로 생성하며, 이는 개인화된 정책 네트워크를 효과적으로 제공합니다. 또한, Rademacher 복잡성과 Banach 고정점 정리를 활용하여 PSL-MORL의 이론적 우수성을 보장합니다.

- **Performance Highlights**: 실험 결과, PSL-MORL은 다양한 벤치마크에서 기존의 최첨단 MORL 방법들을 초월하는 성능을 보여주었습니다. 특히, hypervolume과 sparsity 지표에서 우수한 성과를 기록하여 정책 네트워크의 최적성을 입증했습니다. 이를 통해 PSL-MORL은 복잡한 다중 목표 문제를 처리하는 효과적인 방법임을 입증하였습니다.



### MTPareto: A MultiModal Targeted Pareto Framework for Fake News Detection (https://arxiv.org/abs/2501.06764)
- **What's New**: 이번 논문에서는 다중 모달(fake news) 검출의 필요성을 강조하며, Multimodal Targeted Pareto (MTPareto) 프레임워크를 제안합니다. 이 프레임워크는 bimodal fusion의 한계를 극복하고, 모달 간의 최적화 충돌을 해결하는 것을 목적으로 합니다. TPareto 최적화 알고리즘을 통해 세 가지 융합 레벨을 정의하고, 각 레벨에서 손실을 상정하여 모든 모달 통합을 최적화합니다.

- **Technical Details**: MTPareto 프레임워크는 다중 모달 정보를 효과적으로 융합하기 위해 계층적 융합 네트워크를 설계하였습니다. 이 구조는 BERT, VGG19, Wav2Vec와 같은 사전 학습된 모델을 사용하여 텍스트, 이미지, 오디오 피쳐를 추출하고, 크로스 엔트로피 손실함수를 통해 학습합니다. 특히, 두 개의 스트림을 이용한 크로스 어텐션과 같은 다중 헤드 어텐션 기법을 활용하여 비디오 내의 여러 모달 정보를 보다 정교하게 융합합니다.

- **Performance Highlights**: FakeSV와 FVC 데이터셋에서의 실험 결과, MTPareto 프레임워크는 기존 방법들보다 뛰어난 성능을 보여줬습니다. TPareto 알고리즘은 각각 2.40% 및 1.89%의 정확도 향상을 달성하며, 모든 융합 레벨에서 성능 개선이 이루어졌음을 보여주었습니다. 이러한 결과들은 MTPareto의 효용성을 입증하며, 다중 모달 효과적인 통합 가능성을 제시합니다.



### Procedural Fairness and Its Relationship with Distributive Fairness in Machine Learning (https://arxiv.org/abs/2501.06753)
Comments:
          33 pages, 11 figures

- **What's New**: 본 논문은 기계학습(ML) 모델의 훈련 과정에서 절차적 공정성(procedural fairness)을 달성하기 위한 새로운 방법을 제안합니다. 기존의 연구가 분배적 공정성(distributive fairness)에 주로 집중했던 데 반해, 이 논문은 절차적 공정성이 ML 모델에 미치는 영향과 두 가지 공정성 측정 기준을 최적화하는 차이를 분석합니다. 실험을 통해 제안된 방법의 효과도 검증했습니다.

- **Technical Details**: 절차적 공정성을 평가하기 위한 새로운 지표인 GPFFAE(group procedural fairness feature attribution explanation)를 제안하며, 이 지표는 ML 모델의 의사결정 과정을 설명하는 데 사용됩니다. 논문은 ML 모델의 공정성을 향상시키기 위한 방법을 제안하고, 데이터 세트 편향과 모델의 절차적 공정성이 분배적 공정성에 미치는 영향을 분석합니다. 또한, 절차적 공정성을 최적화함으로써 불공정한 판단으로 인해 발생하는 편향을 완화할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면, 데이터 세트가 편향되지 않으며 ML 모델이 절차적으로 공정할 경우, 분배적 공정성도 확보됩니다. 반대로 데이터 세트 편향이나 모델에서의 절차적 불공정성이 존재하면 분배적 불공정성이 발생하며, 절차적 공정성이 분배적 공정성에 미치는 영향이 더 강하게 나타납니다. 이 연구는 이해관계자들이 적절한 공정성 메트릭과 방법론을 선택하는 데 유용한 지침을 제공합니다.



### ZOQO: Zero-Order Quantized Optimization (https://arxiv.org/abs/2501.06736)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문은 자원이 제한된 환경에서 훈련할 수 있도록 설계된 제로 순서 양자화 최적화(ZOQO) 방법을 제안합니다. 기존의 고정밀도 그래디언트 계산 없이 양자화된 파라미터와 연산에 대해 모델을 훈련할 수 있습니다. ZOQO는 제로 순서 근사를 활용하여 파라미터의 양자화를 유지하면서 학습 과정을 조정합니다. 이 접근 방식은 성능 상의 제약이 있음에도 불구하고 경쟁력 있는 결과를 달성합니다.

- **Technical Details**: ZOQO는 Zero-Sign Stochastic Gradient Descent (ZO-SignSGD) 방법을 사용하여 기초적인 최적화 절차를 수행합니다. 이 방식에서 우리는 파라미터에 양자화된 노이즈를 삽입하고, 학습률을 양자화 규모에 맞추어 조정합니다. 모든 업데이트 과정에서 양자화된 포맷으로 진행되어 메모리 사용을 최소화합니다. 논문에서는 적대적 공격과 대형 언어 모델의 미세 조정을 통해 기법의 유효성을 검증하였습니다.

- **Performance Highlights**: ZOQO는 양자화된 환경에서 훈련된 모델이 고정밀도 공격에 비해 실패율이 미미하게 감소한다는 것을 입증했습니다. 또한, 리소스 제한 조건에서 감정 분석을 위한 대형 언어 모델을 미세 조정하는 데 있어 비약적인 성능을 발휘할 수 있음을 demonstrated하였습니다. 이러한 결과는 ZOQO의 가능성을 강조하며 자원이 제한된 환경에서 모델 훈련 메커니즘을 가능하게 하는 데 기여할 수 있습니다.



### DRDT3: Diffusion-Refined Decision Test-Time Training Mod (https://arxiv.org/abs/2501.06718)
- **What's New**: 이 연구에서는 Conditional Generative Modeling을 활용하여 최적이 아닌 보상 레이블이 있는 경로에서 더 나은 학습이 가능하도록 하고자 합니다. 기존의 Decision Transformer (DT) 방식의 한계를 극복하기 위해 Test-Time Training (TTT) 층을 도입하여 경로를 더 효과적으로 모델링합니다. 최종적으로, Diffusion-Refined Decision TTT (DRDT3)라는 새로운 프레임워크를 제안하여 기존 DT 모델들보다 뛰어난 성능을 발휘하고자 합니다.

- **Technical Details**: 이 프레임워크는 DT3 모듈을 기반으로 하여 Self-Attention 기법과 TTT 층의 이점을 결합합니다. 이 모듈은 최근 맥락을 바탕으로 조치 예측을 수행하며, 이를 통해 Denoising Diffusion Probabilistic Model (DDPM)과 통합됩니다. 통합 최적화 목표를 통해 DT3 모듈은 최적 행동 분포에 가까운 조치 조건을 생성하고, Diffusion 모델은 데이터셋 분포 내에서 행동을 샘플링하도록 제약을 받습니다.

- **Performance Highlights**: 실험 결과, DRDT3 모델은 D4RL 벤치마크에서 Gym과 AntMaze의 다양한 과제에 대해 기존의 DT 모델보다 향상된 성능을 보여줍니다. 특히, DT3는 확산 정제가 없는 상태에서도 DT보다 개선된 결과를 달성합니다. 이러한 결과는 DRDT3가 오프라인 RL 및 DT 기반 방법들에 비해 우수한 성능을 발휘함을 입증합니다.



### Evaluating Sample Utility for Data Selection by Mimicking Model Weights (https://arxiv.org/abs/2501.06708)
- **What's New**: 최근 데이터 선택 기술들은 다양한 한계가 있었으나, 새로운 접근 방식인 Mimic Score는 사전 훈련된 모델을 사용하여 데이터 샘플의 유용성을 평가합니다. 이는 다른 데이터베이스에 접근할 필요 없이 특정 샘플이 훈련에 미치는 영향을 효율적으로 측정할 수 있게 합니다. Mimic Score를 통해 모델 훈련 중의 샘플 우선 순위를 자동화하여 모델 성능을 향상시키는 Grad-Mimic 프레임워크를 개발하였습니다.

- **Technical Details**: Mimic Score는 새로운 모델의 파라미터 기울기와 참조 모델 방향 간의 정렬을 기반으로 합니다. Grad-Mimic은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 훈련 중에 샘플의 우선 순위를 정하고, 두 번째 단계에서는 학습 단계 전반에 걸쳐 샘플 유용성을 평가합니다. 이 과정은 데이터 선택을 자동화하기 위한 필터 단계를 통해 진행됩니다.

- **Performance Highlights**: Empirical results에 따르면 Grad-Mimic은 6개의 이미지 데이터셋에서 일관된 성능 향상을 보여 주었으며, CLIP 모델의 성능을 더욱 향상시키는 데 기여했습니다. 또한, 기존 필터링 방식보다 더 정교하게 낮은 가치의 샘플을 제거하여 모델 성능을 높이는 데 효과적입니다. Mimic Score와 그 필터는 데이터셋 품질을 정확하게 평가하여 기존 필터링 전략을 개선할 수 있음을 보여줍니다.



### Tab-Shapley: Identifying Top-k Tabular Data Quality Insights (https://arxiv.org/abs/2501.06685)
Comments:
          Accepted at AAAI-25

- **What's New**: 이 논문에서는 탭 형식의 데이터셋에서 이상치를 집계하기 위한 비지도학습 방법을 제안합니다. 이 방법은 'top-k' 데이터를 통해 품질 인사이트를 도출하며, 이는 사용자에게 이상치의 증거가 되는 기록의 하위 집합으로 구성됩니다. Tab-Shapley라는 협력 게임 이론 기반의 프레임워크를 도입하여 데이터의 비정상적인 특성을 생성하는 속성의 기여도를 정량화합니다.

- **Technical Details**: Tab-Shapley 프레임워크는 비지도 학습 환경에서 이상치 분석의 효율성을 높이기 위해 Shapley 값을 계산하는 닫힌 형 솔루션을 제공합니다. 이 접근법은 속성과 레코드 간의 복잡한 의존성을 고려하며, 조합 검색 공간을 탐색하여 이상치의 원본을 식별하는 데 도움을 줍니다. 이 방법은 또한 각 속성과 레코드에 대한 증거 집합을 정의하고, 이상행동이 관찰되는 레코드의 하위 집합을 제공합니다.

- **Performance Highlights**: 실제 탭 형 데이터셋에 대한 실증 분석을 통해 제안된 방법의 효과를 검증했습니다. 그 결과, Tab-Shapley 기반의 접근 방식은 비정상적인 유무를 명확히 알려주고, 데이터 품질 향상에 기여하는 주요 속성을 효과적으로 파악할 수 있게 해줍니다. 전반적으로 이 연구는 데이터 이상 감지 및 품질 통찰을 위한 새로운 장을 열어줄 것으로 기대됩니다.



### Challenging reaction prediction models to generalize to novel chemistry (https://arxiv.org/abs/2501.06669)
- **What's New**: 이 연구에서는 유기 반응 결과 예측을 위한 딥 러닝 모델이 벤치마크 테스트에서 높은 성능을 보이지만, 실제로는 잘못된 예측이 발생함을 지적하고 있습니다. 기존 벤치마크는 in-distribution 설정에서 모델을 평가하지만, 실제 상황에서의 앱리케이션은 out-of-distribution 상황에서의 일반화 및 초과 예측이 요구됩니다. 이 논문은 기존 SMILES 기반 딥 러닝 모델의 기능과 한계를 평가하기 위한 새로운 평가 과제를 개발했습니다.

- **Technical Details**: 기존 reaction predictors는 통상적으로 랜덤 샘플링 방식으로 데이터셋을 나누어 훈련, 검증, 테스트를 진행합니다. 그러나 이러한 방법은 상관관계가 있는 반응들이 훈련 세트 및 테스트 세트에 흩어져 있는 경우, 모델이 테스트 예측 시 훈련 데이터와 유사한 반응을 이용해 부당한 성과를 나타낼 수 있음을 보여줍니다. 연구에서는 세 가지 다른 데이터셋 분할 전략—(1) 반응 기반, (2) 문서 기반, (3) 저자 기반—을 통해 이러한 구조가 정확도에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 기존의 랜덤 분할 방식에 비해 문서 기반 및 저자 기반의 분할 방식이 모델 평가에서 더 현실적인 수치를 제공한다는 결과를 얻었습니다. 랜덤 분할을 사용할 경우 모델의 top-1 정확도가 65%로 높지만, 문서 기반 분할에서는 58%로 하락하고, 저자 기반에서는 55%로 더욱 낮아집니다. 이러한 결과는 현실적인 응용을 위해서는 모델이 더욱 다양한 반응과 어떻게 일반화 및 초과 예측을 수행할 수 있는지를 이해하는 것이 필요함을 나타냅니다.



### Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization (https://arxiv.org/abs/2501.06663)
- **What's New**: 이 논문은 저자들이 저메모리 및 저전력 조건 하에 테서(tensor) 압축을 활용하여 FPGA에서 엔드 투 엔드(transformer training)를 수행하는 첫 번째 가속기를 개발했다고 보고합니다. 특히, 디지털 엣지 기기에서 딥러닝 모델 훈련의 중요성이 증가함에 따라, 새로운 알고리즘적 방법과 하드웨어 설계를 통해 인프라스트럭처의 효율성을 극대화하였습니다.

- **Technical Details**: 본 논문에서 소개된 새로운 알고리즘은 양방향 텐서 수축 흐름(bidirectional tensor contraction flow)을 도입하여, 기존의 텐서 연산보다 연산 효율성을 크게 향상시키고, 메모리 비용을 절감하는 데 기여합니다. 또한 각 훈련 단계에서 모든 압축된 모델 매개변수와 기울기(gradient) 정보를 칩 내 저장하는 메모리 관리 전략을 개발하여, Off-chip 통신을 최소화함으로써 지연(latency)과 에너지 비용을 줄였습니다.

- **Performance Highlights**: 실험에서는 36.7MB에서 93.5MB 크기의 transformer 모델이 사용되었으며, AMD Alveo U50 FPGA에서 단일 배치의 온디바이스 훈련이 가능함을 입증하였습니다. 특히, 기존 NVIDIA RTX 3090 GPU에서의 훈련과 비교하여 메모리 소모가 30배에서 51배 줄어들었으며, 에너지 비용 또한 최대 3.6배 감소한 결과를 나타내어 저자들이 제안한 FPGA 가속기의 장점을 실증적으로 보여주었습니다.



### Learning dynamical systems with hit-and-run random feature maps (https://arxiv.org/abs/2501.06661)
- **What's New**: 본 논문에서는 랜덤 피처 맵(Random Feature Maps, RFM)을 사용하여 동적 시스템을 예측하는 방법을 제안합니다. 이 과정에서 tanh 활성화 함수와 데이터 주도 방식으로 선택된 내부 가중치를 활용하여 비선형, 비포화 지역을 탐색합니다. 또한, 여러 유닛을 결합하여 딥 버전의 랜덤 피처 맵을 구성하고, 지역화를 통해 차원의 저주를 완화하는 방법을 제안합니다.

- **Technical Details**: RFM은 고정된 내부 가중치(weights)와 편향(bias)을 사용하여 단일 레이어 피드포워드 네트워크로 볼 수 있으며, 외부 가중치는 최소 제곱 회귀를 통해 결정됩니다. 개선된 RFM은 잔여 네트워크(residual networks)에서 사용되는 스킵 연결(skip connections)을 포함하여, 벡터 필드를 추정하는 방식으로 학습 문제를 구성합니다. 이러한 방식은 고차원 데이터의 예측과 통계적 속성을 효과적으로 처리하는 데 유용합니다.

- **Performance Highlights**: 하나의 하이퍼파라미터(hyperparameter)만 조정하여 시뮬레이션 결과와 비슷한 수준의 예측 능력을 달성할 수 있습니다. RFM은 최대 512차원의 혼돈 동적 시스템에 대해 단일 궤적 예측과 장기 통계적 특성 추정에서 우수한 성능을 보입니다. 본 연구에서는 로렌츠-63 모델과 로렌츠-96 모델, 쿠라마토-시바신스키 방정식 등 세 가지 기준 시스템을 통해 RFM의 다양한 수정 사항들이 강화된 예측 성능을 세밀하게 평가했습니다.



### Personalized Preference Fine-tuning of Diffusion Models (https://arxiv.org/abs/2501.06655)
- **What's New**: 이 논문에서는 기존의 RLHF (Reinforcement Learning from Human Feedback) 기법들로 인해 개인화된 선호를 고려하지 못하는 문제를 해결하기 위해 PPD (Personalized Preference Diffusion)를 소개합니다. PPD는 다중 보상 최적화 목적을 통해 다수의 사용자의 개별 선호에 맞춰 확산 모델을 조정할 수 있습니다. 이를 통해 사용자가 제공하는 최소한의 샘플로도 성능을 극대화하며, 새로운 사용자에게 일반화할 수 있는 능력을 갖추게 됩니다.

- **Technical Details**: PPD는 비전-언어 모델(Vision-Language Model, VLM)을 활용하여 사용자 개별의 선호 임베딩을 몇 가지 쌍별 선호 예시에서 추출합니다. 이후 이 임베딩을 확산 모델에 통합하여 크로스 어텐션(Cross Attention) 기법을 통해 모델을 미세 조정합니다. DPO의 변형을 채택하여 사용자 선호에 따라 텍스트-이미지 모델을 최적화함으로써, 여러 사용자 선호와 일치하도록 훈련됩니다.

- **Performance Highlights**: PPD의 실험 결과는 여러 보상 함수(CLP, Aesthetic 등)에 대해 효과적으로 최적화하며, 추론 중 보상 사이의 원활한 보간(interpolation)을 보여줍니다. 실제 사용자 시나리오에서 PPD는 새로운 사용자로부터 제공된 최소 4개의 선호 예시로도 그 사용자의 특정 선호에 맞춘 이미지를 생성하며, Stable Cascade에 비해 평균 76%의 승률을 기록하는 성과를 보였습니다.



### Dual-Modality Representation Learning for Molecular Property Prediction (https://arxiv.org/abs/2501.06608)
- **What's New**: 이번 논문에서는 Dual-Modality Cross-Attention (DMCA)라는 새로운 방법을 제안하였습니다. 이 방법은 그래프와 SMILES 표현을 결합해 이들 각각의 강점을 효과적으로 활용할 수 있도록 설계되었습니다. DMCA는 최근 깊이 있는 학습(Dense Learning) 기반의 다중 모달리티(Multi-Modality) 학습 기술을 사용해 최고의 성능을 달성하며, 기준 성능을 초월하는 결과를 얻었습니다.

- **Technical Details**: DMCA 모델은 GNN 기반 인코더와 Transformer 기반 인코더가 두 가지 모달리티의 분자 표현을 학습하는 구조를 가지고 있습니다. 이후 Transformer 기반 다중 모달 인코더가 서로 다른 모달리티에서 학습된 특성을 융합하여 화합물의 특성을 예측하는 다층 퍼셉트론(MLP) 네트워크에 전달됩니다. 이 계층화된 네트워크 모델은 MoleculeNet 벤치마크의 8개 데이터 셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, DMCA는 최고 성능을 기록하여 그래프와 SMILES 모달리티로부터의 보완 정보를 효과적으로 활용할 수 있는 방법임을 입증하였습니다. 본 연구에서 제안하는 방법은 향후 약물 발견 및 분자 속성 예측 분야에서 강력한 도구로 자리 잡을 것으로 기대됩니다.



### Preconditioned Sharpness-Aware Minimization: Unifying Analysis and a Novel Learning Algorithm (https://arxiv.org/abs/2501.06603)
Comments:
          Accepted by International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2025

- **What's New**: 본 연구는 sharpness-aware minimization(SAM)의 다양한 변형을 통합하고 수학적 근거를 제공하는 새로운 알고리즘인 infoSAM을 소개합니다. SAM은 손실 경량에서 ‘평탄한’ 지역을 목표로 하여 딥러닝의 일반화 능력을 개선합니다. 기존의 SAM 변형들이 통합된 프레임워크를 통해 이론적 분석과 개선된 알고리즘 설계를 지원하게 됩니다.

- **Technical Details**: 연구는 first-order Taylor expansion을 사용해 손실의 미니맥스 최적화 문제를 근사화하고, 기존 SAM 변형들을 전처리(preconditioning) 방법으로 통합한 preSAM을 제안합니다. 이 방식은 관계 없는 데이터 특성이 아닌, 데이터와 모델의 속성을 함께 고려하여 최적화하는 새로운 방향을 제시합니다. 이 과정에서 gradient noise를 조정하여 adversarial model degradation(AMD) 문제를 해결하는 infoSAM을 제안합니다.

- **Performance Highlights**: Numerical tests에서 infoSAM은 다양한 벤치마크에서 일반화 능력을 향상시킬 수 있음을 보여주었습니다. 이는 SAM의 한계를 극복하고, 특히 복잡한 모델에 적합한 방법임을 입증합니다. infoSAM의 도입으로 데이터의 손실 경량을 더 효과적으로 탐색할 수 있는 가능성이 증명되었습니다.



### EmoXpt: Analyzing Emotional Variances in Human Comments and LLM-Generated Responses (https://arxiv.org/abs/2501.06597)
Comments:
          7 pages, 10 figures, 5 tables. This paper has been accepted and presented at the 2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC)

- **What's New**: 이 연구는 생성적 AI, 특히 ChatGPT와 관련된 감정 동태를 검토하고, EmoXpt라는 새로운 감정 분석 프레임워크를 소개합니다. 이 프레임워크는 인간의 감정과 ChatGPT의 반응에서 감정 표현을 정량적으로 평가하여, 과거 연구와는 달리 AI의 감정 지능을 분석합니다. 실험 결과, LLM(large language model) 생성 응답은 인간 반응보다 일관되게 긍정적이고 효율적이라는 점을 강조합니다.

- **Technical Details**: EmoXpt는 감정 분석을 위한 네 가지 주요 단계, 즉 데이터 수집, 탐색적 데이터 분석(Exploratory Data Analysis, EDA), 데이터 전처리 및 데이터 모델링을 수행합니다. 데이터는 2023년 3월 7일부터 4월 29일까지 X(구 Twitter)에서 수집된 512개의 트윗과 대응하는 ChatGPT의 반응 및 사용자 코멘트를 포함합니다. 연구에서 BERT와 K-means 알고리즘을 사용하여 감정 분석을 진행하며, 이는 높은 품질의 자연어 처리(NLP) 응용 프로그램을 지원합니다.

- **Performance Highlights**: 결과 분석에 따르면, ChatGPT의 응답은 자주 사용되는 단어와 사용자의 코멘트 분석을 바탕으로 매우 높은 응답 품질을 유지하고 있습니다. 특히, LLM 생성 응답은 일반적으로 더 응집성 있고 긍정적인 감정을 나타내며, 이는 생성적 AI가 인간의 의사소통을 지원하는 데 핵심적인 역할을 한다는 것을 시사합니다. 이러한 발견은 감정 지능을 개선한 대화형 에이전트의 개발에 기여할 것으로 기대됩니다.



### Ladder-residual: parallelism-aware architecture for accelerating large model inference with communication overlapping (https://arxiv.org/abs/2501.06589)
- **What's New**: 이번 논문에서는 대형 언어 모델의 추론에서 메모리 및 시간이 많이 소모되는 문제를 해결하기 위해 Ladder Residual이라는 간단한 아키텍처 수정을 제안합니다. 이는 모든 잔여 기반 모델에 적용 가능하며, 통신의 지연을 효과적으로 숨길 수 있도록 돕습니다. 이 방법은 기존의 모델 아키텍처를 변경하여 계산과 통신을 분리할 수 있는 가능성을 열어줍니다. 결과적으로, 70B 파라미터의 Transformer 모델에 Ladder Residual을 적용하면 8개의 장치에서 30%의 속도 향상을 이룰 수 있음을 보여줍니다.

- **Technical Details**: Ladder Residual은 Transformer 아키텍처에서 모델 레이어 간의 잔여 스트림을 재조정하여 통신과 계산을 분리합니다. 이 분리는 다양한 장치 간의 데이터 교환 중 발생하는 지연을 최소화하며, 이를 통해 추론 과정의 속도가 개선됩니다. 이 방법은 Tensor Parallelism에서 특히 효과적이며, 논문에서는 이를 중심으로 두 가지 모델(1B 및 3B Ladder Transformer)을 훈련시켰습니다. Ladder Residual은 고급 기계 학습 프레임워크인 PyTorch나 JAX에서도 쉽게 적용할 수 있습니다.

- **Performance Highlights**: Ladder Residual을 적용한 70B Transformer 모델은 8개 장치에서 추론 시 속도가 약 30% 향상되었습니다. 이를 통해 잔여 기반 모델의 성능을 유지하면서 통신 지연을 숨기는 데 기여할 수 있음을 보여줍니다. 훈련될 때도 다른 형태의 병렬 처리에 대한 속도 향상이 이루어지며, 실험 결과 기존의 변환 모델의 성능과 비슷한 결과를 보여줍니다. 이 모델은 Transformer와 같은 대형 언어 모델에 널리 활용될 수 있을 것입니다.



### Boundary-enhanced time series data imputation with long-term dependency diffusion models (https://arxiv.org/abs/2501.06585)
Comments:
          Accepted by Knowledge-Based Systems

- **What's New**: 이번 연구에서는 다변량 시계열 데이터의 결측값 예측을 위한 새로운 방법론, 즉 Diffusion-based time Series Data Imputation (DSDI) 프레임워크를 제안합니다. 이 프레임워크에서는 결측값 예측을 위해 무게 감소 주입 전략을 도입하여 경계 비일치 문제를 완화합니다. 또한, 다층 S4 기반 U-Net을 통해 다양한 해상도의 정보를 통합하여 장기 종속성(long-range dependencies)을 효과적으로 캡처합니다.

- **Technical Details**: DSDI 프레임워크는 결측점의 예측값을 반전 확산 과정에 적응적으로 주입하여 경계 문제를 해결합니다. 예측값은 선형 자기회귀(AR) 모델을 통해 초기 단계에서 더 높은 비율로 주입되어 정확성을 높이며, 반전 확산이 진행됨에 따라 주입 비율이 점진적으로 감소합니다. 이와 함께, S4 모델을 통합한 U-Net 구조를 사용하여 다차원 시계열 데이터의 장기 종속성을 효과적으로 모형화합니다.

- **Performance Highlights**: 실험 결과 DSDI 모델은 기존의 여러 결측값 보완 방법들과 비교하여 우수한 성능을 보였습니다. 특히, 다변량 시계열 데이터의 경우 결측값의 경계 부정확성을 해소하고, 장기적 의존성을 잘 포착하여 정확도를 높였습니다. 이러한 접근 방식은 의료, 교통, 경제 등 다양한 분야에서 결측값 처리의 가능성을 더욱 확장할 것입니다.



### Recommending the right academic programs: An interest mining approach using BERTopic (https://arxiv.org/abs/2501.06581)
Comments:
          Accepted at Data Mining and Knowledge Discovery (Springer)

- **What's New**: 본 논문은 학생들이 개인적인 선호와 프로그램 내용을 기반으로 효율적인 추천을 받을 수 있는 최초의 정보 시스템을 제안합니다. BERTopic 알고리즘을 활용하여 모든 강좌 설명에서 흥미로운 주제를 추출하고, 이를 통해 각 프로그램의 지식을 포괄적으로 나타냅니다. 학생의 선택한 주제를 바탕으로 가장 적절한 프로그램 목록을 계산하는 독창적인 방법론을 개발하였습니다.

- **Technical Details**: 이 시스템은 비지도 머신러닝 기법인 topic modeling을 사용하여 관련 키워드의 관리 가능한 집합을 탐색하고, 이를 통해 학생들이 관심 있는 주제를 탐색할 수 있는 종합적인 세트를 제공합니다. 옵션으로 주어진 다중 관심 주제를 통해 프로그램 관련성 점수를 계산하고, 최종적으로 가장 관련성이 높은 전공을 나열한 순위를 생성합니다. 시스템은 Python 프로시저와 공개된 프로그램 데이터를 활용하여 자동화된 방식으로 즉각적인 의사 결정을 지원합니다.

- **Performance Highlights**: 사례 연구에서 이 시스템은 80개 프로그램과 5,000개 이상의 강좌를 통해 신속하고 효과적인 의사 결정을 지원하는 것으로 나타났습니다. 65명의 학생을 대상으로 한 질적 연구 결과, 98% 이상의 사용자가 추천이 본인의 관심사와 일치한다고 응답하였고, 약 94%는 향후 도구를 사용할 것이라고 밝혔습니다. 정량적 분석에서는 공정성을 보장하며 98%의 프로그램 커버리지를 달성했음을 보여줘, 데이터 중심의 실시간 사용자 중심 시스템이 전공 선택 과정을 개선할 수 있음을 시사합니다.



### Active Rule Mining for Multivariate Anomaly Detection in Radio Access Networks (https://arxiv.org/abs/2501.06571)
- **What's New**: 이 논문에서는 다변량(anomaly detection) 이상 탐지 시스템의 필요성을 해결하기 위해 반자율(anomaly rule miner) 이상 규칙 발굴기를 제안한다. 이번 연구는 어떤 이상(anomaly)이 특이한지에 대한 설명이 필요하다는 점을 강조하며, 이를 통해 네트워크 운영자들이 실제 상황에서 발생하는 문제를 이해하고 대응할 수 있도록 돕고자 한다. 실험은 시간 시계열(time series) RAN 데이터를 통해 진행되었으며, 주요 특징은 규칙 매핑을 통해 이상 탐지 결과에 대한 행동 가능성을 제시한다.

- **Technical Details**: 제안한 방법론은 이산치(discrete) 및 시계열 데이터(time series data) 모두에 적용 가능하며, 기존의 이상 탐지 방법의 제약 없이 사용할 수 있다. 이 시스템은 이상을 설명할 뿐만 아니라 이를 바탕으로 actionable rules(행동 가능한 규칙)을 수립할 수 있는 방법도 포함되어 있다. 또한, 제안 시스템은 정적(threshold) 기준이 아닌, 도메인 요구에 따라 발전할 수 있는 규칙 관리 방법을 포함하고 있어 개념 변화(concept drift)에 능동적으로 대응할 수 있다.

- **Performance Highlights**: 이 시스템을 사용한 결과, 2400개의 이상을 126개의 조건으로 그룹화하여 관리를 용이하게 만들었으며, 운영자들은 각 조건의 샘플 이상을 평가하고 검증하기가 훨씬 쉬워졌다. 이를 통해 운영 효율성이 크게 개선되었고, 사람이 이상을 검증하는 데 소요되는 시간이 절약되어 인적 오류를 줄였다. 제안된 방법은 비즈니스에 적합한 규칙을 미리 마련할 수 있는 기반을 마련하였다.



### Hierarchical Reinforcement Learning for Optimal Agent Grouping in Cooperative Systems (https://arxiv.org/abs/2501.06554)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문은 다중 에이전트 시스템에서 에이전트 그룹화 및 페어링 문제를 해결하기 위한 계층적 강화 학습(Hierarchical Reinforcement Learning) 접근 방식을 제시합니다. 본 연구의 목표는 최적의 그룹화 및 에이전트 정책을 동시에 학습하는 것입니다. 계층적 RL 프레임워크를 통해 그룹화의 고차원 결정과 저차원 에이전트 행동을 구분하여 효율적인 학습과 확장성을 보장합니다.

- **Technical Details**: 이 방법은 CTDE(Centralized Training with Decentralized Execution) 패러다임을 활용하여 에이전트 간의 동질성과 협력을 처리할 수 있도록 순열 불변(neural networks) 신경망을 사용하는 것을 포함합니다. 옵션-비평가(option-critic) 알고리즘이 계층적 의사결정 과정을 관리하기 위해 조정되어, 동적이고 최적의 정책 조정을 가능하게 합니다. 또한 에이전트 조합 정보를 네트워크 아키텍처에 통합하여 팀 조합의 순열 불변성을 보장하고 모델링 효율성을 높입니다.

- **Performance Highlights**: 이 접근 방식은 기존의 전통적인 Q-learning 또는 옵션-비평가 접근법보다 더 적은 계산 복잡도로 최적 팀 페어링 문제를 재구성합니다. 기본적으로, 제안된 아키텍처는 차원 축소 문제를 피하고 팀 수가 증가함에 따라 정책 매개변수 공간의 기하급수적 성장을 방지합니다. 이를 통해 대규모 문제에 대한 확장성을 확보하며, 복잡한 페어링 및 그룹화 문제에 대한 해결책을 제공합니다.



### Online Algorithm for Aggregating Experts' Predictions with Unbounded Quadratic Loss (https://arxiv.org/abs/2501.06505)
- **What's New**: 이번 논문에서는 전문가 예측을 온라인으로 집계하는 문제를 다룹니다. 기존의 방법들과는 달리, 손실의 상한(upper bound)에 대한 사전 지식이 필요 없는 알고리즘을 제안하였습니다. 이 알고리즘은 전문가 손실의 지수적 재가중치(exponential reweighing)를 기반으로 하고 있습니다.

- **Technical Details**: 제안된 알고리즘은 전문가의 예측을 집계하기 위해 손실 함수로서 제곱 손실(quadratic loss function)을 사용합니다. 이 알고리즘은 손실에 대한 정보를 수집하며, 무작위성(randomness)을 통해 각 전문가의 정확성을 반영하여 예측을 개선합니다. 이는 experts의 다양성(diversity)을 최대한 활용하고 손실을 최소화하는 방향으로 설계되었습니다.

- **Performance Highlights**: 알고리즘의 성능은 다양한 시뮬레이션을 통해 검증되었습니다. 결과적으로, 기존 방법들보다 더 효율적으로 전문가 예측을 집계할 수 있는 능력을 보여주었습니다. 특히, 사전 지식 없이는 어려운 환경에서도 안정적인 성능을 발휘하는 것으로 나타났습니다.



### A New Flexible Train-Test Split Algorithm, an approach for choosing among the Hold-out, K-fold cross-validation, and Hold-out iteration (https://arxiv.org/abs/2501.06492)
- **What's New**: 이 논문은 인공지능이 다양한 산업에 미치는 영향을 다루며, 특히 머신러닝에서 중요한 역할을 하는 예측 모델의 평가 방법을 집중적으로 개선하고자 합니다. 연구는 여러 머신러닝 알고리즘의 정확도를 높이는 데 중점을 두고 있으며, Hold-out, Hold-out with iteration, K-fold 교차 검증의 효과성을 평가합니다. 이러한 방법들을 활용하여, 데이터셋의 특징에 따라 최적의 K 값을 찾는 필요성을 강조합니다.

- **Technical Details**: 이 연구에서는 파라미터 조정(Parameters tuning)을 통해 Hold-out 및 K-fold 교차 검증 방법을 평가하기 위해 유연한 Python 프로그램을 작성했습니다. 테스트 크기(Test size), Random State, 'k' 값 같은 변수를 수정하여 정확도 평가를 개선했습니다. Hold-out 검증 방법은 특히 10%의 테스트 크기에서 지속적으로 우수한 성능을 보이며, 반복 및 Random State 설정에 따라 소폭의 정확도 차이를 나타냅니다.

- **Performance Highlights**: 결과적으로 결정 트리(Decision Tree)는 Framingham 데이터셋에서 우수한 성능을 보였으며, Naive Bayes와 K Nearest Neighbors는 COVID-19 데이터셋에서 최고의 결과를 기록했습니다. 또한 K-fold 교차 검증을 통해 데이터셋에 따라 최적의 K 값이 달라짐을 보여주며, 10%의 테스트 크기와 90%의 훈련 크기를 권장합니다. 이 연구는 데이터셋의 기능, 샘플 크기 및 선택한 방법론들이 결과에 미치는 맥락적인 영향을 강조합니다.



### Automated Detection and Analysis of Minor Deformations in Flat Walls Due to Railway Vibrations Using LiDAR and Machine Learning (https://arxiv.org/abs/2501.06457)
Comments:
          I am requesting the withdrawal of my paper due to the need for significant revisions to ensure the accuracy and integrity of the presented findings

- **What's New**: 이번 연구에서는 인근 철도 선로의 진동으로 인한 평면 벽의 미세 변형(Deformation)을 자동으로 식별하기 위한 고급 방법론을 소개합니다. 이 방법은 고밀도 Terrestrial Laser Scanner (TLS) LiDAR 조사와 AI/ML 기법을 활용하여 데이터를 수집 및 분석합니다. 이는 도시 인프라의 구조적 무결성(Integrity)과 공공 안전을 보장하는 데 중요한 지속적인 모니터링의 필요성을 강조합니다.

- **Technical Details**: 스캔 데이터는 세밀한 포인트 클라우드(Point Cloud)로 처리되어 땅, 나무, 건물 및 기타 물체를 구분하기 위해 세분화(Segmentation)됩니다. 분석은 평면 벽을 따라 특정 구역을 식별하고, 지면 방향 대비 변형을 추정하는 데 초점을 맞추고 있습니다. 연구는 RGIPT 캠퍼스에서 수행되었으며, 철도 복도 근처의 벽에서 발견된 변형은 최대 7~8cm, 평균 3~4cm에 달합니다.

- **Performance Highlights**: 이 자동화된 특성 추출 및 변형 모니터링 프로세스는 구조적 건강 모니터링(Structural Health Monitoring)에 대한 잠재력을 보여줍니다. LiDAR 데이터와 머신러닝(Machine Learning)을 통합함으로써, 이 방법론은 구조적 변형을 식별하고 분석하는 효율적인 시스템을 제공합니다. 도시 인프라 관리를 보다 효과적으로 할 수 있는 중요한 발전을 나타냅니다.



### On the Computational Capability of Graph Neural Networks: A Circuit Complexity Bound Perspectiv (https://arxiv.org/abs/2501.06444)
- **What's New**: 본 논문은 Graph Neural Networks (GNNs)의 이론적 한계를 회로 복잡성(circuit complexity) 관점에서 분석한 새로운 접근 방식을 제시합니다. 기존 연구들은 주로 Weisfeiler-Lehman (WL) 그래프 동형성 검사를 통해 GNN의 표현력을 특성화하는 데 집중하였으나, 본 논문은 GNN 아키텍처의 회로 복잡성을 분석하여 일정 깊이의 층, 선형 또는 하위 선형 임베딩 크기 및 다항 정밀도 조건 하에서 GNN이 그래프 연결성 및 그래프 동형성 문제를 해결할 수 없음을 증명합니다. 이러한 결과는 GNN의 경험적 성공 이면에 있는 본질적인 표현력의 한계를 드러냅니다.

- **Technical Details**: GNN의 기본 활성화 함수에서부터 전체 그래프 컨볼루션 프로세스에 이르는 GNN 구성 요소의 회로 복잡성을 평가합니다. 우리는 상수 개수의 층, 다항(n-polynomial) 정밀도, d=O(n) 임베딩 크기를 가진 GNN이 균일한 TC0 회로로 근사 가능하다는 것을 보여줍니다. 이에 따라, TC0 = NC1이 아닌 이상, 이러한 GNN은 그래프 연결성 문제나 그래프 동형성 문제와 같은 문제를 해결할 수 없음을 규명합니다.

- **Performance Highlights**: 기존 GNN의 표현력을 측정하는 데 있어 Weisfeiler-Lehman (WL) 계층을 사용하는 전통적인 접근법과 달리, 본 논문은 회로 이론을 기반으로 하여 GNN의 계산적 한계를 정량화합니다. 연구 결과는 GNN이 특정한 조건에서 해결할 수 없는 문제들을 명확히 하며, GNN의 신뢰성을 높이는 데 기여할 수 있는 새로운 프레임워크를 소개합니다. 이와 같은 접근법은 다양한 GNN 모델 및 그래프 결정을 분석하는 데 활용될 수 있습니다.



### Deep Learning on Hester Davis Scores for Inpatient Fall Prediction (https://arxiv.org/abs/2501.06432)
Comments:
          Accepted for presentation at IEEE SSCI 2025

- **What's New**: 본 연구는 병원에 입원한 환자의 낙상 위험 예측을 위한 새로운 기계 학습 접근 방식을 제안합니다. 기존의 Hester Davis 점수(HDS)를 기반으로 한 임계값(threshold) 접근법과 비교하여, 기계 학습을 통해 더 정교한 예측 모델을 개발합니다. 새로운 모델은 한 단계 앞선 낙상 예측(one-step ahead fall prediction) 및 시퀀스-투-포인트 예측(sequence-to-point prediction) 두 가지로 구성됩니다.

- **Technical Details**: 제안된 모델은 HDS 값을 현재 시각에서 사용하여 다음 시각의 낙상 위험을 예측하는 한 단계 앞선 예측 전략을 사용합니다. 다른 한편으로, 모든 이전 HDS 값을 활용하여 낙상 위험을 예측하는 시퀀스-투-포인트 모델은 딥 러닝(deep learning)을 활용하여 시간에 따른 패턴을 포착합니다. 특히 순환 신경망(Recurrent Neural Networks, RNN), 장단기 메모리(Long Short-Term Memory, LSTM) 네트워크, 및 게이트 순환 유닛(Gated Recurrent Unit, GRU)을 통해 진행됩니다.

- **Performance Highlights**: 기계 학습 모델은 전통적인 HDS 기반의 임계값 접근법보다 향상된 정확도로 낙상 위험을 예측할 수 있음을 보여주었습니다. 연구 결과는 이들 모델이 환자의 변화를 동적으로 반영하며, 낙상 예방을 위한 데이터 기반 접근 방식의 가능성을 강조합니다. 보다 안전한 환자 관리를 위한 예측 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Reliable Imputed-Sample Assisted Vertical Federated Learning (https://arxiv.org/abs/2501.06429)
- **What's New**: 이번 논문에서는 Reliable Imputed-Sample Assisted (RISA) VFL 프레임워크를 제안하여 비공유 샘플을 효과적으로 활용하는 방법을 다룹니다. 기존의 VFL 접근법들은 제한된 겹치는 샘플จำนวน에 의존하다 보니 비공유 샘플을 적절히 이용하지 못하는 문제점을 가지고 있었습니다. RISA 프레임워크는 이러한 문제를 해결하기 위해 불확실성 추정을 통해 신뢰할 수 있는 비공유 샘플만 선별하여 VFL 모델 학습에 활용합니다.

- **Technical Details**: RISA는 비공유 샘플의 결측 속성을 평균 대치(mean imputation) 기법을 통해 보완하며, 자가 학습(self-training)을 통해 레이블을 추정합니다. 특히, 증거 이론(evidence theory)에 기반한 불확실성 추정을 도입하여, 각 샘플의 품질을 평가하고 신뢰도 높은 비공유 샘플만 선택할 수 있도록 합니다. 이를 통해 RISA는 동적인 가중치 협력(dynamically weighted collaboration)을 달성하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RISA는 특히 제한된 겹치는 샘플을 가지고 있을 때 성능 향상이 두드러지며, CIFAR-10 데이터셋에서는 1%의 겹치는 샘플로 48%의 정확도 개선을 달성했습니다. 두 개의 널리 사용되는 데이터셋(CIFAR-10 및 Criteo)에서 RISA는 최첨단 기술(SOTA) 방법들보다 우수한 성능을 보였습니다. 이러한 성과는 RISA가 VFL 모델의 성능을 최적화하는 데 기여하는 점을 입증합니다.



### DiscQuant: A Quantization Method for Neural Networks Inspired by Discrepancy Theory (https://arxiv.org/abs/2501.06417)
- **What's New**: 이 논문에서는 신경망의 가중치를 양자화하는 과정에서 가중치를 최적의 방식으로 반올림하는 문제를 연구합니다. 최신 방법인 DiscQuant를 제안하여 Round-to-Nearest(RTN) 및 GPTQ와 같은 기존의 양자화 방법들보다 성능을 크게 향상시킵니다. DiscQuant는 데이터 의존적인 방식을 사용하여 양자화 모델의 질을 개선합니다.

- **Technical Details**: 양자화 과정은 (1) 가중치를 위한 저비트 복잡도 표현을 구성하고 (2) 원래 가중치를 양자화 그리드의 값으로 반올림하는 두 단계를 포함합니다. 이 연구에서는 discrepancy theory의 관점으로 이러한 반올림 문제를 탐구하고, m = poly(1/ε) 샘플에서 약한 저랭크(Hisck한, low-rank) 그라디언트를 가진 경우 양자화 모델의 예상 근사 오차가 ε 이하가 되도록 할 수 있음을 증명합니다.

- **Performance Highlights**: DiscQuant를 사용하여 Phi3mini-3.8B 모델을 3.25 비트 양자화 그리드로 반올림한 결과, GSM8k 데이터셋에서 64%의 정확도를 달성했습니다. 반면 기존의 GPTQ는 54% 및 RTN은 31%의 정확도를 기록하였습니다. 이 결과는 DiscQuant가 기술적으로 우수한 성능을 보임을 입증합니다.



### Influencing Humans to Conform to Preference Models for RLHF (https://arxiv.org/abs/2501.06416)
- **What's New**: 본 논문은 인간의 보상 함수를 근사하기 위해 인간 피드백을 통한 강화 학습(RLHF) 알고리즘 설계를 다룹니다. 연구자들은 인간의 선호 모델에 대한 적절한 가정이 없을 경우 잘못된 보상 함수 근사를 배우는 위험이 있음을 지적합니다. 이를 해결하기 위해, 연구진은 세 가지 개입 방법을 제시하여 인간의 선호가 원하는 모델에 더 잘 부합하도록 영향을 미칠 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 선호 모델로서 부분적 수익(partial return) 모델과 후회 탓(regret) 모델을 분석합니다. 연구진은 MDP(Markov Decision Process)가 작업 환경을 어떻게 나타내는지에 대한 설명을 제시하고, 연산에서 보상 함수의 역할을 명확하게 합니다. 각각의 개입 방법인 특권(privileged) 실험, 훈련된(trained) 실험, 질문(question) 실험을 통해 인간의 선호 모델 적합성을 테스트했습니다.

- **Performance Highlights**: 모든 개입 방법은 유의미한 효과를 보여주었으며, 이는 선호 데이터 품질을 개선하고 학습된 보상 함수의 조정을 촉진하는 실용적 도구가 될 수 있음을 나타냅니다. 특히, 특권 방법과 훈련된 방법은 목표 선호 모델과의 높은 부합을 이끌어냈습니다. 이러한 결과는 RLHF 분야의 실무자들에게 유용한 지침을 제공하며, 미래의 연구 방향을 제시합니다.



### Task Delay and Energy Consumption Minimization for Low-altitude MEC via Evolutionary Multi-objective Deep Reinforcement Learning (https://arxiv.org/abs/2501.06410)
- **What's New**: 이 논문은 저고도 경제(LaE)의 UAV(무인 항공기) 지원 모바일 엣지 컴퓨팅(MEC) 시스템에서 작업 지연과 UAV의 에너지 소비 간의 균형을 다루는 동적 최적화 전략을 제안합니다. UAV가 엣지 서버를 장착하고, 지상 장치(GDs)를 위한 작업 오프로딩을 위해 복잡한 최적화 문제를 해결하는 접근 방식이 포함되어 있습니다. 이 연구는 멀티 객체 최적화 문제(CDECMOP)를 설정하고 이를 해결하기 위해 딥 강화 학습(DRL) 알고리즘을 적용했습니다.

- **Technical Details**: 제안된 시스템은 UAV가 엣지 서버를 운반하여 GDs의 계산 서비스 및 작업 스케줄링을 지원하는 복잡한 최적화 문제를 포함합니다. 이 논문에서는 작업 지연과 UAV 에너지 소비 간의 본질적인 충돌을 최적화하기 위해 다중 목표 마르코프 결정 과정(MOMDP)을 활용했습니다. 또한, 에볼루셔너리 멀티-오브젝티브 DRL(EMODRL) 프레임워크와 다중 목표 분포 학습(TDL) 알고리즘을 연계하여 성능과 안정적 수렴을 보장했습니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안된 알고리즘이 고품질의 파레토 솔루션을 달성하며, 기존 방법보다 여러 최적화 목표 간의 균형을 잘 이루었다고 보고합니다. 또한, 이 방법은 다른 EMODRL 방법에 비해 수렴성과 비 지배 솔루션의 효과성 면에서도 우수한 성능을 보여주었습니다. 전체적으로, 이 연구는 UAV 지원 MEC 시스템에서의 에너지 효율성과 작업 처리 지연 문제를 효과적으로 해결할 수 있는 새로운 접근 방식을 제시합니다.



### Mathematics of Digital Twins and Transfer Learning for PDE Models (https://arxiv.org/abs/2501.06400)
Comments:
          22 pages, 7 figures

- **What's New**: 이 논문에서는 디지털 트윈(Digital Twin, DT)을 부분 미분 방정식(Partial Differential Equations, PDE)으로 제어되는 물리적 시스템의 실시간 시뮬레이션 및 제어를 위한 모델로 정의합니다. 저자들은 Karhunen-Loève Neural Network (KL-NN) 대체 모델과 전이 학습(Transfer Learning, TL)을 사용하여 DT를 구축하며, 이러한 방법론은 빠른 추론과 제어 파라미터에 대한 미분 가능성을 제공합니다.

- **Technical Details**: KL-NN 대체 모델을 사용하여 DT를 구성하고, 확률 분포를 통해 특정 범위 내에서 제어 변수의 변동성을 모델링합니다. 저자들은 상태와 제어 변수를 KL 확장을 사용하여 표현하며, 개별 PDE 문제에 대한 TL 속성을 분석합니다. 이 방법은 시스템의 평균 함수 및 공분산의 변화를 분석하여 전이 가능성을 가이드합니다.

- **Performance Highlights**: 이 연구에서 선형 PDE 문제의 경우 KL-NN 대체 모델의 바뀔 수 없는 파라미터는 새로운 목표 조건 아래 제어 변수의 평균값에 해당하는 PDE의 단일 해로부터 정확하게 추정이 가능하다고 합니다. 반면 비선형 PDE 문제에서는 파라미터 전이에 오류가 발생하지만, 저자는 특정 제어 변수의 범위 안에 있을 경우 큰 오류 없이 몇몇 파라미터를 전이할 수 있음을 발견했습니다.



### Kolmogorov-Arnold networks for metal surface defect classification (https://arxiv.org/abs/2501.06389)
- **What's New**: 이번 논문에서는 금속 표면 결함(classifying metal surface defects) 분류에 Kolmogorov-Arnold Networks (KAN)을 적용한 사례를 제시합니다. 특히, 스틸 표면을 분석하여 균열(cracks), 이물질(inclusions), 패치(patches), 움푹 들어간 표면(pitted surfaces), 긁힌 표면(scratches)과 같은 결함을 탐지합니다.

- **Technical Details**: Kolmogorov-Arnold 정리(Kolmogorov-Arnold theorem)를 기반으로 한 KAN은 기존의 다층 퍼셉트론(multilayer perceptrons, MLPs)보다 더 효과적인 함수 근사(function approximation)를 가능하게 합니다. KAN은 스플라인 함수(spline functions)를 사용하여 더 효율적인 모델을 만들며, 이로 인해 이미지 분류에서의 성능이 향상됩니다.

- **Performance Highlights**: 결과적으로 KAN 네트워크는 보다 적은 파라미터로 컨볼루션 신경망(convolutional neural networks, CNNs)보다 뛰어난 정확도를 달성할 수 있음을 보여주었습니다. 이는 더 빠른 수렴(faster convergence)과 향상된 이미지 분류 성능을 의미합니다.



### Using Pre-trained LLMs for Multivariate Time Series Forecasting (https://arxiv.org/abs/2501.06386)
- **What's New**: 이 논문에서는 사전 훈련된 대규모 언어 모델(LLM)을 활용하여 다변량 수요 시계열 예측에 접근하는 새로운 방법론을 제안합니다. 특히, 다변량 패칭(multivariate patching) 전략을 통해 시계열 특성을 디코더 전용으로 사전 훈련된 트랜스포머에 임베딩하는 방식으로, 최신 시계열 예측 모델과 경쟁할 수 있는 결과를 보여줍니다. 이 혁신적인 접근은 다변량 데이터에 대한 예측 정확도를 개선할 수 있는 가능성을 발견하게 합니다.

- **Technical Details**: 주요 기술적 기법으로는 다변량 시계열을 LLM의 토큰 임베딩 공간으로 매핑하는 방법과, 그 후에 다시 원래의 시계열 공간으로 매핑하는 역방향 방법이 포함됩니다. 또한, 층 정규화(layer norms)만 미세 조정하여 훈련 가능 파라미터의 수를 획기적으로 줄입니다. 논문에서는 두 가지 접근 방식을 통해 LLM 기반 시계열 예측의 실증 분석을 제공합니다.

- **Performance Highlights**: 모델의 성능 검증을 위해 MQCNN 모델을 베이스라인으로 설정하고, 사전 훈련된 LLM에서 소수의 파라미터만 조정하여도 전문 엔지니어링 아키텍처와 유사한 성능에 도달할 수 있음을 보여줍니다. 훈련 데이터 접근 없이도 모델 진단을 수행할 수 있는 중량 기반 진단 기법을 사용하여 예측 정확도를 평가합니다. Layer-specific weight analysis를 통해 품질과 예측 정확도 간의 관계를 분석하였음을 밝혔습니다.



### On the Partial Identifiability in Reward Learning: Choosing the Best Reward (https://arxiv.org/abs/2501.06376)
- **What's New**: 이번 논문에서는 Reward Learning (ReL) 분야에서 목표 보상(target reward)을 효과적으로 식별하기 위해 기존의 부분적으로 식별 가능한 상태를 초월하는 새로운 보상 선택 방법을 도입합니다. 특히, 기존의 feasible set 내에서 보상을 임의로 선택하기보다 ReL의 특정 응용에 따라 성능이 개선되는 보상을 선택할 수 있음을 보여줍니다. 이를 통해 ReL의 응용 가능성을 확장하고, 예를 들어 reward transfer와 같은 특정 사례에 적합한 알고리즘을 제시합니다.

- **Technical Details**: 논문은 새로운 양적 프레임워크(quantitative framework)를 제안하며, 이를 통해 ReL 문제를 간단하지만 표현력이 있도록 분석할 수 있습니다. 이 프레임워크를 활용하여, 다양한 보상을 수치적으로 비교하는 방법과 새로운 보상 선택을 제시하고 그 이점을 강조합니다. 연구에서는 특정 사용 사례와 관련하여 세 가지 효율적인 ReL 알고리즘을 개발하며, 이를 일러스트레이션 시뮬레이션을 통해 검증하였습니다.

- **Performance Highlights**: 새로운 보상 선택 방법은 기존 보상 선택 방법보다 특정 응용에 대해 더 나은 성능을 보임을 증명합니다. 제안된 알고리즘은 기존 알고리즘들에 비해 효율성을 뒷받침하는 증명 가능한 성과를 보여줍니다. 이러한 결과는 ReL 알고리즘의 확장과 실제 응용에서의 활용 가능성을 높여 줄 것으로 기대됩니다.



### Towards Robust Nonlinear Subspace Clustering: A Kernel Learning Approach (https://arxiv.org/abs/2501.06368)
- **What's New**: 이번 논문에서는 비선형 서브스페이스 클러스터링을 위한 새로운 패러다임인 DKLM을 제안합니다. 기존의 방법들이 고정된 커널에 의존하던 반면, DKLM은 데이터의 자기 표현에서 직접 커널을 학습하여 적응형 가중치와 삼각 부등식 제약 조건을 만족합니다. 이로써 비선형 공간에서 데이터의 지역 매니폴드 구조를 보존하고 최적의 블록 대각 행렬을 형성하는데 기여합니다.

- **Technical Details**: DKLM은 특정 조건 하에 자기 표현을 통해 커널을 학습하며, 이는 적응형 가중치 커널 학습으로 볼 수 있습니다. 이 접근법은 데이터 포인트가 속하는 서브스페이스를 효과적으로 찾는 방법을 제시하며, 자기 표현 기반의 서브스페이스 클러스터링 프레임워크 내에서 구현될 수 있습니다. 이론적 분석을 통해 DKLM이 기존 방법과의 관계를 설명하고, 다양한 데이터 세트에 대한 실험을 통해 그 효과를 입증하였습니다.

- **Performance Highlights**: DKLM은 첨단의 자기 표현 및 커널 기반 서브스페이스 클러스터링 방식에 비해 우수한 클러스터링 성능을 보여줍니다. 특히, 네 개의 합성 데이터 세트와 아홉 개의 실제 데이터 세트에서 우리의 접근법의 정확성과 강인성을 확인하였으며, 고차원 시계열 데이터에 대한 실험에서도 비선형 패턴 발견의 효과성을 입증하였습니다.



### On The Statistical Complexity of Offline Decision-Making (https://arxiv.org/abs/2501.06339)
Comments:
          arXiv version for the ICML'24 paper

- **What's New**: 이 논문은 함수 근사를 통한 오프라인 의사 결정의 통계적 복잡성을 연구하고, 확률적 컨텍스트 밴디트 및 마르코프 의사 결정 프로세스에 대해 (근사) 미니맥스 최적 비율을 설정합니다. 특히, 가치 함수 클래스의 유사 차원(pseudo-dimension)과 행동 정책에 대한 새로운 특성을 도입하여 오프라인 의사 결정 문헌의 데이터 커버리지 개념을 보완합니다. 또한, 오프라인 데이터를 온라인 의사 결정에 사용할 때의 이점을 이해하고 다양한 환경에서 거의 미니맥스 최적 비율을 보여줍니다.

- **Technical Details**: 저자들은 정책 전이 계수(policy transfer coefficients)라는 개념을 도입하여 고전적인 데이터 커버리지 개념을 포함하는 방식으로 오프라인 학습 가능성을 보다 정밀하게 설명합니다. 이 연구는 선형 및 신경망 기반의 함수 근사 클래스 등 다양한 근사 클래스에서의 오프라인 학습을 다루며, 하이브리드 오프라인-온라인 학습 설정에서도 결과를 확장합니다. 논문은 벨먼 유사 손실에 대한 균일한 번시안(Bernstein's) 불평등을 제공하고, Hegde 알고리즘의 반복 횟수를 줄이는 기술적인 문제를 해결합니다.

- **Performance Highlights**: 연구 결과는 다중 무장 밴디트, 컨텍스트 밴디트 및 마르코프 의사 결정 프로세스와 같은 다양한 오프라인 학습 문제에 대한 최적의 하한 및 상한을 제시합니다. 특히, 제안된 정책 전이 계수는 이전 문헌에서 다룬 학습 가능성 문제를 포함하며, 오프라인 데이터의 질의 중요성을 강조합니다. 이 논문은 오프라인 의사 결정 문제의 더 포괄적이고 정교한 특성을 제공하며, 실제 문제 해결에 기여할 수 있는 중요한 통찰을 제공합니다.



### Aggregating Low Rank Adapters in Federated Fine-tuning (https://arxiv.org/abs/2501.06332)
Comments:
          presented at conference this https URL

- **What's New**: 본 논문에서는 Federated Learning 컨텍스트에서 Low-Rank Adaptation(LoRA) 방법을 적용한 새로운 집계 방법을 제안합니다. 저자들은 기존의 여러 집계 방법들과 비교하며, 저희의 제안된 방법이 어떻게 성능을 개선할 수 있는지를 평가했습니다. 특히, GLUE benchmark 데이터셋에서의 성과를 기반으로 새로운 접근법의 유용성을 입증합니다.

- **Technical Details**: LoRA 방법은 각 Fine-tuning 이터레이션에서 기울기의 저차원 근사를 훈련하는 방식으로, 이를 통해 학습 파라미터 수를 대폭 줄일 수 있습니다. 본 연구에서는 LoRA를 Federated Learning 환경에 적용하며, 중앙 서버가 여러 클라이언트의 파라미터를 집계하는 방식을 채택합니다. 이 과정에서 등장하는 초기 모드의 기여도를 강화하는 체계적인 방법론을 도입하고, 실험을 통해 조기 수렴 개선 사례를 제시합니다.

- **Performance Highlights**: 제안된 집계 방법은 모델 훈련 초기 단계에서 더욱 안정적이고 빠른 수렴을 제공합니다. LoRA의 효과를 최대화하기 위해 여러 실험을 진행했으며, 실질적으로 각 클라이언트에서 훈련된 저각(adaptive) 매트릭스의 집계 결과를 분석했습니다. 특히, GLUE benchmark 데이터셋에서의 성능 평가는 제안 방법의 우수성을 보여줍니다.



### On Creating A Brain-To-Text Decoder (https://arxiv.org/abs/2501.06326)
- **What's New**: 본 논문에서는 생체 신호 중 하나인 자연스러운 전기 뇌파(EEG) 신호를 활용하여 인지 언어 과정을 해독하려는 첫 번째 시도를 다루고 있습니다. 이 접근 방식은 fMRI와 같은 기존의 고비용 및 시간 소모적인 방법에 비해 실시간으로 더 효율적이고 저렴한 방법이 될 것으로 기대됩니다. 연구는 뇌-컴퓨터 인터페이스(BCI)가 말의 생성과 관련된 신경 신호를 해독하는 데 효과적임을 강조하며, 단어 오류율(Word Error Rate, WER)이 기존 기술보다 우수한 성과를 나타냅니다.

- **Technical Details**: EEG는 두피에서 뇌의 전기적 활동을 추출하는 절차로, 뇌의 피질(cerebral cortex)의 동적 활동을 반영한다고 가정됩니다. 이 연구에서는 국제 10/20 시스템을 사용하여 지정된 위치에 전극(electrode)을 부착하고, EEG 신호를 실시간 수치 값으로 변환하여 해석 가능한 방식으로 뇌 활동을 기록합니다. 연구 방법론은 회귀 및 자가 지도 학습(self-supervised learning)을 활용하여 EEG 신호를 텍스트로 빠르게 변환하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 본 연구에서는 대규모의 비지도 학습에 기반해 한정된 라벨 수로도 뛰어난 성능을 발휘하는 음성 인식 모델이 도출되었습니다. 특히, 이 연구는 Librispeech 벤치마크에서 경쟁력 있는 단어 오류율(WER)을 달성하며, 기존의최첨단 기술을 초과하는 성과를 보여주고 있습니다. 또한, 모델 크기와 비지도 학습 데이터를 포함하여 음성 인식 오류 패턴에 대한 종합적인 분석이 포함되어 향후 BCI 성능을 향상시키기 위한 기초 자료를 제공합니다.



### Uncertainty Estimation for Path Loss and Radio Metric Models (https://arxiv.org/abs/2501.06308)
Comments:
          5 pages, 12 figures

- **What's New**: 이번 연구에서는 Conformal Prediction (CP)을 활용하여 Conformal Predictive Systems (CPS) 형태로 불확실성을 정확하게 추정하는 방법을 제시합니다. 기계 학습(ML) 기반의 라디오 메트릭 모델들과 2-D 맵 기반의 ML 경로 손실 모델에 적용하였습니다. 다양한 난이도 추정기를 사용하여 통계적으로 강건한 95% 신뢰 예측 구간(PIs)을 구축하는 방법론이 혁신적임을 보여주고 있습니다.

- **Technical Details**: CPS 모델은 토론토를 포함한 여러 데이터셋에서 훈련되었으며, 밴쿠버와 몬트리올 같은 다른 도시로 일반화되는 능력이 우수합니다. 훈련된 모델은 높은 커버리지와 신뢰성을 유지하면서 통계적 접근 방법을 통해 불확실성을 추정하였고, 난이도 추정기들은 도전적인 샘플을 식별하여 RMSE의 측정을 줄이는 데 기여했습니다.

- **Performance Highlights**: 실험 결과, CPS를 통한 불확실성 추정의 효과가 입증되었으며, 네트워크 계획, 운영, 스펙트럼 관리에 중요한 통찰력을 제공할 가능성이 높습니다. 데이터셋 난이도가 감소할수록 모델의 성능이 개선되어, 더 효과적인 무선 네트워크 모델링을 보여주는 것으로 확인되었습니다.



### Cluster Catch Digraphs with the Nearest Neighbor Distanc (https://arxiv.org/abs/2501.06268)
Comments:
          28 pages, 4 figures, and 10 tables

- **What's New**: 본 논문에서는 Cluster Catch Digraphs (CCDs)을 기반으로 한 새로운 클러스터링 방법을 소개합니다. 이 방법은 Nearest Neighbor Distance (NND)를 사용하는 새로운 종류의 spatial randomness test를 채택하여 기존의 RK-CCDs의 한계를 극복합니다. Monte Carlo 분석을 통해 이 방법이 높은 차원의 데이터셋에서 특히 효과적임을 보여줍니다.

- **Technical Details**: 제안된 UN-CCDs 방법은 Ripley의 K function 대신 NND를 활용하여 데이터의 점 중심 패턴을 테스트합니다. RK-CCDs와 마찬가지로 UN-CCDs는 거의 파라미터가 필요 없어 사용이 간편하며, SU-MCCD 및 U-MCCD와 같은 이상치 탐지 방법도 소개됩니다. 이러한 새로운 방법은 다양한 형상의 클러스터에 대응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 클러스터링 방법은 KS-CCDs 및 RK-CCDs와 비교했을 때 유사하거나 더 뛰어난 성능을 보입니다. 실질적인 데이터셋에서 전통적인 클러스터링 방법과 비교하였으며, 경쟁력 있는 성능을 보여주며 높은 품질의 클러스터를 생성합니다. 특히, 차원이 10 이상인 경우 SU-MCCD 방법이 U-MCCD 방법보다 월등한 성능을 보이는 것으로 나타났습니다.



### Contextual Bandit Optimization with Pre-Trained Neural Networks (https://arxiv.org/abs/2501.06258)
Comments:
          Master's thesis

- **What's New**: 이 논문에서는 Bandit optimization 문제를 다루며, 특히 고차원 보상 모델에서의 도전 과제를 강조합니다. 연구는 작은 모델에서도 pre-training이 어떻게 도움이 되는지를 조사하며, stochastic contextual bandit 문제에 multi-layer neural network를 적용합니다. 여기서 마지막 층은 linear predictor로 설정되고, 앞의 층은 black box neural architecture로 구성된 representation network로 봅니다.

- **Technical Details**: 연구에서는 pre-training을 representation network의 초기 가정으로 모델링하고, Explore Twice then Commit (E2TC)라는 새로운 알고리즘을 도입합니다. 이 알고리즘은 탐사 단계에서 Ridge regression을 활용해 마지막 층의 가중치를 추정하고, 이후 모든 가중치에 대해 Stochastic Gradient Descent를 수행합니다. 특정 조건 하에, 마지막 층의 차원과 행동 수 K가 horizon T보다 훨씬 작은 경우 sublinear regret가 발생함을 증명합니다.

- **Performance Highlights**: E2TC 알고리즘은 misspecified linear bandit 문제로 환원되는 약한 학습 환경에서 효율성을 증명하고, 보상에 대한 범위를 $O(\epsilon_0\sqrt{d}KT+(KT)^{4 /5})$ 또는 $\tilde{O}(\epsilon_0\sqrt{d}KT+d^{1 /3}(KT)^{2 /3})$로 제공하여, 정규화 강도에 따라 다릅니다. 또한, E2TC의 regret과 탐사의 샘플 복잡성을 평가하기 위한 실험을 수행했습니다.



### Progressive Supervision via Label Decomposition: An Long-Term and Large-Scale Wireless Traffic Forecasting Method (https://arxiv.org/abs/2501.06255)
Comments:
          Published at Knowledge-Based Systems. arXiv admin note: substantial text overlap with arXiv:2412.00108

- **What's New**: 장기 및 대규모 무선 트래픽 예측(LL-WTF)의 중요성이 강조되었습니다. 무선 트래픽의 비정상성(non-stationarity) 문제와 도시 규모에서의 많은 네트워크 노드로 인해 큰 도전이 되는 점을 다룹니다. 이 문제를 해결하기 위해 제안된 Progressive Supervision 방법은 Label Decomposition(PSLD)을 기반으로 하며, 이는 여러 개의 쉽게 학습 가능한 구성 요소로 데이터를 분해하는 방식을 사용합니다.

- **Technical Details**: PSLD의 핵심 기술은 Random Subgraph Sampling(RSS) 알고리즘입니다. 이 알고리즘은 대규모 트래픽 데이터를 샘플링하여 효율적으로 네트워크 학습을 가능하게 합니다. PSLD는 비정상 문제를 해결하기 위해 얕은 층에서 점진적으로 학습되는 구성 요소들을 결합하는 방식을 사용하며, 이를 통해 학습 과정을 개선할 수 있습니다.

- **Performance Highlights**: 제안된 PSLD 방법은 세 가지 대규모 WT 데이터셋에서 기존의 최신 기술(SOTA) 방법들과 비교했을 때 평균 2%, 4%, 11%의 성능 향상을 나타냈습니다. 또한, PSLD는 예측 정확도뿐 아니라 처리 속도에서도 뛰어난 성능을 보입니다. 연구를 위해 개발된 오픈 소스 라이브러리인 WTFlib는 다양한 SOTA 방법을 포함하여 관련 연구의 기초를 제공합니다.



### $\text{Transformer}^2$: Self-adaptive LLMs (https://arxiv.org/abs/2501.06252)
Comments:
          18 panges, 11 figures, 9 tables

- **What's New**: 최근의 연구에서 소개된 $	ext{Transformer}^2$는 자가 적응형 대형 언어 모델(LLMs)의 새로운 프레임워크로, 기존의 훈련 방식보다 적은 자원으로 미지의 작업에 즉각적으로 적응할 수 있는 기술을 제공합니다. 이 방법은 기계 학습에서 흔히 사용되는 LoRA와 같은 기법보다 더 나은 성능을 보여주며, 가벼운 메모리 용량과 높은 효율성을 자랑합니다. $	ext{Transformer}^2$는 여러 LLM 아키텍처 및 모달리티에 걸쳐 다재다능함을 입증하였습니다.

- **Technical Details**: $	ext{Transformer}^2$의 두 가지 주요 메커니즘은 작업 속성을 식별하는 디스패치 시스템과, 강화학습(강화 학습, RL)으로 학습된 작업 특정 '전문가' 벡터들을 동적으로 조합하는 것입니다. 이 프레임워크는 단일 입력에 대한 두 번의 패스를 사용하여 모델의 가중치를 조절하는 방식을 통해 이루어집니다. 이를 위해, Singular Value Fine-tuning (SVF)라는 새로운 기법을 도입하여 모델의 가중치 행렬 내의 특이값만을 조정하여 과적합(overfitting) 문제를 줄이고 자원 소비를 최소화합니다.

- **Performance Highlights**: 대체로 다양한 실험을 통해 SVF와 전체 $	ext{Transformer}^2$ 프레임워크가 기존의 전통적인 효율적 파인튜닝 방법보다 우수하다는 것을 입증하였습니다. 특히, SVF는 적은 수의 매개변수로 토픽별 성능을 최적화하는 효과적인 전문가 벡터를 제공함으로써 비용 절감 효과를 선보였습니다. 또한, $	ext{Transformer}^2$는 시각적 질문 응답(visual question answering)과 같은 새로운 과제에서도 전반적인 성능 향상을 이루어냈으며, 진화 가능성이 있는 자가 적응형 AI 시스템 개발에 기여할 수 있는 기반을 마련하였습니다.



### Utility-inspired Reward Transformations Improve Reinforcement Learning Training of Language Models (https://arxiv.org/abs/2501.06248)
- **What's New**: 본 논문에서는 기존의 보상 함수가 평균화되는 방식의 단점을 지적하며, 특히 보상 함수의 선형 집합이 생성된 텍스트의 품질에 미치는 영향을 분석합니다. 저자들은 경제학 이론인 유틸리티 함수(Inada conditions)에서 영감을 받아 새로운 보상 변환 방법인 Inada Reward Transformation (IRT)를 제안합니다. 이 방식은 낮은 보상 값에 대한 감도를 높이고, 이미 높은 보상 값에는 낮은 감도를 적용하여 더 정교한 보상 처리를 가능하게 합니다.

- **Technical Details**: Inada Reward Transformation (IRT)는 보상 함수에 적용할 수 있는 변환 방법으로, 강한 편향을 최소화하며 특정 보상 차원을 잊지 않도록 설계되었습니다. 기존의 선형 평균화 방법과 달리, 이 방법은 다양한 보상의 상대적 중요성을 반영하여 텍스트 생성 품질을 높이고 더 적절한 모델 학습을 이끕니다. RLHF(강화 학습에 대한 인간 피드백) 과정에서 보상 모델은 사람의 선호를 예측하고, 이를 바탕으로 모델의 생성 품질을 향상시킵니다.

- **Performance Highlights**: IRT을 적용한 모델은 기존의 선형 보상 집합 방식에 비해 더 많은 도움을 주고 덜 해로운 결과를 생성하는 것으로 평가되었습니다. 정량적 및 정성적 분석을 통해, Inada 변환을 경험한 모델은 응답의 질이 향상되어 사용자 기대에 더 잘 부합하는 것으로 나타났습니다. 이는 향후 LLM(대형 언어 모델) 훈련 과정에서 보상 구조의 설계에 중요한 인사이트를 제공할 것입니다.



### Predicting House Rental Prices in Ghana Using Machine Learning (https://arxiv.org/abs/2501.06241)
Comments:
          13 pages, 8 figures, 2 tables

- **What's New**: 이번 연구는 가나의 주택 임대 가격을 예측하기 위한 머신 러닝 모델의 효율성을 조사합니다. 주택 시장 정보의 정확성과 접근성이 필요하다는 점을 강조하고 있으며, 다양한 임대 목록 데이터셋을 활용하였습니다.

- **Technical Details**: 다양한 모델을 훈련하고 평가했으며, CatBoost, XGBoost, Random Forest가 포함되었습니다. 그 중에서도 CatBoost가 가장 우수한 성능을 보이며, $R^2$ 값이 0.876에 달했습니다. 이 모델은 주택 시장에서 복잡한 관계를 효과적으로 포착하는 능력을 보여주었습니다.

- **Performance Highlights**: 특징 중요도 분석에 따르면, 임대 가격의 주요 요인은 지역 기반 특성, 침실 수, 욕실 수, 가구 상태 등입니다. 연구 결과는 부동산 전문가, 투자자, 정책 입안자 등 다양한 이해 관계자에게 유용한 통찰을 제공하며, 향후 연구의 기회로는 시간적 데이터 통합 및 지역 변동 탐색 등이 제안됩니다.



### The Convergence of Dynamic Routing between Capsules (https://arxiv.org/abs/2501.06240)
- **What's New**: 이 논문에서는 Capsule Network (CapsNet)의 routing 알고리즘에 대한 깊은 수학적 분석을 제시하여, 그 수렴(convergence)과 최적화 목표(objective)를 명확히 하고자 한다. 기존 연구들이 CapsNet의 성능에 대한 부정적인 결과를 보고했으나, 이 논문은 이러한 모순의 원인을 분석하고 해결할 방법을 제안한다.

- **Technical Details**: CapsNet의 dynamic routing algorithm의 목표 함수는 오목 함수(concave function)으로 정의되고, 이는 선형 제약 조건 하에서 최적화 알고리즘을 해결하기 위한 비선형(Nonlinear) 경량 경량화 방법으로 볼 수 있다. 또한, routing 알고리즘의 반대 효력(negative effect)과 연속적 반복이 link strength를 과도하게 분극화(polarize)할 수 있다는 점을 지적하고, 수학적으로 엄격한 수렴 증명을 제공한다.

- **Performance Highlights**: CapsNet은 많은 응용 분야에서 CNN보다 나은 성능을 보여주지만, 라우팅 알고리즘의 성능은 다소 일관되지 못하며 적절한 반복 횟수가 필요함을 강조한다. 이 연구를 통해 동적 라우팅 알고리즘의 수렴 증명과 함께 최적 문제의 구성을 통해 계산 복잡성을 줄일 수 있다는 점에서 실험 결과도 주목할 만하다.



### Multi-field Visualization: Trait design and trait-induced merge trees (https://arxiv.org/abs/2501.06238)
Comments:
          IEEE Transactions on Visualization and Computer Graphics. arXiv admin note: text overlap with arXiv:2308.09015

- **What's New**: 이 논문에서는 Feature Level Sets (FLS)를 이용한 다중 필드 데이터 분석을 위한 실용적인 방법론을 제시합니다. 특히, trait 설계와 feature 선택의 과제를 해결하기 위해 Cartesian decomposition을 통한 단순화된 trait 설계 방법과, trait-induced merge trees (TIMTs)를 도입하였습니다. 이 TIMTs는 주어진 trait에 가장 가까운 데이터 영역을 효과적으로 분석할 수 있는 새로운 가능성을 제공합니다.

- **Technical Details**: FLS는 속성 공간에서 정의된 trait에 의해 유도된 다변량 feature를 탐구하고, TIMT는 이러한 feature를 위계적으로 분석합니다. 특성 정의가 이루어진 후에는 FLS의 이등급(isovalue)을 설정함으로써 trait에 근접한 영역을 강조할 수 있습니다. 이 과정에서 Cartesian traits와 sparse dictionary learning을 통해 자동으로 단순한 점 trait을 제안하는 방법을 사용하여 trait 설계를 간소화하고 계산 효율성을 높입니다.

- **Performance Highlights**: 여러 도메인에서의 5가지 사례 연구를 통해 제안된 방법의 실용성을 입증하였습니다. 자동 trait 제안과 수동 설계의 효과를 비교하여, 새로운 trait은 다차원 데이터 분석과 와류 재연결(vortex-reconnection) 데이터의 탐색에 있어 실질적인 이점을 제공합니다. 이 연구는 다중 필드의 위상적 분석을 위한 고급 기술을 제공하며, 사용자 정의 trait을 통해 보다 직관적으로 시각화할 수 있는 가능성을 보여줍니다.



### Data-Driven Radio Propagation Modeling using Graph Neural Networks (https://arxiv.org/abs/2501.06236)
- **What's New**: 이 연구에서는 기존의 물리 기반 모델 대신 그래프 신경망(Graph Neural Networks)을 사용하여 무선 네트워크에서의 라디오 전파( radio propagation ) 행동을 실제 데이터로부터 직접 학습하는 새로운 방법을 제안합니다. 이 방법은 라디오 전파 환경을 그래프 구조로 변환하여 노드(node)는 위치를, 엣지(edge)는 지리적 관계를 나타내게 됩니다. 특히 이 연구는 그래프 신경망을 통해 점(point) 측정만으로도 신호 전파의 범위를 예측하는 능력을 보여줍니다.

- **Technical Details**: 본 논문에서는 그래프 신경망 아키텍처를 활용하여 전파 거동을 원활히 통합하는 방법을 제안합니다. 데이터 기반 접근 방식으로 실제 환경에서 수집한 센서 데이터를 그래프 구조로 변환하고, 이를 통해 라디오 신호의 전파를 예측하는 훈련 절차를 개발했습니다. 또한, 마스킹 출력 훈련(masked output training) 기법을 도입하여 제한된 데이터로부터 범위 맵을 생성하도록 모델을 일반화하는 방법도 소개됩니다.

- **Performance Highlights**: 타 전통적인 휴리스틱 모델과 비교할 때, 개발한 그래프 신경망은 속도와 정확성 면에서 우수한 성능을 보였습니다. 해당 연구에서 제안하는 그래프 기반 접근 방식은 물리적 모델에 비해 빠르고 정확한 범위 추정을 가능하게 하며, 현실 세계 데이터를 활용하여 커버리지 맵을 생성할 수 있는 첫 번째 사례로, 이는 미래 무선 네트워크 디자인에 중요한 기여를 할 것으로 기대됩니다.



### Mechanics and Design of Metastructured Auxetic Patches with Bio-inspired Materials (https://arxiv.org/abs/2501.06233)
- **What's New**: 이 연구는 음의 포아송 비율을 가진 메타구조 형태의 보강 패치(Auxetic Patches)에 대한 새로운 데이터 기반 설계 프레임워크를 소개합니다. 실크 피브로인(Silk Fibroin)으로 제작된 패치의 체계적인 모델링을 위해 인공지능 신경망(neural networks)을 활용하였습니다. 또한 이 연구는 인공신경망을 통해 각 스트레인(strain)에 대해 포아송 비율과 응력을 예측하고, 기존의 최적화 방법에 비해 우수한 성능을 시연하였습니다.

- **Technical Details**: 이 연구는 실험적 제작과 기계적 테스트을 통해 메타구조 보강 패치의 물성(properties)을 규명하고, 이에 대응하는 유한 요소 모델(finite element models)을 검증했습니다. GREEDY 샘플링(greedy sampling) 방법을 사용하여 데이터 레이블링에 대한 계산 비용을 줄이면서 두 개의 신경망을 훈련시켰습니다. 이 신경망 모델은 구조적 특성을 예측하는 데 높은 신뢰성을 보였으며, $R^2$ 점수가 0.995를 초과하였습니다.

- **Performance Highlights**: 제안된 프레임워크는 생체모방 메타구조의 설계에서 중요한 발전을 나타내며, 생체 공학 및 재생 의학 분야의 혁신을 위한 기초를 제공합니다. 또한 기계적 특성을 효과적으로 달성하기 위해 패치 디자인을 맞춤화하는 신경망 기반 모델은 전통적인 최적화 방법보다 더 효율적이고 정밀한 설계 솔루션을 제공했습니다. 이는 향후 장기적인 조직 재생 치료의 성공 가능성을 높이는 데 기여할 것으로 기대됩니다.



### An Interpretable ML-based Model for Predicting p-y Curves of Monopile Foundations in Sand (https://arxiv.org/abs/2501.06232)
- **What's New**: 이 연구는 복잡한 파일-토양 상호작용으로 인해 진행하기 어려운 측면 파일 반응 예측에 머신 러닝(ML) 기술을 활용했습니다. 특히, 단일파일(monopile) 기초의 p-y 곡선을 예측하기 위한 해석 가능한 ML 모델을 개발했습니다.

- **Technical Details**: 연구는 기존 연구에서 수집한 데이터베이스를 기반으로 XGBoost 모델을 훈련했습니다. 이 모델은 비선형 분석에 효과적인 접근법을 제공하며, Shapley Additive Explanations (SHAP)를 사용하여 모델의 해석 가능성을 향상시켰습니다.

- **Performance Highlights**: 모델의 예측 정확도는 우수한 성과를 보였으며, 각 변수에 대한 SHAP 값 분포는 기초의 측면 반응에 영향을 미치는 요인에 관한 기존 이론적 지식과 강한 일치를 보였습니다.



### asanAI: In-Browser, No-Code, Offline-First Machine Learning Toolk (https://arxiv.org/abs/2501.06226)
Comments:
          7 pages, 8 figures

- **What's New**: 최근 기계 학습(Machine Learning, ML)에 대한 관심이 높아지고 있지만, 비전문가가 이를 쉽게 이해하고 적용하기 어려운 진입 장벽이 존재합니다. 이 논문에서는 asanAI라는 오프라인 우선의 오픈소스 및 노코드(No-Code) 머신 러닝 툴킷을 소개합니다. asanAI는 사용자가 복잡한 소프트웨어 설치 없이 웹 브라우저에서 직접 ML 모델을 설계하고 디버깅하며 훈련할 수 있는 환경을 제공합니다.

- **Technical Details**: asanAI는 현대 웹 브라우저가 설치된 어떤 장치에서도 작동하며, 특히 스마트폰에서도 사용이 가능합니다. 사용자는 개인 정보 보호를 보장받으면서 로컬 계산을 수행할 수 있으며, GPU 성능을 향상시키기 위해 WebGL을 활용합니다. 이 툴킷은 직관적인 시각화를 통해 네트워크 구조와 데이터 흐름을 쉽게 이해할 수 있도록 지원합니다.

- **Performance Highlights**: asanAI는 연구자들이 신속하게 머신 러닝 아이디어를 초안 및 테스트할 수 있도록 하며, 교육자들이 효과적으로 학습자를 교육하는 데 도움을 줍니다. 또한 교사들이 최신 ML 주제를 최소한의 노력으로 학급에 소개할 수 있게 해줍니다. 오픈소스 MIT 라이선스 하에 배포되어 수정이 가능하고, 산업 사용을 위한 모델 내보내기 형식도 지원하여 다양한 사용자들이 머신 러닝을 효과적으로 배워 활용할 수 있는 기반을 제공합니다.



### Can Explainable AI Assess Personalized Health Risks from Indoor Air Pollution? (https://arxiv.org/abs/2501.06222)
- **What's New**: 본 연구에서는 실내 공기 오염 원인을 정밀하게 파악하고, 개인의 실내 활동이 오염 수준에 미치는 영향을 조사하였습니다. 조사 결과, 실내 공기 오염에 대한 인식이 낮다는 점을 발견하였고, 65일간의 다양한 데이터를 통해 오염원이 발생하는 상황을 분석하였습니다.

- **Technical Details**: 연구의 데이터 수집 과정에서 LIME, SHAP와 같은 해석 가능 모델과 클러스터링 분석을 활용하여, 의사 결정 트리, 랜덤 포레스트, 나이브 베이즈와 SVM 모델을 통합하여 99.8%의 정확도를 달성했습니다. 24시간 연속 데이터를 활용하여 개인 맞춤형 오염 평가가 가능하고, 91%의 정확도로 활동 및 오염 노출을 예측할 수 있습니다.

- **Performance Highlights**: 연구 결과, 실내 활동의 오염 안전 기준 초과 여부를 예측하고, 개인 맞춤형 오염 관리가 가능함을 보여주었습니다. 이를 통해 개인이 실내 공기 오염의 주요 원인과 해당 활동의 영향을 이해하고, 이를 바탕으로 건강한 실내 환경을 조성하는 데 필요한 정보를 제공받을 수 있습니다.



### Optimizing Supply Chain Networks with the Power of Graph Neural Networks (https://arxiv.org/abs/2501.06221)
- **What's New**: 이 연구는 공급망 (supply chain) 네트워크 내에서 Graph Neural Networks (GNNs)를 수요 예측 (demand forecasting)에 적용하는 내용을 다룹니다. SupplyGraph 데이터셋을 활용하여 GNN 방법론을 통해 예측 모델의 정확성을 높이고, 잠재적인 의존성 (latent dependencies)과 공급망 운영의 시간적 복잡성 (temporal complexities)을 탐구합니다. 이 연구는 특히 단일 노드 (single-node) 수요 예측 작업에서 GNN 기반 모델이 전통적인 접근법보다 현저히 뛰어난 성능을 보인다는 점을 강조합니다.

- **Technical Details**: 연구에서 제안하는 SupplyGraph 데이터셋은 방글라데시의 선도적인 빠르게 움직이는 소비재(FMCG) 회사의 실제 운영 데이터를 바탕으로 하며, 공급망 요소를 노드 (nodes)로 모델링하고 이들의 상호 의존성을 엣지 (edges)로 표현합니다. 각 노드는 생산량, 판매 주문, 배송 지표 등 시간적 특징을 포함하여 공급망의 복잡한 구조를 반영합니다. 이 데이터셋은 GNN을 통한 수요 예측과 자원 최적화와 같은 여러 과제를 탐구할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 연구 결과에 따르면, GNN 기반 모델은 멀티레이어 퍼셉트론 (Multilayer Perceptrons, MLPs) 및 그래프 합성곱 네트워크 (Graph Convolutional Networks, GCNs) 같은 전통적 방법보다 뛰어난 성능을 발휘했습니다. 특히, 단일 노드 수요 예측 작업에 있어 GNN은 정확한 예측을 제공하여 재고 관리, 생산 일정 수립 및 운영 효율성을 향상시킬 수 있는 잠재력을 보여줍니다. 이 연구는 GNN의 이론적, 실용적 가치를 입증하며 공급망 관리의 혁신적 기회를 제시합니다.



### Powerful Design of Small Vision Transformer on CIFAR10 (https://arxiv.org/abs/2501.06220)
- **What's New**: 본 논문에서는 작은 데이터셋에서 Vision Transformers (ViTs)의 성능을 개선하기 위해 Tiny ViTs의 설계와 최적화에 대해 다룹니다. CIFAR-10을 벤치마크로 활용하여 데이터 증강(data augmentation), 패치 토큰 초기화(patch token initialization), 저랭크 압축(low-rank compression), 다중 클래스 토큰 전략(multi-class token strategies)의 영향을 체계적으로 평가하였습니다. 저랭크 압축이 성능 손실을 최소화하며, 여러 CLS 토큰을 도입함으로써 모델의 전반적인 표현 능력이 향상된다고 보고합니다.

- **Technical Details**: ViT는 이미지 토큰화(image tokenization), 토큰 변환(token transformation), 과제 투영(task projection)의 주요 절차를 통해 작동합니다. 이미지가 여러 패치로 나눠지고, 각 패치 토큰은 특정 위치의 의미 정보를 나타냅니다. Attention 메커니즘을 통해 각 패치의 중요성을 상대적으로 평가하며, 다중 헤드 주의(multi-head attention)를 사용하여 다양한 관계를 포착합니다.

- **Performance Highlights**: 실험 결과, 저랭크 압축을 사용한 Multi-Head Latent Attention (MLA)은 ViTs에서 다소 중복된 성능을 보여 성능 손실이 거의 없음을 나타냅니다. 또한, 여러 CLS 토큰을 도입함으로써 더 나은 정확도를 달성할 수 있음을 발견했습니다. 이를 통해 Tiny ViTs의 최적화를 위한 포괄적인 프레임워크를 제공하며, 효율적이고 효과적인 설계에 대한 실용적인 통찰을 제시합니다.



### Imagine while Reasoning in Space: Multimodal Visualization-of-Though (https://arxiv.org/abs/2501.07542)
Comments:
          11 pages, 6 figures, 4 tables (27 pages, 10 figures, 16 tables including references and appendices)

- **What's New**: 이번 연구에서 제안한 Multimodal Visualization-of-Thought (MVoT) 방법론은 이미지와 언어를 결합하여 복잡한 사고 과정을 시각적으로 표현할 수 있게 해줍니다. MVoT는 MLLM (Multimodal Large Language Model)이 자연스럽게 시각적 사고를 생성하면서 언어적 사고를 동시에 진행하는 새로운 패러다임을 세웁니다. 이는 기존의 Chain-of-Thought (CoT) 접근 방식이 복잡한 공간적 추론에서 한계가 있었던 점을 극복하는 데 기여합니다. 최종적으로 MVoT는 언어적 추론을 보완할 수 있는 가능성을 열어줍니다.

- **Technical Details**: MVoT는 두 가지 모드 (textual, visual)를 결합하여 다중 모드 생각을 생성하는 프로세스를 정의합니다. 주어진 입력 시퀀스에 대해 모델은 중간 단계에서 이미지 시각화를 추가하여 언어적 사고를 보완합니다. 이는 언어 시퀀스와 이미지 시퀀스를 병행 생성하며, 각 단계는 이전의 단계와 결합하여 더욱 풍부한 데이터를 생성합니다. 실험에서는 Chameleon-7B 모델을 통해 MVoT의 효과를 검증하며, 토큰 불일치 손실(token discrepancy loss)을 도입하여 고품질의 시각화를 보장합니다.

- **Performance Highlights**: MVoT는 여러 동적인 공간 추론 작업에서 경쟁력 있는 성능을 입증하였습니다. Maze, MiniBehavior, FrozenLake와 같은 벤치마크에서 전통적인 CoT 방식을 20% 이상 초과하는 성능을 보였습니다. 특히, MVoT는 CoT가 실패하는 가장 도전적인 시나리오에서도 부분적으로 안정적이며 신뢰할 수 있는 개선을 보여줍니다. 이번 연구는 복잡한 과제에서 시각적 사고의 효용성을 입증하는 중요한 기초 연구로 평가받고 있습니다.



### RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignmen (https://arxiv.org/abs/2501.07525)
- **What's New**: 본 논문에서는 RadAlign이라는 새로운 프레임워크를 제안합니다. RadAlign은 병리학적 진단의 정확성과 해석 가능성을 동시에 중시하면서 방사선 사진 해석과 보고서 생성을 통합합니다. 이 프레임워크는 시각-언어 모델(Vision-Language Models, VLM)과 대형 언어 모델(Large Language Models, LLM)의 장점을 결합하여 의사 소통의 품질을 향상시키고 의료 영상 분석의 신뢰성을 높입니다.

- **Technical Details**: RadAlign은 먼저 VLM을 사용하여 시각적 특징을 의료 개념과 정렬합니다. 이는 방사선과 의사가 특정 진단 기준을 확인하고 이를 바탕으로 판단하는 과정을 모방합니다. 그런 다음 정렬된 시각적-언어 공간에서 저희가 인식한 질병을 텍스트 기반 개념으로 표현하고, 이를 활용해 LLM에 의해 보고서를 생성하도록 유도합니다.

- **Performance Highlights**: RadAlign은 다양한 질병에 대해 평균 AUC가 0.885로 우수한 질병 분류 성능을 달성합니다. 보고서 품질은 GREEN 점수가 0.678로, 현재까지의 최첨단 방법인 0.634를 초과합니다. 이 프레임워크는 임상 해석 가능성을 유지하면서 착각을 줄이고, 통합된 예측 및 생성 AI를 통해 자동화된 의료 영상 및 보고서 분석을 발전시킵니다.



### Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning (https://arxiv.org/abs/2501.07508)
Comments:
          9 pages, 5 figures. Accepted at AI for Social Impact: Bridging Innovations in Finance, Social Media, and Crime Prevention Workshop at AAAI 2025

- **What's New**: 이번 연구는 Uniswap v3의 유동성 제공(liquidity provisioning) 최적화를 위해 심층 강화 학습(deep reinforcement learning, DRL) 기법을 적용한다. 우리는 유동성 제공 작업을 마르코프 결정 과정(Markov Decision Process, MDP)으로 모델링하고, Proximal Policy Optimization (PPO) 알고리즘을 사용하여 능동적인 유동성 제공자(agent)를 교육한다. 본 연구의 목표는 보다 효율적인 유동성 관리(promotion of efficient liquidity management)를 통해 분산 금융(DeFi) 시장을 더 많은 참여자가 접근할 수 있도록 하는 것이다.

- **Technical Details**: 유동성 제공을 불확실성 통제 문제로 설정하고 유동성 제공자의 결정 과정을 RL로 모델링하였다. PPO 알고리즘을 적용하여 시장의 동적 변화에 따라 유동성 포지션을 조정하는 능동적인 유동성 제공자(agent)를 통해 수수료 최대화와 손실 완화를 동시에 고려한다. 이 연구에서 제안한 보상 함수는 수수료 수익을 포함하고, 유동성 포지션 배치에 따른 비용 및 기회 비용을 고려하여 LVR을 패널티로 설정함으로써 소형 소매 트레이더의 입장에 적합한 전략을 개발하고자 한다.

- **Performance Highlights**: 본 연구는 DRL 기반 전략의 성능을 일반적인 휴리스틱 시스템과 비교하여, DRL이 작은 소매 유동성 제공자들이 일반적으로 사용하던 방식에 비해 효과적인 결과를 나타냄을 보여준다. 이 연구는 DeFi 시스템의 안정성(stability)과 전체 가치를 향상시키는 한편, 더 많은 유동성 제공자들이 시장에 참여할 수 있는 여건을 마련하여 지속 가능한 금융 포괄성을 달성하려는 목표를 가지고 있다. 앞으로 DeFi 생태계의 효율성을 높임으로써 전통 금융에 대한 혁신적 대안으로 자리를 잡는데 기여할 것으로 예상된다.



### Synthesis and Analysis of Data as Probability Measures with Entropy-Regularized Optimal Transpor (https://arxiv.org/abs/2501.07446)
Comments:
          58 pages. Code to reproduce experiments: this https URL

- **What's New**: 본 논문에서는 엔트로피 정규화된 Wasserstein-2 비용을 사용한 확률 측정의 합성 및 분석에 관한 문제를 다룬다. 특히, 주어진 m개의 기준 측정에 대해 이 비용에 대한 barycenter를 계산하는 합성 문제와, 주어진 측정에 대해 가장 가까운 barycenter의 계수를 찾는 분석 문제를 정의한다. 이러한 분석은 기존 문헌에서의 가장 약한 가정 하에서도 비용의 도함수를 계산할 수 있도록 한다.

- **Technical Details**: 논문에서는 barycenter의 정규화된 특성을 고정점 방정식(fixed-point equation) 해결책으로서 제시하며, 이는 엔트로피 지도의 평균에 대한 것이다. 연속된 entropic maps를 통해 우리가 얻은 경량의 convex, quadratic program을 통해 분석 문제를 해결할 수 있는 방법이 마련되었다. 엔트로피 정규화된 최적 수송의 특성으로 인해 표본에서 얻은 이러한 좌표는 차원 독립적인 수렴 속도를 보인다.

- **Performance Highlights**: 제안된 barycentric 계수는 Wasserstein-2 메트릭의 pertubation에 대해 안정적인 것으로 입증된다. 이는 이러한 계수가 손상에 강인함을 나타내며, 손상된 포인트 클라우드 데이터에 대한 분류 기능으로 사용된다. 실험 결과, 우리의 접근법이 신경망 기반 방식보다 작은 훈련 데이터 영역에서 더 효율적임을 보인다.



### Pairwise Comparisons without Stochastic Transitivity: Model, Theory and Applications (https://arxiv.org/abs/2501.07437)
Comments:
          34 pages, 1 figure

- **What's New**: 본 논문에서는 페어와이즈 비교(pairwise comparison) 데이터를 위해 기존의 stochastic transitivity 가정을 제거한 새로운 통계 모델 가족을 제안합니다. 기존의 Bradley-Terry (BT) 및 Thurstone 모델에 대한 확장을 통해, 이 모델은 낮은 차원의 skew-symmetric 행렬을 사용하여 페어와이즈 확률을 결정합니다. 또한, 데이터의 희소성을 고려하여 최적의 추정기를 개발하고, 이를 통해 실제 데이터에서도 뛰어난 성능을 보여줍니다.

- **Technical Details**: 제안된 모델은 승리 확률 행렬에 대해 저차원 구조를 가정하여 페어와이즈 비교를 모델링합니다. 이 논문에서는 convex optimization 프로그램을 통해 효율적으로 확인할 수 있는 확률 추정기를 제안하고, 이 추정기는 minimax-optimality를 달성할 수 있음을 입증합니다. 논문에서는 또한 skew-symmetric 행렬에 대한 스펙트럼 이론이 구현 및 이론적 분석에서 중요한 역할을 함을 강조합니다.

- **Performance Highlights**: 시뮬레이션과 실제 데이터 분석을 통해 제안된 방법의 우수성이 입증되었습니다. e-sport인 스타크래프트 II와 프로 테니스 데이터를 포함한 다양한 실제 데이터셋에 적용한 결과, 제안된 모델이 BT 모델에 비해 전반적으로 더 좋은 성능을 발휘하며, 비전이성(intransitivity)이 관찰되는 상황에서도 안정적으로 작동함을 보여줍니다.



### Distance Measure Based on an Embedding of the Manifold of K-Component Gaussian Mixture Models into the Manifold of Symmetric Positive Definite Matrices (https://arxiv.org/abs/2501.07429)
- **What's New**: 이 논문에서는 K-component Gaussian Mixture Model(GMM)과 대칭 양의 정부 행렬의 매니폴드 간의 임베딩을 기반으로 한 새로운 거리 개념이 제안됩니다. K-component GMMs의 매니폴드가 대칭 양의 정부 행렬의 서브 매니폴드임을 증명하고, GMM 매니폴드의 거리 측정 방식의 요약 또한 포함됩니다. 특히, 이 방법은 반드시 닫힌 형식의 유사도 측정을 요구하지 않으며, Fisher-Rao metric의 일반적인 하한을 도출합니다.

- **Technical Details**: 이 연구에서는 GMM의 매니폴드를 대칭 양의 정부 행렬의 매니폴드에 임베딩함으로써 K-component GMM간의 유사성을 보다 효과적으로 측정할 수 있는 방법을 구현합니다. 이 임베딩을 통해 GMM의 기하학적 구조를 보존하면서 K-component의 차원 상승을 활용합니다. 연구 결과, 우리의 방법은 GMM의 매니폴드가 자연적인 리만 기하학적 메트릭을 가진 대칭 양의 정부를 구성함을 증명하였습니다.

- **Performance Highlights**: 실험을 통해 제안된 방법의 효과성이 입증되었으며, UIUC, KTH-TIPS, UMD 텍스처 인식 데이터셋에서 각각 98%, 92%, 93.33%의 정확도를 기록했습니다. 이는 기존의 GMM 기반 유사도 측정보다 상당한 정확도의 개선을 보여줍니다. 이러한 결과들은 이 프레임워크의 이론적 및 실제적 응용 가능성을 검증합니다.



### Simulating the Hubbard Model with Equivariant Normalizing Flows (https://arxiv.org/abs/2501.07371)
Comments:
          14 pages, 5 figures, contribution to the 41st International Symposium on Lattice Field Theory (Lattice 2024), July 28th - August 3rd, 2024, Liverpool, UK

- **What's New**: 이 논문은 normalizing flows가 Hubbard 모델의 Boltzmann 분포를 학습하는 데 사용될 수 있음을 입증하는 개념 검증(proof-of-concept) 데모를 제공합니다. 이 모델은 그래핀 및 기타 탄소 나노물질의 전자 구조 연구에 널리 사용되고 있습니다. 이러한 연구는 statistical mechanics 및 collider physics와 같은 다양한 물리학 분야에서의 발전에 기여합니다.

- **Technical Details**: normalizing flows는 다양한 도메인에서 확률 분포를 학습하는 데 탁월한 성능을 보이고 있습니다. 특히 lattice field theory와 관련하여, normalizing flows를 활용하여 Boltzmann 분포를 정확하게 학습할 수 있습니다. 본 연구에서는 Hybrid Monte Carlo (HMC) 방법을 기반으로 한 Hubbard 모델의 숫자 시뮬레이션이 ergodicity 문제로 어려움을 겪고 있음을 설명합니다.

- **Performance Highlights**: 이 논문의 수치 실험은 normalizing flow에서의 i.i.d. 샘플링을 활용하여 ergodicity 문제를 효과적으로 해결할 수 있음을 보여줍니다. 이러한 접근 방식은 thermodynamic observables의 직접 추정 및 독립 동치 구성(configurations) 샘플링을 가능하게 합니다. 그 결과, 실험은 물리적 관찰치의 편향 추정을 줄이는 데 기여합니다.



### Multimodal semantic retrieval for product search (https://arxiv.org/abs/2501.07365)
- **What's New**: 이 연구에서는 기존의 텍스트 기반 (textual) 의미 검색 (semantic retrieval)에 대한 연구를 확장하여, 전통적인 텍스트 표현과 대조적으로 전자상거래 (e-commerce) 검색에서 제품 항목에 대한 다중 모달 (multimodal) 표현을 구축했습니다. 이런 새로운 접근법은 제품 이미지가 검색 상호작용에 미치는 영향을 분석하며, 고객 제품 탐색 시 중요하다는 점을 강조합니다.

- **Technical Details**: 이번 연구에서는 전자상거래 데이터셋을 기반으로 모델을 개발하고 평가했습니다. 다중 모달 표현을 통해 제품을 나타내는 새로운 방식을 제안하고, 이 표현이 구매 회수 (purchase recall)와 의미 검색의 관련성 정확도 (relevance accuracy)에서 개선점을 나타내는지 조사합니다.

- **Performance Highlights**: 연구 결과, 다중 모달 의미 검색 모델이 텍스트 전용 모델에 비해 독점적으로 검색된 일치 항목에 대한 수치 분석을 통해 성능 개선을 보여주었습니다. 이는 다중 모달 솔루션의 유효성을 입증하는 데 중요한 정보입니다.



### TimberVision: A Multi-Task Dataset and Framework for Log-Component Segmentation and Tracking in Autonomous Forestry Operations (https://arxiv.org/abs/2501.07360)
Comments:
          Accepted at Winter Conference on Applications of Computer Vision (WACV) 2025. Code and dataset available at this https URL

- **What's New**: 본 연구에서는 TimberVision 데이터셋을 소개합니다. 이 데이터셋은 2천 개 이상의 주석이 달린 RGB 이미지로 구성되며, 총 5만 1천 개의 줄기 구성 요소를 포함하고 있어 기존 데이터셋에 비해 규모와 세부 사항에서 큰 차별성을 보입니다. 게다가, 이 데이터셋은 다양한 장면 매개변수들의 모델 성능에 미치는 영향을 분석하기 위한 일련의 ablation 실험을 수행했습니다.

- **Technical Details**: TimberVision 데이터셋은 나무줄기와 그 구성 요소를 다양한 응용 프로그램 시나리오에서 차별화된 학습을 가능하게 하는 구조로 제공됩니다. 데이터셋은 RGB 이미지에서 줄기의 위치와 방향을 유도할 수 있는 방법론을 포함하고 있으며, 이를 통해 로그 배치 처리 및 중량 중심 추정이 가능합니다. 실시간 객체 감지(Oriented Object Detection)와 인스턴스 분할(Instance Segmentation)에 대한 종합적인 연구도 진행되었으며, 이를 통해 여러 가지 장면 배치에 대한 효율성을 극대화했습니다.

- **Performance Highlights**: 최신 알고리즘을 통해 RGB 이미지 데이터만으로도 나무줄기를 높은 정의로 정확하게 표현할 수 있습니다. 이 방식은 도전적인 환경에서도 높은 내구성을 보이며, 여러 센서 모드와 쉽게 결합하여 사용할 수 있는 유연성을 가집니다. 이러한 접근 방식은 나무 자원의 자동화된 수확, 운반, 측정에 적합하여 작업자의 안전을 더욱 높이는 데 기여할 것입니다.



### Code and Pixels: Multi-Modal Contrastive Pre-training for Enhanced Tabular Data Analysis (https://arxiv.org/abs/2501.07304)
- **What's New**: 이번 연구에서는 Multi-task Contrastive Masked Tabular Modeling (MT-CMTM)이라는 혁신적인 방법을 제안하였습니다. 이 방법은 tabular 데이터와 이미지 간의 상관관계를 활용하여 tabular 모델을 개선하는 것을 목표로 하고 있습니다. MT-CMTM은 대조 학습(contrastive learning)과 마스킹된 탭 데이터 모델링(masked tabular modeling)을 결합한 이중 전략을 사용합니다.

- **Technical Details**: MT-CMTM의 핵심은 1D-ResNet-CBAM 구조로, 잔차 연결과 주의 메커니즘(attention mechanism)을 갖춘 1차원 합성곱 신경망(1D Convolutional Neural Network)입니다. 이 모델은 입력 데이터를 효율적으로 처리할 수 있으며, 별도의 이미지 의존성 없이도 downstream 작업을 수행할 수 있게 설계되었습니다. 새로운 HIPMP 데이터세트와 DVM 데이터세트에서 MT-CMTM의 성능이 실험적으로 검증되었습니다.

- **Performance Highlights**: MT-CMTM 모델은 DVM 및 HIPMP 데이터세트에서 우수한 성능을 보였습니다. HIPMP 데이터세트에서 상대 평균 제곱 오차(relative MSE)를 1.48% 개선하고, DVM 데이터세트에서는 절대 정확도를 2.38% 증가시켰습니다. 이러한 결과는 MT-CMTM의 견고성과 다중 모달 학습(multi-modal learning) 분야에 기여할 가능성을 보여줍니다.



### The Lessons of Developing Process Reward Models in Mathematical Reasoning (https://arxiv.org/abs/2501.07301)
- **What's New**: 이 논문에서는 Process Reward Models (PRMs)의 개발에서 발생하는 여러 챌린지를 다룹니다. 기존의 Monte Carlo (MC) 기반 데이터 합성과 LLM-as-a-judge 활용 방법이 PRM의 성능 평가에 있어 단점이 있음을 보여주었습니다. 특히, 기존 BoN (Best-of-N) 평가 방식의 편향과 이를 통한 과정 검증의 비효율성을 지적하고 있습니다.

- **Technical Details**: 연구에서는 MC 추정이 현재 단계의 정확성을 평가하기 위해 완료 모델에 의존하며, 이로 인한 부정확한 단계 검증이 PRM 성능에 부정적인 영향을 미친다고 설명합니다. 또한, PRM이 던지는 응답에 대한 허용 기준이 과도하게 높아 BoN 점수를 비정상적으로 증가시키는 경향이 있다는 점도 짚었습니다. 여기서는 새로운 합의 필터링 메커니즘을 개발하여 MC 추정과 LLM-as-a-judge의 통합을 시도하였습니다.

- **Performance Highlights**: 제안된 방법론은 BoN 평가와 단계별 오류 식별 작업에서 모델 성능과 데이터 효율성을 크게 향상시켰습니다. 최종적으로, 이 논문은 기존의 오픈 소스 PRM보다 더 우수한 성능을 지닌 최신 PRM을 발표하였으며, 향후 과정 감독 모델 개발에 대한 실용적인 지침을 제공합니다.



### Dataset-Agnostic Recommender Systems (https://arxiv.org/abs/2501.07294)
- **What's New**: 추천 시스템(Recommendation Systems)은 개인화된 사용자 경험을 제공하는 데 필수적입니다. 그러나 전통적인 추천 시스템은 데이터 세트에 따라 수동으로 조정해야 하기 때문에 확장성과 재사용성이 제한됩니다. 이를 해결하기 위해 제안된 DAReS(Dataset-Agnostic Recommender Systems)는 데이터 세트 특성에 따라 자동으로 조정되는 새로운 패러다임을 제시합니다.

- **Technical Details**: DAReS의 핵심 요소는 데이터 세트 설명 언어(Dataset Description Language, DsDL)입니다. 이는 데이터 세트의 특성과 레이블에 대한 메타데이터를 제공하는 구조화된 포맷을 갖추고 있습니다. DAReS는 이 정보를 활용해 자동으로 특징 선택, 결측값 처리, 잡음 제거 및 하이퍼파라미터 최적화와 같은 프로세스를 관리합니다.

- **Performance Highlights**: DAReS는 높은 적응성 및 재사용성을 제공하며, 다양한 추천 시나리오에 적용할 수 있는 가능성을 내포하고 있습니다. 하지만 계산적 오버헤드와 데이터 세트 특정 최적화의 제한이 있으며, 이러한 한계를 극복하는 것이 중요합니다. DAReS는 사용자가 직접 코드를 수정할 필요 없이 추천 시스템 구축을 자동화하여 사용자 맞춤형 솔루션을 제공합니다.



### Estimating quantum relative entropies on quantum computers (https://arxiv.org/abs/2501.07292)
Comments:
          24 pages, 10 figures; comments are welcome

- **What's New**: 본 논문에서는 두 개의 알 수 없는 양자 상태로부터 양자 상대 엔트로피(quantum relative entropy)와 Petz Rényi 다이버전스(Petz Rényi divergence)를 추정하기 위한 최초의 양자 알고리즘을 제안합니다. 이 알고리즘은 상대 엔트로피의 사변적 근사(quadrature approximations)와 새로운 파라미터화 기법을 조합하여 제시됩니다. 특히, 이 접근법은 분산 양자 컴퓨팅 시나리오에서도 직접적으로 적용이 가능하다는 장점이 있습니다.

- **Technical Details**: 제안된 양자 알고리즘은 양자 f-다이버전스(𝑓-divergences)의 사변적 표현을 활용하여, 특정 양자 상태에 대한 추정 값을 평가하는 변형 양자 알고리즘(VQA)을 개발합니다. 이 방법은 고전-양자 하이브리드 프레임워크를 통해 확률 분포를 양자 컴퓨터에서 샘플링하고, 그 후 고전 통계량을 처리하는 방식입니다. 이로 인해 원래의 밀도 행렬(density matrix)을 재구성할 필요가 없으며, 이는 계산 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 알고리즘의 유효성과 효율성은 수치 시뮬레이션을 통해 검증되었으며, 기본적인 최적화 방법인 경량 길게조정(gradient descent)에서도 훈련이 가능합니다. 손실 함수(Loss Function)의 변동에 따라 학습률을 동적으로 조정함으로써, 상대 오차율이 약 2%로 효율적인 수렴을 달성하였습니다. 이러한 결과는 양자 하드웨어 장치에 이 방법을 배포하기 위한 기초를 마련합니다.



### Bridging Smart Meter Gaps: A Benchmark of Statistical, Machine Learning and Time Series Foundation Models for Data Imputation (https://arxiv.org/abs/2501.07276)
- **What's New**: 이 논문은 스마트 그리드에서의 시간 시계열 데이터의 결함을 해결하기 위해 Generative Artificial Intelligence (GenAI)와 빅데이터 분석 기법을 접목한 내용입니다. 특히 스마트 미터 데이터의 결함을 보완하기 위해 두 가지 대규모 언어 모델(LLMs)과 다섯 가지 시간 시리즈 기초 모델(Time Series Foundation Models, TSFMs)을 평가합니다. 기존의 기계 학습 모델과 통계 모델과 비교하며, 이러한 접근법이 전통적인 방법보다 더 우수한 성능을 발휘할 수 있음을 제시합니다.

- **Technical Details**: 연구는 2013년 런던의 가정용 에너지 소비 데이터를 포함한 공개 데이터셋을 사용하였습니다. 시간 간격은 30분 단위로 측정되며, 결측값을 유도하기 위해 인위적인 간극을 생성하였습니다. 모델 생성을 위해 다양한 통계 모델과 기계 학습 모델, LLM 및 TSFM을 선택하고 평가하는 방식을 사용했습니다. 이를 통해 각 모델의 성능과 신뢰성을 평가하였습니다.

- **Performance Highlights**: 결과적으로, TSFM은 특정 경우에 임퓨테이션 정확도를 크게 향상시킬 수 있는 가능성을 보여주었습니다. 그러나 계산 비용과 성능 향상 간의 트레이드오프는 여전히 중요한 고려사항으로 남아 있습니다. 이 연구는 결측값 처리와 예측 모델링의 새로운 접근법을 제시하며, 스마트 미터 데이터의 결함 보완에 대한 기존 연구와 차별화됩니다.



### Interpretable machine-learning for predicting molecular weight of PLA based on artificial bee colony optimization algorithm and adaptive neurofuzzy inference system (https://arxiv.org/abs/2501.07247)
- **What's New**: 이번 연구는 인공 벌집 집단 알고리즘(Artificial Bee Colony, ABC)을 인공 신경망(Artificial Neural Networks, ANNs) 및 적응형 네트워크 기반 퍼지 추론 시스템(Adaptive Network-based Fuzzy Inference System, ANFIS)과 통합하여 의약용 폴리락타이드(Polylactic Acid, PLA)의 분자량을 예측하기 위한 특징을 선택하는 방법을 제시합니다. 특히, 실시간으로 NIR(근적외선) 스펙트럼을 캡처하며 PLA의 압출 처리 제조 과정 중에 얻어진 데이터를 활용했습니다.

- **Technical Details**: 이 연구에서는 63개의 관측치와 512개의 입력 특징을 포함한 데이터셋을 사용하였습니다. ABC 최적화 알고리즘은 ANN 및 ANFIS와 결합되어 PLA의 분자량을 예측하며, 목표 함수는 실험적 및 예측된 PLA 분자량 사이의 제곱 평균 근 오류(RMSE)를 최소화하고 입력 특징의 수를 줄이는 것입니다.

- **Performance Highlights**: ABC-ANFIS를 사용한 결과, RMSE 282 Da를 기록하였으며, 4개의 중요한 파라미터(6158 cm-1, 6310 cm-1, 6349 cm-1의 NIR 파장과 용융 온도)를 식별하였습니다. 이 연구는 ABC 알고리즘과 ANFIS를 결합하여 고도의 정확도로 PLA 분자량을 예측하기 위한 최소 입력 특징 집합 선택의 효율성을 입증하였습니다.



### Lung Cancer detection using Deep Learning (https://arxiv.org/abs/2501.07197)
- **What's New**: 이번 논문에서는 Convolutional Neural Networks (CNN)와 Support Vector Machines (SVM)을 결합한 하이브리드 모델을 사용하여 폐암을 조기 진단하는 방법에 대해 논의합니다. 이 모델은 CT 스캔(computed tomography scans) 이미지를 데이터셋으로 활용하여 종양의 양성과 악성을 구별하는 데 도움을 줍니다.

- **Technical Details**: 하이브리드 모델은 CNN과 SVM의 장점을 결합하여 다양한 패턴 인식을 통해 폐암을 조기 탐지하는 데 혁신적인 접근법을 제시합니다. 데이터는 Computed Tomography(CT) 스캔을 기반으로 하며, 심층 학습(deep learning) 기술이 적용되었습니다.

- **Performance Highlights**: 이 연구는 폐암 조기 발견을 위한 최첨단 방법으로, 고성능의 정확한 진단을 가능하게 합니다. 이를 통해 환자의 생존율을 향상시킬 수 있는 잠재력을 지닌다고 평가됩니다.



### Pre-Trained Large Language Model Based Remaining Useful Life Transfer Prediction of Bearing (https://arxiv.org/abs/2501.07191)
- **What's New**: 이 논문은 회전 기계(예: 베어링)의 남은 유용 수명(Remaining Useful Life, RUL)을 정확하게 예측하기 위한 새로운 접근 방식을 제시합니다. 기존의 전통적인 데이터 기반 심층 학습(deep learning) 방법이 직면했던 문제들을 해결하고자 합니다. 이러한 문제는 일관성이 없는 훈련 및 테스트 데이터 분포와 장기 예측에 대한 제한된 일반화로 요약될 수 있습니다.

- **Technical Details**: 저자들은 최신 알고리즘 및 모델을 활용하여 RUL 예측의 신뢰성을 높이기 위한 방법론을 개발합니다. 이를 통해 기계의 상태를 적절히 평가하고, 예측 정확도를 향상시키는 데 중점을 두고 있습니다. 기존 방법론들에 비해 보다 효율적인 데이터 활용이 핵심적인 접근으로 설명됩니다.

- **Performance Highlights**: 이 연구는 새로운 모델이 기존의 예측 방법보다 더 나은 성능을 보임을 입증하고 있습니다. 예측 정확도와 신뢰성을 높임으로써 산업에서의 예상치 못한 고장을 줄이고, 장비의 신뢰성을 증가시키는 데 기여할 수 있습니다. 이를 통해 RUL 예측 분야에서의 기술적인 혁신을 기대할 수 있습니다.



### Uncertainty Guarantees on Automated Precision Weeding using Conformal Prediction (https://arxiv.org/abs/2501.07185)
- **What's New**: 본 논문에서는 정밀 농업 분야에서의 트러스트 구축을 위한 새로운 접근 방식으로, 블랙 박스 모델에 대한 예측 보증을 제공하는 기능을 소개합니다. 기계 학습 커뮤니티에서 확립된 방법론인 "Conformal Prediction" (컨포멀 예측)이 정밀 제초 작업에 적합하게 적용된다는 점이 핵심입니다. 이 방법론은 사용자가 예측의 품질에 대한 신뢰를 갖도록 해주며, 농업 시스템의 채택을 촉진할 수 있을 것으로 기대됩니다.

- **Technical Details**: 컨포멀 예측은 정확한 예측의 보장을 위해 강력한 불확실성을 제어하는 프레임워크입니다. 이 방법은 포인트 예측(orginal predictions) 대신 예측 집합(predictive sets) 또는 간격 예측(interval predictions)을 생성하여 진정한 값의 포함을 보장합니다. 본 연구에서는 유럽에서 수집한 잡초 및 농작물 이미지 데이터베이스를 활용하여, 컨포멀 예측 기반의 정밀 분사 파이프라인을 개발하였습니다.

- **Performance Highlights**: 실험 결과, 개발한 파이프라인은 최소 90%의 잡초를 정확하게 감지할 수 있는 능력을 보여주었습니다. 두 가지 현실 세계 시나리오(데이터의 분포 조건에 따른 실험)에서 평가되었으며, 그 결과는 농업 기술 분야에서 신뢰할 수 있는 모델의 개발 가능성을 시사합니다. 이러한 결과는 정밀 농업 시스템의 효율성을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### A User's Guide to $\texttt{KSig}$: GPU-Accelerated Computation of the Signature Kern (https://arxiv.org/abs/2501.07145)
- **What's New**: 이번 연구에서는 KSig이라는 Scikit-Learn 호환의 Python 라이브러리를 소개합니다. 이 라이브러리는 GPU 가속화된 signature kernel 알고리즘을 구현하고 있으며, Random Fourier Signature Features (RFSF)와 같은 확장 가능한 변형을 포함하여 대규모 시계열 데이터셋의 머신러닝 문제에 실질적인 응용을 가능하게 합니다.

- **Technical Details**: signature kernel은 입력 데이터 도메인에서 정의되는 positive definite이고 대칭인 kernel입니다. 이 kernel은 시퀀스 데이터에 대해 정의되는 또 다른 positive definite kernel로 변환됩니다. 전체 하이퍼파라미터 세트는 원본 kernel의 하이퍼파라미터 집합을 포함하며, 추가 하이퍼파라미터로는 truncation level, preprocessing options, algebraic structure 등이 있습니다.

- **Performance Highlights**: Ksig 라이브러리는 GPU 가속을 통해 signature kernel을 계산하고, 하전된 경험적 성능을 제공하도록 설계되었습니다. 여러 하이퍼파라미터에 대한 최적화가 성능 개선에 크게 기여할 수 있으며. 연구에서는 기존 알고리즘 대비 우수한 성능을 제공하는 새로운 tensor sketches 기반 알고리즘도 소개됩니다.



### Inferring Interpretable Models of Fragmentation Functions using Symbolic Regression (https://arxiv.org/abs/2501.07123)
- **What's New**: 이 논문은 기계학습(Machine Learning)을 사용하여 실험 데이터에서 분열 함수(fragmentation functions)의 기능적 형태를 직접 추론한 최초의 연구를 보고합니다. 분열 함수는 강 상호작용을 설명하는 중요한 요소로, 이 연구는 기존의 가정된 함수 형태에 의존하지 않고 데이터로부터 직접 학습합니다. 특히, 상징 회귀(symbolic regression)라는 ML 기법을 활용하여 데이터를 기반으로 분석적 모델을 학습시킴으로써, 현대 AI 시대에 적합한 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: 분열 함수는 다양한 고에너지 물리학(High Energy Physics) 프로세스에서 하드론 생산 단면(section)을 설명하는데 필요한 핵심 요소입니다. 고전적인 방법론은 주로 글로벌 QCD 적합(global QCD fits)을 통해 데이터를 피팅하는 방식으로 기능적 형태를 추정합니다. 그러나 이 연구에서는 상징 회귀 기법을 통해 하드론의 다수성을 측정하여 분석적 모델을 학습하게 되며, 이는 과학적 발견 도구로서의 잠재력을 보여줍니다.

- **Performance Highlights**: 상징 회귀는 실험 데이터를 직접적으로 사용할 때 높은 효율성을 나타내며, 분열 함수의 기능적 형태를 검증하는 데 적합한 후보로 자리 잡고 있습니다. 이 연구는 전통적인 글로벌 QCD 적합 방법을 보완하며, 데이터에서 제시되는 정보를 기존 기능적 형태와 비교하는 역할을 수행합니다. 실제 실험 데이터를 사용함으로써, 상징 회귀의 신뢰성을 테스트하고 물리학 응용에서의 가능성을 탐색하는 중요한 연구입니다.



### SFC-GAN: A Generative Adversarial Network for Brain Functional and Structural Connectome Translation (https://arxiv.org/abs/2501.07055)
Comments:
          5 pages, 2 figures

- **What's New**: 이 연구에서는 구조적 연결(connectivity)과 기능적 연결의 이원적(translational) 관계를 탐구하는 SFC-GAN(Structural-Functional Connectivity GAN)을 제안합니다. 기존의 연구는 주로 한쪽 방향의 예측에 집중하였으나, 본 연구에서는 CycleGAN 아키텍처를 활용하여 양방향 번역을 가능하게 합니다. 이를 통해 연결체(connectome)의 전체 및 지역 패턴을 포착하고, 네트워크의 대칭성을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: SFC-GAN은 두 개의 생성기(GSC 및 GFC)와 두 개의 판별기(DSC 및 DFC)로 구성되어 있으며, 이들은 connectome의 토폴로지(topology)를 보존하면서 각각의 FC와 SC 쌍 간의 변환을 수행합니다. 일반적으로, fMRI에서 도출된 통계적 관계를 FC로, dMRI에서 유도된 백질 경로를 SC로 사용하여 상호 변환하는 방향성을 가집니다. 이를 통해 상호 작용하는 구조 및 기능적 데이터의 관계를 모델링함으로써 동시에 존재하지 않는 연결체에 대한 유용한 정보를 제공합니다.

- **Performance Highlights**: SFC-GAN은 기존의 모델보다 SC와 FC 간의 변환에서 높은 유사도를 보이며, 기초 모델들보다 그래프 속성 평가에서 탁월한 성능을 입증하였습니다. 각 변환된 모달리티는 다운스트림 분류 작업에 효과적으로 활용될 수 있어, 보건 및 신경학적 장애 진단의 정확도를 크게 향상시킬 수 있습니다. 이를 통해 구조적 및 기능적 연결체 간의 복잡한 관계를 해석하는 데 중요한 도구가 될 수 있습니다.



### Differentially Private Kernelized Contextual Bandits (https://arxiv.org/abs/2501.07046)
- **What's New**: 이 논문은 변별력을 고려한 커널 밴디트 문제에 대해 다루고 있으며, 주목할 점은 차별적 프라이버시(differential privacy)가 적용된다는 것입니다. 저자들은 유명한 복원 커널 힐베르트 공간(Reproducing Kernel Hilbert Space, RKHS)에 속한 보상 함수를 사용하여 알고리즘을 제안하였으며, 제안된 알고리즘은 선형 모델보다 더 많은 모델링 능력을 제공합니다. 이는 개인 정보 보호와 유틸리티를 함께 고려한 접근을 통해 더 나은 학습 성능을 달성합니다.

- **Technical Details**: 본 논문은 주어진 커널 패밀리의 경우 특정한 보상 함수 및 맥락에 대한 차별적 프라이버시 제약 조건 하에서 이러한 커널 밴디트 문제를 해결하기 위한 알고리즘을 제안합니다. 이 알고리즘은 $	ext{O}igg(rac{	ext{γ}_T}{T}+rac{	ext{γ}_T}{T	ext{ε}}igg)$의 오류율을 달성함으로써, 고유한 사양에 따라 프라이버시와 유틸리티의 균형을 유지합니다. 특히 매트른(Matern) 커널과 제곱 지수(Square exponential) 커널과 같은 일반적인 커널 가족을 포함한 알고리즘 이론적으로 보장됩니다.

- **Performance Highlights**: 새롭게 제안된 알고리즘은 기존의 방법보다 어드밴티지를 가지며, 다양한 커널 가족에 대해 우수한 성능을 보여줍니다. 실험 결과는 알고리즘이 적절한 프라이버시 파라미터에서 차별적 프라이버시를 확보하면서도 오류율이 수렴함을 나타내고 있습니다. 이러한 성능 개선은 특히 사용자 개인 정보 보호가 중요한 응용 프로그램에서 더욱 중요한 의미를 가집니다.



### Protego: Detecting Adversarial Examples for Vision Transformers via Intrinsic Capabilities (https://arxiv.org/abs/2501.07044)
Comments:
          Accepted by IEEE MetaCom 2024

- **What's New**: 이 논문에서는 Vision Transformer(ViT) 모델의 취약성을 드러내기 위해 6개의 일반적인 adversarial attack 방법의 공격 능력을 조사합니다. 연구는 adversarial attack에 대한 기존 접근법의 한계를 극복하고, 새로운 탐지 프레임워크인 Protego를 제안하여 ViT 모델의 공격 탐지 효율성을 높이고자 합니다. Protego는 다양한 attack 전략에 대처할 수 있는 강력한 기능을 갖춘 탐지기를 목표로 합니다.

- **Technical Details**: Protego 프레임워크는 ViT의 구조적 정보를 활용하여 이미지를 패치로 나눈 후, Transformer encoder를 통해 패치 임베딩을 수행합니다. 이 과정에서 생성된 최종 특징 출력을 바탕으로 adversarial 예와 정상 예 간의 비일치성을 찾아내어 경량의 플러그인 탐지기를 훈련합니다. 이 탐지기는 입력의 피처 정보를 효과적으로 캡처할 수 있는 적절한 상호작용 메커니즘을 형성하여 공격 예제를 식별합니다.

- **Performance Highlights**: 실험 결과, Protego의 AUC 점수는 0.95를 초과하며, 기존의 탐지 방법들보다 우수한 성능을 보여줍니다. 이 연구는 Metaverse 보안 분야에서의 탐구를 진전시킬 수 있는 잠재력을 지니고 있습니다. ViT 모델의 adversarial example 탐지에 대한 연구 기여와 함께, Protego는 다양한 Transformer 구조에 보편적으로 적용 가능한 기능 추출 기법을 발전시켰습니다.



### Erasing Noise in Signal Detection with Diffusion Model: From Theory to Application (https://arxiv.org/abs/2501.07030)
- **What's New**: 본 논문에서는 denoise diffusion model (DM)을 기반으로 한 새로운 신호 탐지 방법을 제안합니다. 이 방법은 오랫동안 최적의 신호 탐지 기법으로 여겨진 최대 우도 추정법 (ML)에 비해 우수한 성능을 자랑합니다. 또한 Stochastic Differential Equations (SDEs)을 활용한 이론적 기초를 제시하며, 수신 신호에서의 가우시안 잡음을 효과적으로 줄일 수 있음을 보여줍니다.

- **Technical Details**: 제안된 DM 기반 신호 탐지 방법은 diffusion transformer (DiT)를 기반으로 하며, 계산 복잡도는 𝒪⁢(n²)로 설계되었습니다. 이 방법은 수신 신호의 다양한 신호 대 잡음 비율 (SNR)에서 효과적으로 작동할 수 있도록 수학적인 스케일링 기법을 도입하여, 신호 탐지의 일반화 문제를 해결합니다. 또한, 분산 변환기를 통해 수신 신호를 여러 토큰으로 나누어 입력으로 사용함으로써 계산 효율성을 높였습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 DM 기반 신호 탐지 방법은 BPSK 및 4QAM 변조 방식에서 ML 추정 방법보다 월등히 낮은 기호 오류율 (SER)을 달성하는 것으로 나타났습니다. 이는 기존의 신호 탐지 이론에 중대한 혁신을 가져오는 성과로, 전통적인 최적 신호 탐지 기법의 성능 한계를 넘어서는 가능성을 보여줍니다.



### Improved Regret Bounds for Online Fair Division with Bandit Learning (https://arxiv.org/abs/2501.07022)
- **What's New**: 이번 연구에서는 알려지지 않은 평균을 가진 플레이어의 가치로 구성된 온라인 공정 분배(online fair division) 문제에 대해 다룹니다. 이 문제는 한정된 종류의 아이템이 무작위로 배분되는 과정에서 공정성을 높이는 것을 목표로 합니다. 연구팀은 특히, 의사 결정 과정에서 시간이 지남에 따라 성능을 높일 수 있는 알고리즘에 대해 설명합니다.

- **Technical Details**: 알고리즘은 upper confidence bound (UCB) 접근법과 두 번의 선형 최적화(linear optimization) 라운드를 사용하는 방식으로 설계되었습니다. 각 플레이어의 가치가 아직 알려지지 않은 상황에서도 각 단계에서 기대값에 대한 비례성(proportionality)을 충족할 수 있도록 고안되었습니다. 이를 통해 연구팀은 미지의 평균값을 학습하며 각 플레이어에게 공정하게 자원을 배분할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 제안된 알고리즘은 $	ilde{O}(	ext{sqrt}(T))$의 레그렛(regret) 값으로, 이전 알고리즘의 $	ilde{O}(T^{2/3})$에 비해 상당한 개선을 보입니다. 이 결과는 비례성 제약 조건을 유지하면서 높은 확률로 성과를 달성할 수 있음을 보여줍니다. 또한 jealousy가 없는 분배에 대한 불가능성을 시사하며, 비례성 유지의 어려움과 envy-freeness 간의 본질적인 차이를 강조합니다.



### Global Search for Optimal Low Thrust Spacecraft Trajectories using Diffusion Models and the Indirect Method (https://arxiv.org/abs/2501.07005)
- **What's New**: 이번 연구에서는 미션 매개변수가 변화하는 우주선 궤적 최적화 문제의 글로벌 탐색을 가속하기 위해 생성적 기계 학습 모델과 최신 확산 모델(diffusion models)를 통합한 프레임워크를 제안합니다. 이 연구는 우주 임무 설계 단계에서 연료 소비와 비행 시간을 최소화하는 복잡한 문제를 해결하기 위한 새로운 접근 방식을 제공합니다. 제안된 방법은 두 가지 저추력 전이 문제에 대해 테스트되었으며, 특히 ESA와 JAXA의 BepiColombo 우주선과 같은 실제 사례를 반영하고 있습니다.

- **Technical Details**: 문제는 우주선의 초기 및 최종 상태 조건을 만족하면서 성능 측정을 최소화하는 궤적을 결정하는 복잡한 비선형 최적 제어 문제입니다. 이 연구에서 제시된 최적화의 목표는 propellant(추진제)의 양을 최소화하여 임무를 수행하는 데 필요한 propellant을 감소시키는 것입니다. 또한, 비행 시간이 지나치게 지연되지 않도록 상한선을 두어 최적화 과정에서 시간이 과도하게 소비되는 것을 방지합니다.

- **Performance Highlights**: 확산 모델은 새로운 조건에서의 초기 상태를 예측하는 데에 성공적으로 사용되어, 기존의 균일 분포나 인접 제어 변환에서 샘플링한 것에 비해 새로운 추진력의 경우 1-2 배 더 많은 해를 생성할 수 있도록 하였습니다. 결과적으로, 글로벌 탐색 과정의 시간 절약과 솔루션 품질의 향상을 도모했으며, 연구 결과는 저추력 전이에 대한 효율적인 해결책을 제시합니다.



### Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning (https://arxiv.org/abs/2501.06994)
- **What's New**: 이 연구에서는 로봇이 자율적으로 일상적인 작업을 완료할 수 있도록 하기 위한 새로운 접근 방식을 제안합니다. 기존의 Imitation Learning (IL) 방식의 한계를 극복하기 위해, 우리는 동작을 짧은 시간 지평선의 2D 궤적으로 표현하는 방법을 도입하였습니다. Motion Track Policy (MT-π𝜋	extit{pi})는 이 매우 적은 양의 데이터만으로 임무를 성공적으로 수행할 수 있는 새로운 방식으로, 인식된 2D 궤적을 통해 로봇과 인간의 동작을 통합합니다.

- **Technical Details**: MT-π𝜋	extit{pi} 정책은 이미지 관찰을 입력으로 받고 동작을 궤적 형태로 출력합니다. 이 정책은 인간 손과 로봇 말단 장치의 동작을 2D 궤적으로 변환하여, 필요한 데이터는 약 10분의 인간 비디오와 몇 개의 로봇 시연으로 최소화합니다. 테스트 시간 동안 두 개의 카메라 뷰에서 궤적을 예측하고, 다중 뷰 기하학을 통해 6DoF(6 Degrees of Freedom) 궤적을 재구성합니다.

- **Performance Highlights**: MT-π𝜋	extit{pi}는 4개의 실제 작업에서 평균 86.5%의 성공률을 달성하였으며, 이는 기존 IL 방식에 비해 40% 개선된 수치입니다. 이 정책은 인간 비디오에서만 포착된 새로운 시나리오로도 일반화하여, 오픈 소스 데이터셋으로 배포 가능하다는 점이 특징입니다. 연구팀은 코드와 비디오를 웹사이트에서 제공하여 재현 가능한 훈련과 실제 환경에서의 배포를 지원합니다.



### Sanidha: A Studio Quality Multi-Modal Dataset for Carnatic Music (https://arxiv.org/abs/2501.06959)
Comments:
          Accepted to the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 본 논문에서는 Carnatic 음악을 위한 첫 번째 오픈소스 데이터셋인 'Sanidha'를 소개합니다. 이 데이터셋은 스튜디오 품질의 멀티트랙 녹음으로, 최소한의 오버랩과 블리드(bleed)를 제공하여 소스 분리(source separation) 과제를 위한 새로운 가능성을 제시합니다. 또한 기존의 다중 트랙 Carnatic 데이터셋과 비교했을 때 개선된 SDR(신호 왜곡 비율) 성능을 보여, Spleeter 모델이 잘 조정되었다는 점을 강조합니다.

- **Technical Details**: Carnatic 음악은 북인도 음악과 마찬가지로 라이브 전통이기 때문에 녹음 시 여러 악기 및 청중의 소리가 혼합되어 출력됩니다. 이러한 혼합은 마이크에서 수직적으로 발생하는 블리드를 유발하여 소스 분리가 매우 어려워지게 만듭니다. 그 결과, 현재 상용화된 소스 분리 모델을 사용할 수 없으며, 이로 인해 Carnatic 음악에 대한 기존 데이터셋은 신뢰할 수 없는 결과를 초래합니다.

- **Performance Highlights**: Sanidha 데이터셋을 통해 학습된 Spleeter 모델은 새로운 듣기 연구(listening study)를 통해 평가되었으며, 결과적으로 소스 분리 성능이 향상되었습니다. 이러한 결과는 Carnatic 음악을 위한 데이터의 부족 문제를 해결함으로써 향후 연구에 유용한 기초 자료를 제공할 것입니다. 이는 특히 수년 간 소스 분리 모델 개발에 큰 저해요소로 작용해온 기존 데이터셋의 한계를 극복할 가능성을 가지고 있습니다.



### Super-Resolution of 3D Micro-CT Images Using Generative Adversarial Networks: Enhancing Resolution and Segmentation Accuracy (https://arxiv.org/abs/2501.06939)
Comments:
          24 pages, 9 figures

- **What's New**: 본 연구에서는 머신 러닝(Machine Learning) 생성 모델을 활용하여 암석의 3D 마이크로 전산 단층 촬영(micro-CT) 이미지 품질을 크게 향상시키는 절차를 개발했습니다. 제안된 모델은 해상도를 8배(8x) 향상시키며, 서로 다른 암석 광물 및 상에 대한 중첩된 X-ray 감쇠로 인한 세분화 불일치를 해결합니다.

- **Technical Details**: 이 생성 모델은 그래디언트 패널티(Gradient Penalty)를 포함한 3D 깊이 합성곱 와서슈타인 생성 적대 신경망(3D DC WGAN-GP)입니다. 이 알고리즘은 세분화된 3D 저해상도 micro-CT 이미지와 세분화된 쌍이 없는 고해상도 레이저 스캐닝 현미경(LSM) 이미지를 기반으로 훈련되었습니다. 여러 견본 샘플에 대해 알고리즘을 검증하였습니다.

- **Performance Highlights**: 최종적으로 우리는 0.4375 마이크로 미터/부피(voxel)의 해상도를 가진 고품질의 초해상도 3D 이미지를 달성했으며, 구성 광물과 기공 공간에 대한 정확한 세분화를 구현하였습니다. 이 절차는 디지털 암석 물리학의 현대적 역량을 크게 확장할 수 있는 가능성을 보여줍니다.



### Harnessing Large Language Models for Disaster Management: A Survey (https://arxiv.org/abs/2501.06932)
- **What's New**: 본 논문은 자연재해 관리에 사용되는 대형 언어 모델(LLMs)에 대한 종합적인 조사를 제공합니다. 기존의 연구가 부족했던 분야인 자연재해 관리에 대한 LLMs의 체계적인 리뷰를 통해 재난 관리의 네 가지 단계(완화, 준비, 대응, 회복)에 따른 LLMs의 응용을 정리하였습니다. 저자는 또한 다양한 작업과 응용 시나리오에 따라 LLMs를 분류하는 새로운 분류체계를 제안합니다.

- **Technical Details**: 재난 관리의 네 가지 단계는 사고 예방을 위한 위험 식별, 재난 발생에 대한 준비 상태의 확립, 즉각적인 대응 및 복구 과정을 포함합니다. 이는 사회적 및 기술적 도구의 동원 및 공급을 통해 이루어지며, LLM들은 다양한 모달리티의 데이터를 처리하여 실시간 데이터 분석을 지원합니다. 이 과정에서 encoder-based LLM, decoder-based LLM, 그리고 멀티모달 LLM이 LLM 아키텍처로 구분되어 연구되고 있습니다.

- **Performance Highlights**: LLMs는 재난 관리에서 여러 신뢰할 수 있는 수행 결과를 보여줍니다. 예를 들어, decoder 기반 모델은 커뮤니티의 취약성 관련 질문에 대한 답변 생성을 통해 효과적인 자원 할당을 지원합니다. 또한, 향상된 공공 인식을 위한 지식 추출 및 예측 기능을 통해 재난 대응 및 경고 생성에서 중요한 역할을 수행하고 있습니다.



### Automatic Double Reinforcement Learning in Semiparametric Markov Decision Processes with Applications to Long-Term Causal Inferenc (https://arxiv.org/abs/2501.06926)
- **What's New**: 이번 연구에서는 Double Reinforcement Learning (DRL)의 요구 사항인 상태 분포 간의 격렬한 중첩(overlap)을 완화하는 새로운 방법을 제시합니다. 특히 무한 지평선(infinite-horizon)과 시간 불변(time-invariant) 마르코프 결정 과정(Markov Decision Process, MDP)에서의 Q-function의 선형 함수(linear functional)에 대한 효율적인 추정을 연구하였습니다. 이 연구는 특히 도메인 적응(domain adaptation) 하에서의 장기 가치(long-term value) 평가에 중점을 두고 있으며, 이는 새로운 도메인의 짧은 궤적(trajectory)으로부터 정보량을 증가시킬 수 있습니다.

- **Technical Details**: Q-function에 대한 세미파라메트릭(semiparametric) 제약 조건을 도입하여 효율적인 추정을 가능하게 하는 새로운 프레임워크를 마련하였습니다. 이 방법은 적응형 디바이징 머신러닝(adaptive debiased machine learning, ADML) 프레임워크를 확장하여 Q-function의 함수적 형태에 적응하는 비모수적으로 유효(nonparametrically valid)하고 매우 효율적인 추정기를 구성하는 방법을 포함합니다. 또한, 컴퓨팅의 어려움을 해결하기 위해 이소토닉 보정(isotonic calibration)된 적합 Q-iteration을 활용한 새로운 디바이즈드 플러그인 추정기를 제안합니다.

- **Performance Highlights**: 최신 알고리즘을 통해 자원 추정의 변동성을 줄이고 더 타이트한 신뢰 구간(confidence interval)을 제공하는데 성공하였습니다. 연구의 결과는 엄격한 중첩 요구 조건을 완화함으로써 장기적인 정책 효과를 보다 정확하게 추정할 수 있음을 보여줍니다. 이러한 방법은 다양한 산업에서 단기 실험의 결과를 장기적인 실적과 연결할 수 있는 가능성을 제시하며, 실제 응용에 잘 맞는 비모수적(nonparametric) 접근 방식을 기반으로 하고 있습니다.



### Optimal Online Bookmaking for Binary Games (https://arxiv.org/abs/2501.06923)
- **What's New**: 이 논문은 온라인 베팅의 문제를 분석하며, bookmakers가 특정 이벤트에 제공하는 베팅의 payoffs를 업데이트하는 방법에 대해 설명합니다. 저자들은 	extit{Optimal Online Bookmaking game}으로 문제를 공식화하고, 이 문제에 대한 최적의 해결책을 제공합니다. 특히 binary case에서 최적 베팅 전략을 도출하며, 이는 bi-balancing trees라는 새로운 기법에 기반합니다.

- **Technical Details**: 실험 I는 m개의 가능한 결과를 가진 이산 실험으로, 각 결과에 대해 gambler는 돈을 투자할 수 있습니다. 이 베팅 시스템에서는 gambler가 선택한 outcome i에 대해 $1을 투자할 경우, 베팅 결과가 I=i일 때 γ(i)만큼 반환받으며, γ(i)는 1 이상이어야 합니다. bookmakers의 관점에서 최악의 시나리오는 gambler의 베팅분포(q)와 payoffs(r)에 따라 최대 손실을 겪는 상황이며, 이를 최소화하기 위한 전략이 논의됩니다.

- **Performance Highlights**: 최적의 전략을 성공적으로 적용할 경우, bookmaker는 이익을 1 - 1/Γ로 확보할 수 있으며, Γ가 1보다 클 경우 긍정적인 이익을 실현할 수 있습니다. 그러나 단일 gambler가 q=r로 예산을 분배하는 것은 손실을 초래하기 때문에 유도되지 않는다는 점이 강조됩니다. 따라서 이 연구는 bookmaker가 위험을 최소화하면서도 최적의 이익을 추구할 수 있는 방법을 제시합니다.



### MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis (https://arxiv.org/abs/2501.06887)
Comments:
          Accepted to 2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)

- **What's New**: 이 연구는 피부암 진단 분야에서 CLIP(Contrastive Language-Image Pretraining) 모델을 활용함으로써 의사들이 AI의 결정 과정을 이해하고 신뢰할 수 있도록 하는 방법을 제안합니다. 특히, MedGrad E-CLIP이라는 새로운 설명 가능성(explainability) 방법을 도입하여, 복잡한 의료 이미지를 위한 가중 엔트로피 메커니즘을 통합합니다. 이 접근 방식은 특정 진단 설명과 연결된 이미지의 중요한 영역을 강조하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 PH² 및 Derm7pt 데이터셋을 사용하여 dermoscopic(피부 경화의) 구조 기준이 포함된 이미지를 연구했습니다. CLIP은 이미지 인코더와 텍스트 인코더로 구성되어 있으며, 이미지를 통해 진단 기준과의 관계를 학습합니다. 샘플 이미지와 설명의 커플을 통해, CLIP의 가중치를 학습하여 새로운 이미지-텍스트 쌍을 분류하는 방식으로 작동합니다.

- **Performance Highlights**: 제안된 메서드는 이미지에서 특징을 추출하고 이를 텍스트 기준과 일치시켜 피부 병변을 분류합니다. 이를 통해 기존의 진단 방법보다 향상된 신뢰성 및 투명성을 제공하며, AI 기반 진단 시스템에서 의사들 사이의 신뢰를 증진시키는 데 기여합니다. 이와 같은 성과는 깊은 신경망(Deep Neural Networks)의 의료 분석 적용에 있어 중요한 진전을 보여줍니다.



### A Foundational Generative Model for Breast Ultrasound Image Analysis (https://arxiv.org/abs/2501.06869)
Comments:
          Peking University; Stanford University; Peking University Cancer Hospital & Institute; Peking Union Medical College Hospital; Cancer Hospital, Chinese Academy of Medical Sciences

- **What's New**: 이번 논문에서는 브레스트 초음파(image analysis) 분석을 위해 특별히 설계된 BUSGen이라는 첫 번째 기초 생성 모델(generative model)을 소개합니다. 기존의 기초 모델들이 여러 임상 작업에 사용되는 것과 달리, BUSGen은 350만 개 이상의 브레스트 초음파 이미지를 기반으로 사전 학습(pretrained)되어 브레스트 구조 및 병리학적 특성에 대한 방대한 지식을 축적했습니다. 이러한 발전은 브레스트 초음파 분석 분야에서의 잠재력을 실현하는 중요한 진전을 나타냅니다.

- **Technical Details**: BUSGen은 few-shot adaptation 방식을 통해 실제적이고 정보가 풍부한 작업 특화 데이터의 저장소를 생성할 수 있습니다. 이를 통해 다양한 하위 작업(downstream tasks)을 위한 모델 개발을 가속화할 수 있습니다. 연구 결과에 따르면, BUSGen은 브레스트 암 검진(screening), 진단(diagnosis), 예후(prognosis) 측면에서 실제 데이터 기반 기초 모델보다 현저히 뛰어난 적응력을 보여주었습니다. 또한, 생성된 데이터의 스케일링 효과(scaling effect)가 수집된 실제 데이터만큼 효과적임을 입증했습니다.

- **Performance Highlights**: 특히, BUSGen은 브레스트 암 조기 진단에서 9명의 인증된 방사선의사(board-certified radiologists)를 초과 성과를 기록하며 평균 민감도(sensitivity)를 16.5% 향상시켰습니다(P-value<0.0001). 또한, 연구를 통해 하위 모델의 일반화 능력(generalization ability)을 개선하며 환자의 프라이버시를 보호하는 Fully de-identified data 공유를 가능하게 했습니다. BUSGen의 온라인 데모는 제공되는 링크를 통해 확인할 수 있습니다.



### Variable Selection Methods for Multivariate, Functional, and Complex Biomedical Data in the AI Ag (https://arxiv.org/abs/2501.06868)
- **What's New**: 이 논문은 고차원 임상 데이터의 분석을 위한 새로운 최적화 기반 변수 선택 방법을 제안합니다. 제안된 방법론은 선형 회귀모델, 분량 회귀모델 및 비모수 가법 모델을 포함하여 다양한 회귀 모델에 적용될 수 있습니다. 또한, 환자의 연속적 모니터링에서 발생하는 기능적 데이터 및 임의의 응답에 적합할 뿐만 아니라, 이러한 고급 기술을 이용한 변동 선택 문제에 대한 실질적인 통찰을 제공합니다.

- **Technical Details**: 이 연구는 프레셰 리지 선택 연산자(FRISO)와 같은 기존 전통적 방법과 비교하여 계산 시간을 크게 단축하는 새로운 서브그래디언트 프로젝션 기법을 활용합니다. 이 방법론은 밀리언 단위의 환자 데이터를 처리할 수 있는 효과성과 확장성을 보여줍니다. 우리의 변수 선택 프레임워크는 다양한 정보 출처를 통합하여 환자의 건강을 포괄적으로 보는 것을 가능하게 하며, 일반화된 가법 모델, 분포 모델 및 기능 간 회귀 모델 등의 복잡한 구조를 지원합니다.

- **Performance Highlights**: 제안된 방법론은 정확성과 속도에서 기존 최신 기술들을 초월하며, 특히 대규모 데이터셋에 대한 적용 가능성을 강조합니다. 확장성 높은 변수 선택 프레임워크는 정밀 의학 및 공중 보건의 발전 가능성을 열어줍니다. 또한, CGM(Continuous Glucose Monitoring) 데이터를 활용한 연구에서 보여준 바와 같은 임상적 활용 사례들도 다루어져, 다양한 통계적 응답 유형에 대해 여러 배 이상의 개선을 달성했습니다.



### COMPASS: A Compiler Framework for Resource-Constrained Crossbar-Array Based In-Memory Deep Learning Accelerators (https://arxiv.org/abs/2501.06780)
Comments:
          Accepted IEEE DATE 2025

- **What's New**: 최근 메모리 내 가속기(accelerator)의 수요가 증가하고 있으며, 이 논문에서는 COMPASS라는 새로운 컴파일러 프레임워크를 소개합니다. 이 프레임워크는 대규모 심층 신경망 모델을 지원할 수 있도록 설계되었으며, 메모리 한계를 극복하기 위해 외부 메모리와의 연결을 필요로 합니다. 특히, COMPASS는 레이어를 최적의 파티셔닝으로 나누어 각 파티션이 온칩에서 가속화될 수 있도록 합니다.

- **Technical Details**: COMPASS는 자원 제약이 있는 크로스바 기반의 메모리 내 딥 뉴럴 네트워크 가속기를 위한 컴파일러 프레임워크입니다. 이 프레임워크는 데이터 의존성, 코어 활용도(core utilization), 쓰기 명령 수를 고려하여 레이어를 나누고, 지연(latency) 및 메모리 접근 방식을 최소화하여 에너지 효율성을 개선합니다. COMPASS 알고리즘은 유전 알고리즘(genetic algorithm, GA)을 사용하여 최적의 파티셔닝을 찾습니다.

- **Performance Highlights**: 시뮬레이션 결과, COMPASS는 메모리 사용량을 최소화하면서도 더 많은 네트워크를 수용할 수 있음을 보여줍니다. 이는 기본 파티셔닝 방법에 비해 처리량(thoroughput)을 1.78배 증가시키고, 에너지-지연 제품(EDP)을 1.28배 절약하는 성과를 달성했습니다. 이러한 성과는 제한된 자원에서 대규모 DNN을 효과적으로 실행할 수 있는 가능성을 제시합니다.



### Hierarchy-Boosted Funnel Learning for Identifying Semiconductors with Ultralow Lattice Thermal Conductivity (https://arxiv.org/abs/2501.06775)
Comments:
          13 pages, 6 figures

- **What's New**: 본 논문에서는 계층 강화된 깔때기 학습(HiBoFL, hierarchy-boosted funnel learning) 프레임워크를 제안하며, 이를 통해 초저 격자 열 전도도(κL, lattice thermal conductivity)를 가진 반도체 소재를 효과적으로 식별하는 방법을 소개합니다. 수백 개의 재료로만 훈련해도 우수한 결과를 얻을 수 있는 이 접근법은 기존의 대규모 계산을 피할 수 있도록 합니다. 이를 통해 열전기 응용에 적합한 초저 κL 후보 리스트를 제공하고, 구조적 비선형성에 영향을 미치는 새로운 변수를 발견했습니다.

- **Technical Details**: 기계 학습(ML)은 재료의 특성을 예측하는 데 있어 강력한 도구로 떠오르며, 밀도 함수 이론(DFT) 기반의 고속 계산을 활용한 다양한 데이터베이스의 발전으로 더 빠르고 효율적인 물질 탐색이 가능해졌습니다. 지도학습과 비지도학습이 결합되어, 복잡한 특성을 가진 재료에 대한 데이터 라벨링 비용을 절감하고 예측 효율성을 높일 수 있는 방법을 모색합니다. 이 연구는 특히 열전기(Electric effect)의 전환 효율과 밀접하게 연관된 초저 κL의 예측과 설명을 다루고 있습니다.

- **Performance Highlights**: 제안된 HiBoFL 프레임워크는 수천 개의 재료 pool에서 몇 백 개의 목표 재료를 이용하여 비지도 학습을 통해 효과적인 지도 예측을 가능하게 했습니다. 이를 바탕으로 저렴한 비용으로 라벨링을 하고도 복잡한 재료의 열 전도도 특성을 효율적으로 예측할 수 있음을 입증했습니다. 연구 결과로 도출된 새로운 후보 물질들은 열전기 분야의 산업 발전에 크게 기여할 것으로 예상됩니다.



### Improving the adaptive and continuous learning capabilities of artificial neural networks: Lessons from multi-neuromodulatory dynamics (https://arxiv.org/abs/2501.06762)
- **What's New**: 이 논문은 생물학적 학습 시스템의 핵심 요소인 neuromodulation이 인공 신경망(ANN)에서 연속 학습 시 발생하는 문제들을 해결하고 성능을 향상시킬 수 있는 방법을 탐구합니다. 특히, 생물체가 다양한 환경에 적응하며 지식을 습득하고 유지하는 과정을 모방하여 ANN의 지속적인 학습을 지원하고자 합니다.

- **Technical Details**: Neuromodulation은 도파민(dopamine), 아세틸콜린(acetylcholine), 세로토닌(serotonin), 노르아드레날린(noradrenaline)과 같은 신경전달물질에 의해 조절됩니다. 뇌에서의 neuromodulatory process는 지역적인 시냅스 가소성(local synaptic plasticity)부터 전역적인 네트워크 적응성(global network-wide adaptability)까지 다양한 메커니즘을 통해 발생하며, 이는 환경 변화에 대한 동적인 반응을 Facilitates합니다.

- **Performance Highlights**: 우리는 neuro-modulation에 기반한 메커니즘이 ANN의 Go/No-Go(task) 성능을 향상시킬 수 있는 사례 연구를 제시합니다. 다양한 공간적 및 시간적 스케일에서의 neuromodulators의 영향을 통합함으로써, 생물학적 학습과 인공 시스템 간의 격차를 해소하고 보다 유연하고, 강인하며 적응성이 뛰어난 ANN을 위한 길을 열고자 합니다.



### Better Prompt Compression Without Multi-Layer Perceptrons (https://arxiv.org/abs/2501.06730)
Comments:
          7 pages, 0 figures

- **What's New**: 이번 논문에서는 Attention-Only Compressor (AOC)를 도입하여, 프롬프트를 더 효율적으로 압축하는 방법을 제안합니다. 이 방법은 기존의 인퍼런스 언어 모델 아키텍처의 다층 퍼셉트론(MLP) 계층을 제거하면서 학습 가능한 압축 인코더를 생성하여, 약 67% 적은 파라미터로도 유사한 성능을 유지할 수 있음을 보여줍니다. 특히 AOC는 최대 480배의 압축비에서도 기본적인 압축 인코더보다 더 나은 성능을 나타냅니다.

- **Technical Details**: AOC는 입력으로 주어진 프롬프트 토큰을 학습된 메모리 토큰과 결합하여 잠재 표현을 생성하는 구조를 가지고 있습니다. 이 모델은 기존 인퍼런스 프롬프트의 정보를 유지하면서 계산 리소스를 크게 절약합니다. 논문에서는 Llama 3.2 1B Instruct를 사전 학습된 모델로 사용하고, AdamW 옵티마이저를 통한 훈련 과정도 자세히 설명하고 있습니다.

- **Performance Highlights**: AOC는 기존의 압축 방법들과 비교했을 때 압축 성능이 우수할 뿐만 아니라, 인퍼런스 시간 동안 계산 필요량이 적은 장점을 가지고 있습니다. 실험 결과, AOC는 거의 세 배 더 많은 파라미터를 가진 기본 프롬프트 압축 인코더와 비슷한 성능을 발휘하며, 더욱 효율적인 압축기의 개발에 대한 가능성을 보여줍니다. 이러한 결과는 미래의 연구에서 압축 아키텍처의 다양성을 탐구하는 데 중요한 기초 자료가 될 것입니다.



### Sequential Portfolio Selection under Latent Side Information-Dependence Structure: Optimality and Universal Learning Algorithms (https://arxiv.org/abs/2501.06701)
Comments:
          34 pages, working paper, first draft (errors may exist)

- **What's New**: 이 논문은 자산 가격과 관측되지 않은 측면 정보 간의 잠재적 의존 구조가 있는 시장에서 최적의 노 쇼트(no-short) 순차 포트폴리오 전략을 구성하는 투자 문제를 다룹니다. 특히, 동적 전략이 고정 전략보다 무한히 높은 성장률을 나타내지 않는다는 점을 조명합니다. 이러한 발견은 정보 이론 및 금융 분야에서의 전통적인 관점을 재검토하게 만듭니다.

- **Technical Details**: 이 연구에서는 고차원 데이터에 대한 계산 복잡성과 함께 관측되지 않은 측면 정보의 불완전성을 해결하는 일반 모델을 제시합니다. 로그-최적(log-optimal) 전략을 도입하여 최적 벤치마크로 설정하며, 이 전략이 측면 정보 없이도 존재할 수 있는 랜덤 최적 상수 전략을 수립한다는 결과를 보여줍니다. 이 분석은 또한 포트폴리오 구성에 대한 두 가지 학습 알고리즘 접근 방식을 제시합니다.

- **Performance Highlights**: 결과적으로, 최신 알고리즘이 측면 정보를 제거하더라도 최적 동적 전략과 유사한 비대칭적 성장률을 보장할 수 있음을 입증합니다. 이 연구는 인과적 정보의 제한에도 불구하고, 무작위 최적 상수 전략이 로그-최적 전략의 성장률에 근접할 수 있는 가능성을 열게 됩니다.



### Average Reward Reinforcement Learning for Wireless Radio Resource Managemen (https://arxiv.org/abs/2501.06700)
Comments:
          Accepted by Asilomar 2024

- **What's New**: 이 논문에서는 무선 통신에서 방사 자원 관리(RRM)에 강화 학습(RL)을 적용할 때 간과되기 쉬운 요소인 할인된 보상 RL 구성과 무할인 목표 간의 불일치 문제를 다룹니다. 저자들은 이러한 불일치를 체계적으로 조사한 최초의 연구로, RRM 문제를 평균 보상 RL 프레임워크로 전환하여 해결할 수 있는 방안을 모색합니다. 새롭게 제안된 Average Reward Off policy Soft Actor Critic (ARO SAC) 방법은 평균 보상으로 접근하여 효과성을 높이고, 기존의 할인된 보상 RL 방식에 비해 성능을 15% 향상시키는 결과를 보여줍니다.

- **Technical Details**: 이 연구는 RAN 네트워크 슬라이싱 문제를 평균 보상 RL 문제로 다시 포맷하여 기존 RL 접근 방식의 디자인 목표와의 불일치를 강조합니다. 저자들은 평균 보상 RL의 도전 과제를 다루기 위한 알고리즘에서의 효과적인 자원 할당 및 RL 업데이트 방식을 탐구합니다. ARO-DDPG의 오프 정책 Rl 알고리즘 전략을 기반으로 SAC를 평균 보상 버전인 ARO-SAC으로 확장하여 TD 오류 및 벨만 방정식의 수정을 통해 평균 보상 목표를 설정합니다.

- **Performance Highlights**: 실험 결과에 따르면, ARO-SAC는 적절히 선택된 하이퍼파라미터를 통해 기존 SAC보다 약 15% 더 우수한 성능을 보여줍니다. 또한 평균 보상 비율 및 환경 기간의 학습 속도가 ARO-SAC의 성능에 미치는 영향을 분석합니다. 이러한 성과는 무선 네트워크 최적화에 있어 평균 보상 RL의 가능성을 더욱 부각시킵니다.



### Understanding and Mitigating Membership Inference Risks of Neural Ordinary Differential Equations (https://arxiv.org/abs/2501.06686)
- **What's New**: 본 연구는 Neural Ordinary Differential Equations (NODEs)의 membership inference 위험을 처음으로 조사한 것으로, 이 모델은 데이터 분포를 모델링하는 fundamentally different approach를 사용합니다. NODEs는 전통적인 feedforward neural networks인 ResNets에 비해 최대 2배 낮은 membership risks을 보이는 것을 확인했습니다. 또한, 위험을 줄이는 요인들을 분석하였습니다.

- **Technical Details**: NODEs는 학습 가능한 differential equations를 통해 데이터의 기저 분포를 모델링하며, 이 접근 방식은 시간 연속성과 다이나믹 시스템을 이해하는 데 강력한 도구입니다. 본 연구는 NODEs 뿐만 아니라 Neural Stochastic Differential Equations (NSDEs)의 stochastic variations을 통해 membership inference 위험을 줄일 수 있는 방법을 제안합니다. NSDEs는 differential privacy (DP) 원리를 적용하여 학습 중에 개인정보 유출을 최소화합니다.

- **Performance Highlights**: 실험 결과 NSDEs는 membership inference 공격에 대해 기존 NODEs보다 1.8배에서 10배 더 효과적으로 위험을 줄임을 보여주었습니다. 또한, NSDEs는 전통적인 DP-SGD로 훈련된 private 모델들과 비슷한 수준의 유용성을 유지하면서 개인정보 보호-유용성 간의 trade-off를 개선할 수 있는 가능성을 제시합니다. 마지막으로, 기존의 사전 훈련된 모델에 NSDEs를 통합함으로써 privacy risk를 10배까지 감소시킬 수 있음을 실증하였습니다.



### Application of Vision-Language Model to Pedestrians Behavior and Scene Understanding in Autonomous Driving (https://arxiv.org/abs/2501.06680)
- **What's New**: 최근 자율주행(Autonomous Driving) 기술이 크게 발전하여 3D 탐지, 분류, 로컬라이제이션 결과를 보여주고 있습니다. 그러나 보행자 행동의 의미적 이해 및 보행자와의 인터랙션 처리와 같은 많은 도전 과제가 여전히 남아 있습니다. 이 연구에서는 대규모 언어 모델(LLM) 및 비전-언어 모델(VLM)의 지식을 소형 비전 네트워크로 효과적으로 증류하는 방법을 분석하여, 복잡한 장면의 의미적 표현으로 의사결정 및 제어에 활용할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 비전-언어 기초 모델은 시각 데이터와 텍스트 데이터를 통합하여 다중 모달 AI의 최전선에 있는 기술입니다. 이 논문에서는 GPT4-V와 같은 사전 훈련된 비전-언어 모델로부터 일반 지식을 전이하는 지식 증류(Knowledge Distillation)를 수행하였으며, 이를 통해 보행자의 행동과 의미적 속성을 이해하는 효율적인 비전 모델을 개발하였습니다. 이 과정에서 다중 라벨 분류 문제로 설정하고, GPT의 주석을 통해 생성된 의미적 텍스트 라벨로 비전 네트워크를 감독하여 정확한 예측을 목표로 하였습니다.

- **Performance Highlights**: 우리는 이 연구를 통해 열거된 과제들에 대해 의미적으로 풍부한 보행자 속성과 분류 체계를 통해 개선된 성과를 달성하였습니다. 전통적인 분류법보다 더 많은 보행자 속성과 의미적 카테고리를 수집하여, 보다 정교한 자율주행 시스템을 위한 기반을 마련하였습니다. 이를 통해 보행자의 행동을 보다 정확하게 예측하고, 안전하고 신뢰할 수 있는 자율주행 내비게이션을 위한 기반을 강화하였습니다.



### SafeSplit: A Novel Defense Against Client-Side Backdoor Attacks in Split Learning (https://arxiv.org/abs/2501.06650)
Comments:
          To appear at NDSS 2025; 18 pages, 6 Tables, and 11 figures

- **What's New**: 이 논문에서는 Split Learning(SL)에서 클라이언트 측의 백도어 공격에 대한 첫 번째 방어 시스템인 SafeSplit을 제안합니다. SafeSplit은 원형 역분석(circular backward analysis)을 활용하여 서버가 악의적인 클라이언트의 행동을 감지하고 필터링할 수 있게 합니다. 이 시스템은 트레이닝이 완료된 후 모델을 검사하여 안전한 체크포인트로 되돌리는 방식으로 작동합니다.

- **Technical Details**: SafeSplit의 두 가지 주요 분석 방식은 정적(static) 및 동적(dynamic) 분석입니다. 정적 분석은 서버의 레이어 파라미터의 변화 빈도를 측정하며, 동적 분석은 회전 거리 미터(circular distance metric)를 통해 파라미터의 방향 변화를 평가합니다. 이 두 가지 접근 방식은 백도어 공격에 의해 악의적으로 변화한 모델을 효과적으로 감지하는 데 도움을 줍니다.

- **Performance Highlights**: SafeSplit은 다양한 데이터 분포, 클라이언트 수 및 공격 시나리오를 포함한 종합적인 평가를 통해 높은 효율성을 입증하였습니다. 이 시스템은 모델 유용성을 보존하면서 백도어 공격을 효과적으로 완화하는 데 기여합니다. 또한, 이 모델은 GDPR 및 HIPAA와 같은 데이터 프라이버시 표준을 준수하는 리소스 제약 장치에서도 대규모 모델을 안전하게 교육할 수 있도록 합니다.



### A Tight VC-Dimension Analysis of Clustering Coresets with Applications (https://arxiv.org/abs/2501.06588)
- **What's New**: 이 논문에서는 k-클러스터링 문제를 위한 coreset을 다루고 있으며, 이는 거리의 거듭제곱을 최소화하는 점들을 중심에 할당하는 것을 목표로 합니다. 특히, k-median 목적 함수에 대한 새로운 VC 차원 기반의 분석을 제시하여 더욱 개선된 coreset 경계치를 제공합니다. 이는 planar graphs의 최단 경로 메트릭 및 Frechet 메트릭에 대한 클러스터링에 대한 기존의 경계치를 조정한 결과를 담고 있습니다.

- **Technical Details**: Coresets는 주로 학습 이론과 빅데이터 알고리즘에서 사용되며, 특정 점 집합 P와 쿼리 집합 Q, 그리고 관련 손실 함수 f가 주어졌을 때, P의 c의 코어셋은 모든 쿼리에 대해 손실을 근사하는 작은 가중치 집합입니다. 연구에서는 k-median 문제를 해결하기 위한 coreset의 크기를 최소화하는데 중점을 두며, 다양한 거리 메트릭에 따라 적절한 경계식을 제시합니다. 특히, VC 차원을 활용하여 효율적인 coreset 구성을 보여주고, 비드론 확률적 방법의 한계를 극복하는 방향으로 나아갑니다.

- **Performance Highlights**: 논문에서 제안된 방법론에 따라, planar graphs의 최단 경로 메트릭에서 기존의 coreset 크기 $	ilde{O}(k	ext{ε}^{-6})$를 $	ilde{O}(k	ext{ε}^{-2})$로 개선할 수 있었습니다. 또한, d차원 다각형 곡선 클러스터링에 대해서도 더 나은 경계치를 제시하여, $	ilde{O}(kd	ext{ε}^{-2}	ext{log} m)$을 달성했습니다. 이러한 결과는 조화적 경계 및 기존 연구 결과들과 비교해 볼 때 상당한 개선을 나타냅니다.



### Physics-Informed Neuro-Evolution (PINE): A Survey and Prospects (https://arxiv.org/abs/2501.06572)
Comments:
          20 pages, 8 figures, 1 table

- **What's New**: 이 논문은 물리 법칙을 통합하여 훈련된 Physics-Informed Neural Networks (PINNs)에 대한 새로운 시각을 제시합니다. PINNs는 한정된 데이터 제약에서 제공하는 뛰어난 성능으로 과학 머신 러닝의 최전선에 있으며, 과거의 데이터 기반 모델들보다 유리하다는 점이 강조됩니다. 그러나 PINN의 최적화 및 일반화와 관련된 문제들을 해결하기 위한 새로운 알고리즘적 진보가 필요하다고 주장합니다.

- **Technical Details**: PINNs는 수학적으로 정의된 학습 편향을 신경망의 손실 함수에 통합하여 데이터 라벨이 부족한 경우에도 물리적 법칙을 기반으로 학습할 수 있도록 설계되었습니다. 이러한 방식은 PDE(Partial Differential Equations)와 같은 다양한 과학적 지식을 모델에 포함시킬 수 있으며, 비선형 및 선형 조합을 잘 다룰 수 있는 유연성을 제공합니다. 또한 이 논문에서는 neuroevolution 기법을 통해 PINN 훈련 과정에서 발생하는 복잡한 손실 공간을 최적화하는 방법도 탐구합니다.

- **Performance Highlights**: PINNs는 데이터가 부족한 과학 분야에서 유망한 잠재력을 보여주며, 그래디언트 기반 최적화 할당과 비교하여 지역 최적해에 갇힐 위험성이 낮은 gradient-free 알고리즘이 보다 효과적일 것으로 기대됩니다. 본 논문은 PINN의 효율적인 학습을 위한 새로운 최적화 기법을 제안하고, 다양한 물리적 문제에 대한 해결책에 적합한 맞춤형 아키텍처를 탐색할 수 있는 가능성을 강조합니다. 향후 연구 방향으로는 PINN 모델의 일반화 가능성을 높이기 위한 다양한 메타 학습 접근 방식을 제시합니다.



### Natural Language Processing and Deep Learning Models to Classify Phase of Flight in Aviation Safety Occurrences (https://arxiv.org/abs/2501.06564)
Comments:
          NLP, Aviation reports, Text analysis, Deep learning algorithms, Flight phase classification

- **What's New**: 이 연구에서는 항공사고의 전조 사건들을 설명하는 비구조화된 텍스트를 분석하기 위해, 자연어 처리(NLP)와 인공지능(AI) 모델을 적용했습니다. 이는 항공 산업 이해관계자들이 안전 사건을 분류하고 결정할 수 있는 기반을 제공하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구진은 NTSB(국가교통안전위원회)의 27,000개의 안전 사건 보고서를 활용하여 두 개의 딥러닝 모델, 즉 ResNet과 sRNN의 분류 성능을 평가했습니다. 이들 모델은 비행 안전 사건의 단계(classification of flight phases)를 정확하게 분류하는 데 사용되었습니다.

- **Performance Highlights**: 평가 결과, 두 모델 모두 68% 이상의 정확도를 기록하며, 7개 클래스 분류 문제에서 무작위 추측률인 14%를 크게 초과하는 성과를 보였습니다. 특히, sRNN 모델이 이 연구에 사용된 단순화된 ResNet 모델 아키텍처보다 훨씬 우수한 성능을 나타냈습니다.



### Discrete Speech Unit Extraction via Independent Component Analysis (https://arxiv.org/abs/2501.06562)
Comments:
          Accepted to ICASSP 2025 SALMA Workshop. Code available at this https URL

- **What's New**: 이 연구에서는 Self-supervised speech models (S3Ms)을 활용하여 speech processing 분야에서 분리된 음성 단위인 discrete speech units (DSUs)를 추출하기 위한 선형 전처리 방법(linear preprocessing methods)의 가능성을 조사합니다. 현재까지 S3M의 표현(representation)을 더욱 효과적으로 클러스터링하기 위한 전처리 연구는 부족했던 점을 강조합니다.

- **Technical Details**: 본 논문에서는 표준화(standardization), 주성분 분석(principal component analysis), 화이트닝(whitening), 독립 성분 분석(independent component analysis, ICA) 방법을 DSU 기반 자동 음성 인식(ASR) 벤치마크에서 평가합니다. 이러한 전처리 방법들이 k-means 클러스터링을 위한 전처리로서 효과적임을 입증하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: DSUs를 사용하는 경우 자동 음성 인식에서 강력한 성능 향상을 보여주며, ICA의 개별 성분의 직교성(orthogonality) 및 해석 가능성(interpretability)에 대한 방대한 분석을 수행합니다. 이러한 발견은 DSUs의 품질을 향상시키는 새로운 접근 방식을 제시합니다.



### Dynamic Causal Structure Discovery and Causal Effect Estimation (https://arxiv.org/abs/2501.06534)
- **What's New**: 이 논문에서는 기존의 고정된 인과관계를 다루는 방법에서 벗어나, 시간에 따라 변화할 수 있는 인과관계를 모델링하기 위한 새로운 프레임워크를 개발하였습니다. 이를 통해 인과 그래프의 동적 패턴을 포착하고, 기존의 제한적인 가정 없이 다양한 인과 관계를 도출할 수 있습니다. 특히, covid 데이터 분석에 이 방법을 적용하여 정책 제한 효과의 변화를 평가한 사례를 제공합니다.

- **Technical Details**: 이 연구에서는 시간에 따라 변할 수 있는 인과 그래프를 모델링하기 위해, 기본 근사 방법(basis approximation method)과 점수 기반 방법(score-based approach)을 결합하였습니다. 저자는 동적 선형 구조 방정식 모델(dynamic linear structural equation model)을 활용하여 자가회귀 모델 구조(autoregressive model structure)에서 동시적(causation) 및 시간 지연 인과관계를 파악할 수 있도록 하였습니다. 또한, 이 알고리즘은 과거 추정 및 미래 예측을 동시에 제공할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 알고리즘은 인과 그래프의 시간 가변적 동적 인과 효과를 제시합니다. 다양한 시나리오에서 적용 가능하며, 특히 covid 데이터 분석을 통해 정책 결정이 어떻게 인과관계에 영향을 미치는지를 정량적으로 나타냅니다. 이 방법은 다양한 시간 간격을 고려할 수 있어 실질적인 예측과 분석에 강력한 도구가 될 것입니다.



### Sequential Classification of Aviation Safety Occurrences with Natural Language Processing (https://arxiv.org/abs/2501.06490)
- **What's New**: 이 연구에서는 항공 안전 시스템에서의 안전 발생 사건을 분류하고 분류하기 위해 자연어 처리(NLP)와 인공지능(AI) 모델을 활용했습니다. 기존의 안전 관련 사건 보고서는 비구조적이고 알아보기 힘든 텍스트로 작성되어 있었는데, 이를 컴퓨터 시스템이 이해할 수 있도록 처리함으로써 안전 또한 개선될 수 있게 됩니다.

- **Technical Details**: 연구에서는 LSTM, BLSTM, GRU, sRNN과 이들 모델의 조합(예: LSTM+GRU, BLSTM+GRU 등)을 포함한 여러 딥러닝(deep learning) 모델의 분류 성능을 평가했습니다. 이 모델들은 27,000개의 NTSB(National Transportation Safety Board) 안전 사건 보고서를 기반으로 하여 그 성능이 87.9% 이상의 정확도를 기록했습니다. 이는 네 가지 클래스 분류 문제에서 우연히 맞힐 확률인 25%를 상회하는 수치입니다.

- **Performance Highlights**: 모델들은 80%, 88%, 85% 이상의 높은 정밀도(precision), 재현율(recall), F1 점수를 기록하며 경쟁력 있는 퍼포먼스를 보였습니다. 특히 sRNN 모델이 재현율(90%)과 정확도(90%)에서 가장 뛰어난 성능을 보였고, LSTM은 정밀도(87%)에서 약간 더 나쁜 성과를 기록했습니다.



### Enhancing Multi-Modal Video Sentiment Classification Through Semi-Supervised Clustering (https://arxiv.org/abs/2501.06475)
- **What's New**: 본 연구는 비디오 감정 분류 성능을 향상시키기 위해 비디오, 텍스트 및 음향 특징을 이용한 다중 모달 접근 방식을 제안합니다. 기존의 감정 분석 모델들이 대량의 라벨링된 데이터에 의존하는 단점을 극복하기 위해, 클러스터링 기반의 반지도 학습 방법을 개발하여 의미 있는 데이터 표현을 추출합니다. 이는 감정 인식을 지원하는 모델의 기반 구조를 학습하는 데 필요한 라벨된 데이터의 양을 줄여줍니다.

- **Technical Details**: 연구 방법론은 두 가지 단계로 나뉘며, 첫 번째 단계에서는 기존의 라벨링된 데이터셋의 작은 부분집합을 사용하여 초기 감정 분류를 수행하고, 여러 기존 모델과 비교합니다. 두 번째 단계에서는 Deep Embedded Clustering (DEC) 기반의 반지도 클러스터링 네트워크를 구현하여 특징 표현을 개선합니다. DEC 네트워크는 전체 데이터셋을 미리 학습하여 복잡한 다중 모달 관계를 캡처한 후, 이를 통해 초기화된 가중치를 사용하여 분류 성능을 최적화합니다.

- **Performance Highlights**: CMU-MOSI 데이터셋을 통한 평가 결과, 제안된 방법이 기존의 지도 학습 기준선과 비교하여 효과적인 성능 향상을 보여주었습니다. 특히, 라벨된 데이터가 제한적인 상황에서도 모델이 높은 정확도와 F1 점수를 기록하였으며, 이러한 접근 방식이 다중 모달 감정 분석의 최신 한계를 넘는 데 기여할 가능성을 보여주었습니다.



### CNN-powered micro- to macro-scale flow modeling in deformable porous media (https://arxiv.org/abs/2501.06466)
Comments:
          21 pages, 12 figures, research paper

- **What's New**: 이 논문은 변형 가능한 다공성 매체에서 미세 CT 이미지 세트를 활용하여 거시적 고유 투과율 텐서를 예측하는 새로운 기법을 제안합니다. 기존의 투과율 추정 기법들의 한계를 극복하기 위해, 딥러닝 기술인 Convolutional Neural Networks (CNN)을 사용하여 변형 및 비등방 유동 조건에서의 유체 흐름 특성을 예측하는 방법론을 소개하고 있습니다. 이 approaches은 딥러닝을 활용하여 다공성 미세구조의 CT 이미지를 입력으로 받고, 대칭 2차 고유 투과율 텐서를 예측하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 방법론은 4단계로 구성됩니다: 1) 다양한 체적 변형 수준에서 Bentheim sandstone의 CT 이미지 데이터세트를 구축; 2) 단상 유동의 pore-scale 시뮬레이션 수행; 3) 처리된 CT 이미지를 입력으로 CNN 모델 학습; 4) 데이터 증강 및 대체 CNN 아키텍처를 활용하여 모델 일반화 개선. 이 과정에서 lattice Boltzmann method (LBM)와 같은 고급 시뮬레이션 기법을 활용하여 투과율 데이터를 생성합니다.

- **Performance Highlights**: CNN 기반 접근법은 시뮬레이션 요약 모델들이 좋은 성능을 발휘하는 데 필요한 데이터 양이 충분하다는 데 기반을 두고 있으나, 데이터가 부족할 경우에는 데이터 증강 기법이 도움이 될 수 있습니다. 예를 들어, GAN을 활용하여 합성 데이터의 생성을 통해 일반 패턴을 포착하고, 이후 Transfer Learning 기법을 통해 기존 데이터셋에 대한 모델 성능을 향상시키는 방법도 논의됩니다. 이러한 접근법들은 지반공학, 수문학, 재료 과학 등 다양한 분야에서 중요한 고유 투과율 텐서를 예측하는 데 기여할 수 있습니다.



### Reinforcement Learning for Enhancing Sensing Estimation in Bistatic ISAC Systems with UAV Swarms (https://arxiv.org/abs/2501.06454)
- **What's New**: 이 논문은 무인 항공기(UAV) 군집을 활용하여 통합 감지 및 통신(ISAC) 네트워크를 향상시키기 위한 새로운 다중 에이전트 강화 학습(MARL) 프레임워크를 제안합니다. UAV의 위치와 궤적 최적화를 부분 관찰 마르코프 결정 과정(PO-MDP)으로 설정하여, 중앙 집중식 훈련과 분산 실행을 활용하는 방법론을 개발하였습니다. 특히, UAV간의 효과적인 통신 프로토콜을 구축하는 분산 협력 MARL 전략을 구현하여 감지 성능을 극대화합니다.

- **Technical Details**: 제안된 시스템 모델은 K개의 기지국(BS)이 사전에 정의된 지역 내에서 사용자들을 지원하고, q개의 대상을 모니터링하는 이동 네트워크입니다. M개의 UAV들이 분산된 방식으로 감지 레이더로 배치되어 OFDM(Orthogonal Frequency Division Multiplexing) 신호를 활용하여 환경의 감지 매개변수를 추정합니다. 각 UAV의 적절한 궤적 설계가 감지 성능을 향상시키는 주요 요소로 작용하며, 전체 감지 성능은 총 감지 신호 대 잡음 비(SNR)로 측정됩니다.

- **Performance Highlights**: 이 논문에서 제안된 MARL 기반 프레임워크는 UAV의 배치 최적화를 통해 레이더 감지 메트릭을 극대화하며, 실질적인 무선 환경을 고려하여 효율적이고 견고한 통신 프로토콜을 학습할 수 있게 합니다. 또한, 통신 간섭을 줄이기 위한 전송 전력 적응 기법을 통해 UAV 간 신호 간섭 대 잡음 비(SINR)를 극대화합니다. 복잡성이 증가함에도 불구하고, 다양한 시나리오에서의 강력한 성능과 적응성이 입증되어 향후 ISAC 네트워크의 확장 가능하고 비용 효율적인 개선을 제공합니다.



### Cross-Technology Interference: Detection, Avoidance, and Coexistence Mechanisms in the ISM Bands (https://arxiv.org/abs/2501.06446)
- **What's New**: 본 논문은 다양한 무선 네트워크가 공유하는 ISM(Industry, Scientific, and Medicine) 주파수 대역 내에서 발생하는 크로스-테크놀로지 인터페어런스(Cross-Technology Interference, CTI)의 영향을 분석합니다. 공통의 매체 접근 규칙 없이 다르게 설계된 네트워크들 간의 간섭은 저전력 네트워크와 같은 자원이 제한된 기술에 특히 심각한 영향이 있습니다. 저자들은 CTI 감지, 공존 및 회피 메커니즘에 대한 최신 발전을 논의하고 이질 네트워크 간의 직접적인 통신을 가능하게 하는 메세징 방식도 다룹니다.

- **Technical Details**: 논문에서는 IEEE 802.15.4 표준을 기반으로 한 저전력 네트워크에서의 CTI의 영향을 조명합니다. CTI는 고전력 및 고성능 전송 방식을 사용하는 기술이 자원 제약이 있는 기술에 비해 더 큰 간섭을 초래함으로써 발생합니다. 저자들은 이 문제를 해결하기 위한 최신 CTI 감지 전략과 회피 및 공존 전략을 구체적으로 소개하고, 특히 실제 구현 및 배치의 관점에서 프로토콜과 알고리즘을 강조합니다.

- **Performance Highlights**: 실험 결과, 다수의 기술이 같은 주파수 대역에서 작동할 때 CTI가 발생하여 네트워크의 패킷 손실, 재전송 비용, 지연 및 예측 불가능성을 유발하게 됨을 보여줍니다. 저자들은 또한 CTI가 저전력 네트워크의 성능 및 에너지 소비에 미치는 영향을 상세히 논의하며, 최근의 연구 결과와 비교하여 CTI로 인한 성능 저하를 구체적으로 언급합니다.



### Tensor Product Attention Is All You Need (https://arxiv.org/abs/2501.06425)
Comments:
          23 pages, 5 figures

- **What's New**: 이번 논문에서는 Tensor Product Attention (TPA)이라는 새로운 어텐션 메커니즘을 제안하여, 메모리 효율성과 성능을 모두 향상시킵니다. TPA는 텐서 분해(tensor decompositions)를 활용하여 질의(query), 키(key), 값(value)의 표현을 압축이 가능하게 하고, 이는 추론 과정에서 KV 캐시의 크기를 대폭 줄이는데 기여합니다. 또한, TPA는 Rotary Position Embedding (RoPE)과의 원활한 통합을 통해 기존 LLM 아키텍처에서 쉽게 적용될 수 있습니다.

- **Technical Details**: TPA는 계층적 텐서를 활용해 질의(Q), 키(K), 값(V)을 동적으로 분해하는 방식을 채택합니다. 이를 통해, TPA는 저성능의 KV 캐시 메모리 사용을 실질적으로 운용할 수 있으며, 더욱 강력한 표현 능력을 제공합니다. 이 논문에서는 TPA를 기반으로 한 새로운 모델 아키텍처인 Tensor ProducT ATTenTion Transformer (T6)도 소개하며, 다양한 언어 모델링 작업에서의 성능 향상을 보여줍니다.

- **Performance Highlights**: T6 모델은 기존의 표준 Transformer 모델, 즉 MHA, MQA, GQA, MLA에 비해 모든 평가 메트릭에서 더 우수한 성능을 달성했습니다. 특히, T6는 KV 캐시 크기를 줄이면서도 검증 perplexity와 downstream 성능에서도 일관된 향상을 보였습니다. 이러한 성능은 TPA의 메모리 효율성 덕분에 가능해졌으며, 이는 더 긴 입력 시퀀스를 처리할 수 있는 가능성을 제시합니다.



### IPP-Net: A Generalizable Deep Neural Network Model for Indoor Pathloss Radio Map Prediction (https://arxiv.org/abs/2501.06414)
Comments:
          2 pages, 1 figure, Accepted to ICASSP 2025

- **What's New**: 이 논문에서는 실내에서의 데이터 전송 손실(pathloss) 예측을 위한 일반화된 심층 신경망 모델(IPP-Net)을 제안합니다. IPP-Net은 UNet 아키텍처에 기반을 두고 있으며, 대규모 레이 추적(ray tracing) 시뮬레이션 데이터와 수정된 3GPP 실내 핫스팟 모델로부터 학습합니다. ICASSP 2025의 첫 번째 실내 데이터 전송 손실 예측 챌린지에서 평가 결과, IPP-Net은 세 가지 경쟁 작업에서 가중된 평균 제곱근 오차(RMSE) 9.501 dB를 달성하며, 전체 2위로 평가되었습니다.

- **Technical Details**: IPP-Net은 입력으로 5개의 채널을 사용하는 UNet 아키텍처로 설계되었습니다. 초기 레이어는 다섯 개의 채널 입력을 준비하고, 인코딩/디코딩 레이어와 병목(bottleneck) 레이어는 연속된 팽창(convolution) 레이어와 표준 convolution 레이어를 공유합니다. 또한, Non-Line-Of-Sight(NLOS) 레벨 행렬과 가중치 행렬을 도입하여 모델 정확성을 극대화하며, 단계적(curriculum) 학습 전략을 통해 지속적으로 학습 작업의 복잡성을 증가시킵니다.

- **Performance Highlights**: IPP-Net은 세 가지 작업에 대해 엄격한 실험을 거쳐 평균적으로 10.51 ms의 실행 시간으로 하나의 라디오 맵을 예측할 수 있습니다. 훈련 시간은 각각 약 20분, 4시간, 8시간이 소요되었으며, 최종 평가를 위해 validation 세트에서 가장 낮은 RMSE를 실현하는 네트워크 가중치를 저장했습니다. 이러한 결과들은 IPP-Net이 실제 실내 환경에서 라디오 맵 예측의 성능을 크게 향상시킬 수 있음을 보여줍니다.



### Computational and Statistical Asymptotic Analysis of the JKO Scheme for Iterative Algorithms to update distributions (https://arxiv.org/abs/2501.06408)
- **What's New**: 이번 논문에서는 Jordan, Kinderlehrer 및 Otto의 JKO 스킴(JKO scheme)을 확장하여 미지의 매개변수를 갖는 모델에 적용하는 방법을 제안합니다. 전통적으로 JKO 스킴은 모든 모델이 완전히 알려져 있다고 가정하지만, 본 연구에서는 이러한 가정을 완화하고 통계적 방법을 통해 이러한 매개변수를 추정합니다. 이를 통해 JKO 스킴은 추정된 값을 포함하여 작동하도록 조정됩니다.

- **Technical Details**: Langevin 방정식(Langevin equations) 및 Fokker-Planck 방정식을 이용하여 모델의 동적 거동을 분석합니다. 이 과정에서, 통계적 추정치를 기반으로 한 JKO 스킴을 구성하고, 샘플 크기와 알고리즘 반복 수가 무한대로 갈 때의 한계 동작을 수학적으로 설명합니다. 또한, 랜덤 한계 행동과 확률 분포의 특성을 이해하기 위한 새로운 비율 이론(asymptotic distributional theory)을 제시합니다.

- **Performance Highlights**: 제안된 방법의 성능을 평가하기 위해 수치 시뮬레이션을 수행했습니다. 이 시뮬레이션을 통해 최종적으로 도출된 이론의 타당성을 확인하고, 알고리즘의 반복 횟수가 증가함에 따라 동적 행동이 어떻게 변화하는지를 분석합니다. 이러한 분석은 학습 알고리즘의 동적 및 수렴 성질을 정량화하는 데 도움을 줍니다.



### A Hybrid Framework for Reinsurance Optimization: Integrating Generative Models and Reinforcement Learning (https://arxiv.org/abs/2501.06404)
- **What's New**: 이 논문은 재보험 최적화에 대한 새로운 하이브리드 프레임워크를 제안합니다. 이 프레임워크는 생성 모델인 Variational Autoencoders (VAEs)와 Proximal Policy Optimization (PPO)을 이용한 강화 학습 (Reinforcement Learning) 기술을 결합하여 동적이고 확장 가능한 재보험 전략을 최적화합니다. 이 방법론은 복잡한 청구 분포의 생성 모델링과 강화 학습의 적응적 의사 결정 기능을 통합하여 재보험 전략의 커스터마이징을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 보험사의 운영을 모델링하기 위해 이산 시간 모델링을 기반으로 합니다. 각 시간 간격 동안 청구를 발생시키고, 보험료 수익을 창출하며, 포트폴리오의 리스크를 관리하는 과정을 포함합니다. 또한, 이러한 시간적 구조는 보험업계에서의 재보험 전략 최적화 시 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과는 전통적인 최적화 기법에 비해 우수한 적응성, 확장성 및 강건성을 보여줍니다. 하이브리드 프레임워크는 높은 최종 잉여금과 컴퓨터 효율성을 달성하여 다양한 스트레스 테스트 시나리오(예: 팬데믹 영향, 재해 사건)에서도 안정성을 유지합니다. 이 연구는 재보험 최적화에서 머신러닝과 AI 기술의 통합이 가지는 변혁적 가능성을 입증하였습니다.



### Has an AI model been trained on your images? (https://arxiv.org/abs/2501.06399)
- **What's New**: 이 논문에서는 생성 AI 모델이 특정 이미지 또는 이미지 집합으로 훈련되었는지를 판단할 수 있는 새로운 방법을 제시합니다. 이 방법은 컴퓨터 자원을 효율적으로 소모하며, 모델 아키텍처나 가중치에 대한 명시적인 지식이 필요하지 않습니다(black-box membership inference). 이는 기존 모델의 감사( auditing )와 더 공정한 생성 AI 모델의 발전 및 배포에 중요한 역할을 할 것으로 예상됩니다.

- **Technical Details**: 연구에서 활용된 기법은 대규모 웹 스크래핑을 통해 수집된 방대한 데이터셋을 기반으로 하여, 이미지의 내용을 기반으로 in-training(훈련)에 사용된 이미지와 out-of-training(훈련에서 제외된) 이미지를 비교합니다. 연구에서는 Stable Diffusion(스테이블 디퓨전), Midjourney(미드저니), DALL-E(달리)와 같은 다양한 생성 모델에 대해 이 방법을 적용했습니다. 각 이미지 쌍은 콘텐츠의 의미 면에서 유사성을 가지도록 구성되며, 이를 통해 이미지 간의 관측된 차이가 의미적 차이에 의한 것이 아님을 보장합니다.

- **Performance Highlights**: 제안된 방법은 강력한 생성 AI 모델의 훈련 데이터셋에 대한 투명성을 높이고, 생성 모델이 특정 콘텐츠를 사용하여 훈련되었는지 여부를 판단할 수 있게 해줍니다. 이는 특히 저작권 및 공정 사용 문제에 대한 논의가 있을 때 중요하게 여겨지며, AI 모델의 윤리적 개발과 관련된 중요한 기준을 제시합니다. 기술적으로, 이 방법은 계산 효율성이 뛰어나고 다양한 모델 아키텍처에 일반화될 수 있는 가능성을 가지고 있습니다.



### Counterfactually Fair Reinforcement Learning via Sequential Data Preprocessing (https://arxiv.org/abs/2501.06366)
- **What's New**: 이 논문은 의학 분야에서 강화 학습(Reinforcement Learning, RL)을 활용하여 공정한 연속적 의사 결정을 위한 새로운 프레임워크를 제안합니다. 특히, 반사실적 공정성(Counterfactual Fairness, CF)을 이용하여 특정 하위 집단에 대한 치료 자원의 불공정 배분 문제를 해결하고자 합니다. 이론적으로 최적의 CF 정책을 특성화하고, 그 정지성을 입증하여 기존 RL 알고리즘을 활용해 최적의 CF 정책 탐색을 단순화합니다.

- **Technical Details**: 이 연구에서는 CF를 기반으로 한 의사 결정 과정을 위해 RL 알고리즘 내의 민감한 속성이 미치는 직접적 및 간접적 영향을 분리하는 것이 대단히 중요하다고 강조합니다. 논문에서 제안된 알고리즘은 추가적 소음 가정 하에서 최적 CF 정책 학습을 위한 연속 데이터 전처리 방법을 포함하고 있으며, 이는 불공정성 관리를 위한 이론적 보장을 제공합니다. 결과적으로, 본 연구는 RL 알고리즘의 성능을 개선할 수 있는 획기적인 접근 방식을 제공합니다.

- **Performance Highlights**: 디지털 건강 데이터 세트를 분석한 결과, 제안된 방법이 상담 접근의 공정성을 크게 향상시켰음을 보여줍니다. 실제 응용 프로그램인 PowerED 연구에서 도출된 결과는 의료 분야에서의 불공정성을 제어하고 최적의 가치를 달성할 수 있음을 입증합니다. 시뮬레이션 및 실제 데이터 분석을 통해 불공정성을 효과적으로 제어하고 최적의 결과를 달성하는 것을 확인하였습니다.



### MEt3R: Measuring Multi-View Consistency in Generated Images (https://arxiv.org/abs/2501.06336)
Comments:
          Project website: this https URL

- **What's New**: 새로운 메트릭 MEt3R가 제안되었으며, 이는 생성된 이미지의 다중 시점 일관성을 측정하는 데 중점을 두고 있습니다. 기존의 복원 메트릭들은 생성된 출력의 품질을 정확하게 측정할 수 없기 때문에, 비효율적인 대안이 필요했습니다. MEt3R는 특정 장면에 국한되지 않고, 다양한 조명 조건에서도 일관성을 평가할 수 있도록 설계되었습니다.

- **Technical Details**: MEt3R는 DUSt3R를 활용하여 이미지 쌍으로부터 밀접한 3D 재구성을 수행하고, 영상의 특징 맵을 비교하여 유사성을 평가합니다. DINO와 FeatUp을 특징 추출기로 사용하여 조명과 같은 시점 관련 효과에 강건한 특징을 얻습니다. 이 메트릭은 이미지 품질 메트릭과는 독립적으로 작동하며, 이진적이지 않고 점진적인 일관성 측정을 제공합니다.

- **Performance Highlights**: MEt3R는 다중 시점 이미지를 생성하는 기존 방법들의 일관성을 평가하는 데 사용되며, 특히 MV-LDM이 품질과 일관성 간의 trade-off에서 좋은 성능을 보임을 증명했습니다. 이전 메트릭들과의 비교에서 MEt3R는 아주 일관된 시퀀스와 거의 일관된 시퀀스를 구분할 수 있습니다. 이 메트릭은 개방형 소스 다중 시점 잠재적 확산 모델인 MV-LDM과 함께 잘 정렬된 일관성 측정을 가능하게 합니다.



### Tensorization of neural networks for improved privacy and interpretability (https://arxiv.org/abs/2501.06300)
Comments:
          39 pages, 9 figures

- **What's New**: 이번 논문에서는 텐서 트레인(Tensor Train) 표현을 구축하기 위한 텐서화(tensorization) 알고리즘을 소개합니다. 이 방법은 목표 함수에 대해 블랙 박스 액세스와 관심 영역을 정의하는 적은 수의 샘플 포인트만을 요구하며, 이는 특히 머신러닝 모델에 적합합니다. 이 알고리즘은 신경망 모델의 프라이버시와 해석 가능성을 향상하는 데 활용될 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 Hur et al.의 스케치 방법을 적응하여 신경망 모델을 텐서 트레인 형식으로 근사화하는 Tensor Train via Recursive Sketching from Samples (TT-RSS)을 포함합니다. 이 방법은 고차원 및 희소성 환경에 적합하며, 일반 함수에 대한 계산을 가속화하는 데에도 사용될 수 있습니다. 텐서화를 통해 우리는 neural network의 데이터 유출을 줄일 수 있다는 점이 주목할 만합니다.

- **Performance Highlights**: 우리가 제안한 알고리즘을 통해 MNIST 및 소리 분류와 같은 다양한 모델을 근사하는 데 성공했습니다. 실험 결과, 텐서화된 모델이 원래 신경망에 비해 비슷한 정확도를 유지하면서 데이터 유출을 크게 줄이는 것으로 나타났습니다. 이러한 성과는 신경망 양자 상태(NNQS)의 특성을 탐구할 수 있는 유망한 길을 제시합니다.



### Punctuation's Semantic Role between Brain and Transformers Models (https://arxiv.org/abs/2501.06278)
- **What's New**: 최근 자연어 처리(NLP) 모델들이 인간의 뇌 활동과 어떻게 정합하는지를 탐구하는 연구가 진행되고 있습니다. 본 연구에서는 RoBERTa 동일한 방법론을 적용하여 새로운 NLP 모델의 뇌 활동과의 정합성을 평가하였습니다. 특히, 구두점(punctuation)을 제거한 텍스트가 세멘틱(processing) 처리에 미치는 영향을 조사하였습니다.

- **Technical Details**: 연구는 BERT를 기반으로 한 네 가지 변형 모델(RoBERTa, DistiliBERT, ELECTRA, ALBERT)에 대한 뇌-신경망 정합(matching) 평가를 통해 진행됩니다. 비슷한 방법을 사용하여, 인간의 뇌가 언어를 이해하는 방식을 보다 잘 이해하기 위한 고안된 실험들이 포함됩니다. fMRI 및 MEG 뇌 기록을 사용하여 모델의 특성과 뇌 데이터를 일치시키는 방법이 제안되었습니다.

- **Performance Highlights**: 연구 결과, RoBERTa 모델이 BERT보다 뇌 활동과 더 잘 일치하며 정확도에서도 우수한 성과를 보였습니다. BERT의 경우 구두점을 제외했을 때 더 높은 정확도를 기록하였고, 구두점을 포함한 원래 결과와 비교했을 때 문맥 길이가 늘어나도 정확도에 큰 영향을 미치지 않았습니다.



### What Matters for In-Context Learning: A Balancing Act of Look-up and In-Weight Learning (https://arxiv.org/abs/2501.06256)
- **What's New**: 대형 언어 모델(LLM)에서의 In-Context Learning (ICL) 메커니즘을 체계적으로 분석한 연구입니다. 이 연구에서는 훈련 데이터의 특성과 모델 아키텍처의 역할을 넘어서서 ICL의 근본적인 요소를 파악하고자 했습니다. 데이터 시퀀스에서 개념적 반복(conceptual repetitions)의 중요성을 강조하고, 이는 훈련 과정에서 안정적이고 비일시적인 ICL 성능을 보장하는 데 기여한다고 주장합니다.

- **Technical Details**: 연구에서 저자들은 깊은 자회귀 모델을 통해 훈련 데이터 시퀀스의 개념적 반복이 ICL에 미치는 영향을 조사했습니다. 특히, 2048 토큰의 컨텍스트 윈도우에서 n-gram 반복성을 관찰했고, 이는 훈련 예시 내에서의 전환 가능성을 높이는 데 중요하다고 설명합니다. 또한 복잡한 인-웨이트 러닝(in-weight learning) 목표가 ICL의 일관된 성능에 중요한 역할을 한다고 제안하며, 반복과 복잡한 목표 간의 상호작용을 연구했습니다.

- **Performance Highlights**: 저자들은 데이터 시퀀스의 단일 반복만으로도 ICL이 가능하다는 것을 발견하였고, 이는 높은 폭발성(burstiness)이나 skewed label distribution과 같은 다른 특성이 없더라도 성립됩니다. 연구 결과, LLM에서 ICL 성능은 훈련 초기에 일시적인 경향을 보일 수 있으나 반복과 복잡한 목표를 통해 안정적으로 유지될 수 있음을 확인했습니다. 이러한 발견은 대형 모델에서 ICL의 메커니즘을 이해하는 데 중요한 기여를 할 것으로 기대됩니다.



### Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words (https://arxiv.org/abs/2501.06254)
- **What's New**: 이번 논문에서는 Sparse Autoencoders (SAEs)의 새로운 평가 방식을 제안하며, polysemous words (다의어)에서 monosemantic features (단어의 의미를 명확하게 표현하는 특징)를 추출할 수 있는지 분석합니다. 기존의 MSE-L0 Pareto frontier 성능 향상이 단순히 interpretability (해석 가능성)를 증가시키지 않는다는 점을 강조합니다. 또한 Poly-Semantic Evaluations (PS-Eval)이라는 새로운 메트릭을 통해 SAEs의 성과를 정량적으로 평가할 수 있는 방법을 제시합니다.

- **Technical Details**: SAEs는 LLM의 복잡한 세 가지 차원의 활성도를 단일 의미의 피쳐로 변환하기 위해 설계되었습니다. 본 연구는 PS-Eval을 통해 polysemantic activations (다의적 활성화)에서 monosemantic features (단일 의미 특징)를 효과적으로 추출할 수 있는지 평가합니다. 실험에서는 다양한 activation functions (활성화 함수)와 latent dimensions (잠재 변수를 사용하는 차원)을 시험했고, Attention module의 깊이에 따라 해석의 정확성이 어떻게 변화하는지를 탐구했습니다.

- **Performance Highlights**: PS-Eval의 결과는 SAEs가 다양한 polysemous words에서 명확한 단일 의미의 피쳐를 추출한다는 것을 보여줍니다. 특히, deeper layers (더 깊은 층)에서는 specificity (특정성) 점수가 높아지며, Attention mechanism (어텐션 메커니즘)이 다의어를 구분하는 데 효과적임을 확인했습니다. 최종적으로, MSE와 L0 sparsity에만 집중하는 기존의 연구들이 monosemantic feature의 효과적인 추출을 간과함을 일깨워주는 성과를 도출하였습니다.



### A Survey on Algorithmic Developments in Optimal Transport Problem with Applications (https://arxiv.org/abs/2501.06247)
- **What's New**: 이번 논문에서는 Optimal Transport (OT) 이론이 다양한 분야에서 분포 간의 차이를 정량화하기 위한 강력한 체계로 자리 잡았음을 강조합니다. 특히, OT는 기계 학습, 데이터 과학 및 컴퓨터 비전 등에서의 응용을 포함하여, 이론적 기초인 Monge 및 Kantorovich의 고전적 정식과 이의 현대적 컴퓨팅 기법으로의 확장을 자세히 다룹니다. 또한, Sinkhorn iterations와 같은 최신 알고리즘의 효율성과 고차원 문제 해결을 위한 확장성을 강조합니다.

- **Technical Details**: OT 이론은 확률 분포의 비교 및 변환을 위한 강력한 수학적 틀을 제공합니다. 초기 Monge의 기여에서 시작하여 후에 Kantorovich에 의해 확장되고 형식화된 OT는 이산 및 연속 확률 측정을 다루기 위한 통합된 접근 방식을 제공합니다. 이 논문에서는 OT의 기본 이론 개념과 함께 각 개념의 수학적 특성을 설명하며, 최신 알고리즘적 접근 방식과 계산 기술에 대한 혁신을 심도 있게 탐구합니다.

- **Performance Highlights**: OT의 발전은 특히 대규모 데이터셋에 대한 효율성을 개선하여, 다양한 분야에서의 응용 가능성을 확장하였습니다. Wasserstein 거리 사용의 향상은 확률 분포 비교를 통한 OT의 적용 폭을 넓히고 있습니다. 그러나 OT 알고리즘의 계산 복잡성 및 해결 정확성과 실행 속도 간의 트레이드오프에 관한 논의는 여전히 진행 중이며, 윤리적 고려 사항 또한 중요한 문제로 남아있습니다.



### Microservice Deployment in Space Computing Power Networks via Robust Reinforcement Learning (https://arxiv.org/abs/2501.06244)
Comments:
          14 pages

- **What's New**: 최근 지구 관측에 대한 수요가 증가함에 따라, 실시간 원격 감지 추론 서비스의 신뢰성을 제공하는 것이 중요해졌습니다. 본 논문에서는 발사체에 통합된 컴퓨팅 및 넓은 커버리지를 제공하는 Space Computing Power Network (Space-CPN)을 통해 저지연 요구를 충족하고자 하는 새로운 원격 감지 인공지능 응용 프로그램 배포 프레임워크를 제안합니다. 이 프레임워크는 마이크로서비스 아키텍처(microservice architecture)를 사용하여 단일 서비스 과제를 재사용 가능한 독립 모듈로 분해하고, 위성 집합체에서의 실시간 추론 성능을 달성하는 방법을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 고전적인 강화 학습(robust reinforcement learning)과 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)을 활용하여 데이터 불확실성을 다루고 있습니다. 이를 통해 분산된 마이크로서비스 배포를 최적화하여 리소스를 효율적으로 사용하면서 품질(Quality of Service, QoS)과 기능 요구 사항을 충족할 수 있습니다. 주의해야 할 점은 위성의 이질성(heterogeneity)으로 인해 각 위성이 다른 처리 능력과 통신 능력을 가지므로 가장 적합한 위성을 선택하는 알고리즘의 필요성이 강조됩니다.

- **Performance Highlights**: 시뮬레이션 결과는 제안된 프레임워크가 리소스 소비를 최소화하고 QoS 페널티를 피하는 데 효과적임을 보여줍니다. 또한, 론칭 시 미리 할당된 작업과 긴급 임무를 동시에 처리할 수 있는 가능성을 제공하여, 지구 관측 및 재해 모니터링 작업의 효율성을 크게 향상시킬 수 있습니다. 이러한 접근은 기존의 LEO 위성 환경에서 다수의 고정된 자원 한계를 극복할 수 있는 가능성을 나타냅니다.



### Forecasting Anonymized Electricity Load Profiles (https://arxiv.org/abs/2501.06237)
- **What's New**: 전력 소비 패턴의 익명성을 보장하기 위한 방법으로, 이 글에서는 마이크로 집합(microaggregation) 기법이 에너지 수요 예측에 미치는 영향을 분석했습니다. 특히 개인 행동 데이터로 간주되는 전력 소비 데이터는 GDPR(General Data Protection Regulation)에 의해 엄격한 보호 조치가 요구됩니다. 이를 통해 개인정보를 보호하면서도 예측력에 미치는 부정적인 영향을 최소화할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 영국 파워 네트웍스의 Low Carbon London 프로젝트에서 수집된 데이터를 사용하였으며, 주로 전력 소비량의 시계열(time series) 분석을 통해 마이크로 집합 기법의 효과를 평가했습니다. 마이크로 집합은 비슷한 데이터 포인트를 그룹화하여 개별 신원을 숨기면서도 데이터 세트의 주요 속성을 유지합니다. 사용된 기법에는 k-익명성과 마이크로 집합이 포함되며, DFTMicroagg 알고리즘은 추가적인 데이터 보호를 위해 제안되었지만 이는 사용하지 않았습니다.

- **Performance Highlights**: 결과적으로 마이크로 집합을 적용한 데이터는 예측 정확도에 미치는 영향이 거의 없어 높은 데이터 유용성을 유지하며 사용자 프라이버시를 보장합니다. 연구는 에너지 공급업체가 사용자 동의 없이 스마 부하 프로파일 데이터에 접근하더라도 GDPR을 준수하면서 협력할 수 있는 가능성을 제시합니다. 이는 에너지 공급업체와 제3자가 스마트 미터 데이터를 안전하게 이용할 수 있는 방법을 제공합니다.



### Generating and Detecting Various Types of Fake Image and Audio Content: A Review of Modern Deep Learning Technologies and Tools (https://arxiv.org/abs/2501.06227)
- **What's New**: 이 논문은 최신 과학 발전에 기반한 현대 딥러닝 기술을 활용한 딥페이크 생성 및 탐지의 최전선 상태를 검토합니다. 딥페이크의 증가는 개인 정보, 보안 및 민주주의에 상당한 위협을 주며, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs) 및 Diffusion 모델과 같은 기법을 활용합니다. 또한, 딥페이크가 개인과 조직을 속이고, 협박을 용이하게 하며, 법적 및 사회적 시스템의 무결성을 위협하는 방식에 대해 논의합니다.

- **Technical Details**: 이 연구에서는 얼굴 전환(face swapping), 음성 변환(voice conversion), 재연(reenactment) 및 입술 동기화(lip synchronization)와 같은 다양한 딥페이크 생성 방법을 다루며, 이들이 선의 및 악의 맥락에서 어떻게 사용되는지를 강조합니다. 또한, 딥페이크 생성과 탐지 간의 "무기 경쟁(arms race)"를 살펴보고, 조작된 콘텐츠를 식별하는 데 있는 어려움을 분석합니다. 이는 이 분야의 복잡성을 이해하는 데 기여합니다.

- **Performance Highlights**: 현재의 탐지 방법을 검토하고 향후의 연구 방향을 강조함으로써, 이 논문은 딥페이크 기술의 남용을 막기 위한 강력한 탐지 전략이 시급하다는 점을 명확히 합니다. 본 연구는 오디오, 이미지 및 비디오 분야의 최신 동향을 쉽게 이해할 수 있도록 구성되어 있어, 딥페이크 생성 및 탐지에서의 최신 발전을 파악하는 데 큰 도움이 됩니다.



### A Distributed Hybrid Quantum Convolutional Neural Network for Medical Image Classification (https://arxiv.org/abs/2501.06225)
- **What's New**: 이 논문에서는 분산 하이브리드 양자 컨볼루션 신경망(distributed hybrid quantum convolutional neural network)을 제안하여 의료 영상 분류의 효율성과 정확성을 개선하려고 합니다. 이를 위해 양자 회로 분할(quantum circuit splitting) 기술을 활용하여 양자 자원을 절약하면서도 8-qubit QCNN을 5-qubit 시스템에서 실행할 수 있도록 설계되었습니다. 이 모델은 이전 기술에 비해 적은 수의 파라미터로도 뛰어난 성능을 보입니다.

- **Technical Details**: 모델은 MobileNetV2 아키텍처를 활용하여 의료 영상의 고차원 특징을 효과적으로 추출합니다. 훈련 데이터셋을 통해 학습한 후, 분산 8-qubit 양자 컨볼루션 신경망에서 특징 벡터를 입력으로 사용하여 분류 작업을 수행합니다. 제안된 시스템은 CNN을 통하여 특징 추출 및 차원 축소를 먼저 수행하고, 이후에 양자 회로 분할 기술을 통한 양자 컨볼루션 신경망으로 전환합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 3개의 데이터셋에서 이진 및 다중 클래스 분류 작업 모두에서 우수한 성능을 기록했습니다. 특히, 최근의 여러 모델에 비해 적은 수의 파라미터로도 높은 정확도를 달성하여 양자 계산의 장점을 잘 보여줍니다. 이 모델의 설계는 의료 영상 분석을 위한 QCNN의 활용 가능성을 더욱 확장합니다.



### Interpretable Auto Window Setting for Deep-Learning-Based CT Analysis (https://arxiv.org/abs/2501.06223)
- **What's New**: 이번 연구에서는 의료 영상 분석에서 중요한 CT의 Auto Window Setting을 위한 새로운 플러그앤플레이 모듈을 제안합니다. 이 모듈은 Tanh 활성화 함수에서 유래하며, 주요 딥러닝 아키텍처와 호환됩니다. 기존의 수동적인 창 설정 방법을 대체할 수 있는 자동화된 접근 방식으로, 임상적으로 직관적인 관점에서 결과를 해석할 수 있는 기능을 제공합니다.

- **Technical Details**: 제안된 모듈은 두 가지 주요 설계 원칙에 따라 구성됩니다. 첫째, 도메인 불변(design)을 기반으로 하여 다양한 HU(hounsfield unit) 값을 동적으로 처리할 수 있습니다. 둘째, 각 서브 모듈이 독립적으로 분석 및 최적화될 수 있는 구조로 되어 있습니다. 이로 인해 사용자는 모델 조정에 더 많은 유연성을 갖게 되며, 명료한 분석 및 관찰 통찰을 제공합니다.

- **Performance Highlights**: 모듈의 효과는 여러 개방형 데이터셋에서 검증되었으며, 하드 세그먼트 타겟에서 10%에서 200%까지의 Dice score 개선을 기록했습니다. 이는 임상 현장에서 CT 이미지 분석을 수행할 때 보다 나은 성능을 제공할 것으로 기대됩니다. 특히, 자동화된 방식으로 인해 수동 설정의 변수를 줄이고 더욱 신뢰할 수 있는 결과를 도출할 수 있습니다.



### WhACC: Whisker Automatic Contact Classifier with Expert Human-Level Performanc (https://arxiv.org/abs/2501.06219)
- **What's New**: 이번 연구에서는 설치류의 촉각 인식을 자동으로 식별할 수 있는 Whisker Automatic Contact Classifier(WhACC)라는 파이썬 패키지를 개발하였습니다. 이 패키지는 ResNet50V2를 기반으로 한 특징 추출 및 LightGBM을 활용한 분류 기술을 사용하여, 사람 수준의 성능으로 고속 비디오에서의 터치를 식별할 수 있습니다. WhACC는 100백만 프레임 데이터셋에서 수작업으로 필요한 시간을 약 333시간에서 6시간으로 줄이는 데 기여했습니다.

- **Technical Details**: WhACC는 설치류의 머리가 고정된 상태에서의 행동을 기록한 고속 비디오 데이터를 기반으로 하여, 61x61 픽셀의 아무것도 없는 객체 주변의 픽셀을 추출하여 사용합니다. 이 모델은 2048개의 특징에서 99.5%의 동의율로 전문가의 커스터마이징 결과와 일치하여, 터치 시간을 밀리초 수준의 정확도로 식별할 수 있습니다. 이미지 데이터의 증강과 후행 이미지 기법을 통해 다양한 실험 환경에서도 좋은 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: WhACC의 성능은 1백만 프레임 이상의 비디오 데이터에서 세 명의 전문가와의 쌍별 터치 분류 동의 비율이 99.46%에 달하는 것으로 평가되었습니다. 모델의 오류 측정을 통해 주로 발생하는 에러 유형인 엣지 오류는 7배 더 빈번하게 발생하였으며, 이는 신경 반응 분석에 미치는 영향을 파악하기 위해 중요한 지표입니다. 전체적인 목표는 다양한 촬영 조건에서도 전문가 수준의 정확도를 유지하며 손으로 커스터마이징하는 시간적 부담을 최소화하는 것이었습니다.



### Fitting Different Interactive Information: Joint Classification of Emotion and Intention (https://arxiv.org/abs/2501.06215)
- **What's New**: 이 논문은 ICASSP MEIJU@2025 트랙 I의 1등 해법으로, 저자원 환경에서의 다중모드 감정 및 의도 인식을 중심으로 하고 있습니다. 특정한 대화에서 다양한 난이도의 작업 증진을 위한 접근법과 막대한 양의 레이블 없는 데이터를 효과적으로 활용하는 방법이 주된 초점입니다. 이를 통해 저자들은 0.5532의 점수로 트랙에서 우승을 차지했습니다.

- **Technical Details**: 연구의 모델 아키텍처는 다중 헤드 자기 주의(multi-head self-attention)와 게이팅 메커니즘을 통한 상호 작용 모듈(combination module)을 기반으로 구성됩니다. 비디오에서 매 30프레임마다 샘플링된 이미지를 사용하고, 오디오 데이터는 ffmpeg를 활용하여 WAV 형식으로 변환합니다. 텍스트 데이터에서 발견된 오류는 수작업으로 수정되어 감정과 의도를 더 잘 분류할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면 감정 인식과 의도 인식은 서로 관련이 있으나, 두 과제를 동시에 최적의 정확도로 수행하기는 어렵다는 점이 발견되었습니다. 다중 헤드 자기 주의 방식을 통해 의도 인식의 성능이 향상되었으며, 두 단계의 훈련 전략을 통해 최종 점수 0.5532를 달성했습니다. 이러한 접근 방식은 작업 성능을 크게 향상시켰습니다.



### Applications of natural language processing in aviation safety: A review and qualitative analysis (https://arxiv.org/abs/2501.06210)
- **What's New**: 이번 연구는 항공 안전 분야에서 자연어 처리(Natural Language Processing, NLP)를 이용한 기계 학습 알고리즘의 적용 가능성을 탐구하고 있습니다. 현재까지 '자연어 처리'와 '항공 안전'을 키워드로 검색한 결과, 2024년 5월 기준 34개의 스코퍼스(Scopus) 논문이 발견되었습니다. 이러한 연구들을 분석함으로써, NLP의 항공 분야에서의 방법론, 발견 사항 및 시사점을 파악할 수 있습니다.

- **Technical Details**: 연구에서는 현재 항공 안전을 위한 NLP 문헌의 상태를 조사하기 위해 질적(qualitative) 및 양적(quantitative) 도구를 활용하였습니다. 질적 분석에서는 연구의 동기, 목표 및 결과를 요약하여 NLP가 항공 안전의 중요한 문제를 식별하고 향상하는 데 어떻게 기여할 수 있는지를 보여줍니다. 또한 이 연구는 연구의 공백을 식별하고 항공 산업을 위한 실질적인 권장 사항을 제시합니다.

- **Performance Highlights**: NLP를 항공 안전에 적용할 때 발생하는 도전 과제로는 대규모로 주석이 달린 데이터셋의 필요성과 복잡한 모델 해석의 어려움이 있습니다. 연구에서는 데이터 주석을 위한 능동 학습(active learning)과 모델 해석을 위한 설명 가능한 AI(explainable AI)와 같은 해결 방안을 제안합니다. 사례 연구들은 NLP가 항공 안전을 증진시키는데 성공적으로 적용된 예시를 보여주며, 항공 안전이 보다 안전하고 효율적이 될 수 있는 잠재력을 강조합니다.



### CoNOAir: A Neural Operator for Forecasting Carbon Monoxide Evolution in Cities (https://arxiv.org/abs/2501.06007)
Comments:
          28 pages, 14 figures, under submission process

- **What's New**: 본 논문에서는 도시 지역에서의 주요 오염 물질인 일산화탄소(CO) 예측을 위한 새로운 머신러닝 모델, Complex Neural Operator for Air Quality (CoNOAir)를 소개합니다. 이 모델은 짧은 시간(시간 단위)과 긴 시간(72시간) 예측 모두에서 뛰어난 성능을 발휘하며, 기존의 최첨단 모델인 Fourier Neural Operators (FNO)를 능가하는 것으로 나타났습니다.

- **Technical Details**: CoNOAir는 물리학과 화학 기반 시뮬레이션의 높은 계산 비용 문제를 해결하기 위해 설계되었습니다. 이 모델은 도시와 국가 범위에서 신뢰할 수 있는 CO 농도 예측을 가능하게 하며, 극단적인 사건을 포착하는 능력 또한 분석되었습니다. 모델은 인도의 여러 도시에서 가능한 예측을 통해 R2 값을 0.95 이상 기록하며, 매우 정확한 예측 결과를 제공합니다.

- **Performance Highlights**: 이러한 성과는 도시 정부가 조기 경고 시스템을 제공하고, 개입 전략을 계획하며, 다양한 시나리오를 고려하여 효과적인 대책을 개발하는 데 큰 도움이 될 것입니다. 결국, 본 연구의 접근 방식은 도시 지역에서의 CO 오염 실시간 예측 제공을 크게 촉진할 것으로 기대됩니다.



### RMTransformer: Accurate Radio Map Construction and Coverage Prediction (https://arxiv.org/abs/2501.05190)
Comments:
          Submitted to IEEE VTC 2025 Spring

- **What's New**: 이번 논문에서는 무선 네트워크 모델링 및 관리에 중요한 라디오 맵 예측을 위해 하이브리드 변환기-합성곱 모델인 RMTransformer를 소개합니다. RMTransformer는 전통적인 CNN 기반 방법에 비해 계산 오버헤드가 적고 예측 오차를 낮추며, 지리적 맵에서 경로 손실 패턴을 구성하기 위해 멀티 스케일 변환기 기반 인코더와 합성곱 기반 디코더를 특징으로 합니다. 이 연구는 30% 이상의 RMSE 감소를 보여주며, 이는 기존의 최첨단(SOTA) 접근 방식에 비해 상당한 개선을 나타냅니다.

- **Technical Details**: RMTransformer는 인코더-디코더 구조를 기반으로 설계되어, 변환기 기반 인코더가 지리적 맵으로부터 효율적으로 특징을 추출합니다. 이를 통해 다양한 크기를 갖는 다차원 특징을 생성하고, CNN 디코더로 픽셀 수준의 라디오 맵을 정밀하게 재구성합니다. 또한, 멀티 스케일 변환기 아키텍처를 사용하여 다양한 차원을 가진 특징을 생성하고, 이를 건너뛰기 연결을 통해 CNN 블록으로 전달하여 이미지 재구성의 효율성을 높입니다.

- **Performance Highlights**: 제안된 RMTransformer는 공개 라디오 맵 데이터셋에서 평가되었고, 뛰어난 예측 정확성을 달성했습니다. 예를 들어, RMSE의 10^{-3} 수준에 도달하며, PMNet과 비교하여 30% 이상의 개선을 이루었습니다. 이는 복잡한 무선 전파 환경을 세밀하게 이해할 수 있는 능력을 강조합니다.




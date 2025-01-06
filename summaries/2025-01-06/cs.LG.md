New uploads on arXiv(cs.CL)

### Metadata Conditioning Accelerates Language Model Pre-training (https://arxiv.org/abs/2501.01956)
Comments:
          Code available at this https URL

- **What's New**: 이번 논문에서는 Metadata Conditioning then Cooldown(메타데이터 조건부 훈련 후 쿨다운, MeCo)이라는 새로운 방법을 제안합니다. 이 방법은 언어 모델의 미리 훈련 과정에서 메타데이터를 추가하여 효율적으로 다양한 스타일과 품질 수준을 반영합니다. MeCo는 메타데이터를 포함한 훈련과 쿨다운 단계를 통해 메모리 소모를 줄이고 메타데이터 없이도 정상 작동할 수 있게 하여, 훈련 데이터량을 33% 감소시키면서도 유사한 성능을 보여줍니다.

- **Technical Details**: MeCo 방법론은 훈련 데이터에 메타데이터(예: 문서 URL)를 포함하여 90%의 훈련을 진행하고, 마지막 10%는 메타데이터 없이 표준 데이터로 훈련하는 두 단계로 구성됩니다. 이를 통해 모델은 메타데이터에 의존하지 않고도 성능을 유지합니다. 추가적으로, MeCo는 다양한 유형의 메타데이터와 호환되며, 메타데이터가 없을 때 성능 저하를 방지합니다.

- **Performance Highlights**: MeCo는 600M부터 8B 매개변수까지 다양한 모델 크기에서 일관된 성능 향상을 보여주며, DCLM, RefinedWeb, 그리고 C4 등의 훈련 데이터 원천에서 실험하였습니다. 특히, 1.6B 모델은 표준 훈련 모델과 동일한 다운스트림 성능을 유지하면서도 훈련 데이터 사용량을 33% 줄이는 성과를 보였습니다. 이러한 결과는 MeCo가 언어 모델의 데이터 효율성을 크게 개선할 수 있는 가능성을 제시합니다.



### Abstractive Text Summarization for Contemporary Sanskrit Prose: Issues and Challenges (https://arxiv.org/abs/2501.01933)
Comments:
          PhD Thesis

- **What's New**: 이 논문은 현대 산스크리트 산문에 대한 추상적 텍스트 요약(abstractive text summarization) 모델을 제안합니다. 주요 연구 질문은 산스크리트어에 대한 추상적 텍스트 요약을 개발하는 데 있어 어떤 도전 과제가 있는지에 대한 것입니다. 이 연구는 낮은 자원(low-resource) 인플렉션(inflectional) 언어인 산스크리트어의 특수를 다루고 있습니다.

- **Technical Details**: 본 논문의 두 번째 장인 문헌 리뷰(literature review)에서는 이전 연구들을 조사하였고, 세 번째 장에서는 데이터 준비(data preparation) 단계에서 언어 모델(language model)과 요약 모델(summarization model) 훈련을 위한 데이터 수집 및 전처리(preprocessing) 문제를 다루었습니다. 네 번째 장에서는 모델의 훈련(training)과 추론(inference) 과정 및 그 결과를 보고합니다.

- **Performance Highlights**: 이 연구는 산스크리트어 추상적 텍스트 요약을 위한 파이프라인(pipeline)을 시작하였으며, 개발의 각 단계에서 직면한 도전 과제를 보고하였습니다. 모든 주제에 기반한 연구 질문들은 주요 연구 질문에 대한 답변을 제공하기 위해 다루어졌습니다.



### Long Context vs. RAG for LLMs: An Evaluation and Revisits (https://arxiv.org/abs/2501.01880)
Comments:
          14 pages excluding references and appendix

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 외부 컨텍스트 통합을 위한 두 가지 주요 전략인 긴 컨텍스트(LC)와 검색 증강 생성(RAG)의 최근 연구를 재조명합니다. 연구자들은 질문의 외부 맥락의 필요성을 강조하며, 가장 효과적인 검색 방법을 식별하고 데이터 세트를 확장함으로써 보다 포괄적인 평가를 제공하고 있습니다. 실험 결과는 LC가 RAG에 비해 일반적으로 더 우수하다는 것을 보여주며, 특히 위키백과 기반 문제에서 그 성능이 두드러집니다.

- **Technical Details**: LLMs는 제한된 컨텍스트 창으로 인해 신뢰할 수 있는 최신 데이터를 제공하기 위해 외부 메모리를 통합해야 한다고 언급합니다. 이를 위해 LC와 RAG 두 가지 접근 방식을 채택하며, LC는 더 많은 정보를 읽어들일 수 있는 모델을 구축하고, RAG는 적절한 텍스트 조각을 포함하는 방식을 사용합니다. 연구자들은 12개의 질문 응답 데이터 세트에서 필터링된 소규모 데이터셋을 통해 최고의 리트리버를 식별하고, 두 가지 설정(즉, LC와 RAG)으로 생성된 답변을 비교해 심도 있는 분석을 수행했습니다.

- **Performance Highlights**: 연구 결과, LC 모델은 자기 포함 정보 처리에서 일반적으로 RAG보다 우수하며, RAG는 대화 기반의 정보 처리에서 장점을 보입니다. 이러한 실험은 LC와 RAG의 강점과 한계를 깊이 이해하는 데 기여하며, 효과적인 검색 전략 최적화와 성능 향상을 위한 통합 접근 방안을 제공하고 있습니다. 또한, 기존 연구에서 간과된 컨텍스트의 중요성을 강조하며 향후 연구 방향에 대한 통찰을 제공합니다.



### Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions (https://arxiv.org/abs/2501.01872)
Comments:
          Our code is publicly available at this https URL

- **What's New**: 이번 연구에서는 새로운 jailbreak 기법인 POATE(Polar Opposite query generation, Adversarial Template construction, and Elaboration)를 소개합니다. POATE는 대조적 추론(contrastive reasoning)을 활용하여 비윤리적인 응답을 유도하는 독특한 방법으로, 반대 의미의 프롬프트와 적대적 템플릿을 결합하여 언어 모델을 해치는 방향으로 유도합니다. 기존 안전 방어 시스템의 한계를 드러내면서, 이 공격 기법은 약 44%의 성공률을 기록하며, 이전 방법보다 확연히 향상된 결과를 보여줍니다.

- **Technical Details**: POATE의 공격 방식은 두 단계로 구성됩니다. 첫 번째 단계는 원래의 악의적인 의도를 반대하는 의미의 프롬프트를 생성하는 것으로, 이는 언어 모델이 쉽게 해석할 수 있도록 도와줍니다. 두 번째 단계에서는 첫 번째 단계에서 생성된 프롬프트의 의도에 반하는 행동을 수행하기 위한 적대적 템플릿을 구성하여, 시스템적으로 유해한 프롬프트를 생성하는 체계를 완성합니다.

- **Performance Highlights**: POATE를 통해 여러 개의 언어 모델, 예를 들면 LLaMA-2-7B-chat과 GPT-4o에서 평균 공격 성공률(~57%)을 달성하였는데, 이는 기존의 최첨단 공격 방법(평균 ~22%의 성공률)보다 월등히 높은 수치입니다. 또한, POATE 방어를 위한 체인-오브-생각(chain-of-thought) 접근 방식을 도입하여, POATE의 공격 성공률을 평균 98% 감소시키며, 기존의 방어 기술보다 더 효과적인 성과를 보였습니다.



### Time Series Language Model for Descriptive Caption Generation (https://arxiv.org/abs/2501.01832)
- **What's New**: 본 논문에서는 시계열 데이터의 캡셔닝을 위한 새로운 다중 모달 모델인 TSLM(Time Series Language Model)을 소개합니다. TSLM은 시계열 및 텍스트 데이터를 통합하여 시계열 패턴을 정확하게 설명하는 자연어 문장을 생성하는 능력을 가지고 있습니다. 또한, 기존의 대형 언어 모델(LLM)을 사용하는 데이터 생성 및 노이즈 제거 방법을 통해 교육 데이터 세트를 효과적으로 보강하는 접근 방식을 제공합니다.

- **Technical Details**: TSLM은 인코더-디코더 모델로, 텍스트 프롬프트와 시계열 데이터 표현을 함께 활용하여 미세한 시간적 패턴을 캡처하고 텍스트 설명을 생성합니다. 데이터 부족 문제를 해결하기 위해, TSLM은 고품질 샘플로부터 소수의 학습(few-shot learning)과 오픈 소스 LLM을 이용한 컨텍스트 프롬프트 기법을 적용합니다. 또한, 생성된 데이터를 크로스 모달 밀집 검색 스코어링을 통해 노이즈를 제거하는 새로운 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, TSLM은 다양한 시계열 캡셔닝 데이터 세트에서 기존 최첨단 기법들을 유의미하게 능가하는 성과를 보였습니다. 다중 모달 인코더의 추가적 조정을 통해 시계열 정보와 텍스트 표현 간의 일치를 향상시켜, 보다 정밀하고 유용한 해석 결과를 도출합니다. 이러한 성과는 TSLM이 시계열 데이터에 대한 자동 캡셔닝에서의 큰 가능성을 입증합니다.



### The Proof is in the Almond Cookies (https://arxiv.org/abs/2501.01827)
- **What's New**: 이 논문은 로봇이나 인공지능 요리 보조기가 주방에서 사람 요리사를 지원할 수 있도록 하는 조리 레시피 처리 방법에 대한 사례 연구를 제시합니다. 이러한 AI 보조기는 노인이나 신체적 제한이 있는 사람들의 자율성을 유지하고, 전문 요리사들의 스트레스를 줄이는 데 큰 도움이 될 것입니다. 특히 본 연구는 사람의 사고 방식을 모방한 서사 기반(narrative-based) 조리법 이해의 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문은 조리법을 서사로 취급하며, 이를 통해 구문론(syntax)과 의미론(semantics) 등의 언어 처리를 통합합니다. 조리법은 사건과 행동의 집합인 fabula, 인과 네트워크로 구조화된 plot, 그리고 서술 방식인 narration으로 구성됩니다. 본 연구는 이러한 세 가지 층위를 통해 요리 에이전트가 이해할 수 있는 구조적인 경로를 구축합니다.

- **Performance Highlights**: 이 연구의 주목할 만한 점은 조리법 언어의 복잡성을 다루면서 로봇의 계획 과정 최적화와 AI 시스템의 현재 작업 이해도를 측정하는 방법을 제시한다는 것입니다. 또한, 조리법 주석(annotation)이 언어 독립적으로 변환될 수 있도록 허용하는 과정을 설명합니다. 이러한 접근법은 요리 지식의 통합 및 다중 모드 언어 처리 기술을 사용하여 의미 있는 인간-로봇 상호작용을 지원하는 데 기여할 것입니다.



### End-to-End Long Document Summarization using Gradient Caching (https://arxiv.org/abs/2501.01805)
- **What's New**: 이 논문에서는 CachED(Gradient Caching for Encoder-Decoder models)라는 새로운 방법을 제안합니다. CachED는 기존의 transformer 기반 인코더-디코더 모델을 사용하여 문서의 전체 내용을 잘라내지 않고도 훈련할 수 있게 해줍니다. 이 접근법은 비오버랩 슬라이딩 윈도우를 활용하여 입력 문서를 처리하고, 디코더에서의 융합(fusion)을 수행합니다.

- **Technical Details**: CachED 방법은 인코더의 중간 결과를 메모리에 유지하지 않고, 최종 출력만을 보존합니다. 전방파 전파(backpropagation) 과정에서 그래디언트는 디코더에서 캐시(caching)되고 다시 인코더를 통해 청크(chunk)별로 전달됩니다. 이 방식은 메모리 사용량을 크게 줄여줍니다.

- **Performance Highlights**: CachED BART는 GovReport, SummScreenFD, QMSum, ScriptBase 및 BookSum과 같은 여러 긴 문서 요약 벤치마크에서 성능을 테스트하였고, 기존 방법들보다 우수한 성능을 기록했습니다. 추가 매개변수를 사용하지 않고도 500K 이상의 토큰을 처리할 수 있으며, 특히 1024 토큰의 컨텍스트 크기를 가진 작은 모델로도 뛰어난 결과를 보여주었습니다.



### Reading Between the Lines: A dataset and a study on why some texts are tougher than others (https://arxiv.org/abs/2501.01796)
Comments:
          Published at Writing Aids at the Crossroads of AI, Cognitive Science and NLP WR-AI-CogS, at COLING'2025, Abu Dhabi

- **What's New**: 본 연구는 지적 장애가 있는 특정 대상을 위해 텍스트의 읽기 난이도를 이해하고 이를 단순화하는 방법을 제시합니다. 연구팀은 정서적 특성과 번역 연구를 기반으로 하는 난이도 주석 체계를 도입하였으며, 다양한 공개 서비스에서 수집된 복잡한 텍스트와 단순화된 텍스트의 병렬 코퍼스를 생성했습니다. 또한, 텍스트 단순화 전략을 예측하기 위해 transformer 모델을 미세 조정하였고, 이를 설명 가능한 AI(Explainable AI) 기법으로 해석하는 방법도 탐구하였습니다.

- **Technical Details**: 연구에서 사용된 원본 코퍼스는 스코틀랜드의 복지 서비스 자료, 2024년 영국 총선 정치적 공약 및 장애 평등 스코틀랜드의 뉴스레터 등 76개의 병렬 텍스트로 구성되어 있습니다. 이들 텍스트는 건강 관리, 환경 정책, 법제도 등 다양한 주제를 다루며, 글자 수와 문장 길이 통계가 기존 텍스트에 비해 유의미하게 줄어들면서 읽기 용이성이 증가하는 구조적 조정을 나타냅니다. 또한, 이 연구에서는 복잡한 문장을 단순화하기 위한 다양한 전략을 분석하고, 문장 단위 변환에 중점을 두었습니다.

- **Performance Highlights**: 연구 결과, 단순화된 텍스트는 일반적으로 문장 길이가 줄어들고 문장 수가 증가하여 텍스트 접근성을 개선하는 데 기여한 것으로 나타났습니다. 또한, semantic 및 explanation 카테고리가 두드러지게 나타났으며, 이는 명확성과 독자 접근성 향상에 중점을 두었다는 것을 보여줍니다. 최종적으로, 이 연구는 transformer 기반 모델이 특정 단순화 전략을 예측할 수 있는 가능성을 제시하며, 예측 결과의 해석 가능성을 강화하기 위해 Integrated Gradients(XAI 기법)를 적용하여 모델의 신뢰성을 구축하였습니다.



### Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation (https://arxiv.org/abs/2501.01743)
- **What's New**: 이 논문에서는 모호한 법적 개념을 해석하기 위한 새로운 Retrieval-Augmented Generation (RAG) 프레임워크인 ATRI를 소개합니다. 법적 전문가의 주석 작업을 자동화하여 과거 판례에서 관련 정보를 검색하고 모호한 개념을 해석하며, 법적 개념 포함성(Legal Concept Entailment)이라는 새로운 벤치마크를 제안합니다. 이 프레임워크는 대규모 법률 모델(LLMs)이 모호한 법적 개념을 이해하는 데 도움을 주고, 전문가의 고용 없이 생성된 개념 해석의 품질을 평가할 수 있게 합니다.

- **Technical Details**: ATRI 프레임워크는 다음 세 단계로 구성됩니다: 1) Retrieve: 모호한 개념을 언급한 판례를 검색합니다; 2) Filter & Extract: 판례에서 개념이 상세히 분석되는 경우를 필터링하고 그 결정 이유를 추출합니다; 3) Interpret: 추출된 이유를 바탕으로 LLMs를 사용하여 개념 해석을 생성합니다. 이 방법은 법률 판결의 엄밀한 용어를 기반으로 하여 정확한 문자열 매칭을 통해 판례를 검색합니다.

- **Performance Highlights**: 실험 결과, ATRI는 생성된 해석의 품질이 인간 전문가와 비교할 만하다는 것을 보여주었습니다. 자동 평가와 법률 전문가에 의한 다면적 평가 모두에서 LLM이 생성한 개념 해석이 모호한 개념의 이해를 지원하고 높은 품질을 달성할 수 있음을 입증하였습니다. 이러한 결과는 LLM이 법률 전문가들을 지원하여 모호한 개념 해석을 더욱 간소화할 수 있음을 시사합니다.



### The Essence of Contextual Understanding in Theory of Mind: A Study on Question Answering with Story Characters (https://arxiv.org/abs/2501.01705)
Comments:
          17 pages, under review

- **What's New**: 이 논문은 Theory-of-Mind (ToM)의 중요성을 강조하며, 기존의 평가 방법이 사람이 갖고 있는 복잡한 사회적 관계와 상호작용을 반영하지 못한다는 점을 지적합니다. 새로운 벤치마크인 CharToM-QA를 도입하여, 고전 소설의 등장인물에 기반한 1,035개의 ToM 질문을 포함하고 있습니다. 이를 통해 LLM들이 효율적으로 사람의 정신 상태를 이해하는 능력을 평가합니다. 이 연구는 LLM이 인간보다 어떻게 성능이 떨어지는지를 보여줍니다.

- **Technical Details**: ToM은 인간의 심리적 능력으로, 타인의 마음 상태를 이해하고 해석하는 과정입니다. 연구진은 CharToM-QA 벤치마크를 개발하여 LLM의 ToM 이해 능력을 평가합니다. 이 벤치마크는 소설의 등장인물과 관련된 ToM 질문을 제시하며, 고유한 요구에 대한 이해가 더 깊어야 합니다. 평가 과정은 생성적 질문 응답(Generative QA)과 다중 선택 질문 응답(Multiple Choice QA)으로 나뉘며, GPT-4o를 평가자로 활용하여 대답의 질을 체크합니다.

- **Performance Highlights**: 실험 결과 GPT-4o는 다양한 ToM 차원에서 다른 LLM보다 뛰어난 성능을 보였습니다. 인간 참가자들은 소설을 읽었을 때 LLM보다 월등히 높은 성과를 보였으며, 읽은 길이가 길어질수록 정확도가 향상되었습니다. 반면 LLM의 성능은 다양한 줄거리 길이에 관계없이 큰 변화가 없었습니다. 이는 LLM이 복잡한 시나리오와 미세한 역사적 맥락 정보를 효과적으로 포착하지 못하고 있음을 나타냅니다.



### Adaptive Few-shot Prompting for Machine Translation with Pre-trained Language Models (https://arxiv.org/abs/2501.01679)
Comments:
          published to AAAI2025

- **What's New**: 최근에 제안된 Adaptive Few-Shot Prompting (AFSP) 프레임워크는 다양한 출처 입력 문장에 대해 적절한 번역 시연을 자동으로 선택하여 LLM의 번역 능력을 더 끌어내어 기계 번역(Machine Translation) 품질을 향상시킵니다. 이 프레임워크는 LLM의 임베딩을 기반으로 번역 시연 검색 모듈을 구축하여 상응하는 평행 번역 말뭉치에서 의미적으로 유사한 번역 시연을 효율적으로 검색합니다. 또한, AFSP는 번역 결과의 의미 일관성을 보장하기 위해 LLM이 여러 출력 후보를 생성하고 이를 재평가하는 과정을 포함합니다.

- **Technical Details**: 본 논문에서는 LLM의 임베딩 계층을 기반으로 한 하이브리드 시연 검색 모듈을 통해 더 의미적으로 관련된 번역 시연을 검색하는 방법을 구축합니다. 제안된 AFSP는 세 가지 유형의 임베딩—밀집 임베딩(Dense Embedding), 희소 임베딩(Sparse Embedding), 다중 벡터 임베딩(Multi-Vector Embedding)—을 활용하여 기계 번역 품질을 향상시키기 위한 입력 표현을 개선합니다. 최종적으로, LLM의 확률적 샘플링으로 인해 발생할 수 있는 의미적 편향을 완화하기 위해 다수의 출력 후보를 생성하고, 작은 언어 모델(SLM)을 기반으로 한 재평가 모델로 최적의 번역 결과를 선택합니다.

- **Performance Highlights**: 논문에서 제안한 AFSP 프레임워크는 5,528개의 병행 중국어-영어 문장으로 구성된 고품질 외교 중국어-영어 평행 데이터셋과 유엔 평행 말뭉치에서 연구되었으며, 그 성능이 정량적 및 정성적으로 입증되었습니다. AFSP를 통해 LLM의 의미적 일관성 및 번역 품질이 현저하게 개선되었음을 보여줍니다. 결과적으로, 이 연구는 LLM의 기계 번역 성능을 향상시키기 위한 새로운 접근 방식을 제시하며, 우리의 데이터셋이 기계 번역 연구의 경계를 넓히는 데 기여할 것으로 기대됩니다.



### CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis (https://arxiv.org/abs/2501.01668)
- **What's New**: 이 논문에서는 CoT-based Synthesizer라는 새로운 추론 스케일링 전략을 제안합니다. 이 방법은 여러 후보 응답에서 보완 정보를 분석하여 우수한 답변을 합성합니다. 이전의 Self-consistency나 Best-of-N 방법이 후보 응답의 질에 의존하는 반면, 제안된 방법은 후보가 모두 틀렸을 때도 유용하게 작동합니다.

- **Technical Details**: CoT-based Synthesizer는 Chain-of-Thought (CoT) 추론을 활용하여 후보 응답을 체계적으로 분석하고 새로운 답변을 합성합니다. 이와 함께 우리는 다양한 훈련 데이터를 생성하는 자동화된 데이터 생성 파이프라인을 설계했습니다. 이 파이프라인은 후보 응답 합성을 위한 훈련 데이터를 생성하고, 이렇게 생성된 데이터로 작은 LLM을 훈련시킵니다.

- **Performance Highlights**: 실험 결과, 이 방법은 Llama3-8B 모델에서 11.8%의 성능 향상을, GPT-4o 모델에서는 10.3%의 향상을 나타냈습니다. 다양한 NLP 태스크에서 제안된 방법은 기존의 Self-consistency 및 Best-of-N 방법보다 유의미한 개선을 보였으며, 훈련된 작은 모델이 대형 모델의 후보 응답 성능을 효과적으로 향상시킬 수 있음을 보여줍니다.



### MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments (https://arxiv.org/abs/2501.01652)
- **What's New**: 이 논문은 LLMs(Large Language Models)의 고급 인격 표현 능력을 평가하기 위한 포괄적인 프레임워크인 MIRAGE(Multiverse Interactive Role-play Ability General Evaluation)를 소개합니다. MIRAGE는 살인 미스터리 게임을 기초로 하여 LLMs의 사회적 능력을 평가하는 혁신적인 방법론으로, 신뢰와 의심의 동역학을 측정하는 TII(Trust Inclination Index), 정보 탐색 능력을 평가하는 CIC(Clue Investigation Capability) 등을 포함합니다. 이를 통해 LLMs의 역할 수행 능력과 규칙 준수 능력을 심층적으로 분석할 수 있습니다.

- **Technical Details**: MIRAGE는 캐릭터의 이야기, 스크립트, 관계, 역할 수행, 목표 및 기타 능력으로 구성된 여섯 개 주요 부분으로 나뉘어 있습니다. 각 게임은 오픈 대화, 환경 상호작용, 살인 투표의 세 가지 주요 단계로 진행됩니다. 추가적으로, LLMs의 성과를 효율적으로 평가하기 위해 여러 보조 모듈이 표준화되었으며, TII, CIC, ICI(Interactivity Capability Index), SCI(Script Compliance Index)와 같은 네 가지 객관적인 평가 지표가 사용됩니다.

- **Performance Highlights**: 본 연구에서는 GPT-3.5, GPT-4 및 다양한 오픈소스 모델을 활용하여 LLMs의 성능 평가를 수행하였습니다. 실험 결과, 인기 모델인 GPT-4조차도 MIRAGE의 복잡성을 탐색하는 데 상당한 어려움을 겪는 것으로 나타났습니다. MIRAGE의 데이터셋과 시뮬레이션 코드는 GitHub를 통해 공개되어 있으며, 이를 통해 다른 연구자들이 이 평가 메커니즘을 활용하고 개선할 수 있는 기회를 제공합니다.



### Multimodal Contrastive Representation Learning in Augmented Biomedical Knowledge Graphs (https://arxiv.org/abs/2501.01644)
- **What's New**: 이번 연구는 다중 모드 접근 방식을 도입하여 생물 의학 지식 그래프(BKG)에서 링크 예측(link prediction)의 성능을 향상시키기 위해 특수 언어 모델(Language Models)로부터의 임베딩(embedding)과 그래프 대조 학습(Graph Contrastive Learning)을 통합한 새로운 방법론을 제시합니다. 또한, 생물학적 서열과 텍스트 정보를 포함하는 PrimeKG++라는 향상된 지식 그래프를 제안하여 노드 간 관계를 풍부하게 표현하고 있습니다. 이 접근 방식은 보편성을 극대화하여 보지 못한 노드에 대해서도 정확한 링크 예측을 가능하게 합니다.

- **Technical Details**: 본 연구에서는 언어 모델(LM)에서 유도된 임베딩을 통합하고, GCL을 활용하여 각 노드의 상호 정보를 최적화하여 노드 간 관계를 처리합니다. 이와 함께, 지식 그래프 임베딩(Knowledge Graph Embedding) 모델을 사용하여 다양한 생물학적 개체 간의 상호 정보를 캡처합니다. 연구의 핵심은 다양한 입력 데이터의 세미틱과 관계 정보를 통합하여, 생물 의학 지식 그래프(BKG)의 링크 예측 성능을 강화하는 것입니다.

- **Performance Highlights**: PrimeKG++와 DrugBank 약물-타겟 상호작용 데이터셋에서의 실험 결과, 제안된 방법이 다양한 생물 의학 데이터셋에서 강력하고 일관되며 정확한 링크 예측을 보여주었습니다. 기존 모델과 비교했을 때, 우리의 사전 훈련된 노드 표현 모델이 성능 향상에 크게 기여했으며, link prediction뿐만 아니라 생물 의학 연구 커뮤니티에 가치 있는 자원을 제공하고 있습니다.



### A non-ergodic framework for understanding emergent capabilities in Large Language Models (https://arxiv.org/abs/2501.01638)
- **What's New**: 이번 연구에서는 대형 언어 모델이 비에르고딕(non-ergodic) 시스템이라는 점을 증명하고, 능력 출현을 설명하기 위한 수학적 프레임워크를 제시합니다. 특히, Stuart Kauffman의 인접 가능성 이론(TAP)을 기반으로 한 접근 방식을 통해 능력의 출현 메커니즘을 제시합니다. 이는 기존 언어 모델 연구에서 진일보한 이해를 제공합니다.

- **Technical Details**: 연구에서는 TAP 방정식이 아키텍처, 훈련, 그리고 맥락의 제약 사항들과 어떻게 상호작용하여 모델의 능력을 형성하는지를 설명합니다. 특히, 의미 공간의 단계 전이(phase transitions)를 통해 자원 제약이 모델의 능력에 미치는 영향을 강조합니다. 실험 결과, 세 가지 다른 언어 모델에서 제약 상호작용과 경로 의존 탐색(path-dependent exploration)에 의해 능력이 불연속적으로 나타나는 것을 보여줍니다.

- **Performance Highlights**: 이 연구는 언어 모델의 출현(emergence)을 이해하기 위한 이론적 기초를 제공합니다. 또한, 향후 능력 출현을 이끌어낼 수 있는 아키텍처 개발에 대한 가이드를 제시하여, 향후 연구 및 응용에 기여할 가능성을 제시합니다.



### ICPC: In-context Prompt Compression with Faster Inferenc (https://arxiv.org/abs/2501.01625)
- **What's New**: ICPC(인-컨텍스트 프롬프트 압축)은 긴 프롬프트를 효과적으로 압축하여 LLM의 계산 비용과 메모리 오버헤드를 줄이는 새로운 방법론입니다. 기존의 프롬프트 압축 방식과 달리, ICPC는 단어가 프롬프트에 등장할 확률을 계산하고 정보 함수(information function)를 통해 각 단어가 담고 있는 정보를 평가하여 정보 손실을 최소화합니다. 이러한 과정은 압축 속도를 높이면서도 의미의 본질을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: ICPC의 핵심 메커니즘은 변환기 인코더(transformer encoder)를 사용하여 프롬프트의 단어를 평가하고, 이를 기반으로 필요 없는 토큰을 제거하여 문장을 압축합니다. 중복된 정보가 있는 토큰을 제거하기 위해, 문장 내에서 손실을 계산하여 중요하지 않은 단어를 선택적으로 삭제합니다. 이러한 방식은 기존의 LLM에 비해 더 적은 자원으로도 높은 성능을 유지할 수 있도록 합니다.

- **Performance Highlights**: 제안된 ICPC 방법은 다양한 텍스트 카테고리에 걸쳐 긴 문장을 효과적으로 압축하여 여러 NLP 작업에서 더 나은 성능과 속도를 달성함을 실험적으로 입증했습니다. AWS EC2 환경에서 실시한 실험들은 다른 최첨단 접근 방식과 성능을 비교 관찰하였으며, Wikipedia와 같은 긴 컨텍스트 데이터셋에서 뛰어난 결과를 나타냈습니다. 이로써 ICPC는 LLM의 장기적인 컨텍스트 처리 능력을 강화하는 데 기여합니다.



### PSYCHE: A Multi-faceted Patient Simulation Framework for Evaluation of Psychiatric Assessment Conversational Agents (https://arxiv.org/abs/2501.01594)
Comments:
          The first two authors contributed equally

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 인간과 유사한 반응을 생성할 수 있는 대화형 에이전트의 개발을 가속화했습니다. 이 연구에서는 정신과 평가 대화형 에이전트(PACAs)를 위한 새로운 프레임워크인 PSYCHE를 제안하여 임상 평가에서 정신과 의사의 역할을 시뮬레이션하는 방법을 다룹니다. PSYCHE는 PACAs의 임상 적합성을 1) 임상적 관련성, 2) 윤리적 안전성, 3) 비용 효율성 및 4) 정량적 평가를 통해 평가할 수 있는 구조를 제공합니다.

- **Technical Details**: PSYCHE 프레임워크는 다면적 정신과 구조(MFC)를 기반으로 환자 발화를 시뮬레이션하고 PACAs를 평가하는 방식으로 설계되었습니다. 이 프레임워크는 네 가지 단계로 구성되며, 사용자 입력, MFC 생성, 발화 시뮬레이션 및 평가 세션으로 이어집니다. 각 단계는 PACA의 성능을 평가하기 위한 체계적 프로세스를 통해 PSYCHE-SP를 생성하여 환자와의 상호작용을 가능하게 하며, 이 결과는 PACA의 성과 지표인 PSYCHE SCORE로 나타납니다.

- **Performance Highlights**: 연구 결과, 10명의 정신과 의사가 평가한 PSYCHE-SP는 다양한 정신 장애를 시뮬레이션하는 데 있어서 높은 일관성을 보였습니다. 총 7가지 장애에 대해 평균 93%의 적합성을 달성하였으며, 주요 우울 장애(MDD)와 사회 불안 장애(SAD)에서 각각 97%의 가장 높은 적합성을 기록했습니다. PSYCHE는 임상 환경에서 PACAs의 성능 평가를 효과적으로 진행할 수 있는 가능성을 보여주며, 향후 정신 건강 분야의 자동화와 효율성을 높이는 데 기여할 것으로 기대됩니다.



### (WhyPHI) Fine-Tuning PHI-3 for Multiple-Choice Question Answering: Methodology, Results, and Challenges (https://arxiv.org/abs/2501.01588)
- **What's New**: 이번 연구는 Microsoft의 PHI-3 모델을 활용하여 다중 선택 질문(MCQ)에 대한 답변 능력을 향상시키는 방법을 탐구합니다. 본 논문에서는 TruthfulQA 데이터셋을 사용하여 모델을 세밀하게 조정하고, 최적화된 프롬프트를 설계하여 모델 성능을 개선했습니다. 결과적으로, PHI-3 모델의 MCQ 처리 능력이 대폭 향상되었으며, 이는 교육적 응용 프로그램에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: 연구 방법론은 데이터 전처리, 프롬프트 설계, 모델 세부 조정(fine-tuning), 그리고 평가를 포함합니다. 이를 위해 TruthfulQA 데이터셋을 사용하여 1,000개의 MCQ를 기준으로 입력 형식을 표준화했습니다. PHI-3 모델의 정확한 답변 생성을 돕기 위해 Alpaca 스타일의 프롬프트와 기본 텍스트 보완 프롬프트를 결합하여 성능을 개선했습니다.

- **Performance Highlights**: 모델 세부 조정 후 PHI-3는 perplexity가 4.68에서 2.27로 감소하고, 정확도는 62%에서 90.8%로 상승했습니다. 이러한 결과는 세밀하게 조정된 소형 언어 모델이 교육적 과제 처리에서 매우 효과적일 수 있음을 보여주며, 특히 자원 제약이 있는 환경에서의 성공적인 적용 가능성을 나타냅니다.



### Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models (https://arxiv.org/abs/2501.01830)
- **What's New**: 이 논문에서는 Auto-RT라는 새로운 강화 학습 기반의 프레임워크를 제안합니다. 이는 공격 쿼리를 통해 보안 취약점을 효율적으로 발견하기 위한 복잡한 공격 전략을 탐색하고 최적화합니다. Auto-RT는 탐색 복잡성을 줄이고 전략 최적화를 개선하는 두 가지 핵심 메커니즘을 도입합니다: Early-terminated Exploration과 Progressive Reward Tracking 알고리즘입니다.

- **Technical Details**: 자동적인 red-teaming의 목표는 공격 모델 AM을 사용하여 공격 프롬프트를 생성하고, 이를 통해 타겟 모델 TM의 반응을 평가하는 것입니다. 공격 모델의 최적화 과정에서 여러 제약 조건을 추가하여 최적화를 실행하며, 이 논문에서는 전략적 red-teaming이라는 개념을 통해 공격 전략 생성 모델 AMg와 특정 공격 프롬프트를 생성하는 공격 재구성 모델 AMr로 나누어 설명합니다.

- **Performance Highlights**: Auto-RT를 통해 다양한 LLM에 대한 실험을 수행한 결과, 탐색 효율성을 크게 개선하고 공격 전략을 자동으로 최적화하여 더 넓은 범위의 취약점을 탐지할 수 있었습니다. 이 방법은 기존 방법에 비해 16.63% 높은 성공률을 달성하며, 빠른 탐지가 가능합니다.



### SDPO: Segment-Level Direct Preference Optimization for Social Agents (https://arxiv.org/abs/2501.01821)
- **What's New**: 이번 연구에서는 Segment-Level Direct Preference Optimization (SDPO)라는 새로운 방법을 제안하여, LLM 기반 사회 에이전트의 다중 대화 세션에서의 의사결정 및 행동을 개선하고자 했습니다. 기존의 turn-level DPO와 session-level DPO 방법론의 한계를 극복하고자, 특정 키 세그먼트를 집중하여 최적화하는 방식을 채택했습니다. SDPO는 교육 과정에서 발생할 수 있는 노이즈를 최소화하면서 대화의 질을 높일 수 있도록 구성되었으며, 이를 통해 LLM의 사회적 지능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: SDPO 방법론은 불량 세션에서 발생하는 오류가 있는 턴을 식별하고, 그 이전의 상호작용 이력을 바탕으로 긍정적인 세션을 생성합니다. 이후 SDPO는 긍정적인 세션에서의 주요 세그먼트를 파악하여, 같은 길이의 부정적인 세션의 세그먼트와 데이터 쌍을 형성합니다. 다차원적인 대화의 정렬을 위해 SDPO는 잘못된 세그먼트와 해당 긍정적 세그먼트에 대해서만 손실(loss) 값을 산출함으로써, 비어떤 턴으로 인한 교육 노이즈를 제거합니다.

- **Performance Highlights**: SDPO 방법은 SOTOPIA 벤치마크에서 GPT-4o 및 기타 경쟁 모델들과의 상호작용을 통해 검증되었으며, DPO, ETO, DMPO와 같은 기존 기법들을 일관되게 초과 성능을 보여주었습니다. 이 연구의 결과는 세그먼트 레벨의 정렬이 LLM 기반 에이전트의 성능을 획기적으로 향상시킬 수 있음을 입증하고, 기존 기법보다 유연하고 일반적으로 사용 가능한 솔루션임을 시사합니다. 더 나아가 SDPO는 다양한 도메인에 걸쳐 에이전트의 역량을 향상시킬 가능성을 지니고 있습니다.



### How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models (https://arxiv.org/abs/2501.01741)
- **What's New**: 본 논문에서는 자동화된 LLM(대형 언어 모델) 독성 테스트 프레임워크인 EvoTox를 소개합니다. 이 프레임워크는 LLM의 독성 응답 생성 가능성을 정량적으로 평가하여, 기존의 정렬(alignment) 조치에도 불구하고 여전히 존재하는 독성 반응을 탐지할 수 있도록 설계되었습니다. EvoTox는 시스템의 응답을 더욱 독성적으로 유도하기 위해 Prompt Generator라는 두 번째 LLM과의 상호작용을 활용하는 반복적인 진화 전략을 채택합니다.

- **Technical Details**: EvoTox는 기초적인 프롬프트(seed prompt)로 시작해 매 반복마다 새로운 프롬프트(뮤턴트)를 생성하는 방식으로 작동하며, 이 과정에서 기존의 독성 분류기를 기반으로 한 자동화된 오라클을 사용합니다. 이러한 접근 방식은 LLM의 내부 정보에 대한 요구 없이 블랙박스 방식으로 운영됩니다. EvoTox는 다양한 프롬프트 진화 전략을 제공하며, 이러한 전략은 효과적인 변이 방향을 파악하는 데 도움이 됩니다.

- **Performance Highlights**: EvoTox의 평가 결과는 기존의 기준 방법과 비교하여 독성 수준 탐지에서 유의미하게 높은 효과성을 보였습니다. 실험에서 EvoTox는 랜덤 탐색(random search) 및 Jailbreak 기법 대비 각각 1.0 및 0.99의 효과 크기를 보였으며, 실행 시간 측면에서도 평균 22%에서 35%의 제한된 비용 오버헤드를 기록하였습니다. 또한, 인적 평가를 통해 EvoTox로 생성된 프롬프트의 유창성과 독성 수준이 경쟁 방식보다 상당히 우수하다는 결과를 도출했습니다.



### AgentRefine: Enhancing Agent Generalization through Refinement Tuning (https://arxiv.org/abs/2501.01702)
- **What's New**: 이 연구는 Large Language Model (LLM) 기반 에이전트의 일반화 능력을 향상시키기 위한 새로운 프레임워크인 AgentRefine를 제안합니다. 기존 연구는 특정 에이전트 환경에서의 과적합(overfitting) 문제로 인해 일반화 성능이 떨어지는 것으로 분석되었으며, 이에 대한 해결책으로 자기 수정 능력을 강조하고 있습니다.

- **Technical Details**: AgentRefine는 에이전트가 관찰(observation)을 통해 자신의 실수를 수정하도록 학습할 수 있도록 돕는 구조입니다. 이를 위해 다양한 환경(environments)과 작업(tasks)을 포함하는 에이전트 합성(agent synthesis) 프레임워크를 도입하여, 환경 피드백에 따라 에이전트가 동작을 개선할 수 있도록 유도합니다.

- **Performance Highlights**: AgentRefine는 다양한 에이전트 작업에서 높은 일반화 능력을 보여주며, 기존 최첨단(agent-tuning) 기술들을 능가합니다. 또한 잡음(perturbation)에 대한 견고성(robustness)이 높고 추론(inference) 과정에서도 다양화된 사고(thought)를 생성할 수 있는 특징을 가지고 있습니다.



### Crossing Language Borders: A Pipeline for Indonesian Manhwa Translation (https://arxiv.org/abs/2501.01629)
- **What's New**: 본 연구에서는 인도네시아어에서 영어로의 만화(Manhwa) 번역을 자동화하기 위한 효율적인 솔루션을 제안합니다. 컴퓨터 비전, 텍스트 인식, 자연어 처리 기법을 결합하여 전통적인 번역 방식의 비효율성을 해소하고자 합니다. 이 시스템은 의사 대화의 구간을 탐지하고, 문자를 인식하며, 이를 번역하여 만화 패널에 다시 오버레이하는 단계를 포함합니다.

- **Technical Details**: 연구에 사용된 주요 기법은 YOLOv5xu를 활용한 스피치 버블(spoech bubble) 탐지, Tesseract를 통한 광학 문자 인식(Optical Character Recognition), 그리고 MarianMT를 통한 기계 번역입니다. YOLOv5xu는 고정밀 객체 감지를 위해 미세 조정되었고, Tesseract는 인도네시아어 모델을 사용하여 문자를 효율적으로 인식합니다. 마지막으로 번역된 텍스트는 OpenCV와 Pillow 라이브러리를 통해 원본 이미지의 대화 상자 형태에 맞게 재배치됩니다.

- **Performance Highlights**: 모델의 성능은 YOLOv5xu가 90.7%의 F1 점수를 나타내어 스피치 버블을 효과적으로 검출했다는 것을 보여줍니다. OCR의 경우, Tesseract를 활용해 CER 3.1%, WER 8.6%의 성능을 기록하며 기존 방법보다 우수한 결과를 달성하였습니다. 또한, MarianMT 모델은 BLEU 점수와 Meteor 점수를 통해 번역의 의미적 보존이 잘 이루어졌음을 증명했으며, 이 모든 단계를 통합한 자동화 파이프라인은 만화 번역 작업의 효율성을 높이는데 기여하고 있습니다.



### Predicting the Performance of Black-box LLMs through Self-Queries (https://arxiv.org/abs/2501.01558)
Comments:
          28 pages

- **What's New**: 이 논문은 블랙박스(black-box) 접근 방식에서 대형 언어 모델(LLM)의 특징을 추출하는 새로운 방법을 제안합니다. 전통적으로 LLM은 화이트박스(white-box) 접근 방식에서 모델의 상태나 활성화(hidden states)에 대한 접근이 요구되었으나, 이 연구는 후속(prompt) 질문과 응답 확률을 통해 저차원(low-dimensional) 특징을 추출합니다. 본 연구의 접근 방식은 모델의 성능 예측을 개선하며, 흥미롭게도 화이트박스 접근보다 더 뛰어난 성능을 발휘할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 LLM의 출력을 기반으로 후속 질문을 활용하여 성능을 예측하는 방법을 개발했습니다. 연구자들은 LLM이 생성한 응답에 대해 스스로 설명할 수 있는 능력을 활용하여, 응답의 확률 분포가 정확성 및 모델의 차이에 따라 크게 달라진다고 가정합니다. 이러한 저차원 특징을 학습하여 LLM의 성능을 예측할 수 있으며, 예측 과정에서 여러 다양한 LLM을 사용한 결과가 나타났습니다.

- **Performance Highlights**: 연구 결과, 제안된 후속 질문 접근 방식은 특정 클래스의 예측이나 텍스트 생성이 정확한지를 예측하는 데 매우 효과적일 뿐만 아니라 강력한 일반화 보장을 제공했습니다. 모델의 다양성을 조사한 결과, 후속 질문뿐만 아니라 다양한 자연어 시퀀스도 예측 성능을 높일 수 있다는 흥미로운 발견이 있었습니다. 마지막으로, 해당 방법은 LLM의 아키텍처와 크기 판별에서도 신뢰할 수 있는 결과를 도출할 수 있음을 보였습니다.



### Many of Your DPOs are Secretly One: Attempting Unification Through Mutual Information (https://arxiv.org/abs/2501.01544)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 포스트 정렬(post-alignment) 과정을 단순화하고 명확하게 이해할 수 있도록 도와주는 새로운 통합 프레임워크를 제안합니다. 직접 선호 최적화(Direct Preference Optimisation, DPO) 알고리즘들의 다양성으로 인해 연구자들이 각 기법의 관계를 파악하기 어려운 상황을 개선하고자 합니다. 제안된 프레임워크는 상호 정보(mutual information)에 영감을 얻어 만들어졌으며, 유연한 사전(prior)을 갖는 새로운 손실 함수(loss function)를 포함하고 있습니다.

- **Technical Details**: 본 프레임워크를 통해 SimPO, TDPO, SparsePO 등 여러 기존 알고리즘들이 파생될 수 있음을 보여줍니다. 사전의 세심한 정의는 다양한 DPO 기법들 사이의 명확한 관계를 성립시키는 데 중점을 두고 있습니다. 이로 인해 DPO 알고리즘의 복잡성이 감소하고, 연구자들이 더 많은 통찰을 얻을 수 있는 기회가 마련됩니다.

- **Performance Highlights**: 우리는 제안된 프레임워크가 LLM의 정렬 기술을 더 견고하고 해석 가능하게 만드는 기초가 되기를 바랍니다. 이 논문은 연구 커뮤니티가 LLM 정렬 분야에서 더 발전된 기술을 개발하도록 지원하는 데 중점을 두고 있습니다. 새롭게 제안된 통합 접근 방식은 DPO 알고리즘의 이해를 증진시켜, 향후 연구 활동에 긍정적인 영향을 미칠 것으로 기대됩니다.



### A Metasemantic-Metapragmatic Framework for Taxonomizing Multimodal Communicative Alignmen (https://arxiv.org/abs/2501.01535)
Comments:
          34 pages, 1 figure, 3 tables. Draft presented at 2023 ZJU Logic and AI Summit EAI Workshop

- **What's New**: 이 논문은 현대의 실용주의 철학과 인지, 의미, 커뮤니케이션에 대한 언어 이론을 바탕으로 인간과 유사한 다중 모달 커뮤니케이션 정렬을 정립하는 동적 메타 의미론-메타 실용주의 세분화를 제시합니다. Charles Sanders Peirce에 의해 처음 제안된 세 가지 기본적인 커뮤니케이션 능력인 아이코닉(iconic), 인덱시컬(indexical), 룰 라이크(rule-like)에 대한 현대적 발전을 토대로 진행됩니다.

- **Technical Details**: 세 가지 커뮤니케이션 능력의 발전을 바탕으로, 인덱시컬 맥락화(indexical contextualization) 개념이 도입되며, 다중 모달 커뮤니케이션의 의미적(semantic) 및 실용적(pragmatic) 모드를 유지하고 탐색하는 데 필요한 메타 실용주의 능력인 '맥락화 방향성(contextualization directionality)' 원칙이 제안됩니다. 이 논문은 현재의 인지-사회적 컴퓨팅(cognitive-social computational) 방법론이 의미론/메타 의미론 영역에 집중하고 있음을 비판하며, 메타 실용주의 인덱시컬리티의 중요성을 강조합니다.

- **Performance Highlights**: 논문의 방법론은 의도(intentionality), 정체성(identity), 정서(affect), 윤리(ethics)와 같은 광범위한 주제에 대한 넓은 함의를 가지고 있으며, 이는 인간-기계 간의 내적 모달과 교차 모달 정렬에서 중요한 역할을 할 수 있습니다. 이 연구는 커뮤니케이션의 의미-실용 스펙트럼을 탐험하는 데 있어 메타 실용주의의 중심 역할을 부각시키고 있습니다.



### Improving Robustness Estimates in Natural Language Explainable AI though Synonymity Weighted Similarity Measures (https://arxiv.org/abs/2501.01516)
Comments:
          10 pages, 2 figures, 4 tables

- **What's New**: 이 논문에서는 Explainable AI (XAI) 기술에 있어 적대적 예제(adversarial examples)가 중요한 역할을 한다고 강조합니다. 기존의 유사도 측정 방법들이 적대적 XAI에서 효과적인 비교를 제공하지 못한다는 점을 지적하고, 동의어 가중치(synonymity weighting)를 도입하여 이러한 문제를 해결하고자 합니다. 이 접근 방식은 XAI 방법의 내구성 및 안정성, 즉 안정성을 평가하는 새로운 기준을 제시합니다.

- **Technical Details**: 문헌에서는 XAI 모델의 설명이 복잡한 블랙박스 모델의 출력을 이해하기 위한 방법으로 사용되지만, 기존의 유사도 측정 지표들이 두 가지 주된 결함(sensitivity와 indifference) 때문에 신뢰성이 떨어진다고 설명합니다. 이 연구는 동의어 가중치를 통해 perturbed 단어와 원본 단어 간의 의미적 유사성을 고려하여 단순한 비교 방식을 개선합니다. 이러한 접근법은 텍스트 기반 입력에서의 XAI의 적대적 공격 과정을 평가하기 위한 새로운 기초를 제공합니다.

- **Performance Highlights**: 동의어 가중치를 적용한 유사도 측정은 기존의 일반적인 유사도 측정보다 훨씬 더 정확한 결과를 도출합니다. 실험 결과, 전통적인 유사도 측정 방법으로는 XAI의 불안정성을 잘못 평가할 수 있음을 보여주며, 새로운 방법이 XAI 시스템의 신뢰성을 높이는 데 기여할 수 있다는 결론을 도출합니다. 이를 통해 XAI 방법이 적대적 예제에 대해 보다 강건함을 가질 수 있도록 하는 실질적인 기여를 하고 있습니다.



### Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search (https://arxiv.org/abs/2501.01478)
Comments:
          5 pages, 1 figure, 2 tables accepted by aaai 2025 NeurMAD workshop

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 능력을 개선하기 위해 Monte Carlo Tree Search (MCTS)를 이용하여 자체적으로 과정 감독(process supervision) 데이터를 생성하는 방법을 제안합니다. 발생한 reasoning step에 각각 '상대적 정합성(relative correctness)' 점수를 부여하고, 이를 통해 LLM을 훈련시키는 과정을 반복적으로 수행했습니다. 이 방법은 결과적으로 두 개의 수학 reasoning 데이터셋에서 LLM의 성능을 상당히 개선하는 것으로 나타났으며, 이는 향상된 추론 능력의 전이성(transferability)도 보여줍니다.

- **Technical Details**: 제안된 방법론에서는 LLM이 문제를 해결하기 위해 생성한 reasoning 경로에 대해 MCTS를 활용하여 각 단계의 '상대적 정합성' 점수를 부여합니다. 이 점수는 이진 선호(binary preferences)보다 각 단계의 품질을 더욱 정확하게 반영하며, 이를 통해 LLM의 가중치 음 로그 우도 손실 함수(weighted negative log-likelihood loss function)에 통합하여 훈련을 진행합니다. 이러한 generate-then-train 방식은 수학적 reasoning 문제에 대한 LLM의 성과를 향상시키는 반복적인 과정으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LLM의 수학적 reasoning 성능을 대폭 향상시켰습니다. 두 개의 데이터셋 각각에서 훈련된 모델이 다른 데이터셋에서도 성능 개선을 보여주었으며, 이는 향상된 추론 능력의 전이성을 시사합니다. 또한, 인간 주석이 필요하지 않은 방식으로 과정 감독을 통해 훈련할 수 있는 가능성을 제시하였습니다.



### Reinforcing Thinking through Reasoning-Enhanced Reward Models (https://arxiv.org/abs/2501.01457)
- **What's New**: 본 연구는 LLM의 복잡한 다단계 추론을 개선하기 위해 Distillation-Reinforcement-Reasoning (DRR)이라는 새로운 3단계 프레임워크를 제안합니다. 이 프레임워크는 LLM의 내부 행동을 외부 피드백으로 활용하여, LLM이 언제 멈춰야 할지 결정할 수 있도록 돕습니다. DRR은 중간 단계의 수작업 레이블이 필요 없으며, 경량 디자인으로 다양한 LLM 중심의 작업에 쉽게 적용할 수 있습니다.

- **Technical Details**: DRR 프레임워크는 LLM의 추론 능력을 반영하는 행동 데이터를 생성을 통해 시작합니다. 이후, 행동 데이터를 기반으로 경량 식별 보상 모델(Discriminative Model, DM)을 훈련하여 추론 시 판단을 돕습니다. 이 과정은 언어적 보상(verbal reward)을 통해 LLM의 평가를 제공하며 모델 파라미터를 변경하지 않고도 동적 피드백 메커니즘을 구축합니다.

- **Performance Highlights**: 실험 결과, DRR 프레임워크는 자가 비평(self-critique) 방식을 적용한 방법들보다 우수한 성능을 보였으며, 추가적인 복잡한 데이터 주석에 의존하지 않는 것으로 나타났습니다. 연구팀은 모든 코드베이스와 체크포인트, 생성된 데이터를 공개할 예정이며, 이는 개방형 및 닫힌 소스 LLM에 유익할 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### Cold-Start Recommendation towards the Era of Large Language Models (LLMs): A Comprehensive Survey and Roadmap (https://arxiv.org/abs/2501.01945)
- **What's New**: 이번 논문은 차가운 시작 문제(cold-start problem)를 해결하기 위해 대규모 언어 모델(large language models, LLMs)을 활용하는 방법에 대한 포괄적인 검토를 제공합니다. 최근 사용자와 아이템의 폭발적인 증가로 인해 추천 시스템(recommender systems)에서 차가운 시작 추천(cold-start recommendation, CSR)의 중요성이 더욱 부각되고 있습니다. 연구 커뮤니티는 아직 CSR 분야에 대한 포괄적인 리뷰가 부족하기 때문에 이 논문은 새로운 통찰력을 제공합니다.

- **Technical Details**: CSR의 기존 정보 활용 경로를 콘텐츠 특징(content features), 그래프 관계(graph relations), 도메인 정보(domain information)부터 LLM이 보유한 세계지식까지 탐색했습니다. 이 논문에서는 연구 및 산업 커뮤니티에 CSR에 관한 새로운 통찰력을 제공하기 위해 기존 문헌을 기반으로 차가운 시작 추천의 로드맵을 제시합니다. 또한, 관련 자료들은 지속적으로 커뮤니티에 제공될 예정입니다.

- **Performance Highlights**: 대규모 언어 모델의 성공적인 적용은 차가운 시작 문제 해결에 새로운 가능성을 제시합니다. 본 연구의 목적은 차가운 시작 추천의 발전 경로에 대한 탐색과 현대 LLM의 혜택을 결합하여 추천 품질을 개선하고 사용자 경험을 향상시키는 것입니다. 이러한 접근 방식은 CSR 연구와 산업 모두에 큰 기여를 할 것입니다.



### Item Association Factorization Mixed Markov Chains for Sequential Recommendation (https://arxiv.org/abs/2501.01429)
- **What's New**: 이 연구는 Sequential Recommendation(순차 추천) 분야에서 새로운 접근법을 제시합니다. 기존 연구들은 Markov Chain(마코프 체인)을 기반으로 한 모델이 많았으나, 사용자의 행동 기록에만 집중했던 반면, 본 연구는 아이템 간의 상관관계에 주목했습니다. 새롭게 도입된 알고리즘은 Item Association Factorization Mixed Markov Chains(아이템 연관 요인 분해 혼합 마코프 체인)으로, 아이템 간의 연관 정보를 통합하여 추천 정확도를 향상시킵니다.

- **Technical Details**: 이 알고리즘은 아이템 연관 그래프(item association graph)를 사용하여, 사용자의 행동 시퀀스 정보와 아이템 간의 연관성을 결합합니다. 이를 통해 아이템 간의 상관관계를 고려한 추천이 가능해지며, 추천의 정확도를 높일 수 있습니다. 연구는 또한 prior balancing parameters(사전 균형 파라미터) 조정에 대한 중요성을 강조하며, 다양한 데이터셋에 걸쳐 아이템 연관 정보를 통합하는 것의 중요성을 강조합니다.

- **Performance Highlights**: 실험 결과는 네 개의 공개 데이터셋에서 시행되었으며, 새롭게 도입된 알고리즘이 추천 순위 결과를 상당히 개선시켰음을 보여줍니다. 특히, 파라미터 수를 크게 증가시키지 않으면서도 추천 성능을 높일 수 있음을 입증하였습니다. 이러한 결과는 아이템 간의 연관정보가 추천 시스템에서 필수적이라는 점을 잘 보여줍니다.



New uploads on arXiv(cs.CV)

### VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction (https://arxiv.org/abs/2501.01957)
Comments:
this https URL

- **What's New**: 최근 여러 모달리티(Modalities) 대형 언어 모델(MLLMs)은 주로 시각적(Visual) 및 텍스트(텍스트) 모달리티를 통합하는 데 초점을 맞추었지만, 음성(Speech)의 역할 또한 중요하다는 점에 주목해야 합니다. 본 논문에서는 음성과 시각 정보를 이해할 수 있는 다단계 훈련 방법론을 제안합니다. 이 방법론을 통해 VITA-1.5 모델은 이미지, 비디오 그리고 음성 작업에서 뛰어난 성능을 발휘하며, 별도의 음성 인식(ASR) 및 텍스트-음성 변환(TTS) 모듈 없이도 원활한 음성 대화를 가능하게 합니다.

- **Technical Details**: VITA-1.5는 세 단계의 훈련 전략을 통해 개발되었습니다. 첫 단계는 시각-언어 모델의 구축을 위한 훈련 과정으로, 이미지와 비디오 이해를 위한 비주얼 어댑터(Visual Adaptor)를 포함합니다. 둘째 단계에서는 오디오 인코더(Audio Encoder)를 훈련하여 음성 입력을 처리하고, 마지막 단계에서는 오디오 디코더(Audio Decoder)를 훈련하여 최종적으로 말하기 응답을 생성합니다. 이를 통해 VITA-1.5는 강력한 멀티모달 이해를 유지하면서도 즉각적인 응답 기능을 제공합니다.

- **Performance Highlights**: VITA-1.5는 이미지, 비디오 및 음성 기준에서 최신 모델과 비교했을 때 우수한 성능을 보여줍니다. 특히, 기존 MLLMs와 비교하여 음성 처리 능력에서 중요한 개선을 이루었습니다. 이러한 성능 덕분에 VITA-1.5는 거의 실시간에 가까운 음성 및 시각 상호작용이 가능하며, 다양한 멀티모달 태스크에 적합합니다.



### VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignmen (https://arxiv.org/abs/2501.01949)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 모노큘러 비디오로부터 효율적으로 정확한 3D 모델을 재구성하는 VideoLifter라는 새로운 프레임워크를 소개합니다. 기존의 방법들은 카메라 파라미터를 사전 계산해야 하며, 매 프레임마다 재구성 파이프라인이 필요해 오류가 누적되기 쉬웠습니다. VideoLifter는 비디오 시퀀스로부터 전역적으로 희소에서 밀도로의 3D 표현을 최적화하여 이러한 제약 조건을 극복합니다.

- **Technical Details**: VideoLifter는 비디오 시퀀스를 여러 개의 로컬 윈도로 분할한 후, 이를 기반으로 프레임을 매칭하고 등록합니다. 또한 일관된 조각을 구성하고 이를 계층적으로 정렬하여 통합된 3D 모델을 만듭니다. 이 구현은 카메라 포즈와 3D 구조를 점진적으로 정제하여 재투영 오류를 최소화함으로써 더 높은 정확성과 견고성을 제공합니다.

- **Performance Highlights**: VideoLifter는 기존 방법들에 비해 훈련 시간을 82% 이상 단축시키고, 시각적 정확성 및 계산 효율성 측면에서 뛰어난 성능을 보입니다. 다양한 최신 방법들과 비교했을 때, VideoLifter는 5배 이상의 속도 향상 및 향상된 뷰 합성 품질을 기록했습니다.



### Bridging Classification and Segmentation in Osteosarcoma Assessment via Foundation and Discrete Diffusion Models (https://arxiv.org/abs/2501.01932)
Comments:
          Accepted for presentation at the 2025 IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이번 연구에서는 FDDM이라는 새로운 프레임워크를 제안하여 골육종(osteosarcoma)의 정확한 괴사(nnecrosis) 평가를 향상시키고 있습니다. 이 프레임워크는 패치(classification)와 영역 기반 세분화(segmentation)를 연결하여, 전체 슬라이드 이미지(WSI) 분석에서 종합적인 성능을 개선합니다. FDDM은 90% 이상의 괴사 비율에서 80% 이상의 5년 생존률과 관계가 있는 중요한 평가 방법입니다.

- **Technical Details**: FDDM은 두 단계로 작동하며, 첫 번째 단계에서는 패치 기반 분류(patch-based classification)를 수행하고, 두 번째 단계에서는 지역 기반 정제를 통해 분류된 결과를 개선합니다. 이 방법은 작은 패치의 풍부한 정보를 활용하여 패치 분류의 특징을 최대한 살리고, 이를 통해 대규모 병리 이미지 분석의 한계를 극복합니다. 모델 학습에는 CNN 대신 저차원 적응(Low-rank Adaptation, LoRA) 방법을 사용하여 Vision Transformer (ViT)의 파라미터를 효과적으로 조정합니다.

- **Performance Highlights**: FDDM은 기존의 방법들보다 최대 10% 높은 mIOU(mean Intersection over Union)와 32.12% 향상된 괴사 비율 추정 능력을 보여주며 성능 평가에서 탁월한 결과를 얻었습니다. 이러한 성과는 복잡한 의료 영상 과제에서의 파운데이션 모델과 확산 기반 정제의 잠재력을 강조합니다. FDDM은.opengraph.
이 분야에서 새로운 기준을 설정하며, 향후 의료 이미지 분석에서의 활용 가능성을 제시합니다.



### Mitigating Hallucination for Large Vision Language Model by Inter-Modality Correlation Calibration Decoding (https://arxiv.org/abs/2501.01926)
- **What's New**: 이 논문에서는 LVLMs(대형 비전-언어 모델)의 환각(hallucination) 문제를 해결하기 위해 Inter-Modality Correlation Calibration Decoding (IMCCD) 방법을 제안합니다. 기존의 방법들이 언어 선입견에 대한 과도한 의존성을 줄이기 위한 개선을 시행하는 동안, IMCCD는 훈련 없이 이 문제를 해결할 수 있도록 설계되었습니다. 이 접근법은 CMVED(Cross-Modal Value-Enhanced Decoding) 모듈을 통해 환각을 완화하는 새로운 대비 디코딩 메커니즘을 통합합니다.

- **Technical Details**: 제안된 방법은 CMVED를 포함하여 시각적 콘텐츠와 관련된 중요한 크로스 모달 주의(attention) 가중치를 마스킹(masking)하면서 왜곡된 분포를 추정하는 과정을 포함합니다. 이는 환각을 유발하는 스푸리어스(spurious) 상관관계를 조절하는 방식이며, 또한 CDAR(Content-Driven Attention Refinement) 모듈이 시각적 콘텐츠에 집중할 수 있도록 도움을 줍니다. 이 과정에서 단일 모달(uni-modality)에 대한 의존성을 방지합니다.

- **Performance Highlights**: 다양한 환각 벤치마크에서 실험을 통해 IMCCD가 기존의 최첨단 기법들보다 더 효과적으로 LVLM의 환각을 줄이는 것으로 나타났습니다. 이 새로운 접근법은 튜닝(tuning)이나 추가적인 학습 없이도 LVLM의 응용 가능성을 높이는 데 기여할 것으로 기대됩니다. 연구 결과는 LVLM에서의 환각 완화에 대한 효율성과 일반성을 명확하게 입증합니다.



### Transformer-Driven Inverse Problem Transform for Fast Blind Hyperspectral Image Dehazing (https://arxiv.org/abs/2501.01924)
Comments:
          This work has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS)

- **What's New**: 이번 연구에서는 하이퍼스펙트럼 이미지(Hyperspectral Image, HSI)에서 발생하는 안개를 제거하는 새로운 알고리즘인 T2HyDHZ를 소개합니다. 이 알고리즘은 씬을 복원하기 위해 스펙트럴 슈퍼 해상도(Spectral Super-Resolution, SSR) 문제로 재구성하여 하이퍼스펙트럼 이미지를 얻는 방식으로 작동합니다. T2HyDHZ는 사용자에게 부과되는 사전 선택 작업 없이도 안개가 있는 영역을 자동으로 처리할 수 있는 블라인드 알고리즘입니다.

- **Technical Details**: T2HyDHZ 알고리즘은 먼저 비오염 스펙트럴 밴드를 자동 선택한 후, 선택된 밴드에 대해 스펙트럴 업샘플링을 적용하여 클린 HSI를 생성합니다. 이 후 깊은 변환기 네트워크를 통해 최종적으로 디헤이즈된 HSI가 정제됩니다. 이 과정에서 글로벌 어텐션 메커니즘이 적용되어 비국소적 정보가 효과적으로 포착됩니다.

- **Performance Highlights**: C웜터칭 T2HyDHZ는 기존의 하이퍼스펙트럼 디헤이징 알고리즘에 비해 색상 왜곡이 적고 더 나은 성능을 나타냅니다. 실험 결과, T2HyDHZ는 이미지 복원에서 상대적으로 높은 품질을 유지하며, 이는 다양한 실험을 통해 입증되었습니다.



### Detecting and Mitigating Adversarial Attacks on Deep Learning-Based MRI Reconstruction Without Any Retraining (https://arxiv.org/abs/2501.01908)
- **What's New**: 본 연구에서는 магнитной резонансной томографии(MRI) 복원 모델의 적대적 공격(adversarial attack)을 탐지하고 완화하는 새로운 접근 방식을 제안합니다. 기존의 방법과 달리 재학습이 필요 없이 루프 측정 일관성(cyclic measurement consistency)에 기반하여 적대적 교란을 감지합니다. 제안된 방식은 기존의 다수의 방법들보다 실험적으로 보다 효과적인 성능을 보여주며, 다양한 데이터셋과 공격 유형에서도 우수한 결과를 기록했습니다.

- **Technical Details**: MRI는 주파수 영역에서 수집된 원시 측정 데이터를 사용하는데, 이 데이터는 수신 코일(receiver coils)에서 수집됩니다. 본 연구에서는 PD-DL(Physics-driven Deep Learning) 기술을 채택하여 구조화된 손실 함수(objective function)를 사용하여 적대적 공격을 탐지하고 완화합니다. 공격 없는 경우 재구성 결과가 서로 일관된 결과를 보여야 하며, 공격이 있을 경우에는 두 재구성 간의 큰 불일치가 나타나는 것을 기반으로 합니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋, PD-DL 네트워크, 공격 유형 및 강도에서 적대적 혼란의 영향을 크게 줄였습니다. 특히 재학습 없이도 공격 크기나 생성된 공격 알고리즘에 상관없이 효과적으로 적용 가능한 것을 확인했습니다. 이러한 성과는 기존의 기법들보다 정량적 및 정성적으로 높은 성능을 제공하며, 비교적 깨끗한 이미지에서의 성능에도 영향을 미치지 않는 장점을 가집니다.



### Virgo: A Preliminary Exploration on Reproducing o1-like MLLM (https://arxiv.org/abs/2501.01904)
Comments:
          Technical Report on Slow Thinking with LLMs: Visual Reasoning

- **What's New**: 최근 대규모 언어 모델(LLM)을 기반으로 한 느린 사고(reasoning) 시스템이 주목받고 있으며, 이 연구에서는 다중모드 대규모 언어 모델(MLLM)에 이러한 능력을 적용할 수 있는 방법을 탐구합니다. 연구의 주요 결과로, 텍스트 기반의 긴 사고 데이터를 활용하여 MLLM을 세밀하게 조정(fine-tune)함으로써 다중모드 느린 사고 시스템인 Virgo가 개발되었습니다. 이를 통해 텍스트 기반의 합리적인 과정이 MLLM에 효과적으로 전이될 수 있음을 보여주며, 일반적으로 시각적 데이터보다 텍스트 데이터가 느린 사고 능력을 이끌어내는 데 더 효과적이라는 점을 밝혔습니다.

- **Technical Details**: 이 연구에서는 Qwen2-VL-72B-Instruct 모델을 중심으로 MLLM을 조정하기 위해 약 5,000개의 긴 사고 지침 인스턴스를 수집했습니다. 이 데이터는 주로 수학, 과학, 코드 및 퍼즐 도메인에 걸쳐 있으며, 텍스트 형식은 특별한 기호로 정의된 사고 과정과 최종 솔루션으로 나뉩니다. 실험은 MathVerse, MathVision, OlympiadBench 및 MMMU와 같은 4개의 도전적인 벤치마크에서 수행되어, Virgo가 상업적 추론 시스템과 비슷한 성능을 나타냄을 확인하였습니다.

- **Performance Highlights**: Virgo 시스템은 텍스트 기반의 긴 사고 데이터를 사용하여 놀라운 성과를 이루었으며, 상업적 느린 사고 시스템과 비교했을 때 동등하거나 그 이상의 결과를 보였습니다. 실험 결과, 일반적으로 텍스트 기반의 지침이 MLLM의 느린 사고 능력을 유도하는 데 더 효과적이라는 것이 확인되었습니다. 이 연구는 다중모드 느린 사고 시스템 개발에 있어 잠재적인 도전과 기회를 제시하며, 언어 모델의 특성과 전이 가능성에 대한 깊은 통찰을 제공합니다.



### ANTHROPOS-V: benchmarking the novel task of Crowd Volume Estimation (https://arxiv.org/abs/2501.01877)
- **What's New**: 이번 논문에서는 Crowd Volume Estimation (CVE)이라는 새로운 과제를 소개합니다. CVE는 RGB 이미지를 기반으로 군중의 전체 체적을 추정하는 과정입니다. 이 과제는 이벤트 관리 및 공공 안전 외에도 인간의 체중을 추정하는 데 유용하여 인프라의 스트레스 평가와 균형 잡힌 체중 관리를 가능하게 합니다. 또한, ANTHROPOS-V라는 합성 및 포토리얼리스틱 비디오 데이터셋을 기반으로 CVE에 대한 최초의 벤치마크를 제안합니다.

- **Technical Details**: CVE를 위한 기술적인 접근으로는 두 가지 주요 방향이 있습니다. 첫 번째는 Crowd Counting (군중 카운팅) 기법에 기반한 접근이고, 두 번째는 Human Mesh Recovery (인간 형상 복원) 파이프라인을 재구성한 것입니다. 우리는 Volume Density Maps라는 새로운 감독 방식과 Per-Part Volume Density Maps를 도입하여 각 인간의 체적을 더 정확하게 추정하는 방법을 제안합니다. ANTHROPOS-V 데이터셋은 다양한 도시 환경에서 대규모 군중을 포착하고 각 개인의 체적을 주석으로 포함하고 있습니다.

- **Performance Highlights**: 논문에서 제안하는 STEERER-V 모델은 기존의 베이스라인을 초월하여 우수한 성능을 보입니다. 이 모델은 RGB 이미지에서 직접적으로 군중의 체적을 추정하는 능력을 입증하였으며, 제안된 감독 방법인 Per-Part Volume Density Maps를 통해 더욱 향상된 결과를 얻었습니다. 또한 ANTHROPOS-V 데이터셋은 많은 수의 대상에 대해 주석된 체적 정보를 제공하여 CVE 연구의 새로운 기준을 제시합니다.



### Towards Hard and Soft Shadow Removal via Dual-Branch Separation Network and Vision Transformer (https://arxiv.org/abs/2501.01864)
Comments:
          11 pages, 5 figures, IEEE International Conference on Machine Learning and Cybernetics (ICMLC) 2024

- **What's New**: 이 논문에서는 컴퓨터 비전에서 중요한 이미지 그림자 제거를 위한 새로운 접근 방식을 제시하고 있습니다. 특히, 기존의 방법들이 하드(shadow)와 소프트(shadow)를 구분하지 않고 처리하는 데에 대한 한계를 보완하기 위해, 이중 경로 모델(dual-path model)을 제안하였습니다.

- **Technical Details**: 제안된 모델은 하드 그림자와 소프트 그림자를 각각 별도로 처리하며, 이를 위해 특수 설계된 손실 함수(loss functions)를 활용합니다. 그림자 유형을 분류하고 적합한 경로를 통해 처리하여 그림자가 없는 출력을 생성합니다. 또한, Vision Transformer와 UNet++를 통합하여 엣지 세부 사항과 특징 융합(feature fusion)을 개선하고 있습니다.

- **Performance Highlights**: 제안된 모델은 최신 방법들 대비 우수한 성능을 보이며, ISTD 데이터셋에서 2.905 RMSE (Root Mean Square Error) 값을 달성하여 일반적인 단일 경로(single-path) 접근법보다 더 나은 효과를 입증하였습니다.



### UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery (https://arxiv.org/abs/2501.01855)
- **What's New**: 이 논문은 UAV 이미지를 위한 효율적인 객체 탐지 변환기(UAV-DETR) 프레임워크를 제안합니다. 기존의 UAV 객체 탐지 알고리즘들이 수작업으로 설계된 구성 요소에 의존하는 반면, UAV-DETR은 이러한 수작업 요소에 의존하지 않고도 강력한 성능을 발휘합니다. 다중 스케일 특징 융합 및 주파수 증강 모듈이 포함되어, 다양한 스케일에서 공간적 및 주파수 정보를 포착합니다.

- **Technical Details**: UAV-DETR 프레임워크는 주파수 중점을 둔 다운 샘플링 모듈과 의미론적 정렬 및 보정 모듈을 통해 다양한 융합 경로에서의 특징을 정렬하고 융합합니다. 이 모델은 고주파 정보를 유지하고 소규모 및 가려진 객체를 감지하는 데 유리한 성능을 보여줍니다. 특히, Inner Scylla intersection over union (Inner-SIoU)을 적용하여 객체 감지 성능을 더욱 높이고 있습니다.

- **Performance Highlights**: 실험 결과, 본 연구의 방법은 VisDrone 데이터셋에서 AP를 3.1% 및 AP₅₀을 4.2% 향상시키며, UAVVaste 데이터셋에서도 유사한 성능 개선이 관찰되었습니다. UAV-DETR은 기존의 YOLO 모델을 초월하여 정확도와 실시간 성능 모두에서 뛰어난 결과를 나타냅니다.



### Semantic Segmentation for Sequential Historical Maps by Learning from Only One Map (https://arxiv.org/abs/2501.01845)
- **What's New**: 이 논문은 역사적 지도를 디지털화하기 위한 자동화된 접근법을 제안합니다. 특히, 이 연구에서는 역사적 지도에서 각 픽셀에 의미적 레이블을 할당하는 깊은 학습 기반의 semantic segmentation 방법을 사용합니다. 이 과정에서 큰 도전 과제는 깊은 신경망 모델을 훈련시키기 위한 ground-truth annotations의 부족이며, 이를 해결하기 위해 약하게 지도된 age-tracing 전략을 도입합니다. 이 방법은 인접한 시간대의 지도와의 유사성을 활용하여 훈련 과정을 안내합니다.

- **Technical Details**: 이 연구는 1897년부터 2017년까지의 독일 하멜른의 역사적 지도를 대상으로 다섯 가지 클래스—삼림, 초지, 정착지, 흐르는 물, 고인 물—를 정확하게 식별하는 semantic segmentation 모델을 훈련합니다. 기존의 지도는 복잡한 카토그래픽 스타일과 일관되지 않은 레이블링으로 구성되어 있어 모델 훈련에 도전을 제공합니다. 이 논문에서는 anchor year에서의 레이블이 있는 데이터를 활용하여 pseudo-labels를 생성하고 이러한 레이블을 사용하여 전체 시간 범위의 지도를 통해 모델을 보완하는 fine-tuning 전략을 제시합니다.

- **Performance Highlights**: 제안된 age-tracing 전략은 baseline 모델에 비해 segmentation 성능을 크게 향상시킵니다. 실험 결과, 최상의 경우 mean Intersection over Union (mIoU) 지표에서 77.3%를 달성하며, 이는 약 20% 향상된 수치입니다. 또한, 세분화된 모델은 평균 전체 정확도 97%를 기록하여 역사적 지도 디지털화의 효과성을 입증합니다.



### Dedicated Inference Engine and Binary-Weight Neural Networks for Lightweight Instance Segmentation (https://arxiv.org/abs/2501.01841)
Comments:
          Camera-ready version for CVPR 2024 workshop (Embedded Vision Workshop)

- **What's New**: 본 논문은 Binary-weight Neural Networks (BNNs)에서 두 가지 작동 모드를 갖춘 추론 엔진의 하드웨어 아키텍처 설계 방법론을 제안합니다. 컴퓨터 비전의 다양한 애플리케이션에서 계산 비용을 줄이기 위해 가중치 이진화 및 활성화 양자화를 통해 계산 비용을 절감합니다. 제안된 방법은 Multiply-Accumulate (MAC) 연산을 단순화하여 추론 엔진의 게이트 수를 효과적으로 줄일 수 있습니다.

- **Technical Details**: 제안된 하드웨어 아키텍처는 BNNs의 효율적인 처리를 가능하게 하며, MAC 연산에서 곱셈 연산을 제거하고 비트 단위의 연산을 수행함으로써 하드웨어 비용을 52% 절감합니다. 이 엔진은 SegNeXt의 백본과 SparseInst의 디코더를 결합한 경량 네트워크를 활용하며, 단지 비트별 연산과 덧셈 연산만을 사용하여 결과를 처리할 수 있습니다. 논문에서는 정확성과 비트 너비 간의 균형을 찾는 중요성을 강조합니다.

- **Performance Highlights**: 제안된 추론 엔진은 관련 연구에 비해 9.8%의 계산 비용만으로 인스턴스 분할 알고리즘을 처리할 수 있으며, YOLACT 모델에 비해 모델 크기가 77.7배 더 작지만 'Person' 카테고리에서 더 높은 정확도를 달성합니다. 이 결과는 경량 네트워크가 실제 애플리케이션에서 효과적으로 사용될 수 있음을 보여줍니다.



### MoColl: Agent-Based Specific and General Model Collaboration for Image Captioning (https://arxiv.org/abs/2501.01834)
- **What's New**: 이번 논문에서는 복잡한 이미지 캡셔닝(Imaging Captioning) 작업에 대한 새로운 접근 방식을 제안합니다. 저자들은 도메인 특화 모델과 일반 지식을 효과적으로 통합할 수 있는 에이전트-강화 모델 협업 프레임워크인 MoColl을 소개합니다. 이를 통해 VQA(Visual Question Answering) 모델과 LLM(Large Language Model) 기반 에이전트를 결합하여 이미지 캡셔닝 작업을 보다 효율적으로 처리할 수 있습니다.

- **Technical Details**: MoColl 프레임워크는 복잡한 이미지 캡셔닝 작업을 일련의 질문-답변 하위 작업으로 분해합니다. 전문화된 VQA 모델은 특정 도메인에 대한 시각적 분석을 수행하는 도구 역할을 하고, LLM 기반 에이전트는 의미 있는 질문을 formulates하고 결과적 질문-답변 쌍을 통합하여 일관성 있는 캡션을 생성합니다. 이 과정에서 에이전트는 VQA 모델의 도메인 특화 능력을 향상시키기 위한 훈련 양식을 유도합니다.

- **Performance Highlights**: 방사선 보고서 생성을 위한 실험 결과는 제안된 MoColl 프레임워크의 효과성을 입증하며 생성된 캡션의 품질에서 상당한 개선을 보여줍니다. 이 발견은 복잡한 도메인 특화 작업에서 도메인 전문성과 일반 적응성을 연결할 수 있는 에이전트 기반 시스템의 잠재력을 강조합니다. 따라서 이 연구는 이미지 캡셔닝 분야에서의 새로운 가능성을 제시합니다.



### Uncertainty-Aware Label Refinement on Hypergraphs for Personalized Federated Facial Expression Recognition (https://arxiv.org/abs/2501.01816)
- **What's New**: 최근 얼굴 표정 인식(FER) 모델은 개인화된 연합 학습(personalized federated learning) 프레임워크를 통해 개선되고 있으며, 이는 개인 데이터의 프라이버시를 보호할 수 있는 유용한 방법입니다. 본 논문에서는 하이퍼그래프(hypergraph)를 활용한 불확실성-인식 레이블 정제 방법(AMY)을 개발하였습니다. 이를 통해 각 클라이언트에서 개인화된 FER 모델을 학습하고 불확실성 제거가 가능합니다.

- **Technical Details**: AMY 방법은 각 클라이언트에서 백본(backbone), 불확실성 추정 (UE) 블록 및 표현 분류 (EC) 블록으로 구성됩니다. UE 블록에서는 하이퍼그래프를 활용하여 복잡한 고차원 관계를 모델링하고, 이를 바탕으로 불확실성 피처를 도출합니다. EC 블록에서는 하이퍼그래프에서 레이블 전파(label propagation)를 수행하여 재훈련에 필요한 고품질 레이블을 얻습니다.

- **Performance Highlights**: 두 개의 실세계 얼굴 표정 데이터베이스에 대한 실험 결과, 제안된 방법이 여러 최첨단 방법들보다 consistently 성능을 개선했습니다. 이는 불확실성 추정 및 레이블 정제에 있어 하이퍼그래프 모델링의 장점을 보여줍니다. 따라서 개인화된 FER 작업에서 효과적인 성능을 달성할 수 있음을 확인했습니다.



### MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation (https://arxiv.org/abs/2501.01808)
- **What's New**: 이 논문은 감정을 보다 효과적으로 모델링하기 위한 혁신적인 접근 방식을 제안합니다. Mixture of Emotion Experts (MoEE) 모델을 통해 여섯 가지 기본 감정을 분리하고 복합 감정을 정밀하게 합성할 수 있게 되었습니다. 또한, DH-FaceEmoVid-150 데이터셋을 구축하여 다양한 감정 표현을 포함하는 고해상도 비디오 콘텐츠를 제공합니다.

- **Technical Details**: MoEE 모델은 기본 감정과 복합 감정을 모델링하는 데 초점을 맞추고 있습니다. 이 모델은 각 감정을 정밀히 동기화하여 출력할 수 있는 능력을 갖추고 있으며, 다양한 멀티모달 입력을 통해 감정 제어의 유연성을 강화합니다. 전체 데이터셋은 여섯 가지 기본 감정과 네 가지 복합 감정을 포함하여 총 150시간의 비디오 데이터를 제공합니다.

- **Performance Highlights**: MoEE 프레임워크와 DH-FaceEmoVid-150 데이터셋을 이용한 실험 결과는 복합적인 감정 표현과 미세한 얼굴 세부 묘사에서 뛰어난 성능을 보여주었습니다. 이들은 오디오 기반 포트레이트 애니메이션의 새로운 기준을 제시하며, 다양한 감정을 보다 자연스럽게 생성하는 데 있어 우수한 결과를 입증하였습니다.



### JoyGen: Audio-Driven 3D Depth-Aware Talking-Face Video Editing (https://arxiv.org/abs/2501.01798)
- **What's New**: JoyGen은 음성 기반의 입술 움직임 생성 및 시각적 외관 합성을 포함하는 새로운 두 단계 프레임워크입니다. 이 프레임워크는 3D 재구성 모델과 audio2motion 모델을 사용하여 정확한 입술-음성 동기화를 위한 포괄적인 제어를 제공합니다. 또한, 130시간 분량의 고품질 중국어 말하기 얼굴 데이터셋을 구축하였습니다.

- **Technical Details**: JoyGen의 첫 번째 단계에서는 3D 재구성 모델이 정체성 계수(identity coefficients)를 예측하고, audio2motion 모델이 표현 계수(expression coefficients)를 추론하여 정확한 입술 움직임 생성을 지원합니다. 두 번째 단계에서는 오디오 특징과 얼굴 깊이 맵(facial depth map)을 통합하여 정밀한 입술-음성 동기화를 위한 감독을 제공합니다. 이러한 기술적 접근 방식은 기존 방법들이 겪는 문제들을 해결합니다.

- **Performance Highlights**: 실험 결과, JoyGen은 기존 기술들보다 뛰어난 입술-음성 동기화 및 시각적 품질을 달성하였습니다. 특히, HDTF 데이터셋과 추가로 구축된 데이터셋 모두에서 차별화된 성능을 보여주었습니다. 이러한 성능 향상은 고유의 프레임워크 덕분에 가능하였습니다.



### A Minimal Subset Approach for Efficient and Scalable Loop Closur (https://arxiv.org/abs/2501.01791)
Comments:
          7 pages, 8 Figures, 2 Tables. Submitted

- **What's New**: 본 논문에서는 Pose Graph Optimization (PGO)과 루프 클로저 검출(loop closure detection)의 결합된 문제를 해결하기 위해 최적화된 키프레임 샘플링 방법을 제안합니다. 이를 통해 컴퓨테이셔널 오버헤드를 줄이면서 루프 클로저 검출의 성능을 유지할 수 있습니다. 제안된 Minimal Subset Approach (MSA)는 슬라이딩 윈도우 프레임워크 내에서 중복 최소화와 정보 보존을 통해 효율적으로 키프레임을 축소하는 방법입니다. 결과적으로, MSA는 다양한 환경에서 일관된 성능을 보여주며 수동 파라미터 튜닝이 필요하지 않습니다.

- **Technical Details**: PGO의 목적은 로봇의 경로를 그래프 기반으로 표현하여 오류를 최소화하는 것입니다. 이 과정에서 각각의 노드는 로봇의 자세(pose)를 나타내며, 엣지는 두 개의 자세 사이의 공간 제약을 인코딩합니다. 따라서 에지의 쌍에서 발생하는 상대 변환의 차이를 수치적으로 표현하는 오류를 정의하고, 최적의 포즈 구성(configuration)을 찾는 것을 목표로 합니다. MSA는 이러한 PGO와 루프 클로저 검출 문제를 통합하여 다루며, 슬라이딩 윈도우 최적화를 통해 환경 변화에 적응합니다.

- **Performance Highlights**: MSA는 기존의 기준 방법과 유사한 성능을 유지하면서도 컴퓨테이셔널 요구 사항을 크게 줄이는 데 성공했습니다. 특히, MSA는 다양한 환경에서 일관된 결과를 보여주었으며, 실제로 150,000개의 후보를 처리해야 하는 복잡한 데이터셋에서도 우수한 성능을 달성했습니다. 이는 MSA의 환경 구애 받지 않는 특성과 수동 파라미터 튜닝이 필요 없는 점에서 특히 주목할 만합니다. 결론적으로 MSA는 SLAM 시스템의 성능을 크게 향상시키는 가능성을 보여줍니다.



### Ingredients: Blending Custom Photos with Video Diffusion Transformers (https://arxiv.org/abs/2501.01790)
- **What's New**: 이 논문에서는 여러 개의 특정 정체성(Identity, ID) 사진을 통합하여 비디오 생성 맞춤화하는 강력한 프레임워크인 Ingredients를 제안합니다. 이 방법은 세 가지 주요 모듈로 구성되어 있으며, 얼굴 특징을 추출하고, 얼굴 임베딩을 변환하며, 여러 ID 임베딩을 동적으로 결합하여 공간-시간 영역으로 할당합니다. 우리의 접근 방식은 사용자가 정의한 텍스트 프롬프트를 반영하여 개인화된 비디오 콘텐츠를 생성하는 데 도움을 줍니다.

- **Technical Details**: Ingredients는 크게 세 가지 구성 요소로 이루어져 있습니다: (i) 얼굴 추출기(facial extractor)로 개인 ID의 다양한 얼굴 특징을 추출하고, (ii) 다중 스케일 프로젝터(multi-scale projector)로 얼굴 임베딩을 비디오 확산 변환기(image query)의 맥락 공간으로 매핑합니다. (iii) ID 라우터(ID router)는 각 ID 임베딩을 그에 해당하는 영역으로 동적으로 할당하고 통합합니다. 이 멀티 스테이지 훈련 프로세스는 ID 임베딩과 라우터 구성 요소를 순차적으로 최적화하여, 비디오 합성에서 뛰어난 적응성과 정밀도를 제공합니다.

- **Performance Highlights**: 제시된 방법은 기존 기술들에 비해 탁월한 성능을 보여줍니다. 사용자 정의에 적합한 유연성을 제공하며, 개인적인 스토리텔링, 프로모션 비디오 및 창의적 프로젝트와 같은 다양한 애플리케이션에 잘 어울립니다. 또한, 각 생성된 비디오는 사용자 비전과 조화를 이루고 개인화된 터치를 유지하여 고객의 요구를 충족합니다.



### TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation (https://arxiv.org/abs/2501.01770)
Comments:
          Accpeted by the 39th Annual AAAl Conference on Artificial Intelligence (AAAl 2025)

- **What's New**: 이번 연구에서는 TCPFormer라는 새로운 방법을 제안합니다. TCPFormer는 복잡한 2D 포즈 시퀀스 내의 시간적 상관관계를 보다 효과적으로 모델링하기 위해 암시적 포즈 프록시(implicit pose proxy)를 중간 표현으로 활용합니다. 이 방법은 Proxy Update Module (PUM), Proxy Invocation Module (PIM), Proxy Attention Module (PAM)이라는 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: 아키텍처는 세 가지 모듈로 구성되어 있습니다. PUM은 포즈 시퀀스에서 유용한 정보를 통해 포즈 프록시를 업데이트합니다. PIM은 포즈 프록시를 사용하여 포즈 시퀀스의 기능 표현 능력을 향상시키며, PAM은 두 개의 크로스-어텐션 행렬을 활용하여보다 효과적인 시간적 상관관계를 얻습니다.

- **Performance Highlights**: Human3.6M 및 MPI-INF-3DHP 데이터셋에서의 실험 결과, TCPFormer는 이전의 최첨단 방법들보다 뛰어난 성능을 보였습니다. 또한, 각 구성 요소의 기여를 평가하기 위한 포괄적인 ablation study도 수행하였습니다.



### LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction (https://arxiv.org/abs/2501.01767)
Comments:
          project page: this https URL

- **What's New**: 이 논문은 Autoregressive, Multimodal Vision Language Models (AVLMs)를 사용하여 논리 이상 탐지(logical anomaly detection, LA) 문제를 해결하는 새로운 알고리즘인 LogicAD를 소개합니다. LogicAD는 기존의 불필요한 수작업 주석과 데이터에 의존하지 않으며, AVLM에서 추출된 텍스트 특징을 활용하여 뛰어난 성과를 보입니다. 이 방법은 MVTec LOCO AD에서 SOTA 성능을 달성하며, AUROC 86.0%와 F1-max 83.7%로 기존 방법을 상당히 능가합니다.

- **Technical Details**: 논문의 핵심 기술은 AVLM과 논리 추론기(logic reasoner)를 결합하여 LA 탐지에 적합한 모델을 만드는 것입니다. 텍스트 특징을 메모리 뱅크로 활용하여 논리적이고 신뢰성 있는 설명을 생성하는 파이프라인을 설계하였습니다. 이 과정에서는 자동 정리 이론 증명기(automated theorem prover, ATP)를 활용하여 탐지된 이상에 대한 자세한 설명을 제공합니다.

- **Performance Highlights**: LogicAD는 기존 SOTA 방법에 비해 상당히 우수한 성능을 보여주며, 특히 한 번의 학습으로 LA 탐지를 수행할 수 있다는 점에서 큰 장점을 가집니다. 이 모델은 다양한 공개 데이터셋에서 평가되었으며, 전통적인 방법보다 효율적이고 효과적인 방법으로 자리매김하고 있습니다. 이러한 혁신적인 접근 방식은 최신 AVLM 기술이 AD 분야에서의 역할을 확장할 가능성을 보여줍니다.



### Adverse Weather Conditions Augmentation of LiDAR Scenes with Latent Diffusion Models (https://arxiv.org/abs/2501.01761)
Comments:
          This is an intermediate version of our work

- **What's New**: 이 연구에서는 자율 주행 응용 프로그램을 위한 LiDAR 장면의 생성 방법을 제안합니다. 특히, 악천후에서의 장면 데이터를 생성하기 위해 새로운 처리 방법인 latent diffusion process를 도입하였습니다. 이 방법은 오토인코더(autoencoder)와 latent diffusion 모델을 기반으로 하며, 명확한 날씨 조건에서의 LiDAR 장면을 활용하여 더 현실적인 악천후 조건의 장면을 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 입력으로 명확한 날씨의 3333D 포인트 클라우드 이미지를 받아서 악천후를 생성합니다. 이 과정에서 discrete latent space를 생성하기 위해 latent quantization(LQ) 레이어를 사용하며, latent diffusion 모델을 통해 노이즈를 제거합니다. 악천후 조건을 위한 feature-wise linear modulation(FiLM) 레이어를 통해 생성되는 latent 공간의 구조를 복원하고, afterward autoencoder를 통해 최종적인 악천후 장면을 생성 및 세련되게 합니다.

- **Performance Highlights**: 연구자들은 CenterPoint라는 3333D 객체 탐지 모델을 통해 제안된 방법의 효과를 검증할 계획입니다. 생성된 악천후를 활용하여 모델을 학습하고, 실제 악천후 조건에서의 성능을 평가할 예정입니다. 이 연구는 명확한 날씨의 장면을 악천후로 보강함으로써 더 강력한 객체 탐지 성능을 달성할 수 있도록 기대하고 있습니다.



### From Age Estimation to Age-Invariant Face Recognition: Generalized Age Feature Extraction Using Order-Enhanced Contrastive Learning (https://arxiv.org/abs/2501.01760)
- **What's New**: 이번 연구에서는 Age-related facial analysis에 필수적인 일반화된 나이(feature) 추출을 위해, Order-Enhanced Contrastive Learning (OrdCon)이라는 새로운 방법론을 제안합니다. OrdCon는 다양한 데이터 세트 간의 도메인 격차를 최소화하는 데 초점을 맞추며, 나이 진행 방향을 모델링하여 나이 특성을 효과적으로 추출합니다. 또한 새로운 soft proxy matching loss를 활용하여 각 나이 클러스터의 중심에 특징을 위치시키는 것을 목표로 합니다.

- **Technical Details**: OrdCon은 일반적인 contrastive learning 접근방식과 달리, 특징 공간(feature space) 내에서의 방향 벡터를 대비합니다. 이를 통해 나이의 자연스러운 진행 방향을 고려하여 얼굴 특징에서의 연속적인 변화를 포착합니다. 또한, proxy-based metric learning 기법을 통해 intra-class variance를 최소화하며, 각 나이 클러스터의 중심을 나타내는 학습 가능한 proxy를 사용합니다.

- **Performance Highlights**: OrdCon는 다양한 벤치마크 데이터 세트에서 기존 최첨단 방법들과 비교해 동등한 성능을 달성하며, cross-dataset 실험에서는 나이 추정(task) 평균 절대 오차를 약 1.38 감소시키고, AIFR의 평균 정확도를 1.87% 향상시킵니다. 이로써 OrdCon이 Robust한 특징 추출 및 일반화 능력을 갖추었다는 것을 입증합니다.



### Augmentation Matters: A Mix-Paste Method for X-Ray Prohibited Item Detection under Noisy Annotations (https://arxiv.org/abs/2501.01733)
Comments:
          The manuscript has been ACCEPTED for publication as a regular paper in the IEEE Transactions on Information Forensics & Security

- **What's New**: 이번 연구에서는 노이즈가 있는 주석(annotations) 하에서 강력한 X-ray 금지 물품 탐지기를 학습하는 방법을 제안합니다. 기존의 방법들은 주석의 정확성을 가정하지만, X-ray 이미지에서의 물체 겹침으로 인해 정확한 주석을 얻기가 매우 어렵습니다. 이를 해결하기 위해, 저자는 데이터 증강(data augmentation) 관점에서 새로운 Mix-Paste 방법을 통해 복잡한 문제에 접근합니다.

- **Technical Details**: Mix-Paste 방법은 동일한 카테고리 레이블을 가진 여러 개의 아이템 패치를 섞어 원본 패치를 대체하여 새로운 이미지를 생성하는 방식입니다. 이 과정에서 X-ray 이미지의 물체 겹침을 모방하여 탐지 모델이 이미지를 더 잘 이해할 수 있도록 합니다. 또한, 이 모델은 혼합된 패치에서 발생할 수 있는 큰 손실(large losses)을 억제하기 위한 LLS 전략을 설계하여 모델의 학습 효과를 개선합니다.

- **Performance Highlights**: 저자들은 X-ray 데이터셋에서 노이즈가 있는 주석 하에 모델의 우수성을 입증하며, 일반 객체 탐지 작업에서도 성능 개선을 보여줍니다. 더불어, 이 결과들은 노이즈 주석 문제를 해결하기 위한 데이터 증강의 잠재력을 명확히 나타냅니다. 이 연구는 노이즈 주석을 다루기 위한 첫 번째 접근법으로써 특히 주목받고 있습니다.



### Multi-modal classification of forest biodiversity potential from 2D orthophotos and 3D airborne laser scanning point clouds (https://arxiv.org/abs/2501.01728)
- **What's New**: 이 연구는 전통적인 현장 조사 방식의 한계를 극복하기 위해 2D 정사 영상(orthophotos)과 3D 공중 레이저 스캐닝(point clouds) 데이터를 융합하여 숲의 생물 다양성을 평가하는 새로운 방법을 제안합니다. BioVista 데이터셋을 통해 44,378 쌍의 샘플을 수집하였으며, 이는 덴마크의 온대 숲에서 수집된 것입니다. 이 데이터셋은 생물 다양성을 평가할 수 있는 다중 모달 융합 접근 방식을 탐색하는 데 사용됩니다.

- **Technical Details**: 연구에서는 ResNet과 PointVector 네트워크를 사용하여 각각 2D 정사 영상과 3D ALS 포인트 클라우드를 분석하였습니다. 두 가지 융합 방법인 신뢰 기반 앙상블(confidence-based ensemble)과 특성 수준 연결(feature-level concatenation)을 적용했습니다. 특히, 특성 수준 연결 방식을 통해 평균 75.5%의 정확도를 달성하며, 두 데이터 조합의 상호 보완적 이점을 강조했습니다.

- **Performance Highlights**: 이 연구에서는 2D 정사 영상과 3D ALS 포인트 클라우드를 사용하여 숲의 생물 다양성 잠재력을 평가했습니다. 개별 데이터 모달리티에서는 평균 정확도가 각각 69.4% 및 72.8%에 도달했으며, 융합 접근법을 적용했을 때 향상된 정확도를 기록했습니다. 이러한 결과는 생물 다양성 평가에 있어 기존 센서 데이터의 융합이 중요한 역할을 할 수 있음을 보여줍니다.



### IGAF: Incremental Guided Attention Fusion for Depth Super-Resolution (https://arxiv.org/abs/2501.01723)
- **What's New**: 이번 논문에서는 로봇 공학, 내비게이션, 의료 영상 등의 분야에 중요한 깊이 추정 문제를 다룹니다. 기존의 깊이 센서들이 저해상도( Low-Resolution, LR) 깊이 맵을 생성하여 세밀한 장면 인식이 어려운 점을 개선하기 위해, 깊이 초해상도( Super-Resolution) 기술을 제안합니다. 우리는 새로운 센서 융합 방법론인 Incremental guided attention fusion (IGAF) 모듈을 도입하여 RGB 이미지와 LR 깊이 맵을 효과적으로 융합합니다.

- **Technical Details**: IGAF 모듈은 RGB 이미지와 LR 깊이 맵의 특징을 학습하여 고해상도( High-Resolution, HR) 깊이 맵을 생성합니다. 이 접근법은 HR로 구조화된 입력을 통해 LR 깊이 맵을 HR 깊이 맵으로 변환합니다. 이러한 방법론을 통해 우리는 강건한 초해상도 모델을 구축하고 여러 벤치마크 데이터셋을 통해 평가하였습니다.

- **Performance Highlights**: 우리의 모델은 NYU v2 데이터셋에서 $	imes 4$, $	imes 8$, $	imes 16$ 업샘플링에 대해 모든 기준 모델들에 비해 최첨단 결과를 달성했습니다. 또한, Middlebury, Lu, RGB-D-D 데이터셋의 제로샷 설정에서도 모든 기준 모델들을 초월하는 성능을 보였습니다. 관련 코드와 환경 설정, 모델은 GitHub에서 제공됩니다.



### AR4D: Autoregressive 4D Generation from Monocular Videos (https://arxiv.org/abs/2501.01722)
Comments:
          TL;DR: We present a novel method for 4D generation from monocular videos without relying on SDS, delivering greater diversity, improved spatial-temporal consistency, and better alignment with input prompts. Project page: this https URL

- **What's New**: 최근 생성 모델(Generative Models)은 동적 3D 콘텐츠 생성(4D 생성)에 대한 관심을 높였습니다. 기존 방법들은 주로 Score Distillation Sampling(SDS)에 의존해왔지만, 이는 다양성 부족, 공간-시간 불일치 및 입력 프롬프트와의 불일치 문제를 초래합니다. 이러한 문제를 해결하기 위해 AR4D라는 새로운 패러다임을 제안하며, 이는 SDS 없이 4D 자산을 생성할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: AR4D는 세 가지 주요 단계로 구성되어 있습니다. 첫 번째 단계에서는 단안 비디오의 첫 번째 프레임에 대해 전문가 모델을 활용해 3D 표현을 생성하고 이를 정제하여 정규 공간을 만듭니다. 두 번째 단계에서는 반환 비디오의 첫 번째 프레임을 바탕으로 각각의 프레임의 3D 표현을 자율 회귀적(Autoregressive)으로 생성하며, 이 과정에서 로컬 변형 필드를 사용해 정확한 모션과 기하학 추정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 AR4D는 SDS 없이도 최첨단 4D 생성 성능을 달성할 수 있음을 보여주었습니다. 생성된 4D 콘텐츠는 더욱 다양하고 공간-시간 일관성을 향상시켰으며, 입력 프롬프트와의 정렬 성능도 개선되었습니다. 이는 4D 생성의 품질을 한층 높이는 결과를 가져오게 됩니다.



### Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models (https://arxiv.org/abs/2501.01720)
Comments:
          Accepted to AAAI2025

- **What's New**: 이 논문에서는 기존의 이진 분류 방식의 Face Anti-Spoofing (FAS) 방법을 해석 가능한 Visual Question Answering (VQA) 프레임워크로 전환한 Interpretable Face Anti-Spoofing (I-FAS) 모델을 제안하고 있습니다. 이 새로운 접근법은 모델이 결정과 판단을 자연어로 해석할 수 있도록 하여 FAS의 해석 가능성을 증가시킵니다. 또한, Spoof-aware Captioning and Filtering(제목에서 스푸핑에 대한 설명을 제공하는 자막 생성 전략)을 통해 높은 품질의 캡션을 생성하여 모델의 감독을 자연어 해석으로 보강합니다.

- **Technical Details**: 모델의 안정적인 훈련을 지원하기 위해 Lopsided Language Model(L-LM) 손실 함수를 도입하였으며, 이는 판단과 해석의 손실 계산을 분리하여 처리합니다. 이 방법은 판단 개선에 비중을 두어 모델의 수렴 속도와 안정성을 높이는 데 기여합니다. 또한, Globally Aware Connector(GAC)를 통해 다층적인 시각적 표현과 언어 모델을 정렬시킴으로써 모델이 전역적 시각적 특징을 인식하는 데 도움을 줍니다.

- **Performance Highlights**: 기존 방법들과 비교 시, One to Eleven 크로스 도메인 벤치마크에서 자주 최신 모델에 비해 유의미한 향상을 보여주었습니다. 이 벤치마크는 총 12개의 공개된 데이터 세트를 포함하고 있으며, 연구 결과는 I-FAS 방법이 해석 가능성과 강건성을 향상시키고 크로스 도메인 일반화 성능을 크게 개선함을 나타냅니다.



### KeyNode-Driven Geometry Coding for Real-World Scanned Human Dynamic Mesh Compression (https://arxiv.org/abs/2501.01717)
- **What's New**: 본 논문에서는 실제 환경에서 스캔한 3D 인간 동적 메쉬의 압축을 위한 새로운 방법론을 제안합니다. 이 방법은 KeyNode를 활용하여 복잡한 토폴로지 변수와 스캔 결함을 효과적으로 처리합니다. 특히, 임베딩된 키 노드의 변환만을 전송하여 효율적인 압축이 가능함을 보여줍니다. 이 연구는 휴먼 모션의 복잡성을 통합한 변환의 거리 가중 합으로 각 정점의 시간적 움직임을 수식합니다.

- **Technical Details**: 제안된 KeyNode 기반 압축 방법은 서로 다른 키 노드와의 관계를 통해 정점의 움직임을 포착합니다. Octree 기반의 잔여 물체 인코딩(residual coding)과 양방향 예측(Dual-direction prediction) 모드를 통해 예측 정확도를 높입니다. 이러한 방법을 통해 임베딩된 변형을 이용하여 시간이 지남에 따라 동적 메쉬의 위치를 예측하는 과정을 포함합니다. 특히, KeyNode를 통해 정점 변환을 수량화하고, 카우시 분포에 적합한 허프만 딕셔너리를 통해 엔트로피 코딩 최적화를 수행합니다.

- **Performance Highlights**: 실험 결과, 제안한 압축 방법은 기존의 최신 기술들에 비해 평균 비트레이트를 24.51% 절감하며, 특히 저비트레이트 상황에서 두드러진 성능 향상을 보였습니다. 이러한 성과는 복잡한 휴먼 모션을 효과적으로 예측하고 압축하는 데 있어 중요한 의미를 가진다고 할 수 있습니다. 더불어, 다양한 애플리케이션에서 사용될 수 있는 가능성을 제시하며, 향후 연구의 방향성을 제시하고 있습니다.



### Cloth-Splatting: 3D Cloth State Estimation from RGB Supervision (https://arxiv.org/abs/2501.01715)
Comments:
          Accepted at the 8th Conference on Robot Learning (CoRL 2024). Code and videos available at: this http URL

- **What's New**: Cloth-Splatting이라는 새로운 방법을 소개합니다. 이 방법은 RGB 이미지에서 3D 상태를 추정하는 데 필요한 프레임워크를 통해 작동합니다. Cloth-Splatting은 향후 상태를 예측하기 위해 동작 조건 모델(action-conditioned dynamics model)을 활용하고, 예측된 상태를 3D Gaussian Splatting을 사용하여 업데이트합니다.

- **Technical Details**: 이 방법의 핵심은 3D 메쉬 기반 표현과 Gaussian Splatting을 결합하여 섬유 상태 공간과 이미지 공간 간의 미분 가능한 매핑을 정의하는 것입니다. 이를 통해 RGB 감독만을 사용해 부정확한 상태 추정을 개선하는 gradient 기반 최적화 기술을 사용할 수 있습니다. Cloth-Splatting은 Bayesian filtering과 유사한 예측-업데이트 프레임워크를 사용하여 동작 기반 다이나믹 모델로 다음 상태를 예측합니다.

- **Performance Highlights**: 실험 결과, Cloth-Splatting은 2D 및 3D 기준 추적 기술보다 뛰어난 성능을 보여주며, 정확도가 57575757% 향상되고 최상의 기준보다 약 85% 빠른 속도를 기록했습니다. 이 방법은 RGB 관찰만으로 형태가 변형되는 물체를 효율적이고 정확하게 추적할 수 있다는 것을 시사합니다.



### Enhancing Large Vision Model in Street Scene Semantic Understanding through Leveraging Posterior Optimization Trajectory (https://arxiv.org/abs/2501.01710)
Comments:
          7 pages

- **What's New**: 이 논문에서는 자율주행 (AD) 모델의 일반화를 향상시키기 위해 사전 훈련된 Large Vision Models (LVMs)를 이용한 새로운 접근 방식을 제안합니다. AD 모델이 수집된 데이터를 기반으로 시간이 지남에 따라 모델을 업데이트할 수 있도록 하여, 데이터 증가에 따른 언더피팅 문제를 해결합니다. 이 방법은 LVM의 강력한 적합 능력 덕분에 특히 효과적이며, 다양한 훈련 데이터를 통해 모델의 일반화 능력을 크게 향상시킵니다.

- **Technical Details**: LVM을 백본으로 사용하여, 차량의 온보드 데이터셋에 기반한 다운스트림 인식 헤드를 훈련하는 방식을 제안합니다. 이는 LVM이 이미지의 복잡한 패턴을 추출할 수 있도록 도와줍니다. 또한, Posterior Optimization Trajectory (POT)-Guided 최적화 스킴을 도입하여 인식 헤드의 훈련 속도를 가속화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 모델 대비 성능을 66.48% 향상시키고, 수렴 속도는 6배 이상 빨라짐을 입증하는 광범위한 실험 결과를 보여줍니다. 이러한 성능 향상은 AD 차량에 필수적인 계산 효율성을 제공하며, 신속한 의사 결정을 위한 핵심 요소가 됩니다.



### MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders (https://arxiv.org/abs/2501.01709)
Comments:
          11 pages, 5 figures

- **What's New**: 본 논문은 복수의 비전 인코더의 독특한 능력을 하나의 효율적인 인코더 모델로 증류하는 새로운 프레임워크인 Mixture-of-Visual-Encoder Knowledge Distillation(MoVE-KD)을 제안합니다. MoVE-KD는 Low-Rank Adaptation(LoRA)과 Mixture-of-Experts(MoEs)를 활용하여 각 인코더 고유 특성을 유지하면서 선택적으로 지식을 활성화합니다. 또한, Attention 기반의 증류 전략을 사용하여 비전 인코더의 중요도를 고려하고, 시각적 토큰의 가치를 강조합니다.

- **Technical Details**: MoVE-KD는 여러 개의 Teacher 인코더로부터 지식을 효과적으로 전이하며, 이를 통해 단일 인코더의 효율성을 개선합니다. encoder adapters를 통해 다양한 Teacher 인코더의 출력 결과를 통합된 표현 공간으로 투영하여, 비전 토큰 간의 일치를 이루는 것이 핵심입니다. Mixture-of-LoRA-experts(MoLE) 구조를 통해 학습 과정에서 발생할 수 있는 갈등을 완화하며, 최종 목표는 텍스트 손실과 KD 손실을 최소화하는 것입니다.

- **Performance Highlights**: LLaVA 및 LLaVA-NeXT와 같은 유력한 VLM에서의 광범위한 실험 결과, MoVE-KD 방법이 기존 방법들에 비해 성능 및 효율성 면에서 상당한 향상을 보여주었습니다. 본 연구의 기여는 비전-언어 모델의 다중 비전 인코더 융합을 위한 MoVE-KD 프레임워크를 제안한 점과 중요 비주얼 토큰의 증류를 향상시키기 위한 Attention-guided KD 정규화를 도입한 점입니다. 이와 같은 접근은 다양한 인코더 간의 지식을 통합하여 향상된 성능을 달성하게 합니다.



### Aesthetic Matters in Music Perception for Image Stylization: A Emotion-driven Music-to-Visual Manipulation (https://arxiv.org/abs/2501.01700)
- **What's New**: 이번 논문에서는 EmoMV라는 감정 기반 음악 및 비주얼 조작 방법을 제안합니다. 이 방법은 음악의 감정을 바탕으로 이미지를 조작하여 감정과 시각적 요소를 통합합니다. 특히, EmoMV는 음악의 구성 요소(예: pitch 및 rhythm)를 바탕으로 감정을 추출하고 이를 시각적 측면(예: 색상 및 조명)에 적용하는 방식으로 작동합니다. 이를 통해 감정의 시각적 변환을 효과적으로 진행할 수 있다는 점에서 혁신적입니다.

- **Technical Details**: EmoMV는 두 단계로 구성된 프레임워크로, Mus-Vis Textual Alignment 및 Emotion-aware Aesthetic Image Refinement를 포함합니다. Mus-Vis Textual Alignment에서는 음악 요소를 감정 표현과 연관시키고, Emotion-aware Aesthetic Image Refinement에서는 감정 정보를 이미지의 미적 속성으로 외부화하여 이미지의 감정적 깊이를 피드백합니다. 학습에는 HT-SAT 오디오 인코더와 BART 디코더를 사용하여 음악 요소 설명을 생성하고, LLaVA 모델을 활용해 감정, 내러티브, 음악, 관객 반응을 포함한 네 차원 설명을 생성합니다.

- **Performance Highlights**: EmoMV는 38,000개의 음악-이미지 쌍 데이터셋을 사용하여 다각적인 성능 평가를 실시하였고, 감정 인식 및 이미지 품질 모두에서 긍정적인 결과를 나타냈습니다. EEG 측정치를 통해 실시간 감정 반응을 캡처함으로써 감정적 콘텐츠 캡처의 효과를 더욱 향상시켰습니다. 이러한 결과는 EmoMV가 예술 치료와 같은 창의적 산업에서 감정적 웰빙에 긍정적인 영향을 미칠 수 있음을 입증합니다.



### Robust Self-Paced Hashing for Cross-Modal Retrieval with Noisy Labels (https://arxiv.org/abs/2501.01699)
Comments:
          9 pages, AAAI 25 conference

- **What's New**: 본 논문은 Robust Self-paced Hashing with Noisy Labels (RSHNL)이라는 새로운 인지 기반 교차 모드 검색 방법을 제안합니다. 이 방법은 인간의 인지 과정을 모방하여 각 인스턴스를 쉬운 것에서 어려운 것으로 학습하며, 노이즈 라벨에 대한 강건성을 수용합니다. RSHNL은 노이즈 라벨을 식별하면서 점진적으로 해시 코드를 학습하는 방식으로, 현재의 Self-paced Learning (SPL) 패러다임을 활용합니다.

- **Technical Details**: RSHNL은 Contrastive Hashing Learning (CHL) 스킴을 통해 다중 모드 일관성을 극대화하고, Center Aggregation Learning (CAL)를 통해 intra-class 변동성을 완화합니다. 더불어, Noise-tolerance Self-paced Hashing (NSH)은 각 인스턴스의 학습 난이도를 동적으로 측정하고, 노이즈 라벨을 분별하는 역할을 합니다. 이 과정에서 Easy-to-Hard 학습을 통해 전체 깨끗한 샘플 페어로부터 해시 코드를 점진적으로 학습합니다.

- **Performance Highlights**: 제안된 RSHNL은 기존의 최첨단 CMH 방법들에 비해 놀라운 성능을 발휘함을 입증하는 광범위한 실험을 수행했습니다. 특히, 다양한 노이즈 비율에서 상대적인 성능이 크게 향상된 것으로 나타났습니다. RSHNL은 노이즈 라벨의 부정적 영향을 완화시키면서 효과적인 다중 모드 검색을 가능하게 합니다.



### CrossView-GS: Cross-view Gaussian Splatting For Large-scale Scene Reconstruction (https://arxiv.org/abs/2501.01695)
- **What's New**: 이 논문에서는 대규모 장면 재구성을 위한 새로운 cross-view Gaussian Splatting 기법을 제안합니다. 기존의 3D Gaussian Splatting(3DGS) 방법은 작은 시점 변화에서는 효과적이나, 큰 시점 변화가 있는 장면에서는 최적화의 어려움을 겪습니다. 이 방법은 두 개의 독립적인 브랜치를 사용하여 공중 및 지상 뷰로부터 모델을 재구성함으로써 cross-view 재구성을 위한 신뢰할 수 있는 선행 정보를 제공합니다.

- **Technical Details**: 제안된 방법은 초기화 및 밀집화 과정에서 cross-view 재구성을 위해 공중 및 지상 뷰로부터 독립적으로 재구성된 두 가지 모델을 사용하는 dual-branch 구조입니다. 또한, significant view disparity에 의해 발생하는 smoothing 문제를 완화하기 위해 gradient-aware regularization 전략이 도입됩니다. 이 외에도 Gaussian supplementation 전략을 통해 두 브랜치의 보완 정보를 cross-view 모델에 포함시키는 방법이 적용됩니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 실시한 실험 결과, 제안된 방법은 최신 기법들과 비교했을 때 새로운 뷰 합성에서 뛰어난 성능을 나타냈습니다. 특히, 대규모 장면 재구성을 위해 cross-view 데이터의 최적화 문제를 해결하고, 이 과정에서 다양한 시점의 정보를 효과적으로 활용하였음을 입증하였습니다.



### VidFormer: A novel end-to-end framework fused by 3DCNN and Transformer for Video-based Remote Physiological Measuremen (https://arxiv.org/abs/2501.01691)
- **What's New**: 이번 논문에서는 얼굴 비디오를 기반으로 한 원거리 생리 신호 측정을 위한 새로운 방식인 VidFormer를 소개합니다. VidFormer는 3차원 합성곱 신경망(3DCNN)과 Transformer 모델을 통합하여 기존의 깊이 있는 학습 방법들이 갖고 있던 데이터셋의 크기 간 조화로운 성능 부족 문제를 해결하고자 합니다. 이 프레임워크는 모듈 간의 정보 교환과 융합을 쉽게 할 수 있도록 설계되어 있습니다.

- **Technical Details**: VidFormer는 얼굴 비디오로부터 지역적(local) 및 전역적(global) 특징을 추출하기 위해 3DCNN과 Transformer를 각각 활용합니다. 또한, VidFormer는 시공간적(spatiotemporal) 주의 메커니즘을 채택하여 데이터의 시공간적 특징을 더욱 잘 포착할 수 있도록 개선되었습니다. 이 모델은 Stem, Local Convolution Branch, Global Transformer Branch, CNN and Transformer Interaction Module(CTIM), 그리고 rPPG 생성 모듈(RGM)으로 구성됩니다.

- **Performance Highlights**: 우리의 실험 결과, VidFormer는 UBFC-rPPG, PURE, DEAP, ECG-fitness, COHFACE와 같은 다수의 표준 데이터셋에서 현재 최첨단(SOTA) 방법들을 초월하는 성능을 보였습니다. 특히, DA-3DCNN과 ST-MHSA와 같은 새로운 글로벌 주의 메커니즘을 도입하여 모델이 비디오 데이터의다양한 차원적 특성을 효과적으로 포착할 수 있도록 하였습니다. 예를 들어, 이 프레임워크는 심박수(HR), 심박변동성(HRV), 호흡수(RR) 등을 정밀하게 측정하기 위한 기반을 제공했습니다.



### Quantitative Gait Analysis from Single RGB Videos Using a Dual-Input Transformer-Based Network (https://arxiv.org/abs/2501.01689)
Comments:
          Accepted for presentation at The IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 논문은 단일 RGB 비디오로부터 임상 보행 분석을 위한 효율적인 방법을 제안합니다. 다중 패턴 입력을 사용하는 컨볼루션 Transformer 네트워크를 도입하여 전통적인 모션 캡처 시스템의 비용과 복잡성을 줄이며, 임상 환경에서의 적용 가능성을 높이고 있습니다. 특히, 이 시스템은 보행 편차 지수(GDI), 무릎 굴곡각, 보폭, 보행 회전수와 같은 주요 지표를 정확하게 추정하는 신규 방법론을 제공합니다.

- **Technical Details**: 제안된 방법은 입력되는 비디오의 모든 프레임을 연속적인 운동 데이터(X∈ℝT×N×2)로 구성하고, 각 프레임에서의 인간의 자세를 포착합니다. 이 과정을 통해 특정 보행 메트릭을 예측하는 비선형 매핑(fθ:ℝT×N×2→ℝ1)을 학습하게 됩니다. 자기 주목 메커니즘을 활용하여 시간에 따른 주요 해부학적 표적의 관계에 집중할 수 있어, 보행 관련 파라미터의 예측 정확성이 향상됩니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 보행에 관련된 주요 파라미터를 예측하는 데 있어 현재 최첨단 기법을 초월하는 성과를 보였습니다. 적은 자원을 사용하여 훈련이 가능하며, 다양한 임상 환경에서 적용하기에 적합한 것으로 판단됩니다. 이 모델은 더욱 개선된 성과를 제공하고, 기존의 복잡하고 비용 높은 시스템보다 간편한 방법으로 기능성을 보여주고 있습니다.



### IAM: Enhancing RGB-D Instance Segmentation with New Benchmarks (https://arxiv.org/abs/2501.01685)
- **What's New**: 이 논문에서는 RGB-D 인스턴스 세분화(instance segmentation)의 주요 결핍을 해결하기 위해 세 가지 새로운 벤치마크를 도입하고 있습니다. 기존의 데이터 세트들은 주로 RGB 또는 RGB-D 의미론적(segmentation) 세분화에 초점이 맞춰져 있었으며, 개별 인스턴스를 구분하기에는 부족했습니다. 따라서 RGB-D 기반의 세밀한 인스턴스 분리를 지원하는 데이터 세트가 필요합니다.

- **Technical Details**: 이 연구에서는 NYUDv2와 SUN-RGBD라는 두 가지 잘 알려진 RGB-D 데이터 세트를 세밀한 인스턴스 분할 작업을 위해 재주석하고 재구성하여 새로운 RGB-D 벤치마크를 구축하였습니다. 또한, 인간-로봇 상호작용과 같은 특정 응용 프로그램을 지원하기 위해 RGB-D Box 데이터 세트를 개발했습니다. 이러한 데이터 세트는 각각 철저한 문서화와 통계 분석을 포함하여 신뢰할 수 있는 자원으로 활용될 수 있도록 하고 있습니다.

- **Performance Highlights**: IAM(Intra-modal Attention Mix) 모듈을 통해 RGB-D 인스턴스 세분화의 성능이 현저히 향상되었습니다. 실험 결과, IAM은 기존의 융합 방법들에 비해 경계를 더욱 정확하게 구분하고 인스턴스를 효과적으로 분리하는 데 성공했습니다. NYUDv2-IS에서는 IAM이 intra-modal attention에 비해 2.8%, early fusion에 비해 6.7%의 성능 향상을 보였으며, 이는 RGB-D 인스턴스 세분화 연구의 강력한 기준점으로 자리 잡을 것입니다.



### PG-SAG: Parallel Gaussian Splatting for Fine-Grained Large-Scale Urban Buildings Reconstruction via Semantic-Aware Grouping (https://arxiv.org/abs/2501.01677)
- **What's New**: 최근 3D Gaussian Splatting(3DGS) 기반의 새로운 병렬 Gaussian splatting 방식인 PG-SAG가 제안되었습니다. 이 방법은 대규모 도시 지역의 건물 표면 재구성을 정밀하게 수행할 수 있도록 시맨틱 큐(semantic cues)를 활용하여, 이미지 해상도를 저하시키지 않고 최적화를 가능하게 합니다. 특히, 새로운 경계 인식(normal loss) 및 기울기 제약(gradient-constrained balance-load loss) 손실 함수가 도입되어 효율성을 더욱 높이고 있습니다.

- **Technical Details**: PG-SAG는 대규모 장면을 효과적으로 분할하기 위해 Language Segment Anything(LSA)를 활용하여 건물 마스크를 생성하고, 이를 기반으로 여러 카메라 간 가시성을 고려하여 서브 그룹으로 최대로 효율적으로 최적화합니다. 이때, 각 서브 그룹별로 독립적으로 최적화가 가능하고, 마스크된 픽셀을 사용하여 원본 고해상도 이미지를 직접 수용할 수 있습니다. 이러한 접근은 대규모 장면의 복잡성을 고려하고, 최적화 시 픽셀 병렬 렌더링 단계에서의 스레드 대기 시간을 최소화합니다.

- **Performance Highlights**: 실험 결과, PG-SAG는 다양한 도시 데이터 세트에 대해 기존의 3DGS 기반 방법들보다 우수한 성능을 보였습니다. 특히, 건물 표면 재구성에서의 성과가 두드러졌으며, 대규모 도시 장면에서의 복잡성을 효과적으로 처리할 수 있는 가능성을 보여줍니다. 이러한 발전은 대규모 도시 환경에서의 세밀한 건물 재구성을 위한 새로운 기준을 제시합니다.



### EAUWSeg: Eliminating annotation uncertainty in weakly-supervised medical image segmentation (https://arxiv.org/abs/2501.01658)
- **What's New**: 이번 연구에서는 약한 주석(annotation) 방법인 Bounded Polygon Annotation(BPAnno)와 이를 활용한 학습 프레임워크 EAUWSeg를 제안합니다. 이 방법은 레지온을 두 개의 다각형으로 간단하게 레이블링함으로써 주석의 불확실성을 제거하고 더 안정적인 모델 훈련을 지원합니다. 이를 통해 기존 약한 주석 기반 방법들과 비교하여 성능이 뛰어나며 적은 주석 작업량으로도 우수한 결과를 제공합니다.

- **Technical Details**: EAUWSeg는 레이블이 부정확한 영역에서 더욱 신뢰할 수 있는 감독 신호를 제공하기 위해 상반된 다각형을 두 개의 별도 주석으로 취급하는 특별한 학습 메커니즘을 갖추고 있습니다. 이를 통해 적절한 픽셀의 카테고리 예측을 보장하면서 불확실한 픽셀들도 같은 카테고리로 유도하여 특징 표현의 일관성을 유지합니다. 이러한 방법은 주석의 불확실성을 줄이고, 모델 학습의 불안정함을 제거합니다.

- **Performance Highlights**: 실험 결과, EAUWSeg는 ISIC2017 및 Kvasir-SEG와 같은 기존의 약한 주석 분할 방법들보다 우수한 성능을 보이며, 완전 감독 방식과 비교했을 때 주석 작업량을 20% 미만으로 줄이는 성과를 달성했습니다. 이로 인해 비용 효율적인 의료 이미지 분할 솔루션으로서의 가능성을 보여주며, 다양한 의료 이미지 분할 모델에 적용 가능함을 입증했습니다.



### Dual Mutual Learning Network with Global-local Awareness for RGB-D Salient Object Detection (https://arxiv.org/abs/2501.01648)
- **What's New**: RGB-D salient object detection (SOD)는 RGB 및 깊이 정보(Depth Information)를 통합하여 장면의 주요 영역을 강조하는 어려운 작업입니다. 본 연구에서는 GL-DMNet이라는 새로운 이중 상호 학습 네트워크를 제안하여 이러한 문제를 해결하고, 글로벌 및 로컬 인식(Global-Local Awareness)을 통해 이질적인 정보 간의 관계를 개선합니다. 이를 통해 더 나은 성능을 달성하고 24개 기존 RGB-D SOD 방법에 비해 평균적으로 약 3% 향상된 결과를 보여줍니다.

- **Technical Details**: GL-DMNet은 위치 상호 융합(Position Mutual Fusion) 및 채널 상호 융합(Channel Mutual Fusion) 모듈을 통해 서로 다른 모드 간의 의존성을 탐색합니다. 새로운 효율적인 디코더는 다단계 융합 기능을 통합하여 최종 SOD 예측의 정확성을 높입니다. 이 모델은 크로스 모달러티 학습(Cross-modality Learning)에서 더 나은 성능을 발휘하기 위해 이중 주의 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: 본 연구에서 제안된 GL-DMNet은 6개의 벤치마크 데이터셋에서 수행된 실험을 통해 우수한 성능을 입증했습니다. 특히, 기존의 24개 RGB-D SOD 방법들과 비교하여 뛰어난 성능을 달성하였으며, 각 스테이지별로 차별화된 특징을 탐색하여 더 나은 학습 효율을 확보했습니다. 이 모델은 복잡한 환경에서도 더욱 효과적으로 작동할 수 있는 잠재력을 보여주었습니다.



### HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding (https://arxiv.org/abs/2501.01645)
- **What's New**: 이 논문에서는 HLV-1K라는 새로운 대규모 비디오 벤치마크를 제시합니다. HLV-1K는 1,009개의 시간당 비디오로 구성되어 있으며, 다양한 고품질 QA(질문 응답) 및 MCQA(다중 선택 질문 응답) 쌍을 포함하고 있습니다. 이 벤치마크는 긴 비디오 이해 모델을 평가하기 위해 설계되었습니다.

- **Technical Details**: HLV-1K는 100,000개 이상의 프레임을 포함한 비디오의 복잡성을 해결하기 위해 시간 인식 쿼리와 다양한 주석을 포함하여 비디오의 미세한 관계를 명확하게 탐구합니다. 수집 단계에서 HD-VILA 데이터셋을 기반으로 하여 YouTube에서 1,500개 이상의 긴 비디오를 다운로드하고, 수동으로 저품질 비디오를 필터링했습니다. 이를 통해 엔터테인먼트, 영화, 여행 등 다양한 주제를 다루는 약 1,009개의 긴 비디오를 엄선했습니다.

- **Performance Highlights**: 현재의 최첨단 방법을 이용하여 HLV-1K를 평가한 결과, 다양한 수준과 여러 과제에서 깊은 긴 비디오 이해 능력을 테스트하는 데 유용하다는 점을 보여주었습니다. HLV-1K는 향후 긴 비디오 이해 작업을 촉진하고, 실시간 비디오, 회의 녹화, 영화에 대한 심층적인 이해를 가능하게 하는 모델의 발전을 이끌 것으로 기대됩니다.



### iCBIR-Sli: Interpretable Content-Based Image Retrieval with 2D Slice Embeddings (https://arxiv.org/abs/2501.01642)
Comments:
          8 pages, 2 figures. Accepted at the SPIE Medical Imaging

- **What's New**: 본 연구에서는 뇌 MR 이미지를 위한 새로운 해석 가능한 콘텐츠 기반 이미지 검색 시스템인 iCBIR-Sli를 제안합니다. 이 시스템은 2D 슬라이스의 연속성을 활용하여 뇌의 구조적 정보를 전체적으로 보존하면서도 우수한 검색 성능을 보여줍니다. iCBIR-Sli는 낮은 차원의 표현을 효율적으로 집계하여 CBIR 시스템의 필수적인 특성을 충족시키고 있습니다.

- **Technical Details**: iCBIR-Sli는 2D 슬라이스 임베딩 기법을 사용해 뇌 MR 이미지의 정보를 집계하고, 이를 통해 높은 완전성, 사용성, 견고성 및 해석성을 갖춘 낮은 차원 표현을 생성합니다. 이 과정에서 5개의 공개된 뇌 MR 데이터셋을 사용하여 알츠하이머병 및 인지적으로 정상인에 대한 검색 평가 실험을 수행합니다.

- **Performance Highlights**: iCBIR-Sli는 매크로 F1 0.859의 top-1 검색 성능을 보여, 분류를 위해 명시적으로 설계된 기존의 심층 학습 모델과 동등한 성능을 발휘합니다. 또한, 이 방법은 검색된 질병에 대한 뇌 영역을 명확히 식별함으로써 높은 해석성을 제공합니다.



### Uncertainty and Energy based Loss Guided Semi-Supervised Semantic Segmentation (https://arxiv.org/abs/2501.01640)
Comments:
          Accepted in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 연구에서는 반지도 학습(Semi-supervised, SS)에서 픽셀 수준의 주석 문제를 극복하기 위해 라벨이 있는 이미지와 라벨이 없는 이미지를 동시에 활용하는 방법을 제안합니다. 핵심적으로, aleatoric 불확실성(aleatoric uncertainty)과 에너지 기반 모델링(energy-based modeling)을 사용하여 교차-합집합(pseudo intersection-union) 수퍼바이즈드 학습을 수행합니다. 이 과정에서는 네트워크의 예측 분기 두 개에서 발생하는 데이터의 고유한 노이즈 변동을 모델링합니다.

- **Technical Details**: 알고리즘 1에서 제시된 바와 같이, 이 세미 수퍼바이즈드 학습 모델을 학습하기 위한 알고리즘의 포괄적인 설명이 제공됩니다. 손실 함수는 메인 논문에서 상세히 논의되며, 에너지 기반 손실은 하위 세그멘테이션 작업의 생성 모델링 가능성을 실현합니다. 논문은 pseudo-intersection 라벨, pseudo-union 라벨 및 ground-truth와 함께 aleatoric 및 에너지 기반 손실을 적용합니다.

- **Performance Highlights**: 윗 섹션에서 제시된 결과들은 CPCL과 같은 최첨단 방법들과 비교하여 성능 개선을 보여줍니다. Cityscapes 데이터세트를 사용한 시각화 결과에서는 CPCL 방법이 특정 클래스에 대해 잘못된 긍정(양성)을 보이는 반면, 제안한 방법은 ground truth와의 일치를 더욱 정확하게 유지하고 있음을 확인할 수 있습니다. 통계적 유의성 분석에서는 DUEB 방법이 CPCL 및 감독 학습(supervised) 방법과 유의미한 차이를 보이는 것으로 나타났습니다.



### ACE: Anti-Editing Concept Erasure in Text-to-Image Models (https://arxiv.org/abs/2501.01633)
Comments:
          25 pages, code available at this https URL

- **What's New**: 최근 텍스트-이미지(T2I) 확산 모델이 고품질 이미지 생성을 크게 개선하고 있으나, 저작권 이미지와 같은 유해 콘텐츠의 불법 생성에 대한 우려가 커지고 있습니다. 기존의 개념 삭제 방법은 프롬프트로부터 삭제된 개념을 잘 방지하지만, 불필요한 편집을 예방하는 데는 취약합니다. 이에 대한 대안으로, Anti-Editing Concept Erasure(ACE) 방법을 제안하며, 생성 및 편집 과정에서도 목표 개념을 효율적으로 삭제합니다.

- **Technical Details**: ACE 방법은 조건부 및 비조건부 노이즈 예측 모두에 삭제 지침을 주입하여, 편집 및 생성 과정에서 유해한 개념의 생성을 사전에 방지합니다. 또한, 무작위 보정 안내를 도입하여 관련 없는 개념의 침식을 방지하고, 목표 개념 삭제 시 비목표 개념의 생성을 보존합니다. Stable Diffusion 1.4를 텍스트-이미지 모델로 사용하며, 이 모델의 변량 오토인코더(VAE)와 텍스트 조건부 확산 모델을 결합합니다.

- **Performance Highlights**: ACE 방법은 지적 재산권(IP), 노골적인 콘텐츠, 예술적 스타일 등 다양한 삭제 작업에 대해 우수한 성능을 보였습니다. 실험 결과, ACE는 장기 소실 개념 및 예술 스타일 삭제에서 기존의 최첨단 방법보다 우수한 필터링 능력을 보여주었습니다. 이러한 연구 결과는 ACE 방법이 안전하지 않은 콘텐츠 생성을 효율적으로 방지할 수 있는 잠재력을 가지고 있음을 강조합니다.



### Merging Context Clustering with Visual State Space Models for Medical Image Segmentation (https://arxiv.org/abs/2501.01618)
Comments:
          Our paper has been accepted by the IEEE Transactions on Medical Imaging. Our code can be found at this https URL

- **What's New**: 이 논문에서는 기존의 ViM(vision mamba) 모델의 한계를 극복하기 위해 Context Clustering ViM(CCViM)이라는 새롭고 효과적인 방법을 제안합니다. CCViM은 이미지 토큰을 서로 다른 클러스터로 구분하는 컨텍스트 클러스터링 모듈을 포함하여, 전역 및 지역(feature interactions) 피처 간의 상호작용을 동시에 극대화해 의학 이미지 세분화에서의 성능을 개선합니다. 이 방법은 Kumar, CPM17, ISIC17, ISIC18 및 Synapse와 같은 다양한 공공 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수한 성능을 보입니다.

- **Technical Details**: CCViM은 U자형 아키텍처를 채택하여, 고정 스캔 방식의 한계를 극복하고 다이나믹하게 지역의 시각적 컨텍스트를 캡처할 수 있도록 설계되었습니다. CCS6(layer)라는 새로운 기능을 도입해 전역 스캔 방향과 CC(layer)를 결합하여 피처 표현 능력을 향상시킵니다. 이로 인해 필드에서의 실제 수행 능력이 크게 향상되며, 각기 다른 지역 특성을 동적으로 잘 잡아낼 수 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 CCViM이 MedISeg의 여러 태스크에서 효율적이고 효과적으로 뛰어난 성능을 발휘함을 입증했습니다. 특히, 세포 세분화, 피부 병변 세분화 및 다기관 세분화와 같은 의료 분야에서 기존의 CNN 및 ViT 기반 모델들과 비교하여 더욱 우수한 성과를 보였습니다. 이러한 결과는 의료 영상의 자동화 및 정확성을 크게 향상시킬 것으로 기대됩니다.



### Google is all you need: Semi-Supervised Transfer Learning Strategy For Light Multimodal Multi-Task Classification Mod (https://arxiv.org/abs/2501.01611)
- **What's New**: 이번 연구에서는 디지털 이미지 데이터가 증가함에 따라 이미지 분류의 중요성이 커지고 있다는 점을 다룬다. 우리는 하나의 이미지에 대해 여러 개의 라벨을 동시에 할당할 수 있는 강력한 다중 라벨 분류 시스템을 제안한다. 이 시스템은 Convolutional Neural Networks (CNN)와 Natural Language Processing (NLP) 모델을 통합한 다중 모달 분류기를 포함하여, 이미지와 텍스트 정보를 결합해 더 높은 정확도를 추구한다.

- **Technical Details**: 제안된 모델은 이미지와 텍스트의 기능을 각각 추출하고 이를 효과적으로 융합하여 분류 작업에 활용하는 과정을 포함한다. 비전 모듈은 CNN을 통해 이미지에서 특징을 추출하고, NLP 모듈은 텍스트의 구조적 및 의미적 nuance를 mining하여 관련된 텍스트 기능을 생성한다. 이 두 모듈의 출력을 통합하는 기능 융합 모듈은 정보의 흐름을 개선하여 분류 성능을 높이는 중요한 역할을 한다.

- **Performance Highlights**: 초기 결과는 제안된 분류기가 높은 정확도와 효율성을 보임을 나타낸다. 이 모델은 자동 이미지 라벨링 시스템으로서의 잠재력이 크며, 특히 의료 영상과 같이 다중 라벨이 필요한 분야에서 실질적으로 유용하게 사용될 수 있다. 다중 라벨 분류는 이미지에 여러 개체가 포함될 수 있는 현실적인 문제를 해결하며, 모델이 라벨 간의 관계를 인식할 수 있게 해준다.



### Few-shot Implicit Function Generation via Equivarianc (https://arxiv.org/abs/2501.01601)
Comments:
          11 pages, 8 figures, 4 tables

- **What's New**: 이 논문에서는 Few-shot Implicit Function Generation이라는 새로운 문제 설정을 제안하며, 제한된 몇 가지 예시를 통해 다양한 기능적으로 일관된 Implicit Neural Representation (INR) 가중치를 생성하는 방법을 탐구합니다. EquiGen이라는 프레임워크를 통해 제한된 데이터에서도 기능적 유사성을 유지하면서 새로운 INR을 생성할 수 있는 방안을 제시합니다. 이 접근법은 불완전한 데이터 환경에서 효과적이며, 높은 품질의 생성 결과를 도출할 수 있습니다.

- **Technical Details**: EquiGen 프레임워크는 가중치를 중요하게 다루며, 세 가지 주요 단계로 구성됩니다: 1) Equivariant Encoder, 이는 대조 학습(contrastive learning)과 부드러운 증강(smooth augmentation)을 통해 가중치를 equivariant latent space로 프로젝션합니다. 2) Equivariance-Guided Diffusion, 이는 이전의 equivariant 특성(conditioned features)에 기초하여 현재 제너레이션 단계를 규제하는 프로세스입니다. 3) Controlled Perturbations for Diversity, 이는 고유의 여유 공간에서 균형 잡힌 변 perturbation을 적용하여 다양한 기능적 일관성을 갖춘 가중치를 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, EquiGen은 2D 이미지 및 3D 형상 INRs 데이터셋에서 다양성과 기능적 특성을 모두 성공적으로 유지하며 고품질의 가중치 생성을 달성했습니다. 이를 통해 한정된 샘플로도 확장성과 유연성을 갖춘 데이터를 생성하는 능력을 입증하였습니다. Ablation 연구는 각 모듈의 효과를 지지하는 결과를 보여주며, equivariance 개념의 중요성을 실증적으로 입증합니다.



### Adaptive Homophily Clustering: A Structure Homophily Graph Learning with Adaptive Filter for Hyperspectral Imag (https://arxiv.org/abs/2501.01595)
Comments:
          14 pages, 85 figure

- **What's New**: 이 논문에서는 HSI(하이퍼스펙트럼 이미지) 클러스터링을 위한 새로운 방법인 AHSGC(Adaptive Homophily Structure Graph Clustering)가 제안되었습니다. 기존의 그래프 클러스터링 방법들이 가지고 있는 제한점들을 극복하기 위해, 동질적 지역 생성(homogeneous region generation)을 통해 원래 그래프를 구축하는 과정을 포함합니다. 이를 통해 공간 구조 정보 활용도를 높이고, 그래프의 업데이트 능력을 개선하고자 하였습니다.

- **Technical Details**: AHSGC에서는 적응형 필터 그래프 인코더(adaptive filter graph encoder)를 설계하여 그래프에서 고주파(high frequency) 및 저주파(low frequency) 특성을 잡아내는 데 중점을 둡니다. 이후 KL Divergence를 이용한 그래프 임베딩 클러스터링 자기 훈련 디코더가 개발되어 네트워크 학습을 위한 의사 라벨(pseudo-label) 생성에 활용됩니다. 또한, 클러스터링 작업에 맞게 그래프를 업데이트하기 위해 동질성 향상 구조 학습(homophily-enhanced structure learning)이 도입되었습니다.

- **Performance Highlights**: 다양한 실험과 반복적인 비교 분석을 통해 AHSGC는 높은 클러스터링 정확도와 낮은 계산 복잡도, 강력한 강건성을 보유하고 있음이 입증되었습니다. 최종적으로, K-means를 활용하여 잠재적 특성을 표현하며, 제안된 방법의 코드 소스는 제공될 예정입니다.



### D$^3$-Human: Dynamic Disentangled Digital Human from Monocular Video (https://arxiv.org/abs/2501.01589)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 D$^3$-Human이라는 새로운 방법을 제안하여 단일 영상(monocular video)으로부터 동적으로 분리된 디지털 인간 지오메트리(reconstruction)를 재구성합니다. 기존의 연구는 주로 옷이 입혀진 인간 신체를 분리없이 재구성하거나 옷만을 재구성하는 데 집중하여, 애니메이션 제작과 같은 주요 응용 분야에 직접 적용하기 어려웠습니다. 본 연구는 육안으로 보이는 영역과 보이지 않는 영역의 세부사항과 가능성(plausibility)을 보장하여 옷과 신체를 효율적으로 분리하는 방법을 제시합니다.

- **Technical Details**: D$^3$-Human 방법은 명시적(explicit) 및 암시적(implicit) 표현을 결합하여, 분리된 옷을 입은 인간 신체를 모델링하는 방식으로 진행됩니다. 육안으로 보이는 영역은 Signed Distance Function(SDF) 방식으로 재구성하고, 새로운 인간 매니폴드 서명 거리 필드(hmSDF)를 도입하여 가시적인 옷과 신체를 분리해 냅니다. 이 과정에서 SMPL 모델을 이용해 보이지 않는 영역의 신체가 자연스럽게 통합되도록 보장하며, 쉽게 얻을 수 있는 2D 영상 파싱(segmentation)만으로도 작업할 수 있음을 보여줍니다.

- **Performance Highlights**: D$^3$-Human은 기존의 재구성 방식보다 약 20분 만에 의류와 신체의 분리 템플릿을 재구성할 수 있으며, 전체 시간 안에 순차적으로 작업을 마칠 수 있습니다. 애니메이션 제작 및 의류 전송 등 다양한 응용 분야에 적용할 수 있는 가능성을 제시하며, 고품질의 분리된 재구성을 성공적으로 달성하였다고 보고합니다. 실험 결과, 제안된 방법은 기존 방식에 비해 경쟁력 있는 재구성 정확도를 자랑합니다.



### Click-Calib: A Robust Extrinsic Calibration Method for Surround-View Systems (https://arxiv.org/abs/2501.01557)
- **What's New**: 이번 연구에서는 Click-Calib이라는 오프라인 Surround-View System (SVS) 외부 캘리브레이션 방법을 제안합니다. 이 방법은 특별한 설정이 필요 없이 사용자가 자연 장면에서 몇몇 키포인트를 클릭하는 것만으로 캘리브레이션을 수행할 수 있습니다. 기존의 패턴 기반 방법과는 달리 Click-Calib은 원거리에서의 정확한 캘리브레이션을 지원하여 10미터 이상의 거리에서도 높은 정확도를 유지합니다.

- **Technical Details**: Click-Calib는 차량이 정지해 있을 때나 저속(30km/h 미만)으로 주행 중일 때 적용할 수 있는 방법입니다. 기존의 fisheye 카메라 이미지에서 직접 캘리브레이션 매개변수를 최적화하여 정보 손실을 피할 수 있습니다. 또한 새로운 평가 지표인 Mean Distance Error (MDE)를 도입하여 대형 BEV 이미지의 품질을 보다 정확하게 반영합니다.

- **Performance Highlights**: Click-Calib은 세 대의 서로 다른 차량에서 평가되었으며, 기존의 오프라인 캘리브레이션 방법에 비해 특히 먼 거리에서의 개선 효과가 두드러졌습니다. 실험 결과는 Click-Calib의 정확성과 강건함이 기존의 방법들보다 우수하다는 것을 입증합니다. 이를 통해 환경의 불확실성에 대한 강인성까지 보여줍니다.



### Task-Driven Fixation Network: An Efficient Architecture with Fixation Selection (https://arxiv.org/abs/2501.01548)
Comments:
          9 pages, 2 figures, 2 tables

- **What's New**: 이 논문은 자동 고정 포인트 선택 기능을 갖춘 새로운 신경망 아키텍처를 제안하고 있습니다. 이 모델은 크기와 계산 오버헤드를 줄이면서도 복잡한 작업을 효율적으로 처리하도록 설계되었습니다. 저해상도 채널과 고해상도 채널로 구성되어 있으며, 하이브리드 인코딩 모듈을 통해 두 채널의 특징을 통합합니다. 고해상도 채널의 주요 기능은 작업 중심으로 동적으로 고정 포인트를 생성하여 관심 영역에 집중하는 것입니다.

- **Technical Details**: 제안된 TDFN 모델은 Transformer 아키텍처를 기반으로 구현되었습니다. 이 구조는 고해상도 정보와 저해상도 정보를 통합하여 복잡한 작업을 수행하며, 작업 수행을 위한 작업 메모리를 통해 입력 토큰 시퀀스를 관리합니다. 모델의 처리는 저해상도 이미지를 패치로 나누고, 이를 인코딩하여 고해상도 채널과 하이브리드 인코더로 전달하게 됩니다. 이러한 처리 과정을 반복하여 모델이 필요한 성능 기준에 도달할 때까지 수행됩니다.

- **Performance Highlights**: MNIST 데이터셋을 통한 실험 결과, TDFN 모델은 자원 요구 사항을 크게 줄이면서도 높은 작업 성능을 보여주었습니다. 모델은 고해상도 분석을 전체 이미지에 대해 반복적으로 수행하는 것이 아니라, 동적으로 선택된 고정 포인트를 통해 관심 영역에 집중함으로써 효율성을 극대화합니다. 이러한 접근은 특히 비용 효율적인 작업 수행에 유리하며, 신경망의 크기를 줄이는 동시에 성능 지표는 유지할 수 있음을 보여줍니다.



### SAFER: Sharpness Aware layer-selective Finetuning for Enhanced Robustness in vision transformers (https://arxiv.org/abs/2501.01529)
- **What's New**: 비전 트랜스포머(ViTs)는 고급 컴퓨터 비전 애플리케이션 및 다중 모달 기초 모델에서 필수적인 기반을 형성하고 있습니다. 하지만 이들은 적대적 변형(adversarial perturbations)에 취약해, CNN과 비슷한 수준 또는 그 이상으로 취약해질 수 있습니다. 본 논문에서는 적대적 과적합(adversarial overfitting)을 완화하기 위한 새로운 계층 선택적 미세 조정 방법인 SAFER를 제안합니다.

- **Technical Details**: SAFER는 전체 모델을 최적화하는 대신, 과적합의 영향을 많이 받는 소수의 계층을 선택적으로 미세 조정합니다. 이를 위해 샤프니스 인식 최소화(sharpness-aware minimization) 기법을 적용하여 나머지 모델은 동결(freeze)한 상태로 유지합니다. 이러한 접근 방식의 결과로, 일반적인 경우 5% 정도의 정확도를 향상시킬 수 있으며, 최댓값으로는 20%까지도 도달할 수 있습니다.

- **Performance Highlights**: SAFER는 다양한 ViT 아키텍처와 데이터셋에 대해 일관되게 성능을 향상시켰습니다. 본 연구는 SAFER의 적응성을 통해 다양한 훈련 방법과 매개변수 효율적인 미세 조정(PEFT) 프레임워크와의 통합에서 뛰어난 성능을 입증하였습니다. 이러한 결과는 SAFER가 다변화된 응용 프로그램에 잘 적합하다는 것을 보여줍니다.



### LS-GAN: Human Motion Synthesis with Latent-space GANs (https://arxiv.org/abs/2501.01449)
Comments:
          6 pages

- **What's New**: 이번 논문에서는 Generative Adversarial Networks (GANs)를 활용하여 텍스트 입력에 기반한 사람의 모션 합성을 위한 새로운 프레임워크를 제안합니다. 기존의 디퓨전 모델에 비해 훨씬 더 빠른 훈련 및 추론 속도를 자랑하며, 높은 품질의 3D 모션을 생성할 수 있습니다. HumanML3D 및 HumanAct12 벤치마크에서 실험을 진행하였고, 저는 GAN 아키텍처가 뛰어난 성능을 보여줍니다.

- **Technical Details**: 이 연구는 Variational Autoencoder (VAE)를 통해 모션을 잠재 공간(latent space)으로 인코딩하고, Pre-trained CLIP 모델을 사용하여 텍스트 입력에 대한 조건을 설정합니다. 이 과정에서 GAN 아키텍처를 도입하여 텍스트 임베딩과 잠재 공간 간의 매핑을 가속화하고, 더 높은 품질의 모션 시퀀스를 효율적으로 생성하는 것을 목표로 합니다. 특히, GAN의 훈련 동적을 활용하여 성능과 충실도를 최적화하고 다양한 GAN 아키텍처를 실험합니다.

- **Performance Highlights**: 실험 결과, 단순한 GAN 아키텍처가 0.482의 FID를 기록하며, MLD에 비해 91%의 FLOPs 감소를 보였습니다. 또한, Action-to-motion HumanAct12 벤치마크에서도 경쟁력 있는 성능을 나타내었습니다. 이러한 GAN의 잠재 공간 내의 적용은 기존 디퓨전 기반 모델의 계산 효율성을 해결하며 실시간 애플리케이션에 적합한 고품질 모션 합성을 위한 잠재력을 열어줍니다.



### Exoplanet Detection via Differentiable Rendering (https://arxiv.org/abs/2501.01912)
Comments:
          Webpage: this https URL

- **What's New**: 이 논문에서는 파장 전단 데이터(wavefront sensing data)를 활용하여 외계 행성 탐지를 개선하는 미분 가능 렌더링( differentiable rendering) 접근법을 제안합니다. 기존의 이미지 강도 도메인(intensity domain)에서 작동하는 전통적인 후처리(post-processing) 방법들은 이러한 데이터를 통합하지 못하고 있었던 반면, 본 연구는 이러한 데이터를 활용하여 보다 나은 외계 행성 신호 감지를 가능하게 합니다. 또한, 제임스 웹 우주 망원경(JWST) 구성에 기반한 시뮬레이션 실험을 통해 개선된 감도를 입증합니다.

- **Technical Details**: 본 연구에서 제안하는 미분 가능 렌더러(differentiable renderer)는 코로나그래픽 망원경 시스템을 통해 전파되는 빛의 파동 기반(light wave) 전파를 모델링합니다. 이를 통해 그래디언트 기반 최적화(gradient-based optimization)가 이루어져 별빛의 제거(starlight subtraction)를 효율적으로 수행하고, 외계 행성에 대한 감도를 높입니다. 특히, 파장 전단 데이터와의 통합을 통해 시각화하며, 과거의 추정치를 활용해 잔여 빛 신호를 정확히 보정하는 과정이 포함되어 있습니다.

- **Performance Highlights**: 시뮬레이션 실험 결과, 제안된 방법은 기존 후처리 기술에 비해 별빛 제거에서 상당한 개선을 보여주었으며, 외계 행성 탐지의 한계치를 획기적으로 높였습니다. 이 접근법은 다양한 관측 조건에서 Robustness를 유지하며 향후 외계 행성 촬영 임무에 미치는 잠재적인 영향을 제시합니다. 따라서, 이 연구는 관측 전략의 진전을 통해 외계 행성의 이미지 및 특성을 향상시킬 수 있는 새로운 길을 열어줍니다.



### EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation (https://arxiv.org/abs/2501.01895)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 EnerVerse라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 로봇 조작 작업을 위한 미래 공간 생성을 포괄적으로 지원합니다. EnerVerse는 내부 청크(space chunk) 모델링을 위한 합성곱(convolutional) 및 양방향(attention) 메커니즘을 통합하여 저수준 일관성과 연속성을 보장합니다.

- **Technical Details**: EnerVerse 프레임워크는 비디오 데이터의 본질적인 중복성을 인식하고, 희소 메모리(context) 문맥 및 청크 단위(unidirectional) 생성 패러다임을 사용하여 무한 길이 시퀀스 생성을 가능하게 합니다. 또한, 자유 앵커 뷰(Free Anchor View, FAV) 공간을 도입하여 작업 관찰 및 분석을 향상시키는 유연한 관점을 제공합니다. 이 공간은 물리적 제약을 제거하고 로봇의 일반화 및 적응성을 개선합니다.

- **Performance Highlights**: 실험을 통해 EnerVerse의 미래 공간 생성 접근법이 정책 예측 능력을 크게 향상시켜 전체적인 성능 개선에 기여함을 보여주었습니다. 특히 긴 거리의 로봇 조작 작업에서 눈에 띄는 성과를 보여주며, 4D Gaussian Splatting과 결합된 데이터 엔진 파이프라인이 데이터 품질과 다양성을 반복적으로 향상시키는 구조적 이점을 제공함에도 기여하고 있습니다.



### Universal Online Temporal Calibration for Optimization-based Visual-Inertial Navigation Systems (https://arxiv.org/abs/2501.01788)
Comments:
          7 pages

- **What's New**: 본 연구에서는 시각 센서와 관성 센서를 결합한 6자유도(6DoF) 운동 추정의 정밀한 시간 오프셋 보정 전략을 제안합니다. 기존의 최적화 기반 비주얼-관성 내비게이션 시스템에 적합한 온라인 시간 보정 방법을 마련하여, 다른 추적 프론트엔드와도 호환성을 가집니다. 특히 노이즈가 많은 센서 데이터에서도 높은 정확도의 시간 오프셋 추정이 가능합니다.

- **Technical Details**: 제안된 방법은 최적화 잔차 모델의 상태 매개변수로 시간 오프셋(td)을 통합하여 IMU 상태와 이미지 타임스탬프를 일치시킵니다. 이러한 방식으로 td의 시간 비대칭을 다른 추적 상태와 동시에 최적화하고, 잔차 모델의 구조만을 수정하여 다양한 최적화 기반 프레임워크에 적용할 수 있습니다. 이는 기존 방법들과 비교해 유연성을 높이며, 여러 VINS 프레임워크에 적용 가능합니다.

- **Performance Highlights**: 우리의 보정 방법은 EuRoC 및 시뮬레이션 데이터를 باستخدام하여 평가되었으며, 실험 결과 더 높은 정확도의 시간 오프셋 추정과 더 빠른 수렴 속도를 달성했습니다. 특히, 센서 데이터가 노이즈로 오염되었을 때에도 성능이 향상됨을 보여줍니다. 이러한 결과는 향후 비주얼-관성 내비게이션 시스템의 성능을 크게 개선할 잠재력을 지니고 있습니다.



### Compressed Domain Prior-Guided Video Super-Resolution for Cloud Gaming Conten (https://arxiv.org/abs/2501.01773)
Comments:
          10 pages, 4 figures, Data Compression Conference2025

- **What's New**: 최근 클라우드 게임은 네트워크 전송의 발전으로 인해 각광받고 있으며, 이에 따른 비디오 콘텐츠 압축 및 해상도 저하의 도전이 커지고 있습니다. 이 논문에서는 압축된 게임 비디오 콘텐츠의 Super-Resolution(SR) 문제 해결을 위해 Coding Prior-Guided Super-Resolution(CPGSR) 네트워크를 제안합니다. 이를 통해 낮은 해상도의 이미지나 비디오를 복원하는 데 필요한 중요한 요소들을 효과적으로 활용할 수 있도록 설계되었습니다.

- **Technical Details**: CPGSR 네트워크는 세 부분으로 나뉘며, Compressed Domain Guided Block(CDGB)을 통해 코딩 우선 정보를 추출하고, 이를 U-net 백본과 통합하여 깊은 특성을 융합합니다. 이후 재구성을 위한 여러 개의 재매개변수화 블록이 사용됩니다. 또한 약화된 고주파 정보를 복원하기 위한 partitioned focal frequency loss가 도입되어, 비디오 압축 과정에서 손실된 정보를 효과적으로 복구하도록 안내합니다.

- **Performance Highlights**: 실험 결과, 제안된 CPGSR 네트워크는 기존의 슈퍼 해상도 방법보다 비디오 콘텐츠 복원에서 더 나은 성능을 보이며, 60Hz 이상의 재생률을 지원합니다. 새로운 VVC(가변 비디오 코딩) 표준 기반의 데이터셋이 구축되어 압축된 클라우드 게임 콘텐츠 향상을 위한 연구를 촉진하고 있습니다. 이러한 접근은 클라우드 게임의 전반적인 품질 향상에 중요한 기여를 할 것으로 예상됩니다.



### Laparoscopic Scene Analysis for Intraoperative Visualisation of Gamma Probe Signals in Minimally Invasive Cancer Surgery (https://arxiv.org/abs/2501.01752)
Comments:
          Doctoral thesis

- **What's New**: 본 논문에서는 미세침습 수술(Minimally Invasive Surgery, MIS)을 위한 새로운 intraoperative visualisation tool을 개발하는 과정에 대해 다룹니다. 특히, SENSEI라는 미니어처 암 탐지 프로브를 사용하여 방사선 시그널을 통해 암을 보다 정확하게 식별하는 방법이 소개되었습니다. 이러한 도구는 기존의 수술 기술에서의 한계를 극복하고, 외과의사가 수술 중 암 조직을 보다 효과적으로 제거할 수 있도록 돕기 위해 고안되었습니다. 이는 암 치료의 혁신적인 전환을 기대하게 합니다.

- **Technical Details**: 연구에서는 카메라와 감마 프로브를 사용하는 새로운 시스템을 통해 시각화 문제를 해결하기 위한 다양한 기술적 방법을 제시했습니다. 프로브 감지 영역의 위치를 추론하는 문제는 사용자 맞춤형 레이저 모듈을 활용하여 접근성을 높혔습니다. 이 시스템은 단순한 네트워크 디자인을 통해 실시간으로 수술 중 시각적 피드백을 제공하며, 이는 외과 시각화 커뮤니티에 새로운 기준을 설정합니다.

- **Performance Highlights**: 이 연구의 성과는 미세침습 수술 분야에서 감마 프로브의 고도화된 통합 및 다양한 컴퓨터 비전 알고리즘 개발에 있습니다. 이러한 혁신들은 수술 중 암을 인식하고 제거하는 과정의 시각적 인식을 향상시킵니다. 특히, 향후 증강 현실(Augmented Reality, AR) 및 가상 현실(Virtual Reality, VR) 시스템으로의 응용 가능성은 외과의사들에게 실질적인 도움을 줄 수 있을 것으로 기대됩니다.



### Optimal Fiducial Marker Placement for Satellite Proximity Operations Using Observability Gramians (https://arxiv.org/abs/2501.01704)
Comments:
          18 pages, 7 figures, 1 table, presented at 45th Annual American Astronautical Society (AAS) Guidance, Navigation and Control (GNC) Conference

- **What's New**: 이 논문은 두 위성이 상대적으로 접근 작업을 수행할 때 최적의 fiducial marker 위치를 연구합니다. 일반적으로 위성은 작은 크기와 높은 자율성을 갖는 대세로 변화하고 있으며, 이는 상대적 접근 작업의 중요성을 더욱 부각시킵니다. 논문에서는 dual quaternions를 사용하여 두 위성의 운동 방정식을 모델링하고, 상대 시스템의 관측 가능성을 실험적인 Gramian 방법으로 분석합니다.

- **Technical Details**: 아울러, fiducial markers의 최적 배치 방안을 고안하여 각각의 마커가 동시에 시각적 거리 및 자세 측정을 제공하도록 설계되었습니다. 따라서 각 마커의 위치는 비선형 궤도를 따라 상태 변화를 측정하는 데 가장 민감하도록 선정되어야 합니다. 이 연구는 또한 quaternions과 dual quaternions 간의 정의 및 속성에 대한 대한 논의를 포함하고 있습니다.

- **Performance Highlights**: 결과적으로, 5개 및 10개의 fiducial markers의 최적 배치 세트를 수치적으로 해결하였으며, 최적 해법은 fiducial markers 간의 거리를 극대화하는 방안을 제시합니다. 이 해법은 각 마커 위치를 통해 비록 다른 위치보다 더 적은 시간 동안 시각화되더라도 상대적으로 가장 효과적인 변화를 측정하는 것을 목표로 하고 있습니다. 따라서, 이 연구는 향후 우주 임무의 효율성 및 안전성을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### SNeRV: Spectra-preserving Neural Representation for Video (https://arxiv.org/abs/2501.01681)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 스펙트라 보존 NeRV(SNeRV)를 제안하여 비디오 표현에서의 스펙트럴 바이어스를 해결하고자 하였습니다. SNeRV는 2D 이산 웨이브릿 변환(Discrete Wavelet Transform, DWT)을 사용하여 비디오를 저주파 및 고주파 features로 분해함으로써 공간 구조를 보존하면서도 스펙트럴 바이어스를 직접 다룹니다. 또한, 시간적 상관관계를 잘 포착하기 위하여 시간 도메인으로의 주파수 분해의 확장을 제안합니다.

- **Technical Details**: SNeRV는 다중 해상도 융합 유닛(Multi-resolution Fusion Unit, MFU)과 고주파 복원기(High-Frequency Restorer, HFR)를 포함한 여러 전문 모듈을 통합하여 비디오 표현을 용이하게 만듭니다. 이 모델은 저주파(LF) 구성 요소만을 인코딩하고, 고주파(HF) 구성 요소는 복원기에 의해 생성됨으로써 모델의 Compactness를 유지합니다. 또한, 비디오 프레임 간의 시간적 상관관계를 효과적으로 캡처하기 위해 시공간 LF features를 네트워크에 내장하는 방식으로 시간 모델링을 수행합니다.

- **Performance Highlights**: 실험 결과 SNeRV는 기존 NeRV 모델보다 세부 사항을 보다 잘 포착하고 복원 성능이 향상되었습니다. 다양한 비디오 처리 작업 및 데이터 셋에서 제안된 방법의 성능을 평가한 결과, 스펙트럴 바이어스 문제를 완화시킴으로써 향상된 효과를 보였습니다. 따라서 SNeRV는 비디오의 암묵적 표현 분야에서 유망한 접근법으로 자기 자신의 성능을 입증하게 되었습니다.



### Crossing Language Borders: A Pipeline for Indonesian Manhwa Translation (https://arxiv.org/abs/2501.01629)
- **What's New**: 본 연구에서는 인도네시아어에서 영어로의 만화(Manhwa) 번역을 자동화하기 위한 효율적인 솔루션을 제안합니다. 컴퓨터 비전, 텍스트 인식, 자연어 처리 기법을 결합하여 전통적인 번역 방식의 비효율성을 해소하고자 합니다. 이 시스템은 의사 대화의 구간을 탐지하고, 문자를 인식하며, 이를 번역하여 만화 패널에 다시 오버레이하는 단계를 포함합니다.

- **Technical Details**: 연구에 사용된 주요 기법은 YOLOv5xu를 활용한 스피치 버블(spoech bubble) 탐지, Tesseract를 통한 광학 문자 인식(Optical Character Recognition), 그리고 MarianMT를 통한 기계 번역입니다. YOLOv5xu는 고정밀 객체 감지를 위해 미세 조정되었고, Tesseract는 인도네시아어 모델을 사용하여 문자를 효율적으로 인식합니다. 마지막으로 번역된 텍스트는 OpenCV와 Pillow 라이브러리를 통해 원본 이미지의 대화 상자 형태에 맞게 재배치됩니다.

- **Performance Highlights**: 모델의 성능은 YOLOv5xu가 90.7%의 F1 점수를 나타내어 스피치 버블을 효과적으로 검출했다는 것을 보여줍니다. OCR의 경우, Tesseract를 활용해 CER 3.1%, WER 8.6%의 성능을 기록하며 기존 방법보다 우수한 결과를 달성하였습니다. 또한, MarianMT 모델은 BLEU 점수와 Meteor 점수를 통해 번역의 의미적 보존이 잘 이루어졌음을 증명했으며, 이 모든 단계를 통합한 자동화 파이프라인은 만화 번역 작업의 효율성을 높이는데 기여하고 있습니다.



### Embedding Similarity Guided License Plate Super Resolution (https://arxiv.org/abs/2501.01483)
Comments:
          Submitted to Neurocomputing

- **What's New**: 본 연구는 라이센스 판 슈퍼 해상도(LPSR) 분야에서의 새로운 접근법을 제안합니다. 새로운 프레임워크는 픽셀 기반 손실(pixel-based loss)과 임베딩 유사성 학습(embedding similarity learning)을 결합하여 LPSR의 고유한 문제를 해결하는 데 초점을 맞춥니다. 특히, 픽셀 및 임베딩 일관성 손실(PECL)을 도입하여, 시암 네트워크(Siamese network)를 이용해 임베딩의 유사성을 개선합니다.

- **Technical Details**: 제안된 RDASRNet 모델은 레지듀얼 밀집 블록(residual dense blocks)과 채널 주의 메커니즘(channel attention mechanisms)을 활용하여 저해상도 라이센스 판 이미지를 복원합니다. 이 방법은 픽셀 수준(pixel-level) 손실과 특성 수준(feature-level) 손실을 통합하여 높은 해상도와 슈퍼 해상도 이미지 간의 임베딩 유사성을 강화합니다. 또한, 대조 손실(contrastive loss)을 활용하여 모델이 더 분별력 있는 특성을 학습하도록 유도합니다.

- **Performance Highlights**: CCPD 데이터셋에서의 광범위한 실험을 통해, 제안된 프레임워크는 PSNR_RGB, PSNR_Y 및 광학 문자 인식(OCR) 정확도 면에서 기존의 최신 기법들보다 지속적인 향상을 보였습니다. 이러한 결과는 극한의 슈퍼 해상도 시나리오에서 감각 품질과 작업별 성능을 개선하는 데 있어 임베딩 유사성 학습의 잠재력을 강조합니다.



### An unsupervised method for MRI recovery: Deep image prior with structured sparsity (https://arxiv.org/abs/2501.01482)
- **What's New**: 본 연구에서는 전체 샘플링된 k-space 데이터가 필요 없는 비지도 MRI 복원 방법인 구조적 희소성을 적용한 Deep Image Prior (DIP) 확장 모델인 DISCUS를 제안하고 검증합니다. DISCUS는 그룹 희소성을 도입하여 프레임 특정 코드 벡터를 개선하고, 이를 통해 시간 변화를 포착할 수 있는 저차원 매니폴드를 발견할 수 있게 합니다. 또한, DISCUS는 기존의 방법들과 달리 매니폴드의 차원 수를 미리 정할 필요 없이 동적 코드 벡터에 그룹 희소성을 부과하여 차원 수를 찾습니다.

- **Technical Details**: MRI에서 복원 과정은 노이즈가 있는 비샘플링 k-space 데이터에서 기본 이미지를 추정하는 것을 포함합니다. DISCUS는 무작위 코드 벡터를 네트워크에 공급하여 영상 시퀀스를 생성하며, 전통적인 방법들과 비교하여 이미지 유사성을 더 많이 반영하지 않고도 시간적으로 근접한 데이터 간의 유사성을 가정하지 않습니다. 이 방법은 특히 단일 샷의 자유 호흡 LGE 촬영 및 매개변수 매핑에 적합합니다.

- **Performance Highlights**: DISCUS는 여러 연구를 통해 기존 방법들보다 뛰어난 성능을 보여주었습니다. 시뮬레이션 및 실제 데이터를 기반으로 한 평가에서 NMSE와 SSIM 기준의 복원 품질 향상을 입증하였으며, 전문가 평가에서 높은 점수를 기록했습니다. 이러한 결과는 특히 정상적인 스캔 조건에서의 환자 불편을 줄이고, MRI의 임상 적용 가능성을 높이는 데 기여할 것입니다.



### Unleashing Correlation and Continuity for Hyperspectral Reconstruction from RGB Images (https://arxiv.org/abs/2501.01481)
- **What's New**: 이번 연구에서는 RGB 이미지로부터 고분해능의 Hyperspectral 이미지(HSI)를 저비용으로 재구성하는 새로운 방법을 제안합니다. 특히, HSI 재구성 작업에서 지역적 상관관계(local correlation)와 전역적 연속성(global continuity)이 필수적임을 밝혔습니다. 이를 토대로 Correlation and Continuity Network (CCNet)이라는 혁신적인 네트워크 구조를 설계하여 HSI 재구성을 더욱 효과적으로 수행합니다.

- **Technical Details**: 우리의 모델은 두 개의 주요 모듈을 포함하고 있습니다. 첫 번째는 Group-wise Spectral Correlation Modeling (GrSCM) 모듈로, 지역적인 범위 내에서의 스펙트럼 밴드 유사성을 효과적으로 구축합니다. 두 번째는 Neighborhood-wise Spectral Continuity Modeling (NeSCM) 모듈로, 메모리 유닛(memory units)을 사용해 전역 수준에서의 점진적 변화를 모델링합니다. 이를 통해 HSI 재구성의 질을 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 NTIRE2022 및 NTIRE2020 데이터셋을 기반으로 한 철저한 비교 및 Ablation 실험을 통해 평가되었습니다. 기존의 고급 스펙트럼 재구성 알고리즘과 비교하여, 우리의 방법은 State-Of-The-Art(SOTA) 성능을 달성함으로써 그 우수성을 입증하였습니다.



### Tech Report: Divide and Conquer 3D Real-Time Reconstruction for Improved IGS (https://arxiv.org/abs/2501.01465)
- **What's New**: 이 논문에서는 내시경 비디오를 기반으로 한 수술 수정 사항 추적을 효율적으로 처리하기 위한 모듈식 파이프라인을 제안합니다. 이 파이프라인은 프레임 선택, 깊이 추정 및 3D 재구성 구성 요소를 통합하여 새로운 방법의 결합을 유연하게 할 수 있도록 합니다. 최근 발전으로는 Depth-Anything V2 및 EndoDAC이 깊이 추정에 통합되었고, ICP 정렬 과정의 개선이 포함됩니다.

- **Technical Details**: 이 파이프라인은 Python으로 구현되었으며, 모듈 구조와 자세한 인라인 문서가 포함되어 최적화된 환경에서 실행됩니다. 프레임 선택 단계는 여러 선택 방법을 순차적으로 적용하여 정보가 부족한 프레임을 필터링하고, HyperIQA와 R-channel intensity를 통해 프레임 품질을 평가합니다. 깊이 추정 알고리즘으로는 Depth-Anything V2 및 EndoDAC을 사용하여 선택된 프레임의 깊이 맵을 생성하고, 최종적으로 각 프레임의 깊이 정보를 통합하여 3D 포인트 클라우드를 형성합니다.

- **Performance Highlights**: Hamlyn 데이터셋에서 수행된 실험은 통합된 방법들의 효과성을 보여줍니다. 실시간 내시경 비디오와 도구 동작을 이용하여 수술 수정을 추적하는 방법은 IGS의 내비게이션 정밀성과 신뢰성을 향상시킬 가능성이 가 있습니다. 제안된 시스템은 기존의 방법들보다 복잡한 조건에서도 더 나은 성능을 발휘할 것으로 기대되며, 수술 과정에서도 유용한 도구가 될 것입니다.



### Estimation of 3T MR images from 1.5T images regularized with Physics based Constrain (https://arxiv.org/abs/2501.01464)
Comments:
          conference paper

- **What's New**: 최근 1.5T MRI 이미지를 3T와 유사한 고품질 이미지로 개선하기 위한 새로운 비지도 학습 기반 방법이 제안되었습니다. 기존 방법들은 고품질 이미지를 얻기 위해 예제 이미지나 픽셀 대응을 필요로 했으나, 이 방법은 이러한 필요를 없앴습니다. 제안된 방법은 선형 변환을 통해 저자기장(LF) 이미지를 고자기장(HF) 이미지로 변환하는 데 초점을 맞춥니다.

- **Technical Details**: 제안된 방법은 대체 최소화(alternate minimization) 프레임워크를 사용하여 고자기장 이미지(𝐱)와 저자기장 이미지(𝐲) 간의 관계를 모델링합니다. 여기서는 물리 기반의 제약을 도입하여 T1 이완 시간의 차이를 활용하여 HF 이미지를 시뮬레이션하는 정규화자를 사용합니다. 결과적으로, 연산을 통해 생성된 이미지는 고품질 이미지로 노출되어 더 나은 조직(segmentation)와 부피 정량화(a quantification) 성능을 제공합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 1.5T 이미지를 처리하여 3T와 유사한 고품질 이미지를 생성하는 데 성공했음을 입증했습니다. 또한 기존 방법들과 비교했을 때, 조직 경계의 선명함과 이미지 대비에 있어 유의미한 개선을 보여주었습니다. 이는 의료 이미징에서 저자기장 장비의 활용성을 높일 수 있는 잠재력을 가집니다.



### GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution (https://arxiv.org/abs/2501.01460)
Comments:
          GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution

- **What's New**: 이 논문에서는 Remote Sensing Image (RSI) Super-Resolution (SR) 분야에서 새로운 접근 방식을 제시합니다. Receptance Weighted Key Value (RWKV) 모델을 적용하여 긴 범위의 의존성을 선형 복잡도로 캡처함으로써 지역 및 글로벌 기능을 동시에 모델링하는 Global-Detail 구조(GDSR)를 도입하였습니다. 또한 Wavelet Loss를 제안하여 이미지의 고주파 세부 정보를 효과적으로 포착하여 시각적 품질을 향상시킵니다.

- **Technical Details**: GDSR은 RWKV와 convolutional operations를 병렬로 실행하여 대규모 RSI를 처리합니다. Global-Detail Reconstruction Module (GDRM)은 두 브랜치 간의 상호작용을 증진시켜 전체 성능을 향상시키도록 설계되었습니다. Wavelet Loss는 단순하면서도 효과적으로 PSNR과 시각적 품질을 향상시킬 수 있는 손실 함수로, 다양한 네트워크 아키텍처에서의 강력한 성능 개선을 보여줍니다.

- **Performance Highlights**: GDSR은 여러 벤치마크에서 기존의 Transformer 기반 방법에 비해 평균 0.05 dB 높은 PSNR을 달성하면서도 파라미터 수는 63%, FLOPs는 51%로 줄여 2.9배 빠른 추론 속도를 자랑합니다. 다양한 네트워크 아키텍처에서 Wavelet Loss의 일반화 능력도 뛰어나며, 이는 RSI-SR 개선을 위한 새로운 관점을 제시합니다.



### SS-CTML: Self-Supervised Cross-Task Mutual Learning for CT Image Reconstruction (https://arxiv.org/abs/2501.01456)
- **What's New**: 본 논문에서는 CT 이미지 재구성을 위한 새로운 자기 지도 크로스 태스크 상호 학습 프레임워크인 SS-CTML을 제안합니다. 이 프레임워크는 낮은 방사선 노출로 인한 데이터 부족 문제를 해결하기 위해 세 가지 재구성 작업을 하나의 구조로 통합합니다. 특히, 세 개의 신경망이 서로 학습할 수 있는 상호 학습 목표를 설정하여 최종적으로 고품질 CT 이미지를 재구성하는 데 도움을 줍니다.

- **Technical Details**: SS-CTML 프레임워크는 전체 CT(FVCT), 희귀 뷰 CT(SVCT), 제한 뷰 CT(LVCT) 재구성을 위한 독립적인 세 가지 작업을 포함합니다. 이 과정에서 서로 다른 시장 신경망을 설계하였으며, 이들은 FBP(filtered back-projection) 재구성 방식으로 촉진됩니다. 핵심적으로, 세 개의 네트워크는 서로의 출력을 기반으로 최적 학습을 수행하여 데이터의 상호 보완을 이루도록 구성되었습니다.

- **Performance Highlights**: 임상 데이터셋을 통한 실험 결과, SS-CTML 프레임워크는 저 방사선 노출 및 여러 재구성 방식에서 기존 방법들에 비해 우수한 성능을 보였습니다. 본 프레임워크는 잡음 억제, 아티팩트 감소, 세부 구조 보존 측면에서 현저한 향상을 이루어 대상 CT 이미지 품질의 향상에 크게 기여할 것으로 기대됩니다.



### Real-Time Computational Visual Aberration Correcting Display Through High-Contrast Inverse Blurring (https://arxiv.org/abs/2501.01450)
Comments:
          26 pages, 14 figures

- **What's New**: 본 논문은 안경이나 콘택트 렌즈와 같은 전통적인 시력 보정 장치를 사용하지 않고 굴절 시각 왜곡을 해결하기 위해 실시간 비전 보정 디스플레이(VCD) 개발 프레임워크를 제안합니다. 이 프레임워크는 점 확산 함수(point spread function, PSF)를 사용하여 표시된 이미지를 역컨볼루션(deconvolution)하여 시각 왜곡을 교정합니다. 또한 전색 차이를 줄이고 대비를 향상시키기 위해 YUV/YCbCr 색 공간에서 작동하며, 화면을 바라보는 관찰자의 구면 좌표에 기반한 실시간 PSF 계산 기법을 도입합니다.

- **Technical Details**: 방법론에서는 관찰자의 위치에 따라 시간을 두고 스냅샷을 찍고, 이 이미지를 PSF에 따라 역컨볼루션하여 화면에 다시 표시합니다. PSF는 시각적 흐림을 시뮬레이션하기 위해 정의되며, 정규화(normalization)를 통해 이미지의 밝기를 보존합니다. 본 연구는 Zernike 다항식(aberration representation)을 사용하여 여러 종류의 굴절 왜곡을 모델링할 수 있으며, 이를 통해 실시간 비전 보정 디스플레이를 구현합니다.

- **Performance Highlights**: 결과적으로 본 연구의 디스플레이에서는 시각적 선명도가 현저하게 개선되었으며, 구조적 유사성 지수(structural similarity index, SSIM)는 83.04%에 이르렀습니다. 이러한 결과는 제안된 접근 방식의 효과성을 강조하며, 관찰자가 화면을 바라보는 각도에 구애받지 않는 일관된 시각적 보정을 가능하게 합니다.



New uploads on arXiv(cs.AI)

### ASKCOS: an open source software suite for synthesis planning (https://arxiv.org/abs/2501.01835)
- **What's New**: 이번 논문에서는 ASKCOS의 최신 버전을 소개합니다. ASKCOS는 컴퓨터 보조 합성 계획(CASP)을 위한 오픈 소스 소프트웨어로, 합성 계획에 필요한 다양한 기능을 갖추고 있습니다. 이 소프트웨어는 사용자가 설정한 모델을 기반으로 하는 다양한 궤적 탐색 기능과 자동모드로 작동할 수 있는 모듈을 제공합니다.

- **Technical Details**: ASKCOS는 두 가지 작동 모드를 가지고 있으며, 각 모드는 여러 일단계 전략을 사용하여 효율적인 합성 경로를 도출할 수 있습니다. 일단계 문자열(SMILES)로 정의된 목표 분자에 대해 가능성 있는 전구체를 예측하고, plausibility filter를 통해 비현실적인 후보를 제거합니다. ASKCOS는 템플릿 기반의 모델과 신경-기호적 접근 방식을 활용하여 후보 전구체를 제안하고, 각각의 제안을 추적하고 검증할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: ASKCOS는 수 백 명의 화학자들이 일상적인 작업에 도움을 주며, 의약품 및 화학 분야에서 효용성을 인정받고 있습니다. 이번 연구를 통해 ASKCOS의 확장성과 기능을 평가하며, 사용자의 요구에 맞춘 개선사항을 반영하였습니다. 또한, 이 소프트웨어가 현대 화학 연구에서 중요한 역할을 하고 있다는 점에 주목할 필요가 있습니다.



### SDPO: Segment-Level Direct Preference Optimization for Social Agents (https://arxiv.org/abs/2501.01821)
- **What's New**: 이번 연구에서는 Segment-Level Direct Preference Optimization (SDPO)라는 새로운 방법을 제안하여, LLM 기반 사회 에이전트의 다중 대화 세션에서의 의사결정 및 행동을 개선하고자 했습니다. 기존의 turn-level DPO와 session-level DPO 방법론의 한계를 극복하고자, 특정 키 세그먼트를 집중하여 최적화하는 방식을 채택했습니다. SDPO는 교육 과정에서 발생할 수 있는 노이즈를 최소화하면서 대화의 질을 높일 수 있도록 구성되었으며, 이를 통해 LLM의 사회적 지능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: SDPO 방법론은 불량 세션에서 발생하는 오류가 있는 턴을 식별하고, 그 이전의 상호작용 이력을 바탕으로 긍정적인 세션을 생성합니다. 이후 SDPO는 긍정적인 세션에서의 주요 세그먼트를 파악하여, 같은 길이의 부정적인 세션의 세그먼트와 데이터 쌍을 형성합니다. 다차원적인 대화의 정렬을 위해 SDPO는 잘못된 세그먼트와 해당 긍정적 세그먼트에 대해서만 손실(loss) 값을 산출함으로써, 비어떤 턴으로 인한 교육 노이즈를 제거합니다.

- **Performance Highlights**: SDPO 방법은 SOTOPIA 벤치마크에서 GPT-4o 및 기타 경쟁 모델들과의 상호작용을 통해 검증되었으며, DPO, ETO, DMPO와 같은 기존 기법들을 일관되게 초과 성능을 보여주었습니다. 이 연구의 결과는 세그먼트 레벨의 정렬이 LLM 기반 에이전트의 성능을 획기적으로 향상시킬 수 있음을 입증하고, 기존 기법보다 유연하고 일반적으로 사용 가능한 솔루션임을 시사합니다. 더 나아가 SDPO는 다양한 도메인에 걸쳐 에이전트의 역량을 향상시킬 가능성을 지니고 있습니다.



### Proposing Hierarchical Goal-Conditioned Policy Planning in Multi-Goal Reinforcement Learning (https://arxiv.org/abs/2501.01727)
Comments:
          10 pages, 4 figures, this is a preprint of the peer-reviewed version published by SCITEPRESS for ICAART-2025

- **What's New**: 이 논문은 인간형 로봇이 희소 보상(sparse rewards) 문제를 해결하기 위해 강화 학습(reinforcement learning, RL)과 자동 계획(automated planning)을 결합한 새로운 방법론을 제안합니다. 이를 통해 목표 조건 정책(goal-conditioned policies, GCPs)을 계층적으로 구성하고, 고수준 행동(high-level actions, HLA)을 사용하는 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 계획을 적용합니다. 이 프레임워크는 다양한 복잡한 작업을 수행하기 위한 탐색과 계획의 효율성을 개선할 가능성을 보여줍니다.

- **Technical Details**: 본 연구의 기법은 계층적 강화 학습(hierarchical RL)과 GCPs를 활용하여 에이전트가 여러 작업을 학습할 수 있도록 설계되었습니다. MCTS는 저수준의 원시 행동(primitive actions) 대신 고수준 행동을 계획하는 데 사용됩니다. 또한, 단일 계획 트리는 에이전트의 목표 달성을 위한 지식을 보유하고 있으며, HLAs를 재사용함으로써 샘플 효율성을 높이고 미래 행동을 예측할 수 있게 합니다.

- **Performance Highlights**: 제안된 계층적 목표 조건 정책 계획(HGCPP) 프레임워크는 복잡한 문제 해결을 위한 더 나은 탐색 및 계획 방법을 제공할 잠재력이 있습니다. 일반적으로, 에이전트는 MCTS를 통해 계획된 HLAs를 이용하여 빠른 추론을 수행하고 여러 목표에 도달할 수 있습니다. 이 연구는 초기 단계의 연구로 평가가 이루어지지 않았지만, 기존 방법들과의 차별화된 접근을 통해 향후 발전 가능성을 제시합니다.



### AgentRefine: Enhancing Agent Generalization through Refinement Tuning (https://arxiv.org/abs/2501.01702)
- **What's New**: 이 연구는 Large Language Model (LLM) 기반 에이전트의 일반화 능력을 향상시키기 위한 새로운 프레임워크인 AgentRefine를 제안합니다. 기존 연구는 특정 에이전트 환경에서의 과적합(overfitting) 문제로 인해 일반화 성능이 떨어지는 것으로 분석되었으며, 이에 대한 해결책으로 자기 수정 능력을 강조하고 있습니다.

- **Technical Details**: AgentRefine는 에이전트가 관찰(observation)을 통해 자신의 실수를 수정하도록 학습할 수 있도록 돕는 구조입니다. 이를 위해 다양한 환경(environments)과 작업(tasks)을 포함하는 에이전트 합성(agent synthesis) 프레임워크를 도입하여, 환경 피드백에 따라 에이전트가 동작을 개선할 수 있도록 유도합니다.

- **Performance Highlights**: AgentRefine는 다양한 에이전트 작업에서 높은 일반화 능력을 보여주며, 기존 최첨단(agent-tuning) 기술들을 능가합니다. 또한 잡음(perturbation)에 대한 견고성(robustness)이 높고 추론(inference) 과정에서도 다양화된 사고(thought)를 생성할 수 있는 특징을 가지고 있습니다.



### Prism: Mining Task-aware Domains in Non-i.i.d. IMU Data for Flexible User Perception (https://arxiv.org/abs/2501.01598)
Comments:
          in Proceedings of IEEE INFOCOM 2025, London, United Kingdom

- **What's New**: 이 논문은 모바일 디바이스에서 수집된 비독립 동일 분포(non-i.i.d.)의 IMU 데이터를 기반으로 유연한 사용자 인식(Flexible User Perception, FUP) 문제를 해결하기 위한 새로운 방법인 Prism을 제안합니다. 기존의 시스템들이 특정 사용자와 자세에 국한된 제어된 환경에서만 잘 작동했던 것과 달리, Prism은 다양한 장치에서 비정상적인 데이터를 처리할 수 있는 능력을 갖추고 있습니다. 또한, 데이터에서 태스크 인지 도메인을 발견하고, 각 도메인에 대해 도메인 인지 모델을 훈련하는 것을 핵심으로 합니다.

- **Technical Details**: Prism의 핵심은 비독립 동일 분포 데이터셋의 불일치 정도를 측정하고, 이를 기반으로 사용자 인식 태스크에 적합한 도메인으로 데이터셋을 나누는 것입니다. 비정상 데이터를 자동으로 클러스터링하고, 각 클러스터에 대해 개별 태스크 모델을 훈련하기 위해 EM(Expectation-Maximization) 알고리즘을 사용합니다. 이 방법은 데이터셋 내에서의 예측 불일치를 수치적으로 측정하여 비독립 동일 분포의 정도(NID)를 정의하고, 이를 통해 숨겨진 도메인을 효과적으로 추정합니다.

- **Performance Highlights**: Prism은 다양한 모바일 기기에서 실행되어 비정상 공공 IMU 데이터셋을 기반으로 실험을 진행하였으며, FUP 성능에서 최첨단 결과를 달성했습니다. 높은 NID를 가진 데이터셋에서 Prism은 일반 딥러닝 모델보다 F1 점수에서 최대 16.79% 향상된 성능을 보였습니다. 또한, Prism은 경량화되어 저사양 스마트폰에서도 60ms 미만의 낮은 지연 시간을 유지하며 손쉽게 배포될 수 있습니다.



### BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems (https://arxiv.org/abs/2501.01593)
Comments:
          12. arXiv admin note: substantial text overlap with arXiv:2409.07775

- **What's New**: 이 논문에서는 협력적 다중 에이전트 심층 강화 학습(c-MADRL)에 대한 새로운 유형의 백도어 공격인 BLAST(Backdoor Leverage Attack)를 제안합니다. 이 공격은 단일 에이전트에만 백도어를 심어 전체 팀에 영향을 미치는 방식으로, 이를 통해 공격의 은폐성과 효과성을 극대화합니다. 특히, BLAST는 특정 시각 패턴 대신 적대적인 시공간 행동 패턴을 백도어 트리거로 사용하여 은폐성을 향상시킵니다.

- **Technical Details**: BLAST는 단일 에이전트의 보상 함수 해킹을 통해 전체 다중 에이전트 시스템에 대한 leaverage attack 효과를 구현합니다. 이 방법에 의해, 공격자는 BLAST 에이전트의 행동이 다른 에이전트에 미치는 영향을 극대화하며 불필요한 반대 영향을 제거합니다. 이 공격 방식은 분산형 학습 구조인 중앙집중식 훈련과 분산 실행(CTDE) 프레임워크에 적용되며, 이 구조 때문에 공격자가 보이지 않는 상태에서 트리거를 숨길 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 BLAST 공격은 VDN, QMIX 및 MAPPO와 같은 3개의 전통적인 c-MADRL 알고리즘에 대해 높은 공격 성공률을 보이며, 청정 성능 변동률은 낮은 수준을 유지합니다. BLAST의 유효성은 두 가지 인기있는 c-MADRL 환경(SMAC 및 Pursuit)과 기존의 두 방어 메커니즘을 테스트하여 확인되었습니다. 이러한 성과는 BLAST가 기존의 백도어 방어 전략에 대한 저항성을 갖고 있음을 보여줍니다.



### Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search (https://arxiv.org/abs/2501.01478)
Comments:
          5 pages, 1 figure, 2 tables accepted by aaai 2025 NeurMAD workshop

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 능력을 개선하기 위해 Monte Carlo Tree Search (MCTS)를 이용하여 자체적으로 과정 감독(process supervision) 데이터를 생성하는 방법을 제안합니다. 발생한 reasoning step에 각각 '상대적 정합성(relative correctness)' 점수를 부여하고, 이를 통해 LLM을 훈련시키는 과정을 반복적으로 수행했습니다. 이 방법은 결과적으로 두 개의 수학 reasoning 데이터셋에서 LLM의 성능을 상당히 개선하는 것으로 나타났으며, 이는 향상된 추론 능력의 전이성(transferability)도 보여줍니다.

- **Technical Details**: 제안된 방법론에서는 LLM이 문제를 해결하기 위해 생성한 reasoning 경로에 대해 MCTS를 활용하여 각 단계의 '상대적 정합성' 점수를 부여합니다. 이 점수는 이진 선호(binary preferences)보다 각 단계의 품질을 더욱 정확하게 반영하며, 이를 통해 LLM의 가중치 음 로그 우도 손실 함수(weighted negative log-likelihood loss function)에 통합하여 훈련을 진행합니다. 이러한 generate-then-train 방식은 수학적 reasoning 문제에 대한 LLM의 성과를 향상시키는 반복적인 과정으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LLM의 수학적 reasoning 성능을 대폭 향상시켰습니다. 두 개의 데이터셋 각각에서 훈련된 모델이 다른 데이터셋에서도 성능 개선을 보여주었으며, 이는 향상된 추론 능력의 전이성을 시사합니다. 또한, 인간 주석이 필요하지 않은 방식으로 과정 감독을 통해 훈련할 수 있는 가능성을 제시하였습니다.



### Probabilistic Mission Design in Neuro-Symbolic Systems (https://arxiv.org/abs/2501.01439)
Comments:
          arXiv admin note: text overlap with arXiv:2406.03454

- **What's New**: 이번 연구는 Advanced Air Mobility (AAM) 분야에서의 법적 개념과 제약을 정확하게 모델링하는 필요성을 강조하고 있습니다. Unmanned Aircraft Systems (UAS)의 시각적 선을 넘는 운용(BVLOS)에 대한 도전과제를 다루기 위해, 연구팀은 법적 프레임워크와 전문가 지식을 불확실한 공간 관계와 노이즈가 포함된 인식을 통해 이해할 수 있고 적응 가능한 방식으로 인코딩하는 확률적 신경-기호적 아키텍처를 제시합니다.

- **Technical Details**: 제안된 시스템 아키텍처인 Probabilistic Mission Design (ProMis)은 지리공간 및 센서 데이터를 Hybrid Probabilistic Logic Programs (HPLP)와 연결하여 에이전트의 상태 공간과 합법성에 대해 추론하는 기능을 제공합니다. HPLP는 이산 및 연속 분포를 사용하여 확률적 모델을 자유자재로 다루며, 첫 번째 주문 논리(First-Order Logic, FOL)에 확률 분포를 할당합니다. 이를 통하여 ProMis는 다양한 AAM 시나리오에 적용할 수 있는 다중 모드 입력 데이터와 연계를 구축합니다.

- **Performance Highlights**: ProMis는 에이전트의 항법 제약을 충족시키는 임무 조건이 만족될 가능성을 정량화하는 Probabilistic Mission Landscapes (PML)을 생성합니다. 연구팀은 ProMis가 강력한 기계 학습 모델인 Large Language Models (LLM)와 Transformer 기반 비전 모델과 통합될 수 있음을 보여주었으며, 이를 통해 비전 모델이 미션 설계에 어떻게 접목될 수 있는지를 입증합니다. ProMis 프레임워크의 오픈 소스 구현은 향후 지능형 운송 시스템이 법률과 규제를 준수하며 작업을 수행할 수 있도록 돕습니다.



### Mathematical Definition and Systematization of Puzzle Rules (https://arxiv.org/abs/2501.01433)
Comments:
          15pages

- **What's New**: 이 연구에서는 연필 퍼즐의 규칙을 정의하고 체계화하기 위한 수학적 프레임워크를 소개합니다. 이 프레임워크는 그리드 요소의 구성 요소 및 위치 관계를 형식적으로 정의하고, 퍼즐 규칙을 기반으로 하는 구조를 점진적으로 구성할 수 있게 해줍니다. 또한, 퍼즐의 해답성과 일관성을 보장하기 위해 각 구조에 대한 제약 조건과 도메인을 기술하는 공식을 마련했습니다.

- **Technical Details**: 연구에서는 m×n 크기의 2차원 그리드에서 각각의 그리드 점과 셀에 좌표를 부여하는 방법을 설명합니다. 그리드 요소는 점(p), 셀(c), 수평 및 수직 엣지를 포함하며, 각 요소는 특정 변수로 주석이 달립니다. 이를 통해 퍼즐 규칙의 수학적 설명 및 계산적 구현을 가능하게 하는 방법론이 제시되었습니다.

- **Performance Highlights**: 이 프레임워크를 활용하여 Slitherlink 및 Sudoku와 같은 잘 알려진 Nikoli 퍼즐의 규칙을 형식화하는 데 성공하였으며, 이는 기존 퍼즐의 약 1/4에 해당합니다. 결과적으로 이 프레임워크는 퍼즐 규칙 디자인을 체계화하고 혁신할 잠재력을 검증하며, AI를 이용한 자동 규칙 생성과 같은 미래 연구 방향을 제시합니다.



### MixGCN: Scalable GCN Training by Mixture of Parallelism and Mixture of Accelerators (https://arxiv.org/abs/2501.01951)
Comments:
          15 pages, 12 figures, 5 tables

- **What's New**: 본 논문에서는 그래프 기반 머신 러닝 과제에서의 그래프 합성곱 네트워크(GCN)의 효율적인 학습을 위한 MixGCN을 제안합니다. MixGCN은 거대 피쳐 텐서와 혼합 희소-밀집 연산과 같은 두 가지 주요 도전 과제를 동시에 해결하려고 합니다. 이러한 접근법을 통해 메모리 사용량과 통신 대역폭의 비효율성을 감소시킵니다.

- **Technical Details**: MixGCN은 Mixture of Parallelism (MoP)과 Mixture of Accelerators (MoA)라는 두 가지 주요 기술을 기반으로 합니다. MoP는 피쳐 수준과 노드 수준의 병렬성을 결합하여 GCN의 스케일러블한 훈련을 가능하게 하고, MoA는 희소 및 밀집 연산을 각각 다른 가속기에 분산시킵니다. 특히, S-SpMM이라는 고유한 희소 연산을 통해 두 개의 연속적인 희소 연산을 융합하여 효율적인 계산을 구현합니다.

- **Performance Highlights**: MixGCN은 실험을 통해 5개의 대규모 데이터셋에서 우수한 성능을 입증하였으며, 4개의 GPU 클러스터에서 최신 기법 대비 10.4배의 성능 향상을 기록했습니다. 추가적으로 S-SpMM을 사용할 경우 성능은 17.2배로 증가합니다. 제안된 MoP는 안정적인 통신량과 피쳐 메모리 사용을 유지하면서도 균형 잡힌 작업량을 보장합니다.



### MADGEN -- Mass-Spec attends to De Novo Molecular generation (https://arxiv.org/abs/2501.01950)
Comments:
          preprint

- **What's New**: 이번 논문에서는 MADGEN(Mass-spec Attends to De Novo Molecular GENeration)이라는 새로운 접근법을 제안합니다. 이는 질량 분석 데이터(mass spectrometry data)를 기반으로 한 scaffold 기반의 분자 구조 생성을 위한 방법으로, 현재 많은 MS/MS 스펙트럼이 '어두운 화학 공간(dark chemical space)'에 위치하고 있다는 문제를 해결하고자 합니다. MADGEN은 scaffold 검색 및 스펙트럼 조건화된 분자 생성을 두 단계로 나눠 진행합니다.

- **Technical Details**: MADGEN의 첫 단계는 주어진 MS/MS 스펙트럼에 대해 scaffold 검색을 순위(rank) 문제로 설정하고, 대조 학습(contrastive learning)을 이용하여 질량 스펙트럼과 후보 분자 scaffold를 정렬하는 과정입니다. 두 번째 단계에서는 검색된 scaffold에서 시작하여 MS/MS 스펙트럼에 의해 안내되는 주의(attention) 기반 생성 모델을 활용하여 최종 분자를 생성합니다. 이러한 접근 방법은 분자의 생성 검색 공간을 제한하고 복잡성을 줄이며 정확성을 향상시키는 데 기여합니다.

- **Performance Highlights**: MADGEN은 NIST23, CANOPUS, MassSpecGym의 세 가지 데이터셋에서 평가되었으며, 예측된 scaffold와 오라클(oracle) 검색기를 통해 성능을 비교했습니다. 결과적으로, 주의 메커니즘을 통해 스펙트럼 정보를 통합하여 오라클 검색기를 사용할 때 강력한 결과를 달성하는 것을 입증했습니다. 이는 새로운 대사물질, 생리활성 화합물 및 미지의 화합물 탐색에 필수적인 의의를 가집니다.



### Cold-Start Recommendation towards the Era of Large Language Models (LLMs): A Comprehensive Survey and Roadmap (https://arxiv.org/abs/2501.01945)
- **What's New**: 이번 논문은 차가운 시작 문제(cold-start problem)를 해결하기 위해 대규모 언어 모델(large language models, LLMs)을 활용하는 방법에 대한 포괄적인 검토를 제공합니다. 최근 사용자와 아이템의 폭발적인 증가로 인해 추천 시스템(recommender systems)에서 차가운 시작 추천(cold-start recommendation, CSR)의 중요성이 더욱 부각되고 있습니다. 연구 커뮤니티는 아직 CSR 분야에 대한 포괄적인 리뷰가 부족하기 때문에 이 논문은 새로운 통찰력을 제공합니다.

- **Technical Details**: CSR의 기존 정보 활용 경로를 콘텐츠 특징(content features), 그래프 관계(graph relations), 도메인 정보(domain information)부터 LLM이 보유한 세계지식까지 탐색했습니다. 이 논문에서는 연구 및 산업 커뮤니티에 CSR에 관한 새로운 통찰력을 제공하기 위해 기존 문헌을 기반으로 차가운 시작 추천의 로드맵을 제시합니다. 또한, 관련 자료들은 지속적으로 커뮤니티에 제공될 예정입니다.

- **Performance Highlights**: 대규모 언어 모델의 성공적인 적용은 차가운 시작 문제 해결에 새로운 가능성을 제시합니다. 본 연구의 목적은 차가운 시작 추천의 발전 경로에 대한 탐색과 현대 LLM의 혜택을 결합하여 추천 품질을 개선하고 사용자 경험을 향상시키는 것입니다. 이러한 접근 방식은 CSR 연구와 산업 모두에 큰 기여를 할 것입니다.



### Abstractive Text Summarization for Contemporary Sanskrit Prose: Issues and Challenges (https://arxiv.org/abs/2501.01933)
Comments:
          PhD Thesis

- **What's New**: 이 논문은 현대 산스크리트 산문에 대한 추상적 텍스트 요약(abstractive text summarization) 모델을 제안합니다. 주요 연구 질문은 산스크리트어에 대한 추상적 텍스트 요약을 개발하는 데 있어 어떤 도전 과제가 있는지에 대한 것입니다. 이 연구는 낮은 자원(low-resource) 인플렉션(inflectional) 언어인 산스크리트어의 특수를 다루고 있습니다.

- **Technical Details**: 본 논문의 두 번째 장인 문헌 리뷰(literature review)에서는 이전 연구들을 조사하였고, 세 번째 장에서는 데이터 준비(data preparation) 단계에서 언어 모델(language model)과 요약 모델(summarization model) 훈련을 위한 데이터 수집 및 전처리(preprocessing) 문제를 다루었습니다. 네 번째 장에서는 모델의 훈련(training)과 추론(inference) 과정 및 그 결과를 보고합니다.

- **Performance Highlights**: 이 연구는 산스크리트어 추상적 텍스트 요약을 위한 파이프라인(pipeline)을 시작하였으며, 개발의 각 단계에서 직면한 도전 과제를 보고하였습니다. 모든 주제에 기반한 연구 질문들은 주요 연구 질문에 대한 답변을 제공하기 위해 다루어졌습니다.



### Mitigating Hallucination for Large Vision Language Model by Inter-Modality Correlation Calibration Decoding (https://arxiv.org/abs/2501.01926)
- **What's New**: 이 논문에서는 LVLMs(대형 비전-언어 모델)의 환각(hallucination) 문제를 해결하기 위해 Inter-Modality Correlation Calibration Decoding (IMCCD) 방법을 제안합니다. 기존의 방법들이 언어 선입견에 대한 과도한 의존성을 줄이기 위한 개선을 시행하는 동안, IMCCD는 훈련 없이 이 문제를 해결할 수 있도록 설계되었습니다. 이 접근법은 CMVED(Cross-Modal Value-Enhanced Decoding) 모듈을 통해 환각을 완화하는 새로운 대비 디코딩 메커니즘을 통합합니다.

- **Technical Details**: 제안된 방법은 CMVED를 포함하여 시각적 콘텐츠와 관련된 중요한 크로스 모달 주의(attention) 가중치를 마스킹(masking)하면서 왜곡된 분포를 추정하는 과정을 포함합니다. 이는 환각을 유발하는 스푸리어스(spurious) 상관관계를 조절하는 방식이며, 또한 CDAR(Content-Driven Attention Refinement) 모듈이 시각적 콘텐츠에 집중할 수 있도록 도움을 줍니다. 이 과정에서 단일 모달(uni-modality)에 대한 의존성을 방지합니다.

- **Performance Highlights**: 다양한 환각 벤치마크에서 실험을 통해 IMCCD가 기존의 최첨단 기법들보다 더 효과적으로 LVLM의 환각을 줄이는 것으로 나타났습니다. 이 새로운 접근법은 튜닝(tuning)이나 추가적인 학습 없이도 LVLM의 응용 가능성을 높이는 데 기여할 것으로 기대됩니다. 연구 결과는 LVLM에서의 환각 완화에 대한 효율성과 일반성을 명확하게 입증합니다.



### Mingling with the Good to Backdoor Federated Learning (https://arxiv.org/abs/2501.01913)
Comments:
          13 pages, 9 figures, under submission

- **What's New**: 이 논문은 Federated Learning (FL) 환경에서 backdoor 공격을 체계적으로 설치할 수 있는 MIGO라는 새로운 전략을 탐구합니다. MIGO는 악의적인 업데이트를 정상적인 업데이트와 혼합하여 백도어를 점진적으로 통합하는 기술을 사용합니다. 이로 인해 공격이 종료된 후에도 백도어 지속성을 보장하면서 방어 시스템의 효과를 저해하는 충분한 모호성을 생성할 수 있습니다.

- **Technical Details**: FL은 여러 장치에서 공동으로 딥러닝 모델을 학습하는 분산 학습 패러다임이며, 로컬 데이터셋을 공유하지 않아 데이터 프라이버시를 효과적으로 보장합니다. 서버가 클라이언트들을 선택하여 모델 업데이트를 진행하며, 각 클라이언트는 자신의 로컬 데이터셋을 사용하여 글로벌 모델을 훈련합니다. MIGO 전략은 정상 모델과 혼합된 악의적인 모델 업데이트를 생성하여 방어 메커니즘을 우회할 수 있도록 설계되었습니다.

- **Performance Highlights**: MIGO는 5개의 데이터셋과 다양한 DNN 아키텍처를 통해 세 가지 타입의 백도어를 삽입하는 실험에서 놀라운 성과를 보였습니다. MIGO는 백도어 정확도가 90%를 초과했으며, 대부분의 방어 메커니즘에 대해 뛰어난 회피 능력을 발휘했습니다. 또한, MIGO는 다른 네 가지 최신 공격 전략들과 비교했을 때 일관되게 우수한 성능을 보여주었습니다.



### Virgo: A Preliminary Exploration on Reproducing o1-like MLLM (https://arxiv.org/abs/2501.01904)
Comments:
          Technical Report on Slow Thinking with LLMs: Visual Reasoning

- **What's New**: 최근 대규모 언어 모델(LLM)을 기반으로 한 느린 사고(reasoning) 시스템이 주목받고 있으며, 이 연구에서는 다중모드 대규모 언어 모델(MLLM)에 이러한 능력을 적용할 수 있는 방법을 탐구합니다. 연구의 주요 결과로, 텍스트 기반의 긴 사고 데이터를 활용하여 MLLM을 세밀하게 조정(fine-tune)함으로써 다중모드 느린 사고 시스템인 Virgo가 개발되었습니다. 이를 통해 텍스트 기반의 합리적인 과정이 MLLM에 효과적으로 전이될 수 있음을 보여주며, 일반적으로 시각적 데이터보다 텍스트 데이터가 느린 사고 능력을 이끌어내는 데 더 효과적이라는 점을 밝혔습니다.

- **Technical Details**: 이 연구에서는 Qwen2-VL-72B-Instruct 모델을 중심으로 MLLM을 조정하기 위해 약 5,000개의 긴 사고 지침 인스턴스를 수집했습니다. 이 데이터는 주로 수학, 과학, 코드 및 퍼즐 도메인에 걸쳐 있으며, 텍스트 형식은 특별한 기호로 정의된 사고 과정과 최종 솔루션으로 나뉩니다. 실험은 MathVerse, MathVision, OlympiadBench 및 MMMU와 같은 4개의 도전적인 벤치마크에서 수행되어, Virgo가 상업적 추론 시스템과 비슷한 성능을 나타냄을 확인하였습니다.

- **Performance Highlights**: Virgo 시스템은 텍스트 기반의 긴 사고 데이터를 사용하여 놀라운 성과를 이루었으며, 상업적 느린 사고 시스템과 비교했을 때 동등하거나 그 이상의 결과를 보였습니다. 실험 결과, 일반적으로 텍스트 기반의 지침이 MLLM의 느린 사고 능력을 유도하는 데 더 효과적이라는 것이 확인되었습니다. 이 연구는 다중모드 느린 사고 시스템 개발에 있어 잠재적인 도전과 기회를 제시하며, 언어 모델의 특성과 전이 가능성에 대한 깊은 통찰을 제공합니다.



### QuArch: A Question-Answering Dataset for AI Agents in Computer Architectur (https://arxiv.org/abs/2501.01892)
- **What's New**: QuArch는 컴퓨터 아키텍처에 대한 언어 모델(LMs)의 이해도를 평가하고 향상시키기 위해 설계된 1500개의 인간 검증 질문-답변 쌍 데이터셋입니다. 이 데이터셋은 프로세서 설계, 메모리 시스템 및 성능 최적화와 같은 분야를 포괄하며, 기존 모델이 겪고 있는 성능 격차를 분석합니다. 특히, 최고의 폐쇄형 모델이 84% 정확도를 기록한 반면, 상위 소형 오픈 소스 모델은 72%에 불과하다는 점을 강조하고 있습니다.

- **Technical Details**: QuArch는 프로세서 실행, 메모리 계층 구조 및 병렬성 같은 핵심 개념에 대한 심도 있는 이해 없이 AI 드리븐 솔루션을 개선할 수 없는 현 상황을 해결하기 위해 고안되었습니다. 이 데이터셋은 기초 컴퓨터 아키텍처 원칙 및 최근 주제, 예를 들어, 딥 러닝 가속기 및 양자 컴퓨팅 아키텍처를 포함하고 있습니다. 데이터셋 구성 과정에서는 Archipedia라는 방대한 지식 자료를 근거로 삼아 고품질의 질문-답변 쌍을 생성하며, 각 질문은 전문가의 검증을 거치게 됩니다.

- **Performance Highlights**: 분석 결과, 다양한 언어 모델의 아키텍처 지식이 39%에서 84%까지 드러났으며, 소형 오픈 소스 모델과 대형 폐쇄형 모델 간에는 12%의 지식 격차가 존재하는 것으로 나타났습니다. 지식 검색 능력을 평가한 결과, 가장 높은 정확도를 기록한 모델은 84%에 이르렀으며, 이는 여전히 아키텍처 개념의 이해에 상당한 개선 여지가 있음을 시사합니다. QuArch를 활용한 미세 조정은 소형 모델의 정확도를 5.4%에서 8.3%까지 향상시키는 효과가 있었습니다.



### Evaluating Scenario-based Decision-making for Interactive Autonomous Driving Using Rational Criteria: A Survey (https://arxiv.org/abs/2501.01886)
- **What's New**: 이 논문은 자율주행차(AV)에 대한 심층 강화 학습(DRL) 알고리즘의 적용을 종합적으로 리뷰하며, 다양한 주행 시나리오에서의 의사 결정 방안을 평가합니다. 특히, 고속도로, 진입로 병합, 원형 교차로, 신호가 없는 교차로와 같은 특정 시나리오에 대해 논의합니다. DRL이 기존 규칙 기반 방법에 비해 복잡하고 동적인 주행 환경에서 더 적합하다는 점을 강조합니다.

- **Technical Details**: 이 연구는 AV가 환경에서의 상호작용을 통해 의사 결정을 학습하는 시스템을 설명합니다. 주요 요소는 인식, 계획 및 제어로 이루어진 운영 의사 결정 지원 시스템을 포함합니다. DDTUI(Driving Safety, Efficiency, Training Efficiency, Unselfishness, Interpretability)라는 다섯 가지 기준을 통해 DRL 알고리즘을 평가하는 방법론이 제시됩니다.

- **Performance Highlights**: DRL 기반 자율주행 시스템은 긴급 상황에서의 대처 능력이 입증되었습니다. 알고리즘은 다양한 주행 요구 사항을 충족하기 위해 각 주행 시나리오에 맞춤형으로 조정되어야 하며, 그 결과로 고속도로에서는 HDV와의 충돌 회피에 중점을 둡니다. 이러한 평가와 더불어 AV의 안전성 및 효율성 향상이 기대됩니다.



### LCFed: An Efficient Clustered Federated Learning Framework for Heterogeneous Data (https://arxiv.org/abs/2501.01850)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 LCFed라는 새로운 클러스터 연합 학습(CFL) 프레임워크를 제안합니다. LCFed는 사용자 정의 모델 파티셔닝(model partitioning) 방식을 채택하며, 각 서브 모델의 집계 전략을 구분함으로써 클러스터 내 공동 학습(intra-cluster co-training) 과정에 글로벌 지식을 통합합니다. 이를 통해 프레임워크는 데이터 이질성(data heterogeneity)으로 인한 성능 문제를 개선하고 최적의 교육 성능을 달성합니다.

- **Technical Details**: LCFed는 각 기기에서 개인화된 모델을 훈련하고, 이를 클러스터로 그룹화하여 클러스터 중심 모델을 학습합니다. 이 과정에서 모델 유사성 측정을 위해 저차원 모델(low-rank models)을 사용하는 계산 효율적인 방법을 사용자 정의하여, 서버 측 클러스터 업데이트의 연산 오버헤드를 최소화합니다. 이러한 방법은 클러스터 할당에 필요한 서버의 계산 부담을 줄이게 해줍니다.

- **Performance Highlights**: 실험 결과, LCFed는 최신의 벤치마크 모델들보다 테스트 정확도(test accuracy) 및 클러스터링 연산 효율(clustering computational efficiency)에서 우수한 성능을 보였습니다. 다양한 실제 데이터셋을 통해 LCFed의 전반적인 성능, 견고성(robustness), 및 계산 효율성을 입증하였습니다. 이러한 성과는 LCFed가 클러스터 연합 학습 분야에서 중요한 진전을 이룩했음을 시사합니다.



### Multi-Agent Conversational Online Learning for Adaptive LLM Response Identification (https://arxiv.org/abs/2501.01849)
- **What's New**: 이 논문은 사용자 선호에 따라 LLM(대형 언어 모델) 응답을 효율적으로 식별하는 온라인 학습 알고리즘, MACO(Multi-Agent Conversational Online Learning)를 제안합니다. 기존의 중앙 집중형 접근 방식에 비해 여러 로컬 에이전트를 활용하여 데이터 프라이버시를 강조하고, 생성된 응답에 대한 사용자 선호를 반영하는 대화 메커니즘을 도입합니다. 이러한 접근 방식은 LLM 응답 식별 과정의 불확실성을 최소화하는 데 기여합니다.

- **Technical Details**: MACO는 두 가지 주요 구성 요소, 즉 로컬 에이전트에 의해 실행되는 MACO-A와 클라우드 서버에 의해 수행되는 MACO-S로 나눌 수 있습니다. MACO는 기존의 사전 결정된 대화 빈도에 의존하지 않고 현재의 상황에 맞춰 적응적으로 대화 시점을 결정합니다. 이러한 분산형 대화 모델은 비선형 복잡성을 해결하며, LLM에 대한 온라인 식별을 가능하게 합니다.

- **Performance Highlights**: 이론 분석에 따르면 MACO는 누적 후회(잔여 손실)에서 거의 최적이며, 통신 비용 및 계산 복잡성을 줄입니다. 실험 결과에 따르면, 공개 LLM인 Llama와 Google 및 OpenAI의 두 가지 서로 다른 임베딩 모델을 통해 MACO는 온라인 LLM 응답 식별의 최신 기술을 능가하는 것으로 나타났습니다.



### Practical machine learning is learning on small samples (https://arxiv.org/abs/2501.01836)
- **What's New**: 이 논문은 기계 학습(Machine Learning)이 실제 상황에서 의존성을 식별하는 방식에 대한 새로운 관점을 제안합니다. 저자는 기계 학습이 무한히 증가하는 훈련 샘플에 의존해야 한다고 가정하는 통계적 학습 이론의 한계를 비판하며, 실제 학습은 데이터 포인트 간의 피드백의 급격한 변화가 없다는 암묵적 가정에 기반한다고 주장합니다. 이를 통해 저자는 '실용 학습 패러다임(Practical Learning Paradigm)'을 소개하며, 다양한 기계 학습 알고리즘들이 이 패러다임의 구현이라고 설명합니다.

- **Technical Details**: 논문에서 제안하는 실용 학습 패러다임은 여러 기술적 개념을 포함합니다. 객체의 특징은 두 가지 속성, 즉 숨겨진(hidden) 피드백과 드러난(manifested) 특징으로 나뉘며, 이들은 각각 수치적 값으로 표현됩니다. 저자는 학습할 의존성을 나타내는 함수와 사례의 정의, 훈련 세트의 형성 과정을 상세히 기술하고 있으며, 이러한 과정에서 '가설(hypothesis)'의 선택이 중요하다고 강조합니다.

- **Performance Highlights**: 저자는 다양한 인기 있는 학습 알고리즘(k-NN, 의사결정 트리, SVM 등)이 실용 학습 패러다임의 원칙을 따르고 있음을 보여줍니다. 이는 실제 데이터에서 효과적인 학습을 가능하게 하며, 기존의 통계적 학습 이론보다 더 현실적인 접근 방식을 제시합니다. 이로 인해 많은 실제 문제에 대한 해결책을 제공할 수 있는 가능성을 내포하고 있습니다.



### MoColl: Agent-Based Specific and General Model Collaboration for Image Captioning (https://arxiv.org/abs/2501.01834)
- **What's New**: 이번 논문에서는 복잡한 이미지 캡셔닝(Imaging Captioning) 작업에 대한 새로운 접근 방식을 제안합니다. 저자들은 도메인 특화 모델과 일반 지식을 효과적으로 통합할 수 있는 에이전트-강화 모델 협업 프레임워크인 MoColl을 소개합니다. 이를 통해 VQA(Visual Question Answering) 모델과 LLM(Large Language Model) 기반 에이전트를 결합하여 이미지 캡셔닝 작업을 보다 효율적으로 처리할 수 있습니다.

- **Technical Details**: MoColl 프레임워크는 복잡한 이미지 캡셔닝 작업을 일련의 질문-답변 하위 작업으로 분해합니다. 전문화된 VQA 모델은 특정 도메인에 대한 시각적 분석을 수행하는 도구 역할을 하고, LLM 기반 에이전트는 의미 있는 질문을 formulates하고 결과적 질문-답변 쌍을 통합하여 일관성 있는 캡션을 생성합니다. 이 과정에서 에이전트는 VQA 모델의 도메인 특화 능력을 향상시키기 위한 훈련 양식을 유도합니다.

- **Performance Highlights**: 방사선 보고서 생성을 위한 실험 결과는 제안된 MoColl 프레임워크의 효과성을 입증하며 생성된 캡션의 품질에서 상당한 개선을 보여줍니다. 이 발견은 복잡한 도메인 특화 작업에서 도메인 전문성과 일반 적응성을 연결할 수 있는 에이전트 기반 시스템의 잠재력을 강조합니다. 따라서 이 연구는 이미지 캡셔닝 분야에서의 새로운 가능성을 제시합니다.



### Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models (https://arxiv.org/abs/2501.01830)
- **What's New**: 이 논문에서는 Auto-RT라는 새로운 강화 학습 기반의 프레임워크를 제안합니다. 이는 공격 쿼리를 통해 보안 취약점을 효율적으로 발견하기 위한 복잡한 공격 전략을 탐색하고 최적화합니다. Auto-RT는 탐색 복잡성을 줄이고 전략 최적화를 개선하는 두 가지 핵심 메커니즘을 도입합니다: Early-terminated Exploration과 Progressive Reward Tracking 알고리즘입니다.

- **Technical Details**: 자동적인 red-teaming의 목표는 공격 모델 AM을 사용하여 공격 프롬프트를 생성하고, 이를 통해 타겟 모델 TM의 반응을 평가하는 것입니다. 공격 모델의 최적화 과정에서 여러 제약 조건을 추가하여 최적화를 실행하며, 이 논문에서는 전략적 red-teaming이라는 개념을 통해 공격 전략 생성 모델 AMg와 특정 공격 프롬프트를 생성하는 공격 재구성 모델 AMr로 나누어 설명합니다.

- **Performance Highlights**: Auto-RT를 통해 다양한 LLM에 대한 실험을 수행한 결과, 탐색 효율성을 크게 개선하고 공격 전략을 자동으로 최적화하여 더 넓은 범위의 취약점을 탐지할 수 있었습니다. 이 방법은 기존 방법에 비해 16.63% 높은 성공률을 달성하며, 빠른 탐지가 가능합니다.



### The Proof is in the Almond Cookies (https://arxiv.org/abs/2501.01827)
- **What's New**: 이 논문은 로봇이나 인공지능 요리 보조기가 주방에서 사람 요리사를 지원할 수 있도록 하는 조리 레시피 처리 방법에 대한 사례 연구를 제시합니다. 이러한 AI 보조기는 노인이나 신체적 제한이 있는 사람들의 자율성을 유지하고, 전문 요리사들의 스트레스를 줄이는 데 큰 도움이 될 것입니다. 특히 본 연구는 사람의 사고 방식을 모방한 서사 기반(narrative-based) 조리법 이해의 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문은 조리법을 서사로 취급하며, 이를 통해 구문론(syntax)과 의미론(semantics) 등의 언어 처리를 통합합니다. 조리법은 사건과 행동의 집합인 fabula, 인과 네트워크로 구조화된 plot, 그리고 서술 방식인 narration으로 구성됩니다. 본 연구는 이러한 세 가지 층위를 통해 요리 에이전트가 이해할 수 있는 구조적인 경로를 구축합니다.

- **Performance Highlights**: 이 연구의 주목할 만한 점은 조리법 언어의 복잡성을 다루면서 로봇의 계획 과정 최적화와 AI 시스템의 현재 작업 이해도를 측정하는 방법을 제시한다는 것입니다. 또한, 조리법 주석(annotation)이 언어 독립적으로 변환될 수 있도록 허용하는 과정을 설명합니다. 이러한 접근법은 요리 지식의 통합 및 다중 모드 언어 처리 기술을 사용하여 의미 있는 인간-로봇 상호작용을 지원하는 데 기여할 것입니다.



### End-to-End Long Document Summarization using Gradient Caching (https://arxiv.org/abs/2501.01805)
- **What's New**: 이 논문에서는 CachED(Gradient Caching for Encoder-Decoder models)라는 새로운 방법을 제안합니다. CachED는 기존의 transformer 기반 인코더-디코더 모델을 사용하여 문서의 전체 내용을 잘라내지 않고도 훈련할 수 있게 해줍니다. 이 접근법은 비오버랩 슬라이딩 윈도우를 활용하여 입력 문서를 처리하고, 디코더에서의 융합(fusion)을 수행합니다.

- **Technical Details**: CachED 방법은 인코더의 중간 결과를 메모리에 유지하지 않고, 최종 출력만을 보존합니다. 전방파 전파(backpropagation) 과정에서 그래디언트는 디코더에서 캐시(caching)되고 다시 인코더를 통해 청크(chunk)별로 전달됩니다. 이 방식은 메모리 사용량을 크게 줄여줍니다.

- **Performance Highlights**: CachED BART는 GovReport, SummScreenFD, QMSum, ScriptBase 및 BookSum과 같은 여러 긴 문서 요약 벤치마크에서 성능을 테스트하였고, 기존 방법들보다 우수한 성능을 기록했습니다. 추가 매개변수를 사용하지 않고도 500K 이상의 토큰을 처리할 수 있으며, 특히 1024 토큰의 컨텍스트 크기를 가진 작은 모델로도 뛰어난 결과를 보여주었습니다.



### BERT4MIMO: A Foundation Model using BERT Architecture for Massive MIMO Channel State Information Prediction (https://arxiv.org/abs/2501.01802)
Comments:
          10 pages

- **What's New**: 이 논문에서는 Massive MIMO 시스템에서 Channel State Information (CSI)의 예측을 위한 새로운 기초 모델인 BERT4MIMO를 제안합니다. BERT4MIMO는 Deep Learning과 Attention Mechanism을 활용하여 다양한 이동 시나리오 및 채널 조건에서 CSI를 재구성하는 데 탁월한 성능을 보입니다. 이 모델은 BERT에서 영감을 받아 고차원 CSI 데이터를 처리하도록 설계되었습니다.

- **Technical Details**: Massive MIMO 시스템은 많은 수의 안테나를 기반으로 하여 용량, 스펙트럼 효율성, 신뢰성 및 에너지 효율성을 향상시키는 기술입니다. 이러한 시스템에서 CSI는 채널 특성을 포착하는 중요한 요소로, 시스템의 성능은 CSI의 품질에 크게 의존합니다. BERT4MIMO는 MATLAB을 사용하여 다양한 시나리오에 대한 CSI 매트릭스를 시뮬레이션하여 새로운 데이터셋을 생성하고, 트랜스포머 아키텍처를 활용하여 CSI 재구성을 향상시키는 즉, CSI의 복잡한 패턴을 효과적으로 추출합니다.

- **Performance Highlights**: 실험 결과, BERT4MIMO는 다양한 무선 환경에서 높은 강인성을 보여주었으며, 도플러 이동 및 다양한 마스킹 비율과 같은 다양한 조건에서 평가되었습니다. 이러한 성능은 BERT4MIMO가 다가오는 차세대 네트워크에서의 MIMO 시스템 응용 프로그램에 매우 유용하며, CSI 예측의 질을 대폭 향상시킬 수 있는 가능성을 보여줍니다. 또한, 이 연구는 향후 무선 커뮤니케이션 및 센싱 시스템에서의 기초 모델 개발을 위한 방향성을 제시합니다.



### Creating Artificial Students that Never Existed: Leveraging Large Language Models and CTGANs for Synthetic Data Generation (https://arxiv.org/abs/2501.01793)
- **What's New**: 이 연구에서는 AI와 딥러닝 기술의 잠재력을 탐색하며, 특히 Generative Adversarial Networks (GANs)와 Large Language Models (LLMs)을 통해 합성 표 데이터(synthetic tabular data)를 생성하는 가능성을 검토합니다. 품질 높은 학생 데이터에 대한 접근은 학습 분석(learning analytics)을 발전시키기 위해 중요하지만, 개인 정보 보호 우려와 엄격한 데이터 보호 규정으로 인해 이러한 데이터의 접근이 제한되고 있습니다. 이에 따라 합성 데이터가 효과적인 대안이 될 수 있음을 제시합니다.

- **Technical Details**: 본 연구에서는 CTGAN이라는 인기 있는 GAN 모델과 GPT2, DistilGPT2, DialoGPT의 세 가지 LLM을 사용하여 합성 학생 데이터를 생성합니다. 생성된 합성 데이터는 실제 학생 데이터와 유사한 품질을 보여주며, 이는 강력한 가능성을 시사합니다. 다양한 유틸리티 평가 메트릭을 사용하여 합성 데이터의 통계적 및 예측 성능을 평가하고, 사용된 생성기 모델 간의 성과를 비교합니다.

- **Performance Highlights**: 연구 결과는 고품질의 합성 데이터 세트를 생성할 수 있는 방법들의 강력함을 뒷받침합니다. 특히 LLM의 성과를 강조하며, 학습 분석 분야에 합성 데이터 사용에 대한 귀중한 통찰력을 제공합니다. 이 연구는 학습 분석 데이터 생성을 위한 새로운 혁신적인 접근법으로의 연구 기초를 마련하는 것을 목표로 합니다.



### Can Synthetic Data be Fair and Private? A Comparative Study of Synthetic Data Generation and Fairness Algorithms (https://arxiv.org/abs/2501.01785)
- **What's New**: 이 연구에서는 Synthetic Data Generators (SDGs)가 프라이버시와 공정성 사이에서 균형을 이루는 최상의 방법을 탐구합니다. 기존 연구에서는 공정성과 프라이버시 간의 역관계가 있다고 여겨졌으나, 본 연구는 DEbiasing CAusal Fairness (DECAF) 알고리즘이 두 가지 간의 최적 균형을 달성했다고 발표합니다. 또한, 전처리 공정성 알고리즘을 사용한 경우 합성 데이터의 공정성이 실제 데이터보다 더 향상된다는 점을 발견했습니다. 이러한 결과는 공정성을 높이기 위한 유망한 접근 방식을 나타냅니다.

- **Technical Details**: 종합적인 방법론을 통해 3개의 실제 데이터셋으로부터 5개의 SDG를 사용하여 합성 데이터셋을 생성하고, 4개의 광범위하게 사용되는 프라이버시 메트릭을 통해 평가했습니다. 이후 합성 데이터셋을 기반으로 4개의 ML 모델을 훈련시키고 3개의 공정성 메트릭을 사용하여 모델의 공정성을 평가했습니다. 결과적으로 프라이버시와 공정성 간의 상충 문제를 분석하며, 다양한 SDG에 대한 연구 질문을 통해 과거 연구의 한계를 극복하고자 하였습니다.

- **Performance Highlights**: DECAF 알고리즘을 통해 프라이버시와 공정성의 균형이 잘 맞춰졌으나, 예측 정확도 측면에서는 유틸리티가 저하되었습니다. 합성 데이터로 생성된 데이터셋과 전처리 알고리즘의 결합이 실제 데이터와 공정성 알고리즘의 조합보다 더 나은 예측 결과를 도출하였음을 확인했습니다. 이 연구는 LA 연구 분야에서 프라이버시와 공정성을 모두 고려한 데이터 생성 및 알고리즘의 사용에 대한 중요한 정책적 시사점을 제공합니다.



### Quantifying A Firm's AI Engagement: Constructing Objective, Data-Driven, AI Stock Indices Using 10-K Filings (https://arxiv.org/abs/2501.01763)
Comments:
          43 pages, 5 tables, 3 figures, 1 appendix figure

- **What's New**: 이 논문은 기존 AI 관련 상장지수펀드(ETFs)에 대한 분석을 통해 주식 선택 기준이 모호하고 주관적인 판단에 의존한다는 점을 밝혀냈습니다. 새로운 접근법을 제안하며, 자연어 처리(NLP) 기술을 이용하여 2011년부터 2023년까지 NASDAQ에 상장된 3,395개 기업의 연례 10-K 보고서를 분석하여 AI 주식을 분류합니다.

- **Technical Details**: 본 연구는 AI 관련 용어의 빈도와 맥락에 따라 각 기업의 AI 관련성을 양적인 지표와 가중치 AI 점수로 정량화합니다. 이를 바탕으로 다양한 AI 투자 관점을 제공하는 네 가지 AI 주식 지수를 구축하였으며, 각각의 지수는 시간가중치 및 크기 가중치마다 편차를 달리합니다.

- **Performance Highlights**: 개발된 지수는 14개의 기존 AI 테마 ETFs 및 나스닥 종합 지수와 비교하여 위험-수익 프로필에서 견줄만한 성과를 보였습니다. 특히, AI에 더 많이 관련된 기업들은 ChatGPT 출시에 따른 긍정적인 초과 수익을 경험했으며, 평균 일일 수익과 위험 조정 지표에서도 더 높은 성과를 거두었습니다.



### Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation (https://arxiv.org/abs/2501.01743)
- **What's New**: 이 논문에서는 모호한 법적 개념을 해석하기 위한 새로운 Retrieval-Augmented Generation (RAG) 프레임워크인 ATRI를 소개합니다. 법적 전문가의 주석 작업을 자동화하여 과거 판례에서 관련 정보를 검색하고 모호한 개념을 해석하며, 법적 개념 포함성(Legal Concept Entailment)이라는 새로운 벤치마크를 제안합니다. 이 프레임워크는 대규모 법률 모델(LLMs)이 모호한 법적 개념을 이해하는 데 도움을 주고, 전문가의 고용 없이 생성된 개념 해석의 품질을 평가할 수 있게 합니다.

- **Technical Details**: ATRI 프레임워크는 다음 세 단계로 구성됩니다: 1) Retrieve: 모호한 개념을 언급한 판례를 검색합니다; 2) Filter & Extract: 판례에서 개념이 상세히 분석되는 경우를 필터링하고 그 결정 이유를 추출합니다; 3) Interpret: 추출된 이유를 바탕으로 LLMs를 사용하여 개념 해석을 생성합니다. 이 방법은 법률 판결의 엄밀한 용어를 기반으로 하여 정확한 문자열 매칭을 통해 판례를 검색합니다.

- **Performance Highlights**: 실험 결과, ATRI는 생성된 해석의 품질이 인간 전문가와 비교할 만하다는 것을 보여주었습니다. 자동 평가와 법률 전문가에 의한 다면적 평가 모두에서 LLM이 생성한 개념 해석이 모호한 개념의 이해를 지원하고 높은 품질을 달성할 수 있음을 입증하였습니다. 이러한 결과는 LLM이 법률 전문가들을 지원하여 모호한 개념 해석을 더욱 간소화할 수 있음을 시사합니다.



### How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models (https://arxiv.org/abs/2501.01741)
- **What's New**: 본 논문에서는 자동화된 LLM(대형 언어 모델) 독성 테스트 프레임워크인 EvoTox를 소개합니다. 이 프레임워크는 LLM의 독성 응답 생성 가능성을 정량적으로 평가하여, 기존의 정렬(alignment) 조치에도 불구하고 여전히 존재하는 독성 반응을 탐지할 수 있도록 설계되었습니다. EvoTox는 시스템의 응답을 더욱 독성적으로 유도하기 위해 Prompt Generator라는 두 번째 LLM과의 상호작용을 활용하는 반복적인 진화 전략을 채택합니다.

- **Technical Details**: EvoTox는 기초적인 프롬프트(seed prompt)로 시작해 매 반복마다 새로운 프롬프트(뮤턴트)를 생성하는 방식으로 작동하며, 이 과정에서 기존의 독성 분류기를 기반으로 한 자동화된 오라클을 사용합니다. 이러한 접근 방식은 LLM의 내부 정보에 대한 요구 없이 블랙박스 방식으로 운영됩니다. EvoTox는 다양한 프롬프트 진화 전략을 제공하며, 이러한 전략은 효과적인 변이 방향을 파악하는 데 도움이 됩니다.

- **Performance Highlights**: EvoTox의 평가 결과는 기존의 기준 방법과 비교하여 독성 수준 탐지에서 유의미하게 높은 효과성을 보였습니다. 실험에서 EvoTox는 랜덤 탐색(random search) 및 Jailbreak 기법 대비 각각 1.0 및 0.99의 효과 크기를 보였으며, 실행 시간 측면에서도 평균 22%에서 35%의 제한된 비용 오버헤드를 기록하였습니다. 또한, 인적 평가를 통해 EvoTox로 생성된 프롬프트의 유창성과 독성 수준이 경쟁 방식보다 상당히 우수하다는 결과를 도출했습니다.



### Augmentation Matters: A Mix-Paste Method for X-Ray Prohibited Item Detection under Noisy Annotations (https://arxiv.org/abs/2501.01733)
Comments:
          The manuscript has been ACCEPTED for publication as a regular paper in the IEEE Transactions on Information Forensics & Security

- **What's New**: 이번 연구에서는 노이즈가 있는 주석(annotations) 하에서 강력한 X-ray 금지 물품 탐지기를 학습하는 방법을 제안합니다. 기존의 방법들은 주석의 정확성을 가정하지만, X-ray 이미지에서의 물체 겹침으로 인해 정확한 주석을 얻기가 매우 어렵습니다. 이를 해결하기 위해, 저자는 데이터 증강(data augmentation) 관점에서 새로운 Mix-Paste 방법을 통해 복잡한 문제에 접근합니다.

- **Technical Details**: Mix-Paste 방법은 동일한 카테고리 레이블을 가진 여러 개의 아이템 패치를 섞어 원본 패치를 대체하여 새로운 이미지를 생성하는 방식입니다. 이 과정에서 X-ray 이미지의 물체 겹침을 모방하여 탐지 모델이 이미지를 더 잘 이해할 수 있도록 합니다. 또한, 이 모델은 혼합된 패치에서 발생할 수 있는 큰 손실(large losses)을 억제하기 위한 LLS 전략을 설계하여 모델의 학습 효과를 개선합니다.

- **Performance Highlights**: 저자들은 X-ray 데이터셋에서 노이즈가 있는 주석 하에 모델의 우수성을 입증하며, 일반 객체 탐지 작업에서도 성능 개선을 보여줍니다. 더불어, 이 결과들은 노이즈 주석 문제를 해결하기 위한 데이터 증강의 잠재력을 명확히 나타냅니다. 이 연구는 노이즈 주석을 다루기 위한 첫 번째 접근법으로써 특히 주목받고 있습니다.



### Combined Hyper-Extensible Extremely-Secured Zero-Trust CIAM-PAM architectur (https://arxiv.org/abs/2501.01732)
- **What's New**: 이번 논문에서는 대규모 기업을 위한 Combined Hyper-Extensible Extremely-Secured Zero-Trust (CHEZ) CIAM-PAM 아키텍처를 제안합니다. 이 아키텍처는 기존의 Customer Identity and Access Management 시스템의 복잡성을 해결하기 위해 설계되었습니다. 특히, AI와 클라우드 컴퓨팅의 발전에 따른 최신 사이버 위협에 대응하기 위한 adaptive 및 zero-trust 보안 프레임워크로의 패러다임 전환을 강조합니다.

- **Technical Details**: CHEZ PL CIAM-PAM 프레임워크는 federated identity management, password-less authentication 및 adaptive multi-factor authentication (MFA)을 통합해 주요 보안 공백을 해소합니다. 또한, microservice 기반의 PEP (Policy Entitlement Point), multi-layer RBAC (Role Based Access Control), 그리고 multi-level trust 시스템을 통해 더욱 체계적인 접근 제어를 구현합니다. 이 디자인은 end-to-end data encryption을 포함하고 있으며, 최첨단 AI 기반의 위협 탐지 시스템과 원활한 통합을 지원합니다.

- **Performance Highlights**: CHEZ 아키텍처는 기존 CIAM 시스템보다 높은 유연성과 보안성을 제공합니다. 따라서, 기업들이 높은 Return on Investment (RoI)를 확보할 수 있도록 도와주며, 규제 기준 준수를 위한 기능을 포함하고 있습니다. 이 접근 방식은 다양한 고객 인구 분포를 지원하면서도 사이버 공격에 대한 지속적인 보안을 보장합니다.



### LLMs & Legal Aid: Understanding Legal Needs Exhibited Through User Queries (https://arxiv.org/abs/2501.01711)
Comments:
          Accepted at AI for Access to Justice Workshop at Jurix 2024, Brno, Czechia

- **What's New**: 이번 연구는 체코의 Frank Bold가 진행한 실험을 기반으로 하여, 사용자들이 법적 질문을 해결하기 위해 GPT-4와 상호작용한 방식을 분석합니다. 2023년 5월 3일부터 7월 25일까지 1,252명의 사용자가 3,847개의 질문을 제출했습니다. 본 연구는 대규모 언어 모델(LLM)의 정확성이나 사실성 대신, 사용자 쿼리와 상호작용의 측면을 중점적으로 분석한 점이 특징입니다. 이를 통해 사용자들이 어떤 법적 요구를 가지고 있는지를 심층적으로 이해하고자 합니다.

- **Technical Details**: 실험은 GPT-4 모델을 활용하여 법적 질문에 대한 답변을 제공하는 형태로 진행되었습니다. 사용자는 질문을 제출하기 위해 계정을 생성하고, 단일 질문-단일 답변 방식으로 쿼리를 입력했습니다. LLM의 답변은 retrieval-augmented generation (RAG) 방식으로 생성되었으며, 사용자의 쿼리와 관련된 내용을 검색하고 결합하여 최종 답변을 작성하는 프로세스를 포함했습니다. 또한, 사용자는 제공된 답변에 대해 평가하고 피드백을 줄 수 있는 선택권이 있었습니다.

- **Performance Highlights**: 총 1,563명의 사용자 중 1,262명이 실제로 질문을 제출하였으며, 72%의 쿼리는 실험 첫 절반 동안 이루어졌습니다. 쿼리의 평균 길이는 약 224자였으며, 최대 8,499자까지 다양했습니다. 주간 실행 분석 결과에 따르면, 많은 쿼리가 근무 시간 동안 제출되었으며, 특히 금요일에 집중되는 경향을 보였습니다. 이러한 통계는 온라인 법률 지원 도구의 접근성을 잘 보여줍니다.



### MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders (https://arxiv.org/abs/2501.01709)
Comments:
          11 pages, 5 figures

- **What's New**: 본 논문은 복수의 비전 인코더의 독특한 능력을 하나의 효율적인 인코더 모델로 증류하는 새로운 프레임워크인 Mixture-of-Visual-Encoder Knowledge Distillation(MoVE-KD)을 제안합니다. MoVE-KD는 Low-Rank Adaptation(LoRA)과 Mixture-of-Experts(MoEs)를 활용하여 각 인코더 고유 특성을 유지하면서 선택적으로 지식을 활성화합니다. 또한, Attention 기반의 증류 전략을 사용하여 비전 인코더의 중요도를 고려하고, 시각적 토큰의 가치를 강조합니다.

- **Technical Details**: MoVE-KD는 여러 개의 Teacher 인코더로부터 지식을 효과적으로 전이하며, 이를 통해 단일 인코더의 효율성을 개선합니다. encoder adapters를 통해 다양한 Teacher 인코더의 출력 결과를 통합된 표현 공간으로 투영하여, 비전 토큰 간의 일치를 이루는 것이 핵심입니다. Mixture-of-LoRA-experts(MoLE) 구조를 통해 학습 과정에서 발생할 수 있는 갈등을 완화하며, 최종 목표는 텍스트 손실과 KD 손실을 최소화하는 것입니다.

- **Performance Highlights**: LLaVA 및 LLaVA-NeXT와 같은 유력한 VLM에서의 광범위한 실험 결과, MoVE-KD 방법이 기존 방법들에 비해 성능 및 효율성 면에서 상당한 향상을 보여주었습니다. 본 연구의 기여는 비전-언어 모델의 다중 비전 인코더 융합을 위한 MoVE-KD 프레임워크를 제안한 점과 중요 비주얼 토큰의 증류를 향상시키기 위한 Attention-guided KD 정규화를 도입한 점입니다. 이와 같은 접근은 다양한 인코더 간의 지식을 통합하여 향상된 성능을 달성하게 합니다.



### The Essence of Contextual Understanding in Theory of Mind: A Study on Question Answering with Story Characters (https://arxiv.org/abs/2501.01705)
Comments:
          17 pages, under review

- **What's New**: 이 논문은 Theory-of-Mind (ToM)의 중요성을 강조하며, 기존의 평가 방법이 사람이 갖고 있는 복잡한 사회적 관계와 상호작용을 반영하지 못한다는 점을 지적합니다. 새로운 벤치마크인 CharToM-QA를 도입하여, 고전 소설의 등장인물에 기반한 1,035개의 ToM 질문을 포함하고 있습니다. 이를 통해 LLM들이 효율적으로 사람의 정신 상태를 이해하는 능력을 평가합니다. 이 연구는 LLM이 인간보다 어떻게 성능이 떨어지는지를 보여줍니다.

- **Technical Details**: ToM은 인간의 심리적 능력으로, 타인의 마음 상태를 이해하고 해석하는 과정입니다. 연구진은 CharToM-QA 벤치마크를 개발하여 LLM의 ToM 이해 능력을 평가합니다. 이 벤치마크는 소설의 등장인물과 관련된 ToM 질문을 제시하며, 고유한 요구에 대한 이해가 더 깊어야 합니다. 평가 과정은 생성적 질문 응답(Generative QA)과 다중 선택 질문 응답(Multiple Choice QA)으로 나뉘며, GPT-4o를 평가자로 활용하여 대답의 질을 체크합니다.

- **Performance Highlights**: 실험 결과 GPT-4o는 다양한 ToM 차원에서 다른 LLM보다 뛰어난 성능을 보였습니다. 인간 참가자들은 소설을 읽었을 때 LLM보다 월등히 높은 성과를 보였으며, 읽은 길이가 길어질수록 정확도가 향상되었습니다. 반면 LLM의 성능은 다양한 줄거리 길이에 관계없이 큰 변화가 없었습니다. 이는 LLM이 복잡한 시나리오와 미세한 역사적 맥락 정보를 효과적으로 포착하지 못하고 있음을 나타냅니다.



### VidFormer: A novel end-to-end framework fused by 3DCNN and Transformer for Video-based Remote Physiological Measuremen (https://arxiv.org/abs/2501.01691)
- **What's New**: 이번 논문에서는 얼굴 비디오를 기반으로 한 원거리 생리 신호 측정을 위한 새로운 방식인 VidFormer를 소개합니다. VidFormer는 3차원 합성곱 신경망(3DCNN)과 Transformer 모델을 통합하여 기존의 깊이 있는 학습 방법들이 갖고 있던 데이터셋의 크기 간 조화로운 성능 부족 문제를 해결하고자 합니다. 이 프레임워크는 모듈 간의 정보 교환과 융합을 쉽게 할 수 있도록 설계되어 있습니다.

- **Technical Details**: VidFormer는 얼굴 비디오로부터 지역적(local) 및 전역적(global) 특징을 추출하기 위해 3DCNN과 Transformer를 각각 활용합니다. 또한, VidFormer는 시공간적(spatiotemporal) 주의 메커니즘을 채택하여 데이터의 시공간적 특징을 더욱 잘 포착할 수 있도록 개선되었습니다. 이 모델은 Stem, Local Convolution Branch, Global Transformer Branch, CNN and Transformer Interaction Module(CTIM), 그리고 rPPG 생성 모듈(RGM)으로 구성됩니다.

- **Performance Highlights**: 우리의 실험 결과, VidFormer는 UBFC-rPPG, PURE, DEAP, ECG-fitness, COHFACE와 같은 다수의 표준 데이터셋에서 현재 최첨단(SOTA) 방법들을 초월하는 성능을 보였습니다. 특히, DA-3DCNN과 ST-MHSA와 같은 새로운 글로벌 주의 메커니즘을 도입하여 모델이 비디오 데이터의다양한 차원적 특성을 효과적으로 포착할 수 있도록 하였습니다. 예를 들어, 이 프레임워크는 심박수(HR), 심박변동성(HRV), 호흡수(RR) 등을 정밀하게 측정하기 위한 기반을 제공했습니다.



### Adaptive Few-shot Prompting for Machine Translation with Pre-trained Language Models (https://arxiv.org/abs/2501.01679)
Comments:
          published to AAAI2025

- **What's New**: 최근에 제안된 Adaptive Few-Shot Prompting (AFSP) 프레임워크는 다양한 출처 입력 문장에 대해 적절한 번역 시연을 자동으로 선택하여 LLM의 번역 능력을 더 끌어내어 기계 번역(Machine Translation) 품질을 향상시킵니다. 이 프레임워크는 LLM의 임베딩을 기반으로 번역 시연 검색 모듈을 구축하여 상응하는 평행 번역 말뭉치에서 의미적으로 유사한 번역 시연을 효율적으로 검색합니다. 또한, AFSP는 번역 결과의 의미 일관성을 보장하기 위해 LLM이 여러 출력 후보를 생성하고 이를 재평가하는 과정을 포함합니다.

- **Technical Details**: 본 논문에서는 LLM의 임베딩 계층을 기반으로 한 하이브리드 시연 검색 모듈을 통해 더 의미적으로 관련된 번역 시연을 검색하는 방법을 구축합니다. 제안된 AFSP는 세 가지 유형의 임베딩—밀집 임베딩(Dense Embedding), 희소 임베딩(Sparse Embedding), 다중 벡터 임베딩(Multi-Vector Embedding)—을 활용하여 기계 번역 품질을 향상시키기 위한 입력 표현을 개선합니다. 최종적으로, LLM의 확률적 샘플링으로 인해 발생할 수 있는 의미적 편향을 완화하기 위해 다수의 출력 후보를 생성하고, 작은 언어 모델(SLM)을 기반으로 한 재평가 모델로 최적의 번역 결과를 선택합니다.

- **Performance Highlights**: 논문에서 제안한 AFSP 프레임워크는 5,528개의 병행 중국어-영어 문장으로 구성된 고품질 외교 중국어-영어 평행 데이터셋과 유엔 평행 말뭉치에서 연구되었으며, 그 성능이 정량적 및 정성적으로 입증되었습니다. AFSP를 통해 LLM의 의미적 일관성 및 번역 품질이 현저하게 개선되었음을 보여줍니다. 결과적으로, 이 연구는 LLM의 기계 번역 성능을 향상시키기 위한 새로운 접근 방식을 제시하며, 우리의 데이터셋이 기계 번역 연구의 경계를 넓히는 데 기여할 것으로 기대됩니다.



### BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction (https://arxiv.org/abs/2501.01664)
- **What's New**: 본 논문은 IoT 환경에서의 사이버 보안 위협을 해결하기 위해 새로운 침입 예측 프레임워크를 제안합니다. 본 프레임워크는 사전 훈련된 대형 언어 모델인 BART와 BERT를 활용하여 네트워크 트래픽을 예측하고 평가하는 혁신적인 접근 방식을 통해 비정상적인 패킷을 식별할 수 있습니다. 주요 목표는 악의적인 활동을 사전에 미리 파악하여 피해를 예방하는 것입니다.

- **Technical Details**: 제안된 프레임워크는 패킷 파싱 및 사전 처리와 함께, 다음 패킷을 예측하고 분류하기 위해 미세 조정된 두 개의 BART 모델과 BERT 모델을 사용합니다. BART는 과거 데이터를 기반으로 다음 패킷을 예측하며, BERT는 예측된 패킷의 관계를 평가합니다. 이 과정에서 패킷 헤더 분석을 통해 데이터의 사전 처리를 효율적으로 수행하고, IoT 장치의 보안 요구 사항을 충족하기 위해 MEC(다중 접속 엣지 컴퓨팅) 서버에서 구현됩니다.

- **Performance Highlights**: CICIoT2023 IoT 공격 데이터셋을 사용하여 수행된 평가 결과, 본 프레임워크는 인공지능 기반 사이버 보안 문제에 대한 예측 성능을 크게 향상시키며 98%의 높은 정확도를 달성했습니다. 이러한 성능 향상은 IoT 네트워크의 실시간 침입 탐지 및 대응 능력을 크게 높이고 있습니다. 이 연구는 LLM을 활용하여 네트워크 침입 예측을 시도한 최초의 사례로, 향후 IoT 보안 강화에 중요한 기여를 할 것으로 기대됩니다.



### EAUWSeg: Eliminating annotation uncertainty in weakly-supervised medical image segmentation (https://arxiv.org/abs/2501.01658)
- **What's New**: 이번 연구에서는 약한 주석(annotation) 방법인 Bounded Polygon Annotation(BPAnno)와 이를 활용한 학습 프레임워크 EAUWSeg를 제안합니다. 이 방법은 레지온을 두 개의 다각형으로 간단하게 레이블링함으로써 주석의 불확실성을 제거하고 더 안정적인 모델 훈련을 지원합니다. 이를 통해 기존 약한 주석 기반 방법들과 비교하여 성능이 뛰어나며 적은 주석 작업량으로도 우수한 결과를 제공합니다.

- **Technical Details**: EAUWSeg는 레이블이 부정확한 영역에서 더욱 신뢰할 수 있는 감독 신호를 제공하기 위해 상반된 다각형을 두 개의 별도 주석으로 취급하는 특별한 학습 메커니즘을 갖추고 있습니다. 이를 통해 적절한 픽셀의 카테고리 예측을 보장하면서 불확실한 픽셀들도 같은 카테고리로 유도하여 특징 표현의 일관성을 유지합니다. 이러한 방법은 주석의 불확실성을 줄이고, 모델 학습의 불안정함을 제거합니다.

- **Performance Highlights**: 실험 결과, EAUWSeg는 ISIC2017 및 Kvasir-SEG와 같은 기존의 약한 주석 분할 방법들보다 우수한 성능을 보이며, 완전 감독 방식과 비교했을 때 주석 작업량을 20% 미만으로 줄이는 성과를 달성했습니다. 이로 인해 비용 효율적인 의료 이미지 분할 솔루션으로서의 가능성을 보여주며, 다양한 의료 이미지 분할 모델에 적용 가능함을 입증했습니다.



### AVATAR: Adversarial Autoencoders with Autoregressive Refinement for Time Series Generation (https://arxiv.org/abs/2501.01649)
Comments:
          This work has been accepted to the SDM 2025 on December 20, 2024

- **What's New**: 본 연구에서는 AVATAR라는 새로운 프레임워크를 소개합니다. AVATAR는 Adversarial Autoencoders (AAE)와 Autoregressive Learning을 결합하여 시간 시계열 데이터의 생성과 관련된 독특한 문제를 해결합니다. 이 프레임워크는 감독 네트워크를 도입하여 시퀀스에서 조건부 분포를 학습하는 데 도움을 주며, 혁신적인 손실 함수인 distribution loss를 통해 효율성을 극대화합니다.

- **Technical Details**: AVATAR는 시계열 데이터의 고유한 동적 특성을 캡처하기 위해 모든 네트워크를 동시에 훈련하는 공동 훈련 메커니즘을 활용합니다. 이 방법은 Latent Representation의 집합적 후방 분포를 사전 Gaussian 분포와 정렬하는 데 도움을 주며, GRU 모델에서 배치 정규화(batch normalization)를 정규화 기법으로 채택하여 네트워크 성능을 최적화합니다.

- **Performance Highlights**: 다양한 실제 및 합성 다변량 시계열 데이터셋을 통해 AVATAR의 효과를 입증했습니다. 실험 결과, AVATAR는 현실적인 시계열 데이터를 생성하는 데 있어 기존의 벤치마크를 지속적으로 초과 달성했습니다. 이는 AVATAR가 시계열 생성 시 품질과 실제 유용성을 모두 향상시키는 데 기여함을 보여줍니다.



### HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding (https://arxiv.org/abs/2501.01645)
- **What's New**: 이 논문에서는 HLV-1K라는 새로운 대규모 비디오 벤치마크를 제시합니다. HLV-1K는 1,009개의 시간당 비디오로 구성되어 있으며, 다양한 고품질 QA(질문 응답) 및 MCQA(다중 선택 질문 응답) 쌍을 포함하고 있습니다. 이 벤치마크는 긴 비디오 이해 모델을 평가하기 위해 설계되었습니다.

- **Technical Details**: HLV-1K는 100,000개 이상의 프레임을 포함한 비디오의 복잡성을 해결하기 위해 시간 인식 쿼리와 다양한 주석을 포함하여 비디오의 미세한 관계를 명확하게 탐구합니다. 수집 단계에서 HD-VILA 데이터셋을 기반으로 하여 YouTube에서 1,500개 이상의 긴 비디오를 다운로드하고, 수동으로 저품질 비디오를 필터링했습니다. 이를 통해 엔터테인먼트, 영화, 여행 등 다양한 주제를 다루는 약 1,009개의 긴 비디오를 엄선했습니다.

- **Performance Highlights**: 현재의 최첨단 방법을 이용하여 HLV-1K를 평가한 결과, 다양한 수준과 여러 과제에서 깊은 긴 비디오 이해 능력을 테스트하는 데 유용하다는 점을 보여주었습니다. HLV-1K는 향후 긴 비디오 이해 작업을 촉진하고, 실시간 비디오, 회의 녹화, 영화에 대한 심층적인 이해를 가능하게 하는 모델의 발전을 이끌 것으로 기대됩니다.



### Artificial Intelligent Implications on Health Data Privacy and Confidentiality (https://arxiv.org/abs/2501.01639)
- **What's New**: 이 논문은 인공지능(AI)이 의료 분야에서 의료 진단, 개인화된 의학, 운영 효율성을 혁신하는 과정을 다루고 있습니다. AI의 도입으로 인한 혜택 외에도 환자 데이터 프라이버시와 윤리적 고려사항, 규제 준수와 같은 심각한 도전 과제가 발생하고 있음을 강조합니다. 또한, 건강보험이동성과책임법(HIPAA)을 통해 데이터 프라이버시와 보안을 확보하는 것이 중요하다는 점도 부각됩니다.

- **Technical Details**: 논문에서는 당뇨병성 망막병증(diabetic retinopathy), 종양학(oncology) 및 데이터 공유에 관한 논란 등 AI 응용 사례를 통해 AI 구현의 윤리적 및 법적 복잡성을 탐구합니다. 특히, AI 기반 건강 관리 시스템에서 강력한 보호 장치와 윤리 기준이 필요하다는 점을 강조합니다. 이 연구는 혁신을 촉진하면서도 환자의 신뢰와 프라이버시를 유지하는 균형 잡힌 접근 방식이 필수적임을 밝혀냅니다.

- **Performance Highlights**: AI의 잠재력을 책임감 있고 윤리적으로 활용하기 위해서는 지속적인 교육, 투명성 및 규제 프레임워크 준수의 중요성이 강조됩니다. 연구 결과는 AI와 의료 간의 상호작용이 의료 서비스의 질을 높일 수 있지만, 환자의 정보를 보호하기 위한 조치를 소홀히 해서는 안 된다는 점을 분명히 하고 있습니다.



### A non-ergodic framework for understanding emergent capabilities in Large Language Models (https://arxiv.org/abs/2501.01638)
- **What's New**: 이번 연구에서는 대형 언어 모델이 비에르고딕(non-ergodic) 시스템이라는 점을 증명하고, 능력 출현을 설명하기 위한 수학적 프레임워크를 제시합니다. 특히, Stuart Kauffman의 인접 가능성 이론(TAP)을 기반으로 한 접근 방식을 통해 능력의 출현 메커니즘을 제시합니다. 이는 기존 언어 모델 연구에서 진일보한 이해를 제공합니다.

- **Technical Details**: 연구에서는 TAP 방정식이 아키텍처, 훈련, 그리고 맥락의 제약 사항들과 어떻게 상호작용하여 모델의 능력을 형성하는지를 설명합니다. 특히, 의미 공간의 단계 전이(phase transitions)를 통해 자원 제약이 모델의 능력에 미치는 영향을 강조합니다. 실험 결과, 세 가지 다른 언어 모델에서 제약 상호작용과 경로 의존 탐색(path-dependent exploration)에 의해 능력이 불연속적으로 나타나는 것을 보여줍니다.

- **Performance Highlights**: 이 연구는 언어 모델의 출현(emergence)을 이해하기 위한 이론적 기초를 제공합니다. 또한, 향후 능력 출현을 이끌어낼 수 있는 아키텍처 개발에 대한 가이드를 제시하여, 향후 연구 및 응용에 기여할 가능성을 제시합니다.



### ICPC: In-context Prompt Compression with Faster Inferenc (https://arxiv.org/abs/2501.01625)
- **What's New**: ICPC(인-컨텍스트 프롬프트 압축)은 긴 프롬프트를 효과적으로 압축하여 LLM의 계산 비용과 메모리 오버헤드를 줄이는 새로운 방법론입니다. 기존의 프롬프트 압축 방식과 달리, ICPC는 단어가 프롬프트에 등장할 확률을 계산하고 정보 함수(information function)를 통해 각 단어가 담고 있는 정보를 평가하여 정보 손실을 최소화합니다. 이러한 과정은 압축 속도를 높이면서도 의미의 본질을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: ICPC의 핵심 메커니즘은 변환기 인코더(transformer encoder)를 사용하여 프롬프트의 단어를 평가하고, 이를 기반으로 필요 없는 토큰을 제거하여 문장을 압축합니다. 중복된 정보가 있는 토큰을 제거하기 위해, 문장 내에서 손실을 계산하여 중요하지 않은 단어를 선택적으로 삭제합니다. 이러한 방식은 기존의 LLM에 비해 더 적은 자원으로도 높은 성능을 유지할 수 있도록 합니다.

- **Performance Highlights**: 제안된 ICPC 방법은 다양한 텍스트 카테고리에 걸쳐 긴 문장을 효과적으로 압축하여 여러 NLP 작업에서 더 나은 성능과 속도를 달성함을 실험적으로 입증했습니다. AWS EC2 환경에서 실시한 실험들은 다른 최첨단 접근 방식과 성능을 비교 관찰하였으며, Wikipedia와 같은 긴 컨텍스트 데이터셋에서 뛰어난 결과를 나타냈습니다. 이로써 ICPC는 LLM의 장기적인 컨텍스트 처리 능력을 강화하는 데 기여합니다.



### Merging Context Clustering with Visual State Space Models for Medical Image Segmentation (https://arxiv.org/abs/2501.01618)
Comments:
          Our paper has been accepted by the IEEE Transactions on Medical Imaging. Our code can be found at this https URL

- **What's New**: 이 논문에서는 기존의 ViM(vision mamba) 모델의 한계를 극복하기 위해 Context Clustering ViM(CCViM)이라는 새롭고 효과적인 방법을 제안합니다. CCViM은 이미지 토큰을 서로 다른 클러스터로 구분하는 컨텍스트 클러스터링 모듈을 포함하여, 전역 및 지역(feature interactions) 피처 간의 상호작용을 동시에 극대화해 의학 이미지 세분화에서의 성능을 개선합니다. 이 방법은 Kumar, CPM17, ISIC17, ISIC18 및 Synapse와 같은 다양한 공공 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수한 성능을 보입니다.

- **Technical Details**: CCViM은 U자형 아키텍처를 채택하여, 고정 스캔 방식의 한계를 극복하고 다이나믹하게 지역의 시각적 컨텍스트를 캡처할 수 있도록 설계되었습니다. CCS6(layer)라는 새로운 기능을 도입해 전역 스캔 방향과 CC(layer)를 결합하여 피처 표현 능력을 향상시킵니다. 이로 인해 필드에서의 실제 수행 능력이 크게 향상되며, 각기 다른 지역 특성을 동적으로 잘 잡아낼 수 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 CCViM이 MedISeg의 여러 태스크에서 효율적이고 효과적으로 뛰어난 성능을 발휘함을 입증했습니다. 특히, 세포 세분화, 피부 병변 세분화 및 다기관 세분화와 같은 의료 분야에서 기존의 CNN 및 ViT 기반 모델들과 비교하여 더욱 우수한 성과를 보였습니다. 이러한 결과는 의료 영상의 자동화 및 정확성을 크게 향상시킬 것으로 기대됩니다.



### Google is all you need: Semi-Supervised Transfer Learning Strategy For Light Multimodal Multi-Task Classification Mod (https://arxiv.org/abs/2501.01611)
- **What's New**: 이번 연구에서는 디지털 이미지 데이터가 증가함에 따라 이미지 분류의 중요성이 커지고 있다는 점을 다룬다. 우리는 하나의 이미지에 대해 여러 개의 라벨을 동시에 할당할 수 있는 강력한 다중 라벨 분류 시스템을 제안한다. 이 시스템은 Convolutional Neural Networks (CNN)와 Natural Language Processing (NLP) 모델을 통합한 다중 모달 분류기를 포함하여, 이미지와 텍스트 정보를 결합해 더 높은 정확도를 추구한다.

- **Technical Details**: 제안된 모델은 이미지와 텍스트의 기능을 각각 추출하고 이를 효과적으로 융합하여 분류 작업에 활용하는 과정을 포함한다. 비전 모듈은 CNN을 통해 이미지에서 특징을 추출하고, NLP 모듈은 텍스트의 구조적 및 의미적 nuance를 mining하여 관련된 텍스트 기능을 생성한다. 이 두 모듈의 출력을 통합하는 기능 융합 모듈은 정보의 흐름을 개선하여 분류 성능을 높이는 중요한 역할을 한다.

- **Performance Highlights**: 초기 결과는 제안된 분류기가 높은 정확도와 효율성을 보임을 나타낸다. 이 모델은 자동 이미지 라벨링 시스템으로서의 잠재력이 크며, 특히 의료 영상과 같이 다중 라벨이 필요한 분야에서 실질적으로 유용하게 사용될 수 있다. 다중 라벨 분류는 이미지에 여러 개체가 포함될 수 있는 현실적인 문제를 해결하며, 모델이 라벨 간의 관계를 인식할 수 있게 해준다.



### Few-shot Implicit Function Generation via Equivarianc (https://arxiv.org/abs/2501.01601)
Comments:
          11 pages, 8 figures, 4 tables

- **What's New**: 이 논문에서는 Few-shot Implicit Function Generation이라는 새로운 문제 설정을 제안하며, 제한된 몇 가지 예시를 통해 다양한 기능적으로 일관된 Implicit Neural Representation (INR) 가중치를 생성하는 방법을 탐구합니다. EquiGen이라는 프레임워크를 통해 제한된 데이터에서도 기능적 유사성을 유지하면서 새로운 INR을 생성할 수 있는 방안을 제시합니다. 이 접근법은 불완전한 데이터 환경에서 효과적이며, 높은 품질의 생성 결과를 도출할 수 있습니다.

- **Technical Details**: EquiGen 프레임워크는 가중치를 중요하게 다루며, 세 가지 주요 단계로 구성됩니다: 1) Equivariant Encoder, 이는 대조 학습(contrastive learning)과 부드러운 증강(smooth augmentation)을 통해 가중치를 equivariant latent space로 프로젝션합니다. 2) Equivariance-Guided Diffusion, 이는 이전의 equivariant 특성(conditioned features)에 기초하여 현재 제너레이션 단계를 규제하는 프로세스입니다. 3) Controlled Perturbations for Diversity, 이는 고유의 여유 공간에서 균형 잡힌 변 perturbation을 적용하여 다양한 기능적 일관성을 갖춘 가중치를 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, EquiGen은 2D 이미지 및 3D 형상 INRs 데이터셋에서 다양성과 기능적 특성을 모두 성공적으로 유지하며 고품질의 가중치 생성을 달성했습니다. 이를 통해 한정된 샘플로도 확장성과 유연성을 갖춘 데이터를 생성하는 능력을 입증하였습니다. Ablation 연구는 각 모듈의 효과를 지지하는 결과를 보여주며, equivariance 개념의 중요성을 실증적으로 입증합니다.



### PSYCHE: A Multi-faceted Patient Simulation Framework for Evaluation of Psychiatric Assessment Conversational Agents (https://arxiv.org/abs/2501.01594)
Comments:
          The first two authors contributed equally

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 인간과 유사한 반응을 생성할 수 있는 대화형 에이전트의 개발을 가속화했습니다. 이 연구에서는 정신과 평가 대화형 에이전트(PACAs)를 위한 새로운 프레임워크인 PSYCHE를 제안하여 임상 평가에서 정신과 의사의 역할을 시뮬레이션하는 방법을 다룹니다. PSYCHE는 PACAs의 임상 적합성을 1) 임상적 관련성, 2) 윤리적 안전성, 3) 비용 효율성 및 4) 정량적 평가를 통해 평가할 수 있는 구조를 제공합니다.

- **Technical Details**: PSYCHE 프레임워크는 다면적 정신과 구조(MFC)를 기반으로 환자 발화를 시뮬레이션하고 PACAs를 평가하는 방식으로 설계되었습니다. 이 프레임워크는 네 가지 단계로 구성되며, 사용자 입력, MFC 생성, 발화 시뮬레이션 및 평가 세션으로 이어집니다. 각 단계는 PACA의 성능을 평가하기 위한 체계적 프로세스를 통해 PSYCHE-SP를 생성하여 환자와의 상호작용을 가능하게 하며, 이 결과는 PACA의 성과 지표인 PSYCHE SCORE로 나타납니다.

- **Performance Highlights**: 연구 결과, 10명의 정신과 의사가 평가한 PSYCHE-SP는 다양한 정신 장애를 시뮬레이션하는 데 있어서 높은 일관성을 보였습니다. 총 7가지 장애에 대해 평균 93%의 적합성을 달성하였으며, 주요 우울 장애(MDD)와 사회 불안 장애(SAD)에서 각각 97%의 가장 높은 적합성을 기록했습니다. PSYCHE는 임상 환경에서 PACAs의 성능 평가를 효과적으로 진행할 수 있는 가능성을 보여주며, 향후 정신 건강 분야의 자동화와 효율성을 높이는 데 기여할 것으로 기대됩니다.



### (WhyPHI) Fine-Tuning PHI-3 for Multiple-Choice Question Answering: Methodology, Results, and Challenges (https://arxiv.org/abs/2501.01588)
- **What's New**: 이번 연구는 Microsoft의 PHI-3 모델을 활용하여 다중 선택 질문(MCQ)에 대한 답변 능력을 향상시키는 방법을 탐구합니다. 본 논문에서는 TruthfulQA 데이터셋을 사용하여 모델을 세밀하게 조정하고, 최적화된 프롬프트를 설계하여 모델 성능을 개선했습니다. 결과적으로, PHI-3 모델의 MCQ 처리 능력이 대폭 향상되었으며, 이는 교육적 응용 프로그램에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: 연구 방법론은 데이터 전처리, 프롬프트 설계, 모델 세부 조정(fine-tuning), 그리고 평가를 포함합니다. 이를 위해 TruthfulQA 데이터셋을 사용하여 1,000개의 MCQ를 기준으로 입력 형식을 표준화했습니다. PHI-3 모델의 정확한 답변 생성을 돕기 위해 Alpaca 스타일의 프롬프트와 기본 텍스트 보완 프롬프트를 결합하여 성능을 개선했습니다.

- **Performance Highlights**: 모델 세부 조정 후 PHI-3는 perplexity가 4.68에서 2.27로 감소하고, 정확도는 62%에서 90.8%로 상승했습니다. 이러한 결과는 세밀하게 조정된 소형 언어 모델이 교육적 과제 처리에서 매우 효과적일 수 있음을 보여주며, 특히 자원 제약이 있는 환경에서의 성공적인 적용 가능성을 나타냅니다.



### Constructing and explaining machine learning models for chemistry: example of the exploration and design of boron-based Lewis acids (https://arxiv.org/abs/2501.01576)
Comments:
          Main text is 12 pages, 5 figures, 3 extended-data figures. Supplementary information is 25 pages. For associated code and datasets, see this https URL

- **What's New**: 이 논문에서는 화학 분야에 기계 학습(ML)을 접목하여 분자 설계를 혁신할 수 있는 가능성을 제시합니다. 특히, 기존의 효율적인 예측 모델 구축에 초점을 맞추는 대신, 설명 가능한 AI 기술을 활용하여 붕소 기반 루이스 산(boron-based Lewis acids)의 설계를 탐구합니다. 이러한 접근은 화학적 설명력을 결합하여 ML과 화학자의 사고 방식을 연결짓는 새로운 방법론을 제공합니다.

- **Technical Details**: 우리는 Fluoride Ion Affinity를 루이스 산도의 대리 변수로 사용하여, ab initio 특징(ab initio features)과 치환기 기반 매개변수(substituent-based parameters)와 같은 화학적으로 의미 있는 설명자를 기반으로 해석 가능한 ML 모델을 개발했습니다. 이는 잘 정의된 분자 구조화(molecular scaffolds)로 화학적 공간을 제약하여, 데이터가 적을 때도 일반적인 블랙 박스 심층 학습 모델을 초월하는 높은 예측 정확도를 달성했습니다.

- **Performance Highlights**: 모델의 설명 가능성 분석을 통해 이러한 화합물의 루이스 산도 근원을 밝혀내고, 이를 조절할 수 있는 실행 가능한 전략을 식별하였습니다. 이 연구는 ML 모델의 설명 가능성이 분자 설계를 자극할 수 있으며, 화학 반응성에 대한 과학적 이해를 향상시킬 수 있음을 보여줍니다.



### BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery (https://arxiv.org/abs/2501.01540)
Comments:
          KG and MYL contributed equally

- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)에 기반한 과학적 에이전트의 능력을 평가하기 위한 새로운 벤치마크인 BoxingGym을 도입합니다. 10개의 환경을 통해 실험 설계와 모델 발견 능력을 체계적으로 테스트할 수 있습니다. 이를 통해 LLM이 과학 모델을 제안하고 실험 데이터를 수집하며, 새로운 데이터에 따라 모델을 수정하는 능력을 평가하는 시스템이 마련되었습니다.

- **Technical Details**: BoxingGym은 생성적 확률 모델을 사용하여 각 환경을 모델링하며, 과학 에이전트는 인터랙티브한 실험을 수행할 수 있습니다. 연구는 Bayesian Optimal Experimental Design(BOED)과 기대 정보 이득(Expected Information Gain, EIG)을 결합하여 실험 디자인과 모델 발견을 통합적으로 평가합니다. 이를 통해 LLM의 설계된 환경에서 정보 검색 및 실험적 설계를 체계적으로 수행할 수 있는 방법론론을 제시합니다.

- **Performance Highlights**: 현재의 LLM인 GPT-4o는 실험 설계 및 모델 발견에 있어 어려움을 겪고 있으며, 통계 모델을 추가하는 것으로도 이러한 문제를 해결하기 어려운 결과를 보여줍니다. 연구는 LLM의 잠재력을 드러내는 동시에, 실험적 설계와 모델 발견의 통합적 접근 방식을 강조하였습니다. 이는 앞으로 AI 모델이 과학적 발견 과정을 지원할 수 있는 보다 튼튼한 프레임워크 개발을 위한 기반이 됩니다.



### In Search of a Lost Metric: Human Empowerment as a Pillar of Socially Conscious Navigation (https://arxiv.org/abs/2501.01539)
Comments:
          9 pages, 8 figures, 2 tables, Accepted to 20th edition of the IEEE/ACM International Conference on Human-Robot Interaction (HRI)

- **What's New**: 이 논문은 사회적 로봇 내비게이션의 새로운 평가 기준으로 'human empowerment'라는 정보 이론적 개념을 제안합니다. 이 지표는 인간이 미래 상태에 영향을 미치는 능력을 측정하며, 로봇의 내비게이션 정책이 인간의 자율성에 미치는 간접적인 영향을 밝혀냅니다. 이를 통해 로봇 행동의 사회적 적합성을 평가하는 새로운 방법을 제시합니다.

- **Technical Details**: 연구에서는 2차원 환경에서 로봇과 인간 보행자가 함께 작동하는 내비게이션 시나리오를 다루며, 로봇의 행동이 인간의 행동 양식에 미치는 영향을 분석합니다. 'human empowerment'는 로봇의 사회적 성과 평가에 통합되어 있으며, 기존의 proxemics 기반 지표와 비교하여 동적 환경에서 사회적 행동을 보다 포괄적으로 포착할 수 있는 가능성을 보여줍니다. 다양한 로봇 내비게이션 정책에 대한 통계적 검정을 통해 이 지표의 효과성을 입증합니다.

- **Performance Highlights**: 실험 결과, 'human empowerment'는 직관적인 사회적 행동과 일치하며, 각기 다른 로봇 내비게이션 정책 사이에서 통계적으로 유의미한 차이를 보였습니다. 기존 평가 기준들이 미비한 점을 보완하여, 로봇의 사회적 적합성과 인간 자율성에 미치는 영향을 더욱 정교하게 이해할 수 있는 기회를 제공합니다. 이는 향후 사회적 내비게이션에 대한 연구에 있어 유망한 보완 지표로 작용할 수 있음을 시사합니다.



### A Metasemantic-Metapragmatic Framework for Taxonomizing Multimodal Communicative Alignmen (https://arxiv.org/abs/2501.01535)
Comments:
          34 pages, 1 figure, 3 tables. Draft presented at 2023 ZJU Logic and AI Summit EAI Workshop

- **What's New**: 이 논문은 현대의 실용주의 철학과 인지, 의미, 커뮤니케이션에 대한 언어 이론을 바탕으로 인간과 유사한 다중 모달 커뮤니케이션 정렬을 정립하는 동적 메타 의미론-메타 실용주의 세분화를 제시합니다. Charles Sanders Peirce에 의해 처음 제안된 세 가지 기본적인 커뮤니케이션 능력인 아이코닉(iconic), 인덱시컬(indexical), 룰 라이크(rule-like)에 대한 현대적 발전을 토대로 진행됩니다.

- **Technical Details**: 세 가지 커뮤니케이션 능력의 발전을 바탕으로, 인덱시컬 맥락화(indexical contextualization) 개념이 도입되며, 다중 모달 커뮤니케이션의 의미적(semantic) 및 실용적(pragmatic) 모드를 유지하고 탐색하는 데 필요한 메타 실용주의 능력인 '맥락화 방향성(contextualization directionality)' 원칙이 제안됩니다. 이 논문은 현재의 인지-사회적 컴퓨팅(cognitive-social computational) 방법론이 의미론/메타 의미론 영역에 집중하고 있음을 비판하며, 메타 실용주의 인덱시컬리티의 중요성을 강조합니다.

- **Performance Highlights**: 논문의 방법론은 의도(intentionality), 정체성(identity), 정서(affect), 윤리(ethics)와 같은 광범위한 주제에 대한 넓은 함의를 가지고 있으며, 이는 인간-기계 간의 내적 모달과 교차 모달 정렬에서 중요한 역할을 할 수 있습니다. 이 연구는 커뮤니케이션의 의미-실용 스펙트럼을 탐험하는 데 있어 메타 실용주의의 중심 역할을 부각시키고 있습니다.



### Improving Robustness Estimates in Natural Language Explainable AI though Synonymity Weighted Similarity Measures (https://arxiv.org/abs/2501.01516)
Comments:
          10 pages, 2 figures, 4 tables

- **What's New**: 이 논문에서는 Explainable AI (XAI) 기술에 있어 적대적 예제(adversarial examples)가 중요한 역할을 한다고 강조합니다. 기존의 유사도 측정 방법들이 적대적 XAI에서 효과적인 비교를 제공하지 못한다는 점을 지적하고, 동의어 가중치(synonymity weighting)를 도입하여 이러한 문제를 해결하고자 합니다. 이 접근 방식은 XAI 방법의 내구성 및 안정성, 즉 안정성을 평가하는 새로운 기준을 제시합니다.

- **Technical Details**: 문헌에서는 XAI 모델의 설명이 복잡한 블랙박스 모델의 출력을 이해하기 위한 방법으로 사용되지만, 기존의 유사도 측정 지표들이 두 가지 주된 결함(sensitivity와 indifference) 때문에 신뢰성이 떨어진다고 설명합니다. 이 연구는 동의어 가중치를 통해 perturbed 단어와 원본 단어 간의 의미적 유사성을 고려하여 단순한 비교 방식을 개선합니다. 이러한 접근법은 텍스트 기반 입력에서의 XAI의 적대적 공격 과정을 평가하기 위한 새로운 기초를 제공합니다.

- **Performance Highlights**: 동의어 가중치를 적용한 유사도 측정은 기존의 일반적인 유사도 측정보다 훨씬 더 정확한 결과를 도출합니다. 실험 결과, 전통적인 유사도 측정 방법으로는 XAI의 불안정성을 잘못 평가할 수 있음을 보여주며, 새로운 방법이 XAI 시스템의 신뢰성을 높이는 데 기여할 수 있다는 결론을 도출합니다. 이를 통해 XAI 방법이 적대적 예제에 대해 보다 강건함을 가질 수 있도록 하는 실질적인 기여를 하고 있습니다.



### DiagrammaticLearning: A Graphical Language for Compositional Training Regimes (https://arxiv.org/abs/2501.01515)
- **What's New**: 이 논문에서는 여러 개의 상호 작용하는 모델 구성 요소들을 사용하는 딥 러닝 체제를 바탕으로 학습 다이어그램을 소개합니다. 학습 다이어그램은 코드가 아닌 데이터로 매개변수화된 학습을 포착하는 그래픽 표현입니다. 이 개념을 통해 사용자들은 복잡한 모델을 더 작고 구성 요소들로 구성할 수 있습니다.

- **Technical Details**: 학습 다이어그램은 고유한 손실 함수(loss function)로 컴파일되며, 이 손실 함수에서 훈련된 모델들은 서로 '합의'하는 예측을 생성합니다. 논문에서는 few-shot multi-task learning, knowledge distillation, multi-modal learning과 같은 인기 있는 학습 설정을 학습 다이어그램으로 표현할 수 있음을 보여줍니다. 또한, PyTorch 및 기타 모델의 다이어그램을 구축할 수 있는 라이브러리를 구현하였습니다.

- **Performance Highlights**: 전통적인 머신 러닝 사용 사례를 실행하여 학습 다이어그램이 사용자들이 복잡한 모델을 구축하고, 워크플로우 간의 관계를 확인하며, 훈련 중 또는 훈련 후에 모델을 조작하는 방법을 보여줍니다. 범주 이론적(framework) 틀을 활용하여 학습 다이어그램에 대한 엄격한 의미론을 도입하였으며, 이는 이러한 작업을 수학적으로 탄탄한 기반 위에 놓이게 합니다.



### AI-Enabled Operations at Fermi Complex: Multivariate Time Series Prediction for Outage Prediction and Diagnosis (https://arxiv.org/abs/2501.01509)
Comments:
          Presented in the AAAI Workshop on AI for Time Series Analysis 2025

- **What's New**: 이 논문에서는 Fermilab 가속기 컴플렉스에서의 예기치 않은 사건으로 인한 빔 중단 현상에 대한 예측 및 라벨링을 위한 AI 기반 프레임워크를 제안합니다. 기존의 임계값 기반 경고 시스템은 반응적이며 허위 경고가 자주 발생하고, 중단 원인 라벨링이 일관되지 않아 문제를 야기합니다. 새로운 접근법으로는 예측 분석과 자동 라벨링을 결합하여 장애 예측 성능을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 2703개의 Linac 장치에서 수집한 시간 시계열 데이터를 활용하며, 고급 딥러닝 아키텍처를 평가합니다. 이 연구는 순환 신경망(Recurrent Neural Networks), 주의 기반(Attention-based) 모델 및 선형 모델을 포함한 딥러닝 아키텍처의 성능을 테스트합니다. 또한, 랜덤 포레스트(Random Forest) 기반의 라벨링 시스템을 평가하여 일관성 있고 신뢰할 수 있는 장애 주석을 제공하는 방안을 모색합니다.

- **Performance Highlights**: 이 연구의 결과는 제안된 AI 프레임워크가 다운타임을 줄이고 결정 결정의 질을 개선하는 데 도움을 줄 수 있음을 보여줍니다. 연구 결과는 다양한 딥러닝 아키텍처의 강점과 약점을 강조하며, 적절한 라벨링을 통해 다운타임 관리에 있어 AI의 활용 가능성을 극대화할 수 있는 방법을 제시합니다.



### Transfer Learning Analysis of Variational Quantum Circuits (https://arxiv.org/abs/2501.01507)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 논문은 변분 양자 회로(Variational Quantum Circuit, VQC)의 전이 학습(transfer learning)을 분석합니다. 우리가 제안하는 프레임워크는 하나의 도메인에서 사전 훈련된 VQC로 시작하여, 새로운 도메인에서 필요한 1-매개변수 유니타리(1-parameter unitary subgroup) 그룹의 전이를 계산합니다. 기존 연구와의 차별점은 우리가 사전 훈련된 VQC에서 시작하여 기존 데이터를 기억하면서 새로운 데이터를 다룰 수 있도록 최적의 양자 매개변수를 활용한다는 점입니다.

- **Technical Details**: 논문에서는 변분 양자 알고리즘(Variational Quantum Algorithms, VQAs)을 통해 양자 신경망(Quantum Neural Networks, QNNs) 아키텍처를 다양한 머신러닝(Machine Learning, ML) 작업에 적용할 수 있도록 하는 방법을 설명합니다. VQC는 외부 신호에서 학습할 수 있는 조정 가능한 매개변수를 가진 양자 게이트로 구성되어 있으며, 고전적 데이터와 함께 작동합니다. 본 연구는 비슷한 데이터 세트 간의 VQC 모델 매개변수를 조정하는 분석적 솔루션을 도출합니다.

- **Performance Highlights**: 연구 결과는 변분 양자 회로의 전이 학습 메커니즘을 밝히고, 이에 대한 문서화된 물리적 의미를 제공합니다. 특히, 본 방법론은 기존의 경량 기계 학습 기법과 비교할 때 매우 효율적임을 보여줍니다. 전이 학습을 통해 VQC가 새로운 데이터 도메인에 적응하는 능력을 나아가면, 이는 양자 컴퓨팅과 고전적 머신러닝의 융합을 통해 더 나은 성능을 가능하게 할 것입니다.



### ORACLE: A Real-Time, Hierarchical, Deep-Learning Photometric Classifier for the LSS (https://arxiv.org/abs/2501.01496)
Comments:
          29 pages, 19 figures, 9 tables. Submitted to ApJ

- **What's New**: 이 논문에서는 ORACLE이라는 최초의 계층적 깊이 학습 모델을 제시합니다. ORACLE은 실시간 및 상황 인식적인 분류를 수행하며, 광학 관측 정보를 바탕으로 고신뢰 기분류(classification)를 제공합니다. 이 모델은 Gated Recurrent Units (GRUs)를 사용하는 순환 신경망(recurrent neural network)으로 구축되었으며, 고유한 계층적 교차 엔트로피 손실 함수(custom hierarchical cross-entropy loss function)를 통해 훈련되었습니다.

- **Technical Details**: ORACLE은 약 50만 개의 이벤트로부터 훈련되었으며, 기본적으로 1일간의 광도 관측 정보와 상황적 정보를 통해 고성능의 분류 결과를 도출합니다. 이 모델은 64일의 데이터를 확보했을 경우 정확도가 99%를 초과하고, 1024일 후에는 19종 분류에서 83%의 성능을 기록합니다. 또한, ORACLE은 다른 최첨단 분류기들과 비교했을 때 비슷한 성능을 보이며 빠른 분류 결과를 제공합니다.

- **Performance Highlights**: ORACLE은 1일간의 관측 정보로도 뛰어난 분류 성능을 보이고, 64일 시점에서 99%가 넘는 정확도를 달성하는 성과를 보여줍니다. 이 논문에서는 ORACLE의 성능을 더 깊이 탐구하고, 기존의 분류기들과의 비교를 통해 그 효과성을 입증하였습니다. 연구진은 ORACLE의 코드와 모델 가중치가 GitHub에 공개되어 있어 커뮤니티에서 활용할 수 있음을 강조합니다.



### Drift2Matrix: Kernel-Induced Self Representation for Concept Drift Adaptation in Co-evolving Time Series (https://arxiv.org/abs/2501.01480)
- **What's New**: 이 논문의 주제인 Drift2Matrix는 시간 시계열 데이터에서 개념 변화를 다루는 새로운 프레임워크입니다. 이는 커널 기반의 자기 표현 방법을 활용하여 복합적이고 동적인 시계열 데이터 분석에 적응할 수 있는 능력을 제공합니다. Drift2Matrix는 기존의 정적인 모델에서 발생하는 한계를 극복하고, 동시 변화하는 시계열 간의 상호작용을 캡처하여 보다 효과적으로 개념 변화를 식별하고 적응할 수 있게 합니다.

- **Technical Details**: Drift2Matrix는 커널 유도 자기 표현(kernel-induced self-representation) 기법을 사용하여 시간 시계열 데이터를 행렬 형태로 변환합니다. 이 방법은 시계열 데이터 내의 비선형 상관관계를 포착하여 개념 변화를 추적하고 예측하는 데 유용합니다. 또한, 이 프레임워크는 개념 추적, 예측 및 동태 분석을 위한 세 가지 주요 목적을 설정하여 개념 식별의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, Drift2Matrix는 각기 다른 데이터 세트에서 개념 변화를 효과적으로 다룰 수 있는 능력을 증명했습니다. 예를 들어, 예측에서 Drift2Matrix는 새로운 개념의 출현을 예측할 수 있었으며, 이는 시계열의 동적 상호작용을 고려하는 확률 모델에서 기인합니다. 이러한 결과는 Drift2Matrix가 전통적인 방법에 비해 동적 데이터 환경에서 더 나은 적응성과 정확성을 보여주는 것을 확인시켜줍니다.



### A Survey of Deep Learning Methods in Protein Bioinformatics and its Impact on Protein Design (https://arxiv.org/abs/2501.01477)
Comments:
          PhD Qualifying Exam (2021)

- **What's New**: 이 논문에서는 단백질 생물정보학에서의 딥러닝 기법의 적용을 광범위하게 검토하고, 특히 단백질 디자인 문제에서의 도전과제를 강조합니다. 저자들은 단백질 구조 예측, 기능 예측, 단백질 디자인의 세 가지 주요 문제를 분류하고 각 분야에서의 진전을 논의합니다. 또한, 최근 딥러닝 기술을 사용한 성과들이 어떻게 단백질 디자인에 기여했는지를 다룹니다.

- **Technical Details**: 딥러닝 기법은 데이터 세트에서 관련 특징을 자동으로 학습할 수 있는 능력으로 정의됩니다. 특히, 단백질 구조 예측 문제에 대한 딥러닝 적용은 Deep Mind의 Alphafold2 모델처럼 3D 구조 예측 정확도를 실험 검증 방법과 비슷하게 끌어올리는 데 성공했습니다. 또한, 단백질 기능 예측, 단백질-단백질 상호작용(PPI) 및 결합 부위 식별 등에서도 비슷한 진전이 이루어졌습니다.

- **Performance Highlights**: 딥러닝은 전통적인 모델링 기법을 지속적으로 뛰어넘는 성과를 보이고 있으며, 이는 단백질 디자인 등 사회적 이익이 큰 문제에 대해 더 큰 잠재력을 제공합니다. 특히, 단백질 디자인 문제는 자연에서 드물게 발생하는 서열이나 구조를 발견해야 하기 때문에 도전적이며, 이러한 문제 해결을 통해 약물 디자인 및 질병 치료와 같은 분야에서 큰 사회적 이익을 가져올 수 있습니다.



### Unraveling Indirect In-Context Learning Using Influence Functions (https://arxiv.org/abs/2501.01473)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 일반화된 In-Context Learning (ICL)을 위한 새로운 패러다임인 간접 ICL(Indirect In-Context Learning)을 소개합니다. 간접 ICL에서는 Mixture of Tasks와 Noisy Demonstrations이라는 두 가지 실제 시나리오에 맞춘 demonstration selection 전략을 탐구합니다. 본 연구에서는 Influence Functions (IFs)를 선택 도구로 활용하여 시연 풀 내에서 사례의 정보성을 보다 효과적으로 포착할 수 있는 가능성을 강조합니다.

- **Technical Details**: 간접 ICL은 대부분이 최종 작업에 적합하지 않은 사례 집합에서 demonstration을 선택하여 간접적인 감독을 제공하는 것을 목표로 합니다. 본 연구에서는 높은 정확도의 demonstration 선택을 위해 최종 작업과의 인덕티브 편향을 나타내는 후보 demonstrations을 식별하는 데 IF를 활용하는 방법이 Practical하다는 것을 보여줍니다. 이를 위해, BertScore-Recall (BSR)과 결합한 IF Surrogate 모델이 Mixture of Tasks 설정에서 3-shot 및 5-shot 설정에 대한 평균 절대 정확도를 각각 0.37% 및 1.45% 향상시킬 수 있음을 입증합니다.

- **Performance Highlights**: Noisy Demonstrations 설정에서는 IF 기반 선택기를 통해 전통적인 ICL 선택기(분석적 평가 방식인 BSR 및 Cosine Similarity)의 가중 평균 선택을 관찰하였으며, 이로 인해 노이즈가 있는 GLUE 벤치마크에서 각각 2.90% 및 2.94%의 정확도 향상을 가져왔습니다. 연구 결과는 IFs가 demonstration 선택에서 단순한 의미적 유사성보다 더 뛰어난 성능을 발휘할 수 있음을 입증합니다. 전반적으로, 이 연구는 ICL의 demonstration 선택을 발전시키고, 과거의 방법론을 넘어서는 강력한 프레임워크를 제안합니다.



### Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation (https://arxiv.org/abs/2501.01472)
- **What's New**: 이번 연구는 실시간 시계열 데이터에 대한 테스트 단계 적응(test-time adaptation, TTA)의 발전을 목표로 하며, 기존의 시각적 작업을 위한 TTA 방법들이 가지고 있는 한계를 극복하려고 합니다. 본 연구에서는 Uncertainty-aware Prototyping(불확실성 인지 프로토타입)과 결합된 Augmented Contrastive Clustering(자극적인 대조 클러스터링) 방법을 제안하면서, 다채로운 시간 정보를 수집하고 신뢰할 수 있는 pseudo label(유사 라벨)을 생성하는 새로운 접근법을 소개합니다.

- **Technical Details**: ACCUP(Augmented Contrastive Clustering with Uncertainty-aware Prototyping)는 시계열 데이터에 최적화된 새로운 TTA 기법으로, 불확실성을 인식하는 프로토타입 앙상블 모듈을 도입합니다. 이 모듈은 magnitude warping augmentation(진폭 왜곡 증대)을 활용하여 시계열 데이터의 변화에 내성이 있는 시간 패턴을 학습하며, 이를 통해 모델의 출력 신뢰도를 높이고 신뢰 가능한 예측을 보장합니다. 또한, 엔트로피 비교 기법을 도입하여 고신뢰 영역의 예측을 선택적으로 수집하고, 이러한 신뢰도를 통해 학습 과정을 개선하는 전략을 채택합니다.

- **Performance Highlights**: 본 연구는 세 개의 실제 시계열 데이터셋과 추가적인 시각적 데이터셋을 통해 ACCUP의 효과를 검증하였습니다. 실험 결과, 제안된 방법은 다양한 시계열 애플리케이션에서 기존 방법들보다 우수한 성능을 보이며, 불확실성이 높은 pseudo label의 부정적인 영향을 최소화하는 데 많은 기여를 하였습니다. 또한, 타임 시리즈 데이터의 독특한 변화를 잘 잡아내며, 각 클래스 간의 명확한 구분을 가능하게 하는 집합적 클러스터링을 증진시켰습니다.



### Balance-aware Sequence Sampling Makes Multi-modal Learning Better (https://arxiv.org/abs/2501.01470)
- **What's New**: 이번 연구에서는 데이터 이질성(data heterogeneity)으로 인해 발생하는 모달리티 불균형(modality imbalance) 문제를 해결하기 위해 Balance-aware Sequence Sampling (BSS) 기법을 제안합니다. 기존의 다중 모달 학습(multi-modal learning) 접근법이 최적화 목표에만 집중했던 반면, 우리는 샘플 순서가 학습 편향을 초래할 수 있음을 강조했습니다. BSS는 학습된 샘플의 균형 정도를 평가하고, 이를 기반으로 점진적으로 훈련 샘플을 제공하는 휴리스틱 스케줄러를 사용합니다.

- **Technical Details**: BSS는 여러 관점에서 샘플의 균형 정도를 평가하기 위해 다중 관점 측정기를 정의한 후, 커리큘럼 학습(curriculum learning) 원칙을 적용하고 있습니다. 이는 균형 잡힌 샘플에서 시작해 점차 불균형 샘플로 진행하는 방식으로, 모델의 성능 향상을 꾀합니다. 또한, 모델 능력이 향상됨에 따라 샘플 균형이 어떻게 발전하는지를 고려하여, 학습 기반 확률적 샘플링 기법을 통해 훈련 순서를 동적으로 업데이트합니다.

- **Performance Highlights**: 많은 실험을 통해 제안한 방법은 최신의 다중 모달 학습 접근법들과 비교하여 우수한 성능을 보이는 것으로 나타났습니다. 실험은 CREMA-D, Kinetics-Sounds, VGGSound 등 다양한 데이터셋에서 수행되었습니다. 특히, Twitter2015 데이터셋에서 균형 잡힌 샘플이 학습 초기에 높은 의미적 일관성을 보임을 확인하였으며, 이는 다중 모달 학습의 효과성에 기여하고 있습니다.



### Goal Recognition using Actor-Critic Optimization (https://arxiv.org/abs/2501.01463)
- **What's New**: 이 논문은 Deep Recognition using Actor-Critic Optimization (DRACO)라는 새로운 목표 인식 알고리즘을 소개합니다. DRACO는 비구조화된 데이터로부터 정책 네트워크를 학습하여 목표를 추론하는 첫 번째 알고리즘이며, 연속적인 정책 표현을 통해 목표 가설을 평가하는 새로운 메트릭을 도입합니다. 이 방법을 통해 기존 방식이 가진 한계를 극복하고, 더 많은 가능한 목표를 인식할 수 있는 기반을 제공합니다.

- **Technical Details**: DRACO는 환경과의 상호작용을 통해 목표 종속 신경망( Neural Networks, NNs)을 학습하며, 이러한 신경망은 다양한 학습 가능한 도메인에서 관찰된 에이전트가 추구하는 목표의 가능성을 추정합니다. 이는 기존의 비싼 계획자(planner)를 실시간으로 실행할 필요를 없애며, 전이 학습(transfer learning)을 통해 새로운 목표를 표현하는데도 활용될 수 있습니다. 논문에서는 Wasserstein 거리와 통계적 Z-score 메트릭을 기반으로 한 두 가지 거리 측정 방식을 개발하여 학습된 목표 종속 정책과 관측 결과를 비교합니다.

- **Performance Highlights**: DRACO는 다양한 시나리오에서 테스트되었으며, 주어진 입력이 적은 경우에도 기존 알고리즘들보다 더 뛰어난 성능을 보입니다. 특히 연속적 환경에서 과거 방법들을 현저히 초월하는 성과를 달성하였으며, 제시된 테스트베드(test bed)를 통해 목표 인식의 가능성을 보여줍니다. 이 결과는 DRACO의 강력함을 증명하며, 전통적인 목표 인식과 딥 강화 학습(Deep Reinforcement Learning) 간의 교량 역할을 합니다.



### Pan-infection Foundation Framework Enables Multiple Pathogen Prediction (https://arxiv.org/abs/2501.01462)
Comments:
          15 pages, 8 figures

- **What's New**: 이번 논문에서는 호스트 반응 기반 진단방법이 세균 및 바이러스 감염 진단의 정확성을 향상시킬 수 있다는 점을 강조합니다. 저자들은 13개국과 21개 플랫폼에서 수집된 11,247개의 샘플을 포함한 가장 큰 감염 호스트 반응 전사체(transcriptome) 데이터를 구축하였습니다. 이는 기존의 작은 샘플 크기 및 제한된 감염 유형으로는 이루기 어려운 일반화 가능한 진단 모델의 탐색을 가능하게 합니다.

- **Technical Details**: 연구팀은 감염 관련 데이터셋을 바탕으로 포함된 데이터를 사용하여 예측 모델을 구축하였으며, 이 모델의 AUC는 0.97로 나타났습니다. 이후 지식 증류(knowledge distillation) 기법을 활용하여 이 "교사" 모델에서 네 가지 경량화된 병원체 "학생" 모델로 인사이트를 효과적으로 전달합니다. 연구에서 다룬 경량 모델은 황색포도상구균(infection), 연쇄상구균(infection), HIV 감염, RSV 감염 및 패혈증(sepsis)과 관련되어 있습니다.

- **Performance Highlights**: 각 모델은 다음과 같은 AUC 성능을 보였습니다: 황색포도상구균 감염 0.99, 연쇄상구균 감염 0.94, HIV 감염 0.93, RSV 감염 0.94, 패혈증 모델 0.99입니다. 이러한 성능 덕분에 제안된 지식 증류 프레임워크는 다양한 감염 진단을 위한 현장 적응 가능성을 높이며, 경량 디자인을 통해 임상 환경에서 효과적으로 활용될 것으로 기대됩니다.



### GAN-TAT: A Novel Framework Using Protein Interaction Networks in Druggable Gene Identification (https://arxiv.org/abs/2501.01458)
Comments:
          4 pages, 2 figures

- **What's New**: 이번 연구에서는 약물 타겟 지식을 확대하기 위한 새로운 접근 방식을 제안하고, GAN-TAT라는 프레임워크를 통해 고차원 단백질 상호작용 네트워크(Protein Interaction Network, PIN)를 직접적으로 활용하고 있습니다. 특히, ImGAGN 알고리즘을 통해 생성된 잠재 표현(latent representation)을 사용하여 약물로 타겟이 가능한 유전자를 추론하는 데 초점을 맞추었습니다. GAN-TAT는 임상 증거를 기반으로 한 예측 결과로, 약물유전체학(pharmacogenomics)에서의 실제적인 응용 가능성을 강조합니다.

- **Technical Details**: 연구에서 사용된 PIN은 신호 전달 경로 데이터를 기반으로 구축되었으며, 총 6,048개의 유전자 노드와 20,697개의 방향성 비가중 엣지로 구성됩니다. ImGAGN-GraphSAGE 모델을 활용한 네트워크 임베딩 기술을 도입하여 각 유전자 대한 80차원 임베딩을 생성하였고, 이어서 XgBoost 분류기를 사용해 데이터 불균형 문제를 해결하기 위한 서브샘플링 전략을 적용하였습니다. GAN-TAT 구조는 생성적 적대 신경망(Generative Adversarial Network, GAN)을 포함하며, 그래프 생성기, 인코더, 판별기로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, GAN-TAT 구조와 ImGAGN-GraphSAGE + XgBoost 조합이 최고 AUC-ROC 점수 0.951을 달성하며, 다른 임베딩 알고리즘보다 지속적으로 우수한 성능을 보였습니다. 특히, XgBoost는 모든 임베딩 방법에서 가장 효과적인 분류기로 확인되었으며, 임상 진단 관련 유전자(label sets)에서 최고의 효능을 발휘했습니다. 이 연구는 GAN-TAT가 약물 타겟 발굴에서 가진 중요성을 보여주는 다양한 실험적 데이터를 제공합니다.



### Reinforcing Thinking through Reasoning-Enhanced Reward Models (https://arxiv.org/abs/2501.01457)
- **What's New**: 본 연구는 LLM의 복잡한 다단계 추론을 개선하기 위해 Distillation-Reinforcement-Reasoning (DRR)이라는 새로운 3단계 프레임워크를 제안합니다. 이 프레임워크는 LLM의 내부 행동을 외부 피드백으로 활용하여, LLM이 언제 멈춰야 할지 결정할 수 있도록 돕습니다. DRR은 중간 단계의 수작업 레이블이 필요 없으며, 경량 디자인으로 다양한 LLM 중심의 작업에 쉽게 적용할 수 있습니다.

- **Technical Details**: DRR 프레임워크는 LLM의 추론 능력을 반영하는 행동 데이터를 생성을 통해 시작합니다. 이후, 행동 데이터를 기반으로 경량 식별 보상 모델(Discriminative Model, DM)을 훈련하여 추론 시 판단을 돕습니다. 이 과정은 언어적 보상(verbal reward)을 통해 LLM의 평가를 제공하며 모델 파라미터를 변경하지 않고도 동적 피드백 메커니즘을 구축합니다.

- **Performance Highlights**: 실험 결과, DRR 프레임워크는 자가 비평(self-critique) 방식을 적용한 방법들보다 우수한 성능을 보였으며, 추가적인 복잡한 데이터 주석에 의존하지 않는 것으로 나타났습니다. 연구팀은 모든 코드베이스와 체크포인트, 생성된 데이터를 공개할 예정이며, 이는 개방형 및 닫힌 소스 LLM에 유익할 것으로 기대됩니다.



### A Fourfold Pathogen Reference Ontology Su (https://arxiv.org/abs/2501.01454)
Comments:
          25 pages

- **What's New**: 이번 논문은 감염병 관련 데이터를 관리하는 데 중요한 역할을 하는 표준화된 온톨로지(ontology)의 통합에 대해 다룹니다. 특히, 감염병 온톨로지(Infectious Disease Ontology, IDO)와 그 확장인 코로나 감염병 온톨로지(Coronavirus Infectious Disease Ontology, CIDO)의 중요성을 강조합니다. COVID-19 팬데믹은 IDO와 그 바이러스 특이적 확장의 업데이트 필요성을 부각시켰습니다.

- **Technical Details**: 이 논문에서는 IDO의 병원체 특이적 확장을 생성하기 위해 '허브와 스포크' 방법론을 채택했습니다. 그 결과 바이러스 감염병 온톨로지(Virus Infectious Disease Ontology, VIDO), 박테리아 감염병 온톨로지(Bacteria Infectious Disease Ontology, BIDO), 곰팡이 감염병 온톨로지(Mycosis Infectious Disease Ontology, MIDO), 기생충 감염병 온톨로지(Parasite Infectious Disease Ontology, PIDO)가 개발되었습니다. 이러한 병원체 특이적 참조 온톨로지의 생성은 감염병 데이터의 모듈화(modularization) 및 재사용성(reusability)을 촉진합니다.

- **Performance Highlights**: 미래의 연구는 이러한 온톨로지를 더욱 정교화하고 새로운 확장을 생성하며, 데이터를 더 잘 공유하고 분석할 수 있도록 생물학적 및 생의학적 용어를 표준화하는 지속적인 노력을 반영하여 응용 온톨로지를 개발하는 데 중점을 둘 것입니다. 이로 인해 감염병 정보의 조직화 및 배포가 더욱 용이해질 것입니다.



### Human-AI Teaming Using Large Language Models: Boosting Brain-Computer Interfacing (BCI) and Brain Research (https://arxiv.org/abs/2501.01451)
Comments:
          13 pages, 5 figures

- **What's New**: 최근 인공지능(AI)을 사용하여 연구 과정을 자동화하거나 아이디어 생성에서 데이터 분석, 논문 작성 및 평가에 이르기까지 전체 연구 사이클을 수행하려는 관심이 증가하고 있습니다. 특히, 본 논문에서는 Brain-Computer Interface(BCI) 개발과 신경과학 분야에서 AI와 인간의 협업을 위한 새로운 접근 방식을 제안하고 있습니다. 저자들은 AI BCI 연구자가 아닌, 인간과 AI의 협업에 중점을 두는 것이 더 유망하다고 주장하며, 이를 위한 협업 작업공간(concept of collaborative workspaces)을 소개합니다.

- **Technical Details**: 저자들은 Python 기반의 ChatBCI 툴박스를 소개하며, 이는 대규모 언어 모델(LLMs)과의 상호작용을 통해 인간과 AI의 협업을 가능하게 합니다. ChatBCI는 EEG 신호에서 운동 이미징을 디코딩하는 BCI 프로젝트의 성공적인 적용 사례를 통해 AI의 활용 가능성을 보여줍니다. 이 툴박스는 데이터 전처리, 분석, 해석, 시각화 등의 기능을 포함하고 있으며, PyTorch를 기반으로 한 딥러닝 기능을 가지고 있습니다.

- **Performance Highlights**: ChatBCI를 통해 연구자들은 짧은 시간 안에 EEG 데이터 세트를 활용한 BCI 프로젝트를 수행할 수 있었습니다. 이 연구에는 데이터 가져오기, 탐색, 검증뿐만 아니라 머신 러닝 모델 선택 및 구현이 포함되어 있으며, 전문가 지식을 AI 시스템에 효과적으로 전달하여 효율적인 협업을 이끌어냈습니다. 이러한 접근은 BCI 연구에서 의미 있는 인간-AI 공동 학습의 가능성을 보여주며, 기타 신경 과학 및 신경 기술 분야에서도 확장 가능성을 지니고 있습니다.



### LS-GAN: Human Motion Synthesis with Latent-space GANs (https://arxiv.org/abs/2501.01449)
Comments:
          6 pages

- **What's New**: 이번 논문에서는 Generative Adversarial Networks (GANs)를 활용하여 텍스트 입력에 기반한 사람의 모션 합성을 위한 새로운 프레임워크를 제안합니다. 기존의 디퓨전 모델에 비해 훨씬 더 빠른 훈련 및 추론 속도를 자랑하며, 높은 품질의 3D 모션을 생성할 수 있습니다. HumanML3D 및 HumanAct12 벤치마크에서 실험을 진행하였고, 저는 GAN 아키텍처가 뛰어난 성능을 보여줍니다.

- **Technical Details**: 이 연구는 Variational Autoencoder (VAE)를 통해 모션을 잠재 공간(latent space)으로 인코딩하고, Pre-trained CLIP 모델을 사용하여 텍스트 입력에 대한 조건을 설정합니다. 이 과정에서 GAN 아키텍처를 도입하여 텍스트 임베딩과 잠재 공간 간의 매핑을 가속화하고, 더 높은 품질의 모션 시퀀스를 효율적으로 생성하는 것을 목표로 합니다. 특히, GAN의 훈련 동적을 활용하여 성능과 충실도를 최적화하고 다양한 GAN 아키텍처를 실험합니다.

- **Performance Highlights**: 실험 결과, 단순한 GAN 아키텍처가 0.482의 FID를 기록하며, MLD에 비해 91%의 FLOPs 감소를 보였습니다. 또한, Action-to-motion HumanAct12 벤치마크에서도 경쟁력 있는 성능을 나타내었습니다. 이러한 GAN의 잠재 공간 내의 적용은 기존 디퓨전 기반 모델의 계산 효율성을 해결하며 실시간 애플리케이션에 적합한 고품질 모션 합성을 위한 잠재력을 열어줍니다.



### Explanatory Debiasing: Involving Domain Experts in the Data Generation Process to Mitigate Representation Bias in AI Systems (https://arxiv.org/abs/2501.01441)
Comments:
          Pre-print version, please cite the main article instead of the pre-print version

- **What's New**: 이 논문은 인공지능(AI) 시스템에서의 표현 편향(representation bias)을 해결하기 위해 도메인 전문가(domain expert)를 효과적으로 참여시키기 위한 일반적인 디자인 가이드라인을 제시합니다. 이 가이드라인은 헬스케어 중심의 애플리케이션에 적용되어 35명의 헬스케어 전문가와 Mixed-methods 사용자 연구를 통해 평가되었습니다. 연구 결과, 도메인 전문가의 참여가 모델의 정확성을 해치지 않고 표현 편향을 줄이는 데 도움이 된다는 것을 보여주었습니다.

- **Technical Details**: 표현 편향은 훈련 데이터에 특정 그룹의 샘플이 부족할 때 발생하며, 이로 인해 모델이 높은 대표성을 가진 다른 그룹에 편향됩니다. 기존의 데이터 증강(data augmentation) 기법들이 이러한 문제를 해결하기 위한 한 방법으로 제안되었으나, 종종 문제된 데이터 포인트를 생성하는 비판을 받았습니다. 이 연구에서는 헬스케어 전문가들이 데이터 증강 및 표현 편향 완화에 어떻게 기여할 수 있을지를 탐구하며, 관련 지식과 측정 기준을 소개합니다.

- **Performance Highlights**: 연구에 따르면, 도메인 전문가가 참여할 경우 AI 모델의 표현 편향을 줄이는 동시에 예측 정확성도 유지할 수 있다는 것을 발견했습니다. 또한, 이 과정에 포함된 도메인 전문가의 전반적인 신뢰가 높아지는 경향이 나타났습니다. 이 연구는 개발자들이 더 효과적으로 도메인 전문가를 포함하여 강력한 디바이싱(debiasing) 시스템을 구축할 수 있도록 다양한 권고 사항을 제공합니다.



### Fundamental Risks in the Current Deployment of General-Purpose AI Models: What Have We (Not) Learnt From Cybersecurity? (https://arxiv.org/abs/2501.01435)
- **What's New**: 이번 연구는 일반 목적 AI, 특히 Large Language Models (LLMs)의 급속한 발전과 그 활용 범위 확대에 대해 다룹니다. 대화형 AI에서부터 운영체제와 유사한 지위로 발전하여 애플리케이션의 의사결정과 논리를 제어할 수 있는 수준에 이르렀습니다. Microsoft의 Co-pilot과 Office 통합, OpenAI의 Altera와 같은 사례들이 그 예시입니다.

- **Technical Details**: 이 논문에서는 LLMs의 도구 사용(tool-use)과 자율성 증가, 데이터 접근성을 포함한 실행 능력(execution capabilities)에 대한 성과를 분석합니다. 이러한 발전은 사이버 보안(cybersecurity) 문제와 같은 다양한 도전 과제를 동반합니다. 연구팀은 평가(evaluation)에 대한 자신들의 작업을 강조하고 향후 비즈니스 기회와 도전에 대해 설명합니다.

- **Performance Highlights**: LLMs가 단순한 언어 모델에서 진화하여 다양한 기능을 갖춘 챗봇 역할을 할 수 있게 된 점이 두드러집니다. 이러한 발전은 더 높은 자율성과 효율성을 제공하며, 비즈니스 환경에서의 적용 가능성을 높이고 있습니다. 그러나 이러한 기술이 발전함에 따라 사이버 보안 관련 문제와 그것에 대한 대비 필요성이 강조되고 있습니다.



### Survey on safe robot control via learning (https://arxiv.org/abs/2501.01432)
- **What's New**: 이번 연구에서는 안전한 로봇 학습의 경향을 탐구하며, 고성능 제어와 엄격한 안전 제약 조건 간의 균형을 맞추는 방법에 대해 논의합니다. 로봇 시스템이 위험한 상태에 도달하지 않도록 설계될 수 있는지에 대한 이해를 구하는 것이 중요합니다. 이러한 연구는 항공우주, 의료 등 다양한 산업 분야에서의 로봇 기술 발전에 기여할 것입니다.

- **Technical Details**: 연구는 고전 제어 이론과 학습 기반 접근법, 임베디드 시스템 설계를 포괄적으로 분석합니다. 고전적인 제어 기법은 모델 기반 및 비모델 기반 방식으로 구분되며, 모델 예측 제어(MPC)와 같은 고급 기법을 활용하여 주어진 제한 조건을 온라인에서 해결하는 방법을 제시합니다. 또한, 심층 강화 학습(Deep Reinforcement Learning) 알고리즘을 통해 데이터 기반으로 제어 정책을 개선하는 방법도 다룹니다.

- **Performance Highlights**: 안전성이 요구되는 상황에서 제어 시스템이 충족해야 할 조건들, 예를 들어 안정성이나 장애물 회피와 같은 측면을 강조합니다. 또한 클래스 제어 기술, PID 제어기 및 온라인 시스템 식별 절차를 통해 성능을 향상시키는 다양한 방법을 논의합니다. 최종적으로, 이러한 다양한 접근 방식을 통해 복잡한 로봇 시스템 내에서 안전성을 유지하면서 고성능 제어를 구현할 수 있는 가능성을 제시합니다.



New uploads on arXiv(cs.LG)

### MixGCN: Scalable GCN Training by Mixture of Parallelism and Mixture of Accelerators (https://arxiv.org/abs/2501.01951)
Comments:
          15 pages, 12 figures, 5 tables

- **What's New**: 본 논문에서는 그래프 기반 머신 러닝 과제에서의 그래프 합성곱 네트워크(GCN)의 효율적인 학습을 위한 MixGCN을 제안합니다. MixGCN은 거대 피쳐 텐서와 혼합 희소-밀집 연산과 같은 두 가지 주요 도전 과제를 동시에 해결하려고 합니다. 이러한 접근법을 통해 메모리 사용량과 통신 대역폭의 비효율성을 감소시킵니다.

- **Technical Details**: MixGCN은 Mixture of Parallelism (MoP)과 Mixture of Accelerators (MoA)라는 두 가지 주요 기술을 기반으로 합니다. MoP는 피쳐 수준과 노드 수준의 병렬성을 결합하여 GCN의 스케일러블한 훈련을 가능하게 하고, MoA는 희소 및 밀집 연산을 각각 다른 가속기에 분산시킵니다. 특히, S-SpMM이라는 고유한 희소 연산을 통해 두 개의 연속적인 희소 연산을 융합하여 효율적인 계산을 구현합니다.

- **Performance Highlights**: MixGCN은 실험을 통해 5개의 대규모 데이터셋에서 우수한 성능을 입증하였으며, 4개의 GPU 클러스터에서 최신 기법 대비 10.4배의 성능 향상을 기록했습니다. 추가적으로 S-SpMM을 사용할 경우 성능은 17.2배로 증가합니다. 제안된 MoP는 안정적인 통신량과 피쳐 메모리 사용을 유지하면서도 균형 잡힌 작업량을 보장합니다.



### MADGEN -- Mass-Spec attends to De Novo Molecular generation (https://arxiv.org/abs/2501.01950)
Comments:
          preprint

- **What's New**: 이번 논문에서는 MADGEN(Mass-spec Attends to De Novo Molecular GENeration)이라는 새로운 접근법을 제안합니다. 이는 질량 분석 데이터(mass spectrometry data)를 기반으로 한 scaffold 기반의 분자 구조 생성을 위한 방법으로, 현재 많은 MS/MS 스펙트럼이 '어두운 화학 공간(dark chemical space)'에 위치하고 있다는 문제를 해결하고자 합니다. MADGEN은 scaffold 검색 및 스펙트럼 조건화된 분자 생성을 두 단계로 나눠 진행합니다.

- **Technical Details**: MADGEN의 첫 단계는 주어진 MS/MS 스펙트럼에 대해 scaffold 검색을 순위(rank) 문제로 설정하고, 대조 학습(contrastive learning)을 이용하여 질량 스펙트럼과 후보 분자 scaffold를 정렬하는 과정입니다. 두 번째 단계에서는 검색된 scaffold에서 시작하여 MS/MS 스펙트럼에 의해 안내되는 주의(attention) 기반 생성 모델을 활용하여 최종 분자를 생성합니다. 이러한 접근 방법은 분자의 생성 검색 공간을 제한하고 복잡성을 줄이며 정확성을 향상시키는 데 기여합니다.

- **Performance Highlights**: MADGEN은 NIST23, CANOPUS, MassSpecGym의 세 가지 데이터셋에서 평가되었으며, 예측된 scaffold와 오라클(oracle) 검색기를 통해 성능을 비교했습니다. 결과적으로, 주의 메커니즘을 통해 스펙트럼 정보를 통합하여 오라클 검색기를 사용할 때 강력한 결과를 달성하는 것을 입증했습니다. 이는 새로운 대사물질, 생리활성 화합물 및 미지의 화합물 탐색에 필수적인 의의를 가집니다.



### Improving Transducer-Based Spoken Language Understanding with Self-Conditioned CTC and Knowledge Transfer (https://arxiv.org/abs/2501.01936)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 논문에서는 RNN 변환기 모델(RNN Transducer, RNN-T)을 기반으로 한 엔드투엔드(End-to-End, E2E) 음성 언어 이해(Spoken Language Understanding, SLU) 성능을 높이기 위해 자가 조건화된 CTC 자동 음성 인식(Automatic Speech Recognition, ASR) 목표를 도입하는 방법을 제안합니다. 이 모델은 ASR과 SLU를 순차적으로 실행하며, CTC 자가 조건화를 통해 SLU 작업이 ASR 작업에 의해 조건화되도록 보장합니다. 이러한 새로운 공동 모델링은 SLU 성능을 상당히 향상시키며, 저자들은 BERT 모델과의 정렬을 통해 추가적인 성능 개선을 이루었습니다.

- **Technical Details**: 이 연구에서는 RNN-T 기반 E2E SLU 모델을 활용하며, CTC 목표를 이용해 ASR 출력에 따라 SLU 작업이 조건화되는 방식을 제안합니다. 최근의 자가 조건화된 CTC 손실을 이용하여 E2E로 미분 가능하고 단일 패스 방식으로 작동하는 모델을 구현하였습니다. 추가적으로, 발화에 존재하는 개체의 예측을 위한 레이어를 도입하여 ASR 출력과 SLU 디코딩을 조건화하는 새로운 접근 방식을 사용하였습니다.

- **Performance Highlights**: 제안된 방법은 여러 강력한 베이스라인 모델에 비해 상당한 성능 향상을 보여주며, Whisper과 같은 대규모 모델과 비교했을 때 매개변수 수가 적음에도 불구하고 유사한 성능을 발휘합니다. 이 연구는 E2E SLU 영역에서 자가 조건화된 CTC 기반 목표를 최초로 도입했으며, 기존의 LLM 기반 지식 전이 방법을 개선한 점에서도 의미가 큽니다.



### Fusion DeepONet: A Data-Efficient Neural Operator for Geometry-Dependent Hypersonic Flows on Arbitrary Grids (https://arxiv.org/abs/2501.01934)
- **What's New**: 이번 연구는 고속 비행체 설계를 위해 어려운 기하학 의존 초음속 유동장 예측의 새로운 접근법을 제시합니다. 특히, Fusion DeepONet이라는 새로운 네트워크 구조를 개발하여, 제한된 데이터에서도 다양한 기하학적 형태에 적응하고 일관되게 예측할 수 있도록 합니다. 본 연구의 주요 기여는 기존의 DeepONet, U-Net, FNO, MeshGraphNet 모델을 비교 평가하여, 고속 유동 예측의 정확도를 개선하는 데 있습니다.

- **Technical Details**: 연구에서는 2D 압축 오일러 방정식을 해결하여 기하학적 형태에 따라 생성된 유동장 학습을 목표로 합니다. 이를 위해 36개의 고유한 타원형 기하학을 이용해 고충실도 시뮬레이션을 생성하고, N-Net 아키텍처와 기하학적으로 조건부 매핑을 포함하는 Fusion DeepONet을 설계했습니다. 특히, Fusion DeepONet은 훈련 가능한 매개변수가 적으며, 다양한 기하학과 불규칙 격자에서의 예측 능력이 뛰어납니다.

- **Performance Highlights**: Fusion DeepONet은 제한된 훈련 데이터에도 불구하고 U-Net과 유사한 성능을 발휘하며, 불규칙 격자에서는 MeshGraphNet 및 Vanilla DeepONet보다 뛰어난 예측 성능을 보여줍니다. 서로 다른 운영자 기반 프레임워크의 성능을 비교한 결과, 전통적인 U-Net과 Fusion DeepONet이 규칙적인 격자 설정에서 가장 높은 정확도를 기록했습니다. 또한, Fusion DeepONet의 특정 학습 과정에서는 일반화 오류를 유의미하게 감소시키는 결과를 보였습니다.



### GoBERT: Gene Ontology Graph Informed BERT for Universal Gene Function Prediction (https://arxiv.org/abs/2501.01930)
Comments:
          Accept by AAAI-25

- **What's New**: 이번 연구에서는 Gene Ontology 그래프와 BERT(Bidirectional Encoder Representations from Transformers)를 활용하여 유전자 기능 예측 문제를 접근하는 새로운 방법인 GoBERT를 제안합니다. GoBERT는 기존 기능을 입력으로 사용하여 유전자 및 유전자 산물에 대한 기능 예측을 일반화하는 혁신적인 기능 예측 작업을 특징으로 합니다. 이 모델은 명시적(relations)과 암묵적(implicit) 관계를 포착하는 두 가지 전훈련(task)을 통해 훈련됩니다.

- **Technical Details**: GoBERT는 Directed Acyclic Graph(DAG) 구조로 이루어진 Gene Ontology의 명시적 관계를 포착하기 위해 인접 행렬과 관련된 텍스트의 의미적 정보를 활용합니다. 또한, 임의의 마스킹 언어 모델링(MLM) 작업을 통해 암묵적 관계를 찾아내고, 이를 통해 새로운 유전자 기능을 예측할 수 있는 기반을 형성합니다. 이 접근 방식은 명시적 및 암묵적 기능 관계 사이의 격차를 메워 유전자 기능에 대한 이해를 심화시킵니다.

- **Performance Highlights**: GoBERT는 유전자 및 유전자 산물의 새로운 기능 예측에 대해 실험, 사례 연구 및 배제 연구를 통해 그 효과성을 입증했습니다. 본 연구는 기존의 연구들에 비해 더욱 포괄적으로 유전자 기능 예측을 다루며, 깊은 학습(deep learning) 방법을 적용한 최초의 연구로서 평가됩니다. GoBERT의 성공적인 기능 예측 능력은 생물학적 응용 가능성을 크게 향상시킬 것으로 기대됩니다.



### Social Processes: Probabilistic Meta-learning for Adaptive Multiparty Interaction Forecasting (https://arxiv.org/abs/2501.01915)
Comments:
          This is an extension paper to "Social Processes: Self-Supervised Meta-Learning over Conversational Groups for Forecasting Nonverbal Social Cues", by Raman et al. (arXiv:2107.13576)

- **What's New**: 본 연구는 소셜 상호작용 예측의 새로운 접근 방식을 제안합니다. 특히, 기존 연구에서는 단일 또는 이인 상호작용 모형에 집중했지만, 우리는 그룹 대화 수준에서의 예측에 초점을 맞추었습니다. 이를 통해 각 그룹의 고유한 동적인 특성을 고려한 예측 모델을 개발했습니다.

- **Technical Details**: 이 논문은 메타 학습(meta-learning) 접근 방식을 활용하여 각 그룹을 독립적인 학습 과제로 다루었습니다. 소셜 프로세스(SP) 모델을 도입하여 그룹 구성원의 과거 저수준 다중 모달 신호를 기반으로 미래의 다중 모달 신호에 대한 예측을 수행 합니다. 이러한 방법은 학습되지 않은 그룹에 대한 일반화 능력을 높이고, 각 그룹의 상호작용을 명시적으로 모델링합니다.

- **Performance Highlights**: 성능 분석을 통해 우리는 제안한 SP 모델들이 출력과 잠재 공간(latent space)에서 우수한 일반화 능력을 가지는 것을 확인했습니다. 우리는 합성 데이터셋을 이용한 심층 분석을 통해 모델의 예측력이 강화되었음을 보이며, 이러한 접근 방식은 다수의 실험에서도 효과적으로 입증되었습니다.



### Alleviating Overfitting in Transformation-Interaction-Rational Symbolic Regression with Multi-Objective Optimization (https://arxiv.org/abs/2501.01905)
Comments:
          25 pages, 8 figures, 4 tables, Genetic Programming and Evolvable Machines, vol 24, no 2

- **What's New**: 본 논문은 Symbolic Regression의 새로운 표현인 Transformation-Interaction-Rational(TIR)을 소개하고, 이를 Genetic Programming과 결합하여 성능을 개선하는 방법을 제시합니다. TIR은 비선형 함수의 비율로 정의되며, 상대적으로 간단한 표현에 대한 탐색을 편향하는 것을 목표로 합니다. 특히, 작은 데이터셋에서 모델의 복잡도를 줄이기 위해 추가적인 선택적 압력을 적용할 수 있음을 보여줍니다.

- **Technical Details**: TIR 표현은 비선형 변환 변수를 선형 회귀하는 두 함수의 비율로 구성되어 있습니다. 이를 통해 기존의 방법들보다 성능이 향상되었으며, 특히 NSGA-II 알고리즘을 활용한 다목적 최적화(Multi-Objective Optimization)는 결과를 개선하는 데 기여했습니다. 이 방식은 적합도에 대한 패널티(penalization) 없이도 간편한 모델 선택을 가능하게 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 TIR의 성능은 경쟁력 있는 수준으로 나타났으며, 특히 다목적 최적화 기법이 사용되었을 때 전반적인 성능이 개선되었습니다. 작은 데이터셋에서는 미미한 성능 향상이 관찰되었지만, 이는 통계적으로 유의미하지 않아 추가적인 전략이 필요함을 시사합니다. 결과적으로, TIR은 더 나은 모델을 위한 잠재력을 갖고 있으며, 후속 연구를 통해 개선될 수 있습니다.



### Exploring Equality: An Investigation into Custom Loss Functions for Fairness Definitions (https://arxiv.org/abs/2501.01889)
Comments:
          17 Pages, 12 Figures

- **What's New**: 이 논문은 COMPAS의 다양한 공정성 메트릭(fairness metrics)인 equalized odds, disparate impact, equal opportunity 및 예측 정확성(predictive accuracy) 사이의 복잡한 트레이드오프(tradeoffs)를 탐구합니다. Gupta 외(2024)에 의해 이론적으로 제안된 새로운 Group Accuracy Parity (GAP) 프레임워크를 최초로 공정성 중심의 방식으로 구현하고 이를 COMPAS에 적용하였습니다.

- **Technical Details**: 논문에서 제안한 조합 분석 절차는 Pareto front와 다변량 분석(multivariate analysis)을 포함하여, 바이올린 그래프(violin graphs)와 같은 데이터 시각화를 활용합니다. 이를 통해 서로 다른 공정성 이념에 최적화된 COMPAS 모델의 공정성을 운영상으로 정의하고 정확하게 비교하는 방법을 개발하였습니다.

- **Performance Highlights**: GAP는 COMPAS의 현재 국가적 구현 및 전통적인 공정성 정의에 최적화된 대안 구현과 비교하여 공정성과 정확성 사이의 균형을 개선하였다고 결론짓습니다. 하지만 COMPAS의 알고리즘 개선에도 불구하고 외부적 편향(bias)이 그 구현의 공정성을 저해하고 있으며, 예측 경찰(pred predictive policing) 및 COMPAS의 내부 작업에 관한 투명성 부족 등의 문제는 역사적인 부당함을 초래했습니다.



### DFF: Decision-Focused Fine-tuning for Smarter Predict-then-Optimize with Limited Data (https://arxiv.org/abs/2501.01874)
Comments:
          12 pages, 4 figures, The 39th Annual AAAI Conference on Artificial Intelligence

- **What's New**: 본 논문에서는 Decision-Focused Fine-tuning (DFF)이라는 새로운 프레임워크를 제안하여 Decision-Focused Learning (DFL)을 Predict-then-Optimize (PO) 파이프라인에 통합합니다. 이 프레임워크는 편향 교정 모듈을 통해 DFL 모듈을 내장하여 의사 결정 성능을 향상시키고, 기본 예측 모델과의 근접성을 유지하는 방식으로 설계되었습니다. 제안된 방법은 제한된 데이터셋에 대해서도 예측 편향을 사전 정해진 상한으로 제한할 수 있음을 이론적으로 증명하였습니다.

- **Technical Details**: DFF는 제약 최적화 문제로 형성되며, 이는 의사 결정 손실(Decision Loss) 향상을 위한 모델 훈련에서 발생할 수 있는 다양한 도전 과제를 해결하기 위해 설계되었습니다. 특히, DFF는 예측 모델의 출력을 조정하기 위한 편향 교정 계층을 포함하고 있어, 이를 통해 예측의 물리적 의미를 유지하면서 보다 정교하게 조정이 가능합니다. DFF는 비차별적 또는 블랙박스 모델에도 유연하게 통합될 수 있습니다.

- **Performance Highlights**: 다양한 테스트를 통해 DFF는 종합적인 평가에서 기존의 예측 모델보다 뛰어난 의사 결정 성능을 보여주었습니다. 특히 네트워크 흐름, 포트폴리오 최적화 및 자원 할당 문제와 같은 실제 문제에서 DFF가 우수한 성과를 나타내는 것으로 나타났습니다. 이러한 결과는 DFF의 광범위한 적응성과 다른 시나리오에 대한 내구성을 강조합니다.



### LCFed: An Efficient Clustered Federated Learning Framework for Heterogeneous Data (https://arxiv.org/abs/2501.01850)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 LCFed라는 새로운 클러스터 연합 학습(CFL) 프레임워크를 제안합니다. LCFed는 사용자 정의 모델 파티셔닝(model partitioning) 방식을 채택하며, 각 서브 모델의 집계 전략을 구분함으로써 클러스터 내 공동 학습(intra-cluster co-training) 과정에 글로벌 지식을 통합합니다. 이를 통해 프레임워크는 데이터 이질성(data heterogeneity)으로 인한 성능 문제를 개선하고 최적의 교육 성능을 달성합니다.

- **Technical Details**: LCFed는 각 기기에서 개인화된 모델을 훈련하고, 이를 클러스터로 그룹화하여 클러스터 중심 모델을 학습합니다. 이 과정에서 모델 유사성 측정을 위해 저차원 모델(low-rank models)을 사용하는 계산 효율적인 방법을 사용자 정의하여, 서버 측 클러스터 업데이트의 연산 오버헤드를 최소화합니다. 이러한 방법은 클러스터 할당에 필요한 서버의 계산 부담을 줄이게 해줍니다.

- **Performance Highlights**: 실험 결과, LCFed는 최신의 벤치마크 모델들보다 테스트 정확도(test accuracy) 및 클러스터링 연산 효율(clustering computational efficiency)에서 우수한 성능을 보였습니다. 다양한 실제 데이터셋을 통해 LCFed의 전반적인 성능, 견고성(robustness), 및 계산 효율성을 입증하였습니다. 이러한 성과는 LCFed가 클러스터 연합 학습 분야에서 중요한 진전을 이룩했음을 시사합니다.



### Learning from Ambiguous Data with Hard Labels (https://arxiv.org/abs/2501.01844)
Comments:
          9 pages, 4 figures, accepted by ICASSP 2025

- **What's New**: 본 논문은 모호한 데이터로부터 학습하는 새로운 프레임워크인 Quantized Label Learning (QLL)을 제안합니다. 일반적인 데이터 주석 방식에서는 각 인스턴스가 하나의 확실한 하드 레이블에 연관되도록 가정하지만, 이로 인해 모호한 데이터에서 과도한 확신을 가진 모델이 생성되고 일반화 성능이 저하될 수 있습니다. 저자들은 그러한 문제를 해결하기 위해 최적의 소프트-레이블 분포를 고려하여 하드 레이블을 생성하는 방법론을 개발하였습니다.

- **Technical Details**: QLL은 고유한 노이즈 데이터 설정으로 정식화되었습니다. 인스턴스의 모호성을 수치적으로 측정하기 위해 소프트-레이블 분포의 엔트로피를 사용할 수 있습니다. 또한, Class-wise Positive-Unlabeled (CPU) 리스크 추정기를 제안하여 모호한 데이터와 양자화된 레이블만으로도 정확한 분류기를 훈련할 수 있는 방법을 마련하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 CPU 방법론은 모델의 일반화 성능을 크게 향상시켰으며 기존 베이스라인 방법들과 비교해 우수한 성능을 보여주었습니다. 이를 통해 두 가지 교육 방식을 통해 얻은 데이터의 성격이 다름을 이해하고 활용할 수 있는 가능성을 제시합니다.



### Practical machine learning is learning on small samples (https://arxiv.org/abs/2501.01836)
- **What's New**: 이 논문은 기계 학습(Machine Learning)이 실제 상황에서 의존성을 식별하는 방식에 대한 새로운 관점을 제안합니다. 저자는 기계 학습이 무한히 증가하는 훈련 샘플에 의존해야 한다고 가정하는 통계적 학습 이론의 한계를 비판하며, 실제 학습은 데이터 포인트 간의 피드백의 급격한 변화가 없다는 암묵적 가정에 기반한다고 주장합니다. 이를 통해 저자는 '실용 학습 패러다임(Practical Learning Paradigm)'을 소개하며, 다양한 기계 학습 알고리즘들이 이 패러다임의 구현이라고 설명합니다.

- **Technical Details**: 논문에서 제안하는 실용 학습 패러다임은 여러 기술적 개념을 포함합니다. 객체의 특징은 두 가지 속성, 즉 숨겨진(hidden) 피드백과 드러난(manifested) 특징으로 나뉘며, 이들은 각각 수치적 값으로 표현됩니다. 저자는 학습할 의존성을 나타내는 함수와 사례의 정의, 훈련 세트의 형성 과정을 상세히 기술하고 있으며, 이러한 과정에서 '가설(hypothesis)'의 선택이 중요하다고 강조합니다.

- **Performance Highlights**: 저자는 다양한 인기 있는 학습 알고리즘(k-NN, 의사결정 트리, SVM 등)이 실용 학습 패러다임의 원칙을 따르고 있음을 보여줍니다. 이는 실제 데이터에서 효과적인 학습을 가능하게 하며, 기존의 통계적 학습 이론보다 더 현실적인 접근 방식을 제시합니다. 이로 인해 많은 실제 문제에 대한 해결책을 제공할 수 있는 가능성을 내포하고 있습니다.



### Creating Artificial Students that Never Existed: Leveraging Large Language Models and CTGANs for Synthetic Data Generation (https://arxiv.org/abs/2501.01793)
- **What's New**: 이 연구에서는 AI와 딥러닝 기술의 잠재력을 탐색하며, 특히 Generative Adversarial Networks (GANs)와 Large Language Models (LLMs)을 통해 합성 표 데이터(synthetic tabular data)를 생성하는 가능성을 검토합니다. 품질 높은 학생 데이터에 대한 접근은 학습 분석(learning analytics)을 발전시키기 위해 중요하지만, 개인 정보 보호 우려와 엄격한 데이터 보호 규정으로 인해 이러한 데이터의 접근이 제한되고 있습니다. 이에 따라 합성 데이터가 효과적인 대안이 될 수 있음을 제시합니다.

- **Technical Details**: 본 연구에서는 CTGAN이라는 인기 있는 GAN 모델과 GPT2, DistilGPT2, DialoGPT의 세 가지 LLM을 사용하여 합성 학생 데이터를 생성합니다. 생성된 합성 데이터는 실제 학생 데이터와 유사한 품질을 보여주며, 이는 강력한 가능성을 시사합니다. 다양한 유틸리티 평가 메트릭을 사용하여 합성 데이터의 통계적 및 예측 성능을 평가하고, 사용된 생성기 모델 간의 성과를 비교합니다.

- **Performance Highlights**: 연구 결과는 고품질의 합성 데이터 세트를 생성할 수 있는 방법들의 강력함을 뒷받침합니다. 특히 LLM의 성과를 강조하며, 학습 분석 분야에 합성 데이터 사용에 대한 귀중한 통찰력을 제공합니다. 이 연구는 학습 분석 데이터 생성을 위한 새로운 혁신적인 접근법으로의 연구 기초를 마련하는 것을 목표로 합니다.



### Can Synthetic Data be Fair and Private? A Comparative Study of Synthetic Data Generation and Fairness Algorithms (https://arxiv.org/abs/2501.01785)
- **What's New**: 이 연구에서는 Synthetic Data Generators (SDGs)가 프라이버시와 공정성 사이에서 균형을 이루는 최상의 방법을 탐구합니다. 기존 연구에서는 공정성과 프라이버시 간의 역관계가 있다고 여겨졌으나, 본 연구는 DEbiasing CAusal Fairness (DECAF) 알고리즘이 두 가지 간의 최적 균형을 달성했다고 발표합니다. 또한, 전처리 공정성 알고리즘을 사용한 경우 합성 데이터의 공정성이 실제 데이터보다 더 향상된다는 점을 발견했습니다. 이러한 결과는 공정성을 높이기 위한 유망한 접근 방식을 나타냅니다.

- **Technical Details**: 종합적인 방법론을 통해 3개의 실제 데이터셋으로부터 5개의 SDG를 사용하여 합성 데이터셋을 생성하고, 4개의 광범위하게 사용되는 프라이버시 메트릭을 통해 평가했습니다. 이후 합성 데이터셋을 기반으로 4개의 ML 모델을 훈련시키고 3개의 공정성 메트릭을 사용하여 모델의 공정성을 평가했습니다. 결과적으로 프라이버시와 공정성 간의 상충 문제를 분석하며, 다양한 SDG에 대한 연구 질문을 통해 과거 연구의 한계를 극복하고자 하였습니다.

- **Performance Highlights**: DECAF 알고리즘을 통해 프라이버시와 공정성의 균형이 잘 맞춰졌으나, 예측 정확도 측면에서는 유틸리티가 저하되었습니다. 합성 데이터로 생성된 데이터셋과 전처리 알고리즘의 결합이 실제 데이터와 공정성 알고리즘의 조합보다 더 나은 예측 결과를 도출하였음을 확인했습니다. 이 연구는 LA 연구 분야에서 프라이버시와 공정성을 모두 고려한 데이터 생성 및 알고리즘의 사용에 대한 중요한 정책적 시사점을 제공합니다.



### A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning (https://arxiv.org/abs/2501.01774)
- **What's New**: 이 논문에서는 Temporal Difference (TD) 및 Fitted Q-Iteration (FQI) 알고리즘의 수렴 조건을 새로운 관점에서 통합하여 설명합니다. 기존의 관점에서는 TD가 하나의 업데이트를 수행하고 FQI는 무한히 많은 업데이트를 수행한다는 일반적인 이해가 있었습니다. 그러나 본 연구는 이 알고리즘들이 사실상 같은 반복적 방법으로 동작하며, 서로 다른 전처리기(preconditioner)와 행렬 분할(matrix splitting) 기법을 사용한다는 것을 강조합니다.

- **Technical Details**: 연구는 선형 가치 함수 근사를 강조하고, TD, FQI, Partial Fitted Q-Iteration (PFQI)을 Least Squares Temporal Difference (LSTD) 시스템을 해결하기 위한 동일한 반복적 방법으로 묘사합니다. TD는 상수 전처리기를 사용하고, FQI는 데이터 기반의 적응형 전처리기를 사용하며, PFQI는 이 둘 사이의 전처리기를 전환합니다. 이 관점은 알고리즘 수렴 조건을 분석하는 것을 단순화시키고 많은 혼란을 해소합니다.

- **Performance Highlights**: 이 연구는 TD 및 FQI 각각의 수렴을 완전히 특성화하며, 선택된 특성과 관련된 특정 속성을 가정하지 않고도 이들을 분석합니다. 또, 기능 표현(feature representation)에 대한 일반적인 가정이 수렴에 미치는 영향을 조사하고, 수렴을 위해 중요한 새로운 기능조건을 발견합니다. 이러한 발견은 알고리즘 간의 수렴 연결을 확립하고 중요한 질문들에 답변하는 데 기여합니다.



### SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation (https://arxiv.org/abs/2501.01765)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 세이프티 얼라인먼트(safety alignment)가 LoRA(저순위 적응) 방식으로 미세 조정 시 감소하는 문제를 다룹니다. 연구자들은 SaLoRA라는 새로운 기술을 제안하여, 기존 LoRA 방식의 구조적 문제를 해결하고 LLM의 세이프티 기능을 유지하면서도 효율적인 미세 조정을 가능하게 합니다. SaLoRA는 고정된 세이프티 모듈과 작업별 초기화 방식을 채택하여, 모델의 안전성을 보존하는 동시에 향상된 성능을 보여줍니다.

- **Technical Details**: SaLoRA는 기계 학습 모델에서 컨트롤 가능한 세이프티 기능을 유지하기 위해 설계되었습니다. 이 방법은 약 300개의 사전 수집된 불리한 프롬프트와 그에 대한 안전한 응답을 기반으로 설정된 고정된 세이프티 모듈 𝐂_{SaLoRA}를 사용합니다. SaLoRA는 추가 어댑터 𝐀_{SaLoRA}와 𝐁_{SaLoRA}의 학습 가능한 파라미터를 작업 특정 초기화 방식으로 초기화하여 더 나은 수렴성을 제공합니다.

- **Performance Highlights**: SaLoRA는 다양한 평가 지표에서 기존의 다양한 어댑터 기반 방법들보다 뛰어난 성능을 보였습니다. 실험 결과, SaLoRA는 LLM의 원본 안전성을 유지하면서도 다운스트림 평가 작업에서 동등하거나 우수한 결과를 기록했습니다. 이는 SaLoRA가 LLM의 미세 조정에 적절한 솔루션을 제시한다고 할 수 있습니다.



### Catch Causal Signals from Edges for Label Imbalance in Graph Classification (https://arxiv.org/abs/2501.01707)
Comments:
          ICASSP 2025

- **What's New**: 이번 연구에서는 그래프 구조의 causal effects를 탐지할 때 edge features의 중요성을 강조합니다. 기존의 방법들이 edge 정보를 간과하고 있는 상황에서, 이 논문은 causal attention 메커니즘을 강화하여 이러한 edge 정보를 효과적으로 활용합니다. 이를 통해 label imbalance 문제를 가진 그래프 분류 작업에서의 성능을 향상시키고, 더욱 포괄적인 causal 신호를 포착할 수 있는 새로운 방향을 제시합니다.

- **Technical Details**: 본 논문에서는 Graph Neural Networks (GNNs)와 structured causal models에 기반하여 edge-enhanced causal attention 메커니즘을 구현합니다. 특히, 우리는 두 가지 edge-enhanced 모듈을 도입하여 causal subgraph를 효과적으로 탐지하도록 설계한 EGATv1 및 EGATv2를 활용합니다. 또한, causal subgraph와 trivial subgraph를 구분하는 과정에서 각 그래프의 edge 및 node attention score를 산정하여 최적의 그래프 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, PTC, Tox21, 및 ogbg-molhiv와 같은 실제 데이터셋에서 label imbalance 그래프 분류 작업의 성능이 기존의 baselines 대비 향상되었음을 확인했습니다. 이 접근 방식은 edge features를 포함하는 causal 모델이 보다 효과적으로 causal 구조를 탐지할 수 있도록 하는 데 기여하고 있습니다. 따라서, edge features의 활용이 그래프 causal detection 및 label imbalance 문제 해결에 중요하다는 점을 강조합니다.



### Comparative Study of Deep Learning Architectures for Textual Damage Level Classification (https://arxiv.org/abs/2501.01694)
- **What's New**: 이 연구는 항공 산업에서 안전의 중요성을 강조하며, 경미한 운영 이상도 큰 영향을 미칠 수 있음을 지적합니다. 사건 및 사고에 관한 포괄적인 문서화는 근본 원인 파악과 안전 대책 제안에 기여합니다. 그러나 사건 서술의 비구조적 특성이 컴퓨터 시스템의 해석에 도전을 제공합니다.

- **Technical Details**: 본 연구는 자연어 처리(Natural Language Processing, NLP)와 심층 학습(deep learning) 모델을 활용하여 사건 서술을 분석하고 항공기 손상 수준을 분류하는 것을 목표로 했습니다. LSTM, BLSTM, GRU, sRNN 등 다양한 심층 학습 모델을 구현하여 성과를 도출하였으며, 모든 모델이 88% 이상의 정확도를 달성하여 25%의 무작위 추측(threshold)을 크게 초과하는 성과를 보였습니다.

- **Performance Highlights**: 특히 sRNN 모델은 리콜(recall)과 정확도에서 최고의 성과를 보여주었으며, 89%라는 뛰어난 수치를 기록했습니다. 이러한 발견은 NLP와 심층 학습 모델이 비구조적 텍스트 서술에서 실행 가능한 통찰력을 추출하는 데 큰 잠재력을 가지고 있음을 강조합니다.



### Denoising and Adaptive Online Vertical Federated Learning for Sequential Multi-Sensor Data in Industrial Internet of Things (https://arxiv.org/abs/2501.01693)
- **What's New**: 본 연구에서는 Denoising and Adaptive Online Vertical Federated Learning (DAO-VFL) 알고리즘을 제안하여 실시간 데이터 수집 및 처리에서 발생하는 통신 오버헤드와 프라이버시 문제를 해결합니다. DAO-VFL은 산업 조립 라인 시나리오에 맞춰 설계되었으며, 여러 센서들이 데이터를 수집하는 과정에서 발생하는 도전 과제를 효과적으로 다룹니다. 특히, 신호 잡음을 줄이고 적응적 로컬 반복 결정을 통해 실시간 데이터를 지속적으로 처리할 수 있습니다.

- **Technical Details**: DAO-VFL은 온라인 학습 접근 방식을 기반으로 하며, 센서의 이종성과 통신 잡음 문제를 해결하기 위해 설계되었습니다. 이 알고리즘은 각 센서가 자신의 로컬 데이터를 처리한 후 피처 임베딩을 생성하여 서버로 전송하고, 최종 결과를 도출하기 위해 서버에서 이를 처리합니다. 또한, 스스로 학습 성능과 지연을 최적화하는 문제를 깊이 강화 학습(d deep reinforcement learning) 접근으로 해결합니다.

- **Performance Highlights**: 실험 결과, CIFAR-10 및 C-MAPSS와 같은 두 가지 실제 데이터셋에서 DAO-VFL 알고리즘은 기존의 벤치마크 알고리즘에 비해 우수한 성능을 보여주었습니다. DAO-VFL의 잡음 감소 기능과 적응적 로컬 반복 결정 메커니즘의 상세 분석을 통해 그 효율성을 검증하였습니다. 이러한 결과는 DAO-VFL이 IIoT 환경에서 널리 적용 가능함을 시사합니다.



### Analyzing Aviation Safety Narratives with LDA, NMF and PLSA: A Case Study Using Socrata Datasets (https://arxiv.org/abs/2501.01690)
- **What's New**: 이 연구는 1908년부터 2009년까지의 Socrata 데이터셋에 대해 Latent Dirichlet Allocation (LDA), Nonnegative Matrix Factorization (NMF), Probabilistic Latent Semantic Analysis (PLSA)와 같은 주제 모델링 기법의 적용을 탐구합니다. 주체 유형(군사, 상업, 개인)별로 분류하여 조종사 오류, 기계적 고장, 기상 조건 및 훈련 부족 등의 주요 주제를 식별했습니다. 각 기법의 고유한 장점을 강조하며, LDA는 중첩된 주제를 발견하는 데 능하고, NMF는 구별 가능하고 해석 가능한 주제를 생성하며, PLSA는 해석적 복잡성에도 불구하고 미세한 확률적 통찰력을 제공합니다.

- **Technical Details**: 통계 분석 결과, PLSA는 0.32의 coherence score와 -4.6의 perplexity 값을 기록했으며, NMF는 0.34와 37.1, LDA는 0.36의 최고 coherence를 달성했으나 38.2로 가장 높은 perplexity를 나타냈습니다. 이러한 결과는 비구조화된 항공 안전 내러티브에서 실행 가능한 통찰력을 추출하는 데 있어 주제 모델링의 가치를 보여줍니다. 이 연구는 향후 추가적인 맥락 변수를 통합하고 신경 주제 모델을 활용하며 항공 안전 프로토콜을 개선하는 방향을 제시합니다.

- **Performance Highlights**: 이 연구는 항공 안전 관리 분야에서의 고급 텍스트 마이닝 응용의 기초를 제공합니다. 각 주제 모델링 기법이 어떻게 서로 다른 인사이트를 제공하는지를 보여주며, 항공 산업의 위험 요소와 개선이 필요한 영역을 식별하는 데 기여하고 있습니다. 결과적으로, 주제 모델링은 항공 안전 분야의 데이터 분석에 필수적인 도구로 자리 잡을 것으로 기대됩니다.



### Inversely Learning Transferable Rewards via Abstracted States (https://arxiv.org/abs/2501.01669)
- **What's New**: 이 논문에서는 새로운 방법을 제안하여, 관찰된 작업의 고유한 선호도를 공유하는 이전에 보지 못한 작업에 대해 IRL을 일반화할 수 있도록 합니다. 여기서는 두 개 이상의 서로 다른 작업 인스턴스에서 행동 경로를 사용하여 추상적 보상 함수(abstract reward function)를 역으로 학습하는 프로세스를 다룹니다. 이는 로봇 작업의 경우 새로운 작업에 대한 학습을 필요로 하면서도 이를 처음부터 프로그래밍하지 않고도 처리할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 방법은 Variational Autoencoder (VAE)를 활용하여 두 개 이상의 다른 작업 도메인 인스턴스의 관찰된 행동 데이터를 입력으로 사용합니다. 여러 디코더와 결합된 단일 인코더를 통해 인스턴스 경로를 재구성하며, 공통의 잠재 변수(common latent variable)를 기반으로 추상적 보상 함수를 형성합니다. 이 과정은 서로 다른 인스턴스에서 관찰된 두 개 이상의 작업 행동이 필요하며, 이를 통해 작업을 수행하는 데 필요한 공통의 내재적 선호도를 학습합니다.

- **Performance Highlights**: T-IRL로 레이블이 붙은 이 방법은 OpenAI Gym 도메인과 AssistiveGym의 여러 벤치마크를 평가하였습니다. 우리는 각 도메인에서 두 개의 다른 인스턴스의 경로를 VAE의 입력으로 사용하고, 역으로 학습된 추상적 보상 함수가 도메인의 세 번째 정렬된 인스턴스에서 올바른 행동을 배우는 데 어떻게 도움이 되는지를 보여주었습니다. 이 결과는 단순한 도메인에서 이루어졌지만, 이전 문헌에서는 제공되지 않았던 일반화 가능성의 수준을 제안합니다.



### FairSense: Long-Term Fairness Analysis of ML-Enabled Systems (https://arxiv.org/abs/2501.01665)
Comments:
          In Proceedings of the 47th International Conference on Software Engineering (ICSE 2025)

- **What's New**: 이번 논문에서는 ML(기계 학습) 모델의 장기적인 공정성 문제를 다루는 새로운 프레임워크인 FairSense를 제안합니다. 기존의 공정성 평가 방법들이 정적 환경에서 모델 중심으로 이루어졌다면, FairSense는 역동적인 환경에서 시스템의 결정이 환경에 미치는 영향을 분석합니다. 이를 통해 개발자들이 초기 설계 단계에서 장기적인 공정성에 중점을 두고 의사 결정을 할 수 있도록 지원합니다.

- **Technical Details**: FairSense는 Monte-Carlo 시뮬레이션을 활용하여 각 시스템 구성의 진화 추적을 생성하고, 가능한 구성의 민감도 분석을 통해 시스템과 환경 요소가 장기적인 공정성에 미치는 영향을 파악합니다. 이 과정에서 시스템 매개변수와 환경 모델을 정의하고, 시스템 결정으로 인한 환경 변화의 동적 특성을 고려하여 불확실성을 반영한 분포 변화 모델을 사용합니다. 이러한 방식으로 FairSense는 설계 옵션과 환경 요인이 언급된 공정성을 유지하는 데 얼마나 중요한지 평가합니다.

- **Performance Highlights**: FairSense의 유용성을 입증하기 위해 논문에서는 대출, 오피오이드 위험 점수 매기기, 예측 경찰 업무와 같은 세 가지 실제 사례 연구를 수행하였습니다. 사례 연구를 통해 FairSense가 시스템 결정의 디자인 옵션이 장기적인 공정성에 미치는 영향을 체계적으로 분석할 수 있음을 보여주었습니다. 특히, 시스템의 유틸리티와 장기적인 공정성 지표 간의 균형을 맞추는 방법에 대한 인사이트도 제공하였습니다.



### Look Back for More: Harnessing Historical Sequential Updates for Personalized Federated Adapter Tuning (https://arxiv.org/abs/2501.01653)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 데이터 이질성 문제를 해결하기 위해 개인화된 연합 학습(PFL: Personalized Federated Learning) 프레임워크인 pFedSeq를 제안합니다. pFedSeq는 클라이언트의 과거 업데이트 정보를 활용하여 적응형 모델을 개인화하는 방식으로, 기존의 접근 방식에서 나타나는 최적화 문제를 해결하고 성능을 향상시킵니다. pFedSeq는 클라이언트의 적절한 업데이트를 반영함으로써 모델 개인화를 위한 보다 강력하고 일관된 솔루션을 제공합니다.

- **Technical Details**: pFedSeq는 클라이언트의 적응형 업데이트를 순차적으로 처리하는 기능을 갖춘 서버에서 훈련되는 순차적 학습기(Sequential Learner)를 포함하고 있습니다. 이 과정에서 선택적 상태 공간 모델(SSM: Selective State Space Model)을 사용하여 시간을 의식한 선택성 및 효율성을 갖춘 데이터 처리를 구현합니다. 클라이언트의 과거 업데이트 정보를 통합함으로써 pFedSeq는 고차원 데이터의 상호작용을 효과적으로 캡처할 수 있습니다.

- **Performance Highlights**: pFedSeq는 CIFAR-100, Tiny-ImageNet, DomainNet, Omniglot의 네 개 주요 벤치마크 데이터셋에서 평가되었습니다. 그 결과, pFedSeq는 기존의 최첨단 PFL 메서드보다 최고 5.39% 높은 성능을 달성하며 그 우수성을 입증하였습니다. 이러한 성능 개선은 개인화된 적응형 모델 훈련의 새로운 경로를 제공하며 연합 학습의 잠재력을 더욱 끌어올립니다.



### AVATAR: Adversarial Autoencoders with Autoregressive Refinement for Time Series Generation (https://arxiv.org/abs/2501.01649)
Comments:
          This work has been accepted to the SDM 2025 on December 20, 2024

- **What's New**: 본 연구에서는 AVATAR라는 새로운 프레임워크를 소개합니다. AVATAR는 Adversarial Autoencoders (AAE)와 Autoregressive Learning을 결합하여 시간 시계열 데이터의 생성과 관련된 독특한 문제를 해결합니다. 이 프레임워크는 감독 네트워크를 도입하여 시퀀스에서 조건부 분포를 학습하는 데 도움을 주며, 혁신적인 손실 함수인 distribution loss를 통해 효율성을 극대화합니다.

- **Technical Details**: AVATAR는 시계열 데이터의 고유한 동적 특성을 캡처하기 위해 모든 네트워크를 동시에 훈련하는 공동 훈련 메커니즘을 활용합니다. 이 방법은 Latent Representation의 집합적 후방 분포를 사전 Gaussian 분포와 정렬하는 데 도움을 주며, GRU 모델에서 배치 정규화(batch normalization)를 정규화 기법으로 채택하여 네트워크 성능을 최적화합니다.

- **Performance Highlights**: 다양한 실제 및 합성 다변량 시계열 데이터셋을 통해 AVATAR의 효과를 입증했습니다. 실험 결과, AVATAR는 현실적인 시계열 데이터를 생성하는 데 있어 기존의 벤치마크를 지속적으로 초과 달성했습니다. 이는 AVATAR가 시계열 생성 시 품질과 실제 유용성을 모두 향상시키는 데 기여함을 보여줍니다.



### A Probabilistic Model for Node Classification in Directed Graphs (https://arxiv.org/abs/2501.01630)
Comments:
          33 pages, 5 figures

- **What's New**: 이번 연구에서는 노드에 속성이 있으며 레이블이 있는 방향 그래프를 위한 확률 모델을 제안합니다. 이 모델은 최대 우도(maximum likelihood) 또는 최대 사후(estimation) 추정을 통해 보이지 않는 노드의 레이블을 예측할 수 있는 생성적 분류기(generative classifier) 역할을 합니다. 제안된 모델은 Node Classification 분야에서 그래프 신경망(Graph Neural Networks)보다 더 높은 해석 가능성을 제공합니다.

- **Technical Details**: 제안하는 확률 모델은 방향성 가중 그래프의 근본적인 행동을 설명하며, 연결된 노드의 레이블 분포 및 행동을 정의합니다. 이를 통해 우리는 최대 우도 추정과 최대 사후 추정의 두 가지 접근법을 통해 예측을 수행합니다. 또한, 모델 추론 과정은 계산적으로 비용이 많이 들지 않으며, 예측 과정이 병렬화(parallelized)될 수 있어 실행 시간을 향상시킬 수 있습니다.

- **Performance Highlights**: 모델을 적용한 두 개의 데이터셋에서 분류 성능을 평가하였으며, 그래프 신경망을 포함한 다른 방법들과 비교했을 때 경쟁력 있는 결과를 얻었습니다. 특히 Math Genealogy Project에서 파생된 새로운 데이터셋을 사용하여 우리 모델의 성능을 검증했습니다. 이 연구는 방향성이 있는 속성 그래프를 위한 새로운 확률 모델 및 node classification에 대해 해석 가능한 예측을 생성하는 방법을 제안하고 있습니다.



### Crossing Language Borders: A Pipeline for Indonesian Manhwa Translation (https://arxiv.org/abs/2501.01629)
- **What's New**: 본 연구에서는 인도네시아어에서 영어로의 만화(Manhwa) 번역을 자동화하기 위한 효율적인 솔루션을 제안합니다. 컴퓨터 비전, 텍스트 인식, 자연어 처리 기법을 결합하여 전통적인 번역 방식의 비효율성을 해소하고자 합니다. 이 시스템은 의사 대화의 구간을 탐지하고, 문자를 인식하며, 이를 번역하여 만화 패널에 다시 오버레이하는 단계를 포함합니다.

- **Technical Details**: 연구에 사용된 주요 기법은 YOLOv5xu를 활용한 스피치 버블(spoech bubble) 탐지, Tesseract를 통한 광학 문자 인식(Optical Character Recognition), 그리고 MarianMT를 통한 기계 번역입니다. YOLOv5xu는 고정밀 객체 감지를 위해 미세 조정되었고, Tesseract는 인도네시아어 모델을 사용하여 문자를 효율적으로 인식합니다. 마지막으로 번역된 텍스트는 OpenCV와 Pillow 라이브러리를 통해 원본 이미지의 대화 상자 형태에 맞게 재배치됩니다.

- **Performance Highlights**: 모델의 성능은 YOLOv5xu가 90.7%의 F1 점수를 나타내어 스피치 버블을 효과적으로 검출했다는 것을 보여줍니다. OCR의 경우, Tesseract를 활용해 CER 3.1%, WER 8.6%의 성능을 기록하며 기존 방법보다 우수한 결과를 달성하였습니다. 또한, MarianMT 모델은 BLEU 점수와 Meteor 점수를 통해 번역의 의미적 보존이 잘 이루어졌음을 증명했으며, 이 모든 단계를 통합한 자동화 파이프라인은 만화 번역 작업의 효율성을 높이는데 기여하고 있습니다.



### Adaptive Meta-learning-based Adversarial Training for Robust Automatic Modulation Classification (https://arxiv.org/abs/2501.01620)
Comments:
          Submitted to IEEE International Conference on Communications (ICC) 2025

- **What's New**: 이 논문은 딥러닝 기반의 자동 변조 분류(AMC) 모델이 새로운 적대적 공격에 대해서도 강력하게 적응할 수 있도록 하는 메타 학습 기반의 적대적 훈련 프레임워크를 제안합니다. 이 프레임워크는 리얼타임 상황에서의 새로운 공격에 신속하게 적응할 수 있는 접근 방식을 구현하여 AMC 모델의 강건성을 대폭 향상시킵니다. 기존의 적대적 훈련 방식과는 달리, 최소한의 훈련 샘플 만으로도 새로운 적대적 공격에 대응할 수 있는 모델을 개발하였습니다.

- **Technical Details**: 이 프레임워크는 기존의 적대적 훈련 메커니즘과 달리, 특정한 공격 유형에 대한 강건성을 넘어서는 일반화된 적응 전략을 학습할 수 있게 설계되었습니다. 이는 모델이 새로운 적대적 샘플에 대한 훈련이 부족하더라도 뛰어난 일반화 성능과 강건성을 달성할 수 있게 합니다. 메타 학습 기법을 적용하여 AMC 모델이 실시간 환경에서 적대적 공격에 효과적으로 대응할 수 있도록 돕습니다.

- **Performance Highlights**: 제안된 메타 학습 기반의 훈련 프레임워크는 기존의 AMC 모델에 비해 대폭 더 낮은 온라인 훈련 시간으로 높은 정확성과 강건성을 보여줍니다. 이는 실제 환경에서 이 모델의 활용 가능성을 극대화하며, 적대적 공격에 대한 저항력을 일반화하여 새로운 공격을 효과적으로 회피할 수 있도록 합니다. 따라서 이 연구는 AMC 응용 프로그램의 실용적인 구현에 중요한 기여를 할 것으로 기대됩니다.



### Online Meta-Learning Channel Autoencoder for Dynamic End-to-end Physical Layer Optimization (https://arxiv.org/abs/2501.01608)
Comments:
          To be published in IEEE Wireless Communications and Networking Conference (WCNC) 2025

- **What's New**: 본 논문에서는 동적 채널을 위한 few-shot Channel Autoencoder(CAE) 시나리오를 위한 Online Meta Learning Channel AE (OML-CAE) 프레임워크를 제안합니다. 전통적인 CAE 디자인에서는 정적 시나리오에만 초점을 맞추었지만, OML-CAE는 변화하는 채널 조건에 실시간으로 적응할 수 있도록 설계되었습니다. 이로 인해 제한된 수의 파일럿 신호를 사용하여도 효과적으로 동작할 수 있는 가능성을 보여줍니다.

- **Technical Details**: OML-CAE는 메타-러닝(Meta-learning) 기법을 활용하여 새로운 채널 조건에 적응할 수 있는 능력을 강화합니다. 이는 CAE가 공급하는 데이터를 기반으로 한 통합 최적화 접근 방식의 중요한 발전을 의미합니다. 더욱이 이 연구는 CAE 설계를 온라인 학습 환경으로 변경하여 기존의 오프라인 기반 훈련 방법의 제약을 극복하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 제안된 OML-CAE는 실제 동적 시나리오에서도 기존의 CAE보다 적은 파일럿 신호로 높은 정확도를 달성할 수 있습니다. 이는 통신 시스템의 효율성을 증가시키고 데이터 전송 속도를 향상시킵니다. 제한된 훈련 샘플로도 개선된 성능을 발휘할 수 있는 OML-CAE는 현실적인 요건을 충족하는 CAE 디자인의 실행 가능성을 크게 높입니다.



### Multivariate Time Series Anomaly Detection using DiffGAN Mod (https://arxiv.org/abs/2501.01591)
Comments:
          19 pages, 3 figures, 1 table

- **What's New**: 본 논문은 멀티변량 시계열 이상 탐지에 있어 새로운 방법인 DiffGAN을 제안합니다. 이 방법은 기존의 고정 확산 단계의 한계를 극복하고, 생성적 적대 신경망(Generative Adversarial Network, GAN) 요소를 추가하여 노이즈가 있는 데이터를 동시에 생성하고 확산 단계를 예측할 수 있도록 합니다. 실험 결과, DiffGAN은 여러 최신 모델과 비교했을 때 이상 탐지 성능이 우수함을 보여주었습니다.

- **Technical Details**: DiffGAN은 전방 확산 응답 과정에 생성기를 결합하고, 판별기가 데이터에서 추가된 노이즈 수준을 평가하여 복원에 필요한 단계를 예측합니다. 이 방식은 전통적인 확산 모델에서의 고정된 확산 단계의 한계를 극복하여 모델의 유연성과 적응성을 강화합니다. 또한, 본 논문에서는 효과적인 이상 탐지를 위한 판별자의 역할을 재조명하며, 이를 데이터 인코딩 과정의 제어기로 간주합니다.

- **Performance Highlights**: DiffGAN은 여러 멀티변량 시계열 데이터셋에서 기존 벤치마크에 비해 뛰어난 탐지 정확도를 보였습니다. 이 연구 결과는 이상 탐지 분야에서 DiffGAN의 상당한 가능성을 시사합니다. 제안된 방법의 관련 코드와 데이터셋은 DiffGAN에서 공개 검토할 수 있습니다.



### Stackelberg Game Based Performance Optimization in Digital Twin Assisted Federated Learning over NOMA Networks (https://arxiv.org/abs/2501.01584)
- **What's New**: 이번 논문에서는 Digital Twin (DT) 기술을 활용하여 Federated Learning (FL) 시스템의 성능을 향상시키는 새로운 방안을 제안합니다. 기존 FL 시스템의 문제인 straggler 문제를 해결하기 위해 DT를 비추어보고, NOMA 네트워크를 통해 FL 교육 과정을 지원합니다. 더불어, 악의적인 클라이언트 업데이트로부터 시스템을 보호하기 위한 신뢰성 기반 클라이언트 선택 방안도 포함되어 있습니다.

- **Technical Details**: 제안된 시스템은 Stackelberg 게임 이론을 활용하여 클라이언트와 서버 간의 에너지 소비 및 지연 시간을 최소화하는 최적화 문제를 다룹니다. 클라이언트는 에너지 소비를 최소화하는 것을 목표로 하며, 서버는 FL 교육의 지연 시간을 줄이는 데 중점을 둡니다. 시스템은 DT의 실시간 매핑 기능을 활용하여 클라이언트의 상태 정보를 분석하고 이로 인해 발생하는 리소스 부족 문제를 해결합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 시스템은 전반적인 성능을 크게 향상시키는 것으로 나타났습니다. 신뢰성 기반 클라이언트 선택 방안이 FL 시스템의 안정성을 높이는 데 기여하고 있으며, 에너지 소비와 지연 시간을 동시에 효과적으로 줄이고 있습니다. 이 연구는 FL 시스템의 효율성과 안정성을 극대화하는 방향으로 새로운 통찰력을 제공하고 있습니다.



### Semialgebraic Neural Networks: From roots to representations (https://arxiv.org/abs/2501.01564)
- **What's New**: 이 논문에서는 Semialgebraic Neural Networks (SANNs)라는 새로운 신경망 아키텍처를 소개하고 있습니다. 이 아키텍처는 모든 제한된 semialgebraic function을 정확히 표현하고 계산할 수 있는 능력을 가지고 있습니다. 새로운 접근법으로, 고전적인 수치 해석 기법과 ReLU 활성 함수가 결합되어 특정 수치 ODE 솔버의 정확도에 맞게 기능을 학습합니다.

- **Technical Details**: SANNs는 입력으로 받은 데이터에 대해 ODE 초기값 문제를 통해 정의된 벡터 필드를 계산합니다. 이는 연속적인 piecewise polynomial G를 사용하여 구현됩니다. 이 방법은 비연속 semialgebraic function도 표현할 수 있으며, 각 연결된 구성 요소에서의 연속적인 방법을 통해까지 계산할 수 있습니다. SANNs는 전통적인 딥러닝 기술을 통해 훈련될 수 있습니다.

- **Performance Highlights**: SANN 아키텍처는 모든 Bounded semialgebraic function을 정확하게 계산할 수 있는 최초의 신경망이라고 할 수 있습니다. 이전 연구에 비해 고차원 데이터에 대해 임의의 제한된 semialgebraic function을 처리하는 데 있어 강력함을 발휘합니다. 논문에서는 homotopy continuation 방법을 통해 성능 향상을 입증하는 다양한 실험 결과도 제시하고 있습니다.



### Predicting the Performance of Black-box LLMs through Self-Queries (https://arxiv.org/abs/2501.01558)
Comments:
          28 pages

- **What's New**: 이 논문은 블랙박스(black-box) 접근 방식에서 대형 언어 모델(LLM)의 특징을 추출하는 새로운 방법을 제안합니다. 전통적으로 LLM은 화이트박스(white-box) 접근 방식에서 모델의 상태나 활성화(hidden states)에 대한 접근이 요구되었으나, 이 연구는 후속(prompt) 질문과 응답 확률을 통해 저차원(low-dimensional) 특징을 추출합니다. 본 연구의 접근 방식은 모델의 성능 예측을 개선하며, 흥미롭게도 화이트박스 접근보다 더 뛰어난 성능을 발휘할 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 LLM의 출력을 기반으로 후속 질문을 활용하여 성능을 예측하는 방법을 개발했습니다. 연구자들은 LLM이 생성한 응답에 대해 스스로 설명할 수 있는 능력을 활용하여, 응답의 확률 분포가 정확성 및 모델의 차이에 따라 크게 달라진다고 가정합니다. 이러한 저차원 특징을 학습하여 LLM의 성능을 예측할 수 있으며, 예측 과정에서 여러 다양한 LLM을 사용한 결과가 나타났습니다.

- **Performance Highlights**: 연구 결과, 제안된 후속 질문 접근 방식은 특정 클래스의 예측이나 텍스트 생성이 정확한지를 예측하는 데 매우 효과적일 뿐만 아니라 강력한 일반화 보장을 제공했습니다. 모델의 다양성을 조사한 결과, 후속 질문뿐만 아니라 다양한 자연어 시퀀스도 예측 성능을 높일 수 있다는 흥미로운 발견이 있었습니다. 마지막으로, 해당 방법은 LLM의 아키텍처와 크기 판별에서도 신뢰할 수 있는 결과를 도출할 수 있음을 보였습니다.



### Many of Your DPOs are Secretly One: Attempting Unification Through Mutual Information (https://arxiv.org/abs/2501.01544)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 포스트 정렬(post-alignment) 과정을 단순화하고 명확하게 이해할 수 있도록 도와주는 새로운 통합 프레임워크를 제안합니다. 직접 선호 최적화(Direct Preference Optimisation, DPO) 알고리즘들의 다양성으로 인해 연구자들이 각 기법의 관계를 파악하기 어려운 상황을 개선하고자 합니다. 제안된 프레임워크는 상호 정보(mutual information)에 영감을 얻어 만들어졌으며, 유연한 사전(prior)을 갖는 새로운 손실 함수(loss function)를 포함하고 있습니다.

- **Technical Details**: 본 프레임워크를 통해 SimPO, TDPO, SparsePO 등 여러 기존 알고리즘들이 파생될 수 있음을 보여줍니다. 사전의 세심한 정의는 다양한 DPO 기법들 사이의 명확한 관계를 성립시키는 데 중점을 두고 있습니다. 이로 인해 DPO 알고리즘의 복잡성이 감소하고, 연구자들이 더 많은 통찰을 얻을 수 있는 기회가 마련됩니다.

- **Performance Highlights**: 우리는 제안된 프레임워크가 LLM의 정렬 기술을 더 견고하고 해석 가능하게 만드는 기초가 되기를 바랍니다. 이 논문은 연구 커뮤니티가 LLM 정렬 분야에서 더 발전된 기술을 개발하도록 지원하는 데 중점을 두고 있습니다. 새롭게 제안된 통합 접근 방식은 DPO 알고리즘의 이해를 증진시켜, 향후 연구 활동에 긍정적인 영향을 미칠 것으로 기대됩니다.



### BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery (https://arxiv.org/abs/2501.01540)
Comments:
          KG and MYL contributed equally

- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)에 기반한 과학적 에이전트의 능력을 평가하기 위한 새로운 벤치마크인 BoxingGym을 도입합니다. 10개의 환경을 통해 실험 설계와 모델 발견 능력을 체계적으로 테스트할 수 있습니다. 이를 통해 LLM이 과학 모델을 제안하고 실험 데이터를 수집하며, 새로운 데이터에 따라 모델을 수정하는 능력을 평가하는 시스템이 마련되었습니다.

- **Technical Details**: BoxingGym은 생성적 확률 모델을 사용하여 각 환경을 모델링하며, 과학 에이전트는 인터랙티브한 실험을 수행할 수 있습니다. 연구는 Bayesian Optimal Experimental Design(BOED)과 기대 정보 이득(Expected Information Gain, EIG)을 결합하여 실험 디자인과 모델 발견을 통합적으로 평가합니다. 이를 통해 LLM의 설계된 환경에서 정보 검색 및 실험적 설계를 체계적으로 수행할 수 있는 방법론론을 제시합니다.

- **Performance Highlights**: 현재의 LLM인 GPT-4o는 실험 설계 및 모델 발견에 있어 어려움을 겪고 있으며, 통계 모델을 추가하는 것으로도 이러한 문제를 해결하기 어려운 결과를 보여줍니다. 연구는 LLM의 잠재력을 드러내는 동시에, 실험적 설계와 모델 발견의 통합적 접근 방식을 강조하였습니다. 이는 앞으로 AI 모델이 과학적 발견 과정을 지원할 수 있는 보다 튼튼한 프레임워크 개발을 위한 기반이 됩니다.



### Transfer Neyman-Pearson Algorithm for Outlier Detection (https://arxiv.org/abs/2501.01525)
- **What's New**: 이 논문에서는 목표 비정상 데이터(rare outlier data)가 드문 상황에서의 전이 학습(transfer learning) 문제를 다룹니다. 기존의 균형 잡힌 분류(balanced classification)에서는 전이 학습이 많이 연구되었으나, 비정상 탐지(outlier detection) 및 불균형 분류(imbalanced classification) 환경에서의 전이에 대한 연구는 상대적으로 적었습니다.

- **Technical Details**: 저자들은 이론적으로 비정상 분포(abnormal distribution)의 다양한 변화에 강한 보장을 제공하는 일반 메타 알고리즘(meta-algorithm)을 제안합니다. 이 알고리즘은 다층 신경망(multi-layer neural networks) 기반의 다양한 구현(instantiations)을 조사하며, 전통적인 균형 잡힌 분류를 위한 방법들의 자연스러운 확장이 아닌 성능 향상을 입증합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존 전이 방법들의 평균 성능을 초과하며 더 나은 결과를 보였습니다. 이는 비정상 탐지 문제에 더욱 효과적인 접근법이 될 수 있음을 시사합니다.



### Improving Robustness Estimates in Natural Language Explainable AI though Synonymity Weighted Similarity Measures (https://arxiv.org/abs/2501.01516)
Comments:
          10 pages, 2 figures, 4 tables

- **What's New**: 이 논문에서는 Explainable AI (XAI) 기술에 있어 적대적 예제(adversarial examples)가 중요한 역할을 한다고 강조합니다. 기존의 유사도 측정 방법들이 적대적 XAI에서 효과적인 비교를 제공하지 못한다는 점을 지적하고, 동의어 가중치(synonymity weighting)를 도입하여 이러한 문제를 해결하고자 합니다. 이 접근 방식은 XAI 방법의 내구성 및 안정성, 즉 안정성을 평가하는 새로운 기준을 제시합니다.

- **Technical Details**: 문헌에서는 XAI 모델의 설명이 복잡한 블랙박스 모델의 출력을 이해하기 위한 방법으로 사용되지만, 기존의 유사도 측정 지표들이 두 가지 주된 결함(sensitivity와 indifference) 때문에 신뢰성이 떨어진다고 설명합니다. 이 연구는 동의어 가중치를 통해 perturbed 단어와 원본 단어 간의 의미적 유사성을 고려하여 단순한 비교 방식을 개선합니다. 이러한 접근법은 텍스트 기반 입력에서의 XAI의 적대적 공격 과정을 평가하기 위한 새로운 기초를 제공합니다.

- **Performance Highlights**: 동의어 가중치를 적용한 유사도 측정은 기존의 일반적인 유사도 측정보다 훨씬 더 정확한 결과를 도출합니다. 실험 결과, 전통적인 유사도 측정 방법으로는 XAI의 불안정성을 잘못 평가할 수 있음을 보여주며, 새로운 방법이 XAI 시스템의 신뢰성을 높이는 데 기여할 수 있다는 결론을 도출합니다. 이를 통해 XAI 방법이 적대적 예제에 대해 보다 강건함을 가질 수 있도록 하는 실질적인 기여를 하고 있습니다.



### DiagrammaticLearning: A Graphical Language for Compositional Training Regimes (https://arxiv.org/abs/2501.01515)
- **What's New**: 이 논문에서는 여러 개의 상호 작용하는 모델 구성 요소들을 사용하는 딥 러닝 체제를 바탕으로 학습 다이어그램을 소개합니다. 학습 다이어그램은 코드가 아닌 데이터로 매개변수화된 학습을 포착하는 그래픽 표현입니다. 이 개념을 통해 사용자들은 복잡한 모델을 더 작고 구성 요소들로 구성할 수 있습니다.

- **Technical Details**: 학습 다이어그램은 고유한 손실 함수(loss function)로 컴파일되며, 이 손실 함수에서 훈련된 모델들은 서로 '합의'하는 예측을 생성합니다. 논문에서는 few-shot multi-task learning, knowledge distillation, multi-modal learning과 같은 인기 있는 학습 설정을 학습 다이어그램으로 표현할 수 있음을 보여줍니다. 또한, PyTorch 및 기타 모델의 다이어그램을 구축할 수 있는 라이브러리를 구현하였습니다.

- **Performance Highlights**: 전통적인 머신 러닝 사용 사례를 실행하여 학습 다이어그램이 사용자들이 복잡한 모델을 구축하고, 워크플로우 간의 관계를 확인하며, 훈련 중 또는 훈련 후에 모델을 조작하는 방법을 보여줍니다. 범주 이론적(framework) 틀을 활용하여 학습 다이어그램에 대한 엄격한 의미론을 도입하였으며, 이는 이러한 작업을 수학적으로 탄탄한 기반 위에 놓이게 합니다.



### TreeLUT: An Efficient Alternative to Deep Neural Networks for Inference Acceleration Using Gradient Boosted Decision Trees (https://arxiv.org/abs/2501.01511)
Comments:
          Accepted by FPGA'25 conference

- **What's New**: 이 논문에서는 FPGA(Field-Programmable Gate Array)에서 GBDT(Gradient Boosted Decision Trees)를 효율적으로 구현하기 위한 오픈 소스 도구인 TreeLUT를 소개합니다. 이 도구는 LUT(Lookup Table)를 활용하여 하드웨어 구현 시 필요한 메모리와 DSP(Digital Signal Processing) 없이도 높은 효율성을 자랑합니다. TreeLUT는 기존의 복잡한 HLS(High-Level Synthesis) 도구에 의존하지 않고, 직접 Verilog 하드웨어 파일로 변환할 수 있는 Python 라이브러리입니다.

- **Technical Details**: TreeLUT는 가지치기된(quantized) GBDT 모델을 효율적으로 하드웨어 구현하기 위한 접근 방식을 제공합니다. 이 도구는 사전 훈련(pre-training)과 사후 훈련(post-training) 합친 양자화 방식으로 훈련을 효율적으로 진행하며, 결정적 트리를 목표 비트폭보다 적은 비트로 양자화할 수 있도록 설계되었습니다. 하드웨어 아키텍처는 완전히 전개된(fully unrolled) 3계층 구조를 사용하여 모듈화, 확장성 및 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과, TreeLUT는 MNIST, JSC, NID 분류 데이터셋에서 경쟁력 있는 정확도를 유지하며 우수한 성능을 발휘했습니다. 기존의 DNN 및 다른 GBDT 방법들과 비교했을 때, TreeLUT는 하드웨어 활용도와 대기 시간(latency), 처리량(throughput) 면에서显著 향상된 결과를 보여주었습니다. 이 연구는 FPGA에서의 GBDT 구현 가능성을 높여주는 중요한 이정표가 될 것으로 기대됩니다.



### Explainable Brain Age Gap Prediction in Neurodegenerative Conditions using coVariance Neural Networks (https://arxiv.org/abs/2501.01510)
Comments:
          Accepted at ISBI, 2025

- **What's New**: 이번 연구에서는 coVariance neural networks (VNN)을 기반으로 뇌 나이 격차(brain age gap)를 분석하였습니다. VNN은 신경퇴행성 질환에 대한 해석 가능한 분석 방법으로, 해부학적으로 이해할 수 있는 특징과 방법론적 해석 가능성을 통해 뇌 건강을 모니터링하는 바이오마커로 활용됩니다. 이 연구는 알츠하이머병, 전두측두치매 및 비정형 파킨슨병과 같은 다양한 신경퇴행성 질환에서의 뇌 나이 격차를 연구하였습니다.

- **Technical Details**: VNN 모델은 샘플 공분산 행렬(sample covariance matrix)을 기반으로 작동하는 그래프 신경망(graph neural networks)으로, 컨볼루션(convolution) 연산을 통해 뇌의 구조적 특징을 학습합니다. 특히 이 모델은 공분산 행렬의 고유벡터(eigenvectors)를 활용하여 다양한 신경퇴행성 질환 간의 해부학적 패턴을 비교하고 설명할 수 있습니다. 이 연구에서는 VNN을 사용하여 두 가지 주요 기능을 제공하며, 해부학적 바이오마커의 파생을 위한 해석 가능성을 높이고 있습니다.

- **Performance Highlights**: 연구 결과, 알츠하이머병, 전두측두치매, 비정형 파킨슨 질환에서 뇌 나이 격차가 건강한 집단에 비해 유의미하게 증가함을 보여주었습니다. VNN 모델은 각 신경퇴행성 질환에 따라 고유하게 공분산 행렬의 고유벡터를 활용하여 해부학적 패턴의 차이를 설명하였으며, 이는 뇌 나이 격차의 해석 가능성을 높이는 데 기여합니다. 또한, 이 연구는 VNN의 적용 가능성을 여러 신경퇴행성 질환에 확대하여 유의미한 결과를 도출한 점에서 이전 연구들보다 더 포괄적인 평가를 제공합니다.



### AI-Enabled Operations at Fermi Complex: Multivariate Time Series Prediction for Outage Prediction and Diagnosis (https://arxiv.org/abs/2501.01509)
Comments:
          Presented in the AAAI Workshop on AI for Time Series Analysis 2025

- **What's New**: 이 논문에서는 Fermilab 가속기 컴플렉스에서의 예기치 않은 사건으로 인한 빔 중단 현상에 대한 예측 및 라벨링을 위한 AI 기반 프레임워크를 제안합니다. 기존의 임계값 기반 경고 시스템은 반응적이며 허위 경고가 자주 발생하고, 중단 원인 라벨링이 일관되지 않아 문제를 야기합니다. 새로운 접근법으로는 예측 분석과 자동 라벨링을 결합하여 장애 예측 성능을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 2703개의 Linac 장치에서 수집한 시간 시계열 데이터를 활용하며, 고급 딥러닝 아키텍처를 평가합니다. 이 연구는 순환 신경망(Recurrent Neural Networks), 주의 기반(Attention-based) 모델 및 선형 모델을 포함한 딥러닝 아키텍처의 성능을 테스트합니다. 또한, 랜덤 포레스트(Random Forest) 기반의 라벨링 시스템을 평가하여 일관성 있고 신뢰할 수 있는 장애 주석을 제공하는 방안을 모색합니다.

- **Performance Highlights**: 이 연구의 결과는 제안된 AI 프레임워크가 다운타임을 줄이고 결정 결정의 질을 개선하는 데 도움을 줄 수 있음을 보여줍니다. 연구 결과는 다양한 딥러닝 아키텍처의 강점과 약점을 강조하며, 적절한 라벨링을 통해 다운타임 관리에 있어 AI의 활용 가능성을 극대화할 수 있는 방법을 제시합니다.



### Drift2Matrix: Kernel-Induced Self Representation for Concept Drift Adaptation in Co-evolving Time Series (https://arxiv.org/abs/2501.01480)
- **What's New**: 이 논문의 주제인 Drift2Matrix는 시간 시계열 데이터에서 개념 변화를 다루는 새로운 프레임워크입니다. 이는 커널 기반의 자기 표현 방법을 활용하여 복합적이고 동적인 시계열 데이터 분석에 적응할 수 있는 능력을 제공합니다. Drift2Matrix는 기존의 정적인 모델에서 발생하는 한계를 극복하고, 동시 변화하는 시계열 간의 상호작용을 캡처하여 보다 효과적으로 개념 변화를 식별하고 적응할 수 있게 합니다.

- **Technical Details**: Drift2Matrix는 커널 유도 자기 표현(kernel-induced self-representation) 기법을 사용하여 시간 시계열 데이터를 행렬 형태로 변환합니다. 이 방법은 시계열 데이터 내의 비선형 상관관계를 포착하여 개념 변화를 추적하고 예측하는 데 유용합니다. 또한, 이 프레임워크는 개념 추적, 예측 및 동태 분석을 위한 세 가지 주요 목적을 설정하여 개념 식별의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, Drift2Matrix는 각기 다른 데이터 세트에서 개념 변화를 효과적으로 다룰 수 있는 능력을 증명했습니다. 예를 들어, 예측에서 Drift2Matrix는 새로운 개념의 출현을 예측할 수 있었으며, 이는 시계열의 동적 상호작용을 고려하는 확률 모델에서 기인합니다. 이러한 결과는 Drift2Matrix가 전통적인 방법에 비해 동적 데이터 환경에서 더 나은 적응성과 정확성을 보여주는 것을 확인시켜줍니다.



### Unraveling Indirect In-Context Learning Using Influence Functions (https://arxiv.org/abs/2501.01473)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 일반화된 In-Context Learning (ICL)을 위한 새로운 패러다임인 간접 ICL(Indirect In-Context Learning)을 소개합니다. 간접 ICL에서는 Mixture of Tasks와 Noisy Demonstrations이라는 두 가지 실제 시나리오에 맞춘 demonstration selection 전략을 탐구합니다. 본 연구에서는 Influence Functions (IFs)를 선택 도구로 활용하여 시연 풀 내에서 사례의 정보성을 보다 효과적으로 포착할 수 있는 가능성을 강조합니다.

- **Technical Details**: 간접 ICL은 대부분이 최종 작업에 적합하지 않은 사례 집합에서 demonstration을 선택하여 간접적인 감독을 제공하는 것을 목표로 합니다. 본 연구에서는 높은 정확도의 demonstration 선택을 위해 최종 작업과의 인덕티브 편향을 나타내는 후보 demonstrations을 식별하는 데 IF를 활용하는 방법이 Practical하다는 것을 보여줍니다. 이를 위해, BertScore-Recall (BSR)과 결합한 IF Surrogate 모델이 Mixture of Tasks 설정에서 3-shot 및 5-shot 설정에 대한 평균 절대 정확도를 각각 0.37% 및 1.45% 향상시킬 수 있음을 입증합니다.

- **Performance Highlights**: Noisy Demonstrations 설정에서는 IF 기반 선택기를 통해 전통적인 ICL 선택기(분석적 평가 방식인 BSR 및 Cosine Similarity)의 가중 평균 선택을 관찰하였으며, 이로 인해 노이즈가 있는 GLUE 벤치마크에서 각각 2.90% 및 2.94%의 정확도 향상을 가져왔습니다. 연구 결과는 IFs가 demonstration 선택에서 단순한 의미적 유사성보다 더 뛰어난 성능을 발휘할 수 있음을 입증합니다. 전반적으로, 이 연구는 ICL의 demonstration 선택을 발전시키고, 과거의 방법론을 넘어서는 강력한 프레임워크를 제안합니다.



### Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation (https://arxiv.org/abs/2501.01472)
- **What's New**: 이번 연구는 실시간 시계열 데이터에 대한 테스트 단계 적응(test-time adaptation, TTA)의 발전을 목표로 하며, 기존의 시각적 작업을 위한 TTA 방법들이 가지고 있는 한계를 극복하려고 합니다. 본 연구에서는 Uncertainty-aware Prototyping(불확실성 인지 프로토타입)과 결합된 Augmented Contrastive Clustering(자극적인 대조 클러스터링) 방법을 제안하면서, 다채로운 시간 정보를 수집하고 신뢰할 수 있는 pseudo label(유사 라벨)을 생성하는 새로운 접근법을 소개합니다.

- **Technical Details**: ACCUP(Augmented Contrastive Clustering with Uncertainty-aware Prototyping)는 시계열 데이터에 최적화된 새로운 TTA 기법으로, 불확실성을 인식하는 프로토타입 앙상블 모듈을 도입합니다. 이 모듈은 magnitude warping augmentation(진폭 왜곡 증대)을 활용하여 시계열 데이터의 변화에 내성이 있는 시간 패턴을 학습하며, 이를 통해 모델의 출력 신뢰도를 높이고 신뢰 가능한 예측을 보장합니다. 또한, 엔트로피 비교 기법을 도입하여 고신뢰 영역의 예측을 선택적으로 수집하고, 이러한 신뢰도를 통해 학습 과정을 개선하는 전략을 채택합니다.

- **Performance Highlights**: 본 연구는 세 개의 실제 시계열 데이터셋과 추가적인 시각적 데이터셋을 통해 ACCUP의 효과를 검증하였습니다. 실험 결과, 제안된 방법은 다양한 시계열 애플리케이션에서 기존 방법들보다 우수한 성능을 보이며, 불확실성이 높은 pseudo label의 부정적인 영향을 최소화하는 데 많은 기여를 하였습니다. 또한, 타임 시리즈 데이터의 독특한 변화를 잘 잡아내며, 각 클래스 간의 명확한 구분을 가능하게 하는 집합적 클러스터링을 증진시켰습니다.



### Balance-aware Sequence Sampling Makes Multi-modal Learning Better (https://arxiv.org/abs/2501.01470)
- **What's New**: 이번 연구에서는 데이터 이질성(data heterogeneity)으로 인해 발생하는 모달리티 불균형(modality imbalance) 문제를 해결하기 위해 Balance-aware Sequence Sampling (BSS) 기법을 제안합니다. 기존의 다중 모달 학습(multi-modal learning) 접근법이 최적화 목표에만 집중했던 반면, 우리는 샘플 순서가 학습 편향을 초래할 수 있음을 강조했습니다. BSS는 학습된 샘플의 균형 정도를 평가하고, 이를 기반으로 점진적으로 훈련 샘플을 제공하는 휴리스틱 스케줄러를 사용합니다.

- **Technical Details**: BSS는 여러 관점에서 샘플의 균형 정도를 평가하기 위해 다중 관점 측정기를 정의한 후, 커리큘럼 학습(curriculum learning) 원칙을 적용하고 있습니다. 이는 균형 잡힌 샘플에서 시작해 점차 불균형 샘플로 진행하는 방식으로, 모델의 성능 향상을 꾀합니다. 또한, 모델 능력이 향상됨에 따라 샘플 균형이 어떻게 발전하는지를 고려하여, 학습 기반 확률적 샘플링 기법을 통해 훈련 순서를 동적으로 업데이트합니다.

- **Performance Highlights**: 많은 실험을 통해 제안한 방법은 최신의 다중 모달 학습 접근법들과 비교하여 우수한 성능을 보이는 것으로 나타났습니다. 실험은 CREMA-D, Kinetics-Sounds, VGGSound 등 다양한 데이터셋에서 수행되었습니다. 특히, Twitter2015 데이터셋에서 균형 잡힌 샘플이 학습 초기에 높은 의미적 일관성을 보임을 확인하였으며, 이는 다중 모달 학습의 효과성에 기여하고 있습니다.



### Goal Recognition using Actor-Critic Optimization (https://arxiv.org/abs/2501.01463)
- **What's New**: 이 논문은 Deep Recognition using Actor-Critic Optimization (DRACO)라는 새로운 목표 인식 알고리즘을 소개합니다. DRACO는 비구조화된 데이터로부터 정책 네트워크를 학습하여 목표를 추론하는 첫 번째 알고리즘이며, 연속적인 정책 표현을 통해 목표 가설을 평가하는 새로운 메트릭을 도입합니다. 이 방법을 통해 기존 방식이 가진 한계를 극복하고, 더 많은 가능한 목표를 인식할 수 있는 기반을 제공합니다.

- **Technical Details**: DRACO는 환경과의 상호작용을 통해 목표 종속 신경망( Neural Networks, NNs)을 학습하며, 이러한 신경망은 다양한 학습 가능한 도메인에서 관찰된 에이전트가 추구하는 목표의 가능성을 추정합니다. 이는 기존의 비싼 계획자(planner)를 실시간으로 실행할 필요를 없애며, 전이 학습(transfer learning)을 통해 새로운 목표를 표현하는데도 활용될 수 있습니다. 논문에서는 Wasserstein 거리와 통계적 Z-score 메트릭을 기반으로 한 두 가지 거리 측정 방식을 개발하여 학습된 목표 종속 정책과 관측 결과를 비교합니다.

- **Performance Highlights**: DRACO는 다양한 시나리오에서 테스트되었으며, 주어진 입력이 적은 경우에도 기존 알고리즘들보다 더 뛰어난 성능을 보입니다. 특히 연속적 환경에서 과거 방법들을 현저히 초월하는 성과를 달성하였으며, 제시된 테스트베드(test bed)를 통해 목표 인식의 가능성을 보여줍니다. 이 결과는 DRACO의 강력함을 증명하며, 전통적인 목표 인식과 딥 강화 학습(Deep Reinforcement Learning) 간의 교량 역할을 합니다.



### Pan-infection Foundation Framework Enables Multiple Pathogen Prediction (https://arxiv.org/abs/2501.01462)
Comments:
          15 pages, 8 figures

- **What's New**: 이번 논문에서는 호스트 반응 기반 진단방법이 세균 및 바이러스 감염 진단의 정확성을 향상시킬 수 있다는 점을 강조합니다. 저자들은 13개국과 21개 플랫폼에서 수집된 11,247개의 샘플을 포함한 가장 큰 감염 호스트 반응 전사체(transcriptome) 데이터를 구축하였습니다. 이는 기존의 작은 샘플 크기 및 제한된 감염 유형으로는 이루기 어려운 일반화 가능한 진단 모델의 탐색을 가능하게 합니다.

- **Technical Details**: 연구팀은 감염 관련 데이터셋을 바탕으로 포함된 데이터를 사용하여 예측 모델을 구축하였으며, 이 모델의 AUC는 0.97로 나타났습니다. 이후 지식 증류(knowledge distillation) 기법을 활용하여 이 "교사" 모델에서 네 가지 경량화된 병원체 "학생" 모델로 인사이트를 효과적으로 전달합니다. 연구에서 다룬 경량 모델은 황색포도상구균(infection), 연쇄상구균(infection), HIV 감염, RSV 감염 및 패혈증(sepsis)과 관련되어 있습니다.

- **Performance Highlights**: 각 모델은 다음과 같은 AUC 성능을 보였습니다: 황색포도상구균 감염 0.99, 연쇄상구균 감염 0.94, HIV 감염 0.93, RSV 감염 0.94, 패혈증 모델 0.99입니다. 이러한 성능 덕분에 제안된 지식 증류 프레임워크는 다양한 감염 진단을 위한 현장 적응 가능성을 높이며, 경량 디자인을 통해 임상 환경에서 효과적으로 활용될 것으로 기대됩니다.



### GAN-TAT: A Novel Framework Using Protein Interaction Networks in Druggable Gene Identification (https://arxiv.org/abs/2501.01458)
Comments:
          4 pages, 2 figures

- **What's New**: 이번 연구에서는 약물 타겟 지식을 확대하기 위한 새로운 접근 방식을 제안하고, GAN-TAT라는 프레임워크를 통해 고차원 단백질 상호작용 네트워크(Protein Interaction Network, PIN)를 직접적으로 활용하고 있습니다. 특히, ImGAGN 알고리즘을 통해 생성된 잠재 표현(latent representation)을 사용하여 약물로 타겟이 가능한 유전자를 추론하는 데 초점을 맞추었습니다. GAN-TAT는 임상 증거를 기반으로 한 예측 결과로, 약물유전체학(pharmacogenomics)에서의 실제적인 응용 가능성을 강조합니다.

- **Technical Details**: 연구에서 사용된 PIN은 신호 전달 경로 데이터를 기반으로 구축되었으며, 총 6,048개의 유전자 노드와 20,697개의 방향성 비가중 엣지로 구성됩니다. ImGAGN-GraphSAGE 모델을 활용한 네트워크 임베딩 기술을 도입하여 각 유전자 대한 80차원 임베딩을 생성하였고, 이어서 XgBoost 분류기를 사용해 데이터 불균형 문제를 해결하기 위한 서브샘플링 전략을 적용하였습니다. GAN-TAT 구조는 생성적 적대 신경망(Generative Adversarial Network, GAN)을 포함하며, 그래프 생성기, 인코더, 판별기로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, GAN-TAT 구조와 ImGAGN-GraphSAGE + XgBoost 조합이 최고 AUC-ROC 점수 0.951을 달성하며, 다른 임베딩 알고리즘보다 지속적으로 우수한 성능을 보였습니다. 특히, XgBoost는 모든 임베딩 방법에서 가장 효과적인 분류기로 확인되었으며, 임상 진단 관련 유전자(label sets)에서 최고의 효능을 발휘했습니다. 이 연구는 GAN-TAT가 약물 타겟 발굴에서 가진 중요성을 보여주는 다양한 실험적 데이터를 제공합니다.



### Reinforcing Thinking through Reasoning-Enhanced Reward Models (https://arxiv.org/abs/2501.01457)
- **What's New**: 본 연구는 LLM의 복잡한 다단계 추론을 개선하기 위해 Distillation-Reinforcement-Reasoning (DRR)이라는 새로운 3단계 프레임워크를 제안합니다. 이 프레임워크는 LLM의 내부 행동을 외부 피드백으로 활용하여, LLM이 언제 멈춰야 할지 결정할 수 있도록 돕습니다. DRR은 중간 단계의 수작업 레이블이 필요 없으며, 경량 디자인으로 다양한 LLM 중심의 작업에 쉽게 적용할 수 있습니다.

- **Technical Details**: DRR 프레임워크는 LLM의 추론 능력을 반영하는 행동 데이터를 생성을 통해 시작합니다. 이후, 행동 데이터를 기반으로 경량 식별 보상 모델(Discriminative Model, DM)을 훈련하여 추론 시 판단을 돕습니다. 이 과정은 언어적 보상(verbal reward)을 통해 LLM의 평가를 제공하며 모델 파라미터를 변경하지 않고도 동적 피드백 메커니즘을 구축합니다.

- **Performance Highlights**: 실험 결과, DRR 프레임워크는 자가 비평(self-critique) 방식을 적용한 방법들보다 우수한 성능을 보였으며, 추가적인 복잡한 데이터 주석에 의존하지 않는 것으로 나타났습니다. 연구팀은 모든 코드베이스와 체크포인트, 생성된 데이터를 공개할 예정이며, 이는 개방형 및 닫힌 소스 LLM에 유익할 것으로 기대됩니다.



### Geometry Matters: Benchmarking Scientific ML Approaches for Flow Prediction around Complex Geometries (https://arxiv.org/abs/2501.01453)
- **What's New**: 이 연구는 복잡한 형상의 유동 예측을 위한 다양한 SciML 모델의 벤치마킹을 통해 그 격차를 해소하려고 합니다. 특히, 신경 연산자(neural operators)와 비전 변환기(vision transformer) 기반의 기초 모델을 포함하여, 고차원 유동 데이터셋을 활용한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구에서 사용된 기하학적 표현은 Signed Distance Fields (SDF)와 이진 마스크(binary masks)로, 이는 모델의 정확도와 일반화 능력에 큰 영향을 미칩니다. 논문의 핵심은 전역 정확도(global accuracy), 경계층 충실도(boundary layer fidelity), 물리적 일관성(physical consistency)을 통합한 새로운 평가 프레임워크를 도입한 것입니다.

- **Performance Highlights**: 결과적으로 기초 모델들은 데이터가 제한된 시나리오에서 신경 연산자들보다 우수한 성능을 보였습니다. SDF 표현은 충분한 훈련 데이터가 있을 경우 더욱 뛰어난 결과를 보여주었습니다. 그러나 모든 모델이 분포 외의 일반화(out-of-distribution generalization)에는 어려움을 겪고 있어, 이는 향후 SciML 응용의 중요한 도전 과제로 남아 있습니다.



### Detecting and Mitigating Adversarial Attacks on Deep Learning-Based MRI Reconstruction Without Any Retraining (https://arxiv.org/abs/2501.01908)
- **What's New**: 본 연구에서는 магнитной резонансной томографии(MRI) 복원 모델의 적대적 공격(adversarial attack)을 탐지하고 완화하는 새로운 접근 방식을 제안합니다. 기존의 방법과 달리 재학습이 필요 없이 루프 측정 일관성(cyclic measurement consistency)에 기반하여 적대적 교란을 감지합니다. 제안된 방식은 기존의 다수의 방법들보다 실험적으로 보다 효과적인 성능을 보여주며, 다양한 데이터셋과 공격 유형에서도 우수한 결과를 기록했습니다.

- **Technical Details**: MRI는 주파수 영역에서 수집된 원시 측정 데이터를 사용하는데, 이 데이터는 수신 코일(receiver coils)에서 수집됩니다. 본 연구에서는 PD-DL(Physics-driven Deep Learning) 기술을 채택하여 구조화된 손실 함수(objective function)를 사용하여 적대적 공격을 탐지하고 완화합니다. 공격 없는 경우 재구성 결과가 서로 일관된 결과를 보여야 하며, 공격이 있을 경우에는 두 재구성 간의 큰 불일치가 나타나는 것을 기반으로 합니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋, PD-DL 네트워크, 공격 유형 및 강도에서 적대적 혼란의 영향을 크게 줄였습니다. 특히 재학습 없이도 공격 크기나 생성된 공격 알고리즘에 상관없이 효과적으로 적용 가능한 것을 확인했습니다. 이러한 성과는 기존의 기법들보다 정량적 및 정성적으로 높은 성능을 제공하며, 비교적 깨끗한 이미지에서의 성능에도 영향을 미치지 않는 장점을 가집니다.



### EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation (https://arxiv.org/abs/2501.01895)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 EnerVerse라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 로봇 조작 작업을 위한 미래 공간 생성을 포괄적으로 지원합니다. EnerVerse는 내부 청크(space chunk) 모델링을 위한 합성곱(convolutional) 및 양방향(attention) 메커니즘을 통합하여 저수준 일관성과 연속성을 보장합니다.

- **Technical Details**: EnerVerse 프레임워크는 비디오 데이터의 본질적인 중복성을 인식하고, 희소 메모리(context) 문맥 및 청크 단위(unidirectional) 생성 패러다임을 사용하여 무한 길이 시퀀스 생성을 가능하게 합니다. 또한, 자유 앵커 뷰(Free Anchor View, FAV) 공간을 도입하여 작업 관찰 및 분석을 향상시키는 유연한 관점을 제공합니다. 이 공간은 물리적 제약을 제거하고 로봇의 일반화 및 적응성을 개선합니다.

- **Performance Highlights**: 실험을 통해 EnerVerse의 미래 공간 생성 접근법이 정책 예측 능력을 크게 향상시켜 전체적인 성능 개선에 기여함을 보여주었습니다. 특히 긴 거리의 로봇 조작 작업에서 눈에 띄는 성과를 보여주며, 4D Gaussian Splatting과 결합된 데이터 엔진 파이프라인이 데이터 품질과 다양성을 반복적으로 향상시키는 구조적 이점을 제공함에도 기여하고 있습니다.



### QuArch: A Question-Answering Dataset for AI Agents in Computer Architectur (https://arxiv.org/abs/2501.01892)
- **What's New**: QuArch는 컴퓨터 아키텍처에 대한 언어 모델(LMs)의 이해도를 평가하고 향상시키기 위해 설계된 1500개의 인간 검증 질문-답변 쌍 데이터셋입니다. 이 데이터셋은 프로세서 설계, 메모리 시스템 및 성능 최적화와 같은 분야를 포괄하며, 기존 모델이 겪고 있는 성능 격차를 분석합니다. 특히, 최고의 폐쇄형 모델이 84% 정확도를 기록한 반면, 상위 소형 오픈 소스 모델은 72%에 불과하다는 점을 강조하고 있습니다.

- **Technical Details**: QuArch는 프로세서 실행, 메모리 계층 구조 및 병렬성 같은 핵심 개념에 대한 심도 있는 이해 없이 AI 드리븐 솔루션을 개선할 수 없는 현 상황을 해결하기 위해 고안되었습니다. 이 데이터셋은 기초 컴퓨터 아키텍처 원칙 및 최근 주제, 예를 들어, 딥 러닝 가속기 및 양자 컴퓨팅 아키텍처를 포함하고 있습니다. 데이터셋 구성 과정에서는 Archipedia라는 방대한 지식 자료를 근거로 삼아 고품질의 질문-답변 쌍을 생성하며, 각 질문은 전문가의 검증을 거치게 됩니다.

- **Performance Highlights**: 분석 결과, 다양한 언어 모델의 아키텍처 지식이 39%에서 84%까지 드러났으며, 소형 오픈 소스 모델과 대형 폐쇄형 모델 간에는 12%의 지식 격차가 존재하는 것으로 나타났습니다. 지식 검색 능력을 평가한 결과, 가장 높은 정확도를 기록한 모델은 84%에 이르렀으며, 이는 여전히 아키텍처 개념의 이해에 상당한 개선 여지가 있음을 시사합니다. QuArch를 활용한 미세 조정은 소형 모델의 정확도를 5.4%에서 8.3%까지 향상시키는 효과가 있었습니다.



### Signal Recovery Using a Spiked Mixture Mod (https://arxiv.org/abs/2501.01840)
- **What's New**: 이번 연구에서는 많은 랜덤하게 스케일된 노이즈 관측치에서 신호 세트를 추정하는 문제를 해결하기 위해 spiked mixture model (SMM)을 도입하였습니다. 이를 위해 SMM의 모든 매개변수를 복구하기 위한 새로운 expectation-maximization (EM) 알고리즘을 설계하였습니다. 낮은 신호 대 잡음 비율(signal-to-noise ratio, SNR) 환경에서 SMM은 기존의 Gaussian mixture model (GMM)보다 신호 복구 성능에서 우수함을 보여주었습니다. 다양한 데이터 유형에 적용되는 이 알고리즘의 광범위한 적합성을 확인할 수 있었습니다.

- **Technical Details**: SMM은 관측치가 특정 신호를 랜덤하게 스케일한 형태로 이루어지며, 이 과정에서 추가적인 노이즈가 더해지는 구조입니다. 연구는 독립적인 관측치들을 사용하여 이들의 신호와 노이즈를 모델링하는 방식을 따릅니다. EM 알고리즘은 관측치들이 어떤 혼합 성분과 연관될 확률을 부여하며, 이를 통해 노이즈 variances와 spike responsibilities를 추정합니다. 구체적으로, SMM 모델에서는 관측치의 분산과 신호의 사전 확률이 중요한 역할을 하게 됩니다.

- **Performance Highlights**: 두 가지 사례 연구를 통해 SMM의 성능을 검증하였습니다. 첫 번째는 생물 의학 분야의 응용으로, 랫의 뇌 조직 샘플에서 분자의 패턴을 탐색하는 이미지 질량 스펙트로메트리 데이터셋을 이용하였습니다. 두 번째는 컴퓨터 비전 분야의 응용으로, SMM을 사용하여 하이퍼스펙트럼 이미징 데이터셋을 기저 패턴으로 분할하는 작업을 수행하였습니다. 두 경우 모두 SMM은 GMM이나 k-means clustering과 같은 전통적인 방법으로는 복구되지 않은 성능 향상을 확인할 수 있었습니다.



### Time Series Language Model for Descriptive Caption Generation (https://arxiv.org/abs/2501.01832)
- **What's New**: 본 논문에서는 시계열 데이터의 캡셔닝을 위한 새로운 다중 모달 모델인 TSLM(Time Series Language Model)을 소개합니다. TSLM은 시계열 및 텍스트 데이터를 통합하여 시계열 패턴을 정확하게 설명하는 자연어 문장을 생성하는 능력을 가지고 있습니다. 또한, 기존의 대형 언어 모델(LLM)을 사용하는 데이터 생성 및 노이즈 제거 방법을 통해 교육 데이터 세트를 효과적으로 보강하는 접근 방식을 제공합니다.

- **Technical Details**: TSLM은 인코더-디코더 모델로, 텍스트 프롬프트와 시계열 데이터 표현을 함께 활용하여 미세한 시간적 패턴을 캡처하고 텍스트 설명을 생성합니다. 데이터 부족 문제를 해결하기 위해, TSLM은 고품질 샘플로부터 소수의 학습(few-shot learning)과 오픈 소스 LLM을 이용한 컨텍스트 프롬프트 기법을 적용합니다. 또한, 생성된 데이터를 크로스 모달 밀집 검색 스코어링을 통해 노이즈를 제거하는 새로운 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, TSLM은 다양한 시계열 캡셔닝 데이터 세트에서 기존 최첨단 기법들을 유의미하게 능가하는 성과를 보였습니다. 다중 모달 인코더의 추가적 조정을 통해 시계열 정보와 텍스트 표현 간의 일치를 향상시켜, 보다 정밀하고 유용한 해석 결과를 도출합니다. 이러한 성과는 TSLM이 시계열 데이터에 대한 자동 캡셔닝에서의 큰 가능성을 입증합니다.



### Age-Based Device Selection and Transmit Power Optimization in Over-the-Air Federated Learning (https://arxiv.org/abs/2501.01828)
- **What's New**: 최근, 오버-더-에어(Federated Learning, FL)에 대한 관심이 증가하고 있습니다. FL은 통신 효율성을 높일 수 있는 가능성을 가지고 있습니다. 그러나 장치 선택 전략과 신호 집계 오류로 인해 성능이 제한될 수 있습니다. 특히, 느린 장치(straggler) 무시 시 모델 업데이트의 공정성이 저하되고, 글로벌 모델의 편향이 강화되는 문제가 발생할 수 있습니다.

- **Technical Details**: 이 논문에서 저자들은 느린 장치의 적절한 참여를 보장하고, 훈련 성능을 효율적으로 유지하며, 적시의 업데이트를 보장하는 장치 선택 및 전송 전력 최적화 프레임워크를 제안합니다. 그들은 신호 집계 오류와 선택된 장치의 수가 FL 성능에 미치는 영향을 명확히 하게 됩니다. 이를 위해 Lyapunov 최적화를 통해 각 통신 라운드의 장치 우선 순위를 계산하고, 그리디 알고리즘을 통해 가장 높은 우선 순위의 장치를 선택합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 MSE(mean squared error)를 줄이고 모델 성능을 향상시킵니다. 또한, 공정성과 훈련 효율성을 균형 있게 유지하면서 적절한 적시성을 보장하여 안정적인 모델 성능을 확보합니다. 최종적으로 FedAirAoI 방법은 CIFAR-10 및 CIFAR-100 데이터셋에서 우수한 성능을 발휘합니다.



### Unified Native Spaces in Kernel Methods (https://arxiv.org/abs/2501.01825)
- **What's New**: 본 논문은 기존에 알려진 여러 개의 커널을 단일 파라메트릭 클래스에 통합하여 특별한 경우로서 다룬다. 이 새로운 커널 클래스는 매트른 (Matérn) 커널 및 웬들랜드 (Wendland) 커널과 같은 잘 알려진 커널들을 포함하고 있으며, 이들이 보유하는 주요 속성을 설명한다. 저자는 또한 이 커널이 부정적인 값을 가질 수 있는 경우를 제시하며, 이는 자연과학 및 공학 등 여러 분야에서 중요한 의미를 가진다.

- **Technical Details**: 논문에서는 새로운 커널 클래스의 소벨 (Sobolev) 공간과 관련된 일반적인 성질을 설명한다. 이 커널 클래스는 매개변수화를 통한 직접적인 방법 또는 매개변수의 비대칭적 방법을 통해 얻을 수 있는 다양한 커널들을 포함한다. 또한 새로운 커널에 연관된 RKHS(Redeeming Kernel Hilbert Spaces)를 통해 특정 소벨 공간과의 노름 동등성을 보여준다.

- **Performance Highlights**: 제안된 커널 클래스는 부정 지원 또는 글로벌 지원을 포함하는 이질적인 커널을 포괄하며, 이는 연구자들이 모델링에서 부드러움, 집중된 지원 그리고 홀 효과를 제어할 수 있게 해준다. 저자는 새로운 커널 클래스의 다양한 응용과 관련된 데이터 예시를 통해 이 커널들이 어떻게 적용될 수 있는지 설명하고 있다. 또한 이 연구가 통계학 및 기계학습 분야에서의 현재 및 미래 문헌에 미치는 여러 영향을 논의한다.



### Rerouting LLM Routers (https://arxiv.org/abs/2501.01818)
- **What's New**: 본 논문에서는 LLM 라우터(LLM routers)의 적대적 강인성(adversarial robustness)을 조사합니다. LLM 라우터는 쿼리(query)의 복잡성에 따라 비용이 저렴하거나 비싼 LLM으로 라우팅 함으로써 품질과 비용의 균형을 맞추는 시스템입니다. 연구의 초점은 이런 라우팅 시스템의 안전성 문제를 명확히 정의하고 적대적 입력에 대한 강인성 문제를 제기하는 것입니다.

- **Technical Details**: 본 연구에서 제안하는 'confounder gadgets'라는 새로운 개념은 입력 쿼리에 추가되어 원하지 않는 LLM으로 라우팅을 유도하는 토큰 시퀀스(token sequences)입니다. 이러한 가젯을 사용한 공격 방식은 두 가지 시나리오, 즉 화이트 박스(white-box) 및 블랙 박스(black-box) 환경에서 검증되었습니다. 실험 결과는 여러 개방형 및 상용 LLM 라우터에 대해 이 공격 방법이 유효하다는 것을 보여주었습니다.

- **Performance Highlights**: 본 논문에서는 LLM 라우터에 대한 방어 방법도 논의하고 있습니다. 전통적인 perplexity 기반 필터링은 새로운 적대적 가젯에 대해 효과적이지 않음을 발견하였고, 대신 다른 방어 전략인 사용자 쿼리의 비정상적 빈도를 식별하는 방법을 제안합니다. 이러한 라우팅 공격은 적대자의 다양한 목표에 부합하며, LLM 제어 시스템에 대한 향후 연구를 촉진할 수 있기를 기대합니다.



### QuantumBind-RBFE: Accurate Relative Binding Free Energy Calculations Using Neural Network Potentials (https://arxiv.org/abs/2501.01811)
- **What's New**: 이번 연구에서는 단백질-리간드(binding affinity) 결합의 정확한 예측이 약물 발견에서 얼마나 중요한지를 강조합니다. 기존의 리간드 포스 필드(ligand force fields)의 한계를 극복하기 위해 새로운 신경망 포텐셜(neural network potentials, NNP) 모델인 AceForce 1.0을 소개합니다. 이 모델은 다양한 약물 유사 화합물에 적용 가능하며, 전체적으로 향상된 정확도를 보여줍니다.

- **Technical Details**: 연구에서는 NNP/MM 접근 방식을 사용하여 리간드 상호작용을 모델링합니다. 이 하이브리드 방법은 리간드를 신경망 포텐셜로 더 정확하게 시뮬레이션하고, 주변 단백질 환경은 전통적인 분자 역학(molecular mechanics)으로 처리하여 속도를 높입니다. AceForce 1.0은 TensorNet 아키텍처에 기반하여 훈련된 첫 번째 모델로, 다양한 원자 요소와 하전 분자를 지원합니다.

- **Performance Highlights**: 제공된 벤치마크 연구를 통해 AceForce에서 리간드 결합 자유 에너지(relative binding free energy, RBFE) 계산의 정확도를 GAFF2 및 ANI2-x 모델과 비교하여 개선된 결과를 얻었습니다. AceForce는 이전 NNP 모델보다 두 배 큰 타임스텝(2 fs)에서 시뮬레이션을 실행할 수 있어 컴퓨팅 속도에 유리합니다. 결론적으로, AceForce는 약물 발견을 위한 무료 에너지 계산의 향후 발전 가능성을 보여줍니다.



### John Ellipsoids via Lazy Updates (https://arxiv.org/abs/2501.01801)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 n개의 점을 포함한 John ellipsoid을 계산하기 위한 보다 빠른 알고리즘을 제안하고 있습니다. 기존의 알고리즘보다 Leverage score를 지속적으로 계산하는 방식에서 고정점 반복법을 사용하는 새로운 접근 방식을 통해 속도가 크게 향상될 수 있음을 보여줍니다. 또한, John ellipsoid을 위해 저공간 스트리밍 알고리즘(low-space streaming algorithms)도 제시하여 더욱 효율적인 방법을 제공합니다.

- **Technical Details**: John ellipsoid 문제는 n개의 점에 대한 최소 부피를 포함하는 타원체(MVEE)를 찾는 클래식한 문제입니다. 본 연구에서는 대칭 케이스에 초점을 맞추고 있으며, 입력점들을 제약조건으로 하는 다각형(polytope)을 고려하여, 타원체 Q가 주어진 다각형 P를 포함하도록 하는 것이 목표입니다. 알고리즘은 고정점 조건을 이용한 반복적 방법으로, Leverage scores의 정확한 계산을 지연시키면서 효율적으로 처리합니다.

- **Performance Highlights**: 제안된 알고리즘은 n개의 점이 주어질 때 빠른 rectangular matrix multiplication 기술을 활용하여 성능을 개선합니다. 최신 알고리즘의 반복 횟수를 O(ε⁻¹ log(n/d))로 줄이며, 이는 실질적으로 데이터의 크기가 커질수록 더욱 유리하게 작용합니다. 따라서 John ellipsoid을 계산하는 데에서 기존 알고리즘에 비해 계산 시간과 자원 소모가 획기적으로 감소할 것으로 기대됩니다.



### Proposing Hierarchical Goal-Conditioned Policy Planning in Multi-Goal Reinforcement Learning (https://arxiv.org/abs/2501.01727)
Comments:
          10 pages, 4 figures, this is a preprint of the peer-reviewed version published by SCITEPRESS for ICAART-2025

- **What's New**: 이 논문은 인간형 로봇이 희소 보상(sparse rewards) 문제를 해결하기 위해 강화 학습(reinforcement learning, RL)과 자동 계획(automated planning)을 결합한 새로운 방법론을 제안합니다. 이를 통해 목표 조건 정책(goal-conditioned policies, GCPs)을 계층적으로 구성하고, 고수준 행동(high-level actions, HLA)을 사용하는 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS) 계획을 적용합니다. 이 프레임워크는 다양한 복잡한 작업을 수행하기 위한 탐색과 계획의 효율성을 개선할 가능성을 보여줍니다.

- **Technical Details**: 본 연구의 기법은 계층적 강화 학습(hierarchical RL)과 GCPs를 활용하여 에이전트가 여러 작업을 학습할 수 있도록 설계되었습니다. MCTS는 저수준의 원시 행동(primitive actions) 대신 고수준 행동을 계획하는 데 사용됩니다. 또한, 단일 계획 트리는 에이전트의 목표 달성을 위한 지식을 보유하고 있으며, HLAs를 재사용함으로써 샘플 효율성을 높이고 미래 행동을 예측할 수 있게 합니다.

- **Performance Highlights**: 제안된 계층적 목표 조건 정책 계획(HGCPP) 프레임워크는 복잡한 문제 해결을 위한 더 나은 탐색 및 계획 방법을 제공할 잠재력이 있습니다. 일반적으로, 에이전트는 MCTS를 통해 계획된 HLAs를 이용하여 빠른 추론을 수행하고 여러 목표에 도달할 수 있습니다. 이 연구는 초기 단계의 연구로 평가가 이루어지지 않았지만, 기존 방법들과의 차별화된 접근을 통해 향후 발전 가능성을 제시합니다.



### Beyond Non-Degeneracy: Revisiting Certainty Equivalent Heuristic for Online Linear Programming (https://arxiv.org/abs/2501.01716)
- **What's New**: 이번 연구에서는 Certainty Equivalent (CE) 휴리스틱 알고리즘의 성능을 더 정교하게 분석하였습니다. 이전에는 이 알고리즘이 특정한 fluid regularity 조건을 만족해야 성능 보장이 가능하다는 일반적인 신념이 있었으나, 우리는 더 완화된 가정을 통해 CE가 넓은 문제 사례에서 효과적임을 입증하였습니다.

- **Technical Details**: AGeneral framework of online linear programming을 활용하여, CE가 경량의 조건 하에서도 근사 최적 성능을 달성한다는 사실을 보여주었습니다. 연구 결과에 따르면, CE는 연속적인 조건부 보상 분포를 가진 문제들에서 비생성적 현상의 저주를 이길 수 있음을 시사합니다. 우리는 $(	ext{log} T)^2$와 최악의 경우인 $	ext{sqrt}(T)$의 상황을 매개변수 
β에 따라 interpolating하여 성능을 제시합니다.

- **Performance Highlights**: 새로운 알고리즘 분석 기법을 개발하여, 무작위 선형 프로그래밍 문제의 해에 대한 집중 분석을 수립하였습니다. 이러한 기법은 더 완화된 가정 하에서도 향상된 후회(regret) 분석을 가능하게 하며, 광범위한 온라인 의사결정 맥락에서도 응용될 가능성을 가진다는 점에서 의미가 큽니다.



### Enhancing Large Vision Model in Street Scene Semantic Understanding through Leveraging Posterior Optimization Trajectory (https://arxiv.org/abs/2501.01710)
Comments:
          7 pages

- **What's New**: 이 논문에서는 자율주행 (AD) 모델의 일반화를 향상시키기 위해 사전 훈련된 Large Vision Models (LVMs)를 이용한 새로운 접근 방식을 제안합니다. AD 모델이 수집된 데이터를 기반으로 시간이 지남에 따라 모델을 업데이트할 수 있도록 하여, 데이터 증가에 따른 언더피팅 문제를 해결합니다. 이 방법은 LVM의 강력한 적합 능력 덕분에 특히 효과적이며, 다양한 훈련 데이터를 통해 모델의 일반화 능력을 크게 향상시킵니다.

- **Technical Details**: LVM을 백본으로 사용하여, 차량의 온보드 데이터셋에 기반한 다운스트림 인식 헤드를 훈련하는 방식을 제안합니다. 이는 LVM이 이미지의 복잡한 패턴을 추출할 수 있도록 도와줍니다. 또한, Posterior Optimization Trajectory (POT)-Guided 최적화 스킴을 도입하여 인식 헤드의 훈련 속도를 가속화합니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 모델 대비 성능을 66.48% 향상시키고, 수렴 속도는 6배 이상 빨라짐을 입증하는 광범위한 실험 결과를 보여줍니다. 이러한 성능 향상은 AD 차량에 필수적인 계산 효율성을 제공하며, 신속한 의사 결정을 위한 핵심 요소가 됩니다.



### Guaranteed Nonconvex Low-Rank Tensor Estimation via Scaled Gradient Descen (https://arxiv.org/abs/2501.01696)
- **What's New**: 이번 논문에서는 다차원 데이터를 효과적으로 표현하기 위한 텐서(tensor)의 중요성을 강조하며, 특히 손상된 텐서 데이터에서 유의미한 정보를 추출하는 데 중점을 둡니다. Scaled Gradient Descent (ScaledGD) 알고리즘을 개발하였으며, 이는 텐서-텐서 곱(t-product)과 텐서 특이값 분해(t-SVD) 프레임워크를 기반으로 하여 최적의 스펙트럼 초기화를 통해 텐서 인자를 직접 추정합니다.

- **Technical Details**: ScaledGD 알고리즘은 두 가지 주요 문제인 텐서 강건 주성분 분석(tensor robust principal component analysis)과 텐서 완성(tensor completion)에 대해, 저위(rank) 텐서의 조건 숫자(condition number)와 무관하게 일정한 비율로 선형 수렴(linear convergence) 속도를 보입니다. 손상 정도가 너무 크지 않고 샘플 크기가 충분할 경우에도 극히 낮은 반복 비용으로 그래디언트 하강법을 유지합니다.

- **Performance Highlights**: ScaledGD는 저위(rank) 텐서 추정을 위한 t-SVD 분해에서 이러한 특성을 명확히 증명한 최초의 알고리즘으로, 수치 예제를 통해 저조도 저위(rank) 텐서 추정에서의 수렴 속도 향상을 실제로 시연합니다. 이러한 성능은 특히 ill-conditioned 환경에서 더욱 두드러지며, 알고리즘의 적용 가능성을 더욱 넓히는 데 기여합니다.



### Multimodal Contrastive Representation Learning in Augmented Biomedical Knowledge Graphs (https://arxiv.org/abs/2501.01644)
- **What's New**: 이번 연구는 다중 모드 접근 방식을 도입하여 생물 의학 지식 그래프(BKG)에서 링크 예측(link prediction)의 성능을 향상시키기 위해 특수 언어 모델(Language Models)로부터의 임베딩(embedding)과 그래프 대조 학습(Graph Contrastive Learning)을 통합한 새로운 방법론을 제시합니다. 또한, 생물학적 서열과 텍스트 정보를 포함하는 PrimeKG++라는 향상된 지식 그래프를 제안하여 노드 간 관계를 풍부하게 표현하고 있습니다. 이 접근 방식은 보편성을 극대화하여 보지 못한 노드에 대해서도 정확한 링크 예측을 가능하게 합니다.

- **Technical Details**: 본 연구에서는 언어 모델(LM)에서 유도된 임베딩을 통합하고, GCL을 활용하여 각 노드의 상호 정보를 최적화하여 노드 간 관계를 처리합니다. 이와 함께, 지식 그래프 임베딩(Knowledge Graph Embedding) 모델을 사용하여 다양한 생물학적 개체 간의 상호 정보를 캡처합니다. 연구의 핵심은 다양한 입력 데이터의 세미틱과 관계 정보를 통합하여, 생물 의학 지식 그래프(BKG)의 링크 예측 성능을 강화하는 것입니다.

- **Performance Highlights**: PrimeKG++와 DrugBank 약물-타겟 상호작용 데이터셋에서의 실험 결과, 제안된 방법이 다양한 생물 의학 데이터셋에서 강력하고 일관되며 정확한 링크 예측을 보여주었습니다. 기존 모델과 비교했을 때, 우리의 사전 훈련된 노드 표현 모델이 성능 향상에 크게 기여했으며, link prediction뿐만 아니라 생물 의학 연구 커뮤니티에 가치 있는 자원을 제공하고 있습니다.



### iCBIR-Sli: Interpretable Content-Based Image Retrieval with 2D Slice Embeddings (https://arxiv.org/abs/2501.01642)
Comments:
          8 pages, 2 figures. Accepted at the SPIE Medical Imaging

- **What's New**: 본 연구에서는 뇌 MR 이미지를 위한 새로운 해석 가능한 콘텐츠 기반 이미지 검색 시스템인 iCBIR-Sli를 제안합니다. 이 시스템은 2D 슬라이스의 연속성을 활용하여 뇌의 구조적 정보를 전체적으로 보존하면서도 우수한 검색 성능을 보여줍니다. iCBIR-Sli는 낮은 차원의 표현을 효율적으로 집계하여 CBIR 시스템의 필수적인 특성을 충족시키고 있습니다.

- **Technical Details**: iCBIR-Sli는 2D 슬라이스 임베딩 기법을 사용해 뇌 MR 이미지의 정보를 집계하고, 이를 통해 높은 완전성, 사용성, 견고성 및 해석성을 갖춘 낮은 차원 표현을 생성합니다. 이 과정에서 5개의 공개된 뇌 MR 데이터셋을 사용하여 알츠하이머병 및 인지적으로 정상인에 대한 검색 평가 실험을 수행합니다.

- **Performance Highlights**: iCBIR-Sli는 매크로 F1 0.859의 top-1 검색 성능을 보여, 분류를 위해 명시적으로 설계된 기존의 심층 학습 모델과 동등한 성능을 발휘합니다. 또한, 이 방법은 검색된 질병에 대한 뇌 영역을 명확히 식별함으로써 높은 해석성을 제공합니다.



### A non-ergodic framework for understanding emergent capabilities in Large Language Models (https://arxiv.org/abs/2501.01638)
- **What's New**: 이번 연구에서는 대형 언어 모델이 비에르고딕(non-ergodic) 시스템이라는 점을 증명하고, 능력 출현을 설명하기 위한 수학적 프레임워크를 제시합니다. 특히, Stuart Kauffman의 인접 가능성 이론(TAP)을 기반으로 한 접근 방식을 통해 능력의 출현 메커니즘을 제시합니다. 이는 기존 언어 모델 연구에서 진일보한 이해를 제공합니다.

- **Technical Details**: 연구에서는 TAP 방정식이 아키텍처, 훈련, 그리고 맥락의 제약 사항들과 어떻게 상호작용하여 모델의 능력을 형성하는지를 설명합니다. 특히, 의미 공간의 단계 전이(phase transitions)를 통해 자원 제약이 모델의 능력에 미치는 영향을 강조합니다. 실험 결과, 세 가지 다른 언어 모델에서 제약 상호작용과 경로 의존 탐색(path-dependent exploration)에 의해 능력이 불연속적으로 나타나는 것을 보여줍니다.

- **Performance Highlights**: 이 연구는 언어 모델의 출현(emergence)을 이해하기 위한 이론적 기초를 제공합니다. 또한, 향후 능력 출현을 이끌어낼 수 있는 아키텍처 개발에 대한 가이드를 제시하여, 향후 연구 및 응용에 기여할 가능성을 제시합니다.



### PSYCHE: A Multi-faceted Patient Simulation Framework for Evaluation of Psychiatric Assessment Conversational Agents (https://arxiv.org/abs/2501.01594)
Comments:
          The first two authors contributed equally

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 인간과 유사한 반응을 생성할 수 있는 대화형 에이전트의 개발을 가속화했습니다. 이 연구에서는 정신과 평가 대화형 에이전트(PACAs)를 위한 새로운 프레임워크인 PSYCHE를 제안하여 임상 평가에서 정신과 의사의 역할을 시뮬레이션하는 방법을 다룹니다. PSYCHE는 PACAs의 임상 적합성을 1) 임상적 관련성, 2) 윤리적 안전성, 3) 비용 효율성 및 4) 정량적 평가를 통해 평가할 수 있는 구조를 제공합니다.

- **Technical Details**: PSYCHE 프레임워크는 다면적 정신과 구조(MFC)를 기반으로 환자 발화를 시뮬레이션하고 PACAs를 평가하는 방식으로 설계되었습니다. 이 프레임워크는 네 가지 단계로 구성되며, 사용자 입력, MFC 생성, 발화 시뮬레이션 및 평가 세션으로 이어집니다. 각 단계는 PACA의 성능을 평가하기 위한 체계적 프로세스를 통해 PSYCHE-SP를 생성하여 환자와의 상호작용을 가능하게 하며, 이 결과는 PACA의 성과 지표인 PSYCHE SCORE로 나타납니다.

- **Performance Highlights**: 연구 결과, 10명의 정신과 의사가 평가한 PSYCHE-SP는 다양한 정신 장애를 시뮬레이션하는 데 있어서 높은 일관성을 보였습니다. 총 7가지 장애에 대해 평균 93%의 적합성을 달성하였으며, 주요 우울 장애(MDD)와 사회 불안 장애(SAD)에서 각각 97%의 가장 높은 적합성을 기록했습니다. PSYCHE는 임상 환경에서 PACAs의 성능 평가를 효과적으로 진행할 수 있는 가능성을 보여주며, 향후 정신 건강 분야의 자동화와 효율성을 높이는 데 기여할 것으로 기대됩니다.



### BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems (https://arxiv.org/abs/2501.01593)
Comments:
          12. arXiv admin note: substantial text overlap with arXiv:2409.07775

- **What's New**: 이 논문에서는 협력적 다중 에이전트 심층 강화 학습(c-MADRL)에 대한 새로운 유형의 백도어 공격인 BLAST(Backdoor Leverage Attack)를 제안합니다. 이 공격은 단일 에이전트에만 백도어를 심어 전체 팀에 영향을 미치는 방식으로, 이를 통해 공격의 은폐성과 효과성을 극대화합니다. 특히, BLAST는 특정 시각 패턴 대신 적대적인 시공간 행동 패턴을 백도어 트리거로 사용하여 은폐성을 향상시킵니다.

- **Technical Details**: BLAST는 단일 에이전트의 보상 함수 해킹을 통해 전체 다중 에이전트 시스템에 대한 leaverage attack 효과를 구현합니다. 이 방법에 의해, 공격자는 BLAST 에이전트의 행동이 다른 에이전트에 미치는 영향을 극대화하며 불필요한 반대 영향을 제거합니다. 이 공격 방식은 분산형 학습 구조인 중앙집중식 훈련과 분산 실행(CTDE) 프레임워크에 적용되며, 이 구조 때문에 공격자가 보이지 않는 상태에서 트리거를 숨길 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 BLAST 공격은 VDN, QMIX 및 MAPPO와 같은 3개의 전통적인 c-MADRL 알고리즘에 대해 높은 공격 성공률을 보이며, 청정 성능 변동률은 낮은 수준을 유지합니다. BLAST의 유효성은 두 가지 인기있는 c-MADRL 환경(SMAC 및 Pursuit)과 기존의 두 방어 메커니즘을 테스트하여 확인되었습니다. 이러한 성과는 BLAST가 기존의 백도어 방어 전략에 대한 저항성을 갖고 있음을 보여줍니다.



### Unsupervised learning for anticipating critical transitions (https://arxiv.org/abs/2501.01579)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문에서는 복잡한 동적 시스템에서의 중대한 전환(critical transitions) 예측을 위해 변형 오토인코더(variational autoencoder, VAE)와 리저버 컴퓨팅(reservoir computing)을 결합한 새로운 접근 방식을 제안합니다. 이 프레임워크는 시계열(time series) 데이터를 기반으로 하여 매개변수를 학습할 수 있는 비지도 학습(unsupervised learning) 방식으로 작동합니다. 필수적인 매개변수 정보가 없더라도 이 방법이 중대한 전환을 예측할 수 있음을 보여줍니다.

- **Technical Details**: 변형 오토인코더(VAE)는 복잡한 데이터 세트의 구조를 효율적으로 캡처하기 위해 변형 추정(variational inference)과 심층 신경망(deep neural networks)의 원리를 결합하여 만들어진 생성 모델입니다. 이 모델의 핵심은 인코더-디코더(encoder-decoder) 구조로, 인코더가 입력 데이터에서 물리적 매개변수를 추출하고 디코더가 이 매개변수를 기반으로 예측 모델로 작용하여 초기 조건을 시간에 따라 전파합니다. 이러한 접근 방식은 다양한 물리적 현상을 연구하는 데 유용하게 사용됩니다.

- **Performance Highlights**: 이 프레임워크는 시간 시퀀스 데이터로부터 주요 매개변수 정보를 추출하는 데 성공하여, 리저버 컴퓨터(reservoir computer)를 사용해 중대한 전환을 예측하는 능력을 보여줍니다. 스페이티오템포랄 큐라모토-시바신스키 시스템과 같은 프로토타입 동적 시스템에 대해 비지도 학습 스킴의 효용성을 입증하였습니다. 또한 여러 독립 매개변수나 부분 상태 관측(partial state observations)이 있는 시나리오에도 확장이 가능합니다.



### Transfer Learning Analysis of Variational Quantum Circuits (https://arxiv.org/abs/2501.01507)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 논문은 변분 양자 회로(Variational Quantum Circuit, VQC)의 전이 학습(transfer learning)을 분석합니다. 우리가 제안하는 프레임워크는 하나의 도메인에서 사전 훈련된 VQC로 시작하여, 새로운 도메인에서 필요한 1-매개변수 유니타리(1-parameter unitary subgroup) 그룹의 전이를 계산합니다. 기존 연구와의 차별점은 우리가 사전 훈련된 VQC에서 시작하여 기존 데이터를 기억하면서 새로운 데이터를 다룰 수 있도록 최적의 양자 매개변수를 활용한다는 점입니다.

- **Technical Details**: 논문에서는 변분 양자 알고리즘(Variational Quantum Algorithms, VQAs)을 통해 양자 신경망(Quantum Neural Networks, QNNs) 아키텍처를 다양한 머신러닝(Machine Learning, ML) 작업에 적용할 수 있도록 하는 방법을 설명합니다. VQC는 외부 신호에서 학습할 수 있는 조정 가능한 매개변수를 가진 양자 게이트로 구성되어 있으며, 고전적 데이터와 함께 작동합니다. 본 연구는 비슷한 데이터 세트 간의 VQC 모델 매개변수를 조정하는 분석적 솔루션을 도출합니다.

- **Performance Highlights**: 연구 결과는 변분 양자 회로의 전이 학습 메커니즘을 밝히고, 이에 대한 문서화된 물리적 의미를 제공합니다. 특히, 본 방법론은 기존의 경량 기계 학습 기법과 비교할 때 매우 효율적임을 보여줍니다. 전이 학습을 통해 VQC가 새로운 데이터 도메인에 적응하는 능력을 나아가면, 이는 양자 컴퓨팅과 고전적 머신러닝의 융합을 통해 더 나은 성능을 가능하게 할 것입니다.



### ORACLE: A Real-Time, Hierarchical, Deep-Learning Photometric Classifier for the LSS (https://arxiv.org/abs/2501.01496)
Comments:
          29 pages, 19 figures, 9 tables. Submitted to ApJ

- **What's New**: 이 논문에서는 ORACLE이라는 최초의 계층적 깊이 학습 모델을 제시합니다. ORACLE은 실시간 및 상황 인식적인 분류를 수행하며, 광학 관측 정보를 바탕으로 고신뢰 기분류(classification)를 제공합니다. 이 모델은 Gated Recurrent Units (GRUs)를 사용하는 순환 신경망(recurrent neural network)으로 구축되었으며, 고유한 계층적 교차 엔트로피 손실 함수(custom hierarchical cross-entropy loss function)를 통해 훈련되었습니다.

- **Technical Details**: ORACLE은 약 50만 개의 이벤트로부터 훈련되었으며, 기본적으로 1일간의 광도 관측 정보와 상황적 정보를 통해 고성능의 분류 결과를 도출합니다. 이 모델은 64일의 데이터를 확보했을 경우 정확도가 99%를 초과하고, 1024일 후에는 19종 분류에서 83%의 성능을 기록합니다. 또한, ORACLE은 다른 최첨단 분류기들과 비교했을 때 비슷한 성능을 보이며 빠른 분류 결과를 제공합니다.

- **Performance Highlights**: ORACLE은 1일간의 관측 정보로도 뛰어난 분류 성능을 보이고, 64일 시점에서 99%가 넘는 정확도를 달성하는 성과를 보여줍니다. 이 논문에서는 ORACLE의 성능을 더 깊이 탐구하고, 기존의 분류기들과의 비교를 통해 그 효과성을 입증하였습니다. 연구진은 ORACLE의 코드와 모델 가중치가 GitHub에 공개되어 있어 커뮤니티에서 활용할 수 있음을 강조합니다.



### Sequencing Silicates in the IRS Debris Disk Catalog I: Methodology for Unsupervised Clustering (https://arxiv.org/abs/2501.01484)
Comments:
          23 pages, 16 figures, Accepted to ApJS, $\texttt{CLUES}$ software available on GitHub

- **What's New**: 이번 연구에서는 $	exttt{CLUES}$ (CLustering UnsupErvised with Sequencer)라는 새로운 비모수적(non-parametric) 기계 학습 도구를 소개합니다. 이 도구는 잔해 원반(debris disks)의 스펙트럼 데이터를 분석하고 분류하기 위해 설계되었습니다. 특히, 수천 개의 잔해 원반에 대한 체계적인 연구가 부족한 상황에서, $	exttt{CLUES}$는 비지도 클러스터링(unsupervised clustering) 기법을 활용합니다.

- **Technical Details**: $	exttt{CLUES}$는 다양한 비지도 클러스터링 방법과 다중 척도 거리 측정(multi-scale distance measures)을 결합하여 새로운 그룹화(groupings)와 경향(trends)을 식별합니다. 이를 통해 잔해 원반 내의 조성 다양성(compositional diversity)과 지구물리학적 과정(geophysical processes)을 이해하는 데 기여합니다. 이 연구는 잔해 원반의 광물학(mineralogy) 파라미터 공간(parameter space)을 광범위하게 탐색할 수 있게 해줍니다.

- **Performance Highlights**: 연구 결과, $	exttt{CLUES}$는 초기 단계에서 잔해 원반의 조성과 인구 통계학적 특성을 분석하는 데 효과적인 접근 방식을 제공합니다. 이는 향후 잔해 원반에 대한 자세한 후속 연구의 기초를 마련합니다. 이번 연구는 원시 행성계(protoplanetary disks) 및 태양계 물체(solar system objects)와 같은 다른 분야에도 응용 가능성을 제시합니다.



### An unsupervised method for MRI recovery: Deep image prior with structured sparsity (https://arxiv.org/abs/2501.01482)
- **What's New**: 본 연구에서는 전체 샘플링된 k-space 데이터가 필요 없는 비지도 MRI 복원 방법인 구조적 희소성을 적용한 Deep Image Prior (DIP) 확장 모델인 DISCUS를 제안하고 검증합니다. DISCUS는 그룹 희소성을 도입하여 프레임 특정 코드 벡터를 개선하고, 이를 통해 시간 변화를 포착할 수 있는 저차원 매니폴드를 발견할 수 있게 합니다. 또한, DISCUS는 기존의 방법들과 달리 매니폴드의 차원 수를 미리 정할 필요 없이 동적 코드 벡터에 그룹 희소성을 부과하여 차원 수를 찾습니다.

- **Technical Details**: MRI에서 복원 과정은 노이즈가 있는 비샘플링 k-space 데이터에서 기본 이미지를 추정하는 것을 포함합니다. DISCUS는 무작위 코드 벡터를 네트워크에 공급하여 영상 시퀀스를 생성하며, 전통적인 방법들과 비교하여 이미지 유사성을 더 많이 반영하지 않고도 시간적으로 근접한 데이터 간의 유사성을 가정하지 않습니다. 이 방법은 특히 단일 샷의 자유 호흡 LGE 촬영 및 매개변수 매핑에 적합합니다.

- **Performance Highlights**: DISCUS는 여러 연구를 통해 기존 방법들보다 뛰어난 성능을 보여주었습니다. 시뮬레이션 및 실제 데이터를 기반으로 한 평가에서 NMSE와 SSIM 기준의 복원 품질 향상을 입증하였으며, 전문가 평가에서 높은 점수를 기록했습니다. 이러한 결과는 특히 정상적인 스캔 조건에서의 환자 불편을 줄이고, MRI의 임상 적용 가능성을 높이는 데 기여할 것입니다.



### Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search (https://arxiv.org/abs/2501.01478)
Comments:
          5 pages, 1 figure, 2 tables accepted by aaai 2025 NeurMAD workshop

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 능력을 개선하기 위해 Monte Carlo Tree Search (MCTS)를 이용하여 자체적으로 과정 감독(process supervision) 데이터를 생성하는 방법을 제안합니다. 발생한 reasoning step에 각각 '상대적 정합성(relative correctness)' 점수를 부여하고, 이를 통해 LLM을 훈련시키는 과정을 반복적으로 수행했습니다. 이 방법은 결과적으로 두 개의 수학 reasoning 데이터셋에서 LLM의 성능을 상당히 개선하는 것으로 나타났으며, 이는 향상된 추론 능력의 전이성(transferability)도 보여줍니다.

- **Technical Details**: 제안된 방법론에서는 LLM이 문제를 해결하기 위해 생성한 reasoning 경로에 대해 MCTS를 활용하여 각 단계의 '상대적 정합성' 점수를 부여합니다. 이 점수는 이진 선호(binary preferences)보다 각 단계의 품질을 더욱 정확하게 반영하며, 이를 통해 LLM의 가중치 음 로그 우도 손실 함수(weighted negative log-likelihood loss function)에 통합하여 훈련을 진행합니다. 이러한 generate-then-train 방식은 수학적 reasoning 문제에 대한 LLM의 성과를 향상시키는 반복적인 과정으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LLM의 수학적 reasoning 성능을 대폭 향상시켰습니다. 두 개의 데이터셋 각각에서 훈련된 모델이 다른 데이터셋에서도 성능 개선을 보여주었으며, 이는 향상된 추론 능력의 전이성을 시사합니다. 또한, 인간 주석이 필요하지 않은 방식으로 과정 감독을 통해 훈련할 수 있는 가능성을 제시하였습니다.



### Estimation of 3T MR images from 1.5T images regularized with Physics based Constrain (https://arxiv.org/abs/2501.01464)
Comments:
          conference paper

- **What's New**: 최근 1.5T MRI 이미지를 3T와 유사한 고품질 이미지로 개선하기 위한 새로운 비지도 학습 기반 방법이 제안되었습니다. 기존 방법들은 고품질 이미지를 얻기 위해 예제 이미지나 픽셀 대응을 필요로 했으나, 이 방법은 이러한 필요를 없앴습니다. 제안된 방법은 선형 변환을 통해 저자기장(LF) 이미지를 고자기장(HF) 이미지로 변환하는 데 초점을 맞춥니다.

- **Technical Details**: 제안된 방법은 대체 최소화(alternate minimization) 프레임워크를 사용하여 고자기장 이미지(𝐱)와 저자기장 이미지(𝐲) 간의 관계를 모델링합니다. 여기서는 물리 기반의 제약을 도입하여 T1 이완 시간의 차이를 활용하여 HF 이미지를 시뮬레이션하는 정규화자를 사용합니다. 결과적으로, 연산을 통해 생성된 이미지는 고품질 이미지로 노출되어 더 나은 조직(segmentation)와 부피 정량화(a quantification) 성능을 제공합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 1.5T 이미지를 처리하여 3T와 유사한 고품질 이미지를 생성하는 데 성공했음을 입증했습니다. 또한 기존 방법들과 비교했을 때, 조직 경계의 선명함과 이미지 대비에 있어 유의미한 개선을 보여주었습니다. 이는 의료 이미징에서 저자기장 장비의 활용성을 높일 수 있는 잠재력을 가집니다.



### GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution (https://arxiv.org/abs/2501.01460)
Comments:
          GDSR: Global-Detail Integration through Dual-Branch Network with Wavelet Losses for Remote Sensing Image Super-Resolution

- **What's New**: 이 논문에서는 Remote Sensing Image (RSI) Super-Resolution (SR) 분야에서 새로운 접근 방식을 제시합니다. Receptance Weighted Key Value (RWKV) 모델을 적용하여 긴 범위의 의존성을 선형 복잡도로 캡처함으로써 지역 및 글로벌 기능을 동시에 모델링하는 Global-Detail 구조(GDSR)를 도입하였습니다. 또한 Wavelet Loss를 제안하여 이미지의 고주파 세부 정보를 효과적으로 포착하여 시각적 품질을 향상시킵니다.

- **Technical Details**: GDSR은 RWKV와 convolutional operations를 병렬로 실행하여 대규모 RSI를 처리합니다. Global-Detail Reconstruction Module (GDRM)은 두 브랜치 간의 상호작용을 증진시켜 전체 성능을 향상시키도록 설계되었습니다. Wavelet Loss는 단순하면서도 효과적으로 PSNR과 시각적 품질을 향상시킬 수 있는 손실 함수로, 다양한 네트워크 아키텍처에서의 강력한 성능 개선을 보여줍니다.

- **Performance Highlights**: GDSR은 여러 벤치마크에서 기존의 Transformer 기반 방법에 비해 평균 0.05 dB 높은 PSNR을 달성하면서도 파라미터 수는 63%, FLOPs는 51%로 줄여 2.9배 빠른 추론 속도를 자랑합니다. 다양한 네트워크 아키텍처에서 Wavelet Loss의 일반화 능력도 뛰어나며, 이는 RSI-SR 개선을 위한 새로운 관점을 제시합니다.



### SS-CTML: Self-Supervised Cross-Task Mutual Learning for CT Image Reconstruction (https://arxiv.org/abs/2501.01456)
- **What's New**: 본 논문에서는 CT 이미지 재구성을 위한 새로운 자기 지도 크로스 태스크 상호 학습 프레임워크인 SS-CTML을 제안합니다. 이 프레임워크는 낮은 방사선 노출로 인한 데이터 부족 문제를 해결하기 위해 세 가지 재구성 작업을 하나의 구조로 통합합니다. 특히, 세 개의 신경망이 서로 학습할 수 있는 상호 학습 목표를 설정하여 최종적으로 고품질 CT 이미지를 재구성하는 데 도움을 줍니다.

- **Technical Details**: SS-CTML 프레임워크는 전체 CT(FVCT), 희귀 뷰 CT(SVCT), 제한 뷰 CT(LVCT) 재구성을 위한 독립적인 세 가지 작업을 포함합니다. 이 과정에서 서로 다른 시장 신경망을 설계하였으며, 이들은 FBP(filtered back-projection) 재구성 방식으로 촉진됩니다. 핵심적으로, 세 개의 네트워크는 서로의 출력을 기반으로 최적 학습을 수행하여 데이터의 상호 보완을 이루도록 구성되었습니다.

- **Performance Highlights**: 임상 데이터셋을 통한 실험 결과, SS-CTML 프레임워크는 저 방사선 노출 및 여러 재구성 방식에서 기존 방법들에 비해 우수한 성능을 보였습니다. 본 프레임워크는 잡음 억제, 아티팩트 감소, 세부 구조 보존 측면에서 현저한 향상을 이루어 대상 CT 이미지 품질의 향상에 크게 기여할 것으로 기대됩니다.



### Analyzing Country-Level Vaccination Rates and Determinants of Practical Capacity to Administer COVID-19 Vaccines (https://arxiv.org/abs/2501.01447)
Comments:
          31 pages, 7 figures. A previous version was presented at the 102nd Transportation Research Board Annual Meeting in Washington, D.C. in 2023

- **What's New**: 이 연구는 COVID-19 백신 접종의 글로벌 물류 운영의 복잡성을 분석하며, 각국의 백신 접종률을 대기 이론(queuing theory) 프레임워크를 통해 평가합니다. 이에 따라, 백신 접종을 위한 각국의 실제 능력을 나타내는 서비스 요율(service rates)을 도출했습니다. 또한, 여러 정부 협력체(COVAX 등)의 참여가 백신 접종 능력 향상에 기여할 수 있다는 점을 강조합니다.

- **Technical Details**: 백신 접종률은 회귀 분석(regression) 및 해석 가능한 기계 학습(interpretable machine learning) 기법을 통해 인구 통계, 정부 정책 및 사회 경제적 다양한 요소들과 연결해 분석되었습니다. 저소득 국가의 경우 도로 밀집도(roads per area)와 고소득 국가의 철도 밀집도(rail lines per area)가 백신 접종률을 개선하는 데 중요한 변수로 나타났습니다. 특히 저소득 국가에서는 기본 및 보건 인프라(health infrastructure)의 향상이 접종률 증가에 기여하는 것으로 나타났습니다.

- **Performance Highlights**: 저소득 국가에서 의료 예산, 의사 및 병상 수 증가 등 기본 인프라 개선이 백신 접종률을 높였다는 연구 결과가 도출되었습니다. 또한, 고소득 국가 내에서 65세 이상 고령자 인구 비율이 높은 경우 백신 접종률이 낮아지는 경향이 나타났습니다. 이는 노인층의 접종 접근성 문제(accessibility issues)를 시사하며, 마지막 단계에서의 접근성을 강화하고 글로벌 파트너십을 육성하는 것이 중요하다는 결론을 내립니다.



### CSI Compression using Channel Charting (https://arxiv.org/abs/2501.01431)
- **What's New**: 본 논문에서는 다중 안테나 통신 시스템의 채널 상태 정보(CSI) 보고의 오버헤드를 줄이기 위한 새로운 방법으로 채널 차팅(channel charting) 기법을 제안합니다. 기존의 CSI 압축 기법과는 달리, 이 방법은 무감독(unsupervised) 방식으로 저차원(dimensionality reduction) 표현으로 CSIs의 환경 지도를 구축하는 접근법을 사용합니다. 이를 통해, 기초로 하는 데이터와 비교했을 때 성능이 향상됨을 보여줍니다.

- **Technical Details**: 제안된 채널 차팅 기법은 통신 채널의 다양한 특성을 반영하는 저차원 표현을 생성합니다. 이러한 접근법은 실시간 데이터에서의 CSI 압축 문제 해결에 도움을 줄 수 있으며, 빠르게 변화하는 무선 환경 속에서도 효율적인 성능을 제공합니다. 이 연구는 주어진 CSI 분석을 위한 응용 프로그램에 초점을 맞추고 있으며, 다양한 기법들과의 비교를 통해 엄밀한 평가를 진행합니다.

- **Performance Highlights**: 비교 연구를 통해, 제안된 채널 차팅 기법은 현실적인 합성 데이터에 대한 성능에서 기존 방법들보다 유리한 결과를 나타냅니다. 실험 결과, CSIs의 저차원 표현을 통해 CSI의 정보 손실을 최소화하면서도 통신 시스템의 효율성을 향상시키는 데 기여할 수 있음을 입증하였습니다. 이러한 성과는 다중 안테나 시스템에서 CSI 보고의 효율성을 크게 향상시킬 것으로 기대됩니다.



### TabTreeFormer: Tabular Data Generation Using Hybrid Tree-Transformer (https://arxiv.org/abs/2501.01216)
- **What's New**: 이 논문에서는 TabTreeFormer라는 새로운 하이브리드 트랜스포머 아키텍처를 제안합니다. 이 모델은 트리 기반 모델을 통합하여 테이블 데이터의 고유한 특성을 보존할 수 있는 inductive bias를 반영합니다. 또한 듀얼 양자화(tokenization) 방식을 통해 다중 모달 연속 분포를 포착하고, 모델의 크기를 대폭 줄이면서 성능을 향상시킵니다.

- **Technical Details**: TabTreeFormer는 비선형적이고 낮은 상관 패턴을 캡처할 수 있는 트리 기반 접근 방식을 적용하여 테이블 데이터의 특수한 inductive bias를 활용합니다. 이를 통해 테이블 특성에 적합한 고유한 구조를 유지하면서도, K-Means 양자화와 분위수 양자화를 통해 다중 모달 분포를 효과적으로 모델링합니다. 이와 함께, 제한된 차원 의미와 훈련 세트를 고려하여 토크나이저를 최적화함으로써 모델의 크기를 작게 유지할 수 있습니다.

- **Performance Highlights**: TabTreeFormer는 10개의 데이터셋에서 여러 생성 모델과 비교하여 우수한 성능을 보였습니다. 실험 결과, TabTreeFormer는 1/16 크기의 모델로도 40%의 유용성 향상을 달성하였으며, 이는 높은 정확도와 효율성을 의미합니다. 따라서 이 모델은 데이터 프라이버시와 유틸리티를 모두 개선할 수 있는 가능성을 보여줍니다.




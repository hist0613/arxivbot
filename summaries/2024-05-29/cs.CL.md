### Notes on Applicability of GPT-4 to Document Understanding (https://arxiv.org/abs/2405.18433)
- **What's New**: 본 논문은 공개된 모든 GPT-4 계열 모델을 문서 이해(Document Understanding) 분야에 대해 평가합니다. 특히, 텍스트의 공간적 배열과 시각적 단서를 포함한 문서 이미지를 이해하는 데 중점을 둡니다. 평가 결과, 텍스트만을 사용하는 모델로는 만족스러운 결과를 얻기 어렵지만, 외부 OCR 엔진이 인식한 텍스트와 문서 이미지를 함께 제공하면 GPT-4 Vision Turbo 모델이 좋은 성능을 발휘한다고 보고합니다.

- **Technical Details**: 이번 연구에서는 DocVQA, InfographicsVQA, SlideVQA 및 DUDE와 같은 다양한 문서 유형을 대표하는 데이터셋을 사용하여 모델의 성능을 평가합니다. TURBO V와 TURBO V + OCR 모델은 문서의 시각적 측면(레이아웃 포함)을 처리하여 텍스트 전용 모델보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: GPT-4 Vision 모델은 OCR 엔진이 인식한 텍스트를 입력으로 제공할 때 성능이 향상됩니다. 특히, DocVQA와 같은 데이터셋에서 TURBO V + OCR 모델은 텍스트로만 구성된 모델보다 훨씬 높은 성능을 보여주며, 그림이나 이미지보다 텍스트가 많이 포함된 형태에서 더 큰 성능 향상을 보입니다. GPT-4의 텍스트 전용 모델의 경우, 입력 문맥이 길어질수록 성능이 급격히 떨어집니다.



### Don't Forget to Connect! Improving RAG with Graph-based Reranking (https://arxiv.org/abs/2405.18414)
- **What's New**: 이번 연구에서는 RAG(Retrieval Augmented Generation)의 향상을 위해 GNN(Graph Neural Networks)을 기반으로 한 새로운 리랭커 G-RAG을 제안합니다. G-RAG는 문서 간의 연결성과 의미 정보를 결합해 더 나은 문맥 정보를 제공합니다. 또한, PaLM 2를 리랭커로 사용한 경우 성능이 크게 떨어진다는 점을 발견했습니다.

- **Technical Details**: G-RAG은 문서 간의 연결성을 반영하여 각 문서노드 간의 공통 개념을 에지(feature)로 설정하고, 메시지 전달 메커니즘을 통해 에지 기능을 업데이트합니다. 노드 특징으로는 AMR(추상 의미 표현) 정보를 사용하여 복잡한 의미를 더 잘 이해할 수 있도록 했습니다. 또한, 크로스 엔트로피 손실 대신 페어와이즈(pairwise) 랭킹 손실을 적용했습니다.

- **Performance Highlights**: 새롭게 제안된 G-RAG은 현재 최고 성능을 보이는 방법들보다 우수하며, 컴퓨팅 리소스도 적게 필요합니다. MHits@10 및 Mean Tied Reciprocal Ranking과 같은 새로운 랭킹 지표를 통해 평가했으며, G-RAG은 성능, 특히 정확한 매치 성능에서 탁월한 결과를 보였습니다.



### Superposed Decoding: Multiple Generations from a Single Autoregressive Inference Pass (https://arxiv.org/abs/2405.18400)
Comments:
          22 pages, 15 figures

- **What's New**: GitHub의 코드 완성, Gmail의 스마트 컴포즈, Apple의 메시지 자동 제안 등 많은 애플리케이션이 사용자에게 여러 자동 완성 초안을 제공합니다. 그러나 이러한 초안을 제공하기 위해 언어 모델(language model)을 여러 번 실행하는 것은 비용이 큽니다. 이를 해결하기 위해, 단 한 번의 자동회귀(inference) 패스로 k개의 초안을 생성할 수 있는 새로운 디코딩 알고리즘인 Superposed Decoding(SPD)를 제안합니다. SPD는 k개의 최근 토큰 임베딩(embeddings)의 중첩을 입력값으로 사용하여 다음 디코딩 단계로 전달합니다.

- **Technical Details**: SPD는 k개의 초안을 생성하는 데 있어서 일반적인 디코딩 방법보다 2.44배 이상 빠릅니다. 디코딩 단계에서, SPD는 상위 k개의 출력 토큰을 선택하고, k개의 기존 초안을 확장하여 k^2개의 새로운 초안을 만듭니다. n-그램(n-gram) 모델을 사용하여 최대 확률과 일관된 초안을 선택하는 보간(interpolation) 과정을 거칩니다. n-그램 보간은 계산 비용이 낮고, 프로그래밍, 헬스케어, 금융 등 다양한 도메인에서 유연하게 사용할 수 있습니다.

- **Performance Highlights**: SPD가 생성하는 k개의 초안은 일관성과 사실성 면에서 Nucleus Sampling 및 Greedy Decoding과 동일하거나 더 뛰어난 성능을 보이며, 특히 k가 3 이상일 때 약 2.44배 높은 효율성을 보여줍니다. 또한 인간 평가 실험에서 SPD가 생성한 텍스트는 compute normalized 설정에서 최대 20% 더 높은 선호도를 보였습니다. SPD는 다양한 언어 모델(Mistral 7B 등)에 적용할 수 있으며, 긴 형식의 콘텐츠 생성에도 신뢰성 있게 사용될 수 있습니다.



### Thai Winograd Schemas: A Benchmark for Thai Commonsense Reasoning (https://arxiv.org/abs/2405.18375)
- **What's New**: 이번 연구는 아직까지 존재하지 않았던 태국어 Winograd Schema를 새롭게 소개하며, 이를 통해 태국어에서의 상식 추론 능력을 평가하기 위한 새로운 데이터셋을 개발했습니다. 이 데이터셋은 태국어의 특성, 관용구 및 문화적 참조를 반영하며, 모호성과 상식 도전 과제를 유지하려고 합니다.

- **Technical Details**: 두 명의 태국어 원어민 전문 번역가가 처음 85개의 문장을 번역한 후, 나머지 200개의 문장은 추가 지침에 따라 번역되었습니다. 번역 가이드라인은 이름과 문맥을 태국어에 맞게 조정하되 원래 스키마의 모호성과 뉘앙스를 유지하도록 지시했습니다. 번역 후, 세 명의 태국어 원어민이 검토하고 최종 조정을 거쳤습니다. 대규모 언어 모델 Typhoon은 Mistral-7B 아키텍처를 기반으로 구축되었고, 태국어 작업에서 성능을 향상시키기 위해 태국어 서브워드 토크나이저(subword tokenizer)가 추가되었습니다.

- **Performance Highlights**: 태국어 Winograd Schema는 태국어 자연어 처리 및 상식 추론의 교차 언어 능력을 평가했을 때, GPT-4와 Claude-3-Opus 같은 최신 모델이 영어에서는 높은 정확도를 기록했지만, 태국어에서는 성능이 크게 떨어지는 것으로 나타났습니다. 이는 다국어 상식 추론의 발전이 필요함을 강조합니다.



### PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework (https://arxiv.org/abs/2405.18369)
- **What's New**: PromptWizard라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLMs(대형 언어 모델)를 활용하여, 특정 작업에 최적화된 프롬프트를 반복적으로 합성(synthesize)하고 정제(refine)합니다. 이로써 수동 프롬프트 엔지니어링(prompt engineering)의 노동 집약적이고 도메인 특정적인 문제를 해결할 수 있습니다.

- **Technical Details**: PromptWizard는 프롬프트 지침(prompt instructions)과 컨텍스트 예시(in-context examples)를 최적화하여 모델 성능을 극대화합니다. 지침을 돌연변이(mutation)시키고 부정 예시(negative examples)를 통합해 모델의 이해도를 높이고 다양성을 보장합니다. 또한, 평론가(critic)를 통해 새로운 지침과 예시를 합성하고, 구체적인 추론 단계(reasoning steps)를 포함하여 성능을 최적화합니다. 이 과정은 컴퓨팅 효율성을 유지하면서도 다양한 훈련 데이터 양에 적응하며, 더 작은 LLMs에서도 효과적으로 작동합니다.

- **Performance Highlights**: PromptWizard는 8개의 데이터셋에서 35가지 작업을 엄격하게 평가한 결과, 기존의 프롬프트 전략보다 우수한 성능을 보여주었습니다. 이는 프롬프트 최적화에서의 효율성과 확장성을 입증합니다.



### Bridging the Gap: Dynamic Learning Strategies for Improving Multilingual Performance in LLMs (https://arxiv.org/abs/2405.18359)
- **What's New**: 이번 연구는 대형언어모델(LLMs)의 다언어 성능을 향상시키기 위한 새로운 기법을 제안합니다. LLMs는 현재 다양한 분야에서 중요한 역할을 하고 있지만, 특히 비라틴 대본과 저자원 언어에 대해서는 성능이 제한적입니다. 이를 해결하기 위해, 본 연구는 새로운 프롬프트 최적화 기법, 하이브리드 접근법, 그리고 동적 학습 접근법을 도입하여 다언어 환경에서 LLMs의 잠재능력을 최대로 이끌어냅니다.

- **Technical Details**: 연구는 세 가지 주요 전략을 포함합니다. 첫째, 폴리글롯 LLMs에 맞춰 최적화된 프롬프트를 사용함으로써 성능을 크게 향상시킵니다. 둘째, LLM Retrieval Augmented Generation (RAG)을 다언어 임베딩과 결합한 하이브리드 접근법을 도입하여 성능을 강화합니다. 셋째, 실행 시점에서 최적의 프롬프트 전략, LLM 모델, 임베딩 모델을 동적으로 선택하는 새로운 학습 접근법을 통해 다언어 작업 성능을 최대화합니다.

- **Performance Highlights**: 우리의 접근법은 IndicQA와 TyDiQA 데이터셋을 사용한 실험에서 18개 언어에 대해 15-20% 이상의 성능 향상을 보여줍니다. 본 연구는 제한된 학습 데이터와 높은 컴퓨팅 비용의 문제를 해결하며, 새로운 언어와 데이터셋에 신속하게 적응할 수 있는 능력을 갖추고 있습니다.



### MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning (https://arxiv.org/abs/2405.18358)
- **What's New**: MMCTAgent는 복잡한 시각적 추론 작업에서 현재 MLLMs(Multi-modal Large Language Models)의 한계를 해결하기 위해 설계된 새로운 멀티모달 비판적 사고 에이전트 프레임워크입니다. 인간의 인지 과정과 비판적 사고에서 영감을 받아 반복적으로 멀티모달 정보를 분석하고, 쿼리를 분해하며, 전략을 계획하고, 동적으로 추론을 발전시킵니다. 또한, MMCTAgent는 최종 답변 검증과 자기 반성을 포함하여 새로운 방식으로 비전 기반 심판(vision-based critic)을 정의하고 작업별 평가 기준을 식별하여 의사결정 능력을 강화합니다.

- **Technical Details**: MMCTAgent는 동적 계획 및 추론(dynamic planning and reasoning), 도구 보강(tool augmentation), 그리고 비전 기반 심판(vision-based critic)으로 구성됩니다. 동적 계획 컴포넌트는 사용자의 쿼리를 분해하고 문제 해결 전략을 구상하며, 도구 보강 컴포넌트를 통해 추가 정보를 수집합니다. 비전 기반 심판은 증거와 가정을 평가하고, 자동으로 작업별 평가 기준을 결정하며 최종 답변의 정확성과 추론 일관성을 평가합니다. 이를 통해 MMCTAgent는 멀티모달 데이터를 종합적으로 분석하고 검증하며 자기 반성을 통해 성능을 향상시킵니다.

- **Performance Highlights**: MMCTAgent는 MMMU, MMVET, MathVista, MMBench, OKVQA 등 이미지 이해 데이터셋에서 현존하는 최고 성능의 모델을 초과하는 성과를 달성했습니다. 예를 들어, 이미지 이해 데이터셋에서 MMCTAgent는 기존 최첨단 모델들을 최대 10% 초과하는 성과를 보였습니다. 또한, EgoSchema와 새로운 데이터셋인 MMCT-QA에서도 최첨단 접근 방식보다 최대 10% 더 높은 정확도를 기록하며 복잡한 시각적 추론 문제를 해결하는 데 탁월한 성능을 입증했습니다.



### Faithful Logical Reasoning via Symbolic Chain-of-Though (https://arxiv.org/abs/2405.18357)
Comments:
          Accepted by ACL 2024 (main proceeding)

- **What's New**: 최근 Chain-of-Thought (CoT) 기법이 대형 언어 모델(LLMs)의 추론 능력을 향상시켰으나, 여전히 기호적 표현과 엄격한 추론 규칙에 의존하는 논리적 추론을 처리하는 데 어려움을 겪을 수 있습니다. 이러한 논리적 추론 능력을 강화하기 위해, 우리는 기호적 표현과 논리 규칙을 CoT 프롬포팅(pompting)과 통합한 새로운 기법, SymbCoT(Symbolic Chain-of-Thought)를 제안합니다.

- **Technical Details**: SymbCoT는 LLM을 기반으로 1) 자연어 문맥을 기호적 형식으로 번역하고, 2) 기호 논리 규칙을 사용하여 문제를 단계별로 해결하는 계획을 도출하며, 3) 번역 및 추론 체인을 확인하는 검증기를 포함하는 프레임워크입니다. 이러한 접근 방식은 First-Order Logic 및 Constraint Optimization 기호 표현을 포함하는 5개의 표준 데이터셋에서 철저히 평가되었습니다.

- **Performance Highlights**: 평가 결과, SymbCoT는 CoT 방법에 비해 일관되게 큰 개선을 보여주었으며, 현재의 최고 성능을 갱신하였습니다. 또한, 더 신뢰할 수 있고 유연하며 설명 가능한 논리적 추론에서 시스템이 진보를 이루었습니다. 이는 LLM의 논리적 추론을 위해 기호적 표현과 규칙을 CoT에 통합한 첫 사례입니다.



### A System for Automatic English Text Expansion (https://arxiv.org/abs/2405.18350)
- **What's New**: 이 논문에서는 NLG (Natural Language Generation)를 통해 영어 문장을 자동으로 생성하는 텍스트 확장 시스템을 소개합니다. 이 시스템은 최소한의 단어 세트로부터 일관되고 올바른 문장을 자동으로 생성할 수 있습니다. 해당 시스템의 가장 큰 이점 중 하나는 모듈화된 설계로 다양한 언어에 쉽게 적용할 수 있다는 점입니다.

- **Technical Details**: 이 시스템은 통계적 접근과 언어 규칙을 결합한 하이브리드 방식으로, aLexiE라는 고정밀도 영어 어휘 라이브러리를 사용합니다. 주요 구성 요소는 영어 사전(aLexiE)와 문법 규칙, 문장 계획기와 표면 실현 장치입니다. AAC(Augmentative and Alternative Communication) 개념 증명을 통해 평가했으며, 영어와 스페인어 간의 병렬 코퍼스를 사용해 비교 분석을 수행했습니다.

- **Performance Highlights**: AAC를 위한 개념 증명에서, 텍스트 확장 결과를 직접 재생성하거나 주석에서 수동으로 평가하여 테스트를 진행했습니다. 평가 결과, 문장의 일관성, 유창성, 가독성 측면에서 높은 성과를 보였습니다. 특히, 템플릿이 적합하지 않은 경우에도 다양한 언어와 도메인에 쉽게 적용할 수 있는 유연성을 입증했습니다.



### Can Automatic Metrics Assess High-Quality Translations? (https://arxiv.org/abs/2405.18348)
Comments:
          work in progress

- **What's New**: 이 논문은 기존 번역 품질 평가 메트릭이 미세한 품질 차이에 둔감하다는 것을 확인하였으며, 이는 특히 고품질 번역에서 두드러집니다. 저자들은 현재 메트릭이 무오류 번역을 얼마나 잘 식별할 수 있는지를 체계적으로 테스트하였고, 개선의 여지가 많음을 발견했습니다.

- **Technical Details**: 특히 Multidimensional Quality Metrics (MQM) 프레임워크를 골드 스탠다드로 사용하여, 현재 메트릭이 인간이 표기한 오류가 없는 번역을 얼마나 잘 식별할 수 있는지를 평가했습니다. 이를 통해, 기존 메트릭이 고품질 번역의 품질을 과대평가하거나 과소평가하는 경향이 있음을 발견했습니다. 주요한 발견 사항은 Gemba-MQM이 최고의 F1 점수를 기록하여 무오류 번역을 잘 감지하는 반면, 오류가 있는 GPT-4 번역에도 높은 점수를 부여하는 편향을 보인다는 점입니다.

- **Performance Highlights**: 현재 메트릭은 주어진 소스에 대해 여러 번역을 비교할 때, 특히 고품질 번역을 비교할 때 어려움을 겪고 있습니다. 비교 결과, 참조 기반 메트릭에 비해 참조 없는 메트릭이 비슷한 상관 점수를 가져오는 경우도 있었습니다. 특히, Gemba-MQM은 무오류 번역을 감지하는 데 최고 성능을 보였습니다.



### The Battle of LLMs: A Comparative Study in Conversational QA Tasks (https://arxiv.org/abs/2405.18344)
Comments:
          9 pages, 4 figures, 2 tables

- **What's New**: 최신 연구에서 OpenAI의 ChatGPT와 GPT-4, Google의 Gemini, Mistral AI의 Mixtral, 및 Anthropic의 Claude 등 다양한 첨단 언어 모델들이 평가되었습니다. 이 연구는 이러한 모델들이 고객 서비스, 교육, 헬스케어, 금융 등의 분야에서 어떻게 적용될 수 있는지에 대해 깊이 있게 탐구하였습니다.

- **Technical Details**: 연구팀은 ChatGPT, GPT-4, Gemini, Mixtral 및 Claude 모델들이 다양한 Conversational QA(CQA) 코퍼스에서 생성한 응답을 분석했습니다. 평가 점수를 계산하고 비교하여 각 모델의 전반적인 성능을 파악했으며, 모델들이 질문에 대한 부정확한 응답을 제공한 사례를 분석했습니다. 본 연구는 BLEU, ROUGE, METEOR, BART 등의 메트릭스를 활용하여 모델들의 응답의 정확성, 유창성, 일관성을 평가하였습니다.

- **Performance Highlights**: 평가 결과에 따르면, GPT-4와 Claude 모델은 정확성과 관련성, 일관성 면에서 ChatGPT, Gemini, Mixtral 모델을 능가했습니다. 특히 GPT-4와 Claude는 Chain of Thought 평가, Zero Shot 및 3-shot 학습 시나리오에서 우수한 성능을 보였습니다. 그러나 ChatGPT, Gemini, Mixtral 모델은 동일한 맥락에서 질문에 답변할 때 일관성이 떨어지는 경우가 많았습니다. GPT-4와 Claude는 이러한 문제를 효과적으로 해결하여 더욱 신뢰할 수 있는 응답을 생성했습니다.



### Interpretable classification of wiki-review streams (https://arxiv.org/abs/2405.18335)
- **What's New**: 이 연구는 위키 플랫폼에서 악의적 편집이나 신뢰할 수 없는 편집자를 실시간으로 식별하여 위키 콘텐츠의 품질과 신뢰성을 개선하는 새로운 해석 가능한 분류 솔루션을 제안합니다. 최종 분류를 더 공정하게 하기 위해 클래스(balance, 균형)를 맞추는 합성 데이터 생성 알고리즘도 기여합니다. 이 방법은 실시간 스트림 기반 처리(stream-based processing)를 사용합니다.

- **Technical Details**: 제안된 방법은 프로파일링 및 분류 모델을 스트림 기반으로 업데이트합니다. 편집자 프로파일링에는 사이드 및 콘텐츠 기반 피처를 사용하는 자연어 처리(NLP, Natural Language Processing)를 사용하며, 자체 설명 가능한 분류 알고리즘(e.g., 결정 트리, decision trees)을 통해 리뷰가 반전(revert) 또는 비반전(non-revert)으로 분류된 이유를 이해할 수 있습니다. 합성 데이터 생성 알고리즘은 클래스 균형을 위해 사용됩니다.

- **Performance Highlights**: 위키보야지(Wikivoyage)에서 수집한 실제 데이터 세트로 테스트한 결과, 모든 평가 메트릭(정확도, 정밀도, 재현율, F-점수)에서 90%에 가까운 값을 달성했습니다. 이는 제안된 방법이 효율적으로 신뢰할 수 없는 편집과 악의적 편집을 식별할 수 있음을 나타냅니다.



### Joint Lemmatization and Morphological Tagging with LEMMING (https://arxiv.org/abs/2405.18308)
Comments:
          EMNLP 2015; Honorable Mention for Best Short Paper

- **What's New**: LEMMING은 레마타이제이션(lemmatization)과 태깅(tagging)을 공동으로 모델링하기 위한 모듈형 로그-선형 모델(log-linear model)입니다. 이 모델은 임의의 글로벌 피처(global features)를 통합할 수 있는 기능을 지원하며, 표준 태그와 레미타(gold standard tags 및 lemmata)로 주석된 코퍼스를 통해 훈련될 수 있습니다. 기존의 형태적 사전(morphological dictionaries)이나 해석기(analyzers)에 의존하지 않는 것이 큰 장점입니다.

- **Technical Details**: LEMMING은 레마타이제이션과 태깅을 동시에 모델링하여 상호 보완적인 효과를 달성합니다. 특히, 체코어(Czech)의 레마타이제이션에서 에러율을 60% 줄였으며(4.05에서 1.58로), 이는 토큰 기반 통계적 레마타이제이션에서 새로운 최고 성능을 달성하였습니다. 또한, 임의의 글로벌 피처를 통합할 수 있어 다양한 언어에 적용할 수 있습니다.

- **Performance Highlights**: 여섯 가지 언어에서 토큰 기반 통계적 레마타이제이션의 새로운 최고 기록을 세웠습니다. 예를 들어, 체코어 레마타이제이션의 경우 에러율을 4.05에서 1.58로 60% 감소시켰습니다. 실증적 증거를 통해 형태적 태그와 레마타를 공동 모델링하는 것이 서로에게 이익이 된다는 점을 확인했습니다.



### Semantic are Beacons: A Semantic Perspective for Unveiling Parameter-Efficient Fine-Tuning in Knowledge Learning (https://arxiv.org/abs/2405.18292)
Comments:
          Accepted at Findings of ACL 2024

- **What's New**: 이 논문은 다양한 다운스트림 작업에 대형 언어 모델(LLMs)을 효율적으로 적응시키는 Parameter-Efficient Fine-Tuning(PEFT) 방법이 사실적 지식을 학습하는 작업에서 효과가 떨어지는 이유를 탐구합니다. 저자들은 PEFT 중에 모델이 의도한 지식 목표에서 멀어질 위험이 있고, 여러 지식이 서로 간섭하여 지식 특징의 학습과 표현을 억제한다는 사실을 발견했습니다.

- **Technical Details**: PEFT 방법은 Adapter-tuning 및 LoRA와 같은 접근법을 사용하여 모델의 일부 파라미터만을 선택적으로 조정합니다. 그러나 이 방법의 효율성은 특히 고유 명사, 시간 정보, 지리적 위치 등의 사실적 지식을 학습할 때 감소합니다. 저자들은 PEFT의 한계를 시맨틱한 관점에서 분석하고, 데이터 필터링 및 가중치 재조정 전략을 통해 모델의 지식 학습 성능을 개선하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 방법들은 데이터 세트에서 지식 학습에 해로운 데이터를 배제하고, 모델이 시맨틱한 거리(semantic distance)에 주의하도록 만들기 위해 가중치를 재조정하는 전략을 사용했습니다. 실험 결과, 이러한 방법이 오픈 소스 대형 언어 모델에서 효과적임이 입증되었고, PEFT의 시맨틱한 도전 과제를 확인했습니다.



### Active Use of Latent Constituency Representation in both Humans and Large Language Models (https://arxiv.org/abs/2405.18241)
Comments:
          62 pages, 5 figures. Under review

- **What's New**: 이번 연구는 인간 뇌와 대형 언어 모델(LLMs)이 문장을 내부적으로 어떻게 표현하는지를 분석합니다. 새로운 one-shot 학습(task) 실험을 통해 인간과 LLMs가 계층적으로 구성된 구성요소(hierarchical linguistic constituents)를 유사하게 삭제(word deletion)하는 행동을 보인다는 점을 밝혔습니다.

- **Technical Details**: 연구팀은 문장에서 삭제해야 할 단어(word)를 추론하는 실험을 통해, 인간과 LLMs가 구성요소(constituent)를 삭제하는 경향을 보인다고 설명합니다. 반면 단순한 순차 처리 모델(naive sequence processing model)은 이러한 특성을 보이지 않습니다. 삭제 행동(word deletion behaviors)을 바탕으로, 문장의 잠재적인 구성 트리(constituency tree) 구조를 재구성할 수 있습니다.

- **Performance Highlights**: 이번 결과는 인간의 뇌와 LLMs 모두에서 잠재적인 트리 구조의 구성 표현(latent tree-structured constituency representation)이 나타날 수 있음을 보여줍니다. 이는 인지과학에서의 중요한 발견으로, LLMs와 인간 인지 과정의 유사성을 강조합니다.



### IAPT: Instruction-Aware Prompt Tuning for Large Language Models (https://arxiv.org/abs/2405.18203)
Comments:
          Accepted by ACL-2024

- **What's New**: 새로운 논문에서는 Instruction-Aware Prompt Tuning (IAPT)이라는 참신한 프롬프트 튜닝 방식을 소개합니다. 기존의 soft prompt tuning 방식은 많은 소프트 토큰을 입력 시퀀스에 삽입해야 하는 단점이 있어, LoRA (Low-rank adaptation)에 비해 덜 주목받았습니다. 새로운 IAPT 방식은 단 4개의 소프트 토큰만을 필요로 하면서도 고성능을 보장합니다.

- **Technical Details**: IAPT 방식은 각 Transformer 레이어에 파라미터 효율적인 소프트 프롬프트 생성기를 설치해, 입력 명령어마다 고유한 소프트 프롬프트를 생성합니다. 이러한 생성된 소프트 프롬프트는 입력 명령어의 의미적 요약으로 볼 수 있으며, 출력 생성에 효과적으로 안내할 수 있습니다. 소프트 프롬프트 생성기는 self-attention 풀링 연산, 두 개의 선형 투영 및 활성 함수로 구성된 병목 아키텍처를 채택합니다. 실험을 통해 레이어마다 다른 활성 함수가 필요함을 발견하고, 이를 자동으로 학습하는 방식을 제안합니다.

- **Performance Highlights**: 다양한 과제를 대상으로 한 실험 결과, IAPT 방식이 최근의 경쟁 기법들보다도 우수한 성능을 발휘함을 입증했습니다. 특히, 단일 백본 다중 테넌트 (single-backbone multi-tenant) 환경에서 LoRA보다 더 효율적인 것으로 나타났습니다.



### The Knesset Corpus: An Annotated Corpus of Hebrew Parliamentary Proceedings (https://arxiv.org/abs/2405.18115)
Comments:
          28 pages, 7 figures

- **What's New**: 최근 발표된 Knesset Corpus는 1998년부터 2022년까지 이스라엘 국회에서 열린 모든 본회의와 위원회 회의의 회의록을 포함하는 히브리어 의회 회의 코퍼스를 소개합니다. 이 코퍼스에는 3천만 개 이상의 문장과 3억 8천 4백만 개 이상의 토큰이 포함되어 있습니다. 문장은 형태-구문론적 정보로 주석이 달려 있으며, 발언자의 인구통계 및 정치적 속성을 반영하는 상세한 메타정보와 연결되어 있습니다.

- **Technical Details**: Knesset Corpus는 원문 데이터를 Microsoft Word (.doc 및 .docx)와 PDF 파일 형태로 Knesset Archives에서 대량으로 받아 정리하였으며, 문서마다 형식적, 인구통계적, 정치적 정보를 추출하여 정리했습니다. 추출된 문장은 풍부한 메타데이터와 함께 저장되었으며, 사용자가 쉽게 검색하고 탐색할 수 있도록 하는 그래픽 사용자 인터페이스도 개발하였습니다.

- **Performance Highlights**: 이 코퍼스를 사용하여 두 가지 연구 사례를 보여주었습니다. 첫 번째로, 시간 경과에 따른 정치적 논의 스타일의 변화를 분석하여 어휘의 풍부도가 감소하는 현상을 발견했습니다. 두 번째로, 남성과 여성 연설자의 주제 및 스타일적 차이를 조사했습니다. 이러한 연구 사례들은 이 코퍼스가 사회과학 연구에서 중요한 이스라엘 사회의 트렌드를 밝히는 데 매우 유용할 수 있음을 보여줍니다.



### Facilitating Multi-Role and Multi-Behavior Collaboration of Large Language Models for Online Job Seeking and Recruiting (https://arxiv.org/abs/2405.18113)
- **What's New**: 최근 온라인 채용 서비스의 등장으로, 기존의 취업 및 채용 방식이 혁신되고 있습니다. 이에 따라 인공지능을 사용하여 개인과 직무의 매칭을 향상시키는 고품질의 산업용 애플리케이션 개발이 필수적입니다. 새로운 연구로 MockLLM이라는 프레임워크가 제안되었으며, 이는 Large Language Models (LLMs)의 강력한 역할 수행 능력을 활용하여 모의 면접 과정을 통해 후보자 평가를 보완할 수 있습니다.

- **Technical Details**: MockLLM은 LLM을 활용한 모의 면접 생성을 통해 면접관과 후보자 간의 대화를 가능하게 합니다. 이 프레임워크는 역할 수행 모델을 다중 역할 및 다중 행동 협업 패러다임으로 설계하여 면접 단계에서의 질문 생성 및 면접 후 평가를 포함한 다양한 기능을 수행할 수 있습니다. 또한, 두 가지 주요 기술: reflection memory generation과 dynamic prompt modification을 도입하여 양측의 행동을 지속적으로 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과, MockLLM은 높은 품질의 모의 면접과 함께 최상의 개인-직무 매칭 성능을 달성할 수 있음을 보여줍니다. 이 프레임워크는 상호평가(handshake protocol) 결과를 통합하여 최종 매칭 결정을 내리며, 인터뷰어와 후보자가 새로운 인터뷰에서 기존 경험을 반영할 수 있도록 합니다.



### ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator (https://arxiv.org/abs/2405.18111)
Comments:
          16 pages

- **What's New**: 이번 연구에서는 복수의 에이전트(adversarial-defensive system)를 사용하여 RAG 생성기(generator)의 성능과 안정성을 향상시키는 새로운 방법으로 'Adversarial Tuning in a Multi-agent (ATM)' 시스템을 제안합니다. 이 시스템은 생성기가 특정 문서가 질문에 대한 답변에 도움을 주는지를 더 잘 판단할 수 있도록 유도합니다. 이러한 접근방식은 생성기의 강건성을 강화하고 LLM(Large Language Model)이 생성한 잘못된 정보(일명 'fabricated knowledge')로 야기될 수 있는 노이즈를 줄입니다.

- **Technical Details**: ATM 시스템은 두 개의 에이전트로 구성됩니다: 생성기(Generator)와 공격자(Attacker). 공격자는 거짓 정보를 생성하여 문서 리스트에 추가함으로써 노이즈를 유발하고, 생성기는 이러한 노이즈를 견디며 올바른 답변을 제공하려 노력합니다. 이 과정은 다중 에이전트 최적화를 통해 반복적으로 이루어지며, 생성기는 점점 더 강력해지고 공격자는 더 정교한 공격 패턴을 개발합니다. 이를 통해 최종적으로 생성기는 노이즈에 저항하면서 정확한 답변을 제공할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 ATM 생성기는 강력한 베이스라인보다 더 나은 성능을 보여주었으며, LLM의 조작된 정보와 같은 노이즈에 대한 내성을 크게 향상시켰습니다. 이는 대량의 AI 생성 콘텐츠가 존재하는 실제 환경에서도 유효하다는 강력한 증거로 확인되었습니다.



### Context is Important in Depressive Language: A Study of the Interaction Between the Sentiments and Linguistic Markers in Reddit Discussions (https://arxiv.org/abs/2405.18061)
- **What's New**: 최근 연구에 따르면 우울증 환자의 언어 사용은 정신 건강의 지표가 될 수 있습니다. 본 연구는 Reddit 데이터셋을 사용하여 토론 주제가 언어적 지표와 감정 표현에 미치는 영향을 조사했습니다. 특히, 우울증 환자는 더 높은 부정적 감정과 긍정적 감정을 표현하는 경향이 있음을 발견했습니다. 이는 전통적인 어휘 기반 접근 방식(lexicon based approaches)의 한계를 시사합니다.

- **Technical Details**: 연구는 Reddit 기반 데이터셋을 활용했습니다. 우울증 환자가 자진 보고한 포스트와 일반 사용자의 포스트를 비교했습니다. LIWC 도구를 사용하여 심리언어적 특징을 추출하고, 포스트의 감정 톤을 분석했습니다. RoBERTa 기반 모델과 VADER를 비교하여 RoBERTa 모델을 선택했습니다. 또한, BERTopic 알고리즘을 사용해 주제를 분류했습니다.

- **Performance Highlights**: 우울증 그룹은 컨트롤 그룹에 비해 더 많은 부정적 감정 단어를 사용했으나, 동시에 긍정적 감정이 포함된 포스트도 많았습니다. 특히, 1인칭 대명사와 분노/슬픔 관련 단어 사용이 긍정적 감정과 연관되었으며, 현재 시제 단어 사용은 부정적 감정과 연관되었습니다. 이러한 결과는 토론 주제가 언어적 지표 해석에 중요함을 나타냅니다.



### PRFashion24: A Dataset for Sentiment Analysis of Fashion Products Reviews in Persian (https://arxiv.org/abs/2405.18060)
Comments:
          8 page

- **What's New**: PRFashion24 데이터셋은 다양한 온라인 패션 스토어에서 수집된 종합적인 페르시아어 데이터셋으로, 2020년 4월부터 2024년 3월까지의 리뷰 767,272개를 포함하고 있습니다. 이는 패션 산업 내 다양한 카테고리를 포괄하는 첫번째 페르시아어 데이터셋입니다.

- **Technical Details**: 이번 연구에서는 딥러닝 기법, 특히 Long Short-Term Memory (LSTM) 네트워크와 Bidirectional LSTM과 Convolutional Neural Network (BiLSTM-CNN)의 결합을 사용하여 온라인 패션 쇼핑에 대한 감정을 분석했습니다.

- **Performance Highlights**: LSTM 모델은 81.23%의 정확도를, BiLSTM-CNN 모델은 82.89%의 정확도를 달성했습니다. 이를 통해 온라인 패션 쇼핑에 대한 긍정적인 감정을 주로 반영하는 결과를 얻었습니다. 최적화된 모델과 PRFashion24 데이터셋은 GitHub에서 공개될 예정입니다.



### Instruction Tuning with Retrieval-based Examples Ranking for Aspect-based Sentiment Analysis (https://arxiv.org/abs/2405.18035)
Comments:
          ACL Findings 2024

- **What's New**: 이번 연구에서는 ABSA(Aspect-based Sentiment Analysis) 과제를 위해 retrieval 기반 예시 랭킹을 활용한 새로운 명령 튜닝(instruction tuning) 방법을 제안합니다. 대상 샘플마다 언어 모델(Language Model, LM)을 채점기로 활용하여 입력을 기준으로 출력의 가능성을 추정하고, 후보 예시(prompt)를 이용해 예시들의 긍정/부정 여부를 랭킹으로 표시합니다.

- **Technical Details**: 이 연구에서는 LM과 retriever를 동시에 훈련시키는 교대(training schema)를 제안합니다. 목적 샘플에 따라 후보들을 긍정 및 부정 예시로 나누고, 대비 학습(contrastive learning)을 적용하여 샘플이 긍정 예시에는 가까워지고 부정 예시에는 멀어지도록 합니다. 이 방법은 고품질 명령어를 구성하고, LM을 생성 목표(generative objective)로 미세 조정할 수 있습니다. 랭킹 평가를 위해 T5 모델을 사용하며, 이는 코드와 데이터가 함께 제공됩니다.

- **Performance Highlights**: 세 가지 ABSA 하위 작업에 대한 광범위한 실험을 통해 제안된 방법의 효과가 입증되었습니다. 제안된 모델은 여러 강력한 베이스라인 모델을 능가하는 성능을 보여주었습니다.



### Edinburgh Clinical NLP at MEDIQA-CORR 2024: Guiding Large Language Models with Hints (https://arxiv.org/abs/2405.18028)
- **What's New**: MEDIQA-CORR 2024 공유 태스크는 대형 언어 모델(LLMs)이 임상 기록에서 의료 오류를 식별하고 수정할 수 있는 능력을 평가하는 것입니다. 이번 연구에서는 GPT-3.5와 GPT-4를 중심으로 다양한 프롬프트 전략을 사용하여 이러한 오류를 식별하고 수정하는 능력을 평가했습니다. 단순 프롬프트 전략의 한계를 깨닫고, 정확한 오류-스팬 예측을 제공하는 더 작은 모델을 프롬프트에 힌트로 제공하거나 다지선다형 질문으로 제시하여 성능을 개선했습니다.

- **Technical Details**: 이번 연구에서는 다양한 프롬프트 전략, 특히 In-Context Learning(ICL)와 Chain-of-Thought(CoT)을 사용하여 성능을 높였습니다. 작은 모델인 BioLinkBERT가 오류-스팬을 예측하여 이를 프롬프트에 힌트로 제공하거나 다지선다형 질문으로 설정하여 LLM 성능을 개선했습니다. 특히 8-shot ICL과 Brief CoT reasoning의 조합이 가장 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 제안된 프롬프트 전략들이 주요 공유 태스크 리더보드에서 여섯 번째로 높은 성능을 기록했습니다. 또한, 위치, 프롬프트 역할, 다지선다형 옵션의 위치 등 몇 가지 요소가 LLM의 성능에 미치는 영향을 분석했습니다. 결과적으로, 다양한 프롬프트와 힌트 전략들이 LLM의 오류 수정 능력을 현저히 향상시켰습니다.



### TimeChara: Evaluating Point-in-Time Character Hallucination of Role-Playing Large Language Models (https://arxiv.org/abs/2405.18027)
Comments:
          ACL 2024 Findings. Code and dataset are released at this https URL

- **What's New**: 최근 대형 언어 모델(LLMs) 분야에서 대형 언어 모델을 활용한 롤플레잉 에이전트(agent)의 역할이 주목받고 있습니다. 이 연구에서는 특정 시점에서의 롤플레잉(시점별 롤플레잉, point-in-time role-playing)의 중요성을 강조하며, 이를 평가하기 위한 새로운 벤치마크인 'TimeChara'를 제안합니다. 'TimeChara'는 롤플레잉 LLM에서 시점별 캐릭터 환각(character hallucination) 문제를 평가하는 데 사용됩니다.

- **Technical Details**: TimeChara 벤치마크는 네러티브 진행의 특정 시점에 캐릭터를 배치함으로써 네러티브 몰입(narrative immersion)을 높이고 스포일러를 회피하며 팬덤 롤플레잉(fandom role-playing)을 촉진합니다. 이 벤치마크는 자동화된 파이프라인을 통해 생성된 10,895개의 인스턴스로 구성되어 있습니다. 하지만 현존하는 최첨단 LLM(GPT-4o 등)에서도 여전히 시점별 캐릭터 환각 문제가 나타나고 있습니다. 이를 해결하기 위해 'Narrative-Experts'라는 방법을 제안하여 이야기 전문가(narrative experts)를 통해 추론 단계를 분할하고 캐릭터 환각을 줄이는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, Narrative-Experts 방법을 적용한 경우 시점별 캐릭터 환각을 현저히 줄이고 시공간 일관성을 향상시키는 것으로 나타났습니다. 하지만 여전히 TimeChara 벤치마크를 통해 시점별 캐릭터 환각 문제의 지속적인 도전과 향후 개선 가능성을 강조하고 있습니다.



### MultiADE: A Multi-domain Benchmark for Adverse Drug Event Extraction (https://arxiv.org/abs/2405.18015)
Comments:
          Under review; feedback welcome

- **What's New**: 이번 연구는 다양한 텍스트 유형에서 효과적인 단일 ADE(Averse Drug Event) 추출 모델 개발 여부를 탐구하기 위해 다중 도메인 벤치마크 'MultiADE'를 구축했습니다. 이 벤치마크는 기존의 여러 데이터셋과 새롭게 생성된 CADECv2 데이터셋으로 구성되어 있습니다.

- **Technical Details**: MultiADE 벤치마크는 다양한 텍스트 유형(학술 논문, 소셜 미디어 게시물 등)을 샘플하여 ADE를 추출하는 성능을 평가합니다. 새로운 CADECv2 데이터셋은 CADEC를 확장한 것으로, 더욱 다양한 약물에 대한 온라인 게시물을 포함하고 있으며, 인간 주석자가 상세한 주석 지침을 따라 주석을 달았습니다. 또한, 중간 전이 학습(transfer learning)을 통해 기존 자원을 활용하는 접근 방식이 유망함을 보였으나, 비용 효율적인 도메인 적응(domain adaptation) 방법에 대한 추가 연구가 필요함을 언급했습니다.

- **Performance Highlights**: 벤치마크 결과에 따르면, 훈련된 모델의 도메인 일반화(generalisation) 성능이 아직 완벽하지 않아서 다양한 텍스트 소스를 처리하는 데에는 한계가 있는 것으로 나타났습니다.



### Exploring Context Window of Large Language Models via Decomposed Positional Vectors (https://arxiv.org/abs/2405.18009)
- **What's New**: 최근 Transformer 기반의 대형 언어 모델(LLMs)은 입력 시퀀스의 길이 제한인 컨텍스트 윈도우(context window)로 인해 성능 저하를 겪었습니다. 이번 연구에서는 LLM의 컨텍스트 윈도우를 확장하기 위한 훈련 없이 사용할 수 있는 새로운 방법인 positional vector replacement와 attention window extension을 제안합니다. 이 방법들은 기존의 훈련된 모델을 변경하지 않고도 컨텍스트 윈도우 길이를 효과적으로 확장할 수 있습니다.

- **Technical Details**: 이번 연구는 LLM의 히든 스테이트(hidden states)에서 위치 정보를 분리(analyze)했습니다. 이를 통해 컨텍스트 윈도우 안팎의 위치 벡터(positional vectors)가 형성되는 방식과 주의(attention)에 미치는 영향을 분석했습니다. 방법론적으로는 평균 기반 분해(mean-based decomposition) 방식을 사용해 위치 벡터를 분리했으며, 두 가지 설정(직접 외삽(direct extrapolation) 및 컨텍스트 윈도우 확장(context window extension))에서의 변화를 관찰했습니다. 이를 바탕으로 두 가지 훈련 불필요 컨텍스트 윈도우 확장 방법을 고안했습니다: positional vector replacement와 attention window extension

- **Performance Highlights**: PG-19 데이터셋으로 평가한 결과, 제안된 방법들이 기존의 방법들과 비교해도 성능 저하 없이 길이가 더 긴 텍스트에 대해 일반화(generalize)할 수 있음을 확인했습니다. 특히, 이러한 접근법들은 훈련 없이도 효과적으로 작동해 기존 모델을 쉽게 확장할 수 있는 잠재력을 보여줍니다.



### fMRI predictors based on language models of increasing complexity recover brain left lateralization (https://arxiv.org/abs/2405.17992)
- **What's New**: 이번 연구에서는 자연스러운 텍스트를 들을 때 뇌의 반응을 예측하는 모델을 사용하여 좌-우 비대칭성을 분석하였습니다. 이 연구는 8개의 다양한 대형 언어 모델과 이를 통해 예측된 뇌 신호를 fMRI 데이터와 비교하였으며, 모델 크기와 성능이 증가함에 따라 왼쪽 대뇌 반구의 활성화가 오른쪽 대뇌 반구보다 더 잘 예측되는 비대칭성을 발견하였습니다.

- **Technical Details**: 이번 연구는 Le Petit Prince 오디오북을 듣는 동안 48명의 영어권 참가자의 fMRI 데이터를 사용하였습니다. 모델은 Hugging Face hub에서 제공되는 사전 학습된 언어 모델 28개를 사용하였으며, 모델의 파라미터 수는 124M에서 14.2B까지 다양합니다. 각 모델의 예측 성능은 자연 언어 처리 작업에서의 성능과 파라미터 수의 로그 값에 따라 선형적으로 증가되는 스케일링 법칙(scaling law)을 따르는 것으로 나타났습니다.

- **Performance Highlights**: 모델의 크기와 성능이 증가할수록 fMRI 뇌 반응의 왼쪽-오른쪽 대칭성 비율이 감소하고, 왼쪽 대뇌 반구의 활성화가 더 잘 예측되었습니다. 이는 언어 처리에서 왼쪽 대뇌 반구의 우세성을 보여주는 전통적인 관찰 결과와 일치합니다. 가장 작은 모델은 비대칭성을 전혀 보이지 않았지만, 큰 모델들은 점점 더 왼쪽 반구 활성화와 더 잘 맞아떨어졌습니다.



### Peering into the Mind of Language Models: An Approach for Attribution in Contextual Question Answering (https://arxiv.org/abs/2405.17980)
- **What's New**: 본 논문은 생성 AI 분야, 특히 대규모 언어 모델(LLMs)을 사용하는 상황에서의 질문 응답에 있어서 모델이 답변을 생성할 때 입력 소스 문서의 텍스트를 그대로 복사해 사용하고 있다는 점을 관찰하였습니다. 이를 기반으로 LLM의 숨겨진 상태 표현(hidden state representations)을 이용해 텍스트의 출처를 명확히 나타내는 새로운 방법을 제안합니다. 이 방법은 추가적인 모델 재훈련이나 검색 모델의 오버헤드를 피하면서도 세밀한 속성을 제공하고 답변의 품질을 유지합니다.

- **Technical Details**: 제안된 방법은 LLM이 답변을 생성할 때 생성된 토큰의 숨겨진 상태(hidden state) 표현을 활용합니다. 이 표현은 입력 텍스트의 특정 부분에서 가져온 텍스트를 명확히 구분할 수 있게 합니다. 이를 통해 우리는 구문 단계에서의 세밀한 속성을 식별할 수 있습니다. 이 방법은 추가적인 재훈련 없이도 작동하며 LLM의 생성 과정 중 내재된 컨텍스트 정보에서 토큰 레벨의 속성을 추출합니다. Verifiability-granular 데이터셋도 도입하였으며, 이는 컨텍스트 질문 응답 설정에서 LLM 생성에 대한 토큰 레벨 주석을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 텍스트의 출처를 식별하는 데 있어 GPT-4와 동등하거나 더 나은 성능을 보였습니다. 여러 LLM 아키텍처에서 견고하게 작동하며, 널리 적용 가능함을 입증했습니다.



### FASTopic: A Fast, Adaptive, Stable, and Transferable Topic Modeling Paradigm (https://arxiv.org/abs/2405.17978)
- **What's New**: 현존하는 주제 모델(topic models)의 효과성, 효율성 및 안정성을 향상시키기 위해 FASTopic을 소개합니다. FASTopic은 새로운 패러다임인 이중 의미 관계 재구성(Dual Semantic-relation Reconstruction, DSR)을 따릅니다. 기존의 전통적, 신경 VAE 기반 또는 클러스터링 기반 방법 대신, DSR은 문서, 주제 및 단어 임베딩(embedings) 간의 의미 관계를 모델링하여 잠재 주제를 발견합니다. 또한, 새로운 임베딩 운송 계획(Embedding Transport Plan, ETP) 방법을 제안합니다.

- **Technical Details**: FASTopic은 문서, 주제 및 단어 임베딩 간의 관계를 최적의 운송 계획(optimal transport plans)으로 규제하여 의미 관계를 명확히 정규화함으로써 관계 편향 문제를 해결합니다. DSR을 통해 문서-주제와 주제-단어 임베딩 간의 이중 의미 관계를 모델링하며, 이를 주제 모델링의 분포로 해석합니다. ETP는 단순한 매개변수화된 소프트맥스(parameterized softmax) 대신, 관계를 최적의 운송 계획으로 정규화하여 명확한 주제와 정확한 주제 분포를 생성하도록 돕습니다.

- **Performance Highlights**: 광범위한 실험을 통해 FASTopic이 다양한 시나리오에서 최신 기법들보다 우수한 효과성, 효율성, 적응성 및 안정성을 보임을 입증했습니다. 고효율성 및 효과성 외에도, FASTopic은 뛰어난 전이성(transferability)과 강력한 적응성(adaptivity)을 가지고 있어, 초매개변수 조정 없이도 높은 성능을 제공합니다.



### Aligning to Thousands of Preferences via System Message Generalization (https://arxiv.org/abs/2405.17977)
Comments:
          Work in progress

- **What's New**: 연구팀은 LLM(대형 언어 모델)을 다양한 사용자 선호도에 맞게 조정하는 새로운 패러다임을 제안했습니다. 이 방법은 사용자가 시스템 메시지에서 자신의 선호도를 명시함으로써 모델의 생성 행동을 사용자의 의도에 맞게 유도합니다. 이를 통해 개별적인 LLM 재훈련 없이도 사용자의 다양한 선호도에 맞출 수 있습니다.

- **Technical Details**: 연구팀은 192,000개의 다양한 시스템 메시지를 포함하는 'Multifaceted Collection'이라는 선호도 데이터셋을 만들었습니다. 이 데이터셋은 65,000개의 사용자 명령과 그에 대한 3개의 시스템 메시지가 포함되어 있으며, 각 메시지에 대한 응답이 포함되어 있습니다. 이 데이터셋을 사용하여 Janus라는 7B LLM을 훈련했으며, 5개의 벤치마크(AlpacaEval 2.0, FLASK, Koala, MT-Bench, Self-Instruct)에서 테스트했습니다. Janus는 사용자 선호도를 반영하는 다양한 새로운 시스템 메시지를 추가하여 테스트되었습니다.

- **Performance Highlights**: Janus는 Mistral 7B Instruct v0.2, GPT-3.5 Turbo, GPT-4와 비교하여 각각 75.2%, 72.4%, 66.4%의 승리율을 기록했습니다. 또한 AlpacaEval 2.0, MT-Bench, Arena Hard Auto v0.1 벤치마크에서 (응답의 유용성에 중점을 둔) LLaMA 3 8B Instruct를 +4.0%, +0.1%, +3.0%의 차이로 능가하는 성과를 보였습니다.



### Recent Trends in Personalized Dialogue Generation: A Review of Datasets, Methodologies, and Evaluations (https://arxiv.org/abs/2405.17974)
Comments:
          Presented in LREC-COLING 2024

- **What's New**: 개인화된 대화 생성에 대한 최근 연구 동향을 체계적으로 조사한 논문이 출간되었습니다. 개인화된 대화 생성(Personlized Dialogue Generation)의 정의, 사용된 데이터셋, 개발된 방법론 및 평가 지표 등을 포괄적으로 다루고 있습니다. 특히 대규모 언어 모델(LLM)의 발전과 함께, 개인화된 응답 생성의 중요성이 더욱 부각되고 있습니다.

- **Technical Details**: 논문에서는 22개의 데이터셋과 17개의 주요 작업을 분석하였으며, 개인화된 대화 생성의 5가지 문제 유형을 식별합니다. 데이터셋 부분에서는 PersonaChat, MPChat, MSC 등 다양한 데이터셋의 특성과 개인화 정보(예: persona grounding)에 대해 설명합니다. 방법론 부분에서는 최근 3년간 주요 학회(ACL, NAACL, EMNLP, AAAI 등)에 발표된 작업들을 중심으로 개인화를 구현하는 방식을 분석합니다.

- **Performance Highlights**: LLM의 개인화된 대화 생성 능력이 급격히 발전하고 있음을 강조하며, 데이터셋의 크기와 다양성, 방법론의 가정 검토, 평가 기준의 표준화 필요성을 지적합니다. 또한, 최근에 제안된 BlendedSkillTalk(BST), Persona-based Empathetic Conversation(PEC) 데이터셋과 유사한 새로운 데이터셋들이 추가적인 기능(지식, 공감, 비전 등)을 포함해 기존 데이터셋을 보완하는 방식으로 등장하고 있습니다.



### Knowledge Circuits in Pretrained Transformers (https://arxiv.org/abs/2405.17969)
Comments:
          Work in progress, 25 pages

- **What's New**: 최근 연구는 언어모델이 지식을 저장하는 메커니즘을 깊이 탐구하여 특정 지식을 표현하는 데 중요한 'Knowledge Circuits'를 발견하였습니다. GPT-2와 TinyLLAMA를 사용한 실험을 통해, 정보 헤드, 관계 헤드, 그리고 Multilayer Perceptrons가 지식을 공동으로 인코딩하는 방식을 관찰하였습니다. 또한, 지식 편집 기술이 이러한 지식 회로에 미치는 영향을 평가하고, 언어 모델의 환각현상 및 문맥 학습과 같은 행동을 해석했습니다.

- **Technical Details**: 이 연구는 Transformer 기반 언어모델의 내부 지식 회로를 연구하여 지식 저장 및 표현 메커니즘을 이해하려고 합니다. 'Knowledge Circuits'는 주어진 작업을 수행하는데 중요한 하위 그래프로, 특정 지식의 표현에 연관된 주목할만한 현상을 발견했습니다. 예를 들어, 지식 회로는 모델의 초기 단계에서 중간 단계로 지식이 집계되며, 나중 단계에서 이 정보를 더욱 강화한다는 것을 시사합니다.

- **Performance Highlights**: 지식 회로를 이용한 연구는 모델이 독립적으로 관련 지식을 회상할 수 있는 능력을 보여주었으며, 실험 결과 지식 회로가 편집된 지식을 대부분 원래 위치에서 통합하는 것으로 나타났습니다. Fine-tuning 시에는 편집된 토큰이 라는 모델의 최종 출력에 강력한 영향을 미치는 것으로 관찰되었습니다. 또한, 환각 발생 시 언어 모델이 초기 단계에서 결국 토큰으로 옮길 지식을 올바르게 전송하지 못하거나, 잘못된 정보를 선택하는 등의 문제를 확인하였습니다.



### Transformer and Hybrid Deep Learning Based Models for Machine-Generated Text Detection (https://arxiv.org/abs/2405.17964)
- **What's New**: UniBuc-NLP 팀이 SemEval 2024 Task 8에서 뛰어난 성과를 기록했습니다. 주요 과제는 인간과 AI가 생성한 텍스트를 구별하는 것으로, 특히 변환기 기반 모델(transformer-based models)과 혼합 딥러닝 모델(hybrid deep learning models)을 탐구하여 중요한 기여를 했습니다.

- **Technical Details**: 시스템은 변환기 모델(Transformer model)을 기반으로 하여 서로 다른 레이어 선택 및 병합 전략으로 구성되며, 완전히 연결된 레이어(fully connected layers)로 이어집니다. 주요 학습 단계는 두 가지로 나뉘며, 첫 번째는 변환기 가중치를 업데이트하지 않고 완전 연결된 레이어만 특정 학습률로 업데이트하는 '동결 단계(freezing phase)', 두 번째는 선택한 변환기 레이어와 완전 연결된 레이어를 더 작은 학습률로 미세 조정하는 '미세 조정 단계(fine-tuning phase)'입니다. 세 번째 서브태스크(C)의 경우 문자 수준의 특징(character level features)을 추출하는 CNN 모델을 사용하고, 단어 임베딩을 Bi-directional LSTM에 결합해 사용했습니다.

- **Performance Highlights**: 변환기 기반 모델(transformer-based model)은 서브태스크 B에서 86.95%의 정확도로 77개 팀 중 2위를 차지했습니다. 그러나 서브태스크 A 및 C에서는 과대적합(overfitting)이 주요 문제로 나타났으며, 이를 해결하기 위해 미래의 미세 조정 시 더 많은 노력이 필요하다는 점을 발견했습니다. 이러한 모델과 시스템은 GitHub에서 오픈 소스로 제공됩니다.



### Modeling Dynamic Topics in Chain-Free Fashion by Evolution-Tracking Contrastive Learning and Unassociated Word Exclusion (https://arxiv.org/abs/2405.17957)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 새로운 동적 주제 모델인 Chain-Free Dynamic Topic Model (CFDTM)이 소개되었습니다. 이는 반복적인 주제 문제와 관련 없는 주제 문제를 해결하며, 진화 추적 대조 학습 (Evolution-Tracking Contrastive Learning) 방법을 도입하여 동적 주제 사이의 유사성 관계를 구축합니다. 또한, 관련 없는 단어 배제 (Unassociated Word Exclusion) 방법을 제안하여 주제에서 관련 없는 단어를 일관되게 배제합니다.

- **Technical Details**: 기존 모델들은 Markov chains를 통해 주제를 연계했으나, CFDTM은 이를 탈피하여 진화 추적 대조 학습 (ETC) 방법을 사용합니다. 이 방법은 동적 주제 사이의 긍정적 및 부정적 관계를 설정하여 주제의 다양성을 유지하고 반복적인 주제를 완화합니다. 또한, 관련 없는 주제를 피하기 위해 UWE (Unassociated Word Exclusion) 방법을 사용하여 관련 없는 단어를 주제에서 배제합니다.

- **Performance Highlights**: CFDTM은 종합적인 실험에서 최첨단 기반 모델보다 성능이 뛰어나다는 것이 입증되었습니다. 이 모델은 고품질의 주제를 통해 주제의 진화를 효과적으로 추적하며, 하위 작업에서도 더 나은 성능을 보입니다. 또한, 진화 강도에 대한 하이퍼파라미터에 대해 강건합니다.



### Tool Learning with Large Language Models: A Survey (https://arxiv.org/abs/2405.17935)
- **What's New**: 최근 대형 언어 모델(LLMs)을 이용한 도구 학습이 복잡한 문제 해결 능력을 강화하는 유망한 패러다임으로 떠오르고 있습니다. 이에 따라 기존 문헌에 대한 체계적인 리뷰를 통해 도구 학습의 장점과 구현 방법을 종합적으로 이해할 수 있도록 하는 설문 조사가 진행되었습니다.

- **Technical Details**: 이번 설문 조사는 도구 학습이 왜 유용한지, 그리고 어떻게 구현되는지에 대한 두 가지 주요 측면을 중점으로 다룹니다. '왜 도구 학습이 유용한가'에 대한 논의는 도구 통합의 장점과 도구 학습 패러다임의 본질적 이점을 여섯 가지 측면에서 다루고 있으며, '어떻게 구현되는가'에 대한 부분은 작업 계획, 도구 선택, 도구 호출, 응답 생성의 네 가지 주요 단계로 분류하여 체계적으로 리뷰하고 있습니다. 또한, 기존 벤치마크와 평가 방법을 요약하고, 관련 단계를 중심으로 분류하였습니다.

- **Performance Highlights**: 도구 학습은 LLM의 문제 해결 능력을 크게 향상시키며, 복잡한 계산을 정확하게 수행하거나 실시간 정보를 외부 API를 통해 확인하는 등의 기능을 포함합니다. 최근 연구에서는 GPT-4와 같은 모델이 플러그인을 호출하여 외부 도구의 결과를 통합함으로써 사용자에게 더 나은 응답을 제공하는 방법을 사용하고 있습니다. 이는 LLM의 응답 정확성을 크게 개선시킵니다.

- **Future Directions**: 현재 도구 학습의 발전을 위해 풀어야 할 여러 가지 과제들이 남아 있으며, 이러한 도전 과제를 해결하는 것이 연구자들과 산업 개발자들에게 중요한 방향이 될 것입니다. 이러한 방향성을 바탕으로 도구 학습의 미래 발전을 위한 중요한 통찰력을 제공합니다.



### Online Merging Optimizers for Boosting Rewards and Mitigating Tax in Alignmen (https://arxiv.org/abs/2405.17931)
- **What's New**: 이번 논문에서는 인간 피드백을 통한 강화 학습(RLHF)의 주요 도전 과제인 인간 중심의 가치와 기본적인 능력을 유지하면서도 이익 개선을 위한 새로운 접근 방식을 제안하고 있습니다. 이를 위해 '온라인 병합 최적화기(Online Merging Optimizer)'를 도입하여 모델의 최적화 방향을 지속적으로 조절합니다.

- **Technical Details**: 온라인 병합 최적화기는 RLHF 최적화 단계마다 RL 정책과 SFT 모델을 병합하여 학습 방향을 조절합니다. 특히, SFT와 사전 훈련된 모델 간의 파라미터 차이를 사용하여 그라디언트를 조정함으로써 SFT 최적화를 최대화하는 방향으로 그라디언트를 유도합니다. 이는 다양한 LLM 패밀리와 RLHF 알고리즘에 대해 효과적임을 입증하였으며, 다양한 벤치마크에서 높은 성과를 보였습니다.

- **Performance Highlights**: 온라인 병합 최적화기는 Qwen, LLaMA 등의 LLM 패밀리와 다양한 모델 크기(1.8B에서 8B까지), 여러 RLHF 알고리즘(DPO, KTO 등)과 기존 모델 병합 방법에서 높은 성과를 보였습니다. 이는 14개의 벤치마크에서 전체 성능을 더욱 향상시키면서도 '정렬 세금(alignment tax)'을 줄여 높은 '정렬 보상(alignment reward)'을 달성했습니다.



### Long Context is Not Long at All: A Prospector of Long-Dependency Data for Large Language Models (https://arxiv.org/abs/2405.17915)
Comments:
          13 pages, 5 figures, ACL 2024

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 긴 문맥(long-context) 모델링 능력을 향상시키기 위해 새로운 데이터 마이닝 프레임워크 ProLong을 제안합니다. ProLong은 각 학습 샘플에 대해 긴 의존성 점수(long dependency score)를 할당하여, 긴 의존성을 가지는 데이터를 효과적으로 선별할 수 있게 합니다.

- **Technical Details**: ProLong은 텍스트 세그먼트 간의 의존성 강도(Dependency Strength)를 다루기 위해 델타 퍼플렉시티(delta perplexity) 점수를 사용하고, 의존성 거리(Dependency Distance)와 의존성 특이도(Dependency Specificity)라는 메트릭을 도입하여 문서의 긴 문맥 데이터를 정교하게 평가합니다. 이 점수를 바탕으로 긴 의존성을 가진 데이터를 선별하여 학습 데이터로 활용합니다.

- **Performance Highlights**: 여러 벤치마크 실험 결과, ProLong은 긴 의존성을 확인하는데 효과적이며, 이를 통해 학습된 LLM은 기존 모델보다 긴 문맥 모델링 능력이 크게 향상되었습니다. 특히, ProLong 프레임워크를 통해 학습된 모델(ProLong-7b, ProLong-13b)은 동등한 크기의 경쟁 모델을 능가하는 성능을 보여줍니다.



### Enhancing Emotion Recognition in Conversation through Emotional Cross-Modal Fusion and Inter-class Contrastive Learning (https://arxiv.org/abs/2405.17900)
Comments:
          Accepted by the 20th International Conference on Intelligent Computing (ICIC 2024)

- **What's New**: 이번 연구에서는 대화 속 감정 인식을 위한 벡터 연결 기반의 다중 모달 융합 감정 예측 네트워크를 제안합니다. 기존 방법들은 모달 간 정보 차이를 무시하고 단순 연결 방식을 사용해 정보 중복 문제를 야기했으나, 제안된 모델은 이를 극복합니다.

- **Technical Details**: 제안된 모델은 두 주요 단계로 구성됩니다. 첫 번째 단계에서는 오디오의 멜-스펙트로그램(mel-spectrogram)과 텍스트 임베딩(text embeddings)을 통해 각 모달의 정보를 추출합니다. 두 번째 단계에서는 학습 가능한 연결 벡터(joint vectors)를 사용해 두 모달의 정보를 융합하고 이를 통해 최종적으로 감정 인식을 수행합니다. 또한, 감정 레이블을 기반으로 한 감독하에 진행되는 상호 클래스 대조 학습(inter-class contrastive learning) 모듈을 설계했습니다.

- **Performance Highlights**: 제안된 모델은 IEMOCAP와 MELD 데이터셋에서 우수한 성능을 입증했습니다. 실험 결과, 모델이 감정 인식 정확도와 포괄성에서 뛰어난 성능을 보여줍니다.



### Arithmetic Reasoning with LLM: Prolog Generation & Permutation (https://arxiv.org/abs/2405.17893)
Comments:
          12 pages, 4 figures, accepted by NAACL 2024 Main Conference

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)들이 초등학교 수학 문제를 해결하는 데 있어 Prolog 프로그램을 생성하는 방식을 탐구했습니다. 전통적인 Chain of Thought (CoT) 접근 방식은 연산 과정에서 오류가 누적될 가능성이 있었지만, Prolog 코드를 이용하면 외부 코드 인터프리터를 통해 정확하게 연산을 수행할 수 있습니다.

- **Technical Details**: Prolog는 논리 프로그래밍 언어로, 비순차적인 서술 논리를 사용하여 산술 문제를 해결합니다. 연구진은 GSM8K 데이터셋을 Prolog로 변환했으며, Prolog의 논리적 성질을 이용해 데이터를 증강하여 더욱 강력한 모델 훈련을 가능하게 했습니다.

- **Performance Highlights**: Prolog 프로그램 기반의 문제 해결 방식이 GSM8K 벤치마크에서 CoT 접근 방식을 넘어서 세 가지 다른 LLM에서 우수한 성능을 보였습니다. 이러한 결과는 특히 대형 언어 모델의 훈련 강화와 더불어 높은 정확도를 가져왔습니다.



### Benchmark Underestimates the Readiness of Multi-lingual Dialogue Agents (https://arxiv.org/abs/2405.17840)
- **What's New**: 첫 번째로 우리는 컨텍스트 내 학습(in-context learning)이 다국어 TOD(task-oriented dialogue)에 충분하다는 것을 보여줍니다. 특히, 복잡한 대화 상태 추적(DST) 하위 작업을 단순화된 단계로 세분화하여 몇 가지 예제만으로도 다룰 수 있는 방식을 제안합니다. 이를 통해 X-RiSAWOZ 데이터셋(중국어, 영어, 프랑스어, 한국어, 힌디어, 힌디-영어 코드 믹스)에 대한 테스트 결과를 도출했습니다.

- **Technical Details**: 이번 연구의 주요 기여는 다국어, 다도메인 TOD에 대해 LLMs(대규모 언어 모델)가 언어의 특성을 잘 활용하면서 효과적인 학습을 할 수 있음을 처음으로 입증한 것입니다. GPT-4를 통해 DST 하위 작업에서는 LLM 기반 엔터티 정규화를 포함한 다단계 파이프라인을 활용하고, 다른 TOD 하위 작업에서는 단순한 프롬프트를 사용했습니다. 실험 데이터셋으로는 다국어, 다도메인 데이터셋인 X-RiSAWOZ를 사용하였습니다.

- **Performance Highlights**: 자동화된 평가지표로는 우리의 DST 정확도가 55.6%에서 80.3%로 나타나 SOTA 모델 대비 낮게 나타났으며 RG 하위 작업의 BLEU 점수도 상당히 낮았습니다. 그러나 검증 세트를 수동으로 평가한 결과, 원본 레이블 오류 수정과 데이터셋 주석 체계 개선을 통해 GPT-4의 DST 정확도는 89.6%-96.8%, 응답 생성은 언어와 관계없이 99% 이상의 정확도를 달성했습니다. 이는 자동화된 평가지표가 컨텍스트 내 학습의 성능을 심각하게 과소평가하고 있음을 시사합니다.



### More Than Catastrophic Forgetting: Integrating General Capabilities For Domain-Specific LLMs (https://arxiv.org/abs/2405.17830)
- **What's New**: 최근 연구들이 도메인 특화 태스크에 대해 LLM(Large Language Models)을 미세 조정하면 일반 태스크 성능이 하락하는 Catastrophic Forgetting(CF)을 겪는다고 밝혔습니다. 이와 달리, 이 논문은 새로운 도전 과제인 General Capabilities Integration(GCI)를 제시하며, 이는 일반적인 능력과 도메인 지식을 단일 인스턴스에서 통합하는 것을 목표로 합니다.

- **Technical Details**: 논문은 법률 도메인을 예시로 GCI 설정을 통한 세 그룹의 훈련 및 테스트 태스크를 설계하고, 해당 데이터셋을 구축했습니다. 이를 위해 ALoRA라는 새로운 어댑터 아키텍처를 도입했으며, 이는 LoRA를 기반으로 여러 개의 헤드 어텐션 모듈을 활용해 이전 토큰에서 현재 토큰으로 직접적인 정보 이동을 촉진합니다. 이로써 도메인 지식과 일반 능력 간의 동적인 전환이 가능해졌습니다.

- **Performance Highlights**: 제안된 GCI와 기존 CF 문제를 비교한 실험에서, GCI 설정의 중요성과 ALoRA 방법의 효과성을 강조하는 결과가 도출되었습니다. 이를 통해 도메인 지식과 일반 능력을 효과적으로 통합한 법률 도메인 LLM의 성능이 입증되었습니다.



### Conv-CoA: Improving Open-domain Question Answering in Large Language Models via Conversational Chain-of-Action (https://arxiv.org/abs/2405.17822)
- **What's New**: 이번 연구에서는 Open-domain Conversational Question Answering(OCQA)를 위한 대화형 행동 사슬(Conversational Chain-of-Action, Conv-CoA) 프레임워크를 제안합니다. Conv-CoA는 실시간 또는 도메인 사실과 일치하지 않는 비신뢰성 생성(unfaithful hallucination), 대화 시나리오에서의 약한 추론 성능, 대화형 정보 검색의 부진한 성능 등 세 가지 주요 문제를 해결합니다.

- **Technical Details**: Conv-CoA의 핵심은 질문의 의도를 추출하고, 이를 체계적인 프롬프트, 사전 설계된 액션, 컨텍스트 지식 세트(Contextual Knowledge Set, CKS) 업데이트, Hopfield 기반 검색기를 통해 해결하기 위해 추론 체인으로 분해하는 동적 추론-검색 메커니즘입니다. 특히 효율성과 정확성을 높이기 위해 자원 효율적인 Hopfield 검색기와 대화형 다중 참조 신뢰도 점수(Conversational-multi-reference faith score, Conv-MRFS)를 제안하였습니다.

- **Performance Highlights**: 다섯 가지 연구 방향과 두 개의 공개 벤치마크에서 23개의 최신 방법들과 비교한 결과, Conv-CoA는 정확성과 효율성 면에서 다른 방법들보다 우수한 성능을 보였습니다.



### TransVIP: Speech to Speech Translation System with Voice and Isochrony Preservation (https://arxiv.org/abs/2405.17809)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 TransVIP라는 새로운 음성-음성 번역 모델 프레임워크를 소개합니다. TransVIP는 다양한 데이터셋을 사용하여 단계적(cascade) 방식으로 동작하면서도, 공동확률(joint probability)을 통해 엔드 투 엔드 추론(end-to-end inference)을 가능하게 합니다. 특히, 원본 음성의 화자 보이스 특성 및 동시성을 유지하도록 두 개의 분리된 인코더를 제안하였으며, 이는 비디오 더빙과 같은 시나리오에 매우 적합합니다.

- **Technical Details**: TransVIP는 크게 세 부분으로 구성됩니다: 첫째, 음성 양자화(quantization) 및 재구성을 위한 코덱 모델, 둘째, 입력 음성을 처음에 거친 음성(coarse-grained speech)으로 번역하는 자동회귀식(joint) 번역 모델, 그리고 셋째, 출력 음성에 음향적 세부사항을 추가하는 비자동회귀식(textless non-autoregressive acoustic) 모델입니다. 이 모델은 여러 입력 모달(Speech/Text), 출력 모달(Speech/Text), 제어 신호(Voice/Isochrony)를 지원하며, 텍스트 및 음성 생성의 공동 확률 모델링을 통해 최적화됩니다.

- **Performance Highlights**: 프랑스어-영어 상호 번역 실험에서 TransVIP 모델은 기존의 최첨단(SOTA) 음성-음성 번역 모델보다 뛰어난 성능을 보여주었습니다. 모델의 결과물은 https://aka.ms/transvip 에서 확인할 수 있습니다.



### Detection-Correction Structure via General Language Model for Grammatical Error Correction (https://arxiv.org/abs/2405.17804)
Comments:
          Long paper. Accepted by ACL 2024 Main Conference

- **What's New**: 이 논문은 기존의 직접 교정 접근 방식과는 달리 검출 및 교정을 하나의 모델에 통합한 새로운 문법 오류 수정(GEC) 모델, DeCoGLM을 소개합니다. 이 모델은 General Language Model(GLM)을 기반으로 하며, 오류 검출 및 지역적 오류 교정을 통합하여 성능을 향상시킵니다.

- **Technical Details**: DeCoGLM은 검출 단계에서 fault-tolerant detection template(결함 허용 검출 템플릿)을 사용하고, 교정 단계에서는 autoregressive mask infilling(자가회귀 마스크 채우기)을 활용하여 오류를 교정합니다. 입력 토큰의 구성과 attention mask의 조정을 통해 멀티태스크 학습을 구현했습니다. 이로써 검출과 교정을 하나의 모델 내에서 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: DeCoGLM은 영어와 중국어 GEC 데이터셋에서 기존 SOTA(State-of-the-Art) 모델들과 비교했을 때 우수한 성능을 보였습니다. 특히 작은 검출 모델과 결합된 LLM(대형 언어 모델) 교정기를 포함하는 단일 시스템은 다른 Seq2Seq LLM들보다 뛰어난 성능을 발휘했습니다.



### On the Sequence Evaluation based on Stochastic Processes (https://arxiv.org/abs/2405.17764)
- **What's New**: 이 논문은 장문의 텍스트 시퀀스를 모델링하고 분석하기 위한 새로운 접근법을 제시합니다. 이 작업은 기계 번역, 텍스트 생성 등 많은 후속 작업에 중요한 역할을 합니다. 제안된 방법은 확률적 과정(stochastic process)을 통해 시퀀스를 모델링하며, 텍스트 인코더(text encoder)를 위한 새로운 학습 목적함수(likelihood-based training objective)를 도입합니다.

- **Technical Details**: 논문은 장문의 텍스트 평가를 위한 새로운 점수 척도(score)를 설계했습니다. 이 점수는 기존 접근법보다 시간적 및 공간적 종속성(temporal and spatial dependencies)을 더 철저하게 포착합니다. 제안된 학습 목적함수는 시퀀스의 연속성을 잘 보존하는 반면, 새로운 점수는 시퀀스를 평가하는 데 더 포괄적인 특성을 가집니다. 이론적 분석을 통해 이 새로운 점수의 장점을 보여줍니다.

- **Performance Highlights**: 실험 결과는 다양한 시퀀스 평가 작업에서 우수한 성능을 보였습니다. 특히, 문서 내외의 전역(global) 및 지역(local) 차별화에서 뛰어난 성능을 입증했습니다. 또한, 인코더는 인간과 AI가 작성한 텍스트를 구분하는 데 있어서도 경쟁력 있는 결과를 보여주었습니다.



### XL3M: A Training-free Framework for LLM Length Extension Based on Segment-wise Inferenc (https://arxiv.org/abs/2405.17755)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문에서는 매우 긴 텍스트 시퀀스에 대한 추론 능력을 갖춘 새로운 프레임워크 XL3M(extra-long large language model)을 제안했습니다. XL3M은 추가 훈련이나 파인튜닝 없이, 짧은 시퀀스로 훈련된 대형 언어 모델(LLM)이 긴 시퀀스를 추론할 수 있도록 합니다. 이 프레임워크는 기존의 하드 코스트와 정밀도 손실 문제를 해결합니다.

- **Technical Details**: XL3M은 입력 컨텍스트를 독립적인 세그먼트와 원래 컨텍스트의 끝 부분에서 몇 개의 토큰을 가져와 공통 '질문'으로 변환하는 다수의 짧은 서브 컨텍스트로 분해합니다. 각 서브 컨텍스트에 대해 로컬 조건부 확률 분포(cpd)를 계산하고 대응하는 엔트로피를 측정하여, 엔트로피 값이 작은 관련 서브 컨텍스트를 선택해 시간순으로 재구성합니다. 이렇게 추출된 '핵심 컨텍스트'를 사용하여 원래 컨텍스트를 대신해 추론 작업을 완료합니다.

- **Performance Highlights**: XL3M 프레임워크는 종합 벤치마크 및 'Needle in a Haystack' 과제에서 선두적인 방법들과 비교하여 우수한 성능을 입증했습니다. Llama2-7B 모델을 사용하여 8개 카드의 Huawei Ascend 910B NPU 머신에서 20M 이상의 긴 시퀀스를 추론할 수 있음을 보여주었습니다. XL3M은 메모리와 시간 효율적이며 추가적인 훈련을 요구하지 않습니다.



### ORLM: Training Large Language Models for Optimization Modeling (https://arxiv.org/abs/2405.17743)
Comments:
          Work in progress

- **What's New**: 이 논문은 데이터 프라이버시 문제를 해결하고자 오픈 소스 LLMs (Large Language Models)을 최적화 모델링에 적용하는 방안을 제안합니다. 이를 위해, OR-Instruct라는 반자동화된 프로세스를 통해 요구 사항에 맞는 합성 데이터를 생성하며, IndustryOR이라는 최초의 산업용 벤치마크를 도입했습니다. 또한, 이 모델을 사용하여 공개 소스 LLMs를 교육시키고, 이로부터 뛰어난 성능을 보이는 ORLMs를 개발했습니다.

- **Technical Details**: OR-Instruct는 초기 시드 데이터 세트를 수집하고, GPT-4를 사용해 다양한 시나리오와 질문 유형의 데이터를 생성하는 부트스트래핑 과정을 포함합니다. 또한, 목표와 제약 조건을 변경하고 질문을 다시 표현하며 여러 모델링 기술을 포함시켜 교육 데이터 풀을 확장합니다. 최종적으로, 7b 크기의 공개 소스 LLMs를 사용해 ORLMs를 교육합니다.

- **Performance Highlights**: ORLMs는 NL4OPT, MAMO, IndustryOR 벤치마크에서 뛰어난 성능을 보이며, 최신 기술 수준의 성과를 달성했습니다.



### MobileConvRec: A Conversational Dataset for Mobile Apps Recommendations (https://arxiv.org/abs/2405.17740)
- **What's New**: 기존 추천 시스템은 주로 사용자-아이템 상호작용(history) 또는 대화형 추천 시스템(conversational recommendation system)이라는 두 가지 패러다임에 중점을 두어 왔습니다. MobileConvRec는 모바일 앱에 특화된 대화형 추천 시스템 연구를 지원하기 위한 새로운 데이터셋입니다. Google Play 스토어의 실제 사용자 상호작용을 활용하여 다중턴(multi-turn) 대화를 시뮬레이트합니다. 이 데이터셋은 45개의 앱 카테고리에서 12K 이상의 다중턴 대화를 포함하고 있습니다.

- **Technical Details**: MobileConvRec는 사용자의 내재적(implicit) 선호를 반영하는 순차적 사용자-아이템 상호작용과 명시적(explicit) 요구를 효과적으로 파악하는 다중턴 대화를 결합합니다. 시뮬레이션 과정은 두 단계로 나뉘며, 먼저 의미적 수준에서 대화 개요를 생성하고, 이를 맥락적 자연어 발화로 변환합니다. 또한, 각 앱에 대한 권한 데이터, 보안 및 개인정보 보호 관련 정보 등의 메타데이터도 풍부하게 포함되어 있습니다.

- **Performance Highlights**: MobileConvRec는 11.8K명의 고유 사용자와 1,730개의 앱을 포함하며, 총 156K 턴의 대화를 다룹니다. 다양한 대화형 추천 시스템 연구를 검증할 수 있는 우수한 테스트베드로 사용될 수 있으며, 여러 사전 훈련된 대형 언어 모델의 비교 연구를 통해 그 효용성이 입증되었습니다.



### C$^{3}$Bench: A Comprehensive Classical Chinese Understanding Benchmark for Large Language Models (https://arxiv.org/abs/2405.17732)
Comments:
          4 figures and 5 tables

- **What's New**: 이번 논문은 대형 언어 모델 (Large Language Models, LLMs)의 고전 중국어 이해 기량을 평가하기 위해 C$^{3}$bench라는 포괄적인 공정 시험셋 (benchmark)을 새롭게 제안합니다. 이 시험셋은 분류, 검색, 명명 엔터티 인식, 구두점, 번역 등의 5가지 주요 CCU (Classical Chinese Understanding) 작업을 포함한 50,000개의 텍스트 쌍으로 구성되어 있습니다.

- **Technical Details**: C$^{3}$bench는 10개의 다른 도메인에서 수집된 데이터를 포함하여, 다양한 카테고리를 아우르는 고전 중국어 데이터를 제공합니다. 이 시험셋은 자연 언어 생성 (Natural Language Generation, NLG)을 이용하였으며, 모델이 넓고 다양한 이해 능력을 갖추도록 설계되었습니다. 특히 고전 중국어의 특징적인 언어 구조와 어휘를 감안하여, 모델은 다채로운 도메인 지식을 요구받습니다.

- **Performance Highlights**: 15개의 대표적인 LLM을 사용하여 C$^{3}$bench를 통해 평가한 결과, 현재의 LLM들이 CCU 과제에서 어려움을 겪고 있으며, 감독 학습 모델 (supervised models)보다 여전히 열등하다는 것이 드러났습니다. 이는 CCU가 특별한 주의가 필요한 과제임을 시사합니다.



### CLAIM Your Data: Enhancing Imputation Accuracy with Contextual Large Language Models (https://arxiv.org/abs/2405.17712)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 확장된 지식과 추론 능력을 활용하여 테이블 형식 데이터셋에서 누락된 데이터를 처리하는 새로운 전략인 CLAIM(Contextual Language model for Accurate Imputation Method)을 소개합니다. 전통적인 보간 방법이 주로 수치 추정에 의존하는 것과 달리, CLAIM은 문맥적으로 관련 있는 자연어 설명을 사용하여 누락된 값을 채웁니다. 이 접근법은 데이터셋을 자연어 문맥화된 형식으로 변환하여 LLM의 기능과 보다 잘 맞춰집니다.

- **Technical Details**: CLAIM은 먼저 누락된 값 설명자를 생성하기 위해 LLM을 사용하고, 그런 다음 풍부해진 데이터셋에서 LLM을 미세 조정하여 다운스트림 작업에서 성능을 향상시킵니다. 다양한 데이터셋과 누락 패턴에 대한 평가 결과, CLAIM이 기존의 보간 기술보다 우수한 성능을 보였습니다. 또한, 문맥 특정 설명자와 일반 설명자의 효율성을 조사하여 누락된 데이터의 문맥적 정확성이 LLM 성능 향상에 중요한 역할을 한다는 것을 강조합니다.

- **Performance Highlights**: CLAIM은 데이터 분석과 머신러닝 모델의 신뢰성과 품질을 크게 향상시킬 수 있는 잠재력을 가지고 있습니다. 이 방법은 특히 누락된 데이터를 처리하는 데 있어 보다 미세하고 효과적인 솔루션을 제공합니다.



### HEART-felt Narratives: Tracing Empathy and Narrative Style in Personal Stories with LLMs (https://arxiv.org/abs/2405.17633)
- **What's New**: 이번 연구에서는 LLMs (Large Language Models)과 대규모 크라우드소싱 연구를 통해 서사 스타일과 공감의 관계를 정량적으로 분석하였습니다. 새로운 이론 기반의 분류 체계인 HEART(Human Empathy and Narrative Taxonomy)를 도입하여 서사 스타일의 요소가 공감을 유발하는 방식을 정의합니다.

- **Technical Details**: HEART 분류 체계를 활용하여 서사 스타일의 요소들을 분류하고, LLM을 사용해 이러한 요소들을 정량화했습니다. 기존의 사전 기반 방법과 비교했을 때, LLM (특히 GPT-4)이 더 우수한 성능을 보였습니다. 또한, 크라우드소싱을 통해 2,624명의 참여자로부터 이야기에 대한 공감 판단 데이터를 수집하여, 인간 중심의 사회적·행동적 통찰력을 제공하는 데 사용되었습니다.

- **Performance Highlights**: 이번 연구에서는 서사 스타일의 감정 생동감과 플롯 볼륨이 이야기에 대한 공감을 키우는 경로를 설명하는 데 중요한 역할을 한다는 점을 밝혀냈습니다. 또한, 개별 독자의 특성과 이야기 스타일 이외에도 독자의 공감 능력과 화자와의 경험 유사성 등이 공감을 크게 좌우한다는 점을 확인했습니다.



### Explainable machine learning multi-label classification of Spanish legal judgements (https://arxiv.org/abs/2405.17610)
- **What's New**: 이번 연구에서는 법적 판결의 다중 라벨(Multi-label) 분류를 위한 혁신적인 시스템을 제안합니다. 이 시스템은 판결의 분류뿐만 아니라 자연어 설명도 제공합니다. ML(머신 러닝)을 활용하여 판결의 내용을 다중 라벨로 분류하고, 자연어 처리(NLP) 기술을 사용해 설명을 생성합니다.

- **Technical Details**: 이 연구에서는 판결문을 다중 라벨로 분류하기 위해 여러 ML 알고리즘을 사용합니다. 특히, 텍스트 분석을 위해 NLP 기술과 심층 법적 추론(Deep Legal Reasoning)을 결합하여 관련 법적 엔티티(정당 등)를 식별합니다. 기존의 다중 라벨 분류 접근법인 변환(Transformation)과 적응(Adaptation)을 결합하여 더 나은 결과를 도출합니다.

- **Performance Highlights**: 제안된 시스템은 법적 전문가들이 주석을 단 데이터 셋에서 85% 이상의 마이크로 정밀도(micro precision)를 기록했습니다. 이는 인간 전문가가 수행하던 단순한 반복 작업을 크게 줄여줄 수 있는 가능성을 보여줍니다.



### Why are Visually-Grounded Language Models Bad at Image Classification? (https://arxiv.org/abs/2405.18415)
- **What's New**: 새로운 연구에서는 시각 신호를 통합한 언어 모델(VLMs)을 통해 이미지 분류 작업을 재조명합니다. 기존의 독점 및 공개 VLM들이 CLIP과 같은 비전 인코더를 사용하지만, ImageNet과 같은 표준 이미지 분류 벤치마크에서 CLIP보다 성능이 현저히 떨어진다는 것을 발견했습니다.

- **Technical Details**: 본 연구에서는 VLM들이 이미지 분류에서 왜 성능이 낮은지 이해하기 위해 여러 가설을 탐구했습니다. 주로 데이터와 관련된 문제임을 밝혔습니다. 이미지 분류에 필요한 정보는 VLM의 잠재 공간에 인코딩되어 있지만, 이를 효과적으로 디코딩하려면 충분한 훈련 데이터가 필요합니다. VLM 훈련 중 클래스 노출 빈도와 해당 클래스의 성능 사이에 강한 상관관계가 있음을 발견했습니다. 데이터를 충분히 사용하면, VLM들도 최첨단 분류 모델의 정확성을 맞출 수 있습니다.

- **Performance Highlights**: 분석 결과를 바탕으로, 분류 특화 데이터셋을 VLM 훈련에 통합하여 성능을 향상시켰습니다. 이러한 향상된 분류 성능은 VLM의 일반적인 능력으로 전이되었으며, 새로 수집된 ImageWikiQA 데이터셋에서 11.8%의 성능 향상을 달성했습니다. 또한, LLaVA1.5-13B가 LLaVA1.5-7B보다, LLaVANeXT-M7B가 LLaVANeXT-V7B보다 성능이 우수함을 발견했습니다.



### RACCooN: Remove, Add, and Change Video Content with Auto-Generated Narratives (https://arxiv.org/abs/2405.18406)
Comments:
          The first two authors contribute equally. Project Page: this https URL

- **What's New**: 이 논문에서는 RACCooN이라는 비디오-단락-비디오 생성 프레임워크를 제안합니다. 이는 다양한 비디오 편집 기능(삭제, 추가, 수정)을 지원하며, 자동으로 생성된 텍스트 설명을 활용합니다. 이 시스템은 두 가지 주요 단계로 나뉩니다: V2P(Video-to-Paragraph)와 P2V(Paragraph-to-Video)입니다. V2P 단계에서는 비디오 장면을 자연어로 자동 설명하며, P2V 단계에서는 사용자가 이 설명을 수정하여 비디오 확산 모델(Video Diffusion Model)이 이를 기반으로 편집된 비디오를 생성합니다.

- **Technical Details**: RACCooN은 다중 그레뉼러 비디오 인식 전략(Multi-granular video perception strategy)을 사용하여 비디오의 지역화된 맥락을 포착합니다. 이를 위해 슈퍼픽셀(Superpixels)을 사용해 비디오를 여러 수준의 그레뉼러리티로 분할하고, 중첩 K-평균 군집화를 적용합니다. 나아가, 자동으로 생성된 텍스트 설명을 활용하여 사용자가 수정한 프롬프트(Prompt)를 기반으로 비디오의 특정 영역을 정확히 칠할 수 있는 비디오 확산 모델을 개발했습니다. 이 모델의 훈련을 지원하기 위해 'VPLM(Video Paragraph with Localized Mask)' 데이터셋도 수집했습니다. 이 데이터셋에는 7.2K개의 비디오-단락 설명과 5.5K개의 자세한 객체 설명 및 마스크가 포함되어 있습니다.

- **Performance Highlights**: RACCooN 시스템은 다양한 비디오 데이터셋 (ActivityNet, YouCook2, UCF101, DAVIS, VPLM)에서 시험되었으며, 기존의 비디오 캡션 생성 및 편집 성능을 뛰어넘었습니다. 특히 YouCook2 데이터셋에서 기존 모델 대비 평균 +9.4% 성능 향상을 보였습니다. 또한, RACCooN 프레임워크는 기존 비디오 생성 모델에서도 텍스트 설명의 정확성을 높여주는데 기여했습니다.



### OwLore: Outlier-weighed Layerwise Sampled Low-Rank Projection for Memory-Efficient LLM Fine-tuning (https://arxiv.org/abs/2405.18380)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)을 효율적으로 미세 조정하기 위한 새로운 접근 방식인 Outlier-weighed Layerwise Sampled Low-Rank Projection(OwLore)을 제안합니다. 이 방법은 사전 학습된 LLM 레이어의 이상값(Outlier) 분포를 동적으로 샘플링하여 미세 조정을 수행하며 추가 어댑터를 사용하는 기존 방법과 다릅니다. 이 접근 방식은 중(HT-SR) 이론에 의해 영감을 받아 레이어별 이상값에 가중치를 부여하여, 더욱 효율적인 메모리-성능 트레이드 오프를 제공합니다.

- **Technical Details**: OwLore은 Heavy-Tailed Self-Regularization(HT-SR) 이론을 통해 분석된 LLM 레이어의 이상값 분포를 활용해 레이어별로 높은 샘플링 확률을 부여합니다. 또한, 메모리 요구량을 줄이기 위해 그레디언트 저랭크 프로젝션(gradient low-rank projection)을 통합하여 레이어별로 저랭크 방식으로 효율적으로 학습할 수 있습니다. 이러한 접근 방식은 원본 최적화 경로를 유지하면서 메모리 효율성을 높입니다.

- **Performance Highlights**: OwLore은 LLaMa2, LLaMa3, Mistral 등의 다양한 LLM 아키텍처와 벤치마크에서 탁월한 성능을 보였습니다. Commonsense Reasoning 벤치마크에서 평균 1.1%의 정확도 향상, MMLU에서 3.0%의 성능 개선, 그리고 MT-Bench에서 10%의 성능 증가를 달성하였으며, LLaMa2-7B를 21GB의 메모리로만 미세 조정할 수 있음을 보였습니다.



### Self-Supervised Learning Based Handwriting Verification (https://arxiv.org/abs/2405.18320)
Comments:
          14 pages, 6 figures, 2 tables

- **What's New**: 이번에 소개할 논문은 'SSL-HV: Self-Supervised Learning을 Handwriting Verification에 적용'입니다. 이 연구는 손글씨 검증 작업에서 자기지도 학습(Self-Supervised Learning) 접근법을 활용하고 있습니다. 손글씨 검증은 주어진 손글씨 이미지 쌍이 동일한 작가에 의해 쓰여졌는지를 판별하는 작업입니다.

- **Technical Details**: 다양한 생성적(generative) 및 대조적(contrastive) SSL 접근법을 비지도 학습(handcrafted feature extractors) 및 지도 학습(supervised learning)과 비교하였습니다. 주요 기술적 세부사항으로는 ResNet 기반의 변이형 오토인코더(Variational Auto-Encoder, VAE)와 ResNet-18을 활용한 Variance-Invariance-Covariance Regularization (VICReg) 등이 있습니다.

- **Performance Highlights**: 실험 결과, 생성적 접근법 중에서는 ResNet 기반의 VAE가 76.3% 정확도로 가장 높은 성능을 보였습니다. 대조적 접근법 중에서는 VICReg로 미세 조정된 ResNet-18이 78% 정확도로 최고의 성능을 나타냈습니다. 이는 10%의 작가 라벨만을 사용하여, ResNet-18 지도 학습 기준선에 비해 각각 6.7% 및 9%의 상대적 정확도 향상을 보여줍니다.



### Text-only Synthesis for Image Captioning (https://arxiv.org/abs/2405.18258)
- **What's New**: 이미지 캡션 작업에서 인간의 노동력과 컴퓨팅 시간을 줄이기 위해 텍스트 전용 합성(Text-only Synthesis) 방법 ToCa를 제안했습니다. ToCa는 캡션 텍스트를 구조와 어휘 단어들로 분해하고, 이를 결합하여 대형 언어 모델(LLM)에서 다양한 패턴의 캡션을 생성합니다. 이를 통해 모델의 제로샷 일반화 능력을 향상시키고, 새로운 캡션을 생성할 수 있습니다.

- **Technical Details**: ToCa는 캡션을 의미 있는 어휘 단어(pair)와 구조 템플릿으로 분해하여, LLM을 이용해 이들을 결합한 다양한 캡션을 생성합니다. 텍스트만을 활용해 캡션을 생성하므로, 고비용의 이미지-텍스트 페어 데이터 작성이 필요 없으며, 실용성과 유연성을 제공합니다. 또한, LLM은 오픈소스이므로 접근성이 높고, 데이터 프라이버시를 보호할 수 있습니다.

- **Performance Highlights**: ToCa는 3가지 합성 시나리오(도메인 내 합성, 교차 도메인 합성, 데이터 효율적 합성)로 실험을 진행해 약 5 CIDEr의 제로샷 교차 도메인 캡션 성능 향상과 20 CIDEr 이상의 데이터 효율적 캡션 성능 향상을 확인했습니다.



### A Human-Like Reasoning Framework for Multi-Phases Planning Task with Large Language Models (https://arxiv.org/abs/2405.18208)
- **What's New**: 이번 연구는 LLM 에이전트가 종합적인 계획 수립이 필요한 여행 계획 문제를 효과적으로 해결하도록 돕는 인간 유사 계획 프레임워크를 개발하는 데 중점을 두었습니다. 이 프레임워크는 Strategy Block과 Knowledge Block을 통합하여 정보 수집 및 세부 계획 수립을 돕습니다.

- **Technical Details**: 프레임워크는 세 가지 주요 단계로 구성됩니다: (1) Outline Generation Phase: 여행 계획 윤곽을 작성, (2) Information Collection Phase: 필요한 정보 수집, (3) Plan Making Phase: 수집된 정보를 바탕으로 세부 여행 계획 수립. 각 단계는 다중 에이전트 협업을 통해 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존의 GPT-4-Turbo 기반 프레임워크에 비해 10배의 성능 향상을 보여주었습니다. 이는 LLM 에이전트가 여행 계획과 같은 복잡한 계획 문제를 해결하는 데 크게 기여함을 나타냅니다.



### Source Echo Chamber: Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop (https://arxiv.org/abs/2405.17998)
- **What's New**: 최근 연구에 따르면 뉴럴 검색 모델(neural retrieval models)이 인공지능 생성 컨텐츠(AI-generated content, AIGC)를 선호하는 소스 바이어스(source bias)를 가지는 것을 발견했습니다. 추천 시스템의 피드백 루프(feedback loop) 내에서 이 소스 바이어스가 어떤 영향을 미치는지에 대한 연구는 부족한 상황입니다. 본 연구에서는 AIGC가 추천 시스템의 컨텐츠 생태계로 통합되는 과정을 세 가지 단계로 나누어 HGC-지배 단계(Human Generated Content dominate phase), HGC-AIGC 공존 단계(coexist phase), AIGC-지배 단계(dominance phase)로 정의하고, 각각의 단계에서 AIGC가 소스 바이어스를 어떻게 증폭시키는지를 조사했습니다.

- **Technical Details**: 연구의 첫 번째 단계에서는 세 가지 도메인의 데이터를 통해 여러 인기 있는 추천 모델들이 AIGC를 더 높은 순위에 위치시킨다는 것을 발견했습니다. 두 번째 단계에서는 사용자와의 상호작용 데이터를 활용해 피드백 루프 내에서 AIGC가 소스 바이어스를 어떻게 증폭시키는지를 조사했습니다. 세 번째 단계에서는 데이터의 오염을 막기 위해 블랙박스 디바이싱(black-box debiasing) 방법을 도입하여 AIGC와 HGC를 균등하게 처리함으로써 소스 바이어스를 방지하고자 했습니다.

- **Performance Highlights**: 세 가지 도메인의 데이터를 통해 진행된 실험 결과, 피드백 루프 내에서 소스 바이어스가 점점 더 증폭되는 것을 확인했습니다. 피드백 루프에서의 소스 바이어스 증폭을 막기 위한 디바이싱 방법을 통해 모델의 예측 중립성을 유지할 수 있음을 확인했습니다. 이러한 방법은 AIGC 비율에 관계없이 소스 바이어스를 수용 가능한 한계 내에서 안정화시킬 수 있었습니다.



### Yuan 2.0-M32: Mixture of Experts with Attention Router (https://arxiv.org/abs/2405.17976)
Comments:
          14 pages,3 figures, 7 tables

- **What's New**: Yuan 2.0-M32 모델은 Yuan-2.0 2B와 유사한 기본 아키텍처를 가지고 있으며, 32개의 전문가(Experts) 중 2개의 전문가가 활성화되는 mixture-of-experts 아키텍처를 사용합니다. 더욱 효율적인 전문가 선택을 위해 새로운 라우터(Router) 네트워크인 Attention Router가 도입되었습니다. 이는 전통적인 라우터 네트워크에 비해 정확도를 3.8% 향상시킵니다.

- **Technical Details**: Yuan 2.0-M32는 2000B 토큰으로 처음부터 훈련되었으며, 동일한 파라미터 규모의 밀집 모델(Dense Model) 대비 훈련 계산 소비량이 9.25%에 불과합니다. 활성화된 3.7B 파라미터와 40B의 총 파라미터 수를 가지고 있으며, 토큰당 7.4 GFlops의 전진 전파 계산을 수행합니다. 이는 Llama3-70B의 1/19 수준입니다.

- **Performance Highlights**: Yuan 2.0-M32는 코딩, 수학 및 다양한 전문 영역에서 경쟁력 있는 성능을 보여줍니다. 특히 MATH 및 ARC-Challenge 벤치마크에서 각각 55.89와 95.8의 정확도로 Llama3-70B를 능가합니다. 모델과 소스 코드는 Github에서 사용할 수 있습니다.



### The Evolution of Multimodal Model Architectures (https://arxiv.org/abs/2405.17927)
Comments:
          30 pages, 6 tables, 7 figures

- **What's New**: 이 논문은 현대의 멀티모달 (multimodal) 모델 아키텍처 패턴을 네 가지로 분류하고, 이를 통해 멀티모달 도메인 내의 발전을 체계적으로 모니터링할 수 있습니다. 특히, 네 가지 특정 아키텍처 유형을 식별하고 설명하며, 멀티모달 입력의 통합 방법에 따라 구분합니다.

- **Technical Details**: 논문에서는 네 가지 아키텍처 유형을 설명합니다: Type-A, Type-B, Type-C, Type-D. Type-A와 Type-B는 모델 내부 층에서 멀티모달 입력을 융합하고, Type-C와 Type-D는 입력 단계에서 초기 융합을 수행합니다. Type-A는 표준 크로스-어텐션(cross-attention)을 사용하고, Type-B는 모달리티 융합을 위한 맞춤형 레이어를 사용합니다. Type-C는 모달리티-특정 인코더를 사용하고, Type-D는 토크나이저를 활용합니다.

- **Performance Highlights**: 특히, Type-C와 Type-D 아키텍처는 'any-to-any' 멀티모달 모델 개발에서 선호되고 있으며, Type-C는 토크나이징 기법을 사용하지 않는 비-토크나이징 아키텍처로 Type-D의 대안으로 떠오르고 있습니다. 논문은 각 아키텍처 유형의 장단점을 데이터 및 컴퓨팅 요구 사항, 아키텍처 복잡성, 확장성, 모달리티 추가의 용이성, 훈련 목표 등을 기준으로 강조합니다.



### Boosting Protein Language Models with Negative Sample Mining (https://arxiv.org/abs/2405.17902)
Comments:
          17 pages, 4 figures

- **What's New**: 본 논문은 단백질 표현 학습(protein representation learning)에서 대형 언어 모델(LLM)을 개선하기 위한 혁신적 방법론을 소개합니다. 특히, 공진화 지식(co-evolution knowledge)에 대한 과도한 의존성을 조정하여 네트워크가 서로 다른 카테고리에서 추출한 단백질 쌍의 부정적 샘플(negative samples)에서 가치 있는 통찰을 추출하도록 훈련합니다. 이를 통해 주의 점수 공간(attention score space)에서 트랜스포머 기반 모델의 성능을 향상시킵니다.

- **Technical Details**: 이 방법론은 부정적 샘플 마이닝(negative mining)을 사용해 PLM(pre-trained language models)의 공진화 지식에 대한 편향을 줄이고, 단백질 기능 예측, 단백질 서열 설계, 단백질 접힘 구조 예측 등 다양한 다운스트림 작업에서 더 나은 성능을 달성합니다. NM-Transformer라고 명명된 이 프레임워크는 PLM이 다른 라벨을 가진 단백질 쌍에서 주의 매트릭스를 일정하게 유지하도록 하여 주의 공간에서의 정렬을 감소시킵니다.

- **Performance Highlights**: 이 새로운 부정적 샘플 마이닝 접근법은 여러 다운스트림 작업에서 PLM 성능을 향상시키며, 작은 규모의 PLM과 대규모 PLM 간의 성능 차이를 줄일 수 있습니다. 특히, 단백질-단백질 상호작용(PPI) 작업에서 실제 상호작용이 발생하는 결합 경계에서 아미노산 잔여물의 정렬을 강조함으로써 그 유효성을 실험적으로 입증했습니다.



### SLMRec: Empowering Small Language Models for Sequential Recommendation (https://arxiv.org/abs/2405.17890)
- **What's New**: 이 논문에서는 대형 언어 모델 (Large Language Models, LLMs)이 연속 추천 시스템(Sequential Recommendation, SR)에서 얼마나 필요하고 효과적인지에 대한 연구를 수행합니다. 연구 결과, LLM의 중간 계층이 상당 부분 불필요하다는 것을 발견하고, 이를 기반으로 작은 언어 모델(Small Language Models, SLM)을 적용한 SR 모델(SLMRec)을 제안합니다.

- **Technical Details**: 연구팀은 대형 언어 모델(LLM)의 깊이에 대한 영향을 대규모 산업 데이터셋에서 실험을 통해 조사했습니다. 그 결과, 대부분의 중간 계층이 불필요하다는 사실을 발견했습니다. 이를 바탕으로 '지식 증류(Knowledge Distillation)' 방식으로 작은 언어 모델(SLMRec)을 개발했습니다. SLMRec는 큰 모델의 성능을 유지하면서도 계산 효율성을 높이는 데 중점을 두고 있습니다. 또한, 이는 양자화(Quantization)와 프루닝(Pruning) 등의 효율성 기술과도 호환됩니다.

- **Performance Highlights**: 제안된 SLMRec 모델은 기존의 LLM 기반 추천 모델보다 적은 매개변수(13%)를 사용하면서도, 훈련 및 추론 시간에서 각각 최대 6.6배와 8.0배의 속도 향상을 이뤘습니다. SLMRec는 불필요한 계층을 제거하고, 여전히 높은 성능을 보여줌으로써 실질적인 SR 환경에 더 적합합니다.



### Seeing the Image: Prioritizing Visual Correlation by Contrastive Alignmen (https://arxiv.org/abs/2405.17871)
- **What's New**: CAL (Contrastive ALignment)은 Vision Language Models(VLMs)의 이미지와 텍스트 크로스 모달 정렬 방식을 개선하는 간단하지만 효과적인 방법론을 소개합니다. 기존 방법론은 모든 텍스트 토큰을 동등하게 취급했지만, CAL은 시각적 상관성에 기반하여 텍스트 토큰의 기여도를 다르게 부여합니다.

- **Technical Details**: CAL은 이미지 입력을 대조하여 각 텍스트 토큰의 예측 로그잇(logits)의 변화를 분석합니다. 이를 통해 시각적으로 상관성 높은 토큰에 우선순위를 두는 재가중 전략을 적용합니다. 실험에서는 다양한 해상도와 모델 크기의 VLMs에서 지속적으로 성능 향상을 보였습니다. 이 방법은 각 훈련 단계에서 추가적인 한 번의 보조 포워드 연산만 필요로 하며, 거의 추가적인 계산 부담을 주지 않습니다.

- **Performance Highlights**: CAL은 LLaVA-Next-13B 모델에서 VQADoc, VQAChart, COCO 및 TextCaps와 같은 여러 벤치마크에서 뛰어난 성능 향상을 보여주었습니다. 특히, LLaVA-Next-13B 모델에서는 VQADoc에서 1.7 ANLS, VQAChart에서 3.4 relaxed accuracy, COCO에서 2.2 CIDEr, TextCaps에서 6.3 CIDEr, RefCOCOg 검증/테스트 세트에서 0.6/0.7 IoU를 기록했습니다.



### I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models (https://arxiv.org/abs/2405.17849)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 추론을 가속화할 수 있는 사후 양자화(Post-training quantization, PTQ) 프레임워크인 I-LLM을 제안합니다. I-LLM은 기존 양자화 방식들이 가지고 있는 부동소수점 연산(FP) 의존성을 해결하여 에지 및 클라우드 디바이스에서도 효율적으로 작동할 수 있습니다.

- **Technical Details**: I-LLM은 완전 정수(integer-only) 사후 양자화 프레임워크로, 주요 구성 요소는 다음과 같습니다: (1) 채널 간 변동을 부드럽게 조정하는 '완전 부드러운 블록 재구축(Fully-Smooth Block-Reconstruction, FSBR)' 기술, (2) 동적 정수 행렬 곱셈을 가능하게 하는 '동적 정수-전용 MatMul(Dynamic Integer-only MatMul, DI-MatMul)', (3) 비선형 연산을 효율적으로 처리하면서 정확성을 유지하는 'DI-ClippedSoftmax', 'DI-Exp', 및 'DI-Normalization' 설계.

- **Performance Highlights**: I-LLM은 부동소수점(Floating Point, FP) 성능과 거의 동일한 정확성을 달성하며, 비정수 양자화 방법들을 능가합니다. 예를 들어, W4A4 상태에서도 정확도의 손실이 거의 없이 작동할 수 있습니다. 이번 연구는 정수-전용 양자화와 대형 언어 모델 사이의 격차를 좁히는 중요한 진전을 이루었습니다.



### Exploring Activation Patterns of Parameters in Language Models (https://arxiv.org/abs/2405.17799)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)의 내부 표현을 설명하기 위해, 모델 파라미터의 활성화 수준을 평가할 수 있는 그래디언트 기반 측정 지표를 제안합니다. 이를 통해 동일한 도메인의 입력에서는 얕은 층의 파라미터가 밀집 활성화되지만, 깊은 층의 파라미터는 드문드문 활성화된다는 점을 발견했습니다. 또한, 다른 도메인의 입력에서는 얕은 층의 파라미터 활성화 유사도가 깊은 층보다 높습니다. 마지막으로, 깊은 층에서는 활성화된 파라미터의 분포 유사성이 경험적 데이터 관련성에 비례함을 발견했습니다.

- **Technical Details**: 본 연구는 네트워크 가지치기(network pruning) 방법을 활용해 파라미터 영향력을 평가했습니다. 모델의 출력에 미치는 영향을 평가하기 위해 Taylor 전개의 1차 항을 사용해 파라미터 영향력을 측정했습니다. 두 입력의 코사인 유사도를 비교하여 파라미터 활성화 패턴을 분석했습니다. 또한, 결과를 확인하기 위해 세 가지 실험을 설계했습니다: 층별 가지치기 비율 설정, 교정 집합 기반 모델 가지치기, 문장 의미 일관성 평가 실험을 수행했습니다.

- **Performance Highlights**: 제안된 가지치기 방법은 기존 방법보다 우수한 성능을 보였습니다. 교정 집합 기반 가지치기 방법은 교정 과제와 관련된 작업에서 더 우수한 성능을 보이며, LLMDcos 측정값은 의미 유사성과 관련이 있음을 확인했습니다. 우리의 연구는 LLM의 파라미터 활성화 행동에 대한 통찰을 제공하며, 향후 실용적인 응용 프로그램에 영감을 줄 수 있기를 희망합니다.



### Linguistic Collapse: Neural Collapse in (Large) Language Models (https://arxiv.org/abs/2405.17767)
Comments:
          29 pages, 27 figures

- **What's New**: 이번 연구에서는 **Neural Collapse (NC)**가 언어 모델에 어떻게 발현되는지를 실험적으로 조사하였습니다. 특히, 언어 모델이 클래식 분류 조건을 충족하지 않는 환경에서, NC 현상의 확장을 중심으로 연구가 이루어졌습니다. 이를 통해 자연 언어 처리에서의 모델 일반화 능력과 NC 사이의 관계를 탐구하고자 하였습니다.

- **Technical Details**: Neural Collapse는 분류 작업에서 최종 레이어 표현들이 클래스 평균으로 수렴하는 현상으로, 클래스 간의 균일성과 각형적 배치가 나타나는 특성을 가집니다. 이 연구에서는 **Causal Language Models (CLMs)**을 대상으로 모델의 크기와 훈련을 확장하면서 이러한 특성들이 어떻게 나타나는지 분석하였습니다. 특히, **Transformer 기반의 CLMs**를 다양한 모델 폭, 깊이, 훈련 에폭에서 실험하였습니다.

- **Performance Highlights**: 모델 크기와 훈련이 확장됨에 따라 NC 특성이 뚜렷이 나타났으며, 이는 모델의 일반화 성능과 직결되었습니다. 특히, 클래스 내 변동성이 감소하고, 초구체적 균일성 및 클래스 평균에의 수렴도가 향상되었습니다. 이러한 결과는 언어 모델에서도 NC 현상이 나타날 수 있음을 시사하며, 이에 대한 추가 연구의 필요성을 강조합니다.



### Does Geo-co-location Matter? A Case Study of Public Health Conversations during COVID-19 (https://arxiv.org/abs/2405.17710)
- **What's New**: 이 연구는 코로나19 기간 동안 공중보건 전문가(PHE)와 일반 대중 사이의 지리적 공동 위치(geo-co-location)가 트위터(X) 상의 소셜 미디어 참여도에 미치는 영향을 분석했습니다. 코로나19 관련 정보 전달과 참여를 높이기 위한 공중보건 전문가의 전략을 이해하는데 중요한 새로운 통찰을 제공합니다.

- **Technical Details**: 연구팀은 2020년 1월부터 2021년 11월까지 거의 500명의 공중보건 전문가가 작성한 19,000개 이상의 트윗과 350,000명의 참가자가 작성한 약 80만 개의 응답 트윗을 분석했습니다. Carmen 도구를 사용하여 트윗 및 참가자의 지리적 위치를 분석하고, LIWC를 통해 심리적 과정과 자기 공개 수준을 파악했습니다. COVID-19 주제를 분류하기 위해 머신러닝 모델을 사용하고, 공중보건 전문가의 직업 프로필을 결정하기 위해 대규모 언어 모델(Large Language Models, LLM)을 사용했습니다.

- **Performance Highlights**: 통계 테스트 결과, 공중보건 전문가와 일반 대중이 동일한 지역에 있는 경우(geo-co-located) 참여율이 현저히 높았으며, 감정과 개인 경험을 공유하는 트윗에서는 이러한 경향이 더욱 두드러졌습니다. 반면, 백신 관련 대화에서는 비(非) 지역 공동 위치의 참여가 더 높았습니다. 학술 및 의료 전문직 출신의 공중보건 전문가가 시작한 대화에서는 지역 공동 위치가 더 큰 영향을 미쳤습니다.



### Mechanistic Interpretability of Binary and Ternary Transformers (https://arxiv.org/abs/2405.17703)
- **What's New**: 최근 연구들은 메모리를 크게 줄이고 정확도를 유지하면서 대형 언어 모델(LLMs)에서 추론 속도를 개선할 수 있는 이진(binary) 및 삼진(ternary) 트랜스포머 네트워크를 제안했습니다. 본 연구에서는 기계적 해석 가능성(mechanistic interpretability) 기법을 적용하여 이러한 네트워크가 정밀 네트워크와 비교하여 유사한 알고리즘을 학습하는지 탐구했습니다.

- **Technical Details**: 본 연구는 기계적 해석 가능성 기법을 사용하여 이진 및 삼진 네트워크가 정밀 네트워크와 유사한 알고리즘을 학습하는지 비교했습니다. 주요 연구 문제는 모듈러 덧셈(modular addition)으로 선택되었으며, 기본적인 트랜스포머 구조를 사용했습니다. 특히, 실험 설정은 주요 구성 요소를 제거한 1층 트랜스포머로 구성되었습니다. 데이터 학습은 AdamW 옵티마이저와 binarization을 사용했으며, 비닝 및 삼진 네트워크는 해당하는 이진 및 삼진 선형 계층으로 구성되었습니다.

- **Performance Highlights**: 본 연구에서 명확한 결과는 이진 및 삼진 네트워크가 정밀 네트워크와 유사한 주기성을 보이며, 경험된 '그로킹(grokking)' 현상이 매우 유사하다는 점입니다. 그로킹 현상은 주요 구조 유사성과 주기성을 나타내었으며, 이는 이진 및 삼진 네트워크가 정밀 네트워크와 같은 알고리즘을 학습할 가능성을 시사합니다. 그러나, 이진 및 삼진 네트워크는 오히려 해석가능성이 낮은 것으로 평가되었습니다.



### Generative Query Reformulation Using Ensemble Prompting, Document Fusion, and Relevance Feedback (https://arxiv.org/abs/2405.17658)
Comments:
          Extended Work of GenQREnsemble: Zero-Shot LLM Ensemble Prompting for Generative Query Reformulation, Dhole and Agichtein, ECIR 2024. arXiv admin note: text overlap with arXiv:2404.03746

- **What's New**: 최근 검색 질의(queries)의 재구성(Query Reformulation, QR)을 위해 제안된 새로운 접근 방식들은 큰 관심을 받고 있습니다. 이번 연구는 기존의 제로샷(zero-shot) QR 접근법을 개선하고, 대규모 언어 모델(Large Language Models)을 활용하여 검색 성능을 향상시킬 수 있는 두 가지 앙상블 기반 프롬프팅(ensemble prompting) 기법인 'GenQREnsemble'과 'GenQRFusion'을 제안합니다. 이는 제로샷 명령어의 다양한 패러프레이즈(paraphrases)를 활용하여 여러 세트의 키워드를 생성하고, 사용자의 의도에 더 잘 맞는 검색 결과를 제공합니다.

- **Technical Details**: 이 논문은 검색 전(pre-retrieval)과 검색 후(post-retrieval) 설정에서 QR을 개선하기 위해 두 가지 앙상블 기반 프롬프팅 기법을 도입합니다. 'GenQREnsemble'과 'GenQRFusion'은 패러프레이즈된 명령어를 사용하여 다양한 키워드를 생성하며, 이를 통해 보다 효과적인 검색 쿼리 재구성을 이룹니다. 추가적으로, 검색 후 설정에서는 'GenQREnsemble-RF'와 'GenQRFusion-RF'을 도입하여 인간 검색자 또는 'critic' LLM의 피드백을 반영합니다. 이 접근법을 통해 검색 전 설정에서 최대 18%, 검색 후 설정에서 최대 9%의 nDCG@10 성능 향상을 달성했습니다.

- **Performance Highlights**: 제안된 방법들은 최신의 검색 벤치마크에서 실험되었으며, 기존의 최신 결과(State Of The Art)를 능가하는 성능을 보였습니다. 특히, 검색 전 설정에서 nDCG@10 지표의 최대 18% 향상을, 검색 후 설정에서도 최대 9%의 성능 향상을 이루었습니다. 이는 피드백 문서의 수, 명령어의 수 및 도메인 특화 명령어, 유창한 재구성의 생성 등 다양한 요소들이 성능에 미치는 영향을 분석하는 결과를 통해 입증되었습니다.



### A Framework for Multi-modal Learning: Jointly Modeling Inter- & Intra-Modality Dependencies (https://arxiv.org/abs/2405.17613)
- **What's New**: 이 연구에서는 기존의 멀티모달(다중 모달리티) 학습 접근 방식을 개선한 inter- & intra-modality modeling (I2M2) 프레임워크를 제안합니다. 기존 연구들은 모달리티 간 상호작용(inter-modality dependencies)이나 단일 모달리티 내부 상호작용(intra-modality dependencies) 중 하나에만 집중한 반면, I2M2는 두 가지 상호작용을 모두 통합하여 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: I2M2는 각 모달리티마다 개별 분류기를 구축하여 intra-modality dependencies를 포착하고, 출력 레이블과 모달리티 간 상호작용(inter-modality interactions) 간의 의존성을 포착하는 분류기를 결합하여 두 의존성을 모두 모델링합니다. 이 접근 방식은 모달리티와 레이블 간의 상호작용을 선택 변수(selection variable)을 통해 설명하고, 이를 통해 다양한 조건 하에서도 효율적이고 적응 가능한 모델을 만듭니다.

- **Performance Highlights**: 의료 데이터셋(예: MRI 검사 및 MIMIC-III 데이터셋)과 비전-언어(vision-and-language) 데이터셋(예: VQA 및 NLVR2)에서 I2M2를 사용해 평가한 결과, 기존 방법보다 우수한 성능을 보였습니다. 특히, intra-modality dependencies가 중요한 fastMRI 데이터셋과 inter-modality dependencies가 중요한 NLVR2 데이터셋 모두에서 I2M2의 강점을 입증했습니다. 두 의존성이 모두 중요한 AV-MNIST, MIMIC-III 및 VQA 데이터셋에서도 뛰어난 성과를 나타냈습니다.



### LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters (https://arxiv.org/abs/2405.17604)
- **What's New**: 최근 언어 모델의 확장 추세에 따라, 매개변수 효율적인 튜닝(PEFT) 방법의 수요가 증가하고 있습니다. 이 논문에서는 이러한 문제를 해결하기 위해 LoRA-XS(Low-Rank Adaptation with eXtremely Small number of parameters)를 소개합니다. 이는 Singular Value Decomposition(SVD)을 활용하여 매개변수 효율적인 파인 튜닝을 제공합니다.

- **Technical Details**: LoRA-XS는 원래의 가중치 행렬을 SVD로 분해하여, 고정된 LoRA 행렬 사이에 작은 r x r 가중치 행렬을 도입합니다. 이 가중치 행렬들만을 학습함으로써 모델 차원에 독립적으로 작동할 수 있으며, 특히 더 큰 모델에 대해 더 매개변수 효율적인 파인 튜닝을 가능하게 합니다.

- **Performance Highlights**: LoRA-XS는 LoRA와 비교해 7B 모델에서 학습 가능한 매개변수를 100배 이상 줄이는 놀라운 성과를 보여줍니다. 또한 GLUE, GSM8k, MATH 벤치마크를 포함한 다양한 스케일에서 최근의 최첨단 접근법인 VeRA보다 매개변수 효율성 면에서 더 우수하면서도 경쟁력 있는 성능을 유지합니다.



### RAGSys: Item-Cold-Start Recommender as RAG System (https://arxiv.org/abs/2405.17587)
- **What's New**: Large Language Models(LLM)의 실제 응용 가능성을 탐구하고, 특히 도메인별 요구에 맞춘 시연 검색 시스템(Demonstration Retrieval System)의 성질을 연구합니다. 이 논문은 시연 검색 시스템(ICL Retrieval)을 아이템 콜드 스타트 추천 시스템(item-cold-start recommender systems)으로 간주하고, 정보 이득을 극대화하는 것에 초점을 맞추는 새로운 평가 방법을 제안합니다.

- **Technical Details**: 논문에서는 In-Context Learning(ICL)과 Retrieval-Augmented Generation(RAG)의 개념을 결합한 새로운 접근법을 제시합니다. ICL은 '메타 학습(meta-learning)' 능력을 활용하여 LLM의 컨텍스트 내에서 올바른 답변을 생성하는 방법론입니다. 시연 검색 시스템에서 다루는 중요한 특성인 다양성과 품질 편향을 강조하며, 이론적 배경과 실험을 통해 이를 증명합니다. 또한, 소수의 잘 선택된 시연 예제(top-k examples)만으로도 큰 성과를 낼 수 있음을 보여줍니다.

- **Performance Highlights**: 논문의 실험 결과, LLM이 올바른 답변을 생성하는 능력에 다양성과 품질 편향이 중요한 영향을 미침을 확인했습니다. 저자들은 ICL 검색 알고리즘이 실제 환경에서도 매우 효과적일 수 있음을 강조하며, 추천 시스템 기술이 의미론적 검색 엔진보다 더 나은 해결책이 될 수 있음을 보여줍니다.



### BIOSCAN-CLIP: Bridging Vision and Genomics for Biodiversity Monitoring at Sca (https://arxiv.org/abs/2405.17537)
Comments:
          16 pages with 9 figures

- **What's New**: 이 논문에서는 생물다양성을 측정하기 위한 새로운 접근법을 소개합니다. 기존 연구와는 달리, 이번 연구는 이미지, DNA 바코드, 텍스트 데이터를 결합하여 CLIP 스타일의 대조 학습(contrastive learning)을 사용해 통합 임베딩 공간(unified embedding space)을 구축했습니다. 이는 학습 시 DNA 정보를 활용하고 추론 시 이미지 정보만으로도 정확한 분류를 가능하게 합니다.

- **Technical Details**: BIOSCAN-CLIP는 대조 학습을 사용하여 생물학적 이미지, 텍스트 분류 레이블, DNA 바코드를 같은 잠재 공간(latent space)에 매핑합니다. 이를 통해, 종 간의 섬세한 시각적 차이를 학습할 수 있으며, DNA와 이미지 데이터를 결합하여 학습의 다양한 가능성을 엽니다. 기존과 달리 종 간의 진화적 관계와 유사점을 더 잘 포착할 수 있습니다. 모델은 사전 학습된 인코더(pretrained encoder)와 LoRA 미세 조정(finetuning)을 결합하여 개발되었습니다.

- **Performance Highlights**: 이 새로운 접근법은 제로샷 학습(zero-shot learning) 작업에서 기존 단일 모달리티 접근법보다 11% 이상 정확도가 향상되었습니다. 이는 생물다양성 연구에 있어 이 방법의 효과성을 잘 보여줍니다.



### Predicting Rental Price of Lane Houses in Shanghai with Machine Learning Methods and Large Language Models (https://arxiv.org/abs/2405.17505)
Comments:
          13 pages, 11 figures, 39 references

- **What's New**: 이번 연구는 전통적인 머신 러닝 방법과 대형 언어 모델(Large Language Model, LLM)인 ChatGPT를 활용하여 상하이의 골목 주택 임대 가격을 예측하는 방법을 비교한 것입니다. 전통적인 방법 중 랜덤 포레스트(Random Forest, RF)가 가장 우수한 성능을 보였으나, ChatGPT의 성능이 특히 10샷(10-shot) 시나리오에서 더 나은 예측력을 보였습니다.

- **Technical Details**: 사용된 전통적인 머신 러닝 방법으로는 다중 선형 회귀(Multiple Linear Regression, MLR), 릿지 회귀(Ridge Regression, RR), 라쏘 회귀(Lasso Regression, LR), 결정 트리(Decision Tree, DT), 및 랜덤 포레스트(Random Forest, RF)가 있습니다. 이들 모델들은 약 2,609건의 2021년 상하이 골목 주택 임대 거래 데이터를 활용하여 비교되었습니다. LLM 접근 방식은 ChatGPT를 0샷, 1샷, 5샷, 10샷 시나리오에서 사용하여 예측을 하였습니다.

- **Performance Highlights**: RF 모델은 전통적인 방법 중 가장 높은 예측력을 보였으며, R-Squared 값은 최고 수준이었습니다. 그러나 ChatGPT를 사용한 LLM 접근 방식이 특히 10샷 시나리오에서 전통적인 방법보다 더 나은 R-Squared 값을 기록했습니다. 평가에는 평균 제곱 오차(Mean Squared Error, MSE), 평균 절대 오차(Mean Absolute Error, MAE), 및 결정 계수(R-Squared)가 사용되었습니다.



### Code Repair with LLMs gives an Exploration-Exploitation Tradeoff (https://arxiv.org/abs/2405.17503)
- **What's New**: 새로운 연구에서는 대규모 언어 모델(LLMs)을 이용해 복잡한 프로그램을 적층적으로 개선하는 '정제' 기법을 소개합니다. 특히, 테스트 케이스와 후보 프로그램을 바탕으로 실패한 테스트 케이스를 모델에 제공해 프로그램을 개선하는 방법을 제안합니다. 이 연구는 기존의 단순 탐욕적(greedy)이나 너비 우선(breadth-first) 전략 대신, '탐색-활용(explore-exploit)' 트레이드오프를 밝혀내고 이를 개선하기 위한 방법을 제시했습니다.

- **Technical Details**: 연구진은 이러한 '탐색-활용' 트레이드오프를 'arm-acquiring bandit 문제'로 프레임하고 이를 해결하기 위해 톰슨 샘플링(Thompson Sampling) 기법을 사용했습니다. 이 새로운 LLM 기반 프로그램 합성 알고리즘은 여러 도메인에 광범위하게 적용될 수 있습니다. 예를 들어 루프 불변식(loop invariant) 합성, 시각적 추론 퍼즐, 그리고 대회 프로그래밍 문제에까지 적용할 수 있습니다.

- **Performance Highlights**: 우리의 새로운 방법은 언어 모델 호출(call)의 수를 줄이면서 더 많은 문제를 해결할 수 있음을 발견했습니다. 이는 기존의 접근 방식보다 효율성이 높다는 것을 의미합니다.



### How Culturally Aware are Vision-Language Models? (https://arxiv.org/abs/2405.17475)
- **What's New**: 이번 연구는 네 가지의 인기 있는 비전-언어 모델(GPT-4V, Gemini Pro Vision, LLaVA, OpenFlamingo)을 비교하여 신화, 민속 춤, 문화적 기호와 같은 문화적으로 특정한 이미지를 정확하게 인식하고 캡션을 생성하는 성능을 분석합니다. 또한, 이미지 캡션의 문화적 인식을 측정하는 새로운 평가 지표인 '문화 인식 점수(Cultural Awareness Score, CAS)'를 제안합니다.

- **Technical Details**: 연구는 다문화적 콘텐츠를 포함한 이미지를 다루는 MOSAIC-1.5k 데이터셋을 구성하고, 각 이미지에 대한 자세한 문화적 배경을 주석으로 추가했습니다. 또한, 새로운 평가 지표인 문화 인식 점수(CAS)를 통해 모델이 이미지 속 문화적 요소를 얼마나 잘 이해하고 있는지 측정합니다.

- **Performance Highlights**: 연구에서 제안된 CAS를 사용하여 각 비전-언어 모델의 성능을 평가한 결과, 모델들은 문화적 차이를 인식하고 이를 이미지 캡션에 반영하는 능력을 보여주었습니다. 특히, 일부 모델은 높은 문화 감수성을 갖춘 정확한 캡션을 생성하는 데 성공했습니다. 이를 통해 AI 시스템이 더 포용적이고 문화적으로 배려 있는 디지털 생태계를 조성할 수 있는 가능성을 확인했습니다.



### Athena: Efficient Block-Wise Post-Training Quantization for Large Language Models Using Second-Order Matrix Derivative Information (https://arxiv.org/abs/2405.17470)
- **What's New**: 이번 논문에서는 Athena라는 새로운 알고리즘을 제안하여 대형 언어 모델(LLMs)에 대한 효율적인 블록 단위 사후 훈련 양자화를 수행합니다. 기존의 압축 방법은 파라미터의 불균형 분포를 고려하지 않아 정확도가 크게 감소하는 문제가 있었습니다. Athena는 손실 곡면의 커브 정보를 이용하여 양자화 프로세스를 안내하며, 파라미터를 열이나 행 단위로 그룹화하고 반복적으로 최적화하여 모델 파라미터와 Hessian 행렬을 업데이트합니다.

- **Technical Details**: Athena는 손실 함수의 이차 미분 정보를 활용합니다. 모델의 파라미터를 열이나 행 단위로 그룹화한 후 각 그룹에 대해 반복적인 양자화 과정을 수행합니다. 이 과정에서 각 파라미터가 모델 성능에 미치는 영향을 평가하고, 최적화 문제를 풀어 Lagrangian 함수를 구성하며, 파라미터와 Hessian 행렬을 업데이트합니다.

- **Performance Highlights**: Athena는 높은 정확도를 유지하면서도 모델의 압축을 크게 향상시킵니다. 이를 통해 다양한 환경에서 LLMs를 실용적으로 배포할 수 있는 가능성을 제공합니다.



### Integrating Medical Imaging and Clinical Reports Using Multimodal Deep Learning for Advanced Disease Analysis (https://arxiv.org/abs/2405.17459)
- **What's New**: 이 논문에서는 의료 이미지와 임상 보고서로부터 이질적인 정보를 깊게 통합하는 혁신적인 다중 모달 딥러닝 모델을 제안합니다.

- **Technical Details**: 먼저, 의료 이미지의 경우 고차원 특징을 추출하고 초점 세부 사항, 텍스처 및 공간 분포와 같은 핵심 시각 정보를 포착하기 위해 Convolutional Neural Networks(컨벌루션 신경망)을 사용합니다. 그 다음, 임상 보고서 텍스트의 경우 양방향 Long Short-Term Memory Network(양방향 장단기 기억 네트워크)와 주의 메커니즘을 결합하여 깊은 의미적 이해를 제공하고 질병과 관련된 주요 문장을 정확하게 포착합니다. 두 특징은 설계된 다중 모달 융합층을 통해 효과적으로 상호 작용하고 통합되어 이미지와 텍스트의 공동 표현 학습을 실현합니다.

- **Performance Highlights**: 경험적 연구에서 다양한 질병을 포괄하는 대규모 의료 이미지 데이터베이스를 선택하고 해당 임상 보고서와 결합하여 모델 훈련 및 검증을 수행했습니다. 제안된 다중 모달 딥러닝 모델은 질병 분류, 병변 위치 지정 및 임상 설명 생성 분야에서 실험 결과로 상당한 우월성을 입증했습니다.



### When Large Language Models Meet Optical Networks: Paving the Way for Automation (https://arxiv.org/abs/2405.17441)
- **What's New**: 본 연구는 GPT와 같은 대형 언어 모델(Large Language Models, LLM) 기술을 이용하여 광 네트워크(optical networks)에서의 지능형 제어(intelligent control)와 효율적 상호작용을 구현하는 새로운 프레임워크를 제안합니다. 이를 통해 물리적 계층에서의 제어와 애플리케이션 계층 간의 상호작용을 LLM 기반 에이전트(AI-Agent)가 수행하게 됩니다.

- **Technical Details**: 제안된 프레임워크는 LLM 에이전트를 활용하여 외부 도구를 활용하고, 광 네트워크에 특화된 도메인 지식 라이브러리(domain knowledge library)에서 정보를 추출합니다. LLM의 성능을 개선하고 복잡한 작업을 수행할 수 있도록, 프롬프트 공학(prompt engineering), 도메인 지식 라이브러리 구축, 복잡한 작업 구현의 세부 사항을 자세히 설명합니다.

- **Performance Highlights**: 제안된 프레임워크는 네트워크 알람 분석(network alarm analysis) 및 네트워크 성능 최적화(network performance optimization)와 같은 두 가지 전형적인 작업에서 검증되었습니다. 2,400개의 테스트 상황에서 높은 응답 정확도와 의미적 유사도(semtatic similarities)를 보임으로써 LLM이 광 네트워크에서 큰 잠재력을 가지고 있음을 입증했습니다.



### CataLM: Empowering Catalyst Design Through Large Language Models (https://arxiv.org/abs/2405.17440)
- **What's New**: CataLM은 electrocatalytic materials(전기촉매 물질) 도메인에 특화된 최초의 대형 언어 모델(LLM)로, 촉매 지식 검색 및 설계를 위한 인간-AI 협업을 촉진할 수 있는 잠재력을 보여주고 있습니다. 이를 통해 새로운 촉매 발굴 및 개발에 혁신적인 접근을 제시합니다.

- **Technical Details**: CataLM의 학습은 두 단계로 구성됩니다. 첫 번째는 'Domain Pre-training' 단계로, 선택된 고품질 저널의 오픈 액세스 촉매 논문에서 발췌한 텍스트 코퍼스를 사용하여 화학 지식을 습득합니다. 두 번째는 'Instruction Tuning' 단계로, 전문가가 주석을 단 코퍼스와 대형 언어 모델이 생성한 코퍼스를 사용하여 모델이 하위 작업에 대한 요구사항을 더 잘 이해할 수 있도록 합니다. 추가적으로, LLM의 추론 능력을 향상시키기 위해 벡터 데이터베이스와 Sci-BERT를 임베딩 모델로 사용하였습니다.

- **Performance Highlights**: CataLM은 두 가지 작업, 즉 엔티티 추출 작업과 제어 방법 추천 작업을 통해 검증되었습니다. 구축된 지식 베이스를 사용한 검증 외에도, 도메인 전문가를 초청하여 CataLM의 답변을 평가한 결과, 촉매 지식 검색 및 설계에서 CataLM의 일반화 능력이 입증되었습니다.




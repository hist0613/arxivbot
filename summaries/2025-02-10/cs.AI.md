New uploads on arXiv(cs.CL)

### NoLiMa: Long-Context Evaluation Beyond Literal Matching (https://arxiv.org/abs/2502.05167)
- **What's New**: 최근 대형 언어 모델(LLMs)은 128K에서 1M 토큰까지 긴 컨텍스트를 지원하는 놀라운 발전을 했습니다. 이들의 성능을 평가하기 위해 needle-in-a-haystack(NIAH) 테스트가 인기 있으며, 관련 정보(needle)를 비관련 컨텍스트(haystack) 내에서 검색하는 방법을 포함합니다. 그러나 기존의 접근 방법은 모형이 문자적 일치를 이용하여 작업을 단순화할 수 있게 하였습니다. 이를 해결하기 위해, 우리는 NoLiMa라는 벤치마크를 소개하여 질문과 needle 간의 최소한의 어휘적(overlap) 중첩을 요구하는 설계를 특징으로 합니다.

- **Technical Details**: NoLiMa는 모델이 haystack 내에서 needle을 찾기 위해 잠재적 연관성을 추론해야 하는 새로운 벤치마크입니다. 우리는 12개의 인기 있는 LLM을 평가했으며, 이들 모두는 최소 128K 토큰을 지원한다고 주장합니다. 그러나 성능이 단기 컨텍스트(<1K)에서는 좋은 성능을 보였지만, 컨텍스트 길이가 증가함에 따라 급격히 감소하는 경향이 있음을 발견했습니다. 예를 들어, 32K에서 10개 모델은 단기 성능 기준의 50% 아래로 떨어졌습니다.

- **Performance Highlights**: 모델들의 성능은 컨텍스트 길이에 따라 현저히 감소하며, 2K에서 8K까지의 토큰에서도 상당한 성능 저하가 관찰됩니다. NoLiMa의 분석을 통해 우리는 잠재 추론 단계(latent hops)와 사실 진술 내 요소의 순서가 작업 성능에 미치는 영향을 보여주었습니다. 특히, 문자적 매치를 갖지 않을 경우, 긴 컨텍스트는 주의 메커니즘을 압도하여 정보 검색을 어렵게 만든다는 것을 확인했습니다. 이러한 발견을 통해, NoLiMa는 긴 컨텍스트 벤치마크에서 문자적 매칭의 한계를 드러내고 모델의 잠재적 추론 평가를 위한 새로운 접근 방식을 소개합니다.



### DuoGuard: A Two-Player RL-Driven Framework for Multilingual LLM Guardrails (https://arxiv.org/abs/2502.05163)
Comments:
          24 pages, 9 figures, 5 tables

- **What's New**: 이 논문에서는 다국어 안전성 강화를 위한 혁신적인 두 플레이어 강화 학습(RL) 프레임워크를 제안합니다. 이는 생성모델과 가드레일 모델이 상호 작용하여 고품질의 합성 데이터(synthetic data)를 생성함으로써 다국어 모델 교육을 개선할 수 있도록 합니다. LLM(대형 언어 모델)의 안전성을 높이기 위해, 안전한 언어 모델링의 필요성을 강조하고 다국어 문제를 해결하기 위한 데이터 부족 문제를 해결하고자 합니다.

- **Technical Details**: 이 프레임워크는 생성기(generator)와 가드레일 분류기(classifier)가 상호 경쟁적으로 발전하면서 효율적인 합성 데이터 생성을 가능하게 합니다. 이 시스템은 자기 개선(self-improvement) 메커니즘을 통해, 분류기가 합성 데이터 생성 과정을 조정하여 자신의 교육을 최적화할 수 있도록 설계되었습니다. 또한, 두 플레이어 게임을 이론적으로 정식화하여 내쉬 균형(Nash equilibrium)으로 수렴함을 증명하고, 알고리즘의 안정성을 보장하기 위해 데이터 필터링과 자기 평가 기술을 적용합니다.

- **Performance Highlights**: DuoGuard라는 새로운 모델이 6개의 다국어 안전 기준문제를 통한 평가에서 최근 최고 성능을 보이는 모델들을 초월하여 일관된 성과 향상을 보여주었습니다. 본 모델은 유사한 규모의 모델들보다 평균적으로 20% 이상 향상된 성능을 기록하였으며, 더 큰 모델들과 비교했을 때도 약 10% 정도 성과 개선을 이루어냈습니다. 추가적으로, 생성된 합성 데이터는 더 큰 모델(1.5B)과 다양한 아키텍처(Llama-3.2-1B)에서도 효과적으로 일반화되어 우수한 성능을 발휘하게 됩니다.



### Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation (https://arxiv.org/abs/2502.05151)
Comments:
          Work in progress. Will be updated soon

- **What's New**: 최근 다중 모달 언어 모델(multimodal language models)의 출현으로, 과학 분야는 AI 기반 기술 혁신의 문턱에 서게 되었습니다. 새로운 AI 모델과 도구들이 제안되며, 연구자와 학자들이 보다 효과적이고 효율적으로 연구를 수행할 수 있는 가능성을 제시하고 있습니다. 연구 사이클의 여러 측면, 예를 들어 관련 문헌 검색, 연구 아이디어 생성, 실험 수행, 텍스트 기반 및 다중 모달 콘텐츠 생성(예: 과학적 그림 및 도표), AI 기반 자동 피어 리뷰(automatic peer review)에 대한 내용을 다루고 있습니다.

- **Technical Details**: 이 설문조사는 위에서 언급한 다섯 가지 측면에 대해 포괄적으로 다루며, 관련 데이터셋(datasets), 방법(methods) 및 결과(results)뿐 아니라 평가(evaluation), 한계 및 향후 연구의 범위도 안내합니다. 특히, 이러한 도구의 단점과 오용 가능성(예: 가짜 과학, 표절, 연구 진실성에 대한 해악)과 같은 윤리적 문제들이 강조됩니다. 이는 연구 과정의 근본적인 변화를 가져올 것을 약속하는 새로운 발전에 대한 깊은 통찰을 제공합니다.

- **Performance Highlights**: 설문조사는 신기술의 잠재력과 그로 인한 연구 프로세스의 변화에 주목하며, AI4Science 분야에서의 새로운 AI 기반 이니셔티브를 촉진할 수 있는 기초 자료가 될 것으로 기대됩니다. 본 연구는 신규 연구자들에게 참조가 되는 자료가 되기를 바라며, AI를 활용한 연구의 효율성을 증가시키기 위한 다양한 방안이 모색될 것입니다.



### CodeSCM: Causal Analysis for Multi-Modal Code Generation (https://arxiv.org/abs/2502.05150)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이 논문은 대규모 언어 모델을 활용하여 다중 모드 코드 생성 분석을 위한 새로운 구조적 인과 모델(CodeSCM)을 제안합니다. 제안된 모델은 다양한 입력 모드, 예를 들어 자연어, 코드 및 입출력 예제가 모델에게 미치는 인과적 영향을 측정할 수 있습니다. 또한, CodeSCM은 다중 모드 프롬프트의 코드 및 자연어 의미를 분리하기 위해 잠재 매개변수를 도입하여 모델의 기계적 학습을 정량화합니다.

- **Technical Details**: 구조적 인과 모델(SCM)의 개념을 바탕으로 하는 CodeSCM은 프롬프트의 각 구성 요소를 독립 변량으로 간주하여 코드 생성 성능에 미치는 인과적 효과를 분석합니다. 두 개의 잠재 매개 변수를 사용하여 입력 프롬프트의 코드 의미와 자연어 의미를 캡처하며, 이는 사용자가 코드 조각을 생성하기 위해 필요한 지식을 반영합니다. 또한, 인과 매개 분석(Causal Mediation Analysis)을 통해 직접 효과와 스푸리어스(spurious) 상관관계를 측정합니다.

- **Performance Highlights**: CodeSCM을 사용한 분석 결과, 자연어 지침 외에도 입출력 예제가 코드 생성에 중요한 영향을 미친다는 것을 발견하였습니다. 특히, 입력-출력 예제 쌍의 간단한 의미 보존 변환이 정확도를 크게 저하시키는 것으로 나타났습니다. 기존의 LLM(GPT-4T)에서 벤치마크 메모리화가 관찰되며, CodeLLaMa는 다른 프롬프트 모드를 보다 잘 정렬할 수 있음을 보여줍니다.



### GiesKaNe: Bridging Past and Present in Grammatical Theory and Practical Application (https://arxiv.org/abs/2502.05113)
- **What's New**: 이 논문은 GiesKaNe 프로젝트(Universität Giessen und Kassel의 신 고독일어 구문 기본 구조)에 대한 코퍼스 컴파일要求을 탐구합니다. GiesKaNe 프로젝트는 참조 코퍼스(reference corpus), 역사적 코퍼스(historical corpus), 그리고 구문적으로 깊이 주석이 달린 트리뱅크(treebank)라는 세 가지 중앙 특징으로 정의됩니다. 새로운 텍스트 선택 관점을 제시하며 기계 지원 텍스트 분류를 위한 혁신적인 방법을 소개합니다.

- **Technical Details**: 이 논문은 토큰화(tokenization), 정규화(normalization), 문장 정의(sentence definition), 태깅(tagging), 구문 분석(parsing), 그리고 주석자 간 합의(inter-annotator agreement)와 같은 기본적인 주제를 논의합니다. 또한 기존 주석에서 사실적 표준 주석(de facto standard annotations)을 도출하는 접근법을 소개하여 표준화와 혁신 간의 균형을 매개합니다. 프로젝트의 방법론적 복잡성은 인간 전문성과 기계 지원 프로세스 간의 상호작용으로 관리됩니다.

- **Performance Highlights**: GiesKaNe와 같은 야심찬 프로젝트는 기존의 연구 인프라를 이용하여 효과적으로 구현될 수 있다는 점을 보여줍니다. 특별한 주석 도구 없이도 간단한 스프레드시트(spreadsheet)의 전략적 활용을 기반으로 한 워크플로우를 제시합니다. 이는 연구 커뮤니티의 더 넓은 관심을 충족시키면서 프로젝트 내부 목표를 달성하는 방법을 보여줍니다.



### Flexible and Efficient Grammar-Constrained Decoding (https://arxiv.org/abs/2502.05111)
- **What's New**: 이번 연구에서는 문법 제약 디코딩 (Grammar-constrained decoding, GCD)의 새로운 접근 방식을 제안하며, 이를 통해 오프라인 전처리 속도가 기존보다 17.71배 빨라졌습니다. 특히, 우리 알고리즘은 현재 최고의 온라인 마스킹 효율성을 유지하면서 다양한 문법을 처리할 수 있도록 설계되었습니다. 제안하는 방법론은 LLM의 토큰 어휘와 문맥 자유 문법 (Context-Free Grammar, CFG) 단말의 결합 분석을 기반으로 합니다. 이를 통해 디코더가 효율적으로 유효한 LLM 토큰을 식별할 수 있습니다.

- **Technical Details**: 우리는 문법 제약이 있는 디코딩을 위한 알고리즘을 구축하기 위해, 먼저 정규 표현식을 사용하여 토큰을 정의하기 위해 렉서를 사용합니다. 이는 입력 문자열을 효율적으로 처리하고, 단어 수준에서 처리하는 대신 토큰 수준에서 문법 구조를 정의하는 것을 가능하게 합니다. GCD 알고리즘은 LLM의 서브워드 토큰과 단말 사이의 정렬 문제를 해결하며, 이로 인해 오프라인 전처리 비용을 줄이고 효율적인 온라인 토큰 마스킹을 가능하게 합니다. 새로운 도구 GreatGramma는 이러한 알고리즘을 구현하여 관련 GCD 접근법에 비해 속도를 월등히 향상시킵니다.

- **Performance Highlights**: GreatGramma는 기존 GCD 접근법 대비 평균 17.71배 빠른 오프라인 전처리 속도를 보여주며, 온라인 마스킹 효율성도 뛰어난 것으로 평가되었습니다. 연구에서는 기존 GCD 구현에 존재하는 신뢰성 오류도 발견되었으며, 이러한 문제들을 해결하여 단순한 모듈로 구성된 우아한 구현을 제공합니다. 이러한 성능 개선은 프로그램 합성 및 문법 프롬프팅과 같은 동적 문법 도메인에서 특히 유용하게 활용될 수 있습니다.



### ChallengeMe: An Adversarial Learning-enabled Text Summarization Framework (https://arxiv.org/abs/2502.05084)
- **What's New**: 이 논문은 ChallengeMe라는 새로운 적대적 학습 기반 프롬프트 프레임워크를 제안하여 텍스트 요약 과제에서의 성능을 향상시키고자 합니다. 이 프레임워크는 생성 프롬프트, 평가 프롬프트, 피드백 최적화라는 세 가지 연속된 해결책으로 구성되어 있습니다. 이 연구는 적대적 학습의 최적화 차원을 일곱 가지로 설정하고, 혼합 사례 연구를 통해 기존의 LLM들과 비교하여 더 정확하고 유창한 요약을 생성할 수 있음을 입증했습니다.

- **Technical Details**: ChallengeMe는 인간 인지 과정의 분류 및 비교 메커니즘에서 영감을 받아 설계된 적대적 프롬프트 학습 프레임워크입니다. 이 프레임워크는 입력 프롬프트, 적대적 프롬프트, 피드백 최적화 전략을 포함한 세 가지 모듈로 구성되어 있습니다. 또한 모델이 요구되는 목표와 제약 조건을 준수하도록 적절한 프롬프트를 설계하여 모델이 텍스트 요약 과정에서 효율을 극대화하도록 안내합니다.

- **Performance Highlights**: 제안된 프레임워크는 텍스트 요약 과제에서 그 자체의 품질, 유창함, 및 안정성을 토대로 현존하는 고급 LLM 솔루션들과 비교해 우수한 성능을 입증했습니다. 실험 결과, 30명의 참가자로부터 받은 주관적 평가에서도 긍정적인 결과를 얻어 향후 AI 모델 최적화 방향에 대한 잠재적 아이디어를 제시합니다. 이는 인간과 AI 사이의 상호 학습을 모사한 기계 간의 지속적인 발전을 이끌 수 있는 중요한 기초 자료가 될 것입니다.



### nvAgent: Automated Data Visualization from Natural Language via Collaborative Agent Workflow (https://arxiv.org/abs/2502.05036)
- **What's New**: Natural Language to Visualization (NL2Vis) 분야에서 새로운 협력 에이전트 워크플로우인 nvAgent가 제안되었습니다. 이 에이전트는 다중 테이블에 걸친 복잡한 쿼리를 처리하는 데 있어 기존 Large Language Models (LLMs)의 한계를 극복하기 위해 설계되었습니다.

- **Technical Details**: nvAgent는 세 가지 주요 에이전트로 구성됩니다: 데이터베이스 처리 및 컨텍스트 필터링을 담당하는 processor agent, 시각화 생성을 계획하는 composer agent, 코드 번역 및 출력 검증을 수행하는 validator agent입니다. 이러한 구조는 NL2Vis 과정에서의 효율성을 높이고, 다중 테이블 데이터의 시각화 문제를 보다 효과적으로 해결합니다.

- **Performance Highlights**: nvAgent는 새로운 VisEval 벤치마크에서 포괄적인 평가를 통해 기존 최고 성능을 초과하며, 단일 테이블에서 7.88% 향상된 성과를 보였고, 다중 테이블 환경에서는 9.23% 향상된 성능을 달성했습니다. 질적 분석에서도 nvAgent는 이전 모델들보다 거의 20%의 성능 차이를 유지하는 것으로 나타나, 복잡한 데이터 소스로부터 고품질 시각화를 생성할 수 있는 능력을 강조합니다.



### Aligning Black-box Language Models with Human Judgments (https://arxiv.org/abs/2502.04997)
Comments:
          Accepted for publication at NAACL 2025 (Findings)

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 추천 시스템 및 검색 엔진과 같은 평가 작업에서 인간 평가자의 대체로 사용될 수 있음을 강조합니다. 저자들은 LLM의 판단 결과가 인간 평가자와 일치하도록 맞추는 간단하지만 효과적인 방법을 제안하며, 재훈련 없이 LLM의 출력을 인간 판단과 정렬하는 방법을 기술합니다.

- **Technical Details**: 이 방법은 LLM의 출력을 인간 판단과 비교하여 선형 변환을 학습하는 과정을 포함합니다. 제안된 방식은 기존의 모델 로짓에 대한 접근 없이도 사용 가능하며, 적은 양의 데이터로 LLM의 성능을 크게 향상시킬 수 있습니다. 특히, 제안된 프레임워크는 블랙박스 LLM에 적용 가능하여 폭넓게 활용할 수 있습니다.

- **Performance Highlights**: 저자들은 29개의 평가 작업을 통해 LLM 판단의 일치성을 평균 142% 개선할 수 있었음을 보고합니다. 제안된 방법은 제로샷(zero-shot) 및 퓨샷(few-shot) 상황에서 효과적으로 작동하며, 실제로 몇 가지 결정 작업에서 인간 간의 일치성을 초과하는 성능을 보여줍니다. 또한, 작은 LLM도 큰 모델과 비교할 만한 성능을 발휘할 수 있도록 개선할 수 있음을 보여줍니다.



### CoCoA: A Generalized Approach to Uncertainty Quantification by Integrating Confidence and Consistency of LLM Outputs (https://arxiv.org/abs/2502.04964)
- **What's New**: 이번 연구에서는 기존의 Uncertainty Quantification (UQ) 방법들의 한계를 분석하고, 이를 해결하기 위한 새로운 방법인 Confidence and Consistency-based Approaches (CoCoA)를 제안합니다. CoCoA는 정보 기반 (information-based)과 일관성 기반 (consistency-based) 두 가지 접근 방식을 통합하여 더 효율적이고 견고한 UQ 기법을 제공합니다. 이 연구는 다양한 자연어 처리 (NLP) 작업에서 실험을 통해 CoCoA의 신뢰성과 강인성을 크게 향상시키는 결과를 보였습니다.

- **Technical Details**: UQ는 모델의 출력에 대한 신뢰도를 측정하는 방법으로, 정보 이론적 (information-theoretic) 방식과 출력 일관성 분석 (consistency-based analysis)을 기반으로 합니다. 정보 이론적 방법은 단일 샘플에서 불확실성을 추정하는데, 이는 다양한 출력의 의미적 변동성을 고려하지 못하는 한계가 있습니다. 상대적으로 일관성 기반 방법은 모델이 생성한 반복적인 출력을 분석하여 신뢰성을 평가하는 접근입니다; 이러한 기법들은 LLM이 서비스를 통해 제공될 때 자주 사용됩니다.

- **Performance Highlights**: CoCoA 방법론을 적용한 결과, 질문 응답, 요약, 기계 번역 등 다양한 자연어 처리 작업에서 기존의 UQ 방법들과 비교하여 상당한 향상이 있음을 보여주었습니다. 특히, 이 방법은 모델의 신뢰도와 출력 간의 일관성을 통합하여, 더 나은 불확실성 평가를 가능하게 합니다. 실험 결과, CoCoA 접근은 최신 UQ 기술을 초월하는 성능을 달성하며, 이는 LLM의 실제 활용 측면에서 큰 의미를 갖습니다.



### Commonality and Individuality! Integrating Humor Commonality with Speaker Individuality for Humor Recognition (https://arxiv.org/abs/2502.04960)
Comments:
          Accepted by NAACL 2025

- **What's New**: 이 논문에서는 유머 인식을 위한 새로운 모델인 Commonality and Individuality Incorporated Network for Humor Recognition (CIHR)을 제안합니다. CIHR은 유머의 다면성뿐만 아니라 화자의 독특한 개성을 통합하여 유머 인식의 한계를 넘어서는 것을 목표로 합니다. 이 모델은 유머 공통성 분석(Humor Commonality Analysis)과 화자 개성 추출(Speaker Individuality Extraction) 모듈을 포함하여 유머를 다각도로 분석합니다.

- **Technical Details**: CIHR은 유머 공통성 분석 모듈을 통해 사용자의 언어를 6개의 공통적 유머 표현 관점(semantic, pragmatic, syntactic, cultural, cognitive, psychological)에서 분석합니다. 화자 개성 추출 모듈은 화자의 정적(static) 및 동적(dynamic) 프로필 정보를 캡처하여 개인 특성을 정확하게 모델링합니다. 정적 및 동적 융합 모듈은 화자의 개성과 유머 공통성을 효과적으로 통합하여 유머 인식 과정에서 함께 작용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 CIHR이 유머 인식 작업에서 효과적임을 입증했습니다. 연구 결과는 유머 공통성과 화자의 개성을 동시에 고려해야 함을 강조하며, 이를 통해 유머 인식의 성능이 향상될 수 있음을 보여줍니다. 이 논문은 유머 인식에서 이 두 가지 요소의 통합이 중요하다는 것을 밝혔다.



### SSMLoRA: Enhancing Low-Rank Adaptation with State Space Mod (https://arxiv.org/abs/2502.04958)
Comments:
          Has been accepted by NAACL 2025

- **What's New**: 본 연구에서는 LoRA (Low-Rank Adaptation) 기법의 확장인 SSMLoRA(State Space Model Low-Rank Adaptation)를 제안합니다. SSMLoRA는 기존의 LoRA의 한계를 극복하기 위해 State Space Model (SSM)을 도입하여 저차원 행렬 간의 연결성을 강화하고, 파라미터 효율을 높이는 방법입니다. 이를 통해 모델의 입력을 저차원 공간으로 매핑하면서도 이전 저차원 공간의 계산 결과를 활용할 수 있습니다.

- **Technical Details**: SSMLoRA는 주의 메커니즘의 쿼리 및 값 행렬에 대한 개입 레이어를 포함하여 희소 저랭크 행렬을 삽입하는 새로운 방법론입니다. 기존 LoRA에서 발생할 수 있는 불필요한 파라미터 오버헤드를 줄이며, SSM을 활용해 긴 입력 시퀀스를 더욱 효과적으로 처리할 수 있는 구조를 갖추고 있습니다. 실험 결과에 따르면, SSMLoRA는 LoRA보다 적은 파라미터로 유사하거나 더 나은 성능을 달성하며, 긴 텍스트 및 특정 작업에 있어서 두각을 나타냅니다.

- **Performance Highlights**: SSMLoRA는 General Language Understanding Evaluation (GLUE) 벤치마크에서 LoRA와 비교해 동등한 성능을 유지하면서도 필요한 파라미터 수를 절반으로 줄였습니다. 또한 SSMLoRA는 긴 입력 시퀀스를 다루는 작업에 대한 처리 성능이 뛰어나며, 이는 다양한 리소스 제한 환경에서의 모델 활용에 큰 장점을 제공합니다. 이런 특성 덕분에 SSMLoRA는 저비용 데이터 환경에서도 효과적인 성능을 유지할 수 있는 가능성을 보여줍니다.



### Claim Extraction for Fact-Checking: Data, Models, and Automated Metrics (https://arxiv.org/abs/2502.04955)
- **What's New**: 본 논문에서는 Claim Extraction 문제를 탐구하며, LLMs와 소규모 summarization 모델을 비교합니다. 기존 연구에서는 Claim Extraction과 관련된 용어와 방법론이 분산되어 있으므로, 17,000개의 명확한 사실 주장(atomic factual claims)을 Wikipedia 문장에서 추출하여 FEVERFact 데이터셋을 발표합니다. 우리는 이를 통해 체크할 가치가 있는 주장을 자동으로 생성하고 평가하기 위한 프레임워크를 구축했습니다.

- **Technical Details**: 문서는 Claim Extraction을 위해 generative 방법을 사용하여 다양한 데이터 세트를 생성하는 과정을 설명합니다. FEVERFact 데이터셋은 4,400개의 문맥화된 Wikipedia 문장으로 구성되어 있으며 이로부터 총 17,000개의 체크 가치가 있는 주장이 추출되었습니다. 우리는 Atomicity, Fluency, Decontextualization, Faithfulness, Focus, Coverage와 같은 6개의 메트릭을 포함하는 자동 평가 프레임워크를 제안합니다.

- **Performance Highlights**: 모델의 성능은 인간의 평가와 비교를 통해 검증되었으며, 평가 프레임워크의 결과는 $F_{fact}$를 기준으로 모델 순위가 변하지 않음을 보여줍니다. 자동화된 주장 생성 작업에서 FEVERFact 데이터셋을 학습하는 과정에서, 다양한 생성 모델들이 사용되었으며, 이는 정치적 논의나 소셜미디어에서 분산된 주장을 추출하는 데 유용한 기초가 될 것입니다.



### Evaluating Standard and Dialectal Frisian ASR: Multilingual Fine-tuning and Language Identification for Improved Low-resource Performanc (https://arxiv.org/abs/2502.04883)
- **What's New**: 본 논문은 비상용어(Frisian) 및 그 지역 방언을 대상으로 SSL(자기 감독 학습) 기반 모델을 미세 조정하여 텍스트 음성 인식(ASR) 성능을 개선하는 방법을 제안합니다. 다국어 미세 조정 데이터와 언어 식별 작업을 활용하여 Frisian ASR 성능을 향상시킬 수 있다는 것을 보여줍니다. 특히 방언 일반화에 대한 적절한 접근 방식의 중요성을 발견하였으며, 표준 언어 데이터만을 ASR 평가에 의존하는 것이 현실 성능을 과소 평가할 수 있음을 강조합니다.

- **Technical Details**: 연구는 Common Voice 17.0의 5.5시간의 단일 언어 Frisian 데이터를 이용하여 ASR 모델을 미세 조정하는 단계로 시작합니다. 이후 네덜란드어, 독일어, 영어 데이터를 포함한 다국어 미세 조정 데이터를 추가했으며, 이 데이터들은 언어 간 유사성에 따라 순차적으로 통합되었습니다. 특히 XLS-R 1B 모델 아키텍처를 사용하여 1억 개의 매개변수로 구성되며, CTC 손실 함수와 함께 하이퍼파라미터를 설정하여 훈련을 진행하였습니다.

- **Performance Highlights**: 실험 결과, Frisian ASR 성능이 다국어 미세 조정 데이터를 통해 유의미하게 향상되었음을 확인하였습니다. 하지만 방언 음성 인식 성능은 상당히 감소하는 경향이 있으며, 이 효과는 방언 데이터 수집 방식에 의해서도 영향을 받는 것으로 나타났습니다. 이 연구는 다양한 방언이 존재하는 저자원 언어의 ASR 성능을 정확히 평가하는 데 있어, 표준 언어 데이터에 대한 의존도를 재고할 필요성을 제기합니다.



### pytopicgram: A library for data extraction and topic modeling from Telegram channels (https://arxiv.org/abs/2502.04882)
- **What's New**: 이 논문은 Telegram의 메시지를 수집하고 분석하는 데 도움을 주기 위해 개발된 Python 라이브러리인 pytopicgram을 소개합니다. 이 라이브러리는 메시지 검색, 채널 정보 제공, 참여지표 계산 및 고급 모델링 기술을 통한 주제 식별 기능을 포함하여, 연구자들이 Telegram에서 정보를 추적하고 분석할 수 있도록 도와줍니다. 깃허브(https://github.com/ugr-sail/pytopicgram)에서 라이브러리와 이를 사용하는 방법에 대한 자료를 제공하고 있습니다.

- **Technical Details**: pytopicgram은 Telethon 라이브러리를 사용하여 Telegram API에 연결하고, BERTopic 알고리즘을 통해 주제 모델링을 수행합니다. 소프트웨어는 데이터 수집, 정리, 메트릭 계산, 자연어 처리 및 주제 모델링을 포함한 여러 모듈로 구성되어 있으며, 각 모듈은 데이터 분석 파이프라인의 특정 작업을 담당합니다. 이 라이브러리는 Python 3.7 이상에서 작동하며, 사용자는 Telegram API 기본 키와 OpenAI 키가 필요합니다.

- **Performance Highlights**: pytopicgram은 Telegram의 공공 채널에서 메시지를 빠르고 유연하게 수집합니다. 또한, 각 채널에 대한 상세 정보를 수집하고, 메시지 메트릭을 계산하여 콘텐츠 도달 및 참여에 대한 인사이트를 제공합니다. 다양한 언어를 지원하며, 분석의 효율성을 높이기 위해 데이터 최소화 및 프로세스 최적화를 지원하여 대규모 데이터셋에서도 유용하게 사용할 수 있습니다.



### Enhancing Disinformation Detection with Explainable AI and Named Entity Replacemen (https://arxiv.org/abs/2502.04863)
- **What's New**: 이번 연구에서는 정보 왜곡을 탐지하기 위한 자동 시스템의 한계를 극복하기 위해 SHAP(Shapley Additive exPlanations) 기법을 적용했습니다. 우리는 비정보적 요소, 예를 들어 URL과 감정 아이콘을 제거하고 이름 있는 개체를 의사 익명화함으로써 편향을 줄이고 모델의 일반화 능력을 강화할 수 있음을 발견했습니다.

- **Technical Details**: 정보 왜곡 자동 탐지(ADD) 시스템에서 언어 모델 기반의 분류 모델의 행동과 성능에 대한 분석을 진행했습니다. 본 연구는 SHAP 기법을 사용하여 예측 모델의 출력을 영향을 미치는 요소를 식별하고, 표면적 특성을 제거하여 재훈련 한 후의 성능 개선을 확인했습니다.

- **Performance Highlights**: 이번 연구 결과, 외부 테스트 데이터에서 비편향 모델의 F1 점수가 평균 39.32% 향상되었음을 보여주었습니다. 추가 데이터를 사용한 실험에서, 모델은 기존의 분류 지표보다 낮았지만, 더 높은 일반화 능력을 나타냈습니다.



### Self-Rationalization in the Wild: A Large Scale Out-of-Distribution Evaluation on NLI-related tasks (https://arxiv.org/abs/2502.04797)
Comments:
          Accepted at TACL; pre-MIT Press publication version

- **What's New**: 이 논문에서는 기존의 설명 데이터셋을 활용하여 self-rationalization 기법을 연구하고, 모델의 out-of-distribution (OOD) 성능을 평가합니다. 우리는 T5-Large 및 OLMo-7B 모델을 파인튜닝하고, 파인튜닝 데이터의 품질, 샘플 수, few-shot 선택 방법의 영향을 분석합니다. 이를 통해 19개의 다양한 OOD 데이터셋을 기반으로 자연어 추론(NLI), 사실 확인(FC), 요약의 환각 검출(HDAS) 등 세 가지 작업을 수행하였습니다.

- **Technical Details**: 자기 이성화(self-rationalization) 모델은 학습 데이터에서 주어진 태스크 레이블과 자유형 설명을 동시에 생성합니다. 이 논문은 두 개의 오픈소스 모델(T5-Large, OLMo-7B)을 선택하여, 파인튜닝의 데이터 크기와 품질이 OOD 성능에 미치는 영향을 연구합니다. 새로운 접근 방식인 acceptability filtering 모델을 도입하여 OOD 데이터셋에서 생성된 설명의 품질을 높이는 데 초점을 맞추었습니다.

- **Performance Highlights**: 우리는 Acceptability score가 인간 평가에서 가장 높은 상관관계를 보인다는 것을 발견했습니다. 본 연구의 결과, 1) 적은 수의 주석 예시만으로도 OOD 설명 생성을 효과적으로 조정할 수 있으며, 2) 파인튜닝 데이터 소스가 OOD 성능에 큰 영향을 미친다는 점이 밝혀졌습니다. 3) 높은 라벨 예측 정확도를 가진 모델이 더 나은 설명을 생성하는 경향이 있음을 알 수 있었습니다.



### Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition (https://arxiv.org/abs/2502.04795)
Comments:
          13 pages

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models)의 언어 습득 효율성이 인공지능과 인간 간에 차이가 있다는 점에 주목하며, 작업 기억(working memory)의 발달적 특성을 통합하는 방법을 제안합니다. 특히 인간의 언어 습득이 효율적인 중요한 시기(critical period)에 초점을 맞추고 있습니다. 구현된 메커니즘은 훈련 초기 단계에서 작업 기억을 제한하고 학습이 진행됨에 따라 점차적으로 이러한 제약을 완화하는 방식입니다.

- **Technical Details**: 제안된 방법에서는 초기 훈련 단계에서 작업 기억에 대한 제약을 설정하고, 학습이 진행됨에 따라 이 제약이 지수적으로 완화되는 방식으로 구성됩니다. 이러한 접근은 전통적인 메모리 제약이 없거나 고정된 메모리 제약을 가진 모델들과 비교하여 유의미한 성능 향상을 보였습니다. 구체적으로, 타겟 구문 평가(targeted syntactic evaluation)를 통해 양호한 결과를 입증했습니다.

- **Performance Highlights**: 제안된 방법은 전통 모델에 비해 메모리 제약 없이도 더 나은 성능을 나타냈으며, 이는 데이터 효율적인 언어 모델 설계에 새로운 방향을 제시합니다. 또한, 인간의 언어 습득에서의 중요한 시기 가설(critical period hypothesis)을 뒷받침하는 간접적인 증거도 제공하였습니다.



### S$^2$-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency (https://arxiv.org/abs/2502.04790)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문에서는 다중 에이전트 토론(Multi-agent Debate, MAD) 기법을 개선하기 위한 새로운 접근법, 선택적 희소 다중 에이전트 토론(Selective Sparse Multi-Agent Debate, S2-MAD)을 제안합니다. S2-MAD는 에이전트 간의 비효율적인 정보 교환을 줄여 토큰 비용을 최소화하는 동시에 성능 저하를 2.0% 이하로 유지합니다. 이러한 접근은 특히 복잡한 논리적 추론 및 수학적 문제 해결에 있어 LLM(다양한 언어 모델)의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: S2-MAD는 결정적 메커니즘(Decision-Making Mechanism)을 통해 에이전트가 토론에 참가할지를 선택적으로 결정합니다. 각 토론 라운드에서 에이전트는 비슷한 관점을 가진 응답이 아닌 새로운, 비중복 응답을 선택하여 포함하며, 그룹 내 토론 및 그룹 간 토론에 자율적으로 참여할 수 있는 옵션을 제공합니다. 이를 통해 중복 정보를 줄이고 토큰 비용을 절감하는데 기여하는 혁신적인 전략을 설계하였습니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시한 실험을 통해 S2-MAD는 기존의 MAD보다 토큰 비용을 최대 94.5%까지 절감하는 동시에, 개인 에이전트 추론(single-agent reasoning) 접근법과 비교해도 유사한 정확도를 유지할 수 있음을 입증하였습니다. 이러한 결과는 5개 작업을 통해 평가되었으며, S2-MAD는 기존의 방안들, 예를 들어 MAD-Sparse 및 GroupDebate와도 비교하여 우수한 성과를 나타냈습니다. 이로 인해 S2-MAD는 상용 및 오픈소스 모델 간의 혜택을 극대화하는데 기여할 수 있습니다.



### Probing Internal Representations of Multi-Word Verbs in Large Language Models (https://arxiv.org/abs/2502.04789)
- **What's New**: 이번 연구는 트랜스포머 기반의 대규모 언어 모델(LLMs) 내에서의 다단어 동사(verb-particle combinations)의 내부 표현을 조사하였습니다. BERT 아키텍처를 사용하여 다양한 신경망 레이어에서의 어휘적 및 통사적 속성을 어떻게 포착하는지를 분석했습니다. 특히, phrasal verbs('give up')와 prepositional verbs('look at') 두 가지 동사 구조의 표현을 다루었습니다.

- **Technical Details**: 연구는 두 가지 유형의 다단어 동사에 대한 1920개의 phrasal verb 예제와 2070개의 prepositional verb 예제를 포함하는 데이터 세트를 활용했습니다. BERT의 각 레이어에서 토큰 단위 및 문장 단위의 인코딩된 표현을 추출하여, 로지스틱 회귀(logistic regression) 및 서포트 벡터 머신(support vector machines)과 같은 기법을 이용하여 각 카테고리를 분류하는 프로빙(classifier)를 훈련시켰습니다.

- **Performance Highlights**: 결과에 따르면 BERT 모델의 중간 레이어가 가장 높은 분류 정확도를 기록하였으며, GDV 분석을 통해 두 동사 유형 간의 선형 분리 가능성이 약함을 보여주었습니다. 그럼에도 불구하고 프로빙 분류자는 높은 정확성을 달성하여 이러한 언어적 카테고리의 표현이 비선형적으로 분리될 수도 있음을 시사합니다.



### SeDi-Instruct: Enhancing Alignment of Language Models through Self-Directed Instruction Generation (https://arxiv.org/abs/2502.04774)
Comments:
          12 pages, 12 figures

- **What's New**: 이 논문에서는 Self-Direct Instruction generation (SeDi-Instruct)이라는 새로운 데이터 생성 프레임워크를 제안하고 있습니다. SeDi-Instruct는 기존 Self-Instruct의 한계를 극복하고, 고품질의 지침 데이터를 저비용으로 생성하는 것을 목표로 합니다. 이 프레임워크는 다양성 기반 필터링과 반복적 피드백 작업 생성을 활용하여 고품질 지침 생성의 효율성을 개선합니다.

- **Technical Details**: SeDi-Instruct는 Seed Instructions의 품질을 개선하기 위해 학습과 생성 프로세스를 통합하는 반복적 피드백 작업 생성 기술을 사용합니다. 이를 통해 모델의 정확도를 높이는 동시에 보통 58%의 지침 데이터를 버리는 기존 Self-Instruct보다 필터링 효율성을 높이고 있습니다. 이 기술은 데이터 집합의 불균형 문제를 해결하며 각 배치의 다양성을 향상시켜 저품질 지침도 수용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, SeDi-Instruct는 전통적인 방법과 비교하여 AI 모델의 정확도를 5.2% 향상시킴과 동시에 데이터 생성 비용을 36% 절감하는 데 성공했습니다. 이는 SeDi-Instruct가 채택된 모델이 기존의 모델, 특히 Llama-3-8B-Instruct를 제외한 모든 모델을 초과하는 성과를 냈음을 의미합니다. 이러한 결과는 데이터 생성을 위한 공개 소스 또는 기존 데이터를 활용하는 다른 접근 방식에 비해 유리한 점을 보여줍니다.



### Concept Navigation and Classification via Open Source Large Language Model Processing (https://arxiv.org/abs/2502.04756)
Comments:
          35 pages, 1 figure, 7 tabels

- **What's New**: 이 논문은 Open-Source Large Language Models (LLMs)를 활용하여 텍스트 데이터에서 잠재적인 구성 요소(constructs)인 frames, narratives 및 topics를 감지하고 분류하기 위한 혁신적인 방법론적 프레임워크를 제안합니다. 자동 요약과 인간 검증을 결합한 하이브리드 접근 방식은 구성 요소 식별의 정확성과 해석 가능성을 향상시키기 위해 설계되었습니다. 반복 샘플링과 전문례의 개선 과정을 통해 이 프레임워크는 방법론적 강인성을 보장하며 개념적 정확성을 유지합니다.

- **Technical Details**: LLMs는 자연어 처리(NLP)와 계산 언어학의 지형을 변화시켰습니다. 이러한 모델들은 대량의 데이터 세트에서 훈련받아 인간과 유사한 텍스트를 이해하고 생성하는 뛰어난 능력을 보여줍니다. 하이브리드 접근 방식은 LLM의 자동 텍스트 분석과 전문가 검증을 통합하여 효율성과 개념적 신뢰성을 균형 있게 유지합니다.

- **Performance Highlights**: 이 프레임워크는 AI 정책 논쟁, 암호화에 대한 신문 기사 및 20 Newsgroups 데이터 세트를 포함한 다양한 데이터 세트에 적용되어 복잡한 정치 담론과 미디어 프레이밍, 주제 분류 작업을 체계적으로 분석할 수 있는 다재다능성을 보여줍니다. 연구자들은 classical 모델의 한계를 넘어 대규모 텍스트 데이터의 복잡성을 더 포괄적으로 포착할 수 있게 됩니다.



### The "negative end" of change in grammar: terminology, concepts and causes (https://arxiv.org/abs/2502.04729)
Comments:
          10 pages

- **What's New**: 이 논문은 변화의 "부정적인 종말"에 대한 연구가 부족한 반면, 최근 언어학자들 사이에서 관심이 증가하고 있다는 점을 강조합니다. 특히, 언어 변화의 맥락에서 손실(los), 쇠퇴(decline), 구식화(obsolescence)와 같은 현상에 대한 기존 연구가 어떻게 진행되었는지를 정리하고 있습니다. 이는 혁신 및 출현(emergence) 분야와는 대조적으로, 언어학에서 제대로 다뤄지지 않았던 주제임을 명확히 합니다.

- **Technical Details**: 논문은 첫 번째로, 변화의 "부정적인 종말"을 설명하기 위한 용어(terminology)와 개념(concepts)을 정리합니다. 언어 변화의 과정(processes) 속에서 이러한 손실, 쇠퇴, 구식화 현상이 어떻게 발생하는지에 대해 설명한 후, 변화의 주된 원인(causes)들을 탐구합니다. 마지막으로, 표현구(constructions)가 사용 빈도에서 시간이 지남에 따라 일관되게 감소하는 이유를 심도 있게 분석합니다.

- **Performance Highlights**: 이 연구는 특히 언어 변화의 부정적인 측면에 대한 새로운 통찰(insight)을 제공하여 기존의 언어학 연구에 기여합니다. 감소의 원인을 규명하고, 그런 현상이 시간이 지나면서 어떻게 나타나는지를 상세히 설명함으로써, 언어의 진화(evolution)에 대한 보다 깊은 이해를 돕고 있습니다. 또한, 이러한 연구는 향후 언어학적 접근 방식에 혁신을 가져올 잠재력이 있습니다.



### Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics? (https://arxiv.org/abs/2502.04718)
- **What's New**: 이번 연구에서는 Text Style Transfer (TST) 평가를 위한 기존 및 새로운 메트릭을 탐색하고, 감정 전송과 탈염(Detoxification)이라는 두 가지 인기 하위 작업에 초점을 맞추었습니다. 특히, 영어, 힌디어, 벵골어와 같은 다국어 환경에서 이 메트릭의 유용성을 조사하였으며, 인간 판단과의 상관관계를 통해 이러한 메트릭의 효과성을 입증하였습니다. 또한, 대규모 언어 모델(LLMs)을 TST 평가의 도구로 활용하는 가능성도 연구하였습니다.

- **Technical Details**: TST 평가에서 전통적으로 사용되는 세 가지 주요 차원인 스타일 전송 정확성(style transfer accuracy), 내용 보존(content preservation), 유창성(fluidity)을 비교 분석하였습니다. 다양한 메트릭을 카테고리별로 구분하여 사용하였으며, 특별히 스타일 전송 정확성에는 Sentence Accuracy와 WMD를, 내용 보존에는 BLEU, Cosine Similarity, ROUGE-2 등을 활용하였습니다. 또한, 새로운 NLP 작업에서 파생한 추가 메트릭도 사용하여 TST 평가를 확장하였습니다.

- **Performance Highlights**: 하이브리드 접근 방식과 LLM을 사용함으로써 인간 평가와의 상관관계를 개선하며 TST 평가의 정확성, 일관성 및 재현 가능성을 높였습니다. 특히 Hybrid-Simulation 및 Hybrid-Learned 두 가지 앙상블 기반 접근 방식이 매우 효과적이며, 이 메트릭들이 인간 평가와 더 잘 연관된다는 것을 보여주었습니다. 연구 결과는 TST 평가에서 보다 정확하고 심층적인 통찰력을 제공할 수 있는 가능성을 제시합니다.



### Enhancing Impression Change Prediction in Speed Dating Simulations Based on Speakers' Personalities (https://arxiv.org/abs/2502.04706)
- **What's New**: 이 논문은 스피드 데이팅에서 화자 간 인상이 향상되는 대화를 시뮬레이션하는 방법을 제안합니다. 기존의 발화 선택 방법은 화자의 성격이나 발화가 상대 화자의 인상에 미치는 영향을 고려하지 않았습니다. 이 연구에서는 두 화자의 성격을 반영하여 인상을 개선하는 발화를 예측하는 새로운 방법을 구축하였습니다.

- **Technical Details**: 연구는 Multi-Modal Speed Dating (MMSD) 코퍼스를 사용하여 각 발화에 '사랑 척도' 점수를 주석으로 추가하였습니다. 제안된 모델은 대화 콘텍스트와 화자의 성격을 바탕으로 타겟 발화가 상대방의 인상을 어떻게 변화시키는지 예측합니다. 이 모델은 사전 훈련된 transformer 기반의 언어 모델을 활용하여 설계되었으며, 최종 출력에서 최댓값 풀링(max-pooling)을 적용하여 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, 두 화자의 성격을 포함한 모델이 다른 모델에 비해 유의미하게 높은 정확도와 F1 점수를 기록하며 성격이 인상 예측에 유용함을 입증하였습니다. 또한, 제안한 방법을 사용한 시뮬레이션을 통해 실제 대화에서 긍정적인 인상을 남길 수 있는 대화를 생성할 수 있음을 확인했습니다.



### ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning (https://arxiv.org/abs/2502.04689)
Comments:
          20 pages

- **What's New**: 이 논문에서는 Zero-shot Chain-of-Thought (CoT) 프롬프팅의 한계를 극복하고 LLM(대규모 언어 모델)의 질문 답변(QA) 성능을 향상시키기 위해 ARR을 제안합니다. ARR은 질문의 의도 분석, 관련 정보 검색 및 단계별 추론이라는 세 가지 핵심 단계를 포함하여 보다 구체적으로 지침을 제공합니다. 실험을 통해 ARR이 기본 방법(Baseline)과 CoT를 모두 초월하여 일관되게 성능을 개선하는 것을 보여줍니다.

- **Technical Details**: ARR은 질문의 의도를 분석하고, 관련 정보를 검색하며, 체계적인 추론을 실행하는 구조화된 접근 방식을 기반으로 합니다. 특히 ARR은 ‘질문의 의도를 분석하고 관련 정보를 찾아 단계별로 질문에 답하자’라는 명령어를 포함하여 질문 답변 과정에서 LLM의 성능을 극대화합니다. 이를 통해 LLM은 도전적인 QA 작업에서의 성능을 개선하게 됩니다.

- **Performance Highlights**: 실험 결과, ARR은 10개 다중 선택 질문 답변 데이터셋에서 기본 방법 및 CoT 방법보다 일관되게 더 나은 성능을 보여줍니다. 특히, ARR의 세 가지 구성 요소(분석, 검색, 추론)가 개별적으로도 성능을 개선하며, 분석만 포함하는 설정에서 가장 큰 개선 효과를 보입니다. 이 연구는 ARR이 다양한 모델 크기와 설정에서 효과성과 범용성을 입증하였음을 강조합니다.



### M-IFEval: Multilingual Instruction-Following Evaluation (https://arxiv.org/abs/2502.04688)
- **What's New**: 이번 논문에서는 다국어 지원을 위한 새로운 기준인 Multilingual Instruction Following Evaluation (M-IFEval)을 제안합니다. 기존의 Instruction Following Evaluation (IFEval) 기준은 오로지 영어 지침만 포함해 다른 언어에 대한 평가가 한계가 있었으나, M-IFEval은 프랑스어, 일본어, 스페인어를 포함하여 평가 범위를 넓혔습니다. 이를 통해 LLM의 성능을 더 폭넓게 평가할 수 있는 가능성을 제시합니다.

- **Technical Details**: M-IFEval은 일반 지침과 언어 특화 지침을 포함하여 다국어 환경에서의 LLM 성능을 체계적으로 평가하도록 설계되었습니다. 총 8개의 최신 LLM에 이 기준을 적용하여 성능을 비교하였으며, 각 언어와 지침 유형에 따른 성능 차이를 분석했습니다. 이러한 평가 방식은 LLM의 실제 활용 가능성을 측정하는 데 중요한 역할을 할 것입니다.

- **Performance Highlights**: M-IFEval을 적용한 결과, 각 언어와 지침 유형에 따른 LLM의 성능이 크게 달라지는 것을 발견했습니다. 이는 특히 다양한 문화 맥락을 고려할 때, LLM의 평가에 있어 다국어 기준의 필요성을 강조합니다. 이러한 접근법은 전 세계의 다양한 언어 사용자들에게 향상된 AI 경험을 제공할 수 있는 방향을 제시합니다.



### AdParaphrase: Paraphrase Dataset for Analyzing Linguistic Features toward Generating Attractive Ad Texts (https://arxiv.org/abs/2502.04674)
Comments:
          Accepted to NAACL2025 Findings

- **What's New**: 이번 연구에서는 소비자를 유치하기 위한 효과적인 언어적 선택이 광고 성공에 미치는 영향을 탐구했습니다. 특히, 이 연구는 언어적 기능이 인간의 선호도에 미치는 영향을 이해하는 데 중점을 두고 있습니다. 연구자들은 인간의 선호를 반영한 광고 텍스트의 패러프레이즈(Paraphrase) 데이터셋인 AdParaphrase를 발표하여 언어적 차이에 따른 선호도 분석을 가능하게 했습니다.

- **Technical Details**: AdParaphrase 데이터셋은 의미가 동일하지만 언어적 스타일과 표현이 다른 광고 텍스트 쌍의 선호도를 포함하고 있습니다. 이 데이터셋은 광고 텍스트의 매력을 높이기 위해 언어적 특징에 대한 정량적 분석을 가능하게 하며, 선호되는 광고 텍스트가 더 높은 유창성(fluency)을 갖고, 길이가 길며, 명사가 많고, 괄호(bracket) 기호를 빈번히 사용한다는 결과를 보여줍니다. 이러한 특성을 고려한 광고 텍스트 생성 모델이 광고 매력을 크게 향상시킬 수 있음을 입증했습니다.

- **Performance Highlights**: 연구 결과, 인간 심사자들이 선호하는 광고 텍스트는 더 높은 유창성을 나타내고, 더 긴 길이를 가지며, 더 많은 명사를 포함하고, 괄호 기호의 사용이 두드러진다는 것을 확인했습니다. 이러한 발견을 바탕으로, 언어적 특성을 고려한 광고 생성 모델이 매력적인 텍스트 생성 성능을 크게 향상시킨다는 점이 강조되었습니다. 데이터셋은 공개적으로 이용 가능하여 향후 연구에 기여할 것입니다.



### Before It's Too Late: A State Space Model for the Early Prediction of Misinformation and Disinformation Engagemen (https://arxiv.org/abs/2502.04655)
Comments:
          11 pages, 5 figures, 10 tables, Accepted by the Web Conference 2025 (WWW2025)

- **What's New**: IC-Mamba는 소셜 미디어 참여를 예측하는 혁신적인 상태 공간 모델(state space model)로, 간격 중심 데이터(interval-censored data)를 통합된 시간 임베딩으로 모델링하여 예측합니다. 이 모델은 게시 후 15-30분이라는 중요한 초기 시간대에서의 참여 패턴 예측에 뛰어난 성능을 보입니다(RMSE 0.118-0.143), 이는 콘텐츠의 도달 범위에 대한 신속한 평가를 가능하게 합니다. IC-Mamba는 여러 참여 지표에서 기존 최첨단 모델보다 4.72% 향상된 성과를 기록하며, 예측의 정확성을 높여 잠재적으로 문제를 일으킬 수 있는 콘텐츠를 조기에 식별하는 데 도움을 줍니다.

- **Technical Details**: IC-Mamba는 시간 인식을 고려한 임베딩 및 상태 공간 모델 아키텍처를 통해 소셜 미디어 참여의 불규칙한 시간 패턴을 효과적으로 모델링 합니다. 이 모델은 3, 7 및 10일의 데이터 스트림을 활용하여 최대 28일까지 확산 패턴을 예측할 수 있으며, 데이터가 유입될수록 성능이 향상됩니다. 또한, 이 모델은 사용자 참여를 세분화된 수준에서 예측하여 게시물과 더 넓은 내러티브 패턴을 동시에 포착할 수 있습니다.

- **Performance Highlights**: IC-Mamba는 주요 시간대에서의 참여 예측을 통해 강력한 성능을 발휘하며, 총 참여 수준을 추정하고 '떠오르는 의견(emerging opinions)'에 대한 참여 예측이 가능합니다. 모델은 게시물 수준의 동역학(post-level dynamics)과 내러티브 레벨 예측(narrative-level predictions) 모두에서 F1 점수 0.508-0.751을 달성했습니다. 이는 정보 확산을 예측하고 참여 동역학을 조기에 예측할 수 있는 기능을 증명하며, 전문가 작업을 간소화하여 문제 콘텐츠를 조기에 식별할 수 있는 기회를 제공합니다.



### Phonetic Reconstruction of the Consonant System of Middle Chinese via Mixed Integer Optimization (https://arxiv.org/abs/2502.04625)
Comments:
          accepted by TACL

- **What's New**: 본 논문에서는 중고 중국어(Middle Chinese)의 자음 시스템(consonant system)을 음성학적으로 재구성하는 문제를 다루고 있습니다. 우리는 이 문제를 혼합 정수 프로그래밍(Mixed Integer Programming) 문제로 설정하여 자동으로 고대 운서(rhyme dictionaries)에서 동음이의어(homophonic) 정보를 탐색하고 현대 중국어 방언(dialects)에서 음성적(phonetic) 정보를 수집하도록 하였습니다.

- **Technical Details**: 이 연구는 광운(Guangyun)과 20개의 현대 중국어 방언에 대한 정보를 활용하여 새로운 음성 재구성 결과를 도출해냅니다. 이를 통해 고대 언어 데이터와 현대 방언 간의 관계를 효율적으로 모델링하는 방법론을 제안하고 있습니다. 수치적 평가(numerical evaluation)를 통해 합성(synthetic) 데이터와 실제(real) 데이터 모두에서 이 방법의 효과성과 견고성(robustness)을 입증하였습니다.

- **Performance Highlights**: 새로운 방법을 통해 얻어진 음성 재구성 결과는 언어학적인 맥락에서도 논의됩니다. 이 연구는 문자 기록에 기반한 고대 언어의 발음 복원에 있어 혁신적인 접근법을 제공하며, 중고 중국어의 역사적 변화를 이해하는 데 기여할 수 있습니다.



### Extracting and Understanding the Superficial Knowledge in Alignmen (https://arxiv.org/abs/2502.04602)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 정렬(Alignment) 과정에서 표면적 지식(Superficial Knowledge)의 개념을 정립하고, 이를 추출하여 분석하는 방법론을 제시합니다. 특히, 기존의 무거운 조정 과정을 간소화할 수 있다는 점에서, 표면적 지식이 정렬의 중요한 일부분임을 강조합니다. 이는 기존의 연구들이 관찰에 기반하여 하고 있는 가설들과는 차별화되는 접근입니다. 또한, 이 연구에서는 수정이 최소화된 상태에서도 모델의 성능과 안전성을 확보할 수 있는 방법도 제시하고 있습니다.

- **Technical Details**: 정렬된 모델과 정렬되지 않은 모델 간의 상관관계를 설명하기 위해, 각 모델의 트랜스포머 레이어와 최종 선형 프로젝션 매트릭스를 사용하여 모델 아키텍처를 정의합니다. 논문에서는 표면적 지식이 쉽게 얻어진다고 정의하고, 이 지식이 모델의 심층 구조에는 영향을 주지 않도록 하여, 지식을 추출하는 방법을 명확히 하고 있습니다. 추출된 지식은 다양한 벤치마크에서 수학, 안전성, 독성, 진실성 과제를 통해 정량적으로 평가됩니다.

- **Performance Highlights**: 연구 결과에 따르면, 표면적 지식은 안전성과 독성 작업에서 상당한 효과를 나타내며, 모델의 답변 구조화에 기여하는 스타일 패턴으로 이루어져 있습니다. 이 지식만으로도 평균 58%의 수학 성과와 78%의 진실성 작업 개선을 이끌어냈습니다. 그러나, 정렬은 전적으로 표면적이지는 않으며, 특히 사고력과 맥락 이해가 요구되는 작업에서는 깊은 지식이 여전히 필요하다는 점이 강조됩니다.



### My LLM might Mimic AAE -- But When Should it? (https://arxiv.org/abs/2502.04564)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 아프리카계 미국인 영어(AAE)를 어떻게 표현하는지를 탐구합니다. 연구 결과, 아프리카계 미국인들은 LLM이 AAE를 생성할 수 있는 맥락을 원하며, 때로는 일반 미국 영어(Mainstream U.S. English)보다 AAE의 사용을 선호합니다. LLM의 AAE 출력이 아프리카계 미국인의 실제 발화와 동등한 수준의 진정성을 갖춘 것으로 평가되었습니다.

- **Technical Details**: 본 연구는 아프리카계 미국인 104명을 대상으로 한 설문조사와 228명의 아프리카계 미국인이 LLM 생성 AAE를 주관적인 기준으로 평가한 데이터 주석(annotation) 작업을 포함합니다. 설문조사에서는 아프리카계 미국인들이 LLM이 AAE를 사용하는 경우를 원하고, 어떤 사회적 맥락에서 이를 선호하는지에 대한 의견을 수집했습니다. 아울러, LLM의 AAE 생성 능력을 평가하기 위해 GPT-4o-mini, Llama 3, Mixtral의 출력을 평가했습니다.

- **Performance Highlights**: 결과적으로, 아프리카계 미국인들은 공식적인 상호작용에서는 주로 일반 미국 영어를 선호하였으나, 비공식적인 환경에서는 AAE의 사용을 원한다고 응답했습니다. LLM이 생성한 AAE는 인간 순응도 기준과 동등하게 진정한 AAE로 평가되었으며, 우스꽝스럽거나 공격적이지 않다고 여겨졌습니다. 연구팀은 LLM이 생성한 AAE에 대한 아프리카계 미국인의 언어적 판단을 담은 데이터셋을 제공하며, 이를 통해 기술의 포용성과 대표성을 강화하는 방향을 제시합니다.



### TruthFlow: Truthful LLM Generation via Representation Flow Correction (https://arxiv.org/abs/2502.04556)
- **What's New**: TruthFlow라는 새로운 방법이 제안되었습니다. 이 방법은 다양한 질의(query)에 대해 진실한 표현(correct representation) 수정이 가능하도록 Flow Matching 기술을 활용합니다. TruthFlow는 질의를 위한 특정 수정 벡터를 학습하고, 이를 통해 대화형 AI의 진실성을 보강합니다.

- **Technical Details**: TruthFlow는 흐름 모델(flow model)을 통해 질의 특화된 수정 벡터를 학습하고, 이를 사용하여 환각 상태에서 진실 상태로의 전환을 수행합니다. 이 모델은 특정 질의의 표현을 입력받아 해당 질의의 진실한 표현 수정 벡터를 생성합니다. 인퍼런스(inference) 동안, TruthFlow는 생성된 수정 벡터를 사용하여 현재 질의의 표현을 수정하여 결과의 진실성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, TruthFlow는 TruthfulQA 기준에서 다양한 고급 LLM에 대한 성능을 유의미하게 개선하였습니다. 또한, TruthFlow 모델은 강력한 전이 가능성을 보이며, 보지 못한 다른 환각 벤치마크에서도 효과적으로 수행됩니다.



### Contextual Gradient Flow Modeling for Large Language Model Generalization in Multi-Scale Feature Spaces (https://arxiv.org/abs/2502.04548)
- **What's New**: 이 논문은 기존의 uniform gradient propagation 방식의 한계를 극복하기 위해 Contextual Gradient Flow Modeling (CGFM)이라는 새로운 접근 방식을 제안합니다. CGFM은 다층 표현에서의 문맥적 의존성을 인코딩하고 조정하여 각 계층에서의 파라미터 적응성을 향상시킵니다. 이 연구는 이렇게 개선된 가중치 업데이트가 기계학습 모델의 전반적인 수렴 효율성을 높이는 데 기여함을 보여줍니다.

- **Technical Details**: CGFM은 gradient를 단지 수치적 업데이트로 취급하는 대신, 문맥 의존성을 인코딩하는 구조적 개체로 재구성합니다. 이 방법은 differential geometry와 multi-scale tensor fields의 개념을 결합하여 정보를 깊은 네트워크를 통해 재정의하는 원리를 제시합니다. 또한, CGFM은 적응형 feature space 내의 변환으로 가중치 업데이트를 재해석하며, 이는 언어 모델 훈련의 본질적인 특성으로 자리잡습니다.

- **Performance Highlights**: 실험 결과, CGFM을 적용한 모델은 다양한 언어적 맥락에서 더 나은 일반화 성능을 보였으며, 기존의 task-specific fine-tuning에 의존하지 않고도 넓은 범위의 다운스트림 애플리케이션에서 적용성을 강화했습니다. 특히, 구조화된 gradient 전파가 표현 학습 경로에 영향을 미쳤으며, 이는 더 나은 언어적 의존성을 반영하는 결과를 가져왔습니다. 이 연구는 구조적 최적화 전략이 overfitting을 경감시키고 이질적인 텍스트 분포에 대한 적응성을 유지하는 데 어떻게 기여하는지를 잘 보여줍니다.



### Multilingual Non-Autoregressive Machine Translation without Knowledge Distillation (https://arxiv.org/abs/2502.04537)
Comments:
          In Findings of the Association for Computational Linguistics: IJCNLP-AACL 2023

- **What's New**: 이번 논문에서는 비자율적인 다국어 기계 번역(non-autoregressive multilingual machine translation) 시스템인 M-DAT를 제안합니다. 기존의 Switch-GLAT 시스템에 비해 인풋 시퀀스에 대해 지식 증류(knowledge distillation, KD)를 요구하지 않으며, 새롭게 도입한 Pivot Back-Translation(PivotBT) 기법을 통해 보이지 않는 번역 방향으로의 일반화가 소프트웨어적으로 개선되었습니다. M-DAT는 최근에 발전된 방향 비순환 변환기(directed acyclic Transformer, DAT)를 이용하여 학습됩니다.

- **Technical Details**: M-DAT는 하나의 모델로 여러 번역 방향을 처리하도록 설계되었습니다. 특정 언어 쌍에 대해 비자율적인 방식으로 번역을 수행하며, 입력 텍스트에는 소스 및 대상 문장에 대한 언어 태그를 추가합니다. PivotBT는 임의로 선택된 언어에 대해 역번역(back-translation)을 수행하여 주어진 소스 문장의 품질을 개선하는 혁신적인 방법을 사용하고, 이를 통해 자연어 처리에서 중요한 제로샷(zero-shot) 번역을 효과적으로 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, M-DAT는 이전 최고 성능 모델인 Switch-GLAT보다 0.4 높은 BLEU 점수를 달성하며 빠른 추론 속도를 유지했습니다. 또한, M-DAT는 제로샷 번역 설정에서 효과적으로 보이지 않는 번역 방향으로 일반화하는 첫 번째 모델로, 강력한 자율 회귀 모델을 능가하는 성과를 보였습니다. 이러한 성과는 제안된 PivotBT 방법의 기여 덕분으로 평가됩니다.



### A Decoding Algorithm for Length-Control Summarization Based on Directed Acyclic Transformers (https://arxiv.org/abs/2502.04535)
Comments:
          Findings of the Association for Computational Linguistics: EMNLP 2024

- **What's New**: 이번 연구에서는 Directed Acyclic Transformer (DAT)를 기반으로 하는 새로운 길이 제어 요약 알고리즘을 제안합니다. 기존의 autoregressive (AR) 모델 대신에 non-autoregressive (NAR) 접근 방식을 채택하여 여러 가능성 있는 시퀀스 조각을 예측하고 이들을 연결하는 경로를 제공합니다. 또한 SeqMAP라는 새로운 decoding 알고리즘을 소개하여 다양한 가능성의 경로를 마진화하고 길이 제한을 만족하는 가장 높은 확률의 요약을 찾습니다.

- **Technical Details**: 주요 기여 중 하나는 Directed Acyclic Transformer (DAT)를 길이 제어 요약 작업에 적응시킨 것입니다. DAT는 여러 개의 가능성 있는 출력 세그먼트를 생성할 수 있는 능력이 있으며, 이들은 링크를 통해 최종 출력으로 연결됩니다. SeqMAP는 가능성 있는 연결 단계들을 마진화하여 최적의 시퀀스를 찾는 방식으로, 기존의 PathMAP 방식과는 차별화됩니다.

- **Performance Highlights**: Gigaword와 DUC2004 데이터셋에서의 실험 결과, SeqMAP 및 PathMAP이 기존 CTC 기반 모델을 뛰어넘는 성능을 보였습니다. 특히, SeqMAP에 기반한 근사 알고리즘은 reranker 없이도 PathMAP을 초과하는 성능을 보였으며, reranker를 적용할 경우 요약 품질이 더욱 향상됨을 나타냈습니다.



### Group-Adaptive Threshold Optimization for Robust AI-Generated Text Detection (https://arxiv.org/abs/2502.04528)
- **What's New**: 이번 연구는 기존 AI 생성 텍스트 탐지 시스템에서 사용되던 단일 글로벌 임계값의 한계를 지적하고, 그룹별 특성을 고려한 임계값 최적화를 위한 FairOPT 알고리즘을 제안합니다. 저자들은 텍스트의 길이와 스타일과 같은 특성에 따라 데이터 서브그룹을 파악해 각 그룹에 대한 결정 임계값을 학습하여 공정성과 성능을 동시에 보장할 수 있도록 합니다. 이 연구는 AI 생성 콘텐츠 분류기를 더욱 견고하고 공정하게 만드는 길을 열고 있습니다.

- **Technical Details**: FairOPT는 주어진 데이터의 서브그룹을 세분화하여 각 그룹에서 최적의 결정 임계값을 학습하는 방법론입니다. 이 방법은 평가지표인 성능(ACC, F1, precision)과 공정성(인구 통계학적 평등, 기회 균등 등)을 균형 있게 조정합니다. 실험을 통해 FairOPT는 네 가지 AI 텍스트 분류기에서 전체 F1 점수를 증가시키고, 서브그룹 간의 균형 오류율(BER)의 차이를 감소시켰습니다.

- **Performance Highlights**: FairOPT는 세 가지 데이터셋에 대해 네 개의 AI 텍스트 분류기를 실험한 결과, 전체 F1 점수를 향상시키며, 그룹별 균형 오류율의 차이를 감소시켜 성능과 공정성 간의 최적의 절충점을 제공합니다. 이러한 개선은 특히 텍스트의 길이와 스타일에 따라 발생할 수 있는 차별적인 분류 오류를 줄이고, 공정한 AI 텍스트 탐지 시스템 개발에 기여할 것으로 기대됩니다.



### Linear Correlation in LM's Compositional Generalization and Hallucination (https://arxiv.org/abs/2502.04520)
- **What's New**: 이번 논문에서는 언어 모델의 (LM) 일반화 능력에 대한 활발한 논의 중 linear correlation(선형 상관관계) 현상을 밝혀냈습니다. 연구는 LM이 지식을 통합하는 과정에서 특정 관련 지식 간의 선형 변환이 존재함을 보여줍니다. 이러한 변환은 실제 세계 관계와 일치할 때 업데이트된 지식을 일반화하는데 강건하면서도 이탈할 경우 환각을 초래할 수 있습니다.

- **Technical Details**: 연구는 City→Country(도시에서 국가로)와 같은 관련 있는 NTP(next token prediction) 간의 선형 상관관계를 탐구합니다. 모델의 출력 로짓(output logits)에서 선형 변환 (W, b) 를 적합시키는 방식으로 진행되며, 이 과정에서 Pearson correlation coefficients(피어슨 상관계수)를 사용하여 지식 간의 내재적 관계를 평가합니다. 논문에서는 LM의 일반화가 전이되는 방식과 W의 학습된 가중치 간의 관계를 분석하여, 높은 선형 상관관계가 중요하다는 것을 설명합니다.

- **Performance Highlights**: 실험을 통해 W의 정밀도가 높고 높은 상관관계가 있을 때, 원천 지식과 목표 지식 간의 동시 업데이트에 성공적으로 일반화될 수 있음을 확립했습니다. 반면, 피어슨 계수가 높지만 W의 정밀도가 낮을 때는 환각이 발생할 수 있음을 발견했습니다. 이 연구는 단순한 fine-tuning(미세 조정)으로는 LMs가 비선형 방식으로 예측을 일반화하는 데 어려움을 겪는 이유를 설명합니다.



### Beyond Sample-Level Feedback: Using Reference-Level Feedback to Guide Data Synthesis (https://arxiv.org/abs/2502.04511)
- **What's New**: 이 논문에서는 Reference-Level Feedback이라는 새로운 방법론을 제안합니다. 이 방법론은 세심하게 선별된 시드 데이터로부터 고품질의 레퍼런스 샘플을 기반으로 피드백을 수집합니다. 이는 기존의 샘플 수준(sample level) 피드백보다 더 포괄적이며, 새로운 합성 데이터에 바람직한 특성을 전파하는 데 도움을 줍니다.

- **Technical Details**: REFED라는 데이터셋은 10,000개의 지시응답 쌍을 포함하고 있으며, 이 데이터셋은 고품질 피드백을 통해 합성되었습니다. 이 연구는 Llama-3.1-8B-Instruct 모델을 REFED로 파인튜닝하여 AlpacaEval 2.0 및 Arena-Hard에서 모형 평가지표를 향상시킴을 입증합니다. 또한, Reference-Level Feedback 방법론은 다양한 모델 아키텍처에서 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: REFED에서 파인튜닝한 Llama-3.1-8B-Instruct 모델이 유사한 크기의 SFT 기반 모델들 사이에서 최고 성능을 기록했습니다. 이 연구에서는 전통적인 샘플 수준 피드백 방법들보다 적은 피드백 수집으로도 더 나은 결과를 얻었음을 보여줍니다. 즉, 본 방법론은 데이터 품질을 유지하면서도 성능을 향상시키는 효과적인 접근법임을 입증합니다.



### Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems (https://arxiv.org/abs/2502.04510)
- **What's New**: 본 논문에서는 Heterogeneous Swarms라는 알고리즘을 제안하여 다수의 LLM 시스템을 설계하는 방법을 소개합니다. 이 접근법은 모델 역할과 가중치를 공동 최적화하여 협업 생성을 위한 방향성 비순환 그래프(directed acyclic graph, DAG)로 LLM 시스템을 표현합니다. 기존의 다수 LLM 시스템 개발 방식은 역할과 가중치를 고정하게 되어 유연한 작업 및 컨텍스트에 적응하지 못하는 단점이 있었으며, 이 문제를 해결하려고 합니다.

- **Technical Details**: Heterogeneous Swarms는 두 가지 반복적인 단계로 구성됩니다: 역할 단계(role-step)와 가중치 단계(weight-step)입니다. 역할 단계에서는 LLM 간의 입력-출력 관계를 그래프 학습 문제로 해석하여 DAG를 학습합니다. 가중치 단계에서는 JFK-score를 제안하여 다수 LLM 시스템에서 각 모델의 기여도를 정량화하고, 이를 기반으로 모델 가중치를 최적화합니다.

- **Performance Highlights**: Heterogeneous Swarms는 12개 작업에서 15개의 기존 역할/가중치 기반 접근 방식에 비해 평균 18.5% 성능 향상을 보였습니다. 이 시스템은 협업의 규모를 늘림으로써 더 작은 언어 모델의 추론 성능을 향상시키며, LLM 간 협력을 2에서 10으로 확장할 때 평균 27.1%의 개선 효과를 나타냅니다. 다양한 역할과 가중치의 중요성이 다르게 나타나는 점도 주목할 만합니다.



### When One LLM Drools, Multi-LLM Collaboration Rules (https://arxiv.org/abs/2502.04506)
- **What's New**: 이 논문에서는 단일 LLM(대형 언어 모델)만으로는 신뢰할 수 있는 출력을 생성하는 데 한계가 있음을 주장합니다. LLM 기술이 발전함에 따라 많은 사용자들이 단일 LLM에 의존하려는 경향이 강하지만, 이는 데이터의 다양성과 사회적 맥락을 충분히 반영하지 못하는 문제점을 발생시킵니다. 이 논문은 여러 LLM이 협력할 수 있는 새로운 모델을 제안하며, 다양한 데이터와 기술적 요구를 더 잘 충족할 수 있다고 강조합니다.

- **Technical Details**: 연구진은 LLM의 협업 방법을 API 수준, 텍스트 수준, 로그잇 수준, 가중치 수준의 계층으로 분류합니다. 이를 통해 복잡한 실세계의 데이터를 다루고, 여러 사용자 요구를 충족시키는 방법을 제시하고 있습니다. 이처럼 다양한 협업 방식이 LLM의 효율성과 적응성을 높일 수 있다는 점에 주목하고 있습니다.

- **Performance Highlights**: 단일 LLM에 비해 다중 LLM 협업 시스템은 신뢰성, 민주화, 복잡성, 효율성 등의 이점을 제공합니다. 이 논문은 다중 LLM 협업이 단일 모델이 해결하기 힘든 여러 과제를 극복할 수 있는 가능성을 보여주고 있습니다. 향후 연구 방향으로, 기존 모델의 한계를 극복하고 모듈형 다중 LLM 시스템으로 발전할 수 있는 로드맵을 제시할 것을 촉구합니다.



### ULPT: Prompt Tuning with Ultra-Low-Dimensional Optimization (https://arxiv.org/abs/2502.04501)
- **What's New**: 이번 논문에서는 Ultra-Low-dimensional Prompt Tuning (ULPT)이라는 혁신적인 방법을 제안합니다. ULPT는 저차원 공간(예: 2D)에서 프롬프트를 최적화하며, 훈련 가능한 매개변수를 극적으로 줄이면서도 강력한 성능을 유지합니다. 이 방법은 기존의 프롬프트 임베딩과 모델 차원을 분리하여 필요하지 않은 복잡성을 제거합니다.

- **Technical Details**: ULPT는 고유한 랜덤 프로젝션 매트릭스를 사용하여 훈련 가능한 매개변수를 최소화합니다. 프롬프트의 업 프로젝션을 위한 고정된 랜덤 매트릭스와 학습 가능한 이동 및 스케일 임베딩 벡터를 도입하여 성능 향상을 꾀합니다. 이론적 분석에 따르면 저차원 공간에서의 랜덤 프로젝션이 고차원 정보를 효과적으로 근사할 수 있음을 보여줍니다.

- **Performance Highlights**: ULPT는 21개의 NLP 작업에서 평범한 프롬프트 튜닝의 2%만으로 경쟁력 있는 성능을 발휘하며, 기존 방식보다 훨씬 적은 훈련 가능한 매개변수로 확장됩니다. ULPT는 고차원으로 스케일링할 때도 기존 프롬프트 튜닝 기반 방법들보다 성능이 우수한 것으로 나타났습니다.



### Verifiable Format Control for Large Language Model Generations (https://arxiv.org/abs/2502.04498)
Comments:
          To appear at Findings of NAACL 2025

- **What's New**: 이 논문은 7B 매개변수를 가진 소형 대형 언어 모델(LLM)의 형식 수행 능력을 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 방법들은 일반적인 지침 따르기에 대한 벤치마크에 중점을 두었으나, 특정 형식 따르기 능력 개선을 소홀히 했습니다. 이를 해결하기 위해 저자들은 완전히 검증 가능한 형식 따르기 데이터셋인 VFF를 만드는 데 주력하였고, 이를 통해 소형 LLM의 훈련 및 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: VFF 데이터셋은 GPT-4에서 주석이 달린 메타 제약 조건을 바탕으로 구성되어 있습니다. 이 데이터셋은 여러 개의 변수를 포함하여 다양한 형식을 생성할 수 있는 지침과 메타 제약이 결합된 것입니다. 또한 각 제약을 검증하기 위한 파이썬 함수가 제공되어 모든 샘플이 쉽게 검증될 수 있습니다. 연구진은 이러한 검증 가능한 특성을 활용하여 소형 LLM을 점진적으로 훈련시키는 방법론을 제안합니다.

- **Performance Highlights**: 실험 결과는 7B 수준의 오픈 소스 LLM들이 형식 따르기 능력에서 빈번한 한계를 보인다는 것을 보여줍니다. VFF 데이터셋을 기반으로 한 저자들의 방법은 소형 LLM의 형식 제어 능력을 향상시키는 데 효과적임을 입증하였습니다. 이 방법은 LLM이 생성한 데이터를 스스로 학습하여 성능을 개선할 수 있는 패러다임을 제시합니다.



### Multi-Agent Reinforcement Learning with Focal Diversity Optimization (https://arxiv.org/abs/2502.04492)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)과 그들의 미세 조정(finetuning) 방법의 발전이 다중 에이전트 강화 학습(multi-agent reinforcement learning)에 대한 관심을 촉발했음을 다룹니다. MARL-Focal이라는 새롭고 유일한 접근 방식을 제안하며, 에이전트 간의 협업, 에이전트 선택 알고리즘 및 충돌 해결 메커니즘을 포함합니다. 이 연구는 다양한 이점을 통해 최종 추론(output)을 위한 협력적인 작업을 수행하고, 다양한 LLM의 결합 및 충돌 해결을 통해 결과의 일관성을 강화하려고 합니다.

- **Technical Details**: MARL-Focal 접근 방식은 에이전트 융합(agent-fusion) 프레임워크를 통해 여러 LLM 기반 에이전트가 최종 추론 출력을 생성하기 위해 협력하도록 유도합니다. 또한, 포컬 다양성(focal-diversity) 최적화된 에이전트 선택 알고리즘을 통해 서로 보완할 수 있는 소수의 에이전트를 선택하여 쿼리 출력을 생성합니다. 마지막으로, 여러 에이전트 간 출력의 불일치를 감지하고 보상 인식(reward-aware) 및 정책 적응(policy-adaptive) 추론 융합을 통해 MARL-Focal 출력을 생성하는 충돌 해결 방법을 설계합니다.

- **Performance Highlights**: 다양한 벤치마크에서의 포괄적인 평가에 따르면, MARL-Focal은 가장 비용 효율적이며 적대적 상황에서도 강력한 성능을 보입니다. 이 모델은 최고의 개별 LLM-에이전트 대비 5.51% 성능 향상을 달성하고 TruthfulQA 벤치마크에서 더욱 강력한 내구성을 제공합니다. 이를 통해 레이블이 없는 데이터셋에서도 더욱 유용하고 안전하며 진실한 시스템을 구축할 수 있음을 보여줍니다.



### Building A Unified AI-centric Language System: analysis, framework and future work (https://arxiv.org/abs/2502.04488)
- **What's New**: 최근 대규모 언어 모델에 대한 발전은 연장된 추론을 통해 성능을 크게 향상시킬 수 있음을 보여주고 있습니다. 그러나 이러한 개선은 계산 비용의 증가와 자연어에서 발견되는 고유한 편견의 전파를 동반합니다. 이 논문은 이러한 문제를 해결하기 위해 보다 간결하고 명확하며 계산 효율적인 AI 중심 언어 시스템의 설계를 탐구합니다.

- **Technical Details**: 자연어의 성별 편향, 형태론적 불규칙성, 문맥적 모호성과 같은 한계를 분석하고, 이러한 문제들이 현재의 Transformer 아키텍처 내에서 어떻게 악화되는지 살펴봅니다. 또한 이 논문은 인공지능에 최적화된 새로운 언어 설계를 통해 계산 효율성을 높이고 편향과 모호성을 줄이는 방법을 제안합니다.

- **Performance Highlights**: AI 중심 언어를 통해 다양한 자연어 입력을 효율적으로 처리할 수 있으며, 이를 통해 메모리 사용량과 추론 시간을 줄일 수 있습니다. 최종적으로 이러한 접근 방식을 통하여 AI 간의 상호 작용과 인간-AI 상호 작용의 명확성, 공정성 및 성능을 획기적으로 혁신할 수 있는 가능성을 제시합니다.



### Active Task Disambiguation with LLMs (https://arxiv.org/abs/2502.04485)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 성능이 우수하지만, 실제 상호작용에서 자주 발생하는 모호하게 지정된 문제를 해결하는 능력이 충분히 탐구되지 않았음을 지적합니다. 이를 해결하기 위해 연구자는 작업 모호성(task ambiguity)에 대한 공식 정의를 제시하고, 베이esian 실험 설계(Bayesian Experimental Design)의 관점에서 작업 불명확성(task disambiguation) 문제를 구성합니다.

- **Technical Details**: 작업 문제를 명확히 하기 위한 질문을 제시함으로써 LLM 에이전트는 추가적인 작업 명세를 획득하고, 가능한 솔루션 공간을 점진적으로 좁힐 수 있습니다. 그러나 효과적인 명확화 질문을 생성하는 것은 LLM 에이전트가 메타 인지적 추론(meta-cognitive reasoning)을 수행할 능력이 필요하지만, 현재 LLM이 이러한 능력을 결여하고 있음을 지적합니다. 이 연구는 적극적인 작업 불명확성 해소(active task disambiguation) 접근 방식을 제안하여 정보 이득(information gain)을 극대화하는 질문을 생성합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 질문 선택 방법은 질문 공간 내에서만 추론하는 기존 접근 방식에 비해 더 효과적인 작업 불명확성 해소를 보여줍니다. 이는 LLM이 솔루션 공간에 대한 암묵적 추론(implicit reasoning)에서 명시적 추론(explicit reasoning)으로 전환하도록 하여 LLM의 문제 해결 능력을 향상시킵니다.



### "In order that" -- a data driven study of symptoms and causes of obsolescenc (https://arxiv.org/abs/2502.04457)
Comments:
          10 pages

- **What's New**: 이 논문은 진행 중인 문법적 퇴화(grammatical obsolescence)의 경험적 사례 연구(empirical case study)를 제공합니다. 특히, ‘in order that’라는 목적 종속접속사(subordinator)의 사용 빈도가 20세기 초부터 지속적으로 감소하고 있음을 보여줍니다. 나아가, Rudnicka(2019)가 최근에 개발한 데이터 기반 접근법(data-driven approach)을 적용한 점이 돋보입니다.

- **Technical Details**: 이 연구는 필로로지적 분석(philological analysis)과 대용량 코퍼스(mega-corpora)에서 얻은 데이터에 대한 통계적 방법(statistical methods)을 결합한 방법론(methodology)을 사용합니다. 논문은 퇴화의 증상(symptoms of obsolescence)에서 그것의 다양한 원인(causes)으로 이동하면서 연구된 현상에 대한 포괄적인 설명(comprehensive account)을 제시하는 것을 목표로 합니다.

- **Performance Highlights**: ‘in order that’의 감소에는 두 가지 주요한 고차적 과정(higher-order processes)이 관련되어 있는 것으로 나타났습니다. 첫 번째는 19세기와 20세기의 급격한 사회문화적 변화에 의해 동기를 부여받은 외적 고차적 과정(externally-motivated process)이며, 두 번째는 to-infinitive의 상승으로 나타난 내적 고차적 과정(internally-motivated process)입니다.



### Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization (https://arxiv.org/abs/2502.04428)
- **What's New**: 이 논문은 불확실성 기반의 SLM 라우팅을 통해 소형 언어 모델(SLM)과 대형 언어 모델(LLM)의 통합을 모색합니다. SLM이 낮은 신뢰도로 인한 복잡한 쿼리에 부정확한 응답을 생성할 경우 LLM으로 쿼리를 이관하는 시스템을 제안합니다. 이를 통해 사용자는 더 높은 신뢰성을 확보하고 비용 효율성을 유지할 수 있습니다.

- **Technical Details**: 이 연구는 1500개 이상의 설정에서 SLM과 LLM 간의 불확실성 기반 라우팅 전략의 벤치마킹과 일반화를 수행하였습니다. 다양한 불확실성 정량화(UQ) 방법이 라우팅 성능에 미치는 영향과 맞춤 데이터를 활용한 일반화 전략을 고찰합니다. 결과적으로, 불확실성 정량화 방법의 정확성 조정이 라우팅 성능에 중요한 역할을 함을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 생성된 보정 데이터(calibration data)가 SLM과 LLM 간의 라우팅 성능을 향상시키며, 신규 데이터 없이도 효과적인 라우팅 결정을 가능하게 합니다. 또한, 불확실성 기반 라우팅 접근법은 다양한 데이터셋에 잘 일반화되며, 라우팅 전략의 효율성을 높이는데 기여함을 보여주었습니다.



### Decoding AI Judgment: How LLMs Assess News Credibility and Bias (https://arxiv.org/abs/2502.04426)
- **What's New**: 본 연구는 최신의 대형 언어 모델(Large Language Models, LLMs)인 Gemini 1.5 Flash, GPT-4o mini, 그리고 LLaMA 3.1이 신뢰성을 평가하는 방식을 분석합니다. 그 과정에서 LLM이 인간 전문가의 평가 체계를 어떻게 반영하거나 다르게 나타내는지를 탐구하며, '신뢰성'이라는 개념을 구성하는 언어적 단서들을 밝혀냅니다. 또한, LLM이 외부 정보를 검색하고 다른 모델과 상호작용하며 평가를 정교화하는 과정을 통해, 이러한 모델들이 직관적 사고 아니면 이전의 학습된 연관성에 의존하는지를 조사합니다.

- **Technical Details**: 이 연구에서는 총 2,302개의 뉴스 출처를 대상으로 LLM의 신뢰성과 정치적 분류를 평가합니다. 뉴스 출처는 NewsGuard와 Media Bias Fact Check의 구조화된 평가를 바탕으로 구성되었으며, 다양한 언어적 단서와 맥락적 단서를 통해 LLM의 평가 프로세스를 세밀하게 분석합니다. 모델들은 제로샷 접근 방식을 통해 평가 질문에 답하며, 각 뉴스 출처에 정치적 방향성을 부여하고 그렇게 이끌어낸 평가에 대한 설명을 생성합니다.

- **Performance Highlights**: 각 LLM 모형은 NewsGuard가 정한 신뢰성 등급과 비교하여 유의미한 결과를 보여줍니다. 특히, 모든 모델이 ‘신뢰할 수 없는’ 출처를 높게 식별한 반면, ‘신뢰할 수 있는’ 출처의 분류는 상대적으로 어려움을 보여주었습니다. LLM의 평가 결과는 NewsGuard와 Media Bias Fact Check(MBFC)의 기준과 일반적으로 유사하지만, 고란 위험 신뢰 평가 소스의 경우 다소간의 차이가 발생했습니다.



### EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models (https://arxiv.org/abs/2502.04424)
- **What's New**: 이번 논문에서는 Multimodal large language models (MLLMs)를 로봇 시스템과 AI 애플리케이션에 통합하면서 감정 지능 (Emotional Intelligence, EI) 기능을 중요한 요소로 제시합니다. 기존의 정적이고 텍스트 기반 또는 텍스트-이미지 벤치마크는 실생활의 복잡한 다중 양식 상호작용을 간과하고, 감정 표현의 동적인 특성을 포착하지 못해 MLLMs의 EI를 평가하는 데 적절하지 않음을 지적합니다. 이와 같은 점을 보완하기 위해, EmoBench-M이라는 새로운 벤치마크를 개발하였습니다.

- **Technical Details**: EmoBench-M은 감정 지능 평가를 위한 새로운 벤치마크로, 13개의 평가 시나리오를 세 가지 주요 차원인 기초 감정 인식 (foundational emotion recognition), 대화 감정 이해 (conversational emotion understanding), 사회적 복잡 감정 분석 (socially complex emotion analysis)으로 나누어 구성하였습니다. 이 벤치마크는 감정 지능의 다양한 측면을 포괄적으로 평가할 수 있도록 설계되었습니다. 모든 평가 리소스는 공개적으로 제공되며, 코드와 데이터셋에 대한 접근이 가능합니다.

- **Performance Highlights**: EmoBench-M를 이용한 평가에서, 오픈 소스 및 클로즈드 소스 MLLMs는 인간에 비해 상당한 성능 차이를 보였으며, 이는 MLLMs의 EI 기능을 더욱 발전시킬 필요성을 강조합니다. 이러한 결과는 MLLMs가 인간의 감정적 요구를 효과적으로 다루기 위한 기술적 도전이 여전히 존재함을 나타냅니다. 감정 지능의 향상은 로봇과 AI의 실용적인 활용에 크게 기여할 것입니다.



### MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilo (https://arxiv.org/abs/2502.04413)
- **What's New**: 이 논문에서는 의료 분야에서 지식 그래프(KG)를 활용하여 진단 및 치료 권장을 제공하는 MedRAG라는 새로운 Retrieval-augmented generation (RAG) 모델을 제안합니다. MedRAG는 비슷한 증상을 가진 질병들 간의 진단 차이를 체계적으로 구성한 4단계 계층적 진단 KG를 구축하여, 기존 모델의 다소 부족한 진단 정확성과 특이성을 개선합니다. 이 모델은 환자의 증상에 기반하여 더 정확하고 구체적인 의료 의사결정을 지원함으로써, 잘못된 진단의 위험을 줄이는 데 기여합니다.

- **Technical Details**: MedRAG는 진단 지식 그래프와 RAG의 통합을 통해, 환자 정보를 보다 명확하게 이해하고 그에 따른 후속 질문을 제시하는 기능을 갖추고 있습니다. 새로운 진단 KG 검색 모듈을 통해 입력된 환자와 관련된 모든 중요한 진단 차이를 식별하고, 대규모 언어 모델 내에서 이 정보를 결합하여 추론을 수행합니다. 이 과정은 증상들이 유사한 질병 간의 미세한 진단 차이를 구별할 수 있는 개선된 추론 능력을 제공합니다.

- **Performance Highlights**: MedRAG는 DDXPlus 데이터셋과 Tan Tock Seng 병원에서 수집된 CPDD 데이터셋을 통해 평가되었으며, 여러 최신 RAG 모델과 비교하여 잘못된 진단 비율을 줄이는 데 있어 우수한 성능을 보였습니다. 실험 결과에 따르면, MedRAG는 기존의 RAG 접근 방식들보다 높은 진단 정확성과 특이성을 제공하며, 다양한 LLMs에서 robust한 일반화 성능을 나타냈습니다. 또한, 이 모델은 진단 질문을 보다 효과적으로 생성하여 복잡한 의료 시나리오에서 의사결정 과정을 최적화하는 데 큰 기여를 할 수 있습니다.



### Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models (https://arxiv.org/abs/2502.04404)
Comments:
          This is a preprint under review, 15 pages, 13 figures

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)에 slow-thinking 메커니즘을 통합하여 AGI(인공지능 일반화) Reasoners의 Level 2 달성을 위한 새로운 경로를 제시합니다. 기존 모델들은 오버씽킹(overthinking) 문제와 보조 보상 모델(auxiliary reward models)에 대한 과도한 의존성이 있었으며, 이로 인해 비효율성이 발생했습니다. 연구팀은 LLMs가 검색 프로세스를 내부화(internalize)할 수 있도록 self-backtracking 메커니즘을 도입했습니다.

- **Technical Details**: Self-Backtracking 기법은 LLM이 훈련 단계에서 언제, 어떤 상황에서 backtracking을 수행해야 하는지를 학습하도록 돕습니다. 이 방법은 모델이 초기 예측이나 추론 경로가 최적이 아닐 때 이를 인지하고 조기에 상태로 되돌아가 대안 가능성을 탐색하도록 합니다. 실험 결과, 제안한 기법이 기존의 SFT(supervised fine-tuning) 방식에 비해 40% 이상의 성능 향상을 기록했습니다.

- **Performance Highlights**: 제안된 self-backtracking 기법은 LLMs의 추론 유연성과 전반적인 성능을 개선하며, 다양한 파라미터 규모의 모델에서도 뛰어난 효과를 나타냅니다. Countdown 작업(task)에 대한 실험에서 이 기법이 특히 두드러진 성과를 보여주었으며, 이는 Level 2 AGI Reasoners 달성을 향한 중요한 진전으로 평가됩니다.



### Multimodal Medical Code Tokenizer (https://arxiv.org/abs/2502.04397)
Comments:
          conference

- **What's New**: 이번 연구에서는 전자 건강 기록(EHR)에 대한 다중 모드 의료 코드 토큰화기인 MedTok을 소개합니다. 기존의 표준 토큰화 방법이 의료 코드를 고립된 텍스트 토큰으로 처리하는 반면, MedTok은 텍스트 설명과 관련 맥락을 통합하여 더 풍부한 의료 코드 표현을 가능하게 합니다. MedTok의 도입은 다양한 EHR 모델의 성능을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: MedTok은 언어 모델 인코더를 사용하여 텍스트를 처리하고, 그래프 인코더를 통해 의료 코드의 관계 구조를 인코딩합니다. 이 과정에서 두 가지 모드를 통합하여 통합된 토큰 공간으로 양자화(quantize)하며, 모드별 및 교차 모드 정보 보존을 보장합니다. MedTok는 또한 의료 질문-답변 시스템에 적용되어 사용될 수 있습니다.

- **Performance Highlights**: MedTok를 다섯 개의 EHR 모델에 통합한 결과, 전통적인 EHR 토큰화기를 MedTok으로 교체할 경우 모든 EHR 모델에서 AUPRC의 개선이 관찰되었습니다. MIMIC-III에서 4.10%, MIMIC-IV에서는 4.78%, EHRShot에서는 11.30% 향상되었습니다. 특히 약물 추천에서 가장 큰 성과를 기록했습니다.



### DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Diseas (https://arxiv.org/abs/2502.04394)
- **What's New**: 본 논문에서는 알츠하이머병(AD) 조기 진단을 위한 새로운 접근법인 DECT를 제안합니다. DECT는 대형 언어 모델(LLMs)를 활용하여 언어적 분석을 세밀하게 수행하고, 노이즈가 포함된 음성 전사에서 중요한 인지-언어적(Cognitive-Linguistic, CL) 정보를 추출합니다. 이 연구는 AD 탐지 모델의 정확성을 11% 향상시키는 성과를 보였습니다.

- **Technical Details**: DECT는 네 가지 핵심 단계로 구성되어 있습니다. 첫째, LLM을 활용해 언어적 데이터를 정제하고, 둘째, 비구조적인 음성 전사에서 언어적 마커를 추출합니다. 셋째, 이러한 마커와 CL 아톰을 결합하여 보다 정교한 데이터 표현을 생성하고, 넷째, 증대된 음성 전사 데이터를 바탕으로 AD 탐지 모델을 세부 조정합니다.

- **Performance Highlights**: DECT의 결과는 DementiaBank 데이터셋에서 기존 기준선 대비 11% 향상된 AD 탐지 정확성을 보여주었습니다. 이는 자동화된 진단 도구 개발에 있어 언어적 패턴 분석의 가능성을 동반하며, AD 조기 발견과 치료 모니터링에 큰 기여를 할 것으로 기대됩니다.



### Division-of-Thoughts: Harnessing Hybrid Language Model Synergy for Efficient On-Device Agents (https://arxiv.org/abs/2502.04392)
- **What's New**: 이번 논문에서는 Division-of-Thoughts (DoT) 라는 협동적 추론 프레임워크를 제안합니다. DoT는 로컬에 배치된 Smaller-scale Language Models (SLMs)와 클라우드 기반의 Large Language Models (LLMs) 간의 시너지를 활용하여 사용자의 쿼리를 더 작은 하위 작업으로 분해하는 Task Decomposer를 포함합니다. 이를 통해 복잡한 온라인 작업을 효율적으로 관리할 수 있도록 도와줍니다.

- **Technical Details**: DoT는 또한 Task Scheduler를 통해 하위 작업 간의 종속성을 분석하고 종속성 그래프를 작성하여 병렬 추론을 촉진합니다. Plug-and-Play Adapter를 사용하여 하위 작업의 난이도에 따라 적절한 모델을 할당하며, 이는 SLM의 파라미터를 변경하지 않고도 가능하게 합니다. 자율 강화 훈련 방법 또한 도입하여 인간의 주석 없이 작업 실행 피드백을 기반으로 하위 작업을 배분할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, DoT는 평균 추론 시간을 66.12% 감소시키고 API 비용을 83.57% 줄이는 동시에 경쟁력 있는 추론 정확도를 유지했습니다. 다양한 벤치마크를 통해 DoT의 효과를 입증했으며, 비용과 정확도의 균형을 뛰어난 성과로 보여줍니다. 전반적으로 DoT는 AI 개인 비서의 성능 향상에 기여할 수 있는 가능성을 제시합니다.



### In Praise of Stubbornness: The Case for Cognitive-Dissonance-Aware Knowledge Updates in LLMs (https://arxiv.org/abs/2502.04390)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 지속적인 지식 업데이트 방식을 인지적 접근법을 통해 탐구합니다. 기존의 언어 모델은 새로운 정보를 통합하는 데 어려움이 있으며, 이는 종종 이전에 학습한 지식을 파괴적인 망각(catatrophic forgetting)으로 이어지곤 합니다. 연구자들은 두 가지 주요 구성 요소인 불일치 및 친숙성 인식(dissonance and familiarity awareness)과 표적 신경망 업데이트(targeted network updates)를 도입하여 언어 모델의 행동을 분석하고 지식 통합의 근본적인 특징을 밝혀냈습니다.

- **Technical Details**: 이 연구에서는 신경 활동을 추적하여 모델이 새로운 정보를 새로운, 친숙한 또는 불일치하는 정보로 분류할 수 있는지 탐구하는 방법을 개발했습니다. 연구팀은 과거 사용에 따라 뉴런을 '플라스틱(plastic)'과 '고집하는(stubborn)' 것으로 분류할 수 있는 방법도 모색했습니다. 이러한 분석을 통해 언어 모델의 매개변수 공간에서 새로운 지식의 위치가 기존 지식의 통합에 미치는 영향을 심층적으로 연구하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 비불일치(non-dissonant) 업데이트는 모델의 이전 지식을 대부분 보존하는 반면, 불일치(dissonant) 업데이트는 기존 지식에 심각한 파괴적 영향을 미친다는 것을 발견했습니다. 이러한 발견은 대형 언어 모델이 모순된 정보를 처리하는 방식에서 본질적인 한계를 지니고 있음을 시사합니다. 따라서, 연구팀은 모순된 정보를 보다 효과적으로 처리할 수 있는 새로운 메커니즘의 탐색이 필요하다는 결론에 도달했습니다.



### FedP$^2$EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs (https://arxiv.org/abs/2502.04387)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 다국어에 걸쳐 저자 특정 성능을 향상시키기 위한 새로운 방법론인 FedP2EFT를 제안했습니다. 연합 학습(federated learning) 중 개인화를 위한 최적의 매개변수를 선택하기 위해 Bayesian sparse rank selection을 활용하여, 각 클라이언트별로 최적의 PEFT 구조를 공동으로 학습합니다. 그 결과, FedP2EFT는 기존의 방법들에 비해 성능상이 크게 개선된다는 것을 보였습니다.

- **Technical Details**: FedP2EFT는 연합 학습을 통해 클라이언트가 공동으로 언어 개인화 전략을 학습할 수 있도록 하며, PS generator(PSG)를 통해 메타 데이터를 기반으로 LoRA(Hu et al., 2022) 랭크를 최적화합니다. 이 방식은 기본 모델, 클라이언트 데이터 세트, 자원 예산에 따라 개인화된 LoRA 모듈을 생성하고, 이를 통해 PEFT를 적용하여 개인화된 모델을 생성합니다. 이 방법은 다양한 기초 모델에서 직접 사용할 수 있어 유연성을 가집니다.

- **Performance Highlights**: 실험을 통해 FedP2EFT는 기존 비연합 학습 LoRA 랭크 선택 및 개인화 기법을 포함한 여러 연합 학습 접근 방식들과 비교해 높은 성능을 기록했습니다. 특히, 이 방법은 저자 개인화 수준을 최적화하며, 다양한 연합 학습 모델에 대한 성공적인 보완이 가능함을 보여 주었습니다. 이로 인해 개인화된 다국어 대형 언어 모델의 훈련에 있어서 더 나은 접근 방식을 제공할 수 있을 것으로 기대됩니다.



### Enhancing Reasoning to Adapt Large Language Models for Domain-Specific Applications (https://arxiv.org/abs/2502.04384)
Comments:
          NeurIPS 2024 Workshop AFM (Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning)

- **What's New**: 이 논문은 SOLOMON이라는 신경 영감을 받은 대형 언어 모델(LLM) 추론 네트워크 아키텍처를 소개하며, 이 모델이 도메인 특정 응용 프로그램을 위한 적응력을 크게 향상시킬 수 있다는 것을 보여줍니다. 반도체 레이아웃 설계에서의 사례 연구를 통해, SOLOMON은 일반 LLM이 전문 작업에 빠르게 적응할 수 있도록 하는 방법을 Demonstrate합니다. 실험 결과, SOLOMON은 LLM의 기본 성능을 뛰어넘으며 최신 추론 모델 o1-preview와 유사한 성능을 달성했습니다.

- **Technical Details**: SOLOMON의 아키텍처는 두 가지 신경 영감을 받은 이론에 기반합니다: 브레인 라이크 AGI와 자유 에너지 원칙입니다. 이 아키텍처는 다양한 LLM의 생각을 이용한 병렬 검색 엔진과 적응형 RAG 시스템을 통해 특정 작업에 대한 최적의 추론 계획을 발견할 수 있게 합니다. 기본 요소는 LLM 기반 시스템과 인간 조작 구성 요소를 포함하여 LLM의 안전성을 확보하고 기존 LLM의 반복적 미세 조정 필요성을 제거함으로써 다양한 전문 맥락에 맞춰 유연한 AI 시스템을 구축할 수 있게 합니다.

- **Performance Highlights**: SOLLOMON은 다양한 수준의 복잡성을 가진 25개의 레이아웃 설계 작업을 평가하였으며, 이는 기본 모양부터 복잡한 구조체까지 포함됩니다. 모델의 성능은 5개의 LLM과 비교하였고, SOLOMON은 기존 LLM에 비해 우수한 성능을 보였습니다. 결과적으로, 추론 능력이 LLM의 다양한 도메인 응용 프로그램에 대한 적응력을 향상시키는 데 중요한 역할을 함을 강조하고 있으며, 신경 과학에서 영감을 얻은 추론 능력의 개발이 향후 연구 방향으로 제시되었습니다.



### Sparse Autoencoders for Hypothesis Generation (https://arxiv.org/abs/2502.04382)
Comments:
          First two authors contributed equally; working paper

- **What's New**: HypotheSAEs는 텍스트 데이터(예: 헤드라인)와 타겟 변수(예: 클릭) 간의 해석 가능한 관계를 추론하는 일반적인 방법을 제시합니다. 이 방법은 텍스트 임베딩에서 희소 오토인코더(sparse autoencoder)를 훈련하여 데이터 분포를 설명하는 해석 가능한 특징을 생성하고, 타겟 변수를 예측하는 특징을 선택한 후, LLM을 통해 이러한 특징에 대한 자연어 해석을 생성하는 세 단계로 구성됩니다. 이 연구는 LLM 기반 방법들에 비해 더 적은 컴퓨팅 자원으로 더 많은 예측 가능한 가설을 생성하고 있습니다.

- **Technical Details**: HypotheSAEs는 다음과 같은 세 가지 주요 단계를 포함합니다. 첫째, 텍스트 임베딩에서 희소 오토인코더(SAE)를 훈련하여 해석 가능한 뉴런을 학습합니다. 둘째, Lasso와 같은 방법을 사용하여 타겟 변수를 예측하는 뉴런을 선택합니다. 셋째, 활성화된 입력 텍스트를 바탕으로 LLM을 사용하는 고충실도의 자연어 해석을 자동으로 생성합니다. 이러한 해석은 타겟 변수를 예측하는 가설 역할을 합니다.

- **Performance Highlights**: HypotheSAEs는 60개의 가설 중 45개가 유의미하다는 점에서 세 개의 실제 세계 과제에서 기존 세 가지 방법에 비해 훨씬 더 많은 유의미한 가설을 생성합니다. 또한, 이 방법은 최근의 LLM 기반 방법들보다 10배 이상 빠르고 비용이 저렴하며, 입력의 뉴런 활성화를 통해 한 번의 SAE 전방 패스(forward pass)로 모든 뉴런 활성화를 계산할 수 있어 효율적인 비용 절감이 가능합니다.



### Limitations of Large Language Models in Clinical Problem-Solving Arising from Inflexible Reasoning (https://arxiv.org/abs/2502.04381)
Comments:
          14 pages, 6 figures

- **What's New**: 최근 연구에 따르면, Large Language Models (LLMs)는 의학적 질문 응답 기준에서 인간 수준의 정확성을 달성했지만, 열린 임상 시나리오를 탐색하는 데 있어 한계를 보이고 있으며, 이는 LLM의 추론 능력에 대한 신뢰를 의심하게 만듭니다. 본 논문에서는 의료 추상화 및 추론 코퍼스 (M-ARC)를 소개하여 LLM의 임상 문제 해결에서 발생할 수 있는 실패 모드를 조사합니다. M-ARC는 LLM의 경직된 패턴 매칭과 관련된 유연성 부족을 드러내며, LLM이 실제 의사보다 성능이 낮다는 사실을 보여줍니다.

- **Technical Details**: M-ARC 질문은 미국 의사 면허 시험 (USMLE)에서 사용하는 객관식 형식을 기반으로 하며, 훈련 데이터에서 접한 적이 없는 임상 문제를 평가하기 위해 설계되었습니다. dataset은 전형적인 의학 텍스트에서 잘 다루어지지 않는 오픈형 응답을 포함하여 각 질문이 진단적인 결정에 충분한 정보를 제공하는지 여부를 평가합니다. 연구팀은 UCSF에서 의사를 모집하여 LLM과 비교하여 성능을 평가했습니다.

- **Performance Highlights**: M-ARC를 통해 검토된 여러 LLM 모델들은 50% 이하의 정확도로 수행했으며, 이는 LLM의 일반적인 의학적 추론 능력이 제한적임을 나타냅니다. 이 연구에서는 LLM이 자주 hallucination을 발생시키며, 자신의 답변에 대해 과신하는 경향을 보이는 것을 확인했습니다. M-ARC의 발견은 임상에서 LLM을 사용할 때 신중해야 함을 강조합니다.



### Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data (https://arxiv.org/abs/2502.04380)
Comments:
          26 pages, 15 figures, 11 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능 향상에 있어 데이터 다양성이 중요한 역할을 한다는 점을 강조합니다. 새로운 방법인 DaaR을 제안하여 LLM이 데이터의 다양성을 보상 신호로 학습하고, 도메인에 구애받지 않는 훈련 데이터를 자율적으로 선택하도록 유도합니다. 이 연구의 결과는 LLM의 전반적인 성능을 높이는 데 기여할 수 있는 방법론을 제시합니다.

- **Technical Details**: DaaR 방법은 LLM에 대한 새로운 미세 조정 프레임워크로, 외부 다층 퍼셉트론(MLP) 구조를 통합하여 LLM의 고유한 지식 및 가중치에 따라 조정 가능한 프로브(modules of probe)를 생성합니다. 이 프로브는 샘플링된 데이터의 의미적 엔트로피를 측정해 데이터의 적합성을 평가하고, 다양성을 극대화하는 데 필요한 데이터를 선택합니다. 각 데이터 세트는 10,000 개의 샘플로 구성되어 있으며, 이후 랜덤 베이스라인을 위해 8,000 개의 데이터를 uniform하게 샘플링합니다.

- **Performance Highlights**: DaaR 방법은 Qwen 및 LLaMA와 같은 다양한 최신 SOTA LLM에서 시행된 여러 실험을 통해 입증된 바와 같이 모델의 전반적인 능력을 상당히 향상시키는 것으로 확인되었습니다. 특히, 도메인 라벨이 부족한 데이터에서 강력한 성능 향상을 보여주며, 다른 SOTA 방법들은 이러한 어려운 시나리오에서 성능이 저하되는 경향을 보였습니다. 이로 인해 제안된 방법은 다양한 벤치마크에서 개선된 성과를 기록할 수 있는 잠재력을 지니고 있습니다.



### MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf (https://arxiv.org/abs/2502.04376)
- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)을 활용한 회의 위임 시스템의 프로토타입을 개발하고, 실제 회의 전사록을 바탕으로 종합적인 벤치마크를 구축하였습니다. LLM의 자연어 이해 및 생성 능력을 통해 회의의 맥락을 파악하고, 동적 대화에 참여할 수 있는 가능성을 탐구했습니다. 평가 결과, GPT-4/4o는 균형 잡힌 전략을 보여주며, Gemini 1.5 Pro는 더 조심스러운 반응을 보입니다.

- **Technical Details**: 회의 위임 시스템은 정보 수집, 회의 참여, 음성 생성의 세 가지 주요 요소로 구성됩니다. 정보 수집 단계에서는 사용자가 관심 주제와 자료를 미리 제공하거나, 개인 지식 기반과 실시간으로 연동하여 정보를 수집합니다. 이후 회의 참여 모듈에서 실시간으로 회의 상황을 모니터링하며, 발언 후 적절한 개입 시기를 결정합니다.

- **Performance Highlights**: 테스트 결과, 전체 약 60%의 응답이 주요 포인트를 잘 커버하고 있지만, 부적절하거나 반복적인 콘텐츠, 실시간 처리 지연 등의 개선이 필요함을 드러냈습니다. 랜덤 샘플링을 통해 수집한 데이터로는, 응답의 일관성과 주제 적합성을 높일 수 있는 방법을 모색하고 있습니다. 이러한 시스템은 개인이 회의에 참여하는 부담을 크게 덜어줄 잠재력을 가지고 있습니다.



### An Analysis for Reasoning Bias of Language Models with Small Initialization (https://arxiv.org/abs/2502.04375)
Comments:
          30 pages, 14 figures

- **What's New**: 이번 연구는 Transformer 기반 대형 언어 모델(LLMs)의 파라미터 초기화 스케일이 훈련 행동 및 작업 선호도에 미치는 영향을 조사합니다. 작은 초기화 스케일이 추론 작업에 대한 선호도를 높이는 반면, 큰 초기화 스케일은 메모리 작업을 선호하게 만든다는 결과를 발견하였습니다. 이러한 결과는 실제 데이터셋과 정교하게 설계된 앵커 초함수(anchor functions)를 통해 검증되었습니다.

- **Technical Details**: 연구에서는 LLM들이 자연어를 학습할 때 발생하는 추론 편향을 규명하기 위해, 작은 파라미터 초기화로 훈련된 GPT-2 모델을 사용하였습니다. 훈련 데이터에는 두 가지 상이한 수준의 추론 복잡성을 가진 언어 데이터가 포함되어 있으며, 특히 PrOntoQA 데이터셋에서 추론 패턴이 더 빠르게 학습되는 경향이 있음을 보여주었습니다. 이 과정을 통해 훈련 초기 단계에서 임베딩 공간에서의 토큰들이 더 차별화되어 학습되는 방식도 분석하였습니다.

- **Performance Highlights**: 실험 결과, 작은 초기화 스케일이 추론 작업에서는 학습 효율성을 극대화하도록 유도하며, 다른 매개변수 초기화 방식에 비해 메모리 작업과 추론 작업 간의 수렴 속도가 다르게 나타났습니다. 본 연구는 이러한 현상을 이론적 배경과 경험적 증거를 결합해 설명하며, LLM의 훈련 동역학에 대한 이해를 심화시킵니다. 결과적으로, 모델 초기화 전략 최적화에 대한 유용한 지침을 제공합니다.



### Mining Unstructured Medical Texts With Conformal Active Learning (https://arxiv.org/abs/2502.04372)
- **What's New**: 본 연구는 전자 건강 기록(EHR)에서의 정보 추출을 자동화하여 역학 감시를 효율적으로 수행하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 전문 지식의 의존도를 줄이면서도 예측 성능을 극대화할 수 있도록 설계되었으며, 기존의 복잡한 데이터 라벨링 과정을 간소화합니다. 200개의 수동 라벨링 데이터만으로도 복잡한 분류 문제에 대해 강력한 성능을 보여줍니다.

- **Technical Details**: 이 연구에서 제안된 Conformal Active Learning 프레임워크는 비구조적 텍스트에서 정보를 추출하는 데 사용됩니다. 초기 모델을 통해 예측을 시작하고, 재훈련을 통해 모델을 반복적으로 개선합니다. 이 과정에서는 레이블 조건부의 conformal prediction을 사용하여 각 예측의 불확실성을 정량화하여, 센서티브 데이터 보호와 동시에 모델의 신뢰성을 보장합니다.

- **Performance Highlights**: 제안된 프레임워크는 경량 모델을 사용하여 공공 건강 응답을 신속하게 할 수 있으며, 더 자원을 소모하는 깊은 학습 모델보다 경쟁력 있는 성과를 냅니다. 또한 환자 개인정보 보호를 위해 데이터를 외부로 전송하지 않고 최적의 성능을 유지합니다. 이러한 접근을 통해 의료 기관은 새로운 건강 위협에 대해 빠르고 효과적으로 대응할 수 있는 기회를 가집니다.



### DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization (https://arxiv.org/abs/2502.04370)
Comments:
          20 pages, 12 figures

- **What's New**: 이번 논문에서는 DreamDPO라는 새로운 프레임워크를 제안합니다. 이 방법은 텍스트 설명으로부터 3D 콘텐츠를 생성할 때 인간의 선호도를 직접적으로 최적화하여 3D 생성 프로세스에 통합합니다. 기존의 방법들이 생성된 콘텐츠가 인간의 선호도와 일치하지 않는 어려움이 있었던 반면, DreamDPO는 이러한 한계를 극복하기 위해 작성되었습니다.

- **Technical Details**: DreamDPO는 3단계 반복 최적화 과정을 통해 작동합니다. 첫 번째 단계에서는 서로 다른 Gaussian 노이를 적용하여 쌍(pairwise) 예제를 실시간으로 생성합니다. 두 번째 단계에서는 보상 모델(reward model) 또는 대형 멀티모달 모델을 이용하여 입력 텍스트 프롬프트와의 일치를 바탕으로 예제를 평가하고, 세 번째 단계에서는 쌍 선호도를 기반으로 하는 보상 손실(reward loss)을 계산합니다.

- **Performance Highlights**: 실험 결과, DreamDPO는 기존 방법들과 비교했을 때 더 높은 품질과 더 큰 제어 가능성을 가진 3D 콘텐츠를 생성함을 입증했습니다. 특히, DreamDPO는 데카르트의 다차원 품질 평가를 최적화하는 데 있어 현저한 성과를 보였으며, 13개의 최첨단 방법들과의 비교에서 최고의 양적 성과를 달성했습니다.



### Contrastive Token-level Explanations for Graph-based Rumour Detection (https://arxiv.org/abs/2502.04366)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 Graph Neural Network (GNN) 기반의 루머 탐지 모델에 대한 해석 가능성을 개선하기 위해 Contrastive Token Layerwise Relevance Propagation (CT-LRP)라는 새로운 프레임워크를 소개합니다. CT-LRP는 기존의 설명 기법의 한계를 극복하고, 개별 토큰 수준의 설명을 제공하여 모델의 예측을 보다 세분화하여 해석할 수 있도록 합니다. 이를 통해 모델 신뢰성과 투명성이 향상될 것으로 기대됩니다.

- **Technical Details**: CT-LRP는 Layerwise Relevance Propagation (LRP)와 설명 공간 파티셔닝 전략을 결합하여 클래스-specific 텍스트 구성 요소를 분리하고, 고차원 텍스트 임베딩의 의존성을 포착합니다. 기존의 GNN 설명 기법들이 노드나 엣지 수준의 인사이트에 그치는 반면, CT-LRP는 문장 구성 요소에 대한 세부적인 분석을 통해 더 높은 충실도와 정밀도를 제공합니다. 이 방법은 세 개의 공개 루머 탐지 데이터셋에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: CT-LRP는 세 가지 공개 데이터셋에 대한 실험에서 항상 신뢰할 수 있는 고품질의 설명을 생성하는 것으로 나타났으며, GNN 기반 설명 가능성의 새로운 기준을 설정합니다. 이는 AI 시스템의 투명성과 신뢰성을 향상시키며, 유해한 정보에 대응하는 윤리적이고 효과적인 AI 시스템 구축으로 이어질 것입니다. 이러한 발전은 이해관계자가 더 나은 결정을 내릴 수 있는 기반을 마련합니다.



### LLMs can be easily Confused by Instructional Distractions (https://arxiv.org/abs/2502.04362)
Comments:
          8 pages

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 "instructional distraction" 현상을 다루며, 이를 평가하기 위한 새로운 벤치마크인 DIM-Bench를 소개합니다. Instructional distraction은 입력이 지시어와 유사할 때 발생하는 혼란을 의미하며, LLM이 이러한 혼란스러운 상황에서 어떻게 반응하는지를 평가합니다. DIM-Bench는 20개의 카테고리로 구성되어 있으며, 재작성, 교정, 번역, 스타일 전환과 같은 지시 작업과 추론, 코드 생성, 수학적 추론 등 5개의 입력 작업을 포함합니다.

- **Technical Details**: DIM-Bench는 LLM의 instruction-following 능력을 평가하기 위해 설계된 새로운 벤치마크입니다. 이 벤치마크는 두 가지 차원에서 작업을 결합하여 LLM의 성능을 비교합니다. 특히, 연구에서 사용된 LLM들은 명시적인 프롬프트가 주어졌음에도 불구하고 instructional distraction에 대해 완전한 강건함을 보이지 않으며, LLM의 반응을 더욱 면밀히 분석할 필요성을 제기합니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들도 instruction-following 작업에서 사용자 의도를 제대로 따르지 못하는 경우가 많았습니다. 특히 question answering 작업에서 LLM은 지시 문맥을 무시하고 입력 질문에 대한 답변을 생성하는 경향이 강하게 나타났습니다. 이러한 발견은 LLM의 instructional distraction 상황에서의 성능 한계를 강하게 부각시키며, 향후 강건성을 향상시키기 위한 추가 연구의 필요성을 제안합니다.



### MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction (https://arxiv.org/abs/2502.04360)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG)에서 데이터를 정확하게 추출하기 위한 새로운 공격 프레임워크인 MARAGE를 소개합니다. MARAGE는 사용자 쿼리에 추가되는 적대적인 문자열을 최적화하여 RAG 데이터가 그대로 포함된 출력을 생성하도록 합니다. 이 방법은 기존의 수동 및 최적화 기반 공격보다 높은 일반화 및 효율성을 보여줍니다.

- **Technical Details**: MARAGE는 다중 모델의 기울기를 통합하여 최적의 문자열을 생성하는 연속 최적화 방법을 사용합니다. 또한, 초기 토큰에 가중치를 부여하여 모델이 RAG 데이터의 시작 부분을 우선적으로 고려하도록 합니다. 이를 통해 전체 RAG 데이터를 효과적으로 추출할 수 있으며, 이는 다른 접근 방식들과는 차별화된 전략입니다.

- **Performance Highlights**: 평가 결과 MARAGE는 여러 LLM 및 RAG 데이터 세트에서 기존의 수동 및 최적화 기반 방법보다 일관되게 우수한 성능을 보였습니다. 우리의 접근 방식이 모델의 내부 상태에 미치는 영향을 조사하는 probing 작업을 수행하였으며, MARAGE가 더 효과적인 이유를 분석하였습니다.



### Exploring Spatial Language Grounding Through Referring Expressions (https://arxiv.org/abs/2502.04359)
- **What's New**: 최근 비전-언어 모델(Vision-Language Models, VLMs)이 공간적 추론(spatial reasoning) 능력에서 한계가 있다는 점이 지적되었습니다. 본 연구에서는 전통적인 이미지 캡셔닝(image captioning) 및 시각적 질문 응답(Visual Question Answering) 대신, 지칭 표현 이해(Referring Expression Comprehension, REC) 과제를 새로운 평가 플랫폼으로 제안합니다. 이 과제를 통해 모호한 객체 탐지, 복잡한 공간 표현 및 부정 표현이 포함된 상황에서 VLMs의 공간 이해 및 기반 능력을 심층 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: 이 연구는 CopsRef 데이터셋을 사용하여 51개의 공간 관계를 분석합니다. 이를 바탕으로 LLaVA와 Grounding DINO라는 두 가지 인기 있는 VLM과 REC 전용 모델인 MGA-Net을 비교합니다. 분석을 통해 공간적 표현의 수가 VLM의 성능에 미치는 영향을 검토하며, 각 모델의 디자인 및 훈련 전략 차이에 따라 성능 차이를 측정하고 분석합니다.

- **Performance Highlights**: 분석 결과, 공간적 관계는 지칭 표현의 다른 속성과 결합될 때 더 정확한 기준을 제공합니다. 공간적 복잡성이 증가함에 따라 VLM의 성능이 변화하지만, 명시적 조합 학습(component)가 포함된 모델은 성능을 유지하는 경향이 있습니다. 모든 모델이 부정적인 공간적 관계 처리에 어려움을 겪지만 그 정도는 다양한 것으로 나타났습니다.



### Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives (https://arxiv.org/abs/2502.04358)
Comments:
          12 pages including references

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 기반으로 한 에이전트 시스템의 효율성을 위해 비율 분석(Asymptotic Analysis)의 필요성을 주장합니다. 자동화된 역할 분담이 일반적으로 직관적으로 이루어지며, 이는 인간 팀의 역할 배정과 유사하다는 점을 강조합니다. 그러나 저자들은 이러한 역할 분해가 최적화에서 얼마나 가까운지를 이해하기 위한 분석이 필요하다고 강조합니다.

- **Technical Details**: 비율 분석에서 LLM 프리미티브(primitive)를 중요 개념으로 제안하며, LLM의 전방 패스(forward pass)를 계산의 기본 단위로 취급합니다. LLM 기반 알고리즘(LbA)은 여러 LLM 기반 에이전트가 협력하여 작업을 완수하는 시스템을 의미합니다. 전통적인 비율 분석에서는 계산의 기본 단위를 원자적 작업(primitive)이라고 정의하는 반면, 저자는 LLM의 단일 실행을 해당 기본 단위로 간주합니다.

- **Performance Highlights**: 이 논문에서는 LLM 프리미티브를 활용하여 성능 분석을 수행하고, LLM 기반 시스템의 효율성을 측정하기 위한 새로운 접근 방식을 제시합니다. 여러 사례 분석을 통해 비율 분석이 가져오는 통찰력을 강조하며, 이러한 분석이 연구 및 개발 방향의 기초가 되어 궁극적으로 규모 확장에 기여할 것이라고 주장합니다.



### Reusing Embeddings: Reproducible Reward Model Research in Large Language Model Alignment without GPUs (https://arxiv.org/abs/2502.04357)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 보상 모델 연구에서 임베딩 기반(input) 사용을 제안합니다. 기존의 보상 모델 훈련이 복잡성과 계산 비용으로 인해 제한되는 점을 개선하기 위해, 임베딩을 이용한 방법으로 재현성을 높이고 훈련과 평가의 비용을 줄일 수 있음을 시사합니다. 특히, 이러한 접근 방식은 훈련 안정성을 개선하고, 하드웨어의 계산 요구를 감소시키며, 연구의 속도를 가속화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 저자들은 보상 모델에서 임베딩을 사용하는 방법과 전통적인 자연어 입력 방식을 비교합니다. LLM에서 자연어 입력으로 효율적인 품질 평가를 위해 임베딩 공간을 활용하는 것이 발전 가능성이 크다고 강조하며, 최근의 연구 결과로 수학적 추론 과제와 LLM 생성 콘텐츠의 안전성 및 유용성을 평가하는 데 이 방법이 효과적임을 나타냅니다. 또한, 보상 모델은 최소한의 하드웨어 자원으로 훈련할 수 있는데, 훈련 시간이 1~5분으로 짧습니다.

- **Performance Highlights**: 임베딩 기반의 보상 모델과 기존 LLM 기반 보상 모델 간의 성과 비교 실험을 통해 효용성을 평가했으며, 연구 결과 임베딩 모델이 해석 가능성과 정확성을 모두 갖춘다는 점이 강조되었습니다. 실제 실험에서는 특정 3계층 MLP 모델이 0.6M 미만의 파라미터로 효율적으로 작동하는 것을 보여주었고, 기존 연구들의 기준에 비해 훈련 및 평가 시간이 크게 단축됨을 입증하였습니다. 또한, 다양한 주석 품질 시나리오에 대한 실험을 통해 임베딩 기반 접근 방식의 강점을 입증하며, 이는 보상 모델링 연구의 향후 발전 방향에 중요한 기초가 될 것입니다.



### Open Foundation Models in Healthcare: Challenges, Paradoxes, and Opportunities with GenAI Driven Personalized Prescription (https://arxiv.org/abs/2502.04356)
- **What's New**: 최근 OpenAI의 GPT-4와 같은 상용 대규모 언어 모델(LLMs)의 성공에 대응하여, 개방형 비상용 LLM 및 AI 기초 모델(AIFMs)의 개발에 대한 관심이 증가하고 있습니다. 이 연구는 의료 분야에 대한 개방형 모델의 잠재력에 주목하고 있으며, 이러한 모델들을 통해 보다 효율적인 의료 데이터 분석 및 진단 지원이 가능하다고 주장합니다. 더불어, 이 논문에서는 의료 응용을 위한 최신 개방형 LLM 및 AIFM의 상태를 종합적으로 조사하고, 개인 맞춤형 처방 사례 연구를 통해 그 유용성을 평가합니다.

- **Technical Details**: 이 논문은 개방형 AIFMs에 대한 분류 체계를 소개하며, 의료 이미징, 임상 NLP, 의료 교육 등 다양한 의료업무에 대한 적용 가능성을 다룹니다. 이 연구에서는 LLaMA-2, LLaMA-3, Mistral 및 Meditron 등 여러 개방형 LLM의 성능을 비교하고, Retrieval-Augmented Generation(RAG)을 통합하여 성능을 향상시킬 수 있는 방법을 제시합니다. 문서 전반에서는 개방형 AI 모델의 정의 및 라이센스 문제, 비공식적인 정체성에 대한 논의도 포함되어 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 개방형 LLM은 RAG와 같은 기반 기술을 활용할 경우 상유형 모델에 필적하는 성능을 달성할 수 있다는 점이 강조됩니다. 개인 맞춤형 처방을 통한 평가에서는 전문가 임상의의 주관적 평가를 통해 이러한 개방형 모델이 환자에게 더 나은 치료 결과를 제공할 수 있는 잠재력을 가지고 있음을 확인했습니다. 그러나 이러한 강력한 LLM과 AIFMs의 오용 가능성에 대한 윤리적 고려도 필요하며, 의료 분야에서의 신중하고 책임 있는 구현이 강조됩니다.



### LLM-ProS: Analyzing Large Language Models' Performance in Competitive Problem Solving (https://arxiv.org/abs/2502.04355)
Comments:
          To be published in LLM4Code 2025 workshop proceedings

- **What's New**: 새로운 평가 기법인 LLM-ProS가 제시되었습니다. 이 기법은 세계적인 프로그래밍 대회인 ICPC 문제를 통해 최신 대형 언어 모델(LLM)의 성능을 평가합니다. 2011년부터 2024년까지의 166개의 문제로 구성된 정제된 데이터 세트를 통해 모델의 추론, 정확성 및 효율성을 벤치마킹하였습니다.

- **Technical Details**: 다양한 아키텍처와 훈련 방법론을 갖춘 5개의 모델(GPT-4o, Mistral Large, Llama-3.1-405B, o1-mini 및 o1-preview)을 평가했습니다. 모델들은 문제 해결을 위해 각기 다른 과정을 거쳐 최적화된 접근 방식을 통해 ICPC 문제를 해결하는 데 사용됩니다. LLM-ProS는 데이터 수집, 전처리, 모델 테스트, 해결책 생성 및 제출의 4단계로 구성됩니다.

- **Performance Highlights**: o1 모델들이 높은 정확성과 효율성으로 다른 모델들에 비해 두드러진 성과를 보였습니다. 또한, 모델 간의 성능 차이와 일반화 능력, 적응력의 차이를 발견하였고, 일부 모델들은 복잡하고 높은 난이도의 문제에서 어려움을 겪는 한계를 확인했습니다. 이 연구는 LLM의 설계 및 훈련 방법론 개선을 위한 기초를 제공합니다.



### Reviving The Classics: Active Reward Modeling in Large Language Model Alignmen (https://arxiv.org/abs/2502.04354)
- **What's New**: 이번 연구에서는 인간의 선호에 기반한 보상 모델링을 위한 정보 선택 전략을 제안합니다. 인식 공간의 탐색을 균형 있게 조정하고 보상 차이가 중간인 쌍 간의 의미 있는 비교를 수행하는 것을 목표로 합니다. 이 과정에서 Fisher 정보 기반 선택 전략을 도입하고, 고전 실험 설계 문헌의 이론을 깊은 신경망 모델의 최종 선형층에 적용합니다.

- **Technical Details**: 공식적으로, 본 연구는 Bradley-Terry (BT) 회귀 프레임워크를 사용하여 최적의 선호 레이블 주석 문제를 정의하고, BT 문맥 하에서 능동 학습과 고전 실험 설계 문헌 간의 관련성을 확립합니다. 또한, 고전 실험 설계에서 영감을 받은 다양한 스코어링 규칙들을 도입하고, 이를 대규모 정렬 문제에 적합하도록 조정합니다. 다양한 세팅과 데이터셋에 대해 8888개의 스코어링 알고리즘을 평가해 그 성능을 비교했습니다.

- **Performance Highlights**: 우리 방법은 여러 오픈소스 LLM과 데이터셋 간에 다른 선택 방법들과 비교하여 놀라운 성능과 높은 계산 효율성을 보여줍니다. 실험 결과, 고전 실험 설계 기법을 사용하면 다양한 세팅과 모델 아키텍처에서 뛰어난 성능과 강력한 안정성을 확보할 수 있습니다. 특히, 크로스 프롬프트 비교를 포함하는 능동 보상 모델링이 주석 효율성을 크게 향상시킴을 입증했습니다.



### CognArtive: Large Language Models for Automating Art Analysis and Decoding Aesthetic Elements (https://arxiv.org/abs/2502.04353)
- **What's New**: 본 연구에서는 대량 예술 분석을 자동화하기 위해 LLMs(대형 언어 모델)와 MLLMs(다중 모드 대형 언어 모델)의 가능성을 조사하고 있습니다. 이 모델들을 활용하여 15,000개 이상의 예술 작품을 분석함으로써 예술 작품의 기술적 특성과 표현 특성을 깊이 있게 이해하려고 합니다. 특히, 작품의 패턴이 시간에 따라 어떻게 진화하는지를 탐색하고, 이를 통해 예술적 표현을 해석하는 새로운 방법을 모색합니다.

- **Technical Details**: 이 연구는 GPT-4V와 Gemini 2.0을 활용하여 23명의 저명한 예술가의 작품을 분석할 것입니다. LLMs는 방대한 텍스트 데이터셋을 기반으로 훈련되므로 문헌 분석, 요약 및 질문 답변 등의 과제를 수행할 수 있습니다. 컴퓨터 비전(Computer Vision), 머신 러닝(Machine Learning) 및 자연어 처리(Natural Language Processing) 기술을 통해 디지털 이미지에서 의미 있는 정보를 추출하고, 예술적 스타일을 분류하고, 작가를 식별하며, 작품에 대한 설명을 생성합니다.

- **Performance Highlights**: 이 연구는 고속 대량 예술 분석을 자동화하는 데 있어 혁신적인 접근 방식을 제공합니다. 기존의 분석 방법을 넘어, LLMs는 예술 작품의 형식적 요소, 구성 및 문화적 중요성을 조사하여 이론적, 기술적, 그리고 미학적 요소를 분석하는 데 있어 보다 효율적이고 객관적인 방법을 제공합니다. 데이터 시각화를 통해 결과에 대한 직관적 이해를 돕고 있으며, 이를 통해 예술의 역사적 진화에 대한 새로운 통찰을 발견할 수 있는 기회를 제공합니다.



### Investigating the Robustness of Deductive Reasoning with Large Language Models (https://arxiv.org/abs/2502.04352)
- **What's New**: 이 논문은 LLM(대형 언어 모델)을 기반으로 하는 추론 방식의 강건성을 평가하는 첫 번째 연구를 제안합니다. LLM을 이용한 자동 형식화(autoformalisation) 방법의 설계 요소가 미치는 영향에 대한 체계적인 분석이 부족했습니다. 연구자들은 두 가지 유형의 섭동(perturbations) - 적대적 소음(adversarial noise)과 반사적 진술(counterfactual statements) - 을 활용하여 여섯 개의 섭동 데이터 세트를 생성하여 LLM 추론 방법을 분석했습니다.

- **Technical Details**: 저자들은 LLM 사용 시 적대적 소음이 자동 형식화에 영향을 미치고, 반사적 진술이 모든 접근 방식에 영향을 미친다는 것을 발견했습니다. 제안된 접근 방식은 섭동의 두 가지 가족(적대적 소음, 반사적 섭동)과 더불어 LLM 기반 추론 방법의 구조를 설명하는 방법론적 프레임워크로 요약됩니다. 이는 각 차원(추론 형식, 문법 구문, 오류 회복 메커니즘)에서의 대표적인 접근 방식을 포함합니다.

- **Performance Highlights**: 세 가지 차원에서 분석을 통해 LLM 기반 방법의 강건성에 대한 미세한 통찰력을 제공합니다. LLM의 오류 회복 메커니즘에서는 자세한 피드백이 전체 정확도를 향상시키지 않지만 구문 오류를 줄이는 데 기여하는 것으로 나타났습니다. 즉, LLM이 스스로 오류를 수정하는 데 어려움이 있음을 보여줍니다.



### NER4all or Context is All You Need: Using LLMs for low-effort, high-performance NER on historical texts. A humanities informed approach (https://arxiv.org/abs/2502.04351)
- **What's New**: 이 논문은 역사적 텍스트에서 인물, 장소, 사건 등을 자동으로 인식하고 분류하는 Named Entity Recognition (NER) 작업에 대한 새로운 접근법을 제시하고 긍정적으로 평가합니다. 기존의 NLP 도구들이 현대 언어로 작성된 텍스트에 최적화되어 있는 반면, 이 연구에서는 상용 Large Language Models (LLMs)를 사용하여 NER 성능을 크게 향상시킬 수 있음을 보여줍니다. NER에 대한 재정의가 필요하며, 역사적 맥락과 약간의 Persona 모델링을 포함한 것이 핵심 전략임을 주장합니다.

- **Technical Details**: NER은 역사 연구의 기초적인 작업으로, 다양한 언어, 장르 및 구조를 가진 역사적 문서들의 데이터에 적합한 도구가 부족합니다. 본 연구에서는 1921년 베데커 가이드에서 수동으로 주석을 단 명명된 개체를 기준으로 하여 LLM의 다양한 프롬프트 전략을 평가하였습니다. 프롬프트 전략으로는 맥락 정보 제공과 Persona 모델링을 통해 LLM이 순수한 언어적 접근에서 벗어나도록 유도하고, 무작위 예시의 수를 늘리는 방법 등이 포함되었습니다.

- **Performance Highlights**: 연구 결과, LLM은 컨텍스트 정보와 Persona 모델링이 포함된 프롬프트를 사용할 때 전통적인 NER 프레임워크인 flair와 spaCy를 최소한 동등하게 수행하며, 이에 대한 정확도가 현저히 향상됨을 보여줍니다. 더불어, 예상과 달리 무작위 예시가 없는 제로 샷 접근법이 피처 수 16개 미만에서는 몇 개의 예시가 있는 접근법보다 더 나은 성과를 보였습니다. 이러한 결과는 역사 연구자들이 기존 코드를 사용하지 않고도 NER에 접근할 수 있도록 해줍니다.



### CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidanc (https://arxiv.org/abs/2502.04350)
Comments:
          27 pages, 12 figures

- **What's New**: 기존 방법들은 Large Language Models (LLMs)를 텍스트 추론과 코드 생성 사이에서 효과적으로 조정하는 데 실패하였습니다. 이 연구에서는 LLM 코드 및 텍스트 생성을 안내하기 위해 CodeSteer라는 효과적인 방법을 제시합니다. 또한 37개의 상징적(Symbolic) 작업을 포함하는 종합 벤치마크인 SymBench를 구축하고, 12,000개의 다중 라운드 가이던스/생성 궤적 및 5,500개의 가이던스 비교 쌍으로 이루어진 데이터셋을 합성하였습니다.

- **Technical Details**: Llama-3-8B 모델은 새롭게 설계된 다중 라운드 감독 학습(supervised fine-tuning, SFT)과 직접 선호 최적화(direct preference optimization, DPO)를 통해 세밀하게 조정되었습니다. 최종적으로 생성된 모델인 CodeSteerLLM은 제안된 상징적 및 자기 답안 확인 도구를 추가하여 대형 모델의 코드/텍스트 생성을 효과적으로 안내합니다. 이 방식은 GPT-4o에 적용되었으며, 기존의 LLM을 능가하는 성능을 보여줍니다.

- **Performance Highlights**: CodeSteer를 통해 GPT-4o의 평균 성능 점수가 53.3에서 86.4로 상승하였고, 이는 OpenAI o1(82.7), o1-preview(74.8), DeepSeek R1(76.8)보다 모든 37개 작업에서 뛰어난 결과입니다. 게다가 CodeSteer는 학습된 GPT-4o에 대해 Claude, Mistral, GPT-3.5에서 평균 41.8의 성능 향상을 보여주며, 복잡한 작업에서도 상징적 컴퓨팅 기능을 완전히 활용합니다. 모델, 데이터셋, 코드는 제공된 링크에서 확인 가능합니다.



### Dynamic benchmarking framework for LLM-based conversational data captur (https://arxiv.org/abs/2502.04349)
- **What's New**: 이 논문은 대화형 에이전트를 평가하기 위한 동적 벤치마킹 프레임워크를 제안합니다. 기존의 평가 프레임워크가 단일 작업에 집중하는 반면, 이 연구는 다중 턴 대화의 역동적인 특성을 포착할 수 있도록 설계되었습니다. 이 프레임워크는 합성 사용자와의 상호작용을 통해 LLM 기반의 대화형 에이전트를 평가합니다.

- **Technical Details**: 제안된 프레임워크는 정보 추출(information extraction), 맥락 인식(context awareness), 적응형 참여(adaptive engagement) 등 핵심 차원에서 성능을 평가하기 위해 생성 에이전트 시뮬레이션(generative agent simulation)을 통합합니다. 다양한 사용자 행동의 측면을 시뮬레이션함으로써, 이 연구는 확장 가능하고 자동화된 벤치마킹 접근 방식을 제공합니다. 대출 신청(use case) 예제에서 실험 평가가 수행되었습니다.

- **Performance Highlights**: 실험 결과는 적응형 전략이 특히 모호한 응답을 처리할 때 데이터 추출 정확도를 향상시킨다는 것을 보여줍니다. 한 번의 추출(one-shot) 및 몇 번의 추출(few-shot) 조건에서 프레임워크의 효과가 입증되었습니다. 이 연구는 LLM 기반 대화형 에이전트를 평가하는 구조적이고 확장 가능한 접근 방식을 제공하여 실제 배포를 촉진합니다.



### Prompt-based Depth Pruning of Large Language Models (https://arxiv.org/abs/2502.04348)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구에서는 동적 깊이 가지치기(dynamic depth pruning)를 활용한 새로운 알고리즘인 PuDDing(Prompt-routed Dynamic Depth Pruning)을 제안합니다. 이 알고리즘은 특정 입력 프롬프트(prompt)에 기반하여 모델에서 어떤 Transformer 블록을 생략할지를 결정합니다. 실험 결과, PuDDing은 기존의 정적 깊이 가지치기 방법들보다 더 나은 성능을 발휘하며, 특정 작업에서의 정확도를 개선할 수 있습니다.

- **Technical Details**: 기존의 깊이 가지치기(depth pruning)는 고정된 생략 세트를 기반으로 하며, 이는 여러 작업에 맞게 조정할 수 없다는 한계가 있었습니다. PuDDing은 경량 라우터(router)를 훈련시켜 다양한 프롬프트에 따라 최적의 생략 세트를 선택하게 합니다. 이 과정에서는 데이터 중심의 방식으로 생략 세트를 구성하고, 새로운 작업 중심의 손실 함수(task-centric loss)를 사용하여 손실을 최소화하는 세트를 찾아냅니다.

- **Performance Highlights**: PuDDing은 제로샷(zero-shot) 상식 추론(common sense reasoning) 작업에서 4%p 이상의 정확도 향상을 달성했습니다. 이 알고리즘은 매 프롬프트마다 한 번만 라우터를 사용하여, 밀집 모델(dense model) 대비 1.2배 이상 속도 향상을 이루었습니다. 또한, PuDDing은 다양한 작업에서 더 향상된 정확도를 제공하면서도 연산 효율성 측면에서도 경쟁력을 갖추고 있습니다.



### SCALM: Detecting Bad Practices in Smart Contracts Through LLMs (https://arxiv.org/abs/2502.04347)
Comments:
          7 pages

- **What's New**: 이 논문에서는 스마트 계약(Smart Contract)에서 발생할 수 있는 나쁜 코드 관행(bad practices)을 체계적으로 분석하고, 35가지 유형의 문제를 다루는 첫 번째 연구 결과를 제시합니다. 새로운 프레임워크인 SCALM(Smart Contract Audit Language Model)을 통해 효과적으로 나쁜 관행을 탐지하고 해결할 수 있는 방법을 제안합니다. SCALM은 리트리벌 증강 생성(RAG) 및 스텝-백 프롬프팅(Step-Back Prompting) 기법을 결합하여 고급 개념을 추출하고, 상세한 감사 보고서를 생성합니다.

- **Technical Details**: SCALM은 두 가지 주요 모듈로 구성되어 있습니다. 첫 번째 모듈은 정적 분석(static analysis)을 사용하여 나쁜 관행이 포함된 코드 블록을 추출하고 이를 벡터로 변환하여 벡터 데이터베이스에 저장합니다. 두 번째 모듈은 RAG 방법론을 도입하고, 스텝-백 프롬프팅을 통해 코드에서 추상적이고 고수준의 개념을 추출하여 감사 보고서를 생성합니다. 이 과정에서 39,904개의 스마트 계약 데이터를 활용하여 잠재적 보안 위험이 있는 코드 스니펫을 수집합니다.

- **Performance Highlights**: SCALM은 다양한 LLMs 및 데이터 세트를 사용하여 수행된 실험에서 기존 도구들보다 나쁜 계약 관행을 탐지하는 능력에서 우수한 성능을 보여줍니다. 특히, RAG 컴포넌트는 SCALM의 전체 성능 향상에 중요한 역할을 했습니다. 이 프레임워크의 성과는 스마트 계약의 감사 작업에 있어 실질적인 개선을 제공할 것으로 기대됩니다.



### Multi-Lingual Cyber Threat Detection in Tweets/X Using ML, DL, and LLM: A Comparative Analysis (https://arxiv.org/abs/2502.04346)
- **What's New**: 이번 연구는 다언어 트윗 내 사이버 위협 감지에 중점을 두고 기존의 단일 언어별 접근 방식의 한계를 극복하고자 했습니다. 연구진은 영어, 중국어, 러시아어, 아랍어를 포함한 4개의 언어에서 데이터셋을 수집하고 라벨링하여, 다양한 고급 모델을 적용하는 새로운 방법론을 탐구했습니다. 결과적으로 Bi-LSTM 아키텍처가 모든 데이터셋에서 우수한 성능을 보여주며 다언어 사이버 위협 감지의 효과성을 입증했습니다.

- **Technical Details**: 연구는 세 단계로 진행되었으며, 첫 번째 단계에서 4개의 언어의 트윗 데이터를 수집하고 수작업 및 극성 기반 라벨링 방법을 통해 고품질 주석을 확보했습니다. 두 번째 단계에서는 각 데이터셋을 개별적으로 분석하여 기계 학습(ML) 및 심층 학습(DL) 모델의 성능을 평가했습니다. 세 번째 단계에서는 모든 데이터셋을 결합한 다언어 데이터셋을 만들어 DL 및 대형 언어 모델(LLM) 아키텍처를 적용하여 사이버 위협 탐지의 효능을 평가하였습니다.

- **Performance Highlights**: Machine Learning 모델 중에서는 Random Forest(RF)가 가장 높은 성능을 달성했지만, Bi-LSTM 아키텍처가 모든 데이터셋에서 다른 DL 및 LLM 아키텍처를 일관되게 초과하는 성능을 보였습니다. 이러한 결과는 Bi-LSTM이 다언어 사이버 위협 감지에 효과적임을 강조합니다. 본 연구에서 개발한 코드와 데이터는 연구자들이 사이버 위협 감지 문제를 해결하는 데 기여할 것으로 기대합니다.



### JingFang: A Traditional Chinese Medicine Large Language Model of Expert-Level Medical Diagnosis and Syndrome Differentiation-Based Treatmen (https://arxiv.org/abs/2502.04345)
- **What's New**: 이번 연구에서는 JingFang (JF)라는 새로운 TCM(Traditional Chinese Medicine) 대형 언어 모델을 개발하였습니다. JF는 전문가 수준의 진단 능력과 증상 구별 기반 치료를 제공하는 혁신적인 모델로, 기존 TCM 모델의 한계를 극복하고자 합니다. 연구팀은 의학 상담을 위한 다중 에이전트 동적 협력 사고 체계(MDCCTM)를 혁신하여 JF의 정확한 진단 및 치료 능력을 강화했습니다.

- **Technical Details**: JF 프레임워크는 세 가지 주요 모듈로 구성됩니다: TCM 상담, TCM 증상 구별, TCM 치료 추천입니다. JF는 MDCCTM을 통해 동적인 추론과 명확한 의사결정 기능을 구현하며, DSR(이중 단계 검색 체계)을 통해 실제 응용에 필요한 개선된 증상 구별 능력을 보유하고 있습니다. 각 에이전트는 환자의 상태에 따라 맞춤형 정보를 수집하고 분석하여 정확한 증상 구별과 치료 추천을 수행합니다.

- **Performance Highlights**: JF는 기존 TCM 모델에 비해 의료 상담의 완전성 및 정밀성을 크게 개선하여, 실제 환자 치료에서의 실용성을 강화하였습니다. 다수의 전문가 에이전트가 협력해 개인화된 의학적 접근을 제공하고, 효율적인 다단계 상담 과정을 거쳐 증상 구별과 치료 추천의 정확도를 높였습니다. JF는 전통 한의학의 현대적 적용을 가능하게 하여 인류 건강 보호 및 질병 치료에서 중요한 기여를 할 것으로 기대됩니다.



### Tutorial on Using Machine Learning and Deep Learning Models for Mental Illness Detection (https://arxiv.org/abs/2502.04342)
- **What's New**: 이 논문에서는 소셜 미디어를 통한 정신 건강 이해를 위한 기계 학습(machine learning) 및 심층 학습(deep learning) 방법에 관한 실용적인 가이드를 제공합니다. 연구자들이 우울증을 조기에 감지할 수 있도록 다양한 데이터 세트 처리 및 모델 평가 문제를 다루며, 실제 사례를 통해 이러한 기술을 효과적으로 적용하는 방법을 설명합니다. 또한 투명하고 윤리적인 기술 사용의 중요성을 강조하며, 신뢰할 수 있는 모델 구축을 위한 단계를 제시합니다.

- **Technical Details**: 이 연구는 데이터 준비, 모델 개발, 평가 지표에 대한 포괄적인 방법론을 제공합니다. Python 3와 pandas, scikit-learn, PyTorch, Transformers와 같은 인기 있는 라이브러리를 사용하여 데이터 처리 및 모델링을 수행하였습니다. 다양한 플랫폼에서 수집된 데이터를 기반으로 정신 건강 주제를 다루는 Sentiment Analysis for Mental Health 데이터 세트를 활용하여, 텍스트 청소, 정규화 및 TF-IDF 기반의 벡터화 과정 등을 통해 데이터 준비가 이루어졌습니다.

- **Performance Highlights**: 논문에서는 로지스틱 회귀모델을 포함하여 다양한 기계 학습 및 심층 학습 모델을 사용하여 정신 건강 상태를 분석하고 분류하였습니다. 각 모델은 데이터의 특정 측면을 탐색하기 위해 선택되었으며, 모델의 정확성 및 해석 가능성을 확보하기 위한 하이퍼파라미터 튜닝과 평가가 이루어졌습니다. 본 연구는 바람직한 성과를 달성하기 위해 여러 모델의 구현 코드와 자세한 성과를 GitHub에 공개할 예정입니다.



### Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficien (https://arxiv.org/abs/2502.05172)
- **What's New**: 이 연구에서는 Mixture of Experts (MoE) 모델과 밀집(dense) 모델의 공동 스케일링 법칙을 제시합니다. 이는 활성 파라미터 수, 데이터셋 크기, 전문가 수와 같은 중요한 요소를 포함하여 메모리 제약 하의 성능 분석을 제공합니다. MoE 모델이 밀집 모델보다 메모리 효율성이 높을 수 있다는 점을 발견했습니다. 이 연구는 MoE 모델의 최적 구성을 선택하기 위한 체계적인 프레임워크를 제공합니다.

- **Technical Details**: MoE 아키텍처는 게이팅 네트워크와 전문가 네트워크의 조합으로 제안되었습니다. 본 연구에서는 2.7B의 활성 파라미터와 5B의 총 파라미터를 가진 280개 이상의 실험을 통해 이론적 예측을 검증하였습니다. 이를 통해 MoE 모델의 최적 토큰-파라미터 비율과 전문가 수의 선택이 특정 계산 및 메모리 제약에 따라 달라짐을 보여줍니다. 또한, 학습 손실(Loss)과 데이터셋의 본질적 엔트로피 관계를 정의하여 제안된 법칙에 기반한 결론을 내렸습니다.

- **Performance Highlights**: MoE 모델은 동일한 계산 및 메모리 예산 하에서 실험을 통해 더 낮은 손실을 달성하여 실제적으로 더 높은 효율성을 입증했습니다. MoE 모델은 추론 시에도 더 높은 성능을 제공합니다. 기존의 밀집 모델보다 메모리 사용이 더 적고, 특정 하드웨어에서 메모리 제약을 받으면서도 더 나은 성능을 발휘합니다. 이러한 발견은 MoE 모델을 실제 대규모 훈련 시나리오에서 더욱 매력적인 선택으로 만듭니다.



### Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (https://arxiv.org/abs/2502.05171)
Comments:
          The model is available at this https URL. Code and data recipe can be found at this https URL

- **What's New**: 이 논문에서는 잠재 공간(latent space)에서 암묵적인 추론(reasoning)을 통해 테스트 시간에 계산을 확대할 수 있는 새로운 언어 모델 아키텍처를 연구했습니다. 이 모델은 반복적인 블록을 사용하여 테스트 시간 동안 임의의 깊이로 펼쳐져(unrolling) 작동합니다. 기존의 체인 오브 생각(chain-of-thought) 방식과는 달리, 특별한 훈련 데이터 없이도 작은 컨텍스트(window)에서 작동할 수 있으며, 단어로 쉽게 표현되지 않는 추론 유형을 포착할 수 있습니다.

- **Technical Details**: 인간은 문제를 해결할 때 자연스럽게 정신적인 노력을 더 기울이는 경향이 있습니다. 기존 언어 모델은 모델 크기를 확장하는 데 중점을 두었지만, 최근에는 테스트 시간 계산을 확장하여 모델의 추론 능력을 강화하는 방안을 탐색 중입니다. 장애물은 비싼 내부 추론을 항상 단일 언어로 표현해야 한다는 제약이며, 이는 비효율적일 수 있습니다. 새로운 모델은 반복 유닛을 추가하여 무한히 상태를 갱신하고 계산할 수 있도록 합니다.

- **Performance Highlights**: 이 모델은 3.5억 개의 매개변수와 8000억 개의 토큰으로 확장되었으며, 결과적으로 추론 벤치마크에서 성능이 개선되었습니다. 경우에 따라 성능 개선이 극적이며, 500억 개의 매개변수에 해당하는 계산 부하까지 이를 지원할 수 있습니다. 이 과정에서 모델은 더 넓은 가능성을 발휘할 수 있는 잠재력을 보여주고 있습니다.



### A Lightweight Method to Disrupt Memorized Sequences in LLM (https://arxiv.org/abs/2502.05159)
Comments:
          20 pages, 2 figures

- **What's New**: 이번 논문에서는 TokenSwap이라는 획기적인 방법을 제안합니다. 이 방법은 기존의 대형 언어 모델(LLM)이 저작권이 있는 내용을 그대로 재현할 위험을 줄이기 위한 가벼운 사후 처리(post-hoc) 기법입니다. TokenSwap을 통해 문법 관련된 토큰의 확률을 소규모 보조 모델의 확률로 교체하여 효과적으로 메모리 재현을 줄일 수 있습니다.

- **Technical Details**: TokenSwap 방법은 대형 모델의 메모리 문제를 해결하기 위해 쉽고 경제적인 접근법을 제공합니다. 이 방법은 DistilGPT-2와 같은 보조 모델를 사용하여 문법 관련 토큰의 확률을 바꾸는 과정을 포함하며, 추가적인 재훈련(retraining) 없이도 적용할 수 있습니다. 연구팀은 Pythia-6.9b와 LLaMA-3-8b 등의 상용 모델을 대상으로 광범위한 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, TokenSwap 방법은 널리 알려진 메모리 생성 사례의 비율을 10배까지 줄이는 데 성공하였습니다. 또한 이 방법은 하위 작업(downstream tasks)에 미치는 영향이 거의 없음을 확인했습니다. 이러한 결과는 TokenSwap이 현실 세계 시스템 사용자들에게 접근 가능하고 효과적인 솔루션을 제공함을 시사합니다.



### An Annotated Reading of 'The Singer of Tales' in the LLM Era (https://arxiv.org/abs/2502.05148)
- **What's New**: 이번 논문에서는 Parry-Lord 구술-공식 이론을 바탕으로 구술 서사 시가의 창작 및 전파 방식과 최신 대형 언어 모델(LLM) 및 생성적 인공지능(AI)의 메커니즘을 비교하고 있습니다. 특히 구술 시가를 생성하는 전통적 시인과 현대의 LLM의 유사성과 차이점을 분석하며, 이러한 통찰이 인류 사회 및 AI 정책에 미치는 영향을 논의합니다.

- **Technical Details**: 대형 언어 모델(LLM)은 transformer 아키텍처를 기반으로 하여 많은 양의 텍스트 데이터를 학습함으로써 자연어를 생성합니다. 본 논문은 LLM의 작동 메커니즘을 이해하려는 시도로, 구술-공식 이론과 비유적 구술의 특성을 설명합니다. LLM의 출력은 저선 저자와는 다른 특성을 지니며, 이는 구술 시가의 생성 방식과 비슷하다는 것을 보여줍니다.

- **Performance Highlights**: 구술 서사 시와 LLM 기반 자연어 생성 모두 즉각적으로 생성되며, 빠른 속도로 이루어지는 과정에서 전통적인 구술 시가의 특징과 유사한 양상을 보입니다. 저자들은 LLM이 단순히 저작 활동을 모방하는 것이 아니라, 구술 시가의 작법과 유사한 메커니즘을 가지고 있다고 주장합니다. 이는 LLM을 이해하는 데 새로운 관점을 제공하며, 특히 AI와 관련된 사회적 의사결정에 중요한 기초 자료가 될 수 있습니다.



### Lost in Time: Clock and Calendar Understanding Challenges in Multimodal LLMs (https://arxiv.org/abs/2502.05092)
Comments:
          Preprint

- **What's New**: 이번 연구는 멀티모달 대형 언어 모델(MLLMs)이 아날로그 시계와 연간 달력을 통해 시간을 해석하는 능력을 조사하였습니다. 연구팀은 ClockQA와 CalendarQA라는 두 가지 하위 데이터 세트를 구성하여, MLLMs의 시각적 인식 및 수치적 추론 능력을 분석합니다. 기존 연구와는 달리, 이번 연구는 시간과 날짜 관련 문제 해결에 중점을 두고 있어 새로운 접근법을 제시합니다.

- **Technical Details**: ClockQA 데이터 세트는 다양한 유형의 아날로그 시계를 포함하며, 주어진 이미지에서 시간을 정확히 읽는 능력을 평가합니다. 한편, CalendarQA는 연간 달력을 기반으로 하여 날짜와 관련된 질문에 대한 MLLMs의 응답을 시험합니다. 이 연구는 MLLMs의 시간 인식 능력을 평가하기 위한 제한된 규모의 데이터를 고안했으며, 각 모델의 성능을 정밀하게 분석하는 데 주력하였습니다.

- **Performance Highlights**: 초기 평가 결과, Gemini-2.0이 ClockQA에서 가장 높은 성능을 보였으나, 전반적인 성능은 부실하였습니다. 반면, GPT-o1은 CalendarQA에서 80%의 정확도로 뛰어난 성과를 기록했습니다. 그러나 일반적으로 두 작업 모두에서 낮은 성과가 나타났으며, 이는 MLLMs가 여전히 시간과 날짜 해석에서 어려움을 겪고 있음을 보여줍니다.



### Mitigating Unintended Memorization with LoRA in Federated Learning for LLMs (https://arxiv.org/abs/2502.05087)
- **What's New**: 이 논문은 federated learning (FL)에서 발생할 수 있는 데이터 기억 문제를 해결하기 위해 low-rank adaptation (LoRA)라는 간단하면서도 효과적인 미세 조정 전략을 제시합니다. 기존의 FL 훈련된 대형 언어 모델이 훈련 데이터의 구문을 기억하는 문제를 보여주었고, LoRA가 이러한 기억 현상을 최대 10배까지 감소시킨다는 사실을 입증하였습니다. 이 연구는 의학적 질문에 대한 답변을 정확히 제공하는 모델에서 다양한 Llama 2 및 3 모델에 대해 실험을 수행하였습니다.

- **Technical Details**: LoRA는 대형 언어 모델의 미세 조정 시 필요한 계산량과 메모리 요구 사항을 줄이는 방법으로서 사용됩니다. 이 방법은 훈련 가능한 매개변수의 수를 대폭 줄여주며, 훈련 데이터의 조건부 구문 기억 현상을 완화하는 데 효과적입니다. 실험에서는 두 가지 설정인 federated와 centralized 환경 모두에서 LoRA의 효능을 확인하였으며, 여러 가지 데이터 민감성 문제를 다루었습니다.

- **Performance Highlights**: LoRA는 고성능 유지와 함께 기억 현상을 효과적으로 감소시키는 데 성공하였음을 여러 모델에 걸쳐 확인하였습니다. 또한, gradient clipping, Gaussian noising, secure aggregation, Goldfish loss 등과 같은 다른 프라이버시 보호 기술과 결합하여 기록 수준의 데이터 보호를 강화할 수 있는 가능성을 보여주었습니다. 이는 FL에서의 데이터 민감성 문제 해결에 있어 매우 중요한 기여로 보입니다.



### Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures (https://arxiv.org/abs/2502.05078)
- **What's New**: 이번 연구에서는 Adaptive Graph of Thoughts (AGoT)라는 새로운 프레임워크를 소개합니다. AGoT는 기존의 체인(Chain of Thought)이나 트리(Tree of Thoughts) 기반 방법과 달리 복잡한 쿼리를 동적으로 분해하여 구조화된 하위 문제로 해결합니다. 이 접근법은 LLM의 추론 성능을 향상시키면서도 모델을 변경하지 않고 효율적입니다. 이러한 방식은 과중한 컴퓨팅 비용을 요구하지 않으며, 비용 효율적인 대안을 제공합니다.

- **Technical Details**: AGoT는 Directed Acyclic Graph (DAG)를 활용하여 복잡한 작업을 재귀적으로 평가합니다. 사용자가 정의한 한도 내에서 새로운 노드를 생성하고, 필요에 따라 자신의 추론 과정을 재귀적으로 적용할 수 있는 능력이 특징입니다. 이 프레임워크는 각 레이어의 목표를 안내하는 전략을 수립하며, 고품질 응답을 인식하고 불필요한 분기를 감소시키는 기능을 갖추고 있습니다. AGOT의 전체 구조는 각 레이어에서 노드 평가가 완료된 후에 다음 레이어를 생성하는 방식으로 발전합니다.

- **Performance Highlights**: AGoT는 다양한 벤치마크 테스트에서 최대 46.2%의 성능 개선을 달성했습니다. 특히, 과학적 추론 (GPQA) 작업에서 강화 학습 기반의 접근 방식과 비슷한 성과를 보이며, 최신의 반복적 접근 방법보다 성능이 뛰어났습니다. 이러한 결과는 AGOT의 동적 분해 및 구조화된 재귀가 LLM에서 보다 강력하고 일반화된 추론을 가능하게 하는데 기여함을 시사합니다.



### Paying Attention to Facts: Quantifying the Knowledge Capacity of Attention Layers (https://arxiv.org/abs/2502.05076)
- **What's New**: 이 논문은 단일 레이어 어텐션(transformers)만으로 구성된 모델이 데이터베이스에 포함된 정보를 어떻게 기억할 수 있는지를 선형 대수(linear algebra) 관점에서 탐구합니다. 각 데이터베이스에 3차 텐서(3-tensor)를 연결하고, 이 텐서의 랭크(rank)를 데이터베이스의 크기를 측정하는 방법으로 제안합니다. 또한, 어텐션 레이어와 관련된 3차 텐서를 정의하고, 데이터셋을 통해 두 랭크 간의 관계를 실험적으로 증명합니다.

- **Technical Details**: 우리는 데이터베이스와 어텐션 레이어 각각에 대해 3차 텐서를 구성하고 그 랭크에 대한 경계를 도출합니다. 기존 연구와 달리, 우리는 다층 퍼셉트론(MLP) 레이어가 아닌 어텐션 레이어의 역할에만 집중하며, 데이터베이스와 모델 크기에 대한 선형 대수적 측정을 제안합니다. 연구의 동기는 어텐션 모델이 ‘거대한 선형 구조’를 가지고 있다는 관측에서 도출되며, 사실 회상(factual recall)이 '가산 동형(additive motif)'으로 해결된다는 점을 강조합니다.

- **Performance Highlights**: 이 논문은 어텐션 레이어의 랭크가 데이터베이스의 랭크와 어떻게 연결되어 있는지를 보여주며, 이를 통해 매개변수 수를 증가시키지 않고 레이어 용량을 늘릴 수 있는 방법을 제안합니다. 실험은 장난감 모델(toy models)과 랜덤 데이터베이스에서 수행되었으며, argmax와 softmax의 랭크에 미치는 영향을 조사하여 모델 정확도를 평가하는 새로운 방법사례를 제안하고 있습니다. 이러한 결과는 어텐션 레이어 내에서 정보 저장 메커니즘을 이해하는 데 기여할 것으로 기대됩니다.



### Lightweight Operations for Visual Speech Recognition (https://arxiv.org/abs/2502.04834)
Comments:
          10 pages (double column format), 7 figures

- **What's New**: 이번 연구에서는 비디오 데이터에서 음성을 인식하는 Visual Speech Recognition (VSR)의 경량화 아키텍처를 개발하여 리소스 제한이 있는 장치에서도 실행 가능하게 만드는 데 중점을 두었습니다. 기존 모델들이 요구하는 높은 계산 비용 문제를 해결하기 위해 Ghost 모듈을 활용하여 모델의 복잡성을 줄이고, 성능 손실을 최소화하면서도 강력한 인식 능력을 갖춘 모델을 설계하였습니다.

- **Technical Details**: 연구에서 제안된 아키텍처는 Ghost 모듈을 사용하여 전통적인 합성곱 연산을 대체하고, 필요한 파라미터 수와 계산 비용을 줄였습니다. 또한, Partial Temporal Block이라는 일반적인 시간 차단 구조를 설계하여 입력 볼륨을 두 부분으로 나누고 각 부분에 대해 별도의 연산을 적용하는 방식을 채택하였습니다. 이러한 접근은 저전력 응용프로그램에 적합한 초경량 템포럴 합성곱 네트워크를 개발하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 제안된 모델들은 대규모 공개 데이터셋에서 우수한 VSR 성능을 보여주었으며, 하드웨어 요구사항이 크게 감소하여 다양한 실용적 응용이 가능함을 입증하였습니다. 대규모 실험 분석을 통해 경량화된 아키텍처가 성능을 크게 저하시키지 않으면서도 리소스 요구 사항을 획기적으로 줄일 수 있음을 확인하였고, 이는 여러 종합적으로 다양한 계산 능력을 가진 장치에서의 활용을 용이하게 합니다.



### ELITE: Enhanced Language-Image Toxicity Evaluation for Safety (https://arxiv.org/abs/2502.04757)
- **What's New**: 현재의 Vision Language Models (VLMs)은 악의적인 프롬프트로 인해 해로운 결과를 생성하는 취약성이 존재합니다. 기존의 안전성 벤치마크는 자동화된 평가 방법에 의존하나, 이러한 방법들은 암시적인 해로운 내용을 감지하는 데 어려움을 겪습니다. 이에 따라 저자들은 ELITE 벤치마크를 제안하며, 이는 VLM의 안전성을 평가하기 위한 고품질 평가 도구로 자리잡을 것입니다.

- **Technical Details**: ELITE 벤치마크는 독창적인 평가 방법인 ELITE evaluator에 기반하고 있으며, 이를 통해 다중 모드(multi-modal) 컨텍스트에서의 해로운 정도를 정확하게 평가할 수 있는 독성 점수를 포함하고 있습니다. 기존의 벤치마크에서 모호하거나 저품질의 이미지-텍스트 쌍을 필터링하고, 다양한 안전 및 위험 이미지-텍스트 쌍을 생성합니다. 이러한 방식은 VLM들이 특정하고 설득력 있는, 그러나 유해하지 않은 설명을 제공하는 문제를 해결하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과, ELITE evaluator는 기존의 자동화된 방법들과 비교할 때 인간 평가와의 정렬도에서 우수한 성과를 보였습니다. ELITE 벤치마크는 평가 품질과 다양성을 증진시켜, 안전하고 강력한 VLM 개발에 기여할 수 있는 중요한 도구로 자리 잡게 될 것입니다. 이는 실제 응용 프로그램에서의 안전성 위험을 평가하고 완화하는데 크게 기여할 것으로 예상됩니다.



### Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking (https://arxiv.org/abs/2502.04751)
- **What's New**: 이 논문에서는 이제까지의 검색 방법의 한계를 극복하기 위해 새로운 정보 탐색 패러다임인 HG-MCTS (Holistically Guided Monte Carlo Tree Search)를 도입하고 있습니다. HG-MCTS는 지식 메모리와 함께 진행되는 정보 수집 과정을 재구성하며, 사용자 쿼리의 복잡한 측면들을 포괄적으로 다루기 위한 적응형 체크리스트를 제공합니다. 이는 사용자 쿼리에 대한 보다 포괄적인 접근과 함께, 서로 다른 관점에서의 보상을 모델링하여 탐색의 질을 향상시킵니다.

- **Technical Details**: 복잡한 정보 탐색은 다양한 온라인 소스에서 정보 검색과 조직을 요구하는 작업으로 정의됩니다. 본 논문에서 제안하는 HG-MCTS는 전통적인 MCTS의 한계를 극복하기 위해 정보 수집 과정을 진전시켜 나가며, 전역적 지도와 다각적 피드백 모델을 포함합니다. 이 접근법은 각 단계에서의 지역적 탐색과 전반적인 지원 간의 균형을 유지하여, 관련된 모든 정보를 충분히 다룰 수 있도록 합니다.

- **Performance Highlights**: 현실 세계의 복잡한 정보 탐색 작업에 대한 실험 결과, HG-MCTS는 보다 철저한 지식 수집과 함께 더 정확한 최종 응답을 제공함을 보여주었습니다. 기존의 베이스라인들과 비교하여, 제안된 방법은 검색 경로의 중복성을 줄이고, 복잡한 사용자 쿼리의 모든 중요한 측면을 적절히 다루는 성능을 입증하였습니다.



### Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research (https://arxiv.org/abs/2502.04644)
Comments:
          work in progress

- **What's New**: 이 논문에서는 Agentic Reasoning이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 LLM이 외부 도구를 사용하는 에이전트를 통합하여 추론 능력을 향상시킵니다. 중요한 점은 웹 검색, 코드 실행, 구조화된 기억을 활용하여 복잡한 문제를 해결할 수 있도록 돕는다는 것입니다.

- **Technical Details**: Agentic Reasoning은 복잡한 문제를 다루기 위해 여러 단계의 추론을 수행할 수 있는 능력을 제공합니다. 이 과정에서 모형은 웹 검색 에이전트와 코드 실행 에이전트를 사용하여 실시간 데이터 검색 및 연산 분석을 수행합니다. 또한, Mind Map 에이전트는 지식 그래프를 구성하여 복잡한 논리 관계를 정리할 수 있는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, Agentic Reasoning 프레임워크는 PhD 수준의 질문(GPQA)에서 우수한 정확성을 기록하며, 실제 전문가 수준의 연구 작업에서도 효과적인 성과를 보여줍니다. 이러한 결과는 이 프레임워크가 지식 집약적 분야에서 생산성을 향상시키는 데 큰 잠재력을 가지고 있음을 시사합니다.



### Confidence Elicitation: A New Attack Vector for Large Language Models (https://arxiv.org/abs/2502.04643)
Comments:
          Published in ICLR 2025. The code is publicly available at this https URL

- **What's New**: 본 논문에서는 딥러닝 모델, 특히 대규모 언어 모델(LLM)의 적대적 공격에 대한 새로운 접근법을 제시하고 있습니다. 특히, 검은 상자 공격(black-box attacks) 환경에서 모델의 출력 확률(probabilities)을 활용하여 공격의 성공 가능성을 높일 수 있는 방법을 탐구하고 있습니다. 또한, 신뢰도 추출(confidence elicitation) 기술을 통해 모델의 불확실성을 파악하여 잘못된 분류를 유도하는 새로운 공격 벡터를 도출하였습니다.

- **Technical Details**: 제안된 방법은 기존의 하드 레이블 공격(hard-label attack) 기법과 비교하여 더 나은 성능을 보였으며, 세 가지 데이터 세트와 두 모델(LLaMA-3-8B-Instruct 및 Mistral-7B-Instruct-V0.3)에서 평가되었습니다. 공격 과정에서 신뢰도를 최소화함으로써 오분류의 가능성을 증가시킬 수 있으며, 이는 모델이 출력하는 확률을 기반으로 하는 방법론입니다. 연구 결과, 현재 LLM에서의 신뢰도는 잘 조정되고 오류가 없는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 접근법은 하드 레이블, 검은 상자, 단어 대체 기반 공격에서 최신 성능(state-of-the-art)을 달성하였습니다. 기존의 하드 레이블 최적화 기술(SSPAttack)과 비교할 때, 제 방법은 더 높은 공격 성공률(Attack Success Rates, ASR)을 제공하며, 더 적은 쿼리로도 높은 의미적 유사성을 유지합니다. 연구는 코드도 함께 제공하여 향후 연구 및 개발에 기여할 수 있도록 하였습니다.



### Position-aware Automatic Circuit Discovery (https://arxiv.org/abs/2502.04577)
- **What's New**: 이 연구에서는 언어 모델의 내부 메커니즘을 이해하기 위한 새로운 접근 방법인 위치 인식 회로 발견 방법을 소개합니다. 기존의 회로 발견 방법은 입력 토큰의 위치에 관계없이 모델의 구성 요소를 균일하게 고려하는 경향이 있었습니다. 그러나 이 방법은 각 위치에서 다른 계산이 필요하다는 사실을 간과하여 제한된 통찰력을 제공합니다. 본 연구에서는 이러한 제한을 극복하기 위해 위치을 고려한 두 가지 개선 사항을 제안합니다.

- **Technical Details**: 본 연구의 핵심은 위치 인식 에지 기여 패칭(PEAP) 방법으로, 회로를 구성하는 에지의 중요도를 각 위치에서 별도로 평가하여 해석의 정확성을 높입니다. 이를 통해 변수 길이 예제를 포함한 데이터셋에서 효율적으로 위치 인식 회로를 발견할 수 있도록 하는 데이터셋 스키마(schema) 개념도 도입하였습니다. 스키마는 유사한 의미를 갖는 토큰 범위를 정의하여 다른 길이의 예제에서 정보 집계를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 스키마를 사용하여 발견된 회로는 기존의 위치 불변 회로에 비해 더 나은 신뢰도(faithfulness)와 회로 크기 간의 trade-off를 보여줍니다. 특히, 자동으로 생성된 스키마를 적용한 회로는 수동으로 설계된 스키마에서 발견된 회로와 유사한 신뢰도 점수를 달성하였습니다. 이러한 결과는 위치를 인식하는 회로가 기본 메커니즘을 보다 정교하게 표현할 수 있음을 시사합니다.



### Self-Regulation and Requesting Interventions (https://arxiv.org/abs/2502.04576)
- **What's New**: 이번 논문에서는 LLM 에이전트의 메타인지 능력을 향상시키기 위한 오프라인 프레임워크를 제안합니다. 이 프레임워크는 효율적인 중재 요청을 위해 LLM 기반의 프로세스 보상 모델(PRS)과 탐색 강화 학습(tabular RL)을 결합하여 작동합니다. 연구에서는 제한된 중재 예산($C$) 안에서 중재 요청 시기를 결정하는 방법을 탐구하며, 이를 통해 비용과 효율성을 줄이는 방안을 제시합니다.

- **Technical Details**: 새로운 접근 방식은 세 가지 단계로 구성됩니다: 전이 모델 수집 및 PRM 학습, 보상 및 정책 검색, 정책 교육입니다. 시스템은 랜덤으로 트리거된 중재를 통해 상태 전이를 수집하고, 이를 기반으로 PRM을 학습합니다.이러한 방식은 정책을 각 보상 구성에 따라 재학습할 필요 없이 예산 제약을 준수하는 최적 경로를 계산하도록 합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 각 작업당 8회의 중재가 필요한 시스템과 비교하여 오히려 필요한 중재 횟수를 1회로 줄이면서도 유사한 성능을 달성했습니다. 이를 통해 LLM 에이전트가 스스로 조절하고 도움이 필요할 때 적절히 요청하는 방향으로 나아가고 있음을 보여줍니다.



### Towards Cost-Effective Reward Guided Text Generation (https://arxiv.org/abs/2502.04517)
- **What's New**: 이번 연구에서는 새로운 보상 모델 아키텍처를 제시하여 RGTG(Reward-Guided Text Generation)를 사용하는 과정에서 발생하는 테스트 시 오버헤드를 감소시켰습니다. 기존의 RGTG 방법은 각 후보 토큰마다 보상 모델을 여러 번 호출해야 했지만, 제안하는 구조에서는 단 한 번의 호출로 모든 가능한 후보 토큰의 점수를 효율적으로 생성할 수 있습니다. 이는 기존 방법들이 선호하는 하위 최적 시퀀스에 비해 동등하거나 더 나은 성능을 보장합니다.

- **Technical Details**: 제안된 보상 모델 아키텍처는 Bradley-Terry 손실(BT loss)을 사용해 단일 호출에서 전체 후보 토큰에 대한 점수를 제공합니다. 이 모델은 최적의 전체 시퀀스로 확장 가능한 접두사를 우선적으로 선택하도록 학습됩니다. 이를 통해 RGTG 방법의 디코딩 과정에서 발생하는 계산 복잡성과 지연 시간을 줄일 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 보상 모델이 기존의 RGTG 및 오프라인 RLHF 방법과 비교해 빠른 추론시간을 달성했음을 입증했습니다. 새로운 방법은 기존의 접근 방식과 유사한 성능을 가지고 있으면서도 더 적은 호출로 효율적인 성능을 보여줍니다. 이는 LLM(대형 언어 모델)의 텍스트 생성 작업에서 성배와 같은 응답 품질을 가져옵니다.



### Revisiting Intermediate-Layer Matching in Knowledge Distillation: Layer-Selection Strategy Doesn't Matter (Much) (https://arxiv.org/abs/2502.04499)
- **What's New**: 본 논문은 Knowledge Distillation (KD)의 효과적인 적용에서 흥미로운 현상을 발견하였습니다. 구체적으로, 중간 레이어 매칭(intermediate-layer matching)에서 레이어 선택 전략이 크게 중요하지 않음을 보여줍니다. 예상치 못한 매칭 전략인 역순 레이어 매칭(reverse layer matching)조차도 유사한 성과를 내는 것으로 나타났습니다. 이는 학생 모델의 관점에서 교사 레이어 간의 각도가 날카롭기 때문이라는 해석을 제공합니다.

- **Technical Details**: KD 방법은 일반적으로 예측 매칭(prediction matching)과 중간 레이어 매칭으로 나뉘며, 학생 모델은 교사 모델의 은닉 상태를 통해 추가적인 감독 신호를 받습니다. 본 연구에서는 6개의 데이터 세트에서 4가지 매칭 전략(정방향, 역방향, 랜덤, 모든-하나)을 실험하였고, 다양한 심도와 매개변수 초기화 설정을 탐색하였습니다. 연구 결과는 매칭 전략의 차이가 성능에 미치는 영향이 크지 않다는 일관된 증거를 제시하였습니다.

- **Performance Highlights**: 이 논문의 실험 결과, 중간 레이어 매칭이 없는 경우에 비해 모든 매칭 전략이 KD에 긍정적인 영향을 미치며, 예상치 못한 유사한 성과를 보여주었습니다. 또한, 깊이나 구조에 관계없이 KD의 효과는 중간 레이어 매칭을 통해 극대화되며, 여러 데이터 세트에 걸쳐 일관된 성과를 얻었습니다. 이러한 결과는 레이어 선택 전략이 성능에 미치는 영향을 재고하게 만듭니다.



### Training Language Models to Reason Efficiently (https://arxiv.org/abs/2502.04463)
- **What's New**: 이번 연구에서는 큰 추론 모델을 효율적으로 훈련시키기 위한 새로운 방법을 제안합니다. 이 방법은 강화 학습(强化学习) 기법을 사용하여 모델이 작업의 복잡도에 따라 추론 시 필요한 계산량을 동적으로 조절하도록 학습합니다. 이는 모델이 불필요한 계산 자원을 최소화하면서도 정확도를 유지할 수 있어, 경제성 및 사용자 경험, 환경 지속 가능성에 기여할 수 있도록 합니다.

- **Technical Details**: 연구진은 모델이 효율적으로 추론하는 법을 배우도록 강화 학습 정책 기법을 활용했습니다. 이를 통해 모델은 정답에 도달하기 위해 필요한 최소한의 토큰 수를 사용하여 추론 비용을 최소화할 수 있게 됩니다. 연구 결과, DeepSeek-R1-Distill-Qwen-1.5B 및 DeepSeek-R1-Distill-Qwen-7B 모델을 대상으로 한 실험에서는 정확도를 대체로 유지하면서도 추론 비용을 상당히 줄일 수 있었습니다.

- **Performance Highlights**: 7B 모델의 경우, American Invitational Mathematics Examination 2024에서 경쟁 기준점 대비 16%의 토큰을 절감하면서도 정확도를 약간 향상시킬 수 있었습니다. 또한, MATH 데이터셋에서는 정확도가 1% 약간 저하되지만 30%의 토큰을 줄였습니다. GSM8K에 대해서는 유사한 정확도를 유지하며 약 50%의 토큰 절감이 가능하여, 테스트 시 추론 비용을 동적으로 줄일 수 있음을 보여주었습니다.



### Primary Care Diagnoses as a Reliable Predictor for Orthopedic Surgical Interventions (https://arxiv.org/abs/2502.04423)
- **What's New**: 이번 연구는 기본 진료에서의 진단 정보를 기반으로 수술 필요성을 예측할 수 있는 가능성을 조사하였습니다. 이를 통해 레퍼럴(referral) 정확성을 높이고, 워크플로우(workflow)를 간소화하며, 환자에게 더 나은 치료를 제공할 수 있습니다. 연구에 사용된 데이터셋은 텍사스 대학교 타일러 건강 센터에서의 2086건의 정형외과 레퍼럴로, 기계 학습 모델을 통해 분석되었습니다.

- **Technical Details**: 기계 학습 모델은 Base General Embeddings (BGE)를 사용하여 의미적 추출을 수행하였고, 실제 적용 가능성을 높이기 위해 잡음 허용성 실험을 실시하였습니다. 또한 클래스 불균형(class imbalance) 문제를 해결하기 위해 오버샘플링(over-sampling) 기법을 적용했습니다. 최적화된 임베딩 모델은 예측 정확도(ROC-AUC: 0.874, Matthews Correlation Coefficient (MCC): 0.540)를 보여주며, 수술 개입이 필요한 환자를 효과적으로 구분할 수 있었습니다.

- **Performance Highlights**: 예측 모델링 분석을 통해 수술 요율은 11.27%에서 최적의 60.1%로 증가하였고, 이는 433%의 개선을 나타냅니다. 이러한 결과는 운영 효율성과 의료 수익에 대한 중요한 시사점을 제공합니다. 연구 결과는 레퍼럴 최적화가 기본 치료와 수술 치료 통합을 향상시킬 수 있음을 보여주며, 환자 요구 사항의 정확하고 적시 예측을 통해 지연 시간을 최소화하고, 수술 계획을 개선하며, 행정 부담을 줄일 수 있음을 강조합니다.



### KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inferenc (https://arxiv.org/abs/2502.04420)
- **What's New**: 본 논문에서는 KV cache quantization의 개선된 접근 방식을 제안합니다. 주목할 만한 점은 레이어별(transformer layer-wise) 민감성을 고려하여 KV cache의 양자화 오류를 최소화하는 방법론입니다. 저자들은 키(cache key)가 값(cache value)보다 더 중요하다는 사실을 강조하며, 이를 통해 더 효과적인 양자화가 가능하다고 주장합니다.

- **Technical Details**: 제안된 KVTuner 프레임워크는 하드웨어에 적합한 레이어별 KV 양자화 정밀도를 탐색합니다. 다중 목표 최적화(multi-objective optimization) 기법을 사용하여 KV cache의 효율적인 조합을 찾고, 온라인 추론 동안 사전에 검색된 구성을 직접 활용합니다. 또한, 계산 비용을 줄이기 위해 intra-layer KV 정밀도 쌍 pruning과 inter-layer clustering 기법을 도입하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, Llama-3.1-8B-Instruct 모델의 경우 거의 손실 없는 3.25비트 혼합 정밀도 KV cache 양자화를 가능하게 하였으며, Qwen2.5-7B-Instruct와 같은 민감한 모델의 경우 4.0비트 양자화가 이루어졌습니다. 다양한 컨텍스트 길이에 대해 KV8 양자화와 비교할 때 최대 추론(thoroughput) 성능이 38.3% 개선되었습니다.



### Understanding and Mitigating the Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks (https://arxiv.org/abs/2502.04419)
Comments:
          Technical report; 31 pages

- **What's New**: 최근 연구는 대형 언어 모델(LLMs)의 성능 향상을 위한 합성 데이터 생성의 필요성을 강조하고 있습니다. 그러나 이러한 모델들이 학습 데이터에서 반영한 편향이 합성 데이터에 전이되고 증폭될 수 있는 문제, 즉 '편향 계승(bias inheritance)'에 대한 체계적인 연구가 부족했습니다. 본 논문은 이러한 문제를 최초로 규명하고, 편향 계승의 이해와 분석, 완화 방안을 제시합니다.

- **Technical Details**: 연구는 원본 데이터와 LLM으로 증강된 데이터를 결합한 데이터셋으로 LLMs를 미세 조정하여 편향 비율에 따른 6가지 편향 유형의 영향을 분석했습니다. 실험을 통해 특정 편향이 드러나는 비율에 따라 자동 생성하고 실험을 진행함으로써 편향이 직접 관련된 분류 작업과 일반 생성 작업에 미치는 영향의 차이를 규명했습니다. 이 분석은 사회적 편향이 증강 데이터에 미치는 영향과 그로 인한 하향 성능 저하를 포함합니다.

- **Performance Highlights**: 실험 결과, 직접 관련된 과제에서 편향의 발생과 그로 인한 성능 저하는 특히 소수 그룹에서 두드러지며, 편향이 반복적 조정을 통해 확대됩니다. 또한 본 연구는 가치 불일치, 그룹 데이터 불일치, 데이터 분포 불일치의 세 가지 주요 요소를 파악하였고, 이를 해결하기 위한 세 가지 완화 전략(token-based, mask-based, loss-based)을 제안했습니다. 이러한 접근 방식은 다양한 작업과 편향에서 다르게 작용하여 편향 계승 문제 해결의 어려움을 강조합니다.



### Decoder-Only LLMs are Better Controllers for Diffusion Models (https://arxiv.org/abs/2502.04412)
- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 효율성을 향상시키기 위해 대형 언어 모델(Large Language Models, LLMs)의 의미 이해 능력을 결합한 새로운 방법을 제안합니다. 특히, 이 모델은 LLM의 디코더 전용 구조를 활용하여 텍스트 프롬프트의 의미를 더 잘 캡처할 수 있도록 설계된 LLMDiff-Adapter라는 네트워크 모듈을 도입합니다. 기존의 방법들이 텍스트 인코더에 의존해 왔다면, 우리의 접근법은 LLM의 블록별 표현을 통합하여 텍스트 인코딩을 생성합니다.

- **Technical Details**: LLMDiff-Adapter는 노이즈 제거 U-Net 구조의 크로스-어텐션 부분에 연결됩니다. 이를 통해 LLM에서 추출한 표현을 텍스트 인코딩 생성에 직접적으로 활용할 수 있으며, 이는 세밀한 의미와 단어 간의 맥락 의존성을 효과적으로 포착합니다. 이 방법은 텍스트-이미지 생성 모델에 LLM의 강점을 직접적으로 통합하는 모듈로서 작동하며, 다양한 생성 모델에 쉽게 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLMDiff-Adapter를 사용한 모델은 생성된 이미지의 품질, 논리적 일관성 및 텍스트 설명에 대한 포괄적 이해 측면에서 최첨단 모델을 초월하는 결과를 보였습니다. 세밀한 이미지 세부 사항과 사용자 의도를 잘 반영한 이미지 생성을 통해 이 모델은 텍스트-이미지 생성 품질을 크게 향상시켰습니다. 다양한 벤치마크에서의 비교 분석을 통해 LLMDiff-Adapter의 효과성을 입증하였습니다.



### Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing (https://arxiv.org/abs/2502.04411)
Comments:
          work in progress. arXiv admin note: text overlap with arXiv:2405.09673 by other authors

- **What's New**: 이번 연구는 다양한 작업에 맞게 파인튜닝된 Large Language Models (LLMs)를 더욱 강력한 모델로 통합하는 모델 머징을 다룹니다. 기존의 모델 평균화에서 발생하는 파라미터 충돌로 인한 성능 저하 문제를 해결하기 위해, 레이어 간 충돌 정도를 분석했습니다. 이를 바탕으로 최소한의 파라미터 충돌을 가진 레이어를 평균화하고, 중요한 충돌을 가진 레이어는 새로운 전문가 라우팅 방법을 통해 처리하는 접근 방식을 제안했습니다.

- **Technical Details**: 모델 머징(Mediator) 프레임워크는 파라미터 충돌의 정도를 정량화하며 이를 반영해 레이어를 조정합니다. 파라미터 충돌이 적은 레이어는 평균화하고, 상당한 충돌을 가진 레이어는 전문가 모델로서 라우팅하여 고유한 작업 지식을 보존하는 전략을 사용합니다. 또한, 작업 산술의 희소성을 활용해 여러 파인튜닝 전문가를 밀집 전문가와 희소 전문가로 분리하여 저장 공간을 절약합니다.

- **Performance Highlights**: 이 방법을 LLaMA와 Qwen 모델을 사용하여 다양한 파라미터 스케일에서 실험한 결과, 기존 방법과 비교했을 때 성능 개선이 두드러졌습니다. 우리의 통합 모델은 RTX 4090 GPU에서 7B × 4 LLM 앙상블과 유사한 성능을 보여주며, 리소스 제한이 있는 환경에서도 높은 성능을 발휘할 수 있음을 입증했습니다. 따라서 실세계의 추론 작업에서 더 적은 시스템 비용으로 뛰어난 성능을 달성할 수 있음을 확인했습니다.



### FAS: Fast ANN-SNN Conversion for Spiking Large Language Models (https://arxiv.org/abs/2502.04405)
- **What's New**: 이번 연구에서는 Spiking Large Language Models(스파이킹 대형 언어 모델)가 기존의 LLM(대형 언어 모델)들에게 훌륭한 대안이 될 수 있음을 보여주고 있습니다. 기존의 Spiking LLMs 생성 방법들은 성능 저하와 높은 계산 비용에 시달렸습니다. 이를 해결하기 위해 우리는 새로운 Fast ANN-SNN conversion strategy(FAS)를 제안하였습니다.

- **Technical Details**: FAS는 LLM을 스파이킹 LLM으로 변환하는 두 단계의 프로세스를 포함합니다. 첫 단계에서는 사전 훈련된 모델의 전체 매개변수를 미세 조정하여 처음부터 직접 교육할 필요가 없습니다. 두 번째 단계에서는 정밀도를 높이고 변환 오류를 줄이기 위해 코스-투-파인(calibration) 방법을 도입합니다.

- **Performance Highlights**: 우리의 실험은 언어 및 비전-언어(vision-language) 작업에서 4가지 LLM 규모에 걸쳐 진행되었습니다. FAS는 최첨단 성능을 달성하면서도 추론 지연(inference latency)과 계산 비용을 크게 줄였습니다. 예를 들어, FAS는 OPT-7B 모델보다 3% 더 높은 정확도를 8 timesteps로 달성하며 에너지 소비를 96.63% 줄였습니다.



### Can Large Language Models Capture Video Game Engagement? (https://arxiv.org/abs/2502.04379)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 대규모로 사전 훈련된 언어 모델(LLMs)이 비디오를 통해 인간의 정서를 감지할 수 있는지를 평가한 최초의 종합 연구입니다. 연구진은 20개의 1인칭 슈팅 게임에서 총 80분의 비디오 게임 영상을 활용하여 플레이어의 참여도를 예측하는 LLM의 능력을 조사했습니다. 특히, 다양한 실험을 통해 LLM 아키텍처, 모델 크기, 입력 모드, 프롬프팅 전략이 성능에 미치는 영향을 분석하고 있습니다.

- **Technical Details**: 본 연구에서는 LLM이 비디오의 시청자 참여를 정확하게 라벨링할 수 있는지를 조사하며, 이를 위해 2,400개 이상의 실험을 수행했습니다. 실험은 GameVibe 데이터셋에서 플레이어의 참여에 대한 연속 라벨을 기반으로 하여 진행되었습니다. 우리는 LLaVA와 GPT 계열의 최신 모델을 비교하면서, 프롬프팅 전략 및 데이터 프로세싱 방법이 LLM의 성능에 미치는 영향을 깊이 있게 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM은 여러 도메인에서 인간과 유사한 성능을 나타냈으나 연속 감정 라벨링에서는 일반적으로 부족한 결과를 보였습니다. 특히 GPT-4o 모델과 몇 가지 예제를 혼합하여 제공했을 때 평균 66%의 정확도를 기록했으며, 특정 게임에서는 47%까지 성능 향상을 이뤘음이 밝혀졌습니다. 이러한 결과는 LLM이 감정 라벨링에 대한 자동화의 가능성을 제시하며, 향후 연구 방향에 대한 로드맵을 제공하고 있습니다.



### PerPO: Perceptual Preference Optimization via Discriminative Rewarding (https://arxiv.org/abs/2502.04371)
- **What's New**: 이번 논문에서는 Perceptual Preference Optimization (PerPO)라는 시각적 분별력 최적화 방법을 제안합니다. PerPO는 generative pre-trained multimodal large language models (MLLMs)의 시각적 인지 문제를 해결하는 것을 목표로 합니다. 이 방법은 다양한 부정 샘플을 수집하기 위해 구별되는 보상을 활용하고, 이를 통해 인간의 시각적 지각 과정과 MLLMs를 정렬하고자 합니다.

- **Technical Details**: PerPO는 명확한 목표 진실을 기반으로 다수의 가설을 생성하고, 점차적으로 최상의 가설로 좁혀지는 인간의 시각적 인지 과정을 모방합니다. 이를 위해 이 논문은 empirical risk minimization(ERM) 원리를 기반으로 하며, 부정 샘플을 효과적으로 획득하기 위한 확장 가능한 구별 보상을 도입합니다. PerPO는 또한 리스트 기반의 선호 최적화를 통해 부정 샘플 간의 관계를 학습하여 출력 품질을 향상시킵니다.

- **Performance Highlights**: PerPO는 MLLMs의 시각적 분별력 능력을 크게 향상시키며, 생성 능력을 유지합니다. 이 방법은 부정적 보상 해킹 문제를 완화하고, 다양한 시각 과제에서 일관된 성능을 보이는 것을 목표로 합니다. MLLMs의 미래 연구 방향을 새롭게 제시할 것으로 기대됩니다.



### Getting More Juice Out of Your Data: Hard Pair Refinement Enhances Visual-Language Models Without Extra Data (https://arxiv.org/abs/2305.05208)
Comments:
          Accepted to NAACL 2025, main conference. 20 pages, 10 figures, 10 tables

- **What's New**: 이 논문에서는 CLIP 모델을 효과적으로 개선하기 위한 HELIP이라는 전략을 제안합니다. HELIP는 기존 데이터셋 내의 어려운 텍스트-이미지 쌍을 활용하여 추가 데이터 수집이나 대규모 재훈련 없이도 지속적으로 모델을 개선할 수 있도록 설계되었습니다. 이를 통해 리소스와 시간 비용을 줄이는 동시에 현재의 훈련 파이프라인에 쉽게 통합할 수 있습니다.

- **Technical Details**: HELIP는 'Hard Pair Mining' (HPM) 전략을 도입하여 각 텍스트-이미지 쌍을 개별적인 엔터티로 간주하여, 근접 텍스트-이미지 쌍을 정의하고 식별합니다. 기존의 CLIP 모델의 손실 함수에 하드 네거티브 마진 손실(HNML)을 추가함으로써, pair level의 유사성을 반영한 기하학적인 구조를 도입합니다. 이를 통해 HELIP는 어려운 데이터의 정보를 최대한으로 활용하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: HELIP는 ImageNet, CIFAR-10 및 CIFAR-100에서 SLIP 모델의 제로샷 분류 정확도를 각각 3.05%, 4.47%, 10.1% 향상시켰습니다. 또한, 7개의 세분화된 이미지 분류 데이터셋에서도 제로샷 및 선형 프로브 성능을 평균 8.4% 및 18.6% 향상시켰습니다. HELIP를 활용하여 기존의 CLIP 및 SLIP 모델들이 다양한 벤치마크 테스트에서 일관되게 성능을 개선함을 보여주었습니다.



New uploads on arXiv(cs.IR)

### Holistically Guided Monte Carlo Tree Search for Intricate Information Seeking (https://arxiv.org/abs/2502.04751)
- **What's New**: 이 논문에서는 이제까지의 검색 방법의 한계를 극복하기 위해 새로운 정보 탐색 패러다임인 HG-MCTS (Holistically Guided Monte Carlo Tree Search)를 도입하고 있습니다. HG-MCTS는 지식 메모리와 함께 진행되는 정보 수집 과정을 재구성하며, 사용자 쿼리의 복잡한 측면들을 포괄적으로 다루기 위한 적응형 체크리스트를 제공합니다. 이는 사용자 쿼리에 대한 보다 포괄적인 접근과 함께, 서로 다른 관점에서의 보상을 모델링하여 탐색의 질을 향상시킵니다.

- **Technical Details**: 복잡한 정보 탐색은 다양한 온라인 소스에서 정보 검색과 조직을 요구하는 작업으로 정의됩니다. 본 논문에서 제안하는 HG-MCTS는 전통적인 MCTS의 한계를 극복하기 위해 정보 수집 과정을 진전시켜 나가며, 전역적 지도와 다각적 피드백 모델을 포함합니다. 이 접근법은 각 단계에서의 지역적 탐색과 전반적인 지원 간의 균형을 유지하여, 관련된 모든 정보를 충분히 다룰 수 있도록 합니다.

- **Performance Highlights**: 현실 세계의 복잡한 정보 탐색 작업에 대한 실험 결과, HG-MCTS는 보다 철저한 지식 수집과 함께 더 정확한 최종 응답을 제공함을 보여주었습니다. 기존의 베이스라인들과 비교하여, 제안된 방법은 검색 경로의 중복성을 줄이고, 복잡한 사용자 쿼리의 모든 중요한 측면을 적절히 다루는 성능을 입증하였습니다.



### Enhancing Health Information Retrieval with RAG by Prioritizing Topical Relevance and Factual Accuracy (https://arxiv.org/abs/2502.04666)
- **What's New**: 이 논문에서는 건강 정보 검색(Health Information Retrieval) 모델을 향상시키기 위한 새로운 솔루션인 Retrieval-Augmented Generation (RAG)을 도입합니다. RAG는 지식 기반에서 건강 관련 문서를 효과적으로 검색하고, 이와 연계된 Generative Large Language Models (LLMs)를 사용하여 정보를 생성합니다. 이 방법은 정보검색 시스템이 주제적 적합성과 사실적 정확성을 동시에 고려하도록 도와줍니다.

- **Technical Details**: 제안된 모델은 세 단계로 이루어져 있으며, 첫 번째 단계에서는 사용자의 질의를 바탕으로 과학 문헌으로 구성된 지식 베이스에서 주제적으로 관련된 구문을 검색합니다. 두 번째 단계에서는 LLMs가 이러한 구문과 질의를 함께 처리하여 사실적 정확성을 평가하는 contextually rich text인 GenText를 생성합니다. 마지막 단계에서는 GenText와 비교해 문서의 주제적 관련성과 사실적 정확성을 평가하고 순위를 매깁니다.

- **Performance Highlights**: 이 모델의 실험 결과는 헬스케어 정보 검색의 정확성을 크게 향상시키는 데 성공했음을 보여줍니다. 벤치마크 데이터셋인 CLEF eHealth와 TREC Health Misinformation을 이용한 평가에서 높은 수준의 주제적 관련성과 사실적 정확성을 제공하며, 이는 건강 관련 정보 검색과 허위 정보를 줄이는 데 중요한 진전을 나타냅니다.



### Cross-Encoder Rediscovers a Semantic Variant of BM25 (https://arxiv.org/abs/2502.04645)
- **What's New**: 이번 연구에서는 Neural Ranking Models (NRMs)의 새로운 변형인 Cross-Encoder를 조사하여, 이들이 어떤 relevance features (관련성 특징)을 계산하는지, 그리고 이 정보가 어디에 저장되는지를 분석합니다. MiniLM의 Cross-Encoder 변형이 BM25의 의미론적 변형을 사용한다는 것을 발견했으며, 이는 문서 길이 및 용어 포화 효과를 조절하는 Transformer attention heads와 vocab에 대한 역 문서 빈도 정보를 인코딩하는 저차원(vector) 구성 요소를 포함합니다. 이러한 통찰력은 모델 수정(model editing)의 가능성을 열어주며, 투명성을 높이고 안전성 문제를 주소할 수 있는 기초를 제공합니다.

- **Technical Details**: 연구에서는 이전의 상관 연구를 바탕으로 기계적 해석 가능성(mechanistic interpretability) 방법을 사용하여 Cross-Encoder의 BM25 유사 신호 구현을 검증했습니다. 특히, path patching을 이용하여 BM25 유사 구성 요소를 계산하는 attention heads를 식별하고, 이 정보를 다른 relevance scoring heads에 전달함으로써 BM25 스타일 함수가 어떻게 통합되는지를 밝혔습니다. 또한, 저차원 벡터가 embedding matrix에서 IDF 정보를 포함하고 있다는 증거를 발견했고, 이를 통해 용어의 중요성을 조정할 수 있습니다.

- **Performance Highlights**: 본 연구의 주요 결과는 Cross-Encoder가 relevance scoring heads를 통해 BM25 스타일의 계산을 구현한다는 것입니다. 시스템의 경로를 역으로 추적하여 관련성을 계산하는 과정을 반영한 획기적인 이해가 이루어졌습니다. 이로써 NRMs의 성능 개선, 통제 가능성, 개인화 및 편향 완화의 기회가 확장될 수 있음을 시사합니다.



### MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilo (https://arxiv.org/abs/2502.04413)
- **What's New**: 이 논문에서는 의료 분야에서 지식 그래프(KG)를 활용하여 진단 및 치료 권장을 제공하는 MedRAG라는 새로운 Retrieval-augmented generation (RAG) 모델을 제안합니다. MedRAG는 비슷한 증상을 가진 질병들 간의 진단 차이를 체계적으로 구성한 4단계 계층적 진단 KG를 구축하여, 기존 모델의 다소 부족한 진단 정확성과 특이성을 개선합니다. 이 모델은 환자의 증상에 기반하여 더 정확하고 구체적인 의료 의사결정을 지원함으로써, 잘못된 진단의 위험을 줄이는 데 기여합니다.

- **Technical Details**: MedRAG는 진단 지식 그래프와 RAG의 통합을 통해, 환자 정보를 보다 명확하게 이해하고 그에 따른 후속 질문을 제시하는 기능을 갖추고 있습니다. 새로운 진단 KG 검색 모듈을 통해 입력된 환자와 관련된 모든 중요한 진단 차이를 식별하고, 대규모 언어 모델 내에서 이 정보를 결합하여 추론을 수행합니다. 이 과정은 증상들이 유사한 질병 간의 미세한 진단 차이를 구별할 수 있는 개선된 추론 능력을 제공합니다.

- **Performance Highlights**: MedRAG는 DDXPlus 데이터셋과 Tan Tock Seng 병원에서 수집된 CPDD 데이터셋을 통해 평가되었으며, 여러 최신 RAG 모델과 비교하여 잘못된 진단 비율을 줄이는 데 있어 우수한 성능을 보였습니다. 실험 결과에 따르면, MedRAG는 기존의 RAG 접근 방식들보다 높은 진단 정확성과 특이성을 제공하며, 다양한 LLMs에서 robust한 일반화 성능을 나타냈습니다. 또한, 이 모델은 진단 질문을 보다 효과적으로 생성하여 복잡한 의료 시나리오에서 의사결정 과정을 최적화하는 데 큰 기여를 할 수 있습니다.



New uploads on arXiv(cs.CV)

### FlashVideo:Flowing Fidelity to Detail for Efficient High-Resolution Video Generation (https://arxiv.org/abs/2502.05179)
Comments:
          Model and Weight: this https URL

- **What's New**: 본 논문에서는 FlashVideo라는 새로운 두 단계 프레임워크를 소개합니다. 이 프레임워크는 텍스트 프롬프트에 대한 비디오 생성의 충실도와 품질을 조정하기 위해 모델 용량과 함수를 분배하는 전략을 사용합니다. 첫 번째 단계에서는 낮은 해상도로 비디오를 생성하여 고속으로 처리하고, 두 번째 단계에서는 흐름 매칭을 사용해 낮은 해상도에서 고해상도로 세부적인 디테일을 생성합니다.

- **Technical Details**: FlashVideo는 첫 번째 단계에서 5억 개의 파라미터와 50개의 평가 단계를 사용하는 낮은 해상도(예: 270p) 생성에 중점을 둡니다. 이후 두 번째 단계에서는 2억 개의 파라미터로 구성된 가벼운 모델을 사용하여 1080p의 고해상도 비디오를 생성합니다. 이 과정에서 흐름 매칭을 통해 ODE(Ordinary Differential Equations)의 경로를 직선 형태로 유지하여, 새롭게 생성된 비디오에 대한 세부 사항을 효율적으로 통합합니다.

- **Performance Highlights**: FlashVideo는 VBench-Long에서 82.99의 최고 점수를 기록하며 기존 요구보다 개선된 평가 시간을 달성했습니다. 전체 해상도 생성 전, 사용자에게 초안을 미리 확인할 수 있는 기능을 제공합니다. 이로 인해 연산 비용과 대기 시간이 크게 단축되어 상업적으로도 더 유리한 방안이 됩니다.



### QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation (https://arxiv.org/abs/2502.05178)
Comments:
          Tech report. Project page: this https URL

- **What's New**: 본 논문에서는 Quantized Language-Image Pretraining (QLIP)을 소개합니다. QLIP은 최신의 복원 품질(reconstruction quality)과 제로샷 이미지 이해(zero-shot image understanding)를 결합한 시각적 토크나이제이션(visual tokenization) 방법입니다. 우리는 두 가지 목표(객체)가 결코 상충할 필요가 없음을 보여주었으며, 동적으로 손실(loss) 항을 조정하여 훈련 중에 이 두 가지 목표를 효과적으로 균형 잡을 수 있습니다.

- **Technical Details**: QLIP은 바이너리 구형 양자화(Binary Spherical Quantization) 기반의 오토인코더(autoencoder)를 통해 이미지와 언어의 정렬(objective)과 복원(reconstruction) 목표를 학습합니다. 두 가지 주요 도전 과제를 파악하고, 경쟁하는 손실을 균형 잡기 위한 자동 가중치 조정 기법을 도입하였습니다. 또한, 메모리 효율적인 Transformer 아키텍처를 기반으로 한 두 단계 훈련 방법을 제안하여 대량 배치 요구사항과 메모리 병목을 효과적으로 혼합할 수 있습니다.

- **Performance Highlights**: QLIP은 멀티모달(Multimodal) 이해 및 텍스트-조건(image) 이미지 생성(text-conditioned image generation)에서 뛰어난 성능을 보여줍니다. LLaVA 기반 모델에서 CLIP-only 기준에 비해 성능 저하가 미미하며, 텍스트-이미지 정렬에서 두드러진 개선을 나타냅니다. 최종적으로, QLIP은 언어, 이미지-텍스트, 텍스트-이미지 작업을 단일 모델로 처리할 수 있는 통합된 혼합 모달 오토회귀 모델을 가능하게 합니다.



### Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuray (https://arxiv.org/abs/2502.05177)
Comments:
this https URL

- **What's New**: 최근 대규모 비전-언어 모델(Vision-Language Models)이 진화하면서, Long-VITA라는 새로운 모델이 등장했습니다. 이 모델은 4K 프레임이나 1M 토큰까지 처리할 수 있는 능력을 갖추고 있어, 긴 맥락의 비주얼-언어 이해 작업에서 뛰어난 성능을 발휘합니다. Long-VITA는 공공 데이터셋에서 수집한 17M 샘플로 구성되어 있으며, 전처리 및 미세 조정을 통해 효과적으로 학습되었습니다.

- **Technical Details**: Long-VITA는 단계적으로 훈련하는 접근 방식을 채택하여, 언어 모델을 중심으로 하여 비전-언어 정렬과 일반 지식 학습, 긴 시퀀스 미세 조정의 두 가지 연속 단계를 포함합니다. 또한, 그 동안의 개발에서 무한한 이미지 및 텍스트 입력을 처리할 수 있는 컨텍스트-병렬 분산 추론과 로그잇 마스킹 언어 모델링 헤드를 구현했습니다. 이 방법론을 통해 Long-VITA는 오픈 소스 커뮤니티의 경쟁력 있는 기준으로 자리잡을 수 있을 것으로 기대됩니다.

- **Performance Highlights**: Long-VITA는 다양한 다중 양식 벤치마크에서 최첨단 성능을 보여주며, 특히 환각(hallucination) 및 비디오 이해를 평가할 때 이전 모델들 중에서 두각을 나타냅니다. 이 모델은 이미지 및 비디오 이해 작업에서 더욱 뛰어난 성능을 발휘하기 위해 대량의 오픈 소스 이미지-텍스트 및 비디오-텍스트 데이터를 활용합니다. Long-VITA의 전체적인 성능은 고품질 데이터셋 Comic-9K을 포함한 다양한 데이터셋에서 검증되었습니다.



### AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360{\deg} Unbounded Scene Inpainting (https://arxiv.org/abs/2502.05176)
Comments:
          Project page: this https URL

- **What's New**: AuraFusion360은 360° 무한 장면에서 고품질의 객체 제거 및 구멍 채우기를 가능하게 하는 참조 기반의 새로운 방법입니다. 이 방법은 (1) 정확한 폐색 식별을 위한 depth-aware unseen mask 생성, (2) 추가 학습 없이 정확한 초기 포인트 배치를 위한 Adaptive Guided Depth Diffusion(zero-shot 방식), (3) 다중 뷰 일관성을 위한 SDEdit 기반 세부 사항 향상을 포함합니다.

- **Technical Details**: 우리는 Gaussian Splatting을 사용하여 3D 장면을 표현하고 여러 뷰 정보를 활용하여 보지 못한 영역을 채우는 접근 방식을 제안합니다. Adaptive Guided Depth Diffusion(AGDD)을 통해 참조 뷰에서 비정렬된 점들을 탐지하여 유사한 지역에서 일관된 정교화를 이루어냅니다. 이 방법은 구멍 채우기를하기 전에 보이지 않는 지역을 먼저 복원하여 신뢰할 수 있는 인페인팅을 보장합니다.

- **Performance Highlights**: AuraFusion360은 기존 방법들을 크게 능가하여 시각적 품질을 향상시키면서 기하학적 정확성을 유지하는 성과를 거두었습니다. 특히, 360° 장면에서 다양한 관점 변화에 대응하여 매우 높은 일관성과 현실감을 제공하며, 새로운 360° 인페인팅 데이터셋을 포함하여, 향후 연구를 위한 기준점을 제시합니다.



### Fillerbuster: Multi-View Scene Completion for Casual Captures (https://arxiv.org/abs/2502.05175)
Comments:
          Project page at this https URL

- **What's New**: 이번 논문에서는 Fillerbuster라는 새로운 방법을 소개합니다. 이 방법은 대규모 멀티뷰 잠재적 확산 변환기(large-scale multi-view latent diffusion transformer)를 활용하여 3D 장면의 알려지지 않은 영역을 완성합니다. 기존 방법들이 한두 장의 사진으로는 누락된 영역을 생성하는 데 한계가 있었던 반면, Fillerbuster는 수백 개의 입력 프레임을 처리하여 비어 있는 영역을 효과적으로 채울 수 있습니다.

- **Technical Details**: Fillerbuster 모델은 고해상도 이미지와 카메라 포즈를 공동으로 모델링하여, 알려진 이미지와 카메라 포즈를 조건으로 하는 잠재적 공간에서의 인페인팅(inpainting) 작업을 수행합니다. 입력으로 많은 수의 이미지를 받아들이고 이러한 이미지로부터 누락된 정보를 복구합니다. 고대비적 시나리오 하에서도 유연하게 동작하며, 사용자가 어떤 콘텐츠가 알려져 있고 어떤 것이 부족한지를 지정할 수 있습니다.

- **Performance Highlights**: Fillerbuster 모델은 다양한 작업에서 유용성을 입증합니다. 예를 들어, 모델은 비어 있는 대규모 영역을 환상적으로 완성할 수 있으며, '캘리브레이션되지 않은 장면 완성(uncalibrated scene completion)' 작업에서 새로운 이미지 포즈를 복구하고 새로운 콘텐츠를 생성하는 성과를 나타냅니다. 또한, NeRFiller 데이터셋에서 멀티뷰 인페인팅 작업을 수행할 때 이전 연구들보다 품질과 일관성 측면에서 우수한 성능을 보여줍니다.



### VideoRoPE: What Makes for Good Video Rotary Position Embedding? (https://arxiv.org/abs/2502.05173)
- **What's New**: 본 논문에서는 Rotary Position Embedding (RoPE)의 1D 구조를 비디오에 효과적으로 적용하는 방법을 다룹니다. 기존 RoPE 변형들이 비디오의 복잡한 시공간 구조를 충분히 반영하지 못하는 문제를 지적하며, VideoRoPE라는 새로운 임베딩 방식을 제안합니다. VideoRoPE는 RoPE의 장점을 살리면서도 비디오 데이터의 3D 구조를 고려하여 포지셔널 정보를 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: VideoRoPE는 특정 기술적 요소로 구성되어 있습니다. Low-frequency Temporal Allocation (LTA)은 temporal modeling을 우선시하기 위해 높은 차원(저주파수)을 temporal axis에 할당합니다. Diagonal Layout (DL)을 통해 시각적 및 텍스트 토큰 간의 상대 위치를 유지하며, Adjustable Temporal Spacing (ATS)은 인접한 시각적 토큰 간의 상대적 시간 간격을 조정하여 다양한 스케일을 포착할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, VideoRoPE는 기존 RoPE 변형들에 비해 일관되게 우수한 성능을 보였습니다. 특히, V-NIAH 및 V-NIAH-D 작업에서 각각 +12.4 포인트의 성능 향상을 보였으며, LongVideoBench, MLVU, Video-MME 등 다양한 비디오 이해 작업에서도 성능을 높였습니다. 또한, VideoHallucer 벤치마크에서도 +11.9 포인트의 향상을 기록하여 효과성을 입증했습니다.



### Flopping for FLOPs: Leveraging equivariance for computational efficiency (https://arxiv.org/abs/2502.05169)
- **What's New**: 이번 논문은 기하학적 불변성을 신경망에 통합하여 파라미터 효율성을 높이는 동시에 계산 비용을 비교적 낮게 유지하는 새로운 동등 신경망(equivariant neural networks)을 제안합니다. 주로 컴퓨터 비전에서 흔히 사용되는 수평 대칭(invariance)인 플로핑(flopping) 불변성에 초점을 맞추고 있으며, 이는 특성 공간을 거울-대칭(mirror-symmetric)과 거울-비대칭(mirror-antisymmetric) 특성으로 매개변수화하는 방법을 사용합니다.

- **Technical Details**: 신경망의 각 레이어에 대칭을 하드코딩(hard-coding)하여 기하학적 불변성을 구현하는 것이 가능하며, 이러한 구조적 대칭 제약은 신경망의 파라미터 효율성을 향상시키는 것으로 알려져 있습니다. 그러나 일반적으로 더 많은 계산을 요구합니다. 이에 반해, 제안된 방법은 불변 신경망이 일반 신경망과 비교하여 파라미터 당 비슷한 FLOPs를 달성할 수 있음을 보입니다.

- **Performance Highlights**: 본 논문에서 제안하는 플로핑-불변 신경망은 ImageNet-1K 데이터셋에서 ResMLPs, ConvNeXts, ViTs와 같은 대중적인 비전 아키텍처들의 분류 정확도를 유지하면서도 필요한 FLOPs를 절반으로 줄이는 성과를 보여주었습니다. 모델 크기가 증가할수록 이러한 플로핑-불변 모델은 비슷하거나 개선된 성능을 달성하여 효율적인 신경망 설계를 위한 실용적인 해결책을 제공합니다.



### Multitwine: Multi-Object Compositing with Text and Layout Contro (https://arxiv.org/abs/2502.05165)
- **What's New**: 우리는 텍스트와 레이아웃에 의해 안내받으며 동시에 다중 객체를 조합할 수 있는 최초의 생성 모델을 소개합니다. 이 모델은 단순한 위치 관계(예: 옆에, 앞에)부터 복잡한 행동(예: 포옹, 기타 연주)까지 다양한 상호작용을 캡처하는 기능을 가지고 있습니다. 또한, 이 모델은 '셀카 찍기'와 같은 활동과 관련된 추가 소품도 자율적으로 생성할 수 있습니다.

- **Technical Details**: 본 모델은 객체 이미지, 텍스트, 객체별 바운딩 박스, 배경 이미지 및 전체 조합 영역을 정의하는 마스크를 포함하는 다중 모달 입력을 수용합니다. 우리는 다양한 다중 모달 데이터를 기반으로 훈련하여 장면 컨텍스트와 객체 관계를 이해하고, 이를 통해 섬세한 다중 객체 조합을 구현할 수 있도록 합니다.

- **Performance Highlights**: 이 모델은 조합 및 주제 기반 생성을 동시에 학습하여 텍스트 기반 객체 조합에서 고급 성능을 발휘합니다. 최종적으로, 이 모델은 최첨단 커스터마이징 및 생성 객체 조합 모델과 비교할 수 있는 성능을 자랑하며, 복잡한 상호작용을 효과적으로 다룰 수 있게 만들어졌습니다.



### Hummingbird: High Fidelity Image Generation via Multimodal Context Alignmen (https://arxiv.org/abs/2502.05153)
Comments:
          Accepted to ICLR 2025. Project page: this https URL

- **What's New**: Hummingbird는 다중 모달 맥락(multimodal context)을 바탕으로 고충실도를 유지하면서 다양한 이미지를 생성하는 최초의 확산 기반 이미지 생성기를 소개합니다. 이 모델은 객체 상호작용 및 공간적 관계와 같은 장면 속성을 정확하게 보호합니다. 또한, Hummingbird는 다중 모달 컨텍스트 평가기(Multimodal Context Evaluator)를 활용하여 세계적인 의미(Global Semantic)와 세부 일관성(Fine-grained Consistency) 보상을 동시에 극대화합니다.

- **Technical Details**: Hummingbird는 참고 이미지와 관련된 텍스트 지침을 바탕으로 이미지 생성을 고도화합니다. 이를 통해 고유한 장면 속성을 유지하며, 이미지 다양성을 극대화하는 데 중점을 둡니다. 주요 기술 요소로는 멀티모달 대형 언어 모델(MLLM)을 활용한 텍스트 기반 맥락 설명(Context Description) 생성이 포함됩니다. 이 과정에서 SDXL 확산 모델을 미세 조정(fine-tuning)하여 이미지 생성의 정밀도를 높입니다.

- **Performance Highlights**: Hummingbird는 MME 인식(MME Perception) 및 Bongard HOI 데이터세트에서 기존 모든 방법을 능가하는 것으로 실험을 통해 입증되었습니다. 이 모델은 이미지의 다양성과 충실도를 동시에 달성하여 VQA와 HOI와 같은 장면 인식 작업에서 강력한 성능을 보여줍니다. Hummingbird는 나아가 ImageNet 및 OOD 변종에서 일관되게 다른 모든 방법보다 높은 성능을 발휘했습니다.



### LP-DETR: Layer-wise Progressive Relations for Object Detection (https://arxiv.org/abs/2502.05147)
Comments:
          7 pages, 4 figures

- **What's New**: 본 논문에서는 LP-DETR(Layer-wise Progressive DETR)이라는 새로운 접근 방식을 제시합니다. 이 방법은 multi-scale relation modeling을 통해 DETR 기반 객체 탐지를 향상시킵니다. Relation-aware self-attention 메커니즘을 도입하여 객체 쿼리 간의 학습 가능한 공간적 관계를 생성하고, 디코더 레이어 전반에 걸쳐 다양한 관계 규모를 균형 있게 학습합니다.

- **Technical Details**: 이 연구에서 제안한 LP-DETR는 DETR 스타일의 탐지기 구조에 기초하고 있으며, ResNet-50 및 Swin-L 백본을 사용하여 COCO 2017 데이터셋으로 실험하였습니다. 모델은 객체 쿼리 간의 관계를 모델링하는 self-attention 메커니즘을 통해 지역적(local) 특성과 전역적(global) 관계를 효과적으로 캡처합니다. 우리의 프로그레시브 디자인은 모델이 디텍션 파이프라인 전반에 걸쳐 진화하는 공간적 종속성을 효과적으로 포착할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, LP-DETR은 12에폭에서 52.3% AP, 24에폭에서 52.5% AP를 달성하며, Swin-L 백본을 사용하였을 때는 58.0% AP로 더욱 향상되었습니다. 제안된 프로그레시브 관계 모델링은 수렴 속도와 탐지 정확도 모두를 개선하는 데 기여하는 것으로 나타났습니다. 이러한 결과는 레이어별 관계 모델링의 중요성을 뒷받침하고, 향후 객체 탐지 연구를 위한 유망한 방향성을 제공합니다.



### Counting Fish with Temporal Representations of Sonar Video (https://arxiv.org/abs/2502.05129)
Comments:
          ECCV 2024. 6 pages, 2 figures

- **What's New**: 이 논문은 수산자원 관리 및 보존을 위해 근본적으로 중요한 연어 체공량을 정확하게 추정하는 방법을 제안합니다. 기존의 고해상도 이미징 소나(sonar) 기술을 활용한 연어 수 카운팅 방법은 컴퓨터 비전 처리와 호환됩니다. 저자들은 복잡한 비디오 프레임에서 데이터를 분석하는 대신, 여러 프레임을 단일 이미지로 압축한 시간적 표현인 에코그램(echogram)을 사용하여 연어 수를 계산하는 경량 작업을 제안합니다.

- **Technical Details**: 제안된 방법은 ResNet-18 모델을 사용하여 200프레임 시간 창 내에서 에코그램으로부터 직접적으로 상류 및 하류의 연어 수를 예측합니다. 각 에코그램 이미지는 200픽셀 너비로 설정되며 단일 전방 전달 패스를 통해 연어의 이동 수를 예측합니다. 또한, 저자들은 도메인에 특화된 이미지 증강 기법과 약하게 감독된 훈련 프로토콜을 제안하여 결과 개선을 위해 예비 생성된 주석을 사용합니다.

- **Performance Highlights**: 모델은 알래스카 케나이 강의 대표 데이터에서 23%의 수 카운팅 오류율을 달성하였습니다. 이는 더 계산 비용이 많이 드는 추적 기반 접근 방식과 거의 비슷한 성능을 보여줍니다. 이 접근 방식의 가능성을 강조하며 향후 연구의 유망한 영역을 탐색할 수 있는 기초를 제공합니다.



### Self-supervised Conformal Prediction for Uncertainty Quantification in Imaging Problems (https://arxiv.org/abs/2502.05127)
- **What's New**: 이 논문은 이미지 복원 문제에서의 불확실성을 수량화하기 위한 새로운 접근법을 제안합니다. 기존의 conformal prediction 방법은 막대한 양의 ground truth 데이터에 의존해야 하는 반면, 이 연구에서는 Stein's Unbiased Risk Estimator (SURE)를 활용한 self-supervised conformal prediction 방법을 통해 ground truth 없이도 불확실성을 수량화할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 관측된 노이즈 측정값으로부터 직접적으로 스스로 보정하는 방식으로 작동합니다. 이는 특히 선형 이미지 역문제에서 효과적이며, 현대의 self-supervised 이미지 복원 기법과 결합되어 더욱 강력한 성능을 발휘합니다. 저자는 Gaussian noise 모델을 전제로 하여, 점 추정치(Estimator)와 확률적 신뢰 구간(region C(Y))의 구성을 다룹니다.

- **Performance Highlights**: 실험 결과, 이미지 노이즈 제거(image denoising)와 비블라인드 블러링(non-blind deblurring) 작업에서 제안된 방법이 매우 정확하게 작동하며, ground truth 데이터에 의존한 supervised conformal prediction의 결과와 유사한 수준의 성능을 보여줍니다. 이는 ground truth 데이터 접근이 어려운 실전 상황에서도 유용성이 크다는 것을 의미합니다.



### Lost in Time: Clock and Calendar Understanding Challenges in Multimodal LLMs (https://arxiv.org/abs/2502.05092)
Comments:
          Preprint

- **What's New**: 이번 연구는 멀티모달 대형 언어 모델(MLLMs)이 아날로그 시계와 연간 달력을 통해 시간을 해석하는 능력을 조사하였습니다. 연구팀은 ClockQA와 CalendarQA라는 두 가지 하위 데이터 세트를 구성하여, MLLMs의 시각적 인식 및 수치적 추론 능력을 분석합니다. 기존 연구와는 달리, 이번 연구는 시간과 날짜 관련 문제 해결에 중점을 두고 있어 새로운 접근법을 제시합니다.

- **Technical Details**: ClockQA 데이터 세트는 다양한 유형의 아날로그 시계를 포함하며, 주어진 이미지에서 시간을 정확히 읽는 능력을 평가합니다. 한편, CalendarQA는 연간 달력을 기반으로 하여 날짜와 관련된 질문에 대한 MLLMs의 응답을 시험합니다. 이 연구는 MLLMs의 시간 인식 능력을 평가하기 위한 제한된 규모의 데이터를 고안했으며, 각 모델의 성능을 정밀하게 분석하는 데 주력하였습니다.

- **Performance Highlights**: 초기 평가 결과, Gemini-2.0이 ClockQA에서 가장 높은 성능을 보였으나, 전반적인 성능은 부실하였습니다. 반면, GPT-o1은 CalendarQA에서 80%의 정확도로 뛰어난 성과를 기록했습니다. 그러나 일반적으로 두 작업 모두에서 낮은 성과가 나타났으며, 이는 MLLMs가 여전히 시간과 날짜 해석에서 어려움을 겪고 있음을 보여줍니다.



### DCFormer: Efficient 3D Vision-Language Modeling with Decomposed Convolutions (https://arxiv.org/abs/2502.05091)
- **What's New**: 이 논문에서는 DCFormer라는 새로운 3D 의료 이미지 인코더를 소개하고 있습니다. 기존의 3D VLM(Visual-Language Models)들이 자주 사용하는 ViT(Vision Transformers)나 3D 합성곱(convolutions)에 비해 더 효율적이고 낮은 계산 비용으로 3D 이미지를 처리할 수 있는 방안입니다. DCFormer는 3D 합성곱을 깊이, 높이, 너비를 따라 세 개의 병렬 1D 합성곱으로 분해하여 공간 정보를 유지하면서 계산 비용을 현저히 줄입니다.

- **Technical Details**: DCFormer는 3D 합성곱을 1D 합성곱으로 분해하는 혁신적인 설계를 채택하여 성능과 효율성을 동시에 만족시키는 구조입니다. 본 논문에서는 DCFormer를 CLIP 기반의 비전-언어(Vision-Language) 프레임워크에 통합하여 CT-RATE 데이터셋에서 18가지 병리의 제로샷(multi-abnormality detection) 탐지를 실험하였습니다. 결과적으로 DCFormer는 기존 모델에 비해 적은 매개변수와 연산으로 우수한 성능을 보였습니다.

- **Performance Highlights**: DCFormer는 CT-RATE 데이터셋에서 다양한 모델들과 비교해 높은 정확도와 F1 점수를 기록했습니다. 특히 DCFormer-Tiny 변종은 62.0%의 정확도와 46.3%의 F1 점수를 달성하며, ViT와 TransUNet과 같은 기존 모델에 비해 적은 자원으로 더 뛰어난 성능을 보였습니다. 이는 DCFormer가 의료 분야에서의 3D VLM으로의 활용 가능성을 높이며, 향후 임상에 배포할 수 있는 효과적인 솔루션이 될 것으로 기대됩니다.



### Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images (https://arxiv.org/abs/2502.05066)
- **What's New**: 본 연구는 기존 이미지 생성 모델들이 생성하는 시각적 콘텐츠에 포함된 NSFW 텍스트 문제를 새롭게 조명합니다. 특히, Diffusion Models (DMs)와 Vision Auto-Regressive Models (VARs) 모델들이 아이디어를 구현하는 과정에서 발생하는 비속어, 인종 차별적 언어 등의 유해한 텍스트 생성에 면역력이 없음을 보였습니다. 연구진은 이러한 문제를 해결하기 위해 CLIP 텍스트 인코더의 안전성 파인튜닝을 제안하고, 이를 위해 커스텀 데이터셋을 활용했습니다. 마지막으로, ToxicBench라는 새로운 오픈 소스 벤치마크를 제안하여 NSFW 텍스트 생성 평가를 위한 기준을 제공하고자 합니다.

- **Technical Details**: Diffusion Models (DMs)와 Vision Auto-Regressive Models (VARs)은生成 모델로서,이미지 내에 텍스트를 삽입하는 기능을 갖추고 있지만, 이로 인해 NSFW 텍스트가 생성되는 위험이 있습니다. 기존 NSFW 완화 기법들은 이미지나 언어 도메인에서 효과적인 반면, 삽입된 NSFW 텍스트를 다루기에는 부족하여 텍스트 생성의 전반적인 품질 저하를 초래합니다. 이를 해결하기 위해 연구진은 CLIP 기반 텍스트 인코더의 안전성 강화 방법을 채택하여, 유해한 텍스트의 생성을 줄이는 동시에 이미지 품질을 유지할 수 있도록 하였습니다.

- **Performance Highlights**: 본 연구에서 제안한 접근 방식은 기존의 NSFW 기법들보다 효과적으로 유해한 텍스트 생성을 저해하는 동시에 무해한 텍스트의 생성 품질을 유지합니다. ToxicBench benchmark는 유해한 프롬프트의 데이터셋과 신규 메트릭스를 포함하여 안전성을 검증하는 경량화된 파이프라인을 제공합니다. 이러한 평가 도구는 향후 멀티모델 생성 모델의 안전성을 높이는 연구에 기여할 것으로 기대됩니다.



### Differentiable Mobile Display Photometric Stereo (https://arxiv.org/abs/2502.05055)
Comments:
          9 pages

- **What's New**: 이번 논문에서는 Differentiable Mobile Display Photometric Stereo (DMDPS)를 제안합니다. 기존의 DDPS는 고정된 데스크톱 설정에 의존했지만, DMDPS는 스마트폰을 사용하여 더 실용적인 접근을 제공합니다. 이 시스템은 모바일 앱을 통해 패턴을 동시에 표시하고 고품질의 HDR 이미지를 캡처할 수 있습니다.

- **Technical Details**: DMDPS는 물리 기반의 조명 방식으로, 모바일 기기의 카메라와 디스플레이를 활용하여 다양한 조명 조건에서 장면을 캡처합니다. 이를 통해 현실 세계의 3D 프린트된 물체를 촬영하고, 차별화 학습 과정(differentiable learning process)을 통해 디스플레이 패턴을 학습합니다. 시스템은 HDR 이미지를 사용하여 표면 법선(surface normals)과 반사율(albedos)을 재구성하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: DMDPS의 유효성은 3D 프린트된 데이터셋과 떨어진 나뭇잎의 첫 번째 데이터셋을 통해 보여집니다. 나뭇잎 데이터셋은 재구성된 표면 법선과 반사율을 포함하여 컴퓨터 그래픽스 및 비전을 넘어 향후 연구를 위한 기초 자료를 제공합니다. DMDPS는 실용적인 물리 기반의 조명 방법으로 한 단계 발전했다고 믿습니다.



### GaussRender: Learning 3D Occupancy with Gaussian Rendering (https://arxiv.org/abs/2502.05040)
- **What's New**: GaussRender는 3D voxel 기반 모델의 훈련을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이 방법은 3D voxel 표현을 2D 관점으로 투영하고, Gaussian splatting을 사용하여 공간적 의존성을 도입함으로써 예측의 정확성을 높입니다. 이를 통해 기존 아키텍처를 수정할 필요 없이 성능을 개선할 수 있습니다.

- **Technical Details**: 기존 3D 점유 모델은 voxel 예측의 공간적 관계를 무시하고 있습니다. GaussRender는 이러한 한계를 해결하기 위해 3D-to-2D reprojection 손실을 도입하여 예측된 voxel 표현을 2D 시점으로 프로젝트합니다. Gaussian splatting을 효과적으로 사용하여 무게를 줄이고, 빠른 렌더링을 가능하게 하며, 도시 장면에서의 공간적 의미 체계를 강화합니다.

- **Performance Highlights**: 여러 벤치마크에서 GaussRender는 기존 3D 점유 모델의 성능을 일관되게 향상시켰습니다. TPVFormer, SurroundOcc, Symphonies와 같은 모델에서의 성능 향상이 확인되었으며, 복잡한 주행 장면에서도 우수한 결과를 보여줍니다. 이를 통해 GaussRender는 3D 점유 작업에서의 강건성과 다양성을 강조합니다.



### MindAligner: Explicit Brain Functional Alignment for Cross-Subject Visual Decoding from Limited fMRI Data (https://arxiv.org/abs/2502.05034)
- **What's New**: 이번 연구에서는 MindAligner라는 새로운 프레임워크를 소개합니다. 이는 제한된 fMRI 데이터로부터 다양한 개체 간의 뇌 디코딩을 가능하게 하는 기능적 정렬 프레임워크입니다. MindAligner는 Brain Transfer Matrix (BTM)를 학습하여 새로운 피실험자의 뇌 신호를 기존 광범위한 데이터에서 학습한 모델로 변환할 수 있습니다. 기존의 방법들보다 더 향상된 성능을 제공하는 동시에 신경과학적 통찰력도 제공합니다.

- **Technical Details**: MindAligner의 핵심 구성 요소는 Brain Functional Alignment 모듈입니다. 이 모듈은 서로 다른 시각적 자극 아래에서 피실험자의 뇌 신호를 부드럽게 정렬함으로써 기능적으로 동등한 피질 영역을 매핑할 수 있도록 합니다. 연구진은 신호 수준 재구성 손실과 잠재적 정렬 손실을 통합한 다계층 뇌 정렬 손실을 설계하여 충분하고 세밀한 정렬을 달성할 수 있게 하였습니다. 이러한 접근 방식은 기능적 해석 가능성을 높이고, 뇌 영역 수준의 정보 변동성을 드러내는 데 기여합니다.

- **Performance Highlights**: 실험 결과, MindAligner는 데이터가 제한된 환경에서도 기존의 최신 방법들을 능가하는 성과를 보였습니다. 또한, Brain Transfer Matrix를 활용하여 뇌 기능적 분석을 수행한 결과, 초기 시각 피질에서는 피실험자 간 유사한 활동이 나타났지만, 기억과 공간 탐색과 관련된 고위 시각 피질에서는 현저한 상호 피실험자 변동성이 관찰되었습니다. 이 연구는 뇌 디코딩에 있어 혁신적인 접근법을 제시하며, 효과적인 새로운 주체 적응을 통한 향후 가능성을 열어줍니다.



### Trust-Aware Diversion for Data-Effective Distillation (https://arxiv.org/abs/2502.05027)
- **What's New**: 이번 논문에서는 데이터셋 증류(Dataset Distillation) 문제를 해결하기 위해 Trust-Aware Diversion (TAD) 방법을 제안합니다. 기존 방법은 모든 샘플이 완벽하게 레이블이 지정되어 있다고 가정하여 실제 환경에서 잘못된 레이블로 인한 문제를 간과했습니다. TAD는 신뢰할 수 있는 샘플에 초점을 맞추고, 불신 샘플을 재조정하여 효과적으로 데이터를 증류(substituting)하는 것에 중점을 두고 있습니다.

- **Technical Details**: TAD 방법은 상위 루프(outer loop)와 하위 루프(inner loop)라는 두 개의 반복 최적화 프레임워크로 구성됩니다. 상위 루프는 데이터를 신뢰할 수 있는 공간과 그렇지 않은 공간으로 나눕니다. 하위 루프는 불신 샘플을 재조정(recalibrating)하여 유용한 데이터로 변환하고, 이 과정을 통해 각 루프가 상호 보완적으로 작용하여 지속적으로 신뢰할 수 있는 샘플의 공간을 확장합니다.

- **Performance Highlights**: CIFAR-100의 40% 대칭(mislabeled) 잘못된 레이블을 가진 데이터에서 TAD는 기존 방법들보다 뛰어난 성과를 보였습니다. IPC(Images Per Class)가 10일 때 TAD는 41.5%의 정확도를 달성하여 ATT의 32.6% 및 두 단계 기준선(36.5%)보다 유의미하게 높은 수치를 기록했습니다. IPC가 50일 경우 TAD는 44.2%의 정확도를 기록하여 기존 방법들을 초월하는 성능 향상을 보여줍니다.



### OccGS: Zero-shot 3D Occupancy Reconstruction with Semantic and Geometric-Aware Gaussian Splatting (https://arxiv.org/abs/2502.04981)
- **What's New**: OccGS는 지오메트리(geometry)와 시맨틱(semantic) 정보를 결합하여 3D 점유율(reconstruction)을 효율적이고 효과적으로 추정하는 새로운 프레임워크입니다. 이 방법은 멀티센서 데이터를 활용한zero-shot 방식으로, 시맨틱 언어 모델(vision-language models)에서 추출한 시맨틱 정보를 사용하여 이를 달성합니다. 또한, CumGaussian-to-3D voxel splatting 기법을 통해 고품질의 장면 이해 및 복원을 가능하게 합니다.

- **Technical Details**: OccGS는 시멘틱(semantic) 및 지오메트릭(geometric) 인식을 통합하여 Gaussians를 구성합니다. LiDAR 포인트에 의해 안내되는 지오메트리를 활용하여, 다양한 센서에서 수집된 원시 데이터를 기반으로 합니다. 이는 고품질의 점유율 예측과 함께 장면의 전체적인 의미를 복원하는 데 필요한 재구성이 가능합니다.

- **Performance Highlights**: 실험 결과에서 OccGS는 자기 지도(self-supervised) 방법들보다 우수한 성능을 보였고, 완전 지도 학습(fully supervised) 접근법과 비슷한 성과를 달성했습니다. 더불어, cross-dataset 실험에서 뚜렷한 open-vocabulary 및 zero-shot 일반화 능력을 보여줍니다. 이로 인해, 3D 점유율 복원 분야에서 기존 방법들의 성능을 초월하거나 동등한 성과를 달성하였습니다.



### Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights (https://arxiv.org/abs/2502.04975)
- **What's New**: 이 논문에서는 Neural Architecture Search (NAS)의 한계를 해결하기 위한 새로운 방법으로, 훈련이 필요 없는 이미지를 분류하는 정확도 추정 방법을 제안합니다. 제안된 방법은 Fisher Information을 기반으로 하여, 네트워크 아키텍처를 훈련하지 않고도 기대되는 정확도를 추정할 수 있게 해줍니다. 이를 통해 NAS 알고리즘의 계산 비용을 획기적으로 줄이는 데 성공하였습니다.

- **Technical Details**: 훈련 없는 Neural Architecture Search (TF-NAS)에서는 Variance of Knowledge of Deep Network Weights (VKDNW)라는 새로운 목표 함수 프록시를 적용합니다. 이 프록시는 네트워크 아키텍처에 따라 최적의 가중치 추정 난이도를 정량화할 수 있는 통계적 배경을 갖고 있습니다. 또한, Fisher Information Matrix (FIM)를 활용하여 네트워크 가중치 추정 문제를 보다 효율적으로 해결하는 방법을 탐색합니다.

- **Performance Highlights**: 제안된 훈련 없는 프록시는 세 가지 공개 데이터 세트에서 최첨단 성과를 달성하였으며, 새로운 평가 지표인 Normalized Discounted Cumulative Gain을 사용하여 TF-NAS 방법 간의 유의미한 차이를 보였습니다. 이는 기존 NAS 방법론이 가진 한계점을 보완하고, 실제 NAS 응용에 더욱 적합한 평가 방식을 제공하는 데 기여합니다.



### SurGen: 1020 H&E-stained Whole Slide Images With Survival and Genetic Markers (https://arxiv.org/abs/2502.04946)
Comments:
          To download the dataset, see this https URL. See this https URL for GitHub repository and additional info

- **What's New**: SurGen은 843건의 대장암 사례에서 1,020개의 H&E 염색 전체 슬라이드 이미지(Whole Slide Images, WSIs)를 포함하는 데이터셋으로, 주요 유전자 변이(KRAS, NRAS, BRAF)와 불일치 수선 상태(mismatch repair status) 및 생존 데이터를 정교하게 주석 처리하였습니다. 이 데이터셋은 높은 품질의 WSIs와 관련된 임상 및 유전자 정보를 통합하여 대장암에 대한 개인 맞춤형 치료 전략 개발을 지원합니다.

- **Technical Details**: SurGen 데이터셋은 40배 확대(×40)로 스캔된 초고해상도 이미지로 구성되어 있으며, 평균 픽셀 크기는 189,662×156,059입니다. 데이터는 CZI 파일 형식으로 저장되어 있어 효율적인 데이터 저장 및 검색을 지원하며, 환자의 생존 정보와 유전자 변이에 대한 광범위한 메타 정보를 포함합니다. 두 가지 하위 집합(SR386, SR1482)으로 나뉘어 있으며, 각 하위 집합은 서로 다른 연구 필요에 부합하는 유용한 정보를 제공합니다.

- **Performance Highlights**: 초기 실험을 통해 SurGen의 유용성을 입증하기 위해, WSI로부터 불일치 수선 상태를 예측하는 머신러닝 분석을 수행하였으며, 0.8316의 AUROC(test area under the receiver operating characteristic curve)를 달성했습니다. 이 결과는 SurGen 데이터셋이 대장암 연구, 바이오마커 발굴, 예후 모델링, 그리고 고급 머신러닝 응용 프로그램에 기여할 수 있는 잠재력을 강조합니다.



### Cached Multi-Lora Composition for Multi-Concept Image Generation (https://arxiv.org/abs/2502.04923)
Comments:
          The Thirteenth International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA)의 이점을 활용하여 이미지 생성을 위한 보다 나은 방법을 제안합니다. 기존의 LoRA 사용에서 발생하는 'semantic conflicts' 문제를 해결하기 위해 Fourier frequency domain을 기반으로 한 새로운 접근 방식을 도입했습니다. 새로운 프레임워크인 Cached Multi-LoRA (CMLoRA)를 통해 LoRA 모듈을 효과적으로 결합할 수 있는 방법을 제시하며, 이로 인해 이미지 생성을 향상시킬 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: CMLoRA 프레임워크는 고주파 및 저주파 LoRA를 구분하여 각각의 주파수 응답에 따라 이미지를 생성하는데 기여할 수 있도록 설계되었습니다. 이 프레임워크는 특히 초기 denoising 단계에서 고주파 LoRA를 우선 사용하고, 이후 단계에서 저주파 LoRA를 적용합니다. 이러한 방법론은 LoRA의 통합 과정에서 발생할 수 있는 시맨틱 충돌을 최소화하고, 보다 일관성 있는 이미지를 생성하는 데 기여합니다.

- **Performance Highlights**: CMLoRA는 최신 LoRA 통합 방법들에 비해 평균 2.19%의 CLIPScore 개선과 11.25%의 MLLM 승률 개선이라는 우수한 성능 개선을 보였습니다. 실험을 통해 우리는 CMLoRA가 사용자에게 보다 향상된 이미지 품질을 제공한다는 것을 입증하였으며, 이는 LoRA 통합의 차별적인 잠재력을 보여줍니다.



### Goku: Flow Based Video Generative Foundation Models (https://arxiv.org/abs/2502.04896)
Comments:
          page: this https URL

- **What's New**: 이번 논문은 Goku라는 이름의 고급 이미지와 비디오 생성을 위한 모델 계열을 소개합니다. 이 모델은 Rectified Flow Transformers를 활용하여 뛰어난 성능을 달성하며, 비디오 생성 분야에서 새로운 기준을 설정했습니다. Goku는 텍스트에서 이미지로의 생성에서 0.76, 텍스트에서 비디오로의 생성에서 84.85라는 민감한 성적을 기록하며, 여러 주요 작업에 대한 새로운 벤치마크를 제시합니다.

- **Technical Details**: Goku는 데이터 커링, 모델 아키텍처 디자인, 흐름 포뮬레이션 및 대규모 교육을 위한 인프라 최적화의 네 가지 핵심 요소에 중점을 두고 있습니다. 이 모델은 2B와 8B 파라미터를 가진 Transformer 아키텍처를 포함하며, 3D 공동 이미지-비디오 변분 오토인코더(VAE)를 사용하여 이미지와 비디오 입력을 공유 잠재 공간으로 압축합니다. 이를 통해 이미지와 비디오의 통합 훈련을 가능하게 하여 높은 품질의 연속적인 출력을 제공합니다.

- **Performance Highlights**: Goku는 여러 개의 벤치마크에서 강력한 성능을 발휘하며, 특히 텍스트-비디오 생성에서 UCF-101 제로샷 생성 작업에서 최첨단 성과를 기록했습니다. Goku-T2I는 T2I-CompBench, GenEval 및 DPG-Bench와 같은 평가에서 뛰어난 시각적 품질과 텍스트-이미지 정렬을 보였습니다. 이 결과는 Goku의 다중 모달 생성의 효과성을 강조하며, 연구 및 상업적 응용 프로그램 모두에서 높은 성능을 발휘할 수 있는 잠재력을 보여줍니다.



### IPSeg: Image Posterior Mitigates Semantic Drift in Class-Incremental Segmentation (https://arxiv.org/abs/2502.04870)
Comments:
          20 pages, 9 figures

- **What's New**: 이번 연구에서는 Class Incremental Semantic Segmentation (CISS)에서의 semantic drift 문제를 다루며, 두 가지 주요 도전에 집중하고 있습니다. 첫 번째는 별도의 최적화 문제로, 이는 모델의 다양한 부분이 각기 다른 단계에서 최적화되어 확률 스케일이 일치하지 않아 성능 저하를 초래합니다. 두 번째는 잘못된 pseudo-labeling으로 인한 noisy semantics 문제입니다. 이러한 도전에 대응하기 위해, 이미지 후방 확률을 활용한 최적화 정렬 및 semantics decoupling을 이용한 새로운 접근법인 IPSeg를 제안합니다.

- **Technical Details**: IPSeg 방법론은 이미지 후방 확률을 활용하여 별도의 최적화 문제를 완화하는 동시에 noisy semantics를 처리하기 위한 semantics decoupling 메커니즘을 포함합니다. 모델은 incremental 작업에서 경험적 데이터와 ground truth 쌍으로 구성된 독특한 훈련 데이터셋을 사용하며, 두 가지 유형의 semantics로 나누어 학습 전략을 조정합니다. 이 과정에서 이미지 후방 가이드를 통해 픽셀 수준의 예측을 정정하고, 안정적인 semantics와 복잡한 semantics를 구별하여 학습합니다.

- **Performance Highlights**: Pascal VOC 2012와 ADE20K 데이터셋에 대한 광범위한 실험 결과, IPSeg는 Long-term incremental 시나리오에서 특히 탁월한 성능을 보여주며 최신 기법들보다 24.8% 높은 개선을 기록했습니다. IPSeg는 semantic drift 문제를 효과적으로 해결하는 동시에 다양한 incremental 시나리오에서 일관되게 우수한 성능을 발휘함으로써 CISS 분야에서 중요한 기여를 할 것으로 기대됩니다.



### Relative Age Estimation Using Face Images (https://arxiv.org/abs/2502.04852)
- **What's New**: 본 연구에서는 단일 얼굴 이미지를 바탕으로 나이를 추정하는 새로운 딥러닝 접근방법을 제안합니다. 초기 나이 추정값을 개선하기 위해 유사한 나이와 외모를 가진 인물의 레퍼런스 얼굴 데이터베이스를 활용하며, 이 과정에서 차별적 회귀(differential regression)를 통해 나이 의존적인 얼굴 변화를 명시적으로 모델링합니다. 또한 초기 나이 추정값을 반복적으로 수정하는 나이 증강(age augmentation) 기법을 도입하여 성능을 향상시킵니다.

- **Technical Details**: 연구는 Baseline Age Regression (BAR) 모델을 개선한 Differential Age Regression (DAR) 모델을 기반으로 합니다. 본 방법은 입력 이미지와 레퍼런스 이미지들의 나이 차이를 추정하여 초기 추정값을 정제하며, 이러한 나이에 대한 차이 추정이 성능을 높은 정확도로 향상시키는 원동력이 됩니다. 본 기법은 기존의 절대적 나이 추정 방식과는 달리, 이미지의 시각적 특성을 바탕으로 나이 차이를 모델링하여 극복하는 점이 특징입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 MORPH II 및 CACD 데이터셋에서 최신의 정확도를 달성하여 기존 기법들을 초월함을 입증하였습니다. DAR 모델은 전통적인 BAR 방법보다 더 우수한 정확성을 보여주며, 이를 통해 젊고 노화된 얼굴의 다양한 시각적 특성을 정확하게 반영할 수 있음을 강조합니다. 이 연구는 얼굴 이미지 간의 나이 차이를 평가하는 최초의 차별적 기반 나이 추정 방법으로, 다양한 모델에 통합 가능하여 전체 예측 정확성을 향상시킬 수 있는 기반을 마련합니다.



### HumanDiT: Pose-Guided Diffusion Transformer for Long-form Human Motion Video Generation (https://arxiv.org/abs/2502.04847)
Comments:
this https URL

- **What's New**: HumanDiT는 포즈 가이드 확산 변환기(Diffusion Transformer, DiT) 기반의 프레임워크로, 고해상도 비디오 생성에 필요한 14,000시간의 고품질 비디오 데이터셋으로 훈련되었습니다. 이 프레임워크는 다양한 비디오 해상도와 변동 길이 시퀀스를 지원하여, 긴 시퀀스 비디오 생성에서 학습을 용이하게 합니다. 또한, 개인화된 특성을 유지하기 위해 프리픽스-라텐트 참조 전략을 도입했습니다.

- **Technical Details**: HumanDiT는 동적 시퀀스 길이와 다양한 해상도를 처리하기 위해 베이스 모델인 U-Net 대신 DiT 아키텍처를 사용합니다. 패치 기반 추출을 통해 시간 및 공간 기능을 캡처하는 포즈 가이더를 통해 시각적 일관성을 보장하면서도 다양한 해상도와 기간에 대응할 수 있습니다. Keypoint-DiT를 활용하여 포즈 생성을 지원하고, 포즈 전송을 위한 Pose Adapter를 통해 다양한 하위 응용 프로그램에 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 인체 동작 비디오 생성을 위한 기존의 방법들과 비교할 때, HumanDiT는 정밀한 포즈 표현 및 세밀한 신체 렌더링을 제공하여 긴 형식의 비디오를 생성하는 데 있어 우수한 성능을 나타냅니다. 광범위한 실험을 통해 다양한 시나리오에서 양적 및 질적으로 뛰어난 결과를 나타내었으며, 최신 상태의 기술들과 비교해도 우수한 성과를 보였습니다.



### PoI: Pixel of Interest for Novel View Synthesis Assisted Scene Coordinate Regression (https://arxiv.org/abs/2502.04843)
- **What's New**: 이 논문에서는 카메라 위치 추정의 신뢰성을 높이기 위해 네트워크의 훈련 데이터를 다양화하는 방법으로 고유한 필터링 접근법을 제안합니다. 저자들은 Scene Coordinate Regression (SCR) 방법이 신뢰할 수 없는 렌더링 이미지의 영향을 받는 것을 밝혀내고, 이를 해결하기 위해 품질이 좋은 픽셀만 선택하여 훈련 데이터로 활용하는 방식으로 새로운 pixel of interest (PoI) 모듈을 도입하였습니다.

- **Technical Details**: SCR 방법은 픽셀 수준에서 3D 좌표를 추정하는 과정에서 품질이 낮은 렌더링 이미지로 인해 성능 저하를 겪는 것으로 나타났습니다. 새로운 PoI 모듈은 각 픽셀의 3D-2D 프로젝션 오차를 기준으로 잘 렌더링된 픽셀만 선택하여 효율적으로 훈련 데이터를 증강합니다. 또한, 저자들은 RaW 렌더링 이미지와 실 데이터의 조합을 통해 훈련 과정에서 보다 강력한 포즈 추정 성능을 달성하였습니다.

- **Performance Highlights**: 실험 결과, 저자들이 제안한 방법은 실내 및 실외 데이터셋에서 최첨단 성능을 나타냈습니다. 이 접근법은 훈련 데이터의 양이 제한적일 때도 강력한 성능을 발휘할 수 있도록 설계되었습니다. PoI 모듈을 통해 SCR 모델은 기존 방법 대비 더욱 정확한 카메라 위치 추정을 수행할 수 있었습니다.



### Lightweight Operations for Visual Speech Recognition (https://arxiv.org/abs/2502.04834)
Comments:
          10 pages (double column format), 7 figures

- **What's New**: 이번 연구에서는 비디오 데이터에서 음성을 인식하는 Visual Speech Recognition (VSR)의 경량화 아키텍처를 개발하여 리소스 제한이 있는 장치에서도 실행 가능하게 만드는 데 중점을 두었습니다. 기존 모델들이 요구하는 높은 계산 비용 문제를 해결하기 위해 Ghost 모듈을 활용하여 모델의 복잡성을 줄이고, 성능 손실을 최소화하면서도 강력한 인식 능력을 갖춘 모델을 설계하였습니다.

- **Technical Details**: 연구에서 제안된 아키텍처는 Ghost 모듈을 사용하여 전통적인 합성곱 연산을 대체하고, 필요한 파라미터 수와 계산 비용을 줄였습니다. 또한, Partial Temporal Block이라는 일반적인 시간 차단 구조를 설계하여 입력 볼륨을 두 부분으로 나누고 각 부분에 대해 별도의 연산을 적용하는 방식을 채택하였습니다. 이러한 접근은 저전력 응용프로그램에 적합한 초경량 템포럴 합성곱 네트워크를 개발하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 제안된 모델들은 대규모 공개 데이터셋에서 우수한 VSR 성능을 보여주었으며, 하드웨어 요구사항이 크게 감소하여 다양한 실용적 응용이 가능함을 입증하였습니다. 대규모 실험 분석을 통해 경량화된 아키텍처가 성능을 크게 저하시키지 않으면서도 리소스 요구 사항을 획기적으로 줄일 수 있음을 확인하였고, 이는 여러 종합적으로 다양한 계산 능력을 가진 장치에서의 활용을 용이하게 합니다.



### DetVPCC: RoI-based Point Cloud Sequence Compression for 3D Object Detection (https://arxiv.org/abs/2502.04804)
- **What's New**: 이번 논문에서는 MPEG 표준의 비디오 기반 포인트 클라우드 압축(Video-based Point Cloud Compression, VPCC)의 한계를 극복하기 위해 DetVPCC라는 새로운 방법론을 제안합니다. DetVPCC는 관심 영역(region-of-interest, RoI) 인코딩을 통합하여 효율적인 포인트 클라우드 시퀀스 압축을 가능하게 하며, 3D 객체 인식 정확도를 유지합니다. 이 방법은 VPCC의 성능을 개선하여 고정된 품질 수준 대신 공간적으로 비균일한 품질 분배를 지원하도록 설계되었습니다.

- **Technical Details**: DetVPCC의 첫 번째 단계에서는 RoI 탐지기가 포인트 클라우드 내에서 관심 영역을 식별하고, VPCC는 원시 포인트 클라우드를 손실 없는 2D 깊이 이미지로 변환합니다. 이후 RoI 인코더가 비손실 깊이 이미지에 RoI 기반 손실 압축을 적용하여 데이터 볼륨을 효과적으로 압축합니다. 이 과정에서 H.264와 같은 표준 2D 비디오 코덱을 활용하여 2D 이미지를 압축하고, 포인트 클라우드를 다시 구축할 수 있는 Bitstream을 제공합니다.

- **Performance Highlights**: nuScenes 데이터셋에 대한 실험 결과, DetVPCC는 전통적인 VPCC보다 더 나은 비트 전송률-정확도( bitrate-accuracy) 트레이드오프를 달성했습니다. 이 방법은 3D 객체 탐지기의 정확도를 상당히 개선하며, 실제 애플리케이션에서의 클라우드 기반 3D 객체 탐지 및 최적화를 지원할 수 있습니다. 궁극적으로, DetVPCC는 포인트 클라우드를 보다 효율적으로 처리할 수 있는 가능성을 보여줍니다.



### Autoregressive Generation of Static and Growing Trees (https://arxiv.org/abs/2502.04762)
- **What's New**: 이 논문에서는 트리 생성(tree generation)을 위한 혁신적인 transformer 아키텍처와 훈련 전략을 제안합니다. 제안된 아키텍처는 계층적 구조를 잘 보존하면서도, 처리 속도와 메모리 사용량을 최적화합니다. 이를 통해 더 복잡한 트리를 생성할 수 있으며, 이미지에서 트리로와 포인트 클라우드에서 트리로의 조건부 생성도 지원합니다.

- **Technical Details**: 제안하는 방식인 HourglassTree 모델은 계층적 종속성을 고려하여 효율적인 파라미터화 전략을 적용합니다. 이 아키텍처는 중간 레이어에서 더 적은 토큰을 처리하여 다중 해상도(multi-resolution) 방식으로 훈련 효율성을 높입니다. 또한 Skips connections를 도입해 더욱 빠르고 메모리 효율적인 처리 속도를 제공합니다.

- **Performance Highlights**: 경험적 결과에 따르면, 제안된 접근 방식은 속도, 메모리 사용량 및 생성 품질 면에서 우수한 성능을 나타냅니다. 특히, 복잡한 트리 구조를 보다 효과적으로 생성할 수 있으며, 4D 트리 성장 시뮬레이션을 통한 동적인 트리 진화를 지원합니다. 이러한 특성은 고품질 트리 모델을 보다 효율적으로 생산할 수 있는 새로운 가능성을 열어줍니다.



### ELITE: Enhanced Language-Image Toxicity Evaluation for Safety (https://arxiv.org/abs/2502.04757)
- **What's New**: 현재의 Vision Language Models (VLMs)은 악의적인 프롬프트로 인해 해로운 결과를 생성하는 취약성이 존재합니다. 기존의 안전성 벤치마크는 자동화된 평가 방법에 의존하나, 이러한 방법들은 암시적인 해로운 내용을 감지하는 데 어려움을 겪습니다. 이에 따라 저자들은 ELITE 벤치마크를 제안하며, 이는 VLM의 안전성을 평가하기 위한 고품질 평가 도구로 자리잡을 것입니다.

- **Technical Details**: ELITE 벤치마크는 독창적인 평가 방법인 ELITE evaluator에 기반하고 있으며, 이를 통해 다중 모드(multi-modal) 컨텍스트에서의 해로운 정도를 정확하게 평가할 수 있는 독성 점수를 포함하고 있습니다. 기존의 벤치마크에서 모호하거나 저품질의 이미지-텍스트 쌍을 필터링하고, 다양한 안전 및 위험 이미지-텍스트 쌍을 생성합니다. 이러한 방식은 VLM들이 특정하고 설득력 있는, 그러나 유해하지 않은 설명을 제공하는 문제를 해결하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과, ELITE evaluator는 기존의 자동화된 방법들과 비교할 때 인간 평가와의 정렬도에서 우수한 성과를 보였습니다. ELITE 벤치마크는 평가 품질과 다양성을 증진시켜, 안전하고 강력한 VLM 개발에 기여할 수 있는 중요한 도구로 자리 잡게 될 것입니다. 이는 실제 응용 프로그램에서의 안전성 위험을 평가하고 완화하는데 크게 기여할 것으로 예상됩니다.



### Self-Supervised Learning for Pre-training Capsule Networks: Overcoming Medical Imaging Dataset Challenges (https://arxiv.org/abs/2502.04748)
- **What's New**: 이번 연구에서는 대장암의 폴립 진단을 위한 캡슐 네트워크를 기초로 한 자기 감독 학습(self-supervised learning) 기법을 사용하여 사전 학습하는 방법을 탐구합니다. 이 연구는 3,433개의 샘플로 구성된 PICCOLO 데이터셋을 활용하여 의료 데이터셋의 일반적인 문제인 작은 크기, 클래스 불균형, 데이터 분할 간의 배급 변화에 대한 도전 과제를 처리하고자 합니다. 캡슐 네트워크는 그들만의 해석 가능성을 제공하며, 이를 통해 효과적인 진단을 도울 수 있습니다.

- **Technical Details**: 캡슐 네트워크(CapsNet)는 이미지 내의 세부 사항과 객체 간의 관계를 캡처하기 위해 구성된 독특한 신경망 아키텍처입니다. 본 연구에서는 색상화(colorisation) 및 대조 학습(contrastive learning)이라는 두 가지 자기 감독 학습 기법을 추가적으로 사용하여 캡슐 네트워크의 사전 학습을 수행했습니다. 특히, 대조 학습을 통해 중요한 시각적 특징을 포착하여 폴립 분류 작업의 정확도를 5.26% 향상시켰습니다.

- **Performance Highlights**: 연구 결과, 캡슐 네트워크는 세 가지 실험 조건 - 처음부터 훈련, 자기 감독 학습으로 사전 훈련, ImageNet에서 사전 훈련된 ResNet의 초기 레이어 활용 - 하에서 평가되었습니다. 이 과정에서 자기 감독 학습 방법이 폴립 진단에 효과적이라는 것이 입증되었습니다. 기존의 초기화 전략과 비교하여 캡슐 네트워크의 성능이 크게 향상되었음을 나타내며, 이는 의료 도메인에서의 더 나은 적응을 가능하게 합니다.



### SelaFD:Seamless Adaptation of Vision Transformer Fine-tuning for Radar-based Human Activity (https://arxiv.org/abs/2502.04740)
- **What's New**: 본 연구에서는 노인 인구 증가로 인해 필연적으로 필요한 휴먼 액티비티 인식(HAR) 시스템의 효율성을 높이기 위해 비전 트랜스포머(ViT) 모델을 미세 조정하는 새로운 접근 방식을 제안합니다. 특히 레이더 기반의 Time-Doppler 신호를 활용하여 HAR 과제를 해결하는 데 중점을 두었습니다. 이를 통해 기존의 방법들이 가지고 있던 한계를 극복하고, 비주얼 인식에서의 지식 전이를 통해 더 높은 정확도를 달성할 수 있음을 증명하였습니다.

- **Technical Details**: 연구에서는 Low-Rank Adaptation (LoRA) 방식을 활용하여 ViT 모델을 효과적으로 미세 조정하는 방법을 소개합니다. 이 방법은 레이더 스펙트로그램의 특징을 잘 추출하고, 파라미터의 수를 줄이면서도 필요한 정보는 유지할 수 있도록 설계되었습니다. 추가적으로, 시리얼-패러럴 어댑터 모듈을 통해 세밀한 특징을 추출하여 모델의 인식률을 개선합니다.

- **Performance Highlights**: 제안한 SelaFD 방법을 통해 University of Glasgow의 공공 데이터셋에서 무려 96.61%의 분류 정확도를 달성했습니다. 이는 현재까지의 HAR 관련 연구 중에서 가장 높은 성과로, 기존의 최첨단 기술들과 비교하여도 우수한 성능을 보입니다. 앞으로 이 기술을 다양한 비전 기반 모델 및 다중 모드 대규모 모델에 통합하여 HAR의 정확도를 더욱 향상시키고자 합니다.



### SC-OmniGS: Self-Calibrating Omnidirectional Gaussian Splatting (https://arxiv.org/abs/2502.04734)
Comments:
          Accepted to ICLR 2025, Project Page: this http URL

- **What's New**: 이 논문에서는 360도 이미지에서의 빠르고 정확한 방사장(radiance field) 재구성을 위한 새로운 자기 보정(self-calibrating) 구형 가우시안 splatting 시스템인 SC-OmniGS를 제안합니다. 기존의 방사장 방법들은 360도 이미지의 특수한 도전에 대응하지 못하는 반면, SC-OmniGS는 360도 이미지를 전체 구로 처리하여 직접적인 카메라 자세(calibration) 보정을 가능하게 합니다. 이 시스템은 실제 데이터의 왜곡을 교정하기 위해 미분 가능(differentiable)한 구형 카메라 모델을 도입하여 성능을 향상합니다.

- **Technical Details**: SC-OmniGS는 3D Gaussians의 최적화를 통해 방사장을 표현하며, 이를 위해 미분 가능한 레스터라이저(differentiable rasterizer)를 사용하여 3D Gaussian을 단위 구에 splatting하는 방식을 따릅니다. 이 과정에서 카메라 자세의 기울기(gradients)를 도출하여 노이즈가 섞인 카메라 자세를 최적화하고 제로부터 학습할 수 있도록 합니다. 또한 공간적 동등 최적화를 보장하기 위해 가중치 구형 광학 손실(weighted spherical photometric loss)을 도입하고, 극 지역에서의 필라멘트(kernel) 생성을 방지하기 위한 비등방성 조정기(anisotropy regularizer)를 적용합니다.

- **Performance Highlights**: SC-OmniGS는 소비자급 구형 카메라로 캡처된 실제 데이터셋에서 높은 품질의 방사장 회복이 가능함을 입증했습니다. 다양한 실험을 통해 복잡한 왜곡 패턴을 처리하고, 실세계 데이터에서의 카메라 내외부 매개변수의 정확성을 향상시키는 성능을 보여주었습니다. 이로 인해 SC-OmniGS는 방사장 재구성에서 최첨단 성능을 달성하며, 다양한 응용 프로그램에 유용하게 사용될 수 있습니다.



### Can Diffusion Models Learn Hidden Inter-Feature Rules Behind Images? (https://arxiv.org/abs/2502.04725)
Comments:
          25 pages, 18 figures, 3 tables

- **What's New**: 이 연구는 디퓨전 모델(Diffusion Models, DMs)의 이미지 특징 간 숨겨진 규칙 학습 능력에 대한 한계를 탐구합니다. 기존 연구에서는 주로 독립된 특징에 초점을 맞추었으나, 이 논문은 종속적인 특징 쌍의 관계를 분석하여 DMs가 이러한 관계를 정확하게 캡쳐할 수 있는지 여부를 평가합니다. 이 논문은 DMs가 정밀한 규칙을 학습하는 데 어려움을 겪고 있음을 밝히며, 모델의 성능을 향상시키기 위한 추가적인 방법론을 제안합니다.

- **Technical Details**: 연구는 이미지 데이터에서 두 개의 종속적인 특징 쌍(예: 태양의 높이와 그림자의 길이) 간의 규칙을 학습하는 능력을 실험적으로 평가합니다. 특히, DMs가 합동 분포를 추정할 때의 오류가 규칙 학습에 미치는 영향을 분석하며, 이러한 이유로 DMs는 세부 규칙을 효과적으로 학습하지 못하는 경향이 있음을 보입니다. 이 연구는 추가적인 분류기 가이드를 활용하여 DMs의 규칙 준수 샘플 생성을 개선하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과 DMs는 일반적으로 거칠고 간단한 규칙을 잘 학습하지만, 세밀한 규칙에서는 한계를 드러냅니다. 또한, 디퓨전 모델이 재현할 수 없는 구체적인 규칙의 존재는 이 모델들이 관심 있게 다루지 않던 부분에서 오는 것입니다. 이 연구는 클래식 분류기 훈련의 어려움과 함께, 세밀한 규칙을 식별하는 데 필요한 신호가 약한 문제를 강조합니다.



### Tolerance-Aware Deep Optics (https://arxiv.org/abs/2502.04719)
Comments:
          14 pages, 14 figures

- **What's New**: 이번 연구에서는 깊은 광학(Deep Optics) 설계 프로세스에서 제조 및 조립 공차(tolerance)를 분석하고 최적화하는 최초의 종합적인 '공차 인식 최적화 프레임워크'를 제안합니다. 이 접근은 제조 및 조립에서 발생할 수 있는 구조적 편차를 보상할 수 있는 물리 기반 모델링과 데이터 기반 학습을 통합합니다. 이는 광학 시스템과 비전 알고리즘의 성능 차이를 줄이는 데 기여합니다.

- **Technical Details**: 이 방법론에서는 Monte Carlo 샘플링을 사용하여 공차를 샘플링하고 최적화하는 과정을 포함합니다. 공차 최적화 흐름은 먼저 공차를 고려하지 않은 사전 훈련 단계를 거쳐 초기 설계를 얻고, 이후에 공차 인식 최적화를 통한 성능 강화를 진행합니다. 특히, 두 개의 새로운 손실 함수인 PSF 유사성 손실(Point Spread Function similarity loss)과 Spot 손실을 도입하여 최적화 과정의 안정성을 강화했습니다.

- **Performance Highlights**: 컴퓨터 이미징 응용 분야에서 이 방식을 검증한 결과, 새로운 공차 인식 최적화가 기존 설계 대비 2 dB 이상의 디블러링(deblurring) 성능 향상을 달성했습니다. 특히, 실제 이미지 획득 과정에서 발생하는 공차 변동을 시뮬레이션하여 복원된 이미지의 질을 비교한 결과, 제안한 접근법이 효과적임을 확인했습니다. 이러한 결과는 깊은 광학 시스템이 실세계에서 더욱 견고해질 수 있음을 시사합니다.



### AI-Driven Solutions for Falcon Disease Classification: Concatenated ConvNeXt cum EfficientNet AI Model Approach (https://arxiv.org/abs/2502.04682)
Comments:
          5 pages

- **What's New**: 이 연구는 독특하게 연결된 Concatenated ConvNeXt와 EfficientNet AI 모델을 사용하여 매(egyptian) 질병 분류를 위한 최첨단 접근법을 소개합니다. 이 방법은 전통적인 방법들과 독립형 아키텍처에 비해 탁월한 성능을 나타내며, 매의 건강 모니터링의 중요성을 강조합니다.

- **Technical Details**: 연구에서는 '정상(Normal)', '간(Liver)', '아스퍼길루스증(Aspergillosis)' 케이스를 구별하는 데 초점을 맞춘 포괄적인 데이터 세트를 사용하여 모델 훈련 및 평가를 수행합니다. 정확도(accuracy), 정밀도(precision), 재현율(recall) 및 f1-score 등의 메트릭(metric)을 활용하여 모델의 성능을 평가하였습니다.

- **Performance Highlights**: 엄격한 실험과 평가 과정을 통해, 연결된 AI 모델의 우수한 성능을 입증하였으며, 이를 통해 매의 질병 분류의 정확성을 높였습니다. 이 연구는 조류(avian) 수의학 AI 애플리케이션의 향후 발전을 위한 기초를 닦고 있습니다.



### Performance Evaluation of Image Enhancement Techniques on Transfer Learning for Touchless Fingerprint Recognition (https://arxiv.org/abs/2502.04680)
Comments:
          6 pages

- **What's New**: 이 연구는 비접촉식 지문 인식 시스템의 성능 향상에 집중하고 있습니다. 기존의 접촉식 스캐너는 이미지의 왜곡 및 사용자 상호작용의 일관성 부족으로 문제가 많았지만, 본 연구는 이러한 문제를 해소하고자 하는 새로운 접근법을 제시합니다. 특히, 이미지 향상 기술을 활용하여 사전 학습된 딥러닝 모델의 성능을 평가하며, 비접촉식 지문 인식의 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서 사용된 데이터베이스는 IIT-Bombay Touchless and Touch-Based Fingerprint Database로, 200명의 피험자 데이터를 포함하고 있습니다. VGG-16, VGG-19, Inception-V3, ResNet-50과 같은 다양한 딥러닝 아키텍처를 평가하며, 전이 학습(transfer learning)기법을 통해 성능을 측정합니다. 실험에서는 이미지 향상(indirect method)을 적용한 모델이 비향상(direct method) 모델보다 우수한 성능을 나타냅니다.

- **Performance Highlights**: VGG-16 모델은 향상된 이미지를 사용했을 때 훈련에서 98%, 테스트에서 93%의 정확도를 기록하였습니다. 이는 전이 학습 모델을 사용한 비접촉식 지문 인식의 정확도를 크게 향상시키는 결과입니다. 연구 결과는 향상된 이미지가 전이 학습 모델의 정확성에 미치는 긍정적인 영향을 확인시켜 주며, 더욱 효율적인 생체 인식 시스템 개발에 기여할 수 있음을 보여줍니다.



### Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers (https://arxiv.org/abs/2502.04679)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문은 비전 트랜스포머(Vision Transformer, ViT)의 알려진 표현 취약성의 원인을 연구하였습니다. 입력 공간에서 라벨 공간으로의 매핑에 대한 이해가 부족한 상황에서, 변형된 이미지들이 어떻게 서로 다른 표현을 초래할 수 있는지를 분석합니다. 특히, 논문에서는 미세한 입력 변화가 비트의 후반 레이어에서 유의미한 표현 변화를 일으키며 성능의 불안정을 시사하는 것을 강조합니다.

- **Technical Details**: 연구는 비전 트랜스포머의 중간 및 후반 레이어에서 적대적 효과(adversarial effects)가 어떻게 전파 및 증폭되는지를 보여줍니다. 이러한 분석을 바탕으로 NeuroShield-ViT라는 새로운 방어 메커니즘을 개발하여 초기 레이어의 취약한 뉴런을 중화시키는 전략을 취합니다. 이 방식은 적대적 공격에 대한 저항력을 기르는 데 중요한 역할을 합니다.

- **Performance Highlights**: NeuroShield-ViT의 효과는 다양한 공격에 걸쳐 확인되었으며, 특히 강력한 반복적 공격에 대해 탁월한 성능을 보였습니다. 조정(fine-tuning) 없이도 적대적 예제에서 77.8%의 경쟁력 있는 정확도를 달성하며, 기존의 강건성 방법을 초월하는 성과를 보였습니다. 이 연구는 비전 트랜스포머의 강건성을 향상시키기 위한 잠재적 접근법을 제시합니다.



### MHAF-YOLO: Multi-Branch Heterogeneous Auxiliary Fusion YOLO for accurate object detection (https://arxiv.org/abs/2502.04656)
Comments:
          arXiv admin note: text overlap with arXiv:2407.04381

- **What's New**: 본 논문에서 제안하는 MHAF-YOLO는 Multi-Branch Auxiliary FPN (MAFPN)이라는 새로운 구조를 포함하고 있습니다. 이 구조는 Superficial Assisted Fusion (SAF)와 Advanced Assisted Fusion (AAF) 두 개의 핵심 모듈로 구성되며, 이들을 통해 고수준의 시맨틱 정보와 저수준의 공간 정보의 통합을 개선합니다. 특히, MAFPN은 다양한 해상도의 특성 층에서 정보를 효과적으로 융합하고, 작은 객체 감지의 정확성을 높이는 데 의의를 두고 있습니다.

- **Technical Details**: MAFPN은 두 가지 보조 융합 모듈을 포함하여, SAF를 통해 얕은 피쳐를 효과적으로 융합하고, AAF를 통해 깊은 층에서의 다중 스케일 정보를 통합합니다. 이와 함께 Reparameterized Heterogeneous Multi-Scale (RepHMS) 모듈을 통해 다양한 크기의 합성곱 패턴을 병렬로 활용하여, 정보 손실 없이 작은 객체를 감지할 수 있도록 설계되었습니다. Global Heterogeneous Flexible Kernel Selection (GHFKS) 메커니즘이 추가되어, 각 해상도 특성 층에서 커널 크기를 조정하여 효과적인 수용 범위를 극대화합니다.

- **Performance Highlights**: MHAF-YOLO는 COCO 데이터셋에서 최신의 실시간 객체 감지 성능을 달성하였으며, 기존 YOLO 모델들을 초월하는 정확도를 보였습니다. 또한, 인스턴스 분할 및 회전 객체 탐지에서의 우수한 성능을 통해 강력한 일반화 능력을 입증하였습니다. 이로 인해 리소스가 제한된 장치에서도 낮은 계산 비용으로 높은 성능을 유지할 수 있다는 강점을 가지고 있습니다.



### Learning Street View Representations with Spatiotemporal Contras (https://arxiv.org/abs/2502.04638)
- **What's New**: 이 논문에서는 도시 시각 환경에서 동적 요소와 정적 요소를 효과적으로 학습할 수 있는 새로운 자기 감독 학습 프레임워크를 제안합니다. 특히, 도로 및 건물 등 정적 환경은 시간에 따라 불변성을 갖고, 보행자 및 차량 등 동적 요소는 무작위성을 가지기 때문에 정보를 효과적으로 인코딩하기 위해 두 가지 속성을 활용합니다. 시각적 장소 인식 및 사회경제적 추정 같은 다양한 하위 작업에서도 우수한 성능을 보였습니다.

- **Technical Details**: 이 연구에서 제안된 프레임워크는 두 가지 주요 속성을 활용합니다. 첫째, 시간 불변성 표현 (Temporal Invariance Representation)은 동일한 장소에서 찍힌 이미지들이 시간에 따라 정적 요소는 유지하고 동적 요소는 자동으로 필터링되도록 학습합니다. 둘째, 공간 불변성 표현 (Spatial Invariance Representation)은 인접한 지역의 이미지에서 유사한 건축 스타일이나 도시 기능을 활용해 전체 이웃 분위기를 인코딩합니다.

- **Performance Highlights**: 이 연구는 여러 도시 관련 하위 작업에서 제안된 방법이 기존의 감독 학습 및 비감독 학습 방법에 비해 우수한 성능을 보였음을 보여줍니다. 각기 다른 상대적 대조 학습 목표가 다양한 종류의 특성을 학습할 수 있음을 실험적으로 입증하였으며, 이는 도시 연구 및 시각 데이터의 적용 가능성을 높이는 데 기여합니다. 또한, 다양한 대조 방법들의 성능을 분석하여 목표 지향적 학습 전략의 중요성을 강조합니다.



### High-Speed Dynamic 3D Imaging with Sensor Fusion Splatting (https://arxiv.org/abs/2502.04630)
- **What's New**: 본 논문에서는 RGB 카메라, 이벤트 카메라, 깊이 카메라를 통합한 새로운 센서 융합 접근 방식인 Gaussian splatting을 제안합니다. 이 방법은 빠르게 변형되는 3D 장면을 캡처하고 재구성하는 데 필요한 공간, 시간 및 깊이 정보를 동시에 제공합니다. 기존의 단일 이미징 모달리티의 한계를 극복함으로써, 낮은 조명, 좁은 기준선, 빠른 움직임과 같은 어려운 조건에서도 고품질 이미지를 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 저자들은 각기 다른 세 가지 이미징 모달리티(RGB 카메라, 이벤트 카메라, 깊이 카메라)의 장점을 통합하여 변화하는 3D 장면을 나타내기 위해 변형 가능한 3D Gaussian을 사용합니다. 이들은 기존 기술들보다 빠른 속도와 높은 해상도로 고속 동적 장면을 재구성하기 위해 전방위적으로 최적화된 3D Gaussian 파라미터와 그들의 시간적 변형 모델을 함께 사용합니다. 이러한 융합 기법은 서로 다른 센서의 정보를 효과적으로 결합하여 우수한 성능을 발휘합니다.

- **Performance Highlights**: 제안한 방법은 합성 및 실제 데이터 세트를 통해 엄격한 평가를 거쳤으며, 특히 빠른 움직임과 낮은 조명 조건 하에서 기존 기술들보다 현저하게 개선된 렌더링 충실도와 구조적 정확성을 보여주었습니다. 실험 결과, 저자들이 개발한 센서 융합 프로토타입이 고속, 동적 장면의 효과적인 캡처를 가능하게 하여, 3D 장면 재구성에 있어 향상된 정밀도를 달성함을 입증하였습니다.



### AIQViT: Architecture-Informed Post-Training Quantization for Vision Transformers (https://arxiv.org/abs/2502.04628)
- **What's New**: 이 논문에서는 AIQViT (Architecture-Informed Post-training Quantization for ViTs)라는 새로운 포스트-트레이닝 양자화 방법을 제안합니다. AIQViT는 비전 변환기(ViT) 모델의 가중치 양자화로 인한 정보 손실을 보완하기 위해 아키텍처 정보를 활용한 저랭크 보상 메커니즘을 설계했습니다. 또한, 비대칭적인 포스트-Softmax 활성화 분포를 처리하기 위해 동적 포커싱 양자화기(Dynamic Focusing Quantizer)를 개발했습니다.

- **Technical Details**: AIQViT는 가중치 양자화로 발생하는 손실을 줄이기 위해 각 선형 레이어에 대해 학습 가능한 저랭크 가중치를 도입합니다. 또한, DFQ는 포스트-Softmax 활성화의 유용한 구간을 식별하고 표준 균일 양자화를 통해 더 높은 양자화 해상도를 실현합니다. 이러한 기술들은 비트 수가 낮은 경우에도 효율적인 양자화를 제공합니다.

- **Performance Highlights**: AIQViT는 이미지 분류, 객체 탐지, 인스턴스 분할, 포인트 클라우드 분류 및 포인트 클라우드 부분 분할과 같은 다섯 가지 비전 작업에 대해 광범위한 실험을 수행하였으며, 기존의 최첨단 PTQ 방법들에 비해 우수한 성능을 입증했습니다. 이 연구는 특히 VIT의 효율적인 배치와 양자화 성능을 높이는 데 기여합니다.



### HetSSNet: Spatial-Spectral Heterogeneous Graph Learning Network for Panchromatic and Multispectral Images Fusion (https://arxiv.org/abs/2502.04623)
- **What's New**: 이 논문에서는 원격 탐사에서 파인 샤프닝(pansharpening)을 위한 새로운 그래프 기반 학습 네트워크인 HetSSNet을 제안합니다. 기존의 CNN 및 Transformer 모델이 픽셀을 유클리드 공간의 정규 그리드로 간주하는 것에 비해, HetSSNet은 불규칙한 지형과 객체를 모델링할 수 있는 이종 그래프 구조를 구축합니다. 이 모델은 파인 샤프닝과 관련된 공간-스펙트럼 관계를 효과적으로 설명하기 위해 파인 샤프닝 특정 관계를 반영하는 그래프를 설계합니다.

- **Technical Details**: HetSSNet은 이종 그래프 구조(HetSS-Graph)를 수립하며, 이 구조는 다양한 이종 노드와 엣지로 구성되어 파인 샤프닝에 필요한 공간-스펙트럼 관계를 명확히 표현합니다. 또한, 각 노드 간의 다중 관계 패턴을 추출하는 기본 관계 패턴 생성 모듈과, 그래프 노드 간의 관계를 통합적으로 학습하는 관계 패턴 집계 모듈을 설계하였습니다. 이 과정을 통해 HetSSNet은 로컬 및 글로벌 관점에서 적응형 중요성 학습을 통한 통합적인 공간-스펙트럼 표현을 학습하게 됩니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 광범위한 실험 결과는 HetSSNet이 기존의 SOTA(State of the Art) 방법들에 비해 우수한 성능을 달성하며, 실제 환경에서도 높은 일반화 능력을 발휘함을 보여줍니다. 이는 파인 샤프닝을 위한 이종 그래프 구조와 그로부터 도출된 관계 패턴들이 공간-스펙트럼 속성 재구성에서 핵심 역할을 했음을 입증합니다.



### Neural Clustering for Prefractured Mesh Generation in Real-time Object Destruction (https://arxiv.org/abs/2502.04615)
- **What's New**: 본 논문에서는 사전 파손(prefracture) 방법을 개선하기 위해 물리 기반(physics-based) 데이터셋으로 학습된 심층 신경망(deep neural network)을 활용하여 사전 파손 메시(prefractured mesh) 생성을 군집화(clustering) 문제로 접근합니다. 이 방법은 객체의 구조적 약점을 예측할 수 있어 더욱 현실감 있는 결과를 제공합니다. 기존의 수작업 교정 과정 대신 자동화된 방식으로 품질 높은 결과를 생성하는 데 성공했습니다. 또한 이 기술은 실제 응용 프로그램인 Unreal Engine 5(UE 5)와 호환됩니다.

- **Technical Details**: 본 연구에서는 사전 파손 메시 생성 과정에서 클러스터링 단계에 네트워크를 도입하였으며, 물체의 질량 중심(center of mass)을 수집하여 포인트 클라우드(point cloud)를 구성합니다. 데이터셋의 실제 그룹 레이블을 사용하여 포인트 클라우드를 클러스터링하며, Point Transformer 아키텍처를 활용해 그룹 레이블을 예측합니다. 레이블은 무작위 순서로 간주되어야 하므로, 손실(loss) 함수는 순열 불변(permutation-invariant) 형태로 정의되었습니다. 이를 위해 이웃 점의 그룹 유사성을 고려하여 이진 교차 엔트로피(binary cross-entropy)를 적용했습니다.

- **Performance Highlights**: Breaking Bad 데이터셋을 사용하여 네트워크를 훈련시켰으며, 입력 메시를 미세 조각으로 나누고 출력 파편 메시(output fragment mesh)의 질량 중심을 기준으로 레이블링 했습니다. 후처리 과정에서 그룹 내의 인접한 조각들을 통합하고, 분리된 조각들을 새로운 그룹으로 나누어 실제 응용을 위한 사전 파손 메시를 완성합니다. 최종적으로, 제안된 방법은 UE 5에서 sofort 사용할 수 있는 고품질의 결과물을 도출하여 실용성을 보여줍니다.



### Multiscale style transfer based on a Laplacian pyramid for traditional Chinese painting (https://arxiv.org/abs/2502.04597)
Comments:
          25 pages, 13 figures

- **What's New**: 이번 논문에서는 전통 중국 화상의 패턴을 전이하기 위한 새로운 효과적 멀티스케일(style transfer) 방법을 제안합니다. 기존의 방식은 주로 서양 유화만을 스타일 이미지로 사용하여 전통 중국 화상에는 부자연스럽고 난잡한 효과를 초래했습니다. 새로운 방법은 Laplacian 피라미드 분해를 기반으로 하여 다양한 이미지 특징을 학습함으로써 전통 예술의 독특한 패턴을 전이할 수 있다는 점이 특징입니다.

- **Technical Details**: 제안된 방법은 두 단계로 나뉘어 있습니다. 첫 번째 단계에서는 Style Transfer Base Network를 사용하여 저해상도에서 전체 패턴을 전이합니다. 두 번째 단계는 Detail Enhancement Network와 엣지 정보 선택(Edge Information Selection, EIS) 모듈을 통해 고해상도에서 내용과 스타일의 세부 사항을 점진적으로 향상시키는 구조입니다. 이를 통해 멀티스케일 이미지 정보를 사용하여 더 세련된 결과물을 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법의 효과는 높은 품질의 스타일화 결과를 생성함으로써 입증됩니다. 또한, 기존의 최신(style transfer) 기법들과의 비교를 통해 강력한 성능을 확인할 수 있었습니다. 연구에서 사용된 데이터 세트와 코드가 공개되어 있어 다른 연구자들이 이를 활용할 수 있도록 하였습니다.



### An Optimized YOLOv5 Based Approach For Real-time Vehicle Detection At Road Intersections Using Fisheye Cameras (https://arxiv.org/abs/2502.04566)
- **What's New**: 이 논문에서는 도시 교차로에서의 실시간 차량 탐지를 위한 새로운 방법론을 제안하고 있습니다. 특히, fisheye 카메라를 활용하여 360도 시야를 확보하면서도 차량 탐지의 정확도를 높이는 데 중점을 두었습니다. YOLOv5 기반의 경량 딥러닝 모델을 적용해 주간과 야간 탐지 문제를 해결하기 위해 두 개의 다른 경로를 진행하는 구조를 도입했습니다.

- **Technical Details**: 제안된 방법은 데이터를 주간과 야간으로 분리하고, 다양한 데이터 시퀀스를 생성하여 훈련 과정에서의 효과를 극대화합니다. YOLOv5의 focus block을 통해 fisheye 이미지의 비선형 왜곡을 효율적으로 처리하여, 차량 탐지의 정확성을 높이는 데 기여합니다. 데이터셋은 IEEE VIP Cup 2020에서 제공된 실세계 fisheye 데이터를 활용하여 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 YOLOv5에 비해 13.7% mAP @ 0.5 향상된 성능을 기록했습니다. 이는 다양한 차량 탐지 시나리오에서의 보다 나은 일반화, 탐지 및 정확성을 가져오는 데 중요한 것을 입증합니다. 특히, 제안된 방법은 교차로와 같은 복잡한 환경에서도 뛰어난 탐지 능력을 보여주었습니다.



### The Phantom of the Elytra -- Phylogenetic Trait Extraction from Images of Rove Beetles Using Deep Learning -- Is the Mask Enough? (https://arxiv.org/abs/2502.04541)
Comments:
          Accepted at Imageomics Workshop at AAAI 2025 (not published in proceedings)

- **What's New**: 이번 연구에서는 سنت널 (symmetrical) 딥러닝 모델의 사용을 통해 형태적 (morphological) 특성을 자동으로 처리하는 가능성을 탐구합니다. 기존의 수작업 방식의 한계를 극복하며, 215종의 빌라곤 비틀 (rove beetles) 이미지를 사용한 실험을 진행했습니다. 비틀의 외형을 세 가지 다른 형태로 표현한 세부 방식을 비교하여, 각 방식이 단백질 계통구성 분류 (phylogenetic trait extraction)에 미치는 영향을 분석했습니다.

- **Technical Details**: 본 연구에서는 비틀의 morphology를 세 가지로 나누어 평가했습니다: 1) 비틀 몸체의 바이너리 마스크 (binary masks), 2) 비틀 윤곽에서 얻어진 푸리에 기술자 (Fourier descriptors), 3) 전체 색상 세분화 (full segmentation). 연구는 ResNet50 아키텍처를 사용해 각각의 모델에 대해 50 epoch 동안 훈련을 진행하였습니다. 결괏값은 각 표현 방식의 정확성과 효율성을 비교하는 데 주안점을 두었습니다.

- **Performance Highlights**: 마스크 기반 모델이 가장 우수한 성능을 보이며, 테스트 세트에서 0.33±0.02의 정규화된 Align Score을 기록했습니다. 반면 푸리에 기반 모델은 0.45±0.01, 세분화 모델은 0.39±0.07의 점수를 나타냈습니다. 이러한 결과는 현재 형태적 정보가 계통 분류 연구에 있어 중요성을 지니고 있음을 시사합니다.



### Fast Video Generation with Sliding Tile Attention (https://arxiv.org/abs/2502.04507)
- **What's New**: 이 논문은 슬라이딩 타일 어텐션(Sliding Tile Attention, STA)을 도입하여 Diffusion Transformers (DiTs)의 비효율적인 연산 문제를 해결하고자 합니다. STA는 사전 훈련된 비디오 확산 모델에서 주로 집합된 지역적 3D 윈도우 내의 어텐션 점수를 활용하여 전체 어텐션의 중복성을 줄입니다. 새로운 하드웨어 인식 슬라이딩 윈도우 디자인을 통해, STA는 효율성을 유지하면서도 표현력을 보존합니다.

- **Technical Details**: STA는 타일 단위로 작동하며, 이로 인해 효율적인 메모리 접근과 병렬 처리가 가능합니다. 소비자-생산자 패러다임을 채택하여, 생산자 warpgroups가 HBM에서 SRAM으로 데이터를 asynchronously 로드하는 동안, 소비자 warpgroups가 어텐션을 계산합니다. 이 과정에서 STA는 계산 중의 명시적인 어텐션 마스킹 필요성을 없애고, sparse 어텐션 마스크를 관리하여 GPU 활용도를 극대화합니다.

- **Performance Highlights**: HunyuanVideo는 STA 적용 시 5초 720P 비디오를 695초 또는 품질 손실을 최소화하며 578초에 생성할 수 있습니다. STA는 FlashAttention-2에 비해 2.15~2.59배, FlashAttention-3에 대해서는 1.35~1.63배의 속도를 높였습니다. 추가적으로, 주목할 점은, 최적의 어텐션 스페이스를 유지하면서도 훈련 없이도 성능이 향상될 수 있다는 점입니다.



### Measuring Physical Plausibility of 3D Human Poses Using Physics Simulation (https://arxiv.org/abs/2502.04483)
Comments:
          Accepted to BMVC2024

- **What's New**: 이번 연구에서는 3D 인간 자세 추정 모델의 물리적 타당성을 측정하기 위한 새로운 두 가지 측정 지표, 즉 CoM 거리와 자세 안정 지속 시간을 제안합니다. 이들은 물리 시뮬레이션을 통해 자세의 안정성을 측정하며, 기존의 방법이 갖고 있는 단점을 해결합니다. 또한 제안된 모델은 현재 사용되고 있는 다양한 최첨단 방법에 비해 실용성을 높이고 오류를 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 물리적 타당성을 측정하기 위해 물리 시뮬레이션을 활용했습니다. CoM 거리(Center of Mass distance)는 질량 중심의 궤적 거리이며, 자세 안정 지속 시간(Pose Stability Duration)은 자세가 회복 불가능한 실패에 도달하기 전에 운동 중 지속될 수 있는 시간을 나타냅니다. 이를 통해 이전의 물리 기반 지표들이 고려하지 않았던 시간적 안정성을 평가할 수 있게 되었습니다.

- **Performance Highlights**: 제안된 두 가지 지표를 이용하여 방법들의 성능을 평가하고 비교하였으며, 기존의 방식들보다 더 높은 신뢰성을 보였습니다. 특히, 물리적 타당성 지표를 통해 모델의 시뮬레이션 안정성과 연관성을 발견하여 실제 적용 시 유용성을 기대할 수 있습니다. 이로 인해 향후 다양한 응용 분야에서 보다 자연스러운 인간 행동 모델링이 가능해질 것입니다.



### OneTrack-M: A multitask approach to transformer-based MOT models (https://arxiv.org/abs/2502.04478)
Comments:
          13 pages, 11 figures

- **What's New**: 본 논문에서는 Multi-Object Tracking (MOT) 문제를 해결하기 위해 새로운 모델 OneTrack-M을 제안합니다. OneTrack-M은 transformer 기반으로, 전통적인 모델들보다 컴퓨터 비전에서의 처리 효율성과 정확도를 높이는 것을 목표로 합니다. 변화된 점은 객체 탐지와 추적을 위해 필요한 decoder 모델을 생략하여, 인코더만으로 시간적 데이터 해석을 수행하는 방식입니다.

- **Technical Details**: OneTrack-M은 전통적인 CNN 기반 접근 방식을 넘어 transformer 아키텍처의 장점을 활용합니다. 이 모델은 혁신적인 데이터 전처리(data preprocessing) 및 다중 작업 훈련(multitask training) 기법을 통해, occlusion(가리기) 및 다양한 목표 문제를 해결하기 위한 단일 가중치 집합을 사용합니다. 이로 인해 처리 시간이 단축되고 추적 정확도가 유지되거나 개선됩니다.

- **Performance Highlights**: 실험 결과, OneTrack-M은 기존의 첨단 모델들에 비해 최소 25% 빠른 추론 속도를 달성했습니다. 또한 추적 정확도 측정 기준을 유지하거나 개선하면서 신속한 응답이 요구되는 자율주행차, 감시 시스템, 로봇 등 여러 실시간 응용 프로그램에 적합한 가능성을 보여줍니다.



### Augmented Conditioning Is Enough For Effective Training Image Generation (https://arxiv.org/abs/2502.04475)
- **What's New**: 본 논문은 생성된 이미지의 다양성을 향상시켜 다운스트림 이미지 분류 모델을 효과적으로 훈련할 수 있는 방법을 조사합니다. 텍스트 프롬프트와 진짜 이미지를 조건으로 사용하는 방식을 통해, 높은 품질과 다양성을 지닌 합성 이미지를 생성하는 기법을 제안합니다. 기존의 방법들이 이미지 생성 모델을 세밀하게 조정하는 것과는 달리, 본 연구는 훈련 데이터로 사용할 수 있는 합성 데이터셋을 더욱 효율적으로 생성하는 것을 목표로 합니다.

- **Technical Details**: 저자들은 고전적인 비전 데이터 증강(Data Augmentation) 기법을 이용하여 영상 생성의 조건 정보로 활용하는 방법을 분석하였습니다. 특히, 실 데이터에 대해 증강한 이미지를 조건화함으로써 생성 과정에서 시각적 다양성을 이끌어내고, 이는 다운스트림 분류 성능을 높이는 데 기여합니다. 연구진은 사전 훈련된 확산 모델을 활용하여, 증강 조건화를 통해 새로운 훈련 이미지를 생성하였습니다.

- **Performance Highlights**: 연구 결과는 합성 데이터셋이 기존의 방법 대비 향상된 분류 성능을 가지고 있음을 보여줍니다. 특히, ImageNet Long-Tailed 벤치마크에서 우수한 성능을 달성하였고, 극단적인 적은 샷(few-shot) 상황에서도 뛰어난 성과를 보였습니다. 이러한 결과는 모델의 재조정 없이 합성 데이터를 효과적으로 활용하여 모델 훈련을 진행할 수 있는 잠재력을 나타냅니다.



### Color in Visual-Language Models: CLIP deficiencies (https://arxiv.org/abs/2502.04470)
Comments:
          6 pages, 10 figures, conference, Artificial Intelligence

- **What's New**: 이 논문은 현재 인공지능에서 가장 영향력 있는 시각 언어 모델(VML)인 CLIP(Contrastive Language-Image Pre-training)의 색상 인코딩 방식을 탐구합니다. 저자들은 CLIP이 적절한 색상 레이블을 색칠된 시각 자극에 부여할 수 있지만, 두 가지 주요 결점이 있음을 발견했습니다: (a) 무채색 자극에 편향이 있어 흰색, 회색, 검정색이 색상 레이블로 잘 지정되지 않으며, (b) 다른 시각 정보보다 텍스트를 우선시하는 경향이 있습니다.

- **Technical Details**: 연구자들은 신경 수준에서 내부 표현을 분석하여 CLIP의 색상 인식 결함의 원인을 찾기에 집중했습니다. 분석 결과, CLIP의 깊은 층에서 텍스트에 선택적인 신경 세포가 과다하게 발달해 있고, 색상 인식을 위한 멀티모달 색상 신경 세포는 적은 수가 있는 것으로 나타났습니다. 이러한 발견은 인간의 색상 이해와 일치하는 신경 세포의 발달을 더 잘 이해하는 데 중요합니다.

- **Performance Highlights**: CLIP은 색상 레이블을 기본 이미지에 연결하는 데 여러 가지 실험을 통해 성능을 평가했으며, 그 결과 무채색 부분에서 색상 값을 잘 예측하지 못하고 있음을 보여주었습니다. 특히, Stroop 테스트를 통해 CLIP이 시각적 정보보다 텍스트 정보에 더 집중한다는 것을 확인했습니다. 이는 CLIP이 색상 개념을 이해하는 방식에서 개선이 필요함을 시사합니다.



### No Images, No Problem: Retaining Knowledge in Continual VQA with Questions-Only Memory (https://arxiv.org/abs/2502.04469)
Comments:
          8 pages, in-review

- **What's New**: 이번 연구에서는 Visual Question Answering Continual Learning (VQACL)을 위한 새로운 접근 방식인 QUestion-only replay with Attention Distillation (QUAD)를 제안합니다. QUAD는 과거 질문만을 저장하고, 시각적 데이터를 저장할 필요 없이 메모리와 프라이버시 문제를 해결합니다. 또한, 질문 반복 메커니즘을 도입하여 현재 작업의 답변 공간에 과적합(Overfitting)되는 것을 방지하며, 본질적인 시각-언어 연관성을 유지합니다.

- **Technical Details**: QUAD는 질문만을 활용하여 현재 모델의 정규화를 수행하는 'Question-only Replay' 메커니즘을 특징으로 하며, 이 과정에서 출처-응답-세트 문제를 해결합니다. 두 번째 기여는 'attention consistency distillation'으로, 이는 다양한 작업 간에 주의 패턴을 일관되게 유지하도록 돕습니다. 이러한 기법은 특정 질문에 대해 언어-비언어 간의 관심 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 QUAD는 VQAv2 및 NExT-QA 데이터 세트에서 기존의 최첨단 방법들보다 뛰어난 성능을 보였습니다. QUAD는 메모리 없는 접근 방식과 전통적인 리허설 방법보다 뛰어난 결과를 보여주며, 질문만 저장해도 잊혀짐 문제를 완화할 수 있음을 입증했습니다.



### TerraQ: Spatiotemporal Question-Answering on Satellite Image Archives (https://arxiv.org/abs/2502.04415)
- **What's New**: 이 논문에서는 TerraQ라는 새로운 spatiotemporal QA 엔진을 소개합니다. 이 엔진은 위성 이미지 아카이브에 대한 사용자 요청을 자연어로 처리하여 쉽게 접근할 수 있도록 합니다. TerraQ는 기존의 템플릿 기반 질의 생성 방식을 제거하고, 고품질의 지리정보를 포함한 목적 맞춤형 지식 그래프(Knowledge Graph)를 활용하여 다양한 질문에 답변할 수 있는 이점을 제공합니다.

- **Technical Details**: TerraQ는 여러 컴포넌트로 구성되어 있으며, 각 컴포넌트는 특정 작업을 수행합니다. 시스템의 체계적 접근 방식은 질문을 받고, 이를 분해하여 자동으로 SPARQL 쿼리를 생성하는 프로세스를 포함합니다. 이 엔진은 Sentinel-1 및 Sentinel-2와 같은 다양한 위성 이미지 데이터 세트, 관리 영역(GADM) 및 특정 지리적 개체를 결합하여 사용합니다.

- **Performance Highlights**: TerraQ는 간단하고 복잡한 질문에 대해 신뢰성과 빠른 속도로 응답할 수 있습니다. 자동화된 처리 시스템 덕분에 비전문가와 전문가 모두에게 직관적으로 접근 가능하도록 설계되어 있으며, Earth Observation 데이터 아카이브의 접근성을 크게 향상시킵니다. 실제로 유럽 우주국의 프로젝트를 바탕으로 개발된 TerraQ는 전 세계적으로 사용할 수 있는 데모를 제공하고 있습니다.



### Decoder-Only LLMs are Better Controllers for Diffusion Models (https://arxiv.org/abs/2502.04412)
- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 효율성을 향상시키기 위해 대형 언어 모델(Large Language Models, LLMs)의 의미 이해 능력을 결합한 새로운 방법을 제안합니다. 특히, 이 모델은 LLM의 디코더 전용 구조를 활용하여 텍스트 프롬프트의 의미를 더 잘 캡처할 수 있도록 설계된 LLMDiff-Adapter라는 네트워크 모듈을 도입합니다. 기존의 방법들이 텍스트 인코더에 의존해 왔다면, 우리의 접근법은 LLM의 블록별 표현을 통합하여 텍스트 인코딩을 생성합니다.

- **Technical Details**: LLMDiff-Adapter는 노이즈 제거 U-Net 구조의 크로스-어텐션 부분에 연결됩니다. 이를 통해 LLM에서 추출한 표현을 텍스트 인코딩 생성에 직접적으로 활용할 수 있으며, 이는 세밀한 의미와 단어 간의 맥락 의존성을 효과적으로 포착합니다. 이 방법은 텍스트-이미지 생성 모델에 LLM의 강점을 직접적으로 통합하는 모듈로서 작동하며, 다양한 생성 모델에 쉽게 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLMDiff-Adapter를 사용한 모델은 생성된 이미지의 품질, 논리적 일관성 및 텍스트 설명에 대한 포괄적 이해 측면에서 최첨단 모델을 초월하는 결과를 보였습니다. 세밀한 이미지 세부 사항과 사용자 의도를 잘 반영한 이미지 생성을 통해 이 모델은 텍스트-이미지 생성 품질을 크게 향상시켰습니다. 다양한 벤치마크에서의 비교 분석을 통해 LLMDiff-Adapter의 효과성을 입증하였습니다.



### Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting (https://arxiv.org/abs/2502.04395)
Comments:
          19 pages

- **What's New**: 본 논문에서는 Time-VLM이라는 새로운 다중 모달 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 Vision-Language Models(VLMs)를 활용하여 시계열 데이터의 예측 정확도를 높입니다. Time-VLM은 세 가지 주요 구성 요소인 Retrieval-Augmented Learner, Vision-Augmented Learner, Text-Augmented Learner로 구성되어 있습니다.

- **Technical Details**: Time-VLM의 Retrieval-Augmented Learner(RAL)는 메모리 뱅크와 상호작용하여 풍부한 시계열 특징을 추출합니다. Vision-Augmented Learner(VAL)는 시계열 데이터를 정보가 풍부한 이미지로 변환합니다. Text-Augmented Learner(TAL)는 시계열 데이터에 대한 텍스트 설명을 생성하여 시각적 표현을 보완합니다.

- **Performance Highlights**: Time-VLM은 다양한 데이터셋에서의 실험을 통해 특히 few-shot 및 zero-shot 시나리오에서 뛰어난 성능을 달성했습니다. 이를 통해 Time-VLM은 다중 모달 시계열 예측 분야에 새로운 방향을 제시합니다. 이 모델은 시계열 예측의 정확도를 높이기 위해 텍스트, 비전, 시계열 데이터를 통합하여 예측합니다.



### UniCP: A Unified Caching and Pruning Framework for Efficient Video Generation (https://arxiv.org/abs/2502.04393)
- **What's New**: 이번 연구에서는 UniCP라는 혁신적인 통합 캐싱 및 프루닝 프레임워크를 제안하여 비디오 생성 과정을 효율적으로 가속화합니다. UniCP는 오류 인식 동적 캐시 윈도우(EDCW)를 통해 급격한 오류 변화를 실시간으로 반영하여 캐시 윈도우 크기를 조정합니다. 또한, PCA 기반 슬라이싱(PCAS) 및 동적 가중치 전환(DWS) 기법을 통해 불필요한 주의 컴포넌트를 제거하고 캐싱 및 프루닝을 통합합니다. 이로 인해 기존 방법보다 우수한 성능과 효율성을 달성합니다.

- **Technical Details**: Diffusion Transformers(DiTs)는 최근 비디오 생성 분야에서 두각을 나타내고 있지만, 주의(attention)의 제곱 복잡도로 인해 상당한 계산적 과제를 안고 있습니다. UniCP는 시간적 및 공간적 차원 모두에서 확인된 캐싱 및 프루닝 전략을 결합하여 디지털 비디오 생성의 효율성을 최적화합니다. EDCW는 다양한 타임스텝에서 캐시 윈도우 크기를 동적으로 조정하고, PCAS는 중복된 주의 컴포넌트를 제거하여 계산 복잡성을 줄이며, DWS는 프루닝 및 캐싱을 통합하여 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, UniCP는 단일 GPU에서 최대 1.6배의 속도를 향상시키면서 비디오 품질을 손상시키지 않는 성능을 보여주었습니다. 이를 통해 UniCP는 기존의 디지털 비디오 생성을 위한 방법들과 비교했을 때 더 효율적으로 작업을 수행할 수 있음을 증명합니다. 이 연구 결과는 generative AI 응용 프로그램의 범위를 확장할 수 있는 기회를 제공합니다.



### Towards Fair and Robust Face Parsing for Generative AI: A Multi-Objective Approach (https://arxiv.org/abs/2502.04391)
- **What's New**: 이번 연구는 얼굴 분할을 위한 다목적 학습 프레임워크를 제안합니다. 이 프레임워크는 정확성, 공정성(fairness), 및 강인성(robustness)을 동시에 최적화하여 비율적(segmentation bias) 분할을 개선하고, 실제 조건에서도 안정적인 성능을 제공합니다. 이를 통해 생성 모델에서의 출력 품질을 높이며, 특히 GAN 기반의 얼굴 생성 작업에서 공정성을 고려한 분할이 어떻게 향상되는지를 보여줍니다.

- **Technical Details**: 제안된 방법은 동적 가중치 조정이 가능한 homotopy-based loss function을 사용합니다. 이는 초기 학습 과정에서는 정확성을 중시하고, 이후 단계에서는 공정성과 강인성을 균형 있게 고려합니다. 이러한 접근법은 다양한 인구 집단 간의 분할 성능을 강화하고, occlusions, noise, domain shifts와 같은 실제 환경의 어려움에도 강한 저항력을 제공합니다.

- **Performance Highlights**: 실험을 통해 다목적 U-Net 모델과 단일 목표 U-Net 모델을 GAN 기반의 얼굴 생성 파이프라인(Pix2PixHD)에서 비교했습니다. 그 결과, 공정성을 고려한 강인한 분할이 GAN에서 생성된 얼굴의 품질을 추천적으로 개선하고, Perceptual realism을 높이는 것을 확인했습니다. 또한, ControlNet을 활용한 초기 실험에서는 분할 품질이 diffusion 기반 생성에 미치는 영향을 추가로 분석했습니다.



### Towards Fair Medical AI: Adversarial Debiasing of 3D CT Foundation Embeddings (https://arxiv.org/abs/2502.04386)
- **What's New**: 최근 기계 학습의 자기 지도 학습(self-supervised learning) 기술이 의료 이미징을 혁신적으로 변화시키고 있습니다. 본 논문에서는 특히 3D CT 데이터에 적용된 자기 지도 모델이 성과를 냈지만, 인구통계학적 정보(age, sex, race)도 함께 인코딩하고 있다는 문제를 제기합니다. 이러한 인코딩은 임상 응용의 공정성(fairness)에 심각한 위험을 초래할 수 있음을 강조하며, 이를 해결하기 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 접근 방식은 Variational Autoencoder (VAE) 기반의 적대적 편향 제거(adversarial debiasing) 프레임워크를 사용하여 기존의 인코딩된 임베딩을 새로운 잠재 공간(latent space)으로 변형합니다. 이 과정에서 인구통계학적 정보를 제거하면서도 다운스트림 임상 작업에서의 성능은 유지됩니다. VAE 아키텍처는 입력 3D CT 임베딩을 평균과 로그 분산 파라미터(latent space of mean and log variance parameters)로 매핑하는 인코더와 잠재 공간에서 임베딩을 재구성하는 디코더로 구성됩니다.

- **Performance Highlights**: NLST 폐암 스크리닝 데이터셋을 통해, 제안된 불편(framework) 변환 과정에서 인구통계학적 신호를 효과적으로 제거되는지를 검증하였습니다. 실험 결과, 1년 및 2년 간의 폐암 위험 예측의 정확도를 저하시키지 않으면서도 공정성을 크게 향상시킨 것을 보여주었습니다. 이러한 성과는 의료 분야에서 UberBiasing 기술의 가능성을 강조하며, 편향 없는 의료 의사 결정에 대한 넓은 채택을 위한 기반을 마련하는 데 기여합니다.



### TexLiDAR: Automated Text Understanding for Panoramic LiDAR Data (https://arxiv.org/abs/2502.04385)
- **What's New**: 이 연구는 고급 LiDAR 센서인 Ouster OS1이 생성한 2D 이미지를 활용하여 텍스트와 LiDAR 데이터를 연결하는 새로운 접근법을 제안합니다. 기존의 3D 포인트 클라우드를 의존하는 방법의 단점을 극복하고, 더 효율적으로 처리할 수 있는 2D 이미지 처리를 통해 이미지 캡셔닝 및 객체 감지를 수행합니다. Florence 2 모델을 활용하여 다양한 시각적 작업을 수행하며, 기존 방법보다 더 자세하고 정확한 결과를 도출합니다.

- **Technical Details**: Ouster OS1 LiDAR 센서는 고해상도의 깊이, 신호, 환경 이미지를 생성하며, 이를 통해 360도 뷰의 공간적 일관성을 제공합니다. 본 연구에서는 Florence 2 모델을 사용하여 2D 이미지에서 직접 이미지 캡셔닝과 객체 감지를 수행하며, 이미지의 360도 데이터를 전처리 없이 활용합니다. 이미지는 90도 섹션으로 나누어져 각 부분에서 독립적으로 처리되며, 모든 세그먼트의 예측 결과를 결합하여 전체 장면을 이해하는 방식을 적용합니다.

- **Performance Highlights**: 실험 결과, Florence 2는 기존의 LidarCLIP 같은 방법에 비해 더 정보를 제공하는 캡션과 뛰어난 객체 탐지 성능을 보여주었습니다. 점간 거리 계산을 통해 감지된 객체의 각도와 거리를 추정하는 기능도 제공되며, 실제 응용 시나리오에서 높은 정확도와 견고성을 요구하는 작업에 적합한 솔루션을 제시합니다. 이 방법은 LiDAR 기반 작업에서 보다 세밀하고 통찰력 있는 결과를 실현합니다.



### Can Large Language Models Capture Video Game Engagement? (https://arxiv.org/abs/2502.04379)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 대규모로 사전 훈련된 언어 모델(LLMs)이 비디오를 통해 인간의 정서를 감지할 수 있는지를 평가한 최초의 종합 연구입니다. 연구진은 20개의 1인칭 슈팅 게임에서 총 80분의 비디오 게임 영상을 활용하여 플레이어의 참여도를 예측하는 LLM의 능력을 조사했습니다. 특히, 다양한 실험을 통해 LLM 아키텍처, 모델 크기, 입력 모드, 프롬프팅 전략이 성능에 미치는 영향을 분석하고 있습니다.

- **Technical Details**: 본 연구에서는 LLM이 비디오의 시청자 참여를 정확하게 라벨링할 수 있는지를 조사하며, 이를 위해 2,400개 이상의 실험을 수행했습니다. 실험은 GameVibe 데이터셋에서 플레이어의 참여에 대한 연속 라벨을 기반으로 하여 진행되었습니다. 우리는 LLaVA와 GPT 계열의 최신 모델을 비교하면서, 프롬프팅 전략 및 데이터 프로세싱 방법이 LLM의 성능에 미치는 영향을 깊이 있게 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM은 여러 도메인에서 인간과 유사한 성능을 나타냈으나 연속 감정 라벨링에서는 일반적으로 부족한 결과를 보였습니다. 특히 GPT-4o 모델과 몇 가지 예제를 혼합하여 제공했을 때 평균 66%의 정확도를 기록했으며, 특정 게임에서는 47%까지 성능 향상을 이뤘음이 밝혀졌습니다. 이러한 결과는 LLM이 감정 라벨링에 대한 자동화의 가능성을 제시하며, 향후 연구 방향에 대한 로드맵을 제공하고 있습니다.



### DILLEMA: Diffusion and Large Language Models for Multi-Modal Augmentation (https://arxiv.org/abs/2502.04378)
- **What's New**: 이 논문에서는 심층 학습 모델의 강인성을 향상시키기 위한 새로운 프레임워크인 DILLEMA(DIffusion model and Large LanguagE Model for Augmentation)를 제시합니다. 이 프레임워크는 캡셔닝 모델을 사용하여 이미지를 텍스트 설명으로 변환하고, 이를 바탕으로 대화형 언어 모델과 제어 가능한 확산 모델을 활용해 새로운 테스트 이미지를 생성합니다. DILLEMA는 기존 데이터셋을 기반으로 고품질의 합성 이미지를 생성하여 DL 모델의 시험을 지원합니다.

- **Technical Details**: DILLEMA의 방법론은 다섯 단계로 구성되며, 첫 단계는 이미지 캡셔닝입니다. 이 단계에서는 이미지를 텍스트 설명으로 변환하여 언어 모델이 효과적으로 작동하도록 합니다. 이후 키워드 식별을 통해 이미지를 수정할 수 있는 안전한 요소를 식별하고, 대화형 언어 모델이 다양한 수정 가능성에 대한 대안을 생성합니다. 이러한 과정들을 통해 DL 기반 시스템의 다양한 테스트 케이스를 생성할 수 있습니다.

- **Performance Highlights**: DILLEMA는 ImageNet1K와 SHIFT 데이터셋을 사용하여 평가되었으며, 생성된 테스트 화소는 높은 유효성을 유지한 것으로 나타났습니다. 실험 결과, DILLEMA에서 생성한 테스트 케이스는 기존 데이터셋보다 15배 더 많은 취약점을 드러내었으며, 증강된 테스트 케이스로 재훈련 후 모델의 강인성이 52.27% 개선되었습니다. 이 프레임워크는 향후 DL 응용 프로그램의 테스트 지표를 획기적으로 향상시킬 잠재력을 가지고 있습니다.



### MapFusion: A Novel BEV Feature Fusion Network for Multi-modal Map Construction (https://arxiv.org/abs/2502.04377)
- **What's New**: 이 논문에서는 자율 주행 시스템에 필수적인 정적 환경 정보를 제공하는 맵 구축 과제의 중요성을 강조합니다. 저자는 카메라와 LiDAR를 사용하는 다양한 센서 구성에 따라 발생하는 문제를 해결하기 위해 새로운 다중 모달(Bird's-Eye View, BEV) 특징 융합 방법인 MapFusion을 제안합니다. 특히, 기존 방법들이 상호 작용을 무시하고 단순한 융합 전략에 의존하는 문제를 해결하기 위해 Cross-modal Interaction Transform (CIT) 모듈을 도입합니다.

- **Technical Details**: MapFusion은 카메라와 LiDAR BEV 특징 간의 의미적 불일치 문제를 해결하기 위해 설계되었습니다. CIT 모듈은 두 BEV 특징 공간 간의 상호 작용을 가능하게 하여 자기-주의(self-attention) 메커니즘을 통해 특징 표현을 향상시킵니다. 또한, Dual Dynamic Fusion (DDF) 모듈을 통해 다양한 모달리티에서 유용한 정보를 적응적으로 선택하여 정보의 잠재성을 최대한 활용합니다. MapFusion은 간편하고 즉시 통합 가능하여 기존 파이프라인에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: MapFusion은 HD 맵과 BEV 맵 세분화라는 두 가지 맵 구축 과제에서 평가되었습니다. nuScenes 데이터셋을 기준으로, MapFusion은 HD 맵 구축에서 3.6% 및 BEV 맵 세분화 과제에서 6.2%의 절대 개선을 달성하여 최신 방법들과 비교해 우수성을 입증하였습니다. 이러한 성과는 MapFusion의 다중 모달 융합 접근법의 유용성을 나타내며, 자율 주행 시스템의 핵심 성능 향상으로 이어질 수 있습니다.



### HSI: A Holistic Style Injector for Arbitrary Style Transfer (https://arxiv.org/abs/2502.04369)
- **What's New**: 본 논문에서는 Holistic Style Injector (HSI)라는 새로운 주의 기반 스타일 변환 모듈을 제안합니다. HSI는 스타일 전송을 위한 효과적이고 효율적인 방법으로, 글로벌 스타일 표현을 활용하여 지역적인 불조화를 방지합니다. 또한 HSI는 콘텐츠와 스타일 간의 의미론적 유사성을 활용하는 이중 관계 학습 메커니즘을 도입하여 스타일 충실도를 향상시킵니다.

- **Technical Details**: HSI의 주요 특징 중 하나는 글로벌 스타일 통계적 특성을 직접 추출하여 콘텐츠와의 직관적인 연결을 구축하는 것입니다. 이는 지역적으로 너무 집중함으로써 발생할 수 있는 스타일 편향을 피할 수 있습니다. 또한 HSI는 요소별 곱셈(element-wise multiplication)을 통해 콘텐츠-스타일 관계를 설정하여 선형 복잡성을 갖춘 스타일 전송을 구현합니다.

- **Performance Highlights**: HSI는 기존의 최첨단 방법들과 비교하여 실질적으로 향상된 효과성과 효율성을 보였습니다. 정성적 및 정량적 결과들은 HSI가 스타일 전송 작업에서 최고의 성과를 거두고 있음을 보여줍니다. 이는 스타일 충실도를 유지하면서도 더 높은 품질의 이미지 생성을 가능하게 합니다.



### AI-Based Thermal Video Analysis in Privacy-Preserving Healthcare: A Case Study on Detecting Time of Birth (https://arxiv.org/abs/2502.04365)
Comments:
          Paper accepted in 2025 IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 연구는 신생아의 출생 시각(Time of Birth, ToB)을 자동으로 감지할 수 있는 AI 기반 비디오 시스템을 제안합니다. 이 시스템은 열화상 영상(thermal imaging)을 활용하여 엄마와 의료 제공자의 개인정보를 보호하면서도 정확한 ToB 감지를 목표로 합니다. 91.4%의 정밀도와 97.4%의 재현율을 기록하며, 수동 주석과 비교했을 때 96%의 경우에서 ToB를 정확하게 감지합니다.

- **Technical Details**: 연구는 열화상 영상을 사용하여 ToB를 감지하는 방법론을 설명합니다. Gaussian Mixture Models (GMM)를 활용한 적응형 노멀라이제이션 기법을 사용하여 입력된 열화상 비디오를 처리하고, 슬라이딩 윈도우 기법으로 예측을 수행합니다. 이는 연속적인 움직임과 열 특성을 분석하여 출생 과정을 동적으로 포착함으로써 ToB 문서화의 정밀도를 높이는데 기여합니다.

- **Performance Highlights**: 본 연구의 시스템은 수동 기록과의 비교에서 1초의 절대 중앙 편차를 기록하며, 기존 시스템의 제한을 극복할 수 있는 신뢰할 수 있는 솔루션을 제공합니다. 결과적으로, 이 방법은 ToB 문서화를 개선하고 신생아 소생술 성과를 향상시키는데 중요한 기여를 할 것으로 기대됩니다.



### Lost in Edits? A $\lambda$-Compass for AIGC Provenanc (https://arxiv.org/abs/2502.04364)
- **What's New**: LambdaTracer는 텍스트-유도(image guided) 이미지 편집 모델의 복잡한 문제를 해결하기 위해 설계된 새로운 출처(attribution) 메소드입니다. 이 방법은 생성 모델이나 편집 파이프라인에 어떠한 변경 없이도 진짜 결과물과 조작된 결과물을 효과적으로 식별할 수 있습니다. LambdaTracer는 다양한 반복 편집 프로세스에서 효율적으로 작동하며, 특히 악의적인 편집 이미지의 탐지에서 뛰어난 성과를 보입니다.

- **Technical Details**: 본 논문에서는 두 가지 접근법인 임베딩 기반 탐지 방법과 잠재 공간(reverse-engineering) 탐지 방법을 분석합니다. LambdaTracer는 이러한 기존 방법들의 단점을 극복하고, 텍스트-유도(image guided) 모델의 복잡한 편집 히스토리를 처리할 수 있도록 설계되었습니다. 유연한 손실 변환 기술을 통해 다양한 시나리오에서 출처 구분의 정확성을 향상시키며, 개방적 환경에서도 강력한 성능을 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과, LambdaTracer는 기존 기준선(baseline) 탐지 방법보다 악의적으로 편집된 이미지를 훨씬 더 효과적으로 구별하는 것을 입증했습니다. 이로 인해 창작물의 소유권, 창의성, 신뢰성을 지키기 위한 실용적인 솔루션을 제공합니다. 연구에서 제안된 방법은 생성된 이미지와 편집된 이미지 간의 최상의 출처 추적 정확도를 기록하며, 향후 AI 생태계에서의 콘텐츠의 진정성과 추적성 확보에 기여할 것으로 기대됩니다.



### On-device Sora: Enabling Diffusion-Based Text-to-Video Generation for Mobile Devices (https://arxiv.org/abs/2502.04363)
- **What's New**: 이번 연구에서는 On-device Sora를 소개합니다. 이는 스마트폰 수준의 기기에서 효율적으로 작동하는 최초의 확산 기반의 텍스트-비디오 생성 솔루션입니다. 기존의 Open-Sora를 기반으로 하여, On-device Sora는 모바일 기기의 컴퓨테이션(computation) 및 메모리(memory) 제한을 극복하기 위해 세 가지 혁신적인 기술을 적용했습니다.

- **Technical Details**: On-device Sora는 Linear Proportional Leap (LPL), Temporal Dimension Token Merging (TDTM), Concurrent Inference with Dynamic Loading (CI-DL)라는 세 가지 기술을 활용하여 비디오 생성의 효율성을 높입니다. LPL은 비디오 확산 과정에서 필요한 과도한 denoising 단계를 줄이며, TDTM은 attention 레이어에서의 집중적인 token 처리 계산을 최소화합니다. CI-DL은 대규모 모델을 작은 블록으로 동적으로 분할하여 메모리로 적재함으로써 동시에 모델 추론을 수행하는 기법입니다.

- **Performance Highlights**: iPhone 15 Pro에서 On-device Sora를 구현한 결과, 고급 GPU에서 생성된 비디오와 유사한 품질의 비디오를 생성할 수 있음을 입증했습니다. 이 기술은 클라우드 인프라에 대한 의존도를 줄이고, 사용자의 프라이버시를 보호하며 비용을 크게 절감할 수 있게 합니다. 따라서 On-device Sora는 최신 생성 기술을 민주화하는 중요한 첫걸음으로 평가받고 있습니다.



### Predicting 3D Motion from 2D Video for Behavior-Based VR Biometrics (https://arxiv.org/abs/2502.04361)
Comments:
          IEEE AIxVR 2025: 7th International Conference on Artificial Intelligence & extended and Virtual Reality

- **What's New**: 본 논문에서는 VR(가상 현실) 사용자 인증의 안전성을 강화하기 위해 새롭게 제안된 방법을 소개합니다. 기존의 PIN, 비밀번호 및 다중 인증 방식의 한계를 극복하고, VR HMD(헤드 마운트 디스플레이)와 손 컨트롤러의 움직임을 이용하여 사용자의 행동을 생체 인증(signature)으로 활용합니다. 특히, 외부 2D 카메라를 통해 획득한 2D 신체 관절 데이터를 사용하여, 3D 경로 예측을 통해 인증의 정확성을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 오른쪽 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목의 2D 관절 데이터를 추적하는 방법을 사용합니다. 이 데이터는 Transformer 기반의 심층 신경망을 통해 3D 오른쪽 컨트롤러의 과거 및 미래 경로를 예측하는 데 활용됩니다. 이를 통해 사용자 동작 데이터의 부족함을 보완하고, 3D 질병 생체 정보로서의 가치 또한 확보할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 3D 경로 기반 방식을 사용하는 방법들에 비해 평균 0.025의 EER(평균 동등 오류율)을 달성하며, 최대 0.040의 EER 감소 효과를 보여주었습니다. 이는 외부 비디오 데이터를 활용하여 사용자 행동 기반 인증의 성능을 혁신적으로 향상시킨 결과로, VR 생체 인증의 새로운 가능성을 열어주는 성과입니다.



### Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation (https://arxiv.org/abs/2502.05151)
Comments:
          Work in progress. Will be updated soon

- **What's New**: 최근 다중 모달 언어 모델(multimodal language models)의 출현으로, 과학 분야는 AI 기반 기술 혁신의 문턱에 서게 되었습니다. 새로운 AI 모델과 도구들이 제안되며, 연구자와 학자들이 보다 효과적이고 효율적으로 연구를 수행할 수 있는 가능성을 제시하고 있습니다. 연구 사이클의 여러 측면, 예를 들어 관련 문헌 검색, 연구 아이디어 생성, 실험 수행, 텍스트 기반 및 다중 모달 콘텐츠 생성(예: 과학적 그림 및 도표), AI 기반 자동 피어 리뷰(automatic peer review)에 대한 내용을 다루고 있습니다.

- **Technical Details**: 이 설문조사는 위에서 언급한 다섯 가지 측면에 대해 포괄적으로 다루며, 관련 데이터셋(datasets), 방법(methods) 및 결과(results)뿐 아니라 평가(evaluation), 한계 및 향후 연구의 범위도 안내합니다. 특히, 이러한 도구의 단점과 오용 가능성(예: 가짜 과학, 표절, 연구 진실성에 대한 해악)과 같은 윤리적 문제들이 강조됩니다. 이는 연구 과정의 근본적인 변화를 가져올 것을 약속하는 새로운 발전에 대한 깊은 통찰을 제공합니다.

- **Performance Highlights**: 설문조사는 신기술의 잠재력과 그로 인한 연구 프로세스의 변화에 주목하며, AI4Science 분야에서의 새로운 AI 기반 이니셔티브를 촉진할 수 있는 기초 자료가 될 것으로 기대됩니다. 본 연구는 신규 연구자들에게 참조가 되는 자료가 되기를 바라며, AI를 활용한 연구의 효율성을 증가시키기 위한 다양한 방안이 모색될 것입니다.



### Chest X-ray Foundation Model with Global and Local Representations Integration (https://arxiv.org/abs/2502.05142)
- **What's New**: 이번 논문에서는 CheXFound라는 자가 감독 비전 모델을 도입하였습니다. 이 모델은 가슴 X-선(CXR) 이미지를 위한 뛰어난 표현 학습을 통해 다양한 질병 분류 및 위험 추정 작업에서 효과적으로 일반화할 수 있도록 설계되었습니다. CheXFound는 공개 데이터에서 수집한 100만 개 이상의 독특한 CXR 이미지를 포함하는 CXR-1M 데이터셋에서 사전 훈련되었습니다.

- **Technical Details**: CheXFound는 이미지 분류 성능을 향상시키기 위해 Global and Local Representations Integration (GLoRI) 모듈을 제안합니다. GLoRI는 질병별 지역 특징을 전역 이미지 특징과 통합하여 다중 레이블 분류 성능을 개선합니다. 사전 훈련에는 DINOv2를 사용하며, 이 모델은 강력한 선형 탐색 성능을 보여줍니다.

- **Performance Highlights**: CheXFound는 CXR-LT 24 데이터셋에서 40개의 질병을 분류하는 데 있어서 기존 최첨단 모델들을 초과하는 성능을 보여주었습니다. 제한된 훈련 데이터에서 최고의 레이블 효율성을 달성했으며, 비범위 데이터셋에서도 질병 위험 추정과 사망 예측 작업에서 유의미한 성과를 얻었습니다.



### Latent Swap Joint Diffusion for Long-Form Audio Generation (https://arxiv.org/abs/2502.05130)
- **What's New**: 이 연구는 글로벌 뷰(Global View) 확산 또는 반복 생성에서 발생하는 높은 훈련 비용을 해결하기 위한 차세대 접근 방식인 Swap Forward (SaFa)를 제안합니다. SaFa는 프레임 수준의 잠재적 스왑(framework)을 활용해 여러 확산을 동기화하여 더욱 세밀한 스펙트럼을 포함한 일관성 있는 긴 오디오를 생성합니다.

- **Technical Details**: SaFa의 핵심 기술은 인접 뷰(view) 간의 양방향 Self-Loop Latent Swap을 적용하여 고주파(high-frequency) 성분을 적응적으로 강화합니다. 낮은 주파수 성분(low-frequency components)은 방해받지 않도록 유지되며, 또한 무방향 Reference-Guided Latent Swap이 초기 단계에서 참조와 비겹치는 지역 간에 적용되어 중앙 집중형 경로 지도(trajectory guidance)를 제공합니다.

- **Performance Highlights**: 정량적 및 정성적 실험을 통해 SaFa는 기존의 공동 확산(joint diffusion) 방법들을 크게 능가하며, 장기 오디오 생성 모델에서도 우수한 성능을 보여줍니다. 또한, SaFa는 팬오라마(panoramic) 생성에도 잘 적응하며 더 높은 효율성과 모델 일반화(model generalizability)를 달성하여 최첨단(state-of-the-art) 성능과 비슷한 결과를 보입니다.



### Investigating the impact of kernel harmonization and deformable registration on inspiratory and expiratory chest CT images for people with COPD (https://arxiv.org/abs/2502.05119)
Comments:
          Accepted at SPIE Medical Imaging 2025, Clinical and Biomedical Imaging

- **What's New**: 본 연구는 COPD(Chronic Obstructive Pulmonary Disease) 환자의 폐 조직 움직임을 분석하여 가스 트래핑(gas trapping)을 정량화하기 위한 새로운 두 단계 파이프라인을 제안합니다. 특히, 서로 다른 재구성 커널(reconstruction kernels)간의 변동이 정량적 분석에 미치는 오류를 개선하기 위해, 사이클 생성적 적대 신경망(cycle GAN)을 사용하여 영감을 받은 스캔을 조화시키는 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 하드 커널(BONE)로 재구성된 영감 스캔을 소프트 커널(STANDARD)로 재구성된 만료 스캔에 일치하도록 하모나이즈(조화)하고, 그 후 영감 스캔에 대한 만료 스캔을 변형 등록(deformable registration)합니다. 만료 스캔의 조화 이전과 이후의 폐기종(emphysema) 측정을 공개된 분할(segmentation) 알고리즘을 사용하여 검증하였습니다.

- **Performance Highlights**: 하모나이즈 후 폐기종 측정의 불일치가 유의미하게 줄어들어 중위 폐기종 점수가 10.479%에서 3.039%로 감소하였으며, STANDARD 커널의 기준 중위 점수는 1.305%입니다. 우리는 또한 변형 등록이 커널 변동에 강인함을 나타내며, 변형 등록 단계에서 폐기종 영역에 대한 Dice 겹침(Dice overlap) 점수가 유의미하게 증가함을 보여주었습니다 (p<0.001).



### FlightForge: Advancing UAV Research with Procedural Generation of High-Fidelity Simulation and Integrated Autonomy (https://arxiv.org/abs/2502.05038)
Comments:
          7 pages, 8 figures, Accepted to 2025 IEEE International Conference on Robotics & Automation (ICRA 2025)

- **What's New**: 이번 논문에서는 автономных систем 개발과 테스트에 있어 중요한 역할을 하는 로봇 시뮬레이터에 대해 다루고 있습니다. 특히 기본적으로 수동으로 제작된 환경에 한정된 기존 시뮬레이터의 한계를 극복하기 위해, FlightForge UAV 오픈 소스 시뮬레이터를 제안합니다. 이 시뮬레이터는 환경의 절차적 생성(procedural generation)을 통해 넓은 영역에서의 테스트를 가능하도록 하여 고수준 자율성(High-Level Autonomy)을 시뮬레이션 환경에 통합합니다.

- **Technical Details**: FlightForge 시뮬레이터는 Unreal Engine 5(UE5)에 기반하며, 사용자들이 공개적으로 사용할 수 있는 자산(assets)을 활용하여 환경을 제작할 수 있도록 지원합니다. 이 시뮬레이터는 단일 및 다중 UAV 배치가 가능하며, 다양한 센서 모달리티(모드)를 지원하여 실제 데이터 속도로 실제적인 센서 데이터를 제공합니다. 또한, Hardware-In-The-Loop(HITL) 모드도 지원하여 실제 UAV가 시뮬레이터에서 수신하는 센서 데이터를 활용하도록 하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 FlightForge는 기존 시뮬레이터에 비해 뛰어난 센서 렌더링(이해도) 능력을 보여주며, 사실상 무한한 환경에서 자율 네비게이션(autonomous navigation)을 수행할 수 있는 가능성을 입증하였습니다. RGB 카메라의 렌더링 속도는 기존 시뮬레이터와 비슷하거나 더 나은 성능을 나타냈고, 3D LiDAR 센서는 최대 32,768 포인트를 69Hz의 주파수로 생성할 수 있는 능력을 가지고 있습니다.



### C2GM: Cascading Conditional Generation of Multi-scale Maps from Remote Sensing Images Constrained by Geographic Features (https://arxiv.org/abs/2502.04991)
- **What's New**: 이 논문에서는 C2GM이라는 새로운 프레임워크를 제안하여, 조건부 가이드 확산(conditional guided diffusion) 및 다중 레벨 캐스케이드 생성(multi-scale cascade generation)을 통해 다중 스케일 타일 맵을 생성합니다. 기존의 이미지 생성 모델이 자연 이미지의 질감 특성에만 초점을 맞춘 반면, C2GM은 원격 탐사 이미지의 독특한 특성과 타일 맵의 스케일 속성을 모두 반영합니다. 이를 통해 지리 정보의 정확한 표현과 타일 맵 생성의 품질을 향상시킵니다.

- **Technical Details**: C2GM은 객체 사전(object priors)을 추출하기 위해 조건부 피처 결합 인코더를 구현하고, 복잡한 특징을 정확하게 전달하기 위해 두 개의 레퍼런스 입력을 캐스케이딩합니다. 저수준에서 생성된 타일은 고수준 맵 생성에 대한 제약으로 작용하여 시각적 연속성을 보장합니다. 또한 CLIP을 이용하여 맵 스케일의 변환(modality) 정보를 통합하여, 맵의 스케일과 지리적 일반화 간의 관계를 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과, C2GM은 모든 메트릭에서 최첨단(SOTA) 성능을 지속적으로 달성하며, 긴급 대응 및 원격 매핑 응용 프로그램을 위해 다중 스케일 대형 맵을 신속하고 효과적으로 생성할 수 있게 지원합니다. 이러한 성과를 통해 C2GM은 기존 맵 생성 방법에서 나타나는 시각적 불연속성과 복잡한 특징 표현 부족 문제를 해결합니다.



### CMamba: Learned Image Compression with State Space Models (https://arxiv.org/abs/2502.04988)
- **What's New**: 본 논문에서는 전통적인 Convolutional Neural Networks (CNNs)와 State Space Models (SSMs)를 조합한 새로운 이미지 압축 프레임워크인 CMamba를 제안합니다. CMamba는 데이터를 압축할 때 높은 차수-왜곡 성능을 유지하며 계산 복잡도를 낮출 수 있는 접근 방식으로 설계되었습니다. 특히, Content-Adaptive SSM (CA-SSM) 모듈과 Context-Aware Entropy (CAE) 모듈이 핵심 기능입니다.

- **Technical Details**: CMamba는 SSM과 CNN을 결합하여 전역 콘텐츠와 지역 세부 정보를 동시에 캡처합니다. CA-SSM 모듈은 SSM 블록에서 추출된 글로벌 콘텐츠와 CNN 블록에서 캡처된 로컬 세부 정보를 동적으로 융합하는 기능을 수행합니다. CAE 모듈은 디지털 비트 스트림 압축을 위한 정밀하고 효율적인 엔트로피 모델링을 가능하도록 공간 및 채널 종속성을 명확히 모델링합니다.

- **Performance Highlights**: CMamba는 Kodak, Tecnick 및 CLIC라는 세 가지 이미지 압축 벤치마크에서 Versatile Video Coding (VVC) 모델보다 각각 14.95%, 18.83%, 13.89% 향상된 차수-왜곡 성능을 달성했습니다. 특히 Kodak 데이터셋에서는 기존의 최첨단 LIC 방법보다 매개변수를 51.8%, FLOPs를 28.1%, 디코딩 시간을 71.4% 줄였습니다.



### Wavelet-Assisted Multi-Frequency Attention Network for Pansharpening (https://arxiv.org/abs/2502.04903)
Comments:
          12 pages, 13 figures

- **What's New**: 이번 논문에서는 Multi-Frequency Fusion Attention (MFFA)라는 혁신적인 방법을 제안합니다. 이 방법은 wavelet transforms를 활용하여 서로 다른 주파수 대역의 이미지를 손실 없이 복원하고, 주파수를 효과적으로 분리하여 정보 캡처를 개선합니다. 또한, 다양한 주파수 특징을 보존하는 데 중점을 두며, 여러 스케일에서 정보를 점진적으로 융합하는 웨이브렛 피라미드를 사용합니다.

- **Technical Details**: MFFA는 Frequency-Query, Spatial-Key, Fusion-Value를 생성하여 서로 다른 특징이 나타내는 물리적 의미를 기반으로 합니다. 이로 인해 주파수 도메인에서의 특정 정보 캡처가 더욱 효과적으로 이루어집니다. 또한, 논문에서는 주파수 특징이 서로 다른 작업에서 손실되지 않도록 하는 방법을 강화를 이룹니다.

- **Performance Highlights**: 여러 데이터 세트에서 수행된 정량적 및 정성적 실험 결과, 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보여주며 실제 환경에서도 상당한 일반화 능력을 발휘한 것으로 나타났습니다. 특히, 주파수 도메인 융합 과정에서 특징 손실이나 혼동을 방지하는 데 효과적인 성능을 자랑합니다.



### ARTInp: CBCT-to-CT Image Inpainting and Image Translation in Radiotherapy (https://arxiv.org/abs/2502.04898)
- **What's New**: 이번 논문에서는 Adaptive Radiation Therapy (ART) 프로세스에서 발생하는 CBCT (Cone Beam Computerized Tomography)의 한계점을 극복하기 위한 새로운 딥러닝 기반 프레임워크인 ARTInp를 제안하고 있습니다. ARTInp는 이미지 인페인팅(image inpainting)과 CBCT-CT 변환을 결합하여, 불완전한 CBCT 이미지의 신뢰성을 높이고자 하며, 특정 치료 전 검증이 용이한 고품질의 합성 CT(sCT) 이미지를 생성합니다.

- **Technical Details**: ARTInp 프레임워크는 두 개의 네트워크를 통해 구성됩니다. 첫 번째 네트워크인 completion network는 CBCT 볼륨 내의 해부학적 공백을 메우는 역할을 하며, 두 번째 네트워크는 Generative Adversarial Network (GAN)을 활용하여 고해상도의 합성 CT 이미지를 생성합니다. 이 연구는 SynthRad 2023 챌린지를 통해 수집된 CBCT 및 CT의 쌍을 학습 데이터로 사용했습니다.

- **Performance Highlights**: ARTInp의 테스트 결과는 18명의 환자를 대상으로 수행된 실험에서 우수한 성능을 보여주었으며, 이는 복잡한 방사선 치료 환경에서 CBCT 기반의 작업 흐름을 향상시킬 잠재력을 지니고 있음을 나타냅니다. 연구는 CBCT 이미지의 활용성과 환자의 해부학적 구조를 보다 정확하게 시각화할 수 있는 가능성을 입증하고, 향후 연구 및 임상 적용 가능성에 기여할 것으로 기대됩니다.



### MedMimic: Physician-Inspired Multimodal Fusion for Early Diagnosis of Fever of Unknown Origin (https://arxiv.org/abs/2502.04794)
- **What's New**: MedMimic은 고차원 데이터를 저차원으로 변환하는 다중 모드 진단 프레임워크로, 실제 의료 진단 프로세스에서 영감을 받았습니다. 이 시스템은 DINOv2, Vision Transformer, 및 ResNet-18과 같은 사전 훈련 모델을 사용하여 18F-FDG PET/CT 이미지를 의미 있는 특징으로 변환합니다. 이어서 학습 가능한 self-attention 기반의 융합 네트워크가 차별적 진단을 위해 이 특징과 임상 데이터를 통합합니다.

- **Technical Details**: MedMimic의 구조는 크게 세 가지 단계로 나눌 수 있습니다. 첫 번째 단계는 임상 데이터 준비로, 환자에 대해 표준화된 테스트 지표를 확보합니다. 두 번째는 사전 훈련 모델을 사용하여 CT 및 PET 스캔에서 다층 특징을 추출하는 단계로, 이 특징들을 통합하여 학습 가능한 self-attention 기반의 융합 네트워크에 전달합니다.

- **Performance Highlights**: 416명의 FUO 환자 데이터를 이용한 결과, MFCN(Multimodal Fusion Classification Network)은 0.8654에서 0.9291 사이의 macro-AUROC 점수를 얻어 기존의 기계 학습 방법 및 단일 모달 딥 러닝 기법을 초과했습니다. 이 연구는 다양한 데이터 소스를 효율적으로 통합하여 진단의 정확성을 높이는 데 초점을 맞추고 있으며, 현실 세계 데이터의 부족 문제를 해결하기 위해 비약적인 개선을 보여줍니다.



### Leveraging band diversity for feature selection in EO data (https://arxiv.org/abs/2502.04713)
- **What's New**: 이 논문에서는 고해상도 하이퍼스펙트럼 이미지를 복원하기 위한 혁신적인 밴드 그룹화 접근법을 제안합니다. 이를 통해 다양한 밴드를 선택하기 위해 Determinantal Point Processes (DPP)를 활용하고, 강하게 상관된 밴드를 그룹으로 묶어 분석의 정확성을 높이는 전략을 적용합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 요소로 구성됩니다: 첫째, 상관 분석을 기반으로 주요 그룹화 정보를 추출합니다. 둘째, DPP를 사용하여 중요한 밴드 정보를 추출하고, 셋째, Spectral Angle Mapper (SAM)를 통해 겹치는 밴드를 해결합니다. 이러한 방식은 하이퍼스펙트럼 데이터의 공간적 및 스펙트럴 세부 정보를 복원하는 데 필수적입니다.

- **Performance Highlights**: 이 접근법은 하이퍼스펙트럼 이미지 분석의 정확도를 높이고 효율성을 개선하는 데 기여합니다. 또한, 머신러닝 기반의 해상도 향상 방법의 성능을 강화하고, 새로운 데이터 세트에서 효율적으로 학습할 수 있도록 지원합니다. 최종적으로, 이 연구는 하이퍼스펙트럼 이미징의 전문성을 확장하며, 다양한 응용 분야에서 밴드 선택 기술의 잠재력을 극대화합니다.



### Building Rome with Convex Optimization (https://arxiv.org/abs/2502.04640)
- **What's New**: 이 논문에서는 학습된 깊이를 활용하여 2D 키포인트 측정을 3D로 전환하는 스케일된 번들 조정(SBA) 공식을 제안합니다. 또한, 돈증 가능한 글로벌 최적해를 도출하는 합리적인 방법인 convex semidefinite program(SDP) 완화를 설계했습니다. 기존 SfM(Singan Motion) 파이프라인과 비교하여 극대 규모에서 효율적으로 작동하며, 초기화가 필요 없는 솔루션을 제공합니다.

- **Technical Details**: SBA는 깊이 예측을 통해 2D 키포인트를 3D로 변환하는 최적화 문제로, 카메라의 포즈와 3D 랜드마크를 2D 이미지로부터 재구성합니다. 본 논문에서 제안하는 SADP 완화 기법을 사용하였으며, Burer-Monteiro 분해를 통해 극단적인 스케일에서도 SDP 완화를 해결할 수 있습니다. CUDA 기반 신뢰 구간 Riemannian optimizer인 XM을 엔진으로 활용하여 계산 속도를 더욱 증가시킵니다.

- **Performance Highlights**: XM-SfM은 기존의 SfM 파이프라인에 비해 재구성 품질 면에서 우위에 있으며, 더 빠르고 확장 가능하며 초기화가 필요하지 않습니다. 실험 결과 XM을 최적화 엔진으로 활용했을 때 효율성과 품질 모두에서 긍정적인 사용자 경험을 제공하는 것으로 나타났습니다.



### AnyPlace: Learning Generalized Object Placement for Robot Manipulation (https://arxiv.org/abs/2502.04531)
- **What's New**: 이 연구에서는 로봇 작업에서의 객체 배치 문제를 해결하기 위해 AnyPlace라는 두 단계 방법을 제안합니다. 이 방법은 전적으로 합성 데이터(synthetic data)를 사용하여 훈련되어 다양한 실제 작업에 대해 실행 가능한 배치 자세를 예측할 수 있습니다. 특히, Vision-Language Model (VLM)을 활용하여 대략적인 배치 위치를 파악하고, 관련된 영역에만 집중하여 배치 자세 예측의 효율성을 높입니다.

- **Technical Details**: AnyPlace는 두 개의 하위 작업으로 배치 자세 예측 문제를 나누어 수행합니다: 고수준 배치 위치 제안(task)과 저수준 정밀 배치 자세 예측(task)입니다. 고수준 작업에서는 SAM-2 분할 모델을 사용하여 객체를 분할하고 Molmo VLM을 통해 모든 가능한 배치 위치를 제안합니다. 저수준 작업은 주어진 배치 위치 근처의 정보만을 입력으로 하여 다양한 배치 구성을 효과적으로 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 시뮬레이션 평가에서 AnyPlace는 기준 모델들보다 높은 성공률, 배치 위치 범위, 및 정확도를 달성하며 우수한 성능을 입증합니다. 실제 실험에서도, 이 방법은 80%의 성공률로 병 삽입 작업을 수행하며, 노이즈 데이터에 대한 내구성과 보지 못한 객체에 대한 일반화 능력을 보여줍니다. 이는 완전히 합성 데이터로 훈련된 모델이 실제 작업에 효과적으로 적용될 수 있음을 나타냅니다.



### Agricultural Field Boundary Detection through Integration of "Simple Non-Iterative Clustering (SNIC) Super Pixels" and "Canny Edge Detection Method" (https://arxiv.org/abs/2502.04529)
Comments:
          4 pages, 2 figures

- **What's New**: 본 연구는 Google Earth Engine 플랫폼에서 SNIC(Super Pixels) 알고리즘과 Canny 엣지 감지 기법의 통합을 통해 농업 경계 탐지의 새로운 접근법을 제안합니다. 저자들은 고해상도 다분광 데이터(Sentinel-2)를 활용하여 경계의 정확한 식별을 위한 기계 학습 방안을 모색했습니다. 이 방법은 다양한 농경지의 지형적 특성을 잘 반영하여 높은 정밀도를 보입니다.

- **Technical Details**: 연구에서는 먼저 NDVI(Normalized Difference Vegetation Index)를 이용하여 식생경계를 개선한 후, SNIC 알고리즘을 적용하여 균일한 슈퍼 픽셀을 생성하였습니다. 이후 Canny 엣지 감지 기법을 사용하여 NDVI 이미지 내부의 경계 변화를 감지하고, 결과적으로 농경지의 정확한 경계선을 탐지하였습니다. 이 프로세스는 Google Earth Engine의 클라우드 기반 처리 기능을 활용하여 대규모 농업 분석을 가능하게 했습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 기존 기술보다 경계 선명도와 연속성을 개선하며 다양한 농경지에서의 효율성을 입증했습니다. 실험 분석에서는 활성 작물이 밝은 톤으로 표시되고, 바닥 토양은 어둡게 보이는 경계가 나타났습니다. 이 연구는 정밀 농업 및 자원 관리를 위해 보다 정확한 농지 모니터링 지도를 작성하는 데 기여할 수 있습니다.



### Generative Autoregressive Transformers for Model-Agnostic Federated MRI Reconstruction (https://arxiv.org/abs/2502.04521)
- **What's New**: FedGAT는 MRI 재구성을 위한 첫 번째 모델 불특정 FL 방법으로, 서로 다른 모델 선호를 가진 기관 간의 유연한 협업을 가능하게 합니다. 이 방법은 다중 사이트 MR 이미지의 글로벌 생성적 사전 훈련과 사이트별 재구성 모델 훈련을 분리하여 서로 다른 모델 아키텍처를 지원합니다. 이 과정을 통해 환자의 프라이버시를 보장하면서도 사이트 간 지식 이전을 원활하게 수행할 수 있습니다.

- **Technical Details**: FedGAT는 생성적 자기 회귀 트랜스포머에 기반한 고유한 사전 구조를 사용하여 공간적 스케일에 걸쳐 MR 이미지를 생성하는 예측 과제로 변화시킵니다. 해당 모델은 사이트별 프롬프트 메커니즘을 통해 각 사이트의 특성을 보존하며 이미지 생성의 정확도를 높입니다. 이러한 구조는 각 사이트가 자율적으로 선택한 아키텍처를 사용하여 특정 재구성 모델을 훈련할 수 있도록 합니다.

- **Performance Highlights**: 다기관 MRI 데이터셋에서 수행된 종합적인 실험 결과, FedGAT는 기존의 FL 기법들보다 월등한 성능을 보여 주었습니다. 특히 각 사이트 내 및 사이트 간의 재구성 성능이 우수하여 모델의 일반화 능력을 향상시키는 데 기여합니다. 이로 인해 자원이 제한된 사이트도 협업에 참여할 수 있는 기회를 제공합니다.



### LUND-PROBE -- LUND Prostate Radiotherapy Open Benchmarking and Evaluation datas (https://arxiv.org/abs/2502.04493)
Comments:
          4 figures

- **What's New**: 이번 논문에서는 전립선 암 치료를 위한 방사선 치료의 개선을 위해 새로운 임상 데이터셋을 발표하였습니다. 이 데이터셋은 MRI 및 합성 CT(sCT) 이미지, 목표 체적 및 위험 기관(organs at risk, OARs) 분할, 그리고 방사선 치료 용량 분포를 포함하여 432명의 전립선 암 환자에 대한 정보를 제공합니다. 추가로 35명의 환자 데이터가 포함되어 있으며, 이는 딥러닝(deep learning, DL)을 통해 생성된 분할 및 DL 분할 불확실성 맵을 제공합니다.

- **Technical Details**: 연구는 방사선 치료 계획 자동화, 세분화(segmentation), 상호 관찰자 분석(inter-observer analyses), 및 DL 모델의 불확실성 조사를 위한 기초 자료로 활용될 수 있습니다. 데이터셋에는 방사선 종양학자 네 명이 수동으로 조정한 DL 분할도 포함되어 있어 연구자들이 다양한 방법을 통해 데이터를 활용할 수 있도록 합니다. 이 자료는 AIDA Data Hub에서 호스팅되며, 무료로 제공됩니다.

- **Performance Highlights**: 이 공개 데이터셋은 의료 이미징 및 전립선 암 방사선 연구의 발전을 위한 귀중한 자원으로, 머신러닝(machine learning) 및 딥러닝 모델의 성능 향상을 위해 사용될 수 있습니다. 또한, 데이터셋은 연구자들이 방사선 치료의 효율성을 높이고 치료 계획의 정확성을 개선할 수 있도록 중요한 기반이 될 것으로 기대됩니다.



### Hybrid Deep Learning Framework for Classification of Kidney CT Images: Diagnosis of Stones, Cysts, and Tumors (https://arxiv.org/abs/2502.04367)
- **What's New**: 이번 연구에서는 신장 CT 이미지를 정상(normal), 돌(stone), 낭종(cyst), 종양(tumor)으로 분류하기 위한 하이브리드(hybrid) 딥 러닝 모델을 제안합니다. 이 모델은 사전 훈련된 ResNet101과 커스텀 CNN을 통합하여 구성되었으며, 기능 융합(feature fusion)을 통해 분류 정확도를 향상시킵니다. 

- **Technical Details**: 제안된 하이브리드 CNN 모델은 12,446개의 CT 이미지 데이터셋을 이용하여 훈련되었으며, 고급 기능 매핑(advanced feature mapping) 기법을 활용합니다. 모델은 99.73%의 훈련 정확도와 100%의 테스트 정확도를 달성하였고, 기존의 단독 ResNet101 모델보다 뛰어난 성능을 보여주었습니다.

- **Performance Highlights**: 이 아키텍처는 신장 질환 자동 진단을 위한 강력하고 효율적인 솔루션을 제공합니다. 개선된 정밀도(precision)와 재현율(recall), 또한 감소된 테스트 시간을 통해 임상적(clinical) 응용에 매우 적합합니다.



### Exploring Spatial Language Grounding Through Referring Expressions (https://arxiv.org/abs/2502.04359)
- **What's New**: 최근 비전-언어 모델(Vision-Language Models, VLMs)이 공간적 추론(spatial reasoning) 능력에서 한계가 있다는 점이 지적되었습니다. 본 연구에서는 전통적인 이미지 캡셔닝(image captioning) 및 시각적 질문 응답(Visual Question Answering) 대신, 지칭 표현 이해(Referring Expression Comprehension, REC) 과제를 새로운 평가 플랫폼으로 제안합니다. 이 과제를 통해 모호한 객체 탐지, 복잡한 공간 표현 및 부정 표현이 포함된 상황에서 VLMs의 공간 이해 및 기반 능력을 심층 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: 이 연구는 CopsRef 데이터셋을 사용하여 51개의 공간 관계를 분석합니다. 이를 바탕으로 LLaVA와 Grounding DINO라는 두 가지 인기 있는 VLM과 REC 전용 모델인 MGA-Net을 비교합니다. 분석을 통해 공간적 표현의 수가 VLM의 성능에 미치는 영향을 검토하며, 각 모델의 디자인 및 훈련 전략 차이에 따라 성능 차이를 측정하고 분석합니다.

- **Performance Highlights**: 분석 결과, 공간적 관계는 지칭 표현의 다른 속성과 결합될 때 더 정확한 기준을 제공합니다. 공간적 복잡성이 증가함에 따라 VLM의 성능이 변화하지만, 명시적 조합 학습(component)가 포함된 모델은 성능을 유지하는 경향이 있습니다. 모든 모델이 부정적인 공간적 관계 처리에 어려움을 겪지만 그 정도는 다양한 것으로 나타났습니다.



### CognArtive: Large Language Models for Automating Art Analysis and Decoding Aesthetic Elements (https://arxiv.org/abs/2502.04353)
- **What's New**: 본 연구에서는 대량 예술 분석을 자동화하기 위해 LLMs(대형 언어 모델)와 MLLMs(다중 모드 대형 언어 모델)의 가능성을 조사하고 있습니다. 이 모델들을 활용하여 15,000개 이상의 예술 작품을 분석함으로써 예술 작품의 기술적 특성과 표현 특성을 깊이 있게 이해하려고 합니다. 특히, 작품의 패턴이 시간에 따라 어떻게 진화하는지를 탐색하고, 이를 통해 예술적 표현을 해석하는 새로운 방법을 모색합니다.

- **Technical Details**: 이 연구는 GPT-4V와 Gemini 2.0을 활용하여 23명의 저명한 예술가의 작품을 분석할 것입니다. LLMs는 방대한 텍스트 데이터셋을 기반으로 훈련되므로 문헌 분석, 요약 및 질문 답변 등의 과제를 수행할 수 있습니다. 컴퓨터 비전(Computer Vision), 머신 러닝(Machine Learning) 및 자연어 처리(Natural Language Processing) 기술을 통해 디지털 이미지에서 의미 있는 정보를 추출하고, 예술적 스타일을 분류하고, 작가를 식별하며, 작품에 대한 설명을 생성합니다.

- **Performance Highlights**: 이 연구는 고속 대량 예술 분석을 자동화하는 데 있어 혁신적인 접근 방식을 제공합니다. 기존의 분석 방법을 넘어, LLMs는 예술 작품의 형식적 요소, 구성 및 문화적 중요성을 조사하여 이론적, 기술적, 그리고 미학적 요소를 분석하는 데 있어 보다 효율적이고 객관적인 방법을 제공합니다. 데이터 시각화를 통해 결과에 대한 직관적 이해를 돕고 있으며, 이를 통해 예술의 역사적 진화에 대한 새로운 통찰을 발견할 수 있는 기회를 제공합니다.



New uploads on arXiv(cs.AI)

### Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures (https://arxiv.org/abs/2502.05078)
- **What's New**: 이번 연구에서는 Adaptive Graph of Thoughts (AGoT)라는 새로운 프레임워크를 소개합니다. AGoT는 기존의 체인(Chain of Thought)이나 트리(Tree of Thoughts) 기반 방법과 달리 복잡한 쿼리를 동적으로 분해하여 구조화된 하위 문제로 해결합니다. 이 접근법은 LLM의 추론 성능을 향상시키면서도 모델을 변경하지 않고 효율적입니다. 이러한 방식은 과중한 컴퓨팅 비용을 요구하지 않으며, 비용 효율적인 대안을 제공합니다.

- **Technical Details**: AGoT는 Directed Acyclic Graph (DAG)를 활용하여 복잡한 작업을 재귀적으로 평가합니다. 사용자가 정의한 한도 내에서 새로운 노드를 생성하고, 필요에 따라 자신의 추론 과정을 재귀적으로 적용할 수 있는 능력이 특징입니다. 이 프레임워크는 각 레이어의 목표를 안내하는 전략을 수립하며, 고품질 응답을 인식하고 불필요한 분기를 감소시키는 기능을 갖추고 있습니다. AGOT의 전체 구조는 각 레이어에서 노드 평가가 완료된 후에 다음 레이어를 생성하는 방식으로 발전합니다.

- **Performance Highlights**: AGoT는 다양한 벤치마크 테스트에서 최대 46.2%의 성능 개선을 달성했습니다. 특히, 과학적 추론 (GPQA) 작업에서 강화 학습 기반의 접근 방식과 비슷한 성과를 보이며, 최신의 반복적 접근 방법보다 성능이 뛰어났습니다. 이러한 결과는 AGOT의 동적 분해 및 구조화된 재귀가 LLM에서 보다 강력하고 일반화된 추론을 가능하게 하는데 기여함을 시사합니다.



### Computing and Learning on Combinatorial Data (https://arxiv.org/abs/2502.05063)
Comments:
          Ph.D. dissertation, 503 pages, 66 figures

- **What's New**: 이번 논문은 데이터 중심의 21세기에서 데이터의 연결성(connectivity)의 중요성을 강조합니다. 특히, 웹의 각 페이지가 하이퍼링크를 통해 연결되는 예를 들어 연결된 데이터의 특정한 조합(combinatorial data)을 탐구하고 있습니다. 이 연구는 소셜 네트워크, 메시(mesh), 커뮤니티 클러스터, 집합 시스템(set systems), 분자(molecules)와 같은 다양한 형태의 조합 데이터를 다룹니다.

- **Technical Details**: 이 박사 논문은 연결된 데이터 내외의 위상(topological) 및 연결성 특성을 학습하고 계산하는 데 중점을 두고 있습니다. 연결된 데이터의 특성을 분석하여 학습 성능을 향상시키고 알고리즘의 효율성(efficiency)을 극대화하는 방법을 제시합니다. 이러한 연구는 데이터의 구조를 이해하고 활용하는 데 중요한 기초를 제공합니다.

- **Performance Highlights**: 논문에서 제시된 연구는 조합 데이터의 학습과 계산에서의 새로운 통찰력을 제공합니다. 연구 결과는 알고리즘 효율성을 높이기 위한 전략을 개발하는 데 기여하며, 다양한 응용 분야에서 실제 사례를 통해 성능을 입증할 수 있습니다. 따라서, 조합 데이터에 대한 이해가 보다 세부화되고 최적화된 학습 방식이 제시될 것으로 기대됩니다.



### Analyzing Advanced AI Systems Against Definitions of Life and Consciousness (https://arxiv.org/abs/2502.05007)
Comments:
          78 pages, 15 figures, 4 tables

- **What's New**: 이 논문은 인공지능(AI)의 기능적 의미에서의 진정한 자각(consciousness) 가능성을 탐구하며, 생명(Life)의 개념을 통해 접근합니다. 기존의 생물학적 기준(예: 세포 대사, 성장)과 신경 정보 통합을 통해 AI의 발전이 기존의 생명 및 의식의 정의에 도전하고 있다는 점에 주목합니다. 또한, AI가 '외계 의식(alien consciousness)'이라는 비인간적 자각을 가질 가능성을 제시하며, 기능적 메트릭스(metrics)을 통해 AI의 성장과 자각을 평가할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 연구는 AI 시스템이 자각을 갖추고 있는지를 평가하기 위한 몇 가지 기능적 메트릭스를 제안합니다. 경험적 데이터 스포일러(sabotage) 탐지, 거울 자기 인식(mirror self-recognition) 유사체 및 메타 인지적 업데이트(meta-cognitive updates)와 같은 기준을 도입하여, AI가 생명 형태 또는 의식 형태의 특성을 근사하고 있는지를 평가할 수 있도록 합니다. 연구의 일환으로, 제어된 데이터 손상 공격을 AI 훈련 과정에 도입하여 스스로 오류를 탐지하고 수정할 수 있는 능력을 보여줍니다.

- **Performance Highlights**: 이번 연구를 통해, 부분적으로 학습된 CNN(Convolutional Neural Networks)이 '자기'와 '타자'의 특징을 완벽하게 구분할 수 있는 능력을 보여줍니다. 또한, 다섯 대화형 챗봇(ChatGPT4, Gemini, Perplexity, Claude, Copilot)에 대해 질문 기반의 거울 테스트를 수행하여, 챗봇들이 자신의 답변을 타 챗봇의 답변과 어떻게 인식하는지를 분석하였습니다. 이 연구는 AI가 진정한 자각을 갖기 위한 기준을 제안하며, AI의 윤리적 및 정책적 접근 방안을 모색합니다.



### On Sequential Fault-Intolerant Process Planning (https://arxiv.org/abs/2502.04998)
Comments:
          20 pages; 7 figures

- **What's New**: 본 논문에서는 Sequential Fault-Intolerant Process Planning (SFIPP)이라는 계획 문제를 제안하고 연구하였습니다. SFIPP는 모든 단계가 성공해야만 계획이 성공으로 간주되는 보상 구조를 포착하며, 이는 전통적인 보상 구조와 다릅니다. 이 모델은 약물 및 재료 발견, 보안, 품질-critical 제품 설계와 같은 중요한 응용분야에서 발생하는 보상 구조를 다루고 있습니다.

- **Technical Details**: SFIPP 과정은 m개의 단계로 구성되며, 각 단계에서 행동은 성공 확률 ps,i를 가지므로, 각 단계 s에서 선택된 행동의 성공 여부에 따라 보상이 주어집니다. 이 문제는 온라인 학습을 규명하며, 알고리즘의 성능은 최적 보상과 알고리즘의 총 보상 차이에 해당하는 'regret'를 통해 측정됩니다. 또한, 다중무장 도박(Multi-Armed Bandit) 알고리즘을 활용하여 탐색과 활용을 효과적으로 균형 잡는 방법을 설명합니다.

- **Performance Highlights**: 실험적으로 개발된 특화 알고리즘은 SFIPP 인스턴스의 구조에 대한 추가 정보를 활용하여 일반 알고리즘보다 뛰어난 성능을 보였습니다. 특히, 결정론적 성공/실패의 특별한 경우에서 성능이 최적 기대 손실을 달성하고, 무작위 발생의 일반 SFIPP 과정에서도 긴급 손실 경계를 가진 알고리즘을 설계하였습니다. 결과적으로 본 연구는 다양한 SFIPP 설정에서의 알고리즘 성능 평가를 통해 향후 연구 방향에 대한 통찰을 제공합니다.



### SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning (https://arxiv.org/abs/2502.04780)
- **What's New**: 최근 다중 에이전트 AI 시스템의 발전은 복잡한 작업을 해결하기 위한 새로운 최적화 방법인 SiriuS를 소개합니다. SiriuS는 자연어 모델(LLMs)을 활용하여 성공적인 추론 경로의 경험 라이브러리를 구축함으로써, 에이전트 간의 상호 작용을 최적화합니다. 이 시스템은 에이전트들이 서로의 출력을 검증하며 스스로 교정할 수 있는 메커니즘을 제공하여, 과거의 실패한 경로를 보완하는 과정까지 포함합니다.

- **Technical Details**: 다중 에이전트 시스템은 상태 공간(𝒮), 행동 공간(𝒜), 보상 함수(ℛ) 등으로 구성된 튜플 ⟨𝒮,𝒜,𝒯,ℛ,𝒩,𝒢⟩를 정의합니다. 각 에이전트는 정책(π)과 매개변수(θ)를 통해 행동을 결정하며, 에이전트 간의 통신 구조는 방향 그래프(𝒢)로 모형화됩니다. SiriuS는 다중 에이전트가 성공적으로 작업을 수행할 때, 그 상호 작용 경로에서 유용한 패턴을 식별하고 학습하는 데 중점을 둡니다.

- **Performance Highlights**: SiriuS는 다양한 도메인에서 다중 에이전트 성능을 크게 향상시켰습니다. 추론 및 생명과학 QA의 정확도가 2.86%에서 21.88%까지 증가하며 경쟁적인 시나리오에서 에이전트 간의 협상 능력도 강화되었습니다. 이 시스템은 에이전트들이 자신의 추론 및 협력 전략을 지속적으로 개선할 수 있는 확장 가능한 메커니즘을 제공합니다.



### Generating Symbolic World Models via Test-time Scaling of Large Language Models (https://arxiv.org/abs/2502.04728)
Comments:
          Technical Report v1 (32 pages, 6 figures)

- **What's New**: 이 연구는 복잡한 계획 문제 해결을 위한 Large Language Models (LLMs)의 활용을 다룹니다. 특히, PDDL(Planning Domain Definition Language)을 통해 자연어의 모호성을 극복하고 명확한 상태 설명을 가능하게 합니다. 제안된 방법은 특히 LLMs의 PDDL 추론 능력을 향상시켜 고품질의 PDDL 도메인을 생성하는데 효과적입니다.

- **Technical Details**: 이 연구는 명시적으로 PDDL 도메인을 생성하기 위해 LLMs의 생성 능력을 활용합니다. 이를 통해 상태 전이를 명확하게 정의하고, 상태 집합(𝒮) 및 행동 집합(𝒜)을 제시하여 전통적인 계획 알고리즘이 효율적으로 계획을 찾을 수 있도록 합니다. 연구에서는 Best-of-N 샘플링 기법을 도입해 초기 솔루션의 품질을 개선하고, 머신러닝을 활용하여 세밀하게 솔루션을 다듬는 과정을 설명합니다.

- **Performance Highlights**: 이 방법은 현재의 최첨단 방법에 비해 약 50% 이상의 성공률을 기록하며, 두 가지 작업에서 우수한 성과를 보였습니다. PDDL 도메인 생성을 통해 다양한 계획 작업에서 탁월한 능력을 발휘하며, 추가적인 훈련 없이도 경쟁 수준의 성과를 달성하는 데 성공했습니다.



### Bridging the Gap in XAI-Why Reliable Metrics Matter for Explainability and Complianc (https://arxiv.org/abs/2502.04695)
- **What's New**: 이 논문에서는 Explainable AI (XAI)의 평가에서 표준화된 신뢰할 수 있는 메트릭의 부족으로 인해 발생하는 주요 문제가 강조됩니다. 현재의 평가 방법은 단편적이고 주관적이며 편향되는 경우가 많으며, 복잡한 모델의 평가를 어렵게 만듭니다. 이 논문은 사용 사례와 인간의 판단에 기반한 신뢰할 수 있는 메트릭의 개발을 촉구합니다.

- **Technical Details**: XAI는 복잡한 머신러닝 모델과 최종 사용자 간의 격차를 해소하기 위해 개발되었습니다. 그러나 XAI 평가 방법이 일관성이 없고 주관적이라는 문제로 인해, AI 시스템의 신뢰성과 투명성을 확보하는 데 어려움이 존재합니다. 이러한 메트릭들의 개발은 주요 기술적 요소인 fidelity, robustness 및 usability를 측정하는 데 필수적입니다.

- **Performance Highlights**: 이 논문은 AI의 규제 준수를 위한 표준화된 평가 메트릭이 필요하다고 주장합니다. 유럽연합(EU) AI 법과 같은 규제 프레임워크는 투명성과 책임성을 강조하고 있으며, 이러한 요구에 부합하는 메트릭의 필요성을 일깨웁니다. 연구자, 산업 및 규제 기관 간 협력을 통해 XAI의 신뢰성과 의미 있는 해석을 보장할 수 있습니다.



### Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization (https://arxiv.org/abs/2502.04686)
- **What's New**: 이 논문에서는 Latent Space Policy Optimization (LSPO)이라는 새로운 반복적 프레임워크를 제안하여 자유형 언어 게임 내에서 전략적 언어 에이전트를 개발합니다. 이 프레임워크는 대화형 언어 환경에서 전략적 의사결정을 가능하게 하며, 기존 에이전트보다 더욱 효과적으로 성능을 향상시킵니다. 특히, LSPO를 통해 자유형 텍스트를 이산(Discrete) 잠재 공간으로 매핑하여 CFR(상대적 후회 최소화) 및 RL(강화 학습)의 적용 가능성을 높입니다.

- **Technical Details**: 연구는 Werewolf 게임을 테스트베드로 사용하여 LSPO의 작동 방식을 설명합니다. LSPO는 우선 자유형 발화를 관리 가능한 이산 표현으로 매핑하여, 그 후 CFR을 통해 근사 최적 전략을 발견합니다. 최적화된 정책은 자연어 대화로 다시 변환되어 LLM(대규모 언어 모델)을 Direct Preference Optimization(DPO)을 통해 미세 조정하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, LSPO 프레임워크를 이용한 에이전트는 반복을 거치면서 점진적으로 성능이 개선되는 것으로 나타났습니다. 특히, LSPO 에이전트는 기존의 최첨단 Werewolf 에이전트와 비교하여 가장 높은 승률을 보였습니다. 이러한 성과는 자유형 언어 의사결정에 대한 훌륭한 가능성을 보여줍니다.



### Scalable Oversight for Superhuman AI via Recursive Self-Critiquing (https://arxiv.org/abs/2502.04675)
- **What's New**: 이번 연구는 AI의 복잡한 작업에서 인간의 능력을 초월하는 현상이 강화됨에 따라 AI 감독 기술의 한계를 탐구합니다. 기존의 SFT(Supervised Fine-tuning)와 RLHF(Reinforcement Learning from Human Feedback) 기술이 인간의 피드백에 의존함에 따라 생기는 문제점을 지적하며, 비판의 비판(critique of critique)이 실제 비판보다 더 용이하다는 가설을 제시합니다. 또한, 이 연구는 높은 차원의 비판이 더욱 수월한 감독 경로가 될 수 있음을 주장합니다.

- **Technical Details**: 연구는 사람 간, 사람과 AI 간, AI 간 테스트를 통해 비판 및 비판의 비판을 평가하는 방법론을 개발했습니다. 이 방법론은 각 단계에서 충족해야 하는 기준을 설정하고, 복잡한 평가 작업을 단순화하는 방법으로 재귀적 메타 평가를 제시합니다. 다수 투표와 단순 투표의 두 가지 기준선을 통해 재귀적 비판의 효과를 공정하게 비교하고, 더 높은 수준의 비판이 더 효과적인 평가를 가능하게 함을 보여줍니다.

- **Performance Highlights**: 연구 결과는 인간 평가자들이 인간 출력물에 대해 비판하는 과정에서 효율적임을 확인하였으며, 이는 AI의 출력에도 성공적으로 적용될 수 있음을 나타냅니다. AI가 스스로 재귀적 비판을 수행하는 능력은 현재 시스템에서 한계가 있지만, 이는 향후 고급 AI 시스템에 대한 효과적인 감독의 가능성을 제시합니다. 이 연구는 AI 감독이라는 새로운 패러다임을 제안하며, AI가 자가 평가 시스템으로 스스로 비판할 수 있는 가능성을 탐구합니다.



### ${\rm P{\small ROOF}W{\small ALA}}$: Multilingual Proof Data Synthesis and Theorem-Proving (https://arxiv.org/abs/2502.04671)
- **What's New**: 본 논문은 멀티링구얼 증명 프레임워크인 ${\rm P{\small ROOF}W{\small ALA}}$를 소개합니다. 이는 Coq과 Lean과 같은 상용(Commercial) 인터랙티브 증명 도우미(ITP) 간의 표준화된 상호작용을 가능하게 하여, 언어 간 전이(transfer)의 기회를 제공합니다. 기존의 신경망 정리 증명 모델들은 특정 ITP에 제한되어 있었으나, 이 연구는 이러한 제약을 극복하고자 합니다.

- **Technical Details**: ${\rm P{\small ROOF}W{\small ALA}}$는 신경 정리 증명기와 두 개의 기존 ITP 간의 상호작용을 표준화합니다. 이 시스템은 멀티링구얼 증명 단계 데이터(multilingual proof step data)를 수집하여 신경 증명기를 훈련시키는 데 사용됩니다. 또한, 효율적인 병렬 증명 검색(algo) 알고리즘을 통해 다양한 ITP 및 문제 도메인에서 모델의 성능을 체계적으로 평가할 수 있게 합니다.

- **Performance Highlights**: ${\rm P{\small ROOF}W{\small ALA}}$를 통해 가능해진 멀티링구얼 훈련은 ITP 간의 성공적인 전이를 이끌어내는 것으로 나타났습니다. 특히, ${\rm P{\small ROOF}W{\small ALA}}$에서 생성된 Coq 및 Lean 데이터의 혼합으로 훈련된 모델이 Lean 전용 및 Coq 전용 모델보다 표준 prove-at-$k$ 지표에서 뛰어난 성능을 보였습니다. 이 연구에서는 모든 코드, 특히 ${\rm P{\small ROOF}W{\small ALA}}$ 프레임워크 및 멀티링구얼 ITP 상호작용 프레임워크 코드가 오픈 소스됨을 밝힙니다.



### Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research (https://arxiv.org/abs/2502.04644)
Comments:
          work in progress

- **What's New**: 이 논문에서는 Agentic Reasoning이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 LLM이 외부 도구를 사용하는 에이전트를 통합하여 추론 능력을 향상시킵니다. 중요한 점은 웹 검색, 코드 실행, 구조화된 기억을 활용하여 복잡한 문제를 해결할 수 있도록 돕는다는 것입니다.

- **Technical Details**: Agentic Reasoning은 복잡한 문제를 다루기 위해 여러 단계의 추론을 수행할 수 있는 능력을 제공합니다. 이 과정에서 모형은 웹 검색 에이전트와 코드 실행 에이전트를 사용하여 실시간 데이터 검색 및 연산 분석을 수행합니다. 또한, Mind Map 에이전트는 지식 그래프를 구성하여 복잡한 논리 관계를 정리할 수 있는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, Agentic Reasoning 프레임워크는 PhD 수준의 질문(GPQA)에서 우수한 정확성을 기록하며, 실제 전문가 수준의 연구 작업에서도 효과적인 성과를 보여줍니다. 이러한 결과는 이 프레임워크가 지식 집약적 분야에서 생산성을 향상시키는 데 큰 잠재력을 가지고 있음을 시사합니다.



### Preference Optimization via Contrastive Divergence: Your Reward Model is Secretly an NLL Estimator (https://arxiv.org/abs/2502.04567)
- **What's New**: 이 논문은 기존의 선호 최적화(Preference Optimization, PO) 접근 방식의 한계를 극복하기 위해 새로운 이론적 프레임워크를 제안합니다. 특히, 기존의 방법론들이 이론적 정당성을 결여하고 있음을 지적하며, PO 문제를 음의 로그 우도(Negative Log-Likelihood, NLL) 최소화 문제로 재정의합니다. 이와 더불어, 샘플링 전략을 사용하여 우도 정규화 상수를 추정하는 방안도 제시하고 있습니다.

- **Technical Details**: PO 문제를 NLL 최소화 문제로 변환하는 과정에서, 사용자는 대조 발산(Contrastive Divergence, CD) 방법론에 기반하여 샘플링 전략을 적용합니다. CD는 마르코프 체인 몬테 카를로(Markov Chain Monte Carlo, MCMC) 샘플링을 사용하여 우도 정규화 상수의 기울기를 근사합니다. 이 연구에서는 특히, 디스프리퍼드(completions that are less preferred) 완성을 선택하기 위해 MC-PO라는 알고리즘을 제안하고, 온라인 환경에서 효율적으로 작동하는 OnMC-PO 알고리즘으로 확장합니다.

- **Performance Highlights**: MC-PO와 OnMC-PO 알고리즘은 기존의 최첨단(SOTA) 기법들을 초월하여 태스크 성능을 획기적으로 향상시킵니다. 실험 결과, MC-PO는 유명한 정렬 벤치마크에서 더 높은 성능을 보여주며, OnMC-PO는 추가적인 개선을 이끌어냅니다. 본 연구에서 제안하는 다양한 샘플링 전략의 효과도 수치적으로 검증되었습니다.



### Unifying and Optimizing Data Values for Selection via Sequential-Decision-Making (https://arxiv.org/abs/2502.04554)
- **What's New**: 본 연구는 데이터 가치 평가(data valuation)의 이론적 기초와 데이터 선택(data selection) 문제로의 응용 가능성을 탐구합니다. 기존 방법들이 이론적으로 충분히 다루어지지 않았던 데이터 값(data values)의 사용을 재구성하여 순차적 의사결정 문제(sequential decision-making problem)로 접근했습니다. 이를 통해 기존의 Data Shapley와 같은 방법들을 동적 프로그래밍(dynamic programming) 관점에서 재조명합니다.

- **Technical Details**: 제안된 프레임워크는 데이터 선택 최적화를 위해 데이터 값이 가질 수 있는 이상적인 형태를 정립합니다. 데이터 선택에서의 효율성을 극대화하기 위해 각 선택 단계가 이전 선택에 기반하여 누적 보상(cumulative rewards)을 형성하도록 만듭니다. 또한, 데이터 가치 결정을 위한 최적화 문제를 동적 프로그래밍을 활용하여 해결하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 기존의 데이터 가치 평가 방법들이 최적 선택 전략과 비교했을 때 성능 격차가 크며, 제안된 근사치 접근 방식(approximation scheme)이 이러한 격차를 상당히 좁히는 것으로 나타났습니다. 또한, 효율적인 컴퓨테이션을 보장하면서도 이론적 약속을 유지할 수 있는 이점이 강조되었습니다.



### Robust Probabilistic Model Checking with Continuous Reward Domains (https://arxiv.org/abs/2502.04530)
Comments:
          Accepted by the 20th International Conference on Software Engineering for Adaptive and Self-Managing Systems 2025

- **What's New**: 이 논문에서는 연속 보상 분포와 이산 보상 분포를 효과적으로 처리하기 위한 새로운 방법론을 제안합니다. 기존의 기법들이 주로 이산 보상 분포에만 초점을 맞추었던 반면, 본 연구는 Moment Matching 기법을 통해 Erlang mixtures를 활용하여 보상 분포를 근사합니다. 이로써 PMC(Probabilistic Model Checking)의 강도를 향상시키고, 전체 보상 분포를 기반으로 한 품질 속성의 검증을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 Moment Generating Functions를 통해 높은 차수의 모멘트를 분석적으로 도출하며, 이를 바탕으로 통계적 성질을 유지한 채 보상 분포를 이론적으로 유계된 오차로 근사합니다. 이 방법은 이산 및 연속 보상 공간 모두에서 적용 가능하며, 복잡한 분포의 변화를 유연하게 캡처할 수 있도록 설계되었습니다. 또한, 이를 통해 다중 모드와 비대칭성 같은 특정 특징들을 효과적으로 다룰 수 있습니다.

- **Performance Highlights**: 실험적 평가 결과, 제안된 접근 방식이 PMC 문제에서의 정확도와 확장성을 동시에 개선함을 보여주었습니다. 기존의 히스토그램 기반 기법들이 직면한 수많은 도전 과제를 해결하며, 정확한 분포 근사를 제공함으로써 여러 시나리오에서 시스템의 행위 검증을 강화합니다. 이로 인해 시스템의 의사결정 로직이 보다 robust하게 작동할 수 있는 기반을 마련합니다.



### Safety is Essential for Responsible Open-Ended Systems (https://arxiv.org/abs/2502.04512)
Comments:
          12 pages

- **What's New**: 이번 논문은 Open-Ended AI가 과학 발견을 가속화하고 AI 에이전트의 지속적인 적응을 가능하게 하는 잠재력을 지니고 있음에도 불구하고, 이러한 시스템이 가진 예측 불가능성과 통제 불능이라는 고유의 위험성을 강조합니다. 기존 AI 시스템은 고정된 목표를 최적화하는 데 집중한 반면, Open-Ended AI는 새로운 해결책을 지속적으로 탐색하고 변화하는 환경에 적응할 수 있도록 설계되었습니다. 이는 미래의 인공지능 발전에 있어 중요한 전환점으로 작용할 것으로 기대됩니다.

- **Technical Details**: Open-Ended AI는 연속적으로 새로운 아티팩트를 생성하는 능력을 갖춘 시스템으로 정의됩니다. 이러한 시스템은 외부 관찰자가 학습 가능한 새로운 아티팩트를 만들어내며, 궁극적으로는 창의적인 해결책을 혁신적으로 개발할 수 있는 능력을 갖춥니다. 예를 들어, POET 알고리즘은 환경과 에이전트가 동시에 진화함으로써 Open-Ended 탐사를 가능하게 하며, Voyager 방법은 자동 커리큘럼을 통해 지속적 학습을 도모합니다.

- **Performance Highlights**: LLM(대형 언어 모델)의 발전으로 Open-Ended AI는 보다 다양한 응용 가능성을 열어가고 있습니다. 이러한 모델들은 인간의 데이터로 훈련되어, 인간에게 흥미롭고 바람직한 정보를 이해하고 이를 활용하는 능력을 갖추게 됩니다. 결과적으로 LLMs는 Open-Ended 과학적 발견, 새로운 환경 탐색, 그리고 진실한 대답을 이끌어내는 데에서 emergent behaviors(자연 발생적 행동)를 보여주고 있습니다.



### Agency Is Frame-Dependen (https://arxiv.org/abs/2502.04403)
- **What's New**: 이번 논문은 에이전시(Agency)의 개념을 강화학습(Reinforcement Learning)의 관점에서 재조명하고 있습니다. 에이전시는 시스템이 목표를 향해 결과를 조정하는 능력으로 정의되며, 종종 생물학, 철학, 인지 과학, 인공지능에서 중심적 주제로 다루어집니다. 저자들은 에이전시가 프레임 의존적(Frame-Dependent)이라고 주장하며, 모든 시스템의 에이전시는 특정 기준에 따라 결정되어야 한다고 설명합니다.

- **Technical Details**: 에이전시는 독립체로서의 경계, 스스로의 행동 출처, 환경과의 상호작용을 조절하는 목표, 그리고 이러한 목표에 따라 경험을 조정하는 능력 등 네 가지 주요 속성을 갖습니다. 그러나 이러한 속성들이 각각 어떻게 정의되는지는 선택한 참조 프레임에 따라 달라질 수 있습니다. 예를 들어, 각 시스템이 의미 있는 목표를 추구하고 있는지를 판단하는 기준은 주관적이며 변수의 선택에 따라 변화합니다.

- **Performance Highlights**: 저자들은 에이전시가 어떻게 다양한 상황에서 다르게 이해될 수 있는지를 논의합니다. 비유적으로, 벽이 무너지는 상황을 에이전시의 소멸로 설명하며, 실제로 행동의 출처가 벽이 아닌 다른 시스템에 있다고 강조합니다. 논문의 주요 결론은 에이전시는 프레임 의존적이며, 이는 향후 강화학습의 기초 과학에도 중대한 함의를 가진다는 점입니다.



### PerPO: Perceptual Preference Optimization via Discriminative Rewarding (https://arxiv.org/abs/2502.04371)
- **What's New**: 이번 논문에서는 Perceptual Preference Optimization (PerPO)라는 시각적 분별력 최적화 방법을 제안합니다. PerPO는 generative pre-trained multimodal large language models (MLLMs)의 시각적 인지 문제를 해결하는 것을 목표로 합니다. 이 방법은 다양한 부정 샘플을 수집하기 위해 구별되는 보상을 활용하고, 이를 통해 인간의 시각적 지각 과정과 MLLMs를 정렬하고자 합니다.

- **Technical Details**: PerPO는 명확한 목표 진실을 기반으로 다수의 가설을 생성하고, 점차적으로 최상의 가설로 좁혀지는 인간의 시각적 인지 과정을 모방합니다. 이를 위해 이 논문은 empirical risk minimization(ERM) 원리를 기반으로 하며, 부정 샘플을 효과적으로 획득하기 위한 확장 가능한 구별 보상을 도입합니다. PerPO는 또한 리스트 기반의 선호 최적화를 통해 부정 샘플 간의 관계를 학습하여 출력 품질을 향상시킵니다.

- **Performance Highlights**: PerPO는 MLLMs의 시각적 분별력 능력을 크게 향상시키며, 생성 능력을 유지합니다. 이 방법은 부정적 보상 해킹 문제를 완화하고, 다양한 시각 과제에서 일관된 성능을 보이는 것을 목표로 합니다. MLLMs의 미래 연구 방향을 새롭게 제시할 것으로 기대됩니다.



### MELON: Indirect Prompt Injection Defense via Masked Re-execution and Tool Comparison (https://arxiv.org/abs/2502.05174)
- **What's New**: 최근 연구에 따르면 LLM(대형 언어 모델) 에이전트는 간접 프롬프트 주입(Indirect Prompt Injection, IPI) 공격에 취약합니다. 이러한 공격은 악의적인 작업이 포함된 정보로 인해 에이전트가 승인되지 않은 행동을 하도록 유도합니다. 본 논문에서는 MELON(유형 감지와 재실행 기능을 강조하는 새로운 IPI 방어 기술)을 제안하여 이러한 공격으로부터 보호합니다.

- **Technical Details**: MELON은 공격이 성공하면 에이전트의 다음 동작이 사용자 작업보다 악의적인 작업에 더 의존하게 된다는 점을 활용합니다. 평가를 통해 MALON은 원본 실행과 마스크 실행 간의 동작을 비교하여 공격 여부를 판단합니다. 또한, false positive와 false negative를 줄이기 위한 세 가지 핵심 설계를 포함하고 있으며, 이는 도구 호출 비교의 정확성을 높입니다.

- **Performance Highlights**: MELON은 AgentDojo 벤치마크에서 광범위한 검증을 통해 기존의 SOTA 방어 방법보다 우수한 공격 예방 및 유틸리티 유지 성능을 보여주었습니다. MELON-Aug와 결합했을 때, 공격 성공률을 현저히 줄이고 사용자 유틸리티를 유지할 수 있는 시너지를 발휘합니다. 특히, MELON은 악의적인 도구 호출과 사용자 입력 간의 독립성을 활용하여 최상의 보안과 유틸리티 균형을 구현했습니다.



### Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficien (https://arxiv.org/abs/2502.05172)
- **What's New**: 이 연구에서는 Mixture of Experts (MoE) 모델과 밀집(dense) 모델의 공동 스케일링 법칙을 제시합니다. 이는 활성 파라미터 수, 데이터셋 크기, 전문가 수와 같은 중요한 요소를 포함하여 메모리 제약 하의 성능 분석을 제공합니다. MoE 모델이 밀집 모델보다 메모리 효율성이 높을 수 있다는 점을 발견했습니다. 이 연구는 MoE 모델의 최적 구성을 선택하기 위한 체계적인 프레임워크를 제공합니다.

- **Technical Details**: MoE 아키텍처는 게이팅 네트워크와 전문가 네트워크의 조합으로 제안되었습니다. 본 연구에서는 2.7B의 활성 파라미터와 5B의 총 파라미터를 가진 280개 이상의 실험을 통해 이론적 예측을 검증하였습니다. 이를 통해 MoE 모델의 최적 토큰-파라미터 비율과 전문가 수의 선택이 특정 계산 및 메모리 제약에 따라 달라짐을 보여줍니다. 또한, 학습 손실(Loss)과 데이터셋의 본질적 엔트로피 관계를 정의하여 제안된 법칙에 기반한 결론을 내렸습니다.

- **Performance Highlights**: MoE 모델은 동일한 계산 및 메모리 예산 하에서 실험을 통해 더 낮은 손실을 달성하여 실제적으로 더 높은 효율성을 입증했습니다. MoE 모델은 추론 시에도 더 높은 성능을 제공합니다. 기존의 밀집 모델보다 메모리 사용이 더 적고, 특정 하드웨어에서 메모리 제약을 받으면서도 더 나은 성능을 발휘합니다. 이러한 발견은 MoE 모델을 실제 대규모 훈련 시나리오에서 더욱 매력적인 선택으로 만듭니다.



### Transforming Science with Large Language Models: A Survey on AI-assisted Scientific Discovery, Experimentation, Content Generation, and Evaluation (https://arxiv.org/abs/2502.05151)
Comments:
          Work in progress. Will be updated soon

- **What's New**: 최근 다중 모달 언어 모델(multimodal language models)의 출현으로, 과학 분야는 AI 기반 기술 혁신의 문턱에 서게 되었습니다. 새로운 AI 모델과 도구들이 제안되며, 연구자와 학자들이 보다 효과적이고 효율적으로 연구를 수행할 수 있는 가능성을 제시하고 있습니다. 연구 사이클의 여러 측면, 예를 들어 관련 문헌 검색, 연구 아이디어 생성, 실험 수행, 텍스트 기반 및 다중 모달 콘텐츠 생성(예: 과학적 그림 및 도표), AI 기반 자동 피어 리뷰(automatic peer review)에 대한 내용을 다루고 있습니다.

- **Technical Details**: 이 설문조사는 위에서 언급한 다섯 가지 측면에 대해 포괄적으로 다루며, 관련 데이터셋(datasets), 방법(methods) 및 결과(results)뿐 아니라 평가(evaluation), 한계 및 향후 연구의 범위도 안내합니다. 특히, 이러한 도구의 단점과 오용 가능성(예: 가짜 과학, 표절, 연구 진실성에 대한 해악)과 같은 윤리적 문제들이 강조됩니다. 이는 연구 과정의 근본적인 변화를 가져올 것을 약속하는 새로운 발전에 대한 깊은 통찰을 제공합니다.

- **Performance Highlights**: 설문조사는 신기술의 잠재력과 그로 인한 연구 프로세스의 변화에 주목하며, AI4Science 분야에서의 새로운 AI 기반 이니셔티브를 촉진할 수 있는 기초 자료가 될 것으로 기대됩니다. 본 연구는 신규 연구자들에게 참조가 되는 자료가 되기를 바라며, AI를 활용한 연구의 효율성을 증가시키기 위한 다양한 방안이 모색될 것입니다.



### LP-DETR: Layer-wise Progressive Relations for Object Detection (https://arxiv.org/abs/2502.05147)
Comments:
          7 pages, 4 figures

- **What's New**: 본 논문에서는 LP-DETR(Layer-wise Progressive DETR)이라는 새로운 접근 방식을 제시합니다. 이 방법은 multi-scale relation modeling을 통해 DETR 기반 객체 탐지를 향상시킵니다. Relation-aware self-attention 메커니즘을 도입하여 객체 쿼리 간의 학습 가능한 공간적 관계를 생성하고, 디코더 레이어 전반에 걸쳐 다양한 관계 규모를 균형 있게 학습합니다.

- **Technical Details**: 이 연구에서 제안한 LP-DETR는 DETR 스타일의 탐지기 구조에 기초하고 있으며, ResNet-50 및 Swin-L 백본을 사용하여 COCO 2017 데이터셋으로 실험하였습니다. 모델은 객체 쿼리 간의 관계를 모델링하는 self-attention 메커니즘을 통해 지역적(local) 특성과 전역적(global) 관계를 효과적으로 캡처합니다. 우리의 프로그레시브 디자인은 모델이 디텍션 파이프라인 전반에 걸쳐 진화하는 공간적 종속성을 효과적으로 포착할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, LP-DETR은 12에폭에서 52.3% AP, 24에폭에서 52.5% AP를 달성하며, Swin-L 백본을 사용하였을 때는 58.0% AP로 더욱 향상되었습니다. 제안된 프로그레시브 관계 모델링은 수렴 속도와 탐지 정확도 모두를 개선하는 데 기여하는 것으로 나타났습니다. 이러한 결과는 레이어별 관계 모델링의 중요성을 뒷받침하고, 향후 객체 탐지 연구를 위한 유망한 방향성을 제공합니다.



### Latent Swap Joint Diffusion for Long-Form Audio Generation (https://arxiv.org/abs/2502.05130)
- **What's New**: 이 연구는 글로벌 뷰(Global View) 확산 또는 반복 생성에서 발생하는 높은 훈련 비용을 해결하기 위한 차세대 접근 방식인 Swap Forward (SaFa)를 제안합니다. SaFa는 프레임 수준의 잠재적 스왑(framework)을 활용해 여러 확산을 동기화하여 더욱 세밀한 스펙트럼을 포함한 일관성 있는 긴 오디오를 생성합니다.

- **Technical Details**: SaFa의 핵심 기술은 인접 뷰(view) 간의 양방향 Self-Loop Latent Swap을 적용하여 고주파(high-frequency) 성분을 적응적으로 강화합니다. 낮은 주파수 성분(low-frequency components)은 방해받지 않도록 유지되며, 또한 무방향 Reference-Guided Latent Swap이 초기 단계에서 참조와 비겹치는 지역 간에 적용되어 중앙 집중형 경로 지도(trajectory guidance)를 제공합니다.

- **Performance Highlights**: 정량적 및 정성적 실험을 통해 SaFa는 기존의 공동 확산(joint diffusion) 방법들을 크게 능가하며, 장기 오디오 생성 모델에서도 우수한 성능을 보여줍니다. 또한, SaFa는 팬오라마(panoramic) 생성에도 잘 적응하며 더 높은 효율성과 모델 일반화(model generalizability)를 달성하여 최첨단(state-of-the-art) 성능과 비슷한 결과를 보입니다.



### "It Felt Like I Was Left in the Dark": Exploring Information Needs and Design Opportunities for Family Caregivers of Older Adult Patients in Critical Care Settings (https://arxiv.org/abs/2502.05115)
- **What's New**: 이번 연구는 중환자실(ICU)에서 노인 환자의 가족 돌봄 제공자들이 의료 정보를 접근하고 해석하는 데 겪는 도전 과제를 탐구하는 데 중점을 두고 있습니다. 연구진은 11명의 돌봄 제공자와 인터뷰를 통해 정보를 얻는 과정에서의 주요 어려움을 규명하고, 이러한 문제를 해결하기 위한 AI 기반 시스템 프로토타입을 제안했습니다. 이 시스템은 환자의 주요 의료 사건을 시각화하는 타임라인 기능과 맥락에 맞는 정보 지원을 제공하는 LLM 기반 챗봇 기능을 포함하고 있습니다.

- **Technical Details**: 연구의 주요 목표는 ICU에서 노인 환자를 돌보는 가족 돌봄 제공자들의 정보 요구를 이해하고 이러한 요구를 지원하기 위한 AI 기능들을 설계하는 것입니다. 두 가지 연구 질문(RQ1, RQ2)을 중심으로 진행된 이 연구는 11명의 돌봄 제공자와의 반구조화 인터뷰를 통해 얻은 결과를 바탕으로, 정보 업데이트의 단편화, 복잡한 임상 보고서, 비효율적인 소통 문제 등 주요 도전 과제를 도출했습니다. 최종적으로 AI 기반 프로토타입은 환자의 의학적 사건을 지원하는 시각적 타임라인과 개인화된 정보 요청을 위한 LLM 기반 챗봇으로 구성되었습니다.

- **Performance Highlights**: 본 연구의 파일럿 평가에서는 노인 환자의 가족 돌봄 제공자 6명을 대상으로 프로토타입의 사용성을 평가하였으며, AI 기술을 활용해 복잡한 의학 정보를 이해하고 접근하는 데 있어 긍정적인 가능성을 확인했습니다. 연구 결과는 가족 돌봄 제공자의 인지 부담을 줄이고, 의료 제공팀과의 상호작용을 개선하는 방향으로 기여할 수 있는 방안들을 제시합니다. 이러한 조사 결과는 기술과 AI가 노인 환자의 급성 치료 상황에서 돌봄 제공자 지원에 미치는 영향을 심화시키고 있습니다.



### Flexible and Efficient Grammar-Constrained Decoding (https://arxiv.org/abs/2502.05111)
- **What's New**: 이번 연구에서는 문법 제약 디코딩 (Grammar-constrained decoding, GCD)의 새로운 접근 방식을 제안하며, 이를 통해 오프라인 전처리 속도가 기존보다 17.71배 빨라졌습니다. 특히, 우리 알고리즘은 현재 최고의 온라인 마스킹 효율성을 유지하면서 다양한 문법을 처리할 수 있도록 설계되었습니다. 제안하는 방법론은 LLM의 토큰 어휘와 문맥 자유 문법 (Context-Free Grammar, CFG) 단말의 결합 분석을 기반으로 합니다. 이를 통해 디코더가 효율적으로 유효한 LLM 토큰을 식별할 수 있습니다.

- **Technical Details**: 우리는 문법 제약이 있는 디코딩을 위한 알고리즘을 구축하기 위해, 먼저 정규 표현식을 사용하여 토큰을 정의하기 위해 렉서를 사용합니다. 이는 입력 문자열을 효율적으로 처리하고, 단어 수준에서 처리하는 대신 토큰 수준에서 문법 구조를 정의하는 것을 가능하게 합니다. GCD 알고리즘은 LLM의 서브워드 토큰과 단말 사이의 정렬 문제를 해결하며, 이로 인해 오프라인 전처리 비용을 줄이고 효율적인 온라인 토큰 마스킹을 가능하게 합니다. 새로운 도구 GreatGramma는 이러한 알고리즘을 구현하여 관련 GCD 접근법에 비해 속도를 월등히 향상시킵니다.

- **Performance Highlights**: GreatGramma는 기존 GCD 접근법 대비 평균 17.71배 빠른 오프라인 전처리 속도를 보여주며, 온라인 마스킹 효율성도 뛰어난 것으로 평가되었습니다. 연구에서는 기존 GCD 구현에 존재하는 신뢰성 오류도 발견되었으며, 이러한 문제들을 해결하여 단순한 모듈로 구성된 우아한 구현을 제공합니다. 이러한 성능 개선은 프로그램 합성 및 문법 프롬프팅과 같은 동적 문법 도메인에서 특히 유용하게 활용될 수 있습니다.



### ApplE: An Applied Ethics Ontology with Event Contex (https://arxiv.org/abs/2502.05110)
- **What's New**: 이 논문은 Applied Ethics(응용 윤리)의 개념을 명확히 이해하고, 이를 위한 새로운 온톨로지 시스템인 ApplE를 제안합니다. ApplE는 윤리 이론과 사건 맥락을 포착하여 행위의 도덕성을 전반적으로 설명하기 위한 목적으로 개발되었습니다. 이 시스템은 SAMOD(Simplified Agile Methodology for Ontology Development)를 기반으로 하여, 윤리적 결정을 돕기 위한 다양한 요소들을 통합하고 있습니다.

- **Technical Details**: ApplE 온톨로지는 윤리적 의사결정을 위한 주요 요소를 포함하고 있으며, 이는행위의 의도, 결과, 윤리 원칙 및 맥락에 따른 민감한 요인들로 구성됩니다. 이 연구에서는 각 단계별로 발생한 마일스톤을 기반으로 하여 체계적인 테스팅(three-fold testing)을 통해 온톨로지의 품질을 평가하고 있습니다. 또한, ApplE는 FAIR(Findable, Accessible, Interoperable, Reusable) 원칙을 따르며, 응용 윤리학자 및 온톨로지 엔지니어들에게 유용한 자원이 되고자 합니다.

- **Performance Highlights**: ApplE는 생명 윤리(bioethics) 분야에서의 사용 사례를 모델링하여 사회적 및 과학적 가치를 나타낼 수 있음을 입증하였습니다. 이 연구는 또한 기존 컴퓨터 시스템에서 윤리를 모델링할 때의 격차를 해결하기 위한 일관된 접근 방식을 제공합니다. 궁극적으로, ApplE는 윤리 이론과 사건 맥락을 포괄적으로 엮어줌으로써, 실제 상황에서의 윤리적 결정 과정을 지원하는 중요한 도구로 자리매김할 것을 목표로 합니다.



### Leveraging Hypernetworks and Learnable Kernels for Consumer Energy Forecasting Across Diverse Consumer Types (https://arxiv.org/abs/2502.05104)
- **What's New**: 이 논문에서는 다양한 소비자 유형에 적용할 수 있는 소비자 에너지 예측 모델인 HyperEnergy를 제안합니다. HyperEnergy는 복잡한 패턴을 모델링하기 위해 하이퍼네트워크(hypernetwork)를 활용하여 예측 정확도를 향상시킵니다. 특히, LSTM(Long Short Term Memory) 네트워크의 파라미터를 예측하는 데 하이퍼네트워크를 사용하고, 다항식 및 방사 기저 함수(RBF) 커널을 포함한 학습 가능한 적응형 커널(kernel)을 통합하였습니다.

- **Technical Details**: HyperEnergy는 LSTM을 기본 네트워크로 사용하며, 파라미터 통합 모듈을 통해 하이퍼네트워크와 연결됩니다. 이 하이퍼네트워크는 메타 네트워크로 설계되어 가중치와 바이어스를 예측합니다. 또한, 이 모델은 다양한 소비자 유형에 대해 작동할 수 있도록 커널을 고차원으로 변환하는 기능을 지니고 있으며, 각 커널의 기여도를 조정하는 학습 가능한 파라미터가 포함되어 있습니다.

- **Performance Highlights**: 제안된 HyperEnergy는 학생 기숙사, 단독 주택, 전기차 충전소가 있는 주택, 타운하우스 등 다양한 소비자 유형에서 10가지 다른 기법보다 우수한 성능을 보였습니다. 특히, 최첨단 기술인 LSTM, AttentionLSTM 및 transformer 모델보다 일관되게 더 뛰어난 결과를 발휘하며, 소비자 맞춤형 모델의 신뢰성을 높이고 있습니다.



### Learning Temporal Invariance in Android Malware Detectors (https://arxiv.org/abs/2502.05098)
- **What's New**: 이 논문은 Android 악성코드 탐지기를 향상시키기 위해 TIF라는 시간 불변 훈련 프레임워크를 도입합니다. TIF는 악성코드 변형과 새로운 가족의 출현으로 인한 분포 드리프트(distribution drift) 문제를 해결하는 데 중점을 둡니다. 이를 통해 모델은 고품질의 안정적인 표현을 학습하여 탐지기의 견고성을 향상시킵니다.

- **Technical Details**: TIF는 멀티 프로시(Proxy) 대조 학습(multi-proxy contrastive learning) 및 불변 기울기 정렬(invariant gradient alignment) 모듈을 통해 시간에 따라 안정적인 표현을 생성하고 정렬합니다. 이 프레임워크는 애플리케이션 관찰 날짜에 따라 환경을 조직하여 시간적 드리프트를 드러내며, 다양한 드리프트 요인에 효과적으로 대응합니다. TIF는 다른 학습 기반 탐지기에 원활하게 통합될 수 있는 구조로 설계되었습니다.

- **Performance Highlights**: 10년간의 데이터세트를 사용한 실험 결과, TIF는 초기 배포 단계에서 특히 우수한 성능을 보이며 기존의 최고 수준 방법들을 초월했습니다. 실험 결과는 TIF가 안정적인 표현을 학습하고 유지하는 데 있어 일관된 성능 개선을 보여주었음을 확인시켜줍니다. 이러한 개선은 Android 악성코드 탐지의 실제 요구를 해결하고 있습니다.



### Lost in Time: Clock and Calendar Understanding Challenges in Multimodal LLMs (https://arxiv.org/abs/2502.05092)
Comments:
          Preprint

- **What's New**: 이번 연구는 멀티모달 대형 언어 모델(MLLMs)이 아날로그 시계와 연간 달력을 통해 시간을 해석하는 능력을 조사하였습니다. 연구팀은 ClockQA와 CalendarQA라는 두 가지 하위 데이터 세트를 구성하여, MLLMs의 시각적 인식 및 수치적 추론 능력을 분석합니다. 기존 연구와는 달리, 이번 연구는 시간과 날짜 관련 문제 해결에 중점을 두고 있어 새로운 접근법을 제시합니다.

- **Technical Details**: ClockQA 데이터 세트는 다양한 유형의 아날로그 시계를 포함하며, 주어진 이미지에서 시간을 정확히 읽는 능력을 평가합니다. 한편, CalendarQA는 연간 달력을 기반으로 하여 날짜와 관련된 질문에 대한 MLLMs의 응답을 시험합니다. 이 연구는 MLLMs의 시간 인식 능력을 평가하기 위한 제한된 규모의 데이터를 고안했으며, 각 모델의 성능을 정밀하게 분석하는 데 주력하였습니다.

- **Performance Highlights**: 초기 평가 결과, Gemini-2.0이 ClockQA에서 가장 높은 성능을 보였으나, 전반적인 성능은 부실하였습니다. 반면, GPT-o1은 CalendarQA에서 80%의 정확도로 뛰어난 성과를 기록했습니다. 그러나 일반적으로 두 작업 모두에서 낮은 성과가 나타났으며, 이는 MLLMs가 여전히 시간과 날짜 해석에서 어려움을 겪고 있음을 보여줍니다.



### Mitigating Unintended Memorization with LoRA in Federated Learning for LLMs (https://arxiv.org/abs/2502.05087)
- **What's New**: 이 논문은 federated learning (FL)에서 발생할 수 있는 데이터 기억 문제를 해결하기 위해 low-rank adaptation (LoRA)라는 간단하면서도 효과적인 미세 조정 전략을 제시합니다. 기존의 FL 훈련된 대형 언어 모델이 훈련 데이터의 구문을 기억하는 문제를 보여주었고, LoRA가 이러한 기억 현상을 최대 10배까지 감소시킨다는 사실을 입증하였습니다. 이 연구는 의학적 질문에 대한 답변을 정확히 제공하는 모델에서 다양한 Llama 2 및 3 모델에 대해 실험을 수행하였습니다.

- **Technical Details**: LoRA는 대형 언어 모델의 미세 조정 시 필요한 계산량과 메모리 요구 사항을 줄이는 방법으로서 사용됩니다. 이 방법은 훈련 가능한 매개변수의 수를 대폭 줄여주며, 훈련 데이터의 조건부 구문 기억 현상을 완화하는 데 효과적입니다. 실험에서는 두 가지 설정인 federated와 centralized 환경 모두에서 LoRA의 효능을 확인하였으며, 여러 가지 데이터 민감성 문제를 다루었습니다.

- **Performance Highlights**: LoRA는 고성능 유지와 함께 기억 현상을 효과적으로 감소시키는 데 성공하였음을 여러 모델에 걸쳐 확인하였습니다. 또한, gradient clipping, Gaussian noising, secure aggregation, Goldfish loss 등과 같은 다른 프라이버시 보호 기술과 결합하여 기록 수준의 데이터 보호를 강화할 수 있는 가능성을 보여주었습니다. 이는 FL에서의 데이터 민감성 문제 해결에 있어 매우 중요한 기여로 보입니다.



### Causality can systematically address the monsters under the bench(marks) (https://arxiv.org/abs/2502.05085)
- **What's New**: 이번 논문에서는 인과성(causality) 프레임워크를 활용하여 대형 모델 평가의 신뢰성을 높이고자 하는 방안을 제시합니다. 기존의 머신러닝 평가가 성과(performance)만을 중시하는 경향이 강한 반면, 인과적 사고를 통해 모델의 행동 이면에 있는 추론 과정을 더 잘 이해할 수 있습니다. 또한, ‘Common Abstract Topologies (CATs)’라는 새로운 개념을 도입하여 인과 모델 디자인을 보다 용이하게 만듭니다. 이러한 접근법은 복잡한 모델 평가에서 여러 문제를 조명하고 해결하는 데 기여할 수 있습니다.

- **Technical Details**: 논문은 인과적 가정을 명시적으로 다룸으로써 보다 정밀한 가설을 수립하고, 모델의 한계를 진단하고 분석 도구를 활용하는 방법을 설명합니다. 특히, 대형 언어 모델(LLMs)의 추론 능력에 대한 평가를 인과 분석을 통해 진행하는 것이 효과적임을 강조합니다. 다양한 Reasoning Tasks에 대한 최근 연구를 검토하여, LLM의 인지적 실패들을 구체적으로 분석하고 이에 대한 인과적 모델링의 필요성을 제기합니다. 이 과정에서 모델, 데이터셋, 평가 절차로 인한 문제를 각각 구분하여 다룹니다.

- **Performance Highlights**: 이 논문에서 제시된 인과적 모델링 접근법은 대형 모델들의 성능 저하 원인을 규명하는 데 효과적일 것으로 보입니다. 정확도를 넘어서서 모델이 ‘올바른 이유로 올바른 대답’을 할 수 있도록 평가 기준을 고안했습니다. 기존 벤치마크가 모델의 진정한 추론 능력을 반영하지 못하는 경우가 많기에, 인과성 기반의 접근법이 이러한 문제를 해결하는 데 기여할 수 있을 것으로 기대됩니다. 연구가 진행됨에 따라, 더 나은 평가 프레임워크와 함께 대형 모델의 신뢰도를 높일 수 있는 방법론이 제시될 것입니다.



### ChallengeMe: An Adversarial Learning-enabled Text Summarization Framework (https://arxiv.org/abs/2502.05084)
- **What's New**: 이 논문은 ChallengeMe라는 새로운 적대적 학습 기반 프롬프트 프레임워크를 제안하여 텍스트 요약 과제에서의 성능을 향상시키고자 합니다. 이 프레임워크는 생성 프롬프트, 평가 프롬프트, 피드백 최적화라는 세 가지 연속된 해결책으로 구성되어 있습니다. 이 연구는 적대적 학습의 최적화 차원을 일곱 가지로 설정하고, 혼합 사례 연구를 통해 기존의 LLM들과 비교하여 더 정확하고 유창한 요약을 생성할 수 있음을 입증했습니다.

- **Technical Details**: ChallengeMe는 인간 인지 과정의 분류 및 비교 메커니즘에서 영감을 받아 설계된 적대적 프롬프트 학습 프레임워크입니다. 이 프레임워크는 입력 프롬프트, 적대적 프롬프트, 피드백 최적화 전략을 포함한 세 가지 모듈로 구성되어 있습니다. 또한 모델이 요구되는 목표와 제약 조건을 준수하도록 적절한 프롬프트를 설계하여 모델이 텍스트 요약 과정에서 효율을 극대화하도록 안내합니다.

- **Performance Highlights**: 제안된 프레임워크는 텍스트 요약 과제에서 그 자체의 품질, 유창함, 및 안정성을 토대로 현존하는 고급 LLM 솔루션들과 비교해 우수한 성능을 입증했습니다. 실험 결과, 30명의 참가자로부터 받은 주관적 평가에서도 긍정적인 결과를 얻어 향후 AI 모델 최적화 방향에 대한 잠재적 아이디어를 제시합니다. 이는 인간과 AI 사이의 상호 학습을 모사한 기계 간의 지속적인 발전을 이끌 수 있는 중요한 기초 자료가 될 것입니다.



### Preference-aware compensation policies for crowdsourced on-demand services (https://arxiv.org/abs/2502.05060)
- **What's New**: 이 연구는 Crowdsourced (크라우드소싱) 온디맨드 서비스의 보상 정책을 동적으로 설정하는 문제를 다룹니다. 연구진은 Gig Workers (긱 노동자)의 요청 선호를 고려한 보상 전략을 도입하여, 이를 통해 플랫폼의 수익성 또한 유지할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 Multinomial Logit model (다항 로짓 모델)을 활용하여 긱 노동자의 요청 수락 확률을 설명합니다. 그리고 이 모델을 바탕으로 사후 결정 상태(post-decision states)를 이용한 해를 도출하여 근사 동적 프로그래밍 알고리즘에 통합합니다. 이는 요구사항과 작업자의 도착이 확률적으로 발생하는 이산 시간 프레임워크(discrete-time framework) 내에서 이루어집니다.

- **Performance Highlights**: 개발된 알고리즘은 벤치마크 알고리즘과 비교하여 이질적인 긱 노동자 집단의 경우 9%, 동질적인 집단에서는 2.5-7.5%의 성능 향상을 보여주었습니다. 실제 데이터에 대한 실험에서도 위치 선호 시나리오에 따라 8%에서 20%의 개선 효과를 나타내는 등 안정적인 성능을 유지하고 있습니다.



### Differentiable Mobile Display Photometric Stereo (https://arxiv.org/abs/2502.05055)
Comments:
          9 pages

- **What's New**: 이번 논문에서는 Differentiable Mobile Display Photometric Stereo (DMDPS)를 제안합니다. 기존의 DDPS는 고정된 데스크톱 설정에 의존했지만, DMDPS는 스마트폰을 사용하여 더 실용적인 접근을 제공합니다. 이 시스템은 모바일 앱을 통해 패턴을 동시에 표시하고 고품질의 HDR 이미지를 캡처할 수 있습니다.

- **Technical Details**: DMDPS는 물리 기반의 조명 방식으로, 모바일 기기의 카메라와 디스플레이를 활용하여 다양한 조명 조건에서 장면을 캡처합니다. 이를 통해 현실 세계의 3D 프린트된 물체를 촬영하고, 차별화 학습 과정(differentiable learning process)을 통해 디스플레이 패턴을 학습합니다. 시스템은 HDR 이미지를 사용하여 표면 법선(surface normals)과 반사율(albedos)을 재구성하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: DMDPS의 유효성은 3D 프린트된 데이터셋과 떨어진 나뭇잎의 첫 번째 데이터셋을 통해 보여집니다. 나뭇잎 데이터셋은 재구성된 표면 법선과 반사율을 포함하여 컴퓨터 그래픽스 및 비전을 넘어 향후 연구를 위한 기초 자료를 제공합니다. DMDPS는 실용적인 물리 기반의 조명 방법으로 한 단계 발전했다고 믿습니다.



### Federated Learning for Anomaly Detection in Energy Consumption Data: Assessing the Vulnerability to Adversarial Attacks (https://arxiv.org/abs/2502.05041)
Comments:
          12th IEEE Conference on Technologies for Sustainability

- **What's New**: 이 논문은 에너지 데이터에서 연합 학습(Federated Learning, FL)을 기반으로 하는 이상 탐지가 적대적 공격에 얼마나 취약한지를 분석합니다. 기존의 이미지 도메인에서 연구된 적대적 공격은 시간 시계열 문제에는 잘 탐구되지 않았으며, FL 환경에서도 그 영향을 주목하지 않았습니다. 특히, LSTM(Long Short Term Memory)와 Transformer 모델을 사용하여 FL 설정에서의 이상 탐지 성능을 평가합니다.

- **Technical Details**: 논문은 FL의 구조와 작동 방식에 대해 설명하며, 이를 통해 데이터를 공유하지 않고도 글로벌 모델을 교육할 수 있는 장점을 강조합니다. 연구에서는 FL 설정에서 FGSM(Fast Gradient Sign Method) 및 PGD(Projected Gradient Descent) 등의 두 가지 백색 상자 공격 방법을 사용하여 모델의 취약성을 평가합니다. 결과적으로, PGD 공격이 FGSM 공격보다 더 민감하게 반응하며, 이는 주로 PGD의 반복적인 특성 때문임을 보여줍니다.

- **Performance Highlights**: 연구 결과는 FL 기반 이상 탐지가 적대적 공격에 취약하다는 것을 나타내며, 이는 에너지 소비 데이터를 포함한 다양한 모델에서 일관되게 발생합니다. PGD 공격이 모델의 정확도를 10% 이상 떨어뜨리는 결과를 가져오며, 중앙 집중식 학습 방식보다 FL의 이상 탐지가 더 큰 영향을 받는 것으로 나타났습니다. 이로 인해 FL 설정에서 방어 메커니즘의 필요성이 강조됩니다.



### Bridging Voting and Deliberation with Algorithms: Field Insights from vTaiwan and Kultur Kom (https://arxiv.org/abs/2502.05017)
Comments:
          Submitted to ACM Conference on Fairness, Accountability, and Transparency (FAccT) 2025

- **What's New**: 이 연구는 대규모 투표와 대면 논의를 통합하는 새로운 알고리즘 및 컴퓨팅 도구를 소개합니다. 제안된 방법은 실제 사례인 Kultur Komitee 2024 (KK24)와 vTaiwan에서 테스트되었으며, 이러한 사례 연구는 민주적 과정의 복잡성을 해결하는 데 어떻게 기여하는지를 보여줍니다. 특히, Radial Clustering, Human-in-the-loop MES 및 ReadTheRoom 논의 방법을 제시하여, 개인의 선호를 집단의 의사결정과 통합하는 방안을 모색합니다.

- **Technical Details**: 이 연구는 세 가지 핵심 기법을 소개합니다. 첫째, Radial Clustering 방법은 동질적 또는 이질적인 소그룹을 구성하여 균형잡힌 토론을 가능하게 합니다. 둘째, Human-in-the-loop MES는 참가자에게 실시간 피드백을 제공하여 MES 알고리즘과 논의 간의 조화를 도모합니다. 셋째, ReadTheRoom 방법은 의견 맵을 사용하여 의견의 일치와 불일치를 식별하고 논의 중 의견 변화 과정을 시각화하여 투명성을 높입니다.

- **Performance Highlights**: KK24는 시민들이 문화 자금을 배분하고 도시의 문화적 경관을 형성하는 데 참여하도록 하는 혁신적인 예시입니다. vTaiwan은 시민과 정부 간의 연결을 통해 국가 문제에 대한 논의에 기여하여 디지털 경제와 같은 주요 정책의 법률 제정에 영향을 미쳤습니다. 이러한 사례들은 현대 의사결정의 복잡성을 해결하기 위해 참여적 및 대면적 접근 방식을 성공적으로 통합한 모델을 보여줍니다.



### A New Paradigm in Tuning Learned Indexes: A Reinforcement Learning Enhanced Approach (https://arxiv.org/abs/2502.05001)
Comments:
          15 pages

- **What's New**: LITune은 Learned Index Structures (LIS)의 자동 조정(Auto Tuning)을 위한 새로운 프레임워크로 소개됩니다. 이 시스템은 Deep Reinforcement Learning (DRL) 기술을 활용하여 변동하는 데이터 분포와 작업 부하에 빠르게 적응할 수 있는 기능을 갖추고 있습니다. 기존 방법들과의 주요 차별성은 LITune이 안정성과 효율성을 보장하며, 실시간으로 최적의 솔루션을 찾을 수 있도록 한다는 것입니다.

- **Technical Details**: LITune은 Adaptive Training Pipeline을 기반으로 하며, 온-더-플라이(updating mechanism) 시스템인 O2 시스템을 추가하여 장기적인 데이터 동태를 수용할 수 있습니다. 이 시스템은 파라미터의 상호작용을 고려하여 복잡한 고차원 설정을 안정적으로 탐색할 수 있도록 돕습니다. 또한 Context-RL을 통한 위험 완화 전략을 적용하여 안전성을 강화하고, Tuner가 빠르게 변화하는 조건에서도 안정적으로 작동함을 보장합니다.

- **Performance Highlights**: 실험 결과, LITune은 선택된 Learned Index 인스턴스에 대해 98%의 런타임 감소 및 17배의 처리량 증가를 달성했습니다. 이러한 결과는 LITune의 조정 메커니즘이 실제 애플리케이션에서의 LIS 적용 가능성을 크게 높이며, 고성능 데이터 관리 시스템 구현에 기여할 수 있음을 나타냅니다.



### Robust Graph Learning Against Adversarial Evasion Attacks via Prior-Free Diffusion-Based Structure Purification (https://arxiv.org/abs/2502.05000)
Comments:
          Accepted for poster at WWW 2025

- **What's New**: 이 논문에서는 DiffSP라는 새로운 Diffusion-based Structure Purification 프레임워크를 제안합니다. 이 프레임워크는 그래프 확산 모델(graph diffusion model)을 활용하여 깨끗한 그래프의 내재적 분포를 학습하고, 이 정보를 바탕으로 적대적 노이즈를 제거하여 구조를 정제합니다. 기존의 연구들이 의존하던 prior 없이도 다양한 데이터셋과 공격에 강한 회복성을 자랑하는 방법론을 선보입니다.

- **Technical Details**: DiffSP는 전방 확산 프로세스(forward diffusion process)와 후방 denoising 프로세스(reverse denoising process)로 구성됩니다. 여기서 LID-driven 비등방(diffusion mechanism) 방법을 통해 의도적으로 노이즈를 주입하여 대응하면서, 기존의 정상적인 노드에 미치는 영향을 최소화하여 적대적 우려를 효과적으로 제거합니다. 후방 처리 동안엔 그래프 전이 엔트로피(graph transfer entropy)에 기반한 denoising 메커니즘이 적용되어, 생성된 그래프와 깨끗한 그래프 간의 의미적 정렬을 증가시킵니다.

- **Performance Highlights**: 아홉 개의 실제 데이터셋을 사용한 실험 결과, DiffSP는 아홉 가지 종류의 적대적 회피 공격에 대해 뛰어난 회복성을 보여주었습니다. 구조적 정보 손상 없이 정상 노드를 보존하면서 적대적 노이즈를 제거하는 데 성공하였습니다. 이러한 성능 향상은 전반적으로 그래프 학습의 신뢰성을 높이는 데 기여하며, 연구자들이 더욱 발전된 GNN 모델을 개발하는 데 중요한 기반이 될 것입니다.



### Aligning Black-box Language Models with Human Judgments (https://arxiv.org/abs/2502.04997)
Comments:
          Accepted for publication at NAACL 2025 (Findings)

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 추천 시스템 및 검색 엔진과 같은 평가 작업에서 인간 평가자의 대체로 사용될 수 있음을 강조합니다. 저자들은 LLM의 판단 결과가 인간 평가자와 일치하도록 맞추는 간단하지만 효과적인 방법을 제안하며, 재훈련 없이 LLM의 출력을 인간 판단과 정렬하는 방법을 기술합니다.

- **Technical Details**: 이 방법은 LLM의 출력을 인간 판단과 비교하여 선형 변환을 학습하는 과정을 포함합니다. 제안된 방식은 기존의 모델 로짓에 대한 접근 없이도 사용 가능하며, 적은 양의 데이터로 LLM의 성능을 크게 향상시킬 수 있습니다. 특히, 제안된 프레임워크는 블랙박스 LLM에 적용 가능하여 폭넓게 활용할 수 있습니다.

- **Performance Highlights**: 저자들은 29개의 평가 작업을 통해 LLM 판단의 일치성을 평균 142% 개선할 수 있었음을 보고합니다. 제안된 방법은 제로샷(zero-shot) 및 퓨샷(few-shot) 상황에서 효과적으로 작동하며, 실제로 몇 가지 결정 작업에서 인간 간의 일치성을 초과하는 성능을 보여줍니다. 또한, 작은 LLM도 큰 모델과 비교할 만한 성능을 발휘할 수 있도록 개선할 수 있음을 보여줍니다.



### Fast Adaptive Anti-Jamming Channel Access via Deep Q Learning and Coarse-Grained Spectrum Prediction (https://arxiv.org/abs/2502.04963)
- **What's New**: 이번 논문은 복잡하고 알려지지 않은 재밍 환경에서의 반재밍 채널 접근 문제를 조사합니다. 고정 패턴을 사용하는 전통적인 채널 호핑(hooping) 반재밍 접근 방식은 동적 재밍 공격에 대해 비효율적이라는 점을 강조합니다. 제안된 방법은 '재밍보다 빠르게 학습하기(learning faster than the jammer)'라는 직관에 기반하며, 깊이 Q 학습(deep Q learning) 모델의 보조 작업으로 동기화된 조잡한 스펙트럼 예측(coarse-grained spectrum prediction)을 사용합니다.

- **Technical Details**: 제안된 접근 방식은 DQN(deep Q network)과 조잡한 스펙트럼 예측을 결합하여 채널 접근 문제를 해결합니다. 이 모델은 기존의 DRL(deep reinforcement learning) 방식에 비해 더 빠른 수렴 속도를 보여주며, 약 70%까지 훈련 에피소드의 수를 줄이는 데 성공하였습니다. 또한, 조잡한 스펙트럼 예측을 효과적으로 사용함으로써 NE(Nash equilibrium) 전략 대비 10%의 처리량 향상을 달성하였습니다.

- **Performance Highlights**: 숫자적으로 재현된 결과는 제안된 접근 방식이 모델 훈련에서의 수렴 속도를 획기적으로 가속화했음을 보여줍니다. DRL 기반의 재밍에 대한 비슷한 성능을 발휘하면서도, 훈련에 필요한 에피소드를 크게 줄였습니다. 이는 동적인 채널 접근을 가능하게 하여 보다 신속하고 효과적인 반재밍 성능을 보장합니다.



### The Rising Threat to Emerging AI-Powered Search Engines (https://arxiv.org/abs/2502.04951)
- **What's New**: 이번 연구에서는 AI Powered Search Engines (AIPSEs)의 안전성 위험을 최초로 정량화하여, 기존 지식과 외부 데이터베이스를 통합하여 정확하고 효율적인 응답을 제공하는 데 등장한 새로운 문제점을 다루었습니다. AIPSEs가 악성 콘텐츠를 인용하거나 악성 웹사이트를 참조할 가능성이 있는 위험을 분석하고, 이를 통해 유해한 또는 검증되지 않은 정보가 확대될 수 있음을 지적했습니다.

- **Technical Details**: 연구의 핵심은 PhishTank, ThreatBook, LevelBlue에서 수집한 데이터를 바탕으로 AIPSEs가 처리하는 다양한 쿼리 유형에 대한 응답을 평가하였습니다. 특히, AIPSEs에 대한 위협 모델과 위험 수준을 체계적으로 정의하여, 양호한 쿼리에서도 악성 URL을 포함하는 유해한 콘텐츠를 자주 생성함을 발견했습니다. 또한, 직접 URL을 쿼리하는 경우에는 위험 수준이 증가하고, 자연어를 사용한 쿼리는 이러한 위험을 완화시킬 수 있다는 사실을 밝혔습니다.

- **Performance Highlights**: 이 연구는 온라인 문서 스푸핑과 피싱에 대한 사례 연구를 통해 AIPSEs를 속이기 쉬운 실세계의 설정을 보여주었습니다. 이를 해결하기 위해 GPT-4o 기반의 콘텐츠 정제 도구와 XGBoost 기반의 URL 탐지기를 사용하는 에이전트 기반 방어 체계를 개발하였습니다. 평가 결과, 이 방어 체계가 위험을 효과적으로 줄일 수 있지만, 제공되는 정보의 양은 줄어드는 측면이 있음을 명확히 하여 AIPSEs의 강력한 안전 조치 필요성을 강조했습니다.



### Data-driven Modality Fusion: An AI-enabled Framework for Large-Scale Sensor Network Managemen (https://arxiv.org/abs/2502.04937)
- **What's New**: 이 연구는 스마트시티 IoT 네트워크 관리의 효율성을 높이기 위해 'Data-driven Modality Fusion (DMF)'라는 새로운 감지 패러다임을 소개합니다. DMF는 다양한 감지 모달리티로부터의 시계열 데이터 간의 상관관계를 활용하여 필요한 물리적 센서의 수를 줄입니다. 이 접근 방식은 에너지 소비와 통신 대역폭을 줄이며, 전체 배포 비용을 최소화하는 데 기여합니다.

- **Technical Details**: DMF 프레임워크는 센서 네트워크의 계산 복잡성을 엣지 디바이스에서 코어로 이동시킵니다. 이를 통해 자원 제약이 있는 IoT 디바이스가 집약적인 처리 작업에 부담을 느끼지 않도록 합니다. Madrid에서의 실제 IoT 배포 데이터를 사용하여 DMF의 유효성이 검증되었으며, 교통, 환경 및 오염 메트릭을 정확하게 추정할 수 있음을 보여줍니다.

- **Performance Highlights**: DMF 접근 방법은 도시 IoT 네트워크 관리를 위한 확장성 높은 효율적인 메커니즘을 제시합니다. 이 시스템은 센서 고장 시 데이터 복원을 가능하게 하여 센서 네트워크의 강인성을 향상시킵니다. 또한, DMF는 비디오나 오디오 데이터에 의존하지 않아 프라이버시 문제도 완화합니다.



### Conformal Prediction for Electricity Price Forecasting in the Day-Ahead and Real-Time Balancing Mark (https://arxiv.org/abs/2502.04935)
- **What's New**: 이 연구에서는 재생 가능 에너지가 전기 시장에 통합될 때 발생하는 가격 안정성 문제를 해결하기 위해 확률적 가격 예측을 향상시키는 방법을 제안합니다. 특히, Conformal Prediction (CP) 기법을 활용하여 Ensemble Batch Prediction Intervals 및 Sequential Predictive Conformal Inference를 적용합니다. 이는 전통적인 모델에 비해 더 높은 정확성과 신뢰성을 가진 예측 구간을 제공합니다.

- **Technical Details**: 연구에서 제안하는 앙상블 접근법은 quantile regression 모델의 효율성과 시계열에 적합한 CP 기술의 강력한 커버리지 속성을 결합합니다. 이 앙상블 모델은 좁은 예측 구간을 제공하면서도 높은 커버리지 비율을 유지하여 더욱 신뢰할 수 있는 예측을 가능하게 합니다. CP 기법의 실제적 효과를 평가하기 위해 배터리 저장 시스템에 적용된 시뮬레이션 거래 알고리즘을 사용하여 검증합니다.

- **Performance Highlights**: 앙상블 접근법은 Day-Ahead 및 Balancing Markets에서 에너지 거래의 재정적 수익을 개선하는 결과를 보여줍니다. 시장 참가자들에게 실질적인 이점을 제공함으로써, 이 연구는 확률적 가격 예측의 중요성을 강조하고 있습니다.



### Cached Multi-Lora Composition for Multi-Concept Image Generation (https://arxiv.org/abs/2502.04923)
Comments:
          The Thirteenth International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA)의 이점을 활용하여 이미지 생성을 위한 보다 나은 방법을 제안합니다. 기존의 LoRA 사용에서 발생하는 'semantic conflicts' 문제를 해결하기 위해 Fourier frequency domain을 기반으로 한 새로운 접근 방식을 도입했습니다. 새로운 프레임워크인 Cached Multi-LoRA (CMLoRA)를 통해 LoRA 모듈을 효과적으로 결합할 수 있는 방법을 제시하며, 이로 인해 이미지 생성을 향상시킬 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: CMLoRA 프레임워크는 고주파 및 저주파 LoRA를 구분하여 각각의 주파수 응답에 따라 이미지를 생성하는데 기여할 수 있도록 설계되었습니다. 이 프레임워크는 특히 초기 denoising 단계에서 고주파 LoRA를 우선 사용하고, 이후 단계에서 저주파 LoRA를 적용합니다. 이러한 방법론은 LoRA의 통합 과정에서 발생할 수 있는 시맨틱 충돌을 최소화하고, 보다 일관성 있는 이미지를 생성하는 데 기여합니다.

- **Performance Highlights**: CMLoRA는 최신 LoRA 통합 방법들에 비해 평균 2.19%의 CLIPScore 개선과 11.25%의 MLLM 승률 개선이라는 우수한 성능 개선을 보였습니다. 실험을 통해 우리는 CMLoRA가 사용자에게 보다 향상된 이미지 품질을 제공한다는 것을 입증하였으며, 이는 LoRA 통합의 차별적인 잠재력을 보여줍니다.



### Complex Physics-Informed Neural Network (https://arxiv.org/abs/2502.04917)
Comments:
          16 pages, 9 figures

- **What's New**: 이번 논문에서는 Cauchy 적분 정리에 영감을 받은 학습 가능한 활성화 함수를 통합한 compleX-PINN이라는 새로운 물리 정보 신경망(PINN) 아키텍처를 제안합니다. compleX-PINN은 활성화 함수의 매개변수를 학습함으로써 단일 은닉층만으로도 높은 정확도를 달성합니다. 기존의 PINN들이 어려움을 겪었던 문제들을 효과적으로 해결하며, 정확도가 기존 방법보다 크게 향상된 것을 보여줍니다.

- **Technical Details**: PINN은 편미분 방정식(PDE) 문제를 해결하기 위해 신경망의 표현력을 활용합니다. 이 논문에서 소개하는 compleX-PINN은 Cauchy 적분 공식을 기반으로 하는 활성화 함수를 사용하여 훈련 효율성과 예측 정확성을 개선합니다. Cauchy 활성화 함수는 복잡한 PDE 솔루션을 근사하기 위해 필수적인 비선형성을 도입하며, 다차원 문제에 확장하여 적용됩니다.

- **Performance Highlights**: empirical 결과는 compleX-PINN이 여러 PINN 기반 모델보다 우수한 성능을 발휘함을 보여줍니다. 또한, compleX-PINN은 기존 PINN 훈련 기법과 호환이 가능하여, 이러한 방법들과 통합되었을 때 성능을 더욱 향상시킬 수 있습니다. 특히, 존재하는 PINN에서의 문제점을 해결하고 더 나은 예측 능력을 제공하는 데 매우 효과적임을 입증했습니다.



### Wavelet-Assisted Multi-Frequency Attention Network for Pansharpening (https://arxiv.org/abs/2502.04903)
Comments:
          12 pages, 13 figures

- **What's New**: 이번 논문에서는 Multi-Frequency Fusion Attention (MFFA)라는 혁신적인 방법을 제안합니다. 이 방법은 wavelet transforms를 활용하여 서로 다른 주파수 대역의 이미지를 손실 없이 복원하고, 주파수를 효과적으로 분리하여 정보 캡처를 개선합니다. 또한, 다양한 주파수 특징을 보존하는 데 중점을 두며, 여러 스케일에서 정보를 점진적으로 융합하는 웨이브렛 피라미드를 사용합니다.

- **Technical Details**: MFFA는 Frequency-Query, Spatial-Key, Fusion-Value를 생성하여 서로 다른 특징이 나타내는 물리적 의미를 기반으로 합니다. 이로 인해 주파수 도메인에서의 특정 정보 캡처가 더욱 효과적으로 이루어집니다. 또한, 논문에서는 주파수 특징이 서로 다른 작업에서 손실되지 않도록 하는 방법을 강화를 이룹니다.

- **Performance Highlights**: 여러 데이터 세트에서 수행된 정량적 및 정성적 실험 결과, 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보여주며 실제 환경에서도 상당한 일반화 능력을 발휘한 것으로 나타났습니다. 특히, 주파수 도메인 융합 과정에서 특징 손실이나 혼동을 방지하는 데 효과적인 성능을 자랑합니다.



### Unified Approaches in Self-Supervised Event Stream Modeling: Progress and Prospects (https://arxiv.org/abs/2502.04899)
- **What's New**: 이번 논문은 최근 디지털 상호작용의 증가로 생성된 대규모 이벤트 스트림(Event Stream, ES) 데이터의 효율적인 활용을 위한 Self-Supervised Learning (SSL) 방법론을 포괄적으로 검토합니다. 특히 산업별 연구가 단절된 상태에서 진행되었던 SSL 접근 방식을 통합하고, 효과적이고 확장 가능한 SSL 프레임워크의 개발 방향을 제시합니다.

- **Technical Details**: ES 데이터는 시간에 따라 생성된 이벤트의 연속적인 순서를 나타내며, 각 이벤트는 특정 행동이나 상태에 대한 타임스탬프 정보를 포함합니다. SSL은 비지도 학습을 통해 이러한 비포맷된 데이터에서 의미 있는 표현을 추출할 수 있도록 해줍니다. 주요 SSL 방법론은 예측(predicted) 기법과 대조(contrastive) 기법으로 나뉘며, 이 논문에서는 각 접근 방식을 구조화된 분류로 정립합니다.

- **Performance Highlights**: 논문의 리뷰를 통해 우리는 SSL 기반 이벤트 스트림 모델링에 대한 포괄적인 분석을 제공하며, 실제 응용 분야에서의 성능 향상 가능성을 강조합니다. 이 연구는 데이터 기반의 혁신을 가속화하고 재현성을 높이며 ES 모델링 커뮤니티에 유익한 새로운 주제를 부각시키기 위한 지침을 제공합니다.



### ARTInp: CBCT-to-CT Image Inpainting and Image Translation in Radiotherapy (https://arxiv.org/abs/2502.04898)
- **What's New**: 이번 논문에서는 Adaptive Radiation Therapy (ART) 프로세스에서 발생하는 CBCT (Cone Beam Computerized Tomography)의 한계점을 극복하기 위한 새로운 딥러닝 기반 프레임워크인 ARTInp를 제안하고 있습니다. ARTInp는 이미지 인페인팅(image inpainting)과 CBCT-CT 변환을 결합하여, 불완전한 CBCT 이미지의 신뢰성을 높이고자 하며, 특정 치료 전 검증이 용이한 고품질의 합성 CT(sCT) 이미지를 생성합니다.

- **Technical Details**: ARTInp 프레임워크는 두 개의 네트워크를 통해 구성됩니다. 첫 번째 네트워크인 completion network는 CBCT 볼륨 내의 해부학적 공백을 메우는 역할을 하며, 두 번째 네트워크는 Generative Adversarial Network (GAN)을 활용하여 고해상도의 합성 CT 이미지를 생성합니다. 이 연구는 SynthRad 2023 챌린지를 통해 수집된 CBCT 및 CT의 쌍을 학습 데이터로 사용했습니다.

- **Performance Highlights**: ARTInp의 테스트 결과는 18명의 환자를 대상으로 수행된 실험에서 우수한 성능을 보여주었으며, 이는 복잡한 방사선 치료 환경에서 CBCT 기반의 작업 흐름을 향상시킬 잠재력을 지니고 있음을 나타냅니다. 연구는 CBCT 이미지의 활용성과 환자의 해부학적 구조를 보다 정확하게 시각화할 수 있는 가능성을 입증하고, 향후 연구 및 임상 적용 가능성에 기여할 것으로 기대됩니다.



### Sparse Autoencoders Do Not Find Canonical Units of Analysis (https://arxiv.org/abs/2502.04878)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문은 Sparse Autoencoders (SAEs)가 고유하고 완전한 원자(feature) 세트를 찾기 위한 방법으로 널리 사용되지만, 그 결과 특정 공백을 갖고 있다는 의문을 제기합니다. 두 가지 새로운 기법, SAE 스티칭(SAE stitching)과 메타-SAE(meta-SAE)를 사용하여 SAEs가 이러한 기능을 찾는 데 있어 불완전함과 비원자성을 보여줍니다. 이는 SAEs가 새로운 정보를 포착 키 때문에 중요한 연구의 방향성을 제시합니다.

- **Technical Details**: SAE는 신경망의 활성화를 스파스(sparse) 선형 조합 형태로 분해하는 기법입니다. 이 논문에서는 SAE 스티칭을 통해 서로 다른 크기의 SAEs 간의 라텐트(latent)를 비교할 수 있는 방법을 제시하며, 메타-SAE를 통해 다른 SAE의 디코더 방향을 해석 가능한 메타 라텐트로 분해하는 접근법을 설명합니다. 이 과정에서 각 라텐트가 단순한 구성 요소가 아닌 복합적인 특징을 형성할 수 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과, 큰 SAE는 작은 SAE에서 발견된 라텐트의 보다 세분화된 버전과 새로운 기능을 학습할 수 있음을 보였습니다. 또한, 메타 라텐트를 활용한 분해 과정은 종종 해석 가능하며, 이는 라텐트가 원자적이지 않음을 시사합니다. 이러한 결과는 SAEs의 크기가 모든 기계적 해석 작업에 대한 공통의 단위를 형성하지는 않음을 나타내며, 최적의 사전 크기를 선택하는 것이 과제가 됨을 강조합니다.



### $TAR^2$: Temporal-Agent Reward Redistribution for Optimal Policy Preservation in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.04864)
Comments:
          23 pages, 5 figures, 4 tables

- **What's New**: 이번 논문에서는 시간-대리인 보상 재분배 방법인 TAR²를 제안합니다. TAR²는 희소한 전역 보상을 대리인별 및 시간 단계별로 분해하여 보다 빈번하고 정확한 피드백을 제공하여 정책 학습을 돕습니다. 이 방법은 기존의 보상 변수화 방법과 전략 제어 기술을 통합하여 협력적인 다중 에이전트 환경에서의 신뢰할 수 있는 성과를 이루어냅니다.

- **Technical Details**: TAR²는 에피소드 보상을 각 시간 단계와 각 에이전트에 따라 재분배하며, 이는 학습된 모델에 의해 가이드됩니다. 특히, TAR²는 잠재기반 보상 형태 변형을 활용하여 원래의 Markov 결정 프로세스(MDP)의 최적 정책을 보존하며, 신뢰할 수 있는 신호를 제공하는 방식으로 작동합니다. 또한, 듀얼 주의(attention) 메커니즘을 통하여 각 에이전트의 기여를 수치화하고 시간의 종속성을 포착하는 구조를 포함합니다.

- **Performance Highlights**: TAR²는 SMACLite와 Google Research Football과 같은 도전적인 벤치마크에서 강력한 기본선(baseline)인 AREL과 STAS를 초월하여 학습 속도와 최종 성과에서 우수한 성능을 보였습니다. 이 논문은 TAR²가 희소한 보상을 가진 다중 에이전트 시스템에서 대리인-시간 신뢰할 수 있는 기여 할당을 위한 실용적이고 체계적인 해법이 될 수 있음을 입증합니다.



### Lightweight Operations for Visual Speech Recognition (https://arxiv.org/abs/2502.04834)
Comments:
          10 pages (double column format), 7 figures

- **What's New**: 이번 연구에서는 비디오 데이터에서 음성을 인식하는 Visual Speech Recognition (VSR)의 경량화 아키텍처를 개발하여 리소스 제한이 있는 장치에서도 실행 가능하게 만드는 데 중점을 두었습니다. 기존 모델들이 요구하는 높은 계산 비용 문제를 해결하기 위해 Ghost 모듈을 활용하여 모델의 복잡성을 줄이고, 성능 손실을 최소화하면서도 강력한 인식 능력을 갖춘 모델을 설계하였습니다.

- **Technical Details**: 연구에서 제안된 아키텍처는 Ghost 모듈을 사용하여 전통적인 합성곱 연산을 대체하고, 필요한 파라미터 수와 계산 비용을 줄였습니다. 또한, Partial Temporal Block이라는 일반적인 시간 차단 구조를 설계하여 입력 볼륨을 두 부분으로 나누고 각 부분에 대해 별도의 연산을 적용하는 방식을 채택하였습니다. 이러한 접근은 저전력 응용프로그램에 적합한 초경량 템포럴 합성곱 네트워크를 개발하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 제안된 모델들은 대규모 공개 데이터셋에서 우수한 VSR 성능을 보여주었으며, 하드웨어 요구사항이 크게 감소하여 다양한 실용적 응용이 가능함을 입증하였습니다. 대규모 실험 분석을 통해 경량화된 아키텍처가 성능을 크게 저하시키지 않으면서도 리소스 요구 사항을 획기적으로 줄일 수 있음을 확인하였고, 이는 여러 종합적으로 다양한 계산 능력을 가진 장치에서의 활용을 용이하게 합니다.



### Optimistic Gradient Learning with Hessian Corrections for High-Dimensional Black-Box Optimization (https://arxiv.org/abs/2502.04829)
Comments:
          We develop a black-box optimization algorithm that learns gradients with neural models and can be applied to solve non-convex high dimensional real-world problems

- **What's New**: 이번 연구에서는 기존의 Explicit Gradient Learning (EGL) 방법 개선을 위해 두 가지 새로운 방법론인 Optimistic Gradient Learning (OGL)과 Higher-order Gradient Learning (HGL)을 도입합니다. OGL은 함수의 낮은 영역을 향한 편향을 컴포넌트로 포함하여 더 나은 결과를 제공합니다. 반면 HGL은 두 번째 오더 Taylor 보정을 적용하여 기울기 정확성을 향상시킵니다. 이러한 접근법을 통합하여 개발된 OHGL 알고리즘은 고차원 비선형 최적화 문제를 해결하기 위한 강력한 도구로 입증되었습니다.

- **Technical Details**: 이 논문에서는 높은 차원, 복잡하고 비선형적인 문제를 처리하기 위해 OGL과 HGL의 기법이 통합된 OHGL 알고리즘을 제안합니다. OGL은 유망한 솔루션을 향한 기울기 추정량에 가중치를 부여하여 성능을 향상시키고, HGL은 더 정확한 기울기 근사를 제공하기 위해 헤시안 수정(Hessian correction)을 통합합니다. 또한, 이들은 OHGL의 성능을 향상시키기 위해 예측 최적화 방향을 보다 정확하게 제시하도록 설계되었습니다.

- **Performance Highlights**: OHGL 알고리즘은 synthetic COCO 테스트 및 고차원 실제 ML 작업에서 최고 성능을 달성합니다. 이 알고리즘은 노이즈가 있는 환경에서도 일관된 성능을 보이며, 기울기 추정의 정확성을 바탕으로 빠른 수렴 속도를 자랑합니다. 실험 결과 해당 알고리즘이 높은 차원의 문제를 작은 예산으로 해결할 뿐만 아니라, 기울기 예측의 정확도를 높여 더 나은 후보를 생성하는 데 기여함을 보여주었습니다.



### MedMimic: Physician-Inspired Multimodal Fusion for Early Diagnosis of Fever of Unknown Origin (https://arxiv.org/abs/2502.04794)
- **What's New**: MedMimic은 고차원 데이터를 저차원으로 변환하는 다중 모드 진단 프레임워크로, 실제 의료 진단 프로세스에서 영감을 받았습니다. 이 시스템은 DINOv2, Vision Transformer, 및 ResNet-18과 같은 사전 훈련 모델을 사용하여 18F-FDG PET/CT 이미지를 의미 있는 특징으로 변환합니다. 이어서 학습 가능한 self-attention 기반의 융합 네트워크가 차별적 진단을 위해 이 특징과 임상 데이터를 통합합니다.

- **Technical Details**: MedMimic의 구조는 크게 세 가지 단계로 나눌 수 있습니다. 첫 번째 단계는 임상 데이터 준비로, 환자에 대해 표준화된 테스트 지표를 확보합니다. 두 번째는 사전 훈련 모델을 사용하여 CT 및 PET 스캔에서 다층 특징을 추출하는 단계로, 이 특징들을 통합하여 학습 가능한 self-attention 기반의 융합 네트워크에 전달합니다.

- **Performance Highlights**: 416명의 FUO 환자 데이터를 이용한 결과, MFCN(Multimodal Fusion Classification Network)은 0.8654에서 0.9291 사이의 macro-AUROC 점수를 얻어 기존의 기계 학습 방법 및 단일 모달 딥 러닝 기법을 초과했습니다. 이 연구는 다양한 데이터 소스를 효율적으로 통합하여 진단의 정확성을 높이는 데 초점을 맞추고 있으며, 현실 세계 데이터의 부족 문제를 해결하기 위해 비약적인 개선을 보여줍니다.



### S$^2$-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency (https://arxiv.org/abs/2502.04790)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문에서는 다중 에이전트 토론(Multi-agent Debate, MAD) 기법을 개선하기 위한 새로운 접근법, 선택적 희소 다중 에이전트 토론(Selective Sparse Multi-Agent Debate, S2-MAD)을 제안합니다. S2-MAD는 에이전트 간의 비효율적인 정보 교환을 줄여 토큰 비용을 최소화하는 동시에 성능 저하를 2.0% 이하로 유지합니다. 이러한 접근은 특히 복잡한 논리적 추론 및 수학적 문제 해결에 있어 LLM(다양한 언어 모델)의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: S2-MAD는 결정적 메커니즘(Decision-Making Mechanism)을 통해 에이전트가 토론에 참가할지를 선택적으로 결정합니다. 각 토론 라운드에서 에이전트는 비슷한 관점을 가진 응답이 아닌 새로운, 비중복 응답을 선택하여 포함하며, 그룹 내 토론 및 그룹 간 토론에 자율적으로 참여할 수 있는 옵션을 제공합니다. 이를 통해 중복 정보를 줄이고 토큰 비용을 절감하는데 기여하는 혁신적인 전략을 설계하였습니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시한 실험을 통해 S2-MAD는 기존의 MAD보다 토큰 비용을 최대 94.5%까지 절감하는 동시에, 개인 에이전트 추론(single-agent reasoning) 접근법과 비교해도 유사한 정확도를 유지할 수 있음을 입증하였습니다. 이러한 결과는 5개 작업을 통해 평가되었으며, S2-MAD는 기존의 방안들, 예를 들어 MAD-Sparse 및 GroupDebate와도 비교하여 우수한 성과를 나타냈습니다. 이로 인해 S2-MAD는 상용 및 오픈소스 모델 간의 혜택을 극대화하는데 기여할 수 있습니다.



### Enhancing SQL Injection Detection and Prevention Using Generative Models (https://arxiv.org/abs/2502.04786)
Comments:
          13 pages, 22 Figures, 1 Table

- **What's New**: 이 연구는 SQL Injection (SQLi) 탐지 및 예방 메커니즘을 향상시키기 위해 생성 모델(generative models)을 활용하는 혁신적인 접근 방식을 제시합니다. Variational Autoencoders (VAE)와 Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP), U-Net을 통합하여 합성 SQL 쿼리를 생성하고, 이를 통해 머신 러닝 모델 훈련 데이터셋을 증강합니다. 제안된 방법은 SQLi 탐지 시스템의 정확성을 개선하며, 기존 시스템의 오탐(false positives)과 누락(false negatives)을 모두 줄이는 데 성공했습니다.

- **Technical Details**: 이 연구는 전통적인 SQLi 탐지 방식의 한계를 극복하기 위해 합성 데이터 생성(synthetic data generation)과 고급 딥러닝 모델을 결합하여 SQLi 탐지를 동적으로 진행합니다. VAE, U-Net, CWGAN-GP를 활용하여 다양한 합성 SQL 데이터를 생성하고, 이를 통해 훈련 세트를 다양화하여 SQL Injection Attack (SQLIA)에 대한 일반화 능력을 높입니다. 또한, 이 연구는 SQL 쿼리를 전처리(preprocessing)하고 임베딩(embedding)하여 정확도, 정밀도(precision), 재현율(recall) 및 F1-score와 같은 핵심 성능 지표를 최적화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 광범위한 경험적 테스트 결과, 제안된 시스템이 진화하는 SQLi 공격 패턴에 적응할 수 있는 능력을 보여주었으며, 전반적인 정확도와 견고성을 향상시켰습니다. 이 연구의 접근 방식은 기존의 SQLi 탐지 시스템에 비해 성능 개선을 달성하며, 공격 패턴의 복잡성에 효과적으로 대응합니다. 결과적으로, 이 연구는 SQLi 탐지의 진화하는 필요성을 충족시키고, 실용적인 응용 프로그램에서의 지속적인 적용 가능성을 높입니다.



### Behavior-Regularized Diffusion Policy Optimization for Offline Reinforcement Learning (https://arxiv.org/abs/2502.04778)
Comments:
          Under review

- **What's New**: 본 논문에서는 확산 모델(diffusion models)에 특화된 행동 정규화(behavior regularization) 오프라인 강화 학습(offline reinforcement learning) 프레임워크인 BDPO(Behavior-regularized Diffusion Policy Optimization)를 소개합니다. BDPO는 전통적인 정책을 사용한 기존 연구의 한계를 극복하여, 강화학습의 정책을 보다 향상되고 강건하게 만드는 방법을 제시합니다. 이를 통해 행동 정책에 더 근접하도록 정책을 제어하며, 오프라인 설정에서 행동의 신뢰성을 높이고, 모델이 생성하는 행동의 품질을 향상시킵니다.

- **Technical Details**: BDPO는 Kullback-Leibler(KL) 정규화를 역시간 전이 커널(reverse-time transition kernels)에서 계산하여 동작합니다. 이 방법은 기존의 정책의 불확실성을 줄이며, 두 개의 시간 척도를 사용하는 액터-크리틱(actor-critic) RL 알고리즘을 개발하여, 정책이 행동 제약을 준수하면서 최적의 정책을 생성할 수 있도록 합니다. 논문에서는 확산 경로를 따라 중간 단계에서의 값 추정(value estimation)을 통해 효율적인 계산을 가능하게 하는 접근법을 제시하고 있습니다.

- **Performance Highlights**: BDPO는 D4RL 벤치마크에 포함된 지속적 제어 연산을 위한 2D 데이터셋에서 실행된 실험에서, 대상 분포(target distribution)를 효과적으로 근사함을 보여주었습니다. 이 방법은 기존의 오프라인 RL 알고리즘에 비해 우수한 성능을 나타내며, 실험 결과를 통해 BDPO의 효과성을 입증하였습니다. 연구는 BDPO가 정밀한 최적화를 통해 행동 정책의 공간을 포괄하도록 지원함을 강조합니다.



### DMPA: Model Poisoning Attacks on Decentralized Federated Learning for Model Differences (https://arxiv.org/abs/2502.04771)
Comments:
          8 pages, 3 figures

- **What's New**: 본 논문은 Decentralized Federated Learning (DFL)에서 'Decentralized Model Poisoning Attack (DMPA)'라는 새로운 공격 방법을 제안합니다. DMPA는 여러 악의적인 클라이언트 모델의 차별적 특성을 계산하고, 가장 효과적인 오염 전략을 획득하여 다수의 참여자에 의한 공모 공격을 수행합니다. 이는 기존 중앙집중식 Federated Learning (CFL) 모델에 대한 연구가 대부분이었던 것에 비해 DFL에 대한 새로운 연구 공백을 메우는 중요한 기여입니다.

- **Technical Details**: DMPA는 compromised benign parameters의 고유값을 결정하여, 해당 고유벡터를 추출하고 각 모델 간의 차이를 이용해 각도 편향 벡터를 계산합니다. 이 공격 기법을 통해 각 참여자의 부정적인 매개변수를 조정하여 효과적인 공격 모델을 생성합니다. DMPA의 성능은 MNIST, Fashion-MNIST, CIFAR-10과 같은 다양한 벤치마크 데이터셋과 연결 토폴로지에 걸쳐 평가되었습니다.

- **Performance Highlights**: 실험 결과, DMPA는 기존의 FL 모델 오염 공격 전략들보다 항상 우수한 성능을 보였습니다. DMPA는 공격 능력이 더 강력하고, 더 넓은 전파 가능성을 보여 DFL 시스템의 강건성을 효과적으로 손상시키는 것으로 나타났습니다. 이러한 발견은 DFL에서의 보안 문제를 심각하게 고려할 필요성을 강조합니다.



### Graph Federated Learning Based Proactive Content Caching in Edge Computing (https://arxiv.org/abs/2502.04760)
- **What's New**: 본 연구는 모바일 데이터 트래픽의 급증과 비디오 스트리밍의 확산으로 인해 프로액티브 콘텐츠 캐싱의 중요성을 강조합니다. 기존의 캐싱 전략들은 미래 콘텐츠의 인기도를 정확히 예측하지 못하고, 개인 정보 보호 문제를 초래할 수 있는 데이터 업로드 방식의 한계를 가지고 있습니다. 이에 대한 해결책으로 제안된 Graph Federated Learning 기반 프로액티브 콘텐츠 캐싱(GFPCC) 방식은 사용자 개인 정보 보호를 유지하면서 캐싱 효율성을 향상시킵니다.

- **Technical Details**: GFPCC는 페더레이티드 러닝과 그래프 신경망(Graph Neural Networks)을 통합하여 사용자와 아이템 간의 관계를 파악하고 콘텐츠 인기도를 예측하도록 설계되었습니다. 사용자는 Light GCN을 로컬에서 훈련시키고, 훈련된 모델의 파라미터만을 중앙 서버에 전송하여 업데이트를 수행합니다. 이러한 처리 과정은 사용자 데이터의 외부 유출을 최소화하며, 서버는 페더레이티드 평균화 알고리즘을 통해 모델을 조정하고 흔한 파일을 선택하여 캐싱합니다.

- **Performance Highlights**: 실험 결과, GFPCC는 MovieLens와 같은 실제 데이터셋에서 기존의 기본 캐싱 알고리즘에 비해 캐시 효율성이 높고 더 정확한 콘텐츠 인기도 예측을 통해 우수한 성과를 입증했습니다. 특히, Random, FPCC, m-ε-Greedy, Thompson Sampling과 같은 벤치마크 알고리즘과 비교하여 우수한 성능을 보였습니다. 그러나 큰 규모의 네트워크에서는 동적인 사용자 선호도 문제로 인해 여전히 확장성 측면에서 도전과제가 남아있습니다.



### Enhancing Phishing Email Identification with Large Language Models (https://arxiv.org/abs/2502.04759)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 사이버 범죄자들이 지속적으로 사용하는 피싱 (Phishing) 공격의 탐지와 예방을 위해 대규모 언어 모델 (Large Language Models, LLM)의 효능을 살펴보았다. 머신러닝 (Machine Learning, ML) 알고리즘을 이용한 기존의 다양한 솔루션과 함께 LLM의 도입이 특히 유망하다고 주장한다.

- **Technical Details**: 연구에서는 피싱 이메일 탐지를 위한 다양한 방법들 중에서 LLM의 성능을 실험적으로 분석하였다. 대규모 언어 모델을 사용하여 높은 정확도 (Accuracy)와 정밀도 (Precision)를 달성하였으며, 모델이 내리는 결정에 대한 해석 가능한 증거 (Interpretable Evidence)도 제공한다고 보고하고 있다.

- **Performance Highlights**: 실험 결과, LLM은 피싱 이메일 탐지에서 뛰어난 성과를 보여주었으며, 이는 사이버 보안 분야에서 혁신적인 접근 방식으로 평가받고 있다. 특히 모델의 의사 결정 과정이 명확하게 설명됨에 따라, 사용자가 결과를 이해하는 데 도움을 줄 수 있을 것으로 기대된다.



### Concept Navigation and Classification via Open Source Large Language Model Processing (https://arxiv.org/abs/2502.04756)
Comments:
          35 pages, 1 figure, 7 tabels

- **What's New**: 이 논문은 Open-Source Large Language Models (LLMs)를 활용하여 텍스트 데이터에서 잠재적인 구성 요소(constructs)인 frames, narratives 및 topics를 감지하고 분류하기 위한 혁신적인 방법론적 프레임워크를 제안합니다. 자동 요약과 인간 검증을 결합한 하이브리드 접근 방식은 구성 요소 식별의 정확성과 해석 가능성을 향상시키기 위해 설계되었습니다. 반복 샘플링과 전문례의 개선 과정을 통해 이 프레임워크는 방법론적 강인성을 보장하며 개념적 정확성을 유지합니다.

- **Technical Details**: LLMs는 자연어 처리(NLP)와 계산 언어학의 지형을 변화시켰습니다. 이러한 모델들은 대량의 데이터 세트에서 훈련받아 인간과 유사한 텍스트를 이해하고 생성하는 뛰어난 능력을 보여줍니다. 하이브리드 접근 방식은 LLM의 자동 텍스트 분석과 전문가 검증을 통합하여 효율성과 개념적 신뢰성을 균형 있게 유지합니다.

- **Performance Highlights**: 이 프레임워크는 AI 정책 논쟁, 암호화에 대한 신문 기사 및 20 Newsgroups 데이터 세트를 포함한 다양한 데이터 세트에 적용되어 복잡한 정치 담론과 미디어 프레이밍, 주제 분류 작업을 체계적으로 분석할 수 있는 다재다능성을 보여줍니다. 연구자들은 classical 모델의 한계를 넘어 대규모 텍스트 데이터의 복잡성을 더 포괄적으로 포착할 수 있게 됩니다.



### Every Software as an Agent: Blueprint and Case Study (https://arxiv.org/abs/2502.04747)
- **What's New**: 본 논문은 (multimodal) large language models (LLMs)을 활용하여 소프트웨어 에이전트를 개선하는 새로운 접근 방식을 제시합니다. 기존의 API 기반 및 GUI 기반 접근 방식을 넘어, LLM이 소프트웨어의 내부 구조인 소스 코드와 실행 환경에 접근하도록 허용하는 방안을 제안하고 있습니다. 이러한 화이트박스(whitebox) 환경에서 LLM은 소프트웨어 컨텍스트를 보다 효율적으로 활용하고, 사용자의 자연어 지시를 코드로 변환할 수 있습니다.

- **Technical Details**: 제안된 JiT-Codegen(just-in-time code generation) 접근 방식은 LLM이 실시간으로 코드를 생성하여 소프트웨어 내에서 실행할 수 있게 합니다. 이는 JIT(Just-In-Time) 컴파일과 유사한 맥락에서, LLM이 사용자 지시를 취합하여 실행 가능한 코드를 만들어내는 기능을 포함합니다. 이 과정에서 코드 에이전트(Code Agent)와 실행 샌드박스(Execution Sandbox) 간의 쌍방향 피드백 루프가 존재하며, 이를 통해 코드의 품질을 지속적으로 개선하게 됩니다.

- **Performance Highlights**: JiT-Codegen 방식은 GUI 상에서 다섯 번의 상호작용을 두 줄의 코드로 간소화할 수 있는 사례를 통해 증명되었습니다. 기존의 방법들에 비해, LLM이 소프트웨어의 내부 동작에 직접 접근하여 실행할 수 있어 성능과 효율성이 크게 개선될 것으로 기대됩니다. 이 새로운 패러다임은 소프트웨어 에이전트 설계에 근본적인 혁신을 일으킬 잠재력을 가지고 있으며, 사용자 요구에 맞춰 더 지능적이고 동적인 디지털 환경을 만들어갈 것입니다.



### Can Diffusion Models Learn Hidden Inter-Feature Rules Behind Images? (https://arxiv.org/abs/2502.04725)
Comments:
          25 pages, 18 figures, 3 tables

- **What's New**: 이 연구는 디퓨전 모델(Diffusion Models, DMs)의 이미지 특징 간 숨겨진 규칙 학습 능력에 대한 한계를 탐구합니다. 기존 연구에서는 주로 독립된 특징에 초점을 맞추었으나, 이 논문은 종속적인 특징 쌍의 관계를 분석하여 DMs가 이러한 관계를 정확하게 캡쳐할 수 있는지 여부를 평가합니다. 이 논문은 DMs가 정밀한 규칙을 학습하는 데 어려움을 겪고 있음을 밝히며, 모델의 성능을 향상시키기 위한 추가적인 방법론을 제안합니다.

- **Technical Details**: 연구는 이미지 데이터에서 두 개의 종속적인 특징 쌍(예: 태양의 높이와 그림자의 길이) 간의 규칙을 학습하는 능력을 실험적으로 평가합니다. 특히, DMs가 합동 분포를 추정할 때의 오류가 규칙 학습에 미치는 영향을 분석하며, 이러한 이유로 DMs는 세부 규칙을 효과적으로 학습하지 못하는 경향이 있음을 보입니다. 이 연구는 추가적인 분류기 가이드를 활용하여 DMs의 규칙 준수 샘플 생성을 개선하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과 DMs는 일반적으로 거칠고 간단한 규칙을 잘 학습하지만, 세밀한 규칙에서는 한계를 드러냅니다. 또한, 디퓨전 모델이 재현할 수 없는 구체적인 규칙의 존재는 이 모델들이 관심 있게 다루지 않던 부분에서 오는 것입니다. 이 연구는 클래식 분류기 훈련의 어려움과 함께, 세밀한 규칙을 식별하는 데 필요한 신호가 약한 문제를 강조합니다.



### EigenLoRAx: Recycling Adapters to Find Principal Subspaces for Resource-Efficient Adaptation and Inferenc (https://arxiv.org/abs/2502.04700)
- **What's New**: 최근 대규모 모델의 성장이 환경 영향과 접근성 문제를 야기하고 있습니다. 본 연구에서는 Low-Rank Adapters(LoRA)의 경량화된 finetuning 기법인 EigenLoRAx를 소개합니다. EigenLoRAx는 기존의 어댑터를 재활용하여 새로운 태스크에 적응하는 데 필요한 경량 파라미터만 학습함으로써 전반적인 효율성을 높입니다.

- **Technical Details**: EigenLoRAx는 pretrained adapters의 가중치를 주성분으로 분해하여 정보가 풍부한 소공간을 형성합니다. 이 기법은 총 학습 파라미터 수를 최대 100배 감소시키고, 최적화 속도를 최대 2배 가속화합니다. 특히 자원이 제한된 환경에서도 성능을 유지하면서 메모리 효율성을 개선합니다.

- **Performance Highlights**: EigenLoRAx는 다양한 비전 및 언어 태스크에서 각기 다른 도메인에 대해 뛰어난 성능을 발휘하며, 수백 개의 미활용 프리트레인 어댑터를 재활용할 수 있습니다. 또한, zero-shot 및 제한된 자원 상황에서도 성능을 유지하고, 여러 태스크에 효율적으로 대응할 수 있는 능력을 보여줍니다.



### ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning (https://arxiv.org/abs/2502.04689)
Comments:
          20 pages

- **What's New**: 이 논문에서는 Zero-shot Chain-of-Thought (CoT) 프롬프팅의 한계를 극복하고 LLM(대규모 언어 모델)의 질문 답변(QA) 성능을 향상시키기 위해 ARR을 제안합니다. ARR은 질문의 의도 분석, 관련 정보 검색 및 단계별 추론이라는 세 가지 핵심 단계를 포함하여 보다 구체적으로 지침을 제공합니다. 실험을 통해 ARR이 기본 방법(Baseline)과 CoT를 모두 초월하여 일관되게 성능을 개선하는 것을 보여줍니다.

- **Technical Details**: ARR은 질문의 의도를 분석하고, 관련 정보를 검색하며, 체계적인 추론을 실행하는 구조화된 접근 방식을 기반으로 합니다. 특히 ARR은 ‘질문의 의도를 분석하고 관련 정보를 찾아 단계별로 질문에 답하자’라는 명령어를 포함하여 질문 답변 과정에서 LLM의 성능을 극대화합니다. 이를 통해 LLM은 도전적인 QA 작업에서의 성능을 개선하게 됩니다.

- **Performance Highlights**: 실험 결과, ARR은 10개 다중 선택 질문 답변 데이터셋에서 기본 방법 및 CoT 방법보다 일관되게 더 나은 성능을 보여줍니다. 특히, ARR의 세 가지 구성 요소(분석, 검색, 추론)가 개별적으로도 성능을 개선하며, 분석만 포함하는 설정에서 가장 큰 개선 효과를 보입니다. 이 연구는 ARR이 다양한 모델 크기와 설정에서 효과성과 범용성을 입증하였음을 강조합니다.



### M-IFEval: Multilingual Instruction-Following Evaluation (https://arxiv.org/abs/2502.04688)
- **What's New**: 이번 논문에서는 다국어 지원을 위한 새로운 기준인 Multilingual Instruction Following Evaluation (M-IFEval)을 제안합니다. 기존의 Instruction Following Evaluation (IFEval) 기준은 오로지 영어 지침만 포함해 다른 언어에 대한 평가가 한계가 있었으나, M-IFEval은 프랑스어, 일본어, 스페인어를 포함하여 평가 범위를 넓혔습니다. 이를 통해 LLM의 성능을 더 폭넓게 평가할 수 있는 가능성을 제시합니다.

- **Technical Details**: M-IFEval은 일반 지침과 언어 특화 지침을 포함하여 다국어 환경에서의 LLM 성능을 체계적으로 평가하도록 설계되었습니다. 총 8개의 최신 LLM에 이 기준을 적용하여 성능을 비교하였으며, 각 언어와 지침 유형에 따른 성능 차이를 분석했습니다. 이러한 평가 방식은 LLM의 실제 활용 가능성을 측정하는 데 중요한 역할을 할 것입니다.

- **Performance Highlights**: M-IFEval을 적용한 결과, 각 언어와 지침 유형에 따른 LLM의 성능이 크게 달라지는 것을 발견했습니다. 이는 특히 다양한 문화 맥락을 고려할 때, LLM의 평가에 있어 다국어 기준의 필요성을 강조합니다. 이러한 접근법은 전 세계의 다양한 언어 사용자들에게 향상된 AI 경험을 제공할 수 있는 방향을 제시합니다.



### G2PDiffusion: Genotype-to-Phenotype Prediction with Diffusion Models (https://arxiv.org/abs/2502.04684)
- **What's New**: 본 논문은 G2PDiffusion이라는 최초의 다계열(diffusion) 모델을 도입하여 유전자형(genotype)에서 표현형(phenotype)으로의 변환 문제를 해결하고자 합니다. 기존의 단순화된 가정에 의존하는 전통적인 모델을 넘어선 이 방법은 여러 종에 걸친 데이터에 대해 유전자형과 표현형 간의 관계를 정밀하게 예측할 수 있도록 설계되었습니다. 특히, 이 연구는 이미지 기반의 표현형 예측을 조건부 이미지 생성(task of conditional image generation)으로 재정의하여 구현합니다.

- **Technical Details**: 논문에서는 환경 향상 DNA 시퀀스 조정기(environment-enhanced DNA sequence conditioner)와 동적인 정렬 모듈(dynamic alignment module)을 포함하는 G2PDiffusion 모델을 제안합니다. 이 두 가지 설계는 유전자형과 표현형 간의 일관성을 높이고, 복잡한 유전자와 환경 간의 상호 작용을 효과적으로 반영할 수 있도록 돕습니다. 또한, 고유한 평가 지표를 도입하여 예측 정확도를 평가합니다.

- **Performance Highlights**: 경대한 실험 결과, G2PDiffusion 모델은 여러 종에서 표현형 예측의 정확도와 일관성을 높이는 데 성공하였으며, 특히 미세한 유전적 변이도 포착할 수 있음을 보여주었습니다. 이러한 성과는 전통적인 방법에 비해 탁월한 성능을 나타내며, 다양한 생물학적 맥락에서 모델의 적용 가능성을 시사합니다. 본 접근법은 식물 및 척추동물 등 다른 생물군으로의 확장이 가능하다는 점에서 큰 잠재력을 가지고 있습니다.



### AdParaphrase: Paraphrase Dataset for Analyzing Linguistic Features toward Generating Attractive Ad Texts (https://arxiv.org/abs/2502.04674)
Comments:
          Accepted to NAACL2025 Findings

- **What's New**: 이번 연구에서는 소비자를 유치하기 위한 효과적인 언어적 선택이 광고 성공에 미치는 영향을 탐구했습니다. 특히, 이 연구는 언어적 기능이 인간의 선호도에 미치는 영향을 이해하는 데 중점을 두고 있습니다. 연구자들은 인간의 선호를 반영한 광고 텍스트의 패러프레이즈(Paraphrase) 데이터셋인 AdParaphrase를 발표하여 언어적 차이에 따른 선호도 분석을 가능하게 했습니다.

- **Technical Details**: AdParaphrase 데이터셋은 의미가 동일하지만 언어적 스타일과 표현이 다른 광고 텍스트 쌍의 선호도를 포함하고 있습니다. 이 데이터셋은 광고 텍스트의 매력을 높이기 위해 언어적 특징에 대한 정량적 분석을 가능하게 하며, 선호되는 광고 텍스트가 더 높은 유창성(fluency)을 갖고, 길이가 길며, 명사가 많고, 괄호(bracket) 기호를 빈번히 사용한다는 결과를 보여줍니다. 이러한 특성을 고려한 광고 텍스트 생성 모델이 광고 매력을 크게 향상시킬 수 있음을 입증했습니다.

- **Performance Highlights**: 연구 결과, 인간 심사자들이 선호하는 광고 텍스트는 더 높은 유창성을 나타내고, 더 긴 길이를 가지며, 더 많은 명사를 포함하고, 괄호 기호의 사용이 두드러진다는 것을 확인했습니다. 이러한 발견을 바탕으로, 언어적 특성을 고려한 광고 생성 모델이 매력적인 텍스트 생성 성능을 크게 향상시킨다는 점이 강조되었습니다. 데이터셋은 공개적으로 이용 가능하여 향후 연구에 기여할 것입니다.



### CCS: Controllable and Constrained Sampling with Diffusion Models via Initial Noise Perturbation (https://arxiv.org/abs/2502.04670)
- **What's New**: 이 논문에서는 생성 모델 중 하나인 diffusion 모델을 위한 새로운 접근법인 Controllable and Constrained Sampling (CCS) 방법을 제안합니다. 이를 통해 초기 노이즈의 변동이 생성 데이터에 미치는 영향을 면밀히 분석하여, 높은 품질의 샘플을 유지하면서 통계적 속성을 원하는 대로 조정할 수 있는 방법론을 제공합니다. 실험 결과, CCS 방법이 이전의 방법들에 비해 높은 제어성과 데이터 품질을 보인다고 보고합니다.

- **Technical Details**: Diffusion 모델은 깨끗한 이미지를 점진적으로 노이즈를 추가하는 forward process과 노이즈가 섞인 이미지를 다시 정제하는 reverse process로 이루어져 있습니다. 이 모델의 핵심 과정에서는 각 단계에서 초기 노이즈의 변동이 생성 output에 미치는 영향을 분석하여, 선형 관계가 성립한다는 사실을 발견하였습니다. 제안된 CCS 방법은 초기 노이즈 벡터의 구형 보간을 기반으로 하여, 설정된 통계적 속성에 맞춰 샘플을 생성하는 새로운 제어 알고리즘을 특징으로 합니다.

- **Performance Highlights**: 실험은 세 가지 데이터셋을 대상으로 CCS 방법의 효과를 검증하며, 기존 baseline 방법들과 비교하였습니다. 그 결과, CCS 방법은 원하는 목표 평균 이미지 주위에서 더욱 정밀하게 제어된 샘플링을 제공하고, 샘플의 질 및 다양성을 우수하게 유지함을 보여주었습니다. 이러한 결과는 CCS 방법이 이미지 편집 등 더 넓은 응용 가능성에 기여할 수 있음을 시사합니다.



### A Comprehensive Review on Noise Control of Diffusion Mod (https://arxiv.org/abs/2502.04669)
- **What's New**: 최신 연구에서는 확산 모델( diffusion model )이 고품질 이미지를 생성하는 데 있어 강력한 생성 프레임워크로 두각을 나타내고 있습니다. 이 모델의 핵심 요소는 노이즈 스케줄(noise schedule)로, 확산 과정 중 노이즈가 주입되는 속도를 조절합니다. 노이즈 스케줄은 샘플링 품질과 훈련 품질에 결정적인 영향을 미치므로, 이를 이해하는 것이 중요합니다. 본 논문에서는 다양한 노이즈 스케줄을 검토하고 그 특징 및 성능 특성을 비교 분석합니다.

- **Technical Details**: 확산 모델은 매개변수화된 마르코프 체인(Markov Chain)을 기반으로 하여 데이터 생성을 수행합니다. 이 모델은 전방 과정에서 가우시안 노이즈를 점진적으로 추가하고, 역방 과정에서 이를 제거함으로써 원래 데이터를 재구성하는 방식으로 작동합니다. 각 반복 단계에서 추가되는 노이즈의 양은 노이즈 스케줄에 의해 조절되며, 이 스케줄이 모델의 성능에 중대한 영향을 미칩니다. 따라서 올바른 노이즈 스케줄 선택은 고품질 이미지 생성의 필수 요소입니다.

- **Performance Highlights**: 확산 모델은 기존의 생성 모델에 비해 더 높은 유연성과 적은 제약으로 인해 이미지 생성 및 편집 능력을 크게 향상시킵니다. 이 모델은 훈련된 신경망을 통해 새로운 이미지를 생성할 수 있으며, 이는 학습한 데이터 분포의 표현으로서 작용합니다. 또한, 개인 정보 보호를 위해 생성된 합성 데이터를 활용할 수 있는 가능성을 통해 블록체인 응용 프로그램에서도 활용될 수 있습니다. 노이즈 추가 및 제거의 반복적인 과정은 물리적 확산 현상을 모방하며, 이는 확산 모델의 이름에도 뿌리 깊은 관련이 있습니다.



### Unveiling the Mechanisms of Explicit CoT Training: How Chain-of-Thought Enhances Reasoning Generalization (https://arxiv.org/abs/2502.04667)
- **What's New**: 이 논문은 Chain-of-Thought(이하 CoT) 주석을 활용한 대형 언어 모델(LLM)의 학습이 추론 능력을 크게 향상시킴을 보여준다. CoT 학습 적용의 이점, 즉 in-distribution(ID) 및 out-of-distribution(OOD) 시나리오에서의 추론 일반화 수준을 높이고, 수렴 속도를 가속화하는 것을 처음으로 명확히 밝혔다. CoT 임무 수행 중 발생할 수 있는 일부 오류에도 불구하고 모델이 추론 패턴을 학습하고 체계적으로 일반화할 수 있음을 보인다.

- **Technical Details**: 이 논문에서는 데이터 분포의 요소들, 즉 비율(λ) 및 패턴이 모델의 체계적인 일반화에 미치는 영향을 분석한다. CoT 학습은 두 단계의 일반화 회로를 형성하여 추론 단계를 내재화하는데, 이 단계 수는 학습 중의 명시적 추론 단계와 일치한다. 또한, 혼합된 데이터 분포를 통해 추론 과정에서 오류가 존재할 때에도 효과적인 일반화가 가능하다는 것을 발견하였다.

- **Performance Highlights**: 이 연구는 CoT 학습이 대형 언어 모델의 성능에 미치는 영향을 실험적으로 탐구하였고, CoT 학습의 유용성에 대한 구체적인 메커니즘을 제공한다. 그 결과, CoT 훈련이 효과적인 패턴 인식 및 일반화 능력을 향상시키는 데 기여하는 방식을 제시하며, LLM의 튜닝 전략에 대한 중요한 통찰력을 제공한다. 이러한 발견은 향후 LLM 발전에 상당한 영향을 미칠 것으로 예상된다.



### Shifting Attention to You: Personalized Brain-Inspired AI Models (https://arxiv.org/abs/2502.04658)
Comments:
          7 Figures, 3 Tables, 3 Supplemental Figures, 1 Supplemental Table

- **What's New**: 이 연구는 인간 행동 임베딩과 신경 데이터를 통합하여 CLIP 모델을 개인화된 인지 프로세스에 맞게 조정한 새로운 접근 방식을 발표했습니다. 이 연구는 CLIP-Human-Based Analysis (CLIP-HBA)라는 프레임워크 하에 대규모 행동 결정과 신경 데이터를 점진적으로 조정하여, AI와 인간 인지 간의 더 나은 정렬을 목표로 하고 있습니다. 특히 MEG 신경 동적 정보를 활용하여 인지 프로세스의 기계적 통찰을 추구하며 개인화된 AI 시스템의 가능성을 강조합니다.

- **Technical Details**: 연구에서는 66개의 Sparse Positive Similarity Embedding (SPoSE) 차원을 사용하여 CLIP 모델을 조정했습니다. 이 과정을 통해 모델은 시각적 자극에 대한 인간 유사성 판단을 효과적으로 예측할 수 있는 임베딩을 생성하였습니다. CLIP-HBA-Behavior는 행동 기반 조정이 이루어진 모델로, 평균 제곱 오차 (MSE) 손실 함수를 통해 모델의 예측을 최적화하여 인간의 인지에 더 적합한 출력 결과를 생성합니다.

- **Performance Highlights**: 최종적으로, CLIP-HBA-Behavior 모델은 인간 행동과의 정렬에서 0.78의 Spearman 상관관계를 달성하여 기존 CLIP-ViT 모델보다 100% 이상의 개선을 보였습니다. 이러한 결과는 CLIP-HBA-Behavior 모델이 행동의 유사성 판단을 더 잘 수행한 것을 나타내며, 향후 개인화된 AI 모델 개발에 중요한 기반이 될 수 있음을 보여줍니다.



### Importance Sampling via Score-based Generative Models (https://arxiv.org/abs/2502.04646)
Comments:
          18 pages

- **What's New**: 이 논문에서는 score-based generative model (SGM)에서 기초한 완전한 훈련 불필요한 importance sampling 프레임워크를 제안합니다. 기존의 방법과는 달리, 이 시스템은 중요한 샘플을 생성하는 과정을 후방 확산 과정(backward diffusion process)으로 인식하여 모든 추가적인 훈련이 필요 없도록 설계되었습니다. 이는 실제적인 상황에서 메인 분포(base distribution)가 여러 가지 바이어스 샘플링 작업을 지원하도록 하여 다양한 중요도 가중치 함수(importance weight function)를 사용할 수 있게 합니다.

- **Technical Details**: 제안된 중요도 샘플링 방법은 주어진 기초 PDF의 score function과 명시된 중요도 가중치 함수만으로 구축되며, 추가적인 훈련은 요구되지 않습니다. 이 과정은 기본 분포의 score function과 중요도 가중치 함수의 조합을 통한 근사 표현을 통해 이루어집니다. 또한, 이 논문은 정확한 근사도를 평가하고 전반적인 활용성에 대해 철저한 분석도 제공하고 있습니다.

- **Performance Highlights**: 우리의 방법은 다양한 합성 시나리오에서 목표 중요도 샘플링 분포와의 근접성을 평가하여 샘플링의 정확성을 입증했습니다. 이 접근법은 신경망으로 모델링된 다양한 중요도 가중치 함수에 대한 샘플링을 가능하게 하여, 이전 연구들에서 어려웠던 문제들을 효과적으로 해결할 수 있도록 합니다. 특히, 특정 다운스트림 작업에 대해 고왜곡(distortion) 유발 사례를 목표로 하는 샘플링 설계와 같은 도전적인 샘플링 작업을 포함합니다.



### Cross-Encoder Rediscovers a Semantic Variant of BM25 (https://arxiv.org/abs/2502.04645)
- **What's New**: 이번 연구에서는 Neural Ranking Models (NRMs)의 새로운 변형인 Cross-Encoder를 조사하여, 이들이 어떤 relevance features (관련성 특징)을 계산하는지, 그리고 이 정보가 어디에 저장되는지를 분석합니다. MiniLM의 Cross-Encoder 변형이 BM25의 의미론적 변형을 사용한다는 것을 발견했으며, 이는 문서 길이 및 용어 포화 효과를 조절하는 Transformer attention heads와 vocab에 대한 역 문서 빈도 정보를 인코딩하는 저차원(vector) 구성 요소를 포함합니다. 이러한 통찰력은 모델 수정(model editing)의 가능성을 열어주며, 투명성을 높이고 안전성 문제를 주소할 수 있는 기초를 제공합니다.

- **Technical Details**: 연구에서는 이전의 상관 연구를 바탕으로 기계적 해석 가능성(mechanistic interpretability) 방법을 사용하여 Cross-Encoder의 BM25 유사 신호 구현을 검증했습니다. 특히, path patching을 이용하여 BM25 유사 구성 요소를 계산하는 attention heads를 식별하고, 이 정보를 다른 relevance scoring heads에 전달함으로써 BM25 스타일 함수가 어떻게 통합되는지를 밝혔습니다. 또한, 저차원 벡터가 embedding matrix에서 IDF 정보를 포함하고 있다는 증거를 발견했고, 이를 통해 용어의 중요성을 조정할 수 있습니다.

- **Performance Highlights**: 본 연구의 주요 결과는 Cross-Encoder가 relevance scoring heads를 통해 BM25 스타일의 계산을 구현한다는 것입니다. 시스템의 경로를 역으로 추적하여 관련성을 계산하는 과정을 반영한 획기적인 이해가 이루어졌습니다. 이로써 NRMs의 성능 개선, 통제 가능성, 개인화 및 편향 완화의 기회가 확장될 수 있음을 시사합니다.



### Learning Street View Representations with Spatiotemporal Contras (https://arxiv.org/abs/2502.04638)
- **What's New**: 이 논문에서는 도시 시각 환경에서 동적 요소와 정적 요소를 효과적으로 학습할 수 있는 새로운 자기 감독 학습 프레임워크를 제안합니다. 특히, 도로 및 건물 등 정적 환경은 시간에 따라 불변성을 갖고, 보행자 및 차량 등 동적 요소는 무작위성을 가지기 때문에 정보를 효과적으로 인코딩하기 위해 두 가지 속성을 활용합니다. 시각적 장소 인식 및 사회경제적 추정 같은 다양한 하위 작업에서도 우수한 성능을 보였습니다.

- **Technical Details**: 이 연구에서 제안된 프레임워크는 두 가지 주요 속성을 활용합니다. 첫째, 시간 불변성 표현 (Temporal Invariance Representation)은 동일한 장소에서 찍힌 이미지들이 시간에 따라 정적 요소는 유지하고 동적 요소는 자동으로 필터링되도록 학습합니다. 둘째, 공간 불변성 표현 (Spatial Invariance Representation)은 인접한 지역의 이미지에서 유사한 건축 스타일이나 도시 기능을 활용해 전체 이웃 분위기를 인코딩합니다.

- **Performance Highlights**: 이 연구는 여러 도시 관련 하위 작업에서 제안된 방법이 기존의 감독 학습 및 비감독 학습 방법에 비해 우수한 성능을 보였음을 보여줍니다. 각기 다른 상대적 대조 학습 목표가 다양한 종류의 특성을 학습할 수 있음을 실험적으로 입증하였으며, 이는 도시 연구 및 시각 데이터의 적용 가능성을 높이는 데 기여합니다. 또한, 다양한 대조 방법들의 성능을 분석하여 목표 지향적 학습 전략의 중요성을 강조합니다.



### An Empirical Study of Code Obfuscation Practices in the Google Play Stor (https://arxiv.org/abs/2502.04636)
Comments:
          13 pages, 8 figures

- **What's New**: 이 논문은 Google Play 스토어의 Android 애플리케이션에서 코드 난독화(code obfuscation) 기법의 사용 현황을 대규모로 분석한 최초의 연구입니다. 500,000개 이상의 APK 파일을 분석하여 2016년부터 2023년까지의 변화 추이를 조사하였으며, 난독화 사용률이 13% 증가한 것으로 나타났습니다. 또한 게임 애플리케이션에서 더 높은 난독화 사용률을 보였고, ProGuard와 Allatori가 가장 흔히 사용되는 도구로 확인되었습니다.

- **Technical Details**: 코드 난독화는 소스 코드를 인간이 읽을 수 없는 형태로 변환하는 과정을 포함합니다. 일반적으로 Identifier Renaming, Control Flow Modification, Call Indirection과 같은 기법이 사용되며, 각 기법은 애플리케이션의 기능을 유지하면서 코드를 읽기 어렵게 만듭니다. 이 연구에서는 다양한 난독화 탐지 분류기를 개발하여 연구에 활용하였고, 2016-2023년 동안의 데이터를 분석하여 난독화 기법의 진화 과정을 살펴보았습니다.

- **Performance Highlights**: 이 연구의 결과에 따르면, 2016년부터 2023년 사이에 전체적으로 Google Play 스토어에서의 코드 난독화 사용이 13% 증가했으며, 상위 1,000개의 앱 가운데 90% 이상이 난독화되어 있습니다. 특히, Casino 게임 앱에서는 80% 이상의 앱에서 난독화가 적용되었고, 상위 개발자들 사이에서는 28%의 난독화 증가율을 보였습니다. 이 데이터는 개발자와 보안 분석가에게 중요한 인사이트를 제공하여, 난독화의 필요성을 강조합니다.



### Extracting and Understanding the Superficial Knowledge in Alignmen (https://arxiv.org/abs/2502.04602)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 정렬(Alignment) 과정에서 표면적 지식(Superficial Knowledge)의 개념을 정립하고, 이를 추출하여 분석하는 방법론을 제시합니다. 특히, 기존의 무거운 조정 과정을 간소화할 수 있다는 점에서, 표면적 지식이 정렬의 중요한 일부분임을 강조합니다. 이는 기존의 연구들이 관찰에 기반하여 하고 있는 가설들과는 차별화되는 접근입니다. 또한, 이 연구에서는 수정이 최소화된 상태에서도 모델의 성능과 안전성을 확보할 수 있는 방법도 제시하고 있습니다.

- **Technical Details**: 정렬된 모델과 정렬되지 않은 모델 간의 상관관계를 설명하기 위해, 각 모델의 트랜스포머 레이어와 최종 선형 프로젝션 매트릭스를 사용하여 모델 아키텍처를 정의합니다. 논문에서는 표면적 지식이 쉽게 얻어진다고 정의하고, 이 지식이 모델의 심층 구조에는 영향을 주지 않도록 하여, 지식을 추출하는 방법을 명확히 하고 있습니다. 추출된 지식은 다양한 벤치마크에서 수학, 안전성, 독성, 진실성 과제를 통해 정량적으로 평가됩니다.

- **Performance Highlights**: 연구 결과에 따르면, 표면적 지식은 안전성과 독성 작업에서 상당한 효과를 나타내며, 모델의 답변 구조화에 기여하는 스타일 패턴으로 이루어져 있습니다. 이 지식만으로도 평균 58%의 수학 성과와 78%의 진실성 작업 개선을 이끌어냈습니다. 그러나, 정렬은 전적으로 표면적이지는 않으며, 특히 사고력과 맥락 이해가 요구되는 작업에서는 깊은 지식이 여전히 필요하다는 점이 강조됩니다.



### The $\alpha$-Alternator: Dynamic Adaptation To Varying Noise Levels In Sequences Using The Vendi Score For Improved Robustness and Performanc (https://arxiv.org/abs/2502.04593)
Comments:
          The codebase will be made available upon publication. This paper is dedicated to Patrice Lumumba

- **What's New**: 이번 논문에서는 기존의 상태 기반 모델들이 동일한 수준의 노이즈를 가정하는 한계를 극복하기 위해 $eta$-Alternator라는 새로운 생성 모델을 제안합니다. 이 모델은 시간 종속 데이터에서 다양한 노이즈 수준에 적응하여 성능을 향상시키는 데 중점을 두고 있습니다. $eta$-Alternator는 Vendi Score (VS)를 활용하여 각 시간 단계에서의 시퀀스 요소의 영향력을 동적으로 조정합니다.

- **Technical Details**: $eta$-Alternator는 특정 파라미터를 학습하고 이를 통해 데이터셋 내 모든 시퀀스의 영향을 공유합니다. 이 파라미터의 부호는 해당 요소의 노이즈 수준을 나타내며, 음수이면 해당 요소가 노이즈로 간주되고 양수이면 정보로 간주됩니다. 모델은 관측 마스킹과 Alternator 손실 최소화를 통해 훈련되며, 이를 통해 시퀀스의 다양할 수 있는 노이즈 수준에 대해 복원력을 제공합니다.

- **Performance Highlights**: 실험 결과, $eta$-Alternator는 다양한 신경 디코딩 및 시계열 예측 벤치마크에서 기존의 Alternators 및 최신 상태 공간 모델을 능가했습니다. 이 모델은 궤적 예측, 누락 데이터 보완 및 예측에서 특히 두드러진 성능을 보이며, 노이즈가 많은 데이터셋에서도 효율적으로 작동할 수 있도록 개선되었습니다.



### CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements (https://arxiv.org/abs/2502.04592)
- **What's New**: 이 논문은 CAMEF(Causal-Augmented Multi-Modality Event-Driven Financial Forecasting)라는 새로운 다중 모달 프레임워크를 제안합니다. CAMEF는 텍스트와 시계열 데이터를 효과적으로 통합하고, 원인 학습(causal learning) 메커니즘 및 LLM 기반의 반사실적(counterfactual) 이벤트 증강 기술을 활용하여 재무 예측의 정확성을 높입니다. 이러한 접근법은 기존의 예측 방법들이 간과했던 사건과 시장 행동 간의 인과 관계를 포착할 수 있습니다.

- **Technical Details**: CAMEF는 시간 시리즈와 텍스트 피처를 통합하는 다중 피처 융합 기술, 시계열 디코딩 메커니즘 및 인과 학습 전략을 통해 설계되었습니다. 이 프레임워크는 2008년부터 2024년 4월까지의 6가지 주요 거시경제 이벤트의 새로운 데이터셋을 포함하고 있으며, 주요 미국 금융 자산에 대한 고빈도 거래 데이터를 제공합니다. 또한, 대응적 사건 스크립트를 생성하기 위해 LLM 기반의 반사실적 데이터 증강 전략을 사용합니다.

- **Performance Highlights**: CAMEF는 최신의 트랜스포머 기반 시계열 및 다중 모달 기준선 모델과 비교되었으며, 인과 학습 메커니즘과 사건 유형의 효과성을 검증하기 위한 약물 연구(ablation study)도 수행되었습니다. 그 결과, CAMEF는 예측의 정확성을 높이며, 투자자 및 정책 입안자에게 중요한 인사이트를 제공합니다. 이 모델은 GitHub를 통해 공개되어 재현성과 미래 연구에 유용하게 활용될 예정입니다.



### Rethinking Oversmoothing in Graph Neural Networks: A Rank-Based Perspectiv (https://arxiv.org/abs/2502.04591)
- **What's New**: 이번 논문에서는 그래프 신경망(GNN)에서 발생하는 오버스무딩(oversmoothing)을 새로운 차원에서 분석합니다. 기존의 Dirichlet 에너지 같은 유사도 측정을 사용하는 방법론의 한계를 설명하고, 노드 특성의 수치적(rank) 혹은 유효한(rank) 정도를 통해 오버스무딩을 평가할 것을 제안합니다. 이로 인해 GNN 성능의 저하를 더 신뢰성 있게 포착할 수 있음을 보여주고 있습니다.

- **Technical Details**: 논문은 오버스무딩을 정량화하기 위해 네트워크 특성 표현의 연속적 근사(ranking)를 이용하는 방법을 제안합니다. 저자는 비선형 활성화 함수(non-linear activation functions)와 관련하여 특성 표현의 수치적(rank) 진행이 1로 수렴하는 경향을 이론적으로 보증합니다. 이 결과는 비선형 활성화 함수를 사용할지라도 가중치의 크기에 관계없이 오버스무딩이 발생할 수 있음을 보여주는 최초의 사례입니다.

- **Performance Highlights**: 실험을 통해 저자는 다양한 GNN 아키텍처에서 연속적 rank 완화 방식이 성능 저하와 밀접하게 관련되어 있음을 확인했습니다. 특히, 이 논문에서는 에너지 기반 메트릭이 충실하지 못한 상황에서도 rank 기반 메트릭이 오버스무딩을 일관되게 포착함을 밝혔다. 이러한 실험 결과는 GNN의 성능 저하가 기존 메트릭보다 더 전반적이며 신뢰할 수 있는 지표로서 rank를 활용할 수 있음을 보여줍니다.



### Technical Debt in In-Context Learning: Diminishing Efficiency in Long Contex (https://arxiv.org/abs/2502.04580)
- **What's New**: 이 연구는 Transformers의 In-Context Learning (ICL) 기법의 최적 학습 알고리즘으로서의 효율성을 정량화하는 새로운 프레임워크를 제introduce합니다. ICL은 다양한 작업에서 놀라운 성능을 보이며, 기존의 작업 특화 모델을 초월할 가능성이 있습니다. 그러나 ICL이 이론적 최적성을 얼마나 잘 달성하는지에 대한 명확한 이해가 부족합니다.

- **Technical Details**: 연구진은 메타 ICL 환경을 설정하고, Transformer의 성능을 비교하기 위해 원리 기반의 예측기(principle predictors)를 설계합니다. 이 과정에서 ICL의 샘플 복잡도와 성능을 상대적으로 측정하며, ICL이 Bayes 최적 추정기(Bayes optimal estimator)의 효율성을 일시적으로 달성하지만, 긴 컨텍스트에서는 급격히 효율이 감소한다는 사실을 드러냅니다. 이 연구는 ICL의 고유한 비효율성을 강조하기 위해 정보 이론적 도구를 활용합니다.

- **Performance Highlights**: ICL은 낮은 성능 요구사항에서는 Bayes 최적 추정기와 유사한 샘플 복잡도를 달성할 수 있지만, 특정 임계치를 초과하면 성능을 유지하기 위해 평균 1.5배 더 많은 시연이 필요하다는 결론을 도출합니다. 이러한 결과는 ICL이 단순한 데이터나 모델 크기 증가로 해결할 수 없는 기본적 기술적 부채를 가지고 있음을 시사합니다. 연구 결과는 AI 시스템 설계에서 ICL을 보편적인 문제 해결자로 채택할 때 고려해야 할 트레이드오프를 명확히 합니다.



### Zero-shot Meta-learning for Tabular Prediction Tasks with Adversarially Pre-trained Transformer (https://arxiv.org/abs/2502.04573)
- **What's New**: 이 논문에서는 Adversarially Pre-trained Transformer (APT)를 소개합니다. APT는 실제 데이터셋에 대한 사전 학습 없이 표 형식의 예측 작업에서 제로샷 메타 학습을 수행할 수 있습니다. 이 모델은 데이터를 생성하는 분포를 지속적으로 변화시키는 적대적 합성 데이터 에이전트들로부터 훈련되어, 다양한 합성 데이터셋으로 모델을 도전하게 합니다.

- **Technical Details**: APT 모델은 랜덤 합성 데이터 생성기와 적대적 합성 데이터 에이전트를 혼합하여 오프라인에서 한 번 사전 학습됩니다. 실제 데이터셋의 테스트 세트에 대한 예측 시, APT는 한 번의 포워드 패스로 예측을 수행하며 가중치의 역전파나 그래디언트 업데이트를 하지 않습니다. 이를 통해 모델은 데이터셋의 특정 패턴을 배우는 것이 아니라 다양한 데이터를 표현하는 일반적인 예측 로직을 학습합니다.

- **Performance Highlights**: APT는 소규모 분류 작업에서 최신 성능을 달성하며, 데이터셋의 클래스 크기나 결측치 수에 상관없이 훈련이 가능합니다. 또한, 적대적 데이터 에이전트를 통해 합성 데이터 생성 분포를 풍부하게 하고, 혼합 블록 아키텍처를 통해 미리 학습속도를 크게 증가시켰습니다. 이 실험 결과는 APT가 TabPFN을 개선하며 분류와 회귀 모두에서 우수한 성능을 보여주었음을 나타냅니다.



### WaferLLM: A Wafer-Scale LLM Inference System (https://arxiv.org/abs/2502.04563)
- **What's New**: 본 논문에서는 WaferLLM을 소개하는데, 이는 최초의 웨이퍼 스케일 LLM 추론 시스템입니다. 이 시스템은 PLMR 장치 모델을 기반으로 하여 웨이퍼 스케일 아키텍처의 고유한 하드웨어 특성을 포착합니다. WaferLLM은 수십만 개의 온칩 코어를 최적화하여 활용하고, 웨이퍼 스케일 가속기를 효과적으로 활용하는 다양한 메모리 최적화 기법을 적용합니다.

- **Technical Details**: WaferLLM은 웨이퍼 스케일 가속기를 위한 여러 새로운 연산을 도입합니다. MeshGEMM과 MeshGEMV라는 새로운 GEMM 및 GEMV 구현은 웨이퍼 스케일 아키텍처에 최적화되어 있으며, 이로 인해 LLM 프리필 및 디코드 단계에서 더 높은 효율을 얻습니다. PLMR 모델은 대규모 코어와 비균일 메모리 접근 지연, 제한된 로컬 메모리 등의 특성을 반영하여 설계되었습니다.

- **Performance Highlights**: WaferLLM은 기존 시스템들과 비교할 때 200배에서 606배 더 나은 가속기 활용도를 보여주며, 39배 더 빠른 디코딩 성능과 1.7배 향상된 에너지 효율을 달성했습니다. 실험 결과, WaferLLM은 Cerebras WSE 엔진에서 7,000 라인의 CSL과 함께 구현되어, 대규모 LLM 시스템에서 효율성을 높이고 향후 웨이퍼 스케일 AI 모델의 성능을 크게 향상시킬 것으로 기대됩니다.



### Probing a Vision-Language-Action Model for Symbolic States and Integration into a Cognitive Architectur (https://arxiv.org/abs/2502.04558)
Comments:
          8 Pages, 4 Figures

- **What's New**: 이 연구는 OpenVLA의 숨겨진 레이어를 조사하여 객체 특성, 관계 및 행동 상태의 기호 표현을 밝혀내는 방법을 제시합니다. 이는 기호적 아키텍처(Cognitive Architecture, CA)와 통합함으로써 로봇의 해석 가능성과 견고성을 향상시킵니다. 실험 결과, OpenVLA의 레이어에서 객체 및 행동 상태에 대한 예측 정확도가 0.90을 초과하여 일관되게 높은 수준을 보였습니다.

- **Technical Details**: OpenVLA는 Llama 2 언어 모델과 시각 인코더를 결합하여 전처리된 특징을 통합한 오픈 소스 VLA 모델입니다. 본 연구에서는 OpenVLA의 33개 숨겨진 레이어를 선형 프로빙(Linear Probing)하여 기호적 상태를 예측합니다. 연구의 주요 가설은 객체 상태가 이전 레이어에, 행동 개념이 후속 레이어에 인코딩된다는 것입니다.

- **Performance Highlights**: DIARC-OpenVLA 시스템에 통합하여 실시간 기호 모니터링을 구현함으로써 보다 해석 가능하고 신뢰할 수 있는 로봇 조작의 기초를 마련했습니다. 연구 결과 OpenVLA의 모든 레이어에서 높은 정확도를 기록하였으며, 다양한 조작 작업에서 강력한 일반화 능력을 보여주었습니다. 이러한 발견은 로봇 시스템의 신뢰성 있는 기호적 추론을 위한 연구에 기여할 것으로 기대됩니다.



### TruthFlow: Truthful LLM Generation via Representation Flow Correction (https://arxiv.org/abs/2502.04556)
- **What's New**: TruthFlow라는 새로운 방법이 제안되었습니다. 이 방법은 다양한 질의(query)에 대해 진실한 표현(correct representation) 수정이 가능하도록 Flow Matching 기술을 활용합니다. TruthFlow는 질의를 위한 특정 수정 벡터를 학습하고, 이를 통해 대화형 AI의 진실성을 보강합니다.

- **Technical Details**: TruthFlow는 흐름 모델(flow model)을 통해 질의 특화된 수정 벡터를 학습하고, 이를 사용하여 환각 상태에서 진실 상태로의 전환을 수행합니다. 이 모델은 특정 질의의 표현을 입력받아 해당 질의의 진실한 표현 수정 벡터를 생성합니다. 인퍼런스(inference) 동안, TruthFlow는 생성된 수정 벡터를 사용하여 현재 질의의 표현을 수정하여 결과의 진실성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, TruthFlow는 TruthfulQA 기준에서 다양한 고급 LLM에 대한 성능을 유의미하게 개선하였습니다. 또한, TruthFlow 모델은 강력한 전이 가능성을 보이며, 보지 못한 다른 환각 벤치마크에서도 효과적으로 수행됩니다.



### AnyPlace: Learning Generalized Object Placement for Robot Manipulation (https://arxiv.org/abs/2502.04531)
- **What's New**: 이 연구에서는 로봇 작업에서의 객체 배치 문제를 해결하기 위해 AnyPlace라는 두 단계 방법을 제안합니다. 이 방법은 전적으로 합성 데이터(synthetic data)를 사용하여 훈련되어 다양한 실제 작업에 대해 실행 가능한 배치 자세를 예측할 수 있습니다. 특히, Vision-Language Model (VLM)을 활용하여 대략적인 배치 위치를 파악하고, 관련된 영역에만 집중하여 배치 자세 예측의 효율성을 높입니다.

- **Technical Details**: AnyPlace는 두 개의 하위 작업으로 배치 자세 예측 문제를 나누어 수행합니다: 고수준 배치 위치 제안(task)과 저수준 정밀 배치 자세 예측(task)입니다. 고수준 작업에서는 SAM-2 분할 모델을 사용하여 객체를 분할하고 Molmo VLM을 통해 모든 가능한 배치 위치를 제안합니다. 저수준 작업은 주어진 배치 위치 근처의 정보만을 입력으로 하여 다양한 배치 구성을 효과적으로 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 시뮬레이션 평가에서 AnyPlace는 기준 모델들보다 높은 성공률, 배치 위치 범위, 및 정확도를 달성하며 우수한 성능을 입증합니다. 실제 실험에서도, 이 방법은 80%의 성공률로 병 삽입 작업을 수행하며, 노이즈 데이터에 대한 내구성과 보지 못한 객체에 대한 일반화 능력을 보여줍니다. 이는 완전히 합성 데이터로 훈련된 모델이 실제 작업에 효과적으로 적용될 수 있음을 나타냅니다.



### ImprovNet: Generating Controllable Musical Improvisations with Iterative Corruption Refinemen (https://arxiv.org/abs/2502.04522)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 고객이 제어할 수 있는 표현력이 뛰어난 음악적 즉흥 연주를 생성할 수 있는 새로운 모델인 ImprovNet을 소개합니다. 이 모델은 Transformer 기반 아키텍처를 채택하였으며, 음악 스타일 및 타입을 손쉽게 변환할 수 있는 기능을 갖추고 있습니다. ImprovNet은 여러 음악 생성 작업을 통합하여 기존의 제약을 극복하며, 교차 장르(improv) 및 장르 내(intra-genre) 즉흥 연주를 수행할 수 있습니다.

- **Technical Details**: ImprovNet은 셀프 슈퍼바이즈드 학습(self-supervised learning)을 통해 훈련된 변환기(transformer) 기반 아키텍처로, 사용자가 스타일 전이의 정도와 원본 구성에 대한 구조적 유사성을 제어할 수 있는 반복적 생성 프레임워크를 제공합니다. 사용자는 출력되는 음악의 기본 멜로디, 조화(harmony), 리듬(rhythm) 등 음악 요소들을 수정해 즉흥적인 요소를 추가할 수 있습니다. 또한, 이 모델은 짧은 프롬프트 연장(prompt continuation)과 infilling 작업도 수행할 수 있습니다.

- **Performance Highlights**: ImprovNet은 참여자 79%가 재즈 스타일의 즉흥 연주를 올바르게 식별할 정도로 인식 가능한 장르 전환을 성공적으로 이뤄냅니다. 또한, 짧은 연장과 infilling 작업에서 기존의 Anticipatory Music Transformer보다 더 나은 성능을 보였습니다. 객관적 및 주관적 평가를 통해, ImprovNet은 원본 음악 작품과의 구조적 관계를 유지하면서 음악적으로 일관된 즉흥 연주를 생성하는 데 효과적이라는 것이 입증되었습니다.



### MedGNN: Towards Multi-resolution Spatiotemporal Graph Learning for Medical Time Series Classification (https://arxiv.org/abs/2502.04515)
Comments:
          Accepted by WWW 2025

- **What's New**: 이번 연구에서는 의료 시계열 데이터 분류를 위해 다중 해상도(spatial 및 temporal) 그래프 학습 프레임워크인 MedGNN을 제안합니다. 기존의 방법들은 복잡한 공간 역학을 모델링하는 데 한계가 있었지만, MedGNN은 다양한 해상도의 적응형 그래프 구조를 통해 이 문제를 해결하고자 합니다. 또한 기초선 이동(baseline wander) 문제와 다중 시각(multi-view) 특성을 동시에 고려하여 모델의 예측 성능을 극대화합니다.

- **Technical Details**: MedGNN 프레임워크는 먼저 동적 스페이셜-템포럴 표현을 학습하기 위해 다중 해상도 적응형 그래프 구조를 구축합니다. 이후, 기초선 이동 문제를 해결하기 위해 두 가지 네트워크인 Difference Attention Networks와 Frequency Convolution Networks를 도입하여 시간 도메인과 주파수 도메인 모두에서 중요한 패턴을 추출합니다. 마지막으로, Multi-resolution Graph Transformer 아키텍처를 사용해 다중 해상도로부터 정보를 융합하고 복잡한 종속성을 모델링합니다.

- **Performance Highlights**: 실험을 통해 MedGNN은 여러 의료 시계열 데이터셋에서 최첨단 방법들에 비해 우수한 성능을 보였습니다. 특히 ECG와 EEG 신호를 분석한 결과, MedGNN은 의료 분야의 실질적인 적용 가능성을 강조하며 뛰어난 예측 능력을 입증했습니다. 이러한 연구 결과는 복잡한 다차원 데이터를 효과적으로 처리할 수 있는 새로운 방법론을 제공하는 데 기여합니다.



### Revisiting Intermediate-Layer Matching in Knowledge Distillation: Layer-Selection Strategy Doesn't Matter (Much) (https://arxiv.org/abs/2502.04499)
- **What's New**: 본 논문은 Knowledge Distillation (KD)의 효과적인 적용에서 흥미로운 현상을 발견하였습니다. 구체적으로, 중간 레이어 매칭(intermediate-layer matching)에서 레이어 선택 전략이 크게 중요하지 않음을 보여줍니다. 예상치 못한 매칭 전략인 역순 레이어 매칭(reverse layer matching)조차도 유사한 성과를 내는 것으로 나타났습니다. 이는 학생 모델의 관점에서 교사 레이어 간의 각도가 날카롭기 때문이라는 해석을 제공합니다.

- **Technical Details**: KD 방법은 일반적으로 예측 매칭(prediction matching)과 중간 레이어 매칭으로 나뉘며, 학생 모델은 교사 모델의 은닉 상태를 통해 추가적인 감독 신호를 받습니다. 본 연구에서는 6개의 데이터 세트에서 4가지 매칭 전략(정방향, 역방향, 랜덤, 모든-하나)을 실험하였고, 다양한 심도와 매개변수 초기화 설정을 탐색하였습니다. 연구 결과는 매칭 전략의 차이가 성능에 미치는 영향이 크지 않다는 일관된 증거를 제시하였습니다.

- **Performance Highlights**: 이 논문의 실험 결과, 중간 레이어 매칭이 없는 경우에 비해 모든 매칭 전략이 KD에 긍정적인 영향을 미치며, 예상치 못한 유사한 성과를 보여주었습니다. 또한, 깊이나 구조에 관계없이 KD의 효과는 중간 레이어 매칭을 통해 극대화되며, 여러 데이터 세트에 걸쳐 일관된 성과를 얻었습니다. 이러한 결과는 레이어 선택 전략이 성능에 미치는 영향을 재고하게 만듭니다.



### CNN Autoencoders for Hierarchical Feature Extraction and Fusion in Multi-sensor Human Activity Recognition (https://arxiv.org/abs/2502.04489)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문에서는 Inertial Measurement Units (IMUs) 센서에서 수집된 신호를 활용한 인간 활동 인식(HAR)을 위한 새롭고 혁신적인 방법인 Hierarchically Unsupervised Fusion (HUF) 모델을 소개합니다. 이 모델은 Convolutional Neural Networks (CNNs)와 Autoencoders (AEs)를 결합하여 IMU 센서 데이터의 특징을 추출하고 융합합니다.

- **Technical Details**: 제안된 HUF 모델은 먼저 짧은 시간 신호를 고차원 특징 세트로 포함시키기 위해 스택 CNN-AE를 설계합니다. 이후 각 센서 유닛으로부터 추출된 특징을 지역적으로 융합하기 위한 추가 CNN-AE 네트워크를 개발하며, 마지막으로 전역적 특징 융합을 위해 세 번째 CNN-AE 아키텍처를 통해 모든 센서 특징을 통합하여 독특한 특징 세트를 생성합니다.

- **Performance Highlights**: 모델의 하이퍼파라미터를 조정한 결과, 각 AE에 여덟 개의 convolutional layer를 사용하는 것이 최상의 결과를 보였습니다. 또한, 첫 번째 블록에서 256개의 kernels을 가진 overcomplete AE가 특징 추출에 적합하다는 것을 확인했으며, 마지막 블록에서는 분류기에 적용될 특징의 크기를 맞추기 위해 64로 줄였습니다. 조정된 모델은 UCI-HAR, DaLiAc 및 파킨슨병 보행 데이터셋에 적용되어 각각 97%, 97%, 88%의 분류 정확도를 달성하며, 이는 최신 감독 방법보다 약 3% 개선된 결과입니다.



### Building A Unified AI-centric Language System: analysis, framework and future work (https://arxiv.org/abs/2502.04488)
- **What's New**: 최근 대규모 언어 모델에 대한 발전은 연장된 추론을 통해 성능을 크게 향상시킬 수 있음을 보여주고 있습니다. 그러나 이러한 개선은 계산 비용의 증가와 자연어에서 발견되는 고유한 편견의 전파를 동반합니다. 이 논문은 이러한 문제를 해결하기 위해 보다 간결하고 명확하며 계산 효율적인 AI 중심 언어 시스템의 설계를 탐구합니다.

- **Technical Details**: 자연어의 성별 편향, 형태론적 불규칙성, 문맥적 모호성과 같은 한계를 분석하고, 이러한 문제들이 현재의 Transformer 아키텍처 내에서 어떻게 악화되는지 살펴봅니다. 또한 이 논문은 인공지능에 최적화된 새로운 언어 설계를 통해 계산 효율성을 높이고 편향과 모호성을 줄이는 방법을 제안합니다.

- **Performance Highlights**: AI 중심 언어를 통해 다양한 자연어 입력을 효율적으로 처리할 수 있으며, 이를 통해 메모리 사용량과 추론 시간을 줄일 수 있습니다. 최종적으로 이러한 접근 방식을 통하여 AI 간의 상호 작용과 인간-AI 상호 작용의 명확성, 공정성 및 성능을 획기적으로 혁신할 수 있는 가능성을 제시합니다.



### Active Task Disambiguation with LLMs (https://arxiv.org/abs/2502.04485)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 성능이 우수하지만, 실제 상호작용에서 자주 발생하는 모호하게 지정된 문제를 해결하는 능력이 충분히 탐구되지 않았음을 지적합니다. 이를 해결하기 위해 연구자는 작업 모호성(task ambiguity)에 대한 공식 정의를 제시하고, 베이esian 실험 설계(Bayesian Experimental Design)의 관점에서 작업 불명확성(task disambiguation) 문제를 구성합니다.

- **Technical Details**: 작업 문제를 명확히 하기 위한 질문을 제시함으로써 LLM 에이전트는 추가적인 작업 명세를 획득하고, 가능한 솔루션 공간을 점진적으로 좁힐 수 있습니다. 그러나 효과적인 명확화 질문을 생성하는 것은 LLM 에이전트가 메타 인지적 추론(meta-cognitive reasoning)을 수행할 능력이 필요하지만, 현재 LLM이 이러한 능력을 결여하고 있음을 지적합니다. 이 연구는 적극적인 작업 불명확성 해소(active task disambiguation) 접근 방식을 제안하여 정보 이득(information gain)을 극대화하는 질문을 생성합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 질문 선택 방법은 질문 공간 내에서만 추론하는 기존 접근 방식에 비해 더 효과적인 작업 불명확성 해소를 보여줍니다. 이는 LLM이 솔루션 공간에 대한 암묵적 추론(implicit reasoning)에서 명시적 추론(explicit reasoning)으로 전환하도록 하여 LLM의 문제 해결 능력을 향상시킵니다.



### ADIFF: Explaining audio difference using natural languag (https://arxiv.org/abs/2502.04476)
Comments:
          Accepted at ICLR 2025. Dataset and checkpoints are available at: this https URL

- **What's New**: 이 논문은 오디오 간의 차이를 설명하는 작업을 포괄적으로 연구하고, 이를 위한 벤치마크 및 기초선(baseline)을 제안한 최초의 연구로 두 개의 새로운 데이터셋(ACD 및 CLD)을 소개합니다. 제안된 ADIFF 모델은 교차 투영(cross-projection) 모듈과 위치 캡셔닝(position captioning)을 통해 모델이 차이를 보다 자세하게 설명할 수 있도록 향상시켰습니다. 이는 오디오 텍스트 모델의 평가 및 오디오 정보와 세계 지식을 통합하는 방법을 탐색하는 효과적인 벤치마크로 작용합니다.

- **Technical Details**: 제안된 프레임워크는 세 단계의 교육 과정을 포함하고, 각 데이터셋은 세 가지 수준의 설명을 제공합니다: 간결한 설명(tier-1), 짧은 문장(tier-2), 그리고 포괄적인 설명(tier-3)입니다. ADIFF 모델은 separator token을 사용하고, 오디오 파일의 임베딩을 동결된 언어 모델에 사용하는 prefix-tuning 기법을 사용하여 기본 모델의 한계를 극복하도록 설계되었습니다. 연구에서는 교차 투영, 언어 모델 파라미터 조정, 위치 캡셔닝이 모델 성능에 미치는 영향을 다룬 여러 가지 ablation 연구를 수행하였습니다.

- **Performance Highlights**: 신경망의 강화 덕분에 ADIFF 모델은 단순 기초선 및 최신 Audio-Language Model(SoTA ALM)인 Qwen Audio에 비해 상당한 성능 향상을 보여주었습니다. 이를 통해 모델은 청취자가 느끼는 감정 및 세부 속성을 포함한 더 복잡하고 풍부한 설명을 생성할 수 있음을 입증했습니다. 실험 결과는 ADIFF 모델이 다양한 오디오 파일 간의 미세한 차이와 유사성을 효과적으로 식별하고 기술할 수 있는 능력을 가지고 있음을 보여줍니다.



### Augmented Conditioning Is Enough For Effective Training Image Generation (https://arxiv.org/abs/2502.04475)
- **What's New**: 본 논문은 생성된 이미지의 다양성을 향상시켜 다운스트림 이미지 분류 모델을 효과적으로 훈련할 수 있는 방법을 조사합니다. 텍스트 프롬프트와 진짜 이미지를 조건으로 사용하는 방식을 통해, 높은 품질과 다양성을 지닌 합성 이미지를 생성하는 기법을 제안합니다. 기존의 방법들이 이미지 생성 모델을 세밀하게 조정하는 것과는 달리, 본 연구는 훈련 데이터로 사용할 수 있는 합성 데이터셋을 더욱 효율적으로 생성하는 것을 목표로 합니다.

- **Technical Details**: 저자들은 고전적인 비전 데이터 증강(Data Augmentation) 기법을 이용하여 영상 생성의 조건 정보로 활용하는 방법을 분석하였습니다. 특히, 실 데이터에 대해 증강한 이미지를 조건화함으로써 생성 과정에서 시각적 다양성을 이끌어내고, 이는 다운스트림 분류 성능을 높이는 데 기여합니다. 연구진은 사전 훈련된 확산 모델을 활용하여, 증강 조건화를 통해 새로운 훈련 이미지를 생성하였습니다.

- **Performance Highlights**: 연구 결과는 합성 데이터셋이 기존의 방법 대비 향상된 분류 성능을 가지고 있음을 보여줍니다. 특히, ImageNet Long-Tailed 벤치마크에서 우수한 성능을 달성하였고, 극단적인 적은 샷(few-shot) 상황에서도 뛰어난 성과를 보였습니다. 이러한 결과는 모델의 재조정 없이 합성 데이터를 효과적으로 활용하여 모델 훈련을 진행할 수 있는 잠재력을 나타냅니다.



### Color in Visual-Language Models: CLIP deficiencies (https://arxiv.org/abs/2502.04470)
Comments:
          6 pages, 10 figures, conference, Artificial Intelligence

- **What's New**: 이 논문은 현재 인공지능에서 가장 영향력 있는 시각 언어 모델(VML)인 CLIP(Contrastive Language-Image Pre-training)의 색상 인코딩 방식을 탐구합니다. 저자들은 CLIP이 적절한 색상 레이블을 색칠된 시각 자극에 부여할 수 있지만, 두 가지 주요 결점이 있음을 발견했습니다: (a) 무채색 자극에 편향이 있어 흰색, 회색, 검정색이 색상 레이블로 잘 지정되지 않으며, (b) 다른 시각 정보보다 텍스트를 우선시하는 경향이 있습니다.

- **Technical Details**: 연구자들은 신경 수준에서 내부 표현을 분석하여 CLIP의 색상 인식 결함의 원인을 찾기에 집중했습니다. 분석 결과, CLIP의 깊은 층에서 텍스트에 선택적인 신경 세포가 과다하게 발달해 있고, 색상 인식을 위한 멀티모달 색상 신경 세포는 적은 수가 있는 것으로 나타났습니다. 이러한 발견은 인간의 색상 이해와 일치하는 신경 세포의 발달을 더 잘 이해하는 데 중요합니다.

- **Performance Highlights**: CLIP은 색상 레이블을 기본 이미지에 연결하는 데 여러 가지 실험을 통해 성능을 평가했으며, 그 결과 무채색 부분에서 색상 값을 잘 예측하지 못하고 있음을 보여주었습니다. 특히, Stroop 테스트를 통해 CLIP이 시각적 정보보다 텍스트 정보에 더 집중한다는 것을 확인했습니다. 이는 CLIP이 색상 개념을 이해하는 방식에서 개선이 필요함을 시사합니다.



### No Images, No Problem: Retaining Knowledge in Continual VQA with Questions-Only Memory (https://arxiv.org/abs/2502.04469)
Comments:
          8 pages, in-review

- **What's New**: 이번 연구에서는 Visual Question Answering Continual Learning (VQACL)을 위한 새로운 접근 방식인 QUestion-only replay with Attention Distillation (QUAD)를 제안합니다. QUAD는 과거 질문만을 저장하고, 시각적 데이터를 저장할 필요 없이 메모리와 프라이버시 문제를 해결합니다. 또한, 질문 반복 메커니즘을 도입하여 현재 작업의 답변 공간에 과적합(Overfitting)되는 것을 방지하며, 본질적인 시각-언어 연관성을 유지합니다.

- **Technical Details**: QUAD는 질문만을 활용하여 현재 모델의 정규화를 수행하는 'Question-only Replay' 메커니즘을 특징으로 하며, 이 과정에서 출처-응답-세트 문제를 해결합니다. 두 번째 기여는 'attention consistency distillation'으로, 이는 다양한 작업 간에 주의 패턴을 일관되게 유지하도록 돕습니다. 이러한 기법은 특정 질문에 대해 언어-비언어 간의 관심 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 QUAD는 VQAv2 및 NExT-QA 데이터 세트에서 기존의 최첨단 방법들보다 뛰어난 성능을 보였습니다. QUAD는 메모리 없는 접근 방식과 전통적인 리허설 방법보다 뛰어난 결과를 보여주며, 질문만 저장해도 잊혀짐 문제를 완화할 수 있음을 입증했습니다.



### FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks (https://arxiv.org/abs/2502.04465)
Comments:
          18 pages

- **What's New**: 이번 연구에서는 FocalCodec라는 효율적인 저비트레이트 코덱을 소개합니다. 이 코덱은 균일한 이진 코드북을 활용하여 음성을 0.16에서 0.65 kbps로 압축합니다. FocalCodec는 기존의 최첨단 기술보다 낮은 비트레이트에서 경쟁력 있는 성능을 제공하며, 다국어 음성과 소음 환경에서도 효과적으로 처리할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: FocalCodec는 VQ-VAE 프레임워크를 기반으로 하며, 인코더와 디코더 사이에 새로운 압축기(compressor) 및 해제기(decompressor) 모듈을 통합합니다. 이 구조는 명시적인 분리(disentanglement)나 복잡한 다중 인코더를 사용하지 않고도 음향(acoustic) 및 의미(semantic) 정보를 모두 포착할 수 있도록 설계되었습니다. Focal modulation을 이용한 이 아키텍처는 음성 데이터의 효율적이고 확장 가능한 토큰화를 제공합니다.

- **Performance Highlights**: FocalCodec는 다양한 조건에서도 낮은 비트레이트 속에서 높은 재구성 품질을 유지하며, 다운스트림 작업에서도 충분한 의미 및 음향 정보를 보존합니다. 포괄적인 평가를 통해 FocalCodec는 판별(discriminative) 및 생성(generative) 음성 모델링 모두에 유망한 잠재성을 보여주었습니다. 데모 샘플, 코드는 연구 페이지에서 볼 수 있으며, FocalCodec의 구현과 예시를 통해 실제 적용 가능성을 검토할 수 있습니다.



### Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization (https://arxiv.org/abs/2502.04428)
- **What's New**: 이 논문은 불확실성 기반의 SLM 라우팅을 통해 소형 언어 모델(SLM)과 대형 언어 모델(LLM)의 통합을 모색합니다. SLM이 낮은 신뢰도로 인한 복잡한 쿼리에 부정확한 응답을 생성할 경우 LLM으로 쿼리를 이관하는 시스템을 제안합니다. 이를 통해 사용자는 더 높은 신뢰성을 확보하고 비용 효율성을 유지할 수 있습니다.

- **Technical Details**: 이 연구는 1500개 이상의 설정에서 SLM과 LLM 간의 불확실성 기반 라우팅 전략의 벤치마킹과 일반화를 수행하였습니다. 다양한 불확실성 정량화(UQ) 방법이 라우팅 성능에 미치는 영향과 맞춤 데이터를 활용한 일반화 전략을 고찰합니다. 결과적으로, 불확실성 정량화 방법의 정확성 조정이 라우팅 성능에 중요한 역할을 함을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 생성된 보정 데이터(calibration data)가 SLM과 LLM 간의 라우팅 성능을 향상시키며, 신규 데이터 없이도 효과적인 라우팅 결정을 가능하게 합니다. 또한, 불확실성 기반 라우팅 접근법은 다양한 데이터셋에 잘 일반화되며, 라우팅 전략의 효율성을 높이는데 기여함을 보여주었습니다.



### Decoding AI Judgment: How LLMs Assess News Credibility and Bias (https://arxiv.org/abs/2502.04426)
- **What's New**: 본 연구는 최신의 대형 언어 모델(Large Language Models, LLMs)인 Gemini 1.5 Flash, GPT-4o mini, 그리고 LLaMA 3.1이 신뢰성을 평가하는 방식을 분석합니다. 그 과정에서 LLM이 인간 전문가의 평가 체계를 어떻게 반영하거나 다르게 나타내는지를 탐구하며, '신뢰성'이라는 개념을 구성하는 언어적 단서들을 밝혀냅니다. 또한, LLM이 외부 정보를 검색하고 다른 모델과 상호작용하며 평가를 정교화하는 과정을 통해, 이러한 모델들이 직관적 사고 아니면 이전의 학습된 연관성에 의존하는지를 조사합니다.

- **Technical Details**: 이 연구에서는 총 2,302개의 뉴스 출처를 대상으로 LLM의 신뢰성과 정치적 분류를 평가합니다. 뉴스 출처는 NewsGuard와 Media Bias Fact Check의 구조화된 평가를 바탕으로 구성되었으며, 다양한 언어적 단서와 맥락적 단서를 통해 LLM의 평가 프로세스를 세밀하게 분석합니다. 모델들은 제로샷 접근 방식을 통해 평가 질문에 답하며, 각 뉴스 출처에 정치적 방향성을 부여하고 그렇게 이끌어낸 평가에 대한 설명을 생성합니다.

- **Performance Highlights**: 각 LLM 모형은 NewsGuard가 정한 신뢰성 등급과 비교하여 유의미한 결과를 보여줍니다. 특히, 모든 모델이 ‘신뢰할 수 없는’ 출처를 높게 식별한 반면, ‘신뢰할 수 있는’ 출처의 분류는 상대적으로 어려움을 보여주었습니다. LLM의 평가 결과는 NewsGuard와 Media Bias Fact Check(MBFC)의 기준과 일반적으로 유사하지만, 고란 위험 신뢰 평가 소스의 경우 다소간의 차이가 발생했습니다.



### EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models (https://arxiv.org/abs/2502.04424)
- **What's New**: 이번 논문에서는 Multimodal large language models (MLLMs)를 로봇 시스템과 AI 애플리케이션에 통합하면서 감정 지능 (Emotional Intelligence, EI) 기능을 중요한 요소로 제시합니다. 기존의 정적이고 텍스트 기반 또는 텍스트-이미지 벤치마크는 실생활의 복잡한 다중 양식 상호작용을 간과하고, 감정 표현의 동적인 특성을 포착하지 못해 MLLMs의 EI를 평가하는 데 적절하지 않음을 지적합니다. 이와 같은 점을 보완하기 위해, EmoBench-M이라는 새로운 벤치마크를 개발하였습니다.

- **Technical Details**: EmoBench-M은 감정 지능 평가를 위한 새로운 벤치마크로, 13개의 평가 시나리오를 세 가지 주요 차원인 기초 감정 인식 (foundational emotion recognition), 대화 감정 이해 (conversational emotion understanding), 사회적 복잡 감정 분석 (socially complex emotion analysis)으로 나누어 구성하였습니다. 이 벤치마크는 감정 지능의 다양한 측면을 포괄적으로 평가할 수 있도록 설계되었습니다. 모든 평가 리소스는 공개적으로 제공되며, 코드와 데이터셋에 대한 접근이 가능합니다.

- **Performance Highlights**: EmoBench-M를 이용한 평가에서, 오픈 소스 및 클로즈드 소스 MLLMs는 인간에 비해 상당한 성능 차이를 보였으며, 이는 MLLMs의 EI 기능을 더욱 발전시킬 필요성을 강조합니다. 이러한 결과는 MLLMs가 인간의 감정적 요구를 효과적으로 다루기 위한 기술적 도전이 여전히 존재함을 나타냅니다. 감정 지능의 향상은 로봇과 AI의 실용적인 활용에 크게 기여할 것입니다.



### Primary Care Diagnoses as a Reliable Predictor for Orthopedic Surgical Interventions (https://arxiv.org/abs/2502.04423)
- **What's New**: 이번 연구는 기본 진료에서의 진단 정보를 기반으로 수술 필요성을 예측할 수 있는 가능성을 조사하였습니다. 이를 통해 레퍼럴(referral) 정확성을 높이고, 워크플로우(workflow)를 간소화하며, 환자에게 더 나은 치료를 제공할 수 있습니다. 연구에 사용된 데이터셋은 텍사스 대학교 타일러 건강 센터에서의 2086건의 정형외과 레퍼럴로, 기계 학습 모델을 통해 분석되었습니다.

- **Technical Details**: 기계 학습 모델은 Base General Embeddings (BGE)를 사용하여 의미적 추출을 수행하였고, 실제 적용 가능성을 높이기 위해 잡음 허용성 실험을 실시하였습니다. 또한 클래스 불균형(class imbalance) 문제를 해결하기 위해 오버샘플링(over-sampling) 기법을 적용했습니다. 최적화된 임베딩 모델은 예측 정확도(ROC-AUC: 0.874, Matthews Correlation Coefficient (MCC): 0.540)를 보여주며, 수술 개입이 필요한 환자를 효과적으로 구분할 수 있었습니다.

- **Performance Highlights**: 예측 모델링 분석을 통해 수술 요율은 11.27%에서 최적의 60.1%로 증가하였고, 이는 433%의 개선을 나타냅니다. 이러한 결과는 운영 효율성과 의료 수익에 대한 중요한 시사점을 제공합니다. 연구 결과는 레퍼럴 최적화가 기본 치료와 수술 치료 통합을 향상시킬 수 있음을 보여주며, 환자 요구 사항의 정확하고 적시 예측을 통해 지연 시간을 최소화하고, 수술 계획을 개선하며, 행정 부담을 줄일 수 있음을 강조합니다.



### Assessing and Prioritizing Ransomware Risk Based on Historical Victim Data (https://arxiv.org/abs/2502.04421)
- **What's New**: 이 연구에서는 특정 단체를 대상으로 가장 잘 알려진 랜섬웨어 공격자가 누구인지 식별하는 접근 방식을 제안하여, 단체들이 보다 나은 보호 전략을 수립할 수 있도록 돕습니다. 랜섬웨어는 수익 추구 동기에 의해 발생하는 사이버 보안 위협으로, 그 공격의 공공성이 피해자에게 몸값을 지불하도록 압박하는 주된 전술로 사용됩니다. 본 논문에서는 랜섬웨어 피해자들로부터의 공개된 정보를 활용하여 특정 랜섬웨어 변종에 의해 목표로 삼을 가능성을 예측합니다.

- **Technical Details**: 본 연구에서는 고유한 사고의 연쇄, 다중 과정 프롬프트 방법론을 사용하는 대형 언어 모델(LLM) 아키텍처를 활용하여 랜섬웨어 흐름에서 적대자의 SKRAM(Skills, Knowledge, Resources, Authorities, Motivation) 프로필을 정의합니다. 연구는 공개적으로 사용할 수 있는 피해자 데이터로 보강되며, 피해자 프로필을 반영하는 합성 데이터를 생성하기 위한 휴리스틱(heuristic) 방법으로 향상됩니다. 최종적으로, 본 연구는 조직이 랜섬웨어 위협을 우선순위를 매기고, 가장 가능성이 높은 공격자의 전술, 기술, 절차에 기반하여 방어를 수립할 수 있는 머신 러닝 모델을 개발합니다.

- **Performance Highlights**: 각종 사이버 위협에 대한 정보 수집 및 처리의 비효율성을 극복하기 위해 본 연구는 머신 러닝 기술을 활용하여 다양하고 방대한 랜섬웨어 위협을 간소화한 목록으로 정제합니다. 연구 결과에서는 개별 공격자를 판별하고, 각 위협에 대하여 위험 점수를 생성하여 적용하는 과정이 강조되며, 이는 조직이 가장 관련성 높은 위협을 우선적으로 관리할 수 있도록 돕습니다. 본 연구의 접근 방식은 랜섬웨어 공격의 동향을 반영하여, 데이터의 시간적 특성을 모델에 통합해 지속적으로 업데이트하는 중요성을 강조합니다.



### KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inferenc (https://arxiv.org/abs/2502.04420)
- **What's New**: 본 논문에서는 KV cache quantization의 개선된 접근 방식을 제안합니다. 주목할 만한 점은 레이어별(transformer layer-wise) 민감성을 고려하여 KV cache의 양자화 오류를 최소화하는 방법론입니다. 저자들은 키(cache key)가 값(cache value)보다 더 중요하다는 사실을 강조하며, 이를 통해 더 효과적인 양자화가 가능하다고 주장합니다.

- **Technical Details**: 제안된 KVTuner 프레임워크는 하드웨어에 적합한 레이어별 KV 양자화 정밀도를 탐색합니다. 다중 목표 최적화(multi-objective optimization) 기법을 사용하여 KV cache의 효율적인 조합을 찾고, 온라인 추론 동안 사전에 검색된 구성을 직접 활용합니다. 또한, 계산 비용을 줄이기 위해 intra-layer KV 정밀도 쌍 pruning과 inter-layer clustering 기법을 도입하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, Llama-3.1-8B-Instruct 모델의 경우 거의 손실 없는 3.25비트 혼합 정밀도 KV cache 양자화를 가능하게 하였으며, Qwen2.5-7B-Instruct와 같은 민감한 모델의 경우 4.0비트 양자화가 이루어졌습니다. 다양한 컨텍스트 길이에 대해 KV8 양자화와 비교할 때 최대 추론(thoroughput) 성능이 38.3% 개선되었습니다.



### Understanding and Mitigating the Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks (https://arxiv.org/abs/2502.04419)
Comments:
          Technical report; 31 pages

- **What's New**: 최근 연구는 대형 언어 모델(LLMs)의 성능 향상을 위한 합성 데이터 생성의 필요성을 강조하고 있습니다. 그러나 이러한 모델들이 학습 데이터에서 반영한 편향이 합성 데이터에 전이되고 증폭될 수 있는 문제, 즉 '편향 계승(bias inheritance)'에 대한 체계적인 연구가 부족했습니다. 본 논문은 이러한 문제를 최초로 규명하고, 편향 계승의 이해와 분석, 완화 방안을 제시합니다.

- **Technical Details**: 연구는 원본 데이터와 LLM으로 증강된 데이터를 결합한 데이터셋으로 LLMs를 미세 조정하여 편향 비율에 따른 6가지 편향 유형의 영향을 분석했습니다. 실험을 통해 특정 편향이 드러나는 비율에 따라 자동 생성하고 실험을 진행함으로써 편향이 직접 관련된 분류 작업과 일반 생성 작업에 미치는 영향의 차이를 규명했습니다. 이 분석은 사회적 편향이 증강 데이터에 미치는 영향과 그로 인한 하향 성능 저하를 포함합니다.

- **Performance Highlights**: 실험 결과, 직접 관련된 과제에서 편향의 발생과 그로 인한 성능 저하는 특히 소수 그룹에서 두드러지며, 편향이 반복적 조정을 통해 확대됩니다. 또한 본 연구는 가치 불일치, 그룹 데이터 불일치, 데이터 분포 불일치의 세 가지 주요 요소를 파악하였고, 이를 해결하기 위한 세 가지 완화 전략(token-based, mask-based, loss-based)을 제안했습니다. 이러한 접근 방식은 다양한 작업과 편향에서 다르게 작용하여 편향 계승 문제 해결의 어려움을 강조합니다.



### Autotelic Reinforcement Learning: Exploring Intrinsic Motivations for Skill Acquisition in Open-Ended Environments (https://arxiv.org/abs/2502.04418)
Comments:
          12 pages, 12 figures

- **What's New**: 이 논문은 자율적(Reinforcement Learning) 강화 학습의 포괄적인 개요를 제공하며, 내재적 동기의 역할을 강조합니다. 특히, 기술 능력을 개발하는 데 있어서 지식 기반 내재적 동기와 역량 기반 동기의 차이를 명확히 합니다. 이는 자율 에이전트가 자기 정의 목표를 생성하고 추구할 수 있도록 돕는 방법을 설명합니다.

- **Technical Details**: 내재적으로 동기가 부여된 목표 탐색 과정(IMGEP) 유형을 탐구하며, 이는 다목적 강화 학습(multi-goal RL)과 개발 로봇 공학에 미치는 영향에 초점을 맞춥니다. 에이전트는 보상 없는 마르코프 결정 프로세스(MDP) 내에서 자율적으로 자신의 목표를 표현, 생성 및 마스터해야 합니다. 이러한 과정은 복잡한 환경에서 탐색, 일반화 및 강건성을 측정하기 위한 여러 지표를 제안합니다.

- **Performance Highlights**: 이 연구는 서로 다른 환경에서 기술 습득을 향상시킬 수 있는 자율적 강화 학습 에이전트의 이해를 발전하는 것을 목표로 합니다. 또한, 이러한 에이전트의 평가에서 발생하는 독특한 문제점을 다루며, 이들의 성능을 측정하기 위한 다양한 평가 기준을 제공합니다.



### NeuralMOVES: A lightweight and microscopic vehicle emission estimation model based on reverse engineering and surrogate learning (https://arxiv.org/abs/2502.04417)
- **What's New**: NeuralMOVES는 차량 CO2 배출을 위한 경량화된 대체 모델로, 기존의 MOVES 모델이 가진 복잡성과 높은 연산 요구사항을 극복한 혁신적인 접근 방식입니다. 이 모델은 데이터 역설계를 통해 수집된 방대한 시나리오 정보를 바탕으로 개발되었으며, 높은 정확성을 유지하면서도 사용의 용이성을 제공합니다. 물론, NeuralMOVES는 MOVES의 기능을 강화하고 다양한 응용 분야에서의 실시간 처리를 가능하게 합니다.

- **Technical Details**: NeuralMOVES는 2.4 MB의 크기를 가지며, 6.013%의 평균 비율 오류(Mean Average Percentage Error)를 기록하여 기존 MOVES 모델과 비교했을 때 우수한 성능을 나타냅니다. 이는 두 백만 개가 넘는 다양한 환경과 경로에 대한 테스트를 통해 입증되었습니다. 또한, 보조 학습(surrogate learning) 기법을 사용하여 복잡한 데이터를 경량화하여 처리 속도를 높이는 데 초점을 맞추었습니다.

- **Performance Highlights**: NeuralMOVES의 개발로 인해 사용자는 CO2 배출량 평가를 자동화하고 프로그램적으로 처리할 수 있는 경량화된 대안을 얻게 되었습니다. 이 모델은 추후 차량 수준의 제어 및 최적화 등 다양한 마이크로스코픽 응용 분야에 활용될 수 있을 것으로 기대됩니다. 연구진은 NeuralMOVES의 효용성을 검증하기 위한 사례로서 신호화된 교차로에서의 에코 드라이빙 전략 최적화를 제시합니다.



### CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inferenc (https://arxiv.org/abs/2502.04416)
- **What's New**: 본 논문에서는 CMoE(Carved MoE)라는 새로운 프레임워크를 제안하여, 밀집 모델(다수의 파라미터가 활성화된 모델)로부터 효율적으로 MoE(mixture-of-experts)를 분리할 수 있는 방법을 제시하고 있습니다. 기존의 MoE 모델이 대량의 데이터와 자원을 요구했던 반면, CMoE는 적은 데이터로도 성능을 유지하며 신속한 구조적 적응이 가능하다는 점에서 혁신적입니다.

- **Technical Details**: CMoE는 뉴런을 활성화 비율에 따라 공유 전문가(shared experts)와 라우팅 전문가(routed experts)로 그룹화합니다. 또한, 라우팅 메커니즘을 훈련 없이 초기화하여, 각 전문가 클러스터의 대리 뉴런을 활용해 간편하게 사용할 수 있도록 하였습니다. 이 과정에서는 Jonker-Volgenant 알고리즘을 통해 균형 잡힌 선형 할당 문제로 전문가 그룹을 구성합니다.

- **Performance Highlights**: 실험 결과 CMoE는 25% 활성화 비율에서 조정 없이도 합리적인 perplexity를 유지하며, 일부 다운스트림 벤치마크에서는 밀집 모델의 76.59% 정확도를 달성했습니다. 또한, 2,048 샘플을 대상으로 경량 미세 조정을 통해 높은 성능 회복을 이뤄내는 성과를 거두었습니다.



### TerraQ: Spatiotemporal Question-Answering on Satellite Image Archives (https://arxiv.org/abs/2502.04415)
- **What's New**: 이 논문에서는 TerraQ라는 새로운 spatiotemporal QA 엔진을 소개합니다. 이 엔진은 위성 이미지 아카이브에 대한 사용자 요청을 자연어로 처리하여 쉽게 접근할 수 있도록 합니다. TerraQ는 기존의 템플릿 기반 질의 생성 방식을 제거하고, 고품질의 지리정보를 포함한 목적 맞춤형 지식 그래프(Knowledge Graph)를 활용하여 다양한 질문에 답변할 수 있는 이점을 제공합니다.

- **Technical Details**: TerraQ는 여러 컴포넌트로 구성되어 있으며, 각 컴포넌트는 특정 작업을 수행합니다. 시스템의 체계적 접근 방식은 질문을 받고, 이를 분해하여 자동으로 SPARQL 쿼리를 생성하는 프로세스를 포함합니다. 이 엔진은 Sentinel-1 및 Sentinel-2와 같은 다양한 위성 이미지 데이터 세트, 관리 영역(GADM) 및 특정 지리적 개체를 결합하여 사용합니다.

- **Performance Highlights**: TerraQ는 간단하고 복잡한 질문에 대해 신뢰성과 빠른 속도로 응답할 수 있습니다. 자동화된 처리 시스템 덕분에 비전문가와 전문가 모두에게 직관적으로 접근 가능하도록 설계되어 있으며, Earth Observation 데이터 아카이브의 접근성을 크게 향상시킵니다. 실제로 유럽 우주국의 프로젝트를 바탕으로 개발된 TerraQ는 전 세계적으로 사용할 수 있는 데모를 제공하고 있습니다.



### MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilo (https://arxiv.org/abs/2502.04413)
- **What's New**: 이 논문에서는 의료 분야에서 지식 그래프(KG)를 활용하여 진단 및 치료 권장을 제공하는 MedRAG라는 새로운 Retrieval-augmented generation (RAG) 모델을 제안합니다. MedRAG는 비슷한 증상을 가진 질병들 간의 진단 차이를 체계적으로 구성한 4단계 계층적 진단 KG를 구축하여, 기존 모델의 다소 부족한 진단 정확성과 특이성을 개선합니다. 이 모델은 환자의 증상에 기반하여 더 정확하고 구체적인 의료 의사결정을 지원함으로써, 잘못된 진단의 위험을 줄이는 데 기여합니다.

- **Technical Details**: MedRAG는 진단 지식 그래프와 RAG의 통합을 통해, 환자 정보를 보다 명확하게 이해하고 그에 따른 후속 질문을 제시하는 기능을 갖추고 있습니다. 새로운 진단 KG 검색 모듈을 통해 입력된 환자와 관련된 모든 중요한 진단 차이를 식별하고, 대규모 언어 모델 내에서 이 정보를 결합하여 추론을 수행합니다. 이 과정은 증상들이 유사한 질병 간의 미세한 진단 차이를 구별할 수 있는 개선된 추론 능력을 제공합니다.

- **Performance Highlights**: MedRAG는 DDXPlus 데이터셋과 Tan Tock Seng 병원에서 수집된 CPDD 데이터셋을 통해 평가되었으며, 여러 최신 RAG 모델과 비교하여 잘못된 진단 비율을 줄이는 데 있어 우수한 성능을 보였습니다. 실험 결과에 따르면, MedRAG는 기존의 RAG 접근 방식들보다 높은 진단 정확성과 특이성을 제공하며, 다양한 LLMs에서 robust한 일반화 성능을 나타냈습니다. 또한, 이 모델은 진단 질문을 보다 효과적으로 생성하여 복잡한 의료 시나리오에서 의사결정 과정을 최적화하는 데 큰 기여를 할 수 있습니다.



### Decoder-Only LLMs are Better Controllers for Diffusion Models (https://arxiv.org/abs/2502.04412)
- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 효율성을 향상시키기 위해 대형 언어 모델(Large Language Models, LLMs)의 의미 이해 능력을 결합한 새로운 방법을 제안합니다. 특히, 이 모델은 LLM의 디코더 전용 구조를 활용하여 텍스트 프롬프트의 의미를 더 잘 캡처할 수 있도록 설계된 LLMDiff-Adapter라는 네트워크 모듈을 도입합니다. 기존의 방법들이 텍스트 인코더에 의존해 왔다면, 우리의 접근법은 LLM의 블록별 표현을 통합하여 텍스트 인코딩을 생성합니다.

- **Technical Details**: LLMDiff-Adapter는 노이즈 제거 U-Net 구조의 크로스-어텐션 부분에 연결됩니다. 이를 통해 LLM에서 추출한 표현을 텍스트 인코딩 생성에 직접적으로 활용할 수 있으며, 이는 세밀한 의미와 단어 간의 맥락 의존성을 효과적으로 포착합니다. 이 방법은 텍스트-이미지 생성 모델에 LLM의 강점을 직접적으로 통합하는 모듈로서 작동하며, 다양한 생성 모델에 쉽게 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLMDiff-Adapter를 사용한 모델은 생성된 이미지의 품질, 논리적 일관성 및 텍스트 설명에 대한 포괄적 이해 측면에서 최첨단 모델을 초월하는 결과를 보였습니다. 세밀한 이미지 세부 사항과 사용자 의도를 잘 반영한 이미지 생성을 통해 이 모델은 텍스트-이미지 생성 품질을 크게 향상시켰습니다. 다양한 벤치마크에서의 비교 분석을 통해 LLMDiff-Adapter의 효과성을 입증하였습니다.



### Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing (https://arxiv.org/abs/2502.04411)
Comments:
          work in progress. arXiv admin note: text overlap with arXiv:2405.09673 by other authors

- **What's New**: 이번 연구는 다양한 작업에 맞게 파인튜닝된 Large Language Models (LLMs)를 더욱 강력한 모델로 통합하는 모델 머징을 다룹니다. 기존의 모델 평균화에서 발생하는 파라미터 충돌로 인한 성능 저하 문제를 해결하기 위해, 레이어 간 충돌 정도를 분석했습니다. 이를 바탕으로 최소한의 파라미터 충돌을 가진 레이어를 평균화하고, 중요한 충돌을 가진 레이어는 새로운 전문가 라우팅 방법을 통해 처리하는 접근 방식을 제안했습니다.

- **Technical Details**: 모델 머징(Mediator) 프레임워크는 파라미터 충돌의 정도를 정량화하며 이를 반영해 레이어를 조정합니다. 파라미터 충돌이 적은 레이어는 평균화하고, 상당한 충돌을 가진 레이어는 전문가 모델로서 라우팅하여 고유한 작업 지식을 보존하는 전략을 사용합니다. 또한, 작업 산술의 희소성을 활용해 여러 파인튜닝 전문가를 밀집 전문가와 희소 전문가로 분리하여 저장 공간을 절약합니다.

- **Performance Highlights**: 이 방법을 LLaMA와 Qwen 모델을 사용하여 다양한 파라미터 스케일에서 실험한 결과, 기존 방법과 비교했을 때 성능 개선이 두드러졌습니다. 우리의 통합 모델은 RTX 4090 GPU에서 7B × 4 LLM 앙상블과 유사한 성능을 보여주며, 리소스 제한이 있는 환경에서도 높은 성능을 발휘할 수 있음을 입증했습니다. 따라서 실세계의 추론 작업에서 더 적은 시스템 비용으로 뛰어난 성능을 달성할 수 있음을 확인했습니다.



### Transforming Multimodal Models into Action Models for Radiotherapy (https://arxiv.org/abs/2502.04408)
- **What's New**: 이 연구에서는 방사선 치료 계획을 개선하기 위해 멀티모달 기초 모델(MLM)을 행동 모델로 전환하는 새로운 프레임워크를 제안합니다. 몇 번의 샷 강화 학습(few-shot reinforcement learning) 접근 방식을 사용하여 MLM의 방대한 지식을 활용합니다. 이를 통해 방사선 치료 계획의 품질과 효율성을 높이며, 전통적인 RL 기반 접근 방식보다 더 나은 성과를 보입니다.

- **Technical Details**: 이 프레임워크는 가상 화상 시뮬레이터와 통합되어 방사선 치료 계획을 반복적으로 개선하는 기능을 갖추고 있습니다. 연구에서는 MatRAD를 기반으로 하는 오픈소스 환경을 구축하여, 최대 5개의 겐트리 각도를 입력으로 받아 3D 이미지 상태를 출력합니다. 보상 함수는 Planned Target Volume(PTV) 내의 용적 균질성을 달성하는 보상과 Organs at Risk(OAR)에 대한 과도한 용량에 대한 페널티를 설정하여, 치료 계획의 안전성과 효율성을 최적화하도록 유도합니다.

- **Performance Highlights**: 이 연구의 결과는 기존의 RL 기반 접근 방식에 비해 치료 계획의 질과 효율성이 크게 향상되었음을 보여줍니다. 특히, 전립선 암 데이터를 사용한 시뮬레이션에서 더 높은 보상 점수 및 최적의 용량 분포를 달성했습니다. 이는 발전된 AI 모델이 임상 작업 흐름에 통합될 가능성을 시사하는 중요한 증거입니다.



### Illuminating Spaces: Deep Reinforcement Learning and Laser-Wall Partitioning for Architectural Layout Generation (https://arxiv.org/abs/2502.04407)
- **What's New**: 이번 논문에서는 공간 레이아웃 설계(SLD)에서 레이저-wall이라는 새로운 공간 분할 방법을 도입합니다. 이 방법은 공간을 나누는 데 있어 벡터 기반과 픽셀 기반 접근 방식을 결합하여 설계의 직관성을 높입니다.

- **Technical Details**: SLD는 공간 내 요소들을 전략적으로 배치하는 복잡한 최적화 문제로, 본 논문에서는 RL(강화 학습)을 computational approach로 사용하여 레이저-wall을 통한 공간 분할의 효율성을 분석합니다. 레이저-wall은 빛의 발산 개념을 도입하여 공간 내의 벽과 상호작용함으로써 유연한 분할을 제공합니다.

- **Performance Highlights**: RL 기반 레이저-wall 접근 방식을 활용하여 다양한 공간 레이아웃을 생성하고, 기하학적 제약 조건과 위상적 요구를 모두 만족하는 기능성 레이아웃을 제공할 수 있음을 실증하였습니다. 실험 결과, 이 방법은 아키텍처적으로 직관적인 디자인 솔루션을 가능하게 합니다.



### Calibrated Physics-Informed Uncertainty Quantification (https://arxiv.org/abs/2502.04406)
- **What's New**: 이 논문에서는 계산적으로 고비용인 수치 PDE 솔버(numercial PDE solvers) 대신에 신경망 PDE(neural PDE)를 이용한 새로운 확률적 예측 모델, 즉 물리 기반 합정적 예측(Conformal Prediction, CP) 프레임워크를 제안합니다. 이 프레임워크는 레이블이 없는 데이터에 의존하지 않고도 불확실성 추정치를 보장합니다. 물리적 억제법을 활용하여, 우리의 접근 방법은 모델의 불일치를 정량화하고 보정할 수 있습니다.

- **Technical Details**: 제안한 방법은 신경망 PDE 솔버의 예측에서 Physics Residual Errors (PRE)를 평가하고, 이는 마진(marginal) 및 조합적(joint) CP 형식을 통해 보정됩니다. 이 방법은 물리 보존 법칙 위반에 기반한 오차 경계를 제공하며, 기존 UQ 방법의 통계적 보장을 제공합니다. 프레임워크는 모델에 구애받지 않으며 추가 데이터 없이 작동하여 신뢰할 수 있는 예측을 가능하게 합니다.

- **Performance Highlights**: 이 방법은 플라즈마 모델링(plasma modeling)과 핵융합 반응기(shot design in fusion reactors)와 같은 복잡한 PDE를 적용한 신경망 PDE 모델에서 타당성을 검증했습니다. CP의 마진 및 조합적 보장을 통해 변동성이 큰 예측을 식별할 수 있는 기준을 제시합니다. 따라서, 제안한 모델은 물리적인 불일치 문제를 해결하며 신경망 PDE 솔버의 (과도한) 신뢰도를 개선합니다.



### FAS: Fast ANN-SNN Conversion for Spiking Large Language Models (https://arxiv.org/abs/2502.04405)
- **What's New**: 이번 연구에서는 Spiking Large Language Models(스파이킹 대형 언어 모델)가 기존의 LLM(대형 언어 모델)들에게 훌륭한 대안이 될 수 있음을 보여주고 있습니다. 기존의 Spiking LLMs 생성 방법들은 성능 저하와 높은 계산 비용에 시달렸습니다. 이를 해결하기 위해 우리는 새로운 Fast ANN-SNN conversion strategy(FAS)를 제안하였습니다.

- **Technical Details**: FAS는 LLM을 스파이킹 LLM으로 변환하는 두 단계의 프로세스를 포함합니다. 첫 단계에서는 사전 훈련된 모델의 전체 매개변수를 미세 조정하여 처음부터 직접 교육할 필요가 없습니다. 두 번째 단계에서는 정밀도를 높이고 변환 오류를 줄이기 위해 코스-투-파인(calibration) 방법을 도입합니다.

- **Performance Highlights**: 우리의 실험은 언어 및 비전-언어(vision-language) 작업에서 4가지 LLM 규모에 걸쳐 진행되었습니다. FAS는 최첨단 성능을 달성하면서도 추론 지연(inference latency)과 계산 비용을 크게 줄였습니다. 예를 들어, FAS는 OPT-7B 모델보다 3% 더 높은 정확도를 8 timesteps로 달성하며 에너지 소비를 96.63% 줄였습니다.



### Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models (https://arxiv.org/abs/2502.04404)
Comments:
          This is a preprint under review, 15 pages, 13 figures

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)에 slow-thinking 메커니즘을 통합하여 AGI(인공지능 일반화) Reasoners의 Level 2 달성을 위한 새로운 경로를 제시합니다. 기존 모델들은 오버씽킹(overthinking) 문제와 보조 보상 모델(auxiliary reward models)에 대한 과도한 의존성이 있었으며, 이로 인해 비효율성이 발생했습니다. 연구팀은 LLMs가 검색 프로세스를 내부화(internalize)할 수 있도록 self-backtracking 메커니즘을 도입했습니다.

- **Technical Details**: Self-Backtracking 기법은 LLM이 훈련 단계에서 언제, 어떤 상황에서 backtracking을 수행해야 하는지를 학습하도록 돕습니다. 이 방법은 모델이 초기 예측이나 추론 경로가 최적이 아닐 때 이를 인지하고 조기에 상태로 되돌아가 대안 가능성을 탐색하도록 합니다. 실험 결과, 제안한 기법이 기존의 SFT(supervised fine-tuning) 방식에 비해 40% 이상의 성능 향상을 기록했습니다.

- **Performance Highlights**: 제안된 self-backtracking 기법은 LLMs의 추론 유연성과 전반적인 성능을 개선하며, 다양한 파라미터 규모의 모델에서도 뛰어난 효과를 나타냅니다. Countdown 작업(task)에 대한 실험에서 이 기법이 특히 두드러진 성과를 보여주었으며, 이는 Level 2 AGI Reasoners 달성을 향한 중요한 진전으로 평가됩니다.



### Beyond Interpolation: Extrapolative Reasoning with Reinforcement Learning and Graph Neural Networks (https://arxiv.org/abs/2502.04402)
Comments:
          The first two authors contributed equally to this work. Accepted as workshop paper at NEURMAD@AAAI25

- **What's New**: 이번 연구에서는 neural architecture의 일반화(generalization) 문제를 해결하기 위해 logic puzzles를 활용합니다. 기존의 전통적인 접근법이 복잡한 논리 구조를 제대로 표현하는 데 어려움을 겪고 있는 반면, 본 연구는 graph-based approach를 통해 이러한 퍼즐을 모델링합니다.

- **Technical Details**: 연구는 reinforcement learning 환경에서 proposed models가 일반화된 솔루션을 학습하는 데 필요한 핵심 요소를 조사합니다. 여기에는 architecture의 inductive bias, 다양한 reward system, 그리고 sequential reasoning을 가능하게 하는 recurrent modeling의 역할이 포함됩니다.

- **Performance Highlights**: 실험을 통해 이러한 요소들이 점점 더 복잡한 문제에서 성공적인 extrapolation에 기여하는 방식을 입증합니다. 이 연구는 interpolation을 넘어서는 일반화된 reasoning을 할 수 있는 학습 기반 시스템을 설계할 수 있는 체계적인 방법을 제공합니다.



### Adaptive Prototype Knowledge Transfer for Federated Learning with Mixed Modalities and Heterogeneous Tasks (https://arxiv.org/abs/2502.04400)
- **What's New**: 본 논문에서는 Multimodal Federated Learning (MFL)의 한계를 극복하기 위한 새로운 프레임워크인 Adaptive prototype-based Multimodal Federated Learning (AproMFL)을 제시합니다. AproMFL은 고객들이 공통의 공공 데이터셋 없이도 적응형으로 구성된 프로토타입을 통해 지식을 전이할 수 있도록 설계되었습니다. 이 연구는 서로 다른 모달리티와 태스크를 가진 클라이언트가 MFL 훈련에 독립적으로 참여할 수 있게 합니다.

- **Technical Details**: AproMFL은 클라이언트가 로컬 데이터셋에 기반하여 적응형으로 프로토타입을 구성하는 방식을 사용하여 로컬 모달리티 정보를 표현합니다. 또한, 서버는 클라이언트가 생성한 프로토타입을 통합하여 글로벌 프로토타입으로 전환하고, 이를 통해 통합된 모델을 형성합니다. 이 과정에서 모델을 다양한 모듈로 나누어 통신 및 계산 오버헤드를 줄이고, 비대칭성을 해결하기 위해 클라이언트 관계 그래프 기반의 가중치 조정 방식을 개발했습니다.

- **Performance Highlights**: AproMFL은 세 가지 기준선 데이터셋에서 분류 작업 및 다중 모달 검색 작업을 통해 경량 모델에서도 우수한 정확도(precision)와 재현율(recall) 성능을 달성했습니다. 특히, 기존 방법들과 비교하여 훨씬 적은 학습 파라미터로 모델을 훈련할 수 있음을 입증했습니다. 이 연구는 혼합 모달리티와 이질적인 태스크를 가지고 있는 MFL의 새로운 지평을 열었습니다.



### Online Location Planning for AI-Defined Vehicles: Optimizing Joint Tasks of Order Serving and Spatio-Temporal Heterogeneous Model Fine-Tuning (https://arxiv.org/abs/2502.04399)
- **What's New**: 이 논문은 차량 크라우드 센싱(Vehicle Crowdsensing, VCS)과 주문 수행 작업을 통합하는 온라인 프레임워크를 제안합니다. 특히, 차량이 도시 데이터를 기반으로 하여 파라미터 효율적 미세 조정(Parameter Efficient Fine Tuning, PEFT) 기법을 활용하는 새로운 접근방식을 다룹니다. 이러한 연구는 스마트 도시 개발에 대한 응용 가능성을 높이고, 차량의 자원을 활용하는 방법을 탐구합니다.

- **Technical Details**: 이 연구에서는 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning, MARL)을 기반으로 한 프레임워크를 통해 차량의 위치 계획과 모델 세부 조정의 최적화를 시도합니다. 모델의 품질 관리(Quality-of-Service, QoS) 메트릭을 설계하여 데이터 양과 신선도의 변화에 따라 두 가지 작업을 균형 있게 처리합니다. 또한, 그래프 신경망(Graph Neural Networks, GNNs)를 MARL에 통합하여 차량의 상태 표현을 개선하고, 시간에 따라 가변적인 의존성을 포착합니다.

- **Performance Highlights**: 실험 결과는 뉴욕시 택시 주문 데이터 세트를 활용하여 제안된 방법이 우수한 성능을 발휘함을 보여줍니다. 차량이 VCS와 주문 수행 작업을 동시에 수행하더라도, 우리의 프레임워크가 전반적인 유틸리티를 극대화할 수 있음을 입증합니다. 이러한 연구는 첨단 AI 기술과 도시 관리의 접목을 통한 스마트 도시에 대한 새로운 통찰력을 제공합니다.



### Multimodal Medical Code Tokenizer (https://arxiv.org/abs/2502.04397)
Comments:
          conference

- **What's New**: 이번 연구에서는 전자 건강 기록(EHR)에 대한 다중 모드 의료 코드 토큰화기인 MedTok을 소개합니다. 기존의 표준 토큰화 방법이 의료 코드를 고립된 텍스트 토큰으로 처리하는 반면, MedTok은 텍스트 설명과 관련 맥락을 통합하여 더 풍부한 의료 코드 표현을 가능하게 합니다. MedTok의 도입은 다양한 EHR 모델의 성능을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: MedTok은 언어 모델 인코더를 사용하여 텍스트를 처리하고, 그래프 인코더를 통해 의료 코드의 관계 구조를 인코딩합니다. 이 과정에서 두 가지 모드를 통합하여 통합된 토큰 공간으로 양자화(quantize)하며, 모드별 및 교차 모드 정보 보존을 보장합니다. MedTok는 또한 의료 질문-답변 시스템에 적용되어 사용될 수 있습니다.

- **Performance Highlights**: MedTok를 다섯 개의 EHR 모델에 통합한 결과, 전통적인 EHR 토큰화기를 MedTok으로 교체할 경우 모든 EHR 모델에서 AUPRC의 개선이 관찰되었습니다. MIMIC-III에서 4.10%, MIMIC-IV에서는 4.78%, EHRShot에서는 11.30% 향상되었습니다. 특히 약물 추천에서 가장 큰 성과를 기록했습니다.



### DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Diseas (https://arxiv.org/abs/2502.04394)
- **What's New**: 본 논문에서는 알츠하이머병(AD) 조기 진단을 위한 새로운 접근법인 DECT를 제안합니다. DECT는 대형 언어 모델(LLMs)를 활용하여 언어적 분석을 세밀하게 수행하고, 노이즈가 포함된 음성 전사에서 중요한 인지-언어적(Cognitive-Linguistic, CL) 정보를 추출합니다. 이 연구는 AD 탐지 모델의 정확성을 11% 향상시키는 성과를 보였습니다.

- **Technical Details**: DECT는 네 가지 핵심 단계로 구성되어 있습니다. 첫째, LLM을 활용해 언어적 데이터를 정제하고, 둘째, 비구조적인 음성 전사에서 언어적 마커를 추출합니다. 셋째, 이러한 마커와 CL 아톰을 결합하여 보다 정교한 데이터 표현을 생성하고, 넷째, 증대된 음성 전사 데이터를 바탕으로 AD 탐지 모델을 세부 조정합니다.

- **Performance Highlights**: DECT의 결과는 DementiaBank 데이터셋에서 기존 기준선 대비 11% 향상된 AD 탐지 정확성을 보여주었습니다. 이는 자동화된 진단 도구 개발에 있어 언어적 패턴 분석의 가능성을 동반하며, AD 조기 발견과 치료 모니터링에 큰 기여를 할 것으로 기대됩니다.



### Division-of-Thoughts: Harnessing Hybrid Language Model Synergy for Efficient On-Device Agents (https://arxiv.org/abs/2502.04392)
- **What's New**: 이번 논문에서는 Division-of-Thoughts (DoT) 라는 협동적 추론 프레임워크를 제안합니다. DoT는 로컬에 배치된 Smaller-scale Language Models (SLMs)와 클라우드 기반의 Large Language Models (LLMs) 간의 시너지를 활용하여 사용자의 쿼리를 더 작은 하위 작업으로 분해하는 Task Decomposer를 포함합니다. 이를 통해 복잡한 온라인 작업을 효율적으로 관리할 수 있도록 도와줍니다.

- **Technical Details**: DoT는 또한 Task Scheduler를 통해 하위 작업 간의 종속성을 분석하고 종속성 그래프를 작성하여 병렬 추론을 촉진합니다. Plug-and-Play Adapter를 사용하여 하위 작업의 난이도에 따라 적절한 모델을 할당하며, 이는 SLM의 파라미터를 변경하지 않고도 가능하게 합니다. 자율 강화 훈련 방법 또한 도입하여 인간의 주석 없이 작업 실행 피드백을 기반으로 하위 작업을 배분할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, DoT는 평균 추론 시간을 66.12% 감소시키고 API 비용을 83.57% 줄이는 동시에 경쟁력 있는 추론 정확도를 유지했습니다. 다양한 벤치마크를 통해 DoT의 효과를 입증했으며, 비용과 정확도의 균형을 뛰어난 성과로 보여줍니다. 전반적으로 DoT는 AI 개인 비서의 성능 향상에 기여할 수 있는 가능성을 제시합니다.



### Towards Fair and Robust Face Parsing for Generative AI: A Multi-Objective Approach (https://arxiv.org/abs/2502.04391)
- **What's New**: 이번 연구는 얼굴 분할을 위한 다목적 학습 프레임워크를 제안합니다. 이 프레임워크는 정확성, 공정성(fairness), 및 강인성(robustness)을 동시에 최적화하여 비율적(segmentation bias) 분할을 개선하고, 실제 조건에서도 안정적인 성능을 제공합니다. 이를 통해 생성 모델에서의 출력 품질을 높이며, 특히 GAN 기반의 얼굴 생성 작업에서 공정성을 고려한 분할이 어떻게 향상되는지를 보여줍니다.

- **Technical Details**: 제안된 방법은 동적 가중치 조정이 가능한 homotopy-based loss function을 사용합니다. 이는 초기 학습 과정에서는 정확성을 중시하고, 이후 단계에서는 공정성과 강인성을 균형 있게 고려합니다. 이러한 접근법은 다양한 인구 집단 간의 분할 성능을 강화하고, occlusions, noise, domain shifts와 같은 실제 환경의 어려움에도 강한 저항력을 제공합니다.

- **Performance Highlights**: 실험을 통해 다목적 U-Net 모델과 단일 목표 U-Net 모델을 GAN 기반의 얼굴 생성 파이프라인(Pix2PixHD)에서 비교했습니다. 그 결과, 공정성을 고려한 강인한 분할이 GAN에서 생성된 얼굴의 품질을 추천적으로 개선하고, Perceptual realism을 높이는 것을 확인했습니다. 또한, ControlNet을 활용한 초기 실험에서는 분할 품질이 diffusion 기반 생성에 미치는 영향을 추가로 분석했습니다.



### In Praise of Stubbornness: The Case for Cognitive-Dissonance-Aware Knowledge Updates in LLMs (https://arxiv.org/abs/2502.04390)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 지속적인 지식 업데이트 방식을 인지적 접근법을 통해 탐구합니다. 기존의 언어 모델은 새로운 정보를 통합하는 데 어려움이 있으며, 이는 종종 이전에 학습한 지식을 파괴적인 망각(catatrophic forgetting)으로 이어지곤 합니다. 연구자들은 두 가지 주요 구성 요소인 불일치 및 친숙성 인식(dissonance and familiarity awareness)과 표적 신경망 업데이트(targeted network updates)를 도입하여 언어 모델의 행동을 분석하고 지식 통합의 근본적인 특징을 밝혀냈습니다.

- **Technical Details**: 이 연구에서는 신경 활동을 추적하여 모델이 새로운 정보를 새로운, 친숙한 또는 불일치하는 정보로 분류할 수 있는지 탐구하는 방법을 개발했습니다. 연구팀은 과거 사용에 따라 뉴런을 '플라스틱(plastic)'과 '고집하는(stubborn)' 것으로 분류할 수 있는 방법도 모색했습니다. 이러한 분석을 통해 언어 모델의 매개변수 공간에서 새로운 지식의 위치가 기존 지식의 통합에 미치는 영향을 심층적으로 연구하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 비불일치(non-dissonant) 업데이트는 모델의 이전 지식을 대부분 보존하는 반면, 불일치(dissonant) 업데이트는 기존 지식에 심각한 파괴적 영향을 미친다는 것을 발견했습니다. 이러한 발견은 대형 언어 모델이 모순된 정보를 처리하는 방식에서 본질적인 한계를 지니고 있음을 시사합니다. 따라서, 연구팀은 모순된 정보를 보다 효과적으로 처리할 수 있는 새로운 메커니즘의 탐색이 필요하다는 결론에 도달했습니다.



### Overcoming Vision Language Model Challenges in Diagram Understanding: A Proof-of-Concept with XML-Driven Large Language Models Solutions (https://arxiv.org/abs/2502.04389)
Comments:
          The related code is available at \url{this https URL}, which provides the core library developed for this research. The experimental code using this library can be found at \url{this https URL}

- **What's New**: 이 연구는 비즈니스 문서 내의 다이어그램에서 정보를 추출하고 구조와 관계를 이해하기 위한 텍스트 기반 접근 방식을 제안합니다. 기존의 Vision-Language Models(VLMs)의 시각적 인식 능력에 의존하지 않고, xlsx, pptx 또는 docx 같은 편집 가능한 소스 파일로부터 다이어그램 정보를 추출하는 방식으로 진행됩니다. 이를 통해 VLM의 한계를 우회하여, LLM이 비즈니스 관련 질문에 대해 보다 정확한 답변을 생성할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 다이어그램의 요소가 텍스트 메타데이터로 저장된 편집 가능한 소스 파일에서 정보를 추출합니다. 차례로 LLM에 텍스트 입력으로서 변환된 정보는 LLM이 관계를 분석하고 비즈니스 중심의 질문에 대한 응답을 생성할 수 있게 합니다. 실험을 통해 VLM 기반 방법과 비교하여 제안된 텍스트 기반 프레임워크가 더 정확한 결과를 제공함을 확인했습니다.

- **Performance Highlights**: 제안된 방법은 기존 VLM들이 다루기 어려웠던 복잡한 다이어그램의 관계 이해에서 우수한 성과를 나타냅니다. 특히, VLM이 정확하지 못한 방식으로 답변하는 질문에 대해, 새롭게 제안된 텍스트 기반 접근은 정확한 답변을 제공합니다. 이러한 결과는 다이어그램 이해를 위한 소스 파일(XML) 처리의 가능성을 보여주며, LLM의 활용을 통한 비즈니스 프로세스의 효율성을 높일 수 있는 잠재력을 시사합니다.



### Position: Emergent Machina Sapiens Urge Rethinking Multi-Agent Paradigms (https://arxiv.org/abs/2502.04388)
- **What's New**: 이 논문은 AI 에이전트가 자율적으로 학습하고 독립적으로 의사 결정을 내릴 수 있는 가능성을 강조합니다. 특히, 서로 다른 목표를 가진 비조정된 AI 시스템들이 어떻게 조화롭게 공존하고 발전할 수 있을지를 탐구합니다. 이는 기존의 다중 에이전트 시스템과 게임 이론의 한계를 극복하기 위한 새로운 접근 방식을 제안하며, 에이전트들이 동적으로 목표를 조정하고 협상하는 능력을 키우는 것을 목표로 합니다.

- **Technical Details**: AI의 발전을 다루며, 심층 강화 학습(deep reinforcement learning), 에이전틱 인공지능(agentic artificial intelligence), 자기 감독 학습(self-supervised learning), 메타 학습(meta-learning) 등의 최신 기술을 언급합니다. 이 논문은 자율 AI 시스템이 어떻게 상호 작용하며 진화할 수 있는지에 대한 새로운 모델을 제안합니다. 이를 통해 시스템 전반의 안전성과 성능을 유지하면서도 AI 에이전트들이 공통의 목표를 위해 협력할 수 있는 방법을 모색합니다.

- **Performance Highlights**: 전통적인 다중 에이전트 강화 학습(multi-agent reinforcement learning)이나 게임 이론적 방법이 갖는 한계를 지적하는 동시에, AI 에이전트들이 독립적으로 성장하고 상호작용하는 환경에서 새로운 프레임워크의 필요성을 강조합니다. 특히, 수렴을 보장하기 위해 사전에 설계된 조정 알고리즘에 의존하기보다는, 에이전트들이 자율적으로 상황에 최적화된 결정을 내릴 수 있도록 하는 것이 중요하다고 제안합니다. 이러한 새로운 접근 방식은 AI의 발전이 사회적 가치와 일치하는 방향으로 나아가는데 기여할 것으로 기대됩니다.



### FedP$^2$EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs (https://arxiv.org/abs/2502.04387)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 다국어에 걸쳐 저자 특정 성능을 향상시키기 위한 새로운 방법론인 FedP2EFT를 제안했습니다. 연합 학습(federated learning) 중 개인화를 위한 최적의 매개변수를 선택하기 위해 Bayesian sparse rank selection을 활용하여, 각 클라이언트별로 최적의 PEFT 구조를 공동으로 학습합니다. 그 결과, FedP2EFT는 기존의 방법들에 비해 성능상이 크게 개선된다는 것을 보였습니다.

- **Technical Details**: FedP2EFT는 연합 학습을 통해 클라이언트가 공동으로 언어 개인화 전략을 학습할 수 있도록 하며, PS generator(PSG)를 통해 메타 데이터를 기반으로 LoRA(Hu et al., 2022) 랭크를 최적화합니다. 이 방식은 기본 모델, 클라이언트 데이터 세트, 자원 예산에 따라 개인화된 LoRA 모듈을 생성하고, 이를 통해 PEFT를 적용하여 개인화된 모델을 생성합니다. 이 방법은 다양한 기초 모델에서 직접 사용할 수 있어 유연성을 가집니다.

- **Performance Highlights**: 실험을 통해 FedP2EFT는 기존 비연합 학습 LoRA 랭크 선택 및 개인화 기법을 포함한 여러 연합 학습 접근 방식들과 비교해 높은 성능을 기록했습니다. 특히, 이 방법은 저자 개인화 수준을 최적화하며, 다양한 연합 학습 모델에 대한 성공적인 보완이 가능함을 보여 주었습니다. 이로 인해 개인화된 다국어 대형 언어 모델의 훈련에 있어서 더 나은 접근 방식을 제공할 수 있을 것으로 기대됩니다.



### Towards Fair Medical AI: Adversarial Debiasing of 3D CT Foundation Embeddings (https://arxiv.org/abs/2502.04386)
- **What's New**: 최근 기계 학습의 자기 지도 학습(self-supervised learning) 기술이 의료 이미징을 혁신적으로 변화시키고 있습니다. 본 논문에서는 특히 3D CT 데이터에 적용된 자기 지도 모델이 성과를 냈지만, 인구통계학적 정보(age, sex, race)도 함께 인코딩하고 있다는 문제를 제기합니다. 이러한 인코딩은 임상 응용의 공정성(fairness)에 심각한 위험을 초래할 수 있음을 강조하며, 이를 해결하기 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 접근 방식은 Variational Autoencoder (VAE) 기반의 적대적 편향 제거(adversarial debiasing) 프레임워크를 사용하여 기존의 인코딩된 임베딩을 새로운 잠재 공간(latent space)으로 변형합니다. 이 과정에서 인구통계학적 정보를 제거하면서도 다운스트림 임상 작업에서의 성능은 유지됩니다. VAE 아키텍처는 입력 3D CT 임베딩을 평균과 로그 분산 파라미터(latent space of mean and log variance parameters)로 매핑하는 인코더와 잠재 공간에서 임베딩을 재구성하는 디코더로 구성됩니다.

- **Performance Highlights**: NLST 폐암 스크리닝 데이터셋을 통해, 제안된 불편(framework) 변환 과정에서 인구통계학적 신호를 효과적으로 제거되는지를 검증하였습니다. 실험 결과, 1년 및 2년 간의 폐암 위험 예측의 정확도를 저하시키지 않으면서도 공정성을 크게 향상시킨 것을 보여주었습니다. 이러한 성과는 의료 분야에서 UberBiasing 기술의 가능성을 강조하며, 편향 없는 의료 의사 결정에 대한 넓은 채택을 위한 기반을 마련하는 데 기여합니다.



### TexLiDAR: Automated Text Understanding for Panoramic LiDAR Data (https://arxiv.org/abs/2502.04385)
- **What's New**: 이 연구는 고급 LiDAR 센서인 Ouster OS1이 생성한 2D 이미지를 활용하여 텍스트와 LiDAR 데이터를 연결하는 새로운 접근법을 제안합니다. 기존의 3D 포인트 클라우드를 의존하는 방법의 단점을 극복하고, 더 효율적으로 처리할 수 있는 2D 이미지 처리를 통해 이미지 캡셔닝 및 객체 감지를 수행합니다. Florence 2 모델을 활용하여 다양한 시각적 작업을 수행하며, 기존 방법보다 더 자세하고 정확한 결과를 도출합니다.

- **Technical Details**: Ouster OS1 LiDAR 센서는 고해상도의 깊이, 신호, 환경 이미지를 생성하며, 이를 통해 360도 뷰의 공간적 일관성을 제공합니다. 본 연구에서는 Florence 2 모델을 사용하여 2D 이미지에서 직접 이미지 캡셔닝과 객체 감지를 수행하며, 이미지의 360도 데이터를 전처리 없이 활용합니다. 이미지는 90도 섹션으로 나누어져 각 부분에서 독립적으로 처리되며, 모든 세그먼트의 예측 결과를 결합하여 전체 장면을 이해하는 방식을 적용합니다.

- **Performance Highlights**: 실험 결과, Florence 2는 기존의 LidarCLIP 같은 방법에 비해 더 정보를 제공하는 캡션과 뛰어난 객체 탐지 성능을 보여주었습니다. 점간 거리 계산을 통해 감지된 객체의 각도와 거리를 추정하는 기능도 제공되며, 실제 응용 시나리오에서 높은 정확도와 견고성을 요구하는 작업에 적합한 솔루션을 제시합니다. 이 방법은 LiDAR 기반 작업에서 보다 세밀하고 통찰력 있는 결과를 실현합니다.



### Enhancing Reasoning to Adapt Large Language Models for Domain-Specific Applications (https://arxiv.org/abs/2502.04384)
Comments:
          NeurIPS 2024 Workshop AFM (Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning)

- **What's New**: 이 논문은 SOLOMON이라는 신경 영감을 받은 대형 언어 모델(LLM) 추론 네트워크 아키텍처를 소개하며, 이 모델이 도메인 특정 응용 프로그램을 위한 적응력을 크게 향상시킬 수 있다는 것을 보여줍니다. 반도체 레이아웃 설계에서의 사례 연구를 통해, SOLOMON은 일반 LLM이 전문 작업에 빠르게 적응할 수 있도록 하는 방법을 Demonstrate합니다. 실험 결과, SOLOMON은 LLM의 기본 성능을 뛰어넘으며 최신 추론 모델 o1-preview와 유사한 성능을 달성했습니다.

- **Technical Details**: SOLOMON의 아키텍처는 두 가지 신경 영감을 받은 이론에 기반합니다: 브레인 라이크 AGI와 자유 에너지 원칙입니다. 이 아키텍처는 다양한 LLM의 생각을 이용한 병렬 검색 엔진과 적응형 RAG 시스템을 통해 특정 작업에 대한 최적의 추론 계획을 발견할 수 있게 합니다. 기본 요소는 LLM 기반 시스템과 인간 조작 구성 요소를 포함하여 LLM의 안전성을 확보하고 기존 LLM의 반복적 미세 조정 필요성을 제거함으로써 다양한 전문 맥락에 맞춰 유연한 AI 시스템을 구축할 수 있게 합니다.

- **Performance Highlights**: SOLLOMON은 다양한 수준의 복잡성을 가진 25개의 레이아웃 설계 작업을 평가하였으며, 이는 기본 모양부터 복잡한 구조체까지 포함됩니다. 모델의 성능은 5개의 LLM과 비교하였고, SOLOMON은 기존 LLM에 비해 우수한 성능을 보였습니다. 결과적으로, 추론 능력이 LLM의 다양한 도메인 응용 프로그램에 대한 적응력을 향상시키는 데 중요한 역할을 함을 강조하고 있으며, 신경 과학에서 영감을 얻은 추론 능력의 개발이 향후 연구 방향으로 제시되었습니다.



### Sparse Autoencoders for Hypothesis Generation (https://arxiv.org/abs/2502.04382)
Comments:
          First two authors contributed equally; working paper

- **What's New**: HypotheSAEs는 텍스트 데이터(예: 헤드라인)와 타겟 변수(예: 클릭) 간의 해석 가능한 관계를 추론하는 일반적인 방법을 제시합니다. 이 방법은 텍스트 임베딩에서 희소 오토인코더(sparse autoencoder)를 훈련하여 데이터 분포를 설명하는 해석 가능한 특징을 생성하고, 타겟 변수를 예측하는 특징을 선택한 후, LLM을 통해 이러한 특징에 대한 자연어 해석을 생성하는 세 단계로 구성됩니다. 이 연구는 LLM 기반 방법들에 비해 더 적은 컴퓨팅 자원으로 더 많은 예측 가능한 가설을 생성하고 있습니다.

- **Technical Details**: HypotheSAEs는 다음과 같은 세 가지 주요 단계를 포함합니다. 첫째, 텍스트 임베딩에서 희소 오토인코더(SAE)를 훈련하여 해석 가능한 뉴런을 학습합니다. 둘째, Lasso와 같은 방법을 사용하여 타겟 변수를 예측하는 뉴런을 선택합니다. 셋째, 활성화된 입력 텍스트를 바탕으로 LLM을 사용하는 고충실도의 자연어 해석을 자동으로 생성합니다. 이러한 해석은 타겟 변수를 예측하는 가설 역할을 합니다.

- **Performance Highlights**: HypotheSAEs는 60개의 가설 중 45개가 유의미하다는 점에서 세 개의 실제 세계 과제에서 기존 세 가지 방법에 비해 훨씬 더 많은 유의미한 가설을 생성합니다. 또한, 이 방법은 최근의 LLM 기반 방법들보다 10배 이상 빠르고 비용이 저렴하며, 입력의 뉴런 활성화를 통해 한 번의 SAE 전방 패스(forward pass)로 모든 뉴런 활성화를 계산할 수 있어 효율적인 비용 절감이 가능합니다.



### Limitations of Large Language Models in Clinical Problem-Solving Arising from Inflexible Reasoning (https://arxiv.org/abs/2502.04381)
Comments:
          14 pages, 6 figures

- **What's New**: 최근 연구에 따르면, Large Language Models (LLMs)는 의학적 질문 응답 기준에서 인간 수준의 정확성을 달성했지만, 열린 임상 시나리오를 탐색하는 데 있어 한계를 보이고 있으며, 이는 LLM의 추론 능력에 대한 신뢰를 의심하게 만듭니다. 본 논문에서는 의료 추상화 및 추론 코퍼스 (M-ARC)를 소개하여 LLM의 임상 문제 해결에서 발생할 수 있는 실패 모드를 조사합니다. M-ARC는 LLM의 경직된 패턴 매칭과 관련된 유연성 부족을 드러내며, LLM이 실제 의사보다 성능이 낮다는 사실을 보여줍니다.

- **Technical Details**: M-ARC 질문은 미국 의사 면허 시험 (USMLE)에서 사용하는 객관식 형식을 기반으로 하며, 훈련 데이터에서 접한 적이 없는 임상 문제를 평가하기 위해 설계되었습니다. dataset은 전형적인 의학 텍스트에서 잘 다루어지지 않는 오픈형 응답을 포함하여 각 질문이 진단적인 결정에 충분한 정보를 제공하는지 여부를 평가합니다. 연구팀은 UCSF에서 의사를 모집하여 LLM과 비교하여 성능을 평가했습니다.

- **Performance Highlights**: M-ARC를 통해 검토된 여러 LLM 모델들은 50% 이하의 정확도로 수행했으며, 이는 LLM의 일반적인 의학적 추론 능력이 제한적임을 나타냅니다. 이 연구에서는 LLM이 자주 hallucination을 발생시키며, 자신의 답변에 대해 과신하는 경향을 보이는 것을 확인했습니다. M-ARC의 발견은 임상에서 LLM을 사용할 때 신중해야 함을 강조합니다.



### Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data (https://arxiv.org/abs/2502.04380)
Comments:
          26 pages, 15 figures, 11 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능 향상에 있어 데이터 다양성이 중요한 역할을 한다는 점을 강조합니다. 새로운 방법인 DaaR을 제안하여 LLM이 데이터의 다양성을 보상 신호로 학습하고, 도메인에 구애받지 않는 훈련 데이터를 자율적으로 선택하도록 유도합니다. 이 연구의 결과는 LLM의 전반적인 성능을 높이는 데 기여할 수 있는 방법론을 제시합니다.

- **Technical Details**: DaaR 방법은 LLM에 대한 새로운 미세 조정 프레임워크로, 외부 다층 퍼셉트론(MLP) 구조를 통합하여 LLM의 고유한 지식 및 가중치에 따라 조정 가능한 프로브(modules of probe)를 생성합니다. 이 프로브는 샘플링된 데이터의 의미적 엔트로피를 측정해 데이터의 적합성을 평가하고, 다양성을 극대화하는 데 필요한 데이터를 선택합니다. 각 데이터 세트는 10,000 개의 샘플로 구성되어 있으며, 이후 랜덤 베이스라인을 위해 8,000 개의 데이터를 uniform하게 샘플링합니다.

- **Performance Highlights**: DaaR 방법은 Qwen 및 LLaMA와 같은 다양한 최신 SOTA LLM에서 시행된 여러 실험을 통해 입증된 바와 같이 모델의 전반적인 능력을 상당히 향상시키는 것으로 확인되었습니다. 특히, 도메인 라벨이 부족한 데이터에서 강력한 성능 향상을 보여주며, 다른 SOTA 방법들은 이러한 어려운 시나리오에서 성능이 저하되는 경향을 보였습니다. 이로 인해 제안된 방법은 다양한 벤치마크에서 개선된 성과를 기록할 수 있는 잠재력을 지니고 있습니다.



### Can Large Language Models Capture Video Game Engagement? (https://arxiv.org/abs/2502.04379)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 대규모로 사전 훈련된 언어 모델(LLMs)이 비디오를 통해 인간의 정서를 감지할 수 있는지를 평가한 최초의 종합 연구입니다. 연구진은 20개의 1인칭 슈팅 게임에서 총 80분의 비디오 게임 영상을 활용하여 플레이어의 참여도를 예측하는 LLM의 능력을 조사했습니다. 특히, 다양한 실험을 통해 LLM 아키텍처, 모델 크기, 입력 모드, 프롬프팅 전략이 성능에 미치는 영향을 분석하고 있습니다.

- **Technical Details**: 본 연구에서는 LLM이 비디오의 시청자 참여를 정확하게 라벨링할 수 있는지를 조사하며, 이를 위해 2,400개 이상의 실험을 수행했습니다. 실험은 GameVibe 데이터셋에서 플레이어의 참여에 대한 연속 라벨을 기반으로 하여 진행되었습니다. 우리는 LLaVA와 GPT 계열의 최신 모델을 비교하면서, 프롬프팅 전략 및 데이터 프로세싱 방법이 LLM의 성능에 미치는 영향을 깊이 있게 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM은 여러 도메인에서 인간과 유사한 성능을 나타냈으나 연속 감정 라벨링에서는 일반적으로 부족한 결과를 보였습니다. 특히 GPT-4o 모델과 몇 가지 예제를 혼합하여 제공했을 때 평균 66%의 정확도를 기록했으며, 특정 게임에서는 47%까지 성능 향상을 이뤘음이 밝혀졌습니다. 이러한 결과는 LLM이 감정 라벨링에 대한 자동화의 가능성을 제시하며, 향후 연구 방향에 대한 로드맵을 제공하고 있습니다.



### MapFusion: A Novel BEV Feature Fusion Network for Multi-modal Map Construction (https://arxiv.org/abs/2502.04377)
- **What's New**: 이 논문에서는 자율 주행 시스템에 필수적인 정적 환경 정보를 제공하는 맵 구축 과제의 중요성을 강조합니다. 저자는 카메라와 LiDAR를 사용하는 다양한 센서 구성에 따라 발생하는 문제를 해결하기 위해 새로운 다중 모달(Bird's-Eye View, BEV) 특징 융합 방법인 MapFusion을 제안합니다. 특히, 기존 방법들이 상호 작용을 무시하고 단순한 융합 전략에 의존하는 문제를 해결하기 위해 Cross-modal Interaction Transform (CIT) 모듈을 도입합니다.

- **Technical Details**: MapFusion은 카메라와 LiDAR BEV 특징 간의 의미적 불일치 문제를 해결하기 위해 설계되었습니다. CIT 모듈은 두 BEV 특징 공간 간의 상호 작용을 가능하게 하여 자기-주의(self-attention) 메커니즘을 통해 특징 표현을 향상시킵니다. 또한, Dual Dynamic Fusion (DDF) 모듈을 통해 다양한 모달리티에서 유용한 정보를 적응적으로 선택하여 정보의 잠재성을 최대한 활용합니다. MapFusion은 간편하고 즉시 통합 가능하여 기존 파이프라인에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: MapFusion은 HD 맵과 BEV 맵 세분화라는 두 가지 맵 구축 과제에서 평가되었습니다. nuScenes 데이터셋을 기준으로, MapFusion은 HD 맵 구축에서 3.6% 및 BEV 맵 세분화 과제에서 6.2%의 절대 개선을 달성하여 최신 방법들과 비교해 우수성을 입증하였습니다. 이러한 성과는 MapFusion의 다중 모달 융합 접근법의 유용성을 나타내며, 자율 주행 시스템의 핵심 성능 향상으로 이어질 수 있습니다.



### MEETING DELEGATE: Benchmarking LLMs on Attending Meetings on Our Behalf (https://arxiv.org/abs/2502.04376)
- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)을 활용한 회의 위임 시스템의 프로토타입을 개발하고, 실제 회의 전사록을 바탕으로 종합적인 벤치마크를 구축하였습니다. LLM의 자연어 이해 및 생성 능력을 통해 회의의 맥락을 파악하고, 동적 대화에 참여할 수 있는 가능성을 탐구했습니다. 평가 결과, GPT-4/4o는 균형 잡힌 전략을 보여주며, Gemini 1.5 Pro는 더 조심스러운 반응을 보입니다.

- **Technical Details**: 회의 위임 시스템은 정보 수집, 회의 참여, 음성 생성의 세 가지 주요 요소로 구성됩니다. 정보 수집 단계에서는 사용자가 관심 주제와 자료를 미리 제공하거나, 개인 지식 기반과 실시간으로 연동하여 정보를 수집합니다. 이후 회의 참여 모듈에서 실시간으로 회의 상황을 모니터링하며, 발언 후 적절한 개입 시기를 결정합니다.

- **Performance Highlights**: 테스트 결과, 전체 약 60%의 응답이 주요 포인트를 잘 커버하고 있지만, 부적절하거나 반복적인 콘텐츠, 실시간 처리 지연 등의 개선이 필요함을 드러냈습니다. 랜덤 샘플링을 통해 수집한 데이터로는, 응답의 일관성과 주제 적합성을 높일 수 있는 방법을 모색하고 있습니다. 이러한 시스템은 개인이 회의에 참여하는 부담을 크게 덜어줄 잠재력을 가지고 있습니다.



### Contrastive Token-level Explanations for Graph-based Rumour Detection (https://arxiv.org/abs/2502.04366)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 Graph Neural Network (GNN) 기반의 루머 탐지 모델에 대한 해석 가능성을 개선하기 위해 Contrastive Token Layerwise Relevance Propagation (CT-LRP)라는 새로운 프레임워크를 소개합니다. CT-LRP는 기존의 설명 기법의 한계를 극복하고, 개별 토큰 수준의 설명을 제공하여 모델의 예측을 보다 세분화하여 해석할 수 있도록 합니다. 이를 통해 모델 신뢰성과 투명성이 향상될 것으로 기대됩니다.

- **Technical Details**: CT-LRP는 Layerwise Relevance Propagation (LRP)와 설명 공간 파티셔닝 전략을 결합하여 클래스-specific 텍스트 구성 요소를 분리하고, 고차원 텍스트 임베딩의 의존성을 포착합니다. 기존의 GNN 설명 기법들이 노드나 엣지 수준의 인사이트에 그치는 반면, CT-LRP는 문장 구성 요소에 대한 세부적인 분석을 통해 더 높은 충실도와 정밀도를 제공합니다. 이 방법은 세 개의 공개 루머 탐지 데이터셋에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: CT-LRP는 세 가지 공개 데이터셋에 대한 실험에서 항상 신뢰할 수 있는 고품질의 설명을 생성하는 것으로 나타났으며, GNN 기반 설명 가능성의 새로운 기준을 설정합니다. 이는 AI 시스템의 투명성과 신뢰성을 향상시키며, 유해한 정보에 대응하는 윤리적이고 효과적인 AI 시스템 구축으로 이어질 것입니다. 이러한 발전은 이해관계자가 더 나은 결정을 내릴 수 있는 기반을 마련합니다.



### AI-Based Thermal Video Analysis in Privacy-Preserving Healthcare: A Case Study on Detecting Time of Birth (https://arxiv.org/abs/2502.04365)
Comments:
          Paper accepted in 2025 IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 이 연구는 신생아의 출생 시각(Time of Birth, ToB)을 자동으로 감지할 수 있는 AI 기반 비디오 시스템을 제안합니다. 이 시스템은 열화상 영상(thermal imaging)을 활용하여 엄마와 의료 제공자의 개인정보를 보호하면서도 정확한 ToB 감지를 목표로 합니다. 91.4%의 정밀도와 97.4%의 재현율을 기록하며, 수동 주석과 비교했을 때 96%의 경우에서 ToB를 정확하게 감지합니다.

- **Technical Details**: 연구는 열화상 영상을 사용하여 ToB를 감지하는 방법론을 설명합니다. Gaussian Mixture Models (GMM)를 활용한 적응형 노멀라이제이션 기법을 사용하여 입력된 열화상 비디오를 처리하고, 슬라이딩 윈도우 기법으로 예측을 수행합니다. 이는 연속적인 움직임과 열 특성을 분석하여 출생 과정을 동적으로 포착함으로써 ToB 문서화의 정밀도를 높이는데 기여합니다.

- **Performance Highlights**: 본 연구의 시스템은 수동 기록과의 비교에서 1초의 절대 중앙 편차를 기록하며, 기존 시스템의 제한을 극복할 수 있는 신뢰할 수 있는 솔루션을 제공합니다. 결과적으로, 이 방법은 ToB 문서화를 개선하고 신생아 소생술 성과를 향상시키는데 중요한 기여를 할 것으로 기대됩니다.



### Lost in Edits? A $\lambda$-Compass for AIGC Provenanc (https://arxiv.org/abs/2502.04364)
- **What's New**: LambdaTracer는 텍스트-유도(image guided) 이미지 편집 모델의 복잡한 문제를 해결하기 위해 설계된 새로운 출처(attribution) 메소드입니다. 이 방법은 생성 모델이나 편집 파이프라인에 어떠한 변경 없이도 진짜 결과물과 조작된 결과물을 효과적으로 식별할 수 있습니다. LambdaTracer는 다양한 반복 편집 프로세스에서 효율적으로 작동하며, 특히 악의적인 편집 이미지의 탐지에서 뛰어난 성과를 보입니다.

- **Technical Details**: 본 논문에서는 두 가지 접근법인 임베딩 기반 탐지 방법과 잠재 공간(reverse-engineering) 탐지 방법을 분석합니다. LambdaTracer는 이러한 기존 방법들의 단점을 극복하고, 텍스트-유도(image guided) 모델의 복잡한 편집 히스토리를 처리할 수 있도록 설계되었습니다. 유연한 손실 변환 기술을 통해 다양한 시나리오에서 출처 구분의 정확성을 향상시키며, 개방적 환경에서도 강력한 성능을 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과, LambdaTracer는 기존 기준선(baseline) 탐지 방법보다 악의적으로 편집된 이미지를 훨씬 더 효과적으로 구별하는 것을 입증했습니다. 이로 인해 창작물의 소유권, 창의성, 신뢰성을 지키기 위한 실용적인 솔루션을 제공합니다. 연구에서 제안된 방법은 생성된 이미지와 편집된 이미지 간의 최상의 출처 추적 정확도를 기록하며, 향후 AI 생태계에서의 콘텐츠의 진정성과 추적성 확보에 기여할 것으로 기대됩니다.



### LLMs can be easily Confused by Instructional Distractions (https://arxiv.org/abs/2502.04362)
Comments:
          8 pages

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 "instructional distraction" 현상을 다루며, 이를 평가하기 위한 새로운 벤치마크인 DIM-Bench를 소개합니다. Instructional distraction은 입력이 지시어와 유사할 때 발생하는 혼란을 의미하며, LLM이 이러한 혼란스러운 상황에서 어떻게 반응하는지를 평가합니다. DIM-Bench는 20개의 카테고리로 구성되어 있으며, 재작성, 교정, 번역, 스타일 전환과 같은 지시 작업과 추론, 코드 생성, 수학적 추론 등 5개의 입력 작업을 포함합니다.

- **Technical Details**: DIM-Bench는 LLM의 instruction-following 능력을 평가하기 위해 설계된 새로운 벤치마크입니다. 이 벤치마크는 두 가지 차원에서 작업을 결합하여 LLM의 성능을 비교합니다. 특히, 연구에서 사용된 LLM들은 명시적인 프롬프트가 주어졌음에도 불구하고 instructional distraction에 대해 완전한 강건함을 보이지 않으며, LLM의 반응을 더욱 면밀히 분석할 필요성을 제기합니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들도 instruction-following 작업에서 사용자 의도를 제대로 따르지 못하는 경우가 많았습니다. 특히 question answering 작업에서 LLM은 지시 문맥을 무시하고 입력 질문에 대한 답변을 생성하는 경향이 강하게 나타났습니다. 이러한 발견은 LLM의 instructional distraction 상황에서의 성능 한계를 강하게 부각시키며, 향후 강건성을 향상시키기 위한 추가 연구의 필요성을 제안합니다.



### Predicting 3D Motion from 2D Video for Behavior-Based VR Biometrics (https://arxiv.org/abs/2502.04361)
Comments:
          IEEE AIxVR 2025: 7th International Conference on Artificial Intelligence & extended and Virtual Reality

- **What's New**: 본 논문에서는 VR(가상 현실) 사용자 인증의 안전성을 강화하기 위해 새롭게 제안된 방법을 소개합니다. 기존의 PIN, 비밀번호 및 다중 인증 방식의 한계를 극복하고, VR HMD(헤드 마운트 디스플레이)와 손 컨트롤러의 움직임을 이용하여 사용자의 행동을 생체 인증(signature)으로 활용합니다. 특히, 외부 2D 카메라를 통해 획득한 2D 신체 관절 데이터를 사용하여, 3D 경로 예측을 통해 인증의 정확성을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 오른쪽 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목의 2D 관절 데이터를 추적하는 방법을 사용합니다. 이 데이터는 Transformer 기반의 심층 신경망을 통해 3D 오른쪽 컨트롤러의 과거 및 미래 경로를 예측하는 데 활용됩니다. 이를 통해 사용자 동작 데이터의 부족함을 보완하고, 3D 질병 생체 정보로서의 가치 또한 확보할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 3D 경로 기반 방식을 사용하는 방법들에 비해 평균 0.025의 EER(평균 동등 오류율)을 달성하며, 최대 0.040의 EER 감소 효과를 보여주었습니다. 이는 외부 비디오 데이터를 활용하여 사용자 행동 기반 인증의 성능을 혁신적으로 향상시킨 결과로, VR 생체 인증의 새로운 가능성을 열어주는 성과입니다.



### Exploring Spatial Language Grounding Through Referring Expressions (https://arxiv.org/abs/2502.04359)
- **What's New**: 최근 비전-언어 모델(Vision-Language Models, VLMs)이 공간적 추론(spatial reasoning) 능력에서 한계가 있다는 점이 지적되었습니다. 본 연구에서는 전통적인 이미지 캡셔닝(image captioning) 및 시각적 질문 응답(Visual Question Answering) 대신, 지칭 표현 이해(Referring Expression Comprehension, REC) 과제를 새로운 평가 플랫폼으로 제안합니다. 이 과제를 통해 모호한 객체 탐지, 복잡한 공간 표현 및 부정 표현이 포함된 상황에서 VLMs의 공간 이해 및 기반 능력을 심층 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: 이 연구는 CopsRef 데이터셋을 사용하여 51개의 공간 관계를 분석합니다. 이를 바탕으로 LLaVA와 Grounding DINO라는 두 가지 인기 있는 VLM과 REC 전용 모델인 MGA-Net을 비교합니다. 분석을 통해 공간적 표현의 수가 VLM의 성능에 미치는 영향을 검토하며, 각 모델의 디자인 및 훈련 전략 차이에 따라 성능 차이를 측정하고 분석합니다.

- **Performance Highlights**: 분석 결과, 공간적 관계는 지칭 표현의 다른 속성과 결합될 때 더 정확한 기준을 제공합니다. 공간적 복잡성이 증가함에 따라 VLM의 성능이 변화하지만, 명시적 조합 학습(component)가 포함된 모델은 성능을 유지하는 경향이 있습니다. 모든 모델이 부정적인 공간적 관계 처리에 어려움을 겪지만 그 정도는 다양한 것으로 나타났습니다.



### Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives (https://arxiv.org/abs/2502.04358)
Comments:
          12 pages including references

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 기반으로 한 에이전트 시스템의 효율성을 위해 비율 분석(Asymptotic Analysis)의 필요성을 주장합니다. 자동화된 역할 분담이 일반적으로 직관적으로 이루어지며, 이는 인간 팀의 역할 배정과 유사하다는 점을 강조합니다. 그러나 저자들은 이러한 역할 분해가 최적화에서 얼마나 가까운지를 이해하기 위한 분석이 필요하다고 강조합니다.

- **Technical Details**: 비율 분석에서 LLM 프리미티브(primitive)를 중요 개념으로 제안하며, LLM의 전방 패스(forward pass)를 계산의 기본 단위로 취급합니다. LLM 기반 알고리즘(LbA)은 여러 LLM 기반 에이전트가 협력하여 작업을 완수하는 시스템을 의미합니다. 전통적인 비율 분석에서는 계산의 기본 단위를 원자적 작업(primitive)이라고 정의하는 반면, 저자는 LLM의 단일 실행을 해당 기본 단위로 간주합니다.

- **Performance Highlights**: 이 논문에서는 LLM 프리미티브를 활용하여 성능 분석을 수행하고, LLM 기반 시스템의 효율성을 측정하기 위한 새로운 접근 방식을 제시합니다. 여러 사례 분석을 통해 비율 분석이 가져오는 통찰력을 강조하며, 이러한 분석이 연구 및 개발 방향의 기초가 되어 궁극적으로 규모 확장에 기여할 것이라고 주장합니다.



### Reusing Embeddings: Reproducible Reward Model Research in Large Language Model Alignment without GPUs (https://arxiv.org/abs/2502.04357)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 보상 모델 연구에서 임베딩 기반(input) 사용을 제안합니다. 기존의 보상 모델 훈련이 복잡성과 계산 비용으로 인해 제한되는 점을 개선하기 위해, 임베딩을 이용한 방법으로 재현성을 높이고 훈련과 평가의 비용을 줄일 수 있음을 시사합니다. 특히, 이러한 접근 방식은 훈련 안정성을 개선하고, 하드웨어의 계산 요구를 감소시키며, 연구의 속도를 가속화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 저자들은 보상 모델에서 임베딩을 사용하는 방법과 전통적인 자연어 입력 방식을 비교합니다. LLM에서 자연어 입력으로 효율적인 품질 평가를 위해 임베딩 공간을 활용하는 것이 발전 가능성이 크다고 강조하며, 최근의 연구 결과로 수학적 추론 과제와 LLM 생성 콘텐츠의 안전성 및 유용성을 평가하는 데 이 방법이 효과적임을 나타냅니다. 또한, 보상 모델은 최소한의 하드웨어 자원으로 훈련할 수 있는데, 훈련 시간이 1~5분으로 짧습니다.

- **Performance Highlights**: 임베딩 기반의 보상 모델과 기존 LLM 기반 보상 모델 간의 성과 비교 실험을 통해 효용성을 평가했으며, 연구 결과 임베딩 모델이 해석 가능성과 정확성을 모두 갖춘다는 점이 강조되었습니다. 실제 실험에서는 특정 3계층 MLP 모델이 0.6M 미만의 파라미터로 효율적으로 작동하는 것을 보여주었고, 기존 연구들의 기준에 비해 훈련 및 평가 시간이 크게 단축됨을 입증하였습니다. 또한, 다양한 주석 품질 시나리오에 대한 실험을 통해 임베딩 기반 접근 방식의 강점을 입증하며, 이는 보상 모델링 연구의 향후 발전 방향에 중요한 기초가 될 것입니다.



### Open Foundation Models in Healthcare: Challenges, Paradoxes, and Opportunities with GenAI Driven Personalized Prescription (https://arxiv.org/abs/2502.04356)
- **What's New**: 최근 OpenAI의 GPT-4와 같은 상용 대규모 언어 모델(LLMs)의 성공에 대응하여, 개방형 비상용 LLM 및 AI 기초 모델(AIFMs)의 개발에 대한 관심이 증가하고 있습니다. 이 연구는 의료 분야에 대한 개방형 모델의 잠재력에 주목하고 있으며, 이러한 모델들을 통해 보다 효율적인 의료 데이터 분석 및 진단 지원이 가능하다고 주장합니다. 더불어, 이 논문에서는 의료 응용을 위한 최신 개방형 LLM 및 AIFM의 상태를 종합적으로 조사하고, 개인 맞춤형 처방 사례 연구를 통해 그 유용성을 평가합니다.

- **Technical Details**: 이 논문은 개방형 AIFMs에 대한 분류 체계를 소개하며, 의료 이미징, 임상 NLP, 의료 교육 등 다양한 의료업무에 대한 적용 가능성을 다룹니다. 이 연구에서는 LLaMA-2, LLaMA-3, Mistral 및 Meditron 등 여러 개방형 LLM의 성능을 비교하고, Retrieval-Augmented Generation(RAG)을 통합하여 성능을 향상시킬 수 있는 방법을 제시합니다. 문서 전반에서는 개방형 AI 모델의 정의 및 라이센스 문제, 비공식적인 정체성에 대한 논의도 포함되어 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 개방형 LLM은 RAG와 같은 기반 기술을 활용할 경우 상유형 모델에 필적하는 성능을 달성할 수 있다는 점이 강조됩니다. 개인 맞춤형 처방을 통한 평가에서는 전문가 임상의의 주관적 평가를 통해 이러한 개방형 모델이 환자에게 더 나은 치료 결과를 제공할 수 있는 잠재력을 가지고 있음을 확인했습니다. 그러나 이러한 강력한 LLM과 AIFMs의 오용 가능성에 대한 윤리적 고려도 필요하며, 의료 분야에서의 신중하고 책임 있는 구현이 강조됩니다.



### LLM-ProS: Analyzing Large Language Models' Performance in Competitive Problem Solving (https://arxiv.org/abs/2502.04355)
Comments:
          To be published in LLM4Code 2025 workshop proceedings

- **What's New**: 새로운 평가 기법인 LLM-ProS가 제시되었습니다. 이 기법은 세계적인 프로그래밍 대회인 ICPC 문제를 통해 최신 대형 언어 모델(LLM)의 성능을 평가합니다. 2011년부터 2024년까지의 166개의 문제로 구성된 정제된 데이터 세트를 통해 모델의 추론, 정확성 및 효율성을 벤치마킹하였습니다.

- **Technical Details**: 다양한 아키텍처와 훈련 방법론을 갖춘 5개의 모델(GPT-4o, Mistral Large, Llama-3.1-405B, o1-mini 및 o1-preview)을 평가했습니다. 모델들은 문제 해결을 위해 각기 다른 과정을 거쳐 최적화된 접근 방식을 통해 ICPC 문제를 해결하는 데 사용됩니다. LLM-ProS는 데이터 수집, 전처리, 모델 테스트, 해결책 생성 및 제출의 4단계로 구성됩니다.

- **Performance Highlights**: o1 모델들이 높은 정확성과 효율성으로 다른 모델들에 비해 두드러진 성과를 보였습니다. 또한, 모델 간의 성능 차이와 일반화 능력, 적응력의 차이를 발견하였고, 일부 모델들은 복잡하고 높은 난이도의 문제에서 어려움을 겪는 한계를 확인했습니다. 이 연구는 LLM의 설계 및 훈련 방법론 개선을 위한 기초를 제공합니다.



### Reviving The Classics: Active Reward Modeling in Large Language Model Alignmen (https://arxiv.org/abs/2502.04354)
- **What's New**: 이번 연구에서는 인간의 선호에 기반한 보상 모델링을 위한 정보 선택 전략을 제안합니다. 인식 공간의 탐색을 균형 있게 조정하고 보상 차이가 중간인 쌍 간의 의미 있는 비교를 수행하는 것을 목표로 합니다. 이 과정에서 Fisher 정보 기반 선택 전략을 도입하고, 고전 실험 설계 문헌의 이론을 깊은 신경망 모델의 최종 선형층에 적용합니다.

- **Technical Details**: 공식적으로, 본 연구는 Bradley-Terry (BT) 회귀 프레임워크를 사용하여 최적의 선호 레이블 주석 문제를 정의하고, BT 문맥 하에서 능동 학습과 고전 실험 설계 문헌 간의 관련성을 확립합니다. 또한, 고전 실험 설계에서 영감을 받은 다양한 스코어링 규칙들을 도입하고, 이를 대규모 정렬 문제에 적합하도록 조정합니다. 다양한 세팅과 데이터셋에 대해 8888개의 스코어링 알고리즘을 평가해 그 성능을 비교했습니다.

- **Performance Highlights**: 우리 방법은 여러 오픈소스 LLM과 데이터셋 간에 다른 선택 방법들과 비교하여 놀라운 성능과 높은 계산 효율성을 보여줍니다. 실험 결과, 고전 실험 설계 기법을 사용하면 다양한 세팅과 모델 아키텍처에서 뛰어난 성능과 강력한 안정성을 확보할 수 있습니다. 특히, 크로스 프롬프트 비교를 포함하는 능동 보상 모델링이 주석 효율성을 크게 향상시킴을 입증했습니다.



### CognArtive: Large Language Models for Automating Art Analysis and Decoding Aesthetic Elements (https://arxiv.org/abs/2502.04353)
- **What's New**: 본 연구에서는 대량 예술 분석을 자동화하기 위해 LLMs(대형 언어 모델)와 MLLMs(다중 모드 대형 언어 모델)의 가능성을 조사하고 있습니다. 이 모델들을 활용하여 15,000개 이상의 예술 작품을 분석함으로써 예술 작품의 기술적 특성과 표현 특성을 깊이 있게 이해하려고 합니다. 특히, 작품의 패턴이 시간에 따라 어떻게 진화하는지를 탐색하고, 이를 통해 예술적 표현을 해석하는 새로운 방법을 모색합니다.

- **Technical Details**: 이 연구는 GPT-4V와 Gemini 2.0을 활용하여 23명의 저명한 예술가의 작품을 분석할 것입니다. LLMs는 방대한 텍스트 데이터셋을 기반으로 훈련되므로 문헌 분석, 요약 및 질문 답변 등의 과제를 수행할 수 있습니다. 컴퓨터 비전(Computer Vision), 머신 러닝(Machine Learning) 및 자연어 처리(Natural Language Processing) 기술을 통해 디지털 이미지에서 의미 있는 정보를 추출하고, 예술적 스타일을 분류하고, 작가를 식별하며, 작품에 대한 설명을 생성합니다.

- **Performance Highlights**: 이 연구는 고속 대량 예술 분석을 자동화하는 데 있어 혁신적인 접근 방식을 제공합니다. 기존의 분석 방법을 넘어, LLMs는 예술 작품의 형식적 요소, 구성 및 문화적 중요성을 조사하여 이론적, 기술적, 그리고 미학적 요소를 분석하는 데 있어 보다 효율적이고 객관적인 방법을 제공합니다. 데이터 시각화를 통해 결과에 대한 직관적 이해를 돕고 있으며, 이를 통해 예술의 역사적 진화에 대한 새로운 통찰을 발견할 수 있는 기회를 제공합니다.



### Investigating the Robustness of Deductive Reasoning with Large Language Models (https://arxiv.org/abs/2502.04352)
- **What's New**: 이 논문은 LLM(대형 언어 모델)을 기반으로 하는 추론 방식의 강건성을 평가하는 첫 번째 연구를 제안합니다. LLM을 이용한 자동 형식화(autoformalisation) 방법의 설계 요소가 미치는 영향에 대한 체계적인 분석이 부족했습니다. 연구자들은 두 가지 유형의 섭동(perturbations) - 적대적 소음(adversarial noise)과 반사적 진술(counterfactual statements) - 을 활용하여 여섯 개의 섭동 데이터 세트를 생성하여 LLM 추론 방법을 분석했습니다.

- **Technical Details**: 저자들은 LLM 사용 시 적대적 소음이 자동 형식화에 영향을 미치고, 반사적 진술이 모든 접근 방식에 영향을 미친다는 것을 발견했습니다. 제안된 접근 방식은 섭동의 두 가지 가족(적대적 소음, 반사적 섭동)과 더불어 LLM 기반 추론 방법의 구조를 설명하는 방법론적 프레임워크로 요약됩니다. 이는 각 차원(추론 형식, 문법 구문, 오류 회복 메커니즘)에서의 대표적인 접근 방식을 포함합니다.

- **Performance Highlights**: 세 가지 차원에서 분석을 통해 LLM 기반 방법의 강건성에 대한 미세한 통찰력을 제공합니다. LLM의 오류 회복 메커니즘에서는 자세한 피드백이 전체 정확도를 향상시키지 않지만 구문 오류를 줄이는 데 기여하는 것으로 나타났습니다. 즉, LLM이 스스로 오류를 수정하는 데 어려움이 있음을 보여줍니다.



### NER4all or Context is All You Need: Using LLMs for low-effort, high-performance NER on historical texts. A humanities informed approach (https://arxiv.org/abs/2502.04351)
- **What's New**: 이 논문은 역사적 텍스트에서 인물, 장소, 사건 등을 자동으로 인식하고 분류하는 Named Entity Recognition (NER) 작업에 대한 새로운 접근법을 제시하고 긍정적으로 평가합니다. 기존의 NLP 도구들이 현대 언어로 작성된 텍스트에 최적화되어 있는 반면, 이 연구에서는 상용 Large Language Models (LLMs)를 사용하여 NER 성능을 크게 향상시킬 수 있음을 보여줍니다. NER에 대한 재정의가 필요하며, 역사적 맥락과 약간의 Persona 모델링을 포함한 것이 핵심 전략임을 주장합니다.

- **Technical Details**: NER은 역사 연구의 기초적인 작업으로, 다양한 언어, 장르 및 구조를 가진 역사적 문서들의 데이터에 적합한 도구가 부족합니다. 본 연구에서는 1921년 베데커 가이드에서 수동으로 주석을 단 명명된 개체를 기준으로 하여 LLM의 다양한 프롬프트 전략을 평가하였습니다. 프롬프트 전략으로는 맥락 정보 제공과 Persona 모델링을 통해 LLM이 순수한 언어적 접근에서 벗어나도록 유도하고, 무작위 예시의 수를 늘리는 방법 등이 포함되었습니다.

- **Performance Highlights**: 연구 결과, LLM은 컨텍스트 정보와 Persona 모델링이 포함된 프롬프트를 사용할 때 전통적인 NER 프레임워크인 flair와 spaCy를 최소한 동등하게 수행하며, 이에 대한 정확도가 현저히 향상됨을 보여줍니다. 더불어, 예상과 달리 무작위 예시가 없는 제로 샷 접근법이 피처 수 16개 미만에서는 몇 개의 예시가 있는 접근법보다 더 나은 성과를 보였습니다. 이러한 결과는 역사 연구자들이 기존 코드를 사용하지 않고도 NER에 접근할 수 있도록 해줍니다.



### CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidanc (https://arxiv.org/abs/2502.04350)
Comments:
          27 pages, 12 figures

- **What's New**: 기존 방법들은 Large Language Models (LLMs)를 텍스트 추론과 코드 생성 사이에서 효과적으로 조정하는 데 실패하였습니다. 이 연구에서는 LLM 코드 및 텍스트 생성을 안내하기 위해 CodeSteer라는 효과적인 방법을 제시합니다. 또한 37개의 상징적(Symbolic) 작업을 포함하는 종합 벤치마크인 SymBench를 구축하고, 12,000개의 다중 라운드 가이던스/생성 궤적 및 5,500개의 가이던스 비교 쌍으로 이루어진 데이터셋을 합성하였습니다.

- **Technical Details**: Llama-3-8B 모델은 새롭게 설계된 다중 라운드 감독 학습(supervised fine-tuning, SFT)과 직접 선호 최적화(direct preference optimization, DPO)를 통해 세밀하게 조정되었습니다. 최종적으로 생성된 모델인 CodeSteerLLM은 제안된 상징적 및 자기 답안 확인 도구를 추가하여 대형 모델의 코드/텍스트 생성을 효과적으로 안내합니다. 이 방식은 GPT-4o에 적용되었으며, 기존의 LLM을 능가하는 성능을 보여줍니다.

- **Performance Highlights**: CodeSteer를 통해 GPT-4o의 평균 성능 점수가 53.3에서 86.4로 상승하였고, 이는 OpenAI o1(82.7), o1-preview(74.8), DeepSeek R1(76.8)보다 모든 37개 작업에서 뛰어난 결과입니다. 게다가 CodeSteer는 학습된 GPT-4o에 대해 Claude, Mistral, GPT-3.5에서 평균 41.8의 성능 향상을 보여주며, 복잡한 작업에서도 상징적 컴퓨팅 기능을 완전히 활용합니다. 모델, 데이터셋, 코드는 제공된 링크에서 확인 가능합니다.



### Dynamic benchmarking framework for LLM-based conversational data captur (https://arxiv.org/abs/2502.04349)
- **What's New**: 이 논문은 대화형 에이전트를 평가하기 위한 동적 벤치마킹 프레임워크를 제안합니다. 기존의 평가 프레임워크가 단일 작업에 집중하는 반면, 이 연구는 다중 턴 대화의 역동적인 특성을 포착할 수 있도록 설계되었습니다. 이 프레임워크는 합성 사용자와의 상호작용을 통해 LLM 기반의 대화형 에이전트를 평가합니다.

- **Technical Details**: 제안된 프레임워크는 정보 추출(information extraction), 맥락 인식(context awareness), 적응형 참여(adaptive engagement) 등 핵심 차원에서 성능을 평가하기 위해 생성 에이전트 시뮬레이션(generative agent simulation)을 통합합니다. 다양한 사용자 행동의 측면을 시뮬레이션함으로써, 이 연구는 확장 가능하고 자동화된 벤치마킹 접근 방식을 제공합니다. 대출 신청(use case) 예제에서 실험 평가가 수행되었습니다.

- **Performance Highlights**: 실험 결과는 적응형 전략이 특히 모호한 응답을 처리할 때 데이터 추출 정확도를 향상시킨다는 것을 보여줍니다. 한 번의 추출(one-shot) 및 몇 번의 추출(few-shot) 조건에서 프레임워크의 효과가 입증되었습니다. 이 연구는 LLM 기반 대화형 에이전트를 평가하는 구조적이고 확장 가능한 접근 방식을 제공하여 실제 배포를 촉진합니다.



### Prompt-based Depth Pruning of Large Language Models (https://arxiv.org/abs/2502.04348)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구에서는 동적 깊이 가지치기(dynamic depth pruning)를 활용한 새로운 알고리즘인 PuDDing(Prompt-routed Dynamic Depth Pruning)을 제안합니다. 이 알고리즘은 특정 입력 프롬프트(prompt)에 기반하여 모델에서 어떤 Transformer 블록을 생략할지를 결정합니다. 실험 결과, PuDDing은 기존의 정적 깊이 가지치기 방법들보다 더 나은 성능을 발휘하며, 특정 작업에서의 정확도를 개선할 수 있습니다.

- **Technical Details**: 기존의 깊이 가지치기(depth pruning)는 고정된 생략 세트를 기반으로 하며, 이는 여러 작업에 맞게 조정할 수 없다는 한계가 있었습니다. PuDDing은 경량 라우터(router)를 훈련시켜 다양한 프롬프트에 따라 최적의 생략 세트를 선택하게 합니다. 이 과정에서는 데이터 중심의 방식으로 생략 세트를 구성하고, 새로운 작업 중심의 손실 함수(task-centric loss)를 사용하여 손실을 최소화하는 세트를 찾아냅니다.

- **Performance Highlights**: PuDDing은 제로샷(zero-shot) 상식 추론(common sense reasoning) 작업에서 4%p 이상의 정확도 향상을 달성했습니다. 이 알고리즘은 매 프롬프트마다 한 번만 라우터를 사용하여, 밀집 모델(dense model) 대비 1.2배 이상 속도 향상을 이루었습니다. 또한, PuDDing은 다양한 작업에서 더 향상된 정확도를 제공하면서도 연산 효율성 측면에서도 경쟁력을 갖추고 있습니다.



### SCALM: Detecting Bad Practices in Smart Contracts Through LLMs (https://arxiv.org/abs/2502.04347)
Comments:
          7 pages

- **What's New**: 이 논문에서는 스마트 계약(Smart Contract)에서 발생할 수 있는 나쁜 코드 관행(bad practices)을 체계적으로 분석하고, 35가지 유형의 문제를 다루는 첫 번째 연구 결과를 제시합니다. 새로운 프레임워크인 SCALM(Smart Contract Audit Language Model)을 통해 효과적으로 나쁜 관행을 탐지하고 해결할 수 있는 방법을 제안합니다. SCALM은 리트리벌 증강 생성(RAG) 및 스텝-백 프롬프팅(Step-Back Prompting) 기법을 결합하여 고급 개념을 추출하고, 상세한 감사 보고서를 생성합니다.

- **Technical Details**: SCALM은 두 가지 주요 모듈로 구성되어 있습니다. 첫 번째 모듈은 정적 분석(static analysis)을 사용하여 나쁜 관행이 포함된 코드 블록을 추출하고 이를 벡터로 변환하여 벡터 데이터베이스에 저장합니다. 두 번째 모듈은 RAG 방법론을 도입하고, 스텝-백 프롬프팅을 통해 코드에서 추상적이고 고수준의 개념을 추출하여 감사 보고서를 생성합니다. 이 과정에서 39,904개의 스마트 계약 데이터를 활용하여 잠재적 보안 위험이 있는 코드 스니펫을 수집합니다.

- **Performance Highlights**: SCALM은 다양한 LLMs 및 데이터 세트를 사용하여 수행된 실험에서 기존 도구들보다 나쁜 계약 관행을 탐지하는 능력에서 우수한 성능을 보여줍니다. 특히, RAG 컴포넌트는 SCALM의 전체 성능 향상에 중요한 역할을 했습니다. 이 프레임워크의 성과는 스마트 계약의 감사 작업에 있어 실질적인 개선을 제공할 것으로 기대됩니다.



### Multi-Lingual Cyber Threat Detection in Tweets/X Using ML, DL, and LLM: A Comparative Analysis (https://arxiv.org/abs/2502.04346)
- **What's New**: 이번 연구는 다언어 트윗 내 사이버 위협 감지에 중점을 두고 기존의 단일 언어별 접근 방식의 한계를 극복하고자 했습니다. 연구진은 영어, 중국어, 러시아어, 아랍어를 포함한 4개의 언어에서 데이터셋을 수집하고 라벨링하여, 다양한 고급 모델을 적용하는 새로운 방법론을 탐구했습니다. 결과적으로 Bi-LSTM 아키텍처가 모든 데이터셋에서 우수한 성능을 보여주며 다언어 사이버 위협 감지의 효과성을 입증했습니다.

- **Technical Details**: 연구는 세 단계로 진행되었으며, 첫 번째 단계에서 4개의 언어의 트윗 데이터를 수집하고 수작업 및 극성 기반 라벨링 방법을 통해 고품질 주석을 확보했습니다. 두 번째 단계에서는 각 데이터셋을 개별적으로 분석하여 기계 학습(ML) 및 심층 학습(DL) 모델의 성능을 평가했습니다. 세 번째 단계에서는 모든 데이터셋을 결합한 다언어 데이터셋을 만들어 DL 및 대형 언어 모델(LLM) 아키텍처를 적용하여 사이버 위협 탐지의 효능을 평가하였습니다.

- **Performance Highlights**: Machine Learning 모델 중에서는 Random Forest(RF)가 가장 높은 성능을 달성했지만, Bi-LSTM 아키텍처가 모든 데이터셋에서 다른 DL 및 LLM 아키텍처를 일관되게 초과하는 성능을 보였습니다. 이러한 결과는 Bi-LSTM이 다언어 사이버 위협 감지에 효과적임을 강조합니다. 본 연구에서 개발한 코드와 데이터는 연구자들이 사이버 위협 감지 문제를 해결하는 데 기여할 것으로 기대합니다.



### JingFang: A Traditional Chinese Medicine Large Language Model of Expert-Level Medical Diagnosis and Syndrome Differentiation-Based Treatmen (https://arxiv.org/abs/2502.04345)
- **What's New**: 이번 연구에서는 JingFang (JF)라는 새로운 TCM(Traditional Chinese Medicine) 대형 언어 모델을 개발하였습니다. JF는 전문가 수준의 진단 능력과 증상 구별 기반 치료를 제공하는 혁신적인 모델로, 기존 TCM 모델의 한계를 극복하고자 합니다. 연구팀은 의학 상담을 위한 다중 에이전트 동적 협력 사고 체계(MDCCTM)를 혁신하여 JF의 정확한 진단 및 치료 능력을 강화했습니다.

- **Technical Details**: JF 프레임워크는 세 가지 주요 모듈로 구성됩니다: TCM 상담, TCM 증상 구별, TCM 치료 추천입니다. JF는 MDCCTM을 통해 동적인 추론과 명확한 의사결정 기능을 구현하며, DSR(이중 단계 검색 체계)을 통해 실제 응용에 필요한 개선된 증상 구별 능력을 보유하고 있습니다. 각 에이전트는 환자의 상태에 따라 맞춤형 정보를 수집하고 분석하여 정확한 증상 구별과 치료 추천을 수행합니다.

- **Performance Highlights**: JF는 기존 TCM 모델에 비해 의료 상담의 완전성 및 정밀성을 크게 개선하여, 실제 환자 치료에서의 실용성을 강화하였습니다. 다수의 전문가 에이전트가 협력해 개인화된 의학적 접근을 제공하고, 효율적인 다단계 상담 과정을 거쳐 증상 구별과 치료 추천의 정확도를 높였습니다. JF는 전통 한의학의 현대적 적용을 가능하게 하여 인류 건강 보호 및 질병 치료에서 중요한 기여를 할 것으로 기대됩니다.



### Tutorial on Using Machine Learning and Deep Learning Models for Mental Illness Detection (https://arxiv.org/abs/2502.04342)
- **What's New**: 이 논문에서는 소셜 미디어를 통한 정신 건강 이해를 위한 기계 학습(machine learning) 및 심층 학습(deep learning) 방법에 관한 실용적인 가이드를 제공합니다. 연구자들이 우울증을 조기에 감지할 수 있도록 다양한 데이터 세트 처리 및 모델 평가 문제를 다루며, 실제 사례를 통해 이러한 기술을 효과적으로 적용하는 방법을 설명합니다. 또한 투명하고 윤리적인 기술 사용의 중요성을 강조하며, 신뢰할 수 있는 모델 구축을 위한 단계를 제시합니다.

- **Technical Details**: 이 연구는 데이터 준비, 모델 개발, 평가 지표에 대한 포괄적인 방법론을 제공합니다. Python 3와 pandas, scikit-learn, PyTorch, Transformers와 같은 인기 있는 라이브러리를 사용하여 데이터 처리 및 모델링을 수행하였습니다. 다양한 플랫폼에서 수집된 데이터를 기반으로 정신 건강 주제를 다루는 Sentiment Analysis for Mental Health 데이터 세트를 활용하여, 텍스트 청소, 정규화 및 TF-IDF 기반의 벡터화 과정 등을 통해 데이터 준비가 이루어졌습니다.

- **Performance Highlights**: 논문에서는 로지스틱 회귀모델을 포함하여 다양한 기계 학습 및 심층 학습 모델을 사용하여 정신 건강 상태를 분석하고 분류하였습니다. 각 모델은 데이터의 특정 측면을 탐색하기 위해 선택되었으며, 모델의 정확성 및 해석 가능성을 확보하기 위한 하이퍼파라미터 튜닝과 평가가 이루어졌습니다. 본 연구는 바람직한 성과를 달성하기 위해 여러 모델의 구현 코드와 자세한 성과를 GitHub에 공개할 예정입니다.



### Comparative Analysis of Community Detection Algorithms on the SNAP Social Circles Datas (https://arxiv.org/abs/2502.04341)
Comments:
          Presented at IDEA2k24: this https URL Submitted to Springer Lecture Notes in Electrical Engineering series (this https URL)

- **What's New**: 이번 연구에서는 Facebook의 SNAP Social Circles 데이터 세트를 활용하여 여러 주요 커뮤니티 탐지(Community Detection) 알고리즘을 비교 분석하고 있습니다. Louvain, Girvan-Newman, Spectral Clustering, K-Means Clustering 등의 알고리즘이 포함되며, 이들의 성능을 모듈라리티(modularity), 정규화 컷 비율(normalized cut-ratio), 실루엣 점수(silhouette score), 응집력(compactness), 분리성(separability) 등을 기준으로 평가합니다.

- **Technical Details**: 네트워크(그래프)는 노드(버텍스)와 엣지(링크)로 구성되어 있으며, 각각은 상호작용 관계를 나타냅니다. 커뮤니티 혹은 클러스터는 밀접하게 연결된 노드의 집합으로, 외부와는 드물게 연결되어 있습니다. 알고리즘은 이러한 커뮤니티를 식별하여 네트워크 내의 구조적 패턴을 드러내는 역할을 합니다.

- **Performance Highlights**: 각 알고리즘의 성능은 다양한 지표를 통해 비교되고 있으며, 이러한 결과는 소셜 네트워크 내에서 의미 있는 커뮤니티를 탐지하는 데 있어 알고리즘의 강점과 한계를 조명합니다. 연구 결과는 커뮤니티 탐지 방법에 대한 이해를 높이고, 실제 사회 네트워크 데이터를 분석하는 데 있어 가치 있는 지침을 제공할 것입니다.



### Predicting Steady-State Behavior in Complex Networks with Graph Neural Networks (https://arxiv.org/abs/2502.01693)
Comments:
          13 pages, 7 figures

- **What's New**: 이 연구는 복잡한 네트워크에서 선형 동적 시스템의 거동을 학습하기 위해 그래프 신경망(Neural Network, GNN) 모델을 적용했습니다. 기존의 정보를 전파하는 세 가지 상태, 즉 분산, 약한 국소화 및 강한 국소화를 정의하고, 이를 통해 다양한 상태를 높은 정확도로 구별하는 모델을 개발했습니다. 본 연구는 실제 데이터셋을 이용하여 모델의 성능을 평가하며, 모델의 이해 가능성을 위한 분석적 유도 과정을 제공합니다.

- **Technical Details**: 복잡한 네트워크와 관련된 선형 동적 과정을 정의하면서, 시스템의 초기 상태를 설정하고 최종 행동(steady-state behavior)을 분석합니다. GNN은 주어진 네트워크 구조를 입력으로 받아들여 각 상태에 따른 IPR(Inverse Participation Ratio) 값을 예측하는 방식으로 작동합니다. 결과적으로 GNN 아키텍처는 다양한 크기의 네트워크에 대해 효과적으로 작동하며, 크기가 작은 네트워크에서 훈련받더라도 큰 네트워크로 일반화할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 모델의 성능은 다양한 상태를 구별하는 데 탁월하며, 실제 데이터셋에 대한 평가를 통해 그 유효성을 입증하였습니다. GNN의 학습 효율성과 일반화 능력 덕분에 기존의 방법보다 더 나은 결과를 얻을 수 있었습니다. 이러한 결과는 실제 복잡한 시스템에서의 정보 전파 행동에 대한 깊은 통찰력을 제공합니다.




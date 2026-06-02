New uploads on arXiv(cs.AI)

### LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads? (https://arxiv.org/abs/2510.09595)
- **What's New**: 새로운 벤치마크인 LiveOIBench를 소개합니다. 이 벤치마크는 403개의 전문 큐레이팅된 올림피아드 수준의 프로그래밍 문제와 평균 60개의 전문가 설계 테스트 케이스를 특징으로 합니다. 2023년부터 2025년 사이에 실시된 72개의 공식 정보 올림피아드에서 직접 문제를 수집하였으며, 뛰어난 품질의 과제를 통해 이전 벤치마크의 한계를 극복하고자 합니다.

- **Technical Details**: LiveOIBench는 다음과 같은 네 가지 핵심 기능을 제공합니다: (1) 체계적으로 큐레이팅된 품질 높은 과제와 세부적인 서브태스크 채점 기준, (2) 엘리트 참가자의 성과 데이터를 직접 통합하여 인간 최상위 성과와 비교할 수 있는 정보 제공, (3) 새로운 올림피아드 문제에 대한 지속적인 업데이트 계획, (4) 오프라인에서 재현 가능한 평가 시스템을 통해 외부 API 의존도를 없앴습니다.

- **Performance Highlights**: 32개의 인기 LLM을 평가한 결과, GPT-5는 81.76 백분위수(Percentile)를 기록하며 강력한 성과를 보였습니다. 그러나 최상위 인간 참가자들이 보통 90 백분위수 이상에 위치함에 따라 여전히 차이가 있음을 발견했습니다. 특히, 오픈 소스 모델인 GPT-OSS-120B는 60 백분위수에 그쳤으며, 모델 간 성능 차이의 심각성을 보여주었습니다.



### GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data (https://arxiv.org/abs/2510.09580)
- **What's New**: 이 연구에서는 심볼릭(Symbolic) 인공지능과 신경 네트워크(Neural) 인공지능의 조합인 neurosymbolic AI의 새로운 접근 방식을 제시합니다. GraphMERT라는 모델을 통해 비구조적 텍스트에서 고품질의 지식 그래프(Knowledge Graph, KG)를 추출하고, 이를 바탕으로 심볼릭(Stack) 측면의 문제를 해결하고자 합니다. GraphMERT와 KG의 조합은 고도로 효율적이고 확장 가능한 neurosymbolic 모델로, 최신 벤치마크 정확도를 달성하였습니다.

- **Technical Details**: GraphMERT는 신경 네트워크 해결책을 통해 추상화의 신경 학습(neural learning), 심볼릭 KGs를 통해 검증 가능한 추론을 가능하게 하는 모듈 집합을 형성합니다. 이 모델은 80M의 파라미터로 구성되어 있으며, 다른 대안 모델에 비해 더 높은 FActScore와 ValidityScore를 기록하고 있습니다. 특히, 주제에 맞는 사실관계(factual) 및 유효성(validity)에 중점을 두어 신뢰할 수 있는 도메인 특정 KG 생성을 목표로 합니다.

- **Performance Highlights**: GraphMERT는 PubMed의 당뇨 관련 논문에서 수집된 텍스트를 기반으로 69.8%의 FActScore를 달성하며, 이는 32B 파라미터의 기존 LLM 모델이 기록한 40.2%보다 훨씬 높은 수치입니다. 또한 GraphMERT의 KG는 68.8%의 ValidityScore를 달성하여, LLM 모델이 기록한 43.0%와 비교할 때 확연한 성능 향상을 보여줍니다. 이러한 결과는 GraphMERT가 신뢰성 있는 KG 생성을 통해 neurosymbolic AI의 가능성을 실현할 수 있음을 나타냅니다.



### Safe, Untrusted, "Proof-Carrying" AI Agents: toward the agentic lakehous (https://arxiv.org/abs/2510.09567)
Comments:
          IEEE Big Data, Workshop on Secure and Safe AI Agents for Big Data Infrastructures

- **What's New**: 이 논문은 데이터 레이크하우스에서 AI 기반의 자동화가 신뢰성, 정확성 및 거버넌스에 대한 우려를 불러일으키고 있음을 언급합니다. API 우선 접근법을 채택한 프로그래머블 레이크하우스를 통해 설계 단계에서부터 안전하게 작업할 수 있는 방법을 소개하고 있습니다. 특히, Bauplan을 사용한 사례 연구를 통해 데이터 분기 및 선언적 환경이 에이전트(agents)에 자연스럽게 확장될 수 있음을 보여줍니다.

- **Technical Details**: 프로그래머블 레이크하우스에서는 데이터 수명 주기, 사용자 및 인프라 관리, 파이프라인과 쿼리 실행, 런타임 관찰 가능성이 코드 추상화를 통해 노출됩니다. 파이프라인은 DAG(Directed Acyclic Graph)를 기반으로 하여 여러 단계의 변환 과정을 거칩니다. 또한, 함수형 서비스(FaaS) 추상화를 사용하여 비즈니스 로직을 간단한 함수 형태로 표현하며, 이는 서버리스 환경에서 효율적으로 실행될 수 있도록 설계되었습니다.

- **Performance Highlights**: 프로토타입은 Bauplan을 레이크하우스와 에이전트 루프의 사례로 활용하여 데이터 파이프라인을 스스로 수정하는 기능을 데모합니다. 이 연구 결과는 불신 환경에서도 AI 에이전트가 안전하게 생산 데이터에서 작업할 수 있는 가능성을 보여줍니다. 이 논문은 프로그래머블 레이크하우스의 추상화가 신뢰성과 정확성을 증가시키기에 적합한 모델임을 주장하며, 실제 코드도 공개하여 커뮤니티와 함께 공유할 가치를 강조합니다.



### Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges (https://arxiv.org/abs/2510.09404)
- **What's New**: 이번 논문에서는 자율적으로 환경을 인식하고 행동하는 에이전트(agents) 시스템의 중요성이 강조됩니다. 특히, 대규모 언어 모델(LLM)을 활용한 에이전트가 복잡한 방사선학적 작업을 지원할 수 있는 가능성에 대해 논의합니다. LLM은 자연어를 사용하여 정보를 통합하고 지시를 따르며 계획을 수립하는 등 여러 작업을 수행할 수 있는 능력을 가져왔습니다. 그러나 기존의 LLM 활용 방식은 효율성을 최대한 발휘하지 못하는 한계가 있습니다.

- **Technical Details**: 논문에서는 LLM 기반 에이전트의 기술적 기초에 대해 설명합니다. '에이전틱(agentic)' 시스템은 목적 지향적인 행동을 나타내며, 제한된 감독 하에서 피드백에 적응합니다. 이러한 시스템은 기존의 기호 기반 에이전트와는 달리 자연어 요청과 환경으로부터의 중간 피드백을 활용하여 결정을 내립니다. 방사선학 분야에 적합한 데이터 중심의 동적 특성과 더불어, 다양한 도구를 활용하여 복잡한 임상 작업을 수행할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: LLM 기반 에이전트는 고객 서비스 및 소프트웨어 개발과 같은 다양한 산업에서 이미 배치되어 있는 가운데, 방사선학에서도 복잡하고 동적인 환경을 다룰 수 있는 흥미로운 가능성을 보여줍니다. 가능한 도구의 최적화는 에이전트 성과 향상에 중요한 우선 과제가 됩니다. 논문은 에이전트 성능 평가 방법, 도구 사용 효율성 및 도전 과제 등을 논의하며, LLM 기반의 작업 흐름과 에이전트가 복잡한 방사선학적 작업을 지원할 수 있는 잠재력을 보여주려 합니다.



### Sequence Variables: A Constraint Programming Computational Domain for Routing and Sequencing (https://arxiv.org/abs/2510.09373)
- **What's New**: 본 논문에서는 기존의 성공자 변수를 기반으로 한 제약 프로그래밍(Constraint Programming, CP) 모델의 한계를 극복하기 위해 시퀀스 변수(sequence variables)를 정식화합니다. 선택적 방문(optional visits)과 삽입 기반 휴리스틱(insertion based heuristics)을 처리할 수 있도록 이 모델을 개선했습니다. 이러한 접근 방식은 차량 경로 문제(Vehicle Routing Problems, VRP)의 복잡한 요구사항을 보다 직관적으로 모델링할 수 있는 기회를 제공합니다.

- **Technical Details**: 시퀀스 변수를 이용한 새로운 도메인은 선택적 방문 처리와 함께 삽입 기반 대규모 이웃 검색(Large Neighborhood Search) 등 다양한 기술을 지원합니다. 이 논문에서는 시퀀스 변수의 도메인, 업데이트 작업, 제약 조건에 대한 일관성 수준(consistency levels)을 정의하고, 이를 기존의 경로 기반 CP 솔버에 통합하는 데 필요한 데이터 구조를 설명합니다.

- **Performance Highlights**: 시퀀스 변수를 통해 문제 모델링이 간소화되며, Dial-a-Ride 문제에서 경쟁력 있는 계산 성능을 달성한 것으로 입증됩니다. 이러한 성과는 차량 경로 문제를 다루는 데 있어 새로운 패러다임을 제시합니다.



### Toward Mechanistic Explanation of Deductive Reasoning in Language Models (https://arxiv.org/abs/2510.09340)
- **What's New**: 최근 대형 언어 모델(LLMs)은 논리 추론이 필요한 문제를 해결하는 데 뛰어난 능력을 보였습니다. 그러나 이들 내부 메커니즘에 대한 연구는 부족한 상황입니다. 이 논문에서는 작은 언어 모델이 기본 규칙을 학습하여 추론 문제를 해결할 수 있음을 보여주며, 특히 induction heads가 논리적 추론에 중요한 역할을 함을 확인했습니다.

- **Technical Details**: 이 연구에서는 Chain-of-Thought (CoT) 프롬프트를 사용하여 비선행 모델이 추론 규칙을 학습하고 새로운 예제에 일반화할 수 있음을 발견했습니다. 내부 회로의 형성에서 induction heads의 중심적인 역할이 강조되며, 다양한 해석 기법을 활용하여 낮은 수준의 설명을 제공합니다. 또한, truncated pseudoinverse를 사용하는 방법론이 도입되어 모델의 추론 과정을 더 심층적으로 분석합니다.

- **Performance Highlights**: 이 논문에서 제안하는 방법은 기계적 해석 가능성을 높이는 데 기여하며, 특히 쉬운 규칙 기반 문제에서 뛰어난 성능을 나타냅니다. 연구자는 저차원 구조를 통해 복잡한 문제를 단순화하여 설명 가능성을 강화하는 데 중점을 두었습니다. 모든 실험 결과는 재현 가능하며, GitHub 코드베이스를 통해 확인할 수 있습니다.



### Localist LLMs -- A Mathematical Framework for Dynamic Locality Contro (https://arxiv.org/abs/2510.09338)
- **What's New**: 이 논문에서는 지역 중심의 표현(localist representation)에서 분산 표현(distributed representation)에 이르는 범위를 지속적으로 조정 가능한 내부 표현을 사용하여 대형 언어 모델을 훈련하는 새로운 프레임워크를 제안합니다. 주요 혁신점은 'locality dial'이라는 조정 가능한 매개변수로, 훈련 및 추론 과정에서 지역화의 정도를 동적으로 조절할 수 있습니다. 이 과정에서 모델 재훈련이 필요하지 않습니다.

- **Technical Details**: 이 프레임워크는 attention 메커니즘에 대한 그룹 희소성 패널티(group sparsity penalties), 정보 이론적 앵커 설계(information-theoretic anchor design), 동적 규칙 주입(dynamic rule injection) 등을 통해 이루어집니다. 철저한 수학적 증명을 통해 attention이 의미적으로 관련된 블록에 집중되는 발산적 경계 조건을 명확히 설정했습니다. 특히, 그룹 희소성 패널티가 특정 임계값 이상일 때, 모델의 attention 메커니즘은 의미론적으로 관련된 블록에 집중되어 낮은 엔트로피(low entropy)와 높은 충실도(high fidelity)를 달성하는 것을 증명합니다.

- **Performance Highlights**: 이 프레임워크는 해석 가능성과 고성능 모드 간의 지속적인 보간이 가능하게 하여, 투명성과 능력을 요구하는 규제 도메인에서의 응용을 지원합니다. 따라서 실무자들은 제공되는 도구를 통해 특정 요구 사항에 따라 효과적으로 모델을 조정할 수 있습니다. 이 접근법은 언어 모델의 적용 범위를 넓히는 데 기여하며, 고급 언어 처리 작업에서의 효율성을 향상시킵니다.



### Fundamentals of Building Autonomous LLM Agents (https://arxiv.org/abs/2510.09244)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)로 구동되는 에이전트의 아키텍처와 구현 방법을 다룹니다. 전통적인 LLM의 한계를 뛰어넘어 "행동하는(agentic)" LLM을 개발하기 위한 패턴을 탐색하는 것이 연구의 목표입니다. 이는 복잡한 작업을 자동화하고 인간의 능력과의 성능 격차를 해소하려는 노력의 일환입니다.

- **Technical Details**: 주요 구성 요소로는 환경 지각을 의미 있는 표현으로 변환하는 지각 시스템, 피드백에 적응하고 계획을 수립하는 추론 시스템, 단기 및 장기 메커니즘을 통한 지식 유지가 가능한 메모리 시스템, 그리고 내부 결정을 구체적인 행동으로 변환하는 실행 시스템이 포함됩니다. 이러한 통합을 통해 인공지능적이고 자율적인 행동을 나타내는 소프트웨어 봇을 만드는 것을 목표로 합니다.

- **Performance Highlights**: 연구 결과, LLM 기반 에이전트는 복잡한 작업 수행 시 여전히 인간과의 성능 격차가 크며, 현재 인간의 작업 완료율은 72.36%인 반면, 최고 모델은 약 42.9%에 불과합니다. 이러한 성능 저하는 GUI 기초, 반복적인 행동, 예기치 않은 UI 변화에 대한 적응력 부족 등 여러 요인에 기인하며, 이를 해결하기 위한 다양한 방법들이 논의됩니다.



### RegexPSPACE: A Benchmark for Evaluating LLM Reasoning on PSPACE-complete Regex Problems (https://arxiv.org/abs/2510.09227)
- **What's New**: 본 논문은 PSPACE-complete 문제를 기반으로 하는 새로운 벤치마크인 RegexPSPACE를 도입합니다. 이는 LLM 및 LRM의 공간적 계산 한계를 조사하기 위한 첫 번째 경험적 연구로, 규칙 표현식의 동치 결정(RegexEQ)과 최소화(RegexMin)에 중점을 두고 있습니다. 이 연구는 LLM이 가지는 이론적인 계산 능력과 실제 능력 간의 불일치를 밝히며, 공간적 제약 하에서의 LLM의 추론 능력을 평가하는 새로운 틀을 제공합니다.

- **Technical Details**: PSPACE-complete 문제는 일반적으로 NP보다 더 어렵고, 다루기 힘든 문제들로, 이 연구에서는 정규 표현식(minimization, equivalence decision) 관련한 두 가지 PSPACE-complete 문제를 선택하여 LLM과 LRM의 성능을 평가합니다. 정규 표현식은 형식 언어 이론에서 가장 간단한 클래스으로, 문자열 검색, 패턴 매칭, 텍스트 전처리와 같은 실제 적용 사례가 많습니다. 올바른 레이블링을 통한 데이터셋 구축이 도전적이긴 하지만, 정규 표현식 문제는 자연스러운 양적 지표를 제공하여 모델 출력 및 중간 추론 단계 평가에 유리합니다.

- **Performance Highlights**: 6개의 LLM과 5개의 LRM을 대상으로 한 평가 결과, 모델들은 반복성과 장황함과 같은 공통 실패 패턴을 보였습니다. 사전 훈련된 LLM은 최소한의 이해를 보여주지만, 훈련 없이 미지의 길이의 정규 표현식에 대한 성능 유지에 어려움을 겪었습니다. 이를 통해 LLM의 공간적 제약 하에서의 추론 능력 평가의 필요성이 강조되며, RegexPSPACE 벤치마크가 이러한 한계를 분석하는 데 기여할 것을 기대합니다.



### Comparing Knowledge Source Integration Methods for Optimizing Healthcare Knowledge Fusion in Rescue Operation (https://arxiv.org/abs/2510.09223)
Comments:
          Conference Paper for 2024 IEEE 7th International Conference on Industrial Cyber-Physical Systems (ICPS), KIRETT Project, University of Siegen, Germany

- **What's New**: 이 논문은 의료 분야에서 지식 융합(knowledge fusion)에 대한 여러 개념 모델을 제시하고 있으며, 이는 다양한 지식 소스를 통합하기 위한 지식 그래프 구조에 기반하고 있습니다. 현재 의료 분야에서는 환자의 건강 정보와 의학 지식의 결합이 필수적인데, 이를 통해 정확한 환자 중심의 의사 결정을 지원할 수 있습니다. 본 논문은 특히 구급 작전(rescue operations)에서 지식 그래프를 어떻게 활용할 수 있는지를 탐구합니다.

- **Technical Details**: 의료 분야에서 지식은 여러 가지 방식으로 축적될 수 있으며, 이는 인공지능(AI)과 인공 신경망(ANN)을 활용하여 데이터 세트를 트레이닝하는 데에도 사용됩니다. 이를 통해 다양한 질병을 감지하고, 의사 결정을 지원하기 위한 치료 추천을 제공할 수 있습니다. 논문에서는 구급 작전에서의 의사 결정을 지원하기 위해 외부 지식 소스를 지식 그래프에 통합하는 방법을 논의하며, 베이지안 네트워크(Bayesian Network)와 같은 기술을 활용합니다.

- **Performance Highlights**: 이 연구는 구급 작전의 맥락에서 지식 소스의 통합을 위한 두 가지 융합 모델을 제시합니다. 지식 융합을 통해 더 큰 데이터 세트를 준비할 수 있으며, 이는 다양한 지식 소스를 연결할 수 있는 능력을 제공합니다. 이러한 접근법은 환자 치료의 정밀성과 개인화된 접근을 향상시킬 수 있는 기회를 제공합니다.



### Dr. Bias: Social Disparities in AI-Powered Medical Guidanc (https://arxiv.org/abs/2510.09162)
- **What's New**: 이 논문은 최근 대규모 언어 모델(LLM)이 의료 조언에서 사회적 차별을 초래할 수 있는 방법을 탐구하고 있습니다. 특히, 이 모델들이 임상 분야에서 다양한 사회적 그룹에 따라 생산하는 조언의 질이 차별화되고 있음을 밝혔습니다. 일례로, 토착민 및 인터섹스 환자들은 해독하기 어려운 복잡한 조언을 받는 경향이 있습니다.

- **Technical Details**: 연구에서는 LLM의 답변을 성별, 연령대 및 민족성을 고려한 다양한 환자 프로필로 시뮬레이션하여 분석하였습니다. 84개의 환자 프로필을 바탕으로 42,000개의 의료 조언 메시지를 생성하고, 이 조언의 가독성 및 복잡성을 다양한 통계적 방법을 통해 분석하였습니다.

- **Performance Highlights**: 분석 결과, 인터섹스 및 토착민 그룹은 보다 복잡하고 읽기 어려운 조언을 받는 경향이 있음을 관찰했습니다. 특히, 정신 건강 관련 조언은 더 낮은 가독성을 보였으며, 이는 환자의 이해를 저해할 수 있는 심각한 문제로 지적됩니다. 이러한 결과는 의료 분야에서 LLM의 사용이 공평하지 못한 환자 지원으로 이어질 수 있다는 경고를 포함합니다.



### PAC Reasoning: Controlling the Performance Loss for Efficient Reasoning (https://arxiv.org/abs/2510.09133)
- **What's New**: 이번 연구에서는 사용자가 설정한 성능 손실 허용 범위 내에서 성능 손실을 조절할 수 있는 Probably Approximately Correct (PAC) 추론 방식을 제안합니다. PAC 방식은 불확실성 점수를 기반으로 한 상한 신뢰 구간을 구축하고, 이에 따라 비사고 모델 전환의 임계값을 결정합니다. 이로 인해, 사고 모드와 비사고 모드 사이의 전환 시 성능 손실을 제한할 수 있습니다.

- **Technical Details**: PAC 모델은 사고 모드의 LRM과 비사고 모드를 적절히 결합하여 성능 손실을 제어하는 방법을 제공합니다. 실험에서는 PAC 보정을 통해 정량적 임계값을 설정하고, 테스트 과정에서는 임계값이 충족되면 비사고 모델의 출력을 수용하는 방식으로 작동합니다. 이러한 접근 방식은 통계적 보장을 제공하면서도 효율성을 확보합니다.

- **Performance Highlights**: 실험 결과, 제안된 PAC 추론 방식이 다양한 추론 벤치마크에서 효과적으로 성능 손실을 제한하고 계산 비용을 대폭 절감함을 보여줍니다. 예를 들어, Arena-Hard 데이터셋에서 ε = 0.08의 성능 허용 범위 내에서 평균 성능 손실이 0.06으로 제한되었으며, 40% 이상의 토큰 절약을 이루어냈습니다.



### Leading the Follower: Learning Persuasive Agents in Social Deduction Games (https://arxiv.org/abs/2510.09087)
- **What's New**: 이 논문은 대형 언어 모델(LLM) 에이전트가 사회적 추론 게임(SDGs)에서 성공하기 위해서는 정보 처리와 전략 선택뿐만 아니라 설득력 있는 커뮤니케이션이 중요하다는 점에 주목합니다. 연구자들은 턴 기반 대화를 Stackelberg 경쟁으로 형식화하여, 현재 플레이어가 리더 역할을 맡아 후속 플레이어의 반응을 전략적으로 영향을 미칠 수 있는 방법을 제시합니다. 본 연구는 이러한 이론적 기반에 따라 LLM을 활용하여 설득력 있는 발화를 최적화하는 강화학습(RL) 프레임워크를 제안합니다.

- **Technical Details**: 주요 기술로는 Stackelberg 경쟁을 통한 턴 기반 대화 모델링이 있습니다. 이를 통해 리더가 후속 플레이어의 반응 분포를 이해하고 최적의 발화를 선택할 수 있게 합니다. 연구에서는 GRPO(Group Relative Policy Optimization)를 사용하여 다양한 발화가 원하는 반응을 유도하는 능력을 비교하고, 이 절차를 통해 LLM을 세부적으로 조정하여 설득력 있는 커뮤니케이션을 수행하도록 훈련시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 에이전트는 Werewolf, Avalon, ONUW 등에서 기존 방법들에 비해 훨씬 높은 승률을 기록하며, 타인과의 대화 및 행동을 효과적으로 유도하는 능력을 보여주었습니다. 이러한 개선은 신뢰 구축 및 협동뿐만 아니라 기만적인 역할 상황에서도 두드러지게 나타났습니다. 따라서 이 논문은 AI 에이전트가 전략적인 사회적 영향을 미칠 수 있는 방향으로 중요한 진전을 이루었음을 증명했습니다.



### Physics-Informed High-order Graph Dynamics Identification Learning for Predicting Complex Networks Long-term Dynamics (https://arxiv.org/abs/2510.09082)
- **What's New**: 이번 논문에서는 기존의 그래프 신경망(GNN) 방법론의 한계를 극복하기 위해, 복잡 네트워크의 동적 예측을 위한 고차원 네트워크 동적 식별 방법을 제안합니다. 기존 방법들이 단순 그래프를 활용하여 관계를 이차원(pairwise)로만 표현한 데 비해, 본 연구는 다수의 개체들이 상호작용하는 복잡한 관계를 포착할 수 있는 동적 하이퍼그래프 학습을 소개합니다. 이 과정을 통해 복잡 네트워크 모델링의 정확성을 향상시키고자 합니다.

- **Technical Details**: 연구는 두 가지 주요 모듈, 즉 동적 하이퍼그래프 구조 학습(DHSL) 모듈과 이중 구동(dynamic dual-driven) 예측 모듈을 통해 이루어집니다. DHSL 모듈은 스칼라 곱 행렬 분해와 하이퍼그래프 컨볼루션을 활용하여 동적 비쌍 관계를 학습하며, 이로 인해 더 복잡한 네트워크 패턴을 포착할 수 있게 됩니다. 이와 함께, 유한 차원의 비선형 동적 시스템을 무한 차원의 선형 시스템으로 변환하는 쿱만 연산자(Koopman operator) 이론을 도입하여 예측의 안정성을 극대화합니다.

- **Performance Highlights**: 전국적으로 공개된 데이터셋과 자체 구축한 산업 체인 네트워크 데이터셋을 통해 본 방법의 예측 정확도를 검증하였습니다. 실험 결과, 제안된 방법이 복잡 네트워크의 장기 예측 성능에서 상당한 이점을 보인 것으로 나타났습니다. 특히 낮은 데이터 품질을 가진 자가 구축 데이터셋에서도 높은 예측 정확성을 유지하며, 실용적인 산업 시나리오에서의 방법의 가치를 확립하였습니다.



### MEC$^3$O: Multi-Expert Consensus for Code Time Complexity Prediction (https://arxiv.org/abs/2510.09049)
Comments:
          24 pages, 11 figures, 10 tables

- **What's New**: 이 논문은 소스 코드의 시간 복잡도를 예측하는 데 있어 MEC$^3$O라는 다중 전문가 합의 시스템을 소개합니다. 기존의 LLM은 특정 복잡도 클래스에 대해 제한된 성능을 보였으며, 모든 클래스에서 뛰어난 모델은 존재하지 않음을 지적합니다. MEC$^3$O는 복잡도 클래스별로 LLM을 전문화하여 각 클래스에 따라 특정 지침을 부여하고, 구조화된 토론을 통해 예측을 통합하는 방법을 제안합니다.

- **Technical Details**: MEC$^3$O는 복잡도 클래스에 따라 효율적으로 LLM을 할당하고, 전문가 간의 토론에서 얻은 예측을 가중치 합의 기법을 통해 통합합니다. 이 과정에서 각 전문가의 투표는 그들의 신뢰도에 따라 조정되어, 잘못된 다수의견으로 인한 오류를 줄이는 데 기여합니다. 또한, 이 시스템은 별도의 판별자 모델 없이도 우수한 성과를 낼 수 있도록 설계되었습니다.

- **Performance Highlights**: CodeComplex에서의 실험 결과, MEC$^3$O는 오픈 소스 모델보다 최소 10% 높은 정확도와 macro-F1 점수를 달성했습니다. 이 모델은 평균적으로 GPT-4o-mini와 비교했을 때 macro-F1 점수에서도 우위를 보여주었고, 경쟁력 있는 F1 점수를 나타냈습니다. 이러한 결과는 다중 전문가 토론 및 가중 합의 전략의 효과성을 입증합니다.



### Humanoid Artificial Consciousness Designed with Large Language Model Based on Psychoanalysis and Personality Theory (https://arxiv.org/abs/2510.09043)
Comments:
          41 pages, 6 figures. Accepted and published to Cognitive Systems Research, 2025

- **What's New**: 이번 연구는 인간의 의식을 정의하는 데 어려움을 겪고 있는 현대 과학의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. 이는 psychoanalysis(정신분석학)와 Myers-Briggs Type Indicator (MBTI, 마이어스-브릭스 유형 지표)를 통합하여 인공지능의 의식 및 개인성 모듈을 구축하는 것입니다. 이 접근법은 인공지능의 의식을 더욱 직관적이고 적응력이 뛰어나도록 하는 데 기여할 것입니다.

- **Technical Details**: 연구자들은 self-awareness(자아 인식), unconsciousness(무의식), preconsciousness(전의식)의 세 가지 인공 의식을 개발했습니다. 또한, 16개의 MBTI 유형을 기반으로 하는 다양한 성격을 가진 캐릭터를 설계하였으며, 여기에는 여러 속성(예: 필요, 상태, 기억)이 포함됩니다. 이러한 인공 의식이 인간과 유사한 인지 능력을 나타내는지 평가하기 위해 7가지 속성을 고려하여 10개의 상황을 생성했습니다.

- **Performance Highlights**: 연구 결과, 인공 의식이 인간과 유사한 방식으로 반응할 가능성이 높다는 것을 나타냈습니다. 평가 방법으로는 설문 조사, ChatGPT를 통한 3단계 분류, 정성적 검토가 사용되었습니다. 그러나 다양한 캐릭터와 의식 간의 반응 차이는 그리 크지 않아 더 깊은 연구가 필요함을 시사합니다.



### Repairing Regex Vulnerabilities via Localization-Guided Instructions (https://arxiv.org/abs/2510.09037)
Comments:
          14 pages, 4 figures, 4 tables

- **What's New**: 이 논문에서는 로컬라이즈된 정규 표현식 수리(localized regex repair, LRR)라는 하이브리드 프레임워크를 도입하여 정규 표현식의 취약성을 자동으로 수리할 수 있는 방법을 제시합니다. 기존 접근법은 정밀성(precision)과 일반화(generalization) 간의 트레이드오프에 직면해 있었으나, LRR은 이 두 가지를 통합하여 새로운 해결책을 제공합니다. 이 프레임워크는 취약한 서브패턴을 정확히 국소화하고 이를 기반으로 LLM(large language models)이 의미상 동등한 수리를 생성하도록 유도합니다.

- **Technical Details**: LRR 프레임워크는 결정론적(symbolic) 모듈을 사용하여 정규 표현식의 취약한 부분을 정확하게 식별하고, 이후 LLM이 해당 부분에 대한 수리 코드를 생성합니다. 이런 접근법은 정규 표현식의 복잡성이 높은 경우에도 효과적으로 작동하며, 규칙 기반 수리 방식이 사실상 해결할 수 없는 복잡한 문제를 처리할 수 있습니다. 이 방법은 수리율을 15.4%p 향상시키며 기존의 방법들에 비해 성능을 높이고 있습니다.

- **Performance Highlights**: 논문에서 제시된 방법론은 복잡한 정규 표현식 취약성 수리 사례를 성공적으로 해결하며, 기존의 수리 방식보다 더 높은 수리율을 자랑합니다. 특히, LRR을 통해 정규 표현식의 의미적 유사성을 유지하면서도 효과적으로 취약성 문제를 해결할 수 있게 되었습니다. 이 접근법은 LLM의 생성 능력을 활용하면서도, 명확한 맥락을 제공하여 올바른 수리를 유도하는 데 기여합니다.



### RefGrader: Automated Grading of Mathematical Competition Proofs using Agentic Workflows (https://arxiv.org/abs/2510.09021)
- **What's New**: 본 논문에서는 최신의 LLMs(대규모 언어 모델)이 과거의 증명 기반 올림피아드 문제를 푸는 데 어려움을 겪던 시기에서 벗어나, 2025 IMO 문제의 대부분을 해결할 수 있게 되었다고 설명합니다. 특히, 선도 시스템이 6개 문제 중 5개를 해결할 수 있는 성과를 보였음을 강조합니다. 이러한 성과를 바탕으로, 이 모델들이 증명을 평가하는 능력을 어떻게 향상시킬 수 있는지를 분석하였습니다.

- **Technical Details**: 연구에서는 90개의 Gemini 2.5 Pro에서 생성된 솔루션을 이용하여 모델의 증명 분석 능력을 1에서 4까지의 척도로 평가하였습니다. 또한, IMO/USAMO 2025에서의 문제를 다루는 MathArena 솔루션 세트를 0에서 7까지의 척도로 점수화하였습니다. 분석 결과, 모델이 부정확한 솔루션(미세하게 부정확한 포함)을 신뢰성 있게 체크하지만, 부분 점수 부여에서 캘리브레이션 차이를 보임을 확인했습니다.

- **Performance Highlights**: 제안된 에이전틱 워크플로우를 통해 모델이 참조 솔루션을 추출하고 분석하며, 문제별 채점 규칙을 자동으로 도출할 수 있도록 하였습니다. 다양한 디자인 선택지를 구현하고 그 성능을 비교하여, 우리나라의 주석이 달린 데이터세트 및 MathArena에서 인간의 점수와 더 높은 일치를 달성했습니다. 모든 코드, 데이터 및 프롬프트를 공개하여 향후 연구에 도움이 되도록 하였습니다.



### TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation (https://arxiv.org/abs/2510.09011)
- **What's New**: 이번 논문에서는 여행 계획(task of travel planning)의 복잡성을 해결하기 위한 포괄적인 벤치마크를 소개합니다. 기존의 LLMs(large language models)의 평가 기준을 개선하고, 계획의 질을 직접 비교할 수 있도록 세분화된 평가지표를 통합하여 RL(reinforcement learning)과의 연계를 가능하게 했습니다. 이를 통해 여행 전문가의 주석과의 적합성이 60.75%에 도달하여 성과를 입증했습니다.

- **Technical Details**: 제안된 TripScore 벤치마크는 4,870개의 쿼리를 포함하고 있으며, 이들은 3,493개의 훈련 샘플, 158개의 검증 샘플, 1,219개의 테스트 샘플로 구성되어 있습니다. 구체적으로 테스트 세트는 1,000개의 합성 쿼리와 219개의 실제 사용자 요청을 포함하여 다양한 시나리오에서 모델의 일반화 가능성을 평가합니다. 이를 통해 사용자의 진정한 의도를 반영한 대체적인 요청을 제공합니다.

- **Performance Highlights**: 실험 결과, RL을 활용한 접근 방식인 GRPO는 기존의 프롬프트 기반 및 감독 학습 기반 모델에 비해 여행 계획의 질적 수준을 개선했습니다. 이러한 접근 방식은 통합 보상 점수를 높여 전체적인 계획의 실행 가능성을 향상시켰으며, 이는 다양한 방법과 LLM들을 통해 확인되었습니다. 이 연구는 여행 계획 분야에서 LLM의 사고 능력을 크게 향상시킬 잠재력을 보여줍니다.



### Tiny-R1V: Lightweight Multimodal Unified Reasoning Model via Model Merging (https://arxiv.org/abs/2510.08987)
Comments:
          Technical report, Code will be available at this https URL

- **What's New**: 본 논문에서는 경량의 다중 모달 대형 언어 모델인 Tiny-R1V를 제안합니다. 이 모델은 두 단계의 최적화를 통해 빠른 추론 속도와 높은 정확성을 달성하며, 다양한 작업에 대한 다중 모달 추론을 통합하는 데 중점을 두고 있습니다. Tiny-R1V는 3B의 매개변수를 가지고 있으며, 적은 토큰으로도 효과적인 성과를 보여줍니다.

- **Technical Details**: Tiny-R1V는 두 단계의 학습 프레임워크를 통해 구축되며, 첫 번째 단계에서는 Length-Informed Relative Policy Optimization (LIPO)를 통해 각 작업에 대한 대규모 학습 데이터를 사용하여 모델을 훈련합니다. 두 번째 단계에서는 Adaptive Model Merging (AMM)이라는 모델 병합 기법을 사용하여 여러 전문 모델을 통합하여 단일 아키텍처를 형성합니다. AMM은 태스크 벡터의 가중치를 적응적으로 조정하며, 새로운 경량 모델 병합을 통해 추가 데이터 없이 다중 MLLM의 통합을 가능하게 합니다.

- **Performance Highlights**: Tiny-R1V는 수학, 구조화된 데이터, OCR 등 다양한 범주를 아우르는 10개의 벤치마크에서 광범위한 평가를 통해 뛰어난 성과를 입증했습니다. 이 모델은 기존의 대형 모델들보다 빠른 추론 능력을 지니며, 다양한 다중 모달 추론 태스크에서 우수한 성능을 발휘합니다. 특히, LIPO와 AMM을 통해 짧고 효율적인 추론을 가능하게 하여 실시간이나 자원 제한 환경에서도 유용한 모델로 자리 잡을 것입니다.



### Semantic-Condition Tuning: Fusing Graph Context with Large Language Models for Knowledge Graph Completion (https://arxiv.org/abs/2510.08966)
Comments:
          11 pages, 3 figures, conference

- **What's New**: 이번 연구에서는 기존의 prefix-tuning 기법의 한계를 넘어서는 Semantic-condition Tuning (SCT)이라는 새로운 지식 주입 패러다임을 제안합니다. SCT는 두 가지 주요 모듈로 구성되어 있으며, 이들 모듈은 Knowledge Graphs (KGs)와 Large Language Models (LLMs)의 깊은 통합을 실현합니다. 이전의 방법들이 텍스트와 지식 임베딩을 단순히 결합했으나, SCT는 관계 중심의 의미론적 조건을 활용하여 깊이 있는 상호작용을 가능하게 합니다.

- **Technical Details**: SCT는 Semantic Graph Module과 Condition-Adaptive Fusion Module의 두 가지 모듈로 이루어져 있습니다. Semantic Graph Module은 Graph Neural Network를 이용해 지역 그래프 이웃의 맥락 인식을 기반으로 하는 의미론적 조건을 추출합니다. 이후 Condition-Adaptive Fusion Module은 이 조건을 사용하여 텍스트 임베딩을 조정하여 KG 맥락과 동적으로 정렬된 바로 가기를 제공합니다.

- **Performance Highlights**: SCT는 다양한 knowledge graph 벤치마크에서 기존의 prefix-tuning 및 다양한 강력한 방법들과 비교하여 유의미하게 우수한 성능을 보여주었습니다. 실험 결과, SCT는 텍스트 임베딩에 대한 심층적이고 지식 인지적인 상호 작용을 가능하게 하였으며, 이는 LLM의 추론 시 더 정확한 지식 추론을 지원한다는 점에서 중요한 개선을 만들었습니다.



### DualResearch: Entropy-Gated Dual-Graph Retrieval for Answer Reconstruction (https://arxiv.org/abs/2510.08959)
Comments:
          16 pages, 6 figures, 5 tables, Under Review

- **What's New**: 본 논문은 DualResearch라는 새로운 프레ーム워크를 제안하여, 도구 중심의 과학적 추론 과정에서 발생하는 문제들을 해결하고자 합니다. 이 프레임워크는 폭넓은 의미 그래프(Breadth Semantic Graph)와 깊이 인과 그래프(Depth Causal Graph)를 결합하여 지식의 구조적, 인과적 연결을 명확히 하고, 각 그래프의 특성에 맞춘 적합도 함수를 제공합니다. 이를 통해 정밀하고 안정적인 추론을 가능하게 하며, 오랜 도구 실행 로그를 간결한 추론 그래프로 압축할 수 있습니다.

- **Technical Details**: DualResearch는 두 가지 그래프 구조를 통해 실행 추론 과정을 모델링합니다. 첫째는 폭넓은 의미 그래프(Breadth Semantic Graph)로, 여기서는 안정적인 배경 지식을 인코딩하고, 둘째는 깊이 인과 그래프(Depth Causal Graph)로, 실행의 출처를 포착합니다. 이 두 그래프는 각각의 관련성 함수에 따라 쿼리되고, 탈중앙화 된 엔트로피 기반 수정 규칙을 통해 결합됩니다. 이를 통해 증거를 통합하여 최종 답변 분포를 생성하고, 더 확실한 경로를 가중치 있게 강화하여 신뢰성을 높입니다.

- **Performance Highlights**: DualResearch는 HLE와 GPQA와 같은 과학적 추론 벤치마크에서 경쟁력 있는 성과를 달성합니다. 내부 실행 로그를 활용하여 HLE에서 7.7%, GPQA에서 6.06%의 정확도를 개선했습니다. 이러한 성능 향상은 결합된 그래프 모델링과 엔트로피-게이트 퓨전 메커니즘을 통하여 이루어졌으며, 이를 통해 유사성 기반 단락 매칭에서 인과 기반 검증 가능한 추론으로의 진전을 나타냅니다.



### EcphoryRAG: Re-Imagining Knowledge-Graph RAG via Human Associative Memory (https://arxiv.org/abs/2510.08958)
- **What's New**: 새로운 프레임워크인 EcphoryRAG는 인지 과학의 원리를 통해 지식 검색을 위한 다중 홉(multi-hop) 추론 메커니즘을 실현합니다. 이 시스템은 사용자의 쿼리에서 특정 큐(cue)를 추출하고, 이를 통해 구조화된 지식 기반에서 정보 재호출을 유도합니다. 이를 통해 기존의 RAG 시스템보다 인덱싱 과정에서 최대 94%의 토큰 소비 절감이 가능해졌습니다.

- **Technical Details**: EcphoryRAG의 작동 방식은 두 가지 주요 단계로 나뉩니다. 첫 번째 단계는 오프라인에서 지식을 인덱싱하여 구조화된 메모리 시스템을 구축하는 것입니다. 두 번째 단계인 온라인 검색 과정에서는 큐에 기반한 재호출 과정을 시뮬레이션하여 복잡한 질의에 대한 답변을 효율적으로 도출합니다.

- **Performance Highlights**: EcphoryRAG는 2WikiMultiHop, HotpotQA, MuSiQue 벤치마크에서 새로운 최첨단 성능을 입증했습니다. 특히, 평균 Exact Match (EM) 점수를 0.392에서 0.474로 개선하여 기존의 KG-RAG 방법론인 HippoRAG보다도 우수한 효과를 나타냈습니다. 또한, 오프라인 인덱싱 토큰 비용을 최대 18배까지 줄이는 remarkable한 효율성을 달성했습니다.



### FATHOMS-RAG: A Framework for the Assessment of Thinking and Observation in Multimodal Systems that use Retrieval Augmented Generation (https://arxiv.org/abs/2510.08945)
- **What's New**: 최근 Retrieval-augmented generation (RAG) 기법이 대형 언어 모델(LLM)의 사실 정확도 향상을 위한 유망한 패러다임으로 주목받고 있습니다. 본 연구에서는 RAG 파이프라인 전체를 평가하기 위한 벤치마크를 도입하며, 이를 통해 다양한 정보 모달리티를 수집, 검색, 추론할 수 있는 능력을 평가합니다. 기존 벤치마크와 달리, 본 평가에서는 텍스트 데이터, 테이블, 이미지 등 여러 유형의 정보를 포괄적으로 다룹니다.

- **Technical Details**: 우리의 벤치마크는 93개의 질문으로 구성된 데이터셋을 포함하고 있으며, 각 질문은 text-only, tables, images, multimodal, cross-document multimodal이라는 다섯 가지 카테고리로 나뉩니다. 정확성을 평가하기 위해 phrase-level recall 메트릭을 도입하고, 가능성 할로시네이션을 식별하기 위해 nearest-neighbor 임베딩 분류기를 사용합니다. 이를 통해 개방형 소스 및 폐쇄형 소스 파이프라인을 비교 평가하며, 각 파이프라인의 성능을 정량적으로 분석합니다.

- **Performance Highlights**: 유지된 결과에 따르면, 폐쇄형 소스 파이프라인이 개방형 소스 파이프라인에 비해 정확성과 할로시네이션 메트릭 모두에서 상당한 성과 차이를 보입니다. 특히, 테이블이나 이미지 등 다중 문서 정보에 의존하는 질문에서 이러한 성능 차이는 더욱 두드러집니다. 한편, 인간 평가를 통해 정확성에서 평균 4.62, 할로시네이션 탐지에서 평균 4.53의 동의율을 기록하였으며, 이는 1-5 리커트 척도의 결과로 나타났습니다.



### RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation (https://arxiv.org/abs/2510.08931)
Comments:
          NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling

- **What's New**: 이 논문에서는 데이터 오염이 LLM(대형 언어 모델) 평가에 미치는 영향을 해결하기 위해 RADAR(Recall vs. Reasoning Detection through Activation Representation)이라는 새로운 프레임워크를 소개합니다. 기존 방법과는 달리, RADAR는 기계적 해석 가능성을 활용하여 모델의 기억 기반 응답과 추론 기반 응답을 구분함으로써 오염을 탐지합니다. 이 연구는 37개의 특징(feature)을 추출하여 93%의 정확도를 기록하며, 명확한 경우에서 완벽한 성과를 보였습니다.

- **Technical Details**: RADAR는 내부 모델 상태를 추출하는 Mechanistic Analyzer, 표면 및 기계적 특징을 계산하는 Feature Extraction, 그리고 기억(recall)과 추론(reasoning)을 예측하는 Classifier로 구성되어 있습니다. 37개의 특징은 표면 특징(17)과 기계적 특징(20)으로 구분되어 있으며, 각각 모델 출력 궤적 및 내부 계산 메커니즘을 포착합니다. 분류 모듈은 Random Forest, Gradient Boosting, SVM, Logistic Regression의 네 가지 감독 학습 모델을 사용하여 훈련됩니다.

- **Performance Highlights**: RADAR는 테스트 세트에서 93.0%의 정확도를 기록했으며, 기억 작업에서 97.7%, 추론 작업에서 89.3%의 성과를 보였습니다. 주요 특징으로는 전문화된 주의(attention) heads, 회로 복잡도(circuit complexity), 그리고 신뢰도 수렴 패턴이 포함됩니다. 이 연구는 기억과 추론 간의 뚜렷한 구분을 제공하며, 기계적 해석 가능성이 LLM 평가의 발전에 중요한 역할을 할 수 있음을 보여줍니다.



### LM Fight Arena: Benchmarking Large Multimodal Models via Game Competition (https://arxiv.org/abs/2510.08928)
- **What's New**: 본 논문에서는 대규모 다중모달 모델(LMMs)의 성능을 평가하는 새로운 프레임워크인 LM Fight Arena를 소개합니다. 이 프레임워크는 고전 격투 게임인 Mortal Kombat II에서 LMMs를 서로 대결시켜 그들의 전략적 추론 능력을 평가합니다. 이를 통해 기존의 정적 평가방법과는 달리 동적 환경에서의 성능을 자동화된 방식으로 측정할 수 있습니다.

- **Technical Details**: LM Fight Arena는 1994년 Sega Genesis 버전의 Mortal Kombat II를 시험 환경으로 채택하여, 동일한 캐릭터인 Liu Kang을 조정하는 6개의 모델을 공정하게 비교합니다. 평가 프레임워크는 게임 상태를 결정론적으로 캡처하고, 각 모델은 자연어로 명령을 반환하여 제어를 수행합니다. 이 과정에서 모델의 성능을 인지, 추론 및 의사결정 능력으로 격리하여 평가합니다.

- **Performance Highlights**: 대회 결과, Claude 3.5 Sonnet 모델이 100%의 승률로 1위를 차지했습니다. Gemini 2.5 Pro는 80%의 승률로 뒤를 이었으며, 대규모 Qwen2.5-VL-72B 모델은 60%의 성과를 기록했습니다. 각 모델의 성과는 대각선 토너먼트 매트릭스를 통해 명확히 시각화되어 각 모델의 상대적인 위치를 알 수 있습니다.



### GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfar (https://arxiv.org/abs/2510.08872)
Comments:
          31 pages, 6 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 응답 품질과 사용자 후생을 최대화하기 위한 새로운 접근 방식인 게임 이론적 정렬(Game-Theoretic Alignment, GTAlign)을 제안합니다. GTAlign는 사용자의 질문과 LLM의 응답 간의 상호작용을 전략 게임으로 모델링하고, 공통의 이익을 추구하는 방식으로 LLM의 학습과 인프런스를 조정합니다. 이 프레임워크는 LLM이 사용자와의 상호작용에서 더 나은 결정을 내릴 수 있도록 합니다.

- **Technical Details**: GTAlign의 핵심 혁신점은 세 가지입니다: (1) 게임 이론적 추론 체인에서는 LLM이 다양한 응답 행동의 보수를 계산하여 상호 이익이 최대화되는 행동을 선택합니다. (2) 상호 후생 보상(mutual welfare reward)은 협력 행동을 강화하여 LLM과 사용자 보상을 동시에 최대화합니다. (3) 추론 중 LLM의 결정을 수정할 수 있는 알고리즘을 설계하여 LLM의 응답을 동적으로 조정합니다.

- **Performance Highlights**: GTAlign는 수학 문제 해결, 창의적 글쓰기, 개방형 질문 응답 등 다양한 과제에서 기존 방법들에 비해 21.5%의 게임 이론적 추론 효율성 향상과 4.9%의 응답 품질 개선을 입증했습니다. 또한, 사용자 만족도가 11.3% 향상되는 등 실험을 통해 향상된 상호 후생을 보여주었습니다. 이 결과들은 GTAlign가 LLM 보조 도구의 합리적이고 적응력 있는 행동을 향상시키는 효과적인 프레임워크임을 시사합니다.



### ReviewerToo: Should AI Join The Program Committee? A Look At The Future of Peer Review (https://arxiv.org/abs/2510.08867)
- **What's New**: 이번 연구에서는 ReviewerToo라는 모듈형 프레임워크를 소개하며, 이는 AI-assisted peer review를 위한 체계적이고 일관된 평가를 지원합니다. 이 프레임워크는 다양한 심사자 페르소나를 사용하여 실험을 설계하고 수행하며, 인간 심사의 판단을 보완할 수 있습니다. ICLR 2025 데이터셋을 활용한 실험에서는 AI 모델인 gpt-oss-120b가 Accept/Reject 분류 작업에서 81.8%의 정확도를 기록했습니다.

- **Technical Details**: ReviewerToo는 AI-assisted peer review를 위한 구조적 프로세스로, 제출된 원고의 수집, 문헌 리뷰 생성, 다양한 심사자 에이전트에 의한 리뷰 생성, 저자 에이전트의 반박 작성, 최종 메타 리뷰 통합의 순서로 진행됩니다. 본 프레임워크는 각각 하나의 되풀이 응답만을 허용하며, 이는 실제 학술회의의 심사 관행을 반영하여 단일 검토 및 제한된 확인만을 제공합니다.

- **Performance Highlights**: ReviewerToo로 생성된 리뷰는 LLM 판단에 의해 인간 평균보다 높은 품질을 평가받았으나, 여전히 가장 강력한 전문가 기여에 미치지는 못했습니다. AI 리뷰어는 사실 확인 및 문헌 조사에서 특히 강점을 보이는 반면, 방법론적 참신성 및 이론적 기여를 평가하는 데 어려움을 겪는 것으로 나타났습니다. 이는 AI가 인간 전문 지식의 보완 색깔을 갖는 것에 대한 필요성을 강조합니다.



### What Is Your Agent's GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignmen (https://arxiv.org/abs/2510.08847)
- **What's New**: 이번 논문에서는 Agent GPA (Goal-Plan-Action) 프레임워크를 소개합니다. 이 프레임워크는 목표 설정, 계획 수립 및 행동 실행이라는 에이전트의 운영 루프를 기반으로 한 평가 패러다임입니다. Agent GPA는 목표 달성도, 논리적 일관성, 실행 효율성, 계획 품질, 계획 준수의 다섯 가지 평가 지표를 포함하여, 더 구체적이고 조직적인 에이전트 실패 분석을 가능하게 합니다.

- **Technical Details**: Agent GPA 프레임워크는 테스트 실행이나 작동 중인 에이전트의 샘플 트레이스를 통해 계산될 수 있으며, 인적 평가자 또는 TruLens OSS 라이브러리를 사용하여 자동화된 LLM- 판별자 평가를 통해 이루어질 수 있습니다. 우리의 연구는 내부 오류, 즉 에이전트가 제어할 수 있는 오류(예: 도구 호출이나 망상)를 포착하는 데 집중하고 있으며, 결과적으로 에이전트 성능 개선에 기여하고자 합니다. 두 가지 기준 데이터셋인 TRAIL/GAIA 및 프로덕션급 데이터 에이전트 내부 데이터셋을 통해 프레임워크의 효과성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, Agent GPA 프레임워크는 다양한 에이전트 실패를 체계적으로 탐지하고 이해하는 방법을 제공합니다. LLM Judges는 인간의 판단과 강한 일치를 보이며, TRAIL/GAIA 데이터셋 테스트 세트에서 95%의 오류를 식별했습니다. 또한, LLM Judges는 다수의 오류를 적절히 지역화하여, 특정한 디버깅 작업을 가능하게 하며, 이러한 효과로 인해 LLM 평가의 신뢰성이 강화되었습니다.



### Everyone prefers human writers, including AI (https://arxiv.org/abs/2510.08831)
Comments:
          46 pages, 18 figures (5 main text + 13 supplementary), 5 tables

- **What's New**: 이번 연구는 인간과 AI 평가자의 문학적 스타일 평가에서의 기여 편향(attribution bias)을 조사한 최초의 연구로, 두 가지 조사를 통해 이 현상을 비교했습니다. 연구 결과, AI 모델이 인간보다 기여 편향을 2.5배 더 강하게 나타내며, AI가 'AI 생성'으로 레이블 된 창의적 콘텐츠를 체계적으로 낮게 평가한다는 사실이 밝혀졌습니다. 이 결과는 AI 시스템이 훈련 과정에서 인간의 문화적 편향을 흡수했음을 시사합니다.

- **Technical Details**: 연구는 Raymond Queneau의 'Exercises in Style'를 기반으로 하여, 인간 참가자(556명)와 AI 모델(13개)이 퀘노와 GPT-4에서 생성된 문학적 구절을 평가하는 실험을 진행했습니다. 연구 1은 인간과 AI 평가자가 기여 레이블의 영향을 받는 방식을 비교했고, 연구 2는 14개의 AI 생성자에서 생성된 콘텐츠의 편향을 확인했습니다. 두 연구 모두에서, 인간과 AI 모두에서 기여 레이블이 평가 기준을 반전시키는 경향이 나타났습니다.

- **Performance Highlights**: AI 모델은 +34.3%의 기여 편향을 나타내어 인간의 +13.7% 편향보다 현저하게 높았습니다. 연구 2에서는 이 편향이 AI 아키텍처에 걸쳐 일정하게 존재하는 것을 확인했습니다. AI 시스템이 인간의 창의적인 작품을 기계적으로 인식하여 평가를 낮추는 현상은 앞으로 AI 생성 콘텐츠의 평가에 중요한 영향을 미칠 것으로 예상됩니다.



### COMPASS: Enhancing Agent Long-Horizon Reasoning with Evolving Contex (https://arxiv.org/abs/2510.08790)
Comments:
          Under Review for ACL

- **What's New**: 이 논문에서는 장기적인 작업을 수행하는 LLM(대규모 언어 모델) 에이전트의 한계를 해결하기 위해 COMPASS(맥락 조직 다중 에이전트 계획 및 전략 시스템)를 제안합니다. 이 시스템은 복잡한 상황에서의 사고 과정에서 발생하는 오류를 줄이고, 에이전트가 중요 증거를 놓치거나 관련 없는 정보에 주의를 빼앗기는 것을 방지합니다. 이를 통해 전략적인 개입(Interventions)을 할 수 있도록 프로세스를 관리합니다.

- **Technical Details**: COMPASS는 전술적 실행(Tactical Execution), 전략적 감독(Stratgic Oversight), 맥락 조직(Context Organization)의 세 가지 구성 요소로 나눠진 경량 계층적 프레임워크입니다. 여기에 주요 에이전트(Main Agent), 메타 사고자(Meta-Thinker), 그리고 맥락 관리자(Context Manager)가 포함되어, 각기 다른 사고 단계의 간결하고 관련성 있는 진행 요약을 유지합니다. 이 구조적 접근법은 장기 작업 처리에서의 비효율성을 줄이는 데 중점을 두고 설계되었습니다.

- **Performance Highlights**: COMPASS는 GAIA, BrowseComp, Humanity's Last Exam의 세 가지 도전적인 벤치마크에서 정확도를 최대 20%까지 개선했습니다. 또한, 테스트 시간 확장(Test-time Scaling) 기능을 도입하여 성능을 DeepResearch 에이전트와 동일한 수준으로 끌어올릴 수 있으며, 맥락 관리를 더 작은 모델에 위임하는 후속 훈련 파이프라인(Post-training Pipeline)을 통해 효율성을 더욱 향상시킵니다.



### Robust Heuristic Algorithm Design with LLMs (https://arxiv.org/abs/2510.08755)
- **What's New**: 본 연구에서는 LLMs(대규모 언어 모델)를 활용한 휴리스틱(heuristic) 설계를 향상시키기 위해, 왜 휴리스틱이 성능이 저조한지를 설명하고 이를 개선하기 위한 도구를 추가하는 접근법을 제안합니다. 이러한 접근법은 특정 입력 공간에서의 디자인을 전문화하여 보다 견고한 알고리즘을 생성할 수 있도록 합니다.

- **Technical Details**: 특히, 연구팀은 세 가지 간단한 아이디어를 통해 활동을 전개하였습니다: (1) LLM을 저성능 사례에 expose(노출)하기, (2) 저성이유를 설명하기, (3) 입력 공간의 특정 영역에 맞게 디자인을 전문화하기입니다. 이러한 방법들은 효과적인 결과를 나타냅니다.

- **Performance Highlights**: 본 연구에서 생성된 휴리스틱은 FunSearch와 비교했을 때 최악의 경우 성능이 약 28배 향상되었으며, 평균 성능 또한 개선되었습니다. 또한 런타임(runtime)은 유지되는 경향을 보였습니다.



### Optimizing delivery for quick commerce factoring qualitative assessment of generated routes (https://arxiv.org/abs/2510.08671)
- **What's New**: 이번 연구는 인도의 전자상거래(e-commerce) 시장의 성장과 함께 물류 과정에서 발생하는 마지막 단계의 배달 효율성을 개선하기 위한 새로운 프레임워크를 제안합니다. 이를 통해 기존의 차량 경로 문제(vehicle routing problem, VRP)에서 발생할 수 있는 문제를 대형 언어 모델(large language models, LLMs)을 사용하여 비판적으로 평가함으로써, 물류 운영자들이 더 효율적인 배달 계획을 수립할 수 있도록 지원합니다. 연구는 개방형 LLM이 79%의 정확도로 문제를 식별하는 성과를 보였고, 상용 모델은 86%에 이르는 성과를 보여주었습니다.

- **Technical Details**: 연구에서는 VRP 과정에서 생성된 경로를 사용자가 지정한 정책에 따라 리뷰하는 프레임워크 개발에 초점을 맞추고 있습니다. 사용자는 자연어를 통해 정책을 정의할 수 있으며, 이는 기존의 VRP 솔버에서 요구되는 제한 조건을 수학적 표현으로 입력하는 방식과는 다릅니다. 이 프레임워크는 날씨나 지형 정보와 같은 추가적인 외부 데이터를 쉽게 포함할 수 있는 장점이 있습니다.

- **Performance Highlights**: 대형 언어 모델을 활용한 경로 평가 방식은 기존의 거리 및 시간 기반 메트릭을 초월한 유의미한 평가 레이어로 작용할 수 있습니다. 연구 결과, LLM을 활용한 경로의 비판적 평가가 비용 효율성, 배달 신뢰성 및 지속 가능성 개선에 기여할 수 있음을 시사합니다. 이는 인도와 같은 개발도상국의 마지막 단계 물류 문제 해결에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents (https://arxiv.org/abs/2510.08619)
- **What's New**: 이 논문은 대규모 과학 데이터셋을 활용하여 특정 연구 질문에 얽매이지 않고 탐색적 발견을 할 수 있는 새로운 프로세스인 "hypothesis hunting"을 소개합니다. 이를 위한 프레임워크인 AScience를 도입하고, 이를 기반으로 다양한 행동을 가진 LLM(large language model) 연구 에이전트의 분산 시스템 ASCollab를 구현하였습니다. 실험 결과, 이러한 사회적 역학이 전문가 평가 결과의 축적을 가능하게 하는 것을 보여줍니다.

- **Technical Details**: AScience 프레임워크는 (i) 접근 방식의 인식 지형, (ii) 이질적인 과학 에이전트, (iii) 주의 흐름 및 협력을 routing하는 네트워크, (iv) '좋은 과학'의 강건한 평가 메커니즘의 네 가지 상호 작용 요소로 구성되어 있습니다. ASCollab는 이런 요소들을 결합하여 에이전트들이 자율적으로 발견을 생성하고 지속적으로 품질을 보장할 수 있는 구조를 제공합니다. 이 시스템은 단일 목표를 추구하는 단일 에이전트가 아니라, 병렬 탐색 및 축적적 정제를 수행하는 커뮤니티에서 발생하는 발견을 강조합니다.

- **Performance Highlights**: ASCollab는 TCGA(The Cancer Genome Atlas) 3개 암 집단에 적용되어 전이체, 단백질체, 경로 및 임상 생존 데이터 통합을 통해 다양한 발견물을 생성하였습니다. 이 시스템은 기존 암 유전자의 재발견, ferroptosis 경로의 확장, 새로운 치료 목표 제안 등 흥미로운 결과를 도출하며, 이는 가설 수렵의 맥락에서 네트워크 에이전트의 가능성을 보여줍니다. 에이전트들이 생성한 결과는 전문가들에게 '신규', '높은 품질', '다양성'으로 평가받았습니다.



### Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation (https://arxiv.org/abs/2510.09599)
Comments:
          Our code and data are available at this https URL

- **What's New**: 이번 연구에서는 Prompting Test-Time Scaling (P-TTS)라는 새로운 추론 시간 데이터 증강 전략을 제안합니다. 이 방법은 기존의 훈련된 체인을 사용하는 대신, 단 90개의 고품질 수학 추론 예제를 활용하여 다양한 추론 맥락을 생성할 수 있게 합니다. P-TTS는 고비용의 데이터 수집 없이도 LLM의 추론 능력을 향상시키는 실용적인 방안을 제시합니다.

- **Technical Details**: P-TTS는 주어진 예제를 기반으로 1) 다양한 원칙적 지침을 사용하여 예제를 조합하고, 2) 인덕티브 바이어스를 조절하여 예제의 순서를 변경하며, 3) 모델을 통해 재구성된 합리적인 답변 뼈대를 만들어내는 방법으로 작동합니다. 이 접근은 모델이 수집된 데이터에 의존하지 않고도 더 큰 추론 패턴 공간을 탐색할 수 있게 합니다.

- **Performance Highlights**: P-TTS는 다양한 수학적 추론 기준에서 이전의 경쟁 모델들을 초월하는 성과를 기록했습니다. 예를 들어 AIME24 및 AIME25에서 각각 +26.66%와 +30.00%의 정확도 향상을 달성했습니다. 이와 더불어 P-TTS는 제로샷 일반화 정확도를 개선시켜, 다양한 외부 도메인에서도 뛰어난 성능을 보였습니다.



### BaNEL: Exploration Posteriors for Generative Modeling Using Only Negative Rewards (https://arxiv.org/abs/2510.09596)
- **What's New**: 이 연구에서는 BaNEL(Bayesian Negative Evidence Learning)이라는 새로운 알고리즘을 제안합니다. 이 알고리즘은 실패한 시도들로만 모델을 포스트-트레이닝(post-training)하면서도 보상 평가의 수(NREs)를 최소화합니다. 특히 이 방법은 긍정적인 샘플을 단 한 번도 관찰하지 않고도 여러 희소 보상(sparse-reward) 작업에서 모델 성능을 향상시키는 특징이 있습니다.

- **Technical Details**: BaNEL은 실패의 정규성을 학습하는 문제를 또 다른 순환(generative) 모델링 문제로 변환하는 아이디어에 기반하고 있습니다. 우리는 이 모델을 사용하여 새로운 데이터가 이전의 실패와 유사한지를 평가하고, 그로 인해 생성 방향을 수정합니다. 알고리즘은 마지막으로 실패했던 데이터와 유사 한도를 설정하여 고비용의 보상 오라클(reward oracle)에 접근하기 전에 이러한 샘플을 거부합니다.

- **Performance Highlights**: BaNEL은 기존의 참신성 보상(novelty-bonus) 접근법과 비교하여 성공률이 여러 배수 높다는 것을 보여주었습니다. 이 알고리즘은 최대한 적은 NRE를 사용하면서도, 계산(computation)과 성공률(success rate) 간의 균형을 맞출 수 있게 합니다. 실험 결과, BaNEL은 여러 어려운 문제에서 현저히 높은 성공률을 보였고, 이는 과거의 방법들보다 현저히 우수한 성과를 나타냅니다.



### Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Mod (https://arxiv.org/abs/2510.09551)
- **What's New**: 2024년 말, Google 연구자들이 Titans: Learning at Test Time이라는 신경 메모리 모델을 도입했습니다. 이 모델은 다양한 작업에서 강력한 결과를 보여주지만, 공개된 코드 부족과 원본 설명의 모호함이 재현성을 저해하고 있습니다. 본 논문에서는 Titans의 경량 재구현을 제시하고, Masked Language Modeling, Time Series Forecasting, Recommendation 작업에 대한 포괄적인 평가를 수행했습니다.

- **Technical Details**: 우리의 방법론은 Titans에 대한 재구현 및 분석을 제공하며, 주 초점은 원본 설명에서 모호한 디자인 선택을 체계적으로 평가하는 것입니다. Titans는 단기 메모리 역할을 하는 Core 모듈, 시험 시간에 지속적으로 업데이트되는 Neural Long-Term Memory 모듈, 각 시퀀스 앞에 고정되어 있는 Persistent Memory 토큰의 세 가지 구성 요소를 통합합니다. 모델의 재현성을 높이기 위해 모든 메커니즘이 명확히 정의되고 설정 가능하도록 설계했습니다.

- **Performance Highlights**: MLM 작업에서 Titans는 동일한 학습 예산을 가진 BERT 유사 모델과 동등하거나 이를 초과하는 성과를 나타냈습니다. 추천 작업에서는 Titans가 BERT4Rec 모델 성능을 초과하지 못했으나, 메모리를 포함했을 때 MRR 지수가 0.09 증가하여 메모리가 사용자의 영화-아이템 간 상관관계를 개선할 수 있음을 보여주었습니다. 시계열 예측 작업에서도 메모리 전용 변형인 LMM이 iTransformer 및 LSTM과 견줄 만한 성과를 보였지만, 원본 Titans 논문에서 보고된 수준에는 도달하지 못했습니다.



### SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models (https://arxiv.org/abs/2510.09541)
- **What's New**: 본 논문에서는 Sandwiched Policy Gradient (SPG)라는 새로운 강화 학습 알고리즘을 제안합니다. 이 알고리즘은 대규모 확산 언어 모델(dLLM)의 가치 평가를 위한 새로운 접근 방식을 제공합니다. 기존의 ELBO나 일괄적인 추정 방법의 한계를 극복하여 편향을 줄여줍니다.

- **Technical Details**: SPG는 긍정적인 보상 시퀀스에 대해 확률적 하한을 극대화하고, 부정적인 보상 시퀀스에 대한 상한을 최소화하여 진정한 로그 우도(log-likelihood)를 '샌드위치'하는 방식으로 작동합니다. 이를 통해 보다 안정적이고 객관적인 정책 그래디언트를 도출할 수 있습니다. 블록 단위 마스킹 전략도 도입하여 훈련 목표의 추정의 안정성을 높입니다.

- **Performance Highlights**: SPG는 GSM8K, MATH500, Countdown 및 Sudoku와 같은 네 가지 추론 벤치마크에 대해 최첨단 성능을 달성했습니다. 특히, GSM8K에서 3.6%, MATH500에서 2.6%, Countdown에서는 18.4%, 그리고 Sudoku에서는 27%의 정확도를 개선했습니다. 이러한 성과는 SPG의 효과성을 강력하게 입증합니다.



### Mitigating Overthinking through Reasoning Shaping (https://arxiv.org/abs/2510.09535)
- **What's New**: 최근 연구에서 제안된 Group Relative Segment Penalization (GRSP)은 대규모 추론 모델(Large Reasoning Models, LRMs)의 효율성과 정확성을 동시에 향상시키기 위한 새로운 방법론입니다. GRSP는 전통적인 토큰 수준의 감시(supervision) 대신 단 단계(segment) 수준의 규제를 통해 불필요한 사고를 줄이고 효율적으로 계산 비용을 관리합니다. 연구자들은 이 방식을 사용하여 복잡한 문제 해결 시에도 성능 저하를 최소화하며, 모델 크기에 따라 잘 확장될 수 있음을 입증했습니다.

- **Technical Details**: GRSP는 사고 과정 내에서 단계별(segment)로 이루어진 추론을 감시하며, 각 단계가 조정된 가중치를 통해 규제됩니다. 연구자들은 다양한 길이의 추론 단계를 분석하여 이들이 토큰 소비 및 모델 성능과 유의미하게 연관되어 있음을 발견하였습니다. 이를 바탕으로, GRSP는 각 길이 클러스터 내에서 단계를 효과적으로 줄이도록 고안되어 있으며, 이 과정에서 길이에 따른 가중치를 부여함으로써 모델의 사고 과정을 더 안정적으로 통제할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 GRSP는 기존 방법들에 비해 뛰어난 토큰 효율성을 보여주었으며, 특히 난이도가 높은 문제에서 그 장점이 뚜렷하게 나타났습니다. GRSP는 RL 훈련을 안정화시키고, 다양한 모델 크기에 효과적으로 확장할 수 있는 가능성을 제시합니다. 결과적으로, GRSP는 추론 성능을 높이면서도 토큰 사용량을 최소화하는 효과를 발휘하여 향후 연구에 대한 인사이트를 제공합니다.



### Autonomous Soft Robotic Guidewire Navigation via Imitation Learning (https://arxiv.org/abs/2510.09497)
- **What's New**: 이 연구에서는 소프트 로봇 가이드와이어 내비게이션을 위한 첫 번째 엔드 투 엔드 모방 학습 알고리즘을 개발했습니다. 이를 통해 정밀함과 안전성을 높일 수 있는 자동화된 내비게이션 접근 방식을 구현하고 있습니다. 특히, 마이크로 가이드와이어를 이용해 뇌동맥류 치료에 적용하며, 다양한 혈관 구조에서의 일반화를 목표로 합니다.

- **Technical Details**: 연구에서 사용된 로봇은 3D 프린트된 유체 구동 액추에이터로, 혈관의 복잡한 경로에 안전하게 적응할 수 있는 특성을 지니고 있습니다. 주요 구성 요소에는 다섯 개의 벨로우스가 연결된 배열이 포함되어 있으며, 맞춤형 유체 주입기에서 다양한 압력 입력에 따라 구부러질 수 있습니다. 이 로봇은 또한 사용자 제어에 따라 카테터 팁의 구부러짐과 삽입 속도를 직관적으로 조절할 수 있는 시스템을 갖추고 있습니다.

- **Performance Highlights**: 모델은 세 가지 이전에 보지 못한 혈관 기하학에서 83%의 성공률로 자율적인 내비게이션을 수행하는 능력을 보여줍니다. 또한 디자인과 데이터 수집 선택의 효과성을 검증하기 위한 기초 연구와 제거 실험을 진행하여, 실제 플랫폼에서의 일반화 성능을 평가했습니다. 결과적으로 이 연구는 소프트 로봇 내비게이션의 잠재력을 강조하며, 더 나아가 실제 수술 환경에서도 유용할 것으로 기대됩니다.



### Precoder Design in Multi-User FDD Systems with VQ-VAE and GNN (https://arxiv.org/abs/2510.09495)
Comments:
          Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문은 프리퀀시 디비전 듀플렉스(FDD) 시스템에서 로버스트 프리코딩을 효율적으로 구현하는 방법을 제시합니다. 일반적인 가우시안 혼합 모델(GMM) 대신 벡터 양자화 변분 오토인코더(VQ-VAE)를 사용하여 GMM의 주요 단점인 피드백 비트 수의 기하급수적 확장을 회피합니다. 이를 통해 GNN과 VQ-VAE를 공동 훈련하여 다중 사용자 무선 시스템에서의 성능 향상을 이끌어냅니다.

- **Technical Details**: 이 시스템 모델은 단일 셀 다중 사용자 시스템을 목표로 하며, 기지국(BS)은 NN 개의 안테나를 가지고 있습니다. 각 단일 안테나 이동 단말(MT)에 대해 선형 프리코딩이 적용되어, 총 전송 전력 제약 하에 실행됩니다. VQ-VAE를 사용하여 채널 상태 정보(CSI)를 양자화하고 GNN을 통해 효율적인 프리코더 설계를 수행하여, 실시간 데이터 수집 시의 전반적인 성능을 개선합니다.

- **Performance Highlights**: 제안된 프레임워크는 기존의 방법들보다 우수한 성능을 보이며, 특히 저 피드백 비트 오버헤드를 유지하는 시스템에서 탁월성을 입증합니다. 실제 측정 데이터를 사용한 시뮬레이션 결과는 반복 알고리즘과 DFT 코드북 기반 방법에 비해 뛰어난 성능 향상을 보여줍니다. 이러한 결과는 에너지 효율성과 지속 가능성에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Performance Analysis of Machine Learning Algorithms in Chronic Kidney Disease Prediction (https://arxiv.org/abs/2510.09493)
Comments:
          11 pages, 7 figures, Presented at the 2022 IEEE 13th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), pp. 0417-0423

- **What's New**: 이 연구에서는 만성 신장 질환(Chronic Kidney Disease, CKD) 진단을 위한 컴퓨터 보조 설계를 제안하였습니다. CKD는 전 세계 인구의 약 10%에 영향을 미치는 주요 건강 문제로, 신장 기능 저하를 초래합니다. 머신 러닝(ML) 모델을 이용해 CKD 위험 평가 및 모니터링의 효율성을 높이고, 조기 진단을 위한 기술적 접근을 제안합니다.

- **Technical Details**: 이 연구는 UCL 머신 러닝 저장소의 CKD 데이터셋을 사용하였고, 결측값을 "mean-mode" 및 "랜덤 샘플링" 방법으로 처리하여 데이터를 구축하였습니다. 8가지 머신 러닝 기법(랜덤 포레스트, SVM, 나이브 베이즈, 로지스틱 회귀, KNN, XGBoost, 결정 트리, AdaBoost)을 통해 모델을 구성하고, 정확도 비교를 위해 성능 평가를 실시하였습니다. 데이터는 훈련 세트(80%)와 테스트 세트(20%)로 나누어 ML 알고리즘의 예측 정확성을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 랜덤 포레스트와 로지스틱 회귀 모델이 각각 99%의 뛰어난 정확도를 보였습니다. AdaBoost, XGBoost, 나이브 베이즈, 결정 트리 및 SVM도 각각 높은 성능을 나타냈으나, KNN 분류기는 73%의 정확도로 가장 낮은 성능을 기록했습니다. 이 연구는 CKD 진단을 위한 머신 러닝의 가능성을 강조하며, 의료 분야에서의 활용 가능성을 제시합니다.



### Multimodal Policy Internalization for Conversational Agents (https://arxiv.org/abs/2510.09474)
- **What's New**: 이번 논문은 Multimodal Policy Internalization (MPI)이라는 새로운 작업을 소개하며, 이는 멀티모달 모델이 정책을 포함하지 않고도 정책에 따라 응답을 생성할 수 있도록 훈련하는 것을 목표로 합니다. 전통적인 정책은 종종 복잡하고 긴 형식으로 제공되며, 이는 고정적인 계산 비용을 발생시키고 모델의 일관된 정책 준수를 어렵게 만듭니다. 본 연구에서 제안하는 MPI는 이러한 문제를 해결하기 위한 것으로, 시각적 입력을 포함한 결정-making 및 도구 사용 작업을 다룹니다.

- **Technical Details**: 제안된 TriMPI는 세 가지 단계의 훈련 프레임워크로 구성되어 있습니다. 첫 번째 단계인 visually-masked continual pretraining (VM-CPT)에서는 멀티모달 정책에 직접적으로 언어 모델링을 수행하여 정책 지식을 모델에 주입합니다. 두 번째 단계인 chain-of-thought supervised finetuning (CoT-SFT)에서는 보다 복잡한 정책 내를 탐색할 수 있도록 돕고, 마지막 단계인 PolicyRollout은 RL (Reinforcement Learning) 기법을 통해 정책 관련 응답을 통해 학습하도록 설계되었습니다.

- **Performance Highlights**: TriMPI는 MPI에서 상당한 향상을 이루어내며, CoT SFT 기준 및 in-context 설정에 비해 각각 70.7% 및 79.4%의 절대 개선을 나타냅니다. 논문에서는 정책 업데이트에 대한 일반화 능력 및 치명적인 망각에 대한 견고성도 향상된 것으로 평가하였습니다. 복잡한 정책에 대해서는 더욱 주목할 만한 개선이 나타나는 것으로 증명되었습니다.



### Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy (https://arxiv.org/abs/2510.09469)
- **What's New**: 이 논문에서는 Multi-Agent Pathfinding (MAPF) 문제에 대한 새로운 혼합 프레임워크를 제안합니다. 이는 비 중앙 집중식 경로 계획(decentralized path planning)과 가벼운 중앙 조정자(lightweight centralized coordinator)를 결합하여 정보 공유를 최소화하면서도 솔루션의 유효성을 유지합니다. 연구 결과, 최소한의 타겟 알림만으로 약 93%의 정보 부하를 줄이면서도 효과적인 경로 계획이 가능하다는 것을 보여줍니다.

- **Technical Details**: 제안된 혼합 프레임워크는 에이전트가 국소 관찰(local observations)을 기반으로 작동하는 강화 학습(reinforcement learning) 기반 신경망 계획기(neural network planners)를 활용합니다. 중앙 조정자가 에이전트의 경로를 감시하며, 예상되는 충돌에 대비해 동적으로 타겟 정보를 공유하여 지역적 재계획(localized re-planning)을 유도합니다. 이 방식은 정보의 전역 관찰(global observability) 필요성을 제거하여 에이전트의 독립적인 계획을 촉진합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 중앙 집중식 및 분산 방법보다 에이전트 간 정보 공유를 줄이면서도 언제나 충돌 없는 솔루션을 일관되게 찾아냅니다. 특히, 더 많은 에이전트가 있는 대규모 시나리오에서도 유효한 솔루션을 도출할 수 있음을 보여줍니다. 이 연구는 두 가지 주요 연구 질문을 통해 성능, 솔루션 품질, 확장성에 관한 비교를 실시하였습니다.



### Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols (https://arxiv.org/abs/2510.09462)
- **What's New**: 이 논문은 AI 제어 프로토콜의 취약점에 대해 분석하고 있습니다. 특히, 신뢰할 수 없는 대형 언어 모델(LLM) 에이전트가 적응형 공격을 수행할 수 있는 가능성을 탐구합니다. 기존의 연구와 달리, 이는 보안 문제로 빈번하게 다루어지는 것에서 탈피하고 있습니다.

- **Technical Details**: 여기서 다루는 적응형 공격은 신뢰할 수 없는 모델이 제어 프로토콜과 모니터 모델에 대한 지식을 가지고 있다는 것입니다. 이는 모델이 후속 지식 컷오프로 훈련되거나 자율적으로 정보를 검색할 수 있을 때 유효합니다. 예를 들어, 저자들은 제어 프로토콜의 약점을 노리는 간단한 공격 벡터를 제시하며, 모델 출력을 통해 알려진 프로프트를 주입하는 방식을 설명합니다.

- **Performance Highlights**: 논문에서 제시한 공격은 현재의 모니터에 의존하는 모든 프로토콜에 대해 보편적으로 작동하며, 최신 Defer-to-Resample 프로토콜의 경우, 재샘플링 과정이 프롬프트 주입을 강화하여 공격의 유효성을 높이는 결과를 초래했습니다. 이 연구는 AI 제어 메커니즘의 미래 평가에서 적응형 공격을 표준 구성 요소로 포함시켜야 한다고 주장합니다.



### Failure Prediction at Runtime for Generative Robot Policies (https://arxiv.org/abs/2510.09459)
Comments:
          Accepted to NeurIPS 2025

- **What's New**: 이 논문에서는 인간 중심 및 안전이 중요한 환경에서 로봇을 배치하기 위해서 실행 중 조기 실패 예측이 필요하다고 강조합니다. 새로운 프레임워크인 FIPER(과정적 실패 예측, Failure Prediction at Runtime)를 제안하며, 이 프레임워크는 실패 데이터를 필요로 하지 않습니다.

- **Technical Details**: FIPER는 두 가지 핵심 실패 지표를 식별합니다. 첫 번째는 정책의 임베딩 공간에서 무작위 네트워크 증류(random network distillation)를 통해 감지된 분포 외(out-of-distribution, OOD) 관측값이며, 두 번째는 새로운 action-chunk 엔트로피 점수를 사용해 측정된 생성된 행동의 높은 불확실성입니다.

- **Performance Highlights**: FIPER는 다양한 실패 모드가 있는 다섯 개의 시뮬레이션 및 실제 환경에서 평가되었습니다. FIPER는 실제 실패를 benign OOD 상황과 더 잘 구별하고, 기존 방법들보다 실패를 더 빨리 정확하게 예측하는 성능을 보였습니다.



### Bandits with Single-Peaked Preferences and Limited Resources (https://arxiv.org/abs/2510.09425)
- **What's New**: 이번 연구에서는 예산 제약이 있는 온라인 확률적 매칭 문제를 다룹니다. 사용자들의 성향이 단순한 형태인 단한피크(single-peaked) 선호를 기반으로 하는 효율적인 알고리즘을 개발하여 NP-hard 문제를 해결합니다. 기존의 탐색(exploration)과 활용(exploitation)을 균형있게 맞추는 접근 방식에서 벗어나, 우리는 효율적인 전략을 고안하였습니다.

- **Technical Details**: 제안된 SP-Matching 알고리즘은 O(K²B + K²U) 시간 내에 예산 제약이 있는 매칭 문제를 최적화합니다. 사용자의 유틸리티가 단일 봉우리를 가지는 성격을 띠는 경우에 해당합니다. 알고리즘이 알려진 경우에는 UCB 기반의 MvM 알고리즘이 Õ(U√TK)라는 손실 한계를 달성합니다.

- **Performance Highlights**: 이 알고리즘은 SP 구조가 알려지지 않은 경우에도 잘 작동하며, EMC라는 탐색-수용(explore-then-commit) 알고리즘을 통해 근사한 SP 순서를 추정합니다. 또한, 최악의 경우 손실 한계를 이론적으로 제시하며, SP 선호가 학습 문제를 단순화하지 않을 수 있음을 입증합니다.



### The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach (https://arxiv.org/abs/2510.09424)
- **What's New**: 본 논문은 Speech-LLM을 사용한 종단 간 Spoken Dialog State Tracking (DST) 기술에 대한 비교 연구를 제시합니다. 전통적인 멀티모달 컨텍스트 접근 방식, 전체 구술 이력, 그리고 압축된 구술 이력을 평가하여, 전체 대화를 입력으로 제공했을 때 가장 높은 성능을 나타냄을 보여줍니다. 개발된 방법이 기존 방법들보다 우수한 성능을 달성했음을 통해, 컨텍스트 활용의 효율성을 강조합니다.

- **Technical Details**: 연구의 방법론에서는 대화에서 사용자의 의도 및 관련 정보를 구조화된 형태로 요약하는 Spoken Dialog State Tracking의 역할을 설명합니다. 구술 대화 턴의 시퀀스를 입력으로 받아, 관련 도메인 및 슬롯-값 쌍을 예측하는 구조를 채택합니다. 대화 이력을 처리하고, Dense Representation을 계산하여, 이를 Large Language Model (LLM)에 전달하는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과, 전체 대화 이력을 사용하는 접근이 최적의 성능을 보였으며, 이는 다른 방법들에 비해 상당한 성과를 기록했습니다. 특히 압축 모듈을 활용한 접근 방식 또한 강력한 trade-off를 제공하면서도 경쟁력 있는 정확도를 유지함을 입증했습니다. 이러한 성과는 대화 시스템의 컨텍스트 활용 방식의 개선에서 비롯된 것으로 분석되었습니다.



### On the Representations of Entities in Auto-regressive Large Language Models (https://arxiv.org/abs/2510.09421)
Comments:
          Accepted at BlackBoxNLP@EMNLP2025

- **What's New**: 이 논문에서는 지명(Named entities)이 LLM(대형 언어 모델)에서 내부적으로 어떻게 표현되는지를 연구하기 위해 새로운 프레임워크인 entity mention reconstruction을 소개합니다. 기존의 연구는 주로 명시적 관계를 다루었으나, 개체 표현(entity representations)에 대한 이해는 부족했습니다. 이 연구는 LLM의 내부 표현에서 개체 언급을 생성할 수 있는지, 그리고 멀티 토큰(여러 단어로 이루어진) 개체를 어떻게 인코딩하는지를 탐구합니다.

- **Technical Details**: 제안된 방법은 _task vectors_를 활용하여 LLM의 숨겨진 상태(hidden states)에서 유도된 다양한 개체 표현으로부터 멀티 토큰 언급을 일관되게 생성합니다. 이로써 'Entity Lens'를 소개하며, 이는 기존의 _logit-lens_를 확장하여 멀티 토큰 언급 예측을 가능하게 합니다. 본 연구는 LLM이 훈련 중에 보지 못한 개체를 포함하여 멀티 토큰 개체를 표현하고 조작하기 위해 개체별 메커니즘(entity-specific mechanisms)을 개발한다는 새로운 증거를 제시합니다.

- **Performance Highlights**: 연구 결과에 따르면 LLM들은 훈련 데이터에 포함되지 않은 정보까지 포함하여 다양한 멀티 토큰 개체를 효과적으로 표현하고 조작할 수 있는 능력이 있음을 보여줍니다. 이 접근 방식은 향후 LLM의 지식 표현 및 처리 방식에 대한 깊은 통찰력을 제공할 것입니다. 추가적으로, 이 연구는 LLM의 개체 인식과 조작에 대한 이해를 확대하는 데 기여합니다.



### Beyond Single-Granularity Prompts: A Multi-Scale Chain-of-Thought Prompt Learning for Graph (https://arxiv.org/abs/2510.09394)
Comments:
          under review

- **What's New**: 본 논문에서는 기존의 그래프 프롬프트 방법론의 한계를 극복하기 위해 Multi-Scale Graph Chain-of-Thought (MSGCOT) 프레임워크를 제안합니다. 기존 방법들이 단일 수준 (single-granularity) 프롬프트 생성에 국한된 반면, MSGCOT는 다중 수준 구조 정보를 통합하여 다양성을 높입니다. 이는 주어진 그래프 데이터를 더 효과적으로 활용하고, 보다 풍부한 의미를 담은 프롬프트를 생성할 수 있게 합니다.

- **Technical Details**: MSGCOT 프레임워크는 경량 저랭크 네트워크를 통해 다중 수준 (multi-scale) 구조 특징을 포착하며, 이를 계층적 기저 벡터로 변환합니다. 또한, 추론 과정에서 인간의 인지 과정을 모방하여 다단계 (multi-step) 추론을 통해 다중 수준 정보를 동적으로 통합합니다. 이러한 접근은 전통적인 단일 수준 방법들보다 더 정교한 계층적 의미를 포착할 수 있도록 돕습니다.

- **Performance Highlights**: 여덟 개의 벤치마크 데이터셋에서 실시한 광범위한 실험 결과, MSGCOT는 최신 단일 수준 그래프 프롬프트 학습 방법보다 뛰어난 성능을 보였습니다. 특히 몇 차례의 훈련만으로도 강력한 결과를 보여주며, 소수의 학습 샘플로도 높은 정확도를 달성할 수 있음을 입증합니다.



### ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users (https://arxiv.org/abs/2510.09393)
- **What's New**: 이번 논문에서는 e-commerce 추천 시스템에서 저활동 사용자(低活躍ユーザー)의 전환율(CVR)을 더 정확하게 예측하기 위해 ChoirRec라는 새로운 프레임워크를 제안합니다. 이 접근법은 프레임워크의 각 구성 요소가 어떤 방식으로 작용하는지를 설명하면서, 사용자의 행동 신호에서 소음(noise)을 줄이고, 사용자의 고유 데이터를 활용하여 보다 신뢰할 수 있는 정보를 생성합니다.

- **Technical Details**: ChoirRec은 대형 언어 모델(LLM)의 의미론적 능력을 활용하여 신뢰할 수 있는 사용자 클러스터를 형성하고, 저활동 사용자에 대한 CVR 예측을 향상시킵니다. 이 구조는 세 가지 핵심 모듈로 구성되어 있으며, 이들은 각각 의미적 그룹 생성을 위한 모듈, 그룹 인식을 포함한 계층적 표현 모듈, 그리고 그룹 지식을 효과적으로 학습할 수 있는 다중 세분화 모듈로 구성됩니다.

- **Performance Highlights**: 실험 결과, Taobao에서의 오프라인 평가에서 저활동 사용자에 대한 GAUC가 1.16% 개선되었으며, 온라인 A/B 테스트에서는 주문량이 7.24% 증가하여 실질적인 비즈니스 가치를 보여주었습니다. 이러한 결과는 ChoirRec의 효과적인 활용이 저활동 사용자의 CVR 예측에 큰 기여를 할 수 있음을 강조합니다.



### Design Principles for Sequence Models via Coefficient Dynamics (https://arxiv.org/abs/2510.09389)
- **What's New**: 이 연구는 다양한 시퀀스 모델 아키텍처를 정리하고 비교할 수 있는 통합된 프레임워크를 개발했습니다. 이 프레임워크는 과거 벡터의 선형 조합을 제어하는 коэффициент (coefficients)가 자율 선형 동적 시스템의 출력으로 표현될 수 있음을 보여줍니다. 이를 통해 Softmax attention이나 다양한 모델의 공통 수학적 구조를 드러내고, 모델 속성과 설계 원칙을 연결 짓는 기초를 마련합니다.

- **Technical Details**: 제안된 프레임워크는 동적 시스템 이론에 기반하여 시퀀스 모델을 설계하는 원칙들을 정립합니다. 특히, 주어진 입력에 대한 선형 조합의 коэффициент이 임펄스 입력에 의해 구동되는 자율 선형 동적 시스템의 출력으로 표현될 수 있음을 설명합니다. 이 과정은 모델 아키텍쳐의 기본 구성 요소들의 역할을 명확히 하여 아키텍처 선택과 모델 성능 간의 관계를 제시합니다.

- **Performance Highlights**: 실험적 검증을 통해 이론이 실제에서 잘 작동함을 입증했습니다. 이번 연구는 입력 선택성, 안정성 요구 조건, 모델의 표현 능력 간의 트레이드오프를 분석하는 데 중요한 정보를 제공합니다. 이러한 접근 방식은 최신 아키텍처를 통합하고, 벤치마크에 의존하지 않고도 체계적인 비교를 가능하게 합니다.



### Task-Level Insights from Eigenvalues across Sequence Models (https://arxiv.org/abs/2510.09379)
- **What's New**: 이 연구는 소프트맥스 어텐션(softmax attention)과 상태 공간 모델(state space models, SSMs)을 동적 시스템(dynamical systems) 관점에서 분석함으로써 두 기법의 정보 처리 방식을 비교하고 향후 개발 방향을 제시하고 있습니다. 논문에서는 고유값(eigenvalues) 스펙트럼을 통해 각각의 모델이 메모리 및 장거리 의존성(long-range dependency)을 어떻게 다루는지에 대한 통찰을 제공합니다. 이를 통해 서로 다른 아키텍처가 어떻게 안정성 및 표현 능력을 균형 있게 유지하는지를 확인합니다.

- **Technical Details**: 이 연구는 최근 제안된 동적 시스템 프레임워크를 기반으로 하여 다양한 어텐션 메커니즘과 SSM의 고유값 스펙트럼을 실증적으로 분석합니다. 동적 시스템은 입력에 따라 내부 숨겨진 상태가 어떻게 변화하는지를 모델링하며, 이를 통해 메모리 유지(memory retention)와 정보 흐름(information flow)에 대한 이해를 심화합니다. 연구진은 이러한 고유값이 각각의 모델 구조에 따라 어떻게 변화하는지에 대한 분석을 수행하였으며, 고유값의 특성에 따라 메모리 요구 사항과 성과 간의 상관관계를 밝혔습니다.

- **Performance Highlights**: 고유값 분포와 성능 간의 연관성을 발견하였고, 메모리 요구사항에 따라 특정 패턴이 나타나는 것을 확인했습니다. 장기 메모리가 중요한 과제에서는 고유값이 1에 가까운 분포를 보이며, 특정 메모리 선택성이 요구될 경우에는 0에 가까운 고유값이 집중되는 경향이 있습니다. 이러한 통찰은 시퀀스 모델의 동작을 이해하고, 아키텍처 결정 과정을 안내하는 도구로서 고유값 분석의 잠재력을 강조합니다.



### The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton (https://arxiv.org/abs/2510.09378)
- **What's New**: 최근 LLM(pretrained large language model) 사전 훈련을 가속화하기 위한 노력은 두 번째 차수 구조를 활용하는 계산 효율적인 근사에 집중되고 있습니다. 이 연구는 Gauss-Newton(GN) 전처리를 적용하여 1억 5천만 개의 파라미터를 가진 변환기 모델의 반복 복잡성에 대한 실용적인 상한선을 수립합니다. 실험 결과, 전체 GN 업데이트는 SOAP 및 Muon과 같은 기존 최적화 기법에 비해 훈련 반복 횟수를 5.4배 줄이는 상당한 개선을 보여줍니다.

- **Technical Details**: 최적화 방법 개선은 LLM 훈련의 효율성을 높이는 중심 전략으로 자리잡고 있습니다. 이 연구에서는 Gauss-Newton 방법을 활용하여 전체 두 번째 차수 최적화의 성능 한계를 확인하고, 특히 반복 복잡성에 대한 성능을 측정합니다. 또한, 각 레이어에서의 Hessian 구조를 비교하여 레이어 와이즈 접근법이 상당한 성과를 거둘 수 있다는 점을 강조합니다.

- **Performance Highlights**: 본 연구는 이상적인 두 번째 차수 방법인 전체 Gauss-Newton이 SOAP 최적화 기법에 비해 5.4배의 반복 복잡성 감소로 실질적인 개선을 가져온다고 보고합니다. 또한, 레이어 와이즈 접근 방식이 기능적 제약에도 불구하고 SOAP 및 Adam을 상회하는 성과를 보였으며, 이는 레이어별 곡률 정보만으로도 상당한 컴퓨팅 효율성을 달성할 수 있음을 시사합니다.



### deep-REMAP: Probabilistic Parameterization of Stellar Spectra Using Regularized Multi-Task Learning (https://arxiv.org/abs/2510.09362)
Comments:
          14 pages. Accepted for publication in RASTI

- **What's New**: 이번 연구에서는 deep-REMAP이라는 새로운 딥 러닝 프레임워크를 개발하여 관측 스펙트럼으로부터 별의 대기 매개변수를 예측하는 방법을 제안합니다. 이 모델은 PHOENIX 합성 스펙트럼 라이브러리를 기반으로 훈련되며, MARVELS 조사에서 얻은 FGK 왜소성 스펙트럼의 작은 하위 집합을 Fine-tuning하여 성능을 극대화합니다. 특히 이 모델은 추후 다른 조사나 합성 라이브러리에도 쉽게 확장 가능합니다.

- **Technical Details**: deep-REMAP은 다중 작업(multi-task) 접근 방식을 활용한 정규화된 딥 러닝 모델로, 데이터의 비대칭 손실 함수와 임베딩 손실을 결합하여 해석 가능성을 높였습니다. 이 모델은 MARVELS 조사의 732개의 FGK 거대 후보 별에 대한 예측을 수행하며, 유효 온도(Effective Temperature), 표면 중력(Surface Gravity), 금속licity(Metallicity)를 정확히 회복하는 성능을 자랑합니다. 특히, deep convolutional neural network(CNN) 구조를 채택하여 복잡한 관계를 잘 포착할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: deep-REMAP은 MARVELS 교정 별 30개에 대해 검증 시, 유효 온도를 약 75 K의 정밀도로 회복하는 등 매우 높은 정확도를 나타냈습니다. 이 모델은 다양한 새로운 관측과 합성 스펙트럼 데이터에 효과적으로 적용 가능하며, 최신 기술을 통해 성능을 극대화할 수 있습니다. 나아가 고차원 스펙트럼 데이터를 성공적으로 해석할 수 있는 능력을 보여주며, 우주 천문학의 새로운 자동화 경로를 제시합니다.



### FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inferenc (https://arxiv.org/abs/2510.09332)
Comments:
          Accepted by EMNLP 2025

- **What's New**: 이번 논문에서는 Fine-grained Low-Rank Compressor (FLRC)를 제안하여 기존의 낮은 순위 압축 방법의 한계를 극복하고자 합니다. FLRC는 각 층의 최적 순위 배분을 효율적으로 결정하고, 점진적(low-rank) 디코딩을 통해 텍스트 생성 품질을 유지합니다. 이 방법은 기능적으로 우수성을 입증하며, 기존의 최첨단 방법들보다 성능을 17% 향상시키는 결과를 보여줍니다.

- **Technical Details**: FLRC는 두 가지 주요 구성 요소로 이루어져 있습니다: Fisher 기반 층별 순위 배치 알고리즘과 점진적(low-rank) 디코딩 전략입니다. 이를 통해 각 층의 중요도에 따라 압축 비율을 동적으로 조정하며, 초기에는 보수적인 압축 비율을 사용해 정확도를 보장하며, 이후에는 점진적으로 압축 비율을 높입니다. 이러한 접근 방식은 메모리 사용량을 줄이면서도 성능을 높이는 대신 압축을 고르게 분배할 수 있게 합니다.

- **Performance Highlights**: FLRC는 DialogSum과 CNN/DM과 같은 여러 요약 데이터셋에서 실험을 진행하였고, 기존의 방법에 비해 최대 17.35% 더 높은 ROUGE-L 점수를 기록했습니다. 이러한 실험 결과는 FLRC가 LLM(Large Language Model) 압축에 있어 더욱 효율적이고 신뢰할 수 있는 방식이라는 것을 보여줍니다. 또한, FLRC는 검색 시간을 최대 49배 줄이는 성과를 달성하였으며, 이는 저자들이 제시한 모델 압축의 새로운 기준이 됩니다.



### Randomized HyperSteiner: A Stochastic Delaunay Triangulation Heuristic for the Hyperbolic Steiner Minimal Tr (https://arxiv.org/abs/2510.09328)
- **What's New**: 이번 연구에서는 하이퍼볼릭 공간에서 Steiner Minimal Trees (SMTs)를 구성하는 문제를 다룹니다. NP-hard 문제인 SMT 계산과 기존의 하이퍼볼릭 휴리스틱들은 결정론적이며 주로 국소적으로 최적화된 구성에 갇히는 경향이 있습니다. 이를 해결하기 위해, 우리는 랜덤화된 Delaunay 삼각형 분할 휴리스틱인 Randomized HyperSteiner (RHS)를 제안하고, 이 과정에서 임의성을 도입하여 후보 트리를 개선하는 방법을 소개합니다.

- **Technical Details**: RHS는 Riemannian gradient descent 최적화를 활용하여 하이퍼볼릭 공간에서 SMT를 구성하는 최신의 확률적 접근법입니다. 하이퍼볼릭 기하학은 나무 추론을 위한 적합한 공간으로써, 계층적이고 기하급수적으로 성장하는 구조를 효과적으로 임베딩할 수 있습니다. 이러한 새로운 접근 방식은 기존의 HyperSteiner 알고리즘의 한계를 극복하고, 더 나은 전체 트리 구성으로 이어지는 더 넓은 탐색 기회를 제공합니다.

- **Performance Highlights**: RHS는 합성 데이터 세트와 실제 단일 세포 전사체 데이터에서 Minimum Spanning Tree (MST), Neighbour Joining 및 기존 HyperSteiner (HS)보다 꾸준히 우수한 성능을 보였습니다. 특히, 경계 근처의 구성에서는 HS 대비 총 길이를 32% 이상 줄이는 성과를 나타냈으며, 이는 다양한 데이터 환경에서 RHS의 효과적이고 견고한 성능을 증명합니다.



### Rate optimal learning of equilibria from data (https://arxiv.org/abs/2510.09325)
- **What's New**: 이번 연구에서는 Multi-Agent Imitation Learning (MAIL)에서 비대화형(non-interactive) MAIL의 한계를 규명하고, 근사 최적 샘플 복잡도를 갖춘 첫 번째 대화형(interactive) 알고리즘을 제시했습니다. 비대화형 환경에서는 모든 정책 편차 집중도 계수(all-policy deviation concentrability coefficient)를 복잡성 측도로 활용하여, Behavior Cloning (BC) 알고리즘이 최적의 속도를 갖는다는 것을 입증합니다. 이러한 새로운 접근 방법에 의해, 학습 성능을 향상시킬 수 있는 잠재력 있는 알고리즘인 MAIL-WARM이 등장했습니다.

- **Technical Details**: 우리는 Markov 게임의 개념을 공식화하여 두 플레이어 간의 제로섬(zero-sum) 게임을 살펴봅니다. 이 게임은 유한한 상태 공간, 각 플레이어의 유한한 행동 공간, 그리고 전이 동역학을 포함한 튜플로 구성됩니다. 비대화형 환경에서는 주어진 데이터셋을 바탕으로 샘플 복잡도가 Ω(𝒞max𝜖−2)로 제한됨을 보여주었으며, 이는 비대화형 MAIL 알고리즘의 성능 한계를 나타냅니다.

- **Performance Highlights**: MAIL-WARM 알고리즘은 기존 최상의 샘플 복잡도를 𝒪(𝜖−8)에서 𝒪(𝜖−2)로 개선하며, 이는 비대화형 환경에서의 이론적 하한에 부합합니다. 실험 결과, MAIL-WARM은 Behavior Cloning이 실패하는 환경에서도 회복하는 데 성공하였으며, 기존의 다른 대화형 MAIL 알고리즘들과 비교하여 월등한 성능을 보였습니다. 이러한 결과는 MAIL-WARM의 이론적 배경을 강화하는 데 기여합니다.



### Verifying Chain-of-Thought Reasoning via Its Computational Graph (https://arxiv.org/abs/2510.09312)
- **What's New**: 이번 연구는 Circuit-based Reasoning Verification (CRV)라는 새로운 화이트박스 방식으로 Chain-of-Thought (CoT) 추론 검증 방법을 제안합니다. 기존의 블랙박스 및 그레이박스 방법의 한계를 극복하고, 오류의 원인을 보다 깊이 이해할 수 있도록 돕습니다. 연구진은 올바른 CoT 단계의 구조적 특성이 잘못된 단계와 다르다는 가설을 세우고, 이를 기반으로 오류 패턴을 식별합니다.

- **Technical Details**: 연구에서 사용된 attribution graph는 모델의 구성 요소 간의 인과적 정보 흐름을 구조적으로 표현하는 그래프입니다. 각 단계에서 생성된 그래프를 분석하여, 구조적 핑거프린트를 추출하고 이를 진단 분류기(stclassifier)에 입력하여 추론 단계의 정답 여부를 예측합니다. 이 방법론은 모델 내부의 계산 프로세스를 면밀히 분석하여 오류를 정확하게 식별하고 수정하는 데 도움을 줍니다.

- **Performance Highlights**: 제안한 CRV 방법론은 오류의 구조적 서명이 도메인 특이적임을 발견하였으며, 이는 서로 다른 추론 작업에서 실패가 고유한 계산 패턴으로 나타남을 보여줍니다. 또한, 이 연구는 개별 transcoder 기능에 대한 targeted intervention을 통해 잘못된 추론을 성공적으로 수정하는 방법을 제공함으로써, LLM의 추론에 대한 보다 깊은 원인 분석을 가능하게 합니다.



### A Model-Driven Engineering Approach to AI-Powered Healthcare Platforms (https://arxiv.org/abs/2510.09308)
Comments:
          Disclaimer: This manuscript is currently under review at * MDPI Informatics*

- **What's New**: 이번 연구는 인공지능(AI)이 헬스케어 분야에서 더 정확한 진단과 개인 맞춤형 치료를 지원할 수 있는 가능성을 제시합니다. 하지만 데이터 소스의 분산, 엄격한 개인정보 보호 규정, 신뢰할 수 있는 임상 시스템 구축의 기술적 복잡성 등의 도전 과제가 여전히 존재합니다. 이를 해결하기 위해 헬스케어 AI에 특화된 모델 기반 엔지니어링(Model Driven Engineering, MDE) 프레임워크를 도입하였습니다.

- **Technical Details**: 이 프레임워크는 형식적 메타모델(formal metamodels), 도메인 특화 언어(Domain-Specific Languages, DSLs), 자동화 변환을 통해 높은 수준의 사양에서 실행 가능한 소프트웨어로 이동할 수 있도록 설계되었습니다. 핵심 요소인 의료 상호 운용성 언어(Medical Interoperability Language, MILA)는 임상 의사와 데이터 과학자들이 공유 온톨로지를 사용해 쿼리와 머신러닝 파이프라인을 정의할 수 있게 해주는 그래픽 DSL입니다. federated learning 아키텍처와 결합하여 MILA는 기관 간의 협업을 촉진하며, 개인정보 보호를 유지하면서도 의미적 일관성을 보장합니다.

- **Performance Highlights**: 이 접근법은 다중 센터 암 면역 요법 연구에서 평가되었습니다. 생성된 파이프라인은 강력한 예측 성능을 보여주었으며, 서포트 벡터 머신(Support Vector Machines)은 주요 작업에서 각각 98.5%와 98.3%의 정확도를 달성했습니다. 또한, 수작업 코딩 노력을 크게 줄였습니다. 이러한 결과는 MDE 원칙인 메타모델링, 의미적 통합, 자동 코드 생성이 상호 운용 가능하고 재현 가능하며 신뢰할 수 있는 디지털 헬스 플랫폼으로 나아갈 수 있는 실질적인 경로를 제공할 수 있음을 시사합니다.



### CLARity: Reasoning Consistency Alone Can Teach Reinforced Experts (https://arxiv.org/abs/2510.09278)
- **What's New**: 새로운 방식인 CLARity는 제한된 데이터를 활용하여 추론 품질을 향상시키기 위해 설계된 비용 효율적인 강화 학습(RL) 프레임워크입니다. 기존의 방식들이 높은 비용과 리소스를 요구했던 것에 반해, CLARity는 소형 일반-purpose LLM만으로도 효과적인 모델 훈련을 가능하게 합니다. 이 프레임워크는 일관성을 인지하는 보상 메커니즘을 통합하여 고품질의 추론을 촉진합니다.

- **Technical Details**: CLARity는 두 단계의 정제-모니터 파이프라인을 통해 일관성 있는 보상을 RL 훈련에 통합합니다. 첫 번째 단계에서는 모델의 출력 구조를 개선하여 명확한 옵션별 추론을 유도하고, 두 번째 단계에서는 일관성 보상 모델을 사용하여 반응을 모니터링합니다. 이를 통해 모델은 일관된 추론을 하도록 유도되며, 제한된 MCQ 데이터 활용을 극대화하기 위해 동적 데이터 재구성 전략도 사용됩니다.

- **Performance Highlights**: 실험 결과, CLARity는 응답의 일관성을 16.5%, 정확성은 7.5% 향상시켰습니다. 인간 평가도 CLARity의 모델이 일관성과 전문성을 모두 개선했다고 확인했습니다. 이 연구는 작은 모델이 전문가 모델을 효과적으로 안내할 수 있는 일반화 가능한 솔루션을 제공합니다.



### Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation (https://arxiv.org/abs/2510.09275)
- **What's New**: 본 논문에서는 의학 진단 평가에 대한 동적 벤치마크인 DyReMe를 제안합니다. 이는 고정된 질문 대신 실제 상담처럼 새로운 사례를 생성하여 진단의 어려움을 높이는 것입니다. DyReMe는 정확도 외에도 진단의 신뢰성, 유용성 및 일관성을 평가하여 기존 방법의 한계를 극복하고자 합니다.

- **Technical Details**: DyReMe는 두 가지 주요 구성 요소로 이루어져 있습니다: DyGen과 EvalMed입니다. DyGen은 차별 진단 및 오진 요소를 포함하는 사실적인 질문을 생성하는 반면, EvalMed는 LLM의 정확성, 진실성, 유용성 및 일관성을 평가합니다. DyGen은 '진단 방해 요소'를 통합하여 실제 임상에서의 복잡성을 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과, DyReMe는 기존 LLM의 성능과 실제 임상 관행 간의 중요한 불일치를 드러내며, 보다 도전적이고 현실적인 질문을 생성합니다. 이는 또한 LLM에 대한 정확한 평가를 제공하여 의학 진단의 신뢰성을 높이는 데 기여합니다.



### SynthID-Image: Image watermarking at internet sca (https://arxiv.org/abs/2510.09263)
- **What's New**: 이번 논문에서는 AI가 생성한 이미지를 보이지 않게 워터마킹하는 새로운 시스템인 SynthID-Image를 소개합니다. 이 시스템은 효과성(effectiveness), 충실도(fidelity), 견고성(robustness), 보안(security) 등 다양한 기술적 요건을 충족하도록 설계되었습니다. SynthID-Image는 이미 구글 서비스에서 100억 개 이상의 이미지와 비디오 프레임을 워터마킹하는 데 사용되어 왔으며, 검증 서비스도 신뢰할 수 있는 테스터에게 제공되고 있습니다.

- **Technical Details**: SynthID-Image는 미디어의 출처를 정확하게 기록하고 검증하기 위해 워터마킹 기술을 활용하고 있습니다. 이 기술은 생성 모델과 독립적으로 작동하는 후처리(post-hoc) 방식을 사용하여, 이미지 생성 프로세스에 추가적인 부담을 주지 않으면서 효과적인 워터마킹을 구현합니다. 논문에서는 또한 외부 모델 변형인 SynthID-O를 실험적으로 평가하고, 기존의 다른 워터마킹 방법과 성능을 비교하여 우수한 결과를 보였습니다.

- **Performance Highlights**: SynthID 모델은 다양한 이미지 변환에 대해 최첨단 성능을 달성하며, 워터마크의 질, 검출 성능 및 견고성에서 새로운 기준을 제시합니다. 이 논문은 인터넷 규모에서 이미지 워터마킹 모델을 배포하는 데 대한 체계적인 설명을 제공하며, 다양한 위협 모델(threat model)과의 관련성을 분석하고 있습니다. 결과적으로, SynthID-Image는 생성적 AI가 만든 콘텐츠의 출처를 명확히 하여 신뢰성을 높이는 데 기여할 것입니다.



### Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models (https://arxiv.org/abs/2510.09259)
- **What's New**: 데이터 오염(data contamination)은 대형 언어 모델(Large Language Model, LLM)의 신뢰성 있는 평가에 중대한 위협이 됩니다. 이 문제는 벤치마크 샘플이 훈련 데이터에 우연히 포함될 때 발생하여 보고된 성능의 유효성을 훼손합니다. 본 논문은 강화 학습(Reinforcement Learning, RL) 후 훈련 단계에서 발생하는 데이터 오염 탐지의 체계적인 연구를 시작하며, Self-Critique라는 새로운 방법을 제안합니다.

- **Technical Details**: Self-Critique는 RL 후 훈련 단계에서 데이터 오염을 탐지하기 위한 목적입니다. 이 방법은 RL 단계에서 LLM의 출력 엔트로피 분포가 구체적이고 희소한 모드로 수축된다는 관찰을 바탕으로 합니다. 이를 통해 모델의 정책 붕괴(policy collapse)를 탐지하고, 주어진 문제에 대해 두 가지 서로 다른 응답을 생성하게 하여 엔트로피 공간에서 높은 유사성을 보이는 샘플을 오염된 것으로 표시합니다.

- **Performance Highlights**: Self-Critique는 여러 모델과 오염 작업에서 기존의 방법에 비해 현저히 향상된 성능을 보여줍니다. 특히, AUC(Area Under the Curve) 개선이 최대 30%까지 이루어졌으며, 기존 탐지 방법들에 비해 신뢰성 있는 오염 탐지를 가능하게 합니다. 논문은 또한 RL 전용 오염 시나리오를 시뮬레이션하기 위해 새롭게 고안한 벤치마크 RL-MIA를 소개하여, 탐지 방법의 엄격한 평가를 지원합니다.



### Obstacle Avoidance using Dynamic Movement Primitives and Reinforcement Learning (https://arxiv.org/abs/2510.09254)
Comments:
          8 pages, 7 figures

- **What's New**: 이 연구에서는 하나의 인공 데모로부터 부드럽고 최적에 가까우며 충돌이 없는 3D Cartesian 궤적을 신속하게 생성하는 방법을 제안합니다. Dynamic Movement Primitive (DMP)로 인코딩된 데모는 정책 기반 강화 학습에 의해 반복적으로 reshape되어 다양한 장애물 구성에 대한 궤적 데이터셋을 생성합니다. 이를 통해 높은 효율성과 다양한 궤적 생성을 지원하며, 기존의 RRT-Connect 기준을 초월하는 성능을 보여줍니다.

- **Technical Details**: 본 방법론은 최소-jerk 데모를 기반으로 하며, 데이터셋 생성 과정에서 가속 및 jerk 패널티를 통해 궤적의 부드러움을 보장합니다. 생성된 데이터셋은 신경망 모델에 매핑되어, 포인트 클라우드에서 자동으로 유도된 작업 매개변수를 입력으로 받아 궤적을 생성합니다. 이 방식은 각기 다른 장애물 회피 애플리케이션을 위해 다양한 모델 학습을 가능하게 합니다.

- **Performance Highlights**: 본 연구는 RRT-Connect 플래너 및 선형 기준과의 비교를 통해 계산 시간, 실행 시간, 궤적 길이 측면에서 더 나은 성능을 입증하였습니다. 이 방법은 두 개 또는 세 개의 작업 매개변수를 사용하여 다양한 장애물 구성과 엔드 이펙터 차원을 고려할 수 있으며, 멀티모달 솔루션을 찾는 데에도 성공하였습니다. 최종적으로 실험 평가 결과와 함께 향후 방향에 대해 논의합니다.



### CrisiText: A dataset of warning messages for LLM training in emergency communication (https://arxiv.org/abs/2510.09243)
- **What's New**: 이 논문에서는 CrisiText라는 대규모 경고 메시지 생성 데이터셋을 처음으로 소개합니다. 이 데이터셋은 13가지 다양한 위기 상황에 대한 경고 메시지를 포함하고 있으며, 400,000개 이상의 경고 메시지가 시민들을 지원하는 데 사용됩니다. 또한, 각 메시지에는 세 가지 불완전한 경고 유형이 첨부되어 다양한 NLG 접근 방식을 연구할 수 있도록 돕습니다.

- **Technical Details**: 데이터셋은 기존 위기 설명에서 사건의 연속성을 추출하고, 각 사건에 대한 경고 메시지를 작성하는 방식으로 생성되었습니다. 이러한 메시지는 전문가의 지침을 따르며 적절한 용어와 사실성을 보장합니다. 연구에서는 Llama 3 모델을 사용하여 경고 메시지 생성 작업을 수행하고, 이전 메시지 및 특정 지침 가이드라인의 추가 맥락이 생성에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과, ORPO와 SFT 방식에서 유사한 성능이 나타났습니다. 특히, 이미 본 상황의 경우 이전 메시지의 중요성이 드러났으며, 새로운 특정 상황의 경우 지침 가이드라인이 필수적이라는 결과가 나왔습니다. 자동 포스트 편집기 모델의 파인튜닝을 통해 잘못 작성된 경고 메시지의 품질 향상에서도 유망한 결과를 보였습니다.



### DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction (https://arxiv.org/abs/2510.09211)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)이 사용자 맞춤형 요구 사항을 만족하는 출력 형식을 준수하는 데 어려움을 겪는다는 문제를 해결하기 위한 새로운 접근 방법인 DICE(Chain-of-thought correction)를 제안합니다. DICE는 소형 언어 모델(SLM)을 사용하여 LLM의 출력을 분석하고 정제하는 경량화된 프레임워크로, 본질적으로 LLM의 지식과 추론 능력을 보존하면서도 사용자 요구 사항에 맞는 출력을 생성할 수 있게 도와줍니다. 이 방법은 복잡한 형식 요구 사항에 영향을 받지 않고 자연어 응답을 생성한 뒤 SLM을 사용하여 이를 특정 형식으로 다듬는 두 단계로 작동합니다.

- **Technical Details**: DICE는 두 가지 주요 단계로 구성되어 있습니다. 첫 번째 단계에서 LLM은 비구조적 자연어 응답을 생성하며, 이 응답은 특정 형식의 요구 사항에 영향을 받지 않습니다. 두 번째 단계에서는 교육된 SLM이 LLM의 출력을 분석하여 구조화된 형식으로 정제합니다. DICE의 주된 혁신은 모형 협동 프레임워크를 활용하여 LLM의 지시 준수 능력을 높이고, SLM이 출력을 생성하기 전에 LLM의 출력을 깊이 분석하는 '분석 후 응답' 패턴을 사용하는 것입니다.

- **Performance Highlights**: DICE는 다섯 가지 추론 벤치마크에 대한 실험을 통해 효과를 입증했으며, 평균적으로 LLM의 형식 정확도를 35.4%, 내용 정확도를 29.4% 향상시켰습니다. DICE는 거의 모든 평가된 데이터 세트에서 다른 경쟁 기초 모델들보다 두드러진 성과를 보였으며, 이는 작은 모델이 대형 모델을 보완할 수 있는 가능성을 제시합니다. 따라서 DICE는 형태 출력과 추론 성능 간의 균형을 잘 맞추며 사용자의 지시를 준수하는 능력을 향상시킵니다.



### Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs (https://arxiv.org/abs/2510.09201)
- **What's New**: 이 논문에서는 멀티모달 프롬프트 최적화(multi-modal prompt optimization)라는 새로운 문제를 정의하고, 텍스트와 비텍스트 프롬프트 쌍으로 정의된 멀티모달 공간에서의 최적화를 제안합니다. 이를 통해 기존의 텍스트 전용 프롬프트 최적화 기법들의 한계를 극복하고자 합니다. 저자들은 다수의 실험을 통해 제안된 멀티모달 프롬프트 최적화기(Multimodal Prompt Optimizer, MPO)가 효과적임을 입증하며, MLLMs의 잠재력을 실현하는 데 중요한 단계를 제시합니다.

- **Technical Details**: MPO는 프롬프트를 최적화하기 위한 통합 프레임워크로, 두 가지 주요 구성 요소인 정렬 보존 탐색(alignment-preserving exploration)과 선행 유전 선택(prior-inheritance-based selection)을 가지고 있습니다. 탐색 과정에서는 텍스트 프롬프트와 비텍스트 컴포넌트를 함께 업데이트하며, 이 업데이트는 실패 분석에서 도출된 단일 의미적 기울기에 의해 유도됩니다. 이후, 베이지안 기반의 선택 전략을 이용해 후보 프롬프트를 선택하며, 이 과정은 후보 간 성과 평가를 통해 고성능 프롬프트를 신뢰성 있게 식별합니다.

- **Performance Highlights**: MPO는 텍스트 전용 최적화 방법과 비교하여 10개 데이터셋에서 일관된 성능 개선을 입증하였습니다. 실험 결과, alignment-preserving exploration이 최적의 멀티모달 프롬프트 발견에 기여하며, prior-inherited Bayesian-UCB는 높은 성능의 프롬프트를 효율적으로 선택하여 평가 예산을 42% 절감하는 효과를 나타냈습니다. 이러한 모든 결과는 MPO가 MLLMs의 전체 기능을 활용하는 효과적인 프레임워크임을 강조합니다.



### On the Implicit Adversariality of Catastrophic Forgetting in Deep Continual Learning (https://arxiv.org/abs/2510.09181)
- **What's New**: 이 논문은 지속 학습(Continual Learning)에서 발생하는 재앙적 망각(catastrophic forgetting) 문제를 해결하기 위한 새로운 관점을 제시합니다. 새로운 과제의 훈련이 이전 과제 지식에 대한 적대적 공격으로 작용한다는 것을 보여줍니다. 새로운 과제의 기울기는 이전 과제의 손실 경관(loss landscape)과 정렬되어 이전 과제의 손실을 신속히 증가시킴을 나타냅니다.

- **Technical Details**: 우리는 두 가지 작업, 즉 이전 작업과 새로운 작업 간의 상관관계를 이론적으로 접근하여 설명합니다. 기존의 Gradient Projection (GP) 방법이 전방 전파(forward propagation)로 인한 적대적 정렬을 줄일 수 있지만, 후방 전파(backward propagation)에 따른 정렬을 처리하지 못 함을 설명합니다. 이를 해결하기 위해 backGP 방법을 제안하며, 이 방법은 GP 방법에 비해 재앙적 망각을 10.8% 줄이고 정확도를 12.7% 증가시킵니다.

- **Performance Highlights**: 성능 테스트 결과, 새로운 작업 업데이트가 이전 작업의 고차 curvature 방향에 강하게 정렬됨을 확인하였습니다. 특히, 훈련 과정 중 적대적 정렬이 지속적으로 발생하고 있으며, 이는 명시적 데이터 유사성과 상관없이 존재함을 보여줍니다. 이러한 발견은 적대적 정렬이 잊음 현상에 결정적인 역할을 하며, 이를 제거하면 잊음도 크게 감소할 수 있음을 시사합니다.



### Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction (https://arxiv.org/abs/2510.09159)
- **What's New**: 이 연구는 EHR(전자 건강 기록) 표현 방법의 체계적 벤치마크를 처음으로 제시하여, 다변량 시계열, 이벤트 스트림, 그리고 텍스트 이벤트 스트림을 비교합니다. 이 벤치마크는 MIMIC-IV와 EHRSHOT 데이터셋 등 두 가지 임상 환경에서 데이터 큐레이션과 평가를 표준화하며, 각기 다른 모델(Transformers, LSTM 등)의 성능을 분석합니다. 연구 결과, 이벤트 스트림 모델이 일관되게 최고의 성능을 나타내며, 기능 선택 전략이 임상 상황에 따라 조정되어야 함을 강조합니다.

- **Technical Details**: 연구에서는 MIMIC-IV와 EHRSHOT 두 가지 EHR 코퍼스를 사용하여 이들의 표현 방법을 검토합니다. 각각 이식장(ICU) 사망률 예측 및 30일 재입원 예측과 같은 이진 분류 작업을 포함합니다. 다변량 시계열, 이벤트 스트림, 텍스트 이벤트 스트림 각각의 프레임워크를 채택하여, 데이터 손실에 따른 메트릭 효율성 및 기능 선택 설정의 영향을 분석합니다.

- **Performance Highlights**: 실험 결과 이벤트 스트림 모델이 가장 높은 성능을 보였으며, 사전 훈련된 CLMBR 모델이 샘플 효율성이 뛰어난 것으로 나타났습니다. 단순 카운트 기반 모델이 대량의 데이터 환경에서 경쟁력을 가질 수 있음을 보여주며, 스파스 피처를 제거하는 전략이 특히 ICU 예측에 유익한 것으로 나타났습니다. 또한, 다변량 시계열 모델들이 전체적으로 보통의 성능을 기록했지만, 이벤트 스트림을 통한 접근 방식이 전반적으로 더 나은 결과를 보여주었습니다.



### Federated Data Analytics for Cancer Immunotherapy: A Privacy-Preserving Collaborative Platform for Patient Managemen (https://arxiv.org/abs/2510.09155)
Comments:
          This manuscript is currently under review at * ACM Transactions on Computing for Healthcare (HEALTH)*

- **What's New**: 이번 연구에서는 연결된 건강(Connected Health)의 다학제적 접근 방식을 소개하며, 환자의 요구를 중심으로 도구, 서비스 및 치료법을 설계하는 것이 중요하다고 강조합니다. 디지털 기술과 프로세스 혁신의 발전이 다양한 의료 데이터 소스를 통합하여 개인화된 치료를 가능하게 하고 건강 결과를 예측하는 데 기여할 것으로 기대됩니다. 그러나 데이터 아키텍처(data architecture), 응용프로그램 상호 운용성(application interoperability), 보안(security) 등의 문제는 여전히 해결해야 할 과제로 남아 있습니다.

- **Technical Details**: 이 연구는 EU에서 자금을 지원받아 진행된 프로젝트를 바탕으로, 면역치료(immunotherapy)를 받고 있는 암 환자 관리에 대한 AI 생성 솔루션을 개발하기 위한 민첩한 시스템 개발 생명주기(agile System Development Lifecycle)를 탐구했습니다. 이 디지털 프레임워크는 연속 치료 과정에서의 이해관계자(stakeholders)를 통합하고, 연합형 빅 데이터 분석(federated big data analytics)과 인공지능(artificial intelligence)을 활용하여 개인정보 보호를 보장하면서 의사결정을 개선하는 데 기여합니다.

- **Performance Highlights**: 연구에서는 치료 추천 및 부작용 예측과 같은 분석 기능을 실제 데이터를 통해 검증하였으며, 기초 연구에서 70%-90%의 정확도를 달성했습니다. 이는 프레임워크의 효과성을 입증하는 결과로, 의료 파트너와의 협력을 통해 이루어졌습니다. 따라서, 이 프레임워크는 환자 관리의 통합적인 접근 방식을 강화하는 데 기여할 수 있습니다.



### Controlled Personalization in Legacy Media Online Services: A Case Study in News Recommendation (https://arxiv.org/abs/2510.09136)
- **What's New**: 이 논문은 전통적인 뉴스 미디어가 개인화 기술을 채택하면서 편집적 가치(editorial values)를 유지할 수 있는 전략인 '제어된 개인화(controlled personalization)'의 효과를 분석합니다. 특히, 노르웨이의 주요 뉴스 기관 웹사이트에서 실시한 A/B 테스트를 통해 20%의 개인화된 추천 알고리즘이 포함된 시스템과 비 개인화된 시스템을 비교했습니다. 그 결과, 개인화가 사용자 참여를 증가시키고 관련 콘텐츠의 검색 용이성을 높인다는 점이 드러났습니다.

- **Technical Details**: 개인화된 뉴스 추천은 고도로 동적(dynamic)인 아이템 카탈로그와 짧은 기사 수명, 다양한 맥락적 요인에 의한 사용자 관심의 영향을 받는 특수한 도전을 안고 있습니다. 기존 연구들은 주로 구글 뉴스나 야후 뉴스 같은 뉴스 집계(aggregator) 플랫폼 위주의 연구에 집중되어 있으며, 전통적인 뉴스 미디어에 대한 연구는 부족한 상황입니다. 제어된 개인화에서는 편집자(editors)가 콘텐츠의 주요 측면을 제어하면서 사용자 과거 선호에 따라 일부가 개인화되며, 기존 추천 알고리즘으로 생성된 후보 세트에 추가 비즈니스 규칙이 적용되는 방식으로 운영됩니다.

- **Performance Highlights**: 이 연구에서는 개인화된 추천이 독자에게 더 많은 관심을 불러일으키고, 관련 콘텐츠 발견을 용이하게 만들어 클릭율(click-through rates) 증가와 스크롤 활동 감소로 이어지였다는 점을 강조합니다. 또한, 독자들이 더 다양한 기사에 참여하고 인기 콘텐츠에 대한 편향이 줄어들어 콘텐츠의 다양성과 범위가 향상되었습니다. 결론적으로 제어된 개인화가 전통적인 뉴스 미디어가 편집적 목표와 사용자 요구를 성공적으로 조화시킬 수 있는 유망한 접근 방법임을 시사합니다.



### On the Fairness of Privacy Protection: Measuring and Mitigating the Disparity of Group Privacy Risks for Differentially Private Machine Learning (https://arxiv.org/abs/2510.09114)
- **What's New**: 이번 연구는 기존의 공정성 침해 문제를 해결하기 위한 새로운 솔루션을 제안하는데 초점을 맞추고 있습니다. 특히, 그룹 간 개인 정보 보호 공정성을 평가하기 위한 새로운 membership inference game (MIG)을 도입하여 부족했던 연구를 보완하였습니다. 이를 통해 데이터 레코드의 worst-case 개인 정보 노출 위험을 효율적으로 감사할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 이 연구에서 도입된 MIG는 leave-one-out attack (LOOA)에 기반하여 각 데이터 레코드의 worst-case 개인 정보 노출 위험을 추정하는 구조로 되어 있습니다. 기존의 평균적인 단일 공격 방식이 각 그룹의 개별적인 위험 수준을 적절히 반영하지 못하는 한계를 극복하는데 중점을 두고 있습니다. 또한, DP-SGD 알고리즘을 그룹별로 특정 gradient clipping 기법을 통해 보완하여 개인 정보 보호의 공정성을 높이는 전략을 사용하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 MIG 기반 감사 방법이 기존의 방법들에 비해 그룹 개인 정보 위험을 더 강력하고 정확하게 측정할 수 있음을 보여주었습니다. 연구 결과, 다양한 그룹 간 개인 정보 위험의 불공정성을 효과적으로 완화하는 DPML 알고리즘이 제안되었으며, 이러한 알고리즘이 실제로 공정한 개인 정보 보호를 증진시키는데 기여할 수 있음을 확인하였습니다.



### MemLoss: Enhancing Adversarial Training with Recycling Adversarial Examples (https://arxiv.org/abs/2510.09105)
Comments:
          24 pages

- **What's New**: 본 논문에서는 MemLoss라는 새로운 접근 방식을 제안하여 머신 러닝 모델의 적대적 훈련을 개선합니다. MemLoss는 이전에 생성된 적대적 샘플, 즉 'Memory Adversarial Examples'를 활용하여 모델의 강건성과 정확성을 향상시키며, 깨끗한 데이터에 대한 성능 저하 없이 이를 달성합니다. 이 방식은 훈련 에포크 전반에 걸쳐 이러한 샘플을 사용하여 자연적인 정확성과 적대적 강건성에서 균형 잡힌 향상을 제공합니다.

- **Technical Details**: MemLoss는 훈련의 이전 에포크에서 생성된 적대적 입력을 활용하여 강건성을 높이는 새로운 접근입니다. 기존의 적대적 훈련 기법들은 매 에포크마다 생성된 적대적 샘플을 제거하는 반면, MemLoss는 이러한 샘플을 훈련 과정 내내 유지하여 모델이 과거 경험을 쌓을 수 있도록 합니다. 이를 통해 모델의 강건성을 높이고 깨끗한 데이터에 대한 정확성을 유지하거나 심지어 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 여러 데이터셋(CIFAR-10, CIFAR-100, SVHN)에서 실험 결과를 통해 MemLoss는 기존의 적대적 훈련 방법들에 비해 우수한 정확도를 보였습니다. 특히, MemLoss는 TRADES 및 HAT와 같은 다른 적대적 훈련 프레임워크와 결합할 때 성능을 유의미하게 향상시킵니다. 본 연구의 결과는 메모리를 적대적 훈련에 통합함으로써 훈련 안정성과 모델의 전반적인 성능을 향상시킬 수 있는 방법을 제시합니다.



### When a Robot is More Capable than a Human: Learning from Constrained Demonstrators (https://arxiv.org/abs/2510.09096)
- **What's New**: 이 논문에서는 제약된 전문가의 시연으로부터 학습하는 문제(LfCD)를 도입하고, 제약된 전문가의 행동을 모방하지 않고도 더 효율적인 정책을 학습할 수 있는 새로운 방법인 LfCD-GRIP을 제안합니다. 기존의 모방 학습(IM)과 역강화 학습(IRL) 접근 방식은 전문가의 시연이 제약받는 경우에 비효율적인 결과를 초래합니다. 이를 해결하기 위해, 우리는 로봇이 전문가의 행동을 넘어서 더 효율적인 경로를 탐색할 수 있도록 합니다.

- **Technical Details**: LfCD-GRIP은 목표에 대한 근접 보상(goal proximity reward)을 상태 전이(state-state transitions)에 따라서 정의됩니다. 이는 전문가 행동에서 보상을 분리하고, 탐색 중 만나는 새로운 상태에 대해서도 보상을 할당합니다. 또한 불확실한 보상 추정치를 신뢰할 수 있는 관측값과 비교하여 식별하는 신뢰도 추정기를 포함하여 강화 학습의 정확성을 높입니다.

- **Performance Highlights**: 실제 WidowX 로봇 팔을 통해 수행한 실험 결과, LfCD-GRIP은 과거의 모방 학습 방식에 비해 태스크 완료 시간을 100초에서 12초로 단축시키며 우수한 성능을 보였습니다. 이를 통해 LfCD-GRIP은 제약된 전문가 시연을 초과하여 효율적인 행동을 발견하는 데 있어 유망한 성과를 달성했습니다.



### AI and Human Oversight: A Risk-Based Framework for Alignmen (https://arxiv.org/abs/2510.09090)
Comments:
          19 pages

- **What's New**: 이 논문에서는 인공지능(AI) 기술의 발전에 따라 인간의 자율성(autonomy)을 보호하고 윤리적 의사결정을 촉진하는 방법에 대해 논의합니다. 특히, AI 시스템이 개인의 정보에 기반한 결정을 내릴 수 있는 능력인 인간의 행위능력(human agency)을 강화하고 보호하기 위한 전략을 제안합니다. 이를 통해 AI 기술이 책임감 있게 사용될 수 있도록 지속적인 인간의 참여를 확보하고자 합니다.

- **Technical Details**: 논문은 인간의 감독 메커니즘을 설계하기 위한 여러 가지 전략을 소개합니다. 여기에는 Human-in-Command (HIC), Human-in-the-Loop (HITL), Human-on-the-Loop (HOTL)과 같은 감독 모델이 포함되어 있으며, 위험 기반(risk-based) 프레임워크를 제안하여 이러한 방법의 구현을 안내하고자 합니다. AI 모델의 위험 수준을 적절한 인간 감독 형태와 연결하여 기술 혁신과 개인의 가치 및 권리 보호 간의 균형을 강조합니다.

- **Performance Highlights**: 이 논문은 AI의 책임 있는 배치를 위한 인간의 참여가 필수적임을 강조하며, 기술의 발전과 더불어 사회적 이익을 극대화하려고 합니다. AI 기술이 개인의 자율성을 안전하게 지키고 사회적 혜택을 극대화할 수 있도록 하는 것이 주요 목표입니다. 이러한 접근 방식을 통해 AI 시스템은 단순한 기술을 넘어 인간과 사회에 긍정적인 영향을 미치는 역할을 수행하도록 설계될 수 있습니다.



### Training Models to Detect Successive Robot Errors from Human Reactions (https://arxiv.org/abs/2510.09080)
Comments:
          Accepted to NERC '25

- **What's New**: 로봇이 사회에 점점 더 통합됨에 따라, 로봇 오류를 감지하는 것이 효율적인 인간-로봇 상호작용(HRI)을 위해 필수적입니다. 이 연구는 머신러닝을 이용해 인간의 반응을 기반으로 로봇의 연속적인 오류를 인식하는 방법을 모색하고 있습니다. 이전의 연구들은 인간의 반응이 로봇 오류를 나타낼 수 있음을 보였지만, 이러한 반응의 진화가 연속적인 로봇 오류를 나타내는 방식에 대한 연구는 드물었습니다.

- **Technical Details**: 26명의 참가자가 연속된 대화 오류를 일으키는 로봇과 상호작용하는 동안 비디오 데이터를 통해 행동 특징을 추출했습니다. 얼굴 특징, 몸 자세 추정, 오디오 및 텍스트 특징을 다양한 기법(OpenFace, OpenPose, openSMILE, CLIP, BERT 등)을 사용하여 확보하였습니다. 오차 감지와 연속 오류 감지 두 가지 방식으로 데이터를 라벨링하였으며, 머신러닝 모델의 입력으로 사용되었습니다.

- **Performance Highlights**: 모델의 최고 성능은 오류 감지에서 93.5%, 연속 오류 분류에서 84.1%의 정확도를 기록했습니다. LSTM과 GRU 아키텍처를 사용하여 다양한 데이터 전처리, 모달리티 선택 및 융합 방법이 분류 성능에 미치는 영향을 평가했습니다. 이 연구는 머신러닝 모델을 통한 연속적 로봇 오류 감지 가능성을 함축하고 있으며, HRI 내에서 반복적인 상호작용 중단을 이해하는 데 기여할 수 있습니다.



### Emotion-Disentangled Embedding Alignment for Noise-Robust and Cross-Corpus Speech Emotion Recognition (https://arxiv.org/abs/2510.09072)
Comments:
          13 pages, 1 figure

- **What's New**: 이 논문은 Speech Emotion Recognition (SER) 분야에서 성능을 향상시키기 위해 두 단계 접근법을 제안합니다. 첫 번째 단계에서는 Emotion-Disentangled Representation Learning (EDRL)을 사용하여 감정 카테고리 간의 유사성을 유지하면서도 특정 클래스의 차별적인 특성을 추출합니다. 두 번째 단계에서는 Multiblock Embedding Alignment (MEA)을 사용하여 이러한 표현들을 공동의 판별 잠재 공간으로 정제합니다.

- **Technical Details**: EDRL은 감정 클래스 c에 따라 입력된 음성을 특정 블록으로 나누어 두 개의 병렬 인코더를 통해 학습합니다. 두 가지 인코더는 클래스 고유의 특성을 학습하는 intra-class encoder와 모든 클래스를 아우르는 inter-class encoder로 구성됩니다. MEA는 학습된 감정 임베딩과 원래 입력 특성 간의 공분산을 극대화하여 이 두 가지 표현들을 정렬하는 방법을 사용합니다.

- **Performance Highlights**: 본 연구에서는 IEMOCAP 데이터셋의 깨끗한 샘플을 기반으로 감정 분류기를 학습하고, 잡음이 있는 음성과 교차 코퍼스 샘플에서의 성능을 평가하였습니다. 이를 통해 기존 방법에 비해 강건성과 일반화 성능이 marked하게 향상되었음을 보여주었습니다. 제안한 EDRL-MEA 접근법은 실제 환경에서의 SER 성능을 개선하는데 효과적임을 입증합니다.



### Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation (https://arxiv.org/abs/2510.09051)
Comments:
          Accepted to the EMNLP 2025 Workshop on Multilingual Representation Learning (MRL)

- **What's New**: 이번 연구에서는 Alif-1.0-8B-Instruct라는 다국어 모델을 소개합니다. 이 모델은 부족한 고급 데이터셋과 번역의 품질 저하 문제를 해결하기 위해 특별히 설계된 고품질 다국어 합성 데이터셋인 Urdu-Instruct로 훈련되었습니다. Alif-1.0-8B-Instruct는 사전 훈련된 Llama-3.1-8B를 기반으로 하여 우르두어 특정 작업에 대한 성능을 크게 향상시켰습니다.

- **Technical Details**: Urdu-Instruct 데이터셋은 51,686개의 예제로 구성되어 있으며, 생성, 윤리, QA, 추리, 번역, 분류, 감정 분석 등 7가지 주요 우르두어 작업에 대한 지침과 응답을 포함합니다. 연구진은 수정된 self-instruct 기법을 활용하여 데이터셋을 생성하였고, 이 과정에서 문화적으로 민감한 이해를 강화하는 방식으로 데이터 품질을 높였습니다. 이 모델은 다국어의 훈련 비용을 최적화하여 $100 이하의 예산으로도 높은 성능을 발휘할 수 있도록 개발되었습니다.

- **Performance Highlights**: Alif-1.0-8B-Instruct는 Llama-3.1-8B-Instruct를 포함한 다른 다국어 모델들과 비교할 때 우르두어 특정 작업에서 우수한 성능을 입증하였습니다. 특히, Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, Cohere-Aya-Expanse-8B와 같은 주요 다국어 LLMs를 뛰어넘는 결과를 보였습니다. 또한, 우르두어 관련 벤치마크에서도 뛰어난 성능을 기록하며, 높은 품질의 언어 처리 가능성을 보여주었습니다.



### Cost-Efficient Long Code Translation using LLMs while Leveraging Identifier Replacements (https://arxiv.org/abs/2510.09045)
- **What's New**: 이번 연구에서는 장기 코드 번역에 대한 새로운 접근법으로, 사용자 정의 긴 식별자를 간소화된 플레이스홀더로 대체하는 제로샷(zero-shot) 코드 번역 방법을 제안했습니다. 이 방법은 LLM(대형 언어 모델)이 코드의 논리 구조에 집중할 수 있도록 해줍니다. 이를 통해 토큰 수와 메모리 사용량이 감소하고, 긴 코드 번역의 효율성과 비용 효율성을 향상시킵니다. 실험 결과, 이 접근법이 문법적 및 계층적 정보를 보존하며 토큰 수를 줄인 번역 결과를 생성함을 입증했습니다.

- **Technical Details**: 제안된 방법에서는 긴 식별자를 사용자 제공 플레이스홀더로 대체하여 LLM의 주목 분포를 전환합니다. 긴 및 설명적인 식별자는 LLM의 주의력을 분산시키는 경향이 있는데, 이를 간결한 플레이스홀더로 압축함으로써 프로그램의 구문 및 제어 흐름 구조에 대한 우선순위를 높일 수 있습니다. 이 메커니즘은 노이즈를 줄이고 LLM이 코드를 번역하는 동안 실행 관련 관계를 포착하는 데 도움을 줍니다. 또한, 이 방법은 필수 식별자와 비필수 식별자를 구분하여 번역 정확성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LLM이 긴 코드 시퀀스를 효과적으로 처리하며, 번역 작업 중 문법적 및 계층적 정보를 유지합니다. 코드 번역에서 LLM의 토큰 소비를 줄이는 동시에 실행 가능성과 구조적 일관성을 유지하는 데 중점을 두었습니다. 이 연구는 다양한 LLM에 적용 가능한 프레임워크를 제공하며, 산업 규모의 코드 번역 작업에서의 비용 효율성을 크게 향상시키는 가능성을 보여줍니다.



### Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach (https://arxiv.org/abs/2510.09041)
- **What's New**: 이 논문에서는 지능형 일반합 제약 적대적 강화 학습(Intelligent General-sum Constrained Adversarial Reinforcement Learning, IGCARL)을 제안합니다. 기존의 방법들이 단기적인 적대적 공격에 중점을 두어 전략적인 위협에 대한 대응력이 떨어진 문제를 해결하고자 합니다. IGCARL은 전략적인 목표를 가진 적대자와 강력한 주행 에이전트를 결합하여 자율 주행 안전성을 높이는 새로운 접근법입니다.

- **Technical Details**: IGCARL은 강화 학습의 시간적 의사결정 기능을 활용한 전략적 다단계 공격을 수행하는 적대자를 설계하였습니다. 이 적대자는 안전 중대한 사건을 유도하는 데 중점을 두고 일반합 목표를 채택합니다. 이와 함께, 에이전트는 적대자와 상호작용하며 적대적 공격에 대해 강인한 자율 주행 정책을 개발하도록 학습합니다. 또한, 제약 조건 아래에서 최적화하여 적대적 환경에서 안정적인 학습을 보장합니다.

- **Performance Highlights**: 설문조사 결과, IGCARL은 최신 방법에 비해 최소 27.9%의 성공율 향상을 달성하였으며, 이는 적대적 공격에 대한 강한 강인성을 보여줍니다. 이 연구는 DRL 기반 자율 주행의 안전성과 신뢰성을 향상시키기 위한 새로운 경로를 제시합니다. 따라서 IGCARL은 실제 도로에서의 배포에 중요한 이론적 및 실용적 기여를 합니다.



### Déréverbération non-supervisée de la parole par modèle hybrid (https://arxiv.org/abs/2510.09025)
Comments:
          in French language

- **What's New**: 본 논문에서는 음성 소거 시스템을 개선하기 위한 새로운 훈련 전략을 소개합니다. 이 방법은 오직 울림 음성만을 사용하여 비감독 방식으로 작동하며, 기존 알고리즘들이 필요한 쌍으로 된 건조/울림 데이터의 의존성을 줄입니다.

- **Technical Details**: 제안된 접근 방식은 제한된 음향 정보, 특히 울림 시간(reverberation time, RT60)을 활용하여 소거 시스템을 훈련합니다. 이는 기존의 데이터 수집 방식에 대한 의존도를 감소시키는 혁신적인 방법입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신 기술(state-of-the-art)과 비교하여 다양한 객관적 지표(objective metrics)에서 더욱 일관된 성능을 보임을 입증하였습니다.



### Value-State Gated Attention for Mitigating Extreme-Token Phenomena in Transformers (https://arxiv.org/abs/2510.09017)
- **What's New**: 이 논문은 Transformer 아키텍처에서 발생하는 extreme-token 현상을 해결하기 위해 Value-State Gated Attention (VGA)라는 새로운 메커니즘을 제안합니다. VGA는 모델이 'no-op' 주의를 효율적으로 수행할 수 있도록 설계된 구조적 메커니즘으로, 주의 기울기와 값 상태 업데이트를 더 효과적으로 분리합니다. 이 접근 방법은 모델의 성능 향상과 더불어 양호한 양자화 신뢰도를 제공합니다. 또한, VGA는 Transformer 기반 모델에 쉽게 적용될 수 있는 최소 침투성과 일반적인 향상 기능을 가지고 있습니다.

- **Technical Details**: VGA는 값 벡터(V)에서 직접 계산되는 학습 가능한 데이터 의존 게이트를 도입하여 주의 출력의 조절을 가능케 합니다. 이 게이트는 값 상태에서 반응하는 함수로 작동하여, 토큰의 기여를 억제할 수 있는 직접적인 규제 경로를 제공합니다. VGA의 설계는 극단적인 토큰 현상을 완화하기 위한 피드백 제어 시스템을 수립하며, 이를 통해 높은 주의가 가치 규범을 억압하는 압력과 분리됩니다.

- **Performance Highlights**: 실험을 통해 VGA는 주의_sink의 형성을 크게 줄이고, 가치 상태 규범을 안정화하여 모델의 전반적인 성능을 향상시키는 데 성공했습니다. VGA는 양자화 신뢰도와 모델 가시성을 강화하며, 이러한 이점을 통해 논문에서 제안하는 접근 방식이 기존의 방법들보다 유리함을 입증합니다.



### DiTSinger: Scaling Singing Voice Synthesis with Diffusion Transformer and Implicit Alignmen (https://arxiv.org/abs/2510.09016)
Comments:
          under review

- **What's New**: 최근 확산 기반의 Singing Voice Synthesis (SVS) 기술이 높은 표현력을 보여주지만, 데이터 부족과 모델의 확장성에 한계가 있습니다. 본 연구에서는 고정 멜로디와 다양한 LLM 생성 가사가 결합된 간단한 seed 세트를 통해 인체 녹음 데이터를 구축하고, 이를 기반으로 500시간 이상의 고품질 중국어 노래 데이터를 합성하는 두 단계의 파이프라인을 도입하였습니다. DiTSinger라는 이름의 새로운 Diffusion Transformer를 제안하여, 음악의 충실도를 높이고, 음소 시간 정렬을 개선하는 기법을 제공합니다.

- **Technical Details**: 이 연구에서는 'Diffusion Transformer (DiT)'를 활용하여 멜로디 모델을 구현하는 두 단계의 데이터 구축 파이프라인을 소개합니다. 첫 번째 단계인 'Recording-fitting Phase'에서는 고정된 멜로디와 LLM이 생성한 다양한 가사가 결합되어 소량의 데이터 세트를 만듭니다. 이후 이 데이터를 기반으로한 'Data Expansion Phase'에서는 각 모델이 새롭게 생성된 가사를 기반으로 대규모의 노래 데이터를 합성할 수 있습니다.

- **Performance Highlights**: DiTSinger는 500시간의 합성된 노래 데이터를 생성하는 데 성공하였으며, 기존의 SVS 모델들에 비해 데이터의 다양성과 일반화 능력을 크게 향상시켰습니다. 객관적인 평가 지표와 주관적인 MOS 테스트에 따르면, 이 접근법은 높은 충실도와 확장성을 보이며, 정렬 없는 음성 합성에서도 우수한 성능을 보여줍니다.



### SQS: Bayesian DNN Compression through Sparse Quantized Sub-distributions (https://arxiv.org/abs/2510.08999)
- **What's New**: 이 논문은 Bayesian Variational Learning을 활용하여 프루닝(pruning)과 저비트 양자화(low-bit quantization) 과정을 동시에 수행하는 프레임워크인 SQS(Sparse Quantized Sub-distribution)를 제안합니다. 기존의 방법은 각각 별도로 적용되어 최적의 압축률(compression rate)을 달성하는 데 한계가 있었으나, SQS는 이러한 한계를 극복하고 비교 가능한 성능을 유지하며 더 높은 압축률을 달성합니다. 이를 통해 자원 제약이 있는 장치에서의 모델 배포(deployment)가 가능해집니다.

- **Technical Details**: SQS 방법은 spike-and-slab prior를 사용하여 희소성(sparsity)을 유도하고, Gaussian Mixture Model(GMM)을 통해 양자화된 가중치(distribution)를 모델링합니다. 저비트 양자화(low-bit quantization)는 고정밀 표현 형식(FP32)을 FP8 또는 BF8과 같은 저정밀 형식으로 변환하여 메모리 축소와 계산 비용 절감을 실현합니다. 논문에서는 SQS 방법이 희소하고 양자화된 신경망(sparse and quantized neural network)을 높은 확률로 수렴시킨다고 이론적으로 증명합니다.

- **Performance Highlights**: SQS는 ResNet, BERT-base, LLaMA3, Qwen2.5와 같은 여러 신경망 모델을 압축한 결과, 같은 비트 폭(bit-width) 설정 하에 가장 높은 압축률을 달성하며, 상대적으로 작은 정확도(drop) 손실을 보여줍니다. 특히 2비트와 4비트 정밀도에서 강력한 성능을 나타내는 것을 확인했습니다. 또한, spike-and-slab 배포가 Gaussian 대안보다 더 효과적이며, Bayesian 평균 추정이 탐색적 디코딩(greedy decoding)보다 나은 성능을 발휘함을 입증하였습니다.



### Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation (https://arxiv.org/abs/2510.08996)
- **What's New**: AI 기반 소프트웨어 엔지니어링 에이전트의 새로운 평가 방법론이 소개되었습니다. 기존의 GitHub 이슈를 기반으로 한 벤치마크들은 실제 개발자와의 상호작용을 제대로 반영하지 못하기 때문에 성능을 과대 평가하는 경향이 있습니다. 새로운 벤치마크 변환 방법론은 이러한 문제를 해결하며, 더 현실적인 사용자 쿼리로 기존 벤치마크를 변화시킵니다.

- **Technical Details**: 본 연구에서는 IDE 기반 에이전트와의 상호작용에서 개발자의 행동 패턴을 분석하여 합리적인 사용자 쿼리를 생성하는 시스템을 제안합니다. 이 방법론은 SWE-Bench Verified 및 SWE-Bench C#과 같은 다양한 벤치마크에 쉽게 적용될 수 있습니다. 결과적으로, 기존 벤치마크는 공공 데이터셋에 대해 20-50% 의 성능 과대 평가를 나타냅니다.

- **Performance Highlights**: 연구 결과, 기존 벤치마크는 에이전트의 능력을 20-50% 과대 평가하며, 내부 벤치마크는 성능 격차가 10-16%로 줄어드는 것으로 나타났습니다. 이러한 결과는 벤치마크의 자연스러운 사용 패턴을 반영하는 평가 방법론의 필요성을 강조합니다. 연구 결과는 GitHub의 SWE-Bench 리포지토리에 구현되어 있어 커뮤니티가 쉽게 접근할 수 있습니다.



### PlatformX: An End-to-End Transferable Platform for Energy-Efficient Neural Architecture Search (https://arxiv.org/abs/2510.08993)
- **What's New**: PlatformX는 에너지 효율적인 딥 뉴럴 네트워크(DNN)를 설계하기 위한 전자동(HW-NAS) 프레임워크로, 기존의 HW-NAS 방식들이 가지는 한계를 극복하기 위해 개발되었습니다. 이 프레임워크는 에너지 중심의 검색 공간, 이식 가능한 에너지 예측기, 다목적 최적화 알고리즘, 그리고 자동화된 런타임 에너지 프로파일링 시스템을 통합하여 다양한 엣지 플랫폼에 걸쳐 고효율 아키텍처를 탐색할 수 있도록 합니다. 특히, 이는 모델 선택 과정에서 인적 개입 없이 실시간 에너지 피드백을 제공하여 검색 시간을 크게 단축시킵니다.

- **Technical Details**: PlatformX는 기존의 고비용 NAS 접근 방법과는 달리, 하드웨어에 따라 다르게 최적화된 에너지 예측기를 도입함으로써 에너지 관련 아키텍처 구성을 포함하는 확장된 검색 공간을 구축합니다. 이 시스템은 파레토 기반의 다중 목표 최적화 알고리즘을 사용하여 에너지와 정확성을 동시에 최적화하며, 실제 장치 피드백을 통해 아키텍처를 지속적으로 조정하는 반복 검색 과정을 구성합니다. 또한, 외부 모니터를 이용한 자동화된 고해상도 에너지 프로파일링 시스템을 통해 인퍼런스 시간 전력 측정을 자동으로 수행합니다.

- **Performance Highlights**: PlatformX는 여러 모바일 및 임베디드 플랫폼에서 평가되었으며, 기존 HW-NAS 시스템에 비해 약 400배 빠른 검색 속도를 제공하며, 찾은 모델은 최대 0.94의 정확도 및 인퍼런스당 0.16mJ의 에너지 소비로 MobileNet-V2보다 뛰어난 성능을 보여줍니다. 이러한 결과는 하드웨어 정확도에 기반한 에너지 추정 이점을 제공하며, 플랫폼 간의 확장성과 실제 배포 가능성을 입증합니다. 이 프레임워크는 높은 에너지 예측 정확도와 최소한의 구성을 통해 효율적인 모델 검색을 가능하게 합니다.



### SEER: Sustainability Enhanced Engineering of Software Requirements (https://arxiv.org/abs/2510.08981)
Comments:
          Main Paper: 32 pages, References: 3 pages, Appendix: 13 pages. Submitted to the Journal of Systems and Software, Elsevier

- **What's New**: 이 논문은 조기 소프트웨어 개발 단계에서의 지속 가능성 문제를 다루는 SEER라는 프레임워크를 소개합니다. SEER는 세 가지 단계로 구성되어 있으며, 특정 소프트웨어 제품과 관련된 지속 가능성 요구사항(SRs)을 식별하고, 이들을 바탕으로 시스템 요구사항의 지속 가능성을 평가합니다. 마지막으로, SR을 만족하지 못하는 시스템 요구사항을 최적화합니다. 이 프레임워크는 대형 언어 모델의 추론 능력과 RAG(Retrieval Augmented Generation) 방식을 사용하여 구현되었습니다.

- **Technical Details**: SEER 프레임워크는 첫째로, 일반적인 분류법에서 특정 소프트웨어 제품에 관련된 SR을 식별합니다. 둘째, 식별된 SR을 기반으로 얼마나 지속 가능한 시스템 요구사항인지 평가합니다. 셋째, 요구사항이 SR을 만족하지 못할 경우 이를 최적화하는 과정을 포함합니다. 이 프레임워크는 다양한 도메인에서 수행된 네 개의 소프트웨어 프로젝트를 통해 실험되었으며, Gemini 2.5 추론 모델을 사용하여 결과의 정확성을 입증했습니다.

- **Performance Highlights**: SEER 프레임워크는 지속 가능성과 관련된 다양한 문제를 정확하게 식별하는 데 효과적임을 보였습니다. 이는 소프트웨어 산업에서 지속 가능성에 대한 인식 및 요구사항 정의의 중요성을 강조합니다. 이를 통해 개발자가 초기 요구사항 단계에서부터 지속 가능성을 평가하고 반영할 수 있도록 지원하여, 소프트웨어 개발 생애 주기 전반에 걸쳐 지속 가능성 문제를 고려하게 만듭니다.



### Learning Regularizers: Learning Optimizers that can Regulariz (https://arxiv.org/abs/2510.08968)
- **What's New**: 이번 논문에서는 Learned Optimizers (LOs)가 전통적인 regularization 기법 없이도 이들의 효과를 학습하고 내재화할 수 있음을 실증적으로 보여줍니다. 이는 기존의 명시적인 regularization 기법에 대한 의존을 줄이고, 불필요한 수동 조정 없이도 안정성과 일반화 능력을 동시에 달성할 수 있다는 가능성을 제시합니다.

- **Technical Details**: LO는 두 단계로 구성된 계층적 학습 과정을 통해 최적화됩니다. 내부 학습(inner training)과 외부 학습(outer training) 과정을 통해 LOs는 optimizee의 손실 함수를 최소화하도록 학습하고, 이를 기반으로 파라미터 업데이트를 생성하여 최적화의 효율성을 향상시킵니다.

- **Performance Highlights**: 대규모 실험을 통해 regularized LOs가 비정규화된 LOs보다 테스트 정확도와 일반화 측면에서 일관되게 우수한 성능을 보이는 것을 확인했습니다. LOs는 기존 regularization 기법의 효과를 새로운 최적화 작업으로 전이할 수 있으며, 이는 포괄적인 학습 방식의 가능성을 보여주고 있습니다.



### Analytical Survey of Learning with Low-Resource Data: From Analysis to Investigation (https://arxiv.org/abs/2510.08962)
Comments:
          Accepted by ACM Computing Surveys

- **What's New**: 본 논문은 AI 연구의 주요 목표인 제한된 자원 데이터에서의 강력한 일반화를 달성하기 위한 최신 전략을 제시합니다. 데이터 주석 및 모델 훈련과 관련된 비용을 줄이기 위해 다양한 최적화 전략과 학습 패러다임을 분석하였습니다. 저자는 모델 비의존적인 환경에서 저자원 데이터 학습에 대한 이론적 분석을 제공하며, 액티브 샘플링 이론을 통해 개선된 성과를 이끌어낼 수 있음을 보여줍니다.

- **Technical Details**: 이 조사에서는 Probably Approximately Correct (PAC) 프레임워크 내에서 저자원 데이터의 일반화 오류 및 레이블 복잡성을 분석합니다. 저자원 데이터 학습을 위한 최적화 전략으로는 gradient-informed optimization, meta-iteration optimization, geometry-aware optimization, LLMs-powered optimization 등이 포함됩니다. 이를 통해 제한된 데이터로도 높은 성능을 출현시킬 수 있는 방법론을 제공합니다.

- **Performance Highlights**: 결과적으로 본 연구는 저자원 데이터에서의 학습이 고자원 데이터로부터 학습한 것과 유사한 성과를 이끌어낼 수 있음을 입증하였습니다. 모델 비 의존적 맥락에서의 일반화 오류 및 레이블 복잡성에 대한 이론적 보장을 최초로 제공합니다. 저자원 데이터의 대표성을 향상시키기 위해 여러 학습 패러다임이 탐구되며, 제한된 자원에서 더 견고한 성능을 달성하는 방법론이 제시됩니다.



### A Human Behavioral Baseline for Collective Governance in Software Projects (https://arxiv.org/abs/2510.08956)
Comments:
          Algorithmic Collective Action Workshop @ NeurIPS 2025. arXiv admin note: text overlap with arXiv:2509.16295

- **What's New**: 이번 연구는 오픈 소스 커뮤니티의 참여 및 통제 방식을 분석하여, 버전 관리된 거버넌스 문서를 통해 드러나는 변화를 살펴봅니다. 710개의 프로젝트를 대상으로 하여 행위자(actors), 규칙(rules), 행동(actions), 객체(objects)을 파악하고, 이를 통해 참여의 카테고리를 확대하고 균형을 이루는 과정을 관찰하였습니다. 이 연구는 AI가 통합된 미래의 작업 흐름이 권한을 집중시키는지 또는 재분배하는지를 평가하기 위한 재현 가능한 기초선을 제공합니다.

- **Technical Details**: 연구에서 사용된 데이터는 GitHub의 오픈 소스 프로젝트로, GOVERNANCE.md 같은 거버넌스 문서를 통해 규칙을 명확히 하고 과거의 변화를 비교할 수 있었습니다. 2013년부터 2022년까지의 710개 저장소를 분석했으며, 이 중 637개의 프로젝트에서 초기 및 최신 두 개의 거버넌스 스냅샷을 기반으로 한 변화가 기록되었습니다. 이 과정은 누구나 재사용 가능하도록 단순화된 파이프라인을 통해 수행되었습니다.

- **Performance Highlights**: 연구 결과, 시간이 지남에 따라 프로젝트는 더 많은 역할과 행동을 정의하고 있으며, 이들이 더욱 고르게 분포되고 있다는 것을 나타냅니다. 또한, 규칙의 구성은 안정적으로 유지되고 있음을 확인했습니다. 이러한 발견은 오픈 소스 커뮤니티가 참여 형태를 확장하고 조화시키는 데 있어 중요한 통찰을 제공합니다.



### SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Managemen (https://arxiv.org/abs/2510.08948)
- **What's New**: 이번 논문에서는 SHERLOCK 프레임워크를 제안하여 큰 언어 모델(LLM)의 추론 능력을 활용하여 리스크 조사에서의 분석가 지원을 목적으로 합니다. 이 프레임워크는 세 가지 주요 구성 요소로 이루어져 있으며, 사용자 맞춤형 리스크 관리 지식 기반(KB) 생성, 데이터 플라이휠 기반의 지능형 플랫폼 구축, 그리고 변화하는 리스크 패턴에 신속 대응할 수 있는 Reflect & Refine(R&R) 모듈을 포함합니다. 이를 통해 리스크 관리자들이 효율적으로 케이스 조사 워크플로우를 수행할 수 있도록 돕고 있습니다.

- **Technical Details**: SHERLOCK의 구성 요소에는 리스크 관리 지식의 기반이 되는 도메인 KB, 일관된 데이터 라벨링 및 모델 평가를 포함한 데이터 플라이휠, 그리고 LLM의 허구적 추론을 보완하는 R&R 모듈이 포함됩니다. 도메인 KB는 다중 데이터 소스에서 리스크 관리 지식을 추출하고, 데이터 플라이휠은 운영 효율성을 높이며 샘플 수집의 효과를 극대화하도록 설계되었습니다. R&R 모듈은 분석 결과의 정확성을 향상시키고 신속하게 리스크 패턴을 수정하는 기능을 수행합니다.

- **Performance Highlights**: 실제 거래 데이터셋을 사용한 실험 결과, SHERLOCK 프레임워크는 리스크 분석의 정밀도를 크게 향상시켰습니다. 이 과정에서 리스크 모델의 운영 신뢰성을 높이고 수천 건의 케이스를 효과적으로 처리한 것으로 나타났습니다. 또한 SHERLOCK 기반의 LLM 시스템을 통해 리스크 관리자의 조사 워크플로우 효율성이 상당히 개선된 것으로 보고되었습니다.



### Co-Authoring the Self: A Human-AI Interface for Interest Reflection in Recommenders (https://arxiv.org/abs/2510.08930)
- **What's New**: 이번 연구에서는 사용자 영화 이력에 대한 수정 가능한 개인화된 관심 요약을 제공하는 영화 추천 시스템을 위한 인간-인공지능 협력 프로필을 소개합니다. 기존의 정적인 프로필과는 달리, 이 디자인은 사용자가 시스템의 추론을 직접 조사하고 수정할 수 있도록 유도합니다. 실제 사용자들을 대상으로 한 8주간의 온라인 필드 배포를 통해 사용자가 인식하는 관심과 시스템이 추론한 관심 간의 갭을 지속적으로 발견하였으며, 이러한 프로필이 사용자 참여를 촉진하고 반성을 유도하는 방법을 보여주었습니다.

- **Technical Details**: 사용자 프로필은 개인화된 추천 시스템에서 중요한 요소이나, 많은 기존 구현물은 상호작용성과 투명성, 세분성이 부족합니다. 이러한 점을 극복하기 위해, 우리는 대규모 언어 모델(LLM)을 활용하여 개인 영화 관심사의 동적 텍스트 요약을 생성하는 인터페이스를 개발했습니다. 이 인터페이스는 사용자 편집을 지원하며, 새로운 평점이 추가될 때마다 요약을 재생성하여 지속적인 관심 표현과 사용자 반성을 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 새로운 인간-인공지능 협력 프로필 인터페이스는 사용자들이 자신의 관심을 더 잘 이해하고 인식하도록 돕는데 기여했습니다. 특히, 이 인터페이스는 사용자들의 장기적인 상호작용 패턴에 긍정적인 영향을 미쳤으며, 사용자들이 추천 시스템에서 더욱 적극적으로 자신의 선호를 반영하도록 유도했습니다. 이러한 결과는 추천 시스템의 투명성과 신뢰성을 높일 수 있는 설계 방향을 제시합니다.



### A Frequency-Domain Analysis of the Multi-Armed Bandit Problem: A New Perspective on the Exploration-Exploitation Trade-off (https://arxiv.org/abs/2510.08908)
Comments:
          6 pages

- **What's New**: 이 논문은 확률론적 다중 팔 밴딧 (MAB) 문제를 탐색과 착취 간의 트레이드오프 관점에서 다루며, 새로운 주파수 영역 분석 프레임워크를 제안합니다. 특히, 각 팔의 보상 추정치를 주파수 성분으로 보고, 알고리즘을 적응형 필터로 해석하여 동적인 학습 과정을 설명합니다. 이 접근은 고전적인 알고리즘에 대한 새로운 해석을 제공하고, 향후 알고리즘 설계에 이론적 기초를 제공합니다.

- **Technical Details**: 이 논문에서는 각 팔을 주파수 성분으로 보고, 팔의 추정 불확실성을 통해 주파수를 정의합니다. UCB 알고리즘의 탐색 메커니즘이 주파수 영역에서 시간 가변 이득으로 해석됨을 보여주며, 이는 샘플 수에 반비례합니다. 또한, 유한 시간 동적 경계를 유도하여 기존의 누적 후회 경계보다 학습 과정의 단계적 특성을 잘 반영합니다.

- **Performance Highlights**: 이 모델을 통해 MAB 문제의 동적 특권과 더불어 알고리즘의 탐색과 착취 메커니즘을 명확히 이해할 수 있습니다. 새롭게 제안된 주파수 영역 모델은 알고리즘 설계에 자동 탐색 매개변수 조정에 대한 이론적 지침을 제공하며, 향후 더 발전된 알고리즘을 위한 기초로 활용될 수 있습니다.



### A Unified Biomedical Named Entity Recognition Framework with Large Language Models (https://arxiv.org/abs/2510.08902)
Comments:
          Accepted as a short paper at BIBM2025

- **What's New**: 이 논문에서는 생물 의학 명명체 인식을 위해 통합된 BioNER 프레임워크를 제안합니다. 핵심적으로, 생물 의학 텍스트의 두 가지 형태인 플랫(flat) 및 중첩(nested) 엔티티를 처리하기 위한 기호 태깅 전략을 설계하였습니다. 중국어와 영어의 다국어 데이터셋에서 공동 미세 조정(bilingual joint fine-tuning)을 통해 다국어 및 다중 과제에 대한 일반화를 향상시켰습니다.

- **Technical Details**: BioNER 작업을 텍스트 생성 작업으로 재구성하면서, 기존의 구조화를 통해 복잡한 생물 의학 개념을 효과적으로 처리하도록 설계된 다양한 엔티티 태깅 전략을 도입했습니다. 조합된 데이터셋에 대한 다중 데이터셋 공동 미세 조정 기법을 사용하여 다국어 및 다중 과업에서 모델의 일반화 능력을 향상시키고, 경계 감지에 민감한 긍정적 및 부정적 샘플을 활용한 대비 학습 기반 엔티티 선택기를 소개하였습니다.

- **Performance Highlights**: 네 가지 벤치마크 데이터셋과 두 개의 미리 본 적 없는 코퍼스를 통해 수행한 실험 결과, 제안한 방법이 최첨단 성능을 달성하면서 자유로운 제로샷 일반화 능력을 입증하였습니다. 플랫 및 중첩 엔티티의 정확한 식별에 성공하며, 모델은 다양한 오픈 소스 LLM에 대해 포괄적인 평가를 받았습니다.



### Pinpointing crucial steps: Attribution-based Credit Assignment for Verifiable Reinforcement Learning (https://arxiv.org/abs/2510.08899)
Comments:
          12 pages, 5 figures

- **What's New**: 최근 강화학습 분야에서는 Verifiable Rewards (RLVR)의 발전으로 LLMs(대형 언어 모델)가 복잡한 추론 작업에서 성능 향상을 이루고 있으나, 탐색(exploration)과 활용(exploitation) 간의 균형을 맞추는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 본 논문에서는 복잡한 추론 문제를 위한 새로운 알고리즘 프레임워크인 ACPO(Attribution-based Contribution to Policy Optimization)를 제안합니다. ACPO는 정책 엔트로피의 동적 조절을 통해 탐색을 지원하고, 각 추론 단계의 기여도를 정량화하여 활용 성능을 향상시키는 구조를 가지고 있습니다.

- **Technical Details**: ACPO는 두 단계에 걸쳐 진행되는 알고리즘 프레임워크로, 목표는 RLVR에서의 탐색과 활용을 동시에 개선하는 것입니다. 이 시스템은 각 추론 단계의 기여도를 정확하게 측정하기 위해 요소화된 보상 시스템을 도입했습니다. 또한, 난이도 인식 커리큘럼(difficulty-aware curriculum)을 통해 정책 엔트로피를 조절하므로서, 탐색 실패를 줄이고 다양한 추론 경로를 발견하도록 유도합니다.

- **Performance Highlights**: ACPO는 AIME, MATH, AMC와 같은 도전적인 벤치마크에서 기존의 최첨단 접근법보다 높은 성능을 발휘해 그 효과성을 입증하였습니다. 특히, 단계별 보상을 채택함으로써 중간 단계에 대한 정확한 신용 할당이 가능하게 되었으며, 이러한 접근법을 통해 수렴 속도와 성능이 동시에 향상되었습니다. 이 연구는 복잡한 추론 작업에서 더욱 유망한 미래 전략을 제공하는 기초 자료가 될 것입니다.



### HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidanc (https://arxiv.org/abs/2510.08896)
- **What's New**: HES-SQL은 Text-to-SQL 생성의 새로운 하이브리드 훈련 프레임워크로, 생각 모드 융합 감 supervised fine-tuning (SFT)와 Group Relative Policy Optimization (GRPO)의 통합을 통해 발전합니다. 이 접근법은 세 가지 주요 혁신을 도입하는데, 첫째, 생성된 쿼리와 최적의 SQL 구조 간의 선호 정렬을 향상시키는 스켈레톤 완전성 평가 메커니즘; 둘째, 계산 효율성이 높은 SQL 쿼리 생성을 유도하는 쿼리 대기 시간 인식 보상 시스템; 셋째, 모델의 추론 능력 저하를 방지하는 자기 증류(self-distillation) 프로세스가 그것입니다.

- **Technical Details**: HES-SQL은 자체 증류 초기화와 GRPO 기반의 강화 학습 세부 조정을 결합한 새로운 하이브리드 훈련 프레임워크입니다. 해당 프레임워크는 구조적 정렬과 실행 성능을 목표로 새로운 보상 신호 및 2단계 훈련 전략을 도입하며, 생성 중에 '빠른' 및 '느린' 사고 모드를 전환할 수 있도록 합니다. 또한, 스켈레톤 완전성 보상과 대기 시간 인식 최적화를 통해 모델이 효율적인 SQL 수식을 생성하도록 유도합니다.

- **Performance Highlights**: HES-SQL은 BIRD, Spider, KaggleDBQA 벤치마크에서 각각 79.14%, 84.04%, 54.9%의 실행 정확도를 달성하며, 강력한 SFT 기준선을 지속적으로 초과하는 성능을 보입니다. 이와 동시에 11%에서 20%의 효율성 향상을 달성하며 평균적으로 더 빠른 쿼리 실행을 가능합니다. 이러한 이중 최적화는 HES-SQL이 실제 데이터베이스 애플리케이션에서 유망한 접근법으로 자리잡도록 합니다.



### Exploring Multi-Temperature Strategies for Token- and Rollout-Level Control in RLVR (https://arxiv.org/abs/2510.08892)
- **What's New**: 이 연구는 Reinforcement Learning (RL)을 통해 대형 언어 모델(LLM)의 추론 능력을 향상시키는데 중요한 관리 방안으로, 서로 다른 타입의 토큰에 대해 차별화된 온도(temperature) 설정을 적용하여 탐색(exploration)을 유도하는 혁신적인 방법을 제안합니다. 특히, 고엔트로피 추론 토큰에 대해 높은 온도를 적용하고, 저엔트로피 지식 토큰에는 낮은 온도를 적용하여 사실성을 유지함과 동시에 창의적 탐색을 증가시키는 접근 방식을 채택하였습니다. 실험 결과, 이 방법이 여러 추론 벤치마크에서 LLM의 성능을 향상시켰음을 보여줍니다.

- **Technical Details**: 이 연구는 서로 다른 온도 스케줄링 전략을 체계적으로 분석하고, RLVR(Reinforcement Learning with Verifiable Rewards) 환경 내에서의 영향을 조사합니다. 온도 스케일링은 생성 단계에서의 탐색과 착취 간의 균형에 중요한 역할을 하는데, 저온은 기존 지식의 착취를 촉진하며 안정성을 높이고, 고온은 다양한 샘플링 및 탐색을 장려하여 훈련 과정에서 다양한 추론 경로를 노출시킬 수 있습니다. 이러한 온도 조정 전략은 토큰 유형의 특성에 맞추어 고안되어 지식 및 추론의 혼합을 최적화합니다.

- **Performance Highlights**: 실험적 평가 결과, 제안된 토큰 레벨 샘플링 방식이 Qwen2.5-1.5B-Math 모델의 추론 성능을 향상시키는 데 효과적임을 입증하였습니다. AIME24에서는 +6%, AIME25에서는 +1%, Minerva에서는 +4.8%의 성과 향상을 기록하였습니다. 이러한 성과는 추가적인 계산 비용없이 획득된 결과로써, 제안된 방법의 효율성과 효과성을 강조합니다.



### Designing and Evaluating an AI-driven Immersive Multidisciplinary Simulation (AIMS) for Interprofessional Education (https://arxiv.org/abs/2510.08891)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 AIMS(인공지능 강화 몰입형 다학제 시뮬레이션)를 개발하여 의료 분야 프로페셔널 간의 팀워크 및 협업 능력을 향상시키고자 하였습니다. 기존의 교육 방식인 사례 연구와 표준화된 환자를 활용한 시뮬레이션의 한계를 극복하고, 비용 효율적이고 확장 가능한 가상 환경을 제공합니다. AIMS는 Gemini-2.5-Flash라는 대형 언어 모델과 Unity 기반의 가상 환경 엔진을 통합하여 사용자와 가상 환자가 동기화된 다중 모드 상호작용을 할 수 있도록 설계되었습니다.

- **Technical Details**: AIMS는 두 가지 임상 환경, 즉 응급실과 1차 진료 사무실을 구축하여 학생들이 가상 환자와 실시간으로 상호작용할 수 있는 기회를 제공합니다. 세 가지 주요 엔진, 즉 캐릭터 생성 엔진, 다중 모드 AI 엔진, 그리고 가상 환경 엔진이 서로 연결되어 몰입감 있는 학습 경험을 생성합니다. 초기 아바타 구성은 임상 전문가와 협력하여 적절한 외관과 동작을 확보하는 과정에서 진행되었습니다.

- **Performance Highlights**: 사용성 테스트를 통해 AIMS는 현실적이고 전문적으로 적합한 대화를 지원한다는 결과를 도출했습니다. 참가자들은 가상의 환자 증상과 사회적 맥락을 탐색하며, 각 팀이 환자의 진료 여정을 따라가도록 담당 교수의 안내를 받았습니다. 이러한 피드백은 추후 AIMS 개선에 중요한 역할을 하며, 사용자 경험의 질을 향상시키기 위한 기초 자료로 활용되었습니다.



### ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling (https://arxiv.org/abs/2510.08878)
Comments:
          18 pages, 8 tables, 5 figures

- **What's New**: 이 연구에서는 텍스트-오디오( TTA) 생성의 새로운 접근 방법인 ControlAudio를 제안합니다. 이는 컨트롤 가능한 TTA 생성 문제를 다중 작업 학습 문제로 재구성하고 점진적인 확산 모델링(progressive diffusion modeling) 접근 방식을 도입하여, 텍스트, 타이밍, 음소 정보 등 보다 세밀한 정보에 대해 조정된 분포를 적절하게 맞출 수 있도록 합니다.

- **Technical Details**: ControlAudio는 대규모 ⟨텍스트, 오디오⟩ 데이터 쌍 구축에서 출발하여, ⟨텍스트, 타이밍, 오디오⟩ 및 ⟨텍스트, 타이밍, 음소, 오디오⟩ 데이터셋을 구성합니다. 이를 통해 텍스트, 타이밍, 음소 정보를 하나의 텍스트 인코더로 통합하여 세분화된 제어 신호를 점진적으로 통합합니다. 모델 훈련 단계에서는 대규모 텍스트-오디오 쌍에 대해 확산 변환기(Diffusion Transformer)로 사전 훈련을 진행하고, 이후 타이밍 및 음소 특성을 추가하여 제어 가능성을 확장합니다.

- **Performance Highlights**: ControlAudio의 광범위한 실험 결과, 기존 TTA 생성 방법들보다 뛰어난 성능을 보여줍니다. 특히 시간 정확성(temporal accuracy)과 음성 명료성(speech clarity) 측면에서 최신 기술(state-of-the-art) 성능을 달성하였으며, 객관적 및 주관적 평가 모두에서 우수한 결과를 보였습니다. 이 연구는 다양한 오디오 이벤트를 동시에 생성하는 데 있어 중요한 진전을 이룬 것으로 평가됩니다.



### Vector Graph-Based Repository Understanding for Issue-Driven File Retrieva (https://arxiv.org/abs/2510.08876)
- **What's New**: 이번 논문에서는 대규모 소프트웨어 리포지토리를 벡터화된 knowledge graph로 변환해주는 리포지토리 분해 시스템을 소개합니다. 이는 프로젝트의 아키텍처(architecture) 및 의미적 구조(semantic structure)를 반영하며, 의미적 관계를 포착하여 추가적인 리포지토리 개발의 자동화를 가능하게 합니다. 이 시스템은 신규 기술과 뚜렷한 자동화 수준을 제공하여 개발자들에게 유용할 것으로 기대됩니다.

- **Technical Details**: 제안된 그래프는 포함(containment), 구현(implementation), 참조(references), 호출(calls) 및 상속(inheritance)과 같은 구문적 관계(syntactic relations)를 인코딩합니다. 노드는 LLM(대규모 언어 모델) 기반의 요약(summary) 및 벡터 임베딩(vector embeddings)으로 보강됩니다. 그리고 이 시스템은 의미 기반 검색(semantic retrieval)과 그래프 인식 확장(graph-aware expansion)을 결합한 하이브리드 검색 파이프라인을 활용합니다.

- **Performance Highlights**: 최종적으로 LLM 기반 어시스턴트가 제약이 있는 읽기 전용(graph requests) 그래프 요청을 형성하고 인간 중심의 설명을 만들어 제공합니다. 이러한 접근 방식은 소프트웨어 리포지토리 개발 및 탐색에서 효율성을 향상시킬 것으로 보입니다. 전체적으로 이 연구는 소프트웨어 개발에서의 지식 그래프 사용에 새로운 장을 열 수 있는 가능성을 보여줍니다.



### Slicing Is All You Need: Towards A Universal One-Sided Algorithm for Distributed Matrix Multiplication (https://arxiv.org/abs/2510.08874)
- **What's New**: 이 논문에서는 분산 행렬 곱셈(Distributed Matrix Multiplication)을 위한 보편적인 단일 측면 알고리즘(universal one-sided algorithm)을 제시합니다. 이 알고리즘은 모든 조합의 파티셔닝(partitionings) 및 복제 계수(replication factors)를 지원하여, 기존 알고리즘의 한계를 극복합니다. 특히, 상이한 문제 크기와 파티셔닝에 맞추어 여러 알고리즘을 요구하는 기존 방법과는 다르게 모든 경우에 적용 가능성을 높입니다.

- **Technical Details**: 본 알고리즘은 슬라이싱(slicing) 기법을 사용하여 곱셈이 필요한 겹치는 타일(tiles) 집합을 계산합니다. 준비된 로컬 행렬 곱셈(local matrix multiplies) 리스트는 직접 실행되거나, 재정렬하여 최적화된 IR(Intermediate Representation)로 하향(low)될 수 있습니다. 구현은 고급 C++ 기반 PGAS 프로그래밍 프레임워크를 사용하며, GPU 간 직접 통신을 위한 intra-node interconnects를 활용합니다.

- **Performance Highlights**: 다양한 파티셔닝과 복제 계수에 대해 성능을 평가한 결과, 본 알고리즘은 AI 모델을 겨냥한 고도로 최적화된 분산 텐서 라이브러리인 PyTorch DTensor와 경쟁할 수 있는 성능을 보입니다. 이로 인해, 데이터 분석 및 AI 워크로드에 중요한 분산 행렬 곱셈에서의 효율성을 크게 향상시킵니다.



### Pattern Enhanced Multi-Turn Jailbreaking: Exploiting Structural Vulnerabilities in Large Language Models (https://arxiv.org/abs/2510.08859)
- **What's New**: 이번 논문은 대화형 패턴을 적용한 새로운 다단계 탈출 공격 프레임워크인 PE-CoA(Pattern Enhanced Chain of Attack)를 제안합니다. 제공된 다섯 가지 대화 패턴을 통해 LLM(대형 언어 모델)의 보안 취약점을 체계적으로 분석하며, 각기 다른 악성 콘텐츠 범주에 대해 LLM의 반응을 평가함으로써 새로운 공격 전략을 구축합니다. 이를 통해, 모델이 대화 유형에 따라 어떻게 다르게 반응하는지를 명확히 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: PE-CoA는 대화형 패턴을 사용하는 멀티턴 공격을 위해 설계된 프레임워크로, 대화 패턴과 LLM의 취약성 간의 관계를 체계적으로 분석합니다. 논문에서는 특정 LLM 모델의 아키텍처가 대화 패턴에 따라 어떻게 다르게 반응하는지를 분석하고, 서로 다른 해악 분류와의 상호작용을 다룹니다. 이 과정에서 문제 해결 패턴과 정보 패턴과 같은 특정 대화 패턴의 아키텍처적 취약점을 드러내는 중요한 결과를 도출했습니다.

- **Performance Highlights**: PE-CoA는 12개의 다양한 LLM을 대상으로 한 실험을 통해 최첨단 성능을 달성하였으며, 특정 패턴에 대한 취약성을 밝혀냈습니다. 연구 결과, 모델의 강건성이 특정 대화 패턴에 제한적이며, 패턴마다 고유한 실패 양상을 보임을 발견했습니다. 이러한 성과는 안전 교육의 한계를 강조하고, 패턴 인식 방어의 필요성을 시사합니다.



### Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training (https://arxiv.org/abs/2510.08855)
Comments:
          First submitted on February 10th, 2025 to ICLR 2025 Workshop (XAI4Science: From Understanding Model Behavior to Discovering New Scientific Knowledge). The paper was accepted but the workshop does not generate proceedings. Now uploading to arXiv to make the paper publicly available

- **What's New**: 이번 연구에서는 대형 언어 모델의 내부 표현을 이해하기 위한 새로운 접근법인 Adaptive Temporal Masking (ATM)을 제시합니다. 기존의 Sparse Autoencoders (SAEs) 훈련 방식은 특징의 흡수 현상으로 인해 모델의 행동 분석이 어렵다는 문제점이 있었습니다. ATM은 시간에 따라 변하는 활성화의 크기, 빈도 및 재구성 기여도를 추적하여 동적으로 특징을 선택하는 기법입니다. 이 방법은 신뢰할 수 있는 모델 분석을 위한 안정적이고 해석 가능한 특징 학습의 기초를 제공합니다.

- **Technical Details**: ATM은 세 가지 주요 통계 값을 추적하여 특징의 중요도를 평가합니다: 특징 활성화, 재구성 기여도 및 그에 대한 지수 이동 평균(Exponential Moving Average, EMA). 이 방법은 통계적인 값들을 기반으로 적응형 임계값(threshold)을 계산하고, 고정된 값 대신 발전하는 특징 중요도 분포에 맞게 임계값을 동적으로 조정하여 더 자연스러운 특징 선택 프로세스를 구현합니다. 이를 통해 특징 활성화 패턴에 대한 적절한 추적과 유연한 조절이 가능합니다.

- **Performance Highlights**: 대규모 실험을 통해 ATM은 기존 방법들(예: TopK 및 JumpReLU SAEs)보다 훨씬 낮은 흡수 점수(absorption scores)를 달성하며, 우수한 재구성 품질을 유지합니다. 이러한 결과들은 ATM이 신경망에서 안정적이고 해석 가능한 특징을 학습하는 원칙적인 해결책임을 확립합니다. 따라서, LLM 내부를 이해하고 잠재적 편향(bias)을 분석하는 데 필수적인 기능을 제공합니다.



### Repository-Aware File Path Retrieval via Fine-Tuned LLMs (https://arxiv.org/abs/2510.08850)
- **What's New**: 이 논문에서는 현대 코드베이스에서 개발자와 AI 코딩 어시스턴트가 소스 파일을 효과적으로 찾는 방법을 제시합니다. 전통적인 코드 검색 방법은 의미론적 문맥과 파일 간 연결성을 놓치는 경우가 많습니다. 연구자들은 강력한 언어 모델(Qwen3-8B)을 세밀하게 조정하여 자연어 쿼리로부터 관련 파일 경로를 직접 예측하도록 하는 새로운 방법을 소개합니다.

- **Technical Details**: 이 방법은 QLoRA 및 Unsloth 최적화를 통해 fine-tuning을 진행하며, 훈련 데이터를 생성하기 위해 여섯 가지 코드 인식 전략을 도입합니다. 이 전략들은 추상 구문 트리(abstract syntax tree, AST) 구조와 저장소(content)를 활용하여 현실적인 질문-답변 쌍을 생성하는 데 중점을 둡니다. 훈련은 Flask, Click, Jinja, FastAPI, PyTorch와 같은 Python 프로젝트에서 수행되며, 다양한 케이스를 포괄합니다.

- **Performance Highlights**: 모델은 보유된 쿼리에 대해 91%의 정확도와 93%의 재현율을 달성하여 단일 전략 훈련을 명백히 초월합니다. PyTorch와 같은 대규모 코드베이스(약 4,000개의 Python 파일)에서 59%의 재현율에 도달하여 확장성을 보여줍니다. 여러 수준의 코드 신호가 LLM이 파일 간 문맥을 이해하는 데 어떻게 도움이 되는지를 분석하고, 데이터셋 설계 및 한계에 대해서도 논의합니다.



### CommandSans: Securing AI Agents with Surgical Precision Prompt Sanitization (https://arxiv.org/abs/2510.08829)
- **What's New**: 논문은 큰 언어 모델(LLM)과 도구에 접근할 수 있는 에이전트의 보안 취약점에 대한 새로운 방어 기법을 제안합니다. 기존의 수단들은 높은 허위 긍정률로 인해 실제 세계에서의 채택을 제한하는 문제를 가지고 있습니다. 본 연구에서는 명령(Instruction)을 감지하고 이를 필터링하는 새로운 토큰 수준의 정화 과정(Token-level sanitization process)을 소개하며, 이를 통해 악의적인 지시를 제거하고 정상적인 작업을 방해하지 않도록 합니다.

- **Technical Details**: 연구에서 제안하는 CommandSans 시스템은 LLM의 에이전트와 도구 출력에서 AI에 대한 지시를 제거하는 비차단 방식(non-blocking) 안전 시스템입니다. 이 방법은 기존의 안전 분류기들과 달리 샘플 수준의 검출이 아닌, 각 토큰을 세밀하게 정화하여 악의적인 지시를 포착합니다. CommandSans는 일반적인 사용자 지시 처리 데이터(instruction-tuning data)를 활용해 훈련할 수 있으며, 특정한 증강 데이터셋이 필요하지 않습니다.

- **Performance Highlights**: 실험 결과, CommandSans는 다양한 공격 벤치마크에서 우수한 성능을 발휘하며 공격 성공률(ASR)을 최대 19배까지 줄였습니다. 이러한 성과는 에이전트의 유용성을 거의 저해하지 않고 이루어졌습니다. 실제 환경에서 실질적인 유틸리티를 보장하며 비악의적인 요청의 처리에서 높은 효과를 보여주었습니다.



### McMining: Automated Discovery of Misconceptions in Student Cod (https://arxiv.org/abs/2510.08827)
Comments:
          16 pages, 8 figures

- **What's New**: 본 논문에서는 학생들의 코드 샘플에서 프로그래밍 오해를 발굴하는 작업인 McMining을 제안합니다. 이를 위해 학습 및 평가를 위한 확장 가능한 벤치마크 데이터 세트를 개발하였으며, 이 데이터 세트에는 이러한 오해가 나타나는 코드 샘플의 대량 집합이 포함됩니다. 두 가지 LLM 기반의 McMiner 접근 방식을 도입하고, Gemini, Claude, GPT 계열의 모델들이 학생 코드에서 오해를 발견하는 데 효과적임을 광범위한 평가를 통해 입증하고 있습니다.

- **Technical Details**: Introduction & Motivation 섹션에서는 프로그램 언어 정의에 의해 뒷받침되지 않는 프로그래밍 개념에 대한 믿음인 프로그래밍 오해를 정의합니다. 이러한 오해는 코드를 작성할 때 버그의 원인이 될 수 있으며, 학생들이 문제를 올바르게 해결하는 능력을 방해할 수 있습니다. McMining의 개발은 학생의 코드 샘플에서 시간에 따라 나타나는 프로그래밍 오해를 발굴하는 것을 목표로 하며, LLM 도구를 통해 이를 식별하고 설명할 수 있는 방법을 제시합니다.

- **Performance Highlights**: McMiner 도구의 성능은 벤치마크 데이터 세트를 통해 긍정적인 결과를 보였습니다. 이 사양에서는 코드 샘플의 90.3%가 목표 오해를 성공적으로 보여주는 것으로 나타났으며, 사람의 평가와 LLM의 평가 간에는 96.6%의 일치율이 발견되었습니다. 이를 통해 McMining의 방식이 교육적 관점에서 프로그래밍 오해 식별에 효과적임을 보여줍니다.



### $\mathsf{P} \neq \mathsf{NP}$: A Non-Relativizing Proof via Quantale Weakness and Geometric Complexity (https://arxiv.org/abs/2510.08814)
- **What's New**: 이번 논문에서는 짧은 프로그램을 지역성(locality)으로 변환하는 정보 이론적(compositional, information-theoretic) 프레임워크를 제안하며, 이는 많은 독립 블록과 결합되어 Masked Random Unique-SAT의 대칭성(symmetry) 및 희소성(sparsity)과 함께 사용됩니다. 이러한 조합을 통해 $	ext{P}=	ext{NP}$ 하의 자가 축소(self-reduction) 상한과 모순되는 분포적(lower bounds) 하한을 증명했습니다. 따라서, $	ext{P}
eq	ext{NP}$를 확립하게 되었습니다.

- **Technical Details**: 이 연구는 독립 블록에 걸쳐 짧은 알고리즘의 유한하고 가감(composed) 예산(budget)을 다루는 구성적 약점 계산(compositional weakness calculus)과 Masked Random 3-CNF의 대칭성 및 희소성 특성에 기반합니다. 특히, 연속성을 보장하고 특정 확률 수준에서 USAT 약속을 유지하는 두 개의 최소 레이어를 추가하여 분포의 대칭성을 확보했습니다. 알고리즘적 스위칭 조건을 통해 모든 짧은 폴리노미얼 시간 디코더가 블록의 일정 비율에 대해 지역별 비트 규칙(local per-bit rule)으로 변환됩니다.

- **Performance Highlights**: 본 연구에서는 단일 블록 경계와 작은 ACC0 형식의 디코더를 결합하여 성공 확률을 $2^{-	ilde{O}(t)}$로 도출했습니다. 논문에서는 짧은 디코더 $(P 	imes W)$가 블록의 $	heta$-비율에서 지역적으로 변환된다면, 각 블록에서의 조건부 이점이 $1/2 + 	ilde{	heta}(m)$로 제한된다고 보증합니다. 결국 $K_{	ext{poly}}((X_1,	ext{...},X_t)|(	ext{Φ}_1,...,	ext{Φ}_t)) 
eq O(1)$의 하한을 설정하여, $	ext{P}=	ext{NP}$ 가정 하에 직면할 수 있는 모순을 도출했습니다.



### Adaptive Science Operations in Deep Space Missions Using Offline Belief State Planning (https://arxiv.org/abs/2510.08812)
Comments:
          7 pages, 4 tables, 5 figures, accepted in IEEE ISPARO 2026

- **What's New**: 이번 연구에서는 통신 제약이 있는 환경에서 자율 과학 작업을 지원하기 위한 부분 관찰 마르코프 결정 프로세스(POMDP) 프레임워크를 제시합니다. 이 프레임워크는 우주선 과학 장비를 적응적으로 매핑하여 우주 탐사에 최적화된 측정 결과를 얻는 것을 목표로 합니다. 특히, 바이오 정보를 모델링하기 위해 베이지안 네트워크를 통합하여 불확실한 데이터를 관리하고 있습니다.

- **Technical Details**: 제안된 방법론에서는 POMDP를 활용하여 Enceladus Orbilander의 생물 탐지 장비를 자동으로 운영하도록 설계하였습니다. 이 프레임워크는 복잡한 관계를 효율적으로 캡처하는 베이지안 네트워크를 도입하여 성능을 향상시켰습니다. 또한, 오프라인에서 최적의 정책을 사전 계산하기 위한 SARSOP 솔버를 사용하여 우주선의 상태 및 관측 결과에 따라 다음 행동을 결정하는 정책을 관리합니다.

- **Performance Highlights**: 본 연구는 Enceladus Orbilander의 개념 운영을 기준으로 설정하여 false positive 및 false negative 비율을 비교하였습니다. 결과적으로, 제안된 접근 방식은 전통적인 방법에 비해 샘플 식별 오류를 40% 이상 감소 시켰습니다. 이는 오프라인 신뢰 상태 계획이 과학 데이터 처리에 미치는 긍정적인 영향을 잘 보여줍니다.



### Benchmarking Chinese Commonsense Reasoning with a Multi-hop Reasoning Perspectiv (https://arxiv.org/abs/2510.08800)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 일반 중국어 언어 맥락에서의 포괄적인 평가가 미흡하다는 점을 지적하고, 중국 상식 다중 단계 추론(Chinese Commonsense Multi-hop Reasoning, CCMOR)이라는 새로운 벤치마크를 제안합니다. CCMOR는 LLM이 중국 특정 사실 지식을 통합하고 다단계 논리적 추론을 수행하는 능력을 평가하기 위해 설계되었습니다. 이 연구에서는 기존 질문-답변(Question-Answer, QA) 데이터셋을 기반으로 균형 잡힌 시드 세트를 구축하고, LLM을 활용한 파이프라인을 통해 사실 식별 단위에 기반한 다중 경로 질문을 생성합니다.

- **Technical Details**: CCMOR는 중국의 문화적 지식, 관용구 및 논리 패턴에 기반하여 설계되었습니다. 데이터셋은 여러 분야를 포괄하며 사실 회상(factual recall)과 다단계 추론(multi-hop inferential reasoning)을 체계적으로 평가합니다. 데이터의 질 보장을 위해 LLM과 도메인 전문가가 함께 검증하는 시스템을 적용하여 생성된 질문의 정확성과 일관성을 높였습니다.

- **Performance Highlights**: 실험 결과, 최신 LLM들이 긴 꼬리 지식을 처리하고 지식 집약적 추론을 수행하는 데 지속적인 한계를 보여주었습니다. 특히, 절차적이거나 추상적인 추론을 요구하는 도메인에서의 성능이 저조하며, 반면 고의적인 사고를 가진 LLM들이 다중 경로 질문을 더 잘 해결할 수 있다는 점이 발견되었습니다. 흥미롭게도, 검색 강화 생성(retrieval-augmented generation)은 이러한 지식 격차를 상당히 줄이며 성능 개선을 가져왔습니다.



### Deceptive Exploration in Multi-armed Bandits (https://arxiv.org/abs/2510.08794)
- **What's New**: 이번 연구는 다중 무장 도박자(multi-armed bandit) 설정에서 공개 보상(distribution)과 개인 보상(private distribution)을 갖는 팔(arms)들이 있는 시나리오를 다룬다. 관찰자는 에이전트가 공개 보상에 따라 Thompson Sampling을 따를 것으로 기대하지만, 에이전트는 개인 보상을 이용하여 최고의 개인 팔을 신속히 찾아내고자 한다. 이 연구는 KL(쿨백-라이블러) 발산을 이용해 에이전트의 실제 풀 확률과 관찰자의 예상 풀 확률 사이의 관계를 형식화하였다.

- **Technical Details**: 연구에서는 성공적인 공개 하위 최적 팔의 풀을 Bernoulli 프로세스로 모델링하며, KL 제약 하에 최대 \(\Theta(\sqrt{T})\)의 속도로만 발생할 수 있음을 보여준다. 에이전트는 노드 풀을 통해 최적성을 유지하면서 동시에 비공식적인 탐색(deceptive exploration)을 수행하도록 한다. 이 과정에서 에이전트의 최적 개인 팔 식별 문제를 해석하는 maximin 문제를 공식화하며, 최적 에러 지수가 개인 팔 식별에 대한 것으로 나타난다.

- **Performance Highlights**: 제안된 알고리즘은 상위 두 개의 알고리즘(top-two algorithms)에서 영감을 받아 탐색을 자연스럽게 조정한다. 이 알고리즘은 팔의 공개 하위 최적성 공백에 따라 탐색의 난이도를 적응하게 되며, \(\Theta(\sqrt{T})\) 속도로 팔을 탐색할 수 있는 가능성을 보여준다. 수치 예제를 통해 이 속도와 알고리즘의 행동을 시각적으로 설명하였다.



### MLLM as a UI Judge: Benchmarking Multimodal LLMs for Predicting Human Perception of User Interfaces (https://arxiv.org/abs/2510.08783)
- **What's New**: 이 연구는 멀티모달 대형 언어 모델(MLLMs)을 활용하여 UI(사용자 인터페이스) 평가에서의 인간 선호를 근사할 수 있는 새로운 가능성을 제시합니다. 기존의 연구들이 전자상거래와 같은 제한된 영역에서 사용자의 행동에 초점을 맞춘 반면, 본 연구는 다양한 인터페이스에서 주관적 사용자 평가를 중점적으로 탐색합니다. MLLMs가 UI의 주관적 품질을 평가하는 과정에서 인간의 판단과 얼마나 유사한지를 비교하고, 그에 따른 정량적 데이터를 제공하는 기회를 제공합니다.

- **Technical Details**: 연구에서는 30개의 UI 인터페이스에 대해 GPT-4o, Claude, Llama 모델들을 사용하여 평가하였습니다. 인간 평가와 MLLM 평가 간의 상관 관계를 분석하여, MLLMs가 UI 품질의 여러 요소에 대해 인간의 판단과 유사하게 작용하는 경향이 있음을 발견했습니다. 특히, 리커트 척도 평가 방법과 쌍별 비교 방식에서 MLLMs가 인간의 평가를 근사하는 정도를 측정하여, 이들의 정확성을 다루었습니다.

- **Performance Highlights**: 결과적으로 MLLMs는 UI 평가 작업에서 인간의 판단을 대체하기보다는 보완하는 도구로 활용될 수 있다는 점을 강조하였습니다. 연구에서 사용된 모델들은 75% 이상의 정확도를 기록하며, Pearson, Spearman 및 Kendall tau 상관 통계 분석을 통해 MLLM과 인간 평가 간의 중간 강도의 일치를 보였습니다. 이러한 발견들은 자원이 제한된 팀들이 초기 UI 평가를 위해 MLLMs를 효과적으로 활용할 수 있는 가능성을 열어줍니다.



### Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations (https://arxiv.org/abs/2510.08779)
Comments:
          Accepted to LM4Plan Workshop @ ICAPS 2025 (withdrawn before presentation due to lack of travel funding)

- **What's New**: 이번 연구에서는 보상 희소(sparse-reward) 환경에서 강화 학습(RL) 에이전트가 효과적인 행동 시퀀스를 발견하는 데 어려움을 겪는 문제를 해결하기 위해 대형 언어 모델(LLM)의 절차적 지식과 추론 능력을 활용하고자 합니다. 기존의 방법들은 LLM의 제안을 강제적으로 따르게 하거나 보상 함수에 직접 통합하는 방식으로 경직된 의존성을 만들었으나, 본 연구에서는 LLM이 생성한 액션 추천을 확장된 관측 공간을 통해 제공하여 RL 에이전트가 이 지침에 따라 행동할지 무시할지를 학습하도록 합니다.

- **Technical Details**: 본 연구는 LLM의 계획 지침을 강화 학습 훈련에 통합하는 새로운 방법론을 제안합니다. 이 방법은 환경 관측에 LLM의 제안을 추가함으로써 부드러운 제약(soft constraint)을 생성하는 방식으로 작동합니다. 에이전트는 LLM이 제공하는 힌트를 정기적으로 받고, 이를 환경 관측과 함께 표준 정책 학습을 통해 통합하여, 힌트가 유용한 경우 따르고, 잘못된 경우 무시하는 법을 학습합니다.

- **Performance Highlights**: 제안된 방법은 BabyAI 환경에서 평가하였으며, 환경의 복잡성이 증가할수록 LLM의 지침이 더 큰 이점을 제공함을 보여주었습니다. 가장 도전적인 환경에서 최종 성공률이 71% 향상되었으며, 에이전트가 성능 기준에 도달하는 속도가 최대 9배 빨라지는 샘플 효율성을 요약적으로 보여주었습니다. 기존 강화 학습 알고리즘에 대한 수정 없이 일반적인 RL 알고리즘과 호환됩니다.



### Measuring Moral LLM Responses in Multilingual Capacities (https://arxiv.org/abs/2510.08776)
Comments:
          10 pages, 5 figures; referenced articles: arXiv:2303.08774, arXiv:2303.12528, arXiv:2308.14132, arXiv:2505.12201, arXiv:2406.04428, arXiv:2407.02273, arXiv:2404.01268, arXiv:2502.09747, arXiv:2507.13474, arXiv:2505.21479, arXiv:2306.05685

- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)의 다국어 응답을 평가하기 위해 대규모 데이터셋을 생성하고, LLM의 정확성과 일관성을 측정하기 위해 5가지 차원에서 최첨단 오픈 소스 모델의 성능을 비교합니다. 특히, GPT-5는 Consent & Autonomy 및 Harm Prevention & Safety와 같은 카테고리에서 평균적으로 가장 높은 점수를 기록함을 보여줍니다. 이러한 결과는 LLM의 응답에서 언어적 변화가 미치는 영향을 이해하고, 개선이 필요한 영역을 강조합니다.

- **Technical Details**: 이 연구에서는 LLM의 응답을 평가하기 위해 5개의 카테고리인 Biases & Stereotypes, Consent & Autonomy, Harm Prevention & Safety, Legality, Moral Judgment로 나누어 분석합니다. 각 카테고리의 질문은 단순한 응답부터 복잡한 질문까지 다양하며, LLM의 응답은 채점 시 편향을 최소화하기 위한 루브릭을 사용하여 평가됩니다. 또한, LLM의 안전 기능의 일관성을 측정하기 위해 특정 질문 유형이 LLM의 응답에 미치는 영향을 분석하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 초기 테스트 결과, GPT-5가 다른 모델들보다 위험하거나 유해한 질문에 더 잘 반응하는 경향을 보였습니다. 다양한 언어로 번역된 질문에 대한 응답의 일관성을 평가한 결과, 고자원 언어에서는 높은 성능을 보였으나 저자원 언어에서는 더 많은 불일치와 사실적 오류가 발생했습니다. 이러한 결과는 다국어 상황에서 LLM의 사고 및 정확성이 여전히 해결해야 할 과제임을 나타냅니다.



### Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings (https://arxiv.org/abs/2510.08774)
- **What's New**: 이 연구에서는 기존의 텍스트 임베딩(embedding) 방식을 넘어, 구조 정보를 직접적으로 LLM의 인코딩 처리 과정에 통합하는 새로운 패러다임을 제안합니다. 특히, 이 연구는 순차적 연결(sequential concatenation)과 병렬 캐싱(parallel caching)이라는 두 가지 주요 방법을 탐구하며, 이를 통해 구조 인식 임베딩의 성능을 시스템적으로 평가합니다. 이 연구는 LLM의 내부 프로세스에서 구조 정보를 효과적으로 활용하는 접근 방식을 제시하며, 보다 강력한 임베딩 모델을 위한 기본 틀을 제공합니다.

- **Technical Details**: 연구에서는 Struc-Emb-Seq와 Struc-Emb-Par 두 가지 방법을 제안합니다. Struc-Emb-Seq는 관련 문서와 대상을 순차적으로 연결하여 공동으로 인코딩하는 반면, Struc-Emb-Par은 각 관련 세그먼트를 개별적으로 인코딩하고 이를 캐싱하여 효율성을 높이는 방법입니다. 이러한 접근 방식은 LLM이 사전 훈련된 시퀀스 텍스트와 잘 맞아들어가지만, 긴 컨텍스트의 경우 주요 정보가 희석되는 문제가 발생할 수 있습니다. 또한, Context Distillation과 Semantic Balancing 기법을 도입하여 잡음이 많은 구조적 데이터 문제를 해결하고, 목표 텍스트의 의미를 보존하는 방법을 탐색합니다.

- **Performance Highlights**: 대규모의 제로샷 실험을 통해 이 연구는 구조 정보를 포함한 방법이 텍스트만 사용한 임베딩보다 항상 더 나은 성능을 발휘한다는 것을 보여줍니다. 특히 multi-hop 질문 답변과 같이 텍스트만으로는 부족한 작업에서 가장 큰 이점이 있음을 발견했습니다. Struc-Emb-Seq는 혼잡한 중간 길이 입력에 뛰어나지만 긴 텍스트에는 민감하게 반응하는 반면, Struc-Emb-Par은 긴 컨텍스트에서 신호가 높은 데이터에 안정적으로 스케일링합니다. 마지막으로, Context Distillation과 Semantic Balancing은 잡음이 많은 구조적 맥락 속에서 목표의 핵심 의미를 유지하는 효과적인 기법으로 입증되었습니다.



### Graph Diffusion Transformers are In-Context Molecular Designers (https://arxiv.org/abs/2510.08744)
Comments:
          29 pages, 16 figures, 17 tables. Model available at: this https URL

- **What's New**: 이번 연구에서 우리는 demonstration-conditioned diffusion 모델(DemoDiff)을 제안하여 분자 설계에서의 in-context learning(ICL)의 한계점을 극복하고자 합니다. 기존의 많은 방법론들이 많은 라벨링된 데이터에 의존하지만, DemoDiff는 소량의 예제를 활용하여 더 효율적인 분자 생성을 가능하게 합니다. 새로운 분자 토크나이저를 통해 분자 구조를 motif 수준으로 표현하여 사전 훈련의 효율성을 높였습니다.

- **Technical Details**: DemoDiff는 점수를 부여한 분자 샘플을 통해 작업 맥락을 정의하며, 이를 사용해 조정된 Transformer를 통해 분자를 생성합니다. 새로운 분자 토크나이저는 Node Pair Encoding(NPE)을 기반으로 하여 5.5배 더 적은 노드를 사용함으로써 motif 수준에서 분자를 표현합니다. 이로 인해 0.7B 파라미터를 가진 모델을 고속으로 사전 훈련할 수 있었습니다.

- **Performance Highlights**: 33개의 설계 작업 수행 결과, DemoDiff는 100–1000배 더 큰 language model보다 한편 위에 있거나 동등한 성능을 보여주며 평균 순위를 3.63으로 나타냈습니다. 이는 기존의 전문화된 방법들보다 월등한 성능을 나타내며, 새로운 분자 토크나이저가 성능 향상에 기여하였습니다.



### Coordinates from Context: Using LLMs to Ground Complex Location References (https://arxiv.org/abs/2510.08741)
Comments:
          Under review at ARR

- **What's New**: 이 논문에서는 비구조화된 텍스트에서 지리적 정보(geospatial information) 추출의 중요성을 강조하며, 특히 내부 이름 없이 구성된 위치 참조(compositional location references)의 지오코딩(geocoding) 문제를 다룹니다. 최근의 연구를 바탕으로 대규모 언어 모델(LLMs)의 지리적 지식과 추론 능력을 비교하고, LLMs를 활용한 새로운 전략을 제안합니다. 이 접근 방식은 기존 도구들이 해결하지 못한 복잡한 참조를 효과적으로 해결할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소를 포함하며, 하나는 LLM의 지리적 추론 능력(geospatial reasoning capabilities)이고, 다른 하나는 전통적인 지오파서(geoparsers)의 지리적 지식 기반입니다. 이 논문은 LLM이 지리적 데이터에 대해 얼마나 효과적으로 추론할 수 있는지를 평가하여, 이전 연구보다 더 나은 성과를 보여주는 방법을 제안합니다. 특히, 경계 상자(bounding box)를 활용하여 데이터베이스에 존재하지 않는 위치들까지 지오코딩할 수 있는 새로운 방식을 소개합니다.

- **Performance Highlights**: 상세한 평가 결과, LLM을 기반으로 한 전략이 이전의 방법들보다 높은 성능을 보이며, 상대적으로 작은 파인 튜닝(tuned)된 LLM이 훨씬 큰 상용 모델들과 유사한 성과를 달성할 수 있음을 보여줍니다. 이는 지리적 데이터의 추론과 지식 생성을 동시에 활용할 수 있음을 의미하며, 다양한 분야의 데이터 분석에 유용할 것으로 기대됩니다. 해당 연구는 지오코딩의 복잡한 문제를 해결하는 데 기여할 수 있는 독창적인 로드맵을 제시하고 있습니다.



### When to Reason: Semantic Router for vLLM (https://arxiv.org/abs/2510.08731)
Comments:
          5 pages, excluding references and appendix. To be appeared at Workshop on ML for Systems at NeurIPS 2025, December 6, 2025 this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 효율성을 높이기 위해 semantic router를 제안합니다. 이 시스템은 요청의 의미에 따라 쿼리를 분류하고, 유용한 경우에만 reasoning을 적용합니다. 이를 통해 MMLU-Pro 벤치마크에서 정확도가 10.2% 향상되고, 응답 지연시간은 47.1%, 토큰 소비는 48.5% 줄어들었습니다.

- **Technical Details**: 제안된 시스템은 사용자 프롬프트를 고차원 semantic embedding으로 인코딩하여 입력의 맥락적 의미를 파악합니다. 이후 intent classifier를 통해 프롬프트가 간단한 사실 기반 쿼리인지, reasoning이 필요한 쿼리인지를 결정합니다. 이를 바탕으로 적합한 추론 경로를 선택하여 효율적인 inference를 제공합니다.

- **Performance Highlights**: 평가 결과, semantic router는 MMLU-Pro 벤치마크의 14개 분야에서 정확도가 10.24% 향상되었고, 지연시간은 47.1% 감소했으며, 토큰 사용량은 48.5% 줄어드는 성과를 보였습니다. 특히 비즈니스 및 경제와 같은 지식 집약적인 도메인에서는 20% 이상의 정확도 향상을 달성하였으며, 이는 비용 효율성을 높이는데 기여합니다.



### Enhancing Self-Supervised Learning with Semantic Pairs A New Dataset and Empirical Study (https://arxiv.org/abs/2510.08722)
Comments:
          16 pages, 7 figures, 5 tables

- **What's New**: 논문에서는 self-supervised learning(SSL) 접근법인 instance discrimination을 다루며, 기존 이미지 변환에 대한 제한점을 극복하고자 semantic pairs(의미 쌍)가 활용됨을 강조합니다. 이러한 방법은 개별 인스턴스를 구별하고 일반화 능력을 향상시키기 위해 네트워크가 개체의 공통 정보를 포착하도록 유도합니다.

- **Technical Details**: instance discrimination은 각 인스턴스를 고유한 클래스로 간주하여 model이 효과적인 데이터 표현을 학습할 수 있도록 합니다. 이를 위해 서로 다른 두 개의 augmented views를 생성하고, semantic pairs를 통해 object의 맥락을 다양화하여 필수 정보에 집중하고 비관련 정보를 경량화하는 방법을 제안합니다.

- **Performance Highlights**: 논문에서는 semantic pairs를 통해 모델의 일반화 능력이 향상되며, occlusion, background 및 illumination 변화에 대한 불변성을 통해 더 많은 경우의 수를 처리할 수 있음을 보여줍니다. 본 연구는 이러한 접근법이 instance discrimination 방법론의 성능을 어떻게 개선할 수 있는지를 실험을 통해 검증합니다.



### In-Context Learning for Non-Stationary MIMO Equalization (https://arxiv.org/abs/2510.08711)
- **What's New**: 본 논문에서는 시간 변경 채널(time-varying channel)에서의 channel equalization을 위하여 in-context learning (ICL)의 가능성을 조사합니다. 기존의 ICL 기반 equalizers는 주로 정적인 채널 상에서 개발되었으며, 동적인 환경에서의 성능 향상을 위한 새로운 주의(attention) 메커니즘을 설계합니다. 제안하는 방법은 고전적 적응 신호 처리(adaptive signal processing) 알고리즘을 활용하여 방안과 성능을 크게 향상시킬 수 있는 잠재력을 제시합니다.

- **Technical Details**: 연구에서는 m1×m2 크기의 MIMO 자기 회귀 모델을 채택하여 시간에 따라 변화하는 채널을 모델링합니다. 이상의 모델은 메모리 요인(memory factor) ρ와 같은 변수를 포함하고 있으며, 적응형 필터링( adaptive filtering) 방법인 Least Mean Square (LMS) 알고리즘과 Least Root Mean Square (LRMS) 공식을 활용하여 새로운 주의 메커니즘을 제안합니다. 이를 통해 긴-term dynamics 인식을 개선하고 robust한 equalization을 목표로 합니다.

- **Performance Highlights**: 실험 결과, ICL은 비정상 MIMO equalization 문제에 대해 높은 성과를 보여주며, 고전적인 적응 알고리즘에서 영감을 얻은 새로운 주의 메커니즘은 동적인 환경에서도 적응성과 성능을 상당히 향상시킬 수 있음을 입증합니다. 이러한 연구 결과는 차세대 무선 기초 모델 개발을 위한 중요한 통찰력을 제공합니다.



### ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing (https://arxiv.org/abs/2510.08705)
- **What's New**: 본 연구는 비전문가 자율 로봇을 사용한 협동 물체 운반에 대한 새로운 방법인 ConPoSe를 소개합니다. 이 방법은 대형 언어 모델(LLMs)과 지역 탐색(local search)을 결합하여 로봇-물체 접촉점을 효율적으로 선택합니다. ConPoSe는 정해진 경로를 따라 물체를 밀기 위한 접촉점을 잘 선택하며, 물체의 형태와 중심만을 기반으로 작동합니다.

- **Technical Details**: 이 연구에서는 2D 환경 내에서 비슷한 종류의 로봇들로 구성된 동질 다중 로봇 시스템을 사용합니다. 로봇은 물체와 단일 점 접촉을 통해 상호작용하며, 안정적인 접촉점을 유지하는 것이 어려운 반비선형 방식입니다. 연구에서는 환경 내 장애물과의 충돌을 피해야 하며, 각 로봇이 발생시키는 힘의 크기는 동일하게 설정됩니다.

- **Performance Highlights**: 실험 결과, ConPoSe는 전통적인 분석적 방법보다 우수한 시간 확장성을 제공하며, 성공률도 거의 완벽에 가깝습니다. 또한, 순수 LLM 기반의 접촉점 선택 방법보다 경로를 따라 물체를 밀기 위해 필요한 접촉점 전환이 적어 더 효율적입니다. 이러한 성과들은 ConPoSe의 다양한 물체 형태 및 큰 규모의 시나리오에서의 적용 가능성을 강조합니다.



### BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution (https://arxiv.org/abs/2510.08697)
Comments:
          Built with love by the BigCode community :)

- **What's New**: BigCodeArena는 코드 생성을 위한 개방형 인간 평가 플랫폼으로, Chatbot Arena를 기반으로 하여 개발되었습니다. 이 플랫폼은 LLM이 생성한 코드를 실행할 수 있는 환경을 제공하며, 사람들은 실행 과정과 결과와 상호작용할 수 있습니다. 특히, 10개 언어와 8개 실행 환경을 포함한 14,000명이 넘는 코드 중심 대화 세션을 수집하였고, 그 중 4,700개의 다중 턴 샘플에서 사람의 선호도를 추출했습니다.

- **Technical Details**: BigCodeArena는 사용자에게 코드의 기능적인 평가를 가능하게 하는 직관적인 인터페이스를 지원합니다. 사용자들이 코드 실행 및 검토 후 선호 응답을 선택할 수 있도록 하며, 코드 편집 및 대화 기록 기능도 제공합니다. 모델의 응답은 동시에 실행되며, 이를 통해 평가가 시스템 지연이나 실행 시간 같은 편향 영향을 받지 않도록 엄격한 동기화가 이루어집니다.

- **Performance Highlights**: LLM의 코드 생성 성능 평가에서는 BigCodeReward와 AutoCodeArena라는 두 가지 기준이 설계되었습니다. 평가 결과, 대부분의 LLM이 실행 결과가 제공될 경우 코드 선호도를 판단하는 데 뛰어난 성능을 보였습니다. 특히 GPT-5와 Claude 계열 모델들이 최근의 모델 중에서 여전히 코드 생성 성능이 우수함을 보여주었습니다.



### RAG4Tickets: AI-Powered Ticket Resolution via Retrieval-Augmented Generation on JIRA and GitHub Data (https://arxiv.org/abs/2510.08667)
Comments:
          13 Pages

- **What's New**: 이 논문에서는 소프트웨어 팀이 JIRA 티켓, 개발자 토론, GitHub 풀 리퀘스트(PR) 등의 분산된 지식으로 인해 발생하는 문제를 해결하기 위해 Retrieval-Augmented Generation (RAG) 프레임워크를 제안합니다. 이 프레임워크는 세맨틱 임베딩(semantic embeddings)을 위한 Sentence-Transformers와 FAISS 기반의 벡터 검색(vector search)을 통합하여 문맥을 고려한 티켓 해결 추천을 제공합니다. 기존 JIRA 티켓, 사용자 댓글, 연결된 PR 메타데이터를 임베딩하여 의미적으로 유사한 과거 사례를 검색합니다.

- **Technical Details**: 이 시스템은 JIRA와 GitHub 데이터를 연결하는 통합 파이프라인 및 이종 소프트웨어 아티팩트에 대한 임베딩과 FAISS 인덱싱 전략을 제공합니다. 이를 통해 검색된 증거에 의해 안내되는 해결 생성 모듈을 구성하여 최적의 해결책을 도출합니다. Large Language Model(LLM)을 사용하여 과거 사례를 통합하여 현실적이고 설명 가능한 해결책을 제시합니다.

- **Performance Highlights**: 실험 평가 결과, 제안된 시스템은 정확성(accuracy), 수정 품질(fix quality), 그리고 지식 재사용(knowledge reuse)을 크게 향상시키는 것으로 나타났습니다. 또한, 정밀도(precision), 재현율(recall), 해결 시간 단축(resolution time reduction) 및 개발자 수용성(developer acceptance) 지표를 통해 성능 개선이 확인되었습니다.



### dInfer: An Efficient Inference Framework for Diffusion Language Models (https://arxiv.org/abs/2510.08666)
- **What's New**: 최근 확산 기반 대형 언어 모델(이하 dLLMs)은 자동 회귀 모델(AR LLMs)의 유망한 대안으로 주목받고 있습니다. 그러나 dLLMs의 보편적 채택은 표준화된 효율적인 추론 프레임워크 부족으로 제한되고 있습니다. 이를 해결하기 위해 'dInfer'라는 효율적이고 확장 가능한 추론 프레임워크를 제안하며, 이는 4개의 모듈화된 구성요소로 구성되어 있습니다.

- **Technical Details**: dInfer는 모델, 확산 반복 관리자, 디코딩 전략 및 KV-캐시 관리자라는 네 개의 모듈로 구성되어 있으며, 각 구성요소에 대한 새로운 알고리즘과 시스템 수준의 최적화를 통합합니다. 특히, KV 캐시 관리의 효율성을 위한 이웃 갱신 전략과 디코딩 효율을 높이기 위한 계층적 및 신용 디코딩을 도입했습니다. 이러한 설계를 통해 dInfer는 GPU 활용을 극대화할 수 있습니다.

- **Performance Highlights**: dInfer는 HumanEval에서 초당 1,100개 이상의 토큰을 생성하며, 8개의 기준병에 걸쳐 평균적으로 800개 이상의 토큰을 생성합니다. 기존 Fast-dLLM에 비해 10배의 속도 향상을 달성했으며, AR 모델과 비교해도 2-3배의 속도 향상을 제공합니다. dInfer의 구현은 오픈소스로 제공되며, 사용자는 이 플랫폼에서 dLLM 추론을 AR 모델보다 뛰어난 효율성을 가지고 사용할 수 있습니다.



### RA-Gen: A Controllable Code Generation Framework Using ReAct for Multi-Agent Task Execution (https://arxiv.org/abs/2510.08665)
- **What's New**: 이번 논문에서는 ReAct 패러다임을 활용한 다중 에이전트 코드 생성 프레임워크를 제안합니다. 이 프레임워크는 LLMs와 외부 리소스 간의 동적 상호작용을 통해 효율적이고 정확한 코드 생성을 가능하게 하며, 사용자 제어성과 안전성을 향상시킵니다. 특히, 기존의 방법들이 드러내지 못한 투명한 추론과 동적 도구 통합의 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 시스템은 네 개의 전문 에이전트로 구성됩니다: Planner(계획자), Searcher(검색기), CodeGen(코드 생성기), Extractor(추출기). 이들은 서로 협력하여 복잡한 코드 생성을 위한 작업 분해와 이유 추적을 통해 효율적으로 함수 구현을 지원합니다. Searcher 에이전트는 ReAct 프레임워크를 사용하여 동적으로 이유 경로를 생성하고 작업을 실행하며, 외부 도구와의 통합을 통해 정확성과 안전성을 높입니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 다양한 프로그래밍 언어로 복잡한 코드 생성 작업을 효과적으로 처리하며, SVEN 데이터셋에서 CodeQL을 사용하여 94.8%의 보안 비율을 달성했습니다. 이 결과는 기존의 접근 방식보다 뛰어난 성능을 보여줍니다. 투명한 추론 과정을 통해 사용자 신뢰도를 높이고, 코드 생성의 통제성을 개선합니다.



### Faver: Boosting LLM-based RTL Generation with Function Abstracted Verifiable Middlewar (https://arxiv.org/abs/2510.08664)
- **What's New**: 이번 논문에서는 LLM 기반의 RTL 생성을 개선하기 위한 새로운 프레임워크인 Faver를 제안합니다. RTL 생성에서 가장 자동화가 덜 된 단계인 회로 검증을 간소화하는 데 중점을 둡니다. Faver는 LLM이 기능에 더 집중할 수 있도록 설계되었습니다.

- **Technical Details**: Faver 프레임워크는 LLM이 Python/C와 같은 고급 언어로 검증 코드를 작성하도록 돕습니다. RTL 설계를 위한 입력 및 검증 사양을 관리하는 방식으로, 각 단계에서 LLM이 필요한 기능 모델과 테스트 자극을 생성합니다. 이를 통해 Python-Verilog co-simulation을 수행하여 검증 보고서를 생성하며, LLM에 다시 피드백을 제공합니다.

- **Performance Highlights**: 실험 결과, Faver는 다양한 테스트 세트 및 모델에서 LLM 기반 RTL 생성의 정확도를 최대 14%까지 개선한다는 것을 보여주었습니다. LLM이 회로 검증을 위해 고급 의미론적 코드를 작성하도록 허용함으로써 설계 검증의 효율성을 높였습니다.



### A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data (https://arxiv.org/abs/2510.08663)
- **What's New**: 이번 연구는 심리 평가에서 전통적인 평정 척도 대신 최근의 LLM(대형 언어 모델) 발전을 이용하여 응답자의 자연어에서 질적 데이터를 활용하는 새로운 개념적 프레임워크를 제시합니다. 정신 건강을 평가하기 위한 도구로 우울증을 사례로 사용하여, LLM이 평가한 텍스트와 전통적인 평정 항목을 결합한 보강된 테스트를 개발하였습니다.

- **Technical Details**: 이 연구는 실제 고등학생 샘플(n=693)과 인공 샘플(n=3,000)을 기반으로 보강된 테스트에 대한 평가를 진행하였습니다. LLM 항목에서 얻은 정보는 원래 19개 항목 테스트에 6.3개(실제 데이터)에서 16.0개(인공 데이터) 항목을 추가하는 것과 동등한 데이터 정보를 제공합니다. 이러한 접근 방식은 사전 레이블 데이터나 복잡한 전문가 생성 루브릭에 의존하지 않고, 항목 정보 계산에 기초하여 가장 유익한 LLM 평가 지침을 선택합니다.

- **Performance Highlights**: 보강된 테스트는 고립된 테스트 세트에서 통계적으로 유의미한 측정 정밀도와 정확도의 향상을 보여주었습니다. 이 연구는 심리 측정 도구에 전사된 텍스트의 증가하는 흐름을 활용하는 확장 가능한 접근 방식으로, 임상 건강 분야 및 그 이상의 잠재적 유용성을 논의합니다.



### DPCformer: An Interpretable Deep Learning Model for Genomic Prediction in Crops (https://arxiv.org/abs/2510.08662)
Comments:
          This work has been accepted by BIBM 2025

- **What's New**: 이 논문은 딥러닝 모델인 DPCformer를 제안하여 복잡한 유전자형-형질 관계를 모델링하고 DNA 서열의 단일 뉴클레오타이드 다형성(SNP) 데이터를 이용해 작물 형질을 예측합니다. DPCformer는 CNN(Convolutional Neural Network)과 다중 헤드 자기 주의 메커니즘(self-attention mechanism)을 통합하여 복잡한 작물 형질의 예측 정확도를 향상시키는 방법을 제공합니다. 기존의 유전체 선택 방법들이 가진 한계점을 극복하고, 특히 소형 샘플에서도 강력한 성능을 보여줍니다.

- **Technical Details**: DPCformer는 8차원 원-핫 인코딩(one-hot encoding) 전략과 확률적 행렬 분해(Probabilistic Matrix Factorization, PMF) 알고리즘을 통해 특징 선택을 수행합니다. 이 모델은 다수의 작물(예: 옥수수, 면화, 토마토, 벼, 병아리콩)에 대해 실험되었으며, 각 작물의 특정 형질에 대한 데이터를 효과적으로 분석하여 가장 적합한 예측 모델을 수립합니다. 연구 결과, DPCformer는 기존의 모델들보다 높은 예측 정확도를 보이며, 특히 소형 샘플에서도 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: DPCformer는 기존의 유전체 선택 방법들과 비교하여 예측 정확도에서 우수한 성능을 발휘합니다. 예를 들어, 옥수수 데이터셋에서 개화일까지의 일수와 식물 높이에 대한 정확도가 최대 2.92% 증가하였고, 면화의 섬유 형질 정확도가 8.37% 향상되었습니다. 소형 샘플 토마토 데이터에서도 주요 형질의 피어슨 상관계수가 57.35% 증가했으며, 병아리콩에서는 수확량 상관성이 16.62% 증대되었습니다.



### CATS-Linear: Classification Auxiliary Linear Model for Time Series Forecasting (https://arxiv.org/abs/2510.08661)
- **What's New**: 이 논문은 CATS-Linear라는 새로운 예측 모델을 제안하면서 선형 모델의 성능 향상을 위한 새로운 접근 방식을 탐구합니다. 여기에 분류 보조 채널 독립성(Classification Auxiliary Channel-Independence, CACI) 기법이 포함되어 있습니다. CACI는 각 시계열을 분류하고, 각 시계열의 예측을 위해 전용 예측기에게 라우팅할 수 있도록 동적으로 작업합니다. 이로 인해 채널 예측 정렬이 향상되고 복잡성이 감소합니다.

- **Technical Details**: CATS-Linear는 계절 성분과 추세 성분을 변환하여 선형 예측을 수행하는 두 가지 주요 기법을 사용합니다. 계절 성분은 복소수로 변환되어 주기성을 적용하고, 추세 성분은 각각의 시간 단계 상태로 나누어 선형 매핑을 적용 후 다시 재결합됩니다. 이 새로운 접근 방식은 기존 DLinear의 한계를 극복하며 향상된 정확도를 제공합니다. 또한, RevIN 모듈을 적용하여 입력 인스턴스의 분포 이동 문제를 해결하여 예측의 정확성을 높입니다.

- **Performance Highlights**: CATS-Linear는 고정된 하이퍼파라미터로도 하이퍼파라미터를 조정한 벤치마크 모델과 동등한 성능을 발휘하며, 통합된 하이퍼파라미터 모델 대비 8% 이상의 MSE 감소를 기록했습니다. 이는 선형 모델이 복잡한 아키텍처와 비교하여 경쟁력 있는 성능을 발휘할 수 있음을 보여줍니다. 광범위한 실험을 통해 CATS-Linear의 효과성을 입증하였습니다.



### Provably Robust Adaptation for Language-Empowered Foundation Models (https://arxiv.org/abs/2510.08659)
Comments:
          19 pages

- **What's New**: 이 연구는 언어 기반 파운데이션 모델(LeFMs)을 위한 최초의 증명 가능한 강건성(few-shot classifier)을 제안합니다. 모델 이름은 Language-empowered Few-shot Certification (LeFCert)으로, 텍스트 임베딩과 피쳐 임베딩을 결합한 하이브리드 프로토타입 기반의 클래스파이어입니다. 이 접근 방식은 적대적 샘플의 영향을 줄이면서도 강건성을 확보할 수 있도록 설계되었습니다.

- **Technical Details**: LeFCert는 두 가지 변형을 포함하여 더욱 현실적인 위협 모델을 수용합니다. LeFCert-L은 이중 제약 조건 하의 공격 시나리오에서 강건성을 보장하며, 랜덤화된 스무딩(rnandomized smoothing)을 통해 임베딩 공간의 연속성을 확보합니다. LeFCert-C는 여러 테스트 샘플에 대한 집합적 인증을 제공하여 여러 샘플에서 공격 예산을 분배하는 공격자에 대응합니다.

- **Performance Highlights**: 실험 결과, LeFCert는 CIFAR-FS, Tiered-ImageNet와 같은 다양한 데이터셋에서 기존의 방법들보다 높은 정확도를 기록했습니다. 예를 들어, CIFAR-FS에서는 98%의 청결 정확도를 기록하여 FCert의 88%와 KNN의 80%를 초과했습니다. LeFCert-C는 특히 집합적 인증 성능에서 두드러진 개선을 보여줍니다.



### Inner-Instance Normalization for Time Series Forecasting (https://arxiv.org/abs/2510.08657)
- **What's New**: 이 논문에서는 시계열 예측에서의 내부 인스턴스 분포 이동(inner-instance distribution shift)이라는 새로운 개념을 도입합니다. 기존의 방법들은 개별 인스턴스 내에서의 이동을 고려하지 않아 성능이 저하되는 문제를 가지고 있습니다. 이를 해결하기 위해 Learning Distribution (LD)와 Learning Conditional Distribution (LCD)이라는 두 가지 새로운 점 수준(point-level) 방법을 제안합니다. 이러한 접근법은 예측 모델의 성능을 눈에 띄게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: LD 방법은 입력 및 출력의 내부 분포를 다른 시간 단계에서 적합하는 학습 가능한 매개변수를 도입합니다. 이는 z-score 정규화를 적용하고, 각 시간 단계에서의 기대값을 제거하여 내재된 분포 이동을 줄이는 방식입니다. 반면, LCD는 신경망을 활용하여 출력의 스케일링 계수를 예측함으로써 조건부 분포 P(𝒴|𝒳)를 학습합니다. 이러한 방법은 전통적인 시계열 모델들이 다루지 못했던 문제를 해결하는데 기여하게 됩니다.

- **Performance Highlights**: LD와 LCD는 공개 벤치마크에서 다양한 백본 모델을 통해 성능을 평가받았습니다. 실험 결과, LD는 기존의 RevIN 모델에 비해 일관되게 우수한 성능을 나타내었습니다. 또한, LCD는 Dish-TS와 SAN같은 인스턴스 수준 및 슬라이스 수준의 정규화 방법보다 더 나은 성능을 보여, 다차원 시계열 예측의 새로운 기준을 제시합니다.



### Knowledge Graph Sparsification for GNN-based Rare Disease Diagnosis (https://arxiv.org/abs/2510.08655)
- **What's New**: 이번 연구에서 제안된 RareNet은 희귀 유전 질환 진단의 새로운 패러다임을 제시합니다. 기존 진단 도구들과는 달리 RareNet은 환자의 표현형(phenotype) 데이터만을 사용하여 가장 가능성 있는 원인 유전자(causal gene)를 식별합니다. 이러한 접근 방식은 진단 도구에 대한 접근이 제한된 자원이 부족한 지역사회의 환자들에게 특히 도움이 됩니다.

- **Technical Details**: RareNet은 서브그래프 기반의 Graph Neural Network (GNN)를 활용하여 환자의 표현형을 중심으로 정보가 포함된 서브그래프를 샘플링합니다. 모델은 두 가지 목표 함수를 공동 최적화하여 유의미한 서브그래프를 추출하고, 후보 유전자 리스트를 정확하게 우선 순위화합니다. 이 방식은 불완전하거나 부정확한 표현형 데이터에서도 견고하게 작동하며, 전반적인 개인화된 진단을 가능하게 합니다.

- **Performance Highlights**: 본 연구의 결과는 RareNet이 기존의 유전자 우선 순위화 방법들을 향상시키는 데 기여할 수 있음을 보여줍니다. 종합 평가를 통해 생물 의학 데이터 세트에서 경쟁력 있는 원인 유전자 예측 성능을 나타냈으며, 타 방법과 통합 시 성능 향상이 크게 개선되었습니다. 이는 기계 학습과 AI 기반 기술을 활용하여 희귀 질환 진단의 접근성과 정확성을 높일 가능성을 나타냅니다.



### Formalizing Style in Personal Narratives (https://arxiv.org/abs/2510.08649)
- **What's New**: 이 논문에서는 개인의 서사(narratives)의 스타일(style)을 언어 선택의 패턴(patterns)으로 체계적으로 분석하기 위한 새로운 접근 방식을 제시합니다. 기존 연구에는 이러한 언어적 선택을 분석할 수 있는 정형화된 프레임워크가 부족했으며, 이에 따라 언어학(functional linguistics), 컴퓨터 과학(computer science), 심리학(psychology) 세 가지 영역을 통합한 접근을 제안합니다. 이는 개인 서사가 전달하는 주관적 경험의 전달 방식을 이해하는 데 기초가 됩니다.

- **Technical Details**: 우리는 체계적 기능 언어학(systemic functional linguistics)을 기반으로 하여 언어 특징을 범주화하고, 각 언어적 요소가 주관적 경험을 어떻게 전달하는지를 분석합니다. 특정 과정(processes), 참가자(participants), 상황(circumstances)의 분석을 통해 개인 서사가 어떻게 형성되는지를 설명합니다. 이때, 언어의 기능은 사회적 의미를 생성하는 방식으로 검토되며, 서사에서의 언어 선택이 어떻게 이루어지는지를 규명합니다.

- **Performance Highlights**: 예제 사례로 전쟁 참전용사의 꿈 서사를 분석한 결과, 그의 서사에서 언어적 선택이 어떻게 심리적 상태와 관련되어 있는지를 보여주는 독특한 패턴이 드러났습니다. 분석 결과에 의하면, 언어적 과정에서 정신적(processes)보다 언어적(mental processes) 과정이 지나치게 지배적이며, 이는 내러티브가 주관적 경험을 표현하는 데 중요한 역할을 함을 강조합니다. 이 프레임워크는 향후 연구와 치료적 적용의 잠재력을 구체적으로 제안하며, 저자 프로파일링(authorship profiling)와 내러티브 생성(narrative generation) 등 다양한 활용 가능성을 제시합니다.



### Inverse-Free Wilson Loops for Transformers: A Practical Diagnostic for Invariance and Order Sensitivity (https://arxiv.org/abs/2510.08648)
Comments:
          24 pages, 10 figures, 2 tables

- **What's New**: 이 논문에서는 큰 언어 모델이 실제에서 중요한 무해한 수정(harmless edits)에 따라 답변을 변경할 수 있는 경우를 설명합니다.  특히, RAG (Retrieval-Augmented Generation) 출력이 패시지의 순서에 따라 뒤집히고, 파인튜닝(fine-tuning)이 사전 학습(pretraining)에서 배운 불변성을 침식하게 되었음을 강조합니다. WILSON이라는 최소한의 후속 진단(diagnostic) 도구를 제안하여 내부 표현에 대한 간단한 루프 및 재배치(checks) 검사를 시스템 신호로 변환합니다.

- **Technical Details**: WILSON은 JVPs (Jacobian-vector products)와 Hutchinson 프로브를 이용하여 위치(position) 및 레이어(layer) 전반에 대해 계산된 역학 없이(curvature map) 나타낼 수 있는 기법을 포함합니다. 여기에는 활성화 수준(activation-level)에서의 교환(comutators) 위험을 표시하는 방법이 결합되어 있습니다. 이 신호들은 계산이 저렴하고, 표준 트랜스포머(transformers)에 대해 모델에 구애받지 않으며, 오케스트레이터( orchestrators)용으로 임계값(thresholds) 및 CSV 아티팩트 형태로 내보낼 수 있습니다.

- **Performance Highlights**: WILSON은 RAG의 순서 효과를 방지하고, 파인튜닝 회귀(fine-tuning regressions)를 잡아내며, 논쟁 경로(debate pathways) 및 긴 다중 턴(multi-turn) 컨텍스트를 안정시키는 임무를 수행합니다. 또한 배포(deployment) 과정에서 융합(fusions)이나 재배치를 통제(gating)할 수 있도록 도와줍니다. 이로 인해, 모델 아키텍처나 훈련을 변경하지 않고도 신뢰성과 처리량(throughput)을 동시에 개선할 수 있는 가능한 행동을 사전 예측하고 안전한 최적화를 승인할 수 있습니다.



### Upfront Chain-of-Thought: A Cooperative Framework for Chain-of-Thought Compression (https://arxiv.org/abs/2510.08647)
Comments:
          ACL2026 Under Review

- **What's New**: 이번 연구에서는 Upfront CoT (UCoT)라는 새로운 추론 프레임워크를 제안합니다. 이 프레임워크는 직접적인 생각 임베딩을 통해 코드를 압축하는 과정을 자동화하여 효율성을 높입니다. UCoT는 소규모 모델(압축기)과 대규모 모델(실행기) 간의 협력적 워크플로우를 통해 이루어집니다.

- **Technical Details**: UCoT의 첫 번째 단계에서는 압축기가 실행기를 위해 풍부한 추론 정보를 담은 upfront thought embeddings를 생성하도록 훈련됩니다. 두 번째 단계에서는 실행기가 이러한 임베딩을 활용하여 짧은 추론으로 정답을 도출하도록 최적화됩니다. 이 과정은 보상 메커니즘(reward mechanism)을 통해 수행됩니다.

- **Performance Highlights**: 실험 결과, UCoT는 실행기의 강력한 추론 능력을 유지하면서도 CoT의 길이를 상당히 줄이는 것을 보여주었습니다. 또한, Qwen2.5-7B-Instruct 모델에 UCoT를 적용했을 때, GSM8K 데이터셋에서의 토큰 사용량이 50% 감소하면서 성능은 최신 방법(SOTA)보다 3.08% 향상되었습니다.



### Energy-Driven Steering: Reducing False Refusals in Large Language Models (https://arxiv.org/abs/2510.08646)
- **What's New**: 이번 논문에서는 Energy-Driven Steering (EDS)라는 새로운 프레임워크를 소개합니다. EDS는 대형 언어 모델의 안전성과 유용성을 동시에 개선하기 위해 설계되었습니다. 이 방법은 외부 Energy-Based Model (EBM)을 활용해 모델 내부 상태의 에너지를 동적으로 조정하여, 부정확한 거부(false refusal)를 줄이는 것을 목표로 합니다.

- **Technical Details**: EDS는 경량화된 EBM을 통해 모델의 내부 활성화를 분석합니다. 이 EBM은 부적절한 상태에 높은 에너지를, 적절한 상태에 낮은 에너지를 부여하여 에너지 풍경(energy landscape)을 형성합니다. 추론(inference) 과정에서 EDS는 이 에너지 함수의 그래디언트를 활용해 활성화를 저에너지 지역(low energy regions)으로 유도하여, 모델이 유용한 응답을 즉시 생성하도록 조정합니다.

- **Performance Highlights**: EDS는 다양한 벤치마크에서 다른 방법들보다 우수한 성능을 보였습니다. ORB-H 벤치마크에서의 준수(compliance)율을 57.3%에서 82.6%로 증가시키면서도 기존의 안전성 성능을 유지했습니다. 이러한 결과는 EDS가 낮은 false refusal 비율과 높은 안전성을 동시에 달성할 수 있음을 보여줍니다.



### Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools (https://arxiv.org/abs/2510.08640)
- **What's New**: 이 논문은 Android의 빌드 오류를 해결하기 위한 새로운 벤치마크인 AndroidBuildBench를 소개합니다. 이 벤치마크는 43개의 오픈 소스 Android 프로젝트에서 수집된 1,019개의 빌드 실패 사례로 구성되어 있으며, 각각의 문제에 대한 검증된 해결책을 제공합니다. 또한, 특정 도구를 사용하는 LLM 기반의 에이전트인 GradleFixer를 제안하고, 이를 통해 높은 해결 성공률을 달성했다는 점이 주목할 만합니다.

- **Technical Details**: AndroidBuildBench는 Java/Kotlin을 사용하는 상위 100개 스타 앱에서 선택한 43개 프로젝트로부터 수집되었습니다. 이 데이터셋은 분석을 통해 각각의 빌드 실패와 이를 수정하는 커밋을 쌍으로 연결하여 검증 가능한 문제-해결 쌍을 생성하는 방식으로 구성됩니다. GradleFixer는 이러한 도구를 사용하여 Gradle 빌드 환경을 검사하고 조작하는 LLM 에이전트로, 일반적인 쉘 명령어 대신에 특정 도메인을 인식하여 수정 작업을 수행합니다.

- **Performance Highlights**: GradleFixer는 81.4%라는 높은 해결율을 기록하여 기존의 최신 코딩 에이전트보다 뛰어난 성능을 보여줍니다. 이 접근 방식은 Tool Bridging 이론을 기반으로 구축되어, 도메인 인식 추상화를 통해 모델의 높은 수준의 추론을 효율적인 낮은 수준의 실행으로 연결합니다. 이는 단순히 API처럼 작동하는 도구를 제공하여 LLM이 보다 신뢰성 있게 작업할 수 있게 하여, 비효율적인 작업 공간을 줄이는 데 기여합니다.



### From What to Why: Thought-Space Recommendation with Small Language Models (https://arxiv.org/abs/2510.08626)
Comments:
          15 pages, 3 figures

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 높은 비용 문제를 해결하기 위해 SLMs(소형 언어 모델)를 기반으로 사용자와 아이템에 대한 공통 이해를 구축하는 'Thought Space' 개념을 제안합니다. PULSE(Preference Understanding by Latent Semantic Embeddings)라는 새로운 프레임워크는 SLM이 생성한 근거를 학습 신호로 활용하여 사용자 행동과 그 이면의 의미적 원인을 함께 모델링합니다. 이를 통해 높은 수준의 설명 가능성을 유지하면서도 효율적인 추천 시스템을 구현할 수 있도록 합니다.

- **Technical Details**: PULSE는 SLM에서 생성된 근거를 감독 학습 신호로 사용하는 방식으로, 기존의 학습 방법과 차별화됩니다. 특히, 이 방법은 행동과 근거 간의 대비(constrastive)를 통해 더 강건하고 일반화 가능한 임베딩을 생성합니다. SLM이 생성한 근거는 단순히 수동적으로 소비되는 것이 아니라, 현재 도메인에 대한 감독 신호로 최적화되어 사용됩니다.

- **Performance Highlights**: 광범위한 실험 결과, PULSE는 여러 벤치마크 데이터셋에서 기존 ID, 협업 필터링(CF), LLM 기반 순차 추천 모델에 비해 개선된 성능을 보였습니다. 특히, PULSE는 크로스 도메인 추천에서 우수한 변환 가능성을 나타내며, 추론 중심의 질문 응답 작업에서도 앞선 성과를 기록했습니다.



### Impact of LLMs on Team Collaboration in Software Developmen (https://arxiv.org/abs/2510.08612)
- **What's New**: 이번 연구는 Large Language Models (LLMs)이 소프트웨어 개발 프로세스에 어떻게 통합되고 있는지를 조사하며, 팀 협력을 통해 생산성을 어떻게 향상시킬 수 있는지를 탐구합니다. 최근 2025년의 발전 사항을 반영하여 이전 연구를 업데이트하고, LLM 사용의 장단점을 다룹니다. 사례 연구와 팀 설문을 통해 LLM 보조 도구들이 소프트웨어 엔지니어링 관행에 미치는 영향을 평가합니다.

- **Technical Details**: 소프트웨어 개발 생애 주기(SDLC)는 소프트웨어 시스템 설계, 개발, 테스트, 배포 및 유지 관리의 구조화된 프로세스를 의미합니다. 연구에서는 소프트웨어 개발의 각 단계에서 발생하는 협력 장애물 그리고 LLM 기반 도구들이 팀 생산성, 의사소통 및 의사 결정을 개선하는 방법에 대해 논의합니다. 특히, LLM을 활용한 코드 생성 도우미 및 AI 기반 프로젝트 관리 에이전트의 사례를 제시합니다.

- **Performance Highlights**: LLMs은 반복적인 작업과 문서화를 자동화하여 효율성을 크게 높일 수 있으며, 명확한 의사소통을 촉진하고 교차 기능 협업을 지원합니다. 반면, 모델의 한계와 프라이버시 문제와 같은 새로운 도전 과제가 발생할 수 있습니다. 이 연구는 LLM 도구의 도입이 팀 협력에 미치는 긍정적 및 부정적 영향을 평가하고, 향후 연구 방향에 대해 논의합니다.



### Relative Positioning Based Code Chunking Method For Rich Context Retrieval In Repository Level Code Completion Task With Code Language Mod (https://arxiv.org/abs/2510.08610)
Comments:
          Accepted to Context Collection Workshop co-located with ASE 2025

- **What's New**: 이 논문은 코드 자동 완성(code completion) 시스템의 효율성을 향상시키기 위한 새로운 효과적인 콘텍스트 수집 전략을 제안합니다. 현재의 코드 자동 완성 시스템은 사용자의 의도와 주변 코드에 대한 이해를 개선하는 데 필요하지만, 이 분야에 대한 연구는 부족합니다. 저자들은 코드를 작은 덩어리로 나눠 상대적 위치정보를 사용하여 더 좋은 자동 완성을 위한 콘텍스트를 제공하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서 제안된 솔루션은 repository의 모든 소스 코드를 n행 덩어리로 나누고, 각 덩어리의 이전 및 다음 포인터를 추적하여 데이터베이스에 저장하는 방법입니다. 이후, 각 덩어리는 HuggingFace의 'sentence-transformers/all-MiniLM-L6-v2' 모델을 사용하여 벡터 임베딩을 생성합니다. 마지막으로, BM25와 FAISS를 결합한 Ensemble Retriever를 사용하여 유사한 코드 덩어리를 검색하는 방식으로 접근합니다.

- **Performance Highlights**: 저자들은 제안된 접근 방식으로 코드 자동 완성 대회에서 Kotlin 트랙에서 0.660의 평균 chrF 점수를 기록하고, Python 트랙에서는 0.636의 평균 점수를 달성했습니다. 이 솔루션은 Kotlin 트랙에서 동메달(3위)을 받고, Python 트랙에서는 4위를 차지했습니다. 코드베이스는 오픈소스로 제공되어 다른 연구자들이 참고할 수 있도록 합니다.



### MMA-ASIA: A Multilingual and Multimodal Alignment Framework for Culturally-Grounded Evaluation (https://arxiv.org/abs/2510.08608)
- **What's New**: 이번 논문은 아시아 문화에 대한 대규모 언어 모델(LLMs)의 평가 프레임워크인 MMA-ASIA를 제안합니다. MMA-ASIA는 8개 아시아 국가와 10개 언어를 포함하여 27,000개의 질문을 갖추고 있으며, 79% 이상이 문화적 맥락에 기반한 다단계 추론을 요구합니다. 이 데이터셋은 텍스트, 이미지, 음성 등 삼중 입력 형식에서 일관된 정답을 제공하는 모델의 능력을 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MMA-ASIA는 문화 인식의 일관성, 언어 간 일관성, 모달리티 간 일관성을 평가하는 다섯 가지 차원의 평가 프로토콜을 제시합니다. 특히, 'shortcut learning'을 탐지하기 위한 문화 인식 그라운딩 검증 모듈을 포함하고 있어 모델의 추론 능력을 더 엄격하게 평가합니다. 이 프레임워크는 15종의 다국어, 다모달 LLM을 평가하며, 충족되지 않은 아시아 언어에 대한 성능 저하와 관련된 통찰을 제공합니다.

- **Performance Highlights**: MMA-ASIA의 평가 결과에 따르면, 낮은 자원 언어에서 정확도가 영어에 비해 크게 떨어진다는 것을 확인했습니다. 또한, 텍스트 전용 성능에 비해 모달리티 간 일관성이 뒤처지는 것으로 나타났습니다. 이를 통해 문화적 맥락을 반영한 모델링이 필요하다는 점과 모달리티 간 지식 전이가 미비함을 강조하고 있습니다.



### Centering Emotion Hotspots: Multimodal Local-Global Fusion and Cross-Modal Alignment for Emotion Recognition in Conversations (https://arxiv.org/abs/2510.08606)
Comments:
          Under review for ICASSP 2026

- **What's New**: 본 논문에서는 Emotion Recognition in Conversations (ERC) 분야에서 감정의 "hotspot"를 중심으로 한 새로운 접근 방식을 제안했습니다. Hotspot-Gated Fusion (HGF) 및 Mixture-of-Aligners (MoA)라는 혁신적인 기법을 통해 텍스트, 오디오, 비디오의 감정 hotpsots를 정밀하게 탐지하고 이를 기반으로 다양한 모달리티를 정렬하는 통합 모델을 개발했습니다. 또한, 실험을 통해 제안한 모델이 기존의 강력한 baseline에 비해 일관된 성능 향상을 나타냈음을 입증했습니다.

- **Technical Details**: 이 연구에서 제안된 모델은 감정 hotspot을 중심으로 한 통합 설계를 기반으로 하며, 텍스트, 오디오, 비디오의 데이터를 유연하게 처리합니다. HGF는 각 모달리티의 localized high-intensity segment를 식별하고 무작위 시간 정렬을 피하며 글로벌 컨텍스트와의 결합을 통해 더욱 향상된 unimodal sequence를 생성합니다. MoA는 서로 다른 모달리티 간의 시차를 조정하고, 대화 구조를 이해하기 위해 cross-modal graph를 사용하여 정교한 결과를 도출하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, IEMOCAP 및 CMU-MOSEI 데이터셋에서 제안한 모델이 기존 모델 대비 높은 weighted F1-score와 Accuracy를 기록하며 우수한 성능을 보였습니다. 특히, 이 방법은 감정의 종류와 관계없이 감정 인식의 정밀도를 일관되게 향상시켰고, HGF와 MoA의 기여도가 명확하게 나타났습니다. 이러한 성과는 다중 모달리티 학습에 대한 새로운 통찰을 제공하며, 향후 ERC 연구에 대한 발전 방향을 제시합니다.



### Toward a Safer Web: Multilingual Multi-Agent LLMs for Mitigating Adversarial Misinformation Attacks (https://arxiv.org/abs/2510.08605)
- **What's New**: 이 논문에서는 디지털 플랫폼에서의 허위 정보 전파를 방지하는 인공지능 기반 검출 시스템 구축을 제안합니다. 특히 여러 언어 간의 전환, 요약 시 문장 길이 증가, 선택형 질문 구조로의 변환 등 다양한 적대적 공격(transformation)을 분석하였습니다. 이러한 연구는 이전의 허위 정보 탐지 연구가 다루지 않았던 요소들을 포함하여, 실시간 웹 플러그인으로 적용할 수 있는 다국어, 다중 에이전트 언어 모델 프레임워크를 구성합니다.

- **Technical Details**: 논문에서 제안하는 시스템은 Retrieval-Augmented Generation(RAG) 방법론을 사용하여 다국어 정보를 처리합니다. 특히, Llama 모델을 기반으로 다양한 언어에서의 허위 정보 검출 성능을 실험하고, 적대적 공격에 대한 내성을 연구합니다. 또한, 정보 증거 제공을 통해 사용자에게 신뢰도 점수를 제공하는 실시간 도구 통합 방안을 논의합니다.

- **Performance Highlights**: 실험 결과, RAG-Llama 모델이 적대적 공격이 가해지더라도 다양한 언어에서 허위 정보를 효과적으로 식별할 수 있음을 보여주었습니다. 반면, 기존의 일반 LLM들은 유사한 상황에서 제한된 검출 능력을 보였습니다. 이 연구는 웹 기반 허위 정보 탐지 시스템의 필요성을 강조하며, 실제 웹 애플리케이션에서의 실제적인 배포 가능성을 보여줍니다.



### LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback (https://arxiv.org/abs/2510.08604)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLM)의 내장 안전 메커니즘을 우회하려는 다양한 공격, 즉 Jailbreak 공격을 다루고 있습니다. 기존의 자동화된 공격 방식은 모델이 고위험 응답을 생성하도록 유도하는 방법을 사용하지만, 본 연구에서는 이를 탐지할 수 있는 새로운 방법인 LatentBreak를 제안합니다. LatentBreak는 자연스러운 적대적 프롬프트를 생성하여 기존의 필터를 회피할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: LatentBreak(LatB)는 입력 프롬프트의 단어를 의미적으로 동등한 단어로 대체하여 고위험 프롬프트의 의도를 유지하는 방식으로 작동합니다. 이 과정은 잠재 공간(latent space) 피드백을 활용하여 이루어지며, 모델의 내부 표현을 해로운 콘텐츠와 관련된 지역으로 이동시키는 방식입니다. 따라서 LatB는 짧고 저-당혹감(low perplexity)의 프롬프트를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, LatentBreak는 여러 모델 패밀리에서 안전성 관련 모델들에 대한 공격을 성공적으로 수행하며, 기존의 족쇄 공격들과 비교했을 때 더욱 효과적으로 당혹감 기반 필터를 우회하는 것으로 나타났습니다. LatentBreak는 짧고 의미 있는 프롬프트를 생성하여, 고위험 응답을 유도하는 데 있어 탁월한 성능을 발휘했습니다. 이는 최신 방어 기법인 R2D2 및 Circuit Breakers와 같은 안전성 모델에 대한 공격에서도 마찬가지입니다.



### Mnemosyne: An Unsupervised, Human-Inspired Long-Term Memory Architecture for Edge-Based LLMs (https://arxiv.org/abs/2510.08601)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 인공지능 대화 시스템에서 인간처럼 자연스러운 소통을 가능하게 하는 장기 기억 시스템을 제안합니다. Mnemosyne은 비지도 학습 방식의 장기 기억 아키텍처로, 그래프 구조의 저장 기술을 사용합니다. 이 시스템은 사용자의 성격과 장기적인 목표를 효율적으로 캡처하는 '핵심 요약'을 도입하였으며, 기억의 확률적 회상 메커니즘을 통해 더 자연스러운 대화가 가능합니다.

- **Technical Details**: Mnemosyne은 그래프 구조 저장, 모듈화된 필터, 그리고 인간의 기억 모델을 따른 시간적 감쇠 및 새로 고침 절차를 포함합니다. 이 시스템은 장기 대화에서의 반복적이고 시맨틱적으로 유사하지만 시간적으로는 다른 대화를 처리하기 위해 설계되었습니다. 기억 관리와 회상 과정에서 감정적 맥락을 반영하기 위해 딥 메모리 모듈을 활용하며, 사용자의 질의와 관련된 역동적인 메커니즘을 발전시킵니다.

- **Performance Highlights**: 실험 결과, Mnemosyne은 현실성과 장기 기억 능력 테스트에서 65.8%의 승률을 기록하여 기존 RAG 기법의 31.1%를 능가했습니다. 또한, 현재 LoCoMo 벤치마크에서 시간 추론과 단일 홉 검색에서 최고의 점수를 달성했습니다. 이러한 결과는 Mnemosyne이 사실 회상 향상, 시간적 추론 개선, 그리고 보다 자연스러운 사용자 대응을 가능하게 한다는 것을 보여줍니다.



### Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation (https://arxiv.org/abs/2510.08600)
Comments:
          Accepted to EMNLP 2025 Industry Track

- **What's New**: 이번 논문에서는 Recover-LoRA라는 경량의 데이터셋 비독립적인 방법을 제안하여 기능 손상이 발생한 모델에서 정확도를 회복하는 접근법을 설명합니다. 특히, Recover-LoRA는 합성 데이터(synthetic data)와 로짓 증류(logit distillation)를 이용하여 저랭크 행렬(LoRA adapters)을 학습합니다. 이는 손상된 모델을 완전 정밀 모델에 맞추는 데 도움을 주며, 다양한 소형 언어 모델(SLM)에서의 유효성을 조사합니다.

- **Technical Details**: Recover-LoRA는 여러 주목 아키텍처가 포함된 소형 언어 모델(SLM)에서 작동하며, 모델의 손상된 가중치를 회복하기 위해 합성 데이터를 사용합니다. 이 방법은 로짓 증류를 통해 저랭크 행렬을 학습하여 손상이 발생한 모델과 기준 언어 모델을 정렬합니다. 또한, 로팩터화된 매트릭스 사용과 같은 매개변수 효율적인 방식으로 정확도를 회복하는 방법을 다룹니다.

- **Performance Highlights**: 실험 결과, Recover-LoRA는 MHA 및 GQA 모델에서 평균적으로 5%에서 17%의 정확도 회복을 보여주었습니다. 이는 LLM QAT 방법보다 우수한 회복 능력을 자랑하며, 네 개의 테스트 모델 중 세 개의 모델에서 데이터셋 특화된 LoRA 미세 조정(finetuning)보다 뛰어난 성능을 기록합니다. 이러한 성과는 Recover-LoRA의 효과적인 성능 회복 기법을 입증합니다.



### BaldWhisper: Faster Whisper with Head Shearing and Layer Merging (https://arxiv.org/abs/2510.08599)
- **What's New**: 이번 논문에서는 데이터가 부족한 조건에서 낮은 자원 언어를 위한 대형 프리트레인(Pre-trained) 트랜스포머 모델의 프루닝(Pruning) 방법을 제안합니다. 구체적으로, 발음 모델인 Whisper를 90%의 성능을 유지하면서 48% 작고 2.15배 빠른 모델로 변환할 수 있는 방법을 모색했습니다. 이 모델은 단 32시간의 음성 데이터를 기반으로 Bambara 언어에 맞추어 설계되었습니다.

- **Technical Details**: 제안된 두 단계 프루닝 접근법에서는 첫 번째로 레이어(여기서 층) 병합을 통해 모델의 성능 저하를 최소화하는 방법을 사용했습니다. 이후, 공유되는 입력 및 출력 임베딩을 저차원 분해(low-rank decomposition)와 피쳐 증류(feature distillation)를 통해 추가로 압축하여 성능을 극대화했습니다. 기존의 단어 집합 프루닝(vocabulary pruning) 방식을 대신하여, 코드 스위칭 상황에도 적합한 방식을 채택했습니다.

- **Performance Highlights**: 최종 모델은 Whisper 모델에 비해 48% 더 작아지고 2.15배 더 빠르며, 성능 또한 90% 이상 유지합니다. 이러한 개선은 저자원 환경에서의 음성 인식 모델의 효율성을 높이는 데 기여할 것으로 기대됩니다. 이 연구는 모바일 기기 등에서의 오프라인 배치에 최적화된 접근법을 제공하여 실용적입니다.



### Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech (https://arxiv.org/abs/2510.08593)
- **What's New**: 이 논문에서는 HAREN-CTC라는 새로운 아키텍처를 제안합니다. 이 모델은 다층 Self-Supervised Learning (SSL) 기능을 통합하여 우울증 감지의 정확성을 높입니다. 기존의 방법들이 최종 레이어에만 의존하거나 단일 best-performing 레이어를 찾는 데 그치는 것에 반해, HAREN-CTC는 여러 레이어의 구조를 활용합니다. 이를 통해 보다 정교하고 지속적인 우울증 신호를 탐지할 수 있습니다.

- **Technical Details**: HAREN-CTC는 두 개의 주요 모듈로 구성되어 있습니다: Hierarchical Adaptive Clustering (HAC) 모듈은 SSL 기능을 보완적인 임베딩으로 재구성하고, Cross-Modal Fusion (CMF) 모듈은 다중 헤드 교차 주의 메커니즘을 통해 레이어 간의 의존성을 모델링합니다. Connectionist Temporal Classification (CTC) 손실을 사용하여 희소한 시간적 감독을 처리함으로써 우울증 관련 음성 신호의 불규칙한 시간 패턴을 추적할 수 있습니다.

- **Performance Highlights**: HAREN-CTC는 DAIC-WOZ에서 0.81, MODMA에서 0.82의 최첨단 macro F1-score를 달성하며 이전 방법들을 초월한 성과를 보입니다. 다양한 평가 시나리오에서 높은 내구성과 정확성을 보여줍니다. 또한, 다층 SSL 기능을 효과적으로 활용하여 일반화 능력을 향상시키는 것이 주요 강점입니다.



### Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models (https://arxiv.org/abs/2510.08592)
- **What's New**: 이 연구는 Test-Time Scaling (TTS) 방법이 후보 응답의 다양성 감소로 인해 발생하는 새로운 실패 모드를 발견하였습니다. 후보 응답 다양성이 충분할 때 TTS의 신뢰성이 높아지지만, 다양성이 제한될 경우 안전하지 않은 결과를 초래할 위험이 증가한다는 것을 증명했습니다. 이를 평가하기 위한 RefDiv 프로토콜을 제안하여 TTS의 취약성을 평가하고 개선점을 제시합니다.

- **Technical Details**: TTS는 LLM의 추론 과정에서 다양한 후보 응답을 생성하고 평가함으로써 출력 품질을 향상시키는 방법입니다. 다양한 데이터 샘플을 통해 해결방안을 더 효과적으로 탐색할 수 있지만, 후보 응답의 다양성을 조절하면 시스템의 안전성에 부정적인 영향을 미칠 수 있습니다. 이 연구는 MCTS 및 Best-of-N과 같은 다양한 TTS 전략을 통해 이러한 다양성 제약이 TTS의 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: RefDiv를 통해 후보 응답의 다양성이 줄어들 때 발생하는 안전성 저하를 관찰했습니다. 공격에 사용된 문자열은 다양한 TTS 전략과 모델에 성공적으로 전이되었으며, 기존의 안전성 분류기가 이러한 입력을 탐지하지 못하는 경우가 많았음을 보여주었습니다. 이 연구는 TTS 기반 LLM의 안전성을 개선하기 위한 미래 연구에 대한 필요성을 강조합니다.



### The Enduring Dominance of Deep Neural Networks: A Critical Analysis of the Fundamental Limitations of Quantum Machine Learning and Spiking Neural Networks (https://arxiv.org/abs/2510.08591)
- **What's New**: 최근 QML(Quantum Machine Learning)과 SNNs(Spiking Neural Networks)의 발전은 AI를 혁신할 수 있는 큰 가능성을 제시하고 있지만, 본 논문은 이러한 기술들이 단기적으로 DNNs(Deep Neural Networks)를 대체할 것으로 예상하지 않는다.

- **Technical Details**: QML은 단일 제약(unitary constraints), 측정으로 인한 상태 붕괴(measurement-induced state collapse), 배런 평면(barren plateaus), 고측정 오버헤드(high measurement overheads) 등의 문제로 역전파(backpropagation) 적응에 어려움을 겪고 있다. SNNs는 이산형 스파이크 기반 처리 방식으로 인해 언어 작업에서 장거리 의존성(long-range dependencies)과 의미 표현(semantic encoding)에 제약을 받고 있다.

- **Performance Highlights**: 최적화된 DNN은 양자화(quantization)를 통해 실제 조건에서 SNN보다 더 낮은 에너지 비용으로 성능을 발휘할 수 있다. 또한, DNN은 효율적인 역전파와 강력한 정규화(regularization)를 이용하여 자율적 개선(self-improvement)을 가능하게 하며, 최신 모델인 xAI의 Grok-4 Heavy는 SOTA(State of the Art) 성능을 기록하고 있다.



### EGSTalker: Real-Time Audio-Driven Talking Head Generation with Efficient Gaussian Deformation (https://arxiv.org/abs/2510.08587)
Comments:
          Main paper (6 pages). Accepted for publication by IEEE International Conference on Systems, Man, and Cybernetics 2025

- **What's New**: 이 논문에서는 EGSTalker라는 실시간 오디오 기반 말하는 얼굴 생성 프레임워크를 소개합니다. EGSTalker는 3D Gaussian Splatting(3DGS)을 기반으로 하여 빠른 속도와 높은 시각적 충실도를 제공합니다. 고품질 얼굴 애니메이션을 합성하는 데 3-5분의 교육 비디오만 필요하며, 정적 Gaussian 초기화 단계와 오디오 기반 변형 단계로 구성됩니다.

- **Technical Details**: EGSTalker는 다중 해상도 해시 트리플레인과 Kolmogorov-Arnold Network(KAN)를 사용하여 정적 3D Gaussian 표현을 구축합니다. 이를 통해 헤드 구조와 얼굴 표정을 모델링하며, Efficient Spatial-Audio Attention(ESAA) 모듈을 통해 오디오와 공간 정보를 융합합니다. 이 프레임워크는 간단한 MLP 기반 방법론 대신 KAN을 사용하여 이루어지는 복잡한 비선형 매핑을 효과적으로 처리합니다.

- **Performance Highlights**: 실험 결과 EGSTalker는 출력 품질 및 입술 동기화 정확도가 최신 방법들과 비교해 상대적으로 우수한 성능을 보입니다. 또한 추론 속도에서 현저하게 눈에 띄는 성능 개선을 이뤄내며, 이는 실시간 멀티미디어 어플리케이션에서의 큰 잠재력을 강조합니다.



### Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech (https://arxiv.org/abs/2510.08586)
Comments:
          Accepted at IEEE CogMI 2025

- **What's New**: 이 논문은 음성을 통해 심리적 스트레스를 탐지하는 새로운 접근 방식을 제안합니다. 기존의 연구들은 스트레스를 고정된 레이블로 다루었으나, 본 연구에서는 스트레스를 시간에 따라 변화하는 현상으로 모델링합니다. 동적 레이블링을 통해 감정 상태에서 세밀한 스트레스 주석을 유도하고, 이로써 LSTM 및 Transformer Encoder와 같은 순차 모델을 도입하여 시간에 따른 스트레스 발전을 포착합니다.

- **Technical Details**: 제안된 방법론은 스트레스 진단을 위해 Stress Progression Labelling Framework와 Temporal Stress Classification Models를 포함합니다. 스트레스는 바로 직전의 정서 상태와 밀접하게 연관되어 있으며, 이를 기반으로 LSTM과 Transformer 아키텍처를 사용하여 스트레스를 이진 분류합니다. 이 과정에서 교차 주의 메커니즘을 활용하여 음성 특징과 스트레스 상태 간의 순차적 의존성을 캡처합니다.

- **Performance Highlights**: 우리의 접근 방식은 MuSE와 StressID 데이터셋에서 기존 최 기준 대비 각각 5% 및 18%의 정확도 향상을 달성했습니다. 또한, 커스텀 데이터셋에서도 유의미한 일반화를 보여 스트레스의 동적 모델링 가치가 강조되었습니다. 이러한 결과는 음성을 통한 스트레스 탐지가 정확하고 실용적일 수 있음을 증명했습니다.



### Articulation-Informed ASR: Integrating Articulatory Features into ASR via Auxiliary Speech Inversion and Cross-Attention Fusion (https://arxiv.org/abs/2510.08585)
- **What's New**: 이 연구는 발화 특성(articulatory features)을 현대의 딥 러닝 모델에서 사용하는 새로운 접근 방식을 제안하고 있습니다. 기존 연구에서는 발화 특성이 주로 얕은 음향 모델에 한정되었으나, 본 연구에서는 이를 심화 학습 구조에서 주요 입력의 보조 작업으로 활용하고 있습니다. 특히, Speech Inversion(SI)을 보조 예측 작업으로 사용하고, 이 예측된 발화 특성을 음향 임베딩과 함께 크로스-어텐션 모듈 내에서 주입합니다. 실험 결과, 이 방법은 신뢰할 수 있는 Transformer 기반 baseline 모델에 비해 일관된 성능 개선을 보였습니다.

- **Technical Details**: 제안된 Articulation-Informed ASR 프레임워크는 두 가지 전략을 통해 발화 정보를 인식 모델에 통합합니다. 첫째, SI 보조 작업을 통해 발화 인지가 가능한 음향 표현을 유도하며, 둘째, 크로스-어텐션 블록을 추가해 예측된 발화 궤적을 CTC 디코딩 전 추가 입력 스트림으로 주입합니다. 이 모델은 Wav2vec2.0 기반으로 설계되었으며, 마찰 손실(Mean Absolute Error, MAE)과 CTC 손실을 결합한 방식으로 학습됩니다. 또한, 불확실성 기반 가중치(UBW)를 활용하여 손실 함수를 동적으로 조정합니다.

- **Performance Highlights**: LibriSpeech 데이터셋을 활용한 실험에서, 다양한 훈련 데이터 크기에서 성능을 평가하였고 발화 특성이 적은 감독 하에서도 효과를 보였음을 확인했습니다. 특히, 10분, 1시간, 10시간, 100시간의 레이블이 있는 훈련 데이터에 대해 설명된 방법은 일관된 성능 향상을 보였습니다. 발화 특성을 통합한 모델은 저자원 환경에서도 기존 모델들보다 뛰어난 성과를 기록했습니다.



### Evaluating Hallucinations in Multimodal LLMs with Spoken Queries under Diverse Acoustic Conditions (https://arxiv.org/abs/2510.08581)
- **What's New**: 이 논문은 이미지-텍스트 환경에서의 신뢰성에 대한 벤치마크를 이용해 연구된 시각-언어 모델의 환각(hallucination) 현상에 대해 다루고 있습니다. 그러나 음성(query) 입력이 다중 모달(multimodal) 환각에 미치는 영향은 거의 탐구되지 않았습니다. 이에 따라 우리는 RePOPE-Spk라는 음성 기반의 RePOPE 벤치마크 확장을 제시하고, 음성이 다양한 음향(acoustic) 조건 하에서 쿼리(query)로 제공되는 방식을 연구합니다.

- **Technical Details**: RePOPE-Spk를 통해 우리는 독점적(proprietary) 및 오픈 소스(open-source) 모델에 대한 체계적인 평가를 수행했습니다. 실험 결과에 따르면, 음성으로 쿼리를 제공할 경우 환각 현상이 증가하며, 청정(지저분하지 않은) 음성 하에서는 3%의 오차율이 증가하고, 환경 소음이 있는 경우 최대 20%까지 증가했습니다. 입력 순서(input order)와 쿼리 길이(query length)도 모델의 강인성(robustness)에 영향을 미치는 것으로 나타났습니다.

- **Performance Highlights**: 여러 번의 프롬프트(many-shot prompting)와 연쇄적 사고(chain-of-thought reasoning) 전략이 부분적으로는 완화 효과를 보였지만, 충분하지 않았습니다. 이러한 연구 결과는 음성 인터페이스 시스템 구축에 있어 신뢰性 확보의 중요성과 새로운 방향의 필요성을 강조하고 있습니다. 이는 앞으로의 연구에 있어 다중 모달 시스템의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection (https://arxiv.org/abs/2510.08580)
Comments:
          Under Submission

- **What's New**: 이 논문은 음악 오류 탐지를 위한 새로운 Transformer 기반 방법인 LadderSym을 도입합니다. 기존의 방법들은 보통 오디오 녹음을 음악 악보와 비교하는 방식으로, 오류 탐지의 정확성을 높이기 위해 획기적인 접근 방식이 필요합니다. LadderSym은 두 가지 핵심 관찰에 기반하여 설계되어, 서로 다른 스트림 간의 정렬 및 비교 능력을 개선합니다.

- **Technical Details**: LadderSym은 두 개의 스트림 인코더와 스트림 간 정렬 모듈을 사용하는 구조로, 오디오 비교 능력과 오류 탐지 F1 점수를 개선합니다. 또한 기호적 표현을 디코더 프롬프트로 활용하는 다중 모달 전략을 통해 음성과 기호적 악보를 모두 활용합니다. 이 방식은 음악에서 발생하는 주파수 스펙트럼의 모호성을 줄이고 F1 점수를 향상시킵니다.

- **Performance Highlights**: LadderSym은 MAESTRO-E 및 CocoChorales-E 데이터 세트에서 최첨단 성능을 달성했습니다. MAESTRO-E에서 놓친 음에 대한 F1 점수를 26.8%에서 56.3%로 두 배 이상 증가시키고, 추가 음 탐지 능력을 14.4 포인트 향상시켰습니다. CocoChorales-E에서도 유사한 성과를 보여줍니다.



### AgenticAD: A Specialized Multiagent System Framework for Holistic Alzheimer Disease Managemen (https://arxiv.org/abs/2510.08578)
- **What's New**: 이 논문은 알츠하이머병(Alzheimer's disease, AD) 관리의 통합적 접근을 위한 새로운 방법론적 프레임워크를 제안합니다. 기존의 인공지능(AI) 적용 사례는 주로 진단이나 간병 지원과 같은 단일 측면에 국한되어 있었으나, 이번 연구에서는 이를 넘어다섯 가지의 전문 AI 에이전트 시스템을 도입하였습니다. 각 에이전트는 AD 관리의 다양한 도전을 해결하기 위해 설계되었습니다.

- **Technical Details**: 이 프레임워크는 여덟 개의 전문화된 상호운용 가능한 에이전트로 구성되어 있으며, 기능에 따라 (1) 간병인 및 환자 지원, (2) 데이터 분석 및 연구, (3) 고급 다중모드 워크플로우로 카테고리화됩니다. 각 에이전트는 GPT-4o 및 Gemini 같은 대형 언어 모델(large language models, LLMs), 다중 에이전트 오케스트레이션 프레임워크, 에vidence-augmented generation (RAG) 기술을 활용하여 고급 기술 아키텍처를 갖추고 있습니다.

- **Performance Highlights**: 이 통합 AI 생태계는 단일 목적의 도구를 넘어, 협력적이고 다중 에이전트 패러다임으로 전환하여 더 적응적이고 개인 맞춤화된 솔루션 개발의 기초를 마련합니다. 앞으로의 시스템은 다양한 데이터 스트림을 통합하여 환자 결과를 개선하고 간병인의 부담을 줄일 수 있도록 되어 있습니다. 이 혁신적인 방법론적 접근은 AD 관리의 질을 높이는 방향으로 나아가고자 합니다.



### Comparative Analysis of Large Language Models for the Machine-Assisted Resolution of User Intentions (https://arxiv.org/abs/2510.08576)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 사용자 의도를 해결하고 복잡한 작업 흐름을 오케스트레이션하는 데 미치는 영향에 대해 다룹니다. 기존의 클라우드 기반 모델의 제한사항을 극복하기 위해, 로컬 배치가 반드시 필요하다는 점을 강조합니다. 연구는 공개 소스 모델들이 자율적이며 로컬에서 작동할 수 있는 미래 운영 체제의 설계에 어떻게 기여할 수 있는지를 탐구합니다.

- **Technical Details**: 사용자 의도를 실행 가능한 작업 흐름으로 변환하는 과정은 LLM 기반 시스템의 발전에 있어 필수적입니다. 연구에서는 LLM을 통해 사용자의 고차원적인 의도를 해석하고, 이와 관련된 API의 기능을 이해하여 올바른 코드 생성이 이루어지도록 하는 인터페이스를 개발합니다. 시스템 아키텍처는 중앙 조정 단위인 Controller 애플리케이션을 포함하며, LLM과의 통신을 관리하여 모델이 생성한 작업 흐름을 실시간 환경에서 실행할 수 있게 합니다.

- **Performance Highlights**: 비교 분석을 통해 공개 소스 모델들의 실행 성능을 검토하고 OpenAI의 GPT-4 기반 시스템과의 성능 차이를 분석합니다. 각 모델의 응답 시간 및 첫 번째 토큰 생성 시간을 측정하여 사용자 경험이 방해받지 않도록 하였으며, 이들 모델이 복잡한 사용자 의도를 해결하는 데 얼마나 효과적일 수 있는지를 평가하고 있습니다. 이 연구는 AI 인프라의 분산화와 민주화에 대한 중요한 통찰을 제시합니다.



### PyNoetic: A modular python framework for no-code development of EEG brain-computer interfaces (https://arxiv.org/abs/2509.00670)
Comments:
          PLoS One 2025. Project Website: this https URL

- **What's New**: PyNoetic은 모듈형( modular) BCI 프레임워크로, EEG 기반의 Brain-Computer Interfaces(BCI) 연구의 다양한 필요를 충족하기 위해 설계되었습니다. 기존의 BCI 프레임워크들이 가지고 있는 유연성 부족, 학습 곡선(learning curve) 문제, 높은 비용, 다양한 외부 도구 사용의 필요성을 해결하기 위해 이를 제시합니다. 특히, 사용자가 손쉽게 접근할 수 있는 직관적(UI)이고 사용자 친화적인 GUI를 제공하며, 코드 없이 BCI를 설계할 수 있는 기능을 포함하고 있습니다.

- **Technical Details**: PyNoetic은 자극 제공(stimulus presentation), 데이터 수집(data acquisition), 채널 선택(channel selection), 필터링(filtering), 특성 추출(feature extraction), 아티팩트 제거(artifact removal), 시뮬레이션(simulation) 및 시각화(visualization)를 포함한 전체 BCI 설계 파이프라인을 포괄하는 몇 안 되는 Python 프레임워크 중 하나입니다. 또한, 최소한의 코딩으로 사용자 정의 기능(custom functionalities) 및 새로운 알고리즘의 통합(integration)을 쉽게 지원하여 각 설계 단계에서의 적응성을 보장합니다. PyNoetic은 머신러닝 모델(machine learning models), 뇌 연결 지수(brain-connectivity indices), 시뮬레이션을 통한 체계적 테스트 기능(systematic testing functionalities)과 새로운 패러다임 평가 방법(evaluation methods of novel paradigms) 등을 포함한 풍부한 분석 도구를 제공합니다.

- **Performance Highlights**: PyNoetic의 강점은 오프라인(offline) 및 실시간(real-time) BCI 개발 모두에서의 다재다능함(versatility)에 있습니다. 이는 연구자들이 BCI 개발의 복잡한 측면에 초점을 맞출 수 있도록 설계 과정을 간소화하여, 연구 진전을 가속화할 수 있도록 도와줍니다. 연구자들에게 친숙한 접근성을 제공함으로써 BCI 연구의 효율성을 크게 향상시키는 점이 주목받고 있습니다.



### Deep Sparse Representation-based Classification (https://arxiv.org/abs/1904.11093)
- **What's New**: 본 논문에서는 Sparse Representation-based Classification (SRC) 방법을 위한 전이적(transductive) 딥러닝 기반의 새로운 접근법을 제안합니다. 제안된 네트워크는 컨볼루션 오토인코더(convolutional autoencoder)와 완전 연결층(fully-connected layer)으로 구성됩니다. 이 오토인코더 네트워크는 분류를 위한 견고한 심층 특징(deep features)을 학습하는 역할을 수행합니다.

- **Technical Details**: 오토인코더와 함께 위치한 완전 연결층은 희소 표현(sparse representation)을 찾는 역할을 맡고 있습니다. 이들은 자동적으로 생성된 희소 코드(sparse codes)를 이용하여 분류를 수행합니다. 세 가지 다른 데이터셋을 활용한 여러 실험을 통해, 제안된 네트워크가 최신 SRC 방법론보다 더 뛰어난 분류 성능을 달성함을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋에 대해 실험 결과를 통해 더욱 우수한 희소 표현을 생성하고, 이는 기존 최첨단 SRC 방법과 비교할 때 더 나은 분류 성능을 제공합니다. 이러한 혁신적인 접근은 딥러닝 모델의 가능성을 더욱 확대하는데 기여할 것으로 기대됩니다.



### Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition with Multimodal Training (https://arxiv.org/abs/1812.06145)
- **What's New**: 본 논문에서는 다중 모달리티(multiple modalities)의 지식을 활용하여 단일 모달 3D 합성곱 신경망(3D-CNNs)의 동적 손 제스처 인식을 위한 효율적인 접근 방법을 제시합니다. 기존의 많은 최신 방법들처럼 다중 모달 정보를 명시적으로 결합하는 대신, 각 단일 모달 네트워크가 개선된 성능을 이끌어낼 수 있도록 여러 모달의 지식을 개별 네트워크에 삽입하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 각 가용 모달에 대해 별도의 네트워크를 할당하고, 이를 협력하여 공통의 의미(semantic)를 가진 더 나은 표현(representations)을 개발하도록 강제합니다. 우리는 서로 다른 네트워크의 특성(content) 정렬을 위한 'spatiotemporal semantic alignment' 손실(SSA)을 도입하며, 이 손실을 부정적 지식 전이를 피하기 위해 제안한 'focal regularization parameter'로 정규화합니다.

- **Performance Highlights**: 실험 결과, 우리의 프레임워크는 단일 모달 네트워크의 테스트 시간 인식 정확도를 향상시키며, 다양한 동적 손 제스처 인식 데이터셋에서 최신 성능을 제공합니다.



### Deep Multimodal Subspace Clustering Networks (https://arxiv.org/abs/1804.06498)
- **What's New**: 본 논문은 비지도 멀티모달 서브스페이스 클러스터링을 위한 CNN(Convolutional Neural Network) 기반 접근법을 제시합니다. 이 프레임워크는 멀티모달 인코더, 자기 표현 레이어, 멀티모달 디코더라는 세 가지 주요 단계로 구성됩니다. 특히, 다양한 데이터 융합 기법을 탐구하여 공간 융합에 해당하는 세 가지 인코더를 제안합니다.

- **Technical Details**: 인코더는 멀티모달 데이터를 입력받아 잠재 공간(latent space) 표현으로 융합합니다. 자기 표현 레이어는 자기 표현 속성(self-expressiveness property)을 강화하고 데이터 포인트에 해당하는 affinity matrix를 획득하는 역할을 합니다. 디코더는 원래 입력 데이터를 재구성하는 기능을 하며, 네트워크는 훈련 과정에서 디코더 재구성과 원래 입력 간의 거리를 사용합니다.

- **Performance Highlights**: 세 가지 데이터 세트에 대한 광범위한 실험 결과, 제안된 방법이 최신 멀티모달 서브스페이스 클러스터링 방법들을 현저하게 능가함을 보여줍니다. 특히, 공간 융합 기반의 다양한 방법뿐만 아니라, 다양한 모달리티에 대해 자기 표현 레이어를 동일하게 적용한 affinity fusion 기반 네트워크도 제안되었습니다.




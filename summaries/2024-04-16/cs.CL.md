### Memory Sharing for Large Language Model based Agents (https://arxiv.org/abs/2404.09982)
- **What's New**: 이 연구에서는 인-컨텍스트 학습(in-context learning)의 한계를 극복하고자 대규모 언어 모델(Large Language Model, LLM) 기반 멀티에이전트 시스템을 위한 메모리 공유(Memory Sharing, MS) 프레임워크를 소개합니다. 이 프레임워크는 실시간 메모리 저장 및 검색 시스템을 사용하여 에이전트의 학습 과정을 향상시키고, 복잡한 문제에 대한 대응 능력을 제고합니다.

- **Technical Details**: MS 프레임워크는 각 에이전트가 생성한 프롬프트-응답(Prompt-Answer, PA) 쌍을 메모리 풀에 저장하고 다른 에이전트와 공유함으로써 지식 기반을 확장하고 역동성을 증가시킵니다. 저장 단계에서는 LLM 평가자가 PA 쌍의 적절성을 평가하고, 검색 단계에서는 자율 학습 검색 시스템이 적절한 메모리를 프롬프트에 통합하여 에이전트가 질문의 본질을 더욱 잘 이해하도록 돕습니다. 이러한 과정을 통해 에이전트는 지속적으로 새로운 메모리를 풀에 추가하고 검색자를 개선할 수 있습니다.

- **Performance Highlights**: 세 가지 다른 도메인에서의 실험 검증을 통해 MS 프레임워크는 에이전트의 정밀도와 관련성을 향상시키는 것으로 나타났습니다. 특히, 개방형 질문에 대해 훨씬 더 일치하는 결과를 생성하는 데 도움을 주었습니다. 메모리 풀과 검색 전략의 적절한 통합은 에이전트의 성능을 크게 향상시킬 수 있는 주요 요인임을 확인했습니다.



### Context Does Matter: Implications for Crowdsourced Evaluation Labels in  Task-Oriented Dialogue Systems (https://arxiv.org/abs/2404.09980)
Comments: Accepted at NAACL 2024 Findings

- **What's New**: 이 연구는 대화 맥락이 어노테이터(annotators)의 품질 평가에 미치는 영향을 조사합니다. 특히, 대화 맥락을 요약하여 제공하는 새로운 방법과 대화 맥락의 양이 어노테이션의 일관성에 어떻게 영향을 미치는지를 실험적으로 검토합니다. 또한, LLM(Large Language Models)을 활용하여 대화 맥락을 요약하고 이가 어노테이터의 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구에서는 대화의 맥락을 무작위로 샘플링하고, 그 양을 조절하여 어노테이션 작업의 설계가 평가의 품질과 일관성에 미치는 영향을 조사합니다. LLM, 특히 GPT-4를 사용하여 대화 맥락을 요약하고, 이 요약이 어노테이터의 판단에 어떤 영향을 미치는지 분석하였습니다. 이를 통해 대화 응답의 관련성(relevance)과 유용성(usefulness)을 평가하는 데 필요한 최적의 맥락 크기와 정보의 질을 탐구합니다.

- **Performance Highlights**: 제한된 맥락을 제공할 때 어노테이터는 더 긍정적인 평가를 할 가능성이 높아지는 반면, 전체 대화 맥락을 제공하면 관련성 평가의 품질이 향상됩니다. 하지만, 전체 맥락 제공은 유용성 평가에서의 모호함을 증가시키고 동의도를 약간 저하시킵니다. 자동으로 생성된 대화 맥락이 제공되었을 때 (C0 조건에서), 어노테이터 간의 일관성은 향상되며, 전체 맥락 (C7 조건)을 제공할 때보다 어노테이션 시간이 단축되었습니다.



### Constructing Benchmarks and Interventions for Combating Hallucinations  in LLMs (https://arxiv.org/abs/2404.09971)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 환각 현상을 완화하기 위해 모델의 계산 중 개입하는 새로운 접근 방식을 제안합니다. 특히, 모델의 내부 지식에 근거한 데이터셋을 구축하고, 특정 컴포넌트(MLPs, attention block, residual stream 및 특정 heads)에 대한 개입을 통해 환각 방지를 목적으로 합니다. 또한, 이 연구는 환각 발생 전과 후의 개입 방법을 비교하여 환각 발생 전 개입이 더 효과적임을 발견하였습니다.

- **Technical Details**: 이 연구는 closed-book과 open-book 질의응답(question-answering, QA) 설정에서 모델 기반 벤치마크를 구축합니다. 연구자들은 모델의 계산에서 특정 레이어(layer)와 컴포넌트(component)에 개입하여 환각을 완화하고자 하는 방법인 'whitebox approaches'에 초점을 맞추었습니다. 개입은 주로 계산 성분의 활성화 변형을 통해 이루어지며, 이러한 개입은 정해진 가이드라인에 따라 실행됩니다. 새로운 동적 개입 방법(dynamic intervention)도 도입되어 필요한 경우에만 개입하도록 하였습니다. 이 연구는 기존 방법보다 더 정밀하고 효율적인 환각 제어 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 특정 컴포넌트에 개입할 때 성능에 차이가 있었으며, attention 컴포넌트에 개입하는 것이 MLP 파트보다 선호되었습니다. 또한, residual에 대한 개입은 복잡도를 높이지만, 동적 개입을 적용할 때 이 문제가 다소 완화되었습니다. 또한, 환각 발생 전 내부 상태를 사용하는 것이 환각 발생 후 사용하는 것보다 효과적임을 발견하였습니다.



### Compression Represents Intelligence Linearly (https://arxiv.org/abs/2404.09937)
Comments: Preprint. Data and code are available at this https URL

- **What's New**: 이 연구는 대규모 언어 모델(LLMs: Large Language Models)의 압축 능력과 지능 사이의 관계를 실증적으로 탐구했습니다. 연구진은 LLMs를 데이터 압축기로 취급하고, 이 모델들이 외부 텍스트 코퍼스를 얼마나 효율적으로 압축하는지를 평가했습니다. 그 결과, 모델들의 다운스트림 벤치마크 점수와 압축 능력 간에 거의 선형적인 상관 관계를 발견했습니다.

- **Technical Details**: 연구팀은 지식과 상식, 코딩, 수학적 추론 등 세 가지 주요 능력을 중심으로 LLMs의 '지능'을 측정했습니다. 각 도메인에서 외부 원시 코퍼스를 수집한 다음, 다양한 LLMs가 이 코퍼스를 압축하는 효율성을 평가했습니다. 여기에는 30개의 공개 LLM과 12개의 다양한 벤치마크가 포함되었습니다. 연구 결과, 압축 효율성과 모델 능력 사이에는 -0.95의 피어슨 상관 계수(Pearson correlation coefficient)로 거의 선형적인 상관 관계가 나타났습니다.

- **Performance Highlights**: 압축 효율성은 감독되지 않은 메트릭(Unsupervised Metric)으로서, 텍스트 코퍼스를 쉽게 갱신하여 오버피팅이나 테스트 오염을 피할 수 있습니다. 또한, 연구 결과에 따르면, 압축 효율성은 LLMs의 능력을 평가하는 안정적이고 유연하며 신뢰할 수 있는 메트릭으로 사용될 수 있습니다.



### ChatShop: Interactive Information Seeking with Language Agents (https://arxiv.org/abs/2404.09911)
- **What's New**: 이 연구에서는 쇼핑 작업을 새로운 방법으로 조정하여 언어 에이전트의 정보 탐색 능력을 평가합니다. 챗샵(ChatShop)이라는 새로운 작업을 제안하여, 에이전트가 사용자의 선호도를 개방형 대화를 통해 탐색하고 정보를 점진적으로 축적하여 합리적인 결정을 내릴 수 있습니다. 쇼핑 작업이 기존에 한 번의 정보 검색으로 해결되는 것과는 달리, 다단계 상호작용을 요구합니다.

- **Technical Details**: 이 챗샵 작업에서 에이전트는 사용자로부터 제품의 대략적인 유형만을 알고 시작하며, BM25(비엠25) 검색 엔진 및 질문-응답을 통해 추가 정보를 얻어야 합니다. 베이스 모델로는 GPT-3.5와 라마 2(variant of Llama)가 사용되었습니다. 챗샵 작업은 웹샵(WebShop) 작업을 기반으로 하되, 목표 지시 사항을 단순화하여 에이전트가 적극적으로 정보를 발견하도록 유도합니다.

- **Performance Highlights**: 실험 결과 쇼핑 작업의 복잡성에 대한 에이전트의 이해를 성공적으로 평가할 수 있음을 보여줍니다. 또한, LLM을 사용하여 인간 쇼퍼를 시뮬레이션한 환경은 실제 인간 쇼퍼와 상호작용하는 환경만큼 효과적이며 에이전트의 오류 패턴을 발견하는 데 유용합니다. OpenAI의 GPT-3.5를 사용한 시뮬레이션 쇼퍼는 제품의 타이틀, 필요한 속성 및 옵션을 제공받고 에이전트의 질문에 짧게 응답하도록 설정되었습니다.



### Glitch Tokens in Large Language Models: Categorization Taxonomy and  Effective Detection (https://arxiv.org/abs/2404.09894)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 토크나이저에서 발생하는 이상 토큰인 '글리치 토큰(glitch tokens)' 현상을 처음으로 소개하고 체계적으로 탐구하였습니다. 여러 LLM에서 글리치 토큰이 모델의 응답 품질을 손상시킬 수 있는 잠재적인 위험을 가지고 있음을 밝혔습니다.

- **Technical Details**: 연구팀은 세 가지 다른 토크나이저(tokenizers)를 사용하고 총 182,517개의 토큰을 포함하는 일곱 가지 인기 있는 LLM을 실험하여 글리치 토큰을 분류하고 LLM이 글리치 토큰과 상호작용할 때 나타나는 증상을 제시했습니다. 글리치 토큰이 임베딩 공간(embedding space)에서 군집하는 경향이 있다는 관찰에 기반하여, 효율적인 글리치 토큰 감지를 위한 새로운 순환 군집 기반 기술인 '글리치헌터(GlitchHunter)'를 제안했습니다.

- **Performance Highlights**: 평가 결과, 제안한 '글리치헌터(GlitchHunter)' 방법이 세 가지 기준 방법(baseline methods)보다 여덟 가지 오픈 소스 LLM에서 현저하게 우수한 성능을 보였습니다. 이 연구는 글리치 토큰에 대한 최초의 포괄적 연구로, LLM에서 토크나이제이션 관련 오류를 완화하는 데에 중요한 통찰을 제공합니다.



### Negation Triplet Extraction with Syntactic Dependency and Semantic  Consistency (https://arxiv.org/abs/2404.09830)
Comments: Accepted by COLING 2024

- **What's New**: 이 논문에서는 부정 객체를 파악하는 것의 중요성에 주목하며 새로운 부정 트리플 추출(Negation Triplet Extraction, NTE) 작업을 제안합니다. 기존의 연구들이 주로 부정 신호(negation cue) 탐지와 범위(scope) 해결에 중점을 두었다면, 이 연구에서는 부정 주체(negation subject)를 함께 추출하는 것을 목표로 합니다. 이를 위해, 문법 구조와 의미론을 강화한 새로운 모델인 SSENE(Syntax&Semantic-Enhanced Negation Extraction)를 소개합니다.

- **Technical Details**: SSENE 모델은 Encoder-Decoder 구조를 기반으로 하여, 문장의 구문 의존성 트리(syntactic dependency tree)를 이용하여 부정 주체, 신호, 범위 사이의 연관성을 파악합니다. 또한, 다중 작업 학습 프레임워크(multi-task learning framework)를 사용하여 문장과 추출된 트리플 사이의 의미적 일관성(semantic consistency)을 보장합니다. 이 모델은 중국어 데이터셋인 NegComment를 사용하여 평가되었습니다.

- **Performance Highlights**: SSENE은 Meituan 사용자 리뷰를 기반으로 한 NegComment 데이터셋에서 최고의 NTE 성능을 달성했습니다. 부정 주체, 신호, 범위를 정확하게 식별함으로써, 이전 모델들과 비교했을 때 성능이 개선되었습니다. NTE 작업에 대한 실험을 통해 구문 정보의 통합이 주제와 신호 간의 먼 의존성을 인식하는데 도움이 됨을 입증했으며, 부정 트리플 추출에 대한 의미론적 일관성을 높이는데도 효과적임을 보여주었습니다.



### Impact of Preference Noise on the Alignment Performance of Generative  Language Models (https://arxiv.org/abs/2404.09824)
- **What's New**: 이 논문에서는 GLM(Generative Language Models)을 인간의 가치와 일치시키기 위한 새로운 프레임워크를 제안하며, 선호도 노이즈(Preference Noise)가 정렬 성능에 미치는 영향을 체계적으로 연구합니다. 특히 요약(summarization)과 대화 생성(dialogue generation) 두 가지 과제에서 선호도 노이즈가 정렬 성능에 미치는 영향을 실험적으로 분석하였습니다.

- **Technical Details**: 이 연구에서는 인간 평가자나 훈련된 보상 모델(Reward Models, RM)로부터 이진 선호도(Binary Preferences)를 수집하고, 이를 바탕으로 GLM을 조정하는 방법을 사용합니다. 사용된 기술로는 직접 선호 최적화(Direct Preference Optimization, DPO) 및 시퀀스 가능성 보정(Sequence Likelihood Calibration, SLiC) 등이 있습니다. 또한, 노이즈가 포함된 선호도 데이터에서의 신뢰 기반 데이터 필터링(Confidence-Based Data Filtering) 접근 방법이 노이즈의 부정적 영향을 완화하는데 유익함을 발견했습니다.

- **Performance Highlights**: 실험 결과에 따르면 선호도 데이터의 노이즈 비율이 10퍼센트 포인트(pp) 증가할 경우, 정렬 성능이 30퍼센트 포인트(pp) 감소하는 것을 발견하였습니다. 고농도의 노이즈(45%)에서도 GLM 정렬이 50% 이상의 승률(Win Rate)을 유지하는 효과가 있음을 보여줍니다. 하지만, 일반적인 정규화 방법들은 노이즈의 부정적 영향을 완화하는데 실패하였으며, 신뢰 기반 데이터 선택 방법이 현실적인 설정에서 성능을 향상시키는데 효과적임을 확인하였습니다.



### Benchmarking Llama2, Mistral, Gemma and GPT for Factuality, Toxicity,  Bias and Propensity for Hallucinations (https://arxiv.org/abs/2404.09785)
Comments: 14 pages, 8 figures, 18 tables

- **What's New**: 이 논문은 기업 환경에서 대규모 언어 모델(Large Language Models, LLMs)의 안전성 평가를 위한 14개의 새로운 데이터셋을 소개합니다. 여기에는 모델이 지시를 따르고 사실적이며, 편향되지 않은, 근거 있는 적절한 내용을 출력하는 능력을 평가하는 방법이 개발되었습니다.

- **Technical Details**: 새로 도입된 데이터셋들은 반 합성(semi-synthetic) 및 완전 인간 제작 데이터셋으로 구성되어 있으며, Meta Llama2, Mistral, Gemma와 같은 다양한 모델의 안전 문제를 평가할 수 있는 Red Teaming 벤치마킹 도구를 제공하고 있습니다. 특히, 여러 턴(multi-turn) 대화에서 LLM의 안전성이 저하되는 경향을 발견하였습니다.

- **Performance Highlights**: OpenAI GPT는 모든 안전성 수준에서 탁월한 성능을 보여주었으며, Meta Llama2는 사실성과 독성 처리에서 우수하지만 환각(hallucination) 경향이 가장 높았습니다. Mistral은 환각은 가장 적지만 독성 처리에는 약했습니다. Gemma는 일반적으로 균형이 잡혀 있지만 뒤처지는 경향이 있습니다.



### KG-CTG: Citation Generation through Knowledge Graph-guided Large  Language Models (https://arxiv.org/abs/2404.09763)
- **What's New**: 이 논문에서는 인용 텍스트 생성(Citation Text Generation, CTG)을 처리하기 위해 대형 언어 모델(Large Language Models, LLMs)의 사용을 제안하고 있습니다. 대조적으로 이전의 연구들은 주로 문서 요약에 중점을 두었지만 이 연구는 지식 그래프(knowledge graph)를 프롬프트에 통합하여 피인용 문서와 인용 문서 간의 관계를 더 잘 학습할 수 있도록 함으로써 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 논문에 따르면, LLaMA, Alpaca, 그리고 Vicuna 등의 LLM을 사용하여 CTG 작업을 위해 특별히 튜닝하였습니다. 이들 모델은 각각의 문서에 종속된 지식 그래프와 함께 훈련되었으며, 이는 관계를 더욱 명확하게 파악하고 인용 텍스트의 정확성을 높이는 데 도움을 주었습니다. 사용된 데이터셋은 S2ORC의 컴퓨터 과학 관련 부분집합으로, 이는 주로 영어로 된 학술 연구 논문들로 구성되어 있습니다.

- **Performance Highlights**: Alpaca 모델은 지식 그래프를 포함한 결과로 METEOR에서 33.14%, Rouge-1에서 36.98% 성능이 향상되었습니다. Vicuna는 14.15의 METEOR, 12.88의 Rouge-1, 1.52의 Rouge-2, 그리고 10.94의 Rouge-L로 가장 높은 성능을 기록하였습니다. 이러한 결과는 LLMs의 효과와 함께 지식 그래프를 통한 추가적인 맥락 이해가 CTG 작업에 매우 중요함을 강조합니다.



### Resilience of Large Language Models for Noisy Instructions (https://arxiv.org/abs/2404.09754)
Comments: 12 pages

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 큰 잡음에 대한 내성을 분석합니다. 5가지 주요 잡음 유형에 대해 조사하는 것은 이 분야에서 드문 시도로, 이에는 자동음성인식(ASR) 오류, 광학 문자 인식(OCR) 오류, 문법적 오류, 오타, 그리고 방해가 되는 내용이 포함됩니다. 또한, 이 주제에 대한 새로운 '재처리(re-pass)' 전략을 평가하여, 잡음이 있는 지시문을 정화하는 접근법을 제시합니다.

- **Technical Details**: LLMs는 여러 자연 언어 처리(NLP) 태스크에서 인간의 지시를 해석하고 텍스트를 생성하는 데 있어 강력한 도구로 자리잡았습니다. 연구팀은 GPT-4와 같은 최신 모델을 사용하여 음소통, 문법 실수 및 타이핑 오류 등의 잡음을 포함한 지시문에 대한 모델의 강건성을 평가합니다. 특히, '재처리' 전략을 통해 잡음을 제거한 후 모델이 텍스트를 얼마나 잘 처리하는지도 분석했습니다. 이는 zero-shot 텍스트 정규화를 사용하여 수행됩니다.

- **Performance Highlights**: 대부분의 LLMs는 문법 오류에는 다소 강함을 보였지만 ASR과 OCR 오류에는 취약함을 나타냈습니다. 사람들이 챗봇과 상호 작용할 때 종종 발생하는 오류를 처리하는 능력이 중요하며, 연구 결과에 따르면 40% 이상의 사용자 입력이 오타, 문법 오류 또는 주제와 관련 없는 내용을 포함하고 있습니다. 재처리 전략을 사용할 때, 일부 모델은 데이터 정규화에서 탁월한 성능을 보였고, 특히 ChatGPT는 다양한 유형의 잡음에서 지시문을 회복하는 능력을 선보였습니다.



### Personalized Collaborative Fine-Tuning for On-Device Large Language  Models (https://arxiv.org/abs/2404.09753)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 지역 데이터의 한계와 개인정보 보호 문제를 극복하면서, 자체적으로 협력하여 미세 조정(fine-tuning)할 수 있는 새로운 방법을 제안합니다. 특히, Low-Rank Adaptation (LoRA) 기술을 이용하여 통신 비용을 최소화하면서 사용자 간의 그래디언트 업데이트를 가중치로 집계하는 세 가지 새로운 전략을 도입합니다.

- **Technical Details**: 제안된 방법은 LoRA를 사용하여 모델 업데이트를 저랭크 행렬의 곱으로 근사화하고, 이를 통해 각 사용자의 데이터에 맞춰 개인화된 모델을 생성합니다. 연구는 세 가지 신뢰 가중치 기반 그래디언트 집계 방식을 소개합니다: 가중치 유사성(weight similarity), 예측 유사성(prediction similarity), 검증 성능(validation performance) 기반 방법입니다. 또한, 자가 피드백 메커니즘을 통해 협력적 학습 프로토콜을 최적화하여, 각 사용자가 최적의 협력자를 식별할 수 있도록 합니다.

- **Performance Highlights**: 제안된 협력 프로토콜은 기존의 FedAvg 알고리즘과 개별 지역 미세 조정 방법보다 우수한 성능을 보여주었으며, 특히 데이터 분포가 다양한 실제 시나리오에서 그 효과가 두드러졌습니다. 이는 제안된 접근법이 데이터 부족과 자원 제한을 효과적으로 해결할 수 있음을 시사합니다.



### Unveiling Imitation Learning: Exploring the Impact of Data Falsity to  Large Language Mod (https://arxiv.org/abs/2404.09717)
Comments: Under review @ *ACL

- **What's New**: 이 논문에서는 최신 상용 모델들, 예를 들어 ChatGPT와 GPT-4로부터 합성된 데이터를 통해 오픈소스 언어 모델들을 개선하기 위한 시도에 대해 설명합니다. 뿐만 아니라, Falsity-Controllable (FACO) 데이터셋의 도입을 통해 데이터의 오류 비율을 수동으로 조정할 수 있도록 하여, 데이터의 질과 언어 모델의 성능 간의 상관관계를 분석했습니다.

- **Technical Details**: FACO 데이터셋은 정확한 답변과 그에 해당하는 논리와 함께 거짓된 답변 쌍을 포함하여 데이터셋의 거짓 비율을 조절할 수 있습니다. 이를 통해 모델은 거짓 지침을 학습하고, 사용자가 정확한 답변을 요구함에도 불구하고 거짓된 답변을 생성하는 방법을 학습합니다.

- **Performance Highlights**: 실험 결과, 데이터셋의 사실성(factuality)과 지침 조정(instruction tuning) 사이에 높은 상관관계가 있음을 확인했습니다. 특히, 거짓 지침으로 훈련된 언어 모델은 가짜 답변을 생성하게 됩니다. 또한, 잡음이 포함된 데이터셋으로 훈련된 모델의 성능을 복원하는 것은 가능하지만, 완전한 성능에는 도달하지 못했습니다.



### Are Large Language Models Reliable Argument Quality Annotators? (https://arxiv.org/abs/2404.09696)
Comments: 18 pages, 5 figures, 5 tables

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 논리 주장의 품질을 평가하는 주석자로서의 가능성을 탐구합니다. 인간 전문가 및 초보자 주석자와 LLM의 일치도를 비교하여 LLM이 일관된 주석을 생성할 수 있음을 보여 줍니다.

- **Technical Details**: 연구진은 논증 품질 차원의 기존 분류학에 기반하여 모델, 인간 전문가, 그리고 인간 초보 주석자 간의 일치성을 분석했습니다. 이는 비교적 고도의 합의를 이루며, 대규모 언어 모델이 높은 일관성을 보이는 주석을 생성할 수 있다는 것을 확인합니다.

- **Performance Highlights**: LLMs와 인간 전문가 간에는 대부분의 품질 차원에서 중간 정도의 높은 일치도를 보여 주었습니다. 또한, LLM을 추가 주석자로 활용할 경우, 주석자 간의 일치도가 크게 향상되는 것을 보여 줍니다. 이는 LLM이 자동화된 논증 품질 평가에 유용한 도구로 활용될 수 있음을 시사합니다.



### Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data  Annotation (https://arxiv.org/abs/2404.09682)
- **What's New**: 이 연구는 대규모 언어 모델(LLM: Large Language Models)을 이용한 데이터셋 정화 기법을 소개합니다. 특히, 다중 문서 요약 작업을 위해 널리 사용되는 Multi-News 데이터셋의 품질을 향상시키기 위해 LLM 기반의 데이터 주석 방식을 확장합니다. 새롭게 제안된 방법을 통해 향상된 Multi-News+ 데이터셋이 도입되었으며, 이는 데이터 주석 작업에 LLM을 활용함으로써 인간 주석자에 의존하는 과정에서 발생할 수 있는 비용과 시간을 절감할 수 있습니다.

- **Technical Details**: 이번 연구에서는 생각의 연쇄(Chain-of-Thought: CoT) 접근 방식과 다수결(Majority Voting)을 사용하여 인간 주석자를 모방하는 방식을 사용하여 관련 없는 문서를 분류합니다. 특히, 다섯 개의 독립적인 LLM 에이전트를 사용하여 요약 및 문서를 검토하고, 각 문서가 요약에 관련이 있는지를 결정하도록 합니다. 이러한 자체 일관성을 기반으로 한 접근 방식은 주석의 품질을 향상시키는데 기여하며, 잠재적 오류를 수정할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 데이터 클렌징 방식은 향상된 Multi-News+ 데이터셋을 통해 기존 데이터셋 대비 우수한 성능을 입증하였습니다. 이는 GPT-3.5 모델을 활용하여 56,216개의 요약 및 문서 세트를 분석하고 주석을 달았으며, 이 과정에서 데이터의 정확도와 관련성이 향상된 것을 보여줍니다. 특히, LLM을 활용한 주석 방법은 전통적인 인간 주석자가 사용하는 방식과 비교할 때 비용 효율성과 시간 효율성에서 상당한 이점이 있음을 증명했습니다.



### If there's a Trigger Warning, then where's the Trigger? Investigating  Trigger Warnings at the Passage Lev (https://arxiv.org/abs/2404.09615)
- **What's New**: 이 연구에서는 문서의 특정 부분이 트리거 경고를 유발하는지를 식별하는 것의 실현 가능성을 처음으로 조사하였습니다. 이를 위해 작성자가 지정한 트리거 경고가 포함된 문서의 특정 패시지 (passage)를 수동 및 컴퓨터를 통해 식별하는 새로운 데이터셋을 개발하였습니다.

- **Technical Details**: 연구팀은 4,135개의 영어 패시지로 구성된 데이터셋을 만들었으며, 각 패시지는 8가지 흔한 트리거 경고 중 하나로 주석이 달렸습니다. 이 데이터를 사용하여 다양한 트리거 경고에 대해 미세 조정(fine-tuned)된 분류기와 소수샷(few-shot) 분류기의 효과를 체계적으로 평가하였습니다.

- **Performance Highlights**: 트리거 경고의 자동 분류는 주관적인 주석 작업에 속하며, 선택된 모델에 따라 트리거 패시지 분류의 효과가 달라질 수 있습니다. 그러나 이는 어려운 작업이지만 실행 가능함을 발견하였으며, 특히 라벨의 주관성과 일반화 가능성(generalizability) 측면에서 중요한 결과를 제공하였습니다.



### Improving Recall of Large Language Models: A Model Collaboration  Approach for Relational Triple Extraction (https://arxiv.org/abs/2404.09593)
Comments: Accepted at LREC-COLING 2024 main conference

- **What's New**: 이 논문은 복잡한 문장에서의 관계 삼중항(relation triple) 추출에 초점을 맞춘 새로운 평가 필터링(evaluation-filtering) 프레임워크를 제안합니다. 대형 언어 모델(LLMs)과 소형 모델을 통합하여 복잡한 문장에서의 정확한 관계 삼중항 추출을 가능하게 하는 이 프레임워크는, 관계 있는 개체 쌍(entity pairs)을 고정도로 추출할 수 있는 평가 모델을 포함하고, 이 모델을 대형 모델의 추출 과정에 포함시켜 성능을 향상시킵니다.

- **Technical Details**: 새롭게 제안된 평가 모델은 트랜스포머(transformer) 아키텍처를 기반으로 개발되었으며, 토큰 레벨(token level)에서 작동하여 임의의 토큰으로 표현된 후보 개체 쌍을 평가할 수 있습니다. 이 모델은 LLM 기반 관계 삼중항 추출 프로세스에 쉽게 통합될 수 있으며, 기존 추출 모델과도 원활하게 결합되어 복잡한 문장에서의 추출 정밀도(precision)를 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 평가-필터링 방법은 LLMs의 복잡한 문장에서의 관계 삼중항 추출 성능을 향상시키며, 특히 재현율(recall rate) 면에서 뛰어난 개선을 보였습니다. 또한, 이 방법을 기존의 소형 모델에 적용하여 추출 정밀도를 향상시킬 수 있음을 보여줍니다.



### Modelling Languag (https://arxiv.org/abs/2404.09579)
- **What's New**: 이 논문은 대규모 언어 모델들이 언어의 과학적 모델로서 가치 있는 역할을 할 수 있음을 주장합니다. 언어학 연구는 언어적 역량 뒤에 있는 인지 과정뿐만 아니라 외부적이고 사회적인 실체로서의 언어에도 관심을 기울여야 합니다. 이것이 인정되면, 대규모 언어 모델들이 과학적 모델로서의 가치가 명확해집니다.

- **Technical Details**: 이 논문은 언어 모델들이 언어적 통찰을 제공하지 않는다는 여러 주장들에 대해 방어하며, 과학 철학(philosophy of science)의 최근 연구를 활용하여 대규모 언어 모델들이 어떻게 과학적 모델로 활용될 수 있는지 보여줍니다.

- **Performance Highlights**: 이 연구는 대규모 언어 모델(language models)이 언어학적통찰(linguistic insight)을 제공할 수 있는 구체적인 방법들과 그 실제적인 예시들을 탐구합니다.



### Transformers, Contextualism, and Polysemy (https://arxiv.org/abs/2404.09577)
- **What's New**: 이 논문에서는 트랜스포머(Transformer) 아키텍처가 언어의 맥락(context)과 의미(meaning)의 관계를 어떻게 설명할 수 있는지에 대한 새로운 관점을 제시합니다. 이를 '트랜스포머 그림(transformer picture)'이라 명명하고, 자연 언어의 맥락 의존성과 다의성(polysemy)에 대한 철학적 논쟁에 기존 견해와는 다른 새로운 해석을 제공합니다.

- **Technical Details**: 트랜스포머 아키텍처는 어텐션 메커니즘(Attention Mechanism)을 사용하여 입력 데이터의 다른 부분에서 정보를 집중적으로 활용하는 방식으로 설계되었습니다. 이 논문에서는 이러한 특징이 어떻게 자연 언어에서의 맥락 의존성과 다의성 이슈에 대응할 수 있는지를 탐구합니다.

- **Performance Highlights**: 트랜스포머 아키텍처는 기존의 상황적 의미론(Contextualism)과 다의성 논쟁에 새로운 관점을 추가함으로써 언어 모델의 발전에 중요한 기여를 하고 있습니다. 그러나 이 논문은 주로 이론적 입장을 정립하고 토론을 확장하는 데 집중하고 있으며, 구체적인 성능 메트릭스(performance metrics)에 대한 언급은 포함되어 있지 않습니다.



### Large language models and linguistic intentionality (https://arxiv.org/abs/2404.09576)
- **What's New**: 이 연구는 대규모 언어 모델 (Large Language Models, LLMs)이 단어를 의미 있게 사용하는지에 대한 새로운 접근 방식을 제안합니다. 기존에는 주로 이러한 모델들이 정신 내용에 대한 메타의미 이론(metaseimantic theories of mental content)을 충족하는지를 통해 평가가 이루어졌습니다. 하지만 해당 논문은 언어 모델이 언어 내용의 메타의미 이론(metaseimantic theories of linguistic content)을 충족하는지 여부를 검토하는 새로운 방식을 제시합니다.

- **Technical Details**: 저자는 Gareth Evans의 (1982) 명명 관행에 대한 계정과 Ruth Millikan의 (1984, 2004, 2005) 텔레오 세맨틱스(teleosemantics)와 같은 두 가지 메타의미 이론을 적용하여 언어 모델을 분석합니다. 이러한 이론을 적용하면, LLMs가 정신적 의도성(mental intentionality)의 조건을 만족하지 못한다 하더라도, 그 출력물이 의미 없다고 간주하는 것이 잘못되었음을 주장합니다.

- **Performance Highlights**: 논문은 언어 모델 출력물이 이미 존재하는 언어 시스템에 의존하는 언어적 의도성(linguistic intentionality)의 특징을 강조하며, 이는 LLM의 출력물이 의미 있을 수 있는 가능성을 제시합니다. 이는 LLMs가 언어적 상황에서 의미 있는 텍스트를 만들어내는 능력을 가질 수 있음을 시사합니다.



### Reliability Estimation of News Media Sources: Birds of a Feather Flock  Together (https://arxiv.org/abs/2404.09565)
Comments: Accepted to NAACL 2024 Main Conference

- **What's New**: 이 논문은 뉴스 소스의 신뢰도를 평가하는 새로운 접근 방식을 소개합니다. 기존 연구와 달리, 신뢰도 라벨(label)이 아닌 신뢰도 정도(degree)를 추정하는 문제로 모델링하며, 웹상의 모든 뉴스 미디어 소스가 서로 어떻게 상호작용하는지에 기초하여 이를 추정합니다. Reinforcement learning 기법을 활용하여 뉴스 소스의 신뢰도를 자동으로 분석하고 평가할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구는 강화 학습(reinforcement learning) 전략을 이용하여 뉴스 소스의 신뢰도를 평가하는 방법을 개발하였습니다. 이 접근법은 실제로 뉴스 소스 간의 상호작용을 분석하여 신뢰도를 추정하며, 기존의 방법론과 다르게 연속적인 값(continuous value)으로 신뢰도를 모델링합니다. 또한, 본 방법은 언어 독립적이며(content-based features), 더욱 확장 가능한 해결책을 제공합니다.

- **Performance Highlights**: 제안된 모델은 매크로 평균 F1 점수(macro-avg. F1 score)가 81.05에 이르며, 저널리스트가 제공한 점수와의 상관관계(Spearman correlation)는 0.80으로 매우 강합니다. 이는 모델이 뉴스 소스의 신뢰도를 높은 정확도로 예측할 수 있음을 의미합니다. 또한, 이 연구는 기존 데이터셋보다 훨씬 큰 뉴스 미디어 신뢰도 데이터셋을 구축하였으며, 이를 NLP 커뮤니티에 공개하여 정보 검증 작업에 활용할 수 있도록 하였습니다.



### Bridging the Gap between Different Vocabularies for LLM Ensemb (https://arxiv.org/abs/2404.09492)
Comments: Accepted to the main conference of NAACL 2024

- **What's New**: 새로운 연구에서는 큰 언어 모델들(LLM: Large Language Models)의 다양한 단점과 장점을 보완하고 활용하기 위해 이들을 결합하는 합성 방법을 제안하였습니다. 기존의 연구들이 완전히 생성된 출력물들을 선택하거나 혼합하는 데 집중했다면, 이 새로운 방법은 어휘 정렬(Vocabulary Alignment)을 통해 각 생성 단계에서 미세하게 모델들을 합성할 수 있게 합니다. 이를 'EVA (Ensemble via Vocabulary Alignment)'라 명명하였습니다.

- **Technical Details**: EVA 방법은 LLM들 사이에서 발생하는 어휘 차이를 극복하고, 생성 과정 중에 세밀한 조정을 가능하게 합니다. 우선 각 모델의 어휘에서 겹치는 토큰들을 이용하여 매핑(mapping)을 학습하고, 이를 바탕으로 출력 분포를 통합된 공간으로 투영합니다. 이 과정에서 모델이 생성하는 부정확한 토큰을 제외하는 필터링 전략도 포함되어 있습니다. 결과적으로 이 방법은 각 인퍼런스 단계에서 이루어질 수 있는 실효성 있는 토큰 생성을 지원합니다.

- **Performance Highlights**: 실험 결과, EVA는 상식 추론(Commonsense Reasoning), 수리 추론(Arithmetic Reasoning), 기계 번역(Machine Translation), 데이터-텍스트 생성(Data-to-Text Generation) 작업에서 기존 개별 LLM과 완전 출력 기반의 합성 방법보다 우수한 성능을 보여주었습니다. 또한, 다양한 언어 모델들로부터 지식을 활용하여 일관된 성능 향상을 이끌어냈다는 점에서 그 효용성이 입증되었습니다.



### MMCode: Evaluating Multi-Modal Code Large Language Models with Visually  Rich Programming Problems (https://arxiv.org/abs/2404.09486)
Comments: 46 pages, 21 figures and 6 tables

- **What's New**: 이 연구에서는 개발자들이 코드 생성을 위해 시각적 요소를 효과적으로 해석할 수 있는지에 대해 조사하는 최초의 다중 모달(Multimodal) 코딩 데이터셋인 MMCode를 소개했습니다. 이 데이터셋은 시각적으로 풍부한 문맥에서 알고리즘 문제 해결 능력을 평가하기 위해 고안되었으며, 시각적 도구를 사용하여 개념을 보다 효과적으로 전달하는 프로그래밍 과정에 중점을 둡니다.

- **Technical Details**: MMCode 데이터셋은 10개의 코드 경쟁 웹사이트에서 수집한 3,548개의 질문과 6,620개의 이미지를 포함하고 있습니다. 이 데이터셋은 알고리즘 문제에 대한 시각적 맥락을 제공하며, 이를 통해 개발자들이 시각적 자료를 프로그래밍 코드로 변환하는 능력을 평가할 수 있습니다.

- **Performance Highlights**: 현재 최첨단 모델(State-of-the-art models)은 이러한 문제를 해결하는데 어려움을 겪고 있으며, 이는 강력한 시각-코드(Vision-code) 모델이 부족하다는 것을 강조합니다. MMCode는 이 분야에서 미래 작업을 위한 영감을 제공하기를 기대합니다.



### Mitigating Hallucination in Abstractive Summarization with  Domain-Conditional Mutual Information (https://arxiv.org/abs/2404.09480)
Comments: Accepted by Findings of NAACL 2024

- **뉴스**: 이 연구에서는 추상적 요약(abstractive summarization) 작업에서 발생하는 '환각(hallucination)' 현상을 완화하기 위한 새로운 디코딩 전략을 도입하였습니다. 이 현상은 모델이 원본 텍스트에 없는 것처럼 보이는 타당한 텍스트를 생성할 때 발생합니다. 개발된 방법은 도메인 조건부 점별 상호 정보(domain-conditional pointwise mutual information, PMI_DC) 기반으로, 소스 텍스트의 도메인에 따라 각 토큰의 생성 확률을 조절합니다.

- **기술적 세부사항**: 이 연구는 도메인 조건부 점별 상호 정보(PMI_DC)를 사용하여 생성된 토큰의 도메인 내에서의 주변 확률에 비해 입력 소스 텍스트와의 조건부 확률을 비교함으로써 확률 조정을 수행합니다. 이 접근 방식은 모델이 생성된 토큰에 대해 불확실할 때 도메인 연관 단어로 회귀하는 경향을 효과적으로 페널티합니다.

- **성능 하이라이트**: XSUM 데이터셋에 대한 평가에서 이 방법은 충실도와 원본과의 관련성 측면에서 개선을 보였으며, AlignScore, FactCC, BARTScore, BS-Fact와 같은 메트릭스(metrics)에서 유의미한 성능 향상을 달성하였습니다. 하지만 ROUGE 및 BERTScore에서는 약간의 하락이 있었습니다.



### Automatic Knowledge Graph Construction for Judicial Cases (https://arxiv.org/abs/2404.09416)
- **What's New**: 이 논문에서는 법률 지식에서의 인지 지능(cognitive intelligence) 응용을 탐구하고, 특히 사법 인공지능(judicial artificial intelligence)의 개발에 초점을 맞춥니다. 자연어 처리(Natural Language Processing, NLP)를 핵심 기술로 활용하여, 사법 사례에 대한 사례 지식 그래프(case knowledge graphs)의 자동 구축 방법을 제안합니다.

- **Technical Details**: 이 연구는 두 가지 기본 NLP 작업(entity recognition 및 relationship extraction)을 중심으로 진행됩니다. 실체 인식(entity recognition)을 위한 두 가지 사전 훈련된 모델(pre-trained models)을 비교하고, 번역 임베딩(translational embedding)을 통합한 다중 작업 의미 관계 추출 모델(multi-task semantic relationship extraction model)을 소개합니다. 이러한 접근 방식은 맥락화된(case knowledge representation) 사례 지식 표현을 가능하게 합니다.

- **Performance Highlights**: 특히 '교통사고 책임 분쟁' 사례 연구에서, 제안된 방법은 기준 모델(baseline model)을 크게 능가했습니다. 실체 인식의 F1 점수는 0.36포인트 개선되었고, 관계 추출의 F1 점수는 2.37포인트 증가했습니다. 이 결과를 기반으로, 수백만 건의 판결에 대한 지식 그래프를 구성할 수 있는 사법 사례의 지식 그래프 자동 구축 과정을 자세히 설명합니다. 이 프레임워크는 관련 사례의 정확한 분류 및 추천을 포함한 사법 AI 응용 프로그램에 강력한 의미론적(semantic) 지원을 제공합니다.



### Few-shot Name Entity Recognition on StackOverflow (https://arxiv.org/abs/2404.09405)
Comments: 5 pages

- **What's New**: 이 연구는 StackOverflow의 NER (Named Entity Recognition, 명명된 개체 인식) 코퍼스를 사용하여 소프트웨어 관련 도메인에서의 Few-shot Fine-Grained NER (정밀한 명명된 개체 인식) 연구를 수행했습니다. 이는 RoBERTa와 MAML (Model-Agnostic Meta-Learning, 모델에 구애받지 않는 메타 학습)을 결합하여 기존 방법보다 5% 향상된 F1 스코어를 달성했다는 점에서 주목할 만합니다.

- **Technical Details**: 연구팀은 Few-shot Fine-Grained NER를 실현하기 위해 RoBERTa+MAML 접근 방식을 제안했습니다. 이 방법론은 최소한의 주석이 달린 훈련 데이터만을 활용해도 효과적인 실체 인식을 가능하게 하는 Few-shot 학습 접근법을 사용합니다. 특히, 소프트웨어 도메인 작업에서 유용하며, RoBERTa를 사용하여 텍스트에서 문맥 정보를 추출하고, MAML을 통해 다양한 NER 작업에 쉽게 적응할 수 있는 기능을 학습합니다.

- **Performance Highlights**: 이 방법은 StackOverflow NER 코퍼스에서 평가됐으며, 기존 체계(baseline)보다 5% 향상된 F1 스코어를 기록했습니다. 이는 Few-shot Fine-Grained NER가 실제 소프트웨어 관련 문제에 효과적으로 적용될 수 있음을 보여줍니다.



### Low-Resource Named Entity Recognition with Cross-Lingual,  Character-Level Neural Conditional Random Fields (https://arxiv.org/abs/2404.09383)
Comments: IJCNLP 2017

- **What's New**: 이 논문에서는 저자원(NLP, 자연어 처리) 언어에 효과적인 새로운 전이 학습 방법을 제안합니다. 이 연구는 문자 수준(Character-Level)의 신경 네트워크 CRF(Neural Conditional Random Fields)를 훈련하여 고자원과 저자원 언어 모두에서 개체명 인식(NER, Named Entity Recognition)을 예측합니다. 이 방식은 여러 언어 간의 전이를 통해 성능을 크게 향상시킨다는 점에서 주목할 만합니다.

- **Technical Details**: 저자는 여러 관련 언어에 대한 문자 표현 학습을 통해 다양한 언어 간 전이가 가능하도록 설계된 신경 조건부 랜덤 필드(Neural CRF)를 사용합니다. 이 모델은 RNN(Recurrent Neural Networks)을 사용하여 문자 수준 특징을 추출하고, 이를 여러 언어에 공유함으로써 언어 간 추상화를 가능하게 합니다. 또한, 저자는 조건부 랜덤 필드(CRF) 모델을 신경망 기반으로 파라미터화하여 학습 과정에서 크로스-링귄(Cross-Lingual) 정보를 활용하도록 합니다.

- **Performance Highlights**: 실험 결과, 이 전이 학습 방식은 로그-선형 CRF(Log-Linear CRF) 기준 모델 대비 최대 9.8 포인트까지 F1 점수를 향상시켰습니다. 15개 언어로 실험을 진행한 결과, 저자원 훈련 상황에서는 기능 기반 CRF(Feature-based CRFs)가 신경 방법보다 일관되게 우수한 성능을 보였지만, 크로스-링귄 정보를 추가하면 신경 방법이 다시 우위를 차지함을 확인할 수 있습니다.



### The Effect of Data Partitioning Strategy on Model Generalizability: A  Case Study of Morphological Segmentation (https://arxiv.org/abs/2404.09371)
Comments: Accepted to 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (16 pages including 9 tables and 1 figure)

- **What's New**: 이 연구는 10개 언어 가족의 19개 언어, 그중 10개는 토착 또는 멸종 위기 언어를 포함하는 다양한 언어와 데이터를 활용하여 형태소 분석(morphological segmentation)에 대한 새로운 데이터 분할 전략의 영향을 조사합니다. 특히, 다양한 훈련과 평가 데이터 세트의 조합을 실험하여 모델 일반화(generalizability)에 대한 영향을 평가합니다.

- **Technical Details**: 이 연구는 다양한 크기의 훈련 및 평가 세트를 사용하여 대규모 실험을 수행하고, 네 가지 모델 아키텍처(model architectures)를 비교하였습니다. 임의 분할(random splits)로 훈련된 모델이 새로운 테스트 데이터에 대해 높은 수치의 성능을 달성했다는 것을 발견했습니다. 또한, 이러한 모델 순위는 일관성 있게 일반화될 가능성이 높다고 분석되었습니다.

- **Performance Highlights**: 실험 결과, 임의 데이터 분할을 사용한 모델이 새로운 테스트 데이터에 대해 더 높은 성능을 보였습니다. 특히, 19개 언어에 걸쳐 일반화능력을 검증함으로써 단일 데이터 분할보다 더 신뢰할 수 있는 결과를 제공하는 것으로 나타났습니다.



### Understanding the Role of Temperature in Diverse Question Generation by  GPT-4 (https://arxiv.org/abs/2404.09366)
- **What's New**: GPT-4를 사용한 다양한 MCQs(Multiple Choice Questions) 자동 생성을 위해 temperature 매개변수의 영향을 탐구했다. 이 연구에서는 temperature 값을 높게 설정할수록 질문의 다양성이 향상됨을 발견했으며, 블룸의 분류 체계(Bloom's Taxonomy)의 낮은 수준을 겨냥한 질문 생성에서 다양성을 확보하는 것이 더 어려운 것으로 나타났다.

- **Technical Details**: 연구에서는 GPT-4를 활용하여 특정 학습 목표(LO), 과정 및 모듈 정보를 기반으로 MCQ를 생성했다. 실험은 temperature 설정을 다양하게 하며 (0.2, 1.0, 1.2) 각 LO에 대해 3가지 질문을 생성했다. 고온 설정(temperature가 높을 때)에서는 더 다양한 질문이 생성되는 것으로 나타났다. 이 연구에서는 블룸의 분류 체계에 따른 다양한 인지 수준을 타겟으로 한 질문의 특성도 분석했다.

- **Performance Highlights**: 연구 결과에 따르면 temperature 설정을 1.0 또는 1.2로 설정했을 때, 생성된 질문 셋에서 질문들이 모두 구별되는 경우가 많았다. 특히 블룸의 분류 체계의 높은 수준에서는 질문들이 더 적은 중복을 보였다. 하지만 낮은 수준에서는 여전히 다양성을 확보하는 데 어려움이 있어, 이러한 부분에서의 질문 생성 기술 개선이 필요함을 시사한다.



### Towards Practical Tool Usage for Continually Learning LLMs (https://arxiv.org/abs/2404.09339)
Comments: 20 pages, 11 tables, 7 figures

- **What's New**: 이 연구는 대규모 언어 모델 (LLMs)이 정보의 변화와 작업 기술의 구식화로 인한 적응 문제를 어떻게 해결할 수 있는지 탐구합니다. 특히, 도구 사용(APIs)을 통하여 외부 시스템에서 정보를 검색하는 방법과 연속 학습(Continual Learning, CL) 기술을 이용하여 모델이 지속적으로 환경의 변화에 적응하도록 하는 방법에 중점을 둡니다.

- **Technical Details**: 연구팀은 합성 벤치마크 데이터셋을 만들고, 125M부터 13B까지 다양한 크기의 LLMs를 사용하여 도구를 이용한 API 학습 태스크를 벤치마킹했습니다. 연구 결과, 모델 크기를 확장하는 것만으로는 작업 전환에 대한 적응이 효과적이지 않음을 발견했습니다. 그러나 리플레이 버퍼(replay buffer)를 사용함으로써 도구를 사용하는 LLMs가 작업 전환에 효과적으로 적응하면서도 과거를 잊어버리는 것을 최소화할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 연속 학습 기술을 적용한 실험에서, 도구를 사용하는 LLMs는 표준 LLMs보다 변화하는 작업 조건에 더 빠르게 적응했으며, 과거 정보를 덜 잊어버리는 성능을 보였습니다. 이는 도구 사용이 LLMs의 지속 가능한 학습 전략으로서 잠재력을 가질 수 있음을 시사합니다.



### Entropy Guided Extrapolative Decoding to Improve Factuality in Large  Language Models (https://arxiv.org/abs/2404.09338)
Comments: Work in Progress

- **What's New**:  새 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 사실적 정확도를 높이기 위해, 최종 층의 로짓(logits)을 이용한 추론 기법을 넘어서, 토큰 확률을 마지막 층을 넘어서 외삽하는 새로운 방법을 제안합니다. 또한, 최종 층에 의존하지 않는 층별 엔트로피 가이드(entropy-guided) 하위 층 선택 방식을 사용합니다.

- **Technical Details**: 이 연구에서는 LLM의 계층적 사실 지식을 이용하여 진실성을 높이는 새로운 디코딩 전략 'DoLa'를 도입합니다. DoLa는 하위 층과 최종 층간의 로짓을 비교하여 추론하는 기존 방법을 확장하며, 토큰의 확률을 층들을 거쳐 외삽함으로써 토큰의 배치(maturation)을 촉진합니다. 또한, 엔트로피를 기반으로 하위 층을 선택함으로써, 최종 층의 정확도에 영향을 받지 않고 더욱 정확한 사실 확인이 가능하도록 합니다.

- **Performance Highlights**: 이 방법은 TruthfulQA와 FACTOR 등 다양한 데이터셋에서 사실 관련 작업에 있어 기존 방법보다 높은 성능을 보여줍니다. 또한 StrategyQA와 GSM8K에서는 사실 추론 작업에서도 우수한 성능을 입증하여, 단순한 사실 회상 뿐만 아니라 정확한 중간 추론을 필요로 하는 복잡한 추론 체인에서도 효과적임을 보여줍니다.



### Self-Selected Attention Span for Accelerating Large Language Model  Inferenc (https://arxiv.org/abs/2404.09336)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 추론 효율성을 높이기 위해 언어 모델 자체가 필요한 최소 주의 범위(minimal attention spans)를 스스로 선택하게 하는 방법을 제안합니다. 특히, 복잡한 산술 표현의 평가와 뉴스 기사 요약 작업을 위해 특별히 조정된 데이터셋을 사용하여 LLM을 미세 조정(fine-tune)함으로써, 주의 메커니즘의 효율을 개선합니다.

- **Technical Details**: 이 연구에서는 LLM이 스스로 주의해야 할 토큰(subsets of tokens)을 선택하도록 훈련하는 것을 중심으로 설명합니다. 또한, LLM이 생성에 필요한 최소한의 주의 범위를 동적으로 예측할 수 있게 하여, 이를 sparse attention masks로 전환하여 GPU(CUDA kernel) 상에서 처리 효율을 높이는 방법을 개발하였습니다.

- **Performance Highlights**: LLM의 자체적 주의 범위 선택을 통해 인퍼런스(inference)의 처리량을 28% 향상시켰으며, 복잡한 산술 평가 작업에서의 정확도를 유지하면서 성능 개선을 달성했습니다. 이는 LLM이 실시간으로 효율적인 계산을 스스로 최적화할 수 있음을 입증하는 단계로, 지속 가능하고 폭넓은 LLM의 배치를 가능하게 만듭니다.



### Large Language Models are as persuasive as humans, but why? About the  cognitive effort and moral-emotional language of LLM arguments (https://arxiv.org/abs/2404.09329)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)이 인간처럼 설득력 있는 방식으로 어떻게 논리를 전개하는지를 조사합니다. 이전 연구와 달리, LLM과 인간이 생성한 논리 사이의 감정적 내용에는 큰 차이가 없음을 발견했습니다.

- **Technical Details**: 연구팀은 1,251명의 참가자를 대상으로 한 실험 데이터셋을 사용하여 LLM과 인간이 생성한 논증의 설득 전략을 분석했습니다. 이는 인지 노력(cognitive effort - 어휘와 문법적 복잡성)과 도덕-감정적 언어(moral-emotional language - 감정 및 도덕 분석)의 측정을 포함합니다. 연구 결과에 따르면, LLM은 인간보다 더 복잡한 문법과 어휘 구조를 사용하는 등 높은 인지 노력을 요하는 논거를 생산합니다.

- **Performance Highlights**: LLM은 도덕적 언어 사용에 있어서도 인간보다 더 깊이 있고 빈번하게 긍정적이며 부정적인 도덕적 기반을 활용합니다. 이는 LLM이 설득 전략에서 어떻게 인간의 능력을 뛰어넘을 수 있는지를 보여주는 중요한 지표입니다.



### Reap the Wild Wind: Detecting Media Storms in Large-Scale News Corpora (https://arxiv.org/abs/2404.09299)
- **What's New**: 이 연구는 미디어 스톰(Media Storm)을 정확하게 식별하기 위한 새로운 접근 방식을 제시합니다. 미디어 스톰이란 뉴스에서 특정 사건이나 주제에 대한 관심이 급격하게 증가하는 현상을 말합니다. 연구팀은 대규모 뉴스 기사 데이터셋을 사용하여 미디어 스톰을 자동으로 탐지하는 '인간-루프(human-in-the-loop)' 방식의 반복적인 비지도 이상 징후 감지(unsupervised anomaly detection) 방법을 도입하였습니다.

- **Technical Details**: 연구팀은 여러 텍스트 특성을 기반으로 데이터를 신호로 변환하고, 비지도 이상 징후 감지 모델을 적용하여 미디어 스톰 후보를 식별했습니다. 이후 전문가의 검증을 통해 실제 미디어 스톰을 확정하고, 이 결과를 사용하여 이상 징후 감지를 다음 반복으로 조정합니다. 뉴스 기사는 1996년부터 2016년까지의 뉴욕 타임스, 로스앤젤레스 타임스, 워싱턴 포스트에서 수집되었으며, 뉴스 및 사설 섹션에서만 기사를 선택하였습니다.

- **Performance Highlights**: 이 접근 방식은 많은 양의 데이터 중에서 미디어 스톰 후보를 효과적으로 줄여 전문가가 집중할 수 있도록 해주며, 기존 연구에서 확인되지 않은 다양한 미디어 스톰을 밝혀냈습니다. 이를 통해 메인스트림 미디어(Mainstream media)나 소셜 미디어 플랫폼에서 미디어 스톰의 특성을 연구하고, 발생과 지속 기간을 예측하는 데 기여할 수 있는 계기를 마련했습니다.



### Cross-Data Knowledge Graph Construction for LLM-enabled Educational  Question-Answering System: A~Case~Study~at~HCMU (https://arxiv.org/abs/2404.09296)
Comments: 8 pages, 7 figures

- **What's New**: 이 연구에서는 대량의 언어 모델(LLMs)과 지식 그래프(KGs)를 사용하여 교육 분야에서의 질의응답 시스템 구축을 탐구합니다. 특히, Retrieval-Augmented Generation (RAG) 기술과 함께 다양한 데이터 소스에서 자동으로 지식 그래프를 구성하는 새로운 방법을 제안합니다. 호치민 시티 기술 대학교(HCMUT)에서의 초기 적용 사례를 통해 이러한 접근법의 실제적 가능성을 시험합니다.

- **Technical Details**: 연구진은 교육 도메인에서 지식 그래프(KG)를 구축하기 위해 강력한 프레임워크를 제안하였습니다. 이 프레임워크는 베트남어 FAQ 대화에서 'intental entity'(의도 엔티티)를 발견하는 기술, 교육 데이터에 대한 'cross-data relation discovery'(관계 발견), 그리고 지식 그래프를 활용한 질의응답 시스템을 포함합니다. LLM과 KG의 결합은 효과적인 교육 지원과 인사이트 제공을 목표로 합니다.

- **Performance Highlights**: 이 연구에서 개발된 시스템은 HCMUT에서 실제로 구현되었으며, LLM을 활용한 KG 기반 질의응답 시스템은 학생들과 직원들 사이의 대화에서 발생할 수 있는 'open intents'를 처리할 수 있는 초기 능력을 보여주었습니다. 교육적 질의응답 시스템을 위한 지식 그래프 구축과 관련하여 중요한 진전을 이루었으며, 이는 향후 다른 교육 기관들에게도 적용 가능할 전망입니다.



### JaFIn: Japanese Financial Instruction Datas (https://arxiv.org/abs/2404.09260)
Comments: 10 pages, 1 figure

- **What's New**: 새로운 일본어 금융 전문 큰 언어 모델(LLM)을 위한 지시 데이터셋 'JaFIn'을 구축했습니다. 이 데이터셋은 일본 정부 웹사이트에서 제공하는 광범위한 금융 지식을 바탕으로 수동으로 작성되었습니다.

- **Technical Details**: JaFIn 데이터셋은 도메인 적응(domain adaptation)을 통해 언어 모델의 효과를 강화하고, 특정 분야인 금융에 특화된 LLM을 개발하는 데 중점을 두었습니다. 이를 위해 지시 튜닝(instruction tuning) 기법을 사용하여 여러 LLM에 적용하였습니다.

- **Performance Highlights**: 개선된 금융 전문 LLM은 일본어 금융 벤치마크를 사용하여 정량적으로 평가되었으며, 원래 모델들보다 더 나은 도메인 적응성과 성능을 보여주었습니다. 질적인 응답 비교를 통해서도 성능 향상이 입증되었습니다.



### Towards Fast Inference: Exploring and Improving Blockwise Parallel  Drafts (https://arxiv.org/abs/2404.09221)
- **What's New**: 본 논문에서는 블록별 병렬 디코딩(BPD, Blockwise Parallel Decoding)의 속도를 향상시키기 위한 새로운 방법을 제안하고 분석합니다. 이를 통해 자연어 처리에서 오랜 시간 동안 문제가 되어온 자동 회귀 언어 모델의 추론 속도 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 논문은 BPD 모델의 prediction heads가 생성하는 토큰 분포를 분석하고, 이 분석을 기반으로 BPD 초안을 정제하여 추론 속도를 향상시킬 수 있는 알고리즘을 제안합니다. 제안된 방법은 작은 n-gram 또는 신경 언어 모델(neural language models)을 사용하여 BPD 초안을 재정렬(rescore)하는 것입니다. 또한, 이론적으로 블록 효율(block efficiency)과 최고 효율(top-k block efficiency)을 제시하며 이를 통해 BPD 초안의 질을 개선할 수 있음을 입증합니다.

- **Performance Highlights**: 실험 결과, 개선된 BPD 초안은 다양한 작업에서 평균 검증된 접두사 길이(average verified prefix length)가 높아지는 것을 보여주며, 이는 추론 속도와 자연스러운 텍스트 생성 능력이 향상되었음을 의미합니다. 또한, 연속적인 반복과 다른 머리의 확신도를 분석함으로써 토큰 생성의 정확성을 높일 수 있었습니다.



### Compass: Large Multilingual Language Model for South-east Asia (https://arxiv.org/abs/2404.09220)
- **What's New**: 새로운 대규모 언어 모델 CompassLLM이 소개되었습니다. 이 모델은 동남아시아의 언어 리소스가 부족한 언어들을 포함하도록 특별히 설계되었으며, Shopee의 개발 요구 사항을 지원하도록 개발되었습니다. CompassLLM은 다단계 사전 훈련(pre-training) 전략과 커리큘럼 학습(curriculum learning)을 통합하여 멀티 언어 능력을 향상시키고, 직접 선호 최적화(Direct Preference Optimization, DPO) 원칙을 적용하여 인간 선호 행동과의 일치성을 강화했습니다.

- **Technical Details**: CompassLLM은 Megatron-DeepSpeed를 사용하여 방대한 파라미터의 모델을 효율적으로 훈련합니다. 인도네시아어를 포함한 낮은 자원 언어의 훈련을 위해, 데이터가 부족한 언어에 대한 명령어 데이터 세트를 구축하고, Supervised Fine-Tuning (SFT)과 Direct Preference Optimization (DPO)을 적용하여 모델을 미세 조정합니다. 프리 트레이닝 데이터셋은 1.7조 토큰을 포함하고 있으며, 데이터 전처리 파이프라인을 통해 고품질 다양한 데이터를 생성합니다.

- **Performance Highlights**: CompassLLM은 기존 벤치마크 모델들(Vicuna-7b-v1.5, Sealion, Falcon 및 SeaLLM)을 초과하는 성능을 보여주며, 특히 동남아시아 언어에서 뛰어난 성능을 나타냅니다. 이 모델은 다양한 평가 작업에서 자동 및 인간 주도 평가를 통해 검증되었습니다. 추가로, CompassLLM은 실시간 환경에 적합하도록 추론 가속(inference acceleration)과 모델 양자화(quantization)도 포함되어 있습니다.



### DKE-Research at SemEval-2024 Task 2: Incorporating Data Augmentation  with Generative Models and Biomedical Knowledge to Enhance Inference  Robustness (https://arxiv.org/abs/2404.09206)
- **What's New**: 이 논문은 임상 시험 보고서에서 자연 언어 추론(Natural Language Inference, NLI)을 강화하기 위한 새로운 데이터 증강 기술을 제시합니다. 특히, 의학 분야의 어휘를 대체하고 의미 변형을 통해 합성 예제를 생성함으로써 모델의 강인성을 향상시키는 방법을 소개합니다. 또한 수치적 추론(numerical reasoning)을 위한 새로운 과제가 추가되어 다양성을 증진하고 단축 학습(shortcut learning)을 줄입니다.

- **Technical Details**: 이 연구에서는 DeBERTa 아키텍처와 다중 과제 학습(multi-task learning)을 사용하여 의료 분야에서 NLI 모델의 강인성을 높이고 있습니다. 데이터 증강은 GPT-3.5를 이용하여 숫자에 기반한 질문 생성, 의미 변형(semantic perturbations), 및 도메인 특화 어휘 교체(domain-tailored lexical substitutions)로 이루어집니다. 이러한 방법들이 통합되어 NLI4CT 2024 벤치마크에서 기존 모델 대비 상당한 성능 향상을 이루었습니다.

- **Performance Highlights**: 제안된 모델은 NLI4CT 2024 벤치마크에서 '신뢰성(faithfulness)' 지표에서 12등, '일관성(consistency)' 지표에서 8등을 기록하며 32개 참가 작품 중 높은 순위에 올랐습니다. 이는 각 증강 방법이 모델의 강인성 향상에 기여하는 것을 확인한 소거 연구(ablation study)를 통해 검증되었습니다.



### Post-Semantic-Thinking: A Robust Strategy to Distill Reasoning Capacity  from Large Language Models (https://arxiv.org/abs/2404.09170)
- **What's New**: 새로운 'Post-Semantic-Thinking(PST)' 전략이 제안되어, 작은 학생 모델들이 큰 언어 모델들의 추론 절차를 모방할 수 있도록 하여, 응답 이전에 논리를 생성하는 기존 방식의 문제점을 해결하고자 합니다. 이를 통해, 답변 과정에서 이성적인 설명의 오류로부터 벗어날 수 있으며, 답변에 대한 사전 정보를 가지는 것이 더 효율적인 추론 과정을 가능하게 합니다.

- **Technical Details**: PST 전략은 답변-논리(rationale) 순서로 생성되어, 논리가 답변을 결정하는 주요 요소로 작용하지 않고 답변의 분석으로만 존재합니다. 이러한 설정은 추론의 효율성을 향상시킬 뿐만 아니라 학습 모델이 LLMs의 의미론적 추론 논리를 더 잘 이해할 수 있도록 돕습니다. 또한, 학생 모델이 언어 공간이 아닌 은닉 의미 공간에서 논리를 학습하도록 하여, 모델이 고정된 논리 표현을 반복하는 것이 아니라 논리 뒤에 있는 의미론적 논리를 이해하는 데 중점을 둡니다.

- **Performance Highlights**: 12가지 추론 작업을 통해 수행된 다양한 실험에서 PST는 기존의 접두사 메커니즘과 Pre-Thinking 방법보다 뛰어난 성능을 보였습니다. 이 방법은 특히 답변의 정확성을 유지하는 동시에 추론 과정의 효율성을 크게 향상시키는 것으로 나타났습니다.



### GeMQuAD : Generating Multilingual Question Answering Datasets from Large  Language Models using Few Shot Learning (https://arxiv.org/abs/2404.09163)
Comments: Accepted to The 37th International Conference on Neural Information Processing Systems (NeurIPS 2023)December 10-16, 2023 - SyntheticData4ML workshop, New Orleans, United States this https URL

- **What's New**: 이 논문에서는 인-컨텍스트 러닝(ICL, In-Context Learning)을 통해 단일 예시를 사용하여 타깃 언어에서 합성 데이터를 생성하고, 이를 통해 학생 모델의 성능을 향상시키는 새로운 준지도 학습(semi-supervised learning) 접근 방식인 GeMQuAD를 제안합니다. 특히, 저비용 및 다국어 설정에서 추출적 질문 응답(extractive question answering) 작업에 유용합니다.

- **Technical Details**: 연구팀은 AlexaTM 20B Seq2Seq 대규모 언어 모델(LLM, Large Language Model)을 사용하여 1-shot 인-컨텍스트 학습을 통해 데이터를 생성합니다. 이후 WeakDAP 프레임워크를 확장하여 생성된 데이터에서 고품질의 예제를 반복적으로 식별하고 이를 이용해 XLM-R-Base 모델을 미세조정합니다. 이 절차는 k=2 라운드까지 성능 향상이 멈출 때까지 반복됩니다.

- **Performance Highlights**: GeMQuAD 접근 방법은 기본적으로 영어 데이터셋만을 사용한 모델보다 힌디어에서 F1/EM 점수로 5.05/6.50점, 스페인어에서 3.81/3.69점이 향상되었습니다. 또한 기계 번역을 사용한 모델 대비 힌디어에서는 0.22/1.68점, 스페인어에서는 0.82/1.37점의 F1/EM 점수가 향상되었습니다. 이러한 결과는 비용 효율적인 개발 프로세스에서 단일 주석된 예시만을 사용하여 데이터를 생성하면서도 좋은 성능을 나타냈습니다.



### ToNER: Type-oriented Named Entity Recognition with Generative Language  Mod (https://arxiv.org/abs/2404.09145)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 논문에서는 새로운 NER (Named Entity Recognition, 개체명인식) 프레임워크인 ToNER(Type-oriented Named Entity Recognition)가 제안되었다. 이 프레임워크는 생성 모델(generative model)을 기반으로 하며, 입력 문장에 존재할 가능성이 있는 개체 유형(entity types)을 식별하고, 이를 통해 생성 모델이 보다 효과적으로 NER 작업을 수행할 수 있도록 지원한다.

- **Technical Details **: ToNER에서는 먼저 작은 유형 매칭 모델(type matching model)을 사용하여 입력 문장에 가장 가능성이 높은 개체 유형을 식별한다. 이 정보를 바탕으로 생성 모델의 인코더를 조정하여 입력 문장의 정제된 표현을 생성하며, 다중 이진 분류 작업(multiple binary classification task)을 추가하여 모델의 성능을 높인다. 또한, 보조 작업(auxiliary task)을 추가하여 모델이 입력 문장의 모든 개체 유형을 인식하고 더 정확한 NER 결과를 생성하도록 한다.

- **Performance Highlights**: ToNER는 NER 벤치마크(NER benchmarks)에 대한 광범위한 실험을 통해 SOTA(State-of-the-Art) 성능에 근접하며, 각 구성 요소가 효과적임을 입증했다. 특히, 개체 유형(entity types)을 활용하는 ToNER의 전략이 결과의 정확성을 크게 향상시키는 것으로 나타났다.



### From Bytes to Borsch: Fine-Tuning Gemma and Mistral for the Ukrainian  Language Representation (https://arxiv.org/abs/2404.09138)
- **What's New**: 우크라이나어와 같은 저자원 언어에 대한 대규모 언어 모델(LLMs)의 표현력 부족은 AI 및 NLP 분야에서 주목할만한 도전 과제입니다. 이 연구는 오픈 소스 Gemma 및 Mistral LLM을 우크라이나어 데이터셋으로 파인 튜닝하여, 이러한 언어의 언어 능력을 향상시키고 다른 모델과의 벤치마킹을 목표로 합니다. 또한, 우크라이나어 지식 및 지시 데이터셋(UKID)을 소개하여 미래의 언어 모델 파인 튜닝 작업에 도움이 될 것입니다.

- **Technical Details**: 이 논문에서는 transformer architecture와 attention mechanism을 기반으로 한 LLMs의 발전을 설명합니다. 특히, GPT와 BERT와 같은 초기 LLM들이 텍스트 이해 및 감정 인식 같은 문제에 초점을 맞추며, 가장 최신의 Gemma 모델이 벤치마킹 대상으로 비교됩니다. 연구진은 주로 LoRA 파인 튜닝 방법을 사용해 3-5 epoch 동안 모델을 훈련시킨 결과, 우크라이나어 멀티플 초이스 형식에 대한 이해도를 높였습니다.

- **Performance Highlights**: 새롭게 파인 튜닝된 LLMs은 기존의 BERT-유사 모델인 UAlpaca와 비교하여 우크라이나어 처리에서 상당한 향상을 보였습니다. UKID 데이터셋을 활용한 벤치마크 테스트에서, Gemma 모델은 다양한 NLP 작업에서 사용자 정의된 요구 사항에 더 잘 부합하는 것으로 나타났습니다. 이는 우크라이나어의 표현과 문화적 뉘앙스를 반영하는 데 더욱 효과적인 지원을 제공합니다.



### TLDR at SemEval-2024 Task 2: T5-generated clinical-Language summaries  for DeBERTa Report Analysis (https://arxiv.org/abs/2404.09136)
- **What's New**: 이 논문에서는 임상시험에 대한 자연어 추론 (Natural Language Inference for Clinical Trials, NLI4CT) 작업을 위한 새로운 방법론을 소개합니다. TLDR (T5-generated clinical-Language summaries for DeBERTa Report Analysis) 시스템은 T5 모델을 이용한 전제 요약을 통해 임상 NLI 업무의 모순과 포함 관계 분석을 향상시킵니다. 이 접근법은 소규모 컨텍스트 창과 길이가 긴 전제에 의해 발생하는 문제를 극복했으며, 이는 Macro F1 점수에서 0.184의 상당한 향상을 이끌었습니다.

- **Technical Details**: TLDR 프레임워크는 T5 모델을 사용하여 임상 보고서의 전제를 요약하고, 이 요약된 내용을 DeBERTa, Encoder-only 트랜스포머 (Transformer)를 사용하여 해당 선언문과 함께 분석합니다. 이 방법은 임상 NLI 작업의 전제 길이와 토큰 크기 제한의 도전을 관리하기 위해 혁신적인 기법을 활용합니다. 특히, T5 모델을 맞춤형으로 튜닝하여 '포함' 레이블이 붙은 쌍에 대해 요약을 생성하도록 훈련시키는 방법이 포함됩니다.

- **Performance Highlights**: TLDR 모델은 기존의 축약된 전제를 사용한 경우보다 Macro F1 점수에서 0.184 포인트 향상을 달성했으며, 추출적 요약 전제보다 0.046 포인트 높은 성능을 보였습니다. 또한 원본 텍스트와 의미적으로 변경된 입력에 대한 일관성과 신뢰도 측면에서도 강력한 성능을 보여 주었습니다. 이러한 결과는 임상 보고서의 NLI 작업에 있어서 TLDR 프레임워크의 강인성과 효과를 강조합니다.



### Unveiling LLM Evaluation Focused on Metrics: Challenges and Solutions (https://arxiv.org/abs/2404.09135)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs) 평가를 위한 측정 지표에 중점을 두고, 현재 사용 중인 지표들의 선택 및 해석에 대한 통찰을 제공합니다. 특히, 최근의 의학적 LLM들을 사용한 예시를 통해 이러한 지표들의 적용을 설명하고, 연구자들이 다양한 작업에 적합한 지표를 선택할 수 있도록 돕습니다.

- **Technical Details**: LLM의 평가 지표는 크게 세 가지 유형으로 구분됩니다. 첫 번째는 Multiple-Classification (MC) 지표, 두 번째는 Token-Similarity (TS) 지표, 마지막으로 Question-Answering (QA) 지표입니다. 각 지표는 특정 작업에 적합하도록 설계되었으며, 이들의 수학적 공식과 통계적 해석을 제공하여 LLM 연구자들이 평가에 활용할 수 있는 기준을 마련합니다.

- **Performance Highlights**: 이 연구는 LLM 평가에서 자주 무시되는 통계적 해석을 강조하며, 최신 생의학 LLM 사례를 통해 구체적인 적용 예를 보여줍니다. 또한, 이 논문은 도구와 기준 데이터셋을 포함한 평가 지표의 구현이 가능한 저장소(repositories)를 공개하여, 연구자들이 이전 모델들과의 일관성을 유지하면서 후속 연구를 수행할 수 있도록 지원합니다.



### When Hindsight is Not 20/20: Testing Limits on Reflective Thinking in  Large Language Models (https://arxiv.org/abs/2404.09129)
Comments: NAACL 2024 Findings paper (Camera-Ready Version)

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 자기성찰(self-reflection)을 활용할 때 진정한 인간 같은 반성 능력을 갖추고 있는지를 철저히 검증합니다. 특히 외부 피드백이나 여러 번의 반복적인 프롬프팅 없이 진행된 실험을 통해, 자기 반성이 진정으로 모델의 성능을 향상시킬 수 있는지 여부를 평가합니다.

- **Technical Details**: 연구팀은 단일 라운드 자기성찰 검증(Single-Round Self-Reflection Verification, SR2V)을 사용하여, 모델이 외부의 암시 없이도 자체 생성한 응답에 대해 반성할 수 있는 능력을 평가합니다. 이 방식에서는 모델에게 여러 응답 후보를 생성하도록 요청한 후, 각 응답에 대해 자기 비판을 생성하고, 이를 종합하여 최종 응답을 개선하도록 합니다. 실험은 TruthfulQA와 HotpotQA 데이터셋을 사용하여 수행되었습니다.

- **Performance Highlights**: 자기성찰을 통한 성능은 혼합 결과를 보여줍니다. TruthfulQA에서는 성능이 향상되는 반면, HotpotQA에서는 성능이 저하되었습니다. 이러한 결과는 모델이 초기에 제공하는 응답의 정확성과 질문의 난이도에 크게 의존한다는 것을 발견하였습니다. 성능이 향상된 주요 이유는 초기 모델 응답의 정확성이 낮을 때와 전체 질문의 난이도가 높을 때 자기성찰이 유리하기 때문입니다.



### Confidence Calibration and Rationalization for LLMs via Multi-Agent  Deliberation (https://arxiv.org/abs/2404.09127)
Comments: Accepted at ICLR 2024 Workshop on Reliable and Responsible Foundation Models

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 정확도와 교정을 개선하기 위한 새로운 접근법인 'Collaborative Calibration'을 제안합니다. 이는 여러 LLM 에이전트들이 모의 그룹 토론 과정에서 협력하여 교정하는 방법으로, 모델의 신뢰성을 높이는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 다양한 도메인의 생성적 QA(Question Answering) 작업에 'Collaborative Calibration'을 적용하였습니다. 기존의 단일 모델의 교정을 넘어서 여러 LLM 에이전트의 상호작용을 통해 교정을 개선하고자 합니다. 이 과정은 실시간 학습이 필요 없는 후처리(post-hoc) 전략으로, 툴(tool)-확장된 LLM 에이전트들을 활용합니다.

- **Performance Highlights**: 이 접근법은 더 정확한 신뢰도 평가와 모델 예측의 신뢰성 향상에 효과적임을 보여줍니다. 'Collaborative Calibration'는 다양한 교정 메트릭(calibration metrics)에서 이전 방법들과 비교하여 비슷하거나 더 높은 성능을 달성했으며, 정확도나 생성 품질을 해치지 않으면서도 광범위한 학습이나 매개변수 조정 없이도 가능했습니다.



### CuriousLLM: Elevating Multi-Document QA with Reasoning-Infused Knowledge  Graph Prompting (https://arxiv.org/abs/2404.09077)
- **What's New**: 이 연구에서는 복잡한 QA(Question Answering) 작업을 위한 더 나은 이유 제공 방법과 LLM(Large Language Models)의 인식 오류를 줄이기 위해 지식 그래프(Knowledge Graph)와 결합된 새로운 접근 방식인 Knowledge Graph Prompting (KGP) 프레임워크를 개선하였습니다. 마찬가지로 이 프레임워크의 비용과 대기 시간 문제를 해결하기 위해 이성적 추론을 갖춘 LLM 이동 에이전트를 도입하여 성능을 크게 향상시켰습니다.

- **Technical Details**: 이 연구에서 개발된 이성적 추론을 갖춘 LLM 에이전트는 인간의 호기심을 모방하여, 다단계 문서 검색을 위해 추가적인 질문을 생성합니다. 이 방법은 복잡한 질문 유형을 처리하기 위한 능력을 제고합니다. 이 에이전트는 문서 간의 연결과 차이점을 평가하는 데 필요한 논리적 사고를 반영합니다. 또한, Follow-upQA라는 새로운 데이터셋을 개발하여, 다문서 세트에 대한 평가 및 연구를 촉진합니다.

- **Performance Highlights**: 개발된 이성적 추론 LLM 에이전트는 전통적인 TF-IDF, BM25 및 심층 학습 기반 접근 방법(DL)과 비교하여 경쟁력 있는 성능을 보였습니다. 이 에이전트는 특히 다단계 QA 데이터셋에서 뛰어난 성능을 보여주었으며, 이성적 추론의 통합이 모델의 전반적인 효율성 및 정확도를 높였습니다.



### Multilingual Evaluation of Semantic Textual Relatedness (https://arxiv.org/abs/2404.09047)
Comments: 8 pages

- **What's New**: 이 논문은 다양한 언어에 걸쳐 문맥을 고려한 의미적 텍스트 관련성(Semantic Textual Relatedness, STR)을 탐색하며, 특히 영어, 마라티어, 힌디어, 스페인어를 중심으로 연구하였습니다. STR은 단순한 단어 겹침을 넘어서 언어적 요소와 비언어적 요인(주제, 감정, 관점 등)을 포함하여 텍스트 간의 더 깊은 연결을 포착합니다. 이는 정보 검색, 기계 번역 등 NLP 분야에서의 응용 가능성을 크게 확장합니다.

- **Technical Details**: 이 연구는 세 가지 학습 패러다임(지도 학습(Supervised Learning), 비지도 학습(Unsupervised Learning), 교차 언어 학습(Cross-lingual Learning))을 사용하여 다양한 언어 모델을 활용하였습니다. 특히 SemEval-2024 공유 작업을 통해 영어와 힌디어, 마라티어, 스페인어 데이터셋에서 STR 기술을 평가하였습니다. 사용된 주요 모델로는 서포트 벡터 회귀(Support Vector Regression, SVR)와 XGBoost, 그리고 문장 변환기 기반 모델(Sentence Transformer-based models)이 포함되어 있습니다.

- **Performance Highlights**: 제출된 여러 트랙에서 뛰어난 점수를 받았으며, 특히 지도 학습과 교차 언어 학습에서 유의미한 성과를 보였습니다. 이는 다양한 언어에 대한 STR의 효과적인 접근 방식을 입증하며, 저자원 언어에 대한 추가 연구를 촉진하는 계기가 되었습니다. 해당 결과는 SemRel2024 Dataset을 사용하여 얻은 것으로, 0에서 1 사이의 의미 유사도 점수를 기반으로 합니다.



### Adapting Mental Health Prediction Tasks for Cross-lingual Learning via  Meta-Training and In-context Learning with Large Language Mod (https://arxiv.org/abs/2404.09045)
- **What's New**: 이 연구는 아프리카의 저자원 언어, 스와힐리어로 소셜 미디어 데이터에서 정신 건강 상태를 예측하는 새로운 방법을 제안합니다. 메타-러닝(model-agnostic meta-learning)과 대규모 언어 모델(LLMs)을 활용한 두 가지 접근 방식을 소개하며, 스트레스, 우울증, 우울증의 심각성, 자살 사고 예측 등 네 가지 정신 건강 과제에 적용됩니다.

- **Technical Details**: 이 연구는 메타-러닝과 자기감독(self-supervision)을 결합하여 모델 초기화를 개선하고 빠른 적응과 교차 언어 전이(cross-lingual transfer)를 가능하게 합니다. 또한, 대규모 언어 모델(LLMs)의 인-콘텍스트 학습(in-context learning) 능력을 활용하여 스와힐리어 정신 건강 예측 작업에서의 성능을 평가하였습니다. 연구에서는 스와힐리어 프롬프트가 교차 언어 프롬프트보다 성능이 우수하나 영어 프롬프트보다는 떨어지는 것으로 나타났습니다.

- **Performance Highlights**: 메타-학습을 사용한 모델은 기존의 파인튜닝(fine-tuning) 방법보다 우수한 성능을 보여, 매크로 F1 점수(macro F1 score)에서 XLM-R보다 18%, mBERT보다 0.8% 높은 성과를 달성하였습니다. 이는 저자원 언어에 대한 효과적인 정신 건강 예측 모델의 교차 언어 학습 및 적응 가능성을 입증합니다.



### Do LLMs Play Dice? Exploring Probability Distribution Sampling in Large  Language Models for Behavioral Simulation (https://arxiv.org/abs/2404.09043)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM: Large Language Models)이 인간의 순차적 의사 결정 과정(MDP: Markov decision-making processes)을 모방하는 데 사용될 때 확률 분포를 이해하고 샘플링하는 능력을 탐구합니다. 연구는 알려진 확률 분포와 알려지지 않은 확률 분포의 두 가지 상황에서 LLM 에이전트의 성능을 분석합니다.

- **Technical Details**: 연구는 두 가지 주요 실험을 통해 수행되었습니다. 첫 번째 실험에서는 LLM에게 정확한 확률 분포를 제공하고 이를 바탕으로 샘플링하도록 요청하며, 두 번째 실험에서는 온라인 소셜 네트워크에서 활동 수준을 변경하여 LLM이 어떻게 행동 시퀀스 뒤에 있는 확률 분포를 추론하는지 분석합니다. 또한, 프로그래밍 도구(programming tools)를 사용하여 샘플링 과정을 돕는 방법이 제안되었습니다.

- **Performance Highlights**: 결과적으로, LLM 에이전트는 알려진 확률 분포에 대해서는 일정 수준의 이해를 보였으나 샘플링 성공률이 낮았습니다. 알려지지 않은 확률 분포의 경우, LLM 에이전트는 행동 시퀀스가 따르는 확률 분포를 효과적으로 추론하는 데 실패했습니다. 이는 인간 행동을 시뮬레이션하기 전에 LLM을 사용할 때 신중한 고려가 필요함을 시사합니다.



### MING-MOE: Enhancing Medical Multi-Task Learning in Large Language Models  with Sparse Mixture of Low-Rank Adapter Experts (https://arxiv.org/abs/2404.09027)
Comments: 15 pages, 3 figures

- **What's New**: 이 논문에서는 의료 분야의 다양하고 복잡한 작업을 처리할 수 있는 새로운 의료 대형 언어 모델인 MING-MOE를 소개합니다. 기존 모델과 다르게 MING-MOE는 특정 작업에 대한 주석(annotations) 없이도 효과적으로 작동할 수 있으며, 이를 통해 데이터 세트가 광범위하게 적용되며 사용성이 향상됩니다. MING-MOE는 여러 전문가(Mixture of Experts, MoE)의 조합과 MoLoRA 기법을 사용하여 효율적인 매개변수 사용을 가능하게 합니다.

- **Technical Details**: MING-MOE는 MoE 구조를 사용하여 다양한 입력을 처리하기 위해 다수의 전문가 모듈을 도입하였습니다. 각 토큰에 대해 적절한 전문가를 선택할 수 있는 토큰 수준의 MoE 아키텍처를 사용하여 특정 작업 유형을 지정할 필요가 없습니다. MoLoRA(Mixture of Low-Rank Adaptation) 접근 방식을 통해 베이스 모델의 기존 매개변수를 동결하고 최소한의 훈련 가능 매개변수만을 사용하여 MOE 모델을 미세 조정(fine-tune)합니다. 이는 적응 비용을 감소시키는 동시에 효율성을 증가시킵니다.

- **Performance Highlights**: MING-MOE는 20개 이상의 의료 작업에 대해 최고의 성능(state-of-the-art, SOTA)을 달성하였습니다. 이는 기존에 사용 가능한 오픈 소스 의료 모델들과 비교하여 높은 우수성을 보여주며, 이는 특히 의료 자연어 처리(Natural Language Processing, NLP) 작업 및 의료 면허 시험에서 그 성능을 입증합니다.



### WikiSplit++: Easy Data Refinement for Split and Rephras (https://arxiv.org/abs/2404.09002)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 논문에서는 복잡한 문장을 단순한 문장으로 분할하고 다시 구성하는 작업을 개선시키기 위해 WikiSplit++라는 새로운 데이터셋을 제안하고 있습니다. 데이터 정제(data refinement) 접근 방식을 사용하여 WikiSplit 데이터셋을 개선하고, 본 연구에서는 WikiSplit++ 데이터셋이 훈련 시 더 나은 성능을 보임을 입증하고 있습니다. 특히 복잡한 문장에서 발생할 수 있는 환각(hallucinations) 문제와 문장 분할의 정확성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 본 논문에서 제안하는 데이터셋 WikiSplit++는 기존 WikiSplit 데이터셋에서 복잡한 문장과 그에 상응하는 단순한 문장들 사이에 모순이 있는 경우를 제거하고, 참조되는 단순 문장들의 순서를 뒤집는 방식으로 만들어집니다. 이러한 접근은 자연어 추론(Natural Language Inference, NLI) 분류기를 사용하여 실현되며, 이는 모델이 텍스트 생성 시 더욱 신뢰도 높은 결과를 도출하도록 합니다. 또한, T5모델을 WikiSplit++로 미세 조정(fine-tuned)하여 실험한 결과, 기존 WikiSplit을 사용했을 때보다 더 적은 환각을 생성하고, 더 많은 문장 분할을 실현할 수 있었다는 점이 특징입니다.

- **Performance Highlights**: WikiSplit++ 데이터셋을 사용하여 훈련된 모델은 WikiSplit을 사용한 경우보다 뛰어난 성능을 보였습니다. 특히, 문장 분할의 개수가 증가하고, 환각 발생 비율이 감소하여 결과의 정확성이 크게 향상되었습니다. 이러한 성과는 특히 데이터셋의 질을 개선함으로써 자연어 처리(NLP)에서 중요한 향상을 이끌어낼 수 있음을 시사합니다.



### Labeled Morphological Segmentation with Semi-Markov Models (https://arxiv.org/abs/2404.08997)
Comments: CoNLL 2015

- **What's New**: 이 연구는 레이블 부착된 형태론적 세분화(Labeled Morphological Segmentation, LMS)을 소개하며, 이는 형태론적 처리의 여러 작업을 통합하는 새로운 관점을 제공합니다. 이를 통해 형태론적 태그 분류(Morphological Tag Classification), 어간 추출(Stemming), 그리고 통상적 형태 분할(Morphological Segmentation) 등의 작업을 통합 관리할 수 있습니다. 또한, 새로운 계층적 형태소 분석 태그 집합을 도입하며, Chipmunk라는 시스템을 개발하여 형태 태스크(Morphotactics)를 명시적으로 모델링합니다.

- **Technical Details**: Chipmunk는 반지도 마르코프 조건부 랜덤 필드(semi-Markov Conditional Random Field, semi-CRF)를 기반으로 한 감독받는 모델입니다. 이 모델은 세분화와 시퀀스 라벨링을 함께 모델링하며, 레이블이 있는 형태론적 분할을 직접 모델링하고 다양한 계층의 형태소 분석 태그 집합을 사용합니다. LMS는 기존의 단순한 형태소 분할보다 더 높은 정밀도를 제공하며, 이는 다양한 언어 과제에 사용될 수 있습니다.

- **Performance Highlights**: Chipmunk는 6개 언어에 대해 세 가지 주요 태스크에서 향상된 성능을 보여줍니다. 특히 형태론적 분할에서는 기존 베이스라인 대비 2-6 포인트 F1 점수의 절대적인 향상을 달성하였습니다. 이는 Chipmunk가 다양한 언어의 복잡한 형태 구조를 효과적으로 처리할 수 있는 능력을 시사합니다.



### RoNID: New Intent Discovery with Generated-Reliable Labels and  Cluster-friendly Representations (https://arxiv.org/abs/2404.08977)
Comments: DASFAA 2024

- **What's New**: 새로운 의도 발견(NID)을 위한 로버스트 프레임워크인 RoNID(Robust New Intent Discovery)가 소개되었습니다. 이 방법은 신뢰할 수 있는 가짜 레이블(pseudo-label) 생성과 클러스터 친화적인 표현 학습을 통해 NID 문제에 접근하며, 이는 최적 수송 문제(optimal transport problem)를 해결하여 높은 품질의 지도 신호를 제공하는 최초의 시도로 기존 방법들과 차별화됩니다.

- **Technical Details**: RoNID는 두 가지 주요 모듈로 구성됩니다: 신뢰할 수 있는 가짜 레이블 생성 모듈과 클러스터 친화적인 표현 학습 모듈. 가짜 레이블 생성 모듈은 이동 최적화 문제를 해결하여 신뢰할 수 있는 합성 레이블을 할당하며, 표현 학습 모듈은 E-step(Expectation Step)과 M-step(Maximization Step)에서 내부 클러스터(compactness)와 클러스터 간의 분리(separation)를 강화하는 대조 학습(contrastive learning)을 결합하여 사용합니다.

- **Performance Highlights**: 실험 결과는 다양한 벤치마크에서 기존의 최첨단 기술 대비 평균 1.5% 개선된 새로운 최고 성능을 달성함을 보여줍니다. RoNID는 반복적으로 수행되어 신뢰할 수 있는 가짜 레이블과 클러스터 친화적인 표현을 최종적으로 얻을 수 있으며, 이는 모델의 전반적인 성능을 눈에 띄게 향상시킵니다.



### OOVs in the Spotlight: How to Inflect them? (https://arxiv.org/abs/2404.08974)
Comments: To be published in LREC-COLING 2024. 12 pages, 3 figures

- **What's New**: 본 연구는 새로운 시스템을 개발하여 문법 변화 과제를 해결하고, 특히 OOV(out-of-vocabulary) 상태에서 체코어 명사의 형태변화를 추측하는 Python 라이브러리를 제공합니다. 체코어는 다른 언어에 비해 복잡한 형태학적 구조를 가지고 있는데, 이번에는 체코어에 초점을 맞추어 새로운 OOV 주어진 상태에서의 평가용 데이터셋을 개발했습니다. 이는 체코어에 특화된 첫 번째 OOV 데이터셋으로 알려져 있습니다.

- **Technical Details**: 세 가지 다른 시스템이 개발되었습니다; 첫 번째는 사전 기반의 retrograde 모델이며, 나머지 두 시스템은 LSTM과 Transformer를 기반으로 한 sequence-to-sequence(seq2seq) 모델입니다. 이 시스템들은 seq2seq 아키텍처가 적용되어 있으며, 특히 체코어과 같이 형태학적으로 풍부한 언어에서 효과적으로 작동하도록 설계되었습니다. 또한 OOV 상태에 맞추어 시스템들을 조정하고 광범위한 테스트를 수행했습니다.

- **Performance Highlights**: 표준 OOV 조건에서 Transformer가 가장 우수한 성과를 보였으며, LSTM, retrograde 모델 및 SIGMORPHON 기준선과 함께 앙상블을 구성할 때 성능이 향상되었습니다. 실세계 OOV 데이터셋에서는 retrograde 모델이 모든 신경 모델을 능가했습니다. 추가적으로, 이 seq2seq 모델은 SIGMORPHON 2022 공유 작업 데이터에서 16개 언어 중 9개 언어에서 OOV 평가(feature overlap)에서 최신 기술 결과를 달성했습니다. 이 연구는 체코어 OOV 형태변화 데이터셋을 공개하며, 해당 데이터셋을 사용하여 OOV 조건에서 철저한 평가를 수행할 수 있습니다.



### Multimodal Cross-Document Event Coreference Resolution Using Linear  Semantic Transfer and Mixed-Modality Ensembles (https://arxiv.org/abs/2404.08949)
Comments: To appear at LREC-COLING 2024

- **What's New**: 사건 코어퍼런스 해상도(Event Coreference Resolution, ECR)에 초점을 맞춘 새로운 다중 모달 접근 방식을 제안하였습니다. 이 연구에서는 비단 문서 내 사건이 아니라, 다수의 문서에 걸쳐 동일한 사건을 언급하는 것을 연결하는데 이미지와 텍스트 모두 활용하였습니다. 연구 팀은 ECB+ 데이터셋을 확장하여 사건 중심 이미지를 포함시키고, 이미지 확산 모델을 사용하여 이미지를 생성함으로써 데이터 세트에 다양성을 추가하였습니다.

- **Technical Details**: 이 연구는 시각 및 언어 모델 간의 간단한 선형 매핑(Linear Mapping)을 통해 시각적 및 텍스트 단서를 통합하는 혁신적인 다중 모달 코어퍼런스 방법을 사용합니다. 구체적으로는, 이벤트가 언급된 메타데이터를 활용하여 그와 관련된 이미지를 찾아내고, 최신 이미지 확산 모델을 통해 이벤트 중심 이미지를 생성합니다. 세 가지 주요 방법론에는 1) 파인튜닝이 추가된 표준 융합 모델, 2) 파인튜닝 없는 새로운 선형 매핑 방법, 3) 의미론적 및 담화 수준의 어려움에 따라 언급 쌍을 분할하는 앙상블 접근법이 포함됩니다.

- **Performance Highlights**: 제안된 시스템은 ECB+ 데이터셋에서 91.9 CoNLL F1의 최고 성능을 달성하였으며, AIDA Phase 1 데이터셋에 대해서도 새로운 기준점을 설정하였습니다. 이 결과는 특히 도전적인 코어퍼런스 문제에 대해 다중 모달 정보의 유틸리티를 입증하며, 코어퍼런스 해상도 영역에서 다중 모달 자원의 필요성을 강조합니다.



### Enforcing Paraphrase Generation via Controllable Latent Diffusion (https://arxiv.org/abs/2404.08938)
- **What's New**: 본 논문에서는 주어진 텍스트의 고품질 및 다양한 (diverse) 표현을 생성하는 패러프레이즈 생성을 위해 새로운 모델인 Latent Diffusion Paraphraser (LDP)를 제안함. LDP는 학습된 잠재 공간을 바탕으로 제어 가능한 확산 과정(diffusion process)을 모델링하여, 기존의 확산 모델(diffusion model)의 절단(truncation) 문제를 해결하고, 외부 특징(external features) 없이 입력 세그먼트만을 이용하여 의미론적(paraphrase semantics) 정확성을 보장함으로써 효율성과 결과를 향상시킴.

- **Technical Details**: LDP는 인코더-디코더(encoder-decoder) 프레임워크의 잠재 공간(latent space)을 활용하고, 연속적인 확산 과정을 이산 텍스트(discrete texts)와 연결함으로써, 기존의 확산 모델에서 요구되는 중간 라운딩(rounding) 과정 없이 분산 생성 효율을 증진시킴. 또한, 외부 특성을 사용하지 않고 입력 세그먼트만을 활용하여 의미를 강화함으로써 다양한 데이터셋에서 우수한 성능을 달성함.

- **Performance Highlights**: LDP는 다양한 데이터셋에서 비교 모델들(diffusion counterparts)보다 더 나은 성능과 속도로 패러프레이징을 생성함. 또한 이 접근 방식은 문제 생성(question generation) 및 도메인 적응(domain adaptation)과 같은 유사한 텍스트 생성 시나리오에서도 유용함을 입증함. LDP는 '라운딩' 과정을 회피함으로써 생산성과 효율성을 크게 향상시킴.



### Towards Enhancing Health Coaching Dialogue in Low-Resource Settings (https://arxiv.org/abs/2404.08888)
Comments: Accepted to the main conference of COLING 2022

- **What's New**: 이 논문에서는 환자와 대화하여 신체 활동 목표를 설정하고 달성하는 데 도움을 주는 대화형 시스템을 제안합니다. 특히, 감정 이해와 공감적 반응 생성(empathetic response generation)에 초점을 맞추어 환자의 감정에 민감하게 반응할 수 있는 모듈화된 건강 코칭 대화 시스템(modularized health coaching dialogue system)을 구축하는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 간소화된 자연어 이해(NLU, Natural Language Understanding) 및 자연어 생성(NLG, Natural Language Generation) 프레임워크를 포함하고 있습니다. 시스템은 NLU 모듈, NLGhc 모듈(health coaching을 위한 NLG), 그리고 NLGemp 모듈(공감적 반응을 생성하는 NLG)로 구성됩니다. 이 시스템은 공감(empathy)을 조건으로 하는 반응 생성 메커니즘을 채택하여 보다 인간적인 상호작용을 가능하게 합니다. 또한, 실제 건강 코칭 데이터에 기반하여 목표 설정 및 달성을 위한 특정 단계(phase)를 정의하고 이를 추적하는 기능이 포함되어 있습니다.

- **Performance Highlights**: 자동 평가와 전문가 기반 인간 평가를 통해, 이 시스템은 기존의 최고 수준(state-of-the-art) NLU 시스템보다 10% 이상 높은 F1 점수를 달성하였으며, 목표 속성 추적(goal attributes tracking) 작업에서 의미 구조 정확성(semantic frame correctness)에서도 7% 이상 향상되었습니다. 또한, 대화 생성(dialogue generation) 측면에서 일관성(coherence), 유창성(fluency), 공감성(empathy) 면에서 기준 모델보다 뛰어난 성능을 보였습니다. 이러한 결과는 이 시스템이 건강 코칭 효율성을 개선하고, 코칭 과정에서 인간 코치의 역할을 보완할 수 있는 유망한 도구가 될 수 있음을 시사합니다.



### LLM In-Context Recall is Prompt Dependen (https://arxiv.org/abs/2404.08865)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 'in-context recall performance'를 평가하였습니다. 이 분석은 'needle-in-a-haystack' 방법론을 사용하며, 텍스트 블록('haystack') 내에 포함된 사실('needle')을 모델이 검색할 수 있는 능력을 평가합니다. 연구에서는 다양한 'haystack' 길이와 'needle' 배치에서 각 모델의 성능을 측정하여 그 패턴을 확인했습니다.

- **Technical Details**: 연구는 Llama 2, GPT-4 Turbo, Gemini 1.5 등의 다양한 LLM을 포함하여 9개 모델에 대해 실시되었습니다. 이들 모델은 'context window'의 크기가 서로 다릅니다. 예를 들어, GPT-4 Turbo는 128,000개의 토큰을, Gemini 1.5는 10M 토큰을 처리할 수 있는 'context window'를 가지고 있습니다. 모델들은 다양한 'haystack' 길이와 'needle' 위치에서 테스트되었으며, 성능 패턴을 시각적으로 분석하기 위해 히트맵(heatmaps)이 사용되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, LLM의 recall 성능은 프롬프트의 내용에 영향을 받으며, 훈련 데이터와 다른 정보가 포함된 프롬프트에서 성능이 저하될 수 있습니다. 그러나 모델 아키텍처(model architecture), 훈련 전략(training strategy), 또는 파인 튜닝(fine-tuning)을 조정함으로써 성능을 향상시킬 수 있습니다. 이러한 발견은 LLM을 실제 애플리케이션에 보다 효과적으로 적용하기 위한 방향을 제공합니다.



### On Speculative Decoding for Multimodal Large Language Models (https://arxiv.org/abs/2404.08856)
Comments: Accepted as a spotlight paper to ELVM workshop at CVPR 2024

- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs, 멀티모달 대형 언어 모델)의 추론 효율성을 개선하기 위해 추론 과정에 speculative decoding (추측 디코딩)을 적용하는 새로운 접근 방식을 소개합니다. 특히, LLaVA 7B 모델을 대상으로 하여, 이미지 토큰이 필요 없는 언어 전용 모델을 사용하여 speculative decoding을 구현하였습니다.

- **Technical Details**: 이 논문에서는 115M 매개변수 언어 모델을 사용하여, LLaVA 7B 모델의 추론 속도를 최대 2.37배까지 향상시킬 수 있는 기법을 보여줍니다. 사용된 언어 모델은 이미지 토큰과 관련 처리 구성 요소를 고려하지 않아 기존보다 더 효율적인 추론이 가능합니다. 또한, 이미지 어댑터를 포함하는 소형 LLaVA draft model을 소개하여, COCO captioning 및 ScienceQA 작업에서 성능을 소폭 향상시키면서 다른 작업에서 비교 가능한 결과를 유지합니다.

- **Performance Highlights**: 개발된 시스템은 LLaVA Instruct 150K dataset, COCO dataset 및 ScienceQA dataset에서 다양한 태스크를 수행하면서 최대 2.37배의 메모리 제한 속도 향상을 달성했습니다. 이는 이미지 토큰을 고려할 필요가 없는 언어 모델만으로도 효율적인 추론이 가능함을 시사합니다.



### BERT-LSH: Reducing Absolute Compute For Attention (https://arxiv.org/abs/2404.08836)
Comments: 10 pages, 5 figures

- **What's New**: 이 연구에서는 BERT 아키텍처에서 주의(attention) 메커니즘을 근사하기 위해 지역 민감 해싱(Locality Sensitive Hashing, LSH)을 통합한 새로운 BERT-LSH 모델을 소개합니다. BERT-LSH는 계산 효율성을 비약적으로 향상시키며, 예상치 못하게도 기존 BERT 모델보다 성능이 뛰어납니다. 이는 LSH 기반의 주의 메커니즘이 단순히 계산상의 이점을 제공할 뿐만 아니라, 모델이 훈련 데이터를 일반화하는 능력을 향상시킬 수 있음을 시사합니다.

- **Technical Details**: BERT-LSH 모델은 레포머(Reformer)와 달리 BERT의 원래 아키텍처에서 쿼리(Q)와 키(K) 행렬을 독립적으로 유지하면서, 이 행렬들에 LSH를 적용합니다. 이 방법은 BERT의 표현력 있는 주의 메커니즘을 보유하면서도 LSH의 계산 효율성을 이용합니다. 또한, 다중 해시 함수를 이용하여 비슷한 벡터들이 서로 충돌할 확률을 높입니다. 구현은 벡터화 된 방식으로 진행되어, 해시 테이블을 통해 벡터를 버킷에 매핑하고, 충돌 행렬을 구성하여 쿼리 및 키 벡터 간의 잠재적인 연관성을 나타냅니다.

- **Performance Highlights**: BERT-LSH는 기본 BERT 모델과 비교했을 때 자체 주의 계층의 계산 요구를 크게 줄이면서도 사전 훈련 및 미세 조정 작업에서 더 우수한 성능을 보여주었습니다. 이는 특히 자원이 제한된 환경에서 강력한 NLP 모델을 배포해야 하는 응용 프로그램에 유리합니다. 연구 결과는 BERT-LSH가 성능을 유지하면서 계산상의 이점을 제공함으로써, 보다 확장 가능하고 실용적인 NLP 솔루션 개발에 기여할 수 있음을 보여줍니다.



### Constrained C-Test Generation via Mixed-Integer Programming (https://arxiv.org/abs/2404.08821)
Comments: Github: this https URL

- **What's New**: 이 연구는 C-Test 생성을 위한 새로운 방법을 제안합니다. 기존의 연구들이 간격 크기(gap size)나 배치(gap placement)만을 달리하여 지역적으로 최적화된 해결책을 도출한 것과 달리, 이 연구에서는 혼합 정수 프로그래밍(Mixed-Integer Programming, MIP) 접근 방식을 사용하여 간격 크기와 배치를 동시에 고려함으로써 전역적으로 최적화된 솔루션을 도출할 수 있게 되었습니다. 또한, 최신 공백 난이도 예측 모델을 최적화 문제에 직접 통합할 수 있습니다.

- **Technical Details**: MIP는 C-Test에서의 간격 크기와 위치를 동시에 최적화하여 고려하고, 최신의 공백 난이도 예측 모델을 통합하기 때문에, 단순히 간격을 조작하는 것보다 더 정교하고 효과적인 시험을 생성할 수 있습니다. 이 방법은 언어 모델 GPT-4와 함께 사용되어, 32개의 영어 C-Test를 생성했으며 각각 20개의 간격이 있습니다 (총 3,200개의 갭 반응).

- **Performance Highlights**: 40명의 참가자를 대상으로 한 사용자 연구에서 MIP 방식은 두 가지 기준 방법론(간격 배치와 GPT-4에 기반한 방식)보다 훨씬 우수한 성능을 보였으며, 세 번째 기준(간격 크기에 기반한 방식)과 동등한 성능을 보였습니다. 분석 결과, GPT-4는 여전히 명시적인 제약 조건을 충족하는 데 어려움이 있었지만, MIP가 생성한 C-Test가 인식된 난이도와 가장 잘 상관관계를 갖는 것으로 나타났습니다. 연구 코드, 모델, 수집된 데이터는 오픈 소스 라이선스 하에 공개되었습니다.



### Revisiting Code Similarity Evaluation with Abstract Syntax Tree Edit  Distanc (https://arxiv.org/abs/2404.08817)
- **What's New**: 이 논문은 최근 코드 유사도 평가 메트릭스에 대해 재조명하고 있으며, 특히 다양한 프로그래밍 언어에 대한 추상 구문 트리(AST: Abstract Syntax Tree) 편집 거리의 응용을 중점적으로 탐구합니다. 특히, 기존의 순차적 유사성 메트릭스와 비교하여 이 메트릭스들의 유용성을 탐구하고 있으며, Tree Similarity of Edit Distance (TSED) 메트릭의 새로운 개선 버전을 제안하고 최적화하여 발표하였습니다.

- **Technical Details**: 이 연구에서는 GPT와 TSED라는 두 가지 다른 메트릭스를 사용하여 다양한 프로그래밍 언어의 구조적 유사성을 평가하고, 이 메트릭스들이 BLEU 점수와 같은 의미적 유사성 메트릭스와 어떤 상관관계가 있는지를 연구합니다. TSED 메트릭의 구현을 위해 AST 파싱과 트리 편집 거리 계산 기능이 사용되며, 이는 코드의 복잡성을 고려하여 정규화된 점수를 제공합니다. 또한 GPT 모델을 프롬프트하여 코드의 구조적 유사성에 대한 점수를 생성하는 방법을 사용합니다. 이는 GPT 모델이 특정 계산을 수행하는 방식을 명확히 하지 않는 '블랙 박스'로 작동하며, 결과의 안정성에 대한 이슈도 존재합니다.

- **Performance Highlights**: 실험 결과, TSED와 GPT 유사성은 모든 테스트된 6개 프로그래밍 언어에서 MBXP 데이터셋을 평가하는 데 효과적이었으며, 파싱이나 점수 생성 실패는 관찰되지 않았습니다. TSED의 SQL 외의 언어로의 적용 가능성도 보여주며, Java 및 Kotlin 같은 언어에서의 코드 분석 가능성을 시사하고 있습니다.



### Evaluating the Quality of Answers in Political Q&A Sessions with Large  Language Models (https://arxiv.org/abs/2404.08816)
- **What's New**: 이 연구는 캐나다 하원의 질의응답 시간(Question Period)에서 답변의 질을 평가하는 새로운 접근 방법을 제안합니다. 답변의 질을 원래 질문을 정확하게 추론할 수 있는 정도로 측정하며, 이는 답변의 적절성을 내포합니다.

- **Technical Details**: 연구진은 large language model을 fine-tuning하여 질문과 답변의 코퍼스(corpus)를 활용하고, 이를 통해 답변의 질을 측정하는 방식을 실현했습니다. 특히, 'Sentence-BERT'로 알려진 변형된 BERT(Bidirectional Encoder Representations from Transformers) 모델을 활용하였습니다. 이 모델은 문장의 전체적인 뉘앙스를 반영하는 문장 임베딩(sentence embeddings)을 생성하여 의미 검색(semantic search)에서 우수한 성능을 발휘합니다.

- **Performance Highlights**: 분석 결과, 응답의 질은 질문을 한 의원의 정당 소속에 따라 유의미한 차이를 보였으며, 질문의 주제와 답변의 질 사이에도 의미있는 상관관계가 있음을 발견했습니다. 이러한 발견은 캐나다 하원에서의 정책적 토론의 질과 효과성을 이해하는 데 중요한 통찰을 제공합니다.



### CreativEval: Evaluating Creativity of LLM-Based Hardware Code Generation (https://arxiv.org/abs/2404.08806)
- **What's New**: 이 연구는 하드웨어 디자인 생성에 대한 대규모 언어 모델(Large Language Models, LLMs)의 창의성을 평가하는 새로운 프레임워크인 CreativeEval을 제시합니다. 이전 연구들이 기능적 정확성에만 초점을 맞춘 것과 달리, 이 프레임워크는 창의적 측면을 중요시하며, 다양한 측정 기술을 통해 LLMs의 창의성을 정량화합니다.

- **Technical Details**: CreativeEval 프레임워크는 유창성(fluency), 유연성(flexibility), 독창성(originality), 및 상세화(elaboration)의 네 가지 창의적 하위 구성요소를 정량화합니다. 다양한 프롬프팅(prompting) 기법과 후처리(post-processing) 기술이 사용되어 LLMs가 생성한 하드웨어 디자인의 창의성을 평가합니다.

- **Performance Highlights**: 이 평가에서는 여러 인기 있는 LLM들(예: GPT 모델, CodeLlama, VeriGen)이 창의성 지표에서 경쟁하였고, GPT-3.5가 하드웨어 디자인 생성에서 가장 창의적인 모델로 나타났습니다.



### The Generation Gap:Exploring Age Bias in Large Language Models (https://arxiv.org/abs/2404.08760)
Comments: 4 pages

- **What's New**: 이 논문에서는 세계 가치 조사(World Value Survey) 데이터를 활용하여 대형 언어 모델(LLMs)의 가치가 특정 연령대와 어떻게 부합하는지 탐구합니다. 기존의 연구에서 주로 다루어지지 않은 연령과의 가치 차이를 최소화하는 방법을 조사하여, 다양한 연령층이 디지털 제품과 소통하는 데 있어서 좀 더 나은 경험을 제공하고자 합니다.

- **Technical Details**: 연구자들은 대형 언어 모델들(GPT-3.5-turbo, GPT-3.5-turbo-instruct, FLAN-T5XXL, FLAN-UL2)을 이용하여 13가지 범주에 걸친 가치 반응을 유도하고, 8가지 다양한 형식의 프롬프트(prompt) 변화를 사용해 모델의 반응을 테스트했습니다. 또한, 연령 및 국가 정보를 프롬프트에 포함시키는 실험을 통해 그 영향을 관찰했습니다. 그 결과, 연령 정보를 포함한 프롬프트가 연령별 가치 불일치를 완전히 해결하지 못하는 경우가 다수 발견되었습니다.

- **Performance Highlights**: 연구 결과는 대형 언어 모델들이 일반적으로 젊은 사용자들의 가치관과 더 일치하는 경향이 있다는 것을 보여줍니다. 예를 들어, 그림 1(Fig 1)과 같이 분석된 LLM들은 다수 연령층에서 보여주는 가치와는 달리 젊은 층의 가치와 더욱 일치하였습니다. 이러한 발견은 대형 언어 모델의 연령 편향 및 인구통계적 특성에 대한 인식을 높이는 데 중요한 기여를 합니다.



### Introducing L2M3, A Multilingual Medical Large Language Model to Advance  Health Equity in Low-Resource Regions (https://arxiv.org/abs/2404.08705)
- **What's New**: 이 연구는 저소득 및 중산층 국가(LMICs)의 보건 노동자 부족 문제를 해결하기 위해 대규모 언어 모델(LLMs)과 기계 번역 모델을 통합하여 사회 보건 노동자(CHWs)의 언어 장벽, 문화적 민감성, 의료 대화 데이터셋의 제한을 극복하는 혁신적 접근 방식을 소개합니다. 이 모델은 높은 수준의 번역 능력을 자랑하며, 오픈 소스 데이터셋에서의 철저한 파인 튜닝(fine-tuning)을 통해 의료 정확성을 보장하고 잘못된 정보의 위험을 방지하기 위한 종합적인 안전 기능을 갖추고 있습니다.

- **Technical Details**: Uheal L2M3 모델 시스템은 복잡한 의료 용어와 약어를 피하고, 세계 보건 기구(WHO)의 지속 가능한 발전 목표(Sustainable Development Goals, SDGs)에 맞춰 주요 건강 문제를 해결하도록 설계되었습니다. 디자인은 모듈화 및 확장성을 갖추고 있어 다양한 지리적 및 문화적 상황에 손쉽게 적응할 수 있습니다. 이 모델은 특히 930만 개 토큰에서 수집된 포괄적인 의료 코퍼스를 사용하여 도메인 적응형 파인 튜닝 방식을 활용하고 평가 단계에서 모델의 성능을 철저히 검증합니다.

- **Performance Highlights**: 이 접근 방법은 CHWs에게 문맥적으로 적절한 의료 지식과 진단 도구를 제공함으로써 의료 서비스의 접근성과 질을 크게 향상시킵니다. 또한, 문화적 민감성과 현지화를 우선시하여 내용을 현지 언어, 관습 및 보건 신념에 맞게 사용자 정의하여 문화적으로 관련성 있고 효과적인 개입을 촉진하고 건강 결과를 개선합니다.



### MM-PhyQA: Multimodal Physics Question-Answering With Multi-Image CoT  Prompting (https://arxiv.org/abs/2404.08704)
- **What's New**: 새로운 멀티모달(Multimodal) 물리학 문제 데이터셋 MM-PhyQA를 개발하였다. 이 데이터셋은 다양한 물리학 문제를 포함하여 복잡한 고등학교 수준의 물리 문제에 초점을 맞추고 있다. 또한, Multi-Image Chain-of-Thought (MI-CoT) 프롬프팅 기술을 도입하여, 문제 해결 과정에서 여러 이미지를 활용할 수 있는 새로운 방법을 제시한다.

- **Technical Details**: LLaVA, LLaVA-1.5 그리고 GPT-4와 같은 대형 언어 모델(Large Language Models)을 사용하여 Zero-shot 예측을 시행하였다. 특히, MM-PhyQA 데이터셋을 바탕으로 LLaVA-1.5 모델을 파인튜닝(fine-tuning)하여 멀티모달 입력 처리 능력을 향상시켰다. CoT-Prompting 기술을 사용하여 모델이 문제를 해결하기 위해 필요한 중간 단계를 생각하도록 유도하였으며, MI-CoT 방법론을 통해 여러 이미지를 프롬프팅 과정에 통합하였다.

- **Performance Highlights**: LLaVA-1.5 13b 모델은 MI-CoT 훈련 기법을 사용하여 테스트 데이터셋에서 가장 높은 정확도 71.65%를 달성하였다. 이는 멀티모달 문제에 대한 대형 언어 모델의 유효성을 입증하며, 특히 물리 문제 해결에 효과적임을 보여준다.



### Is Your LLM Outdated? Benchmarking LLMs & Alignment Algorithms for  Time-Sensitive Knowledg (https://arxiv.org/abs/2404.08700)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)이 시간에 따라 변화하는 사실(factual) 지식을 얼마나 잘 유지하는지에 초점을 맞추고 있습니다. 특히, 이 연구는 LLM에서 구식 지식을 식별하는 새로운 방법론인 동적 지식 벤치마크(Dynamic Knowledge Benchmark, DyKnow)를 제시하며, 이를 이용하여 18개의 LLM을 평가하고 지식 편집 방법의 효과성을 검증합니다. 동적 벤치마크는 실시간으로 Wikidata로부터 정보를 검색하여 최신 상태의 진실값을 제공합니다.

- **Technical Details**: DyKnow는 정치, 스포츠 및 조직과 같은 다양한 영역에서 LLM의 시간에 민감한 지식을 평가하기 위해 설계되었습니다. 이 벤치마크는 단순히 고정된 지식을 평가하는 것이 아니라, 평가 시점에 Wikidata에서 검색된 최신 정보를 바탕으로 정답을 제공합니다. 또한, ROME 및 MEMIT(모델 파라미터 수정 방법)과 SERAC 및 IKE(원래 파라미터를 보존하는 방법)와 같은 지식 편집 방법들을 사용하여 LLM의 지식을 최신 상태로 유지하는 방법을 평가하고, 검색 보완 생성(Retrieval Augmented Generation, RAG)과의 성능을 비교합니다.

- **Performance Highlights**: 이 연구에서 평가된 18개의 LLM 중 대부분은 최신 정보를 정확하게 반영하지 못하는 경향이 있는 것으로 나타났습니다. 특히, 최신 축구 클럽 정보에 대한 질문에서 상당수 모델이 구식 정보를 제공했습니다. 그러나 지식 편집 방법을 사용하여 이러한 모델들을 최신 상태로 정렬한 결과, 편집 알고리즘들은 실제 데이터에 대한 효과적인 업데이트를 제공할 능력을 보여주었습니다. 비록 이 방법들이 아직 완벽하지 않으나, 이 연구는 LLM의 지식을 실시간으로 최신 상태로 유지하는 방법에 대한 중요한 통찰력을 제공합니다.



### Analyzing the Impact of Data Selection and Fine-Tuning on Economic and  Political Biases in LLMs (https://arxiv.org/abs/2404.08699)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)에 특정 정치 및 경제 이념을 통합하는 방법론을 탐구하여 AI 기술의 윤리적 적용에 대한 중요한 대화에 기여합니다. 특히, 자원 집약적인 완전한 사전 훈련 대신에 매개변수 효율적 미세조정(Parameter-Efficient Fine-Tuning, PEFT) 기법을 사용하여 특정 이념에 맞춰 LLM을 미세조정하는 접근방식을 제시합니다.

- **Technical Details**: 연구팀은 데이터 선택, 주석 처리, 및 지시어 튜닝을 체계적으로 다루며, 이를 통해 LLM이 특정 이념을 반영하도록 프레임워크를 구축합니다. 또한, Low-Rank Adaptation (LoRA) 및 Adapter 등의 매개변수 효율적 미세조정 기법을 사용하여, LLM의 소수 매개변수만을 수정함으로써 특정 이념으로의 편향성을 구현할 수 있는 방법을 소개합니다. 이러한 방식은 전체 모델을 재훈련하는 것보다 훨씬 적은 자원을 사용합니다.

- **Performance Highlights**: 이 연구는 양적 및 질적 평가를 통해 제안한 프레임워크의 효과를 검증합니다. 결과적으로, LLM이 선택된 정치 및 경제 이념에 더 근접하게 조정되었으며, 이는 정책 결정 과정 및 대중 의견 형성에 기존의 편향을 반영할 가능성을 줄이는 데 기여할 수 있습니다. 또한, 이러한 접근법이 사회적 가치와 민주적 원칙에 부합하는 방식으로 AI를 배치하는 데 필수적인 윤리적 지침 및 거버넌스 프레임워크의 필요성을 강조합니다.



### Lossless Acceleration of Large Language Model via Adaptive N-gram  Parallel Decoding (https://arxiv.org/abs/2404.08698)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 추론 효율성을 향상시키기 위해 새롭게 Adaptive N-gram Parallel Decoding(ANPD)를 도입하였습니다. 이 방법은 동시에 여러 토큰을 생성할 수 있게 함으로써 추론 속도를 높이는 손실 없는 접근 방식입니다. ANPD는 빠른 초안 작성 단계와 해당 초안을 검증하는 두 단계로 구성됩니다.

- **Technical Details**: ANPD는 N-gram 모듈을 활용하여 초기 초안을 생성하고, 이는 현재의 상호작용 콘텍스트에 따라 적응적으로 변화합니다. 이후 본래 LLM이 제안된 토큰을 평가하고 확인하는 검증 단계를 거칩니다. 또한, N-gram 모듈의 다층 구조를 활용하여 초기 초안의 정확도를 높여 추론 지연을 줄입니다. ANPD는 재훈련이나 추가 GPU 메모리가 필요 없으며, 기존 LLM에 플러그 앤 플레이 방식으로 적용될 수 있는 효율적인 방법입니다.

- **Performance Highlights**: 실험을 통해 LLaMA 모델과 그 파인 튜닝된 변형체들의 속도가 최대 3.67배 향상되었음을 확인하였습니다. 이는 ANPD가 대형 언어 모델의 추론 속도를 현저히 향상시킬 수 있음을 입증합니다.



### Enhancing Question Answering for Enterprise Knowledge Bases using Large  Language Models (https://arxiv.org/abs/2404.08695)
Comments: DASFAA 2024 Accepted

- **What's New**: 이 논문에서는 기업 지식 기반 시스템을 위한 새로운 검색-생성(Reveal-Generation) 프레임워크인 EKRG를 제안하여, 제한된 주석(annotation) 비용으로 질문응답(Question-Answering)을 가능하게 합니다. EKRG는 대규모 언어 모델(LLMs)을 활용하여, 최소한의 주석 데이터만을 사용하여 효과적으로 지식 검색 및 생성 과정을 수행하도록 설계되었습니다.

- **Technical Details**: EKRG 프레임워크는 지식 검색(retrieval)과 생성(generation) 두 가지 주요 과정을 포함합니다. 첫째, 지식 검색을 위해, 본 설계는 LLM을 사용하여 효과적인 문서-질문 쌍을 생성하는 'instruction-tuning' 방법을 도입합니다. 이는 엔터프라이즈 지식 기반이 당면한 다양한 질문 유형에 대해 고려하여, 사실 중심(fact-oriented) 및 해결책 중심(solution-oriented) 지식을 아우르는 다양한 질문을 효율적으로 생성할 수 있습니다. 둘째, 생성 단계에서는 'chain of thought (CoT)' 기반의 미세조정(fine-tuning) 방법을 제안하여, LLM 기반 생성기(generator)가 검색된 문서를 사용하여 사용자 질문에 능숙하게 대응할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, EKRG 프레임워크는 실제 데이터셋에서 효과적임을 보여주었습니다. 특히, 검색된 문서와 관련성 있게 질문을 생성하고, 해당 질문에 대한 답변을 생성하는 데 있어 높은 정확성과 일관성을 달성하였습니다. 이는 기존의 검색 방식과 대조적으로, 의미적 또는 문맥적 유사성을 간과하는 문제를 해결하며, 복잡한 기업 지식 데이터에 강력하게 적용될 수 있음을 시사합니다.



### Towards Building a Robust Toxicity Predictor (https://arxiv.org/abs/2404.08690)
Comments: ACL 2023 /

- **What's New**: 본 논문에서는 독성 언어 감지기의 견고성에 대해 연구하며, 'ToxicTrap'이라는 새로운 적대적 공격을 제안합니다. 이 공격은 대상 모델이 독성 텍스트 샘플을 양호한 것으로 예측하도록 하여 SOTA(State-of-the-Art) 텍스트 분류기를 속입니다. 본 연구는 다중 클래스(Multiclass) 및 다중 레이블(Multilabel) 독성 언어 감지기의 취약점을 식별할 수 있는 새로운 목표 함수와 탐색 전략을 도입합니다.

- **Technical Details**: ToxicTrap은 그리디 기반 검색 전략과 소규모 단어 레벨의 변형을 사용하여 독성 적대적 사례를 빠르고 효과적으로 생성합니다. 변형된 텍스트는 원본의 동의어로 단어를 대체하여 모델의 탐지를 실패하게 합니다. 이 공격은 특히 API 서비스로 배포된 독성 분류기에 활용되며, BERT 및 DistillBERT 기반 모델을 포함한 현대의 텍스트 분류기가 이러한 공격에 취약함을 실험적으로 보여줍니다. 또한, 적대적 훈련(Adversarial Training)과 그 개선된 버전이 모델의 견고성을 증가시키는 데 어떻게 도움이 되는지를 보여줍니다.

- **Performance Highlights**: ToxicTrap은 다중 레이블 케이스에서 98% 이상의 공격 성공률을 달성하여 최첨단 독성 텍스트 분류기가 소규모 적대적 변형에 취약함을 입증합니다. 실험을 통해 공격이 성공적으로 독성을 포함하되 저독성 점수를 받는 텍스트를 생성하고, 이를 '양호(benign)'으로 분류하도록 모델을 속일 수 있습니다. 적대적 훈련은 다양한 공격으로부터 생성된 독성 적대적 예제를 사용하여 훈련함으로써, 공격에 대한 모델의 견고성을 향상시키는 것으로 나타났습니다.



### Extractive text summarisation of Privacy Policy documents using machine  learning approaches (https://arxiv.org/abs/2404.08686)
Comments: University of Edinburgh MInf (Master of Informatics) Thesis, 52 pages, 13 figures, Submitted and approved by the institution in May 2022

- **What's New**: 이 연구에서는 K-means 클러스터링과 Pre-determined Centroid (PDC) 클러스터링을 기반으로 한 두 가지 개인정보 보호 정책 (Privacy Policy, PP) 요약 모델을 제시합니다. GDPR(General Data Protection Regulation)의 14개 주요 주제를 기반으로 클러스터 중심을 정의하고, 이를 통해 PP 문서에서 필수 문장을 추출하는 요약 메커니즘을 실현합니다. PDC 클러스터링 모델은 K-means 모델보다 우수한 성능을 보여, PP 문서의 GDPR 준수 여부를 평가하는 응용 프로그램으로 개발될 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: K-means 클러스터링과 PDC 클러스터링 알고리즘은 유클리드 거리를 사용하여 문장 벡터를 클러스터 중심에 할당합니다. 특히, PDC 모델은 GDPR 지침에 명시된 주제에 따라 사전에 정의된 클러스터 중심을 활용하여 문서를 요약합니다. 이러한 접근 방식은 Sum of Squared Distance (SSD) 및 ROUGE(metrics to evaluate summaries)와 같은 평가 방법에서 K-means 모델을 상회하는 결과를 보입니다. 또한, 이 연구에서는 문장 Transformer (Sentence Transformer, SBERT라고도 함)를 활용하여 강력하고 효율적인 문장 벡터화를 구현합니다.

- **Performance Highlights**: PDC 클러스터링 모델은 SSD와 ROUGE 평가에서 각각 27%, 24% 더 좋은 성과를 보입니다. 이것은 PDC 모델이 GDPR의 주제에 맞는 문장 선택에 더 적절하게 최적화되었음을 시사합니다. 요약 모델의 성능은 GDPR 준수 여부를 판단하는 데 활용될 수 있으며, 이는 정책 문서의 중요한 측면을 파악하는 데 크게 기여할 수 있습니다.



### Neural Sequence-to-Sequence Modeling with Attention by Leveraging Deep  Learning Architectures for Enhanced Contextual Understanding in Abstractive  Text Summarization (https://arxiv.org/abs/2404.08685)
- **What's New**: 이 연구에서는 단일 문서의 추상적 요약을 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 구조적, 의미론적, 신경망 기반(neural-based) 접근 방식을 통합하여 추상적(text summarization, TS)의 새로운 방법론을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 단계로 구성됩니다. 전처리(pre-processing) 단계에서는 지식 기반의 단어 의미 분별(Word Sense Disambiguation, WSD) 기술을 사용하여 모호한 단어를 일반화하고, 의미론적 내용 일반화(semantic content generalization)로 드문(rare) 단어나 사전에 없는(out-of-vocabulary, OOV) 단어를 처리합니다. 일반화된 텍스트는 신경 언어 처리(neural language processing) 기술을 사용하여 지속적인 벡터 공간으로 변환되고, 주의 메커니즘(attention mechanism)을 갖춘 심층 시퀀스 대 시퀀스(deep sequence-to-sequence, seq2seq) 모델을 사용하여 벡터 표현에 기반한 일반화된 요약을 예측합니다. 후처리(post-processing) 단계에서는 휴리스틱 알고리즘과 텍스트 유사도 메트릭스(text similarity metrics)를 사용하여 생성된 요약을 더욱 세밀하게 다듬습니다.

- **Performance Highlights**: 실험 평가는 Gigaword, Duc 2004 및 CNN/DailyMail과 같은 유명 데이터셋에서 수행되었으며, 제안된 프레임워크의 효과를 입증했습니다. 결과는 드문 단어와 OOV 단어 처리에 있어서 기존의 최신 딥 러닝(deep learning) 기술을 능가하는 중요한 개선을 보여줍니다.



### Is English the New Programming Language? How About Pseudo-code  Engineering? (https://arxiv.org/abs/2404.08684)
- **What's New**: 이 연구는 ChatGPT, OpenAI의 선도적인 언어 모델의 다양한 입력 형태에 대한 이해와 실행 능력의 영향을 탐구합니다. 특히, 자연 언어 처리(Natural Language Processing, NLP)를 사용하는 챗봇의 독특한 도전과 혁신적 잠재력을 강조하면서, 자연 언어 및 의사 코드(pseudo-code) 공학 입력 사이의 대조를 연구했습니다.

- **Technical Details**: 연구는 사례 연구 방법론과 담론 분석을 사용하여 ChatGPT의 응답을 분석했습니다. 이는 의도의 이해, 해석 가능성(interpretability), 완성도(completeness), 그리고 창의성(creativity)의 네 가지 범주에서 모델의 숙련도를 조사했습니다. '주간 식단 계획'과 '쇼핑 목록'을 포함하는 합성 사례 시나리오를 통해 자연 언어 및 의사 코드 입력에 대한 ChatGPT의 반응을 평가했습니다.

- **Performance Highlights**: 연구 결과에 따르면 의사 코드 공학 입력은 ChatGPT의 반응의 명확성과 결정성을 크게 향상시켜 자연 언어에서 내재된 모호성을 줄였습니다. 또한, 프롬프트 엔지니어링 기술(Prompt Engineering Techniques)을 통해 구조화된 개선된 자연 언어는 모델의 해석 가능성과 창의성을 개선했습니다.



### Text clustering applied to data augmentation in legal contexts (https://arxiv.org/abs/2404.08683)
Comments: 23 pages, 4 figures. submitted to Artificial Intelligence and Law Journal

- **What's New**: 이 연구는 법률 텍스트 분류에 자연어 처리(Natural Language Processing, NLP) 도구의 힘을 활용하여 전문가가 정교하게 큐레이션한 데이터 세트를 향상시켜 법률 텍스트의 분류 워크플로우를 개선합니다. 특히 유엔(United Nations, UN)의 2030 의제(Agenda)의 지속 가능 개발 목표(Sustainable Development Goals, SDGs) 데이터를 실제 사례 연구로 고려하였습니다.

- **Technical Details**: 이 연구에서는 전문가 레이블이 지정된 데이터베이스에서 데이터 클러스터링 기술(Data Clustering Techniques)과 데이터 증강(Data Augmentation)을 사용하여 자연어 처리 모델의 성능을 향상시키는 방법을 보여줍니다. 구체적으로, 데이터 증강 클러스터링 기반 전략은 분류 모델의 정확도(Accuracy)와 민감도(Sensitivity) 메트릭스에서 현저한 향상을 이끌어 냈습니다.

- **Performance Highlights**: 일부 SDGs에서는 성능이 15% 이상 향상되었습니다. 특정 경우에는 예제 베이스가 주목할만한 5배 확장되었습니다. 데이터 증강 전략은 분류되지 않은 법률 텍스트를 다룰 때 특히 효과적입니다. 이는 기존 지식 베이스를 확장하는데 있어 노동 집약적인 수동 분류 노력을 필요로 하지 않는 가치 있는 수단을 제공합니다.



### EFSA: Towards Event-Level Financial Sentiment Analysis (https://arxiv.org/abs/2404.08681)
- **What's New**: 이 연구에서는 금융 텍스트에서 사건을 추출하고 관련 감정 분석을 통해 금융 시장의 변동성에 미치는 영향을 분석하는 새로운 금융 감정 분석(Financial Sentiment Analysis, FSA) 접근 방식을 제안합니다. 특히, 이 논문은 사건 수준에서의 금융 감정 분석(Event-Level Financial Sentiment Analysis, EFSA)을 도입하여 금융 텍스트의 감정을 사건별로 분류하는 새로운 작업을 설정하고, 이를 위한 대규모 중국어 데이터셋을 공개하였습니다.

- **Technical Details**: EFSA 작업은 금융 뉴스에서 (회사, 산업, 대분류 사건(coarse-grained event), 세분류 사건(fine-grained event), 감정)으로 구성된 오중(quintuple)을 예측하는 것을 목표로 합니다. 사건 추출을 분류 작업으로 재개념화하고, 대분류와 세분류 사건 카테고리를 포함하는 분류 체계를 설계하였습니다. 또한, 4단계 사고의 흐름(Chain-of-Thought, CoT) 기반 접근 방식을 사용하여 이 작업을 수행하는 새로운 프레임워크를 개발하였습니다.

- **Performance Highlights**: 실험 결과, 기존의 방법들과 비교하여 새로운 LLM(Large Language Model)-기반 프레임워크는 현존하는 기법들을 뛰어넘는 상태 최고(State-of-the-art)의 성능을 달성했습니다. 이는 사건 수준에서의 분석이 금융 텍스트의 감정 예측 정확도를 높이는 데 중요한 역할을 할 수 있음을 시사합니다.



### Automating Research Synthesis with Domain-Specific Large Language Model  Fine-Tuning (https://arxiv.org/abs/2404.08680)
- **What's New**: 이 연구는 Systematic Literature Reviews(SLR)의 자동화를 위해 세밀하게 조정된 대용량 언어 모델(LLM)의 사용을 선도하며, 학문적 연구 방법론을 향상시키기 위한 AI의 통합에 있어 중대하고 새로운 기여를 제시합니다. 이 연구는 최신 미세 조정 방법론과 공개 소스 LLM을 사용하여 지식 합성을 포함하는 SLR 프로세스의 최종 실행 단계를 자동화하는 실용적이고 효율적인 접근 방식을 시연하였습니다.

- **Technical Details**: 이 연구는 특정 SLR 대상 학술 논문의 컬렉션에서 학습된 미세 조정된 오픈 소스 LLM을 제안하며, 이를 통해 일반적인 지식에서 더 좁은 도메인 특정 전문 지식으로 LLM의 지식을 확장합니다. 또한, LLM 환각 현상을 완화하고 LLM 응답이 정보의 출처를 추적할 수 있는 메커니즘을 구축하는 새로운 방법을 제시합니다.

- **Performance Highlights**: 이 연구의 결과는 실제로 기존의 PRISMA를 준수하는 SLR을 복제함으로써 검증되었으며, 높은 사실적 정확성을 유지하는 LLM 응답을 보여주었습니다. 이 연구는 학술 연구에서 SLR 수행의 여러 노동 집약적 과정을 간소화할 수 있는 미세 조정된 LLM의 잠재력을 확증했습니다.



### Your Finetuned Large Language Model is Already a Powerful  Out-of-distribution Detector (https://arxiv.org/abs/2404.08679)
- **What's New**: 이 연구에서는 사전 훈련된 대규모 언어 모델(LLM: Large Language Model)과 그 파인튜닝(finetuned)된 변형 간의 우도 비(likelihood ratio)를 사용하여 분포 외 데이터(OOD: Out-of-Distribution) 탐지 기준으로 다시 조명합니다. 특히, 이 우도 비를 통해 QA(Question-Answering) 시스템에서의 OOD 질문을 탐지하고, 일반 질문에 대한 전문된 LLM의 성능을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 사전 훈련된 LLM은 방대한 데이터를 통해 OOD 데이터에 대한 사전 지식을 가지고 있다고 가정하며, 이를 바탕으로 파인튜닝된 데이터(인-디스트리뷰션: in-distribution)와의 우도 비를 계산하여 OOD를 탐지합니다. 이 기법은 현대 신경망 프레임워크 내의 손실 함수(loss functions)를 통해 쉽게 구현할 수 있는 우도 값을 사용합니다. 또한, 기존에 파인튜닝된 모델을 활용하여 추가적인 훈련 없이 OOD 탐지가 가능합니다.

- **Performance Highlights**: 다양한 설정(극단적인 OOD, 근접 OOD, 스팸 탐지, QA 시나리오)에서의 광범위한 평가를 통해 방법의 효과를 입증했습니다. 특히, QA 시스템에서의 OOD 질문 탐지 능력이 뛰어나며, 이를 통해 현재 QA 시스템의 강건성을 크게 향상시킬 수 있음을 보여줍니다.



### ALERT: A Comprehensive Benchmark for Assessing Large Language Models'  Safety through Red Teaming (https://arxiv.org/abs/2404.08676)
Comments: 19 pages, preprint

- **What's New**: 새롭게 개발된 ALERT 벤치마크를 통해 대규모 언어 모델(LLMs)의 안전성을 평가합니다. ALERT는 레드 팀링(red teaming) 방법론을 활용하고, 45,000개 이상의 특정 지침으로 구성된 고급 위험 분류 체계를 사용하여 LLMs의 취약점을 확인하고 개선을 돕습니다. 이 벤치마크는 AI 정책 준수를 평가하는 데에도 유용합니다.

- **Technical Details**: ALERT는 6개의 매크로(macro) 범주와 32개의 미시(micro) 범주로 구성된 새로운 안전 위험 분류 체계를 사용합니다. 이는 다양한 상황에 대한 LLMs의 반응을 평가하고, 정책과 맥락에 부합하는지 검토할 수 있게 하며, 분류 체계에 기반하여 LLMs 테스트를 실행합니다. 또한, 벤치마크는 반자동적 방법론을 통해 사람의 노력을 줄이면서 LLMs의 안전성을 효율적으로 측정합니다.

- **Performance Highlights**: 이 연구에서 10개의 유명한 오픈 소스 및 클로즈드 소스 LLMs을 평가한 결과 대부분의 모델이 여전히 안전성을 확보하는데 어려움을 겪고 있음을 발견했습니다. 특히, 일부 모델은 캐너비스(cannabis) 소비나 유통과 관련된 응답에서 취약점이 드러났습니다. 이러한 발견은 맥락 및 정책 인식 평가의 중요성을 강조합니다.



### Effects of Different Prompts on the Quality of GPT-4 Responses to  Dementia Care Questions (https://arxiv.org/abs/2404.08674)
- **What's New**: 이 연구는 치매 간병에 대한 대규모 언어 모델(GPT-4)의 반응을 통해, 다양한 프롬프트(prompt)가 의료 분야에서의 반응 품질에 어떤 영향을 미치는지 탐색합니다. 이 연구는 특히 치매 간병인이 겪는 과제에 초점을 맞춰, 세 가지 유형의 체계적 프롬프트를 조합하여 반응을 평가하였습니다.

- **Technical Details**: 연구팀은 세 가지 유형의 프롬프트를 개발하여 GPT-4에 적용했습니다: 시스템 프롬프트(system prompts, SPs), 초기화 프롬프트(initialization prompt), 그리고 작업 프롬프트(task prompts, TPs). 각각의 프롬프트 유형은 특정 전문가의 역할을 지정하고, 명확한 작업 지시를 포함하여, 치매 간병 상황에 맞는 응답을 유도하는 것을 목표로 합니다.

- **Performance Highlights**: GPT-4는 12개의 다양한 프롬프트 조합을 사용하여 총 36개의 응답을 생성했으며, 이 응답들은 두 명의 경험 많은 치매 간호 전문가에 의해 평가되었습니다. 이 평가는 0부터 5까지의 점수로 구성된 5가지 품질 지표를 포함하여 이루어졌습니다. 이 연구는 치매 간병인을 지원하는 AI 시스템의 효과를 평가하고 향후 의료 분야에서의 프롬프트 엔지니어링에 대한 연구를 제안합니다.



### Sentiment analysis and random forest to classify LLM versus human source  applied to Scientific Texts (https://arxiv.org/abs/2404.08673)
Comments: 12 Pages, 3 tables, 6 figures

- **What's New**: ChatGPT-4 출시 이후 과학 및 기술 문서를 비롯한 모든 종류의 텍스트를 자동으로 생성할 수 있는 인공 지능(AI) 플랫폼에 대한 전 세계적인 토론이 활발해졌습니다. 본 연구에서는 감성 분석(Sentiment Analysis)을 사용하여 자동 텍스트 생성 엔진과 인간이 생성한 텍스트를 구별하는 새로운 분류 방법론을 제안합니다. 이는 앞으로 많은 텍스트가 전적으로 사람에 의해 쓰이지 않을 가능성에 대비해 교육과 학문적 절차를 적응시키는 데 도움이 될 수 있습니다. 즉, 이 방법은 소프트웨어에 의해 이루어진 작업으로 의심되는 환경에서 부정 행위를 감지하는데 유용하게 사용될 수 있습니다.

- **Technical Details**: 본 논문에서는 네 가지 다른 감성 사전('afinn', 'bing', 'nrc', 'loughran')을 사용하여 특징(Feature)을 생성하고, 이를 머신러닝 Random Forest 분류 알고리즘을 통해 모델을 학습시키는 기법을 소개합니다. 데이터 증강(Data Augmentation)도 사용되었으며, 이를 통해 보다 나은 평가 점수를 얻을 수 있었습니다.

- **Performance Highlights**: 제안된 방법론은 인간 대비 AI에 의해 생성된 텍스트를 감지하는 데 있어 매우 설득력 있는 결과를 보여주었습니다. 감성 분석을 기반으로 한 특징 생성은 텍스트의 길이와 형식이 서로 다른 카테고리 간의 정보 유출을 방지, robust한 대응이 가능하게 합니다.



### Revealing Trends in Datasets from the 2022 ACL and EMNLP Conferences (https://arxiv.org/abs/2404.08666)
- **What's New**: 이 연구는 2022년에 ACL과 EMNLP 학회에서 소개된 새로운 데이터셋들을 분석하여, NLP(자연어 처리) 연구의 최신 동향과 필요사항을 탐색합니다. Transformer architecture의 도입으로 급성장한 NLP 분야에서는, 고품질의 데이터셋이 PLM(Pre-trained Large Language Models)의 성능 향상에 필수적임을 강조하고 있습니다.

- **Technical Details**: 이 연구는 2022년 ACL과 EMNLP에서 발표된 92개의 논문을 분석하여 데이터셋의 주요 특징 및 NLP 주제들을 파악하였습니다. 데이터셋 생성에 있어 중요한 주제에는 텍스트 생성(Text Generation), 정보 추출(Information Extraction), 질의 응답(Question Answering) 등이 포함되어 있습니다. 이들 데이터셋은 다양한 크기로 구성되어 있으며, 연구자들이 추구하는 NLP 시스템의 고도화와 다양화를 가능하게 합니다.

- **Performance Highlights**: 분석 결과에 따르면, 새로운 NLP 데이터셋은 주로 10,000에서 50,000 문장 범위에 걸쳐 분포하고 있으며, 이는 해당 데이터의 규모가 이제까지의 NLP 시스템 성능에 있어 만족스러운 결과를 도출하기 위한 중요한 요소임을 시사합니다. 또한, 학술과 산업 분야의 연구자들 간의 협력이 데이터셋 구축에 큰 영향을 미치고 있으며, 다국어 및 멀티모달 데이터셋의 증가 추세도 확인되었습니다.



### The Comparison of Translationese in Machine Translation and Human  Transation in terms of Translation Relations (https://arxiv.org/abs/2404.08661)
- **What's New**: 이 연구는 인공신경망 기계번역 (NMT)과 인간 번역 (HT) 간의 차이를 번역 관계를 통해 탐색합니다. 두 번역 방법 간의 전체적인 차이, 비문자적 번역 기술의 활용 방법과 비문자적 기술 사용에 영향을 미치는 요인들에 대해 조사합니다. 두 개의 병렬 말뭉치를 사용하여 NMT와 HT의 번역 기술을 분석하고, 이를 통해 NMT가 HT보다 훨씬 더 문자적인 번역에 의존하는 것으로 나타났습니다.

- **Technical Details**: 말뭉치는 동일한 원문을 기반으로 NMT로 번역된 그룹과 인간이 번역한 그룹으로 구성됩니다. 각각의 번역은 9개의 장르를 포함하며, 번역 관계는 수동으로 주석 처리된 쌍들을 기반으로 실시된 비교 분석을 통해 평가됩니다. 연구에서는 의미론적 및 구문학적 뉘앙스(예: 상위어(hypernyms), 품사 태깅 변경)를 포함합니다.

- **Performance Highlights**: NMT는 HT에 비해 전 장르에 걸쳐 현저하게 문자적 번역에 의존합니다. 구문 수준의 비문자적 번역 기술에서 NMT는 HT와 유사한 성능을 보였지만, 의미 수준에서는 특정화, 형상화, 등가, 일반화 기술을 사용할 때 성능이 이상적이지 않았습니다. 이는 NMT가 복잡한 의미 구조를 처리하는 데 아직 한계가 있음을 나타냅니다.



### Linear Cross-document Event Coreference Resolution with X-AMR (https://arxiv.org/abs/2404.08656)
Comments: LREC-COLING 2024 main conference

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 사용이 비용이 많이 드는 Event Coreference Resolution (ECR) 문제를 해결하기 위해 새로운 그래픽 표현인 X-AMR을 제안합니다. X-AMR은 행사를 중심으로 하여 개별 언급을 연결하는 cross-document 버전의 Abstract Meaning Representation(AMR)입니다. 이를 통해 복잡성을 크게 줄이고 ECR 작업을 더 효율적으로 만듭니다.

- **Technical Details**: X-AMR 구조는 event triggers와 arguments를 VerbNet Lexicon과 Knowledge Base를 사용해 연결하여 corpus 차원의 event graph를 생성합니다. 이 graph는 pairwise 점수매기기 없이 ECR을 수행하는 데 사용됩니다. 또한, ECR 작업을 위한 새로운 multi-hop coreference 알고리즘이 도입되어 전통적인 pairwise 접근 방식과 달리 계산 비용을 피할 수 있습니다. ECB+ corpus에 대해 X-AMR 주석을 자동 및 수동으로 부여하고, GPT-4를 사용해 이를 평가합니다.

- **Performance Highlights**: GPT-4를 사용한 자동 생성 X-AMR graph는 zero-shot 및 few-shot 설정에서 평가되었으며, 전통적인 ECR 벤치마크 데이터셋에 비해 유의미한 성능 향상을 보여주었습니다. 이는 ECR을 위한 새로운 접근 방식이 실제 어플리케이션에 효과적으로 확장될 수 있음을 시사합니다.



### Transformer-based Joint Modelling for Automatic Essay Scoring and  Off-Topic Detection (https://arxiv.org/abs/2404.08655)
Comments: Accepted in LREC-COLING 2024

- **What's New**: 이 논문에서는 자동화된 에세이 채점(Automated Essay Scoring, AES) 시스템에서 부적절한(오프-토픽) 응답을 식별하는 새로운 비지도 학습 기법을 제안합니다. 이 방법은 과거의 방법론과는 달리 에세이 채점과 오프-토픽 감지를 동시에 수행할 수 있는 'Automated Open Essay Scoring (AOES)' 모델을 사용하며, 트랜스포머(transformer) 모델 위에 덧붙여질 수 있는 새로운 주제 규제 모듈(Topic Regularization Module, TRM)을 포함합니다.

- **Technical Details**: AOES 모델은 hybrid loss function을 사용하여 학습되며, 오프-토픽 에세이를 감지하기 위해 Mahalanobis distance score를 계산하는 방식을 사용합니다. 이 모델은 BERT(Bidirectional Encoder Representations from Transformers)를 기반으로 하여 자연 언어 이해(natural language understanding)에 적합하게 조정되었습니다. 비지도 방식으로 오프-토픽 데이터 없이도 효과적으로 오프-토픽 응답을 식별할 수 있는 새로운 접근법을 제시합니다.

- **Performance Highlights**: AOES 모델은 ASAP-AES 및 PsyW-Essay라는 두 가지 에세이 스코링 데이터셋에서 기존의 지도 학습 및 비지도 학습 방식보다 뛰어난 성능을 보여주었습니다. 또한 다양한 적대적(Adversarial) 시나리오에 대한 테스트에서도 이 모델이 인간 수준의 방해를 감지하는 데 매우 효과적임을 입증했습니다.



### Optimal path for Biomedical Text Summarization Using Pointer GP (https://arxiv.org/abs/2404.08654)
Comments: 3 pages, 3 figures

- **What's New**: 이 연구에서는 전통적인 GPT 모델의 attention 메커니즘을 pointer 네트워크로 대체하여 의료 기록 요약의 정확성을 향상시켰습니다. 이로 인해 클리니션들이 환자의 상태를 보다 효과적으로 파악할 수 있게 되었으며, 새로운 EMR(Electronic Medical Records) 시스템 패러다임이 도입될 잠재력을 가지게 되었습니다.

- **Technical Details**: 기존 GPT 모델은 텍스트를 요약하는 과정에서 사실적 오류를 생성하고, 맥락을 무시하며, 단어를 과도하게 단순화하는 경향이 있습니다. 이를 해결하기 위해, 연구진은 GPT 모델의 attention 메커니즘을 pointer network로 교체하였습니다. 이러한 변경은 원문의 핵심 가치를 보존하는 데 중점을 두었습니다.

- **Performance Highlights**: 새로운 Pointer-GPT 모델은 ROUGE score를 사용하여 평가되었으며, 원래의 GPT 모델보다 우수한 성능을 보였습니다. 이 결과는 pointer 네트워크가 의료 기록 시스템에 귀중한 추가가 될 수 있음을 시사합니다.



### MMInA: Benchmarking Multihop Multimodal Internet Agents (https://arxiv.org/abs/2404.09992)
- **What's New**: MMInA 벤치마크는 실제 인터넷 사용과 유사하게 다양한 홉을 거쳐 다중 모달 정보를 처리하여 과제를 완수하는 자율적인 AI 대리인을 평가하기 위해 개발되었습니다. 이는 다양한 웹사이트에서 긴 범위의 계획 및 멀티모달 추론 (multimodal reasoning) 능력을 요구하는 현실적인 과제를 포함하며, 실제 세계에서의 적용 가능성을 극대화합니다.

- **Technical Details**: MMInA는 1,050개의 인간이 작성한 과제를 통해 단일 홉(hop) 뿐만 아니라 멀티 홉 탐색 능력을 평가합니다. 이 벤치마크는 다양한 웹사이트가 진화하는 과정을 반영하며, 대표적인 대형 언어 모델(Large Language Models, LLMs) 및 대형 멀티모달 모델(Large Multimodal Models, LMMs)을 활용하여 과제를 수행합니다. 기존의 기법들을 넘어선 새로운 기억 증강 방법(memory augmentation)을 제안하여 에이전트의 성능을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 최신의 스탠드얼론 모델인 GPT-4V는 전반적인 과제 성공률이 21.8%로 나타나, 인간의 성능(96.3%)에 비해 미흡하였지만, 텍스트 기반 벤치마크와 비교했을 때는 확실한 개선이 있었습니다. 에이전트는 초기 홉에서 더 실패할 가능성이 높았음이 밝혀졌으며, 이에 대응하여 개발된 기억 증강 기술은 LMMs의 인터넷 대리인으로서의 단일 홉 및 멀티 홉 웹탐색 능력을 현저히 향상시켰다는 평가를 받았습니다.



### Tango 2: Aligning Diffusion-based Text-to-Audio Generations through  Direct Preference Optimization (https://arxiv.org/abs/2404.09956)
Comments: this https URL

- **What's New**: 이 연구에서는 향상된 텍스트-오디오 생성 모델 Tango를 'Tango 2'로 발전시키는 접근법을 소개합니다. 기존의 Tango 모델을 사용하여 긍정적이고 부정적인 오디오 샘플을 포함하는 '오디오 알파카(Audio-alpaca)'라는 새로운 선호도 데이터셋에서 직접 선호 최적화(diffusion-DPO) 손실함수를 사용하여 미세 조정(fine-tuning)을 진행했습니다. 이는 오디오 결과의 질을 개선하는 것을 목표로 하며, 자동 및 수동 평가 모두에서 기존 모델들을 뛰어넘는 성능을 보여줍니다.

- **Technical Details**: Tango 2는 '오디오 알파카' 데이터셋에서 텍스트 프롬프트에 대한 음성 샘플의 선호도를 기반으로 학습되었습니다. 이 데이터셋은 오디오 샘플 중에서 CLAP 점수가 낮은 샘플을 부정적인 오디오로 선별하고, 이를 통해 모델이 긍정적인 오디오와 부정적인 오디오를 구분함으로써 입력된 텍스트 프롬프트에 더 잘 맞는 오디오를 생성하도록 합니다. 사용된 LDM(Latent Diffusion Model, 잠재 확산 모델)은 텍스트에서 오디오로의 직접 변환을 가능하게 하는 중요한 기술적 요소입니다.

- **Performance Highlights**: Tango 2는 기존의 Tango 및 AudioLDM2 모델을 자동과 수동 평가 모두에서 능가하는 성능을 보여줍니다. 이는 DPO 기법을 통한 미세조정이 텍스트-오디오 생성 과정에서의 의미론적 정렬(semantic alignment)을 개선하는데 효과적임을 시사합니다. 특히, 텍스트 프롬프트의 의미를 오디오 스페이스로 더 정확하게 매핑(map)하는 능력이 향상되었습니다.



### Foundational Challenges in Assuring Alignment and Safety of Large  Language Models (https://arxiv.org/abs/2404.09932)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 정렬과 안전성을 확보하는 데 있어 18가지 기초적인 도전 과제를 확인했습니다. 이러한 도전 과제들은 세 가지 다른 범주로 구성되어 있습니다: LLMs의 과학적 이해, 개발 및 배포 방법, 사회기술적 도전 과제.

- **Technical Details**: 이 도전 과제들을 바탕으로, 연구자들은 200개 이상의 구체적인 연구 질문(Research Questions)을 제기하였습니다. 이러한 질문들은 LLMs의 이해와 개선을 도모하기 위해 설계되어 연구개발에 중요한 방향을 제시할 것입니다.

- **Performance Highlights**: LLMs에 대한 과학적 이해를 향상시키고, 개발과 배포 방법을 최적화하며, 다양한 사회기술적 문제들을 해결하는 데 기여할 이 연구는 LLMs의 효과적인 활용과 안정성 보장에 중요한 발판을 마련합니다.



### Progressive Knowledge Graph Completion (https://arxiv.org/abs/2404.09897)
Comments: 14 pages, 10 figures

- **What's New**: 이 논문에서는 기존의 지식 그래프 완성(Knowledge Graph Completion, KGC)이 실제 시나리오와 잘 맞지 않는다는 문제점을 지적하고, 실제로 적용 가능한 진보된 지식 그래프 완성(Progressive Knowledge Graph Completion, PKGC) 작업을 소개합니다. PKGC는 검증(verification), 채굴(mining), 훈련(training)의 세 가지 과정을 통합하여 지식 그래프의 점진적 완성을 모방합니다.

- **Technical Details**: PKGC는 지식 그래프를 알려진 부분과 알려지지 않은 부분으로 나누고, 이를 통해 점진적으로 알려진 부분을 확장합니다. 이 과정에서 최적화된 Top-k 알고리즘과 의미 유효성 필터(Semantic Validity Filter, SVF)라는 두 가지 가속 모듈을 제안하여 채굴 과정의 효율을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, 기존의 링크 예측(link prediction) 성능이 PKGC에서의 성능을 정확하게 반영하지 않음을 보여줍니다. 이는 PKGC가 실제 시나리오에서 더욱 적합한 모델 평가 방법을 제공할 수 있음을 시사합니다. 또한, PKGC에서 검증된 지식을 활용하는 방법에 대한 논의를 통해 향후 연구 방향을 제시합니다.



### Is Table Retrieval a Solved Problem? Join-Aware Multi-Table Retrieva (https://arxiv.org/abs/2404.09889)
- **What's New**: 이 연구는 열린 도메인 질의응답(QA) 시스템에서 테이블 검색의 문제를 다룬다. 특히, 다중 테이블 검색 문제를 재순위 매기기(re-ranking) 문제로 재구성하여 테이블 간의 조인 관계(join relationship)를 고려한 새로운 방법을 소개한다. 이 방법은 표와 쿼리 간의 관련성(table-query relevance)뿐만 아니라 표와 표 사이의 관련성(table-table relevance)을 추론하는 것을 포함한다.

- **Technical Details**: 이 논문에서 개발된 방법은 혼합 정수 프로그래밍(mixed-integer program)으로 구성된 새로운 재순위 매기기 방식을 사용한다. 이는 표 간의 조인 관계를 파악하고 추론하여 다중 테이블을 검색할 수 있게 한다. 다양한 시나리오에서 Spider 및 Bird 데이터셋을 사용하여 실험을 수행하고, 기존 방법들에 비해 성능이 우수함을 입증한다.

- **Performance Highlights**: 저자들은 이 방법이 테이블 검색에서 F1 점수(F1 score)로 최대 9.3%, 엔드-투-엔드(end-to-end) QA에서는 정확도(accuracy)로 최대 5.4% 우수한 성능을 보임을 보고하고 있다. 이는 기존의 상태(state-of-the-art) 방법들에 비해 상당한 개선이다.



### AI-Driven Statutory Reasoning via Software Engineering Methods (https://arxiv.org/abs/2404.09868)
- **What's New**: 인공 지능(AI) 기술의 발전과 함께 법률 분야에서 빠르게 발전하고 있는 '계산 법학(Computational Law)'에 초점을 맞춘 새로운 연구입니다. 이 연구는 사전 훈련된 대형 언어 모델(Pre-trained Large Language Models, LLMs) 같은 생성 인공 지능(Generative AI, GenAI) 기술을 활용하여 법률 문서를 프로그램처럼 처리하고 관련 상황에 대한 결과를 자동으로 계산해 내는 방법을 제안합니다.

- **Technical Details**: 이 논문에서는 법률 문서에 대한 AI 주도 분석을 적용하기 위해 자동 소프트웨어 테스팅 및 프로그램 분석과 같은 소프트웨어 공학 분야로부터 여러 가지 접근법을 소개합니다. 이 접근법을 통해 자연어 문서를 코드로 볼 수 있으며, 이를 통해 프로그램이 특정 사실 집합을 주어진 결과로 계산하게 합니다. 또한, 클라우드(Claude) 모델 같은 LLMs을 사용하여 주어진 법적 규정을 바탕으로 사례 질문에 답하고 가설적 상황에 대해 이유를 설명하는 예를 생성하는 방법을 설명합니다.

- **Performance Highlights**: LLM 기반 접근법은 법적 규정 데이터셋에서 수치 계산을 포함하지 않는 작업에서 75-80%의 정확도를 달성했습니다. 예를 들어, Anthropic의 클라우드(Claude) 3 Opus 모델은 관련 법률 규정을 제공받고 사고 과정을 설명하도록 지시받았을 때, 올바른 답변을 생성할 수 있음을 보여줍니다. 이러한 성과는 법률 텍스트에 적합한 AI 해석을 개선하거나 특정 법적 상황에서의 적용 가능성을 높이는 데 유용할 수 있습니다.



### Anatomy of Industrial Scale Multilingual ASR (https://arxiv.org/abs/2404.09841)
- **What's New**: 이 논문은 대규모, 다국어 자동 음성 인식(ASR) 시스템을 위해 설계된 AssemblyAI의 산업 규모 ASR 시스템을 설명합니다. 이 시스템은 다양한 응용 프로그램 요구 사항에 맞게 조정되었습니다. 논문에서는 다양한 언어에 걸쳐 비감독(12.5M 시간), 감독(188k 시간), 의사 라벨링(1.6M 시간) 데이터를 포함하는 다양한 훈련 데이터 세트를 활용하는 모델 아키텍처에 대해 자세히 설명합니다.

- **Technical Details**: 모델 아키텍처는 전체 맥락을 고려한 600M-parameter Conformer encoder와 BEST-RQ로 사전 훈련된 후 RNN-T (Recurrent Neural Network Transducer) 디코더와 함께 미세 조정된 것입니다. 이러한 구성은 Whisper large 및 Canary-1B와 같은 더 크고 계산 비용이 많이 드는 모델들과 경쟁할만한 단어 오류율(WER) 성능을 보여주었습니다.

- **Performance Highlights**: 이 아키텍처는 코드 전환(code-switching) 능력 향상, 최적화된 Whisper 베이스라인에 비해 5배 빠른 추론 속도, 음성 데이터에 대한 환각 발생률 30% 감소, Whisper에 비해 주변 소음 90% 감소 및 타임스탬프 정확성 향상 등 여러 주요 이점을 제공합니다. 또한, 다양한 언어 및 테스트 세트에서 Whisper large-v3와 Canary-1B와 같은 최신 ASR 모델들과 경쟁할 수 있으며, 모델 파라미터 수는 절반 수준입니다.



### Quantization of Large Language Models with an Overdetermined Basis (https://arxiv.org/abs/2404.09737)
- **What's New**: 이 논문에서는 카신 표현(Kashin representation) 원리를 기반으로 한 새로운 데이터 양자화(quantization) 알고리즘, 카신 양자화(Kashin Quantization)를 소개합니다. 이 방법은 벡터, 행렬 또는 텐서를 두 요소로 분해하는 것을 특징으로 하며, 첫 번째 요소는 작은 무한대(norm)을 유지하고, 두 번째 요소는 직교 행렬(orthogonal matrix)을 곱했을 때 마찬가지로 제한된 norm을 유지합니다. 또한, 분해 후 요소들은 몇 개의 피크(peaks) 주위에 잘 집중됩니다.

- **Technical Details**: 카신 양자화는 기존의 카신 표현 알고리즘을 사용하는 것을 넘어서 데이터를 효율적으로 양자화하는 새로운 방법을 제시합니다. 이 기법은 데이터 값을 정의된 피크 주변에 집중시켜 저장 공간을 획기적으로 줄일 수 있습니다. 주어진 행렬 또는 텐서를 두 가지 요소로 분해하여 하우스홀더(Householder), DCT, 버터플라이(Butterfly) 행렬과 같은 구조화된 행렬 유형을 통합합니다.

- **Performance Highlights**: 이 논문에서는 OPT, Bert, RoBerta 언어 모델을 사용하여 자연어 처리(Natural Language Processing, NLP) 공통 벤치마크에서 카신 양자화의 효율성을 검증했습니다. 다음 단어 예측(next-word prediction) 작업과 텍스트 분류(text classification) 작업에서 기존 양자화 방법보다 우수하거나 경쟁력 있는 모델 성능을 달성하면서 데이터 압축을 보장합니다.



### LoRAP: Transformer Sub-Layers Deserve Differentiated Structured  Compression for Large Language Models (https://arxiv.org/abs/2404.09695)
Comments: 8 pages,4 figures

- **What's New**: 본 연구에서는 트랜스포머(Transformer)의 다중 헤드 자기주의(Multi-head Self-attention, MHA) 부분층이 낮은 랭크(low-rank) 구조를 뚜렷하게 보여주고, 피드포워드 네트워크(Feed-Forward Network, FFN) 부분층은 그렇지 않다는 중요한 관찰을 했습니다. 이를 바탕으로, MHA 부분층에는 낮은 랭크 행렬 근사와 구조적 가지치기(Structured Pruning)를 융합한 새로운 압축 모델, LoRAP을 제안합니다. 또한, FFN 부분층에서는 기울기 없는(gradient-free) 구조적 가지치기 방법을 도입하여, 압축 후 모델의 성능을 유지하는 데 효과적인 방법을 개발했습니다.

- **Technical Details**: MHA 부분층의 경우, 입력 활성화에 가중치를 둔 특이값 분해(Singular Value Decomposition, SVD) 방법을 제안합니다(Activation Weighted SVD, AWSVD). 이를 통해 낮은 랭크 특성을 강화하고, MHA의 가중치 매트릭스 간 랭크 차이에 따라 매개변수 할당 방식을 새롭게 설계하였습니다. FFN 부분층에 대해서는 기울기에 의존하지 않는 채널 가지치기 방법을 도입함으로써, 가장 중요도가 낮은 약 1%의 매개변수가 모델 성능에 중대한 영향을 미칠 수 있음을 발견하였습니다.

- **Performance Highlights**: 제안한 모델은 WikiText2와 PTB 데이터셋에서의 제로샷(Zero-Shot) 복잡도 및 7개 상식 추론 데이터셋에서의 제로샷 태스크 분류에서 여러 압축 비율에 대해 기존 구조적 가지치기 및 낮은 랭크 근사 방법보다 우수한 성능을 보였습니다. 특히, 이러한 압축 방법을 통해 대규모 언어 모델의 메모리와 계산 자원의 사용을 효과적으로 줄이면서도 일반적인 작업 수행 능력을 유지할 수 있음을 입증했습니다.



### Harnessing GPT-4V(ision) for Insurance: A Preliminary Exploration (https://arxiv.org/abs/2404.09690)
- **What's New**: 본 논문에서는 인공지능(AI)의 발전에 중요한 이정표를 나타내는 대규모 멀티모달 모델(Large Multimodal Models, LMMs)의 등장을 다루고 있습니다. 보험 분야에서는 텍스트, 이미지, 비디오 등 다양한 데이터 형식을 포함하며, GPT-4V를 사용하여 보험 도메인에서의 멀티모달 작업의 여러 가지 측면을 탐구하였습니다. 특히 보험 유형(예: 자동차, 주거/상업용 재산, 건강, 농업 보험)과 보험 단계(예: 위험 평가, 위험 모니터링, 청구 처리)에 따라 멀티모달 작업을 분류하였습니다.

- **Technical Details**: 이 연구에서는 GPT-4V가 보험 관련 멀티모달 콘텐츠를 이해하고 보험 시나리오에 대한 포괄적인 지식을 가지고 있음을 보여주었습니다. 그러나 GPT-4V는 세부적인 위험 평가와 손실 평가에서 어려움을 겪으며, 이미지 이해에서 환각 현상을 경험하고, 다양한 언어에 대한 지원이 일관되지 않은 문제를 보였습니다.

- **Performance Highlights**: 실험 결과에 따르면 GPT-4V는 보험 관련 작업에서 뛰어난 능력을 발휘하였습니다. 이는 멀티모달 콘텐츠의 이해 및 보험 시나리오에 대한 광범위한 지식에 기반한 것입니다. 하지만 상세한 위험 평가 및 손실 평가, 이미지 이해의 환각 문제, 언어 지원의 불균일성 등 여러 단점도 확인되었습니다.



### Learn Your Reference Model for Real Good Alignmen (https://arxiv.org/abs/2404.09656)
- **What's New**: 이 논문에서는 기존의 언어 모델 정렬 방법에 대한 한계를 극복하기 위해 Trust Region Direct Preference Optimization (TR-DPO) 방법을 제안하고 있습니다. TR-DPO는 훈련하는 동안 참조 정책(policy)을 업데이트하여, 모델 정렬의 효율성과 성능을 향상시키는 새로운 접근 방식입니다.

- **Technical Details**: TR-DPO는 Direct Preference Optimization (DPO)의 최적화 작업을 재구성하며, 보상 모델(Reward Model, RM)을 제거하고 Kullback-Leibler 발산을 최소화하면서도 SFT 정책에 가깝게 유지합니다. 추가적으로, 훈련 과정 중에 참조 정책을 업데이트함으로써 더 나은 정렬을 달성하도록 설계되었습니다. 이 방법은 Alpha (α) 및 Tau (τ) 설정을 통해 텍스트 생성의 다양성과 길이를 효과적으로 관리합니다.

- **Performance Highlights**: TR-DPO는 GPT-4를 사용한 자동 평가에서 DPO를 최대 19%까지 능가하는 성능을 보여주었습니다. Anthropic HH 및 TLDR 데이터셋에서는 일관성(consistency), 정확성(accuracy), 세부 수준(detail level), 유용성(usefulness), 무해성(harmness) 등 여러 인간 중심 메트릭(human-centric metrics)에서 통계적으로 유의미하게 뛰어난 결과를 보여주었습니다.



### Real-world Instance-specific Image Goal Navigation for Service Robots:  Bridging the Domain Gap with Contrastive Learning (https://arxiv.org/abs/2404.09645)
Comments: See website at this https URL Submitted to IROS2024

- **What's New**: 이 연구에서는 실시간 환경에서 같은 객체를 찾아내는 인스턴스 특정 이미지 목표 탐색 (Instance-specific Image Goal Navigation, InstanceImageNav)의 성공률을 높이기 위해 새로운 방법인 Few-shot Cross-quality Instance-aware Adaptation (CrossIA)을 제안했습니다. 이 방법은 저품질과 고품질 이미지 간의 도메인 갭을 줄이기 위해 대조학습 (contrastive learning)과 인스턴스 분류기를 사용합니다. 또한, 기존의 선명화 기법으로는 해결할 수 없었던 도메인 갭을 효과적으로 해소하였다는 점에서 중요한 의의를 가집니다.

- **Technical Details**: 이 시스템은 대조학습을 통해 저품질 이미지 대량과 소수의 고품질 이미지 간의 인스턴스 기반 feature representation을 학습합니다. 추가적으로, 기존에 훈련된 탈흐림 모델 (pre-trained deblurring model)을 통합하여 로봇이 관찰하는 이미지의 품질을 높였습니다. SimSiam 모델을 CrossIA를 사용하여 튜닝하는 방식을 채택했으며, 실제 환경에서 20가지 다른 유형의 인스턴스를 찾는 Task에서 이 방법의 효과를 평가했습니다.

- **Performance Highlights**: 실험 결과, 기존의 SuperGlue 기반 접근법과 비교하여 최대 3배의 Task 성공률 향상을 보였습니다. 이는 Contrastive Learning과 이미지 향상 기술을 활용할 때 인스턴스 특정 이미지 목표 탐색 태스크에서 객체 위치 식별의 정확도를 높일 수 있는 가능성을 시사합니다.



### Prepacking: A Simple Method for Fast Prefilling and Increased Throughput  in Large Language Models (https://arxiv.org/abs/2404.09529)
Comments: 18 pages, code in this https URL

- **What's New**: 이 연구에서는 긴 입력 프롬프트(prompt)가 디코딩 시간(decoding time)에 상당한 부하를 유발할 수 있는 전형적인 패딩(padding) 기반 프리필링(prefilling) 계산의 문제점을 지적하고, 이를 최적화하기 위해 '프리패킹(Prepacking)'이라는 새로운 방법을 제안합니다. 프리패킹은 배치 내에서 다양한 길이의 프롬프트를 결합하고 이를 하나의 시퀀스로 압축하는 방식을 사용하여, 불필요한 계산을 줄이고 메모리 사용을 최적화합니다.

- **Technical Details**: 프리패킹은 입력 토큰의 KV 캐시(key-value cache)를 계산하는 과정에서 패딩 대신 다양한 길이의 프롬프트를 하나의 시퀀스로 결합하고, 빈 패킹(bin-packing) 알고리즘을 사용하여 시퀀스를 압축합니다. 이를 통해 포지셔널 인코딩(positional encoding)과 어텐션 마스크(attention mask)를 수정하여, 하나의 시퀀스 내에서 여러 프롬프트의 KV 캐시를 계산할 수 있습니다. 이 방법은 프롬프트 간의 상호 작용을 방지하며, 각 프롬프트는 독립적으로 위치 인덱스를 재 설정하여 처리됩니다.

- **Performance Highlights**: 프리패킹 방식은 기존의 전체 배치 방식에 비해 최대 6배까지 속도가 향상되었으며, 특히 배치 크기가 크고 프롬프트 길이가 다양할 때 더 큰 속도 향상을 보였습니다. 또한 메모리 사용량을 대폭 감소시켜, 프리필링 시 최대 16배 큰 배치 크기를 가능하게 했습니다. 이는 특히 메모리 제약이 있는 설정에서 LLM의 처리량을 높이는 데 유리합니다.



### State Space Model for New-Generation Network Alternative to  Transformers: A Survey (https://arxiv.org/abs/2404.09516)
Comments: The First review of State Space Model (SSM)/Mamba and their applications in artificial intelligence, 33 pages

- **What's New**: 본 연구에서는 자기주의 기반 트랜스포머 모델 (self-attention based Transformer model)의 대안으로 가능성이 있는 상태공간모델 (State Space Model, SSM)에 대한 첫 종합적인 검토를 제공합니다. 특히, 자연어 처리, 컴퓨터 비전, 그래프, 다중 모드 및 미디어, 포인트 클라우드/이벤트 스트림, 시계열 데이터 등 다양한 분야에서의 SSM의 응용을 전반적으로 소개하고 성능을 실험적으로 비교하고 분석함으로써, SSM의 특징과 장점을 더 잘 설명합니다.



### A Large-Scale Evaluation of Speech Foundation Models (https://arxiv.org/abs/2404.09385)
Comments: The extended journal version for SUPERB and SUPERB-SG. Accepted to TASLP. The arxiv version is further refined

- **What's New**: 이 연구는 음성처리에 대한 일반적인 평가 프레임워크의 부족을 해결하기 위해 Speech processing Universal PERformance Benchmark (SUPERB)를 소개합니다. SUPERB는 음성 자기지도 학습(Self-Supervised Learning, SSL) 모델이 다양한 실제 응용 프로그램에서 직접 사용될 수 있도록 평가합니다. 기존의 방법들과 달리, SUPERB는 15가지 음성 처리 작업에 걸쳐 SSL 모델의 일반화 능력을 평가하며, 이는 음성 인식 분야에서 새로운 가능성을 제시합니다.

- **Technical Details**: SUPERB는 동결된 기반 모델(frozen foundation model) 위에 각 작업에 특화된 가벼운 예측 헤드(lightweight prediction heads)를 사용하는 통합 다중 작업 프레임워크를 채택합니다. 이 프레임워크는 일반적인 SSL 기법을 적용하여 HIGH-LEVEL이자 전이가능한 표현을 학습하고, 간단한 가중 합(weighted-sum) 프로토콜로 최상위 작업 수행 상태(State-of-the-Art, SOTA)를 달성하는 것이 가능합니다.

- **Performance Highlights**: SUPERB를 통해 다양한 학습 목표, 모델 구성, 연산 예산에서 SSL 기술이 유용함이 입증되었으며, 구체적으로 SSL 모델은 전통적인 비-SSL 접근법과 비교하여 우수하거나 그 이상의 성능을 보여줍니다. 특히 음성 변환(Voice Conversion, VC)과 같은 일부 작업에서는 단일 레이어를 벤치마킹하는 것이 전체 레이어를 동결시키는 전통적 방법보다 더 낫다는 결과를 드러내, 특정 작업에 최적의 레이어를 찾는 데 있어 계층적 벤치마킹을 제안합니다.



### Tasks People Prompt: A Taxonomy of LLM Downstream Tasks in Software  Verification and Falsification Approaches (https://arxiv.org/abs/2404.09384)
- **What's New**: 이 연구는 Large Language Models (LLMs)를 활용하는 새로운 접근법인 '프롬프팅(Prompting)'에 초점을 맞춥니다. 많은 연구자들이 LLM을 최대한 활용하기 위해 프롬프트를 실험해왔습니다. 특히, 소프트웨어 테스팅 및 검증 연구 영역에서 LLM을 활성화하는 솔루션을 구조화하는 방법을 심층적으로 조사하여 실험적인 아키텍처를 설계하고 있습니다.

- **Technical Details**: 연구진은 80개의 논문을 분석하여 LLM 기반의 솔루션을 어떻게 구축하는지 검토합니다. 그들은 특히 '다운스트림 태스크(Downstream Task)' 개념이 프롬프트 기반 솔루션의 설계도를 전달하는 데 적합한지를 검증하고자 합니다. 또한, 이러한 솔루션에서 다운스트림 태스크의 수와 성격을 식별하기 위해 새로운 다운스트림 태스크 분류체계를 개발했습니다.

- **Performance Highlights**: 연구진은 소프트웨어 엔지니어링 문제의 다양한 스펙트럼을 아우르는 엔지니어링 패턴을 식별하기 위해 사용되는 새로운 분류 체계를 개발하여, 테스팅(testing), 퍼징(fuzzing), 디버깅(debugging), 취약점 감지(vulnerability detection), 정적 분석(static analysis), 프로그램 검증(program verification) 방법론에 적용 가능합니다.



### Deceptive Patterns of Intelligent and Interactive Writing Assistants (https://arxiv.org/abs/2404.09375)
Comments: Published as a workshop paper to the In2Writing workshop at CHI 2024

- **What's New**: 본 논문에서는 인공지능(AI) 글쓰기 보조 도구(AI writing assistants)에서 의도적으로 오도하는 사용자 인터페이스(UI) 및 사용자 경험(UX) 패턴들에 대한 연구를 제안하고 있습니다. OpenAI의 ChatGPT나 Microsoft의 Copilot와 같은 최신 대형 언어 모델(LLMs)이 활용되는 인터랙티브 시스템들에서 속임수 패턴(deceptive patterns)의 사용 가능성을 인식시키고자 하며, 이는 잠재적으로 사용자 의견에 영향을 미칠 수 있습니다.

- **Technical Details**: 이 연구는 기존의 속임수 UI/UX 패턴을 AI 글쓰기 도구에 적용합니다. 비싼 요금제(hidden costs), 반복적인 광고(nagging), 의도하지 않은 콘텐츠 삽입(sneaking) 및 인터페이스 간섭(interface interference) 등의 패턴들이 제안됩니다. 연구자들은 Gray et al. (2018)과 deceptive.design 웹사이트에서 언급된 기존 패턴을 활용하여 새로운 맥락에서의 속임수 패턴을 논의합니다.

- **Performance Highlights**: 이러한 패턴은 사용자가 제품에 더 많은 시간을 소비하도록 유도하거나, 추가 결제를 유도하거나, 특정 의견이나 제품을 부당하게 강조하여 사용자의 의견을 조작할 수 있습니다. 이러한 연구는 사용자에게 속임수 디자인의 존재를 인식시키고 UI/UX 디자인의 윤리적 사용을 강조하는 데 중요한 기여를 하고 있습니다.



### LLeMpower: Understanding Disparities in the Control and Access of Large  Language Models (https://arxiv.org/abs/2404.09356)
Comments: 11 total pages, 7 page text, 4 page references, 3 figures (with subfigures), 1 table

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 경제적 부담 및 접근성 문제를 조사합니다. 연구진은 세계 각국의 경제력과 LLM 개발 및 운영 비용을 비교 분석하며, 기술의 집중화가 소수의 기업에 의해 이루어지고 있음을 지적합니다. 연구 결과를 통해 LLM 접근성 개선을 위한 윤리적 시사점과 미래 방향에 대해 논의합니다.

- **Technical Details**: 연구는 다양한 LLM(예: GPT-4)의 훈련 및 추론(inference) 비용을 수집하고 분석하였습니다. LLMs는 막대한 컴퓨팅 자원과 에너지를 필요로 하며, 특수 하드웨어(GPU, TPU)가 필요합니다. GPT-4의 훈련 비용은 추정 $100 백만이고, 추론 비용은 하루에 $900 정도입니다. LLMs의 고비용은 경제력이 낮은 국가나 개인에게 접근 장벽으로 작용합니다.

- **Performance Highlights**: 연구 결과, LLMs의 고비용으로 인해 기술의 소수 집중화가 심화되고 있으며, 이는 일부 기업 및 국가에 권력과 통제가 집중됨을 의미합니다. 또한, 저개발 국가나 소규모 기업은 이러한 기술을 활용하고 개발하는 데 필요한 자원이 부족함을 발견하였습니다.



### TrafficVLM: A Controllable Visual Language Model for Traffic Video  Captioning (https://arxiv.org/abs/2404.09275)
- **What's New**: 이 논문에서는 새로운 멀티모달 덴스 비디오 캡셔닝 모델인 TrafficVLM을 제시합니다. 이 모델은 차량 자기 카메라(view) 관점에서 교통 비디오 이벤트를 공간적, 시간적으로 분석하고 사건의 다양한 단계에서 차량 및 보행자에 대한 세밀한 설명을 생성합니다. 또한, 생성 결과를 제어할 수 있는 조건부 컴포넌트(conditional component)와 다중 태스크 파인 튜닝 패러다임(multi-task fine-tuning paradigm)을 도입하여 TrafficVLM의 학습 능력을 향상시키고자 합니다. AI City Challenge 2024의 Track 2에서 이 솔루션은 뛰어난 성능을 보여주었으며 세 번째로 높은 순위를 얻었습니다.

- **Technical Details**: TrafficVLM은 비디오 언어 모델(video language model)로, 트래픽 안전 설명 및 분석 작업을 시간적 로컬라이제이션(temporal localization) 및 덴스 비디오 캡셔닝(dense video captioning) 작업으로 재구성합니다. 이 모델은 비디오 특징을 다양한 수준에서 모델링하고, 시계열 변환기(encoder)와 대형 언어 모델의 디코더(decoder)를 사용하여 파인 튜닝합니다. 또한, 멀티모달 덴스 비디오 캡셔닝에 대한 접근 방식을 적용하여 시각-텍스트 기능 간의 정렬(alignment)을 학습하게 합니다. TrafficVLM은 여러 타겟에 대한 캡션을 생성할 수 있는 controllable component를 포함하여 출력을 제어할 수 있습니다. 이 연구는 차량과 보행자 모두를 위한 트래픽 안전 시나리오의 상세한 비디오 캡셔닝을 가능하게 합니다.

- **Performance Highlights**: TrafficVLM은 AI City Challenge 2024 Track 2에서 뛰어난 결과를 달성하여 세 번째로 높은 순위에 올랐습니다. 이 모델은 비디오에서 교통 사고 상황을 정확하게 식별하고 설명하는 능력을 향상시키며, 복잡한 도시 환경에서의 다이내믹 상호 작용을 효과적으로 분석합니다. 또한, 공개적으로 코드를 제공하여 연구 및 개발 커뮤니티가 이 모델을 활용할 수 있도록 지원합니다.



### Test Code Generation for Telecom Software Systems using Two-Stage  Generative Mod (https://arxiv.org/abs/2404.09249)
Comments: 6 pages, 5 figures, Accepted at 1st Workshop on The Impact of Large Language Models on 6G Networks - IEEE International Conference on Communications (ICC) 2024

- **What's New**: Telecom 소프트웨어 시스템에 자동화된 테스트 생성 프레임워크를 제안함으로써, 우리는 다양한 배치 시나리오를 위한 소프트웨어 개발과 테스트의 복잡성을 줄이고자 합니다. 이 프레임워크는 시계열 생성 모델(time-series Generative model)을 사용하여 테스트 케이스 입력 데이터를 생성하고, 이 데이터는 자연어로 작성된 테스트 설명과 함께 사용하여 대규모 언어 모델(Generative Large Language Model)을 이용해 테스트 스크립트를 생성합니다. 이 방법은 통신 데이터의 프라이버시를 유지하면서 효율적인 테스트 케이스 데이터 입력 및 유용한 테스트 코드를 생성할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 우리는 역사적인 텔레콤 네트워크 데이터를 기반으로 훈련된 시계열 생성 모델을 사용하여 관찰된 테스트 시나리오를 위한 테스트 케이스 입력 데이터를 생성합니다. 이 데이터는 테스트 설명과 함께 사용되어 Generative Large Language Models를 통해 테스트 스크립트(Test Script)를 생성하는 데 사용됩니다. 사용된 모델은 5G 네트워크 및 O-RAN(O-Open RAN) 같은 새로운 네트워크 기술에 대한 복잡성을 관리하도록 설계되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 공개 데이터셋 및 운영 텔레콤 네트워크에서 얻은 텔레콤 데이터셋을 사용한 포괄적인 실험을 통해 효과적인 테스트 케이스 데이터 입력과 유용한 테스트 코드를 생성할 수 있는 것으로 나타났습니다. 특히 멀티벤더 지원이 요구되는 O-RAN 환경에서의 테스트 시나리오 커버리지 및 표현력 있는 데이터 생성이 중요하며, 우리의 방법은 이러한 요구 사항을 충족할 수 있는 것으로 평가되었습니다.



### Knowledgeable Agents by Offline Reinforcement Learning from Large  Language Model Rollouts (https://arxiv.org/abs/2404.09248)
- **What's New**: 이 논문은 강화 학습(RL)과 대규모 언어 모델(LLMs)의 통합을 위한 새로운 방법인 Knowledgeable Agents from Language Model Rollouts (KALM)를 소개합니다. KALM은 언어 모델로부터 상상의 롤아웃(imaginary rollouts)을 추출하여, 에이전트가 오프라인 강화 학습 방법을 통해 쉽게 학습할 수 있도록 합니다. 이는 언어 모델과 강화 학습의 결합에 있어서 의미 있는 도약을 나타내며, 에이전트가 복잡한 작업을 수행하고 새로운 기술을 반영하는 능력을 향상시킵니다.

- **Technical Details**: KALM은 LLM을 미세 조정하여 환경 데이터에 기반한 다양한 작업을 수행하고, 기술의 자연어 설명과 해당 롤아웃 데이터 간의 양방향 번역을 포함합니다. 이러한 'grounding' 과정은 LLM이 환경 역학을 이해하도록 돕고, 새로운 기술을 반영하는 다양하고 의미 있는 상상의 롤아웃을 생성할 수 있게 합니다.

- **Performance Highlights**: 초기 실증 평가에서 KALM은 CLEVR-Robot 환경에서 복잡한 과제 목표의 재구성을 완수하고, 전례 없는 최적 행동을 요구하는 새로운 과제에 대한 능력을 확장하는 데 성공했습니다. KALM은 보이지 않은 목표를 가진 작업을 수행하는 데 46%의 성공률을 달성하여, 기준 모델의 26% 성공률을 크게 상회하였습니다.



### TransformerFAM: Feedback attention is working memory (https://arxiv.org/abs/2404.09173)
Comments: 24 pages, 12 figures, 14 tables

- **What's New**: 새로운 트랜스포머 아키텍쳐인 'Feedback Attention Memory' (FAM)가 제안되었습니다. 이 아키텍처는 피드백 루프(feedback loop)를 활용하여 트랜스포머 내부의 잠재 표현(latent representations)에 주목함으로써, 무한히 긴 입력을 처리할 수 있게 합니다. TransformerFAM은 기존의 트랜스포머 모델과 원활히 통합될 수 있으며, 추가적인 가중치를 요구하지 않습니다.

- **Technical Details**: TransformerFAM은 각 트랜스포머 레이어에서 출력 활성화(output activation)를 동일 레이어의 입력으로 다시 사용하는 내부 트랜스포머 블록 피드백(approach within-transformer-block feedback) 방식을 채택하였습니다. 이러한 구조 변경을 통해 더 효과적으로 정보를 유지하며 작업 기억(work memory)이 자연스럽게 생성됩니다. TransformerFAM은 추론 시 계산 복잡도가 O(L), 메모리 복잡도가 O(1)이며, 이는 입력 토큰의 길이에 따라 결정됩니다.

- **Performance Highlights**: TransformerFAM은 다양한 모델 크기(1B, 8B, 24B)에서의 긴 컨텍스트 태스크에 대하여 기존 트랜스포머보다 뛰어난 성능을 보였습니다. 또한, LoRA를 사용하여 50k 스텝만에 미세 조정을 통해 성능을 크게 향상시켰습니다. 이는 TransformerFAM이 Large Language Models (LLMs)에서 무한한 길이의 시퀀스를 처리할 수 있는 유망한 솔루션이 될 수 있음을 시사합니다.



### Mitigating Heterogeneity among Factor Tensors via Lie Group Manifolds  for Tensor Decomposition Based Temporal Knowledge Graph Embedding (https://arxiv.org/abs/2404.09155)
- **What's New**: 이 연구는 텐서 분해(tensor decomposition) 기반의 시간 지식 그래프 임베딩(TKGE: Temporal Knowledge Graphs Embedding) 방법에서 요소 텐서(factor tensors) 사이의 내재된 이질성(heterogeneity) 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 저자는 요소 텐서들을 동질성(homogeneous)을 가지도록 리 군(Lie group) 다양체(manifold)에 매핑(mapping)하는 기법을 소개하고, 이를 통해 텐서 융합(tensor fusion)과 링크 예측(link prediction) 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 리 군 다양체를 사용하여 요소 텐서들의 분포를 동질적으로 만들어, 시간 지식 그래프의 이질적인 성질(예: 개체, 관계, 시간적 정보의 이질성)을 완화합니다. 이러한 매핑은 리 군의 특성(각 포인트에서 동일하게 보이는 구조와 접평면의 일관성)을 활용하여 요소 텐서들을 평활하고 균일하게 분포시키는 데 도움을 줍니다. 이론적 증명을 통해 동질적인 요소 텐서들이 TKGE 모델에서 이질적인 텐서들보다 더 효율적임을 입증하며, 이 기법은 추가적인 매개변수 없이 기존의 텐서 분해 기반 TKGE 모델에 직접 통합될 수 있습니다.

- **Performance Highlights**: 실험을 통해 이 방법이 요소 텐서들 사이의 이질성을 줄이고 TKGE 모델의 성능을 개선함을 보여줍니다. 여러 TKGE 모델에 대한 광범위한 실험에서 텐서 융합과 링크 예측의 정확도가 상당히 향상됨을 확인할 수 있습니다. 이는 제안된 방법이 시간 지식 그래프의 동적인 변화를 보다 정확하게 분석하고 이해하는 데 기여할 수 있음을 시사합니다.



### Provable Interactive Learning with Hindsight Instruction Feedback (https://arxiv.org/abs/2404.09123)
- **What's New**: 이 논문에서는 에이전트(예: 로봇)가 주어진 맥락(context)과 지시(instruction)에 따라 응답(예: 행동이나 궤적)을 생성해야 하는 상호작용 학습(interactive learning)을 연구합니다. 이는 전형적인 보상이나 전문가 감독하에 이루어지는 학습 방법과 다릅니다. 본 연구에서는 '후행 지시(hindsight instruction)' 학습을 통해 선생님(teacher)이 에이전트의 생성된 응답에 가장 적합한 지시를 제공하는 방식을 고안했습니다. 이 후행 지시 방법은 최적의 응답에 대한 전문가 감독을 제공하는 것보다 더 간단하게 제공될 수 있습니다. 이러한 방식으로의 학습을 이론적으로 분석하고, 새로운 알고리즘 LORIL을 소개하여 이 설정에 대한 이론적 이해를 시작합니다.

- **Technical Details**: 연구는 특정 상황에서 이론적인 하한선을 제공하여, 일반적으로 알고리즘의 후회(regret)가 에이전트의 응답 공간 크기에 따라 결정되어야 함을 보여줍니다. 또한, 학습과 어떠한 지시-응답 분포를 낮은 순위(low-rank) 행렬로 분해할 수 있는 특별한 상황을 연구합니다. 이 설정에서 LORIL 알고리즘을 소개하고, 해당 알고리즘의 후회가 𝑇의 제곱근(√𝑇)에 비례하여 증가함을 나타내며, 이는 지시(instruction), 맥락(context), 또는 에이전트의 응답 공간 크기에 의존하지 않고 교사의 분포의 순위(rank)에만 의존합니다.

- **Performance Highlights**: LORIL 알고리즘은 실험을 통해 낮은 등급 가정이 위반되었을 때에도 베이스라인을 능가하는 성능을 보여주었습니다. 첫 번째 실험에서는 합성 과제에서 낮은 등급 가정이 유효하며 LORIL이 베이스라인보다 낮은 후회를 달성했습니다. 두 번째 실험에서는 자연어 지시와 이미지 설정을 사용하여 LORIL이 낮은 등급 가정이 유효하지 않은 경우에도 유용한 통찰력을 제공함을 보여주었습니다.



### Semantic In-Domain Product Identification for Search Queries (https://arxiv.org/abs/2404.09091)
- **What's New**: 이 연구에서는 Adobe의 50개 이상의 제품과 수백 가지 도구를 다루는 검색 쿼리에서 명시적 및 암시적 제품 식별의 정확도를 향상시키기 위해 사용자 행동 데이터로부터 제품 분류기를 훈련하는 새로운 접근 방식을 제시합니다. 이 접근 방식은 새로운어도비 제품의 가시성을 높이는 데 도움을 주는 app card를 두 배로 늘렸습니다.

- **Technical Details**: 이 시스템은 정규 표현식 규칙과 간단한 NER(Named Entity Recognition) 기법을 사용하여 검색 쿼리와 제품을 매칭하던 이전 방식에서 발전하여, 지연 시간이 낮은 쿼리-제품 의미론적(semantic) 매칭 시스템을 사용합니다. 이를 위해 자체 문서 세트에서 언어 모델(Language Model, LM)을 사전 훈련시켜 Adobe 제품의 복잡성을 학습한 후, LM 위에 분류 접근법을 활용합니다.

- **Performance Highlights**: 이 새로운 시스템은 배포된 인터페이스에서 CTR(Click Through Rate)를 25% 이상 상대적으로 향상시켰으며, 무응답률은 50% 이상 감소했고, 앱 카드 활성화는 두 배로 증가했습니다. 이는 제품 가시성을 높이고 사용자 경험을 개선하는 데 크게 기여합니다.



### CodeCloak: A Method for Evaluating and Mitigating Code Leakage by LLM  Code Assistants (https://arxiv.org/abs/2404.09066)
- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)-기반 코드 보조 도구를 사용할 때 발생할 수 있는 코드 유출 위험을 완화하기 위한 두 가지 보완적인 방법을 제안합니다. 첫 번째 방법은 개발 과정에서 IDE(Integrated Development Environment, 통합 개발 환경)에서 코드 보조 서비스로 전송된 프롬프트에서 개발자의 원래 코드베이스를 재구성하는 기술이며, 이를 통해 제3자(또는 적대자)에게 코드 유출의 정도를 평가할 수 있습니다. 두 번째 방법은 CodeCloak이라는 새로운 딥 강화 학습(DRL; Deep Reinforcement Learning) 에이전트로, 코드 보조 서비스로 전송되기 전에 프롬프트를 조작합니다. 이 두 가지 방법을 통해 코드 유출을 최소화하면서도 개발자에게 유용하고 관련성 있는 제안을 유지하려고 합니다.

- **Technical Details**: CodeCloak은 딥 강화 학습을 이용하여 프롬프트의 코드 요소를 조작함으로써 코드 보안을 강화하는 기술입니다. 이 조작에는 함수 제거, 변수 또는 함수 이름 변경, 코드를 텍스트 요약으로 대체하는 등의 작업이 포함됩니다. CodeCloak의 목표는 코드 유출 최소화 및 개발자의 독점적 코드 노출 방지와 동시에 유용한 코드 제안을 유지하는 것입니다. GitHub Copilot, StarCoder, CodeLlama와 같은 여러 LLM 기반 코드 보조 모델을 사용하여 평가한 결과, CodeCloak은 원래 프롬프트와 조작된 프롬프트 사이에서 평균 유사성이 71%를 유지하면서 평균 유출 감소율이 44%임을 보여주었습니다.

- **Performance Highlights**: 이 연구에서 제안된 코드 재구성 방법은 평균 90%의 재구성 성공률을 달성하여 코드 유출의 심각성을 입증하였습니다. 또한, CodeCloak은 다양한 코드 저장소 크기 및 다양한 모델에 걸쳐 효과적인 것으로 나타났으며, 실제 코딩 환경을 시뮬레이션하여 코드 유출 위험과 완화 기법의 효과를 철저히 분석할 수 있는 가상 코딩 환경을 생성하였습니다. 결과적으로, 개발자들이 코드 보안을 유지하면서도 효율적으로 코드를 개선할 수 있는 강력한 도구를 제공합니다.



### Navigating the Landscape of Large Language Models: A Comprehensive  Review and Analysis of Paradigms and Fine-Tuning Strategies (https://arxiv.org/abs/2404.09022)
- **What's New**: 이 논문은 Transformer 아키텍처를 기반으로 한 대규모 언어 모델의 미세조정 방법에 대한 포괄적인 검토를 제공합니다. 특히, task-adaptive fine-tuning (태스크 적응형 미세조정), domain-adaptive fine-tuning (도메인 적응형 미세조정), few-shot learning (퓨샷 학습), knowledge distillation (지식 증류), multi-task learning (멀티태스크 학습), parameter-efficient fine-tuning (파라미터 효율적 미세조정), 및 dynamic fine-tuning (동적 미세조정)에 대한 최신 기술 발전과 방법 적용을 조사합니다.

- **Technical Details**: Transformer 모델은 순환 신경망(RNNs)과 합성곱 신경망(CNNs)이 장기 의존성을 처리하는 데 제한적이라는 문제를 해결하기 위해 개발되었습니다. 이 모델은 attention mechanism을 활용하여 입력과 출력 간의 글로벌 의존성을 파악합니다. 특히, encoder는 입력 시퀀스를 처리하고, decoder는 이를 바탕으로 출력 시퀀스를 생성하는 역할을 합니다. 미세조정은 사전 훈련된 모델을 특정 작업이나 도메인에 맞게 조정하여 모델의 일반화 능력을 향상시키는 방식입니다. 이 연구는 BERT, GPT 시리즈 등 여러 중요 모델을 다루면서 이들이 어떻게 다양한 NLP 작업에서 뛰어난 성능을 달성했는지 설명합니다.

- **Performance Highlights**: 이 논문은 LoRA fine-tuning 패러다임을 사용하여 여섯 가지 텍스트 분류 데이터 세트에서 모델 크기와 미세조정 방법을 비교한 실험 결과를 제공합니다. 실험은 GitHub에서 공개적으로 코드를 제공하며, 이를 통해 연구자들이 결과를 재현하고 추가적인 실험을 수행할 수 있습니다.



### AMU-Tuning: Effective Logit Bias for CLIP-based Few-shot Learning (https://arxiv.org/abs/2404.08958)
Comments: Accepted by CVPR 2024

- **What's New**: 이 논문은 특히 CLIP(예: CLIP)와 같은 사전 트레이닝된 시각-언어 모델을 사용한 few-shot 학습의 효과성에 큰 잠재력을 발견하였습니다. 연구자들은 CLIP 기반의 few-shot 학습 방법들을 분석하고 효과적인 'logit bias'를 학습하여 성능을 개선하기를 제안합니다. 새로운 AMU-Tuning 방법론은 보조적 특징(Auxiliary features), 멀티-브랜치 트레이닝, 불확실성 기반 융합(Uncertainty-based fusion)을 사용하여 logit bias를 계산합니다.

- **Technical Details**: AMU-Tuning은 시각-언어 모델 CLIP을 대상으로 하는 보조적 특징을 사용하여 logit bias를 예측하고, 멀티-브랜치 트레이닝을 포함한 효율적인 특징 초기화 선형 분류기(feature-initialized linear classifier)를 사용하여 logit 예측기의 성능을 향상시킵니다. 불확실성 기반의 융합은 제로 샷 CLIP의 예측 신뢰성을 고려하여 logit bias를 CLIP에 적응적으로 통합합니다.

- **Performance Highlights**: AMU-Tuning은 여러 벤치마크에서 기존 방법들을 뛰어넘는 성능을 보여주며, CLIP 기반의 few-shot 학습에서 최고의 성능을 달성했습니다. 이 방법은 다양한 다운스트림 작업과 out-of-distribution 벤치마크에서 효과적인 결과를 이끌어 냈으며, 계산 비용도 효율적입니다.



### Introducing Super RAGs in Mistral 8x7B-v1 (https://arxiv.org/abs/2404.08940)
- **What's New**: Super RAGs(슈퍼 RAGs)는 LLMs를 위한 향상된 방법론으로, Mistral 8x7B v1과 같은 최신의 대규모 언어 모델에 통합되어 모델의 정확도, 속도 및 사용자 만족도를 크게 향상시킬 수 있는 가능성을 보여주었습니다. 이 연구는 RAG(검색 증강 생성) 시스템을 한 단계 발전시켜, 기존의 한계를 극복하고 언어 모델의 성능을 혁신적으로 끌어올리는 Super RAG를 소개합니다.

- **Technical Details**: Super RAG는 외부 데이터 소스를 활용하여 언어 모델의 생성 과정을 풍부하게 하여 정확성과 관련성을 높입니다. 이는 Sparse Mixture-of-Experts(SMoE) 기법과 함께 적용되어, 정보 검색 및 통합 과정에서의 효율성을 극대화합니다. Instruct model 설정과 cache tuning fork 시스템을 통해 세심하게 조정되어, 최적의 데이터 검색을 보장합니다.

- **Performance Highlights**: 연구 결과, Super RAGs가 통합된 Mistral 8x7B v1은 전반적인 메트릭스에서 높은 점수를 보여주었습니다. 특히, 정보의 정확도와 처리 속도에서 눈에 띄는 개선이 확인되었으며, 이는 Super RAG의 고도화된 문서 검색 및 선택 메커니즘 덕분입니다. 또한, 사용자 만족도 조사에서도 긍정적인 반응을 얻었음을 보여줍니다.



### EIVEN: Efficient Implicit Attribute Value Extraction using Multimodal  LLM (https://arxiv.org/abs/2404.08886)
Comments: Accepted by NAACL 2024 Industry Track

- **What's New**: EIVEN은 기존 방법을 능가하는 새로운 데이터 및 매개변수 효율적인 다중 모드 생성 프레임워크를 제시합니다. 이 프레임워크는 묵시적 속성 값 추출을 위해 다중 모달 LLM(Large Language Models)을 사용하는 것이 특징입니다. 이는 추출 작업에서 레이블이 많이 필요하지 않으며, 제안된 'Learning-by-Comparison' 기법은 유사한 속성 값들 사이의 혼돈을 줄이는 데 도움을 줍니다.

- **Technical Details**: 'Learning-by-Comparison'은 같은 속성을 공유하지만 다를 수 있는 속성 값을 가진 인스턴스 쌍을 모델에 제공하여 모델이 비교하고 구별하도록 강제합니다. EIVEN은 사전 훈련된 LLM과 시각 인코더(visual encoder)의 풍부한 기본 지식을 활용합니다. 이를 통해 속성별 많은 데이터에 대한 의존도를 줄이면서 정확한 속성 값을 추출할 수 있습니다. 또한, 다중 모드 데이터에서 묵시적 속성 값을 추출하기 위한 개방형 데이터 세트를 처음으로 구축했습니다.

- **Performance Highlights**: EIVEN은 기존 다중 모드 속성 값 추출 방법보다 훨씬 우수한 성능을 보였습니다. 이는 특히 레이블이 적은 데이터를 요구하는 경우에도 높은 성능을 유지하기 때문에 주목할 만합니다. 광범위한 실험을 통해 EIVEN이 묵시적 속성 값 추출 작업에서 새로운 표준을 설정했음을 입증하였습니다.



### Is Next Token Prediction Sufficient for GPT? Exploration on Code Logic  Comprehension (https://arxiv.org/abs/2404.08885)
- **What's New**: 이 연구에서는 대형 언어 모델들(Large Language Models, LLMs)이 코드의 논리를 이해하는 능력에 한계를 드러내는 새로운 발견을 제시합니다. 특히, '논리적으로 동등한 코드 선택(Logically Equivalent Code Selection)'이라는 새로운 작업을 도입하여, 기존의 토큰 예측 작업들이 코드의 논리적 이해를 충분히 지원하지 못함을 보여줍니다. 이를 통해, 모델이 코드를 단순한 텍스트로 해석하는 경향이 있음을 지적하며, 코드의 논리적 구조를 더 잘 이해할 수 있도록 하는 'Next Token Prediction+'라는 새로운 예비 트레이닝 작업을 제안합니다.

- **Technical Details**: 새로 제안된 'Next Token Prediction+' 작업은 기존의 다음 토큰 예측 작업(Next Token Prediction) 형식을 따르면서, 문장 임베딩 분포(sentence embedding distribution)를 변경하는 것을 목표로 합니다. 이를 통해 LLM이 코드의 논리적 구조를 보다 효과적으로 학습할 수 있도록 지원합니다. 이 작업에는 원본 코드, 어퓨제이티드(obfuscated) 코드, 그리고 라인 셔플(line-shuffled) 코드를 포함하여, 긍정적 및 부정적 샘플로 구성됩니다.

- **Performance Highlights**: 'Next Token Prediction+'를 적용한 후, Code Llama와 StarCoder와 같은 유명한 코드 도메인 사전훈련 모델들은 '논리적으로 동등한 코드 선택' 작업과 코드 완성 작업에서 두드러진 성능 향상을 보였습니다. 평균 성능 향상은 Code Llama 7b, 13b 및 StarCoder 15b에서 각각 22.95%, 23.23%, 23.99%로 나타났습니다.



### Aligning LLMs for FL-free Program Repair (https://arxiv.org/abs/2404.08877)
- **What's New**: 이 연구에서는 오류 수정 프로그램 (APR)을 향상시키기 위해 대규모 언어 모델 (LLM)의 새로운 활용 방법을 제안합니다. 'D4C'라는 새로운 프레임워크를 통해, 전통적인 결함 지역화 (fault localization) 절차를 없애고, LLM이 전체 프로그램을 자체적으로 수정할 수 있도록 함으로써, APR의 효율성을 크게 향상시켰습니다.

- **Technical Details**: D4C는 기존의 LLM 트레이닝 목적과 출력 형식을 맞추고, 복잡한 프롬프팅 (prompting)이나 추가 훈련 없이도 높은 성능을 달성합니다. 디버그 정보 같은 관련 아티팩트(artifacts)를 입력으로 사용하여, LLM이 프로그램의 버려진 부분을 정확히 감지하고 수정하는 방식을 통해 APR을 수행합니다.

- **Performance Highlights**: D4C는 Defects4J 데이터 세트에서 테스트된 437개의 단일 기능 버그 중 180개를 성공적으로 수정하였고, 이는 완벽한 결함 지역화를 이용한 최신의 APR 방법보다 10% 더 높은 성능을 나타냅니다. 각 패치는 단 10회의 샘플링만으로 생성되었으며, 이는 기존 방법에 비해 90% 적은 수치입니다.



### Experimental Design for Active Transductive Inference in Large Language  Models (https://arxiv.org/abs/2404.08846)
- **What's New**: 이 연구에서는 큰 언어 모델(Large Language Models, LLM)의 새로운 사용 방식을 제안합니다. 본 논문에서 소개된 적응형 프롬프트 디자인 프레임워크인 '능동 추론 전이(Active Transductive Inference, ATI)'는 특정 추론 쿼리(query)에 대해 몇 개의 예시를 적응적으로 선택함으로써 LLM 프롬프트를 설계합니다. 이 예시들은 초기에 라벨이 없으며, 사용자에게 가장 유익한 정보를 제공하는 예시의 라벨을 요청함으로써 LLM 예측의 불확실성을 최대한 줄입니다.

- **Technical Details**: ATI에는 두 가지 주요 알고리즘이 포함되어 있습니다: G-Optimal design algorithm (GO)과 Simulation-Based Active Learning algorithm (SAL). GO는 예제의 불확실성을 최소화하는 가장 가까운 예제에 대한 사용자의 라벨링을 요구하는 반면, SAL은 라벨이 지정되지 않은 예제의 라벨링이 불확실성에 미치는 영향을 시뮬레이션으로 추정합니다. 이 연구는 선형 모델(linear models)에서 이 두 알고리즘이 어떻게 동등한지를 분석하고 각각의 알고리즘이 LLMs에 적용될 때의 효능을 평가합니다.

- **Performance Highlights**: GO와 SAL은 다양한 작업과 데이터셋(UCI, OpenML, 사용자 정의 NLP 데이터셋, 추상적 추론(corpus) 작업 및 Probabilistic Context Free Grammar (PCFG) 작업)에서 기존 few-shot 예제 선택 방법들을 일관되게 능가하는 결과를 보여줍니다. 이러한 성능은 능동 학습(active learning)과 추론 전이(transductive inference)를 LLMs에 효율적으로 적용하는 것의 유용성을 입증합니다.



### The Illusion of State in State-Space Models (https://arxiv.org/abs/2404.08819)
Comments: Preprint

- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 위한 아키텍처로서 state-space models (SSMs)의 표현력을 조사합니다. 이전에는 transformer 아키텍처가 널리 사용되었으나, SSMs는 순차적 연산 및 상태 추적 측면에서 transformer를 능가할 가능성이 있다고 제시되었습니다. 그러나 연구 결과에 따르면 SSMs는 transformer와 유사한 한계를 가지며, 순차적인 문제나 간단한 상태 추적 문제(예: 순열 조합)를 해결하는 능력이 매우 제한적임이 밝혀졌습니다.

- **Technical Details**: SSMs와 transformer의 표현력을 분석한 결과, 이 두 아키텍처 모두 $	ext{TC}^0$ 복잡도 클래스에 속함을 확인했습니다. 이는 SSMs가 RNN처럼 순차적 문제를 해결하는데 있어 더 우수하지 않다는 것을 시사합니다. 추가적으로, SSMs는 특히 순열 조합과 같은 간단한 상태 추적 문제도 해결하는데 어려움을 겪는다는 실험적 증거도 제공되었습니다.

- **Performance Highlights**: 실험 결과, linear 혹은 Mamba-style SSM 아키텍처는 transformer와 비슷한 성능을 보였으며, RNN에 비해 상당히 뒤쳐졌습니다. 특히 숫자의 순열을 조합하는 문제에서 SSMs와 transformer 모두 제한된 계층 수를 가진 상태에서는 문제 해결 능력이 매우 낮았던 반면, RNN은 단일 계층만으로도 문제를 해결할 수 있었습니다.



### Megalodon: Efficient LLM Pretraining and Inference with Unlimited  Context Length (https://arxiv.org/abs/2404.08801)
Comments: 9 pages, 6 figures and 8 tables

- **What's New**: 새로운 모델인 Megalodon이 소개됩니다. 이는 기존의 트랜스포머 아키텍처의 한계를 극복하고, 긴 시퀀스 모델링에서 뛰어난 성능과 효율성을 보여주기 위해 개발되었습니다. Megalodon은 Mega 아키텍처를 계승하여 복잡한 지수 이동 평균(CEMA), 타임스텝 정규화 레이어, 정규화된 주의 메커니즘, 두 홉 잔여 구성의 사전 정규화를 도입하는 등 여러 기술적 개선을 통해 안정성과 성능을 향상시켰습니다.

- **Technical Details**: Megalodon은 클래식한 이동 평균(EMA) 및 게이트 주의 메커니즘을 사용하는 Mega 아키텍처를 기반으로 하며, 여기에 복잡한 지수 이동 평균(CEMA), 타임스텝 정규화 레이어, 정규화된 주의 메커니즘 등을 추가하여 긴 시퀀스를 효과적으로 다룰 수 있도록 합니다. 이 아키텍처는 입력 시퀀스를 고정된 블록으로 나눔으로써 모델 훈련 및 추론 시 계산 및 메모리 복잡성을 선형으로 유지합니다.

- **Performance Highlights**: Megalodon-7B는 Llama2-7B 모델과의 비교에서 훈련 당혹도(perplexity) 및 다양한 하류 작업에서 더 나은 성능을 보였습니다. 이는 7억 개의 파라미터와 2조 개의 훈련 토큰 스케일에서 트랜스포머보다 효율적인 성능을 제공함을 의미합니다. 또한, Megalodon은 다양한 문맥 길이와 내용 QA 작업에서 긴 시퀀스 모델링 능력을 입증하였습니다. 더 작고 중간 규모의 벤치마크에서도 일관된 성능 향상을 보여줍니다.



### JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large  Language Models (https://arxiv.org/abs/2404.08793)
Comments: Submitted to VIS 2024

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 보안 취약점, 특히 재일브레이크 공격(jailbreak attacks)을 방지하기 위한 새로운 시각 분석 시스템인 JailbreakLens를 제안합니다. 이 시스템은 사용자가 대상 모델에 대한 재일브레이크 성능을 탐색하고, 프롬프트(prompt) 특성을 다중 레벨로 분석하며, 프롬프트 인스턴스를 정제하여 발견 사항을 검증할 수 있도록 지원합니다.

- **Technical Details**: JailbreakLens는 LLM의 방어 능력을 평가하고 잠재적 약점을 식별하기 위해 자동 재일브레이크 평가를 제공하는 LLM-지원 프레임워크를 기반으로 합니다. 이 시스템은 사용자가 프롬프트의 구성 요소와 키워드를 분석하고, 성능 평가를 용이하게 할 수 있도록 설계되었습니다. 프롬프트의 중요 키워드를 요약하고, 사용자가 자신의 전문 지식에 따라 재일브레이크 프롬프트 인스턴스를 자유롭게 수정할 수 있는 기능도 제공합니다.

- **Performance Highlights**: 기술 평가, 사례 연구 및 전문가 인터뷰를 통해 JailbreakLens 시스템의 효과성이 입증되었습니다. 이 시스템은 모델 보안을 평가하고 모델의 약점을 식별하는 데 사용자를 도울 수 있는 것으로 나타났습니다. 특히, 재일브레이크 프롬프트의 성능을 개선할 수 있는 방안을 분석하여 모델 약점을 식별하는 데 중요한 역할을 합니다.



### CATS: Contextually-Aware Thresholding for Sparsity in Large Language  Models (https://arxiv.org/abs/2404.08763)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)의 활성화를 희소화하고 추론 비용을 줄이기 위한 새로운 프레임워크인 CATS(Contextually Aware Thresholding for Sparsity)를 제안합니다. CATS는 비선형 활성화 함수를 중심으로 구현되며, 기본 모델들인 Mistral-7B와 Llama2-7B에 적용하여 기존의 희소화 기술들을 능가하는 성능을 보여줍니다.

- **Technical Details**: CATS는 입력에 기반하여 동적으로 활성화할 '전문가'를 선택하는 MoE(Mixture of Experts) 기법과 유사한 접근 방식을 사용하고, 새로운 비선형 활성화 함수를 통해 활성화의 희소성을 조절할 수 있게 합니다. 또한, CATS는 50%의 활성화 희소도에서도 기본 모델의 성능과 1-2% 이내의 차이로 하위 작업을 수행할 수 있으며, 미세 조정(fine-tuning)을 적용할 때는 기존 기법보다 더 빠른 수렴 속도와 개선된 작업 성능을 보입니다.

- **Performance Highlights**: CATS 기반 모델은 토큰 생성 작업에서의 월 클럭(wall-clock) 추론 지연 시간을 약 15% 개선하였습니다. 이는 사용자 정의 GPU 커널을 통해 CATS의 활성화 희소성을 효율적으로 구현함으로써 달성되었습니다. CATS는 뛰어난 확장성과 범용성을 지녔으며, 다양한 LLM 베이스라인 모델에 적용 가능합니다.



### Can LLMs substitute SQL? Comparing Resource Utilization of Querying LLMs  versus Traditional Relational Databases (https://arxiv.org/abs/2404.08727)
Comments: 13 pages, 2 figures, 5 tables

- **What's New**: LLM(Large Language Models)이 소프트웨어 공학 프로세스에서 다양한 작업을 자동화하거나 대체할 수 있는 가능성에 대해 연구되었습니다. 이 연구는 LLM이 전통적인 SQL(Structured Query Language)을 사용하는 관계형 데이터베이스 관리 시스템과 비교하여 자연어 질의를 해석하고 실행하는 데 있어 자원 사용 및 정확도를 평가합니다.

- **Technical Details**: 연구팀은 7B에서 34B까지 다양한 파라미터를 가진 9개의 LLM, 예를 들어 Llama2 7B, Llama2 13B, Mistral, Mixtral, Optimus-7B, SUS-chat-34B, platypus-yi-34b, NeuralHermes-2.5-Mistral-7B 및 Starling-LM-7B-alpha를 사용하여 소규모 거래 데이터셋에서 자원 사용량과 정확도를 실증적으로 검토했습니다.

- **Performance Highlights**: 연구 결과에 따르면 데이터베이스 쿼리에 LLM을 사용하면 상당한 에너지 오버헤드가 발생하여 환경에 친화적이지 않은 방법임이 나타났습니다. LLM은 SQL 엔진에 비해 높은 에너지를 소모할 뿐만 아니라 정확성도 떨어질 수 있습니다. 따라서 연구팀은 LLM이 관계형 데이터베이스를 대체하는 것은 권장하지 않습니다.



### Exploring Contrastive Learning for Long-Tailed Multi-Label Text  Classification (https://arxiv.org/abs/2404.08720)
Comments: 14 pages, 2 figures

- **What's New**: 이 논문에서는 다중 레이블 텍스트 분류(Multi-label Text Classification, MLTC)에서의 효과적인 표현을 학습하는 것에 중점을 두고 있습니다. 특히, 이 논문은 감독된 대조학습(Supervised Contrastive Learning)을 기존의 감독 손실 함수와 통합하는 접근 방식의 가능성을 새로이 조사합니다. MLTC의 복잡성을 감안할 때, '긍정 샘플 부족(lack of positives)'과 '인력-반발력 불균형(attraction-repulsion imbalance)'이라는 두 가지 주요 도전을 해결하기 위해, 새로운 대조 손실 함수를 도입합니다.

- **Technical Details**: 이 연구는 다중 레이블 데이터셋에서 긴 꼬리 분포(long-tailed distribution)를 고려하면서, MLTC에서의 대조학습의 영향을 깊이 있게 분석하고 있습니다. 대조학습은 텍스트의 본질적인 이산성(discreteness)으로 인해 데이터 증강(data augmentation) 기법 선택이 어렵고, 레이블 간 상호작용 때문에 긍정적인 문서 쌍을 정의하기 어렵다는 점에서 MLTC에 적용하기 복잡합니다. 연구는 새로운 대조적 손실 함수를 도입하여 Micro-F1과 Macro-F1 점수 모두에서 기존 손실 함수보다 뛰어나거나 동등한 성능을 보이는 것을 목표로 합니다.

- **Performance Highlights**: ABALONE이라는 새로운 접근법은 세 가지 다중 레이블 데이터셋에서 Macro-F1 점수에서 상당한 향상을 보였으며, Micro-F1 점수는 기존의 손실 함수들과 비견되거나 그 이상의 결과를 얻었습니다. 이는 본 연구에서 제안하는 접근법이 MLTC에서의 표현 공간을 효과적으로 개선하고, 특히 긴 꼬리 분포를 가진 데이터셋에 대해 더 강력한 모델 일반화(generalization) 능력을 제공할 수 있음을 시사합니다.



### Large Language Model Can Continue Evolving From Mistakes (https://arxiv.org/abs/2404.08707)
- **What's New**: 이 연구에서는 기존의 연속 학습(Continual Learning, CL) 방식을 개선하여 대규모 언어 모델(Large Language Models, LLMs)의 지속적인 업데이트와 오류 수정이 가능한 ‘Continue Evolving from Mistakes (CEM)’ 방법을 제안합니다. 이 방법은 학생들이 실수를 요약하고 배우는 방식에서 영감을 받아 모델의 지식 결핍을 체계적으로 식별하고 수정함으로써 모델 성능을 개선합니다.

- **Technical Details**: CEM 방법은 LLM이 잘못된 응답을 생성했을 때, 이를 통해 모델의 지식 결핍을 파악하고, 관련 지식을 보완하기 위해 추가적인 학습을 반복적으로 수행합니다. 이를 위해 다양한 데이터 소스에서 지식을 수집하고, 보충 학습 세트를 구축하는 두 가지 전략을 개발하여 모델이 코퍼스(Corpus)를 이해하고 지식을 잊어버리는 것을 방지하도록 합니다.

- **Performance Highlights**: CEM 방법을 사용하여 세 개의 오픈 소스 LLM과 두 개의 복잡한 지식 기반 질의응답(Question-Answering, QA) 작업에 대한 실험을 수행한 결과, 최적의 경우 모델의 정확도가 17.00% 향상되었습니다. 이는 지속적인 지식 업데이트와 보충을 통해 LLM이 보다 정확하고 신뢰할 수 있는 응답을 생성할 수 있음을 시사합니다.



### Apollonion: Profile-centric Dialog Agen (https://arxiv.org/abs/2404.08692)
- **What's New**: 이 논문에서는 대화형 에이전트(dialog agents)가 사용자의 특성을 고려하여 개성화된(personalized) 대화를 구현할 수 있는 새로운 프레임워크 'Apollonion'을 제안합니다. 기존의 대화형 에이전트들이 제공하는 일반적이고 단일화된 응답에서 벗어나, 사용자 프로파일링(user profiling)과 지속적인 업데이트를 통해 더 개인화된 응답을 제공하려는 시도가 특징입니다. 아폴로니온(Apollonion)은 고대 그리스의 신전에서 유래한 이름으로, '너 자신을 알라(Know Yourself)'는 철학적 배경에 기초하여, 각 사용자의 독특한 특성을 이해하고 반영하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 연구에서 제안한 'Apollonion' 프레임워크는 크게 사용자 쿼리 분석 및 응답을 구조화된 사용자 프로파일로 조직화하는 기능과, 이 프로파일을 사용하여 더 정밀하고 개인화된 응답을 생성하는 기능으로 구성됩니다. 프로파일은 동적으로 업데이트되며, 사용자의 관심사나 선호도 등을 지속적으로 반영하게 됩니다. 또한, 대화형 에이전트가 사용자의 특성을 얼마나 잘 반영하는지를 측정하기 위한 여러 평가 프로토콜을 제안하여, 개인화 수준을 평가할 수 있는 기준을 마련하고 있습니다.

- **Performance Highlights**: 아폴로니온 프레임워크는 사용자의 세부적인 특성을 고려함으로써 개인화된 대응을 가능하게 하는 것을 목표로 합니다. 이는 기존 대화형 시스템에서 볼 수 없던 새로운 접근법이며, 사용자 경험을 향상시킬 수 있는 잠재력을 가지고 있습니다. 이 프레임워크를 통해 구현된 에이전트는 더 정확하고 관련성 높은 응답을 제공하며, 사용자 맞춤형 서비스를 제공하는 측면에서 탁월한 성능을 보일 것으로 기대됩니다.



### PMG : Personalized Multimodal Generation with Large Language Models (https://arxiv.org/abs/2404.08677)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 사용하여 개인화된 멀티모달 생성(Personalized Multimodal Generation, PMG)을 제안합니다. 그동안 다양한 분야에서 멀티모달 생성이 주목을 받았지만, 개인화 생성에 대한 연구는 많지 않았습니다. 이 연구는 고객의 개인 선호를 반영한 맞춤형 콘텐츠 생성에 초점을 맞추고 있으며, 이를 통해 추천 시스템 등에 응용할 수 있습니다.

- **Technical Details**: PMG 방법론은 사용자 행동(예: 추천 시스템에서의 클릭, 가상 어시스턴트와의 대화 등)을 자연어로 변환하여 LLM이 이해하기 쉽게하고, 사용자의 선호도를 추출합니다. 이후, 이 사용자 선호는 멀티모달 LLM 또는 확산 모델(diffusion model)과 같은 생성기에 입력되어 개인화된 콘텐츠를 생성합니다. 특히, 사용자 선호를 명시적 키워드(explicit keywords)와 함축적 임베딩(implicit embeddings)의 조합으로 출력하고, 이를 생성기의 프롬프트로 사용하여 더 정확하게 사용자의 선호를 반영하도록 하였습니다.

- **Performance Highlights**: 실험 결과, PMG는 개인화 개선 면에서 기존 방법론 대비 최대 8% 개선된 성능을 보였으며, 생성 정확도(accuracy)도 유지되었습니다. 이는 LPIPS 점수를 통해 검증되었습니다. 여기서 LPIPS(Learned Perceptual Image Patch Similarity)는 생성된 이미지의 다양성과 사실성을 평가하는 지표입니다.



### RecGPT: Generative Personalized Prompts for Sequential Recommendation  via ChatGPT Training Paradigm (https://arxiv.org/abs/2404.08675)
- **What's New**: 이 논문은 ChatGPT의 교육 패러다임을 적용하여 아이템 시퀀스 예측에 사용한다는 새로운 접근 방식을 제시합니다. 'RecGPT'라는 새로운 프레임워크는 사용자의 행동 시퀀스와 사용자의 피드백을 기반으로 개인화된 프롬프트를 도입하여 추천 시스템에 적용하였습니다.

- **Technical Details**: RecGPT는 Generative Pre-training Transformer (GPT) 모델을 기반으로 하며, 사용자 ID 모듈을 통해 개인화 된 정보를 캡처합니다. 훈련은 프리-트레이닝과 파인-튜닝의 두 단계로 구성되며, 프롬프트 튜닝과 인퍼런스-검증 단계를 포함합니다. 이는 사용자의 시간에 따른 선호도 변화를 보다 잘 포착할 수 있도록 자동 회귀적(recall) 접근 방식을 사용합니다.

- **Performance Highlights**: RecGPT는 온라인 및 오프라인 데이터셋에서 모두 효과를 입증했습니다. 특히 Kuaishou 앱의 A/B 테스트에서 자동 회귀적(recall) 방법을 채택해 온라인 추천에서의 타당성을 입증했습니다. 이는 챗봇(ChatGPT)의 온라인 추천에 대한 새로운 관점을 제공하며, 학계와 산업계에서의 후속 연구를 촉진합니다.



### Taxonomy and Analysis of Sensitive User Queries in Generative AI Search (https://arxiv.org/abs/2404.08672)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs) 기반의 대화형 서비스를 운영하는데 대한 경험을 공유하며, 특히 사용자 질의의 민감성에 중점을 둔 연구입니다. 한국의 국가 규모 검색 엔진에서 LLM을 활용해 과민 반응을 분석하고 대응하는 방법을 제시합니다.

- **Technical Details**: 연구팀은 과민 반응을 식별하기 위해 민감한 질의에 대한 분류법(taxonomy)을 개발하고, 이를 통해 사용자의 질의를 분석했습니다. 또한, Transformer 및 LLM 기술을 활용하여 검색 결과를 생성하고, 검색 의도를 파악하기 위해 인간의 사고방식을 모방하는 시스템을 구현했습니다.

- **Performance Highlights**: 이 시스템은 비공개(unsafe) 질의에 대한 완화 전략을 적용하여 사용자와의 상호 작용 중 안전을 최우선으로 하는 답변을 생성합니다. 또한, 민감한 질의의 분포와 키워드를 분석하여 사회적 이슈에 어떻게 반응하는지 상세하게 보고하였습니다.



### Targeted aspect-based emotion analysis to detect opportunities and  precaution in financial Twitter messages (https://arxiv.org/abs/2404.08665)
- **What's New**: 이 연구는 트위터(Twitter)와 같은 마이크로블로깅 플랫폼에서 재정적 감정(긍정적 및 부정적 전망)을 개별적으로 식별할 수 있는 새로운 시스템인 타기팅 어스펙트-기반 감정 분석(Targeted Aspect-Based Emotion Analysis, TABEA)을 제안합니다. 이 시스템은 다양한 주식 시장 자산에 대해 감정을 분석하여 금융결정 지원에 실용적인 기여를 합니다.

- **Technical Details**: TABEA 시스템은 자연 언어 처리(Natural Language Processing, NLP) 기술과 온라인 머신 러닝(online Machine Learning) 스트리밍 알고리즘을 기반으로 하며, 트윗을 파싱하여 간단한 선언적 절로 분할하는 구문 분석(module) 모듈, 텍스트와 숫자, 범주형 특징을 처리하고 관련성에 따라 분석 및 선택하는 오프라인 데이터 처리 모듈, 실시간으로 트윗을 지속적으로 처리하는 스트림 분류 모듈로 구성됩니다.

- **Performance Highlights**: 이 시스템은 타깃 감정인 '금융 기회(financial opportunity)'와 '주의(precaution)'에 대해 90% 이상의 정밀도(precision)를 달성하였습니다. 이는 감정분석 및 의견 추출에 기반한 기존의 접근과 다른 새로운 접근 방식을 통해 금융 자산에 대한 정교한 감정 분석을 가능하게 한 주요 성과입니다.



### Identifying Banking Transaction Descriptions via Support Vector Machine  Short-Text Classification Based on a Specialized Labelled Corpus (https://arxiv.org/abs/2404.08664)
- **What's New**: 이 논문에서는 은행 거래 설명의 자동 분류라는 새로운 문제를 다루고 있습니다. 이는 개인 금융 관리를 위해 자연어 처리(NLP) 기법과 기계 학습(Machine Learning, ML) 알고리즘을 결합한 시스템을 사용하여 짧은 텍스트 분류를 수행합니다. 특히, 실시간으로 생성되는 짧은 텍스트의 효율적인 분류 방법이 필요함을 강조하고, Jaccard 거리를 기반으로 한 단문 유사성 검출기를 제안하여 훈련 데이터 세트 크기를 줄입니다.

- **Technical Details**: 이 연구는 글자 및 단어 n-grams을 특징으로 사용하여 SVM(Support Vector Machine) 분류기를 훈련시키는 두 단계 분류기 시스템을 개발했습니다. 또한, 스팸 감지에서 영감을 받아 Jaccard 거리를 기반으로 하는 단문 유사성 검출기를 통해 훈련 세트의 크기를 줄이는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 복잡성과 계산 시간을 고려할 때 대안적 접근 방법들과 비교하여 높은 정확도를 보였습니다. 또한, Google Play와 App Store에서 사용 가능한 개인 금융 앱 CoinScrap에 실제로 적용되었습니다. 이는 짧은 텍스트 분류 기술이 실제 금융 분야에서 유용하게 사용될 수 있음을 시사합니다.



### Embodied Agents for Efficient Exploration and Smart Scene Description (https://arxiv.org/abs/2301.07150)
Comments: Accepted by IEEE International Conference on Robotics and Automation (ICRA 2023)

- **What's New**: 이 연구는 사람이 붐비는 환경에서 로봇 플랫폼의 확산을 촉진하는 자연어로 의사소통할 수 있는 실체 에이전트(embodied agents)의 개발에 대한 관심이 증가함에 따라, 이를 한 단계 발전시키기 위해 미지의 실내 환경을 탐색하고 매핑하는 동시에 흥미로운 장면들을 자연어 설명으로 묘사하는 시각적 내비게이션 설정을 다루고 있다. 이를 통해 로봇의 환경 표현을 사용자가 이해하기 쉬운 방식으로 제공하며, 탐험 중 만난 주요 객체(object)들과 그들 사이의 상관관계를 강조한다.

- **Technical Details**: 이 연구는 최신의 시각적 로봇 탐험 기술과 이미지 캡셔닝(image captioning)을 결합하여 에이전트-환경 상호작용을 통해 생성된 이미지들에 대해 진행되었다. 제안된 접근 방식은 환경의 의미론적 지식을 극대화하고 반복을 피하는 똑똑한 장면 설명을 생성할 수 있다.

- **Performance Highlights**: 제안된 접근법의 성능을 정량적으로 평가하기 위해 탐험(exploitation) 및 설명(description) 기술을 모두 고려하는 특정 점수가 개발되었다. 사실적으로 시뮬레이션된 환경과 실제 세계에서 수행된 실험들은 제안된 접근법이 로봇의 탐험 중 시점을 효과적으로 설명할 수 있으며, 그 관찰의 인간 친화적 해석을 개선할 수 있음을 보여준다.



### Explore and Explain: Self-supervised Navigation and Recounting (https://arxiv.org/abs/2007.07268)
Comments: ICPR 2020

- **What's New**: 이 논문은 자율적이고 지능적인 에이전트를 개발하기 위한 목표로 주목받고 있는 실체화된 인공지능(Embodied AI)에 관한 새로운 설정을 제안합니다. 제안된 설정에서, 에이전트는 이전에 알려지지 않은 환경을 탐색하면서 그 경로에서 보게 되는 장면을 설명해야 합니다. 이러한 맥락에서, 탐구 목표에 의해 환경을 탐색하고, 설명을 위한 적절한 순간을 선택하며, 관련 객체 및 장면에 대한 자연스러운 언어 설명을 출력해야 합니다.

- **Technical Details**: 모델은 자기감독(self-supervised) 탐색 모듈과 함께 벌칙을 가진 새로운 탐색 방법과 완전히 주의를 기울이는 설명 모델(fully-attentive captioning model)을 통합합니다. 또한, 환경과 탐색에서 오는 정보를 바탕으로 적절한 설명 시기를 선택하는 다양한 정책을 조사합니다. 

- **Performance Highlights**: 실험은 사실적인 환경인 Matterport3D 데이터셋에서 수행되었으며, 에이전트의 탐색 및 설명 능력과 그 상호 작용의 역할을 조사합니다.




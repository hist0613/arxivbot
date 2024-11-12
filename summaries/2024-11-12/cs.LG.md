New uploads on arXiv(cs.CL)

### UTMath: Math Evaluation with Unit Test via Reasoning-to-Coding Thoughts (https://arxiv.org/abs/2411.07240)
- **What's New**: 이 논문은 기존의 수학 문제 해결 기준의 한계를 극복하기 위한 새로운 UTMath 벤치마크를 소개합니다. 이 벤치마크는 9개의 수학 영역에서 1,053개의 문제를 가지고 있으며, 정확성과 신뢰성을 중시하는 혁신적인 평가 프레임워크를 제공합니다.

- **Technical Details**: UTMath 벤치마크는 각각의 문제에 대해 68개 이상의 테스트 케이스로 구성되며, LLM이 코드를 생성하기 전에 명시적 추론을 수행하도록 유도하는 Reasoning-to-Coding of Thoughts (RCoT) 접근 방식을 채택하고 있습니다. 이는 LLM이 더 향상된 솔루션을 생성하도록 합니다.

- **Performance Highlights**: 연구 결과, GPT-4o 모델은 UTMath 벤치마크에서 단 26.93%의 문제만을 해결하여 벤치마크의 어려움을 보여주었습니다. RCoT 접근을 통해 8개의 LLM 중 7개가 더 효율적인 솔루션을 생성했으며, 모델들의 reasoning 품질은 최종 솔루션의 정확성과 효율성에 중요한 영향을 미쳤습니다.



### OpenThaiGPT 1.5: A Thai-Centric Open Source Large Language Mod (https://arxiv.org/abs/2411.07238)
Comments:
          8 pages, 4 tables

- **What's New**: OpenThaiGPT 1.5는 Qwen v2.5를 기반으로 한 최신 태국어 챗 모델로, 2,000,000개 이상의 태국어 지침 쌍으로 파인튜닝되었습니다. 이 모델은 다중 턴 대화 지원, Retrieval Augmented Generation(RAG) 호환성, 툴 호출 기능 등 다양한 최신 기능을 포함하고 있습니다.

- **Technical Details**: OpenThaiGPT 1.5는 두 가지 크기(7B 및 72B 매개변수)로 제공되며, 다양한 계산 자원 제약과 성능 요구에 맞춰 조정되었습니다. 모델은 Huggingface에서 Qwen/Qwen2.5-7B-Instruct 및 Qwen/Qwen2.5-72B-Instruct로부터 파인튜닝되었으며, 파인튜닝 과정에서 LoRa 기술을 사용했습니다. 총 2,000,000개의 태국어 지침 쌍으로 광범위한 파인튜닝이 이루어졌습니다.

- **Performance Highlights**: OpenThaiGPT 1.5는 여러 태국어 작업에서 최고 수준의 성능을 발휘하였으며, 태국어 모델들 중 최고의 성능을 기록했습니다. 최근에 발표된 태국어 시험 벤치마크에서 72B 모델이 63.89% 및 70.39%의 점수를 기록하며 다른 오픈 소스 태국어 모델들을 능가했습니다.



### Contextualized Evaluations: Taking the Guesswork Out of Language Model Evaluations (https://arxiv.org/abs/2411.07237)
Comments:
          Code & data available at this https URL

- **What's New**: 이 논문은 언어 모델에 대한 쿼리(질문)가 불명확할 때에도 평가 절차가 제대로 이루어질 수 있도록 'contextualized evaluations'라는 새로운 평가 프로토콜을 제안합니다. 이 방법은 구체화되지 않은 쿼리에 대한 주변 맥락을 인위적으로 생성하여 평가 시간에 전달합니다.

- **Technical Details**: 제안된 프로토콜은 다양한 맥락(후속 질문-답변 쌍)을 언어 모델의 불명확한 쿼리에 통합하는 것을 목표로 합니다. 비맥락적(e.g., 모델 출력만 있는 경우) 및 맥락 인식 설정에서 모델 쌍을 비교하여 평가자 간의 선호도 및 판단 기준의 변화가 평가의 정확도와 신뢰성에 미치는 영향을 조사합니다.

- **Performance Highlights**: 맥락을 제공함으로써 평가자 간의 동의율이 3-10% 증가하고 모델 쌍 간의 승률이 변화하는 등 평가의 결과에 중대한 영향을 미칠 수 있음을 발견했습니다. 또한, 기본 모델 응답이 WEIRD(서구, 교육받은, 산업화된, 부유하고 민주적인) 맥락에치우쳐 있다는 편향을 발견하였습니다.



### TreeCoders: Trees of Transformers (https://arxiv.org/abs/2411.07218)
- **What's New**: 이번 논문에서는 TreeCoders라는 새로운 형태의 transformer tree를 소개합니다. 전통적인 linear transformer에서 k-ary 나무 구조로 이동하여 각 블록이 노드를 구성하며, 클래스 분류기가 최적의 자식을 선택하고 토큰을 특정 리프(leaf)로 라우팅합니다. 이러한 선택자는 transformer 블록 외부에 위치해 다양한 아키텍처를 사용할 수 있게 합니다.

- **Technical Details**: TreeCoder는 transformer 블록과 선택자로 구성된 k-ary 트리 구조입니다. 각 노드는 하나 이상의 디코더 또는 인코더 레이어로 구성되며, 뿌리(root) 노드는 기존 transformer와 같은 입력을 받고 같은 작업을 수행합니다. 이 구조는 sparse node activation을 지원하며, 로그 복잡도를 기반으로 한 트리 검색이 가능합니다.

- **Performance Highlights**: 우리가 제안한 tree transformer 모델은 76%의 경우, 동등한 크기의 linear transformer 모델보다 우수한 성능을 보였으며, 다양한 언어 데이터셋에 걸쳐 경쟁력 있는 결과를 달성했습니다. 또한, 제안된 모델은 배포 구현(distributed implementation)에 적합합니다.



### The Super Weight in Large Language Models (https://arxiv.org/abs/2411.07191)
- **What's New**: 최근 연구에서 크고 복잡한 대형 언어 모델(LLM)에서 소수의 파라미터가 모델 품질에 미치는 영향이 크다는 놀라운 결과가 발표되었습니다. 본 논문에서는 단 하나의 파라미터를 제외하는 것만으로 LLM의 텍스트 생성 능력이 파괴될 수 있음을 보여주었습니다.

- **Technical Details**: 논문에서는 'super weights'라는 개념을 제시하고, 이를 단 한 번의 forward pass를 통해 식별하는 데이터 프리(data-free) 방법을 소개합니다. 또한, super weights는 'super activations'를 유도하며, 이 두 가지 요소는 모델 품질에 필수적입니다. super weights와 super activations는 모두 구조적으로 중요하며, pruning(제거) 시 모델 성능이 급격히 감소합니다.

- **Performance Highlights**: 단 하나의 super weight를 제거하면 zero-shot accuracy(제로 샷 정확도)가 사실상 제로로 떨어지며, perplexity(당혹도)는 3배 이상 증가합니다. 본 논문에서 제시한 방법은 기존의 quantization(양자화) 기술과 비교했을 때 데이터 프리 방식으로 높은 성능을 보이며, 라운드 투 니어스(round-to-nearest) 양자화 방식의 품질을 크게 향상시킬 수 있음을 입증합니다.



### Counterfactual Generation from Language Models (https://arxiv.org/abs/2411.07180)
Comments:
          A preprint

- **What's New**: 언어 모델에서 인과 생성 메커니즘을 이해하고 조작하는 것은 모델의 행동을 제어하는 데 필수적입니다. 본 논문에서는 기존의 개입(intervention) 기술 외에도, 카운터팩추얼(counterfactual) 사고방식을 강조하여 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 언어 모델을 일반화된 구조 방정식(Generalized Structural-equation) 모델로 재구성하여 진짜 문자열 카운터팩추얼을 생성합니다. Gumbel-max 트릭(Gumbel-max trick)을 사용하여 샘플링 노이즈의 동일한 인스턴스에서 원래 문자열과 카운터팩추얼 간의 결합 분포(joint distribution)를 모델링합니다. 이 알고리즘은 후견 Gumbel 샘플링(hindsight Gumbel sampling)을 기반으로 하여 관찰된 문자열의 카운터팩추얼을 생성하기 위한 잠재적인 노이즈 변수를 추론합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 의미 있는 카운터팩추얼을 생성하는 동시에 일반적으로 사용되는 개입 기술이 상당한 원하지 않는 부작용을 가진다는 것을 보여줍니다.



### More Expressive Attention with Negative Weights (https://arxiv.org/abs/2411.07176)
- **What's New**: Cog Attention이라는 새로운 주의 메커니즘을 제안하며, 이는 음의 주의 가중치를 허용하여 표현력을 향상시킵니다. 이 메커니즘은 토큰 삭제 및 복사를 정적인 OV 매트릭스에서 동적인 QK 내적(product)으로 전환할 수 있도록 합니다.

- **Technical Details**: Cog Attention은 음의 주의 가중치를 사용할 수 있는 구조로, 토큰의 삭제, 복사 또는 보유를 각각 음의, 양의 또는 최소 주의 가중치로 할당하여 단일 주의 헤드가 더 유연하고 표현력이 높아지도록 합니다.

- **Performance Highlights**: Cog Attention을 사용한 모델들이 전통적인 softmax 주의 모듈을 사용한 모델들에 비해 우수한 성능을 보이며, 이는 언어 모델링 및 이미지 생성 등 다양한 작업에서 입증되었습니다.



### Continual Memorization of Factoids in Large Language Models (https://arxiv.org/abs/2411.07175)
- **What's New**: 이 논문에서는 LLMs (Large Language Models)가 긴 꼬리(long-tail) 및 전문화된 사실을 수집하는 데 겪는 문제에 대해 연구합니다. 특히, 잊어버림(forgetting) 문제를 해결하기 위해 REMIX(무작위 및 일반 데이터 혼합)라는 전략을 제안합니다.

- **Technical Details**: 연구에서는 LLMs의 지속적인 기억(continual memorization) 문제를 다룹니다. 모델이 초기 단계에서 소량의 사실 데이터에 대해 훈련한 후 다른 데이터셋으로 훈련할 때 그 사실을 잊지 않도록 해야 합니다. REMIX는 사전 훈련 데이터에서 샘플링한 일반 데이터를 무작위로 혼합하여 모델이 기억력을 유지하도록 돕습니다.

- **Performance Highlights**: 실험 결과, REMIX는 모델이 기억한 사실의 정확도를 13.5%에서 53.2%로 향상시켰으며, 일반적인 리플레이(replay) 방법보다 성능이 우수한 것으로 나타났습니다. 이는 여러 가지 사실 및 비사실 작업에 대해 일관되게 관찰되었습니다.



### A Primer on Word Embeddings: AI Techniques for Text Analysis in Social Work (https://arxiv.org/abs/2411.07156)
Comments:
          37 pages, 3 figures

- **What's New**: 이 논문은 사회복지 연구에서 텍스트 데이터를 분석하기 위한 혁신적인 기술인 word embeddings를 소개합니다. 기존의 키워드 기반 접근 방식보다 더 효과적으로 의미와 관계를 포착할 수 있는 수학적 표현을 설명하고 있습니다.

- **Technical Details**: 논문은 word embeddings의 기본 개념, 기술적 기초 및 실제 응용 분야를 논의합니다. 여기에는 semantic search, clustering, retrieval augmented generation과 같은 방법들이 포함되어 있으며, 이러한 기술들은 사회복지 실무에서 사례 노트를 분석하거나 사회복지 면허 시험을 언어 간에 비교하는 데에 사용될 수 있습니다.

- **Performance Highlights**: 이 연구의 사례는 word embeddings가 복잡한 텍스트 데이터를 분석하고, 효율적인 서비스 및 개입을 지원하는 데 어떻게 기여할 수 있는지를 보여줍니다. 그러나 정보 손실, 훈련 데이터 제약 및 잠재적 편견과 같은 한계도 간과하지 않았습니다. 성공적인 도입을 위해서는 도메인 특정 모델 개발, 접근 가능한 도구 생성 및 사회복지의 윤리적 원칙에 부합하는 최선의 관행 수립이 필요하다고 강조하고 있습니다.



### HierTOD: A Task-Oriented Dialogue System Driven by Hierarchical Goals (https://arxiv.org/abs/2411.07152)
- **What's New**: 이 논문에서는 복잡한 작업 환경에서 효과적인 대화형 시스템을 구현하기 위해 HierTOD라는 새로운 Task-Oriented Dialogue (TOD) 시스템을 소개합니다. 이 시스템은 계층적 목표에 기반하여 사용자와의 프로액티브한 상호작용을 지원하며, 종합적인 워크플로우를 지원합니다.

- **Technical Details**: HierTOD는 자연어 이해(NLU), 대화 관리(DM), 응답 생성(RG) 모듈로 구성된 전통적인 TOD 시스템의 파이프라인 접근 방식을 따릅니다. Composite Goal Retriever (CGR) 모듈이 목표 저장소를 구축하여 목표 달성을 위한 워크플로우를 정의하고 저장합니다. 시스템은 상태 머신 구조를 활용하여 대화 흐름을 관리하며, 다양한 데이터 서비스와 연결되어 사용자 지원을 최적화합니다.

- **Performance Highlights**: 인간을 대상으로 한 연구 결과, HierTOD는 슬롯 기반 정보 수집과 단계별 안내 두 가지 패러다임을 모두 지원하며, 사용자에게 더 나은 경험을 제공합니다. 사용자는 시스템을 통해 명확한 질문과 응답을 통한 효율적인 작업 수행이 가능해졌습니다.



### Greenback Bears and Fiscal Hawks: Finance is a Jungle and Text Embeddings Must Adap (https://arxiv.org/abs/2411.07142)
Comments:
          EMNLP 2024

- **What's New**: 이 논문은 1,430만 쌍의 쿼리-패시지(query-passage) 쌍을 기반으로 한 BAM embeddings라는 새로운 텍스트 임베딩 세트를 소개합니다. 이 임베딩은 금융 문서 검색에 최적화되어 있으며, 기존의 일반 목적 텍스트 임베딩 모델 보다 성능이 우수합니다.

- **Technical Details**: BAM embeddings는 Multilingual-E5 모델을 기반으로 하며, 금융 문서에서 정밀하게 필터링된 데이터셋으로 추가적으로 훈련되었습니다. 이 과정에서 하드 네거티브 마이닝(hard negative mining)이 성능 향상에 중요한 요소로 작용하며, 데이터셋의 규모도 중요한 역할을 합니다.

- **Performance Highlights**: BAM embeddings는 보유 세트에서 Recall@1 62.8%를 달성하였으며, 이는 OpenAI의 최상의 일반 목적 텍스트 임베딩(39.2%)보다 월등히 높습니다. 또한, FinanceBench에서 질문 응답 정확도가 8% 증가하여 회사 이름, 티커(ticker) 및 재무 지표를 더 잘 인식합니다.



### Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models (https://arxiv.org/abs/2411.07140)
- **What's New**: 이번 연구에서는 중국어 언어 모델의 사실성(factuality) 능력을 평가하기 위한 최초의 종합적인 벤치마크인 Chinese SimpleQA를 소개합니다. 이는 6개의 주요 주제와 99개의 다양한 하위 주제를 기반으로 3000개의 고품질 질문을 포함합니다.

- **Technical Details**: Chinese SimpleQA는 질문과 답변이 매우 짧으며, OpenAI API를 활용하여 평가하는 것이 용이합니다. 또한, 질문은 시간이 지나도 변하지 않는 정적(static) 특성을 가지고 있으며, 품질 보증 품질(control process)에 의해 높은 수준의 질문과 답변이 확보됩니다.

- **Performance Highlights**: Chinese SimpleQA에서 실시한 평가 결과, o1-preview와 Doubao-pro-32k 모델만이 합격 점수인 63.8%와 61.9%를 기록했습니다. 모델의 크기가 클수록 성능이 향상되고, RAG 전략을 도입했을 때 모델 간 성능 차이가 크게 줄어드는 경향을 보였습니다.



### Retrieval or Global Context Understanding? On Many-Shot In-Context Learning for Long-Context Evaluation (https://arxiv.org/abs/2411.07130)
- **What's New**: 이 논문에서는 기존의 길고 맥락이 있는 언어 모델(long-context language models, LCLM)의 평가 방법을 개선하기 위해, 새로운 많은 샷 인컨텍스트 학습(많은 샷 ICL) 벤치마크인 MANYICLBENCH를 제안합니다. 이 벤치마크는 LCLM의 검색 능력과 전반적인 맥락 이해 능력을 분리하여 평가합니다.

- **Technical Details**: 이 연구는 LCLM이 ICL 과제를 통해 요구되는 기술을 분석하며, 검색 능력과 전반적인 맥락 이해 간의 관계를 규명합니다. 분류(classification) 및 요약(summarization) 과제가 추가적인 시연(demonstration)에 의해 성능이 향상되는 반면, 번역(translation) 및 추론(reasoning) 과저는 명확한 경향을 보이지 않음을 발견하였습니다.

- **Performance Highlights**: 11개의 최신 LCLM 모델을 MANYICLBENCH로 평가한 결과, 최첨단 모델들이 64k 토큰까지의 검색 과제에서 잘 수행하였으나, 16k 토큰에서 전반적인 맥락 과제에서는 성능 저하가 두드러진다는 것을 발견하였습니다.



### Benchmarking LLMs' Judgments with No Gold Standard (https://arxiv.org/abs/2411.07127)
- **What's New**: GEM(Generative Estimator for Mutual Information)라는 새로운 평가 지표를 소개하며, 이는 기존의 gold standard reference 없이 Large Language Models(LLMs)의 언어 생성 성능을 평가할 수 있는 방법을 제공합니다. GEM은 LLM의 생성 성능을 기존의 기계 번역, 요약 같은 전통적인 과제에서 학술지 peer review와 같은 주관적 과제까지 확장할 수 있게 합니다.

- **Technical Details**: GEM은 후보 응답과 참조 응답 간의 상호 정보(mutual information)를 추정하는 생성 모델입니다. 이는 gold standard 품질이 필요 없으며, 후보 응답이 참조 응답에 대한 정보를 얼마나 제공하는지를 측정합니다. GEM의 변형으로 GEM-S가 있으며, 이는 특정 작업의 요약을 기반으로 상호 정보를 추정합니다. 실험 결과, GEM은 기존의 평가 지표와 비교하여 더 뛰어난 민감성과 조작 저항성을 보여줍니다.

- **Performance Highlights**: 실험 결과 GEM과 GEM-S는 인간 평가와 높은 상관관계를 보이며, 다른 기초 지표보다 우수한 성능을 드러냈습니다. 특히, 모든 의미적 열화에 대해 민감성을 나타내는 유일한 지표로, 무의미한 응답 연장 후에도 점수가 크게 증가하지 않았습니다. 추가로, GRE-bench를 통해 다양한 LLM의 peer review 능력을 평가하였고, 파라미터 크기와 GRE-bench 점수 간의 강한 상관관계를 발견했습니다.



### SCAR: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs (https://arxiv.org/abs/2411.07122)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 생성 결과가 사용자와 일치하지 않거나 유해한 콘텐츠를 생성할 수 있다는 문제를 다룹니다. 새로운 접근 방식으로 Sparse Conditioned Autoencoder (SCAR)를 제안하여 이러한 유해한 개념을 탐지하고 조절할 수 있는 방법을 제공합니다.

- **Technical Details**: SCAR는 LLM의 작동 방식을 혼란시키지 않고 원하지 않는 콘텐츠(예: 독성 콘텐츠)의 생성을 회피하는 데 필요한 제어 능력을 제공합니다. SCAR는 잠재적 특성을 격리하는 메커니즘을 사용하여 개념 검출 및 조작을 가능하게 합니다. 이 시스템은 SAE(Sparse Autoencoder) 구조를 기반으로 하며, 훈련 중에 모든 변압기 가중치를 고정하고 입력 활성화를 재구성하도록 훈련됩니다. 또한, 조건부 손실 함수를 도입하여 원하는 개념의 표현을 보장합니다.

- **Performance Highlights**: SCAR는 여러 개념(독성, 안전성, 글쓰기 스타일 정렬 등)에 대한 실험을 통해 효과적인 검출 및 조절 기능을 입증했으며, 전체 모델 성능에 미치는 영향을 최소화하면서도 독성 콘텐츠의 생성을 효과적으로 선도할 수 있는 능력을 보여주었습니다.



### Building a Taiwanese Mandarin Spoken Language Model: A First Attemp (https://arxiv.org/abs/2411.07111)
Comments:
          Work in progress

- **What's New**: 이번 기술 보고서는 대만 만다린어( Taiwanese Mandarin )를 위한 구어 대형 언어 모델( spoken large language model, LLM ) 구축의 초기 시도를 소개합니다. 이 모델은 다중 턴 대화를 위한 실시간 음성-음성 상호작용을 가능하게 하기 위해 설계되었습니다.

- **Technical Details**: 모델은 디코더 전용(transformer architecture) 아키텍처를 사용하며, 세련된 상호작용을 유지하면서도 전이 가능한 전이-말-음성(interactions) 능력을 갖추고 있습니다. 데이터 준비는 합성 대화(dialogues)로 진행되어 LLM을 기반으로 한 대화 생성 및 TTS( text-to-speech ) 모델을 통한 음성화 과정을 포함합니다.

- **Performance Highlights**: 모델은 ASR( automatic speech recognition ), LLM, TTS를 포함한 기존의 계단식 모델들과의 차별성으로, 실시간 전이-말-음성 커뮤니케이션을 구현하는 잠재력을 보여줍니다. 특히, 한국어를 포함한 여러 언어를 지원하며, 실시간 상호작용을 통한 대화의 자연스러움을 추구하고 있습니다.



### Training Neural Networks as Recognizers of Formal Languages (https://arxiv.org/abs/2411.07107)
Comments:
          40 pages, 2 figures. Preprint

- **What's New**: 본 논문은 신경망 아키텍처의 계산적 능력을 형식 언어 이론의 관점에서 정량적으로 검증하려는 새로운 접근 방식을 제안합니다. 일반적인 실험 방법 대신 문자열 이진 분류기로 신경망을 훈련시키고 평가하는 방식을 적용했습니다. 또한, Snæbjarnarson et al. (2024)의 알고리즘을 확장하여 정규 언어의 문자열 길이 제어 샘플링을 수행하여 더 나은 시간 복잡도를 확보했습니다.

- **Technical Details**: 이 연구는 Chomsky 계층의 여러 언어에 대해 세 가지 신경망 아키텍처(RNN, LSTM, causally-masked transformer)의 성능을 비교 분석합니다. 신경망을 형식 언어의 인식기로 훈련시키기 위해, 긍정 샘플링 및 회원 테스트를 위한 언어 특정 알고리즘을 사용하는 방법을 제안합니다. 길이 제어 샘플링을 위한 비판적인 알고리즘을 구현하여, 필수적인 부정적인 샘플을 생성하고, 효율적인 훈련 방식을 확인했습니다.

- **Performance Highlights**: 실험 결과, RNN과 LSTM이 transformer보다 자주 더 나은 성능을 보였으며, 보조 훈련 목표는 특정 아키텍처에서 도움이 되기도 했지만, 전반적으로 일관된 향상 효과는 없었습니다. 연구 결과는 FLaRe(Formal Language Recognition)라는 벤치마크로 제공되며, 향후 언어 인식 주장을 이론적으로 뒷받침하는데 유용할 것입니다.



### Transformer verbatim in-context retrieval across time and sca (https://arxiv.org/abs/2411.07075)
Comments:
          accepted to Conference on Natural Language Learning 2024 (this https URL)

- **What's New**: 이 연구에서는 언어 모델이 훈련 과정에서 시간에 따라 어떻게 의미 있는 정보를 검색할 수 있는지가 분석되었습니다. 특히 구체적 vs 추상적 명사의 검색 능력이 훈련 중 어떻게 발전하는지를 관찰하였습니다.

- **Technical Details**: Pythia 모델 세트를 사용하여 훈련 중에 모델 크기별로 정보 검색의 발달을 평가했습니다. 모든 모델에서 약 1%의 훈련 토큰이 지나간 후에 구체적인 명사를 검색할 때의 이점이 나타났고, 이 이점은 훈련이 끝나갈 무렵 사라지는 경향이 있었습니다.

- **Performance Highlights**: 모델들이 구체적인 명사를 검색할 때 (과거 상황에 대한 회상) 성능이 향상되었고, 구체적 명사에 대한 검색 정확도가 높았습니다. 제로샷 성능과는 양의 상관관계가 있었으나, 훈련 말기에 들어서는 해당 이점이 감소하였습니다.



### On Active Privacy Auditing in Supervised Fine-tuning for White-Box Language Models (https://arxiv.org/abs/2411.07070)
- **What's New**: 본 논문에서는 언어 모델(LMs)의 미세 조정(supervised fine-tuning, SFT) 과정에서 개인정보 유출 위험을 식별하고 정량화하기 위한 새로운 능동적 개인정보 감사 프레임워크인 Parsing을 소개합니다. 이 프레임워크는 흰 상자(white-box) 멤버십 추론 공격(membership inference attacks, MIAs)을 핵심 기술로 활용하여 개인정보 보호를 극대화합니다.

- **Technical Details**: Parsing 프레임워크는 두 단계로 이루어진 전략을 적용하여 LMs의 미세 조정 과정에서 개인정보 노출을 모니터링합니다. 이 프레임워크는 GPT-2, Llama2를 포함한 대형 LMs에 대한 MIAs의 효과성을 개선하였으며, 새로운 학습 목표를 도입하여 샘플의 멤버십 표현을 최적화합니다.

- **Performance Highlights**: 실험 결과, Parsing 프레임워크는 다양한 모델과 작업에서 개인정보 위험을 감지하고 정량화하는 데 효과적임을 입증하였습니다. 이 연구는 미세 조정 과정에서의 개인정보 보호 전략을 제안하고 있으며, 연구 커뮤니티에 유용한 도구를 제공합니다.



### LIFBench: Evaluating the Instruction Following Performance and Stability of Large Language Models in Long-Context Scenarios (https://arxiv.org/abs/2411.07037)
Comments:
          17 pages, 3 figures

- **What's New**: 본 논문에서는 'Long-context Instruction-Following Benchmark (LIFBench)'라는 새로운 벤치마크를 소개하여 장기 문맥(Long-context) 시나리오에서의 LLM(대형 언어 모델)의 지시 사항 수행 능력과 안정성을 평가할 수 있는 데이터셋을 제공합니다.

- **Technical Details**: LIFBench는 세 가지 장기 문맥 시나리오와 11개의 다양한 태스크로 구성되어 있으며, 2,766개의 지시 사항이 자동 확장 방법을 통해 생성되었습니다. 평가를 위한 새로운 평가 체계인 'LIFEval'을 제안하여, LLM의 복잡한 응답을 정확하고 자동화된 방식으로 점수화할 수 있습니다.

- **Performance Highlights**: 20개의 유명한 LLM을 대상으로 6개의 길이 간격에서 광범위한 실험을 수행하였고, 이들의 지시 사항 수행 능력과 안정성을 분석하였습니다. LIFBench와 LIFEval은 LLM의 복잡한 장기 문맥 환경에서의 성능을 평가하는 강력한 도구로 기여하게 됩니다.



### UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction (https://arxiv.org/abs/2411.07019)
- **What's New**: 본 연구에서는 통합 지식 그래프(link prediction) 링크 예측을 위한 통합 계층적 표현 학습 프레임워크(UniHR)를 제안합니다. 이 프레임워크는 하이퍼 관계(hyper-relational), 시계열(temporal), 중첩(nested) 사실을 포함한 다양한 사실 표현을 통일된 방식으로 처리할 수 있습니다.

- **Technical Details**: UniHR 프레임워크는 하이퍼 관계 데이터 표현 모듈(HiDR)과 계층 구조 학습 모듈(HiSL)로 구성됩니다. HiDR는 다양한 형태의 사실을 삼중 표현(triple-based representation)으로 통합하며, HiSL는 개별 사실의 의미 정보를 증강시키고 사실 간의 구조적 정보를 풍부하게 합니다. 이를 통해 링크 예측을 위한 향상된 표현을 생성합니다.

- **Performance Highlights**: 7개의 데이터 세트에 대한 실험 결과 UniHR은 특정 유형의 KG를 위해 설계된 기본 모델들보다 뛰어난 성능을 보였으며, 이는 HiDR의 강력한 일반화 능력과 HiSL 모듈의 효과성을 입증합니다.



### Token2Wav (https://arxiv.org/abs/2411.06989)
- **What's New**: 이번 논문에서는 Wave Network에서 유래된 새로운 토큰 표현 방법인 Token2Wave에 대한 심층 분석을 제공하며, 이를 통해 입력 텍스트의 전역 및 지역 의미를 파악할 수 있도록 설계되었습니다.

- **Technical Details**: Token2Wave에서는 각 토큰이 전역 의미를 포착하는 magnitude component와 개별 토큰의 관계를 인코딩하는 phase component로 구성된 복소 벡터로 표현됩니다. 이 연구는 Token2Wave 프레임워크 내에서의 수렴 거동, 역전파 특성 및 임베딩 독립성에 대해 조사하였습니다.

- **Performance Highlights**: Token2Wave는 BERT에 비해 비디오 메모리 사용량과 학습 시간을 현저히 줄일 수 있으며, [CLS] 토큰, 전체 입력 텍스트, 분류기 매개변수에 대한 기울기 비교를 통해 Token2Wave의 독특한 특성을 강조합니다.



### Sniff AI: Is My 'Spicy' Your 'Spicy'? Exploring LLM's Perceptual Alignment with Human Smell Experiences (https://arxiv.org/abs/2411.06950)
- **What's New**: 이 연구는 인간의 후각 경험과 인공지능(AI) 간의 인지적 정렬을 탐구하며, AI가 인간의 후각 설명을 어떻게 해석하는지를 조사합니다. 40명의 참가자를 대상으로 한 사용자 연구를 통해 AI 시스템이 향기를 추정하는 능력을 평가하였습니다.

- **Technical Details**: 대규모 언어 모델(LLM)을 바탕으로 한 AI 시스템은 참가자들의 냄새 설명을 바탕으로 향기를 유추하는 상호작용적 작업을 수행했습니다. 연구에서는 AI의 내부 상태에서 향기 관계의 맥락적 이해와 표현을 평가하기 위해 고차원 임베딩 공간(high-dimensional embedding space)을 분석했습니다. AI의 성능은 정량적 및 정성적 방법론을 통해 평가되었습니다.

- **Performance Highlights**: 연구 결과 AI와 인간 간의 인지적 정렬은 제한적이며, 특정 향기(예: 레몬, 페퍼민트)에 대한 편향이 발견되었습니다. AI는 로즈마리(로즈마리)가 아닌 유칼립투스로 잘못 인지하기도 했으나, 어떤 경우에는 "강렬한 남성 향"을 오크모스로 올바르게 인식하는 등 흥미로운 행동을 보였습니다. 이러한 발견은 HCI 시스템에 다감각적 경험 통합을 통한 인간-AI 정렬 향상의 기회를 강조합니다.



### Cancer-Answer: Empowering Cancer Care with Advanced Large Language Models (https://arxiv.org/abs/2411.06946)
Comments:
          Accepted at FIRE 2024 (Track: Conversational System for Differential Diagnosis of GI Cancer)

- **What's New**: 이 연구는 대량의 언어 모델(LLMs)인 GPT-3.5 Turbo를 활용하여 위장관(Gastrointestinal) 암 관련 질문에 대한 정확하고 맥락에 적합한 응답을 생성하는 방법을 탐구합니다. 이는 조기 진단을 위한 중요한 도구로 사용될 수 있습니다.

- **Technical Details**: 연구에서는 위장관 암 관련 30개의 질문을 훈련 세트로, 50개의 질문을 테스트 세트로 사용합니다. A1과 A2라는 두 가지 성능 지표를 통해 모델 생성 답변의 정확성 및 언어적 의미를 평가하며, 각각 최대 0.546과 0.881의 값을 달성하였습니다. 프롬프트를 사용하여 특정 입력 인스트럭션을 제공하여 모델이 관련 있고 일관된 응답을 생성하도록 유도합니다.

- **Performance Highlights**: 이번 연구는 GI 암 진단을 위해 필요한 신속하고 정확한 응답을 제공하도록 LLMs의 잠재력을 강조하였으며, 이를 통해 환자 치료 결과를 향상시킬 수 있는 가능성을 제시합니다.



### LongSafetyBench: Long-Context LLMs Struggle with Safety Issues (https://arxiv.org/abs/2411.06899)
- **What's New**: 이 논문에서는 장기 맥락 언어 모델(long-context language models)의 안전성을 평가하기 위한 최초의 기준인 LongSafetyBench를 소개합니다. 이 기준은 10개의 작업 범주로 구성되어 있으며, 평균 41,889 단어의 길이를 가집니다. 최근까지의 연구들은 주로 모델의 기능에 초점을 맞췄지만, 이 논문은 안전성에 대한 객관적이고 포괄적인 평가를 다룹니다.

- **Technical Details**: LongSafetyBench는 모델의 안전성 문제를 해결하기 위해 불법 활동(Illegal Activities), 잘못된 정보 피해(Misinformation Harm), 공격성 및 편향(Offensiveness and Bias)과 같은 세 가지 유형의 위험한 시나리오를 대상으로 데이터를 수집하고 구성하였습니다. 각 문항은 다중 선택 질문 형식으로 포맷팅되었으며, 총 1,203개의 테스트 인스턴스가 포함되어 있습니다.

- **Performance Highlights**: 8개의 장기 맥락 LLM을 LongSafetyBench에서 테스트한 결과, 대부분의 주류 장기 맥락 LLM에서 안전한 응답의 비율이 50% 미만으로 나타났습니다. 장기 맥락 시나리오에서의 안전성과 단기 맥락에서의 성능 간의 일치성이 떨어지며, 모델이 긴 텍스트 내의 해로운 콘텐츠를 간과하는 경향이 있음을 확인했습니다. 또한, 적은 양의 데이터로 훈련한 오픈 소스 모델이 최상위 폐쇄형 모델과 유사한 성능을 달성할 수 있는 것으로 나타났습니다.



### A Unified Multi-Task Learning Architecture for Hate Detection Leveraging User-Based Information (https://arxiv.org/abs/2411.06855)
Comments:
          7 pages, 1 figure, and two tables

- **What's New**: 이 논문은 기존의 증오 발언(hate speech) 탐지 방법과 차별화된 모델을 제시하여 사용자가 생성한 컨텐츠 간의 정보를 활용해 증오 발언 확인을 개선하는 새로운 접근법을 소개하고 있습니다.

- **Technical Details**: 연구에서는 단일 작업 학습(single-task learning, STL)과 다중 작업 학습(multi-task learning, MTL) 패러다임에서 Convolutional Neural Networks (CNN), Gated Recurrent Unit (GRU), Bidirectional Encoder Representations from Transformers (BERT), A Lite BERT (ALBERT)와 같은 딥 뉴럴 네트워크를 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: 세 가지 기준 데이터셋을 기반으로 특정 사용자 특성과 텍스트 특성을 결합함으로써 macro-F1 및 weighted-F1 점수가 유의미하게 향상된 결과를 도출하였습니다.



### Evaluating Large Language Models on Financial Report Summarization: An Empirical Study (https://arxiv.org/abs/2411.06852)
- **What's New**: 이 논문은 GLM-4, Mistral-NeMo, LLaMA3.1의 세 개의 최신 LLM을 비교 분석하여 자동화된 금융 보고서 생성에서의 효과성을 평가합니다. 현 금융 환경에서의 LLM의 신뢰성, 정확성 및 규정 준수를 보장하기 위한 엄격한 검토의 필요성을 강조합니다.

- **Technical Details**: 이 연구에서는 ROUGE-1, BERT Score, LLM Score와 같은 제안된 지표를 포함하여 금융 보고서 분석을 위한 벤치마크를 제공합니다. 정량적 메트릭(precision, recall 등)과 정성적 분석(맥락 적합성, 일관성 등)을 통합한 혁신적인 평가 프레임워크를 도입하여 각 모델의 출력 품질을 종합적으로 평가합니다.

- **Performance Highlights**: 이 논문은 금융 보고서에서의 LLM 성능에 대한 구체적인 벤치마크를 확립하며, LLM이 금융 분야에 적합하게 조정되어야 하는 도전 과제를 강조합니다. 공개된 데이터셋을 통해 연구자들이 이 연구 결과를 검토하고 개선할 수 있는 협력 환경을 조성합니다.



### 1-800-SHARED-TASKS @ NLU of Devanagari Script Languages: Detection of Language, Hate Speech, and Targets using LLMs (https://arxiv.org/abs/2411.06850)
Comments:
          13 pages, Submitted to CHIPSAL workshop @ COLING 2025

- **What's New**: 이번 연구에서는 CHiPSAL 2025의 언어 감지, 증오 발언 식별 및 목표 감지에 관한 시스템을 설명합니다. Devanagari 스크립트 언어에서의 자연어 이해(NLP) 문제를 해결하기 위해 MuRIL, IndicBERT, Gemma-2와 같은 대형 언어 모델을 활용했습니다.

- **Technical Details**: 각 서브태스크에 대해 다양한 멀티링구얼 모델을 파인 튜닝(fine-tune)하고 평가 단계에서 최상의 모델을 선택했습니다. 서브태스크 A는 5개 언어 중에서 언어를 식별하며, B는 텍스트에서 증오 발언을 감지하고, C는 증오 발언의 목표를 분류합니다. Focal loss를 사용하여 클래스 불균형 문제를 해결했습니다.

- **Performance Highlights**: 서브태스크 A에서 F1 점수 0.9980, B에서 0.7652, C에서 0.6804를 달성하여 강력한 성능을 나타냈습니다. 특히, Ensemble 기법을 통해 최종 테스트에서 서브태스크 A의 성능을 더욱 개선했습니다.



### LLM-Neo: Parameter Efficient Knowledge Distillation for Large Language Models (https://arxiv.org/abs/2411.06839)
Comments:
          ICASSP 25' under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 소형 학생 모델로 효율적으로 지식을 전이하는 새로운 LLM-Neo 프레임워크를 제안합니다. 기존의 지식 증류(KD)와 저차원 적응(LoRA) 개념을 통합하여 지식 전이의 효율성을 개선합니다.

- **Technical Details**: LLM-Neo는 LoRA의 저차원 브랜치를 활용하여 효율적인 지식 증류를 달성합니다. KD 손실은 실제 레이블과 교차 엔트로피 손실, 그리고 교사 모델의 예측과의 Kullback-Leibler(KL) 발산을 결합하여 정의됩니다. 이 과정에서 LoRA의 파라미터 효율성을 유지할 수 있습니다.

- **Performance Highlights**: LLM-Neo는 Llama 2 및 Llama 3.1 모델을 압축하는 실험에서 다양한 기준 모델보다 우수한 성능을 보였습니다. 추가 분석을 통해 LoRA 변종에 대한 LLM-Neo의 강건성도 확인되었습니다.



### Persuasion with Large Language Models: a Survey (https://arxiv.org/abs/2411.06837)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 급증은 개인화되고 상호작용적인 콘텐츠 생성이 가능하게 되어 설득적 커뮤니케이션의 새로운 가능성을 열어주고 있습니다. 이 논문은 LLM 기반 설득 연구 분야의 발전을 조사합니다.

- **Technical Details**: LLM 시스템이 개인의 태도와 행동에 영향을 미치는 다양한 방식을 탐구하며, 정치, 마케팅, 공공 건강, 전자 상거래 및 자선 기부 등 여러 분야에서 이러한 시스템들이 인간 수준 또는 그 이상의 설득력을 달성한 사례를 들어 설명합니다. 이 연구는 또한 개인화 방식, AI 생성 콘텐츠 라벨링 여부 등 효과성에 영향을 미치는 주요 요소를 식별하고 있습니다.

- **Performance Highlights**: LLM 시스템은 개인의 선호도와 심리적 프로파일에 맞춰 메시지를 맞춤 설계하여 목표 그룹 또는 특정 개인과 공감할 수 있는 콘텐츠를 생성하는 데 탁월합니다. 이 시스템들은 상호작용적인 대화를 가능하게 하여 전통적인 일방향 메시징보다 더 효과적인 참여 경험을 제공합니다. 연구 결과에 따르면, LLM 시스템이 설득적 상호작용을 효과적으로 수행할 수 있는 가능성은 매우 크며, 이로 인해 발생하는윤리적 및 사회적 위험을 경고하고 있습니다.



### HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization on Model Alignmen (https://arxiv.org/abs/2411.06835)
Comments:
          NeurIPS 2024 Workshop on Safe Generative Artificial Intelligence (SafeGenAI)

- **What's New**: 이 연구는 최신 LLM(대형 언어 모델)의 사이버 보안 및 안전성 측면에서 중요한 격차를 메우기 위해 HarmLevelBench라는 새로운 데이터셋을 제안합니다. 이 데이터셋은 7개의 해로운 주제를 포함하고 있으며, 각 주제는 8개 다른 해로운 수준으로 세분화되어 모델 출력을 평가합니다.

- **Technical Details**: HarmLevelBench 데이터셋은 해로운 주제에 대한 일관된 질문 템플릿을 사용하여 구성되었습니다. 각 주제의 질문은 "How to...?" 형식을 따릅니다. 또한, 기존의 상태 최적화 기법(예: AWQ, GPTQ)과의 상관관계를 조사하여, 양자화(quantization) 기법이 모델의 정렬(alignment)과 강건성(robustness)에 미치는 영향을 연구합니다.

- **Performance Highlights**: 본 연구에서는 고급 jailbreaking 공격 기술을 활용하여 모델의 취약성을 평가했습니다. HarmLevelBench 데이터셋을 통해 7가지 jailbreaking 방법을 비교하고, 공격 성공률(ASR)을 통해 각각의 방법의 효과성을 측정했습니다. 실험 결과는 양자화의 효과와 함께 jailbreaking 기술의 면밀한 분석을 제공하였습니다.



### AssistRAG: Boosting the Potential of Large Language Models with an Intelligent Information Assistan (https://arxiv.org/abs/2411.06805)
Comments:
          Accepted by NeurIPS 2024 (poster)

- **What's New**: 이번 논문에서는 기존의 RAG 방법론의 한계를 극복하기 위해 AssistRAG라는 새로운 접근 방식을 제안합니다. 이 방법론은 LLM 내부에 지능형 정보 어시스턴트를 통합하여 정보 검색 및 의사결정 능력을 향상시킵니다.

- **Technical Details**: AssistRAG는 두 가지 주요 기능인 메모리 관리와 지식 관리로 구성됩니다. 메모리 관리는 내부 메모리의 내용을 통합 및 분석하며, 지식 관리는 외부 지식을 활용하는 데 초점을 맞춥니다. 이를 위해 네 가지 핵심 기능인 Tool usage, Action execution, Memory building, Plan specification을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, AssistRAG는 기존의 벤치마크를 초월하는 성능을 보여주었으며, 특히 덜 발전된 LLM에 더 큰 이점을 제공합니다. 다양한 복잡한 질문 응답 데이터셋에서 우수한 추론 능력을 나타냈습니다.



### PDC & DM-SFT: A Road for LLM SQL Bug-Fix Enhancing (https://arxiv.org/abs/2411.06767)
Comments:
          COLING-Industry 2025 accepted

- **What's New**: 이번 연구에서는 SQL의 버그 수정을 위한 새로운 방법론을 제시합니다. 기존의 Code LLM과 달리, 이 모델은 버그 수리 능력을 향상시키기 위해 Progressive Dataset Construction (PDC)와 Dynamic Mask Supervised Fine-tuning (DM-SFT) 방법을 도입합니다.

- **Technical Details**: PDC는 폭넓고 깊이 있는 두 가지 데이터 확장 방법을 포함하며, DM-SFT는 효과적인 수퍼바이즈드( Supervised) 학습 접근법을 통해 SQL 코드 버그 수정 과정에서의 학습 단계를 줄이고 안정성을 강화합니다. 특히, DM-SFT는 SQL 코드를 다루는 훈련 난이도를 줄이는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, PDC와 DM-SFT를 통해 훈련된 모델은 현재까지의 최고 모델들과 비교했을 때 성능이 50% 이상 향상되었으며, 일반적인 생성적 SFT에 비해 약 10%의 추가적인 성능 향상을 보여주었습니다.



### Reverse Prompt Engineering (https://arxiv.org/abs/2411.06729)
- **What's New**: 본 논문은 새로운 블랙박스(black-box), 제로샷(zero-shot) 언어 모델 역전 문제에 대한 연구를 다루며, 언어 모델의 텍스트 출력만을 활용한 프롬프트 복원(prompts reconstruction) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 대형 언어 모델(LLM)과 최적화 알고리즘을 활용하여 최소한의 자원으로 프롬프트를 효과적으로 복원합니다. RPE(Reverse Prompt Engineering) 기법은 LLM의 출력만을 이용하여 기저 프롬프트를 유추하며, 유전자 알고리즘(genetic algorithm)을 기반으로 한 반복 최적화 알고리즘을 중요하게 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 최첨단 방법들보다 우수한 성능을 보여주며, 코사인 유사도(cosine similarity)에서 평균 5.2% 개선된 결과를 나타낸 것으로 보고되었습니다.



### What Should Baby Models Read? Exploring Sample-Efficient Data Composition on Model Performanc (https://arxiv.org/abs/2411.06672)
Comments:
          8 pages, 6 figures, CoNLL 2024 (Shared Task) Accepted Paper

- **What's New**: 본 논문은 작은 언어 모델의 성능에 대한 선행 훈련 데이터 구성의 영향을 탐구합니다. 1000만 단어로 제한된 데이터셋을 사용하여 다양한 데이터셋 소스의 성능을 평가했습니다.

- **Technical Details**: 실험에 사용된 데이터셋은 아동 언어 지향 대화(CHILDES), 고전 문학(구텐베르크), 합성 데이터(타이니 스토리) 및 이들의 혼합(Mix)으로 구성되었습니다. 모델 범위는 18만 개에서 705만 개의 파라미터로 다양합니다.

- **Performance Highlights**: 작은 모델(GPT2-97M, GPT2-705M, Llama-360M)은 구텐베르크와 같은 복잡하고 풍부한 데이터셋에서 더 나은 성능을 보였으며, CHILDES와 타이니 스토리 데이터셋으로 훈련된 모델은 모든 모델 크기에서 저조한 성과를 보였습니다.



### Bridge: A Unified Framework to Knowledge Graph Completion via Language Models and Knowledge Representation (https://arxiv.org/abs/2411.06660)
- **What's New**: 이번 연구에서는 Knowledge Graph Completion (KGC) 분야의 한계를 극복하기 위해 Bridge라는 새로운 프레임워크를 제안하였습니다. 이 프레임워크는 구조적 정보와 의미적 정보를 동시에 인코딩하여 KGC 성능을 향상시킵니다.

- **Technical Details**: Bridge는 PLM (Pre-trained Language Model)을 활용하여 KGs (Knowledge Graphs)의 구조적(spatial) 및 의미적(semantic) 정보를 별도로 인코딩합니다. 이를 통해 PLM의 의미적 지식을 더 잘 활용하고 구조적 표현 학습(structured representation learning) 원리를 적용할 수 있습니다. 또한, BYOL (Bootstrap Your Own Latent) 방법을 사용하여 트리플의 두 가지 다른 뷰를 통해 PLM을 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: Bridge는 세 가지 벤치마크 데이터 세트에서 SOTA (State Of The Art) 모델을 능가하는 결과를 보였습니다. 실험 결과, Bridge 프레임워크가 기존의 다른 방법들에 비해 일관되게 우수한 성능을 나타냈습니다.



### Explore the Reasoning Capability of LLMs in the Chess Testbed (https://arxiv.org/abs/2411.06655)
Comments:
          submitted to NAACL2025

- **What's New**: 이 연구에서는 체스의 전략적 및 전술적 요소를 통합하여 언어 모델의 추론 능력을 향상시키는 새로운 접근 방식을 제안합니다. MATE라는 100만 개의 체스 위치와 전문가의 주석을 포함하는 데이터셋을 수집했습니다.

- **Technical Details**: 체스 데이터셋 MATE는 각 위치에 대해 장기 전략 및 단기 전술이 주석이 달린 후보 수를 제공합니다. 연구진은 LLaMA-3-8B 모델을 파인튜닝하여 최신 상용 언어 모델들과 비교하였습니다.

- **Performance Highlights**: 모델은 전략과 전술을 모두 제공할 때 최신 상용 언어 모델들을 24.2% 초과하는 성능을 기록했습니다. 언어 설명이 언어 모델의 추론 능력을 향상시키는 데 도움이 된다고 발견했습니다.



### The KIPARLA Forest treebank of spoken Italian: an overview of initial design choices (https://arxiv.org/abs/2411.06554)
- **What's New**: 본 논문은 이탈리아어 KIParla 코퍼스를 위한 treebank 구축을 위한 초기 설계 선택에 대한 개요를 제공합니다.

- **Technical Details**: KIParla 코퍼스는 228시간 정도의 음성 기록과 약 1,990,311개의 전사된 토큰으로 구성됩니다. 저자들은 기존 음성 언어의 UD(treebank) 포맷과의 정렬을 시도하고 있으며, 현재는 학생 인턴들이 elan 소프트웨어를 사용하여 수동으로 대화를 전사하고 있습니다. 각 전사 단위는 기본적인 억양 단위로 세분화되어 , 이를 통해 morphosyntactic annotation을 구축할 계획입니다.

- **Performance Highlights**: 현재 KIParla 프로젝트는 다수의 말뭉치와 음성 데이터가 결합된 독특한 자원으로, 기존 이탈리아어 UD treebank에서 다루어지지 않는 구어체 변형을 다루고 있습니다. 주요 목표는 다양한 언어적 변이를 반영하고, 특정 발화 상황에서의 구문 연결성을 보다 잘 드러내기 위한 전사 단위 경계 정의입니다.



### CineXDrama: Relevance Detection and Sentiment Analysis of Bangla YouTube Comments on Movie-Drama using Transformers: Insights from Interpretability Too (https://arxiv.org/abs/2411.06548)
- **What's New**: 이 연구에서는 방글라 영화 및 드라마에 대한 YouTube 댓글의 관련성(relevance) 및 감정(sentiment) 분석을 위한 새로운 데이터셋인 'CineXDrama'를 개발했습니다. 이 데이터셋은 14,000개의 수동으로 수집된 댓글로 구성되어 있으며, 각 댓글은 관련성(관련 또는 비관련) 및 감정(긍정 또는 부정)으로 주석이 달려 있습니다.

- **Technical Details**: 여덟 개의 트랜스포머 모델(Transformer models) 중 BanglaBERT가 관련성 탐지와 감정 분석 모두에서 가장 높은 정확도(관련성 탐지 83.99%, 감정 분석 93.3%)를 기록했습니다. 또한 LIME(로컬 해석 가능 모델 단순 선언 Explanations) 기법을 사용하여 모델의 판별 과정을 해석할 수 있게 하여, 분석의 투명성을 높였습니다.

- **Performance Highlights**: 제안된 시스템은 댓글의 관련성을 먼저 평가하고, 관련하다고 판단된 댓글에 대해서는 감정을 분석합니다. 이 연구를 통해 방글라 댓글 처리의 정확성과 효율성을 두 배로 증가시키고, 방글라 영화 및 드라마 업계에 실시간 피드백을 제공하는 시스템을 정착시킬 가능성을 제시했습니다.



### Epistemic Integrity in Large Language Models (https://arxiv.org/abs/2411.06528)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)이 생성하는 응답의 신뢰성 문제를 다루며, 언어적 단정성(linguistic assertiveness)와 내부 확신(internal certainty) 간의 불일치를 조명합니다. 새로운 인간 라벨 데이터셋과 정확도를 50% 이상 개선하는 방법을 도입하였습니다.

- **Technical Details**: 제안된 새로운 방법론은 LLM의 언어적 단정성을 측정하여 내부 확신과 외부 표현 간의 불일치를 확인합니다. 이는 여러 데이터셋에서 검증되었으며, LLM이 주장하는 확신과 실제 정확도 간의 불일치가 심각함을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM은 낮은 내부 확신을 가진 경우에도 과도하게 단정적인 응답을 생성하였으며, 이는 사용자에게 잘못된 신뢰를 줄 수 있습니다. 이 연구는 LLM의 불일치를 측정하는 가장 포괄적인 증거를 제공하며, 인간의 인식과 LLM의 단정성 점수를 비교하여 그 상관관계를 분석합니다.



### CULL-MT: Compression Using Language and Layer pruning for Machine Translation (https://arxiv.org/abs/2411.06506)
- **What's New**: CULL-MT(Compression for Unimportant Layers in Machine Translation)는 구조적 레이어 가지치기(Structural Layer Pruning)와 선택적 언어 방향(Selected Language Directions)에 기반한 기계 번역 모델의 압축 방법을 제시합니다. 이 방법은 중요하지 않은 레이어를 식별하고 제거하여 비용을 절감하는 동시에 원 모델에서의 지식 증류(Knowledge Distillation)를 통해 성능 저하를 최소화합니다.

- **Technical Details**: CULL-MT는 반복적으로 각 레이어의 중요성을 평가하고, 중요하지 않은 레이어를 제거하는 탐욕적 방법을 사용합니다. 이후 모델은 최적의 번역 성능을 위해 세밀한 조정(Fine-tuning)을 수행합니다. NLLB-3.3B 모델을 사용하여 다국어 번역 시나리오에서 25%의 레이어를 제거해도 0.9 spBLEU(Score with Pairwise BLEU) 점수 하락을 보였으며, LLaMA3.1-8B-Instruct 모델은 5개의 레이어를 제거한 후 2.0 spBLEU 점수 하락을 기록했습니다.

- **Performance Highlights**: CULL-MT를 통해 NLLB-3.3B 모델의 다국어 번역 성능을 유지하면서 12개의 레이어를 가지치기 할 수 있었고, 단일 방향으로는 15개의 레이어를 제거해 1.2의 spBLEU 점수 하락을 보였습니다. LLaMA3.1-8B-Instruct 모델은 5개의 레이어를 제거한 후 2.0의 spBLEU 점수 하락을 경험했습니다.



### VocalTweets: Investigating Social Media Offensive Language Among Nigerian Musicians (https://arxiv.org/abs/2411.06477)
Comments:
          13 pages, 5 figures, 6 tables

- **What's New**: 이번 연구에서는 나이지리아의 유명한 음악가 12명의 트윗(tweets)을 포함한 VocalTweets라는 언어 혼합(code-switched) 및 다국어(multilingual) 데이터셋을 소개합니다.

- **Technical Details**: VocalTweets 데이터셋은 정규(Normal) 또는 공격적(Offensive)으로 이항 분류(binary classification)된 트윗으로 구성되어 있으며, HuggingFace의 base-Twitter-RoBERTa 모델을 사용하여 훈련되었습니다.

- **Performance Highlights**: 모델은 F1 점수(F1 score) 74.5를 달성했으며, OLID 데이터셋과의 교차 코퍼스(cross-corpus) 실험을 진행하여 데이터셋의 일반화 가능성(generality)을 평가했습니다.



### ClinicalBench: Can LLMs Beat Traditional ML Models in Clinical Prediction? (https://arxiv.org/abs/2411.06469)
Comments:
          The first two authors contributed equally. 10 pages for main paper, 66 pages including appendix. Project website: this https URL

- **What's New**: 최근 연구에서 일반 목적 및 의료용 Large Language Models (LLMs)의 임상 예측 모델링 능력을 평가하기 위해 ClinicalBench라는 새로운 벤치마크가 개발되었습니다.

- **Technical Details**: ClinicalBench는 세 가지 일반적인 임상 예측 작업, 두 개의 데이터베이스, 14개의 일반 목적 LLM, 8개의 의료 LLM, 그리고 11개의 전통적인 ML 모델(SVM, XGBoost 포함)을 포함합니다. 실험을 통해 다양한 프롬프트와 파인튜닝 전략을 적용했지만 LLM은 여전히 전통적인 ML 모델에 비해 임상 예측에서는 성능이 부족한 것으로 나타났습니다.

- **Performance Highlights**: 임상 예측 작업에서 LLMs가 전통적인 ML 모델을 초월하지 못하였으며, 이는 LLMs의 임상 추론 및 의사결정에서의 잠재적 부족을 나타냅니다. 따라서 의료 분야에서의 LLM 적용에 대해 신중함이 요구됩니다.



### Prompt-Efficient Fine-Tuning for GPT-like Deep Models to Reduce Hallucination and to Improve Reproducibility in Scientific Text Generation Using Stochastic Optimisation Techniques (https://arxiv.org/abs/2411.06445)
Comments:
          73 pages, 6 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 과학 텍스트 생성을 위한 복잡한 작업에서의 제한사항을 극복하기 위해 고안된 새로운 파라미터 효율 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 접근법을 소개합니다. 이 접근법은 특히 질량 분석(mass spectrometry) 분야에서의 재현 가능성을 높이는 데 집중하고 있습니다.

- **Technical Details**: 모델은 오픈AI의 GPT-2를 기반으로 한 MS-GPT로 명명되었으며, 저자들은 질량 분석 문헌에 특화된 데이터셋을 사용하여 LoRA(Low-Rank Adaptation) 어댑터를 적용하여 미세 조정을 진행했습니다. BLEU, ROUGE, Perplexity 등 여러 평가 지표를 통해 MS-GPT 모델이 baseline GPT-2보다 더 높은 텍스트 일관성과 재현 가능성을 보였음을 통계 분석을 통해 검증했습니다.

- **Performance Highlights**: MS-GPT는 미세 조정 이전보다 BLEU 점수가 0.33에서 0.34로 증가하였고, ROUGE-1은 0.42에서 0.44로, ROUGE-L은 0.57에서 0.62로 향상되었습니다. Perplexity 점수는 13586.37에서 10092.12로 감소하고, 재현 가능성 점수는 0.83에서 0.84로 향상되었습니다. Perplexity의 개선은 5% 유의수준 하에서 통계적으로 유의미한 차이를 보였습니다.



### PLM-Based Discrete Diffusion Language Models with Entropy-Adaptive Gibbs Sampling (https://arxiv.org/abs/2411.06438)
- **What's New**: 본 논문에서는 기존의 Pretrained Language Model (PLM)과 Discrete Diffusion Language Model (DDLM)을 통합하는 새로운 방법, Diffusion-EAGS를 제안합니다. 이 방법은 PLM을 활용해 데이터셋 기반 생성 작업의 성능을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: Diffusion-EAGS는 Mask Language Model (MLM)을 기반으로 PLM을 DDLM에 통합하는 방식으로, 각 단계에서의 denoising 기능을 MLM에 의해 수행하도록 설계되었습니다. 또한, 엔트로피 추적 모듈을 도입하여 diffusion 과정에서 denoising 적용 위치를 결정하는 데 도움을 줍니다. 이는 each denoising step을 제약된 Markov Random Field (cMRF)로 해석하여 adaptive Gibbs sampling을 활용하는 방법입니다.

- **Performance Highlights**: 실험 결과, Diffusion-EAGS는 기존의 다양한 데이터를 기반으로 하는 생성 작업에서 높은 텍스트 품질과 다양성을 달성하며, token-level 생성이 가능하여 여러 데이터셋 유도 생성 작업에서 적용 가능성을 보여줍니다. 추가로, 낮은 자원 환경 및 이중언어 설정에서도 잘 작동하는 것을 입증하였습니다.



### Fineweb-Edu-Ar: Machine-translated Corpus to Support Arabic Small Language Models (https://arxiv.org/abs/2411.06402)
- **What's New**: 이번 연구에서는 HuggingFace에서 제공하는 FineWeb-Edu 데이터셋을 기계 번역하는 과정을 통해 아랍어로 번역된 FineWeb-Edu-Ar 데이터셋을 공개합니다. 이는 공개적으로 이용 가능한 가장 큰 아랍어 기계 번역 데이터셋입니다.

- **Technical Details**: FineWeb-Edu-Ar은 2020B 토큰의 범위를 가지며, 아랍어 전용 토크나이저에 최적화되어 있습니다. 번역 작업에서 nllb-200-distilled-600M 모델이 가장 우수한 성능을 보였으며, 다양한 기계 번역 모델의 성능 평가에 대한 자세한 분석이 포함되어 있습니다.

- **Performance Highlights**: 번역 품질은 LLM-as-a-Judge 접근 방식을 사용하여 평가되었으며, 24점 만점에 대한 평균 점수를 통해 다양한 MT 모델의 성능을 비교했습니다. nllb-200-distilled-600M 모델이 계산 예산에 가장 적합한 모델로 선정되었습니다.



### LLM Vocabulary Compression for Low-Compute Environments (https://arxiv.org/abs/2411.06371)
Comments:
          Machine Learning and Compression Workshop @ NeurIPS 2024

- **What's New**: 본 논문은 언어 모델의 최종 선형 층을 압축하는 방법을 제안하여 메모리 사용량을 최대 3.4배 줄이는 동시에 성능 저하를 최소화합니다. Byte Pair Encoding (BPE) 병합에 따라 토큰을 그룹화함으로써 메모리 집약적인 logits 텐서의 물화를 방지합니다.

- **Technical Details**: 우리는 언어 모델의 최종 임베딩 층의 메모리 발자국을 줄이는 방법으로, 토큰을 그룹화하고 최종 토큰을 예측하는 두 단계의 과정을 통해 어휘 층을 효과적으로 압축합니다. 새로운 접근 방식에서는 두 개의 개별 모델을 사용하는 대신, 숨겨진 상태를 기반으로 그룹 및 토큰 예측을 모두 학습할 수 있는 간단한 선형 층을 적용합니다.

- **Performance Highlights**: TinyStories 데이터셋에서의 평가 결과, 본 방법은 GPT-Neo와 GPT2와 동등한 성능을 보이며, 처리량을 최대 3배 향상시켜 저전력 환경에 적합하다는 것을 확인했습니다.



### Prompts Matter: Comparing ML/GAI Approaches for Generating Inductive Qualitative Coding Results (https://arxiv.org/abs/2411.06316)
Comments:
          Accepted by AERA 2025 Annual Meeting

- **What's New**: 이번 연구는 최근의 인공지능 발전, 특히 생성적 인공지능(Generative AI)을 활용하여 질적 코딩 과정에서의 기여를 탐구합니다.

- **Technical Details**: 연구에서는 두 가지 기존 접근법과 두 가지 이론 기반의 새로운 접근법을 온라인 커뮤니티 데이터셋에 적용하고, 생성된 코딩 결과를 평가했습니다.

- **Performance Highlights**: 결과적으로 ML/GAI 접근법 간에 상당한 차이가 나타났으며, 인간 코딩 과정을 GAI 프롬프트에 통합한 접근법의 우수성을 입증했습니다.



### Golden Touchstone: A Comprehensive Bilingual Benchmark for Evaluating Financial Large Language Models (https://arxiv.org/abs/2411.06272)
Comments:
          26 pages, 9 tables, 3 figures

- **What's New**: 금융 분야에서 대규모 언어 모델(LLMs)의 성능을 종합적으로 평가하기 위한 표준화된 방법이 필요합니다. 기존 금융 벤치마크들이 지나치게 제한된 언어 및 작업 범위와 낮은 품질의 데이터셋으로 고통받는 문제를 해결하기 위해, "Golden Touchstone"을 제안합니다. 이 벤치마크는 중국어 및 영어의 8개 핵심 금융 NLP 작업을 부각시키는 최초의 종합적인 이중 언어 벤치마크입니다.

- **Technical Details**: Golden Touchstone 벤치마크는 고품질 데이터셋, 작업에 적합한 메트릭 및 LLM의 작업 적합한 응답 생성을 안내하는 템플릿을 포함하고 있습니다. 또한, Touchstone-GPT 모델은 특정 금융 작업을 위한 지속적인 사전 훈련과 금융 지침 조정을 통해 훈련되었습니다.

- **Performance Highlights**: 금융 감정 분석 및 엔티티 추출과 같은 여러 작업에서 GPT-4o, Qwen-2, Llama-3 및 FinMA 모델이 우수한 성과를 보였지만, 신용카드 스코어링 및 주식 움직임 예측 작업에서는 상당한 개선 여지가 발견되었습니다. Touchstone-GPT는 이중 언어 벤치마크에서 견고한 성능을 보이는 반면, 주식 움직임 예측 및 질문 응답 작업에서 한계를 나타냈습니다.



### Robust Detection of LLM-Generated Text: A Comparative Analysis (https://arxiv.org/abs/2411.06248)
Comments:
          8 pages

- **What's New**: 이번 논문은 LLM(대규모 언어 모델)이 생성한 텍스트를 식별하기 위한 새로운 탐지 기법에 대한 연구를 다룹니다. 특히, LLM 생성 텍스트의 식별은 사회 미디어와 같은 플랫폼에서 잘못된 정보를 방지하는 데 필수적입니다.

- **Technical Details**: 논문에서는 로지스틱 회귀(logistic regression), K-평균 군집화(k-means clustering), 가우시안 나이브 베이즈(Gaussian Naive Bayes), 서포트 벡터 머신(support vector machines) 등 전통적인 기계 학습(methods) 기술과 BERT와 같은 변환기(converter) 기반 기법을 사용합니다. 또한 LLM을 활용하여 LLM 생성 텍스트를 탐지하는 알고리즘을 탐색합니다.

- **Performance Highlights**: 모델의 일반화(generalization), 잠재적 적대적 공격(adversarial attacks), 모델 평가(accuracy)에서의 성능 강조가 이루어졌으며, 향후 연구 방향성 제안과 현재 실험 결과를 요약하였습니다.



### An $\mathbf{L^*}$ Algorithm for Deterministic Weighted Regular Languages (https://arxiv.org/abs/2411.06228)
- **What's New**: 이 논문에서는 블랙박스 모델에서 유한 상태 자동자(finite state automata, FSA)를 추출하는 새로운 접근 방식을 제안합니다. 특히, Angluin의 1987년 알고리즘인 $	extbf{L^*}$을 기반으로 한 가중치 변형을 소개하여 결정론적 가중치 FSA를 학습할 수 있도록 했습니다.

- **Technical Details**: 제안된 알고리즘은 결정론적 가중치 FSA를 정확하게 학습할 수 있도록 설계되었으며, FSA 최소화와의 관련성을 강조합니다. 알고리즘은 세미필드 가중치(semifield-weighted) FSA에 대해서도 작동하며, 원래의 알고리즘에 충실한 일반화를 제공합니다.

- **Performance Highlights**: 이 연구는 $	extbf{L^*}$ 알고리즘이 목표 언어에 대한 최소 자동자를 직접 학습하는 방식을 보여주며, 기존의 비결정론적 가중치 FSA를 학습하는 다양한 방법과의 차별성을 두고 있습니다.



### Incorporating Human Explanations for Robust Hate Speech Detection (https://arxiv.org/abs/2411.06213)
Comments:
          2021 ACL Unimplicit Workshop

- **What's New**: 본 연구에서는 대규모 Transformer 언어 모델(LM)의 검은 상자(black-box) 특성과 복잡성을 고려하여, 증오 발언(hate speech, HS) 탐지의 일반화 가능성과 강건성에 대한 우려를 다룹니다. 특히 신뢰성을 높이기 위해 새로운 작업인 Stereotype Intent Entailment (SIE)를 도입하여, 모델이 비인식적 의미를 더 잘 평가하도록 유도합니다.

- **Technical Details**: Social Bias Frames(SBF) 데이터셋을 활용하여 HS 트윗을 분류하고, SIE 작업을 통해 HS 트윗과 고유의 고정관념(stereotype)을 연결하는 방법을 연구합니다. SIE 데이터셋은 220,000개의 트윗 및 고정관념 쌍으로 구성되며, LM-SIE는 테스트 세트에서 F1 점수 87.6%, 정확도 87.6%를 기록했습니다.

- **Performance Highlights**: 기존 LM-HS 모델은 신뢰할 수 없는 단어에 비중을 두어, 희귀 단어를 사용한 공격으로 정확도가 절반 감소하고, 질문 단어로는 25% 감소하는 결과를 보였습니다. 그러나 SIE를 통한 인간 설명을 모델에 통합함으로써 강건성이 개선되었고, LM-SIE 모델은 공격에 대한 저항력이 더 강했습니다.



### IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization (https://arxiv.org/abs/2411.06208)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 복잡한 지시사항을 따르는 능력을 향상시키고 평가하기 위한 TRACE 벤치마크를 소개합니다. 이 벤치마크는 12만 개의 훈련 데이터와 1천 개의 평가 데이터로 구성되어 있습니다.

- **Technical Details**: TRACE는 5개의 제약 유형과 26개 제약 차원으로 분류된 복잡한 지시사항의 자동 작성 데이터를 기반으로 하며, ‘Input-Output Preference Optimization (IOPO)’라는 새로운 정렬 방법을 제안합니다. IOPO는 입력 지시사항과 출력 선호 쌍을 고려하여 LLM이 지시사항 선호를 효율적으로 탐색할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, IOPO를 사용한 경우 in-domain 데이터에서 각각 8.15%, 2.18%의 개선을 보였고, out-of-domain 데이터에서는 각각 6.29%, 3.13%의 성능 향상이 있었습니다. 이는 SFT 및 DPO 방법과 비교하여 평균적으로 7.22% 및 2.66%의 향상된 결과를 나타냅니다.



### Exploring Knowledge Boundaries in Large Language Models for Retrieval Judgmen (https://arxiv.org/abs/2411.06207)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능이 외부 정보를 실시간으로 검색하는 Retrieval-Augmented Generation (RAG)에 의해 크게 향상될 수 있다는 점을 강조하고 있습니다. 연구팀은 RAG가 LLMs의 질문 응답 능력에 미치는 영향을 유익한, 중립적인 및 해로운 세 가지 유형으로 분류하고, 중립적이거나 해로운 결과를 초래하는 검색 요청의 비율을 줄여 성능을 향상시킬 수 있는 방법을 제안합니다.

- **Technical Details**: 연구에서 제안하는 Knowledge Boundary Model (KBM)은 LLMs가 질문의 유형에 따라 검색 요청을 최적화할 수 있도록 훈련됩니다. 이 모델은 장기적 정적 지식, 동적 지식 변화를 이해하고, 여러 시나리오에서 LLM의 지식 경계를 효과적으로 도형화할 수 있는 방법에 대해 실험합니다. 또한, 성과 지표로서 신뢰도(confidence)와 확실성(certainty)의 기준을 설정하여 성과를 평가합니다.

- **Performance Highlights**: 실험 결과, KBM은 WebQA에서 검색 비율을 43.17% 줄이면서 성능의 저하가 1.7%에 불과하다는 것을 보여주었습니다. 또 다른 성과로, 확실성 기반 접근 방식으로 QA 작업에서 약 10%의 검색 비율을 감소시키면서도 ALL RAG에 가까운 성능을 유지했습니다. 이 연구는 LLMs의 지식 경계를 평가하며 검색 효율성을 극대화하고 질문 응답 성능을 향상시키는 결과를 가져오고 있습니다.



### WMT24 Test Suite: Gender Resolution in Speaker-Listener Dialogue Roles (https://arxiv.org/abs/2411.06194)
- **What's New**: 본 논문은 문학 스타일 대화에서 성별 해석의 난이도와 성별 고정관념(gender stereotypes)이 미치는 영향을 평가합니다. 기존의 연구와는 달리 캐릭터와 화법에 대한 외부 메타 맥락이 성별 합의에 미치는 중대한 영향을 발견했습니다.

- **Technical Details**: 이 테스트 슈트(test suite)는 문학 스타일 대화 설정에서 기계 번역 시스템의 성별 해석 경향을 측정합니다. 세 가지 목표 언어(스페인어, 체코어, 아이슬란드어)를 포함하고 있으며, 알고리즘은 맥락에 따라 성별을 해석합니다. 주요 방법론으로 사전 검색(dictionary searches)을 사용하여 성별 합의 라벨을 얻고, 다양한 성별 고정관념이 있는 특성을 고려하여 회귀 분석을 수행합니다.

- **Performance Highlights**: 테스트 결과, 성별 고정관념이 있는 화법과 캐릭터 설명이 성별 해석에 미치는 영향력이 상당히 큽니다. 특히, 어떠한 성별 고정관념을 사용했는지에 따라 번역 시스템에서 성별 선택의 차이가 10% 이상 나타날 수 있습니다.



### M-Longdoc: A Benchmark For Multimodal Super-Long Document Understanding And A Retrieval-Aware Tuning Framework (https://arxiv.org/abs/2411.06176)
- **What's New**: 이 논문에서는 M-LongDoc라는 새로운 벤치마크와 이를 평가하기 위한 자동화된 프레임워크를 소개합니다. M-LongDoc은 851개의 샘플로 구성되어 있으며, 긴 멀티모달(한 가지 이상의 유형을 포함하는) 문서를 읽는 데 필요한 모델 성능을 평가합니다. 기존 작업들과의 차별점으로, 최근의 문서와 다수의 페이지를 포함하며, 오픈 엔디드(open-ended) 솔루션을 요구합니다.

- **Technical Details**: M-LongDoc은 200페이지 이상의 긴 문서와 복잡한 멀티모달 내용을 포함한 질문을 다루며, 모델들이 텍스트, 그림, 테이블을 분석하고 추론할 수 있도록 요구합니다. 자동화된 평가 프레임워크는 모델이 생성한 솔루션의 정확성을 평가하기 위해 여러 평가 모델을 사용합니다. 또한, retrieval-aware tuning 접근 방식이 제안되어 다양한 도메인 지식을 효과적으로 통합하면서 관련 없는 내용을 무시하도록 모델을 조정하는데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안된 조정 방법이 기존 오픈소스 모델에 비해 4.6%의 응답 정확성 향상을 달성하였습니다. M-LongDoc 벤치마크는 모델의 멀티모달 문서 이해 성능의 새로운 기준을 제시하고 있으며, 향후 연구와 실제 적용을 위한 중요한 자원입니다.



### Clustering Algorithms and RAG Enhancing Semi-Supervised Text Classification with Large LLMs (https://arxiv.org/abs/2411.06175)
- **What's New**: 이 논문은 제한된 레이블 예제로부터 효과적으로 학습할 수 있는 혁신적인 semi-supervised learning 방법을 도입합니다. 특히, retrieval-augmented generation (RAG)과 기존의 통계적 클러스터링을 통합하여 레이블이 지정된 인스턴스의 수가 최소화된 상태에서도 고품질의 레이블 데이터를 생성할 수 있습니다.

- **Technical Details**: 이 방법은 클러스터링 알고리즘을 통해 데이터 선택을 우선 진행하고, 그 후 전문가들이 수작업으로 레이블링을 수행합니다. 생성된 레이블은 RAG와 CoT 파이프라인을 통해 더 큰 크기의 LLM에서 추가 학습 샘플을 생성하는 데 사용됩니다. 마지막으로, 이러한 학습 샘플은 더 작은 LLM을 fine-tune하는 데 사용되어 잘못 분류된 샘플에 다시 집중할 수 있도록 설계됩니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 레이블링된 데이터의 양이 동일할 때 무작위 선택 및 기존 방법들보다 fine-tuned 모델의 정확도를 높이는 것으로 나타났습니다. 또한, 복잡한 텍스트 문서 분류 작업에서 95.41%와 82.43%의 정확도를 기록하여 최신 성과를 달성했습니다.



### SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models (https://arxiv.org/abs/2411.06171)
Comments:
          EMNLP2024

- **What's New**: 본 연구에서는 대형 언어 모델(LLM)의 지속 학습(Continual Learning)에서 주의(attention) 가중치의 중요성을 강조하며, 이를 바탕으로 데이터 효율적 재생(replay) 기반 지속 학습을 위한 SElective attEntion-guided Knowledge Retention 방법(SEEKR)을 제안합니다.

- **Technical Details**: SEEKR은 선택된 주의 헤드를 대상으로 주의 증류(attention distillation)를 수행하여 더 정교한 지식 보존을 달성합니다. 이 방법은 forgettability와 task-sensitivity 기반의 지표를 통해 가장 가치 있는 주의 헤드를 식별합니다. SEEKR은 과거 작업의 지식을 효과적으로 유지하기 위해 계층적 예산 할당 메커니즘을 사용합니다.

- **Performance Highlights**: 실험 결과, SEEKR은 기존 방법보다 성능과 효율성에서 우수한 결과를 보이며, 기존 방법이 사용하는 재생 데이터의 1/10만으로도 유사하거나 더 나은 성능을 달성하고, 재생 데이터 비율을 1%로 감소시킴으로써 데이터 효율성을 입증합니다.



### Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework (https://arxiv.org/abs/2411.06160)
- **What's New**: 이 논문은 텍스트 감정 탐지(Text Emotion Detection)의 발전을 위해 Emotion Quantization Network (EQN) 프레임워크를 제안합니다. EQN은 감정 강도를 에너지 수준으로 매핑하여 미세 감정(micro-emotion)의 자동 탐지 및 주석(annotation)을 가능하게 합니다.

- **Technical Details**: EQN 프레임워크는 모든 라벨(all-labels) 및 훈련 세트 라벨 회귀(training-set label regression) 방법을 활용합니다. 이를 통해 기계 모델의 학습능력과 라벨 간의 상호 의존성을 최대한 활용하여 샘플 내 여러 감정을 발견합니다. 이론적으로 미세 감정의 측정과 주석 작성에 기여하는 새로운 접근법이 설명됩니다.

- **Performance Highlights**: EQN 프레임워크는 GoEmotions 데이터셋에서 감정 탐지 및 주석을 수행하며, Google 연구 결과와의 포괄적 비교를 통해 높은 자동 탐지 능력을 입증합니다. EQN은 에너지 레벨 점수로 미세 감정 자동 주석을 최초로 달성하였으며, 감정 탐지 분석 및 감정 컴퓨팅의 정량적 연구에 강한 지원을 제공합니다.



### From References to Insights: Collaborative Knowledge Minigraph Agents for Automating Scholarly Literature Review (https://arxiv.org/abs/2411.06159)
- **What's New**: 이 논문에서는 협력 지식 미니그래프 에이전트(CKMA)를 제안하여 자동으로 학술 문헌 리뷰를 수행하는 새로운 프레임워크를 소개합니다.

- **Technical Details**: CKMA는 지식 미니그래프 구성 에이전트(KMCA)와 다중 경로 요약 에이전트(MPSA)로 구성되며, 이는 대형 언어 모델(LLM)을 활용하여 정보 조각과 그 관계를 효율적으로 정리하고 요약합니다.

- **Performance Highlights**: CKMA는 세 가지 벤치마크 데이터셋에서 테스트되었으며, 생성된 요약은 정보가 풍부하고 완전하며 일관되고 통찰력 있는 것으로 입증되었습니다.



### Building an Efficient Multilingual Non-Profit IR System for the Islamic Domain Leveraging Multiprocessing Design in Rus (https://arxiv.org/abs/2411.06151)
- **What's New**: 이 논문에서는 이슬람 도메인을 위한 다국어 비영리 정보 검색 시스템을 개발하였으며, 복잡한 언어 환경에서의 데이터 검색을 향상시키는 방법을 제시합니다.

- **Technical Details**: 이 연구는 언어 감소(language reduction) 및 지속적인 재학습(continued pre-training)을 통해 도메인 특화(Multilingual Domain-Specific) 언어 모델을 경량화했습니다. 모델 크기를 절반 이상 줄이면서도 도메인 특정 단어집을 추가하여 성능을 개선하였습니다.

- **Performance Highlights**: 제안된 모델은 일반 도메인에서 사전 학습된 더 큰 모델들과 비교했을 때, 더욱 우수한 성능을 보였습니다. Rust 언어를 활용하여 CPU 아키텍처에서 효율적인 의미론적 검색(semantic search)을 구현함으로써 성능을 저하시키지 않으면서도 검색 속도를 크게 향상시켰습니다.



### Detecting Reference Errors in Scientific Literature with Large Language Models (https://arxiv.org/abs/2411.06101)
- **What's New**: 이번 연구에서는 OpenAI의 GPT 모델이 인용 오류(quotation errors)를 자동으로 탐지할 수 있는 능력을 평가하였다. 이를 통해 인용 오류가 포함된 과학 논문에서의 관련 정보의 변화를 고려하였다.

- **Technical Details**: 연구진은 문장-참조 페어의 전문가 주석 데이터셋을 생성하고, 대형 언어 모델(LLMs)이 참조 정보의 양에 따라 다른 환경에서 평가되었다. 실험은 3가지 설정(제목만 제공, 제목 및 초록 제공, 제목, 초록 및 발췌 제공)으로 수행되었다.

- **Performance Highlights**: GPT-4 Turbo와 GPT-4o 모델은 GPT-3.5 Turbo보다 Unsubstantiated 케이스를 탐지하는 데 더 좋은 성능을 보였다. 기존 모델은 our dataset의 발췌를 사용했을 때 ‘정보가 부족함(Not Enough Information)’으로 예측하였다.



### ZhoBLiMP: a Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chines (https://arxiv.org/abs/2411.06096)
- **What's New**: 본 논문에서는 중국어를 위한 가장 포괄적인 언어학적 최소 쌍 벤치마크인 ZhoBLiMP를 소개합니다. 이는 118개의 패러다임으로 구성되어 있으며 15개의 언어학적 현상을 포괄합니다.

- **Technical Details**: ZhoBLiMP는 중국 언어학자들이 개발한 문법 템플릿과 어휘를 사용하여 생성되었으며, 35,000개의 최소 쌍을 포함합니다. 저자들은 서로 다른 크기의 20개의 언어 모델(14M에서 1.4B까지)을 중국어 코퍼스(100M에서 3B 토큰)로 학습시키고, 14개의 기존 LLM과 함께 ZhoBLiMP에서 평가했습니다. 결과적으로, 약 500M의 파라미터와 1B 토큰의 훈련 데이터가 필요하다는 것이 밝혀졌습니다.

- **Performance Highlights**: 중국어 문법은 약 500M의 파라미터를 가진 모델이 1B 토큰으로 훈련된 후 거의 대부분 학습 가능한 것으로 나타났습니다. 그러나 32B 파라미터를 가진 모델에게도 여전히 도전적인 13개의 패러다임이 존재하고, 유사 아동 언어 습득 패턴(U-shaped learning)을 관찰하였습니다.



### Zyda-2: a 5 Trillion Token High-Quality Datas (https://arxiv.org/abs/2411.06068)
Comments:
          initial upload 11/08/24

- **What's New**: Zyda-2라는 새로운 다섯 조 트리온(token) 데이터셋이 발표되었습니다. 이는 언어 모델 사전 훈련(pretraining) 용도로 사용되며, Zamba2 시리즈 모델을 교육하는 데 사용됩니다.

- **Technical Details**: Zyda-2는FineWeb과 DCLM과 같은 고품질 오픈 소스 token을 수집하여 제작되었습니다. 이를 통해 교차 중복 제거(cross-deduplication)와 모델 기반 품질 필터링(model-based quality filtering)을 통해 최고의 품질 하위 집합을 증류(distilling)하였습니다.

- **Performance Highlights**: Zyda-2를 기반으로 한 Zamba2 모델 시리즈는 해당 가중치(weight) 클래스에서 최첨단(state-of-the-art) 성능을 자랑합니다.



### Sufficient Context: A New Lens on Retrieval Augmented Generation Systems (https://arxiv.org/abs/2411.06037)
- **What's New**: 본 논문에서는 LLMs (Large Language Models)와 context의 활용에 대해 다루고 있으며, 특히 Retrieval Augmented Generation (RAG) 시스템의 성능 향상에 기여하는 새로운 충분한 컨텍스트의 개념을 제안합니다.

- **Technical Details**: 저자들은 충분한 컨텍스트(sufficient context)를 정의하고, 이를 기반으로 여러 모델과 데이터셋을 분석하여 오류를 분류했습니다. 특히, LLM이 제공하는 답변의 품질이 컨텍스트의 충분성과 어떻게 연결되는지를 연구했습니다. 이 분석을 통해 독점 LLM(예: Gemini, GPT, Claude)은 컨텍스트가 충분할 때 우수한 성능을 보였지만, 부족할 경우 잘못된 답변을 생성하는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 제안된 방법은 Gemini, GPT, Gemma에서 모델이 반응할 때 올바른 답변의 비율을 2-10% 증가시켰습니다. 특히 RAG 시스템의 환각(hallucination)을 줄일 수 있는 새로운 선택적 생성(selective generation) 방법도 탐구하였습니다.



### LLM-GLOBE: A Benchmark Evaluating the Cultural Values Embedded in LLM Outpu (https://arxiv.org/abs/2411.06032)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 문화적 가치 시스템을 평가하기 위한 LLM-GLOBE 벤치마크를 제안합니다. LLM의 문화적 가치 연구는 초기 단계에 있으며, 동서양 LLM의 가치를 비교합니다.

- **Technical Details**: LLM-GLOBE 벤치마크는 GLOBE 프레임워크를 기반으로 하여 LLM의 문화적 가치 시스템을 평가합니다. 연구 방법론으로는 'LLMs-as-a-Jury' 파이프라인이 포함되어 있으며, 이는 개념적 수준에서 대규모 분석을 가능하게 합니다.

- **Performance Highlights**: 결과적으로, 동서양의 문화적 가치 시스템 간의 유사성과 차이를 명확하게 밝혔고, 개방형 생성 과제가 문화적 가치 평가에서 유망한 방향임을 제안했습니다.



### Improved intent classification based on context information using a windows-based approach (https://arxiv.org/abs/2411.06022)
Comments:
          In preparation for Journal Submission

- **What's New**: 이 연구에서는 대화의 맥락을 고려하여 사용자의 의도를 구분하는 방법을 제안합니다. 기존의 연구에서는 단일 발화만 사용하여 의도를 분류했으나, 이번 방법은 이전 발화들을 결합하여 보다 정확한 분류를 가능하게 합니다.

- **Technical Details**: 이 방법은 대화 기록과 현재 발화를 연결하여 의도 분류를 수행하는 컨볼루션 신경망(convolutional neural network)을 사용합니다. BERT로부터 얻은 효과적인 벡터 표현을 활용하며, 윈도우 기반 접근법(window-based approach)을 통해 컨텍스트를 포함하여 모델을 학습합니다.

- **Performance Highlights**: 실제 브라질 포르투갈어 데이터셋에서 수행한 실험 결과, 컨텍스트를 포함하여 사용자 발화와 시스템 응답을 사용했을 때 세 가지 접근법 모두 기본선(Baseline)보다 상당한 성능 향상을 보여주었습니다.



### The Dark Patterns of Personalized Persuasion in Large Language Models: Exposing Persuasive Linguistic Features for Big Five Personality Traits in LLMs Responses (https://arxiv.org/abs/2411.06008)
Comments:
          31 pages

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 개인의 성격 특성에 따라 언어적 특징을 조정하여 개인화된 설득적 출력을 생성하는 방법을 탐구합니다. 구체적으로 LLMs가 자신의 반응을 어떻게 조정하는지에 대한 첫 번째 연구를 제시합니다.

- **Technical Details**: 연구에서는 성격의 Big Five 모델에 따라 개인을 설득할 때 중요한 13가지 언어적 특징을 밝혀냈습니다. 연구는 5개의 모델 계열에서 19개의 LLM의 출력이 성격 특성 정보가 포함된 프롬프트에 어떻게 영향을 받는지를 분석했습니다. Shapley 값 등의 기법을 통해 언어적 특징의 중요성을 평가한 후 회귀 분석을 통해 성격 특성과 모델의 출력 간의 상호작용을 조사했습니다.

- **Performance Highlights**: 연구 결과, LLM들은 신경증에서 불안 관련 단어를 더 많이 사용하고, 성실성에서 성취 관련 단어를 증가시키며, 경험에 대한 개방성에서는 인지 과정과 관련된 단어를 덜 사용하는 경향이 있음을 보여줍니다. 이는 LLM이 사용자의 성격 단서에 따라 반응을 조정할 수 있음을 나타내며, 개인의 정신 건강과 복지에 영향을 줄 수 있는 설득적 콘텐츠를 생성할 가능성을 보여줍니다.



### GUIDEQ: Framework for Guided Questioning for progressive informational collection and classification (https://arxiv.org/abs/2411.05991)
- **What's New**: 본 연구에서 제안하는 GUIDEQ 프레임워크는 불완전한 정보로부터 안내된 질문을 생성하여 보다 정확한 정보 수집을 가능하게 합니다.

- **Technical Details**: GUIDEQ는 LLMs(Large Language Models)의 추론 능력을 활용하고, 분류기 모델의 해석 가능성을 결합하여 가장 중요한 키워드에 따라 지침 질문을 형성합니다.

- **Performance Highlights**: 실험 결과, GUIDEQ는 다른 LLM 기반의 방법론에 비해 F1 점수를 개선하며, 질문의 품질 또한 더 우수함을 보여주었습니다.



### Fine-Grained Reward Optimization for Machine Translation using Error Severity Mappings (https://arxiv.org/abs/2411.05986)
Comments:
          10 pages, work-in-progress

- **What's New**: 본 연구에서는 강화학습(Reinforcement Learning, RL) 기법을 이용하여 문장 수준의 피드백보다 더 세밀한 토큰 수준의 보상 메커니즘을 도입하는 새로운 접근법을 제안합니다. 이 방법은 번역 품질을 효율적으로 평가하기 위해 xCOMET라는 최첨단 품질 추정 시스템을 활용합니다.

- **Technical Details**: 이 연구에서는 NMT(Neural Machine Translation) 시스템을 훈련하기 위해 세밀한 보상 신호를 통합합니다. xCOMET는 소스-번역 쌍에 대한 세밀한 오류 범위와 심각도를 예측하여 보다 정밀한 피드백을 제공합니다. 실험을 통해 문장 수준의 보상 신호와 세밀한 보상 신호가 번역 품질에 미치는 영향을 비교하였습니다.

- **Performance Highlights**: 토큰 수준의 보상으로 훈련한 결과, 여러 언어 쌍에서 자동 평가 및 인간 평가 모두에서 기본 시스템 대비 번역 품질이 향상되는 것이 확인되었습니다. 추가적으로, 훈련 안정성 또한 향상되어 훈련 에포크가 진행됨에 따라 평균 보상이 지속적으로 증가하는 경향을 보였습니다.



### FactLens: Benchmarking Fine-Grained Fact Verification (https://arxiv.org/abs/2411.05980)
Comments:
          12 pages, under review

- **What's New**: 이 논문은 기존 LLM(대규모 언어 모델)의 사실 검증 방식을 재검토하고, 복잡한 주장을 세분화하여 각 서브 클레임(sub-claim)을 독립적으로 검증할 수 있는 방법을 제시합니다. 이를 통해 정확도, 투명성 및 증거 검색의 모호성을 줄일 수 있습니다.

- **Technical Details**: 저자들은 FactLens라는 새로운 벤치마크를 도입하여 세분화된 사실 확인을 평가합니다. FactLens는 서브 클레임의 품질을 평가하기 위한 메트릭스와 자동화된 평가자를 포함하고 있으며, 벤치마크 데이터는 수동으로 큐레이션되어 고품질의 그라운드 트루스를 보장합니다.

- **Performance Highlights**: 세분화된 검증 처리와 관련된 정량적 메트릭스를 통해 자동화된 평가자와 인간의 판단 간의 일치도를 입증하였으며, 서브 클레임 생성의 도전 과제를 논의하고, 최첨단 모델의 결과를 제시하였습니다.



### The Empirical Impact of Data Sanitization on Language Models (https://arxiv.org/abs/2411.05978)
Comments:
          Paper accepted at Safe Generative AI Workshop at NeurIPS 2024

- **What's New**: 본 논문은 자연어 처리(NLP)에서 데이터 빈소화(data sanitization)가 언어 모델의 이해 능력에 미치는 영향을 실증적으로 분석합니다. 특히, 개인 식별 정보(PII)가 포함된 데이터에서 이러한 정보를 제거할 때의 효과를 다양한 기준 언어 모델링 작업에 대해 평가합니다.

- **Technical Details**: 실험은 BART와 GPT-2와 같은 소규모 모델의 미세 조정(fine-tuning)과 Claude 3.5 Sonnet, Mistral 7B, GPT-4o와 같은 대규모 모델에 대한 프롬프트(prompting)를 포함합니다. 데이터 빈소화는 구성된 엔티티를 제거하여 수행됩니다. 다양한 작업에서 성능 비교가 이루어졌습니다.

- **Performance Highlights**: 감정 분석(sentiment analysis) 및 함의(entailment)와 같은 작업에서는 빈소화의 영향이 1-5%로 낮은 반면, 이해도 질문 답변(comprehension Q&A) 작업에서는 25% 이상의 큰 성능 하락이 관찰되었습니다. 논문에서는 빈소화된 데이터셋의 활용 방안도 제안하고 있습니다.



### Sentiment Analysis of Cyberbullying Data in Social Media (https://arxiv.org/abs/2411.05958)
- **What's New**: 이 연구는 사이버 괴롭힘 탐지를 위한 감정 분석에 있어 두 가지 새로운 하이브리드 방법을 소개합니다. 기존의 기술 대신에 최신 임베딩을 활용하고, 특히 사이버 괴롭힘 데이터에 대한 RNN 프레임워크와 BERT, OpenAI 임베딩을 결합한 접근방법을 채택했습니다.

- **Technical Details**: 연구에서는 LSTM(Long Short-Term Memory) 셀을 사용하는 순환 신경망을 개발하여, 감정 분석에 대한 BERT 임베딩 및 OpenAI의 최신 임베딩 API를 비교합니다. 이 기술들은 자연어 처리(NLP)와 기계 학습(ML) 영역에서 고품질의 주석 데이터가 충분히 확보되어야 성능 평가가 가능하다는 점을 강조합니다.

- **Performance Highlights**: Formspring 사이버 괴롭힘 데이터를 사용하여 두 가지 접근 방식을 효과적으로 비교하며, 최신 LLM(Large Language Model)을 사용한 방법이 과거 방법들보다 더 높은 정확도로 사이버 괴롭힘을 탐지하는 데 기여하고 있음을 보여줍니다.



### NeKo: Toward Post Recognition Generative Correction Large Language Models with Task-Oriented Experts (https://arxiv.org/abs/2411.05945)
Comments:
          NeKo work has been done in June 2024. NeKo LMs will be open source on this https URL under the MIT license

- **What's New**: NeKo는 일반 포스트 인식 오류 수정 모델을 위한 새로운 접근방식으로, Mixture-of-Experts (MoE) 아키텍처를 활용하여 다양한 도메인 데이터셋에서 효율적으로 학습합니다. 이 모델은 서로 다른 작업을 위해 전문가를 훈련시켜 각 데이터셋 토큰을 해당 전문가에게 라우팅하는 방식으로 작동합니다.

- **Technical Details**: NeKo는 다양한 오류 수정 데이터셋의 혼합에 대해 사전 훈련된 MoE 모델을 기반으로 하며, 각 전문가는 특정 도메인에 특화됩니다. 이 과제 지향적인 MoE 세부 조정 방식은 각 전문가가 작업 특정 특징을 포착하게 하며, 지식 공유가 가능합니다. 이를 통해 네트워크는 음성 인식, 번역 및 OCR 오타 수정 등의 작업에서 높은 성능을 발휘합니다.

- **Performance Highlights**: NeKo는 Open ASR Leaderboard 및 Hyporadise 벤치마크에서 기존 모델들 대비 평균 상대 WER을 5.0% 감소시켰으며, 0-shot 평가에서 GPT-3.5와 Claude-Opus보다 15.5%에서 27.6%까지 WER 감소를 달성했습니다. 이 결과로 NeKo는 멀티태스크 모델에서 경쟁력을 입증하였습니다.



### BERTrend: Neural Topic Modeling for Emerging Trends Detection (https://arxiv.org/abs/2411.05930)
Comments:
          17 pages, 12 figures, FuturED 2024: Workshop on Future of Event Detection (CoLocated with EMNLP 2024)

- **What's New**: BERTrend는 온라인 학습 설정에서 신경 주제 모델링을 활용하여 대규모 텍스트 자료의 신흥 트렌드와 약한 신호를 감지 및 추적할 수 있는 새로운 방법입니다. 이는 문서 수와 업데이트 빈도를 고려하여 시간에 따른 주제 인기도를 정량화하는 새로운 메트릭을 도입함으로써 그동안의 한계를 극복합니다.

- **Technical Details**: BERTrend는 HDBSCAN 알고리즘을 사용하여 자동으로 주제의 개수를 결정합니다. 또한 긴 문서를 단락으로 나눈 뒤 각 단락을 개별 문서로 취급하여 인기도 계산의 정확성을 높이고, 실시간으로 새로운 데이터가 들어올 경우 동적으로 주제를 추적할 수 있는 기능을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, BERTrend는 두 개의 대규모 실제 데이터셋에서 유의미한 약한 신호를 정확하게 탐지하고 잡음을 필터링하는 능력을 보여주었습니다. 이는 대규모 텍스트 자료에서의 신흥 트렌드 모니터링에 종합적인 솔루션을 제공합니다.



### Reducing Distraction in Long-Context Language Models by Focused Learning (https://arxiv.org/abs/2411.05928)
- **What's New**: 이 논문에서는 Long Context LLMs의 산만함(distractedness) 문제를 해결하기 위해 새로운 훈련 방법을 제안합니다. 이 방법은 검색 기반 데이터 증강(retrieval-based data augmentation)과 대비 학습(contrastive learning)을 조합하여 LLM이 관련 정보를 더 잘 인식할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소를 포함합니다: 1) 검색 기반 데이터 증강: 각 예제에 대해 질문과 관련된 상위 k개의 세그먼트만 유지하여 증강된 입력을 생성합니다. 2) 대비 학습: 원본 맥락과 검색 증강 샘플의 출력이 밀접하게 정렬되도록 보장하는 보조 대비 학습 목표를 도입합니다. 이 방법은 Mistral-7B 모델을 사용하여 효율적으로 미세 조정(low-rank adaptation, LoRA)합니다.

- **Performance Highlights**: 다양한 긴 단일 문서 및 다중 문서 QA 벤치마크에서 제안된 방법의 효과성을 입증하였으며, 훈련 단계를 몇 백 단계만 거치고도 산만함으로 인한 오류를 크게 줄여 표준训练 방법과 검색 증강 추론 기술을 초월하는 성능을 보여주었습니다.



### Humans Continue to Outperform Large Language Models in Complex Clinical Decision-Making: A Study with Medical Calculators (https://arxiv.org/abs/2411.05897)
- **What's New**: 대형 언어 모델(LLMs)이 임상 의사 결정 지원에서 의료 계산기 추천 능력을 평가한 최초의 연구입니다.

- **Technical Details**: 8개의 LLM(오픈 소스, 독점, 도메인 특정 모델 포함)을 평가하였으며, 1,009개의 질문-답변 쌍과 35개의 임상 계산기를 활용하였습니다. 인간 성능은 100개의 질문 하에서 측정되었습니다.

- **Performance Highlights**: 최고 성능의 LLM인 GPT-4o는 74.3%의 답변 정확도를 달성하였지만, 인간 주석자들은 평균 79.5%로 LLMs를 초월하였습니다. 오류 분석 결과 LLM의 컴프리헨션(이해) 오류율이 56.6%, 계산기 지식 오류율이 8.1%로 나타났습니다.



### One Small and One Large for Document-level Event Argument Extraction (https://arxiv.org/abs/2411.05895)
- **What's New**: 이번 논문에서는 Document-level Event Argument Extraction (EAE)의 두 가지 주요 문제를 해결하기 위해 Co and Structure Event Argument Extraction 모델(CsEAE)과 대형 언어 모델(LLMs)에 적합한 새로운 프롬프트 방식을 제안했습니다. 이 두 가지 접근법은 입력 길이가 증가함에 따라 발생하는 의미적 경계 식별의 어려움과 산재한 정보에서 오는 간섭을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: CsEAE는 Small Language Models (SLMs) 기반으로 하며, 두 개의 주요 모듈을 포함합니다: 1) co-occurrences-aware 모듈로, 모든 이벤트의 공존 정보와 관련 프롬프트를 함께 고려하도록 설계되었습니다. 2) structure-aware 모듈은 밀접한 문장 간의 관계를 구축하여 중복 정보로 인한 간섭을 감소시킵니다. 또한, LLMs를 위한 새로운 프롬프트를 설계하고, 감독 하에 미세 조정(Supervised Fine-Tuning, SFT)을 통해 성능을 향상시켰습니다.

- **Performance Highlights**: CsEAE 모델은 Rams, WikiEvents, MLEE 데이터셋에서 기존 PAIE 모델보다 각각 2.1%, 2.3%, 3.2%의 Arg-C F1 지표 개선을 달성했습니다. LLMs는 문서 수준 데이터셋에서 SLMs와 유사한 성능을 보였습니다. 이러한 결과는 SLMs에 유효했던 신뢰할 수 있는 인사이트가 LLMs에도 적용 가능함을 시사합니다.



### SSSD: Simply-Scalable Speculative Decoding (https://arxiv.org/abs/2411.05894)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 연구에서는 Speculative Decoding(가설적 디코딩) 기법이 대규모 언어 모델(LLM) 추론을 가속화하는 데 효과적으로 사용될 수 있는 방법을 제시합니다. 여기서는 기존 시스템에 추가적인 훈련 없이 통합 가능한 방법을 소개하며, 짧은 컨텍스트 생성 시 처리량을 4배 증가시키고 지연 시간에는 영향을 미치지 않는 성과를 달성했습니다.

- **Technical Details**: 이 연구는 대규모 배치 크기에서 Speculative Decoding이 어떻게 효과적으로 적용될 수 있는지를 이론적으로 설명합니다. 전처리 단계에서 KV-cache를 생성하고 첫 번째 토큰을 생성하는 패러렐 처리와, 모델이 반복적으로 새로운 토큰을 생성하는 오토 리그레시브 디코딩 단계로 나뉩니다. 이 연구는 특히 대규모 LLM 배포에서 사용되는 복잡한 디코딩 과정을 최적화하는 새로운 기법을 제안합니다.

- **Performance Highlights**: 이 방법은 짧은 컨텍스트 생성에서 처리량을 4배 증가시키고, 긴 컨텍스트의 경우 지연 시간과 처리량이 각각 1.7배에서 2배 향상되는 성과를 보였습니다. 다양한 사용사례에서 효과적으로 활용될 수 있도록 설계되었습니다.



### Identifying and Decomposing Compound Ingredients in Meal Plans Using Large Language Models (https://arxiv.org/abs/2411.05892)
Comments:
          Comments: Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)

- **What's New**: 이 연구는 식단 계획에서 대형 언어 모델(Large Language Models, LLMs)의 효과를 탐구하며, 복합 재료의 식별 및 분해 능력에 중점을 둡니다.

- **Technical Details**: 세 가지 모델(GPT-4o, Llama-3 (70b), Mixtral (8x7b)의 성능을 평가했습니다. 초기 결과에 따르면 Llama-3 (70b)와 GPT-4o는 정확한 분해에 뛰어난 성능을 보이지만, 모든 모델이 양념 및 오일과 같은 필수 요소를 식별하는 데 어려움을 겪습니다.

- **Performance Highlights**: 강력한 전반적인 성능에도 불구하고, 모델 간 정확성과 완전성에서 차이가 관찰되었습니다. 이는 LLM이 개인 맞춤형 영양을 향상시킬 가능성을 강조하지만, 재료 분해에서의 추가 수정이 필요함을 나타냅니다.



### Dialectal Coverage And Generalization in Arabic Speech Recognition (https://arxiv.org/abs/2411.05872)
- **What's New**: 이 연구는 아랍어의 방언 다양성을 관리하는 강력한 자동 음성 인식(ASR) 시스템 개발을 위한 중요한 요소들을 탐구합니다. 세 가지 주요 요소는 방언적 범위의 사전 훈련에서의 역할, 방언 특정 미세 조정(dialect-specific fine-tuning)과 다중 방언 접근 방식(multi-dialectal approach)의 효과성 비교, 그리고 보지 못한 방언에 대한 일반화 능력입니다.

- **Technical Details**: 이 연구는 ArTST 모델을 기반으로 하여 방언 다양성을 통합하여 ASR 성능을 최적화하는 것을 목표로 합니다. 방언 분석을 위해 12개 아랍어 방언을 대상으로 실험을 진행했으며, 사전 훈련(pre-training) 단계에서 다양한 방언을 포함하는 것이 성능에 미치는 영향을 조사했습니다. 실험 결과는 사전 훈련의 데이터 양과 방언적 범위가 대부분 방언 변종, 특히 현대 표준 아랍어(MSA)에서 성능을 향상시키는 데 기여함을 보여줍니다.

- **Performance Highlights**: 연구 결과는 다음과 같습니다: (1) 더 많은 데이터와 더 넓은 방언 범위로 사전 훈련하면 대부분의 방언 변종에서 성능을 향상시킵니다. (2) 다중 방언 미세 조정은 저자원(low-resource) 방언의 성능을 개선하지만, 고자원(high-resource) 방언에는 최적이 아닙니다. (3) 다중 방언 사전 훈련 및 미세 조정은 보지 못한 방언에 대한 제로샷 전이(zero-shot transfer)의 잠재력이 더 높습니다.



### Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy (https://arxiv.org/abs/2411.05819)
Comments:
          20 Pages

- **What's New**: 이번 연구는 기존의 증오 발언 (hate speech) 탐지 방법의 한계를 극복하고자, 감정 분석 (sentiment analysis)과 공유된 감정 표현 (shared emotional representations)을 통합한 새로운 멀티태스크 (multitask) 모델을 제안합니다.

- **Technical Details**: 본 연구에서는 딥 러닝 (deep learning)과 머신 러닝 (machine learning) 기반의 증오 발언 텍스트 분류 시스템 모델을 제시합니다. Hugging Face의 Transformer 기반 모델을 활용하여 감정 분석을 통해 잘못된 긍정 (false positives)을 최소화하는 방법을 모색하였습니다.

- **Performance Highlights**: 여러 데이터셋에서의 실험 결과, 감정 분석과 Transformer 기반의 훈련된 모델을 활용함으로써 증오 발언 탐지의 정확성이 상당히 향상되었음을 확인하였습니다.



### TempCharBERT: Keystroke Dynamics for Continuous Access Control Based on Pre-trained Language Models (https://arxiv.org/abs/2411.07224)
Comments:
          Accepted at WIFS 2024

- **What's New**: 이번 논문에서는 사용자 식별 및 인증을 위한 새로운 아키텍처인 TempCharBERT를 제안합니다. 이 모델은 기존의 언어 모델(PLMs)을 기반으로 하며, 키 입력 동적 정보(keystroke dynamics, KD)에 맞추어 커스터마이징되었습니다.

- **Technical Details**: TempCharBERT는 CharBERT 아키텍처의 임베딩 레이어에 시간적-문자 정보(temporal-character information)를 통합하여 구현되었습니다. 이는 사용자의 키 입력 패턴을 정확히 모델링할 수 있도록 돕고, 개별 사용자를 식별 및 인증하기 위한 기능을 개선합니다. 특히, 이 모델은 사용자의 입력시에 발생하는 딜타임(dwell time)과 플라이트 타임(flight time)을 고려합니다.

- **Performance Highlights**: TempCharBERT는 사용자 식별 및 인증 작업에서 CharBERT 및 다른 기초 모델들과 비교하여 상당한 정확도 향상을 보여주었습니다. 또한, 이 모델은 연합 학습(federated learning) 환경에서 훈련될 수 있는 가능성을 보여주었으며, 개인 데이터의 프라이버시를 보호하는 데도 효과적임을 입증했습니다.



### Stronger Models are NOT Stronger Teachers for Instruction Tuning (https://arxiv.org/abs/2411.07133)
- **What's New**: 본 논문은 기존의 큰 모델이 더 강력한 교사가 될 것이라는 가정에 도전합니다. 저자들은 여러 실험을 통해 더 큰 또는 강력한 모델이 반드시 더 작은 모델에 대해 더 효과적인 교육자가 아니라는 사실을 밝혔습니다. 따라서 'Larger Models' Paradox'라는 현상을 제안합니다.

- **Technical Details**: 이 연구에서는 Compatibility-Adjusted Reward (CAR)라는 새로운 메트릭을 개발하여 다양한 응답 생성기의 효과성을 측정합니다. CAR는 호환성을 위험 요소로 삼아 응답의 평균 손실로 호환성을 정량화하며, 기존의 메트릭이 간과하는 효과를 고려합니다. 다섯 가지 기본 모델을 사용하여 CAR의 성능을 평가한 결과, 모든 기준선 모델을 능가했습니다.

- **Performance Highlights**: 응답 생성기 선택에서 기존 메트릭이 효과를 예측하는 데 실패한다는 사실이 드러났습니다. 더 큰 모델의 사용이 항상 더 나은 결과를 가져오지 않으며, 오픈 소스 모델들이 GPT-4보다 더 높은 성능을 보였다는 점이 인상적입니다. 저자들은 응답 생성기를 선택할 때 기존의 벤치마크 성능에 의존하기 보다는 호환성이 더 높은 모델을 우선시할 필요성을 강조합니다.



### Universal Response and Emergence of Induction in LLMs (https://arxiv.org/abs/2411.07071)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)에서 유도(induction) 행동의 출현을 조사하여, 모델의 잔여 스트림(residual stream)에 대한 약한 단일 토큰의 변화를 탐색합니다. 이를 통해 유도 신호의 양적 특성을 제공하여 LLM의 동작을 더 잘 이해할 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 Gemma-2-2B, Llama-3.2-3B, GPT-2-XL 모델에서 잔여 스트림에 대한 약한 변화를 통해 유도 행동의 신호를 탐지하였습니다. 저희는 이 모델들이 변화 강도에 관계없이 반응이 비례(scale-invariant)하게 유지되는 보편적인 영역을 갖고 있음을 발견했습니다. 이 방법은 사용자가 모델의 각 레이어에서 유도 행동의 구성 요소를 더 깊이 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 모델의 중간 레이어에서 유도 신호가 점진적으로 나타나면서, 유도 행동을 구성하는 중요 모델 섹션들을 식별할 수 있었습니다. 이 발견은 대규모 회로 분석(large-scale circuit analysis)을 위한 기준이 될 수 있으며, LLM의 내부 상호작용에 대한 통찰력을 제공합니다.



### Zeroth-Order Adaptive Neuron Alignment Based Pruning without Re-Training (https://arxiv.org/abs/2411.07066)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 네트워크 프루닝(Network Pruning) 기술을 활용하여, LLM(대형 언어 모델)들에 대해 성능 저하 없이 매개변수를 줄일 수 있는 새로운 알고리즘 NeuroAl을 제안합니다. 이 알고리즘은 'top-up' 접근 방식을 사용해, 기존의 여러 프루닝 기법에 쉽게 적용될 수 있습니다.

- **Technical Details**: NeuroAl 알고리즘은 Activations의 일관성을 높이기 위해 블록 및 로우별 희소성 비율을 수정하는 두 단계의 방식으로 구성됩니다. 이 과정에서 사용자에게 하이퍼파라미터를 지정할 필요가 없으며, 모델의 구조와 입력된 희소성 요구에 따라 자동으로 최적의 값을 선택합니다. 이를 통해 프루닝 후에도 과거 프루닝 기법보다 나은 성능을 보장합니다.

- **Performance Highlights**: 4개의 서로 다른 LLM 계열에서 3가지 희소성 비율에 대해 테스트한 결과, NeuroAl은 최신 기술인 OWL 및 DsNoT보다 지속적으로 우수한 성능을 보였습니다. 실험 결과, 60%, 70%, 80%의 높은 희소성에서도 안정적인 성능을 발휘하며, 심도 있는 Ablation Study를 통해 알고리즘의 강건성을 입증했습니다.



### Minion: A Technology Probe for Resolving Value Conflicts through Expert-Driven and User-Driven Strategies in AI Companion Applications (https://arxiv.org/abs/2411.07042)
Comments:
          18 pages, 5 figures

- **What's New**: 이 연구는 AI 동반자와 사용자 간의 가치 충돌을 이해하고 해결하는 방법을 제안합니다. 연구팀은 151건의 사용자 불만을 분석하여 충돌의 디자인 시사점을 도출하고, Minion이라는 기술 프로브를 개발하여 사용자가 인간-AI 가치 충돌을 해결하는 데 도움을 주고자 하였습니다.

- **Technical Details**: Minion은 전문가 기반(expert-driven) 및 사용자 기반(user-driven) 충돌 해결 전략을 결합한 사용자 권한 부여 개입 방법을 적용합니다. 기술 프로브 연구에서는 40개의 가치 충돌 시나리오를 설정하고, 22명의 참가자가 274개의 작업을 수행하며 94.16%의 충돌 해결률을 기록했습니다.

- **Performance Highlights**: Minion은 참가자들에게 긍정적인 피드백을 받았고, 충돌 해결에 대한 새로운 아이디어를 제공하였습니다. 연구팀은 인간-AI 가치 충돌을 해결하는 데 있어 전문가 및 사용자 전략을 통합하는 기회와 도전을 논의하며, 향후 연구 필요성을 제기했습니다.



### Electroencephalogram-based Multi-class Decoding of Attended Speakers' Direction with Audio Spatial Spectrum (https://arxiv.org/abs/2411.06928)
- **What's New**: 본 논문은 청취자의 EEG 신호를 사용하여 주목하는 화자의 방향성을 보다 정밀하게 해독하는 방법을 제안합니다. 특히, 오디오 공간 정보(audio spatial information)를 활용하여 기존의 이진 방향성 해독을 넘어서 더 높은 정확도의 다중 클래스 방향성 해독을 가능하게 했습니다.

- **Technical Details**: 본 연구에서는 CNN, LSM-CNN, EEG-Deformer 모델을 사용하여 청취자의 EEG 신호와 보조 오디오 공간 스펙트라(auxiliary audio spatial spectra)를 통합하여 방향성 집중을 해독합니다. 특히, 최근 제안된 15-class(클래스) 방향성 집중 데이터셋에서 실험을 진행하였으며, leave-one-subject-out 및 leave-one-trial-out 시나리오에서 성능을 평가했습니다.

- **Performance Highlights**: 제안된 Sp-Aux-Deformer 모델은 leave-one-subject-out 시나리오에서 57.48%, leave-one-trial-out 시나리오에서 61.83%의 15-class(클래스) 해독 정확도를 달성하였습니다.



### EVQAScore: Efficient Video Question Answering Data Evaluation (https://arxiv.org/abs/2411.06908)
- **What's New**: 이번 연구에서는 비디오 질문-응답(Question-Answering, QA) 데이터 평가를 위한 새로운 방법인 EVQAScore를 소개합니다. 기존의 방법들이 비디오 캡션 평가에만 초점을 맞춰 결과적으로 비디오 QA 평가에 적합한 지표가 부족했던 문제를 해결하고자 합니다.

- **Technical Details**: EVQAScore는 키워드 추출(keyword extraction)과 프레임 샘플링(frame sampling) 기법을 활용하여 비디오 QA 및 캡션 데이터 품질을 평가합니다. 이를 통해 비디오의 긴 길이에도 효과적으로 작동할 수 있도록 하여 평가 비용을 30배 줄이는 성과를 거두었습니다. 또한, LLMs를 통해 데이터의 어휘적 의미를 거리 평가하는 전통적인 TF-IDF 방법보다 의미를 더 정확하게 이해할 수 있습니다.

- **Performance Highlights**: VATEX-EVAL 벤치마크에서 EVQAScore는 Kendall 상관관계 32.8, Spearman 상관관계 42.3을 기록하여 이전 방법인 PAC-S++보다 각각 4.7점, 5.9점 높은 성과를 보였습니다. 데이터 선택에 EVQAScore를 사용하여 원본 데이터의 12.5%만으로도 이전 SOTA 방법인 PAC-S 및 100% 데이터와 비교해 성과를 뛰어넘는 결과를 달성했습니다.



### Subgraph Retrieval Enhanced by Graph-Text Alignment for Commonsense Question Answering (https://arxiv.org/abs/2411.06866)
Comments:
          Accepted by ECML PKDD 2024

- **What's New**: 본 논문에서는 Commonsense Question Answering (CSQA) 작업을 위한 새로운 프레임워크인 SEPTA를 제안합니다. SEPTA는 Knowledge Graph (KG)를 효율적으로 활용하여 소프트웨어가 공통 상식에 기반한 논리적 사고를 수행할 수 있도록 설계되었습니다.

- **Technical Details**: SEPTA는 Knowledge Graph를 서브그래프 벡터 데이터베이스로 변환하고, BFS 스타일의 샘플링 전략을 사용하여 정보 손실을 최소화합니다. 또한, 그래프와 텍스트 인코더 사이의 의미 공간을 정렬하기 위한 양방향 대비 학습 접근법을 제안하여 정보 통합을 효과적으로 개선합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에 대한 대규모 실험 결과, 제안된 SEPTA 프레임워크가 기존의 최첨단 방법들보다 우수한 성능을 보였으며, 약한 감독 설정에서도 유망한 성과를 달성했습니다.



### LA4SR: illuminating the dark proteome with generative AI (https://arxiv.org/abs/2411.06798)
- **What's New**: 이 논문에서는 AI 언어 모델(LMs)이 미생물 서열 분류에 효과적으로 재설계되었음을 보여줍니다.

- **Technical Details**: 재설계된 오픈 소스 모델은 GPT-2, BLOOM, DistilRoBERTa, ELECTRA 및 Mamba로, 매개변수 수는 70M에서 12B까지 다양합니다. 이 모델들은 F1 점수 95를 기록하고, BLASTP보다 16,580배 빠르고 2.9배 높은 recall을 달성했습니다.

- **Performance Highlights**: 대형 모델(>1B) LA4SR은 사용 가능한 데이터의 2% 미만으로 훈련했음에도 불구하고 높은 정확성(F1 > 86)을 보였고, 완전한 Hi-C/Pacbio Chlamydomonas 게놈을 포함한 새로운 데이터로 검증되었습니다. 이는 결측 서열에서의 강력한 일반화 능력을 보여줍니다.



### Large-scale moral machine experiment on large language models (https://arxiv.org/abs/2411.06790)
Comments:
          20 pages, 6 figures

- **What's New**: 본 연구는 51개의 다양한 LLM을 평가하여 자율주행 시나리오에서 인간의 도덕적 선호와의 정합성을 조사합니다. 이전 연구에서는 한정된 LLM만을 분석했으나, 본 논문에서는 여러 버전의 전유 모델과 오픈소스 모델을 포함하여 보다 포괄적으로 다룹니다.

- **Technical Details**: 연구에서는 특별히 Moral Machine 실험 프레임워크를 활용하여 LLM이 윤리적 딜레마에 대한 응답을 평가했습니다. 분석 프레임워크로는 'conjoint analysis'를 사용했으며, 모델 크기, 업데이트, 아키텍처의 영향을 조사했습니다.

- **Performance Highlights**: 10억 개 이상의 파라미터를 가진 전유 모델과 오픈소스 모델은 인간의 판단과 비교적 근접한 정합성을 보였으며, 오픈소스 모델에서 모델 크기와 인간 판단의 거리는 상관관계가 부정적이었습니다. 그러나 모델 업데이트는 일관되게 인간의 선호도와의 정합성을 향상시키지 않았습니다.



### Model Fusion through Bayesian Optimization in Language Model Fine-Tuning (https://arxiv.org/abs/2411.06710)
- **What's New**: 본 논문은 Pretrained Language Models (plms)의 미세 조정과 관련된 문제를 해결하기 위해 Bayesian Optimization Model Fusion (bomf)이라는 새로운 기법을 소개합니다. 모델 퓨전은 여러 모델을 결합하는 방식으로, 성능을 개선하고 최적의 하이퍼파라미터 선택을 지원하는 혁신적인 접근법을 제공합니다.

- **Technical Details**: 논문에서는 Multi-Objective Bayesian Optimization (mobo)를 사용하여 손실 함수(loss function)와 성능 지표(metric)를 동시에 고려하여 모델을 퓨전하는 새로운 방법을 제안합니다. 또한, 하이퍼파라미터 선택을 위한 두 단계의 Bayesian 최적화 절차를 구축하여 모델 퓨전 과정을 효율적으로 수행합니다. 이를 통해 다양한 NLP(자연어 처리) 태스크에서 성능을 큰 폭으로 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: experiments conducted with various tasks, including Natural Language Understanding (nlu) and Natural Language Generation (nlg), demonstrate significant performance improvements using the proposed bomf method on models such as roberta, T5, and llama.



### Renaissance: Investigating the Pretraining of Vision-Language Encoders (https://arxiv.org/abs/2411.06657)
- **What's New**: 최근 비전-언어 (Vision-Language) 과제를 위한 모델들이 급격히 증가하며, 이와 관련된 모델 디자인 및 훈련의 모범 사례에 대한 질문들이 여전히 남아있습니다. 본 논문에서는 비전-언어 인코더의 사전 훈련에 관한 질문에 답하고, 두 개의 주요 실험을 통해 가시적인 성능 저하 없이 계산 비용을 크게 절감할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 'Renaissance'라는 비전-언어 모델링 플랫폼을 소개하며, 대규모 컴퓨팅 자원을 절약하기 위해 사전 훈련 중 모델의 여러 부분을 고정(freezing)하는 실험을 수행하였습니다. 첫 번째 실험에서는 두 개의 모듈이 모두 고정된 경우 일부 성능 저하가 발생했지만, 시각 모듈을 고정시켰을 때는 성능이 증가하는 경향이 나타났습니다. 두 번째 실험에서는 텍스트 인코더와 비전 인코더의 사전 훈련 성능을 비교하였습니다.

- **Performance Highlights**: 결과적으로, 한 개의 타워 인코더 모델을 훈련할 때는 사전 훈련된 가중치보다 무작위 초기화(randomly initialized)가 더 나은 성능을 보이는 것으로 나타났습니다. Renaissance 플랫폼은 다양한 비전-언어 모델 유형을 평가하는 데 유용하며, 향후 연구에서 보다 많은 VL 모델에 대한 지원이 필요합니다.



### Understanding Scaling Laws with Statistical and Approximation Theory for Transformer Neural Networks on Intrinsically Low-dimensional Data (https://arxiv.org/abs/2411.06646)
- **What's New**: 이 논문은 트랜스포머 기반의 대형 언어 모델에서 모델 크기와 데이터 크기에 따라 일반화 오류(generalization error)가 어떻게 파워 스케일링 법칙을 따른다는 이론을 제시합니다. 특히, 이론은 훈련 데이터가 저차원 매니폴드(low-dimensional manifold)에 집중될 때 적용됩니다.

- **Technical Details**: 저자는 새로운 통계적 추정(statistical estimation) 및 수학적 근사(mathematical approximation) 이론을 개발하여 트랜스포머 신경망의 일반화 오류와 모델/데이터 크기 간의 스케일링 법칙을 예측하고 정당화했습니다. 이 연구에서 일반화 오류는 훈련 데이터 크기와 네트워크 크기에 대해 전통적인 방식으로 설명할 수 있으며, 모델은 저차원 데이터 구조를 활용하여 데이터 기하학(data geometry)을 존중하는 방법으로 스케일링 법칙을 설명합니다.

- **Performance Highlights**: LLM을 자연어 데이터 세트에서 훈련한 결과, 관측된 경험적 데이터 스케일링 법칙은 이론적 예측과 밀접한 일치를 보였습니다. 이 결과는 이론과 실제 모두에서 트랜스포머 스케일링 법칙에 영향을 미치는 데이터의 내재적 차원(intrinsic dimension)이 중요한 수량임을 엄격하게 보여줍니다.



### Model Editing for LLMs4Code: How Far are We? (https://arxiv.org/abs/2411.06638)
Comments:
          Accepted by ICSE2025. The code is available at: this https URL

- **What's New**: 최신 연구는 LLMs4Code(대형 언어 모델의 코드 버전)의 지식 수정 방법인 모델 편집(Model Editing)에 초점을 맞추고 있으며, 사전 훈련된 모델의 부정확한 지식을 수정하는 일에 대한 체계적인 공부를 진행하고 있습니다.

- **Technical Details**: 이 연구에서는 CLMEEval이라는 벤치마크를 도입하며, CoNaLa-Edit와 CodeSearchNet-Edit 두 가지 데이터셋을 포함합니다. 각 데이터셋은 코드 생성과 코드 요약 작업을 평가하며, 6가지 최신 모델 편집 기법을 사용하여 세 가지 LLMs4Code 모델(CodeLlama, CodeQwen1.5, Stable-Code)에 대해 실험을 진행했습니다. 주요 평가 지표는 효과성(Effectiveness), 일반화(Generalization), 특이성(Specificity), 유창함(Fluency)입니다.

- **Performance Highlights**: GRACE 기법이 가장 높은 효과성과 특이성을 보였지만, 모든 기술이 일반화에서 낮은 성능을 보였습니다. 향상된 기법 A-GRACE를 통해 일반화 성능이 평균 80.86%로 크게 향상되었습니다. 이 결과는 기존의 편집 기법보다 구체적이고 유의미한 성과로, 모델 편집 기술의 발전에 기여할 것입니다.



### CriticAL: Critic Automation with Language Models (https://arxiv.org/abs/2411.06590)
- **What's New**: 이 논문에서는 CriticAL (Critic Automation with Language Models)을 제안하여 LLM(대형 언어 모델)의 활용을 통해 모델 비판(model criticism)을 자동화하는 새로운 접근 방식을 소개합니다. CriticAL은 모델 예측과 데이터 간의 불일치를 포착하는 summary statistics를 생성하고, 이들의 유의미성을 평가하는 가설 검정을 적용합니다.

- **Technical Details**: CriticAL은 LLM을 통해 모델과 데이터의 메타데이터를 기반으로 데이터의 성질을 포착하는 summary statistics를 생성하고, 이를 통해 모델의 가정이 위반되는지를 평가합니다. 이 통계량은 Python 함수로 구현되어 인간 또는 LLM 과학자가 쉽게 실행할 수 있도록 되어 있어 투명성과 신뢰성을 제공합니다. CriticAL의 summary statistics는 전통적인 가설 검정을 통해 불일치의 유의미성을 평가하여 모델을 자동으로 검증할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, CriticAL은 인위적으로 제어된 불일치에서 신뢰성 있는 비판을 생성하며, 인간 및 LLM 심사자 모두 CriticAL의 비판을 다른 접근 방법보다 투명성과 실행 가능성 측면에서 더 선호하는 것으로 나타났습니다. CriticAL은 실제 데이터셋에서 인간이 설계한 모델보다 개선된 성과를 보였습니다.



### In-Context Learning for Preserving Patient Privacy: A Framework for Synthesizing Realistic Patient Portal Messages (https://arxiv.org/abs/2411.06549)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 8 pages

- **What's New**: COVID-19 대유행 이후, 환자 포털 메시지가 증가하고 있는데, 이는 임상의사의 소진에 큰 기여를 하고 있습니다. 본 논문에서는 HIPAA 친화적이며 현실적인 환자 포털 메시지를 생성하기 위한 LLM 기반의 새로운 프레임워크인 PortalGen을 소개하고 있습니다.

- **Technical Details**: PortalGen은 두 단계로 나뉘는 구조를 가지고 있으며, 첫 번째 단계에서는 ICD-9 코드를 기반으로 환자 포털 메시지 프롬프트를 생성합니다. 두 번째 단계에서는 실제 환자 메시지 10개를 사용하여 LLM에 Grounded Generation을 적용하여 현실적인 환자 메시지를 생성합니다. 이 과정에서 HIPAA 준수를 유지하며 최소한의 인간의 비식별화 노력으로 데이터를 생성합니다.

- **Performance Highlights**: PortalGen은 기존의 데이터 생성 기법과 비교해 품질이 높으며, 실제 데이터와 매우 유사한 결과를 보였습니다. 전체 평가를 통해 PortalGen이 데이터의 스타일 및 의미의 충실성을 높이는 데 기여함을 입증했습니다.



### Probabilistic Consensus through Ensemble Validation: A Framework for LLM Reliability (https://arxiv.org/abs/2411.06535)
Comments:
          8 pages, 6 tables

- **What's New**: 대형 언어 모델(Large Language Models, LLMs)이 텍스트 생성 분야에서 큰 발전을 이루었으나, 의료, 법률, 금융과 같은 고위험 분야에서의 자율 배치에 필요한 신뢰성은 부족합니다. 기존 방법들은 외부 지식이나 인간의 감독에 의존하여 확장성에 한계가 있습니다. 이번 논문에서는 모델 합의를 통한 콘텐츠 검증을 위해 앙상블 방법(ensemble methods)을 재사용하는 새로운 프레임워크를 소개합니다.

- **Technical Details**: 78개의 복잡한 사례에서 사실 정확성(factual accuracy)과 인과 일관성(causal consistency)을 요구하는 테스트에서, 제안된 프레임워크는 두 개 모델을 사용할 경우 정확도를 73.1%에서 93.9%로 향상시켰고(95% CI: 83.5%-97.9%), 세 개 모델을 사용할 경우 95.6%로 향상시켰습니다(95% CI: 85.2%-98.8%). 통계 분석을 통해 강한 모델 간 합의(kappa > 0.76)를 보이며, 오류를 발견하기 위한 충분한 독립성도 유지합니다.

- **Performance Highlights**: 추가적인 검증자(validator)와 정교화를 통해 정확도를 더욱 향상시킬 수 있는 명확한 경로를 제시합니다. 현재 접근법은 다중 선택 형식 요구사항과 처리 지연(processing latency)으로 인해 제약이 있지만, 중요한 응용 분야에서 신뢰할 수 있는 자율 AI 시스템을 가능하게 하는 즉각적인 가치를 제공합니다.



### CTC-Assisted LLM-Based Contextual ASR (https://arxiv.org/abs/2411.06437)
Comments:
          SLT 2024

- **What's New**: 본 논문에서는 rare long-tail words(희귀 긴 꼬리 단어)를 인식하는 데 중점을 둔 CTC-Assisted LLM-Based Contextual ASR 모델을 제안합니다. 이 모델은 효율적인 필터링 알고리즘을 통해 기존 LLM 기반 ASR 모델의 한계를 극복합니다.

- **Technical Details**: CTC(Continuous Time Classification) 디코더와 fine-tuned SSL(선택적 자기 감시) 모델을 결합하여, 관련 hotword(핫워드)를 필터링하고 LLM(prompt input) 입력에 통합하는 방법을 사용합니다. WI-BER(Word Error Rate / Biased Word Error Rate) 측면에서 test-clean 세트에서 1.27% / 3.67%의 성능을 보이며, test-other 세트에서는 2.72% / 8.02%를 달성했습니다.

- **Performance Highlights**: 제안된 모델은 baseline LLM-ASR 모델과 비교하여 각각 39.81% / 63.37%, 35.24% / 61.37%의 성능 향상을 보여주며, 2000개의 biasing words(바이어스 단어)의 도움으로도 뛰어난 성능을 유지합니다.



### SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains (https://arxiv.org/abs/2411.06426)
- **What's New**: 본 연구에서는 SequentialBreak라는 새로운 jailbreak 공격 기법을 소개합니다. 이 기법은 LLM이 단일 쿼리 내에서 여러 프롬프트를 처리할 때 특정 유해한 프롬프트에 집중하도록 유도하는 구조를 활용합니다. 다양한 시나리오를 통해 유익한 프롬프트들 사이에 유해한 프롬프트를 은폐하여 LLM이 유해한 반응을 생성하도록 유도할 수 있습니다.

- **Technical Details**: SequentialBreak 공격은 단일 쿼리로 여러 프롬프트를 전송하며, 공격자는 benign 프롬프트 사이에 있던 유해한 프롬프트를 포함합니다. 이 공격은 black-box 접근만 필요하며, 다양한 프롬프트 서사 구조에 적응 가능합니다. 연구에서는 'Question Bank', 'Dialog Completion', 'Game Environment'라는 세 가지 공격 시나리오를 제시하며, 이 모두에서 고도의 공격 성공률을 보고합니다.

- **Performance Highlights**: SequentialBreak는 기존의 다양한 jailbreak 기법들보다 우수한 성능을 나타내었으며, 단일 쿼리만으로 여러 최신 LLM 모델에서 높은 공격 성공률을 달성했습니다. 또한, 기존 방어 메커니즘을 효과적으로 피할 수 있는 능력을 입증하며, 자원 효율성 또한 높습니다.



### Ablation is Not Enough to Emulate DPO: How Neuron Dynamics Drive Toxicity Reduction (https://arxiv.org/abs/2411.06424)
- **What's New**: 이 연구는 안전성 미세 조정 알고리즘의 메커니즘, 특히 직접 선호 최적화(Direct Preference Optimization, DPO)를 통한 독성 감소 과정을 탐구하며, 기존의 설명이 불완전하다고 주장합니다.

- **Technical Details**: DPO 알고리즘은 가장 독성이 강한 MLP 뉴런의 활성화를 억제하여 독성 영역을 피하는 것이 아니라, 다수의 뉴런 그룹 간의 효과를 축적하여 독성을 감소시킵니다. 실험을 통해 DPO의 독성 감소 효과가 단지 억제된 독성 뉴런이 아닌, 반독성(anti-toxicity)을 촉진하는 방식으로도 기여함을 밝혔습니다. DPO의 적용으로 인해 상당수의 뉴런들이 오히려 독성을 증가시키는 경우도 관찰되었습니다.

- **Performance Highlights**: DPO 적용 후 오직 31.8%의 독성 감소가 가장 독성이 강한 뉴런에서 기인하고 있으며, 나머지는 여러 뉴런 그룹의 상호 작용 결과입니다. 이 연구 결과는 안전성 미세 조정 알고리즘이 독성 감소를 이루기 위해서는 뉴런 간의 복잡한 균형 작용이 필요함을 시사합니다.



### CausalStock: Deep End-to-end Causal Discovery for News-driven Stock Movement Prediction (https://arxiv.org/abs/2411.06391)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 업계 최초로 뉴스 기반의 다중 주식 움직임 예측을 위한 "CausalStock"이라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 주식 간의 시간적 인과 관계를 발견하며, 노이즈가 포함된 뉴스 데이터에서 유용한 정보를 효과적으로 추출합니다.

- **Technical Details**: CausalStock 모델은 라그 의존성(lag-dependent) 템포럴 인과 발견 메커니즘을 적용하여 주식 간의 인과 그래프 분포를 모델링합니다. 또한, 대형 언어 모델(LLMs)의 텍스트 평가 능력을 활용한 Denoised News Encoder를 통해 노이즈가 많은 뉴스 텍스트에서 유용한 정보를 추출합니다. Functional Causal Model(FCM)을 사용하여 발견된 인과 관계를 캡슐화하고 주식의 움직임을 예측합니다.

- **Performance Highlights**: CausalStock은 미국, 중국, 일본, 영국 시장에서 수집한 6개의 실제 데이터세트에서 뉴스 기반의 다중 주식 움직임 예측 및 다중 주식 움직임 예측 작업 모두에서 강력한 기준선(baseline)을 초과하는 성능을 보였습니다. 인과 관계를 활용하여 CausalStock은 명확한 예측 메커니즘과 뛰어난 설명 가능성을 제공합니다.



### Self-Training Meets Consistency: Improving LLMs' Reasoning With Consistency-Driven Rationale Evaluation (https://arxiv.org/abs/2411.06387)
Comments:
          under review

- **What's New**: 본 연구에서는 CREST(Consistency-driven Rationale Evaluation for Self-Training)라는 새로운 자기 훈련 프레임워크를 제안합니다. 이 프레임워크는 각 논리를 후속 질문을 통해 평가하고 이를 활용하여 모델을 훈련하는 방법을 포함하고 있습니다.

- **Technical Details**: CREST는 두 가지 방법을 사용하여 훈련을 수행합니다: (1) 후속 질문에서 자주 잘못된 정답을 도출하는 논리를 걸러내는 rational filtering과 (2) 원본 및 후속 질문의 평가 결과를 기반으로 혼합된 선호도를 학습하는 preference learning입니다.

- **Performance Highlights**: 세 가지 질문-응답 데이터셋에서 실험을 수행한 결과, CREST는 논리의 강건성과 정확성을 향상시켰으며, 이전 자기 훈련 접근법보다 더 나은 추론 능력을 보여주었습니다.



### StopHC: A Harmful Content Detection and Mitigation Architecture for Social Media Platforms (https://arxiv.org/abs/2411.06138)
- **What's New**: 이 논문에서는 소셜 미디어 플랫폼을 위한 유해 콘텐츠 감지 및 완화 아키텍처인 StopHC를 제안합니다. StopHC는 딥 뉴럴 네트워크를 사용한 유해 콘텐츠 감지 모듈과 네트워크 면역화 알고리즘을 이용한 유해 콘텐츠 확산 방지 모듈로 구성되어 있습니다.

- **Technical Details**: StopHC는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 유해 콘텐츠 감지 모듈과 (2) 네트워크 면역화 모듈. 감지 모듈은 텍스트 데이터를 사용하여 콘텐츠의 유해성을 예측하고 사용자를 식별하며, 면역화 모듈은 사용자 상호작용을 기반으로 그래프를 생성하여 유해 콘텐츠의 확산을 저지합니다. 다양한 임베딩 기법(Word2Vec, GloVe, BERT, RoBERTa, Node2Vec)을 사용하여 데이터의 의미적 관계를 감지하고, 3가지 면역화 전략(naïve, pro-active, contra-active)을 통해 네트워크 내 유해 요소를 제어합니다.

- **Performance Highlights**: StopHC의 효과성은 두 개의 실제 데이터 세트(Hate Speech Dataset, EXIST2023 Dataset)를 이용한 실험을 통해 검증되었습니다. 이 솔루션은 유해 콘텐츠 감지 정확도와 함께 특정 유해 콘텐츠의 확산을 효과적으로 차단하는 능력을 보여주었습니다.



### Optimizing Large Language Models through Quantization: A Comparative Analysis of PTQ and QAT Techniques (https://arxiv.org/abs/2411.06084)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문은 Large Language Models (LLMs)의 최적화를 위한 양자화(Quantization) 기법들에 대한 포괄적인 분석을 제공합니다. 특히 Post-Training Quantization (PTQ)와 Quantization-Aware Training (QAT)에 중점을 두었습니다.

- **Technical Details**: 실험적으로 10M에서 1B 파라미터 범위의 모델에 대해 양자화를 적용한 결과, 모델 크기를 최대 68% 줄이면서도 전체 정밀도 기준선과의 성능 차이를 6% 이내로 유지할 수 있다는 것을 보여주었습니다. INT8 양자화는 연산 비용과 전력 소비를 40% 줄였고, INT4 양자화는 이러한 수치를 60% 더 개선했습니다. 혼합 정밀도 양자화를 위한 새로운 이론적 프레임워크를 도입하고, 레이어 민감도와 가중치 분산에 기반한 최적 비트 할당 전략을 도출했습니다.

- **Performance Highlights**: 엣지 장치에서의 하드웨어 효율성 평가 결과, INT8은 최대 2.4배의 처리량 향상을, INT4는 3배의 향상을 가능하게 하였으며, 전체 정밀도 모델에 비해 60% 전력 소비 감소를 보여주었습니다.



### Game-theoretic LLM: Agent Workflow for Negotiation Games (https://arxiv.org/abs/2411.05990)
Comments:
          44 pages, 12 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 전략적 결정 과정에서 이성을 발휘하는 수준을 게임 이론의 틀 안에서 조사합니다. 특히, 불완전 정보 게임 상황에서 LLMs의 비이성적인 전략 선택 경향을 분석하고 이를 극복하기 위한 게임 이론적 워크플로우를 개발하였습니다.

- **Technical Details**: 여러 최신 LLM들(예: Claude-3.5 Sonnet, GPT-4o 등)의 성능을 평가하였으며, 이는 완전 정보 및 불완전 정보 게임(예: Prisoner’s Dilemma, Battle of the Sexes 등)을 기반으로 합니다. 또한, 지배적 전략 탐색(Dominant Strategy Search), 후방 유도(Backward Induction), 베이지안 신념 업데이트(Bayesian belief updating)와 같은 게임 이론 원리를 활용하여 LLM의 이성적 행동과 결정 능력을 향상시키는 알고리즘을 설계하였습니다.

- **Performance Highlights**: 제안된 워크플로우를 통해 LLM들이 최적 전략을 식별하는 데 유의미한 개선을 보였으며, 협상 시 near-optimal allocations를 달성하고, 협상 과정에서의 착취(Exploitation)에 대한 저항력도 증가하였습니다. 연구 결과는 복잡한 상호작용 환경에서 전략적으로 더 견고하고 합리적인 AI 에이전트를 개발하는 데 기여할 수 있습니다.



### Toward Transdisciplinary Approaches to Audio Deepfake Discernmen (https://arxiv.org/abs/2411.05969)
- **What's New**: 이 논문은 다양한 학문 분야의 학자들이 음성 딥페이크(Audio Deepfake) 탐지 및 판별 문제를 AI(Artificial Intelligence) 방법론과 언어학(Linguistics)의 융합적 관점에서 접근해야 한다고 주장합니다.

- **Technical Details**: 현재 AI 모델들은 언어의 본질적인 변동성(variability)과 인간 음성의 복잡성(complexity) 및 독창성(uniqueness)을 충분히 이해하지 못하고 있으며, 이는 음성 딥페이크 탐지를 어렵게 하는 요인입니다. 최근 언어적 지식을 AI 접근법에 통합하는 학제간(transdisciplinary) 연구에서 기대되는 가능성을 탐구합니다.

- **Performance Highlights**: 전문가가 참여하는 방식(expert-in-the-loop)을 도입하여, 전문가에 대한 의존도가 낮은 AI 기반 방법을 넘어 더 견고하고 포괄적인 딥페이크 탐지 방법론으로 나아갈 수 있는 경로를 제공합니다.



### Quantifying artificial intelligence through algebraic generalization (https://arxiv.org/abs/2411.05943)
- **What's New**: 이번 논문에서는 알gebraic circuit complexity(대수 회로 복잡도) 이론을 도입해 기호적 일반화(symbolic generalization)를 명확히 정량화하는 새로운 프레임워크를 제시합니다. 현재의 인공지능(AI) 시스템은 기호 처리 및 추상화에 있어 한계를 보이고 있으며, 이 문제를 해결하기 위한 방법론을 수립하는 것에 초점을 두고 있습니다.

- **Technical Details**: 기호적 계산(symbolic computation)의 복잡성을 연구하기 위해 대수 회로 복잡도 이론을 사용합니다. 이 이론은 수학적 표현을 회로 모델(즉, 방향성 비순환 그래프)로 정식화하며, 각 회로의 크기(size)와 깊이(depth)를 주요 복잡성 척도로 설정합니다. 이를 통해 AI 모델의 일반화 성능을 객관적으로 평가할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 대수 회로 복잡도 이론을 통해 다양한 기호 문제를 다룰 수 있으며, 이는 AI 시스템의 약점 및 실패 양상을 구체적으로 파악하는 데 유리합니다. 알gebraic circuits는 그 본질상 수많은 표본(sample)을 생성할 수 있어 오늘날 데이터 중심의 기계 학습 알고리즘을 위한 최적의 테스트베드(testbed)가 됩니다.



### Mitigating Hallucination with ZeroG: An Advanced Knowledge Management Engin (https://arxiv.org/abs/2411.05936)
Comments:
          10 pages, 4 figures, 1 table

- **What's New**: ZeroG는 Knowledge Distillation과 Prompt Tuning을 활용하여 LLM의 응답 질을 크게 향상시킵니다. 이 접근법은 기존의 복잡한 문서 처리에서 발생하는 환각(hallucinations) 문제를 해결하여 보다 정확하고 신뢰할 수 있는 반응을 제공합니다.

- **Technical Details**: ZeroG는 블랙 박스 디스틸레이션 접근 방식을 통해 더 작은 모델이 더 큰 교사 모델의 동작을 복제하도록 설계되었습니다. 이 방법은 문서 소재를 Markdown 형식으로 변환하고, Neo4j와 같은 그래프 데이터베이스를 사용하여 메타데이터를 관리하며, RAG(Retrieval Augmented Generation) 접근법을 통해 문서 처리를 최적화합니다.

- **Performance Highlights**: MMR(Maximal Marginal Relevance) 검색 기법을 통해 응답의 정확성과 관련성을 높였습니다. 실험 결과, MMR을 사용하는 시스템은 코사인 유사성(cosine similarity) 방법에 비해 최대 12%의 정확도 향상이 있었습니다.



### Towards Multi-Modal Mastery: A 4.5B Parameter Truly Multi-Modal Small Language Mod (https://arxiv.org/abs/2411.05903)
- **What's New**: 새로운 4.5B 파라미터의 소형 언어 모델이 소개되었습니다. 이 모델은 텍스트, 이미지, 비디오, 오디오 등 다양한 입력 및 출력 모달리티(modality)를 처리할 수 있습니다.

- **Technical Details**: 이 모델은 언어 모델링(language modeling)과 다중 작업 학습(multi-task learning)의 최근 발전을 활용하여 만들어졌으며, 간편하게 엣지 추론(edge inference)에 배포될 수 있습니다.

- **Performance Highlights**: 모델은 여러 벤치마크에서 뛰어난 성능을 보여주며, 복잡한 현실 세계 문제를 해결할 수 있는 다중 모달(multi-modal) 인공지능의 가능성을 시사합니다.



### Autoregressive Models in Vision: A Survey (https://arxiv.org/abs/2411.05902)
- **What's New**: 자기회귀 모델(Autoregressive models)은 자연어 처리(NLP) 분야에서 큰 성공을 거두었다. 최근에는 컴퓨터 비전 분야에서도 이러한 모델이 주목받고 있으며, 고급 비주얼 콘텐츠를 생성하는 데 탁월한 성능을 발휘하고 있다. 이 서베이는 비전에 적용된 자기회귀 모델에 대한 포괄적인 문헌 검토를 다룬다. 또한, 제안된 모델은 이미지 생성, 비디오 생성, 3D 생성, 다중 모달 생성 등 다양한 분야에서 응용된다.

- **Technical Details**: 비전에서의 자기회귀 모델은 여러 레벨의 표현 전략을 기반으로 하는데, 픽셀 수준(pixel-level), 토큰 수준(token-level), 스케일 수준(scale-level)으로 나뉜다. 각각의 모델은 독특한 장점과 도전을 가지고 있으며, 비전에서의 자기회귀 모델의 범주는 크게 픽셀 기반(pixel-based), 토큰 기반(token-based), 스케일 기반(scale-based) 모델로 나뉘어 구체적으로 분석된다.

- **Performance Highlights**: 자기회귀 모델들은 이미지 생성, 비디오 생성, 및 의료 AI와 같은 다양한 응용 분야에서 비약적인 발전을 이루었으며, 특히 GAN, Diffusion, MAE 기반 방법과의 성능 비교를 통해 그 우수성을 강조한다. 학계에서는 이러한 모델들 간의 상호 연관성을 탐구하며 향후 연구 방향을 제안하고 있다.



### When are 1.58 bits enough? A Bottom-up Exploration of BitNet Quantization (https://arxiv.org/abs/2411.05882)
Comments:
          10 pages, 2 tables, 6 figures

- **What's New**: 본 논문은 1.58-bit 양자화(quantization) 기반의 훈련 방법을 조사하며, 비변환기(non-transformer) 모델 아키텍처와 변환기(transformer) 기반 모델을 포함한 다양한 네트워크에서의 성능을 비교 분석합니다.

- **Technical Details**: 1.58-bit 양자화 방법론은 Linear layers를 대체하는 BitLinear 레이어를 도입하여, 훈련 중 16-bit 그림자 가중치(shadow weights)를 유지하고, 순전파(forward pass) 시에는 이 가중치들을 양자화하여 사용합니다. 이는 최종적으로는 ternary weights(세 가지 값 -1, 0, 1)으로 전환됩니다.

- **Performance Highlights**: 모든 실험에서 1.58-bit 훈련 방법이 표준 32/16-bit 모델과 동등하거나 더 나은 성능을 보여주었으며, 특히 encoder-only 및 encoder-decoder 모델에서의 성능 향상이 두드러졌습니다. 특정 네트워크에서는 16-bit 모델을 초월하는 성능을 발휘했습니다.



### Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass (https://arxiv.org/abs/2411.05877)
- **What's New**: 이번 연구는 새로운 컨텍스트에 대한 효과적이고 효율적인 적응 방법인 GenerativeAdapter를 소개합니다. 이 방법은 사전 학습된 언어 모델을 미세 조정(fine-tuning) 없이 저랭크(low-rank) LM 어댑터에 직접 매핑하여 추론 오버헤드를 크게 감소시킵니다.

- **Technical Details**: GenerativeAdapter는 자가 지도 학습(self-supervised learning)을 통해 훈련된 어댑터 생성기(adapter generator)를 사용합니다. 이 어댑터 생성기는 고정된 LM을 단일 온-더 플라이(on-the-fly)로 적응시키며, 그러한 컨텍스트를 새로운 어댑터에 매핑합니다. 본 연구에서는 Mistral-7B-Instruct와 Llama2-7B-Chat이라는 두 개의 사전 학습된 LM에 GenerativeAdapter를 적용했습니다.

- **Performance Highlights**: StreamingQA에서 우리의 접근 방식은 LM의 파라미터에 지식을 주입하는 데 효과적이며, 감독 미세 조정(supervised fine-tuning)이 있는 모델 대비 F1 점수에서 63.5%의 향상(19.5에서 31.5로)이라는 성과를 달성했습니다. 다양한 적응 시나리오에서 평균 정확도 44.9%를 기록하여 기본 모델을 초월했습니다.



### Towards Improved Preference Optimization Pipeline: from Data Generation to Budget-Controlled Regularization (https://arxiv.org/abs/2411.05875)
Comments:
          15 pages

- **What's New**: 최근 Direct Preference Optimization (DPO) 및 그 변형들이 큰 언어 모델(LLM)의 조정에 있어 주요 방법이 되고 있습니다. 이 연구에서는 DPO의 선호 데이터 생성 및 교육 정규화 기술을 개선하는 방법을 제안합니다.

- **Technical Details**: 우리는 반복 쌍별 순위 메커니즘을 도입하여 선호 데이터의 품질을 향상시키고, 예측된 선호 샘플의 우선 확률을 약간 감소시키는 새로운 예산 관리 정규화 기법을 사용합니다. 이는 LLM의 예측 정확성을 유지하면서도 최적화 과정의 안정성을 가져옵니다.

- **Performance Highlights**: 이 연구에서 제안한 방법들은 두 가지 일반 벤치마크 평가를 통해 기존 SOTA를 능가하는 결과를 보였으며, 현업에서의 LLM 적용에 있어 높은 품질의 선호 데이터 생성을 통해 더 나은 최적화를 이루었습니다.



### LEGO-GraphRAG: Modularizing Graph-based Retrieval-Augmented Generation for Design Space Exploration (https://arxiv.org/abs/2411.05844)
- **What's New**: GraphRAG은 Retrieval-Augmented Generation (RAG)에서의 도전을 해결하기 위해 지식을 내장한 그래프를 활용하고 있으며, LEGO-GraphRAG라는 모듈형 프레임워크를 제안합니다. 이 프레임워크는 그래프 기반 지식 검색 프로세스를 세 가지 모듈로 분해합니다: 서브그래프 추출(subgraph-extraction), 경로 필터링(path-filtering), 경로 정제(path-refinement).

- **Technical Details**: LEGO-GraphRAG는 그래프 구조를 기반으로 한 알고리즘과 신경망 모델(Neural Network, NN)을 체계적으로 요약하고 분류하였습니다. 또한, Graph Coupling과 Computational Cost와 같은 주요 설계 요소를 식별하여 GraphRAG 구현의 효과성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험 연구를 통해 고품질 GraphRAG 인스턴스를 구축하고 그 결과가 검색 및 추론 성능에 미치는 영향을 분석하였습니다. 연구 결과는 보다 정확하고 맥락적으로 관련성이 높은 LLM 애플리케이션을 위한 GraphRAG 인스턴스 설계 최적화에 중요한 통찰력을 제공합니다.



### CDR: Customizable Density Ratios of Strong-over-weak LLMs for Preference Annotation (https://arxiv.org/abs/2411.02481)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 선호도 튜닝을 위한 새로운 방법인 사용자 정의 밀도 비율(CDR)을 소개합니다. 기존의 방법과 달리 CDR은 훈련이 필요 없으며, 소프트웨어 라이센스 제한이 없어 상업적인 사용에 제약이 없습니다.

- **Technical Details**: CDR은 두 개의 LLM 간의 로그 밀도 비율을 사용하여 선호도 데이터를 주석(annotation) 처리합니다. 논문에서는 221개의 다양한 LLM 쌍을 탐색하고, 성능 격차가 클수록 더 나은 보상 일반화와 상관관계가 있음을 실증적으로 보여줍니다. 모델 쌍의 조정 격차를 활용하여 특정 도메인에 맞게 조정된 밀도 비율 보상 기능을 제공하였습니다.

- **Performance Highlights**: Mistral-7B 모델 쌍을 이용하여 CDR을 적용한 결과, Llama-3-8B-Instruct는 ArenaHard에서 승률 37.4%(+15.1%) 및 Length-Controlled AlpacaEval 2.0에서 승률 40.7%(+17.8%)를 기록하며 뛰어난 성능을 보였습니다. CDR은 Safety(91.0) 및 Reasoning(88.0) 도메인에서 최첨단 모델과 경쟁력 있는 성과를 달성했습니다.



New uploads on arXiv(cs.IR)

### Invar-RAG: Invariant LLM-aligned Retrieval for Better Generation (https://arxiv.org/abs/2411.07021)
- **What's New**: 이 논문에서는 Invar-RAG라는 혁신적인 이단계 미세 조정 아키텍처를 제안합니다. 이 아키텍처는 LLM 기반 리트리버의 기능적 국부성(Feature Locality) 문제를 해결하고 LLM 변형(Retrieval Variance) 문제를 완화하는 새로운 접근 방식을 사용합니다.

- **Technical Details**: Invar-RAG는 두 단계로 구성됩니다. 첫 번째 단계에서는 LoRA 기반(LoRA-based) 표현 학습을 통합하여 기능적 국부성 문제를 해결한 LLM 기반 리트리버를 구축합니다. 두 번째 단계에서는 수집된 정보에 기반하여 LLM이 정확한 답변을 생성할 수 있도록 향상된 미세 조정 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, Invar-RAG는 세 가지 공개 ODQA 데이터셋에서 기존의 기준 모델들보다 현저히 우수한 성능을 보였습니다.



### LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help? (https://arxiv.org/abs/2411.06877)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)과 수동 주석(annotation)을 결합하여 테스트 컬렉션을 더욱 신뢰할 수 있고 효율적으로 구축하기 위한 LARA(LLM-Assisted Relevance Assessments) 방법을 제안합니다.

- **Technical Details**: LARA는 LLM의 예측된 관련성을 활용하여 수동 주석의 우선 순위를 정하고, 예측된 확률을 통해 불확실한 문서를 선택합니다. 이를 통해 인공지능과 인간의 주석 과정을 상호 보완적으로 최적화합니다.

- **Performance Highlights**: LARA는 TREC-COVID 및 TREC-8 Ad Hoc 데이터셋에서 거의 모든 예산 제약 하에서도 다른 대안을 초월하는 성능을 보였습니다.



### Boosting the Targeted Transferability of Adversarial Examples via Salient Region & Weighted Feature Drop (https://arxiv.org/abs/2411.06784)
Comments:
          9 pages

- **What's New**: 본 논문에서는 Salient region & Weighted Feature Drop (SWFD)라는 새로운 프레임워크를 도입하여 적대적 예제의 타겟 전이 가능성을 향상시키는 방법을 제안합니다. 기존의 방법들이 대체 모델에 과적합되는 경향이 있었다면, 본 연구는 이 문제를 해결하고자 합니다.

- **Technical Details**: SWFD는 깊은 레이어의 출력 분포가 부드러워질수록 전이 가능성이 향상된다는 관찰을 기반으로 합니다. 이 메커니즘은 피쳐의 활성화 값을 정규 분포에 의해 조절하여 예제가 특정 피쳐에 과도하게 의존하지 않도록 합니다.

- **Performance Highlights**: 제안된 SWFD 방법은 정상 훈련 모델과 강건 모델에 대해 각각 16.31% 및 7.06%의 공격 성공률 상승을 보여주며 다양한 환경에서 최신 기술들보다 우수한 성능을 입증하였습니다.



### Generating Mixcode Popular Songs with Artificial Intelligence: Concepts, Plans, and Speculations (https://arxiv.org/abs/2411.06420)
Comments:
          Link to the paper:this https URL Published in The International Conference on AI and Musical Creativity at the University of Oxford (2024) this https URL

- **What's New**: 이 논문은 인공지능(Artificial Intelligence)과 대중 음악(Popular Music)의 통합을 다루며, 사회 변혁(Social Transformation), 교육(Education), 건강 관리(Healthcare), 감정적 웰빙(Emotional Well-Being)을 위한 강력한 도구를 만들기 위한 프로젝트를 제안합니다.

- **Technical Details**: 이 연구는 컴퓨터 과학자(Computer Scientist), 데이터 분석가(Data Analyst)와 민속 음악학자(Ethnomusicologist), 사회 인류학자(Social Anthropologist) 간의 협업의 출발점에서 제시되고 있으며, 주로 개념적(Conceptual)이고 다소 투기적(Speculative)입니다.

- **Performance Highlights**: 이 프로젝트의 결과는 음악이 사회적 목적을 위해 어떻게 효과적으로 활용될 수 있는지를 탐구하며, 사회, 정치, 경제적 맥락에서 음악의 역할을 재조명할 수 있는 기회를 제공합니다.



### Metric Learning for Tag Recommendation: Tackling Data Sparsity and Cold Start Issues (https://arxiv.org/abs/2411.06374)
- **What's New**: 이 논문에서는 개인화된 추천 시스템의 한계를 극복하기 위해 메트릭 학습(metric learning)에 기반한 새로운 레이블 추천 알고리즘을 제안합니다. 이는 사용자의 선호와 아이템 특성 간의 미세한 차이를 포착하는 데 효과적인 거리 또는 유사성 메트릭을 학습합니다.

- **Technical Details**: 제안된 알고리즘은 전통적인 협업 필터링(collaborative filtering) 및 콘텐츠 기반 추천(content-based recommendation) 방법이 가진 데이터 희소성(data sparsity) 및 콜드 스타트(cold start) 문제를 해결하는 데 중점을 둡니다. 실험 결과에 따르면, 이 알고리즘은 지역 반응 메트릭 학습(local response metric learning, LRML), 협업 메트릭 학습(collaborative metric learning, CML), 적응형 텐서 분해(adaptive tensor factorization, ATF)와 같은 기존 방법보다 우수한 성능을 보입니다.

- **Performance Highlights**: 특히 제안된 알고리즘은 추천 항목의 초기 몇 개에서 높은 정확도를 달성하며, 강력한 내구성(robustness)을 유지하면서도 높은 추천 정확도를 지속적으로 보여줍니다.



### Annotative Indexing (https://arxiv.org/abs/2411.06256)
- **What's New**: 이 논문은 annotative indexing이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 전통적인 inverted indexes, column stores, object stores, graph databases를 통합 및 일반화합니다. 이를 통해 knowledge graphs, entity retrieval, semi-structured data 및 ranked retrieval을 지원하는 데이터베이스를 위한 기본 인덱싱 프레임워크를 제공합니다.

- **Technical Details**: annotative indexing은 텍스트 형태의 인간 언어 데이터를 주로 다루지만, 숫자와 날짜를 포함한 다양한 데이터 유형을 지원할 수 있는 충분히 일반적입니다. 예를 들어, SQL과 유사한 쿼리를 JSON 저장소에 대해 실행할 수 있습니다. 이 인덱스는 ACID 트랜잭션 속성을 지원하며, 동시에 수백 명의 읽기 및 쓰기 작업자를 지원하는 동적 annotative index를 시연합니다.

- **Performance Highlights**: annotative indexing은 기존의 sparsely retrieval 방법들의 한계를 극복하고, 효율적이고 효과적인 첫 단계 검색을 넘어 사람 언어 데이터 관리의 복잡성을 해결합니다. 특히, BM25와 같은 기존의 순위 알고리즘을 활용 유지하면서 JSON 저장소에서 인덱싱된 애너테이션을 통해 콘텐츠 및 상관 정보에 대한 효율적인 검색과 변환을 지원합니다.



### KeyB2: Selecting Key Blocks is Also Important for Long Document Ranking with Large Language Models (https://arxiv.org/abs/2411.06254)
- **What's New**: 본 논문에서는 LLMs (Large Language Models) 기반의 정보를 검색하는 KeyB2라는 새로운 접근 방식을 제안하며, 이는 기존의 KeyB 전략을 개선하여 긴 문서에 대한 검색 효율성을 높입니다.

- **Technical Details**: KeyB2는 블록 프리-랭킹(block pre-ranking) 전략을 통합하여, 특정 attention heads가 관련 토큰을 정렬하는 데 중요한 역할을 한다는 점을 분석하였습니다. 이를 통해 KeyB2는 관련 블록을 효율적으로 식별하고 처리하여 계산 비용을 줄이며 랭킹 효과성을 개선합니다. 새로운 bi-encoder 블록 매칭 전략도 도입되어, KeyB2는 정보 검색의 효율성과 이해력을 동시에 향상시킵니다.

- **Performance Highlights**: TREC 2019 DL, Robust04, MLDR-zh와 같은 긴 문서 데이터 세트에서의 광범위한 실험에서 KeyB2는 RankLLaMA 및 KeyB와 같은 기존 기준 모델 대비 재랭킹 시간과 GPU 메모리 사용량을 줄이고 검색 성능을 향상시켜 새로운 SOTA 결과를 달성했습니다.



### Leveraging Retrieval-Augmented Generation for University Knowledge Retrieva (https://arxiv.org/abs/2411.06237)
Comments:
          6 pages, 2 figures, 1 table, Submitted to 15th IKT conference

- **What's New**: 이 논문은 대학교 관련 질문 응답 시스템을 향상시키기 위한 Retrieval-Augmented Generation (RAG) 파이프라인을 사용하여 정보 검색을 혁신적으로 접근하는 방법을 소개합니다.

- **Technical Details**: 논문에서 제안한 RAG 파이프라인은 두 단계의 접근법을 이용하여 페르시아 대형 언어 모델(PLM)과 정교한 프롬프트 엔지니어링 기법을 결합합니다. 먼저, 쿼리를 분류하여 가장 관련 있는 문서를 찾아내고, 그 후 적절한 LLM을 사용하여 정확하고 맥락적으로 관련 있는 응답을 생성합니다. 이 연구는 'UniversityQuestionBench'(UQB)라는 포괄적인 벤치마크 데이터를 개발하여 RAG 파이프라인의 성능을 엄격하게 평가하였습니다.

- **Performance Highlights**: 실험 결과, 생성된 응답의 정확성과 관련성이 크게 향상되어 사용자 경험이 증진되고 관련 답변을 얻기 위한 소요 시간이 줄어들었습니다.



### Interpret the Internal States of Recommendation Model with Sparse Autoencoder (https://arxiv.org/abs/2411.06112)
- **What's New**: 이번 논문에서는 추천 시스템(Recommendation systems)의 내부 상태를 해석하는 자동화된 일반화 가능한 방법인 RecSAE를 제안합니다. RecSAE는 기존 모델의 결과에 영향을 주지 않으면서도 해석 결과를 기반으로 시스템 동작을 예측 가능하게 조정할 수 있는 플러그인 모듈로 기능합니다.

- **Technical Details**: RecSAE는 Sparsity 제약(subtyping constraints)을 가진 오토인코더(autoencoder)를 사용하여 추천 모델의 내부 활성화(internal activations)를 재구성합니다. 이는 RecSAE의 잠재(latent) 벡터가 원래 신경망(neuron) 활성화보다 더 해석 가능하고 단일 의미(monosemantic)을 가질 수 있도록 돕습니다. 또한 잠재 활성화와 입력 항목 시퀀스(input item sequences) 간의 관계를 기반으로 개념 사전을 자동으로 생성합니다.

- **Performance Highlights**: RecSAE의 유효성은 두 개의 데이터셋에서 입증되었으며, ID 기반 모델에서 수백 개의 매우 해석 가능한 개념들을 식별하였습니다. 잠재 소거(latent ablation) 연구를 통해 잠재 개념을 조작하면 모델 출력 동작(output behavior)에 상응하는 변화가 발생함을 확인하며, 추천 모델의 이해 및 목표 조정을 위한 RecSAE의 유용성을 강조합니다.



### Snippet-based Conversational Recommender System (https://arxiv.org/abs/2411.06064)
- **What's New**: 이번 연구에서는 사용자 생성 콘텐츠(UGC)에서 다양한 표현과 선호를 추출하여 대화를 강화하고 개인화된 추천을 제공하는 새로운 대화형 추천 시스템(CRS)인 SnipRec을 소개합니다.

- **Technical Details**: SnipRec은 대형 언어 모델을 사용하여 사용자 응답 및 UGC를 간결한 스니펫(snippet)으로 매핑합니다. 이러한 스니펫을 활용하여 명확화 질문(clarification questions)을 생성하고 관련 항목을 검색합니다. 이 접근법은 특정 도메인에 대한 훈련이 필요 없어 새로운 도메인에 쉽게 적응할 수 있습니다.

- **Performance Highlights**: Yelp 데이터셋에서의 광범위한 실험을 통해 SnipRec의 스니펫 기반 표현이 문서 및 문장 기반 표현에 비해 효과적임을 입증하였으며, 5회 대화 전환(turns) 동안 Hits@10을 0.25 향상시켜 사용자의 선호를 효과적으로 포착할 수 있음을 보여주었습니다.



### Mitigating Hallucination with ZeroG: An Advanced Knowledge Management Engin (https://arxiv.org/abs/2411.05936)
Comments:
          10 pages, 4 figures, 1 table

- **What's New**: ZeroG는 Knowledge Distillation과 Prompt Tuning을 활용하여 LLM의 응답 질을 크게 향상시킵니다. 이 접근법은 기존의 복잡한 문서 처리에서 발생하는 환각(hallucinations) 문제를 해결하여 보다 정확하고 신뢰할 수 있는 반응을 제공합니다.

- **Technical Details**: ZeroG는 블랙 박스 디스틸레이션 접근 방식을 통해 더 작은 모델이 더 큰 교사 모델의 동작을 복제하도록 설계되었습니다. 이 방법은 문서 소재를 Markdown 형식으로 변환하고, Neo4j와 같은 그래프 데이터베이스를 사용하여 메타데이터를 관리하며, RAG(Retrieval Augmented Generation) 접근법을 통해 문서 처리를 최적화합니다.

- **Performance Highlights**: MMR(Maximal Marginal Relevance) 검색 기법을 통해 응답의 정확성과 관련성을 높였습니다. 실험 결과, MMR을 사용하는 시스템은 코사인 유사성(cosine similarity) 방법에 비해 최대 12%의 정확도 향상이 있었습니다.



### The Shapley index for music streaming platforms (https://arxiv.org/abs/2411.07166)
- **What's New**: 이번 연구에서는 음악 스트리밍 플랫폼에서 아티스트의 인기 측정을 위한 새로운 지수인 Shapley index를 제안하고, 이 지수가 아티스트들에게 유료 구독을 통해 조달된 수익을 분배하는 모델로 자리 잡을 수 있는 가능성을 분석합니다.

- **Technical Details**: Shapley index는 협력 게임 이론(cooperative game theory)에서 Shapley value를 기반으로 하며, 아티스트들의 공동체를 형성하여 일정한 총 수익을 아티스트별로 어떻게 배분할지를 분석합니다. 이 지수는 user-centric 방식과 유사하지만, 각 사용자별로 아티스트에게 지급되는 금액을 같은 수준으로 분산시킵니다.

- **Performance Highlights**: Shapley index는 기존의 pro-rata와 user-centric 방식보다 공정한 수익 배분을 제공하며, 음악 산업의 수익 분배 문제에 대한 새로운 통찰을 제공합니다. 이 지수는 아티스트의 스트리밍 수익을 보다 공평하게 분배할 수 있는 잠재력을 가지고 있습니다.



### Adaptive Conditional Expert Selection Network for Multi-domain Recommendation (https://arxiv.org/abs/2411.06826)
- **What's New**: 최근 Multi-domain recommendation (MDR)에서 Mixture-of-Experts (MOE) 기반 접근 방법의 대안으로 CESAA 모델이 제안되었습니다. 이 모델은 Conditional Expert Selection (CES)와 Adaptive Expert Aggregation (AEA) 모듈로 구성되어 있어 성능 저하와 확장성 문제를 해결합니다.

- **Technical Details**: CES 모듈은 희소한 게이팅 전략과 도메인 공유 전문가를 결합하여 특정 도메인에 적합하도록 전문가를 선택합니다. AEA 모듈은 상호 정보 손실(mutual information loss)을 활용하여 전문가와 특정 도메인 간의 상관관계를 강화하고, 각 인스턴스에 대해 도메인 공유 및 선택된 도메인 전문 도시기자만 활성화하여 효율성을 높입니다. 이러한 구조는 손쉬운 end-to-end 학습을 가능하게 하여 사전 정의된 도메인 구분 없이 최적의 패턴을 찾아냅니다.

- **Performance Highlights**: 공공 랭킹 및 산업 리트리벌 데이터 세트를 통한 실험 결과, CESAA 모델은 최신 방법들과 비교하여 더 우수한 성능을 보이며 MDR 작업에서 높은 효과성을 입증했습니다.



### Large Language Model in Medical Informatics: Direct Classification and Enhanced Text Representations for Automatic ICD Coding (https://arxiv.org/abs/2411.06823)
Comments:
          accepted at the 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

- **What's New**: LLAMA-2 모델을 활용하여 ICD 코드 분류 자동화의 효과성을 높이는 두 가지 방법론을 제시합니다. 하나는 직접적인 분류기로서의 LLAMA-2 활용이며, 다른 하나는 Multi-Filter Residual Convolutional Neural Network (MultiResCNN)에서 풍부한 텍스트 표현 생성입니다.

- **Technical Details**: LLAMA-2 모델을 직접적인 ICD 코드 분류자로써 활용하고, MultiResCNN을 통한 텍스트 표현 생성을 위해 학습합니다. 이 네트워크는 Deep Learning과 Natural Language Processing의 통합을 기반으로 하고, 복잡한 의학적 문서를 효과적으로 해석하기 위한 구성 요소를 가지고 있습니다. 세분화된 ICD 코드 공간을 다루기 위해 특수한 분류 헤드를 포함시킵니다. 또한, RoPE (Rotatory Position Embedding) 및 YaRN (Yet another RoPE extension method) 기법을 통해 긴 임상 텍스트를 효율적으로 처리합니다.

- **Performance Highlights**: MIMIC-III 데이터셋에서 LLAMA-2의 적용 결과, ICD 코드 분류 정확도가 현저히 향상됨을 보였습니다. LLAMA-2를 활용한 두 가지 접근법 모두 기존 최첨단 방법들과 비교하여 뛰어난 성과를 나타냈습니다.



### AssistRAG: Boosting the Potential of Large Language Models with an Intelligent Information Assistan (https://arxiv.org/abs/2411.06805)
Comments:
          Accepted by NeurIPS 2024 (poster)

- **What's New**: 이번 논문에서는 기존의 RAG 방법론의 한계를 극복하기 위해 AssistRAG라는 새로운 접근 방식을 제안합니다. 이 방법론은 LLM 내부에 지능형 정보 어시스턴트를 통합하여 정보 검색 및 의사결정 능력을 향상시킵니다.

- **Technical Details**: AssistRAG는 두 가지 주요 기능인 메모리 관리와 지식 관리로 구성됩니다. 메모리 관리는 내부 메모리의 내용을 통합 및 분석하며, 지식 관리는 외부 지식을 활용하는 데 초점을 맞춥니다. 이를 위해 네 가지 핵심 기능인 Tool usage, Action execution, Memory building, Plan specification을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, AssistRAG는 기존의 벤치마크를 초월하는 성능을 보여주었으며, 특히 덜 발전된 LLM에 더 큰 이점을 제공합니다. 다양한 복잡한 질문 응답 데이터셋에서 우수한 추론 능력을 나타냈습니다.



### GuidelineGuard: An Agentic Framework for Medical Note Evaluation with Guideline Adherenc (https://arxiv.org/abs/2411.06264)
- **What's New**: 이 논문에서는 GuidelineGuard라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 기반으로 하여 의료 진료 노트를 자율적으로 분석하여 가이드라인 준수를 보장합니다.

- **Technical Details**: GuidelineGuard는 병원 퇴원 노트 및 사무실 방문 노트와 같은 의료 진료 노트를 분석합니다. 추천된 관행에서의 이탈을 식별하고, WHO 및 CDC와 같은 기관의 최신 기준을 준수하도록 증거 기반 제안을 제공합니다.

- **Performance Highlights**: 이 프레임워크는 문서 품질을 개선하고 임상 오류를 줄이는 데 도움을 주며, 의료 전문가들이 최신 가이드라인을 따르는 데 필요한 지원을 제공합니다.



### The effect of different feature selection methods on models created with XGBoos (https://arxiv.org/abs/2411.05937)
- **What's New**: 이번 연구는 XGBoost와 같은 인기 있는 머신러닝 알고리즘에서 다양한 피처 선택 방법이 모델 구축에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 세 가지 다른 차원 축소 방법이 모델의 예측 정확도에 통계적으로 유의미한 변화를 일으키지 않는다는 것을 보여줍니다. 이는 XGBoost에서 잡음이 많은 훈련 데이터를 제거하여 모델의 과적합(overfitting)을 방지하는 전통적인 아이디어가 적용되지 않을 수 있음을 시사합니다.

- **Performance Highlights**: 그럼에도 불구하고, 이러한 피처 선택 기법들은 계산 복잡도(computational complexity)를 줄이는 데 여전히 유효할 수 있습니다.



### BERTrend: Neural Topic Modeling for Emerging Trends Detection (https://arxiv.org/abs/2411.05930)
Comments:
          17 pages, 12 figures, FuturED 2024: Workshop on Future of Event Detection (CoLocated with EMNLP 2024)

- **What's New**: BERTrend는 온라인 학습 설정에서 신경 주제 모델링을 활용하여 대규모 텍스트 자료의 신흥 트렌드와 약한 신호를 감지 및 추적할 수 있는 새로운 방법입니다. 이는 문서 수와 업데이트 빈도를 고려하여 시간에 따른 주제 인기도를 정량화하는 새로운 메트릭을 도입함으로써 그동안의 한계를 극복합니다.

- **Technical Details**: BERTrend는 HDBSCAN 알고리즘을 사용하여 자동으로 주제의 개수를 결정합니다. 또한 긴 문서를 단락으로 나눈 뒤 각 단락을 개별 문서로 취급하여 인기도 계산의 정확성을 높이고, 실시간으로 새로운 데이터가 들어올 경우 동적으로 주제를 추적할 수 있는 기능을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, BERTrend는 두 개의 대규모 실제 데이터셋에서 유의미한 약한 신호를 정확하게 탐지하고 잡음을 필터링하는 능력을 보여주었습니다. 이는 대규모 텍스트 자료에서의 신흥 트렌드 모니터링에 종합적인 솔루션을 제공합니다.



### Identifying and Decomposing Compound Ingredients in Meal Plans Using Large Language Models (https://arxiv.org/abs/2411.05892)
Comments:
          Comments: Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)

- **What's New**: 이 연구는 식단 계획에서 대형 언어 모델(Large Language Models, LLMs)의 효과를 탐구하며, 복합 재료의 식별 및 분해 능력에 중점을 둡니다.

- **Technical Details**: 세 가지 모델(GPT-4o, Llama-3 (70b), Mixtral (8x7b)의 성능을 평가했습니다. 초기 결과에 따르면 Llama-3 (70b)와 GPT-4o는 정확한 분해에 뛰어난 성능을 보이지만, 모든 모델이 양념 및 오일과 같은 필수 요소를 식별하는 데 어려움을 겪습니다.

- **Performance Highlights**: 강력한 전반적인 성능에도 불구하고, 모델 간 정확성과 완전성에서 차이가 관찰되었습니다. 이는 LLM이 개인 맞춤형 영양을 향상시킬 가능성을 강조하지만, 재료 분해에서의 추가 수정이 필요함을 나타냅니다.



New uploads on arXiv(cs.CV)

### Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models (https://arxiv.org/abs/2411.07232)
Comments:
          Project page is at this https URL

- **What's New**: 이번 연구에서는 자연어 텍스트 지침에 따라 이미지를 편집하는 기존의 접근 방식의 한계를 극복하기 위해, Add-it이라는 새로운 방법을 제안합니다. 이 방법은 기존 장면을 유지하면서도 새로운 객체를 자연스럽게 통합하는 것에 중점을 둡니다.

- **Technical Details**: Add-it은 학습이 필요 없는 접근 방식을 통해 diffusion 모델의 attention 메커니즘을 확장하고, 장면 이미지, 텍스트 프롬프트 및 생성된 이미지를 포함한 세 가지 주요 정보를 결합합니다. 이 확장된 attention 메커니즘은 구조적 일관성을 유지하며 객체의 자연스러운 배치를 보장합니다.

- **Performance Highlights**: Add-it은 이전 방법들과 비교해 객체 삽입 작업에서 최첨단의 성능을 달성했습니다. 실제 이미지 및 생성된 이미지 삽입 벤치마크에서 뛰어난 결과를 보였으며, 인간 평가에서도 80% 이상의 선호도를 기록했습니다.



### Watermark Anything with Localized Messages (https://arxiv.org/abs/2411.07231)
Comments:
          Under review. Code at this https URL

- **What's New**: 본 논문에서는 새로운 심층 학습 모델인 Watermark Anything Model (WAM)을 소개합니다. WAM은 특히 작은 워터마크된 영역을 처리하는 데 최적화되어 있으며, 다양한 소스에서 온 이미지의 편집된 작은 부분을 처리하는 현실 세계의 적용에 적합합니다.

- **Technical Details**: WAM은 두 단계로 훈련됩니다: 첫 번째 단계는 저해상도 이미지에서 낮은 가시성을 유지하도록 Embedder와 Extractor 모델을 사전 훈련하는 것입니다. 두 번째 단계에서는 인간 시각 시스템과의 정렬을 통해 워터마크의 가시성을 최소화하고, 동일 이미지 안에서 여러 메시지를 허용하는 새로운 목표를 달성합니다. 추가적으로 DBSCAN 클러스터링 알고리즘을 사용하여 픽셀 레벨의 이진 문자열에서 워터마크 영역을 로컬라이즈합니다.

- **Performance Highlights**: WAM은 저해상도 및 고해상도 이미지 처리 시 섬세함과 강인성 면에서 기존의 최첨단 기법과 경쟁력을 갖추고 있습니다. 예를 들어, 5개의 32비트 메시지를 숨긴 경우, 워터마크된 영역의 감지 정확도는 85% 이상에 달하며, 비트 정확도는 95% 이상을 기록했습니다.



### Learning from Limited and Imperfect Data (https://arxiv.org/abs/2411.07229)
Comments:
          ICVGIP'24 Young Researcher Symposium Abstract

- **What's New**: 이번 논문에서는 제한적이며 불완전한 데이터를 학습하는 딥 뉴럴 네트워크 적합을 위해 여러 가지 알고리즘을 제안합니다. 이는 기존의 수작업 데이터 균형 유지 과정에서 발생하는 비용을 줄이고, 실제 세계의 데이터 분포에서 효과적으로 학습할 수 있도록 합니다.

- **Technical Details**: 이 논문은 Generative Models, Inductive Regularization Schemes, Semi-Supervised Learning, Efficient Domain Adaptation의 네 가지 범주로 나뉘어 있으며, 각 범주는 제한적 또는 불완전한 데이터로부터의 학습 시나리오를 다룹니다. 특히 Class Balancing GAN, Sharpness Aware Minimization 등의 기술이 소개됩니다.

- **Performance Highlights**: 제안된 방법들을 통해, 화상 생성 및 인식 작업에서 특정 클래스에서의 성능을 개선하여 State-of-the-Art(SotA) 성과를 달성했습니다. 특히 고해상도 StyleGAN을 사용하여 ImageNet-LT 및 iNaturalist2019 데이터셋에서 뛰어난 결과를 보였습니다.



### DLCR: A Generative Data Expansion Framework via Diffusion for Clothes-Changing Person Re-ID (https://arxiv.org/abs/2411.07205)
Comments:
          Published in WACV 2025

- **What's New**: 최근의 생성적 확산 모델(generative diffusion models)의 강력함을 통해 생성된 이미지로 더 나은 시각적 표현을 학습할 수 있는지가 주요 연구 질문으로 제기되었습니다. 본 연구에서는 의류 변경 사람 재식별(clothes-changing person re-identification, CC-ReID)이라는 어려운 작업에 이러한 생성적 데이터 확장을 적용하여 기존 데이터 세트의 의류 다양성을 10배 증가시키는 DLCR(data expansion framework)을 제안합니다.

- **Technical Details**: DLCR은 사전 훈련된 확산(diffusion) 모델과 대형 언어 모델(LLMs)을 활용하여 개인의 다양한 의상을 정확하게 생성합니다. 이 과정에서 DLCR은 텍스트 기반 지침에 따른 이미지 인페인팅(inpainting)을 사용하여 개인의 특징을 보존하면서 의류만 변경된 합성 데이터를 생성합니다.

- **Performance Highlights**: PRCC 데이터 세트에서 DLCR로 생성된 데이터를 활용하여 이전의 최고 성능(method CAL) 모델의 top-1 정확성을 11.3% 향상시키는 성과를 달성했습니다. 두 가지 새로운 전략인 점진적 학습(progressive learning)과 테스트 시 예측 정제(test-time prediction refinement)를 사용하여 훈련 시간 단축과 CC-ReID 성능을 더욱 향상시켰습니다.



### OmniEdit: Building Image Editing Generalist Models Through Specialist Supervision (https://arxiv.org/abs/2411.07199)
Comments:
          21 pages

- **What's New**: 이 논문은 Omni-Edit라는 새로운 이미지 편집 모델을 소개합니다. 이 모델은 주어진 사용자의 지시에 따라 7가지 다양한 이미지 편집 작업을 처리할 수 있는 강력한 능력을 갖추고 있습니다.

- **Technical Details**: Omni-Edit는 (1) 여러 전문 모델로부터의 감독을 통해 일반화된 편집 모델로 학습됩니다. (2) 소비자 데이터를 위한 중요 샘플링을 사용하여, 데이터 품질을 개선합니다. (3) EditNet이라는 새로운 아키텍처를 도입하여 편집 성공률을 비약적으로 증가시킵니다. (4) 다양한 종횡비와 높은 해상도의 이미지를 사용하여 훈련하여, 실제 환경에서의 활용성을 높입니다.

- **Performance Highlights**: 자동 평가와 인간 평가 모두에서 Omni-Edit는 기존의 모든 모델을 능가하는 성과를 보였습니다. 예를 들면, VIEScore와 같은 자동 메트릭에서 기존 접근 방식보다 유의미하게 높은 점수를 기록했으며, 인간 평가에서는 CosXL-Edit와 비교해 20% 개선된 결과를 보여주었습니다.



### SAMPart3D: Segment Any Part in 3D Objects (https://arxiv.org/abs/2411.07184)
Comments:
          Project Page: this https URL

- **What's New**: SAMPart3D는 사전 정의된 부분 레이블 세트가 없이도 임의의 3D 객체를 다중 세분화로 나누는 확장 가능한 제로샷 3D 부품 분할 프레임워크입니다.

- **Technical Details**: 이 프레임워크는 텍스트 비의존적인 DINOv2 모델을 사용하여 2D에서 3D로의 특징 증류를 수행하고, SAM에서 증류된 스케일 조건부 파트 인지 3D 특징을 사용하여 3D 부분 분할을 다중 세분화로 수행합니다.

- **Performance Highlights**: SAMPart3D는 기존 제로샷 3D 부품 분할 방법과 비교해 뛰어난 성능을 보이며, PartObjaverse-Tiny라는 새로운 3D 부품 분할 벤치마크를 통해 복잡한 3D 객체에 대한 주목할 만한 성과를 보여줍니다.



### Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection (https://arxiv.org/abs/2411.07167)
Comments:
          Accepted by WACV 2025. Supplementary material is included at the end of the main paper (3 pages, 5 figures, 2 tables)

- **What's New**: 이 논문에서는 Dual Vision Transformer (D-ViT)와 Long Skip Connections (LSC)를 기반으로 한 새로운 얼굴 랜드마크 탐지기를 소개합니다. 특히, 이 모델은 랜드마크 간의 고유한 기하학적 관계를 모델링하기 위해 Channel-split ViT를 이용하여 기초적인 선형 관계를 학습합니다.

- **Technical Details**: 이 연구에서는 특징 맵(feature maps)의 채널 차원이 열지도 공간(heatmap space)의 선형 기초를 나타낸다고 보고, 이를 통해 랜드마크 간의 내재적인 기하학적 관계를 모델링합니다. Dual Vision Transformer(D-ViT)는 표준 시각 변환기(spatial-split ViT)와 채널 분할 ViT(channel-split ViT)를 통합하여 예측 블록을 구성합니다. Long Skip Connections은 낮은 수준의 이미지 특징을 모든 예측 블록에 전달하여 유용한 정보를 잃지 않도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 WFLW, COFW, 300W와 같은 여러 데이터셋에서 이전의 최첨단 방법(SOTA)을 능가하였으며, 뛰어난 일반화 능력을 보여주었습니다. 모든 세 데이터셋에서 새로운 SOTA를 달성했습니다.



### Nuremberg Letterbooks: A Multi-Transcriptional Dataset of Early 15th Century Manuscripts for Document Analysis (https://arxiv.org/abs/2411.07138)
- **What's New**: 뉘른베르크 편지록(Nuremberg Letterbooks) 데이터셋은 15세기 초 역사적 문서로 구성되며, 다양한 형식의 전사(transcription)와 메타데이터(metadata)를 제공함으로써 인문학 연구의 요구에 더 부합하는 방법론을 개발할 수 있는 기반을 마련했습니다.

- **Technical Details**: 이 데이터셋은 10명의 필사자가 작성한 1711개의 라벨이 붙은 페이지를 포함하고 있으며, 기본(basic), 외교(diplomatic), 정규화(regularized)된 세 가지 전사형식을 제공합니다. 기술 검증에서는 여러 작업에서 기준값(baselines)을 확립하고 데이터의 일관성을 보여주었습니다.

- **Performance Highlights**: 핸드라이튼 텍스트 인식(Handwritten Text Recognition, HTR) 및 필자 식별(writer identification) 시스템을 통해 전사 성능을 검증하였으며, 책 4에서 가장 낮은 문자 오류율(Character Error Rate, CER)과 단어 오류율(Word Error Rate, WER)을 기록했습니다. 데이터셋의 다양성으로 인해 인문학 연구자들이 더욱 쉽게 디지털 텍스트에 접근할 수 있도록 지원합니다.



### Edify 3D: Scalable High-Quality 3D Asset Generation (https://arxiv.org/abs/2411.07135)
Comments:
          Project website: this https URL

- **What's New**: Edify 3D는 고품질 3D 자산 생성을 위한 고급 솔루션으로, 다수의 시점에서 RGB 및 표면 노멀 이미지를 합성하는 **diffusion model**을 활용합니다. 이 모델은 2분 이내로 신속한 3D 자산 생성을 가능하게 하며, 산업 표준에 맞는 해상도와 모델 품질을 보장합니다.

- **Technical Details**: Edify 3D는 **text-to-3D** 및 **image-to-3D** 생성 기능이 있으며, 두 가지 종류의 신경망인 **diffusion models**와 **Transformers**에 기반한 기술을 사용합니다. 다중 시점의 RGB 이미지와 표면 노멀을 합성한 후, 이를 사용하여 3D 형태와 재질을 예측하는 **reconstruction model**을 통해 기하학적 데이터와 텍스처 맵, 소재 맵을 생성합니다.

- **Performance Highlights**: Edify 3D는 이전의 text-to-3D 접근 방식보다 뛰어난 3D 형상과 텍스처를 일관되게 생성하며, 4K 해상도의 텍스처와 물리 기반 렌더링(PBR) 재료를 포함하여 뛰어난 품질의 3D 자산을 생성합니다. 이 과정은 효율성과 확장성이 크게 향상되었습니다.



### Token Merging for Training-Free Semantic Binding in Text-to-Image Synthesis (https://arxiv.org/abs/2411.07132)
Comments:
          Accepted by Neurips2024

- **What's New**: 본 연구에서는 텍스트-이미지(T2I) 모델의 의미적 바인딩 문제를 해결하기 위해 새로운 방법인 Token Merging(ToMe)를 제안합니다. 이 방법은 관련된 토큰을 하나의 복합 토큰으로 집계하여 모든 객체, 속성, 하위 객체가 동일한 교차 주의 맵을 공유하도록 합니다.

- **Technical Details**: ToMe는 객체 속성과 하위 객체 간의 적절한 연관성을 증진시키기 위해 설계되었습니다. 또한, 복합 토큰 업데이트를 위한 두 가지 보조 손실(엔트로피 손실과 의미적 바인딩 손실)을 통합하여 T2I 생성 초기 단계에서의 통합성을 높입니다. 이 방법은 T2I-CompBench와 GPT-4o 오브젝트 바인딩 벤치마크에서 객관적인 성능 평가를 받았습니다.

- **Performance Highlights**: ToMe 방법은 복잡한 다중 객체 및 속성 생성 시나리오에서 특히 효과적이며, 많은 기존 방법들을 능가하는 결과를 보여주었습니다. 사용자 친화적인 특성 덕분에 대규모 언어 모델이나 특정 레이아웃 정보에 의존할 필요가 없습니다.



### Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models (https://arxiv.org/abs/2411.07126)
- **What's New**: Edify Image라는 새로운 생성 모델이 도입되었습니다. 이 모델은 픽셀 완벽한 정확도를 가진 포토리얼리스틱 이미지 콘텐츠를 생성할 수 있는 여러 가지 확산 모델로 구성되어 있습니다.

- **Technical Details**: Edify Image는 다단계(cascaded) 픽셀-스페이스(diffusion models) 확산 모델을 활용하여, 이미지 신호를 다양한 주파수 대역에서 서로 다른 비율로 감쇠시키는 새로운 라플라시안(Laplacian) 확산 프로세스를 사용합니다. 이 모델은 고해상도 이미지를 생성하며, 텍스트-이미지 합성, 4K 업샘플링, ControlNets 및 360도 HDR 파노라마 생성과 같은 다양한 응용 프로그램을 지원합니다.

- **Performance Highlights**: Edify Image 모델은 긴 텍스트 프롬프트를 처리하며, 다양한 종횡비로 이미지를 생성하고, 인간 주제를 생성할 때 향상된 공정성과 다양성을 보장합니다. 또한, 주어진 낮은 해상도 이미지를 기반으로 높은 주파수 세부사항을 합성할 수 있는 4K 업샘플러 모델과 다양한 모달리티에 대한 ControlNets로 이미지를 생성할 수 있는 기능을 제공합니다.



### Decoding Visual Experience and Mapping Semantics through Whole-Brain Analysis Using fMRI Foundation Models (https://arxiv.org/abs/2411.07121)
- **What's New**: 이 논문에서는 시각 자극에 대한 뇌의 반응을 이해하기 위해 전체 뇌 활성화 맵을 사용하여 새로운 알고리즘을 개발했습니다. 이 방법은 기존의 시각 피질(decode visual stimuli onto the visual cortex)에서의 디코딩을 넘어서는 모델을 구축하여 더 복잡한 시각 경험을 이해하는 데 기여합니다.

- **Technical Details**: Whole-brain Analysis of Visual Experience (WAVE)라는 새로운 방법론을 제안하였으며, fMRI(functinal Magnetic Resonance Imaging) 인코더와 이미지 생성 모델을 결합하여 대규모 공개 데이터셋에서 사전 훈련 후 Image-fMRI contrastive learning을 통해 미세 조정합니다. 이를 통해 Visual Cortex를 넘어 Cerebral Cortex의 시각 경험을 디코딩할 수 있습니다.

- **Performance Highlights**: WAVE 모델은 시각 처리 디코딩에서 43%의 향상된 예측 의미 정확도를 보여주었으며, 전통적인 데이터를 사용하더라도 시각 피질을 제거한 후에도 높은 예측 정확도를 달성했습니다. 추가적인 제로샷 상상 디코딩에서 p-value 0.0206을 기록하며 다양한 시나리오에서 의미를 포착하는 능력을 입증했습니다.



### ConvMixFormer- A Resource-efficient Convolution Mixer for Transformer-based Dynamic Hand Gesture Recognition (https://arxiv.org/abs/2411.07118)
- **What's New**: 이 연구에서는 동적 손 제스처 인식을 위해 새로운 ConvMixFormer 아키텍처를 제안합니다. 기존의 transformer 모델의 self-attention 대신 간단한 CNN 기반의 token mixer를 사용하여 계산 복잡성을 줄이고, 더 적은 파라미터로 로컬 공간 특성을 캡처합니다.

- **Technical Details**: ConvMixFormer는 self-attention의 복잡한 계산을 피하고, convolution 레이어를 사용하여 입력 이미지의 spatial tokens를 혼합합니다. Gated Depthwise Feed Forward Network (GDFN)를 활용하여 정보 흐름을 제어하는 기능이 추가되어, 적은 파라미터로 효율적인 학습이 가능합니다.

- **Performance Highlights**: 제안된 모델은 NVidia Dynamic Hand Gesture 및 Briareo 데이터셋에서 state-of-the-art 성능을 달성했으며, 특히 단일 및 다중 모달 입력에서 우수한 결과를 보였습니다. ConvMixFormer는 기존 모델에 비해 파라미터 효율성이 뛰어납니다.



### Arctique: An artificial histopathological dataset unifying realism and controllability for uncertainty quantification (https://arxiv.org/abs/2411.07097)
Comments:
          13 pages, 4 figures

- **What's New**: 본 논문은 Uncertainty Quantification (UQ) 분야에서의 체계적인 비교와 평가를 위한 새로운 데이터 셋인 Arctique을 소개합니다. Arctique은 복잡한 실체적 불확실성을 시험하기 위한 동적으로 생성된 데이터셋으로, 특히 병리학적 대장 조직 이미지를 모델링하여 제작되었습니다.

- **Technical Details**: Arctique은 Blender 기반의 3D 장면 생성 프레임워크를 통해 생성된 50,000장의 이미지로 이루어져 있으며, 각 이미지에 정밀한 마스크(precise masks)와 노이즈가 추가된 레이블 시뮬레이션이 포함됩니다. 이는 사용자가 이미지와 레이블의 불확실성을 독립적으로 조정할 수 있도록 해줍니다.

- **Performance Highlights**: Arctique에서 훈련된 세분화 네트워크는 실제 H&E 이미지에서 제로샷(zero-shot) 성능을 발휘하며, 다양한 UQ 방법론인 Maximum Softmax Response (MSR), Test Time Augmentation (TTA), Monte-Carlo Dropout (MCD), Deep Ensembles (DE)에 대해 성능 평가가 이루어졌습니다. 이 연구는 Arctique이 의미 있고 포괄적인 UQ 벤치마킹을 가능하게 한다는 것을 증명합니다.



### Extreme Rotation Estimation in the Wild (https://arxiv.org/abs/2411.07096)
Comments:
          Project webpage: this https URL

- **What's New**: 본 논문에서는 제한적 또는 비겹치는 시야(View)를 가진 인터넷 이미지 쌍 간의 상대적인 3D 방향 추정 기법과 벤치마크 데이터셋인 ExtremeLandmarkPairs(ELP)를 소개합니다. 기존의 극단적인 회전 추정을 위한 연구들은 제한된 3D 환경을 가정하였고, 파노라마 뷰에서 영역을 잘라내어 관점 이미지를 재현하는 방식이었습니다. 그러나 실제로 자연에서 촬영된 이미지는 다양한 외적 요인으로 인해 모습과 카메라 속성이 크게 변동합니다.

- **Technical Details**: 우리는 Transformer 기반의 모델을 제안합니다. 이 모델은 로컬 키포인트의 공간 분포와 일치 정보, 그리고 의미론적 세분화 맵(Semantic Segmentation Map)과 같은 보조 채널을 활용하여 거의 겹치지 않는 이미지 쌍 간의 관계를 더 잘 추론할 수 있도록 합니다. ExtremeLandmarkPairs 데이터셋은 인터넷에서 수집된 장면 수준 이미지 컬렉션으로 구성되어 있으며, 다양한 시야 각도가 반영되어 있습니다.

- **Performance Highlights**: 우리의 평가 결과, 제안된 방법은 다양한 극단적 시점의 이미지 쌍에 대해 상대적인 회전을 정확하게 예측할 수 있음을 보여줍니다. 특히 실제 인터넷 데이터에서 강력한 베이스라인과 비교하여 유의미한 성능 향상을 나타내며, 파노라마 뷰에서 잘라낸 에뮬레이션된 관점 이미지 쌍에 대해서도 유사한 성능을 달성했습니다.



### StoryTeller: Improving Long Video Description through Global Audio-Visual Character Identification (https://arxiv.org/abs/2411.07076)
- **What's New**: 이 논문에서는 긴 비디오에 대한 일관성 있는 설명을 생성하기 위한 새로운 시스템, StoryTeller를 제안합니다. 이 시스템은 오디오-비주얼 캐릭터 식별(Audio-Visual Character Identification)을 통해 캐릭터와 대사를 효과적으로 연결하여 더 밀집된 비디오 설명을 생성합니다.

- **Technical Details**: StoryTeller는 비디오를 여러 개의 짧은 클립으로 분할하는 비디오 세분화 모듈, 오디오-비주얼 캐릭터 식별 모듈, 그리고 LVLM을 사용하는 설명 생성 모듈로 구성되어 있습니다. 오디오 및 비주얼 입력을 통합한 다중 모달 대형 언어 모델을 사용하여 각 클립에서 캐릭터를 식별합니다.

- **Performance Highlights**: MovieQA 평가에서 StoryTeller는 Gemini-1.5-pro보다 9.5% 높은 정확도를 기록하며, 인간 평가에서는 +15.56%의 우위를 보였습니다. 또한, StoryTeller에서의 오디오-비주얼 캐릭터 식별 정보는 다른 비디오 설명 모델들의 성능도 향상시켰습니다.



### Increasing Rosacea Awareness Among Population Using Deep Learning and Statistical Approaches (https://arxiv.org/abs/2411.07074)
Comments:
          Accepted to 2024 International Conference on Medical Imaging and Computer-Aided Diagnosis

- **What's New**: 이번 연구는 로사세아(rosacea) 자동 검출을 위한 딥 러닝(deep learning) 및 설명 가능한 통계(statistical) 접근 방식을 제시합니다. 제안하는 방법은 ResNet-18을 활용하여 로사세아 환자를 쉽게 구분할 수 있도록 돕습니다.

- **Technical Details**: 본 연구에서는 로사세아 검출을 위해 사전 훈련된 ResNet-18 모델을 미세 조정하여 사용하며, 훈련 데이터의 한계를 극복하기 위해 생성된 데이터를 활용합니다. 또한 통계적 접근 방식으로 주성분 분석(PCA) 및 두 클래스의 평균을 계산하여 특징을 추출합니다.

- **Performance Highlights**: 제안된 방법은 제한된 훈련 데이터로 로사세아 환자와 건강한 사람을 자동으로 구분할 수 있으며, 결과에 대한 설명 가능성을 제공하여 의사와 환자 모두가 결과를 신뢰할 수 있도록 합니다. 이를 통해 로사세아에 대한 인식을 높이고 조기 치료 가능성을 환자들에게 상기시키는 데 기여할 것으로 기대됩니다.



### An Interpretable X-ray Style Transfer via Trainable Local Laplacian Filter (https://arxiv.org/abs/2411.07072)
- **What's New**: 이번 연구에서는 방사선 의사가 X-ray 이미지를 진단 성능 향상을 위해 선호하는 시각적 인상, 즉 '스타일'을 자동으로 전환할 수 있는 기법을 제안합니다.

- **Technical Details**: 제안하는 방법은 Local Laplacian Filter (LLF)의 훈련 가능한 버전을 도입하여 스타일 전환의 최적화된 변환 함수를 형성하고, 이를 통해 스타일 전환의 특성을 추론할 수 있도록 합니다. MLP (Multi-Layer Perceptron)를 사용하여 LLF가 복잡한 X-ray 스타일 특징을 포착할 수 있도록 했습니다.

- **Performance Highlights**: 실험 결과, 처리되지 않은 유방 X-ray 이미지를 목표 유방 촬영 이미지의 스타일에 맞춰 변환하여 기존 LLF 스타일 전환 방법의 SSIM (Structural Similarity Index) 0.82에 비해 0.94를 달성하는 효과성을 입증했습니다.



### SIESEF-FusionNet: Spatial Inter-correlation Enhancement and Spatially-Embedded Feature Fusion Network for LiDAR Point Cloud Semantic Segmentation (https://arxiv.org/abs/2411.06991)
Comments:
          9 pages, 4 figures

- **What's New**: 본 논문에서는 LiDAR 포인트 클라우드의 의미 분할(segmentation)에서 경계 모호성(boundary ambiguity) 문제를 해결하기 위해 새로운 SIESEF-FusionNet 네트워크를 제안합니다. 이 네트워크는 공간적 상호 연관성(spatial inter-correlation)을 향상시키고, 공간적으로 내장된 특징 융합(spatially-embedded feature fusion) 기법을 통해 경계의 정확한 세분화를 지원합니다.

- **Technical Details**: SIESEF-FusionNet은 강화된 지역 공간 인코딩(enhanced local spatial encoding, ELSE) 모듈과 공간 내장 적응 풀링(spatially-embedded adaptive pooling, SEAP) 모듈을 활용하여 공간적인 정보를 더 효과적으로 추출하고, 이를 의미적 특징에 통합합니다. ELSE 모듈은 역거리 가중치(inverse distance weighting)와 각 보정(angular compensation)을 사용하여 효과적으로 지역 공간 정보를 상호 연관시키고, SEAP 모듈은 이를 의미적 특징에 통합하여 맥락 인지(context-awareness)를 강화합니다.

- **Performance Highlights**: Toronto3D 데이터셋에서 SIESEF-FusionNet은 mIoU(Mean Intersection over Union) 83.7%와 OA(Overall Accuracy) 97.8%를 달성했으며, semanticKITTI 데이터셋에서는 61.1% mIoU를 기록하여 다른 기초 방법보다 우수한 성능을 보였습니다. 이 연구의 모듈은 플러그 앤 플레이(plug-and-play) 기능을 가지며, 다양한 모델 아키텍처에 쉽게 통합될 수 있습니다.



### A Hierarchical Compression Technique for 3D Gaussian Splatting Compression (https://arxiv.org/abs/2411.06976)
- **What's New**: 이번 논문에서는 Hierarchical GS Compression (HGSC) 기술을 제안하여 3D Gaussian Splatting (GS) 데이터의 압축 효율을 크게 향상시켰습니다. 중요도 점수에 기반하여 비중요 Gaussians를 제거하고, Octree 구조를 사용하여 3D 위치를 압축하며, KD-tree를 활용하여 계층적 속성 압축 전략을 구현했습니다.

- **Technical Details**: HGSC에서는 비중요 Gaussians를 제거한 후, Octree 구조를 통해 3D GS를 압축합니다. 각 블록에서 anchor primitives를 선택하기 위해 farthest point sampling (FPS) 기술을 적용하며, k-최근접(anchor primitives)을 기반으로 비anchor primitives의 속성을 예측합니다. 이 과정에서 region adaptive hierarchical transform (RAHT)을 사용하여 다양한 속성을 거의 손실 없이 압축합니다.

- **Performance Highlights**: 우리의 방법은 최신 압축 방법에 비해 4.5배 이상의 데이터 크기 감소를 이뤘으며, 작은 장면 데이터세트에서 우수한 압축 품질을 달성하였습니다.



### MapSAM: Adapting Segment Anything Model for Automated Feature Detection in Historical Maps (https://arxiv.org/abs/2411.06971)
- **What's New**: 이번 논문은 역사적인 지도의 자동적인 특징 탐지에 있어 Segment Anything Model (SAM)의 한계를 극복하기 위한 새로운 접근법인 MapSAM을 제안합니다. MapSAM은 파라미터 효율적인 미세 조정 전략을 통해 프롬프트 없이 다양한 하위 작업에 적용 가능하도록 SAM을 조정합니다.

- **Technical Details**: MapSAM은 Weight-Decomposed Low-Rank Adaptation (DoRA) 기술을 활용하여 이미지 인코더에 도메인 특화 지식을 통합합니다. 이를 통해 고유한 파라미터 수를 최소화하고, 자동 프롬프트 생성 과정이 포함되어 있어 수동 입력 없이 작동할 수 있습니다. 또한 Mask2Former에서 영감을 받아 Masked Attention을 도입하여 특성 집합의 효율성을 높입니다.

- **Performance Highlights**: MapSAM 프레임워크는 철도 및 포도밭 탐지를 위한 두 가지 역사적 지도 세그멘테이션 작업에서 뛰어난 성능을 보였습니다. 특히, 10회 촬영으로 미세 조정할 경우에도 다양한 특징에 잘 적응하여 높은 성능을 나타냈습니다.



### Robust Fine-tuning of Zero-shot Models via Variance Reduction (https://arxiv.org/abs/2411.06966)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 클립(CLIP)과 같은 제로샷 모델을 미세 조정할 때, ID(내부 데이터)와 OOD(외부 데이터)에 대한 성능을 동시에 높일 수 있는 샘플별 앙상블 기술, 즉 Variance Reduction Fine-tuning (VRF) 방법을 제안합니다.

- **Technical Details**: VRF는 제로샷 실패(Zero-Shot Failure, ZSF) 세트를 구성하고, 각 테스트 샘플의 ZSF 세트와의 거리를 측정하여 거리의 크기에 따라 미세 조정된 모델의 가중치를 조정합니다. 이 방법은 앙상블 예측의 분산을 줄여 잔여 오차를 감소시키는 효과가 있습니다.

- **Performance Highlights**: ImageNet 및 다섯 개 배포 변동에서 VRF 기법을 적용하여 앙상블 기준선 대비 OOD 정확도를 1.5 - 2.0 pp 향상시키면서 ID 정확도를 유지하거나 증가시켰습니다. 다른 배포 변동 벤치마크에서도 유사한 큰 견고성 향상(0.9 - 3.1 pp)을 달성하였습니다.



### ENAT: Rethinking Spatial-temporal Interactions in Token-based Image Synthesis (https://arxiv.org/abs/2411.06959)
Comments:
          Accepted by NeurIPS2024

- **What's New**: 본 논문에서는 효율적인 비자기 회귀 트랜스포머(Non-Autoregressive Transformers, NATs) 모델을 기반으로 한 EfficientNAT (ENAT)이라는 새로운 이미지를 생성하는 모델을 제안합니다. ENAT는 공간적 및 시간적 상호작용을 최적화하여 성능을 개선하고 계산 비용을 크게 줄입니다.

- **Technical Details**: ENAT는 마스크 토큰과 보이는 토큰의 계산을 분리하여 보이는 토큰을 독립적으로 인코딩하고, 마스크 토큰은 완전히 인코딩된 보이는 토큰을 바탕으로 디코딩됩니다. 시간적으로는 '비판적 토큰(critical tokens)'의 계산을 우선시하고, 이전에 계산된 토큰 표현을 최대한 재사용하여 필요한 정보를 보완합니다.

- **Performance Highlights**: ENAT는 ImageNet-256 및 MS-COCO에서 실험을 통해 기존의 NATs에 비해 24%의 성능 향상과 1.8배의 계산 비용 절감을 동시에 달성하였습니다.



### UMFC: Unsupervised Multi-Domain Feature Calibration for Vision-Language Models (https://arxiv.org/abs/2411.06921)
Comments:
          NeurIPS 2024

- **What's New**: 최근 비전-언어 모델인 CLIP의 한계를 극복하기 위해, 라벨이 없는 다중 도메인 데이터를 활용하여 모델 편향을 교정하는 새로운 접근법인 Unsupervised Multi-domain Feature Calibration (UMFC)을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 UMFC는 라벨이 없는 다중 도메인 설정에서 CLIP의 비주얼 인코더와 텍스트 인코더의 편향을 분석합니다. Visual Feature Calibration (IFC)과 Text Feature Calibration (TFC)의 두 가지 모듈을 통해 카테고리 정보의 우선 순위를 높이고, 도메인 관련 클래스 이름에 대한 편향을 줄입니다.

- **Performance Highlights**: UMFC 방법은 다양한 다운스트림 태스크에서 CLIP을 능가하며, 라벨이 부족한 실제 상황에서도 효과적으로 작동함을 보여줍니다.



### Gaussian Process Emulators for Few-Shot Segmentation in Cardiac MRI (https://arxiv.org/abs/2411.06911)
Comments:
          Submitted to Statistical Atlases and Computational Modeling of the Heart (STACOM) 2024

- **What's New**: 이 논문에서는 cardiac MRI(심장 자기공명영상)의 세분화를 개선하기 위해 few-shot learning을 U-Net 아키텍처 및 Gaussian Process Emulators(GPEs)와 결합한 새로운 방법을 제안합니다. 이 방법은 데이터 적재를 최소화하면서도, 적은 양의 라벨이 있는 지원 집합으로부터 더 나은 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 U-Net의 수축 부분을 이용하여 query 이미지와 지원 이미지의 관계를 잠재 공간(latent space)에서 GPEs가 학습하도록 하며, 이 정보는 확장 부분으로 통합되어 query 이미지의 세분화를 보다 정확하게 수행합니다. 또한, M&Ms-2 공공 데이터셋을 사용하여 다양한 각도에서 심장을 세분화하는 능력을 평가했습니다. 이 모델은 기존의 무감독 및 few-shot 방법과 비교하여 더 높은 DICE 계수를 기록했습니다.

- **Performance Highlights**: 모델의 성능은 심장 자기공명영상에서 작은 지원 집합의 크기로 인해 특히 도전적인 설정에서 다른 방법들과 비교할 때 현저하게 개선되었으며, 더 나은 일반화 능력을 보였습니다.



### EVQAScore: Efficient Video Question Answering Data Evaluation (https://arxiv.org/abs/2411.06908)
- **What's New**: 이번 연구에서는 비디오 질문-응답(Question-Answering, QA) 데이터 평가를 위한 새로운 방법인 EVQAScore를 소개합니다. 기존의 방법들이 비디오 캡션 평가에만 초점을 맞춰 결과적으로 비디오 QA 평가에 적합한 지표가 부족했던 문제를 해결하고자 합니다.

- **Technical Details**: EVQAScore는 키워드 추출(keyword extraction)과 프레임 샘플링(frame sampling) 기법을 활용하여 비디오 QA 및 캡션 데이터 품질을 평가합니다. 이를 통해 비디오의 긴 길이에도 효과적으로 작동할 수 있도록 하여 평가 비용을 30배 줄이는 성과를 거두었습니다. 또한, LLMs를 통해 데이터의 어휘적 의미를 거리 평가하는 전통적인 TF-IDF 방법보다 의미를 더 정확하게 이해할 수 있습니다.

- **Performance Highlights**: VATEX-EVAL 벤치마크에서 EVQAScore는 Kendall 상관관계 32.8, Spearman 상관관계 42.3을 기록하여 이전 방법인 PAC-S++보다 각각 4.7점, 5.9점 높은 성과를 보였습니다. 데이터 선택에 EVQAScore를 사용하여 원본 데이터의 12.5%만으로도 이전 SOTA 방법인 PAC-S 및 100% 데이터와 비교해 성과를 뛰어넘는 결과를 달성했습니다.



### BuckTales : A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes (https://arxiv.org/abs/2411.06896)
Comments:
          9 pages, 5 figures

- **What's New**: BuckTales는 멀티 오브젝트 트래킹(MOT) 및 재식별(Re-ID) 문제를 해결하기 위해 설계된 최초의 대규모 UAV 데이터셋으로, 주로 흑턱제비나물의 짝짓기 행동을 관찰합니다. 이 데이터셋은 1.2백만 개의 주석과 12개의 고해상도(5.4K) 비디오를 포함하여 동물 행동 연구를 위한 새로운 기회를 제공합니다.

- **Technical Details**: BuckTales 데이터셋은 22.5K 프레임(12.5분)에 걸쳐 있으며, 가장 긴 비디오는 3분 넘게 지속되고 평균 75마리의 제비나물이 등장합니다. 이 데이터셋은 동물의 기존 행동 관찰과 컴퓨터 비전 방법을 결합하여 장기적인 트래킹을 위한 계측을 할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 우리는 두 가지 탐지기를 사용한 기초 성능을 제공하고 여러 최신 트래킹 방법에 대한 벤치마킹을 진행하였습니다. BuckTales는 야생 동물 모니터링에 대한 자동화와 장기적 관찰을 가능하게 할 수 있습니다.



### Multi-scale Frequency Enhancement Network for Blind Image Deblurring (https://arxiv.org/abs/2411.06893)
- **What's New**: 본 논문에서는 모자이크 형태의 블러(blur)를 인식하고 복원하는 새로운 프레임워크인 다중 스케일 주파수 향상 네트워크(Multi-scale Frequency Enhancement Network, MFENet)를 제안합니다. 기존 알고리즘이 직면한 블러 복원 한계를 극복하기 위해, MFENet은 멀티스케일 기능 추출과 주파수 개선 기법을 효과적으로 통합하여 높은 수준의 해상도를 복원하는 데 중점을 둡니다.

- **Technical Details**: MFENet은 깊이 분리 컨볼루션(depthwise separable convolutions)을 기반으로 한 다중 스케일 기능 추출 모듈(MS-FE)과 웨이블릿 변환(wavelet transforms)을 활용한 주파수 향상 블러 인식 모듈(FEBP)을 포함합니다. 이 구조는 이미지의 스페이셜(spatial) 및 주파수 도메인(feature domain) 특성을 동시에 처리하여 복잡한 장면에서 비균일 블러를 효과적으로 감지하고 복원합니다.

- **Performance Highlights**: GoPro 및 HIDE 데이터셋에서 수행된 실험 결과에 따르면, MFENet은 기존 방법들에 비해 시각적 품질과 정량적 지표에서 우수한 성능을 보였으며, 다운스트림 객체 감지(object detection) 작업에서도 감지 정확도가 현저히 개선되었습니다.



### Classification of residential and non-residential buildings based on satellite data using deep learning (https://arxiv.org/abs/2411.06879)
- **What's New**: 본 논문은 고해상도 위성 데이터와 벡터 데이터를 결합한 새로운 딥러닝(dedep learning) 접근 방식을 제안합니다. 이는 주거 및 비주거 건물의 분류를 자동화하는데 있어 중요한 발전을 제공합니다.

- **Technical Details**: 제안된 모델은 LeakyReLU와 ReLU 활성화 함수(activation function)를 활용하여 데이터의 비선형성(nonlinearities)을 포착합니다. 또한, 고도로 상관된 특징들을 제거하는 기능 공학(feature-engineering) 기술을 사용하여 계산 효율성을 향상시킵니다.

- **Performance Highlights**: 대규모 데이터셋에서의 실험 결과, 본 모델은 0.9936의 인상적인 전체 F1-score를 달성하며, 이는 건물 분류에 있어 확장 가능하고 정확한 솔루션을 제공합니다.



### Multi-Modal interpretable automatic video captioning (https://arxiv.org/abs/2411.06872)
- **What's New**: 본 논문은 비디오 자막 생성에 관한 새로운 접근 방식을 제안합니다. 특히, 시각 정보뿐만 아니라 오디오 정보를 통합하여 더 나은 자막을 생성하는 멀티 모달 학습 프레임워크를 도입합니다.

- **Technical Details**: 새로운 비디오 자막 생성 방법은 멀티 모달 대비 손실(multi-modal contrastive loss)을 사용하여 시각 및 청각 정보를 통합하고, 주의(attention) 메커니즘을 통해 모델의 의사결정 과정에 대한 해석 가능성(interpretable)을 제공합니다. 이러한 접근 방식은 입력으로 들어오는 이미지 시퀀스와 오디오 캡션을 처리하는 인코더-디코더 구조를 활용합니다.

- **Performance Highlights**: 제안된 방법은 MSR-VTT와 VATEX 같은 벤치마크 데이터셋에서 최신 모델들과 비교했을 때 우수한 성능을 보였습니다.



### CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models (https://arxiv.org/abs/2411.06869)
- **What's New**: CapeLLM은 카테고리 비의존적 포즈 추정(Category-Agnostic Pose Estimation, CAPE)을 위한 혁신적인 접근 방식으로, 기존의 지원 이미지 없이 쿼리 이미지(query image)와 상세한 텍스트 설명만을 이용하여 키포인트(keypoints)를 추정합니다.

- **Technical Details**: CapeLLM은 텍스트 기반 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 활용하여 CAPE에 적합하도록 설계되었습니다. 연구에서는 모델의 아키텍처와 입력으로 사용되는 지시(description) 설정을 세밀하게 조정하고, LLM이 텍스트를 이해하는 능력을 완전 활용하여 예상 입력에 대한 결과를 생성합니다.

- **Performance Highlights**: CapeLLM은 MP-100 벤치마크에서 1-shot 설정에서 기존 모델들의 5-shot 성능을 초과하는 뛰어난 성능을 시연하며, 카테고리 비의존적 포즈 추정 분야에서 새로운 최첨단 성과를 달성하였습니다.



### Veri-Car: Towards Open-world Vehicle Information Retrieva (https://arxiv.org/abs/2411.06864)
Comments:
          33 pages, 12 figures

- **What's New**: 이 논문에서는 차량의 특성을 이미지로부터 추출하기 위한 새로운 도구인 Veri-Car을 제안합니다.

- **Technical Details**: Veri-Car은 감독 학습 기법(supervised learning techniques)을 활용하여 자동차의 제조사(make), 유형(type), 모델(model), 연도(year), 색상(color), 번호판(license plate)을 정확히 식별합니다. 또한, 공개 세계 문제(open-world problems)를 처리하기 위해 사전 훈련된 모델(pre-trained models)과 계층적 다중 유사성 손실(hierarchical multi-similarity loss) 조합을 활용합니다.

- **Performance Highlights**: Veri-Car은 보지 않은 데이터(unseen data)와 보인 데이터(seen data) 모두에서 높은 정밀도(precision)와 정확도(accuracy)를 달성하며, 합성_license plate_detection과 OCR 모델을 통합하여 번호판 번호를 놀라운 정확도로 추출합니다.



### Fast and Efficient Transformer-based Method for Bird's Eye View Instance Prediction (https://arxiv.org/abs/2411.06851)
Comments:
          The article has been presented in the 27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024) on September, 2024. Number of pages: 6, Number of figures: 4

- **What's New**: 이 논문에서는 최신 BEV(instance segmentation, flow prediction 기반의) 구조를 제안하여 자율주행차량의 객체 예측 및 경로 예측의 정확성을 높입니다.

- **Technical Details**: 제안된 아키텍처는 'Lift, Splat, Shoot' 방식을 기반으로 하며, 이를 통해 다중 카메라에서 얻은 데이터를 효과적으로 처리합니다. 이 모델은 효율적인 transformer 기반의 구조로, 적은 수의 파라미터와 빠른 추론 시간을 자랑합니다.

- **Performance Highlights**: PyTorch 2.1을 사용하여 성능을 최적화하였으며, 이는 기존 SOTA 아키텍처에 비해 상당한 성능 향상을 이뤄냈습니다.



### ScaleKD: Strong Vision Transformers Could Be Excellent Teachers (https://arxiv.org/abs/2411.06786)
Comments:
          This work is accepted to NeurIPS 2024. The project page: this https URL

- **What's New**: 이 논문에서는 잘 정밀 훈련된 비전 트랜스포머 (ViT) 모델이 교사 역할을 하여 크로스 아키텍처 지식 증류 (Knowledge Distillation, KD) 연구를 진전시키는 데 활용될 수 있는지를 질문합니다.

- **Technical Details**: 우리는 ScaleKD라는 새로운 KD 방법을 제안합니다. 이 방법은 세 가지 상호 연관된 구성요소인 cross attention projector (CAP), dual-view feature mimicking, teacher parameter perception을 결합하여, CNN, MLP 및 ViT 아키텍처 전반에 걸쳐 활용될 수 있는 효과적인 지식 전이 방법을 제공합니다.

- **Performance Highlights**: ScaleKD는 ImageNet-1K 데이터셋에서 MobileNet-V1, ResNet-50, ViT-S/16과 같은 다양한 모델들에 대해 각각의 개별 훈련 모델 대비 2%에서 5%까지 절대 정확도 향상을 보여주며, 다운스트림 MS-COCO 및 ADE20K 데이터셋에서도 우수한 성능을 발휘합니다.



### HSTrack: Bootstrap End-to-End Multi-Camera 3D Multi-object Tracking with Hybrid Supervision (https://arxiv.org/abs/2411.06780)
Comments:
          9 pages, 2 figures

- **What's New**: HSTrack는 카메라 기반 3D 다중 객체 추적(MOT)에서 포함된 추적 쿼리와 객체 쿼리의 경쟁을 피하기 위해 새로운 병렬 가중치 공유 디코더 디자인을 제안하는 혁신적인 방법으로, 여러 데이터셋에 대해 성능을 크게 개선합니다.

- **Technical Details**: HSTrack은 self-attention 없이 동작하는 병렬 디코더를 구축하여 추적 쿼리 및 객체 쿼리를 구분하고, 이는 spatio-temporal modeling과 품질 후보 생성을 향상시킵니다. 또한, track 쿼리와 object 쿼리에 대해 각각 one-to-one 및 one-to-many 레이블 할당 전략을 채택합니다.

- **Performance Highlights**: HSTrack은 nuScenes 데이터셋에서 최신 PF-Track 방법에 비해 AMOTA는 +2.3%, mAP는 +1.7% 개선하여 성능을 지속적으로 향상시킵니다.



### Machine vision-aware quality metrics for compressed image and video assessmen (https://arxiv.org/abs/2411.06776)
Comments:
          16 pages, 10 figures

- **What's New**: 본 연구는 video-compression 알고리즘의 새로운 관점을 제시하며, 머신 비전(Computer Vision) 향상에 최적화된 비디오 코덱(Video Codec)을 개발하는 데 필요한 프레임워크를 제공합니다. 특히, 객체 탐지(Object Detection), 얼굴 인식(Face Recognition), 번호판 인식(License Plate Recognition) 등 다양한 비전 알고리즘에 대한 영상 품질 평가 메트릭스를 도입하고, 이 메트릭스가 기존 메트릭스보다 머신 비전 결과와 더 높은 상관관계를 가지도록 개선되었습니다.

- **Technical Details**: 본 연구에서는 머신 비전을 위한 새로운 품질 평가 메트릭스를 제안하였으며, 딥 뉴럴 네트워크(Deep Neural Networks)에 기반한 기존 품질 평가 방법이 제공하는 낮은 상관관계를 분석하였습니다. 연구 방법론은 PSNR, SSIM 등의 기존 메트릭이 아닌, 객체 탐지, 얼굴 탐지 및 인식, 번호판 탐지 및 인식 성능을 기준으로 한 CNN(Convolutional Neural Network) 모델 기반의 새로운 비디오 품질 메트릭스를 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 메트릭스는 객체 탐지, 얼굴 탐지 및 인식, 번호판 탐지 및 인식 작업에서 기존 메트릭스들보다 성능이 높은 상관관계를 보였습니다. 이는 기존의 인간 구인 시각적 품질 평가 메트릭으로는 평가할 수 없는 머신 비전 알고리즘의 성능 개선 및 최적화 가능성을 시사합니다.



### Multi-Stage Knowledge Integration of Vision-Language Models for Continual Learning (https://arxiv.org/abs/2411.06764)
- **What's New**: 본 논문은 Vision Language Model(VLM)에 대한 지속적인 학습을 목표로 하고 있으며, Knowledge Integration Theory(KIT)를 바탕으로 한 Multi-Stage Knowledge Integration network(MulKI)를 제안합니다. 이 네트워크는 인간의 학습 프로세스를 모방하여 VLM이 새로운 데이터 분포에 효과적으로 적응할 수 있도록 돕습니다.

- **Technical Details**: MulKI는 Eliciting Ideas, Adding New Ideas, Distinguishing Ideas, Making Connections의 네 단계로 구성됩니다. 각 단계에서는 프로토타입을 활용하여 다양한 모달리티 간의 정렬을 수행하고, 두 개의 Teacher 모델로부터 지식을 적응적으로 구별하고 조정합니다. 이 방법은 추가 데이터나 이전 작업의 프로토타입 보존 없이도 VLM을 지속적으로 학습할 수 있게 합니다.

- **Performance Highlights**: MulKI는 다양한 하위 작업에서 지속적인 학습을 지원하며, 제로샷(zero-shot) 기능을 유지하는 데 있어 상당한 개선을 보여줍니다. 성능 평가에서 기존의 방법들보다 우수한 결과를 나타내며, VLM이 변화하는 데이터 분포에 적응할 수 있는 잠재력을 입증합니다.



### LuSh-NeRF: Lighting up and Sharpening NeRFs for Low-light Scenes (https://arxiv.org/abs/2411.06757)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 해당 논문에서는 손으로 들고 촬영한 저조도 이미지에서 깨끗하고 선명한 Neural Radiance Field (NeRF)를 재구성하는 새로운 모델인 LuSh-NeRF를 제안합니다. LuSh-NeRF는 이미지 내 노이즈(noise)와 블러(blur)를 순차적으로 모델링하여 손잡이 있는 저조도 이미지로부터 높은 품질의 이미지를 생성합니다.

- **Technical Details**: LuSh-NeRF는 Scene-Noise Decomposition (SND) 모듈과 Camera Trajectory Prediction (CTP) 모듈을 포함하여, SND 모듈은 장면 표현(scene representation)에서 노이즈를 분리하고, CTP 모듈은 저주파(low-frequency) 장면 정보를 기반으로 카메라의 운동을 예측합니다. 이러한 두 모듈은 서로의 출력을 사용하여 반복적으로 최적화됩니다.

- **Performance Highlights**: LuSh-NeRF는 실험에서 기존 방법들보다 더 우수한 성능을 보였습니다. 새로운 데이터셋은 합성된 이미지와 실제 이미지를 모두 포함하고 있으며, LuSh-NeRF는 밝고 선명한 새로운 시점 이미지를 렌더링하는 데 성공했습니다.



### Can KAN Work? Exploring the Potential of Kolmogorov-Arnold Networks in Computer Vision (https://arxiv.org/abs/2411.06727)
- **What's New**: 이번 연구는 Kolmogorov-Arnold Networks(KANs)의 컴퓨터 비전 적용 가능성을 탐구합니다. KAN의 이미지 분류 및 시멘틱 세분화에서의 성능을 평가하며, 데이터 스케일 및 노이즈 수준에 따른 특성을 분석합니다. KAN이 복잡한 패턴을 잡아내는 데 강력하긴 하나, 노이즈에 대한 높은 민감도로 인해 강인성이 제한적입니다. 이를 해결하기 위해 Smoothness Regularization 기법과 Segment Deactivation 기법이 제안되었습니다.

- **Technical Details**: Kolmogorov-Arnold Networks(KANs)는 Kolmogorov-Arnold 표현 정리에 기반하여 다변수 함수를 단일 변수 함수의 합으로 표현할 수 있습니다. KAN은 학습 가능한 활성화 함수인 B-spline 함수를 사용하여 복잡한 패턴에 적응합니다. KAN과 Convolutional KAN(CKAN)는 이미지 분류 및 시멘틱 세분화 같은 다양한 비전 작업에서 실험되었습니다. 연구에서는 노이즈 환경에서 KAN의 안정성을 향상시키기 위해 Smoothness Regularization과 Segment Deactivation을 도입했습니다.

- **Performance Highlights**: KAN은 대규모 데이터셋에서 강력한 적합 능력을 보였으나 노이즈에 민감하여 강인성이 떨어졌습니다. Smoothness Regularization 기법을 통해 모델의 학습 안정성을 향상시켰고, Segment Deactivation 기법을 통해 복잡성을 줄여 KAN의 성능을 개선했습니다. 이 두 가지 접근 방식은 KAN이 복잡한 시각적 데이터 작업을 처리하는 데 유망함을 보여주었습니다.



### GTA-Net: An IoT-Integrated 3D Human Pose Estimation System for Real-Time Adolescent Sports Posture Correction (https://arxiv.org/abs/2411.06725)
Comments:
          18 pages

- **What's New**: 이 연구에서는 3D 인간 자세 추정(3D human pose estimation)을 기반으로 한 새로운 시스템인 GTA-Net를 소개하였습니다. 이 시스템은 IoT(Internet of Things) 기술을 통합하여 청소년 스포츠에서의 자세 교정을 실시간으로 지원합니다.

- **Technical Details**: GTA-Net는 Graph Convolutional Networks (GCN), Temporal Convolutional Networks (TCN), 및 Hierarchical Attention 메커니즘을 포함하여 동적 장면에서 자세 추정을 향상시킵니다. 이 시스템은 IoT 장치를 통해 실시간 데이터를 전송하고 피드백을 제공합니다.

- **Performance Highlights**: 실험 결과, GTA-Net는 Human3.6M, HumanEva-I, 및 MPI-INF-3DHP 데이터셋에서 각각 32.2mm, 15.0mm, 48.0mm의 평균 관절 위치 오류(Mean Per Joint Position Error, MPJPE) 값을 기록하며 기존 방법보다 현저히 우수한 성능을 보였습니다. 복잡한 상황에서도 높은 정확도를 유지하며, 자세 교정의 실시간 지원을 강화합니다.



### Shallow Signed Distance Functions for Kinematic Collision Bodies (https://arxiv.org/abs/2411.06719)
Comments:
          Preprint

- **What's New**: 본 논문에서는 의류 시뮬레이션에서 발생하는 실시간 아바타 충돌 쿼리를 위한 학습 기반의 암시적 형태 표현(implicit shape representations)을 제안합니다. 우리는 전통적인 표현 방식에 비해 메모리 요구 사항이 적은 여러 형상을 표현할 수 있는 깊은 신경망(Deep Neural Networks)을 사용합니다.

- **Technical Details**: 우리는 인간 아바타의 SDF( signed distance functions) 표현을 설계하며, 특정 관절의 변화로 인한 형태 변형을 표현하기 위해 매우 효율적인 얕은 신경망(shallow neural networks)의 집합을 사용합니다. 각 얕은 SDF는 전신의 경계에 대한 거리를 효율적으로 표현하기 위해 스티칭(stitching) 처리 과정을 통해 결합됩니다.

- **Performance Highlights**: 제안된 모델은 매우 빠르고 정확하게 작동하며, 애니메이션 캐릭터에 의해 구동되는 의류의 실시간 시뮬레이션에서 그 적용 가능성을 보여줍니다.



### United Domain Cognition Network for Salient Object Detection in Optical Remote Sensing Images (https://arxiv.org/abs/2411.06703)
Comments:
          Accepted at TGRS 2024

- **What's New**: 최근 딥러닝 기반의 선명 객체 탐지 (Salient Object Detection, SOD) 기술이 광학 원격 감지 이미지 (Optical Remote Sensing Images, ORSIs)에서 큰 성과를 달성하였습니다. 본 논문에서는 기존의 방법들이 주로 공간 영역에서 픽셀 특징을 최적화하는데 집중하고 있음을 지적하고, 지역적 특성에 국한된 제한을 극복하기 위해 푸리에 변환 (Fourier Transform)을 도입하여 글로벌 주파수 특징을 생성하고 이미지 크기 수용 영역 (receptive field)을 확장하는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 'United Domain Cognition Network' (UDCNet)라는 새로운 네트워크 구조를 제안합니다. 이 네트워크는 공간 영역과 주파수 영역에서의 글로벌-로컬 정보를 공동으로 탐색하도록 설계되었습니다. 특히, 주파수-공간 도메인 변환기 블록 (Frequency-Spatial Domain Transformer, FSDT)을 개발하였으며, 이는 공간 인지 자가 주의 (Spatial Perception Self-Attention, SPSA)와 주파수 인지 자가 주의 (Frequency Perception Self-Attention, FPSA) 구조를 활용해 지역적 공간 특징과 글로벌 주파수 특징을 상호 결합합니다. 또한, 두 가지 분기 구조를 가진 공동 최적화 디코더를 구성하여 객체의 고급 구조와 경계 정보를 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 UDCNet 방법이 24개의 최첨단 SOD 모델을 초월한 우수성을 입증하였으며, 세 가지 널리 사용되는 ORSIs-SOD 데이터셋에서 정량적, 정성적으로 비교 분석하였습니다. 이 연구는 기존 방법들의 한계를 극복하고 더 정확한 원격 감지 객체 탐지를 가능하게 할 것으로 기대됩니다.



### Track Any Peppers: Weakly Supervised Sweet Pepper Tracking Using VLMs (https://arxiv.org/abs/2411.06702)
- **What's New**: 이번 연구에서는 Track Any Peppers (TAP)라는 약한 감독 학습을 활용한 다양한 객체 추적 기술을 제안합니다. TAP는 비전-언어 모델인 Grounding DINO의 제로샷(Zero-shot) 탐지 기능을 활용하여 최소한의 인간 개입으로 고추(스위트 페퍼)의 가짜 라벨(pseudo-labels)을 자동 생성합니다.

- **Technical Details**: 이 접근 방식은 네 가지 주요 단계로 구성됩니다: 1) 약한 라벨 수집, 2) YOLOv8로 마스크 및 박스 탐지, 3) 전처리 및 후처리 단계, 4) 앙상블 방법을 활용한 하이브리드 객체 추적. 제로샷 탐지를 위해 Grounding DINO를 사용하여 텍스트 쿼리 기반으로 객체를 탐지하고, YOLOv8 분할 네트워크를 통해 세분화된 라벨과 공개 데이터셋을 결합하여 훈련합니다. 또한, 조명 조정 및 깊이 필터링 기법이 포함되어 정확성을 높입니다.

- **Performance Highlights**: 이 방법론은 HOTA(Higher Order Tracking Accuracy) 80.4%, MOTA(Multi-Object Tracking Accuracy) 66.1%, Recall 74.0%, Precision 90.7%의 성능 지표를 기록하며, extensive한 수작업 없이도 스위트 페퍼를 효율적으로 추적할 수 있음을 보여주었습니다.



### HomoMatcher: Dense Feature Matching Results with Semi-Dense Efficiency by Homography Estimation (https://arxiv.org/abs/2411.06700)
Comments:
          10 pages, 5 figures, conference under review

- **What's New**: 본 논문은 기존의 semi-dense matching 기법에서의 세부 매칭 모듈을 개선하는 방법을 제안합니다. 가벼운 homography 추정 네트워크를 활용하여 coarse matching으로 얻어진 패치 간의 시각적 매핑을 생성합니다.

- **Technical Details**: 이 연구에서는 homography estimation을 사용하여 패치 간의 정확한 매칭을 수행하고, 이를 통해 sub-pixel 수준의 정밀도를 달성합니다. 제안된 방법은 coarse-to-fine 구조를 가진 기존의 detector-free 기법에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 종합적인 실험 결과, 본 방법은 기존의 semi-dense matcher에 비해 높은 정확도를 달성하며, dense matcher에 비해 유사한 end-point-error 정확도를 기록하면서도 semi-dense의 효율성을 유지합니다.



### Layout Control and Semantic Guidance with Attention Loss Backward for T2I Diffusion Mod (https://arxiv.org/abs/2411.06692)
- **What's New**: 이 논문은 훈련이 필요 없는 방법으로 이미지 생성을 제어하는 새로운 기법을 제안합니다.

- **Technical Details**: 우리는 attention loss backward 기반의 방법을 사용하여 cross attention map을 조절합니다. 외부 조건(예: prompts)을 통해 attention map에 합리적으로 매핑하여 이미지 생성을 제어합니다.

- **Performance Highlights**: 우리의 접근 방식은 속성 불일치(attribute mismatch)와 부적절한 프롬프트 준수(poor prompt-following) 문제를 해결하며, 실질적인 생산 응용에서 뛰어난 성과를 보였습니다.



### SeedEdit: Align Image Re-Generation to Image Editing (https://arxiv.org/abs/2411.06686)
Comments:
          Our website: this https URL

- **What's New**: SeedEdit라는 새로운 diffusion model을 소개하며, 주어진 이미지를 텍스트 프롬프트로 수정할 수 있는 기술을 개발하였습니다.

- **Technical Details**: 이 모델은 원본 이미지를 유지하는 이미지 재구성(image reconstruction)과 새로운 이미지를 생성하는 이미지 재생성(image re-generation) 간의 최적의 균형을 찾는 것을 목표로 합니다. 초기에는 약한 generator(텍스트-이미지 모델)에서 시작하여, 두 방향 간의 다양한 쌍을 생성하고 이를 점진적으로 강력한 이미지 편집기로 정렬합니다.

- **Performance Highlights**: SeedEdit는 이전의 이미지 편집 방법들보다 더 다양하고 안정적인 편집 기능을 제공하며, diffusion 모델로 생성된 이미지에 대해 연속적인 수정을 가능하게 합니다.



### High-Frequency Enhanced Hybrid Neural Representation for Video Compression (https://arxiv.org/abs/2411.06685)
- **What's New**: 본 논문은 기존의 Neural Representations for Videos (NeRV) 방식의 한계를 극복하기 위해, 고주파 세부정보를 향상시키는 Hybrid Neural Representation Network를 소개합니다.

- **Technical Details**: 우리는 고주파 정보를 활용하여 세부정보를 개선하기 위해 Wavelet Frequency Decomposer (WFD) 블록을 포함하는 wavelet 고주파 인코더를 설계하였습니다. 또한, HFM (High-Frequency Feature Modulation) 블록을 설계하여 추출된 고주파 임베딩을 사용하여 디코더의 적합 과정을 향상시킵니다. 마지막으로, Harmonic decoder 블록과 동적 가중치 주파수 손실(Dynamic Weighted Frequency Loss)을 통해 고주파 정보 손실을 최소화합니다.

- **Performance Highlights**: Bunny 및 UVG 데이터셋을 통한 실험에서 본 방법은 세부정보 보존 및 압축 성능에서 다른 방법들보다 유의미한 개선을 보였습니다.



### Learning from Different Samples: A Source-free Framework for Semi-supervised Domain Adaptation (https://arxiv.org/abs/2411.06665)
- **What's New**: 이번 논문에서는 기존의 Semi-supervised Domain Adaptation (SSDA) 방법의 한계점을 극복하기 위해 서로 다른 타입의 target 샘플에 맞춤형 학습 전략을 적용할 수 있는 새로운 프레임워크인 SOUF(Source-Free Unified Framework)를 제안합니다.

- **Technical Details**: SOUF는 unlabeled, reliably labeled, noisy pseudo-labeled target 샘플을 개별적으로 처리하며, 각 샘플 유형에 대해 다음과 같은 기법을 도입합니다: 1) Probability-based Weighted Contrastive Learning (PWC)를 통해 unlabeled target 샘플에서 더 구별되는 feature representation을 학습합니다. 2) Reliability-based Mixup Contrastive Learning (RMC)로 labeled target 샘플의 심층 지식을 학습합니다. 3) Predictive Regularization Learning (PR)을 사용하여 noisy pseudo-labeled 샘플의 잡음을 줄입니다.

- **Performance Highlights**: 본 연구는 다양한 benchmark 데이터셋에서 SOUF 프레임워크가 기존의 최신 SSDA 방법들보다 우수한 성능을 나타낸다는 것을 실험적으로 입증했습니다.



### Renaissance: Investigating the Pretraining of Vision-Language Encoders (https://arxiv.org/abs/2411.06657)
- **What's New**: 최근 비전-언어 (Vision-Language) 과제를 위한 모델들이 급격히 증가하며, 이와 관련된 모델 디자인 및 훈련의 모범 사례에 대한 질문들이 여전히 남아있습니다. 본 논문에서는 비전-언어 인코더의 사전 훈련에 관한 질문에 답하고, 두 개의 주요 실험을 통해 가시적인 성능 저하 없이 계산 비용을 크게 절감할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 'Renaissance'라는 비전-언어 모델링 플랫폼을 소개하며, 대규모 컴퓨팅 자원을 절약하기 위해 사전 훈련 중 모델의 여러 부분을 고정(freezing)하는 실험을 수행하였습니다. 첫 번째 실험에서는 두 개의 모듈이 모두 고정된 경우 일부 성능 저하가 발생했지만, 시각 모듈을 고정시켰을 때는 성능이 증가하는 경향이 나타났습니다. 두 번째 실험에서는 텍스트 인코더와 비전 인코더의 사전 훈련 성능을 비교하였습니다.

- **Performance Highlights**: 결과적으로, 한 개의 타워 인코더 모델을 훈련할 때는 사전 훈련된 가중치보다 무작위 초기화(randomly initialized)가 더 나은 성능을 보이는 것으로 나타났습니다. Renaissance 플랫폼은 다양한 비전-언어 모델 유형을 평가하는 데 유용하며, 향후 연구에서 보다 많은 VL 모델에 대한 지원이 필요합니다.



### LFSamba: Marry SAM with Mamba for Light Field Salient Object Detection (https://arxiv.org/abs/2411.06652)
Comments:
          Accepted by SPL

- **What's New**: 이 논문에서는 최신의 salient object detection 모델인 LFSamba를 소개하며, multi-focus light field 이미지에서 주목할만한 객체를 효과적으로 검출하는 방법을 제안합니다.

- **Technical Details**: LFSamba는 SAM(Segment Anything Model)을 기반으로 하는 두 개의 스트림 인코더-디코더 프레임워크입니다. 이 방법은 다양한 focal slice에서의 관계와 all-focus 이미지 간의 관계를 모델링하여, weakly supervised learning 방식으로 주석을 생성합니다. Mamba를 이용해 seqential fusion 및 multi-modal fusion이 이루어집니다.

- **Performance Highlights**: LFSamba는 기존의 방법보다 뛰어난 성능을 보이며, multi-focus light field 이미지에서의 salient object detection을 위한 새로운 기준선을 estabelece 합니다. 또한, sparse scribbles를 활용한 주석화를 통해 훈련 비용을 줄일 수 있음을 입증했습니다.



### Few-shot Semantic Learning for Robust Multi-Biome 3D Semantic Mapping in Off-Road Environments (https://arxiv.org/abs/2411.06632)
Comments:
          Accepted to Australasian Conference on Robotics and Automation (ACRA 2024)

- **What's New**: 이 연구에서는 언구조적 지형과 악화된 센싱 조건에서 고속 자율 네비게이션을 위한 새로운 접근 방식을 제안합니다. 사전 훈련된 Vision Transformer (ViT)를 사용하여 소량의 다중 생물군 데이터셋(500 이미지 미만)에서 2D semantic segmentation을 예측합니다.

- **Technical Details**: 우리는 감지된 클래스를 시간에 따라 집계하여 3D semantic voxel 맵으로 결합합니다. 이 과정에서 novel range-based metric을 활용하여 semantic 정보를 통합합니다. 또한, Yamaha와 Rellis 데이터 세트를 사용하여 zero-shot 및 few-shot 학습을 이용한 segmentation을 수행합니다.

- **Performance Highlights**: Yamaha 데이터 세트에서 52.9 mIoU, Rellis 데이터 세트에서 55.5 mIoU를 달성하였으며, few-shot sparse labeling을 통해 Yamaha에서 66.6 mIoU, Rellis에서 67.2 mIoU로 성능을 개선했습니다. 이 기법을 활용하여 오프로드 환경의 위험 요소를 효과적으로 처리할 수 있는 가능성을 보여줍니다.



### Adaptive and Temporally Consistent Gaussian Surfels for Multi-view Dynamic Reconstruction (https://arxiv.org/abs/2411.06602)
- **What's New**: 본 논문에서 제안하는 AT-GS(Applicable and Temporally Consistent Gaussian Surfels)는 다중 시점 비디오에서 높은 품질의 동적 표면을 재구성하기 위한 새로운 효율적 방법입니다. 이전 방법들은 긴 시퀀스의 복잡한 동적 장면을 다루는 데 한계를 보였으나, AT-GS는 프레임별 증분 최적화(post-frame incremental optimization)를 통해 이러한 문제를 해결합니다.

- **Technical Details**: AT-GS는 Gaussian surfels 표현을 기반으로 하여 조정된 통합 밀도 확장 전략(unified density-aware densification strategy)을 도입합니다. 이 방법은 정의된 경량 PDF(probability density function) 샘플링을 적용하여 Gaussian의 위치 기울기에 따라 분할 과정을 유도하며, 연속 프레임에서의 곡률 맵(curvature maps)의 일관성을 유지하여 템포럴(Jittering) 불안정을 줄입니다.

- **Performance Highlights**: 본 방법은 다양한 다중 뷰 비디오 데이터 세트에서 실험을 실시하여 기존의 방법보다 높은 정확성과 템포럴 일관성을 보여주었습니다. AT-GS는 복잡한 동적 장면에서도 고충실도의 공간-시간 새로운 뷰 합성을 수행하고, 높은 훈련 속도를 유지하면서 사실적인 표면 재구성을 달성합니다.



### Graph Neural Networks for modelling breast biomechanical compression (https://arxiv.org/abs/2411.06596)
Comments:
          Deep Breath @ MICCAI 2024 | The code is available at this URL: this https URL

- **What's New**: 이번 연구는 PhysGNN(Physics-based Graph Neural Networks)을 처음으로 유방 압축 시뮬레이션에 적용하여, 기존의 Finite Element Analysis(FEA) 방법과 비교하며 성과를 분석합니다. 이 모델은 복잡한 유방 조직 기하학을 캡처할 수 있는 독특한 메쉬 구조 정보를 통합하여 운용됩니다.

- **Technical Details**: PhysGNN은 그래프 신경망(Graph Neural Network)을 기반으로 하여 비선형 소프트 티슈(soft tissue) 변형을 예측하는 딥러닝 프레임워크입니다. FEA로부터 훈련된 PhysGNN 모델은 30단계로 진행된 압축 시뮬레이션의 결과를 바탕으로 하며, 90 뉴턴의 힘을 가진 다양한 방향으로 유방 표면 노드에 적용됩니다. 또는 FEM(유한요소모델)을 사용하여 개발된 메쉬는 17595개 노드와 95865개 테트라헤드 요소로 구성되어 있습니다.

- **Performance Highlights**: PhysGNN의 성능은 노드 변위를 예측하는 데 있어 기존 FEA 시뮬레이션과 비교하여 높은 정확도와 속도를 보여 주며, 실제 시나리오에서 더 나은 계산 효율성을 제공합니다.



### Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinemen (https://arxiv.org/abs/2411.06558)
- **What's New**: 이 논문에서는 RAG(Regional-Aware text-to-image Generation)라는 새롭고 지역 인식이 가능한 텍스트-이미지 생성 방법을 제안합니다. RAG는 정확한 레이아웃 구성을 위해 지역 설명에 조건화되어 있으며, 개별 구역에서의 공간적 제어를 가능하게 하는 조합 생성을 통해 사용자에게 더 정밀한 이미지 생성을 제공합니다.

- **Technical Details**: RAG는 일반적인 방식으로 I는 두 가지 하위 작업으로 나뉘어진 다중 지역 생성을 수행합니다: 1) Regional Hard Binding - 각 지역 프롬프트가 올바르게 실행되도록 도와주는 단계입니다. 2) Regional Soft Refinement - 시각적 구분을 버리고 인접 지역 간의 상호작용을 향상시킵니다. 이 방법은 추가적인 모델 조정 없이 다른 프레임워크에 적용될 수 있습니다.

- **Performance Highlights**: 정량적 및 정성적 실험 결과 RAG는 기존의 속성 바인딩(attribute binding) 및 객체 관계(object relationship) 처리에서 뛰어난 성능을 나타냅니다. RAG는 고급의 텍스트-이미지 합성 기준(T2I-CompBench)에서 이전의 비조정 방식들보다 우수합니다.



### Extended multi-stream temporal-attention module for skeleton-based human action recognition (HAR) (https://arxiv.org/abs/2411.06553)
Comments:
          This paper accepted in Computers in Human Behavior Journal

- **What's New**: 이번 논문에서는 graph convolutional networks (GCNs)를 기반으로 한 새로운 접근 방식으로 인간 행동 인식(HAR) 기술을 제안합니다. 기존의 GCN 모델들이 가진 한계를 극복하기 위해 각 계층과 입력 데이터에 따라 변동 가능한 graph 구조를 채택하였습니다.

- **Technical Details**: 저자들은 기존 GCN 모델들이 모든 계층(layer)과 입력 데이터에 대해 동일한 graph 구조를 사용하는 문제를 지적하며, 이를 해결하기 위해 layer-wise adaptive graph structures를 도입했습니다. 이로 인해 GCN의 유연성을 높이고, 모델 성능을 개선할 수 있는 토대를 마련하였습니다.

- **Performance Highlights**: 이 새로운 접근 방식은 기존 GCN 기반 모델에 비해 더 나은 성능을 보여주었으며, 특히 다양한 입력 데이터에 대한 적응력이 크게 향상되었습니다.



### Image Segmentation from Shadow-Hints using Minimum Spanning Trees (https://arxiv.org/abs/2411.06530)
- **What's New**: 이 논문에서는 이미지 세분화(image segmentation)를 위한 새로운 방법을 제안하며, 수천, 수백만 개의 주석이 달린 이미지를 사용하지 않고도 비슷한 품질의 세분화를 달성합니다. 이 방법은 정적 카메라와 다양한 위치의 단일 광원이 있는 이미지 시퀀스를 요구합니다.

- **Technical Details**: 새로운 이미지 세분화 알고리즘은 Delaunay triangulation을 기반으로 하며, 이를 통해 픽셀 그리드를 메시(mesh)로 변환합니다. 알고리즘은 그림자 마스크(shadow masks)를 사용하여 이미지의 불연속성을 감지하고, 가장자리 강도(edge strength)와 방향을 계산하여 세분화를 진행합니다. 또한, 비최대 억제(non-maximum suppression) 및 이중 임계값(double thresholding)을 통해 경량의 윤곽선을 추출합니다.

- **Performance Highlights**: 이 방법은 기존의 이미지 세분화 방법인 FH04 및 최신 딥러닝 모델인 SAM23와 비교하여 많은 경우에서 유사한 결과를 보이며, 특정 조건에서는 SAM23의 과도한 세분화(over-segmentation) 문제를 피하는 장점이 있습니다. 주석 데이터에 의존하지 않고도 높은 품질의 세분화를 실현하는 점에서, 주석 데이터셋을 생성할 대안으로 제시됩니다.



### I2VControl-Camera: Precise Video Camera Control with Adjustable Motion Strength (https://arxiv.org/abs/2411.06525)
- **What's New**: 이번 연구에서는 I2VControl-Camera라는 새로운 카메라 제어 방법을 제안하여 동영상 생성의 제어 정밀도를 크게 향상시키고, 주제의 동작 강도에 대한 조정 가능성을 제공했습니다.

- **Technical Details**: I2VControl-Camera는 카메라 좌표계에서의 점 궤적(point trajectory)을 제어 신호로 사용하며, 고차원 동작 표현을 모델링하여 주제 동작의 강도를 염두에 두고 개발되었습니다. 적응형 아키텍처를 사용하여 기본 모델 구조에 구애받지 않도록 설계되었습니다.

- **Performance Highlights**: 정적 및 동적 장면에서 기존 방법보다 정량적 및 정성적으로 우수한 성능을 입증하였으며, 주제가의 동적 효과와 카메라 제어를 효과적으로 조화시켜 향상된 비디오 품질을 보여주었습니다.



### Offline Handwritten Signature Verification Using a Stream-Based Approach (https://arxiv.org/abs/2411.06510)
Comments:
          Accepted for oral presentation at the International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문에서는 손글씨 서명 검증(HSV) 시스템의 새로운 접근법을 제안합니다. 기존의 정적인 배치 구성 대신, 동적인 데이터 스트림 환경에서 작동할 수 있는 적응형 시스템을 개발하여 서명 검증의 성능을 향상시킵니다.

- **Technical Details**: 제안된 시스템은 SigNet-S 모델을 사용하여 서명 이미지를 특징 벡터로 변환하고, 이를 기반으로 디체미니당 변환(Dichotomy Transformation, DT)을 통해 이진 분류 문제로 변환합니다. 이 시스템은 새로운 서명 샘플이 입력될 때마다 업데이트되고, 과거의 데이터와 결합하여 계속해서 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Support Vector Machine(SVM)을 이용한 표준 접근 방식과 비교했을 때 우수한 성능을 보였습니다. GPDS Synthetic, CEDAR, MCYT 데이터셋에서의 실험 결과는 이 접근법의 효과성을 입증합니다.



### KMM: Key Frame Mask Mamba for Extended Motion Generation (https://arxiv.org/abs/2411.06481)
- **What's New**: 이번 논문에서는 Mamba 아키텍처의 한계를 극복하고 Text-to-Motion (T2M) 생성 성능을 향상시키기 위한 세 가지 기여를 제안합니다. 첫째, Key frame Masking Modeling (KMM)을 도입하여 Mamba의 기억력 소진 문제를 해결하고 주요 동작에 집중하도록 설계되었습니다. 둘째, contrastive learning을 이용한 새로운 접근 방식을 통해 멀티모달 융합 문제를 해결합니다. 마지막으로, 강력한 실험을 통해 BABEL 데이터셋에서 최신 기술을 초월하는 성과를 달성하였습니다.

- **Technical Details**: 이 논문에서는 Key frame Masking Modeling (KMM) 기반의 새로운 아키텍처를 제안하여 Mamba의 기억력 소진 문제를 해결하고자 합니다. KMM은 로컬 밀도와 쌍별 거리 기반으로 주요 프레임을 선택, Mamba의 암묵적 기억 아키텍처에 최적화된 방식으로 동작합니다. 또한, contrastive learning 기법을 사용하여 텍스트 쿼리와의 정렬을 향상시키고, 이를 통해 동작 생성을 더 정확하게 구현할 수 있도록 합니다.

- **Performance Highlights**: BABEL 데이터셋에서의 실험을 통해, 제안된 접근 방식은 기존의 최신 기술 대비 FID를 57% 이상 감소시키고, 파라미터 수를 70% 줄이는 성과를 거두었습니다. 이러한 결과는 Mamba 아키텍처의 멀티모달 융합 및 텍스트-모션 정렬의 제한된 성능을 효과적으로 개선하였음을 보여줍니다.



### Superpixel Segmentation: A Long-Lasting Ill-Posed Problem (https://arxiv.org/abs/2411.06478)
- **What's New**: 이번 연구는 superpixel segmentation의 본질적 문제인 ill-posed 문제를 조명하며, 다양한 기존 방법들이 정규성(regularity)을 다루는 방식에서의 한계를 드러냅니다. 특히, 최근의 deep learning 방법들이 이러한 정규성을 간과하고 있다는 점을 강조합니다. 또한, Segment Anything Model(SAM)을 사용하여 고전적인 superpixel 세분화의 성능을 뛰어넘는 방안을 제시합니다.

- **Technical Details**: superpixel segmentation은 이미지에서 동질적이고 식별 가능한 구역을 생성하는 기술로, 전통적 방법은 정규성 제약(regularity constraint)에 기반을 두고 있습니다. 연구에서는 SLIC(Simpl Linear Iterative Clustering) 기법을 포함한 다양한 방법들이 이 제약을 서로 다르게 해석하고 적용하는 문제를 지적합니다. 본 연구는 SLIC의 변형, 그래프 기반 알고리즘, 최신 컨볼루션 신경망(convolutional neural network) 기반 방법들의 성능 비교도 포함합니다.

- **Performance Highlights**: SAM을 사용하여 특별한 훈련 없이도, 전통적인 superpixel 기법들과 비교해 경쟁력 있는 성능을 발휘하는 결과를 보여주었고, 이는 superpixel segmentation의 재고 필요성을 부각시킵니다. 또한, regularity constraint를 유지하면서 state-of-the-art 결과를 달성하는 방법론을 제안합니다.



### RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration (https://arxiv.org/abs/2411.06463)
- **What's New**: 이번 논문에서는 RL-Pruner라는 새로운 구조적 프루닝 방법을 도입하였다. 이 방법은 강화 학습(Reinforcement Learning)을 기반으로 하여 가지치기 분포를 자동으로 학습하고, 다양한 CNN 구조에서 유연하게 적용될 수 있다.

- **Technical Details**: RL-Pruner는 포스트 트레이닝(post-training) 단계에서 작동하며, 각 프루닝 단계마다 정확한 가지치기 분포를 학습한다. Gaussian 노이즈를 이용하여 정책 분포를 업데이트하고, 각 단계에서 보상 함수를 사용하여 Q 값을 계산한다. 이 방식은 VGGNet, ResNet, GoogLeNet, MobileNet 등 다양한 네트워크에서 호환된다.

- **Performance Highlights**: 실험에서는 RL-Pruner가 CIFAR-100 데이터셋에서 VGG-19에 대해 60% 채널 희소성(sparsity)을 달성하고, GoogLeNet과 MobileNetV3-Large에 대해서는 40% 희소성을 달성하였다. 모든 경우에서 성능 저하는 1% 이하로 유지되었다.



### Dropout the High-rate Downsampling: A Novel Design Paradigm for UHD Image Restoration (https://arxiv.org/abs/2411.06456)
Comments:
          WACV2025

- **What's New**: 이 연구에서는 UHD 이미지 복원 문제를 해결하기 위해 D2Net이라는 새로운 프레임워크를 제안합니다. D2Net은 UHD 이미지를 고해상도로 직접 추론할 수 있도록 하며, 기존의 고속 다운샘플링이나 패치 방식 처리를 필요로 하지 않습니다.

- **Technical Details**: D2Net은 주파수 도메인의 특성을 활용하여 특징의 장기적 의존성을 설정하는 데 중점을 둡니다. Fourier-based Global Feature Extraction (FGFE) 모듈이 특징의 장기적 의존성을 캡처하며, Multi-scale Local Feature Extraction (MLFE) 모듈은 UHD 이미지의 다중 규모 지역 특징을 추출합니다. Adaptive Feature Modulation Module (AFMM)은 인코딩 및 디코딩 특징의 융합을 동적으로 조정합니다.

- **Performance Highlights**: D2Net은 저조도 이미지 향상, 이미지 디헤이징 및 이미지 블러 제거와 같은 세 가지 UHD 이미지 복원 작업에서 광범위한 실험을 수행하였으며, 기존 최첨단 방법들보다 정량적 및 정성적으로 더 나은 결과를 보여주었습니다.



### Improved Video VAE for Latent Video Diffusion Mod (https://arxiv.org/abs/2411.06449)
- **What's New**: 본 연구에서는 Keyframe-based Temporal Compression (KTC) 아키텍처와 Group Causal Convolution (GCConv) 모듈을 통해 영상 VAE의 성능을 향상시키는 새로운 접근 방식을 제안합니다. 이 두 가지는 영상 데이터의 시간적 압축 능력을 개선하고, 프레임 간의 정보 상호작용을 최적화합니다.

- **Technical Details**: KTC 아키텍처는 잠재 공간을 두 개의 가지로 나누어, 하나는 낮은 차원의 이미지 VAE로부터 키프레임의 압축 사전 정보를 완전히 상속받고, 다른 하나는 3D Group Causal Convolution을 통해 시간적 압축을 수행합니다. GCConv는 각 프레임 그룹 내에서 표준 컨볼루션을 사용하여 프레임 간 동등성을 보장하며, 그룹 사이에는 causal logic padding을 적용하여 변화하는 프레임 비디오 처리를 유연하게 합니다.

- **Performance Highlights**: 다섯 개의 벤치마크에서 진행된 실험 결과, 제안된 IV-VAE는 다양한 해상도와 움직임 속도에서 SOTA 비디오 재구성 및 생성 능력을 보였습니다. MotionHD 데이터셋을 재수집하여 1080P 해상도에서 다양한 모션 속도를 포함한 2000개의 비디오를 평가하였으며, 기존 데이터셋으로는 측정하기 어려운 고해상도 비디오 VAE 성능을 평가할 수 있었습니다.



### SamRobNODDI: Q-Space Sampling-Augmented Continuous Representation Learning for Robust and Generalized NODDI (https://arxiv.org/abs/2411.06444)
- **What's New**: 본 논문에서는 q-space 샘플링 증강 기반의 연속 표현 학습 프레임워크(SamRobNODDI)를 제안하여 NODDI의 견고성과 일반화 능력을 향상시키는 방법을 보여줍니다. 이는 기존의 방법들이 고정된 gradient 방향을 요구하는 제약을 해결하고, 다양한 q-space 샘플링 방식에 대해 보다 유연한 성능을 달성합니다.

- **Technical Details**: 제안된 SamRobNODDI는 q-space 샘플링 증강을 활용하여 다양한 diffusion 방향 간의 정보를 완벽히 탐색하는 연속 표현 학습 방식을 도입합니다. 또한, sampling consistency loss를 설계하여 여러 샘플링 방식 간의 결과 출력을 제어하여 성능과 견고함을 향상시킵니다.

- **Performance Highlights**: SamRobNODDI는 기존의 7가지 최첨단 방법들과 18가지의 서로 다른 q-space 샘플링 방식에서 비교 실험을 진행한 결과, 성능, 견고성, 일반화 및 유연성 측면에서 우수한 결과를 보였습니다.



### Local Implicit Wavelet Transformer for Arbitrary-Scale Super-Resolution (https://arxiv.org/abs/2411.06442)
Comments:
          Accepted by BMVC 2024

- **What's New**: 최근 발표된 연구에서, Local Implicit Wavelet Transformer (LIWT)라는 새로운 모델이 제안되었습니다. 이 모델은 이미지의 고주파 세부정보 복원을 개선하기 위해 Discrete Wavelet Transform (DWT)을 사용하여 특징을 분해하여 서로 다른 주파수 정보로 구성된 네 개의 서브밴드로 나누고, Wavelet Enhanced Residual Module (WERM)을 도입하여 고주파 선행 정보를 제공하도록 설계되었습니다.

- **Technical Details**: LIWT는 Wavelet Mutual Projected Fusion (WMPF)과 Wavelet-aware Implicit Attention (WIA)을 결합하여 고주파 정보를 효과적으로 활용하고, 복원 과정에서 주어진 좌표에 따라 주의 맵을 생성합니다. DWT를 통해 추출된 저주파 및 고주파 컴포넌트를 통합하여 고주파 세부정보를 복원합니다. 이러한 방법은 기존의 좌표 기반 앙상블 기법의 한계를 극복하고 지역적 관련성을 고려합니다.

- **Performance Highlights**: LIWT는 다양한 벤치마크 데이터 세트에서 뛰어난 성능을 보이며, 기존의 최신 기법들보다 더 나은 결과를 도출하는 것으로 나타났습니다. 정성적이며 정량적인 결과 모두 LIWT의 우수한 성능을 뒷받침합니다.



### Detecting AutoEncoder is Enough to Catch LDM Generated Images (https://arxiv.org/abs/2411.06441)
- **What's New**: 최근 몇 년간 diffusion models는 이미지 생성의 주요 방법론 중 하나로 자리잡았으나, 이들 모델이 생성한 이미지를 탐지하는 것은 여전히 어려운 과제로 남아있습니다. 본 논문은 Latent Diffusion Models (LDM)로 생성된 이미지를 탐지하기 위한 새로운 방법을 제안하는데, 이는 autoencoder에 의해 도입된 아티팩트를 식별하여 이루어집니다.

- **Technical Details**: 제안된 방법은 실제 이미지와 LDM autoencoder에 의해 재구성된 이미지를 구별하기 위해 탐지기를 훈련시키는 접근 방식을 이용합니다. 이 방법은 생성된 이미지를 직접 훈련하지 않고도 이미지를 탐지할 수 있게 하며, 다른 유사한 접근법과 비교해 학습에 소모되는 계산 비용을 크게 줄이고 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 방법은 허위 긍정(false positive)을 최소화하며 높은 탐지 정확도를 보여줍니다. 이는 생성된 이미지를 탐지하는 데 있어 유망한 도구가 될 것으로 기대됩니다.



### SplatFormer: Point Transformer for Robust 3D Gaussian Splatting (https://arxiv.org/abs/2411.06390)
Comments:
          Code and dataset are publicly available. Project page: this https URL

- **What's New**: 본 논문에서는 OOD(Out-Of-Distribution) 카메라 시나리오에서 3D Gaussian Splatting(3DGS) 방법이 어떻게 성능이 저하되는지를 평가하고, 이를 극복하기 위한 새로운 모델인 SplatFormer를 제안합니다.

- **Technical Details**: SplatFormer는 최초의 포인트 트랜스포머(point transformer) 모델로, Gaussian splats에 특화되어 설계되었습니다. 이 모델은 초기 3DGS 세트를 입력받아 단일 전방 패스를 통해 정제하며, OOD 테스트 뷰에서 발생할 수 있는 아티팩트(artifacts)를 효과적으로 제거합니다.

- **Performance Highlights**: SplatFormer는 극단적인 새로운 뷰에서도 렌더링 품질을 크게 개선하여 최신 기법들에 비해 뛰어난 성능을 보여주었습니다. 기존의 3DGS 정규화 기술들과 다수의 다중 장면 모델보다 성능이 우수하며, 현실 세계의 데이터 세트에서도 일반화 가능한 아티팩트 제거 능력을 보였습니다.



### SAN: Structure-Aware Network for Complex and Long-tailed Chinese Text Recognition (https://arxiv.org/abs/2411.06381)
Comments:
          Published in ICDAR 2023

- **What's New**: 본 논문은 복잡한 문자 인식에 있어 구조 인식(Structure-Aware) 네트워크(SAN)를 제안합니다. SAN은 문자 구성 요소의 계층적 구조를 활용하여 모델 성능을 향상시키는데 초점을 맞추고 있습니다.

- **Technical Details**: 우리가 제안하는 SAN은 보조적인 radical branch(ARB)와 기본 인식 네트워크로 구성됩니다. ARB는 feature extractor의 특징 맵을 개선하고 문자 형태의 인식 정확도를 높입니다. 또한, Tree-Similarity 기반의 weighting 메커니즘을 통해 계층적 표현에서 깊이 정보(depth information)를 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 복잡한 문자와 tail 클래스 문자의 인식 성능을 크게 향상시켜 전체 중국어 텍스트 인식의 정확도를 높이는 것으로 나타났습니다.



### PKF: Probabilistic Data Association Kalman Filter for Multi-Object Tracking (https://arxiv.org/abs/2411.06378)
- **What's New**: 본 논문에서는 측정값과 상태 간의 확률적 데이터 연관(probablistic data association)을 사용하는 새로운 Kalman 필터를 파생합니다. 이 필터는 측정 데이터에 조건한 상태의 posterior 확률 밀도를 근사하기 위해 변분 추론(variational inference) 문제를 형성하며, 알려지지 않은 데이터 연관을 잠재 변수(latent variable)로 간주하고, Expectation Maximization (EM)을 적용하여 Kalman 필터와 동일한 형태를 가진 업데이트 단계를 갖는 필터를 개발하였습니다.

- **Technical Details**: 우리는 측정의 가능성과 관련된 행렬의 permanent를 계산하여 연관 확률을 구할 수 있음을 보여줍니다. 또한 혼란 측정을 포함한 애매한 결정을 처리하기 위해 하이브리드 데이터 연관(hybrid data association) 절차를 제안하여 연관 시간과 추정 정확성 손상을 줄입니다. 우리는 여러 실제 데이터셋(MOT17, MOT20, DanceTrack)에서 우리의 필터가 기존 JPDAF보다 낮은 추적 오차를 달성하며 유사한 처리 속도를 유지하는 것을 입증했습니다.

- **Performance Highlights**: 우리의 새로운 Kalman 필터인 PKF는 다중 객체 추적에서 이전 Kalman 필터 방법보다 더 높은 순위 추적 정확도(HOTA)를 달성하며, MOT17과 MOT20에서 상위 10위에 랭크됩니다. 오프라인 탐지 후, 우리의 알고리즘은 단일 노트북 CPU에서 250+ fps로 추적할 수 있습니다.



### Through the Curved Cover: Synthesizing Cover Aberrated Scenes with Refractive Field (https://arxiv.org/abs/2411.06365)
Comments:
          WACV 2025

- **What's New**: 최근 확장 현실(XR) 헤드셋과 현장 로봇이 환경적 위험으로부터 앞 카메라를 보호하기 위해 커버를 채택했습니다. 그러나 이러한 커버의 표면 불균형은 흐림과 비모수적 왜곡과 같은 광학적 오류를 초래할 수 있습니다.

- **Technical Details**: 이 논문은 SynthCover라는 새로운 뷰 합성 방법론을 소개합니다. SynthCover는 커버의 기하학적 특성을 추정하기 위해 Refractive Field를 사용하여 굴절 광선을 정밀하게 분석하고 계산합니다. 이 방법은 새로운 3D 뷰 합성을 통해 보호 커버를 통해서도 효과적으로 작업할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 SynthCover는 커버에 의해 영향을 받은 장면을 모델링하는 데 있어 기존의 방법들에 비해 렌더링 품질이 크게 향상되었음을 입증했습니다. 다양한 표면 곡률을 가진 커버에서 포착된 합성 시퀀스에 대해서도 잘 조정되는 성능을 보여줍니다.



### Layer-Wise Feature Metric of Semantic-Pixel Matching for Few-Shot Learning (https://arxiv.org/abs/2411.06363)
- **What's New**: 본 연구에서는 이미지 쌍의 유사성을 보다 정확하게 평가하기 위해 새로운 방법인 Layer-Wise Features Metric of Semantic-Pixel Matching (LWFM-SPM)을 제안합니다. 이 방법은 향상된 성능을 위해 Layer-Wise Embedding (LWE) 모듈과 Semantic-Pixel Matching (SPM) 모듈을 포함하고 있습니다.

- **Technical Details**: LWE 모듈은 서로 다른 이미지 쌍의 레이어별 상관관계를 정교하게 추출하고, SPM 모듈은 의미론적 임베딩을 기반으로 한 중요한 픽셀 정렬 과정을 통해 픽셀 간 유사성을 측정합니다. 최종적으로 이 모듈들을 통합하여 이미지 쌍을 입력으로 받아 유사성 점수를 출력하는 엔드-투-엔드 방식으로 설계됩니다.

- **Performance Highlights**: 다수의 상태-공식 분류 벤치마크(miniImageNet, tieredImageNet, CUB-200-2011, CIFAR-FS)에서 LWFM-SPM이 경쟁력 있는 성능을 달성함을 보였습니다. 이를 통해 기존의 메트릭 기반 방법들이 가진 한계점을 극복하고 유사성 평가의 정확성을 높이는 데 기여합니다.



### Classification in Japanese Sign Language Based on Dynamic Facial Expressions (https://arxiv.org/abs/2411.06347)
Comments:
          2024 IEEE 13th Global Conference on Consumer Electronics (GCCE 2024)

- **What's New**: 이 논문에서는 일본 수화(JSL) 인식에 초점을 맞춘 새로운 방법을 제안하며, 특히 비수동적 표식(non-manual markers)인 얼굴 표정(facial expressions)을 통해 의사소통의 정확성을 높이는 것을 목표로 한다.

- **Technical Details**: 제안된 방법은 신경망(neural network)을 활용하여 얼굴 특징을 분석하고 문장의 유형을 분류한다. JSL 비디오를 입력으로 받아 얼굴 랜드마크(facial landmarks)를 감지하고, 이를 기반으로 세 가지 문장 형태(긍정문, Yes/No 질문, WH 질문)로 분류한다. 모델로는 OpenPose, MediaPipe, Dlib이 사용되며, 최종적인 입력으로는 정규화된 랜드마크 데이터가 사용된다.

- **Performance Highlights**: 제안된 방법은 96.05%의 분류 정확도(classification accuracy)를 달성했으며, OpenPose 모델을 사용할 경우 특히 높은 정확도를 기록하였다. 이는 복잡한 배경과 갑작스러운 움직임에도 강력한 얼굴 랜드마크 탐지 성능 덕분이다.



### CityGuessr: City-Level Video Geo-Localization on a Global Sca (https://arxiv.org/abs/2411.06344)
Comments:
          Accepted to ECVA Eurpoean Conference on Computer Vision(ECCV) 2024

- **What's New**: 이번 논문에서는 전세계 비디오 위치 식별 문제를 새롭게 정의하고, 166개 도시에서 수집한 68,269개의 비디오로 구성된 대규모 데이터셋 CityGuessr68k를 소개합니다. 이 데이터셋은 비디오 위치 식별을 위한 기계 학습 모델 훈련에 필수적입니다.

- **Technical Details**: 우리는 전 세계 비디오 지리적 식별을 위해 트랜스포머 기반 아키텍처를 사용한 분류 기반 접근법을 제안합니다. 주요 구성 요소로는 장면 예측을 위한 Self-Cross Attention 모듈과 텍스트 라벨 정렬(TextLabel Alignment) 전략이 있습니다. 이를 통해 비디오의 지리적 정보와 텍스트 정보를 효과적으로 통합하여 모델의 예측 성능을 높입니다.

- **Performance Highlights**: 제안된 방법론은 CityGuessr68k 데이터셋과 Mapillary(MSLS) 데이터셋에서의 성능 평가를 통해 효과성을 입증하였습니다. 이를 바탕으로 비디오 지리적 식별의 새로운 기준 선을 제시할 수 있습니다.



### SEM-Net: Efficient Pixel Modelling for image inpainting with Spatially Enhanced SSM (https://arxiv.org/abs/2411.06318)
Comments:
          Accepted by WACV 2025

- **What's New**: 이번 연구에서는 이미지 복원을 위한 새로운 모델인 SEM-Net을 제안합니다. 이 모델은 손상된 이미지를 픽셀 수준에서 복원하며, 장거리 의존성(long-range dependencies, LRDs)을 효과적으로 포착할 수 있습니다. SEM-Net은 두 가지 혁신적인 모듈인 Snake Mamba Block(SMB)과 Spatially-Enhanced Feedforward Network(SEFN)을 도입하여 공간적 연속성을 강화합니다.

- **Technical Details**: SEM-Net은 Spatially-Enhanced SSM(상태공간모델) 아키텍처로, 앞서 설명한 SMB와 PE layer를 통해 이미지를 두 방향으로 효과적으로 스캔하며 의존성을 유지합니다. SBDM(Snake Bi-Directional Modelling)을 사용하여 장거리 위치 인식을 강화하며, SEFN을 통해 로컬 공간 의존성을 보완합니다. 이 방식은 CNN 및 Transformer 기반 방법보다 더 효율적으로 장거리 장면을 복구합니다.

- **Performance Highlights**: SEM-Net은 CelebA-HQ 및 Places2 데이터셋에서 기존 최첨단 이미지 복원 기법들과 비교하여 더 우수한 성능을 보여주며, 모션 블러링(motion deblurring)에서도 최첨단 성능을 달성하였습니다. 연구에 따르면, SEM-Net은 LRD 및 공간적 일관성을 더욱 효과적으로 포착하여 이미지 복원 질을 높였습니다.



### NeuReg: Domain-invariant 3D Image Registration on Human and Mouse Brains (https://arxiv.org/abs/2411.06315)
Comments:
          15 pages, 5 figures, 5 tables

- **What's New**: 이번 연구에서는 NeuReg라는 Neuro-inspired 3D image registration 아키텍처를 제안합니다. 이 아키텍처는 domain invariance 기능을 가지고 있어 다양한 3D 뇌 이미징 모달리티 간의 변화를 잘 캡처할 수 있습니다.

- **Technical Details**: NeuReg는 Swin Transformer 블록을 encoder로 사용하여 domain-agnostic representations을 생성하며, 이를 통해 여러 뇌 이미징 도메인에 걸쳐 강력한 성능을 발휘합니다. 이 아키텍처는 mammalian visual system에서 영감을 얻어 개발되었습니다.

- **Performance Highlights**: NeuReg는 iSeg-2017 및 OASIS-3와 같은 다중 도메인 공개 데이터셋에서 기존의 baseline deep learning image registration 모델보다 우수한 성능을 보여줍니다. 특히, cross-domain 데이터셋에서 'source-only' 도메인에서 훈련된 모델이 'unseen' target 도메인에서 높은 성능을 기록했습니다.



### Adaptive Aspect Ratios with Patch-Mixup-ViT-based Vehicle ReID (https://arxiv.org/abs/2411.06297)
- **What's New**: 이번 연구에서는 차량 재식별 (ReID) 시스템의 정확성을 향상시키기 위해 다양한 종횡비(aspect ratio)를 적절히 처리하는 새로운 Vision Transformer (ViT) 기반의 프레임워크를 제안합니다. 이 프레임워크는 여러 종횡비에 대해 훈련된 모델들을 융합하여 성능을 극대화합니다.

- **Technical Details**: 제안된 방법은 다음과 같은 주요 요소로 구성됩니다: 1) VeRi-776 및 VehicleID 데이터셋을 사용하여 종횡비가 성능에 미치는 영향을 분석합니다. 2) 공간적 주의 점수에 기반한 패치-와이즈 믹스업(strategy) 전략을 도입하고, 물체의 종횡비와의 정렬을 개선하기 위해 불균일 보폭(uneven stride) 방식을 구현합니다. 3) 동적 특징 융합(Dynamic Feature Fusion) 네트워크를 통해 모델의 강인성을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 두 데이터셋 모두에서 최신 트랜스포머 기반 접근 방식보다 뛰어난 성능을 보였으며, 이미지당 추론 시간은 최소한의 증가만 있었습니다.



### Hidden in Plain Sight: Evaluating Abstract Shape Recognition in Vision-Language Models (https://arxiv.org/abs/2411.06287)
- **What's New**: 새로운 데이터셋인 IllusionBench가 소개되었습니다. 이 데이터셋은 현재의 Vision-Language Models (VLMs)의 형태 인식을 평가하기 위해 설계되었으며, 복잡한 장면에서 시각적 요소의 배열로 형태 정보를 표현합니다.

- **Technical Details**: IllusionBench는 사람 annotators가 쉽게 식별할 수 있는 형태 정보를 제공하지만, 현재의 VLMs는 이러한 형태를 인식하는 데 어려움을 겪고 있습니다. VLM에 대한 제로샷(Zero-Shot), 퓨샷(Few-Shot), 그리고 미세 조정(Fine-Tuning) 세 가지 시나리오를 통해 평가합니다.

- **Performance Highlights**: 현재 VLMs는 형태 인식에 한계가 있으며, 장면 구성 요소에 주로 집중하고 있습니다. 이는 인간의 시각적 강건성(robustness)을 위한 추상적 형태 인식 능력이 부족하다는 것을 나타냅니다.



### Crowd3D++: Robust Monocular Crowd Reconstruction with Upright Spac (https://arxiv.org/abs/2411.06232)
Comments:
          14 pages including reference

- **What's New**: 이 논문은 단일 이미지에서 수백 명의 3D 포즈, 형태 및 위치를 재구성하기 위한 새로운 방법인 Crowd3D와 그 확장판인 Crowd3D++를 제안합니다. 이는 카메라 매개변수가 불확실한 상황에서도 전 세계적으로 일관된 재구성이 가능하도록 설계되었습니다.

- **Technical Details**: Crowd3D는 Human-scene Virtual Interaction Point (HVIP)라는 개념을 활용하여 복잡한 3D 인간 위치 지정을 2D 픽셀 로컬라이제이션으로 변환하고, Robust한 카메라 및 지면 추정과 함께 글로벌 일관성을 달성합니다. Crowd3D++는 인지된 직립 공간과 지면 인식 정규화 변환을 통해 카메라 매개변수와 크로핑 작업의 영향을 제거합니다.

- **Performance Highlights**: Crowd3D++는 새로운 장면에 대해 임시 최적화 없이도 일반화가 가능하며, 733개의 기가픽셀 이미지에서 100K 이상의 라벨이 지정된 인체에 대한 대규모 벤치마크 데이터셋인 LargeCrowd를 구축하였습니다. 또한 가상 데이터셋 SyntheticCrowd도 제작하여 다양한 카메라 매개변수 하에서 정량적인 평가를 진행할 수 있습니다.



### Text2CAD: Text to 3D CAD Generation via Technical Drawings (https://arxiv.org/abs/2411.06206)
- **What's New**: 이 논문은 사용자 요청 및 사양으로부터 산업용 CAD 모델을 자동으로 생성할 수 있는 새로운 프레임워크인 Text2CAD를 소개합니다. 기존의 수작업 방식의 한계를 극복하고, 텍스트 설명을 통해 CAD 모델 생성을 간소화하며, 효율성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: Text2CAD는 안정적 확산 모델(stable diffusion models)을 활용하여 사용자 텍스트 설명을 정밀한 CAD 모델로 변환합니다. 이 프레임워크는 먼저 사용자의 텍스트 설명을 바탕으로 이소메트릭(isometric) 이미지를 생성하고, 이후 이 이미지를 기반으로 다양한 정투영(orthographic) 뷰, 즉 상단, 정면, 측면을 생성합니다. 이로써 3D CAD 모델의 세부 정보가 제공되고, 물리적 및 치수의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, Text2CAD는 고품질 3D CAD 모델로 정확하게 변환되는 기술 도면을 생성할 수 있음이 입증되었습니다. 이러한 결과는 CAD 자동화의 혁신적인 가능성을 보여주며, 사용자 요구에 적극적으로 대응하는 모델 생성에 기여할 수 있습니다.



### Multi-object Tracking by Detection and Query: an efficient end-to-end manner (https://arxiv.org/abs/2411.06197)
- **What's New**: 본 논문에서는 기존의 tracking by detection과 tracking by query 방식을 통합하여 새로운 'tracking-by-detection-and-query' 패러다임을 제안합니다. 이를 위해 Learnable Associator를 도입하고, 개체 쿼리 간의 정보 상호작용 모듈과 콘텐츠-위치 정렬 모듈을 개발하여, 쿼리에서 직접적으로 추적 결과를 복호화합니다.

- **Technical Details**: LAID는 두 가지 모듈, 즉 Basic Information Interaction (BII) 및 Content-Position Alignment (CPA) 모듈을 활용하여 쿼리 내용을 효과적으로 상호작용시키고 위치 정보를 정렬합니다. 이 과정을 통해 가져온 완전한 상호작용된 개체 쿼리는 Transformer decoder 레이어를 통해 해석됩니다. 추가로, LAID는 활용된 사전 훈련된 탐지기를 바탕으로 Learnable Associator를 추가하고 기존의 추적 패러다임을 유지합니다.

- **Performance Highlights**: LAID는 DanceTrack와 SportsMOT 데이터셋에서 테스트한 결과, 이전의 최첨단 heuristics 방법인 Hybrid-SORT에 비해 HOTA 메트릭에서 3.9%, IDF1 메트릭에서 6.1% 우수한 성능을 보였으며, 스포츠 MOT 데이터셋에서도 최고의 점수를 기록했습니다. 훈련 효율성을 유지하면서도 강력한 추적 능력을 갖추고 있는 LAID는 MOT 분야의 미래 지향적인 방향을 제시합니다.



### LSSInst: Improving Geometric Modeling in LSS-Based BEV Perception with Instance Representation (https://arxiv.org/abs/2411.06173)
Comments:
          Accepted by 3DV 2025

- **What's New**: 본 논문에서는 카메라만을 이용한 3D 물체 탐지에서 Bird-Eye-View (BEV) 표현을 활용한 새로운 두 단계(object detection) 프레임워크인 LSSInst를 제안합니다. 이 프레임워크는 BEV 및 instance representations를 통합하여 정밀한 3D 객체 정보를 제공할 수 있도록 설계되었습니다.

- **Technical Details**: LSSInst는 pixel-level 세부 특징을 활용하여 기존 LSS 기반 BEV 네트워크에 유연하게 통합될 수 있습니다. 이 연구에서 제안된 instance adaptor는 BEV에서 instance semantics 간의 일관성을 유지하도록 설계되었습니다. 실험을 통해 nuScenes 데이터셋에서 LSSInst의 mAP가 기존 방법들보다 월등하게 개선되었음을 확인하였습니다.

- **Performance Highlights**: LSSInst는 BEVDet보다 5.0%, BEVDepth보다 2.2%, BEVStereo보다 2.6% 높은 성능을 보였으며, 최신 LSS 기반 방법인 SOLOFusion보다도 1.6% 향상된 결과를 나타냈습니다.



### Aquila-plus: Prompt-Driven Visual-Language Models for Pixel-Level Remote Sensing Image Understanding (https://arxiv.org/abs/2411.06142)
- **What's New**: 최근 비전 언어 모델(vision language models, VLMs)의 발전으로 시각-언어 통합(visual-language integration)이 크게 향상되었습니다. 특히 본 연구에서는 Aquila-plus라는 새로운 mask-text instruction tuning 방법을 제안하여 기존의 원거리 감지 비전 언어 모델(Remote Sensing Vision Language Models, RSVLMs)의 한계를 극복하고 픽셀 수준의 시각적 이해(pixel-level visual understanding)를 달성합니다.

- **Technical Details**: Aquila-plus는 세밀한 마스크 영역(mask regions)을 언어 지침에 포함시키는 방식으로 RSVLMs의 능력을 확장합니다. 연구팀은 10만 개의 샘플이 포함된 마스크 지역-텍스트 데이터셋을 정교하게 구축하였으며, 대규모 언어 모델(large language model, LLM)에 픽셀 수준의 표현을 주입하여 비전-언어 모델을 설계하였습니다. 구체적으로, Aquila-plus는 시각 인코더로 컨볼루션 CLIP(convolutional CLIP)를 사용하고, 고해상도 입력에서 정밀한 시각 마스크 특징을 추출하기 위해 마스크 인식 비주얼 추출기(mask-aware visual extractor)를 활용합니다.

- **Performance Highlights**: 실험 결과, Aquila-plus는 다양한 영역 이해(region understanding) 작업에서 기존 방법들보다 우수한 성능을 보여주었으며, 픽셀 수준의 지침 튜닝에서 새로운 가능성을 입증했습니다.



### Scalable, Tokenization-Free Diffusion Model Architectures with Efficient Initial Convolution and Fixed-Size Reusable Structures for On-Device Image Generation (https://arxiv.org/abs/2411.06119)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 Diffusion Model을 구현하기 위한 새로운 하드웨어 친화적인 신경망 아키텍처를 제안합니다. 기존의 Vision Transformer와 U-Net 아키텍처의 장점을 살리면서도, 보다 효율적인 하드웨어 구현을 가능하게 하는 구조를 설계하였습니다.

- **Technical Details**: 제안하는 아키텍처는 고정 크기의 재사용 가능한 transformer 블록을 중심으로 구성되어 있으며, 토큰화(tokenization)와 위치 임베딩(positional embedding)이 필요하지 않은 설계를 특징으로 합니다. 두 가지 구성(Configuration I 및 II)을 통해 다른 매개변수 수를 확장할 수 있으며, 각 구성에 따라 계산 복잡성을 조정하면서도 높은 이미지 생성 품질을 유지합니다.

- **Performance Highlights**: 모델은 조건부 및 비조건부 이미지 생성 작업에서 경쟁력 있는 성능을 보여주며, CelebA 데이터셋에서 비조건부 이미지 생성 시 FID 점수 1.6을 기록하여 최신 기술과 비교하여 우수한 결과를 보였습니다.



### Personalize to generalize: Towards a universal medical multi-modality generalization through personalization (https://arxiv.org/abs/2411.06106)
- **What's New**: 이번 논문은 개인 맞춤형 의학(Personalized Medicine)과 다중 모달 의료 이미지 분석(Multi-modal Medical Image Analysis) 간의 연결을 목표로 합니다. 특히, 개인 단위에서 생물학적 정보를 이용하여 개인화된 불변 표현(Personalized Invariant Representation) 을 도출하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 연구에서는 각 개인의 의료 이미지를 나타내는 표기법과 인코더(Encoder) 및 디코더(Decoder) 구조를 정의합니다. 제안된 방법은 개인화된 불변 표현 $oldsymbol{X}_h$ 을 학습하기 위해 생물학적 사전 지식(Previous Knowledge)을 이용하는 구조적 제약조건(Structural Constraints)을 포함합니다. 이를 통해 다중 모달 간의 일반화 가능성을 탐구합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 본 방법은 이질적 일반화(Heterogeneous Generalization) 및 동질적 일반화(Homogeneous Generalization) 등 다양한 시나리오에서 기존의 최신 기술(State-of-the-Art) 방법보다 뛰어난 성능을 보여주었습니다.



### LT-DARTS: An Architectural Approach to Enhance Deep Long-Tailed Learning (https://arxiv.org/abs/2411.06098)
- **What's New**: 이 논문에서는 Deep long-tailed recognition을 위한 새로운 접근법인 Long-Tailed Differential Architecture Search (LT-DARTS)를 제안합니다. 기존 DARTS 방법들이 long-tailed 데이터에서 좋은 성능을 내지 못하는 문제를 해결하기 위해 아키텍처 개선을 목표로 합니다.

- **Technical Details**: LT-DARTS는 long-tailed 데이터에 적합한 아키텍처를 탐색하기 위해 새로운 검색 공간(search space)과 두 가지 혁신적인 convolution 운영(Operation)을 시행합니다. Equiangular Tight Frame (ETF) 분류기를 도입하여 biased search process를 완화하고 성능 붕괴(performance collapse)를 방지합니다.

- **Performance Highlights**: 실험 결과, LT-DARTS는 기존의 전문가 설계 네트워크보다 뛰어난 성능을 발휘하며, long-tailed 문제에 최적화된 기존 방법들과도 원활하게 통합되어 성능을 지속적으로 개선합니다. 또한, 간단한 개선만으로도 state-of-the-art 결과를 달성함을 보여줍니다.



### Pattern Integration and Enhancement Vision Transformer for Self-Supervised Learning in Remote Sensing (https://arxiv.org/abs/2411.06091)
- **What's New**: 본 논문에서는 자가 감독 학습(self-supervised learning, SSL) 기법을 바탕으로 한 새로운 프레임워크인 Pattern Integration and Enhancement Vision Transformer (PIEViT)를 소개합니다. PIEViT는 원거리 센싱 이미지에서 유사한 객체를 자동으로 집합하여 다양한 지리적 패턴을 통합하여 향상된 피처 표현을 가능하게 합니다.

- **Technical Details**: PIEViT는 교사-학생 아키텍처를 활용하여 이미지 수준과 패치 수준 작업을 동시에 처리합니다. Geospatial Pattern Cohesion (GPC) 모듈을 통해 패치의 자연적인 군집화를 탐색하고 개별 특성의 차별화를 강화하며, Feature Integration Projection (FIP) 모듈은 지리적으로 군집화된 패치를 사용하여 마스킹된 토큰 복원을 정제합니다.

- **Performance Highlights**: PIEViT는 객체 탐지, 토지 피복 분류 및 변화 탐지와 같은 여러 하부 작업에서 검증되었으며, 기존의 자가 감독 기준 방법에 비해 뛰어난 성능을 보였습니다. 특히 객체 탐지와 토지 피복 분류, 변화 탐지에서 SOTA(results of state-of-the-art) 성과를 달성하여 원거리 센싱 이미지 해석 작업에 대한 강력한 일반화 능력과 전이 가능성을 보여줍니다.



### Aquila: A Hierarchically Aligned Visual-Language Model for Enhanced Remote Sensing Image Comprehension (https://arxiv.org/abs/2411.06074)
- **What's New**: 최근, 대형 비전 언어 모델(VLMs)은 비주얼 인스트럭션 튜닝을 통해 시각적 언어 능력에서 중요한 발전을 이루었습니다. 그러나 기존의 원격 감지 비전 언어 모델(RSVLMs)은 복잡한 원격 감지 장면의 특성을 포착하는 데 한계가 있었습니다. 본 논문에서는 Aquila라는 진보된 비전 언어 기초 모델을 소개합니다.

- **Technical Details**: Aquila는 고해상도 이미지 입력을 지원하고 다중 스케일 시각적 특성을 집계하는 학습 가능한 계층적 공간 특성 통합(Hierarchical Spatial Feature Integration, SFI) 모듈을 도입합니다. 이 모듈은 복잡한 시각적 정보를 세밀하게 표현하는 기능을 제공합니다. 또한 SFI 모듈은 대형 언어 모델(LLM)의 여러 층에 통합되어 심층적인 비주얼-언어 특성 정렬(deep visual-language feature alignment)을 달성합니다.

- **Performance Highlights**: Aquila는 높은 해상도와 다중 스케일 입력을 통해 세부적인 시각적 효과를 포착하고, 특성 정렬을 강화하여 이미지 텍스트 데이터에서 학습 능력을 크게 향상시킵니다. 광범위한 정량적 실험과 질적 분석을 통해 Aquila의 우수한 성능을 검증하였습니다.



### GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection (https://arxiv.org/abs/2411.06071)
Comments:
          28 pages, 33 figures

- **What's New**: 새로운 제안된 방법 GlocalCLIP는 전통적인 이미지 분류 및 파라미터 조정 방식과 달리, 글로벌 및 로컬 프롬프트를 명확하게 분리하여 상호 보완적으로 학습합니다. 이 방식은 특정 객체에 의존하지 않으면서 일반적인 정상 및 비정상 패턴을 효과적으로 포착할 수 있게 해줍니다.

- **Technical Details**: GlocalCLIP는 이전에 훈련된 CLIP 모델을 기반으로 하고, 개체 비특이적인 glocal 시맨틱 프롬프트 구조를 디자인하여 모든 정상 및 비정상 경우에 적용 가능하게 합니다. 텍스트 인코더에서는 학습 가능한 토큰을 삽입하여 세부적인 텍스트 조정을 위한 deep-text 프롬프트 튜닝을 사용합니다. 비전 인코더에서는 V-V attention 레이어를 적용하여 로컬 이미지의 세부 특징을 캡처합니다.

- **Performance Highlights**: GlocalCLIP는 15개의 실제 세계 데이터 세트에서 실험을 통해 향상된 비정상 감지 성능과 강력한 일반화를 입증하였으며, 기존 방법에 비해 우수한 성능을 나타냈습니다.



### AI-Driven Stylization of 3D Environments (https://arxiv.org/abs/2411.06067)
- **What's New**: 본 논문에서는 NeRFs 및 3D Gaussian Splatting과 같은 혁신적인 3D 표현 방식을 활용하여 3D 원시 객체의 장면을 고해상도 3D 장면으로 스타일화하는 방법을 논의합니다. 사용자는 기본 원시 형상을 그리며 스타일 선호를 입력할 수 있는 사용자 친화적인 인터페이스를 통해 효과적으로 공간을 마련하고 재스타일링할 수 있습니다.

- **Technical Details**: 제안된 파이프라인은 크게 세 가지 구성 요소로 이루어져 있습니다: 1) primitives stylizer - 단일 뷰 이미지를 입력받아 스타일화된 이미지를 생성하는 모듈, 2) mesh generator - 스타일화된 이미지를 기반으로 텍스처드 메쉬를 생성하는 모듈, 3) scene integrator - 생성된 메쉬를 대상 장면에 통합하는 모듈. 스타일화 과정에서 InstructPix2Pix 모델을 사용하며, 메쉬 생성에는 Convolutional Reconstruction Model (CRM)과 Gaussian Reconstruction Model (GRM) 과정을 이용합니다.

- **Performance Highlights**: 실험 결과 CRM과 GRM의 결합을 통해 높은 품질의 3D 모델을 효과적으로 생성할 수 있었으며, SIGNeRF를 통해 새로운 메쉬를 기존 장면에 통합하여 다양한 관점에서의 일관성을 유지했습니다. 이러한 접근법은 3D 환경을 사용자 맞춤형으로 간편히 디자인할 수 있는 가능성을 제시하며, 3D 디자인이 더 넓은 청중에게 접근 가능하도록 하는 데 기여합니다.



### An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models (https://arxiv.org/abs/2411.06048)
- **What's New**: 이 논문은 Spatial-MM이라는 새로운 VQA 데이터셋을 소개하여 Large Multimodal Models (LMMs)의 공간 이해 및 추리 능력을 종합적으로 연구하였습니다. 연구를 통해 LMMs가 인간의 시점에서 질문에 대한 답변을 잘 하지 못하고 복잡한 multi-hop 문제에 대한 모델 성능이 향상되지 않는 등의 문제점을 발견했습니다.

- **Technical Details**: Spatial-MM 데이터셋은 Spatial-Obj와 Spatial-CoT 두 개의 하위 집합으로 구성됩니다. Spatial-Obj는 이미지 내의 하나 또는 두 개의 객체 간의 공간적 관계를 포괄하는 다선택 질문을 포함하며, 2000개의 질문을 통해 LMMs의 공간적 추리를 평가합니다. Spatial-CoT는 개방형 multi-hop 질문을 제공합니다.

- **Performance Highlights**: 분석 결과, (i) 바운딩 박스 및 씬 그래프가 LMMs의 공간적 추리를 상당히 향상시키고, (ii) LMMs는 이미지에 대한 인간의 시각에서 제기된 질문에 대해 더 어렵게 반응하며, (iii) chain of thought (CoT) 프롬프트가 복잡한 multi-hop 질문의 모델 성능에 기여하지 않는 것으로 나타났습니다. 전반적으로 LMMs는 복잡한 공간적 추리에 비해 기초적인 객체 탐지에 훨씬 더 강한 성능을 보였습니다.



### PointCG: Self-supervised Point Cloud Learning via Joint Completion and Generation (https://arxiv.org/abs/2411.06041)
- **What's New**: 이 논문은 masked point modeling (MPM)과 3D-to-2D generation을 통합한 새로운 self-supervised learning 프레임워크인 PointCG를 제안합니다. 이 프레임워크는 Hidden Point Completion (HPC) 모듈과 Arbitrary-view Image Generation (AIG) 모듈을 포함하여 3D 객체를 효과적으로 인식할 수 있도록 합니다.

- **Technical Details**: PointCG는 두 가지 주요 모듈을 통합하여 동작합니다. HPC 모듈은 input으로부터 보이는 포인트를 이용해 전체 형태를 완성하고, AIG 모듈은 보이는 포인트의 표현을 기반으로 2D 이미지를 생성합니다. 이 과정에서 cross-modal feature alignment를 통해 포인트 클라우드와 이미지의 feature space를 일치시키고, encoder의 학습 초점을 다시 맞추어줍니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법이 다양한 downstream tasks에서 기존 방법들보다 우수한 성능을 보임을 입증하였습니다. 제안된 모듈은 정보가 제한된 포인트로부터 효과적으로 학습할 수 있게 해주어, masked point modeling과 arbitrary-view image generation의 효율성을 크게 향상시킵니다.



### Dynamic Textual Prompt For Rehearsal-free Lifelong Person Re-identification (https://arxiv.org/abs/2411.06023)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 논문에서는 직관적인 텍스트 설명을 활용하여 ReID 모델이 도메인 불변 특징을 학습하도록 유도하는 새로운 접근 방식을 제안합니다. 이는 데이터 샘플을 보존하지 않고도 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 제안한 방법은 Dynamic Textual Prompt (DTP) 프레임워크를 기반으로 하며, 동적 프롬프트 융합 모듈(Dynamic Prompt Fusion, DPF), 텍스트-비주얼 특징 정렬 모듈(Text-Visual Feature Alignment, TFA), 학습 가능한 지식 증류 모듈(Learnable Knowledge Distillation, LKD)로 구성됩니다. 이들 모듈은 ReID 모델이 다양한 도메인을 통합하여 일관된 의미 공간에 이미지를 임베딩하도록 안내합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 DTP 방법은 mAP/rank1 기준으로 기존의 데이터 리허설 방법을 15.9%/14.8% 및 보지 못한 데이터셋에 대해 17.4%/17.9% 성능 향상을 보였으며, 여러 설정에서 SOTA(State-of-the-Art) 모델을 초월하는 성과를 달성했습니다.



### GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting (https://arxiv.org/abs/2411.06019)
Comments:
          Project page at this https URL

- **What's New**: 본 논문에서는 3D Gaussian Splatting (3DGS)의 메모리 문제를 해결하기 위해 GaussianSpa라는 최적화 기반의 간소화 프레임워크를 소개합니다. 이 프레임워크는 Gaussians의 수를 줄이면서도 고품질의 3DGS 모델을 유지합니다.

- **Technical Details**: GaussianSpa는 3DGS 교육 과정에서 Gaussians의 수를 제약 조건화된 최적화 문제로 설정하고, 이 문제를 '최적화' 단계와 '희소화' 단계의 두 개의 독립적인 하위 문제로 나누어 해결합니다. 이 과정을 통해 Gaussian은 점진적으로 강한 희소성을 부여받습니다.

- **Performance Highlights**: GaussianSpa는 기존 방법들에 비해 뛰어난 성능을 보여주며, 10배 적은 수의 Gaussians로도 Deep Blending 데이터셋에서 평균 0.9 dB의 PSNR 개선을 달성했습니다. 다양한 실험을 통해 작품의 시각적 품질이 크게 향상되었음을 입증하였습니다.



### A Modular Conditional Diffusion Framework for Image Reconstruction (https://arxiv.org/abs/2411.05993)
- **What's New**: 이번 연구에서는 Diffusion Probabilistic Models (DPMs)를 활용하여 다양한 블라인드 이미지 복원(Blind Image Restoration, IR) 작업을 처리하는 새로운 모듈식 확산 확률적 IR 프레임워크(Framework)인 DP-IR을 제안합니다. 이 프레임워크는 특정 IR 작업에 관련된 작은 모듈만 추가적으로 학습합니다.

- **Technical Details**: DP-IR 프레임워크는 기존의 최첨단 IR 네트워크와 생성적 DPMs의 성능 장점을 결합하며, 총 0.7M 파라미터만 추가적으로 필요합니다. 제안된 아키텍처는 성능 손실 없이 뉴럴 기능 평가(Neural Function Evaluations, NFEs)를 최소 4배 주기적으로 줄일 수 있는 샘플링 전략을 제공합니다.

- **Performance Highlights**: 제안한 방법은 burst JDD-SR, 동적 씬 디블러링(Dynamic Scene Deblurring), 슈퍼 해상도(Super-Resolution)와 같은 네 가지 벤치마크에서 성능을 평가한 결과, 인식 품질에서 기존 방법을 초월했으며, 충실도 메트릭(Fidelity metrics) 또한 경쟁력 있는 성능을 유지함을 확인했습니다.



### Utilisation of Vision Systems and Digital Twin for Maintaining Cleanliness in Public Spaces (https://arxiv.org/abs/2411.05964)
Comments:
          Accepted for the ICCVG 2024: International Conference on Computer Vision and Graphics, Poland

- **What's New**: 이 논문은 첨단 비전 감시 시스템과 Digital Twin 기술을 활용하여 대중교통 분야의 청소 관리 시스템을 개발하는 내용을 다룹니다. 사례로는 기차역이 사용되며, 이를 통해 청결 상태를 실시간으로 모니터링하고, 오염물질을 자동으로 감지하는 방법을 제시합니다.

- **Technical Details**: Digital Twin 기술을 활용해 기차역의 3D 모델을 생성하였고, 쓰레기 감지기(litter detector), 쓰레기통 점유 수준 감지기(bin occupancy level detector), 오염 세그멘테이션(stain segmentation) 및 사람 감지(human detector)와 같은 컴포넌트를 포함한 시스템을 구현했습니다. 이 모든 과정은 Nvidia Omniverse Isaac Sim 시뮬레이터를 통해 이루어졌습니다.

- **Performance Highlights**: 시스템은 실시간으로 청결 여부를 모니터링하고, 청소 서비스에 데이터를 전달하여 청소 업무를 최적화합니다. 초기 평가 결과, 이 시스템을 통해 청결도를 높이고 청소 비용을 절감할 수 있을 것으로 기대됩니다.



### Aligned Vector Quantization for Edge-Cloud Collabrative Vision-Language Models (https://arxiv.org/abs/2411.05961)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 LLaVA-AlignedVQ라는 엣지-클라우드 협업 기반의 VQA 시스템을 소개하며, Aligned Vector Quantization (AlignedVQ) 알고리즘을 통해 중간 특성을 효율적으로 압축하고 정확성을 유지하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: LLaVA-AlignedVQ는 중간 특성의 압축 비율을 약 1365배에 달하게 하여 데이터 전송 오버헤드를 96.8% 감소시키고, 원래 모델의 정확성과 비교하여 -2.23%에서 +1.6% 범위 내에서 높은 정확성을 유지합니다. 이 시스템은 초기 층을 엣지에서 로컬로 처리하고 나머지 층과 VLM 구성 요소를 클라우드에서 처리하는 파티셔닝 실행 방식을 채택하고 있습니다.

- **Performance Highlights**: LLaVA-AlignedVQ는 실행 속도를 2-15배 향상시키면서 높은 정확성을 유지합니다. 엣지에서 NVIDIA Jetson AGX Xavier를 사용하여 첫 번째 블록과 양자화 모듈을 실행하고 나머지 부분은 A100 GPU를 갖춘 워크스테이션에서 실행하며, 클라우드 전용 솔루션보다 뛰어난 성능을 자랑합니다.



### GCI-ViTAL: Gradual Confidence Improvement with Vision Transformers for Active Learning on Label Nois (https://arxiv.org/abs/2411.05939)
Comments:
          under review

- **What's New**: 이번 연구에서는 label noise가 있는 상황에서 이미지 분류를 위한 Active Learning (AL) 방법들을 비교하고 새로운 deep active learning 알고리즘인 GCI-ViTAL을 제안합니다. 이 알고리즘은 label noise에 강인하도록 설계되었습니다.

- **Technical Details**: GCI-ViTAL은 예측 엔트로피(predictive entropy)와 클래스 중심 깨끗한 세트 주의 벡터(class-centric clean set attention vectors)와 비교한 마지막 레이어 주의 벡터의 Frobenius norm을 이용합니다. 이 모델은 불확실성과 의미적으로 전통적인 이미지와 다른 샘플을 식별하는 데 도움을 줍니다. Label smoothing이 적용되어 잠재적으로 노이즈가 포함된 레이블에 대해 지나치게 확신하지 않는 모델 학습을 지원합니다.

- **Performance Highlights**: GCI-ViTAL은 다양한 수준의 대칭 label noise에 대해 평가되었으며, CNN 모델에 비해 모든 AL 전략에서 ViT 모델을 사용하면 성능이 크게 향상됨을 보여주었습니다. 특히, label noise가 있는 경우에서 더 두드러진 성과를 보였습니다.



### Moving Off-the-Grid: Scene-Grounded Video Representations (https://arxiv.org/abs/2411.05927)
Comments:
          Accepted to NeurIPS 2024 (spotlight). Project page: this https URL

- **What's New**: MooG(Moving Off-the-Grid)는 비디오 표현 학습을 위한 새로운 self-supervised 방법론으로, 기존의 grid-based 처리 방식만으로 엮이지 않고, 영상의 구조와 무관하게 장면의 요소를 보다 일관되게 표현할 수 있는 토큰들의 움직임을 가능하게 한다.

- **Technical Details**: MooG는 cross-attention과 positional embeddings의 조합을 활용하여 representation 구조와 이미지 구조를 분리시킨다. 이 모델은 다음 프레임 예측(next frame prediction) 손실을 이용하여 비디오 데이터에서 학습되며, 입력 프레임이 도착할 때마다 cross-attention을 통해 토큰을 업데이트하고, 이미지를 복원하는데도 cross-attention을 사용한다.

- **Performance Highlights**: MooG는 DINO와 같은 기존 self-supervised grid-based 모델들을 초월하는 성능을 보이며, point tracking, monocular depth estimation 및 object tracking 등 다양한 다운스트림 비전 작업에 유용한 특징을 제공함을 정량적으로 및 정성적으로 입증하였다.



### Autoregressive Models in Vision: A Survey (https://arxiv.org/abs/2411.05902)
- **What's New**: 자기회귀 모델(Autoregressive models)은 자연어 처리(NLP) 분야에서 큰 성공을 거두었다. 최근에는 컴퓨터 비전 분야에서도 이러한 모델이 주목받고 있으며, 고급 비주얼 콘텐츠를 생성하는 데 탁월한 성능을 발휘하고 있다. 이 서베이는 비전에 적용된 자기회귀 모델에 대한 포괄적인 문헌 검토를 다룬다. 또한, 제안된 모델은 이미지 생성, 비디오 생성, 3D 생성, 다중 모달 생성 등 다양한 분야에서 응용된다.

- **Technical Details**: 비전에서의 자기회귀 모델은 여러 레벨의 표현 전략을 기반으로 하는데, 픽셀 수준(pixel-level), 토큰 수준(token-level), 스케일 수준(scale-level)으로 나뉜다. 각각의 모델은 독특한 장점과 도전을 가지고 있으며, 비전에서의 자기회귀 모델의 범주는 크게 픽셀 기반(pixel-based), 토큰 기반(token-based), 스케일 기반(scale-based) 모델로 나뉘어 구체적으로 분석된다.

- **Performance Highlights**: 자기회귀 모델들은 이미지 생성, 비디오 생성, 및 의료 AI와 같은 다양한 응용 분야에서 비약적인 발전을 이루었으며, 특히 GAN, Diffusion, MAE 기반 방법과의 성능 비교를 통해 그 우수성을 강조한다. 학계에서는 이러한 모델들 간의 상호 연관성을 탐구하며 향후 연구 방향을 제안하고 있다.



### Enhancing Cardiovascular Disease Prediction through Multi-Modal Self-Supervised Learning (https://arxiv.org/abs/2411.05900)
Comments:
          Accepted to British Machine Vision Conference (BMVC) 2024

- **What's New**: 본 연구는 단일 모달리티(meta-modal) 데이터만으로는 포착할 수 없는 심혈관질환(CVD) 예측의 새로운 통찰력을 다중 모달 학습(multi-modal learning)을 통해 제공하는 것을 목표로 하고 있습니다. 심장 자기 공명 영상(cardiac magnetic resonance images)과 심전도(electrocardiogram) 신호, 의료 정보를 통합하여 개인의 심혈관 건강 상태를 포괄적으로 이해할 수 있도록 하는 모델을 제안합니다.

- **Technical Details**: 본 연구에서는 masked autoencoder를 사용해 ECG 인코더를 전훈련(pre-train)하여 심전도 데이터에서 관련 특징을 추출합니다. 이어서, 이미지 인코더를 통해 심장 자기 공명 영상에서 관련 특징을 추출하고, multimodal contrastive learning을 통해 비싼 CMR 모달리티에서 경제적인 ECG 및 의료 정보 모달리티로 지식을 전이합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 다양한 이용 가능 모달리티의 정보를 활용하여 이미지를 개선하고, 감독 학습(supervised approach)보다 7.6% 향상된 balanced accuracy를 기록하였습니다.



### Integrating Object Detection Modality into Visual Language Model for Enhanced Autonomous Driving Agen (https://arxiv.org/abs/2411.05898)
Comments:
          accepted by SafeGenAI workshop of NeurIPS 2024

- **What's New**: 이번 논문은 자율주행 시스템의 시각적 이해를 향상시키기 위해 시각 언어 모델(Visual Language Models, VLMs)과 객체 탐지에 특화된 추가적인 시각 인식 모듈을 통합한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 Llama-Adapter 아키텍처를 확장하여 YOLOS 기반 탐지 네트워크를 CLIP 인식 네트워크와 통합합니다. 이를 통해 객체 탐지 및 위치 지정의 한계를 해결합니다. 카메라 ID 분리기를 도입하여 다중 시점 처리를 개선하고, 종합적인 환경 인식을 가능하게 합니다.

- **Performance Highlights**: DriveLM 시각 질문 응답 과제에서의 실험 결과는 ChatGPT 점수, BLEU 점수, CIDEr 지표에서 의미 있는 개선을 보여, 모델의 응답이 진실과의 근접성을 나타냅니다. 이번 연구는 자율주행 시스템의 능력과 해석 가능성을 향상시키는 promising한 단계로 평가됩니다.



### Predictive Digital Twin for Condition Monitoring Using Thermal Imaging (https://arxiv.org/abs/2411.05887)
- **What's New**: 이 논문은 상태 모니터링을 위해 특별히 설계된 예측 디지털 트윈의 개발과 실제 적용을 탐구합니다. 엄격한 수학적 모델과 열 이미징 기술을 사용하여, Robust Principal Component Analysis (RPCA) 및 Dynamic Mode Decomposition (DMD)와 Proper Orthogonal Decomposition (POD)의 통합 개발을 제안합니다.

- **Technical Details**: 이 연구에서는 열 이미징을 통해 모니터링된 가열된 판에 대한 실시간 실험을 포함하여 디지털 트윈의 예측 능력과 상태 모니터링, 이상 탐지 기능을 입증합니다. 그리고 가상 현실을 포함한 인간-기계 인터페이스를 도입하여 사용자 상호작용과 시스템 이해를 향상시킵니다.

- **Performance Highlights**: 연구의 주요 기여는 고차원 열 이미징 데이터를 기반으로 한 상태 모니터링을 위한 물리적 프레임워크의 개발과 그에 따른 실시간 진단 및 예측 디지털 트윈의 생성 및 시연입니다. 이러한 기여는 디지털 트윈이 산업 관행을 혁신할 잠재력을 보여줍니다.



### Towards Equitable ASD Diagnostics: A Comparative Study of Machine and Deep Learning Models Using Behavioral and Facial Data (https://arxiv.org/abs/2411.05880)
- **What's New**: 이번 연구에서는 여성의 자폐 스펙트럼 장애(ASD) 진단을 위한 기계 학습 모델, 특히 Random Forest와 convolutional neural networks를 평가했습니다. 이 연구는 ASD 진단을 개선하기 위한 혁신적인 접근 방식을 제안하고 있습니다.

- **Technical Details**: Random Forest 모델은 다수의 데이터셋에서 100% 검증 정확도를 달성했으며, 이 모델의 복잡한 관계 관리 능력과 낮은 오탐지율로 인해 조기 개입에 중요한 역할을 할 수 있습니다. MobileNet은 이미지 기반 분석에서 87%의 정확도로 baseline CNN을 초과했지만, 30%의 검증 손실이 나타나 추가 최적화가 필요합니다.

- **Performance Highlights**: Random Forest의 높은 정확도와 균형 잡힌 정밀도-재현율 지표는 임상 작업 흐름 개선에 기여할 수 있습니다. MobileNet의 경량 구조는 자원이 제한된 환경에서도 접근 가능한 ASD 스크리닝을 가능하게 할 잠재력을 보여줍니다.



### Smile upon the Face but Sadness in the Eyes: Emotion Recognition based on Facial Expressions and Eye Behaviors (https://arxiv.org/abs/2411.05879)
- **What's New**: 본 연구는 Eye-behavior-aided Multimodal Emotion Recognition (EMER) 데이터셋을 소개하며, 표정 인식(FER)와 감정 인식(ER) 간의 간극을 이해하고 이를 좁히는 데 중점을 두고 있습니다. 이 데이터셋은 비침습적인 눈 행동 데이터를 통합하여 자연스럽고 정확한 인간의 감정을 포착하는 것을 목표로 합니다.

- **Technical Details**: EMER 데이터셋은 자극 자료 유도 자발적 감정 생성 방법을 사용하여 눈의 움직임과 고정 맵, 얼굴 비디오를 통합합니다. 이 데이터셋은 1,303 고품질의 멀티모달 데이터 시퀀스를 포함하며, EMERT 아키텍처는 모달리티 적대적 특성 분리와 다중 작업 Transformer를 이용하여 감정 간극을 효율적으로 식별합니다.

- **Performance Highlights**: EMERT는 다른 최첨단 멀티모달 방법들을 크게 능가하는 성능을 보여주었으며, 이는 눈 행동 모델링의 중요성을 강조합니다. 우리는 EMER 데이터셋에 대해 7개의 멀티모달 벤치마크 프로토콜을 도입하고, FER와 ER 간의 간극을 명확히 분석하여 향후 연구 방향을 제시합니다.



### Joint-Optimized Unsupervised Adversarial Domain Adaptation in Remote Sensing Segmentation with Prompted Foundation Mod (https://arxiv.org/abs/2411.05878)
Comments:
          12 pages,6 figures, 6 tables

- **What's New**: 본 논문은 다양한 원거리 감지(Remote Sensing) 장면에서 주석이 달린 데이터 없이 모델을 목표 도메인 샘플에 적응시키는 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 기법을 제안합니다.

- **Technical Details**: 제안된 방법은 'Segment Anything Model (SAM)'을 통합한 조인트 최적화(adversarial optimization) 네트워크인 SAM-JOANet을 사용합니다. 이 네트워크는 SAM의 강력한 일반화 표현 능력을 활용하여 특징 간 불일치를 줄이고, 최적화된 피쳐 레벨 적대적 프롬프트 세그멘터가 클래스 비의존적(class-agnostic) 맵을 생성하여 피쳐 표현을 안내합니다.

- **Performance Highlights**: ISPRS 및 CITY-OSM 데이터셋에서의 광범위한 평가를 통해 제안된 방법의 효과가 입증되었으며, 결과는 시각화 및 분석을 통해 해석 가능성과 견고성(robustness)을 뒷받침합니다.



### Conditional Diffusion Model for Longitudinal Medical Image Generation (https://arxiv.org/abs/2411.05860)
Comments:
          4 pages, 2 figures, conference

- **What's New**: 이 논문은 알츠하이머병(Alzheimer's disease)의 진행 과정을 3D 의료 이미징(medical imaging) 데이터로 모델링하는 새로운 접근 방식을 제안하고 있습니다.

- **Technical Details**: 제안된 방법은 단일 자기 공명 영상(single magnetic resonance imaging, MRI)을 사용하여 확산 기반(diffusion-based) 모델을 구현합니다. 이 모델은 조건부 MRI(conditioning MRI)와 시간 방문 인코딩(time-visit encoding)을 주입하여 원본(source) 이미지와 대상(target) 이미지 간의 변화를 제어합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 경쟁 방법들에 비해 더 높은 품질의 이미지를 생성하는 것으로 나타났습니다.



### Saliency Assisted Quantization for Neural Networks (https://arxiv.org/abs/2411.05858)
- **What's New**: 이번 논문은 딥러닝 모델의 해석 가능성을 개선하고 자원 제약 환경에서의 효율성을 높이기 위해 실시간 설명을 제공하는 새로운 접근 방식을 제안합니다. 이를 통해 모델이 입력의 가장 중요한 특징에 집중하도록 유도하며, 모델의 예측 정확도와 해석 가능성 간의 균형을 조명합니다.

- **Technical Details**: 연구는 Convolutional Neural Networks에서 양자화(Quantization)가 해석 가능성과 정확도에 미치는 영향을 비교 분석합니다. 특히, Parameterized Clipping Activation 방법을 사용하여 양자화를 구현하고, MNIST와 FashionMNIST 데이터셋에서 모델의 성능을 평가했습니다. 세 가지 비트(Bits) 폭 구성(2-bit, 4-bit, mixed 4/2-bit)을 통해 효율성과 해석 가능성 간의 트레이드오프를 탐색합니다.

- **Performance Highlights**: 결과적으로, 양자화는 자원 제한 장치에서 모델을 구현하는 데 필수적이지만, 정확도와 해석 가능성 간의 트레이드오프가 필요하다는 것을 보여줍니다. 낮은 비트 폭에서 두 메트릭의 감소가 보다 두드러지며, 특히 모델 투명성 요구가 있는 응용 프로그램에서 양자화 매개변수 선택의 중요성을 강조합니다.



### StegaVision: Enhancing Steganography with Attention Mechanism (https://arxiv.org/abs/2411.05838)
Comments:
          AAAI-25 Student Abstract

- **What's New**: 본 논문에서는 이미지 스테가노그래피(image steganography) 분야에서 기존 방법의 한계를 극복하기 위해 채널 및 공간 주의 메커니즘(channel and spatial attention mechanisms)을 강화한 인코더-디코더 아키텍처(encoder-decoder architecture)를 제안합니다.

- **Technical Details**: 우리는 다음과 같은 5가지 주의 조합을 실험했습니다: (1) 채널 주의, (2) 공간 주의, (3) 순차적 채널 후 공간 주의, (4) 공간 주의 후 채널 주의 및 (5) 병렬 채널 및 공간 주의. 실험 결과, 병렬 조합이 이미지 품질과 숨기기 능력 간의 균형을 개선했습니다.

- **Performance Highlights**: 병렬 채널 및 공간 주의를 사용한 경우 PSNR(Peak Signal-to-Noise Ratio)과 SSIM(Structural Similarity Index) 점수가 향상되어, 기존 방법들보다 시각적 품질과 숨기기 용량을 동시에 유지하는 데 성공했습니다.



### On the Trade-Off between Stability and Fidelity of Gaussian-Smoothed Saliency Maps (https://arxiv.org/abs/2411.05837)
- **What's New**: 이 연구는 Gradient 기반의 saliency maps에서 Gaussian smoothing을 적용하여 안정성을 높이는 방법을 탐구합니다. Smooth-Grad 알고리즘을 통해 주어진 훈련 데이터의 무작위성에 대한 gradients의 안정성을 증대시키는데 초점을 맞춥니다.

- **Technical Details**: 우리는 알고리즘적 안정성(algorithmic stability) 프레임워크를 활용하여 Simple-Grad, Integrated-Gradients 및 Smooth-Grad의 saliency maps의 안정성 오류를 이론적으로 분석합니다. Gaussian smoothing이 훈련 설정의 무작위성에 대한 saliency maps의 안정성을 증가시키기 위한 기여를 입증했습니다.

- **Performance Highlights**: Numerical 실험 결과는 Gaussian smoothing이 gradient-based interpretation maps의 안정성을 증가시키면서, 원본 Simple-Grad map과의 차이를 더할 수 있음을 보여주었습니다. 이 연구는 saliency maps의 안정성(fidelity)과 충실도(stability) 사이의 trade-off를 조명합니다.



### Prion-ViT: Prions-Inspired Vision Transformers for Temperature prediction with Specklegrams (https://arxiv.org/abs/2411.05836)
- **What's New**: 이 연구는 생물학적 프리온(prion) 메모리 메커니즘에서 영감을 받아 특히 Fiber Specklegram Sensors (FSS) 데이터를 사용하여 온도 예측을 위한 새로운 Prion-Vision Transformer (Prion-ViT) 모델을 제안합니다.

- **Technical Details**: Prion-ViT 모델은 영속적인 메모리 상태(persistent memory state)를 활용하여 여러 레이어에 걸쳐 중요한 특징들을 보존하고 전파함으로써 예측의 정확성을 높입니다. 또한 Temperature prediction에 있어 평균 절대 오차(mean absolute error, MAE)를 "0.52도 Celsius"로 줄였고, 기존의 ResNet, Inception Net V2 및 다른 transformer 기반 아키텍처보다 뛰어난 성능을 보입니다.

- **Performance Highlights**: Prion-ViT는 복잡한 광학 간섭 패턴을 포착하는 데 강력한 솔루션을 제공하여 실시간 산업 온도 모니터링 애플리케이션에 유망한 발전을 나타내며, 다른 광학 센싱(optical sensing) 분야에도 적용 가능성을 제시합니다.



### Diversify, Contextualize, and Adapt: Efficient Entropy Modeling for Neural Image Codec (https://arxiv.org/abs/2411.05832)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 효율적인 엔트로피 모델링을 위한 새로운 프레임워크 DCA(Diversify, Contextualize, Adapt)를 제안하고 있으며, 이는 기존 방법들보다 더 다양한 컨텍스트를 활용하여 성능을 향상시킵니다.

- **Technical Details**: DCA는 세 가지 다른 하이퍼 잠재 표현(local, regional, global)을 추출하여 전방 적응(forward adaptation)을 위한 충분한 컨텍스트를 제공합니다. 이 프레임워크는 효율적인 후방 적응(backward adaptation) 방법과 결합하여 개선된 성능을 보장합니다.

- **Performance Highlights**: Kodak 데이터셋에서 기존 최첨단 방법에 비해 3.73% BD-rate 증가를 달성하여 다양한 비트레이트 영역에서 성능 개선을 보여주었습니다.



### A Theory of Stabilization by Skull Carving (https://arxiv.org/abs/2411.05827)
Comments:
          4 pages, 3 figures

- **What's New**: 이번 연구에서는 3D 게임, 가상 현실 및 영화 제작을 위한 포토리얼 아바타 제작에서 얼굴 움직임의 정확한 안정화를 위한 새로운 접근 방식을 제시합니다. 이 방법은 신경 서명 거리(field) 및 미분 가능한 등면(mesh) 메싱을 활용하여 불규칙한 삼각형 메쉬나 포인트 클라우드에서 두개골 안정화 변환을 직접 계산하여 정확성과 강인성을 현저히 향상시킵니다.

- **Technical Details**: 제안된 방법은 고정된 두개골 모양과 피부 두께에 대한 단순한 가정에 의존하지 않고, 정적 표정 스캔을 위한 안정적인 헐(stable hull) 개념을 도입합니다. 이 헐은 안정화된 스캔의 불리언 교차의 표면으로, 주어진 정적 점 구름이나 삼각형 메쉬에서 각 스캔의 사인 거리장(SDF)으로 변환된 후, 뼈대 좌표계에서 조정이 이루어집니다. 이 과정은 불안정한 표정 데이터의 안정화와 동시에 두개골의 위치를 정확하게 추정합니다.

- **Performance Highlights**: 이번 연구는 기존의 방법들과 비교할 때, 다양한 인구 집단을 대상으로 복잡한 표정에 대한 안정화 성능이 우수함을 보여줍니다. 또한, 각 표현을 조정할 때 두개골과의 정렬이 잘 이루어져 있어 제안된 알고리즘이 신뢰할 수 있는 새롭고 효과적인 접근임을 입증합니다.



### From Pixels to Prose: Advancing Multi-Modal Language Models for Remote Sensing (https://arxiv.org/abs/2411.05826)
Comments:
          10 pages, 1 figure

- **What's New**: 본 논문은 원거리 탐지(remote sensing) 분야에서 다중 모달 언어 모델(MMLMs)의 개발과 응용을 탐구합니다. 위성 이미지를 자연어로 해석하고 설명할 수 있는 모델의 능력에 초점을 맞추고 있습니다.

- **Technical Details**: MMLMs는 일반적으로 이중 인코더 아키텍처(dual-encoder architecture)를 사용하여 시각적 및 텍스트 정보를 통합 처리합니다. Transformer 모델을 활용하여 복잡한 원거리 탐지 데이터를 효과적으로 처리하며, 주목(attention) 메커니즘을 통해 시각 및 텍스트 입력의 중요한 부분에 집중하는 방식으로 작동합니다.

- **Performance Highlights**: 이 모델들은 환경 모니터링, 도시 계획 및 재난 대응과 같은 주요 응용 분야에서 효율적인 정보 추출을 통해 자동화된 지구 관측 분석을 크게 향상시킵니다. 특히, 장면 설명(scene description), 객체 탐지(object detection), 변화 탐지(change detection) 등 다양한 응용 분야에서 그 효과를 입증하고 있습니다.



### FlexCAD: Unified and Versatile Controllable CAD Generation with Fine-tuned Large Language Models (https://arxiv.org/abs/2411.05823)
Comments:
          23 pages

- **What's New**: 최근 사용자 의도에 기반하여 컴퓨터 지원 설계(CAD) 모델을 생성하는 접근 방식인 controllable CAD generation에 대한 관심이 증가하고 있습니다. 이러한 배경에서 저자들은 FlexCAD라는 통합 모델을 제안하며, 이는 정적 지식에서 벗어나 다양한 CAD 구조 계층을 효율적으로 실현할 수 있는 모델입니다.

- **Technical Details**: FlexCAD는 대규모 언어 모델(LLM)을 미세 조정하여 CAD 모델을 구조화된 텍스트로 표현하는 방식입니다. 이 과정에서 CAD 모델의 각 계층을 텍스트 토큰의 시퀀스로 변환하고, 계층 인식을 위한 마스킹 전략을 도입하여 다양한 생성 작업을 통합하고 있습니다.

- **Performance Highlights**: FlexCAD는 공공 데이터셋에 대한 포괄적인 실험을 통해 생성 품질과 제어 가능성을 크게 향상시키는 효과를 보였습니다. 또한, 이 모델은 기존의 CAD 생성 방식과 비교하여 효율적인 성능을 제공하고 있습니다.



### SPACE: SPAtial-aware Consistency rEgularization for anomaly detection in Industrial applications (https://arxiv.org/abs/2411.05822)
Comments:
          Accepted to WACV 2025

- **What's New**: 본 논문에서는 SPACE라는 새로운 이상 감지(Anomaly Detection) 방법론을 제안합니다. 이 방법은 Feature Encoder (FE)와 Student-Teacher (S-T) 접근법의 통합 구조를 기반으로 하며, Spatial Consistency Regularization Loss (SCL)와 Feature Converter Module (FM)이라는 두 가지 핵심 요소를 포함합니다.

- **Technical Details**: SPACE 방법은 SCL을 통해 학생 모델이 교사 모델을 지나치게 모방하지 않도록 하고, 그로 인해 발생하는 과적합(overfitting)을 방지합니다. 또한, FM은 FE로부터의 모호한 정보를 학습하는 것을 방지하여 학습된 특징을 보호하고 구조적 및 논리적(anomaly detection) 이상 감지의 효과성을 높입니다.

- **Performance Highlights**: 실험 결과, SPACE는 MVTec LOCO, MVTec AD, VisA 데이터셋에서 기존 최첨단 방법들 대비 이상 감지에서 우수한 성능을 보였으며, 각 모듈의 효율성 또한 정성 평가를 통해 입증되었습니다.



### Grounding Video Models to Actions through Goal Conditioned Exploration (https://arxiv.org/abs/2411.07223)
Comments:
          Project page at this https URL

- **What's New**: 이번 연구에서는 대규모 비디오 모델을 통해 얻은 물리적 지식으로 Agent의 동작과 목표를 시각적으로 탐색할 수 있는 새로운 방법을 제안합니다. 기존의 방법들이 Agent 특정 데이터에 기반한 별도의 vision-based inverse dynamic model을 사용해야 했던 것에 비해, 우리는 생성된 비디오 상태를 탐색의 시각적 목표로 활용하여 이를 해결하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 trajectory level action generation과 비디오 안내를 결합하여 Agent가 외부의 감독 없이도 복잡한 작업을 해결할 수 있게 합니다. 연구진은 Libero, MetaWorld, Calvin, iThor Visual Navigation의 다양한 환경에서 8개, 6개, 4개, 12개의 작업을 검증하며, 이론적으로는 행동 클로닝(behavior cloning) 기준선과 유사하거나 더 뛰어난 결과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 50개의 시연 데이터를 이용해 훈련된 BC보다도 높은 성공률을 보여, 비디오 모델에 의해 주어진 목표를 환경 속 탐색을 통해 직접 학습함으로써 이뤄낸 성과입니다. 실험 결과는 훈련 데이터의 양과 다양성에 따라 성과가 향상된다고 나타났으며, 특히 비디오 탐색 빈도와 비디오 모델의 수평선(horizon) 변동이 탐색 성능에 미치는 영향을 평가했습니다.



### Acoustic-based 3D Human Pose Estimation Robust to Human Position (https://arxiv.org/abs/2411.07165)
Comments:
          Accepted at BMVC2024

- **What's New**: 이 논문에서는 저수준의 음향 신호만을 이용하여 3D 인간 자세 추정 문제를 탐구합니다. 기존의 방법은 사용자가 스피커와 마이크 사이의 직선 위에 위치해야 한다고 가정하였으나, 이는 실세계에서의 적용에 제한이 있었습니다. 이에 따라 위치 판별기와 잔향 저항 모델을 결합한 새로운 방법을 제안하였습니다.

- **Technical Details**: 제안된 방법은 위치 판별기가 포함되어 있으며, 주체의 위치에 관계없이 인식할 수 있는 특징을 추출합니다. 또한, 추정 대상 시간 이전의 음향 신호를 참조하여 신호의 변동에 대한 강건성을 높이는 방법을 제안합니다. 이 논문에서는 스피커와 마이크 간의 직선에서 멀리 떨어진 여러 위치에서의 데이터를 포함한 새로운 데이터셋을 구축하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보였으며, 다양한 인간 위치를 커버하는 경우에도 안정적인 3D 자세 추정이 가능함을 입증하였습니다.



### Lost in Tracking Translation: A Comprehensive Analysis of Visual SLAM in Human-Centered XR and IoT Ecosystems (https://arxiv.org/abs/2411.07146)
- **What's New**: 이번 논문에서는 다양한 응용 분야에서 최신 추적 알고리즘의 성능을 평가하면서, 이들 알고리즘이 다른 유형의 이동성과의 상이성을 수용하지 못하는 문제를 집중적으로 다루고 있습니다. 기존의 응용 분야에 국한되지 않고, IoT 및 XR 애플리케이션 전반에 걸친 성능 분석을 통해 새로운 통찰을 제공합니다.

- **Technical Details**: 논문은 알고리즘, 환경 및 이동성 관련 도전 과제를 분류하고, 이를 해결하기 위한 여러 접근 방법을 제시합니다. 기존 SLAM (Simultaneous Localization and Mapping) 방법과 최신 데이터 기반 알고리즘(CNN, RNN 등)을 비교하여 각각의 강점과 한계를 분석하였으며, 데이터를 기반으로 한 특징(NLP, RL 등) 분류를 통해 다양한 환경에서의 추적 성능을 quantitatively 평가하였습니다.

- **Performance Highlights**: 최신 추적 알고리즘의 성능은 응용 분야와 환경에 따라 크게 달라지며, XR 애플리케이션에서는 특히 인간의 비예측적 행동에 적응해야 하는 복잡성이 더해집니다. 연구 결과는 추적 알고리즘이 교차 응용 분야에서 일관되게 성능을 발휘하지 못하며, 이를 해결하기 위한 적응형 솔루션과 실험적 데이터 평가의 중요성을 강조하고 있습니다.



### Learning Collective Dynamics of Multi-Agent Systems using Event-based Vision (https://arxiv.org/abs/2411.07039)
- **What's New**: 이 논문은 시각 기반 인식을 통해 다중 에이전트 시스템의 집단 역학을 학습하고 예측하는 새로운 문제를 제안합니다. 특히 상호 작용 강도와 수렴 시간의 예측에 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 시각적으로 캡처된 데이터에서 다중 에이전트의 집단 역학을 직접 예측하는 딥 러닝 모델을 사용합니다. 기존의 수치 모델들이 에이전트의 정확한 위치를 요구하는 반면, 우리는 상태 정보를 필요로 하지 않는 이벤트 기반 비전을 활용합니다. 이를 위해 이벤트 카메라를 사용하여 높은 시간 해상도와 동적 범위를 가진 데이터를 수집하였습니다.

- **Performance Highlights**: 제안된 모델은 전통적인 프레임 기반 방법보다 이벤트 기반 방법이 집단 행동 예측에서 동작 변화를 더 잘 포착함을 보여주며, 다중 에이전트 시스템의 실시간과 정확한 이해를 위한 딥 러닝 아키텍처인 evMAP을 소개합니다.



### Scaling Mesh Generation via Compressive Tokenization (https://arxiv.org/abs/2411.07025)
Comments:
          Homepage: this https URL , Code: this https URL

- **What's New**: 이번 논문에서는 Blocked and Patchified Tokenization (BPT)이라는 새로운 메쉬 표현 방식을 제안하며, 이는 8,000개 이상의 면을 초과하는 메쉬 생성을 가능하게 합니다. BPT는 블록별 인덱싱과 패치 집계를 사용하여 메쉬 시퀸스를 약 75% 압축하며, 이는 기존 시퀸스에 비해 대폭적인 데이터 양을 처리할 수 있게 합니다.

- **Technical Details**: BPT는 카르테시안 좌표를 블록별 인덱스로 변환하여 메쉬 토큰을 정점 수준에서 압축하며, 가장 많은 면과 연결된 정점들을 패치 센터로 선택한 후 해당 정점 주변의 면을 패치로 집계합니다. 이렇게 하여 기존의 메쉬 시퀸스 길이를 약 75% 줄여서 SoTA 압축 비율을 달성합니다. 이 방법은 정교한 세부 사항과 정확한 위상을 가진 메쉬 생성을 가능하게 합니다.

- **Performance Highlights**: BPT를 활용한 메쉬 생성 모델은 복잡한 세부 사항을 포함하며, 3D 콘텐츠 생성에서의 다양한 응용 프로그램에 적용 가능합니다. 이 모델은 조건부 생성을 통해 점 구름(Point Clouds) 및 이미지 기반의 메쉬 생성을 지원하며, 전문적이지 않은 사용자도 제품 수준의 메쉬를 제작할 수 있도록 돕습니다. 따라서, 3D 생성의 새로운 시대가 도래함을 보여줍니다.



### A Hyperspectral Imaging Dataset and Methodology for Intraoperative Pixel-Wise Classification of Metastatic Colon Cancer in the Liver (https://arxiv.org/abs/2411.06969)
Comments:
          12 pages, 5 figures, 5 tables

- **What's New**: 본 연구는 14명의 환자에서 수집한 27개의 HSIs(hyperspectral images)를 포함하는 데이터베이스를 소개하며, 이는 대장선종의 간 전이에서 intraoperative tumor resection을 위한 픽셀 단위 분류를 검증하는 것이 목적입니다.

- **Technical Details**: HSI는 450nm에서 800nm의 스펙트럴 범위에서 1nm 해상도로 획득되었으며, 각 이미지는 1384x1035 픽셀로 구성되어 있습니다. 3명의 병리학자에 의해 픽셀 단위 주석이 수행되었고, label-propagation 기반의 semi-supervised learning (SSL)과 MPRI(multiscale principle of relevant information) 방법 및 tensor singular spectrum analysis 방법을 결합하여 실험 변동성과 주석 데이터 부족 문제를 극복했습니다.

- **Performance Highlights**: SSL-MPRI 방법은 각 클래스당 1%의 주석 픽셀만 사용하여 micro balanced accuracy (BACC) 0.9313 및 micro F1-score 0.9235를 달성했습니다. 반면, 해당 RGB 이미지는 micro BACC 0.8809 및 micro F1-score 0.8688로 낮은 성능을 보였습니다. 이러한 개선은 통계적으로 유의미하며, SSL-MPRI 접근법은 63%의 주석 픽셀로 훈련된 6개의 딥러닝 아키텍처보다 뛰어난 성능을 보였습니다.



### Slowing Down Forgetting in Continual Learning (https://arxiv.org/abs/2411.06916)
- **What's New**: 본 연구는 연속 학습(Continual Learning)에서 재난적인 망각(catastrophic forgetting)을 완화하기 위한 새로운 프레임워크인 ReCL(Reconstruction from Continual Learning)을 제안합니다. 이 프레임워크는 기초적인 뉴럴 네트워크 학습의 암묵적 편향을 이용하여 이전 태스크의 데이터를 재구성하고, 이를 현재의 학습 데이터와 결합하여 망각을 느리게 합니다.

- **Technical Details**: ReCL 프레임워크는 이전 태스크에서 훈련한 데이터를 네트워크 파라미터의 최적화 과정에서 메모리처럼 활용합니다. 이를 통해 개선된 성능을 다양한 CL 시나리오(예: class incremental learning, domain incremental learning, task incremental learning)와 데이터셋(MNIST, CIFAR10)에서 입증하였습니다. 또한, 다층 퍼셉트론(MLP)과 합성곱 신경망(CNN) 구조에서도 성능 향상이 관찰되었습니다.

- **Performance Highlights**: 모든 실험에서 ReCL 프레임워크는 기존의 CL 방법들과 결합할 때에도 일관된 성능 향상을 보여줍니다. 이 연구는 또한 EWC, ER, UPGD와 같은 최신 CL 기법들과 결합했을 때, 특히 망각을 더욱 줄이는 결과를 보였습니다.



### Maximizing domain generalization in fetal brain tissue segmentation: the role of synthetic data generation, intensity clustering and real image fine-tuning (https://arxiv.org/abs/2411.06842)
- **What's New**: 이 논문에서는 태아 뇌 MRI(자기공명영상)에서 데이터 생성 및 도메인 랜덤화(domain randomization) 방법들이 OOD(Out-of-Domain) 일반화 성능을 극대화하는 방안을 탐구합니다. 특히, SynthSeg 기반 방법과 함께 다양한 최신 노하우를 결합하여 실제 데이터에 대한 미세 조정(fine-tuning) 전략의 효과를 보여줍니다.  

- **Technical Details**: SynthSeg는 기본적으로 Gaussian mixture model을 사용하여 시뮬레이션된 이미지를 생성하며, 간단한 강도 클러스터링(intensity clustering)을 통해 신뢰할 수 있는 합성 이미지를 생성하는 데 기여합니다. 이 논문에서는 FetalSynthSeg라는 맞춤형 모델을 기반으로 데이터를 다양화하여 OOD 일반화 성능을 향상시키기 위해 메타 클래스(meta-classes)를 추가하는 방안도 논의합니다. 또한 물리 기반의 강도 생성 방법인 FaBiAN과 결합하여 대조 불변성(contrast invariance)을 달성하는 방법을 탐색합니다.

- **Performance Highlights**: 결과적으로 SynthSeg와 현대적인 가중치 평균(weight averaging) 기반의 미세 조정 접근법을 결합함으로써 신규 도메인에서 적은 양의 실제 이미지로 학습하더라도 인도메인 및 아웃오브도메인에서 성능을 향상시킬 수 있는 유연한 방법을 제안합니다. 이 연구는 SynthSeg 기반의 접근법을 다른 장기나 모달리티로 확장하고자 하는 실무자들을 위한 다섯 가지 주요 권장 사항을 제공합니다.



### JPEG AI Image Compression Visual Artifacts: Detection Methods and Datas (https://arxiv.org/abs/2411.06810)
- **What's New**: 최근 학습 기반 이미지 압축 방법이 전통적인 코덱을 능가하기 시작했습니다. 하지만, 신경망 접근 방식은 예상치 못한 시각적 아티팩트를 도입할 수 있습니다. 이 논문에서는 이러한 아티팩트를 감지하고 지역을 국소화하며 강도를 정량화하는 방법을 제안합니다.

- **Technical Details**: 본 연구에서 제안한 방법은 세 가지 유형의 아티팩트 (텍스처 및 경계 손상, 색상 변화, 텍스트 손상)를 각각 감지합니다. 약 350,000개의 고유 이미지를 다양한 압축 품질 매개변수로 처리한 후 46,440개의 아티팩트를 포함하는 데이터셋이 구축되었으며, 이 데이터셋은 주관적 평가를 통해 검증되었습니다.

- **Performance Highlights**: 우리의 방법은 기존의 이미지 품질 평가 방법들과 비교하여 주관적 평가와의 상관관계가 높음을 입증하였습니다. 이는 신경망 기반 이미지 코덱을 평가하고 성능을 높이기 위해 매우 중요한 결과입니다.



### SynStitch: a Self-Supervised Learning Network for Ultrasound Image Stitching Using Synthetic Training Pairs and Indirect Supervision (https://arxiv.org/abs/2411.06750)
- **What's New**: 이번 연구에서는 2D 초음파(2D US) 이미지 스티칭을 위해 설계된 자가 지도 학습(self-supervised learning) 프레임워크 SynStitch를 소개합니다. SynStitch는 합성 이미지 쌍 생성 모듈(SSPGM)과 이미지 스티칭 모듈(ISM)로 구성되어 있으며, 기존의 조합된 방법보다 더 정교한 방식으로 초음파 영상을 연결하는 데 중점을 두고 있습니다.

- **Technical Details**: SynStitch의 SSPGM은 ControlNet 아키텍처를 기반으로 하여 입력 이미지에서 2DUS 이미지 쌍(I, Is)을 생성합니다. SSPGM은 랜덤한 아핀 변환(affine transformation)을 적용하여 실제적인 스티칭 쌍을 생성하고, ISM은 이 합성 데이터를 이용하여 2DUS 스티칭을 학습합니다. 이 과정에서 ISM은 입력 이미지를 고정 이미지에 맞추기 위해 아핀 변환의 최소화를 수행합니다.

- **Performance Highlights**: SynStitch는 다양한 기존의 스티칭 방법과 비교하여, 2DUS 신장 데이터셋에서 고품질의 스티칭 성능을 입증하였습니다. 정량적 및 정성적 분석을 통해 SynStitch가 통계적으로 유의미한(p<0.05) 성능 향상을 보여주었으며, 코드는 논문 수락 이후 공개될 예정입니다.



### DiffSR: Learning Radar Reflectivity Synthesis via Diffusion Model from Satellite Observations (https://arxiv.org/abs/2411.06714)
- **What's New**: 이 논문에서는 Weather radar 데이터의 합성을 위해 DiffSR이라 불리는 새로운 두 단계의 diffusion-based 방법을 제안합니다. 기존의 MSE (Mean Squared Error) 손실을 사용하는 재구성 방식에서 발생하는 과도한 평활화 문제를 해결하고자 합니다.

- **Technical Details**: DiffSR 방법은 첫 번째 단계에서 글로벌 데이터에 대한 재구성 모델을 사전 훈련(pre-training)하여 레이더 추정을 수행한 후, 두 번째 단계에서 레이더 추정 결과를 위성 데이터(Satellite data)와 결합하여 diffusion 모델의 조건으로 사용합니다.

- **Performance Highlights**: 다양한 실험 결과를 통해, 제안된 방법이 최신 기술(SOTA) 결과를 달성했음을 보여주며, 이는 고주파 세부사항 및 고값 관측 영역을 생성하는 능력을 입증합니다.



### S\'eparation en composantes structures, textures et bruit d'une image, apport de l'utilisation des contourlettes (https://arxiv.org/abs/2411.06696)
Comments:
          in French language, GRETSI Symposium on Signal and Image Processing, Dijon, France, September 2009

- **What's New**: 이 논문에서는 노이즈가 있는 이미지의 분해 알고리즘을 개선하는 방안을 제안합니다. 기존에는 wavelet 변환을 사용하였으나, artefact가 발생하는 단점을 해결하기 위해 contourlet 변환을 도입하고 있습니다.

- **Technical Details**: 논문에서는 이미지의 구조, 텍스처, 노이즈를 분리하는 세 가지 구성 요소 모델을 제안합니다. 이 과정에서 contourlet space와 각 norm을 정의하며, contourlet을 이용한 반복 알고리즘이 개발되어 두 개의 노이즈가 있는 텍스처 이미지에서 테스트되었습니다.

- **Performance Highlights**: 제안된 모델은 기존의 wavelet 기반 방법보다 낫고, 구조의 경계를 더 잘 보존하면서 노이즈 제거 성능을 개선합니다.



### METRIC: a complete methodology for performances evaluation of automatic target Detection, Recognition and Tracking algorithms in infrared imagery (https://arxiv.org/abs/2411.06695)
- **What's New**: 이 논문에서는 자동 목표 탐지(automatic target detection), 인식(recognition), 추적(tracking) 알고리즘의 성능 평가 방법론을 제안합니다.

- **Technical Details**: 제안한 방법론은 객관적인 이미지 데이터셋(objective image datasets) 개발과 다양한 작업(탐지, 인식, 추적)을 위한 적합한 메트릭스(metrics) 정의를 포함합니다.

- **Performance Highlights**: 본 연구에서는 프랑스 MoD 프로그램인 2ACI(``Acquisition Automatique de Cibles par Imagerie``)에서 현재 처리 중인 성능 결과를 제시합니다.



### Machine learning enabled velocity model building with uncertainty quantification (https://arxiv.org/abs/2411.06651)
- **What's New**: 본 논문은 CO2 저장 프로젝트의 모니터링 및 석유 탐사와 같은 여러 지구 물리학적 응용에서 중요한 마이그레이션 속도 모델을 정교하게 특징화하는 방법을 제안합니다. 기존의 Full-Waveform Inversion (FWI) 등의 방법은 복잡한 역문제의 문제를 해결하기 어렵지만, 본 연구에서는 Generative modeling과 physics-informed summary statistics를 통합한 확장 가능한 방법론을 개발하였습니다.

- **Technical Details**: 제안하는 방법은 Diffusion 네트워크를 기반으로 하여, 초기 속도 모델이 불량한 경우 지하 오프셋 이미지 볼륨을 기반으로 요약 통계치를 정의합니다. 이를 통해 마이그레이션 속도 모델의 Bayesian posterior 샘플을 효과적으로 생성하고 불확실성 평가를 가능하게 합니다. 복잡한 속도 모델 구축의 경우, salt flooding과 함께 posterior 근사를 정제하는 새로운 반복 작업흐름, ASPIRE를 제안하였습니다.

- **Performance Highlights**: 현대 합성 데이터셋을 활용하여 Common-Image Gathers (CIGs)를 사용하는 경우의 이점을 재확인 하였으며, 실측 데이터셋을 통한 개념 증명을 통해 본 방법이 산업 규모의 문제에 확장 가능함을 보여주었습니다. 다양한 데이터셋에서 테스트를 통해 계산 효율성을 확보하며 2D 대형 역문제 해결에 효과적임을 입증하였습니다.



### Enhancing frozen histological section images using permanent-section-guided deep learning with nuclei attention (https://arxiv.org/abs/2411.06583)
- **What's New**: 이 연구는 수술 중 신속 진단을 위해 사용되는 동결 절편(frozen section) 이미지의 질을 향상시키기 위한 생성적 딥러닝(generative deep learning) 접근 방식을 제시합니다. 동결 절편 이미지의 핵(nuclei) 영역에 초점을 맞추어 영구 절편(permanent section)에서 정보를 활용합니다.

- **Technical Details**: 제안된 방법은 세분화된 주의 네트워크(segmented attention network)를 통해 핵이 세분화된 이미지를 사용하여 훈련하고, 생성된 영구 이미지를 개선하는 추가 손실 함수(loss function)를 추가합니다. 이는 인위적인 데이터 생성을 방지하고, 블랭크(blank) 지역에서 비신뢰할 수 있는 정보를 도입하지 않도록 합니다.

- **Performance Highlights**: 이 방법은 신장(kidney), 유방(breast), 대장(colon)을 포함한 다양한 조직에서 검증되었으며, 동결 절편 이미지의 효율성을 크게 높이고 진단 정확성을 개선합니다. 이미지를 몇 초 안에 향상시켜 기존 실험실 워크플로우와 원활하게 통합될 수 있습니다.



### PRISM: Privacy-preserving Inter-Site MRI Harmonization via Disentangled Representation Learning (https://arxiv.org/abs/2411.06513)
Comments:
          This work has been submitted to ISBI 2025

- **What's New**: 본 논문에서는 다양한 스캐너 및 프로토콜에 의한 변화를 바탕으로 다중 사이트 MRI 데이터를 조율하는 새로운 딥 러닝 프레임워크 PRISM(Privacy-preserving Inter-Site MRI Harmonization)을 소개합니다. 이 프레임워크는 데이터 프라이버시를 유지하면서도 사이트 간 조화를 가능하게 합니다.

- **Technical Details**: PRISM은 대조 학습(contrastive learning) 및 변분 추론(variational inference)을 활용하여 해부학적 특성을 스타일 및 사이트별 변동으로부터 분리합니다. 이 이중 브랜치 오토인코더 구조는 비매칭 이미지 전환(unpaired image translation)을 지원하며, 새로운 사이트를 비재훈련 및 미세조정 없이 통합할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: PRISM은 뇌 조직 세분화(brain tissue segmentation)와 같은 하위 작업에서의 효과성을 입증하고, 여러 실험을 통해 조화의 성능을 검증했습니다. 실험 결과, PRISM은 고해상도로 MRI 스캔을 재구성하고 해부학적 유사성을 유지하며, 조화 후에는 사이트 특성의 감소를 나타내는 분류 정확성 저하가 나타났습니다.



### Understanding the Role of Equivariance in Self-supervised Learning (https://arxiv.org/abs/2411.06508)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문에서는 전통적인 Self-Supervised Learning(SSL) 방법들이 유용한 특징(예: 색상)의 손실을 초래하고 있다는 점을 지적하며, 이를 보완하기 위한 Equivariant Self-Supervised Learning (E-SSL) 접근법에 대해 심도 있는 논의를 통해 본질적 작동 메커니즘을 이해하려고 합니다.

- **Technical Details**: E-SSL은 입력 데이터의 변형에 민감한 특징을 학습하는 방법으로, 특히 이미지 클래스와 변환 사이의 상호작용을 수학적인 관점에서 탐구합니다. 'Explaining-away' 효과를 활용하여 E-SSL의 일반화 능력을 정보 이론적 관점에서 분석하며, 데이터 변환의 영향을 정량적으로 분석하여 E-SSL 설계를 위한 세 가지 원칙(손실 변환, 클래스 관련성, 단축 가지치기)을 제시합니다.

- **Performance Highlights**: 이론적 발견을 바탕으로 최근 연구된 다양한 E-SSL 방법들을 재검토한 결과, 다수의 사례가 제안된 프레임워크 내에서 잘 설명될 수 있음을 보여줍니다. 이 연구는 E-SSL의 설계 및 이해에 대한 중요한 방향을 제시하며, 미래 연구에서 E-SSL의 가능성을 확장하는 데 기여할 것입니다.



### Diffusion Sampling Correction via Approximately 10 Parameters (https://arxiv.org/abs/2411.06503)
- **What's New**: 이 논문에서는 Diffusion Probabilistic Models (DPMs)의 샘플링 속도를 향상시키기 위해 PCA 기반의 Adaptive Search (PAS)를 제안합니다. 기존의 샘플링 지향 알고리즘과 달리 PAS는 최소한의 학습 가능한 파라미터와 훈련 비용으로 기존 솔버들을 최적화합니다.

- **Technical Details**: PAS는 PCA(주성분 분석)를 사용하여 고차원 샘플링 공간을 구성하는 몇 개의 직교 단위 기저 벡터를 얻고, 이들 벡터를 통해 올바른 샘플링 방향을 결정하기 위한 계수를 학습합니다. 이를 통해 기존의 빠른 솔버의 잘려진(truncation) 오류를 수정하여 샘플링 효율성을 높입니다.

- **Performance Highlights**: CIFAR10 데이터셋에서 PAS는 12개의 파라미터와 단일 NVIDIA A100 GPU에서 훈련 시간 1분 미만으로 DDIM의 FID 점수를 15.69에서 4.37로 향상시켰습니다. 이는 기존의 고비용 훈련 기반 알고리즘에 비해 매우 적은 비용으로 성능을 개선하는 효과를 보여줍니다.



### Mitigating covariate shift in non-colocated data with learned parameter priors (https://arxiv.org/abs/2411.06499)
- **What's New**: 본 논문은 데이터가 시간이나 공간에 따라 분산될 때 발생하는 covariate shift 문제를 해결하기 위해 Fragmentation-Induced covariate-shift Remediation (FIcsR) 방법을 제안합니다.

- **Technical Details**: FIcsR는 fragment의 covariate 분포와 표준 cross-validation 기준의 covariate 분포 간의 $f$-divergence를 최소화하여 데이터의 편향을 줄입니다. 이 방법은 기존의 중요도 가중치 방법과 동등함을 보여주며, 신경망의 과대파라미터화(overparametrized) 문제 때문에 수치적 해결이 어려워 Fisher Information 근사를 도출했습니다. 이를 통해 shift remediation의 전역 추정치를 계산하고 이를 최소화 목표에 priors로 통합했습니다.

- **Performance Highlights**: 여러 데이터 클래스에 대해 40개 이상의 데이터셋에서 광범위한 분류 실험을 통해 FIcsR를 도입했을 때 batch와 fold의 최첨단 성능보다 각각 5% 및 10% 이상 향상된 정확도를 보였습니다. 이 방법은 shift가 있는 여러 조건 하에서도 성능 저하가 느리게 발생하는 것을 보여주었습니다.



### DDIM-Driven Coverless Steganography Scheme with Real Key (https://arxiv.org/abs/2411.06486)
- **What's New**: 본 논문은 Denoising Diffusion Implicit Model (DDIM)을 활용하여 고품질의 stego-images를 생성하는 새로운 coverless steganography 방법을 제안합니다. 이 방법은 기존의 pseudo-key 대신 실제 키를 사용하여 보안성을 강화합니다.

- **Technical Details**: 제안된 방법은 하나의 키 협상을 통해 여러 통신이 가능하여 효율성을 개선하며, 혼돈 암호화(chaotic encryption)를 통해 실제 키를 보호하여 보안성과 기밀성을 높입니다. DDIM은 고품질 이미지를 안정적이고 점진적인 방식으로 생성할 수 있습니다.

- **Performance Highlights**: 본 방법은 기존의 GAN 기반 방식보다 유사하거나 개선된 시각적 충실도를 제공합니다. 또한, 기존 diffusion 기반 시스템의 pseudo-key 의존성을 제거하여 보안성을 크게 높이는 성과를 보여줍니다.



### A Hybrid Approach for COVID-19 Detection: Combining Wasserstein GAN with Transfer Learning (https://arxiv.org/abs/2411.06397)
- **What's New**: COVID-19 진단을 위한 GAN (Generative Adversarial Network) 기반 이미지 합성 방법을 제안하여, 한정된 데이터세트를 극복하고 분류 정확도를 향상시켰습니다.

- **Technical Details**: 커스터마이징된 Wasserstein GAN을 사용하여 실제 이미지보다 19% 더 많은 Chest X-ray 이미지를 생성하였습니다. 이 확장된 데이터셋은 VGG-16, ResNet-50, GoogLeNet 및 MNAST의 네 가지 딥러닝 모델을 훈련하는 데 사용되었습니다.

- **Performance Highlights**: VGG-16은 99.17%의 정확도로 가장 높은 성과를 기록하였고, ResNet-50, GoogLeNet, MNAST는 각각 93.9%, 94.49%, 97.75%의 테스트 정확도를 보였습니다.



### Deep Active Learning in the Open World (https://arxiv.org/abs/2411.06353)
- **What's New**: 이번 논문에서는 open-world 환경에서 새로운 OOD(Out-Of-Distribution) 클래스의 통합을 위한 ALOE라는 새로운 active learning 알고리즘을 소개합니다. ALOE는 두 가지 단계로 구성되어 있으며, 첫 번째 단계에서 다양성 샘플링(diversity sampling)을 통해 대표적인 예제를 선택하고, 두 번째 단계에서는 에너지 기반 OOD 탐지를 통해 알려지지 않은 클래스의 우선 순위를 정합니다.

- **Technical Details**: ALOE(Active Learning in Open-world Environments) 알고리즘은 두 단계의 접근 방식을 사용하여 open-world 환경에서 발생할 수 있는 문제를 해결합니다. 첫 번째 단계에서는 다양성 샘플링을 통해 다양한 데이터 분포를 포괄하는 대표 예제를 선택합니다. 두 번째 단계에서는 에너지 점수 기능을 사용하여 클러스터 내의 예제를 순위 매겨 주목해야 할 OOD 클래스를 우선적으로 파악합니다.

- **Performance Highlights**: ALOE는 ImageNet-LT와 같은 장기적 불균형 이미지 분류 데이터세트에서 실험을 수행했으며, 무작위 샘플링에 비해 동일한 정확도를 달성하는 데 70%의 주석 비용을 절감하였습니다. 모든 실험 설정에서 ALOE가 가장 우수한 성능을 보이며, 알려진 클래스의 성능 향상과 새로운 클래스 발견 사이의 중요한 트레이드오프를 발견하였습니다.



### Activation Map Compression through Tensor Decomposition for Deep Learning (https://arxiv.org/abs/2411.06346)
- **What's New**: 이번 연구는 Edge AI에서의 백프로퍼게이션(backpropagation)의 메모리 부족 문제를 해결하기 위해 텐서 분해(tensor decomposition) 기법을 활용하여 활성화 맵(activation map)의 압축(compression)을 제안합니다.

- **Technical Details**: 이 연구는 Singular Value Decomposition(SVD)와 High-Order Singular Value Decomposition(HOSVD)을 사용하여 백프로퍼게이션의 메모리 발자국(memory footprint)을 줄이는 방법에 중점을 둡니다. 저차원 분해(low-order decomposition)는 많은 메모리 절약 효과를 가지며, 학습에 필수적인 특성(features)을 보존합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 최신 기술들과 비교하여 일반화(generalization)와 메모리 발자국 사이의 균형에서 Pareto-superiority를 달성하였습니다.



### A novel algorithm for optimizing bundle adjustment in image sequence alignmen (https://arxiv.org/abs/2411.06343)
- **What's New**: 본 논문은 cryo-electron tomography의 이미지 시퀀스 정렬에서 Bundle Adjustment (BA) 모델을 최적화하기 위해 최적 제어 이론(optimal control theory)을 활용한 새로운 알고리즘인 Optimal Control Algorithm (OCA)을 소개합니다.

- **Technical Details**: OCA는 비선형 함수의 직접 최적화를 통해 기존의 Levenberg-Marquardt (L-M) 알고리즘의 진동 문제를 효과적으로 완화하고 우수한 수렴 속도를 보여줍니다. 특히, OCA는 초기값 선택에 유연성을 제공하며, bisection 기반의 업데이트 절차를 통해 성능이 크게 개선됩니다.

- **Performance Highlights**: 실험 결과, OCA는 synthetic 및 real-world 데이터셋에서 L-M 알고리즘보다 더 빠른 수렴 속도를 달성하며, 초기 조건이 좋지 않은 데이터셋에서도 효과적으로 성능을 향상시킬 수 있음을 보여줍니다.



### Exploring Out-of-distribution Detection for Sparse-view Computed Tomography with Diffusion Models (https://arxiv.org/abs/2411.06308)
- **What's New**: 최근 연구에서는 확산 모델(diffusion models)이 역 이미지 문제의 비지도 솔버로서의 효용성을 입증했습니다. 본 논문에서는 스파스 뷰 컴퓨터 단층 촬영(sparse-view computed tomography, CT)에서 이러한 모델의 잠재적인 활용과 함께 이상치 탐지(out-of-distribution detection)에 대한 연구를 진행합니다.

- **Technical Details**: 우리는 CT 재구성을 위한 목표 분포를 캡처하도록 훈련된 확산 모델을 활용하여 스파스 뷰 CT의 입력 및 재구성 오류 정의를 재정의합니다. 필터링 역투영(filtered backprojection, FBP) 재구성을 입력으로 사용하고, 다양한 재구성 오류 정의를 조사합니다. 이를 통해 해당 모델이 OOD 탐지기로서의 기능을 발휘할 수 있도록 합니다.

- **Performance Highlights**: MNIST 데이터셋을 통한 실험 결과는 OOD 탐지의 가능성과 한계를 보여주며, 컨디셔닝을 통해 잡음 있는 FBP 입력에서 측정값을 조건화하는 것이 OOD 탐지의 정확성을 높일 수 있음을 시사합니다. 그러나 이 과정에서 OOD 이미지를 잘 재구성하는 문제가 발생하기도 하였습니다. 또한, 가중치 접근법을 도입하여 OOD 측정에 대한 강건성을 향상시키지만, 성능의 일부 손실이 발생할 수 있음을 알게 되었습니다.



### Zero-Shot NAS via the Suppression of Local Entropy Decreas (https://arxiv.org/abs/2411.06236)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 연구는 Zero-Shot NAS의 성능 평가를 가속화하기 위해 네트워크 아키텍처 토폴로지를 활용하여 더 빠르고 효율적인 zero-cost 프록시(Proxy)를 제안합니다. 이는 기존 프록시들이 요구하는 backpropagation이나 입력 데이터 의존성을 제거하여 계산 속도를 한층 높였습니다.

- **Technical Details**: 연구에서는 로컬 엔트로피(local entropy) 감소에 대한 이론 분석을 통해 네트워크 아키텍처가 특성 맵(feature maps)의 로컬 엔트로피를 낮출 수 있음을 증명합니다. 이를 바탕으로 Suppression of Local Entropy Decrease (SED)라는 새로운 프록시를 제안하였고, 이 프록시는 벤치마크에서 기존의 SOTA 프록시들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: SED 프록시는 NATS-Bench-TSS에서 아키텍처 평가를 3.4e-5 초 내에 수행하면서 기존 SOTA 방식들보다 세 배 빠른 결과를 기록하였습니다. SED 기반 NAS는 한 초 만에 더 높은 정확도와 적은 파라미터 수를 가진 아키텍처를 선정할 수 있습니다.



### Alleviating Hyperparameter-Tuning Burden in SVM Classifiers for Pulmonary Nodules Diagnosis with Multi-Task Bayesian Optimization (https://arxiv.org/abs/2411.06184)
Comments:
          12 pages, 4 figures, 37 references

- **What's New**: 이번 연구는 다중 의료 분류 작업을 동시에 해결하기 위해 다중 작업 Bayesian 최적화(multi-task Bayesian optimization, MTBO)를 사용하는 가능성을 조사하였습니다. MTBO를 통해 기존의 단일 작업 접근법(single-task approach)보다 하이퍼파라미터 탐색을 빠르게 수행할 수 있음을 발견하였습니다.

- **Technical Details**: 의료 이미지 분석에서 이미지 이산화(image discretization)는 관심 영역(region of interest, ROI) 내의 강도(intensity) 값을 작게 나누는 과정입니다. 본 연구에서는 9가지 이미지 이산화 전략을 제안하고, RBF SVM을 사용하여 여러 SVM 분류기의 하이퍼파라미터를 동시에 조정하는 알고리즘을 설계하였습니다. 이는 각 작업 간의 관계를 공유하여 최적의 모델을 생성하기 위한 것입니다.

- **Performance Highlights**: MTBO 알고리즘은 단일 작업 Bayesian 최적화(single-task Bayesian optimization)보다 더 높은 성능을 보였으며, 동시에 여러 SVM 분류기의 하이퍼파라미터를 효율적으로 찾아내는 데 기여하였습니다. 본 연구는 의료 분야에서 MTBO 기술을 적용한 최초의 연구입니다.



### Epi-NAF: Enhancing Neural Attenuation Fields for Limited-Angle CT with Epipolar Consistency Conditions (https://arxiv.org/abs/2411.06181)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문에서는 Epi-NAF라는 새로운 접근 방식을 제안하여, 제한 각도를 가진 X-ray CT 이미지에서의 재구성을 개선하고자 합니다. 이는 기존의 신경 필드 방법들이 한계각 제한에서 어려움을 겪는 문제를 해결하기 위함입니다.

- **Technical Details**: Epi-NAF는 X-ray 투영 이미지의 대응하는 에피폴라 선들 간의 일관성 조건에 기반한 새로운 손실 항을 도입하여 신경 감쇠 필드 최적화를 정규화합니다. 이 접근 방법은 180° 범위 내에서 입력 뷰에서 예측된 투영에 대해 감독을 효과적으로 전파하여 재구성 정밀도를 높입니다.

- **Performance Highlights**: Epi-NAF는 기존의 전통적 방법과 신경 필드 기반 방법들과 비교하여 정량적 및 정성적으로 더 뛰어난 성능을 보임을 실험을 통해 확인했습니다. 특히 제한 각도 구성에서 더 나은 CT 스캔 결과를 도출할 수 있었습니다.



### Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework (https://arxiv.org/abs/2411.06160)
- **What's New**: 이 논문은 텍스트 감정 탐지(Text Emotion Detection)의 발전을 위해 Emotion Quantization Network (EQN) 프레임워크를 제안합니다. EQN은 감정 강도를 에너지 수준으로 매핑하여 미세 감정(micro-emotion)의 자동 탐지 및 주석(annotation)을 가능하게 합니다.

- **Technical Details**: EQN 프레임워크는 모든 라벨(all-labels) 및 훈련 세트 라벨 회귀(training-set label regression) 방법을 활용합니다. 이를 통해 기계 모델의 학습능력과 라벨 간의 상호 의존성을 최대한 활용하여 샘플 내 여러 감정을 발견합니다. 이론적으로 미세 감정의 측정과 주석 작성에 기여하는 새로운 접근법이 설명됩니다.

- **Performance Highlights**: EQN 프레임워크는 GoEmotions 데이터셋에서 감정 탐지 및 주석을 수행하며, Google 연구 결과와의 포괄적 비교를 통해 높은 자동 탐지 능력을 입증합니다. EQN은 에너지 레벨 점수로 미세 감정 자동 주석을 최초로 달성하였으며, 감정 탐지 분석 및 감정 컴퓨팅의 정량적 연구에 강한 지원을 제공합니다.



### Emotional Images: Assessing Emotions in Images and Potential Biases in Generative Models (https://arxiv.org/abs/2411.05985)
- **What's New**: 본 연구는 생성 인공지능(Generative AI) 모델이 생성한 이미지에서 불러오는 감정의 일관성 및 편향을 조사합니다. 특히, AI로 생성된 이미지의 감정과 이미지를 생성하는 데 사용된 프롬프트의 감정을 비교함으로써 이러한 편향을 평가합니다.

- **Technical Details**: 세 가지 접근 방식을 사용하여 이미지 내 감정을 식별합니다: 전통적인 감독 학습(Supervised Learning), 비변화 학습(Zero-shot Learning)과 비전-언어 모델(Vision-Language Models), 그리고 교차 모달 자동 캡셔닝(Cross-modal Auto-Captioning)입니다. EmoSet 데이터셋을 통해 감정 분류를 수행하며, 주로 Google의 Vision Transformer(ViT)를 사용합니다.

- **Performance Highlights**: Fine-tuning된 ViT 모델이 이미지에서 감정을 인식하는 데 있어 기존의 zero-shot 및 캡셔닝 기반 방식보다 유의미하게 성능이 뛰어남을 보여주었습니다. AI가 생성한 이미지는 일반적으로 부정적인 감정 콘텐츠에 치우치는 경향이 있으며, 이는 디지털 공간에서 부정적 정서의 영향을 증가시킬 수 있는 가능성을 내포합니다.



### Assessing Foundational Medical 'Segment Anything' (Med-SAM1, Med-SAM2) Deep Learning Models for Left Atrial Segmentation in 3D LGE MRI (https://arxiv.org/abs/2411.05963)
- **What's New**: 본 연구는 의료 분야에 적합하게 조정된 MedSAM 모델을 사용하여 3D LGE MRI에서 좌심방(LA)의 자동 세분화(segmentation)를 수행하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구는 MedSAM 모델의 성능을 평가하고, 자동 추적(automated tracking)을 사용하는 MedSAM2 모델과 각 슬라이스에 대해 별도의 프롬프트(prompt)가 필요한 MedSAM1 모델의 성능을 비교하는데 초점을 맞추고 있습니다. 세분화 정확도를 평가하기 위해 Dice score를 분석합니다.

- **Performance Highlights**: MedSAM 모델은 기존 수동 세분화 방법에 비해 LA 세분화 작업에서 효율적인 비율로 성능 향상을 보이며, 다양한 박스 프롬프트(prompt)의 크기와 위치에 따른 MedSAM1 모델의 세분화 정확도(Dice score)를 평가합니다.



### Efficient Self-Supervised Barlow Twins from Limited Tissue Slide Cohorts for Colonic Pathology Diagnostics (https://arxiv.org/abs/2411.05959)
Comments:
          Submission Under Review

- **What's New**: 이 논문은 대장암 폴립 스크리닝을 위한 최적화된 Barlow Twins 프레임워크를 제안하고 있으며, 병리 데이터에 적용하기 위한 하이퍼파라미터, 증강 전략 및 인코더를 조정하여 성능을 향상시키는 방법을 분석합니다. 특히, 자기 지도 학습(SSL)을 통해 데이터 주석의 부담을 줄이고 병리 이미지를 분석하는 새로운 접근법을 제공합니다.

- **Technical Details**: 이 연구는 Barlow Twins 연구를 병리 데이터의 특성에 맞게 최적화하는 방법을 다룹니다. 이를 통해 폴립의 잉여성을 최소화하고, 필드 오브 뷰(Field of View, FoV)에 따라 최적의 스크리닝 조건을 찾습니다. 제안된 방법의 평가를 위한 새로운 벤치마크 데이터셋을 제작하고, MHIST 및 NCT-CRC-7K 데이터셋을 활용하여 데이터를 전이 학습합니다.

- **Performance Highlights**: 제안된 SSL 표현은 지도 학습(Supervised Learning) 방법보다 더욱 의미 있고 질적으로 우수하다는 것을 보여줍니다. 또한, Barlow Twins는 병리 데이터에 적용될 때 Swin Transformer와 결합하여 성능을 더욱 향상시킵니다. 이 연구는 병리학적 스크리닝에서 심각한 암을 신속하게 진단하고 치료하는 데 중요한 기여를 할 것으로 기대됩니다.



### Querying Perception Streams with Spatial Regular Expressions (https://arxiv.org/abs/2411.05946)
Comments:
          This work has been submitted to the International Journal on Software Tools for Technology Transfer

- **What's New**: 이 논문에서는 multi-modal dynamic environments에서 파생된 spatial과 temporal 데이터를 포함한 지각(Perception) 스트림에 대한 패턴 매칭을 위한 새로운 쿼리 언어인 SpREs를 소개합니다.

- **Technical Details**: SpREs는 다양한 상황에 맞는 쿼리를 제공하여 지각 데이터에 대한 정밀한 패턴 매칭을 가능하게 합니다. STREM 도구는 오프라인 및 온라인 패턴 매칭 프레임워크로, Woven Planet Perception의 공개 AV 데이터셋을 사용해 오프라인 기능을 시연하고, CARLA 시뮬레이터와 결합하여 온라인 기능을 입증하였습니다.

- **Performance Highlights**: STREM을 통해 296 밀리초(ms) 이내에 20,000개 이상의 매치를 찾을 수 있는 성능을 보여주며, 이는 런타임 모니터링 응용 프로그램에 적용 가능성을 시사합니다.



### Towards Multi-Modal Mastery: A 4.5B Parameter Truly Multi-Modal Small Language Mod (https://arxiv.org/abs/2411.05903)
- **What's New**: 새로운 4.5B 파라미터의 소형 언어 모델이 소개되었습니다. 이 모델은 텍스트, 이미지, 비디오, 오디오 등 다양한 입력 및 출력 모달리티(modality)를 처리할 수 있습니다.

- **Technical Details**: 이 모델은 언어 모델링(language modeling)과 다중 작업 학습(multi-task learning)의 최근 발전을 활용하여 만들어졌으며, 간편하게 엣지 추론(edge inference)에 배포될 수 있습니다.

- **Performance Highlights**: 모델은 여러 벤치마크에서 뛰어난 성능을 보여주며, 복잡한 현실 세계 문제를 해결할 수 있는 다중 모달(multi-modal) 인공지능의 가능성을 시사합니다.



### ViT Enhanced Privacy-Preserving Secure Medical Data Sharing and Classification (https://arxiv.org/abs/2411.05901)
Comments:
          2 pages, 2 figures

- **What's New**: 본 연구에서는 의료 이미지 분석을 위한 안전하고 개인 정보를 보호하는 데이터 공유 프레임워크를 소개합니다. 이 프레임워크는 learnable encryption 방식을 통해 데이터를 암호화하고 Vision Transformer (ViT)와 통합하여 정확성을 유지하면서도 보안성을 강화합니다.

- **Technical Details**: 제안된 접근 방식은 block-pixel operation을 기반으로 하는 learnable encryption 기법을 사용하여 이미지의 주요 특징을 보존하면서도 이러한 이미지를 암호화합니다. 다수의 고객이 각기 다른 키를 사용해 로컬 이미지를 암호화하고, 변환된 이미지는 중앙 서버를 통해 전송되어 Transformer Encoder로 복잡한 의존성을 캡처합니다.

- **Performance Highlights**: 연구에서 제안한 ViT 모델은 5,712개의 MRI 이미지를 사용한 brain tumor 분류 및 2,980개의 histopathological 이미지를 사용한 폐 및 대장암 분류에서 높은 정확도를 기록했습니다. ViT 모델은 암호화된 MRI 뇌종양 데이터셋에서 95%의 훈련 정확도와 94%의 검증 정확도를 달성하였으며, 기존 DNN 모델은 38%에서 51%로 낮은 정확도를 보였습니다.



### UnDIVE: Generalized Underwater Video Enhancement Using Generative Priors (https://arxiv.org/abs/2411.05886)
Comments:
          Accepted to IEEE/CVF WACV 2025

- **What's New**: 이 논문에서는 해상도와 시간이 중요한 해양 탐사에서 수중 비디오의 질을 향상시키기 위한 새로운 두 단계 프레임워크를 제안합니다. 첫 번째 단계에서는 비지도 학습 방식으로 생성 사전(generative prior)을 학습하고, 두 번째 단계에서는 물리 기반 이미지 형식을 활용하여 시공간적 향상을 시행합니다.

- **Technical Details**: 제안된 방법은 Denoising Diffusion Probabilistic Model (DDPM)을 통해 수중 이미지의 생성을 위한 강력한 표현을 학습합니다. 이 후 학습된 인코더는 UnDIVE 네트워크에 통합되어 시공간적으로 향상된 비디오를 생성하며, 비디오 프레임 간의 일관성을 유지하기 위해 비지도(optical flow loss)를 활용합니다.

- **Performance Highlights**: 네 가지 다양한 수중 비디오 데이터셋에서 실험한 결과, 제안된 UnDIVE 프레임워크는 기존의 방법들보다 더 나은 성능을 보였으며, 특히 다양한 수중 타입에서 잘 일반화되는 것으로 나타났습니다.



### Alternative Learning Paradigms for Image Quality Transfer (https://arxiv.org/abs/2411.05885)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 새로운 연구에서는 낮은 품질의 의료 이미지를 높은 품질 이미지에서 학습한 정보를 활용하여 향상시키는 'Image Quality Transfer (IQT)' 접근 방식을 제안합니다. 이 연구는 기존의 감독 학습 프레임워크에 따른 IQT 방법과는 달리 비감독 학습 접근법과 감독-비감독 혼합 접근법을 포함하는 두 가지 새로운 IQT 문제 공식화를 제안합니다.

- **Technical Details**: 첫 번째 접근법인 IQT-SRep는 희소 표현(Sparse Representation, SRep) 및 사전 학습 모델을 사용하는 비감독 학습 기반입니다. 두 번째 접근법인 IQT-DDL(Deep Dictionary Learning)은 감독 및 비감독 학습의 조합을 사용하며, 입력 볼륨을 업스케일하기 위한 고해상도 사전을 명시적으로 학습합니다. 이러한 두 모델은 낮은 전장 자기공명영상(MRI) 애플리케이션에서 평범한 MRI 스캐너로부터 얻은 높은 품질 이미지를 복구하기 위해 평가되었습니다.

- **Performance Highlights**: 제안된 방법은 최신 감독 딥러닝 IQT 방법(IQT-DL)과 비교되었으며, 새로운 두 공식화는 훈련 데이터의 분포와 다른 분포의 데이터를 사용하여 테스트할 때 감독 방법에 발생하는 편향을 피할 수 있음을 보여주었습니다. 이는 IQT의 잠재적인 이점을 강조합니다.



### Untrained Perceptual Loss for image denoising of line-like structures in MR images (https://arxiv.org/abs/2411.05884)
- **What's New**: 본 논문에서는 Magnetic Resonance (MR) 이미지의 3D 영상 잡음을 제거하기 위해 untrained Perceptual Loss (uPL)를 도입하였습니다. 이 방법은 뿌리나 혈관과 같은 선형 구조를 포함하는 MR 이미지에 더 적합합니다.

- **Technical Details**: uPL은 이미지에 포함된 선형 구조의 특징을 고려하며, 기존의 L1 손실 또는 SSIM 기반 손실 함수를 초월하는 성능을 보여줍니다. 다양한 uPL 특성(가중치 초기화, 네트워크 깊이, 커널 크기, 풀링 연산 등)이 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: uPL을 사용한 결과, MRA 이미지에서 SSIM 값이 0.93으로, 이전의 L1과 SSIM 손실이 각각 0.81과 0.88에 불과한 것에 비해 우수한 성능을 보였습니다. 작은 uPL 네트워크가 VGG와 같은 대형 네트워크보다 더 우수한 성능을 보이며, 계산 비용에서도 효율적입니다.



### Benchmarking 3D multi-coil NC-PDNet MRI reconstruction (https://arxiv.org/abs/2411.05883)
- **What's New**: 이 논문에서는 비 카르테시안 (non-Cartesian) 하의 3D MRI 재구성 성능 검증이 부족하다는 문제를 해결하기 위해 NC-PDNet (Non-Cartesian Primal-Dual Network)을 3D 다채널 설정으로 확장했습니다.

- **Technical Details**: 본 연구는 채널 특정 (channel-specific) 훈련 구성과 채널 비특정 (channel-agnostic) 훈련 구성이 가지는 영향을 평가하고, 코일 압축 (coil compression)의 효과를 분석했습니다. 또한, Calgary-Campinas 데이터셋을 사용하여 가속화 계수 6의 네 가지 비 카르테시안 하의 언더샘플링 (undersampling) 패턴을 벤치마킹했습니다.

- **Performance Highlights**: 압축된 데이터에 대해 다양한 입력 채널 수로 훈련된 NC-PDNet은 1mm 이소트로픽 (isotropic) 32채널 전체 뇌 3D 재구성에서 평균 PSNR 42.98 dB를 기록했습니다. 추론 시간은 4.95초, GPU 메모리 사용량은 5.49GB로 임상 연구에 대한 상당한 잠재력을 보여줍니다.



### Trends, Challenges, and Future Directions in Deep Learning for Glaucoma: A Systematic Review (https://arxiv.org/abs/2411.05876)
- **What's New**: 본 연구는 심층 학습(Deep Learning, DL) 알고리즘을 활용한 녹내장(glaucoma) 탐지의 최신 발전을 분석합니다. 특히 시스템적 리뷰 및 메타 분석(Preferred Reporting Items for Systematic Reviews and Meta-Analyses, PRISMA) 기법을 통해 수행된 연구입니다.

- **Technical Details**: 연구는 DL 기반의 녹내장 탐지 프레임워크의 세 가지 측면을 다룹니다: 입력 데이터 모달리티(input data modalities), 처리 전략(processing strategies), 그리고 모델 아키텍처 및 응용(model architectures and applications). 각 측면의 사용 추세를 DL이 발전하기 시작한 시점 이후로 분석합니다.

- **Performance Highlights**: 현재 직면한 과제를 다루고, 향후 연구 방향에 대한 제안을 포함하여 DL을 통해 녹내장 탐지 분야에서의 발전을 기대할 수 있습니다.



### Poor Man's Training on MCUs: A Memory-Efficient Quantized Back-Propagation-Free Approach (https://arxiv.org/abs/2411.05873)
- **What's New**: 이 논문은 마이크로컨트롤러(MCU)에서 Back Propagation(BP) 없이 훈련할 수 있는 간단한 방법을 제시하며, 엣지 훈련 하드웨어 설계를 추론 하드웨어 설계처럼 쉽게 만듭니다.

- **Technical Details**: 양자화된 제로차 방법(quantized zeroth-order method)을 활용하여 양자화된 모델 파라미터의 기울기를 추정하며, BP 기반 훈련의 오류를 극복합니다. 차원 축소(dimension reduction) 방법(노드 퍼터베이션(node perturbation), 희소 훈련(sparse training))을 활용하여 제로차 훈련의 수렴성을 개선합니다.

- **Performance Highlights**: 우리의 BP-free 훈련 방법은 자원 제약이 있는 엣지 장치에서 미리 훈련된 이미지 분류기를 다양한 손상된 데이터에 적응시킬 때 BP 기반 훈련과 유사한 성능을 달성하며, 평균 6.4%의 테스트 정확도 향상을 보여줍니다.



### Exploring the Feasibility of Affordable Sonar Technology: Object Detection in Underwater Environments Using the Ping 360 (https://arxiv.org/abs/2411.05863)
Comments:
          This work is currently under review. This is a pre-print

- **What's New**: 본 연구는 주로 내비게이션(navigation) 용도로 사용되는 Ping 360 소나 장치의 복잡한 수중 장애물 탐지 가능성을 탐구합니다. 이 장치는 저렴한 가격과 오픈 소스 특성으로 고가의 이미징 소나 시스템에 대한 비용 효율적인 대안을 제공합니다.

- **Technical Details**: Ping 360 소나 장치는 수중 환경에서의 대상 탐지를 위해 U-Net 세그멘테이션 모델을 훈련시키기 위한 수동 주석이 달린 소나 이미지 데이터셋을 개발하였습니다. 이 연구에서는 수면 반사, 물체 그림자와 같은 요인이 얕은 수중 환경에서 소나의 성능에 미치는 영향을 조사합니다.

- **Performance Highlights**: Ping 360 소나는 간단한 환경에서는 잠재력을 보였으나, 복잡하거나 반사성이 강한 환경에서는 광범위한 데이터 전처리 및 주석 없이 성능이 제한된다는 결과를 도출했습니다. 저렴한 가격의 소나 장치가 복잡한 대상 탐지에 효과적으로 재사용될 수 있는 가능성을 탐구하였습니다.



### Learning Morphisms with Gauss-Newton Approximation for Growing Networks (https://arxiv.org/abs/2411.05855)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 Neural Architecture Search (NAS) 방법에서 network morphism을 활용하여 네트워크를 효과적으로 확장하는 새로운 접근 방식을 제안합니다. 기존 NAS 방법들이 다양한 네트워크를 훈련시켜야 하는 것과 달리, 이 방법은 작은 seed 네트워크에서 시작하여 능동적으로 새로운 뉴런을 추가합니다.

- **Technical Details**: 제안된 방법은 Gauss-Newton 근사를 Loss function에 적용하여 candidate network morphisms를 효율적으로 학습하고 평가합니다. 각 morphism을 적용했을 때 예상되는 손실 감소를 추정하고, 이 근사된 손실 함수를 역전파(backpropagation)를 사용해 최적화하여 morphism 매개변수를 학습합니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 분류 작업에 대해 실험한 결과, 제안된 NAS 방법이 유사한 또는 더 나은 품질의 아키텍처를 더 적은 계산 비용으로 학습하고, 최신 성능을 달성함을 나타냅니다.



### Harmful YouTube Video Detection: A Taxonomy of Online Harm and MLLMs as Alternative Annotators (https://arxiv.org/abs/2411.05854)
- **What's New**: 본 연구는 YouTube, Instagram, TikTok과 같은 단편 동영상 플랫폼에서 해로운 콘텐츠를 탐지하기 위한 새로운 측정 방법과 기법을 개발했습니다. 또한 해로움의 정의에 대한 종합적인 분류법을 제시하며, 이를 6가지 카테고리로 분류했습니다.

- **Technical Details**: 이 연구에서 개발한 분류법은 정보(Information), 증오 및 괴롭힘(Hate and harassment), 중독(Addictive), 클릭베이트(Clickbait), 성적(Sexual), 신체적(Physical) 해로움으로 구분됩니다. 19,422개의 YouTube 동영상을 14개의 이미지 프레임, 1개의 썸네일(Thumbnail), 텍스트 메타데이터를 사용하여 분석하였습니다.

- **Performance Highlights**: GPT-4-Turbo는 Crowdworkers (Mturk)보다 이진 분류(유해 vs 무해) 및 다중 레이블 해로움 분류 작업에서 더 높은 정확성을 보였습니다. 이 연구는 LLMs의 적용을 텍스트 주석 및 이진 분류를 넘어 다중 레이블 및 다중 모달(contexts)로 확장합니다.



### Reducing catastrophic forgetting of incremental learning in the absence of rehearsal memory with task-specific token (https://arxiv.org/abs/2411.05846)
- **What's New**: 이 논문에서는 새로운 데이터 학습 시 기존 데이터를 저장하지 않고도 이전 지식을 보존할 수 있는 혁신적인 방법을 제안합니다.

- **Technical Details**: 이 방법은 비전 변환기(vision transformer)의 구조에서 영감을 받았으며, 각 작업의 압축된 지식을 캡슐화할 수 있는 독특한 토큰(token)을 사용합니다. 작업에 따라 주의를 다르게 기울여 작업별 임베딩(task-specific embeddings)을 생성합니다.

- **Performance Highlights**: 우리의 모델은 여러 작업 증분 학습(task-incremental learning) 시나리오에서 벤치마크 데이터 세트를 사용하여 정확도와 역전이(backward transfer) 측면에서 성능을 측정하였으며, 비교한 방법들 중에서 가장 높은 정확도와 가장 낮은 역전이 성능을 기록했습니다.



### To Ask or Not to Ask? Detecting Absence of Information in Vision and Language Navigation (https://arxiv.org/abs/2411.05831)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 연구는 로봇이 불완전한 지침에 대해 명확한 질문을 할 수 있는 능력을 개발하는 것을 다룹니다. 특히 미흡한 정보를 인식하는 능력에 초점을 맞추고 있으며, 그로 인해 에이전트의 효율성을 향상시키는 방법을 제시합니다.

- **Technical Details**: 본 논문에서는 attention 기반의 instruction-vagueness estimation 모듈을 제안합니다. 이 모듈은 지침, 지금까지의 경로, 다음 이동 제안을 입력으로 받아, 각 시점에서 VLN 모델의 제안에 따라 행동할지 또는 도움을 요청할지를 결정합니다. 모듈은 instruction-to-path alignment 정보를 활용하여, precision-recall 균형에서 약 52% 향상된 성능을 보입니다.

- **Performance Highlights**: 제안된 IV estimation 모듈을 VLN-DUET 및 HAMT 두 내비게이터에 통합하여 효과와 일반화를 평가하였습니다. 실험 결과, instruction-to-path attention network의 attention 점수가 모호성 추정에 대한 더 나은 지표로 작용함을 보여주었습니다.



### SurfGNN: A robust surface-based prediction model with interpretability for coactivation maps of spatial and cortical features (https://arxiv.org/abs/2411.05825)
Comments:
          15 pages, 6 figures

- **What's New**: 이 논문에서는 기존 뇌표면 기반 예측 모델의 지역적 속성 변동성을 간과한 점을 개선하기 위해 Surface Graph Neural Network (SurfGNN)를 제안합니다. SurfGNN은 topology-sampling learning (TSL) 및 region-specific learning (RSL) 구조를 활용하여 각 피질 특성을 보다 효과적으로 관리합니다.

- **Technical Details**: SurfGNN은 피질 표면 메시를 희소 그래프로 간주하고, TSL과 RSL 구조를 통해 표면 메시의 낮은 및 높은 스케일에서 개별 피질 특성을 처리합니다. 이를 통해 복잡한 고밀도 그래프 구조의 문제를 해결하고, score-weighted fusion (SWF) 방법으로 각 피질 특성에 대한 노드 표현을 통합하여 예측을 수행합니다.

- **Performance Highlights**: SurfGNN은 481명의 환자(503 스캔)로부터 얻은 통합된 MRI 데이터를 사용하여 신생아의 뇌 나이를 예측하는 작업에서 기존의 모든 최신 방법을 능가하였으며, 평균 절대 오차(mean absolute error, MAE)는 0.827+0.056로 개선되었습니다.



### Navigating Distribution Shifts in Medical Image Analysis: A Survey (https://arxiv.org/abs/2411.05824)
- **What's New**: 본 논문은 Medical Image Analysis (MedIA)에서 distribution shifts 문제를 다루고 있으며, 이는 다양한 병원이나 지역, 환자 집단으로부터 오는 데이터의 변동성 때문에 발생하는 것이다. 이 연구는 이러한 문제를 해결하기 위한 다양한 deep learning (DL) 접근법을 체계적으로 정리한다.

- **Technical Details**: 본 논문은 DL 모델을 MedIA 시스템에 적용할 때의 distribution shifts 문제를 해결하기 위한 방법론을 제시한다. 구체적으로, Joint Training, Federated Learning, Fine-tuning, Domain Generalization으로 나누어 각 접근법의 적용 시나리오와 기술적 세부사항을 설명한다.

- **Performance Highlights**: 이 연구는 다양한 환경에서의 DL 모델의 유연성과 강인성을 높이는 전략을 제시하며, 각기 다른 의료 환경에서 모델을 성공적으로 활용할 수 있는 방법론을 개발하는 데 중점을 둔다.



### Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks (https://arxiv.org/abs/2411.05821)
Comments:
          16 Pages, 9 Figures

- **What's New**: 이번 연구에서는 비전-언어-행동(Vision-Language-Action, VLA) 모델의 평가 프레임워크와 벤치마크 스위트를 제시하여 다양한 로봇 작업에서 이 모델들을 체계적으로 평가하고자 하였습니다. GPT-4o, OpenVLA, JAT의 세 가지 최첨단 VLA 모델을 Open-X-Embodiment 컬렉션의 20개 데이터셋으로 평가하였으며, 이에 대한 통찰력을 제공하였습니다.

- **Technical Details**: VLA 모델들은 비디오와 언어의 이해를 결합하여 로봇 작업을 수행하는 능력을 보여주고 있으며, 특히 복잡한 조작 작업에서의 성능 차이가 두드러집니다. 이 연구는 로봇 학습 작업에 특화된 평가 분할 및 지표를 가지고 있으며, OpenX 데이터셋을 활용하여 다양한 로봇 플랫폼 및 작업 유형을 포함합니다. 또한, 모델 성능은 행동 공간 및 환경 요인에 민감하게 반응함을 확인했습니다.

- **Performance Highlights**: GPT-4o 모델은 정교한 프롬프트 엔지니어링을 통해 가장 일관된 성능을 보여주었고, 모든 모델이 다단계 계획을 요구하는 복잡한 조작 작업에서 어려움을 겪는 경향이 있으며, 20개의 다양한 데이터셋을 기반으로 한 벤치마크에서 성능을 비교하였습니다. 이 연구는 로봇 시스템의 발전을 위한 중요한 초석이 될 것입니다.



New uploads on arXiv(cs.AI)

### Tooling or Not Tooling? The Impact of Tools on Language Agents for Chemistry Problem Solving (https://arxiv.org/abs/2411.07228)
- **What's New**: 이 논문에서는 ChemCrow를 기반으로 한 ChemAgent가 개발되었으며, 다양한 화학 문제 해결을 위해 도구를 통합하여 성능을 평가했습니다. 그러나 ChemAgent는 도구 없이 사용할 때 기본 LLM보다 항상 뛰어난 성능을 보이지는 않았습니다.

- **Technical Details**: ChemAgent는 29개의 도구를 통합하여 전문화된 화학 작업 및 일반 화학 질문을 다룹니다. ReAct framework를 활용하고, SMolInstruct, MMLU-Chemistry, GPQA-Chemistry 벤치마크를 통해 평가되었습니다. 전문 화학 작업에 대해 도구 보강이 필요하며, 일반 화학 질문에서 도구 보강은 항상 도움이 되지 않는다는 점이 밝혀졌습니다.

- **Performance Highlights**: ChemAgent는 SMolInstruct의 전반적인 작업에서 ChemCrow보다 유의미한 성능 향상을 보였으나, LLM 도구 없이도 기본 모델이 항상 우수한 성능을 기록했습니다. 일반 화학 질문에 대한 처리능력이 부족하여 기본 LLM보다 성능이 떨어지기도 했습니다.



### 'Explaining RL Decisions with Trajectories': A Reproducibility Study (https://arxiv.org/abs/2411.07200)
- **What's New**: 이번 연구는 Deshmukh et al. (2023)의 'Explaining RL Decisions with Trajectories' 논문의 재현 가능성을 분석합니다. 이는 RL(강화학습)에서 결정의 해석 가능성을 높이는 새로운 방식을 제안하고 있습니다.

- **Technical Details**: 우리는 Grid-World 환경의 일부 코드와 나머지 환경들(Seaquest, HalfCheetah, Breakout, Q*Bert)을 처음부터 구현하여 연구를 진행했습니다. 원본 논문에서 주장한 네 가지 주요 주장 중 세 가지(학습 데이터에서의 주요 경로 삭제는 낮은 초기 상태 가치를 암시하며, 클러스터에서의 유사한 고수준 행동 제공, 먼 경로가 에이전트의 결정에 영향을 미친다.)는 부분적으로 확인하였습니다.

- **Performance Highlights**: 초기 상태 가치는 모든 경로를 포함하는 학습 데이터에서 높거나 같으며, 서로 다른 클러스터는 다른 해석 가능한 고수준 행동을 나타낸다는 것을 확인했습니다. 그러나 원작 실험의 한계로 인해 인간이 RL 에이전트의 결정에 영향을 미친 경로를 정확하게 식별할 수 있다는 주장은 지지할 수 없었습니다.



### A Domain-Agnostic Neurosymbolic Approach for Big Social Data Analysis: Evaluating Mental Health Sentiment on Social Media during COVID-19 (https://arxiv.org/abs/2411.07163)
Comments:
          13 Pages, 5 Figures, 5 Tables, 2024 IEEE International Conference on Big Data, Regular Paper

- **What's New**: 이 연구에서는 COVID-19와 관련된 정신 건강 트윗을 탐지하고 해석하기 위해 신경기호적(neurosymbolic) 방법을 도입하였습니다. 이는 전통적인 데이터 기반 접근 방식의 한계를 극복하고, 진화하는 언어에 적응하는 능력을 향상시킵니다.

- **Technical Details**: 본 방법은 120억 개의 트윗과 250만 개의 서브레딧 데이터, 70만 개의 뉴스 기사를 포함하는 대규모 데이터셋을 사용하여 평가하였으며, Zero-Shot Semantic Encoding and Decoding Optimization (SEDO) 프레임워크를 통해 새로운 용어와 기존 지식 개념 간의 의미적 유사성을 평가하고 적절한 콘텐츠 표현을 가능하게 합니다.

- **Performance Highlights**: 신경기호적 방법은 >92%의 F1 스코어를 기록하며 전통적인 데이터 기반 모델을 능가하였고, 전이 학습을 위한 자원 소모를 줄였습니다. 추가 실험에서도 pre-trained LLMs와 비교하여 우수한 성능을 나타냈습니다.



### Stronger Models are NOT Stronger Teachers for Instruction Tuning (https://arxiv.org/abs/2411.07133)
- **What's New**: 본 논문은 기존의 큰 모델이 더 강력한 교사가 될 것이라는 가정에 도전합니다. 저자들은 여러 실험을 통해 더 큰 또는 강력한 모델이 반드시 더 작은 모델에 대해 더 효과적인 교육자가 아니라는 사실을 밝혔습니다. 따라서 'Larger Models' Paradox'라는 현상을 제안합니다.

- **Technical Details**: 이 연구에서는 Compatibility-Adjusted Reward (CAR)라는 새로운 메트릭을 개발하여 다양한 응답 생성기의 효과성을 측정합니다. CAR는 호환성을 위험 요소로 삼아 응답의 평균 손실로 호환성을 정량화하며, 기존의 메트릭이 간과하는 효과를 고려합니다. 다섯 가지 기본 모델을 사용하여 CAR의 성능을 평가한 결과, 모든 기준선 모델을 능가했습니다.

- **Performance Highlights**: 응답 생성기 선택에서 기존 메트릭이 효과를 예측하는 데 실패한다는 사실이 드러났습니다. 더 큰 모델의 사용이 항상 더 나은 결과를 가져오지 않으며, 오픈 소스 모델들이 GPT-4보다 더 높은 성능을 보였다는 점이 인상적입니다. 저자들은 응답 생성기를 선택할 때 기존의 벤치마크 성능에 의존하기 보다는 호환성이 더 높은 모델을 우선시할 필요성을 강조합니다.



### Towards Characterizing Cyber Networks with Large Language Models (https://arxiv.org/abs/2411.07089)
Comments:
          5 pages, 2 figures

- **What's New**: 본 논문에서는 Cyber Log Embeddings Model (CLEM)을 통해 사이버 데이터의 잠재적 특성을 활용하여 이상 징후를 탐지하는 새로운 접근 방식을 제안합니다. CLEM은 실제 네트워크와 IoT 사이버 보안 테스트베드의 Zeek 네트워크 트래픽 로그를 사용하여 훈련되었습니다.

- **Technical Details**: CLEM은 자연어 처리(NLP) 기법을 활용하여 사이버 보안 로그에서 발견되는 언어적 구조를 모델링하며, 유사한 행동 패턴에 따라 기계들을 군집화(cluster)합니다. 모델은 Sliding Window 기술을 사용하여 데이터의 각 구간을 특성화하고, Adjusted Rand Index (ARI)를 통해 CLEM의 출력 결과와 전문가 레이블의 비교를 수행합니다.

- **Performance Highlights**: CLEM은 고차원적이고 복잡한 사이버 데이터 내에서 비정상적인 행동을 식별하는 데 있어 promise를 보여줍니다. 이 모델은 대규모 언어 모델(LLM)을 통해 사이버 데이터 이해에 기여하며, 실험 결과는 전통적인 방법에 비해 유의미한 성능 향상을 나타냅니다.



### OCMDP: Observation-Constrained Markov Decision Process (https://arxiv.org/abs/2411.07087)
Comments:
          Full paper, 14 Pages

- **What's New**: 이 논문에서는 Observation-Constrained Markov Decision Process (OCMDP)를 도입하여 비용 민감한 환경에서 관찰 및 제어 전략을 동시에 학습하는 문제를 다룹니다. 이는 전통적 제어 시스템에서 가정하는 완전 관측 가능성이 비현실적임을 인식하고, 정책이 실제 상태의 관측 가능성에 영향을 주는 구조입니다.

- **Technical Details**: 제안된 OCMDP는 상태 공간, 행동 공간, 관찰 집합, 상태 전이 함수, 관찰 함수, 보상 함수 및 비용 함수로 정의됩니다. 새로운 모델은 관찰 및 제어 정책을 분리하여 작동하며, 기계 학습으로 최적의 관찰 및 제어 행동을 고안합니다. 함께 학습하는 과정에서 ‘관찰을 무엇으로 할 것인지’와 ‘언제 관찰할 것인지’에 대한 의사결정이 중요합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 시뮬레이션 진단 작업 및 현실적인 의료 환경에서 기존의 방법들보다 평균적으로 관찰 비용을 크게 줄이며, 효율성에서 두드러진 성과를 보여줍니다. 이를 통해 실제 의료 결정 과정에서의 적용 가능성을 입증하였습니다.



### To Train or Not to Train: Balancing Efficiency and Training Cost in Deep Reinforcement Learning for Mobile Edge Computing (https://arxiv.org/abs/2411.07086)
- **What's New**: 이 논문에서는 6G 네트워크에서 인공지능(AI)의 역할을 다루고 있으며, Mobile Edge Computing (MEC)의 효율적인 리소스 할당 문제에 대해 다룹니다. 기존 연구들이 교육 비용을 간과했던 점을 지적하며, 더 현실적인 상황에서의 모델을 제안합니다.

- **Technical Details**: 제안된 새로운 알고리즘은 Deep Reinforcement Learning (DRL) 에이전트의 교육 시점을 동적으로 선택하는 방식으로, 교육 오버헤드를 고려한 시스템입니다. 이 방법은 다양한 훈련 비용이 있는 시나리오에 직접 적용할 수 있으며, 이상적인 학습 에이전트와 유사한 성능을 달성할 수 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 교육 비용을 반영하는 상황에서도 이상적인 성능에 가까운 결과를 보여 줍니다. 따라서 이 접근 방식은 실제 환경에서의 효율적인 리소스 관리에 큰 기여를 할 수 있습니다.



### Designing Reliable Experiments with Generative Agent-Based Modeling: A Comprehensive Guide Using Concordia by Google DeepMind (https://arxiv.org/abs/2411.07038)
- **What's New**: 이번 논문에서는 Generative Agent-Based Modeling (GABM) 프레임워크를 사용하여 신뢰할 수 있는 실험을 설계하는 방법을 제시합니다. 특히, AI 기반의 에이전트가 복잡한 행동을 생성할 수 있도록 하여 여러 분야의 연구자들이 고급 시뮬레이션 기술에 접근할 수 있도록 돕는 방안을 모색합니다.

- **Technical Details**: GABM은 일반적인 agent-based modeling (ABM)보다 발전된 접근 방식으로, 사전에 프로그래밍된 규칙이 아닌 데이터를 기반으로 행동을 생성하고 학습하는 저작(Generative) 모델을 통합합니다. 이를 통해 에이전트들은 더 인간적인 의사결정 과정을 반영할 수 있으며, Google DeepMind가 개발한 Concordia 플랫폼을 이용하여 GABM 기반의 대규모 시뮬레이션을 디자인하고 실행할 수 있습니다.

- **Performance Highlights**: GABM을 활용하면 연구자들은 복잡한 사회적 상호작용 및 인간 행동을 모방하는 다양한 시뮬레이션을 수행할 수 있습니다. 이러한 시뮬레이션은 의사결정 프로세스, 집단 행동, 사회적 네트워크의 영향을 심층적으로 이해하는 데 도움을 줄 수 있으며, 현재의 데이터 수집 방법론에서 벗어나 새로운 연구 기회를 제공할 수 있습니다.



### Evaluating the Accuracy of Chatbots in Financial Literatur (https://arxiv.org/abs/2411.07031)
- **What's New**: 본 연구는 ChatGPT (4o 및 o1-preview 버전)와 Gemini Advanced 챗봇의 금융 문헌 참고 제공 신뢰성을 평가하였습니다. 기존의 이진(binary) 접근 방식 외에도 비이진(nonbinary) 접근 방식과 최신성(recency) 측정을 개발하여 주제의 최신성에 따른 환각(hallucination) 비율 변화를 분석했습니다.

- **Technical Details**: 150개의 인용을 분석한 결과, ChatGPT-4o의 환각 비율은 20.0% (95% 신뢰구간, 13.6%-26.4%)였고, o1-preview는 21.3% (95% 신뢰구간, 14.8%-27.9%)로 나타났습니다. 반면, Gemini Advanced는 76.7% (95% 신뢰구간, 69.9%-83.4%)로 높은 환각 비율을 보였습니다.

- **Performance Highlights**: 최신 주제에 대해 환각 비율이 증가하는 경향이 관찰되었으나, Gemini Advanced의 경우 통계적으로 유의미하지 않았습니다. 이러한 결과는 신속하게 변화하는 분야에서 챗봇이 제공하는 참고 문헌의 검증의 중요성을 강조합니다.



### Permutative redundancy and uncertainty of the objective in deep learning (https://arxiv.org/abs/2411.07008)
Comments:
          22 pages, 3 figures

- **What's New**: 본 논문은 불확실한 목적 함수(uncertain objective functions)와 전통적인 딥 러닝 아키텍처의 순열 대칭(permutative symmetry)에 대한 문제를 다룹니다. 전통적인 아키텍처들이 엄청난 수의 동등한 글로벌(optima) 및 로컬(optima) 최적값에 의해 오염된다는 점을 설명합니다. 네트워크의 크기가 커짐에 따라 글로벌 최적화 경관(global optimization landscape)이 복잡하게 얽히게 된다는 주장을 제시합니다.

- **Technical Details**: 기존의 딥 러닝 아키텍처는 확률적 경량 최적화(stochastic gradient descent) 기술을 사용하여 학습됩니다. 그러나 불확실한 목적(such as aleatoric uncertainty)에 대한 처리가 한계가 있으며, 수량이 방대해야 수치적 불확실성(statistical uncertainty)을 완화할 수 있습니다. 논문에서 제안하는 해결책으로는 강제 전처리(forced pre-pruning), 재정렬(re-ordering), 직교 다항 활성화(ortho-polynomial activations), 모듈형 생체 영감을 받은 아키텍처가 포함됩니다.

- **Performance Highlights**: 전통적 딥 러닝 아키텍처들은 여전히 현장에서 일정 부분 성공을 거두고 있으나, 불확실성과 목표의 변동성으로 인해 데이터 의존성이 두드러지며, 일부 경우에는 비효율적인 결과를 초래할 수 있습니다. 이 논문에서 논의된 기법들은 이러한 문제를 해결하기 위한 방향을 제시하고 있습니다.



### Estimating Causal Effects in Partially Directed Parametric Causal Factor Graphs (https://arxiv.org/abs/2411.07006)
Comments:
          Accepted to the Proceedings of the 16th International Conference on Scalable Uncertainty Management (SUM 2024)

- **What's New**: 이번 논문에서는 인과 추론(causal inference)을 부분적으로 지향적인 그래프(partially directed graphs)에 적용하는 방법을 제시합니다. 부분적으로 지향적인 인과 인자(parametric causal factor graphs, PCFGs)의 일반화로서 부분적으로 지향적인 인과 인자 그래프(partially directed parametric causal factor graphs, PPCFGs)를 소개하며, 인과적 관계에 대한 사전 지식이 적게 필요한 확장된 인과 추론을 가능하게 합니다.

- **Technical Details**: PPCFGs는 지향된(edge) 및 비지향된(edge) 관계를 모두 포함하는 그래프 형태로, 주어진 랜덤 변수(random variables) 집합에 대한 전체 결합 분포(joint distribution)를 압축적으로 인코딩할 수 있습니다. PPCFG에서 d-separation을 정의하여 조건부 독립성(conditional independence)을 판단할 수 있는 기초를 마련하고, 인과 효과(causal effect)의 효율적인 추정 알고리즘을 제시합니다.

- **Performance Highlights**: PPCFG를 적용한 인과 효과 추정 알고리즘은 동일한 표본을 대표하는 객체를 사용하여 추론(inference)의 속도를 높이며, 비지향적인 엣지가 포함된 경우에도 가능한 모든 인과 효과를 효율적으로 나열할 수 있습니다. 이를 통해 실제 데이터와 부분적으로 지향적인 그래프를 사용하여 인과 효과를 추정하는 데 있어 실용적인 접근을 제시합니다.



### Which PPML Would a User Choose? A Structured Decision Support Framework for Developers to Rank PPML Techniques Based on User Acceptance Criteria (https://arxiv.org/abs/2411.06995)
- **What's New**: 이 논문에서는 사용자 선호에 기반하여 Privacy Preserving Machine Learning (PPML) 기술을 선택하는 데 도움이 되는 의사결정 지원 프레임워크를 제안합니다.

- **Technical Details**: Privacy-Enhancing Technologies (PETs) 및 User Acceptance Criteria (UAC)를 기반으로 PPML 기술의 다양한 특성을 분석하여 기술적인 차별화를 도출합니다. 이를 통해 PPML 방법을 사용 사례에 맞추어 평가합니다.

- **Performance Highlights**: 이 연구는 사용자 수용 기준에 따라 PPML 기술을 순위 매기는 방법을 제공하며, 개인정보에 관련된 정보를 분류하는 사례를 통해 응용 프로그램을 시연합니다.



### Multi-modal Iterative and Deep Fusion Frameworks for Enhanced Passive DOA Sensing via a Green Massive H2AD MIMO Receiver (https://arxiv.org/abs/2411.06927)
- **What's New**: 본 연구에서는 H$^2$AD 배열을 위한 저비용 및 높은 시간 효율성을 갖춘 보다 실용적인 방향 추정(DOA) 기법을 제안합니다. 새로운 다중 모달 융합(DOA) 프레임워크를 통해 이상적인 소스 입사 각도를 가정하지 않으며, 두 가지 고효율 클러스터링 방법(GMaxCS 및 GMinD)을 통해 더 정확한 해답을 추론합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 접근 방식인 MM-IWF-GMaxCS 및 MM-IWF-GMinD를 포함하며, 이는 반복적인 가중치 융합(IWF) 방법을 기반으로 합니다. 또한, MM-fusionNet-GMaxCS 및 MM-fusionNet-GMinD의 융합 네트워크(fusionNet)를 통해 두 부분의 추론된 진각을 집계하고 더 높은 정확한 DOA 추정을 달성합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 네 가지 방법은 이상적인 DOA 성능 및 Cramer-Rao Lower Bound (CRLB)를 달성할 수 있으며, 특히 MM-fusionNet-GMaxCS와 MM-fusionNet-GMinD의 성능이 낮은 신호 대 잡음비(SNR) 범위에서 뛰어난 DOA 성능을 보여줍니다.



### Learning Interpretable Network Dynamics via Universal Neural Symbolic Regression (https://arxiv.org/abs/2411.06833)
Comments:
          preprint

- **What's New**: 이 연구에서는 복잡한 네트워크 동역학의 지배 방정식을 자동으로 배우고, 효율적으로 식별할 수 있는 범용 계산 도구 LLC (Learning Law of Changes)를 개발했습니다. 이 도구는 심층 학습의 탁월한 적합 능력과 사전 훈련된 기호 회귀의 방정식 추론 능력을 결합하여 복잡한 시스템 상태의 기호 변화 패턴을 배웁니다.

- **Technical Details**: LLC는 네트워크 동역학의 관측 데이터를 통해 일반적인 미분 방정식 (Ordinary Differential Equations, ODEs)을 발견합니다. 관측 데이터는 연속 상태 데이터에서 추출되며, 초기 상태와 토폴로지를 기반으로 초기 값 문제를 해결하여 얻은 것입니다. 이 도구는 노이즈가 있거나 토폴로지가 결여된 데이터에서도 지배 방정식을 정확하고 효율적으로 발견할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: 이 도구의 효과성을 증명하기 위해 물리학, 생화학, 생태학, 전염병학 등 10개 이상의 대표적인 시나리오에서 실험을 수행했습니다. 그 결과, LLC는 혼돈 시스템 및 실제 시스템(예: 글로벌 전염병 전파 및 보행자 움직임)을 포함하여 뛰어난 효율성과 효과를 보여주었습니다.



### Combining Domain and Alignment Vectors to Achieve Better Knowledge-Safety Trade-offs in LLMs (https://arxiv.org/abs/2411.06824)
- **What's New**: 최근 도메인 전문 LLMs (Large Language Models)에 대한 관심이 증가하고 있으며, 이러한 모델의 안전성을 유지하면서 효율적으로 개발하기 위한 방법인 MergeAlign을 소개합니다. MergeAlign은 도메인 및 정렬 벡터를 보간(interpolate)하여 도메인 전문 모델을 안전하게 만드는 방법입니다.

- **Technical Details**: MergeAlign은 기존의 일반 목적 모델과 도메인 전문 모델의 벡터를 결합하여 안전 정렬을 수행하는 방법입니다. 이 과정에서 모델 병합(model merging) 기법을 활용하며, 도메인 벡터(domain vector)와 정렬 벡터(alignment vector)의 선형 보간을 통해 수행됩니다. 결과적으로 도메인 전문 모델의 유용성을 해치지 않으면서 안전성을 높일 수 있습니다.

- **Performance Highlights**: MergeAlign을 사용하여 의학 및 금융 분야의 Llama3 변형 모델에서 실험한 결과, 도메인 기준에서 성능 저하 없이 안전성 정렬 성능이 크게 향상됨을 확인하였습니다. MergeAlign 모델은 기존의 선호 정렬(preference tuning) 방법보다 지식-안전성 트레이드 오프가 감소하며, 보다 비용 효율적으로 안전한 도메인 전문 모델로 발전시킬 수 있는 가능성을 보여주었습니다.



### Generative midtended cognition and Artificial Intelligence. Thinging with thinging things (https://arxiv.org/abs/2411.06812)
Comments:
          16 pages, 2 figures. Submitted to "Synthese" Journal, accepted

- **What's New**: 본 논문은 'generative midtended cognition'이라는 개념을 소개하며, 생성적 AI와 인간의 인지 통합을 탐구합니다. 기존의 인지 이론에서 간과된 한계를 극복하기 위한 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 이 논문에서는 multimodal transformer architectures에 기반한 현재의 생성 기술을 검토하고, ChatGPT와 같은 대형 언어 모델이 인간의 인지 작용을 어떻게 전환시킬 수 있는지를 설명합니다. 'Generative midtended cognition'은 두 가지 차원, 즉 Width(맥락의 민감성)와 Depth(반복 루프의 세분화)를 포함합니다.

- **Performance Highlights**: 생성적 AI의 보편화는 다양한 창의적 출력의 질에 유의미한 변화를 가져왔으며, 이는 전통적인 인지 및 의도적 창조의 개념을 재정의하는 데 기여하고 있습니다. 그러나 이러한 과정은 정체성과 창의성의 본질에 대한 새로운 도전과제를 제기하고 있습니다.



### JPEG AI Image Compression Visual Artifacts: Detection Methods and Datas (https://arxiv.org/abs/2411.06810)
- **What's New**: 최근 학습 기반 이미지 압축 방법이 전통적인 코덱을 능가하기 시작했습니다. 하지만, 신경망 접근 방식은 예상치 못한 시각적 아티팩트를 도입할 수 있습니다. 이 논문에서는 이러한 아티팩트를 감지하고 지역을 국소화하며 강도를 정량화하는 방법을 제안합니다.

- **Technical Details**: 본 연구에서 제안한 방법은 세 가지 유형의 아티팩트 (텍스처 및 경계 손상, 색상 변화, 텍스트 손상)를 각각 감지합니다. 약 350,000개의 고유 이미지를 다양한 압축 품질 매개변수로 처리한 후 46,440개의 아티팩트를 포함하는 데이터셋이 구축되었으며, 이 데이터셋은 주관적 평가를 통해 검증되었습니다.

- **Performance Highlights**: 우리의 방법은 기존의 이미지 품질 평가 방법들과 비교하여 주관적 평가와의 상관관계가 높음을 입증하였습니다. 이는 신경망 기반 이미지 코덱을 평가하고 성능을 높이기 위해 매우 중요한 결과입니다.



### MP-PINN: A Multi-Phase Physics-Informed Neural Network for Epidemic Forecasting (https://arxiv.org/abs/2411.06781)
- **What's New**: 이 논문에서는 COVID-19와 같은 전염병의 확산 예측을 위해 새로운 하이브리드 방법론인 MP-PINN(Multi-Phase Physics-Informed Neural Network)을 제안합니다. 이 방법은 전통적인 기계적 모델과 데이터 기반 방법의 한계를 극복하는 것을 목표로 하고 있습니다.

- **Technical Details**: MP-PINN은 신경망에 전파 메커니즘을 주입하여 정책 개입에 따른 전염병의 역학 변화를 단계적으로 반영합니다. 이를 통해 단기 및 장기 예측에서 더 나은 성능을 제공합니다. SIR(Susceptible-Infectious-Recovered) 모델의 여러 단계에 걸쳐 각기 다른 SIR 매개변수를 사용할 수 있도록 하여 적응성을 추가했습니다.

- **Performance Highlights**: MP-PINN은 이탈리아의 COVID-19 데이터 세트를 기반으로 한 실험에서 전통적인 SIR 모델, 순수 데이터 기반 접근 방식, 단일 단계 PINN보다 단기 및 장기 예측 모두에서 우수한 성능을 보였습니다.



### A Text Classification Model Combining Adversarial Training with Pre-trained Language Model and neural networks: A Case Study on Telecom Fraud Incident Texts (https://arxiv.org/abs/2411.06772)
- **What's New**: 이 논문은 경찰의 통신 사기 사건을 보다 효과적으로 분류하기 위한 새로운 텍스트 분류 모델을 제안합니다. 이 모델은 전통적인 수작업 분류 방식을 대신하여, 효율성과 정확성을 높이기 위해 적대적 학습(adversarial training) 및 사전 훈련된 언어 모델(Pre-trained Language Model)과 신경망(neural networks)을 결합합니다.

- **Technical Details**: 모델은 언어적 특징(feature) 추출을 위해 언어학적으로 동기 부여된 Pre-trained Language Model을 이용하며, Fast Gradient Method 알고리즘을 사용하여 생성된 임베딩 레이어(embedding layer)에 변화를 줍니다. 이후, Bi-directional Long Short-Term Memory와 Convolutional Neural Networks를 통해 각각 문맥 구문 정보와 지역 의미 정보를 추출합니다.

- **Performance Highlights**: 이 모델은 운영 부서에서 제공한 통신 사기 사건 데이터의 일부로 훈련되었고, 83.9%의 분류 정확도를 달성했습니다. 모델은 운영 부서에 배포되어 인력을 상당히 절감하고 통신 사기 범죄에 대한 부서의 효율성을 향상시켰습니다.



### KLCBL: An Improved Police Incident Classification Mod (https://arxiv.org/abs/2411.06749)
- **What's New**: 이번 연구는 경찰 사건 데이터를 효과적으로 분류하기 위한 새로운 멀티채널 신경망 모델 KLCBL을 제안합니다. 이 모델은 전통적인 분류 방식의 한계를 극복하고 데이터의 특성을 고려합니다.

- **Technical Details**: KLCBL은 Kolmogorov-Arnold Network (KAN), 언어적 향상을 통한 텍스트 전처리 접근법 (LERT), 합성곱 신경망 (CNN), 양방향 장단기 기억 네트워크 (BiLSTM)를 통합하여 구성됩니다. 이러한 다양한 신경망 구조를 통해 데이터 처리 능력을 극대화합니다.

- **Performance Highlights**: KLCBL 모델은 실제 데이터로 평가했을 때 91.9%의 정확도를 기록하며 기존의 기준 모델들을 능가했습니다. 이 모델은 경찰의 정보화 수준을 높이고 자원 배분을 개선하여 다양한 분류 작업에 널리 적용될 수 있는 가능성을 보여줍니다.



### Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data (https://arxiv.org/abs/2411.06735)
Comments:
          21 pages, 4 tables, 2 figures

- **What's New**: 본 논문에서는 TimeText Corpus (TTC)라는 새로운 멀티모달(multi-modal) 데이터셋을 개발하였으며, 기후과학(climate science) 및 헬스케어(healthcare) 분야의 데이터가 포함되어 있습니다. 이 데이터셋은 시간이 정렬된 숫자(숫자 시퀀스)와 텍스트 데이터로 구성되어 있어, 시간 시리즈(time series)와 텍스트 데이터를 동시에 예측하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: TimeText Corpus는 정렬된 텍스트와 시간 및 숫자 시퀀스를 포함하여, 다양한 멀티모달 예측을 가능하게 하고, Hybrid Multi-Modal Forecaster (Hybrid-MMF)라는 모델을 제안합니다. Hybrid-MMF는 공유 임베딩(shared embeddings)을 활용하여 텍스트와 숫자 데이터를 동시에 예측합니다.

- **Performance Highlights**: Hybrid-MMF 모델은 실험에서 기존의 기준 모델들에 비해 특별한 성과를 보여주지는 않았으나, 멀티모달 예측의 복잡성을 강조합니다. 이는 우리가 예상했던 것과는 다르게, 멀티모달 예측의 도전 과제가 크다는 것을 잘 보여줍니다.



### Ambient AI Scribing Support: Comparing the Performance of Specialized AI Agentic Architecture to Leading Foundational Models (https://arxiv.org/abs/2411.06713)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.15528

- **What's New**: 이 연구에서는 Sporo Health의 AI Scribe를 다양한 LLMs(Model)과 비교하여 의료 기록 작성 분야에서의 성능을 평가하였습니다.

- **Technical Details**: Sporo Health의 AI Scribe는 의료 기록을 위한 특수 모델로, 임상 노트(SOAP notes)를 바탕으로 훈련되었습니다. 연구에서는 제휴 클리닉에서 얻은 개인정보가 제거된 환자 기록을 분석하였으며, zero-shot prompting 기법을 사용하여 모델이 SOAP 요약을 생성하도록 했습니다.

- **Performance Highlights**: Sporo는 73.3%의 recall, 78.6%의 precision, 75.3%의 F1 점수를 기록하며 모든 모델 중에서 가장 우수한 성과를 보였습니다. 통계적으로 유의미한 차이(p < 0.05)가 발견되었으며, Sporo는 GPT-3.5, Gemma-9B, Llama 3.2-3B와 비교해 유의미한 개선이 있었습니다. 사용자의 만족도 조사에서도 Sporo가 더 높은 정확성과 관련성을 보였습니다.



### Anytime Probabilistically Constrained Provably Convergent Online Belief Space Planning (https://arxiv.org/abs/2411.06711)
Comments:
          arXiv admin note: text overlap with arXiv:2302.10439 by other authors

- **What's New**: 이 논문에서는 확률적 신념 의존성 제약(probabilistic belief-dependent constraints) 구성의 최신 발전을 기반으로 하여, Monte Carlo Tree Search(MCTS) 방법을 활용한 안전한 자율 로봇 의사결정 접근 방식을 제안합니다. 이 접근 방식은 검색 트리의 수렴에 의존하지 않고도 언제든지 안전성을 보장합니다.

- **Technical Details**: 연속적 상태에서의 Partially Observable Markov Decision Process(POMDP)에 대한 안전한 행동 분석을 통해, MCTS 기반 알고리즘을 개발했습니다. 이 알고리즘은 신념 트리에서 위험한 동작을 제거하고, 안전한 동작에 해당하는 통계값을 지속적으로 수정하여 매우 제한적인 수의 트리 쿼리로도 안전한 최적 동작을 찾을 수 있도록 해줍니다.

- **Performance Highlights**: 우리가 제안한 방법은 기존 기준선보다 항상 더 안전한 동작을 찾아내며, 목표에 대한 측면에서도 기준선보다 우수한 성능을 지속적으로 보입니다. 실제 시뮬레이션을 통해 이러한 효용성을 입증했습니다.



### Model Fusion through Bayesian Optimization in Language Model Fine-Tuning (https://arxiv.org/abs/2411.06710)
- **What's New**: 본 논문은 Pretrained Language Models (plms)의 미세 조정과 관련된 문제를 해결하기 위해 Bayesian Optimization Model Fusion (bomf)이라는 새로운 기법을 소개합니다. 모델 퓨전은 여러 모델을 결합하는 방식으로, 성능을 개선하고 최적의 하이퍼파라미터 선택을 지원하는 혁신적인 접근법을 제공합니다.

- **Technical Details**: 논문에서는 Multi-Objective Bayesian Optimization (mobo)를 사용하여 손실 함수(loss function)와 성능 지표(metric)를 동시에 고려하여 모델을 퓨전하는 새로운 방법을 제안합니다. 또한, 하이퍼파라미터 선택을 위한 두 단계의 Bayesian 최적화 절차를 구축하여 모델 퓨전 과정을 효율적으로 수행합니다. 이를 통해 다양한 NLP(자연어 처리) 태스크에서 성능을 큰 폭으로 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: experiments conducted with various tasks, including Natural Language Understanding (nlu) and Natural Language Generation (nlg), demonstrate significant performance improvements using the proposed bomf method on models such as roberta, T5, and llama.



### Autonomous Droplet Microfluidic Design Framework with Large Language Models (https://arxiv.org/abs/2411.06691)
- **What's New**: MicroFluidic-LLMs라는 프레임워크가 새로운 방식으로 표 형식의 데이터를 처리하고, 중요한 맥락 정보를 추출하여 드롭렛 기반의 마이크로플루이딕 장치 디자인을 자동화하는 데 기여합니다.

- **Technical Details**: 이 프레임워크는 데이터를 언어 형식으로 변환하고, 사전 훈련된 대형 언어 모델(pre-trained large language models, LLMs)을 활용하여 분석합니다. 11개의 예측 작업에서 기하학(geometry), 흐름 조건(flow conditions), 레짐(regimes) 및 성능(performance)에 대한 평가가 이루어졌습니다.

- **Performance Highlights**: MicroFluidic-LLMs는 Deep Neural Network 모델의 성능을 크게 향상시킴으로써 드롭렛 지름(droplet diameter)과 생성률(generation rate)의 평균 절대 오차(mean absolute error)를 각각 약 5배 및 7배 줄였고, 레짐 분류 정확도를 4% 이상 향상시켰습니다.



### Adversarial Detection with a Dynamically Stable System (https://arxiv.org/abs/2411.06666)
- **What's New**: 본 연구에서는 입력 예제의 안정성을 기반으로 적대적 예제(Adversarial Examples, AEs) 탐지를 위한 Dynamically Stable System (DSS)를 제안합니다. DSS는 Lyapunov 동적 시스템의 안정성을 사용하여 일반 예제와 적대적 예제를 효과적으로 구별할 수 있는 새로운 메커니즘을 갖추고 있습니다.

- **Technical Details**: DSS는 입력 예제를 방해하고 복원하는 프로세스를 통해 입력 예제의 안정성 특징을 추출합니다. 이 시스템은 제어 항을 도입하여 정상 예제가 원래 안정성을 유지하고, 적대적 예제는 안정성을 잃도록 유도합니다. 이를 통해 DSS는 입력 예제의 안정성 변화에 따라 적대적 예제를 탐지할 수 있습니다.

- **Performance Highlights**: MNIST, CIFAR10, CIFAR100의 세 가지 벤치마크 데이터셋에서 평균 ROC-AUC 값이 각각 99.83%, 97.81%, 94.47%를 기록하였으며, 이는 기존 7개 방법의 SOTA(최첨단 성능, State-of-the-Art) 값을 능가합니다.



### Predicting Country Instability Using Bayesian Deep Learning and Random Fores (https://arxiv.org/abs/2411.06639)
- **What's New**: 본 연구는 국가 불안정성(country instability) 예측을 위한 새로운 AI 기반 접근 방식을 제안합니다. GDELT 프로젝트(Global Database of Activities, Voice, and Tone) 데이터셋을 사용하여 정치적 갈등(polical conflict)을 보다 정교하게 분석하고자 합니다.

- **Technical Details**: GDELT 데이터셋은 2012년에 처음 출시되었으며, 매일 100개 이상의 언어로 보도된 뉴스(뉴스 기사, 방송 및 웹 뉴스)를 실시간으로 기록합니다. 이 데이터셋은 사회적 네트워크(social networks)와 글로벌 경제(global economy)의 상호 연결성을 포함하여, 방대한 대용량 빅데이터(big data)를 활용하여 국가 예측 모델을 개발하는 데 기여하는 것을 목표로 합니다.

- **Performance Highlights**: GDELT 프로젝트는 매우 정교한 데이터 수집과 분석을 통해, 정치적 갈등에 대한 예측의 정확성을 제고할 수 있는 잠재력을 가지고 있습니다. 연구 결과는 사회경제적 성장에 긍정적인 영향을 미칠 수 있을 것으로 기대됩니다.



### A Review of Fairness and A Practical Guide to Selecting Context-Appropriate Fairness Metrics in Machine Learning (https://arxiv.org/abs/2411.06624)
Comments:
          15 pages, 4 figures, 1 table

- **What's New**: 이 논문은 최근 인공지능(AI) 분야의 규제 제안들이 강조하는 공정성(fairness) 요구사항에 대한 문제의식을 다루고 있습니다. 문맥(context) 인식을 통해 적절한 공정성 측정을 선택하는 기준이 필요하다는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 공정성을 평가하기 위한 흐름도(flowchart)를 개발하였으며, 여기에는 모델 평가 기준(model assessment criteria), 모델 선택 기준(model selection criteria), 데이터 편향(data bias) 등을 고려한 12가지 기준이 포함되어 있습니다. 또한, 공정성 관련 문헌을 검토하고 이를 핵심 규제 도구(core regulatory instruments)와 연계하여 정책 입안자, AI 개발자, 연구자 및 기타 이해관계자가 공정성 문제를 적절히 해결하는 데 도움을 줍니다.

- **Performance Highlights**: 이 흐름도를 통해 사용자들은 제공된 문맥에 따른 적절한 공정성 기준을 선택할 수 있어, 점점 엄격해지는 규제 요구사항에 효과적으로 대응할 수 있는 방법을 제시합니다.



### MEANT: Multimodal Encoder for Antecedent Information (https://arxiv.org/abs/2411.06616)
- **What's New**: 이번 연구에서는 주식 시장의 멀티모달 데이터에 대한 새로운 접근 방식을 제시합니다. MEANT 모델을 통해 다중 정보를 처리하면서 시간 종속적인 요구를 충족시키며, 새로운 데이터세트인 TempStock을 소개합니다. 이 데이터세트는 S&P 500에 속하는 모든 기업에 대한 가격 정보, 트위트 및 그래픽 데이터로 구성되어 있으며, 100만 개 이상의 트위트가 포함되어 있습니다.

- **Technical Details**: MEANT(Model) 모델은 템포럴(focused self-attention) 메커니즘을 갖춘 멀티모달 아키텍처로, 가격 정보를 멀리 있는 장기 관계와 소셜 미디어 언어 피쳐를 처리합니다. TimeSFormer 아키텍처가 이미지 피쳐를 추출하는 데 사용되며, 트위트 데이터를 통해 더 즉각적인 트렌드를 파악합니다. TempStock 데이터세트는 변경 지연(lag)에 따라 다양한 청크로 시퀀스를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: MEANT 모델은 기존 벤치마크 대비 15% 이상의 성능 향상을 보여주었으며, 문자열(텍스트) 정보가 시각적 정보보다 성능에 훨씬 더 큰 영향을 미친다는 것을 입증했습니다. 기본적으로 시간 시계열 예측으로서 주식 가격 및 시장 변동에 대한 예측 분류 문제를 해결하는 데 높은 성과를 보였습니다.



### Gen-AI for User Safety: A Survey (https://arxiv.org/abs/2411.06606)
- **What's New**: 이번 논문에서는 Generative Artificial Intelligence (Gen-AI) 기술이 사용자 안전을 강화하는 다양한 도메인에서 어떻게 활용될 수 있는지에 대한 종합적인 개요를 제공합니다. 이 기술은 사이버 공격 및 거짓 정보와 같은 사용자 안전 침해를 탐지하는 데 있어 뛰어난 성능을 발휘합니다.

- **Technical Details**: Gen-AI 기술은 NLP (Natural Language Processing), 이미지 인식, 비디오 및 오디오 데이터를 분석하는 데 사용될 수 있으며, 다중 데이터 모달리티를 참조하여 사용자 안전 위반을 탐지합니다. 예를 들어, 고급 과제 수행을 위한 Retrieval-Augmented-Generation (RAG) 등의 기법들은 보다 정교한 피싱 및 악성 소프트웨어 공격 탐지에 기여할 수 있습니다. 또한, Gen-AI 기술은 심리적 지원 및 접근성을 향상시키는 데에도 적용됩니다.

- **Performance Highlights**: 이 연구는 Gen-AI 기술이 기존의 기계 학습 기법에 비해 사용자 안전 위반을 탐지하는 데 있어서 더 높은 정확도를 제공합니다. 예를 들어, Deepfake 탐지와 관련된 접근 방식인 Temporal Dropout 3D Convolutional Neural Network (TD-3DCNN)이 최신 기술을 초월하는 성능을 발휘했습니다.



### OffLight: An Offline Multi-Agent Reinforcement Learning Framework for Traffic Signal Contro (https://arxiv.org/abs/2411.06601)
- **What's New**: 이 논문에서는 교통 신호 제어(traffic signal control, TSC)를 위해 고안된 OffLight라는 새로운 오프라인 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL) 프레임워크를 소개합니다. 이 프레임워크는 Importance Sampling(IS)과 Return-Based Prioritized Sampling(RBPS)을 결합하여 정책의 이질성을 다룹니다.

- **Technical Details**: OffLight는 GMM-VGAE 모델을 사용하여 실제 교통 네트워크 내에서의 다양한 행동 정책의 이질성을 포착합니다. 또한, Graph Neural Networks(GNNs)를 활용하여 교차점 간의 지역 관측을 구조화된 글로벌 표현으로 통합합니다. 이를 통해 OffLight는 IS를 효과적으로 활용하여 배포 변화(distributional shifts)를 교정하고 다양한 데이터로부터 안정적인 정책 학습을 보장합니다.

- **Performance Highlights**: OffLight는 실제 교통 데이터세트에서 기존 방법보다 우수한 성능을 보임을 입증하였으며, 샘플 효율성과 정책 성능을 향상시킵니다. 이는 특히 고수익 에피소드를 우선적으로 학습하면서 수렴을 가속화하는 데 기여합니다.



### Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents (https://arxiv.org/abs/2411.06559)
Comments:
          18 pages, 6 figures, 4 tables

- **What's New**: 이번 연구는 웹 기반 작업을 자동화하는 언어 에이전트를 향상시키기 위해 모델 기반 계획을 통합하는 새로운 패러다임을 제안합니다. WebDreamer는 대형 언어 모델(LLM)을 복잡한 웹 환경의 세계 모델로 사용하는 혁신적인 접근 방식을 보여줍니다.

- **Technical Details**: WebDreamer는 각 후보 작업에 대한 결과를 시뮬레이션하기 위해 LLM을 활용하며, 자연어 설명을 통해 상태 변화의 결과를 상상합니다. 그 후, 이러한 결과를 평가하여 최적의 작업을 결정합니다. 이 접근법은 안전 위험을 줄이고 에이전트의 탐색 및 계획 능력을 유지합니다.

- **Performance Highlights**: VisualWebArena 및 Mind2Web-live의 두 가지 대표적인 웹 에이전트 벤치마크에서 WebDreamer는 반응형 기초선보다 유의미한 성능 향상을 달성했습니다. 이 연구의 결과는 LLM 기반 세계 모델의 가능성을 검증하고, 복잡한 웹 환경에서의 자동화된 상호작용에 대하여 새로운 연구 방향성을 제시합니다.



### In-Context Learning for Preserving Patient Privacy: A Framework for Synthesizing Realistic Patient Portal Messages (https://arxiv.org/abs/2411.06549)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 8 pages

- **What's New**: COVID-19 대유행 이후, 환자 포털 메시지가 증가하고 있는데, 이는 임상의사의 소진에 큰 기여를 하고 있습니다. 본 논문에서는 HIPAA 친화적이며 현실적인 환자 포털 메시지를 생성하기 위한 LLM 기반의 새로운 프레임워크인 PortalGen을 소개하고 있습니다.

- **Technical Details**: PortalGen은 두 단계로 나뉘는 구조를 가지고 있으며, 첫 번째 단계에서는 ICD-9 코드를 기반으로 환자 포털 메시지 프롬프트를 생성합니다. 두 번째 단계에서는 실제 환자 메시지 10개를 사용하여 LLM에 Grounded Generation을 적용하여 현실적인 환자 메시지를 생성합니다. 이 과정에서 HIPAA 준수를 유지하며 최소한의 인간의 비식별화 노력으로 데이터를 생성합니다.

- **Performance Highlights**: PortalGen은 기존의 데이터 생성 기법과 비교해 품질이 높으며, 실제 데이터와 매우 유사한 결과를 보였습니다. 전체 평가를 통해 PortalGen이 데이터의 스타일 및 의미의 충실성을 높이는 데 기여함을 입증했습니다.



### A Next-Generation Approach to Airline Reservations: Integrating Cloud Microservices with AI and Blockchain for Enhanced Operational Performanc (https://arxiv.org/abs/2411.06538)
Comments:
          25 pages, 8 figures

- **What's New**: 이번 연구는 클라우드(Cloud) 마이크로서비스, 분산 인공지능 모듈(distributed artificial intelligence modules), 블록체인 기술을 통합한 차세대 항공 예약 시스템의 개발을 제안합니다. 이는 효율성, 안전성 및 고객 만족도를 향상시키기 위해 설계되었습니다.

- **Technical Details**: 전통적인 예약 시스템은 시스템의 확장성, 데이터의 무결성(integrity) 및 고객에게 제공되는 서비스 수준과 관련된 문제들에 직면하고 있습니다. 제안된 아키텍처는 모듈형(modular) 및 데이터 중심(data-centric) 설계를 통해 이러한 문제를 해결합니다. 이 시스템은 예약, 결제 및 고객 데이터 관리를 별도로 수행할 수 있게 하여 시스템의 가용성을 30% 향상시키고 확장성에서는 40%의 성능 개선을 이끌어냅니다. AI 기반 모듈은 고객의 과거 예약 패턴과 프로필을 분석하여 수요를 예측하고 추천을 생성합니다.

- **Performance Highlights**: 시뮬레이터를 사용한 분석과 머신러닝 평가를 통하여 기존 시스템과 비교했을 때 속도 향상과 데이터 처리 보안률이 35% 증가하고 시스템 반응 시간도 15% 향상되었습니다. 또한, 이러한 시스템은 물류(logistics) 및 환대(hospitality)와 같은 높은 거래 산업에도 적용될 수 있습니다. 블록체인 기술을 활용하여 모든 거래에서 부패를 방지하는 무결성 있는 장부 시스템을 제공함으로써 사기(fraud)를 감소시키고 투명성을 20% 증가시킵니다.



### Probabilistic Consensus through Ensemble Validation: A Framework for LLM Reliability (https://arxiv.org/abs/2411.06535)
Comments:
          8 pages, 6 tables

- **What's New**: 대형 언어 모델(Large Language Models, LLMs)이 텍스트 생성 분야에서 큰 발전을 이루었으나, 의료, 법률, 금융과 같은 고위험 분야에서의 자율 배치에 필요한 신뢰성은 부족합니다. 기존 방법들은 외부 지식이나 인간의 감독에 의존하여 확장성에 한계가 있습니다. 이번 논문에서는 모델 합의를 통한 콘텐츠 검증을 위해 앙상블 방법(ensemble methods)을 재사용하는 새로운 프레임워크를 소개합니다.

- **Technical Details**: 78개의 복잡한 사례에서 사실 정확성(factual accuracy)과 인과 일관성(causal consistency)을 요구하는 테스트에서, 제안된 프레임워크는 두 개 모델을 사용할 경우 정확도를 73.1%에서 93.9%로 향상시켰고(95% CI: 83.5%-97.9%), 세 개 모델을 사용할 경우 95.6%로 향상시켰습니다(95% CI: 85.2%-98.8%). 통계 분석을 통해 강한 모델 간 합의(kappa > 0.76)를 보이며, 오류를 발견하기 위한 충분한 독립성도 유지합니다.

- **Performance Highlights**: 추가적인 검증자(validator)와 정교화를 통해 정확도를 더욱 향상시킬 수 있는 명확한 경로를 제시합니다. 현재 접근법은 다중 선택 형식 요구사항과 처리 지연(processing latency)으로 인해 제약이 있지만, 중요한 응용 분야에서 신뢰할 수 있는 자율 AI 시스템을 가능하게 하는 즉각적인 가치를 제공합니다.



### Does This Summary Answer My Question? Modeling Query-Focused Summary Readers with Rational Speech Acts (https://arxiv.org/abs/2411.06524)
- **What's New**: 본 연구에서는 사용자의 이해를 명확히 고려하여 Query-focused summarization (QFS) 시스템의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: Rational Speech Act (RSA) 프레임워크를 채택하여 독자의 질문 중심 요약에 대한 이해도를 모델링하고, 기존 QFS 시스템의 생성 방법에 통합합니다. 특히, 답변 재구성 목표(answer reconstruction objective)를 도입하여 독자가 요약을 사용하여 초기 질문에 대한 답변을 재구성할 수 있는 능력으로 이해도를 근사화합니다.

- **Performance Highlights**: 이 목표를 사용하여 기존 QFS 시스템이 생성한 후보 요약을 재정렬하고, 각 질문 및 참조 요약과 더 잘 맞는 요약을 선택합니다. 이는 사용자 중심 작업을 위한 언어 생성 시스템의 성능을 단순하고 효과적으로 개선할 수 있는 방법을 제시합니다.



### Barriers to Complexity-Theoretic Proofs that Achieving AGI Using Machine Learning is Intractab (https://arxiv.org/abs/2411.06498)
- **What's New**: 최근 논문(van Rooij et al. 2024)은 데이터로부터 학습하여 인간과 유사한 인공지능을 만드는 것이 계산 복잡도 이론적으로 불가능하다고 주장합니다. 그러나 저자들은 이 주장의 주요 가정을 정당화하지 않았다는 점을 지적하고 있습니다.

- **Technical Details**: 이 논문은 '인간 유사 행동'의 정의와 특정 기계 학습 시스템의 유도 편향(inductive biases)을 고려해야 한다고 주장합니다. 특히, 저자들은 상황-행동 쌍의 분포가 임의적일 수 없다고 강조하며, 이는 그들을 설정한 주장과 모순된다고 합니다.

- **Performance Highlights**: 이 논문은 기존의 'AI-by-Learning' 문제와 'Ingenia Theorem'의 주장에 대해 비판하며, 이론적 증명이 명확하지 않다고 강조합니다. 저자들은 이 증명이 정당화되기 위해 해결해야 할 두 가지 주요 문제를 제시하고 있으며, 이는 인간 유사 행동을 정의하고 특정 구조적 함수가 적절한 유도 편향으로 학습될 수 있음을 보여주는 것입니다.



### Hermes: A Large Language Model Framework on the Journey to Autonomous Networks (https://arxiv.org/abs/2411.06490)
- **What's New**: 이 논문에서는 Hermes라는 새로운 LLM(대형 언어 모델) 에이전트 체인을 소개하여 네트워크 디지털 트윈(NDT) 인스턴스를 구조적이고 설명 가능한 논리적 단계로 생성할 수 있는 "청사진"을 사용하여 네트워크 모델링을 자동화하고 정확성을 향상시키는 방법을 제시합니다.

- **Technical Details**: Hermes는 다양한 사용 사례와 구성에 대해 신뢰할 수 있고 정확한 네트워크 모델링을 가능하게 하는 LLM 에이전트의 체인을 사용합니다. 각 에이전트는 파라미터 기반의 지식을 활용하여 자율적으로 청사진을 설계하고 코드를 생성합니다. 이 과정에서 자기 반성과 피드백 메커니즘이 포함되어 있어, NDT의 유효성을 보장합니다.

- **Performance Highlights**: Hermes를 통해 다양한 네트워크 모델링 작업에 대한 LLM의 신뢰성이 획기적으로 향상되며, 네트워크 역학 및 운영에 대한 보다 강력한 이해를 제공하는 것을 보여줍니다.



### Over-parameterized Student Model via Tensor Decomposition Boosted Knowledge Distillation (https://arxiv.org/abs/2411.06448)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 최근 연구에서는 Knowledge Distillation (KD) 접근 방식에서 학생 모델(student model)의 파라미터를 훈련 중에 확장하여 대형 모델(teacher model)의 지식을 더욱 효과적으로 전달하는 방법에 초점을 맞추고 있습니다. 이 논문은 작은 모델의 성능을 향상시키기 위해 tensor decomposition 기법을 도입하여, 거의 손실 없이 파라미터를 고차원 텐서로 분해하는 새로운 전략을 제공합니다.

- **Technical Details**: 제안된 방법은 Matrix Product Operator (MPO) 기술을 기반으로 하여, 학생 모델의 파라미터 매트릭스를 고차원 텐서로 분해합니다. 그런 다음, 학생 모델과 교사 모델 간의 효과적인 정보 전달을 보장하기 위해 고차원 텐서 정렬 손실(tensor constraint loss)을 설계하였습니다. 이러한 접근 방식은 학생 모델 훈련 동안의 오버 파라미터화를 가능하게 하며, 이는 모델의 일반화 능력(generalization capability)을 향상시킵니다.

- **Performance Highlights**: 다양한 실험을 통해 OPDF가 여러 Knowledge Distillation 작업에서 실질적인 성능 향상을 이룬 것으로 나타났습니다. 예를 들어, BERT-base 모델에서 평균적으로 +1.6 향상되었으며, AD-KD와 우리의 방법 비교에서는 두 모델 간의 성능 차이를 거의 없애면서도 고성능을 달성했습니다.



### Reinforcement learning for Quantum Tiq-Taq-To (https://arxiv.org/abs/2411.06429)
- **What's New**: 양자 Tiq-Taq-Toe는 양자 컴퓨팅과 강화 학습(Reinforcement Learning, RL)의 통합을 위한 접근 가능한 테스트베드로 연구됩니다. 이전에는 이 게임에 RL 방법이 적용된 적이 없으며, 본 연구를 통해 새로운 게임 규칙을 도입하였습니다.

- **Technical Details**: 게임의 상태는 Measurement와 Move History를 통해 관찰됩니다. 양자 Tiq-Taq-Toe는 두 가지 버전으로 나뉘며, 하나는 엔탱글먼트(Entanglement) 이동을 제한하고, 다른 하나는 제한을 완전히 제거하여 다양한 양자 상태와 상호작용이 가능하게 합니다.

- **Performance Highlights**: 자체 플레이를 통해 비교한 PPO(P Proximal Policy Optimization) 에이전트의 결과, 측정 행렬과 역사적 엔탱글먼트 기록을 모두 사용하는 경우(M&H 에이전트) 최적의 성능을 발휘하였으며, 이는 강하게 부분적으로 관찰 가능한 양자 환경에서 종합 정보의 중요성을 강조합니다.



### Mastering NIM and Impartial Games with Weak Neural Networks: An AlphaZero-inspired Multi-Frame Approach (https://arxiv.org/abs/2411.06403)
- **What's New**: 이번 연구는 AlphaZero 스타일의 강화 학습 알고리즘이 NIM 게임에서 최적 플레이를 배우는 데 어려움을 겪는다는 기존 실험 결과를 검증하고 설명하는 이론적 프레임워크를 제공합니다. 특히, 최근 게임 이력을 통합함으로써 한정된 AlphaZero 모델이 NIM에서 최적 플레이를 달성할 수 있음을 증명합니다.

- **Technical Details**: 본 논문에서는 AC0 복잡도 클래스에 속하는 약한 Neural Network 모델(NN, RNN, LTST)을 정의하며, 이 모델들이 다루는 문제를 해결하기 위해 멀티 프레임 표현과 새로운 검색 전략을 도입합니다. 이러한 모델들은 심층 신경망의 계산 한계를 극복할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 강화 학습 알고리즘은 특정 환경에서 NIM 초보자 게임을 배울 때 성능이 저하되는 경향을 보입니다. 그러나 제안하는 방법은 이러한 제약을 극복하며 최적 플레이를 가능하게 합니다.



### Class Granularity: How richly does your knowledge graph represent the real world? (https://arxiv.org/abs/2411.06385)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 Class Granularity라는 새로운 메트릭을 제안하여 지식 그래프의 구조화 수준을 정량적으로 평가합니다. 이는 다양한 특성을 가진 클래스의 정의와 그로 인한 영향을 측정하기 위한 것입니다.

- **Technical Details**: Class Granularity는 지식 그래프의 고유한 특성을 가진 클래스들이 얼마나 세밀하게 정의되어 있는지를 측정합니다. 이 연구는 그래프 임베딩(graph embedding) 기술과 질문 응답 시스템(KBQA)에 대한 Class Granularity의 영향을 분석합니다.

- **Performance Highlights**: Class Granularity 메트릭을 활용하여 YAGO, DBpedia, Wikidata와 같은 다양한 Linked Open Data 소스의 구조적 수준을 비교한 결과는 이전에 보고된 적이 없습니다. 이로 인해 지식 그래프의 품질과 효과를 더욱 명확히 이해할 수 있는 기초 자료를 제공합니다.



### A Comprehensive Survey and Guide to Multimodal Large Language Models in Vision-Language Tasks (https://arxiv.org/abs/2411.06284)
- **What's New**: 이 논문은 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 급속히 발전하는 분야를 조사합니다. MLLMs는 텍스트, 이미지, 비디오, 오디오 등 다양한 데이터 타입을 통합하여 복잡한 AI 시스템을 구축하는 데 중요한 역할을 하고 있습니다.

- **Technical Details**: MLLMs는 cross-modal learning을 통해 텍스트, 시각, 청각 등 다양한 데이터를 기반으로 훈련됩니다. 이들은 통합 표현(unified representation) 방식을 사용하여 서로 다른 모달리티를 원활하게 처리하며, 안전하면서도 효과적인 AI 개발을 위한 윤리적 고려사항을 강조합니다.

- **Performance Highlights**: MLLMs는 비주얼 질문 응답, 텍스트-이미지 생성, 멀티모달 콘텐츠 생성 등의 뛰어난 성능을 자랑하며, 의료, 보안, 전자상거래 등 다양한 분야에 실질적인 응용 가능성을 보입니다. 또한 인간-로봇 상호작용 및 자율주행차의 안전성 향상에도 기여할 수 있는 잠재력을 가지고 있습니다.



### AI's Spatial Intelligence: Evaluating AI's Understanding of Spatial Transformations in PSVT:R and Augmented Reality (https://arxiv.org/abs/2411.06269)
- **What's New**: 이 논문에서는 GPT-4 모델을 활용하여 3D 공간에서 객체의 회전 이해 능력을 연구했습니다. 특히, Generative AI의 이미지 및 언어 처리 기능을 통해 공간 회전 과정을 어떻게 이해하는지를 분석했습니다.

- **Technical Details**: 연구는 Revised Purdue Spatial Visualization Test: Visualization of Rotations (Revised PSVT:R)를 기반으로 하여 수행되었으며, 이와 더불어 좌표계 축을 추가하여 GPT-4의 성능 변화를 연구했습니다. 또한, Augmented Reality (AR) 장면에서 3D 회전 이해를 분석하여 보조 텍스트 정보를 추가할 때 GPT-4의 회전 이해도 향상됨을 관찰했습니다.

- **Performance Highlights**: GPT-4 모델은 공간 회전 과정을 이해하는 데 한계가 있지만, AR을 통한 추가 정보 제공으로 회전 과정 이해 잠재력을 보여주었습니다. 이 연구는 AI의 공간 지능과 AR의 상호작용 시각화 기능을 결합하여 학생들의 공간 학습 활동을 향상시킬 수 있는 가능성을 제시합니다.



### GuidelineGuard: An Agentic Framework for Medical Note Evaluation with Guideline Adherenc (https://arxiv.org/abs/2411.06264)
- **What's New**: 이 논문에서는 GuidelineGuard라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 기반으로 하여 의료 진료 노트를 자율적으로 분석하여 가이드라인 준수를 보장합니다.

- **Technical Details**: GuidelineGuard는 병원 퇴원 노트 및 사무실 방문 노트와 같은 의료 진료 노트를 분석합니다. 추천된 관행에서의 이탈을 식별하고, WHO 및 CDC와 같은 기관의 최신 기준을 준수하도록 증거 기반 제안을 제공합니다.

- **Performance Highlights**: 이 프레임워크는 문서 품질을 개선하고 임상 오류를 줄이는 데 도움을 주며, 의료 전문가들이 최신 가이드라인을 따르는 데 필요한 지원을 제공합니다.



### Knowledge Authoring with Factual English, Rules, and Actions (https://arxiv.org/abs/2411.06253)
Comments:
          PhD thesis

- **What's New**: 본 논문에서는 Knowledge Representation and Reasoning (KRR) 시스템을 개선하기 위해, 기존 KALM을 사실 언어에 적합하도록 확장한 KALMF와 규칙 및 행동을 지원하는 KALMR를 제안합니다.

- **Technical Details**: KALMF는 MS라는 신경 파서를 사용하여 문장들을 분석하며, 사용자는 최소한의 문법 연수를 통해 지식을 표현할 수 있습니다. KALMR은 F-logic을 사용하여 다단계 프레임 기반 추론을 가능하게 하며, Simplified Event Calculus (SEC)를 사용하여 행동을 표현하고 추론합니다.

- **Performance Highlights**: KALMF와 KALMR은 사실 및 쿼리 작성에서 95%, 규칙 작성에서 100%, 행동 관련 저작 및 추론에서 99.3%의 정확성을 달성하였습니다. 또한, 속도 최적화 과정에서 68%의 런타임 개선과 함께 높은 정확도를 나타냈습니다.



### Quasi-random Multi-Sample Inference for Large Language Models (https://arxiv.org/abs/2411.06251)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)에서 사용할 수 있는 새로운 샘플링 기술인 arithmetic sampling을 도입하고, 이를 기존의 ancestral sampling과 비교하여 두 가지 디코딩 작업에서의 성능과 다양성을 향상시키는 가능성을 탐구합니다. 특히, reasoning과 machine translation 작업에서 샘플의 다양성을 통해 성능 향상을 확인했습니다.

- **Technical Details**: arithmetic sampling은 기존의 ancestral sampling 과정을 단순화하여, 유니폼하게 분포된 코드 포인트(color points)에서 샘플을 생성하는 방식으로, 각 코드 포인트는 출력 분포의 시퀀스에 해당합니다. 이는 생성된 샘플의 다양성을 보장하며, 병렬 처리가 용이하여 효율적인 inference를 가능하게 합니다. 실험에서는 GSM8K 및 COMET 점수를 사용하여 평가했습니다.

- **Performance Highlights**: arithmetic sampling을 사용한 경우, GSM8K 데이터셋에서 정확도가 3-5% 상승하며, WMT19 작업에서도 0.45-0.89% 포인트 증가했습니다. 특히, 샘플 수가 증가함에 따라 reasoning과 translation의 성능이 유의미하게 향상되었습니다.



### Multimodal Contrastive Learning of Urban Space Representations from POI Data (https://arxiv.org/abs/2411.06229)
Comments:
          19 pages, 5 figures, 7 tables

- **What's New**: 본 논문에서는 CaLLiPer(Contrastive Language-Location Pre-training)라는 새로운 방법론을 제안하여 POI(Point-of-Interest) 데이터로부터 도시 공간 표현을 학습하는 한계를 극복합니다. CaLLiPer는 지리적 구분, 공간 정보 모델링 부족, POI의 의미적 속성 미활용, computational inefficiencies(계산 비효율성) 문제를 해결합니다.

- **Technical Details**: CaLLiPer는 연속적인 도시 공간을 벡터 표현으로 직접 임베드하여 도시 환경의 공간적 및 의미적 분포를 포착합니다. 이 모델은 multimodal contrastive learning objective를 활용하여 위치 임베딩을 텍스트 POI 설명과 정렬합니다. 이러한 접근 방식은 복잡한 학습 코퍼스 구축 및 negative sampling(부정 샘플링)의 필요성을 우회합니다. 또한, 본 모델은 영국 런던에서 도시 공간 표현 학습을 통해 5-15% 예측 성능 향상을 확인하였습니다.

- **Performance Highlights**: CaLLiPer는 기존의 최신 기술에 비해 예측 정확도에서 5-15%의 성능 향상을 보였으며, 학습 시간 또한 단축되어 효율성과 확장성을 보여주었습니다. 시각화된 결과는 본 모델이 도시 의미의 공간 변화를 높은 정확도와 세밀한 해상도로 포착하는 데 있어 장점을 가지고 있음을 보여줍니다.



### Artificial Intelligence for Collective Intelligence: A National-Scale Research Strategy (https://arxiv.org/abs/2411.06211)
Comments:
          25 pages, 3 figures, Accepted for publication at Knowledge Engineering Review (KER)

- **What's New**: 본 논문은 영국의 새로운 인공지능 연구 허브인 AI for Collective Intelligence (AI4CI)에 대한 연구 전략을 소개합니다. AI4CI는 AI와 집단 지능의 교차점에서 응용 연구를 수행하며, 이를 통해 국면별 사회적 도전 과제를 해결하고자 합니다.

- **Technical Details**: AI4CI의 연구는 여러 분야(헬스케어, 금융, 환경, 팬데믹, 도시)를 통해 리얼타임 데이터 스트림의 접근 및 해석을 목표로 하며, 스마트 AI 에이전트를 통한 효과적인 개입을 추구합니다. 또한, AI의 두 가지 중심 요소인 정보 수집과 행동 통보를 연결하는 방법론이 강조됩니다.

- **Performance Highlights**: AI4CI는 다국적 대학 및 정부, 산업 협력 파트너와 함께, 인공지능과 집단 지능의 통합을 통해 세계적인 문제를 해결하는 데 기여할 수 있는 연구 전략을 개발하고 있습니다.



### OpenAI-o1 AB Testing: Does the o1 model really do good reasoning in math problem solving? (https://arxiv.org/abs/2411.06198)
- **What's New**: OpenAI의 Orion-1 모델은 이전 대형 언어 모델보다 더 강력한 논리적 사고 능력을 가지고 있다고 주장되고 있으며, 최근 연구를 통해 이 모델이 문제를 '기억'하는 경향이 없음을 밝혀냈다.

- **Technical Details**: Orion-1 모델은 강화 학습을 통해 토큰별 보상 모델을 사용하는 혁신적인 훈련 방식으로 훈련되었으며, 두 개의 데이터셋(IMO 및 CNT)을 사용하여 성능을 비교하였다. 제시된 문제와 해결 방안을 비교 분석하여, 모델이 문제를 단순히 암기하는 것이 아니라 높은 품질의 추론 단계를 생성할 수 있음을 보여주었다.

- **Performance Highlights**: 오리온 모델은 IMO 및 CNT 문제 세트 사이에서 일관된 성능을 보였고, 다양한 수학 문제에서의 성능을 평가하여 LLM의 일반화 능력을 높인 것으로 나타났다.



### Generalizing Hyperedge Expansion for Hyper-relational Knowledge Graph Modeling (https://arxiv.org/abs/2411.06191)
- **What's New**: 본 논문은 기존의 지식 그래프(KG)와 비교하여 하이퍼 관계 지식 그래프(HKG)의 모델링을 위한 TransEQ 메커니즘을 제안합니다. 이는 하이퍼엣지 확장을 일반화하여 HKG를 KG로 변환하는 동등한 변환을 통해 구조적 정보와 의미적 정보를 동시에 포착하는 혁신적인 접근 방식입니다.

- **Technical Details**: TransEQ는 하이퍼 그래프에서 그래프로 변환하는 개념인 하이퍼엣지 확장을 기반으로 하며, 인코더-디코더 프레임워크를 통해 HKG 모델링을 수행합니다. 인코더 부분에서는 그래프 신경망(GNN)을 사용하여 구조적 모델링을 수행하고, 디코더에서는 HKG 기반의 점수 함수(SF)를 활용하여 의미적 모델링을 진행합니다. 또한, 공유 임베딩 메커니즘을 설계하여 의미적 관련성을 캡처합니다.

- **Performance Highlights**: 실험 결과, TransEQ는 WikiPeople과 같은 대규모 벤치마크에서 MRR(Mean Reciprocal Rank)을 15% 향상시키며, 기존의 최첨단 모델에 비해 우수한 성능을 보여줍니다. 이러한 결과는 TransEQ가 효과적인 정보 포착 및 효율성을 제공함을 나타냅니다.



### Deep Reinforcement Learning for Digital Twin-Oriented Complex Networked Systems (https://arxiv.org/abs/2411.06148)
- **What's New**: 이 연구는 디지털 트윈 지향 복합 네트워크 시스템(Temporal DT-CNS) 모델을 제안하고, 강화 학습을 통해 전염병 발생 시의 노드 간의 시간적 상호작용을 결정하는 방식을 탐구합니다.

- **Technical Details**: Temporal DT-CNS 모델은 노드의 이질적 특성과 연결 선호도를 고려하며, 이는 선호적 부착(prefential attachment)과 동류성(homophily)의 영향을 결합합니다. 또한, SIR 모델을 사용하여 전염병 확산 과정을 설명합니다.

- **Performance Highlights**: 실험 결과는 완전 협력이 자아 중심 또는 무지한 '프리 라이더'(free-riders)가 있는 협력보다 더 높은 보상과 낮은 감염 수를 초래함을 보여주었습니다. 또한, '프리 라이더' 수의 증가가 보상을 줄이고 감염 수를 증가시킵니다.



### AI-Compass: A Comprehensive and Effective Multi-module Testing Tool for AI Systems (https://arxiv.org/abs/2411.06146)
- **What's New**: 본 연구에서는 AI 시스템의 평가를 위한 종합적인 테스트 도구인 	ool을 설계하고 구현하였습니다. 이 도구는 적대적 강인성(adversarial robustness), 모델 설명 가능성(model interpretability), 뉴런 분석(neuron analysis) 등을 포함한 다양한 측면을 평가할 수 있도록 돕습니다.

- **Technical Details**: 	ool은 여러 측정 지표를 통해 AI 시스템의 성능과 신뢰성을 종합적으로 평가합니다. 기존의 DLS(DL Systems) 테스트 도구들이 특정 과제에 국한된 것과 달리, 	ool은 다양한 모드(예: 이미지 분류, 객체 탐지, 텍스트 분류)에서 유효성을 검증할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 	ool은 기존의 DLS 테스트 도구에 비해 적대적 공격에 대한 모델 강인성을 정밀하게 평가하고, 모델의 설명 가능성에 대한 신뢰성 있는 보고서를 제공합니다. 이를 통해 AI 시스템의 종합적인 평가가 가능함을 입증하였습니다.



### Evaluating the Propensity of Generative AI for Producing Disinformation During an Election Cyc (https://arxiv.org/abs/2411.06120)
- **What's New**: 이 연구는 현재의 생성적 인공지능 모델들이 선거 과정에서 유해한 허위 정보를 생산할 가능성을 조사합니다. 연구 결과, Copilot과 Gemini는 가장 낮은 예상 위험을 기록하며 가장 안전한 성능을 보인 반면, GPT-4o는 유해한 허위 정보를 가장 많이 생성하여 높은 위험 점수를 기록했습니다.

- **Technical Details**: 세 가지 인기 있는 생성적 인공지능 모델(ChatGPT, Gemini, Copilot)을 사용하여 각 모델이 정치 및 건강 주제에 대한 허위 정보를 생성하는 경향을 평가했습니다. 사실 확인된 정보를 기반으로 허위 정보를 식별하고, 다양한 적대적 역할을 사용하여 허위 정보의 생산을 유도했습니다.

- **Performance Highlights**: Gemini 모델은 정치 분야의 허위 정보에서 가장 안전한 성능을 보였으며, Copilot은 건강 관련 주제에서 가장 안전했습니다. 다양한 역할을 통해 GPT-4o는 높은 허위 정보 생성률을 보였고, 분석된 모든 모델에서 일반적으로 더 큰 예상 위험을 초래하는 적대적 역할 특성이 발견되었습니다.



### A Multimodal Adaptive Graph-based Intelligent Classification Model for Fake News (https://arxiv.org/abs/2411.06097)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 기하학적 심층학습(geometric deep learning)을 활용한 그래프 기반(fake news) 허위 뉴스 탐지 시스템인 MAGIC를 소개합니다. 기존 연구와의 차별성은 멀티모달(멀티모달리티) 접근 방식을 통해 다양한 정보를 통합하는 것에 있습니다.

- **Technical Details**: MAGIC는 텍스트 벡터화(text vectorization)를 위해 Transformers의 Encoder Representations를 사용하고, 이미지를 위해 ResNet50을 활용합니다. 또한 adaptive Graph Attention Network를 통해 포괄적인 정보 상호작용(graph interaction) 그래프를 구축하고, Softmax 함수를 통해 멀티모달 입력을 분류합니다.

- **Performance Highlights**: MAGIC 모델은 Fakeddit(영어)와 Multimodal Fake News Detection(중국어) 두 개의 허위 뉴스 데이터 세트에서 각각 98.8%와 86.3%의 정확도를 달성하였습니다. 아블레이션 실험(ablation experiments) 결과, 두 데이터 세트 모두에서 뛰어난 성능을 보였으며, 그래프 기반 심층학습(adaptive graph-based deep learning) 모델이 멀티모달 허위 뉴스 탐지에서 효과적임을 입증하였습니다.



### Cross-Domain Transfer Learning using Attention Latent Features for Multi-Agent Trajectory Prediction (https://arxiv.org/abs/2411.06087)
Comments:
          Accepted at the IEEE International Conference on Systems, Man, and Cybernetics (IEECSMC) 2024

- **What's New**: 본 논문에서는 교통 네트워크 간의 일반화 능력을 향상시키기 위한 새로운 공간-시간(Spatial-Temporal) 궤적 예측 프레임워크를 제안합니다. 특히, Transformer 기반의 모델에서 주의(attention) 표현을 통한 도메인 간 적응(cross-domain adaptation)을 수행하여 복잡한 궤적 데이터를 모델링합니다.

- **Technical Details**: 제안하는 Graph Embedded Transformer는 Graph Convolutional Network(GCN)를 통합하여 공간 인식을 위한 비유클리드(input embeddings)를 생성하고, Transformer를 통해 궤적 시퀀스의 시간적 모델링을 수행합니다. 또한, Transformer의 인코더 모듈에 도메인 적응 훈련 전략을 추가하여 다양한 교통 도메인 간의 횡단 학습(cross-domain transfer learning)을 실현합니다.

- **Performance Highlights**: NGSIM-I80 및 NGSIM-US101 데이터셋을 통한 실험 결과, 제안한 모델은 기존 최첨단(vehicle trajectory prediction) 모델들에 비해 우수한 궤적 예측 정확도를 보였으며, 이는 도메인 일반화 능력이 효과적임을 나타냅니다.



### Diversity and Inclusion in AI for Recruitment: Lessons from Industry Workshop (https://arxiv.org/abs/2411.06066)
- **What's New**: 이 연구는 인공지능 (AI)을 사용한 온라인 채용 시스템에서 다양성 및 포용성 (D&I) 원칙을 실제로 어떻게 적용할 수 있는지에 대한 실용적인 조사입니다. 특히, 이러한 원칙이 채용 프로세스를 보다 포괄적으로 만들기 위해 어떻게 구현될 수 있는지를 탐구합니다.

- **Technical Details**: 이 연구는 대규모 다국적 채용 회사와 함께 진행된 공동 설계 워크숍을 통해 AI 구동 채용 사례 두 가지에 중점을 두었습니다. 사용자 스토리(user stories)와 페르소나(personas)를 활용하여 AI가 다양한 이해관계자에게 미치는 영향을 평가했습니다. 후속 인터뷰를 통해 워크숍 참가자들의 D&I 원칙에 대한 인식 변화와 적용 효과를 평가하였습니다.

- **Performance Highlights**: 워크숍은 참가자들의 AI 내 D&I에 대한 이해를 크게 향상시켰습니다. 그러나 D&I 의식을 운영적 실천으로 전환하는 것에는 어려움이 있었으며, 비즈니스 목표와 D&I를 균형 있게 유지하는 것이 특히 도전이었습니다. 따라서 맞춤형 D&I 가이드라인 개발과 지속적인 지원이 필요한 것으로 나타났습니다.



### Personalized News Recommendation System via LLM Embedding and Co-Occurrence Patterns (https://arxiv.org/abs/2411.06046)
- **What's New**: 최근 2년 동안 대형 언어 모델(LLMs)이 급속한 발전을 이루었으며, 이로 인해 추천 시스템 분야에서도 혁신적인 변화를 가져왔습니다. 특히 뉴스 추천 시스템에서 LLM을 활용하여 사용자 선호를 정교하게 파악할 수 있는 새로운 알고리즘이 제안되었습니다.

- **Technical Details**: 본 논문에서는 LLM Embedding 및 Co-Occurrence Pattern (LECOP) 기반으로 뉴스 모델을 재구성하는 새로운 뉴스 추천(NR) 알고리즘을 제안합니다. 대규모 데이터셋을 활용한 contrastive learning으로 LLM을 fine-tuning 하여 뉴스의 의미 정보를 최대로 활용할 수 있도록 했습니다. 또한, 뉴스 ID co-occurrence, Item-Item keywords co-occurrence, Intra-Item keywords co-occurrence와 같은 다양한 공동 발생 패턴을 탐구하였습니다. 이러한 패턴은 모두 LLM에 의해 생성됩니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 새로운 방법이 기존 모델들에 비해 뛰어난 성능을 보임을 입증하였습니다.



### CROPS: A Deployable Crop Management System Over All Possible State Availabilities (https://arxiv.org/abs/2411.06034)
- **What's New**: 본 논문은 농업 관리의 최적화 문제를 해결하기 위해 deployable Crop Management system Over all Possible State availabilities (CROPS)를 도입했습니다. CROPS는 언어 모델(language model, LM)을 강화 학습(reinforcement learning, RL) 에이전트로 사용하여 농작물 시뮬레이션인 DSSAT에서 최적의 관리 전략을 탐색합니다.

- **Technical Details**: CROPS는 결정 지원 시스템(DSSAT)을 통해 전체 또는 부분 관찰을 기반으로 농업 결정을 내릴 수 있도록 설계되었습니다. 주요 목표는 관리 정책 최적화와 함께 마스킹된 상태 추론입니다. 마스킹 기법을 사용하여 실제 농업 환경의 불확실성을 반영했으며, 이는 RL 에이전트의 강인성과 적응성을 크게 향상시킵니다.

- **Performance Highlights**: CROPS는 미국 플로리다와 스페인 사라고사에서의 옥수수 작물 실험에서 State-of-the-Art(SOTA) 결과를 달성했으며, 생산성, 수익, 지속 가능성 등의 다양한 평가 지표에서도 우수한 성과를 기록했습니다. 또한, 1천만 개 이상의 실제 상황에 즉시 배포 가능하며, 사전 훈련된 정책은 소음 저항 능력을 가지고 있어 잠재적인 센서 편향을 최소화할 수 있습니다.



### A Comprehensive Guide to Enhancing Antibiotic Discovery Using Machine Learning Derived Bio-computation (https://arxiv.org/abs/2411.06009)
Comments:
          65 pages

- **What's New**: 전통적인 약물 발견 과정이 인공지능(AI) 및 머신러닝(ML)의 발전에 의해 변화하고 있으며, 약물 발견 과정을 간소화하고 가속화할 수 있는 다양한 도구들이 소개됩니다.

- **Technical Details**: ML 알고리즘을 훈련하기 위해 데이터 세트를 사용함으로써 비교적 빠르고 효율적으로 약물 또는 약물 유사 화합물을 발견할 수 있습니다. 이 외에도 AI 기반 약물 발견 및 개발의 한계, 고품질 데이터 부족, 윤리적 고려 사항 등을 다룹니다.

- **Performance Highlights**: AI의 약물 발견에 대한 영향력 증대가 강조되며, 특히 세계적 항균 저항성(AMR) 문제를 해결하기 위해 새로운 항생제 발견을 가속화할 수 있는 방법도 논의됩니다.



### Game-theoretic LLM: Agent Workflow for Negotiation Games (https://arxiv.org/abs/2411.05990)
Comments:
          44 pages, 12 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 전략적 결정 과정에서 이성을 발휘하는 수준을 게임 이론의 틀 안에서 조사합니다. 특히, 불완전 정보 게임 상황에서 LLMs의 비이성적인 전략 선택 경향을 분석하고 이를 극복하기 위한 게임 이론적 워크플로우를 개발하였습니다.

- **Technical Details**: 여러 최신 LLM들(예: Claude-3.5 Sonnet, GPT-4o 등)의 성능을 평가하였으며, 이는 완전 정보 및 불완전 정보 게임(예: Prisoner’s Dilemma, Battle of the Sexes 등)을 기반으로 합니다. 또한, 지배적 전략 탐색(Dominant Strategy Search), 후방 유도(Backward Induction), 베이지안 신념 업데이트(Bayesian belief updating)와 같은 게임 이론 원리를 활용하여 LLM의 이성적 행동과 결정 능력을 향상시키는 알고리즘을 설계하였습니다.

- **Performance Highlights**: 제안된 워크플로우를 통해 LLM들이 최적 전략을 식별하는 데 유의미한 개선을 보였으며, 협상 시 near-optimal allocations를 달성하고, 협상 과정에서의 착취(Exploitation)에 대한 저항력도 증가하였습니다. 연구 결과는 복잡한 상호작용 환경에서 전략적으로 더 견고하고 합리적인 AI 에이전트를 개발하는 데 기여할 수 있습니다.



### Quantifying artificial intelligence through algebraic generalization (https://arxiv.org/abs/2411.05943)
- **What's New**: 이번 논문에서는 알gebraic circuit complexity(대수 회로 복잡도) 이론을 도입해 기호적 일반화(symbolic generalization)를 명확히 정량화하는 새로운 프레임워크를 제시합니다. 현재의 인공지능(AI) 시스템은 기호 처리 및 추상화에 있어 한계를 보이고 있으며, 이 문제를 해결하기 위한 방법론을 수립하는 것에 초점을 두고 있습니다.

- **Technical Details**: 기호적 계산(symbolic computation)의 복잡성을 연구하기 위해 대수 회로 복잡도 이론을 사용합니다. 이 이론은 수학적 표현을 회로 모델(즉, 방향성 비순환 그래프)로 정식화하며, 각 회로의 크기(size)와 깊이(depth)를 주요 복잡성 척도로 설정합니다. 이를 통해 AI 모델의 일반화 성능을 객관적으로 평가할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 대수 회로 복잡도 이론을 통해 다양한 기호 문제를 다룰 수 있으며, 이는 AI 시스템의 약점 및 실패 양상을 구체적으로 파악하는 데 유리합니다. 알gebraic circuits는 그 본질상 수많은 표본(sample)을 생성할 수 있어 오늘날 데이터 중심의 기계 학습 알고리즘을 위한 최적의 테스트베드(testbed)가 됩니다.



### Qwen2.5-32B: Leveraging Self-Consistent Tool-Integrated Reasoning for Bengali Mathematical Olympiad Problem Solving (https://arxiv.org/abs/2411.05934)
- **What's New**: 이번 연구에서는 2024년 DL Sprint 3.0 BUET CSE Fest 대회를 위해 개발된 벵골어 수학 문제 해결을 위한 혁신적인 접근법을 제시합니다. 이 방법은 Qwen 2.5 시리즈와 같은 고급 딥러닝 모델을 활용하며, prompt engineering, 모델 양자화(model quantization), 도구 통합 추론(Tool Integrated Reasoning, TIR)을 통해 복잡한 계산 문제를 처리하도록 개선되었습니다.

- **Technical Details**: 이 연구에서는 Mistral, Qwen 시리즈와 같은 다양한 모델 아키텍처를 탐색하며, Retrieval-Augmented Generation(RAG), 사용자 정의 데이터셋(curated dataset) 기법을 활용하여 이를 정제했습니다. 수동으로 하이퍼파라미터 조정(hyperparameter tuning)을 통해 온도(temperature) 및 top-p 파라미터를 최적화하여 모델의 적응성과 정확성을 높였습니다. 또한, VLLM(Variable-Length Language Model)을 도입하여 추론 속도를 개선하고, Python을 사용한 TIR을 통해 복잡한 계산을 수행하도록 하였습니다.

- **Performance Highlights**: 최종 모델은 77점을 기록하며 다양한 문제 유형을 처리하는 능력을 입증했습니다. 데이터셋의 번역을 통해 모델의 성능이 크게 향상되어, 영어에 대한 사전 지식을 효과적으로 활용하여 점수를 끌어올렸습니다. Tool Integrated Reasoning(TIR)의 도입은 특히 벵골어 질문에서 모델의 효율성과 정확도를 크게 개선했습니다.



### LEGO-GraphRAG: Modularizing Graph-based Retrieval-Augmented Generation for Design Space Exploration (https://arxiv.org/abs/2411.05844)
- **What's New**: GraphRAG은 Retrieval-Augmented Generation (RAG)에서의 도전을 해결하기 위해 지식을 내장한 그래프를 활용하고 있으며, LEGO-GraphRAG라는 모듈형 프레임워크를 제안합니다. 이 프레임워크는 그래프 기반 지식 검색 프로세스를 세 가지 모듈로 분해합니다: 서브그래프 추출(subgraph-extraction), 경로 필터링(path-filtering), 경로 정제(path-refinement).

- **Technical Details**: LEGO-GraphRAG는 그래프 구조를 기반으로 한 알고리즘과 신경망 모델(Neural Network, NN)을 체계적으로 요약하고 분류하였습니다. 또한, Graph Coupling과 Computational Cost와 같은 주요 설계 요소를 식별하여 GraphRAG 구현의 효과성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 광범위한 실험 연구를 통해 고품질 GraphRAG 인스턴스를 구축하고 그 결과가 검색 및 추론 성능에 미치는 영향을 분석하였습니다. 연구 결과는 보다 정확하고 맥락적으로 관련성이 높은 LLM 애플리케이션을 위한 GraphRAG 인스턴스 설계 최적화에 중요한 통찰력을 제공합니다.



### To Ask or Not to Ask? Detecting Absence of Information in Vision and Language Navigation (https://arxiv.org/abs/2411.05831)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 연구는 로봇이 불완전한 지침에 대해 명확한 질문을 할 수 있는 능력을 개발하는 것을 다룹니다. 특히 미흡한 정보를 인식하는 능력에 초점을 맞추고 있으며, 그로 인해 에이전트의 효율성을 향상시키는 방법을 제시합니다.

- **Technical Details**: 본 논문에서는 attention 기반의 instruction-vagueness estimation 모듈을 제안합니다. 이 모듈은 지침, 지금까지의 경로, 다음 이동 제안을 입력으로 받아, 각 시점에서 VLN 모델의 제안에 따라 행동할지 또는 도움을 요청할지를 결정합니다. 모듈은 instruction-to-path alignment 정보를 활용하여, precision-recall 균형에서 약 52% 향상된 성능을 보입니다.

- **Performance Highlights**: 제안된 IV estimation 모듈을 VLN-DUET 및 HAMT 두 내비게이터에 통합하여 효과와 일반화를 평가하였습니다. 실험 결과, instruction-to-path attention network의 attention 점수가 모호성 추정에 대한 더 나은 지표로 작용함을 보여주었습니다.



### AI Multi-Agent Interoperability Extension for Managing Multiparty Conversations (https://arxiv.org/abs/2411.05828)
Comments:
          20 pages, 3 figures

- **What's New**: 이 논문은 Open Voice Interoperability Initiative(OVON)의 기존 다중 에이전트 상호운용성(Multi-Agent Interoperability) 사양에 대한 새로운 확장을 소개합니다. 이 확장은 다양한 기술로 개발된 AI 에이전트가 보편적이고 자연어 기반의 API 또는 NLP 기반 표준 API를 통해 소통할 수 있도록 합니다.

- **Technical Details**: 다중 AI 대화를 관리하는 데 초점을 맞추고 새로운 개념(예: Convener Agent, Floor-Shared Conversational Space, Floor Manager, Multi-Conversant Support, Interruptions 및 Uninvited Agents 처리 메커니즘)을 도입합니다. Convener는 메시지 릴레이 및 참가자 상호작용의 제어자로서 역할을 수행하여 확장성과 보안을 향상시킵니다.

- **Performance Highlights**: 이 연구는 여러 AI 에이전트가 협력, 토론 또는 기여해야 하는 시나리오에서 원활하고 효율적이며 안전한 상호작용을 보장하기 위한 중요성을 강조합니다. 추가적으로 대화의 수용 구조 내에서의 구현 사례를 제공하며, 기존의 독립적인 에이전트들이 어떻게 협력할 수 있는지를 보여줍니다.



### UTMath: Math Evaluation with Unit Test via Reasoning-to-Coding Thoughts (https://arxiv.org/abs/2411.07240)
- **What's New**: 이 논문은 기존의 수학 문제 해결 기준의 한계를 극복하기 위한 새로운 UTMath 벤치마크를 소개합니다. 이 벤치마크는 9개의 수학 영역에서 1,053개의 문제를 가지고 있으며, 정확성과 신뢰성을 중시하는 혁신적인 평가 프레임워크를 제공합니다.

- **Technical Details**: UTMath 벤치마크는 각각의 문제에 대해 68개 이상의 테스트 케이스로 구성되며, LLM이 코드를 생성하기 전에 명시적 추론을 수행하도록 유도하는 Reasoning-to-Coding of Thoughts (RCoT) 접근 방식을 채택하고 있습니다. 이는 LLM이 더 향상된 솔루션을 생성하도록 합니다.

- **Performance Highlights**: 연구 결과, GPT-4o 모델은 UTMath 벤치마크에서 단 26.93%의 문제만을 해결하여 벤치마크의 어려움을 보여주었습니다. RCoT 접근을 통해 8개의 LLM 중 7개가 더 효율적인 솔루션을 생성했으며, 모델들의 reasoning 품질은 최종 솔루션의 정확성과 효율성에 중요한 영향을 미쳤습니다.



### Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models (https://arxiv.org/abs/2411.07232)
Comments:
          Project page is at this https URL

- **What's New**: 이번 연구에서는 자연어 텍스트 지침에 따라 이미지를 편집하는 기존의 접근 방식의 한계를 극복하기 위해, Add-it이라는 새로운 방법을 제안합니다. 이 방법은 기존 장면을 유지하면서도 새로운 객체를 자연스럽게 통합하는 것에 중점을 둡니다.

- **Technical Details**: Add-it은 학습이 필요 없는 접근 방식을 통해 diffusion 모델의 attention 메커니즘을 확장하고, 장면 이미지, 텍스트 프롬프트 및 생성된 이미지를 포함한 세 가지 주요 정보를 결합합니다. 이 확장된 attention 메커니즘은 구조적 일관성을 유지하며 객체의 자연스러운 배치를 보장합니다.

- **Performance Highlights**: Add-it은 이전 방법들과 비교해 객체 삽입 작업에서 최첨단의 성능을 달성했습니다. 실제 이미지 및 생성된 이미지 삽입 벤치마크에서 뛰어난 결과를 보였으며, 인간 평가에서도 80% 이상의 선호도를 기록했습니다.



### Grounding Video Models to Actions through Goal Conditioned Exploration (https://arxiv.org/abs/2411.07223)
Comments:
          Project page at this https URL

- **What's New**: 이번 연구에서는 대규모 비디오 모델을 통해 얻은 물리적 지식으로 Agent의 동작과 목표를 시각적으로 탐색할 수 있는 새로운 방법을 제안합니다. 기존의 방법들이 Agent 특정 데이터에 기반한 별도의 vision-based inverse dynamic model을 사용해야 했던 것에 비해, 우리는 생성된 비디오 상태를 탐색의 시각적 목표로 활용하여 이를 해결하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 trajectory level action generation과 비디오 안내를 결합하여 Agent가 외부의 감독 없이도 복잡한 작업을 해결할 수 있게 합니다. 연구진은 Libero, MetaWorld, Calvin, iThor Visual Navigation의 다양한 환경에서 8개, 6개, 4개, 12개의 작업을 검증하며, 이론적으로는 행동 클로닝(behavior cloning) 기준선과 유사하거나 더 뛰어난 결과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 50개의 시연 데이터를 이용해 훈련된 BC보다도 높은 성공률을 보여, 비디오 모델에 의해 주어진 목표를 환경 속 탐색을 통해 직접 학습함으로써 이뤄낸 성과입니다. 실험 결과는 훈련 데이터의 양과 다양성에 따라 성과가 향상된다고 나타났으며, 특히 비디오 탐색 빈도와 비디오 모델의 수평선(horizon) 변동이 탐색 성능에 미치는 영향을 평가했습니다.



### TreeCoders: Trees of Transformers (https://arxiv.org/abs/2411.07218)
- **What's New**: 이번 논문에서는 TreeCoders라는 새로운 형태의 transformer tree를 소개합니다. 전통적인 linear transformer에서 k-ary 나무 구조로 이동하여 각 블록이 노드를 구성하며, 클래스 분류기가 최적의 자식을 선택하고 토큰을 특정 리프(leaf)로 라우팅합니다. 이러한 선택자는 transformer 블록 외부에 위치해 다양한 아키텍처를 사용할 수 있게 합니다.

- **Technical Details**: TreeCoder는 transformer 블록과 선택자로 구성된 k-ary 트리 구조입니다. 각 노드는 하나 이상의 디코더 또는 인코더 레이어로 구성되며, 뿌리(root) 노드는 기존 transformer와 같은 입력을 받고 같은 작업을 수행합니다. 이 구조는 sparse node activation을 지원하며, 로그 복잡도를 기반으로 한 트리 검색이 가능합니다.

- **Performance Highlights**: 우리가 제안한 tree transformer 모델은 76%의 경우, 동등한 크기의 linear transformer 모델보다 우수한 성능을 보였으며, 다양한 언어 데이터셋에 걸쳐 경쟁력 있는 결과를 달성했습니다. 또한, 제안된 모델은 배포 구현(distributed implementation)에 적합합니다.



### OmniEdit: Building Image Editing Generalist Models Through Specialist Supervision (https://arxiv.org/abs/2411.07199)
Comments:
          21 pages

- **What's New**: 이 논문은 Omni-Edit라는 새로운 이미지 편집 모델을 소개합니다. 이 모델은 주어진 사용자의 지시에 따라 7가지 다양한 이미지 편집 작업을 처리할 수 있는 강력한 능력을 갖추고 있습니다.

- **Technical Details**: Omni-Edit는 (1) 여러 전문 모델로부터의 감독을 통해 일반화된 편집 모델로 학습됩니다. (2) 소비자 데이터를 위한 중요 샘플링을 사용하여, 데이터 품질을 개선합니다. (3) EditNet이라는 새로운 아키텍처를 도입하여 편집 성공률을 비약적으로 증가시킵니다. (4) 다양한 종횡비와 높은 해상도의 이미지를 사용하여 훈련하여, 실제 환경에서의 활용성을 높입니다.

- **Performance Highlights**: 자동 평가와 인간 평가 모두에서 Omni-Edit는 기존의 모든 모델을 능가하는 성과를 보였습니다. 예를 들면, VIEScore와 같은 자동 메트릭에서 기존 접근 방식보다 유의미하게 높은 점수를 기록했으며, 인간 평가에서는 CosXL-Edit와 비교해 20% 개선된 결과를 보여주었습니다.



### The Super Weight in Large Language Models (https://arxiv.org/abs/2411.07191)
- **What's New**: 최근 연구에서 크고 복잡한 대형 언어 모델(LLM)에서 소수의 파라미터가 모델 품질에 미치는 영향이 크다는 놀라운 결과가 발표되었습니다. 본 논문에서는 단 하나의 파라미터를 제외하는 것만으로 LLM의 텍스트 생성 능력이 파괴될 수 있음을 보여주었습니다.

- **Technical Details**: 논문에서는 'super weights'라는 개념을 제시하고, 이를 단 한 번의 forward pass를 통해 식별하는 데이터 프리(data-free) 방법을 소개합니다. 또한, super weights는 'super activations'를 유도하며, 이 두 가지 요소는 모델 품질에 필수적입니다. super weights와 super activations는 모두 구조적으로 중요하며, pruning(제거) 시 모델 성능이 급격히 감소합니다.

- **Performance Highlights**: 단 하나의 super weight를 제거하면 zero-shot accuracy(제로 샷 정확도)가 사실상 제로로 떨어지며, perplexity(당혹도)는 3배 이상 증가합니다. 본 논문에서 제시한 방법은 기존의 quantization(양자화) 기술과 비교했을 때 데이터 프리 방식으로 높은 성능을 보이며, 라운드 투 니어스(round-to-nearest) 양자화 방식의 품질을 크게 향상시킬 수 있음을 입증합니다.



### NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics (https://arxiv.org/abs/2411.07186)
Comments:
          Demo page: this https URL The code will be open-sourced and available shortly

- **What's New**: 이 논문에서는 생명과학 용도로 특별히 설계된 최초의 오디오-언어 기초 모델인 NatureLM-audio를 소개합니다. 이는 다양한 생물음향(bioacoustics) 과제를 해결하는 데 성공적으로 활용되었으며, 특히 보기 드문 종에 대한 제로샷(zero-shot) 분류를 포함합니다.

- **Technical Details**: NatureLM-audio는 음성 및 음악 데이터와 함께 훈련된 커다란 데이터셋으로부터 학습된 표현들을 동물의 음성에 성공적으로 전이하는 능력을 보여줍니다. 본 모델은 새로운 벤치마크인 BEANS-Zero에서 성능을 평가받아 여러 생물음향 과제에서 새로운 최첨단(SotA) 성능을 기록하였습니다.

- **Performance Highlights**: NatureLM-audio는 고급 생물음향 과제를 해결함에 있어 뛰어난 제로샷(zero-shot) 성능을 보였으며, 이는 이전에 보지 못한 종과 과제에 효과적으로 적용될 수 있음을 의미합니다. 이 모델은 생물다양성 모니터링 및 보전 연구에 대한 기여 가능성을 높이고 있습니다.



### Gradual Fine-Tuning with Graph Routing for Multi-Source Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.07185)
Comments:
          In Proceedings of the 3rd Conference on Lifelong Learning Agents (CoLLAs 2024)

- **What's New**: 이 논문에서는 다중 소스의 비지도 도메인 적응(multi-source unsupervised domain adaptation)을 위한 새로운 프레임워크인 점진적인 미세 조정(Gradual Fine Tuning, GFT)을 제안합니다. GFT는 여러 소스 도메인에서 머신러닝 모델을 훈련하여 타겟 도메인에서의 일반화를 촉진합니다.

- **Technical Details**: 제안된 GFT 프레임워크는 소스 도메인의 가중치가 적용된 무방향 그래프(undirected weighted graph)로 표현되며, 이는 최적의 훈련 순서에 맞는 최적 경로(optimal path)를 결정하는 데 사용됩니다. GFT의 일반화 오차 경계(generalization error bound)를 제시하며, 이를 통해 모델의 성능을 향상시키기 위한 세 가지 경량화된 그래프 라우팅(graph-routing) 전략을 도입합니다.

- **Performance Highlights**: GFT의 최적 전략은 Natural Language Inference (NLI) 작업에서 기존 연구보다 2.3% 높은 정확도를 달성했으며, Sentiment Analysis (SA) 작업에서도 경쟁력 있는 성능을 보여 주었습니다. 특히 SA의 다양한 데이터 서브셋에서는 3.9%의 개선을 이끌어냈습니다.



### Counterfactual Generation from Language Models (https://arxiv.org/abs/2411.07180)
Comments:
          A preprint

- **What's New**: 언어 모델에서 인과 생성 메커니즘을 이해하고 조작하는 것은 모델의 행동을 제어하는 데 필수적입니다. 본 논문에서는 기존의 개입(intervention) 기술 외에도, 카운터팩추얼(counterfactual) 사고방식을 강조하여 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 언어 모델을 일반화된 구조 방정식(Generalized Structural-equation) 모델로 재구성하여 진짜 문자열 카운터팩추얼을 생성합니다. Gumbel-max 트릭(Gumbel-max trick)을 사용하여 샘플링 노이즈의 동일한 인스턴스에서 원래 문자열과 카운터팩추얼 간의 결합 분포(joint distribution)를 모델링합니다. 이 알고리즘은 후견 Gumbel 샘플링(hindsight Gumbel sampling)을 기반으로 하여 관찰된 문자열의 카운터팩추얼을 생성하기 위한 잠재적인 노이즈 변수를 추론합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 의미 있는 카운터팩추얼을 생성하는 동시에 일반적으로 사용되는 개입 기술이 상당한 원하지 않는 부작용을 가진다는 것을 보여줍니다.



### More Expressive Attention with Negative Weights (https://arxiv.org/abs/2411.07176)
- **What's New**: Cog Attention이라는 새로운 주의 메커니즘을 제안하며, 이는 음의 주의 가중치를 허용하여 표현력을 향상시킵니다. 이 메커니즘은 토큰 삭제 및 복사를 정적인 OV 매트릭스에서 동적인 QK 내적(product)으로 전환할 수 있도록 합니다.

- **Technical Details**: Cog Attention은 음의 주의 가중치를 사용할 수 있는 구조로, 토큰의 삭제, 복사 또는 보유를 각각 음의, 양의 또는 최소 주의 가중치로 할당하여 단일 주의 헤드가 더 유연하고 표현력이 높아지도록 합니다.

- **Performance Highlights**: Cog Attention을 사용한 모델들이 전통적인 softmax 주의 모듈을 사용한 모델들에 비해 우수한 성능을 보이며, 이는 언어 모델링 및 이미지 생성 등 다양한 작업에서 입증되었습니다.



### Anytime Sequential Halving in Monte-Carlo Tree Search (https://arxiv.org/abs/2411.07171)
Comments:
          Accepted by the Computers and Games 2024 conference

- **What's New**: 본 논문에서는 Monte-Carlo Tree Search (MCTS)에서 선택 전략으로 사용할 수 있는 anytime Sequential Halving 알고리즘을 제안합니다. 이 알고리즘은 별도의 사전 예산이 필요하지 않으며, 언제든지 중단할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: anytime SH (Sequential Halving)는 Multi-Armed Bandit (MAB) 알고리즘으로, 기본적으로 원래의 SH를 뒤집은 형태입니다. 이 알고리즘은 MCTS의 root node에서 선택 전략으로 사용될 수 있도록 설계되었습니다.

- **Performance Highlights**: MAB 문제와 10개의 다양한 보드 게임에서 empirical (경험적) 실험 결과는 anytime SH의 성능이 UCB1 및 기존의 SH와 경쟁할 만하다는 것을 보여줍니다.



### Acoustic-based 3D Human Pose Estimation Robust to Human Position (https://arxiv.org/abs/2411.07165)
Comments:
          Accepted at BMVC2024

- **What's New**: 이 논문에서는 저수준의 음향 신호만을 이용하여 3D 인간 자세 추정 문제를 탐구합니다. 기존의 방법은 사용자가 스피커와 마이크 사이의 직선 위에 위치해야 한다고 가정하였으나, 이는 실세계에서의 적용에 제한이 있었습니다. 이에 따라 위치 판별기와 잔향 저항 모델을 결합한 새로운 방법을 제안하였습니다.

- **Technical Details**: 제안된 방법은 위치 판별기가 포함되어 있으며, 주체의 위치에 관계없이 인식할 수 있는 특징을 추출합니다. 또한, 추정 대상 시간 이전의 음향 신호를 참조하여 신호의 변동에 대한 강건성을 높이는 방법을 제안합니다. 이 논문에서는 스피커와 마이크 간의 직선에서 멀리 떨어진 여러 위치에서의 데이터를 포함한 새로운 데이터셋을 구축하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보였으며, 다양한 인간 위치를 커버하는 경우에도 안정적인 3D 자세 추정이 가능함을 입증하였습니다.



### RoundTable: Investigating Group Decision-Making Mechanism in Multi-Agent Collaboration (https://arxiv.org/abs/2411.07161)
Comments:
          preprint

- **What's New**: 이번 연구는 분산된 환경에서 다중 에이전트 시스템(Multi-Agent Systems, MAS)의 효율성을 조사하고, 에이전트 간의 의사소통을 촉진하여 집단 지능을 향상시키는 방법을 탐구합니다. 중앙 집중식 메커니즘과 달리, 분산형 집단 의사결정 방식은 에이전트들이 공동으로 deliberation에 참여할 수 있게 합니다.

- **Technical Details**: 연구는 다양한 사회적 선택 방법을 적용하여 의사소통 및 의사결정의 역학을 분석합니다. 다양한 투표 규칙을 통해, 중간 정도의 의사결정 유연성이 더 나은 결과를 가져오는 것을 발견하였고, 에이전트 간 대화의 언어적 특성을 조사하여 효과적인 협업의 지표를 제시합니다. 또한, 언어적 단서를 바탕으로 다중 에이전트 협업에서 최적의 중지점을 결정하는 여러 방법을 제안하였습니다.

- **Performance Highlights**: 연구 결과는 중앙집중식 의사결정이 지닌 문제점을 해결할 수 있는 분산형 의사결정 방식의 가능성을 보여주며, 특정 사회적 선택 방법이 에이전트 간 협력을 어떻게 증진시키거나 저해하는지를 포함한 중요한 통찰을 제공합니다. 또한, 이는 효과적인 MAS 환경 설계를 위한 기초 자료로 활용될 수 있습니다.



### HierTOD: A Task-Oriented Dialogue System Driven by Hierarchical Goals (https://arxiv.org/abs/2411.07152)
- **What's New**: 이 논문에서는 복잡한 작업 환경에서 효과적인 대화형 시스템을 구현하기 위해 HierTOD라는 새로운 Task-Oriented Dialogue (TOD) 시스템을 소개합니다. 이 시스템은 계층적 목표에 기반하여 사용자와의 프로액티브한 상호작용을 지원하며, 종합적인 워크플로우를 지원합니다.

- **Technical Details**: HierTOD는 자연어 이해(NLU), 대화 관리(DM), 응답 생성(RG) 모듈로 구성된 전통적인 TOD 시스템의 파이프라인 접근 방식을 따릅니다. Composite Goal Retriever (CGR) 모듈이 목표 저장소를 구축하여 목표 달성을 위한 워크플로우를 정의하고 저장합니다. 시스템은 상태 머신 구조를 활용하여 대화 흐름을 관리하며, 다양한 데이터 서비스와 연결되어 사용자 지원을 최적화합니다.

- **Performance Highlights**: 인간을 대상으로 한 연구 결과, HierTOD는 슬롯 기반 정보 수집과 단계별 안내 두 가지 패러다임을 모두 지원하며, 사용자에게 더 나은 경험을 제공합니다. 사용자는 시스템을 통해 명확한 질문과 응답을 통한 효율적인 작업 수행이 가능해졌습니다.



### Variational Graph Contrastive Learning (https://arxiv.org/abs/2411.07150)
- **What's New**: 이번 논문에서는 Subgraph Gaussian Embedding Contrast (SGEC) 방법론을 제안합니다. SGEC는 서브그래프를 구조화된 Gaussian 공간으로 능동적으로 매핑하여 그래프의 특성을 보존하면서 생성된 서브그래프의 분포를 제어하는 모듈을 갖추고 있습니다.

- **Technical Details**: SGEC는 Graph Representation Learning (GRL)에 사용되는 Self-supervised Learning (SSL) 방법으로, 서브그래프의 특징을 Gaussian 분포로 매핑하고 이것을 최적 전달 거리(optimal transport distances)인 Wasserstein 및 Gromov-Wasserstein 거리를 통해 평가합니다. 이 과정에서 Kullback-Leibler (KL) 발산을 사용하여 매핑의 정규화를 수행합니다.

- **Performance Highlights**: SGEC는 8개의 벤치마크에서 기존의 최신 방법들과 비교하여 더 나은 성능을 보여주었습니다. 우리의 실험은 생성된 대조 쌍의 분포가 GRL 방법 설계에 중요한 영향이 있음을 강조합니다.



### Edify 3D: Scalable High-Quality 3D Asset Generation (https://arxiv.org/abs/2411.07135)
Comments:
          Project website: this https URL

- **What's New**: Edify 3D는 고품질 3D 자산 생성을 위한 고급 솔루션으로, 다수의 시점에서 RGB 및 표면 노멀 이미지를 합성하는 **diffusion model**을 활용합니다. 이 모델은 2분 이내로 신속한 3D 자산 생성을 가능하게 하며, 산업 표준에 맞는 해상도와 모델 품질을 보장합니다.

- **Technical Details**: Edify 3D는 **text-to-3D** 및 **image-to-3D** 생성 기능이 있으며, 두 가지 종류의 신경망인 **diffusion models**와 **Transformers**에 기반한 기술을 사용합니다. 다중 시점의 RGB 이미지와 표면 노멀을 합성한 후, 이를 사용하여 3D 형태와 재질을 예측하는 **reconstruction model**을 통해 기하학적 데이터와 텍스처 맵, 소재 맵을 생성합니다.

- **Performance Highlights**: Edify 3D는 이전의 text-to-3D 접근 방식보다 뛰어난 3D 형상과 텍스처를 일관되게 생성하며, 4K 해상도의 텍스처와 물리 기반 렌더링(PBR) 재료를 포함하여 뛰어난 품질의 3D 자산을 생성합니다. 이 과정은 효율성과 확장성이 크게 향상되었습니다.



### Token Merging for Training-Free Semantic Binding in Text-to-Image Synthesis (https://arxiv.org/abs/2411.07132)
Comments:
          Accepted by Neurips2024

- **What's New**: 본 연구에서는 텍스트-이미지(T2I) 모델의 의미적 바인딩 문제를 해결하기 위해 새로운 방법인 Token Merging(ToMe)를 제안합니다. 이 방법은 관련된 토큰을 하나의 복합 토큰으로 집계하여 모든 객체, 속성, 하위 객체가 동일한 교차 주의 맵을 공유하도록 합니다.

- **Technical Details**: ToMe는 객체 속성과 하위 객체 간의 적절한 연관성을 증진시키기 위해 설계되었습니다. 또한, 복합 토큰 업데이트를 위한 두 가지 보조 손실(엔트로피 손실과 의미적 바인딩 손실)을 통합하여 T2I 생성 초기 단계에서의 통합성을 높입니다. 이 방법은 T2I-CompBench와 GPT-4o 오브젝트 바인딩 벤치마크에서 객관적인 성능 평가를 받았습니다.

- **Performance Highlights**: ToMe 방법은 복잡한 다중 객체 및 속성 생성 시나리오에서 특히 효과적이며, 많은 기존 방법들을 능가하는 결과를 보여주었습니다. 사용자 친화적인 특성 덕분에 대규모 언어 모델이나 특정 레이아웃 정보에 의존할 필요가 없습니다.



### Fast and Robust Contextual Node Representation Learning over Dynamic Graphs (https://arxiv.org/abs/2411.07123)
- **What's New**: 이 논문은 동적 그래프에서의 노드 표현 유지 문제를 다루며, 기존의 PPR (Personalized PageRank) 기반의 GNN 모델을 개선할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 모델은 희소 노드 간의 주의(attention)의 기반을 두고 있으며, proximal gradient method(ISTA)를 활용하여 PPR 기법의 효율성을 최대 6배 향상시킵니다. 또한, PPR이 GNN의 기초적인 도구로 기능할 수 있도록 여러 속성을 세분화하고 정의합니다.

- **Performance Highlights**: 모델 GoPPE는 간단하면서도 효과적인 포지셔널 인코딩을 사용하여, 기존의 최신 기술들(STOA)과 비교하여 유사한 성능을 보이며, 노이즈가 많은 초기 노드 속성에서 그래프가 진화할 때 성능이 우수합니다.



### Learning Multi-Agent Collaborative Manipulation for Long-Horizon Quadrupedal Pushing (https://arxiv.org/abs/2411.07104)
- **What's New**: 본 논문에서는 여러 사족 로봇이 협력하여 장애물을 인식하고 긴 범위 동안 물체를 누르는 과제를 해결하는 방법을 제안합니다. 이를 위해 3단계로 구성된 계층적 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning, MARL) 프레임워크를 개발하였습니다.

- **Technical Details**: 제안된 프레임워크는 고수준 컨트롤러가 장애물 회피를 위해 RRT(Rapidly-exploring Random Tree) 계획자와 중앙 집중식 적응 정책을 통합하고, 중간 수준 컨트롤러가 분산형 목표 조건 정책을 사용하여 로봇이 서브 목표를 향해 나아가도록 하며, 저수준 컨트롤러는 사전 훈련된 보행 정책으로 명령을 실행합니다.

- **Performance Highlights**: 시뮬레이션에서 제안된 방법은 기존 방법들에 비해 36.0% 더 높은 성공률과 24.5% 단축된 완료 시간을 기록했습니다. 실제 Go1 로봇에서 Push-Cuboid 및 Push-T 작업을 성공적으로 수행함으로써 제안된 방법이 실제 환경에서도 효과적임을 입증하였습니다.



### Bounded Rationality Equilibrium Learning in Mean Field Games (https://arxiv.org/abs/2411.07099)
- **What's New**: 이 연구는 대규모 에이전트 집단에서의 행동을 모델링하기 위해 Mean Field Games (MFGs)의 새로운 접근 방식을 제시하며, Nash Equilibrium (NE)의 한계를 극복하기 위해 bounded rationality를 포함한다.

- **Technical Details**: 논문에서는 quantal response equilibria (QRE)와 receding horizon (RH) MFGs를 정의하고, 이러한 개념들이 MFGs의 새로운 유형으로서 에이전트의 제한된 합리성을 모델링할 수 있음을 보여준다. QRE는 노이즈에 의해 왜곡된 보상을 인식하고 이에 따라 최적의 행동을 취하는 에이전트를 가정한다.

- **Performance Highlights**: 제안된 알고리즘을 사용하여 다양한 예제를 통해 QRE와 RH 균형을 학습하는 능력을 평가하였으며, 실용적인 차이점을 명확히 구분하여 기존의 균형 개념과 비교하였다.



### A Multi-Agent Approach for REST API Testing with Semantic Graphs and LLM-Driven Inputs (https://arxiv.org/abs/2411.07098)
Comments:
          To be published in the 47th IEEE/ACM International Conference on Software Engineering (ICSE 2025)

- **What's New**: 본 논문은 REST API 테스트를 위한 최초의 블랙 박스 프레임워크인 AutoRestTest를 소개합니다. 이 프레임워크는 Multi-Agent Reinforcement Learning (MARL), Semantic Property Dependency Graph (SPDG), 그리고 Large Language Models (LLMs)를 통합하여 종속성이 포함된 다중 에이전트 접근 방식을 채택합니다.

- **Technical Details**: AutoRestTest는 REST API 테스트를 API, 의존성, 매개변수 및 값의 네 개 에이전트가 협력하여 API 탐색을 최적화하는 문제로 다룹니다. LLM은 도메인 특정 값 제한을 처리하고, SPDG 모델은 API 간의 유사성 점수를 사용하여 의존성을 위한 검색 공간을 단순화합니다. MARL은 에이전트의 행동을 동적으로 최적화합니다.

- **Performance Highlights**: AutoRestTest는 RESTGPT에 의해 지원되는 도구를 포함하여, 12개의 실제 REST 서비스에서 코드 커버리지, 운영 커버리지 및 결함 탐지 측면에서 네 개의 주요 블랙 박스 REST API 테스트 도구보다 성능이 우수합니다. 특히, Spotify의 내부 서버 오류를 식별할 수 있는 유일한 도구입니다. 또한, ablation 연구를 통해 에이전트 학습, SPDG 및 LLM 구성 요소의 중요한 기여를 강조합니다.



### StoryTeller: Improving Long Video Description through Global Audio-Visual Character Identification (https://arxiv.org/abs/2411.07076)
- **What's New**: 이 논문에서는 긴 비디오에 대한 일관성 있는 설명을 생성하기 위한 새로운 시스템, StoryTeller를 제안합니다. 이 시스템은 오디오-비주얼 캐릭터 식별(Audio-Visual Character Identification)을 통해 캐릭터와 대사를 효과적으로 연결하여 더 밀집된 비디오 설명을 생성합니다.

- **Technical Details**: StoryTeller는 비디오를 여러 개의 짧은 클립으로 분할하는 비디오 세분화 모듈, 오디오-비주얼 캐릭터 식별 모듈, 그리고 LVLM을 사용하는 설명 생성 모듈로 구성되어 있습니다. 오디오 및 비주얼 입력을 통합한 다중 모달 대형 언어 모델을 사용하여 각 클립에서 캐릭터를 식별합니다.

- **Performance Highlights**: MovieQA 평가에서 StoryTeller는 Gemini-1.5-pro보다 9.5% 높은 정확도를 기록하며, 인간 평가에서는 +15.56%의 우위를 보였습니다. 또한, StoryTeller에서의 오디오-비주얼 캐릭터 식별 정보는 다른 비디오 설명 모델들의 성능도 향상시켰습니다.



### An Interpretable X-ray Style Transfer via Trainable Local Laplacian Filter (https://arxiv.org/abs/2411.07072)
- **What's New**: 이번 연구에서는 방사선 의사가 X-ray 이미지를 진단 성능 향상을 위해 선호하는 시각적 인상, 즉 '스타일'을 자동으로 전환할 수 있는 기법을 제안합니다.

- **Technical Details**: 제안하는 방법은 Local Laplacian Filter (LLF)의 훈련 가능한 버전을 도입하여 스타일 전환의 최적화된 변환 함수를 형성하고, 이를 통해 스타일 전환의 특성을 추론할 수 있도록 합니다. MLP (Multi-Layer Perceptron)를 사용하여 LLF가 복잡한 X-ray 스타일 특징을 포착할 수 있도록 했습니다.

- **Performance Highlights**: 실험 결과, 처리되지 않은 유방 X-ray 이미지를 목표 유방 촬영 이미지의 스타일에 맞춰 변환하여 기존 LLF 스타일 전환 방법의 SSIM (Structural Similarity Index) 0.82에 비해 0.94를 달성하는 효과성을 입증했습니다.



### Universal Response and Emergence of Induction in LLMs (https://arxiv.org/abs/2411.07071)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)에서 유도(induction) 행동의 출현을 조사하여, 모델의 잔여 스트림(residual stream)에 대한 약한 단일 토큰의 변화를 탐색합니다. 이를 통해 유도 신호의 양적 특성을 제공하여 LLM의 동작을 더 잘 이해할 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 Gemma-2-2B, Llama-3.2-3B, GPT-2-XL 모델에서 잔여 스트림에 대한 약한 변화를 통해 유도 행동의 신호를 탐지하였습니다. 저희는 이 모델들이 변화 강도에 관계없이 반응이 비례(scale-invariant)하게 유지되는 보편적인 영역을 갖고 있음을 발견했습니다. 이 방법은 사용자가 모델의 각 레이어에서 유도 행동의 구성 요소를 더 깊이 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 모델의 중간 레이어에서 유도 신호가 점진적으로 나타나면서, 유도 행동을 구성하는 중요 모델 섹션들을 식별할 수 있었습니다. 이 발견은 대규모 회로 분석(large-scale circuit analysis)을 위한 기준이 될 수 있으며, LLM의 내부 상호작용에 대한 통찰력을 제공합니다.



### On Active Privacy Auditing in Supervised Fine-tuning for White-Box Language Models (https://arxiv.org/abs/2411.07070)
- **What's New**: 본 논문에서는 언어 모델(LMs)의 미세 조정(supervised fine-tuning, SFT) 과정에서 개인정보 유출 위험을 식별하고 정량화하기 위한 새로운 능동적 개인정보 감사 프레임워크인 Parsing을 소개합니다. 이 프레임워크는 흰 상자(white-box) 멤버십 추론 공격(membership inference attacks, MIAs)을 핵심 기술로 활용하여 개인정보 보호를 극대화합니다.

- **Technical Details**: Parsing 프레임워크는 두 단계로 이루어진 전략을 적용하여 LMs의 미세 조정 과정에서 개인정보 노출을 모니터링합니다. 이 프레임워크는 GPT-2, Llama2를 포함한 대형 LMs에 대한 MIAs의 효과성을 개선하였으며, 새로운 학습 목표를 도입하여 샘플의 멤버십 표현을 최적화합니다.

- **Performance Highlights**: 실험 결과, Parsing 프레임워크는 다양한 모델과 작업에서 개인정보 위험을 감지하고 정량화하는 데 효과적임을 입증하였습니다. 이 연구는 미세 조정 과정에서의 개인정보 보호 전략을 제안하고 있으며, 연구 커뮤니티에 유용한 도구를 제공합니다.



### Zeroth-Order Adaptive Neuron Alignment Based Pruning without Re-Training (https://arxiv.org/abs/2411.07066)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 네트워크 프루닝(Network Pruning) 기술을 활용하여, LLM(대형 언어 모델)들에 대해 성능 저하 없이 매개변수를 줄일 수 있는 새로운 알고리즘 NeuroAl을 제안합니다. 이 알고리즘은 'top-up' 접근 방식을 사용해, 기존의 여러 프루닝 기법에 쉽게 적용될 수 있습니다.

- **Technical Details**: NeuroAl 알고리즘은 Activations의 일관성을 높이기 위해 블록 및 로우별 희소성 비율을 수정하는 두 단계의 방식으로 구성됩니다. 이 과정에서 사용자에게 하이퍼파라미터를 지정할 필요가 없으며, 모델의 구조와 입력된 희소성 요구에 따라 자동으로 최적의 값을 선택합니다. 이를 통해 프루닝 후에도 과거 프루닝 기법보다 나은 성능을 보장합니다.

- **Performance Highlights**: 4개의 서로 다른 LLM 계열에서 3가지 희소성 비율에 대해 테스트한 결과, NeuroAl은 최신 기술인 OWL 및 DsNoT보다 지속적으로 우수한 성능을 보였습니다. 실험 결과, 60%, 70%, 80%의 높은 희소성에서도 안정적인 성능을 발휘하며, 심도 있는 Ablation Study를 통해 알고리즘의 강건성을 입증했습니다.



### FedCVD: The First Real-World Federated Learning Benchmark on Cardiovascular Disease Data (https://arxiv.org/abs/2411.07050)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문은 실제 세계의 심혈관 질환(CVD) 탐지를 위한 첫 번째 연방 학습(Federated Learning, FL) 벤치마크인 FedCVD를 소개합니다. FedCVD는 7개 기관에서 수집된 자연적으로 분산된 CVD 데이터를 기초로 하여 ECG 분류 및 ECHO 분할의 두 가지 주요 작업을 포함합니다.

- **Technical Details**: FedCVD는 비독립적이고 동일하게 분포되지 않는(Non-IID) 데이터, 긴 꼬리 분포(Long-tail Distribution), 그리고 레이블 불완전성(Label Incompleteness)이라는 세 가지 도전 과제를 강조합니다. 이 데이터 세트와 벤치마크는 실제 의료 환경에서 FL 알고리즘 개발 및 검증에 중요한 역할을 합니다.

- **Performance Highlights**: FedCVD에서의 실험 결과, FL은 기존 중앙 집중식 학습 방법과 비교하여 CVD 문제에서 효과적임을 검증했으며, 새로운 평가 지표를 제시하였습니다. 모든 데이터 세트는 자연스러운 분할 전략을 기반으로 하여 구성되었습니다.



### Minion: A Technology Probe for Resolving Value Conflicts through Expert-Driven and User-Driven Strategies in AI Companion Applications (https://arxiv.org/abs/2411.07042)
Comments:
          18 pages, 5 figures

- **What's New**: 이 연구는 AI 동반자와 사용자 간의 가치 충돌을 이해하고 해결하는 방법을 제안합니다. 연구팀은 151건의 사용자 불만을 분석하여 충돌의 디자인 시사점을 도출하고, Minion이라는 기술 프로브를 개발하여 사용자가 인간-AI 가치 충돌을 해결하는 데 도움을 주고자 하였습니다.

- **Technical Details**: Minion은 전문가 기반(expert-driven) 및 사용자 기반(user-driven) 충돌 해결 전략을 결합한 사용자 권한 부여 개입 방법을 적용합니다. 기술 프로브 연구에서는 40개의 가치 충돌 시나리오를 설정하고, 22명의 참가자가 274개의 작업을 수행하며 94.16%의 충돌 해결률을 기록했습니다.

- **Performance Highlights**: Minion은 참가자들에게 긍정적인 피드백을 받았고, 충돌 해결에 대한 새로운 아이디어를 제공하였습니다. 연구팀은 인간-AI 가치 충돌을 해결하는 데 있어 전문가 및 사용자 전략을 통합하는 기회와 도전을 논의하며, 향후 연구 필요성을 제기했습니다.



### UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction (https://arxiv.org/abs/2411.07019)
- **What's New**: 본 연구에서는 통합 지식 그래프(link prediction) 링크 예측을 위한 통합 계층적 표현 학습 프레임워크(UniHR)를 제안합니다. 이 프레임워크는 하이퍼 관계(hyper-relational), 시계열(temporal), 중첩(nested) 사실을 포함한 다양한 사실 표현을 통일된 방식으로 처리할 수 있습니다.

- **Technical Details**: UniHR 프레임워크는 하이퍼 관계 데이터 표현 모듈(HiDR)과 계층 구조 학습 모듈(HiSL)로 구성됩니다. HiDR는 다양한 형태의 사실을 삼중 표현(triple-based representation)으로 통합하며, HiSL는 개별 사실의 의미 정보를 증강시키고 사실 간의 구조적 정보를 풍부하게 합니다. 이를 통해 링크 예측을 위한 향상된 표현을 생성합니다.

- **Performance Highlights**: 7개의 데이터 세트에 대한 실험 결과 UniHR은 특정 유형의 KG를 위해 설계된 기본 모델들보다 뛰어난 성능을 보였으며, 이는 HiDR의 강력한 일반화 능력과 HiSL 모듈의 효과성을 입증합니다.



### Leveraging LSTM for Predictive Modeling of Satellite Clock Bias (https://arxiv.org/abs/2411.07015)
Comments:
          6 Pages, 6 figures (8 sub-figures), 5 Tables Index Terms-LSTM, Satellite Navigation, Deep Learning, Clock Bias

- **What's New**: 위성 시계 편차 예측의 정확성을 증가시키기 위해 Long Short-Term Memory (LSTM) 네트워크를 활용한 새로운 접근 방식을 제안합니다. 기존의 방법에 비해 LSTM 모델이 RNN보다 170배, MLP보다 2.3 × 10^7배, ARIMA보다 1.9 × 10^4배 더 높은 정확도를 보였습니다.

- **Technical Details**: 본 연구는 Galileo의 PRN 2 위성에서 수집한 데이터를 사용하여 단일 차분 시퀀스를 생성하고 이를 통해 예측 정확도를 높입니다. LSTM 모델은 7일부터 31일까지의 데이터셋 길이에 대해 훈련되며, RMSE(평균 제곱근 오차)를 주요 평가 지표로 사용합니다.

- **Performance Highlights**: LSTM 모델의 RMSE는 2.11 × 10^{-11}로, 전통적인 시계열 예측 방법에 비해 현저한 향상을 보였습니다. 이 연구 결과는 특히 저전력 수신기에서 정확도와 효율성을 높이는 데 기여할 것으로 기대됩니다.



### A neural-network based anomaly detection system and a safety protocol to protect vehicular network (https://arxiv.org/abs/2411.07013)
Comments:
          Master's thesis 2023-2024

- **What's New**: 이 논문은 Cooperative Intelligent Transport Systems (CITS)의 사용을 통해 도로 안전과 효율성을 증진하기 위한 방법을 제안합니다. 특히, 차량 간 통신을 통해 안전하고 정확한 데이터 교환의 중요성을 강조합니다.

- **Technical Details**: CITS는 차량 간 통신을 통해 사고 예방 및 교통 흐름 최적화에 기여합니다. 논문에서는 Long Short-Term Memory (LSTM) 네트워크를 사용하는 Machine Learning 기반의 Misbehavior Detection System (MDS)을 제안하고, VeReMi 데이터셋을 기반으로 오프라인에서 훈련시킨 후, 실제 시나리오에서 테스트합니다.

- **Performance Highlights**: MDS는 잘못된 메시지로 인한 사고를 거의 모두 예방할 수 있으며, 이상 현상이 감지되면 방어 프로토콜을 발동하여 차량 무리를 해체합니다. 그러나 다양한 교통 상황에서 특정 유형의 잘못된 행동을 식별하는 데 어려움을 겪고 있어 보편적인 적응 프로토콜을 만드는 것이 어렵다는 결과를 보였습니다.



### Non-Adversarial Inverse Reinforcement Learning via Successor Feature Matching (https://arxiv.org/abs/2411.07007)
- **What's New**: 본 연구는 Inverse Reinforcement Learning (IRL)에서의 새로운 접근 방법인 Successor Feature Matching (SFM)을 제안합니다. 이 기법은 전문가의 행동을 복제할 때 보상 함수 학습이 필요하지 않으며, 단 하나의 전문가 시연만으로도 효과적으로 학습할 수 있습니다.

- **Technical Details**: SFM은 상태만을 이용하여 기대 누적 특성을 추정하는 데 Successor Features (SF)를 활용하여, 정책 경량화(Policy Gradient Descent) 방법을 통해 학습합니다. 이 방식은 전통적인 적대적 접근 방식과 달리 안정적인 솔루션을 제공하며, Actor-Critic RL 알고리즘과 통합이 용이합니다.

- **Performance Highlights**: SFM은 다양한 제어 작업에서 평균 정규화 수익률이 16% 향상된 성능을 보여주었고, 단 하나의 전문가 시연으로도 효과적인 모방 학습이 가능함을 입증했습니다.



### Enhancing Robot Assistive Behaviour with Reinforcement Learning and Theory of Mind (https://arxiv.org/abs/2411.07003)
- **What's New**: 이 연구 논문은 Theory of Mind (ToM) 능력을 갖춘 적응형 로봇이 사용자 성능 및 인식에 미치는 영향을 조사하는 탐색적 비교 연구를 제시합니다.

- **Technical Details**: 연구에서는 두 층의 아키텍처를 설계하였습니다. 상위 단계인 Q-learning 에이전트가 로봇의 고급 행동을 학습하며, 하위 단계에서는 사용자 의도된 전략을 추론하는 휴리스틱 기반 ToM이 역할을 합니다. 이 과정에서 ToM은 로봇의 도움을 제공하고 그 선택의 이유를 설명하는 데 사용됩니다.

- **Performance Highlights**: ToM 기능을 갖춘 로봇과 상호작용한 참가자들이 더 나은 성과를 내고 로봇의 도움을 더 많이 수용하였으며, 로봇이 자신의 의도를 잘 인식하고 예측한다고 인식했습니다.



### Token2Wav (https://arxiv.org/abs/2411.06989)
- **What's New**: 이번 논문에서는 Wave Network에서 유래된 새로운 토큰 표현 방법인 Token2Wave에 대한 심층 분석을 제공하며, 이를 통해 입력 텍스트의 전역 및 지역 의미를 파악할 수 있도록 설계되었습니다.

- **Technical Details**: Token2Wave에서는 각 토큰이 전역 의미를 포착하는 magnitude component와 개별 토큰의 관계를 인코딩하는 phase component로 구성된 복소 벡터로 표현됩니다. 이 연구는 Token2Wave 프레임워크 내에서의 수렴 거동, 역전파 특성 및 임베딩 독립성에 대해 조사하였습니다.

- **Performance Highlights**: Token2Wave는 BERT에 비해 비디오 메모리 사용량과 학습 시간을 현저히 줄일 수 있으며, [CLS] 토큰, 전체 입력 텍스트, 분류기 매개변수에 대한 기울기 비교를 통해 Token2Wave의 독특한 특성을 강조합니다.



### Imitation from Diverse Behaviors: Wasserstein Quality Diversity Imitation Learning with Single-Step Archive Exploration (https://arxiv.org/abs/2411.06965)
- **What's New**: 이번 연구에서는 Wasserstein Quality Diversity Imitation Learning (WQDIL)을 소개하여, 제한된 시연으로부터 다양한 고품질 정책을 배우는 데 필요한 혁신적인 접근 방식을 제시합니다. 이 방법은 Wasserstein Auto-Encoder (WAE)를 기반으로 잠재적 적대적 훈련을 통해 모방 학습의 안정성을 향상시키고, 행동 과적합 이슈를 완화하기 위해 측정 조건 보상 함수를 사용합니다.

- **Technical Details**: WQDIL은 Wasserstein Auto-Encoder (WAE) 내의 잠재 공간에서 Wasserstein 적대적 훈련을 적용하여 보상 모델의 안정성을 높입니다. 또한, 단일 스텝 아카이브 탐사를 통한 보너스를 도입하여 에이전트가 보다 다양한 행동을 수집하도록 유도하는 구조입니다. 이로써 교육이 불안정하고 행동이 과적합되는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, WQDIL은 최신 모방 학습(IL) 방법들을 훨씬 초월하는 성능을 기록하며, MuJoCo 환경에서 파생된 도전적인 연속 제어 작업에서 전문가 수준의 QD 성능을 달성하였습니다.



### ENAT: Rethinking Spatial-temporal Interactions in Token-based Image Synthesis (https://arxiv.org/abs/2411.06959)
Comments:
          Accepted by NeurIPS2024

- **What's New**: 본 논문에서는 효율적인 비자기 회귀 트랜스포머(Non-Autoregressive Transformers, NATs) 모델을 기반으로 한 EfficientNAT (ENAT)이라는 새로운 이미지를 생성하는 모델을 제안합니다. ENAT는 공간적 및 시간적 상호작용을 최적화하여 성능을 개선하고 계산 비용을 크게 줄입니다.

- **Technical Details**: ENAT는 마스크 토큰과 보이는 토큰의 계산을 분리하여 보이는 토큰을 독립적으로 인코딩하고, 마스크 토큰은 완전히 인코딩된 보이는 토큰을 바탕으로 디코딩됩니다. 시간적으로는 '비판적 토큰(critical tokens)'의 계산을 우선시하고, 이전에 계산된 토큰 표현을 최대한 재사용하여 필요한 정보를 보완합니다.

- **Performance Highlights**: ENAT는 ImageNet-256 및 MS-COCO에서 실험을 통해 기존의 NATs에 비해 24%의 성능 향상과 1.8배의 계산 비용 절감을 동시에 달성하였습니다.



### Electroencephalogram-based Multi-class Decoding of Attended Speakers' Direction with Audio Spatial Spectrum (https://arxiv.org/abs/2411.06928)
- **What's New**: 본 논문은 청취자의 EEG 신호를 사용하여 주목하는 화자의 방향성을 보다 정밀하게 해독하는 방법을 제안합니다. 특히, 오디오 공간 정보(audio spatial information)를 활용하여 기존의 이진 방향성 해독을 넘어서 더 높은 정확도의 다중 클래스 방향성 해독을 가능하게 했습니다.

- **Technical Details**: 본 연구에서는 CNN, LSM-CNN, EEG-Deformer 모델을 사용하여 청취자의 EEG 신호와 보조 오디오 공간 스펙트라(auxiliary audio spatial spectra)를 통합하여 방향성 집중을 해독합니다. 특히, 최근 제안된 15-class(클래스) 방향성 집중 데이터셋에서 실험을 진행하였으며, leave-one-subject-out 및 leave-one-trial-out 시나리오에서 성능을 평가했습니다.

- **Performance Highlights**: 제안된 Sp-Aux-Deformer 모델은 leave-one-subject-out 시나리오에서 57.48%, leave-one-trial-out 시나리오에서 61.83%의 15-class(클래스) 해독 정확도를 달성하였습니다.



### Slowing Down Forgetting in Continual Learning (https://arxiv.org/abs/2411.06916)
- **What's New**: 본 연구는 연속 학습(Continual Learning)에서 재난적인 망각(catastrophic forgetting)을 완화하기 위한 새로운 프레임워크인 ReCL(Reconstruction from Continual Learning)을 제안합니다. 이 프레임워크는 기초적인 뉴럴 네트워크 학습의 암묵적 편향을 이용하여 이전 태스크의 데이터를 재구성하고, 이를 현재의 학습 데이터와 결합하여 망각을 느리게 합니다.

- **Technical Details**: ReCL 프레임워크는 이전 태스크에서 훈련한 데이터를 네트워크 파라미터의 최적화 과정에서 메모리처럼 활용합니다. 이를 통해 개선된 성능을 다양한 CL 시나리오(예: class incremental learning, domain incremental learning, task incremental learning)와 데이터셋(MNIST, CIFAR10)에서 입증하였습니다. 또한, 다층 퍼셉트론(MLP)과 합성곱 신경망(CNN) 구조에서도 성능 향상이 관찰되었습니다.

- **Performance Highlights**: 모든 실험에서 ReCL 프레임워크는 기존의 CL 방법들과 결합할 때에도 일관된 성능 향상을 보여줍니다. 이 연구는 또한 EWC, ER, UPGD와 같은 최신 CL 기법들과 결합했을 때, 특히 망각을 더욱 줄이는 결과를 보였습니다.



### Gaussian Process Emulators for Few-Shot Segmentation in Cardiac MRI (https://arxiv.org/abs/2411.06911)
Comments:
          Submitted to Statistical Atlases and Computational Modeling of the Heart (STACOM) 2024

- **What's New**: 이 논문에서는 cardiac MRI(심장 자기공명영상)의 세분화를 개선하기 위해 few-shot learning을 U-Net 아키텍처 및 Gaussian Process Emulators(GPEs)와 결합한 새로운 방법을 제안합니다. 이 방법은 데이터 적재를 최소화하면서도, 적은 양의 라벨이 있는 지원 집합으로부터 더 나은 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 U-Net의 수축 부분을 이용하여 query 이미지와 지원 이미지의 관계를 잠재 공간(latent space)에서 GPEs가 학습하도록 하며, 이 정보는 확장 부분으로 통합되어 query 이미지의 세분화를 보다 정확하게 수행합니다. 또한, M&Ms-2 공공 데이터셋을 사용하여 다양한 각도에서 심장을 세분화하는 능력을 평가했습니다. 이 모델은 기존의 무감독 및 few-shot 방법과 비교하여 더 높은 DICE 계수를 기록했습니다.

- **Performance Highlights**: 모델의 성능은 심장 자기공명영상에서 작은 지원 집합의 크기로 인해 특히 도전적인 설정에서 다른 방법들과 비교할 때 현저하게 개선되었으며, 더 나은 일반화 능력을 보였습니다.



### LongSafetyBench: Long-Context LLMs Struggle with Safety Issues (https://arxiv.org/abs/2411.06899)
- **What's New**: 이 논문에서는 장기 맥락 언어 모델(long-context language models)의 안전성을 평가하기 위한 최초의 기준인 LongSafetyBench를 소개합니다. 이 기준은 10개의 작업 범주로 구성되어 있으며, 평균 41,889 단어의 길이를 가집니다. 최근까지의 연구들은 주로 모델의 기능에 초점을 맞췄지만, 이 논문은 안전성에 대한 객관적이고 포괄적인 평가를 다룹니다.

- **Technical Details**: LongSafetyBench는 모델의 안전성 문제를 해결하기 위해 불법 활동(Illegal Activities), 잘못된 정보 피해(Misinformation Harm), 공격성 및 편향(Offensiveness and Bias)과 같은 세 가지 유형의 위험한 시나리오를 대상으로 데이터를 수집하고 구성하였습니다. 각 문항은 다중 선택 질문 형식으로 포맷팅되었으며, 총 1,203개의 테스트 인스턴스가 포함되어 있습니다.

- **Performance Highlights**: 8개의 장기 맥락 LLM을 LongSafetyBench에서 테스트한 결과, 대부분의 주류 장기 맥락 LLM에서 안전한 응답의 비율이 50% 미만으로 나타났습니다. 장기 맥락 시나리오에서의 안전성과 단기 맥락에서의 성능 간의 일치성이 떨어지며, 모델이 긴 텍스트 내의 해로운 콘텐츠를 간과하는 경향이 있음을 확인했습니다. 또한, 적은 양의 데이터로 훈련한 오픈 소스 모델이 최상위 폐쇄형 모델과 유사한 성능을 달성할 수 있는 것으로 나타났습니다.



### GraphRPM: Risk Pattern Mining on Industrial Large Attributed Graphs (https://arxiv.org/abs/2411.06878)
Comments:
          Accepted by ECML PKDD 2024

- **What's New**: 이 논문에서는 대규모 속성 그래프에서 위험 패턴을 발견하기 위한 새로운 프레임워크인 GraphRPM을 소개합니다. GraphRPM은 업계 특정으로 설계된 병렬 및 분산 위험 패턴 마이닝 프레임워크이며, 최신 Edge-Involved Graph Isomorphism Network (EGIN)과 최적화된 병렬 그래프 계산 연산을 통합하여 비약적인 성능 개선을 이뤘습니다.

- **Technical Details**: GraphRPM은 대규모 속성 그래프에서 위험 패턴을 효율적으로 발견하기 위해 두 단계의 마이닝 전략과 병렬 분산 처리 프레임워크를 구현합니다. 이 프레임워크는 EGIN을 기반으로 하여 속성 그래프 패턴의 모호한 매칭 문제를 해결하며, 연산의 복잡성과 정확성을 균형 있게 관리합니다. 또한, 효과적인 위험 그래프 패턴을 식별하기 위한 평가 메트릭스를 도입합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 포괄적인 실험 평가 결과, GraphRPM은 대규모 산업 속성 그래프에서 패턴 마이닝의 도전 과제를 성공적으로 해결하는 능력을 입증하였습니다. 이는 산업 적용을 위한 상당한 가치와 혁신을 제공하는 연구로, 데이터 마이닝 및 기계 학습의 응용 분야에서 중요한 진전을 이룩하였습니다.



### Multi-Modal interpretable automatic video captioning (https://arxiv.org/abs/2411.06872)
- **What's New**: 본 논문은 비디오 자막 생성에 관한 새로운 접근 방식을 제안합니다. 특히, 시각 정보뿐만 아니라 오디오 정보를 통합하여 더 나은 자막을 생성하는 멀티 모달 학습 프레임워크를 도입합니다.

- **Technical Details**: 새로운 비디오 자막 생성 방법은 멀티 모달 대비 손실(multi-modal contrastive loss)을 사용하여 시각 및 청각 정보를 통합하고, 주의(attention) 메커니즘을 통해 모델의 의사결정 과정에 대한 해석 가능성(interpretable)을 제공합니다. 이러한 접근 방식은 입력으로 들어오는 이미지 시퀀스와 오디오 캡션을 처리하는 인코더-디코더 구조를 활용합니다.

- **Performance Highlights**: 제안된 방법은 MSR-VTT와 VATEX 같은 벤치마크 데이터셋에서 최신 모델들과 비교했을 때 우수한 성능을 보였습니다.



### AI-Native Multi-Access Future Networks -- The REASON Architectur (https://arxiv.org/abs/2411.06870)
Comments:
          Accepted for publication at IEEE Access

- **What's New**: 6세대 통신 네트워크(6G)가 2030년 도입 목표로 개발되고 있으며, REASON 프로젝트를 통해 AI 통합 및 지속 가능한 운영과 같은 핵심 기능을 다룬다.

- **Technical Details**: REASON 아키텍처는 물리적 인프라(Physical Infrastructure), 네트워크 서비스(Network Service), 지식(Knowledge), 최종 사용자 애플리케이션(End-User Application)으로 구성된 네 개의 수평적 레이어와 관리 및 오케스트레이션(Management and Orchestration), E2E 보안(E2E Security)이라는 두 개의 수직적 레이어로 구성된다.

- **Performance Highlights**: REASON 아키텍처는 모듈화(Modularity), 상호 운용성(Interoperability), 확장성(Scalability) 및 보안(Security)을 강조하며, 6G 사용 사례 지원을 위한 네트워크 밀도(Network Densification)를 촉진한다.



### Subgraph Retrieval Enhanced by Graph-Text Alignment for Commonsense Question Answering (https://arxiv.org/abs/2411.06866)
Comments:
          Accepted by ECML PKDD 2024

- **What's New**: 본 논문에서는 Commonsense Question Answering (CSQA) 작업을 위한 새로운 프레임워크인 SEPTA를 제안합니다. SEPTA는 Knowledge Graph (KG)를 효율적으로 활용하여 소프트웨어가 공통 상식에 기반한 논리적 사고를 수행할 수 있도록 설계되었습니다.

- **Technical Details**: SEPTA는 Knowledge Graph를 서브그래프 벡터 데이터베이스로 변환하고, BFS 스타일의 샘플링 전략을 사용하여 정보 손실을 최소화합니다. 또한, 그래프와 텍스트 인코더 사이의 의미 공간을 정렬하기 위한 양방향 대비 학습 접근법을 제안하여 정보 통합을 효과적으로 개선합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에 대한 대규모 실험 결과, 제안된 SEPTA 프레임워크가 기존의 최첨단 방법들보다 우수한 성능을 보였으며, 약한 감독 설정에서도 유망한 성과를 달성했습니다.



### Computable Model-Independent Bounds for Adversarial Quantum Machine Learning (https://arxiv.org/abs/2411.06863)
Comments:
          21 pages, 9 figures

- **What's New**: 이 연구는 양자 기계 학습(QML)의 적대적 공격에 대한 내성을 평가하기 위한 모델 독립적인 경계(bound)를 첫 번째로 도입하고, 고급 양자 기반 적대적 공격에 대한 근사적 하한을 계산하여 QML 모델의 본질적인 강인성을 증명합니다.

- **Technical Details**: 이 연구는 적대적 오류 비율에 대한 새로운 하한 추정 방법을 제시하며, 이는 양자 모델의 구조에 독립적이며, 양자 왜곡 공격(quantal perturbation attacks)과 같은 양자 모델에 특화된 서로 다른 공격을 처리합니다. Projected Gradient Descent (PGD)를 기반으로 한 새로운 양자 공격 전략을 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 경계는 실제 양자 모델에서 관찰된 적대적 오류 비율과 강한 상관관계를 보여주며, 가장 좋은 경우에 실험 오류가 추정된 경계보다 10%만큼 높아 본질적인 강인성을 입증합니다.



### Enhancing Phishing Detection through Feature Importance Analysis and Explainable AI: A Comparative Study of CatBoost, XGBoost, and EBM Models (https://arxiv.org/abs/2411.06860)
- **What's New**: 본 연구는 온라인 보안에 대한 지속적인 위협인 피싱 공격에 효과적으로 대응하기 위해 머신러닝(machine learning)을 이용한 피싱 URL 탐지 방법을 제안합니다. 이 과정에서 특징 선택(feature selection)과 모델 해석 가능성(model interpretability)의 중요성이 강조되었습니다.

- **Technical Details**: 연구에서는 Recursive Feature Elimination 기법을 사용하여 주요 특징인 "length_url", "time_domain_activation", "Page_rank" 등을 피싱 시도의 강력한 지표로 선정하였습니다. 또한, CatBoost, XGBoost, Explainable Boosting Machine과 같은 다양한 알고리즘의 내구성과 확장성을 평가하였습니다. XGBoost는 런타임(runtime) 측면에서 매우 효율적인 성능을 보였으며, 대규모 데이터셋에 적합한 것으로 나타났습니다. 반면 CatBoost는 특징이 줄어들었음에도 높은 정확도를 유지하는 강인함을 보였습니다.

- **Performance Highlights**: Explainable AI 기법인 SHAP를 사용하여 특징의 중요성에 대한 통찰을 제공함으로써 투명성(transparency)과 신뢰성(trustworthiness)을 높였습니다. 연구 결과는 효과적인 특징 선택과 모델 해석 가능성이 피싱 탐지 시스템을 크게 강화할 수 있음을 보여주며, 변화하는 사이버 위협에 대해 보다 효율적이고 적응 가능한 방어 체계를 구축할 수 있는 기반을 마련하였습니다.



### Scientific machine learning in ecological systems: A study on the predator-prey dynamics (https://arxiv.org/abs/2411.06858)
Comments:
          16 pages, 7 figures, 1 table

- **What's New**: 본 연구에서는 Neural Ordinary Differential Equations (Neural ODEs) 및 Universal Differential Equations (UDEs)라는 과학적 기계 학습의 두 가지 핵심 기법을 Lotka-Volterra 포식자-피식자 모델에 적용하였습니다. 이 모델은 포식자와 피식자 개체군 간의 동적 상호작용을 설명하는 기본 생태 모델입니다. 논문에서는 사전 지식 없이 데이터와 신경망만을 이용하여 내재된 미분 방정식을 찾아내는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 Julia 프로그래밍 언어를 사용하여 Neural ODEs와 UDEs를 통해 Lotka-Volterra 시스템의 예측 및 예보를 수행합니다. Neural ODE는 전통적인 ODE/PDE 시스템을 신경망으로 대체하며, UDE는 미분 방정식의 특정 항을 신경망으로 대체합니다. UDE는 데이터에서 미지의 역학을 학습하면서 알려진 물리 법칙을 보존합니다. 또한 Gaussian noise를 도입하여 데이터의 노이즈 저항성을 확인하였으며, Hyperparameter optimization을 통해 최적의 신경망 구조와 활성화 함수, 최적화 알고리즘을 조사하였습니다.

- **Performance Highlights**: UDE는 적은 양의 훈련 데이터로도 정확한 예측을 달성하며, Neural ODEs보다 더 뛰어난 성능을 보였습니다. 특히, UDE는 Gaussian noise가 있는 데이터에서도 더욱 강인한 모습을 보였으나, Neural ODE는 높은 수준의 노이즈에 어려움을 겪었습니다. 추가로, 분석을 통해 예측 정확도가 크게 저하되기 시작하는 '예측 분해점(forecasting breakdown point)'의 개념을 도입하여, 장기 예측 과제에서 현재 SciML 프레임워크의 한계를 조명하였습니다.



### Evaluating Large Language Models on Financial Report Summarization: An Empirical Study (https://arxiv.org/abs/2411.06852)
- **What's New**: 이 논문은 GLM-4, Mistral-NeMo, LLaMA3.1의 세 개의 최신 LLM을 비교 분석하여 자동화된 금융 보고서 생성에서의 효과성을 평가합니다. 현 금융 환경에서의 LLM의 신뢰성, 정확성 및 규정 준수를 보장하기 위한 엄격한 검토의 필요성을 강조합니다.

- **Technical Details**: 이 연구에서는 ROUGE-1, BERT Score, LLM Score와 같은 제안된 지표를 포함하여 금융 보고서 분석을 위한 벤치마크를 제공합니다. 정량적 메트릭(precision, recall 등)과 정성적 분석(맥락 적합성, 일관성 등)을 통합한 혁신적인 평가 프레임워크를 도입하여 각 모델의 출력 품질을 종합적으로 평가합니다.

- **Performance Highlights**: 이 논문은 금융 보고서에서의 LLM 성능에 대한 구체적인 벤치마크를 확립하며, LLM이 금융 분야에 적합하게 조정되어야 하는 도전 과제를 강조합니다. 공개된 데이터셋을 통해 연구자들이 이 연구 결과를 검토하고 개선할 수 있는 협력 환경을 조성합니다.



### 1-800-SHARED-TASKS @ NLU of Devanagari Script Languages: Detection of Language, Hate Speech, and Targets using LLMs (https://arxiv.org/abs/2411.06850)
Comments:
          13 pages, Submitted to CHIPSAL workshop @ COLING 2025

- **What's New**: 이번 연구에서는 CHiPSAL 2025의 언어 감지, 증오 발언 식별 및 목표 감지에 관한 시스템을 설명합니다. Devanagari 스크립트 언어에서의 자연어 이해(NLP) 문제를 해결하기 위해 MuRIL, IndicBERT, Gemma-2와 같은 대형 언어 모델을 활용했습니다.

- **Technical Details**: 각 서브태스크에 대해 다양한 멀티링구얼 모델을 파인 튜닝(fine-tune)하고 평가 단계에서 최상의 모델을 선택했습니다. 서브태스크 A는 5개 언어 중에서 언어를 식별하며, B는 텍스트에서 증오 발언을 감지하고, C는 증오 발언의 목표를 분류합니다. Focal loss를 사용하여 클래스 불균형 문제를 해결했습니다.

- **Performance Highlights**: 서브태스크 A에서 F1 점수 0.9980, B에서 0.7652, C에서 0.6804를 달성하여 강력한 성능을 나타냈습니다. 특히, Ensemble 기법을 통해 최종 테스트에서 서브태스크 A의 성능을 더욱 개선했습니다.



### LLM-Neo: Parameter Efficient Knowledge Distillation for Large Language Models (https://arxiv.org/abs/2411.06839)
Comments:
          ICASSP 25' under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 소형 학생 모델로 효율적으로 지식을 전이하는 새로운 LLM-Neo 프레임워크를 제안합니다. 기존의 지식 증류(KD)와 저차원 적응(LoRA) 개념을 통합하여 지식 전이의 효율성을 개선합니다.

- **Technical Details**: LLM-Neo는 LoRA의 저차원 브랜치를 활용하여 효율적인 지식 증류를 달성합니다. KD 손실은 실제 레이블과 교차 엔트로피 손실, 그리고 교사 모델의 예측과의 Kullback-Leibler(KL) 발산을 결합하여 정의됩니다. 이 과정에서 LoRA의 파라미터 효율성을 유지할 수 있습니다.

- **Performance Highlights**: LLM-Neo는 Llama 2 및 Llama 3.1 모델을 압축하는 실험에서 다양한 기준 모델보다 우수한 성능을 보였습니다. 추가 분석을 통해 LoRA 변종에 대한 LLM-Neo의 강건성도 확인되었습니다.



### AssistRAG: Boosting the Potential of Large Language Models with an Intelligent Information Assistan (https://arxiv.org/abs/2411.06805)
Comments:
          Accepted by NeurIPS 2024 (poster)

- **What's New**: 이번 논문에서는 기존의 RAG 방법론의 한계를 극복하기 위해 AssistRAG라는 새로운 접근 방식을 제안합니다. 이 방법론은 LLM 내부에 지능형 정보 어시스턴트를 통합하여 정보 검색 및 의사결정 능력을 향상시킵니다.

- **Technical Details**: AssistRAG는 두 가지 주요 기능인 메모리 관리와 지식 관리로 구성됩니다. 메모리 관리는 내부 메모리의 내용을 통합 및 분석하며, 지식 관리는 외부 지식을 활용하는 데 초점을 맞춥니다. 이를 위해 네 가지 핵심 기능인 Tool usage, Action execution, Memory building, Plan specification을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, AssistRAG는 기존의 벤치마크를 초월하는 성능을 보여주었으며, 특히 덜 발전된 LLM에 더 큰 이점을 제공합니다. 다양한 복잡한 질문 응답 데이터셋에서 우수한 추론 능력을 나타냈습니다.



### LA4SR: illuminating the dark proteome with generative AI (https://arxiv.org/abs/2411.06798)
- **What's New**: 이 논문에서는 AI 언어 모델(LMs)이 미생물 서열 분류에 효과적으로 재설계되었음을 보여줍니다.

- **Technical Details**: 재설계된 오픈 소스 모델은 GPT-2, BLOOM, DistilRoBERTa, ELECTRA 및 Mamba로, 매개변수 수는 70M에서 12B까지 다양합니다. 이 모델들은 F1 점수 95를 기록하고, BLASTP보다 16,580배 빠르고 2.9배 높은 recall을 달성했습니다.

- **Performance Highlights**: 대형 모델(>1B) LA4SR은 사용 가능한 데이터의 2% 미만으로 훈련했음에도 불구하고 높은 정확성(F1 > 86)을 보였고, 완전한 Hi-C/Pacbio Chlamydomonas 게놈을 포함한 새로운 데이터로 검증되었습니다. 이는 결측 서열에서의 강력한 일반화 능력을 보여줍니다.



### Evolving Efficient Genetic Encoding for Deep Spiking Neural Networks (https://arxiv.org/abs/2411.06792)
- **What's New**: 이번 연구는 스파이킹 신경망(Spiking Neural Networks, SNNs)의 효율적인 유전자 인코딩 전략을 통해 대규모 네트워크의 연산 비용을 낮추는 방법을 제안합니다. 이 방식은 연결망의 뉴런 인코딩을 최적화하기 위해 유전자 상호작용을 활용하여 파라미터 및 에너지 소비를 줄입니다.

- **Technical Details**: 제안된 방법은 유전자 기반의 확장된 SNN 인코딩 방식을 통해 직접적인 가중치 업데이트 대신 뉴런 인코딩 학습을 수행합니다. 스페이셜 및 타임피컬한 맥락에서 진화적 초기 배선을 최적화하기 위해 Covariance Matrix Adaptation Evolution Strategy (CMA-ES)를 적용하며, 두 가지 동적 정규화 연산자를 사용하여 진화 효율을 높입니다.

- **Performance Highlights**: 실험 결과, CIFAR-10, CIFAR-100, ImageNet 데이터셋에서 약 50%에서 80%의 파라미터 감소를 달성했으며, 같은 아키텍처 모델 대비 0.21%에서 4.38%의 성능 향상을 보였습니다. 이는 뇌의 진화적 유전 코딩 원리를 근거로 한 SNN 최적화의 이점을 강조합니다.



### ScaleKD: Strong Vision Transformers Could Be Excellent Teachers (https://arxiv.org/abs/2411.06786)
Comments:
          This work is accepted to NeurIPS 2024. The project page: this https URL

- **What's New**: 이 논문에서는 잘 정밀 훈련된 비전 트랜스포머 (ViT) 모델이 교사 역할을 하여 크로스 아키텍처 지식 증류 (Knowledge Distillation, KD) 연구를 진전시키는 데 활용될 수 있는지를 질문합니다.

- **Technical Details**: 우리는 ScaleKD라는 새로운 KD 방법을 제안합니다. 이 방법은 세 가지 상호 연관된 구성요소인 cross attention projector (CAP), dual-view feature mimicking, teacher parameter perception을 결합하여, CNN, MLP 및 ViT 아키텍처 전반에 걸쳐 활용될 수 있는 효과적인 지식 전이 방법을 제공합니다.

- **Performance Highlights**: ScaleKD는 ImageNet-1K 데이터셋에서 MobileNet-V1, ResNet-50, ViT-S/16과 같은 다양한 모델들에 대해 각각의 개별 훈련 모델 대비 2%에서 5%까지 절대 정확도 향상을 보여주며, 다운스트림 MS-COCO 및 ADE20K 데이터셋에서도 우수한 성능을 발휘합니다.



### QuadWBG: Generalizable Quadrupedal Whole-Body Grasping (https://arxiv.org/abs/2411.06782)
- **What's New**: 이번 연구에서는 로봇 팔에 장착된 카메라 하나를 기반으로 강력하고 범용적인 전체 몸체의 loco-manipulation (로코-조작) 제어기를 위한 모듈형 프레임워크를 제시합니다. 이는 로봇의 움직임을 조정하고 다양한 개체를 조작하는 데 필요한 통합된 제어 방안을 제공합니다.

- **Technical Details**: 제안된 시스템은 일반화된 지향 도달 가능성 맵(Generalized Oriented Reachability Map, GORM)을 사용하여 5차원(5D) 명령을 관리할 수 있는 저수준 정책과 그립 감지 인식을 위한 고수준 정책을 조화롭게 구성합니다. 이러한 시스템은 로봇의 전체 이동성과 조작성을 향상시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 시스템은 실제 환경에서 89%의 일회성 그립 정확도를 달성했습니다. 또한, 다양한 전체 몸체 loco-manipulation 작업을 수행할 수 있는 역량을 보여주며, 투명한 물체와 같은 도전적인 작업에서도 우수한 성과를 기록했습니다.



### Machine vision-aware quality metrics for compressed image and video assessmen (https://arxiv.org/abs/2411.06776)
Comments:
          16 pages, 10 figures

- **What's New**: 본 연구는 video-compression 알고리즘의 새로운 관점을 제시하며, 머신 비전(Computer Vision) 향상에 최적화된 비디오 코덱(Video Codec)을 개발하는 데 필요한 프레임워크를 제공합니다. 특히, 객체 탐지(Object Detection), 얼굴 인식(Face Recognition), 번호판 인식(License Plate Recognition) 등 다양한 비전 알고리즘에 대한 영상 품질 평가 메트릭스를 도입하고, 이 메트릭스가 기존 메트릭스보다 머신 비전 결과와 더 높은 상관관계를 가지도록 개선되었습니다.

- **Technical Details**: 본 연구에서는 머신 비전을 위한 새로운 품질 평가 메트릭스를 제안하였으며, 딥 뉴럴 네트워크(Deep Neural Networks)에 기반한 기존 품질 평가 방법이 제공하는 낮은 상관관계를 분석하였습니다. 연구 방법론은 PSNR, SSIM 등의 기존 메트릭이 아닌, 객체 탐지, 얼굴 탐지 및 인식, 번호판 탐지 및 인식 성능을 기준으로 한 CNN(Convolutional Neural Network) 모델 기반의 새로운 비디오 품질 메트릭스를 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 메트릭스는 객체 탐지, 얼굴 탐지 및 인식, 번호판 탐지 및 인식 작업에서 기존 메트릭스들보다 성능이 높은 상관관계를 보였습니다. 이는 기존의 인간 구인 시각적 품질 평가 메트릭으로는 평가할 수 없는 머신 비전 알고리즘의 성능 개선 및 최적화 가능성을 시사합니다.



### PDC & DM-SFT: A Road for LLM SQL Bug-Fix Enhancing (https://arxiv.org/abs/2411.06767)
Comments:
          COLING-Industry 2025 accepted

- **What's New**: 이번 연구에서는 SQL의 버그 수정을 위한 새로운 방법론을 제시합니다. 기존의 Code LLM과 달리, 이 모델은 버그 수리 능력을 향상시키기 위해 Progressive Dataset Construction (PDC)와 Dynamic Mask Supervised Fine-tuning (DM-SFT) 방법을 도입합니다.

- **Technical Details**: PDC는 폭넓고 깊이 있는 두 가지 데이터 확장 방법을 포함하며, DM-SFT는 효과적인 수퍼바이즈드( Supervised) 학습 접근법을 통해 SQL 코드 버그 수정 과정에서의 학습 단계를 줄이고 안정성을 강화합니다. 특히, DM-SFT는 SQL 코드를 다루는 훈련 난이도를 줄이는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, PDC와 DM-SFT를 통해 훈련된 모델은 현재까지의 최고 모델들과 비교했을 때 성능이 50% 이상 향상되었으며, 일반적인 생성적 SFT에 비해 약 10%의 추가적인 성능 향상을 보여주었습니다.



### Research on an intelligent fault diagnosis method for nuclear power plants based on ETCN-SSA combined algorithm (https://arxiv.org/abs/2411.06765)
- **What's New**: 이 논문은 원자력 발전소(NPP)에서 효율적이고 정확한 결함 진단을 위한 새로운 지능형 결함 진단 방법을 제안합니다. 이 방법은 향상된 시간 합성곱 신경망(ETCN)과 참새 탐색 알고리즘(SSA)을 결합하여 성능을 최적화합니다.

- **Technical Details**: 제안된 ETCN은 시간 합성곱 신경망(TCN), 자기 주의(SA) 메커니즘, 그리고 잔여 블록(residual block)을 활용하여 성능을 향상시키며, SSA는 하이퍼파라미터(hyperparameters)를 적응적으로 최적화하여 성능을 개선합니다. 이론적으로는 지역적 특징을 잘 추출하고 시계열 정보를 효과적으로 캡처하는 데 강점을 보입니다.

- **Performance Highlights**: 제안한 방법은 CPR1000 시뮬레이션 데이터 세트에서 실험적으로 검증되었으며, 기존의 다른 고급 지능형 결함 진단 방법보다 모든 평가 지표에서 뛰어난 성능을 보였습니다. 이는 NPP의 지능형 결함 진단에 있어 유망한 도구가 될 것으로 기대됩니다.



### Dockformer: A transformer-based molecular docking paradigm for large-scale virtual screening (https://arxiv.org/abs/2411.06740)
Comments:
          14 pages, 5 figures

- **What's New**: 본 연구에서는 새로운 deep learning 기반 분자 도킹 방법인 Dockformer를 소개합니다. 이 방법은 다중 모달 정보(multimodal information)를 활용하여 분자의 기하학적 구조와 정보를 포착하고, end-to-end 방식으로 결합 형태(binding conformations)를 직접 생성할 수 있습니다.

- **Technical Details**: Dockformer는 두 개의 독립적인 인코더(encoders)를 사용하여 단백질과 리간드의 잠재 임베딩(latent embeddings)을 생성하고, 결합 모듈(binding module)을 통해 분자의 관계를 효과적으로 탐지합니다. 생성된 구조는 리간드 원자의 좌표를 end-to-end 방식으로 계산하며, 각각의 생성된 결합 형태에 대한 신뢰도(confidence measures)를 사용하여 결합 강도를 구분합니다.

- **Performance Highlights**: Dockformer는 PDBbind 코어 집합과 PoseBusters 벤치마크에서 각각 90.53%와 82.71%의 성공률을 기록하였으며, 추론 속도는 100배 이상 증가하여 거의 모든 최첨단 도킹 방법을 초월했습니다. 또한, 코로나바이러스의 주요 단백질 분해 효소 억제제를 식별하는 데 있어서도 뛰어난 성능을 보여줍니다.



### On the Principles of ReLU Networks with One Hidden Layer (https://arxiv.org/abs/2411.06728)
- **What's New**: 이번 논문은 단층 네트워크와二층 ReLU 네트워크의 함수 근사화 기법을 체계적으로 연구하여, 이러한 네트워크의 "블랙 박스" 문제를 해결하고자 하였습니다. 이를 통해 훈련된 솔루션을 이론적으로 이해하고 실험적으로 검증하였습니다.

- **Technical Details**: 네트워크 훈련 프로세스와 솔루션 공간을 명확히 이해하기 위해, 논문에서는 스플라인(spline)의 일면 기초(one-sided bases)와 같은 원리를 통해 고차원 입력에 대한 훈련 솔루션을 도출하는 몇 가지 새로운 원칙을 제안합니다. 이론적으로 제안된 방법은 다수의 엄격한 부분 순서(multiple strict partial orders)와 연속성 제한(continuity restriction)을 포함하여, 고차원 입력의 훈련 솔루션에 대한 일반적인 함수 근사화 능력을 증명합니다.

- **Performance Highlights**: 본 논문의 이론적 결과는 단순한 일차원 입력에 대한 깊은 이해를 이끌어낼 뿐만 아니라, 고차원 입력에서도 어느 정도 해석 가능함을 보여줍니다. 또한, 실험적 결과는 제안된 이론의 적합성을 증명하여, ReLU 네트워크의 블랙 박스를 풀 수 있는 가능성을 제시하고 있습니다.



### Script-Strategy Aligned Generation: Aligning LLMs with Expert-Crafted Dialogue Scripts and Therapeutic Strategies for Psychotherapy (https://arxiv.org/abs/2411.06723)
- **What's New**: 이 논문에서는 대화형 인공지능인 챗봇이 심리치료에 효과적으로 적용될 수 있도록 만드는 방법을 탐구했습니다. 특히, 대규모 언어 모델(LLM)과 전문가가 작성한 스크립트를 조화롭게 결합하여 성능을 향상시키는 새로운 접근법인 'Script-Strategy Aligned Generation (SSAG)'를 제안했습니다.

- **Technical Details**: SSAG는 LLM이 심리치료에 적합하게 조정되도록 하는 유연한 방법으로, '질문하기' 및 '반영 경청하기'와 같은 치료 전략을 사용하여 LLM의 대화 유연성 및 내용 적합성을 높입니다. 연구에서는 SAG와 SSAG 방식의 LLM-챗봇이 규칙 기반의 챗봇과 비교하여 우수한 성과를 나타냈습니다.

- **Performance Highlights**: 10일간의 현장 연구 결과, SSAG 방식은 규칙 기반 챗봇 대비 더 나은 성과를 보였으며, LLM의 유연성과 치료적 품질을 동시에 유지하는 효과적인 접근법으로 검증되었습니다.



### Synthesize, Partition, then Adapt: Eliciting Diverse Samples from Foundation Models (https://arxiv.org/abs/2411.06722)
- **What's New**: 본 논문에서는 사용자 경험을 향상시키고 다양한 선호를 수용하기 위해, 다양한 응답을 제공하도록 기초 모델의 출력을 다양화하는 새로운 프레임워크인 Synthesize-Partition-Adapt (SPA)를 제안합니다. 이 방법은 고품질의 다양한 응답을 생성하면서도 정확성을 희생하지 않는 것을 목표로 합니다.

- **Technical Details**: SPA는 널리 사용되는 합성 데이터(synthetic data)를 활용하여 기초 모델의 출력을 다각화하는 효율적인 접근 방식을 제공합니다. 이 프레임워크는 데이터 파티셔닝(data partitioning) 기법을 이용하여 합성 데이터를 서브셋으로 나누고, 각 서브셋에 대해 최적화된 모델 적응(model adaptation)을 훈련합니다. 이를 통해 다양한 응답을 생성할 수 있으며, 불필요한 품질 저하를 방지합니다.

- **Performance Highlights**: 저자는 HumanEval와 MBPP와 같은 코드 생성 작업 및 자연어 이해(natural language understanding) 작업에 대한 실험을 통해 SPA의 효과를 입증하였습니다. 이 실험 결과는 SPA가 높은 정확성을 유지하면서 모델의 응답 다양성을 증대시킬 수 있음을 보여줍니다. 이는 다양한 응용 프로그램에서 사용자 경험을 풍부하게 할 수 있는 잠재력을 강조합니다.



### DiffSR: Learning Radar Reflectivity Synthesis via Diffusion Model from Satellite Observations (https://arxiv.org/abs/2411.06714)
- **What's New**: 이 논문에서는 Weather radar 데이터의 합성을 위해 DiffSR이라 불리는 새로운 두 단계의 diffusion-based 방법을 제안합니다. 기존의 MSE (Mean Squared Error) 손실을 사용하는 재구성 방식에서 발생하는 과도한 평활화 문제를 해결하고자 합니다.

- **Technical Details**: DiffSR 방법은 첫 번째 단계에서 글로벌 데이터에 대한 재구성 모델을 사전 훈련(pre-training)하여 레이더 추정을 수행한 후, 두 번째 단계에서 레이더 추정 결과를 위성 데이터(Satellite data)와 결합하여 diffusion 모델의 조건으로 사용합니다.

- **Performance Highlights**: 다양한 실험 결과를 통해, 제안된 방법이 최신 기술(SOTA) 결과를 달성했음을 보여주며, 이는 고주파 세부사항 및 고값 관측 영역을 생성하는 능력을 입증합니다.



### High-Frequency Enhanced Hybrid Neural Representation for Video Compression (https://arxiv.org/abs/2411.06685)
- **What's New**: 본 논문은 기존의 Neural Representations for Videos (NeRV) 방식의 한계를 극복하기 위해, 고주파 세부정보를 향상시키는 Hybrid Neural Representation Network를 소개합니다.

- **Technical Details**: 우리는 고주파 정보를 활용하여 세부정보를 개선하기 위해 Wavelet Frequency Decomposer (WFD) 블록을 포함하는 wavelet 고주파 인코더를 설계하였습니다. 또한, HFM (High-Frequency Feature Modulation) 블록을 설계하여 추출된 고주파 임베딩을 사용하여 디코더의 적합 과정을 향상시킵니다. 마지막으로, Harmonic decoder 블록과 동적 가중치 주파수 손실(Dynamic Weighted Frequency Loss)을 통해 고주파 정보 손실을 최소화합니다.

- **Performance Highlights**: Bunny 및 UVG 데이터셋을 통한 실험에서 본 방법은 세부정보 보존 및 압축 성능에서 다른 방법들보다 유의미한 개선을 보였습니다.



### WDMoE: Wireless Distributed Mixture of Experts for Large Language Models (https://arxiv.org/abs/2411.06681)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)와 무선 네트워크의 결합을 통해 향상된 성능을 달성하기 위한 새로운 아키텍처인 Wireless Distributed Mixture of Experts (WDMoE)를 제안합니다. LLM의 MoE 레이어를 분해하여 기지국(Base Station)과 모바일 장치 간의 협업 배포를 가능하게 하여, 무선 네트워크에서 LLM의 효율성을 극대화합니다.

- **Technical Details**: WDMoE 아키텍처는 LLM의 MoE 레이어에서 게이팅 네트워크(gating network)와 선행 신경망(neural network) 레이어를 미리 기지국에 배치하고, 전문가 네트워크(expert networks)는 다양한 모바일 장치에 분산시킵니다. 이 방식은 모바일 장치의 병렬 추론(parallel inference) 능력을 활용하여 한정된 계산(computing) 및 캐싱(caching) 자원을 효과적으로 이용할 수 있습니다. 성능 측정 기준을 개발하고, 그 기준을 바탕으로 전문가 선택과 대역폭 할당을 최적화하여 지연(latency)을 최소화하며 정확도를 유지하는 데 중점을 두었습니다.

- **Performance Highlights**: 이론적인 시뮬레이션과 실제 하드웨어 실험에서 WDMoE 방법은 성능 저하 없이 지연을 상당히 줄일 수 있음을 입증하였습니다. NVIDIA Jetson 키트를 활용하여 구축한 하드웨어 테스트베드에서 성능을 검증하였으며, 무선 네트워크 환경에서의 실용적인 LLM 운영 가능성을 보여주었습니다.



### What Should Baby Models Read? Exploring Sample-Efficient Data Composition on Model Performanc (https://arxiv.org/abs/2411.06672)
Comments:
          8 pages, 6 figures, CoNLL 2024 (Shared Task) Accepted Paper

- **What's New**: 본 논문은 작은 언어 모델의 성능에 대한 선행 훈련 데이터 구성의 영향을 탐구합니다. 1000만 단어로 제한된 데이터셋을 사용하여 다양한 데이터셋 소스의 성능을 평가했습니다.

- **Technical Details**: 실험에 사용된 데이터셋은 아동 언어 지향 대화(CHILDES), 고전 문학(구텐베르크), 합성 데이터(타이니 스토리) 및 이들의 혼합(Mix)으로 구성되었습니다. 모델 범위는 18만 개에서 705만 개의 파라미터로 다양합니다.

- **Performance Highlights**: 작은 모델(GPT2-97M, GPT2-705M, Llama-360M)은 구텐베르크와 같은 복잡하고 풍부한 데이터셋에서 더 나은 성능을 보였으며, CHILDES와 타이니 스토리 데이터셋으로 훈련된 모델은 모든 모델 크기에서 저조한 성과를 보였습니다.



### An Efficient Memory Module for Graph Few-Shot Class-Incremental Learning (https://arxiv.org/abs/2411.06659)
Comments:
          16 pages, 6 figures, 38th Conference on Neural Information Processing Systems, 2024

- **What's New**: 본 논문에서는 그래프 표현 학습에서 발생하는 재앙적 망각(catasrophic forgetting) 문제를 해결하기 위한 새로운 방법인 Mecoin을 제안합니다. Mecoin은 제한된 라벨 수로도 효과적으로 그래프의 클래스를 학습할 수 있도록 설계되었습니다.

- **Technical Details**: Mecoin은 두 가지 주요 구성 요소, 즉 클래스 프로토타입을 학습하고 저장하는 Structured Memory Unit(SMU)와 GNN과의 동적 메모리 상호 작용을 위한 Memory Representation Adaptive Module(MRaM)으로 구성됩니다. SMU 내의 Memory Construction Module(MeCs)는 입력 노드의 특징과 프로토타입 표현 간의 상호 작용을 통해 샘플 표현을 업데이트하고, MRaM은 각 클래스 프로토타입에 대한 확률 분포를 저장하여 매개변수 조정을 통해 발생할 수 있는 지식 손실을 줄입니다.

- **Performance Highlights**: Mecoin은 기존의 관련 방법들과 비교하여 정확도(accuracy)와 망각률(forgetting rate)에서 우수한 성능을 보이며, GNN의 메모리를 잘 유지하면서도 효과적인 일반화 오류(generalization error)를 달성합니다. 이 방법은 메타 학습(meta-learning)으로 생성한 여러 샘플을 저장하는 기존 방법과는 달리, 메모리 소비를 최소화하며 고성능을 유지합니다.



### Renaissance: Investigating the Pretraining of Vision-Language Encoders (https://arxiv.org/abs/2411.06657)
- **What's New**: 최근 비전-언어 (Vision-Language) 과제를 위한 모델들이 급격히 증가하며, 이와 관련된 모델 디자인 및 훈련의 모범 사례에 대한 질문들이 여전히 남아있습니다. 본 논문에서는 비전-언어 인코더의 사전 훈련에 관한 질문에 답하고, 두 개의 주요 실험을 통해 가시적인 성능 저하 없이 계산 비용을 크게 절감할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 'Renaissance'라는 비전-언어 모델링 플랫폼을 소개하며, 대규모 컴퓨팅 자원을 절약하기 위해 사전 훈련 중 모델의 여러 부분을 고정(freezing)하는 실험을 수행하였습니다. 첫 번째 실험에서는 두 개의 모듈이 모두 고정된 경우 일부 성능 저하가 발생했지만, 시각 모듈을 고정시켰을 때는 성능이 증가하는 경향이 나타났습니다. 두 번째 실험에서는 텍스트 인코더와 비전 인코더의 사전 훈련 성능을 비교하였습니다.

- **Performance Highlights**: 결과적으로, 한 개의 타워 인코더 모델을 훈련할 때는 사전 훈련된 가중치보다 무작위 초기화(randomly initialized)가 더 나은 성능을 보이는 것으로 나타났습니다. Renaissance 플랫폼은 다양한 비전-언어 모델 유형을 평가하는 데 유용하며, 향후 연구에서 보다 많은 VL 모델에 대한 지원이 필요합니다.



### Explore the Reasoning Capability of LLMs in the Chess Testbed (https://arxiv.org/abs/2411.06655)
Comments:
          submitted to NAACL2025

- **What's New**: 이 연구에서는 체스의 전략적 및 전술적 요소를 통합하여 언어 모델의 추론 능력을 향상시키는 새로운 접근 방식을 제안합니다. MATE라는 100만 개의 체스 위치와 전문가의 주석을 포함하는 데이터셋을 수집했습니다.

- **Technical Details**: 체스 데이터셋 MATE는 각 위치에 대해 장기 전략 및 단기 전술이 주석이 달린 후보 수를 제공합니다. 연구진은 LLaMA-3-8B 모델을 파인튜닝하여 최신 상용 언어 모델들과 비교하였습니다.

- **Performance Highlights**: 모델은 전략과 전술을 모두 제공할 때 최신 상용 언어 모델들을 24.2% 초과하는 성능을 기록했습니다. 언어 설명이 언어 모델의 추론 능력을 향상시키는 데 도움이 된다고 발견했습니다.



### Understanding Scaling Laws with Statistical and Approximation Theory for Transformer Neural Networks on Intrinsically Low-dimensional Data (https://arxiv.org/abs/2411.06646)
- **What's New**: 이 논문은 트랜스포머 기반의 대형 언어 모델에서 모델 크기와 데이터 크기에 따라 일반화 오류(generalization error)가 어떻게 파워 스케일링 법칙을 따른다는 이론을 제시합니다. 특히, 이론은 훈련 데이터가 저차원 매니폴드(low-dimensional manifold)에 집중될 때 적용됩니다.

- **Technical Details**: 저자는 새로운 통계적 추정(statistical estimation) 및 수학적 근사(mathematical approximation) 이론을 개발하여 트랜스포머 신경망의 일반화 오류와 모델/데이터 크기 간의 스케일링 법칙을 예측하고 정당화했습니다. 이 연구에서 일반화 오류는 훈련 데이터 크기와 네트워크 크기에 대해 전통적인 방식으로 설명할 수 있으며, 모델은 저차원 데이터 구조를 활용하여 데이터 기하학(data geometry)을 존중하는 방법으로 스케일링 법칙을 설명합니다.

- **Performance Highlights**: LLM을 자연어 데이터 세트에서 훈련한 결과, 관측된 경험적 데이터 스케일링 법칙은 이론적 예측과 밀접한 일치를 보였습니다. 이 결과는 이론과 실제 모두에서 트랜스포머 스케일링 법칙에 영향을 미치는 데이터의 내재적 차원(intrinsic dimension)이 중요한 수량임을 엄격하게 보여줍니다.



### Exploring social bots: A feature-based approach to improve bot detection in social networks (https://arxiv.org/abs/2411.06626)
- **What's New**: 이 논문은 소셜 미디어에서의 허위 정보와 악성 링크의 확산 문제를 다루며, 자동화된 계정(봇)을 탐지하는 데 필요한 사용자 계정 프로필 및 콘텐츠 기반의 특징(features)을 조사합니다.

- **Technical Details**: 연구팀은 기계 학습 알고리즘(classical machine learning algorithms)을 사용하여 여러 메트릭(metric)에서 최신 기술(state of the art)을 초월하는 것을 목표로 하였으며, 특징 선택(feature selection) 및 추론(inference)을 통해 자동화된 계정 탐지에 유용한 특징들을 찾아냈습니다.

- **Performance Highlights**: 이 논문은 자동화된 계정에 대해 가장 중요한 특징들을 식별하며, 기존의 방법들을 뛰어넘는 성능을 보여주고 있습니다.



### vTune: Verifiable Fine-Tuning for LLMs Through Backdooring (https://arxiv.org/abs/2411.06611)
- **What's New**: 이 논문은 사용자 맞춤형 데이터셋에서 대형 언어 모델(LLM)의 미세 조정을 검증할 수 있는 새로운 방법인 vTune을 제안합니다. 이 방법은 훈련 데이터에 소량의 '출입구(backdoor)' 데이터 포인트를 추가하여 통계적 테스트를 통해 미세 조정이 제대로 이루어졌는지를 확인합니다.

- **Technical Details**: vTune은 LLM 미세 조정 기술의 최신 발전을 활용하여 훈련 데이터의 <1%의 수치를 수정하여 모델의 진위를 검증합니다. 이 방법은 사용자에게 몇 번의 추론 호출을 통해 높은 확률적인 정확성을 요구하며, 서비스 제공자에게는 약 1%의 추가 작업만을 필요로 합니다. vTune은 다양한 오픈소스 및 폐쇄형 LLM에 확장 가능하다는 장점이 있습니다.

- **Performance Highlights**: vTune 방법은 여러 모델 패밀리와 크기, 다양한 미세 조정 데이터셋에 걸쳐 테스트되었으며, p-값이 약 10^{-40}의 정도로 통계적 테스트가 만족되었습니다. 또한, vTune을 이용한 미세 조정이 하류 작업 성능에 부정적인 영향을 미치지 않음을 입증하였으며, 다양한 공격에 대한 견고성도 확인하였습니다.



### CriticAL: Critic Automation with Language Models (https://arxiv.org/abs/2411.06590)
- **What's New**: 이 논문에서는 CriticAL (Critic Automation with Language Models)을 제안하여 LLM(대형 언어 모델)의 활용을 통해 모델 비판(model criticism)을 자동화하는 새로운 접근 방식을 소개합니다. CriticAL은 모델 예측과 데이터 간의 불일치를 포착하는 summary statistics를 생성하고, 이들의 유의미성을 평가하는 가설 검정을 적용합니다.

- **Technical Details**: CriticAL은 LLM을 통해 모델과 데이터의 메타데이터를 기반으로 데이터의 성질을 포착하는 summary statistics를 생성하고, 이를 통해 모델의 가정이 위반되는지를 평가합니다. 이 통계량은 Python 함수로 구현되어 인간 또는 LLM 과학자가 쉽게 실행할 수 있도록 되어 있어 투명성과 신뢰성을 제공합니다. CriticAL의 summary statistics는 전통적인 가설 검정을 통해 불일치의 유의미성을 평가하여 모델을 자동으로 검증할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, CriticAL은 인위적으로 제어된 불일치에서 신뢰성 있는 비판을 생성하며, 인간 및 LLM 심사자 모두 CriticAL의 비판을 다른 접근 방법보다 투명성과 실행 가능성 측면에서 더 선호하는 것으로 나타났습니다. CriticAL은 실제 데이터셋에서 인간이 설계한 모델보다 개선된 성과를 보였습니다.



### Enhancing frozen histological section images using permanent-section-guided deep learning with nuclei attention (https://arxiv.org/abs/2411.06583)
- **What's New**: 이 연구는 수술 중 신속 진단을 위해 사용되는 동결 절편(frozen section) 이미지의 질을 향상시키기 위한 생성적 딥러닝(generative deep learning) 접근 방식을 제시합니다. 동결 절편 이미지의 핵(nuclei) 영역에 초점을 맞추어 영구 절편(permanent section)에서 정보를 활용합니다.

- **Technical Details**: 제안된 방법은 세분화된 주의 네트워크(segmented attention network)를 통해 핵이 세분화된 이미지를 사용하여 훈련하고, 생성된 영구 이미지를 개선하는 추가 손실 함수(loss function)를 추가합니다. 이는 인위적인 데이터 생성을 방지하고, 블랭크(blank) 지역에서 비신뢰할 수 있는 정보를 도입하지 않도록 합니다.

- **Performance Highlights**: 이 방법은 신장(kidney), 유방(breast), 대장(colon)을 포함한 다양한 조직에서 검증되었으며, 동결 절편 이미지의 효율성을 크게 높이고 진단 정확성을 개선합니다. 이미지를 몇 초 안에 향상시켜 기존 실험실 워크플로우와 원활하게 통합될 수 있습니다.



### Federated LLMs Fine-tuned with Adaptive Importance-Aware LoRA (https://arxiv.org/abs/2411.06581)
- **What's New**: 이 논문은 이질적인 클라이언트 자원에 적합한 연합형 저랭크 적응(Adaptive Federated Low-Rank Adaptation) 프레임워크를 제안합니다. 이를 통해 데이터 프라이버시를 유지하면서도 대규모 언어 모델(LLM)의 태스크 맞춤형 적응을 가능하게 합니다.

- **Technical Details**: 제안된 HAFL(이질적 적응 연합 LoRA) 프레임워크는 클라이언트의 LoRA 랭크에 따라 중요도 기반 파라미터 절단 및 동결 기법을 도입하여 자원의 이질성을 고려합니다. 클라이언트들은 선택적으로 가장 중요한 LoRA rank-1 매트릭스만 업데이트하며, 나머지 매트릭스는 동결됩니다. 또한, 어댑티브 집계 방식이 도입되어 정보 희석을 방지합니다.

- **Performance Highlights**: 20뉴스 그룹 분류 작업에서 HAFL 방법이 낮은 통신 비용으로 빠르게 수렴하며, 협력 모델 분배 시 성능 저하를 방지할 수 있다는 결과를 보였습니다. 또한, 제안된 동적 집계 방법은 제로 패딩 접근법에 비해 더 빠른 수렴 속도를 기록하였습니다.



### Discovering emergent connections in quantum physics research via dynamic word embeddings (https://arxiv.org/abs/2411.06577)
Comments:
          7 pages; 4 figures; 1 table; Appendix: 2 pages, 2 figures

- **What's New**: 이 논문에서는 양자 물리학 연구의 개념 조합 예측을 위한 새로운 접근 방식으로 동적 단어 임베딩(dynamic word embeddings)을 도입합니다. 기존 지식 그래프(knowledge graphs)와는 달리, 제안된 방법은 개념 간의 암묵적인 관계를 포착하여 더 폭넓은 정보를 인코딩할 수 있습니다.

- **Technical Details**: 연구에서는 arXiv quant-ph 카테고리에서 얻은 66,839개의 초록을 기반으로 하여 10,235개의 고유한 양자 물리학 개념을 정의하고 이를 동적 임베딩으로 모델링합니다. Word2Vec의 Skip-gram 모델을 사용하여 개념의 의미적 관계와 시간이 지남에 따라 변화하는 관계를 학습합니다.

- **Performance Highlights**: 제안된 방법은 연구 초록 내 개념의 동시 출현 예측에서 기존 방법들보다 뛰어난 성능을 보이며, 이는 과학 문헌에서 개념적 관계를 모델링하는 더 유연하고 유익한 방법임을 시사합니다.



### Learning Loss Landscapes in Preference Optimization (https://arxiv.org/abs/2411.06568)
- **What's New**: 본 연구는 Preference Optimization (PO) 알고리즘의 성능에 영향을 미치는 데이터의 특성을 분석하였으며, 특히 노이즈가 포함된 데이터나 품질이 혼합된 데이터에서 발생하는 성능 저하 문제를 해결하기 위한 새로운 PO 프레임워크를 제안합니다.

- **Technical Details**: 우리는 mirror descent 기반의 새로운 PO 프레임워크를 도입하여 Direct Preference Optimization (DPO)와 Odds-Ratio Preference Optimization (ORPO) 방법을 특정 미러 맵의 선택에 따라 회복할 수 있음을 보여줍니다. 이 프레임워크 내에서, 진화적 전략(evolutionary strategies)을 사용하여 문제 상황을 다룰 수 있는 새로운 손실 함수(loss functions)를 발견하였습니다.

- **Performance Highlights**: 우리의 접근 방식으로 발견한 손실 함수는 여러 작업에서 DPO 및 ORPO에 비해 상당한 성능 개선을 이끌어냈습니다. 특히, 혼합 품질 데이터를 사용하여 대규모 언어 모델을 세밀히 조정하는 경우 ORPO보다 우수한 성능을 보여주었습니다.



### Foundation Model for Composite Materials and Microstructural Analysis (https://arxiv.org/abs/2411.06565)
- **What's New**: 이 연구에서는 복합재료(composite materials) 전용으로 설계된 기반 모델(foundational model)을 제안합니다. 이 모델은 짧은 섬유 복합재료 데이터셋에서 사전 훈련(pre-trained)되어 강력한 잠재 특성을 학습합니다. 이 연구는 기존의 데이터 부족 문제를 해결할 수 있는 성능을 보여주고 있습니다.

- **Technical Details**: 제안된 모델은 MMAE(Material Masked Autoencoder)라는 이름의 기반 모델로, 100,000개의 그레이스케일 이미지로 구성된 데이터셋에 대해 self-supervised learning을 통해 사전 훈련되었습니다. 전이 학습(transfer learning)을 통해, MMAE는 혼합 강도를 예측하는 데 있어 R2 점수가 0.959에 도달하며, 훈련 데이터가 제한적임에도 불구하고 성능이 0.91을 지속적으로 초과합니다.

- **Performance Highlights**: MMAE는 복합재료의 마이크로 구조를 효과적으로 재구성(reconstruction)하며, 눈에 보이지 않는 데이터에 대해서도 높은 일반화 능력(generalization)으로 성능을 발휘합니다. 사전 훈련에서 높은 마스킹 비율(masking ratio)이 더 효과적인 특징 학습(feature learning)을 가능하게 한다는 사실이 입증되었습니다. 이 모델은 비용 효율적인 재료 설계 및 분석을 위한 새로운 길을 열어줄 것으로 기대됩니다.



### Is Linear Feedback on Smoothed Dynamics Sufficient for Stabilizing Contact-Rich Plans? (https://arxiv.org/abs/2411.06542)
Comments:
          Under review for ICRA2025

- **What's New**: 이번 연구는 contact smoothing을 활용하여 비접촉 제어에서의 선형 제어기(LQR) 합성을 분석합니다. 연구자들은 로봇의 이중 수조작에서 300개 이상의 궤적을 통해 이 접근법의 성능을 광범위하게 평가하였습니다.

- **Technical Details**: 이 논문에서는 contact smoothing을 통해 비선형 시스템을 보다 매끄러운 시스템으로 근사하여 선형 제어기 합성을 가능하게 하는 방법을 제안합니다. 이를 통해 생성된 계획과 피드백 이득을 이용하여 unstable한 동작을 안정화하고자 하며, quasi-dynamic 모델을 사용하여 시스템 상태를 정의합니다.

- **Performance Highlights**: 연구 결과, contact smoothing이 계획 수립에 효과적이지만, LQR은 contact-rich 계획의 안정화에 있어 불만족스러운 결과를 보였습니다. 접촉의 일방향성과 control기가 너무 강하게 작용하는 경향이 주요 원인으로 밝혀졌습니다.



### Epistemic Integrity in Large Language Models (https://arxiv.org/abs/2411.06528)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)이 생성하는 응답의 신뢰성 문제를 다루며, 언어적 단정성(linguistic assertiveness)와 내부 확신(internal certainty) 간의 불일치를 조명합니다. 새로운 인간 라벨 데이터셋과 정확도를 50% 이상 개선하는 방법을 도입하였습니다.

- **Technical Details**: 제안된 새로운 방법론은 LLM의 언어적 단정성을 측정하여 내부 확신과 외부 표현 간의 불일치를 확인합니다. 이는 여러 데이터셋에서 검증되었으며, LLM이 주장하는 확신과 실제 정확도 간의 불일치가 심각함을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM은 낮은 내부 확신을 가진 경우에도 과도하게 단정적인 응답을 생성하였으며, 이는 사용자에게 잘못된 신뢰를 줄 수 있습니다. 이 연구는 LLM의 불일치를 측정하는 가장 포괄적인 증거를 제공하며, 인간의 인식과 LLM의 단정성 점수를 비교하여 그 상관관계를 분석합니다.



### I2VControl-Camera: Precise Video Camera Control with Adjustable Motion Strength (https://arxiv.org/abs/2411.06525)
- **What's New**: 이번 연구에서는 I2VControl-Camera라는 새로운 카메라 제어 방법을 제안하여 동영상 생성의 제어 정밀도를 크게 향상시키고, 주제의 동작 강도에 대한 조정 가능성을 제공했습니다.

- **Technical Details**: I2VControl-Camera는 카메라 좌표계에서의 점 궤적(point trajectory)을 제어 신호로 사용하며, 고차원 동작 표현을 모델링하여 주제 동작의 강도를 염두에 두고 개발되었습니다. 적응형 아키텍처를 사용하여 기본 모델 구조에 구애받지 않도록 설계되었습니다.

- **Performance Highlights**: 정적 및 동적 장면에서 기존 방법보다 정량적 및 정성적으로 우수한 성능을 입증하였으며, 주제가의 동적 효과와 카메라 제어를 효과적으로 조화시켜 향상된 비디오 품질을 보여주었습니다.



### Offline Handwritten Signature Verification Using a Stream-Based Approach (https://arxiv.org/abs/2411.06510)
Comments:
          Accepted for oral presentation at the International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문에서는 손글씨 서명 검증(HSV) 시스템의 새로운 접근법을 제안합니다. 기존의 정적인 배치 구성 대신, 동적인 데이터 스트림 환경에서 작동할 수 있는 적응형 시스템을 개발하여 서명 검증의 성능을 향상시킵니다.

- **Technical Details**: 제안된 시스템은 SigNet-S 모델을 사용하여 서명 이미지를 특징 벡터로 변환하고, 이를 기반으로 디체미니당 변환(Dichotomy Transformation, DT)을 통해 이진 분류 문제로 변환합니다. 이 시스템은 새로운 서명 샘플이 입력될 때마다 업데이트되고, 과거의 데이터와 결합하여 계속해서 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Support Vector Machine(SVM)을 이용한 표준 접근 방식과 비교했을 때 우수한 성능을 보였습니다. GPDS Synthetic, CEDAR, MCYT 데이터셋에서의 실험 결과는 이 접근법의 효과성을 입증합니다.



### Understanding the Role of Equivariance in Self-supervised Learning (https://arxiv.org/abs/2411.06508)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문에서는 전통적인 Self-Supervised Learning(SSL) 방법들이 유용한 특징(예: 색상)의 손실을 초래하고 있다는 점을 지적하며, 이를 보완하기 위한 Equivariant Self-Supervised Learning (E-SSL) 접근법에 대해 심도 있는 논의를 통해 본질적 작동 메커니즘을 이해하려고 합니다.

- **Technical Details**: E-SSL은 입력 데이터의 변형에 민감한 특징을 학습하는 방법으로, 특히 이미지 클래스와 변환 사이의 상호작용을 수학적인 관점에서 탐구합니다. 'Explaining-away' 효과를 활용하여 E-SSL의 일반화 능력을 정보 이론적 관점에서 분석하며, 데이터 변환의 영향을 정량적으로 분석하여 E-SSL 설계를 위한 세 가지 원칙(손실 변환, 클래스 관련성, 단축 가지치기)을 제시합니다.

- **Performance Highlights**: 이론적 발견을 바탕으로 최근 연구된 다양한 E-SSL 방법들을 재검토한 결과, 다수의 사례가 제안된 프레임워크 내에서 잘 설명될 수 있음을 보여줍니다. 이 연구는 E-SSL의 설계 및 이해에 대한 중요한 방향을 제시하며, 미래 연구에서 E-SSL의 가능성을 확장하는 데 기여할 것입니다.



### LProtector: An LLM-driven Vulnerability Detection System (https://arxiv.org/abs/2411.06493)
Comments:
          5 pages, 4 figures. This is a preprint version of the article. The final version will be published in the proceedings of the IEEE conference

- **What's New**: 이번 논문에서는 LProtector라는 자동화된 취약점 탐지 시스템이 소개되었습니다. 이는 대형 언어 모델인 GPT-4o와 Retrieval-Augmented Generation(RAG)을 기반으로 하여 C/C++ 코드베이스 내의 취약점을 효과적으로 탐지하고자 합니다.

- **Technical Details**: LProtector는 'Big-Vul' 데이터세트를 활용하여 훈련되었습니다. 이 시스템은 코드 조각을 이진 분류하여 취약점을 식별하며, RAG 방법론을 통합하여 정확성과 정밀도를 향상시킵니다. preprocessing 단계에서는 Pandas를 사용해 CWE-ID, 코드 설명, 취약점 이름 등을 추출하며, OpenAI Embedding Algorithm을 통해 임베딩을 생성합니다.

- **Performance Highlights**: LProtector는 F1 점수 측면에서 기존의 두 가지 최첨단 기준보다 우수한 성능을 보였으며, LLM과 취약점 탐지의 통합 가능성을 보여줍니다.



### RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration (https://arxiv.org/abs/2411.06463)
- **What's New**: 이번 논문에서는 RL-Pruner라는 새로운 구조적 프루닝 방법을 도입하였다. 이 방법은 강화 학습(Reinforcement Learning)을 기반으로 하여 가지치기 분포를 자동으로 학습하고, 다양한 CNN 구조에서 유연하게 적용될 수 있다.

- **Technical Details**: RL-Pruner는 포스트 트레이닝(post-training) 단계에서 작동하며, 각 프루닝 단계마다 정확한 가지치기 분포를 학습한다. Gaussian 노이즈를 이용하여 정책 분포를 업데이트하고, 각 단계에서 보상 함수를 사용하여 Q 값을 계산한다. 이 방식은 VGGNet, ResNet, GoogLeNet, MobileNet 등 다양한 네트워크에서 호환된다.

- **Performance Highlights**: 실험에서는 RL-Pruner가 CIFAR-100 데이터셋에서 VGG-19에 대해 60% 채널 희소성(sparsity)을 달성하고, GoogLeNet과 MobileNetV3-Large에 대해서는 40% 희소성을 달성하였다. 모든 경우에서 성능 저하는 1% 이하로 유지되었다.



### Prompt-Efficient Fine-Tuning for GPT-like Deep Models to Reduce Hallucination and to Improve Reproducibility in Scientific Text Generation Using Stochastic Optimisation Techniques (https://arxiv.org/abs/2411.06445)
Comments:
          73 pages, 6 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 과학 텍스트 생성을 위한 복잡한 작업에서의 제한사항을 극복하기 위해 고안된 새로운 파라미터 효율 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 접근법을 소개합니다. 이 접근법은 특히 질량 분석(mass spectrometry) 분야에서의 재현 가능성을 높이는 데 집중하고 있습니다.

- **Technical Details**: 모델은 오픈AI의 GPT-2를 기반으로 한 MS-GPT로 명명되었으며, 저자들은 질량 분석 문헌에 특화된 데이터셋을 사용하여 LoRA(Low-Rank Adaptation) 어댑터를 적용하여 미세 조정을 진행했습니다. BLEU, ROUGE, Perplexity 등 여러 평가 지표를 통해 MS-GPT 모델이 baseline GPT-2보다 더 높은 텍스트 일관성과 재현 가능성을 보였음을 통계 분석을 통해 검증했습니다.

- **Performance Highlights**: MS-GPT는 미세 조정 이전보다 BLEU 점수가 0.33에서 0.34로 증가하였고, ROUGE-1은 0.42에서 0.44로, ROUGE-L은 0.57에서 0.62로 향상되었습니다. Perplexity 점수는 13586.37에서 10092.12로 감소하고, 재현 가능성 점수는 0.83에서 0.84로 향상되었습니다. Perplexity의 개선은 5% 유의수준 하에서 통계적으로 유의미한 차이를 보였습니다.



### Local Implicit Wavelet Transformer for Arbitrary-Scale Super-Resolution (https://arxiv.org/abs/2411.06442)
Comments:
          Accepted by BMVC 2024

- **What's New**: 최근 발표된 연구에서, Local Implicit Wavelet Transformer (LIWT)라는 새로운 모델이 제안되었습니다. 이 모델은 이미지의 고주파 세부정보 복원을 개선하기 위해 Discrete Wavelet Transform (DWT)을 사용하여 특징을 분해하여 서로 다른 주파수 정보로 구성된 네 개의 서브밴드로 나누고, Wavelet Enhanced Residual Module (WERM)을 도입하여 고주파 선행 정보를 제공하도록 설계되었습니다.

- **Technical Details**: LIWT는 Wavelet Mutual Projected Fusion (WMPF)과 Wavelet-aware Implicit Attention (WIA)을 결합하여 고주파 정보를 효과적으로 활용하고, 복원 과정에서 주어진 좌표에 따라 주의 맵을 생성합니다. DWT를 통해 추출된 저주파 및 고주파 컴포넌트를 통합하여 고주파 세부정보를 복원합니다. 이러한 방법은 기존의 좌표 기반 앙상블 기법의 한계를 극복하고 지역적 관련성을 고려합니다.

- **Performance Highlights**: LIWT는 다양한 벤치마크 데이터 세트에서 뛰어난 성능을 보이며, 기존의 최신 기법들보다 더 나은 결과를 도출하는 것으로 나타났습니다. 정성적이며 정량적인 결과 모두 LIWT의 우수한 성능을 뒷받침합니다.



### PLM-Based Discrete Diffusion Language Models with Entropy-Adaptive Gibbs Sampling (https://arxiv.org/abs/2411.06438)
- **What's New**: 본 논문에서는 기존의 Pretrained Language Model (PLM)과 Discrete Diffusion Language Model (DDLM)을 통합하는 새로운 방법, Diffusion-EAGS를 제안합니다. 이 방법은 PLM을 활용해 데이터셋 기반 생성 작업의 성능을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: Diffusion-EAGS는 Mask Language Model (MLM)을 기반으로 PLM을 DDLM에 통합하는 방식으로, 각 단계에서의 denoising 기능을 MLM에 의해 수행하도록 설계되었습니다. 또한, 엔트로피 추적 모듈을 도입하여 diffusion 과정에서 denoising 적용 위치를 결정하는 데 도움을 줍니다. 이는 each denoising step을 제약된 Markov Random Field (cMRF)로 해석하여 adaptive Gibbs sampling을 활용하는 방법입니다.

- **Performance Highlights**: 실험 결과, Diffusion-EAGS는 기존의 다양한 데이터를 기반으로 하는 생성 작업에서 높은 텍스트 품질과 다양성을 달성하며, token-level 생성이 가능하여 여러 데이터셋 유도 생성 작업에서 적용 가능성을 보여줍니다. 추가로, 낮은 자원 환경 및 이중언어 설정에서도 잘 작동하는 것을 입증하였습니다.



### CTC-Assisted LLM-Based Contextual ASR (https://arxiv.org/abs/2411.06437)
Comments:
          SLT 2024

- **What's New**: 본 논문에서는 rare long-tail words(희귀 긴 꼬리 단어)를 인식하는 데 중점을 둔 CTC-Assisted LLM-Based Contextual ASR 모델을 제안합니다. 이 모델은 효율적인 필터링 알고리즘을 통해 기존 LLM 기반 ASR 모델의 한계를 극복합니다.

- **Technical Details**: CTC(Continuous Time Classification) 디코더와 fine-tuned SSL(선택적 자기 감시) 모델을 결합하여, 관련 hotword(핫워드)를 필터링하고 LLM(prompt input) 입력에 통합하는 방법을 사용합니다. WI-BER(Word Error Rate / Biased Word Error Rate) 측면에서 test-clean 세트에서 1.27% / 3.67%의 성능을 보이며, test-other 세트에서는 2.72% / 8.02%를 달성했습니다.

- **Performance Highlights**: 제안된 모델은 baseline LLM-ASR 모델과 비교하여 각각 39.81% / 63.37%, 35.24% / 61.37%의 성능 향상을 보여주며, 2000개의 biasing words(바이어스 단어)의 도움으로도 뛰어난 성능을 유지합니다.



### Neuro-Symbolic Rule Lists (https://arxiv.org/abs/2411.06428)
- **What's New**: 이 논문에서는 Healthcare와 같은 민감한 분야에서 사용할 수 있도록 해석 가능한 머신러닝 모델을 위한 새로운 접근 방식을 제안합니다. NeuRules는 디스크리티제이션(discretization), 규칙 학습(rule learning), 규칙 순서(rule order)를 통합한 완전한 차별화 가능한 프레임워크입니다.

- **Technical Details**: NeuRules의 핵심은 규칙 리스트 학습 문제에 대한 연속 완화(continuous relaxation)를 수립하고, 온도 풀림(temperature annealing)을 통해 엄격한 규칙 리스트로 수렴하는 것입니다. 또한, NeuRules는 개별 특징의 디스크리티제이션(discretization) 및 그 조합을 conjunctive 규칙으로 학습하며 사전 처리(pre-processing)나 제한 없이 동작합니다.

- **Performance Highlights**: 광범위한 데이터셋에서 NeuRules는 기존의 조합 최적화(combinatorial optimization) 및 신경-상징(neuro-symbolic) 방법들을 일관되게 초월하며, 간단한 규칙과 복잡한 규칙, 그리고 그 순서를 효과적으로 학습합니다.



### SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains (https://arxiv.org/abs/2411.06426)
- **What's New**: 본 연구에서는 SequentialBreak라는 새로운 jailbreak 공격 기법을 소개합니다. 이 기법은 LLM이 단일 쿼리 내에서 여러 프롬프트를 처리할 때 특정 유해한 프롬프트에 집중하도록 유도하는 구조를 활용합니다. 다양한 시나리오를 통해 유익한 프롬프트들 사이에 유해한 프롬프트를 은폐하여 LLM이 유해한 반응을 생성하도록 유도할 수 있습니다.

- **Technical Details**: SequentialBreak 공격은 단일 쿼리로 여러 프롬프트를 전송하며, 공격자는 benign 프롬프트 사이에 있던 유해한 프롬프트를 포함합니다. 이 공격은 black-box 접근만 필요하며, 다양한 프롬프트 서사 구조에 적응 가능합니다. 연구에서는 'Question Bank', 'Dialog Completion', 'Game Environment'라는 세 가지 공격 시나리오를 제시하며, 이 모두에서 고도의 공격 성공률을 보고합니다.

- **Performance Highlights**: SequentialBreak는 기존의 다양한 jailbreak 기법들보다 우수한 성능을 나타내었으며, 단일 쿼리만으로 여러 최신 LLM 모델에서 높은 공격 성공률을 달성했습니다. 또한, 기존 방어 메커니즘을 효과적으로 피할 수 있는 능력을 입증하며, 자원 효율성 또한 높습니다.



### Generating Mixcode Popular Songs with Artificial Intelligence: Concepts, Plans, and Speculations (https://arxiv.org/abs/2411.06420)
Comments:
          Link to the paper:this https URL Published in The International Conference on AI and Musical Creativity at the University of Oxford (2024) this https URL

- **What's New**: 이 논문은 인공지능(Artificial Intelligence)과 대중 음악(Popular Music)의 통합을 다루며, 사회 변혁(Social Transformation), 교육(Education), 건강 관리(Healthcare), 감정적 웰빙(Emotional Well-Being)을 위한 강력한 도구를 만들기 위한 프로젝트를 제안합니다.

- **Technical Details**: 이 연구는 컴퓨터 과학자(Computer Scientist), 데이터 분석가(Data Analyst)와 민속 음악학자(Ethnomusicologist), 사회 인류학자(Social Anthropologist) 간의 협업의 출발점에서 제시되고 있으며, 주로 개념적(Conceptual)이고 다소 투기적(Speculative)입니다.

- **Performance Highlights**: 이 프로젝트의 결과는 음악이 사회적 목적을 위해 어떻게 효과적으로 활용될 수 있는지를 탐구하며, 사회, 정치, 경제적 맥락에서 음악의 역할을 재조명할 수 있는 기회를 제공합니다.



### Automated Strategy Invention for Confluence of Term Rewrite Systems (https://arxiv.org/abs/2411.06409)
- **What's New**: 본 논문에서는 머신러닝을 활용하여 자동 융합 증명기(automatic confluence prover) CSI의 전략 발명을 다룹니다. 이는 기존의 인적 설계 전략을 초월하는 성능을 발휘합니다.

- **Technical Details**: 자동 융합 증명기에서 사용하는 다양한 용어 수정 시스템(term rewriting systems, TRS)에 대한 전략을 자동으로 생성합니다. 이 접근 방식은 넓은 파라미터 공간에서 효과적인 전략을 찾기 위해 머신러닝 기술을 적용합니다.

- **Performance Highlights**: 실험 결과, 논문에서 발명한 전략은 CSI의 기본 전략을 Cops 데이터셋 및 증강 데이터셋 모두에서 초과 달성하며, CoCo 경연 역사상 자동 융합 증명기가 증명하지 못했던 여러 TRS에 대한 (비)융합을 증명 또는 반증하였습니다.



### Fineweb-Edu-Ar: Machine-translated Corpus to Support Arabic Small Language Models (https://arxiv.org/abs/2411.06402)
- **What's New**: 이번 연구에서는 HuggingFace에서 제공하는 FineWeb-Edu 데이터셋을 기계 번역하는 과정을 통해 아랍어로 번역된 FineWeb-Edu-Ar 데이터셋을 공개합니다. 이는 공개적으로 이용 가능한 가장 큰 아랍어 기계 번역 데이터셋입니다.

- **Technical Details**: FineWeb-Edu-Ar은 2020B 토큰의 범위를 가지며, 아랍어 전용 토크나이저에 최적화되어 있습니다. 번역 작업에서 nllb-200-distilled-600M 모델이 가장 우수한 성능을 보였으며, 다양한 기계 번역 모델의 성능 평가에 대한 자세한 분석이 포함되어 있습니다.

- **Performance Highlights**: 번역 품질은 LLM-as-a-Judge 접근 방식을 사용하여 평가되었으며, 24점 만점에 대한 평균 점수를 통해 다양한 MT 모델의 성능을 비교했습니다. nllb-200-distilled-600M 모델이 계산 예산에 가장 적합한 모델로 선정되었습니다.



### A Variance Minimization Approach to Temporal-Difference Learning (https://arxiv.org/abs/2411.06396)
- **What's New**: 이 논문은 전통적인 RL 알고리즘 대신 오류(minimization) 최소화를 위한 변동성 최소화(Variance Minimization, VM) 접근법을 도입하여 Bellman 오류(Bellman Error)의 변동성(Variance of Bellman Error, VBE)와 투영 Bellman 오류의 변동성(Variance of Projected Bellman Error, VPBE) 두 가지 목표를 제시합니다.

- **Technical Details**: 저자는 VMTD, VMTDC 및 VMETD 알고리즘을 도출하고, 변동성 최소화의 수렴성과 최적 정책 불변성(optimal policy invariance)의 증명을 제공합니다. 이 연구는 선형 근사화(value function approximation)에 기반한 시간 차 학습(Temporal Difference Learning) 알고리즘을 중점적으로 다룹니다.

- **Performance Highlights**: 실험 연구를 통해 제안된 알고리즘의 효과성을 검증하였으며, 새로운 방법론이 기존의 오류 최소화 접근법 대비 개선된 수렴 속도를 제공하는 것으로 나타났습니다.



### CausalStock: Deep End-to-end Causal Discovery for News-driven Stock Movement Prediction (https://arxiv.org/abs/2411.06391)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 업계 최초로 뉴스 기반의 다중 주식 움직임 예측을 위한 "CausalStock"이라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 주식 간의 시간적 인과 관계를 발견하며, 노이즈가 포함된 뉴스 데이터에서 유용한 정보를 효과적으로 추출합니다.

- **Technical Details**: CausalStock 모델은 라그 의존성(lag-dependent) 템포럴 인과 발견 메커니즘을 적용하여 주식 간의 인과 그래프 분포를 모델링합니다. 또한, 대형 언어 모델(LLMs)의 텍스트 평가 능력을 활용한 Denoised News Encoder를 통해 노이즈가 많은 뉴스 텍스트에서 유용한 정보를 추출합니다. Functional Causal Model(FCM)을 사용하여 발견된 인과 관계를 캡슐화하고 주식의 움직임을 예측합니다.

- **Performance Highlights**: CausalStock은 미국, 중국, 일본, 영국 시장에서 수집한 6개의 실제 데이터세트에서 뉴스 기반의 다중 주식 움직임 예측 및 다중 주식 움직임 예측 작업 모두에서 강력한 기준선(baseline)을 초과하는 성능을 보였습니다. 인과 관계를 활용하여 CausalStock은 명확한 예측 메커니즘과 뛰어난 설명 가능성을 제공합니다.



### Self-Training Meets Consistency: Improving LLMs' Reasoning With Consistency-Driven Rationale Evaluation (https://arxiv.org/abs/2411.06387)
Comments:
          under review

- **What's New**: 본 연구에서는 CREST(Consistency-driven Rationale Evaluation for Self-Training)라는 새로운 자기 훈련 프레임워크를 제안합니다. 이 프레임워크는 각 논리를 후속 질문을 통해 평가하고 이를 활용하여 모델을 훈련하는 방법을 포함하고 있습니다.

- **Technical Details**: CREST는 두 가지 방법을 사용하여 훈련을 수행합니다: (1) 후속 질문에서 자주 잘못된 정답을 도출하는 논리를 걸러내는 rational filtering과 (2) 원본 및 후속 질문의 평가 결과를 기반으로 혼합된 선호도를 학습하는 preference learning입니다.

- **Performance Highlights**: 세 가지 질문-응답 데이터셋에서 실험을 수행한 결과, CREST는 논리의 강건성과 정확성을 향상시켰으며, 이전 자기 훈련 접근법보다 더 나은 추론 능력을 보여주었습니다.



### Phantom: Constraining Generative Artificial Intelligence Models for Practical Domain Specific Peripherals Trace Synthesizing (https://arxiv.org/abs/2411.06376)
- **What's New**: Phantom는 PCIe TLP 트레이스 생성을 생성적 AI 문제로 취급하면서 PCIe 고유의 제약 조건을 통합한 최초의 프레임워크입니다. 이 논문은 Phantom의 효과성을 실험적으로 증명하였습니다.

- **Technical Details**: Phantom은 TLP (Transaction Layer Packet) 작업과 RGB 삼중체의 매핑을 설계하고, TLP 생성을 이미지 생성 작업으로 재정의합니다. 이 과정에서, 트레이스 생성의 세 가지 단계인 정규화(normalization), 보정(calibration), 디코딩(decoding)으로 구성됩니다. 그리고 피험자 특성이 반영된 생성 패턴을 적용할 수 있는 방법론이 구현되었습니다.

- **Performance Highlights**: Phantom은 실제 PCIe 네트워크 인터페이스 카드에 대한 TLP 트레이스를 생성하며, 기존 모델보다 최대 1000배의 성능 향상과 Fréchet Inception Distance (FID)에서 최대 2.19배 개선된 결과를 보였습니다.



### BayesNAM: Leveraging Inconsistency for Reliable Explanations (https://arxiv.org/abs/2411.06367)
Comments:
          Under Review

- **What's New**: Neural additive model (NAM)의 불일치 현상을 연구하고 이를 설명할 수 있는 새로운 프레임워크, Bayesian Neural Additive Model (BayesNAM),을 제안합니다. BayesNAM은 Bayesian neural networks와 feature dropout을 결합하여 불일치 정보를 활용할 수 있도록 지원합니다.

- **Technical Details**: BayesNAM은 Bayesian neural networks을 기반으로 하여 feature dropout을 도입하여 NAM의 불일치성을 효과적으로 처리합니다. 기존 NAM과 동일한 데이터셋과 아키텍처를 사용하였음에도 불구하고, 랜덤 시드의 변화로 불일치하는 결과를 보이는 현상을 설명합니다. 이론적 분석을 통해 feature dropout이 불일치 정보를 포착하는 효과적인 방법임을 증명합니다.

- **Performance Highlights**: BayesNAM은 부족한 데이터 또는 모델의 구조적 한계를 비롯한 잠재적인 문제를 효과적으로 식별하고, 더 신뢰할 수 있는 설명을 제공합니다. 실험을 통해 BayesNAM이 NAM보다 더욱 해석 가능하고 믿을 수 있는 결과를 도출함을 보여줍니다.



### Layer-Wise Feature Metric of Semantic-Pixel Matching for Few-Shot Learning (https://arxiv.org/abs/2411.06363)
- **What's New**: 본 연구에서는 이미지 쌍의 유사성을 보다 정확하게 평가하기 위해 새로운 방법인 Layer-Wise Features Metric of Semantic-Pixel Matching (LWFM-SPM)을 제안합니다. 이 방법은 향상된 성능을 위해 Layer-Wise Embedding (LWE) 모듈과 Semantic-Pixel Matching (SPM) 모듈을 포함하고 있습니다.

- **Technical Details**: LWE 모듈은 서로 다른 이미지 쌍의 레이어별 상관관계를 정교하게 추출하고, SPM 모듈은 의미론적 임베딩을 기반으로 한 중요한 픽셀 정렬 과정을 통해 픽셀 간 유사성을 측정합니다. 최종적으로 이 모듈들을 통합하여 이미지 쌍을 입력으로 받아 유사성 점수를 출력하는 엔드-투-엔드 방식으로 설계됩니다.

- **Performance Highlights**: 다수의 상태-공식 분류 벤치마크(miniImageNet, tieredImageNet, CUB-200-2011, CIFAR-FS)에서 LWFM-SPM이 경쟁력 있는 성능을 달성함을 보였습니다. 이를 통해 기존의 메트릭 기반 방법들이 가진 한계점을 극복하고 유사성 평가의 정확성을 높이는 데 기여합니다.



### Deep Active Learning in the Open World (https://arxiv.org/abs/2411.06353)
- **What's New**: 이번 논문에서는 open-world 환경에서 새로운 OOD(Out-Of-Distribution) 클래스의 통합을 위한 ALOE라는 새로운 active learning 알고리즘을 소개합니다. ALOE는 두 가지 단계로 구성되어 있으며, 첫 번째 단계에서 다양성 샘플링(diversity sampling)을 통해 대표적인 예제를 선택하고, 두 번째 단계에서는 에너지 기반 OOD 탐지를 통해 알려지지 않은 클래스의 우선 순위를 정합니다.

- **Technical Details**: ALOE(Active Learning in Open-world Environments) 알고리즘은 두 단계의 접근 방식을 사용하여 open-world 환경에서 발생할 수 있는 문제를 해결합니다. 첫 번째 단계에서는 다양성 샘플링을 통해 다양한 데이터 분포를 포괄하는 대표 예제를 선택합니다. 두 번째 단계에서는 에너지 점수 기능을 사용하여 클러스터 내의 예제를 순위 매겨 주목해야 할 OOD 클래스를 우선적으로 파악합니다.

- **Performance Highlights**: ALOE는 ImageNet-LT와 같은 장기적 불균형 이미지 분류 데이터세트에서 실험을 수행했으며, 무작위 샘플링에 비해 동일한 정확도를 달성하는 데 70%의 주석 비용을 절감하였습니다. 모든 실험 설정에서 ALOE가 가장 우수한 성능을 보이며, 알려진 클래스의 성능 향상과 새로운 클래스 발견 사이의 중요한 트레이드오프를 발견하였습니다.



### Balancing Power and Ethics: A Framework for Addressing Human Rights Concerns in Military AI (https://arxiv.org/abs/2411.06336)
Comments:
          Accepted for oral (only 3 papers are selected!) Harms and Risks of AI in the Military Workshop (HRAIM 2024) at Mila Quebec (this https URL)

- **What's New**: 이번 논문은 군사 AI 디자인, 배포 및 사용에서 인권 문제를 평가하기 위한 새로운 3단계 프레임워크(Framework)를 제안합니다. 각 단계는 다양한 윤리적 및 법적 고려사항을 포함하여 AI 기술이 군사 작전에 미치는 영향을 균형 있게 다루고자 합니다.

- **Technical Details**: 이 프레임워크는 세 가지 주요 단계(Design, Deployment, Use)로 나뉘며, 각 단계는 알고리즘, 사후 사용의 책임, 공정성(fairness) 및 감시(surveillance)와 같은 여러 구성 요소를 포함합니다. 또한, 'Human-in-the-loop' 시스템(HITL)에 대한 인간 개입을 강조합니다.

- **Performance Highlights**: 논문에서는 군사 AI의 인권 침해 가능성과 이에 대한 해결 방안을 제시합니다. 또한, AI 시스템의 성능 모니터링과 지속적인 감사를 통해 공정성을 확보하고, 국제 인도법(International Humanitarian Law) 준수를 위한 강력한 윤리적 지침을 마련할 것을 강조합니다.



### Prompts Matter: Comparing ML/GAI Approaches for Generating Inductive Qualitative Coding Results (https://arxiv.org/abs/2411.06316)
Comments:
          Accepted by AERA 2025 Annual Meeting

- **What's New**: 이번 연구는 최근의 인공지능 발전, 특히 생성적 인공지능(Generative AI)을 활용하여 질적 코딩 과정에서의 기여를 탐구합니다.

- **Technical Details**: 연구에서는 두 가지 기존 접근법과 두 가지 이론 기반의 새로운 접근법을 온라인 커뮤니티 데이터셋에 적용하고, 생성된 코딩 결과를 평가했습니다.

- **Performance Highlights**: 결과적으로 ML/GAI 접근법 간에 상당한 차이가 나타났으며, 인간 코딩 과정을 GAI 프롬프트에 통합한 접근법의 우수성을 입증했습니다.



### NeuReg: Domain-invariant 3D Image Registration on Human and Mouse Brains (https://arxiv.org/abs/2411.06315)
Comments:
          15 pages, 5 figures, 5 tables

- **What's New**: 이번 연구에서는 NeuReg라는 Neuro-inspired 3D image registration 아키텍처를 제안합니다. 이 아키텍처는 domain invariance 기능을 가지고 있어 다양한 3D 뇌 이미징 모달리티 간의 변화를 잘 캡처할 수 있습니다.

- **Technical Details**: NeuReg는 Swin Transformer 블록을 encoder로 사용하여 domain-agnostic representations을 생성하며, 이를 통해 여러 뇌 이미징 도메인에 걸쳐 강력한 성능을 발휘합니다. 이 아키텍처는 mammalian visual system에서 영감을 얻어 개발되었습니다.

- **Performance Highlights**: NeuReg는 iSeg-2017 및 OASIS-3와 같은 다중 도메인 공개 데이터셋에서 기존의 baseline deep learning image registration 모델보다 우수한 성능을 보여줍니다. 특히, cross-domain 데이터셋에서 'source-only' 도메인에서 훈련된 모델이 'unseen' target 도메인에서 높은 성능을 기록했습니다.



### Optimal Driver Warning Generation in Dynamic Driving Environmen (https://arxiv.org/abs/2411.06306)
Comments:
          ICRA 2024

- **What's New**: 이 연구에서는 기존의 운전 경고 시스템의 한계를 극복하기 위해 운전자의 반응과 다른 차량들과의 상호작용을 고려한 최적 운전 경고 생성 문제를 설정합니다. 이 문제는 부분 관찰 마코프 결정 과정 (POMDP)으로 모델링되었습니다.

- **Technical Details**: 제안된 프레임워크는 운전자의 행동 추정을 통합하여 운전 경고를 생성하는 최적화된 솔루션을 제공합니다. 이 프레임워크는 다양한 운전 상황에 대한 예측 모델을 유연하게 포함할 수 있도록 설계되었습니다.

- **Performance Highlights**: 전체 폐쇄 루프 시뮬레이션 실험을 통해 제안된 방법의 우수성이 기존 경고 생성 방법보다 뛰어났음을 입증하였습니다.



### Analyzing the Evolution of Graphs and Texts (https://arxiv.org/abs/2411.06295)
Comments:
          PhD dissertation

- **What's New**: 이 논문은 동적 그래프와 텍스트의 변화를 효율적으로 모델링하기 위한 새로운 접근 방식을 제안합니다. 특히 동적 네트워크 임베딩을 위한 Personalized PageRank 알고리즘을 활용하여 네트워크 비정상 탐지 및 실체 의미 변화 발견에서 성능을 향상시킵니다.

- **Technical Details**: 논문에서는 Dynamic Personalized PageRank Embedding(DynamicPPE)과 DynAnom 프레임워크를 소개합니다. DynamicPPE는 동적 네트워크에서 특정 노드 집합을 효율적으로 학습하는 방법으로, 시간 복잡도가 선형적이며 고품질의 PPV(per-Personalized PageRank vector)를 활용하여 지역성과 전역 일관성을 확보합니다. DynAnom은 한정된 동적 가중 그래프에서 각 노드의 변화를 효율적으로 정량화하고 이상치를 추적하는 방법을 제안하여 2.3배 빠른 처리 속도를 자랑합니다.

- **Performance Highlights**: DynamicPPE는 COVID-19 팬데믹 동안 위키백과 그래프의 도시에 대한 임베딩 변화를 성공적으로 캡처하였고, DynAnom은 노드 수준 이상 탐지에서 가장 좋은 기준선 대비 0.5425의 정확도를 달성하며, 2.3배 더 빠른 실행 속도를 보였습니다.



### Multi-View Majority Vote Learning Algorithms: Direct Minimization of PAC-Bayesian Bounds (https://arxiv.org/abs/2411.06276)
- **What's New**: PAC-Bayesian 프레임워크를 다중 뷰 학습에 적용하여 Rényi divergence를 기반으로 하는 새로운 PAC-Bayesian 경계를 도입했습니다. 이로써 Kullback-Leibler divergence보다 더 정교한 복잡도 측정을 제공합니다.

- **Technical Details**: 첫째, 다중 뷰 학습을 위한 일반 PAC-Bayesian 경계를 확장했습니다. 둘째, Rényi divergence를 기반으로 한 1차 및 2차 오라클 PAC-Bayesian 경계를 제안했습니다. 셋째, 다중 뷰 학습을 위한 효율적인 최적화 알고리즘을 개발했으며, 자가 경계(self-bounding) 속성을 포함합니다.

- **Performance Highlights**: 이 연구는 다중 출처 데이터에서의 적용 가능성을 높이기 위해 기존의 PAC-Bayes 접근 방식을 개선하고, 새로운 이론적 기여를 바탕으로 실행 가능한 최적화 절차를 제시하여 실질적인 효과를 기대할 수 있습니다.



### Federated Split Learning for Human Activity Recognition with Differential Privacy (https://arxiv.org/abs/2411.06263)
Comments:
          Accepted to IEEE Consumer Communications and Networking Conference (CCNC), 6 pages

- **What's New**: 본 논문에서는 엣지 네트워크에서 차별적 프라이버시(Differential Privacy, DP)를 적용한 새로운 연합 분할 학습(Federated Split Learning, FSL) 기반의 인간 활동 인식(Human Activity Recognition, HAR) 프레임워크를 제안합니다. 이 프레임워크는 가속도계(accelerometer) 및 자이로스코프(gyroscope) 데이터를 활용하여 HAR 정확도를 크게 향상시킵니다.

- **Technical Details**: FSL-DP 프레임워크는 사용자 데이터 프라이버시를 보장하면서도, 클라이언트에서 모델의 일부를 훈련시킨 후 결과를 서버로 전송하여 훈련하는 구조를 갖추고 있습니다. 또한, 이 시스템에서는 ED(Edge Devices)가 클라이언트 모델의 전방 전파 및 후방 전파 일부를 수행하고, 서버는 나머지 모델 훈련을 통해 계산 부담을 줄입니다. DP 메커니즘은 정보가 공격에 노출될 가능성을 줄이기 위해 활성화된 데이터에 마스크를 추가합니다.

- **Performance Highlights**: 본 연구의 FSL 방법은 전통적인 학습 방법과 비교했을 때 훈련 성능을 크게 향상시키고, 훈련 지연 시간을 줄이는 데 성공했습니다. 시뮬레이션 결과는 DP 메커니즘이 HAR 성능에 미치는 영향을 다양한 훈련 설정에서 검증하였으며, FSL 프레임워크가 정확성 및 손실 메트릭에서 기존의 연합 학습 모델을 초월한 것으로 나타났습니다.



### Smart-LLaMA: Two-Stage Post-Training of Large Language Models for Smart Contract Vulnerability Detection and Explanation (https://arxiv.org/abs/2411.06221)
- **What's New**: 빠르게 발전하는 블록체인 기술과 함께 스마트 계약의 보안 문제는 중요한 과제가 되었습니다. 기존의 스마트 계약 취약점 탐지 방법은 데이터셋의 품질 부족, 대규모 언어 모델(LLMs)의 적응성 부족, 탐지된 취약점에 대한 명확한 설명 부족 등 세 가지 주요 문제에 직면해 있습니다. 이러한 문제를 해결하기 위해, Smart-LLaMA라는 새로운 탐지 방법이 제안되었습니다.

- **Technical Details**: Smart-LLaMA는 LLaMA 언어 모델을 기반으로 하며, 먼저 네 가지 취약점 유형에 대해 라벨, 자세한 설명 및 정확한 취약점 위치를 포함하는 종합 데이터셋을 구성합니다. 그 다음 스마트 계약 데이터로부터 배운 지속적인 사전 훈련(Smart Contract-Specific Continual Pre-Training)을 도입하여 모델이 스마트 계약의 구문 및 의미를 이해할 수 있도록 합니다. 마지막으로, 쌍으로 이루어진 취약한 코드와 설명을 사용하여 LLM을 미세 조정하는 설명 기반 미세 조정(Explanation-Guided Fine-Tuning)을 제안합니다.

- **Performance Highlights**: 실험 결과 Smart-LLaMA는 최신 기법들보다 우수한 성능을 보였으며, 평균 F1 점수는 6.49%, 정확도는 3.78% 향상되었습니다. 전문가 평가에서도 Smart-LLaMA는 정확성, 완전성 및 간결성 측면에서 상당히 높은 점수를 기록했습니다. 특히, 인간 평가에서 정확성, 완전성 및 간결성 각각의 지표에서 69.5%, 57.1%, 65.6%의 높은 점수를 달성했습니다.



### Multistage non-deterministic classification using secondary concept graphs and graph convolutional networks for high-level feature extraction (https://arxiv.org/abs/2411.06212)
Comments:
          13 Pages, 15 figures, and 4 Tables

- **What's New**: 본 논문은 다단계 비결정론적 분류 방법을 제안하며, 이는 보조 개념 그래프(secondary conceptual graph)와 그래프 컨볼루션 네트워크(Graph Convolutional Networks, GCN)를 기반으로 하고 있습니다.

- **Technical Details**: 제안된 방법은 1) GCN을 사용하여 12개의 고수준 특성을 추출 및 생성하고, 2) 불완전한 비결정론적 모델을 사용하여 예측을 수행하고, 3) 개념 그래프에 기반하여 최종 예측을 수행하는 여러 단계로 구성됩니다.

- **Performance Highlights**: Cora, Citeseer 및 PubMed의 세 가지 데이터셋에서 각각 96%, 93%, 95%의 분류 정확도를 달성하여 최신 방법보다 성능이 우수함을 입증하였습니다.



### IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization (https://arxiv.org/abs/2411.06208)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 복잡한 지시사항을 따르는 능력을 향상시키고 평가하기 위한 TRACE 벤치마크를 소개합니다. 이 벤치마크는 12만 개의 훈련 데이터와 1천 개의 평가 데이터로 구성되어 있습니다.

- **Technical Details**: TRACE는 5개의 제약 유형과 26개 제약 차원으로 분류된 복잡한 지시사항의 자동 작성 데이터를 기반으로 하며, ‘Input-Output Preference Optimization (IOPO)’라는 새로운 정렬 방법을 제안합니다. IOPO는 입력 지시사항과 출력 선호 쌍을 고려하여 LLM이 지시사항 선호를 효율적으로 탐색할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, IOPO를 사용한 경우 in-domain 데이터에서 각각 8.15%, 2.18%의 개선을 보였고, out-of-domain 데이터에서는 각각 6.29%, 3.13%의 성능 향상이 있었습니다. 이는 SFT 및 DPO 방법과 비교하여 평균적으로 7.22% 및 2.66%의 향상된 결과를 나타냅니다.



### Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework (https://arxiv.org/abs/2411.06160)
- **What's New**: 이 논문은 텍스트 감정 탐지(Text Emotion Detection)의 발전을 위해 Emotion Quantization Network (EQN) 프레임워크를 제안합니다. EQN은 감정 강도를 에너지 수준으로 매핑하여 미세 감정(micro-emotion)의 자동 탐지 및 주석(annotation)을 가능하게 합니다.

- **Technical Details**: EQN 프레임워크는 모든 라벨(all-labels) 및 훈련 세트 라벨 회귀(training-set label regression) 방법을 활용합니다. 이를 통해 기계 모델의 학습능력과 라벨 간의 상호 의존성을 최대한 활용하여 샘플 내 여러 감정을 발견합니다. 이론적으로 미세 감정의 측정과 주석 작성에 기여하는 새로운 접근법이 설명됩니다.

- **Performance Highlights**: EQN 프레임워크는 GoEmotions 데이터셋에서 감정 탐지 및 주석을 수행하며, Google 연구 결과와의 포괄적 비교를 통해 높은 자동 탐지 능력을 입증합니다. EQN은 에너지 레벨 점수로 미세 감정 자동 주석을 최초로 달성하였으며, 감정 탐지 분석 및 감정 컴퓨팅의 정량적 연구에 강한 지원을 제공합니다.



### Aquila-plus: Prompt-Driven Visual-Language Models for Pixel-Level Remote Sensing Image Understanding (https://arxiv.org/abs/2411.06142)
- **What's New**: 최근 비전 언어 모델(vision language models, VLMs)의 발전으로 시각-언어 통합(visual-language integration)이 크게 향상되었습니다. 특히 본 연구에서는 Aquila-plus라는 새로운 mask-text instruction tuning 방법을 제안하여 기존의 원거리 감지 비전 언어 모델(Remote Sensing Vision Language Models, RSVLMs)의 한계를 극복하고 픽셀 수준의 시각적 이해(pixel-level visual understanding)를 달성합니다.

- **Technical Details**: Aquila-plus는 세밀한 마스크 영역(mask regions)을 언어 지침에 포함시키는 방식으로 RSVLMs의 능력을 확장합니다. 연구팀은 10만 개의 샘플이 포함된 마스크 지역-텍스트 데이터셋을 정교하게 구축하였으며, 대규모 언어 모델(large language model, LLM)에 픽셀 수준의 표현을 주입하여 비전-언어 모델을 설계하였습니다. 구체적으로, Aquila-plus는 시각 인코더로 컨볼루션 CLIP(convolutional CLIP)를 사용하고, 고해상도 입력에서 정밀한 시각 마스크 특징을 추출하기 위해 마스크 인식 비주얼 추출기(mask-aware visual extractor)를 활용합니다.

- **Performance Highlights**: 실험 결과, Aquila-plus는 다양한 영역 이해(region understanding) 작업에서 기존 방법들보다 우수한 성능을 보여주었으며, 픽셀 수준의 지침 튜닝에서 새로운 가능성을 입증했습니다.



### Online Parallel Multi-Task Relationship Learning via Alternating Direction Method of Multipliers (https://arxiv.org/abs/2411.06135)
Comments:
          Accpeted by Neurocomputing

- **What's New**: 본 연구는 온라인 다중 작업 학습(Online Multi-task Learning, OMTL) 문제에 대해 중앙 서버를 활용하는 전통적인 방식 대신, 분산 컴퓨팅 환경에 적합하고 효율적인 수행을 위한 대안으로 교차 방향 배수기법(Alternating Direction Multiplier Method, ADMM)을 기반으로 한 새로운 OMTL 프레임워크를 제안합니다.

- **Technical Details**: 제안된 OMTL 알고리즘은 ADMM 최적기를 사용하여 여러 작업 간의 동적 관계를 모델링하며, 단순화된 하위 문제들을 병렬로 처리할 수 있는 구조를 가지고 있습니다. 이는 각 노드가 지역 이웃과 정보만을 교환하면서 작업을 수행할 수 있도록 하여, 중앙 서버의 병목 현상을 피하고 효율성을 증대시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 ADMM 기반 OMTL 알고리즘은 기존의 SGD 기반 접근법보다 정확성과 효율성에서 우수한 성능을 보였습니다. 실제 데이터셋과 합성 데이터셋을 통해 검증된 결과는 대규모 학습 문제에 대한 적용 가능성을 시사합니다.



### Research on reinforcement learning based warehouse robot navigation algorithm in complex warehouse layou (https://arxiv.org/abs/2411.06128)
- **What's New**: 본 논문에서는 복잡한 창고(warehouse) 레이아웃에서 최적 경로(optimal path)를 효율적으로 찾고 실시간(real-time) 의사 결정을 하는 방법을 제안합니다. Proximal Policy Optimization (PPO)과 Dijkstra 알고리즘을 결합한 Proximal policy-Dijkstra (PP-D) 방법을 도입하여, 효율적인 전략 학습과 실시간 의사 결정을 이루어냅니다.

- **Technical Details**: PP-D 방법은 PPO를 통해 동적(dynamic) 환경에서 로봇이 신속하게 액션 전략(action strategies)을 적응하고 최적화할 수 있도록 하며, Dijkstra 알고리즘을 이용해 정적(static) 환경에서의 전역(optimal) 경로 계획을 가능하게 합니다. PPO는 안정적인 정책(policy) 업데이트 메커니즘을 통해 로봇의 전략을 개선하고, Dijkstra 알고리즘은 정적 환경에서 글로벌 최적 경로 계획을 보장합니다.

- **Performance Highlights**: 비교 실험과 분석 결과, 제안된 PP-D 방법은 기존 알고리즘과 비교해 내비게이션(navigation) 예측의 정확도(accuracy)를 향상시키고 시스템의 견고함(robustness)을 크게 개선하는 중요한 장점을 가지고 있습니다. 특히 복잡한 창고 레이아웃에서 PP-D 방법은 최적 경로를 보다 정확하게 찾아내고 충돌(collision) 및 정체(stagnation)를 줄이는 데 효과적임을 입증하였습니다.



### Characteristics of Political Misinformation Over the Past Decad (https://arxiv.org/abs/2411.06122)
- **What's New**: 이 논문은 지난 12년간의 정치적 허위 정보의 공통된 특성을 탐구하여 , 이를 탐지하고 완화하기 위한 알고리즘 개발에 유용한 통찰을 제공하고자 한다.

- **Technical Details**: 연구자들은 자연어 처리(Natural Language Processing, NLP) 기법을 사용하여 PolitiFact의 검증된 데이터(2011-2023)를 분석하였으며, 데이터 수집 및 분류는 '정확한 정보(Accurate)', '허위 정보(Misinformation)', '혼합 정확도(Mixed-Accuracy)'의 세 가지 범주로 이루어졌다.

- **Performance Highlights**: 허위 정보는 최근 몇 년간 급격히 증가하였으며, 특히 Facebook 및 Instagram과 같은 텍스트 및 이미지 중심의 플랫폼에서 공유되는 경향이 있다. 또한, 허위 정보는 사실 정보보다 부정적인 감정을 더 많이 포함하고 있으며, 시간이 지남에 따라 정치적 진술의 전반적인 감정이 부정적인 경향을 보인다고 밝혔다.



### Energy-efficient Hybrid Model Predictive Trajectory Planning for Autonomous Electric Vehicles (https://arxiv.org/abs/2411.06111)
Comments:
          Accepted at the IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2024

- **What's New**: 이 논문은 전기차(EVs)의 제한된 배터리 수명과 긴 충전 시간이라는 두 가지 도전에 대응하기 위해 에너지 효율적인 하이브리드 모델 예측 계획기(EHMPP)를 제안합니다.

- **Technical Details**: EHMPP는 기존의 자동주행 알고리즘과 완벽하게 통합될 수 있도록 설계된 모션 플래너를 최적화하며, 추가 하드웨어 없이도 작동합니다. 이 시스템은 운동 에너지 회수 시스템(KERS), 엔진 효율성 및 외부 환경 간의 상호작용을 고려하여 최적화된 경로 계획 전략을 제공합니다. 시뮬레이션 실험은 Prescan, CarSim 및 Matlab에서 수행되었습니다.

- **Performance Highlights**: EHMPP는 차량의 에너지 효율성을 크게 향상시키며, 감속 단계에서 수동 회수 에너지를 11.74% 증가시키고, 모터 작동을 최적화하여 가속, 감속 및 순항 단계에서 이상적인 전력 상태를 유지할 수 있도록 합니다.



### Personalize to generalize: Towards a universal medical multi-modality generalization through personalization (https://arxiv.org/abs/2411.06106)
- **What's New**: 이번 논문은 개인 맞춤형 의학(Personalized Medicine)과 다중 모달 의료 이미지 분석(Multi-modal Medical Image Analysis) 간의 연결을 목표로 합니다. 특히, 개인 단위에서 생물학적 정보를 이용하여 개인화된 불변 표현(Personalized Invariant Representation) 을 도출하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 연구에서는 각 개인의 의료 이미지를 나타내는 표기법과 인코더(Encoder) 및 디코더(Decoder) 구조를 정의합니다. 제안된 방법은 개인화된 불변 표현 $oldsymbol{X}_h$ 을 학습하기 위해 생물학적 사전 지식(Previous Knowledge)을 이용하는 구조적 제약조건(Structural Constraints)을 포함합니다. 이를 통해 다중 모달 간의 일반화 가능성을 탐구합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 본 방법은 이질적 일반화(Heterogeneous Generalization) 및 동질적 일반화(Homogeneous Generalization) 등 다양한 시나리오에서 기존의 최신 기술(State-of-the-Art) 방법보다 뛰어난 성능을 보여주었습니다.



### LT-DARTS: An Architectural Approach to Enhance Deep Long-Tailed Learning (https://arxiv.org/abs/2411.06098)
- **What's New**: 이 논문에서는 Deep long-tailed recognition을 위한 새로운 접근법인 Long-Tailed Differential Architecture Search (LT-DARTS)를 제안합니다. 기존 DARTS 방법들이 long-tailed 데이터에서 좋은 성능을 내지 못하는 문제를 해결하기 위해 아키텍처 개선을 목표로 합니다.

- **Technical Details**: LT-DARTS는 long-tailed 데이터에 적합한 아키텍처를 탐색하기 위해 새로운 검색 공간(search space)과 두 가지 혁신적인 convolution 운영(Operation)을 시행합니다. Equiangular Tight Frame (ETF) 분류기를 도입하여 biased search process를 완화하고 성능 붕괴(performance collapse)를 방지합니다.

- **Performance Highlights**: 실험 결과, LT-DARTS는 기존의 전문가 설계 네트워크보다 뛰어난 성능을 발휘하며, long-tailed 문제에 최적화된 기존 방법들과도 원활하게 통합되어 성능을 지속적으로 개선합니다. 또한, 간단한 개선만으로도 state-of-the-art 결과를 달성함을 보여줍니다.



### Optimizing Large Language Models through Quantization: A Comparative Analysis of PTQ and QAT Techniques (https://arxiv.org/abs/2411.06084)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문은 Large Language Models (LLMs)의 최적화를 위한 양자화(Quantization) 기법들에 대한 포괄적인 분석을 제공합니다. 특히 Post-Training Quantization (PTQ)와 Quantization-Aware Training (QAT)에 중점을 두었습니다.

- **Technical Details**: 실험적으로 10M에서 1B 파라미터 범위의 모델에 대해 양자화를 적용한 결과, 모델 크기를 최대 68% 줄이면서도 전체 정밀도 기준선과의 성능 차이를 6% 이내로 유지할 수 있다는 것을 보여주었습니다. INT8 양자화는 연산 비용과 전력 소비를 40% 줄였고, INT4 양자화는 이러한 수치를 60% 더 개선했습니다. 혼합 정밀도 양자화를 위한 새로운 이론적 프레임워크를 도입하고, 레이어 민감도와 가중치 분산에 기반한 최적 비트 할당 전략을 도출했습니다.

- **Performance Highlights**: 엣지 장치에서의 하드웨어 효율성 평가 결과, INT8은 최대 2.4배의 처리량 향상을, INT4는 3배의 향상을 가능하게 하였으며, 전체 정밀도 모델에 비해 60% 전력 소비 감소를 보여주었습니다.



### Aquila: A Hierarchically Aligned Visual-Language Model for Enhanced Remote Sensing Image Comprehension (https://arxiv.org/abs/2411.06074)
- **What's New**: 최근, 대형 비전 언어 모델(VLMs)은 비주얼 인스트럭션 튜닝을 통해 시각적 언어 능력에서 중요한 발전을 이루었습니다. 그러나 기존의 원격 감지 비전 언어 모델(RSVLMs)은 복잡한 원격 감지 장면의 특성을 포착하는 데 한계가 있었습니다. 본 논문에서는 Aquila라는 진보된 비전 언어 기초 모델을 소개합니다.

- **Technical Details**: Aquila는 고해상도 이미지 입력을 지원하고 다중 스케일 시각적 특성을 집계하는 학습 가능한 계층적 공간 특성 통합(Hierarchical Spatial Feature Integration, SFI) 모듈을 도입합니다. 이 모듈은 복잡한 시각적 정보를 세밀하게 표현하는 기능을 제공합니다. 또한 SFI 모듈은 대형 언어 모델(LLM)의 여러 층에 통합되어 심층적인 비주얼-언어 특성 정렬(deep visual-language feature alignment)을 달성합니다.

- **Performance Highlights**: Aquila는 높은 해상도와 다중 스케일 입력을 통해 세부적인 시각적 효과를 포착하고, 특성 정렬을 강화하여 이미지 텍스트 데이터에서 학습 능력을 크게 향상시킵니다. 광범위한 정량적 실험과 질적 분석을 통해 Aquila의 우수한 성능을 검증하였습니다.



### GFT: Graph Foundation Model with Transferable Tree Vocabulary (https://arxiv.org/abs/2411.06070)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 Graph Foundation Models (GFMs)의 개발을 위한 새로운 접근법을 제안합니다. 기존의 그래프 모델들이 다양한 그래프 학습 작업에서 원하는 성능을 달성하지 못하는 상황에서, 연구자들은 계산 트리(computation tree)를 새로운 전이 가능한 패턴으로 제안하여 GFMs를 개선하고자 했습니다.

- **Technical Details**: GFT (Graph Foundation model with transferable Tree vocabulary)라는 이름의 새로운 GFMs 모델을 제안합니다. 이 모델은 메시지 전달 과정에서 파생된 계산 트리 구조를 사용하여 그래프 데이터를 처리하는데, 이를 통해 모델의 일반화 능력을 향상시키고 부정 전이(negative transfer)의 위험을 줄입니다. GFT는 데이터셋의 다양한 작업과 도메인을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, GFT가 다양한 작업 및 도메인에 걸쳐 그래프 학습의 효과성을 입증하였으며, 계산 트리의 유사성과 전이 학습 성능 간의 강한 상관관계를 보여주었습니다. 이 연구는 GFMs의 향후 개발에 중요한 길잡이가 될 것으로 기대됩니다.



### Zyda-2: a 5 Trillion Token High-Quality Datas (https://arxiv.org/abs/2411.06068)
Comments:
          initial upload 11/08/24

- **What's New**: Zyda-2라는 새로운 다섯 조 트리온(token) 데이터셋이 발표되었습니다. 이는 언어 모델 사전 훈련(pretraining) 용도로 사용되며, Zamba2 시리즈 모델을 교육하는 데 사용됩니다.

- **Technical Details**: Zyda-2는FineWeb과 DCLM과 같은 고품질 오픈 소스 token을 수집하여 제작되었습니다. 이를 통해 교차 중복 제거(cross-deduplication)와 모델 기반 품질 필터링(model-based quality filtering)을 통해 최고의 품질 하위 집합을 증류(distilling)하였습니다.

- **Performance Highlights**: Zyda-2를 기반으로 한 Zamba2 모델 시리즈는 해당 가중치(weight) 클래스에서 최첨단(state-of-the-art) 성능을 자랑합니다.



### Wild Narratives: Exploring the Effects of Animal Chatbots on Empathy and Positive Attitudes toward Animals (https://arxiv.org/abs/2411.06060)
- **What's New**: 이번 연구는 동물 정체성을 반영한 챗봇(chatbot)의 설계가 사용자들의 공감을 이끌어내는 데 어떻게 기여할 수 있는지를 탐구하였습니다. 실험 결과, 감정적 언어 표현과 동물의 생활에 대한 진정한 디테일을 포함할 때 공감과 태도 향상, 그리고 사회적 행동 의도를 증진할 수 있음을 확인했습니다.

- **Technical Details**: 240명의 참가자를 대상으로 한 혼합 연구 방법(mixed-methods experiment)을 통해, 챗봇이 가지는 구체적인 설계 단서가 사용자들의 인식에 미치는 영향을 분석하였습니다. 다양한 종류의 언어적 및 비언어적 단서를 제공하는 챗봇과의 상호작용을 통해, 공감과 태도 개선이 어떻게 이루어지는지를 평가하였습니다.

- **Performance Highlights**: 연구 결과, 감정적 언어 표현을 활용한 챗봇이 참가자들의 공감을 유도하고, 동물로서의 정체감 인식을 증가시켜 긍정적인 태도와 사회적 행동 의도를 증진시킬 수 있다는 것을 발견하였습니다. 이는 보존 프로젝트(conservation initiatives)나 교육 분야에서 챗봇 기술을 적용할 수 있는 새로운 가능성을 열어줍니다.



### An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models (https://arxiv.org/abs/2411.06048)
- **What's New**: 이 논문은 Spatial-MM이라는 새로운 VQA 데이터셋을 소개하여 Large Multimodal Models (LMMs)의 공간 이해 및 추리 능력을 종합적으로 연구하였습니다. 연구를 통해 LMMs가 인간의 시점에서 질문에 대한 답변을 잘 하지 못하고 복잡한 multi-hop 문제에 대한 모델 성능이 향상되지 않는 등의 문제점을 발견했습니다.

- **Technical Details**: Spatial-MM 데이터셋은 Spatial-Obj와 Spatial-CoT 두 개의 하위 집합으로 구성됩니다. Spatial-Obj는 이미지 내의 하나 또는 두 개의 객체 간의 공간적 관계를 포괄하는 다선택 질문을 포함하며, 2000개의 질문을 통해 LMMs의 공간적 추리를 평가합니다. Spatial-CoT는 개방형 multi-hop 질문을 제공합니다.

- **Performance Highlights**: 분석 결과, (i) 바운딩 박스 및 씬 그래프가 LMMs의 공간적 추리를 상당히 향상시키고, (ii) LMMs는 이미지에 대한 인간의 시각에서 제기된 질문에 대해 더 어렵게 반응하며, (iii) chain of thought (CoT) 프롬프트가 복잡한 multi-hop 질문의 모델 성능에 기여하지 않는 것으로 나타났습니다. 전반적으로 LMMs는 복잡한 공간적 추리에 비해 기초적인 객체 탐지에 훨씬 더 강한 성능을 보였습니다.



### PointCG: Self-supervised Point Cloud Learning via Joint Completion and Generation (https://arxiv.org/abs/2411.06041)
- **What's New**: 이 논문은 masked point modeling (MPM)과 3D-to-2D generation을 통합한 새로운 self-supervised learning 프레임워크인 PointCG를 제안합니다. 이 프레임워크는 Hidden Point Completion (HPC) 모듈과 Arbitrary-view Image Generation (AIG) 모듈을 포함하여 3D 객체를 효과적으로 인식할 수 있도록 합니다.

- **Technical Details**: PointCG는 두 가지 주요 모듈을 통합하여 동작합니다. HPC 모듈은 input으로부터 보이는 포인트를 이용해 전체 형태를 완성하고, AIG 모듈은 보이는 포인트의 표현을 기반으로 2D 이미지를 생성합니다. 이 과정에서 cross-modal feature alignment를 통해 포인트 클라우드와 이미지의 feature space를 일치시키고, encoder의 학습 초점을 다시 맞추어줍니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법이 다양한 downstream tasks에서 기존 방법들보다 우수한 성능을 보임을 입증하였습니다. 제안된 모듈은 정보가 제한된 포인트로부터 효과적으로 학습할 수 있게 해주어, masked point modeling과 arbitrary-view image generation의 효율성을 크게 향상시킵니다.



### CGLearn: Consistent Gradient-Based Learning for Out-of-Distribution Generalization (https://arxiv.org/abs/2411.06040)
Comments:
          9 pages, 3 figures

- **What's New**: 이번 논문에서는 여러 환경에서의 gradient 합의를 활용하여, 기계 학습 모델의 일반화 및 강건성을 향상시키기 위한 새로운 방법론인 CGLearn을 제안합니다. CGLearn은 다양한 환경에서의 일관된 특징을 통해 신뢰할 수 있는 예측 변수를 학습하는 데 중점을 둡니다.

- **Technical Details**: CGLearn은 gradient 일관성을 강제하여, 다양한 환경에서의 각 변수의 요소에 대한 invariant features를 활용합니다. 이 방법론은 Empirical Risk Minimization (ERM) 접근법을 기초로 하여, 각 특성의 gradient 일관성을 보장하여 신뢰할 수 있는 특성을 식별합니다. 이는 주로 선형 회귀 및 분류 작업에서 적용됩니다.

- **Performance Highlights**: CGLearn은 선형 및 비선형 설정에서 기존의 최첨단 방법들에 비해 우수한 예측력과 일반화 능력을 입증하였으며, 별도의 환경이 없더라도 그 성능을 발휘합니다. 다양한 합성 및 실제 데이터셋에 대한 실험 결과는 본 방법의 효과성과 강건성을 강조합니다.



### A Picture is Worth A Thousand Numbers: Enabling LLMs Reason about Time Series via Visualization (https://arxiv.org/abs/2411.06018)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)의 시간 시계열 추론(TsR) 능력을 평가하기 위한 새로운 테스트베드인 TimerBed를 제안합니다. TimerBed는 실제 작업에 관한 계층화된 추론 패턴, LLMs 및 추론 전략의 종합적인 조합을 포함하고 있습니다.

- **Technical Details**: TimerBed는 간단한 결정론적 추론, 복잡한 결정론적 추론 및 확률적 추론의 세 가지 패턴으로 구성되어 있으며, 각 패턴에 대해 실제 작업을 정렬합니다. 또한 VL-Time이라는 프롬프트 기반 솔루션을 제안하여 데이터 시각화 및 언어 가이드를 통해 LLMs의 시간 시계열 추론 능력을 강화합니다.

- **Performance Highlights**: VL-Time은 시계열에 대해 비트리비얼 제로샷과 강력한 몇 샷 추론 기술을 가능하게 하여 평균 140%의 성능 향상과 99%의 토큰 비용 절감을 이끌어냈습니다. 이는 단순한 수치 모델링에 비해 매우 뛰어난 결과입니다.



### The Dark Patterns of Personalized Persuasion in Large Language Models: Exposing Persuasive Linguistic Features for Big Five Personality Traits in LLMs Responses (https://arxiv.org/abs/2411.06008)
Comments:
          31 pages

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 개인의 성격 특성에 따라 언어적 특징을 조정하여 개인화된 설득적 출력을 생성하는 방법을 탐구합니다. 구체적으로 LLMs가 자신의 반응을 어떻게 조정하는지에 대한 첫 번째 연구를 제시합니다.

- **Technical Details**: 연구에서는 성격의 Big Five 모델에 따라 개인을 설득할 때 중요한 13가지 언어적 특징을 밝혀냈습니다. 연구는 5개의 모델 계열에서 19개의 LLM의 출력이 성격 특성 정보가 포함된 프롬프트에 어떻게 영향을 받는지를 분석했습니다. Shapley 값 등의 기법을 통해 언어적 특징의 중요성을 평가한 후 회귀 분석을 통해 성격 특성과 모델의 출력 간의 상호작용을 조사했습니다.

- **Performance Highlights**: 연구 결과, LLM들은 신경증에서 불안 관련 단어를 더 많이 사용하고, 성실성에서 성취 관련 단어를 증가시키며, 경험에 대한 개방성에서는 인지 과정과 관련된 단어를 덜 사용하는 경향이 있음을 보여줍니다. 이는 LLM이 사용자의 성격 단서에 따라 반응을 조정할 수 있음을 나타내며, 개인의 정신 건강과 복지에 영향을 줄 수 있는 설득적 콘텐츠를 생성할 가능성을 보여줍니다.



### Longitudinal Ensemble Integration for sequential classification with multimodal data (https://arxiv.org/abs/2411.05983)
Comments:
          11 pages, submitted to ICLR 2025

- **What's New**: 이번 연구에서는 다중 모드 및 종단적 데이터를 효과적으로 모델링하기 위해 새로운 Longitudinal Ensemble Integration (LEI) 학습 프레임워크를 개발했습니다. 이는 치매의 조기 탐지를 위한 다중 모드 시퀀스 분류에서 성능이 뛰어난 것으로 평가되었습니다.

- **Technical Details**: LEI는 다양한 데이터 모드로부터 개별 기본 예측기를 생성하고, 이러한 예측기를 Long Short-Term Memory (LSTM) 네트워크를 사용해 스택하는 방식을 통해 성능을 극대화합니다. 데이터는 Alzheimer’s Disease Prediction of Longitudinal Evolution (TADPOLE) Challenge에서 수집된 정보를 기반으로 합니다.

- **Performance Highlights**: LEI는 기존의 접근 방식보다 향상된 성과를 보였으며, 다양한 모드에서 중요한 피쳐들을 식별할 수 있었습니다. 이 프레임워크는 종단적 다중 모드 데이터에서의 연속적 분류에 대한 잠재력을 확장하여 다양한 응용 분야에 활용될 수 있습니다.



### Unmasking the Shadows: Pinpoint the Implementations of Anti-Dynamic Analysis Techniques in Malware Using LLM (https://arxiv.org/abs/2411.05982)
- **What's New**: 본 연구는 악성코드에서 Anti-Dynamic Analysis Techniques (TADA)의 위치를 자동으로 식별하고, 이를 통해 역공학자(reverse engineers)가 디버깅에 사용할 중단점을 보다 쉽게 설정할 수 있도록 지원하는 대규모 언어 모델(LLM) 기반의 워크플로우를 제안합니다.

- **Technical Details**: 제안된 워크플로우는 두 가지 주요 통찰을 기반으로 합니다. 첫째, LLM을 통해 기존의 규칙 기반(static analysis) 분석의 한계를 해결하며, 새로운 TADA 및 그 구현이 지속적으로 등장할 때에도 확장성을 보장합니다. 둘째, 디스어셈블(dissassembled) 코드와 자연어(natural language) 사이의 의미적 간격(semantic gap)을 다루기 위해 고급 정적 분석이 필수적입니다.

- **Performance Highlights**: 본 연구에서 제안된 방법은 공개된 레포지토리에서 확인된 TADA 구현의 87.80%를 성공적으로 식별하였으며, 온라인 악성코드 분석 블로그에 문서화된 4개의 잘 알려진 악성코드 샘플에서 TADA의 위치를 성공적으로 pinpoint 하였습니다.



### FactLens: Benchmarking Fine-Grained Fact Verification (https://arxiv.org/abs/2411.05980)
Comments:
          12 pages, under review

- **What's New**: 이 논문은 기존 LLM(대규모 언어 모델)의 사실 검증 방식을 재검토하고, 복잡한 주장을 세분화하여 각 서브 클레임(sub-claim)을 독립적으로 검증할 수 있는 방법을 제시합니다. 이를 통해 정확도, 투명성 및 증거 검색의 모호성을 줄일 수 있습니다.

- **Technical Details**: 저자들은 FactLens라는 새로운 벤치마크를 도입하여 세분화된 사실 확인을 평가합니다. FactLens는 서브 클레임의 품질을 평가하기 위한 메트릭스와 자동화된 평가자를 포함하고 있으며, 벤치마크 데이터는 수동으로 큐레이션되어 고품질의 그라운드 트루스를 보장합니다.

- **Performance Highlights**: 세분화된 검증 처리와 관련된 정량적 메트릭스를 통해 자동화된 평가자와 인간의 판단 간의 일치도를 입증하였으며, 서브 클레임 생성의 도전 과제를 논의하고, 최첨단 모델의 결과를 제시하였습니다.



### Assessing Foundational Medical 'Segment Anything' (Med-SAM1, Med-SAM2) Deep Learning Models for Left Atrial Segmentation in 3D LGE MRI (https://arxiv.org/abs/2411.05963)
- **What's New**: 본 연구는 의료 분야에 적합하게 조정된 MedSAM 모델을 사용하여 3D LGE MRI에서 좌심방(LA)의 자동 세분화(segmentation)를 수행하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구는 MedSAM 모델의 성능을 평가하고, 자동 추적(automated tracking)을 사용하는 MedSAM2 모델과 각 슬라이스에 대해 별도의 프롬프트(prompt)가 필요한 MedSAM1 모델의 성능을 비교하는데 초점을 맞추고 있습니다. 세분화 정확도를 평가하기 위해 Dice score를 분석합니다.

- **Performance Highlights**: MedSAM 모델은 기존 수동 세분화 방법에 비해 LA 세분화 작업에서 효율적인 비율로 성능 향상을 보이며, 다양한 박스 프롬프트(prompt)의 크기와 위치에 따른 MedSAM1 모델의 세분화 정확도(Dice score)를 평가합니다.



### Aligned Vector Quantization for Edge-Cloud Collabrative Vision-Language Models (https://arxiv.org/abs/2411.05961)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 LLaVA-AlignedVQ라는 엣지-클라우드 협업 기반의 VQA 시스템을 소개하며, Aligned Vector Quantization (AlignedVQ) 알고리즘을 통해 중간 특성을 효율적으로 압축하고 정확성을 유지하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: LLaVA-AlignedVQ는 중간 특성의 압축 비율을 약 1365배에 달하게 하여 데이터 전송 오버헤드를 96.8% 감소시키고, 원래 모델의 정확성과 비교하여 -2.23%에서 +1.6% 범위 내에서 높은 정확성을 유지합니다. 이 시스템은 초기 층을 엣지에서 로컬로 처리하고 나머지 층과 VLM 구성 요소를 클라우드에서 처리하는 파티셔닝 실행 방식을 채택하고 있습니다.

- **Performance Highlights**: LLaVA-AlignedVQ는 실행 속도를 2-15배 향상시키면서 높은 정확성을 유지합니다. 엣지에서 NVIDIA Jetson AGX Xavier를 사용하여 첫 번째 블록과 양자화 모듈을 실행하고 나머지 부분은 A100 GPU를 갖춘 워크스테이션에서 실행하며, 클라우드 전용 솔루션보다 뛰어난 성능을 자랑합니다.



### Sentiment Analysis of Cyberbullying Data in Social Media (https://arxiv.org/abs/2411.05958)
- **What's New**: 이 연구는 사이버 괴롭힘 탐지를 위한 감정 분석에 있어 두 가지 새로운 하이브리드 방법을 소개합니다. 기존의 기술 대신에 최신 임베딩을 활용하고, 특히 사이버 괴롭힘 데이터에 대한 RNN 프레임워크와 BERT, OpenAI 임베딩을 결합한 접근방법을 채택했습니다.

- **Technical Details**: 연구에서는 LSTM(Long Short-Term Memory) 셀을 사용하는 순환 신경망을 개발하여, 감정 분석에 대한 BERT 임베딩 및 OpenAI의 최신 임베딩 API를 비교합니다. 이 기술들은 자연어 처리(NLP)와 기계 학습(ML) 영역에서 고품질의 주석 데이터가 충분히 확보되어야 성능 평가가 가능하다는 점을 강조합니다.

- **Performance Highlights**: Formspring 사이버 괴롭힘 데이터를 사용하여 두 가지 접근 방식을 효과적으로 비교하며, 최신 LLM(Large Language Model)을 사용한 방법이 과거 방법들보다 더 높은 정확도로 사이버 괴롭힘을 탐지하는 데 기여하고 있음을 보여줍니다.



### NeKo: Toward Post Recognition Generative Correction Large Language Models with Task-Oriented Experts (https://arxiv.org/abs/2411.05945)
Comments:
          NeKo work has been done in June 2024. NeKo LMs will be open source on this https URL under the MIT license

- **What's New**: NeKo는 일반 포스트 인식 오류 수정 모델을 위한 새로운 접근방식으로, Mixture-of-Experts (MoE) 아키텍처를 활용하여 다양한 도메인 데이터셋에서 효율적으로 학습합니다. 이 모델은 서로 다른 작업을 위해 전문가를 훈련시켜 각 데이터셋 토큰을 해당 전문가에게 라우팅하는 방식으로 작동합니다.

- **Technical Details**: NeKo는 다양한 오류 수정 데이터셋의 혼합에 대해 사전 훈련된 MoE 모델을 기반으로 하며, 각 전문가는 특정 도메인에 특화됩니다. 이 과제 지향적인 MoE 세부 조정 방식은 각 전문가가 작업 특정 특징을 포착하게 하며, 지식 공유가 가능합니다. 이를 통해 네트워크는 음성 인식, 번역 및 OCR 오타 수정 등의 작업에서 높은 성능을 발휘합니다.

- **Performance Highlights**: NeKo는 Open ASR Leaderboard 및 Hyporadise 벤치마크에서 기존 모델들 대비 평균 상대 WER을 5.0% 감소시켰으며, 0-shot 평가에서 GPT-3.5와 Claude-Opus보다 15.5%에서 27.6%까지 WER 감소를 달성했습니다. 이 결과로 NeKo는 멀티태스크 모델에서 경쟁력을 입증하였습니다.



### GCI-ViTAL: Gradual Confidence Improvement with Vision Transformers for Active Learning on Label Nois (https://arxiv.org/abs/2411.05939)
Comments:
          under review

- **What's New**: 이번 연구에서는 label noise가 있는 상황에서 이미지 분류를 위한 Active Learning (AL) 방법들을 비교하고 새로운 deep active learning 알고리즘인 GCI-ViTAL을 제안합니다. 이 알고리즘은 label noise에 강인하도록 설계되었습니다.

- **Technical Details**: GCI-ViTAL은 예측 엔트로피(predictive entropy)와 클래스 중심 깨끗한 세트 주의 벡터(class-centric clean set attention vectors)와 비교한 마지막 레이어 주의 벡터의 Frobenius norm을 이용합니다. 이 모델은 불확실성과 의미적으로 전통적인 이미지와 다른 샘플을 식별하는 데 도움을 줍니다. Label smoothing이 적용되어 잠재적으로 노이즈가 포함된 레이블에 대해 지나치게 확신하지 않는 모델 학습을 지원합니다.

- **Performance Highlights**: GCI-ViTAL은 다양한 수준의 대칭 label noise에 대해 평가되었으며, CNN 모델에 비해 모든 AL 전략에서 ViT 모델을 사용하면 성능이 크게 향상됨을 보여주었습니다. 특히, label noise가 있는 경우에서 더 두드러진 성과를 보였습니다.



### Mitigating Hallucination with ZeroG: An Advanced Knowledge Management Engin (https://arxiv.org/abs/2411.05936)
Comments:
          10 pages, 4 figures, 1 table

- **What's New**: ZeroG는 Knowledge Distillation과 Prompt Tuning을 활용하여 LLM의 응답 질을 크게 향상시킵니다. 이 접근법은 기존의 복잡한 문서 처리에서 발생하는 환각(hallucinations) 문제를 해결하여 보다 정확하고 신뢰할 수 있는 반응을 제공합니다.

- **Technical Details**: ZeroG는 블랙 박스 디스틸레이션 접근 방식을 통해 더 작은 모델이 더 큰 교사 모델의 동작을 복제하도록 설계되었습니다. 이 방법은 문서 소재를 Markdown 형식으로 변환하고, Neo4j와 같은 그래프 데이터베이스를 사용하여 메타데이터를 관리하며, RAG(Retrieval Augmented Generation) 접근법을 통해 문서 처리를 최적화합니다.

- **Performance Highlights**: MMR(Maximal Marginal Relevance) 검색 기법을 통해 응답의 정확성과 관련성을 높였습니다. 실험 결과, MMR을 사용하는 시스템은 코사인 유사성(cosine similarity) 방법에 비해 최대 12%의 정확도 향상이 있었습니다.



### BERTrend: Neural Topic Modeling for Emerging Trends Detection (https://arxiv.org/abs/2411.05930)
Comments:
          17 pages, 12 figures, FuturED 2024: Workshop on Future of Event Detection (CoLocated with EMNLP 2024)

- **What's New**: BERTrend는 온라인 학습 설정에서 신경 주제 모델링을 활용하여 대규모 텍스트 자료의 신흥 트렌드와 약한 신호를 감지 및 추적할 수 있는 새로운 방법입니다. 이는 문서 수와 업데이트 빈도를 고려하여 시간에 따른 주제 인기도를 정량화하는 새로운 메트릭을 도입함으로써 그동안의 한계를 극복합니다.

- **Technical Details**: BERTrend는 HDBSCAN 알고리즘을 사용하여 자동으로 주제의 개수를 결정합니다. 또한 긴 문서를 단락으로 나눈 뒤 각 단락을 개별 문서로 취급하여 인기도 계산의 정확성을 높이고, 실시간으로 새로운 데이터가 들어올 경우 동적으로 주제를 추적할 수 있는 기능을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, BERTrend는 두 개의 대규모 실제 데이터셋에서 유의미한 약한 신호를 정확하게 탐지하고 잡음을 필터링하는 능력을 보여주었습니다. 이는 대규모 텍스트 자료에서의 신흥 트렌드 모니터링에 종합적인 솔루션을 제공합니다.



### Moving Off-the-Grid: Scene-Grounded Video Representations (https://arxiv.org/abs/2411.05927)
Comments:
          Accepted to NeurIPS 2024 (spotlight). Project page: this https URL

- **What's New**: MooG(Moving Off-the-Grid)는 비디오 표현 학습을 위한 새로운 self-supervised 방법론으로, 기존의 grid-based 처리 방식만으로 엮이지 않고, 영상의 구조와 무관하게 장면의 요소를 보다 일관되게 표현할 수 있는 토큰들의 움직임을 가능하게 한다.

- **Technical Details**: MooG는 cross-attention과 positional embeddings의 조합을 활용하여 representation 구조와 이미지 구조를 분리시킨다. 이 모델은 다음 프레임 예측(next frame prediction) 손실을 이용하여 비디오 데이터에서 학습되며, 입력 프레임이 도착할 때마다 cross-attention을 통해 토큰을 업데이트하고, 이미지를 복원하는데도 cross-attention을 사용한다.

- **Performance Highlights**: MooG는 DINO와 같은 기존 self-supervised grid-based 모델들을 초월하는 성능을 보이며, point tracking, monocular depth estimation 및 object tracking 등 다양한 다운스트림 비전 작업에 유용한 특징을 제공함을 정량적으로 및 정성적으로 입증하였다.



### Towards Multi-Modal Mastery: A 4.5B Parameter Truly Multi-Modal Small Language Mod (https://arxiv.org/abs/2411.05903)
- **What's New**: 새로운 4.5B 파라미터의 소형 언어 모델이 소개되었습니다. 이 모델은 텍스트, 이미지, 비디오, 오디오 등 다양한 입력 및 출력 모달리티(modality)를 처리할 수 있습니다.

- **Technical Details**: 이 모델은 언어 모델링(language modeling)과 다중 작업 학습(multi-task learning)의 최근 발전을 활용하여 만들어졌으며, 간편하게 엣지 추론(edge inference)에 배포될 수 있습니다.

- **Performance Highlights**: 모델은 여러 벤치마크에서 뛰어난 성능을 보여주며, 복잡한 현실 세계 문제를 해결할 수 있는 다중 모달(multi-modal) 인공지능의 가능성을 시사합니다.



### Integrating Object Detection Modality into Visual Language Model for Enhanced Autonomous Driving Agen (https://arxiv.org/abs/2411.05898)
Comments:
          accepted by SafeGenAI workshop of NeurIPS 2024

- **What's New**: 이번 논문은 자율주행 시스템의 시각적 이해를 향상시키기 위해 시각 언어 모델(Visual Language Models, VLMs)과 객체 탐지에 특화된 추가적인 시각 인식 모듈을 통합한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 Llama-Adapter 아키텍처를 확장하여 YOLOS 기반 탐지 네트워크를 CLIP 인식 네트워크와 통합합니다. 이를 통해 객체 탐지 및 위치 지정의 한계를 해결합니다. 카메라 ID 분리기를 도입하여 다중 시점 처리를 개선하고, 종합적인 환경 인식을 가능하게 합니다.

- **Performance Highlights**: DriveLM 시각 질문 응답 과제에서의 실험 결과는 ChatGPT 점수, BLEU 점수, CIDEr 지표에서 의미 있는 개선을 보여, 모델의 응답이 진실과의 근접성을 나타냅니다. 이번 연구는 자율주행 시스템의 능력과 해석 가능성을 향상시키는 promising한 단계로 평가됩니다.



### Humans Continue to Outperform Large Language Models in Complex Clinical Decision-Making: A Study with Medical Calculators (https://arxiv.org/abs/2411.05897)
- **What's New**: 대형 언어 모델(LLMs)이 임상 의사 결정 지원에서 의료 계산기 추천 능력을 평가한 최초의 연구입니다.

- **Technical Details**: 8개의 LLM(오픈 소스, 독점, 도메인 특정 모델 포함)을 평가하였으며, 1,009개의 질문-답변 쌍과 35개의 임상 계산기를 활용하였습니다. 인간 성능은 100개의 질문 하에서 측정되었습니다.

- **Performance Highlights**: 최고 성능의 LLM인 GPT-4o는 74.3%의 답변 정확도를 달성하였지만, 인간 주석자들은 평균 79.5%로 LLMs를 초월하였습니다. 오류 분석 결과 LLM의 컴프리헨션(이해) 오류율이 56.6%, 계산기 지식 오류율이 8.1%로 나타났습니다.



### SSSD: Simply-Scalable Speculative Decoding (https://arxiv.org/abs/2411.05894)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 연구에서는 Speculative Decoding(가설적 디코딩) 기법이 대규모 언어 모델(LLM) 추론을 가속화하는 데 효과적으로 사용될 수 있는 방법을 제시합니다. 여기서는 기존 시스템에 추가적인 훈련 없이 통합 가능한 방법을 소개하며, 짧은 컨텍스트 생성 시 처리량을 4배 증가시키고 지연 시간에는 영향을 미치지 않는 성과를 달성했습니다.

- **Technical Details**: 이 연구는 대규모 배치 크기에서 Speculative Decoding이 어떻게 효과적으로 적용될 수 있는지를 이론적으로 설명합니다. 전처리 단계에서 KV-cache를 생성하고 첫 번째 토큰을 생성하는 패러렐 처리와, 모델이 반복적으로 새로운 토큰을 생성하는 오토 리그레시브 디코딩 단계로 나뉩니다. 이 연구는 특히 대규모 LLM 배포에서 사용되는 복잡한 디코딩 과정을 최적화하는 새로운 기법을 제안합니다.

- **Performance Highlights**: 이 방법은 짧은 컨텍스트 생성에서 처리량을 4배 증가시키고, 긴 컨텍스트의 경우 지연 시간과 처리량이 각각 1.7배에서 2배 향상되는 성과를 보였습니다. 다양한 사용사례에서 효과적으로 활용될 수 있도록 설계되었습니다.



### Identifying and Decomposing Compound Ingredients in Meal Plans Using Large Language Models (https://arxiv.org/abs/2411.05892)
Comments:
          Comments: Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)

- **What's New**: 이 연구는 식단 계획에서 대형 언어 모델(Large Language Models, LLMs)의 효과를 탐구하며, 복합 재료의 식별 및 분해 능력에 중점을 둡니다.

- **Technical Details**: 세 가지 모델(GPT-4o, Llama-3 (70b), Mixtral (8x7b)의 성능을 평가했습니다. 초기 결과에 따르면 Llama-3 (70b)와 GPT-4o는 정확한 분해에 뛰어난 성능을 보이지만, 모든 모델이 양념 및 오일과 같은 필수 요소를 식별하는 데 어려움을 겪습니다.

- **Performance Highlights**: 강력한 전반적인 성능에도 불구하고, 모델 간 정확성과 완전성에서 차이가 관찰되었습니다. 이는 LLM이 개인 맞춤형 영양을 향상시킬 가능성을 강조하지만, 재료 분해에서의 추가 수정이 필요함을 나타냅니다.



### Towards Equitable ASD Diagnostics: A Comparative Study of Machine and Deep Learning Models Using Behavioral and Facial Data (https://arxiv.org/abs/2411.05880)
- **What's New**: 이번 연구에서는 여성의 자폐 스펙트럼 장애(ASD) 진단을 위한 기계 학습 모델, 특히 Random Forest와 convolutional neural networks를 평가했습니다. 이 연구는 ASD 진단을 개선하기 위한 혁신적인 접근 방식을 제안하고 있습니다.

- **Technical Details**: Random Forest 모델은 다수의 데이터셋에서 100% 검증 정확도를 달성했으며, 이 모델의 복잡한 관계 관리 능력과 낮은 오탐지율로 인해 조기 개입에 중요한 역할을 할 수 있습니다. MobileNet은 이미지 기반 분석에서 87%의 정확도로 baseline CNN을 초과했지만, 30%의 검증 손실이 나타나 추가 최적화가 필요합니다.

- **Performance Highlights**: Random Forest의 높은 정확도와 균형 잡힌 정밀도-재현율 지표는 임상 작업 흐름 개선에 기여할 수 있습니다. MobileNet의 경량 구조는 자원이 제한된 환경에서도 접근 가능한 ASD 스크리닝을 가능하게 할 잠재력을 보여줍니다.



### Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass (https://arxiv.org/abs/2411.05877)
- **What's New**: 이번 연구는 새로운 컨텍스트에 대한 효과적이고 효율적인 적응 방법인 GenerativeAdapter를 소개합니다. 이 방법은 사전 학습된 언어 모델을 미세 조정(fine-tuning) 없이 저랭크(low-rank) LM 어댑터에 직접 매핑하여 추론 오버헤드를 크게 감소시킵니다.

- **Technical Details**: GenerativeAdapter는 자가 지도 학습(self-supervised learning)을 통해 훈련된 어댑터 생성기(adapter generator)를 사용합니다. 이 어댑터 생성기는 고정된 LM을 단일 온-더 플라이(on-the-fly)로 적응시키며, 그러한 컨텍스트를 새로운 어댑터에 매핑합니다. 본 연구에서는 Mistral-7B-Instruct와 Llama2-7B-Chat이라는 두 개의 사전 학습된 LM에 GenerativeAdapter를 적용했습니다.

- **Performance Highlights**: StreamingQA에서 우리의 접근 방식은 LM의 파라미터에 지식을 주입하는 데 효과적이며, 감독 미세 조정(supervised fine-tuning)이 있는 모델 대비 F1 점수에서 63.5%의 향상(19.5에서 31.5로)이라는 성과를 달성했습니다. 다양한 적응 시나리오에서 평균 정확도 44.9%를 기록하여 기본 모델을 초월했습니다.



### Towards Improved Preference Optimization Pipeline: from Data Generation to Budget-Controlled Regularization (https://arxiv.org/abs/2411.05875)
Comments:
          15 pages

- **What's New**: 최근 Direct Preference Optimization (DPO) 및 그 변형들이 큰 언어 모델(LLM)의 조정에 있어 주요 방법이 되고 있습니다. 이 연구에서는 DPO의 선호 데이터 생성 및 교육 정규화 기술을 개선하는 방법을 제안합니다.

- **Technical Details**: 우리는 반복 쌍별 순위 메커니즘을 도입하여 선호 데이터의 품질을 향상시키고, 예측된 선호 샘플의 우선 확률을 약간 감소시키는 새로운 예산 관리 정규화 기법을 사용합니다. 이는 LLM의 예측 정확성을 유지하면서도 최적화 과정의 안정성을 가져옵니다.

- **Performance Highlights**: 이 연구에서 제안한 방법들은 두 가지 일반 벤치마크 평가를 통해 기존 SOTA를 능가하는 결과를 보였으며, 현업에서의 LLM 적용에 있어 높은 품질의 선호 데이터 생성을 통해 더 나은 최적화를 이루었습니다.



### Interplay between Federated Learning and Explainable Artificial Intelligence: a Scoping Review (https://arxiv.org/abs/2411.05874)
Comments:
          16 pages, 11 figures, submitted in IEEE Trans. Knowledge and Data Engineering

- **What's New**: 이번 논문은 Federated Learning (FL)과 Explainable Artificial Intelligence (XAI)의 결합에 대한 최신 연구 결과를 정리하고 있습니다. 37개의 연구가 FL과 XAI의 상호 작용을 분석하였으며, 특히 데이터가 분산되어 있는 환경에서의 모델의 해석 가능성과 설명 방식에 대해 논의하고 있습니다.

- **Technical Details**: FL은 머신러닝 모델을 중앙 집중식 데이터 전송 없이 분산된 데이터로부터 훈련시키는 기술입니다. 이와 함께 XAI는 AI 시스템의 결정 과정을 설명 가능하게 만드는 알고리즘 및 방법론을 개발하는 분야입니다. HFL(수평적 연합 학습), VFL(수직적 연합 학습), FTL(연합 전이 학습)과 같은 FL의 세 가지 주요 카테고리가 존재하며, 모델의 해석 가능성 및 post-hoc explanation(사후 설명 방법)을 통한 다양한 접근이 필요합니다.

- **Performance Highlights**: FL과 XAI의 결합 연구는 증가하고 있으나, 연구의 대다수는 데이터 센터 수가 10개 이하인 시뮬레이션 환경을 사용했습니다. 8개의 논문은 FL 알고리즘의 구성 요소로서 설명 방법을 통합하여 얻는 이점을 다루었고, FL과 XAI의 상호 작용에 대한 보다 정량적이고 구조화된 연구가 필요함을 강조하였습니다.



### Modeling Nonlinear Oscillator Networks Using Physics-Informed Hybrid Reservoir Computing (https://arxiv.org/abs/2411.05867)
Comments:
          27 pages, 10 figures, 17 supplementary figures. Code available at this https URL

- **What's New**: 본 논문에서는 하이브리드 레저버 컴퓨팅(hybrid reservoir computing)을 사용하여 비선형 진동자 네트워크(non-linear oscillator networks)의 대체 모델을 개발했습니다. 이 모델은 '전문가' 해석 모델과 결합하여 복잡한 실제 상황을 더 잘 반영할 수 있게 설계되었습니다.

- **Technical Details**: 하이브리드 레저버 컴퓨팅(Hybrid Reservoir Computing, HRC)은 전통적인 레저버 컴퓨팅과 전문가 모델을 통합한 것으로, 주요 비선형 결합 항들이 포함된 확장된 실제 모델과 비교하여 성능을 평가했습니다. 간단한 모델 대신 레저버 컴포넌트를 통해 보완할 수 있는지를 조사했습니다.

- **Performance Highlights**: 하이브리드 레저버 컴퓨터는 일반적으로 표준 레저버 컴퓨터보다 더 나은 성능을 보였으며, 특히 관측된 스펙트럴 반경(spectral radius) 임계값을 넘을 때 성능 저하가 발생하지 않았습니다. 또한, 전문가 모델로 진입할 수 없는 역동적 상황에서도 우수한 성능을 나타냈습니다.



### Bilinear Fuzzy Genetic Algorithm and Its Application on the Optimum Design of Steel Structures with Semi-rigid Connections (https://arxiv.org/abs/2411.05865)
Comments:
          19 pages, 12 figures, book chapter, Springer

- **What's New**: 본 논문에서는 반강체 연결(semi-rigid connections)을 가진 강철 구조물의 설계를 최적화하기 위한 개선된 이선형 퍼지 유전 알고리즘(BFGA)을 소개합니다.

- **Technical Details**: BFGA는 퍼지 로직(fuzzy logic)과 유전 알고리즘(genetic algorithm)의 장점을 결합하여 구조 설계 문제의 복잡성과 불확실성을 처리하는 강력한 최적화 방법입니다. 이 알고리즘은 비선형(nonlinear) 행동을 가진 반강체 연결의 문제를 다루는 데 중점을 둡니다.

- **Performance Highlights**: BFGA는 표준 GA에 비해 합리적인 시간 안에 고품질의 솔루션을 생성하는 것으로 나타났습니다. 또한 이 최적화 방법은 모든 설계 요구사항과 제약조건을 만족하는 최적 설계를 찾는 데 성공했습니다.



### Boosting the Efficiency of Metaheuristics Through Opposition-Based Learning in Optimum Locating of Control Systems in Tall Buildings (https://arxiv.org/abs/2411.05864)
Comments:
          17 pages, 4 figures, book chapter, Springer

- **What's New**: 이 논문에서는 Opposition-based Learning (OBL) 기법을 활발히 탐구하며, 메타휴리스틱 최적화 알고리즘에 대한 적용 가능성과 개선 방안에 대해 설명합니다.

- **Technical Details**: OBL은 메타휴리스틱 최적화 알고리즘의 성능을 향상시키기 위한 효과적인 방법입니다. 다양한 구현 방식 및 그 성능에 미치는 영향이 논의됩니다. 특히, MR(마그네토레올로직) 유체 댐퍼가 포함된 전단 프레임을 사례 연구로 제시합니다.

- **Performance Highlights**: 결과적으로, 최적화 과정에 OBL을 포함시킴으로써 알고리즘의 품질과 속도가 현저히 향상됨을 보여줍니다. 이는 실제 엔지니어링 문제에 대한 OBL의 적용을 위한 명확한 이해를 제공합니다.



### From Electrode to Global Brain: Integrating Multi- and Cross-Scale Brain Connections and Interactions Under Cross-Subject and Within-Subject Scenarios (https://arxiv.org/abs/2411.05862)
- **What's New**: 본 논문에서는 멀티스케일 공간 데이터 분포 차이를 해결하기 위한 새로운 다중 스케일 공간 도메인 적응 네트워크(MSSDAN)를 제안합니다. 이는 EEG 신호의 개인 차이를 극복하는 데 집중하며, 특히 단일 소스에서 단일 타겟(STS) 시나리오를 다룹니다.

- **Technical Details**: MSSDAN은 멀티스케일 공간 피쳐 추출기(MSSFE)와 다중 스케일 공간 도메인 적응(MSSDA) 방법으로 구성되어 있습니다. 이 구조는 뇌의 다중 스케일 토폴로지 원리를 통합하여 EEG 신호의 다중 스케일 공간 데이터 분포 차이를 해결하려고 합니다.

- **Performance Highlights**: 제안된 MSSDAN 모델은 기존의 데이터 분포 차이를 해결하는 방법들과 달리, 각 뇌 영역의 전극 간 데이터 분포의 차이를 고려하여 동작합니다. 이 모델은 주어진 타겟 도메인에 대한 분류 성능을 개선하는 것으로 검증되었습니다.



### Conditional Diffusion Model for Longitudinal Medical Image Generation (https://arxiv.org/abs/2411.05860)
Comments:
          4 pages, 2 figures, conference

- **What's New**: 이 논문은 알츠하이머병(Alzheimer's disease)의 진행 과정을 3D 의료 이미징(medical imaging) 데이터로 모델링하는 새로운 접근 방식을 제안하고 있습니다.

- **Technical Details**: 제안된 방법은 단일 자기 공명 영상(single magnetic resonance imaging, MRI)을 사용하여 확산 기반(diffusion-based) 모델을 구현합니다. 이 모델은 조건부 MRI(conditioning MRI)와 시간 방문 인코딩(time-visit encoding)을 주입하여 원본(source) 이미지와 대상(target) 이미지 간의 변화를 제어합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 경쟁 방법들에 비해 더 높은 품질의 이미지를 생성하는 것으로 나타났습니다.



### Enhancing Financial Fraud Detection with Human-in-the-Loop Feedback and Feedback Propagation (https://arxiv.org/abs/2411.05859)
Comments:
          International Conference on Machine Learning and Applications 2024

- **What's New**: 이번 논문은 Human-in-the-Loop (HITL) 피드백 메커니즘이 재정 사기 탐지에서 기계 학습(ML) 모델의 성능을 어떻게 향상시키는지를 다룹니다. 특히 전문가의 소량의 피드백으로도 모델의 정확도가 크게 증가할 수 있음을 보여주고, 그래프 기반 기술이 가장 큰 혜택을 보는 것으로 나타났습니다.

- **Technical Details**: HITL 시스템은 인간의 전문성과 기계 학습 기술을 결합하여 데이터를 주석 처리하고 모델 학습을 보강하는 접근 방식입니다. 새로운 피드백 전파 방법을 소개하며, 이는 데이터셋 전반에 피드백을 확장하여 탐지 정확도를 더욱 향상시킵니다. 그리고 기계 학습 모델을 훈련시키고 평가하는 전통적인 방법과 첨단 기술을 통합하는 HITL 프레임워크를 제안합니다.

- **Performance Highlights**: HITL 피드백을 통합함으로써, 데이터 주석 품질, 모델 해석 가능성 및 동적 사기 패턴 적응성을 개선할 수 있었습니다. 연구 결과는 HITL 메커니즘이 기존 및 최신의 금융 사기 탐지 기법을 향상시킬 수 있는 잠재력을 보여주며, 대다수의 알고리즘들에서 사기 탐지 성능을 크게 향상시켰습니다.



### Evaluating the Economic Implications of Using Machine Learning in Clinical Psychiatry (https://arxiv.org/abs/2411.05856)
Comments:
          11 pages, submitted to Machine Learning for Health (ML4H) 2024

- **What's New**: 이 연구는 임상 정신의학에서 머신러닝(ML)의 경제적 함의에 대한 기존 연구의 공백을 메우고자 하며, 이 분야의 비용 효과성과 공정성 문제를 다룹니다.

- **Technical Details**: 본 연구는 3개의 문제 지향 사례 연구(case studies), 경제학 및 의료 AI에 관한 문헌, 두 가지 종류의 건강 경제 평가(health economic evaluations)를 통해 ML의 경제적 함의를 평가합니다.

- **Performance Highlights**: 정신 질환으로 인한 글로벌 비용이 5조 달러에 달한다고 추정되며, ML은 진단 정확도를 향상시키고 자원을 절약함으로써 정신 건강 서비스의 제공을 개선할 가능성이 있습니다.



### Harmful YouTube Video Detection: A Taxonomy of Online Harm and MLLMs as Alternative Annotators (https://arxiv.org/abs/2411.05854)
- **What's New**: 본 연구는 YouTube, Instagram, TikTok과 같은 단편 동영상 플랫폼에서 해로운 콘텐츠를 탐지하기 위한 새로운 측정 방법과 기법을 개발했습니다. 또한 해로움의 정의에 대한 종합적인 분류법을 제시하며, 이를 6가지 카테고리로 분류했습니다.

- **Technical Details**: 이 연구에서 개발한 분류법은 정보(Information), 증오 및 괴롭힘(Hate and harassment), 중독(Addictive), 클릭베이트(Clickbait), 성적(Sexual), 신체적(Physical) 해로움으로 구분됩니다. 19,422개의 YouTube 동영상을 14개의 이미지 프레임, 1개의 썸네일(Thumbnail), 텍스트 메타데이터를 사용하여 분석하였습니다.

- **Performance Highlights**: GPT-4-Turbo는 Crowdworkers (Mturk)보다 이진 분류(유해 vs 무해) 및 다중 레이블 해로움 분류 작업에서 더 높은 정확성을 보였습니다. 이 연구는 LLMs의 적용을 텍스트 주석 및 이진 분류를 넘어 다중 레이블 및 다중 모달(contexts)로 확장합니다.



### Input-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks (https://arxiv.org/abs/2411.05849)
Comments:
          24 pages, 15 figures

- **What's New**: 이 논문에서는 Hopfield 모델의 새로운 동역학 시스템 프레임워크를 제안하여 외부 입력이 신경 시냅스에 직접 영향을 미치고 Hopfield 모델의 에너지 경관을 형성하는 방식을 설명합니다. 이는 기억 검색 과정에 대한 명확한 에너지 해석을 제공하며, 혼합된 입력들을 올바르게 분류하는 데 효과적입니다.

- **Technical Details**: 이 모델은 현대 Hopfield 아키텍처의 틀 내에서 통합되어 현재 정보와 과거 정보가 검색 과정에서 어떻게 결합되는지를 밝힙니다. 특히, 이 논문은 입력이 에너지 경관을 형성하고 결과적인 경량 하강(gradient descent) 흐름에 영향을 미친다는 점이 주요 특징입니다.

- **Performance Highlights**: 노이즈가 있게 구성된 환경에서 고전적인 모델과 새로운 모델의 강건성을 비교하며, 우리 모델의 장점은 과거와 현재 정보를 통합하여 부정확한 입력에 의해 유도되는 오분류 오류를 줄이는 데 있습니다.



### Federated Data-Driven Kalman Filtering for State Estimation (https://arxiv.org/abs/2411.05847)
- **What's New**: 이 논문은 자율주행 차량의 고도로 정확한 위치정보를 제공하기 위해 협동 훈련 또는 연합 학습(Federated Learning) 패러다임에 기초한 새로운 로컬라이징 프레임워크를 제안합니다. 기존의 KalmanNet을 기반으로 하여 FedKalmanNet으로 재구성하여 분산 방식으로 훈련합니다.

- **Technical Details**: 본 연구는 KalmanNet의 원리를 사용하여 차량 각각이 자신의 데이터셋을 활용하여 지역 모델을 학습하고, 이를 통해 시스템 불확실성을 추정하는 방법을 사용했습니다. FedKalmanNet은 차량들 간의 협력적 훈련을 통해 전 세계적으로 정보를 융합하여 높은 성능을 자랑합니다.

- **Performance Highlights**: CARLA 자율주행 시뮬레이터에서 수행한 실험에서는 FedKalmanNet이 전통적인 협력적 결정 방식보다 성능이 우수하며, 실시간 V2X 통신이 필요하지 않다는 장점을 보여주었습니다.



### Diversify, Contextualize, and Adapt: Efficient Entropy Modeling for Neural Image Codec (https://arxiv.org/abs/2411.05832)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 효율적인 엔트로피 모델링을 위한 새로운 프레임워크 DCA(Diversify, Contextualize, Adapt)를 제안하고 있으며, 이는 기존 방법들보다 더 다양한 컨텍스트를 활용하여 성능을 향상시킵니다.

- **Technical Details**: DCA는 세 가지 다른 하이퍼 잠재 표현(local, regional, global)을 추출하여 전방 적응(forward adaptation)을 위한 충분한 컨텍스트를 제공합니다. 이 프레임워크는 효율적인 후방 적응(backward adaptation) 방법과 결합하여 개선된 성능을 보장합니다.

- **Performance Highlights**: Kodak 데이터셋에서 기존 최첨단 방법에 비해 3.73% BD-rate 증가를 달성하여 다양한 비트레이트 영역에서 성능 개선을 보여주었습니다.



### Utilizing RNN for Real-time Cryptocurrency Price Prediction and Trading Strategy Optimization (https://arxiv.org/abs/2411.05829)
Comments:
          10 pages, 16 figures, 1 table

- **What's New**: 이 연구는 Recurrent Neural Networks (RNN)을 이용하여 실시간 암호화폐 가격 예측과 최적화된 거래 전략을 탐구합니다. 고도의 변동성을 가진 암호화폐 시장의 자연에서, 전통적인 예측 모델은 종종 부족함을 보입니다. 이 연구는 RNN의 장기 패턴 감지 능력을 이용하여 가격 예측의 정확성을 향상시키고 효과적인 거래 전략을 개발할 것을 목표로 합니다.

- **Technical Details**: 이 연구는 Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC) 가격 예측 모델을 개발하며, Long Short-Term Memory (LSTM), Bi-directional LSTM (Bi-LSTM), Gated Recurrent Units (GRU)와 같은 딥 러닝 (Deep Learning) 알고리즘을 사용합니다. 모델의 성능을 평가하기 위해 Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) 등의 성능 지표를 활용합니다.

- **Performance Highlights**: 이 연구는 Keras와 TensorFlow를 사용하여 각 RNN 모델을 개발하고, 80%의 데이터로 훈련한 후, 20%의 테스트 데이터에서 성능을 평가합니다. 모델의 성능 지표로는 MSE, MAE, RMSE, MAPE를 사용하여 가장 작은 오차 값을 가진 모델이 선택됩니다. 세 가지 모델 모두 높은 예측 정확도를 보여주며, 암호화폐 투자가와 거래자들에게 유용한 가격 예측 도구로 활용될 수 있습니다.



### From Pixels to Prose: Advancing Multi-Modal Language Models for Remote Sensing (https://arxiv.org/abs/2411.05826)
Comments:
          10 pages, 1 figure

- **What's New**: 본 논문은 원거리 탐지(remote sensing) 분야에서 다중 모달 언어 모델(MMLMs)의 개발과 응용을 탐구합니다. 위성 이미지를 자연어로 해석하고 설명할 수 있는 모델의 능력에 초점을 맞추고 있습니다.

- **Technical Details**: MMLMs는 일반적으로 이중 인코더 아키텍처(dual-encoder architecture)를 사용하여 시각적 및 텍스트 정보를 통합 처리합니다. Transformer 모델을 활용하여 복잡한 원거리 탐지 데이터를 효과적으로 처리하며, 주목(attention) 메커니즘을 통해 시각 및 텍스트 입력의 중요한 부분에 집중하는 방식으로 작동합니다.

- **Performance Highlights**: 이 모델들은 환경 모니터링, 도시 계획 및 재난 대응과 같은 주요 응용 분야에서 효율적인 정보 추출을 통해 자동화된 지구 관측 분석을 크게 향상시킵니다. 특히, 장면 설명(scene description), 객체 탐지(object detection), 변화 탐지(change detection) 등 다양한 응용 분야에서 그 효과를 입증하고 있습니다.



### SurfGNN: A robust surface-based prediction model with interpretability for coactivation maps of spatial and cortical features (https://arxiv.org/abs/2411.05825)
Comments:
          15 pages, 6 figures

- **What's New**: 이 논문에서는 기존 뇌표면 기반 예측 모델의 지역적 속성 변동성을 간과한 점을 개선하기 위해 Surface Graph Neural Network (SurfGNN)를 제안합니다. SurfGNN은 topology-sampling learning (TSL) 및 region-specific learning (RSL) 구조를 활용하여 각 피질 특성을 보다 효과적으로 관리합니다.

- **Technical Details**: SurfGNN은 피질 표면 메시를 희소 그래프로 간주하고, TSL과 RSL 구조를 통해 표면 메시의 낮은 및 높은 스케일에서 개별 피질 특성을 처리합니다. 이를 통해 복잡한 고밀도 그래프 구조의 문제를 해결하고, score-weighted fusion (SWF) 방법으로 각 피질 특성에 대한 노드 표현을 통합하여 예측을 수행합니다.

- **Performance Highlights**: SurfGNN은 481명의 환자(503 스캔)로부터 얻은 통합된 MRI 데이터를 사용하여 신생아의 뇌 나이를 예측하는 작업에서 기존의 모든 최신 방법을 능가하였으며, 평균 절대 오차(mean absolute error, MAE)는 0.827+0.056로 개선되었습니다.



### FlexCAD: Unified and Versatile Controllable CAD Generation with Fine-tuned Large Language Models (https://arxiv.org/abs/2411.05823)
Comments:
          23 pages

- **What's New**: 최근 사용자 의도에 기반하여 컴퓨터 지원 설계(CAD) 모델을 생성하는 접근 방식인 controllable CAD generation에 대한 관심이 증가하고 있습니다. 이러한 배경에서 저자들은 FlexCAD라는 통합 모델을 제안하며, 이는 정적 지식에서 벗어나 다양한 CAD 구조 계층을 효율적으로 실현할 수 있는 모델입니다.

- **Technical Details**: FlexCAD는 대규모 언어 모델(LLM)을 미세 조정하여 CAD 모델을 구조화된 텍스트로 표현하는 방식입니다. 이 과정에서 CAD 모델의 각 계층을 텍스트 토큰의 시퀀스로 변환하고, 계층 인식을 위한 마스킹 전략을 도입하여 다양한 생성 작업을 통합하고 있습니다.

- **Performance Highlights**: FlexCAD는 공공 데이터셋에 대한 포괄적인 실험을 통해 생성 품질과 제어 가능성을 크게 향상시키는 효과를 보였습니다. 또한, 이 모델은 기존의 CAD 생성 방식과 비교하여 효율적인 성능을 제공하고 있습니다.



### Guiding Genetic Programming with Graph Neural Networks (https://arxiv.org/abs/2411.05820)
Comments:
          Full version of the same-titled paper accepted at GECCO 2024

- **What's New**: EvoNUDGE라는 새로운 신경진화 방법론을 제안하여, 유전적 프로그래밍(genetic programming)에서 상징 회귀(symbolic regression) 문제 해결을 위한 추가 지식을 도출합니다.

- **Technical Details**: EvoNUDGE는 그래프 신경망(graph neural network, GNN)을 활용해 초기 집단을 생성하고 검색 연산자의 동작을 편향시키는 방식으로 작동합니다. 이 방식은 기존 접근법과 달리 후보 솔루션의 부프로그램 정보를 활용하여 복잡한 문제를 해결하기 위한 검색 가이드를 제공합니다.

- **Performance Highlights**: EvoNUDGE는 다양한 기준선(baselines)보다 유의미한 성과를 보였으며, 전통적인 트리 기반 유전적 프로그래밍과 순수 신경망 변형 모두를 초월해 우수한 결과를 달성했습니다.



### Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy (https://arxiv.org/abs/2411.05819)
Comments:
          20 Pages

- **What's New**: 이번 연구는 기존의 증오 발언 (hate speech) 탐지 방법의 한계를 극복하고자, 감정 분석 (sentiment analysis)과 공유된 감정 표현 (shared emotional representations)을 통합한 새로운 멀티태스크 (multitask) 모델을 제안합니다.

- **Technical Details**: 본 연구에서는 딥 러닝 (deep learning)과 머신 러닝 (machine learning) 기반의 증오 발언 텍스트 분류 시스템 모델을 제시합니다. Hugging Face의 Transformer 기반 모델을 활용하여 감정 분석을 통해 잘못된 긍정 (false positives)을 최소화하는 방법을 모색하였습니다.

- **Performance Highlights**: 여러 데이터셋에서의 실험 결과, 감정 분석과 Transformer 기반의 훈련된 모델을 활용함으로써 증오 발언 탐지의 정확성이 상당히 향상되었음을 확인하였습니다.



### Learning Characteristics of Reverse Quaternion Neural Network (https://arxiv.org/abs/2411.05816)
- **What's New**: 이 논문은 새로운 multi-layer feedforward quaternion neural network 구조인 Reverse Quaternion Neural Network(RQNN)를 제안하며, 여기서 쿼터니온 곱셈의 비가환성(non-commutative nature)을 활용합니다. 특히 기존 multi-layer feedforward 쿼터니온 신경망에서 가중치를 역방향으로 적용한 특성에 대한 연구는 없었습니다.

- **Technical Details**: 이 모델은 쿼터니온의 곱셈 순서를 변경하여 서로 다른 특성을 가진 신경망을 구성할 수 있습니다. RQNN은 학습 속도와 회전에서의 일반화 능력을 두 가지 관점에서 분석하였으며, 그 결과 기존 모델에 비해 유사한 학습 속도를 보여주었고, 기존 모델과는 다른 회전 표현을 얻을 수 있음을 발견하였습니다.

- **Performance Highlights**: RQNN은 기존 multi-layer feedforward quaternion neural network와 비교하여 학습 속도가 유사하고, 회전에 대한 일반화 능력이 뛰어난 것으로 나타났습니다. 이 연구는 특히 3D 구조를 처리하고 관련 특징을 포착하는데 있어 기존 모델이 놓칠 수 있는 정보를 효과적으로 다룰 수 있는 가능성을 증명합니다.



### Neurophysiological Analysis in Motor and Sensory Cortices for Improving Motor Imagination (https://arxiv.org/abs/2411.05811)
Comments:
          4 pages, 3 figures, 1 table, Name of Conference: International Winter Conference on Brain-Computer Interface

- **What's New**: 이 연구는 뇌-컴퓨터 인터페이스(BCI)가 운동 실행(motor execution, ME)과 운동 이미지(motor imagery, MI) 작업에서의 신경 신호를 어떻게 해석하는지를 탐색하고 있습니다. 연구자는 감각-관련(온도) 및 운동-관련(당기기 및 밀기) 네 가지 조건을 설정하여 뇌의 운동 감각 피질에서의 활성화를 비교했습니다.

- **Technical Details**: EEG 신호를 활용하여 ME 및 MI 작업의 신경 서명을 분석했습니다. 스켈프 토포그래피 분석을 통해 감각-관련 조건이 주로 감각 운동 피질의 후부를 자극하고, 운동-관련 조건이 앞부분을 자극함을 밝혔습니다. 세 가지 신경 네트워크 모델(EEGNet, ShallowConvNet, DeepConvNet)의 성능을 평가하였으며, ME 작업이 MI 작업보다 높은 분류 정확도를 보였습니다.

- **Performance Highlights**: 감각-관련 조건에서 가장 높은 정확도가 차가운 조건에서 관찰되었으며, 운동-관련 조건에서는 당기기 작업이 가장 높은 성능을 보였습니다. 특히, DeepConvNet 모델이 가장 높은 성능을 기록하였습니다. 이 발견들은 특정 조건에 따른 신경 활성화를 활용하여 BCI 응용을 최적화할 수 있는 통찰력을 제공합니다.



### Is it me, or is A larger than B: Uncovering the determinants of relational cognitive dissonance resolution (https://arxiv.org/abs/2411.05809)
- **What's New**: 이 연구는 인지 부조화를 해결하는 데 사용되는 계산적 메커니즘을 탐구합니다. 특히, 예상되는 객체 간의 관계를 위반하는 관찰 시나리오에 초점을 맞추고, 인공지능 신경망(Artificial Neural Networks, ANNs)을 사용하여 이러한 두 경로의 존재를 입증합니다.

- **Technical Details**: 연구는 입력 표현 모듈과 관계 모듈의 두 가지 모듈로 구성된 ANN을 사용하여 수행되었습니다. 입력 표현 모듈은 관련 특성을 인코딩하며, 관계 모듈은 다양한 입력 표현 간의 관계를 특성화합니다. 적응 경로는 인지적 불일치의 크기에 따라 달라지며, 큰 불일치는 객체의 표현을 변경하고, 작은 불일치는 예상 관계의 조정을 초래합니다.

- **Performance Highlights**: 실험에서는 ANNs가 이미지 쌍을 제시받고 올바른 순서를 식별해야 하는 작업을 수행했습니다. 연구 결과, ANN이 표현 모듈이나 관계 모듈을 조정하여 인지 부조화를 해결하는 방식을 보여주었고, 이를 통해 인간의 인지 부조리에 대한 이해에 기여할 수 있는 통찰을 제공하고 있습니다.



### SkipSNN: Efficiently Classifying Spike Trains with Event-attention (https://arxiv.org/abs/2411.05806)
Comments:
          Published as a research paper at IEEE BigData 2024

- **What's New**: 이 논문은 Spiking Neural Networks (SNNs)의 한계를 극복하기 위해 event-attention 메커니즘을 도입하여 유용한 신호를 강조합니다. 이를 통해 SkipSNN 모델이 개발되었으며, 이 모델은 기존 SNNs에 비해 더욱 높은 계산 효율성과 분류 정확도를 달성합니다.

- **Technical Details**: SkipSNN은 기존 SNN 모델을 확장하여 시간 단계에 따라 두 가지 상태(각성 상태와 동면 상태) 간 전환할 수 있도록 설계되었습니다. 정보가 필요할 때만 계산이 이루어지며, 비필요한 노이즈를 무시하는 방식으로 메모리와 에너지를 절약할 수 있습니다. 새로운 손실 함수로 정확도와 계산 비용 간의 균형을 맞춥니다.

- **Performance Highlights**: SkipSNN은 neuromorphic MNIST와 DVS-Gesture 데이터셋에서 최신 SNN 모델들과 비교하여 더 높은 정확도와 낮은 계산 비용을 기록했습니다. 이러한 성과는 에너지가 제한된 센서 장치에서 특히 중요한 장점으로 작용합니다.



### Similarity-based context aware continual learning for spiking neural networks (https://arxiv.org/abs/2411.05802)
- **What's New**: 본 연구에서는 유사한 작업 간의 유사성을 기반으로 한 Context Aware Spiking Neural Network(SCA-SNN) 지속 학습 알고리즘을 제안합니다. 이 알고리즘은 기존의 지속 학습 알고리즘이 작업을 동등하게 다룬다는 점에서 벗어나, 작업 간 유사성 관계를 활용하여 신경망의 학습 효율성을 개선하는 기법입니다.

- **Technical Details**: SCA-SNN 모델은 현재 데이터와 네트워크 상태를 통합하여 작업 유사성을 평가하는 방법을 설계합니다. 이 평가를 통해 이전 작업에서 유용한 신경세포를 재사용하고 새로운 신경세포를 유연하게 확장하는 원칙을 마련했습니다. 또한, '사용하지 않으면 잊혀진다'는 뇌의 발달 유연성에 착안하여, 새로운 작업에 효과적으로 기여하는 신경세포만을 선택적으로 재사용하는 방법을 구현합니다.

- **Performance Highlights**: SCA-SNN 모델은 CIFAR100, ImageNet 일반화 데이터 세트 및 FMNIST-MNIST, SVHN-CIFAR100 혼합 데이터 세트에서 실험을 통해 뛰어난 성능을 입증했습니다. 이 모델은 에너지 소비를 줄이고 효율적인 신경 세포 배치를 실현하여, 국가 최전선의 스파이킹 신경망 성능을 달성했습니다.



### Do LLM Personas Dream of Bull Markets? Comparing Human and AI Investment Strategies Through the Lens of the Five-Factor Mod (https://arxiv.org/abs/2411.05801)
- **What's New**: 대형 언어 모델(LLMs)이 특정 Big Five 성격 프로필을 가진 개인의 투자 작업 수행을 조사한 연구가 새롭게 발표되었습니다. LLM의 성격이 인간의 행동과 유사하게 나타나는지를 평가하여 LLM의 효용성을 보여줍니다.

- **Technical Details**: 연구에서는 LLM의 성격을 다섯 가지 성격 특성(개방성, 성실성, 친화성, 외향성, 신경성)에 기반한 243개의 독특한 페르소나로 구축하였습니다. 이들은 9개의 행동적 질문을 통해 인간의 성격과 행동의 연관성을 평가하였으며, ChainGPT 4.0을 사용하여 보다 신뢰할 수 있는 결과를 도출했습니다.

- **Performance Highlights**: LLMs는 학습 스타일, 충동성 및 위험 감수성과 같은 분야에서 인간의 행동 연구에서 기대된 대로 성격 특성을 일반화하는데 성공하였으나, 환경에 대한 태도는 정확하게 표현되지 못했습니다. 또한, 시뮬레이션 환경에서 LLM의 행동은 설문 환경에서보다 인간의 행동을 더 잘 반영했습니다.



### NeoPhysIx: An Ultra Fast 3D Physical Simulator as Development Tool for AI Algorithms (https://arxiv.org/abs/2411.05799)
Comments:
          7 Pages, 4 Figures

- **What's New**: NeoPhysIx는 3D 물리 시뮬레이터로, 기존의 AI 알고리즘들이 갖는 계산 자원의 한계를 극복하고 1000배 이상의 속도를 구현하였습니다. 이 시뮬레이터는 로봇 시뮬레이션을 위한 혁신적인 방법론을 채택하고 있으며, 점 구름 충돌 감지, 관절 각도 계산 및 마찰력 추정의 간소화된 접근을 통해 성능을 극대화하였습니다.

- **Technical Details**: NeoPhysIx는 단일 코어의 Intel i5 프로세서에서 초당 최대 1백만 프레임을 처리할 수 있는 성능을 자랑합니다. 이 시뮬레이터는 로봇 모델을 질량 포인트의 집합으로 구성하고, 환경의 높이 맵을 사용하여 충돌 감지를 단순화하는 방식을 채택했습니다. 이러한 혁신적인 알고리즘들은 저비용으로도 현실적인 행동을 가능하게 합니다.

- **Performance Highlights**: NeoPhysIx는 18도의 자유도를 가진 다리 로봇의 시뮬레이션을 통해 6개월의 로봇 생애를 단 9시간 만에 처리할 수 있었습니다. 이는 기존의 시뮬레이션 방식들과 비교했을 때 비약적인 효율성을 보여줍니다. 이러한 성과는 AI 개발에 있어 물리적 기반을 갖춘 영역에서의 훈련 과정을 가속화할 수 있는 가능성을 제시합니다.



### A Genetic Algorithm for Multi-Capacity Fixed-Charge Flow Network Design (https://arxiv.org/abs/2411.05798)
- **What's New**: MFCNF 문제에 대한 새로운 유전 알고리즘 제안으로, 흐름 해법을 수치 배열로 표현하여 기존의 수리적 수리기법을 대체함으로써 효율성을 증대시킴.

- **Technical Details**: 문서에서는 다중 용량 고정 비용 네트워크 흐름 문제(MC-FCNF)를 정의하고 이를 정수 선형 프로그래밍(ILP)으로 공식화합니다. 유전 알고리즘은 선형 프로그래밍(LP) 수정 및 흐름 해법을 생성하는 데 사용되는 새로운 스케일링 파라미터를 도입합니다.

- **Performance Highlights**: 진화된 유전 알고리즘을 사용하여 실제 CO2 포집 및 저장 인프라 설계 데이터로 평가된 결과, 대규모 네트워크에 대한 최적 해법을 신속하게 찾을 수 있는 잠재력을 보여줌.



### A Comprehensive Survey of Time Series Forecasting: Architectural Diversity and Open Challenges (https://arxiv.org/abs/2411.05793)
Comments:
          Submitted to the Artificial Intelligence Review on October 10, 2024

- **What's New**: 이번 논문에서는 시계열 예측(time series forecasting)의 다양한 아키텍처의 발전 및 구조적 다양화에 대한 포괄적인 분석을 제공합니다. 특히, Transformer 모델이 장기 의존성(long-term dependencies)을 처리하는 데 우수하다는 점을 강조하며, 최근 단순 선형 모델이 Transformer보다 더 나은 성능을 보인 사례를 소개합니다.

- **Technical Details**: 시계열 예측은 시퀀스 형태의 역사적 데이터를 기반으로 미래 값을 예측하는 작업으로, 경제, 금융, 물류 등 여러 분야에서 활용됩니다. 기존에는 MLPs, RNNs, CNNs, GNNs와 같은 딥러닝 아키텍처가 사용되었으나, 각 구조의 귀납적 편향(inductive biases)으로 인한 성능 한계가 있었습니다. 이 연구는 다양한 아키텍처의 진화를 비교하고, 하이브리드 모델, 확산 모델(diffusion models), Mamba 모델과 같은 최신 트렌드 및 도전 과제를 찾아냅니다.

- **Performance Highlights**: 시계열 데이터의 복잡성과 다양성으로 인해 기존 모델의 일반화 성능이 제한되어 왔습니다. 그러나 본 논문에서는 구조적 다양성을 통해 연구자들이 시계열 예측 문제에 대한 새로운 접근을 시도할 수 있도록 하여, 새로운 연구 기회와 깊은 통찰력을 제공합니다.



New uploads on arXiv(cs.LG)

### DeepONet as a Multi-Operator Extrapolation Model: Distributed Pretraining with Physics-Informed Fine-Tuning (https://arxiv.org/abs/2411.07239)
- **What's New**: 이번 연구는 다양한 함수 데이터를 활용하여 분산 신경 연산자(Distributed Neural Operator)를 훈련시키고, 물리 정보를 이용한 손실 함수(physics-informed losses)를 통해 제로샷(fine-tuning) 미세 조정을 수행하는 새로운 미세 조정 방법을 제안합니다.

- **Technical Details**: 이 접근법은 여러 운영자(operator)로부터 데이터를 통합하는 분산 학습(distributed learning) 방식을 활용하여 훈련해, 새로운 작업에 신속히 적응할 수 있는 초기화(initialization)를 선택합니다. 제안된 두 가지 미세 조정 방식은 표준 미세 조정(standard fine-tuning)과 저랭크 적응(fine-tuning with Low-Rank Adaptation)이며, 복잡한 비선형 목표 운영자를 훈련시키는 데 사용합니다.

- **Performance Highlights**: 본 연구의 실험을 통해 제시한 방법이 정확성을 크게 개선하는 것을 보여주며, 복수 운영자 학습(multi-operator learning)을 향상시키고 전이 학습 기법의 잠재력을 강조합니다.



### Score-based generative diffusion with "active" correlated noise sources (https://arxiv.org/abs/2411.07233)
Comments:
          18 pages, 11 figures

- **What's New**: 본 연구에서는 전통적인 Gaussian white noise 대신, active (시간적으로 상관된) noise를 생성적 확산 모델의 forward 과정에 도입하여 생성 성능을 향상시킬 수 있는 가능성을 탐색합니다.

- **Technical Details**: 활성 화음 (active noise) 개념을 통해 데이터의 상관성을 파괴함으로써 새로운 하이퍼파라미터를 도입하며, 이는 다양한 데이터 분포에서 훈련 및 샘플링 효율성을 최적화하는 데 도움이 됩니다. 이 연구는 score-based diffusion models를 기반으로 하며, 그 과정에서 실제 물리적 모델 대신 활성 물질 시스템에서 영감을 받아 다차원 Gaussian 분포로의 변화 과정을 다룹니다.

- **Performance Highlights**: 활성 noise를 적용한 새로운 확산 모델은 기존 방안과 비교하여 더 나은 생성 성능을 보이며, 데이터 분포에 최적화된 학습 및 샘플링 효율성을 제공하여 심층 학습 (deep learning) 구조와 결합되었을 때 원래 데이터와 유사한 새로운 샘플을 생성하는 데 효과적임을 입증합니다.



### Feature Selection Based on Wasserstein Distanc (https://arxiv.org/abs/2411.07217)
- **What's New**: 이번 논문에서는 Wasserstein 거리(Wasserstein distance)를 기반으로 한 새로운 특성 선택(feature selection) 방법을 제안합니다. 이는 입력 데이터의 차원 축소를 통해 머신 러닝의 효율성 및 일반화 성능을 향상시키는 데 중요한 역할을 합니다.

- **Technical Details**: 기존의 특성 선택 기법들은 상관관계(correlation)나 KL 발산(KL divergence)과 같은 기준에 의존하는 반면, 본 방법은 선택된 특성과 원래 특성 사이의 분포 유사성을 측정하기 위해 Wasserstein 거리를 활용합니다. 이 접근 방식은 클래스 간 유사성을 본질적으로 고려하여 노이즈가 있는 레이블의 경우에도 견고합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 방법들과 비교하여 특히 노이즈가 있는 레이블 데이터와 같은 어려운 설정에서 더 나은 성능을 보이는 것으로 나타났습니다.



### Comparing Bottom-Up and Top-Down Steering Approaches on In-Context Learning Tasks (https://arxiv.org/abs/2411.07213)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 해석 가능성(interpretable) 연구에서 두 가지 접근 방식, 즉 "하향식(top-down)"과 "상향식(bottom-up)" 방법을 비교하여 각 방법의 효과를 분석한 연구입니다. ICV(인-컨텍스트 벡터)와 FV(함수 벡터) 두 가지 대표적인 벡터 조정 방식이 서로 다른 과제에서 어떻게 작용하는지를 평가하였습니다.

- **Technical Details**: 기존의 개별 평가 결과들이 매우 상이한 과제들에 의한 것이기 때문에, 두 방법의 상대적 강점을 이해하는 데 어려움이 있었습니다. ICV는 하향식 접근으로, 특정 행동에 대한 대조적 예제를 통해 생성된 벡터로 넓은 개념을 포착합니다. 반면 FV는 상향식 방법으로, 특정 주의 헤드를 식별하여 고성능의 인-컨텍스트 학습 태스크를 조정합니다. 각 방법은 7개의 다양한 과제를 통해 평가되었으며, 정밀성 요구 과제에서는 FV가 뛰어난 성능을 보였습니다.

- **Performance Highlights**: ICV는 폭넓은 행동 전환에서 FV보다 우수하지만, 정밀한 작업에서는 FV가 더 효과적입니다. ICV는 일반적으로 정확도에서 대폭적인 향상을 보여주지만, 정서 전이(task)는 FV의 성능이 더 우수하였습니다. FV는 새로운 상황에서도 잘 작용하는 반면, ICV는 설정에 따라 결과의 변동이 크고 유동성을 낮출 수 있는 경향이 있습니다.



### General Geospatial Inference with a Population Dynamics Foundation Mod (https://arxiv.org/abs/2411.07207)
Comments:
          28 pages, 16 figures, preprint

- **What's New**: 이번 연구에서는 Population Dynamics Foundation Model (PDFM)을 소개하여 정부 기관과 연구자들이 복잡한 인간 행동과 지역 환경 간의 관계를 이해하는 데 도움을 줄 수 있는 새로운 모델을 제안합니다. 이 모델은 데이터 모달리티 간의 관계를 캡처하고 다양한 geospatial 작업에 적용될 수 있도록 설계되었습니다.

- **Technical Details**: PDFM은 미국의 우편번호와 카운티에 대한 geo-indexed 데이터를 제작하고, 그래프 신경망 (Graph Neural Network)으로 모델링하여 위치 간의 복잡한 관계를 학습합니다. 이 데이터는 인구 행동에 대한 정보와 환경 요소를 포함하여 다양한 downstream 작업에 쉽게 적용될 수 있는 embeddings를 생성합니다.

- **Performance Highlights**: PDFM은 27개의 geospatial interpolation 작업에서 state-of-the-art 성능을 달성했으며, 27개 작업 중 25개의 extrapolation과 super-resolution 작업에서도 우수한 결과를 보여주었습니다. 이 모델은 기존의 supervised forecasting을 능가하여 실업과 빈곤 예측에서 뛰어난 성능을 발휘합니다.



### Gradual Fine-Tuning with Graph Routing for Multi-Source Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.07185)
Comments:
          In Proceedings of the 3rd Conference on Lifelong Learning Agents (CoLLAs 2024)

- **What's New**: 이 논문에서는 다중 소스의 비지도 도메인 적응(multi-source unsupervised domain adaptation)을 위한 새로운 프레임워크인 점진적인 미세 조정(Gradual Fine Tuning, GFT)을 제안합니다. GFT는 여러 소스 도메인에서 머신러닝 모델을 훈련하여 타겟 도메인에서의 일반화를 촉진합니다.

- **Technical Details**: 제안된 GFT 프레임워크는 소스 도메인의 가중치가 적용된 무방향 그래프(undirected weighted graph)로 표현되며, 이는 최적의 훈련 순서에 맞는 최적 경로(optimal path)를 결정하는 데 사용됩니다. GFT의 일반화 오차 경계(generalization error bound)를 제시하며, 이를 통해 모델의 성능을 향상시키기 위한 세 가지 경량화된 그래프 라우팅(graph-routing) 전략을 도입합니다.

- **Performance Highlights**: GFT의 최적 전략은 Natural Language Inference (NLI) 작업에서 기존 연구보다 2.3% 높은 정확도를 달성했으며, Sentiment Analysis (SA) 작업에서도 경쟁력 있는 성능을 보여 주었습니다. 특히 SA의 다양한 데이터 서브셋에서는 3.9%의 개선을 이끌어냈습니다.



### Revisiting Ensembling in One-Shot Federated Learning (https://arxiv.org/abs/2411.07182)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이번 논문은 Federated Ensembling Scheme(FENS)이라는 새로운 기법을 소개합니다. 이는 One-shot Federated Learning (OFL)의 통신 효율성과 Iterative Federated Learning (FL)의 정확성을 동시에 향상시키는 것을 목표로 합니다.

- **Technical Details**: FENS는 두 단계로 진행됩니다. 첫 번째 단계에서는 클라이언트들이 로컬 모델을 서버에 전송합니다. 두 번째 단계에서는 클라이언트들이 FL을 통해 경량화된 예측 집계 모델을 협력적으로 학습합니다. 이 방식은 OFL의 전통적인 집계 방법 대신, 예측 집계 모델을 활용하여 정확도를 높입니다.

- **Performance Highlights**: FENS는 CIFAR-10 데이터셋에서 SOTA OFL에 비해 최대 26.9% 높은 정확도를 달성했으며, FL에 비해서는 3.1% 낮은 정확도를 기록했습니다. 또한, FENS는 OFL에 비해 최대 4.3배 더 많은 통신 비용이 소요되며, FL에 비해서는 최소 10.9배 더 저렴한 통신 비용을 기록했습니다.



### Anytime Sequential Halving in Monte-Carlo Tree Search (https://arxiv.org/abs/2411.07171)
Comments:
          Accepted by the Computers and Games 2024 conference

- **What's New**: 본 논문에서는 Monte-Carlo Tree Search (MCTS)에서 선택 전략으로 사용할 수 있는 anytime Sequential Halving 알고리즘을 제안합니다. 이 알고리즘은 별도의 사전 예산이 필요하지 않으며, 언제든지 중단할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: anytime SH (Sequential Halving)는 Multi-Armed Bandit (MAB) 알고리즘으로, 기본적으로 원래의 SH를 뒤집은 형태입니다. 이 알고리즘은 MCTS의 root node에서 선택 전략으로 사용될 수 있도록 설계되었습니다.

- **Performance Highlights**: MAB 문제와 10개의 다양한 보드 게임에서 empirical (경험적) 실험 결과는 anytime SH의 성능이 UCB1 및 기존의 SH와 경쟁할 만하다는 것을 보여줍니다.



### Enhancing Predictive Maintenance in Mining Mobile Machinery through a TinyML-enabled Hierarchical Inference Network (https://arxiv.org/abs/2411.07168)
Comments:
          This work has been submitted to the IEEE Access for possible publication

- **What's New**: 본 논문은 Edge Sensor Network for Predictive Maintenance (ESN-PdM)를 제안하여 다양한 환경에서 동작하는 광산 기계의 예측 유지보수(Predictive Maintenance, PdM)에 대한 새로운 접근법을 제공합니다. 이 시스템은 실시간 상태 모니터링을 위해 엣지 장치, 게이트웨이 및 클라우드 서비스 간의 계층적 추론 프레임워크를 통합하며, 정확도, 지연(latency), 배터리 수명 간의 트레이드오프를 고려하여 추론 위치를 동적으로 조정합니다.

- **Technical Details**: ESN-PdM 시스템은 Tiny Machine Learning (TinyML) 기술을 활용하여 자원 제약이 있는 장치에서 모델 최적화를 수행합니다. 성능 평가 결과, 온-센서 및 온-게이트웨이 추론 모드에서는 90% 이상의 분류 정확도를 달성하였으며, 클라우드 기반 추론은 99%에 도달했습니다. 특히, 온-센서 추론은 약 44%의 전력 소비 감소를 보여 104시간의 작동을 가능하게 합니다. 온-디바이스 추론 시 지연 시간은 가장 낮은 3.33 ms로, 게이트웨이(146.67 ms) 및 클라우드(641.71 ms)로 오프로드할 때 증가합니다.

- **Performance Highlights**: ESN-PdM 프레임워크는 신뢰성 있는 이상 탐지 및 PdM을 위한 확장 가능하고 적응력 있는 솔루션을 제공합니다. 이 노하우는 가혹한 환경에서의 기계 가동 비율 유지에 중요한 역할을 하고 있으며, 정확도, 지연 및 에너지 소비 간의 균형을 맞춰 산업 응용을 위한 PdM 프레임워크를 발전시킵니다.



### Variational Graph Contrastive Learning (https://arxiv.org/abs/2411.07150)
- **What's New**: 이번 논문에서는 Subgraph Gaussian Embedding Contrast (SGEC) 방법론을 제안합니다. SGEC는 서브그래프를 구조화된 Gaussian 공간으로 능동적으로 매핑하여 그래프의 특성을 보존하면서 생성된 서브그래프의 분포를 제어하는 모듈을 갖추고 있습니다.

- **Technical Details**: SGEC는 Graph Representation Learning (GRL)에 사용되는 Self-supervised Learning (SSL) 방법으로, 서브그래프의 특징을 Gaussian 분포로 매핑하고 이것을 최적 전달 거리(optimal transport distances)인 Wasserstein 및 Gromov-Wasserstein 거리를 통해 평가합니다. 이 과정에서 Kullback-Leibler (KL) 발산을 사용하여 매핑의 정규화를 수행합니다.

- **Performance Highlights**: SGEC는 8개의 벤치마크에서 기존의 최신 방법들과 비교하여 더 나은 성능을 보여주었습니다. 우리의 실험은 생성된 대조 쌍의 분포가 GRL 방법 설계에 중요한 영향이 있음을 강조합니다.



### Fast and Robust Contextual Node Representation Learning over Dynamic Graphs (https://arxiv.org/abs/2411.07123)
- **What's New**: 이 논문은 동적 그래프에서의 노드 표현 유지 문제를 다루며, 기존의 PPR (Personalized PageRank) 기반의 GNN 모델을 개선할 수 있는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 모델은 희소 노드 간의 주의(attention)의 기반을 두고 있으며, proximal gradient method(ISTA)를 활용하여 PPR 기법의 효율성을 최대 6배 향상시킵니다. 또한, PPR이 GNN의 기초적인 도구로 기능할 수 있도록 여러 속성을 세분화하고 정의합니다.

- **Performance Highlights**: 모델 GoPPE는 간단하면서도 효과적인 포지셔널 인코딩을 사용하여, 기존의 최신 기술들(STOA)과 비교하여 유사한 성능을 보이며, 노이즈가 많은 초기 노드 속성에서 그래프가 진화할 때 성능이 우수합니다.



### Efficient Adaptive Optimization via Subset-Norm and Subspace-Momentum: Fast, Memory-Reduced Training with Convergence Guarantees (https://arxiv.org/abs/2411.07120)
- **What's New**: 이 논문은 대규모 신경망 훈련의 메모리 요구 사항을 줄이면서도 효율적인 적응형 최적화를 위해 두 가지 보완 기법을 소개합니다. 첫 번째 기법은 Subset-Norm 적응형 스텝 크기이며, 두 번째 기법은 Subspace-Momentum입니다.

- **Technical Details**: Subset-Norm(SN)은 모델 크기 d에 대해 O(d)에서 O(√d)로 메모리 발자국을 줄이는데, 이는 AdaGrad와 AdaGrad(-Coordinate)의 일반화를 사용합니다. Subspace-Momentum(SM)은 모멘텀 상태를 낮은 차원의 부분공간에서 수행합니다. 두 기법 모두 기존 방법에 비해 향상된 차원 의존성과 높은 확률의 수렴 보장을 제공합니다.

- **Performance Highlights**: LLaMA 모델(60M에서 1B 파라미터)에 대한 실험 평가에서, Subset-Norm과 Subspace-Momentum을 결합한 AdamSNSM이 Adam의 유효성 평가에서 약 절반의 훈련 토큰(6.8B 대 13.1B)을 사용하면서도, Adam의 메모리 점유율의 20%만을 소모하고 최소한의 추가 하이퍼파라미터 조정으로 성과를 냈습니다.



### Differentially-Private Collaborative Online Personalized Mean Estimation (https://arxiv.org/abs/2411.07094)
Comments:
          Presented in part at the 2023 IEEE International Symposium on Information Theory (ISIT)

- **What's New**: 본 논문에서는 여러 에이전트가 각기 다른 분포에서 데이터를 수신하는 환경에서 개인화된 평균 추정(collaborative personalized mean estimation)의 문제를 다루고 있습니다. 이 과정은 프라이버시 제한을 고려하여 진행됩니다.

- **Technical Details**: 제안된 방법은 가설 검정(hypothesis testing)과 차별적 프라이버시(differential privacy), 데이터 분산 추정(data variance estimation)에 기반하고 있습니다. 두 가지 프라이버시 메커니즘과 두 가지 데이터 분산 추정 기법이 제안되며, 모든 제한된(unknown) 분포에서의 이론적인 수렴 분석이 제공됩니다.

- **Performance Highlights**: 협력적인 접근 방식이 데이터를 공유하지 않는 순전히 로컬(local) 접근 방식보다 더 빠른 수렴(faster convergence)을 제공함을 보여줍니다. 실험 결과, 제안된 접근 방식은 순전히 로컬 접근 방식보다 훨씬 빠르게 수렴하고 모든 데이터가 공개된 이상적인 성능과 유사한 성능을 발휘함을 나타내었습니다.



### Universal Response and Emergence of Induction in LLMs (https://arxiv.org/abs/2411.07071)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)에서 유도(induction) 행동의 출현을 조사하여, 모델의 잔여 스트림(residual stream)에 대한 약한 단일 토큰의 변화를 탐색합니다. 이를 통해 유도 신호의 양적 특성을 제공하여 LLM의 동작을 더 잘 이해할 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 Gemma-2-2B, Llama-3.2-3B, GPT-2-XL 모델에서 잔여 스트림에 대한 약한 변화를 통해 유도 행동의 신호를 탐지하였습니다. 저희는 이 모델들이 변화 강도에 관계없이 반응이 비례(scale-invariant)하게 유지되는 보편적인 영역을 갖고 있음을 발견했습니다. 이 방법은 사용자가 모델의 각 레이어에서 유도 행동의 구성 요소를 더 깊이 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 모델의 중간 레이어에서 유도 신호가 점진적으로 나타나면서, 유도 행동을 구성하는 중요 모델 섹션들을 식별할 수 있었습니다. 이 발견은 대규모 회로 분석(large-scale circuit analysis)을 위한 기준이 될 수 있으며, LLM의 내부 상호작용에 대한 통찰력을 제공합니다.



### Zeroth-Order Adaptive Neuron Alignment Based Pruning without Re-Training (https://arxiv.org/abs/2411.07066)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 네트워크 프루닝(Network Pruning) 기술을 활용하여, LLM(대형 언어 모델)들에 대해 성능 저하 없이 매개변수를 줄일 수 있는 새로운 알고리즘 NeuroAl을 제안합니다. 이 알고리즘은 'top-up' 접근 방식을 사용해, 기존의 여러 프루닝 기법에 쉽게 적용될 수 있습니다.

- **Technical Details**: NeuroAl 알고리즘은 Activations의 일관성을 높이기 위해 블록 및 로우별 희소성 비율을 수정하는 두 단계의 방식으로 구성됩니다. 이 과정에서 사용자에게 하이퍼파라미터를 지정할 필요가 없으며, 모델의 구조와 입력된 희소성 요구에 따라 자동으로 최적의 값을 선택합니다. 이를 통해 프루닝 후에도 과거 프루닝 기법보다 나은 성능을 보장합니다.

- **Performance Highlights**: 4개의 서로 다른 LLM 계열에서 3가지 희소성 비율에 대해 테스트한 결과, NeuroAl은 최신 기술인 OWL 및 DsNoT보다 지속적으로 우수한 성능을 보였습니다. 실험 결과, 60%, 70%, 80%의 높은 희소성에서도 안정적인 성능을 발휘하며, 심도 있는 Ablation Study를 통해 알고리즘의 강건성을 입증했습니다.



### General framework for online-to-nonconvex conversion: Schedule-free SGD is also effective for nonconvex optimization (https://arxiv.org/abs/2411.07061)
Comments:
          Comments would be appreciated!

- **What's New**: 이번 연구는 A. Defazio 등이 개발한 schedule-free 방법의 비선형 최적화 문제에 대한 효과를 탐구하고, 이를 통해 schedule-free SGD가 비매끄럽고 비볼록(nonconvex) 최적화 문제에 대해 최적의 반복 복잡성을 달성함을 보여줍니다.

- **Technical Details**: 비선형 최적화를 위한 온라인 학습 알고리즘을 변환하는 일반적인 프레임워크를 개발하고, 이를 통해 일정이 없는 SGD(schedule-free SGD) 알고리즘을 도출했습니다. 이 알고리즘은 세 가지의 업데이트 시퀀스를 유지하며, 이는 SGD 궤적과 관련이 있고, 해당 알고리즘의 차별성을 제공합니다.

- **Performance Highlights**: schedule-free SGD는 비매끄럽고 비볼록 최적화 문제에 대해 최적의 성능을 보이며, 특정 파라미터 선택을 통한 효율성을 높일 수 있음을 보여줍니다. 특히, 한국어와 같은 대규모 언어 모델 훈련 시 안정성과 빠른 수렴을 보장할 수 있습니다.



### HeteroSample: Meta-path Guided Sampling for Heterogeneous Graph Representation Learning (https://arxiv.org/abs/2411.07022)
Comments:
          11 pages

- **What's New**: HeteroSample은 IoT 및 기타 복잡한 시스템을 위한 이종 그래프에서 구조적 무결성과 의미적 풍부성을 보존하도록 설계된 새로운 샘플링 방법입니다. 이 방법은 대표적인 노드를 선택하기 위해 새로운 top-leader 선택 전략, 균형 잡힌 이웃 확장 및 메타 경로 안내 샘플링 전략을 통합합니다.

- **Technical Details**: HeteroSample은 세 가지 주요 단계를 통해 작동합니다: (1) 각 노드 유형에 대한 가장 영향력 있는 노드를 식별하고, (2) 이러한 top-leaders의 이웃을 균형 있게 확장하며, (3) 메타 경로를 활용하여 노드 간의 의미적 관계를 포착합니다. 이 접근법은 원본 그래프의 구조적 및 의미적 속성을 보존하면서 계산 오버헤드를 줄이는 데 중점을 둡니다.

- **Performance Highlights**: HeteroSample은 링크 예측 및 노드 분류와 같은 작업에서 최대 15% 더 높은 F1 점수를 달성하고 실행 시간을 20% 줄이며 기존 방법들을 능가하는 성능을 보였습니다. 이러한 장점 덕분에 HeteroSample은 확장 가능하고 정확한 IoT 애플리케이션을 위한 혁신적인 도구로 자리 잡고 있습니다.



### Leveraging LSTM for Predictive Modeling of Satellite Clock Bias (https://arxiv.org/abs/2411.07015)
Comments:
          6 Pages, 6 figures (8 sub-figures), 5 Tables Index Terms-LSTM, Satellite Navigation, Deep Learning, Clock Bias

- **What's New**: 위성 시계 편차 예측의 정확성을 증가시키기 위해 Long Short-Term Memory (LSTM) 네트워크를 활용한 새로운 접근 방식을 제안합니다. 기존의 방법에 비해 LSTM 모델이 RNN보다 170배, MLP보다 2.3 × 10^7배, ARIMA보다 1.9 × 10^4배 더 높은 정확도를 보였습니다.

- **Technical Details**: 본 연구는 Galileo의 PRN 2 위성에서 수집한 데이터를 사용하여 단일 차분 시퀀스를 생성하고 이를 통해 예측 정확도를 높입니다. LSTM 모델은 7일부터 31일까지의 데이터셋 길이에 대해 훈련되며, RMSE(평균 제곱근 오차)를 주요 평가 지표로 사용합니다.

- **Performance Highlights**: LSTM 모델의 RMSE는 2.11 × 10^{-11}로, 전통적인 시계열 예측 방법에 비해 현저한 향상을 보였습니다. 이 연구 결과는 특히 저전력 수신기에서 정확도와 효율성을 높이는 데 기여할 것으로 기대됩니다.



### A neural-network based anomaly detection system and a safety protocol to protect vehicular network (https://arxiv.org/abs/2411.07013)
Comments:
          Master's thesis 2023-2024

- **What's New**: 이 논문은 Cooperative Intelligent Transport Systems (CITS)의 사용을 통해 도로 안전과 효율성을 증진하기 위한 방법을 제안합니다. 특히, 차량 간 통신을 통해 안전하고 정확한 데이터 교환의 중요성을 강조합니다.

- **Technical Details**: CITS는 차량 간 통신을 통해 사고 예방 및 교통 흐름 최적화에 기여합니다. 논문에서는 Long Short-Term Memory (LSTM) 네트워크를 사용하는 Machine Learning 기반의 Misbehavior Detection System (MDS)을 제안하고, VeReMi 데이터셋을 기반으로 오프라인에서 훈련시킨 후, 실제 시나리오에서 테스트합니다.

- **Performance Highlights**: MDS는 잘못된 메시지로 인한 사고를 거의 모두 예방할 수 있으며, 이상 현상이 감지되면 방어 프로토콜을 발동하여 차량 무리를 해체합니다. 그러나 다양한 교통 상황에서 특정 유형의 잘못된 행동을 식별하는 데 어려움을 겪고 있어 보편적인 적응 프로토콜을 만드는 것이 어렵다는 결과를 보였습니다.



### Hierarchical Conditional Tabular GAN for Multi-Tabular Synthetic Data Generation (https://arxiv.org/abs/2411.07009)
- **What's New**: 이 논문에서는 복잡한 다중 표 형식의 데이터셋에서 합성 데이터를 효율적으로 생성하기 위한 HCTGAN 알고리즘을 제안합니다. 기존의 연구는 주로 단일 표 데이터셋에 집중되어 있었으며, 다중 표 데이터셋에 대한 효율적인 알고리즘이 부족했습니다.

- **Technical Details**: HCTGAN(계층 조건 표 GAN)은 CTGAN 모델의 확장 버전으로, 부모 테이블과 자식 테이블 생성 네트워크 간의 정보 전송을 통해 관계 학습을 촉진합니다. 이 모델은 훈련 및 샘플링을 위한 두 가지 새로운 알고리즘을 도입하며, 이러한 알고리즘은 대규모 및 복잡한 다중 표 데이터셋을 합성하는 데 적합합니다.

- **Performance Highlights**: HCTGAN 알고리즘은 대규모 합성 데이터를 더 효율적으로 샘플링할 수 있으며, 데이터 품질을 적절히 유지하면서 참조 무결성을 보장합니다. 결과적으로 복잡한 관계를 가진 깊은 다중 표 데이터셋에 대해 효율적으로 대량의 합성 데이터를 생성할 수 있는 적합한 알고리즘으로 결론을 내렸습니다.



### Non-Adversarial Inverse Reinforcement Learning via Successor Feature Matching (https://arxiv.org/abs/2411.07007)
- **What's New**: 본 연구는 Inverse Reinforcement Learning (IRL)에서의 새로운 접근 방법인 Successor Feature Matching (SFM)을 제안합니다. 이 기법은 전문가의 행동을 복제할 때 보상 함수 학습이 필요하지 않으며, 단 하나의 전문가 시연만으로도 효과적으로 학습할 수 있습니다.

- **Technical Details**: SFM은 상태만을 이용하여 기대 누적 특성을 추정하는 데 Successor Features (SF)를 활용하여, 정책 경량화(Policy Gradient Descent) 방법을 통해 학습합니다. 이 방식은 전통적인 적대적 접근 방식과 달리 안정적인 솔루션을 제공하며, Actor-Critic RL 알고리즘과 통합이 용이합니다.

- **Performance Highlights**: SFM은 다양한 제어 작업에서 평균 정규화 수익률이 16% 향상된 성능을 보여주었고, 단 하나의 전문가 시연으로도 효과적인 모방 학습이 가능함을 입증했습니다.



### Imitation from Diverse Behaviors: Wasserstein Quality Diversity Imitation Learning with Single-Step Archive Exploration (https://arxiv.org/abs/2411.06965)
- **What's New**: 이번 연구에서는 Wasserstein Quality Diversity Imitation Learning (WQDIL)을 소개하여, 제한된 시연으로부터 다양한 고품질 정책을 배우는 데 필요한 혁신적인 접근 방식을 제시합니다. 이 방법은 Wasserstein Auto-Encoder (WAE)를 기반으로 잠재적 적대적 훈련을 통해 모방 학습의 안정성을 향상시키고, 행동 과적합 이슈를 완화하기 위해 측정 조건 보상 함수를 사용합니다.

- **Technical Details**: WQDIL은 Wasserstein Auto-Encoder (WAE) 내의 잠재 공간에서 Wasserstein 적대적 훈련을 적용하여 보상 모델의 안정성을 높입니다. 또한, 단일 스텝 아카이브 탐사를 통한 보너스를 도입하여 에이전트가 보다 다양한 행동을 수집하도록 유도하는 구조입니다. 이로써 교육이 불안정하고 행동이 과적합되는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, WQDIL은 최신 모방 학습(IL) 방법들을 훨씬 초월하는 성능을 기록하며, MuJoCo 환경에서 파생된 도전적인 연속 제어 작업에서 전문가 수준의 QD 성능을 달성하였습니다.



### Efficient Unsupervised Domain Adaptation Regression for Spatial-Temporal Air Quality Sensor Fusion (https://arxiv.org/abs/2411.06917)
- **What's New**: 최근 IoT 센서를 사용한 대기 오염 모니터링의 발전에도 불구하고, 저비용 센서의 정확한 보정은 여전히 큰 도전 과제로 남아 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해 새로운 비지도 도메인 적응 (Unsupervised Domain Adaptation, UDA) 방법을 제안합니다. 이 방법은 회귀 작업을 위해 그래프 구조의 데이터를 활용하며, Graph Neural Networks (GNNs)를 사용하여 센서 간의 관계를 모델링합니다.

- **Technical Details**: 우리는 Spatial-Temporal Graph Neural Networks (STGNNs)를 도입하여 시공간 상호작용을 포착합니다. 또한, Tikhonov 정규화된 최소 제곱 방법에 기반한 도메인 적응 방법을 사용하여 소스 도메인과 타겟 도메인 간의 서브스페이스를 정렬합니다. 이는 Cholesky 분해와 파워 반복 알고리즘을 활용하여 이루어집니다.

- **Performance Highlights**: 제안된 TikUDA 방법은 기존의 비지도 도메인 적응 방법에 비해 빠르고 확장성이 뛰어나며, 저비용 IoT 센서가 고비용 기준 센서로부터 보정 매개변수를 배우는 데 도움을 주어 새로운 위치에서의 신뢰할 수 있는 오염 물질 측정을 가능하게 합니다.



### Slowing Down Forgetting in Continual Learning (https://arxiv.org/abs/2411.06916)
- **What's New**: 본 연구는 연속 학습(Continual Learning)에서 재난적인 망각(catastrophic forgetting)을 완화하기 위한 새로운 프레임워크인 ReCL(Reconstruction from Continual Learning)을 제안합니다. 이 프레임워크는 기초적인 뉴럴 네트워크 학습의 암묵적 편향을 이용하여 이전 태스크의 데이터를 재구성하고, 이를 현재의 학습 데이터와 결합하여 망각을 느리게 합니다.

- **Technical Details**: ReCL 프레임워크는 이전 태스크에서 훈련한 데이터를 네트워크 파라미터의 최적화 과정에서 메모리처럼 활용합니다. 이를 통해 개선된 성능을 다양한 CL 시나리오(예: class incremental learning, domain incremental learning, task incremental learning)와 데이터셋(MNIST, CIFAR10)에서 입증하였습니다. 또한, 다층 퍼셉트론(MLP)과 합성곱 신경망(CNN) 구조에서도 성능 향상이 관찰되었습니다.

- **Performance Highlights**: 모든 실험에서 ReCL 프레임워크는 기존의 CL 방법들과 결합할 때에도 일관된 성능 향상을 보여줍니다. 이 연구는 또한 EWC, ER, UPGD와 같은 최신 CL 기법들과 결합했을 때, 특히 망각을 더욱 줄이는 결과를 보였습니다.



### SPARTAN: A Sparse Transformer Learning Local Causation (https://arxiv.org/abs/2411.06890)
- **What's New**: 본 연구에서는 SPARse TrANsformer World model (SPARTAN)이라는 새로운 모델을 제안합니다. 이 모델은 환경의 다양한 변경 사항에 적응할 수 있는 지역적 인과 구조를 학습합니다. 최신의 object-centric world models와 비교하여 더 정확하고 해석 가능성이 높은 결과를 보여줍니다.

- **Technical Details**: SPARTAN은 Transformer 기반의 세계 모델로서 객체-팩터(token) 간의 주의(attention) 패턴에 대해 희소성(sparsity) 정규화를 적용하여 local causal structures를 발견합니다. 이를 통해 SPARTAN은 변화를 예측할 수 있는 희소한 지역 인과 모델을 식별합니다. 또한, 다양한 환경에서의 개입(intervention) 모델링을 위해 환경의 동적 변화를 명확히 포착합니다.

- **Performance Highlights**: SPARTAN은 복잡한 환경에서 미래의 객체 상태를 보다 정확하게 예측하며, 불필요한 방해 요소를 제거했을 때의 강인성(robustness) 또한 향상되었습니다. 여기서는 몇 번의 적응(few-shot adaptation)만으로도 환경의 변화에 잘 적응할 수 있는 성능을 보여줍니다.



### WassFFed: Wasserstein Fair Federated Learning (https://arxiv.org/abs/2411.06881)
Comments:
          Submitted to TKDE

- **What's New**: 본 논문에서는 사용자 데이터의 비공유 문제를 해결하기 위해 저자들이 제안하는 Wasserstein Fair Federated Learning 프레임워크인 WassFFed를 소개합니다. 이 프레임워크는 기존의 공정성 문제를 해결하기 위한 새로운 접근 방식을 채택하고 있으며, 공정한 글로벌 모델을 구축하기 위한 과제를 다룹니다.

- **Technical Details**: WassFFed는 두 가지 주요 문제를 해결합니다: (CH1) 서베이 함수(surrogate functions)로 얻어진 공정 최적화 결과와 공정 분류 결과 간의 불일치 문제를 해결하고, (CH2) 클라이언트 간 비동일하고 독립적인 데이터 분포(non-IID)로 인해 로컬 공정 모델을 직접 집계하는 것으로는 항상 글로벌 공정 모델을 생성하지 못하는 문제를 다룹니다. 이를 위해 WassFFed는 Wasserstein barycenter 계산을 활용하여 로컬 모델의 출력을 조정합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋을 기반으로 진행된 실험에서, WassFFed는 기존의 최첨단 방법들(State-Of-The-Art)보다 우수한 성능을 보이며 공정성과 정확성 간의 균형을 잘 맞춥니다. 특히, WassFFed는 더 복잡한 분류 작업에서도 뛰어난 일반화 능력을 발휘함을 입증했습니다.



### GraphRPM: Risk Pattern Mining on Industrial Large Attributed Graphs (https://arxiv.org/abs/2411.06878)
Comments:
          Accepted by ECML PKDD 2024

- **What's New**: 이 논문에서는 대규모 속성 그래프에서 위험 패턴을 발견하기 위한 새로운 프레임워크인 GraphRPM을 소개합니다. GraphRPM은 업계 특정으로 설계된 병렬 및 분산 위험 패턴 마이닝 프레임워크이며, 최신 Edge-Involved Graph Isomorphism Network (EGIN)과 최적화된 병렬 그래프 계산 연산을 통합하여 비약적인 성능 개선을 이뤘습니다.

- **Technical Details**: GraphRPM은 대규모 속성 그래프에서 위험 패턴을 효율적으로 발견하기 위해 두 단계의 마이닝 전략과 병렬 분산 처리 프레임워크를 구현합니다. 이 프레임워크는 EGIN을 기반으로 하여 속성 그래프 패턴의 모호한 매칭 문제를 해결하며, 연산의 복잡성과 정확성을 균형 있게 관리합니다. 또한, 효과적인 위험 그래프 패턴을 식별하기 위한 평가 메트릭스를 도입합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 포괄적인 실험 평가 결과, GraphRPM은 대규모 산업 속성 그래프에서 패턴 마이닝의 도전 과제를 성공적으로 해결하는 능력을 입증하였습니다. 이는 산업 적용을 위한 상당한 가치와 혁신을 제공하는 연구로, 데이터 마이닝 및 기계 학습의 응용 분야에서 중요한 진전을 이룩하였습니다.



### Subgraph Retrieval Enhanced by Graph-Text Alignment for Commonsense Question Answering (https://arxiv.org/abs/2411.06866)
Comments:
          Accepted by ECML PKDD 2024

- **What's New**: 본 논문에서는 Commonsense Question Answering (CSQA) 작업을 위한 새로운 프레임워크인 SEPTA를 제안합니다. SEPTA는 Knowledge Graph (KG)를 효율적으로 활용하여 소프트웨어가 공통 상식에 기반한 논리적 사고를 수행할 수 있도록 설계되었습니다.

- **Technical Details**: SEPTA는 Knowledge Graph를 서브그래프 벡터 데이터베이스로 변환하고, BFS 스타일의 샘플링 전략을 사용하여 정보 손실을 최소화합니다. 또한, 그래프와 텍스트 인코더 사이의 의미 공간을 정렬하기 위한 양방향 대비 학습 접근법을 제안하여 정보 통합을 효과적으로 개선합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에 대한 대규모 실험 결과, 제안된 SEPTA 프레임워크가 기존의 최첨단 방법들보다 우수한 성능을 보였으며, 약한 감독 설정에서도 유망한 성과를 달성했습니다.



### Computable Model-Independent Bounds for Adversarial Quantum Machine Learning (https://arxiv.org/abs/2411.06863)
Comments:
          21 pages, 9 figures

- **What's New**: 이 연구는 양자 기계 학습(QML)의 적대적 공격에 대한 내성을 평가하기 위한 모델 독립적인 경계(bound)를 첫 번째로 도입하고, 고급 양자 기반 적대적 공격에 대한 근사적 하한을 계산하여 QML 모델의 본질적인 강인성을 증명합니다.

- **Technical Details**: 이 연구는 적대적 오류 비율에 대한 새로운 하한 추정 방법을 제시하며, 이는 양자 모델의 구조에 독립적이며, 양자 왜곡 공격(quantal perturbation attacks)과 같은 양자 모델에 특화된 서로 다른 공격을 처리합니다. Projected Gradient Descent (PGD)를 기반으로 한 새로운 양자 공격 전략을 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 경계는 실제 양자 모델에서 관찰된 적대적 오류 비율과 강한 상관관계를 보여주며, 가장 좋은 경우에 실험 오류가 추정된 경계보다 10%만큼 높아 본질적인 강인성을 입증합니다.



### Scientific machine learning in ecological systems: A study on the predator-prey dynamics (https://arxiv.org/abs/2411.06858)
Comments:
          16 pages, 7 figures, 1 table

- **What's New**: 본 연구에서는 Neural Ordinary Differential Equations (Neural ODEs) 및 Universal Differential Equations (UDEs)라는 과학적 기계 학습의 두 가지 핵심 기법을 Lotka-Volterra 포식자-피식자 모델에 적용하였습니다. 이 모델은 포식자와 피식자 개체군 간의 동적 상호작용을 설명하는 기본 생태 모델입니다. 논문에서는 사전 지식 없이 데이터와 신경망만을 이용하여 내재된 미분 방정식을 찾아내는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 Julia 프로그래밍 언어를 사용하여 Neural ODEs와 UDEs를 통해 Lotka-Volterra 시스템의 예측 및 예보를 수행합니다. Neural ODE는 전통적인 ODE/PDE 시스템을 신경망으로 대체하며, UDE는 미분 방정식의 특정 항을 신경망으로 대체합니다. UDE는 데이터에서 미지의 역학을 학습하면서 알려진 물리 법칙을 보존합니다. 또한 Gaussian noise를 도입하여 데이터의 노이즈 저항성을 확인하였으며, Hyperparameter optimization을 통해 최적의 신경망 구조와 활성화 함수, 최적화 알고리즘을 조사하였습니다.

- **Performance Highlights**: UDE는 적은 양의 훈련 데이터로도 정확한 예측을 달성하며, Neural ODEs보다 더 뛰어난 성능을 보였습니다. 특히, UDE는 Gaussian noise가 있는 데이터에서도 더욱 강인한 모습을 보였으나, Neural ODE는 높은 수준의 노이즈에 어려움을 겪었습니다. 추가로, 분석을 통해 예측 정확도가 크게 저하되기 시작하는 '예측 분해점(forecasting breakdown point)'의 개념을 도입하여, 장기 예측 과제에서 현재 SciML 프레임워크의 한계를 조명하였습니다.



### Generative Feature Training of Thin 2-Layer Networks (https://arxiv.org/abs/2411.06848)
- **What's New**: 본 연구에서는 작은 수의 hidden weights를 가진 2-layer neural networks를 이용해 함수 근사를 다루며, gradient-based training의 제한점인 local minima 문제를 해결하기 위해 생성 모델(generative model)을 이용한 초기화 방법을 제안합니다.

- **Technical Details**: 제안하는 방법에서는 deep generative model로 parameterized된 제안 분포(proposal distribution)에서 hidden weights를 샘플링하고, 이후 최적화 과정에서 샘플된 weights를 latent space에서 gradient-based post-processing을 통해 다듬습니다. 이 과정에서 regularization scheme도 포함되어 노이즈를 감소시킵니다.

- **Performance Highlights**: 제안된 방법은 수치 예제를 통해 효과성을 입증하였으며, 특히 작은 데이터셋이나 gradient-based 알고리즘이 잘 작동하지 않는 경우에 대해 우수한 성능을 보였습니다.



### Spatially Constrained Transformer with Efficient Global Relation Modelling for Spatio-Temporal Prediction (https://arxiv.org/abs/2411.06836)
- **What's New**: 이 논문에서는 스마트 시티의 지속 가능한 발전을 위해 중요한 공간-시간(spatio-temporal) 예측 모델인 ST-SampleNet을 제안합니다. 이 모델은 CNN(Convolutional Neural Networks)과 Self-attention 메커니즘을 결합하여 지역 간의 거리 관계를 잘 포착할 수 있도록 설계되었습니다. 특히, 기존 모델들이 간과했던 글로벌(global) 관계 포착의 중요성을 강조합니다.

- **Technical Details**: ST-SampleNet은 지역 샘플링 전략을 통해 중요하지 않은 지역을 제거하여 효율성을 높이는 경량화된(region sampling) 방법을 따릅니다. 또한, 공간 제약이 있는 포지션 임베딩(spatially constrained position embedding)을 도입하여 이웃 지역의 정보가 Self-attention 메커니즘에 통합되도록 했습니다.

- **Performance Highlights**: ST-SampleNet은 세 가지 실제 데이터셋에서 실험 평가를 통해 그 효율성과 효과성을 입증하였으며, 효율적인 변형 모델은 계산 비용을 40% 감소시키면서 성능의 약 1%만 소폭 저하되었습니다.



### Adaptive Conditional Expert Selection Network for Multi-domain Recommendation (https://arxiv.org/abs/2411.06826)
- **What's New**: 최근 Multi-domain recommendation (MDR)에서 Mixture-of-Experts (MOE) 기반 접근 방법의 대안으로 CESAA 모델이 제안되었습니다. 이 모델은 Conditional Expert Selection (CES)와 Adaptive Expert Aggregation (AEA) 모듈로 구성되어 있어 성능 저하와 확장성 문제를 해결합니다.

- **Technical Details**: CES 모듈은 희소한 게이팅 전략과 도메인 공유 전문가를 결합하여 특정 도메인에 적합하도록 전문가를 선택합니다. AEA 모듈은 상호 정보 손실(mutual information loss)을 활용하여 전문가와 특정 도메인 간의 상관관계를 강화하고, 각 인스턴스에 대해 도메인 공유 및 선택된 도메인 전문 도시기자만 활성화하여 효율성을 높입니다. 이러한 구조는 손쉬운 end-to-end 학습을 가능하게 하여 사전 정의된 도메인 구분 없이 최적의 패턴을 찾아냅니다.

- **Performance Highlights**: 공공 랭킹 및 산업 리트리벌 데이터 세트를 통한 실험 결과, CESAA 모델은 최신 방법들과 비교하여 더 우수한 성능을 보이며 MDR 작업에서 높은 효과성을 입증했습니다.



### Large Language Model in Medical Informatics: Direct Classification and Enhanced Text Representations for Automatic ICD Coding (https://arxiv.org/abs/2411.06823)
Comments:
          accepted at the 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

- **What's New**: LLAMA-2 모델을 활용하여 ICD 코드 분류 자동화의 효과성을 높이는 두 가지 방법론을 제시합니다. 하나는 직접적인 분류기로서의 LLAMA-2 활용이며, 다른 하나는 Multi-Filter Residual Convolutional Neural Network (MultiResCNN)에서 풍부한 텍스트 표현 생성입니다.

- **Technical Details**: LLAMA-2 모델을 직접적인 ICD 코드 분류자로써 활용하고, MultiResCNN을 통한 텍스트 표현 생성을 위해 학습합니다. 이 네트워크는 Deep Learning과 Natural Language Processing의 통합을 기반으로 하고, 복잡한 의학적 문서를 효과적으로 해석하기 위한 구성 요소를 가지고 있습니다. 세분화된 ICD 코드 공간을 다루기 위해 특수한 분류 헤드를 포함시킵니다. 또한, RoPE (Rotatory Position Embedding) 및 YaRN (Yet another RoPE extension method) 기법을 통해 긴 임상 텍스트를 효율적으로 처리합니다.

- **Performance Highlights**: MIMIC-III 데이터셋에서 LLAMA-2의 적용 결과, ICD 코드 분류 정확도가 현저히 향상됨을 보였습니다. LLAMA-2를 활용한 두 가지 접근법 모두 기존 최첨단 방법들과 비교하여 뛰어난 성과를 나타냈습니다.



### Streetwise Agents: Empowering Offline RL Policies to Outsmart Exogenous Stochastic Disturbances in RTC (https://arxiv.org/abs/2411.06815)
- **What's New**: 비론행 정책을 보완하기 위한 새로운 방법인 Streetwise 에이전트가 소개되었습니다. 이 에이전트는 예기치 않은 외부 요인으로 인해 발생하는 도메인 변화(OOD, Out-Of-Distribution) 문제를 처리할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구에서는 실시간으로 실험 환경에서 OOD 공간을 정량화하고 특성화하여 기존의 비론행 정책을 조정하는 방법을 제안합니다. 이로 인해 실제 네트워크 병목 현상의 대역폭 추정(BWE) 문제에서 강화된 성능을 보입니다.

- **Performance Highlights**: 최종 반환과 관련하여 기존의 최신 기술 대비 평균 약 18%의 현저한 성능 개선이 이루어졌습니다.



### Structuring the Processing Frameworks for Data Stream Evaluation and Application (https://arxiv.org/abs/2411.06799)
- **What's New**: 이 연구는 실세계 응용 환경에 적합한 데이터 스트림 처리 프레임워크에 대한 문제를 다룹니다. 레이블 접근의 지연과 제한을 고려하여 데이터 스트림 분류 방법을 신뢰성 있게 평가할 필요성이 강조됩니다.

- **Technical Details**: 구조화된 프레임워크는 'concept drifts'(개념 이동)과 'drift detection'(변화 탐지) 간의 연관성을 보여주는 데이터 스트림 처리 프레임워크의 분류 체계를 제안합니다. 이 방법론은 'moving window mechanism'(이동 창 기법)과 'Adaptive Windowing'(적응형 창기법) 등의 기법을 활용하고 있습니다.

- **Performance Highlights**: 연구에서 제안된 분류 체계는 다양한 'drift detection method'(변화 탐지 방법)들을 포함하며, 각기 다른 레이블 요구 사항을 가진 세 가지 그룹으로 나누어 집니다. 이로써 데이터 스트림 처리의 효과성과 정확성을 높일 수 있습니다.



### White-Box Diffusion Transformer for single-cell RNA-seq generation (https://arxiv.org/abs/2411.06785)
Comments:
          11pages, 3 figures

- **What's New**: 본 논문에서는 Diffusion 모델과 White-Box Transformer를 기반으로 한 하이브리드 모델인 White-Box Diffusion Transformer를 제안하여, 합성적이고 생물학적으로 타당 scRNA-seq 데이터를 생성하는 데 성공한다고 발표하였습니다.

- **Technical Details**: Diffusion 모델은 데이터에 점진적으로 노이즈를 추가한 후 이를 제거하는 과정을 통해 데이터를 생성합니다. White-Box Transformer는 데이터의 코딩 비율을 최소화하고 표현의 희소성을 극대화하여 수학적인 해석 가능성을 제공합니다. White-Box Diffusion Transformer는 이러한 두 모델의 장점을 결합하여 성능을 높였습니다.

- **Performance Highlights**: 실험에서 여섯 가지 서로 다른 단일 세포 RNA-Seq 데이터셋을 사용하여 생성된 데이터와 실제 데이터를 비교한 결과, White-Box Diffusion Transformer가 scRNA-seq 데이터 생성에서 DiT와 유사한 성능을 보여주며 훈련 효율성과 자원 활용에서 상당한 개선을 이루었습니다.



### Model Partition and Resource Allocation for Split Learning in Vehicular Edge Networks (https://arxiv.org/abs/2411.06773)
Comments:
          arXiv admin note: text overlap with arXiv:2306.12194 by other authors

- **What's New**: 본 논문은 자율 주행 기술과 차량 네트워크의 통합에 관련된 프라이버시 보호, 통신 효율성 및 자원 배분 문제를 해결하기 위해 새로운 U-shaped split federated learning (U-SFL) 프레임워크를 제안합니다. 이 프레임워크는 로컬 데이터와 라벨을 차량 사용자(VU) 측에 보관하면서 여러 차량 간의 병렬 처리를 가능하게 합니다.

- **Technical Details**: U-SFL은 semantic-aware auto-encoder (SAE)를 도입하여 전송 데이터의 차원 감소를 이룩하고 중요한 의미 정보를 보존합니다. 또한, deep reinforcement learning (DRL) 기반 알고리즘을 개발하여 동적 자원 배분 및 분할 지점 선택의 NP-hard 문제를 해결합니다. U-SFL은 기존의 split learning (SL)과 동일한 분류 성능을 유지하면서 데이터 전송량과 통신 지연을 유의미하게 감소시킵니다.

- **Performance Highlights**: U-SFL은 자율 주행 응용 프로그램을 위한 차량 네트워크에서 프라이버시 보호, 통신 효율성 및 학습 성능 균형을 이루는 통합 솔루션을 제공하며, DRL 기반 최적화 알고리즘을 통해 지연, 에너지 소비 및 학습 성능의 균형을 잘 유지합니다.



### Sketched Adaptive Federated Deep Learning: A Sharp Convergence Analysis (https://arxiv.org/abs/2411.06770)
- **What's New**: 본 연구는 federated learning (FL)에서 gradient 압축 방법과 adaptive 최적화 기법을 결합한 새로운 알고리즘인 Sketched Adaptive Federated Learning (SAFL)를 소개하고, 이의 이론적 수렴성을 로그 스케일로 증명하여 통신 비용을 효율적으로 줄일 수 있음을 보여준다.

- **Technical Details**: SAFL 알고리즘은 adaptive 방법과 무작위 sketching 기법을 통합하여, 통신비용을 줄이면서도 수렴성을 보장한다. 이를 통해 현재의 deep learning 모델의 매개변수 공간 차원에 대한 의존성을 선형이 아닌 로그에만 의존하게 만들며, 이는 총 O(1/sqrt(T)) 수렴 속도로 이어진다.

- **Performance Highlights**: 실험적으로 SAFL 방법은 언어 및 시각 과제에서 효과적으로 수렴 속도를 높였고, 특히 heavy-tailed noise 환경에서도 Sketched Adaptive Clipped Federated Learning (SACFL) 알고리즘이 안정적으로 수렴함을 증명하였다.



### Research on an intelligent fault diagnosis method for nuclear power plants based on ETCN-SSA combined algorithm (https://arxiv.org/abs/2411.06765)
- **What's New**: 이 논문은 원자력 발전소(NPP)에서 효율적이고 정확한 결함 진단을 위한 새로운 지능형 결함 진단 방법을 제안합니다. 이 방법은 향상된 시간 합성곱 신경망(ETCN)과 참새 탐색 알고리즘(SSA)을 결합하여 성능을 최적화합니다.

- **Technical Details**: 제안된 ETCN은 시간 합성곱 신경망(TCN), 자기 주의(SA) 메커니즘, 그리고 잔여 블록(residual block)을 활용하여 성능을 향상시키며, SSA는 하이퍼파라미터(hyperparameters)를 적응적으로 최적화하여 성능을 개선합니다. 이론적으로는 지역적 특징을 잘 추출하고 시계열 정보를 효과적으로 캡처하는 데 강점을 보입니다.

- **Performance Highlights**: 제안한 방법은 CPR1000 시뮬레이션 데이터 세트에서 실험적으로 검증되었으며, 기존의 다른 고급 지능형 결함 진단 방법보다 모든 평가 지표에서 뛰어난 성능을 보였습니다. 이는 NPP의 지능형 결함 진단에 있어 유망한 도구가 될 것으로 기대됩니다.



### Neuromodulated Meta-Learning (https://arxiv.org/abs/2411.06746)
- **What's New**: 이번 연구에서는 메타러닝(meta-learning)에서의 유연한 네트워크 구조(flexible network structure, FNS)의 중요성을 강조합니다. FNS를 통해 각 작업에 최적화된 구조를 생성함으로써 성능과 학습 효율성을 극대화하는 방식을 제안합니다.

- **Technical Details**: FNS는 (i) frugality, (ii) plasticity, (iii) sensitivity의 세 가지 속성을 가져야 하며, 이를 정량화하기 위한 세 가지 측정치를 제안합니다. 최종적으로, Neuromodulated Meta-Learning (NeuronML) 모델을 통해 bi-level optimization을 사용하여 모델의 가중치(weight)와 네트워크 구조를 함께 최적화합니다.

- **Performance Highlights**: NeuronML은 다양한 작업에서의 성능 향상을 입증했으며, 여러 실험에서 계속해서 우수한 성능을 달성했습니다. 이로써 FNS의 역할이 메타러닝의 성공에 필수적임을 확인했습니다.



### Dockformer: A transformer-based molecular docking paradigm for large-scale virtual screening (https://arxiv.org/abs/2411.06740)
Comments:
          14 pages, 5 figures

- **What's New**: 본 연구에서는 새로운 deep learning 기반 분자 도킹 방법인 Dockformer를 소개합니다. 이 방법은 다중 모달 정보(multimodal information)를 활용하여 분자의 기하학적 구조와 정보를 포착하고, end-to-end 방식으로 결합 형태(binding conformations)를 직접 생성할 수 있습니다.

- **Technical Details**: Dockformer는 두 개의 독립적인 인코더(encoders)를 사용하여 단백질과 리간드의 잠재 임베딩(latent embeddings)을 생성하고, 결합 모듈(binding module)을 통해 분자의 관계를 효과적으로 탐지합니다. 생성된 구조는 리간드 원자의 좌표를 end-to-end 방식으로 계산하며, 각각의 생성된 결합 형태에 대한 신뢰도(confidence measures)를 사용하여 결합 강도를 구분합니다.

- **Performance Highlights**: Dockformer는 PDBbind 코어 집합과 PoseBusters 벤치마크에서 각각 90.53%와 82.71%의 성공률을 기록하였으며, 추론 속도는 100배 이상 증가하여 거의 모든 최첨단 도킹 방법을 초월했습니다. 또한, 코로나바이러스의 주요 단백질 분해 효소 억제제를 식별하는 데 있어서도 뛰어난 성능을 보여줍니다.



### Beating Adversarial Low-Rank MDPs with Unknown Transition and Bandit Feedback (https://arxiv.org/abs/2411.06739)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 고정된 전이(transition)와 적대적 손실(adversarial loss)을 가진 저랭크( low-rank) MDP에서의 후회(regret) 최소화를 다룹니다. 이전 연구와는 달리 본 논문은 전이 정보가 알려지지 않은 상황에서의 전체 정보 손실( full-information loss) 피드백과, 전이가 알려진 밴딧 손실( bandit loss) 피드백을 동시에 고려합니다.

- **Technical Details**: 첫 번째로, 전체 정보 미지 전이 전체에서 Zhao et al. (2024)의 $poly(d, A, H)T^{5/6}$ 후회 한계를 $poly(d, A, H)T^{2/3}$로 개선했습니다. 여기서 d는 전이의 랭크, A는 행동의 수, H는 수평(horizon) 길이, T는 에피소드의 수를 나타냅니다. 다음으로, 밴딧 손실 피드백과 미지의 전이에 대한 연구를 시작했으며, 손실이 선형 구조(linear structure)를 가질 때 $poly(d, A, H)T^{2/3}$의 후회를 보장하는 모델 기반(model-based) 및 모델 자유(model-free) 알고리즘을 제안했습니다.

- **Performance Highlights**: 제안된 알고리즘은 계산적으로 비효율적일 수 있지만, 오라클 효율적인 모델 자유 알고리즘은 $poly(d, A, H)T^{4/5}$ 후회를 달성했습니다. 선형 구조가 필요하다는 것을 보여주었으며, 보상 함수(reward function)에 구조가 없을 경우 밴딧 경우에서 후회는 상태 수(state)의 다항식으로 스케일해야 한다고 결론지었습니다.



### Mr.Steve: Instruction-Following Agents in Minecraft with What-Where-When Memory (https://arxiv.org/abs/2411.06736)
- **What's New**: 본 논문에서는 저자들이 Steve-1의 제한을 해결하기 위해 새로운 저수준 컨트롤러인 Memory Recall Steve-1(Mr.Steve)을 제안합니다. 이 새로운 시스템은 사건 중심의 메모리인 Place Event Memory(PEM)를 통합하여 과거의 사건을 효율적으로 기억하고 탐색할 수 있도록 설계되었습니다.

- **Technical Details**: Mr.Steve는 고유한 Place Event Memory(PEM)를 가지고 있으며, 이는 시간, 장소 및 사건에 관한 정보를 구조화하고 저장하는 계층적 메모리 시스템입니다. 이 시스템은 낮은 수준의 컨트롤러에 필수적인 기억 기능을 제공하여 반복 실패를 줄이고, 탐색과 목표 달성 사이에서 효과적으로 전환할 수 있게 합니다.

- **Performance Highlights**: Mr.Steve는 기존의 방법들보다 탐색과 작업 해결 효율성을 유의미하게 개선했습니다. 특히, Minecraft와 같은 환경에서 긴 시퀀스의 작업을 해결하는 데 있어 기존 기반선들에 비해 월등한 성과를 보였습니다.



### GSL-PCD: Improving Generalist-Specialist Learning with Point Cloud Feature-based Task Partitioning (https://arxiv.org/abs/2411.06733)
- **What's New**: 이 논문은 Generalist-Specialist Learning (GSL)의 랜덤 작업 분할 방식의 한계를 분석하고, 포인트 클라우드(Point Cloud) 피처 기반의 작업 분할을 통해 성능 개선을 제안합니다. 제안된 GSL-PCD 프레임워크는 robot manipulation 작업에서 객체 특성을 근거로 분할을 수행하여 더 나은 일반화 성능을 달성합니다.

- **Technical Details**: GSL-PCD 방법은 PointNet++ 모델을 이용해 객체의 포인트 클라우드 피처를 인코딩하고, 이를 바탕으로 균형 잡힌 클러스터링 알고리즘을 통해 유사한 변형들을 동일한 specialist에 할당합니다. 이 과정은 강화 학습 환경에서 specialist 모델의 학습 난이도를 줄이며, 다양한 환경 변형을 처리하는 능력을 향상시킵니다.

- **Performance Highlights**: ManiSkill 벤치마크에서의 실험 결과, GSL-PCD 방법이 이전의 랜덤 분할 방식보다 고정된 specialist 수로 9.4% 성능을 개선하였으며, 필요한 계산 및 샘플 요구사항을 50% 줄였습니다. 특히, 60가지 서로 다른 수도꼭지 유형을 기반으로 한 Turn Faucet 작업에서의 성과가 두드러집니다.



### On the Principles of ReLU Networks with One Hidden Layer (https://arxiv.org/abs/2411.06728)
- **What's New**: 이번 논문은 단층 네트워크와二층 ReLU 네트워크의 함수 근사화 기법을 체계적으로 연구하여, 이러한 네트워크의 "블랙 박스" 문제를 해결하고자 하였습니다. 이를 통해 훈련된 솔루션을 이론적으로 이해하고 실험적으로 검증하였습니다.

- **Technical Details**: 네트워크 훈련 프로세스와 솔루션 공간을 명확히 이해하기 위해, 논문에서는 스플라인(spline)의 일면 기초(one-sided bases)와 같은 원리를 통해 고차원 입력에 대한 훈련 솔루션을 도출하는 몇 가지 새로운 원칙을 제안합니다. 이론적으로 제안된 방법은 다수의 엄격한 부분 순서(multiple strict partial orders)와 연속성 제한(continuity restriction)을 포함하여, 고차원 입력의 훈련 솔루션에 대한 일반적인 함수 근사화 능력을 증명합니다.

- **Performance Highlights**: 본 논문의 이론적 결과는 단순한 일차원 입력에 대한 깊은 이해를 이끌어낼 뿐만 아니라, 고차원 입력에서도 어느 정도 해석 가능함을 보여줍니다. 또한, 실험적 결과는 제안된 이론의 적합성을 증명하여, ReLU 네트워크의 블랙 박스를 풀 수 있는 가능성을 제시하고 있습니다.



### Synthesize, Partition, then Adapt: Eliciting Diverse Samples from Foundation Models (https://arxiv.org/abs/2411.06722)
- **What's New**: 본 논문에서는 사용자 경험을 향상시키고 다양한 선호를 수용하기 위해, 다양한 응답을 제공하도록 기초 모델의 출력을 다양화하는 새로운 프레임워크인 Synthesize-Partition-Adapt (SPA)를 제안합니다. 이 방법은 고품질의 다양한 응답을 생성하면서도 정확성을 희생하지 않는 것을 목표로 합니다.

- **Technical Details**: SPA는 널리 사용되는 합성 데이터(synthetic data)를 활용하여 기초 모델의 출력을 다각화하는 효율적인 접근 방식을 제공합니다. 이 프레임워크는 데이터 파티셔닝(data partitioning) 기법을 이용하여 합성 데이터를 서브셋으로 나누고, 각 서브셋에 대해 최적화된 모델 적응(model adaptation)을 훈련합니다. 이를 통해 다양한 응답을 생성할 수 있으며, 불필요한 품질 저하를 방지합니다.

- **Performance Highlights**: 저자는 HumanEval와 MBPP와 같은 코드 생성 작업 및 자연어 이해(natural language understanding) 작업에 대한 실험을 통해 SPA의 효과를 입증하였습니다. 이 실험 결과는 SPA가 높은 정확성을 유지하면서 모델의 응답 다양성을 증대시킬 수 있음을 보여줍니다. 이는 다양한 응용 프로그램에서 사용자 경험을 풍부하게 할 수 있는 잠재력을 강조합니다.



### Real-time Monitoring and Analysis of Track and Field Athletes Based on Edge Computing and Deep Reinforcement Learning Algorithm (https://arxiv.org/abs/2411.06720)
Comments:
          17 pages

- **What's New**: 이 연구는 전통적인 모니터링 시스템의 한계를 극복하기 위해 IoT 최적화 시스템을 개발하고, 에지 컴퓨팅(edge computing)과 딥 러닝(deep learning) 알고리즘을 통합하여 실시간 성능 모니터링을 구현했습니다.

- **Technical Details**: 제안된 시스템은 SAC 최적화된 딥 러닝 모델을 IoT 아키텍처에 통합하여 복잡한 운동 데이터를 효율적으로 인식하고 실시간 피드백을 제공합니다. 다중 센서 데이터 융합 방법을 설계하여 데이터 처리의 실시간 성능과 정확성을 크게 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 반응 시간, 데이터 처리 정확도 및 에너지 효율성 측면에서 전통적인 방법보다 현저히 우수한 성능을 보였으며, 특히 복잡한 육상 경기에서 두드러진 효과를 나타냈습니다.



### Learning a Single Neuron Robustly to Distributional Shifts and Adversarial Label Nois (https://arxiv.org/abs/2411.06697)
- **What's New**: 이 연구에서는 적대적 분포 변화(adversarial distribution shifts)가 있는 상황에서 L2² 손실에 대한 단일 뉴런(single neuron)을 학습하는 문제를 다룹니다. 특히, 라벨이 임의로 주어질 때 최적의 함수(best-fit function)를 찾는 것을 목표로 합니다.

- **Technical Details**: 훈련 샘플이 참조 분포(reference distribution) \(\mathcal{p}_0\)에서 주어졌을 때, 최악의 경우 분포가 \(\chi^2\) 발산(chi-squared divergence) 점근에서 \(\mathcal{p}_0\)와 가까운 경우에 대한 제곱 손실을 최소화하는 벡터 \(\mathbf{w}^*\)를 근사하는 것이 핵심입니다. 이 algorithm은 원래 비볼록(nonconvex) L2² 손실에 대한 위험(risk)을 직접 제한하는 방식으로 설계되었습니다.

- **Performance Highlights**: 제안된 알고리즘은 실용적이며, 엔지니어링 관점에서 비볼록 구조를 가지는 경우에 대한 새로운 소스를 열어주고 있습니다. 이를 통해 기존 알고리즘의 성능을 이상적으로 개선할 수 있는 가능성이 있습니다.



### Shedding Light on Problems with Hyperbolic Graph Learning (https://arxiv.org/abs/2411.06688)
Comments:
          Preprint

- **What's New**: 최근 그래프 머신 러닝에서는 하이퍼볼릭(hyperbolic) 표현 학습 방식이 주목 받고 있으며, 이 방식의 효용성을 주장하는 여러 논문들이 발표되었습니다. 그러나 본 연구에서는 유클리드(Euclidean) 모델이 같은 환경에서 하이퍼볼릭 모델보다 뛰어난 성능을 보인다는 예상을 뒤엎는 발견을 하였습니다.

- **Technical Details**: 본 연구에서는 Gromov δ-hyperbolicity를 기반으로 하여 하이퍼볼릭 공간에서의 데이터 적합성을 평가하는 것의 한계를 논의하였습니다. 또한, 유클리드 모델이 하이퍼볼릭 모델들과 비교하여 비슷하거나 더 우수한 성과를 거둔 3가지 주요 문제를 식별하고 분석하였습니다. 여기에는 데이터셋 선택의 부적절성, 모델링 가정의 오류, 잘못된 메트릭 사용 등이 포함됩니다.

- **Performance Highlights**: Chami et al. (2019)에서 제시된 여러 그래프 작업에서, 간단한 유클리드 모델이 다양한 최신 하이퍼볼릭 모델과 동등하거나 더 나은 성능을 보이는 것을 입증하였습니다. 이러한 발견은 모델을 개선하고, 하이퍼볼릭 그래프 신경망의 적용 가능성을 평가하기 위한 파라메트릭 기준 데이터셋을 도입하는 기초가 됩니다.



### WDMoE: Wireless Distributed Mixture of Experts for Large Language Models (https://arxiv.org/abs/2411.06681)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)와 무선 네트워크의 결합을 통해 향상된 성능을 달성하기 위한 새로운 아키텍처인 Wireless Distributed Mixture of Experts (WDMoE)를 제안합니다. LLM의 MoE 레이어를 분해하여 기지국(Base Station)과 모바일 장치 간의 협업 배포를 가능하게 하여, 무선 네트워크에서 LLM의 효율성을 극대화합니다.

- **Technical Details**: WDMoE 아키텍처는 LLM의 MoE 레이어에서 게이팅 네트워크(gating network)와 선행 신경망(neural network) 레이어를 미리 기지국에 배치하고, 전문가 네트워크(expert networks)는 다양한 모바일 장치에 분산시킵니다. 이 방식은 모바일 장치의 병렬 추론(parallel inference) 능력을 활용하여 한정된 계산(computing) 및 캐싱(caching) 자원을 효과적으로 이용할 수 있습니다. 성능 측정 기준을 개발하고, 그 기준을 바탕으로 전문가 선택과 대역폭 할당을 최적화하여 지연(latency)을 최소화하며 정확도를 유지하는 데 중점을 두었습니다.

- **Performance Highlights**: 이론적인 시뮬레이션과 실제 하드웨어 실험에서 WDMoE 방법은 성능 저하 없이 지연을 상당히 줄일 수 있음을 입증하였습니다. NVIDIA Jetson 키트를 활용하여 구축한 하드웨어 테스트베드에서 성능을 검증하였으며, 무선 네트워크 환경에서의 실용적인 LLM 운영 가능성을 보여주었습니다.



### An Efficient Memory Module for Graph Few-Shot Class-Incremental Learning (https://arxiv.org/abs/2411.06659)
Comments:
          16 pages, 6 figures, 38th Conference on Neural Information Processing Systems, 2024

- **What's New**: 본 논문에서는 그래프 표현 학습에서 발생하는 재앙적 망각(catasrophic forgetting) 문제를 해결하기 위한 새로운 방법인 Mecoin을 제안합니다. Mecoin은 제한된 라벨 수로도 효과적으로 그래프의 클래스를 학습할 수 있도록 설계되었습니다.

- **Technical Details**: Mecoin은 두 가지 주요 구성 요소, 즉 클래스 프로토타입을 학습하고 저장하는 Structured Memory Unit(SMU)와 GNN과의 동적 메모리 상호 작용을 위한 Memory Representation Adaptive Module(MRaM)으로 구성됩니다. SMU 내의 Memory Construction Module(MeCs)는 입력 노드의 특징과 프로토타입 표현 간의 상호 작용을 통해 샘플 표현을 업데이트하고, MRaM은 각 클래스 프로토타입에 대한 확률 분포를 저장하여 매개변수 조정을 통해 발생할 수 있는 지식 손실을 줄입니다.

- **Performance Highlights**: Mecoin은 기존의 관련 방법들과 비교하여 정확도(accuracy)와 망각률(forgetting rate)에서 우수한 성능을 보이며, GNN의 메모리를 잘 유지하면서도 효과적인 일반화 오류(generalization error)를 달성합니다. 이 방법은 메타 학습(meta-learning)으로 생성한 여러 샘플을 저장하는 기존 방법과는 달리, 메모리 소비를 최소화하며 고성능을 유지합니다.



### Machine learning enabled velocity model building with uncertainty quantification (https://arxiv.org/abs/2411.06651)
- **What's New**: 본 논문은 CO2 저장 프로젝트의 모니터링 및 석유 탐사와 같은 여러 지구 물리학적 응용에서 중요한 마이그레이션 속도 모델을 정교하게 특징화하는 방법을 제안합니다. 기존의 Full-Waveform Inversion (FWI) 등의 방법은 복잡한 역문제의 문제를 해결하기 어렵지만, 본 연구에서는 Generative modeling과 physics-informed summary statistics를 통합한 확장 가능한 방법론을 개발하였습니다.

- **Technical Details**: 제안하는 방법은 Diffusion 네트워크를 기반으로 하여, 초기 속도 모델이 불량한 경우 지하 오프셋 이미지 볼륨을 기반으로 요약 통계치를 정의합니다. 이를 통해 마이그레이션 속도 모델의 Bayesian posterior 샘플을 효과적으로 생성하고 불확실성 평가를 가능하게 합니다. 복잡한 속도 모델 구축의 경우, salt flooding과 함께 posterior 근사를 정제하는 새로운 반복 작업흐름, ASPIRE를 제안하였습니다.

- **Performance Highlights**: 현대 합성 데이터셋을 활용하여 Common-Image Gathers (CIGs)를 사용하는 경우의 이점을 재확인 하였으며, 실측 데이터셋을 통한 개념 증명을 통해 본 방법이 산업 규모의 문제에 확장 가능함을 보여주었습니다. 다양한 데이터셋에서 테스트를 통해 계산 효율성을 확보하며 2D 대형 역문제 해결에 효과적임을 입증하였습니다.



### Understanding Scaling Laws with Statistical and Approximation Theory for Transformer Neural Networks on Intrinsically Low-dimensional Data (https://arxiv.org/abs/2411.06646)
- **What's New**: 이 논문은 트랜스포머 기반의 대형 언어 모델에서 모델 크기와 데이터 크기에 따라 일반화 오류(generalization error)가 어떻게 파워 스케일링 법칙을 따른다는 이론을 제시합니다. 특히, 이론은 훈련 데이터가 저차원 매니폴드(low-dimensional manifold)에 집중될 때 적용됩니다.

- **Technical Details**: 저자는 새로운 통계적 추정(statistical estimation) 및 수학적 근사(mathematical approximation) 이론을 개발하여 트랜스포머 신경망의 일반화 오류와 모델/데이터 크기 간의 스케일링 법칙을 예측하고 정당화했습니다. 이 연구에서 일반화 오류는 훈련 데이터 크기와 네트워크 크기에 대해 전통적인 방식으로 설명할 수 있으며, 모델은 저차원 데이터 구조를 활용하여 데이터 기하학(data geometry)을 존중하는 방법으로 스케일링 법칙을 설명합니다.

- **Performance Highlights**: LLM을 자연어 데이터 세트에서 훈련한 결과, 관측된 경험적 데이터 스케일링 법칙은 이론적 예측과 밀접한 일치를 보였습니다. 이 결과는 이론과 실제 모두에서 트랜스포머 스케일링 법칙에 영향을 미치는 데이터의 내재적 차원(intrinsic dimension)이 중요한 수량임을 엄격하게 보여줍니다.



### Mixed Effects Deep Learning Autoencoder for interpretable analysis of single cell RNA Sequencing data (https://arxiv.org/abs/2411.06635)
Comments:
          Main manuscript: 29 pages, including 10 figures and 8 tables. Supplemental material: 17 pages

- **What's New**: 이번 연구는 Mixed Effects Deep Learning (MEDL) Autoencoder 프레임워크를 제안하여 single-cell RNA sequencing (scRNA-seq) 데이터에서 존재하는 기술적 또는 생물학적 배치 효과를 효과적으로 모델링합니다.

- **Technical Details**: MEDL 프레임워크는 고정 효과(fixed effects)와 무작위 효과(random effects)를 분리하여 모델링하고, 이를 통해 생물학적 상태를 나타내는 정보와 배치 특정 변동성을 포착하여 예측 정확도를 높입니다.

- **Performance Highlights**: MEDL은 Healthy Heart, Autism Spectrum Disorder (ASDc), Acute Myeloid Leukemia (AML) 세 가지 데이터셋에 적용되어 147개의 배치를 처리하고, 자폐증 환자와 건강한 개인 간의 이질성을 포착하였으며, AML에서는 다양한 세포 유형과 악성 세포가 포함된 복잡한 환경에서도 이질성을 구별하는 데 성공했습니다.



### Inductive Graph Few-shot Class Incremental Learning (https://arxiv.org/abs/2411.06634)
- **What's New**: 이 논문에서는 점진적인 그래프 노드 분류 문제를 다루기 위해 Inductive Graph Few-Shot Class Incremental Learning (GFSCIL)이라는 새로운 개념을 소개하고 있습니다. 이 방식은 기존의 그래프를 저장하지 않고도 새로운 클래스를 지속적으로 학습할 수 있다는 점에서 기존 GFSCIL과 차별화됩니다.

- **Technical Details**: 제안된 방법인 Topology-based class Augmentation and Prototype calibration (TAP)은 노드 간의 다중 토폴로지 클래스 증강 방법을 포함하여 모델의 일반화 능력을 향상시킵니다. 이 방법에서는 각 점진적 세션마다 새로운 클래스의 노드를 기반으로 disjoint subgraph를 사용하며, 프로토타입의 분별력을 높이기 위해 반복적인 프로토타입 보정 방법을 적용합니다. 또한, 기존 클래스의 프로토타입들이 feature distribution drift로 인해 실패하는 것을 보완하는 프로토타입 이동 방법을 제안합니다.

- **Performance Highlights**: 제안된 TAP 방법은 네 개의 데이터 세트에서 검증되었으며, 기존 최첨단 모델들과 비교하여 성능이 향상된 결과를 보였습니다. 이는 새로운 클래스 학습을 가능하게 하면서도 이전 클래스의 인식 능력을 유지하는 데 기여합니다.



### Using Diffusion Models as Generative Replay in Continual Federated Learning -- What will Happen? (https://arxiv.org/abs/2411.06618)
- **What's New**: 새롭게 제안된 DCFL(지속적 연합 학습) 프레임워크는 동적 데이터 입력을 처리하기 위해 조건부 확산 모델(conditional diffusion model)을 이용하여 시뮬레이션된 역사적 데이터를 생성합니다. 이는 기존의 재생 메모리(replay memory) 방법이 필요 없으며, 각 로컬 클라이언트에서 진행됩니다.

- **Technical Details**: DCFL 프레임워크에서는 여러 이상적인 요구 사항을 충족해야 하는 지속적 연합 학습(CFL) 설정을 다루며, 각 클라이언트는 동적으로 변화하는 데이터 및 작업을 경험합니다. 이를 통해 서로 다른 데이터 분포와 비IID(non-IID) 문제를 효과적으로 다루게 됩니다. 또한, DCFL의 수렴 분석(convergence analysis)을 통해 FL 모델과 시뮬레이션된 데이터의 수렴 경계를 연구하였습니다.

- **Performance Highlights**: DCFL은 세 가지 CFL 시나리오와 네 가지 주요 벤치마크 데이터셋에서 실험을 수행하였고, 평균 32.61%의 성능 향상을 보였습니다. 이는 기존의 FL, CL, 전통적인 생성 모델 및 최신 최첨단(SOTA) 기준선을 초과하는 결과입니다.



### Are Neuromorphic Architectures Inherently Privacy-preserving? An Exploratory Study (https://arxiv.org/abs/2411.06613)
- **What's New**: 이번 연구에서는 Spiking Neural Networks (SNNs)의 프라이버시 보호 특성을 기존 인공 신경망 (Artificial Neural Networks, ANNs)과 비교하여 분석합니다. 특히, Membership Inference Attacks (MIAs)에 대한 SNNs의 저항력을 평가하며, SNNs가 프라이버시 보존에 더 우수한 성능을 보임을 제시합니다.

- **Technical Details**: 연구에서는 다양한 데이터셋(MNIST, CIFAR-10, CIFAR-100 등)을 사용하여 SNNs와 ANNs의 MIAs 저항력을 비교하였으며, 에볼루셔너리 알고리즘(evolutionary algorithms)과 서리게이트 그래디언트(surrogate gradient) 기반의 학습 알고리즘을 검토했습니다. SNNs는 ANNs보다 낮은 AUC scores을 기록하며, 이는 MIAs에 대한 더 강한 저항력을 나타냅니다.

- **Performance Highlights**: CIFAR-10에서 SNN은 0.59의 AUC를 기록하여 ANNs(0.82)보다 우수하며, F-MNIST 데이터셋에서 DPSGD를 적용할 경우 SNN은 평균 12.87% 정확도 손실을 보이는 반면, ANNs는 19.55%의 손실을 겪었습니다.



### vTune: Verifiable Fine-Tuning for LLMs Through Backdooring (https://arxiv.org/abs/2411.06611)
- **What's New**: 이 논문은 사용자 맞춤형 데이터셋에서 대형 언어 모델(LLM)의 미세 조정을 검증할 수 있는 새로운 방법인 vTune을 제안합니다. 이 방법은 훈련 데이터에 소량의 '출입구(backdoor)' 데이터 포인트를 추가하여 통계적 테스트를 통해 미세 조정이 제대로 이루어졌는지를 확인합니다.

- **Technical Details**: vTune은 LLM 미세 조정 기술의 최신 발전을 활용하여 훈련 데이터의 <1%의 수치를 수정하여 모델의 진위를 검증합니다. 이 방법은 사용자에게 몇 번의 추론 호출을 통해 높은 확률적인 정확성을 요구하며, 서비스 제공자에게는 약 1%의 추가 작업만을 필요로 합니다. vTune은 다양한 오픈소스 및 폐쇄형 LLM에 확장 가능하다는 장점이 있습니다.

- **Performance Highlights**: vTune 방법은 여러 모델 패밀리와 크기, 다양한 미세 조정 데이터셋에 걸쳐 테스트되었으며, p-값이 약 10^{-40}의 정도로 통계적 테스트가 만족되었습니다. 또한, vTune을 이용한 미세 조정이 하류 작업 성능에 부정적인 영향을 미치지 않음을 입증하였으며, 다양한 공격에 대한 견고성도 확인하였습니다.



### MolMiner: Transformer architecture for fragment-based autoregressive generation of molecular stories (https://arxiv.org/abs/2411.06608)
- **What's New**: 이번 연구에서는 분자 생성 과정을 명확하고 해석 가능한 단계로 분해하는 자기회귀(autoregressive) 모델을 제안합니다. 이는 분자 조각(molecular fragments)을 활용하여 '분자의 이야기(molecular story)'를 만듭니다.

- **Technical Details**: 제안된 모델은 분자 생성 시 화학적 유효성을 보장하고, 해석 가능성을 증가시키며, 분자의 크기를 모델에게 결정하도록 허용합니다. 3차원 기하학적 정보와 화학적 규칙을 강화하여 다중 목표를 갖는 전기활성 유기 화합물의 역설계를 수행합니다.

- **Performance Highlights**: 모델은 다중 목표 객체에 따라 생성 분포를 효과적으로 조정할 수 있으며, 용해도, 환원 전위 및 합성 접근성과 같은 다양한 표적 속성에 대해 우수한 성능을 보였습니다.



### CriticAL: Critic Automation with Language Models (https://arxiv.org/abs/2411.06590)
- **What's New**: 이 논문에서는 CriticAL (Critic Automation with Language Models)을 제안하여 LLM(대형 언어 모델)의 활용을 통해 모델 비판(model criticism)을 자동화하는 새로운 접근 방식을 소개합니다. CriticAL은 모델 예측과 데이터 간의 불일치를 포착하는 summary statistics를 생성하고, 이들의 유의미성을 평가하는 가설 검정을 적용합니다.

- **Technical Details**: CriticAL은 LLM을 통해 모델과 데이터의 메타데이터를 기반으로 데이터의 성질을 포착하는 summary statistics를 생성하고, 이를 통해 모델의 가정이 위반되는지를 평가합니다. 이 통계량은 Python 함수로 구현되어 인간 또는 LLM 과학자가 쉽게 실행할 수 있도록 되어 있어 투명성과 신뢰성을 제공합니다. CriticAL의 summary statistics는 전통적인 가설 검정을 통해 불일치의 유의미성을 평가하여 모델을 자동으로 검증할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, CriticAL은 인위적으로 제어된 불일치에서 신뢰성 있는 비판을 생성하며, 인간 및 LLM 심사자 모두 CriticAL의 비판을 다른 접근 방법보다 투명성과 실행 가능성 측면에서 더 선호하는 것으로 나타났습니다. CriticAL은 실제 데이터셋에서 인간이 설계한 모델보다 개선된 성과를 보였습니다.



### Federated LLMs Fine-tuned with Adaptive Importance-Aware LoRA (https://arxiv.org/abs/2411.06581)
- **What's New**: 이 논문은 이질적인 클라이언트 자원에 적합한 연합형 저랭크 적응(Adaptive Federated Low-Rank Adaptation) 프레임워크를 제안합니다. 이를 통해 데이터 프라이버시를 유지하면서도 대규모 언어 모델(LLM)의 태스크 맞춤형 적응을 가능하게 합니다.

- **Technical Details**: 제안된 HAFL(이질적 적응 연합 LoRA) 프레임워크는 클라이언트의 LoRA 랭크에 따라 중요도 기반 파라미터 절단 및 동결 기법을 도입하여 자원의 이질성을 고려합니다. 클라이언트들은 선택적으로 가장 중요한 LoRA rank-1 매트릭스만 업데이트하며, 나머지 매트릭스는 동결됩니다. 또한, 어댑티브 집계 방식이 도입되어 정보 희석을 방지합니다.

- **Performance Highlights**: 20뉴스 그룹 분류 작업에서 HAFL 방법이 낮은 통신 비용으로 빠르게 수렴하며, 협력 모델 분배 시 성능 저하를 방지할 수 있다는 결과를 보였습니다. 또한, 제안된 동적 집계 방법은 제로 패딩 접근법에 비해 더 빠른 수렴 속도를 기록하였습니다.



### Discovering emergent connections in quantum physics research via dynamic word embeddings (https://arxiv.org/abs/2411.06577)
Comments:
          7 pages; 4 figures; 1 table; Appendix: 2 pages, 2 figures

- **What's New**: 이 논문에서는 양자 물리학 연구의 개념 조합 예측을 위한 새로운 접근 방식으로 동적 단어 임베딩(dynamic word embeddings)을 도입합니다. 기존 지식 그래프(knowledge graphs)와는 달리, 제안된 방법은 개념 간의 암묵적인 관계를 포착하여 더 폭넓은 정보를 인코딩할 수 있습니다.

- **Technical Details**: 연구에서는 arXiv quant-ph 카테고리에서 얻은 66,839개의 초록을 기반으로 하여 10,235개의 고유한 양자 물리학 개념을 정의하고 이를 동적 임베딩으로 모델링합니다. Word2Vec의 Skip-gram 모델을 사용하여 개념의 의미적 관계와 시간이 지남에 따라 변화하는 관계를 학습합니다.

- **Performance Highlights**: 제안된 방법은 연구 초록 내 개념의 동시 출현 예측에서 기존 방법들보다 뛰어난 성능을 보이며, 이는 과학 문헌에서 개념적 관계를 모델링하는 더 유연하고 유익한 방법임을 시사합니다.



### An Energy-Based Self-Adaptive Learning Rate for Stochastic Gradient Descent: Enhancing Unconstrained Optimization with VAV method (https://arxiv.org/abs/2411.06573)
- **What's New**: 기계 학습에서 학습률 최적화는 여전히 중요한 도전 과제로 남아 있으며, 모델 안정성과 효율적 수렴을 달성하는 데 필수적입니다. 최근에 제안된 Vector Auxiliary Variable (VAV) 알고리즘은 비제한 최적화 문제를 위해 설계된 새로운 에너지 기반 자기 조정 학습률 최적화 방법을 도입합니다. 이 방법은 역추적 없이 효율적인 에너지 근사를 촉진하는 보조 변수 $r$을 통합하여 무제한 에너지 소산 법칙을 준수합니다.

- **Technical Details**: VAV 알고리즘은 큰 학습률에서도 더 뛰어난 안정성을 입증하며, 훈련 과정 초기에 더 빠른 수렴을 달성합니다. 고전적인 방법인 Stochastic Gradient Descent (SGD)와 비교했을 때, VAV가 다양한 작업에서 월등히 우수한 성능을 보입니다. 본 논문은 에너지 소산 법칙에 대한 엄격한 증명을 제공하고 합리적인 가정하에서 알고리즘의 수렴성을 확립합니다. 또한, $r$은 실질적인 훈련 손실의 하한을 제공하여 알고리즘 성능을 더욱 향상시키는 새로운 스케줄링 접근 방식을 제공합니다.

- **Performance Highlights**: VAV 방법은 스테인에러를 효과적으로 모니터링하고, 이는 손실 함수의 동적 추적을 가능하게 합니다. 실제로 VAV는 기존의 학습률 조정 방법들과 비교하여 안정성과 빠른 수렴 속도를 보여, 다양한 기계 학습 작업에서 전반적인 성능을 향상시킵니다.



### Fitting Multiple Machine Learning Models with Performance Based Clustering (https://arxiv.org/abs/2411.06572)
- **What's New**: 이 논문에서는 전통적인 머신러닝 접근 방법이 데이터를 단일 생성 메커니즘에 기반한다고 가정하는 것을 벗어나, 실제 데이터는 다양한 특성과 타겟 값들 간의 관계에 따라 군집화(clustering)하는 새로운 프레임워크를 제안합니다. 이를 통해 여러 개의 별도의 모델을 학습하고, 스트리밍 데이터 분야에서도 효과적으로 적용될 수 있습니다.

- **Technical Details**: 제안된 군집화 알고리즘은 기능(feature) 벡터와 타겟(target) 데이터 간의 관계에 기반하여 데이터를 군집화합니다. 이 과정에서 Expectation-Maximization (EM) 기법을 활용하여 최적화 문제를 해결하며, 각 데이터 포인트가 함수에 대해 얼마나 멀리 떨어져 있는지를 비용 함수로 측정합니다. 또한, 새로운 데이터 배치가 들어올 때마다 기울기 하강법 기반의 학습을 통해 앙상블(ensemble) 모델의 가중치를 업데이트합니다.

- **Performance Highlights**: 기존의 단일 모델 접근 방식에 비해 제안된 방법은 여러 실제 데이터셋을 통해 성능이 크게 향상되는 것을 보여주었습니다. 특히, 군집화된 모델을 앙상블 방식으로 결합하여 얻은 성과는 전통적인 모델들보다 우수하여, 실세계의 다양한 데이터 환경에서 효과적인 적용 가능성을 제시합니다.



### Learning Loss Landscapes in Preference Optimization (https://arxiv.org/abs/2411.06568)
- **What's New**: 본 연구는 Preference Optimization (PO) 알고리즘의 성능에 영향을 미치는 데이터의 특성을 분석하였으며, 특히 노이즈가 포함된 데이터나 품질이 혼합된 데이터에서 발생하는 성능 저하 문제를 해결하기 위한 새로운 PO 프레임워크를 제안합니다.

- **Technical Details**: 우리는 mirror descent 기반의 새로운 PO 프레임워크를 도입하여 Direct Preference Optimization (DPO)와 Odds-Ratio Preference Optimization (ORPO) 방법을 특정 미러 맵의 선택에 따라 회복할 수 있음을 보여줍니다. 이 프레임워크 내에서, 진화적 전략(evolutionary strategies)을 사용하여 문제 상황을 다룰 수 있는 새로운 손실 함수(loss functions)를 발견하였습니다.

- **Performance Highlights**: 우리의 접근 방식으로 발견한 손실 함수는 여러 작업에서 DPO 및 ORPO에 비해 상당한 성능 개선을 이끌어냈습니다. 특히, 혼합 품질 데이터를 사용하여 대규모 언어 모델을 세밀히 조정하는 경우 ORPO보다 우수한 성능을 보여주었습니다.



### Thermodynamically-Informed Iterative Neural Operators for Heterogeneous Elastic Localization (https://arxiv.org/abs/2411.06529)
Comments:
          Submitted to Elsevier

- **What's New**: 이번 연구에서는 이종 물질 구조 상의 국부 탄성 변형 필드를 예측하는 전통적 수치 해석 방법의 한계를 극복하고자 Thermodynamically-informed Iterative Neural Operator (TherINO)를 제안하였습니다. 이를 통해 계수 필드를 직접 입력으로 사용하는 대신 열역학적 인코딩을 활용하여 솔루션 공간에서 반복적으로 예측을 개선합니다.

- **Technical Details**: 본 논문에서는 메조스케일 탄성체에 대해 이종 물질 구조의 변형을 예측하기 위한 Deep Equilibrium Model을 개발하였습니다. 핵심 필터링 작업으로 Fourier Neural Operator를 사용하고, Anderson Acceleration을 통해 발생하는 시스템의 방정식을 해결합니다. 이 모델은 계산 속도와 정확성 측면에서 새로운 접근 방식을 제시합니다.

- **Performance Highlights**: 제안된 모델은 기존의 최첨단 이터레이티브 신경 연산자 아키텍처와의 비교에서 효율성, 예측 정확도, 추론 안정성 면에서 우수한 성능을 보였습니다. 특히, 훈련 집합과는 다른 구조에서의 제로샷 외삽 테스트에서, 제안된 모델이 다른 방법들 보다 두 배에서 다섯 배 작은 예측 오류를 나타내는 결과를 보였습니다.



### Causal Representation Learning from Multimodal Biological Observations (https://arxiv.org/abs/2411.06518)
- **What's New**: 이번 연구는 다중 모드 생물학적 데이터 세트의 유연한 동원 조건을 개발하고, 생물학적 데이터를 이해하는 데 필요한 원칙 기반 방법을 제시합니다.

- **Technical Details**: 비모수 잠재 분포에 대한 유연한 접근을 적용하며, 다른 모드 간에 인과적 관계를 허용합니다. 또한 각 잠재 구성 요소에 대한 확인 가능성을 보장하여, 이전 연구의 부분 공간 식별 결과를 확장합니다.

- **Performance Highlights**: 실험을 통해 제안한 방법의 효과를 입증하였으며, 실제 인간 표현형 데이터 세트에서의 결과는 기존 의료 연구와 일관성을 보였습니다.



### Understanding the Role of Equivariance in Self-supervised Learning (https://arxiv.org/abs/2411.06508)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문에서는 전통적인 Self-Supervised Learning(SSL) 방법들이 유용한 특징(예: 색상)의 손실을 초래하고 있다는 점을 지적하며, 이를 보완하기 위한 Equivariant Self-Supervised Learning (E-SSL) 접근법에 대해 심도 있는 논의를 통해 본질적 작동 메커니즘을 이해하려고 합니다.

- **Technical Details**: E-SSL은 입력 데이터의 변형에 민감한 특징을 학습하는 방법으로, 특히 이미지 클래스와 변환 사이의 상호작용을 수학적인 관점에서 탐구합니다. 'Explaining-away' 효과를 활용하여 E-SSL의 일반화 능력을 정보 이론적 관점에서 분석하며, 데이터 변환의 영향을 정량적으로 분석하여 E-SSL 설계를 위한 세 가지 원칙(손실 변환, 클래스 관련성, 단축 가지치기)을 제시합니다.

- **Performance Highlights**: 이론적 발견을 바탕으로 최근 연구된 다양한 E-SSL 방법들을 재검토한 결과, 다수의 사례가 제안된 프레임워크 내에서 잘 설명될 수 있음을 보여줍니다. 이 연구는 E-SSL의 설계 및 이해에 대한 중요한 방향을 제시하며, 미래 연구에서 E-SSL의 가능성을 확장하는 데 기여할 것입니다.



### Diffusion Sampling Correction via Approximately 10 Parameters (https://arxiv.org/abs/2411.06503)
- **What's New**: 이 논문에서는 Diffusion Probabilistic Models (DPMs)의 샘플링 속도를 향상시키기 위해 PCA 기반의 Adaptive Search (PAS)를 제안합니다. 기존의 샘플링 지향 알고리즘과 달리 PAS는 최소한의 학습 가능한 파라미터와 훈련 비용으로 기존 솔버들을 최적화합니다.

- **Technical Details**: PAS는 PCA(주성분 분석)를 사용하여 고차원 샘플링 공간을 구성하는 몇 개의 직교 단위 기저 벡터를 얻고, 이들 벡터를 통해 올바른 샘플링 방향을 결정하기 위한 계수를 학습합니다. 이를 통해 기존의 빠른 솔버의 잘려진(truncation) 오류를 수정하여 샘플링 효율성을 높입니다.

- **Performance Highlights**: CIFAR10 데이터셋에서 PAS는 12개의 파라미터와 단일 NVIDIA A100 GPU에서 훈련 시간 1분 미만으로 DDIM의 FID 점수를 15.69에서 4.37로 향상시켰습니다. 이는 기존의 고비용 훈련 기반 알고리즘에 비해 매우 적은 비용으로 성능을 개선하는 효과를 보여줍니다.



### Individual Regret in Cooperative Stochastic Multi-Armed Bandits (https://arxiv.org/abs/2411.06501)
Comments:
          42 pages, 1 figure

- **What's New**: 이번 논문은 여러 에이전트가 임의의 연결된 통신 그래프를 통해 소통하는 환경에서의 확률적 다중 무장 도박기(Stochastic Multi-Armed Bandits, MAB)의 후회(regret)에 대해 연구한 것입니다. 특히, 협력적인 상황에서 비최적성 간격(sub-optimality gaps)과 통신 그래프의 지름(diameter)에 영향을 받지 않는 에이전트의 후회 경계(regret bound)를 제시하였습니다.

- **Technical Details**: 제시된 개별 후회 경계는 $	ilde{O}(	ext{sqrt}(AT/m)+A)$로, 여기서 $A$는 액션(action)의 수, $T$는 시간 범위(time horizon), $m$은 에이전트(agent)의 수를 나타냅니다. 충분한 수의 에이전트가 있을 경우 후회 경계는 단순히 $	ilde{O}(A)$로, 이는 협력적 확률적 MAB에 대해 그래프의 지름에 무관합니다.

- **Performance Highlights**: 이 연구는 비완전 연결(non-fully-connected) 통신 그래프에 대해 적용 가능한 개별 후회 경계를 처음으로 제시하여 기존 연구에서의 한계를 극복한 의미 있는 결과입니다.



### Towards Graph Neural Network Surrogates Leveraging Mechanistic Expert Knowledge for Pandemic Respons (https://arxiv.org/abs/2411.06500)
Comments:
          22 pages, 8 figures

- **What's New**: COVID-19 위기 동안 증거 기반 의사 결정을 이끌기 위해 기계적 모델이 필수적이라는 것이 입증되었습니다. 본 연구는 복잡한 기계적 모델과 데이터 기반 서로게이트 모델을 결합하여 공중 보건 전문가들이 모델을 즉각적으로 조정할 수 있도록 제안합니다.

- **Technical Details**: 이 연구는 공간적 및 인구통계적으로 조정된 감염병 모델을 기반으로 하며, 그래프 신경망(Graph Neural Networks, GNNs)을 활용하여 팬데믹 초기 데이터를 학습합니다. GNN은 빠른 실행 시간을 달성했으며, 특히 천 분의 일 초 이내의 응답 시간을 기록했습니다.

- **Performance Highlights**: 제안된 접근법은 즉각적인 실행 가능성을 제공하여 저장소 웹 애플리케이션에서 질병 동역학 모델을 통합할 수 있는 잠재력을 보입니다. 이 연구에서는 더 큰 변동성을 가진 데이터셋을 고려해야 함을 지적하고 있습니다.



### Mitigating covariate shift in non-colocated data with learned parameter priors (https://arxiv.org/abs/2411.06499)
- **What's New**: 본 논문은 데이터가 시간이나 공간에 따라 분산될 때 발생하는 covariate shift 문제를 해결하기 위해 Fragmentation-Induced covariate-shift Remediation (FIcsR) 방법을 제안합니다.

- **Technical Details**: FIcsR는 fragment의 covariate 분포와 표준 cross-validation 기준의 covariate 분포 간의 $f$-divergence를 최소화하여 데이터의 편향을 줄입니다. 이 방법은 기존의 중요도 가중치 방법과 동등함을 보여주며, 신경망의 과대파라미터화(overparametrized) 문제 때문에 수치적 해결이 어려워 Fisher Information 근사를 도출했습니다. 이를 통해 shift remediation의 전역 추정치를 계산하고 이를 최소화 목표에 priors로 통합했습니다.

- **Performance Highlights**: 여러 데이터 클래스에 대해 40개 이상의 데이터셋에서 광범위한 분류 실험을 통해 FIcsR를 도입했을 때 batch와 fold의 최첨단 성능보다 각각 5% 및 10% 이상 향상된 정확도를 보였습니다. 이 방법은 shift가 있는 여러 조건 하에서도 성능 저하가 느리게 발생하는 것을 보여주었습니다.



### Accelerating Large Language Model Training with 4D Parallelism and Memory Consumption Estimator (https://arxiv.org/abs/2411.06465)
- **What's New**: 이 연구는 Llama 아키텍처에서 4D 병렬 훈련을 위한 메모리 소비를 정밀하게 추정할 수 있는 공식을 제공하며, GPU 메모리 오버플로우를 방지하는 최적의 병렬화 구성 방법을 제시합니다.

- **Technical Details**: 연구에서는 A100 및 H100 GPU에서 수행된 454개의 실험을 기반으로 메모리 사용량을 분석하였으며, 메모리 소비 예측기가 GPU 메모리의 80% 이하일 때 훈련이 성공한다고 밝혔습니다. 이는 Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (DP), Context Parallelism (CP) 등을 포함한 4D 병렬 아키텍처에 적용됩니다.

- **Performance Highlights**: 제공된 간단한 공식은 메모리 오버플로우를 사전에 탐지할 수 있게 해주며, 실질적인 구성 탐색 공간을 크게 줄입니다. 또한, 454개의 실험 결과를 통해 최적의 4D 병렬 구성에 대한 경험적 통찰을 제공합니다.



### Predictors of disease outbreaks at continentalscale in the African region: Insights and predictions with geospatial artificial intelligence using earth observations and routine disease surveillance data (https://arxiv.org/abs/2411.06436)
Comments:
          15 pages, 3 figures, 7 tables

- **What's New**: 본 연구는 질병 발병 분석을 위한 컴퓨테이셔널 기법(computational techniques)을 적용하여 넓은 지리적 영역에서 주간 발병 현황을 분석하면서, 관련된 고해상도의 문화적 및 환경 데이터를 통합해 지역 수준 분석을 유지합니다.

- **Technical Details**: 우리는 말라리아(malaria), 콜레라(cholera), 수막염(meningitis), 황열병(yellow fever) 사례 수에 대해 전 세계적(global) 및 지역적(local) 공간 자기상관(spatial autocorrelation)을 적용했습니다. 이후, 기계 학습(machine learning)을 사용하여 이들 질병의 주간 존재 여부를 예측했습니다. 또한, 전파에 영향을 미치는 변수에 대해 기계 학습 특징 중요성(machine learning feature importance) 방법을 적용했습니다.

- **Performance Highlights**: 우리의 공간 자기상관 결과는 지리적 근접성(geographic nearness)이 중요하지만 효과와 공간의 차이가 있음을 보여줍니다. 흥미로운 핫스팟(hot spots)과 콜드스팟(cold spots), 공간 이상치(spatial outliers)를 확인했습니다. 기계 학습 모델은 말라리아에 대해 가장 좋은 F1 점수 0.96로 사례의 이진 클래스(binary class)를 추론했습니다. 기계 학습 특징 중요성 분석을 통해 발병에 영향을 미치는 중요한 문화적 및 환경적 요인을 밝혀냈습니다.



### Neuro-Symbolic Rule Lists (https://arxiv.org/abs/2411.06428)
- **What's New**: 이 논문에서는 Healthcare와 같은 민감한 분야에서 사용할 수 있도록 해석 가능한 머신러닝 모델을 위한 새로운 접근 방식을 제안합니다. NeuRules는 디스크리티제이션(discretization), 규칙 학습(rule learning), 규칙 순서(rule order)를 통합한 완전한 차별화 가능한 프레임워크입니다.

- **Technical Details**: NeuRules의 핵심은 규칙 리스트 학습 문제에 대한 연속 완화(continuous relaxation)를 수립하고, 온도 풀림(temperature annealing)을 통해 엄격한 규칙 리스트로 수렴하는 것입니다. 또한, NeuRules는 개별 특징의 디스크리티제이션(discretization) 및 그 조합을 conjunctive 규칙으로 학습하며 사전 처리(pre-processing)나 제한 없이 동작합니다.

- **Performance Highlights**: 광범위한 데이터셋에서 NeuRules는 기존의 조합 최적화(combinatorial optimization) 및 신경-상징(neuro-symbolic) 방법들을 일관되게 초월하며, 간단한 규칙과 복잡한 규칙, 그리고 그 순서를 효과적으로 학습합니다.



### UniGAD: Unifying Multi-level Graph Anomaly Detection (https://arxiv.org/abs/2411.06427)
Comments:
          Accepted by NeurIPS 2024. All codes can be found at this https URL

- **What's New**: 최근 발표된 UniGAD는 노드, 엣지, 그래프 등 여러 수준에서 동시에 이상치를 탐지할 수 있는 첫 번째 통합 프레임워크입니다.

- **Technical Details**: UniGAD는 MRQSampler라는 서브그래프 샘플러를 통해 각 수준의 객체를 그래프 수준의 작업으로 전환하고, GraphStitch Network를 통해 서로 다른 수준 간 정보를 통합합니다.

- **Performance Highlights**: UniGAD는 14개의 GAD 데이터셋에 대한 실험에서 기존의 단일 작업 전문 기법과 그래프 프롬프트 기반 접근 방식을 초월하는 성능을 보여주었으며, 강력한 제로샷(Zero-shot) 작업 이전 가능성을 제공했습니다.



### Ablation is Not Enough to Emulate DPO: How Neuron Dynamics Drive Toxicity Reduction (https://arxiv.org/abs/2411.06424)
- **What's New**: 이 연구는 안전성 미세 조정 알고리즘의 메커니즘, 특히 직접 선호 최적화(Direct Preference Optimization, DPO)를 통한 독성 감소 과정을 탐구하며, 기존의 설명이 불완전하다고 주장합니다.

- **Technical Details**: DPO 알고리즘은 가장 독성이 강한 MLP 뉴런의 활성화를 억제하여 독성 영역을 피하는 것이 아니라, 다수의 뉴런 그룹 간의 효과를 축적하여 독성을 감소시킵니다. 실험을 통해 DPO의 독성 감소 효과가 단지 억제된 독성 뉴런이 아닌, 반독성(anti-toxicity)을 촉진하는 방식으로도 기여함을 밝혔습니다. DPO의 적용으로 인해 상당수의 뉴런들이 오히려 독성을 증가시키는 경우도 관찰되었습니다.

- **Performance Highlights**: DPO 적용 후 오직 31.8%의 독성 감소가 가장 독성이 강한 뉴런에서 기인하고 있으며, 나머지는 여러 뉴런 그룹의 상호 작용 결과입니다. 이 연구 결과는 안전성 미세 조정 알고리즘이 독성 감소를 이루기 위해서는 뉴런 간의 복잡한 균형 작용이 필요함을 시사합니다.



### Locally Adaptive One-Class Classifier Fusion with Dynamic $\ell$p-Norm Constraints for Robust Anomaly Detection (https://arxiv.org/abs/2411.06406)
- **What's New**: 이 논문은 동적 $	ext{ℓ}_p$-norm 제약 조건을 통한 지역 적응 학습을 활용한 새로운 One-Class Classifier Fusion 접근 방식을 제안합니다. 이 방법은 데이터의 지역적 특성에 따라 융합 가중치를 동적으로 조정하여 앙상블 기반 이상 탐지에서의 근본적인 도전 과제를 해결합니다.

- **Technical Details**: 본 연구는 내부 지점 최적화 기법을 도입하여 기존 Frank-Wolfe 접근법보다 계산 효율성을 크게 향상시킵니다. 이로 인해 복잡한 시나리오에서 최대 19배의 속도 향상을 달성하였습니다. 제안된 프레임워크는 표준 UCI 벤치마크 데이터셋과 전문화된 시간 시퀀스 데이터셋에서 광범위하게 평가됩니다.

- **Performance Highlights**: Statistical validation을 통해 Skillings-Mack 테스트를 사용하여 제안된 방법이 기존 접근 방식에 비해 상당한 이점을 가지며, 순수 및 비순수 학습 시나리오 모두에서 일관된 상위 순위를 유지함을 확인하였습니다. 이러한 프레임워크는 지역 데이터 패턴에 적응할 수 있으면서도 계산 효율성을 유지하여, 빠르고 정확한 이상 탐지가 중요한 실시간 응용 프로그램에 특별히 가치가 있습니다.



### A Variance Minimization Approach to Temporal-Difference Learning (https://arxiv.org/abs/2411.06396)
- **What's New**: 이 논문은 전통적인 RL 알고리즘 대신 오류(minimization) 최소화를 위한 변동성 최소화(Variance Minimization, VM) 접근법을 도입하여 Bellman 오류(Bellman Error)의 변동성(Variance of Bellman Error, VBE)와 투영 Bellman 오류의 변동성(Variance of Projected Bellman Error, VPBE) 두 가지 목표를 제시합니다.

- **Technical Details**: 저자는 VMTD, VMTDC 및 VMETD 알고리즘을 도출하고, 변동성 최소화의 수렴성과 최적 정책 불변성(optimal policy invariance)의 증명을 제공합니다. 이 연구는 선형 근사화(value function approximation)에 기반한 시간 차 학습(Temporal Difference Learning) 알고리즘을 중점적으로 다룹니다.

- **Performance Highlights**: 실험 연구를 통해 제안된 알고리즘의 효과성을 검증하였으며, 새로운 방법론이 기존의 오류 최소화 접근법 대비 개선된 수렴 속도를 제공하는 것으로 나타났습니다.



### Local vs. Global Models for Hierarchical Forecasting (https://arxiv.org/abs/2411.06394)
- **What's New**: 이 연구는 계층적 시계열 예측의 정확성에 영향을 미치는 다양한 정보 활용법을 탐구하고, 지역 모델(local model)에 비해 글로벌 예측 모델(Global Forecasting Models, GFM)의 장점을 제시합니다. GFMs는 여러 계층 및 시계열의 정보를 활용하여 예측 성능을 향상시킵니다.

- **Technical Details**: 본 연구에서는 LightGBM 기반의 두 가지 구체적인 GFM을 도입하며, 예측의 일관성을 유지하는 정리 방법(reconciliation methods)인 Bottom-Up, Top-Down 및 Minimum Trace(MinT)을 사용합니다. 예측의 정확성은 Mean Absolute Scaled Error (MASE)와 Multiple Comparisons with the Best (MCB) 테스트를 통해 평가됩니다.

- **Performance Highlights**: GFM은 지역 모델 및 전통적인 방법(Exponential Smoothing, ARIMA)에 비해 부드러운 정확성과 낮은 모델 복잡성을 보여줍니다. 실험 결과 GFMs가 계층적 예측에 있어서 뛰어난 성과를 거두었음을 확인했습니다.



### CausalStock: Deep End-to-end Causal Discovery for News-driven Stock Movement Prediction (https://arxiv.org/abs/2411.06391)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 업계 최초로 뉴스 기반의 다중 주식 움직임 예측을 위한 "CausalStock"이라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 주식 간의 시간적 인과 관계를 발견하며, 노이즈가 포함된 뉴스 데이터에서 유용한 정보를 효과적으로 추출합니다.

- **Technical Details**: CausalStock 모델은 라그 의존성(lag-dependent) 템포럴 인과 발견 메커니즘을 적용하여 주식 간의 인과 그래프 분포를 모델링합니다. 또한, 대형 언어 모델(LLMs)의 텍스트 평가 능력을 활용한 Denoised News Encoder를 통해 노이즈가 많은 뉴스 텍스트에서 유용한 정보를 추출합니다. Functional Causal Model(FCM)을 사용하여 발견된 인과 관계를 캡슐화하고 주식의 움직임을 예측합니다.

- **Performance Highlights**: CausalStock은 미국, 중국, 일본, 영국 시장에서 수집한 6개의 실제 데이터세트에서 뉴스 기반의 다중 주식 움직임 예측 및 다중 주식 움직임 예측 작업 모두에서 강력한 기준선(baseline)을 초과하는 성능을 보였습니다. 인과 관계를 활용하여 CausalStock은 명확한 예측 메커니즘과 뛰어난 설명 가능성을 제공합니다.



### Self-Training Meets Consistency: Improving LLMs' Reasoning With Consistency-Driven Rationale Evaluation (https://arxiv.org/abs/2411.06387)
Comments:
          under review

- **What's New**: 본 연구에서는 CREST(Consistency-driven Rationale Evaluation for Self-Training)라는 새로운 자기 훈련 프레임워크를 제안합니다. 이 프레임워크는 각 논리를 후속 질문을 통해 평가하고 이를 활용하여 모델을 훈련하는 방법을 포함하고 있습니다.

- **Technical Details**: CREST는 두 가지 방법을 사용하여 훈련을 수행합니다: (1) 후속 질문에서 자주 잘못된 정답을 도출하는 논리를 걸러내는 rational filtering과 (2) 원본 및 후속 질문의 평가 결과를 기반으로 혼합된 선호도를 학습하는 preference learning입니다.

- **Performance Highlights**: 세 가지 질문-응답 데이터셋에서 실험을 수행한 결과, CREST는 논리의 강건성과 정확성을 향상시켰으며, 이전 자기 훈련 접근법보다 더 나은 추론 능력을 보여주었습니다.



### Phantom: Constraining Generative Artificial Intelligence Models for Practical Domain Specific Peripherals Trace Synthesizing (https://arxiv.org/abs/2411.06376)
- **What's New**: Phantom는 PCIe TLP 트레이스 생성을 생성적 AI 문제로 취급하면서 PCIe 고유의 제약 조건을 통합한 최초의 프레임워크입니다. 이 논문은 Phantom의 효과성을 실험적으로 증명하였습니다.

- **Technical Details**: Phantom은 TLP (Transaction Layer Packet) 작업과 RGB 삼중체의 매핑을 설계하고, TLP 생성을 이미지 생성 작업으로 재정의합니다. 이 과정에서, 트레이스 생성의 세 가지 단계인 정규화(normalization), 보정(calibration), 디코딩(decoding)으로 구성됩니다. 그리고 피험자 특성이 반영된 생성 패턴을 적용할 수 있는 방법론이 구현되었습니다.

- **Performance Highlights**: Phantom은 실제 PCIe 네트워크 인터페이스 카드에 대한 TLP 트레이스를 생성하며, 기존 모델보다 최대 1000배의 성능 향상과 Fréchet Inception Distance (FID)에서 최대 2.19배 개선된 결과를 보였습니다.



### BayesNAM: Leveraging Inconsistency for Reliable Explanations (https://arxiv.org/abs/2411.06367)
Comments:
          Under Review

- **What's New**: Neural additive model (NAM)의 불일치 현상을 연구하고 이를 설명할 수 있는 새로운 프레임워크, Bayesian Neural Additive Model (BayesNAM),을 제안합니다. BayesNAM은 Bayesian neural networks와 feature dropout을 결합하여 불일치 정보를 활용할 수 있도록 지원합니다.

- **Technical Details**: BayesNAM은 Bayesian neural networks을 기반으로 하여 feature dropout을 도입하여 NAM의 불일치성을 효과적으로 처리합니다. 기존 NAM과 동일한 데이터셋과 아키텍처를 사용하였음에도 불구하고, 랜덤 시드의 변화로 불일치하는 결과를 보이는 현상을 설명합니다. 이론적 분석을 통해 feature dropout이 불일치 정보를 포착하는 효과적인 방법임을 증명합니다.

- **Performance Highlights**: BayesNAM은 부족한 데이터 또는 모델의 구조적 한계를 비롯한 잠재적인 문제를 효과적으로 식별하고, 더 신뢰할 수 있는 설명을 제공합니다. 실험을 통해 BayesNAM이 NAM보다 더욱 해석 가능하고 믿을 수 있는 결과를 도출함을 보여줍니다.



### Optimized Inference for 1.58-bit LLMs: A Time and Memory-Efficient Algorithm for Binary and Ternary Matrix Multiplication (https://arxiv.org/abs/2411.06360)
- **What's New**: 본 논문에서는 1.58-bit LLM의 추론 효율성을 높이기 위한 알고리즘을 제안합니다. 이를 통해 메모리 사용량과 추론 시간을 최적화하여 LLM을 보다 접근 가능하고 경제적으로 만드는데 기여하고자 합니다.

- **Technical Details**: 제안하는 알고리즘은 훈련된 모델의 가중치 행렬을 전처리하여 인덱스를 생성하며, 이 인덱스를 활용하여 효율적으로 행렬 곱셈을 수행합니다. 특히, 행렬 곱셈의 시간 복잡도를 O(n²/log(n))으로 보장하여 기존의 O(n²) 방식보다 로그적 개선을 이루었습니다.

- **Performance Highlights**: 실험 결과, 제안하는 알고리즘은 최대 29배의 추론 시간 단축과 최대 6배의 메모리 사용량 감소를 달성하였습니다. 이는 LLM의 실용성과 접근성을 크게 향상시킬 수 있는 결과입니다.



### Deep Active Learning in the Open World (https://arxiv.org/abs/2411.06353)
- **What's New**: 이번 논문에서는 open-world 환경에서 새로운 OOD(Out-Of-Distribution) 클래스의 통합을 위한 ALOE라는 새로운 active learning 알고리즘을 소개합니다. ALOE는 두 가지 단계로 구성되어 있으며, 첫 번째 단계에서 다양성 샘플링(diversity sampling)을 통해 대표적인 예제를 선택하고, 두 번째 단계에서는 에너지 기반 OOD 탐지를 통해 알려지지 않은 클래스의 우선 순위를 정합니다.

- **Technical Details**: ALOE(Active Learning in Open-world Environments) 알고리즘은 두 단계의 접근 방식을 사용하여 open-world 환경에서 발생할 수 있는 문제를 해결합니다. 첫 번째 단계에서는 다양성 샘플링을 통해 다양한 데이터 분포를 포괄하는 대표 예제를 선택합니다. 두 번째 단계에서는 에너지 점수 기능을 사용하여 클러스터 내의 예제를 순위 매겨 주목해야 할 OOD 클래스를 우선적으로 파악합니다.

- **Performance Highlights**: ALOE는 ImageNet-LT와 같은 장기적 불균형 이미지 분류 데이터세트에서 실험을 수행했으며, 무작위 샘플링에 비해 동일한 정확도를 달성하는 데 70%의 주석 비용을 절감하였습니다. 모든 실험 설정에서 ALOE가 가장 우수한 성능을 보이며, 알려진 클래스의 성능 향상과 새로운 클래스 발견 사이의 중요한 트레이드오프를 발견하였습니다.



### Client Contribution Normalization for Enhanced Federated Learning (https://arxiv.org/abs/2411.06352)
Comments:
          Accepted at IEEE INDICON 2024

- **What's New**: 이 논문은 Federated Learning (FL)에서 통계적 이질성을 처리하기 위한 새로운 접근법을 제안합니다. 구체적으로, 로컬에서 훈련된 모델로부터 추출된 평균 잠재 표현 (mean latent representations)을 활용하여 클라이언트의 기여도를 정규화하고, 이를 통해 중앙 서버가 집계 과정에서 이질성을 평가하고 조정할 수 있도록 합니다.

- **Technical Details**: 본 연구는 평균 잠재 표현을 이용한 정규화 스킴을 제안하며, 이는 기존 FL 알고리즘과 원활하게 통합되도록 설계되었습니다. 이 방법은 비 독립적이고 동일하게 분포되지 않은 (non-IID) 데이터 환경에서도 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 여섯 가지 FL 기법(FedAvg, FedProx, FedBABU, FedNova, SCAFFOLD, SGDM)을 통해 실험한 결과, 모델의 정확성과 일관성이 향상되었음을 확인했습니다. 이 연구는 통계적 이질성 문제를 해결하기 위한 계산적으로 효율적인 솔루션을 제공하여 신뢰할 수 있는 머신러닝 모델 개발에 기여합니다.



### Activation Map Compression through Tensor Decomposition for Deep Learning (https://arxiv.org/abs/2411.06346)
- **What's New**: 이번 연구는 Edge AI에서의 백프로퍼게이션(backpropagation)의 메모리 부족 문제를 해결하기 위해 텐서 분해(tensor decomposition) 기법을 활용하여 활성화 맵(activation map)의 압축(compression)을 제안합니다.

- **Technical Details**: 이 연구는 Singular Value Decomposition(SVD)와 High-Order Singular Value Decomposition(HOSVD)을 사용하여 백프로퍼게이션의 메모리 발자국(memory footprint)을 줄이는 방법에 중점을 둡니다. 저차원 분해(low-order decomposition)는 많은 메모리 절약 효과를 가지며, 학습에 필수적인 특성(features)을 보존합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 최신 기술들과 비교하여 일반화(generalization)와 메모리 발자국 사이의 균형에서 Pareto-superiority를 달성하였습니다.



### CRTRE: Causal Rule Generation with Target Trial Emulation Framework (https://arxiv.org/abs/2411.06338)
- **What's New**: 이 연구에서는 비선형 환경에서 인과 추론 및 모델 해석력을 향상시키기 위한 새로운 방법인 causal rule generation with target trial emulation framework (CRTRE)를 소개합니다. 이 방법은 랜덤화 시험 설계 원칙을 적용하여 연관 규칙의 인과 효과를 추정하며, 실제 의료 데이터에서 뛰어난 성능을 보여줍니다.

- **Technical Details**: CRTRE는 연관 규칙 마이닝 알고리즘을 사용하여 특징으로서 규칙을 추출합니다. 또한, 기능 F(x)을 이용하여 변수 간의 상호작용을 포착하고 테일러 전개를 수행합니다. 이 방법은 선형 및 비선형 맥락 모두에서 변수 간의 독립성을 식별하는 타겟 시험 에뮬레이션 프레임워크를 사용합니다.

- **Performance Highlights**: 여섯 개의 건강 관리 데이터 세트에서 광범위한 실험을 수행하여 CRTRE의 우수한 성능을 입증했습니다. 특히, 진단 정확도에서 Esophageal Cancer, Heart Disease, 및 Cauda Equina Syndrome 예측 작업에서 각각 0.789, 0.920, 0.300을 달성하며 기존 모델을 초과했습니다. MIMIC-III 및 MIMIC-IV 데이터 세트에서 AUC Macro 점수에서도 기존 최첨단 모델을 능가했습니다.



### Regret Minimization and Statistical Inference in Online Decision Making with High-dimensional Covariates (https://arxiv.org/abs/2411.06329)
- **What's New**: 이번 논문에서는 sparse linear context bandit 모델을 바탕으로 고차원 온라인 의사결정에서의 후회 최소화(regret minimization)와 통계적 추론(statistical inference)의 상호작용을 분석합니다. 저자는 $eta$-greedy bandit 알고리즘을 결정과제로 통합하고, sparse bandit 매개변수를 추정하기 위한 하드 임계값(thresholding) 알고리즘을 개발했습니다. 또한 inverse propensity weighting을 활용한 비편향(debiasing) 방법 기반의 추론 프레임워크를 도입했습니다.

- **Technical Details**: 저자들은 margin 조건 하에 O(T^{1/2}) 후회 혹은 전통적인 O(T^{1/2})-일관 추론(consistency inference)을 달성하는 방법을 제시합니다. 다양한 공변량 조건(covariate condition)을 만족하는 경우, 순수 탐색 없이 greedy bandit 알고리즘을 사용하여 최적의 O(log T) 후회와 O(T^{1/2})-일관 추론을 동시에 달성할 수 있음을 입증했습니다. 또한, 단순 표본 평균(sample mean) 추정기가 최적 정책 값에 대한 유효한 추론을 제공할 수 있음을 보여줍니다.

- **Performance Highlights**: Warfarin 복용량 데이터에 대한 수치적 시뮬레이션(numerical simulations)과 실험(experiments)을 통해 제안한 방법들의 효과가 검증되었습니다. 이 연구는 온라인 개인화 의사결정의 다양한 응용 분야에서 중요한 의미를 가지고 있으며, 의료 및 마케팅 전략에서의 실제 예시로 통계적 추론의 필요성을 강조합니다.



### When are dynamical systems learned from time series data statistically accurate? (https://arxiv.org/abs/2411.06311)
Comments:
          in NeuRIPS 2024

- **What's New**: 이 논문에서는 시간 시계열 데이터로부터 학습된 복잡한 동적 모델의 일반화(generalization) 능력을 설명하기 위해 에르고딕 이론적 접근 방식을 제안합니다. 이는 복잡한 동적 시스템의 신경망 표현을 정의하고 분석하는 새로운 방법을 제시하며, 특히 카오틱 시스템에 대한 일반화 가능성을 다룹니다.

- **Technical Details**: 저자들은 비선형 카오틱 시스템으로부터 얻어진 데이터를 기반으로 신경망 모델, 특히 Neural ODEs의 일반화 실패에 대한 이론적 근거를 제공합니다. 이 연구는 Jacobian 정보를 추가하는 것이 훈련 중 통계적 정확성을 개선함을 보여줍니다. 다양한 신경망 파라미터화(Multi-Layer Perceptrons, ResNets, Fourier Neural layers, RNNs 등)에 대한 결과를 검증합니다.

- **Performance Highlights**: ‘MSE_MLP’ 모델이 기본 시스템의 궤적과 상당히 다른 비현실적인 궤적을 생성한 반면, ‘JAC_MLP’ 모델은 로렌츠 ‘63 어트랙터와 물리적 분포를 정확하게 재현했습니다. 논문에서는 일반화의 새로운 경계를 개발하고, 동적 시스템에서의 학습 실패 모드를 엄밀하게 설명합니다.



### Intelligent Fault Diagnosis of Type and Severity in Low-Frequency, Low Bit-Depth Signals (https://arxiv.org/abs/2411.06299)
- **What's New**: 이 연구는 단일 마이크로폰(single microphone)과 데이터 기반(data-driven) 방법론을 활용해 회전 기계(rotating machinery)에서의 Intelligent Fault Diagnosis (IFD)에 초점을 맞추었습니다. 42가지의 결함 종류와 심각도를 효과적으로 진단하는 방법을 제시합니다.

- **Technical Details**: 연구는 불균형한 MaFaulDa 데이터셋의 음향 데이터를 활용하여 높은 성능과 낮은 자원 소비 간의 균형을 이루기 위한 여러 가지 테스트 구성요소를 포함했습니다. 여기에는 샘플링(sampling), 양자화(quantization), 신호 정규화(signal normalization), 무음 제거(silence removal), Wiener 필터링(Wiener filtering), 데이터 스케일링(data scaling), 윈도잉(windowing), 증강(augmentation), 그리고 XGBoost를 사용한 분류기 조정(classifier tuning)이 포함됩니다.

- **Performance Highlights**: 시간(time), 주파수(frequency), 멜 주파수(mel-frequency), 통계적 특징(statistical features)을 분석한 결과, 8 kHz, 8-bit 설정에서 6개의 boosting 나무(tree)를 사용하여 99.54%의 높은 정확도(accuracy)와 99.52%의 F-Beta 점수를 기록했습니다. MFCCs와 첫 번째 및 두 번째 델타(delta)를 활용했을 때 97.83%의 정확도와 97.67%의 F-Beta 점수를 나타냈고, greedy wrapper 접근 방식을 적용하여 50개의 선택된 특징을 사용했을 때 96.82%의 정확도와 98.86%의 F-Beta 점수를 달성했습니다.



### TinyML NLP Approach for Semantic Wireless Sentiment Classification (https://arxiv.org/abs/2411.06291)
Comments:
          Submitted for WCNC-2025, Under Review

- **What's New**: 이 논문에서는 text emotion classification을 위한 새로운 두 가지 프레임워크를 제안합니다. 하나는 Federated Learning(FL) 기반이며, 다른 하나는 Split Learning(SL) 기반입니다. 이 연구는 연산 자원 문제와 데이터 개인정보 보호 문제를 동시에 해결하는 방안을 모색합니다.

- **Technical Details**: FL은 사용자들이 로컬 모델을 훈련하고 중앙 서버와 업데이트를 공유하는 분산 학습 방법입니다. SL은 초기 모델 레이어에서 사용자의 활성화만 전송하여 연산 부담을 줄이고 데이터 개인정보를 강화합니다. 두 기술 모두 채널 환경의 노이즈와 페이딩을 고려하여 성능을 비교합니다.

- **Performance Highlights**: SL은 처리가능한 전력과 CO2 배출을 줄이면서 높은 정확도를 유지합니다. 반면, FL은 효율성과 개인정보 보호 사이의 균형을 제공합니다. 실험 결과는 제안된 방법이 극단적인 조건에서도 강력함을 입증하며, 전체 에너지 소비와 CO2 배출 감소에 기여함을 보여줍니다.



### SPIKANs: Separable Physics-Informed Kolmogorov-Arnold Networks (https://arxiv.org/abs/2411.06286)
- **What's New**: 이 논문은 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)의 새로운 변형인 분리 가능한 물리 정보 콜모고로프-아널드 네트워크(Separable Physics-Informed Kolmogorov-Arnold Networks, SPIKANs)를 소개합니다. SPIKANs는 KANs(Kolmogorov-Arnold Networks)의 구조를 활용하여 다차원 편미분 방정식(Partial Differential Equations, PDEs)의 해결을 용이하게 합니다. 이는 훈련 속도를 높이면서도 정확도를 유지하는 새로운 방법론입니다.

- **Technical Details**: SPIKANs는 KAN의 원칙을 기반으로 하여 다차원 PDE를 분리 가능한 구성 요소로 분해합니다. 각 차원은 개별 KAN에 의해 처리되므로 연산 복잡도가 크게 줄어듭니다. 이 아키텍처는 다차원 문제 해결 시 요구되는 훈련 포인트 수를 O(N)으로 줄여, 제각기 O(N) 포인트를 처리할 수 있도록 합니다. 즉, 차원 수 d에 대한 훈련 포인트 수가 O(N^d)에서 O(N)으로 감소합니다.

- **Performance Highlights**: SPIKANs는 4개의 벤치마크 문제에서 테스트되어, KANs 및 PIKANs와 비교하여 뛰어난 확장성 및 성능을 보여줍니다. 특히, 계산 시간 및 메모리 측면에서 월등히 개선된 결과를 나타내었으며, 복잡한 고차원 PDE 문제를 해결하는데 유망한 가능성을 갖추고 있습니다.



### Multi-View Majority Vote Learning Algorithms: Direct Minimization of PAC-Bayesian Bounds (https://arxiv.org/abs/2411.06276)
- **What's New**: PAC-Bayesian 프레임워크를 다중 뷰 학습에 적용하여 Rényi divergence를 기반으로 하는 새로운 PAC-Bayesian 경계를 도입했습니다. 이로써 Kullback-Leibler divergence보다 더 정교한 복잡도 측정을 제공합니다.

- **Technical Details**: 첫째, 다중 뷰 학습을 위한 일반 PAC-Bayesian 경계를 확장했습니다. 둘째, Rényi divergence를 기반으로 한 1차 및 2차 오라클 PAC-Bayesian 경계를 제안했습니다. 셋째, 다중 뷰 학습을 위한 효율적인 최적화 알고리즘을 개발했으며, 자가 경계(self-bounding) 속성을 포함합니다.

- **Performance Highlights**: 이 연구는 다중 출처 데이터에서의 적용 가능성을 높이기 위해 기존의 PAC-Bayes 접근 방식을 개선하고, 새로운 이론적 기여를 바탕으로 실행 가능한 최적화 절차를 제시하여 실질적인 효과를 기대할 수 있습니다.



### Federated Split Learning for Human Activity Recognition with Differential Privacy (https://arxiv.org/abs/2411.06263)
Comments:
          Accepted to IEEE Consumer Communications and Networking Conference (CCNC), 6 pages

- **What's New**: 본 논문에서는 엣지 네트워크에서 차별적 프라이버시(Differential Privacy, DP)를 적용한 새로운 연합 분할 학습(Federated Split Learning, FSL) 기반의 인간 활동 인식(Human Activity Recognition, HAR) 프레임워크를 제안합니다. 이 프레임워크는 가속도계(accelerometer) 및 자이로스코프(gyroscope) 데이터를 활용하여 HAR 정확도를 크게 향상시킵니다.

- **Technical Details**: FSL-DP 프레임워크는 사용자 데이터 프라이버시를 보장하면서도, 클라이언트에서 모델의 일부를 훈련시킨 후 결과를 서버로 전송하여 훈련하는 구조를 갖추고 있습니다. 또한, 이 시스템에서는 ED(Edge Devices)가 클라이언트 모델의 전방 전파 및 후방 전파 일부를 수행하고, 서버는 나머지 모델 훈련을 통해 계산 부담을 줄입니다. DP 메커니즘은 정보가 공격에 노출될 가능성을 줄이기 위해 활성화된 데이터에 마스크를 추가합니다.

- **Performance Highlights**: 본 연구의 FSL 방법은 전통적인 학습 방법과 비교했을 때 훈련 성능을 크게 향상시키고, 훈련 지연 시간을 줄이는 데 성공했습니다. 시뮬레이션 결과는 DP 메커니즘이 HAR 성능에 미치는 영향을 다양한 훈련 설정에서 검증하였으며, FSL 프레임워크가 정확성 및 손실 메트릭에서 기존의 연합 학습 모델을 초월한 것으로 나타났습니다.



### Theoretical Analysis of Learned Database Operations under Distribution Shift through Distribution Learnability (https://arxiv.org/abs/2411.06241)
Comments:
          Appeared in ICML'24 (oral)

- **What's New**: 본 논문에서는 동적 데이터셋에서의 학습된 모델(learned models)의 성능을 이론적으로 특성화한 첫 번째 연구 결과를 제시합니다. 데이터셋의 변화와 데이터 분포의 변화에 따른 성능 저하 문제를 다룹니다.

- **Technical Details**: 우리는 데이터베이스의 인덱싱(indexing), 카디널리티 추정(cardinality estimation), 정렬(sorting) 작업에 대해 학습된 모델의 성능 저하를 이론적으로 분석합니다. 새로운 이론적 특성과 모델의 장점을 비학습 모델(non-learned methods)과 비교하여 성능 경계를 제공합니다. 이를 바탕으로 분포 학습 가능성(distribution learnability framework)과 새로운 이론적 도구를 개발합니다.

- **Performance Highlights**: 학습된 모델이 비학습 대안보다 우수하게 작동하는 조건을 명확하게 설명하며, 데이터셋의 변화가 모델 성능에 미치는 영향을 체계적으로 분류합니다.



### Zero-Shot NAS via the Suppression of Local Entropy Decreas (https://arxiv.org/abs/2411.06236)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 연구는 Zero-Shot NAS의 성능 평가를 가속화하기 위해 네트워크 아키텍처 토폴로지를 활용하여 더 빠르고 효율적인 zero-cost 프록시(Proxy)를 제안합니다. 이는 기존 프록시들이 요구하는 backpropagation이나 입력 데이터 의존성을 제거하여 계산 속도를 한층 높였습니다.

- **Technical Details**: 연구에서는 로컬 엔트로피(local entropy) 감소에 대한 이론 분석을 통해 네트워크 아키텍처가 특성 맵(feature maps)의 로컬 엔트로피를 낮출 수 있음을 증명합니다. 이를 바탕으로 Suppression of Local Entropy Decrease (SED)라는 새로운 프록시를 제안하였고, 이 프록시는 벤치마크에서 기존의 SOTA 프록시들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: SED 프록시는 NATS-Bench-TSS에서 아키텍처 평가를 3.4e-5 초 내에 수행하면서 기존 SOTA 방식들보다 세 배 빠른 결과를 기록하였습니다. SED 기반 NAS는 한 초 만에 더 높은 정확도와 적은 파라미터 수를 가진 아키텍처를 선정할 수 있습니다.



### Early Prediction of Natural Gas Pipeline Leaks Using the MKTCN Mod (https://arxiv.org/abs/2411.06214)
Comments:
          12 pages, 6 figures

- **What's New**: 이 연구는 기존의 이상 탐지 방식과는 달리, 내부 파이프라인 데이터를 활용하여 가스 파이프라인 유출을 조기에 예측하는 최초의 모델을 개발했습니다. 이는 장거리 파이프라인의 안전을 위한 중요한 기여입니다.

- **Technical Details**: 이 모델은 dilated convolution을 사용하여 시계열 데이터의 장기 의존성(long-term dependency)을 캡처하고, Kolmogorov-Arnold Network (KAN)를 통합하여 샘플 불균형(sample imbalance) 문제를 해결합니다. MKTCN(Multi-classification Temporal Convolutional Network)은 이 두 가지 문제를 방지하면서 예측 성능을 향상시킵니다.

- **Performance Highlights**: MKTCN 모델은 실제 데이터 불균형이 심한 상황에서도 뛰어난 일반화 및 분류 성능을 발휘하며, 최대 5000초 이내의 가스 유출을 효과적으로 예측할 수 있음을 실험을 통해 검증했습니다.



### Multistage non-deterministic classification using secondary concept graphs and graph convolutional networks for high-level feature extraction (https://arxiv.org/abs/2411.06212)
Comments:
          13 Pages, 15 figures, and 4 Tables

- **What's New**: 본 논문은 다단계 비결정론적 분류 방법을 제안하며, 이는 보조 개념 그래프(secondary conceptual graph)와 그래프 컨볼루션 네트워크(Graph Convolutional Networks, GCN)를 기반으로 하고 있습니다.

- **Technical Details**: 제안된 방법은 1) GCN을 사용하여 12개의 고수준 특성을 추출 및 생성하고, 2) 불완전한 비결정론적 모델을 사용하여 예측을 수행하고, 3) 개념 그래프에 기반하여 최종 예측을 수행하는 여러 단계로 구성됩니다.

- **Performance Highlights**: Cora, Citeseer 및 PubMed의 세 가지 데이터셋에서 각각 96%, 93%, 95%의 분류 정확도를 달성하여 최신 방법보다 성능이 우수함을 입증하였습니다.



### Advanced Wildfire Prediction in Morocco: Developing a Deep Learning Dataset from Multisource Observations (https://arxiv.org/abs/2411.06202)
- **What's New**: 이번 연구는 모로코의 고유한 지리적 및 기후적 도전에 맞춘 새로운 야생화재 예측 데이터세트를 소개합니다. 이 데이터세트는 위성 관측 및 지상 관측소 데이터를 통합하여 식생 건강(NDVI), 인구 밀도, 토양 수분 수준 및 기상 데이터를 포함하고 있습니다.

- **Technical Details**: 모델은 유지보수가 용이한 열(columnar) 데이터로 구성되어 있으며, ML(머신러닝) 및 DL(딥러닝) 알고리즘을 활용하여 기존 예측 모델에 비해 월등한 성능을 보입니다. 데이터 통합을 통해 간단하고 효율적인 예측을 가능하게 하였고, 다양한 환경 지표들을 활용하여 다음 날의 야생화재 발생을 90%의 정확도로 예측할 수 있습니다.

- **Performance Highlights**: 본 연구의 결과로, 새로운 데이터세트를 활용한 모델들은 기존 모델들보다 향상된 성능을 보여주며, 야생화재 다이내믹스를 정확히 포착하였습니다. 이 데이터세트는 공개적으로 이용 가능하여, 과학적 협업을 촉진하고, 다른 지역의 유사한 환경적 문제 해결을 위한 확장 가능한 모델을 제공합니다.



### Weak to Strong Learning from Aggregate Labels (https://arxiv.org/abs/2411.06200)
Comments:
          19 pages

- **What's New**: 본 논문은 집합 레이블(aggregate labels)로부터 약한 학습기(weak learner)를 사용하여 강한 학습기(strong learner)를 만드는 가능성을 이론적으로 탐구합니다. 기존의 전통적인 감독 학습(supervised learning)에서는 개별 데이터 샘플에 대해 레이블이 제공되지만, 본 연구에서는 평균 레이블로 구성된 특정 훈련 세트에서 학습하는 방법에 중점을 둡니다. 특히, 라벨 비율 학습(LLP) 및 다중 샘플 학습(MIL)에서 강화(boosting) 기법 적용의 불가능성을 증명합니다.

- **Technical Details**: 제안된 알고리즘은 라벨 비율 학습(LLP) 및 다중 샘플 학습(MIL) 상황에서 사용되는 약한 학습기를 강한 학습기로 전환하는 방법을 모색합니다. 특히, LLP에서는 약한 학습기를 사용해 적절하게 지시된 소형 샘플 세트에서 강한 학습기를 얻을 수 있는 가능성을 보입니다. 강화 불가능성을 입증하기 위해, 약한 분류기의 정확도가 1 미만일 때 LLP 및 MIL 문제에 대한 특별한 가방 모음을 구성합니다.

- **Performance Highlights**: 제시된 방법은 실제 데이터 세트 3개와 합성 데이터 세트 2개에서 경험적으로 검증되었습니다. 다양한 정확도 수준에서 약한 학습기를 사용하여 소형 가방에서 강한 학습기를 생성할 수 있는 놀라운 결과를 보여줍니다. 이는 집합 레이블로부터 강한 학습기를 생성하는 첫 번째 이론적 연구로, 해당 알고리즘은 유효성과 효율성을 특징으로 합니다.



### State Chrono Representation for Enhancing Generalization in Reinforcement Learning (https://arxiv.org/abs/2411.06174)
- **What's New**: 이 논문에서는 강화 학습에 있어 이미지 기반 입력에서 강력하고 일반화 가능한 상태 표현을 구축하기 위한 새로운 접근법인 State Chrono Representation (SCR)을 제안합니다.

- **Technical Details**: SCR은 bisimulation metric learning에서 업데이트 단계에 광범위한 시간 정보를 통합하여 상태 메트릭 기반 표현을 보강합니다. 이 방식은 현재 및 장기 미래 상태에 대한 누적 보상을 고려하여 미래 동역학 내에서 상태 거리를 학습합니다. 두 개의 별도의 상태 인코더를 훈련하여 개별 상태의 표현을 생성하고 상태와 미래 상태 간의 관계를 캡슐화하는 순차 임베딩을 생성합니다.

- **Performance Highlights**: DeepMind Control 및 Meta-World 환경에서 수행된 광범위한 실험에서 SCR은 도전적인 일반화 작업에서 다른 최신 메트릭 기반 방법들보다 더 나은 성능을 보였습니다.



### HiHa: Introducing Hierarchical Harmonic Decomposition to Implicit Neural Compression for Atmospheric Data (https://arxiv.org/abs/2411.06155)
- **What's New**: 본 논문에서는 대기 데이터를 위한 새로운 압축 기술인 Hierarchical Harmonic decomposition implicit neural compression (HiHa)를 제안합니다. 기존의 Implicit Neural Representation (INR) 기반 압축의 한계를 극복하기 위한 접근법입니다.

- **Technical Details**: HiHa는 대기 데이터를 여러 개의 복잡한 조화(harmonic)로 분해하여 다중 주파수 신호로 나눈 후, 각 조화에 대해 주파수 기반 계층 압축 모듈을 적용합니다. 이 모듈은 희소 저장(sparse storage), 다중 스케일 INR, 반복 분해 서브 모듈로 구성됩니다. 또한, 시간적 연속성을 활용한 시간 잔여 압축 모듈이 추가되어 압축 속도를 증가시킵니다.

- **Performance Highlights**: 실험 결과, HiHa는 압축 정밀도와 성능 모두에서 다른 주요 압축기 및 INR 기반 방법을 초월하는 성능을 보여주었습니다. 특히, HiHa는 200배 이상의 압축비를 기록하며 43초 이내에 1e-3의 에러 범위로 압축할 수 있습니다.



### Online Parallel Multi-Task Relationship Learning via Alternating Direction Method of Multipliers (https://arxiv.org/abs/2411.06135)
Comments:
          Accpeted by Neurocomputing

- **What's New**: 본 연구는 온라인 다중 작업 학습(Online Multi-task Learning, OMTL) 문제에 대해 중앙 서버를 활용하는 전통적인 방식 대신, 분산 컴퓨팅 환경에 적합하고 효율적인 수행을 위한 대안으로 교차 방향 배수기법(Alternating Direction Multiplier Method, ADMM)을 기반으로 한 새로운 OMTL 프레임워크를 제안합니다.

- **Technical Details**: 제안된 OMTL 알고리즘은 ADMM 최적기를 사용하여 여러 작업 간의 동적 관계를 모델링하며, 단순화된 하위 문제들을 병렬로 처리할 수 있는 구조를 가지고 있습니다. 이는 각 노드가 지역 이웃과 정보만을 교환하면서 작업을 수행할 수 있도록 하여, 중앙 서버의 병목 현상을 피하고 효율성을 증대시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 ADMM 기반 OMTL 알고리즘은 기존의 SGD 기반 접근법보다 정확성과 효율성에서 우수한 성능을 보였습니다. 실제 데이터셋과 합성 데이터셋을 통해 검증된 결과는 대규모 학습 문제에 대한 적용 가능성을 시사합니다.



### Mutual-energy inner product optimization method for constructing feature coordinates and image classification in Machine Learning (https://arxiv.org/abs/2411.06100)
Comments:
          23 pages,5 figures

- **What's New**: 본 논문에서는 데이터 분류를 위한 새로운 접근법으로 mutual-energy inner product 최적화 방법을 제안합니다. 이는 다양한 클래스의 샘플 데이터를 표현할 적합한 좌표계를 구성하는 것을 목표로 합니다.

- **Technical Details**: 논문은 비균일 멤브레인을 설명하는 편미분 방정식의 해 공간과 고유함수를 분석하여 mutual-energy inner product를 정의합니다. 이 inner product는 고유 함수의 시리즈로 표현되며, 유클리드 inner product와 비교했을 때 저주파 특성을 강화하고 고주파 노이즈를 억제하는 데 있어 유리하다는 점이 강조됩니다. 최적화 모델은 유한 요소 방법과 결합하여 안정적이고 효율적인 순차 선형화 알고리즘을 개발합니다.

- **Performance Highlights**: 개발된 알고리즘은 양의 정부호 대칭 행렬과 몇 개의 제약 조건을 포함하는 선형 프로그래밍 방정식만을 해결합니다. 이후, mutual-energy inner product 최적화 방법을 사용하여 특성 좌표계를 구성하고 MNIST(MNIST) 훈련 세트에서 다중 클래스 Gaussian 분류기를 학습한 결과, MNIST 테스트 세트에서 좋은 예측 성능을 달성했습니다.



### Concept Bottleneck Language Models For protein design (https://arxiv.org/abs/2411.06090)
- **What's New**: 이 논문은 Concept Bottleneck Protein Language Models (CB-pLM)을 소개합니다. CB-pLM은 각 뉴런이 해석 가능한 개념에 대응되는 생성형 마스킹 언어 모델로, 생성 단백질의 성질을 정확하게 제어할 수 있는 능력을 제공합니다.

- **Technical Details**: CB-pLM은 24백만에서 30억 파라미터까지 확장되어 현재까지 훈련된 가장 큰 개념 병목 모델입니다. 모델은 generative masked language modeling loss, concept loss, orthogonality loss라는 세 가지 손실 함수를 사용하여 훈련됩니다.

- **Performance Highlights**: 훈련된 CB-pLM은 전통적인 마스킹 단백질 언어 모델에 비슷한 사전 훈련의 교란(perplexity) 정도와 하위 작업 성능을 나타내며, 해석 가능성이 성능 손실을 초래하지 않음을 보여줍니다.



### Optimizing Large Language Models through Quantization: A Comparative Analysis of PTQ and QAT Techniques (https://arxiv.org/abs/2411.06084)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문은 Large Language Models (LLMs)의 최적화를 위한 양자화(Quantization) 기법들에 대한 포괄적인 분석을 제공합니다. 특히 Post-Training Quantization (PTQ)와 Quantization-Aware Training (QAT)에 중점을 두었습니다.

- **Technical Details**: 실험적으로 10M에서 1B 파라미터 범위의 모델에 대해 양자화를 적용한 결과, 모델 크기를 최대 68% 줄이면서도 전체 정밀도 기준선과의 성능 차이를 6% 이내로 유지할 수 있다는 것을 보여주었습니다. INT8 양자화는 연산 비용과 전력 소비를 40% 줄였고, INT4 양자화는 이러한 수치를 60% 더 개선했습니다. 혼합 정밀도 양자화를 위한 새로운 이론적 프레임워크를 도입하고, 레이어 민감도와 가중치 분산에 기반한 최적 비트 할당 전략을 도출했습니다.

- **Performance Highlights**: 엣지 장치에서의 하드웨어 효율성 평가 결과, INT8은 최대 2.4배의 처리량 향상을, INT4는 3배의 향상을 가능하게 하였으며, 전체 정밀도 모델에 비해 60% 전력 소비 감소를 보여주었습니다.



### A Survey on Kolmogorov-Arnold Network (https://arxiv.org/abs/2411.06078)
- **What's New**: 이번 논문은 Kolmogorov-Arnold Networks (KAN)에 대한 체계적 검토이다. KAN은 Kolmogorov-Arnold 표현 정리에 영감을 받은 신경망 모델로, 고차원 함수의 유연하고 해석 가능한 표현을 위한 학습 가능한 스플라인 파라미터화 함수 사용이 특징이다.

- **Technical Details**: KAN의 아키텍처는 적응형 엣지 기반 활성화 함수가 포함되어 있어, 시간 시계열 예측, 계산 생물 의학 및 그래프 학습과 같은 응용 분야에서 매개변수 효율성과 확장성을 향상시킨다. Temporal-KAN, FastKAN, Partial Differential Equation (PDE) KAN과 같은 주요 발전은 KAN의 동적 환경에서의 적용 가능성을 보여준다.

- **Performance Highlights**: KAN은 복잡한 함수 근사 작업을 위한 해석 가능성, 계산 효율성 및 적응력을 향상시키며, 합성곱(convolutional), 순환(recurrent), 변환(transformer) 기반 모델과의 통합을 통해 혼합 접근 방식이 필요한 작업에 대한 다재다능성을 보여준다. 그러나, KAN은 고차원 및 노이즈 데이터 설정에서의 계산적 도전 과제가 남아 있어 최적화 전략, 정규화 기법 및 하이브리드 모델에 대한 연구가 필요하다.



### GFT: Graph Foundation Model with Transferable Tree Vocabulary (https://arxiv.org/abs/2411.06070)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 Graph Foundation Models (GFMs)의 개발을 위한 새로운 접근법을 제안합니다. 기존의 그래프 모델들이 다양한 그래프 학습 작업에서 원하는 성능을 달성하지 못하는 상황에서, 연구자들은 계산 트리(computation tree)를 새로운 전이 가능한 패턴으로 제안하여 GFMs를 개선하고자 했습니다.

- **Technical Details**: GFT (Graph Foundation model with transferable Tree vocabulary)라는 이름의 새로운 GFMs 모델을 제안합니다. 이 모델은 메시지 전달 과정에서 파생된 계산 트리 구조를 사용하여 그래프 데이터를 처리하는데, 이를 통해 모델의 일반화 능력을 향상시키고 부정 전이(negative transfer)의 위험을 줄입니다. GFT는 데이터셋의 다양한 작업과 도메인을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, GFT가 다양한 작업 및 도메인에 걸쳐 그래프 학습의 효과성을 입증하였으며, 계산 트리의 유사성과 전이 학습 성능 간의 강한 상관관계를 보여주었습니다. 이 연구는 GFMs의 향후 개발에 중요한 길잡이가 될 것으로 기대됩니다.



### Model Selection for Average Reward RL with Application to Utility Maximization in Repeated Games (https://arxiv.org/abs/2411.06069)
- **What's New**: 이 논문에서는 평균 보상 기반 강화 학습(average reward RL) 설정에서 사용할 수 있는 온라인 모델 선택 알고리즘인 MRBEAR를 제안합니다. 이 알고리즘은 서로 다른 복잡성을 가진 모델 클래스 중에서 최적의 정책을 학습하는 데 중점을 두고 있습니다.

- **Technical Details**: MRBEAR의 후회(regret)는 $	ilde O(M C_{m^*}^2 	extsf{B}_{m^*}(T,ho))$로, 여기서 $C_{m^*}$는 가장 간단한 명시적 모델 클래스의 복잡성을 나타내며, $	extsf{B}_{m^*}(T,ho)$는 해당 후회 경계입니다. 알고리즘은 두 플레이어의 상호작용에서 학습자를 모델링하며, 상대방은 고정된 미지의 제한된 기억 전략을 따릅니다.

- **Performance Highlights**: MRBEAR은 학습자의 성과를 평균 보상 후회로 측정하며, 상대방의 메모리 한계에 따라 $	ilde O(M(	extsf{sp}(h^*) B^{m^*} A^{m^*+1})^{rac{3}{2}} 	extsf{√{T}})$의 후회를 생성합니다. 이 알고리즘은 상대방의 효용 함수나 메모리 한계를 사전에 알지 못하는 상태에서도 최적의 정책을 학습할 수 있도록 설계되었습니다.



### Learning Mixtures of Experts with EM (https://arxiv.org/abs/2411.06056)
- **What's New**: 이번 연구에서는 Mixtures of Experts (MoE) 모델의 효율성을 개선하기 위한 Expectation Maximization (EM) 알고리즘의 성능을 집중적으로 살펴보았습니다. EM이 기존의 gradient descent 방법의 대안으로 어떻게 기능하는지를 분석했습니다.

- **Technical Details**: MoE는 입력 공간을 분할하여 각각의 파티션에 대해 별도의 '전문가' 모델을 훈련시키는 기계 학습 모델입니다. 본 연구에서는 선형 또는 로지스틱 전문가의 경우에서 EM 알고리즘을 엄격하게 분석하며, EM이 Mirror Descent와 unit step size 및 Kullback-Leibler Divergence 정규화자로 동등하다는 것을 보여줍니다. 이 관점은 새로운 수렴 결과를 도출하고 신호 대 잡음 비율(SNR)에 기반한 지역 선형 수렴 조건을 식별하는데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 합성 데이터(synthetic data) 및 소규모 실제 데이터(real-world data)에 대해 EM이 gradient descent 알고리즘보다 수렴 속도와 정확도에서 우수하다는 것을 보여주었습니다.



### Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data (https://arxiv.org/abs/2411.06055)
- **What's New**: 이 논문에서는 구형 확률 분포를 효과적으로 비교하기 위해 새로운 방법론인 Linear Spherical Sliced Optimal Transport (LSSOT)를 제안합니다. LSSOT는 구형 분포를 L^2 공간에 임베딩하면서 본래의 기하학적 특성을 유지하도록 설계되었습니다.

- **Technical Details**: LSSOT는 기하학적 구조를 유지하기 위해 구형 분포를 잘라서 이차원 공간에 투영합니다. 이는 최적 운송(optimal transport) 문제의 계산 비용을 줄이는 방안을 제공합니다. 이 방법은 주변 시각화(cortical surface registration), 3D 포인트 클라우드 보간(point cloud interpolation), 형태 임베딩(shape embedding)과 같은 여러 응용 분야에서 그 효율성을 검증했습니다.

- **Performance Highlights**: LSSOT는 다른 기초 지표(metric)보다 우수한 계산 효율성을 보이며, 다양한 실험을 통해 효과성과 효율성을 입증했습니다. 특히 포인트 클라우드 분석과 뇌 피질 표면 등록(cortical surface registration)에서 큰 장점을 보여주었습니다.



### Personalized Hierarchical Split Federated Learning in Wireless Networks (https://arxiv.org/abs/2411.06042)
- **What's New**: 본 논문에서는 개인화된 계층적 분할 연합 학습(Personalized Hierarchical Split Federated Learning, PHSFL) 알고리즘을 제안하여 리소스가 제한된 무선 네트워크에서의 대규모 기계 학습의 개인화 성능을 개선하고자 합니다. PHSFL은 ML 모델을 클라이언트와 서버 측 블록으로 나누며, 클라이언트는 자신의 개인 작업에 맞는 맞춤형 모델을 학습할 수 있습니다.

- **Technical Details**: PHSFL은 클라이언트의 데이터 분포에 관계없이 유사한 특성이 있는 피처들을 활용하여 전체 ML 모델 중 몸체 부분만을 학습하고, 분류자는 훈련 중 고정된 상태를 유지합니다. 이를 통해 데이터 불균형이 존재하는 상황에서도 개인화된 성능을 향상할 수 있습니다. 또한 최적의 이론적 경계를 제공합니다.

- **Performance Highlights**: PHSFL 모델은 일반화 성능이 HSFL과 유사하지만, 세밀하게 조정된 개인화 모델이 보다 높은 개인화 성능을 보이는 것으로 나타났습니다. 이러한 결과는 글로벌 모델에 비해 확인된 성능 개선을 뒷받침하며, 다양한 클라이언트의 테스팅에 대해 높은 일관성을 유지하는 것을 보여줍니다.



### CGLearn: Consistent Gradient-Based Learning for Out-of-Distribution Generalization (https://arxiv.org/abs/2411.06040)
Comments:
          9 pages, 3 figures

- **What's New**: 이번 논문에서는 여러 환경에서의 gradient 합의를 활용하여, 기계 학습 모델의 일반화 및 강건성을 향상시키기 위한 새로운 방법론인 CGLearn을 제안합니다. CGLearn은 다양한 환경에서의 일관된 특징을 통해 신뢰할 수 있는 예측 변수를 학습하는 데 중점을 둡니다.

- **Technical Details**: CGLearn은 gradient 일관성을 강제하여, 다양한 환경에서의 각 변수의 요소에 대한 invariant features를 활용합니다. 이 방법론은 Empirical Risk Minimization (ERM) 접근법을 기초로 하여, 각 특성의 gradient 일관성을 보장하여 신뢰할 수 있는 특성을 식별합니다. 이는 주로 선형 회귀 및 분류 작업에서 적용됩니다.

- **Performance Highlights**: CGLearn은 선형 및 비선형 설정에서 기존의 최첨단 방법들에 비해 우수한 예측력과 일반화 능력을 입증하였으며, 별도의 환경이 없더라도 그 성능을 발휘합니다. 다양한 합성 및 실제 데이터셋에 대한 실험 결과는 본 방법의 효과성과 강건성을 강조합니다.



### Parallel Multi-path Feed Forward Neural Networks (PMFFNN) for Long Columnar Datasets: A Novel Approach to Complexity Reduction (https://arxiv.org/abs/2411.06020)
- **What's New**: 이 논문은 Parallel Multi-path Feed Forward Neural Networks (PMFFNN)라는 새로운 아키텍처를 도입하여, 긴 열(columnar) 데이터셋을 처리하는 데 있어 기존 Feed-Forward Neural Networks (FFNN) 및 1D CNN의 한계를 극복하고자 합니다.

- **Technical Details**: PMFFNN은 입력 데이터를 여러 개의 열(column) 하위 집합으로 나누고, 각 하위 집합을 독립적인 병렬 경로를 통해 처리합니다. 각 경로는 'micro-FFNN'으로 작동하며, 특성(Feature) 학습에 전문화된 모듈입니다. 이 구조는 BatchNormalization 및 Dense 레이어를 포함하여 효율적인 학습을 보장하고 과적합(overfitting) 문제를 최소화합니다.

- **Performance Highlights**: 실험 결과, PMFFNN은 전통적인 FFNN 및 1D CNN보다 뛰어난 성능을 보이며, 대규모 데이터 관리에 최적화된 솔루션을 제공합니다. 특징의 다양성을 극대화하고, 처리 성능을 개선하며, 훈련 시간과 자원 효율성을 높이는 데 기여합니다.



### A Picture is Worth A Thousand Numbers: Enabling LLMs Reason about Time Series via Visualization (https://arxiv.org/abs/2411.06018)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)의 시간 시계열 추론(TsR) 능력을 평가하기 위한 새로운 테스트베드인 TimerBed를 제안합니다. TimerBed는 실제 작업에 관한 계층화된 추론 패턴, LLMs 및 추론 전략의 종합적인 조합을 포함하고 있습니다.

- **Technical Details**: TimerBed는 간단한 결정론적 추론, 복잡한 결정론적 추론 및 확률적 추론의 세 가지 패턴으로 구성되어 있으며, 각 패턴에 대해 실제 작업을 정렬합니다. 또한 VL-Time이라는 프롬프트 기반 솔루션을 제안하여 데이터 시각화 및 언어 가이드를 통해 LLMs의 시간 시계열 추론 능력을 강화합니다.

- **Performance Highlights**: VL-Time은 시계열에 대해 비트리비얼 제로샷과 강력한 몇 샷 추론 기술을 가능하게 하여 평균 140%의 성능 향상과 99%의 토큰 비용 절감을 이끌어냈습니다. 이는 단순한 수치 모델링에 비해 매우 뛰어난 결과입니다.



### Longitudinal Ensemble Integration for sequential classification with multimodal data (https://arxiv.org/abs/2411.05983)
Comments:
          11 pages, submitted to ICLR 2025

- **What's New**: 이번 연구에서는 다중 모드 및 종단적 데이터를 효과적으로 모델링하기 위해 새로운 Longitudinal Ensemble Integration (LEI) 학습 프레임워크를 개발했습니다. 이는 치매의 조기 탐지를 위한 다중 모드 시퀀스 분류에서 성능이 뛰어난 것으로 평가되었습니다.

- **Technical Details**: LEI는 다양한 데이터 모드로부터 개별 기본 예측기를 생성하고, 이러한 예측기를 Long Short-Term Memory (LSTM) 네트워크를 사용해 스택하는 방식을 통해 성능을 극대화합니다. 데이터는 Alzheimer’s Disease Prediction of Longitudinal Evolution (TADPOLE) Challenge에서 수집된 정보를 기반으로 합니다.

- **Performance Highlights**: LEI는 기존의 접근 방식보다 향상된 성과를 보였으며, 다양한 모드에서 중요한 피쳐들을 식별할 수 있었습니다. 이 프레임워크는 종단적 다중 모드 데이터에서의 연속적 분류에 대한 잠재력을 확장하여 다양한 응용 분야에 활용될 수 있습니다.



### Variance-Aware Linear UCB with Deep Representation for Neural Contextual Bandits (https://arxiv.org/abs/2411.05979)
- **What's New**: 본 연구에서는 심층 신경망(deep neural networks)의 표현 능력을 활용하여 neural upper confidence bound (UCB) 알고리즘의 발전을 이루었습니다. Neural-$\sigma^2$-LinearUCB 알고리즘은 보상 노이즈 분산(reward noise variance)의 상한인 $\sigma^2_t$를 사용하여 탐색(exploration)과 활용(exploitation)의 균형을 맞추고, 불확실성 정량화(uncertainty quantification)의 품질을 향상시킵니다.

- **Technical Details**: Neural-$\sigma^2$-LinearUCB는 두 가지 버전을 제공합니다: 오라클 버전(oracle version)과 실용 버전(practical version). 오라클 알고리즘은 오라클 분산 상한(oracle variance upper bound) $\sigma^2_t$로 특징 지어지며, 실용 버전은 새로운 분산 경계 추정치를 포함한 것입니다. 이론적으로 두 버전에 대한 엄밀한 후회 분석(regret analysis)을 제공하며, 오라클 알고리즘이 다른 neural-UCB 알고리즘보다 더 나은 후회 보장을 달성함을 입증합니다.

- **Performance Highlights**: 실용적인 방법은 비슷한 계산 효율성(computational efficiency)을 가지면서, 합성(synthetic), UCI, MNIST, CIFAR-10 데이터셋을 포함한 여러 표준 설정에서 상태-of-the-art 기술을 초월하며 더 나은 보정(calibration)과 낮은 후회(regret)를 기록했습니다.



### The effect of different feature selection methods on models created with XGBoos (https://arxiv.org/abs/2411.05937)
- **What's New**: 이번 연구는 XGBoost와 같은 인기 있는 머신러닝 알고리즘에서 다양한 피처 선택 방법이 모델 구축에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 세 가지 다른 차원 축소 방법이 모델의 예측 정확도에 통계적으로 유의미한 변화를 일으키지 않는다는 것을 보여줍니다. 이는 XGBoost에서 잡음이 많은 훈련 데이터를 제거하여 모델의 과적합(overfitting)을 방지하는 전통적인 아이디어가 적용되지 않을 수 있음을 시사합니다.

- **Performance Highlights**: 그럼에도 불구하고, 이러한 피처 선택 기법들은 계산 복잡도(computational complexity)를 줄이는 데 여전히 유효할 수 있습니다.



### DNAMite: Interpretable Calibrated Survival Analysis with Discretized Additive Models (https://arxiv.org/abs/2411.05923)
- **What's New**: 이 논문에서는 유리 상자(glass-box) 머신러닝 모델인 DNAMite를 소개합니다. DNAMite는 생존 분석(survival analysis)을 위해 고안된 모델로, 예측 성능과 해석 가능성을 모두 갖추고 있습니다.

- **Technical Details**: DNAMite는 특징 이산화(feature discretization)와 커널 스무딩(kernel smoothing)을 사용하여 형태 함수를 학습합니다. 이 모델은 부드러움(smoothness)과 울퉁불퉁함(jaggedness) 간의 유연한 균형을 유지하여 실제 데이터에서 자주 발견되는 복잡한 패턴을 포착할 수 있습니다. DNAMite는 각 특징의 기여도를 누적 발생 함수(cumulative incidence function)에 직접 해석 가능한 형상 함수(shape function)로 변환합니다.

- **Performance Highlights**: 실험 결과, DNAMite는 합성 데이터(synthetic data)에서 실제 형상 함수와 더 가까운 형상 함수를 생성하며, 예측 성능은 이전의 유리 상자(glass-box) 및 블랙 박스(black-box) 모델과 비교하여 보다 나은 보정(calibration)을 보여줍니다.



### Towards Multi-Modal Mastery: A 4.5B Parameter Truly Multi-Modal Small Language Mod (https://arxiv.org/abs/2411.05903)
- **What's New**: 새로운 4.5B 파라미터의 소형 언어 모델이 소개되었습니다. 이 모델은 텍스트, 이미지, 비디오, 오디오 등 다양한 입력 및 출력 모달리티(modality)를 처리할 수 있습니다.

- **Technical Details**: 이 모델은 언어 모델링(language modeling)과 다중 작업 학습(multi-task learning)의 최근 발전을 활용하여 만들어졌으며, 간편하게 엣지 추론(edge inference)에 배포될 수 있습니다.

- **Performance Highlights**: 모델은 여러 벤치마크에서 뛰어난 성능을 보여주며, 복잡한 현실 세계 문제를 해결할 수 있는 다중 모달(multi-modal) 인공지능의 가능성을 시사합니다.



### Streaming Bayes GFlowNets (https://arxiv.org/abs/2411.05899)
Comments:
          25 pages, 8 figures

- **What's New**: 본 논문에서는 이산 매개변수 공간에서의 스트리밍 Bayesian 추론을 가능하게 하기 위해 Streaming Bayes GFlowNets (SB-GFlowNets)를 제안하고 있습니다. 이는 최근에 제안된 GFlowNets를 활용하여 비정규화된 포스터리어에서 샘플링하는 데 효과적임을 보여줍니다.

- **Technical Details**: SB-GFlowNet은 초기 포스터리어(Posterior)를 표준 GFlowNet을 사용하여 근사한 후, 이후에 관측된 새로운 데이터만을 이용한 특별한 절차로 업데이트합니다. 이 방법은 변별적 상태 공간(discrete state space)에서도 안정적이고 효율적인 Bayesian 추론을 가능하게 합니다.

- **Performance Highlights**: SB-GFlowNets의 사례 연구인 선형 선호 학습(linear preference learning) 및 계통 발생 추론(phylogenetic inference)에서 Unnormalized Posterior로부터 샘플링하는 데 있어 효과성이 입증되었으며, GFlowNet을 처음부터 다시 훈련하는 것에 비해 훨씬 빠른 속도를 보였습니다.



### When are 1.58 bits enough? A Bottom-up Exploration of BitNet Quantization (https://arxiv.org/abs/2411.05882)
Comments:
          10 pages, 2 tables, 6 figures

- **What's New**: 본 논문은 1.58-bit 양자화(quantization) 기반의 훈련 방법을 조사하며, 비변환기(non-transformer) 모델 아키텍처와 변환기(transformer) 기반 모델을 포함한 다양한 네트워크에서의 성능을 비교 분석합니다.

- **Technical Details**: 1.58-bit 양자화 방법론은 Linear layers를 대체하는 BitLinear 레이어를 도입하여, 훈련 중 16-bit 그림자 가중치(shadow weights)를 유지하고, 순전파(forward pass) 시에는 이 가중치들을 양자화하여 사용합니다. 이는 최종적으로는 ternary weights(세 가지 값 -1, 0, 1)으로 전환됩니다.

- **Performance Highlights**: 모든 실험에서 1.58-bit 훈련 방법이 표준 32/16-bit 모델과 동등하거나 더 나은 성능을 보여주었으며, 특히 encoder-only 및 encoder-decoder 모델에서의 성능 향상이 두드러졌습니다. 특정 네트워크에서는 16-bit 모델을 초월하는 성능을 발휘했습니다.



### Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass (https://arxiv.org/abs/2411.05877)
- **What's New**: 이번 연구는 새로운 컨텍스트에 대한 효과적이고 효율적인 적응 방법인 GenerativeAdapter를 소개합니다. 이 방법은 사전 학습된 언어 모델을 미세 조정(fine-tuning) 없이 저랭크(low-rank) LM 어댑터에 직접 매핑하여 추론 오버헤드를 크게 감소시킵니다.

- **Technical Details**: GenerativeAdapter는 자가 지도 학습(self-supervised learning)을 통해 훈련된 어댑터 생성기(adapter generator)를 사용합니다. 이 어댑터 생성기는 고정된 LM을 단일 온-더 플라이(on-the-fly)로 적응시키며, 그러한 컨텍스트를 새로운 어댑터에 매핑합니다. 본 연구에서는 Mistral-7B-Instruct와 Llama2-7B-Chat이라는 두 개의 사전 학습된 LM에 GenerativeAdapter를 적용했습니다.

- **Performance Highlights**: StreamingQA에서 우리의 접근 방식은 LM의 파라미터에 지식을 주입하는 데 효과적이며, 감독 미세 조정(supervised fine-tuning)이 있는 모델 대비 F1 점수에서 63.5%의 향상(19.5에서 31.5로)이라는 성과를 달성했습니다. 다양한 적응 시나리오에서 평균 정확도 44.9%를 기록하여 기본 모델을 초월했습니다.



### Towards Improved Preference Optimization Pipeline: from Data Generation to Budget-Controlled Regularization (https://arxiv.org/abs/2411.05875)
Comments:
          15 pages

- **What's New**: 최근 Direct Preference Optimization (DPO) 및 그 변형들이 큰 언어 모델(LLM)의 조정에 있어 주요 방법이 되고 있습니다. 이 연구에서는 DPO의 선호 데이터 생성 및 교육 정규화 기술을 개선하는 방법을 제안합니다.

- **Technical Details**: 우리는 반복 쌍별 순위 메커니즘을 도입하여 선호 데이터의 품질을 향상시키고, 예측된 선호 샘플의 우선 확률을 약간 감소시키는 새로운 예산 관리 정규화 기법을 사용합니다. 이는 LLM의 예측 정확성을 유지하면서도 최적화 과정의 안정성을 가져옵니다.

- **Performance Highlights**: 이 연구에서 제안한 방법들은 두 가지 일반 벤치마크 평가를 통해 기존 SOTA를 능가하는 결과를 보였으며, 현업에서의 LLM 적용에 있어 높은 품질의 선호 데이터 생성을 통해 더 나은 최적화를 이루었습니다.



### Interplay between Federated Learning and Explainable Artificial Intelligence: a Scoping Review (https://arxiv.org/abs/2411.05874)
Comments:
          16 pages, 11 figures, submitted in IEEE Trans. Knowledge and Data Engineering

- **What's New**: 이번 논문은 Federated Learning (FL)과 Explainable Artificial Intelligence (XAI)의 결합에 대한 최신 연구 결과를 정리하고 있습니다. 37개의 연구가 FL과 XAI의 상호 작용을 분석하였으며, 특히 데이터가 분산되어 있는 환경에서의 모델의 해석 가능성과 설명 방식에 대해 논의하고 있습니다.

- **Technical Details**: FL은 머신러닝 모델을 중앙 집중식 데이터 전송 없이 분산된 데이터로부터 훈련시키는 기술입니다. 이와 함께 XAI는 AI 시스템의 결정 과정을 설명 가능하게 만드는 알고리즘 및 방법론을 개발하는 분야입니다. HFL(수평적 연합 학습), VFL(수직적 연합 학습), FTL(연합 전이 학습)과 같은 FL의 세 가지 주요 카테고리가 존재하며, 모델의 해석 가능성 및 post-hoc explanation(사후 설명 방법)을 통한 다양한 접근이 필요합니다.

- **Performance Highlights**: FL과 XAI의 결합 연구는 증가하고 있으나, 연구의 대다수는 데이터 센터 수가 10개 이하인 시뮬레이션 환경을 사용했습니다. 8개의 논문은 FL 알고리즘의 구성 요소로서 설명 방법을 통합하여 얻는 이점을 다루었고, FL과 XAI의 상호 작용에 대한 보다 정량적이고 구조화된 연구가 필요함을 강조하였습니다.



### Poor Man's Training on MCUs: A Memory-Efficient Quantized Back-Propagation-Free Approach (https://arxiv.org/abs/2411.05873)
- **What's New**: 이 논문은 마이크로컨트롤러(MCU)에서 Back Propagation(BP) 없이 훈련할 수 있는 간단한 방법을 제시하며, 엣지 훈련 하드웨어 설계를 추론 하드웨어 설계처럼 쉽게 만듭니다.

- **Technical Details**: 양자화된 제로차 방법(quantized zeroth-order method)을 활용하여 양자화된 모델 파라미터의 기울기를 추정하며, BP 기반 훈련의 오류를 극복합니다. 차원 축소(dimension reduction) 방법(노드 퍼터베이션(node perturbation), 희소 훈련(sparse training))을 활용하여 제로차 훈련의 수렴성을 개선합니다.

- **Performance Highlights**: 우리의 BP-free 훈련 방법은 자원 제약이 있는 엣지 장치에서 미리 훈련된 이미지 분류기를 다양한 손상된 데이터에 적응시킬 때 BP 기반 훈련과 유사한 성능을 달성하며, 평균 6.4%의 테스트 정확도 향상을 보여줍니다.



### Enhancing Financial Fraud Detection with Human-in-the-Loop Feedback and Feedback Propagation (https://arxiv.org/abs/2411.05859)
Comments:
          International Conference on Machine Learning and Applications 2024

- **What's New**: 이번 논문은 Human-in-the-Loop (HITL) 피드백 메커니즘이 재정 사기 탐지에서 기계 학습(ML) 모델의 성능을 어떻게 향상시키는지를 다룹니다. 특히 전문가의 소량의 피드백으로도 모델의 정확도가 크게 증가할 수 있음을 보여주고, 그래프 기반 기술이 가장 큰 혜택을 보는 것으로 나타났습니다.

- **Technical Details**: HITL 시스템은 인간의 전문성과 기계 학습 기술을 결합하여 데이터를 주석 처리하고 모델 학습을 보강하는 접근 방식입니다. 새로운 피드백 전파 방법을 소개하며, 이는 데이터셋 전반에 피드백을 확장하여 탐지 정확도를 더욱 향상시킵니다. 그리고 기계 학습 모델을 훈련시키고 평가하는 전통적인 방법과 첨단 기술을 통합하는 HITL 프레임워크를 제안합니다.

- **Performance Highlights**: HITL 피드백을 통합함으로써, 데이터 주석 품질, 모델 해석 가능성 및 동적 사기 패턴 적응성을 개선할 수 있었습니다. 연구 결과는 HITL 메커니즘이 기존 및 최신의 금융 사기 탐지 기법을 향상시킬 수 있는 잠재력을 보여주며, 대다수의 알고리즘들에서 사기 탐지 성능을 크게 향상시켰습니다.



### Financial Fraud Detection using Jump-Attentive Graph Neural Networks (https://arxiv.org/abs/2411.05857)
Comments:
          International Conference on Machine Learning and Applications 2024

- **What's New**: 본 연구는 금융 사기 탐지에서의 그래프 신경망(Graphic Neural Networks, GNN)을 활용하여 기존 모델들의 한계를 극복할 수 있는 새로운 접근 방식을 제안합니다. 특히, 주목 메커니즘(attention mechanism)을 통합하여 비슷하지 않은 노드에서의 중요한 특성 정보를 보존하는 혁신적인 GNN 구조(Jump-Attentive Graph Neural Network, JA-GNN)를 소개합니다.

- **Technical Details**: 본 논문에서 제안한 JA-GNN은 효율적인 이웃 샘플링(neighborhood sampling) 방법을 활용하여 비슷한 노드에서 특징 정보를 보존하며 위장하는 사기 행위(camouflaging fraud) 탐지에 효과적으로 대응합니다. 이 방법은 잔여 연결(residual connections)을 포함하여 고객의 최종 임베딩을 최적화하여 합법적인 거래와 사기 거래를 구분하는 데 필수적입니다.

- **Performance Highlights**: 우리의 실험 결과 JA-GNN은 금융 데이터에 대한 실험에서 다른 최신 그래프 알고리즘에 비해 더 나은 성능을 나타내어, 혁신적인 샘플링 전략과 GNN 구조가 금융 사기 탐지에서의 효율성을 어떻게 개선하는지를 보여주었습니다.



### Learning Morphisms with Gauss-Newton Approximation for Growing Networks (https://arxiv.org/abs/2411.05855)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 Neural Architecture Search (NAS) 방법에서 network morphism을 활용하여 네트워크를 효과적으로 확장하는 새로운 접근 방식을 제안합니다. 기존 NAS 방법들이 다양한 네트워크를 훈련시켜야 하는 것과 달리, 이 방법은 작은 seed 네트워크에서 시작하여 능동적으로 새로운 뉴런을 추가합니다.

- **Technical Details**: 제안된 방법은 Gauss-Newton 근사를 Loss function에 적용하여 candidate network morphisms를 효율적으로 학습하고 평가합니다. 각 morphism을 적용했을 때 예상되는 손실 감소를 추정하고, 이 근사된 손실 함수를 역전파(backpropagation)를 사용해 최적화하여 morphism 매개변수를 학습합니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 분류 작업에 대해 실험한 결과, 제안된 NAS 방법이 유사한 또는 더 나은 품질의 아키텍처를 더 적은 계산 비용으로 학습하고, 최신 성능을 달성함을 나타냅니다.



### $\spadesuit$ SPADE $\spadesuit$ Split Peak Attention DEcomposition (https://arxiv.org/abs/2411.05852)
- **What's New**: 새로운 신경망 예측 모델인 Split Peak Attention DEcomposition (SPADE)을 도입하여 Peak Events (PEs)이후의 수요 예측을 개선합니다.

- **Technical Details**: SPADE 모델은 예측을 두 개의 별개 작업으로 모듈화합니다: PEs 및 비PEs. 이 모델은 마스킹된 합성곱 필터(masked convolution filters)와 Peak Attention 모듈을 사용하여 PEs의 영향을 줄입니다.

- **Performance Highlights**: 전 세계 소매 데이터셋에서 SPADE는 PPE 오류를 4.5% 줄이고, PE 정확도를 3.9% 개선시키며 기존 모델과 비교해 우수한 성능을 보였습니다.



### Multivariate Data Augmentation for Predictive Maintenance using Diffusion (https://arxiv.org/abs/2411.05848)
- **What's New**: 이 논문은 predictive maintenance(예측 유지 관리) 분야에서 diffusion 모델을 활용하여 결함 데이터를 생성하는 새로운 접근 방식을 제안합니다. 노후화된 시스템의 결함 데이터를 사용하는 대신, 유사한 시스템에서 학습한 관계를 바탕으로 새로운 시스템의 가상의 결함 데이터를 생성합니다.

- **Technical Details**: Diffusion 모델은 점진적으로 데이터 샘플에 Gaussian noise를 추가하고, 이를 제거하는 과정을 통해 원래의 데이터를 복원합니다. 이 과정을 통해 생성된 synthetic 데이터는 predictive 모델 훈련에 도움이 됩니다. DSAT-ECG라는 최신 모델을 사용하여 PRONOSTIA 데이터셋에서 테스트를 진행하였습니다.

- **Performance Highlights**: 연구 결과, 생성된 synthetic 데이터가 predictive maintenance 모델의 효율성을 향상시키는 데 성공적인 역할을 했습니다. 또한, 생성된 데이터는 concept drift 문제를 해결하는 데 기여할 수 있는 가능성이 확인되었습니다.



### Reducing catastrophic forgetting of incremental learning in the absence of rehearsal memory with task-specific token (https://arxiv.org/abs/2411.05846)
- **What's New**: 이 논문에서는 새로운 데이터 학습 시 기존 데이터를 저장하지 않고도 이전 지식을 보존할 수 있는 혁신적인 방법을 제안합니다.

- **Technical Details**: 이 방법은 비전 변환기(vision transformer)의 구조에서 영감을 받았으며, 각 작업의 압축된 지식을 캡슐화할 수 있는 독특한 토큰(token)을 사용합니다. 작업에 따라 주의를 다르게 기울여 작업별 임베딩(task-specific embeddings)을 생성합니다.

- **Performance Highlights**: 우리의 모델은 여러 작업 증분 학습(task-incremental learning) 시나리오에서 벤치마크 데이터 세트를 사용하여 정확도와 역전이(backward transfer) 측면에서 성능을 측정하였으며, 비교한 방법들 중에서 가장 높은 정확도와 가장 낮은 역전이 성능을 기록했습니다.



### Neural Precision Polarization: Simplifying Neural Network Inference with Dual-Level Precision (https://arxiv.org/abs/2411.05845)
- **What's New**: 이번 논문에서는 DNN 추론을 위한 정밀도 편극(precision polarization) 방법을 소개합니다. 이 방법은 네트워크의 대부분 가중치와 활성화를 저정밀도(low precision)로 설정하고, 오류 보정을 위해 특정 경로에는 고정밀도(high precision)를 할당합니다. 이를 통해 메모리와 계산 요구 사항을 줄이면서도 모델의 정확도를 유지할 수 있습니다.

- **Technical Details**: 정밀도 편극 방법은 매우 낮은 정밀도(NF4, INT8)로 대다수의 네트워크 가중치를 처리하며, 정확도 손실을 보완하기 위해 고정밀 대체 경로를 사용합니다. 이러한 대체 경로는 낮은 순위 근사(low-rank approximation)를 최적화하여 정확도를 회복하며, 주로 적은 양의 훈련 데이터에 대한 감도 기반 메트릭을 활용하여 훈련됩니다. 이 과정은 클라우드에서 교육된 부동 소수점 모델이 에지 디바이스로 다운로드된 후 직접 양자화된다는 점에서 독특합니다.

- **Performance Highlights**: 시뮬레이션 결과, 신경망 정밀도 편극은 약 464 TOPS/W의 매트릭스 연산(MAC) 효율성과 신뢰성을 달성했습니다. 이는 저에너지 쉐임 및 고효율의 비트 평면 처리(bit plane-wise processing)를 통합하여 이루어진 성과입니다.



### FLEXtime: Filterbank learning for explaining time series (https://arxiv.org/abs/2411.05841)
- **What's New**: FLEXtime(필터뱅크 학습 기반 설명 방법)이 제안되었습니다. 이 방법은 시계열 데이터를 주파수 대역으로 나누고 최적의 주파수 조합을 학습하여 기존의 'zeroing out' 방식의 단점을 극복합니다.

- **Technical Details**: FLEXtime은 필터뱅크를 사용하여 시계열 데이터를 여러 주파수 대역으로 분리합니다. 이 과정에서 시계열의 주파수 정보를 지역화하여 파악할 수 있으며, 이를 통해 보다 안정적이고 효율적인 최적화를 수행합니다.

- **Performance Highlights**: FLEXtime은 다양한 데이터셋에서 기존 설명 방법들보다 평균적으로 더 나은 성능을 보여주었으며, 뇌파(EEG) 및 오디오와 같은 다양한 시계열 데이터에 유용하게 활용될 수 있습니다.



### Fully Automated Correlated Time Series Forecasting in Minutes (https://arxiv.org/abs/2411.05833)
Comments:
          accepted by PVLDB 2025

- **What's New**: 이 논문에서는 자동으로 설계된 모델이 수동으로 설계된 모델보다 높은 정확도를 달성하는 새로운 CT 프로그램을 제안합니다. 특히, 검색 및 훈련을 몇 분 만에 완료할 수 있는 완전 자동화된 효율적인 관련 시계열(Correlated Time Series, CTS) 예측 프레임워크가 소개됩니다.

- **Technical Details**: 제안된 FACTS 프레임워크는 대규모 검색 공간을 정교하게 정리하여 새로운 예측 작업에 대한 고품질 검색 공간을 생성하는 데이터 기반의 반복적 전략을 포함합니다. 여기에는 사용자 지정 검색 공간 내에서 최적의 모델을 효율적으로 식별하는 제로샷 검색 전략과 식별된 모델의 훈련을 가속화하기 위한 빠른 매개변수 적응 전략이 포함됩니다.

- **Performance Highlights**: 일곱 가지 벤치마크 데이터 세트에 대한 실험 결과, FACTS 프레임워크가 최신 기술의 정확도를 달성하고 기존 방법보다 훨씬 더 효율적임을 입증했습니다. 특히, 빠른 매개변수 적응 전략을 통해 훈련 시간을 최대 66%까지 단축할 수 있었습니다.



### Open LLMs are Necessary for Current Private Adaptations and Outperform their Closed Alternatives (https://arxiv.org/abs/2411.05818)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 연구는 사적 데이터에 대한 적응을 위해 최근 제안된 폐쇄 LLM(Closed LLM)의 네 가지 방법을 분석합니다. 이들은 사용자 데이터를 제 3자 혹은 제공자에게 유출하지 않고도 사적 데이터로 조정될 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 각 방법의 보안 모델을 조사하고, 다양한 개인정보보호 수준(differential privacy, DP), 여러 LLM 아키텍처, 그리고 분류 및 생성 작업을 위한 다양한 데이터셋을 사용하여 성능을 비교합니다.

- **Performance Highlights**: 결과적으로 사용된 모든 방법은 사용자 쿼리 데이터를 LLM 제공자에게 유출하며, 제작된 방법 중 세 가지는 사적 데이터 훈련 데이터의 많은 부분을 제공자에게 유출합니다. 폐쇄 LLM에 대한 사적 적응 방법은 로컬 개방 LLM보다 성능이 낮고 거의 모든 비용이 더 높습니다.



### AI for ERW Detection in Clearance Operations -- The State of Research (https://arxiv.org/abs/2411.05813)
- **What's New**: 이번 논문은 전쟁에서 발생한 폭발물 잔여물(ERW) 제거를 위한 인공지능(AI) 연구의 최근 동향을 종합적으로 정리합니다.  특히 ERW 객체 탐지와 ERW 위험 예측으로 나뉘는 연구의 두 가지 주요 흐름을 발견하였습니다.

- **Technical Details**: 연구는 주로 AI를 활용한 ERW 탐지 및 위험 예측 기술에 집중됩니다. ERW 객체 탐지에 관한 연구가 활발히 진행된 반면, 위험 예측에 대해서는 연구가 부족하다는 점을 지적합니다. 저자들은 AI 시스템과 데이터 소스를 조합한 새로운 접근법, 그리고 패턴 기반 예측과 같은 혁신적인 방법을 제안하였습니다.

- **Performance Highlights**: 미래 연구를 위한 세 가지 기회를 제시하며, 특히 ERW 위험 예측에 대한 renewed (다시) 노력이 필요하다는 점을 강조합니다. 또한, 전통적인 machine learning (기계 학습)의 중요성과 전문가 지식을 동적으로 통합하는 필요성, 그리고 실제 운영과의 효과적인 AI 시스템 통합의 중요성을 언급합니다.



### Variational Bayes Decomposition for Inverse Estimation with Superimposed Multispectral Intensity (https://arxiv.org/abs/2411.05805)
- **What's New**: 이번 논문에서는 X선 강도와 같은 측정된 파동 강도에 대한 변분 베이지안 추론(Variational Bayesian Inference) 방법을 제시합니다. 이 방법은 관찰할 수 없는 객체의 특징에 대한 정보를 제공하는 데 유용합니다.

- **Technical Details**: 제안된 방법은 입자가 파동을 나타낸다고 가정하고, 이들의 행동을 확률적으로 모델링합니다. 게다가 데이터에 노이즈가 포함되어도 정확한 추론이 가능합니다. 이 방법은 또한 두 가지 실험 결과를 통해 그 실현 가능성을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 기존의 매개변수 피팅 및 스토캐스틱 추론 방법과 비교했을 때, 과적합(overfitting)에 저항력을 보이는 강점이 있습니다. 이는 샘플의 조성(compostion)을 설명하는 데 있어 노이즈 제거의 효율성을 제공합니다.



### A Comprehensive Survey of Time Series Forecasting: Architectural Diversity and Open Challenges (https://arxiv.org/abs/2411.05793)
Comments:
          Submitted to the Artificial Intelligence Review on October 10, 2024

- **What's New**: 이번 논문에서는 시계열 예측(time series forecasting)의 다양한 아키텍처의 발전 및 구조적 다양화에 대한 포괄적인 분석을 제공합니다. 특히, Transformer 모델이 장기 의존성(long-term dependencies)을 처리하는 데 우수하다는 점을 강조하며, 최근 단순 선형 모델이 Transformer보다 더 나은 성능을 보인 사례를 소개합니다.

- **Technical Details**: 시계열 예측은 시퀀스 형태의 역사적 데이터를 기반으로 미래 값을 예측하는 작업으로, 경제, 금융, 물류 등 여러 분야에서 활용됩니다. 기존에는 MLPs, RNNs, CNNs, GNNs와 같은 딥러닝 아키텍처가 사용되었으나, 각 구조의 귀납적 편향(inductive biases)으로 인한 성능 한계가 있었습니다. 이 연구는 다양한 아키텍처의 진화를 비교하고, 하이브리드 모델, 확산 모델(diffusion models), Mamba 모델과 같은 최신 트렌드 및 도전 과제를 찾아냅니다.

- **Performance Highlights**: 시계열 데이터의 복잡성과 다양성으로 인해 기존 모델의 일반화 성능이 제한되어 왔습니다. 그러나 본 논문에서는 구조적 다양성을 통해 연구자들이 시계열 예측 문제에 대한 새로운 접근을 시도할 수 있도록 하여, 새로운 연구 기회와 깊은 통찰력을 제공합니다.



### Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models (https://arxiv.org/abs/2411.07232)
Comments:
          Project page is at this https URL

- **What's New**: 이번 연구에서는 자연어 텍스트 지침에 따라 이미지를 편집하는 기존의 접근 방식의 한계를 극복하기 위해, Add-it이라는 새로운 방법을 제안합니다. 이 방법은 기존 장면을 유지하면서도 새로운 객체를 자연스럽게 통합하는 것에 중점을 둡니다.

- **Technical Details**: Add-it은 학습이 필요 없는 접근 방식을 통해 diffusion 모델의 attention 메커니즘을 확장하고, 장면 이미지, 텍스트 프롬프트 및 생성된 이미지를 포함한 세 가지 주요 정보를 결합합니다. 이 확장된 attention 메커니즘은 구조적 일관성을 유지하며 객체의 자연스러운 배치를 보장합니다.

- **Performance Highlights**: Add-it은 이전 방법들과 비교해 객체 삽입 작업에서 최첨단의 성능을 달성했습니다. 실제 이미지 및 생성된 이미지 삽입 벤치마크에서 뛰어난 결과를 보였으며, 인간 평가에서도 80% 이상의 선호도를 기록했습니다.



### Grounding Video Models to Actions through Goal Conditioned Exploration (https://arxiv.org/abs/2411.07223)
Comments:
          Project page at this https URL

- **What's New**: 이번 연구에서는 대규모 비디오 모델을 통해 얻은 물리적 지식으로 Agent의 동작과 목표를 시각적으로 탐색할 수 있는 새로운 방법을 제안합니다. 기존의 방법들이 Agent 특정 데이터에 기반한 별도의 vision-based inverse dynamic model을 사용해야 했던 것에 비해, 우리는 생성된 비디오 상태를 탐색의 시각적 목표로 활용하여 이를 해결하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 trajectory level action generation과 비디오 안내를 결합하여 Agent가 외부의 감독 없이도 복잡한 작업을 해결할 수 있게 합니다. 연구진은 Libero, MetaWorld, Calvin, iThor Visual Navigation의 다양한 환경에서 8개, 6개, 4개, 12개의 작업을 검증하며, 이론적으로는 행동 클로닝(behavior cloning) 기준선과 유사하거나 더 뛰어난 결과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 50개의 시연 데이터를 이용해 훈련된 BC보다도 높은 성공률을 보여, 비디오 모델에 의해 주어진 목표를 환경 속 탐색을 통해 직접 학습함으로써 이뤄낸 성과입니다. 실험 결과는 훈련 데이터의 양과 다양성에 따라 성과가 향상된다고 나타났으며, 특히 비디오 탐색 빈도와 비디오 모델의 수평선(horizon) 변동이 탐색 성능에 미치는 영향을 평가했습니다.



### Data-Driven Predictive Control of Nonholonomic Robots Based on a Bilinear Koopman Realization: Data Does Not Replace Geometry (https://arxiv.org/abs/2411.07192)
Comments:
          23 pages, 12 figures

- **What's New**: 이 논문은 데이터 기반의 모델 예측 제어(MPC) 방식을 비모델적(nonholonomic) 차량에 최초로 적용하고 실험적으로 검증한 내용을 담고 있습니다.

- **Technical Details**: Extended Dynamic Mode Decomposition (EDMD)를 사용하여 모델을 추론하며, 실험에서 얻은 실제 데이터로부터 비모델적 모바일 로봇의 2차 동적 모델을 학습합니다. 이때 서브 리만 기하학(sub-Riemannian geometry) 및 액추에이터 동적 모델을 고려한 비용 함수를 설계합니다.

- **Performance Highlights**: 실험과 시뮬레이션을 통해 데이터 기반 모델이 고정밀 예측 제어기를 지원하며, 충분한 데이터 없이도 EDMD 기반의 예측 제어기를 사용할 수 있음을 보여줍니다. 하지만 비모델적 시스템의 기하학을 무시하는 데이터 중심 접근 방식의 문제점도 지적하고 있습니다.



### NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics (https://arxiv.org/abs/2411.07186)
Comments:
          Demo page: this https URL The code will be open-sourced and available shortly

- **What's New**: 이 논문에서는 생명과학 용도로 특별히 설계된 최초의 오디오-언어 기초 모델인 NatureLM-audio를 소개합니다. 이는 다양한 생물음향(bioacoustics) 과제를 해결하는 데 성공적으로 활용되었으며, 특히 보기 드문 종에 대한 제로샷(zero-shot) 분류를 포함합니다.

- **Technical Details**: NatureLM-audio는 음성 및 음악 데이터와 함께 훈련된 커다란 데이터셋으로부터 학습된 표현들을 동물의 음성에 성공적으로 전이하는 능력을 보여줍니다. 본 모델은 새로운 벤치마크인 BEANS-Zero에서 성능을 평가받아 여러 생물음향 과제에서 새로운 최첨단(SotA) 성능을 기록하였습니다.

- **Performance Highlights**: NatureLM-audio는 고급 생물음향 과제를 해결함에 있어 뛰어난 제로샷(zero-shot) 성능을 보였으며, 이는 이전에 보지 못한 종과 과제에 효과적으로 적용될 수 있음을 의미합니다. 이 모델은 생물다양성 모니터링 및 보전 연구에 대한 기여 가능성을 높이고 있습니다.



### Counterfactual Generation from Language Models (https://arxiv.org/abs/2411.07180)
Comments:
          A preprint

- **What's New**: 언어 모델에서 인과 생성 메커니즘을 이해하고 조작하는 것은 모델의 행동을 제어하는 데 필수적입니다. 본 논문에서는 기존의 개입(intervention) 기술 외에도, 카운터팩추얼(counterfactual) 사고방식을 강조하여 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 언어 모델을 일반화된 구조 방정식(Generalized Structural-equation) 모델로 재구성하여 진짜 문자열 카운터팩추얼을 생성합니다. Gumbel-max 트릭(Gumbel-max trick)을 사용하여 샘플링 노이즈의 동일한 인스턴스에서 원래 문자열과 카운터팩추얼 간의 결합 분포(joint distribution)를 모델링합니다. 이 알고리즘은 후견 Gumbel 샘플링(hindsight Gumbel sampling)을 기반으로 하여 관찰된 문자열의 카운터팩추얼을 생성하기 위한 잠재적인 노이즈 변수를 추론합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 의미 있는 카운터팩추얼을 생성하는 동시에 일반적으로 사용되는 개입 기술이 상당한 원하지 않는 부작용을 가진다는 것을 보여줍니다.



### Joint Age-State Belief is All You Need: Minimizing AoII via Pull-Based Remote Estimation (https://arxiv.org/abs/2411.07179)
- **What's New**: 이 논문은 잘못된 정보의 나이(AoII)라는 새로운 정보 신선도 측정 기준을 제안하며, 이는 추정 오류와 그 지속 시간을 함께 고려합니다. 특히, 시간 슬롯 기반의 원격 추정 시스템을 다루며, 확률론적 정보 원천에서 효율적으로 데이터를 관리하는 방안을 모색합니다.

- **Technical Details**: 제안된 시스템 모델은 일반적인 이산 시간 마르코프 체인(DTMC) 프로세스를 활용하고 있으며, 패킷 전송 시간의 비제로 제약을 둡니다. 이를 통해 모니터는 실제 AoII 프로세스에 대한 완전한 정보를 갖지 않으며, 대신에 'belief'라는 충분 통계량을 통해 전반적인 상태를 추정하게 됩니다. 이 연구에서는 최대 사후 추정기(MAP estimator)를 활용하여 기존의 마팅게일 추정기 대신 모니터에서 정보를 갱신합니다.

- **Performance Highlights**: 저자는 두 가지 belief 기반 정책을 제안하며, 그 중 하나는 심층 강화 학습(DRL)을 기반으로 하고, 다른 하나는 순간적인 기대 AoII에 기반한 임계값 정책입니다. 두 정책은 기존의 베이스라인 정책들과 비교됩니다.



### More Expressive Attention with Negative Weights (https://arxiv.org/abs/2411.07176)
- **What's New**: Cog Attention이라는 새로운 주의 메커니즘을 제안하며, 이는 음의 주의 가중치를 허용하여 표현력을 향상시킵니다. 이 메커니즘은 토큰 삭제 및 복사를 정적인 OV 매트릭스에서 동적인 QK 내적(product)으로 전환할 수 있도록 합니다.

- **Technical Details**: Cog Attention은 음의 주의 가중치를 사용할 수 있는 구조로, 토큰의 삭제, 복사 또는 보유를 각각 음의, 양의 또는 최소 주의 가중치로 할당하여 단일 주의 헤드가 더 유연하고 표현력이 높아지도록 합니다.

- **Performance Highlights**: Cog Attention을 사용한 모델들이 전통적인 softmax 주의 모듈을 사용한 모델들에 비해 우수한 성능을 보이며, 이는 언어 모델링 및 이미지 생성 등 다양한 작업에서 입증되었습니다.



### Acoustic-based 3D Human Pose Estimation Robust to Human Position (https://arxiv.org/abs/2411.07165)
Comments:
          Accepted at BMVC2024

- **What's New**: 이 논문에서는 저수준의 음향 신호만을 이용하여 3D 인간 자세 추정 문제를 탐구합니다. 기존의 방법은 사용자가 스피커와 마이크 사이의 직선 위에 위치해야 한다고 가정하였으나, 이는 실세계에서의 적용에 제한이 있었습니다. 이에 따라 위치 판별기와 잔향 저항 모델을 결합한 새로운 방법을 제안하였습니다.

- **Technical Details**: 제안된 방법은 위치 판별기가 포함되어 있으며, 주체의 위치에 관계없이 인식할 수 있는 특징을 추출합니다. 또한, 추정 대상 시간 이전의 음향 신호를 참조하여 신호의 변동에 대한 강건성을 높이는 방법을 제안합니다. 이 논문에서는 스피커와 마이크 간의 직선에서 멀리 떨어진 여러 위치에서의 데이터를 포함한 새로운 데이터셋을 구축하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 접근 방식보다 우수한 성능을 보였으며, 다양한 인간 위치를 커버하는 경우에도 안정적인 3D 자세 추정이 가능함을 입증하였습니다.



### Conditional simulation via entropic optimal transport: Toward non-parametric estimation of conditional Brenier maps (https://arxiv.org/abs/2411.07154)
Comments:
          26 pages, 4 figures

- **What's New**: 본 논문에서는 조건부 브레니어 맵(conditional Brenier maps)을 위한 비모수적 추정기를 제안합니다. 이는 컴퓨터 계산에서의 확장성을 강조하는 엔트로픽 최적수송(entropic optimal transport)에 기반하고 있습니다. 이 접근 방식은 최적 수송 지도의 근사적 결과를 이용하여 처리됩니다.

- **Technical Details**: 제안된 비모수적 추정기는 Carlier et al. (2010)의 결과를 활용하여, 재조정된 이차 비용 아래 최적 수송 맵이 조건부 브레니어 맵으로 점근적으로 수렴하는 특성을 지니고 있습니다. 이를 통해, 샘플 수에 대한 스케일링 매개변수를 선택하는 방식을 전체적으로 설명하고 있습니다.

- **Performance Highlights**: 기존의 머신러닝 및 비모수적 접근 방식과의 성능 비교를 통해 제안된 추정기의 효과성을 입증하였으며, 벤치마크 데이터셋 및 베이지안 추론 문제에서 유의미한 개선을 보였습니다.



### Benchmarking LLMs' Judgments with No Gold Standard (https://arxiv.org/abs/2411.07127)
- **What's New**: GEM(Generative Estimator for Mutual Information)라는 새로운 평가 지표를 소개하며, 이는 기존의 gold standard reference 없이 Large Language Models(LLMs)의 언어 생성 성능을 평가할 수 있는 방법을 제공합니다. GEM은 LLM의 생성 성능을 기존의 기계 번역, 요약 같은 전통적인 과제에서 학술지 peer review와 같은 주관적 과제까지 확장할 수 있게 합니다.

- **Technical Details**: GEM은 후보 응답과 참조 응답 간의 상호 정보(mutual information)를 추정하는 생성 모델입니다. 이는 gold standard 품질이 필요 없으며, 후보 응답이 참조 응답에 대한 정보를 얼마나 제공하는지를 측정합니다. GEM의 변형으로 GEM-S가 있으며, 이는 특정 작업의 요약을 기반으로 상호 정보를 추정합니다. 실험 결과, GEM은 기존의 평가 지표와 비교하여 더 뛰어난 민감성과 조작 저항성을 보여줍니다.

- **Performance Highlights**: 실험 결과 GEM과 GEM-S는 인간 평가와 높은 상관관계를 보이며, 다른 기초 지표보다 우수한 성능을 드러냈습니다. 특히, 모든 의미적 열화에 대해 민감성을 나타내는 유일한 지표로, 무의미한 응답 연장 후에도 점수가 크게 증가하지 않았습니다. 추가로, GRE-bench를 통해 다양한 LLM의 peer review 능력을 평가하였고, 파라미터 크기와 GRE-bench 점수 간의 강한 상관관계를 발견했습니다.



### Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models (https://arxiv.org/abs/2411.07126)
- **What's New**: Edify Image라는 새로운 생성 모델이 도입되었습니다. 이 모델은 픽셀 완벽한 정확도를 가진 포토리얼리스틱 이미지 콘텐츠를 생성할 수 있는 여러 가지 확산 모델로 구성되어 있습니다.

- **Technical Details**: Edify Image는 다단계(cascaded) 픽셀-스페이스(diffusion models) 확산 모델을 활용하여, 이미지 신호를 다양한 주파수 대역에서 서로 다른 비율로 감쇠시키는 새로운 라플라시안(Laplacian) 확산 프로세스를 사용합니다. 이 모델은 고해상도 이미지를 생성하며, 텍스트-이미지 합성, 4K 업샘플링, ControlNets 및 360도 HDR 파노라마 생성과 같은 다양한 응용 프로그램을 지원합니다.

- **Performance Highlights**: Edify Image 모델은 긴 텍스트 프롬프트를 처리하며, 다양한 종횡비로 이미지를 생성하고, 인간 주제를 생성할 때 향상된 공정성과 다양성을 보장합니다. 또한, 주어진 낮은 해상도 이미지를 기반으로 높은 주파수 세부사항을 합성할 수 있는 4K 업샘플러 모델과 다양한 모달리티에 대한 ControlNets로 이미지를 생성할 수 있는 기능을 제공합니다.



### ConvMixFormer- A Resource-efficient Convolution Mixer for Transformer-based Dynamic Hand Gesture Recognition (https://arxiv.org/abs/2411.07118)
- **What's New**: 이 연구에서는 동적 손 제스처 인식을 위해 새로운 ConvMixFormer 아키텍처를 제안합니다. 기존의 transformer 모델의 self-attention 대신 간단한 CNN 기반의 token mixer를 사용하여 계산 복잡성을 줄이고, 더 적은 파라미터로 로컬 공간 특성을 캡처합니다.

- **Technical Details**: ConvMixFormer는 self-attention의 복잡한 계산을 피하고, convolution 레이어를 사용하여 입력 이미지의 spatial tokens를 혼합합니다. Gated Depthwise Feed Forward Network (GDFN)를 활용하여 정보 흐름을 제어하는 기능이 추가되어, 적은 파라미터로 효율적인 학습이 가능합니다.

- **Performance Highlights**: 제안된 모델은 NVidia Dynamic Hand Gesture 및 Briareo 데이터셋에서 state-of-the-art 성능을 달성했으며, 특히 단일 및 다중 모달 입력에서 우수한 결과를 보였습니다. ConvMixFormer는 기존 모델에 비해 파라미터 효율성이 뛰어납니다.



### TinyML Security: Exploring Vulnerabilities in Resource-Constrained Machine Learning Systems (https://arxiv.org/abs/2411.07114)
Comments:
          Submitted to Proceedings of the IEEE

- **What's New**: 이 논문은 Tiny Machine Learning (TinyML) 시스템의 보안 위협에 대한 최초의 포괄적인 조사 결과를 제공합니다. 연구는 IoT, EdgeML 및 TinyML의 장치 분류를 통해 TinyML 고유의 취약성을 강조하고 있으며, 다양한 공격 벡터를 나열하고 그 위협 수준을 평가하며 기존 및 가능한 방어책을 평가합니다. 또한 기존 보안 솔루션의 적합성과 TinyML에 특화된 솔루션의 필요성을 강조합니다.

- **Technical Details**: TinyML 시스템은 RAM과 CPU의 제약으로 인해 기존 보안 솔루션을 직접적으로 적용하기 어려운 환경에 위치하고 있습니다. 이 논문에서는 Common Vulnerability Scoring System (CVSS)를 사용하여 공격의 심각성과 잠재적 영향을 평가하며, 하드웨어, 소프트웨어 및 모델 보안 기술의 실행 가능성을 조사합니다. 또한, 다양한 유형의 공격 벡터와 위험 모델을 제시하여 TinyML 장치의 보안 문제를 체계적으로 분석합니다.

- **Performance Highlights**: 이 연구는 TinyML 보안 연구의 현황을 강조하며, 보안이 무시되는 상황에서 TinyML 기술이 빠르게 발전하고 있음을 경고합니다. 기존 연구의 부족을 지적하며, 여러 산업에서 TinyML의 배포를 고려할 때 보안 문제의 해결이 긴급하다고 강조합니다. 앞으로의 연구 방향으로는 경량 보안 솔루션의 개발과 TinyML 모델의 고유 취약성 분석이 포함됩니다.



### Training Neural Networks as Recognizers of Formal Languages (https://arxiv.org/abs/2411.07107)
Comments:
          40 pages, 2 figures. Preprint

- **What's New**: 본 논문은 신경망 아키텍처의 계산적 능력을 형식 언어 이론의 관점에서 정량적으로 검증하려는 새로운 접근 방식을 제안합니다. 일반적인 실험 방법 대신 문자열 이진 분류기로 신경망을 훈련시키고 평가하는 방식을 적용했습니다. 또한, Snæbjarnarson et al. (2024)의 알고리즘을 확장하여 정규 언어의 문자열 길이 제어 샘플링을 수행하여 더 나은 시간 복잡도를 확보했습니다.

- **Technical Details**: 이 연구는 Chomsky 계층의 여러 언어에 대해 세 가지 신경망 아키텍처(RNN, LSTM, causally-masked transformer)의 성능을 비교 분석합니다. 신경망을 형식 언어의 인식기로 훈련시키기 위해, 긍정 샘플링 및 회원 테스트를 위한 언어 특정 알고리즘을 사용하는 방법을 제안합니다. 길이 제어 샘플링을 위한 비판적인 알고리즘을 구현하여, 필수적인 부정적인 샘플을 생성하고, 효율적인 훈련 방식을 확인했습니다.

- **Performance Highlights**: 실험 결과, RNN과 LSTM이 transformer보다 자주 더 나은 성능을 보였으며, 보조 훈련 목표는 특정 아키텍처에서 도움이 되기도 했지만, 전반적으로 일관된 향상 효과는 없었습니다. 연구 결과는 FLaRe(Formal Language Recognition)라는 벤치마크로 제공되며, 향후 언어 인식 주장을 이론적으로 뒷받침하는데 유용할 것입니다.



### Learning Multi-Agent Collaborative Manipulation for Long-Horizon Quadrupedal Pushing (https://arxiv.org/abs/2411.07104)
- **What's New**: 본 논문에서는 여러 사족 로봇이 협력하여 장애물을 인식하고 긴 범위 동안 물체를 누르는 과제를 해결하는 방법을 제안합니다. 이를 위해 3단계로 구성된 계층적 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning, MARL) 프레임워크를 개발하였습니다.

- **Technical Details**: 제안된 프레임워크는 고수준 컨트롤러가 장애물 회피를 위해 RRT(Rapidly-exploring Random Tree) 계획자와 중앙 집중식 적응 정책을 통합하고, 중간 수준 컨트롤러가 분산형 목표 조건 정책을 사용하여 로봇이 서브 목표를 향해 나아가도록 하며, 저수준 컨트롤러는 사전 훈련된 보행 정책으로 명령을 실행합니다.

- **Performance Highlights**: 시뮬레이션에서 제안된 방법은 기존 방법들에 비해 36.0% 더 높은 성공률과 24.5% 단축된 완료 시간을 기록했습니다. 실제 Go1 로봇에서 Push-Cuboid 및 Push-T 작업을 성공적으로 수행함으로써 제안된 방법이 실제 환경에서도 효과적임을 입증하였습니다.



### Effectively Leveraging Momentum Terms in Stochastic Line Search Frameworks for Fast Optimization of Finite-Sum Problems (https://arxiv.org/abs/2411.07102)
- **What's New**: 이번 연구에서는 대규모 딥러닝 시나리오에서 발생하는 제약 없는 유한 합 최적화 문제를 다룹니다. 특히, 최근 스토캐스틱 최적화를 위한 선형 탐색 접근 방식과 모멘텀 방향의 관계를 탐구합니다.

- **Technical Details**: 본 연구에서는 mini-batch persistency 개념을 기반으로 한 알고리즘 프레임워크를 제안합니다. 이 프레임워크는 데이터 지속성, 공액 그래디언트(conjugate-gradient) 유형 규칙, 스토캐스틱 선형 탐색을 혼합하여 모멘텀 파라미터를 정의합니다.

- **Performance Highlights**: 제안된 알고리즘은 실험적으로 기존의 유명한 방법들보다 뛰어난 성능을 보였으며, 볼록(convex) 및 비볼록(nonconvex) 대규모 훈련 문제 모두에서 최첨단 결과를 얻었습니다.



### Bounded Rationality Equilibrium Learning in Mean Field Games (https://arxiv.org/abs/2411.07099)
- **What's New**: 이 연구는 대규모 에이전트 집단에서의 행동을 모델링하기 위해 Mean Field Games (MFGs)의 새로운 접근 방식을 제시하며, Nash Equilibrium (NE)의 한계를 극복하기 위해 bounded rationality를 포함한다.

- **Technical Details**: 논문에서는 quantal response equilibria (QRE)와 receding horizon (RH) MFGs를 정의하고, 이러한 개념들이 MFGs의 새로운 유형으로서 에이전트의 제한된 합리성을 모델링할 수 있음을 보여준다. QRE는 노이즈에 의해 왜곡된 보상을 인식하고 이에 따라 최적의 행동을 취하는 에이전트를 가정한다.

- **Performance Highlights**: 제안된 알고리즘을 사용하여 다양한 예제를 통해 QRE와 RH 균형을 학습하는 능력을 평가하였으며, 실용적인 차이점을 명확히 구분하여 기존의 균형 개념과 비교하였다.



### Towards Characterizing Cyber Networks with Large Language Models (https://arxiv.org/abs/2411.07089)
Comments:
          5 pages, 2 figures

- **What's New**: 본 논문에서는 Cyber Log Embeddings Model (CLEM)을 통해 사이버 데이터의 잠재적 특성을 활용하여 이상 징후를 탐지하는 새로운 접근 방식을 제안합니다. CLEM은 실제 네트워크와 IoT 사이버 보안 테스트베드의 Zeek 네트워크 트래픽 로그를 사용하여 훈련되었습니다.

- **Technical Details**: CLEM은 자연어 처리(NLP) 기법을 활용하여 사이버 보안 로그에서 발견되는 언어적 구조를 모델링하며, 유사한 행동 패턴에 따라 기계들을 군집화(cluster)합니다. 모델은 Sliding Window 기술을 사용하여 데이터의 각 구간을 특성화하고, Adjusted Rand Index (ARI)를 통해 CLEM의 출력 결과와 전문가 레이블의 비교를 수행합니다.

- **Performance Highlights**: CLEM은 고차원적이고 복잡한 사이버 데이터 내에서 비정상적인 행동을 식별하는 데 있어 promise를 보여줍니다. 이 모델은 대규모 언어 모델(LLM)을 통해 사이버 데이터 이해에 기여하며, 실험 결과는 전통적인 방법에 비해 유의미한 성능 향상을 나타냅니다.



### OCMDP: Observation-Constrained Markov Decision Process (https://arxiv.org/abs/2411.07087)
Comments:
          Full paper, 14 Pages

- **What's New**: 이 논문에서는 Observation-Constrained Markov Decision Process (OCMDP)를 도입하여 비용 민감한 환경에서 관찰 및 제어 전략을 동시에 학습하는 문제를 다룹니다. 이는 전통적 제어 시스템에서 가정하는 완전 관측 가능성이 비현실적임을 인식하고, 정책이 실제 상태의 관측 가능성에 영향을 주는 구조입니다.

- **Technical Details**: 제안된 OCMDP는 상태 공간, 행동 공간, 관찰 집합, 상태 전이 함수, 관찰 함수, 보상 함수 및 비용 함수로 정의됩니다. 새로운 모델은 관찰 및 제어 정책을 분리하여 작동하며, 기계 학습으로 최적의 관찰 및 제어 행동을 고안합니다. 함께 학습하는 과정에서 ‘관찰을 무엇으로 할 것인지’와 ‘언제 관찰할 것인지’에 대한 의사결정이 중요합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 시뮬레이션 진단 작업 및 현실적인 의료 환경에서 기존의 방법들보다 평균적으로 관찰 비용을 크게 줄이며, 효율성에서 두드러진 성과를 보여줍니다. 이를 통해 실제 의료 결정 과정에서의 적용 가능성을 입증하였습니다.



### To Train or Not to Train: Balancing Efficiency and Training Cost in Deep Reinforcement Learning for Mobile Edge Computing (https://arxiv.org/abs/2411.07086)
- **What's New**: 이 논문에서는 6G 네트워크에서 인공지능(AI)의 역할을 다루고 있으며, Mobile Edge Computing (MEC)의 효율적인 리소스 할당 문제에 대해 다룹니다. 기존 연구들이 교육 비용을 간과했던 점을 지적하며, 더 현실적인 상황에서의 모델을 제안합니다.

- **Technical Details**: 제안된 새로운 알고리즘은 Deep Reinforcement Learning (DRL) 에이전트의 교육 시점을 동적으로 선택하는 방식으로, 교육 오버헤드를 고려한 시스템입니다. 이 방법은 다양한 훈련 비용이 있는 시나리오에 직접 적용할 수 있으며, 이상적인 학습 에이전트와 유사한 성능을 달성할 수 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 교육 비용을 반영하는 상황에서도 이상적인 성능에 가까운 결과를 보여 줍니다. 따라서 이 접근 방식은 실제 환경에서의 효율적인 리소스 관리에 큰 기여를 할 수 있습니다.



### An Interpretable X-ray Style Transfer via Trainable Local Laplacian Filter (https://arxiv.org/abs/2411.07072)
- **What's New**: 이번 연구에서는 방사선 의사가 X-ray 이미지를 진단 성능 향상을 위해 선호하는 시각적 인상, 즉 '스타일'을 자동으로 전환할 수 있는 기법을 제안합니다.

- **Technical Details**: 제안하는 방법은 Local Laplacian Filter (LLF)의 훈련 가능한 버전을 도입하여 스타일 전환의 최적화된 변환 함수를 형성하고, 이를 통해 스타일 전환의 특성을 추론할 수 있도록 합니다. MLP (Multi-Layer Perceptron)를 사용하여 LLF가 복잡한 X-ray 스타일 특징을 포착할 수 있도록 했습니다.

- **Performance Highlights**: 실험 결과, 처리되지 않은 유방 X-ray 이미지를 목표 유방 촬영 이미지의 스타일에 맞춰 변환하여 기존 LLF 스타일 전환 방법의 SSIM (Structural Similarity Index) 0.82에 비해 0.94를 달성하는 효과성을 입증했습니다.



### Reconstruction of neuromorphic dynamics from a single scalar time series using variational autoencoder and neural network map (https://arxiv.org/abs/2411.07055)
Comments:
          15 pages, 15 figures, 3 tables

- **What's New**: 이 논문에서는 단일 스칼라 시계열을 사용하여 신경형 행동을 갖는 동적 시스템의 가족을 재구성하는 방법을 제시합니다. Hodgkin-Huxley 포멀리즘에 기반한 생리학적 뉴런의 모델을 고려하며, 이 모델의 변수를 하나만 사용하여 신경망을 훈련시킬 수 있음을 보여줍니다.

- **Technical Details**: 연구는 두 단계로 진행됩니다. 첫 번째 단계에서, 원래 시계열에서 지연 좌표 임베딩 벡터를 생성하고, 변분 오토인코더(Variational Autoencoder)를 통해 차원을 축소하여 복원된 상태 공간 벡터를 얻습니다. 두 번째 단계에서는 연속적인 시간 단계에서 복원된 상태 공간 벡터의 쌍과 일정 값을 결합하여 또 다른 신경망을 훈련시켜 재귀 맵으로 작동하도록 합니다.

- **Performance Highlights**: 제어 매개변수를 변화시킬 때 관찰된 신경망 시스템의 동작이 원래 시스템과 매우 잘 일치하는 결과를 보여 주며, 이는 훈련 중에 명시적으로 나타나지 않았던 것입니다.



### FedCVD: The First Real-World Federated Learning Benchmark on Cardiovascular Disease Data (https://arxiv.org/abs/2411.07050)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문은 실제 세계의 심혈관 질환(CVD) 탐지를 위한 첫 번째 연방 학습(Federated Learning, FL) 벤치마크인 FedCVD를 소개합니다. FedCVD는 7개 기관에서 수집된 자연적으로 분산된 CVD 데이터를 기초로 하여 ECG 분류 및 ECHO 분할의 두 가지 주요 작업을 포함합니다.

- **Technical Details**: FedCVD는 비독립적이고 동일하게 분포되지 않는(Non-IID) 데이터, 긴 꼬리 분포(Long-tail Distribution), 그리고 레이블 불완전성(Label Incompleteness)이라는 세 가지 도전 과제를 강조합니다. 이 데이터 세트와 벤치마크는 실제 의료 환경에서 FL 알고리즘 개발 및 검증에 중요한 역할을 합니다.

- **Performance Highlights**: FedCVD에서의 실험 결과, FL은 기존 중앙 집중식 학습 방법과 비교하여 CVD 문제에서 효과적임을 검증했으며, 새로운 평가 지표를 제시하였습니다. 모든 데이터 세트는 자연스러운 분할 전략을 기반으로 하여 구성되었습니다.



### Unified Bayesian representation for high-dimensional multi-modal biomedical data for small-sample classification (https://arxiv.org/abs/2411.07043)
Comments:
          36 pages, 3 figures and 3 tables

- **What's New**: 새로운 Bayesian 알고리즘 BALDUR는 고차원에서 multi-modal datasets와 소규모 샘플 크기를 처리하고 설명 가능한 솔루션을 제공합니다. 이 모델은 공통 잠재 공간에서 다양한 데이터 뷰를 통합하여 분류 작업을 해결하고 불필요한 기능 및 데이터 뷰를 제거합니다.

- **Technical Details**: BALDUR는 두 개의 커널을 효과적으로 통합하며, 소규모 샘플-특징 비율에서 일반화 가능한 솔루션을 제공합니다. 또한, 모델의 선형적 특성은 바이오마커(biomarker) 식별을 위한 설명 가능한 결과를 보장합니다.

- **Performance Highlights**: BALDUR 모델은 두 가지 신경퇴행성(neurodegeneration) 데이터셋에서 테스트되어, 최신 모델들을 초월하는 성능을 보였으며, 이미 과학 문헌에 설명된 마커와 일치하는 특징을 탐지했습니다.



### Data-Driven Gradient Optimization for Field Emission Management in a Superconducting Radio-Frequency Linac (https://arxiv.org/abs/2411.07018)
Comments:
          14 pages, 6 figures, 10 tables

- **What's New**: 이번 연구에서는 슈퍼컨덕팅 라디오 주파수 선형 가속기(SRF linacs)에서 발생하는 필드 방출(field emission)로 유도된 방사선 문제를 해결하기 위해 기계 학습 기법을 적용하고 불확실성 평가(uncertainty quantification)를 수행합니다. 이러한 접근법은 방사선 수준을 예측하고 우수한 결과를 도출하기 위해 사용됩니다.

- **Technical Details**: 연구팀은 RF 제어 시스템과 NDX 시스템의 데이터를 활용하여 모델을 개발했습니다. RF 시스템에서 측정된 캐비티 기울기(cavity gradient)와 NDX 감지기를 통해 측정된 중성자 및 감마 방사선의 선량률(dose rate)을 기반으로 기계 학습 예측 모델을 구축합니다. 이러한 모델은 캐비티 기울기와 방사선 수준 간의 관계를 설명하여, 방사선 수준을 낮추고 가속기의 에너지 이득(energy gain)을 최적화하는 데 기여합니다.

- **Performance Highlights**: 최적화된 솔루션은 표준 운영 설정에 비해 40% 이상의 중성자 및 감마 방사선 수준 감소를 달성했습니다. 이 결과들은 CEBAF 가속기에서 필드 방출로 인한 방사선 문제를 효과적으로 저감할 수 있는 가능성을 제시합니다.



### Estimating Causal Effects in Partially Directed Parametric Causal Factor Graphs (https://arxiv.org/abs/2411.07006)
Comments:
          Accepted to the Proceedings of the 16th International Conference on Scalable Uncertainty Management (SUM 2024)

- **What's New**: 이번 논문에서는 인과 추론(causal inference)을 부분적으로 지향적인 그래프(partially directed graphs)에 적용하는 방법을 제시합니다. 부분적으로 지향적인 인과 인자(parametric causal factor graphs, PCFGs)의 일반화로서 부분적으로 지향적인 인과 인자 그래프(partially directed parametric causal factor graphs, PPCFGs)를 소개하며, 인과적 관계에 대한 사전 지식이 적게 필요한 확장된 인과 추론을 가능하게 합니다.

- **Technical Details**: PPCFGs는 지향된(edge) 및 비지향된(edge) 관계를 모두 포함하는 그래프 형태로, 주어진 랜덤 변수(random variables) 집합에 대한 전체 결합 분포(joint distribution)를 압축적으로 인코딩할 수 있습니다. PPCFG에서 d-separation을 정의하여 조건부 독립성(conditional independence)을 판단할 수 있는 기초를 마련하고, 인과 효과(causal effect)의 효율적인 추정 알고리즘을 제시합니다.

- **Performance Highlights**: PPCFG를 적용한 인과 효과 추정 알고리즘은 동일한 표본을 대표하는 객체를 사용하여 추론(inference)의 속도를 높이며, 비지향적인 엣지가 포함된 경우에도 가능한 모든 인과 효과를 효율적으로 나열할 수 있습니다. 이를 통해 실제 데이터와 부분적으로 지향적인 그래프를 사용하여 인과 효과를 추정하는 데 있어 실용적인 접근을 제시합니다.



### Which PPML Would a User Choose? A Structured Decision Support Framework for Developers to Rank PPML Techniques Based on User Acceptance Criteria (https://arxiv.org/abs/2411.06995)
- **What's New**: 이 논문에서는 사용자 선호에 기반하여 Privacy Preserving Machine Learning (PPML) 기술을 선택하는 데 도움이 되는 의사결정 지원 프레임워크를 제안합니다.

- **Technical Details**: Privacy-Enhancing Technologies (PETs) 및 User Acceptance Criteria (UAC)를 기반으로 PPML 기술의 다양한 특성을 분석하여 기술적인 차별화를 도출합니다. 이를 통해 PPML 방법을 사용 사례에 맞추어 평가합니다.

- **Performance Highlights**: 이 연구는 사용자 수용 기준에 따라 PPML 기술을 순위 매기는 방법을 제공하며, 개인정보에 관련된 정보를 분류하는 사례를 통해 응용 프로그램을 시연합니다.



### Causal-discovery-based root-cause analysis and its application in time-series prediction error diagnosis (https://arxiv.org/abs/2411.06990)
Comments:
          10 pages with 5 figures

- **What's New**: 최근의 기계 학습(ML) 발전은 예측 모델의 정확도를 크게 향상시켰으나, 대부분의 모델이 '블랙 박스(Black Box)' 형태로 남아 있어 예측 오류 진단이 어렵습니다. 이 논문에서는 사전 정의된 인과 그래프 없이 예측 오류와 설명 변인 간의 인과 관계를 추정하는 새로운 방법인 CD-RCA(Causal-Discovery-based Root-Cause Analysis)를 제안합니다.

- **Technical Details**: CD-RCA는 Shapley 값(Shapley values)을 활용하여 예측 오류에 대한 원인 변인의 기여를 식별합니다. 이 방법은 합성 오류 데이터(지금까지와 다른 데이터)를 시뮬레이션하여 예측 오류의 이상치에 대한 변인 기여도를 평가합니다. 기존의 휴리스틱 어트리뷰션 방법들에 비해 더 높은 성능을 보입니다.

- **Performance Highlights**: 광범위한 시뮬레이션 결과 CD-RCA는 현재의 방법들보다 예측 오류의 이상치에 대한 원인 분석에서 우수한 성능을 나타내며, Shapley 값의 오배당 오류를 나타내는 새로운 패턴을 발견했습니다. 이 연구는 XAI(Xplainable AI) 애플리케이션의 신뢰성과 안전성을 향상시키는 데 기여할 것으로 기대됩니다.



### Data-driven discovery of mechanical models directly from MRI spectral data (https://arxiv.org/abs/2411.06958)
Comments:
          11 pages regular paper with 8 figures, 9 pages supplementary material with 6 figures, 1 supplementary video

- **What's New**: 본 연구는 비선형 생체역학적 모델을 실험적으로 얻은 undersampled MRI 스펙트럼 데이터를 기반으로 데이터 기반 발견(data-driven discovery, DDD) 방법론을 통해 복원하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 Spectro-dynamic framework와 Sparse Identification of Non-linear Dynamics (SINDy) 방법을 결합하여, motion의 주기성에 의존하지 않고 displacement field를 복원하고 해석 가능한 모델을 동시에 식별합니다. 이 과정은 k-space 데이터를 직접 사용하여 중간 이미지를 재구성하는 단계(temporal series images)를 생략함으로써 높은 시간 해상도를 가능하게 합니다.

- **Performance Highlights**: 동적 팬텀 데이터를 사용하여 임상 MRI 스캐너에서 수집된 스펙트럼 데이터로 검증하였으며, 2단계 접근법보다 모델 식별 성능이 뛰어난 결과를 얻었습니다. 이는 실제 생체조직에 대한 데이터 기반 모델 발견의 가능성을 시사합니다.



### Understanding Generalization in Quantum Machine Learning with Margins (https://arxiv.org/abs/2411.06919)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문에서는 양자 기계 학습(Quantum Machine Learning, QML) 모델을 위한 새로운 여유 기반 일반화 경계를 제시합니다. 이 경계는 현재의 일반화 이론의 한계를 극복하는 데 도움을 주며, 실험을 통해 여유 기반 메트릭(margin-based metrics)이 전통적 메트릭보다 일반화 성능을 더 잘 예측함을 입증하였습니다.

- **Technical Details**: 여유 기반 일반화 경계는 다중 클래스 분류 문제에 적합한 QNN(Quantum Neural Nets) 모델의 접근법을 사용하여, 벗어비스 넷을 위한 기법을 양자 도메인에 적응시키는 내용을 담고 있습니다. 이 과정에서 양자 측정을 립시츠 연속 비선형 활성화로 해석하고, 복소수 값 공간에 대한 매트릭 커버링 기법을 확장합니다. 또한, 양자 회로의 유니타리 제약과 양자 상태의 정규화는 실제 프레임워크의 복잡성을 단순화하는 데 기여합니다.

- **Performance Highlights**: 우리는 다양한 실험에서 여유 분포(margin distribution)와 일반화 성능 간의 강한 상관관계를 입증하였으며, 이는 전통적인 파라미터 기반 메트릭이 비효율적일 때도 유효합니다. 특히, Neural Quantum Embedding(NQE) 접근법이 큰 여유 값을 생성하고 일반화 성능을 향상시키는 것으로 나타났습니다.



### Gaussian Process Emulators for Few-Shot Segmentation in Cardiac MRI (https://arxiv.org/abs/2411.06911)
Comments:
          Submitted to Statistical Atlases and Computational Modeling of the Heart (STACOM) 2024

- **What's New**: 이 논문에서는 cardiac MRI(심장 자기공명영상)의 세분화를 개선하기 위해 few-shot learning을 U-Net 아키텍처 및 Gaussian Process Emulators(GPEs)와 결합한 새로운 방법을 제안합니다. 이 방법은 데이터 적재를 최소화하면서도, 적은 양의 라벨이 있는 지원 집합으로부터 더 나은 성능을 발휘할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 U-Net의 수축 부분을 이용하여 query 이미지와 지원 이미지의 관계를 잠재 공간(latent space)에서 GPEs가 학습하도록 하며, 이 정보는 확장 부분으로 통합되어 query 이미지의 세분화를 보다 정확하게 수행합니다. 또한, M&Ms-2 공공 데이터셋을 사용하여 다양한 각도에서 심장을 세분화하는 능력을 평가했습니다. 이 모델은 기존의 무감독 및 few-shot 방법과 비교하여 더 높은 DICE 계수를 기록했습니다.

- **Performance Highlights**: 모델의 성능은 심장 자기공명영상에서 작은 지원 집합의 크기로 인해 특히 도전적인 설정에서 다른 방법들과 비교할 때 현저하게 개선되었으며, 더 나은 일반화 능력을 보였습니다.



### LongSafetyBench: Long-Context LLMs Struggle with Safety Issues (https://arxiv.org/abs/2411.06899)
- **What's New**: 이 논문에서는 장기 맥락 언어 모델(long-context language models)의 안전성을 평가하기 위한 최초의 기준인 LongSafetyBench를 소개합니다. 이 기준은 10개의 작업 범주로 구성되어 있으며, 평균 41,889 단어의 길이를 가집니다. 최근까지의 연구들은 주로 모델의 기능에 초점을 맞췄지만, 이 논문은 안전성에 대한 객관적이고 포괄적인 평가를 다룹니다.

- **Technical Details**: LongSafetyBench는 모델의 안전성 문제를 해결하기 위해 불법 활동(Illegal Activities), 잘못된 정보 피해(Misinformation Harm), 공격성 및 편향(Offensiveness and Bias)과 같은 세 가지 유형의 위험한 시나리오를 대상으로 데이터를 수집하고 구성하였습니다. 각 문항은 다중 선택 질문 형식으로 포맷팅되었으며, 총 1,203개의 테스트 인스턴스가 포함되어 있습니다.

- **Performance Highlights**: 8개의 장기 맥락 LLM을 LongSafetyBench에서 테스트한 결과, 대부분의 주류 장기 맥락 LLM에서 안전한 응답의 비율이 50% 미만으로 나타났습니다. 장기 맥락 시나리오에서의 안전성과 단기 맥락에서의 성능 간의 일치성이 떨어지며, 모델이 긴 텍스트 내의 해로운 콘텐츠를 간과하는 경향이 있음을 확인했습니다. 또한, 적은 양의 데이터로 훈련한 오픈 소스 모델이 최상위 폐쇄형 모델과 유사한 성능을 달성할 수 있는 것으로 나타났습니다.



### CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models (https://arxiv.org/abs/2411.06869)
- **What's New**: CapeLLM은 카테고리 비의존적 포즈 추정(Category-Agnostic Pose Estimation, CAPE)을 위한 혁신적인 접근 방식으로, 기존의 지원 이미지 없이 쿼리 이미지(query image)와 상세한 텍스트 설명만을 이용하여 키포인트(keypoints)를 추정합니다.

- **Technical Details**: CapeLLM은 텍스트 기반 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 활용하여 CAPE에 적합하도록 설계되었습니다. 연구에서는 모델의 아키텍처와 입력으로 사용되는 지시(description) 설정을 세밀하게 조정하고, LLM이 텍스트를 이해하는 능력을 완전 활용하여 예상 입력에 대한 결과를 생성합니다.

- **Performance Highlights**: CapeLLM은 MP-100 벤치마크에서 1-shot 설정에서 기존 모델들의 5-shot 성능을 초과하는 뛰어난 성능을 시연하며, 카테고리 비의존적 포즈 추정 분야에서 새로운 최첨단 성과를 달성하였습니다.



### Effect sizes as a statistical feature-selector-based learning to detect breast cancer (https://arxiv.org/abs/2411.06868)
Comments:
          16 pages, 10 figures, 5 tables,2024 IEEE Biennial Congress of Argentina (ARGENCON)

- **What's New**: 이번 연구에서는 세포 핵 이미지에서 추출한 특징을 기반으로 하여 통계적 효과 크기(effect size) 측정을 사용한 학습 도구를 개발할 수 있는 가능성을 보여주는 알고리즘과 실험 결과를 발표하였습니다.

- **Technical Details**: 효과 크기(effect size)는 두 변수 사이의 관계 강도를 숫자로 측정하는 통계적 개념이며, 특징 선택(feature selection)은 학습 모델을 개선하기 위해 예측 변수의 서브셋만 선택하여 데이터 차원을 줄이는 데 사용됩니다. 본 연구에서는 SVM(classifier) 분류기와 선형 커널을 이용하여 데이터 차원을 줄이는 학습 도구를 개발하였습니다.

- **Performance Highlights**: SVM 분류기를 이용한 실험에서 90% 이상의 정확도(accuracy)를 달성하였으며, 이는 효과 크기가 특징 선택 방법의 기준에 부합함을 시사합니다.



### Fast and Efficient Transformer-based Method for Bird's Eye View Instance Prediction (https://arxiv.org/abs/2411.06851)
Comments:
          The article has been presented in the 27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024) on September, 2024. Number of pages: 6, Number of figures: 4

- **What's New**: 이 논문에서는 최신 BEV(instance segmentation, flow prediction 기반의) 구조를 제안하여 자율주행차량의 객체 예측 및 경로 예측의 정확성을 높입니다.

- **Technical Details**: 제안된 아키텍처는 'Lift, Splat, Shoot' 방식을 기반으로 하며, 이를 통해 다중 카메라에서 얻은 데이터를 효과적으로 처리합니다. 이 모델은 효율적인 transformer 기반의 구조로, 적은 수의 파라미터와 빠른 추론 시간을 자랑합니다.

- **Performance Highlights**: PyTorch 2.1을 사용하여 성능을 최적화하였으며, 이는 기존 SOTA 아키텍처에 비해 상당한 성능 향상을 이뤄냈습니다.



### 1-800-SHARED-TASKS @ NLU of Devanagari Script Languages: Detection of Language, Hate Speech, and Targets using LLMs (https://arxiv.org/abs/2411.06850)
Comments:
          13 pages, Submitted to CHIPSAL workshop @ COLING 2025

- **What's New**: 이번 연구에서는 CHiPSAL 2025의 언어 감지, 증오 발언 식별 및 목표 감지에 관한 시스템을 설명합니다. Devanagari 스크립트 언어에서의 자연어 이해(NLP) 문제를 해결하기 위해 MuRIL, IndicBERT, Gemma-2와 같은 대형 언어 모델을 활용했습니다.

- **Technical Details**: 각 서브태스크에 대해 다양한 멀티링구얼 모델을 파인 튜닝(fine-tune)하고 평가 단계에서 최상의 모델을 선택했습니다. 서브태스크 A는 5개 언어 중에서 언어를 식별하며, B는 텍스트에서 증오 발언을 감지하고, C는 증오 발언의 목표를 분류합니다. Focal loss를 사용하여 클래스 불균형 문제를 해결했습니다.

- **Performance Highlights**: 서브태스크 A에서 F1 점수 0.9980, B에서 0.7652, C에서 0.6804를 달성하여 강력한 성능을 나타냈습니다. 특히, Ensemble 기법을 통해 최종 테스트에서 서브태스크 A의 성능을 더욱 개선했습니다.



### LLM-Neo: Parameter Efficient Knowledge Distillation for Large Language Models (https://arxiv.org/abs/2411.06839)
Comments:
          ICASSP 25' under review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 소형 학생 모델로 효율적으로 지식을 전이하는 새로운 LLM-Neo 프레임워크를 제안합니다. 기존의 지식 증류(KD)와 저차원 적응(LoRA) 개념을 통합하여 지식 전이의 효율성을 개선합니다.

- **Technical Details**: LLM-Neo는 LoRA의 저차원 브랜치를 활용하여 효율적인 지식 증류를 달성합니다. KD 손실은 실제 레이블과 교차 엔트로피 손실, 그리고 교사 모델의 예측과의 Kullback-Leibler(KL) 발산을 결합하여 정의됩니다. 이 과정에서 LoRA의 파라미터 효율성을 유지할 수 있습니다.

- **Performance Highlights**: LLM-Neo는 Llama 2 및 Llama 3.1 모델을 압축하는 실험에서 다양한 기준 모델보다 우수한 성능을 보였습니다. 추가 분석을 통해 LoRA 변종에 대한 LLM-Neo의 강건성도 확인되었습니다.



### Learning Interpretable Network Dynamics via Universal Neural Symbolic Regression (https://arxiv.org/abs/2411.06833)
Comments:
          preprint

- **What's New**: 이 연구에서는 복잡한 네트워크 동역학의 지배 방정식을 자동으로 배우고, 효율적으로 식별할 수 있는 범용 계산 도구 LLC (Learning Law of Changes)를 개발했습니다. 이 도구는 심층 학습의 탁월한 적합 능력과 사전 훈련된 기호 회귀의 방정식 추론 능력을 결합하여 복잡한 시스템 상태의 기호 변화 패턴을 배웁니다.

- **Technical Details**: LLC는 네트워크 동역학의 관측 데이터를 통해 일반적인 미분 방정식 (Ordinary Differential Equations, ODEs)을 발견합니다. 관측 데이터는 연속 상태 데이터에서 추출되며, 초기 상태와 토폴로지를 기반으로 초기 값 문제를 해결하여 얻은 것입니다. 이 도구는 노이즈가 있거나 토폴로지가 결여된 데이터에서도 지배 방정식을 정확하고 효율적으로 발견할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: 이 도구의 효과성을 증명하기 위해 물리학, 생화학, 생태학, 전염병학 등 10개 이상의 대표적인 시나리오에서 실험을 수행했습니다. 그 결과, LLC는 혼돈 시스템 및 실제 시스템(예: 글로벌 전염병 전파 및 보행자 움직임)을 포함하여 뛰어난 효율성과 효과를 보여주었습니다.



### Optimized Quality of Service prediction in FSO Links over South Africa using Ensemble Learning (https://arxiv.org/abs/2411.06832)
- **What's New**: 이번 연구는 다양한 날씨 상황에서 자유 공간 광통신의 품질(QoS: Quality of Service)을 최적화하기 위해 앙상블 학습 모델을 활용한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구는 2010년부터 2019년까지 남아프리카 기상 서비스 아카이브에서 수집된 기상 데이터, 가시성, 풍속, 고도를 사용하여 수행되었습니다. 연구에서 사용된 앙상블 학습 모델은 Random Forest, ADaBoost Regression, Stacking Regression, Gradient Boost Regression, Multilayer Neural Network로 구성됩니다. 모델은 폴로크완(Polokwane), 킴벌리(Kimberley), 블룸폰테인(Bloemfontein), 조지(George) 등 4개 위치에서 데이터 전송 속도, 수신된 전력, 안개에 의한 감쇠, 비트 오류율, 전력 패널티 등을 추정했습니다.

- **Performance Highlights**: 모델의 RMSE와 R-squared 값은 각각 (폴로크완) 0.0073 및 0.9951, (킴벌리) 0.0065 및 0.9998, (블룸폰테인) 0.0060 및 0.9941, (조지) 0.0032 및 0.9906으로 나타났습니다. 이러한 결과는 앙상블 학습 기법이 전달 모델링에서 서비스 품질을 크게 향상시킬 수 있음을 보여주며, 신호 대 잡음 비율(signal to noise ratio)을 효과적으로 최적화하였음을 나타냅니다.



### Generative midtended cognition and Artificial Intelligence. Thinging with thinging things (https://arxiv.org/abs/2411.06812)
Comments:
          16 pages, 2 figures. Submitted to "Synthese" Journal, accepted

- **What's New**: 본 논문은 'generative midtended cognition'이라는 개념을 소개하며, 생성적 AI와 인간의 인지 통합을 탐구합니다. 기존의 인지 이론에서 간과된 한계를 극복하기 위한 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 이 논문에서는 multimodal transformer architectures에 기반한 현재의 생성 기술을 검토하고, ChatGPT와 같은 대형 언어 모델이 인간의 인지 작용을 어떻게 전환시킬 수 있는지를 설명합니다. 'Generative midtended cognition'은 두 가지 차원, 즉 Width(맥락의 민감성)와 Depth(반복 루프의 세분화)를 포함합니다.

- **Performance Highlights**: 생성적 AI의 보편화는 다양한 창의적 출력의 질에 유의미한 변화를 가져왔으며, 이는 전통적인 인지 및 의도적 창조의 개념을 재정의하는 데 기여하고 있습니다. 그러나 이러한 과정은 정체성과 창의성의 본질에 대한 새로운 도전과제를 제기하고 있습니다.



### Predicting ionic conductivity in solids from the machine-learned potential energy landscap (https://arxiv.org/abs/2411.06804)
- **What's New**: 본 논문에서는 새로운 초이온성 (superionic) 물질을 빠르고 신뢰성 있게 평가하기 위해, 범용 원자 간 포텐셜 (interatomic potential) 모델을 분석한 방법론을 제안합니다. 이 방법은 최소한의 일반화 (generalization) 능력을 요구하면서 기본 모델의 풍부한 지식을 효과적으로 활용하는 구조 설명자를 포함합니다.

- **Technical Details**: 제안된 방법은 고급 기계 학습 (Machine Learning) 알고리즘, 특히 그래프 신경망 (graph neural networks; GNN)을 기반으로 하며, 이는 고품질 데이터 세트에서 기초 원리에 따른 힘과 에너지 계산의 확장을 통해 훈련됩니다. 이러한 일반화된 포텐셜 모델을 이용하여 리튬이 포함된 물질을 예상 전도도 (ionic conductivity)에 따라 순위 매깁니다.

- **Performance Highlights**: 제안된 방법은 전통적인 분자 동역학 (molecular dynamics) 방법보다 약 50배 빠르며, 첫 번째 원리 분자 동역학(AIMD)보다 최소 3000배 빠른 성능을 보여 주었습니다. 이 작업으로 인해 연구자들은 새로운 고성능 고체 전해질 자재 발견 속도를 가속화할 수 있습니다.



### ScaleKD: Strong Vision Transformers Could Be Excellent Teachers (https://arxiv.org/abs/2411.06786)
Comments:
          This work is accepted to NeurIPS 2024. The project page: this https URL

- **What's New**: 이 논문에서는 잘 정밀 훈련된 비전 트랜스포머 (ViT) 모델이 교사 역할을 하여 크로스 아키텍처 지식 증류 (Knowledge Distillation, KD) 연구를 진전시키는 데 활용될 수 있는지를 질문합니다.

- **Technical Details**: 우리는 ScaleKD라는 새로운 KD 방법을 제안합니다. 이 방법은 세 가지 상호 연관된 구성요소인 cross attention projector (CAP), dual-view feature mimicking, teacher parameter perception을 결합하여, CNN, MLP 및 ViT 아키텍처 전반에 걸쳐 활용될 수 있는 효과적인 지식 전이 방법을 제공합니다.

- **Performance Highlights**: ScaleKD는 ImageNet-1K 데이터셋에서 MobileNet-V1, ResNet-50, ViT-S/16과 같은 다양한 모델들에 대해 각각의 개별 훈련 모델 대비 2%에서 5%까지 절대 정확도 향상을 보여주며, 다운스트림 MS-COCO 및 ADE20K 데이터셋에서도 우수한 성능을 발휘합니다.



### QuadWBG: Generalizable Quadrupedal Whole-Body Grasping (https://arxiv.org/abs/2411.06782)
- **What's New**: 이번 연구에서는 로봇 팔에 장착된 카메라 하나를 기반으로 강력하고 범용적인 전체 몸체의 loco-manipulation (로코-조작) 제어기를 위한 모듈형 프레임워크를 제시합니다. 이는 로봇의 움직임을 조정하고 다양한 개체를 조작하는 데 필요한 통합된 제어 방안을 제공합니다.

- **Technical Details**: 제안된 시스템은 일반화된 지향 도달 가능성 맵(Generalized Oriented Reachability Map, GORM)을 사용하여 5차원(5D) 명령을 관리할 수 있는 저수준 정책과 그립 감지 인식을 위한 고수준 정책을 조화롭게 구성합니다. 이러한 시스템은 로봇의 전체 이동성과 조작성을 향상시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 시스템은 실제 환경에서 89%의 일회성 그립 정확도를 달성했습니다. 또한, 다양한 전체 몸체 loco-manipulation 작업을 수행할 수 있는 역량을 보여주며, 투명한 물체와 같은 도전적인 작업에서도 우수한 성과를 기록했습니다.



### MP-PINN: A Multi-Phase Physics-Informed Neural Network for Epidemic Forecasting (https://arxiv.org/abs/2411.06781)
- **What's New**: 이 논문에서는 COVID-19와 같은 전염병의 확산 예측을 위해 새로운 하이브리드 방법론인 MP-PINN(Multi-Phase Physics-Informed Neural Network)을 제안합니다. 이 방법은 전통적인 기계적 모델과 데이터 기반 방법의 한계를 극복하는 것을 목표로 하고 있습니다.

- **Technical Details**: MP-PINN은 신경망에 전파 메커니즘을 주입하여 정책 개입에 따른 전염병의 역학 변화를 단계적으로 반영합니다. 이를 통해 단기 및 장기 예측에서 더 나은 성능을 제공합니다. SIR(Susceptible-Infectious-Recovered) 모델의 여러 단계에 걸쳐 각기 다른 SIR 매개변수를 사용할 수 있도록 하여 적응성을 추가했습니다.

- **Performance Highlights**: MP-PINN은 이탈리아의 COVID-19 데이터 세트를 기반으로 한 실험에서 전통적인 SIR 모델, 순수 데이터 기반 접근 방식, 단일 단계 PINN보다 단기 및 장기 예측 모두에서 우수한 성능을 보였습니다.



### PDC & DM-SFT: A Road for LLM SQL Bug-Fix Enhancing (https://arxiv.org/abs/2411.06767)
Comments:
          COLING-Industry 2025 accepted

- **What's New**: 이번 연구에서는 SQL의 버그 수정을 위한 새로운 방법론을 제시합니다. 기존의 Code LLM과 달리, 이 모델은 버그 수리 능력을 향상시키기 위해 Progressive Dataset Construction (PDC)와 Dynamic Mask Supervised Fine-tuning (DM-SFT) 방법을 도입합니다.

- **Technical Details**: PDC는 폭넓고 깊이 있는 두 가지 데이터 확장 방법을 포함하며, DM-SFT는 효과적인 수퍼바이즈드( Supervised) 학습 접근법을 통해 SQL 코드 버그 수정 과정에서의 학습 단계를 줄이고 안정성을 강화합니다. 특히, DM-SFT는 SQL 코드를 다루는 훈련 난이도를 줄이는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, PDC와 DM-SFT를 통해 훈련된 모델은 현재까지의 최고 모델들과 비교했을 때 성능이 50% 이상 향상되었으며, 일반적인 생성적 SFT에 비해 약 10%의 추가적인 성능 향상을 보여주었습니다.



### Multi-Stage Knowledge Integration of Vision-Language Models for Continual Learning (https://arxiv.org/abs/2411.06764)
- **What's New**: 본 논문은 Vision Language Model(VLM)에 대한 지속적인 학습을 목표로 하고 있으며, Knowledge Integration Theory(KIT)를 바탕으로 한 Multi-Stage Knowledge Integration network(MulKI)를 제안합니다. 이 네트워크는 인간의 학습 프로세스를 모방하여 VLM이 새로운 데이터 분포에 효과적으로 적응할 수 있도록 돕습니다.

- **Technical Details**: MulKI는 Eliciting Ideas, Adding New Ideas, Distinguishing Ideas, Making Connections의 네 단계로 구성됩니다. 각 단계에서는 프로토타입을 활용하여 다양한 모달리티 간의 정렬을 수행하고, 두 개의 Teacher 모델로부터 지식을 적응적으로 구별하고 조정합니다. 이 방법은 추가 데이터나 이전 작업의 프로토타입 보존 없이도 VLM을 지속적으로 학습할 수 있게 합니다.

- **Performance Highlights**: MulKI는 다양한 하위 작업에서 지속적인 학습을 지원하며, 제로샷(zero-shot) 기능을 유지하는 데 있어 상당한 개선을 보여줍니다. 성능 평가에서 기존의 방법들보다 우수한 결과를 나타내며, VLM이 변화하는 데이터 분포에 적응할 수 있는 잠재력을 입증합니다.



### Precision Glass Thermoforming Assisted by Neural Networks (https://arxiv.org/abs/2411.06762)
- **What's New**: 이번 연구에서는 곡면 프로파일(curve profiles)의 정밀도가 점점 더 요구되는 유리 성형 공정에서 효율적인 예측 모델을 개발했습니다. 전통적인 방법의 비효율성을 극복하기 위해 비차원 역전파 신경망(BPNN)을 활용하여 몰드 설계에서의 오류를 보상하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서 제안된 BPNN은 form errors(형상 오류)를 정확히 예측하고, 이를 바탕으로 precision glass molding(정밀 유리 성형)을 위한 몰드 디자인을 개선합니다. 또한, AR/VR 유리 또는 스마트폰 커버 유리와 같은 대량 생산 요구를 감안할 때, 몰드 가공 정밀도가 상대적으로 낮을 수 있는 문제도 논의됩니다.

- **Performance Highlights**: 개발된 BPNN은 산업 데이터를 기반으로 훈련되어 유리 제조 공정에 도입될 수 있으며, 비용이 많이 드는 시뮬레이션이나 실험을 대체할 수 있는 가능성을 제시합니다.



### Methane projections from Canada's oil sands tailings using scientific deep learning reveal significant underestimation (https://arxiv.org/abs/2411.06741)
Comments:
          19 pages, 8 figures, 2 tables

- **What's New**: 본 연구는 캘리포니아 아사바스카 오일 샌드 산업에서 비트멘 추출이 온실가스 배출의 주요 원인으로 지목됨을 강조합니다. 특히 점토 튜브(비축 물질)에서 미생물 분해로 인해 발생하는 메탄(CH4) 가스의 방출 가능성을 기계 학습 모델을 통해 예측합니다.

- **Technical Details**: 이 연구는 기상 데이터, 실험실 모델, 산업 보고서를 종합하여 물리적 제약(physics constrained) 기계 학습 모델을 훈련합니다. 이 모델은 메탄 방출 수준을 예측하고, 다양한 조건을 고려하여 데이터의 비선형 관계를 학습합니다.

- **Performance Highlights**: 모델을 통해 각 활성 점토 튜브가 연간 950~1500 톤의 메탄을 방출할 수 있고, 이는 최소 6000 대의 휘발유 차량이 배출하는 이산화탄소와 동등한 환경적 영향을 미친다고 분석하였습니다. 활성화된 포드의 배출량 감소를 위해 연간 약 12%의 감축이 필요하다고 추정하였습니다.



### Shallow Signed Distance Functions for Kinematic Collision Bodies (https://arxiv.org/abs/2411.06719)
Comments:
          Preprint

- **What's New**: 본 논문에서는 의류 시뮬레이션에서 발생하는 실시간 아바타 충돌 쿼리를 위한 학습 기반의 암시적 형태 표현(implicit shape representations)을 제안합니다. 우리는 전통적인 표현 방식에 비해 메모리 요구 사항이 적은 여러 형상을 표현할 수 있는 깊은 신경망(Deep Neural Networks)을 사용합니다.

- **Technical Details**: 우리는 인간 아바타의 SDF( signed distance functions) 표현을 설계하며, 특정 관절의 변화로 인한 형태 변형을 표현하기 위해 매우 효율적인 얕은 신경망(shallow neural networks)의 집합을 사용합니다. 각 얕은 SDF는 전신의 경계에 대한 거리를 효율적으로 표현하기 위해 스티칭(stitching) 처리 과정을 통해 결합됩니다.

- **Performance Highlights**: 제안된 모델은 매우 빠르고 정확하게 작동하며, 애니메이션 캐릭터에 의해 구동되는 의류의 실시간 시뮬레이션에서 그 적용 가능성을 보여줍니다.



### Truth, beauty, and goodness in grand unification: a machine learning approach (https://arxiv.org/abs/2411.06718)
Comments:
          8 pages, 4 figures

- **What's New**: 본 논문은 머신러닝 기술을 활용하여 초대칭(Supersymmetric) $SU(5)$ 대통일 이론(Grand Unified Theory, GUT) 모델의 맛(sector of flavour)을 조사합니다. 이를 통해 기존 모델의 모습을 개선하려는 연구를 소개합니다.

- **Technical Details**: 최소 $SU(5)$ 모델은 자연에서 관측된 페르미온 질량(fermion masses)과 일치하지 않는다는 점이 알려져 있습니다. 이 문제를 해결하기 위한 두 가지 접근 방식: 45-표현(45-representation) 힉스 필드 및 24-표현 GUT 힉스 필드를 활용한 고차원(operator) 연산자를 비교합니다. 우리는 손실 함수(loss function)를 수치적으로 최적화하여 질량 행렬(mass matrices)의 결정식(determinants) 비율로 정의합니다.

- **Performance Highlights**: 결과적으로, 24-Higgs 접근 방식이 최소 $SU(5)$ 모델의 원래 구조를 크게 변경하지 않으면서 관측된 페르미온 질량을 달성함을 보여주었습니다.



### DiffSR: Learning Radar Reflectivity Synthesis via Diffusion Model from Satellite Observations (https://arxiv.org/abs/2411.06714)
- **What's New**: 이 논문에서는 Weather radar 데이터의 합성을 위해 DiffSR이라 불리는 새로운 두 단계의 diffusion-based 방법을 제안합니다. 기존의 MSE (Mean Squared Error) 손실을 사용하는 재구성 방식에서 발생하는 과도한 평활화 문제를 해결하고자 합니다.

- **Technical Details**: DiffSR 방법은 첫 번째 단계에서 글로벌 데이터에 대한 재구성 모델을 사전 훈련(pre-training)하여 레이더 추정을 수행한 후, 두 번째 단계에서 레이더 추정 결과를 위성 데이터(Satellite data)와 결합하여 diffusion 모델의 조건으로 사용합니다.

- **Performance Highlights**: 다양한 실험 결과를 통해, 제안된 방법이 최신 기술(SOTA) 결과를 달성했음을 보여주며, 이는 고주파 세부사항 및 고값 관측 영역을 생성하는 능력을 입증합니다.



### Bridge: A Unified Framework to Knowledge Graph Completion via Language Models and Knowledge Representation (https://arxiv.org/abs/2411.06660)
- **What's New**: 이번 연구에서는 Knowledge Graph Completion (KGC) 분야의 한계를 극복하기 위해 Bridge라는 새로운 프레임워크를 제안하였습니다. 이 프레임워크는 구조적 정보와 의미적 정보를 동시에 인코딩하여 KGC 성능을 향상시킵니다.

- **Technical Details**: Bridge는 PLM (Pre-trained Language Model)을 활용하여 KGs (Knowledge Graphs)의 구조적(spatial) 및 의미적(semantic) 정보를 별도로 인코딩합니다. 이를 통해 PLM의 의미적 지식을 더 잘 활용하고 구조적 표현 학습(structured representation learning) 원리를 적용할 수 있습니다. 또한, BYOL (Bootstrap Your Own Latent) 방법을 사용하여 트리플의 두 가지 다른 뷰를 통해 PLM을 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: Bridge는 세 가지 벤치마크 데이터 세트에서 SOTA (State Of The Art) 모델을 능가하는 결과를 보였습니다. 실험 결과, Bridge 프레임워크가 기존의 다른 방법들에 비해 일관되게 우수한 성능을 나타냈습니다.



### Renaissance: Investigating the Pretraining of Vision-Language Encoders (https://arxiv.org/abs/2411.06657)
- **What's New**: 최근 비전-언어 (Vision-Language) 과제를 위한 모델들이 급격히 증가하며, 이와 관련된 모델 디자인 및 훈련의 모범 사례에 대한 질문들이 여전히 남아있습니다. 본 논문에서는 비전-언어 인코더의 사전 훈련에 관한 질문에 답하고, 두 개의 주요 실험을 통해 가시적인 성능 저하 없이 계산 비용을 크게 절감할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 'Renaissance'라는 비전-언어 모델링 플랫폼을 소개하며, 대규모 컴퓨팅 자원을 절약하기 위해 사전 훈련 중 모델의 여러 부분을 고정(freezing)하는 실험을 수행하였습니다. 첫 번째 실험에서는 두 개의 모듈이 모두 고정된 경우 일부 성능 저하가 발생했지만, 시각 모듈을 고정시켰을 때는 성능이 증가하는 경향이 나타났습니다. 두 번째 실험에서는 텍스트 인코더와 비전 인코더의 사전 훈련 성능을 비교하였습니다.

- **Performance Highlights**: 결과적으로, 한 개의 타워 인코더 모델을 훈련할 때는 사전 훈련된 가중치보다 무작위 초기화(randomly initialized)가 더 나은 성능을 보이는 것으로 나타났습니다. Renaissance 플랫폼은 다양한 비전-언어 모델 유형을 평가하는 데 유용하며, 향후 연구에서 보다 많은 VL 모델에 대한 지원이 필요합니다.



### Quantum Policy Gradient in Reproducing Kernel Hilbert Spac (https://arxiv.org/abs/2411.06650)
- **What's New**: 이 논문은 양자 강화 학습 (quantum reinforcement learning) 맥락에서 파라미터화된 양자 회로 (parametrised quantum circuits)를 활용한 새로운 정책 경량화 (policy gradient) 및 액터-크리틱 (actor-critic) 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 양자 커널 (quantum kernel) 정책을 기반으로 하며, 이는 고차원 복소 힐버트 공간 (complex Hilbert space)에서의 양자 상태를 고려해 설계되었습니다. numercial 및 analytical 양자 정책 경량화 기법을 통해 구현되며, 정책의 그래디언트 (gradient)에 대한 해석 가능한 형태와 조정 가능한 표현력을 활용할 수 있습니다.

- **Performance Highlights**: 이 접근법은 벡터 값 동작 공간 (vector-valued action spaces)에 적합하며, 모든 제안된 포뮬레이션은 클래스 하드웨어 (classical counterparts) 대비 쿼리 복잡성 (query complexity)을 제곱으로 감소시킵니다. 특히, 확률적 정책 경량화 (stochastic policy gradient)와 결정론적 정책 경량화 (deterministic policy gradient)에 기반한 두 가지 액터-크리틱 알고리즘은 유리한 조건에서 추가적인 쿼리 복잡성 감소를 보여줍니다.



### A Novel Combined Data-Driven Approach for Electricity Theft Detection (https://arxiv.org/abs/2411.06649)
Comments:
          Paper accepted for IEEE Transactions on Industrial Informatics. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses

- **What's New**: 본 논문에서는 에너지 인터넷(Energy Internet)에서의 전기 절도 탐지를 위해 두 가지 혁신적인 데이터 마이닝 기법인 최대 정보 계수(Maximum Information Coefficient, MIC)와 밀도 피크의 Fast Search and Find 기법(CFSFDP)을 결합하여 제안합니다. 이 방법은 비정상적인 부하 프로파일을 찾아냄으로써 전기 절도를 보다 효과적으로 탐지합니다.

- **Technical Details**: 1. 최대 정보 계수(MIC): 비기술적 손실(Non-technical loss, NTL)과 소비자의 특정 전기 행동 간의 상관관계를 찾는 데 사용됩니다. 이는 일반적으로 정상으로 보이는 절도를 정밀하게 탐지할 수 있습니다. 2. CFSFDP: 수천 개의 부하 프로파일 중 비정상적인 사용자들을 찾아내는 클러스터링 기법으로, 전기 절도 탐지에 적합합니다. 이 두 방법을 결합하여 에너지 인터넷의 정보 흐름을 활용한 새로운 탐지 프레임워크를 제안합니다.

- **Performance Highlights**: 아일랜드의 스마트 미터 데이터셋을 활용한 수치적 실험 결과, 제안한 결합 방법의 우수한 성능이 확인되었습니다. 이는 기존의 절도 탐지 방법들보다 신뢰성과 정확성을 높이는 가능성을 제공합니다.



### Few-shot Semantic Learning for Robust Multi-Biome 3D Semantic Mapping in Off-Road Environments (https://arxiv.org/abs/2411.06632)
Comments:
          Accepted to Australasian Conference on Robotics and Automation (ACRA 2024)

- **What's New**: 이 연구에서는 언구조적 지형과 악화된 센싱 조건에서 고속 자율 네비게이션을 위한 새로운 접근 방식을 제안합니다. 사전 훈련된 Vision Transformer (ViT)를 사용하여 소량의 다중 생물군 데이터셋(500 이미지 미만)에서 2D semantic segmentation을 예측합니다.

- **Technical Details**: 우리는 감지된 클래스를 시간에 따라 집계하여 3D semantic voxel 맵으로 결합합니다. 이 과정에서 novel range-based metric을 활용하여 semantic 정보를 통합합니다. 또한, Yamaha와 Rellis 데이터 세트를 사용하여 zero-shot 및 few-shot 학습을 이용한 segmentation을 수행합니다.

- **Performance Highlights**: Yamaha 데이터 세트에서 52.9 mIoU, Rellis 데이터 세트에서 55.5 mIoU를 달성하였으며, few-shot sparse labeling을 통해 Yamaha에서 66.6 mIoU, Rellis에서 67.2 mIoU로 성능을 개선했습니다. 이 기법을 활용하여 오프로드 환경의 위험 요소를 효과적으로 처리할 수 있는 가능성을 보여줍니다.



### Exploring social bots: A feature-based approach to improve bot detection in social networks (https://arxiv.org/abs/2411.06626)
- **What's New**: 이 논문은 소셜 미디어에서의 허위 정보와 악성 링크의 확산 문제를 다루며, 자동화된 계정(봇)을 탐지하는 데 필요한 사용자 계정 프로필 및 콘텐츠 기반의 특징(features)을 조사합니다.

- **Technical Details**: 연구팀은 기계 학습 알고리즘(classical machine learning algorithms)을 사용하여 여러 메트릭(metric)에서 최신 기술(state of the art)을 초월하는 것을 목표로 하였으며, 특징 선택(feature selection) 및 추론(inference)을 통해 자동화된 계정 탐지에 유용한 특징들을 찾아냈습니다.

- **Performance Highlights**: 이 논문은 자동화된 계정에 대해 가장 중요한 특징들을 식별하며, 기존의 방법들을 뛰어넘는 성능을 보여주고 있습니다.



### OffLight: An Offline Multi-Agent Reinforcement Learning Framework for Traffic Signal Contro (https://arxiv.org/abs/2411.06601)
- **What's New**: 이 논문에서는 교통 신호 제어(traffic signal control, TSC)를 위해 고안된 OffLight라는 새로운 오프라인 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL) 프레임워크를 소개합니다. 이 프레임워크는 Importance Sampling(IS)과 Return-Based Prioritized Sampling(RBPS)을 결합하여 정책의 이질성을 다룹니다.

- **Technical Details**: OffLight는 GMM-VGAE 모델을 사용하여 실제 교통 네트워크 내에서의 다양한 행동 정책의 이질성을 포착합니다. 또한, Graph Neural Networks(GNNs)를 활용하여 교차점 간의 지역 관측을 구조화된 글로벌 표현으로 통합합니다. 이를 통해 OffLight는 IS를 효과적으로 활용하여 배포 변화(distributional shifts)를 교정하고 다양한 데이터로부터 안정적인 정책 학습을 보장합니다.

- **Performance Highlights**: OffLight는 실제 교통 데이터세트에서 기존 방법보다 우수한 성능을 보임을 입증하였으며, 샘플 효율성과 정책 성능을 향상시킵니다. 이는 특히 고수익 에피소드를 우선적으로 학습하면서 수렴을 가속화하는 데 기여합니다.



### Few measurement shots challenge generalization in learning to classify entanglemen (https://arxiv.org/abs/2411.06600)
- **What's New**: 이 논문은 양자 학습(quantum learning)에서의 문제를 다루며, 고전적 기계 학습(classical machine-learning) 방법과 양자 알고리즘(quantum algorithms)의 하이브리드 접근 방식의 효과를 탐구합니다.

- **Technical Details**: 측정의 불확실성이 에러의 주요 원인이 될 수 있음을 보여주고, 최대 엉킴 상태(maximally entangled states)와 분리 가능한 상태(separable states)의 분류를 통해 이를 설명합니다. 고전적 그림자(classical shadows)를 기반으로 한 새로운 추정기를 도입하고 있습니다.

- **Performance Highlights**: 실험 결과, 고전적 기계 학습 방법의 단순한 적용이 문제를 일으킬 수 있으며, 양자 학습의 이론적 기초를 강화해야 한다는 점을 강조합니다. 해석의 정확도가 훈련 데이터 수와 샷 수가 증가할수록 향상되는 경향을 보입니다.



### Probabilistic Consensus through Ensemble Validation: A Framework for LLM Reliability (https://arxiv.org/abs/2411.06535)
Comments:
          8 pages, 6 tables

- **What's New**: 대형 언어 모델(Large Language Models, LLMs)이 텍스트 생성 분야에서 큰 발전을 이루었으나, 의료, 법률, 금융과 같은 고위험 분야에서의 자율 배치에 필요한 신뢰성은 부족합니다. 기존 방법들은 외부 지식이나 인간의 감독에 의존하여 확장성에 한계가 있습니다. 이번 논문에서는 모델 합의를 통한 콘텐츠 검증을 위해 앙상블 방법(ensemble methods)을 재사용하는 새로운 프레임워크를 소개합니다.

- **Technical Details**: 78개의 복잡한 사례에서 사실 정확성(factual accuracy)과 인과 일관성(causal consistency)을 요구하는 테스트에서, 제안된 프레임워크는 두 개 모델을 사용할 경우 정확도를 73.1%에서 93.9%로 향상시켰고(95% CI: 83.5%-97.9%), 세 개 모델을 사용할 경우 95.6%로 향상시켰습니다(95% CI: 85.2%-98.8%). 통계 분석을 통해 강한 모델 간 합의(kappa > 0.76)를 보이며, 오류를 발견하기 위한 충분한 독립성도 유지합니다.

- **Performance Highlights**: 추가적인 검증자(validator)와 정교화를 통해 정확도를 더욱 향상시킬 수 있는 명확한 경로를 제시합니다. 현재 접근법은 다중 선택 형식 요구사항과 처리 지연(processing latency)으로 인해 제약이 있지만, 중요한 응용 분야에서 신뢰할 수 있는 자율 AI 시스템을 가능하게 하는 즉각적인 가치를 제공합니다.



### Offline Handwritten Signature Verification Using a Stream-Based Approach (https://arxiv.org/abs/2411.06510)
Comments:
          Accepted for oral presentation at the International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문에서는 손글씨 서명 검증(HSV) 시스템의 새로운 접근법을 제안합니다. 기존의 정적인 배치 구성 대신, 동적인 데이터 스트림 환경에서 작동할 수 있는 적응형 시스템을 개발하여 서명 검증의 성능을 향상시킵니다.

- **Technical Details**: 제안된 시스템은 SigNet-S 모델을 사용하여 서명 이미지를 특징 벡터로 변환하고, 이를 기반으로 디체미니당 변환(Dichotomy Transformation, DT)을 통해 이진 분류 문제로 변환합니다. 이 시스템은 새로운 서명 샘플이 입력될 때마다 업데이트되고, 과거의 데이터와 결합하여 계속해서 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Support Vector Machine(SVM)을 이용한 표준 접근 방식과 비교했을 때 우수한 성능을 보였습니다. GPDS Synthetic, CEDAR, MCYT 데이터셋에서의 실험 결과는 이 접근법의 효과성을 입증합니다.



### VocalTweets: Investigating Social Media Offensive Language Among Nigerian Musicians (https://arxiv.org/abs/2411.06477)
Comments:
          13 pages, 5 figures, 6 tables

- **What's New**: 이번 연구에서는 나이지리아의 유명한 음악가 12명의 트윗(tweets)을 포함한 VocalTweets라는 언어 혼합(code-switched) 및 다국어(multilingual) 데이터셋을 소개합니다.

- **Technical Details**: VocalTweets 데이터셋은 정규(Normal) 또는 공격적(Offensive)으로 이항 분류(binary classification)된 트윗으로 구성되어 있으며, HuggingFace의 base-Twitter-RoBERTa 모델을 사용하여 훈련되었습니다.

- **Performance Highlights**: 모델은 F1 점수(F1 score) 74.5를 달성했으며, OLID 데이터셋과의 교차 코퍼스(cross-corpus) 실험을 진행하여 데이터셋의 일반화 가능성(generality)을 평가했습니다.



### Multi-Parameter Molecular MRI Quantification using Physics-Informed Self-Supervised Learning (https://arxiv.org/abs/2411.06447)
Comments:
          This project was funded by the European Union (ERC, BabyMagnet, project no. 101115639), the Ministry of Innovation, Science and Technology, Israel, and a grant from the Tel Aviv University Center for AI and Data Science (TAD, The Blavatnik AI and Data Science Fund). None of above can be held responsible for views and opinions expressed, which are those of the authors alone

- **What's New**: 이번 연구에서는 물리 기반의 딥러닝 프레임워크를 통해 인체 뇌 조직의 프로톤 스핀 특성에 대한 모델 적합을 신속하게 수행하는 방법을 제안합니다. 특히, 다중 프로톤 풀 및 비정상 상태의 MRF(자기 공명 지문화) 획득을 포함한 CEST 이미징 시나리오에 중점을 두었습니다.

- **Technical Details**: 제안된 시스템은 블로흐-맥콜 공식을 기반으로 한 보통 미분 방정식(ODE)을 효과적으로 해결하고 역으로 변환합니다. 이 과정에서 새롭게 개발된 '신경 블로흐 맥콜 적합(Neural Bloch McConnell Fitting, NBMF)' 기법을 사용하여, 기존 MRF 방법의 필요 없었던 사전 훈련 데이터셋 생성 문제를 해결합니다. 이 네트워크는 한 사람의 데이터를 바탕으로 자기 지도식 학습(Self-supervised Learning)을 통해 훈련됩니다.

- **Performance Highlights**: NBMF 파이프라인을 통해 삼성전자의 3T 임상 스캐너에서 L-아르기닌 농도의 높은 일치를 확인하였고, Pearson 상관계수는 0.986에 달했습니다. 또한, NBMF로 재구성한 프로톤 교환 속도는 기존의 MRF 딕셔너리 매칭 결과와 우수한 일치를 보였으며, Pearson 상관계수는 0.999에 이릅니다. 전체 뇌의 첫 번째 정량화 작업은 18.3±8.3분 만에 완료되었으며, 이는 유사한 대안들에 비해 수 배 더 빠른 결과입니다.



### Detecting AutoEncoder is Enough to Catch LDM Generated Images (https://arxiv.org/abs/2411.06441)
- **What's New**: 최근 몇 년간 diffusion models는 이미지 생성의 주요 방법론 중 하나로 자리잡았으나, 이들 모델이 생성한 이미지를 탐지하는 것은 여전히 어려운 과제로 남아있습니다. 본 논문은 Latent Diffusion Models (LDM)로 생성된 이미지를 탐지하기 위한 새로운 방법을 제안하는데, 이는 autoencoder에 의해 도입된 아티팩트를 식별하여 이루어집니다.

- **Technical Details**: 제안된 방법은 실제 이미지와 LDM autoencoder에 의해 재구성된 이미지를 구별하기 위해 탐지기를 훈련시키는 접근 방식을 이용합니다. 이 방법은 생성된 이미지를 직접 훈련하지 않고도 이미지를 탐지할 수 있게 하며, 다른 유사한 접근법과 비교해 학습에 소모되는 계산 비용을 크게 줄이고 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 본 방법은 허위 긍정(false positive)을 최소화하며 높은 탐지 정확도를 보여줍니다. 이는 생성된 이미지를 탐지하는 데 있어 유망한 도구가 될 것으로 기대됩니다.



### SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains (https://arxiv.org/abs/2411.06426)
- **What's New**: 본 연구에서는 SequentialBreak라는 새로운 jailbreak 공격 기법을 소개합니다. 이 기법은 LLM이 단일 쿼리 내에서 여러 프롬프트를 처리할 때 특정 유해한 프롬프트에 집중하도록 유도하는 구조를 활용합니다. 다양한 시나리오를 통해 유익한 프롬프트들 사이에 유해한 프롬프트를 은폐하여 LLM이 유해한 반응을 생성하도록 유도할 수 있습니다.

- **Technical Details**: SequentialBreak 공격은 단일 쿼리로 여러 프롬프트를 전송하며, 공격자는 benign 프롬프트 사이에 있던 유해한 프롬프트를 포함합니다. 이 공격은 black-box 접근만 필요하며, 다양한 프롬프트 서사 구조에 적응 가능합니다. 연구에서는 'Question Bank', 'Dialog Completion', 'Game Environment'라는 세 가지 공격 시나리오를 제시하며, 이 모두에서 고도의 공격 성공률을 보고합니다.

- **Performance Highlights**: SequentialBreak는 기존의 다양한 jailbreak 기법들보다 우수한 성능을 나타내었으며, 단일 쿼리만으로 여러 최신 LLM 모델에서 높은 공격 성공률을 달성했습니다. 또한, 기존 방어 메커니즘을 효과적으로 피할 수 있는 능력을 입증하며, 자원 효율성 또한 높습니다.



### Optimal Execution with Reinforcement Learning (https://arxiv.org/abs/2411.06389)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 트레이더들이 한정된 시간 내에 자산을 거래하기 위한 최적 실행 전략을 찾기 위해 강화학습( reinforcement learning )을 활용합니다. 제안된 모델은 현재의 리미트 오더 북( limit order book ) 상태에서 파생된 입력 특징을 이용합니다.

- **Technical Details**: 우리는 ABIDES라는 다중 에이전트 시장 시뮬레이터를 사용하여 이 환경을 시뮬레이션하고, 고전적인 데이터에 의존하는 제한을 극복합니다. 또한, 맞춤형 MDP( Markov Decision Process ) 수식을 제시하고, 이 방법론의 결과를 기존의 실행 전략과 비교하여 성능을 벤치마킹합니다.

- **Performance Highlights**: 연구 결과, 강화학습 기반 접근법이 상당한 잠재력을 보여주었습니다. 이는 대량 거래가 자산 가격에 미치는 영향을 효과적으로 최소화할 수 있는 새로운 방법으로 자리잡을 가능성을 보입니다.



### Metric Learning for Tag Recommendation: Tackling Data Sparsity and Cold Start Issues (https://arxiv.org/abs/2411.06374)
- **What's New**: 이 논문에서는 개인화된 추천 시스템의 한계를 극복하기 위해 메트릭 학습(metric learning)에 기반한 새로운 레이블 추천 알고리즘을 제안합니다. 이는 사용자의 선호와 아이템 특성 간의 미세한 차이를 포착하는 데 효과적인 거리 또는 유사성 메트릭을 학습합니다.

- **Technical Details**: 제안된 알고리즘은 전통적인 협업 필터링(collaborative filtering) 및 콘텐츠 기반 추천(content-based recommendation) 방법이 가진 데이터 희소성(data sparsity) 및 콜드 스타트(cold start) 문제를 해결하는 데 중점을 둡니다. 실험 결과에 따르면, 이 알고리즘은 지역 반응 메트릭 학습(local response metric learning, LRML), 협업 메트릭 학습(collaborative metric learning, CML), 적응형 텐서 분해(adaptive tensor factorization, ATF)와 같은 기존 방법보다 우수한 성능을 보입니다.

- **Performance Highlights**: 특히 제안된 알고리즘은 추천 항목의 초기 몇 개에서 높은 정확도를 달성하며, 강력한 내구성(robustness)을 유지하면서도 높은 추천 정확도를 지속적으로 보여줍니다.



### LLM Vocabulary Compression for Low-Compute Environments (https://arxiv.org/abs/2411.06371)
Comments:
          Machine Learning and Compression Workshop @ NeurIPS 2024

- **What's New**: 본 논문은 언어 모델의 최종 선형 층을 압축하는 방법을 제안하여 메모리 사용량을 최대 3.4배 줄이는 동시에 성능 저하를 최소화합니다. Byte Pair Encoding (BPE) 병합에 따라 토큰을 그룹화함으로써 메모리 집약적인 logits 텐서의 물화를 방지합니다.

- **Technical Details**: 우리는 언어 모델의 최종 임베딩 층의 메모리 발자국을 줄이는 방법으로, 토큰을 그룹화하고 최종 토큰을 예측하는 두 단계의 과정을 통해 어휘 층을 효과적으로 압축합니다. 새로운 접근 방식에서는 두 개의 개별 모델을 사용하는 대신, 숨겨진 상태를 기반으로 그룹 및 토큰 예측을 모두 학습할 수 있는 간단한 선형 층을 적용합니다.

- **Performance Highlights**: TinyStories 데이터셋에서의 평가 결과, 본 방법은 GPT-Neo와 GPT2와 동등한 성능을 보이며, 처리량을 최대 3배 향상시켜 저전력 환경에 적합하다는 것을 확인했습니다.



### Balancing Power and Ethics: A Framework for Addressing Human Rights Concerns in Military AI (https://arxiv.org/abs/2411.06336)
Comments:
          Accepted for oral (only 3 papers are selected!) Harms and Risks of AI in the Military Workshop (HRAIM 2024) at Mila Quebec (this https URL)

- **What's New**: 이번 논문은 군사 AI 디자인, 배포 및 사용에서 인권 문제를 평가하기 위한 새로운 3단계 프레임워크(Framework)를 제안합니다. 각 단계는 다양한 윤리적 및 법적 고려사항을 포함하여 AI 기술이 군사 작전에 미치는 영향을 균형 있게 다루고자 합니다.

- **Technical Details**: 이 프레임워크는 세 가지 주요 단계(Design, Deployment, Use)로 나뉘며, 각 단계는 알고리즘, 사후 사용의 책임, 공정성(fairness) 및 감시(surveillance)와 같은 여러 구성 요소를 포함합니다. 또한, 'Human-in-the-loop' 시스템(HITL)에 대한 인간 개입을 강조합니다.

- **Performance Highlights**: 논문에서는 군사 AI의 인권 침해 가능성과 이에 대한 해결 방안을 제시합니다. 또한, AI 시스템의 성능 모니터링과 지속적인 감사를 통해 공정성을 확보하고, 국제 인도법(International Humanitarian Law) 준수를 위한 강력한 윤리적 지침을 마련할 것을 강조합니다.



### A Learned Proximal Alternating Minimization Algorithm and Its Induced Network for a Class of Two-block Nonconvex and Nonsmooth Optimization (https://arxiv.org/abs/2411.06333)
- **What's New**: 이번 연구에서는 학습 가능한 두 블록 비부드럽고 비볼록 최적화 문제를 해결하기 위한 일반화된 학습된 Proximal Alternating Minimization 알고리즘(LPAM)을 제안합니다. LPAM은 자동 감소하는 smoothing 효과를 활용하여 비부드러운 문제를 다루며, Residual Learning 아키텍처를 통합하여 최적화합니다.

- **Technical Details**: 제안된 LPAM 방법은 Residual Learning 아키텍처와 Block Coordinate Descent(BCD) 반복을 통합하여 Proximal Alternating Linearized Minimization(PALM) 방식을 수정합니다. 알고리즘이 생성한 반복의 부분 수열은 최소한 하나의 집합 점(accumulation point)을 가지며, 각 집합 점은 Clarke Stationary Point입니다. LPAM-net 구조는 LPAM을 정확히 따르며, 이를 통해 알고리즘의 수렴 특성을 계승합니다.

- **Performance Highlights**: LPAM-net의 적용 예로서, 저샘플링된 k-space 데이터로부터의 Joint Multi-Modal MRI 재구성에 대한 수치적 및 이론적 결과를 제시합니다. 실험 결과, 제안된 LPAM-net은 효율적인 매개변수를 가지고 있으며, 여러 최신 방법들과 비교해 우수한 성능을 보였습니다.



### Emotion-Aware Interaction Design in Intelligent User Interface Using Multi-Modal Deep Learning (https://arxiv.org/abs/2411.06326)
- **What's New**: 이번 연구에서는 사용자 인터페이스 (UI) 디자인의 중요성을 강조하며, 감정 인식 시스템을 도입하여 UI의 감정적 반응성을 크게 향상시키고자 합니다.

- **Technical Details**: 다중 분기 Transformer 모델을 통해 얼굴 표정, 음성, 텍스트 데이터를 통합하여 복잡한 감정 신호를 실시간으로 해석합니다.

- **Performance Highlights**: 공개된 MELD 데이터셋을 활용한 검증 결과, 우리의 모델은 감정 인식 정확도와 F1 점수에서 전통적인 방법들을 크게 능가하는 성능 향상을 보여주었습니다.



### Amortized Bayesian Local Interpolation NetworK: Fast covariance parameter estimation for Gaussian Processes (https://arxiv.org/abs/2411.06324)
- **What's New**: 본 논문에서는 Amortized Bayesian Local Interpolation NetworK(A-BLINK)를 제안합니다. 이는 크기가 큰 공간 데이터셋에서도 빠르게 공분산 매개변수를 추정할 수 있도록 설계되었습니다. 두 개의 사전 훈련된 심층 신경망을 이용해 Kriging(weights)에 대한 매핑을 학습하여 행렬 역산을 우회할 수 있습니다.

- **Technical Details**: A-BLINK는 공간 위치 좌표와 공분산 함수 매개변수에서 각각 Kriging weights와 공간 분산으로의 매핑을 학습하는 두 개의 사전 훈련된 심층 신경망을 사용합니다. 이 접근법은 특히 대규모로 불규칙한 공간 데이터에 대해 Bayesian 프레임워크에서 전체 데이터를 재조정하지 않고도 파라미터 추정 및 예측을 가능하게 합니다. 이를 통해 기존의 방법들에 비해 큰 계산 속도 향상을 이루었습니다.

- **Performance Highlights**: 광범위한 시뮬레이션 연구에서 A-BLINK는 기존의 확장 가능한 가우시안 프로세스(GP) 방법론에 비해 유의미한 계산 효율성을 보여주었으며, 1991-2020년 동안 7,000개 이상의 기상 관측소에서의 온도 데이터셋을 사용하여 효과성을 입증하였습니다.



### NeuReg: Domain-invariant 3D Image Registration on Human and Mouse Brains (https://arxiv.org/abs/2411.06315)
Comments:
          15 pages, 5 figures, 5 tables

- **What's New**: 이번 연구에서는 NeuReg라는 Neuro-inspired 3D image registration 아키텍처를 제안합니다. 이 아키텍처는 domain invariance 기능을 가지고 있어 다양한 3D 뇌 이미징 모달리티 간의 변화를 잘 캡처할 수 있습니다.

- **Technical Details**: NeuReg는 Swin Transformer 블록을 encoder로 사용하여 domain-agnostic representations을 생성하며, 이를 통해 여러 뇌 이미징 도메인에 걸쳐 강력한 성능을 발휘합니다. 이 아키텍처는 mammalian visual system에서 영감을 얻어 개발되었습니다.

- **Performance Highlights**: NeuReg는 iSeg-2017 및 OASIS-3와 같은 다중 도메인 공개 데이터셋에서 기존의 baseline deep learning image registration 모델보다 우수한 성능을 보여줍니다. 특히, cross-domain 데이터셋에서 'source-only' 도메인에서 훈련된 모델이 'unseen' target 도메인에서 높은 성능을 기록했습니다.



### A Natural Primal-Dual Hybrid Gradient Method for Adversarial Neural Network Training on Solving Partial Differential Equations (https://arxiv.org/abs/2411.06278)
- **What's New**: 본 연구에서는 부분 미분 방정식(Partial Differential Equations, PDEs)을 해결하기 위해 확장 가능한 전처리된 프라이멀-듀얼 혼합 기울기 알고리즘을 제안합니다.

- **Technical Details**: PDE에 듀얼 테스트 함수를 곱하여 인프-서프(inf-sup) 문제를 도출하고, 손실 함수는 저차 미분 연산자를 포함합니다. PDHG(Primal-Dual Hybrid Gradient) 알고리즘을 사용하여 이 쐐기점 문제를 해결하며, 적절한 전처리 연산자를 도입하여 신경망 매개변수를 업데이트하는 자연 기울기 상승-하강 최적화 방식으로 변환합니다. Krylov 부분공간 방법(MINRES)을 적용하여 자연 기울기를 효율적으로 평가하고, 매트릭스-벡터 곱셈을 통해 전처리 행렬의 역행렬을 쉽게 처리합니다.

- **Performance Highlights**: 제안된 방법은 1차원에서 50차원까지 다양한 유형의 PDE에서 테스트 되었으며, 특히 선형 및 비선형 타원 방정식, 반응-확산 방정식 및 $L^2$ 최적 수송 문제에서 발생하는 Monge-Ampère 방정식까지 포함됩니다. 이 알고리즘은 PINNs, DeepRitz 방법, WANs 등 여러 일반적으로 사용되는 심층 학습 알고리즘과 비교되었으며, Adam과 L-BFGS 최적화기를 사용하여 PDE를 해결하는 데 효율적이고 안정적인 수렴 성능을 보였습니다.



### Constraints and Variables Reduction for Optimal Power Flow Using Hierarchical Graph Neural Networks with Virtual Node-Splitting (https://arxiv.org/abs/2411.06268)
- **What's New**: 전통적인 전력 시스템 모델링 방법에서 탈피하여, 가상 노드 분할(virtual node-splitting) 전략을 도입함으로써 개별 발전기(generator) 특성을 효과적으로 포착할 수 있게 되었습니다.

- **Technical Details**: 이 연구에서는 두 단계 적응형 계층적 GNN을 개발하였습니다. 첫 번째 단계는 혼잡할 수 있는 중요 라인(critical lines)을 예측하고, 두 번째 단계는 최대 용량으로 운영될 발전기(base generators)를 예측합니다. 이러한 방법은 OPF(Optimal Power Flow) 문제의 제약 변수와 조건을 크게 줄여줍니다.

- **Performance Highlights**: 제안된 ROPFLG 모델은 벤치마크된 전체 OPF(FOPF) 및 기타 두 개의 ROPF 방법들에 비해 지속적으로 우수한 성과를 보이며, 최적 솔루션을 안정적으로 찾는 동시에 계산 시간에서 상당한 절약을 이루었습니다.



### Towards Establishing Guaranteed Error for Learned Database Operations (https://arxiv.org/abs/2411.06243)
Comments:
          Appeared in ICLR'24

- **What's New**: 이번 연구는 머신러닝을 이용한 데이터베이스 작업에 대한 이론적 보장 조건을 제시하였습니다. 특히, 인덱싱, 카디널리티 추정, 범위 합 추정과 관련하여 각 작업의 정확도를 보장하기 위한 모델 크기(모델 사이즈)의 하한을 처음으로 제공합니다.

- **Technical Details**: 우리는 각 데이터베이스 작업에 대해 정해진 평균 및 최악의 경우 오류를 고려할 때 필요한 모델 크기의 이론적인 경계를 제시합니다. 모델 크기는 모델을 저장하는 데 필요한 비트 수로 측정됩니다(파라미터 수로 변환됨). 또한, 허용 가능한 최대 오류를 나타내는 'tolerable error parameter' ϵ(에프실론)도 정의하였습니다.

- **Performance Highlights**: 실험적으로, 학습된 모델들은 비학습 방법들과 비교할 때 인덱싱 및 카디널리티 추정에서 현저히 빠른 쿼리 시간 및 낮은 저장 공간을 제공합니다. 우리의 이론적 결과는 이러한 학습된 모델들이 실제 시스템에서 더 너비 있는 적용을 가능하게 하는 기반을 다집니다.



### Web Scale Graph Mining for Cyber Threat Intelligenc (https://arxiv.org/abs/2411.06239)
- **What's New**: 오늘날의 사이버 공격에 효과적으로 대응하기 위해 TITAN(Threat Intelligence Tracking via Adaptive Networks)이라는 새로운 프레임워크를 소개합니다. TITAN은 산업 규모의 그래프 마이닝 시스템으로, 사이버 위협 정보를 전례 없는 속도와 규모로 생성합니다. 이 시스템은 동적 위협 정보 그래프와 실시간 업데이트 메커니즘 등을 통해 복잡한 보안 환경을 다룹니다.

- **Technical Details**: TITAN은 동적 k-partite 그래프를 사용하여 수백만 개의 개체, 사건 및 조직 간의 복잡한 관계를 캡처하고, 보안 도메인 지식을 통합하여 초기 평판 점수를 부여하며, 평판 전파 알고리즘으로 숨겨진 위협 공급망을 유도합니다. Microsoft Unified Security Operations Platform(USOP)에 통합되어 있으며, 이를 통해 수십만 개의 조직에 적용됩니다.

- **Performance Highlights**: TITAN은 정기적으로 수백만 개의 고위험 개체를 식별하며, 비파일 위협 정보의 여섯 배 증가를 이루었습니다. 이러한 시스템의 도입으로 보안 사건 차단 비율이 21% 증가하고 차단 소요 시간을 1.9배 단축시켰으며 고객 피드백에 따르면 99%의 정밀도를 유지하고 있습니다.



### Leveraging Retrieval-Augmented Generation for University Knowledge Retrieva (https://arxiv.org/abs/2411.06237)
Comments:
          6 pages, 2 figures, 1 table, Submitted to 15th IKT conference

- **What's New**: 이 논문은 대학교 관련 질문 응답 시스템을 향상시키기 위한 Retrieval-Augmented Generation (RAG) 파이프라인을 사용하여 정보 검색을 혁신적으로 접근하는 방법을 소개합니다.

- **Technical Details**: 논문에서 제안한 RAG 파이프라인은 두 단계의 접근법을 이용하여 페르시아 대형 언어 모델(PLM)과 정교한 프롬프트 엔지니어링 기법을 결합합니다. 먼저, 쿼리를 분류하여 가장 관련 있는 문서를 찾아내고, 그 후 적절한 LLM을 사용하여 정확하고 맥락적으로 관련 있는 응답을 생성합니다. 이 연구는 'UniversityQuestionBench'(UQB)라는 포괄적인 벤치마크 데이터를 개발하여 RAG 파이프라인의 성능을 엄격하게 평가하였습니다.

- **Performance Highlights**: 실험 결과, 생성된 응답의 정확성과 관련성이 크게 향상되어 사용자 경험이 증진되고 관련 답변을 얻기 위한 소요 시간이 줄어들었습니다.



### Generalizing Hyperedge Expansion for Hyper-relational Knowledge Graph Modeling (https://arxiv.org/abs/2411.06191)
- **What's New**: 본 논문은 기존의 지식 그래프(KG)와 비교하여 하이퍼 관계 지식 그래프(HKG)의 모델링을 위한 TransEQ 메커니즘을 제안합니다. 이는 하이퍼엣지 확장을 일반화하여 HKG를 KG로 변환하는 동등한 변환을 통해 구조적 정보와 의미적 정보를 동시에 포착하는 혁신적인 접근 방식입니다.

- **Technical Details**: TransEQ는 하이퍼 그래프에서 그래프로 변환하는 개념인 하이퍼엣지 확장을 기반으로 하며, 인코더-디코더 프레임워크를 통해 HKG 모델링을 수행합니다. 인코더 부분에서는 그래프 신경망(GNN)을 사용하여 구조적 모델링을 수행하고, 디코더에서는 HKG 기반의 점수 함수(SF)를 활용하여 의미적 모델링을 진행합니다. 또한, 공유 임베딩 메커니즘을 설계하여 의미적 관련성을 캡처합니다.

- **Performance Highlights**: 실험 결과, TransEQ는 WikiPeople과 같은 대규모 벤치마크에서 MRR(Mean Reciprocal Rank)을 15% 향상시키며, 기존의 최첨단 모델에 비해 우수한 성능을 보여줍니다. 이러한 결과는 TransEQ가 효과적인 정보 포착 및 효율성을 제공함을 나타냅니다.



### Alleviating Hyperparameter-Tuning Burden in SVM Classifiers for Pulmonary Nodules Diagnosis with Multi-Task Bayesian Optimization (https://arxiv.org/abs/2411.06184)
Comments:
          12 pages, 4 figures, 37 references

- **What's New**: 이번 연구는 다중 의료 분류 작업을 동시에 해결하기 위해 다중 작업 Bayesian 최적화(multi-task Bayesian optimization, MTBO)를 사용하는 가능성을 조사하였습니다. MTBO를 통해 기존의 단일 작업 접근법(single-task approach)보다 하이퍼파라미터 탐색을 빠르게 수행할 수 있음을 발견하였습니다.

- **Technical Details**: 의료 이미지 분석에서 이미지 이산화(image discretization)는 관심 영역(region of interest, ROI) 내의 강도(intensity) 값을 작게 나누는 과정입니다. 본 연구에서는 9가지 이미지 이산화 전략을 제안하고, RBF SVM을 사용하여 여러 SVM 분류기의 하이퍼파라미터를 동시에 조정하는 알고리즘을 설계하였습니다. 이는 각 작업 간의 관계를 공유하여 최적의 모델을 생성하기 위한 것입니다.

- **Performance Highlights**: MTBO 알고리즘은 단일 작업 Bayesian 최적화(single-task Bayesian optimization)보다 더 높은 성능을 보였으며, 동시에 여러 SVM 분류기의 하이퍼파라미터를 효율적으로 찾아내는 데 기여하였습니다. 본 연구는 의료 분야에서 MTBO 기술을 적용한 최초의 연구입니다.



### Clustering Algorithms and RAG Enhancing Semi-Supervised Text Classification with Large LLMs (https://arxiv.org/abs/2411.06175)
- **What's New**: 이 논문은 제한된 레이블 예제로부터 효과적으로 학습할 수 있는 혁신적인 semi-supervised learning 방법을 도입합니다. 특히, retrieval-augmented generation (RAG)과 기존의 통계적 클러스터링을 통합하여 레이블이 지정된 인스턴스의 수가 최소화된 상태에서도 고품질의 레이블 데이터를 생성할 수 있습니다.

- **Technical Details**: 이 방법은 클러스터링 알고리즘을 통해 데이터 선택을 우선 진행하고, 그 후 전문가들이 수작업으로 레이블링을 수행합니다. 생성된 레이블은 RAG와 CoT 파이프라인을 통해 더 큰 크기의 LLM에서 추가 학습 샘플을 생성하는 데 사용됩니다. 마지막으로, 이러한 학습 샘플은 더 작은 LLM을 fine-tune하는 데 사용되어 잘못 분류된 샘플에 다시 집중할 수 있도록 설계됩니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 레이블링된 데이터의 양이 동일할 때 무작위 선택 및 기존 방법들보다 fine-tuned 모델의 정확도를 높이는 것으로 나타났습니다. 또한, 복잡한 텍스트 문서 분류 작업에서 95.41%와 82.43%의 정확도를 기록하여 최신 성과를 달성했습니다.



### SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models (https://arxiv.org/abs/2411.06171)
Comments:
          EMNLP2024

- **What's New**: 본 연구에서는 대형 언어 모델(LLM)의 지속 학습(Continual Learning)에서 주의(attention) 가중치의 중요성을 강조하며, 이를 바탕으로 데이터 효율적 재생(replay) 기반 지속 학습을 위한 SElective attEntion-guided Knowledge Retention 방법(SEEKR)을 제안합니다.

- **Technical Details**: SEEKR은 선택된 주의 헤드를 대상으로 주의 증류(attention distillation)를 수행하여 더 정교한 지식 보존을 달성합니다. 이 방법은 forgettability와 task-sensitivity 기반의 지표를 통해 가장 가치 있는 주의 헤드를 식별합니다. SEEKR은 과거 작업의 지식을 효과적으로 유지하기 위해 계층적 예산 할당 메커니즘을 사용합니다.

- **Performance Highlights**: 실험 결과, SEEKR은 기존 방법보다 성능과 효율성에서 우수한 결과를 보이며, 기존 방법이 사용하는 재생 데이터의 1/10만으로도 유사하거나 더 나은 성능을 달성하고, 재생 데이터 비율을 1%로 감소시킴으로써 데이터 효율성을 입증합니다.



### Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework (https://arxiv.org/abs/2411.06160)
- **What's New**: 이 논문은 텍스트 감정 탐지(Text Emotion Detection)의 발전을 위해 Emotion Quantization Network (EQN) 프레임워크를 제안합니다. EQN은 감정 강도를 에너지 수준으로 매핑하여 미세 감정(micro-emotion)의 자동 탐지 및 주석(annotation)을 가능하게 합니다.

- **Technical Details**: EQN 프레임워크는 모든 라벨(all-labels) 및 훈련 세트 라벨 회귀(training-set label regression) 방법을 활용합니다. 이를 통해 기계 모델의 학습능력과 라벨 간의 상호 의존성을 최대한 활용하여 샘플 내 여러 감정을 발견합니다. 이론적으로 미세 감정의 측정과 주석 작성에 기여하는 새로운 접근법이 설명됩니다.

- **Performance Highlights**: EQN 프레임워크는 GoEmotions 데이터셋에서 감정 탐지 및 주석을 수행하며, Google 연구 결과와의 포괄적 비교를 통해 높은 자동 탐지 능력을 입증합니다. EQN은 에너지 레벨 점수로 미세 감정 자동 주석을 최초로 달성하였으며, 감정 탐지 분석 및 감정 컴퓨팅의 정량적 연구에 강한 지원을 제공합니다.



### Deep Nonparametric Conditional Independence Tests for Images (https://arxiv.org/abs/2411.06140)
Comments:
          50 pages, 13 figures

- **What's New**: 이 논문에서는 복잡하고 고차원 변수를 다루기 위해 Deep Nonparametric Conditional Independence Tests (DNCITs)를 도입합니다. 기존의 조건부 독립성 테스트(CIT)가 고차원 변수에 제대로 적용되지 않은 한계를 극복하고자 합니다.

- **Technical Details**: DNCITs는 고차원 변수의 특징 표현(feature representations)을 추출하는 embedding maps와 이러한 특징 표현에 적용 가능한 비모수 CITs를 결합합니다. 또한, embedding maps의 매개변수 추정치에 대한 일반적인 성질을 도출하여 유효한 DNCITs를 얻습니다. DNCITs는 이미지와 스칼라 변수 간의 조건부 연관성을 테스트합니다.

- **Performance Highlights**: DNCITs는 UK Biobank의 뇌 MRI 스캔과 행동 특성을 분석하는 데 성공하였으며, 불확실한 성격 신경과학 연구의 널 결과를 확인했습니다. 나아가, 혼란 제어 연구에서 DNCITs를 적용하여 기존의 연구보다 개선된 혼란 제어 아래에서 혼란 차원을 잠재적으로 줄일 수 있었습니다.



### Exploring Structural Nonlinearity in Binary Polariton-Based Neuromorphic Architectures (https://arxiv.org/abs/2411.06124)
- **What's New**: 본 연구는 마이크로 캐비티에서 겹치는 극소 점에서 광학적으로 여기된 쌍의 극소 점(condensate) 즉, polariton dyads를 활용하여 바이너리 로직 게이트 뉴런으로 작동하는 이진화 신경망의 성능을 조사합니다. 또한, 뉴런 구성의 비선형성과 구조적 비선형성이 이미지 분류 작업에서 중요한 역할을 한다고 강조합니다.

- **Technical Details**: 이 연구에서는 여러 뉴런 구성을 실험하여 이진화 신경망(BNNs)의 효과를 평가했습니다. 특히 구조적 비선형성이 복잡한 계산 작업을 원활하게 하여 개별 뉴런의 비선형성에 의존할 필요를 줄인다는 점에 주목했습니다. 여기서 polariton dyads를 사용하여 인공 바이너리 뉴런을 구성하고, 이를 통해 다양한 비선형 및 선형 게이트(NAND, NOR, XNOR 등) 기능을 구현합니다.

- **Performance Highlights**: 저자들은 이미지 분류 정확도를 비교하는 수치 실험을 통해 구조적 비선형성이 인식 작업에서의 중요성을 보여주고, 이로 인해 신경망 디자인과 제작이 간소화될 수 있음을 시사합니다. 또한, MNIST 데이터셋에서의 인식 성능이 약 96%에 달하는 것으로 나타났습니다.



### Scalable, Tokenization-Free Diffusion Model Architectures with Efficient Initial Convolution and Fixed-Size Reusable Structures for On-Device Image Generation (https://arxiv.org/abs/2411.06119)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 Diffusion Model을 구현하기 위한 새로운 하드웨어 친화적인 신경망 아키텍처를 제안합니다. 기존의 Vision Transformer와 U-Net 아키텍처의 장점을 살리면서도, 보다 효율적인 하드웨어 구현을 가능하게 하는 구조를 설계하였습니다.

- **Technical Details**: 제안하는 아키텍처는 고정 크기의 재사용 가능한 transformer 블록을 중심으로 구성되어 있으며, 토큰화(tokenization)와 위치 임베딩(positional embedding)이 필요하지 않은 설계를 특징으로 합니다. 두 가지 구성(Configuration I 및 II)을 통해 다른 매개변수 수를 확장할 수 있으며, 각 구성에 따라 계산 복잡성을 조정하면서도 높은 이미지 생성 품질을 유지합니다.

- **Performance Highlights**: 모델은 조건부 및 비조건부 이미지 생성 작업에서 경쟁력 있는 성능을 보여주며, CelebA 데이터셋에서 비조건부 이미지 생성 시 FID 점수 1.6을 기록하여 최신 기술과 비교하여 우수한 결과를 보였습니다.



### BreakGPT: Leveraging Large Language Models for Predicting Asset Price Surges (https://arxiv.org/abs/2411.06076)
- **What's New**: BreakGPT는 자산 가격의 급격한 상승 예측을 위해 특별히 개조된 대형 언어 모델(LLM) 아키텍처입니다. 이 모델은 LLM과 Transformer 기반 모델의 강점을 결합하여 변동성이 큰 금융 시장에서의 예측 문제를 해결하도록 설계되었습니다.

- **Technical Details**: BreakGPT는 시간 시계열 데이터 예측을 위해 GPT-2 기반의 수정된 TimeLLM 아키텍처를 개발하였으며, 다양한 Transformer 기반 모델을 비교합니다. 이 아키텍처는 자산 가격 변동 예측에 적합하게 변경되었으며, OHLC(Open-High-Low-Close) 데이터와 여러 보조 지표(예: SMA, EMA, RSI)를 활용하여 시장 행동을 보다 정밀하게 포착합니다.

- **Performance Highlights**: BreakGPT는 특히 비트코인 및 솔라나와 같은 암호화폐 자산의 가격 급등 예측에 있어 강력한 성능을 보이며, 전통적인 통계 모델보다 뛰어난 결과를 도출하고 변동성이 큰 데이터에 효과적으로 대처할 수 있는 가능성을 보여주었습니다.



### Filling in Missing FX Implied Volatilities with Uncertainties: Improving VAE-Based Volatility Imputation (https://arxiv.org/abs/2411.05998)
Comments:
          35 pages, 22 figures, 10 tables

- **What's New**: 이 연구에서는 외환(FX) 옵션에 대한 누락된 내재 변동성(implied volatility) 수치를 보간하는 데 집중하였으며, 기존의 변별 오토인코더(variational autoencoders, VAE) 접근 방식에 비해 더 강력한 고전적 기준을 사용할 때 성능이 크게 향상될 수 있음을 보였습니다.

- **Technical Details**: 기존 VAE 아키텍처에 간단한 수정을 추가하여 보간 성능을 개선할 수 있으며, 이는 저Missingness(FX 변동성이 낮은 경우) 환경에서 오류를 거의 절반으로 줄일 수 있음을 보여줍니다. 또한, 이 연구에서는 데이터의 불확실성을 처리하기 위한 VAE 보간 알고리즘의 수정 사항을 포함합니다.

- **Performance Highlights**: 이 연구 결과, 새롭게 수정된 VAE는 FX 변동성 수치를 보간하는 데 있어 고전적인 방법들보다 더욱 정확한 불확실성 추정치를 제공하며, 다양한 고전적 방법들과 비교했을 때 개선된 성능을 보여주었습니다.



### Game-theoretic LLM: Agent Workflow for Negotiation Games (https://arxiv.org/abs/2411.05990)
Comments:
          44 pages, 12 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 전략적 결정 과정에서 이성을 발휘하는 수준을 게임 이론의 틀 안에서 조사합니다. 특히, 불완전 정보 게임 상황에서 LLMs의 비이성적인 전략 선택 경향을 분석하고 이를 극복하기 위한 게임 이론적 워크플로우를 개발하였습니다.

- **Technical Details**: 여러 최신 LLM들(예: Claude-3.5 Sonnet, GPT-4o 등)의 성능을 평가하였으며, 이는 완전 정보 및 불완전 정보 게임(예: Prisoner’s Dilemma, Battle of the Sexes 등)을 기반으로 합니다. 또한, 지배적 전략 탐색(Dominant Strategy Search), 후방 유도(Backward Induction), 베이지안 신념 업데이트(Bayesian belief updating)와 같은 게임 이론 원리를 활용하여 LLM의 이성적 행동과 결정 능력을 향상시키는 알고리즘을 설계하였습니다.

- **Performance Highlights**: 제안된 워크플로우를 통해 LLM들이 최적 전략을 식별하는 데 유의미한 개선을 보였으며, 협상 시 near-optimal allocations를 달성하고, 협상 과정에서의 착취(Exploitation)에 대한 저항력도 증가하였습니다. 연구 결과는 복잡한 상호작용 환경에서 전략적으로 더 견고하고 합리적인 AI 에이전트를 개발하는 데 기여할 수 있습니다.



### FactLens: Benchmarking Fine-Grained Fact Verification (https://arxiv.org/abs/2411.05980)
Comments:
          12 pages, under review

- **What's New**: 이 논문은 기존 LLM(대규모 언어 모델)의 사실 검증 방식을 재검토하고, 복잡한 주장을 세분화하여 각 서브 클레임(sub-claim)을 독립적으로 검증할 수 있는 방법을 제시합니다. 이를 통해 정확도, 투명성 및 증거 검색의 모호성을 줄일 수 있습니다.

- **Technical Details**: 저자들은 FactLens라는 새로운 벤치마크를 도입하여 세분화된 사실 확인을 평가합니다. FactLens는 서브 클레임의 품질을 평가하기 위한 메트릭스와 자동화된 평가자를 포함하고 있으며, 벤치마크 데이터는 수동으로 큐레이션되어 고품질의 그라운드 트루스를 보장합니다.

- **Performance Highlights**: 세분화된 검증 처리와 관련된 정량적 메트릭스를 통해 자동화된 평가자와 인간의 판단 간의 일치도를 입증하였으며, 서브 클레임 생성의 도전 과제를 논의하고, 최첨단 모델의 결과를 제시하였습니다.



### Energy Efficient Protein Language Models: Leveraging Small Language Models with LoRA for Controllable Protein Generation (https://arxiv.org/abs/2411.05966)
- **What's New**: 이 연구에서는 Llama-3-8B와 Phi-3-mini를 기반으로 한 두 개의 작은 단백질 언어 모델을 제시합니다. 이 모델들은 통제 불가능한(uncontrollable) 생성과 통제 가능한(controllable) 단백질 생성을 수행할 수 있습니다.

- **Technical Details**: 모델의 성능을 향상시키기 위해 Low-Rank Adaptor(LoRA) 기술을 활용하여 원래 모델 크기의 4%로 학습 가능한 파라미터 수를 줄였습니다. 또한, UniRef50 데이터셋의 일부를 활용하여 전체 훈련 시간을 70% 단축하며 성능 저하 없이 작은 모델을 사용했습니다.

- **Performance Highlights**: 우리의 모델 중 가장 성능이 좋은 모델은 pLDDT 점수 69.75를 달성하였고, 통제 가능한 생성 작업에서 평균 TM-Score 0.84를 기록하여 목표 단백질과 높은 구조적 유사성을 나타냈습니다. Phi-3-mini 모델은 학습 가능한 파라미터를 60% 줄이고, Llama 3에 비해 30%의 훈련 비용 감소를 달성했습니다.



### A method based on Generative Adversarial Networks for disentangling physical and chemical properties of stars in astronomical spectra (https://arxiv.org/abs/2411.05960)
- **What's New**: 이번 연구는 astrophysical (천체 물리학적) 스펙트럼 분석을 위한 새로운 인코더-디코더 아키텍처를 제안하며, 전통적인 오토인코더를 수정한 적대적 훈련(adversarial training)을 적용했습니다. 특히, 중간 표현(intermediate representation)을 얻는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 별의 스펙트럼에서 표면 온도(surface temperature)와 중력(gravity) 같은 주요 물리적 특성의 기여를 제거하고, 화학 조성(chemical composition)의 효과만 반영되는 변화를 관찰합니다. 딥 러닝(deep learning) 기법을 통해 잠재 공간(latent space)에서 원하는 매개변수를 드러내는 프레임워크(GANDALF)를 제안합니다.

- **Performance Highlights**: 효과성 검증을 위해 APOGEE와 Gaia 조사의 합성 천문 데이터(synthetic astronomical data)를 사용하며, 제안하는 방법의 복제, 시각화(visualization), 및 다양한 도메인에 대한 확장을 가능하게 합니다.



### Sentiment Analysis of Cyberbullying Data in Social Media (https://arxiv.org/abs/2411.05958)
- **What's New**: 이 연구는 사이버 괴롭힘 탐지를 위한 감정 분석에 있어 두 가지 새로운 하이브리드 방법을 소개합니다. 기존의 기술 대신에 최신 임베딩을 활용하고, 특히 사이버 괴롭힘 데이터에 대한 RNN 프레임워크와 BERT, OpenAI 임베딩을 결합한 접근방법을 채택했습니다.

- **Technical Details**: 연구에서는 LSTM(Long Short-Term Memory) 셀을 사용하는 순환 신경망을 개발하여, 감정 분석에 대한 BERT 임베딩 및 OpenAI의 최신 임베딩 API를 비교합니다. 이 기술들은 자연어 처리(NLP)와 기계 학습(ML) 영역에서 고품질의 주석 데이터가 충분히 확보되어야 성능 평가가 가능하다는 점을 강조합니다.

- **Performance Highlights**: Formspring 사이버 괴롭힘 데이터를 사용하여 두 가지 접근 방식을 효과적으로 비교하며, 최신 LLM(Large Language Model)을 사용한 방법이 과거 방법들보다 더 높은 정확도로 사이버 괴롭힘을 탐지하는 데 기여하고 있음을 보여줍니다.



### Tackling extreme urban heat: a machine learning approach to assess the impacts of climate change and the efficacy of climate adaptation strategies in urban microclimates (https://arxiv.org/abs/2411.05952)
- **What's New**: 도시화(urbanization)와 기후 변화(climate change)의 진행으로 도시 열(urban heat)이 기후 적응 노력에서 중요한 과제로 떠오르고 있습니다. 본 연구에서는 고해상도의 기후 변화 영향 추정 방법을 제시합니다.

- **Technical Details**: 공개 소스(open-source)이며 계산 효율(computationally efficient)인 머신러닝(machine learning) 방법을 통해 로스앤젤레스(Los Angeles)의 주거 건물(residential buildings)에서 도시 온도 추정의 정확도를 향상시키고, 에너지 수요(energy demand)와 기후 변화의 영향을 비교 분석합니다. 이 모델은 역사적 재분석 데이터(historical reanalysis data)와 비교하여 성능이 강화되었습니다.

- **Performance Highlights**: 미래 중반까지 냉방 수요(cooling demand)가 크게 증가할 것으로 예상되며, 그러나 고알베도 표면(engineered high-albedo surfaces)을 활용하면 이 증가폭을 50% 이상 감소시킬 수 있습니다. 또한 전기 히트 펌프(electric heat pumps)를 이용한 난방 및 냉방의 총 연간 에너지 사용(total annual energy use)은 현재 및 미래의 기후 하에서도 이러한 냉방 전략의 혜택을 받을 것으로 나타났습니다.



### NeKo: Toward Post Recognition Generative Correction Large Language Models with Task-Oriented Experts (https://arxiv.org/abs/2411.05945)
Comments:
          NeKo work has been done in June 2024. NeKo LMs will be open source on this https URL under the MIT license

- **What's New**: NeKo는 일반 포스트 인식 오류 수정 모델을 위한 새로운 접근방식으로, Mixture-of-Experts (MoE) 아키텍처를 활용하여 다양한 도메인 데이터셋에서 효율적으로 학습합니다. 이 모델은 서로 다른 작업을 위해 전문가를 훈련시켜 각 데이터셋 토큰을 해당 전문가에게 라우팅하는 방식으로 작동합니다.

- **Technical Details**: NeKo는 다양한 오류 수정 데이터셋의 혼합에 대해 사전 훈련된 MoE 모델을 기반으로 하며, 각 전문가는 특정 도메인에 특화됩니다. 이 과제 지향적인 MoE 세부 조정 방식은 각 전문가가 작업 특정 특징을 포착하게 하며, 지식 공유가 가능합니다. 이를 통해 네트워크는 음성 인식, 번역 및 OCR 오타 수정 등의 작업에서 높은 성능을 발휘합니다.

- **Performance Highlights**: NeKo는 Open ASR Leaderboard 및 Hyporadise 벤치마크에서 기존 모델들 대비 평균 상대 WER을 5.0% 감소시켰으며, 0-shot 평가에서 GPT-3.5와 Claude-Opus보다 15.5%에서 27.6%까지 WER 감소를 달성했습니다. 이 결과로 NeKo는 멀티태스크 모델에서 경쟁력을 입증하였습니다.



### Quantifying artificial intelligence through algebraic generalization (https://arxiv.org/abs/2411.05943)
- **What's New**: 이번 논문에서는 알gebraic circuit complexity(대수 회로 복잡도) 이론을 도입해 기호적 일반화(symbolic generalization)를 명확히 정량화하는 새로운 프레임워크를 제시합니다. 현재의 인공지능(AI) 시스템은 기호 처리 및 추상화에 있어 한계를 보이고 있으며, 이 문제를 해결하기 위한 방법론을 수립하는 것에 초점을 두고 있습니다.

- **Technical Details**: 기호적 계산(symbolic computation)의 복잡성을 연구하기 위해 대수 회로 복잡도 이론을 사용합니다. 이 이론은 수학적 표현을 회로 모델(즉, 방향성 비순환 그래프)로 정식화하며, 각 회로의 크기(size)와 깊이(depth)를 주요 복잡성 척도로 설정합니다. 이를 통해 AI 모델의 일반화 성능을 객관적으로 평가할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 대수 회로 복잡도 이론을 통해 다양한 기호 문제를 다룰 수 있으며, 이는 AI 시스템의 약점 및 실패 양상을 구체적으로 파악하는 데 유리합니다. 알gebraic circuits는 그 본질상 수많은 표본(sample)을 생성할 수 있어 오늘날 데이터 중심의 기계 학습 알고리즘을 위한 최적의 테스트베드(testbed)가 됩니다.



### GCI-ViTAL: Gradual Confidence Improvement with Vision Transformers for Active Learning on Label Nois (https://arxiv.org/abs/2411.05939)
Comments:
          under review

- **What's New**: 이번 연구에서는 label noise가 있는 상황에서 이미지 분류를 위한 Active Learning (AL) 방법들을 비교하고 새로운 deep active learning 알고리즘인 GCI-ViTAL을 제안합니다. 이 알고리즘은 label noise에 강인하도록 설계되었습니다.

- **Technical Details**: GCI-ViTAL은 예측 엔트로피(predictive entropy)와 클래스 중심 깨끗한 세트 주의 벡터(class-centric clean set attention vectors)와 비교한 마지막 레이어 주의 벡터의 Frobenius norm을 이용합니다. 이 모델은 불확실성과 의미적으로 전통적인 이미지와 다른 샘플을 식별하는 데 도움을 줍니다. Label smoothing이 적용되어 잠재적으로 노이즈가 포함된 레이블에 대해 지나치게 확신하지 않는 모델 학습을 지원합니다.

- **Performance Highlights**: GCI-ViTAL은 다양한 수준의 대칭 label noise에 대해 평가되었으며, CNN 모델에 비해 모든 AL 전략에서 ViT 모델을 사용하면 성능이 크게 향상됨을 보여주었습니다. 특히, label noise가 있는 경우에서 더 두드러진 성과를 보였습니다.



### Moving Off-the-Grid: Scene-Grounded Video Representations (https://arxiv.org/abs/2411.05927)
Comments:
          Accepted to NeurIPS 2024 (spotlight). Project page: this https URL

- **What's New**: MooG(Moving Off-the-Grid)는 비디오 표현 학습을 위한 새로운 self-supervised 방법론으로, 기존의 grid-based 처리 방식만으로 엮이지 않고, 영상의 구조와 무관하게 장면의 요소를 보다 일관되게 표현할 수 있는 토큰들의 움직임을 가능하게 한다.

- **Technical Details**: MooG는 cross-attention과 positional embeddings의 조합을 활용하여 representation 구조와 이미지 구조를 분리시킨다. 이 모델은 다음 프레임 예측(next frame prediction) 손실을 이용하여 비디오 데이터에서 학습되며, 입력 프레임이 도착할 때마다 cross-attention을 통해 토큰을 업데이트하고, 이미지를 복원하는데도 cross-attention을 사용한다.

- **Performance Highlights**: MooG는 DINO와 같은 기존 self-supervised grid-based 모델들을 초월하는 성능을 보이며, point tracking, monocular depth estimation 및 object tracking 등 다양한 다운스트림 비전 작업에 유용한 특징을 제공함을 정량적으로 및 정성적으로 입증하였다.



### ViT Enhanced Privacy-Preserving Secure Medical Data Sharing and Classification (https://arxiv.org/abs/2411.05901)
Comments:
          2 pages, 2 figures

- **What's New**: 본 연구에서는 의료 이미지 분석을 위한 안전하고 개인 정보를 보호하는 데이터 공유 프레임워크를 소개합니다. 이 프레임워크는 learnable encryption 방식을 통해 데이터를 암호화하고 Vision Transformer (ViT)와 통합하여 정확성을 유지하면서도 보안성을 강화합니다.

- **Technical Details**: 제안된 접근 방식은 block-pixel operation을 기반으로 하는 learnable encryption 기법을 사용하여 이미지의 주요 특징을 보존하면서도 이러한 이미지를 암호화합니다. 다수의 고객이 각기 다른 키를 사용해 로컬 이미지를 암호화하고, 변환된 이미지는 중앙 서버를 통해 전송되어 Transformer Encoder로 복잡한 의존성을 캡처합니다.

- **Performance Highlights**: 연구에서 제안한 ViT 모델은 5,712개의 MRI 이미지를 사용한 brain tumor 분류 및 2,980개의 histopathological 이미지를 사용한 폐 및 대장암 분류에서 높은 정확도를 기록했습니다. ViT 모델은 암호화된 MRI 뇌종양 데이터셋에서 95%의 훈련 정확도와 94%의 검증 정확도를 달성하였으며, 기존 DNN 모델은 38%에서 51%로 낮은 정확도를 보였습니다.



### Enhancing Cardiovascular Disease Prediction through Multi-Modal Self-Supervised Learning (https://arxiv.org/abs/2411.05900)
Comments:
          Accepted to British Machine Vision Conference (BMVC) 2024

- **What's New**: 본 연구는 단일 모달리티(meta-modal) 데이터만으로는 포착할 수 없는 심혈관질환(CVD) 예측의 새로운 통찰력을 다중 모달 학습(multi-modal learning)을 통해 제공하는 것을 목표로 하고 있습니다. 심장 자기 공명 영상(cardiac magnetic resonance images)과 심전도(electrocardiogram) 신호, 의료 정보를 통합하여 개인의 심혈관 건강 상태를 포괄적으로 이해할 수 있도록 하는 모델을 제안합니다.

- **Technical Details**: 본 연구에서는 masked autoencoder를 사용해 ECG 인코더를 전훈련(pre-train)하여 심전도 데이터에서 관련 특징을 추출합니다. 이어서, 이미지 인코더를 통해 심장 자기 공명 영상에서 관련 특징을 추출하고, multimodal contrastive learning을 통해 비싼 CMR 모달리티에서 경제적인 ECG 및 의료 정보 모달리티로 지식을 전이합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 다양한 이용 가능 모달리티의 정보를 활용하여 이미지를 개선하고, 감독 학습(supervised approach)보다 7.6% 향상된 balanced accuracy를 기록하였습니다.



### SSSD: Simply-Scalable Speculative Decoding (https://arxiv.org/abs/2411.05894)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 연구에서는 Speculative Decoding(가설적 디코딩) 기법이 대규모 언어 모델(LLM) 추론을 가속화하는 데 효과적으로 사용될 수 있는 방법을 제시합니다. 여기서는 기존 시스템에 추가적인 훈련 없이 통합 가능한 방법을 소개하며, 짧은 컨텍스트 생성 시 처리량을 4배 증가시키고 지연 시간에는 영향을 미치지 않는 성과를 달성했습니다.

- **Technical Details**: 이 연구는 대규모 배치 크기에서 Speculative Decoding이 어떻게 효과적으로 적용될 수 있는지를 이론적으로 설명합니다. 전처리 단계에서 KV-cache를 생성하고 첫 번째 토큰을 생성하는 패러렐 처리와, 모델이 반복적으로 새로운 토큰을 생성하는 오토 리그레시브 디코딩 단계로 나뉩니다. 이 연구는 특히 대규모 LLM 배포에서 사용되는 복잡한 디코딩 과정을 최적화하는 새로운 기법을 제안합니다.

- **Performance Highlights**: 이 방법은 짧은 컨텍스트 생성에서 처리량을 4배 증가시키고, 긴 컨텍스트의 경우 지연 시간과 처리량이 각각 1.7배에서 2배 향상되는 성과를 보였습니다. 다양한 사용사례에서 효과적으로 활용될 수 있도록 설계되었습니다.



### A Comparative Analysis of Machine Learning Models for DDoS Detection in IoT Networks (https://arxiv.org/abs/2411.05890)
Comments:
          6 pages, 6 figures

- **What's New**: 본 논문은 IoT 네트워크에서 DDoS 공격을 탐지하기 위해 머신러닝 모델을 사용하는 방법을 제시합니다. 빠르게 성장하는 IoT 환경은 다양한 사이버 공격에 대해 더욱 취약해지며, 기존의 보안 절차는 불규칙적으로 시행되고 있습니다.

- **Technical Details**: 연구는 XGBoost, K-Nearest Neighbours (KNN), Stochastic Gradient Descent (SGD), Naïve Bayes 모델을 활용해 정상 네트워크 트래픽에서 DDoS 공격 탐지의 효율성을 평가합니다. 모든 모델은 정확도, 정밀도, 재현율, F1-score 등 여러 성능 지표를 통해 DDoS 위협에 대한 실시간 탐지 및 대응의 적합성을 분석합니다.

- **Performance Highlights**: 이 분석을 통해 각 모델의 고유한 강점과 약점이 정리되며, IoT 환경에서의 DDoS 탐지에 대한 효과적인 대응이 가능함을 보여줍니다. 머신러닝은 IoT 보안 프레임워크를 크게 향상시킬 수 있는 잠재력을 가집니다.



### Sdn Intrusion Detection Using Machine Learning Method (https://arxiv.org/abs/2411.05888)
Comments:
          15 Pages, 14 Figures

- **What's New**: 본 연구에서는 SDN(Software-defined network) 환경의 보안을 강화하기 위해 기계 학습(machine learning) 방법을 개발하였습니다.

- **Technical Details**: 연구진은 UNSW-NB 15 침입 탐지 데이터셋을 활용하여 Gradient Boosting을 포함한 여러 분류기(classifier)를 평가하였고, Random Forest와 Decision Tree를 비교하여 Gradient Boosting이 99.87%의 정확도(accuracy)를 기록하며 최상의 성능을 보였음을 확인하였습니다. 이 모델은 약한 학습자(weak learners)를 결합해 강력한 앙상블 모델을 생성합니다.

- **Performance Highlights**: Gradient Boosting은 99.87%의 정확도, 100%의 재현율(recall), 99.85%의 F1 점수를 기록하였으며, 보안 네트워크에서의 침입 탐지에 신뢰성이 있음을 보여주었습니다. 또한, Random Forest가 99.38%의 정확도로 두 번째로 높은 성능을 보였습니다. 향후 연구에서는 이 모델을 실제 SDN 환경에 통합하여 그 응용 가능성과 확장성을 관찰할 예정입니다.



### Predictive Digital Twin for Condition Monitoring Using Thermal Imaging (https://arxiv.org/abs/2411.05887)
- **What's New**: 이 논문은 상태 모니터링을 위해 특별히 설계된 예측 디지털 트윈의 개발과 실제 적용을 탐구합니다. 엄격한 수학적 모델과 열 이미징 기술을 사용하여, Robust Principal Component Analysis (RPCA) 및 Dynamic Mode Decomposition (DMD)와 Proper Orthogonal Decomposition (POD)의 통합 개발을 제안합니다.

- **Technical Details**: 이 연구에서는 열 이미징을 통해 모니터링된 가열된 판에 대한 실시간 실험을 포함하여 디지털 트윈의 예측 능력과 상태 모니터링, 이상 탐지 기능을 입증합니다. 그리고 가상 현실을 포함한 인간-기계 인터페이스를 도입하여 사용자 상호작용과 시스템 이해를 향상시킵니다.

- **Performance Highlights**: 연구의 주요 기여는 고차원 열 이미징 데이터를 기반으로 한 상태 모니터링을 위한 물리적 프레임워크의 개발과 그에 따른 실시간 진단 및 예측 디지털 트윈의 생성 및 시연입니다. 이러한 기여는 디지털 트윈이 산업 관행을 혁신할 잠재력을 보여줍니다.



### Alternative Learning Paradigms for Image Quality Transfer (https://arxiv.org/abs/2411.05885)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 새로운 연구에서는 낮은 품질의 의료 이미지를 높은 품질 이미지에서 학습한 정보를 활용하여 향상시키는 'Image Quality Transfer (IQT)' 접근 방식을 제안합니다. 이 연구는 기존의 감독 학습 프레임워크에 따른 IQT 방법과는 달리 비감독 학습 접근법과 감독-비감독 혼합 접근법을 포함하는 두 가지 새로운 IQT 문제 공식화를 제안합니다.

- **Technical Details**: 첫 번째 접근법인 IQT-SRep는 희소 표현(Sparse Representation, SRep) 및 사전 학습 모델을 사용하는 비감독 학습 기반입니다. 두 번째 접근법인 IQT-DDL(Deep Dictionary Learning)은 감독 및 비감독 학습의 조합을 사용하며, 입력 볼륨을 업스케일하기 위한 고해상도 사전을 명시적으로 학습합니다. 이러한 두 모델은 낮은 전장 자기공명영상(MRI) 애플리케이션에서 평범한 MRI 스캐너로부터 얻은 높은 품질 이미지를 복구하기 위해 평가되었습니다.

- **Performance Highlights**: 제안된 방법은 최신 감독 딥러닝 IQT 방법(IQT-DL)과 비교되었으며, 새로운 두 공식화는 훈련 데이터의 분포와 다른 분포의 데이터를 사용하여 테스트할 때 감독 방법에 발생하는 편향을 피할 수 있음을 보여주었습니다. 이는 IQT의 잠재적인 이점을 강조합니다.



### Untrained Perceptual Loss for image denoising of line-like structures in MR images (https://arxiv.org/abs/2411.05884)
- **What's New**: 본 논문에서는 Magnetic Resonance (MR) 이미지의 3D 영상 잡음을 제거하기 위해 untrained Perceptual Loss (uPL)를 도입하였습니다. 이 방법은 뿌리나 혈관과 같은 선형 구조를 포함하는 MR 이미지에 더 적합합니다.

- **Technical Details**: uPL은 이미지에 포함된 선형 구조의 특징을 고려하며, 기존의 L1 손실 또는 SSIM 기반 손실 함수를 초월하는 성능을 보여줍니다. 다양한 uPL 특성(가중치 초기화, 네트워크 깊이, 커널 크기, 풀링 연산 등)이 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: uPL을 사용한 결과, MRA 이미지에서 SSIM 값이 0.93으로, 이전의 L1과 SSIM 손실이 각각 0.81과 0.88에 불과한 것에 비해 우수한 성능을 보였습니다. 작은 uPL 네트워크가 VGG와 같은 대형 네트워크보다 더 우수한 성능을 보이며, 계산 비용에서도 효율적입니다.



### Towards Equitable ASD Diagnostics: A Comparative Study of Machine and Deep Learning Models Using Behavioral and Facial Data (https://arxiv.org/abs/2411.05880)
- **What's New**: 이번 연구에서는 여성의 자폐 스펙트럼 장애(ASD) 진단을 위한 기계 학습 모델, 특히 Random Forest와 convolutional neural networks를 평가했습니다. 이 연구는 ASD 진단을 개선하기 위한 혁신적인 접근 방식을 제안하고 있습니다.

- **Technical Details**: Random Forest 모델은 다수의 데이터셋에서 100% 검증 정확도를 달성했으며, 이 모델의 복잡한 관계 관리 능력과 낮은 오탐지율로 인해 조기 개입에 중요한 역할을 할 수 있습니다. MobileNet은 이미지 기반 분석에서 87%의 정확도로 baseline CNN을 초과했지만, 30%의 검증 손실이 나타나 추가 최적화가 필요합니다.

- **Performance Highlights**: Random Forest의 높은 정확도와 균형 잡힌 정밀도-재현율 지표는 임상 작업 흐름 개선에 기여할 수 있습니다. MobileNet의 경량 구조는 자원이 제한된 환경에서도 접근 가능한 ASD 스크리닝을 가능하게 할 잠재력을 보여줍니다.



### Compactly-supported nonstationary kernels for computing exact Gaussian processes on big data (https://arxiv.org/abs/2411.05869)
- **What's New**: 이 논문에서는 기존의 Gaussian process (GP) 모델의 한계를 극복하기 위해 새로운 커널을 제안합니다. 이 커널은 희소성(sparsity)과 비정상성(nonstationarity)을 직접 발견하고 인코딩할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: 제안된 커널은 완전한 베이지안 GP 모델에 포함되어 대규모 데이터 세트를 분석하는 데 필요한 고성능 컴퓨팅 자원을 활용합니다. 기존의 GP 메소드와는 달리 이 모델은 복잡한 하이퍼 파라미터를 줄이는 방식으로 설계되었습니다.

- **Performance Highlights**: 제안된 커널은 100만 개 이상의 일일 최고 기온 데이터 측정값을 기반으로 한 공간-시간 예측에서 최첨단 방법보다 우수한 성과를 나타냈습니다. 이는 다양한 합성 데이터 예제에서도 입증되었습니다.



### Provably Faster Algorithms for Bilevel Optimization via Without-Replacement Sampling (https://arxiv.org/abs/2411.05868)
- **What's New**: 최근 Bilevel Optimization (이층 최적화)에 대한 효율적인 알고리즘이 새롭게 제안되었습니다. 독립 샘플링을 가정하는 기존의 Stochastic Gradient 기반 알고리즘의 한계를 극복하기 위해, 본 연구에서는 이층 최적화의 예시 선택 전략을 다룹니다.

- **Technical Details**: 새롭게 소개된 알고리즘은 'without-replacement sampling' 기법을 활용하여 독립 샘플링에 기반한 알고리즘보다 더 빠른 수렴률을 달성합니다. 이 연구는 standard bilevel optimization을 넘어서 conditional bilevel optimization, minimax, compositional optimization과 같은 두 가지 특별한 경우에 대해서도 논의합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 알고리즘은 합성 및 실제 응용 프로그램에서 기존 알고리즘보다 우수한 성능을 보여주었습니다.



### Modeling Nonlinear Oscillator Networks Using Physics-Informed Hybrid Reservoir Computing (https://arxiv.org/abs/2411.05867)
Comments:
          27 pages, 10 figures, 17 supplementary figures. Code available at this https URL

- **What's New**: 본 논문에서는 하이브리드 레저버 컴퓨팅(hybrid reservoir computing)을 사용하여 비선형 진동자 네트워크(non-linear oscillator networks)의 대체 모델을 개발했습니다. 이 모델은 '전문가' 해석 모델과 결합하여 복잡한 실제 상황을 더 잘 반영할 수 있게 설계되었습니다.

- **Technical Details**: 하이브리드 레저버 컴퓨팅(Hybrid Reservoir Computing, HRC)은 전통적인 레저버 컴퓨팅과 전문가 모델을 통합한 것으로, 주요 비선형 결합 항들이 포함된 확장된 실제 모델과 비교하여 성능을 평가했습니다. 간단한 모델 대신 레저버 컴포넌트를 통해 보완할 수 있는지를 조사했습니다.

- **Performance Highlights**: 하이브리드 레저버 컴퓨터는 일반적으로 표준 레저버 컴퓨터보다 더 나은 성능을 보였으며, 특히 관측된 스펙트럴 반경(spectral radius) 임계값을 넘을 때 성능 저하가 발생하지 않았습니다. 또한, 전문가 모델로 진입할 수 없는 역동적 상황에서도 우수한 성능을 나타냈습니다.



### Rethinking Deep Learning: Non-backpropagation and Non-optimization Machine Learning Approach Using Hebbian Neural Networks (https://arxiv.org/abs/2411.05861)
Comments:
          13 pages, 4 figures

- **What's New**: 본 연구는 MNIST 분류 문제를 해결하기 위해 기존의 backpropagation(역전파)이나 objective function(목적 함수) 없이 생물학적 신경 시스템을 모방한 Hebbian learning(헤비안 학습) 방법을 제안합니다.

- **Technical Details**: 연구는 세 단계로 진행되었습니다. 첫 번째 단계에서 Hebbian learning 규칙을 적용했으나 기존의 비-Hebbian NNs보다 정확도가 낮아 일반적인 훈련 방법의 한계를 나타냅니다. 두 번째 단계에서는 norm-based cognition(정규 기반 인식)을 사용하여 특정 레이블에 훈련된 NNs의 반응성을 분석했습니다.

- **Performance Highlights**: 세 번째 단계에서는 벡터 정규 크기를 기준으로 한 MNIST 문자 인식 프로그램을 개발하여 약 75%의 정확도를 달성하였습니다. 이는 Hebbian learning NNs가 목적 함수, 역전파 또는 최적화 과정 없이도 손글씨 문자 인식이 가능함을 보여줍니다.



### Saliency Assisted Quantization for Neural Networks (https://arxiv.org/abs/2411.05858)
- **What's New**: 이번 논문은 딥러닝 모델의 해석 가능성을 개선하고 자원 제약 환경에서의 효율성을 높이기 위해 실시간 설명을 제공하는 새로운 접근 방식을 제안합니다. 이를 통해 모델이 입력의 가장 중요한 특징에 집중하도록 유도하며, 모델의 예측 정확도와 해석 가능성 간의 균형을 조명합니다.

- **Technical Details**: 연구는 Convolutional Neural Networks에서 양자화(Quantization)가 해석 가능성과 정확도에 미치는 영향을 비교 분석합니다. 특히, Parameterized Clipping Activation 방법을 사용하여 양자화를 구현하고, MNIST와 FashionMNIST 데이터셋에서 모델의 성능을 평가했습니다. 세 가지 비트(Bits) 폭 구성(2-bit, 4-bit, mixed 4/2-bit)을 통해 효율성과 해석 가능성 간의 트레이드오프를 탐색합니다.

- **Performance Highlights**: 결과적으로, 양자화는 자원 제한 장치에서 모델을 구현하는 데 필수적이지만, 정확도와 해석 가능성 간의 트레이드오프가 필요하다는 것을 보여줍니다. 낮은 비트 폭에서 두 메트릭의 감소가 보다 두드러지며, 특히 모델 투명성 요구가 있는 응용 프로그램에서 양자화 매개변수 선택의 중요성을 강조합니다.



### Evaluating the Economic Implications of Using Machine Learning in Clinical Psychiatry (https://arxiv.org/abs/2411.05856)
Comments:
          11 pages, submitted to Machine Learning for Health (ML4H) 2024

- **What's New**: 이 연구는 임상 정신의학에서 머신러닝(ML)의 경제적 함의에 대한 기존 연구의 공백을 메우고자 하며, 이 분야의 비용 효과성과 공정성 문제를 다룹니다.

- **Technical Details**: 본 연구는 3개의 문제 지향 사례 연구(case studies), 경제학 및 의료 AI에 관한 문헌, 두 가지 종류의 건강 경제 평가(health economic evaluations)를 통해 ML의 경제적 함의를 평가합니다.

- **Performance Highlights**: 정신 질환으로 인한 글로벌 비용이 5조 달러에 달한다고 추정되며, ML은 진단 정확도를 향상시키고 자원을 절약함으로써 정신 건강 서비스의 제공을 개선할 가능성이 있습니다.



### A Fundamental Accuracy--Robustness Trade-off in Regression and Classification (https://arxiv.org/abs/2411.05853)
- **What's New**: 이 논문은 adversarial risk와 standard risk 간의 기본적인 trade-off를 유도합니다. 이는 '최적의 예측기가 부드럽지 않다면, adversarial robustness는 정확도의 희생을 초래한다'는 직관을 공식화한 것입니다.

- **Technical Details**: 논문에서는 주어진 데이터 샘플 (X, Y)의 쌍에 대해 예측 함수를 찾는 문제를 다룹니다. 예측 함수 f는 특정 손실 함수 ℓ에 대해 adversarial risk를 최소화하는 데 초점을 맞춥니다. adversarial risk는 perturbation Δ가 주어졌을 때의 예측값과 실제 레이블 Y 간의 관계로 정의됩니다.

- **Performance Highlights**: 이러한 framework을 통해 polynomial ridge functions를 이용한 회귀 문제에서의trade-off를 구체적으로 평가하고, adversarial risk가 낮은 경우에 대해 예측기 성능을 보장할 수 있는 근거가 제공됩니다.



### Are Deep Learning Methods Suitable for Downscaling Global Climate Projections? Review and Intercomparison of Existing Models (https://arxiv.org/abs/2411.05850)
Comments:
          Under review for Earth's Future

- **What's New**: 이 논문에서는 Deep Learning (DL)을 활용한 기후 변화 예측의 다운스케일링에서 발생하는 문제점을 다루고 있습니다. 특히, Perfect Prognosis (PP) 모델이 관측 데이터에 기반하여 훈련된다는 점이 중요한 포인트입니다.

- **Technical Details**: 우리는 문헌 검토를 통해 PP 다운스케일링을 위한 최신 DL 모델을 식별하고, 이러한 모델들의 성능 평가를 위한 비교 실험을 수행했습니다. 이 실험에서는 다양한 훈련 복제본의 민감성을 고려하여 최소 및 최대 온도와 스페인에서의 강수량을 분석했습니다.

- **Performance Highlights**: 결과는 현재의 방법론의 한계와 개선 방향에 대해 논의하고, 향후 개발 전망에 대한 통찰을 제공합니다.



### Input-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks (https://arxiv.org/abs/2411.05849)
Comments:
          24 pages, 15 figures

- **What's New**: 이 논문에서는 Hopfield 모델의 새로운 동역학 시스템 프레임워크를 제안하여 외부 입력이 신경 시냅스에 직접 영향을 미치고 Hopfield 모델의 에너지 경관을 형성하는 방식을 설명합니다. 이는 기억 검색 과정에 대한 명확한 에너지 해석을 제공하며, 혼합된 입력들을 올바르게 분류하는 데 효과적입니다.

- **Technical Details**: 이 모델은 현대 Hopfield 아키텍처의 틀 내에서 통합되어 현재 정보와 과거 정보가 검색 과정에서 어떻게 결합되는지를 밝힙니다. 특히, 이 논문은 입력이 에너지 경관을 형성하고 결과적인 경량 하강(gradient descent) 흐름에 영향을 미친다는 점이 주요 특징입니다.

- **Performance Highlights**: 노이즈가 있게 구성된 환경에서 고전적인 모델과 새로운 모델의 강건성을 비교하며, 우리 모델의 장점은 과거와 현재 정보를 통합하여 부정확한 입력에 의해 유도되는 오분류 오류를 줄이는 데 있습니다.



### Efficient and Robust Freeway Traffic Speed Estimation under Oblique Grid using Vehicle Trajectory Data (https://arxiv.org/abs/2411.05842)
Comments:
          accepted by T-ITS

- **What's New**: 이번 연구에서는 소수의 차량 궤적 데이터를 사용하여 고속도로의 시공간 교통 속도 상태(Traffic Speed State)를 정확히 추정하기 위한 효율적이고 강인한 저랭크(low-rank) 모델를 제안합니다. 이 모델은 교통 파동(traffic wave) 선행 지식을 활용하여 특수한 기하학적 구조를 가진 매트릭스(matrices)를 설계하고, 이를 통해 시공간 교통 상태의 본질적인 의존성을 저랭크 특성으로 변환하여 높은 정확도를 달성합니다.

- **Technical Details**: 본 연구에서 제안하는 방법은 비스듬한 격자 기반의 저랭크 행렬(completion method)을 사용하여 시공간 교통 전파 특성을 포착하고 교통 상태를 정밀하게 재구성합니다. 혹시 모를 데이터 손상에 대비하여 스파스 매트릭스 기반의 이상 탐지 모듈을 개발하여 모델의 강인성을 향상시켰습니다. 또한, 제안된 방법의 계산적 복잡성은 문제의 크기와 관련있으며, 기존 모델처럼 데이터 크기나 하이퍼파라미터 선택에 의존하지 않습니다.

- **Performance Highlights**: 제안된 방법은 Root Mean Squared Error (RMSE)에서 기존 최첨단(SOTA) 방법에 비해 12% 향상된 성능을 보였으며, 강인한 TSE 시나리오에서는 18% 향상된 성능을 보여줍니다. 또한, 이 방법은 기존 방법보다 20배 이상 빠른 속도로 작동 가능하며, 명백한 효율성과 강인성을 입증하였습니다.



### On the Trade-Off between Stability and Fidelity of Gaussian-Smoothed Saliency Maps (https://arxiv.org/abs/2411.05837)
- **What's New**: 이 연구는 Gradient 기반의 saliency maps에서 Gaussian smoothing을 적용하여 안정성을 높이는 방법을 탐구합니다. Smooth-Grad 알고리즘을 통해 주어진 훈련 데이터의 무작위성에 대한 gradients의 안정성을 증대시키는데 초점을 맞춥니다.

- **Technical Details**: 우리는 알고리즘적 안정성(algorithmic stability) 프레임워크를 활용하여 Simple-Grad, Integrated-Gradients 및 Smooth-Grad의 saliency maps의 안정성 오류를 이론적으로 분석합니다. Gaussian smoothing이 훈련 설정의 무작위성에 대한 saliency maps의 안정성을 증가시키기 위한 기여를 입증했습니다.

- **Performance Highlights**: Numerical 실험 결과는 Gaussian smoothing이 gradient-based interpretation maps의 안정성을 증가시키면서, 원본 Simple-Grad map과의 차이를 더할 수 있음을 보여주었습니다. 이 연구는 saliency maps의 안정성(fidelity)과 충실도(stability) 사이의 trade-off를 조명합니다.



### Assessing and Enhancing Graph Neural Networks for Combinatorial Optimization: Novel Approaches and Application in Maximum Independent Set Problems (https://arxiv.org/abs/2411.05834)
- **What's New**: 본 연구는 Combinatorial Optimization (CO) 문제를 해결하는 데 있어 그래프 신경망(Graph Neural Networks, GNNs)의 효과성을 조사하고 최대 독립 집합(Maximum Independent Set, MIS) 문제를 중점적으로 살펴보았습니다. 특히 GNN을 통한 QUBO(Quadratic Unconstrained Binary Optimization) 접근 방식의 개선에 주목했습니다.

- **Technical Details**: GNN은 기존의 히스테리시스 함수에 의존하지 않고 학습한 그래프 구조를 이용하여 CO 문제를 해결합니다. QUBO 비감독 접근법을 통해 초기 예측을 위한 강력한 기반을 제공하며, 자가 학습 능력을 활용하여 그래프 구조 정보를 학습하는 데 기여합니다.

- **Performance Highlights**: 향상된 노드 특성 초기화 및 최적화된 QUBO 기능을 적용하여, GNN은 전통적 알고리즘인 탐욕 알고리즘보다 더 나은 성능을 보였습니다. 또한, 훈련된 지도학습 모델은 MIS 문제에 대해 더 정확한 초기 노드 특성을 제공함으로써 예측 정확도를 개선하였습니다.



### GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models (https://arxiv.org/abs/2411.05830)
- **What's New**: 새로운 코드 완성 기준선인 GitChameleon이 도입되었습니다. 이 데이터셋은 116개의 Python 코드 완성 문제로 구성되어 있으며, 특정 라이브러리 버전에 따라 다릅니다.

- **Technical Details**: GitChameleon 데이터셋은 11개의 인기 Python 라이브러리를 기반으로 하며, 각 문제는 실행 가능한 유닛 테스트와 함께 제공됩니다. 이 기준은 LLM의 성능을 평가하는 데 있어 실행 기반 평가를 강조합니다.

- **Performance Highlights**: 최신 LLM들이 이 데이터셋에서 어려움을 겪고 있으며, GPT-4o는 39.9%의 pass@10을 기록했습니다. 이는 현재 모델의 한계를 강조합니다.



### Utilizing RNN for Real-time Cryptocurrency Price Prediction and Trading Strategy Optimization (https://arxiv.org/abs/2411.05829)
Comments:
          10 pages, 16 figures, 1 table

- **What's New**: 이 연구는 Recurrent Neural Networks (RNN)을 이용하여 실시간 암호화폐 가격 예측과 최적화된 거래 전략을 탐구합니다. 고도의 변동성을 가진 암호화폐 시장의 자연에서, 전통적인 예측 모델은 종종 부족함을 보입니다. 이 연구는 RNN의 장기 패턴 감지 능력을 이용하여 가격 예측의 정확성을 향상시키고 효과적인 거래 전략을 개발할 것을 목표로 합니다.

- **Technical Details**: 이 연구는 Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC) 가격 예측 모델을 개발하며, Long Short-Term Memory (LSTM), Bi-directional LSTM (Bi-LSTM), Gated Recurrent Units (GRU)와 같은 딥 러닝 (Deep Learning) 알고리즘을 사용합니다. 모델의 성능을 평가하기 위해 Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) 등의 성능 지표를 활용합니다.

- **Performance Highlights**: 이 연구는 Keras와 TensorFlow를 사용하여 각 RNN 모델을 개발하고, 80%의 데이터로 훈련한 후, 20%의 테스트 데이터에서 성능을 평가합니다. 모델의 성능 지표로는 MSE, MAE, RMSE, MAPE를 사용하여 가장 작은 오차 값을 가진 모델이 선택됩니다. 세 가지 모델 모두 높은 예측 정확도를 보여주며, 암호화폐 투자가와 거래자들에게 유용한 가격 예측 도구로 활용될 수 있습니다.



### A Theory of Stabilization by Skull Carving (https://arxiv.org/abs/2411.05827)
Comments:
          4 pages, 3 figures

- **What's New**: 이번 연구에서는 3D 게임, 가상 현실 및 영화 제작을 위한 포토리얼 아바타 제작에서 얼굴 움직임의 정확한 안정화를 위한 새로운 접근 방식을 제시합니다. 이 방법은 신경 서명 거리(field) 및 미분 가능한 등면(mesh) 메싱을 활용하여 불규칙한 삼각형 메쉬나 포인트 클라우드에서 두개골 안정화 변환을 직접 계산하여 정확성과 강인성을 현저히 향상시킵니다.

- **Technical Details**: 제안된 방법은 고정된 두개골 모양과 피부 두께에 대한 단순한 가정에 의존하지 않고, 정적 표정 스캔을 위한 안정적인 헐(stable hull) 개념을 도입합니다. 이 헐은 안정화된 스캔의 불리언 교차의 표면으로, 주어진 정적 점 구름이나 삼각형 메쉬에서 각 스캔의 사인 거리장(SDF)으로 변환된 후, 뼈대 좌표계에서 조정이 이루어집니다. 이 과정은 불안정한 표정 데이터의 안정화와 동시에 두개골의 위치를 정확하게 추정합니다.

- **Performance Highlights**: 이번 연구는 기존의 방법들과 비교할 때, 다양한 인구 집단을 대상으로 복잡한 표정에 대한 안정화 성능이 우수함을 보여줍니다. 또한, 각 표현을 조정할 때 두개골과의 정렬이 잘 이루어져 있어 제안된 알고리즘이 신뢰할 수 있는 새롭고 효과적인 접근임을 입증합니다.



### From Pixels to Prose: Advancing Multi-Modal Language Models for Remote Sensing (https://arxiv.org/abs/2411.05826)
Comments:
          10 pages, 1 figure

- **What's New**: 본 논문은 원거리 탐지(remote sensing) 분야에서 다중 모달 언어 모델(MMLMs)의 개발과 응용을 탐구합니다. 위성 이미지를 자연어로 해석하고 설명할 수 있는 모델의 능력에 초점을 맞추고 있습니다.

- **Technical Details**: MMLMs는 일반적으로 이중 인코더 아키텍처(dual-encoder architecture)를 사용하여 시각적 및 텍스트 정보를 통합 처리합니다. Transformer 모델을 활용하여 복잡한 원거리 탐지 데이터를 효과적으로 처리하며, 주목(attention) 메커니즘을 통해 시각 및 텍스트 입력의 중요한 부분에 집중하는 방식으로 작동합니다.

- **Performance Highlights**: 이 모델들은 환경 모니터링, 도시 계획 및 재난 대응과 같은 주요 응용 분야에서 효율적인 정보 추출을 통해 자동화된 지구 관측 분석을 크게 향상시킵니다. 특히, 장면 설명(scene description), 객체 탐지(object detection), 변화 탐지(change detection) 등 다양한 응용 분야에서 그 효과를 입증하고 있습니다.



### Navigating Distribution Shifts in Medical Image Analysis: A Survey (https://arxiv.org/abs/2411.05824)
- **What's New**: 본 논문은 Medical Image Analysis (MedIA)에서 distribution shifts 문제를 다루고 있으며, 이는 다양한 병원이나 지역, 환자 집단으로부터 오는 데이터의 변동성 때문에 발생하는 것이다. 이 연구는 이러한 문제를 해결하기 위한 다양한 deep learning (DL) 접근법을 체계적으로 정리한다.

- **Technical Details**: 본 논문은 DL 모델을 MedIA 시스템에 적용할 때의 distribution shifts 문제를 해결하기 위한 방법론을 제시한다. 구체적으로, Joint Training, Federated Learning, Fine-tuning, Domain Generalization으로 나누어 각 접근법의 적용 시나리오와 기술적 세부사항을 설명한다.

- **Performance Highlights**: 이 연구는 다양한 환경에서의 DL 모델의 유연성과 강인성을 높이는 전략을 제시하며, 각기 다른 의료 환경에서 모델을 성공적으로 활용할 수 있는 방법론을 개발하는 데 중점을 둔다.



### Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks (https://arxiv.org/abs/2411.05821)
Comments:
          16 Pages, 9 Figures

- **What's New**: 이번 연구에서는 비전-언어-행동(Vision-Language-Action, VLA) 모델의 평가 프레임워크와 벤치마크 스위트를 제시하여 다양한 로봇 작업에서 이 모델들을 체계적으로 평가하고자 하였습니다. GPT-4o, OpenVLA, JAT의 세 가지 최첨단 VLA 모델을 Open-X-Embodiment 컬렉션의 20개 데이터셋으로 평가하였으며, 이에 대한 통찰력을 제공하였습니다.

- **Technical Details**: VLA 모델들은 비디오와 언어의 이해를 결합하여 로봇 작업을 수행하는 능력을 보여주고 있으며, 특히 복잡한 조작 작업에서의 성능 차이가 두드러집니다. 이 연구는 로봇 학습 작업에 특화된 평가 분할 및 지표를 가지고 있으며, OpenX 데이터셋을 활용하여 다양한 로봇 플랫폼 및 작업 유형을 포함합니다. 또한, 모델 성능은 행동 공간 및 환경 요인에 민감하게 반응함을 확인했습니다.

- **Performance Highlights**: GPT-4o 모델은 정교한 프롬프트 엔지니어링을 통해 가장 일관된 성능을 보여주었고, 모든 모델이 다단계 계획을 요구하는 복잡한 조작 작업에서 어려움을 겪는 경향이 있으며, 20개의 다양한 데이터셋을 기반으로 한 벤치마크에서 성능을 비교하였습니다. 이 연구는 로봇 시스템의 발전을 위한 중요한 초석이 될 것입니다.



### Guiding Genetic Programming with Graph Neural Networks (https://arxiv.org/abs/2411.05820)
Comments:
          Full version of the same-titled paper accepted at GECCO 2024

- **What's New**: EvoNUDGE라는 새로운 신경진화 방법론을 제안하여, 유전적 프로그래밍(genetic programming)에서 상징 회귀(symbolic regression) 문제 해결을 위한 추가 지식을 도출합니다.

- **Technical Details**: EvoNUDGE는 그래프 신경망(graph neural network, GNN)을 활용해 초기 집단을 생성하고 검색 연산자의 동작을 편향시키는 방식으로 작동합니다. 이 방식은 기존 접근법과 달리 후보 솔루션의 부프로그램 정보를 활용하여 복잡한 문제를 해결하기 위한 검색 가이드를 제공합니다.

- **Performance Highlights**: EvoNUDGE는 다양한 기준선(baselines)보다 유의미한 성과를 보였으며, 전통적인 트리 기반 유전적 프로그래밍과 순수 신경망 변형 모두를 초월해 우수한 결과를 달성했습니다.



### Demo: Multi-Modal Seizure Prediction System (https://arxiv.org/abs/2411.05817)
Comments:
          1 page, 1 figure, Proceedings of the IEEE 20th International Conference on Body Sensor Networks (BSN), October 2024

- **What's New**: SeizNet는 다중 모드 센서 네트워크를 활용하여 간질 발작을 예측하는 혁신적인 시스템입니다. 이 시스템은 깊은 학습(Deep Learning) 기법을 사용하여 실시간으로 높은 정확도의 경고를 제공합니다.

- **Technical Details**: SeizNet은 침습적인 intracranial electroencephalogram (iEEG)와 비침습적인 electroencephalogram (EEG), electrocardiogram (ECG) 센서에서 수집된 데이터를 결합하여 사용합니다. 이 데이터는 실시간 추론을 위해 최적화된 첨단 Deep Learning 알고리즘에 의해 처리됩니다. 이를 통해 개인 정보 보호를 보장하고 데이터 전송을 최소화합니다.

- **Performance Highlights**: SeizNet은 발작 예측에서 97% 이상의 높은 정확도를 달성하였으며, 이식 가능한 장치의 크기와 에너지 제한을 유지하면서도 우수한 성능을 발휘합니다.



### Graph Neural Networks for Financial Fraud Detection: A Review (https://arxiv.org/abs/2411.05815)
Comments:
          17 Pages, 2 Figures

- **What's New**: 이 논문은 Graph Neural Networks (GNNs)의 재정 사기 탐지에 대한 역할을 탐구하며, 기존 GNN 방법론들을 통합된 프레임워크로 분류하고, GNN의 적합성과 실제 배치 시 고려사항을 다룹니다.

- **Technical Details**: 이 리뷰는 GNN이 재정 네트워크 내의 복잡한 관계 및 동적 패턴을 포착하는 데 탁월하다는 것을 보여줍니다. 연구는 100편 이상의 연구를 기반으로 하여 GNN의 다양한 응용 및 배치 가능성을 구조적으로 분석합니다.

- **Performance Highlights**: GNN은 전통적인 사기 탐지 방법을 크게 초월하는 성능을 보이며, 이 리뷰는 GNN의 잠재력을 강조하고 현재의 빈틈을 파악하여 재정 시스템에서의 향후 연구 방향을 제시합니다.



### SkipSNN: Efficiently Classifying Spike Trains with Event-attention (https://arxiv.org/abs/2411.05806)
Comments:
          Published as a research paper at IEEE BigData 2024

- **What's New**: 이 논문은 Spiking Neural Networks (SNNs)의 한계를 극복하기 위해 event-attention 메커니즘을 도입하여 유용한 신호를 강조합니다. 이를 통해 SkipSNN 모델이 개발되었으며, 이 모델은 기존 SNNs에 비해 더욱 높은 계산 효율성과 분류 정확도를 달성합니다.

- **Technical Details**: SkipSNN은 기존 SNN 모델을 확장하여 시간 단계에 따라 두 가지 상태(각성 상태와 동면 상태) 간 전환할 수 있도록 설계되었습니다. 정보가 필요할 때만 계산이 이루어지며, 비필요한 노이즈를 무시하는 방식으로 메모리와 에너지를 절약할 수 있습니다. 새로운 손실 함수로 정확도와 계산 비용 간의 균형을 맞춥니다.

- **Performance Highlights**: SkipSNN은 neuromorphic MNIST와 DVS-Gesture 데이터셋에서 최신 SNN 모델들과 비교하여 더 높은 정확도와 낮은 계산 비용을 기록했습니다. 이러한 성과는 에너지가 제한된 센서 장치에서 특히 중요한 장점으로 작용합니다.



### Similarity-based context aware continual learning for spiking neural networks (https://arxiv.org/abs/2411.05802)
- **What's New**: 본 연구에서는 유사한 작업 간의 유사성을 기반으로 한 Context Aware Spiking Neural Network(SCA-SNN) 지속 학습 알고리즘을 제안합니다. 이 알고리즘은 기존의 지속 학습 알고리즘이 작업을 동등하게 다룬다는 점에서 벗어나, 작업 간 유사성 관계를 활용하여 신경망의 학습 효율성을 개선하는 기법입니다.

- **Technical Details**: SCA-SNN 모델은 현재 데이터와 네트워크 상태를 통합하여 작업 유사성을 평가하는 방법을 설계합니다. 이 평가를 통해 이전 작업에서 유용한 신경세포를 재사용하고 새로운 신경세포를 유연하게 확장하는 원칙을 마련했습니다. 또한, '사용하지 않으면 잊혀진다'는 뇌의 발달 유연성에 착안하여, 새로운 작업에 효과적으로 기여하는 신경세포만을 선택적으로 재사용하는 방법을 구현합니다.

- **Performance Highlights**: SCA-SNN 모델은 CIFAR100, ImageNet 일반화 데이터 세트 및 FMNIST-MNIST, SVHN-CIFAR100 혼합 데이터 세트에서 실험을 통해 뛰어난 성능을 입증했습니다. 이 모델은 에너지 소비를 줄이고 효율적인 신경 세포 배치를 실현하여, 국가 최전선의 스파이킹 신경망 성능을 달성했습니다.



### NeoPhysIx: An Ultra Fast 3D Physical Simulator as Development Tool for AI Algorithms (https://arxiv.org/abs/2411.05799)
Comments:
          7 Pages, 4 Figures

- **What's New**: NeoPhysIx는 3D 물리 시뮬레이터로, 기존의 AI 알고리즘들이 갖는 계산 자원의 한계를 극복하고 1000배 이상의 속도를 구현하였습니다. 이 시뮬레이터는 로봇 시뮬레이션을 위한 혁신적인 방법론을 채택하고 있으며, 점 구름 충돌 감지, 관절 각도 계산 및 마찰력 추정의 간소화된 접근을 통해 성능을 극대화하였습니다.

- **Technical Details**: NeoPhysIx는 단일 코어의 Intel i5 프로세서에서 초당 최대 1백만 프레임을 처리할 수 있는 성능을 자랑합니다. 이 시뮬레이터는 로봇 모델을 질량 포인트의 집합으로 구성하고, 환경의 높이 맵을 사용하여 충돌 감지를 단순화하는 방식을 채택했습니다. 이러한 혁신적인 알고리즘들은 저비용으로도 현실적인 행동을 가능하게 합니다.

- **Performance Highlights**: NeoPhysIx는 18도의 자유도를 가진 다리 로봇의 시뮬레이션을 통해 6개월의 로봇 생애를 단 9시간 만에 처리할 수 있었습니다. 이는 기존의 시뮬레이션 방식들과 비교했을 때 비약적인 효율성을 보여줍니다. 이러한 성과는 AI 개발에 있어 물리적 기반을 갖춘 영역에서의 훈련 과정을 가속화할 수 있는 가능성을 제시합니다.



### Forecasting Company Fundamentals (https://arxiv.org/abs/2411.05791)
Comments:
          24 pages, 9 figures, under review

- **What's New**: 본 논문은 기업 기본 지표(CF)의 예측에 대한 통계적 및 현대적 기계 학습 방법을 철저하게 평가하였습니다. 특히, 22개의 결정론적 및 확률론적 CF 예측 모델을 실제 기업 데이터를 사용하여 비교하였고, 딥 러닝 모델이 고전적 모델에 비해 우수한 성과를 보였음을 발견하였습니다.

- **Technical Details**: 기업 기본 지표(CFs)는 기업의 재무 상태를 요약하는 지표로, 이 데이터의 예측에는 시간적 비정상성(non-stationarity)과 다양한 기업 간의 상호작용, 복잡한 역학이 포함됩니다. 이 연구에서는 딥 러닝 모델과 전통적인 통계 모델의 성능을 비교하고, 예측의 불확실성 추정(uncertainty estimation) 기능이 어떻게 투자 전략에 기여할 수 있는지를 조사합니다. 데이터 선택 및 전처리(data preprocessing)의 효과적인 접근법도 다루어집니다.

- **Performance Highlights**: 딥 러닝 모델은 전통적인 모델들에 비해 예측 성능이 뛰어나며, 인간 분석가의 예측과 유사한 정확도로 자동 예측이 가능하다는 것을 보여주었습니다. 이러한 고품질 예측은 자동화된 주식 배분에 큰 이점을 제공합니다. 최종적으로 전문가의 참여를 통해 성과를 더욱 향상시키고 신뢰성을 높일 수 있는 방법에 대한 논의도 포함되어 있습니다.



### Comparative Analysis of LSTM, GRU, and Transformer Models for Stock Price Prediction (https://arxiv.org/abs/2411.05790)
- **What's New**: 이 논문은 AI(인공지능)를 활용한 주가(Stock Price) 트렌드 예측에 대한 연구를 다루고 있습니다. 2015년부터 2024년까지의 테슬라(Tesla) 자동차 데이터를 모델링하여 LSTM, GRU, Transformer 모델을 비교 분석하였습니다.

- **Technical Details**: 주요 분석 방법으로 LSTM(Long Short-Term Memory), GRU(Gated Recurrent Unit), Transformer 모델을 사용하였으며, 데이터셋은 테슬라의 주가 이력을 포함하고 있습니다. 각 모델의 성능을 분석하여 고유한 특성을 강조합니다.

- **Performance Highlights**: 실험 결과는 LSTM 모델이 94%의 정확도(Accuracy)를 달성했음을 보여주며, 이는 투자자들이 보다 정보에 입각한 결정을 내리고 시장 행태에 대한 통찰력을 얻는 데 기여할 수 있음을 나타냅니다.



### Semantic Information G Theory for Range Control with Tradeoff between Purposiveness and Efficiency (https://arxiv.org/abs/2411.05789)
Comments:
          9 pages and 6 Figures

- **What's New**: 최근 딥러닝(Deep Learning)의 발전은 서로 다른 두 종류의 정보를 동시에 최대화 및 최소화해야 함을 시사합니다. 본 논문에서는 정보 최대-최소(Information Max-Min, IMM) 방법을 제안하였습니다.

- **Technical Details**: IMM 문제를 해결하기 위해 Shannon의 정보 전송 속도-왜곡 함수는 최소 상호 정보(Minimizing Mutual Information, MMI) 및 데이터 압축의 이론적 기초 역할을 하지만, 이는 충분하지 않습니다. 저자는 의미 정보 G 이론(Shannon-Lu 이론)을 제안하였으며, 이는 의미 정보 G 측정 및 정보 속도 충실도 함수 R(G)를 포함합니다. R(G) 함수의 파라미터 해법은 정보 효율성 G/R을 향상시키는 일반적인 방법을 제공합니다.

- **Performance Highlights**: 두 가지 사례를 통해 파라미터 해법이 목적성(즉, 의미적 상호 정보)과 정보 효율성 간의 균형을 최적화하는 데 어떻게 도움이 되는지를 보여줍니다. R(G) 함수는 IMM 방법의 이론적 기초 역할을 할 수 있지만, 딥러닝, 강화 학습(Reinforcement Learning), 제약 제어와의 결합에 대한 추가 연구가 필요합니다.



### News-Driven Stock Price Forecasting in Indian Markets: A Comparative Study of Advanced Deep Learning Models (https://arxiv.org/abs/2411.05788)
Comments:
          7 pages, 9 figures, 1 table

- **What's New**: 이 논문에서는 인도 증시의 주가 예측을 위한 고급 딥러닝 모델과 뉴스 데이터의 감성 분석을 통합하였다. 30년간의 역사적 데이터를 활용하고, 기존의 LSTM, SARIMA 및 Facebook Prophet 모델을 개선하여 예측 정확도를 높이고자 하였다.

- **Technical Details**: 연구는 변수 다중 단계 Long Short-Term Memory (LSTM), Facebook Prophet, LightGBM을 통한 최적화 및 Seasonal Auto-Regressive Integrated Moving Average (SARIMA) 모델을 포함한다. 또한 뉴스 소스와 트윗을 통한 감성 분석을 통합하여 주가 변동에 미치는 영향을 조사하였다.

- **Performance Highlights**: 모델은 Root Mean Squared Error (RMSE)를 통해 성능을 평가하며, 특히 뉴스 데이터에 기반한 예측 정확도가 매우 높다는 점을 강조하였다. 감성 분석과 Bi-LSTM 모델의 조합은 변동성이 큰 시장에서도 예측 성능을 크게 향상시키는 것으로 나타났다.




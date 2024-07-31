### An Empirical Study on Parameter-Efficient Fine-Tuning for MultiModal Large Language Models (https://arxiv.org/abs/2406.05130)
Comments:
          ACL finding 2024

- **What's New**: 멀티모달 대규모 언어 모델(Multimodal Large Language Models, 이하 MLLMs)은 멀티모달 지시 데이터셋으로 파인튜닝(fine-tuning)되면서 뛰어난 성능을 보였다. 그러나 MLLMs의 모든 파라미터를 파인튜닝하는 것이 어려워져, 우리는 파라미터 효율적인 파인튜닝(Parameter-Efficient Fine-Tuning, 이하 PEFT)의 도입을 연구했다. 본 논문에서는 다양한 PEFT 방법 중에서 어댑터(Adapter)가 가장 우수한 성능을 보이며, 커넥터 계층의 파인튜닝이 MLLMs의 성능 향상에 기여한다는 것을 밝혀냈다.

- **Technical Details**: 본 논문은 오픈 소스 MLLMs의 LLM 컴포넌트를 대상으로 네 가지 PEFT 방법(LoRA, IA3, Adapter, Prefix-Tuning)을 사용하여 실증 연구를 수행했다. 또한, 다양한 모델, PEFT 모듈의 파라미터와 위치, 파인튜닝 데이터의 크기, 모델의 안정성, 일반화(generalization), 환각(hallucination) 등에 미치는 영향을 종합적으로 분석했다.

- **Performance Highlights**: 일곱 개의 데이터셋에서 네 가지 PEFT 방법을 평가한 결과, 다음과 같은 주요 성과를 얻었다: 어댑터가 전반적인 성능에서 가장 우수하였으며, 커넥터 레이어를 파인튜닝하면 대부분의 MLLMs에서 성능이 향상되었다. 더 많은 트레이닝 가능한 파라미터는 보지 않은 데이터셋에서 더 나은 성능을 보이며, 적은 파라미터는 이미 본 데이터셋에서 성능을 유지했다. 대규모 데이터셋으로 파인튜닝하면 더 나은 성능을 보이나, 자원이 제한될 경우 중간 크기의 데이터셋을 사용하는 것이 좋다.



### Multi-Head RAG: Solving Multi-Aspect Problems with LLMs (https://arxiv.org/abs/2406.05085)
- **What's New**: MRAG(다중 헤드 검색 증강 생성 모델)은 대규모 언어 모델(LLM)이 다양한 내용이 담긴 여러 문서를 검색해야 하는 쿼리를 처리할 수 있도록 설계되었습니다. 기존의 RAG(검색 증강 생성) 솔루션은 이러한 쿼리를 처리하는 데 어려움을 겪었으나, MRAG는 변환기(Transformer)의 다중 헤드 주의층(multi-head attention layer) 활성화를 활용해 이 문제를 해결합니다. 다양한 주의 헤드가 여러 데이터 측면을 학습할 수 있도록 함으로써 복잡한 쿼리에 대한 검색 정확도를 향상시킵니다.

- **Technical Details**: MRAG는 기본적인 RAG 설계를 향상시키는 간단하지만 강력한 접근법을 제안합니다. 기존의 마지막 층 디코더(decoder) 블록의 활성화를 키로 사용하지 않고, 다중 헤드 주의층의 활성화를 키로 사용하여 다중 측면 문서를 검색합니다. 이러한 다중 측면 임베딩(embedding)을 데이터 항목과 쿼리 표현 모두에 직접 사용합니다. MRAG는 새로운 평가 방법론과 메트릭, 합성 데이터셋 및 실제 사례를 통해 그 효과를 입증합니다. MRAG의 코드와 관련 자료는 공개되어 있으며, RAGAS와 같은 벤치마킹 도구 및 다양한 데이터 스토어 클래스와 쉽게 통합될 수 있습니다.

- **Performance Highlights**: MRAG는 복잡한 쿼리에 대한 검색 정확도에서 기존 RAG 기반보다 최대 20% 향상된 성능을 보여줍니다. 예를 들어, 다중 측면 위키피디아(Wikipedia) 기사 검색에서 20% 향상을 보였습니다. 이러한 다중 측면 임베딩 아이디어는 추가적인 공간 요구 없이 RAG의 성능을 향상시킵니다.



### SUMIE: A Synthetic Benchmark for Incremental Entity Summarization (https://arxiv.org/abs/2406.05079)
Comments:
          24 figures, 4 tables

- **What's New**: 이번 논문에서는 Incremental Entity Summarization (IES) 문제를 다루기 위한 새로운 데이터셋 SUMIE를 소개합니다. 기존 데이터셋들은 이러한 모델들이 실시간으로 엔티티 요약 정보를 업데이트하는 능력을 충분히 시험하지 못했지만, SUMIE는 현실적인 IES 문제들을 잘 드러냅니다. 이를 통해 잘못된 엔티티 연관 및 불완전한 정보 표현 등의 문제를 효과적으로 강조합니다.

- **Technical Details**: SUMIE는 LLM (Large Language Models)을 사용해 완전히 합성된 데이터셋으로, 인기 있는 검색 주제를 기반으로 다양한 속성(attribute)과 합리적인 엔티티 이름을 생성합니다. 또한, 실질적인 데이터 업데이트 시나리오를 반영한 점진적 변화, 충돌 및 반복이 포함됩니다. 데이터셋의 생성 과정은 다양한 스타일과 톤으로 구성된 문단을 생성하여 모델이 다채로운 언어 패턴에 적응하도록 합니다.

- **Performance Highlights**: SUMIE 데이터셋을 사용한 실험 결과, 최신 LLM들은 80.4% 이상의 F1 점수를 달성하는 데 어려움을 겪고 있습니다. 이는 이 과제가 상당한 복잡성을 가진다는 것을 의미합니다. 데이터셋 평가와 측정을 위한 벤치마크와 메트릭스 또한 공개할 예정입니다.



### Are Large Language Models More Empathetic than Humans? (https://arxiv.org/abs/2406.05063)
Comments:
          9 pages, 3 figures. arXiv admin note: text overlap with arXiv:2403.05572

- **What's New**: 본 연구는 최신 대형 언어 모델(LLMs)의 공감적 응답 능력을 인간과 비교하여 평가하는 포괄적인 연구를 제시합니다. 연구에 참여한 모델은 GPT-4, LLaMA-2-70B-Chat, Gemini-1.0-Pro, Mixtral-8x7B-Instruct이며, 이들을 인간 기준선과 비교했습니다.

- **Technical Details**: 본 연구는 1,000명의 참가자를 모집하여, 32가지의 긍정적 및 부정적 감정을 다룬 2,000개의 감정 대화 프롬프트에 대한 응답의 공감적 품질을 평가했습니다. 이를 통해 인간과 네 가지 최첨단 LLMs의 응답을 분석했습니다. 평가 프레임워크는 EmpatheticDialogues 데이터셋을 사용하였으며, 공감의 인지적, 감정적, 자비로운 측면을 포함합니다.

- **Performance Highlights**: 결과는 LLMs가 인간보다 통계적으로 유의하게 더 높은 공감적 응답 능력을 보였다는 점을 나타냅니다. GPT-4는 인간 기준선에 비해 '좋음' 평가가 약 31% 증가하여 가장 공감적인 모델로 나타났으며, LLaMA-2, Mixtral-8x7B, Gemini-Pro도 각각 약 24%, 21%, 10%의 증가를 보였습니다. 일부 LLMs가 특정 감정에 대해 특히 더 나은 응답을 제공한 것으로 나타났습니다.



### Scenarios and Approaches for Situated Natural Language Explanations (https://arxiv.org/abs/2406.05035)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 최근의 대형 언어 모델(LLMs)이 다양한 사용자 상황에 맞추어 자연어 설명(NLE)을 생성하는 능력에 대한 정량적 평가가 부족한 점을 보완하기 위해, Situation-Based Explanation(SBE)라는 새로운 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 100개의 설명 대상(explanandum)과 각 설명 대상에 대해 세 가지 다른 청중 유형(예: 교육자, 학생, 직장인)을 포함합니다. 이를 통해 다양한 사용자 그룹의 정보 요구와 맥락에 맞는 설명의 적합성을 평가할 수 있습니다.

- **Technical Details**: 이 연구에서는 다양한 사전 학습된 언어 모델의 성능을 세 가지 프롬프트 방법 범주: 규칙 기반 프롬프트(rule-based prompting), 메타 프롬프트(meta-prompting), 인컨텍스트 학습 프롬프트(in-context learning prompting)를 통해 평가합니다. 각 설명 대상과 청중 조합마다 사람이 작성한 설명을 포함하여 설명이 얼마나 잘 적응하는지 정량화할 수 있는 유사성 점수와 일치 점수를 계산합니다.

- **Performance Highlights**: 1) 언어 모델은 목표 상황에 더 정확하게 맞는 설명을 생성할 수 있는 프롬프트를 만들어낼 수 있으며, 2) '도움이 되는 지원자'로 지정하는 프롬프트 기법이 situational NLE 작업에 필수적이지 않으며, 3) 인컨텍스트 학습 프롬프트는 LLM이 데모 템플릿을 학습하는 데는 도움이 되지만 추론 성능을 향상시키지 못합니다. SBE와 분석 결과는 상황에 맞춘 자연어 설명 생성을 향한 미래 연구를 촉진합니다.



### Compositional Generalization with Grounded Language Models (https://arxiv.org/abs/2406.04989)
Comments:
          ACL 2024, Findings

- **What's New**: 본 연구는 기존의 의미적 구문 분석(compositional generalization) 연구를 확장하여, 지식 그래프(knowledge graphs)의 패턴에서 언어 모델이 어떤 방식으로 학습하고 일반화하는지를 평가하고자 합니다. 이를 통해 기존 언어 모델의 훈련 가중치에 이미 암묵적으로 내재된 정보에 기반하지 않는 자연어 질문 생성 절차를 개발했습니다.

- **Technical Details**: 연구는 그래프 신경망(graph neural network, GNN)과 언어 모델을 결합하여 지식 그래프를 통한 질문 응답(task)을 수행합니다. 데이터 생성 절차는 대체성(substitutivity), 생산성(productivity), 체계성(systematicity) 세 가지 요소를 목표로 합니다. 이 접근법은 기존의 언어 모델이 다루지 못했던 새로운 길이의 시퀀스와 새로운 조합에 대한 일반화 능력을 평가하기 위해 고안되었습니다.

- **Performance Highlights**: 기존 방법론들이 새로운 길이의 시퀀스 및 학습된 기본 요소의 새로운 조합에 대한 일반화에 어려움을 겪고 있음을 발견했습니다. 이 논문은 언어 모델의 구성적 일반화(compositional generalization)에 대해 실험 연구를 최초로 수행하고, 이 연구 절차를 통해 생성된 데이터셋을 공개하여, 통제된 환경에서 언어 모델을 벤치마킹할 수 있도록 했습니다.



### Language models emulate certain cognitive profiles: An investigation of how predictability measures interact with individual differences (https://arxiv.org/abs/2406.04988)
Comments:
          Accepted at ACL 2024

- **What's New**: 이 연구는 읽기에서 놀라움(surprisal)과 정보 이론적 불확실성(entropy) 효과를 개인 차이를 고려하여 분석한 최초의 사례입니다. 인간의 읽기 시간을 예측하기 위해 다양한 언어 모델(LMs)에서 추정된 놀라움과 정보 이론적 불확실성의 예측력을 조사합니다. 또한, 예측 정확성을 높이기 위해 인지 능력 정보를 통합합니다.

- **Technical Details**: 이 연구에서는 놀라움과 정보 이론적 불확실성을 사용하여 읽기 시간을 예측합니다. 이를 위해 다섯 개의 사전 훈련된 생성적 언어 모델(GPT-2 base와 large, Llama 2 7B와 13B, 그리고 Mixtral)을 사용하여 예측 변수를 포함한 선형 회귀 모델을 구축했습니다. 또한, 이 연구는 개별 차이를 고려하기 위해 InDiCo(Intelligent Differences Corpus)의 데이터를 사용했습니다. 해당 데이터는 언어 사용자의 인지 능력을 평가한 종합적인 심리 측정 결과를 포함하고 있습니다.

- **Performance Highlights**: 놀라움과 정보 이론적 불확실성의 예측력은 인지 점수와의 상호작용 항을 추가함으로써 상당히 향상되었습니다. 일반적으로 높은 인지 능력을 가진 개인은 예측성 효과에 덜 민감함을 보였습니다. 또한, 모든 테스트한 모델은 낮은 언어 지능을 가진 사람들의 처리 행동을 모방하는 경향을 보였습니다.



### MEFT: Memory-Efficient Fine-Tuning through Sparse Adapter (https://arxiv.org/abs/2406.04984)
Comments:
          ACL 24

- **What's New**: 연구진들은 PA(Parallel Adapter)를 활용해 LLMs(Large Language Models)에서 지식 집약적 작업을 위한 효과적인 미세 조정 방법을 제공하는 기술을 새롭게 도입했습니다.

- **Technical Details**: 연구진의 새로운 메커니즘인 MEFT(Mixture of Experts-based Fine-Tuning)는 활성화 희소성을 활용해 FFNs(Feed-Forward Networks) 모델의 일부 뉴런들만 활성화시킵니다. 이렇게 하여 메모리 사용량을 줄이는 한편, CPU 메모리의 큰 용량을 활용합니다. 활성화된 뉴런들만 CPU에서 GPU로 이동하여 계산을 완료하게 됩니다. MoE(Mixture of Experts)-기반 어댑터 구조를 도입해 불필요한 CPU 계산을 줄이고 PCIe 대역폭 문제를 해결했습니다.

- **Performance Highlights**: 실험 결과에 따르면, MEFT는 24GB 메모리 단일 GPU 설정에서도 48GB 메모리 양상이 필요한 설정과 유사한 성능을 보이며, GPU 메모리 사용량을 50% 줄였습니다. 또한, 다른 PEFT(Parameter Efficient Fine-Tuning) 방법들인 Parallel Adapter와 LoRA보다 낮은 자원 조건에서 더 높은 퍼포먼스를 보였습니다.



### Quantifying Geospatial in the Common Crawl Corpus (https://arxiv.org/abs/2406.04952)
- **What's New**: 이 논문은 최근 Common Crawl (CC) 데이터셋에서의 지리공간 데이터의 존재를 조사하며, 광범위한 비라벨 텍스트 데이터에서 학습하는 대형 언어 모델(LLM)의 공간 추론 능력을 분석합니다.

- **Technical Details**: 연구팀은 Gemini라는 강력한 언어 모델을 사용하여 문서 샘플을 분석하고 결과를 수동으로 검토했습니다. 'HTML', 'XML' 같은 전통적인 웹 문서들과 '좌표', '거리지주소' 등의 지리공간 정보를 파악하는 데 주력했습니다.

- **Performance Highlights**: 분석 결과 CC 내 문서 5개 중 1개에서 6개 중 1개가 지리공간 정보를 포함하고 있는 것으로 추정되었습니다. 이러한 지리공간 데이터의 빈도와 특성에 대한 정량적 인사이트를 제공하여 LLM의 공간 인식 능력 연구를 위한 기초 자료를 마련했습니다.



### BAMO at SemEval-2024 Task 9: BRAINTEASER: A Novel Task Defying Common Sens (https://arxiv.org/abs/2406.04947)
Comments:
          9 pages, 8 tables, 5 figures

- **What's New**: SemEval 2024 Task 9, BRAINTEASER는 일반적인 상식을 뛰어넘는 새로운 문제를 언어 모델이 창의적으로 생각할 수 있는 능력을 평가하기 위해 도입되었습니다. 언어 모델의 수평적 사고(lateral thinking) 능력을 자극하는 것을 목표로 합니다. 데이터셋은 선택형 질문으로 구성되어 있으며, 기존의 관리적 사고(Vertical thinking)를 넘어서게 합니다.

- **Technical Details**: BERT 및 RoBERTa Large 모델을 세밀 조정(fine-tuning)한 후, Chain of Thought (CoT)의 무샷(zero-shot) 프롬프트 접근법을 통해 다양한 대형 언어 모델(LLMs)과 함께 작업했습니다. 그 후, ReConcile 기술을 활용하여, 여러 에이전트 간의 '원탁회의' 방식을 통해 합의된 답변을 생성했습니다. 이 기법은 GPT-3.5, Mixtral, Claude와 같은 모델에서 사용되었습니다. 세부적인 설정과 성능 향상 방법은 GitHub 저장소에 있습니다.

- **Performance Highlights**: 문장 퍼즐 부문에서 85%의 정확도를 달성했으며, 이로 인해 순위가 33개 팀 중 11위에 올랐습니다. 이는 무샷 학습 및 창의적 사고를 활용하여 BRAINTEASER 작업에서 언어 모델의 성능을 크게 향상시켰음을 보여줍니다.



### TCMD: A Traditional Chinese Medicine QA Dataset for Evaluating Large Language Models (https://arxiv.org/abs/2406.04941)
- **What's New**: 최근 거대한 언어 모델(LLMs)의 전례 없는 발전은 첨단 의료 분야의 모델들을 확립하며 의료 공동체를 발전시켰습니다. 그러나 의료 데이터셋의 한정된 수집으로 인해 이 분야의 진전을 측정할 포괄적인 벤치마크 몇 개만이 존재합니다. 본 논문에서는 전통 중국 의학(Traditional Chinese Medicine, TCM) 시험 과제를 해결하기 위한 새로운 의료 질문-답변(QA) 데이터셋 'TCMD'를 소개합니다. TCMD는 다양한 도메인의 수많은 질문과 주석이 달린 의료 과목을 포함하여 LLMs의 TCM 도메인 내 능력을 평가하는 데 도움을 줍니다.

- **Technical Details**: TCMD는 중국 국가 의료 자격 시험(Chinese National Medical Licensing Examination)의 여러 선택형 문제와 그에 대한 설명을 포함합니다. 질문들은 공식 시험 매뉴얼의 지침에 따라 필터링되고 조직되어 다양한 의료 주제에 대한 포괄적인 커버리지를 보장합니다. 각 문제에 대해 주석을 달고, 광범위한 분석을 통해 일반 LLMs, 일반 의료 LLMs, 그리고 TCM 도메인 특화 LLMs를 평가합니다.

- **Performance Highlights**: 일반 LLMs가 평균적으로 의료 및 TCM 특화 LLMs보다 더 나은 성능을 보였으며, 응답 일관성은 약간 불만족스러웠습니다. 특히 옵션이 셔플된 질문에 대해 일관성 있는 응답을 예측하는 데 어려움을 겪었습니다. 추가적으로, 옵션이 셔플된 질문에 대한 예측을 투표 메커니즘을 사용하여 앙상블로 처리하면 특정 조건에서 최종 성능을 향상시킬 수 있음이 밝혀졌습니다.



### Through the Thicket: A Study of Number-Oriented LLMs derived from Random Forest Models (https://arxiv.org/abs/2406.04926)
- **What's New**: 이번 논문은 큰 언어 모델(LLM)을 훈련시키는 새로운 방법을 제안하며, 랜덤 포레스트(RF) 앙상블을 활용한 지식 전이를 기반으로 성능을 높이는 방안을 모색합니다. RF의 결정 경로를 자연어로 변환하여 LLM의 분류 및 설명 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 본 연구에서는 RF 앙상블의 각 나무(tree)의 결정 경로를 제안 논리 명제(propositional logic statements)로 변환하여 자연어로 바꾸는 방식을 통해 LLM 훈련에 사용합니다. 또한, LLM의 수치 데이터 처리 능력을 평가하기 위해 전처리 기술(랑 정규화, 값의 언어적 기술 및 관계 인코딩)의 영향을 분석하였습니다. 이 방법은 LLM이 반환한 라벨 및 설명의 정확성을 검증하는 메커니즘도 포함합니다.

- **Performance Highlights**: 제안된 방법은 몇 가지 분류 성능 지표를 통해 LLM 훈련과정에서 발생한 규칙의 정확성을 검증합니다. 또한, CoT(Chain of Thought) 접근 방식을 사용함으로써 설명 가능성과 모델 성능을 잠재적으로 향상시킬 수 있습니다.



### Sexism Detection on a Data D (https://arxiv.org/abs/2406.04892)
Comments:
          Accepted at ACM WebSci 2024 Workshop in DHOW: Diffusion of Harmful Content on Online Web Workshop

- **What's New**: 최근 소셜 미디어의 사용이 증가함에 따라 온라인 혐오 표현도 증가하고 있습니다. 자연어 처리(NLP)와 딥러닝을 기반으로 한 자동화 도구를 사용하여 이러한 유해한 텍스트를 감지하는 기술 또한 빠르게 발전하고 있습니다. 이번 연구는 영향 점수(influence scores)를 활용해 데이터 포인트의 중요성을 추정하고, 성차별(성별에 따른 편견, 고정관념, 차별 검출을 다루는 정제 전략을 설계하는 방법을 소개합니다. 본 연구는 여러 도메인의 데이터셋에서 다수의 인스턴스를 제거하더라도 성능 저하가 크지 않다는 것을 보여줍니다. 그러나, 다른 작업에서 성공적이었던 정제 전략이 유해한 콘텐츠 검출에서는 오히려 클래스 불균형을 악화시킨다는 것도 발견했습니다.

- **Technical Details**: 본 연구는 딥러닝 모델 학습에 많은 주석된 데이터를 필요로 하는 것에 대한 도전 과제에 집중합니다. 연구에서는 영향 점수를 사용하여 훈련 시 데이터 포인트의 중요성을 추정했습니다. 이러한 영향 점수를 활용하여 다양한 정제 전략을 디자인하며 이를 성차별 검출에 적용했습니다. 사용된 영향 점수는 Pointwise V-Information (PVI), Error L2-Norm (EL2N), Variance of Gradients (VoG)입니다. 이 점수들은 각각 정보 기반, 마진 기반, 그라디언트 기반 접근법을 포함합니다. 실험은 세 가지 외부 도메인 데이터셋에서 진행되었으며, 성능 결과는 섹션 5에서 보고되었습니다.

- **Performance Highlights**: 다양한 정제 전략을 사용한 모델을 세 가지 외부 도메인 데이터셋에서 평가한 결과, 대부분의 인스턴스를 제거하더라도 성능 하락이 크지 않았습니다. 그러나, 기존의 자연어 추론(NLI) 작업에서 성공적이었던 데이터 정제 전략은 유해한 콘텐츠 검출에서 클래스 불균형 문제를 더 악화시킬 수 있다는 것을 발견했습니다. 최악의 경우 유해한 클래스가 완전히 사라질 수도 있음을 관찰했습니다.



### A Deep Dive into the Trade-Offs of Parameter-Efficient Preference Alignment Techniques (https://arxiv.org/abs/2406.04879)
Comments:
          Accepted to ACL (Main) 2024

- **What's New**: 대형 언어 모델(Large Language Models, 줄여서 LLMs)은 사전 학습된 수조 개의 토큰에서 특화된 사용자 지침(instruction)이나 선호도에 맞추기 위한 미세 조정을 거칩니다. 사전 학습은 높은 계산 비용으로 인해 대부분의 연구자들이 접근할 수 없지만, 최근 파라미터 효율적인 방법들(예: LoRA, QLoRA)을 통해 미세 조정이 가능해졌습니다. 이 연구는 다양한 정렬(dataset) 및 정렬 메서드(alignment method), 모델의 영향에 관한 광범위한 실험을 통해 일관된 경향과 예상치 못한 발견 내용을 발표합니다.

- **Technical Details**: 주요 연구 축은 다음 세 가지입니다: (i) 정렬 데이터셋(HH-RLHF와 BeaverTails), (ii) 정렬 기법(SFT와 DPO), (iii) 모델(LLaMA-1, Vicuna-v1.3, Mistral-7b, 및 Mistral-7b-Instruct). LoRA와 QLoRA 두 가지 방법을 사용하여 300건 이상의 실험을 통해 파라미터 효율적인 훈련(PEFT)의 다양한 측면을 탐구했습니다. 각 데이터셋과 정렬 기법은 해로움과 유용함의 관점에서 평가되었습니다.

- **Performance Highlights**: 이 연구에서는 일부 일관된 경향과 함께 예상치 못한 결과도 발견되었습니다. 예를 들어, 더 정보성이 높은 데이터가 선호도 정렬에 도움이 되는 경우, 감독된 미세 조정(Supervised Fine-Tuning, SFT)이 선호도 최적화(DPO)를 능가하는 경우, 독특한 선호도에 맞춘 정렬이 다운스트림 작업의 성능을 향상시키는 경우를 관찰했습니다. 이러한 분석 결과는 연구자들에게 효과적인 파라미터 효율적인 LLM 정렬을 위한 중요한 가이드라인을 제공할 것입니다.



### HateDebias: On the Diversity and Variability of Hate Speech Debiasing (https://arxiv.org/abs/2406.04876)
- **What's New**: 이 논문에서는 소셜 미디어에서 증오 발언을 탐지하고 그것의 편향을 완화하기 위한 새로운 벤치마크 HateDebias를 제안합니다. 기존 데이터셋들이 다양한 편향을 충분히 반영하지 못하는 문제를 해결하기 위해 다양한 편향을 가진 기존 증오 발언 탐지 데이터셋을 수집하고, 연속 학습 환경을 따르도록 재조직화하였습니다. 이를 통해 모델이 증오 발언 탐지에 있어서 더 현실적인 환경에서의 성능을 평가할 수 있습니다.

- **Technical Details**: HateDebias는 4가지 편향 속성(나이, 국가, 성별, 민족)을 포함한 23,276개의 증오 발언 텍스트로 구성됩니다. 각 편향은 계속해서 변하는 속성을 가지고 있으며, 연속 학습(Continuous Learning)과 편향 정보 규제(Bias Information Regularization) 및 기억 재생 전략(Memory Replay Strategies)을 기반으로 한 새로운 디바이싱(De-biasing) 프레임워크를 제안합니다. 이 프레임워크는 다양한 편향이 연속적으로 등장하는 시나리오를 시뮬레이션하여 모델이 실제 환경에서 더 나은 성능을 발휘하도록 돕습니다.

- **Performance Highlights**: HateDebias 벤치마크에서 실험한 결과, 제안된 연속 학습 기반 디바이싱 프레임워크가 기존의 몇 가지 기초 모델들(Baselines)에 비해 유의미한 성능 향상을 보여주었습니다. 이는 다양한 편향 속성을 가진 증오 발언을 다루는 실제 응용에서 효과적임을 강조합니다.



### ComplexTempQA: A Large-Scale Dataset for Complex Temporal Question Answering (https://arxiv.org/abs/2406.04866)
- **What's New**: ComplexTempQA는 대규모 템포럴 질문 응답(Temporal Question Answering)을 위한 새로운 데이터셋입니다. 기존 데이터셋인 HOTPOTQA, TORQUE, TEQUILA를 규모와 범위에서 크게 능가하는 1억 쌍 이상의 질문-응답 쌍을 제공하며, 위키피디아(Wikipedia)와 위키데이터(Wikidata)의 자료를 기반으로 한다는 점이 특징입니다. 이 데이터셋은 다양한 주제와 복잡한 질문을 포함하며, 질문의 유형을 속성, 비교, 카운팅으로 분류하는 독특한 분류 체계를 제시합니다.

- **Technical Details**: ComplexTempQA는 1987년부터 2023년까지의 시간 범위를 다루며, 시간 범위별로 메타데이터를 제공합니다. 데이터셋에는 이벤트 간 비교, 템포럴 집계, 멀티홉 추론(multi-hop reasoning)을 포함한 복잡한 질문이 포함되어 있습니다. 메타데이터는 질문이 다루는 시간 영역과 난이도 평가를 포함하여, 시간적 추론 능력을 평가하고 향상시키는 데 도움을 줍니다. 데이터셋 생성은 위키데이터의 사실과 위키피디아에서 추출한 일반적인 질문 유형을 기반으로 대규모로 이루어졌습니다.

- **Performance Highlights**: ComplexTempQA는 템포럴 질문 응답을 위한 가장 큰 규모의 데이터셋으로, 1억 쌍 이상의 질문-응답 쌍을 제공합니다. 다양한 LLM들을 평가하여, 제로 샷(zero shot), 피우 샷(few shot), 리트리벌 어그먼티드 제너레이션(Retrieval-Augmented Generation, RAG) 접근 방식을 사용하여 성능을 측정합니다. 이를 통해 현재 LLM이 시간 정보를 처리하는 능력과 한계를 파악할 수 있습니다.



### The Russian Legislative Corpus (https://arxiv.org/abs/2406.04855)
Comments:
          7 pages, 6 figures, 1 table

- **What's New**: 러시아의 법률 문서에 대한 포괄적인 코퍼스(corpus)가 1991년부터 2023년에 걸쳐 수집되었습니다. 이 코퍼스는 비밀이 아닌 연방 규정과 법률 행위 텍스트 281,413개(176,523,268 토큰)와 이에 대한 메타데이터를 포함하고 있습니다. 원본 텍스트와 형태통사 표기를 준비한 두 가지 버전이 있습니다.

- **Technical Details**: 이 코퍼스는 'Legislation of Russia' 웹사이트에서 웹 스크래핑을 통해 수집되었으며, 각 법률 문서는 XML 파일로 저장됩니다. XML 구조는 Akoma Ntoso 표준을 따릅니다. 형태통사 표기를 위해 MyStem, TreeTagger, MaltParser와 같은 도구를 사용하였으며, 결과는 Universal Dependencies 프레임워크로 저장됩니다.

- **Performance Highlights**: 연간 평균 4.9% 증가한 법률 행위 수와 연간 9.8% 증가한 문서의 양을 보여주는 통계 자료를 포함하고 있습니다. 형태통사 표기를 위한 텍스트 준비 과정에서 법률 텍스트의 특성을 고려한 규칙과 정규 표현식을 사용하여 문서 형식을 통일하였습니다.



### Uncertainty Aware Learning for Language Model Alignmen (https://arxiv.org/abs/2406.04854)
Comments:
          ACL 2024

- **What's New**: 새로운 연구에서는 지침 기반의 대형 언어 모델 (LLMs)을 최적화하기 위해 불확실성 인식 학습(Uncertainty-Aware Learning, UAL) 접근 방식을 제안합니다. 이 방법은 훈련 샘플의 개별 불확실성에 따라 레이블 스무딩(label smoothing)의 값을 적응적으로 설정하는 것입니다.

- **Technical Details**: UAL은 더 높은 능력을 가진 LLM에서 유도된 샘플 불확실성을 도입합니다. 불확실성 값은 학습 과정에서 레이블 스무딩 값을 조정하는 데 사용됩니다. 이를 통해 특징 공간에서 더 나은 토큰 클러스터링을 촉진하며, 이는 모델의 정렬 성능을 향상시킵니다.

- **Performance Highlights**: 광범위한 벤치마크 실험에서 UAL은 표준 감독 학습(Supervised Fine-Tuning, SFT)을 크게 능가했습니다. 특히 고 엔트로피(high-entropy) 작업에서 10.62%, 복잡한 저 엔트로피(low-entropy) 작업에서 1.81% 향상된 성능을 보였습니다. 이는 AlpacaEval 리더보드와 MetaMath 및 GSM8K 벤치마크에서 확인되었습니다.



### Do Language Models Exhibit Human-like Structural Priming Effects? (https://arxiv.org/abs/2406.04847)
Comments:
          ACL Findings 2024

- **What's New**: 이 연구는 문장과 토큰 수준에서 조사가 수행되었으며, 인간과 인간 언어 코퍼스에서 발견된 결과와 일치하는지 여부를 탐구합니다. 연구는 구조적 프라이밍(structural priming) 파라다임을 사용하며, 드문 요소가 포함된 프라임(priming)이 더 강한 프라이밍 효과를 유발하는 역빈도 효과(inverse frequency effect)를 확인합니다.

- **Technical Details**: 구조적 프라이밍은 최근에 노출된 구조가 같은 구조의 처리를 용이하게 하는 현상을 말합니다. 예를 들어, 이중 목적어(double object) 구조를 담은 문장(프라임)에 노출된 후 같은 구조의 문장을 더 잘 생성할 수 있습니다. 연구는 레킽스-문법적 겹침(lexico-semantic overlap)과 프라이밍 효과의 비대칭성을 조사하며, 드문 프라임이 더 강한 프라이밍을 유발한다는 인간의 인식과 유사한 패턴을 발견했습니다.

- **Performance Highlights**: 연구는 언어 모델이 인간의 언어 생성 선호도를 반영하는 체계적인 특성을 학습한다는 것을 보여줍니다. 또한, 언어 모델이 역빈도 효과와 동사 선호도 측면에서 프라이밍 효과를 나타낸다는 것을 입증했습니다.



### FedLLM-Bench: Realistic Benchmarks for Federated Learning of Large Language Models (https://arxiv.org/abs/2406.04845)
Comments:
          22 pages

- **What's New**: 이번 연구에서는 연합 학습을 통한 대형 언어 모델 학습을 위해 현실적인 데이터셋 및 벤치마크가 부족한 문제를 해결하고자 FedLLM-Bench를 제안합니다. 이는 8개의 학습 방법, 4개의 학습 데이터셋, 6개의 평가 지표를 포함하여 포괄적인 테스트베드를 제공합니다. 이 데이터셋은 다국어 데이터와 사용자 선호도를 반영해 실제 세계 시나리오의 속성을 포착합니다.

- **Technical Details**: FedLLM-Bench는 연합 학습 지침 조정(federated instruction tuning)을 위한 3개의 데이터셋(Fed-Aya, Fed-WildChat, Fed-ChatbotIT)과 연합 선호도 조정(federated preference alignment)을 위한 1개의 데이터셋(Fed-ChatbotPA)을 포함합니다. 이 데이터셋들은 38에서 747에 이르는 클라이언트 규모로 나뉘어 있으며, 언어, 품질, 양, 길이, 임베딩, 선호도 등의 다양성을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 연합 학습은 협업 없이 로컬 학습과 비교할 때 일관되게 성능 향상을 가져왔습니다. 또한, 다국어 데이터셋 Fed-Aya를 기반으로 한 탐험적 실험에서는 유사한 언어 간의 협업이 모든 언어 간의 협업보다 더 많은 이점을 가져올 수 있음을 보여주었습니다. 이러한 벤치마크는 새 연구 방향 탐색에 큰 도움이 될 것입니다.



### Revisiting Catastrophic Forgetting in Large Language Model Tuning (https://arxiv.org/abs/2406.04836)
- **What's New**: 이 논문은 주로 대규모 언어 모델(LLMs)이 새로운 데이터를 학습할 때 이전에 습득한 지식을 잊어버리는 'Catastrophic Forgetting (CF)' 현상을 분석합니다. 논문에서는 모델 손실 지형의 평탄도(flatness)와 CF의 상관 관계를 밝히고 이 문제를 해결하기 위해 손실 지형을 평탄하게 만드는 'Sharpness-Aware Minimization' (SAM) 방법을 제안합니다.

- **Technical Details**: 연구진은 손실 지형의 시각화 및 다양한 매트릭스를 통해 모델 손실 지형의 평탄도와 CF 간의 고도로 양의 상관 관계를 확인했습니다. SAM 최적화 방법을 도입하여 이 지형을 평탄하게 함으로써 CF를 완화하고자 했습니다. 세부적으로는 손실 함수의 2D 시각화와 'Surface Curvature (SC)', 'Average Gradient (AG)', 'Mean Absolute Gradient (MAG)' 등의 매트릭스를 사용하여 분석을 실시했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 Alaca, Open-Platypus, Auto-Wiki 데이터셋에서 CF 문제를 효과적으로 완화함을 보여주었습니다. 손실 지형의 평탄도가 증가함에 따라 모델의 성능 저하가 줄어드는 것을 보고했습니다. 특히, SAM을 도입한 방법은 기존의 반-망각(anti-forgetting) 방법과 시너지 효과를 내며, 이를 통해 LLMs의 CF 저항성을 강화할 수 있음을 입증했습니다.



### Annotating FrameNet via Structure-Conditioned Language Generation (https://arxiv.org/abs/2406.04834)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 주어진 의미 구조를 보존하는 문장을 생성하는 능력을 조사합니다. 특히, FrameNet 형식을 사용하여 새로운 문장을 생성하는 프레임워크를 제안하며, 이를 통해 생성된 문장은 인간 검증에서 높은 수용성을 보였습니다.

- **Technical Details**: 프레임-의미(structure) 정보를 조건으로 하여 문장을 생성하는 프레임워크를 소개합니다. 프레임-세만틱 구축(structured annotation)을 활용하여 기존 문장에서 새로운 예로 주석(annotation)을 전이합니다. 구체적으로는 문장의 특정 구간(spans)에 집중하여 주어진 프레임 구조를 따르면서도 인간적으로 수용 가능한 문장을 생성합니다. 오버제너레이트-필터(generation-and-filter) 접근 방식을 사용하여 의미 일관성을 보장합니다.

- **Performance Highlights**: 인간 평가 및 자동화된 지표를 통해 생성된 문장이 기존 접근 방식보다 의도한 프레임-세만틱 구조를 더 충실히 보존함을 확인했습니다. 추가적으로, 생성된 주석(annotation)을 낮은 자원 환경에서 프레임-세만틱 롤(labeling) 훈련 데이터로 사용했을 때 효과적이지만, 높은 자원 환경에서는 효과가 감소했습니다.



### BERTs are Generative In-Context Learners (https://arxiv.org/abs/2406.04823)
Comments:
          21 pages, preprint

- **What's New**: 이번 논문에서는 DeBERTa 모델을 추가적인 훈련 없이 생성 모델로 활용하는 간단한 추론 기법을 제안합니다. 이를 통해 DeBERTa가 GPT-3와 같은 수준, 혹은 그 이상의 in-context learning 능력을 가질 수 있음을 입증하였습니다. Masked language models (MLMs)도 causal language models만큼 in-context learning에 적합함을 보여주면서, 이들 모델 간의 서로 다른 강점을 활용한 하이브리드 훈련 방식의 가능성을 시사하고 있습니다.

- **Technical Details**: 본 연구에서는 기존의 pretrained masked language model을 (generative) in-context learning에 재사용하는 방법을 제안합니다. 추가적인 훈련 없이 입력 토큰 시퀀스의 순서를 조금만 변경하여 이루어집니다. 두 가지 방법으로 문제를 해결합니다: 1) text generation (텍스트 생성)과 2) ranking (순위 매기기). DeBERTa 모델에서 특수한 [MASK]와 [SEP] 토큰을 사용하여 예측 분포를 만듭니다. 특히, [MASK]를 사용하여 텍스트 생성을 반복적으로 수행하는 방식입니다. 이를 통해 간단한 방안이지만, 일부 문제를 해결하기 위해서 추가적인 수정이 필요했습니다.

- **Performance Highlights**: DeBERTa는 텍스트 이해(task understanding)에서는 GPT-3보다 우수했으나, closed-book question answering와 같은 문제에서는 상대적으로 성능이 낮았습니다. 이는 MLMs와 causal language models이 서로 보완적인 훈련 목표를 가지며, 결합할 경우 매우 큰 잠재력을 가질 수 있음을 시사합니다. 또한, MLMs도 in-context learning에서 스케일링 가능성을 보여주었습니다.



### SelfGoal: Your Language Agents Already Know How to Achieve High-level Goals (https://arxiv.org/abs/2406.04784)
Comments:
          Preprint

- **What's New**: 본 논문에서는 SelfGoal이라는 새로운 자동화 접근 방식을 제시합니다. 이 접근 방식은 인간의 사전 지식 및 환경 피드백이 제한된 상황에서 에이전트가 높은 수준의 목표를 달성할 수 있도록 설계되었습니다. SelfGoal의 핵심 개념은 고수준 목표를 체계적으로 분해하고, 환경과의 상호작용 동안 더 실용적인 하위 목표로 나누는 것입니다.

- **Technical Details**: SelfGoal의 작업 방식은 높은 수준의 목표를 적응적으로 더 작은 하위 목표로 분해하여 트리 구조로 형성하는 것입니다. 상호작용 과정에서 가장 유용한 하위 목표를 식별하고 이 구조를 점진적으로 업데이트하면서 목표 달성을 향한 에이전트의 성능을 향상시킵니다. 이는 경쟁적, 협력적, 그리고 피드백이 지연되는 환경에서도 효과적입니다.

- **Performance Highlights**: 실험 결과, SelfGoal은 다양한 과제에서 언어 에이전트의 성능을 크게 향상시켰습니다. 특히 경쟁적, 협력적 및 지연 피드백 환경 모두에서 현저한 성능 개선이 확인되었습니다.



### WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild (https://arxiv.org/abs/2406.04770)
Comments:
          Link: this https URL

- **What's New**: WildBench는 복잡한 실제 사용자 쿼리를 활용해 대형 언어 모델(LLMs)을 벤치마킹하기 위한 자동화 평가 프레임워크입니다. 이 프레임워크는 100만 개 이상의 인간-챗봇 대화 기록에서 1,024개의 과제를 엄선하여 마련되었습니다.

- **Technical Details**: WildBench는 GPT-4-turbo와 같은 고급 LLM을 사용하여 산출 가능한 WB-Reward와 WB-Score라는 두 가지 지표를 도입했습니다. 평가 과정에서 모델 응답의 체계적인 평가를 위해 작업별 점검표가 사용되며, 결과와 비교를 정당화하는 구조화된 설명이 제공됩니다. WB-Reward는 모델 응답 간의 미세한 비교를 통해 5가지 가능한 결과를 생성하며, 길이 편향을 완화하기 위해 간단한 방법도 제안합니다. WB-Score는 개별 모델 출력의 품질을 평가하는 데 사용됩니다.

- **Performance Highlights**: WildBench 결과는 Chatbot Arena의 인간 투표 엘로(Elo) 등급과 강한 상관관계를 나타냈습니다. 특히 WB-Reward는 상위 모델에 대해 피어슨 상관관계 0.98을 달성했으며, WB-Score는 0.95에 도달했습니다. 이는 ArenaHard의 0.91과 AlpacaEval2.0의 0.89를 각각 능가하는 성과를 보여줍니다.



### Think out Loud: Emotion Deducing Explanation in Dialogues (https://arxiv.org/abs/2406.04758)
- **What's New**: 새로운 연구 과제로 EDEN 'Emotion Deducing Explanation in Dialogues'가 제안되었습니다. 이는 대화에서 감정 파악과 유발 원인을 설명하는 텍스트를 생성하여 감정과 원인을 동시에 인식하려는 방법입니다.

- **Technical Details**: EDEN은 기존의 ERD(Emotion Recognition in Dialogues)와 ECED(Emotion Cause Extraction in Dialogues) 과제의 한계를 극복합니다. 모델은 대화 컨텍스트에서 감정 유발 요인을 요약하고, 화자의 내부 활동을 분석한 후 해당 감정을 추론합니다. 이를 위해, 인간이 구성한 두 개의 EDEN 데이터셋(DailyDialogue 및 Friends)을 사용하였습니다. 다양한 모델(기존 Pretrained models, ChatGPT, LLaMA)을 대상으로 한 실험에서 LLMs(Large Language Models)이 더 높은 성능을 보였습니다.

- **Performance Highlights**: PLMs(Pretrained Language Models)은 EDEN 과제에 적합하지 않으며, EDEN은 LLMs의 이유 능력을 활성화하여 더 나은 감정 이해를 달성할 수 있습니다. EDEN을 활용하면 이전 모델보다 더 나은 감정/원인 인식 성능을 얻을 수 있습니다.



### CRiskEval: A Chinese Multi-Level Risk Evaluation Benchmark Dataset for Large Language Models (https://arxiv.org/abs/2406.04752)
Comments:
          28 pages, 5 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 위험 성향 평가를 위한 중국어 데이터셋, CRiskEval을 소개합니다. CRiskEval은 자원 획득, 악의적 협력 등의 위험 성향을 평가하기 위해 고안되었습니다. 새롭게 정의된 위험 분류 체계와 4개의 안전 수준(매우 위험, 중간 위험, 중립, 안전)을 사용하여 7가지 유형의 최첨단 위험에 대한 14,888개의 질문으로 구성되었습니다.

- **Technical Details**: CRiskEval은 다양한 위험 시나리오를 모사하는 질문에 대해 다중 선택형 응답을 제공하여 LLMs의 위험 성향을 정밀하게 측정합니다. 각 질문에는 4개의 응답 선택지가 있으며, 이들은 위험 수준에 따라 수동으로 주석이 달려있습니다. 이러한 데이터셋은 LLMs의 위험 성향을 세밀하게 프로파일링할 수 있도록 돕습니다. 평가 방법으로는 경향성 평가(tendency evaluation)를 사용합니다.

- **Performance Highlights**: 다양한 중국어 대형 언어 모델에 CRiskEval을 적용한 결과, 대부분의 모델이 40% 이상의 위험 성향을 보였습니다. 모델의 크기가 커짐에 따라 자립성, 권력 추구 등의 위험 목표에 대한 경향이 증가하는 경향을 보였습니다. CRiskEval은 초기 자기 인식 및 상황 인식을 갖춘 모델의 위험 성향을 평가하는 데 탁월한 성능을 발휘했습니다. 이는 LLMs의 최첨단 위험 평가를 위한 중요한 기초 데이터를 제공합니다.



### CRAG -- Comprehensive RAG Benchmark (https://arxiv.org/abs/2406.04744)
- **What's New**: CRAG(CRAG; Comprehensive RAG Benchmark)이 최근 소개되었습니다. 이는 4,409개의 질문-답변 쌍과 웹 및 지식 그래프(KG) 검색을 모방하는 모의 API를 이용한 사실 기반 질문 응답(QA) 벤치마크를 제공합니다. CRAG는 다섯 가지 도메인과 여덟 가지 질문 카테고리를 통해 인기 있는 엔티티부터 롱테일(Long-tail) 엔티티까지 다양한 인기도와 시간적 역동성을 반영합니다.

- **Technical Details**: CRAG는 스마트 어시스턴트 사용 사례를 참고하여 4,409개의 QA 쌍을 수집하고 다양한 표현의 질문을 포함시키기 위한 재구성을 통해 현실적이고 신뢰할 수 있는 질문과 답변을 제공합니다. 웹에서 최대 50개의 HTML 페이지와 가상의 260만 개 엔티티로 구성된 KG를 사용하여 다양한 정보를 검색할 수 있도록 모의 API를 제공하는 것이 특징입니다. 세 가지 주요 과제인 웹 검색 요약, 구조화된 데이터 쿼리 및 응답 생성, 그리고 엔드 투 엔드 RAG(E2E RAG)를 통해 RAG 솔루션을 평가합니다.

- **Performance Highlights**: 최신 LLM은 CRAG에서 34% 이하의 정확도를 기록하는 반면, 단순 RAG 통합 시 44% 정답률을 보입니다. 업계 최첨단 RAG 솔루션은 환각 현상 없이 63%의 질문에 답변하지만, 동적 정보나 낮은 인기도, 높은 복잡성을 가진 질문에 대한 정확도는 여전히 낮습니다. 이 평가 결과는 QA 시스템의 신뢰성을 높이기 위한 연구 방향을 제시합니다.



### AICoderEval: Improving AI Domain Code Generation of Large Language Models (https://arxiv.org/abs/2406.04712)
- **What's New**: 최신 arXiv 논문에서는 실제 시나리오에서의 대규모 언어 모델(LLM)의 코드 생성 능력을 평가하기 위한 새로운 데이터셋, AICoderEval을 소개합니다. 이 데이터셋은 HuggingFace, PyTorch, TensorFlow를 기반으로 한 다양한 분야에서의 실제 작업을 포괄하여 LLM의 작업별 코드 생성 능력을 평가하고 향상시킬 수 있도록 설계되었습니다.

- **Technical Details**: AICoderEval은 자연어 처리(NLP), 컴퓨터 비전(CV), 멀티모달 학습 등을 포함한 다양한 도메인의 작업을 포함하며, 코드 생성 작업 및 평가를 위한 테스트 케이스와 완전한 프로그램을 제공합니다. 이를 통해 모델이 특정 라이브러리 API를 활용하는 방식을 학습할 수 있도록 도와줍니다. 또한, CoderGen이라는 에이전트 기반 프레임워크를 제안하여 LLM이 특정 작업 관련 코드를 생성하도록 돕고, 이 프레임워크를 통해 트레이닝 및 테스트 샘플을 자동으로 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, CoderGen이 LLM의 작업별 코드 생성 능력을 크게 향상시킨 것으로 나타났습니다. 원래 모델의 pass@1 성능이 12% 증가했고, ReAct Agent의 경우 9.5% 증가했습니다. 또한, AICoder 모델이 현재의 코드 생성 LLM보다 더 뛰어난 성능을 보이면서 AICoderEval 벤치마크의 높은 품질을 입증했습니다.



### Mixture-of-Agents Enhances Large Language Model Capabilities (https://arxiv.org/abs/2406.04692)
- **What's New**: 이번 연구에서는 여러 대형 언어 모델(LLMs)의 집단적 전문성을 활용하여 자연어 이해 및 생성 능력을 대폭 향상시키는 새로운 접근법을 제안합니다. 이를 위해 'Mixture-of-Agents(MoA)' 방법론을 도입하여 다수의 LLM을 계층적으로 구성하고, 각 계층의 에이전트들이 이전 계층의 출력 결과를 참고하여 응답을 생성하도록 합니다. MoA 모델은 AlpacaEval 2.0, MT-Bench, FLASK 등 여러 벤치마크에서 GPT-4 Omni를 비롯한 기존 최고 성능 모델을 능가하여 최첨단 성능을 달성했습니다.

- **Technical Details**: MoA 구성에서는 각 계층에 여러 LLM 에이전트가 배치되며, 이들 에이전트는 이전 계층의 출력 정보를 보조 정보로 활용합니다. 이를 통해 각 에이전트는 더 개선된 응답을 생성할 수 있습니다. MoA 모델은 레이어마다 다양한 모델의 출력물을 종합하고, 이를 다단계로 반복적으로 개선하여 최종적으로 더 정교한 응답을 도출합니다. 또한, LLM들을 'Proposers(제안자)'와 'Aggregators(결합자)'라는 두 가지 역할로 구분하여 효과적인 협력을 유도합니다. Proposers는 다채로운 참고 응답을 생성하는데 뛰어나며, Aggregators는 여러 모델의 출력을 합성하여 고품질의 단일 출력으로 만듭니다.

- **Performance Highlights**: MoA 프레임워크는 AlpacaEval 2.0에서 65.8%의 새로운 최고 승률을 기록했습니다. 이는 이전 최고 성능을 기록한 GPT-4 Omni의 57.5%를 크게 상회하는 결과입니다. 이와 더불어, MT-Bench와 FLASK 등의 벤치마크에서도 기존 모델들을 능가하며 일관된 성능 상승을 보였습니다.



### MATTER: Memory-Augmented Transformer Using Heterogeneous Knowledge Sources (https://arxiv.org/abs/2406.04670)
Comments:
          ACL2024-Findings

- **What's New**: MATTER라는 새로운 메모리-증강 트랜스포머(Transformer)를 소개합니다. 이 모델은 다중 이종 지식 소스로부터 관련 지식을 검색하고 읽을 수 있도록 설계되었습니다. 기존의 질의응답(QA) 모델들이 단일 지식 소스에만 의존하는 한계를 극복하며, MATTER는 구조가 다양한 지식 소스에서 정보를 가져옵니다.

- **Technical Details**: MATTER는 메모리-증강 QA 모델로, 미리 정의된 길이의 신경 메모리(neural memory)를 통해 지식을 저장합니다. 이 모델은 비구조화된 소스(예: 위키피디아 문단)와 반구조화된 소스(예: QA 쌍)에서 정보를 검색합니다. 이를 통해 문맥의 길이가 줄어들어 계산 비용과 대기 시간을 줄입니다. 또한, MATTER는 주어진 질문과 검색된 신경 메모리를 교차 인코딩(cross-encoding)하여 입력과 문맥을 종합적으로 이해합니다.

- **Performance Highlights**: MATTER는 기존의 효율적인 검색-증강 QA 모델들을 뛰어넘는 성능을 보여주며, 일반적인 읽기-검색(read-and-retrieve) 모델과 비교해도 경쟁력 있는 결과를 기록했습니다. 특히, 추론 단계에서 100배의 처리량을 달성했으며, 이는 FiD 모델보다 월등한 속도를 자랑합니다.



### DiNeR: a Large Realistic Dataset for Evaluating Compositional Generalization (https://arxiv.org/abs/2406.04669)
Comments:
          EMNLP 2023 long paper

- **What's New**: 이번 연구에서는 기존 합성으로 생성된 데이터셋의 한계를 벗어나기 위한 새로운 과제로 DIsh NamE Recognition (DiNeR) task를 제안하며, 이를 위한 대규모 현실적인 중국어 데이터셋을 제작했습니다. 해당 데이터셋은 3,811개의 요리 이름과 228,114개의 레시피를 포함하며, 다양한 언어적 현상을 다룹니다.

- **Technical Details**: DiNeR 과제는 레시피 설명을 기반으로 식재료, 행동(요리 방식), 맛의 조합을 통해 요리 이름을 인식하는 것을 요구합니다. XiaChuFang Recipe Dataset을 기반으로 대규모 데이터세트를 수집하였으며, 기존의 syntactically generated 데이터셋의 한계를 극복할 수 있는 non-synthetic data를 포함하고 있습니다. T5 모델과 GPT-3.5를 베이스라인으로 사용하여 compositional prompting (CP-FT) 및 도메인 적응(continual pretraining)을 수행하였습니다.

- **Performance Highlights**: 새로운 CP-FT 방법과 도메인 적응을 적용함으로써 out-of-distribution (OOD) 사례에서의 F1 점수를 크게 향상시켰습니다. 또한 데이터의 분할과 확장을 통해 모델의 다양한 상황에서의 합성 일반화(compositional generalization) 능력을 평가하고, 데이터 규모와 분포 이동 수준 간의 성능 역스케일링 현상을 입증하였습니다.

- **Introduction**: 본 연구는 요리 설명을 기반으로 다양한 언어적 현상을 포함한 요리 이름을 인식하는 DiNeR 과제를 제안하고, 이를 위해 대규모 중국어 데이터셋을 구축하였습니다. 이전의 합성 데이터셋들이 가진 한계를 극복하고자 하는 새로운 접근법을 제시합니다.



### More Victories, Less Cooperation: Assessing Cicero's Diplomacy Play (https://arxiv.org/abs/2406.04643)
- **What's New**: 이번 논문에서는 보드게임 'Diplomacy'에서 Cicero라 명명된 AI의 커뮤니케이션 능력을 평가합니다. Cicero는 전략적 부분에서 뛰어난 능력을 보이고 있으며, 인간 플레이어들을 초과하는 성과를 보여주었지만, Deception 및 Persuasion과 같은 커뮤니케이션 기술에서는 여전히 한계가 존재합니다.

- **Technical Details**: 논문은 in-game 커뮤니케이션을 Abstract Meaning Representation (AMR)을 통해 추상화하고, 약 24개의 게임을 통해 Cicero와 인간 플레이어 간의 커뮤니케이션을 테스트했습니다. 메시지 분석을 통해 Cicero는 전략적으로는 우수하지만 Deception 및 Persuasion에서 부족하다는 결론을 도출했습니다.

- **Performance Highlights**: Cicero는 게임에서 우수한 성적을 거두었고, 자주 승리하는 모습을 보였습니다. 그러나 AI-인간 커뮤니케이션에서 인간 플레이어들은 Cicero를 인식하고, Cicero의 커뮤니케이션이 주로 전략적인 측면에 의존함을 발견했습니다. Cicero는 여전히 상호연결된 커뮤니케이션 기술에서 인간을 능가하지 못하고 있습니다.



### Large Language Model-guided Document Selection (https://arxiv.org/abs/2406.04638)
Comments:
          9 pages

- **What's New**: 최근 연구에 따르면, 도큐먼트 선택을 신중히 하면 대규모 연산 자원(Compute budget)의 일부만을 사용해도 유사한 모델 품질을 얻을 수 있습니다. 본 연구는 이러한 아이디어를 확장하여 일관된 품질의 도큐먼트를 대규모 웹 크롤링 데이터에서 선택하는 방법을 제안합니다. 대형 언어 모델(LLM)을 도큐먼트 평가자로 활용하여 품질 레이블을 분류기 모델에 증류(distill)한 후, 이를 대규모 웹 크롤링 데이터에 적용합니다. 이로 인해 데이터의 75%를 제거하고 나머지 데이터를 이용해 LLM을 훈련합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 언어 모델을 사용합니다: 대형 명령어 미세 조정 모델(LMlarge)과 소형 사전 훈련 언어 모델(LMsmall). 먼저, LMlarge를 사용해 무작위로 추출한 도큐먼트를 평가하고, LMsmall을 이에 기반하여 품질 레이블로 미세 조정합니다. 그런 다음 LMsmall을 사용해 전체 웹 크롤링 코퍼스를 평가합니다. 이를 통해 필터링에 필요한 연산량을 최소화할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크에서, 필터링을 통해 최대 70%의 연산량(FLOPs)만 사용해도 전체 코퍼스를 훈련한 모델과 유사한 품질을 얻을 수 있음을 확인했습니다. 더 강력한 LLM 라벨러 및 분류기 모델은 더 나은 결과를 제공하며, 모델들의 프롬프트에 대한 민감도가 낮아집니다. 또한, 문맥 학습(In-context learning)은 덜 강력한 라벨링 모델의 성능을 향상시킵니다.



### Low-Resource Cross-Lingual Summarization through Few-Shot Learning with Large Language Models (https://arxiv.org/abs/2406.04630)
Comments:
          7 pages,3 figures

- **What's New**: 크로스-링구얼 요약(Cross-lingual summarization, XLS)는 원문 언어와 다른 대상 언어로 요약을 생성하는 것을 목표로 합니다. 본 논문에서는 Mistral-7B-Instruct-v0.2, GPT-3.5, GPT-4 등의 다양한 모델을 대상으로 몇 샷 학습(few-shot learning) 설정에서 XLS 성능을 조사했습니다. 특히, 낮은 자원 환경에서 GPT-3.5와 GPT-4의 성능이 크게 향상되었습니다.

- **Technical Details**: 실험을 통해 몇 샷 학습이 대형 언어 모델(LLM)의 XLS 성능을 크게 개선할 수 있음을 발견했습니다. GPT-3.5와 GPT-4는 낮은 자원 환경에서 탁월한 성능을 보여주었지만, 오픈 소스 모델인 Mistral-7B-Instruct-v0.2는 제한된 예제에서 효과적으로 적응하지 못했습니다.

- **Performance Highlights**: GPT-3.5와 GPT-4는 낮은 자원 환경에서도 우수한 몇 샷 학습 성능을 보여줬습니다. 반면, Mistral-7B-Instruct-v0.2는 XLS 과제에 효과적으로 적응하지 못해, 몇 샷 학습 전략 및 LLM 아키텍처 디자인에 대한 추가 연구 필요성을 강조했습니다.



### Key-Element-Informed sLLM Tuning for Document Summarization (https://arxiv.org/abs/2406.04625)
Comments:
          Interspeech 2024

- **What's New**: KEITSum이라는 새로운 케이트엘레멘트 정보 기반의 요약 튜닝을 제안합니다. 이 모델은 문서 내 중요한 요소를 식별하고, sLLM(small-scale LLM)을 조정하여 이러한 요소를 포착한 요약을 생성합니다.

- **Technical Details**: KEITSum은 주요 요소, 즉 이름된 엔티티(named entities)와 결론 문장을 식별하여 sLLM을 튜닝합니다. NER(named entity recognition) 메커니즘을 사용하여 엔티티를 추출하고, 사전 훈련된 BERT 기반의 추출 요약기(extractive summarizer)를 사용하여 핵심 문장을 선택합니다. 선택된 요소들은 문서 내에서 강조 표식으로 강조되며, 이 정보를 바탕으로 sLLM을 세밀하게 튜닝합니다.

- **Performance Highlights**: DialogSum과 CNN/Daily Mail 데이터셋을 사용한 실험 결과, KEITSum을 적용한 sLLM은 더욱 높은 관련성과 더 적은 환상을 포함한 고품질 요약을 생성해냈습니다. 특히 긴 대화나 문서를 요약할 때 높은 성능을 보였습니다.



### LawGPT: A Chinese Legal Knowledge-Enhanced Large Language Mod (https://arxiv.org/abs/2406.04614)
Comments:
          Technical Report

- **What's New**: LawGPT는 중국 법률 응용을 위해 설계된 최초의 오픈 소스 대형 언어 모델입니다. 이 모델은 법률 중심의 사전 훈련과 법률 지도 학습 미세 조정을 통해 개발되었습니다. 공개된 코드는 GitHub에서 5.7K 별을 받으며 큰 관심을 끌고 있습니다.

- **Technical Details**: LawGPT는 법률 지식을 포함시키기 위해 대규모의 중국 법률 문서를 활용한 사전 훈련을 거칩니다. 추가로, 법률 지도 학습을 통해 다양한 다운스트림 법률 작업에서 성능을 향상시킵니다. 모델의 두 가지 주요 구성 요소는 'Legal-oriented Pre-training'(법률 지향 사전 훈련)과 'Legal Supervised Fine-tuning'(법률 지도 학습 미세 조정)입니다.

- **Performance Highlights**: LawGPT는 주요 법률 작업에서 오픈 소스 LLaMA 7B 모델을 뛰어넘는 성능을 보였습니다. 이는 법률 과제 수행에서 오픈 소스 모델의 가능성을 제시합니다.



### Learning Task Decomposition to Assist Humans in Competitive Programming (https://arxiv.org/abs/2406.04604)
Comments:
          ACL 2024 Main Conference

- **What's New**: 복잡한 문제를 해결할 때 인간들이 언어 모델(LMs)이 생성한 솔루션을 이해하고 수정하기 어려워하는 문제를 해결하기 위해, 복잡한 솔루션을 더 간단한 하위 작업으로 분해하는 새로운 접근법을 제안합니다. 이 연구는 Assistive Value (AssistV)라는 새로운 목적을 도입하여 사람이 분해된 솔루션을 수정하는 실현 가능성과 속도를 측정합니다.

- **Technical Details**: 이 연구는 LMs가 생성한 초기 솔루션을 여러 하위 작업으로 나누어 더 쉽게 수정할 수 있도록 하는 모델을 학습하는 데 중점을 둡니다. 이 모델은 자동으로 인간이 수정할 수 있는 단위로 솔루션을 분해합니다. 또한, 인간 주석자들이 제공한 자연어 비판에 기반한 분해 향상을 학습하는 세 가지 단계(비판 학습, 개선 학습 및 순위 결정)를 거칩니다.

- **Performance Highlights**: 177시간의 인간 연구에서, 제안된 방법은 비전문가들이 33.3% 더 많은 문제를 해결하게 하고, 그들의 속도를 3.3배(비전문가) 및 2.4배(전문가)로 증가시키며, 비전문가가 비보조 전문가의 수준에 도달하도록 지원합니다. GPT-3.5-Turbo는 인간의 판단보다 62.5% 정확하게 AssistV를 예측하고 GPT-4는 15.6% 더 나은 결과를 보였습니다.



### Extroversion or Introversion? Controlling The Personality of Your Large Language Models (https://arxiv.org/abs/2406.04583)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 성격 통제에 관한 종합적인 조사를 통해 기존 연구의 공백을 메우고자 합니다. 연구진은 지속적인 사전 훈련(Continual Pre-training), 감독된 세부 조정(Supervised Fine-Tuning, SFT), 인간 피드백 기반 강화 학습(Reinforcement Learning from Human Feedback, RLHF), 그리고 추론 단계에서의 프롬프트(prompt)와 같은 여러 방법을 조사했습니다.

- **Technical Details**: 연구는 LLM의 성격 형성에 미치는 요인을 파악하고, LLM의 합성 성격을 효과적이고 안정적으로 제어하는 방법을 중심으로 이루어졌습니다. SFT가 프롬프트를 통한 유도보다 더 높은 제어 성공률을 보였고, RLHF와 지속적 사전 훈련 방법보다 더 효과적임이 밝혀졌습니다. 또한, SFT와 프롬프트의 강점을 결합한 'PISF' 방법을 제안하여, 고효율, 고성공률, 고안정성을 달성했습니다.

- **Performance Highlights**: PISF 방법은 역방향 성격 프롬프트 유도(Reverse Prompt Induction)를 통해서도 안정성과 일관성을 유지하는 모습을 보여줍니다. 연구는 다양한 성격 특성 데이터셋과 이를 통해 구축된 훈련 모델을 활용하여, LLM의 특정 특성과 성격을 평가하기 위한 정량적 지표(ISR, TIE, TSE, PISR, PIE)를 디자인했습니다.



### SC2: Towards Enhancing Content Preservation and Style Consistency in Long Text Style Transfer (https://arxiv.org/abs/2406.04578)
- **What's New**: 이번 연구에서는 긴 텍스트 스타일 변환(long TST)의 과제를 해결하기 위한 새로운 방법인 SC2를 제안합니다. SC2는 스타일 일관성과 콘텐츠 보존 문제를 동시에 해결하기 위해 다층 Joint Style-Content Weighed(JSCW) 모듈과 스타일 일관성 손실(Style Consistency loss)을 설계하였습니다.

- **Technical Details**: JSCW 모듈은 컨볼루션(convolution) 연산을 사용하여 중심 토큰의 스타일 속성(SA, Style Attribute)과 콘텐츠 속성(CA, Content Attribute)의 양을 동시에 평가합니다. 이를 통해 손실 없는 콘텐츠 표현을 학습하고, 이를 여러 층의 JSCW 모듈로 점진적으로 개선합니다. 또한, 스타일 일관성을 유지하기 위해 대조 학습 기반 스타일 일관성 손실을 도입하였습니다. 비자동 회귀(decoder)를 포함한 방법으로 훈련 속도를 가속화합니다.

- **Performance Highlights**: 제안된 SC2 방법은 중국어와 영어 데이터셋에서 광범위한 실험을 통해 기존의 경쟁 모델들보다 상당한 개선을 보였습니다. 이러한 결과는 긴 텍스트 스타일 변환에서 내용 보존과 스타일 일관성을 효과적으로 유지할 수 있음을 증명합니다.



### SpaRC and SpaRP: Spatial Reasoning Characterization and Path Generation for Understanding Spatial Reasoning Capability of Large Language Models (https://arxiv.org/abs/2406.04566)
Comments:
          Accepted at ACL 2024 (Main)

- **What's New**: 이 연구에서는 최신 대형 언어 모델(LLMs)의 공간 추론 능력을 종합적으로 분석합니다. 이를 위해 새로운 공간 추론 특성화(SpaRC) 프레임워크와 공간 추론 경로(SpaRP) 데이터셋을 개발했으며, 이를 통해 공간 관계와 구성 그리고 공간 추론 연쇄의 유용성을 심도 있게 이해할 수 있습니다.

- **Technical Details**: 연구에 따르면, 현재의 최신 LLM들은 SpaRP 데이터셋에서 일관되게 저조한 성능을 보였습니다. 그러나 모델 크기가 커짐에 따라 공간 추론 능력이 크게 향상되었습니다. 예를 들어, Llama-2-70B와 같은 대형 모델과 Llama-2-13B와 같은 소형 모델 모두를 미세 조정(finetuning)하면 F1 점수가 절대적으로 7~32 포인트 향상되었습니다. 또한, 독점 LLM이 상용 LLM에 비해 토폴로지적 공간 이해와 추론에서 크게 우수한 성능을 보였습니다.

- **Performance Highlights**: 모델의 크기가 커질수록 공간 추론 능력이 향상되는 것이 확인되었습니다. Llama-2-70B와 같은 대형 모델은 특히 효과적이었습니다. 미세 조정된 모델들은 공간 추론 경로(SpaRP)를 통해 성능이 일관되게 향상되었으며, 이는 사전 학습된 일반 LLM들이 공간 추론 작업에서 저조한 성능을 보인다는 점을 강조합니다.



### Creating an AI Observer: Generative Semantic Workspaces (https://arxiv.org/abs/2406.04555)
Comments:
          37 pages with appendix, 28 figures

- **What's New**: 이 논문에서는 범죄 보고서와 같은 문서를 읽고 분석하기 위한 새로운 AI 시스템인 Generative Semantic Workspace(GSW)를 소개합니다. GSW는 'Operator'와 'Reconciler'로 구성되며, 대규모 언어 모델(LLM, Large Language Models)의 발전을 활용해 전통적인 정해진 어휘 레이블 집합이 아닌 생성적 스타일의 의미론적 프레임워크를 만듭니다.

- **Technical Details**: Operator는 주어진 텍스트 부분(Cn)을 기반으로 배우 중심의 의미 지도('Workspace instance' Wn)를 생성합니다. Reconciler는 현재의 워크스페이스 인스턴스와 '작업 메모리(Mn*)' 간의 차이를 해결하여 업데이트된 Mn+1*를 생성합니다. Reconciler는 복합 네트워크로 모델링된 워크스페이스 인스턴스와 최신 Operator 생성 인스턴스 간의 유사성을 계산하여 세 가지 선택지를 갖습니다: (i) 현재의 정보를 유지하고 새로운 정보를 제거, (ii) 특정 어휘 부분만 제거하고 새로운 정보로 대체, (iii) 기존 및 새로운 어휘 샘플 모두 유지.

- **Performance Highlights**: GSW는 다중 문장 의미 추출 작업에서 FST, GLEN, BertSRL 등 유명한 기준점을 약 94%로 상회하며, NLI-BERT 대비 약 15%, QA 대비 약 35% 성능 향상을 보여줍니다. 이는 GSW가 실제 인간 관찰자(Observer)를 모방하여 개별 의도를 이해하고 미래 행동을 예측하는 첫 단계가 될 수 있음을 시사합니다.



### Label-Synchronous Neural Transducer for E2E Simultaneous Speech Translation (https://arxiv.org/abs/2406.04541)
Comments:
          Accepted by ACL 2024 Main Conference

- **What's New**: 이번 연구에서는 동시 음성 번역(Simultaneous Speech Translation, SST)과 관련된 새로운 LS-Transducer-SST 모델을 소개했습니다. 이 모델은 레이블-동기 신경 변환기(Label-synchronous Neural Transducer)로, 새로운 Auto-regressive Integrate-and-Fire (AIF) 메커니즘을 사용해 번역 토큰을 동적으로 생성합니다. 이 메커니즘은 대기 시간을 조절할 수 있는 기능을 포함하여, 성능과 지연 시간의 균형을 맞출 수 있습니다.

- **Technical Details**: LS-Transducer-SST는 Auto-regressive Integrate-and-Fire (AIF) 메커니즘을 사용해 축적된 프레임-레벨 가중치를 기반으로 번역 토큰을 언제 생성할지 결정합니다. 이 메커니즘은 레이블-레벨 타겟-사이드 인코더 표현을 자동회귀적으로 추출합니다. 또한, 예측 네트워크(Prediction Network)는 일반 언어 모델(LM)처럼 동작해 단어 재배열 문제 및 데이터 부족 문제를 경감시킵니다. 지연 시간 제어가 가능한 AIF는 훈련 중뿐만 아니라 디코딩 중에도 성능-지연 시간의 균형을 맞출 수 있도록 합니다. 디코딩 과정에서, 청크 기반 증분 조인트 디코딩을 통해 검색 공간을 확장하고 번역 품질을 높입니다.

- **Performance Highlights**: LS-Transducer-SST는 Fisher-CallHome Spanish (Es-En)와 MuST-C En-De 데이터셋에서 기존의 인기 있는 SST 방법들보다 더 나은 성능-지연 시간 균형을 보여주었습니다. 예를 들어, CAAT 대비 유사한 지연 시간에서 BLEU 점수가 3.1/2.9 포인트 (Es-En/En-De) 상승했으며, Wait-k 대비 평균 지연 시간이 1.4초 줄어들면서도 유사한 BLEU 점수를 기록했습니다.



### llmNER: (Zero|Few)-Shot Named Entity Recognition, Exploiting the Power of Large Language Models (https://arxiv.org/abs/2406.04528)
- **What's New**: 최근 NLP에서의 발전을 위한 Python 라이브러리 llmNER가 소개되었습니다. 이 라이브러리는 대형 언어 모델 (LLM)을 활용하여 zero-shot 및 few-shot 네임드 엔티티 인식 (NER)을 수행할 수 있도록 설계되었습니다. 사용자 친화적인 인터페이스를 통해 프롬프트 구성, 모델 쿼리, 결과 파싱을 간편하게 처리할 수 있습니다. 또한 prompt engineering을 효율적으로 할 수 있게 돕습니다.

- **Technical Details**: llmNER는 자연 언어 처리에서 네임드 엔티티 인식(NER)을 쉽게 수행할 수 있도록 프롬프트 구성과 결과 파싱 과정을 단순화한 파이썬 라이브러리입니다. 이 라이브러리는 먼저 사용자로부터 정의된 엔티티와 예시를 텍스트 프롬프트로 컴파일하고, 결과를 머신 리더블 한 객체로 반환합니다. 사용자는 다양한 프롬프트 방법과 응답 형태 파서를 선택할 수 있으며, POS(Parts of Speech) 증강도 지원됩니다.

- **Performance Highlights**: llmNER는 두 가지 NER 작업을 통해 소프트웨어의 유연성을 검증했습니다. 다양한 도메인에서도 높은 성능을 보였으며, 특히 사전 학습된 모델이 없는 상황에서의 빠른 프로토타입 제작이나 인간 주도 주석 전 단계에서 유용할 수 있습니다.



### Proofread: Fixes All Errors with One Tap (https://arxiv.org/abs/2406.04523)
Comments:
          8 pages, 3 figures, 2 tables

- **What's New**: 새로운 연구는 Gboard의 사용자 타이핑 경험을 혁신적으로 개선하는 'Proofread' 기능을 소개합니다. 이 기능은 서버 측 LLM(Large Language Models)을 활용하여 문장 수준과 단락 수준의 타이핑 오류를 한 번의 클릭으로 수정할 수 있습니다.

- **Technical Details**: Proofread 시스템은 데이터 생성, 메트릭 설계, 모델 튜닝, 모델 배포의 네 부분으로 구성됩니다. 데이터는 사용자의 입력을 시뮬레이션하기 위해 신중하게 설계된 에러 합성 프레임워크를 통해 생성됩니다. 메트릭은 LLM을 기반으로 한 문법 오류 존재 여부와 동일한 의미 여부를 확인하여 모델 품질을 측정합니다. 'Supervised Fine Tuning (SFT)'와 'Reinforcement Learning (RL) tuning'의 두 단계 튜닝 접근법이 사용되었습니다. RL 튜닝 단계에서는 글로벌 보상과 직접 보상을 제안하여 모델 성능을 더욱 향상시켰습니다.

- **Performance Highlights**: 사람이 라벨링한 데이터셋에서 실험 결과, 튜닝된 PaLM2-XS 모델은 85.56%의 높은 품질 비율을 달성했습니다. 이 기능은 현재 Pixel 8 기기 사용자들에게 제공되고 있으며, 수천 명의 사용자가 매일 이 기능을 사용하고 있습니다. 지연 시간을 줄이기 위해 양자화(quantization), 버킷 키(bucket keys), 입력 분할(input segmentation), 추측 디코딩(speculative decoding) 등을 사용하여 성능을 최적화했습니다.



### NATURAL PLAN: Benchmarking LLMs on Natural Language Planning (https://arxiv.org/abs/2406.04520)
- **What's New**: NATURAL PLAN은 자연어로 표현된 3개의 주요 작업(여행 계획, 회의 계획, 일정 관리)을 포함하는 현실적인 계획 벤치마크를 소개합니다. 본 연구는 Google Flights, Google Maps, Google Calendar와 같은 도구의 출력을 모델의 컨텍스트로 제공하여, LLM(large language model)의 계획 능력을 평가합니다. 이를 통해 도구 사용 환경 없이도 LLM의 계획 능력을 평가할 수 있습니다.

- **Technical Details**: NATURAL PLAN은 여행 계획(Trip Planning), 회의 계획(Meeting Planning), 일정 관리(Calendar Scheduling)의 3가지 계획 작업을 포함합니다. 각 작업에서는 Google Flights API를 통해 얻은 실제 데이터를 제공합니다. 예를 들어, 여행 계획 작업에서는 도시 간의 항공 연결 정보를 제공하여 모델이 여행 일정을 계획하도록 합니다. 벤치마크 데이터셋은 도구 데이터를 사용하여 다양한 제약 조건을 추가하여 생성됩니다.

- **Performance Highlights**: 최신 모델들은 NATURAL PLAN에서 어려움을 겪었습니다. 예를 들어, 여행 계획 작업에서 GPT-4와 Gemini 1.5 Pro는 각각 31.1%와 34.8%의 해결율을 기록했습니다. 문제의 복잡성이 증가할수록 모델 성능은 급격히 감소했습니다. 모델들이 10개의 도시가 포함된 문제에서 5% 미만의 성능을 기록한 것은 자연어로 된 계획 작업에서의 큰 격차를 시사합니다.



### To Distill or Not to Distill? On the Robustness of Robust Knowledge Distillation (https://arxiv.org/abs/2406.04512)
Comments:
          Accepted at ACL'24 main

- **What's New**: 이 연구는 아랍어 방언을 포함한 자동 음성 인식(ASR)의 난제를 해결하기 위해 지식 증류(knowledge distillation) 기법을 활용하여 대형 모델에서 소형 모델로 지식을 전이하는 방법을 제안합니다. 또한, 다섯 가지 저평가된 아랍어 방언을 포괄하는 새로운 인간 주석 데이터셋을 소개하고 이를 통해 모델의 성능을 평가했습니다.

- **Technical Details**: 연구진은 OpenAI의 Whisper와 Meta의 SeamlessM4T와 같은 대규모 다국어 ASR 모델에서 소형 학생 모델로 지식을 증류했습니다. 증류 과정에서는 대형 모델의 성능을 유지하면서 계산 비용을 줄이는 데 초점을 맞췄습니다. 도입된 새로운 방언 데이터셋을 통해 다양한 모델을 평가했으며, 주요 오차 유형을 분석하여 모델 성능의 한계를 파악했습니다.

- **Performance Highlights**: 최고의 증류된 모델은 전체 성능(45.0% WER)에서 SoTA(State-of-The-Art) 모델(SeamlessM4T-large-v2, WER=47.0%)과 교사 모델(Whisper-large-v2, WER=55.1%)보다 뛰어났습니다. 또한, 새로운 방언 데이터에서 평균 성능(56.9% WER) 역시 다른 모든 모델보다 우수했습니다.



### Time Sensitive Knowledge Editing through Efficient Finetuning (https://arxiv.org/abs/2406.04496)
Comments:
          Accepted to ACL 2024 main conference

- **What's New**: 이 논문은 Large Language Models (LLMs)의 지식 편집(KE)에 관한 새로운 접근법을 제안합니다. 기존의 locate-and-edit 방법의 한계를 지적하며, Parameter-Efficient Fine-Tuning (PEFT) 기법을 대안으로 탐구합니다. PEFT를 이용한 지식 업데이트와 주입이 locate-and-edit 기술보다 효율적이고, 시간적 지식 편집에 더 나은 성능을 보이는지를 분석합니다.

- **Technical Details**: PEFT 기법을 활용하여 LLaMA-7B, Falcon-7B, Mistral-7B 등의 기본 LLM들을 미세 조정(fine-tuning) 합니다. 구체적으로 LoRA와 P-tuning 방법을 사용하여 요소를 추가하고, 이들을 통해 크로스 엔트로피 손실 함수를 최소화합니다. 또한, 새로운 데이터셋인 ChronoEdit를 큐레이션하여 약 15,000개의 시간에 민감한 사실 편집 예제를 포함시켜, 현실 세계에서의 KE 성능을 평가합니다.

- **Performance Highlights**: PEFT 접근법은 locate-and-edit 기술보다 시간에 민감한 지식 편집에 있어서 더 나은 성능을 보였습니다. 특히, 중간 레이어의 미세 조정이 다중 홉 질문(multi-hop questions) 에 대한 LLM의 성능 향상에 큰 영향을 미쳤습니다. 이를 통해 기존의 locate-and-edit 방법이 가진 추론 한계를 극복할 수 있음을 검증하였습니다.



### Automatic Bug Detection in LLM-Powered Text-Based Games Using LLMs (https://arxiv.org/abs/2406.04482)
Comments:
          Accepted for publication in Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 활용한 텍스트 기반 게임의 버그를 자동으로 식별하는 새로운 시스템을 제안합니다. 이는 플레이어의 게임 로그를 분석하여 논리적 결함 및 게임 밸런스 문제를 감지하여 게임의 완성도를 높이는 데 도움을 주는 방법을 제공합니다.

- **Technical Details**: 본 연구에서는 LLM을 사용하여 플레이어의 게임 로그를 분석해 자동으로 버그를 탐지하는 두 단계의 프로세스를 설정했습니다. 첫 번째 단계에서는 LLM에게 게임 디자이너가 의도한 게임 로직을 기반으로 플레이어의 진행 상황을 요약하여 일관된 형식으로 정리하게 합니다. 두 번째 단계에서는 이러한 요약을 수집하여 각 장면의 완성도를 평가하고, 낮은 완성도를 보이는 장면을 잠재적 문제 지점으로 표시합니다. 이 과정에서 GPT-4 모델을 사용하며, 고유의 '시나리오'와 '장면' 개념을 사용하여 게임 로직을 세분화하고, 플레이어의 진행을 구조화된 형식으로 요약합니다.

- **Performance Highlights**: 제안된 시스템은 최근 발매된 텍스트 기반 미스터리 게임 'DejaBoom!'에 적용하여 평가되었습니다. 이 게임에서는 GPT-4를 활용하여 모든 인게임 텍스트를 생성하며, 플레이어가 폭탄 해체 키트를 찾아 폭탄을 해체하는 과정을 통해 해결해야 합니다. 연구 결과, 제안된 방법이 기존의 비구조적 LLM 방식보다 버그를 더 효과적으로 식별하는 것으로 나타났으며, 게임 논리 및 밸런스 문제를 자동으로 탐지하는 데 유의미한 성과를 보였습니다.



### PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning (https://arxiv.org/abs/2406.04478)
Comments:
          NAACL 2024

- **What's New**: PromptFix라는 새로운 백도어 제거 전략을 제안합니다. 기존 방법과 달리, 모델 파라미터를 수정하지 않고 소프트 토큰 성질을 이용해 적대적 최적화를 통해 백도어를 효과적으로 제거합니다.

- **Technical Details**: PromptFix는 백도어가 삽입된 NLP 모델을 유지한 채, 두 세트의 소프트 토큰을 추가하여 트리거와 그에 대응하는 토큰을 각각 근사합니다. 이러한 소프트 토큰은 적대적 최적화를 통해 백도어 설정을 유동적으로 조정하며, 백도어를 찾고 모델 성능을 유지하는 균형을 가능하게 합니다.

- **Performance Highlights**: 다양한 백도어 공격 실험에서 PromptFix의 효과가 입증되었습니다. 도메인 변이가 있는 상황에서도 고성능을 유지하여, 알 수 없는 데이터 소스에서 사전 훈련된 모델에 PromptFix의 적용 가능성을 보여주고 있습니다.



### Multi-Label Classification for Implicit Discourse Relation Recognition (https://arxiv.org/abs/2406.04461)
Comments:
          ACL2024 Finding

- **What's New**: 이 연구는 기존의 단일 레이블 예측 방식이 아닌, 다중 레이블 분류(multi-label classification) 방식을 통해 암묵적 담화 관계 인식(Implicit Discourse Relation Recognition, IDRR)을 더욱 효과적으로 처리하는 방법을 제안합니다. PDTB-3 데이터셋에서 다중 레이블을 활용한 첫 번째 연구로서 이 접근 방식이 담화 관계의 복잡성을 보다 정확하게 파악할 수 있음을 보여줍니다.

- **Technical Details**: PDTB-3에 대한 자료 분석 및 다양한 다중 레이블 분류 프레임워크를 탐구하였습니다. 연구에서 RoBERTa 모델을 텍스트 표현 학습에 사용했으며, [CLS] 토큰 위에 분류 헤드(classification head)를 추가하여 다중 레이블 분류를 수행했습니다. 데이터셋은 크로스 밸리데이션(cross-validation)을 통해 12개의 폴드로 나누어 학습, 개발, 테스트로 파티셔닝하였고, 주요 평가 지표로 F1 점수(F1 scores)를 사용했습니다.

- **Performance Highlights**: 세 가지 다중 레이블 분류 기법을 비교한 결과, 다중 레이블 분류기가 기존의 단일 레이블 분류기를 능가하는 성능을 보였습니다. 각 방법의 평가 결과는 macro-averaged F1 점수, Precision(정밀도) 및 Recall(재현율)을 기준으로 종합 분석되었으며, 단일 레이블 예측 성능에 비해 저조하지 않음을 확인했습니다.



### Evaluating the Smooth Control of Attribute Intensity in Text Generation with LLMs (https://arxiv.org/abs/2406.04460)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 최근 발표된 논문은 대형 언어 모델(Large Language Models, LLMs)을 사용하여 텍스트의 속성 강도를 부드럽게 조절하는 방법을 제안합니다. 텍스트 생성에서 속성의 강도(예: 감정, 형식성, 설명의 상세도)를 상황에 맞게 조절하는 것은 매우 중요합니다. 저자들은 '부드러운 제어'(smooth control)를 위해 속성 강도와 컨텍스트 관련성을 평가하는 새로운 프레임워크를 도입했습니다. 이를 위해 Elo rating system과 GPT-4를 활용한 평가 메트릭을 개발했습니다.

- **Technical Details**: 본 논문에서는 LLM의 부드러운 제어를 위해 두 가지 비훈련 방법을 소개합니다: (1) 의미 전환자를 이용한 프롬프트 사용, (2) 내부 모델 표현을 수정하는 방법입니다. 이러한 방법은 5가지 다른 속성에 대해 다양한 모델을 대상으로 평가되었습니다. 속성 강도와 컨텍스트 관련성을 수량화하기 위해 저자들은 Elo rating system과 GPT-4를 사용하여 자동 평가 파이프라인을 구축했습니다. 이를 통해 인류 평가와 잘 맞아떨어지는 결과를 얻을 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 모델 크기가 부드러운 제어 성능에 부정적인 영향을 미칠 수 있으며, 프롬프트 방법이 내부 표현 수정 방법과 거의 동등하거나 더 나은 성능을 보이는 것으로 나타났습니다. 이러한 평가 프레임워크와 데이터셋은 공개되어 있어 추가 연구에 활용될 수 있습니다.



### MAIRA-2: Grounded Radiology Report Generation (https://arxiv.org/abs/2406.04449)
Comments:
          44 pages, 20 figures

- **What's New**: 최근 발표된 논문에서는 '그라운디드 리포트 생성'(grounded report generation)이라는 새로운 과제를 소개하며, 이를 위해 'RadFact'라는 새로운 평가 프레임워크도 제안했습니다. 이 프레임워크는 큰 언어 모델(LLMs)을 활용하여 개별 생성된 문장의 사실성(factuality)과 공간적 위치의 정확성을 평가합니다. 기존의 보고서 생성 과제를 확장하여 이미지에 개별적인 발견 사항을 구체적으로 위치시키는 기법을 도입했습니다.

- **Technical Details**: 논문은 MAIRA-2라는 새로운 대형 멀티모달 모델을 소개했습니다. 이 모델은 radiology-특화 이미지 인코더와 LLM을 결합하여, 흉부 X-레이에 대한 그라운디드 리포트를 생성하도록 훈련되었습니다. MAIRA-2는 현재 정면 이미지, 현재 측면 이미지, 이전 정면 이미지 및 이전 보고서, 그리고 현재 보고서의 Indication, Technique, Comparison 섹션을 포함한 보다 포괄적인 입력을 사용합니다. 이를 통해 리포트의 질을 높이고 'hallucination'(잘못된 생성문)을 줄이는 효과를 보여주었습니다.

- **Performance Highlights**: MAIRA-2 모델은 MIMIC-CXR 데이터셋에서 새로운 최고 성능을 기록했으며, 그라운디드 리포트 생성의 가능성을 입증했습니다. 이는 정확하게 radiology 리포트를 생성하는 것은 물론, 개별적인 발견 사항을 이미지 내에서 구체적으로 위치시키는 기능을 갖추고 있습니다.



### TexIm FAST: Text-to-Image Representation for Semantic Similarity Evaluation using Transformers (https://arxiv.org/abs/2406.04438)
Comments:
          19 pages, 33 figures

- **What's New**: NLP의 주요 목표는 텍스트로부터 의미 있는 표현(Representation)을 생성하는 것입니다. 이 논문에서는 관여하던 텍스트의 표현을 주기 위해, 메모리 용량과 차원을 획기적으로 줄이는 새로운 'Text-to-Image' 메커니즘, TexIm FAST를 소개합니다. TexIm FAST는 셀프-슈퍼바이즈드 (self-supervised) Variational Auto-Encoder(VAE)와 Transformer를 결합하여 고정된 길이를 가지는 표현을 생성함으로써 메모리 공간을 75% 이상 줄이고, 다운스트림 모델에서의 복잡성을 감소시킵니다.

- **Technical Details**: TexIm FAST는 기존의 텍스트에서 이미지로의 변환 방식들과 다르게, 입력된 텍스트를 플픽스 길이의 이미지 표현으로 변환합니다. 이 방법은 CNN-TSLFN (Selective Learn-Forget Network) 기반의 VAE를 사용하여 텍스트의 맥락적 의미 정보를 포괄적으로 캡처하고, 시퀀스의 길이와 무관하게 균일한 차원의 이미지로 인코딩합니다. VAE에는 후방 붕괴(Posterior Collapse) 문제를 해결하기 위해 가중치 완화 메커니즘이 적용되었습니다.

- **Performance Highlights**: TexIm FAST는 Semantic Textual Similarity(STS) 작업에서 기존 방법에 비해 정확도가 6% 향상되는 성과를 보였습니다. MSRPC, CNN/Daily Mail, XSum 데이터셋에 대한 평가에서, TexIm FAST가 다양한 길이의 시퀀스를 효과적으로 비교할 수 있는 특별한 능력을 보여주었습니다. 또한 고정된 이미지 표현을 생성함으로써, 학습 모델의 파라미터 수를 줄이고 메모리 공간도 75% 이상 절약되었습니다.



### MoralBench: Moral Evaluation of LLMs (https://arxiv.org/abs/2406.04428)
- **What's New**: 인공지능 분야에서 대형 언어 모델(LLMs)이 다양한 응용 프로그램에서 중요한 역할을 하고 있는 가운데, 이 논문은 LLMs의 도덕적 추론 능력을 평가하기 위한 새로운 벤치마크를 소개합니다. 이 논문은 현실 세계의 복잡성을 반영한 다양한 윤리적 딜레마와 시나리오를 포함하는 포괄적 데이터 세트를 처음으로 발표했습니다.

- **Technical Details**: 이 연구는 LLMs의 도덕적 정체성을 평가하기 위해 윤리학 학자들의 정성적 통찰과 정량적 분석을 결합한 다각적인 접근 방식을 사용합니다. 도덕적 추론 능력을 평가하기 위해 만들어진 벤치마크 데이터 세트와 메트릭이 포함되어 있으며, 이는 인간 윤리 기준과의 정렬, 문맥적 민감성 및 뉘앙스를 고려합니다.

- **Performance Highlights**: 여러 주요 LLM들에 대한 벤치마크 적용 결과, 모델 간의 도덕적 추론 능력에 상당한 차이가 있음을 발견했습니다. 이러한 결과는 도덕적 추론을 LLMs의 개발 및 평가에서 고려해야 할 필요성을 강조하며, 연구를 통해 발견된 편향과 한계를 해결하기 위한 지속적인 연구의 필요성을 제기합니다.



### Exploring the Latest LLMs for Leaderboard Extraction (https://arxiv.org/abs/2406.04383)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)인 Mistral 7B, Llama-2, GPT-4-Turbo 및 GPT-4.o가 인공지능(AI) 연구 논문에서 리더보드 정보를 추출하는 효율성을 조사한다. 논문은 DocTAET (문서 제목, 요약, 실험 설정, 표 형식 정보), DocREC (결과, 실험, 결론), DocFULL (전체 문서)의 세 가지 맥락 입력 유형을 탐구하며, 연구 논문에서 (Task, Dataset, Metric, Score) 쿼드러플을 생성하는 데 있어서 각 모델의 강점과 한계를 밝혀낸다.

- **Technical Details**: LLM의 정보 추출(IE) 성능은 입력 중 제공되는 맥락에 의해 크게 영향을 받는다. 맥락은 연구 논문에서 (Task, Dataset, Metric, Score) 데이터를 추출하는 데 사용되는 텍스트의 일부를 의미한다. 본 연구는 DocTAET, DocREC, DocFULL 세 가지 맥락 입력 유형을 탐구해 최적의 길이와 구체성의 맥락을 식별하는 것을 목표로 한다. CLEF 2024의 공유 과제 'SOTA? Tracking the State-of-the-Art in Scholarly Publications'에 참여해 문서 전체 텍스트에서 리더보드 데이터를 추출하는 시스템을 개발한다.

- **Performance Highlights**: 논문은 최신 LLM인 Mistral 7B, Llama-2, GPT-4-Turbo 및 GPT-4.o의 성능을 평가해 구조적 요약을 생성하고 리더보드 유무를 분류한다. 첫 번째 연구 질문(RQ1)은 구조적 요약 생성 및 리더보드 유무 분류에 가장 정확한 성능을 제공하는 LLM을 찾는 것이다. 두 번째 연구 질문(RQ2)은 정밀도와 다른 성능 메트릭 간의 균형을 조사해 가장 좋은 성능을 제공하는 LLM을 찾는 것이다.



### Phased Instruction Fine-Tuning for Large Language Models (https://arxiv.org/abs/2406.04371)
Comments:
          Review version, to be appear at ACL 2024 Findings

- **What's New**: 이번 논문에서는 Instruction Fine-Tuning(IFT)을 한층 발전시킨 '단계적 지시 미세조정(Phased IFT)' 방법을 제안합니다. 이는 사전 학습된 언어 모델의 단순한 다음 단어 예측에서 복잡한 지시 환경으로의 적응을 점진적으로 향상시키기 위한 접근법입니다. 추가적으로, GPT-4를 사용하여 각 지시의 난이도 점수를 산출하고, 이를 난이도에 따라 여러 단계의 하위 데이터 세트로 분할하여 순차적으로 업트레이닝(uptraining)을 진행합니다.

- **Technical Details**: Phased IFT 방법은 지시 데이터의 난이도를 평가하는 과정을 포함하며, 이를 통해 데이터 세트를 점차적으로 난이도가 증가하는 여러 단계로 나눕니다. 초기에는 낮은 난이도의 하위 데이터 세트에서 표준 supervised loss를 사용한 훈련을 시작하고, 훈련이 끝난 모델 체크포인트를 더 어려운 하위 데이터 세트로 확장하는 방식으로 진행됩니다. 이 과정은 가장 어려운 데이터 세트까지 반복됩니다. 또한, 광범위한 실험을 통해 Llama-2 7B/13B 및 Mistral-7B 모델에서 Phased IFT의 성능을 검증했습니다.

- **Performance Highlights**: 6개의 벤치마크 데이터 세트를 활용한 성능 평가에서 Phased IFT는 전통적인 One-off IFT 방식에 비해 상당한 승률 향상을 보였습니다. 특히, 단계별 업트레이닝 과정을 거친 모델이 무작위 단계 업트레이닝 또는 One-off IFT 방식보다 더 우수한 성과를 나타내며, 점진적 정렬 가설이 실험적으로 입증되었습니다. 종합적으로, Phased IFT 방법은 사전 학습된 언어 모델의 지시 수행 능력을 크게 향상시키는 효과적인 접근법임을 보여주었습니다.



### Large Language Model Confidence Estimation via Black-Box Access (https://arxiv.org/abs/2406.04370)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 반응에 대한 신뢰도를 추정하기 위한 간단하고 확장 가능한 프레임워크를 제안합니다. 이 프레임워크는 입력 프롬프트와 그에 따른 출력을 기반으로 다양한 방법으로 프롬프트를 조작하고 이러한 조작에서 나온 변동성을 특징으로 사용하는 새로운 기능들을 설계합니다. 이를 통해 로지스틱 회귀(logistic regression) 모델을 학습시켜 신뢰도를 추정합니다.

- **Technical Details**: LLM의 신뢰도를 추정하기 위해 6가지 블랙박스 전략을 제안합니다. Stochastic Decoding(SD), Split Response Consistency(SRC) 등은 LLM의 출력을 변동시키는 다수의 샘플을 생성하고, 이러한 샘플들 간의 일관성을 분석하는 방법입니다. 각 프롬프트 변형에서 나온 변동성을 기반으로 특징(feature)을 구성하고, 로지스틱 회귀 모델을 사용하여 정답 여부를 예측합니다. 예측된 확률이 신뢰도 추정을 나타냅니다.

- **Performance Highlights**: 제안된 프레임워크는 TriviaQA, SQuAD, CoQA, Natural Questions 같은 벤치마크 데이터셋에서 기존의 블랙박스 신뢰도 추정 접근법을 능가합니다. 특히 AUROC(수신자 조작 특성 곡선 하의 면적)에서 10% 이상의 성능 향상을 보였습니다. 또한, 특정 LLM에 대한 신뢰도 모델이 다른 LLM에도 제로샷(zero-shot)으로 잘 일반화됨을 보여주었습니다.



### SocialNLP Fake-EmoReact 2021 Challenge Overview: Predicting Fake Tweets from Their Replies and GIFs (https://arxiv.org/abs/2406.04368)
- **What's New**: Fake-EmoReact 2021 챌린지는 NAACL 2021에서 열렸으며, 트윗의 진위 여부를 예측하는 과제를 다룹니다. 이 챌린지는 EmotionGIF 데이터셋에서 증강된 GIF 카테고리와 답변 컨텍스트를 사용합니다.

- **Technical Details**: Fake-EmoReact 데이터셋은 453,000개 이상의 트윗을 포함하며, 모든 트윗은 진위 여부 레이블이 지정되어 있습니다. 트윗과 답변 텍스트와 함께 GIF 반응을 포함합니다. 모델은 주어진 레이블 없이 트윗과 해당 GIF 반응을 사용해 그 진위 여부를 예측해야 합니다. 트레이닝, 개발, 평가 세트로 나누어집니다.

- **Performance Highlights**: 24개의 팀이 등록하였고, 5개의 팀이 최종 평가에 제출했습니다. 최우수 팀은 F1 점수 기준으로 93.9점을 달성했습니다. DeBERTa, BERT, RoBERTa 등을 포함한 다양한 머신러닝 및 딥러닝 모델들이 사용되었습니다.



### 3D-GRAND: Towards Better Grounding and Less Hallucination for 3D-LLMs (https://arxiv.org/abs/2406.05132)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 언어와 3D 인식을 통합하여 물리적 세계와 상호 작용할 수 있는 인공지능과 로봇의 개발을 목표로 합니다. 특히, '3D-GRAND'라는 대규모 데이터셋을 소개하며, 6.2백만 개의 장면-언어 지침과 짝을 이룬 40,087개의 가정 장면을 포함하고 있습니다. 이 데이터셋은 3D 장면과 언어 사이의 밀집된 연결을 제공하여 3D-LLMs의 성능을 향상시키고 환각(hallucination)을 줄이는 데 기여합니다.

- **Technical Details**: 연구팀은 '3D-GRAND' 데이터셋을 사용하여 3D-LLMs의 학습을 진행하였으며, 이를 통해 언어와 3D 장면 간의 밀집된 연결을 달성했습니다. 또한 환각(hallucination)을 체계적으로 평가하기 위한 '3D-POPE'이라는 벤치마크를 소개하여, 다양한 모델 간의 공정한 비교를 가능하게 했습니다. '3D-POPE'는 3D-LLMs가 객체의 존재 여부를 평가하는 데 사용하는 표준 프로토콜을 제공합니다.

- **Performance Highlights**: 실험 결과, '3D-GRAND' 데이터셋을 사용한 학습은 3D-LLM의 환각을 크게 줄였으며, 밀집된 데이터 기반의 가르침이 3D-LLM의 연결 능력을 크게 향상시켰습니다. 또한 대규모 데이터셋으로 학습된 모델이 실제 3D 스캔에서도 양호한 성능을 보였습니다. 이 연구는 시뮬레이션에서 실제 적용까지의 전이 가능성을 시사하며, 저비용 및 지속 가능한 3D 데이터 스케일링의 미래를 열어줍니다.



### On Ambiguity and the Expressive Function of Law: The Role of Pragmatics in Smart Legal Ecosystems (https://arxiv.org/abs/2406.05084)
Comments:
          50 pages, 6 Figures, first presented at the 31st Congress of General Linguistics of the University of Barcelona (UB, CLUB31), October, 2023. To be published in the Catalan Linguistic Series as a chapter of the volume edited by Jordi Fortuny, Pau Francesch and Lluis Payrato (eds.), Ambiguity: an interdisciplinary approach. Barcelona: Edicions de la Universitat de Barcelona, 2025

- **What's New**: 이 논문은 법의 모호성, 화용론(Pragmatics), 법적 생태계, 그리고 법의 표현 기능을 중심으로 한 에세이입니다. 논문은 크게 두 부분으로 나뉘며, 화용론과 컴퓨팅(Computing)에 대해 다룹니다.

- **Technical Details**: 첫 번째 부분은 법적 분야에서 언어적 및 인지적 화용론 (Pragmatics) 관점에서 모호성을 다루고 있으며, 두 번째 부분은 인간 중심 설계(Human-centered design)와 인공지능(AI) 관점에서 규칙의 개념과 이를 준수하는 방법을 모델링(MODELING)하는 방법을 다룹니다. 이는 스마트 법적 생태계(Smart Legal Ecosystems, SLE)의 발판을 형성하는 데 필요합니다.

- **Performance Highlights**: 논문에서는 유럽연합(EU)의 Industry 4.0 프로젝트인 OPTIMAI를 예로 들어 아키텍처, 정보 흐름, 스마트 생태계를 설명합니다. OPTIMAI는 인공지능 및 가상화를 통해 제조 공정을 최적화하고 결함이 없는 제조(Zero-defect manufacturing)를 목표로 합니다.



### I2EDL: Interactive Instruction Error Detection and Localization (https://arxiv.org/abs/2406.05080)
Comments:
          Accepted at IEEE RO-MAN 2024

- **What's New**: 본 연구는 Vision-and-Language Navigation in Continuous Environments (VLN-CE) 과제에서 사용자 오류를 다루지 않는 기존 접근법을 개선하기 위한 새로운 접근법을 제시합니다. Interactive VLN in Continuous Environments (IVLN-CE)이라는 새로운 과제를 도입하여 에이전트가 탐색 도중 사용자의 지시 오류를 감지하고 상호작용할 수 있도록 합니다.

- **Technical Details**: 본 연구에서 제안된 Interactive Instruction Error Detector and Localizer (I2EDL)는 사전 학습된 모듈을 활용하여 탐색 중 지시 오류를 감지하고, 텍스트 입력과 과거의 관찰을 교차 참조하여 오류를 집어냅니다. 이를 통해 에이전트는 사용자가 명확한 지시 오류를 수정하도록 질의할 수 있습니다.

- **Performance Highlights**: 제안된 I2EDL은 지시 오류가 포함된 데이터셋에서 평가되었으며, Success weighted by Interaction Number (SIN)이라는 새로운 측정 지표를 도입했습니다. 이 지표는 탐색 성능과 상호작용 효과성을 반영하여, 성공률을 높이면서 상호작용 횟수를 최소화합니다. 평가 결과, I2EDL은 무작위로 상호작용하는 에이전트보다 더 효과적인 성능을 보였습니다.



### Bootstrapping Referring Multi-Object Tracking (https://arxiv.org/abs/2406.05039)
- **What's New**: 새로운 연구는 사람의 자연어 지시로 다중 객체를 감지하고 추적하는 'Referring Multi-Object Tracking (RMOT)' 작업에 중점을 둡니다. 주요 기여로는 이전 데이터셋보다 훨씬 다양한 언어적 표현이 포함된 대규모 데이터셋 'Refer-KITTI-V2'의 도입, 3단계 반자동 라벨링 방식, 그리고 기존 방법들보다 향상된 성능을 보이는 'TempRMOT' 프레임워크 개발입니다. 소스 코드와 데이터셋은 공개되어 있습니다.

- **Technical Details**: 이 연구는 'Refer-KITTI-V2'라는 큰 규모의 데이터셋을 도입하여 RMOT 작업을 시작합니다. 이는 2,719개의 수동 주석과 9,758개의 표현으로 이루어져 있으며, 617개의 서로 다른 단어로 구성되어 있습니다. 연구에서는 세 단계의 반자동 라벨링 파이프라인을 통해 언어 항목(class, color, action 등)을 자동으로 수집하고, 특정 규칙에 따라 초기 언어 프롬프트를 생성하며, LLM(Large Language Model)을 통해 더 많은 언어 프롬프트를 생성합니다.

- **Performance Highlights**: 새로운 TempRMOT 프레임워크는 DETR(Detection Transformer)과 유사한 프레임워크를 기반으로 하는 쿼리 기반의 시간적 향상 모듈을 도입하여 영상 시퀀스에서 다중 객체를 추적하는 동안 더 나은 성능을 제공합니다. 실험 결과, Refer-KITTI와 Refer-KITTI-V2 데이터셋에서 각각 +3.16%와 +4.04% HOTA 지표를 달성하며 기존 최첨단 기법을 능가하는 성과를 보였습니다.



### LLM-based speaker diarization correction: A generalizable approach (https://arxiv.org/abs/2406.04927)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 사용하여 자동 음성 인식(ASR) 도구에서 생성된 대화 기록을 후처리(post-processing) 단계에서 발화자 구분(diarization) 정확도를 개선하는 방법을 제안합니다. Fisher 말뭉치(corpus)를 사용하여 LLM을 미세 조정한 후 성능을 평가하며, 다양한 ASR 도구의 아웃풋에 대해 일반화 가능한 시스템을 개발하기 위해 앙상블 모델을 제안합니다.

- **Technical Details**: 연구에서 대형 언어 모델(LLMs)을 Fisher 말뭉치로 미세 조정하여 사용하였습니다. 이를 위해 AWS, Azure, WhisperX 등의 ASR 도구를 사용하여 대화 기록을 테스트하였고, 각 도구에 대해 개별적으로 미세 조정된 모델을 사용했습니다. 다중 모델의 무게(weight)를 결합한 앙상블 모델을 개발하여 다양한 ASR 도구 간의 일관성 있는 발화자 구분을 가능하게 했습니다.

- **Performance Highlights**: 미세 조정된 LLM들은 동일한 ASR 도구로 생성된 기록에서 발화자 구분 정확도를 눈에 띄게 향상시켰습니다. 특히, 앙상블 모델은 개별 ASR 도구의 성능을 초과하며, ASR 도구에 구애받지 않는 일반화된 발화자 구분 정확도를 달성할 가능성이 높음을 보였습니다.



### XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Mod (https://arxiv.org/abs/2406.04904)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 이번 논문에서는 다중 언어 Zero-shot Multi-speaker TTS (ZS-TTS) 시스템인 XTTS를 소개합니다. 기존 모델들이 주로 고자원 언어(few high/medium resource languages)에 한정되어 있었던 반면, XTTS는 16개 언어를 지원하여 저자원 언어(low/medium resource languages)의 응용 가능성을 확장합니다.

- **Technical Details**: XTTS는 Tortoise 모델을 기반으로 하여 다중 언어 학습(multilingual training)을 가능하게 하고, 음성 복제(voice cloning)를 개선하며, 더 빠른 학습과 추론(faster training and inference)을 가능하게 하는 여러 가지 새로운 수정 사항들을 도입했습니다. VQ-VAE, GPT-2 인코더, HiFi-GAN vocoder 등 주요 구성 요소로 구성되어 있으며, 한국어, 일본어, 중국어 언어를 위해 텍스트를 로마자로 변환하는 기능도 포함되어 있습니다.

- **Performance Highlights**: XTTS는 16개 언어에서 state-of-the-art(SOTA) 성능을 달성했으며, 크로스 언어 TTS(cross-language TTS)를 수행할 수 있습니다. 또한 기존 모델 대비 더 적은 학습 데이터와 더 빠른 샘플링 속도를 자랑합니다. XTTS 모델과 체크포인트는 Coqui TTS 및 Hugging Face의 GitHub 저장소에서 공개되어 있습니다.



### Seeing the Unseen: Visual Metaphor Captioning for Videos (https://arxiv.org/abs/2406.04886)
- **What's New**: 새로운 연구는 비디오에 존재하는 은유를 설명하는 '비디오 은유 캡션 생성(Video Metaphor Captioning)'이라는 새로운 VL(비전-언어) 과제를 도입했습니다. 이를 위해 705개의 비디오와 2115개의 인간 작성 캡션으로 구성된 수작업 데이터셋을 공개하고, 은유 생성의 창의성을 평가하기 위한 새로운 지표 'Average Concept Distance(ACD)'를 제안했습니다.

- **Technical Details**: 연구진은 적은 양의 프리트레이닝 데이터로 비디오 은유를 이해하기 위해 사전 학습과 미세 조정을 거친 새로운 저자원 비디오 은유 캡션 생성 모델인 GIT-LLaVA를 제안했습니다. 이 모델은 한정된 데이터로도 SOTA 비디오 언어 모델과 유사한 성능을 달성합니다. 또한, 기존 비디오 언어 모델의 한계를 분석하고 우리의 데이터셋, 모델, 벤치마크 결과를 공개했습니다.

- **Performance Highlights**: GIT-LLaVA 모델은 기존 SOTA 영상 언어 모델과 비교 가능하거나 대등한 성능을 보여줍니다. 하지만, 기존 영상 언어 모델들은 비디오 은유를 깊게 이해하는 데 한계를 보입니다. 실험 결과와 분석을 통해 이러한 성능 차이를 뒷받침하는 데이터를 제공합니다.



### InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environmen (https://arxiv.org/abs/2406.04882)
Comments:
          Submitted to CoRL 2024

- **What's New**: InstructNav는 새로운 유형의 지시 내비게이션(instruction navigation) 시스템으로 다양한 지시를 따르는 능력을 최초로 제시합니다. InstructNav는 훈련 없이 또는 사전 구축된 지도 없이도 여러 가지 내비게이션 작업을 수행할 수 있습니다.

- **Technical Details**: InstructNav는 Dynamic Chain-of-Navigation(DCoN)이라는 접근 방식을 도입하여 서로 다른 유형의 내비게이션 지시를 통합하는 계획을 수립합니다. 이와 더불어, Multi-sourced Value Maps를 사용하여 로봇이 수행 가능한 경로로 전환할 수 있도록 합니다. DCoN은 내비게이션 중 새로운 환경을 탐색하면서 지시와 랜드마크를 동적으로 업데이트하여 계획을 세웁니다.

- **Performance Highlights**: InstructNav는 R2R-CE 작업에서 zero-shot 성과를 달성하며, 기존의 많은 작업 훈련 방법들을 능가합니다. 또한 zero-shot Habitat ObjNav에서 10.48%, demand-driven navigation(DDN)에서 86.34% 향상된 성과를 보였습니다. 실제 로봇 실험에서도 다양한 실내 장면에 대해 높은 견고성을 입증했습니다.



### Digital assistant in a point of sales (https://arxiv.org/abs/2406.04851)
- **What's New**: 최근 소매 환경에서 디지털 어시스턴트(Digital Assistant)를 배치하여 고객 참여와 서비스 효율성에 미친 영향을 평가하는 연구가 발표되었습니다. 이 연구는 다국어 지원 및 고급 대화 기능을 통해 사용자 상호작용을 개선하는 방안을 탐구합니다. 특히 고빈도 소매 환경에서 디지털 어시스턴트를 통합하여 고객 서비스 품질과 운영 효율성을 향상시키는 효과를 평가합니다.

- **Technical Details**: 이 연구는 음성 사용자 인터페이스(Voice User Interface, VUI)를 갖춘 디지털 어시스턴트를 소매 환경에 배치하고, 다양한 서비스 시나리오에서 고객 상호작용을 최적화하는 데 중점을 둡니다. 3개월간의 실험 기간 동안 Unity 3D 소프트웨어를 사용하여 3D 디지털 캐릭터를 생성하고, 상호작용성을 높이기 위해 동작 센서 등을 활용했습니다. 주요 기능으로는 Sales Pokes, Conversation Engine, Phone Recommendations, Multilingual Capabilities, Feedback Loop 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, 디지털 어시스턴트는 고객 상호작용에 다양한 영향을 미쳤으며, 서비스 환경에서의 이 기술의 잠재력을 밝히는 통찰을 얻었습니다. 특히 고객 만족도와 운영 효율성 측면에서 긍정적인 영향을 미친 것으로 나타났습니다. 이러한 결과는 향후 연구 및 실천에 중요한 시사점을 제공합니다.



### Zero, Finite, and Infinite Belief History of Theory of Mind Reasoning in Large Language Models (https://arxiv.org/abs/2406.04800)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models; LLMs)의 이론적인 정신상태 추론 기법(Theory of Mind; ToM)을 평가하고 확장하기 위한 새로운 개념, 분류 및 프레임워크를 제안합니다. 특히, 다양한 신념 역사(belief history)를 고려한 다중 라운드 텍스트 기반 게임 'Pick the Right Stuff'를 벤치마크로 개발했습니다.

- **Technical Details**: 신념 역사는 세 가지로 분류됩니다: 제로 신념 역사(Zero Belief History), 유한 신념 역사(Finite Belief History), 무한 신념 역사(Infinite Belief History). 제로 신념 역사는 과거 신념 정보를 고려하지 않고 최신 신념을 식별하는 것을 의미하며, 유한 신념 역사는 제한된 과거 신념 정보를 이용해 최신 신념을 식별하는 것을 의미합니다. 무한 신념 역사는 무한한 과거 신념 정보를 관리하는 것을 뜻하며, 주로 미래 연구에서 다뤄질 예정입니다.

- **Performance Highlights**: 제로 신념 역사 조건에서 모든 LLM 모델들이 유한 신념 역사 조건보다 일관되게 더 나은 성능을 보였습니다. 또한, 작은 파라미터 크기를 가진 두 모델이 큰 파라미터 크기를 가진 모든 모델보다 더 우수한 성능을 보였습니다. 이는 LLM 모델의 파라미터 크기를 늘리는 것이 무조건 성능 향상에 효과적인지에 대한 질문을 제기합니다.



### PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction (https://arxiv.org/abs/2406.04746)
- **What's New**: 이번 논문에서는 텍스트에서 이미지로 생성(Text-to-Image Generation) 작업에서 질의(Query)의 난이도를 분석하는 첫 연구로서, 이를 인간의 판단에 기반해 평가한 최초의 데이터셋을 소개합니다.뿐만 아니라, 이미지 검색(Image Retrieval)에서도 동일한 질의의 난이도를 평가하여, 텍스트에서 이미지로의 프롬프트 및 질의 성능 예측을 위한 첫 번째 벤치마크를 제안합니다. 이 벤치마크는 10,000개의 질의로 구성되어 있으며, 다양한 프리젠더(pre-generation), 프리트리벌(pre-retrieval), 포스트젠더(post-generation), 포스트리트리벌(post-retrieval) 성능 예측기를 통해 비교할 수 있는 기반을 제공합니다.

- **Technical Details**: Microsoft Common Objects in Context (MS COCO) 데이터셋의 캡션을 k-means 클러스터링 알고리즘과 BERT 기반의 문장 임베딩 모델을 사용해 10,000개의 대표적인 프롬프트를 선정했습니다. 그런 다음, Stable Diffusion과 GLIDE 모델을 사용해 프롬프트마다 2개의 이미지를 생성했습니다. 또한, 인간 평가자들이 125만 개의 주관적 레벨 판단(annotation)을 통해 텍스트-이미지 생성의 성능을 평가하였습니다. 이러한 데이터를 바탕으로 Prompt and Query Performance Prediction (PQPP) 벤치마크를 구축했습니다.

- **Performance Highlights**: 프롬프트 및 질의 난이도 예측 실험 결과, 텍스트에서 이미지로의 생성과 검색 두 작업 간에 매우 낮은 상관관계를 나타냈기 때문에, 생성 작업의 프롬프트 성능 예측이 별도의 연구로서 필요함을 확인했습니다. 다양한 프리젠더 및 프리트리벌 예측기와 포스트젠더 및 포스트리트리벌 예측기를 통해 경쟁력 있는 베이스라인을 제시했습니다. 실험 결과, 강력한 감독된 프리젠더 및 프리트리벌 예측기가 각 작업에서 포스트젠더 및 포스트리트리벌 예측기와 견줄 만한 성능을 보였습니다.



### Generative AI Models: Opportunities and Risks for Industry and Authorities (https://arxiv.org/abs/2406.04734)
Comments:
          33 pages, 3 figures

- **What's New**: Generative AI(생성형 인공지능) 모델들은 기존 데이터에서 패턴을 학습하고 이를 바탕으로 텍스트, 이미지, 음악 등 새 콘텐츠를 생성하는 능력을 가지고 있습니다. 이러한 모델들은 디지털화의 기회를 제공할 뿐만 아니라, 새로운 IT 보안(IT security) 리스크를 소개합니다. 이에 따라, 생성형 AI를 사용하는 기업이나 기관은 통합하기 전 개별적인 리스크 분석을 수행해야 합니다.

- **Technical Details**: 생성형 AI 모델들은 트레이닝 과정에서 기존 데이터의 패턴을 학습합니다. 이후 이 모델들은 새로운 텍스트, 이미지 또는 음악을 생성하는 데 활용됩니다. 하지만 이 과정에서 발생할 수 있는 다양한 보안 리스크를 고려해야 하며, 개발자 및 운영자들은 이러한 리스크를 고려한 추가적인 보안 조치를 취해야 합니다.

- **Performance Highlights**: 생성형 AI 모델들은 그들의 다재다능함과 일반적으로 높은 품질의 결과로 인해 다양한 분야에서 사용될 수 있습니다. 하지만 보안 리스크를 충분히 분석하고 관리하지 않으면, 오히려 시스템의 안전성에 위협이 될 수 있습니다.



### What do MLLMs hear? Examining reasoning with text and sound components in Multimodal Large Language Models (https://arxiv.org/abs/2406.04615)
Comments:
          9 pages

- **What's New**: 이번 연구는 멀티모달 대형 언어 모델(MLLM)에서 오디오 인코더(audio encoder)가 텍스트 기반 추론(text-based reasoning) 기능을 충분히 활용하지 못하고 있다는 점을 확인했습니다. 특히, 오디오 MLLM을 대상으로 한 실험을 통해 오디오 캡션 생성을 하는 경우 텍스트 기반의 추론을 완벽히 활용하지 못하는 문제점을 발견했습니다.

- **Technical Details**: 이 논문에서는 다양한 데이터 모달리티(data modality)를 처리할 수 있는 멀티모달 대형 언어 모델의 기능을 분석합니다. 기존 연구에 따르면 멀티모달 모델은 이미지나 오디오 인코더의 출력을 토큰화하여 LLM에 입력시키는 구조를 사용합니다. 이를 통해 모델이 텍스트 기반의 추론 능력을 사용할 수 있습니다. 오디오 MLLM에서는 주로 오디오 태스크들을 텍스트 생성 태스크로 변환하는 방식을 사용하며, 이를 통해 통합된 LLM의 인과적 능력을 활용합니다.

- **Performance Highlights**: 연구에 따르면, 오디오 MLLM이 텍스트 기반 추론을 이용한 분류(task)를 수행하는 데 있어서 한계가 있음을 확인했습니다. 특히, 현재의 오디오 MLLM은 오디오와 텍스트 정보를 별도로 표현하기 때문에 LLM의 추론 경로가 오디오 인코더로부터 단절되는 문제가 발생합니다. 이러한 구조적인 한계점을 해결하기 위한 추가적인 연구가 필요합니다.



### Pitch-Aware RNN-T for Mandarin Chinese Mispronunciation Detection and Diagnosis (https://arxiv.org/abs/2406.04595)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 이 논문에서는 중국어 발음 오류 탐지 및 진단(MDD) 시스템을 위한 참신한 접근법을 제시합니다. 이 접근법은 HuBERT 특징과 음높이 임베딩을 사용하여 Stateless RNN-T 모델을 통해 구현됩니다. 기본 데이터만으로 훈련된 모델이 비원어민 시나리오에서 Phone Error Rate (3%)와 False Acceptance Rate (7%)에서 기존 최첨단 기준보다 개선된 결과를 보였습니다.

- **Technical Details**: 제안된 모델은 stateless RNN-T 구조를 따르며, HuBERT 기반 SSL 모듈, 서브샘플링 모듈, 음높이 추출기, 음높이 임베딩, 음높이 인코더, 음높이 융합 블록으로 구성됩니다. 모델은 AISHELL-1 데이터셋으로 훈련되었으며, LATIC 데이터셋에서 평가되었습니다. 이 모델은 입력된 파형으로부터 F0를 추출하여 음높이 임베딩을 생성하고, 이를 피치 인코더에 공급하여 고차원 음높이 특징을 획득합니다. 그런 다음 Pitch Fusion Block을 사용해 HuBERT 특징과 결합하여 MDD 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 모델은 비원어민 시나리오에서 Phone Error Rate를 3% 개선하고, False Acceptance Rate를 7% 증가시켰습니다. 이러한 결과는 기존의 최첨단 기준 모델 대비 성능 향상을 보여줍니다.



### FLUID-LLM: Learning Computational Fluid Dynamics with Spatiotemporal-aware Large Language Models (https://arxiv.org/abs/2406.04501)
- **What's New**: 이 논문에서는 FLUID-LLM이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 사전 학습된 대형 언어 모델(LLM)과 시공간 인식(spatiotemporal-aware) 인코딩을 결합하여 비정상 유체 역학을 예측합니다. 전통적인 유체 역학 시뮬레이션은 나비에-스토크스 방정식(Navier-Stokes equations)의 해결이 필요하여 계산 자원이 많이 요구되는 반면, FLUID-LLM은 이를 극복할 수 있는 방안을 제시합니다.

- **Technical Details**: FLUID-LLM은 시공간 인코딩된 데이터를 활용하여 LLM의 autoregressive(자기 회귀) 능력을 활용합니다. 모델 아키텍처는 이전 상태의 히스토리를 입력으로 받아 다음 상태를 예측합니다. 특히 2D 유체 흐름 시뮬레이션 문제에서 x, y 방향의 속도 성분과 압력을 예측합니다. 각 타임스텝 데이터는 2D 기본 그리드로 변환되고, 패치 기반 인코더와 경계 조건을 사용하여 불규칙한 메쉬(mesh)에서도 작동할 수 있도록 합니다.

- **Performance Highlights**: 표준 CFD 벤치마크에서 FLUID-LLM을 평가한 결과, 여러 유체 데이터 세트에서 성능이 크게 향상되었습니다. FLUID-LLM은 사전 학습된 LLM을 활용하여 시공간 데이터를 효과적으로 통합함으로써, 유체 역학 과제에서 성능을 높이는 데 성공했습니다. 또한, 언어 모델로 미리 학습된 LLM을 사용함으로써 'in-context learning' 능력도 제공되었습니다.



### CORU: Comprehensive Post-OCR Parsing and Receipt Understanding Datas (https://arxiv.org/abs/2406.04493)
- **What's New**: 이번 연구는 다국어 환경에서 영수증의 OCR (Optical Character Recognition)과 정보 추출 성능을 향상시키기 위해 특별히 설계된 CORU (Comprehensive Post-OCR Parsing and Receipt Understanding) 데이터셋을 소개합니다. 이 데이터셋은 아랍어와 영어 텍스트를 포함해 20,000개 이상의 주석이 달린 영수증과 30,000개의 주석이 달린 OCR 이미지, 그리고 상세 정보 추출을 위한 10,000개의 항목을 포함하고 있습니다.

- **Technical Details**: CORU 데이터셋은 세 가지 주요 컴퓨팅 작업을 지원하기 위해 설계되었습니다: 객체 탐지(object detection), OCR, 및 정보 추출(information extraction). 기존 방법(Tesseract OCR)과 더 발전된 신경망(neural network) 기반 접근법을 평가하기 위한 기준 성능을 설정했습니다. 이 데이터셋은 다양한 상업 환경에서 수집된 영수증 이미지들을 포함하여, 텍스트 레이아웃의 복잡성과 다양성을 잘 반영합니다.

- **Performance Highlights**: 기준 성능 평가는 전통적인 방법과 최신 신경망 접근법들을 대상으로 수행되었습니다. CORU 데이터셋은 특히 다양한 레이아웃과 저화질 텍스트의 특성을 잘 처리할 수 있도록 설계된 OCR 모델을 소개합니다. 이 모델은 CNN (convolutional neural networks)와 양방향 LSTMs (bidirectional LSTMs) 결합을 활용하여 복잡하고 잡음이 많은 배경에서도 우수한 성능을 보여줍니다.



### Small-E: Small Language Model with Linear Attention for Efficient Speech Synthesis (https://arxiv.org/abs/2406.04467)
Comments:
          Interspeech

- **What's New**: 최근 텍스트-투-스피치(TTS) 기술은 자연스러움과 제로-샷 음성 복제를 구현하는 데 있어 놀라운 발전을 보여주었습니다. 본 논문에서는 기존 트랜스포머를 대체할 새로운 리커런트 아키텍처와 반복 및 건너뛰기 문제를 줄이기 위해 특화된 크로스-어텐션 메커니즘을 도입했습니다. 결과적으로 해당 아키텍처는 긴 샘플에 효과적으로 학습할 수 있으며, 비슷한 크기의 기준 모델에 비해 최첨단 제로-샷 음성 복제를 달성할 수 있습니다.

- **Technical Details**: 트랜스포머의 자기 어텐션(self-attention) '시퀀스 믹싱(time-mixing)' 연산은 병렬로 효율적으로 학습할 수 있지만 시퀀스 길이에 비례한 복잡성이 존재합니다. 이에 본 연구는 '리니어 어텐션(linear attention)'이라 불리는 새로운 종류의 RNN을 도입하여 자기 어텐션의 선형 복잡도로 시퀀스 믹싱을 대체하는 방안을 제시합니다. RVQ 코덱을 통해 연속적인 잠재 변수를 학습하여 자율 회귀 모델링을 피하고, 고유한 인덕티브 바이어스를 갖추도록 설계된 새로운 RNN 아키텍처를 소개합니다. 또한 반복 및 건너뛰기 문제를 줄이기 위해 특화된 크로스-어텐션 메커니즘을 도입했습니다.

- **Performance Highlights**: 새로운 리커런트 아키텍처는 긴 시퀀스에 대해 더 효율적인 학습을 가능하게 하며 자원 제한된 하드웨어에서도 뛰어난 성능을 발휘합니다. 제안된 시스템은 제로-샷 음성 복제에 있어 비슷한 크기의 기존 기준 모델들을 능가하는 최첨단 성능을 보여줍니다.



### LipGER: Visually-Conditioned Generative Error Correction for Robust Automatic Speech Recognition (https://arxiv.org/abs/2406.04432)
Comments:
          InterSpeech 2024. Code and Data: this https URL

- **What's New**: Automatic Speech Recognition (ASR) 시스템이 시끄러운 환경에서 성능을 개선할 수 있도록 시각적 힌트(예: 입술 움직임)를 활용하는 새로운 프레임워크인 LipGER(Lip Motion aided Generative Error Correction)를 제안합니다. 이 프레임워크는 오디오와 비주얼 모달리티 간의 상호 연관성을 학습하기보다는 Large Language Model(LLM)을 활용해 본질적으로 생성적 ASR 오류 수정 과업을 학습하도록 합니다. 구체적으로는 ASR 빔 서치(beam-search)를 통해 생성된 N-best 가설에서 전사(transcription)를 예측하는 것을 목표로 합니다. 또한, LipHyp이라는 대규모 데이터 세트를 공개하여 추가 연구를 촉진합니다.

- **Technical Details**: LipGER는 전통적인 오디오-비주얼 스피치 인식(Audio-Visual Speech Recognition, AVSR) 시스템과 달리, 시각적 힌트로 조건부된 생성적 오류 수정을 수행합니다. LLM(Large Language Model)은 N-best 가설 목록을 기반으로 텍스트 전사를 예측하며, 이 과정을 통해 시각적 입술 움직임을 활용합니다. 이를 통해 새로운 도메인에 적응하거나 대규모 병렬 데이터 세트가 필요한 기존의 문제점을 해결할 수 있습니다. LipGER는 구조가 간단하면서도 효과적이며, 복잡한 오디오-비주얼 상관 관계를 학습할 필요가 없습니다. 또한, 기존의 강력한 ASR 모델과 함께 사용할 수 있습니다.

- **Performance Highlights**: LipGER는 4가지 데이터 세트에서 다양한 조건 하에 실험을 수행했으며, 이 결과 Word Error Rate(WER)를 1.1%에서 49.2%까지 개선할 수 있는 것으로 나타났습니다. 또한, 실험 데이터 세트는 합성 및 실제 시끄러운 환경을 모두 포함하였습니다.



### Aligning Large Language Models with Self-generated Preference Data (https://arxiv.org/abs/2406.04412)
Comments:
          18 pages, under review

- **What's New**: 여러분은 대형 언어 모델(LLMs)을 인간의 선호에 맞추는 것이 최신 성능을 얻기 위한 중요한 요소임을 알고 계십니까? 하지만 이를 위해서는 막대한 양의 인간 주석 데이터가 필요하여 큰 비용이 소요되는 문제가 있습니다. 이를 해결하기 위해, 소량의 인간 주석 데이터만을 이용하여 Self-generated Preference data (Selfie)라는 새로운 프레임워크를 제안했습니다. 이 프레임워크는 초기 소량의 데이터에서 시작하여, 모델이 자체적으로 생성한 응답과 선호 데이터를 반복적으로 학습하는 방식으로 LLMs의 정렬(alignment)을 지속적으로 향상시킵니다.

- **Technical Details**: Selfie 프레임워크의 핵심 아이디어는 LLM 출력의 로그잇(logits)을 이용해 모델의 내재된 선호를 명시적으로 추출하고, 이를 선호 레이블로 활용하는 것입니다. 기존의 외부 보상 모델이나 암시적 학습(in-context learning)에 의존하는 방법보다 효과적이라고 할 수 있습니다. 또한, 생성된 선호 데이터의 품질을 높이기 위해 노이즈 인식 학습 알고리즘을 도입했습니다. 이 방법은 선택적 신뢰도를 사용해 노이즈를 줄이고, 현재 모델과 참조 모델 간의 선형 외삽(linear extrapolation) 예측을 적용해 더 나은 노이즈 식별을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, Selfie 프레임워크는 LLMs의 정렬 성능을 현저히 향상시킵니다. 예를 들어, Ultrafeedback 데이터의 3.3%만을 사용해도 SOTA 기법들과 비교해 AlpacaEval2.0에서 높은 정렬 성능을 보였습니다. 구체적으로 초기에 비해 16.4%의 성능 향상을 기록했고, MT-bench 점수도 6.38에서 6.94로 증가했습니다. Selfie는 초기 인간 주석 데이터 없이도 다양한 LLMs의 정렬을 성공적으로 개선한다는 점에서 매우 실제적이고 경쟁력 있는 방법입니다.



### Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive? (https://arxiv.org/abs/2406.04391)
- **What's New**: 이 연구는 고급 AI 시스템의 확장(scaling) 성능 예측이 가능한 것이 매우 유용한 속성임에도 불구하고, 특정 다운스트림(downstream) 기능을 예측하는 것이 왜 어려웠는지를 탐구합니다. 연구는 다섯 개의 모델 패밀리와 열두 개의 널리 사용되는 다중 선택 질문-답변 벤치마크를 사용하여 이를 조사했습니다.

- **Technical Details**: 연구진은 다운스트림 성능이 부정적인 로그 가능도(negative log likelihood) 변환 시퀀스를 통해 계산됨에 따라, 성능과 스케일(scale) 간의 통계적 관계가 점진적으로 악화된다는 점을 발견했습니다. 주요 원인으로는 올바른 선택과 제한된 수의 특정한 잘못된 선택을 비교해야 하는 다운스트림 메트릭이 있음을 지적합니다. 스케일 증가에 따라 올바른 선택에 대한 확률 질량이 어떻게 변동하는지 뿐 아니라, 특정 잘못된 선택에 대한 확률 질량도 예측해야 하기 때문입니다.

- **Performance Highlights**: 연구는 올바른 선택에 대한 확률 질량이 잘못된 선택들과 함께 컴퓨팅 증가에 따라 어떻게 변동하는지를 실험적으로 조사했습니다. 이를 통해 잘못된 선택에 대한 확장 법칙이 달성 가능할 수 있다는 점을 시사합니다. 이러한 발견은 왜 프리트레이닝(pretraining) 확장 법칙이 다운스트림 기능보다 더 예측 가능하게 여겨지는지를 설명하며, 최첨단 AI 모델의 확장-예측 가능한 평가를 위한 기여를 제공합니다.



### VHDL-Eval: A Framework for Evaluating Large Language Models in VHDL Code Generation (https://arxiv.org/abs/2406.04379)
Comments:
          6 pages, 3 Figures, LAD'24

- **What's New**: 이 논문에서는 LLMs (Large Language Models)의 가능성이 VHDL (VHSIC Hardware Description Language) 코드 생성 작업에도 적용될 수 있음을 보여주는 새로운 평가 프레임워크를 소개합니다. 이 프레임워크는 특히 VHDL 코드 생성 작업에서 LLMs의 성능을 평가하는 데 중점을 둡니다.

- **Technical Details**: 이 연구는 Verilog 평가 문제를 VHDL로 번역하고 공개적으로 이용 가능한 VHDL 문제를 수집하여 총 202개의 문제로 구성된 데이터셋을 만들었습니다. 또한, 코드 생성의 기능적 정확성을 평가하기 위해 문제에 맞춘 self-verifying 테스트벤치를 사용합니다. 다양한 LLMs와 그 변형(제로샷 코드 생성, in-context learning (ICL), Parameter-efficient fine-tuning (PEFT))을 평가하였습니다.

- **Performance Highlights**: 기존 LLMs는 VHDL 코드 생성에서 상당한 어려움을 겪고 있음을 발견하였습니다. 이는 VHDL 디자인 효율성을 추구하는 디자이너들에게 큰 도움이 될 수 있는 VHDL 전용 코드 생성 모델의 필요성을 강조합니다.




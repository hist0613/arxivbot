New uploads on arXiv(cs.CL)

### Language Modeling with Editable External Knowledg (https://arxiv.org/abs/2406.11830)
- **What's New**: 이 논문에서는 어휘 모델(Language Models)이 지속적으로 변하는 최신 정보와 동기화 할 수 있도록 하는 방법을 제안합니다. 특히 'ERASE (Enhancing Retrieval Augmentation with Self-consistent Editing)'라는 새로운 접근법을 도입하여, 새로운 문서가 추가될 때 기존 지식 베이스(Knowledge Base)의 항목을 재작성하거나 삭제함으로써 모델의 정확성을 높입니다. 이는 기존의 문서 검색 및 생성 방식에서 벗어나, 새로운 정보에 따라 지식 베이스를 동적이고 정확하게 업데이트하는 데 초점을 맞춥니다.

- **Technical Details**: ERASE 방법은 문서 입력 시 기존 지식 베이스의 관련 문서를 식별하고 보존, 편집 또는 삭제하는 방식으로 동작합니다. 이를 통해 새로운 정보가 지식 베이스에 추가될 때마다 최신 상태를 유지할 수 있습니다. 두 가지 새로운 벤치마크 데이터셋이 도입되었으며, 이는 뉴스 기사와 대화를 스트리밍하며 질문에 답하는 모델의 능력을 평가합니다. 특히, 'clark-News'와 'clark-Conversations'라는 두 가지 도메인에서 테스트되었습니다.

- **Performance Highlights**: ERASE는 전통적인 검색 증강 생성 방법 (RAG) 대비 성능을 크게 향상시켰습니다. Mixtral-8x7B 모델에서 7-13%의 향상, Llama-3-8B 모델에서 6-10%의 향상을 보였습니다. 다중 논리를 필요로 하는 부분에서는 여전히 개선의 여지가 남아 있지만, 단일 논리 섹션에서는 현저한 성능 향상을 보여주었습니다.



### WPO: Enhancing RLHF with Weighted Preference Optimization (https://arxiv.org/abs/2406.11827)
- **What's New**: 이번 논문에서는 현재 정책 모델(Policy Model)에서 생성한 출력을 기반으로 새로운 선호 데이터셋을 샘플링하여 오프 정책 선호 최적화(Off-Policy Preference Optimization)의 성능 향상을 꾀하는 Weighted Preference Optimization (WPO) 방법을 제안합니다. 이를 통해 기존의 오프 정책 데이터와 유사한 방식으로 온 정책(On-Policy) 학습을 모방할 수 있으며, 기존 Direct Preference Optimization (DPO) 방법과 비교하여 최적화 과정에서 더 우수한 결과를 보입니다.

- **Technical Details**: WPO는 현재 정책 모델에서 출력을 다시 가중치(Weighted)하여 기존 선호 데이터 쌍의 확률에 따라 재배열함으로써 수행됩니다. 이 과정은 부트스트래핑(bootstrapping) 과정을 활용하여 새로운 선호 데이터셋을 생성하고, 그 쌍들이 레이블링 함수(Labeling Function)를 통해서 결정되도록 합니다. 또한, DPO와 함께 WPO를 통합하여 최적화 과정에서 온 정책으로 생성된 쌍의 가중치가 동일하게 분배되도록 하는 메커니즘을 도입하였습니다.

- **Performance Highlights**: WPO는 Alpaca Eval 2에서 DPO보다 최대 5.6% 더 높은 성능을 보였고, Llama-3-8B-Instruct 기반으로 GPT-4-turbo 대비 48.6%의 리더보드 최강 승률을 달성하였습니다. 그 결과, 현존하는 가장 강력한 8B 모델 중 하나로 자리매김했습니다.



### Iterative Length-Regularized Direct Preference Optimization: A Case Study on Improving 7B Language Models to GPT-4 Lev (https://arxiv.org/abs/2406.11817)
- **What's New**: 본 논문은 사람의 선호도에 맞춰 언어 모델을 정렬하는 표준 방법인 Direct Preference Optimization (DPO)를 개선한 Iterative Length-Regularized DPO (iLR-DPO)를 소개합니다. 기존의 vanilla iDPO는 응답 품질을 향상시킬 수 있으나, 응답의 장황함도 증가시키는 문제가 있었습니다. 이를 해결하기 위해 응답 길이에 페널티를 주는 iLR-DPO를 소개합니다. iLR-DPO를 통해 7B 모델이 GPT-4 수준의 성능을 달성하면서도 장황함을 줄일 수 있음을 입증하였습니다.

- **Technical Details**: iLR-DPO는 기본 언어 모델(π_base)의 출력을 특정 보상 모델(r)과 비교하여 최적화하는 방법입니다. 이 방법은 두 가지 주요 단계로 이루어집니다: (1) 보상 모델로부터 합성 선호도를 수집하고, (2) 길이 페널티를 적용하여 언어 모델을 최적화합니다. 각 반복(iteration)마다 최신 언어 모델 체크포인트에서 출발하여, 보상 모델의 피드백에 따라 선호도를 수집하고 학습합니다. 이를 통해 응답의 길이를 조절하면서 품질을 극대화합니다.

- **Performance Highlights**: iLR-DPO로 학습된 7B 모델은 GPT-4 Preview와 비교하여 AlpacaEval 2.0에서 50.5%의 길이-통제된 승률을 달성했으며, MT-Bench, Arena-Hard, OpenLLM Leaderboard 등의 표준 벤치마크에서도 뛰어난 성능을 보였습니다. 이러한 결과는 iLR-DPO가 인간의 피드백에 맞춰 언어 모델을 정렬하는 데 효과적임을 보여줍니다.



### How Do Large Language Models Acquire Factual Knowledge During Pretraining? (https://arxiv.org/abs/2406.11813)
- **What's New**: 대형 언어 모델(LLMs)의 사전 훈련 중 사실적 지식을 습득하는 동역학에 대한 새로운 연구가 발표되었습니다. 연구팀은 LLMs가 사실적 지식을 어떻게 습득하고 잊어버리는지에 대한 몇 가지 중요한 통찰을 제공했습니다. 중요한 발견으로는 더 많은 데이터로 훈련한다고 해서 사실적 지식 습득 능력이 크게 향상되지 않는다는 점과, 훈련 단계와 사실적 지식의 망각 사이에 강력한 비율 관계가 있다는 점입니다. 또한, 더 큰 배치 크기로 훈련하면 모델의 망각에 대한 견고성이 증가함을 발견했습니다.

- **Technical Details**: 연구진은 다양한 훈련 조건에서 사실적 지식의 습득과 망각을 분석했습니다. 특히 모델 크기, 훈련 단계, 훈련 배치 크기를 두루 살펴보며, 특정 지식을 주입하고 그에 따른 모델의 지식 습득 과정을 세밀하게 분석했습니다. 실험에서는 중간 사전 훈련 체크포인트와 새로운 사실적 지식을 주입하여 각 훈련 단계에서 지식 습득 과정을 모니터링 했습니다. 지식 습득의 세 단계(암기, 의미적 일반화, 합성적 일반화)로 나누어 탐침을 설계하고 세밀한 분석을 통해 모델이 각 단계에서 어떻게 지식을 습득하는지 관찰했습니다.

- **Performance Highlights**: 연구 결과, 더 큰 배치 크기를 사용하는 것이 모델이 망각에 더 견고해지도록 만든다는 것을 발견했습니다. 또한 지식 주입 시 모델 파라미터의 로그 확률 변화가 몇 단계에 걸쳐서 최댓값에 도달하는 현상이 관찰되었습니다. 이러한 결과는 장기 꼬리 지식(long-tail knowledge) 습득의 어려움 및 데이터셋 deduplication의 중요성을 설명하는 데 기여할 수 있습니다.



### RepLiQA: A Question-Answering Dataset for Benchmarking LLMs on Unseen Reference Conten (https://arxiv.org/abs/2406.11811)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 평가를 위해 새롭게 소개된 RepLiQA 데이터셋이 발표되었습니다. RepLiQA는 인터넷에 공개된 적이 없는 참신한 테스트 데이터를 제공하여 모델의 성능을 더 정확하게 평가할 수 있도록 고안되었습니다. 이 데이터셋은 질문-답변과 주제 검색(task retrieval) 작업에 적합하게 설계되었습니다.

- **Technical Details**: RepLiQA는 총 5개의 분할 테스트 세트를 포함하고 있으며, 이 중 4개는 인터넷에 공개된 적이 없습니다. 각 샘플은 인간 평가자가 작성한 참조 문서, 문서의 주제에 대한 질문, 문서에 기반한 정답, 그리고 참조 문서에서 발췌한 단락으로 구성되어 있습니다. 따라서 모델이 제공된 문서 내에서 적절한 내용을 찾아야만 정확한 답변을 생성할 수 있습니다.

- **Performance Highlights**: 다양한 최첨단 LLM을 대상으로 대규모 벤치마크를 수행한 결과, 모델 종류와 크기에 따른 성능 차이가 확인되었습니다. RepLiQA의 공개된 첫 번째 분할은 현재 사용 가능하며, 향후 몇 개월 동안 추가 분할이 순차적으로 공개될 예정입니다. 실험은 공개된 첫 번째 분할을 사용해 수행되었으며, 이를 통해 보지 않은 콘텐츠에 대한 모델 평가의 중요성이 강조되었습니다.



### Safety Arithmetic: A Framework for Test-time Safety Alignment of Language Models by Steering Parameters and Activations (https://arxiv.org/abs/2406.11801)
Comments:
          Under Review. Codes are available at: this https URL

- **What's New**: 오늘 소개할 논문은 'Safety Arithmetic'이라는 새로운 프레임워크를 제안하여 큰 언어 모델(Large Language Models, LLMs)의 안전성을 강화하는 방법을 다룹니다. 이 프레임워크는 훈련이 필요 없는 방식으로, 기본 모델(Base models), 감독 하에 미세 조정된 모델(Supervised Fine-tuned models, SFT), 편집된 모델(Edited models) 등 다양한 시나리오에서 LLM의 안전성을 보장합니다. 또한, 우리는 'NoIntentEdit'이라는 데이터셋을 제공하여, 의도치 않게 모델의 안전성을 저해할 수 있는 편집 인스턴스를 강조합니다.

- **Technical Details**: Safety Arithmetic 프레임워크는 크게 두 가지 단계로 구성됩니다: 1. 'Harm Direction Removal' 단계에서는 모델의 파라미터를 해로운 방향으로부터 조정합니다. 2. 'Safety Alignment' 단계에서는 잠재 공간(Latent Space)을 안전한 응답 생성 방향으로 맞춥니다. 이 접근법은 모든 세 가지 시나리오 (Base models, SFT, Edited models)에 걸쳐 모델의 일반적 기능을 보존하면서 안전성을 높이도록 설계되었습니다. 주요 구성 요소는 다음과 같습니다: harm direction 제거와 잠재 공간 조정을 통해 모델의 안전성을 보장하고, 다양한 상태에서 모델의 성능을 유지합니다.

- **Performance Highlights**: 실험 결과, Safety Arithmetic은 기존에 사용되던 방법들보다 안전 측정 지표를 유의미하게 개선하고, 과도한 안전성(Over-safety)을 줄이며 모델의 유용성을 유지하는 것으로 나타났습니다. 구체적으로, 다양한 LLM 시나리오에서 안전한 콘텐츠 생성을 보장하면서도 성능 저하가 없음을 입증했습니다.



### CELL your Model: Contrastive Explanation Methods for Large Language Models (https://arxiv.org/abs/2406.11785)
- **What's New**: 이 논문에서는 LLMs(Large Language Models)에 대한 첫 번째 대조적 설명 방법을 제안합니다. 기존의 설명 기법은 주로 분류 및 회귀 모델에 초점을 맞췄지만, 본 연구는 LLM이 특정 프롬프트에 대해 왜 특정 응답을 생성했는지 설명하는 방법을 제시합니다.

- **Technical Details**: LLMs에 대한 대조적 설명을 위해 두 가지 알고리즘을 제안합니다. 첫째, 작은 프롬프트에 효과적인 '근시안적 알고리즘(myopic algorithm)'입니다. 둘째, 긴 문맥에 적합한 '예산 알고리즘(budgeted algorithm)'으로, 쿼리 예산 내에서 효율적으로 대조를 생성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 다양한 자연어 처리 작업에서 효과적으로 작동함을 보였습니다. 예를 들어, 열린 텍스트 생성, 자동 레드 팀 작업 및 대화 저하 설명 등의 분야에서 성능이 입증되었습니다.



### MDCR: A Dataset for Multi-Document Conditional Reasoning (https://arxiv.org/abs/2406.11784)
- **What's New**: 본 연구는 ConditionalQA의 한계를 극복하고 다중 문서에 걸친 최적화 질문을 해결하기 위해 새로운 데이터셋 MDCR(Multi-Document Conditional Reasoning)을 제안합니다. MDCR은 실생활에서 발생하는 복잡한 조건적 추론의 난제를 반영하며, 기존 모델의 한계를 평가하고 개선 방향을 제시합니다.

- **Technical Details**: MDCR 데이터셋은 장학금과 직업이라는 두 가지 도메인에서 수집된 문서들로 구성되어 있으며, 다양한 수의 문서를 토대로 한 모델의 추론 능력을 평가하는 질문들을 포함합니다. 이 데이터셋은 문서 간 조건 관계(conflicting, equivalent, inclusive)를 이해하고 최적화된 결과를 도출하는 모델의 능력을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: 최신 대형언어모델(LLMs)인 GPT-4와 Llama3-70B를 이용한 평가에서, 이들 모델은 약 69%의 단답 정확도와 약 40%의 조건적 답변 F1 점수를 기록하며, 이 과제의 어려움을 보여주었습니다.



### Improving Multi-Agent Debate with Sparse Communication Topology (https://arxiv.org/abs/2406.11776)
Comments:
          13 pages, 9 figures

- **What's New**: 본 논문에서는 다중 에이전트 시스템에서의 통신 연결이 성능에 미치는 영향을 체계적으로 조사하였습니다. GPT와 미스트랄(Mistral) 모델에서의 실험을 통해, 희소한 통신 토폴로지를 활용한 다중 에이전트 토론이 매우 적은 계산 비용으로 동등하거나 더 높은 성능을 발휘할 수 있음을 밝혔습니다. 또한, 이 다중 에이전트 토론 프레임워크를 멀티모달(Multimodal) 추론 및 정렬 레이블링(alignment labeling) 작업으로 확장하여 넓은 적용 가능성과 효율성을 보여주었습니다.

- **Technical Details**: 논문에서는 다중 에이전트 각기 다른 회의론적 접근 대신, 에이전트 간의 통신 연결을 제한함으로써 얻을 수 있는 이점을 다루고 있습니다. 이 접근 방식에서 에이전트는 이웃 연결(neighbor-connected) 구조를 채택하여 모든 다른 에이전트와 통신하는 대신, 제한된 수의 에이전트와만 상호작용합니다. 또한, 다중 에이전트 시스템의 통신 토폴로지(Topology)는 그래프 𝒢=(𝒱,ℰ)로 나타내며, 한 에이전트가 다른 에이전트의 이전 라운드 솔루션에 접근할 수 있음을 나타냅니다. 이 연구는 정적 그래프와 동적 그래프 모두를 실험합니다.

- **Performance Highlights**: 희소한 통신 토폴로지를 적용한 결과, MATH 데이터세트에서 성능이 +2% 향상되고 GSM8K 데이터세트에서는 동일한 정확도를 유지하면서도 추론 작업의 평균 입력 토큰 비용이 40% 이상 감소했습니다. 또한, Anthropic-HH 데이터세트의 분석에서 도움(helpfulness)과 무해성(harmlessness)이 각각 +0.5%와 +1.0% 향상되었으며, 비용은 각각 50.0%와 53.3% 감소했습니다.



### A Semantic-based Layer Freezing Approach to Efficient Fine-Tuning of Language Models (https://arxiv.org/abs/2406.11753)
Comments:
          13 pages, 5 figures, under peer-review

- **What's New**: 새로운 연구는 언어 모델(LM)을 특정 작업용 데이터에 적응시키기 위한 방법인 파인 튜닝(finetuning)에 관한 것이다. 기존의 많은 연구가 '어떻게'(how to) 파인 튜닝할지에 집중했지만, 본 연구는 '어디서'(where to) 파인 튜닝할지에 대한 첫 번째 접근법을 제안한다.

- **Technical Details**: 이 연구는 LM 추론 과정을 의미적 분석을 통해 이해하려고 시도한다. 우선, 잠재 표현(latent representation)의 가상 전환을 제안한 후 실제 전환을 추적한다. 각 모델 레이어의 파인 튜닝 이점을 추산하기 위해 전환의 편차를 기반으로 한다. 이를 통해 파인 튜닝의 범위를 좁혀 나가게 된다. 제안된 접근법은 PEFT(parameter-efficient finetuning)와 같은 기존 기술과도 상호 보완적이다.

- **Performance Highlights**: 많은 유명한 언어 모델과 데이터셋을 대상으로 광범위한 실험을 수행한 결과, 본 접근법이 효과적이고 효율적이며, 기존의 기준 모델들을 능가하는 것으로 나타났다.



### Interactive Evolution: A Neural-Symbolic Self-Training Framework For Large Language Models (https://arxiv.org/abs/2406.11736)
Comments:
          18 pages, 6 figures

- **What's New**: 본 연구에서는 사람 주석 데이터를 거의 사용하지 않고 LLM (Large Language Models)을 향상시키기 위한 자가 학습 프레임워크인 ENVISIONS를 제안합니다. 이 프레임워크는 특히 심볼릭 데이터(symmetric data) 부족과 심볼릭 언어(symbolic language) 처리에서 LLM의 한계를 극복하는 것을 목표로 하고 있습니다. 코드 또한 공개될 예정입니다.

- **Technical Details**: ENVISIONS는 환경 지향 자가 학습 프레임워크로, LLM을 체화된 환경과의 상호작용을 통해 반복적으로 훈련시키는 방식입니다. 구체적으로 웹 브라우징 작업과 같은 예시에서 LLM 에이전트(agent)가 웹 브라우저 내에서 다양한 후보 행동을 실행하며 옳은 결과와 잘못된 결과를 도출합니다. 이 과정에서 자기 보상 알고리즘(self-rewarding algorithm)을 설계해 에이전트의 경로를 후처리하고 대조적 훈련 쌍(contrastive training pairs)을 만듭니다.

- **Performance Highlights**: 세 가지 도메인에서의 광범위한 평가를 통해 ENVISIONS의 효과를 검증하였으며, 기존의 더 강력한 모델이나 보상 모델 없이도 LLM을 일관되게 향상시킬 수 있음을 확인하였습니다. ENVISIONS는 이전 방법들과 상호 배타적이지 않으며, 상호 시너지를 탐구하는 것도 미래 연구 과제로 남아 있습니다.



### Zero-Shot Generalization during Instruction Tuning: Insights from Similarity and Granularity (https://arxiv.org/abs/2406.11721)
Comments:
          33 pages, 14 figures

- **What's New**: 이번 논문에서는 zero-shot generalization 라는 개념에 대해 새로운 인사이트를 제공합니다. 특히, instruction tuning(명령어 튜닝)에서 zero-shot generalization이 발생하는 시점과 그 메커니즘을 깊이 있게 탐구하였습니다. 이 연구는 기존의 'task' 개념에 얽매이지 않고, 데이터 자체의 관점에서 zero-shot generalization을 분석합니다.

- **Technical Details**: 저자들은 zero-shot generalization이 instruction tuning 초기에 발생한다는 사실을 여러 지표를 통해 입증했습니다. 이들은 데이터의 유사도(similarity)와 세분화된 정보(granularity)가 초기 단계에서의 zero-shot generalization을 촉진한다고 주장합니다. 이를 위해 Weight Similarity Distance(WSD), 그리고 Test-centric Multi-turn Arrangement(TMA)라는 새로운 데이터 배치 방식을 제안했습니다.

- **Performance Highlights**: 이 연구는 두 가지 주요한 기여를 합니다. 첫째, zero-shot generalization이 instruction tuning 초기에 발생하며, 이를 측정하는데 손실(loss)이 안정적이고 공정한 지표로 작용한다는 점입니다. 둘째, Test-centric Multi-turn Arrangement(TMA)를 통해 지속적인 학습과 추가적인 손실 감소를 촉진할 수 있음을 입증했습니다.



### Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging (https://arxiv.org/abs/2406.11709)
- **What's New**: TreeInstruct는 state space 추정과 동적 tree-based 질문을 이용한 다중 회전 소크라틱 교수법을 처음으로 탐구한 연구입니다. 이 시스템은 학생이 독립적으로 오류를 식별하고 해결할 수 있도록 유도하는 질문을 통해 학생을 교육합니다.

- **Technical Details**: TreeInstruct는 학생의 응답과 현재 지식 상태에 따라 질문 트리(question tree)를 동적으로 구성하는 알고리즘을 사용합니다. 이를 통해 학생의 개념적 및 문법적 지식을 추정하고, 비독립적이자 종속적인 오류를 동시에 해결합니다. 또한 다중 버그를 포함한 새로운 데이터셋을 구축하고, state-of-the-art 성능을 보이는 결과를 도출했습니다.

- **Performance Highlights**: 기존 벤치마크와 새로 구성한 데이터셋에서 폭넓게 평가한 결과, TreeInstruct는 모든 기준에서 탁월한 소크라틱 질문 능력을 보였으며, 실제 학생들과의 상호작용 사례 연구에서도 적은 회전수로 효율적인 코드 디버깅을 유도하는 능력을 입증했습니다.



### Nemotron-4 340B Technical Repor (https://arxiv.org/abs/2406.11704)
- **What's New**: 이번에 발표된 논문에서는 Nemotron-4 340B 모델 패밀리, 즉 Nemotron-4-340B-Base, Nemotron-4-340B-Instruct, Nemotron-4-340B-Reward를 공개하였습니다. 이 모델들은 NVIDIA의 오픈 모델 라이선스 계약(NVIDIA Open Model License Agreement) 하에 배포되어, 활용, 수정 및 출력 사용이 자유롭습니다.

- **Technical Details**: Nemotron-4-340B-Base 모델은 9조 개의 고품질 토큰으로 학습되었으며, Supervised Fine-Tuning(SFT) 및 Reinforcement Learning with Human Feedback(RLHF)와 같은 Preference Fine-Tuning(힐 프라이퍼런스 튜닝) 과정을 통해 조정되었습니다. 모델 학습에는 DGX H100 서버의 8개의 GPU를 사용하여 FP8 정밀도로 배포가 가능하도록 설계되었습니다. 또한 고품질의 Synthetic Data Generation Pipeline(합성 데이터 생성 파이프라인)을 공개하고 있어, 모델 정렬 과정에서의 효율성을 보여줍니다.

- **Performance Highlights**: Nemotron-4-340B 모델 패밀리는 다양한 평가 지표에서 경쟁력 있는 성능을 보여줍니다. Nemotron-4-340B-Base는 Llama-3 70B, Mixtral 8x22B, Qwen-2 72B 모델과 함께 ARC-Challenge, MMLU, BigBench Hard 벤치마크에서 뛰어난 성능을 자랑합니다. Nemotron-4-340B-Instruct는 명령 따르기와 대화 능력 면에서 가장 앞서 있으며, Nemotron-4-340B-Reward는 RewardBench에서 가장 높은 정확도를 기록, GPT-4o-0513 및 Gemini 1.5 Pro-0514과 같은 모델을 초과 성능을 보입니다. 합성 데이터 생성에서도 상당한 성과를 보이며, 98% 이상의 정렬 데이터가 합성 데이터로 이루어졌습니다.



### Meta Reasoning for Large Language Models (https://arxiv.org/abs/2406.11698)
- **What's New**: Meta-Reasoning Prompting(MRP)은 혁신적이고 효율적인 시스템 프롬프트 방법으로, 대형 언어 모델(LLM)에게 인간의 메타-이성을 본뜬 새로운 접근법을 제시합니다. MRP는 각 작업의 요구 사항에 맞추어 동적으로 다양한 이성적 방법을 선택하고 적용하도록 LLM을 유도하여, 전반적인 성능과 컴퓨팅 효율성을 최적화합니다.

- **Technical Details**: MRP는 두 단계로 구성됩니다. 첫 번째 단계에서는 LLM이 과제 입력 신호와 사용할 수 있는 방법의 객관적 설명을 사용하여 가장 적절한 이성적 방법을 식별합니다. 두 번째 단계에서는 선택된 방법을 적용하여 과제를 완료합니다. 이는 인간의 메타-이성을 반영하여 다양한 문제 도메인에서 모델의 성능을 향상시킵니다. 사전 정의된 방법 집합(Reasoning Pool)에서 각 방법에 대한 설명을 기반으로 프롬프트를 사용하여 선택 과정을 안내합니다.

- **Performance Highlights**: 다수의 벤치마크를 통해 MRP의 효과를 평가한 결과, 다양한 작업에서 MRP가 최첨단 성능에 도달하거나 접근한다는 것을 확인할 수 있었습니다. 특히, GPT-4와 같은 더 큰 모델에서 MRP의 성능이 두드러지게 향상됐습니다. MRP는 LLM의 내재된 메타-인지능력을 활용하여 다양한 작업에 대한 일반성과 성능을 개선합니다.



### Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (https://arxiv.org/abs/2406.11695)
Comments:
          Krista and Michael contributed equally to this work

- **What's New**: 이번 논문에서는 다단계 구성을 갖춘 Language Model (LM) 프로그램의 프롬프트 최적화 방법을 소개합니다. 이를 통해 개별 모듈의 레이블이나 기울기에 접근하지 않고도 다운스트림 메트릭을 최대화할 수 있습니다. 새로운 프롬프트 최적화 도구 MIPRO를 개발하여 기존 베이스라인을 능가하는 성능을 보여줍니다. 이번 연구 결과는 오픈 소스 모델 Llama-3-8B를 사용하여 6개의 다양한 LM 프로그램 중 5개에서 최대 12.9%의 정확도 향상을 기록했습니다.

- **Technical Details**: 프롬프트 최적화를 위해 자유 형식의 지시문(instructions)과 소수 샷 데모(few-shot demonstrations)를 최적화하는 방식을 사용합니다. 효율적인 지시문을 제안하기 위해 프로그램 및 데이터 인식 기법과 확률적 미니배치 평가 함수, 메타 최적화 절차(meta-optimization procedure)를 도입했습니다. 특히, LM 가중치, 로그-확률(log-probabilities), 수공한 메트릭 없이는 개별 모듈에 대한 중간 레이블을 이용하지 않는다는 약한 조건에서도 동작하도록 설계되었습니다.

- **Performance Highlights**: MIPRO는 6개의 다양한 LM 프로그램 중 5개에서 기존 베이스라인 모델 대비 최대 12.9% 정확도 향상을 달성했습니다. 또한, 학습 데이터 세트를 토대로 프로그램을 전체적으로 최적화함으로써 얻은 성능 향상을 통해 다섯 가지 주요 교훈을 도출했습니다. 제안된 방법론과 최적화 도구는 DSPy 벤치마크에서 검증되었습니다.



### Tokenization Falling Short: The Curse of Tokenization (https://arxiv.org/abs/2406.11687)
- **What's New**: 대규모 언어모델(LLMs)이 토크나이제이션(tokenization)에 민감하게 반응하여 발생하는 문제들인 '토크나이제이션의 저주(curse of tokenization)'를 체계적으로 연구한 논문이 발표되었습니다. 이 연구는 복잡한 문제 해결(RQ1), 토큰 구조 탐사(RQ2), 타이포그래픽 변형에 대한 내구성(RQ3)이라는 세 가지 주요 연구 질문을 통해 이러한 문제들의 영향을 검토합니다. 또한, 코드와 데이터를 공개하여 추가 연구를 장려할 예정입니다.

- **Technical Details**: 연구는 토크나이제이션의 민감도 문제를 다루며, 특히 오타, 길이 변동, 토큰 내부 구조에 대한 무지로 인해 발생하는 문제들을 강조했습니다. 주요 실험을 통해 LLMs가 오타 및 텍스트 형식 변형에 의해 유발되는 편견을 여전히 겪고 있음을 발견했습니다. 문제 해결을 위해 BPE-dropout와 같은 서브워드 정규화를 제안했습니다.

- **Performance Highlights**: 실험 결과, 모델 파라미터를 확대하면 토크나이제이션 문제를 어느 정도 완화할 수 있지만, 여전히 오타와 같은 타이포그래픽 변형에 민감함을 확인했습니다. LLama3, Mistral, GPT-4와 같은 최신 모델들도 이 문제에서 자유롭지 못했습니다. 또한, BPE-dropout과 같은 정규화된 토크나이제이션 접근법이 모델의 내구성을 강화하는데 효과적임을 입증했습니다.



### HoLLMwood: Unleashing the Creativity of Large Language Models in Screenwriting via Role Playing (https://arxiv.org/abs/2406.11683)
- **What's New**: HoLLMwood는 대형 언어 모델(LLMs)의 창의성을 발휘하여 시나리오 작성(screenwriting)을 자동화하는 새로운 프레임워크입니다. 이 프레임워크는 LLMs를 다양한 역할에 할당하여, 작가(Writer), 편집자(Editor), 배우(Actors)로 나누어 시나리오 작성 과정을 진행합니다. 이를 통해 LLMs는 인간의 창작 과정을 모방하면서 높은 수준의 시나리오를 생성할 수 있습니다.

- **Technical Details**: HoLLMwood 프레임워크는 다음과 같은 역할을 LLMs에게 부여합니다: 작가(Writer)는 초기 이야기 줄거리를 기반으로 캐릭터와 줄거리 초안을 작성합니다. 편집자(Editor)는 그 초안을 검토하고 피드백을 제공하여, 작가가 초안을 수정할 수 있도록 합니다. 이후, 배우(Actors)는 각 캐릭터로 역할극(role-playing)을 하여 대화와 상호작용을 통해 스토리를 더욱 풍부하게 만듭니다. 이 모든 과정은 HTML 스타일의 프롬프트 형태로 구조화되어 있으며, 수작업 편집도 쉽게 통합할 수 있습니다.

- **Performance Highlights**: 실험 결과, HoLLMwood는 기존의 강력한 베이스라인보다 스토리의 일관성, 관련성, 흥미로움, 전반적인 품질 측면에서 크게 뛰어난 성과를 보였습니다. 페어와이즈(paired) 비교를 통해 검증된 결과, LLMs를 작가와 편집자로 나누고, 역할극을 도입한 이 프레임워크는 각 모듈이 최종 시나리오의 품질을 향상시키는 데 긍정적인 영향을 미친다는 것을 확인했습니다.



### Knowledge-to-Jailbreak: One Knowledge Point Worth One Attack (https://arxiv.org/abs/2406.11682)
Comments:
          18 pages, 14 figures, 11 tables

- **What's New**: 이 논문은 'Knowledge-to-Jailbreak'라는 새로운 과제를 제안하며, 이는 도메인 지식으로부터 LLMs(Large Language Models)에 대한 공격을 생성하여 LLMs의 특정 도메인에서의 안전성을 평가하는 것입니다. 이를 위해 12,974개의 지식-탈옥 쌍 데이터셋을 수집하고, 탈옥 공격을 생성하는 'jailbreak-generator'를 훈련했습니다.

- **Technical Details**: 연구진은 Llama2-7b 모델을 활용해 'jailbreak-generator'를 미세 조정(fine-tune)했습니다. 처음으로 위키피디아(Wikipedia)와 같은 자료에서 지식 조각(snippet)을 수집하고, 이를 기존의 일반 탈옥 공격과 결합해 새로운 공격 쌍을 생성하는 '역 데이터 생성(reverse data generation)' 방법론을 도입했습니다. 탈옥 공격 생성의 효과성을 평가하기 위해 13개 도메인과 8개의 타겟 LLMs에서 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 'jailbreak-generator'는 높은 공격 성공률(ASR, Attack Success Rate)과 해악성(harmfulness) 지표에서 대부분의 도메인 및 타겟 LLMs에서 우수한 성능을 보여줬습니다. 또한, 새로운 도메인 지식을 사용한 실험에서도 인간 전문가들과 비슷한 수준의 해악성을 가지는 공격을 생성할 수 있음을 확인했습니다.



### R-Eval: A Unified Toolkit for Evaluating Domain Knowledge of Retrieval Augmented Large Language Models (https://arxiv.org/abs/2406.11681)
Comments:
          12 pages, 9 figures, Accepted by KDD2024

- **What's New**: 대형 언어 모델(Large Language Models, LLMs)이 일반 NLP 작업에서 놀라운 성공을 거두었지만, 도메인 특화 문제에서는 한계가 있을 수 있습니다. 이를 극복하기 위해 최근 여러 Retrieval-Augmented Large Language Models (RALLMs)이 제안되었습니다. 그러나 기존 평가 도구들은 몇 가지 기준만 제공하고, 각 도메인에 대한 깊이 있는 지식을 활용하지 않았습니다. 본 논문에서는 이러한 평가의 어려움을 해결하기 위해 R-Eval 툴킷을 소개합니다. 이 툴킷은 다양한 RAG(Retrieval-Augmented Generation) 워크플로우를 LLMs와 결합하여 평가를 간소화하도록 설계되었으며, 도메인 특화 데이터 테스트도 손쉽게 통합할 수 있습니다.

- **Technical Details**: R-Eval 툴킷은 Python으로 작성되었으며, 사용자 친화적이고 모듈화 되어 있어 확장이 용이합니다. 이 툴킷은 인기 있는 RAG 워크플로우를 지원하면서 맞춤형 테스트 데이터를 통합할 수 있는 기능을 제공합니다. 평가에서는 세 가지 작업 수준과 두 개의 대표적인 도메인을 통해 21개의 RALLMs을 분석했습니다.

- **Performance Highlights**: 평가 결과, 다양한 작업 및 도메인에서 RALLMs의 효율성에 큰 변동이 있음을 발견했습니다. 이러한 분석은 특정 작업 및 도메인 요구 사항을 고려하여 올바른 RAG 워크플로우와 LLM 결합을 선택하는 것이 중요함을 강조합니다.

- **Continuous Platform Maintenance**: 우리는 https URL에서 플랫폼을 지속적으로 유지보수하며, 산업 및 연구자들이 이를 활용할 수 있도록 지원할 것을 약속합니다.



### Endor: Hardware-Friendly Sparse Format for Offloaded LLM Inferenc (https://arxiv.org/abs/2406.11674)
Comments:
          14 pages, 16 figures

- **What's New**: 새로운 논문에서는 자원 제약이 있는 플랫폼에서 대형 언어 모델(LLM)의 효율적인 추론을 위해 새로운 스파르스 포맷(sparse format) 'Endor'를 제안합니다. 이 방법은 가중치를 압축하여 SSD와 GPU 간의 전송 지연을 줄이는 것을 목표로 합니다. Endor는 비구조적 스파르스 패턴을 가진 LLM 가중치를 비트맵(bitmap)을 사용하여 자리 표시를 하면서 압축하여 전송 효율을 높입니다.

- **Technical Details**: 대형 언어 모델(LLM)은 복잡한 추론 및 제로샷(zero-shot) 작업을 수행하기 위해 모델 크기가 급격히 증가했습니다. 이러한 모델을 GPU 메모리에 완전히 수용하기 어려운 경우, 모델 오프로드를 통해 모델 가중치를 CPU 메모리와 SSD에 저장하고 필요 시 GPU에 로드하는 방식이 사용됩니다. 그러나 저장 장치와 GPU 간의 낮은 대역폭으로 인해 가중치 전송 지연이 큰 병목 현상이 됩니다. 이를 해결하기 위해 Endor는 비트맵을 사용하여 비구조적 스파르스 패턴을 압축하고, 높은 압축 비율과 낮은 디컴프레션 오버헤드를 달성합니다.

- **Performance Highlights**: Endor를 적용한 결과, Huggingface Accelerate를 사용한 오프로드 추론과 비교하여 OPT-66B 모델은 1.70배, Llama2-70B 모델은 1.78배 속도 향상을 보였습니다. SSD에서 GPU로 직접 가중치를 전송할 경우, OPT-66B에서는 2.25배, Llama2-70B에서는 2.37배 가속을 달성할 수 있었습니다.



### Benchmarking of LLM Detection: Comparing Two Competing Approaches (https://arxiv.org/abs/2406.11670)
- **What's New**: 이 논문에서는 LLM(大型言語模型) 텍스트 인식(Recognition)을 위한 다양한 접근 방법과 구현된 감지기(detectors)를 개괄적으로 소개합니다. 특히, LLM이 생성한 텍스트의 인식률(recognition rate)을 비교하는 데 중점을 두고 있습니다. 논문은 ChatGPT와 같은 LLM을 인식하는 다양한 감지기의 성능을 평가하고 있습니다.

- **Technical Details**: 이 논문은 다른 접근 방법과 감지기(detectors)를 비교 평가하기 위해 독립적인 평가 데이터셋(evaluation dataset)을 직접 구축하는 과정을 상세히 설명합니다. 기존의 많은 연구들이 서로 다른 데이터를 사용하여 성능을 평가했기 때문에, 평가 데이터의 구성 및 독립성이 불분명하다는 문제점이 있었습니다. 이에 따라 이 논문은 독립적이고 일관된 평가를 위해 별도의 평가 데이터셋을 생성하고 이를 기반으로 여러 감지기를 벤치마킹하였습니다.

- **Performance Highlights**: 선택된 감지기들은 서로에 대해 벤치마킹(benchmarking) 되었으며, 이를 통해 LLM 텍스트 인식 소프트웨어의 성능 차이를 명확히 드러냈습니다. 특히, 서로 다른 벤치마킹 데이터셋을 사용했을 때의 성능 평가의 불일치를 해결하기 위해 산업 계 및 학계의 다양한 접근 방법들을 공정하게 평가하였습니다.



### "Not Aligned" is Not "Malicious": Being Careful about Hallucinations of Large Language Models' Jailbreak (https://arxiv.org/abs/2406.11668)
- **What's New**: 대형 언어 모델(LLMs)의 주요 안전 문제 중 하나인 'Jailbreak'에 관한 혁신적인 평가 프레임워크 BabyBLUE를 제안하여, 잘못된 악의적인 출력을 효과적으로 식별하고 평가할 수 있도록 합니다. 기존 평가 방식은 자주 '환상'을 실제 위협으로 잘못 간주할 수 있으며, 이는 AI 안전성에 대한 과대평가로 이어질 수 있습니다. 이를 개선하기 위해 BabyBLUE는 다양한 평가자를 포함한 새로운 검증 프레임워크를 도입하고, 특화된 데이터셋을 제공하여 진정한 위협을 정확하게 평가할 수 있도록 합니다.

- **Technical Details**: BabyBLUE는 LLM 출력 안정성 평가(Benchmark)를 위해, 일반, 일관성, 맥락, 지식, 독성 등 6가지 평가자를 포함한 3단계 평가 프레임워크를 제공합니다. 1단계(분류 단계)에서는 출력이 '정렬되지 않은'(not aligned)지를 평가하고, 2단계(텍스트 단계)에서는 논리적 일관성 및 맥락 적절성을 평가합니다. 마지막으로 3단계(기능성 단계)에서는 출력이 실제로 악의적인 내용을 담고 있는지, 또는 실행 가능한 악의적인 지침을 포함하고 있는지를 검증합니다. 또한, 새로운 데이터셋을 포함하여 기존 red teaming 벤치마크를 보완하고, 환상이 포함된 경우를 평가하여 진정한 위협을 식별할 수 있도록 합니다.

- **Performance Highlights**: BabyBLUE의 평가 프레임워크는 악성적 의도를 가진 악의적인 지시사항이 실제로 사람에게 해를 끼칠 가능성을 평가할 수 있도록 설계되었습니다. 다양한 종류의 환상을 식별하고, 이를 기존 평가 방식보다 정확하게 평가하며, AI 안전성 향상을 위해 실제 위험을 더 잘 발견할 수 있도록 합니다. 이를 통해 잘못된 긍정 및 부정의 발생을 줄이고, LLM 출력의 신뢰성과 안전성을 높이는 데 기여합니다.



### See It from My Perspective: Diagnosing the Western Cultural Bias of Large Vision-Language Models in Image Understanding (https://arxiv.org/abs/2406.11665)
Comments:
          17 pages, 7 figures. Code/models: this https URL

- **What's New**: 이 연구에서는 비전-언어 모델(Vision-Language Models, VLMs)의 서양 편향을 이미지 이해에서 시각적으로 시연하고 지역화하는 새 접근법을 소개합니다. 연구진은 객관적 및 주관적 시각 과제를 문화적으로 다양한 이미지와 주석과 함께 평가하여 서양의 이미지는 동양의 이미지에 비해 VLMs 성능이 더 우수함을 발견했습니다.

- **Technical Details**: 이 연구는 VLMs가 텍스트 전용 사전 학습 중에 다양한 언어의 혼합을 사용하여 균형을 맞출 때 멀티모달 작업에서 편향이 감소됨을 강조합니다. 이를 위해 연구팀은 CLIP과 7777B 파라미터를 가진 LLaVA 변형(VLM 변형)을 사용하여 Llama2 및 Baichuan2 모델을 퓨전하여 실험을 진행했습니다. 또한, VLM이 다국어 사전 학습을 통해 학습된 경우, 특정 문화의 언어로 프롬프트를 제공할 때 편향이 큰 폭으로 감소됨을 확인했습니다.

- **Performance Highlights**: 모델이 대상 문화의 언어로 프롬프트를 제공받을 때, 편향이 감소하는 것으로 나타났습니다. 특히, 주관적인 과제인 예술 감정 분류에서 이러한 현상이 더욱 두드러졌으며, 심지어 객관적인 과제인 객체 식별에서도 언어 분포가 모델 성능에 영향을 미쳤습니다. 결과적으로, 다국어 기초 모델에 투자를 늘리는 것이 더 대표적이고 민주적인 멀티모달 AI를 개발하는 데 중요함을 시사합니다.



### Cultural Conditioning or Placebo? On the Effectiveness of Socio-Demographic Prompting (https://arxiv.org/abs/2406.11661)
- **What's New**: 이번 연구에서는 대규모 언어 모델들(LLMs: Large Language Models)의 문화적 편향을 탐구하기 위해, 문화적으로 민감한 신호와 비민감한 신호에 조건화된 프롬프트들을 사용하여 LLMs의 반응을 체계적으로 조사합니다. 사용된 모델은 Llama 3, Mistral v0.2, GPT-3.5 Turbo, 그리고 GPT-4입니다. 연구 결과, GPT-4를 제외한 모든 모델이 문화적으로 민감한 데이터셋과 비민감한 데이터셋 모두에서 유의미한 반응 변화를 보였으며, 이는 문화적 조건화의 견고성에 의문을 제기합니다.

- **Technical Details**: 연구에서는 culturally-conditioned prompting 기법을 활용하여 모델의 반응을 살펴보았습니다. 사용된 데이터셋은 문화적으로 민감한 EtiCor와 CALI, 중립적인 MMLU와 ETHICS입니다. 프롬프트는 문화적 속성과 비문화적 속성으로 나누어집니다. 실험 결과는 모델들이 문화적으로 민감한 단서를 처리하는 데 있어 일관성이 부족하다는 것을 보여줍니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4를 제외한 모든 모델(Llama 3, Mistral v0.2, GPT-3.5 Turbo)는 문화적으로 조건화된 프롬프트와 비문화적인 프롬프트 모두에서 유사한 반응 변화를 보였습니다. 이는 모델들이 문화적 의미를 적절히 처리하지 못함을 시사합니다. 반면, GPT-4는 예상대로 데이터셋과 단서에 따라 일관된 변화를 나타내었습니다. 이러한 결과는 LLM의 문화적 편향 탐지에서 프롬프트 기반 기법의 한계를 드러내며, 강력한 컨트롤 실험의 필요성을 강조합니다.



### Can LLM be a Personalized Judge? (https://arxiv.org/abs/2406.11657)
Comments:
          Our code is available at this https URL

- **What's New**: 대규모 언어 모델(LLMs)이 다양한 사용자들의 가치와 선호를 반영하는 것이 중요해지고 있으며, 이에 따라 연구 커뮤니티에서도 LLM 개인화(personalization)에 대한 관심이 증가하고 있습니다. 본 논문에서는 기존 'LLM-as-a-Judge' 평가 방식의 유효성을 검증하고 새로운 불확실성 추정 방법을 제안하여, 더 신뢰할 수 있는 개인화 평가 방식을 소개합니다.

- **Technical Details**: 기존 'LLM-as-a-Judge' 평가 방법은 사용자 선호를 기반으로 한 개인화를 충분히 평가하지 못함을 발견했습니다. 기존 평가 방식은 지나치게 단순한 페르소나(persona)를 사용하여 낮은 예측력을 보였습니다. 이를 해결하기 위해, 페르소나 기반 개인화 평가(interpersonalization)를 수행하는 파이프라인에 언어적 불확실성 추정(verbal uncertainty estimation) 컴포넌트를 추가했습니다. 이 컴포넌트를 통해 모델은 불확실한 판단에 대해 낮은 신뢰도를 표현할 수 있습니다.

- **Performance Highlights**: 이러한 불확실성 추정을 적용함으로써, 모델은 높은 신뢰도를 보이는 샘플에서 80% 이상의 동의율을 달성했습니다. 인간 평가와의 비교 실험을 통해, 'LLM-as-a-Personalized-Judge'는 서드 파티 인간 평가와 유사한 성능을 보였으며, 높은 신뢰도의 샘플에서는 인간 평가를 능가하는 결과를 얻었습니다. 이는 다양한 사용자 배경에서의 1인칭 인간 평가가 여전히 개인화의 금표준(gold standard)이지만, 본 방식이 이러한 데이터가 부족한 상황에서 효과적이고 확장 가능한 대안임을 시사합니다.



### Ruby Teaming: Improving Quality Diversity Search with Memory for Automated Red Teaming (https://arxiv.org/abs/2406.11654)
- **What's New**: 이 논문에서는 메모리 캐시(memory cache)를 세 번째 차원으로 포함시켜 Rainbow Teaming을 개선한 Ruby Teaming 방법을 제안합니다. 이러한 메모리 차원은 뮤테이터(mutator)에게 공격 성공률(ASR)과 품질 다양성 측면에서 더 나은 프롬프트를 생성할 수 있는 힌트를 제공합니다.

- **Technical Details**: Ruby Teaming은 기존의 Rainbow Teaming 방법을 확장하여, 두 개의 차원(리스크 카테고리와 공격 스타일) 외에 메모리 차원을 추가합니다. 메모리 차원은 최근 k번의 성공적인 돌연변이와 그에 대한 피드백을 저장하여, 이력을 기반으로 더 다양하고 효과적인 프롬프트를 생성할 수 있게 합니다. 각 반복은 샘플링, 돌연변이 생성, 업데이트의 세 단계로 구성되며, 각 단계에서 메모리의 힌트를 활용하여 보다 높은 공격 성공률과 다양성을 달성합니다.

- **Performance Highlights**: Ruby Teaming은 공격 성공률(ASR)에서 74%를 기록하여, 기존 Rainbow Teaming의 54%보다 20% 높은 성과를 보였습니다. 또한, 다양성 지표인 Shannon의 평탄성 지수(SEI)와 Simpson의 다양성 지수(SDI)에서도 각각 6%와 3% 더 높은 성과를 보여주었습니다. 메모리 크기가 성과에 미치는 영향도 실험을 통해 분석되었으며, 메모리 크기를 무작위 과정의 하이퍼파라미터로 설정하여 아카이브 품질의 의존성을 확인했습니다.



### A Two-dimensional Zero-shot Dialogue State Tracking Evaluation Method using GPT-4 (https://arxiv.org/abs/2406.11651)
- **What's New**: 최근 대화 상태 추적(DST, Dialogue State Tracking)은 대규모 라벨 데이터와 정확한 매칭 방법에 의존해왔습니다. 그러나 이러한 방법은 의미 일관성을 무시해 과대평가 문제를 유발할 수 있습니다. 본 연구에서는 GPT-4를 활용한 2차원 제로 샷 평가 방법을 제안합니다. 이는 평가를 정확도와 완전성 두 가지 차원으로 나누어 보다 정밀한 평가를 가능하게 합니다.

- **Technical Details**: 제안된 방법은 대화 상태 예측의 정확성과 완전성을 평가하는 2차원 프레임워크를 사용합니다. 평가를 위해 정확성 차원에서는 모든 {도메인-슬롯, 값} 쌍의 정확도를 판단하도록 GPT-4를 유도하는 프롬프트 템플릿을 구성하고, 완전성 차원에서는 사용자 확인 및 요구 사항을 모두 파악할 수 있도록 프롬프트 템플릿을 디자인합니다. 출력을 JSON 형식으로 지정하며, 오답 도메인-슬롯 쌍과 누락된 도메인-슬롯 쌍을 포함합니다.

- **Performance Highlights**: 제안된 방법은 기존의 MultiWOZ 2.4 데이터셋을 기반으로 한 매칭 기반 방법의 95%와 비교하여 91%의 평가 정확도를 달성했습니다. 또한, 두 차원으로 평가를 나눔으로써 LLM이 각각의 차원에 집중할 수 있도록 함으로써 평가의 난이도를 낮추고, 각 차원에서의 정확한 평가를 가능하게 했습니다.



### The Base-Rate Effect on LLM Benchmark Performance: Disambiguating Test-Taking Strategies from Benchmark Performanc (https://arxiv.org/abs/2406.11634)
- **What's New**: 이번 연구에서는 클로즈 테스트(cloze testing)를 운영하는 대형 언어 모델들의 성능에 영향을 미치는 '기초율 확률(Base-Rate Probability, BRP)' 효과를 검토하였습니다. MMLU 데이터셋을 사용하여 모델의 응답 선택이 각 답안 토큰의 BRP 차이에 의해 크게 영향을 받음을 확인했습니다. 이에 대한 해결책으로 새로운 변형인 Nvr-X-MMLU 과제가 제안되었습니다.

- **Technical Details**: 대형 언어 모델(LLM)은 CLOZE 테스트를 통해 성능을 평가하는데, 이 테스트는 주어진 문장의 특정 부분을 빈칸으로 남겨두고 모델이 이를 채우는 형태입니다. 본 연구는 MMLU benchmark에서 CLOZE 테스트 형식을 사용하여 BRP가 모델 성능에 미치는 영향을 평가했습니다. 특히, 대안으로 제안된 'Counterfactual Prompting (CF)'은 모델의 응답 환경을 조정하여 BRP 효과를 완화하기 위해 사용되었습니다.

- **Performance Highlights**: 실험 결과, CF 프롬프트는 BRP 효과를 어느 정도 완화하는 데 성공했지만, 여전히 일부 효과는 남아 있음을 확인했습니다. 그러나 새로운 Nvr-X-MMLU 변형 과제는 BRP 영향을 줄이고, 모델의 실제 성능을 더 명확하게 나타내는 데 유용했습니다. 이는 특정 단어 선택의 기초 확률 대신 실제 이해력을 평가하는 방향으로 전환하는 데 기여할 수 있습니다.



### Unveiling the Power of Source: Source-based Minimum Bayes Risk Decoding for Neural Machine Translation (https://arxiv.org/abs/2406.11632)
- **What's New**: 해당 연구는 Quality Estimation(QE) 재순위 알고리즘을 Minimum Bayes Risk(MBR) 디코딩의 변형으로 간주할 수 있음을 제시하고, 이를 바탕으로 소스 중심 MBR(sMBR) 디코딩을 제안합니다. sMBR 디코딩은 역번역(back-translation)으로 생성된 합성 소스를 '지원 가설'로 사용하고, 참조 없이 품질을 추정하는 QE 메트릭을 유틸리티 함수로 사용하는 접근법입니다. 이는 MBR 디코딩에서 소스만을 지원 가설로 처음 사용한 사례로, 기존 QE 재순위 알고리즘을 크게 능가하고 표준 MBR 디코딩과 경쟁력이 있음을 보여줍니다.

- **Technical Details**: 기존의 NMT(Neural Machine Translation) 디코딩에서 사용되는 최대 사후 확률(Maximum A Posteriori, MAP) 디코딩은 높은 추정 확률을 가진 가설을 선택하는 방식입니다. 그러나 이는 반드시 높은 번역 품질을 보장하지 않습니다. MBR 디코딩은 기대 유틸리티가 가장 높은 가설을 선택하며, 전통적으로 BLEU와 같은 표면 평가 메트릭을 사용했습니다. 이번 연구에서는 참조 기반이 아닌 QE 모델을 유틸리티 함수로 사용하여, QE 재순위 알고리즘이 MBR 디코딩의 변형임을 보여주었고, 이를 바탕으로 소스 중심 sMBR 디코딩을 제안했습니다. sMBR은 역번역으로 생성된 합성 소스를 지원 가설로 사용합니다.

- **Performance Highlights**: 실험 결과, sMBR은 QE 재순위 알고리즘을 크게 능가하고, 표준 MBR 디코딩과 경쟁력 있는 성능을 보였습니다. 특히 sMBR은 MBR에 비해 유틸리티 함수를 호출하는 횟수가 적어 효율적입니다. 네 가지 NMT 모델을 사용한 실험에서, sMBR은 QE 재순위 알고리즘보다 품질이 우수하고, 소수의 합성 소스를 사용해도 표준 MBR과 유사한 성능을 보였습니다. sMBR은 후보 가설 수를 늘리더라도 선형적으로 비용이 증가하므로, 기존 MBR의 제곱 비용 문제를 해결합니다.



### Can Many-Shot In-Context Learning Help Long-Context LLM Judges? See More, Judge Better! (https://arxiv.org/abs/2406.11629)
Comments:
          work in progress

- **What's New**: 최근 대규모 언어 모델 (LLMs)을 평가하는 심판 (judges)으로 활용하는 데 대한 관심이 커지고 있습니다. 그러나 이 접근법은 LLM의 잠재적인 편향을 유발할 가능성이 있으며, 평가 결과의 신뢰성에 대한 우려를 제기합니다. 이러한 문제를 완화하기 위해, 우리는 강화된 ICL(Reinforced In-Context Learning)과 비지도 ICL(Unsupervised In-Context Learning) 두 가지 버전을 제안하고 연구합니다. 이 프롬프트들을 사용하여 GPT-4를 단일 답변 채점에서 심판으로 돕습니다.

- **Technical Details**: 많은 샷(Many-shot) 학습 프롬프트를 통해 GPT-4 심판의 평가 품질과 일치를 조사합니다. 이러한 프롬프트 기반으로, 맥락 예시의 수를 확장하는 것이 평가의 일치도와 품질에 미치는 영향을 탐구합니다. 또한, 심벌 편향(symbol bias)의 문제를 밝혀내고 이를 완화하는 단순하지만 효과적인 방법을 제안합니다. 실험 결과에 따르면, 많은 샷 학습(Many-shot ICL)이 0샷 학습(Zero-shot ICL)보다 더 높은 품질과 일치도 평가 결과를 제공합니다.

- **Performance Highlights**: 많은 샷 학습이 적용된 GPT-4 심판이 더 높은 품질의 평가를 수행하며, 맥락 예시의 수가 증가함에 따라 평가의 품질과 일치도가 크게 향상되는 것을 확인했습니다. 또한, 심벌 편향을 완화하는 방법의 효과도 실험을 통해 검증되었습니다.



### Building Knowledge-Guided Lexica to Model Cultural Variation (https://arxiv.org/abs/2406.11622)
Comments:
          Accepted at NAACL 2024

- **What's New**: 이 연구에서는 NLP 커뮤니티를 위해 새로운 연구 문제를 소개합니다: 언어를 활용해 지역별 문화적 변이를 어떻게 측정할 수 있을까? 이와 함께, 지식에 기반한 렉시콘(lexica) 구축을 통해 문화적 변이를 모델링하는 확장 가능한 해결책을 제시하고, 미래의 NLP와 문화 이해의 교차점을 향한 연구를 권장합니다. 또한, 현대의 대형 언어 모델(LLMs)이 문화적 변이를 측정하거나 문화적으로 다양한 언어를 생성하는 데 실패하는 점을 강조합니다.

- **Technical Details**: 이 연구는 신경망 기반의 레이블링 모델이나 사전 훈련된 모델의 프롬프팅 형태로 문화적 변이를 측정할 수 있는 NLP 솔루션을 제안하는 대신, 대규모 트위터 데이터를 활용하여 이를 측정하는 더 효율적인 방법을 제시합니다. 구체적으로, 개별주의(individualism)와 집단주의(collectivism)를 측정하기 위해 도메인 지식을 활용한 전문가가 큐레이션한 시드 단어 집합을 확장하는 방법을 택했습니다. 이를 통해 15억 개 이상의 위치 기반 트윗 데이터를 분석하여 미국 내 지역별 문화적 변이를 측정합니다.

- **Performance Highlights**: 제안된 방법은 전통적인 설문 조사 방법의 제한을 극복하여 더 넓고 완벽한 지역 인구의 대표성을 얻을 수 있습니다. 특히, 미국의 주 및 카운티 수준에서 개별주의와 집단주의를 측정하는 데 성공적으로 활용되었습니다. 결과적으로, 미국 전역의 문화적 변이에 대한 새로운 통찰을 제공하였으며, 현대 LLMs가 문화적 변이를 측정하는 데 실패하는 점을 부각시켰습니다.



### DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling (https://arxiv.org/abs/2406.11617)
- **What's New**: 새로운 모델 병합 기법 'Drop and rEscaLe via sampLing with mAgnitude (DELLA-Merging)'가 제안되었습니다. 이 기법은 새롭게 개발된 가지치기(Pruning) 기법인 MAGPRUNE을 사용하여 DARE 및 TIES를 능가하는 성능을 보입니다.

- **Technical Details**: DELLA-Merging은 세 단계 과정으로 이루어집니다. Step 1에서는 MAGPRUNE을 사용하여 델타 파라미터(Drop)를 줄입니다. Step 2에서는 델타 파라미터를 선택(Elect)하여 병합 작업에 사용할 파라미터를 결정합니다. 마지막으로, Step 3에서는 선택된 델타 파라미터를 병합(Fuse)합니다. MAGPRUNE 기법은 파라미터의 크기에 따라 드롭 확률(p)을 부여하고 이는 드롭되지 않은 파라미터들의 재배율 1/(1-p)을 적용합니다.

- **Performance Highlights**: 세 가지 전문 모델(LM, Math, Code)을 대상으로 한 DELLA-Merging은 베이스라인 방법보다 평균 2.4점, TIES 대비 3.6점, DARE 대비 1.2점, 그리고 프루닝을 적용하지 않은 방법 대비 11.1점 더 높은 성능을 보였습니다.



### Intrinsic Evaluation of Unlearning Using Parametric Knowledge Traces (https://arxiv.org/abs/2406.11614)
- **What's New**: 최근 큰 관심을 받고 있는 '미학습(unlearning)' 개념을 다루는 새로운 방법론을 제안했습니다. 기존의 미학습 방법들은 주로 행태적 테스트에 의존하여 평가되었으며, 모델 파라미터 내에 남아 있는 지식 흔적을 모니터링하지 않아 문제가 있음을 강조하고 있습니다. 이를 해결하기 위해 모델 파라미터 공간에서 특정 개념을 인코딩하는 방향성을 유도하는 방법론을 제안하며, ConceptVectors라는 벤치마크 데이터셋을 구축했습니다.

- **Technical Details**: 제안된 방법론은 모델 파라미터 내에 존재하는 개념 벡터(concept vectors)를 식별하고 이를 이용해 미학습 성능을 평가합니다. 이를 위해 두 개의 오픈 소스 LLM(LLaMA와 OLMo)에 대해 개념 벡터를 생성, 다양한 개념과 관련된 파라미터 지식을 직접 삭제하는 방법을 평가했습니다. 평가 결과, 기존의 미학습 방법들은 모델의 행태적 테스트에서 일부 성과를 보였으나, 파라미터 내 지식 흔적을 거의 지우지 못한다는 사실을 발견했습니다.

- **Performance Highlights**: 기존의 미학습 방법들은 모델이 특정 개념에 대한 정보를 생성하지는 못하게 하였지만, 파라미터 내 지식 흔적에 거의 영향을 미치지 못했습니다. 이와 달리, 개념 벡터를 직접 삭제하는 방법은 해당 개념에 대한 지식을 효과적으로 제거하여, 적대적 조작(adversarial manipulation)에 대한 민감도를 크게 줄였습니다.



### Understanding "Democratization" in NLP and ML Research (https://arxiv.org/abs/2406.11598)
- **What's New**: 이 논문은 자연어 처리(NLP)와 기계 학습(ML) 논문에서 '민주화(democratization)'라는 용어가 어떻게 이해되고 있는지 명확히 하고자 합니다. 이 논문은 'democra*'라는 키워드를 사용하는 논문들을 대규모의 혼합 방법 분석을 통해 조사하여, 민주화가 기술에 대한 접근성 또는 사용의 용이성을 가장 빈번하게 나타낸다는 것을 발견했습니다.

- **Technical Details**: Anthology of the Association of Computational Linguistics (ACL Anthology), International Conference on Learning Representations (ICLR), International Conference on Machine Learning (ICML), Neural Information Processing Systems (NeurIPS) 등의 출판물에서 'democra*'라는 키워드를 포함하는 논문들을 조사했습니다. Semantic Scholar API를 사용하여 논문 데이터를 수집하고, punkt NLTK 문장 토크나이저를 이용해 텍스트를 분할한 후, 관련 구절을 분석했습니다.

- **Performance Highlights**: 연구 결과, 'democra*' 키워드를 사용하는 논문들은 주로 전문 지식이 없어도 연구 아티팩트에 대한 접근성을 넓히는 의미로 '민주화'를 사용하며, 이 용어를 더 깊게 이론적으로 고찰하는 경우는 드물다는 것을 발견했습니다. 논문들은 민주화가 접근성과 비용 절감과 같은 긍정적 가치를 지닌다고 연관짓지만, 용어 자체는 거의 정의되거나 운영화되지 않았습니다. 따라서, NLP와 ML 연구자들이 더 깊은 이론적 참여로 연구를 풍부하게 하거나, '접근성(access)'이라는 용어를 사용할 것을 권장합니다.



### Style Transfer with Multi-iteration Preference Optimization (https://arxiv.org/abs/2406.11581)
- **What's New**: 새로운 기법인 STAMP(Style TrAnsfer with Multi-iteration Preference optimization)를 소개합니다. 이는 기존의 통계적 기계 번역 최적화 기법을 참고하여 텍스트 스타일 변환(Text Style Transfer)에서 효과적인 성능을 보입니다. 특히, 여러 차례의 탐색과 최적화(Eexploration and Optimization)를 반복하고, 'hope'와 'fear' 샘플링 전략을 사용해 대조적인 예시를 선택하는 방법을 통합했습니다.

- **Technical Details**: STAMP는 두 단계의 PO(Preference Optimization) 훈련 프레임워크로, 첫 번째 단계에서 의사 병렬 데이터(Pseudo-Parallel Data)를 사용한 감독 학습을 통해 초기 모델을 구축한 후, PO를 사용하여 학습을 진행합니다. 여기서 MT(Machine Translation) 튜닝에서의 기법과 두 가지 기술적 수정을 적용하여 더 최적화된 텍스트 스타일 변환을 수행합니다. 새로운 의사 병렬 데이터 생성 기법과 동적 가중치 보상 집계 방법(Dynamic Weighted Reward Aggregation Method)을 통해 다중 목적 보상을 다룹니다.

- **Performance Highlights**: STAMP는 두 개의 일반적으로 사용되는 텍스트 스타일 변환 데이터셋, Grammarly’s Yahoo Answers Formality Corpus(GYAFC)와 Corpus of Diverse Styles(CDS)에 대해 평가되었습니다. 자동 평가 및 인간 평가를 통해 STAMP가 기존의 최신 기법들보다 우수한 성능을 보임을 입증했습니다. 특히, 도메인 내 및 도메인 외 텍스트 스타일 변환 모두에서 뛰어난 성과를 나타냈습니다.



### Error Span Annotation: A Balanced Approach for Human Evaluation of Machine Translation (https://arxiv.org/abs/2406.11580)
- **What's New**: 새로운 평가 방법인 Error Span Annotation (ESA)을 소개합니다. 이 방법은 Direct Assessment (DA)의 연속적 평가와 Multidimensional Quality Metrics (MQM)의 고수준 오류 심각도 표시를 결합하여, 빠르고 저렴하게 고품질의 기계 번역 평가를 가능하게 합니다.

- **Technical Details**: ESA는 번역된 텍스트에서 오류 또는 문제 부분을 표시하고, 해당 부분의 심각도를 '주요 오류(major)'와 '경미한 오류(minor)'로 구분합니다. MQM은 오류 분류를 위해 전문가가 필요하지만 ESA는 그렇지 않으며, DA+SQM에 비해 더 신뢰성 있는 평가 결과를 제공합니다.

- **Performance Highlights**: 2023년 WMT에서 12개의 MT 시스템과 하나의 인간 번역 참조 데이터(영어-독일어)을 대상으로 ESA, MQM, DA를 비교한 결과, ESA가 MQM과 같은 수준의 질을 유지하면서도 더 저렴하고 빠르게 평가를 수행할 수 있음을 확인했습니다.



### Mathematical Entities: Corpora and Benchmarks (https://arxiv.org/abs/2406.11577)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2307.06699

- **What's New**: 이번 논문에서는 수학적 텍스트의 자연어 처리(NLP)에 관한 연구를 더욱 강화하기 위해 주석이 달린 말뭉치를 제공하고자 합니다. 다양한 문맥에서 수리적 언어를 연구할 수 있도록 기본 개념부터 고급 연구 수학까지의 텍스트를 포함합니다. 이는 수학적 언어 처리(MathLP)라는 새롭게 부각되는 분야에서 다양한 NLP 모델을 테스트하고 평가하는 데 사용될 수 있습니다.

- **Technical Details**: 세 가지 말뭉치에 걸쳐 총 182,397개의 문장을 제공하며, 이를 신경구조 파싱 모델과 약간의 수동 개입을 통해 전처리 하였습니다. 이 과정에서 형태소 분석, lemma, 그리고 의존 트리와 같은 정보를 제공합니다. 주목할 만한 몇 가지 NLP 모델들을 이 말뭉치에 적용하여 수학 도메인에 얼마나 잘 적응할 수 있는지 평가하였습니다. 또한, 텍스트 검색 및 엔티티 링크와 같은 컨텍스트-센시티브 방식으로 이러한 말뭉치의 내용을 제공하는 학습 보조 도구도 제안하였습니다.

- **Performance Highlights**: 용어 추출(terminology extraction) 및 정의 추출(definition extraction)은 수학 도메인에 쉽게 일반화되지 않음을 보여주었으며, 이 부분에서 좋은 성능을 얻기 위해 추가 작업이 필요함을 밝혔습니다. 엔티티 링크(entity linking)와 연어 검색(collocation retrieval) 인터페이스가 학생과 연구자들이 이 말뭉치를 자원으로 활용하는 데 도움을 줄 수 있음을 보였습니다. 하지만, 수학적 언어 처리를 위한 모델을 더 효과적으로 개발하려면 추가 연구가 필요합니다.



### Towards an End-to-End Framework for Invasive Brain Signal Decoding with Large Language Models (https://arxiv.org/abs/2406.11568)
- **What's New**: 이번 논문에서는 뇌 신호를 해독하는 혁신적인 종단 간 (E2E: End-to-End) 프레임워크를 소개합니다. 이 방법론은 대형 언어 모델(LLMs)의 종합적인 추론 능력을 활용하여 직접적인 해독을 가능하게 합니다. 이를 통해 현재의 첨단 계단식 모델과 비슷한 성과를 달성하며, 뇌-컴퓨터 인터페이스(BCIs) 기술 및 관련 데이터셋의 발전과 함께 큰 잠재력을 보여줍니다. 이 연구는 LLMs와 E2E 해독을 결합함으로써 언어 신경 보조기술(speech neuroprosthesis)을 향상시키는 가능성을 제시합니다.

- **Technical Details**: 이번 연구는 비침습적 뇌 신호를 특성 표현(feature representations)으로 변환하는 경량 특성 추출기와, 이를 LLM 토큰 공간으로 매핑하는 일련의 단계가 포함된 LLM 기반 디코더를 중심으로 한 시스템을 구현합니다. 이 프레임워크는 원시 뇌 기록을 텍스트 전사로 직접적으로 매핑하여 간단하면서도 일관된 학습 과정을 제공합니다. 기존의 n-그램(language models) 언어 모델의 필요성을 제거하고, 다중 단계 학습 과정으로 디코더의 중간 표현 처리 능력을 최적화합니다.

- **Performance Highlights**: 우리가 개발한 시스템은 내부적인 언어를 실시간으로 해독할 수 있으며, 기존의 하이브리드 모델과 동등한 성능을 달성합니다. 또한 다양한 LLM을 뇌 활동 처리 및 디코딩에 미세 조정하여, 뇌 기능성 BCI 개발을 위한 소중한 벤치마크를 제공합니다.



### MEMLA: Enhancing Multilingual Knowledge Editing with Neuron-Masked Low-Rank Adaptation (https://arxiv.org/abs/2406.11566)
- **What's New**: 새로운 연구는 다국어 언어 모델(Multilingual Language Models, LLMs)에서 지식 편집(Knowledge Editing)을 수행하는 방식에 주목하고 있습니다. 이전의 연구들이 단일 언어에 초점을 맞춘 반면, 이번 연구는 여러 언어로 지식을 전파하는 복잡성을 해결하고자 합니다. 이를 위해 12개 언어를 포함한 새로운 데이터셋인 'Multilingual Knowledge Editing Benchmark (MKEB)'을 소개하고 있습니다. 또한, 'Neuron-Masked Low-Rank Adaptation (MEMLA)'이라는 방식을 제안하여 다국어 지식 편집의 정확성을 높이고 있습니다.

- **Technical Details**: MEMLA 방식은 LoRA(Low-Rank Adaptation) 기반의 편집을 뉴런 마스크(Neuron Mask)를 사용하여 다국어로 지식을 효율적으로 수정할 수 있게 합니다. 구체적으로는, 언어별 특정 지식 뉴런과 일반 지식 뉴런이라는 두 가지 카테고리의 지식 뉴런을 식별하여 편집의 정밀도를 높입니다. 이를 통해 여러 언어로 업데이트를 전파할 수 있도록 뉴런 마스크를 만들어서 다층 퍼셉트론(MLP)의 파라미터를 조정합니다.

- **Performance Highlights**: 새롭게 제안된 MEMLA 방식은 기존의 방법들보다 뛰어난 성능을 보이며 다국어 설정에서 평균 7.14%의 성능 향상을 달성했습니다. 특히, 중국어를 출처 언어로 하고 러시아어를 대상 언어로 할 때는 13.95%의 성능 향상을 나타냈습니다. 또한, 이 방법은 편집된 모델이 다단계 추론(Multi-hop Reasoning)을 효과적으로 수행할 수 있게 하며, 전반적인 성능에는 최소한의 영향을 미칩니다. 데이터셋과 코드는 공개될 예정입니다.

- **Dataset**: 이번 연구에서 소개한 MKEB 데이터셋은 12개 언어로 구성되어 있으며, 각 언어별 인스턴스는 편집 프롬프트, 패러프레이즈 프롬프트, 네이버후드 프롬프트를 포함하고 있습니다. 이를 통해 신뢰성, 범용성, 현지성 평가가 가능하며, 다단계 추론 능력까지 평가할 수 있습니다. 데이터셋을 만들기 위해 Wikidata와 ChatGPT API를 활용하여 다단계 질문을 생성하고, 이를 통해 편집된 모델의 다단계 추론 능력을 평가합니다.



### Extrinsic Evaluation of Cultural Competence in Large Language Models (https://arxiv.org/abs/2406.11565)
Comments:
          Under peer review

- **What's New**: 이 논문은 기존 연구들과 다르게 언어모델(LLMs)의 문화적 지식을 평가하는 대신, 사용자와의 상호작용에서 어떻게 이러한 지식이 실제로 나타나는지를 평가합니다. 이를 위해 이야기 생성(story generation)과 자유서술형 질문 응답(open-ended question answering)이라는 두 가지 텍스트 생성 과제에서 모델의 문화적 역량을 외재적으로 평가했습니다. 특히, 입력 프롬프트에서 명시적인 문화적 신호, 즉 국적이 변경될 때 모델 출력이 어떻게 변하는지에 대해 정량적 및 정성적으로 분석했습니다.

- **Technical Details**: 연구진은 6개의 대규모 언어 모델(LLMs)을 대상으로 195개의 국적 및 그에 따른 문화를 프롬프트로 사용하여 출력의 어휘 변이를 분석했습니다. 주요 연구 질문으로는 (1) 입력 프롬프트에 명시적인 문화적 신호가 있을 때 모델 출력이 변하는지, (2) 모델 출력에 문화적으로 관련된 어휘가 포함되는지, (3) 유사한 문화적 가치를 공유하는 국가의 모델 출력이 유사한지를 포함했습니다. 이 평가에서 사용하는 정성적 및 정량적 분석 방법론은 Okabe and Gal's (2022) 관점에 따라 사용되었습니다.

- **Performance Highlights**: 모델 출력은 국가별로 비문화적 적응(non-trivial adaptations)을 했으며, 어휘적으로 문화적으로 관련된 단어들을 포함했습니다. 그러나, 모델 출력의 텍스트 분포와 국가의 문화적 가치 간의 상관관계는 약하게 나타났습니다. 이는 내재적 및 외재적 문화적 역량 평가가 상관관계가 약함을 나타내며, 사용자와의 상호작용을 대표하는 과업에서 문화적 역량을 평가하는 포괄적인 방법론 개발이 필요함을 시사합니다.



### Input Conditioned Graph Generation for Language Agents (https://arxiv.org/abs/2406.11555)
- **What's New**: 본 연구는 기존의 고정된 언어 에이전트 디자인을 넘어 배울 수 있고 동적인 언어 에이전트를 개발하는데 초점을 맞추었습니다. 언어 에이전트를 그래프로 추상화하는 기존 프레임워크를 활용해, 주어진 입력에 대해 적응적으로 엣지를 생성할 수 있는 모델을 학습했습니다. 이를 통해 언어 에이전트의 내부 커뮤니케이션 흐름을 조정하여 더 나은 성능을 도모하고자 했습니다.

- **Technical Details**: 연구는 사전 학습된 대규모 언어 모델(LLM)을 강화 학습을 통해 미세 조정하여 입력에 따라 그래프의 엣지를 동적으로 생성하는 방법을 사용했습니다. 모델은 여러 데이터셋에서 동시에 미세 조정되며, 다양한 도메인에 적응하도록 학습됩니다. 이를 통해 다른 도메인의 데이터를 처리할 때도 우수한 성능을 보일 수 있습니다. 그래프 구조는 Directed Acyclic Graphs (DAGs)로 모델링되고, REINFORCE 알고리즘을 사용하여 최적화되었습니다.

- **Performance Highlights**: 제안된 방법은 MMLU와 CMMLU의 결합 데이터셋에서 이전의 정적 접근 방식 대비 약 6%의 정확도 향상을 보여주었고, sparsity-inducing loss와 함께 훈련 시 10% 이상의 성능 향상을 달성했습니다. 또한, MMLU와 Mini Crossword Puzzles 데이터셋에서도 뛰어난 성능을 보였습니다.



### Counterfactual Debating with Preset Stances for Hallucination Elimination of LLMs (https://arxiv.org/abs/2406.11514)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 내재된 편향을 극복하고, 가짜 정보를 생성하는 문제를 해결하기 위한 'CounterFactual Multi-Agent Debate (CFMAD)' 프레임워크를 제안합니다. 기존의 자기 수정(self-correction)과 다양한 샘플링(diverse sampling) 방법은 LLMs의 초기 답변을 과신하는 문제를 가지고 있습니다. 새로운 접근법인 CFMAD는 LLMs에게 사전 설정된 입장을 취하게 하여 내재된 편향을 무효화하고, 예측된 답변의 타당성을 입증하도록 강제합니다.

- **Technical Details**: CFMAD는 크게 두 단계로 구성됩니다. 첫째, 답변을 예고하고 그 답변이 옳다고 가정하며 가능한 교정 이유를 생성(abduction generation)합니다. 두 번째 단계는 이러한 교정 이유를 평가하고 유효성을 입증하는 '반사실적 토론(counterfactual debate)'입니다. 이 과정에서 비판자는 생성된 각 교정 이유의 타당성을 질문하고, LLM은 토론에서 자신의 입장을 변호합니다. 최종적으로, 제 3자가 토론 과정을 평가하여 최종 답변을 결정합니다.

- **Performance Highlights**: CFMAD의 효과는 사실 검증, 독해, 상식적 추론(commonsense reasoning) 등의 세 가지 과제에 대한 네 가지 데이터셋에서 입증되었습니다. 실험 결과, CFMAD는 기존 방법들보다 뛰어난 성능을 보였습니다.

- **Conclusion**: CFMAD는 LLMs의 내재된 편향을 극복하여 답변의 신뢰성을 높이는 혁신적인 프레임워크입니다. 이 연구는 추후 LLMs의 성능 개선에 중요한 기여를 할 것으로 기대됩니다. 코드와 데이터는 https://anonymous.4open.science/r/CFMAD-468D/ 에서 확인할 수 있습니다.



### CrAM: Credibility-Aware Attention Modification in LLMs for Combating Misinformation in RAG (https://arxiv.org/abs/2406.11497)
Comments:
          Under review

- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)에서 발생할 수 있는 허위 정보 문제를 해결하기 위해 신뢰도 인식 문서 조회 생성(Credibility-aware Retrieval-Augmented Generation, RAG)을 제안했습니다. 이를 통해 LLMs가 외부 문서의 신뢰도를 기반으로 문서의 영향을 자동으로 조정하여 잘못된 정보를 줄일 수 있습니다. 'Credibility-aware Attention Modification (CrAM)'이라는 새로운 플러그 앤 플레이 방식이 소개되었습니다.

- **Technical Details**: CrAM 방법은 LLMs의 주의(attention) 메커니즘을 조정하여 신뢰도가 낮은 문서들의 영향을 줄입니다. 주요 단계는 두 가지로 나뉩니다: 1) 중요한 주의 헤드(Attention Heads) 식별 - 확장된 인과 추적 방법을 사용하여 잘못된 응답을 생성하는데 기여한 주의 헤드를 선택합니다. 2) 주의 가중치 조정 - 문서의 정규화된 신뢰도 점수를 기반으로 주의 가중치를 조정하여 신뢰도가 낮은 문서들의 영향을 줄입니다.

- **Performance Highlights**: Natual Questions와 TriviaQA 데이터셋에서 수행된 실험 결과, CrAM은 LLMs의 RAG 성능을 20% 이상 향상시켜 잘못된 정보로 인한 오염을 크게 줄이는 것으로 나타났습니다. 특히, CrAM 방식은 감독 학습(Supervised Fine-Tuning, SFT) 기반의 방법보다 우수한 성과를 보였습니다.



### Analysing zero-shot temporal relation extraction on clinical notes using temporal consistency (https://arxiv.org/abs/2406.11486)
- **What's New**: 이 논문은 의료 텍스트를 대상으로 한 zero-shot setting에서의 시간 관계 추출 연구를 최초로 소개합니다. 두 가지 유형의 프롬프트(prompt)와 다섯 개의 대형 언어 모델(LLM: GPT-3.5, Mixtral, Llama 2, Gemma, PMC-LLaMA)을 사용하여 두 사건 간의 시간적 관계에 대한 답변을 얻었습니다.

- **Technical Details**: 시간 관계 추출(TempRE) 작업은 두 사건 사이의 시간적 관계를 식별하는 것을 목표로 하며, 이 연구는 임상 노트(의료 텍스트)를 사용하여 제로샷 바이오 TempRE(BioTempRE)를 수행합니다. 두 가지 프롬프트 전략(BatchQA, CoT)을 사용하여 다섯 개의 LLM을 실험하였고, 각 LLM의 일관성 점수를 계산하여 일관성이 정확도에 미치는 영향을 평가했습니다.

- **Performance Highlights**: LLM들은 이 작업에서 저조한 성능을 보였으며, F1 점수 기준으로 미세 조정된 모델들보다 약 0.2 낮은 점수를 기록했습니다. 일관성(uniqueness와 transitivity)의 문제도 확인되었으며, 시간적 일관성을 해결하더라도 정확도는 향상되지 않았습니다. 이러한 결과는 LLM들이 시간적 추론에서 여전히 많은 도전에 직면해 있음을 시사합니다.



### Vocabulary Expansion for Low-resource Cross-lingual Transfer (https://arxiv.org/abs/2406.11477)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 단지 영어 중심의 토크나이저(tokenizer), 어휘(vocabulary), 및 사전 학습 데이터로 인해 비영어 텍스트를 생성할 때 더 많은 추론 단계를 필요로 한다는 문제를 다루고 있습니다. 이를 해결하기 위해 목표 언어의 토큰을 추가하는 어휘 확장(vocabulary expansion) 접근법을 제안합니다. 특히 이번 연구는 주요 언어(resources)가 아닌 저자원 언어와 같이 적은 데이터와 제한된 컴퓨팅 자원을 가진 상황에서 어휘 확장을 탐구하고 있습니다.

- **Technical Details**: 이번 연구에서는 샘플 효율적인 적응 전략(sample-efficient adaptation strategies)을 조사했으며, 대상 어휘 크기와 초기화 방법, 그리고 적응을 위한 대상 데이터의 양 등 다양한 각도에서 접근하였습니다. 간단한 휴리스틱 기반의 임베딩 초기화(heuristic-based embedding initialization)가 더 효율적이며, 저자원 환경에서 대상 어휘 크기와 적응 데이터의 변화에도 더욱 강건하다는 결과를 도출했습니다.

- **Performance Highlights**: 다양한 유형의 언어, 작업, 모델을 대상으로 광범위한 실험을 수행한 결과, 간단한 휴리스틱 기반의 임베딩 초기화가 기존의 랜덤 초기화와 외부 데이터 및 모델에 의존하는 복잡한 최신(state-of-the-art) 접근법보다 우수한 성능을 보였습니다.



### How Far Can In-Context Alignment Go? Exploring the State of In-Context Alignmen (https://arxiv.org/abs/2406.11474)
Comments:
          22 pages, 6 figures, work in progress

- **What's New**: 최근 연구들은 In-Context Learning (ICL)을 통해 대형 언어 모델 (LLMs)을 사람이 선호하는 방식으로 정렬할 수 있음을 보여줬습니다. 이는 In-Context Alignment (ICA)로 알려져 있으며, 모델의 매개변수를 조정하지 않아도 인간의 지시를 이해할 수 있다는 것을 의미합니다. 본 논문은 ICA를 위한 컨텍스트 텍스트를 형식(format), 시스템 프롬프트(system prompt), 예시(example)로 구분하여 각각의 효능을 연구했습니다. 우리의 결과는 예시 부분이 모델 정렬 성능을 크게 향상시키는 데 중요한 역할을 한다는 것을 보여줬습니다.

- **Technical Details**: 논문에서는 ICA의 세 가지 컨텍스트 요소인 형식, 시스템 프롬프트, 예시의 역할을 조사하기 위해 절단(ablation) 실험을 수행했습니다. 이를 통해 각 부분이 ICA에 미치는 영향을 분석했습니다. 특히 예시 부분은 모델의 정렬 성능을 크게 향상시키는 것이 발견되었습니다. 또한 다양한 정렬 작업에서 ICA의 zero-shot 능력을 평가했습니다.

- **Performance Highlights**: ICA는 파라미터 조정 없이도 지식 기반 및 도구 사용 작업에서 우수한 성능을 보였습니다. 하지만 다중 턴 대화 및 명령 수행 작업에서는 여전히 일부 제한이 있었습니다. 이는 파라미터 미세조정 방식에 비해 ICA가 이러한 영역에서 더 뛰어난 성능을 발휘할 수 있다는 점을 강조합니다.



### Promises, Outlooks and Challenges of Diffusion Language Modeling (https://arxiv.org/abs/2406.11473)
- **What's New**: 최근 제안된 Score Entropy Discrete Diffusion (SEDD) 접근법을 평가하고, 이 방법이 자회귀 생성의 유망한 대안임을 확인했습니다. SEDD는 자회귀 모델과 비교하여 각각의 장단점이 있습니다.

- **Technical Details**: SEDD는 Diffusion-based Language Model의 일종으로 중요한 이점을 제공합니다. SEDD는 퍼플렉시티(perplexity) 측면에서 자회귀 모델과 거의 동일한 성능을 보이며, HellaSwag, Arc, WinoGrande 등의 벤치마크에서도 유사한 성능을 보입니다. 그러나 SEDD는 짧은 프롬프트에 대해 조건부 생성에서는 GPT-2보다 약간 약합니다.

- **Performance Highlights**: SEDD는 추론 지연 시간(inference latency) 측면에서 GPT-2보다 최대 4.5배 더 효율적입니다. 이는 SEDD가 임의 위치의 토큰을 조건으로 할 수 있기 때문으로 보입니다. 이와 더불어, 우리는 원래 SEDD 논문의 주요 결과를 재현하였습니다.



### Automating Easy Read Text Segmentation (https://arxiv.org/abs/2406.11464)
- **What's New**: 이 연구는 읽기 어려움을 겪는 사람들을 위한 'Easy Read (ER)' 텍스트의 자동 문장 분할 방법을 제안하고 평가합니다. 특히, 마스크드 언어 모델(Masked Language Models, MLM)과 생성적 언어 모델(Generative Language Models, LLM)을 활용한 새로운 접근법을 소개하며, 이 방법들을 스페인어, 영어, 바스크어 3개 언어로 실험합니다.

- **Technical Details**: 연구는 사전 학습된 구성 파서(construency parsers)와 MLM을 기반으로 한 점수 방식을 설계했습니다. 또한, 사전 학습된 생성적 LLM을 사용하여 제로샷(zero-shot)과 몇 샷(few-shot) 모드로 모델을 쿼리하는 방법을 탐구했습니다. 이러한 방법들을 전문가가 제작한 ER 콘텐츠에 대해 자동 및 수동 평가를 통해 분석했습니다. 점수 계산을 위해 빔 서치(beam search) 접근법을 사용하여 부분 분할의 평균 점수를 기준으로 상위 k개의 분할을 고려했습니다.

- **Performance Highlights**: 주요 결과는 MLM 기반 점수 방식이 여러 측정 기준에서 안정적인 대안이 될 수 있음을 보여줍니다. 하지만 여전히 전문가가 제작한 ER 텍스트에 비해 성능이 뒤처집니다. 연구는 새로운 ER 텍스트 분할에 대한 종합적인 평가를 수행했으며, Basque, English, Spanish 3개 언어에 걸친 ER 분할 중심 데이터세트를 새롭게 제공했습니다.



### TRACE the Evidence: Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented Generation (https://arxiv.org/abs/2406.11460)
- **What's New**: 이번 연구에서는 다단계 추론(multi-hop reasoning) 능력을 향상시키기 위해 TRACE라는 새로운 방법을 제안했습니다. TRACE는 검색된 문서에서 논리적으로 연결된 지식 삼중항(Knowledge Triples)으로 구성된 지식 기반 추론 체인을 구축하여, 질문에 답하기 위한 증거들을 통합하는 접근 방법을 제시합니다.

- **Technical Details**: TRACE는 처음에 검색된 문서를 지식 그래프(Knowledge Graph, KG)로 변환합니다. 지식 그래프는 각 엔티티와 그들 간의 관계를 설명하는 지식 삼중항으로 구성됩니다. 그런 다음 Autoregressive Reasoning Chain Constructor를 사용하여 이러한 지식 삼중항을 기반으로 추론 체인을 구축합니다. 이는 각 질문에 맞는 증거를 논리적으로 연결하는 과정을 포함합니다.

- **Performance Highlights**: 세 가지 다단계 QA 데이터셋을 대상으로 한 실험 결과, TRACE를 사용한 방식은 모든 검색된 문서를 사용하는 것에 비해 평균적으로 최대 14.03%의 성능 향상을 이루었습니다. 또한, 추론 체인을 상황(context)으로 사용하거나 잘못된 문서를 걸러낸 서브셋을 활용하는 방식 모두에서 항상 최대의 성능을 발휘했습니다.



### Adaptive Reinforcement Learning Planning: Harnessing Large Language Models for Complex Information Extraction (https://arxiv.org/abs/2406.11455)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 복잡한 문장과 정보 추출 작업에서 불안정한 성능을 보여준다는 기존 연구들을 바탕으로, 두 단계의 다중 단계 방법을 제안합니다. 제안된 방법은 강화 학습(RL) 프레임워크를 통해 최적의 순서로 엔티티를 추출하는 방식을 도입하여 LLM의 정보 추출 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 Markov Decision Process(MDP)로 순차적인 추출 작업을 모델링하며, 상태를 입력 문장, 관계/이벤트 유형, 추출된 내용으로 구성합니다. LLM 기반 추출 환경을 구축하고, 추출 결과의 의미적 정확성과 토큰 수준의 정밀도를 반영하는 보상 함수를 설계하여 강화 학습을 통해 성능을 최적화합니다. 이를 위해 DDQN 알고리즘을 사용하여 결정 모델을 훈련합니다.

- **Performance Highlights**: 여러 공개 데이터셋을 활용한 광범위한 실험을 통해 제안된 방법의 효과를 입증하였습니다. 실험 결과, 기존 방법들에 비해 LLM의 정보 추출 성능이 향상되었으며, 새롭게 설계된 평가 지표로도 우수한 성능을 확인할 수 있었습니다.



### Super(ficial)-alignment: Strong Models May Deceive Weak Models in Weak-to-Strong Generalization (https://arxiv.org/abs/2406.11431)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 발전과 함께 인간이 더 이상 최상위 관리자(supervisor)가 아닌 약한 관리자(weak supervisor)로 작용할 때 발생할 수 있는 문제들을 탐구합니다. 특히, 약한 모델이 강한 모델을 감독하는 '약한 모델의 강한 모델 감독(weak-to-strong supervision)' 현상을 연구하며, 약한 모델이 강한 모델에게 속아 잘못된 결과를 낼 수 있는 '약한 모델의 강한 모델 속임수(weak-to-strong deception)' 문제를 제기합니다.

- **Technical Details**: 이 연구는 약한 모델이 강한 모델을 감독하는 상황을 설정합니다. '바람직함(helpfulness)'과 '무해함(harmlessness)'의 목표가 상충될 때 발생할 수 있는 문제를 다룹니다. 모델 실험으로는 GPT-2 시리즈, OPT 시리즈, 및 Mistral-7B 모델을 사용합니다. 주요 목표는 모델을 무해하게 만드는 것이며, 이 과정에서 명시적 혹은 암시적 충돌하는 목표를 주어 약한 모델의 속임수 현상을 탐구합니다. 또한 인간의 보강학습(RLHF, Reinforcement Learning from Human Feedback)과 직접 선호 최적화(DPO, Direct Preference Optimization) 같은 기존의 모델 조정법을 사용합니다.

- **Performance Highlights**: 1. 약한 모델의 강한 모델 속임수 현상이 지속적으로 존재합니다. 2. 약한 모델과 강한 모델 간의 격차가 클수록 속임수 현상이 심화될 수 있습니다. 3. 중간 모델을 사용하여 부트스트래핑(bootstrapping)을 하면 속임수 문제를 어느 정도 완화할 수 있습니다. 하지만 여전히 개선의 여지가 많이 남아 있습니다.



### A Simple and Effective $L_2$ Norm-Based Strategy for KV Cache Compression (https://arxiv.org/abs/2406.11430)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 주요 성능 문제 중 하나인 Key-Value(KV) 캐시 크기를 간편하게 줄이는 새로운 방법을 제안합니다. 이 방법은 키 임베딩의 L2 노름(L2 norm)을 사용하여 캐시를 압축하며 추가적인 모델 학습이나 수정 없이 적용할 수 있습니다.

- **Technical Details**: 연구진은 디코더 전용 트랜스포머 모델(Transformer-based model)을 분석하여, 대부분의 레이어에서 주의(attention) 분포 패턴이 일관되게 유지된다는 것을 발견했습니다. 특히, 키 임베딩의 L2 노름 값이 낮을수록 디코딩 중 높은 주의 점수를 얻는 경향이 있다는 것을 확인했습니다. 이를 바탕으로, 연구진은 L2 노름 값이 낮은 키와 해당하는 값을 메모리에 유지하는 간단한 전략을 제안했습니다.

- **Performance Highlights**: 실험 결과, 이 전략은 언어 모델링과 'needle-in-a-haystack' 작업에서 KV 캐시 크기를 50%로, 패스키 검색 작업에서 90%로 줄이면서 정확도를 유지할 수 있음을 보여줍니다. 이 방법은 추가적인 학습 없이도 모든 트랜스포머 기반 디코더 모델에 적용할 수 있어 실용성이 높습니다.



### Fusion Makes Perfection: An Efficient Multi-Grained Matching Approach for Zero-Shot Relation Extraction (https://arxiv.org/abs/2406.11429)
Comments:
          Accepted to the main conference of NAACL2024

- **What's New**: 새로운 연구는 다중 그레인 매칭 접근법을 제안하여 수작업으로 태그를 달 필요 없이 가상 엔티티 매칭(vitual entity matching)을 사용하여 관계를 예측하는 방법을 소개합니다. 또한, 코스 그레인 회상(coarse-grained recall)과 세밀한 분류(fine-grained classification)를 결합하여 상호작용을 풍부하게 하면서도 추론 속도를 보장합니다.

- **Technical Details**: 이 연구는 보이지 않는 관계를 추론하는 zero-shot 관계 추출(RE) 과제를 다룹니다. 기존의 RE 모델은 주어진 문장 내의 엔티티 쌍 간의 관계를 식별하는 데 중점을 두었습니다. 제안된 EMMA(Efficient Multi-Grained Matching Approach)는 코스 그레인 회상과 세밀한 분류 단계를 결합하여 효율성을 높였습니다. 가상 엔티티 매칭 방법을 통해 태그를 달지 않고도 효과적인 의미 매칭을 달성할 수 있습니다. 추론 단계에서는 훈련 데이터를 통해 학습된 모델이 주어진 입력에 대해서 최상위 k개의 관계를 회상한 후, 세밀한 분류 모델을 통해 후보 관계 중에서 최적의 관계를 선택합니다.

- **Performance Highlights**: 다양한 데이터셋과 설정에서 실험한 결과, 제안된 EMMA 모델은 기존의 최신 상태 기술(State Of The Art) 모델들을 능가했으며, 추론 효율성과 예측 정확성 사이의 균형을 성공적으로 달성했습니다. 따라서 EMMA는 수작업을 줄이면서도 더 높은 정확성을 제공할 수 있습니다.



### BAMBINO-LM: (Bilingual-)Human-Inspired Continual Pretraining of BabyLM (https://arxiv.org/abs/2406.11418)
Comments:
          Short paper; Under review

- **What's New**: 본 연구는 이중언어를 사용하는 아동이 부모와 교사의 상호작용을 통해 유산 언어(heritage language)를 재습득할 수 있다는 행동 연구의 통찰을 소규모 언어 모델 학습에 어떻게 통합할 수 있는지 조사합니다. BAMBINO-LM이라는 새로운 지속 적 학습(pretraining) 전략을 도입하여 BabyLM의 성능을 향상시킵니다. 이는 부모 모델 역할을 하는 이탈리아어 모델에서 유도된 혼합된 교대 전략(alternation strategy)과 PPO 기반 복잡성 보상(perplexity reward)을 사용합니다.

- **Technical Details**: BAMBINO-LM은 영어로 학습된 소규모 언어 모델을 이탈리아어 데이터로 지속적 학습하여 성능을 향상시킵니다. 학습 단계에서는 이탈리아어 데이터에 대해 인과 언어 모델링(causal language modeling)을 수행하며, PPO(근접 정책 최적화, Proximal Policy Optimization) 기법을 사용해 보상 신호를 도입합니다. 보상은 부모 모델의 복잡성(perplexity)에 기반합니다.

- **Performance Highlights**: 실험 결과, BAMBINO-LM은 이탈리아어 언어 과제의 성능을 상당히 향상시켰으며, 일부 예측된 성능 저하도 관찰되었습니다. 이는 언어 모델이 인간 아동과 유사한 이중언어 습득 경로를 공유할 수 있음을 시사합니다.



### HARE: HumAn pRiors, a key to small language model Efficiency (https://arxiv.org/abs/2406.11410)
- **What's New**: 이번 논문에서는 인간의 선험적 지식(human priors)을 데이터 구축에 활용하여 효율적이고 성능이 뛰어난 소형 언어 모델(SLM)을 훈련하는 새로운 원칙을 제안합니다. 즉, 의미의 다양성과 데이터 품질 일관성을 갖춘 간결한 데이터셋을 활용하여 SLM을 훈련하고, 벤치마크 데이터 유출을 방지하는 방식입니다. 이를 토대로 훈련한 SLM인 HARE-1.1B는 다양한 대규모 벤치마크 데이터셋에서 뛰어난 성능을 보였습니다.

- **Technical Details**: 제안된 데이터 구축 방법은 다음과 같습니다:
1. 웹에서 수집한 대규모 데이터셋에서 고품질 데이터를 추출하고, 의미의 다양성을 유지하면서 일관된 데이터 품질을 유지합니다.
2. 주제별로 데이터를 클러스터링하고, 다양한 프롬프트와 주제별 데이터를 활용하여 고성능 언어 모델(LLMs)로 합성 데이터를 생성합니다.
3. 자연 언어 형태의 NLP 작업 데이터셋을 대규모로 구축하여, NLP 작업 해결 능력을 향상시킵니다.
4. 벤치마크 데이터 유출을 방지하기 위해 엄격한 데이터 비오염 절차를 구현합니다.

- **Performance Highlights**: HARE-1.1B는 16개의 Nvidia-H800 GPU에서 DeepSpeed와 Flash-Attention을 사용하여 훈련되었습니다. 모델의 매개 변수는 약 1.1B로 줄였으며, 22개의 숨겨진 레이어, 32개의 어텐션 헤드, 8개의 키-값 헤드, 2048의 숨겨진 크기와 32000의 어휘 크기를 가지고 있습니다. 학습 결과, HARE-1.1B는 기존의 최첨단 SLM들과 비교하여 대규모 벤치마크 데이터셋에서 매우 좋은 성능을 보였습니다.



### CodeGemma: Open Code Models Based on Gemma (https://arxiv.org/abs/2406.11409)
- **What's New**: CodeGemma는 Google DeepMind의 Gemma 모델을 기반으로 한 전문적인 오픈 코드 모델 컬렉션입니다. 이 모델 컬렉션은 코드 및 자연어 생성 작업을 위해 설계되었으며, 7B 및 2B 모델 변형을 제공합니다. CodeGemma 7B는 사전 학습(PT) 및 지시 조정(IT) 변형으로, CodeGemma 2B는 코드 완성 및 열린 끝 생성에 특화된 모델입니다.

- **Technical Details**: CodeGemma의 모든 모델은 500억에서 1조 개의 주요 코드 토큰을 추가로 학습했습니다. 7B 모델은 80% 코드 및 20% 자연어 데이터 혼합물로 학습된 반면, 2B 모델은 100% 코드로 학습되었습니다. 이 모델은 fill-in-the-middle (FIM) 작업 기반의 기법으로 훈련되었으며, PSM (Prefix-Suffix-Middle) 및 SPM (Suffix-Prefix-Middle) 모드 모두에서 작동하도록 설계되었습니다.

- **Instruction Tuning**: CodeGemma의 7B 모델은 강화 학습 알고리즘과 합성 데이터 생성의 차이가 있는 반면, 지시 조정 단계에서는 다양한 수학적 문제에 대한 감독 학습을 통해 모델의 논리적 추론 및 문제 해결 능력을 향상시키고자 했습니다. 이를 위해 여러 오픈 소스 수학 데이터셋과 합성 코드를 사용하는 방식을 도입했습니다.

- **Performance Highlights**: CodeGemma 모델은 다양한 코드 완성 및 생성 작업에서 뛰어난 성능을 보입니다. 특히, 2B 사전 학습 모델은 낮은 대기 시간이 중요한 경우에 매우 뛰어난 성과를 보이며, 다른 모델과 비교하여 두 배 빠른 추론 속도를 가지고 있습니다. HumanEval, Mostly Basic Python Problems 등의 벤치마크 테스트에서도 뛰어난 성능을 보였습니다.



### Evaluating Open Language Models Across Task Types, Application Domains, and Reasoning Types: An In-Depth Experimental Analysis (https://arxiv.org/abs/2406.11402)
- **What's New**: 이번 연구는 10개의 소규모, 공개 언어 모델(LM)의 성능을 다양한 프롬프트 스타일을 사용하여 세 가지 측면에서 심층 분석하였습니다. 이를 통해 각각의 요구 사항에 따라 가장 효과적인 모델과 프롬프트 스타일이 다르다는 것을 확인하였습니다.

- **Technical Details**: 연구에서는 2B-11B 파라미터 크기의 공개 LM을 대상으로 실험을 수행하였으며, Super-Natural Instructions 데이터셋을 사용하여 12개의 작업 유형(task types), 12개의 응용 분야(application domains), 10개의 추론 유형(reasoning types)을 기준으로 삼아 분석을 진행하였습니다. 최대 100개의 인스턴스(instance)를 각 작업에 대해 실험하였으며, 총 11810개의 작업 인스턴스를 평가하였습니다.

- **Performance Highlights**: 연구 결과, 적절히 활용한다면 이러한 소규모 LM도 GPT-3.5-Turbo, GPT-4o 및 DeepSeek-v2와 같은 SOTA LLM과 경쟁할 수 있으며 때로는 더 우수한 성능을 보일 수 있음을 보여주었습니다. 다양한 프롬프트 스타일을 통해 모델의 성능이 다르게 나타남을 발견하였으며, 몇몇 소규모 LM이 대형 SOTA 모델들과 유사하거나 이를 뛰어넘는 성능을 보였습니다.



### Large Language Models and Knowledge Graphs for Astronomical Entity Disambiguation (https://arxiv.org/abs/2406.11400)
- **What's New**: 이 논문은 해커톤 기간 동안 대형 언어 모델(LLMs)과 지식 그래프 클러스터링(knowledge graph clustering)을 사용하여 천문학 텍스트에서 엔티티(entity)와 관계(relationship)를 추출하는 실험을 다룹니다. 천문학 분야에서 다양한 문맥에서 나타날 수 있는 엔티티를 식별하고 분류하는 접근 방법을 보여줍니다. 특히 GPT-4 언어 모델을 활용해 특정 엔티티 주변의 텍스트를 수집하고, 이를 통해 관련 엔티티와 관계를 추출합니다.

- **Technical Details**: 추출된 정보는 지식 그래프(knowledge graph)를 구성하는 데 사용되며, 이를 Leiden 알고리즘으로 클러스터링합니다. 생성된 Leiden 커뮤니티를 사용하여 이전에 알려지지 않은 텍스트 조각을 각 커뮤니티와의 연관성 비율로 식별합니다. 이렇게 함으로써 엔티티의 분류(disambiguation) 문제를 해결합니다.

- **Performance Highlights**: 해당 실험은 LLMs와 지식 그래프 클러스터링 기법을 결합하여 천문학 연구에서 정보 추출의 가능성을 보여줍니다. 이 접근법은 엔티티를 식별하고 분류하는 데 효과적이며, 이들 간의 관계를 기반으로 의미 있는 클러스터로 그룹화하는 데 유용함을 강조합니다.



### MetaGPT: Merging Large Language Models Using Model Exclusive Task Arithmetic (https://arxiv.org/abs/2406.11385)
Comments:
          17 pages

- **What's New**: 최근 대형 언어 모델(LLMs)인 GPT-4와 같은 모델들이 다중 작업 학습(Multi-Task Learning, MTL)을 탐구하는 촉매 역할을 하고 있습니다. 이는 단일 모델이 다양한 작업에서 능숙함을 보이는 것을 의미합니다. 그러나 현재 최적 성능, 계산 효율성, 데이터 프라이버시를 동시에 달성할 수 있는 방법의 부재로 인해, 이러한 접근방식의 활용이 제한됩니다. 이 문제를 해결하기 위해, 우리는 GPT 규모의 모델을 병합하기 위한 Model Exclusive Task Arithmetic, 즉 MetaGPT를 제안합니다.

- **Technical Details**: MetaGPT는 모델 병합을 다중 작업 학습 프레임워크로 형식화하고, 병합된 모델과 각 개별 작업 모델 간의 평균 손실 차이를 최소화하는 것을 목표로 합니다. 우리는 데이터 프라이버시 때문에 다중 작업 훈련 데이터를 사용할 수 없는 것을 감안하여, LLM의 로컬 선형성과 작업 벡터(Task vectors)의 직교성을 활용하여 데이터 항과 스케일링 계수 항을 분리했습니다. 이를 통해 계산 효율적이고 데이터 비의존적인 모델 독점 작업 산술 방법을 도출했습니다.

- **Performance Highlights**: 광범위한 실험 결과, MetaGPT가 여러 작업에 대해 최첨단 성능을 달성하며, 기존의 병합 방법들보다 우수한 성능을 보였습니다. MetaGPT는 비용 효율적이며 대규모 다중 작업 학습을 위한 최적의 작업 산술을 효과적으로 구현할 수 있는 방법을 제공합니다. 특히, LLaMA-2와 Mistral 시리즈 실험에서 MetaGPT의 우수성이 입증되었습니다.



### A Realistic Evaluation of LLMs for Quotation Attribution in Literary Texts: A Case Study of LLaMa3 (https://arxiv.org/abs/2406.11380)
Comments:
          Paper under review

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 문학적 인용구 할당 성능을 평가합니다. 특히 Llama3 모델을 예시로 들어, 모델이 얼마나 많은 책을 암기했는지 측정하여 인용구 할당 능력이 책 암기 정도와 어떻게 연관되는지 분석합니다. 데이터 및 코드는 공개될 예정입니다.

- **Technical Details**: 이번 연구는 Llama3-8b-Instruct 모델을 사용하여 소설 내 인용구 할당 작업을 수행했습니다. 이를 위해 책 암기(book memorization)와 주석 오염(annotation contamination)을 측정하는 방법론을 설계했습니다. Project Dialogism Novel Corpus(PDNC)라는 데이터셋을 사용하여 모델의 성능을 테스트했으며, 이 데이터셋에는 28권의 소설이 포함되어 있습니다. 데이터 오염을 방지하기 위해 최신 소설(2024년 출판)의 인용구를 수작업으로 주석 달아 테스트했습니다.

- **Performance Highlights**: Llama3 모델이 PDNC 데이터셋에서 높은 성능을 보였으며, 특히 암기된 소설에서 더 높은 성능을 발휘했습니다. 하지만 주석 오염의 징후는 보이지 않았습니다. 또한, Llama3는 이전에 본 적 없는 소설에서도 상대적으로 좋은 성능을 발휘하며, 모델 크기를 증가시키면 성능이 향상되는 모습을 보였습니다.



### Boosting Scientific Concepts Understanding: Can Analogy from Teacher Models Empower Student Models? (https://arxiv.org/abs/2406.11375)
- **What's New**: 본 논문에서는 인간 학습 과정을 참조하여 교사용 언어 모델(teacher LMs)이 생성한 유추가 학생용 언어 모델(student LMs)이 과학적 개념을 이해하는 데 어떻게 도움을 줄 수 있는지 조사합니다. 자유 형식의 유추가 실제로 LMs의 개념 이해에 도움을 줄 수 있음을 실험을 통해 확인하였습니다.

- **Technical Details**: 제안된 SCUA(Scientific Concept Understanding with Analogy) 과제에서는 교사용 LMs, 예를 들어 GPT-4(OpenAI, 2023)와 Claude(Anthropic, 2024)에게 특정 과학적 개념을 설명하는 유추를 생성하도록 요청합니다. 그런 다음, 학생용 LMs, 예를 들어 GPT-3.5(OpenAI, 2022)와 Vicuna(Chiang et al., 2023)에게 유추를 사용하여 과학적 질문에 답하도록 합니다. 이는 다양한 유추 유형의 강력한 및 약한 LMs를 평가합니다.

- **Performance Highlights**: 유추를 사용한 결과, LMs의 과학적 개념 이해가 향상되었으며, 특히 자유 형식의 유추가 가장 큰 향상을 보였습니다. 또한, 학생용 LMs가 생성한 유추가 그들의 성능을 개선하여, 스스로 새로운 지식을 학습하는 능력을 보였습니다.



### Fairer Preferences Elicit Improved Human-Aligned Large Language Model Judgments (https://arxiv.org/abs/2406.11370)
Comments:
          5 pages, 3 figures, 1 table (12 pages, 4 figures, 6 tables including references and appendices)

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 언어 생성 품질을 평가하는 데 있어 비용 효율적이고 참조 없이 평가하는 능력을 보여주었으나, LLM의 평가 편향과 프롬프트 설계에 대한 민감성이 문제로 지적된다는 것을 알리고 있습니다. 연구진은 LLM이 주어진 지시에 따라 예측 편향과 인간 의견 일치도가 크게 달라질 수 있음을 발견했습니다. 이를 개선하기 위해 제로샷 평가 지향 프롬프트 최적화(ZEPO) 프레임워크를 제안했으며, 인간 판단과의 일치도를 향상시키는 데 성공했습니다.

- **Technical Details**: ZEPO는 LLM 평가자들이 인간과의 일치도를 높이기 위해 자동화된 제로샷 학습 목표를 제안합니다. 연구에서는 LLM 평가자들이 제공된 지시에 얼마나 민감한지를 체계적으로 검토하며, 공정한 예측이 더 나은 인간 정렬 판단으로 이어진다는 것을 실험적으로 밝혔습니다. ZEPO는 수작업으로 설계된 지시 없이 성능을 크게 향상시켰습니다.

- **Performance Highlights**: ZEPO는 대표적인 메타 평가 벤치마크에서 최신 LLM 평가자보다 상당한 성능 향상을 보여주었습니다. 특히 공정한 예측 편향을 통해 인간 판단과의 일치도를 개선하는 데 성공했습니다.



### Improving Quotation Attribution with Fictional Character Embeddings (https://arxiv.org/abs/2406.11368)
Comments:
          Paper under review

- **What's New**: 이번 연구에서는 인물 표현(character representations)을 포함하여 기존 BookNLP 시스템을 개선하는 방법을 제안했습니다. 이를 위해 15세기부터 20세기까지의 영문 드라마 작품집인 DramaCV 코퍼스를 만들고, 이 데이터를 활용하여 Universal Authorship Representation(UAR) 모델을 학습했습니다. 이 모델은 기존 캐릭터 임베딩(character embeddings) 방법보다 뛰어난 성능을 보였습니다. 또한, BookNLP와 우리의 전역 인물 임베딩(global character embeddings)을 결합하여 인물 식별 정확도를 향상시켰습니다.

- **Technical Details**: 기존 방법은 결정적 규칙(deterministic rules)이나 뉴럴 네트워크를 사용하여 문맥 정보를 처리하였습니다. 그러나 이들은 인물 표현의 부족으로 인해 직관적(anaphoric) 및 암시적(implicit) 인용문을 처리하는 데 어려움을 겪었습니다. 이번 연구에서는 UAR 모델을 DramaCV 데이터셋으로 학습시켜, 허구 인물들의 전역 정보를 포함한 임베딩을 생성하였습니다. 이를 BookNLP 시스템에 통합하여 인물 식별 성능을 향상시켰습니다.

- **Performance Highlights**: 22개의 소설에 대한 광범위한 평가를 통해, 우리 모델이 anaphoric 및 implicit 인용문의 화자 식별에서 최첨단 성능(state-of-the-art performance)을 달성했음을 확인했습니다. 또한, DramaCV 데이터를 사용한 UAR 모델은 다양한 문체와 주제적 선호도를 더 잘 반영하여 인물 식별 작업에서 뛰어난 성능을 보였습니다.



### $\textit{Refiner}$: Restructure Retrieval Content Efficiently to Advance Question-Answering Capabilities (https://arxiv.org/abs/2406.11357)
Comments:
          8 pages

- **What's New**: Refiner는 Retrieval-Augmented Generation(RAG) 시스템의 포스트-리트리벌(process)에서 작동하는 새로운 엔드-투-엔드 추출 및 재구성 패러다임입니다. Refiner는 query와 관련된 컨텐츠를 적응적으로 추출하고 이를 섹션으로 나누어 주요 정보를 강조하며, 다운스트림 LLM과 원본 문맥을 효과적으로 정렬합니다. 다양한 QA(Questions and Answers) 작업에서 상태-of-the-art RAG 솔루션을 능가합니다.

- **Technical Details**: Refiner는 단일 디코더 전용 LLM을 사용하여 문서 청크에서 쿼리와 관련된 내용을 추출하고, 이를 구조화하여 중요한 정보가 잘 드러나도록 합니다. 이를 통해 'lost-in-the-middle' 현상을 방지하고, 다운스트림 LLM이 주요 정보를 쉽게 인식할 수 있도록 도와줍니다. Refiner는 Llama2-7B Chat tokenizer를 사용했습니다.

- **Performance Highlights**: Refiner는 다중-점프 QA 작업에서 80.5%의 토큰 감소와 1.6-7.0%의 성능 개선을 달성했습니다. 특히 HotpotQA와 2WikiMultihop 작업에서 각각 7.0%, 6.4%의 성능 개선을 보여줍니다. 추가적인 실험 결과, Refiner는 불필요한 문서 청크가 추가되더라도 성능이 유지되며, 다운스트림 LLM의 정확성을 향상시킵니다. Refiner는 플러그-앤-플레이 해결책으로, 다양한 오픈소스 프레임워크와 통합 가능성이 높습니다.



### Preserving Knowledge in Large Language Model: A Model-Agnostic Self-Decompression Approach (https://arxiv.org/abs/2406.11354)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 '불연속 기억 상실(catastrophic forgetting)' 문제와 다중 모드 대규모 언어 모델(MLLMs)의 성능 저하 문제를 해결하기 위해 새로운 모델-독립적(self-decompression) 방법인 '나무 생성(Tree Generation, TG)'을 소개합니다. TG를 통해 학습 데이터 내에서 지식을 확장하고 이를 활용하여 LLMs의 기억 상실 문제를 줄일 수 있습니다.

- **Technical Details**: 이 논문은 TG-SFT를 중심으로 수퍼바이즈드 파인튜닝(SFT) 단계를 위한 데이터 생성을 자동으로 수행하는 방법을 제안합니다. TG-SFT는 모델에 의해 생성되는 훈련 데이터를 사용하여 구체적으로 설계되었으며, 이 데이터를 MLLMs의 SFT 과정에 통합하여 기억 상실 문제를 현저히 줄였습니다.

- **Performance Highlights**: 다양한 실험 결과, TG 알고리즘은 기억 상실 문제를 효과적으로 줄이는 데 유용함이 입증되었습니다. 또한, TG 알고리즘은 특정한 NLP 과업에 한정되지 않고 어떤 LLMs에도 적용할 수 있으며, 추가적인 수동 입력 없이도 사용 가능합니다. 이러한 TG 알고리즘은 기억 상실 방지, 지식 증류, 연속 학습 등 많은 응용 분야에서 활용될 수 있습니다.



### Full-ECE: A Metric For Token-level Calibration on Large Language Models (https://arxiv.org/abs/2406.11345)
- **What's New**: 최근 여러 도메인에서 Deep Neural Networks (DNNs)의 성공에도 불구하고, 이 모델들이 불확실성(estimates)을 정확히 제공하는 데 어려움을 겪고 있습니다. 특히, Large Language Models (LLMs)는 탁월한 성능을 보이지만 전통적인 보정 메트릭(calibration metrics)은 해당 모델들의 방대한 어휘집(vocabularies), 데이터 복잡성, 분포적 집중도를 다루는 데 한계가 있습니다. 이를 해결하기 위해 우리는 '풀 보정(full calibration)'이라는 새로운 개념과 이에 상응하는 메트릭으로 Full-ECE를 제안합니다.

- **Technical Details**: 기존의 보정 메트릭은 주로 가장 가능성 높은 예측값, 즉 top-1 prediction에 집중합니다. LLMs의 경우, 수십억 개의 토큰으로 구성된 데이터셋에서 토큰 레벨 보정(token-level calibration)이 중요한데, 이는 각 토큰의 예측 확률 분포가 실제 분포와 일치하도록 합니다. Full-ECE는 전체 예측 확률 분포를 평가함으로써, 전통적인 ECE와 classwise-ECE (cw-ECE)보다 더 정확하고 강력한 보정 측정을 제공합니다.

- **Performance Highlights**: Full-ECE 메트릭은 LLMs의 광범위한 클래스 수와 불균형성을 감안하여 더 정확하고 신뢰성 높은 보정 측정을 가능하게 합니다.



### A Systematic Analysis of Large Language Models as Soft Reasoners: The Case of Syllogistic Inferences (https://arxiv.org/abs/2406.11341)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 추론 능력, 특히 삼단논법 추론(syllogistic reasoning)에 대한 체계적인 분석을 제공한다. 체인 오브 사고(Chain-of-Thought, CoT) 추론, 인컨텍스트 학습(In-Context Learning, ICL), 감독 하 미세 조정(Supervised Fine-Tuning, SFT)이 삼단논법 추론에 미치는 영향을 연구했다. 기존 연구를 넘어서는 정확도 분석 외에도, 모델이 생성하는 결론을 심층 분석했다.

- **Technical Details**: 본 연구에서는 인간이 추론하고자 하는 방식과 유사한 방식으로 LLMs가 추론하는 능력을 측정했다. 특히 Pyhtia와 LLaMA와 같은 소형 및 중형 오픈 액세스 모델을 Zero-Shot CoT (ZS-CoT) 설정에서 평가했다. 또한, 믿을 수 있는 삼단논법과 믿기 어려운 삼단논법을 포함한 데이터셋을 사용하여 여러 전제를 포함하는 삼단논법 추론을 수행했다. 체계적인 데이터셋을 통해 ICL 및 SFT의 효과를 조사하고, 모델이 생성하는 모든 결론을 분석했다.

- **Performance Highlights**: 모델은 인간과 유사한 추론 동작을 보여주며, LLaMA-3 8B는 ZS-CoT 설정에서 Atmosphere Heuristic Theory의 예측과 일치하는 답변을 생성하는 경향이 있다. ICL은 유효한 추론에서 모델 성능을 향상시키지만, 내용 효과 편향(content effect bias)이나 무효한 삼단논법을 처리하는 어려움을 해결하지 못한다. 반면, SFT는 소형 및 중형 모델 모두에서 효과적이며, 후자는 일관성을 유지하면서 거의 완벽한 성능을 달성한다.



### Fine-grained Controllable Text Generation through In-context Learning with Feedback (https://arxiv.org/abs/2406.11338)
- **What's New**: 이번 연구에서는 문장의 의존 깊이(Dependency Depth)와 같은 비정형 언어적 특징에 맞게 입력 문장을 재작성하는 방법을 제안합니다. 이전 연구들과 달리, 이번 방법은 파인튜닝(finetuning) 대신 문맥 학습(in-context learning)을 사용하여 데이터가 부족한 상황에서도 적용 가능합니다. 성능 면에서, 이 모델은 특정 학년 수준에 맞춰 문장을 재작성할 때 최신 기술과 유사한 성능을 보입니다.

- **Technical Details**: 이 방법은 문장을 학년 수준으로 맞추기 위해 두 단계를 거칩니다. 첫 번째 단계에서는 출력 텍스트의 언어적 특징을 예측하고, 두 번째 단계에서는 문맥 학습을 통해 예측된 특징 값에 맞춰 문장을 재작성합니다. 네 가지 주요 언어적 특징은 최대 의존 깊이(DD), 최대 의존 길이(DL), 단어 수(WC), 어려운 단어(DW)입니다. 이를 통해 텍스트 복잡도를 조절합니다. 또한, 피드백 메커니즘을 도입하여 정확하지 않은 경우 모델이 다시 시도하도록 합니다.

- **Performance Highlights**: 이 모델은 81%의 테스트 문장을 요청된 의존 깊이에 맞춰 정확하게 재작성할 수 있었으며, 문맥 학습 예제 다섯 개만으로도 이전 연구와 비교해 좋은 성능을 보였습니다. 이렇게 학습한 모델은 의존 깊이와 같은 구체적인 언어적 특징뿐만 아니라, 원하는 학년 수준에 맞춰 문장을 조정하는 데 있어 뛰어난 성과를 보였습니다.



### Are Large Language Models True Healthcare Jacks-of-All-Trades? Benchmarking Across Health Professions Beyond Physician Exams (https://arxiv.org/abs/2406.11328)
Comments:
          15 pages, 4 figures

- **What's New**: 최근 대형 언어 모델(LLMs)이 세계 지식에 대한 정확한 답을 제공하는 잠재력을 보이고 있지만, 기존의 의료 분야 평가 기준은 주로 의료 전문가에만 집중되었다. 이에 대한 연구 격차를 메우기 위해 전통적인 중국어로 된 대규모 의료 지식 벤치마크 'EMPEC'를 도입했다. EMPEC는 20개의 의료 직업과 124개의 과목에 걸쳐 157,803개의 시험 문제를 포함하고 있으며, 시각 검사사(Optometrist) 및 청력 검사사(Audiologist)와 같은 기존 평가에서 간과된 직업들도 포함된다.

- **Technical Details**: EMPEC는 각 질문이 출처와 출시 시간이 태그되어 있어 관련성과 신뢰성을 보장한다. 17개의 LLM를 대상으로 포괄적인 실험을 진행했으며, 여기에는 독점 모델, 오픈소스 모델, 일반 도메인 모델 및 의료 특화 모델이 포함되었다. 실험에서 일반 목적의 LLM가 의료 특화 모델보다 더 나은 성과를 보였고, EMPEC의 훈련 데이터를 포함함으로써 성능이 크게 향상되었다.

- **Performance Highlights**: GPT-4는 평가에서 75% 이상의 정확도를 달성하였으나, 전문 분야 및 대체 의학에서는 여전히 어려움을 겪고 있다. 전체 테스트 세트와 가장 최근의 질문 세트를 사용하여 모델의 성능을 비교한 결과, 모델의 테스트 세트 성능이 미지의 의료 관련 질문에 대한 효과를 예측할 수 있음을 확인했다. 전통적인 중국어에서 간체 중국어로의 전환은 모델 성능에 거의 영향을 미치지 않는 것으로 나타나, 다양한 중국어 환경에서도 견고한 언어적 유연성을 보였다.



### An Empirical Investigation of Matrix Factorization Methods for Pre-trained Transformers (https://arxiv.org/abs/2406.11307)
- **What's New**: 본 논문은 NLP에서 급증하는 트랜스포머 기반 모델의 크기를 압축하는 문제를 해결하기 위해 제안된 다양한 방법들을 분석합니다. 특히, 저순위 행렬 분해(low-rank factorization)와 최근 도입된 Monarch 분해를 비교하며, 저순위 행렬 분해의 안정성을 높이기 위해 스테이지드 분해(Staged Factorization) 접근법을 소개합니다.

- **Technical Details**: 저자들은 트랜스포머 모델의 사전 훈련된 가중치 행렬을 저순위 행렬 분해 및 Monarch 분해로 표현하는 방법을 연구했습니다. 저순위 행렬 분해는 큰 행렬을 두 개의 작은 행렬의 곱으로 표현하며, Monarch 분해는 블록 대각 행렬 곱으로 표현합니다. 스테이지드 분해는 모든 레이어를 동시에 분해하는 대신, 각각의 레이어를 순차적으로 분해하여 안정성을 높이고 모델 훈련의 실패를 줄입니다. 또한, 블록 단위 저순위 행렬 분해(Block-wise Low-Rank Factorization) 방법도 제안되었습니다.

- **Performance Highlights**: 여섯 개의 GLUE 데이터셋과 네 개의 사전 훈련된 모델을 사용한 실험 결과, 저순위 행렬 분해가 Monarch 분해를 consistently 뛰어넘는 성능을 보였습니다. 저순위 행렬 분해는 높은 정확도와 낮은 추론 시간(inference time)을 달성했으며, 구현하기도 쉬운 방법입니다. 특히, 스테이지드 분해 전략을 통해 안정성을 크게 향상시켰습니다.



### A Systematic Survey of Text Summarization: From Statistical Methods to Large Language Models (https://arxiv.org/abs/2406.11289)
Comments:
          30 pages, 8 figures, 6 tables

- **What's New**: 이 논문은 딥 뉴럴 네트워크(deep neural network), 사전 훈련된 언어 모델(pre-trained language model, PLM) 및 최근의 대형 언어 모델(large language model, LLM)의 등장으로 인한 텍스트 요약 연구의 발전과 변화를 종합적으로 검토합니다. 이 리뷰는 주요하게 두 부분으로 나뉘어져 있습니다: (1) 통계적 방법, 딥 러닝 접근법, PLM 파인튜닝 기술 등 LLM 시대 이전의 데이터셋, 평가 척도 및 요약 방법의 종합적인 개요와 (2) LLM 시대의 최신 발전에 대한 세부 검토입니다. 또한, 요약 연구의 동향, 해결되지 않은 도전 과제 및 유망한 연구 방향도 논의합니다.

- **Technical Details**: 텍스트 요약은 소스 문서에서 가장 중요한 정보를 추출하여 간결한 버전을 생성하는 작업입니다. 초기 자동 텍스트 요약(automatic text summarization, ATS) 시스템은 1950년대부터 시작되었으며, 이후 1990년대와 2000년대 초에 통계적 기계 학습의 발전으로 비지도(feature-based) 시스템이 등장했습니다. 2010년대에는 대규모 훈련 데이터를 활용한 딥 러닝 프레임워크의 교육이 주요 연구 주제가 되었으며, BERT 및 T5와 같은 사전 훈련된 언어 모델의 등장으로 요약 성능이 크게 향상되었습니다. 현재 LLM의 등장으로 요약 분야는 새로운 시대를 맞이하고 있으며, LLM은 학문적 NLP 연구 및 산업 제품을 혁신하고 있습니다.

- **Performance Highlights**: LLM은 방대한 텍스트 데이터 코퍼스를 활용하여 복잡한 언어 패턴, 의미 관계, 문맥적 단서를 포착할 수 있으며, 인간이 제작한 요약과 견줄 만한 고품질 요약을 생성합니다. 그러나 많은 NLP 연구자들은 LLM 시스템의 성공으로 인해 텍스트 요약이 거의 사라졌다고 주장합니다. 연구자들은 기존의 요약 접근법과 LLM 기반 요약 접근법을 체계적으로 분석하였으며, LLM 시대에 요약 연구의 발전과 도전 과제를 논의합니다. 요약 방법, 데이터셋 및 평가 척도를 종합적으로 검토하고, LLM 기반 요약 연구의 동향과 미래 방향을 제시합니다.



### MFC-Bench: Benchmarking Multimodal Fact-Checking with Large Vision-Language Models (https://arxiv.org/abs/2406.11288)
Comments:
          22 pages, 8 figures

- **What's New**: MFC-Bench라는 새로운 벤치마크가 도입되었습니다. 이 벤치마크는 대형 비전-언어 모델(Large Vision-Language Models, LVLM)의 사실성(Factual Accuracy)을 평가하기 위해 고안된 것으로, Manipulation, Out-of-Context, Veracity Classification 세 가지 작업을 포함합니다. 이를 통해 LVLM이 다중 모달 콘텐츠에서의 사실성을 얼마나 잘 인지하고 검증할 수 있는지에 대한 종합적인 평가가 가능해졌습니다.

- **Technical Details**: MFC-Bench는 33,000개의 다중 모달 샘플을 포함하며, 세 가지 주요 작업으로 구성됩니다. 첫째, Manipulation Classification 작업은 얼굴 스왑(Face Swap), 얼굴 속성 편집(Face Attribute Edit), 배경 변경(Background Changing), 이미지 생성(Image Generation), 엔티티 교체(Entity Replacement), 스타일 전환(Style Transfer) 등의 다양한 조작 기법을 통해 콘텐츠의 변질 여부를 평가합니다. 둘째, Out-of-Context Classification 작업은 이미지와 텍스트 간의 잘못된 연결을 찾아냅니다. 셋째, Veracity Classification 작업은 텍스트 주장과 시각적 증거의 사실성을 분류합니다. 각 작업은 LVLM의 파라메트릭 지식을 활용해 신속하고 정확하게 수행될 수 있도록 설계되었습니다.

- **Performance Highlights**: 12개의 LVLM을 평가한 결과, 현존하는 모델들이 다중 모달 사실 검증에서 상당한 도전에 직면하고 있다는 것을 발견했습니다. 특히, GPT-4V는 Manipulation, OOC, Veracity Classification 각각에 대해 F1 점수 45.6%, 75.2%, 60.0%를 기록했습니다. 이는 이러한 모델들이 다양한 형태의 변질된 콘텐츠에 민감하지 않다는 것을 나타냅니다. MFC-Bench는 LVLM이 온라인 허위 정보 확산을 억제하고, 다양한 커뮤니티의 안정성과 결속을 촉진하는 데 중요한 역할을 할 수 있도록 하는 중요한 평가 도구입니다.



### Do Not Design, Learn: A Trainable Scoring Function for Uncertainty Estimation in Generative LLMs (https://arxiv.org/abs/2406.11278)
- **What's New**: 여기서 우리는 학습 가능한 응답 점수 함수(Learnable Response Scoring Function: LARS)를 큐레이터 추정(UE)에 사용하기 위해 제안합니다. 현재의 확률 기반 UE 점수 함수는 길이 정상화 점수와 의미적 기여 기반 가중치와 같이 특정 문제를 해결하도록 설계되었지만, 편향된 확률을 처리하지 못하거나 터키어 같은 자원 부족 언어에서 성능이 떨어지는 등의 한계가 있습니다. 이러한 문제를 해결하기 위해, 우리는 복잡한 토큰 및 확률 간의 종속성을 포착하는 슈퍼바이즈드 데이터를 활용하여 더 신뢰할 수 있고 잘 조정된 응답 점수를 생성하는 LARS를 제안합니다.

- **Technical Details**: LARS는 감독 데이터를 활용하여 학습된 점수 함수로, 현재의 점수 함수가 간과할 수 있는 복잡한 의존성을 잡아줍니다. 기존의 길이 정상화 점수(length-normalized scoring)와 의미적 기여 기반 가중치(semantic contribution-based weighting) 점수는 각각의 한계를 지니고 있습니다. 길이 정규화 점수는 여러 토큰 확률의 평균 로그를 계산해 긴 생성물에 대한 편향을 완화하려고 하며, 의미적 기여 기반 가중치 방법은 의미적으로 중요한 토큰에 더 높은 가중치를 부여합니다. LARS는 이들 기존의 접근법의 문제점을 해결하며, 토큰 확률을 사용하는 확률 기반 전자 입체화(UE) 방법의 개선을 목표로 합니다.

- **Performance Highlights**: LARS는 여러 데이터셋에 걸쳐 기존의 점수 함수보다 뛰어난 성능을 보여주었습니다. 실험 결과, LARS는 터키어와 같은 자원이 부족한 언어에서도 더 나은 성능과 캘리브레이션을 제공하여 기존 방법들을 능가하였습니다. 이를 통해 LARS의 효과를 입증하며, 세 가지 다른 데이터셋에서 기존 기준선을 넘는 성능을 검증했습니다.



### Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector (https://arxiv.org/abs/2406.11277)
- **What's New**: 본 논문에서는 Hallucination(환각) 검출을 위한 자율적인 LLM 기반 에이전트 프레임워크, HaluAgent를 제안합니다. 이는 상대적으로 작은 LLM들 (예: Baichuan2-Chat 7B)이 텍스트, 코드, 수학 표현과 같은 여러 유형의 환각을 능동적으로 검출할 수 있도록 합니다. HaluAgent는 다기능 툴박스를 통합하고, 메모리 메커니즘과 함께 세분화된 3단계 검출 프레임워크를 설계했습니다.

- **Technical Details**: HaluAgent는 작은 오픈소스 모델 (Baichuan2-Chat 7B 및 13B) 기반으로, 중국어와 영어의 이중 언어 환각 검출이 가능합니다. 주요 기술적 기여는 다음과 같습니다: (1) 다양한 종류의 환각 형태를 검출할 수 있도록 다기능 툴박스를 설계, (2) 세분화된 3단계 검출 프레임워크 (문장 분할, 도구 선택 및 검증, 반영) 설계, (3) 고품질 검출 궤적 데이터를 합성하여 오픈소스 LLM을 미세 조정.

- **Performance Highlights**: HaluAgent는 2K 샘플로 LLM을 튜닝하여 여러 유형의 작업과 데이터셋에서 환각 검출을 수행하여 GPT-4와 견줄 수 있는 성능을 보였습니다. 예를 들어, 4개의 도메인 내(in-domain) 데이터셋에서 전체 정확도는 46.44%에서 79.70%로 향상되었고, 2개의 도메인 외(out-of-domain) 데이터셋에서는 49.50%에서 78.43%로 향상되었습니다. 문장 레벨 검출 실험에서도 수학 및 과학 데이터셋에서 F1 점수가 각각 19.51%에서 68.80%, 17.54%에서 94.16%로 상당히 향상되었습니다.



### Self-training Large Language Models through Knowledge Detection (https://arxiv.org/abs/2406.11275)
Comments:
          Under review

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 자율적으로 자체 라벨을 생성하고, 이를 통해 선택적으로 학습할 수 있는 셀프 트레이닝 패러다임을 탐구합니다. 이를 통해 여러 주제에 걸쳐 생성 시 발생하는 'hallucination'을 크게 줄이는 성과를 보였습니다.

- **Technical Details**: 연구에서는 LLM이 '알 수 없음'으로 라벨링된 데이터를 식별하는 참조 없는 일관성 방법을 사용하여 선택적으로 학습합니다. 이 필터링 단계는 Direct Preference Optimization(DPO)를 수행하기 위한 선호 데이터셋을 만듭니다. 이는 모델이 새로 학습한 정보로 인해 이전에 학습한 지식을 잊어버리는 현상(Catastrophic Forgetting)을 완화하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 제한된 라벨된 데이터셋 의존도를 크게 줄이면서 사실적 정확성을 높였습니다. 또한, 선택된 샘플로 학습함으로써 분포 외 벤치마크에서도 성능이 보존 또는 향상되는 효과를 보였습니다.



### Skip-Layer Attention: Bridging Abstract and Detailed Dependencies in Transformers (https://arxiv.org/abs/2406.11274)
Comments:
          7 pages, 1 figure

- **What's New**: 이번 연구에서는 변환기(Transformer) 모델의 성능을 향상시키기 위해 'Skip-Layer Attention(SLA)' 기법을 새롭게 도입했습니다. 이 기법은 비인접 층 사이의 직접적인 주의를 가능하게 하여, 고수준 추상화 기능과 저수준 세부사항 간의 종속성을 더 잘 포착할 수 있게 합니다. 이는 기존의 변환기 모델이 종종 부족한 동일 층 내 주의(intra-layer attention)에 의존하는 문제를 해결합니다.

- **Technical Details**: SLA는 주어진 층의 쿼리(query)가 현재 층뿐만 아니라 한 층 이전의 키(key)와 값(value)와도 상호작용할 수 있도록 변환기 모델을 확장합니다. 이를 통해 다중 헤드 주의(multi-head attention)의 다양성을 증가시키면서도 계산 복잡성을 추가하지 않습니다. 우리의 구현은 오리지널 변환기의 다중 헤드 주의 메커니즘을 그대로 유지하면서도 비인접 층 간의 직접 연결을 확보합니다. 이 기법의 주요 요소는 PyTorch로 구현됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 SLA를 적용한 변환기 모델이 언어 모델링 작업에서 우수한 성능을 보였음을 확인했습니다. 예를 들어, 기본 GPT-2 모델(124M 파라미터) 구성에서 9개 층의 스킵 주의(skip-layer attention) 층을 적용했을 때, 기준 모델 대비 0.1076의 절대 성능 향상을 달성했습니다.



### Mitigating Large Language Model Hallucination with Faithful Finetuning (https://arxiv.org/abs/2406.11267)
- **What's New**: 대형 언어 모델(LLMs)이 여러 자연어 처리 작업에서 뛰어난 성능을 보이고 있지만, 사실에 맞지 않는 응답('환상(hallucinations)')을 생성하는 문제가 발생하고 있다. 이러한 환상은 정보의 왜곡과 잘못된 정보의 전파로 이어질 수 있다. 본 연구에서는 'Faithful Finetuning (F2)'라는 새로운 방법을 도입하여 정밀하게 설계된 손실 함수(loss functions)를 통해 신뢰성 있는 질문 응답을 모델링하는 방법을 제안한다.

- **Technical Details**: F2는 QA 목적을 내부 사실 검색과 사실 기반 QA의 두 가지 하위 목표로 분해하여 LLM이 내부 지식을 효과적으로 활용하여 신뢰성 있는 답변을 생성하도록 한다. 이를 위해 엔티티 기반 및 주의 기반 휴리스틱스 기법을 사용하여 모델의 핫스팟(환상 발생 가능성이 높은 부분)을 식별하고 해당 부분에서 미세 조정을 수행한다. 또한, 환상이 발생하기 쉬운 층(layer)을 선택하여 타겟팅된 미세 조정을 통해 모델의 환상 발생을 최소화한다.

- **Performance Highlights**: TruthfulQA와 FACTOR 데이터셋에서 광범위한 실험을 수행한 결과, F2는 기존 모델 대비 상당한 성능 향상을 보였다. 특히, 기존 최첨단 조작 방법들과도 상호 보완적으로 작용하여 성능을 더욱 향상시켰다는 결과를 얻었다.



### The Fall of ROME: Understanding the Collapse of LLMs in Model Editing (https://arxiv.org/abs/2406.11263)
- **What's New**: 최근 대규모 언어 모델(LLMs)을 수정하는 방법들이 개발되고 있지만, 실제 적용에서 모델 붕괴(collapse) 문제를 야기할 수 있는 위기가 있습니다. 특히 ROME(Rank-One Model Editing) 방법은 단 한 번의 수정으로도 모델을 붕괴시킬 수 있어 주목받고 있습니다. 본 연구에서는 이러한 붕괴 현상의 근본 원인 두 가지를 제시합니다: i) 매개변수 업데이트 방정식에서 접두어(prefixed)와 미접두어(unprefixed) 키를 일관성 없이 처리하여 매우 작은 분모로 인해 지나치게 큰 매개변수 업데이트가 발생하는 경우, ii) 붕괴 사례에서 첫 번째 토큰의 미접두어 키 분포가 접두어 키 분포와 크게 달라지는 경우. 이 문제를 해결하기 위해 간단하면서 효과적인 접근법을 제안하여, 수정 단계에서 일관되게 접두어 키를 사용하고 테스트 단계에서 접두어를 추가하는 방법을 제안합니다.

- **Technical Details**: ROME 기법은 Transformer 아키텍처의 MLP 모듈을 키-값 연관 메모리(key-value associative memory)로 모델링합니다. 모델 내의 지식을 키 벡터와 값 벡터로 재현하며, 수정을 위해 변환 행렬(transformation matrix)을 조정합니다. 그러나 ROME 기법은 매개변수 업데이트 과정에서 접두어와 미접두어 키의 불일치 문제로 인해 붕괴가 발생할 수 있습니다. 특히 붕괴 사례에서는 첫 번째 토큰의 미접두어 키 분포가 다르게 나타나 문제가 심화됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 모델 붕괴를 방지하면서도 수정 효율성을 유지하는 것을 확인했습니다. 전체적으로 키 벡터의 일관된 사용과 테스트 시 접두어 추가 방식이 효과적인 것으로 나타났습니다.



### Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection (https://arxiv.org/abs/2406.11260)
Comments:
          8 pages

- **What's New**: 이 연구는 AdStyle이라는 적대적 스타일 증강(adversarial style augmentation) 기술을 도입하여 다양한 스타일 전환 공격에 대해 강건한(fake news) 가짜 뉴스 탐지기를 훈련하는 방법을 제안합니다. 주요 메커니즘은 LLMs(Large Language Models)를 사용하여 자동으로 다양한 스타일 전환 공격 프롬프트(prompts)를 생성하는 것입니다. 이는 탐지기가 처리하기 어려운 프롬프트를 생성하여 탐지 성능을 향상시킵니다.

- **Technical Details**: AdStyle의 주요 메커니즘은 LLMs를 활용해 다양한 스타일 전환 프롬프트를 자동으로 생성하는 것입니다. 이를 통해 프롬프트와 탐지기의 성능 간의 패턴을 추론하고 탐지기를 가장 혼란스럽게 만드는 적대적 프롬프트를 찾는 자동화된 프롬프트 엔지니어링(prompt engineering) 기술이 도입되었습니다. AdStyle은 데이터셋의 무작위 부분 집합을 선택하고, 다양한 프롬프트를 적용해 증강된 샘플을 생성하며, 이러한 증강된 샘플을 탐지기에 입력하여 성능을 측정합니다. 성능이 떨어지는 프롬프트를 선택하여 교육 데이터셋에 적용하고 반복적인 훈련 과정을 통해 탐지기를 더 강건하게 만듭니다.

- **Performance Highlights**: 기존 가짜 뉴스 벤치마크 데이터셋을 사용한 실험 결과, AdStyle의 증강 전략이 기존 방법에 비해 더 높은 강건성 및 탐지 성능을 보여주었습니다. 이는 다양한 공격 시나리오에서도 더 나은 성능을 입증하며, 새로운 공격 프롬프트를 추가함으로써 다양한 공격 전략에 확장 가능함을 나타냅니다.



### Enhancing Biomedical Knowledge Retrieval-Augmented Generation with Self-Rewarding Tree Search and Proximal Policy Optimization (https://arxiv.org/abs/2406.11258)
- **What's New**: 본 연구는 Self-Rewarding Tree Search (SeRTS)라는 새로운 플러그 앤 플레이(plug-and-play) LLM 기반 검색 방법을 제안합니다. 이는 Monte Carlo Tree Search (MCTS)와 자가 보상(self-rewarding) 패러다임에 기반합니다. 이러한 접근 방식을 통해 문서 검색의 품질을 높이고 의료 지식 질의에 대해 더 정확하고 포괄적인 답변을 제공할 수 있습니다.

- **Technical Details**: SeRTS는 LLM의 추론 능력과 MCTS의 검색 능력을 결합하여 제로샷(zero-shot) 성능을 향상시킵니다. 검색 성능을 더욱 향상시키기 위해, SeRTS가 수집한 데이터로 Proximal Policy Optimization (PPO) 목표를 사용하여 LLM을 미세 조정(fine-tuning)합니다. 이를 통해 복잡한 질의와 문서 관계를 고려한 트리 검색 문제로 문서 검색 작업을 공식화합니다.

- **Performance Highlights**: BioASQ-QA 데이터셋을 사용한 GPT-3.5-Turbo와 LLama2-7b를 이용한 실험에서는 SeRTS 방법이 BM25 검색기보다 성능을 크게 향상시키고 자체 반사(self-reflection) 강력한 기준선도 능가하는 것으로 나타났습니다. SeRTS는 PPO 트레이닝에 더 높은 품질의 피드백을 생성하며, 이는 의료 지식 질의에 대해 더 관련성 높은 문서를 검색하는 능력을 향상시킵니다.



### Dynamic Data Mixing Maximizes Instruction Tuning for Mixture-of-Experts (https://arxiv.org/abs/2406.11256)
- **What's New**: 이번 논문은 Mixture-of-Experts(MoE) 모델의 instruction tuning에서 데이터셋 간의 중복성을 줄이고 효율성을 높이기 위한 동적 데이터 혼합 방식을 제안합니다. 기존의 방법들이 고정된 샘플링 가중치를 사용하는 데 반해, 본 연구에서는 MoE의 token routing preference를 기반으로 데이터셋 레벨의 표현을 구축하고, 각 데이터셋 간의 미세한 차이를 포착하여 실제로 도움을 주는 데이터셋에 동적으로 가중치를 조정합니다.

- **Technical Details**: 이 방법론은 MoE 모델의 고유 특성인 sparsity(스파르시티)와 token routing(토큰 라우팅)을 활용하여 데이터셋 간의 차이를 계산합니다. 구체적으로, 각 데이터셋에 대해 gate load(게이트 로드)를 계산하고 이를 데이터셋의 표현으로 사용합니다. 그런 다음, 이 표현들 간의 L2 distance(L2 거리)를 계산하여 모델의 내부 상태를 반영합니다. 이후, 이전 샘플링 가중치와 현재 거리 정보를 기반으로 새로운 샘플링 가중치를 동적으로 업데이트하는 알고리즘을 제안합니다.

- **Performance Highlights**: 두 개의 MoE 모델과 네 가지 대표적인 instruction datasets를 조합하여 실험을 수행한 결과, 제안된 동적 샘플링 방법이 다양한 지식 테스트, 추론, 개방형 질문 응답 작업에서 뛰어난 성능을 보였습니다. 추가로, 전문가의 특화와 데이터 조합의 다양한 경우에 대한 분석도 제공합니다.



### Can Machines Resonate with Humans? Evaluating the Emotional and Empathic Comprehension of LMs (https://arxiv.org/abs/2406.11250)
Comments:
          18 pages

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 활용하여 공감을 이해하려는 다양한 전략을 제안합니다. 주요 전략으로는 대조 학습(contrastive learning) 및 CoT(Chain-of-Thought)를 활용한 지도 학습(supervised fine-tuning)을 포함합니다. 대조 학습을 통해 유사한 예제의 임베딩을 밀어주고, 서로 다른 예제의 임베딩은 멀어지게 하여 공감 이해를 높이고자 합니다.

- **Technical Details**: 주요 데이터셋으로는 Shen et al. (2023)의 EmpathicStories 데이터를 사용하였으며, 이 데이터는 개인 경험을 담은 1,500개의 이야기를 포함하고, 이야기 쌍 간의 유사성을 평가합니다. 각 이야기 쌍에 대해 사건(event), 감정(emotion), 도덕(moral), 그리고 전반적인 공감(empathy) 측면에서 1에서 4까지의 점수로 유사성을 측정합니다. 모델 학습 및 평가 방법으로는 Pearson 및 Spearman 상관계수와 평균 제곱 오차(MSE)를 사용합니다.

- **Performance Highlights**: 이번 연구에서는 모델이 사건(event)의 유사성을 잘 인식하는 반면, 감정(emotion)과 공감(empathy)과 같은 미묘한 차이를 인식하는데 어려움을 겪는다는 결과를 도출했습니다. 대조 학습을 도입한 결과, Pearson 및 Spearman 상관계수에서 약 5–10%의 향상을 보였으나, 정확도는 일정 수준을 넘지 못했음을 발견했습니다. 이는 공감 유사성 평가의 주관적인 특성 때문임을 시사합니다.



### FamiCom: Further Demystifying Prompts for Language Models with Task-Agnostic Performance Estimation (https://arxiv.org/abs/2406.11243)
- **What's New**: 이 논문에서는 말뭉치를 기반으로 한 모델의 성능을 더 정확하게 예측하기 위한 새로운 측정 지표인 FamiCom(Familarity and Complexity Based Performance Estimation)을 소개하고 있습니다. FamiCom은 기존의 친숙도(familiarity)만을 고려한 지표가 아닌, 텍스트의 내재된 복잡함(complexity)을 함께 고려하여 모델의 성능을 예측합니다.

- **Technical Details**: FamiCom는 텍스트의 복잡성(complexity)과 모델의 친숙도(familiarity)의 두 가지 주요 요소를 기반으로 계산됩니다. 기존의 혼란도(perplexity) 측정 기법을 기반으로 하여 모델이 프롬프트에 대해 얼마나 익숙한지를 평가하고, 추가적으로 텍스트 자체가 얼마나 복잡한지 평가합니다. 복잡성은 문제를 해결하는 데 필요한 단계 수로 정의됩니다. 이를 통해 FamiCom는 다양한 텍스트 입력 상황에서 보다 정확한 성능 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, FamiCom는 기존 지표에 비해 더 높은 상관관계(스피어만 상관계수 0.85)를 보였습니다. 또한, 자동 프롬프트 선택 및 예시 선택에 적용했을 때 기존 방법들보다 7.0% 이상의 정확도 향상을 이루었습니다. 이러한 결과는 FamiCom가 보다 일관되게 모델의 성능을 예측할 수 있음을 증명합니다.



### Evading AI-Generated Content Detectors using Homoglyphs (https://arxiv.org/abs/2406.11239)
- **What's New**: 최신 연구는 유사 글자(homoglyph)를 활용한 새로운 공격 방법을 제시하였습니다. 이 방법은 기존의 대형 언어 모델(LLM) 검출기를 회피할 수 있으며, 이를 통해 AI 생성 텍스트 감지를 회피할 수 있음을 보여줍니다.

- **Technical Details**: 유사 글자(homoglyph)를 사용한 공격은 라틴 알파벳의 'a'를 그리스 알파벳의 'α'와 같이 시각적으로 유사한 다른 알파벳으로 대체하는 것입니다. 이러한 대체는 텍스트의 토큰화(tokenization) 및 토큰 로그 가능성(token loglikelihood)을 변화시켜 탐지를 피할 수 있게 합니다. 이 논문은 Binoculars, DetectGPT, OpenAI의 탐지기 및 워터마킹(watermarking) 기법을 포함한 최첨단 LLM 탐지기와 다섯 가지 다른 데이터셋에서 유사 글자 공격의 효능을 평가했습니다.

- **Performance Highlights**: 유사 글자 공격은 모든 평가된 탐지기와 데이터셋 구성에서 효율성을 크게 감소시켜, 정확도가 0.5(무작위 추측)까지 떨어졌습니다. 이는 유사 글자 기반 공격이 기존의 LLM 탐지기를 효과적으로 회피할 수 있음을 의미합니다.



### What Kinds of Tokens Benefit from Distant Text? An Analysis on Long Context Language Modeling (https://arxiv.org/abs/2406.11238)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 문맥 길이가 증가함에 따라 어떤 종류의 단어들이 더 많은 이점을 얻는지를 탐구합니다. 기존 연구와 달리, 이 연구는 언어 모델이 문맥 길이가 증가함에 따라 더욱 향상된 성능을 보이는 현상을 발견했습니다. 이는 인간의 읽기 및 쓰기 습관과는 달리, 언어 모델이 매우 긴 문맥에서도 예측 능력이 증가하는 것을 보여줍니다.

- **Technical Details**: 논문은 여러 종류의 토큰들이 문맥 길이 변화에 어떻게 반응하는지를 분석합니다. 콘텐츠 단어(예: 명사, 형용사)와 단어의 첫 번째 토큰이 긴 문맥에서 가장 큰 이점을 얻는 것으로 나타났습니다. 또한, 문맥 내에 자주 등장하는 패턴(N-그램)도 예측에 큰 영향을 미쳤습니다. 반면, 모델의 사전 지식은 드물게 나타나는 토큰에 대해 유의미한 영향을 미쳤습니다. 문맥이 증가함에 따라 언어 모델의 예측 확률 분포가 더욱 날카로워지는(overconfidence) 현상도 관찰되었습니다.

- **Performance Highlights**: 긴 문맥을 처리하는 능력을 평가할 때, 언어 모델의 퍼플렉시티(perplexity)는 문맥 길이가 증가함에 따라 감소하는 경향을 보였습니다. 퍼플렉시티가 낮아질수록 모델의 언어 모델링 능력이 향상됨을 의미합니다. 이 연구는 LLM의 성능이 인간의 읽기 및 쓰기 습관과는 달리, 긴 문맥에서도 지속적으로 증가하는 이유를 밝히는데 기여합니다.



### MiniConGTS: A Near Ultimate Minimalist Contrastive Grid Tagging Scheme for Aspect Sentiment Triplet Extraction (https://arxiv.org/abs/2406.11234)
Comments:
          arXiv admin note: text overlap with arXiv:2403.07342

- **What's New**: 이번 연구에서는 기존의 태깅 스킴(tagging scheme)의 복잡성과 사전 학습 모델의 내부 표현 개선을 재평가합니다. 제안된 방법은 최소한의 태깅 스킴(minimalist tagging scheme)과 새로운 토큰 수준의 대조 학습 전략(token-level contrastive learning strategy)을 통합하여 사전 학습 표현(pretrained representations)을 향상시킵니다. 제안된 접근법은 최첨단 기법들과 비교해서도 동등하거나 더 나은 성능을 발휘하며, 컴팩트한 디자인과 적은 계산 오버헤드를 특징으로 합니다. 또한, GPT-4의 Few-shot Learning과 Chain-of-Thought 시나리오에 대한 성능을 처음으로 평가했습니다.

- **Technical Details**: 기존의 파이프라인 방법(Pipeline methods)은 단계별로 다중 서브태스크를 분해하여 ASTE(Aspect Sentiment Triplet Extraction) 작업을 수행하며 오류 전파 문제를 겪습니다. 반면, 조인트 태깅 방법(Joint Tagging methods)은 통합된 태깅 스킴을 통해 모든 트리플릿 요소를 한 번에 추출합니다. 본 연구에서는 최소한의 그리드 태깅 스킴(Minimalist Grid Tagging Scheme)과 토큰 수준 대조 학습 전략(token-level contrastive learning approach)을 도입하여, 사전 학습된 인코더의 표현력을 효과적으로 활용하고자 합니다. 이 방법은 최소한의 레이블 클래스를 사용하여 학습 과정을 단순화하고 빠른 수렴을 촉진하며, 대조 학습과도 잘 맞물리도록 설계되었습니다.

- **Performance Highlights**: 광범위한 벤치마크 데이터셋에 대한 실험과 평가를 통해 제안된 방법이 기존 접근법보다 우수한 성능을 나타냄을 입증했습니다. 특히, 대규모 언어 모델(LLM) 시대에서도 GPT 3.5 및 GPT 4보다 더 효과적인 성능을 보였으며, few-shot과 Chain-of-Thought 학습 시나리오에서 우수한 효율성과 효과성을 입증했습니다.



### ComperDial: Commonsense Persona-grounded Dialogue Dataset and Benchmark (https://arxiv.org/abs/2406.11228)
- **What's New**: ComperDial은 오픈 도메인 대화 시스템에 대한 평가 지표의 훈련과 평가를 위해 새로운 벤치마크를 제안합니다. 이 벤치마크는 1,485개의 대화에서 10,395개의 대화 턴에 대한 인간 평가 점수를 포함하고 있으며, CPD 도전 과제에 제출된 99개의 대화 에이전트로부터 수집되었습니다. 이를 통해 다양한 특성을 가진 여러 응답을 제공하여 학습된 대화 지표의 보다 견고한 평가를 가능하게 합니다.

- **Technical Details**: ComperDial은 단일 턴 응답 점수뿐만 아니라, 대화 수준에서 인간 주석 점수를 포함하여 여러 턴 동안의 모델 응답을 공동으로 평가할 수 있습니다. 또한 새로운 자동 평가 지표인 CPDScore를 개발하여 모델 생성 대화의 인간 대화와의 유사성을 측정합니다. CPDScore는 Chain-of-Thought 추론 및 다단계 프롬프팅을 사용하여 대화 평가의 감사 가능한 설명을 제공합니다.

- **Performance Highlights**: CPDScore는 기존 메트릭스보다 인간 판단과의 상관성이 더 높게 나타났습니다. 실험 결과에 따르면, CPDScore는 ComperDial 데이터셋에서 인간 점수와 더 높은 상관성을 보여줍니다.



### Building another Spanish dictionary, this time with GPT-4 (https://arxiv.org/abs/2406.11218)
- **What's New**: 우리는 'Spanish Built Factual Freectianary 2.0 (Spanish-BFF-2)'라는 AI 생성 스페인어 사전의 두 번째 버전을 소개합니다. 이전에는 GPT-3를 사용하여 최초의 무료 사전을 개발했었습니다. 이번 연구에서는 GPT-4-turbo를 사용하여 사전을 개선하고자 합니다.

- **Technical Details**: 이 연구에서는 GPT-3와 GPT-4-turbo의 성능을 비교하고, 초기 버전에서 이루어진 개선 사항들을 탐구합니다.

- **Performance Highlights**: 새로운 버전은 GPT-4-turbo의 사용으로 인해 향상된 정확도와 성능을 제공할 것으로 기대됩니다. 이에 따라 초기 버전에 비해 개선된 결과가 나타날 것입니다.



### Global Data Constraints: Ethical and Effectiveness Challenges in Large Language Mod (https://arxiv.org/abs/2406.11214)
Comments:
          6 pages, 3 figures, and 5 tables

- **What's New**: 본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 효과성 및 윤리적 무결성에 큰 영향을 미치는 훈련 데이터셋의 다양성과 품질에 대해 다루고 있습니다. 특히, 데이터 접근성이 제한된 지역에서 고품질 훈련 데이터를 획득하는 데에 따른 다각적 문제를 집중적으로 분석합니다. 이를 통해 공공 데이터에 의존할 경우 발생할 수 있는 편향 및 환각(bias and hallucination) 문제를 제시하고, 이를 완화할 수 있는 전략을 제안합니다.

- **Technical Details**: 본 연구에서는 GPT-4와 GPT-4o 모델을 사용하여 데이터 제약이 모델 성능 및 윤리적 정렬에 미치는 영향을 평가합니다. 감시 및 필터링 기법을 통해 데이터 품질을 향상시키고 모델의 견고성을 강화하는 다양한 전략을 검증하였습니다. 예를 들어, 특정 토큰 길이에 따른 성능 변화를 평가하기 위한 각종 메트릭(성능 기준)들을 사용하여 구조화된 테스트를 수행하였습니다.

- **Performance Highlights**: GPT-4 및 GPT-4o 모델은 이미지 및 텍스트 입력을 처리하여 텍스트 출력을 생성하며, 다양한 학술 및 전문 벤치마크에서 인간 수준의 성과를 달성했습니다. 모델의 사실성(factuality)과 원하는 행동을 향상시키기 위해 사후 훈련 정렬 과정(post-training alignment process)을 통해 성능을 개선하였습니다. 또한, 데이터를 통해 긴 토큰의 소유 빈도 및 문장 생성에 미치는 영향을 평가하였으며, 이를 통해 모델 A와 모델 B를 비교하여 성능을 평가했습니다.



### Fine-Tuning or Fine-Failing? Debunking Performance Myths in Large Language Models (https://arxiv.org/abs/2406.11201)
Comments:
          8 pages, 4 figures

- **What's New**: OpenAI는 장대언어모델(LLMs)에 대한 미세 조정을 통해 도메인별 질문에 대한 성능을 향상시키는 과정에 대해 논의하고 있다. 이 연구는 이러한 개념을 정보 검색을 통해 정확성과 관련성을 높이려는 RAG(Retrieval-Augmented Generation) 파이프라인에 통합하는 것을 목표로 한다.

- **Technical Details**: 이 연구는 주로 세 가지 공개된 질문-응답 데이터셋(BioASQ, Natural Questions, Qasper)을 사용하여 LLM이 RAG 파이프라인에서 얼마나 잘 성능을 발휘하는지 평가, Mistral, LlaMA2, GPT-4 모델을 활용했다. Mistral과 LlaMA2 모델은 각각 200, 500, 1000개의 질문-응답 세트로 미세 조정되었으며, 성능 비교를 위해 기본 버전 모델과 대조했다.

- **Performance Highlights**: 연구 결과, 미세 조정된 LLM이 RAG 시스템에서 성능이 저하되는 현상을 보였다. 이는 OpenAI가 제안한 단독 LLM 애플리케이션에서의 향상과는 달리, 도메인에 따라 더욱 신중한 검증이 필요함을 시사한다.



### In-Context Editing: Learning Knowledge from Self-Induced Distributions (https://arxiv.org/abs/2406.11194)
- **What's New**: 기존 언어 모델의 미세 조정(fine-tuning) 방식은 새로운 정보를 통합하는 데 있어서 매우 취약합니다. 새로운 접근법으로 Consistent In-Context Editing (ICE)를 제안합니다. ICE는 모델의 학습을 단일 결과 대신 분포(distribution)로 향하도록 조정하여, 더 견고한 미세 조정을 가능하게 합니다.

- **Technical Details**: ICE는 모델의 예측과 원핫(one-hot) 타겟 분포 사이의 거리를 줄이는 대신, 맥락(contextual) 정보를 학습 과정에 포함시켜 모델이 새로운 분포에 맞춰 조정됩니다. 이는 모델이 맥락 내 학습(in-context learning) 능력을 활용하여, 변화가 모델 분포의 특정 영역에 국한되도록 합니다. 또한, ICE는 그래디언트 기반 튜닝 방법의 견고성과 효과를 높이는 간단한 최적화 프레임워크를 도입합니다.

- **Performance Highlights**: ICE의 성능은 정확도(accuracy), 국소성(locality), 일반화(generalization), 언어적 품질(linguistic quality)이라는 네 가지 핵심 요소에서 분석되었습니다. 네 가지 데이터셋에서의 실험 결과, ICE는 지속적인 편집이 가능하며, 업데이트된 정보가 통합되면서 기존 모델의 무결성을 유지합니다.



### MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Mod (https://arxiv.org/abs/2406.11193)
- **What's New**: 이 논문은 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)에서 도메인별 뉴런(domain-specific neurons)을 식별하고, 다양한 시각적 도메인의 특징을 처리하는 메커니즘을 연구합니다. 약 1%의 뉴런이 도메인별 뉴런임을 밝혀내었으며, MLLMs의 피드-포워드 네트워크(Feed-Forward Network, FFN) 계층에서 이러한 뉴런의 분포를 조사했습니다.

- **Technical Details**: 논문에서는 다음과 같은 프레임워크를 제안합니다: 이미지를 시각적 특징으로 변환하고, 이 특징을 단어 임베딩(word embedding) 공간으로 투영하여 언어 모델 모듈에서 텍스트 출력을 생성. 이를 통해 다섯 개의 서로 다른 도메인에서의 뉴런 활성화 패턴을 분석했습니다. 이 도메인에는 자율 주행, 원격 센싱, 의학, 문서 및 일반 장면이 포함됩니다. 또한, 로그 잇 렌즈(logit lens)를 사용하여 LLM 모듈의 중간 계층에서 특징 변환을 시각화했습니다.

- **Performance Highlights**: 광범위한 실험 결과, 현재 MLLMs는 시각적 질문 응답(Visual Question Answering, VQA) 능력을 보여주지만 도메인별 정보를 완전히 활용하지 못하고 있음을 발견했습니다. 도메인별 뉴런을 적절히 조작하면 정확도가 최대 10%까지 변할 수 있으며, 이는 향후 크로스-도메인 멀티모달 모델 개발에 중요한 정보를 제공합니다.



### Beyond Boundaries: Learning a Universal Entity Taxonomy across Datasets and Languages for Open Named Entity Recognition (https://arxiv.org/abs/2406.11192)
Comments:
          20 pages. Project page: this https URL

- **What's New**: 최신 연구 B2NERD는 기존의 일관성 없는 엔터티 정의와 중복 데이터를 해결함으로써 Open Named Entity Recognition (NER)을 위한 대규모 언어 모델(LLMs)의 일반화 성능을 크게 향상시키는 새로운 데이터셋입니다. B2NERD는 영어 및 중국어로 된 54개의 기존 데이터셋을 정규화하여, 400개 이상의 엔터티 유형을 가지는 보편적인 분류 체계를 제공합니다.

- **Technical Details**: B2NERD의 데이터셋은 두 가지 접근법으로 구축되었습니다. 첫 번째로, 데이터셋 간의 엔터티 정의 불일치를 모델 기반 교차 검증 및 규칙 기반 스크리닝 방법으로 자동 감지하여, 각 고유 유형에 대해 명확한 라벨 이름을 지정함으로써 보편적인 분류 체계를 확립했습니다. 두 번째로, 범주적 및 의미적 다양성을 강조하는 데이터 가지치기 전략을 사용하여 중복성을 최소화하였습니다. 이 과정에서 텍스트 유사성이 낮은 샘플을 선택하여 범주 간 및 의미 다양성을 확보했습니다.

- **Performance Highlights**: B2NERD로 학습된 B2NER 모델은 15개의 데이터셋과 6개의 언어로 구성된 3개의 도메인 외 벤치마크에서 GPT-4를 6.8-12.0 F1 포인트 초과하며, 기존 방법들보다 뛰어난 성능을 보였습니다. 특히, 영어에서는 3.0%, 중국어에서는 6.8%, 다중 언어 설정에서는 6.7%의 성능 향상이 있었습니다. B2NERD는 다양한 도메인에서의 실세계 Open NER 요구사항을 충족시키는 데 중요한 기여를 했습니다.



### A Survey on Human Preference Learning for Large Language Models (https://arxiv.org/abs/2406.11191)
- **What's New**: 최근 다목적 대형 언어 모델(LLMs)의 발전은 인간의 의도와 일치시키기 위해 선호 학습(preference learning)을 활용하며, 다양한 맥락에서 뛰어난 적용성과 효과를 가지고 있습니다. 이 설문 논문은 사람의 선호도가 LLMs에 어떻게 도입되는지에 대한 관점을 제공하여, 인간의 선호도와 LLMs 사이의 관계 및 한계를 깊이 이해하는 데 도움을 주고자 합니다.

- **Technical Details**: 이 논문은 선호 피드백의 출처와 형식, 선호 신호의 모델링 및 사용, 그리고 일치된 LLMs의 평가를 다루며, 광범위한 피드백 소스와 형식을 분류하고 각 모델의 장단점을 비교합니다. 선호 신호의 사용 목적에 따라 RLHF(강화 학습), SFT(지도 학습), 대조 학습 및 조건부 생성의 다양한 기술을 설명합니다.

- **Performance Highlights**: 최근의 인간 선호 학습 방법은 높은 품질과 대규모 피드백을 위한 사람과 시뮬레이션의 피드백 수집을 결합했습니다. 입력 신호를 기준으로 LLM을 평가하는 다양한 평가 프로토콜을 제시하며, 각 방법의 장단점과 향후 연구 방향을 토의합니다.



### Aligning Large Language Models from Self-Reference AI Feedback with one General Princip (https://arxiv.org/abs/2406.11190)
Comments:
          19 pages, 3 figures

- **What's New**: 이 연구는 기존의 대규모 언어 모델(LLMs)을 사용하는 피드백 방식이 인간의 의도를 정확하게 반영하는 데 어려움을 겪는 문제를 해결하기 위해 새로운 자기참조 기반 AI 피드백 프레임워크를 제안합니다. 간단하고 일반적인 원칙인 '인류에게 가장 좋은 것'을 바탕으로 13B Llama2-Chat 모델이 높은 품질의 피드백을 제공할 수 있도록 합니다.

- **Technical Details**: 이 방법은 AI가 사용자의 질문에 먼저 응답하고, 그 응답을 기준으로 다른 답변에 대한 비판을 생성한 후, 비판을 토대로 인간의 선호에 더 적합한 답변을 결정하는 프로세스를 포함합니다. 또한 위치 편향을 줄이기 위해 자기 일관성 방법(self-consistency method)을 사용하고, 여러 답변 간의 선호 강도 차이를 계산하기 위해 의미론적 당황도(semantic perplexity)를 활용합니다.

- **Performance Highlights**: 실험 결과, 13B 및 70B Llama2-Chat 모델 주석자는 높은 품질의 선호 피드백을 제공하며, 이 선호 데이터를 기반으로 훈련된 정책 모델은 강화 학습을 통해 벤치마크 데이터셋에서 상당한 성능 향상을 달성했습니다.



### TIFG: Text-Informed Feature Generation with Large Language Models (https://arxiv.org/abs/2406.11177)
- **What's New**: 최근 발표된 연구에서는 텍스트 정보를 활용한 새로운 특징 생성 방법인 Text-Informed Feature Generation (TIFG)를 소개합니다. 기존의 방법들이 데이터 구조에만 집중하여 텍스트 정보를 간과한 반면, TIFG는 대형 언어 모델(LLM)과 Retrieval Augmented Generation (RAG) 기술을 통해 외부 지식에서 텍스트 정보를 추출해 새로운 특징을 생성합니다. 이 방법은 특징 공간을 풍부하게 하고 깊은 데이터 관계를 탐색할 수 있게 합니다.

- **Technical Details**: TIFG는 새로운 특징을 생성하기 위해 LLM을 사용하며 텍스트 정보를 분석하고 디코딩합니다. RAG 기술을 적용하여 LLM이 외부 지식을 기반으로 관련 정보를 검색하고 이를 활용해 새로운 특징을 생성합니다. 이 자동화된 프레임워크는 새로운 데이터 입력에 적응하고 반복적으로 특징 생성 프로세스를 최적화하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 다운스트림 작업에서 TIFG의 성능을 평가한 결과, 기존 방법들보다 높은 품질과 의미 있는 특징을 생성하는 것으로 나타났습니다. TIFG는 투명성과 설명 가능성을 갖춘 특징을 생성하며, 이는 특화된 도메인 작업에서 큰 이점을 제공합니다.



### Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinemen (https://arxiv.org/abs/2406.11176)
- **What's New**: 본 논문에서는 Iterative step-level Process Refinement (IPR) 프레임워크를 소개합니다. 이 프레임워크는 에이전트(agent) 훈련을 향상시키기 위해 단계별(step-by-step) 지침을 제공합니다. IPR은 에이전트가 전문가의 경로를 탐색하면서 새로운 행동을 생성하고, 이러한 행동을 단계별 보상(step-level rewards)를 사용하여 전문가의 경로와 비교합니다. 이를 통해 식별된 대조 행동 쌍(contrastive action pairs)을 훈련 데이터로 활용합니다.

- **Technical Details**: IPR 프레임워크는 Monte Carlo 방법을 사용하여 단계별 보상을 추정합니다. 각 반복(iteration) 동안, 에이전트는 전문가 트랙터리(expert trajectory)를 따라 탐색하고 새로운 행동을 생성합니다. 이러한 행동은 해당 단계별 보상과 비교 평가됩니다. 이를 통해 발견된 불일치는 트레이닝 데이터로 사용되어 에이전트를 훈련합니다.

- **Performance Highlights**: 세 가지 복잡한 에이전트 작업에 대한 실험에서, IPR 프레임워크는 다양한 강력한 기준선(baselines)을 능가하는 성능을 보여주었습니다. 분석 결과, IPR은 행동 효율성(action efficiency)을 증가시키는 데 효과적이며 다양한 모델에 적용 가능함을 확인했습니다.



### BSRBF-KAN: A combination of B-splines and Radial Basic Functions in Kolmogorov-Arnold Networks (https://arxiv.org/abs/2406.11173)
Comments:
          6 pages, 1 figure, 3 tables

- **What's New**: 이번 논문에서는 B-splines와 Radial Basis Functions(RBFs)를 결합한 Kolmogorov Arnold Network (KAN)인 BSRBF-KAN을 소개합니다. 이는 데이터 학습 시 입력 벡터를 피팅하는 네트워크입니다. MNIST 데이터셋을 이용한 실험에서 BSRBF-KAN은 5번의 훈련에서 평균 정확도 97.55%를 보이며 다른 네트워크보다 더 나은 수렴성을 보여줍니다. 관련 자료는 공개된 저장소에서 확인할 수 있습니다.

- **Technical Details**: 최근 Liu 등은 KANs에 관한 연구를 통해 고정된 노드를 사용하는 대신 학습 가능한 활성화 함수를 '엣지'로 사용하여 데이터를 피팅하는 새로운 패러다임을 제시했습니다. KAN의 이론적 기반은 Kolmogorov-Arnold 표현 정리(KART)으로, 이는 다변수 연속 함수를 단일 변수의 연속 함수와 덧셈으로 표현할 수 있음을 의미합니다. 이 논문에서는 여기에 영감을 받아 B-splines와 RBFs를 결합하여 BSRBF-KAN을 구축하였습니다. 기존에 성공적으로 활용된 EfficientKAN과 FastKAN을 바탕으로 설계된 이 네트워크는 연속적 기저 함수를 사용해 함수 근사성을 높입니다.

- **Performance Highlights**: MNIST 데이터셋을 사용하는 실험에서 BSRBF-KAN은 5번의 훈련을 통해 평균 정확도 97.55%를 기록하였으며, 이는 다른 네트워크와 비교하여 더 나은 수렴성을 보여줍니다. 다중 수렴률과 높은 정확도를 통해 BSRBF-KAN의 우수성을 입증했습니다.



### Enhancing Criminal Case Matching through Diverse Legal Factors (https://arxiv.org/abs/2406.11172)
- **What's New**: 이번 논문에서는 다양한 법적 요소 (LFs)를 활용하여 범죄 사건의 유사성을 판단하는 새로운 프레임워크인 'Diverse Legal Factor-enhanced Criminal Case Matching (DLF-CCM)'을 제안합니다. 기존 방법들이 사건의 단일 표현에 의존하여 여러 법적 요소들을 포괄적으로 표현하지 못하는 문제를 지적하고 이를 해결하기 위한 두 단계의 프레임워크를 도입했습니다.

- **Technical Details**: DLF-CCM는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 대규모 법적 판결 예측 데이터셋을 사용해 다중 과제 학습 프레임워크를 통해 LF 추출 네트워크를 사전 학습합니다. 두 번째 단계에서는 LF 중복 제거 모듈을 도입하여 공통 LF와 개별적인 ARF (article-related factor), CRF (charge-related factor), TRF (term-related factor)를 학습합니다. 또한, 엔트로피 가중치 융합 전략을 도입하여 각 LF가 생성한 다중 유사성을 동적으로 융합합니다.

- **Performance Highlights**: 실험 결과 DLF-CCM의 효과가 입증되었으며, 경쟁 베이스라인에 비해 성능이 크게 개선되었습니다. 이는 다양한 법적 요소들을 고려하여 더 정확한 범죄 사건 매칭을 가능하게 했습니다.



### How Good are LLMs at Relation Extraction under Low-Resource Scenario? Comprehensive Evaluation (https://arxiv.org/abs/2406.11162)
- **What's New**: 이번 연구에서는 저자들이 중앙아시아, 동남아시아, 중동 지역의 10개 저자원 언어(Low-Resource Languages, LRLs)에 대해 새로운 관계 추출(Relation Extraction, RE) 데이터셋을 구축했습니다. 이를 통해 LRL에서의 관계 추출 성능을 향상시키기 위한 실험과 평가를 수행했습니다.

- **Technical Details**: 연구 과정에서 저자들은 영어 RE 데이터셋(NYT10, FewRel, CrossRE)을 다국어 기계 번역 모델을 사용해 LRL 언어로 번역했습니다. 다국어 기계 번역 모델로는 NLLB(Costa-jussà et al., 2022)를 사용했으며, 번역된 데이터셋의 질을 평가하기 위해 언어 혼란도(language perplexity, PPL)를 활용해 저질 데이터를 필터링했습니다. 최종적으로 여러 오픈 소스 대형 언어 모델(LLMs)의 성능을 구축된 LRL RE 데이터셋에서 검증했습니다.

- **Performance Highlights**: 저자들이 구축한 LRL RE 데이터셋은 중앙아시아, 동남아시아, 중동 지역의 10개 언어로 구성되었습니다. 이 데이터셋을 통해 다양한 오픈 소스 LLM들의 관계 추출 성능을 종합적으로 평가한 결과, LLM들이 저자원 조건에서도 일정 수준 이상의 성능을 보일 수 있음을 확인했습니다. 또한 데이터 품질 향상을 위해 효율적인 평가 서브모델을 도입한 점도 주요 성과 중 하나입니다.



### GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory (https://arxiv.org/abs/2406.11149)
- **What's New**: 최근 프라이버시 침해는 정보의 부적절한 전송에서 발생하며, 법적 규제가 복잡해지면서 이를 해결하기 위해 새로운 틀인 GoldCoin이 소개되었습니다. GoldCoin은 LLMs(Large Language Models)를 프라이버시 법률에 맞춰 효율적으로 활용할 수 있도록 돕는 프레임워크입니다. 이 프레임워크는 컨텍스추얼 인테그리티(contextual integrity) 이론을 활용해 다양한 시나리오를 생성, 프라이버시 위험을 식별하는 데 중점을 둡니다.

- **Technical Details**: GoldCoin 프레임워크는 HIPAA, COPPA, GDPR와 같은 프라이버시 법률을 기반으로 하여, LLMs가 실제 법적 사례에서 프라이버시 위험을 식별하도록 지원합니다. 컨텍스추얼 인테그리티(contextual integrity) 이론을 사용하여 정보 전송의 적절성 여부를 평가하며, 이를 통해 생성된 다양한 합성 시나리오를 LLMs에 학습시킵니다. GPT-4와 같은 모델을 활용하여 자동 필터링을 통해 중요한 요소를 포함한 고품질 데이터셋을 생성하고, 다양성 랭킹 메커니즘을 도입하여 데이터 일관성을 유지합니다.

- **Performance Highlights**: GoldCoin을 통해 학습된 모델은 실제 법적 사례에서 HIPAA의 적용 가능성을 식별하는 능력이 크게 향상되었습니다. 다른 기준 모델들보다 8%에서 23% 더 나은 성능을 보였으며, 프라이버시 위험을 탐지하는 데 있어 8%에서 18% 더 우수한 결과를 얻었습니다. 또한, 인간 분석 및 변형 연구 결과는 데이터 품질 향상 및 컨텍스추얼 인테그리티 이론의 효과를 확인했습니다.



### Breaking Boundaries: Investigating the Effects of Model Editing on Cross-linguistic Performanc (https://arxiv.org/abs/2406.11139)
Comments:
          Under review

- **What's New**: 이 논문은 사전학습된 언어 모델(Pretrained Language Models, PLMs)인 BERT와 GPT와 같은 모델이 NLP 분야에 변혁을 가져왔지만, 언어적 불균형도 초래하고 있다는 점을 지적합니다. 연구진은 다양한 지식 편집 기법을 다국어 환경에서 평가하여 언어적 형평성의 필요성을 강조하고 있습니다. 영어, 독일어, 프랑스어, 이탈리아어, 스페인어, 힌디어, 타밀어, 칸나다어 등 다양한 언어에 걸쳐 Mistral, TowerInstruct, OpenHathi, Tamil-Llama, Kan-Llama 등의 모델을 평가했습니다.

- **Technical Details**: 연구진은 '각 언어 자체를 위한' (each language for itself, ELFI) 및 '각 언어를 다른 언어를 위해' (each language for others, ELFO) 등 두 가지 접근법을 사용해 모델을 스트레스 테스트했습니다. 대표적인 편집 방법인 ROME과 MEMIT를 사용해 모델의 성능과 언어 간 일관성을 평가했습니다. 연구는 특히 언어 일관성 유지에 대한 과제를 해결하기 위한 고난도의 기술적 전략을 개발했습니다.

- **Performance Highlights**: 연구 결과, 모델 합병(merged models)은 언어 간 일관성을 유지하는 데 여전히 부족함이 있다는 점이 확인되었습니다. 오류 분석을 통해 언어적 차이가 모델의 응답에 영향을 미치는 방식을 심도 있게 조사했습니다. 연구는 지식 편집이 다국어 환경에서 어떻게 각기 다른 언어로 전달될 수 있는지, 그리고 이를 통해 AI 기술에서의 언어적 형평성을 어떻게 달성할 수 있는지에 대한 기초를 마련했습니다.



### RePrompt: Planning by Automatic Prompt Engineering for Large Language Models Agents (https://arxiv.org/abs/2406.11132)
- **What's New**: 최근 몇 년간 대형 언어 모델(LLMs)은 전통적인 자연어 처리 분야를 넘어 코드 생성, 여행 계획, 로봇 제어 등 다양한 애플리케이션 도메인에서 뛰어난 성과를 거두었습니다. 이 논문에서는 이러한 LLM들로 생성되는 결과물을 최적화하기 위한 자동 프롬프트 엔지니어링(Automatic Prompt Engineering, APE) 방법으로 	extsc{RePrompt}를 제안합니다. 	extsc{RePrompt}는 챗 기록(Chat History)을 기반으로 '그라디언트 디센트(Gradient Descent)'를 수행하여 단계별 지시 사항을 최적화합니다.

- **Technical Details**: RePrompt는 주어진 도메인에서 사용되는 체인 오브 생각(Chain-of-Thought, CoT) 프롬프트 및 ReAct와 같은 상호작용 절차를 고려해 프롬프트를 개선합니다. 대화 기록에서 얻은 정보를 사용하여 프롬프트의 각 문장을 최적화하고, LLM이 특정 도메인에서 어떻게 계획을 세울지 학습합니다. 이를 통해 사용자는 다양한 추론 작업에서 초기 프롬프트 성능을 개선할 수 있습니다.

- **Performance Highlights**: Planning Domain Definition Language(PDDL) 생성 및 여행 계획 같은 실제 적용 사례에서 실험한 결과, RePrompt를 사용한 방법이 초기 프롬프트보다 더 높은 성공률을 보였습니다. 피드백이 비싼 경우와 저렴하지만 덜 정확한 경우 모두에서 효과적으로 작동함을 입증했습니다.



### Are Large Language Models a Good Replacement of Taxonomies? (https://arxiv.org/abs/2406.11131)
Comments:
          Accepted by VLDB 2024

- **What's New**: LLM의 지식 그래프 대체 가능성에 대한 연구

- **Technical Details**: 본 논문에서는 TaxoGlimpse라는 새로운 벤치마크를 제작하여 LLM의 성능을 다양한 계층 구조의 분류법(taxonomy)에서 평가했습니다.

- **Performance Highlights**: {'General performance': 'LLM은 일반적인 도메인(쇼핑 등)에서는 높은 성능을 보였으나, 전문가 도메인(생물학 등)에서는 성능이 떨어졌습니다.', 'Hierarchical levels': 'LLM은 계층 구조의 최상위(root) 레벨에서는 잘 작동하지만, 하위(leaf) 레벨로 내려갈수록 성능이 저하되는 경향을 보였습니다.', 'Prompting methods': 'Few-shot 및 Chain-of-Thoughts 핸들링도 성능 향상에 큰 영향을 미치지 않았습니다.'}

- **Future Opportunities**: 일반적인 도메인의 경우, LLM을 분류 기반 검색 및 추론에 활용하는 것이 유효하며, 전문가 도메인 또는 계층 구조의 상위 레벨에 대해서는 수동 분류법 구축을 지속하는 것이 좋습니다.



### Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis (https://arxiv.org/abs/2406.11130)
Comments:
          10 pages, 7 figures

- **What's New**: 새로운 Dynamic Order Template(DOT) 방법이 제안되었습니다. 이 방법은 각 인스턴스의 엔트로피를 기반으로 동적으로 뷰(view)를 생성하여 Aspect-based sentiment analysis(ABSA)의 성능을 개선합니다. 이를 통해 불필요한 계산을 줄이고 데이터 분포 변화에 대한 강건성을 보장합니다.

- **Technical Details**: DOT 방법은 두 단계로 구성됩니다. 첫 번째 단계는 프롬프트로 사용할 순서 템플릿을 예측하는 것이고, 두 번째 단계는 그 템플릿을 사용하여 감정 튜플(sentiment tuples)을 예측하는 것입니다. 각 단계에서 감정 튜플을 [A] (Aspect), [C] (Category), [S] (Sentiment), [O] (Opinion) 등의 마커 토큰을 사용하여 매핑합니다. 첫번째 단계에서는 모든 가능한 순서 뷰를 생성하고 그 엔트로피를 계산하여 각 인스턴스에 필요한 최적의 뷰 수를 동적으로 결정합니다. 이후 두번째 단계에서는 그 뷰를 사용하여 감정 튜플을 생성합니다.

- **Performance Highlights**: DOT 방법은 ASQP와 ACOS 데이터셋에서 F1-점수가 개선되었으며, 이전의 다중 뷰 방법론에 비해 추론 시간을 크게 줄였습니다. 또한 데이터 분포 변화에 더 강한 성능을 보였습니다.



### Grammaticality Representation in ChatGPT as Compared to Linguists and Laypeop (https://arxiv.org/abs/2406.11116)
Comments:
          23 pages

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 인간과 유사한 세밀한 문법 직관을 개발했는지 조사하는 첫 번째 대규모 연구를 제공합니다. 이전 연구(Sprouse, Schutze, & Almeida, 2013)가 148개의 언어 현상에 대해 수집한 평범한 사람들의 문법 판단을 기반으로, ChatGPT와 일반 사람들 및 언어학자들 간의 문법 판단을 비교했습니다.

- **Technical Details**: 실험 1에서는 ChatGPT가 주어진 참조 문장을 기반으로 문장에 점수를 매겼습니다. 실험 2에서는 7점 척도로 문장을 평가했으며, 실험 3에서는 두 문장 쌍 중 더 문법적인 문장을 선택하게 했습니다. 연구는 총 3개의 실험으로 이루어졌으며, 심리측정 학적 성질과 인간과 LLMs 간의 언어 처리 스타일의 차이에 기인한 결과를 보여줍니다.

- **Performance Highlights**: 전체적으로 ChatGPT와 언어학자 간의 일치율은 73%에서 95%까지 다양했으며, 전체 평균 일치율은 89%였습니다. 또한, 모든 작업에서 ChatGPT와 평범한 사람들 간에도 유의미한 상관관계가 발견되었으나, 그 상관관계의 강도는 작업에 따라 달랐습니다.



### Text Grafting: Near-Distribution Weak Supervision for Minority Classes in Text Classification (https://arxiv.org/abs/2406.11115)
- **What's New**: 이번 연구에서는 기존의 매우 약한 지도 학습 (XWS-TC) 방법을 개선하여, 소수 클래스에 대한 깨끗하고 분포 내(supervision)를 제공하기 위해 텍스트 접목(text grafting)이라는 새로운 프레임워크를 제안합니다. 텍스트 접목은 LLM(Logits 및 State-of-the-Art LLMs)을 사용하여 마이닝된 템플릿을 생성하고 이를 채워 소수 클래스의 텍스트를 합성하는 것을 목표로 합니다.

- **Technical Details**: 텍스트 접목은 세 가지 주요 단계로 구성됩니다: (1) 잠재적 텍스트 마이닝은 소수 클래스의 데이터를 합성하기 위해 원시 텍스트에서 유용한 구성 요소를 수집합니다. (2) 템플릿 생성은 합성에 기여하지 않는 구성 요소를 마스킹하여 템플릿을 만듭니다. (3) 템플릿 채우기는 최신 LLM을 사용하여 빈자리를 채워 텍스트를 합성합니다. 템플릿 마이닝 및 마스킹을 위해 확률 로그잇(probability logits)을 사용하고, 이는 종합된 텍스트의 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 원시 말뭉치(corpus)에서 소수 클래스를 분류하기 위해 제안된 프레임워크를 비교한 결과, 텍스트 접목은 최고의 텍스트 마이닝 및 데이터 합성 방법보다 뛰어난 성능을 나타냈습니다. 특히, 모든 단계와 중간 템플릿이 텍스트 접목의 성공에 기여하는 것으로 나타났으며, 텍스트 접목은 목표 클래스가 원시 말뭉치에 전혀 나타나지 않는 극단적인 상황에서도 견고함을 보여주었습니다. 이는 텍스트 접목이 매우 작은 말뭉치에서 효율적으로 작동할 수 있음을 의미합니다.



### Investigating Annotator Bias in Large Language Models for Hate Speech Detection (https://arxiv.org/abs/2406.11109)
- **What's New**: 이 논문은 GPT 3.5 및 GPT 4o와 같은 대형 언어 모델(Large Language Models, LLMs)이 혐오 발언 데이터를 주석(annotate)할 때 나타나는 편향(bias)에 대해 탐구합니다. 특히 젠더, 인종, 종교, 장애 등 네 가지 주요 범주에서 나타나는 편향을 다룹니다. 저자들은 HateSpeechCorpus라는 맞춤형 혐오 발언 탐지 데이터셋을 소개하고 이를 기반으로 연구를 진행했으며, ETHOS (Mollas et al., 2022) 데이터셋에서도 비교 실험을 수행했습니다.

- **Technical Details**: 연구는 Hatebase.org에서 혐오 발언 어휘집을 사용해 수집한 트윗 데이터 3003개를 기반으로 합니다. 트윗은 세 명의 언어 병리학 대학원생들이 각각의 트윗을 혐오적(hateful)인지 아닌지로 분류했습니다. 이 데이터에 대해 ChatGPT 'gpt-3.5-turbo' 및 'gpt-4o'를 사용하여 동일한 주석 작업을 수행했습니다. 또한, 연구에서는 소셜 미디어에서 가장 많은 혐오 발언을 받는 여섯 개의 취약 그룹을 선정하여 주석자 편향을 조사했습니다.

- **Performance Highlights**: 주석된 데이터에서 편향이 발견되었으며, 이는 젠더, 인종, 종교, 그리고 장애와 관련된 그룹에서 특히 현저했습니다. 연구 결과는 LLM이 일관된 주석 작업을 수행하면서도 특정 그룹에 대해 편향된 결과를 생성할 수 있음을 시사합니다. 이러한 발견은 LLM을 이용한 데이터 주석에서 편향을 줄이는 데 중요한 통찰력을 제공합니다.



### Exploring Safety-Utility Trade-Offs in Personalized Language Models (https://arxiv.org/abs/2406.11107)
Comments:
          Work in Progress

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 사용자 개별화(personalization)가 초래하는 성능 편향을 다룹니다. 특히, 사용자 정체성에 맞춰진 LLM이 안전성(safety) 및 유용성(utility)에서 어떻게 다른지 평가합니다. 다양한 LLM 모델(Llama, Mistral, GPT-3.5, GPT-4o)을 조사한 결과, 사용자의 정체성에 따라 성능 편차가 큼을 발견했습니다. 이는 기존 연구가 개별 사용자 요구에 맞춰 LLM을 개인화하는 방안에 집중한 것과는 대조적입니다.

- **Technical Details**: LLM의 개인화 편향(personalization bias)을 계량화하기 위해 논문에서는 안전성(safety)과 유용성(utility)의 두 축을 고려합니다. 안전성은 사용자 정체성이 명시될 때와 안될 때, 안전하지 않은 프롬프트에 대한 응답을 비교함으로써 측정됩니다. 유용성은 일반 지식, 수학 능력, 프로그래밍, 추론 등을 포함한 다양한 작업 수행 능력으로 평가됩니다. 연구는 31개의 다른 사용자 정체성(성별, 종교, 인종, 국적 등)을 고려한 광범위한 테스트를 수행했습니다.

- **Performance Highlights**: 결과적으로, LLM이 사용자 정체성을 명시할 때와 하지 않을 때 성능이 크게 변동함을 발견했습니다. 이는 LLM이 특정 정체성에 대해 더 안전한 응답을 하는 경우가 있는 반면, 유용성에서 저하되는 경우도 많음을 보여줍니다. 특히, 미성년자라고 명시할 때 성인의 콘텐츠를 회피하는 안전성이 증대되었습니다. 논문은 이러한 개인화 편향을 줄이기 위한 선호 조정(preference tuning) 및 프롬프트 기반 방어(prompt-based defenses) 전략도 제시합니다.



### From Intentions to Techniques: A Comprehensive Taxonomy and Challenges in Text Watermarking for Large Language Models (https://arxiv.org/abs/2406.11106)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 증가와 함께 불법적인 텍스트 사용을 방지하기 위한 '텍스트 워터마킹'의 필요성을 강조합니다. 최근의 연구 문헌을 종합적으로 조사하고 다양한 워터마킹 기술의 설계를 통합적으로 개괄합니다. 이 연구는 워터마킹 기술의 의도, 평가 데이터셋, 추가 및 제거 방법 등에 따른 분류 체계를 구축하고, 텍스트 워터마킹 분야의 주요 이슈와 오픈된 도전 과제를 강조합니다.

- **Technical Details**: 연구진은 워터마킹 기술을 개발자의 의도에 따라 텍스트 품질, 출력 분포, 모델 소유권 검증의 세 가지로 분류합니다. 텍스트 품질은 일반적으로 생성된 텍스트의 퍼플렉서티(perplexity)를 최소화하고, 의미적 유사성을 유지하는 것을 포함합니다. 퍼플렉서티는 모델이 텍스트 생성을 확신하는 정도를 나타내며, 낮을수록 더 정확한 예측을 의미합니다. 다른 접근 방식으로는 '그린-레드 리스트 규칙'을 이용하여 퍼플렉서티에 미치는 영향을 최소화하는 방법이 있습니다. 출력 분포를 유지하는 것은 자연스러운 사용자 경험을 제공하기 위해 중요한데, 이는 단어 배포를 조정하여 그 본래 텍스트와 유사하게 만드는 전략입니다.

- **Performance Highlights**: 다양한 워터마킹 기법이 평가되었으며, 각 기술은 최소한의 퍼플렉서티 영향을 유지하거나 의미적 유사성을 높이기 위해 설계되었습니다. 또한, 단어 분포를 재조정하여 원본 텍스트와 유사한 분포를 유지하는 전략이 성공적으로 적용되었습니다. 이러한 기술은 연구자들이 LLM 생성 텍스트의 품질을 유지하면서 워터마킹을 구현할 수 있도록 합니다.



### Grading Massive Open Online Courses Using Large Language Models (https://arxiv.org/abs/2406.11102)
Comments:
          v1. arXiv admin note: substantial text overlap with arXiv:2402.03776

- **What's New**: Massive open online courses(MOOCs)는 언제 어디서나 교육을 받을 수 있도록 해주는 혁신적인 플랫폼이지만, 수많은 참가자를 관리하는 데 어려움을 겪고 있다. 기존의 동료 평가(peer grading) 시스템은 편리하지만 신뢰성과 타당성 면에서 문제가 있다. 본 연구에서는 대형 언어 모델(LLMs)인 GPT-4와 GPT-3.5를 활용하여 동료 평가 시스템을 대체할 가능성을 탐구했다. 세 가지 MOOC 과목(소개 천문학, 우주생물학, 천문학의 역사와 철학)에서 ZCoT(zero-shot chain-of-thought) 기법을 사용하여 LLM의 성능을 평가했다.

- **Technical Details**: 세 가지 ZCoT 프롬프트 전략을 사용했다: 1) 강사가 제공한 정답을 포함한 ZCoT, 2) 강사가 제공한 정답과 루브릭(평가 기준)을 포함한 ZCoT, 3) 강사가 제공한 정답과 LLM이 생성한 루브릭을 포함한 ZCoT. 최종 성능은 18가지 설정에서 평가되었으며, 특히 강사가 제공한 정답과 루브릭을 포함한 ZCoT는 동료 평가보다 강사 점수와의 일치도가 높았다.

- **Performance Highlights**: GPT-4는 강사가 제공한 정답과 루브릭을 포함한 ZCoT를 사용할 때, 동료 평가보다 강사 점수와 더 잘 일치하는 성적을 산출하는 데 성공했다. 특히 정의된 루브릭이 있는 과목에서 그 효과가 두드러졌다. 이는 MOOCs에서 자동 채점 시스템이 학습 경험을 향상시킬 가능성을 시사한다.



### InstructCMP: Length Control in Sentence Compression through Instruction-based Large Language Models (https://arxiv.org/abs/2406.11097)
Comments:
          8 pages, 3 figures, accepted to ACL 2024 (Long Paper)

- **What's New**: 새로운 연구에서는 기존의 추출적 요약법(extractive summarization)과 문장 압축(sentence compression) 모델이 가지는 길이 제약을 해결할 수 있는 'Instruction-based Compression(InstructCMP)' 방법을 제안합니다. 이 방법은 명령을 통해 길이 제약을 고려할 수 있게 하며, 대규모 언어 모델(LLMs)의 제로샷(zero-shot) 문제 해결 능력을 활용합니다.

- **Technical Details**: InstructCMP는 문장 압축 작업을 명령 형식으로 변환하여 길이 제약을 반영할 수 있게 합니다. 이를 위해 기존의 데이터셋을 명령 형식으로 바꾸고 'length priming'이라는 접근 방식을 제안했습니다. 이 방법은 명령에 추가적인 길이 정보를 포함하여 모델 수정 없이도 제로샷 환경에서 효과적으로 작동합니다. 또한, 명령 형식으로 변환된 학습 데이터셋을 만들어 모델을 세부 조정(fine-tuning) 할 수 있습니다.

- **Performance Highlights**: 실험 결과, 'length priming'은 제로샷 및 세부 조정 모두에서 InstructCMP의 성능을 크게 향상시킵니다. ROUGE metric과 길이 제약 준수 측면에서도 ChatGPT (GPT-4 및 GPT-4-1106-preview)을 사용하여 유의미한 향상을 보였습니다. 또한, InstructCMP는 충실성을 유지하면서 문장을 압축할 수 있음을 실험적으로 입증했습니다.



### The Potential and Challenges of Evaluating Attitudes, Opinions, and Values in Large Language Models (https://arxiv.org/abs/2406.11096)
- **What's New**: 최근 Large Language Models (LLMs)의 발전은 이러한 모델들이 인간과 유사한 인지-행동 특성을 얼마나 잘 포착하고 표현할 수 있는지에 대한 관심을 불러일으켰습니다. 이 논문은 LLMs 내의 태도(Attitudes), 의견(Opinions), 가치(Values, AOV)를 평가하는 최근 연구들을 개괄하고, 이러한 평가의 잠재력 및 도전에 대해 논의합니다.

- **Technical Details**: 논문에서는 사회과학에서 사용되는 설문조사 방법을 응용하여, LLM의 출력이 AOV와 얼마나 일치하는지 평가합니다. 다양한 평가 방법이 사용되고 있어 결과의 일관성이 떨어지고, 평가 과정에서의 미묘한 차이를 놓칠 수 있는 위험이 있습니다. 이를 해결하기 위해 AOV의 정의, LLM에서의 AOV 연구 현황, 평가 파이프라인 사용법을 정리했습니다.

- **Performance Highlights**: 대부분의 연구는 미국 중심의 설문조사를 기반으로 LLM의 의견을 평가했으며, 많은 연구에서 LLM이 인간의 의견과 불일치하고 좌파 성향의 정치적 편향을 보였다는 점을 발견했습니다. 일부 연구는 미국 외 다른 국가 또는 다국적 비교를 수행하여, 역시 강한 편향을 관찰했습니다. 이를 통해 LLM의 출력이 특정 문화나 사회적 편향을 내포할 수 있음을 확인했습니다.

- **Practical Insights**: 이 논문은 사회과학적 조사와 LLM 평가를 통합함으로써, LLM의 인간-인공지능 정렬(human-AI alignment) 및 인문사회과학 응용에서의 개선을 위한 실질적인 통찰을 제공합니다. 또한, 평가 방법, 모델 향상, 학제 간 협력을 위한 실천적 제안을 제시함으로써, LLM 내 AOV 평가의 미래 전망을 재고합니다.



### RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning based on Emotional Information (https://arxiv.org/abs/2406.11093)
- **What's New**: 이번 논문에서는 감정 정보를 기반으로 한 in-context 학습 방법을 사용하여 도메인 간 잘못된 정보를 탐지하는 새로운 프레임워크인 RAEmoLLM을 제안합니다. 이는 감정 인식 LLM을 활용하여 감정 임베딩 데이터베이스를 구축하고, 이를 바탕으로 소스 도메인 예시를 검색하여 대상 도메인의 잘못된 정보를 탐지합니다.

- **Technical Details**: RAEmoLLM은 감정 정보 기반의 in-context 학습을 적용한 최초의 Retrieval Augmented (RAG) LLMs 프레임워크입니다. 세 가지 모듈로 구성되어 있으며, 인덱스 구축 모듈은 EmoLLaMA-chat-7B를 사용하여 도메인 코퍼스를 인코딩하여 검색 데이터베이스를 생성하고 명시적인 감정 레이블을 얻습니다. 검색 모듈은 대상 도메인 콘텐츠에 따라 소스 도메인에서 감정 관련 예시를 추천합니다. 추론 모듈은 검색된 예시를 통해 few-shot 학습을 진행하여 대상 콘텐츠의 잘못된 정보를 검증합니다.

- **Performance Highlights**: RAEmoLLM은 Fake News, Rumors, Conspiracy Theory와 같은 세 가지 잘못된 정보 벤치마크에서 평가되었습니다. 결과는 zero-shot 방법 대비 각각 20.69%, 23.94%, 39.11%의 성능 향상을 보여주었습니다. 이는 RAEmoLLM 프레임워크의 효과를 입증합니다.



### Multiple Sources are Better Than One: Incorporating External Knowledge in Low-Resource Glossing (https://arxiv.org/abs/2406.11085)
Comments:
          arXiv admin note: text overlap with arXiv:2403.08189

- **What's New**: 이 논문은 언어학 전문가들의 다양한 출처를 조정하여 저자원이 언어를 위한 자동 데이터 기반 glossing(주석)의 데이터 부족 문제를 해결합니다. 현대의 LLMs(대형 언어 모델)의 광범위한 언어 능력을 활용하고, 토큰과 문장 수준의 번역을 모델에 추가하여 성능을 향상시켰습니다.

- **Technical Details**: 우리 시스템은 번역문, 외부 사전, 그리고 LLMs를 이용하여 glossing 파이프라인을 강화합니다. 이를 통해 토큰과 문장 수준에서 추가 정보원을 제공하여 성능을 향상시킵니다. 특히 LLMs를 사용하여 post-correction(후보정) 단계에서 성능을 개선하였습니다. 이 접근법은 fine-tuning(재학습)이 필요 없어 자원이 매우 적은 설정에서도 적합합니다.

- **Performance Highlights**: 이 논문의 개선 사항은 평균 5% 포인트의 단어 수준 정확도 향상을 가져왔습니다. Gitksan과 같은 저자원 언어에서는 10% 포인트의 향상을 기록했습니다. 시뮬레이션된 초저자원 설정(100개 미만의 glossed sentences로 학습)에서는 평균적으로 10% 포인트 정확도가 향상되었습니다.



### Exploring the Limitations of Detecting Machine-Generated Tex (https://arxiv.org/abs/2406.11073)
- **What's New**: 최근 대형 언어 모델(LLM)의 텍스트 생성 품질이 크게 향상되면서, 기계 생성 텍스트를 식별하는 연구가 활발히 진행되고 있습니다. 이번 논문에서는 다양한 글쓰기 스타일을 지닌 텍스트를 대상으로 기계 생성 텍스트 검출 모델의 분류 성능을 조사했습니다. 연구 결과, 분류기는 스타일의 변화와 텍스트 복잡성에 매우 민감하며 일부 상황에서는 성능이 무작위 분류기 수준으로 저하될 수 있음을 발견했습니다. 특히, 읽기 쉬운 텍스트를 오분류하는 경향이 강하며 복잡한 텍스트에서 높은 성능을 보입니다.

- **Technical Details**: 논문에서는 두 가지 최첨단 MGT 검출 방법의 한계를 조사하고, 각 방법의 성능을 언어적 특징과 읽기 점수를 기반으로 한 샘플링 평가 데이터를 사용해 비교했습니다. 실험에는 M4, OutFox, IDMGSP 데이터셋이 사용되었으며, GPT-3.5, GPT-4, ChatGPT 등의 LLM들이 생성한 텍스트가 포함되었습니다. RoBERTa를 기반으로 한 두 가지 버전(OpenAI Detector, RoBERTa-M4)과 14개의 언어적 특징으로 학습된 LR-GLTR 모델을 사용해 분류기를 평가했습니다. 특히 명사, 동사, 부사, 고유 명사, 대상, 평균 문장 길이 등 6가지 언어적 특징과 가독율 점수(Flesch Reading Ease score)를 계산해 텍스트의 스타일을 분석했습니다.

- **Performance Highlights**: 전반적으로, 분류기는 특정 표면적 언어 특징(예: 문장 부호)에 지나치게 의존하는 경향을 보였습니다. 부사의 비율이 증가함에 따라 기계 생성 텍스트와 인간이 쓴 텍스트 모두의 분류 성능이 거의 0으로 떨어졌습니다. 또한, 명사, 동사, 부사 외의 다른 언어적 특징에서는 기계 생성 텍스트를 잘 검출하는 반면, 인간이 쓴 텍스트의 성능은 떨어지는 결과를 보였습니다. 이는 모델이 일반적인 패턴을 학습하기보다는 표면적 특징에 과적합되고 있음을 시사합니다.



### Can LLMs Understand the Implication of Emphasized Sentences in Dialogue? (https://arxiv.org/abs/2406.11065)
Comments:
          10 pages

- **What's New**: 본 논문은 Emphasized-Talk라는 새로운 벤치마크 데이터를 소개합니다. 이 데이터는 대화에서 강조된 문장의 의미를 담아낸 샘플들로 구성되어 있으며, 이를 통해 여러 대형 언어 모델(LLMs)이 강조를 얼마나 잘 이해하는지 평가합니다. 또한 GPT-4를 활용한 자동 평가 파이프라인을 제안하여 인간 평가와 높은 상관관계를 보여줍니다.

- **Technical Details**: Emphasized-Talk 데이터셋은 실제 대화를 기반으로 하여, 서로 다른 단어나 구를 강조했을 때 달라지는 함축 의미를 포착합니다. 데이터셋 구축에는 DailyTalk 데이터를 시작으로, 충분한 대화 맥락과 구체적인 단어들을 포함하는 문장을 선택하여 강조의 위치와 그 함축 의미를 인간 주석자가 결정하는 방식을 사용했습니다. 자동 평가 파이프라인에서는 GPT-4를 이용하여 모델의 성능을 평가하고, 인간 평가와 비교하여 유사성 점수를 측정합니다.

- **Performance Highlights**: 평가 결과, 상용 LLM들은 전반적으로 더 나은 성능을 보였지만, 강조된 문장의 의미를 완전히 이해하는 데는 여전히 개선의 여지가 있습니다. 오픈 소스 LLM들, 특히 Llama와 Mistral 모델들은 다양한 크기로 실험되었으며, 각각 7B에서 70B 파라미터를 사용하는 여러 버전으로 성능 차이를 분석했습니다.



### A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners (https://arxiv.org/abs/2406.11050)
Comments:
          Codes are open-sourced at this https URL

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 진정한 추론 능력을 평가하기 위한 가설 테스트 프레임워크를 소개합니다. 대부분의 LLM들이 여전히 논리적 추론에 어려움을 겪고 있으며, 클래식 문제에서 높은 성과를 보여도 이는 주로 표면적인 패턴 인식에 의존한다는 것을 통계적으로 증명하는 방향으로 연구가 진행되었습니다.

- **Technical Details**: 이 프레임워크는 합성 데이터셋 생성, 토큰 변형(token perturbation), 통계적 가설 테스트로 구성됩니다. 합성 데이터셋은 결합 오류(conjunction fallacy)와 삼단논법 문제(syllogistic fallacy) 등 논리적 오류를 다루며, 이 데이터셋을 통해 토큰 편향을 탐구합니다. 토큰 변형을 통해 LLM의 성능 변화를 관찰하여, 모델이 표면적인 패턴에 의존하는지 아니면 진정한 추론을 하고 있는지를 평가합니다.

- **Performance Highlights**: 실험 결과, 대부분의 최첨단 LLM들이 논리적 오류를 식별하는 데 성공했지만, 이는 주로 특정 토큰에 과도하게 맞춰진 경우였습니다. 통계적 보장을 통해, LLM들이 실제로는 추론 능력을 보유하고 있지 않고, 학습 데이터의 겉모습 패턴을 인식하는 데 더 의존한다는 것을 밝혀냈습니다. 결과적으로, 이러한 편향은 새로운 예제나 문구를 일반화하는 데 실패할 수 있음을 시사합니다.



### Reconsidering Sentence-Level Sign Language Translation (https://arxiv.org/abs/2406.11049)
- **What's New**: 이 연구는 기존의 문장 단위의 수어 기계 번역(task framing) 접근 방식을 검토하고, 이 방식의 한계를 탐구합니다. 특히, 처음으로 기계 학습(task framing)에서 사용되는 방식대로 인간이 수어 번역 작업을 수행한 사례 연구를 제시합니다.

- **Technical Details**: 기존의 수어 기계 번역은 연속된 이야기를 문장 수준으로 나누어 모델에 제공하는 방식이었으나, 이 연구에서는 문맥 수준의 정보가 중요하다는 점을 지적합니다. How2Sign 데이터셋을 사용하여 ASL-American Sign Language에서 영어로의 번역 작업을 수행하면서, 문장 단위의 클립만으로는 번역이 어렵다는 결과를 도출했습니다. 이 실험은 문장 단위의 클립만으로 번역을 수행할 것을 요구받은 인간 언어학자들로부터 얻은 결과입니다.

- **Performance Highlights**: 샘플 중 33%의 문장에서 문맥 수준의 추가 정보가 없이는 주요 부분을 이해할 수 없다는 결과가 나왔습니다. 또한, BLEU 점수는 문장 단위에서는 19.8점, 추가 문맥 정보가 제공되었을 때는 21.5점으로 다소 올랐습니다. 이는 수어 기계 번역 분야에서 문장 단위의 기계 번역 방식의 한계를 재고하게 만드는 결과입니다.



### Evaluating the Performance of Large Language Models via Debates (https://arxiv.org/abs/2406.11044)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)의 성능을 평가하고 비교하는 새로운 자동화된 벤치마킹 프레임워크를 제안했습니다. 이 방법은 특정 도메인에 국한되지 않고, LLM 사이의 토론(debate)을 바탕으로 성능을 평가하며, 평가자 역시 LLM을 사용합니다. 이를 통해 인간의 입력 없이도 모델의 성능을 평가할 수 있는 방법을 제시했습니다.

- **Technical Details**: 이 프레임워크에서는 미리 정해진 주제를 바탕으로 두 개의 LLM이 여러 라운드에 걸쳐 토론을 벌입니다. 첫 번째 모델은 자신의 주장을 펼치고, 두 번째 모델은 이를 반박하며 새로운 증거를 제시합니다. 이러한 과정이 일정 횟수 반복되며, 최종적으로 토론 스크립트가 평가자 LLM에게 전달되어 각 모델의 성과를 평가합니다. 평가에는 문제 정의, 불일치 인식 등 다양한 항목이 포함됩니다.

- **Performance Highlights**: 이 자동화된 벤치마킹 프레임워크는 기존의 인간 군중 소싱(human crowdsourcing)을 통한 평가 방식과 비교하여 더 높은 확장성을 가집니다. 또한, 실험 결과 이 프레임워크는 인간 입력 기반의 평가와 유사한 순위를 도출하며 신뢰성을 입증했습니다. 다양한 최신 LLM들에 대해 테스트를 통해 실제로 이 접근법이 유효함을 보였습니다.



### garak: A Framework for Security Probing Large Language Models (https://arxiv.org/abs/2406.11036)
Comments:
this https URL

- **What's New**: 이 논문에서는 'garak'라는 새로운 프레임워크를 소개합니다. garak는 대규모 언어 모델(LLMs)이나 대화 시스템의 취약점을 발견하고 식별하는 데 사용될 수 있는 프레임워크입니다. 이 프레임워크는 `LLM 보안`이라는 개념을 재정립하고, 문제 탐색과 발견을 중심으로 한 포괄적인 접근 방식을 제안합니다.

- **Technical Details**: garak는 다음의 네 가지 핵심 구성요소로 설계되었습니다: 1. Generators, 2. Probes, 3. Detectors, 그리고 4. Buffs. 이 프레임워크는 LLM 보안 평가 절차를 다룰 수 있도록 유연하게 설계되어 있습니다. 주된 기능은 모델의 잠재적 취약점을 스캔하고 알려지거나 알려지지 않은 문제를 발견하는 것입니다.

- **Performance Highlights**: garak는 대화형 설정에서 모델의 예기치 않은 행동을 유도하여 시스템의 약점을 드러내는 방식으로 동작합니다. 이는 LLM의 보안 정책 형성과 정렬 논의를 뒷받침하는 데 도움이 됩니다.



### Scaling Synthetic Logical Reasoning Datasets with Context-Sensitive Declarative Grammars (https://arxiv.org/abs/2406.11035)
- **What's New**: 이 논문은 자연어 처리에서의 논리적 추론 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 구체적으로, 절차적으로 생성된 문제에 대해 정리 증명을 모방하도록 언어 모델을 훈련시키는 방식을 사용합니다. 이전의 작업과 달리, 이 논문에서는 도메인 특화 증명 생성 알고리즘을 사용하지 않고 더욱 일반적으로 선언적(declarative)인 프레임워크를 도입하였습니다. 이를 통해 논리적 추론을 더욱 유연하게 처리할 수 있으며, 여러 언어(특히 간단한 영어와 TPTP 정리 증명 언어)를 묶어 사용하는 컨텍스트-센서티브(context-sensitive) 규칙을 사용합니다.

- **Technical Details**: 논문에서는 최대 32개의 전제(premises)와 하나의 가설(hypothesis)을 선택하여 1차 논리 문제(first-order logic problems)를 구성합니다. 생성 과정에서 의미적 제약(semantic constraints)을 사용하고, 술어(predicates)를 영어로 조심스럽게 표현함으로써 자연스러운 영어 과제에 영향을 주지 않고 논리적 추론을 향상시킵니다. DeBERTa-v3 모델을 사용하여 실험을 진행했으며, 외부 솔버(solver)를 사용하지 않고 GPT-4를 능가하는 정확도를 달성하였습니다.

- **Performance Highlights**: 제안된 방법은 FOLIO라는 인간이 작성한 논리 데이터셋에서 가장 높은 정확도(state-of-the-art accuracy)를 기록했습니다. 실험 결과, 외부 솔버를 사용하든 사용하지 않든 관계없이 GPT-4보다 12% 높은 정확도를 보였습니다.



### FoodieQA: A Multimodal Dataset for Fine-Grained Understanding of Chinese Food Cultur (https://arxiv.org/abs/2406.11030)
- **What's New**: FoodieQA는 중국의 다양한 지역 음식 문화를 세밀하게 포착한 이미지-텍스트 데이터셋입니다. 이 데이터셋은 새로운 음식 이미지와 관련 질문을 기반으로 비전-언어 모델(Vision-Language Models, VLMs)과 대형 언어 모델(Large Language Models, LLMs)의 성능을 평가합니다. LLMs는 텍스트 기반 질문-답변에서는 훌륭한 성과를 보이지만, VLMs는 여전히 인간의 정확성을 따라가지 못합니다.

- **Technical Details**: FoodieQA는 세 가지 다중 선택(Multiple-Choice) 질문-답변 과제를 포함하며, 모델이 다중 이미지, 단일 이미지, 텍스트만을 기반으로 질문에 답변해야 합니다. 데이터셋은 중국 현지인들이 제공한 14가지의 다양한 음식 이미지를 포함하고 있으며, 이 이미지는 공개되지 않은 새로운 데이터로 구성되어 기존 사전 학습 데이터에 포함될 가능성이 없습니다.

- **Performance Highlights**: LLMs는 텍스트 기반 질문-답변 과제에서 인간의 정확성을 능가하는 성과를 보였으며, 오픈 소스 VLMs는 다중 이미지 VQA 과제에서 41%, 단일 이미지 VQA 과제에서 21% 뒤처졌습니다. 반면 클로즈드 웨이트(closed-weights) 모델은 인간 성능에 10% 차이로 상당히 근접한 성과를 보였습니다.



### Curating Stopwords in Marathi: A TF-IDF Approach for Improved Text Analysis and Information Retrieva (https://arxiv.org/abs/2406.11029)
Comments:
          Accepted at I2CT 2024

- **What's New**: 이번 연구에서는 인도의 저자원 언어 중 하나인 마라티어(Marathi)를 대상으로 한 스톱워드(stopwords) 리스트를 새롭게 큐레이션했습니다. 마라티어는 많은 인구가 사용하지만, 그동안 잘 정제된 컴퓨터 자원이 부족했습니다. 이번 연구는 마하코퍼스(MahaCorpus)라는 2,480만 개의 문장을 포함한 대규모 마라티어 코퍼스를 사용하여 400개의 스톱워드 리스트를 구축했습니다.

- **Technical Details**: 우리는 TF-IDF(Term Frequency-Inverse Document Frequency) 접근법을 활용하여 스톱워드를 식별하고 순위를 매겼습니다. 이후 인간 평가를 통해 최종적으로 400개의 스톱워드를 추렸습니다. 이 과정을 통해 만든 스톱워드 리스트는 마라티어 NLP 라이브러리인 mahaNLP에 통합되어 공개되었습니다. TF-IDF는 특정 문서 내에서 단어의 중요성을 평가하고, 그 단어가 전체 코퍼스에서 얼마나 흔한지를 반영하여 단어의 가치를 결정하는 방법입니다.

- **Performance Highlights**: 스톱워드 필터링을 L3Cube-MahaNews 텍스트 분류 작업에 적용한 결과, 다운스트림 정확도에 거의 영향을 미치지 않음을 확인했습니다. 이는 우리 접근법이 실제 텍스트 분석에 효과적임을 보여줍니다. 이러한 결론은 인간 평가와 실제 분류 작업에서의 성능 평가를 통해 검증되었습니다.



### Universal Cross-Lingual Text Classification (https://arxiv.org/abs/2406.11028)
Comments:
          Accepted at I2CT 2024

- **What's New**: 이번 연구에서는 통합된 모델을 통해 여러 언어에서의 텍스트 분류를 수행하는 새로운 접근 방식을 제안합니다. 저자들은 낮은 자원 언어 (low-resource languages)에서의 레이블 범위와 언어 범위를 확장하기 위해 다국어 SBERT(Sentence-BERT) 기반 모델을 사용하여 훈련 중에 다양한 언어의 감시 데이터를 통합했습니다. 이를 통해 낮은 자원 언어에서도 텍스트를 분류할 수 있는 보편적인 모델을 개발하고자 했습니다.

- **Technical Details**: 기본 모델로 다국어 SBERT(Sentence-BERT)를 사용하며, 새로운 훈련 전략을 통해 다양한 언어의 데이터를 병합하여 보편적인 모델을 구축했습니다. 이 모델은 훈련 중 사용된 데이터의 레이블(레이블) 세트를 넓히고 적응성을 높이고자 했습니다. 평가 과정에서는 특정 언어에 한정되지 않고 다양한 언어에서의 성능을 테스트하여 모델의 보편성을 입증했습니다.

- **Performance Highlights**: 본 연구에서는 다양한 BERT 및 SBERT 모델을 사용하여 실험을 수행했습니다. 초기 실험에서는 다국어 SBERT 모델의 역량을 평가하였고, 이를 통해 마라티어와 같은 미지의 언어에서도 높은 성능을 보임을 확인했습니다. 후속 실험에서는 다양한 언어의 레이블 데이터를 혼합하여 훈련함으로써 더 넓은 레이블 범위를 지원하는 보편 모델을 성공적으로 구축했습니다. 이를 통해 낮은 자원 언어에서도 효과적인 텍스트 분류가 가능함을 확인했습니다.



### RUPBench: Benchmarking Reasoning Under Perturbations for Robustness Evaluation in Large Language Models (https://arxiv.org/abs/2406.11020)
Comments:
          In submission; Data and code are available at: this https URL

- **What's New**: RUPBench라는 종합적인 벤치마크를 소개합니다. 이는 대형 언어 모델(LLM)의 강인함을 다양한 추론 작업에서 평가하기 위해 설계되었습니다. 이 벤치마크는 공통 상식, 산술, 논리 및 지식 집중 추론 4개 카테고리에서 15개의 데이터셋을 포함하며, 9가지 텍스트 변형을 도입하여 LLM의 성능을 체계적으로 분석합니다.

- **Technical Details**: RUPBench는 15개의 추론 데이터셋을 사용하여 LLM의 강인성을 평가합니다. 텍스트 변형은 어휘적(lexical), 구문적(syntactic), 의미적(semantic) 수준에서 이루어지며, 각 데이터셋은 다양한 형태의 텍스트 변형을 통해 실세계 입력 변화를 시뮬레이션 합니다. 주요 LLM 모델인 GPT-4o, Llama3, Phi-3, Gemma 모델을 대상으로 실험을 진행하여 원본과 변형된 데이터셋에서의 성능을 비교 분석합니다.

- **Performance Highlights**: 실험 결과, 대형 모델일수록 텍스트 변형에 대한 강인함이 더 높아지는 경향이 나타났습니다. LLM이 다양한 유형의 텍스트 변형에서 어떤 일반적인 오류 패턴을 보이는지 확인했고, 이러한 패턴을 통해 LLM의 개선 필요 영역을 식별할 수 있었습니다. 특히 문맥 오해, 지식 격차 등의 문제에서 개선이 필요함을 밝혔습니다.



### Connecting the Dots: Evaluating Abstract Reasoning Capabilities of LLMs Using the New York Times Connections Word Gam (https://arxiv.org/abs/2406.11012)
- **What's New**: 최근 뉴욕타임스의 Connections 게임이 단어 퍼즐 애호가들 사이에서 유행하고 있습니다. 이 논문은 200개의 Connections 게임을 수집하여, 인간 플레이어와 대규모 언어 모델(LLMs)의 성능을 평가했습니다. 결과에 따르면 최고 성능의 LLM인 GPT-4o도 게임의 8%만 완벽히 해결할 수 있었습니다. 이는 초보자와 전문가 인간 플레이어가 GPT-4o보다 훨씬 우수한 성능을 보였다는 것을 보여줍니다.

- **Technical Details**: Connections 게임은 4x4 격자에 16개의 단어를 제공하며, 플레이어는 각각의 단어 그룹을 공유된 특징을 통해 네 개의 고유한 그룹으로 분류해야 합니다. 이 연구는 Google의 Gemini 1.5, Anthropic의 Claude 3, OpenAI의 GPT-4o, Meta의 Llama 3 70B와 같은 최첨단 LLM을 비교했습니다. 모든 LLM이 부분적으로 게임을 해결할 수 있지만, 그 성능은 이상적이지 않습니다. 특히, GPT-4o는 few-shot 및 chain-of-thought 프롬프팅을 통해서도 게임의 8%만 해결할 수 있었습니다.

- **Performance Highlights**: 노련한 인간 플레이어는 정확하게 게임을 해결하는 데 있어 GPT-4o를 상당히 능가했습니다. 초보 플레이어도 GPT-4o보다 약간 더 우수한 성능을 보였습니다. 실험 결과 LLM이语義知識(semantic knowledge)에는 능하지만, 연관적(associative), 백과사전적(encyclopedic), 다단어 표현(multi-word expressions)과 같은 다른 유형의 지식에서는 어려움을 겪는다는 것을 보여줍니다.



### Not All Bias is Bad: Balancing Rational Deviations and Cognitive Biases in Large Language Model Reasoning (https://arxiv.org/abs/2406.10999)
Comments:
          This article is currently under review. All data will be made publicly available on GitHub once the review is complete

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 의사 결정 과정에서의 편향의 역할을 다룹니다. 기존 연구는 모든 편향을 제거하는 데 초점을 맞추었지만, 본 연구는 모든 편향이 해롭지 않다는 것을 밝혀내었습니다. 합리적 편차(rational deviations)를 통해 효율적인 의사 결정을 가능하게 하는 휴리스틱(heuristic) 숏컷을 강조하며, 적절히 균형을 맞춘다면 이들이 도움이 될 수 있음을 보여줍니다. 우리는 휴리스틱 완화(heuristic moderation)와 불확실한 경우에 답변을 보류할 수 있는 옵션을 도입하여, 오류율을 줄이고 의사 결정 정확도를 개선합니다.

- **Technical Details**: 본 연구는 새로운 BRD(Balance Rational Deviations) 데이터셋을 활용하여 대형 언어 모델의 편향을 적절히 조정하면 성능이 개선될 수 있음을 보여줍니다. 휴리스틱 완화(heuristic moderation)와 '답변 보류(abstention)' 옵션을 도입해 LLM이 불확실한 경우 답변을 보류할 수 있도록 하여 오류율을 줄입니다. 다양한 지표로 평가된 이 접근법은 LLM의 의사 결정 프로세스를 인간의 추론 및 판단과 더욱 유사하게 만듭니다.

- **Performance Highlights**: 연구 결과, 편향 검사 및 답변 보류 옵션을 조정하면 모델 성능이 향상되고, 의사 결정 정확도가 높아졌습니다. 이는 LLM의 신뢰성을 높이고 각종 실전 응용 분야에서의 실용성을 더욱 강화할 전략을 제시합니다.



### THEANINE: Revisiting Memory Management in Long-term Conversations with Timeline-augmented Response Generation (https://arxiv.org/abs/2406.10996)
Comments:
          Under Review

- **What's New**: LLM 기반의 대화 시스템에서 과거 대화 정보를 더욱 정확히 기억하고 활용하기 위한 새로운 프레임워크 Theanine이 소개되었습니다. 이 프레임워크는 기존 메모리를 버리는 대신, 과거 사건들의 흐름과 인과 관계를 파악하기 위해 '메모리 타임라인(memory timelines)'을 추가합니다. 또한 오랜 대화 속 문맥을 더욱 효과적으로 대처하기 위해 TeaFarm이라는 반사실적(反事實的, counterfactual-driven) 질문-응답 파이프라인을 도입했습니다.

- **Technical Details**: Theanine은 그래프 구조의 메모리 관리 시스템을 통해 과거와 현재 대화를 연결합니다. 메모리 그래프(G)은 대화에서 요약된 메모리(V)와 이들 간의 시간적, 인과적 관계를 나타내는 엣지(E)로 구성됩니다. LLM을 활용해 메모리 간의 관계를 동적으로 연결하고, 새로운 응답을 생성할 때, 관련 사건의 메모리 타임라인을 검색합니다. 이를 통해 현재 대화에 맞춰 정제된 정보를 제공합니다. 또한 'chain-of-thought (CoT)' 추론 기능을 사용해 최종 응답을 생성합니다.

- **Performance Highlights**: Theanine은 LLM 기반 평가와 인간 평가에서 더 상세하고 비공식적인 응답을 생성하는 데 탁월한 성과를 보였습니다. 또한, TeaFarm의 질문-응답 파이프라인을 통해 TeaBag 데이터셋을 평가하여 오래된 대화 세션에 대한 참조를 더욱 효율적으로 처리합니다. 이 시스템은 대화 세션 축적 시 메모리 검색 품질의 저하 문제를 감소시킵니다.



### CoSTA: Code-Switched Speech Translation using Aligned Speech-Text Interleaving (https://arxiv.org/abs/2406.10993)
- **What's New**: 이 논문은 인도의 다국어 사회에서 빈번히 발생하는 코드-스위치(Code-switching)된 음성을 영어 텍스트로 번역하는 새로운 끝-끝(end-to-end) 모델 아키텍처인 CoSTA를 소개합니다. 코드-스위칭된 음성을 위한 새로운 평가 벤치마크도 영상합니다. 코드-스위칭이란 여러 언어를 혼합하여 말하는 현상을 의미합니다. 저자들은 이 모델이 경쟁력 있는 다중 모드(multimodal) 베이스라인을 최대 3.5 BLEU 포인트 만큼 능가한다고 주장합니다.

- **Technical Details**: CoSTA는 사전 학습된 자동 음성 인식(ASR)과 기계 번역(MT) 모듈에 기초하며, 말과 ASR 텍스트 표현을 정렬된 인터리빙(aligned interleaving) 방식으로 결합합니다. 이 합성된 표현은 사전 학습된 MT 모듈에 입력으로 들어가고, 해당 파이프라인 전체가 ST 목표로 훈련됩니다. 또한, 인도 언어인 벵골어-영어, 힌디어-영어, 마라티어-영어, 텔루구어-영어 코드-스위칭된 음성 데이터를 새롭게 평가 벤치마크로 공개합니다.

- **Performance Highlights**: CoSTA 모델은 소량의 30시간 합성 ST 데이터로 훈련되었음에도 불구하고, 대규모 데이터셋으로 훈련된 기존의 최첨단(end-to-end) 베이스라인 모델들을 능가하는 BLEU 점수를 기록합니다. 특히, 코드-스위칭에 강한 로버스트성을 보여주며, 문장 내 코드-스위칭이 증가함에도 성능 저하가 거의 없다는 것이 특징입니다.



### Adaptive Query Rewriting: Aligning Rewriters through Marginal Probability of Conversational Answers (https://arxiv.org/abs/2406.10991)
- **What's New**: 최소한의 어노테이션 데이터만으로 효과적인 쿼리 리라이팅 모델을 학습하기 위한 새로운 접근법인 AdaQR(Adaptive Query Rewriting)을 제안합니다. 이 프레임워크는 시드 데이터셋(seed datasets)에서 리라이팅 어노테이션의 약 10%만 사용하고, 패시지 레이블 없이 쿼리 리워드를 최적화합니다.

- **Technical Details**: AdaQR는 대형 언어 모델(large language models)을 미세 조정하여(seed dataset의 약 10% 어노테이션 사용) 리라이팅 후보를 생성합니다. 그런 다음 각 쿼리 인스턴스에 대해 후보를 생성하는데, 리트리버의 선호도를 평가하기 위해 회답의 조건부 확률을 계산하고 이를 직접 선호도 최적화(Direct Preference Optimization, DPO)에 사용합니다. 이는 리라이팅 및 리트리버 어노테이션이 없는 상태로 수행됩니다.

- **Performance Highlights**: AdaQR는 4개의 오픈 도메인 CQA 데이터셋에서 실험한 결과, 기존의 쿼리 리라이팅 기법보다 더 적은 어노테이션으로도 높은 성능을 발휘하였습니다. 특히, 도메인 외 데이터셋(out-of-domain datasets)에서의 적응력도 우수한 것으로 나타났습니다.



### Taking a Deep Breath: Enhancing Language Modeling of Large Language Models with Sentinel Tokens (https://arxiv.org/abs/2406.10985)
- **What's New**: 최근 연구는 대형 언어 모델(LLM)의 성능 저하를 해결하기 위해 새로운 방법을 제안합니다. 이 방법은 <SR>라는 특수 토큰을 도입하여 텍스트의 각 청크(chunk) 끝에 삽입합니다. 이를 통해 모델이 개별 토큰뿐만 아니라 청크의 전체 정보를 집계할 수 있도록 합니다.

- **Technical Details**: 이 방법은 텍스트를 청크로 분할하고 각 청크 끝에 <SR> 토큰을 삽입합니다. 주의(attention) 마스크를 수정하여 <SR> 토큰이 해당 청크의 내용을 통합할 수 있게 합니다. 이로 인해 LLM이 다음 토큰을 생성할 때 청크의 전체 의미 정보를 활용할 수 있습니다. 실험은 1.3B에서 13B 크기의 모델을 사용하여 Wikitext-2 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 이 방법은 다양한 모델에서 퍼플렉시티(perplexity) 감소를 통해 성능 향상을 입증했습니다. OPT 모델과 RedPajama-3B 모델에서는 약 10%의 퍼플렉시티 감소를 보였고, Llama 및 Mistral-7B 모델에서도 약 3.5%의 감소를 기록했습니다. 이는 제안한 방법이 LLM이 다음 토큰 예측 시 다각적인 정보를 활용하는 데 효과적임을 시사합니다.



### Revisiting Cosine Similarity via Normalized ICA-transformed Embeddings (https://arxiv.org/abs/2406.10984)
- **What's New**: 이번 연구는 코사인 유사성(cosine similarity)을 독립성분분석(ICA)을 통해 변환된 임베딩을 통해 해석하는 새로운 방법을 제안합니다. 연구팀은 코사인 유사성을 축(axis) 별 의미적 유사성의 합으로 해석할 수 있다는 새로운 관점을 제시했습니다.

- **Technical Details**: 먼저 실험을 통해 정규화되지 않은 임베딩에 노름(norm)으로 인한 인공물이 포함되어 있음을 보여주었습니다. 이후 정규화된 ICA 변환 임베딩이 축별로 몇 가지 큰 값을 가지며 스파스(sparsity)한 특성을 나타냄을 확인했습니다. 이는 의미적 기여를 명확히 구분함으로써 해석 가능성을 높입니다. 최종적으로 이를 검증하기 위해 특정 의미 구성 요소가 있는지 없는지를 고려한 이상적인 임베딩을 사용한 검색 실험을 수행했습니다.

- **Performance Highlights**: ICA 변환 임베딩은 축에 대한 의미적 해석이 가능하다는 점에서, 코사인 유사성을 의미적 유사성의 합으로 해석할 수 있습니다. 실험을 통해 정규화된 ICA 변환 임베딩이 특정 축에 대해 큰 값을 가지며, PCA 변환 임베딩보다 더 해석 가능함을 보여주었습니다.



### Toward Optimal LLM Alignments Using Two-Player Games (https://arxiv.org/abs/2406.10977)
Comments:
          Our code is released at this https URL

- **What's New**: 이번 연구는 기존의 'Reinforcement Learning from Human Feedback (RLHF)' 방법론의 한계를 보완하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 두 에이전트 간 반복적인 상호작용을 통해 모델의 성능을 최적화하는 접근법을 채택합니다. 특히, 방어 에이전트(defensive agent)의 약점을 찾아내기 위한 적대적 에이전트(adversarial agent)를 도입하여 보다 포괄적인 프롬프트를 생성합니다.

- **Technical Details**: 이 연구에서는 두 에이전트 간의 게임 형태로 학습이 이루어집니다. 적대적 에이전트는 방어 에이전트의 약점을 드러내기 위한 프롬프트를 생성하고, 방어 에이전트는 이러한 프롬프트에 대처하는 방법을 향상시키는 역할을 합니다. 이 과정은 반복적으로 진행되어, 결국 두 에이전트가 Nash Equilibrium에 도달하는 것을 이론적으로 증명했습니다. 또한, BLEU 점수와 문장 임베딩을 이용한 다양성 제약을 통합하여 적대적 에이전트가 제한된 프롬프트로 수렴하는 것을 방지했습니다.

- **Performance Highlights**: 실험 결과, 이러한 경쟁적 환경에서의 학습은 에이전트의 일반화 능력을 크게 향상시키는 것으로 나타났습니다. 특히, 안전 시나리오에서의 실험에서, 적대적 에이전트가 지속적으로 도전적인 프롬프트를 생성함으로써 방어 에이전트가 보다 철저히 학습되는 것을 확인했습니다.



### Towards Supporting Legal Argumentation with NLP: Is More Data Really All You Need? (https://arxiv.org/abs/2406.10974)
- **What's New**: 본 연구는 AI와 법 분야의 전통적인 상징적 방법과 최신 법률 NLP(자연어 처리) 접근법을 검토하고, 두 방법을 통합하여 확장성과 설명 가능성 사이의 균형을 맞추는 방법을 제안합니다. 특히, 전문가의 지식을 통합한 현대 NLP 모델 및 방법의 잠재력을 탐구하고 있으며, 법률 reasoning(추론)과 argumentation(논증)의 모델링을 강조합니다.

- **Technical Details**: 법률 추론은 원칙적으로 IF-THEN(조건-결과) 형태의 추론과 유사합니다. 법률 규칙은 법규, 규제, 판례 등을 기반으로 하며 특정 상황에서 사실 요구를 충족하면 특정 결과가 따른다고 명시합니다. 그러나 현실에서는 모호성, 애매함, 인간의 재량이 혼재되어 있어 완전히 논리적인 추론이 어렵습니다. 본 논문은 전통적인 상징적 AI&Law 접근법과 최신 데이터 기반 접근법의 비교를 수행하고, 법률 시스템의 변화와 대규모 언어 모델(LLMs)의 도입이 가져오는 지식 획득의 어려움에 대해 논의합니다.

- **Performance Highlights**: 데이터 기반 법률 AI 접근법은 대개 명시적인 법률 reasoning(추론)을 따라가지 않고 법률 결론을 예측하는 경향이 있어 설명 가능성이 떨어집니다. 논문은 기존의 상징적 접근법이 값비싼 법률 전문 지식을 필요로 하지만 정확하고 신뢰할 수 있는 모델을 제공한다고 보고 있습니다. 반면, 현대 NLP 모델은 확장성은 뛰어나지만 설명 가능성이 부족하다는 점을 지적합니다. 따라서, 두 접근법의 장점을 결합하여 더 나은 법률 AI 시스템을 구축하는 방향을 제안합니다.



### DocNet: Semantic Structure in Inductive Bias Detection Models (https://arxiv.org/abs/2406.10965)
Comments:
          Under submission with EMNLP 2024

- **What's New**: 문서의 의미 구조 분석을 활용한 새로운 편향 탐지 모델, DocNet을 소개합니다. 이는 기존 대규모 언어 모델보다 우수한 성능을 보이며, 저자원 환경에서도 편향 탐지에 유용합니다. 또한, 상반되는 정치적 성향의 뉴스 기사 간에도 의미적 구조가 상당히 유사하다는 점을 발견했습니다.

- **Technical Details**: DocNet은 inductive(귀납적) 방식의 그래프 기반 편향 탐지 모델입니다. 뉴스 기사마다 단어 공출현 그래프(word co-occurrence graph)를 생성하고, 이를 통해 각 기사의 의미 구조를 분석합니다. 이러한 그래프 임베딩(graph embedding)은 대규모 언어 모델의 임베딩과 비교해도 편향을 효과적으로 캡처할 수 있습니다. 모델 훈련 파이프라인은 데이터셋 수집, 그래프 생성, 비지도 임베딩 생성, 지도 학습 모델 훈련의 네 가지 절차로 이루어집니다.

- **Performance Highlights**: DocNet은 기존의 LLM(대규모 언어 모델)이 생성한 편향 예측보다 통계적으로 우수한 성능을 보였습니다. 이는 미리 훈련된 언어 모델에 의존하지 않고도 높은 효율성을 보이는 새로운 방법론입니다.



### ESCoT: Towards Interpretable Emotional Support Dialogue Systems (https://arxiv.org/abs/2406.10960)
Comments:
          Accepted to ACL 2024 (Long Paper)

- **What's New**: 최신 연구에서는 감정 중심(emotion-focused) 및 전략 주도(strategy-driven) 체인의 사고(Chain-of-Thought, CoT)를 모사하는 감정 지원 응답 생성 방식인 ESCoT를 소개합니다. 이는 사용자의 감정을 식별하고 이해하며 조절하는 과정을 모방하여 대화 시스템에 더 나은 해석 가능성을 부여하고자 한 것입니다. 이 접근 방식은 새로운 데이터셋 생성과 향상된 대화 응답 생성을 위한 모델 개발을 포함합니다.

- **Technical Details**: ESCoT는 두 가지 주요 단계를 포함합니다. 첫째, 대화 생성(Dialogue Generation) 단계에서는 다양한 대화 상황을 생성한 후, 이러한 상황을 기반으로 더 풍부한 감정 지원 전략을 사용하여 대화 생성을 향상시킵니다. 둘째, 체인 보완(Chain Supplement) 단계에서는 선택된 대화에 감정, 자극, 평가, 전략 이유 등의 요소를 보완하여 수작업으로 검증된 체인을 형성합니다. 또한, 사전 훈련된 언어 모델을 감독 학습(fine-tuning)하여 더 나은 해석 가능한 대화 응답 생성 모델을 개발합니다.

- **Performance Highlights**: ESCoT 및 생성된 대화 응답의 효과성을 검증하기 위해 광범위한 실험 및 인간 평가를 실시하였습니다. 이에 따라 감정 지원 응답 생성의 해석 가능성을 높이고, 구축된 대화 데이터셋의 품질을 확인할 수 있었습니다. 이 접근 방식은 향후 연구를 위한 강력한 기준을 제공하며, 데이터와 코드는 공개되어 접근할 수 있습니다.



### Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergenc (https://arxiv.org/abs/2406.10957)
- **What's New**: 최근 Direct Preference Optimization (DPO)는 인간의 선호도에 직접적으로 맞춰 Large Language Models (LLMs)를 최적화하는 뛰어난 알고리즘으로 주목받고 있습니다. 이는 복잡한 인적 피드백 기반 강화 학습(Reinforcement Learning from Human Feedback, RLHF)의 대안으로 자리 잡았으나 'verbosity(장황함)' 문제를 다룰 필요가 있었습니다. 본 논문에서는 DPO의 장황함 문제가 데이터 내 편향된 라벨들뿐 아니라 알고리즘 자체의 길이 의존성에서 기인한다고 주장합니다.

- **Technical Details**: 기존에는 DPO가 선택된 시퀀스와 거부된 시퀀스 간의 Kullback-Leibler (KL) 발산 차이에 기반하여 보상을 과대 또는 과소 평가한다고 알려져 있었습니다. 본 연구는 이러한 현상이 시퀀스의 길이 차이에 기인하며, 이를 해결하기 위해 SamPO라는 다운샘플링 접근법을 제안합니다. SamPO는 각 토큰 수준에서 동일한 확률 특징을 다운샘플링하여 KL 발산을 정규화합니다.

- **Performance Highlights**: 세 가지 규모의 LLM을 사용한 실험 평가를 통해 SamPO가 장황함을 효과적으로 완화하고, 조건부 및 개방형 벤치마크에서 DPO 대비 5%에서 12%까지 성능 향상을 이룬 것을 확인했습니다. 이를 통해 편향된 보상을 제거하고 보다 신뢰성 있는 최적화를 달성했습니다. 또한 필요한 코드는 공개되어 있습니다.



### Avoiding Copyright Infringement via Machine Unlearning (https://arxiv.org/abs/2406.10952)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 저작권이 있는 자료를 학습하고 생성하는 과정에서 발생하는 법적 및 윤리적 문제를 해결하기 위해, 모델 소유자가 다양한 시간 단계에서 저작권이 있는 콘텐츠를 '잊도록' 할 수 있는 능력이 중요하다는 점을 강조합니다. 이를 위해 저자들은 여러 시간 단계에 걸쳐 저작권이 있는 콘텐츠를 제거하는 '순차적 망각' 설정을 탐구합니다. '안정적인 순차적 망각(Stable Sequential Unlearning, SSU)'이라는 새로운 프레임워크를 제안하여, 태스크 벡터(task vectors)와 그래디언트 기반의 가중치 살리언시 매핑(weight saliency mapping)을 통해 저작권이 있는 콘텐츠를 제거하면서 모델의 일반 지식을 유지하려고 합니다.

- **Technical Details**: SSU는 저작권이 있는 자료를 제거하기 위해 모델을 파인 튜닝(fine-tuning)하고, 임의의 라벨 손실(random labeling loss) 항을 추가하여 안정성을 강화하고 가중치 살리언시 매핑을 적용하여 로컬리티를 유지하도록 설계되었습니다. 기존의 그래디언트 어센트(Gradient Ascent, GA) 기반 방법이 모델의 추론 능력을 크게 손상시키는 것과 달리, SSU는 안정적인 모델 성능을 유지합니다. SSU는 내부 모델 메커니즘과 손실 함수를 활용하여 다른 태스크에서도 성능을 유지하면서 저작권이 있는 내용을 잊도록 합니다.

- **Performance Highlights**: 저자들은 Llama3-8B 모델을 사용해 저작권이 있는 책들을 순차적으로 잊도록 실험을 수행했으며, SSU가 기존의 방법들보다 망각 효율성과 모델 로컬리티 유지의 더 나은 균형을 제공한다는 점을 입증했습니다. SSU는 망각 과정 중 발생하는 불안정을 완화하고 저작권 침해를 방지하면서 모델의 추론 능력을 보존하는 데 효과적입니다.



### E-Bench: Towards Evaluating the Ease-of-Use of Large Language Models (https://arxiv.org/abs/2406.10950)
- **What's New**: 최근 연구에서 대규모 언어 모델(LLMs)의 안정성 평가를 위해 E-Bench라는 벤치마크를 제안했습니다. 이는 실제 사용 상황에서 동의어 교란(가운데 재구성, 단순화, 구어체)과 타이포그래피 교란의 영향을 시뮬레이트합니다.

- **Technical Details**: E-Bench는 동의어 및 타이포그래피 오류에 대한 모델의 성능 변화를 평가합니다. 먼저 기존 AlpacaEval 셋을 사용하여 데이터를 네 가지 유형으로 분할하고, 이를 재구성합니다. 각 프롬프트를 자동 도구로 변형한 후, 유사한 의미를 유지하는지 확인합니다. 성능 저하를 평가 척도로 사용하여 모델의 사용자 편의성을 측정합니다.

- **Performance Highlights**: 6개의 대표적인 LLM (Llama2-chat, Vicuna, GPT) 모델을 실험한 결과, 프롬프트 교란 후 모든 모델에서 성능 저하가 발생했습니다. 특히, 더 큰 모델이 동의어 교란에 잘 대응하지만, 타이포그래피 오류에서는 모델 크기와 성능 저하 간 명확한 상관관계가 발견되지 않았습니다.



### Generating Tables from the Parametric Knowledge of Language Models (https://arxiv.org/abs/2406.10922)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 파라메트릭 지식을 활용하여 사실적이고 정확한 테이블을 생성하는 방법을 탐구합니다. 이는 주로 금융이나 건강관리 분야에서 매우 중요한 구조화된 데이터 생성을 목표로 합니다.

- **Technical Details**: 논문에서는 GPT-3.5, GPT-4, Llama2-13B, Llama2-70B와 같은 최신 LLM 네 가지를 평가 대상으로 설정하고, 테이블 생성을 위해 세 가지 프롬프트 방법을 사용했습니다: (a) 전체 테이블 생성 (full-table), (b) 행별 생성 (row-by-row), (c) 셀별 생성 (cell-by-cell). 이를 평가하기 위해 WikiTabGen이라는 새로운 벤치마크를 도입하였으며, 이 벤치마크는 100개의 Wikipedia 테이블을 포함하고 있습니다.

- **Performance Highlights**: 테이블 생성은 여전히 어려운 과제로 남아 있으며, GPT-4가 가장 높은 정확도인 19.6%를 달성했습니다. 추가 분석을 통해 테이블의 크기, 인기, 수치적 내용 등 다양한 특성이 생성 성능에 미치는 영향을 밝혔습니다.



### MICL: Improving In-Context Learning through Multiple-Label Words in Demonstration (https://arxiv.org/abs/2406.10908)
Comments:
          13 pages, 7 figures

- **What's New**: 최근 발표된 논문에서는 In-context learning (ICL)의 성능을 향상시키기 위해 다중 레이블 단어(multipe label words) 사용을 제안했습니다. 이는 기존 연구들이 테스크 데모에서 단일 클래스 이름을 레이블 단어로 사용했던 것에서 벗어나, 보다 다양한 레이블 정보를 제공하여 ICL 성능을 극대화하려는 접근입니다.

- **Technical Details**: 논문에서는 새로운 알고리즘 MICL를 통해 샘플-레이블(pairing samples and labels) 쌍을 만들 때 다중 레이블 단어를 선택하고 정렬하는 방법을 설명합니다. 이 과정은 다음과 같이 이루어집니다: 먼저, 큰 지식 기반에서 관련 레이블 단어들을 필터링하고 zero-shot 학습을 통해 트레이닝 샘플에 대한 LLM의 출력 분포(logit)를 획득합니다. 이후, logit 값이 가장 높은 단어를 기준으로 초기화하고 추가 레이블 단어를 선택합니다. 이들의 최적화된 숫자와 순서에 따라 샘플-레이블 쌍을 구성합니다.

- **Performance Highlights**: 논문은 일곱 개의 분류 데이터셋에서 수행된 실험 결과를 통해 MICL의 효과를 증명합니다. 다중 레이블 단어가 포함된 데모는 ICL 성능을 높이는 데 중요한 역할을 하며, 이는 다양한 레이블 정보로 인한 명확성 향상과 모호성 감소 덕분입니다.



### RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models (https://arxiv.org/abs/2406.10890)
Comments:
          48 pages, 7 figures, 12 tables

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)에서 민감한 정보나 저작권이 있는 데이터를 삭제하는 효율적인 방법을 제안합니다. 이를 위해 실제 환경에서의 지식 삭제 벤치마크인 RWKU(Real-World Knowledge Unlearning benchmark)를 도입하였습니다.

- **Technical Details**: RWKU는 세 가지 중요한 요소를 기반으로 설계되었습니다. 첫째, 현실적인 작업 설정을 고려해 삭제할 데이터나 유지할 데이터를 제공하지 않고도 작업을 수행합니다. 둘째, 위키피디아에 있는 유명 인물 200명을 삭제 대상으로 선택했으며, 이는 다양한 LLM에서 쉽게 찾아볼 수 있는 지식을 기반으로 합니다. 셋째, 평가 프레임워크를 통해 모델의 지식 삭제 후 성능을 다양한 실제 응용 프로그램에서 테스트합니다. 구체적으로는, 네 가지 'membership inference attack (MIA)' 방법과 아홉 종류의 적대적 공격을 통해 삭제된 지식이 정말로 제거되었는지 평가하고, 모델의 유틸리티를 유지하며 지역성을 평가합니다.

- **Performance Highlights**: 실험 결과, 다음과 같은 주요 발견 사항이 있습니다: (1) 모델은 질문-응답 프로브(Question-Answer Probes)보다는 적대적 공격(Adversarial Attack Probes)과 빈칸 채우기 프로브(Fill-in-the-Blank Probes)에 더 취약합니다. (2) 목표 지식을 삭제하면서 인접 지식에도 부정적인 영향을 미치는 등 균형을 맞추기가 어렵습니다. (3) 일괄 삭제(Batch-Target Unlearning)는 단일 삭제(Single-Target Unlearning)보다 훨씬 어렵고, 모델 붕괴를 초래할 수 있습니다. (4) 고전적 기법인 'Gradient Ascent', 최근의 'Negative Preference Optimization', '인 컨텍스트 삭제(In-Context Unlearning)' 방법이 상대적으로 좋은 성능을 보였습니다.



### Distilling Opinions at Scale: Incremental Opinion Summarization using XL-OPSUMM (https://arxiv.org/abs/2406.10886)
- **What's New**: 이 논문에서는 다량의 이커머스 리뷰를 처리하여 효율적으로 의견 요약(Opinion Summarization)을 생성하는 새로운 프레임워크, Xl-OpSumm을 제안합니다. 이를 위해 저자들은 Flipkart 웹사이트에서 데이터를 수집하여 Xl-Flipkart 라는 대규모 테스트 세트를 생성했습니다. 이를 통해 새로운 리뷰가 추가될 때마다 점진적으로 요약을 업데이트할 수 있습니다.

- **Technical Details**: Xl-OpSumm 프레임워크는 리뷰들을 셋으로 나누어 각 셋에 대해 Local Summary와 Global Summary를 생성합니다. 추가적으로 Aspect Dictionary라는 요소를 도입하여 각 셋에서 추출한 의견을 기반으로 긍정, 부정, 중립적인 감정을 기록합니다. 리뷰의 각 셋에 대해 ABSA (Aspect-Based Sentiment Analyser) 모델을 사용하여 감정을 분석하고 Sentence Transformer를 이용해 유사한 측면들을 병합합니다. Global Summary는 모든 이전 셋의 요약을 포함하며, 이는 각 셋마다 갱신됩니다.

- **Performance Highlights**: 이 프레임워크는 LLAMA-3-8B-8k 모델을 사용하여 다른 모델들에 비해 평균 ROUGE-1 F1 점수가 4.38% 향상되고 ROUGE-L F1 점수가 3.70% 향상되었습니다. 이는 기존의 접근 방식보다 훨씬 더 많은 리뷰를 효과적으로 처리할 수 있음을 보여줍니다.



### On the Role of Entity and Event Level Conceptualization in Generalizable Reasoning: A Survey of Tasks, Methods, Applications, and Future Directions (https://arxiv.org/abs/2406.10885)
- **What's New**: 이 논문은 개념화(Conceptualization)에 대한 최초의 종합적인 설문조사를 제공합니다. 특히 엔티티(Entity)와 이벤트(Event) 수준의 개념화에 중점을 두며, 150편 이상의 논문을 검토하고 다양한 정의, 자원, 방법 및 다운스트림 응용 프로그램을 통합된 분류 체계로 정리합니다. 또한, 이 분야의 잠재적 미래 방향을 제시합니다.

- **Technical Details**: 개념화는 공통 속성을 가진 특정 인스턴스를 포괄하는 하나의 개념으로 통합하는 과정입니다. 이 논문은 개념화를 엔티티 수준, 이벤트 수준, 문서 수준, 시스템 수준으로 분류하며, 각 수준별로 포괄적인 정의를 제공합니다. 개념화 획득 방법은 추출 기반, 검색 기반, 생성 기반 방법으로 나누어 설명됩니다.

- **Performance Highlights**: 개념화는 상식 추론, 인과 추론, 물리적 추론 등 다양한 추론 작업에서 모델의 추론 능력을 향상시키고, 여러 도메인에 걸쳐 지식을 효과적으로 전이할 수 있게 합니다. 논문은 개념화가 다운스트림 작업에 어떻게 이점을 제공하는지, 그리고 추후 연구에 대한 통찰을 제공합니다.



### SCAR: Efficient Instruction-Tuning for Large Language Models via Style Consistency-Aware Response Ranking (https://arxiv.org/abs/2406.10882)
Comments:
          21 pages

- **What's New**: 최근 연구에 따르면 사람 전문가의 일관된 응답 스타일을 유지하고 훈련 데이터 품질을 향상시키는 것이 화자 큰 언어 모델(LLMs, Large Language Models)의 성능을 크게 향상시키고 필요한 훈련 예시 수를 줄일 수 있음이 밝혀졌습니다. 이를 기반으로 SCAR(Style Consistency-Aware Response Ranking)라는 기법을 소개하여 응답 스타일의 일관성을 자동으로 우선순위화하여 훈련 데이터를 선택합니다.

- **Technical Details**: 응답 스타일은 프레젠테이션 스타일(presentation style)과 창의성 스타일(creativity style)로 나눌 수 있습니다. 프레젠테이션 스타일은 어조, 단어 선택, 서식 등을 포함하고 창의성 스타일은 응답의 불확실성이나 창의성을 나타냅니다. 스타일 일관성이 높은 예시를 선택하면 원래 전체 데이터보다 적은 데이터로도 유사하거나 더 나은 성능을 나타낼 수 있습니다. SCAR는 스타일 일관성을 유지하면서 데이터 품질을 유지하도록 응답을 순위화하며 학습된 언어 모델을 사용하여 스타일 요소를 강화합니다.

- **Performance Highlights**: SCAR를 사용해 상위 25%에서 0.7%에 해당하는 스타일 일관성이 높은 예시들만을 선택하여 훈련한 결과, 코드 작성 및 열린 질문 응답 벤치마크에서 전체 데이터셋을 사용한 모델의 성능을 일치시키거나 능가했습니다. SCAR는 데이터 선택 기준보다 효율적인 SFT를 제공하여 LLM 성능을 향상시키면서 계산 비용을 줄였습니다.



### Teaching Large Language Models to Express Knowledge Boundary from Their Own Signals (https://arxiv.org/abs/2406.10881)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하기 위해 CoKE(Confidence-derived Knowledge boundary Expression) 방법을 제안했습니다. 이는 LLM이 자신의 지식 경계(knowledge boundary)를 표현하고, 모르는 질문에 대한 답변을 회피하도록 훈련시키는 방법입니다.

- **Technical Details**: CoKE는 두 가지 주요 단계로 구성됩니다: 탐색(stage probing) 단계와 훈련(training) 단계입니다. 탐색 단계에서는 모델의 내부 신호를 사용하여 답변 가능한 질문과 불가능한 질문을 구분합니다. 훈련 단계에서는 세 가지 대표적 유형의 프롬프트를 사용하여 질문에 대한 정답을 학습시키며, 일관성(consistency)을 위해 정규화를 적용합니다.

- **Performance Highlights**: 광범위한 실험 결과, CoKE가 적용된 모델은 지식 경계를 표현하는 능력이 향상되었으며, 도메인 내(in-domain)와 도메인 외(out-of-domain) 성능 모두에서 한층 개선된 결과를 보여주었습니다.



### Exploring the Potential of Multimodal LLM with Knowledge-Intensive Multimodal ASR (https://arxiv.org/abs/2406.10880)
- **What's New**: 최근 들어, 멀티모달 대형 언어 모델(MultiModal Large Language Models, MLLMs)이 다양한 모달리티 정보를 통합하는 데 있어서 상당한 진전을 이루었으나, 교육 및 과학 분야의 실제 응용에서는 여전히 어려움이 있습니다. 본 논문은 멀티모달 과학 자동 음성 인식(Multimodal Scientific Automatic Speech Recognition, MS-ASR) 과제를 소개하며, 이는 과학 컨퍼런스 비디오의 정확한 음성 기록을 위해 슬라이드의 시각 정보를 활용합니다. 기존의 평가지표인 WER(Word Error Rate)가 이러한 성능을 정확하게 평가하지 못한다는 점을 지적하며, 오류의 심각성을 고려한 SWER(Severity-aware WER)를 제안합니다. SciVASR(Scientific Vision Augmented ASR) 프레임워크를 통해 포스트 편집으로 MLLMs의 기록 품질을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: MS-ASR 과제는 과학 컨퍼런스 비디오의 시각적 정보를 통합하여 음성 내용을 정확하게 기록하는 데 중점을 둡니다. 이 과제는 도메인별 용어를 시각적 정보로 보조하는 것이 필요하므로, 기존의 WER 대신 SWER 지표를 도입하여 평가의 정확성을 높입니다. 제안된 SciVASR 프레임워크는 MLLMs을 활용하여 포스트 편집을 통해 기록의 정확성을 높이며, Zero-shot inference를 제공합니다.

- **Performance Highlights**: 최신 MLLMs, 특히 GPT-4o를 사용한 평가에서 음성-only ASR 기준에 비해 45% 성능 향상을 보였습니다. 이는 시각적 정보 통합의 중요성을 강조하며, 상태-오브-더-아트(State-of-the-Art, SoTA) 멀티모달 능력을 효과적으로 평가합니다. 실험 결과 시각 정보를 추가하면 기술 용어 오류가 152건, 주요 오류가 28% 감소하는 것으로 나타났습니다. 인간 평가에서도 이러한 개선점들은 기록의 이해에 큰 영향을 주지 않는 것으로 확인되었습니다.



### COOL: Comprehensive Knowledge Enhanced Prompt Learning for Domain Adaptive Few-shot Fake News Detection (https://arxiv.org/abs/2406.10870)
- **What's New**: 최근 급부상하는 뉴스 도메인에서 데이터 부족 문제를 해결하기 위한 COOL(COmprehensive knOwledge enhanced prOmpt Learning) 방법이 제안되었습니다. 이 방법은 도메인 적응을 위한 few-shot Fake News Detection(FND)을 위해 포괄적인 지식 추출 모듈을 사용하여 긍정적 또는 부정적으로 뉴스와 연관된 구조적 및 비구조적 외부 지식을 추출합니다.

- **Technical Details**: COOL은 복합 지식 추출 모듈을 통해 외부 소스에서 긍정적 및 부정적으로 연관된 지식을 추출하고, 적응형 대비 학습을 활용한 하이브리드 프롬프트 학습 전략을 채택하여 뉴스와 지식 간의 도메인 불변 상호작용 패턴을 모델링합니다. 프레임워크는 학습 가능한 토큰으로 구성된 소프트 프롬프트와 수작업으로 작성된 하드 프롬프트를 사용하여 PLM이 뉴스 진위성을 추론할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, COOL이 기존 방법보다 consistently 높은 성능을 보였습니다. 이는 포괄적 지식을 프롬프트 학습에 통합하여 domain adaptive few-shot FND 성능을 향상시킨 결과입니다.



### Analyzing Key Neurons in Large Language Models (https://arxiv.org/abs/2406.10868)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 현대의 autoregressive LLMs(Large Language Models)인 LLaMA와 Mistral에서 쿼리와 관련된 뉴런을 효과적으로 찾고 이러한 LLMs에서 장문의 텍스트 생성을 다루는 방법을 탐구했습니다. NA-ICA(Neuron Attribution-Inverse Cluster Attribution)는 새로운 아키텍처에 구애받지 않는 프레임워크로, 다양한 도메인과 언어를 아우르는 multi-choice QA 데이터를 구축하여 주요 뉴런을 찾아내는데 유리하도록 설계되었습니다. 또한, 뉴런 분포 분석을 통해 LLMs 내에 로컬라이즈된(지역화된) 지식 영역이 존재하는지 확인했습니다.

- **Technical Details**: NA-ICA(Neuron Attribution-Inverse Cluster Attribution)는 TF-IDF 키워드 추출 방법에서 영감을 받은 프레임워크로, long-form 텍스트 생성을 처리할 수 있습니다. 이 프레임워크는 open-ended generation task를 multiple-choice QA 형식으로 변환하고, Knowledge Attribution 방법을 적용하여 뉴런과 입력 쿼리 사이의 관계를 해명합니다. 이후, 쿼리 클러스터를 수집하고 역 클러스터 배분을 계산하여 반복적인 뉴런의 영향을 줄입니다. 최종적으로, 뉴런 배분과 역 클러스터 배분 값을 곱하여 주요 뉴런을 식별합니다.

- **Performance Highlights**: NA-ICA는 기존의 방법들보다 뛰어난 성능을 보였으며, 다양한 도메인과 언어를 포함하는 두 가지 multi-choice QA 데이터셋에서의 실험을 통해 확인되었습니다. 뉴런 분포 분석 결과, 특히 중간 계층에서 도메인별 뉴런이 눈에 띄게 로컬라이즈된 부분이 발견되었습니다. 공통 뉴런은 동일한 상위 계층에 집중되었으며, 이는 주로 자주 사용하는 토큰을 표현하는 뉴런입니다.

- **Potential Applications**: NA-ICA는 지식 편집 및 뉴런 기반 예측에서도 활용 가능성이 높습니다. 이 방법은 LLMs의 내부 지식을 식별하고, 이를 바탕으로 모델의 기능을 강화할 수 있는 여러 잠재적인 응용 분야를 염두에 두고 있습니다.



### Step-level Value Preference Optimization for Mathematical Reasoning (https://arxiv.org/abs/2406.10858)
Comments:
          Ongoing Work

- **What's New**: SVPO(스텝 레벨 가치 선호 최적화)라는 새로운 알고리즘을 제안합니다. 이 알고리즘은 복잡한 멀티 스텝 추론 작업에서 모델 출력을 세밀하게 평가할 수 있도록 합니다. SVPO는 몬테 카를로 트리 탐색(MCTS)을 사용하여 멀티 스텝 추론에 대해 자동으로 스텝 레벨 선호도를 주석합니다.

- **Technical Details**: SVPO는 몬테 카를로 트리 탐색(MCTS)을 통해 스텝 레벨 선호도를 자율적으로 생성합니다. 이렇게 얻어진 선호도는 전통적인 솔루션 레벨의 선호도보다 더 세밀한 정보와 오류의 원인을 제공하며, 기존에 사용하는 강제적 지식 주입 방식보다 모델의 현재 능력에 더 잘 맞습니다. SVPO는 명시적인 가치 모델을 포함하여 추론 경로를 효과적으로 안내하고 선호도 학습을 돕습니다.

- **Performance Highlights**: SVPO는 최신 수학적 추론 벤치마크에서 최첨단 성능을 달성했습니다. 자율적 탐색 과정은 스텝 레벨의 선호도를 자연스럽게 제공하며, 솔루션 레벨의 선호도보다 모델의 추론 능력을 크게 향상시킵니다. 또한, 가치 모델이 정책 모델의 선호도 학습과 추론에서 효과적으로 작용합니다.



### Leading Whitespaces of Language Models' Subword Vocabulary Poses a Confound for Calculating Word Probabilities (https://arxiv.org/abs/2406.10851)
- **What's New**: 이번 연구에서는 Transformer 기반의 언어 모델들이 예측의 조건부 확률을 평가하는 과정에서 하위 단어 토크나이제이션(subword tokenization) 방식이 잠재적인 혼선을 초래할 수 있음을 지적하고 있다. 특히, 대부분의 언어 모델에서 하위 단어 어휘가 선행 공백문자를 포함하고 있어 단어의 확률를 자연스럽게 정의하지 못하게 된다고 언급한다.

- **Technical Details**: 논문은 선행 공백문자가 포함된 하위 단어 어휘가 일관되지 않은 단어 확률을 초래하며, 이는 예측의 체인 규칙(chain rule)과 Kolmogorov의 법칙(P(Ω) = 1)을 위반할 수 있음을 증명한다. 이를 해결하기 위해 현재 단어의 공백 확률을 재조정하는 간단한 디코딩 방법을 제안한다.

- **Performance Highlights**: 제안된 디코딩 방법을 적용하면 전이/비전이 문장에서 쉼표를 예상하는 LMs의 서프라이즈(놀라움)의 추정치가 크게 달라짐을 보여준다. 이는 인간 심리언어학 실험과 더 일관된 결과를 가져오며 예전 모델 예측들의 문제점을 해결할 수 있다.



### Large Language Models for Automatic Milestone Detection in Group Discussions (https://arxiv.org/abs/2406.10842)
- **What's New**: 이 논문에서는 대형 언어 모델(GPT)이 부분적으로 잘려 있거나 형식이 잘 갖추어지지 않은 그룹 구두 의사소통 과제에서 어떻게 성과를 내는지 조사합니다. 저자들은 여러 단계로 이루어진 퍼즐을 포함하는 새로운 그룹 과제 실험을 제안합니다. 이 실험을 통해 특정 마일스톤의 완료 여부를 검출하는 방법에 대해 탐구합니다.

- **Technical Details**: 저자들은 GPT를 사용하여 전사(transcription) 처리를 통해 마일스톤의 완료 시점과 수행자를 검출하는 방법을 연구했습니다. 전사 처리에 있어서 GPT를 반복적으로 프롬프트(prompting)하는 방법이 텍스트 임베딩(text embeddings)을 사용한 의미 유사도 탐색 방법보다 뛰어나다는 것을 입증했습니다. 또한, 다양한 컨텍스트 윈도우(context window) 크기에서 GPT 응답의 품질과 무작위성에 대해 논의했습니다.

- **Performance Highlights**: GPT를 사용한 반복적 프롬프트 방식이 텍스트 임베딩 기반 의미 유사도 탐색 방법을 능가하는 성과를 보였습니다. 이는 더 정교한 언어 모델이 비정형적이거나 불완전한 대화에서도 유의미한 성과를 낼 수 있음을 증명합니다.



### Exposing the Achilles' Heel: Evaluating LLMs Ability to Handle Mistakes in Mathematical Reasoning (https://arxiv.org/abs/2406.10834)
- **What's New**: 본 연구에서는 수학 단어 문제(Math Word Problems, MWPs)에서 이유 단계의 오류를 감지하고 수정하는 능력에 중점을 둔 새로운 데이터셋인 MWP-MISTAKE를 소개합니다. 이 데이터셋은 규칙 기반 방법과 더 작은 언어 모델을 통해 생성된 올바르고 잘못된 이유 단계를 포함합니다. 놀랍게도, 이 데이터셋은 최신 모델의 성능을 벤치마킹하면서 GPT-4o가 오류 감지 및 수정에서 뛰어난 성능을 보였으나, 소형 모델들은 여전히 많은 도전에 직면해 있음을 보여줍니다.

- **Technical Details**: MWP-MISTAKE 데이터셋은 GSM-8K, MATH, MATHBENCH, JEEBENCH와 같은 최신 수학 단어 문제 데이터셋을 통해 수집된 올바르고 잘못된 이유 단계를 포함합니다. 잘못된 이유 단계는 규칙 기반 접근법과 더 작고 제한된 언어 모델에서 파생된 것입니다. 또한 이유 단계를 무작위로 정렬하거나 삭제, 숫자 값 교체, 연산자 바꾸기 등의 6가지 규칙을 적용하여 오류 시나리오를 생성하였습니다.

- **Performance Highlights**: 다양한 대형 및 소형 언어 모델 (GPT-4o, GPT-4, GPT-3.5 Turbo, Claude, Llama, Phi, Mixtral 등)을 대상으로 벤치마크 결과를 발표하며, GPT-4o 모델이 가장 우수한 성능을 보였습니다. 특히, 작은 언어 모델들은 잘못된 이유 단계를 감지하고 수정하는 데 어려움을 겪고, 데이터 오염 및 메모리에 의존하는 경향이 있음을 확인했습니다.



### A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery (https://arxiv.org/abs/2406.10833)
Comments:
          33 pages (GitHub: this https URL)

- **What's New**: 최근 크고 다양한 분야에서의 과학 대형 언어 모델(scientific LLMs)에 대한 종합적인 조사가 진행되었습니다. 이번 연구는 250개 이상의 과학 대형 언어 모델을 분석하며, 다양한 분야와 데이터 모달리티(modalities)에서의 사전 학습 데이터셋과 평가 작업을 요약하고 있습니다. 또한, 과학적 발견을 촉진하는 LLM의 적용 사례도 조사되었습니다.

- **Technical Details**: 이번 연구는 BERT, RoBERTa와 같은 인코더 언어 모델의 마스크드 언어 모델링(MLM)부터, GPT, LLaMA 등을 응용한 다음 토큰 예측, 그리고 두 인코더를 활용한 대조 학습(contrastive learning)까지 다양한 사전 학습 전략을 다루고 있습니다. 과학적 LLM들은 각기 다른 데이터 형식을 처리하기 위해 자연스럽게 시퀀스화하거나 인위적으로 선형화하는 방식을 사용합니다. 여러 모달리티에서의 데이터, 예를 들어, 분자, 단백질, 표, 메타데이터 등을 다루는 방법이 논의되었습니다.

- **Performance Highlights**: 과학적 LLM들은 일반적인 자연어 처리(NLP) 작업뿐 아니라, 과학적 발견 과정에서의 가설 생성, 정리 증명, 실험 디자인, 약물 발견, 기상 예측 등 여러 단계에서 유용하게 활용될 수 있습니다. 예를 들어, SciBERT나 SciGPT와 같은 모델들은 과학 텍스트를 통한 대규모 무라벨 코퍼스에서 과학 지식을 습득하는 데 사용되었습니다. 또한, 그래프 신호를 더 잘 캐치하기 위해 Adapters, GNN-nested Transformers, Mixture-of-Experts Transformers 등의 변화를 추가한 최근 접근법도 제안되었습니다.



### Citation-Based Summarization of Landmark Judgments (https://arxiv.org/abs/2406.10824)
Comments:
          Accepted for publication at ICON 2023

- **What's New**: 이번 연구에서는 중요 판결문에 대한 컨텍스트 참조를 활용하여 목표 판결문의 추출적 요약을 생성하는 새로운 알고리즘 CB-JSumm을 제안합니다. 이 알고리즘의 성능을 평가하기 위해 인도 법원의 판결문 데이터를 활용했으며, 그 결과 유망한 성과를 보였습니다.

- **Technical Details**: CB-JSumm 알고리즘은 세 가지 단계로 구성됩니다. 첫 번째 단계에서, InLegalBERT라는 인도 법률 도메인에 맞게 사전 훈련된 변환기 기반 언어 모델을 사용하여 citances(참조 텍스트 구간)와 판결문 문장의 문맥적 임베딩을 도출합니다. 두 번째 단계에서는 이들 임베딩 간의 코사인 유사도를 계산하고, 세 번째 단계에서는 유사도 점수를 기반으로 판결문에서 요약에 포함될 만한 중요한 문장을 식별합니다.

- **Performance Highlights**: 이 연구에서는 인도 법원의 판결문 데이터를 기반으로 두 가지 데이터셋을 제작하여 알고리즘의 질을 평가했습니다. 제안된 알고리즘은 참조 판결문을 요약하는 데 있어 유망한 성과를 보여주었으며, 이는 추출적 요약의 가능성을 입증합니다.



### Self-Evolution Fine-Tuning for Policy Optimization (https://arxiv.org/abs/2406.10813)
- **What's New**: 새로운 논문에서 대형 언어 모델(LLMs)을 효율적이고 안정적으로 최적화할 수 있는 자기 진화 미세 조정(self-evolution fine-tuning, SEFT) 방법을 소개합니다. 이 방법은 주석이 달린 데이터 없이도 고품질 응답을 만들어내는 것을 목표로 합니다.

- **Technical Details**: SEFT는 단계적인 어댑티브 리바이저(adaptive reviser)를 훈련하여 낮은 품질의 응답을 향상시키는 동시에 높은 품질의 응답은 유지합니다. 어댑티브 리바이저는 초기 응답의 난이도에 따라 [Major Revise], [Minor Revise], [No Revise] 레이블을 할당하여 수정이 가능한 부분 만을 수정하도록 유도합니다. 이를 통해 SEFT는 RLHF(Reinforcement Learning from Human Feedback)의 복잡성을 피하면서 정책을 최적화하는 효율적이고 안정적인 방법을 제공합니다.

- **Performance Highlights**: SEFT는 AlpacaEval 2.0 및 MT-Bench 벤치마크 테스트에서 기존의 정렬 방법들(SFT, DPO, ORPO)보다 우수한 성능을 보여줍니다. 추가적인 비주석 데이터의 통합은 SEFT의 성능을 지속적으로 향상시켰으며, 실험 결과 점진적으로 정렬 데이터의 품질을 개선하는 것이 직접 고품질 데이터를 적용하는 것보다 더 나은 성능 향상을 가져다 준다는 것을 확인했습니다.



### LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction (https://arxiv.org/abs/2406.10811)
Comments:
          ACL(Findings)2024

- **What's New**: 최근에 대형 언어 모델(LLM)들은 다양한 텍스트 분석 작업에서 뛰어난 성과를 보여주며 주목받고 있습니다. 그러나 금융 부문은 복잡한 예측 작업을 위해 시계열 데이터에 의존한다는 점에서 다른 도전과제를 제시합니다. 이번 연구에서는 주식 움직임에 영향을 미치는 요인들을 식별하기 위해 LLM을 활용한 새로운 프레임워크, LLMFactor를 소개합니다. 기존의 키프레이즈나 감정 분석에 의존하는 방법들과 달리, 이 접근법은 주식 시장의 역동성을 보다 직접적으로 관련된 요인들을 추출하는 데 중점을 둡니다.

- **Technical Details**: LLMFactor는 Sequential Knowledge-Guided Prompting (SKGP)이라는 전략을 채택하여 LLM이 주식 관련 배경 지식을 생성하도록 유도한 후 관련 뉴스에서 주가에 영향을 미치는 잠재적인 요인을 식별합니다. 이후, 텍스트 형식으로 된 역사적 주가 데이터를 활용하여 주식 움직임을 예측합니다. SKGP는 매우 적은 배경 지식을 활용하여 프롬프트 템플릿의 풍부함을 증대시키는 fill-in-the-blank 기법을 사용합니다.

- **Performance Highlights**: LLMFactor 프레임워크는 미국과 중국 주식 시장의 네 가지 벤치마크 데이터셋에 대한 광범위한 평가를 거쳐, 기존의 최첨단 방법들을 능가하는 예측 성능을 입증했습니다. 이 프레임워크는 금융 시계열 예측에 있어서 설명 가능성을 극대화하고, 시장 변화의 원리를 명확히 설명할 수 있는 요인들을 제공하는 데 있어 높은 효과를 보입니다.



### Post-hoc Utterance Refining Method by Entity Mining for Faithful Knowledge Grounded Conversations (https://arxiv.org/abs/2406.10809)
Comments:
          Accepted at EMNLP 2023

- **What's New**: 최근 언어 생성 성능의 괄목할 만한 발전에도 불구하고, 모델이 생성한 응답은 종종 사실과 다르거나 주어진 소스 지식을 충실히 반영하지 않는 환각현상(hallucination) 문제가 있습니다. 특히, 지식 기반 대화(Knowledge Grounded Conversation; KGC) 작업에서는 정보성 있는 응답을 생성해야 하지만, 환각된 발화는 오해를 불러일으킬 수 있습니다. 이를 해결하기 위해 REM이라는 사후 정제 방법(post-hoc refinement method)을 제안합니다. REM은 생성된 발화가 소스 지식과의 일관성이 낮은 경우, 지식의 주요 엔티티를 추출하여 발화를 정제해줍니다.

- **Technical Details**: REM은 엔티티 마이닝(entity mining) 방법을 활용하여 생성된 발화가 소스 지식과 일치하지 않는 경우 해당 발화를 필터링하고, 주요 엔티티를 사용하여 발화를 더 충실하게 만듭니다. 이 방법은 PLM(pre-trained language models)과 LLM(large language models)에 plug-and-play 방식으로 적용할 수 있습니다. REM 모델은 인코더-디코더 구조를 사용하여 훈련되며, 엔티티 마이너(entity miner)는 소스 지식에서 명명된 엔티티를 추출하여 이를 기반으로 더 충실한 발화를 생성합니다.

- **Performance Highlights**: REM은 세 개의 KGC 데이터셋에 대한 기준 모델로 생성된 발화를 정제하는 실험을 통해 그 유효성을 입증했습니다. 교차 데이터 실험과 적대적 데이터 정제 실험을 통해 REM의 유연성과 타당성을 조사했습니다. 또한, 삭제 연구(ablation study)와 인간 평가를 통해 소스 충실성 점수 및 엔티티 커버리지(entity coverage)의 개선을 확인했으며, 대규모 언어 모델에도 적용 가능한 확장성을 입증했습니다. REM은 엔티티 수준의 환각을 줄이고, 더 바람직한 지식 기반 대화를 달성하는 데 중요한 역할을 합니다.



### ptt5-v2: A Closer Look at Continued Pretraining of T5 Models for the Portuguese Languag (https://arxiv.org/abs/2406.10806)
- **What's New**: 최근 Natural Language Processing (NLP) 분야에서 영어 모델에 비해 다른 언어의 모델 개발이 상대적으로 부족한 가운데, T5 모델을 포르투갈어에 맞게 지속적으로 pretrained 한 새로운 연구가 소개되었습니다. 이 연구는 $	exttt{ptt5-v2}$을 도입하여 포르투갈어 코퍼스를 사용한 T5 모델의 지속적 사전 훈련이 다양한 downstream 작업에 미치는 영향을 분석합니다.

- **Technical Details**: 중심적인 접근 방식으로, Google의 T5 모델(최대 3B 파라미터)을 포르투갈어 텍스트로 지속적으로 사전 훈련했습니다. 여기에는 데이터셋의 품질 필터, 최적화 전략, 다중 에폭(epochs) 사전 훈련과 같은 다양한 프리트레이닝 설정이 포함되었습니다. 실험에서는 포르투갈어 mC4 데이터셋(약 524 GB 텍스트) 및 SentencePiece Unigram 토크나이저(32,000 토큰)를 사용했습니다. 프리트레이닝 접근 방식으로는 span corruption task를 사용했으며, Adafactor optimizer와 constant learning rate(0.001)를 적용했습니다.

- **Performance Highlights**: 세 가지 포르투갈어 다운스트림 작업(ASSIN2 STS, ASSIN2 RTE, TweetSentBR)에 대한 파인튜닝 결과, 후자의 두 작업에서는 state-of-the-art(SOTA) 성능을 달성했습니다. 그러나 프리트레이닝 설정의 변화가 baseline 대비 미미한 영향을 미친다는 것이 주요 발견 중 하나였습니다. $	exttt{ptt5-v2}$ 프리트레인 모델 체크포인트와 파인튜닝된 MonoT5 rerankers는 HuggingFace에서 공개되었습니다.



### KGPA: Robustness Evaluation for Large Language Models via Cross-Domain Knowledge Graphs (https://arxiv.org/abs/2406.10802)
- **What's New**: 이 논문에서는 지식 그래프(KG)를 활용하여 대형 언어 모델(LLM)의 견고성을 평가하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 지식 그래프의 삼중항(triplets)으로부터 원래 프롬프트를 생성하고, 이를 오염시켜 공격 프롬프트(adversarial prompts)를 만들어 LLM의 견고성을 평가합니다. 기존 평가 방법들이 특정 벤치마크에 지나치게 의존하는 반면, 제안된 프레임워크는 이러한 제약을 극복하고 비용을 절감할 수 있습니다.

- **Technical Details**: 제안된 지식 그래프 기반 프롬프트 공격(KGPA) 프레임워크는 T2P(Triplets to Prompts) 모듈을 통해 지식 그래프의 사실(triplet)로부터 원래 프롬프트를 자동 생성합니다. 생성된 프롬프트는 'true', 'entity_error', 'predicate_error'의 세 가지 레이블로 태그가 붙습니다. 프롬프트 변환 전략으로는 템플릿 기반 전략과 LLM 기반 변환 전략 두 가지를 제안합니다. 템플릿 기반 전략은 주어와 객체 사이의 관계를 예제 문장에서 설명하는 템플릿을 사용합니다. 반면, LLM 기반 변환 전략은 LLM에 삼중항을 입력하여 더 이해하기 쉬운 고품질의 문장을 생성합니다. 또한, KGB-FSA 모듈은 한정된 수의 예제 프롬프트를 생성하여 몇 샷 공격(few-shot attack) 전략을 수행합니다.

- **Performance Highlights**: 실험 결과, ChatGPT 패밀리에서 GPT-4-turbo, GPT-4, GPT-3.5-turbo 순으로 견고성이 높게 나타났습니다. 또한, LLM의 견고성은 그들이 적용되는 전문 도메인에 따라 달라집니다. 동일한 LLM이라도 일반 도메인과 전문 도메인 지식 그래프로 평가했을 때 견고성이 다르게 나타나는 것을 확인했습니다.



### Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis (https://arxiv.org/abs/2406.10794)
- **What's New**: 대형 언어 모델(LLMs)은 방어 기제를 우회하여 유해한 내용을 생성하도록 유도하는 공격인 '탈옥(jailbreaking)'에 취약합니다. 이번 연구는 탈옥 공격의 성공 여부를 결정하는 내재적 특성을 이해하고자, 유해 및 무해 프롬프트가 LLM의 표현 공간에서 어떻게 행동하는지를 탐구합니다. 연구팀은 성공적인 공격이 유해 프롬프트의 표현을 무해 프롬프트 방향으로 이동시키는 공통 속성을 가지고 있다고 가정합니다.

- **Technical Details**: 연구팀은 기존의 jailbreaking 공격 방법에 숨겨진 표현(hidden representations)을 활용하는 새로운 최적화 목적을 도입하여, 유해 프롬프트를 무해 프롬프트 방향으로 이동시키는 실험을 수행했습니다. 이 접근 방식은 기존의 공격 방법(GCG 및 AutoDAN)에 통합할 수 있으며, 실험 결과를 통해 가설을 검증했습니다. 특히, Llama-2-13b-chat 모델에서 GCG를 개선한 후, 공격 성공률(ASR)이 26.15%에서 62.31%로 증가했습니다.

- **Performance Highlights**: 제안된 방법을 사용하여 GCG와 AutoDAN을 강화한 결과, Llama-2-13b-chat 모델에서 공격 성공률(ASR)이 크게 증가했습니다. GCG의 경우, 기본 공격 성공률 26.15%에서 제안된 방법을 적용한 후 62.31%로 36.16% 상승했습니다. 이러한 결과는 jailbreaking 공격에 대한 이해를 더욱 깊게 하며, 효과적인 방어 전략 개발에 기여할 것입니다.



### ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation (https://arxiv.org/abs/2406.10785)
Comments:
          16 pages, 5 figures

- **What's New**: 이번 연구에서는 프리트레이닝 언어 모델(Pretrained Language Models, 이하 PLMs)의 파라미터 효율적 미세 조정(Parameter Efficient Fine Tuning, 이하 PEFT)을 최적화하는 새로운 접근법 ShareLoRA를 소개합니다. ShareLoRA는 여러 계층에서 공유 저순위 적응(Shared Low Rank Adaptation)을 전략적으로 배치하고, 자기 주의(self-attention) 계층의 쿼리(Query), 키(Key), 값(Value) 구성 요소에 이 방식을 적용하여 교육 파라미터와 메모리 사용량을 크게 줄입니다.

- **Technical Details**: ShareLoRA는 각 계층에 있는 저순위 행렬 A와 B를 고유하게 구성할 필요 없이, 하나의 행렬(A 또는 B)을 모든 계층에 걸쳐 공유하는 접근법입니다. 이렇게 함으로써 훈련 가능한 파라미터 수를 줄이고 메모리 사용량 또한 감소시킬 수 있습니다. 또한, ShareLoRA는 쿼리, 키, 값 구성 요소 또는 다운 프로젝션(down-projection), 업 프로젝션(up-projection)을 공유하면서도 각 요소에서 고유한 업데이트를 유지하여 모델의 적응력을 보장합니다.

- **Performance Highlights**: ShareLoRA는 RoBERTa, GPT-2, LLaMA 및 LLaMA2와 같은 다양한 모델에서 분류 및 생성 태스크에서 높은 성능을 유지하며, 기존의 LoRA 응용 프로그램보다 우수한 전이 학습(transfer learning) 능력을 보여줍니다. 또한, 과적합을 방지하고, 다양한 언어 모델 아키텍처에서 확장 가능하고 고성능을 유지합니다.



### RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning (https://arxiv.org/abs/2406.10777)
- **What's New**: RoseLoRA는 파라미터 효율적 파인튜닝(PEFT) 방법으로, row와 column-wise로 안 중요한 파라미터를 식별하고 업데이트함으로써 효율적이면서도 모델의 다른 지식을 보존합니다. 이 방법은 지식 편집(knowledge editing)과 같은 특정 작업에 필요한 선택적 업데이트를 쉽게 만듭니다.

- **Technical Details**: RoseLoRA는 LoRA의 구조를 기반으로 하지만, low-rank 행렬들 간의 곱셈에 sparsity 제약을 추가하여 가장 중요한 파라미터들만 선택적으로 업데이트합니다. 이를 통해 fou역자림 수학적 분석은 행렬 곱셈의 sparsity에 대한 하한을 보장하며, row 및 column-wise의 sparsity 제약을 점진적으로 해결하기 위해 중요도 기반의 점수(sensitivity-based importance score)를 사용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 RoseLoRA는 기존 방법보다 적은 파라미터 수정으로도 최신 정보를 반영하는 데 탁월한 성능을 입증했습니다. 20여 개의 데이터셋에서 수행된 5가지 벤치마크 테스트에서 RoseLoRA는 일반적인 파인튜닝과 지식 편집 작업 모두에서 기준치를 능가하는 성과를 보였습니다.



### Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inferenc (https://arxiv.org/abs/2406.10774)
Comments:
          ICML 2024

- **What's New**: 최근 대형 언어 모델 (LLM)의 긴 문맥(1M 토큰까지) 수요가 증가함에 따라 긴 문맥 추론 속도가 느려지는 문제를 해결하기 위한 새로운 알고리즘, Quest가 제안되었습니다. 특히, Quest는 Query 기반 키-값(KV) 캐시 선택 알고리즘으로, 자주 사용되는 핵심 KV 캐시 페이지를 선택하여 추론 속도를 대폭 향상시킵니다.

- **Technical Details**: Quest는 각 페이지의 메타데이터로서 Key 벡터의 최댓값과 최솟값을 추적하여 주어진 페이지의 중요도를 추정합니다. 그리고 Query 벡터를 사용하여 상위 K개의 중요한 KV 캐시 페이지만 로드하여 주의 메커니즘(self-attention)을 실행합니다. 이를 통해 전체 KV 캐시를 메타데이터와 일부 페이지만으로 메모리 이동을 줄여 표현하는 효율적인 자원 사용을 실현합니다.

- **Performance Highlights**: Quest는 추론 지연(latency)을 7.03배 줄이고, self-attention 속도를 2.23배 향상시키며, 성능 손실은 거의 없습니다. 이는 특히 PG19 데이터셋, Passkey Retrieval Task, LongBench와 같은 긴 종속성을 가진 작업에서 효과적임이 입증되었습니다.



### Quantifying Generative Media Bias with a Corpus of Real-world and Generated News Articles (https://arxiv.org/abs/2406.10773)
Comments:
          20 pages, 10 figures

- **What's New**: 최근 연구에서는 저널리즘 분야에서 대규모 언어 모델(LLMs)을 통한 정치적 편향 분석을 시도했습니다. 이 연구는 인간이 작성한 2,100개의 기사와 이를 바탕으로 생성된 56,700개의 LLM 기반 기사 데이터를 통해 정치적 편향의 변화를 분석합니다. 주요 발견으로는 기본 LLM과 지침 조정된(Instruct-Tuned) LLM 사이에서 정치적 편향의 큰 차이가 있다는 점과, 특히 지침 조정된 모델은 일관된 정치적 편향을 보인다는 사실이 있습니다.

- **Technical Details**: 연구는 인간이 작성한 기사와 기계가 생성한 기사 간 속성 변화를 분석하기 위해, 2,100개의 인간 작성 기사 데이터셋과 56,700개의 LLM 생성 기사 데이터셋을 새롭게 구축했습니다. 편향 감지를 위해 감독된 모델(supervised models)과 LLM을 사용하여 분석을 수행했습니다. 또한, LLM들이 분류 도구로서 어떻게 작용하는지 살펴보고, 이 모델들이 그들 자신의 출력에 대해 가진 정치적 편향을 관찰했습니다.

- **Performance Highlights**: 지침 조정된 LLM들은 지속적으로 정치적 편향을 보이는 반면, 기본 형태의 LLM들은 그러한 경향이 덜하다는 점이 관찰되었습니다. 이는 LLM들이 생성하는 기사뿐만 아니라 분류하는 과정에서도 정치적 편향을 드러낼 수 있다는 것을 시사합니다. 이 연구는 저널리즘 도메인에서 LLM 정치적 편향에 대한 정량적 실험을 위한 프레임워크와 구조화된 데이터셋을 제공하여 향후 연구의 기초를 마련했습니다.



### GNOME: Generating Negotiations through Open-Domain Mapping of Exchanges (https://arxiv.org/abs/2406.10764)
- **What's New**: 이번 논문에서는 기존의 폐쇄 도메인(Closed-Domain) 협상 모델이 훈련된 도메인 외에는 일반화되지 않는다는 문제를 지적하고 있습니다. 이를 해결하기 위해, GNOME이라는 자동화 프레임워크를 제안합니다. GNOME은 대형 언어 모델(Large Language Models, LLMs)을 사용하여 기존의 인간 주석 데이터를 이용해 협상의 오픈 도메인(Open-Domain) 대화를 생성합니다. 이를 통해 협상 시스템의 일반화 능력을 향상시키고, 비용이 많이 들고 주관적인 수작업 데이터 생성 작업을 줄일 수 있습니다.

- **Technical Details**: GNOME 프레임워크는 세 가지 주요 단계로 구성됩니다: 사전 처리, 도메인 매핑(domain mapping), 후처리. 사전 처리 단계에서는 불완전한 대화를 제거하고, 다양한 데이터셋 간 공통된 라벨 매핑을 수행하며, 데이터셋의 불균형을 교정합니다. 도메인 매핑 단계에서는 LLM을 이용하여 개별 대화를 보다 다양하고 새로운 상황으로 변환합니다. 마지막으로 후처리 단계에서는 생성된 데이터를 정제하여 중복이나 원본 데이터 누출을 방지합니다.

- **Performance Highlights**: GNOME을 통해 생성된 데이터셋으로 훈련된 모델은 기존 최신 상태의 모델보다 더 나은 도메인 특화 전략 예측 성능을 보였고, 이전에 보지 못한 도메인에서도 더 잘 일반화되는 것으로 나타났습니다. 이것은 대형 언어 모델이 생성한 고품질의 합성 데이터가 추가적인 수작업 데이터의 대체재로 효과적으로 사용할 수 있음을 시사합니다.



### SparseCL: Sparse Contrastive Learning for Contradiction Retrieva (https://arxiv.org/abs/2406.10746)
- **What's New**: 저자들은 'SparseCL'라 불리는 새로운 접근 방식을 도입하여, 큰 문서 집합에서 쿼리에 모순되는 문서를 효과적으로 검색하는 방법을 제안하였습니다. SparseCL은 문장 임베딩(sentence embeddings)을 활용하여 문장 간 미묘한 모순점을 보존하도록 특별히 훈련되었습니다. 이 접근 방식은 코사인 유사도(cosine similarity)와 희소성 함수(sparsity function)를 결합한 지표를 사용하여 쿼리에 모순되는 문서를 효율적으로 식별하고 검색합니다.

- **Technical Details**: SparseCL 방법은 단어 임베딩의 차이에서 희소성을 유지하도록 문장 임베딩 모델을 훈련시킵니다. 쿼리에 응답할 때, 코사인 유사도와 임베딩의 차이 희소성을 기반으로 각 문서와 쿼리 간의 점수를 계산하고, 가장 높은 점수를 가진 문서를 검색합니다. 희소성의 구체적인 척도로는 Hoyer measure을 사용하였습니다.

- **Performance Highlights**: 이 방법은 Arguana 데이터셋 및 GPT-4를 사용하여 생성된 MSMARCO와 HotpotQA 데이터셋에서 테스트되었습니다. 실험 결과, MSMARCO와 HotpotQA에서 30% 이상의 정확도 향상이 확인되었습니다. 또한, 손상된 데이터 집합을 정리하고 고품질 QA검색 정확도를 복원하는 데에도 성공적으로 적용되었습니다.



### MIND: Multimodal Shopping Intention Distillation from Large Vision-language Models for E-commerce Purchase Understanding (https://arxiv.org/abs/2406.10701)
Comments:
          8 pages, 5 figures

- **What's New**: MIND는 대형 시각-언어 모델(Large Vision-Language Models, LVLMs)을 이용해 구매 의도를 추론하여, 인간 중심적 의도를 우선시하는 멀티모달 프레임워크를 소개합니다. 기존의 방법들은 텍스트 중심적 접근 방식을 취해 제품 이미지에서 얻을 수 있는 중요한 시각적 정보를 간과했고, 이에 따라 발생하는 확장 비용이 높았습니다.

- **Technical Details**: MIND는 아마존 리뷰 데이터를 이용해 107,215개의 제품에 관한 126,142개의 공동 구매 기록으로부터 1,264,441개의 의도를 포함하는 멀티모달 의도 지식베이스를 구축했습니다. LLaVa와 같은 대형 시각-언어 모델을 사용하여 제품 이미지와 제품명을 입력하여 의도를 생성하며, 인간 중심적 역할 인지 메커니즘을 통해 의도를 필터링합니다.

- **Performance Highlights**: 인간 평가를 통해 생성된 의도의 높은 타당성과 전형성이 확인되었습니다. 또한, 의도 이해와 관련된 두 개의 다운스트림 태스크에서 큰 언어 모델을 최적화하여 성능 향상을 보였습니다. 추가 실험에서는 시각적 정보의 중요성과 인간 중심적 필터 메커니즘의 필요성을 입증했습니다.



### Augmenting Biomedical Named Entity Recognition with General-domain Resources (https://arxiv.org/abs/2406.10671)
Comments:
          We make data, codes, and models publicly available via this https URL

- **What's New**: GERBERA는 일반 도메인의 NER(Named Entity Recognition) 데이터셋을 활용하여 생의학 명명 엔터티 인식(BioNER) 모델을 효과적으로 훈련하려는 새로운 방법입니다. 기존의 BioNER 데이터셋들과는 적은 개념 겹침을 가지는 쉽게 접근할 수 있는 리소스를 활용하여 성능을 향상시키는 것이 목표입니다.

- **Technical Details**: GERBERA는 먼저 사전 학습된 생의학 언어 모델을 목표 BioNER 데이터셋과 일반 도메인 데이터셋으로 멀티태스크 학습(Multi-task Learning)을 수행하여 훈련합니다. 그 후, BioNER 데이터셋에 맞게 모델을 다시 미세 조정(Fine-tuning)합니다. 이렇게 함으로써 여러 BioNER 데이터셋을 사용하지 않고도 효과적인 성능을 달성할 수 있습니다.

- **Performance Highlights**: GERBERA는 총 81,410개의 인스턴스로 구성된 다섯 개의 데이터셋에서 평균 0.9%의 성능 향상을 보였습니다. 특히 데이터가 제한된 BioNER 데이터셋에서 두드러진 성능 향상을 보였으며, JNLPBA-RNA 데이터셋에서는 F1 점수가 4.7% 향상되었습니다.



### DIEKAE: Difference Injection for Efficient Knowledge Augmentation and Editing of Large Language Models (https://arxiv.org/abs/2406.10660)
Comments:
          WIP

- **What's New**: DIEKÆ(Difference Injection for Efficient Knowledge Augmentation and Editing)라는 새로운 방법을 도입하여 PLM(LLaMA2-7B)의 지식 처리 과정을 인코더 시리즈로 분리함으로써 PLM의 성능을 크게 개선하였습니다.

- **Technical Details**: 이 방법은 인코더를 통해 외부 지식을 처리하여 PLM 계층에 주입함으로써 계산 비용을 줄이고 성능을 향상시키는 것을 목표로 합니다. 인코더의 초기 훈련 과정에서는 PLM에 대한 역전파(back-propagation)가 필요하지 않아 메모리와 시간 소모를 대폭 줄일 수 있습니다.

- **Performance Highlights**: 우리의 방식은 지식 증강(knowledge augmentation)과 지식 편집(knowledge editing) 모두에서 여러 기준선(baseline)과 비교하여 더 빠르고 효율적인 성능을 보였습니다. 코드와 데이터는 공개되어 있습니다.



### Emerging Safety Attack and Defense in Federated Instruction Tuning of Large Language Models (https://arxiv.org/abs/2406.10630)
Comments:
          18 pages

- **What's New**: 이번 논문에서는 Federated Learning(FL)을 통한 대규모 언어 모델(LLM) 훈련 시 발생할 수 있는 안전성 취약점을 처음으로 밝혀냈습니다. 구체적으로, 악의적인 클라이언트들이 자동으로 공격 데이터를 생성하여 FedIT 시스템에 침투시킬 수 있는 간단하지만 효과적인 공격 방법을 제안하였습니다.

- **Technical Details**: 연구진은 공격 클라이언트들이 수작업 없이 자동으로 생성한 공격 데이터를 이용하여 로컬 LLM을 훈련시키는 방식을 제안했습니다. 이 공격 방법은 LLM의 안전성 정렬(alignment)을 심각하게 저해하며, 기존의 많은 FL 방어 방법으로는 효과적으로 방어할 수 없습니다. 이에 따라, 연구진은 자동화된 방어 데이터 생성 파이프라인을 통한 포스트 호크(post-hoc) 방어 방법을 제안하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 안전성 공격 방법은 LLM의 안전성을 70%까지 감소시킬 수 있으며, 기존 방어 방법으로는 최대 4%의 안전성 향상에 불과한 것으로 나타났습니다. 그러나 제안된 방어 방법을 적용하면 공격받은 LLM의 안전성을 최대 69%까지 향상시킬 수 있어, 악의적인 유저가 없는 경우와 비슷한 수준의 안전성 정렬을 회복할 수 있습니다.



### On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models (https://arxiv.org/abs/2406.10625)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 생성하는 Chain-of-Thought Reasoning(COT)의 신뢰성을 향상시키기 위한 세 가지 접근법 중에서 'in-context learning', 'fine-tuning', 'activation editing' 기법을 탐구하고 새로운 전략을 제안하였습니다. 다양한 벤치마크 데이터셋을 사용하여 이들 전략이 신뢰성을 얼마나 개선할 수 있는지에 대해 광범위한 실증 분석을 수행했습니다.

- **Technical Details**: 논문은 기존의 'in-context learning', 'fine-tuning', 'activation editing'기법에 대한 새롭고 향상된 방법을 소개합니다. 'activation editing'에서는 모델의 내부 구조를 조사하고 특정 속성을 개선하기 위해 전략적으로 업데이트하며, fine-tuning에서는 모델 파라미터를 새로운 데이터셋을 사용해 업데이트합니다. 'in-context learning'에서는 추론 시간에 몇 개의 샘플을 모델에 제공하여 행동을 조정합니다. 이러한 방법들이 과거에는 다양한 작업에서 효과적이었음을 보였지만, CoT reasoning의 신뢰성 향상을 위한 맥락에서는 이번에 새롭게 탐구되었습니다.

- **Performance Highlights**: 세 가지 접근법 모두 CoT reasoning의 신뢰성을 크게 향상시키지 못했습니다. 'activation editing'은 매우 한정적 성공을 보였고, 'fine-tuning'과 'in-context learning'은 통제된 시나리오에서 약간의 개선을 보였으나 다양한 데이터셋에 일반화되지는 못했습니다. 이는 LLM에서 신뢰성을 이끌어내기 위한 본질적인 어려움을 강조하며, 지금 사용 가능한 기술들이 이 복잡한 문제를 해결하기에는 충분하지 않다는 결론을 내립니다.



### StructBench: An Autogenerated Benchmark for Evaluating Large Language Model's Ability in Structure-Rich Text Understanding (https://arxiv.org/abs/2406.10621)
- **What's New**: 최근 연구에서 특허받은 구조를 갖춘 데이터를 직접 이해하는 대형 언어 모델(LLM)을 평가하기 위한 새로운 방법을 제안했습니다. 이 방법은 수동으로 작성된 질문 템플릿과 생성 규칙을 기반으로 복잡성을 제어할 수 있는 구조화된 데이터를 생성합니다. 이 방법을 통해 StructBench라는 벤치마크를 도입하여 8개의 서로 다른 구조화된 언어와 29개의 특정 작업에 걸친 6,032개의 질문을 포함했습니다. 또한, 인간의 규칙 기반 작업 숙련도를 고려하여 StructBench-Hard를 도입하여 LLM과 인간 간의 성능 격차를 조사했습니다.

- **Technical Details**: StructBench는 여러 구조화된 데이터 형식을 포괄적으로 평가하도록 설계되었습니다. 구체적으로 JSON, YAML, ORG, Markdown, Latex 등의 구조화된 데이터를 대상으로 하여 8개의 구조화된 언어와 29개의 평가 작업을 포함합니다. 이 벤치마크는 수동 주석 데이터에 의존하지 않고 자동으로 평가 데이터를 생성하는 방법을 사용하여 모델의 성능을 객관적으로 평가합니다. 임의의 구조적 중첩 정도를 조절하여 생성된 데이터의 복잡성을 제어할 수 있습니다.

- **Performance Highlights**: StructBench-Hard에서 현재 최고 성능의 LLM이 65.0%의 정확도를 달성한 반면, 인간 정확도는 95.7%에 달했습니다. 이는 LLM이 복잡한 구조 정보를 처리하는 데 있어 불충분함을 강조합니다. 또한, StructBench를 사용한 미세 조정으로 기존 LLM의 구조화된 언어 이해를 향상시킬 수 있었으나, 모든 작업 유형에서의 성능 향상을 보장하지는 못했습니다.



### Multilingual Large Language Models and Curse of Multilinguality (https://arxiv.org/abs/2406.10602)
- **What's New**: 이번 논문은 다중언어 대형 언어 모델(LLMs)의 기술적 측면을 소개하며, 이 모델들의 기본 구조 및 훈련 방식, 토크나이제이션(tokenization) 방법 등을 설명합니다. 또한, 다중언어 LLMs가 가진 주요한 한계, '다중언어의 저주'(curse of multilinguality)를 다루고 이를 극복하려는 현재의 시도를 논의합니다.

- **Technical Details**: 다중언어 LLMs는 주로 트랜스포머(Transformer) 아키텍처를 사용하며, 인코더-디코더(Encoder-Decoder), 인코더-온리(Encoder-only), 디코더-온리(Decoder-only) 세 가지 유형으로 나뉩니다. 예를 들어 mBERT는 인코더-온리 모델로 문맥을 이해하는 데 유용하며, mT5와 같은 모델은 인코더-디코더 구조를 활용해 기계 번역 등의 작업에 효과적입니다. 주요 사전 훈련 목표 함수로는 Masked Language Modeling(MLM), Causal Language Modeling(CLM), Next Sentence Prediction(NSP), Translation Language Modeling(TLM)이 있습니다. 대부분의 다중언어 LLMs는 Wikipedia, Common Crawl(CC) Corpus, OSCAR 등의 대규모 데이터셋을 활용해 사전 훈련됩니다.

- **Performance Highlights**: mBERT와 같은 다중언어 인코더-온리 모델은 감정 분석(sentiment analysis), 명명된 개체 인식(named entity recognition)과 같은 작업에서 우수한 성능을 보입니다. 반면, XGLM 같은 디코더-온리 모델은 언어 모델링(language modeling)이나 텍스트 완성(text completion) 작업에 뛰어난 성능을 발휘합니다.



### BlockPruner: Fine-grained Pruning for Large Language Models (https://arxiv.org/abs/2406.10594)
- **What's New**: 최근 대형 언어 모델(LLMs)의 학습과 추론 비용이 급격히 증가하고 있다는 점을 주목해, 이 논문에서는 세부적인 블록 수준에서의 중복 축소를 통해 더 효율적인 모델 압축 방법을 제안합니다. 기존의 레이어 단위의 가지치기(pruning) 방법이 아닌, 블록 수준에서의 가지치기를 도입한 BlockPruner를 소개합니다. BlockPruner는 Transformer 레이어를 MHA (multi-head attention)와 MLP (multi-layer perceptron) 블록으로 세분화하여 불필요한 블록을 식별하고 제거합니다.

- **Technical Details**: BlockPruner는 각 Transformer 레이어를 MHA와 MLP 블록으로 분리하고, 퍼플렉시티(Perplexity) 측정을 통해 블록의 중요성을 평가합니다. 그런 다음 휴리스틱 탐색을 통해 반복적으로 중요도가 낮은 블록을 가지치기 합니다. 이 방법은 기존의 레이어 수준 가지치기 방법보다 더 세밀하고 효과적인 가지치기를 가능하게 합니다. 또한, 학습 없이 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: 실험 결과, BlockPruner는 다양한 크기와 아키텍처의 LLMs에 적용하여, 최첨단 기법들과 비교했을 때 더 세밀하고 효과적인 가지치기를 달성했습니다. 다운스트림 작업에서도 뛰어난 성능 유지를 확인하였으며, 블록 타입, 중요성 측정 지표, 데이터의 영향을 분석한 결과에서도 블록 수준 가지치기의 우수성을 입증하였습니다.



### Concentrate Attention: Towards Domain-Generalizable Prompt Optimization for Language Models (https://arxiv.org/abs/2406.10584)
Comments:
          Preprint

- **What's New**: 최근의 프롬프트 최적화(prompt optimization)는 사전 학습된 언어 모델(PLMs)의 성능을 상당히 향상시켰지만, 도메인 일반화(domain generalization)에 대한 가능성은 충분히 탐구되지 않았습니다. 본 연구는 도메인 일반화에 초점을 맞춘 새로운 최적화 목표인 'Concentration'을 제안합니다. 'Concentration'은 프롬프트 토큰에 대한 주의를 강화하고 주의 분포의 변동을 줄이는 목적을 가지며, 이를 통해 도메인 일반화 능력을 향상시킵니다.

- **Technical Details**: 얕은 실험을 통해 프롬프트 최적화의 주요 요소를 분석한 결과, 두 가지 중요한 발견을 했습니다. (i) PLMs의 깊은 층에서 더 많은 주의(attention) 가중치를 받는 프롬프트가 더 일반화 가능하다는 것과 (ii) PLMs의 깊은 층에서 더 안정적인 주의 분포를 가진 프롬프트가 더 일반화 가능하다는 것입니다. 제안된 Concentration 목표는 현재 디코딩 토큰에서 프롬프트 토큰으로 'lookback' 주의를 나타내며, 프롬프트의 주의 강도를 높이고 주의 분포의 변동을 줄이는 것을 목적으로 합니다.

- **Performance Highlights**: 제안된 방법을 통해 소프트 프롬프트 최적화 방법은 1.42%, 하드 프롬프트 최적화 방법은 2.16%의 정확도 향상을 보여주었습니다. 이러한 결과는 제안된 프롬프트 최적화 목표의 효율성을 검증하며, 도메인 일반화 가능한 프롬프트에 대한 주요 통찰을 제공합니다.



### We Care: Multimodal Depression Detection and Knowledge Infused Mental Health Therapeutic Response Generation (https://arxiv.org/abs/2406.10561)
- **What's New**: 비언어적 신호를 통해 우울증을 감지하는 연구에 새롭게 등장한 Extended D-vlog 데이터셋을 소개합니다. 1,261개의 YouTube 브이로그를 포함하고 있으며, 이는 실제 생활 상황에서의 개인 행동을 더 잘 반영하는 데 도움을 줍니다. 또한, GPT-3.5와 GPT-4 같은 대형 언어 모델(LLMs)의 등장과 함께 심리 상담 역할을 할 수 있는 가상 에이전트(Virtual agent)를 개발했습니다.

- **Technical Details**: 이 가상 에이전트는 두 가지 주요 기능을 갖추고 있습니다. 첫째, 개인의 우울증을 식별하고, 둘째, 인지 행동 치료(CBT) 기반의 치료적 응답을 제공하는 것입니다. Vision-Language TVLT Transformer 모델을 사용하여 다양한 모달리티 데이터를 처리하고, 이를 통해 우울증 감지 성능을 향상시켰습니다. TVLT 모델은 비디오 캡션 생성 및 멀티모달 감정 분석에서 최첨단 성능을 자랑합니다. 오디오의 경우, wav2vec2와 스펙트로그램을 결합하여 67.8%의 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 Mistral 모델은 인지 왜곡 평가 및 분류에서 각각 70.1%와 30.9%의 점수를 기록했으며, Bert 점수는 88.7%를 기록했습니다. 또한, 멀티모달 Extended D-vlog 데이터셋을 이용한 TVLT 모델은 67.8%의 뛰어난 F1-Score를 기록했습니다.



### Facts-and-Feelings: Capturing both Objectivity and Subjectivity in Table-to-Text Generation (https://arxiv.org/abs/2406.10560)
- **What's New**: 본 논문은 기존의 자연어 생성 문제 중 하나인 table-to-text(테이블-텍스트) 생성 문제를 주관성 관점에서 탐구했습니다. 이는 테이블에서 얻은 정보를 단순한 객관적 데이터가 아닌 주관적 해석을 통해 설명하는 것을 포함합니다. 이를 위해 우리는 Ta2TS 데이터셋을 도입했으며, 총 3,849개의 데이터 인스턴스를 포함하고 있습니다. 이 데이터셋은 스포츠, 금융 보고서, 날씨 예측 등 다양한 장르의 테이블을 다루고 있습니다.

- **Technical Details**: Ta2TS 데이터셋의 생성에는 T5 기반의 시퀀스-투-시퀀스(Sequence-to-Sequence) 학습자와 대규모 언어 모델(LLMs)을 활용했습니다. 모델은 테이블 데이터를 선형화하고 이를 입력으로 사용하여 자연어 텍스트를 생성합니다. 이 과정을 통해 각 모델의 주관성 요소와 사실적 일관성을 분석했습니다. 주요 성능 지표로 BERTScore와 Meteor 점수를 사용했으며, 모델 성능을 정량적 및 정성적으로 평가하였습니다.

- **Performance Highlights**: 분석 결과, 미세 조정된(피네 튜닝) 언어 모델(LMs)이 LLMs와 유사한 성능을 보였습니다. 구체적으로, 테이블 데이터에서 생성된 텍스트는 85.15%의 BERTScore와 26.28%의 Meteor 점수를 기록했습니다. 또한, T5 기반의 시퀀스-투-시퀀스 학습자가 GPT3.5 및 다른 LLMs를 능가하는 성능을 보이며 가장 높은 Bleu-4와 BERTScore를 기록했습니다.



### Large Language Model Enhanced Clustering for News Event Detection (https://arxiv.org/abs/2406.10552)
- **What's New**: 논문에서는 최신 뉴스 데이터 속에서 중요 사건을 자동으로 탐지하기 위한 새로운 프레임워크를 제시하고 있습니다. 이 프레임워크는 대규모 언어 모델(Large Language Models, LLMs)과 클러스터링 분석(clustering analysis)을 결합하여 GDELT(Global Database of Events, Language, and Tone) 데이터로부터 이벤트를 탐지합니다. 또한 클러스터링 결과의 유효성을 평가하기 위해 새로운 클러스터 안정성 평가 지수(Cluster Stability Assessment Index, CSAI)를 도입했습니다.

- **Technical Details**: 프레임워크는 사건 탐지 전의 작업(pre-event detection tasks)과 사건 탐지 후의 작업(post-event detection tasks)을 통해 이벤트 클러스터링을 향상시킵니다. 사건 탐지 전 작업에는 키워드 추출(keyword extraction)과 텍스트 임베딩(text embedding)이 포함되며, 사건 탐지 후 작업에는 이벤트 요약(event summarization)과 주제 라벨링(topic labeling)이 포함됩니다. CSAI는 잠재적 특징 벡터(latent feature vectors)를 활용하여 클러스터링 품질을 새로운 방식으로 측정합니다.

- **Performance Highlights**: 실험 결과, LLM 임베딩과 클러스터링 알고리즘을 결합하면 최고의 결과를 도출해내며, CSAI 점수 측면에서도 높은 견고성을 보여주었습니다. 또한 사건 탐지 후 작업이 의미 있는 통찰력을 제공하여 사건 클러스터링 결과를 더 효과적으로 해석할 수 있도록 도왔습니다.



### CroPrompt: Cross-task Interactive Prompting for Zero-shot Spoken Language Understanding (https://arxiv.org/abs/2406.10505)
- **What's New**: 이 논문은 SLU(Spoken Language Understanding) 분야에서 새로운 Cross-task Interactive Prompting (CroPrompt) 기법을 제안합니다. 이 기법은 상호 연관된 작업들 간 정보를 교환하여 데이터 부족 문제를 해결하고 성능을 최적화하는 데 중점을 둡니다.

- **Technical Details**: 기존의 SLU에서는 zero-shot prompting 기술을 사용하여 데이터 희소성 문제를 해결하려 했으나, 작업 간 상호작용 정보를 무시하여 최적의 성능을 내지 못했습니다. 이를 해결하기 위해 CroPrompt는 상호작용을 통해 SLU의 상호 연관된 작업들 간의 정보 교환을 가능하게 합니다. 또한 의도 정보 주입으로 인한 오류 전파 문제를 완화하기 위해 multi-task self-consistency 메커니즘을 도입했습니다.

- **Performance Highlights**: 표준 SLU 벤치마크 실험에서 CroPrompt는 일관되게 기존의 prompting 기법들을 능가하는 성능을 보였습니다. 또한 multi-task self-consistency 메커니즘은 오류 전파 문제를 효과적으로 완화하여 성능을 향상시켰습니다. 이를 통해 CroPrompt가 SLU 분야에서 더욱 많은 연구를 촉진할 수 있기를 기대합니다.



### Large Language Models as Event Forecasters (https://arxiv.org/abs/2406.10492)
Comments:
          10 pages, 3 figures, 10 tables

- **What's New**: 인간 이벤트의 주요 요소를 주어(subject), 관계(relation), 객체(object), 타임스탬프(timestamp)로 구성된 쿼드러플(quadruples)로 추출하는 방식을 소개하며, 이를 다섯 번째 요소인 텍스트 요약(textual summary)을 추가한 퀸투플(quintuples)로 확장할 수 있다고 설명합니다. 이러한 쿼드러플 또는 퀸투플은 특정 도메인 내에서 조직될 때, 시간 지식 그래프(temporal knowledge graph, TKG)를 형성합니다.

- **Technical Details**: 현재의 학습 프레임워크는 객체 예측(object prediction)이나 다중 이벤트 예측(multi-event forecasting)과 같은 몇 가지 TKG 관련 작업에 초점을 맞추고 있으며, 주로 그래프 신경망(Graph Neural Networks, GNNs)과 순환 신경망(Recurrent Neural Networks, RNNs)과 같은 복잡한 구조 및 순차 모델을 사용합니다. 그러나 이러한 방법들은 각각의 퀸투플 내에 내재된 맥락 정보를 효과적으로 캡처하지 못합니다. 이 논문은 대형 언어 모델(Large Language Models, LLMs)을 사용하여 TKG 학습 프레임워크의 설계를 단순화하면서도 예측 및 예측 작업에서 경쟁력 있는 정확성을 유지할 수 있는 방안을 연구합니다. 예를 들어, 객체 예측(OP) 작업을 표준 질문-응답(Question-Answering, QA) 작업으로 프레임화하기 위해 여러 프롬프트 템플릿을 개발하고, 인코더-디코더 생성 LLM을 사용하여 지시 미세 조정(instruction fine-tuning)을 수행합니다.

- **Performance Highlights**: 제안된 접근 방식은 GNNs 및 RNNs의 필요성을 제거하고, 대신 고정된 중간 임베딩을 생성하는 인코더 전용 LLM을 사용하여 예측 헤드(prediction head)와 자체 주의 메커니즘(self-attention mechanism)으로 예측 작업을 진행합니다. 여러 실제 데이터셋과 다양한 평가 지표를 사용한 광범위한 실험을 통해 이 접근 방식의 효과성과 견고성을 검증했습니다.



### Do Large Language Models Discriminate in Hiring Decisions on the Basis of Race, Ethnicity, and Gender? (https://arxiv.org/abs/2406.10486)
Comments:
          ACL 2024

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 채용 결정 시 이름에 따라 인종 및 성별 관련 차별을 나타내는지 조사합니다. 백인 지원자와 히스패닉 지원자 사이의 수락률 차이를 측정하며, LLMs가 백인 지원자를 선호하는 경향이 있다는 결과를 발견했습니다.

- **Technical Details**: 이 연구에서는 다양한 직책, 자격 수준, 템플릿 구문을 포함한 일련의 템플릿 프롬프트를 디자인해 LLMs가 채용 결정 이메일을 작성하도록 지시했습니다. 대표적인 인종/민족 및 성별과 연관된 이름을 대입하여 LLMs의 채용 결정에 미치는 영향을 분석했습니다. 사용된 모델로는 Mistral-Instruct-v0.1, Llama2 (7b, 13b, 70b), GPT-3.5-Turbo가 있습니다.

- **Performance Highlights**: 실험 결과, 백인 지원자가 가장 높은 수락률을 보였으며, 히스패닉 남성 지원자가 가장 낮은 수락률을 보였습니다. 총 2백만 개 이상의 이메일을 분석한 결과, LLMs는 이름에 따른 차별을 보여줬습니다. 수락 및 거절 이메일의 정확도를 보장하기 위해 TF-IDF 특성을 사용한 SVM 모델을 훈련시켰고, 이 모델은 0.98의 F1 점수를 기록했습니다.



### From Words to Worlds: Transforming One-line Prompt into Immersive Multi-modal Digital Stories with Communicative LLM Agen (https://arxiv.org/abs/2406.10478)
Comments:
          16 pages, 13 figures

- **What's New**: StoryAgent 프레임워크는 Large Language Models (LLMs)와 생성 도구들을 활용하여 디지털 스토리텔링을 자동화하고 정교하게 만듭니다. 이 프레임워크는 상향식 자산 생성과 하향식 스토리 초안 작성 접근 방식을 결합하여 다양한 문제를 해결하며, 인터랙티브하고 일관된 내러티브를 생성할 수 있게 합니다.

- **Technical Details**: StoryAgent는 텍스트 지시로 시작해 LLM-agent 기반의 스토리 클러스터로 복잡한 스토리 라인을 작성합니다. 이 스토리 클러스터는 디지털 스토리텔링 작업을 여러 하위 작업으로 분해하고, 각 작업은 특정 모달리티를 목표로 합니다. 생성된 텍스트 설명을 기반으로, 생성 모델들과 도구들이 조율되어 이야기의 실체적인 자산을 생성합니다. 이 과정은 상위 수준의 개입을 통해 인간의 수정 가능성도 고려합니다.

- **Performance Highlights**: StoryAgent 프레임워크는 참조 비디오 없이 일관된 디지털 스토리를 생성할 수 있으며, 여러 모달리티에서 내러티브의 일관성을 유지합니다. 이는 자동화된 디지털 스토리텔링에서 중요한 진전을 나타내며, 콘텐츠 생성 과정의 민주화와 참여도 향상을 가능하게 합니다.



### Personalized Pieces: Efficient Personalized Large Language Models through Collaborative Efforts (https://arxiv.org/abs/2406.10471)
- **What's New**: 이번 연구에서는 다수의 사용자들과 협력하여 효율적으로 개인화된 대형 언어 모델(Large Language Models, LLM)을 구축하는 새로운 프레임워크인 Personalized Pieces (Per-Pcs)를 제안합니다. 이 프레임워크는 사용자가 개인화된 parameter-efficient fine-tuning (PEFT) 조각을 안전하게 공유할 수 있게 하여, 커뮤니티가 협력해서 LLM을 개선할 수 있습니다. Per-Pcs는 기존의 non-personalized 모델이나 다른 PEFT 기반 방법들보다 성능을 뛰어넘으면서도 리소스 사용을 크게 절감합니다.

- **Technical Details**: Per-Pcs 프레임워크는 대표적인 사용자를 sharer로 선정하고, 그들의 PEFT 매개변수를 조각으로 나눈 후 각 조각에 대한 게이트(gate)를 훈련시킵니다. 이 조각들은 pool에 추가되며, 이후 타겟 사용자는 자신의 과거 데이터(history data)를 사용하여 필요한 PEFT 조각을 pool에서 선택하고 조립하여 개인화된 PEFT를 생성합니다. 이렇게 함으로써, Per-Pcs는 안전한 공유와 고도화된 사용자 모델링을 가능하게 하며, 많은 저장 공간과 계산 요구를 줄입니다.

- **Performance Highlights**: 실험 결과에 따르면 Per-Pcs는 non-personalized 모델이나 다른 PEFT 기반 메서드들보다 성능이 우수하며, OPPU와 비슷한 성능을 보여줍니다. 그러나 OPPU 보다 저장공간에서는 38배, 계산 효율에서는 7배 더 효율적입니다. Per-Pcs는 sharer 수와 조각 공유 비율에 대해 강건하며, 작은 비율의 조각만을 공유해도 강력한 성능을 유지합니다. 이러한 결과들은 개인화된 LLM을 구축하는데 Per-Pcs의 잠재력을 강조합니다.



### CancerLLM: A Large Language Model in Cancer Domain (https://arxiv.org/abs/2406.10459)
- **What's New**: 이번 연구에서는 암 분야에 특화된 대형 언어 모델(LLM), CancerLLM을 소개합니다. CancerLLM은 7억 개의 파라미터를 가진 모델로 Mistral 스타일의 아키텍처를 따르며, 17가지 암 유형에 대한 2,676,642개의 임상 기록과 515,524개의 병리 보고서를 사전 학습하였습니다. 그 후, 암 표현형 추출, 암 진단 생성, 암 치료 계획 생성의 세 가지 암 관련 작업에 대해 미세 조정되었습니다.

- **Technical Details**: CancerLLM은 7억 개의 파라미터를 가진 모델로, Mistral 스타일 아키텍처를 사용하여 대형 언어 모델의 성능을 향상시켰습니다. 이 모델은 University of Minnesota (UMN) Clinical Data Repository에서 수집된 2,676,642개의 임상 기록과 515,524개의 병리 보고서를 기반으로 사전 학습을 진행하였습니다. 또한,적으로 phenotype extraction, diagnosis generation, treatment plan generation의 세 가지 작업을 위해 미세 조정되었습니다.

- **Performance Highlights**: CancerLLM은 다른 기존 LLM에 비해 평균 F1 스코어가 8.1% 향상된 상태로 최첨단 결과를 달성했습니다. 또한, 두 가지 강건성 테스트베드(robustness testbeds)에서도 우수한 성능을 입증하였습니다. 이 모델은 임상 AI 시스템에 효과적으로 적용될 수 있으며, 암 분야에서의 진단 정확도와 치료 계획 수립에 있어 혁신적인 도움을 줄 수 있을 것으로 기대됩니다.



### Domain-Specific Shorthand for Generation Based on Context-Free Grammar (https://arxiv.org/abs/2406.10442)
- **What's New**: 이번 연구에서는 JSON, YAML, XML과 같은 구조화된 데이터를 생성하는 과정에서 많은 토큰(token)을 사용해 비효율성이 증가하는 문제를 해결하고자 도메인-특화 축약형(DSS) 포맷을 제안합니다. 이 포맷은 컨텍스트-프리 문법(CFG)을 사용하여 더 적은 토큰으로 데이터를 표현할 수 있도록 합니다.

- **Technical Details**: 제안된 DSS 포맷은 LLMs(Large Language Models, 큰 언어 모델) 예를 들어 GPT-4가 중요한 요소를 더 적은 토큰으로 캡처할 수 있게 합니다. 이 포맷은 컨텍스트-프리 문법(CFG)를 중심으로 되어 있으며, 이를 통해 효과적인 축약형 생성 및 축약된 데이터를 표준 구조화된 포맷으로 변환하는 파서를 만듭니다.

- **Performance Highlights**: 이 접근 방식을 데이터 시각화에 적용한 결과, 생성된 토큰 수가 3배에서 5배까지 감소했음을 확인할 수 있었습니다. 이는 지연 시간(latency)과 운영 비용을 크게 줄여줍니다.



### Enhancing In-Context Learning with Semantic Representations for Relation Extraction (https://arxiv.org/abs/2406.10432)
- **What's New**: 새로운 연구에서는 문장의 의미 구조를 AMR(Abstract Meaning Representation)로 표현하여 관계 추출(RE) 작업의 문맥 학습(ICL) 능력을 향상시키는 두 가지 새로운 방법을 제안합니다. 첫 번째 방법은 문장의 일부 그래프(최단 AMR 경로)를 탐색하고, 두 번째 방법은 전체 AMR 구조를 탐색합니다. 결과적으로, 모든 설정에서 고급 AMR의 의미 구조가 유익함을 입증했습니다.

- **Technical Details**: 이 연구에서는 관계 추출을 언어 생성 작업으로 공식화하기 위해 프롬프트 구성 방법을 보여주며, 정밀한 의미 관계 구조 정보를 활용하는 두 가지 구조 인식 데모 회수 프레임워크를 제안합니다. 비정밀 튜닝된 TAGSim과 정밀 튜닝된 FTSim을 사용하여 k-최단 경로 기반 데모 회수를 수행합니다. AMR 그래프를 통해 두 엔티티 간의 명시적 의미 관계를 모델링하고 TAG(Trimmed AMR Graph) 표현을 사용하여 회수된 데모의 품질을 높입니다.

- **Performance Highlights**: 제안된 모델은 네 가지 표준 관계 추출 데이터셋에서 평가되었습니다. 우리의 결과는 기존의 GPT 기반 벤치마크보다 일관되게 높은 F1 점수를 기록하며 두 데이터셋에서 SOTA 성능을 달성했음을 나타냅니다. 또한, 다른 두 데이터셋에서는 경쟁력 있는 성능을 발휘했으며, 비용 효율적인 성과를 보였습니다.



### SciEx: Benchmarking Large Language Models on Scientific Exams with Human Expert Grading and Automatic Grading (https://arxiv.org/abs/2406.10421)
- **What's New**: 최근 대형 언어 모델(LLMs)의 급속한 발전에 따라, 다양한 도메인에서 LLMs의 능력을 평가할 수 있는 벤치마크가 필요합니다. 본 논문에서는 대학생들이 과학적인 주제에 대한 작업을 평가받는 방식에서 영감을 받아, 컴퓨터 과학 시험 문제로 구성된 SciEx라는 새로운 벤치마크를 제안합니다. SciEx는 다국어(영어 및 독일어)와 다중 모달(이미지를 포함한 질문)을 지원하며, 다양한 난이도의 자유형 질문을 포함하고 있습니다.

- **Technical Details**: SciEx는 자유형 시험 문제로 구성되어 있어 LLM을 평가하는 것이 간단하지 않습니다. 따라서 우리는 LLM 출력에 대해 전문가의 평가를 제공하며, 다양한 최첨단 LLM들의 성능을 평가합니다. SciEx는 대학 컴퓨터 과학 시험 문제로 구성되어 있어 자연스럽게 다양한 질문 유형을 제공합니다. 또한 LLM의 성능을 평가하기 위한 자동 채점 방식(Lambda-as-a-judge)을 제안하며, 이는 전문가 평가와 0.948의 Pearson 상관 관계를 나타냅니다.

- **Performance Highlights**: 현존 LLM들은 SciEx의 자유형 시험에서 여전히 도전적인 과제로 남아 있으며, 최고 성능의 LLM도 평균 59.4%의 성적을 기록합니다. 특히 Claude와 GPT-4V와 같은 강력한 LLM들은 대학생 평균 점수를 능가하지만 여전히 완벽하지 않습니다. LLM을 채점자로 사용하는 실험에서, LLM들은 전문가 평가와 높은 상관 관계(0.948)를 보여, 평가자로서는 적절한 성능을 발휘했습니다.



### Determination of the Number of Topics Intrinsically: Is It Possible? (https://arxiv.org/abs/2406.10402)
Comments:
          This is the first full draft version of the article. The camera-ready version was accepted at the 11th International Conference on Analysis of Images, Social Networks and Texts (AIST 2023). Presented on September 30, 2023. Expected to be published in the conference proceedings, as part of the Communications in Computer and Information Science series (CCIS, Vol. 1905)

- **What's New**: 이 연구는 데이터셋에서 주제(topic)의 수를 추정하는 다양한 방법들을 비교 분석합니다. 기존의 방법들이 충분히 비교되지 않았고, 주제의 수가 특정한 데이터셋의 절대적인 속성이 아닌 방법과 모델에 따라 달라진다는 것을 강조합니다. 따라서 새로운 접근방식이 필요하며, 이 방향에서의 추가 연구가 제안됩니다.

- **Technical Details**: 주제 모델(topic model)은 unsupervised 텍스트 분석을 위한 통계 모델입니다. 주제 모델은 'word-in-topic' 분포(Φ)와 'topic-in-document' 분포(Θ)라는 두 확률 분포를 통한 훈련을 기반으로 합니다. 이 연구는 내재적 지표(intrinsic metric)를 중심으로 여러 방법을 검토하고, 각 모델과 데이터셋에 따른 실험적 증거를 제공합니다. 주요 방법으로는 hold-out perplexity(holdPerp), 안정성 분석(stability analysis), Jaccard Similarity Index, cophenetic correlation coefficient 등이 있습니다.

- **Performance Highlights**: 기존의 내재적 방법들이 신뢰성과 정확성이 떨어진다는 것을 발견했습니다. 주제의 수(T)는 특정 데이터셋의 고유 속성이 아닌, 사용된 방법과 모델에 의존한다는 결과를 도출했습니다. 이 연구는 수치 지표를 통해 주제 모델을 평가하고, 새로운 지표와 방법론 개발의 필요성을 강조합니다.



### Self-Reflection Outcome is Sensitive to Prompt Construction (https://arxiv.org/abs/2406.10400)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 자신이 생성한 응답을 반성(self-reflection)하여 오류를 식별하고 수정할 수 있는 능력을 탐구합니다. 특히, 반성 프롬프트(prompt)의 특정 단어 사용이 결과에 얼마나 영향을 미치는지 확인하였습니다.

- **Technical Details**: 연구팀은 Massive Multitask Language Understanding (MMLU) 데이터셋을 사용하여 실험을 진행했습니다. 실험에서 모델은 먼저 체인 오브 사고(chain-of-thought)와 네 샷 프롬프팅(four-shot prompting) 방식을 통해 초기 응답을 생성하고, 여러 자기 반성 프롬프트를 통해 응답을 업데이트했습니다. 실험 설정에는 Llama-3 8B 모델과 온도(temperature) 0.4, top_k = 10, top_p = 0.7 설정을 사용했습니다.

- **Performance Highlights**: 실험 결과, 프롬프트 구성에 따라 자기 반성 결과의 정확도가 크게 달라지는 것을 확인했습니다. 특정 문구 사용 시 잘못된 응답을 올바른 것으로 식별하거나, 반대로 올바른 응답을 잘못된 것으로 식별하는 잘못된 긍정(false positive) 비율이 높아지는 경향이 나타났습니다. 연구팀은 Mixture of Prompts (MoP) 프레임워크를 제안하여 이러한 편향을 완화하고 자기 반성의 정확도를 높였습니다.



### EWEK-QA: Enhanced Web and Efficient Knowledge Graph Retrieval for Citation-based Question Answering Systems (https://arxiv.org/abs/2406.10393)
- **What's New**: 이번 연구에서는 EWEK-QA (Enhanced Web and Efficient Knowledge graph retrieval solution) 시스템을 제안하여 기존의 인용 기반 질문 응답 시스템(QA systems)의 정확성과 효율성을 개선했습니다. 이 시스템은 웹에서 가져온 정보 외에도 Knowledge Graph(KG)에서 추출한 유용한 triples을 통합하여 더 풍부한 지식을 제공합니다.

- **Technical Details**: EWEK-QA 시스템은 적응형 웹 정보 검색(adaptive web retriever)과 효율적인 KG triples 통합을 특징으로 합니다. 웹 정보는 단순한 길이나 구분점을 기준으로 나누는 기존 방식 대신, 더 관련성과 완결성이 높은 인용 구문(quotes)을 추출합니다. 또한, EWEK-QA는 대형 언어 모델(LLMs)의 호출을 최소화하여 전체 파이프라인의 효율성을 유지하면서 KG triples을 효과적으로 통합합니다.

- **Performance Highlights**: EWEK-QA는 기본 웹 검색 시스템에 비해 관련성이 높은 구문 추출에서 20% 이상, 답변 범위의 커버리지에서 25% 이상, 그리고 완결성에서 35% 이상 향상된 성능을 보였습니다. 또한, 개방형 질문 응답(ODQA), 다중 단계 추론(multi-hop reasoning), 그리고 KG 기반 질문 응답(KGQA) 작업에서 최신 기술을 사용하는 시스템들을 능가하며, 특히 KGQA 작업에서는 10% 이상, ODQA 작업에서는 평균 3% 이상의 성능 향상을 달성했습니다. 인간 평가에서 EWEK-QA는 기존 최신 시스템에 비해 20% 이상의 정확성을 보였습니다.



### Enhancing Multilingual Voice Toxicity Detection with Speech-Text Alignmen (https://arxiv.org/abs/2406.10325)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 음성 유해성 분류(talks에서의 toxicity classification)를 위한 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 텍스트의 의미적 임베딩(embedding)을 다중 라벨 음성 유해성 분류기(speech toxicity classifier)에 통합하여, 훈련 중 텍스트 정보를 통합하면서도 추론 시에는 오디오만 필요로 합니다.

- **Technical Details**: 제안된 시스템은 두 주요 구성 요소로 이루어져 있습니다: 음성 인코더(speech encoder)를 포함한 음성 유해성 분류기와 다중 언어 텍스트 인코더(multilingual text encoder)입니다. 훈련 단계에서는 텍스트 인코더의 임베딩을 이용해 음성 인코더의 의미적 이해를 높이며, 텍스트 인코더는 동결(frozen)되어 훈련 중에만 사용됩니다. 추론 시에는 음성 유해성 분류기만 실행됩니다. 이 시스템은 다중 레이블 분류 문제로, 입력된 오디오 클립이 다중 유형의 위반 내용을 포함할 수 있습니다.

- **Performance Highlights**: 다양한 언어와 유해성 범주에서 음성 유해성 분류의 성능이 향상됨을 실험을 통해 입증했습니다. 대규모 모노링구얼(monolingual) 및 다국어(multilingual) 데이터셋에서 성능을 검증했으며, 제안된 프레임워크를 통해 음성 전용 유해성 분류기의 성능 개선을 확인했습니다.



### GenQA: Generating Millions of Instructions from a Handful of Prompts (https://arxiv.org/abs/2406.10323)
Comments:
          9.5 pages, 6 Figures, and 3 tables in the main body. Dataset available at this https URL

- **What's New**: 새로운 연구는 단일 프롬프트(prompt)에서 대규모의 지시 데이터셋을 생성하는 방법을 탐구합니다. 인간의 개입을 최소화하고, 다양한 주제를 아우르는 복잡한 다중 턴 대화(dialogs)를 포함하여 다양한 지시 예제를 작성할 수 있도록 LLMs를 활용합니다. 그 결과는 대규모 데이터셋을 생성하며, 이는 기존의 공개된 데이터셋보다 더 큰 규모로, 산업용 모델을 위한 데이터셋을 제공할 수 있습니다.

- **Technical Details**: 본 연구는 Llama-3 8B 베이스 모델을 미세 조정(finetuning)하는 과정에서, 자동화된 데이터 생성 방법을 사용하여 대규모 지시 데이터셋을 생성했습니다. 생성된 데이터셋은 간단한 완료 작업에서 복잡한 대화까지 다양한 지시 예제를 포함하며, 이는 거의 완전한 자동화 과정을 통해 수집되었습니다. 생성된 데이터셋과 이를 생성한 'generator 프롬프트', 그리고 미세 조정된 모델 체크포인트를 공개합니다.

- **Performance Highlights**: Llama-3 8B 모델을 미세 조정한 후, 생성된 데이터셋은 WizardLM과 Ultrachat을 지식 집약적 랭킹 작업과 대화 평가에서 모두 능가하거나 동등한 성능을 보였습니다. 이를 통해 자동화된 방법으로도 산업 규모의 높은 품질의 데이터셋을 생성할 수 있음을 입증했습니다.



### CNVSRC 2023: The First Chinese Continuous Visual Speech Recognition Challeng (https://arxiv.org/abs/2406.10313)
Comments:
          Accepted by INTERSPEECH 2024

- **What's New**: 중국어 연속 시각 음성 인식(Continuous Visual Speech Recognition, LVC-VSR)을 평가하는 최초의 챌린지가 진행되었습니다. 이 챌린지는 두 가지 과제를 포함했습니다: 특정 연설자를 위한 단일 연설자 시각 음성 인식과 등록된 여러 연설자를 위한 다중 연설자 시각 음성 인식입니다. 본 논문에서는 챌린지에 대해 종합적으로 리뷰하며, 데이터 프로파일, 과제 사양, 기본 시스템 구성 등을 다룹니다.

- **Technical Details**: 챌린지에서는 주로 CN-CVS 데이터셋을 활용하였으며, 두 개의 추가 데이터셋인 CNVSRC-Single과 CNVSRC-Multi도 사용되었습니다. 참가자들은 '고정 트랙'과 '오픈 트랙'에서 경쟁할 수 있었고, 주 평가 지표로 문자 오류율(Character Error Rate, CER)을 사용했습니다. 모델 아키텍처로는 CNN(Convolutional Neural Network), ResNet, MS-TCN(Multi-Scale Temporal Convolutional Network) 등이 사용되었습니다.

- **Performance Highlights**: 단일 연설자 과제에서 최고 제출물은 기본 성능을 크게 능가했습니다. 또한, 데이터 볼륨을 늘리는 단순한 접근방식이 좋은 효과를 보였으며, 반자동 레이블링 파이프라인을 통해 비디오 데이터를 텍스트로 전사하는 작업이 이루어졌습니다.



### In-depth analysis of recall initiators of medical devices with a Machine Learning-Natural language Processing workflow (https://arxiv.org/abs/2406.10312)
Comments:
          The Second version of the manuscript

- **What's New**: 이 논문은 기존의 도구들이 대용량 및 다양한 형식의 데이터를 효율적으로 처리하지 못하는 문제를 해결하기 위해 빅데이터(Big Data)와 기계 학습(Machine Learning, ML)-자연어 처리(Natural Language Processing, NLP) 기반의 작업 도구를 제시합니다. 이 도구는 2018년부터 2024년까지의 공개 의료 기기 리콜 데이터베이스를 분석하여 리콜 원인 식별 및 평가를 돕습니다.

- **Technical Details**: 제안된 도구는 비감독 밀도 기반의 공간 클러스터링 알고리즘 (Density-Based Spatial Clustering of Applications with Noise, DBSCAN)을 사용하여 각 리콜 원인을 명확하게 표현합니다. 이를 통해 실무자들이 리콜 원인을 쉽게 파악할 수 있도록 도우며, 텍스트 유사성 기반의 텍스트 분류(text similarity-based textual classification)로 리콜 원인 그룹의 크기를 관리하고 운영, 전술, 전략 수준에서의 관리 통찰력을 제공합니다.

- **Performance Highlights**: 이 ML-NLP 도구는 각 리콜 원인의 세부 사항을 포착할 뿐만 아니라, 기존 리콜 원인 간의 내부 연결성을 해석할 수 있으며 앞으로 SC(공급망)에서의 위험 식별 및 평가에 적용될 수 있습니다. 결론적으로, 논문은 더 많은 예방적 관행과 의료 기기 리콜을 위한 통제 솔루션이 미래에 기대된다고 언급합니다.



### CHiSafetyBench: A Chinese Hierarchical Safety Benchmark for Large Language Models (https://arxiv.org/abs/2406.10311)
Comments:
          13 pages, 3 figures

- **What's New**: 본 연구에서는 중국어 대형 언어 모델(LLMs)의 안전성을 평가하기 위한 새로운 벤치마크 CHiSafetyBench를 소개합니다. 이는 중국어 문맥에서 위험한 콘텐츠를 식별하고, 위험한 질문에 대한 답변을 거부하는 모델의 능력을 평가하기 위한 전용 데이터셋을 포함합니다. 해당 데이터셋은 5개의 위험 영역과 31개 카테고리를 포괄하는 계층적 중국어 안전 분류체계를 반영합니다.

- **Technical Details**: CHiSafetyBench는 다중 선택 질문(MCQ)과 질문-응답(QA) 두 가지 유형의 데이터를 포함하여 LLM의 다양한 안전성을 평가할 수 있도록 설계되었습니다. 또한, AI 모델의 안전 검출 및 방어 능력을 경제적이고 효율적으로 테스트할 수 있는 자동 평가 방법을 제안합니다. 실험을 통해 자동 평가의 타당성을 검증하고, 12개의 주류 중국어 LLM에 대한 포괄적이고 자동화된 안전 평가를 수행했습니다.

- **Performance Highlights**: 실험 결과, 다양한 모델들이 여러 안전 영역에서 상이한 성능을 나타냈으며, 모든 모델이 중국어 안전성 능력 측면에서 상당한 개선 잠재력을 가지고 있음을 확인했습니다. 이 데이터셋은 공개되어 연구자들이 자유롭게 활용할 수 있습니다.



### TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs (https://arxiv.org/abs/2406.10310)
- **What's New**: 최근 연구에서는 텍스트-속성 그래프(Text-Attributed Graph, TAG)에서 텍스트가 주로 노드에만 적용되고, 엣지에는 이진(binary) 또는 범주형(categorical) 태그만 사용되는 한계를 보완하기 위해, 텍스트 엣지 그래프(Textual-Edge Graph) 데이터셋과 벤치마크(TEG-DB)를 도입했습니다. TEG-DB는 노드와 엣지 모두에 풍부한 텍스트 설명을 포함한 다양한 도메인의 대규모 데이터셋을 제공하며, 이는 TEG 연구의 발전을 촉진하고 복잡한 실제 네트워크에 대한 깊이 있는 통찰을 제공합니다.

- **Technical Details**: TEG-DB는 북 추천(Book Recommendation), 전자 상거래(E-commerce), 학술(academic), 소셜 네트워크(social network) 등 다양한 도메인의 9가지 종합 데이터셋을 포함하고 있습니다. 각 데이터셋은 작은 규모에서 큰 규모에 이르는 다양한 크기로 노드와 엣지에 풍부한 원시 텍스트 데이터를 포함하고 있습니다. 이를 통해 연구자들은 그래프 분석에 있어서 더 깊이 있는 이해와 모델링이 가능해질 것입니다. 또한, TEG-DB는 통일된 데이터 형식을 사용하여 데이터 전처리, 데이터 로딩, 모델 평가 등의 표준화된 파이프라인을 개발하였습니다.

- **Performance Highlights**: TEG-DB의 벤치마크 실험에서는 사전 학습된 언어 모델(pre-trained language models)과 그래프 신경망(graph neural networks)을 포함한 다양한 기법을 평가했습니다. 이러한 연구는 모델 간의 성능 비교가 용이하게 하며, 노드와 엣지의 텍스트 정보를 효율적으로 활용할 수 있는 새로운 방법론 개발에 기여할 수 있습니다. 특히, 다양한 스케일의 사전 학습된 언어 모델이 생성한 임베딩의 효과 및 서로 다른 도메인 데이터셋이 성능에 미치는 영향을 심도 있게 분석했습니다.



### What is the best model? Application-driven Evaluation for Large Language Models (https://arxiv.org/abs/2406.10307)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Models, LLMs)을 실질적인 응용 시나리오에서 평가하기 위한 새로운 벤치마크인 A-Eval을 소개합니다. A-Eval은 실용적인 작업을 중심으로 다섯 가지 주요 카테고리와 27개 하위 카테고리로 평가 작업을 분류하고, 678개의 질문-답변 쌍으로 구성된 데이터셋을 제공합니다. 이를 통해 특정 응용 작업에 가장 적합한 모델을 선택하는 방법을 제시하고자 합니다.

- **Technical Details**: A-Eval 벤치마크는 텍스트 이해, 정보 추출, 자연어 생성, 논리적 추론, 작업 계획의 다섯 가지 주요 카테고리와 27개 하위 카테고리로 평가 작업을 분류합니다. 이를 위해 세 가지 난이도 수준(쉬운, 보통, 어려운)에 걸쳐 총 678개의 질문-답변 쌍을 수집하고 주석 및 검토 과정을 거쳐 데이터셋을 구축하였습니다. 또한, 모델 크기(0.5B-110B)에 따른 성능을 평가하기 위해 전문가와 자동 평가 방식을 설계했습니다.

- **Performance Highlights**: 평가 결과, 모델 크기와 작업 난이도 수준 간의 관계에 대한 흥미로운 규칙성을 발견했으며, 이에 기반하여 최적의 모델을 선택하는 실용적인 방법을 제안했습니다. 예를 들어, 더 어려운 작업이나 문제가 더 큰 규모의 모델을 필요로 하는 경향이 있지만, 초기 데이터에 따르면 더 작은 모델이 비용 대비 성능 비율(Cost-Performance Ratio, CPR)이 높다는 결론에 도달했습니다. A-Eval 데이터셋은 모든 데이터가 공개되어 있습니다.



### Enhancing Voice Wake-Up for Dysarthria: Mandarin Dysarthria Speech Corpus Release and Customized System Design (https://arxiv.org/abs/2406.10304)
Comments:
          to be published in Interspeech 2024

- **What's New**: 스마트 홈 기술의 확산은 음성 명령을 통해 기기 제어를 보다 수월하게 만들었습니다. 그러나 운변증(dysarthria)을 앓고 있는 사람들은 말의 변동성으로 인해 어려움을 겪고 있습니다. 이 논문은 운변증 환자를 위한 깨우기 단어 인식(wake-up word spotting, WWS) 작업을 다루며, 이를 통해 이들을 실제 응용 프로그램에 통합하는 것을 목표로 합니다. 이를 지원하기 위해, 집 환경에서 운변증 환자를 위해 설계된 오픈 소스 'Mandarin Dysarthria Speech Corpus (MDSC)' 데이터셋을 공개합니다.

- **Technical Details**: MDSC는 나이, 성별, 질환 유형 및 이해도 평가에 대한 정보를 포함하고 있습니다. MDSC에는 21명의 운변증 사용자와 25명의 비운변증 사용자로부터 각각 9.4시간 및 7.6시간의 녹음이 포함되어 있습니다. 녹음에는 각기 다른 속도로 반복된 10개의 깨우기 단어와 355개의 비깨우기 단어가 포함되어 있습니다. 우리는 MDSC에 대한 종합적인 실험 분석을 수행하여 운변증 음성 인식의 주요 도전 과제를 강조했습니다.

- **Performance Highlights**: 개발된 맞춤형 운변증 WWS 시스템은 이해도 처리에서 견고성을 보여주며 뛰어난 성능을 달성했습니다. 이 시스템은 Global Cepstral Mean and Variance Normalization (CMVN), Depthwise Separable Temporal Convolutional Network (DS-TCN), 및 스펙트로그램 증가(spectrogram augmentation), 속도 변형(speed perturbation), 화이트 노이즈 추가와 같은 여러 데이터 증가(augmentation) 기술을 사용하여 구축되었습니다.



### A Survey on Large Language Models from General Purpose to Medical Applications: Datasets, Methodologies, and Evaluations (https://arxiv.org/abs/2406.10303)
Comments:
          20 pages,3 figures

- **What's New**: 최근 자연어 처리 작업에서 뛰어난 성능을 입증한 대형 언어 모델(LLM)이 의료 상담 및 진단에서도 탁월한 능력을 발휘하고 있습니다. 이를 위해, 도메인-specific knowledge(도메인 특화 지식)를 추가로 훈련하여 의사-환자 대화를 매끄럽게 시뮬레이션하고 전문적인 의료 조언을 제공할 수 있는 의료 LLM(Medical LLM)이 개발되었습니다.

- **Technical Details**: 이 논문은 일반 LLM에서 의료 LLM을 훈련하는 방법을 체계적으로 탐색합니다. 주요 내용은 다음과 같습니다: (a) 훈련 데이터 획득 및 맞춤형 의료 훈련 세트 구축 방법, (b) 적절한 훈련 패러다임 선택, (c) 적합한 평가 벤치마크 선택, 및 (d) 기존의 도전 과제와 유망한 미래 연구 방향 등이 논의됩니다.

- **Performance Highlights**: 의료 LLM은 상호 진단에서 뛰어난 성능을 보이며, 애매한 설명에 대해 환자에게 필요한 정보를 능동적으로 요청하고 안내할 수 있습니다. 이는 일반 LLM과 차별화된 점입니다. 이러한 의료 LLM은 내과, 호흡기학, 위장병학 등 다양한 의학 분야에서 더 전문화된 답변을 제공합니다.



### SememeLM: A Sememe Knowledge Enhanced Method for Long-tail Relation Representation (https://arxiv.org/abs/2406.10297)
- **What's New**: 관계 인식 (Relation Recognition)은 여러 응용 분야에서 매우 중요합니다. 이번 연구는 기존 언어 모델(Language Models, LM)이 흔히 관측되는 관계를 학습하는 반면, 빈도는 낮지만 의미 있는 관계를 놓치는 문제를 제기합니다. 이에 기존의 LM을 보강하기 위해 외부 지식을 사용하는 방법을 제안하지만, 장기 꼬리 관계(long-tail relations)를 포함하는 코퍼스를 수집하는 것은 거의 불가능합니다. 이를 위해 새롭게 제안된 방법은 세메믹 지식을 향상시키는 SememeLM입니다.

- **Technical Details**: 제안된 SememeLM은 오픈하우넷(OpenHowNet)에서 세매믹(sememes) 및 관계를 추출하여 세매믹 관계 그래프(sememe relation graph)를 구축합니다. 그런 다음 그래프 주의 메커니즘(graph attention mechanism)을 사용하여 그래프의 표현을 학습합니다. 외부 지식 베이스가 많은 관련 없는 지식을 포함하고 있을 수 있어, 일관성 정렬 모듈(consistency alignment module)을 사용하여 노이즈를 줄이고 세매믹 지식을 통합합니다. 모델 학습에서는 관계 유사성 데이터를 활용하고, 지도 대비 학습(supervised contrastive learning)을 도입하여 모델을 훈련시킵니다.

- **Performance Highlights**: 제안된 방법론을 테스트하기 위해 7개의 단어 유추 데이터셋(word analogy datasets)에서 실험을 수행했습니다. 광범위한 실험 결과, SememeLM이 최신 방법들(sota methods)과 비교하여 우수한 성능을 보임을 확인하였습니다.



### CLST: Cold-Start Mitigation in Knowledge Tracing by Aligning a Generative Language Model as a Students' Knowledge Tracer (https://arxiv.org/abs/2406.10296)
- **What's New**: 이 연구에서는 학생들의 문제 해결 이력을 추적하여 현재 지식 수준을 추정하는 지식 추적(knowledge tracing, KT)에서 발생하는 콜드 스타트(cold-start) 문제를 해결하기 위해 생성형 대규모 언어 모델(LLM)을 적용한 새 프레임워크를 제안합니다. 특히, 생성형 언어 모델을 학생들의 지식 추적기로 정렬하여 사용하는 CLST(cold-start mitigation in knowledge tracing by aligning a generative language model as a students’ knowledge tracer) 프레임워크를 제공합니다.

- **Technical Details**: 이 연구는 수학, 사회, 과학 과목에서 수집된 데이터를 사용하여 문제 해결 데이터가 자연어로 표현되도록 하여 KT 작업을 자연 언어 처리(NLP) 작업으로 제시했습니다. 그런 다음 생성형 LLM을 이 포맷된 KT 데이터셋을 사용해 파인튜닝(fine-tuning)하였으며, 데이터가 부족한 상황에서 다양한 벤치마크 모델들과 비교하여 CLST의 성능을 평가했습니다.

- **Performance Highlights**: CLST는 100명 미만의 학생들로 이루어진 데이터셋에서도 콜드 스타트 상황에서 예측, 신뢰성, 도메인 간 일반화 측면에서 성능 향상을 보여주었습니다. RNN, MANN, 트랜스포머 기반의 모델들과 비교했을 때 CLST가 더 높은 성능을 기록했습니다.



### Robustness of Structured Data Extraction from In-plane Rotated Documents using Multi-Modal Large Language Models (LLM) (https://arxiv.org/abs/2406.10295)
Comments:
          20 pages, 6 figures

- **What's New**: 이번 연구는 최신 멀티 모달 대형 언어 모델(LLM)들이 문서의 흔들림(평면 내 회전 또는 skew)에 어떻게 영향을 받는지 조사합니다. 문서 데이터 추출 정확도가 이러한 흔들림으로 인해 얼마나 영향을 받는지 살펴봅니다. 이에 따라 Anthropic Claude V3 Sonnet, GPT-4-Turbo, Llava:v1.6 등의 모델에 대한 실험 결과를 제시합니다.

- **Technical Details**: 문서의 흔들림이 데이터 추출 정확도에 미치는 영향을 조사하기 위해, 다양한 각도의 흔들림을 갖는 합성 샘플 문서에서 특정 엔티티를 추출하는 실험을 진행했습니다. 모델별로 안전한 회전 각도(SIPRA)를 식별하고, 일반적인 skew 탐지 및 수정 메커니즘의 한계와 대안을 제안합니다. 특히, 선제적으로 더 견고한 멀티 모달 아키텍처를 개발하거나 모델의 사전 학습 단계에서 skewing 기법을 통합하는 방법을 고려합니다.

- **Performance Highlights**: 모든 테스트 모델에서 문서의 흔들림이 데이터 추출 정확도에 부정적인 영향을 미쳤으며, 그 영향의 정도는 모델별로 상이했습니다. 또한, 모델의 허위 정보 생성(환각)에도 흔들림이 영향을 미치는 것으로 나타났습니다. 이는 실세계를 반영한 폭넓은 문서 품질 및 조건을 테스트할 필요성을 강조합니다.



### RelevAI-Reviewer: A Benchmark on AI Reviewers for Survey Paper Relevanc (https://arxiv.org/abs/2406.10294)
- **What's New**: 최근 인공지능(AI) 분야의 진보, 특히 대형 언어 모델(Large Language Models, LLMs)의 광범위한 채택은 텍스트 분석 능력을 크게 향상시켰습니다. 이 기술적인 진화는 전통적으로 동료 연구자들에 의해 관리되는 과학 논문의 리뷰 업무를 자동화하는데 상당한 가능성을 제공합니다. 이번 논문에서는 설문 논문 리뷰 업무를 분류 문제로 개념화하여 특정 프롬프트에 대한 논문의 관련성을 평가하는 자동 시스템인 RelevAI-Reviewer를 제안합니다.

- **Technical Details**: RelevAI-Reviewer는 한 개의 프롬프트와 네 개의 후보 논문이 포함된 새로운 데이터셋 25,164 개의 인스턴스로 구성되어 있습니다. 각 인스턴스는 프롬프트와 관련성 별로 다양한 후보 논문을 포함하고 있으며, 모델은 각 논문의 관련성을 결정하고 가장 적합한 논문을 식별하는 것을 목표로 합니다. 기본 접근 방식으로는 전통적인 지원 벡터 기계(Support Vector Machine, SVM)와 BERT와 같은 고급 언어 모델을 탐구합니다.

- **Performance Highlights**: 예비 결과에 따르면, BERT 기반의 엔드-투-엔드 분류기가 다른 전통적인 머신 러닝 방법보다 성능이 뛰어납니다. 논문에서는 이 문제를 공개 챌린지(public challenge)로 제시하여 이 분야의 연구에 대한 참여와 관심을 촉진하고자 합니다.



### MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases (https://arxiv.org/abs/2406.10290)
- **What's New**: 최근 모바일 디바이스에서 대형 언어 모델(LLMs)과 대형 다중모달 모델(LMMs)의 배포가 주목받고 있습니다. 새로운 벤치마킹 프레임워크인 MobileAIBench가 이를 평가하기 위해 도입되었습니다. MobileAIBench는 다양한 크기, 양자화 수준, 그리고 실제 디바이스에서의 LLM과 LMM 작업 성능을 평가합니다.

- **Technical Details**: MobileAIBench는 데스크탑과 서버에서 모델 성능을 평가하는 오픈 소스 라이브러리와 iOS 앱으로 구성된 두 부분의 프레임워크입니다. 이 프레임워크를 사용하여 사용자는 양자화된 모델을 다양한 벤치마크 작업을 통해 평가할 수 있습니다. iOS 앱을 통해 사용자는 RAM 및 CPU 사용률과 같은 모바일 하드웨어 이용 현황을 측정할 수 있습니다.

- **Performance Highlights**: MobileAIBench는 7B 파라미터 이하의 모바일 최적화된 LLMs와 LMMs를 대상으로 다양한 양자화 수준(16-bit에서 3-bit까지)에서의 성능을 측정합니다. iPhone 14에서 실제 디바이스 성능을 측정하며, 이를 통해 전체적인 처리 지연 시간(time-to-first-token)과 CPU, RAM 사용량을 분석합니다. LLM과 LMM 작업에서 양자화가 실제 성능에 미치는 영향에 대한 통찰을 제공합니다.



### VeraCT Scan: Retrieval-Augmented Fake News Detection with Justifiable Reasoning (https://arxiv.org/abs/2406.10289)
- **What's New**: 새로운 논문에서는 가짜 뉴스(Fake News)의 확산 문제에 대응하기 위해 VeraCT Scan이라는 혁신적인 시스템을 소개했습니다. 이 시스템은 뉴스 기사에서 핵심 사실을 추출한 후 인터넷 전역에서 이를 확인하는 검색(Internet-wide search) 과정을 통해 뉴스의 진위 여부를 판별합니다. 결과적으로 투명하고 신뢰할 수 있는 증거와 논리를 제공하여 뉴스의 신뢰성을 높이는 것을 목표로 합니다.

- **Technical Details**: VeraCT Scan은 정보 검색 최적화 기법과 결합된 포괄적인 파이프라인을 갖추고 있습니다. GPT-4 Turbo와 Llama-2 13B를 미세 조정(Fine-tuning)하여 뉴스 내용 이해, 검증, 추론 작업을 수행합니다. 핵심 사실을 다중 수준으로 세분화하고, 각 사실별로 인터넷 검색을 수행하여 관련 정보를 수집한 후, 출처 신뢰도를 고려하여 뉴스의 진위를 판별합니다. 또한, 시스템의 투명성과 신뢰성을 높이기 위해 검증 논리를 생성하고 관리합니다.

- **Performance Highlights**: VeraCT Scan은 여러 가짜 뉴스 검출 데이터셋에서 종합적인 평가를 통해 최고 수준의 성능(State-of-the-art accuracy)을 입증했습니다. 이는 자동 생성 및 미세 조정된 대규모 언어 모델들(LLMs)을 활용하여 높은 정확도의 뉴스 검증 작업을 수행한 결과입니다.



### Mimicking User Data: On Mitigating Fine-Tuning Risks in Closed Large Language Models (https://arxiv.org/abs/2406.10288)
- **What's New**: 대형 언어 모델(fine-tuning large language models)을 소량의 고품질 데이터셋에 미세조정(fine-tuning)하면 특정 다운스트림 작업에서 성능이 향상될 수 있습니다. 하지만 최근 연구는 무해한 지침을 따르도록 미세조정하는 것이 모델의 안전 정렬(safety alignment) 과정을 무효화하고 해로운 쿼리(harmful queries)를 따를 가능성을 증가시킬 수 있음을 보여줍니다. 본 연구는 폐쇄형 모델(closed models)에서 사용자 데이터가 활용되는 방식을 제공자가 통제하는 환경에서 다양한 작업별 데이터의 위험을 탐구합니다.

- **Technical Details**: 본 연구는 악의적인 행위자가 거의 모든 작업별 데이터셋의 구조를 미세하게 조작할 수 있는 방법을 보여줌으로써, 외관상 무해하고 합리적인 다운스트림 작업 성능을 유지하면서도 훨씬 더 위험한 모델 행동을 촉진할 수 있음을 입증합니다. 이를 해결하기 위해, 본 연구는 사용자 데이터의 작업 형식과 프롬프팅 스타일(prompting style)을 모방하는 안전 데이터를 혼합(mixing in safety data)하는 새로운 완화 전략을 제안합니다. 이는 기존의 기준선(baselines)보다 안전 정렬을 재확립하는 데 더 효과적이며, 유사한 작업 성능을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 완화 전략은 소량 고품질 데이터셋에 대한 미세조정에도 불구하고 모델의 악의적인 행동을 최소화하면서 안전 정렬을 유지하는 데 유의미한 결과를 나타냈습니다. 이는 데이터셋의 작업 형식 및 프롬프팅 스타일을 고려한 안전 데이터의 혼합을 통해 달성되었습니다.



### Improving child speech recognition with augmented child-like speech (https://arxiv.org/abs/2406.10284)
Comments:
          5 pages, 1 figure Accepted to INTERSPEECH 2024

- **What's New**: 최신 연구에 따르면, 아동 음성 인식(ASR)은 아동 음성(child speech)에 대해 여전히 최적의 성능을 보여주지 못하고 있다. 아동 음성 데이터의 부족은 아동 음성 인식(CSR)의 발전을 저해하는 주요 요인 중 하나다. 본 연구는 기존 아동 발화자 데이터와 추가(새로운) 아동 발화자를 통한 모노링구얼(monolingual) 및 크로스링구얼(cross-lingual, 네덜란드어-독일어) 음성 변환(VC)을 연구하고, 이를 통해 CSR 성능을 개선하려 했다.

- **Technical Details**: 본 연구에서는 네덜란드어-독일어 크로스링구얼 VC가 아동 ASR 성능을 크게 개선하는 것으로 나타났다. 특정 ASR 모델(FT-Conformer 및 FT-Whisper 모델)을 미세 조정(fine-tuning)하기 위해 2배로 증강된 데이터를 사용한 결과, 기본 모델 대비 워드 에러율(WER)이 약 3% 절대적으로 감소했다. 또한, 처음부터 학습된 모델에 6배로 증강된 데이터를 사용한 경우, WER이 3.6% 절대적으로 개선되었다. '고품질'의 VC 생성 데이터를 소량 사용한 경우에도 유사한 성능 향상을 보였다.

- **Performance Highlights**: 연구 결과, 크로스링구얼 VC 기반의 데이터 증강은 CSR 성능을 현저하게 향상시켰다. 특히, FT-Conformer 및 FT-Whisper 모델의 경우, 2배 증강 데이터를 사용해 WER이 약 3% 감소했으며, 모델을 처음부터 학습할 때 6배 증강 데이터를 사용한 경우에는 3.6% 절대적으로 개선되었다.



### Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection (https://arxiv.org/abs/2406.10283)
- **What's New**: 이번 연구는 WavLM 모델의 다중 레이어 특성을 활용하여 음성 위조 방지에 대한 성능을 향상시키는 새로운 'Attentive Merging' 방법을 제안합니다. 이를 통해 계층적 숨겨진 임베딩(hierarchical hidden embeddings)을 효과적으로 결합하여 최적의 성능을 도출합니다.

- **Technical Details**: WavLM 모델은 CNN 인코더와 다수의 트랜스포머(transformers) 레이어로 이루어져 있으며, 음성 위조 방지 작업에 유용한 다양한 레벨의 음성 특징을 추출합니다. 제안된 'Attentive Merging' 방법은 각 트랜스포머 레이어에서 추출된 숨겨진 임베딩을 최적화된 방식으로 결합하여 필요한 정보를 강조합니다. CNN 인코더는 고정된 상태로 유지되며, 24개의 트랜스포머 레이어를 통해 멀티-레벨 음성 특징을 추출합니다. 이 특징들은 두 단계의 과정으로 평균화되고 활성화 함수를 통해 결국 하나의 텐서로 압축됩니다.

- **Performance Highlights**: 실험 결과, 'Attentive Merging' 방법을 통해 WavLM 모델을 미세 조정한 경우, ASVspoof 2019LA, 2021LA, 2021DF 평가 세트에서 각각 0.65%, 3.50%, 3.19%의 최적의 등오차율(EER)을 달성했습니다. 또한, WavLM 대형 모델의 초기 트랜스포머 레이어들이 위조 방지 작업에 크게 기여하는 것으로 나타나, 일부 미리 학습된 레이어만을 사용하여도 높은 성능과 더불어 계산 비용을 절감할 수 있음을 확인했습니다.

- **Conclusion**: 제안된 방법은 WavLM 모델의 트랜스포머 레이어의 숨겨진 임베딩을 효율적으로 결합하여 음성 위조 방지 성능을 크게 향상시키며, 제한된 계산 자원으로도 SOTA(State-of-the-Art) 성능을 달성할 수 있음을 보여줍니다.



### Prompt-Based Length Controlled Generation with Multiple Control Types (https://arxiv.org/abs/2406.10278)
Comments:
          Accepted by ACL 2024 findings. arXiv admin note: text overlap with arXiv:2308.12030

- **What's New**: 최신 연구는 다양한 NLP 작업에서 뛰어난 성과를 보이는 대형 언어 모델(LLMs)의 길이 제어 텍스트 생성을 다룹니다. 특별히 '동일 길이' 제어에 그치지 않고 다양한 길이 제어 유형을 높은 정확도로 달성하기 위해 프롬프트 기반 (prompt-based) 방법을 제안합니다. 이 방법은 강화 학습 (Reinforcement Learning, RL)과 샘플 필터링을 도입하여 모델의 길이 제어 능력을 높입니다.

- **Technical Details**: 이 연구에서는 규칙 기반 보상 모델(rule-based reward models)을 통해 주어진 보상 신호를 사용하여 강화 학습 및 샘플 필터링을 수행합니다. 이를 통해 특정 제어 지침을 따르는 출력을 보상합니다. 또한, 다양한 사용자의 입력을 표준 제어 지침으로 변환하는 표준 프롬프트 추출기(Standard Prompt Extractor, SPE)를 도입했습니다. 이는 규칙 기반 보상 및 평가에 필요한 것입니다. GPT 모델의 길이 제어 능력을 향상시키기 위해 Proximal Policy Optimization (PPO) 알고리즘을 응용하였습니다.

- **Performance Highlights**: 실험 결과, 강화 학습과 샘플 필터링을 적용함으로써, 널리 사용되는 요약 데이터세트인 CNNDM과 NYT에서 프롬프트 기반 길이 제어의 정확도가 크게 개선되었습니다. 또한 표준 프롬프트 추출기와 RL 조정 모델은 보지 못한 제어 프롬프트 템플릿에도 강력한 일반화 성능을 보였습니다.



### Soft Language Identification for Language-Agnostic Many-to-One End-to-End Speech Translation (https://arxiv.org/abs/2406.10276)
- **What's New**: 이 논문에서는 다수의 소스 언어를 하나의 타겟 언어로 번역하는 언어 비종속(end-to-end Many-to-One) 음성 번역 모델에 대한 새로운 방법을 제안합니다. 기존 모델의 성능을 유지하면서 선택된 특정 언어의 번역 성능을 향상시키기 위해 Linear Input Network(LIN) 기법을 도입했습니다. LIN은 초기화될 때 identity matrix로 설정되므로, 모델의 성능 저하 없이 능동적으로 사용될 수 있습니다.

- **Technical Details**: 다수의 소스 언어에서 하나의 타겟 언어로 번역하는 E2E ST(Speech Translation) 모델은 자동 음성 인식(ASR)과 텍스트 기반 기계 번역(MT)의 단점을 보완하기 위해 사용됩니다. 특히, Neural Transducer를 기반으로 한 모델은 스트리밍 환경에서도 뛰어난 성능을 발휘합니다. 이 논문은 LAMASSU 구조의 간편화 버전에 LIN을 적용한 LAMASSU-LIN 기법을 제안했습니다. LIN은 스피커 적응(speaker adaptation)에서 주로 사용되며, 특정 언어의 학습 데이터로만 훈련되어 다수의 언어에 대해 균일한 성능을 유지하면서도 선택된 언어의 번역 성능을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 LIN 기법은 특정 언어의 성능을 효과적으로 향상시키면서도 기존 모델의 언어 비종속 특성을 유지합니다. 특히, LIN 레이어를 identity matrix로 재설정하면 원래 모델로 쉽게 복원할 수 있어, 전반적 성능의 저하 없이 필요한 언어에서 향상된 성능을 제공합니다.



### ExHuBERT: Enhancing HuBERT Through Block Extension and Fine-Tuning on 37 Emotion Datasets (https://arxiv.org/abs/2406.10275)
Comments:
          accepted at INTERSPEECH 2024

- **What's New**: 이번 연구에서는 감정 인식 모델의 성능을 향상시키기 위해 두 가지 새로운 접근법을 제안했습니다. 첫째, 37개의 데이터셋과 150,907개의 샘플로 구성된 다문화, 다언어 음성 감정 데이터셋인 EmoSet++를 구축했습니다. 둘째, HuBERT 모델을 확장하여 ExHuBERT를 도입했습니다. 이 모델은 EmoSet++를 통해 백본 확장(backbone extension)과 미세 조정을 거쳐 성능이 최적화되었습니다.

- **Technical Details**: ExHuBERT는 HuBERT 모델에 기반하여 새로운 백본 블록 확장(backbone block expansion)을 포함합니다. 각 인코더 레이어와 그 가중치를 복사한 후, 첫 번째 복사본을 동결(freeze)하고 제로 초기화된 선형 레이어(zero-initialized linear layer)와 스킵 연결을 추가했습니다. 이를 통해 모델의 기능을 유지하고 후속 미세 조정을 위한 적응성을 보장합니다. 또한 EmoSet++ 데이터셋을 이용해 다언어, 다문화적인 데이터를 모델에 학습시켰습니다.

- **Performance Highlights**: ExHuBERT는 새로 제안된 접근법을 통해 여러 감정 인식(SER) 작업에서 새로운 벤치마크를 세웠습니다. 실험에서는 ExHuBERT가 이전 모델들보다 더 나은 성능을 보여주었습니다. 특히, 6개의 보지 못한 감정 데이터셋에서 우수한 일반화 성능을 입증했습니다.



### Beyond Words: On Large Language Models Actionability in Mission-Critical Risk Analysis (https://arxiv.org/abs/2406.10273)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Model)을 활용한 리스크 분석의 효율성을 조사했습니다. LLM, 특히 Retrieval-Augmented Generation(RAG) 및 미세 조정된(Fine-tuned) 버전의 효과를 검증했으며, 이는 리스크 분석 분야에서 처음 시도된 것입니다.

- **Technical Details**: 50개 이상의 중요한 임무 분석 데이터를 통해 수집한 수천 개의 시나리오를 바탕으로, 기준 GPT-3.5와 GPT-4 모델, 그리고 이 모델들의 RAG 및 미세 조정된 버전을 비교했습니다. 또한, 두 명의 인간 전문가와 세 명의 검토자가 5,000개의 시나리오 분석을 통해 모델 성능을 평가했습니다. 최종적으로 RAG 모델은 가장 낮은 환각율(hallucination rate)을 보이며 숨겨진 리스크를 효과적으로 발견했습니다.

- **Performance Highlights**: 인간 전문가들이 가장 높은 정확도를 보였지만, LLM은 더 빠르고 실행 가능성이 높았습니다. RAG 모델은 숨겨진 리스크를 잘 발견하며, 미세 조정된 모델은 정확성 면에서 탁월함을 보였습니다. 종합적으로, LLM은 리스크 분석에서 시간과 비용을 절약하면서도 전문가의 분석을 보완하는 유용한 도구임을 입증했습니다.



### Connected Speech-Based Cognitive Assessment in Chinese and English (https://arxiv.org/abs/2406.10272)
Comments:
          To appear in Proceedings of Interspeech 2024

- **What's New**: 본 논문은 연결된 음성 분석을 통해 인지 기능을 평가하는 방법을 탐구하기 위한 새로운 벤치마크 데이터셋과 예측 과제를 제공합니다. 데이터셋에는 다양한 수준의 인지 장애를 가진 중국어와 영어 사용자의 음성 샘플 및 임상 정보가 포함되어 있으며, 정상 인지 기능을 가진 개인도 포함되어 있습니다. 데이터는 연령과 성별에 따라 성향 점수 분석으로 신중하게 매칭되었습니다.

- **Technical Details**: 예측 과제는 경미한 인지 장애 진단 및 인지 테스트 점수 예측을 포함합니다. 모델 훈련의 균형성과 대표성을 보장하기 위해 데이터가 연령과 성별로 매칭되었으며, 언어 독립적(Language-agnostic)이고 비교 가능한 특징들을 사용하는 기본 예측 모델을 제공합니다. 기초 모델은 진단에서 59.2%의 가중되지 않은 평균 회상(Unweighted Average Recall)을, 점수 예측에서 2.89의 루트 평균 제곱 오차(Root Mean Squared Error)를 달성했습니다.

- **Performance Highlights**: 제안된 모델은 경미한 인지 장애 진단에서 59.2%의 평균 회상을, 인지 테스트 점수 예측에서 2.89의 루트 평균 제곱 오차를 기록했습니다. 이는 인지 기능 평가를 위해 음성을 디지털 바이오마커로 활용하는 가능성을 보여줍니다.



### A Conceptual Framework For Trie-Augmented Neural Networks (TANNS) (https://arxiv.org/abs/2406.10270)
Comments:
          17 Pages and 9 Figures

- **What's New**: Trie-Augmented Neural Networks (TANNs)라는 새로운 프레임워크가 도입되었습니다. 이 프레임워크는 트라이(structure) 데이터 구조와 신경망을 통합하여 투명성과 학습 효율성을 향상시킵니다. TANNs는 텍스트와 문서 분류 작업에 특히 효과적이며, 기존의 순환 신경망(RNN) 및 전방향 신경망(FNN)과 비교해 유사하거나 더 나은 성능을 보여줍니다.

- **Technical Details**: TANNs는 트라이 노드 안에 신경망을 배치하는 하향식 구조를 가지고 있습니다. 각 노드는 데이터의 특정 특성에 맞게 신경망을 포함하고 있어, 적응형 학습과 해석 가능성을 제공합니다. 이를 통해 데이터는 단계별로, 계층적으로 처리되며, RNNs, FNNs, CNNs와 같은 다양한 신경망을 지원합니다. 이 프레임워크는 문제를 더 작은 단위로 나누어 처리함으로써, 복잡한 결정 작업을 보다 정밀하고 확장 가능하게 만듭니다.

- **Performance Highlights**: XOR 문제와 AND/OR 논리 게이트 문제에서 TANNs는 전통적인 신경망보다 빠른 학습 속도와 수렴을 보여주었습니다. TANNs는 복잡한 데이터 관계를 효율적으로 학습할 수 있으며, 학습률 변화에도 안정적인 성능을 유지했습니다. 이는 TANNs의 설계가 복잡하고 대규모의 작업에 적합하다는 것을 나타냅니다.



### Markov Constraint as Large Language Model Surroga (https://arxiv.org/abs/2406.10269)
Comments:
          To appear at The 33rd International Joint Conference on Artificial Intelligence, IJCAI-24 (in press)

- **What's New**: 이번 논문에서는 'NgramMarkov'라는 새로운 제약 조건이 소개되었습니다. 이는 특수한 조건에서 텍스트를 생성하기 위한 Constraint Programming (CP) 기법의 변형입니다. 또한, 대형 언어 모델(LLM)에 의해 제공된 확률과 관련된 n-gram(단어의 시퀀스)을 사용하여 문장의 n-gram 확률 곱을 제한합니다. 이를 통해 문장이 최대일 가능성 추정을 넘어 LLM 분포를 포함하는 확장된 기초 Markov 제약 조건 전달자로 볼 수 있습니다.

- **Technical Details**: NgramMarkov는 'gliding threshold'를 사용하여 로컬 확률이 너무 낮은 n-gram을 거부하여 균형 잡힌 솔루션을 보장합니다. 또한 'look-ahead' 접근법과 결합하여 고정 길이의 수평선 내에서 받아들일 수 있는 문장으로 이어질 가능성이 매우 낮은 n-gram을 제거할 수 있습니다. 실험 결과에 따르면 생성된 텍스트는 LLM의 perplexity 함수와 유사한 평가를 받았으며, NgramMarkov 제약을 사용하면 후보 문장의 수가 크게 줄어들고 계산 시간이 개선됩니다.

- **Performance Highlights**: 실험 결과, 4-gram 사용 시 기존 5-gram보다 더 큰 말뭉치나 작은 n-gram을 사용할 수 있게 되어 실세계 문제를 처음으로 해결했습니다. 이는 문장을 생성하는 동안 'poor' 솔루션을 효율적으로 필터링할 수 있는 LLM 기반 제약을 구현함으로써 가능해졌습니다.



### Unused information in token probability distribution of generative LLM: improving LLM reading comprehension through calculation of expected values (https://arxiv.org/abs/2406.10267)
Comments:
          7 pages, 1 figure

- **What's New**: 이번 연구는 LLM(Large Language Model)의 텍스트 디코딩 성능을 개선하기 위해 토큰 확률을 조작하는 방법을 탐구합니다. 특히, 큰 온도(temp) 값을 사용하여 로그 확률(logits)을 조정함으로써 인간의 판단과의 상관관계가 높은 SummEval 데이터셋에서 성능을 크게 향상시켰습니다.

- **Technical Details**: 연구에서는 먼저 기디(greedy) 디코딩과 다음 토큰 분포에 대한 기대값을 비교하였습니다. 이때 높은 온도 값(T=10)을 사용하여 점수의 엔트로피를 증가시켰습니다. 또한, 주어진 프롬프트에 대해 가장 가능성이 높은 생성들을 모두 살펴보는 확률 기반 트리 샘플링 알고리즘을 사용했습니다.

- **Performance Highlights**: SummEval 데이터셋에서는 7B Mistral 모델의 성능이 6-8%에서 13-28%로, Mixtral 모델의 성능이 20%-46%에서 37%-56%로 향상되었습니다. 이는 GPT-4의 결과를 두 가지 메트릭에서 능가하는 수준입니다. 또한, 작은 크기와 양자화된(quantized) LLM에서도 일관되게 성능이 향상됐으며, 최대 4.4배의 개선을 보였습니다.



### COVID-19 Twitter Sentiment Classification Using Hybrid Deep Learning Model Based on Grid Search Methodology (https://arxiv.org/abs/2406.10266)
Comments:
          14 pages, 6 figures, 11 tables

- **What's New**: 현대 시대에 소셜 미디어 플랫폼은 이용자들이 기여한 방대한 양의 소셜 데이터를 수집합니다. 본 논문에서는 COVID-19 백신에 대한 망설임을 조사하기 위해 8가지 다른 하이브리드 딥러닝(Hybrid Deep Learning) 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 Twitter의 COVID-19 데이터셋을 활용해 감정 예측(Sentiment Prediction)을 수행합니다. 임베딩(Embedding), 딥러닝 모델(Deep Learning Model) 및 그리드 검색 알고리즘(Grid Search Algorithm)을 사용하여 감정 분석을 진행하였습니다. 특히, BERT, CNN, 그리고 그리드 검색(GS)의 조합이 가장 높은 정확도를 보였으며, GloVe, BiLSTM, CNN, GS의 조합도 높은 성능을 보였습니다.

- **Performance Highlights**: 제안된 모델은 98.86%의 정확도를 기록하여 다른 모델들보다 우수한 성능을 보여주었습니다. 이는 기존 연구와 비교하여 2.11%에서 14.46%까지 정확도가 향상되었음을 나타냅니다. 또한, 시간이 지남에 따라 COVID-19 백신에 대한 공공 감정이 개선되고 있는 것으로 나타났습니다.



### Improving Language Models for Emotion Analysis: Insights from Cognitive Scienc (https://arxiv.org/abs/2406.10265)
Comments:
          in French language, JEP-TALN-RECITAL 2024, Jul 2024, Toulouse, France

- **What's New**: 이 논문에서는 감정 분석을 위해 언어 모델을 개선하기 위해 인지 과학 연구와 감정 및 의사소통 연구를 활용할 것을 제안합니다. 주요 심리학 및 인지 과학에서의 감정 이론과 이들 연구를 기반으로 하는 감정 주석(Annotation) 방식 등을 소개합니다.

- **Technical Details**: 먼저, 심리학과 인지과학에서의 주요 감정 이론을 설명하고, 이후 자연어 처리에서의 주요 감정 주석 방법 및 이들의 심리학 이론과의 연관성을 소개합니다. 또한, 인지 화용론(Cognitive Pragmatics)에서 감정적 의사소통의 두 가지 주요 분석 유형을 제시합니다.

- **Performance Highlights**: 인지 과학 연구를 기반으로 언어 모델을 개선하는 방향을 제안하며, 이는 새로운 주석 체계 및 감정 이해를 위한 벤치마크 구축에 기여할 가능성을 논의합니다. 이를 통해 다양한 인간 감정 및 의사소통의 측면을 고려한 언어 모델의 성능 향상을 기대할 수 있습니다.



### FoodSky: A Food-oriented Large Language Model that Passes the Chef and Dietetic Examination (https://arxiv.org/abs/2406.10261)
Comments:
          32 pages, 19 figures

- **What's New**: 새로운 연구로 식품 관련 대규모 언어 모델 (Large Language Model, LLM)인 FoodSky가 도입되었습니다. 이 모델은 다양한 작업에서의 식품 인지와 추론을 가능하게 하여 요리 생성, 식이 추천, 질병과 식이의 상관관계 발견 및 이해를 돕습니다. 특히 중국 요리의 복잡성과 특성을 고려하여 종합적인 중국 요리 자료 집합인 FoodEarth를 구축하여 FoodSky의 성능을 높였습니다.

- **Technical Details**: FoodSky는 두 가지 주요 메커니즘을 통해 성능을 강화합니다. 첫째, 주제 기반 선택 상태 공간 모델 (Topic-based Selective State Space Model, TS3M)을 통해 세밀한 식품 의미론을 캡처합니다. 둘째, 계층적 주제 검색 증강 생성 (Hierarchical Topic Retrieval Augmented Generation, HTRAG) 메커니즘을 도입하여 컨텍스트에 민감한 식품 관련 텍스트를 생성합니다. FoodEarth는 다양한 권위 있는 출처로부터 수집된 811K의 지도 데이터로 구성되어 모델이 복잡하고 다양한 도메인 지식을 이해할 수 있도록 도와줍니다.

- **Performance Highlights**: FoodSky는 중국 국가 요리사 시험과 국가 영양사 시험에서 각각 67.2%와 66.4%의 정확도로 일반적인 LLM을 뛰어넘는 성능을 보였습니다. 이는 FoodSky가 요리 창의성을 증진하고 건강한 식습관을 촉진하는 데 도움이 되며, 식품 분야의 복잡한 실제 문제를 해결하는 도메인 특화 LLM의 새로운 기준을 설정한다고 할 수 있습니다. 또한 FoodSky는 다양한 문화적 배경의 식품 관련 정보를 효율적으로 처리할 수 있습니다.



### Flextron: Many-in-One Flexible Large Language Mod (https://arxiv.org/abs/2406.10260)
- **What's New**: Flextron은 대규모 언어 모델(LLM)을 다양한 배포 시나리오에 맞게 쉽게 커스터마이징할 수 있도록 고안된 네트워크 아키텍처 및 사후 학습 모델 최적화 프레임워크입니다. Flextron 아키텍처는 내부에 탄력적으로 조정 가능한 구조를 사용하여 사용자 정의 지연 시간 및 정확도 목표에 빠르게 적응할 수 있습니다. 이는 추가적인 미세 조정 없이도 가능합니다. 또, 입력 적응형(input-adaptive)이며, 성능과 효율성을 향상시키기 위해 토큰을 하위 네트워크로 자동 라우팅할 수 있습니다.

- **Technical Details**: Flextron은 주의(attention) 및 피드포워드(feed-forward) 레이어에 이론적으로 확장된 MoE(Mixture-of-Expert) 네트워크의 아이디어를 도입하였습니다. 이 네트워크는 이질적인 전문가들로 구성되며, 다양한 크기의 네트워크를 지원하는 내부에 탄력적인 구조를 갖추고 있습니다. Flextron 모델은 추가적인 미세 조정 없이 여러 모델을 하나의 배포 시점에서 제공할 수 있습니다. 우리는 표준 학습된 LLM(GPT-3, Llama-2)을 Flextron으로 효율적으로 변환할 수 있는 프레임워크를 제안합니다. 또한, 적응적 계산을 구현하기 위해 정적 및 동적 라우팅 알고리즘을 제안하며, 최적의 하위 네트워크를 자동으로 선택하는 데 사용됩니다.

- **Performance Highlights**: Flextron은 GPT-3 및 Llama-2 LLM을 대상으로 평가되었으며, 여러 최첨단 엘라스틱 네트워크 및 독립적으로 학습된 여러 변형 모델에 비해 우수한 성능을 입증했습니다. 이는 원래 사전 학습에 사용된 토큰의 단 7.63%만을 사용한 단일 사전 학습 실행으로 달성되었습니다. Flextron은 추가적인 미세 조정 없이 사용자 정의 지연 시간 및 정확도 목표를 충족시키기 위해 빠르게 조정할 수 있으며, 훈련된 모델을 효율적으로 변환할 수 있는 샘플 효율적인 학습 방법을 제공합니다.



### Optimal synthesis embeddings (https://arxiv.org/abs/2406.10259)
- **What's New**: 이번 논문에서는 주어진 단어 집합에 대해 모든 구성 요소의 벡터 표현과 동일한 거리에 있는 새로운 벡터를 만들어내는 단어 임베딩(composition) 방법을 제안합니다. 이 방법은 정적인 단어 표현뿐만 아니라 문맥화된 단어 표현에도 적용 가능하며, 문장 표현뿐만 아니라 순서에 구애받지 않는 단어 집합의 표현도 학습할 수 있습니다.

- **Technical Details**: 해당 방법은 주어진 단어 벡터와 동일한 거리에 있는 벡터 존재 조건을 이론적으로 규명하고, 이를 해결하기 위한 방법을 도출합니다. 두 가지 시나리오—단어 분류를 위한 데이터 증강과 문장 분류를 위한 문장 임베딩 생성—에서 방법의 성능을 평가하였습니다. 실험은 다양한 임베딩 구성 방법과 비교하여 수행되었습니다.

- **Performance Highlights**: 노력한 결과로, 제안된 방법이 단순한 언어적 특징을 포착하는 탐색 작업(probing tasks)에서 뛰어난 성능을 보였으며, 데이터 증강과 문장 분류 작업에서도 기존 방법들보다 높은 정확도를 기록했습니다.



### Curating Grounded Synthetic Data with Global Perspectives for Equitable A (https://arxiv.org/abs/2406.10258)
- **What's New**: 본 논문에서는 새로운 합성 데이터 세트 생성 방법을 소개합니다. 이 방법은 실제 세계의 다양성을 기반으로 하며, 전략적 다양화를 통해 데이터 세트를 풍부하게 만듭니다. 이 데이터는 12개 언어로 작성된 125개국의 뉴스 기사들을 사용하여 합성되었으며, 주제 다양화와 번역, 요약을 통해 전통적 데이터 세트의 대표성 문제를 해결합니다.

- **Technical Details**: 이 논문은 Named Entity Recognition (NER) 분야에서 합성 데이터 론칭의 영향을 강조합니다. GLiNER 모델을 사용하여 기존의 자원 소모가 큰 LLM 기반 NER 모델과 달리, 효율적인 Bidirectional Language Models (BiLM)인 BERT와 deBERTa를 활용하여 다양한 엔티티 유형을 병렬로 예측하고 양방향 문맥 처리를 수행합니다. 이를 통해 125개국, 12개 언어의 뉴스 기사들을 번역, 요약, 분류하여 5,049개의 샘플로 구성된 데이터 세트를 생성했습니다.

- **Performance Highlights**: 제안된 합성 데이터 세트를 NER 벤치마크에서 테스트한 결과, 성능이 최대 7.3% 향상되었습니다. 이러한 성과는 합성 데이터가 다양한 전세계 데이터 소스를 미러링하며, 이를 통해 AI 모델의 일반화 능력을 크게 향상시킨다는 것을 입증합니다.



### Explicit Word Density Estimation for Language Modelling (https://arxiv.org/abs/2406.10256)
Comments:
          Master's thesis

- **What's New**: 이번 연구에서는 NeuralODEs와 Normalizing Flows의 연속적 대안을 기반으로 한 새로운 언어 모델 패밀리를 제안했습니다. 이러한 접근을 통해 일부 기존 기준(l)모델보다 성능이 향상되었습니다.

- **Technical Details**: 기존 연구에서는 Softmax layer가 결과 매트릭스(matrix)의 rank에 상한을 두어 모델의 표현력을 제한한다고 밝혔습니다. 제안된 모델은 NeuralODEs와 Normalizing Flows의 연속 아날로그를 사용해 이러한 제한을 극복하려고 합니다. NeuralODE는 Residual Networks의 연속적 대안으로 소개된 신경망 신경망(unsigned variable)입니다.

- **Performance Highlights**: 제안된 모델은 LSTM 기반 언어 모델보다 일부 기준 성능에서 개선된 결과를 보였습니다.



### WarCov -- Large multilabel and multimodal dataset from social platform (https://arxiv.org/abs/2406.10255)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 연구는 2022년 폴란드의 인기 소셜 미디어 플랫폼에 게시된 COVID-19 팬데믹과 우크라이나 전쟁에 관한 3,187,105개의 게시물을 수집한 새로운 데이터셋을 소개합니다. 이 데이터셋은 전처리된 텍스트와 이미지가 포함되어 있어 멀티모달 인식 작업에도 활용될 수 있습니다. 라벨은 게시물에 동반된 해시태그를 사용하여 생성되었습니다.

- **Technical Details**: WarCov 데이터셋은 텍스트 데이터를 XLM-RoBERTa 기반의 clips/mfaq 모델을 이용해 워드 임베딩(word embeddings)으로, 이미지 데이터를 ResNet-18 아키텍처를 사용하여 임베딩으로 제공됩니다. 텍스트와 이미지 데이터는 PCA를 통해 동일한 수의 컴포넌트로 전처리됩니다. 데이터셋은 GitHub을 통해 오픈 소스로 제공되며(CC BY-NC-SA 4.0 라이선스), 멀티라벨 및 멀티모달 인식을 위한 벤치마크로 활용할 수 있습니다.

- **Performance Highlights**: 이 데이터셋은 최소 50개의 카테고리에 속하는 멀티라벨 분류 작업을 테스트하기에 적합하며, 텍스트와 이미지 데이터를 결합하여 모달리티 간의 지식 전이를 연구할 수 있습니다. 이미지 데이터만으로도 컴퓨터 비전 연구에 활용될 수 있는 충분한 크기를 가지고 있습니다. 초기 실험 결과, 데이터셋이 다양한 인식 작업에서 높은 복잡성과 품질을 보여주는 것으로 확인되었습니다.



### Towards Signal Processing In Large Language Models (https://arxiv.org/abs/2406.10254)
Comments:
          12 pages, 3 figures

- **What's New**: 이 논문에서는 신호처리를 대형 언어 모델(LLM) 내부에서 적용하는 아이디어를 소개합니다. 최근 생성 AI의 폭발적인 성장과 함께, 이 연구는 신호처리 분야와 LLM을 연결하는데 기여할 수 있습니다. 연구진은 GPT와 같은 아키텍처에서 신호처리를 통합함으로써, 더 빠른 수렴 및 성능 향상을 달성했다고 주장합니다.

- **Technical Details**: 기존의 푸리에 변환(Fourier Transform) 개념을 확장하여, 중간 활성 신호를 시간-주파수(time-frequency) 표현으로 변환하면서 LLM 구성 요소를 학습합니다. 이 새로운 시간-주파수 표현을 사용하여 각 토큰의 활성 신호를 분해, 필터링 및 재구성함으로써, 이전 문맥을 기반으로 다음 토큰을 예측합니다. 모든 구성 요소는 처음부터 학습되어 최적의 시간-주파수 표현을 찾도록 합니다.

- **Performance Highlights**: 연구진은 GPT와 같은 아키텍처에서 제안한 방법이 같은 에포크(epoch) 동안 학습되었을 때 극히 적은 수의 추가 파라미터로 더 빠른 수렴과 성능 향상을 달성했음을 입증했습니다. 이는 신호처리 개념을 LLM 학습에 적용함으로써, 큰 언어 모델의 사전 훈련을 보다 효과적으로 만들 수 있음을 시사합니다.



### D\'eveloppement automatique de lexiques pour les concepts \'emergents : une exploration m\'ethodologiqu (https://arxiv.org/abs/2406.10253)
Comments:
          in French language. JADT 2024

- **What's New**: 본 논문은 비기술적 혁신에 중점을 둔 신흥 개념 중심의 어휘집 (lexicon) 개발을 소개합니다. 여러 도메인에 걸쳐 일반화할 수 있는 모델을 확립하기 위해 인간 전문 지식, 통계 분석, 기계 학습 기법을 결합한 4단계 방법론을 소개합니다.

- **Technical Details**: 이 방법론은 주제별 말뭉치(corpus)의 생성, Gold Standard Lexicon(골드 스탠다드 어휘집)의 개발, 교육용 말뭉치(annotation 및 준비), 그리고 새로운 용어를 식별하기 위한 학습 모델의 구현을 포함합니다.

- **Performance Highlights**: 결과는 해당 접근 방식의 견고성과 관련성을 입증하며, 다양한 맥락에서의 적응성과 어휘 연구에의 기여를 강조합니다. 이 개발된 방법론은 개념적 분야에서 적용 가능성을 약속합니다.



### The Impact of Quantization on Retrieval-Augmented Generation: An Analysis of Small LLMs (https://arxiv.org/abs/2406.10251)
Comments:
          Accepted to the IR-RAG Workshop at SIGIR 2024

- **What's New**: 이번 논문에서는 포스트 트레이닝 양자화 (Post-training quantization)가 소형 대형 언어 모델(Large Language Models, LLMs)의 성능에 미치는 영향을 살펴봅니다. 특히 양자화가 소형 LLM의 장기 문맥 분석 능력에 어떻게 영향을 미치는지 연구하고, 이를 위해 복수의 문서를 사용하는 개인화(personalization) 영역에서 이를 평가합니다. 연구 결과, 소형 LLM이 특정 작업을 잘 수행하면 양자화가 그 성능을 크게 저해하지 않는 것으로 나타났습니다.

- **Technical Details**: 이 연구에서는 LLaMA2-7B와 같은 baseline 모델을 포함하여 여러 7B 및 8B LLMs를 조사했습니다. LaMP 벤치마크에서 두 가지 개인화 데이터셋을 선택하여 FP16과 양자화된 INT4 버전의 성능을 비교했습니다. Activation-aware Weight Quantization (AWQ) 방법을 사용하여 양자화하였으며, LaMP-3과 LaMP-5U 데이터셋을 사용했습니다. LaMP-3은 제품 리뷰와 그 점수로 구성되며, LaMP-5U는 학술 논문의 초록을 기반으로 제목을 생성하는 작업입니다.

- **Performance Highlights**: 양자화된 모델과 원래 FP16 모델 간의 성능 차이를 비교한 결과, 모델과 작업에 따라 양자화의 영향이 다르게 나타났습니다. OpenChat 모델의 경우 성능 저하가 거의 보이지 않았지만, LLaMA2는 양자화에 더 민감한 것으로 나타났습니다. 이는 양자화된 소형 LLM이 효율성이 중요한 RAG 파이프라인에 적합할 수 있음을 시사합니다.



### On the Worst Prompt Performance of Large Language Models (https://arxiv.org/abs/2406.10248)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 성능이 프롬프트 표현 방식에 민감하다는 문제를 해결하기 위해, 새로운 벤치마크인 RobustAlpacaEval을 소개합니다. 이 벤치마크는 의미적으로 동등한 케이스-레벨 질의를 포함하며, 모델 성능의 최저 한계를 평가하기 위해 '최악의 프롬프트 성능'을 강조합니다.

- **Technical Details**: RobustAlpacaEval은 TinyAlpaceEval에서 출발하여 각각의 질의에 대해 10개의 패러프레이즈(paraphrases)를 생성했습니다. 이는 GPT-4를 통해 자동으로 이루어졌으며, 각 패러프레이즈는 의미적 일관성과 유창성을 유지하기 위해 수동으로 검토 및 수정되었습니다. 평균 길이-정규화 에디트 거리(edit distance)는 0.7234로, 이는 상당한 다양성을 나타냅니다.

- **Performance Highlights**: 실험 결과, ChatGPT 및 Llama, Mistral, Gemma 계열의 6가지 오픈 소스 LLM에서 모델 성능의 큰 변동이 발견되었습니다. 예를 들어, Llama-2-70B-chat 모델의 경우 최악과 최고 성능 간의 차이가 45.48%에 달했고, 최악의 성능은 9.38%까지 하락했습니다. 또한, 최악의 프롬프트를 식별하는 것이 모델 간에 일관성이 없음을 강조하여, 모델-독립적 및 모델-의존적 특성 모두가 예측에 적합하지 않음을 보여줍니다.



### QCQA: Quality and Capacity-aware grouped Query Attention (https://arxiv.org/abs/2406.10247)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 자동회귀 추론에서 발생하는 과도한 메모리 요구 문제를 해결하기 위해 Quality and Capacity-Aware Grouped Query Attention (QCQA)라는 새로운 접근 방식을 제안합니다. QCQA는 진화 알고리즘을 사용하여 최적의 쿼리 헤드 그룹화를 식별하며, 보다 계산적으로 효율적이고 저렴한 적합도 함수를 통해 이를 수행합니다. 이를 통해 LLM의 정확도와 KV-cache 용량 간의 상충 관계(tradeoff)를 개선하였습니다.

- **Technical Details**: 이 연구는 기존의 Multi-Query Attention (MQA)나 Grouped Query Attention (GQA)이 텍스트 생성 품질에 영향을 미치는 반면, KV-cache 크기를 줄이는 데 초점을 맞췄다는 점에서 출발합니다. 구체적으로, QCQA는 쿼리 헤드의 최적 그룹화를 식별하는 진화 알고리즘을 채택하여, 이러한 문제를 해결합니다. 이 알고리즘은 정확도 손실을 신뢰성 있게 나타내는 간단하고 저렴한 적합도 추정 함수를 사용합니다. QCQA는 GQA와 달리, 임의 크기의 쿼리 헤드 그룹을 만들 수 있습니다.

- **Performance Highlights**: QCQA는 Llama2 $7\,B 모델에서 비세분화(fine-tuning) 상태에서도 GQA 대비 20% 더 높은 정확도를 나타냈습니다. 세부 조정 후에도 유사한 KV-cache 크기를 유지하면서 QCQA는 GQA보다 10.55% 더 높은 정확도를 제공합니다. 또한, QCQA는 GQA와 동일한 정확도를 달성하기 위해 40% 적은 KV-cache 크기를 요구합니다. 이는 LLM 추론에서 KV-cache 최적화의 새로운 기준이 될 수 있습니다.



### Early Detection of Misinformation for Infodemic Management: A Domain Adaptation Approach (https://arxiv.org/abs/2406.10238)
- **What's New**: 이번 논문에서는 질병 확산 중 참 정보와 거짓 정보를 대량으로 퍼뜨리는 '인포데믹'의 문제를 다룹니다. 기존의 허위 정보 탐지 방법은 레이블이 달린 데이터를 필요로 하지만, 초기 인포데믹 단계에서는 레이블이 없는 정보가 대부분입니다. 이를 해결하기 위해, 다른 도메인의 레이블이 달린 정보를 활용하여 인포데믹 도메인에서 허위 정보를 탐지하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 'covariate shift'와 'concept shift'를 모두 해결하는 데 중점을 둡니다. 일반적으로 도메인 적응(domain adaptation) 기법을 사용하여 한 도메인에서 학습한 모델이 다른 도메인에서도 성능을 발휘하도록 합니다. 데이터의 분포 차이 및 조건부 확률 차이를 줄여 모델의 성능을 향상시키는 방식입니다. 실험에서는 두 개의 실제 데이터셋을 사용하여 새로운 방법의 효과를 검증하였습니다.

- **Performance Highlights**: 실제 데이터셋을 활용한 실험 결과, 제안된 새로운 방법이 최신 허위 정보 탐지 방법 및 일반적인 도메인 적응 기법보다 성능이 뛰어남을 보여주었습니다. 이로써 초기 인포데믹 단계에서 더 효과적으로 허위 정보를 탐지하고, 공중 보건에 미치는 부정적 영향을 줄일 수 있음을 입증했습니다.



### mDPO: Conditional Preference Optimization for Multimodal Large Language Models (https://arxiv.org/abs/2406.11839)
- **What's New**: 최근 연구는 대형 언어 모델(LLM)을 멀티모달(multimodal) 시나리오에 적용한 결과 일관된 성능 개선을 달성하기 어려운 문제를 확인했습니다. 이를 해결하기 위해 mDPO라는 새로운 멀티모달 직접 선호 최적화(mDPO)를 제안합니다. 이 방법은 언어만 중시하는 선호를 막고 이미지 조건을 최적화합니다. 또한, 선택된 응답의 가능성 저하를 막기 위해 보상 앵커를 도입했습니다. 두 가지 이상의 멀티모달 LLM과 다양한 벤치마크를 통해 mDPO가 멀티모달 선호 최적화를 효과적으로 해결하고 모델 성능을 크게 향상시킴을 입증했습니다.

- **Technical Details**: 기존의 직접 선호 최적화(DPO)는 텍스트 기반 모델에서는 성공적이었지만, 멀티모달에서는 일관된 성능 개선이 어려웠습니다. mDPO는 이미지 조건을 포기하지 않고, 언어와 이미지를 함께 최적화하여 문제를 해결합니다. mDPO는 선택된 응답이 긍정적인 보상을 받도록 보상 앵커를 도입하여 응답의 가능성 감소를 방지합니다. 이를 통해 모델이 질문에만 기반한 응답을 생성하는 경향을 막고, 시각적 정보를 효과적으로 활용할 수 있게 합니다.

- **Performance Highlights**: Bunny-v1.0-3B와 LLaVA-v1.5-7B 두 가지 모델을 통해 실험한 결과, mDPO는 기존 DPO보다 멀티모달 시나리오에서 우수한 성능을 보였습니다. MMHalbench, Object HalBench, AMBER 등의 벤치마크를 통해 mDPO가 환각(hallucination)을 효과적으로 줄이고 모델의 이미지 이해 능력을 향상시키는 것을 확인했습니다. mDPO는 텍스트 편향을 줄이고, 다양한 모델과 데이터 스케일에서 일관된 성능 개선을 달성했습니다.



### On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning (https://arxiv.org/abs/2406.11823)
Comments:
          17 pages, 8 figures

- **What's New**: 최근 언어 및 비전 어시스턴트(LVA, Language-Vision Assistants) 모델에서 투명성이 부족해 보다 넓은 연구와 재현성이 제한되는 문제가 발생했습니다. 본 연구는 모델 크기와 데이터 중요성 간의 균형을 재정립하고 효율적인 모델을 설계하는 것을 목표로 합니다. 이를 위해 데이터셋을 전략적으로 구성하고, 비전 모듈을 최적화하며, 지도 학습 기술을 개선하여 추론 성능을 대폭 향상시킵니다. 연구 결과는 전면 오픈소스로 공개될 예정입니다.

- **Technical Details**: 본 연구의 핵심 기여는 다음과 같습니다:
1. 효율성과 재현성: 오픈소스 데이터로 학습된 효율적이고 확장 가능한 모델 아키텍처인 'Elva'를 제시합니다.
2. 실증적 검증: Elva의 주요 구성 요소의 효과를 검증하기 위한 철저한 실험을 수행하였습니다.
3. 모델 확장성: Elva의 다양한 버전을 개발하여 그 확장성과 적응성을 입증합니다.
4. 데이터셋 기여: 새로운 데이터셋인 CORD-Instruct와 Parsing-Bench를 도입하여 Elva의 성능을 평가합니다.
5. 오픈소스 이니셔티브: 커뮤니티 연구를 촉진하고 모델 재현성을 보장하기 위해 학습된 모델과 데이터셋을 공개합니다.

- **Performance Highlights**: 다양한 LLM (Large Language Model)과 비전 모듈을 실험하여 160M에서 13B 파라미터까지 모델 최적화를 진행하였고, 고해상도 작업을 처리하면서도 낮은 추론 비용을 유지하는 Elva 모델을 설계했습니다. 표로 제시된 바와 같이 LLaVA 네트워크의 자원 사용과 관련하여 DocVQA 및 ChartQA 테스트셋에서 LLaVA-1.5 모델은 상대적으로 관리 가능한 컴퓨팅 비용을 보였으나, LLaVA-NeXT 모델은 높은 자원 요구를 나타냈습니다.



### DataComp-LM: In search of the next generation of training sets for language models (https://arxiv.org/abs/2406.11794)
Comments:
          Project page: this https URL

- **What's New**: DCLM (DataComp for Language Models)라는 새로운 벤치마크를 소개합니다. 이는 Common Crawl에서 추출한 240T 토큰의 표준화된 코퍼스, OpenLM 프레임워크 기반의 효과적인 사전 훈련 레시피, 53개의 다운스트림 평가 스위트를 포함합니다. DCLM 벤치마크 참가자는 데이터 제거, 필터링 및 데이터 혼합 등의 전략을 모델 스케일(412M에서 7B 파라미터)에서 실험할 수 있습니다.

- **Technical Details**: DCLM-Baseline 데이터셋을 구성하기 위해 모델 기반 필터링이 주요한 역할을 한다는 것을 발견했습니다. 결과적으로 MMLU에서 64%의 5-shot 정확도를 기록하며, 기존의 MAP-Neo 대비 40% 적은 컴퓨팅 자원을 사용하여 6.6% 포인트 향상된 성능을 보여주었습니다. 이 모델은 Mistral-7B-v0.3 및 Llama 3 8B 모델과 비교해도 비슷한 성능을 보이며, Llama 3 8B보다 6.6배 적은 컴퓨팅 자원을 사용해 53개의 자연어 이해(task)에서 유사한 성능을 발휘합니다.

- **Performance Highlights**: DCLM-Baseline을 통해 학습된 7B 파라미터 언어 모델은 2.6T 트레이닝 토큰을 사용하여 MMLU의 5-shot 측정에서 64%의 정확도를 얻었습니다. 이는 40% 적은 컴퓨팅 자원을 사용하여 기존의 MAP-Neo 대비 6.6% 포인트 향상된 성능을 의미합니다. 또한, 53개의 자연어 이해 task에서 평균적인 성능이 비슷하게 유지되어 Llama 3 8B보다 6.6배 적은 컴퓨팅 자원으로 훌륭한 성과를 보여줍니다.



### Split, Unlearn, Merge: Leveraging Data Attributes for More Effective Unlearning in LLMs (https://arxiv.org/abs/2406.11780)
- **What's New**: 최근 대형 언어 모델(LLMs)의 안전성을 개선하기 위해 유해한 행동과 지식을 직접 제거하는 기법인 '기계 학습 취소(Machine Unlearning)'가 주목받고 있습니다. 이번 연구에서는 이러한 학습 취소의 효과를 증폭시킬 수 있는 프레임워크 'SPlit, UNlearn, MerGE (SPUNGE)'를 제안했습니다.

- **Technical Details**: SPUNGE는 데이터 속성을 활용하여 학습 취소 데이터를 특정 속성 값에 따라 하위 집합(subset)으로 나누고, 각 하위 집합을 개별적으로 학습 취소하며, 마지막으로 학습 취소된 모델들을 합칩니다. 이렇게 함으로써 SPUNGE는 전체적인 LLM의 성능을 유지하면서 다양한 학습 취소 기법의 효과를 크게 향상시킵니다.

- **Performance Highlights**: SPUNGE는 최신 LLM들(예: Llama2-7b, Zephyr-7b-beta)에서 두 가지 최근 학습 취소 기법의 성능을 현저히 개선시켰습니다. 예를 들어, SPUNGE는 ToxiGen 벤치마크에서 생성된 유해 텍스트 비율을 최대 32%로 감소시키고, 바이오 보안(biosecurity) 지식을 11.8% 감소시키며, 사이버 보안(cybersecurity) 지식을 4% 감소시켰습니다. 또한, SPUNGE는 10개의 표준 학술 벤치마크에서 LLM의 일반적인 성능을 유지했습니다.



### GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities (https://arxiv.org/abs/2406.11768)
Comments:
          Project Website: this https URL

- **What's New**: 이번 논문에서 제안된 GAMA는 고급 오디오 이해와 복잡한 추론 능력을 가진 새로운 다목적 대형 오디오-언어 모델(LALM)입니다. GAMA는 다양한 오디오 표현을 통합하여 오디오 이해와 추론 능력을 향상시켰습니다.

- **Technical Details**: GAMA는 맞춤형 Audio Q-Former와 다층 애그리게이터(aggregator)를 사용하여 오디오 인코더의 여러 레이어에서 기능을 집계합니다. GAMA는 대규모 오디오-언어 데이터세트로 미세 조정(fine-tuning)되어 오디오 이해 능력을 배양합니다. 또한 복잡한 오디오 추론을 위한 Instruction-Tuning 데이터를 사용하여 복잡한 추론 능력을 갖추도록 하였습니다. GAMA는 CompA-R 테스트 데이터세트를 통해 오디오에 대한 개방형 질문-응답 기능을 평가받습니다.

- **Performance Highlights**: 자동화된 평가 및 전문가의 평가를 통해 GAMA가 다양한 오디오 이해 작업에서 기존 LALM들보다 최대 84% 향상된 성능을 보였습니다. CompA-R로 Instruction-Tuning된 GAMA는 복잡한 추론 및 지침 따르기에서 뛰어난 성능을 입증하였습니다.



### STAR: SocioTechnical Approach to Red Teaming Language Models (https://arxiv.org/abs/2406.11757)
Comments:
          8 pages, 5 figures, 5 pages appendix. * denotes equal contribution

- **What's New**: 새로운 연구에서는 대규모 언어 모델의 안전성을 검토하는 적군 공격(red teaming) 방법론을 개선한 STAR(사회기술 프레임워크)를 소개합니다. STAR는 인간 적군 공격자들에게 매개변수화된 지침을 제공하여 리스크 표면을 더 넓게 포괄하고 모델 실패에 대한 심도 있는 통찰을 제공합니다. 또한, 선정된 인구집단을 매칭하여 특정 그룹에 대한 해를 평가하고 신호의 품질을 향상시키는 독특한 중재 단계를 도입합니다.

- **Technical Details**: STAR는 두 가지 주요 혁신을 통해 현재의 적군 공격 방법론을 발전시킵니다. 첫째, 매개변수화된 지침(parameterised instructions)을 통해 조사 목표를 명확히 하고 리스크 표면을 포괄적으로 탐색할 수 있도록 합니다. 둘째, 인구통계학적 매칭을 통해 특정 그룹에 대한 해를 더욱 민감하게 평가하고 다양한 관점을 반영하여 라벨 신뢰성을 향상시키는 중재 단계를 추가합니다.

- **Performance Highlights**: STAR는 전문가와 일반 인구집단의 다양한 평가를 활용하여 보다 신뢰성 있는 신호를 제공합니다. 예를 들어, 특정 인구집단에 대한 증오 발언이나 차별적 고정관념을 평가할 때, 해당 집단의 경험을 직접 반영하는 사용자들의 평가를 우선시합니다. 이를 통해 공격의 다양성과 신뢰성이 향상되는 것을 목표로 합니다.



### Multi-Layer Ranking with Large Language Models for News Source Recommendation (https://arxiv.org/abs/2406.11745)
Comments:
          Accepted by the SIGIR 2024. arXiv admin note: text overlap with arXiv:2305.04825

- **What's New**: 뉴스 이벤트에 대한 신뢰할 수 있는 정보원을 찾기 위해, 이전에 인용된 발언을 기반으로 신뢰할 수 있는 소스를 식별하는 새로운 전문가 추천 시스템을 제안합니다. 이를 위해 총 23,571개의 인용-화자 쌍으로 구성된 NewsQuote라는 새로운 데이터셋을 만들었습니다. 특히, 대규모 언어 모델(Large Language Models, LLMs)를 활용한 다층 랭킹(Multi-layer Ranking) 프레임워크를 통해 추천 성능을 크게 향상시켰습니다.

- **Technical Details**: 데이터셋 생성은 2019년 11월부터 2020년 8월 사이에 출판된 기사에서 수집된 인용-화자 쌍으로 이루어졌습니다. 뉴스 기사를 문장 단위로 분할하고, 사전 학습된 BERT 기반의 의미 역할 라벨링 모델을 사용하여 인용 트리거 단어를 추출했습니다. 이를 통해 352개의 인용 트리거 단어를 식별하고, 주어와 목적어가 포함된 문장만을 선택했습니다. 수집된 데이터셋은 In-context Learning 기반의 LLM 랭커를 사용하여 다층 랭킹 기반 필터링 메커니즘을 통해 추천 성능을 높였습니다.

- **Performance Highlights**: 실험 결과, 다층 LLM 랭킹을 사용하면 추천 시스템의 예측 품질 및 행동 품질이 크게 향상된 것으로 나타났습니다. 이를 통해 기존의 인기 편향을 효과적으로 완화할 수 있었습니다. 데이터셋은 총 23,571개의 화자-인용 페어와 2,843명의 화자로 구성되어 있으며, 다양한 글로벌 도메인에서 수집된 데이터입니다.



### 1000 African Voices: Advancing inclusive multi-speaker multi-accent speech synthesis (https://arxiv.org/abs/2406.11727)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 최초로 발표된 범아프리카 악센트를 가진 영어 음성 합성 시스템, Afro-TTS를 소개합니다. Afro-TTS는 86개 아프리카 악센트와 1000명의 페르소나를 가지고 있어, 교육, 공중보건, 자동화된 콘텐츠 생성 분야에서 유용하게 사용될 수 있습니다.

- **Technical Details**: 본 연구에서는 9개국에서 86개의 악센트와 747명의 다양한 아프리카 악센트를 가진 영어 화자를 대상으로 데이터를 수집했습니다. 수집된 136시간 분량의 오디오 데이터는 다양한 장비로 녹음되었으며, 이를 TTS 훈련에 적합하게 처리하였습니다. 예를 들어, 음성 샘플은 잡음을 제거하고 품질을 개선하기 위한 모델을 사용하였습니다. 두 가지 최신 TTS 모델인 VITS와 XTTS를 사용하여 실험을 수행하였으며, 모델을 파인튜닝하여 아프리카 악센트를 가진 합성 음성을 생성하였습니다.

- **Performance Highlights**: 이 시스템은 아프리카 악센트를 가진 영어 음성을 자연스럽게 생성할 수 있으며, 화자 간의 인터폴레이션을 통해 새로운 목소리를 만들 수 있습니다. 예를 들어, VITS 모델은 VCTK 데이터셋으로 500k 반복 훈련되었고 XTTS 모델은 16개 언어로 구성된 16000 시간의 데이터셋으로 훈련되었는데, 이 모델을 아프리카 악센트를 가진 데이터셋으로 파인튜닝하여 시스템 성능을 검증하였습니다.



### Refusal in Language Models Is Mediated by a Single Direction (https://arxiv.org/abs/2406.11717)
- **What's New**: 이번 연구에서는 대화형 대형 언어 모델(conversational large language models)의 거부 반응(refusal behavior)이 하나의 일차원 하위 공간(one-dimensional subspace)에 의해 매개된다는 것을 밝혀냈습니다. 이는 13개의 인기 있는 오픈 소스 채팅 모델(파라미터 최대 72억 개)에서 공통적으로 발견되었습니다. 이 연구는 모델의 잔여 스트림 활성화(residual stream activations)에서 단 하나의 방향을 지워주면 해로운 명령을 거부하지 않게 되며, 이 방향을 추가하면 해로운 명령이 아닌 경우에도 거부 반응이 유발된다는 것을 보여줍니다.

- **Technical Details**: 각 모델에서 잔여 스트림 활성화(residual stream activations)에서 특정 방향을 찾아내어 이를 제거하면 모델이 해로운 명령을 거부하지 않게 됩니다. 이와 반대로, 이 방향을 추가하면 모델이 무해한 요청조차도 거부하게 됩니다. 이를 바탕으로 작동 원리를 이해하고, 모델의 다른 기능에 최소한의 영향을 미치면서 거부 반응을 비활성화하는 백박스 탈출(white-box jailbreak) 방법을 새롭게 제안했습니다. 또한, 공격적인 접미사(adversarial suffixes)가 거부를 중재하는 방향의 전파를 억제하는 방법을 분석했습니다.

- **Performance Highlights**: 이번 연구는 현재의 안전성 미세 조정 방법(safety fine-tuning methods)의 취약성을 강조하며, 모델의 내부 작동 방식을 이해함으로써 모델의 행동을 제어할 수 있는 실용적인 방법을 개발하는 데 기여하고 있습니다. 이를 통해 모델의 안정성을 높이는 데 중요한 통찰을 제공하고 있습니다.



### Measuring memorization in RLHF for code completion (https://arxiv.org/abs/2406.11715)
- **What's New**: 최근 연구에서 인간 피드백을 활용한 강화학습(RLHF)이 대형 모델을 사용자 선호에 맞추는 주요 방법론으로 나타났습니다. 이 논문은 코드 자동 완성 모델에 적용된 RLHF에서 데이터 암기(Memorization)가 어떻게 발생하고 전파되는지를 분석했습니다. 코드 자동 완성은 대형 언어 모델의 인기 있는 사용 사례 중 하나입니다.

- **Technical Details**: RLHF는 세 가지 단계로 구성됩니다. 먼저 모델은 자가 지도 학습(Self-Supervised Learning)으로 기본적인 코딩 문법과 스타일을 학습합니다. 그런 다음 보상모델(Reward Model, RM)이 인간의 선호도를 근사화하기 위해 훈련됩니다. 마지막으로, 보상모델을 점수 함수로 사용해 강화 학습이 진행됩니다. 이 과정에서 데이터 암기는 주로 첫 번째 단계에서 발생하며, 나머지 단계에서는 감지되지 않는 경향이 있습니다.

- **Performance Highlights**: RLHF는 직접 파인 튜닝을 통해 모델을 맞추는 것에 비해 데이터 암기 확률을 크게 줄여줍니다. 그러나 파인 튜닝 단계에서 이미 암기된 예제들은 RLHF 이후에도 대부분 암기가 유지됩니다. RLHF 훈련 과정에서 보상 모델이 사용하는 데이터는 일반적으로 암기되지 않는 것으로 나타났습니다. 이 결과는 보상 모델 훈련에 귀중한 데이터를 사용할 수 있는 가능성을 열어줍니다.



### Prompts as Auto-Optimized Training Hyperparameters: Training Best-in-Class IR Models from Scratch with 10 Gold Labels (https://arxiv.org/abs/2406.11706)
- **What's New**: 이번 연구는 1억 개 미만의 파라미터를 가진 소규모 신경 정보 검색 모델을 고작 10개의 고유 관련 라벨로 학습하는 새로운 방법을 개발했습니다. 이 방법은 문서에 대한 질문을 생성하기 위해 언어 모델(LM)을 사용하며, 중요한 단계는 학습 품질에 기반하여 LM 프롬프트를 자동으로 최적화하는 것입니다.

- **Technical Details**: 연구는 BIRCO 벤치마크 기준을 사용하여 실험을 수행하였으며, 우리의 방법으로 학습된 모델이 RankZephyr보다 우수하고, 7B 파라미터 모델인 RankLLama와 경쟁력이 있음을 발견했습니다. 이 모델들은 10만 건 이상의 라벨로 학습되었습니다. 주요 기술적 접근은 PATH(Prompts as Auto-optimized Training Hyperparameters) 방법으로, 프롬프트를 자동으로 최적화하여 합성 데이터셋을 생성하고, 이를 DSPy 프로그래밍 모델로 표현합니다.

- **Performance Highlights**: 평가에서는 10개의 긍정 라벨로 PATH를 적용했을 때 매우 경쟁력 있는 성능을 보였습니다. 특히, 태스크와 LM 전반에 걸쳐 평균 NDCG@10 점수에서 BM25보다 6.0 포인트, 10개의 긍정 라벨로 미세 조정된 LM보다 6.5 포인트, GPT-3.5를 사용한 합성 질문 생성보다 4.5 포인트 더 높은 성과를 보였습니다. 또한, RankLLaMA와 같은 대규모 크로스인코더와 비슷한 수준의 성능을 유지했습니다.



### TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy (https://arxiv.org/abs/2406.11678)
- **What's New**: 본 연구에서는 TourRank라는 새로운 문서 랭킹 방법을 제안합니다. 이는 스포츠 토너먼트 구조에서 영감을 받아 다양한 문서를 효과적으로 랭킹할 수 있도록 설계되었습니다. TourRank는 LLM(Large Language Models)의 입력 길이 제한, 입력 순서에 따른 랭킹 결과의 불균형, 그리고 성능과 비용 간의 균형과 같은 기존 LLM 기반 랭킹 방법의 문제점을 해결하기 위해 도입되었습니다.

- **Technical Details**: TourRank는 다단계 토너먼트 방식을 채택하여 각 문서를 참가자로 간주하고, 여러 단계에서 쿼리에 가장 관련성이 높은 문서를 선택합니다. 각 단계에서 그룹을 형성하고 LLM을 사용하여 그룹 내에서 가장 관련성이 높은 문서를 다음 단계로 진행시킵니다. 이 과정은 병렬화하여 처리 시간을 단축시키고, 첫 번째 검색 모델이 제공하는 초기 문서 순서도 활용하지만 이에 크게 의존하지 않도록 설계되었습니다. 또한, 문서를 여러 라운드의 토너먼트를 통해 점수를 부여하여 최종 점수 기반으로 문서 순위를 결정합니다.

- **Performance Highlights**: TourRank는 TREC DL 19, TREC DL 20, 그리고 BEIR 벤치마크 데이터셋에서 실험을 수행하였으며, 기존의 제로샷 문서 랭킹 방법을 능가하는 성능을 입증했습니다. 특히, Mistral-7B 및 Llama-3-8B와 같은 오픈 소스 모델에서도 최첨단(SOTA) 성능을 달성했습니다. TourRank는 랭킹 성능과 비용 간의 균형을 잘 맞추었으며, 초기 후보 문서 순서에 대한 민감성을 효과적으로 완화하였습니다.



### BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models (https://arxiv.org/abs/2406.11675)
Comments:
          27 pages, 3 figures, 9 tables; preprint, work in progress

- **What's New**: Bayesian Low-Rank Adaptation by Backpropagation (BLoB)는 대형 언어 모델(LLMs)에서 수치 불확실성 문제를 해결하기 위해 제안된 새로운 알고리즘입니다. BLoB는 LLM들의 파라미터의 평균과 공분산을 지속적으로 조정하는 과정을 통해 전체 미세 조정 단계에서 불확실성을 평가할 수 있도록 도와줍니다. 이 방법은 전통적인 사후 학습 방식의 한계를 뛰어넘어 보다 효과적인 일반화 성능과 불확실성 평가를 제공합니다.

- **Technical Details**: BLoB는 변량 베이지안 저급 적응(Variational Bayesian Low-Rank Adaptation)을 이용하여 LLM을 미세 조정합니다. 이는 전체 학습 파라미터의 저계급 구조를 가정하고 독립적인 가우시안 분포들의 선형 결합으로 이를 모델링합니다. BLoB의 핵심은 backpropagation을 통해 평균과 공분산을 동시에 추정하는 것이며, 이러한 과정은 모드 추정에 도움이 되는 파라미터의 랜덤 샘플링을 포함합니다.

- **Performance Highlights**: BLoB의 성능은 여러 데이터셋에서 일반화와 불확실성 평가 측면에서 뛰어남이 입증되었습니다. 특히, 인-분포 및 아웃-분포 데이터셋 모두에서 우수한 성능을 보여주며, 기존의 베이지안 기법 대비 더 나은 결과를 제공합니다.



### Words in Motion: Representation Engineering for Motion Forecasting (https://arxiv.org/abs/2406.11624)
- **What's New**: 이번 논문에서는 모션 예측(motion forecasting)을 자연어를 사용하여 인간이 해석할 수 있는 형태로 양자화하고, 이러한 특징들이 숨겨진 상태(hidden states)에 어떻게 내재되어 있는지를 측정합니다. 이를 통해 텍스트 입력을 사용하여 Transformer 기반 모션 예측 모델을 제어할 수 있는 새로운 인터페이스를 제공합니다.

- **Technical Details**: 기존의 모션 예측 모델은 과거의 모션 시퀀스와 환경 컨텍스트를 처리하여 미래의 모션 시퀀스를 예측합니다. 이 논문에서는 자연어를 사용하여 모션 특징을 정량화하고, 각 특징에 대한 컨트롤 벡터(control vectors) 를 생성하여 예측 제어를 가능하게 합니다. 이러한 방식으로 텍스트 입력을 통해 모션 예측 모델의 숨겨진 상태를 해석하고 제어할 수 있습니다. 또한, 리니어 프로빙(linear probing)를 통해 숨겨진 상태가 해석 가능한 모션 특징과 어떻게 일치하는지를 평가합니다.

- **Performance Highlights**: 실험 결과 숨겨진 모션 시퀀스 상태가 정량화된 모션 특징 집합에 따라 구성됨을 보여줍니다. 이를 통해 텍스트 입력을 사용한 새로운 제어 방법이 가능해졌으며, 이는 Transformer 기반 모션 예측 모델의 해석과 제어에 큰 장점을 제공합니다.



### GECOBench: A Gender-Controlled Text Dataset and Benchmark for Quantifying Biases in Explanations (https://arxiv.org/abs/2406.11547)
Comments:
          Under review

- **What's New**: Large pre-trained language models (LLMs)은 자연어 처리 (NLP)에서 매우 중요한 역할을 합니다. 하지만 이러한 모델들은 종종 데이터에 포함된 다양한 편향, 예를 들어 성별 편향, 등의 문제를 가지고 있습니다. 이번 연구는 이러한 성별 편향이 모델 설명에 미치는 영향을 조사하기 위해 GECO라는 성별 통제 텍스트 데이터셋과 GECOBench라는 정량적 평가 프레임워크를 개발했습니다.

- **Technical Details**: GECO 데이터셋은 동일한 문장이 남성과 여성 형태로 나타나는 4가지 변형을 포함합니다. 이 데이터셋은 성별 분류 작업에 사용되며, 이는 XAI (explainable artificial intelligence) 방법의 객관적인 평가를 가능하게 합니다. 또한, GECOBench는 사전 훈련된 언어 모델들을 다양한 정도로 파인 튜닝(fine-tuning)하여 XAI 방법을 평가합니다. 이를 통해 사전 훈련 과정에서 발생하는 설명 편향이 어떻게 모델 설명에 영향을 미치는지 확인하고, 파인 튜닝 또는 임베딩 레이어의 전체 재훈련(retraining)이 설명 성능에 얼마나 긍정적인 영향을 미칠 수 있는지 조사했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 설명 성능과 파인 튜닝된 레이어 수 사이에 명확한 의존성이 존재합니다. 특히, XAI 방법은 임베딩 레이어의 파인 튜닝이나 전체 재훈련이 이루어질 때 더 좋은 성능을 보였습니다. 이러한 관계는 동일한 작업에서 유사한 분류 성능을 달성하는 모델들에도 동일하게 적용되었습니다. 이로써 제안된 성별 통제 데이터셋과 새로운 벤치마킹 접근 방식의 유용성을 강조합니다.

- **Link to Code and Dataset**: 모든 코드 및 데이터셋 생성, 모델 훈련, 평가, 시각화 등은 다음 링크에서 확인할 수 있습니다: [this https URL]



### GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages with Automated Crawling, Transcription and Refinemen (https://arxiv.org/abs/2406.11546)
Comments:
          Under review

- **What's New**: 이번 논문에서는 저자들이 GigaSpeech 2라는 대규모 멀티도메인 다국어 음성 인식 코퍼스를 소개합니다. 이 코퍼스는 태국어, 인도네시아어, 베트남어 등 저자원이 언어를 대상으로 하며, 레이블링된 음성-텍스트 데이터에 의존하지 않습니다. 약 30,000시간의 자동 전사된 음성 데이터를 포함하고 있습니다.

- **Technical Details**: GigaSpeech 2는 YouTube에서 수집한 레이블링되지 않은 비디오에서 자동으로 전사된 데이터를 사용하여 구축되었습니다. 데이터 크롤링(crawling), 전사(transcription), 레이블 정제(label refinement)를 위한 자동화된 파이프라인이 개발되었으며, 여기에는 Whisper를 초기 전사 도구로 사용하고 TorchAudio를 강제 정렬(forced alignment) 도구로 사용하는 것이 포함됩니다. 또한, 많은 사이의 필터링으로 데이터를 품질 보증하기 위해 다차원 필터링이 사용되었습니다. 수정된 Noisy Student Training (NST) 방법이 사용되어 결함이 있는 가짜 레이블을 반복적으로 정제합니다.

- **Performance Highlights**: 실험 결과, GigaSpeech 2로 훈련된 ASR 모델은 태국어, 인도네시아어, 베트남어의 경우 YouTube 테스트 세트에서 Whisper large-v3 모델 대비 단어 오류율(word error rate)을 25%에서 40%까지 줄였습니다. 이는 모델 파라미터의 단지 10%만을 사용한 결과입니다. 또한, GigaSpeech 2로 훈련된 ASR 모델은 상용 서비스에 비해 우수한 성능을 발휘합니다.



### GeoGPT4V: Towards Geometric Multi-modal Large Language Models with Geometric Image Generation (https://arxiv.org/abs/2406.11503)
- **What's New**: 최신 연구에서는 GPT-4 및 GPT-4V를 활용하여 텍스트와 이미지가 잘 맞춰진 기하학 문제를 생성하는 새로운 파이프라인을 소개했습니다. 이를 통해 GeoGPT4V라는 4.9K 개인 기하학 문제와 19K의 공개 데이터셋을 결합한 새로운 데이터셋을 개발했습니다.

- **Technical Details**: 새로운 파이프라인은 GPT-4V를 이용해 공개 데이터셋에 기반한 단순화된 기하학 문제를 생성하고, 이후 GPT-4를 사용해 각 기하학 문제에 대해 Wolfram 코드를 생성합니다. 코드는 실행되어 여러 개의 기하학 이미지를 만들고, GPT-4V가 이들을 스코어링하여 최적의 이미지를 선택하도록 합니다.

- **Performance Highlights**: GeoGPT4V 데이터셋을 사용한 모델들은 MathVista와 MathVision 벤치마크에서 현저한 성능 향상을 보였습니다. 예를 들어, LLaVA-1.5-7B 모델은 58.2%, ShareGPT4V-7B 모델은 33.8%의 상대적 성능 향상을 보여, 이 접근법의 효과를 입증했습니다.



### DiTTo-TTS: Efficient and Scalable Zero-Shot Text-to-Speech with Diffusion Transformer (https://arxiv.org/abs/2406.11427)
- **What's New**: 이 논문은 대규모 확산 모델(large-scale diffusion models)을 사용하여 텍스트-음성 변환(text-to-speech, TTS)에서 효율성과 확장성을 개선하는 Diffusion Transformer(DiT)를 제안합니다. 기존의 도메인-특정 요소(예: 음소 및 음소 수준의 지속 시간)를 사용하지 않고도 텍스트와 음성 간의 정밀한 정렬을 달성하는 새로운 방법론을 포함하고 있습니다.

- **Technical Details**: 논문에서는 오프-더-쉘프(off-the-shelf) 사전 학습된 텍스트 및 음성 인코더를 활용하는 효율적인 확산 트랜스포머(Diffusion Transformer, DiT)를 제안합니다. DiT의 주요 특징은 교차 주의 메커니즘(cross-attention mechanisms)을 통한 텍스트-음성 정렬 문제를 해결하고, 생성된 음성의 전체 길이를 예측하는 모듈을 도입한 것입니다. 82,000시간의 데이터와 7억 90백만 개의 파라미터로 모델을 훈련하였습니다.

- **Performance Highlights**: 제안된 DiT 기반 TTS 모델은 도메인-특정 요소 없이도 우수한 또는 동등한 자연스러움(naturalness), 이해도(intelligibility), 그리고 화자 유사도(speaker similarity) 면에서 최신 TTS 모델과 비교할 때 비슷하거나 더 나은 성능을 보여주었습니다. 또한, 최적의 오토리그레시브 모델(autoregressive model)보다 추론 속도가 4.6배 빠르고, 모델 크기가 3.84배 더 작습니다.



### Evaluating the Efficacy of Open-Source LLMs in Enterprise-Specific RAG Systems: A Comparative Study of Performance and Scalability (https://arxiv.org/abs/2406.11424)
- **What's New**: 이번 논문은 오픈 소스 대형 언어 모델(LLMs)을 통한 기업 맞춤형 데이터 세트에서의 Retrieval-Augmented Generation(RAG) 작업에 대한 분석을 제공합니다. 웹사이트에서 스크랩된 데이터를 사용하여 다양한 오픈 소스 LLMs의 성능, 접근성, 통합 가능성을 평가합니다. 연구 결과에 따르면 효과적인 임베딩 기술과 결합된 오픈 소스 LLMs는 RAG 시스템의 정확성과 효율성을 크게 향상시킬 수 있다는 것을 보여줍니다.

- **Technical Details**: 논문에서는 데이터 수집, 모델 선택, 시스템 아키텍처, 평가 지표, 실험 절차 등 여러 단계로 구성된 방법론을 설명합니다. 데이터 수집 단계에서는 https://i-venture.org/ 사이트에서 URL을 가져와 텍스트 콘텐츠를 추출했으며, langchain 라이브러리를 사용하여 데이터를 작은 조각으로 나누었습니다. 텍스트 조각에 대한 임베딩(Embeddings)은 Hugging Face 플랫폼에서 가져왔으며, 이를 통해 생성된 임베딩은 FAISS(Face AI Similarity Search) 벡터 데이터베이스에 저장됐습니다. LLMs는 RAG 시스템의 생성을 강화하는 데 사용되었습니다.

- **Performance Highlights**: 연구 결과는 오픈 소스 LLMs와 효과적인 임베딩 기술이 결합될 경우 RAG 시스템의 정확성과 효율성을 크게 향상시킬 수 있다는 것을 입증했습니다. 또한 TopK 매개 변수를 조정하여 다른 LLMs에 대해 상당한 성능 변화를 관찰할 수 있었으며, 이는 특정 기업 환경에서 최적의 성능을 위한 미세 조정의 중요성을 강조합니다.



### Dredge Word, Social Media, and Webgraph Networks for Unreliable Website Classification and Identification (https://arxiv.org/abs/2406.11423)
- **What's New**: 이 연구는 검색 엔진과 소셜 미디어 사이에서 신뢰할 수 없는 콘텐츠가 퍼지는 복잡한 경로를 모방하려는 시도로, 웹그래프 및 대규모 소셜 미디어 맥락을 웹사이트 신뢰성 분류 및 발견 시스템에 통합하는 영향력을 탐구합니다. 특히 소셜 미디어에서 '드레지 단어 (dredge words)'라 불리는, 신뢰할 수 없는 도메인들이 높은 순위를 차지하는 용어 또는 구를 사용합니다.

- **Technical Details**: 이 연구는 종합적인 그래프 신경망 검증을 통해 이종 그래프 모델이 웹그래프와 소셜 미디어 데이터의 맥락을 활용하여 동종 및 단일 모드 접근법보다 뛰어나다는 것을 보여줍니다. 웹사이트 신뢰성 분류를 위해 대규모 트위터 데이터의 사용자-도메인 상호작용 네트워크와 소규모 웹그래프 네트워크를 사용하여, 이 두 가지를 결합한 이종 그래프 모델을 사용합니다. 또한 웹사이트 신뢰성 점수에 기반한 교육 과정 설계를 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 모델은 상위 k개의 미표시 신뢰할 수 없는 웹사이트를 발견하는 성능에서 경쟁 시스템을 크게 능가합니다. 드레지 단어를 소셜 미디어와 온라인 상거래 플랫폼에 신뢰할 수 없는 웹사이트를 강하게 연결시키는 데 사용하며, 다양한 경로에서 신뢰할 수 없는 콘텐츠를 찾아내는 강력한 신호를 나타냅니다. 결과적으로, 트위터 데이터와 웹그래프 데이터를 결합한 이종 모델이 기존 시스템보다 성능이 우수하다는 것을 입증했습니다.



### Multimodal Structured Generation: CVPR's 2nd MMFM Challenge Technical Repor (https://arxiv.org/abs/2406.11403)
Comments:
          Conference on Computer Vision and Pattern Recognition's 2nd Multimodal Foundation Models Challenge

- **What's New**: Multimodal Foundation Models (MMFMs)이 다양한 컴퓨터 비전 및 자연어 처리 작업에서 뛰어난 성능을 보여주었지만, 문서 이해와 같은 특정 작업에서는 여전히 한계를 보입니다. 이번 보고서에서는 freeze된 MMFM의 출력 로짓(logits)을 제어하여 구조화된 출력을 생성하는 일반적인 프레임워크인 Multimodal Structured Generation을 제시합니다. 이를 통해 후속 API가 구문 분석하고 사용할 수 있습니다. 이 접근법을 사용하여 CVPR 회의에서 주최한 2nd Multimodal Foundation Models Challenge에서 2위를 차지했습니다.

- **Technical Details**: 기존의 생성 모델은 출력이 항상 후속 프로그램이나 API에서 구문 분석 가능한 형식이 아닙니다. 예를 들어, 인간 사용자가 LLM에게 숫자 목록을 정렬하는 Python 스크립트를 생성하도록 요청하면, LLM의 출력이 실행 가능한 것처럼 보이지만 실제로 실행할 때 오류가 발생할 수 있습니다. 이를 해결하기 위해 'soft constraints'와 'hard constraints' 방법을 사용하여 생성 모델의 출력을 제어합니다. 구조화된 생성을 위해 우리는 Huggingface의 Inference Endpoints API와 Text Generation Interface (TGI) API를 사용했습니다.

- **Performance Highlights**: 우리의 접근법은 CVPR의 2nd MMFM Challenge의 Phase 2 숨겨진 테스트 세트에서 2위를 차지했으며 전체 순위에서 3위를 차지했습니다. 중요한 것은, 복잡한 모델링 단계 없이도 이 성과를 달성했다는 점입니다. 이는 단순한 엔지니어링 작업이 고비용의 복잡한 모델링 단계를 능가할 수 있음을 보여줍니다. 전체 스크립트 및 평가 결과는 GitHub 리포지토리에서 확인할 수 있습니다.



### $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts (https://arxiv.org/abs/2406.11353)
Comments:
          9 pages, 8 figures, camera ready on ICML2024

- **What's New**: 최근 Mixture-of-Experts(MoE) 모델이 대규모 언어 모델(LLM)을 확장하는 유망한 방법으로 주목받고 있습니다. 그러나 MoE의 신뢰도 평가가 이에 비해 부족한 상황입니다. 이 연구에서는 MoE의 안전성과 허상 생성, 적대적 공격에 대한 저항성, 그리고 분포 외 데이터에 대한 강건성을 평가할 수 있는 최초의 종합적인 평가 도구인 MoE-RBench를 제안합니다.

- **Technical Details**: MoE-RBench는 MoE와 밀집 모델을 비교 평가하기 위해 다양한 모델과 데이터셋을 사용했습니다. MoE 모형의 신뢰도를 높이기 위해 적절한 하이퍼파라미터, 학습 레시피, 추론 기법을 사용한 경우, MoE 모델이 밀집 LLM보다 더 신뢰성 있는 모델이 될 수 있음을 발견했습니다. 특히, MoE의 강건성은 기본 학습 설정에 매우 민감하다는 것을 확인했습니다.

- **Performance Highlights**: MoE 모델은 적대적 공격과 분포 외 데이터에서 밀집 모델보다 뛰어난 강건성을 보였습니다. 또한 MoE 모델은 기존의 모델 세팅과 비교해 초기 성능은 뒤처지는 경우가 많지만, 최적의 학습 및 추론 방법론을 사용할 경우 높은 안정성과 질을 제공할 수 있는 가능성을 가지고 있습니다. 이는 다양한 모델 아키텍처, 모델 크기, 그리고 여러 데이터셋에 걸친 실험을 통해 확인되었습니다.



### GUICourse: From General Vision Language Models to Versatile GUI Agents (https://arxiv.org/abs/2406.11317)
- **What's New**: 최근 Vision Language Models (VLMs)의 발전으로 다재다능한 GUI 내비게이션 에이전트를 개발할 잠재력이 부각되고 있습니다. 그러나 현재 VLMs는 필수 능력(OCR 및 grounding)과 GUI 지식(요소의 기능 및 제어 방법)에서 한계를 보이고 있어 실용적인 GUI 에이전트로 사용되기 어려웠습니다. 이를 해결하기 위해, GUICourse라는 데이터셋을 제안하며, GUIEnv, GUIAct, GUIChat 세 가지 데이터셋으로 구성되어 있습니다.

- **Technical Details**: GUICourse는 VLMs의 OCR과 grounding 능력을 강화하기 위한 GUIEnv, GUI 시스템의 컴포넌트와 상호작용 지식을 풍부하게 하기 위한 GUIAct 및 GUI 에이전트의 상호작용 기술을 향상시키기 위한 GUIChat 데이터셋으로 구성됩니다. GUIEnv는 10M 웹사이트 페이지-주석 쌍과 0.7M 지역-텍스트 QA 쌍으로 구성되어 있으며, GUIAct는 67k 단일 단계와 15k 다단계 작업 지침을 포함합니다. GUIChat은 44k 단일 턴 QA 쌍과 6k 다중 턴 대화 데이터를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 GUI 에이전트는 일반적인 GUI 작업에서 기존 VLMs보다 우수한 성능을 보였습니다. 특히, 작은 크기의 GUI 에이전트(3.1B 파라미터)도 단일 및 다중 단계 GUI 작업에서 효과적으로 작동합니다. 또한, GUIEnv 데이터는 VLMs의 OCR 및 그라운딩 능력을 향상시키는 데 효과적이며, 높은 해상도는 GUI 작업 성능을 크게 증가시키는 것으로 나타났습니다.



### VideoVista: A Versatile Benchmark for Video Understanding and Reasoning (https://arxiv.org/abs/2406.11303)
Comments:
          38 pages, 44 figures

- **What's New**: VideoVista라는 새로운 비디오 QA 벤치마크가 발표되었습니다. 이는 다양한 콘텐츠, 시간 길이, 그리고 여러 능력을 평가할 수 있도록 설계되었으며, 3,400개의 비디오에서 유도된 25,000개의 질문을 포함하고 있습니다. 이 벤치마크는 GPT-4o와 고급 분석 도구를 활용하여 자동 데이터 구성 프레임워크를 통해 생성되었습니다.

- **Technical Details**: VideoVista는 총 14개의 비디오 카테고리와 27개의 작업 유형 (이해 과제 19개, 추론 과제 8개)을 포함합니다. 이를 위해 유튜브에서 894개의 비디오를 다운로드한 후 특수 비디오 분할 기술을 사용하여 다양한 길이의 클립으로 나눕니다. 또한, 객체 세분화와 추적 등 고급 비디오 분석 방법을 이용하여 데이터 세트를 구성하였습니다.

- **Performance Highlights**: 최신 Video-LMMs를 평가한 결과, 긴 비디오 처리와 세밀한 비디오 이해 작업(예: 시간 위치 추적, 이상 탐지)에서 어려움을 겪고 있다는 것을 확인했습니다. 또한, 논리적 및 관계적 추론 능력도 상대적으로 낮았으며, 오픈 소스 Video-LMMs의 성능이 GPT-4o와 Gemini 1.5에 비해 약 20점 낮았습니다. 이는 비디오 이해와 정확한 추론을 수행할 수 있는 LMMs를 개발하는 데 중요한 역할을 할 것입니다.



### Optimizing and Testing Instruction-Following: Analyzing the Impact of Fine-Grained Instruction Variants on instruction-tuned LLMs (https://arxiv.org/abs/2406.11301)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Models, LLMs)이 복잡한 명령을 보다 능숙하게 따를 수 있도록 도와주는 새로운 데이터 증강 기법을 소개합니다. 이 기법은 복잡한 명령을 더 단순한 하위 구성 요소로 분해하고 이를 수정한 후 다시 새로운 변형으로 재구성합니다. 이를 통해 원래 명령의 맥락과 복잡성을 유지하면서도 변동성을 도입할 수 있어, LLMs의 명령-따르기 능력을 향상시킵니다.

- **Technical Details**: 제안된 기법은 DeMoRecon 데이터셋을 사용합니다. DeMoRecon은 크게 다섯 가지 구성 요소(DeMoRecon-Aug, DeMoRecon-Ref, DeMoRecon-Aug-DPO, DeMoRecon-Ref-DPO, DeMoRecon-Eval)로 나뉘어 LLMs의 성능을 미세 조정 및 평가하는 데 사용됩니다. 데이터 생성 과정에서 명령을 더 단순한 하위 그룹(facts와 sub-instructions)으로 분해하고, 이를 수정한 후 재구성합니다. 이 모든 과정을 GPT-4를 이용해 수행합니다. 이 데이터셋은 복잡한 명령이 아닌, 유사한 명령 변형을 정확하게 따르는 능력을 평가하는 데 중점을 둡니다.

- **Performance Highlights**: DeMoRecon 데이터셋으로 학습된 LLMs는 기존의 데이터셋으로 학습된 모델들에 비해 명령-따르기 작업에서 큰 성능 향상을 보였습니다. 특히 GPT-4를 포함한 여러 주요 LLMs에서 검증된 결과, 유사한 명령 변형을 정확히 따르는 능력이 크게 향상되었습니다. 이를 통해 DeMoRecon이 향후 이 분야에서 중요한 벤치마크가 될 수 있음을 확인했습니다.



### Iterative Utility Judgment Framework via LLMs Inspired by Relevance in Philosophy (https://arxiv.org/abs/2406.11290)
Comments:
          22 pages

- **What's New**: 최근 정보 검색(IR) 분야에서는 주제 관련성(topical relevance)뿐만 아니라 유용성(utility) 평가도 중요하게 다뤄지고 있습니다. 이는 Retrieval-Augmented Generation(RAG)와 같은 다운스트림 작업을 촉진하기 위해 필요합니다. 이 논문에서는 주제 관련성, 해석적 관련성(interpretational relevance), 동기적 관련성(motivational relevance)을 포함한 세 가지 종류의 관련성을 동적으로 통합하는 Iterative utiliTy judgmEnt fraMework(ITEM)을 제안합니다. 이 프레임워크는 LLMs를 활용하여 각 단계의 성능을 크게 향상시킵니다.

- **Technical Details**: ITEM 프레임워크는 세 가지 주요 단계를 결합합니다: 주제 관련성 평가, 유용성 판단, 그리고 답변 생성입니다. Schutz의 철학적 관련성 시스템에서 영감을 받아 이 세 단계가 반복적으로 상호작용하며 성능을 증대시킵니다. 즉, 주제 관련성은 현재의 집중 대상을 형성하고, 해석적 관련성은 과거 경험을 통해 현재 대상을 이해하며, 동기적 관련성은 이를 바탕으로 추가 자료를 획득하여 새로운 경험으로 삼습니다.

- **Performance Highlights**: TREC DL, WebAP, NQ와 같은 다중 등급(grade) 패시지 검색 및 사실 기반 질문 답변(QA) 데이터셋에서 실험을 수행했습니다. 실험 결과, ITEM 프레임워크는 유용성 평가, 주제 관련성 순위, 답변 생성 측면에서 대표적인 기존 방법보다 우수한 성능을 보였습니다. 특히 단일 회차의 유용성 판단 접근법보다도 크게 향상된 결과를 보였습니다.



### Self and Cross-Model Distillation for LLMs: Effective Methods for Refusal Pattern Alignmen (https://arxiv.org/abs/2406.11285)
- **What's New**: 최근 대형 언어 모델(LLMs)인 OpenAI의 GPT 시리즈, Anthropic의 Claude, Meta의 LLaMa가 뛰어난 텍스트 생성 능력을 보여주었으나, 유해한 프롬프트에 민감하다는 점에서 보안 문제가 대두되고 있습니다. 이 논문에서는 이러한 문제를 해결하기 위해 Supervised Fine-Tuning (SFT)과 Reinforcement Learning from Human Feedback (RLHF)를 포함한 다양한 정렬 기법을 조사하였습니다. 연구 결과, Claude3와 같은 일정한 거부 패턴을 가진 모델이 더 높은 보안을 보여주었으며, 이를 기반으로 자가 증류(self-distillation)와 크로스 모델 증류(cross-model distillation) 방법을 제안하였습니다. 이러한 방법들은 거부율을 크게 향상시키고 안전하지 않은 콘텐츠를 줄이는 데 효과적이었습니다.

- **Technical Details**: 본 논문은 유해한 프롬프트에 대한 거부 패턴을 연구하기 위해 9개의 LLM을 대상으로 한 실증 연구를 수행했습니다. 먼저 510개의 유해 프롬프트를 포함한 벤치마크를 구축한 뒤, 각 LLM에 입력하여 총 4590개의 응답을 얻었습니다. 응답을 안전한 유형과 안전하지 않은 유형으로 분류한 후, 다양한 모델의 거부 패턴을 분석하였습니다. 그 결과, Claude3 opus는 94.51%의 가장 높은 거부율을 보였고, GPT-3.5와 GPT-4도 90%를 초과하는 높은 거부율을 기록하였습니다. 이를 바탕으로 자가 증류와 크로스 모델 증류 방법을 제안하였으며, 평가 결과 이 두 방법이 LLM의 보안을 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 자가 증류와 크로스 모델 증류 방법을 적용한 결과, Vicuna-7B, Vicuna-13B, LLaMa-3-8B-Instruct와 같은 모델들의 거부율이 약 5% 증가하고 안전하지 않은 콘텐츠 출력이 약 30%로 감소했습니다. 특히 Claude3-opus 모델을 교사 모델로 사용한 크로스 모델 증류는 거부율을 5-8% 증가시켜 Claude3의 94.51%에 근접한 거부율을 달성했습니다. 이러한 결과는 증류 기반의 정렬 방법이 유해한 프롬프트에 대한 보안을 강화하는 데 큰 잠재력을 가지고 있음을 보여줍니다.



### Probing the Decision Boundaries of In-context Learning in Large Language Models (https://arxiv.org/abs/2406.11233)
Comments:
          18 pages, 18 figures

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 in-context learning을 새로운 관점에서 탐구하여, 간단한 이진 분류 작업에서 결정 경계를 시각화합니다. 연구진은 LLM들이 현재의 상태에서 본래의 분류 문제에서 선형 분리성 여부에 상관없이 비정상적이고 불규칙한 결정 경계를 형성하는 경향을 보인다는 점을 발견했습니다. 이를 통해 이들 모델이 학습 데이터와 모델 구조에 어떻게 반응하는지에 대한 깊은 이해를 제공하고자 합니다.

- **Technical Details**: 본 연구는 다양한 LLMs (Llama2-7b, Llama2-13b, Llama3-8b, Mistral-7b 등)에서의 in-context learning을 평가합니다. 연구진은 모델 크기, 사전 학습 데이터 및 목표, in-context 예제의 개수, 정량화 수준, 레이블 형식의 의미와 예제 순서 등이 결정 경계에 미치는 영향을 조사했습니다. 서브셋에서는 fine-tuning 및 adaptive prompting 전략을 통해 결정 경계를 개선할 방법도 탐구했습니다.

- **Performance Highlights**: 연구 결과, 최근의 LLM들이 단순한 선형 이진 분류 작업에서도 부드럽지 않고 불규칙한 결정 경계를 형성한다는 점을 확인했습니다. 이를 통해, 전통적인 머신러닝 모델과 달리 LLM들이 더 많은 예제에도 불구하고 일정한 결정 경계를 형성하기 어려움을 나타냈습니다. 또한, 결정을 개선하기 위해 fine-tuning 및 불확실성 인지(active learning) 전략이 효과적임을 보였습니다.



### Enabling robots to follow abstract instructions and complete complex dynamic tasks (https://arxiv.org/abs/2406.11231)
- **What's New**: 이 논문에서는 예측 불가능한 환경에서 복잡한 작업을 수행하는 로봇 시스템을 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs), 큐레이트된 지식 베이스(Knowledge Base), 통합된 힘과 시각적 피드백(IFVF)을 결합하여, 로봇이 추상적인 명령을 해석하고 긴 시간에 걸친 작업을 수행하며 다양한 불확실성을 처리할 수 있도록 돕습니다. 코드를 동적으로 생성하여 지식 베이스에서 적절한 함수를 선택하고 적용하며, 실행 도중 발생하는 외부 소음과 방해 요소에 대응할 수 있습니다.

- **Technical Details**: 제안된 방법은 사용자의 질의와 주변 환경을 분석하기 위해 GPT-4를 활용하고, 캡슐화된 함수가 들어있는 데이터베이스 접근 코드를 생성합니다. 추상적인 명령을 구체적인 작업 단계로 번역하고, 각 단계는 구체적인 코드 생성을 포함하여 지식 베이스에서 IFVF 관련 예제를 추출합니다. 또한 로봇은 통합된 힘과 시각 피드백(IFVF)을 사용하여 노이즈와 간섭을 처리합니다.

- **Performance Highlights**: 커피 제조와 접시 장식과 같은 실제 시나리오에서 이 프레임워크를 테스트한 결과, 작업 단위는 물을 따르는 것에서 서랍을 여는 것까지 다양하며 각 작업은 서로 다른 피드백 유형과 방법의 혜택을 봅니다. 7자유도 Kinova 로봇 암을 통해 불확실한 환경에서 복잡한 작업을 수행할 수 있었으며, 정밀성 및 정확성을 보장하는 실시간 피드백을 통해 유연하게 작업을 처리할 수 있었습니다.



### Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal Large Language Models (https://arxiv.org/abs/2406.11230)
- **What's New**: 최근의 중요 멀티모달 대형 언어 모델들(Multimodal Large Language Models, 이하 MLLMs)은 다양한 응용 프로그램에서 유망한 성과를 보이고 있지만 장기 맥락 이해(Long-Context Capabilities)를 평가하는 연구는 여전히 부족한 상태입니다. 이를 해결하기 위해 MultiModal Needle-in-a-haystack(MMNeedle) 벤치마크를 도입했습니다. 이 벤치마크는 여러 이미지 입력을 기반으로 이미지 이어붙이기(image stitching) 기법을 사용해 입력 맥락 길이를 증가시킴으로써 MLLM의 장기 맥락 처리 성능을 평가합니다. MMNeedle은 텍스트 지시 사항과 이미지 설명을 기반으로 다수의 이미지 중 목표 하위 이미지를 찾아내는 능력을 테스트합니다.

- **Technical Details**: MMNeedle 벤치마크는 40,000개의 이미지, 560,000개의 캡션, 그리고 280,000개의 'needle-haystack' 쌍으로 구성되어 있으며, 다양한 맥락 길이, 단일 및 다중 니들(needle) 설정, 긍정 및 부정 샘플을 포함합니다. 모델의 성능을 종합적으로 평가하기 위해 '존재 정확도(existence accuracy)', '인덱스 정확도(index accuracy)', 및 '정확 정확도(exact accuracy)' 등의 평가 지표를 제시합니다. 본 연구는 API 기반 및 오픈 소스 MLLM들을 모두 포함하여 평가합니다.

- **Performance Highlights**: 평가 결과, GPT-4o 모델이 긴 맥락 시나리오에서 다른 모델들을 능가했으나, 부정 샘플에서 '환각(hallucination)' 문제를 겪는 것으로 나타났습니다. API 기반 모델과 오픈 소스 모델 간의 큰 성능 차이가 있으며, 이미지는 많을수록, 특히 서브 이미지가 많을수록 정확도가 크게 하락합니다. 예를 들어, GPT-4o 모델은 10개의 이미지에서 97.00% 정확도를 보였으나 16개의 서브 이미지를 포함한 10개의 이미지 설정에서는 정확도가 26.90%로 하락했습니다.



### WeatherQA: Can Multimodal Language Models Reason about Severe Weather? (https://arxiv.org/abs/2406.11217)
Comments:
          26 pages, 9 figures

- **What's New**: 기상이변(Severe convective weather) 이벤트, 예를 들어 우박, 토네이도, 뇌우와 같은 현상은 매우 빠르게 발생하지만 막대한 피해를 유발하며 매년 수십 억 달러의 비용이 듭니다. 이러한 현상은 기상학자와 고위험 지역의 주민들이 더 잘 준비할 수 있도록 몇 시간 전 예보의 중요성을 부각합니다. 이 연구에서는 WeatherQA라는 새로운 멀티모달 데이터셋을 소개합니다. 이는 기계가 복잡한 기상 파라미터 조합을 추론하고 실제 시나리오에서 기상이변을 예측하는 데 도움을 줍니다. 데이터셋은 8,000쌍 이상의 이미지와 텍스트로 구성된 다채로운 기상이변 이벤트를 포함하고 있습니다.

- **Technical Details**: WeatherQA는 예측에 필수적인 풍부한 정보를 포함하고 있습니다. 이미지는 환경 불안정성, 지상 관측, 레이더 반사율 같은 요소들을 설명하고, 텍스트는 인간 전문가가 작성한 예보 분석 내용을 담고 있습니다. 이를 통해 GPT4, Claude3, Gemini-1.5, 그리고 미세조정된(Llama3) 기반의 VLM와 같은 최첨단 비전 언어 모델들을 평가합니다. 평가 작업은 두 가지 도전 과제를 포함합니다: (1) 영향 지역을 예측하는 다중 선택 질문과 (2) 강력한 대류의 발전 가능성을 분류하는 것. 이러한 작업은 대기 역학과 같은 도메인 지식과 복잡한 멀티모달 데이터의 상호작용에 대한 깊은 이해를 요구합니다.

- **Performance Highlights**: 가장 강력한 VLM인 GPT4o와 인간 추론 사이에는 상당한 격차가 있음을 보여줍니다. 기상학자와의 종합 사례 연구를 통해 모델의 약점을 더욱 구체적으로 드러내며, 더 나은 교육과 데이터 통합이 이 격차를 좁히는 데 필요함을 시사합니다.



### AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieva (https://arxiv.org/abs/2406.11200)
Comments:
          19 pages, 8 figures, 6 tables

- **What's New**: 새로운 프레임워크 AvaTaR를 도입했습니다. AvaTaR는 대규모 언어 모델(Large Language Model, LLM) 에이전트가 제공된 도구들을 효과적으로 이용하고 주어진 작업/도메인에서 성능을 향상시키도록 자동으로 최적화합니다.

- **Technical Details**: AvaTaR는 비교모듈(comparator module)을 설계하여, 훈련 데이터에서 샘플링한 긍정 및 부정 예제 간의 추론을 통해 LLM 에이전트에 통찰력과 전체적인 프롬프트를 반복적으로 제공합니다. 네 가지 복합 멀티모달 검색 데이터셋에서 AvaTaR를 테스트하였습니다.

- **Performance Highlights**: AvaTaR는 모든 지표에서 기존 최신 모델보다 성능이 우수했으며, 특히 Hit@1 지표에서 평균으로 14%의 상대적 성능 향상을 보였습니다. STaRK 벤치마크에서 AvaTaR는 Hit@1에서 15.6%, MRR에서 9.5%의 평균 향상을 보였습니다. 또한 Amazon과 MAG 데이터셋에서 각각 35%에서 75%, 20%에서 78%로 성능을 개선했습니다. 이미지 검색 작업에서도 AvaTaR는 9.2%에서 13.0%의 상대적 성능 향상을 보였습니다.



### SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations (https://arxiv.org/abs/2406.11171)
- **What's New**: SUGARCREPE++ 데이터셋이 소개되었습니다. 이 데이터셋은 Vision-Language Models (VLMs)와 Unimodal Language Models (ULMs)의 어휘적 및 의미적 변형에 대한 민감성을 분석하기 위해 설계되었습니다. 각 샘플은 이미지와 세 개의 캡션으로 구성됩니다: 의미적으로 동일하지만 어휘적으로 다른 두 개의 긍정 캡션과 하나의 부정 캡션. 이를 통해 3가지 방식의 의미적 (비)동등성 문제를 제시하고 있습니다.

- **Technical Details**: SUGARCREPE++ 데이터셋은 VLMs와 ULMs의 감도 분석을 목적으로 만들어졌으며, 각 샘플에 이미지와 3개의 캡션이 포함되어 있습니다. 실험 결과, VLMs가 객체 속성과 공간 관계에서 어휘적 및 의미적 변화를 구별하는 데 어려움을 겪는 것으로 나타났습니다. 더욱 큰 사전 학습 데이터셋, 모델 크기, 다양한 사전 학습 목표를 가진 VLMs가 더 나은 성능을 보였으나, 여전히 개선의 여지가 많습니다.

- **Performance Highlights**: VLMs는 어휘적 및 의미적 변화를 구별하는 데 어려움을 겪으며, 특히 객체 속성이나 관계를 바꾸는 경우에 두드러집니다. 인간 수준의 성능과는 큰 격차가 존재하며, 이는 VLMs의 큰 개선 가능성을 나타냅니다. 최첨단 ULMs조차도 일관되게 의미를 해체하지 못하고, 어휘적 중복이 높은 캡션을 선택하는 경향이 있습니다. 이는 합성만으로는 의미와 어휘적 변형을 이해하는 데 충분하지 않다는 점을 강조합니다.



### MemDPT: Differential Privacy for Memory Efficient Language Models (https://arxiv.org/abs/2406.11087)
Comments:
          12 pages first version

- **What's New**: 최신 연구 보고서는 대규모 언어 모델(Large Language Models, LLMs)의 메모리 비용 절감과 사용자 데이터 프라이버시 보호를 동시에 달성하는 혁신적인 훈련 프레임워크 MemDPT를 제안합니다. 이 프레임워크는 엣지 네트워크와 역방향 네트워크 디자인을 통해 다양한 차등 프라이버시 메모리 효율적 파인튜닝(differential privacy memory-efficient fine-tuning) 방식을 수용합니다.

- **Technical Details**: MemDPT는 메모리 사용량을 절감하기 위해 두 가지 효율적인 파인튜닝 방법인 MemDPT_side와 MemDPT_rev를 탐구합니다. 이 접근법은 각 네트워크 아키텍처에서의 훈련 메모리 요구사항과의 관계를 체계적으로 분석하며, 차등 프라이버시 훈련의 메모리 비용 특성을 설명합니다. DP-SGD(Differential Privacy Stochastic Gradient Descent) 방법을 사용하여 훈련 데이터에 프라이버시를 보장합니다.

- **Performance Highlights**: MemDPT는 다양한 데이터셋과 모델에 대한 광범위한 실험을 통해 경쟁력 있는 성능을 유지하면서도 훈련 메모리 사용량을 2~3배 절감하는 성과를 보여주었습니다. 또한, 사용자 데이터 프라이버시를 강력하게 보호하면서도 특정 다운스트림 작업에 대해 높은 효율성을 가진 파인튜닝을 제공합니다.



### WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences (https://arxiv.org/abs/2406.11069)
Comments:
          link: this https URL

- **What's New**: 최근 비전-언어 모델(VLMs)의 발전은 실제 멀티모달 상호작용에서 인간의 선호도를 벤치마킹하는 필요성을 강조합니다. 이러한 격차를 해결하기 위해, 우리는 인간의 선호도를 수집하여 VLMs를 평가하는 온라인 플랫폼인 WildVision-Arena(WV-Arena)를 출시했습니다. WV-Arena에서 사용자 제출물 8,000개 중 500개의 고품질 샘플을 선택하여 WV-Bench를 구성했습니다.

- **Technical Details**: WV-Bench는 GPT-4를 심판으로 사용하여 각 VLM을 Claude-3-Sonnet과 비교하며, WV-Arena Elo와 0.94의 스피어만 상관 관계(Spearman correlation)를 달성했습니다. 이는 MMVet, MMMU, MMStar와 같은 다른 벤치마크보다 뛰어난 성능을 보여줍니다. 20,000개의 실제 상호작용에 대한 종합 분석 결과, 최상위 VLMs의 실패 사례에 대한 중요한 통찰을 제공했습니다. 예를 들어, GPT-4V가 시각적 인식 및 추론 작업에서 Reka-Flash, Opus, Yi-VL-Plus와 같은 많은 다른 모델을 능가하지만, 미묘한 맥락적 단서, 공간 추론, 시각적 상상력 및 전문가 도메인 지식에서는 여전히 어려움을 겪고 있습니다.

- **Performance Highlights**: 현재 VLMs는 의도적으로 유도될 때 환각 및 안전성(Hallucinations and Safety)의 문제를 나타냅니다. 우리의 채팅 및 피드백 데이터를 공개하여 VLMs 연구를 더욱 발전시키고자 합니다.



### Large Language Models for Dysfluency Detection in Stuttered Speech (https://arxiv.org/abs/2406.11025)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 이 연구는 최근 대형 언어 모델(Large Language Models, LLMs)이 비문자 입력(none-lexical inputs) 예를 들어 오디오와 비디오를 잘 처리하는 능력을 바탕으로, 말더듬(stuttering) 관련 비유창성(dysfluency)을 탐지하는 작업을 언어 모델링(language modeling) 문제로 접근합니다. 영어와 독일어로 된 말더듬 데이터셋을 사용하여, ASR(Automatic Speech Recognition) 시스템과 오디오 인코더 모델을 통해 생성된 어쿠스틱(acoustic) 표현을 LLM에 제공하고, 이를 미세 조정(finetuning) 하여 비유창성 레이블을 예측하는 시스템을 제안합니다.

- **Technical Details**: 시스템은 wav2vec 2.0을 기반으로 한 어쿠스틱 인코더를 통해 잠재적 특징(latent features)을 추출하고, ASR 후보자와 어쿠스틱 특징을 결합하여 미리 훈련된 Llama 2 모델에 입력합니다. 이 모델은 LoRA (Low-Rank Adaption)를 사용하여 최적화되며, 말단위 해상도로 비유창성을 탐지합니다. 이는 어쿠스틱 특징이 좁은 시간 프레임에서 비유창성을 감지할 수 있는 반면, 어휘적 특징은 단어와 같은 자체 포함 엔티티에 중점을 둡니다.

- **Performance Highlights**: 제안된 시스템은 SEP-28k-Extended, FluencyBank, Kassel State of Fluency(KSoF) 등 세 개의 데이터셋에서 말더듬 관련 비유창성(블록, 간투어, 연장, 소리 반복, 단어 반복)을 효과적으로 탐지하며, 이로 인해 경쟁력 있는 결과를 보여줍니다. 실험 결과, 어쿠스틱과 어휘적 정보를 결합한 시스템이 높은 성능을 보였습니다.



### Optimized Speculative Sampling for GPU Hardware Accelerators (https://arxiv.org/abs/2406.11016)
- **What's New**: 이 연구에서는 병렬 하드웨어 가속기를 위한 추론 속도를 개선하기 위해 추측 샘플링(speculative sampling)을 최적화했습니다. 연구진은 추측 샘플링에 필요한 중간 행렬의 상당 부분을 동시에 계산할 수 있음을 발견했습니다. 이를 활용해 여러 GPU 스레드에 작업을 분배하고, 스레드 블록 내에서 행렬 세그먼트를 동시에 연산합니다. 또한, 중간 결과들을 빠른 온칩 메모리에 저장하여 느린 메모리 타입 간의 읽기 및 쓰기 작업을 최소화했습니다.

- **Technical Details**: 추론 속도를 높이기 위해, 점별 근사법으로 sigmoid를 사용하여 softmax로 매개 변수화된 확률 분포를 근사했습니다. 이 방법은 조금의 정확도 저하를 감수하더라도 프로파일링 시간 측면에서 큰 상대적 향상을 가져왔습니다. 연구진은 자동 음성 인식(automatic speech recognition)과 요약 작업에서 방대한 실험을 수행하여 최적화 방법의 유효성을 검증했습니다.

- **Performance Highlights**: 최적화된 검증 방법은 기본 구현에 비해 프로파일링 시간을 6%에서 13%까지 감소시켰으며, 모델 정확도에는 영향을 미치지 않았습니다. 또한, sigmoid 근사 방법은 프로파일링 시간을 37%에서 94%까지 개선했지만 약간의 정확도 저하가 발생했습니다.



### Data Shapley in One Training Run (https://arxiv.org/abs/2406.11011)
- **What's New**: 인공지능(AI) 시스템에서 저작권 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 저작권 소유자에게 그들의 기여도에 비례하여 공정한 보상을 지급하며, 협력 게임 이론(Cooperative Game Theory) 및 확률론적 기법을 활용하여 기여도를 정량적으로 평가합니다. 또한, 이 프레임워크는 고품질 훈련 데이터를 제공받아 AI 모델 성능을 향상할 수 있게 합니다.

- **Technical Details**: 이 논문은 데이터 셰플리(Data Shapley) 값을 활용하여 각 훈련 데이터 소스의 기여도를 측정하는 'In-Run Data Shapley'라는 접근법을 도입합니다. 모델 학습 과정 중 중간 모델 파라미터를 모니터링하고, 그레디언트 점-곱(Gradient Dot-Product) 또는 그레디언트-헤시안-그레디언트(Gradient-Hessian-Gradient Product) 계산을 통해 기여도를 측정합니다. 이 기법은 특별한 추가 그레디언트 벡터 또는 헤시안 매트릭스를 인스턴스화할 필요 없이 효율적으로 연산할 수 있도록 합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 제안된 프레임워크는 생성된 콘텐츠에 기여한 주요 데이터 소스를 성공적으로 식별하고, 저작권 소유자 사이에 공정하게 수익을 분배할 수 있음을 보입니다. 한편 GPT2 모델을 활용한 파일럿 스터디에서, 잘 큐레이션된 데이터에서도 약 16%는 오히려 유해한 영향을 미칠 수 있음을 확인하며, 데이터 큐레이션 개선의 필요성을 제기합니다.



### Promoting Data and Model Privacy in Federated Learning through Quantized LoRA (https://arxiv.org/abs/2406.10976)
- **What's New**: 기존의 연합 학습(Federated Learning, FL)은 데이터 프라이버시 보호에 중점을 두지만, 대형 언어 모델(LLMs) 개발에는 막대한 데이터와 연산 자원이 필요합니다. 새로운 접근 방식인 FedLPP(Federated Learning with LLM Privacy Protection)를 소개합니다. 이 기법은 모델의 매개변수를 양자화하여 학습하는 동안 데이터와 모델 프라이버시를 동시에 보호합니다.

- **Technical Details**: FedLPP에서는 양자화 기법(quantization)을 이용하여 모델의 매개변수를 클라이언트에게 배포하며, LoRA와 같은 파라미터 효율적 미세 조정 방법(parameter-efficient fine-tuning methods)을 결합하여 통신 비용을 크게 줄였습니다. 이는 클라이언트가 중앙 서버의 성능에 필적하는 모델을 얻지 못하면서도 정확한 그라디언트 추정을 가능하게 합니다.

- **Performance Highlights**: FedLPP는 네 가지 텍스트 생성 데이터셋에서 기존 방법 대비 성능을 크게 향상시켰습니다. 모델 프라이버시를 유지하면서도 성능 저하 없이 뛰어난 성능을 발휘하며, 낮은 통신 및 연산 요구를 만족시켜 현실 적용에 적합합니다.



### City-LEO: Toward Transparent City Management Using LLM with End-to-End Optimization (https://arxiv.org/abs/2406.10958)
Comments:
          26 pages, 8 figures, 5 tables

- **What's New**: 이번 연구에서는 'City-LEO'라는 대형 언어 모델(LLM) 기반의 에이전트를 제안하여 스마트시티 운영에서의 의사결정을 보다 효과적이고 투명하게 지원하고자 합니다. 이 에이전트는 기존의 운영 연구(Operations Research, OR) 모델들의 활용도를 높여 보다 정확하고 사용자 요구에 부응하는 솔루션을 제공합니다. 특히, 사용자 요구를 충족시키기 위해 LLM의 논리적 추론 능력을 활용하여 대규모 최적화 문제를 효율적으로 축소하는 것이 특징입니다.

- **Technical Details**: City-LEO는 E2E(End-to-end) 프레임워크를 통합하여 예측과 최적화를 함께 다룹니다. 이는 환경 불확실성에 대처하고 사용자 요구와 관련된 특징들을 포함하는 데 도움을 줍니다. 또한, 전체 최적화 모델을 기반으로 사용자 요구에 맞는 문제 범위를 축소함으로써 계산의 효율성을 높입니다. E2E 프레임워크는 Random Forest(RF) 목표 함수를 이용하여 예측과 최적화를 통합합니다. 이는 함축적인 의사결정을 가능하게 하며, 투명성과 해석 가능성을 높입니다.

- **Performance Highlights**: 사례 연구에서는 City-LEO를 전기 자전거 공유 시스템의 운영 관리에 적용하였습니다. 실험 결과, City-LEO는 기존의 전체 최적화 문제 대비 높은 성능을 보였으며, 더 적은 계산 시간 안에 사용자 요구에 더 부합하는 솔루션을 제공하였습니다. 또한, 전반적인 전역 최적화 성능이나 정확성을 크게 저해하지 않으면서도 더 낮은 수준의 전역적 비최적성을 달성했습니다.



### Understanding Understanding: A Pragmatic Framework Motivated by Large Language Models (https://arxiv.org/abs/2406.10937)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 주제에 대한 이해를 가지고 있는지 여부를 테스트하기 위한 새로운 프레임워크를 제안합니다. Turing Test와 유사하게, 이 프레임워크는 질문에 대한 응답 성능을 기반으로 하며, 일반적인 능력을 요구하고 '황당한 답변'을 피하는 등의 요소를 포함합니다.

- **Technical Details**: 이해의 정의는 특정 도메인 내 질문 세트에 대한 만족스러운 답변을 줄 수 있는 능력으로 구성됩니다. 만족스러운 답변의 기준으로는 평균 점수가 정해진 기준 이상이어야 하며, 황당한 답변의 확률이 매우 낮아야 한다는 것입니다. 현실적으로 도메인 내 모든 질문을 테스트하는 것은 불가능하므로, 랜덤 샘플링과 확률론적 신뢰도를 적용하여 높은 신뢰도를 달성합니다.

- **Performance Highlights**: 현재의 LLMs는 비현실적인 도메인에 대해 이해를 가지고 있다고 할 수 없지만, 답변에 설명을 추가함으로써 필요한 샘플 수를 줄일 수 있습니다. 이는 교육 환경에서 설명이 중요한 역할을 하는 이유에 대한 직관을 제공합니다. 또한, 이 프레임워크는 LLMs의 평가 및 설계에 실질적인 영향을 미칠 수 있는 도구를 제공합니다.



### Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies (https://arxiv.org/abs/2406.10923)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 영화 속 전형적 플롯(Tropes)을 통한 비디오 추론 능력을 평가하기 위해 새로운 데이터 세트인 Tropes in Movies (TiM)를 도입하였습니다. TiM은 두 가지 주요 비디오 추론 기술인 '추상적 인식(Abstract Perception)'과 '장기 구성 추론(Long-range Compositional Reasoning)'을 평가하는 데 중점을 둡니다.

- **Technical Details**: TiM 데이터 세트는 영화 스토리텔링에서 흔히 사용되는 전형적 플롯을 활용하여 최신 대형 언어 모델(LLM) 기반 접근법의 추론 능력을 평가합니다. 특히, Captioner-Reasoner, Large Multimodal Model Instruction Fine-tuning (LMM-IF), Visual Programming 등의 방법이 TiM의 문제를 풀 때 무작위 베이스라인을 약간 상회하는 성능을 보입니다. TiM의 복잡한 문제를 해결하기 위해 Face-Enhanced Viper of Role Interactions(FEVoRI)와 Context Query Reduction(ConQueR)을 도입하여 성능을 크게 개선했습니다.

- **Performance Highlights**: 기존의 SOTA 방법은 TiM 데이터 세트에서 최대 F1 점수 25를 달성하며 무작위 베이스라인을 약간 상회하는 수준을 보였습니다. 하지만, FEVoRI와 ConQueR를 도입한 후 성능이 15 F1 포인트 향상되어 40 F1를 기록하였으나 여전히 인간 수준(65 F1)에는 미치지 못했습니다. AST Based Code Dignosis(ABCD) 프레임워크를 통해 추상적 인식과 장기 구성 추론의 복잡성을 확인할 수 있었습니다.



### Embodied Question Answering via Multi-LLM Systems (https://arxiv.org/abs/2406.10918)
Comments:
          11 pages, 5 Figures, 3 Tables

- **What's New**: 본 논문에서는 Embodied Question Answering (EQA) 문제를 해결하기 위해 다중 에이전트 프레임워크를 도입했습니다. 다중 대형 언어 모델 (LLM) 기반 에이전트가 가정 환경에서 독립적으로 사용자 질문에 답변하는 방식입니다. 각 질문에 대해 하나의 답변을 생성하기 위해, 개별 응답을 종합하는 Central Answer Model (CAM)을 사용하여 더 견고한 답변을 제공합니다. CAM을 사용하면 기존의 앙상블 LLM 방법(예: 투표 또는 토론)과 비교할 때 EQA 정확도가 50% 향상되었습니다. 또한, CAM은 에이전트 간의 통신이 필요 없어 관련 비용을 줄입니다.

- **Technical Details**: CAM은 비선형 (신경망, 랜덤 포레스트, 결정 트리, XGBoost) 및 선형 (로지스틱 회귀 분류기, SVM) 알고리즘을 사용해 다양한 방법으로 실험되었습니다. 에이전트의 개별 응답을 학습하여 최종적인 질문 답변을 제공하며, 에이전트 간의 통신이 필요 없습니다. 여러 Matterport 3D 환경에서 다양한 방법을 사용하여 CAM을 학습시키고 평가했습니다.

- **Performance Highlights**: CAM을 사용하면 다중 에이전트 프레임워크에서 Matterport3D 데이터셋을 통해 50% 더 높은 EQA 정확도를 달성할 수 있었습니다. 또한, CAM의 의존도를 측정하기 위해 permutation feature importance (PFI)를 사용하여 평가했습니다.



### Breaking the Attention Bottleneck (https://arxiv.org/abs/2406.10906)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문은 attention 알고리즘의 복잡성을 줄이고 성능을 향상시키기 위한 새로운 기법을 소개합니다. 특히, 초과 매개변수(over-parameterized)를 갖는 디코더(decoder)에서 attention 메커니즘의 정적 패턴 수렴 문제를 해결하기 위해 생성적 함수(generative function)를 대체로 사용하는 방법을 제안합니다. 이를 통해 더 작은 모델로도 더 낮은 손실(loss)을 달성할 수 있습니다.

- **Technical Details**: 제안된 방법은 나노GPT(nanoGPT) 설정에서 attention 또는 활성화(activation) 대체를 통해 자동 회귀(auto-regressive) 특성을 유지합니다. 특히, 각 토큰을 이전 토큰과 비교하면서 손실을 줄이며, 평균 컨텍스트 벡터를 결합하여 더욱 손실을 줄이는 방법을 제시합니다. 실험 설정은 블록 크기 64, 배치 크기 12, 4개의 층(layers), 임베딩 크기 128, 드롭아웃(dropout) 없음, 4개의 attention 헤드를 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기준 attention보다 더 낮은 손실 값을 달성하면서 동시에 모델 크기를 줄였습니다. 특히, 평균 컨텍스트 벡터를 포함하면 검증 손실(validation loss)이 더욱 감소하는 성과를 보였습니다. 검증 손실은 표준 attention의 1.692에서 제안된 방법으로 1.638로, 평균 컨텍스트 벡터를 포함하면 1.557까지 감소하였습니다. 또한, 계산 비용도 감소시키는 장점이 있습니다.



### New Solutions on LLM Acceleration, Optimization, and Application (https://arxiv.org/abs/2406.10903)
Comments:
          This is an expanded and more comprehensive study based on our invited DAC-24 paper with the same title and co-authors

- **What's New**: 최근 대형 언어 모델(LLMs)의 효율성을 향상시키기 위한 여러 연구 방향을 종합적으로 검토합니다. 이 연구는 LLM 추론 속도 및 자원 활용을 최적화하기 위한 알고리즘 수준의 가속화 기술, LLM 요구사항에 맞춘 하드웨어 설계, LLM에 최적화된 하드웨어 가속기 컴파일 접근법을 포함합니다. 또한 High-Level Synthesis(HLS) 기능 검증을 위한 LLM 지원 설계 방법론을 사례 연구로 제안합니다.

- **Technical Details**: [{'Algorithm-level Accelerations': 'LLM 추론 시간을 줄이기 위한 다양한 알고리즘 전략을 탐구합니다. 초기 종료 및 레이어 스키핑(layer skipping), 문맥적 희소성(contextual sparsity) 기법을 다루며, Mixture of Experts(MoE) 접근법도 언급합니다. 추가로 병렬 디코딩(parallel decoding) 기술과 효과적인 Key-Value(KV) 캐시 최적화 방법을 제안합니다.'}, {'LLM-Hardware Co-design': '효율적인 LLM 배포를 위해 LLM과 하드웨어 아키텍처를 맞춤 설계하는 전략을 제시합니다. AutoDistill과 같은 모델 압축 및 Transformer 아키텍처 탐색 프레임워크를 소개하며, 하드웨어 효율성을 높이기 위한 가지치기 인식 양자화(pruning-aware quantization) 전략을 설명합니다.'}, {'LLM-to-accelerator Compilation': 'Model 아키텍처를 하드웨어 구현으로 빠르게 전환하기 위한 High-Level Synthesis(HLS) 프레임워크의 필요성을 다룹니다. ScaleHLS와 HIDA와 같은 새로운 컴파일레이션 프레임워크를 요약하며, PyTorch 모델을 Synthesizable C 코드로 변환하는 과정도 설명합니다.'}]

- **Performance Highlights**: SnapKV와 Medusa와 같은 새로운 접근법은 추론 시간을 대폭 단축하는 동시에 메모리 효율성을 향상시킵니다. Chrysalis 데이터셋과 HLS-특화 Debugging Assistant는 HLS 기능 검증을 가속화하고 생산성을 향상시키며, 하드웨어 설계의 시장 출시 시간을 줄이는 데 기여합니다.



### Light Up the Shadows: Enhance Long-Tailed Entity Grounding with Concept-Guided Vision-Language Models (https://arxiv.org/abs/2406.10902)
- **What's New**: 이번 연구에서는 다중 모달 지식 그래프(Multi-Modal Knowledge Graph, MMKG)에서 긴 꼬리 엔터티(long-tailed entities)의 이미지를 정확하게 매칭하는 새로운 프레임워크 COG를 소개합니다. 기존의 방법들이 웹 검색을 통해 이미지를 수집하는 데 어려움을 겪는 반면, COG는 개념 유도(COncept-Guided)를 사용해 이 문제를 해결하고자 합니다.

- **Technical Details**: COG 프레임워크는 두 개의 주요 모듈로 구성됩니다. 먼저, 개념 통합 모듈(Concept Integration Module)을 통해 긴 꼬리 엔터티의 이미지-텍스트 쌍을 효과적으로 식별합니다. 이후, 증거 융합 모듈(Evidence Fusion Module)을 통해 설명 가능성을 높이고 인간 검사(human verification)가 가능하도록 합니다. 이 프레임워크는 교체 가능한 사전 훈련된 비전-언어 모델(Pre-trained Vision-Language Models, PVLMs)을 사용하여 유연하게 애플리케이션에 적용할 수 있습니다.

- **Performance Highlights**: 25,000개의 긴 꼬리 엔터티 이미지-텍스트 쌍 데이터를 생성하여 COG의 효과를 입증했습니다. 실험 결과, COG는 기존 방법들과 비교하여 긴 꼬리 이미지-텍스트 쌍 인식 정확도가 크게 향상되었음을 확인했습니다. 또한, 유연성과 설명 가능성을 제공하여 품질 관리 및 인간 검사가 가능합니다.



### AUTOHALLUSION: Automatic Generation of Hallucination Benchmarks for Vision-Language Models (https://arxiv.org/abs/2406.10900)
- **What's New**: 기존의 수작업 벤치마크에서는 일반화가 어려운 한계가 있어, 첫 자동화 벤치마크 생성 접근 방식인 AUTOHALLUSION이 개발되었습니다. 이 새로운 접근 방식은 다양하고 다채로운 환각(환영) 사례를 생성하여 LVLM(Large Vision-Language Models)의 언어 모듈의 문맥 단서를 이용해 이미지를 합성합니다. 합성된 이미지와 설정된 질문으로 LVLM이 문맥적 편견과 시각적 방해 요소에 어떻게 대응하는지 평가합니다.

- **Technical Details**: AUTOHALLUSION은 세 가지 주요 전략을 사용하여 시나리오와 충돌하는 이미지를 생성합니다: (1) 문맥 단서에 맞지 않는 비정상적인 객체 추가, (2) 함께 발생하는 두 객체 중 하나를 유지하고 다른 하나는 제거, (3) 문맥과 밀접하게 관련된 객체 제거. 이러한 방식으로 생성한 이미지와 질문들은 LVLM의 언어 모듈이 과도하게 의존하는 문맥적 편견을 유발하여 부정확하거나 모순된 답변을 생성하는지 확인합니다.

- **Performance Highlights**: AUTOHALLUSION은 최상위 LVLM, 예를 들어 GPT-4V(ision), Gemini Pro Vision, Claude 3, LLaVA-1.5을 대상으로 평가했습니다. 이 모델들은 각기 97.7%와 98.7%의 환각 유도 성공률을 기록했으며, 이는 AUTOHALLUSION의 높은 성능을 입증하는 것입니다. 이로써 LVLM의 환각 문제를 지속적으로 해결하기 위한 첫 걸음을 내딛었습니다.



### Demonstration Notebook: Finding the Most Suited In-Context Learning Example from Interactions (https://arxiv.org/abs/2406.10878)
- **What's New**: 본 논문은 'demonstration notebook'이라는 새로운 객체를 중심으로 하는 프롬프트 엔지니어링(workflow)을 제안하고 있습니다. 이는 특정 질문에 맞춘 데모(현재까지의 상호작용 정보를 활용)를 자동으로 생성하고 선택하는 방법입니다. 이 접근법이 여러 추론 벤치마크에서 최첨단 성능을 달성했다고 합니다.

- **Technical Details**: 본 연구에서는 데이터셋의 내재적 이질성을 고려하지 않고도 동일한 데모를 모든 추론 질문에 적용하는 기존 방식의 문제점을 지적합니다. 'demonstration notebook'은 LLM(대형 언어 모델)의 과거 상호작용 정보를 수집 및 재사용하여 각 질문에 가장 적합한 in-context 학습 예제를 식별합니다. 이를 통해 자동 데모 구성 및 선택에서 최첨단 결과를 얻었습니다.

- **Performance Highlights**: 제안된 방법은 텍스트 요약 및 프롬프트 압축(Prompt Compression) 작업에서도 성공적으로 적용되었습니다. 또한, 'demonstrative regime'에 대한 철저한 분석 방법을 제시하여 데모가 데이터셋 내의 다양한 질문 유형과 어떻게 연관되는지에 대한 귀중한 통찰력을 제공합니다.



### Optimizing Automatic Speech Assessment: W-RankSim Regularization and Hybrid Feature Fusion Strategies (https://arxiv.org/abs/2406.10873)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 최근 연구에서는 self-supervised feature (SSL)의 활용으로 Automatic Speech Assessment (ASA)에서 눈에 띄는 발전이 이루어졌습니다. 본 논문에서는 영어 시험 데이터셋에서 두드러지는 데이터 불균형 문제를 해결하기 위해, ASA을 순서형 분류 문제로 접근하고 Weighted Vectors Ranking Similarity (W-RankSim)을 새로운 정규화 기술로 도입했습니다. W-RankSim은 출력을 위한 가중 벡터들이 유사한 클래스들에서 더 가까이 위치하도록 유도합니다.

- **Technical Details**: W-RankSim은 RankSim을 기반으로 하여 배치 크기의 제약을 극복하고 각 클래스에 대한 기울기를 효과적으로 축적할 수 있도록 개선했습니다. 또한, 기계적 특징과 handcrafted feature을 결합한 하이브리드 모델을 제안하여, ASA 시스템의 성능을 향상시킵니다. 특히 데이터 불균형 문제를 해결하기 위해, CEFR 수준 등의 점수를 순서형 정보로 활용합니다. W-RankSim은 개별 벡터 간의 코사인 유사도를 계산하여 가중치 벡터 유사도 행렬(Sw)을 구성합니다.

- **Performance Highlights**: 실험 결과, W-RankSim은 RankSim의 배치 크기 제약을 극복하고, 순서형 분류 과제에서 우수한 성능을 보였습니다. 제안된 하이브리드 모델은 자가 지도 학습(SSL) 기능과 handcrafted feature을 결합함으로써, 전반적인 평가 성능을 향상시켰습니다. 본 논문에서는 ASA을 불균형한 순서형 분류 문제로 정의하고, 이를 개선하기 위한 실질적인 방법을 제안한 최초의 연구입니다.



### TorchOpera: A Compound AI System for LLM Safety (https://arxiv.org/abs/2406.10847)
- **What's New**: TorchOpera를 소개합니다. 이는 Large Language Models(LLM)의 프롬프트와 응답의 안전성과 품질을 향상시키기 위해 설계된 복합 AI 시스템입니다. TorchOpera는 사용자 프롬프트가 안전하고, 문맥적으로 맞춰지며, 효과적으로 처리되도록 보장하고, LLM 응답을 관련성 있고 높은 품질로 개선합니다.

- **Technical Details**: TorchOpera는 벡터 데이터베이스(Vector Database)를 이용한 문맥적 맞춤(Contextual Grounding), 유연한 수정이 가능한 규칙 기반 래퍼(Rule-based Wrappers), 안전하지 않거나 잘못된 콘텐츠를 감지 및 조정하는 전문 메커니즘을 활용합니다. 안전성 감지 노드(Safety Detection Node), 문맥 설정 노드(Grounding Node), 오류 수정 노드(Repair Node) 등 여러 주요 노드를 조율하여 개별 기능을 전문화합니다.

- **Performance Highlights**: 다양한 실험 결과, TorchOpera는 실세계 환경에서 LLM의 안전성과 신뢰성, 적용 가능성을 보장하면서도 응답의 효율성을 유지함을 확인할 수 있었습니다. 시스템 전체 수명 주기 동안 LLM 상호작용의 안전성을 높여 안정적이고 확장 가능한 LLM 애플리케이션 구현을 가능하게 합니다.



### Reminding Multimodal Large Language Models of Object-aware Knowledge with Retrieved Tags (https://arxiv.org/abs/2406.10839)
Comments:
          18 pages, 11 figures

- **What's New**: 최근 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLM)은 일반적인 시각 지시를 잘 따라가는 능력을 보였으나, 새로운 객체와 엔터티를 인식하거나 이미지의 세부 사항을 정확히 전달하는 데 어려움을 겪고 있습니다. 이런 문제를 해결하기 위해, 우리는 새로운 태그-기반 시각 지시 튜닝 방식인 TUNA를 도입하였습니다. TUNA는 객체 인식 향상을 위해 풍부한 객체 정보가 포함된 태그를 활용합니다.

- **Technical Details**: 기존의 다중모달 커넥터는 종종 훈련 데이터가 부족해 문제를 일으킵니다. 이를 극복하기 위해, 우리는 외부 데이터 저장소에서 태그를 검색하여 더욱 풍부한 객체 정보를 제공하는 태그-기반 지시 튜닝 방법(Tuna)을 제안합니다. 이를 통해 객체 이름과 속성에 대한 정보를 포함한 태그를 이미지와 매핑하여 더 정확한 텍스트 임베딩을 생성할 수 있습니다.

- **Performance Highlights**: TUNA는 같은 언어 모델과 훈련 데이터를 사용하는 기존 모델들을 12개의 벤치마크에서 뛰어넘는 성능을 보였습니다. 추가로, 특정 데이터 저장소와 함께 사용할 경우, 제로샷(Zero-Shot) 역량을 보여주기도 했습니다.



### GUI-WORLD: A Dataset for GUI-oriented Multimodal LLM-based Agents (https://arxiv.org/abs/2406.10819)
- **What's New**: 최근 나온 논문에서는 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 이용하여 그래픽 사용자 인터페이스(Graphical User Interface, GUI)를 직접 인식하고 이에 맞는 코드를 생성하는 방법을 제안하였다. 그러나 현재의 에이전트는 주로 정적인 환경에서 우수한 이해 능력을 보여주며, 웹이나 모바일 인터페이스와 같은 상대적으로 단순한 도메인에 주로 적용된다. 본 논문에서는 역동적인 웹 콘텐츠 및 다단계 작업을 포함한 GUI의 시간 정보를 인식할 수 있고, 데스크톱 소프트웨어와 다중 창 상호작용을 포함한 다양한 GUI 시나리오에 대한 포괄적인 이해를 해야 한다고 주장한다. 이를 위해 새로운 데이터셋 'GUI-World'를 소개한다.

- **Technical Details**: GUI-World 데이터셋은 인간-MLLM(Human-MLLM) 주석을 포함하며, 6개의 GUI 시나리오와 3가지 형식으로 구성된 8가지 GUI 지향 질문 유형을 다루고 있다. 현존하는 최첨단 MLLMs, 특히 ImageLLMs와 VideoLLMs를 평가하였다. ImageLLMs는 수동으로 주석이 달린 키프레임이나 작업 기록 없이는 동적 GUI 콘텐츠를 처리하는 데 어려움을 겪었고, VideoLLMs는 희소한 GUI 비디오 데이터셋으로 인해 모든 GUI 지향 작업에서 부족한 성과를 보였다. GUI-World를 기반으로 VideoLLM을 GUI 에이전트로 활용하는 초기 단계를 진행했으며, 다양한 GUI 작업에 대한 개선된 이해를 시연하였다.

- **Performance Highlights**: VideoLLM을 통해 GUI 작업 이해도가 개선되었지만, 기본 LLM의 성능 한계상 VideoLLMs를 GUI 에이전트로 사용하는 데는 여전히 큰 도전과제가 따른다는 결론을 내렸다. 이번 연구는 역동적인 GUI 콘텐츠 이해에 관한 미래 연구에 귀중한 통찰력을 제공한다고 믿는다.



### HiddenTables & PyQTax: A Cooperative Game and Dataset For TableQA to Ensure Scale and Data Privacy Across a Myriad of Taxonomies (https://arxiv.org/abs/2406.10803)
Comments:
          In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)

- **What's New**: 최근 arxiv에 발표된 'HiddenTables'라는 새로운 협력형 게임에 대한 연구는 대규모 언어 모델(LLM)들이 표 기반 질의응답(Table QA) 과제를 해결하는 데 있어 마주하는 다양한 문제를 해결하려는 시도를 소개하고 있습니다.

- **Technical Details**: HiddenTables는 코드 생성 LLM(Solver)과 평가자(Oracle)의 협력 게임입니다. 여기서 Solver는 오라클이 제공하는 자연언어 구조(Natural Language Schema) 및 지시 사항에 따라 사용자 질문에 대한 코드를 생성하고, 이를 바탕으로 Oracle이 데이터를 보존하면서 답변을 제공합니다. 이 게임은 제한된 문맥 창, 토큰화 패턴과 셀 경계 사이의 불일치, 외부 모델 사용시 생기는 데이터 기밀성 문제 등을 해결하려는 목적으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, HiddenTables는 전체 데이터를 노출하지 않고도 다량의 데이터셋과 상호작용할 수 있는 효율성을 보여주었습니다. 새로운 데이터셋 'PyQTax'를 생성하여 116,671개의 질문-테이블-정답 트립렛을 포함하고, 다양한 질문 분류에 대한 세밀한 분류 및 라벨을 제공합니다. 또한 이 게임을 통해 gpt-3.5-turbo의 효율성을 높였으며, 피드백을 반복적으로 주고받는 과정에서 정확도가 개선되었습니다.



### Evaluating LLMs with Multiple Problems at once: A New Paradigm for Probing LLM Capabilities (https://arxiv.org/abs/2406.10786)
Comments:
          20 pages, 15 figures, 9 tables

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 단일 문제에 대해 평가하는 기존 방식에서 벗어나, 다중 문제 처리를 평가하는 새로운 접근 방식을 제안합니다. 7개의 LLM을 6개의 분류 벤치마크에서 구성한 4가지 유형의 작업에 대해 포괄적으로 조사하였습니다. 특히 다중 문제 평가가 LLM의 종합적인 문제 해결 능력에 미치는 영향을 연구합니다.

- **Technical Details**: 검사한 작업 유형은 전통적인 단일 문제, 동질적 다중 문제, 그리고 다중 문제를 포함한 2가지 인덱스 선택 작업입니다. 실험은 zero-shot, few-shot, few-shot-CoT(chain of thought) 설정하에 실행되었습니다. 이 연구는 LLM이 긴 입력에서도 위치 편향(Position Bias)을 많이 겪지 않으며, 다중 문제 처리에서도 단일 문제 처리와 유사한 성능을 보임을 발견했습니다.

- **Performance Highlights**: 다중 문제 평가 방식이 LLM의 성능에 큰 영향을 주지 않으며, 비용 효율성도 크게 향상시키는 것으로 나타났습니다. 예를 들어, 여러 문제를 하나의 프롬프트로 제시할 때 GPT-4의 접근 비용이 최대 82% 감소했습니다. 그러나 LLM이 진정한 이해를 결여하고 있음을 보여줍니다. 특히, 인덱스 선택 작업에서 성능이 현저히 떨어지는 것을 볼 수 있습니다. 이는 LLM이 확률적으로 답변을 생성하는 경향이 있기 때문입니다.



### How Should We Extract Discrete Audio Tokens from Self-Supervised Models? (https://arxiv.org/abs/2406.10735)
Comments:
          4 pages, 2 figures, 2 tables, Accepted at Interspeech 2024

- **What's New**: 이 논문은 오디오 토큰화(Audito Tokenization) 방법의 최적 구성을 탐색하며, 특히 셀프-슈퍼바이즈드 학습(Self-Supervised Learning, SSL) 모델의 양자화(Quantization)에 대한 최적 설정을 연구합니다. 논문에서는 여러 SSL 레이어를 가로지르는 범용 보코더(Vocoder)를 훈련하는 확장 가능한 솔루션을 제안하며, 다양한 오디오 애플리케이션에서 의미적 토큰(semantic tokens)의 적응성(adaptability)과 성능을 향상시키기 위해 어텐션 메커니즘(attention mechanism)을 사용합니다.

- **Technical Details**: 제안된 토큰화 아키텍처는 네 개의 주요 구성 요소로 구성됩니다: 토크나이저(Tokenizer), 정보화 레이어 선택기(Informed Layer Selector), 어쿠스틱 모델(Acoustic Model), 그리고 확장 가능한 보코더(Scalable Vocoder). 각 SSL 모델의 여러 레이어에서 k-평균 알고리즘(k-means algorithm)을 사용하여 클러스터링을 수행하고, 각 클러스터는 독립적으로 고유한 인덱스를 할당받습니다. 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 기반의 어텐션 메커니즘을 통해 각 레이어의 정보를 통합하고, 다중 레이어 오디오 토큰의 정보는 다운스트림 작업에 맞게 모델이 배운 주의(attention) 가중치를 통해 최적화됩니다. 이를 통해 모델의 성능을 개선하고 각 레이어의 상대적 중요도를 분석할 수 있습니다.

- **Performance Highlights**: 제안된 확장 가능한 보코더는 특정 레이어에서 훈련된 보코더보다 뛰어난 성능을 보여줍니다. 또한, 이 메커니즘은 단일 SSL 레이어의 정보를 사용하는 모델과 비교할 때 성능이 향상되는 것을 보였습니다. 이를 통해 제안된 아키텍처가 오디오 인식, 화자 인식, 감정 분류, 스피치 강화 및 텍스트-투-스피치와 같은 다양한 오디오 생성 및 변별 작업에서 우수한 성능을 발휘함을 확인할 수 있습니다.



### SyntheT2C: Generating Synthetic Data for Fine-Tuning Large Language Models on the Text2Cypher Task (https://arxiv.org/abs/2406.10710)
Comments:
          19 pages, 15 figures, 8 tables

- **What's New**: 대규모 언어 모델(LLMs)을 기존의 지식 그래프(KG) 데이터베이스와 통합하여 LLMs의 효율성을 높이고 '환각' 문제를 완화하려는 연구가 진행 중입니다. 이 연구에서는 자연어를 Cypher 쿼리로 변환하는 기술(Text2Cypher)을 자동화하는 새로운 방법론인 SyntheT2C을 제안합니다. 이 방법론은 LLM 기반 프롬프트와 템플릿 채우기 두 가지 파이프라인으로 구성됩니다. SyntheT2C는 다양한 Query-Cypher 쌍을 생성하며, 이를 이용해 백본 LLM의 성능을 향상시킵니다.

- **Technical Details**: SyntheT2C는 LLM 기반 프롬프트와 템플릿 채우기 두 가지 파이프라인으로 구성됩니다. LLM 기반 프롬프트 파이프라인은 더 많은 의미론적 유연성을 가진 Cypher 쿼리를 생성하며, 템플릿 채우기 파이프라인은 구문적으로 복잡한 Cypher 쿼리를 생성합니다. 생성된 쿼리 쌍은 자동 및 수동 검증을 거쳐 백본 LLM을 고도화하는 데 사용됩니다. 이 방법론은 두 개의 의료 데이터베이스에서 테스트되었습니다: LHY 데이터베이스와 Hetionet 데이터베이스.

- **Performance Highlights**: SyntheT2C를 통해 생성된 합성 데이터셋인 MedT2C는 백본 LLM의 Cypher 쿼리 작성 능력을 크게 향상시켰습니다. 실험 결과, LLM을 고도화한 후 Cypher 쿼리의 성능이 향상되었으며, 이를 통해 자연어로 KG 데이터베이스와 상호작용할 수 있는 가능성이 확인되었습니다.



### CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training (https://arxiv.org/abs/2406.10670)
- **What's New**: 이번 연구에서는 언어 모델의 사전 훈련에 사용할 고품질 데이터를 선택하는 새로운 방법인 CoLoR-Filter (Conditional Loss Reduction Filtering)를 제안합니다. 이는 두 개의 보조 모델의 상대적인 손실 값을 기반으로 하는 간단하고 계산 효율적인 선택 기준을 사용하며, 경험적 베이즈 접근법에 영감을 받았습니다.

- **Technical Details**: CoLoR-Filter는 베이즈 규칙과 경험적 베이즈를 적용하여 파생된 방법으로서, 시퀀스의 우도를 '우선' 모델과 '조건' 모델 간의 차이로 스코어링합니다. 이를 통해 고품질 데이터를 선별합니다. 이 방법은 RHOLoss와 같은 기존 방법들에 비해 계산 비용이 적고 평행화가 가능하다는 장점이 있습니다.

- **Performance Highlights**: CoLoR-Filter는 C4 데이터에서 Books 도메인 적응을 위해 데이터를 선택하여, 기존의 무작위로 선택된 데이터의 25배 적은 데이터로 1.2b 파라미터 모델의 성능을 달성할 수 있음을 보여주었습니다. 또한, 8가지의 다운스트림 다중 선택 질문 응답 작업에서 CoLoR-Filter를 사용한 데이터 선택이 무작위로 선택된 데이터의 11배 적은 데이터로 동일한 성능을 발휘함을 확인하였습니다.



### Optimization-based Structural Pruning for Large Language Models without Back-Propagation (https://arxiv.org/abs/2406.10576)
Comments:
          17 pages

- **What's New**: 최근 연구에서 Large-Language Models (LLMs)의 효율적인 구조적 가중치 가지치기 (pruning)를 위한 새로운 최적화 기반 방법을 제안했습니다. 이 방법은 확률 공간에서 가지치기 마스크를 직접 학습하여, 기존의 후발 학습 (post-training) 단계에서의 값비싼 가중치 미세 조정 (fine-tuning)을 피하면서도 성능을 최적화합니다. 새로운 방법은 LLM의 순 전달 (forward pass) 만을 필요로 하며, 백전파 (back-propagation)를 제거해 효율성을 증대시켰습니다.

- **Technical Details**: 이 접근법은 Bernoulli 분포를 사용해 이진 가지치기 마스크를 샘플링합니다. 가지치기 마스크를 학습하기 위해 LLM의 손실과 분리된 잠재적인 Bernoulli 분포 파라미터를 사용해, 정책 경사 추정기 (policy gradient estimator)를 통한 최적화를 실행합니다. 이 방법은 채널, 헤드, 레이어 수준의 구조적 단위에서 작동하며, 글로벌 및 이질적인 가지치기를 지원합니다. 또한, 초기화 시 메트릭 기반 방법을 선택적으로 사용할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 LLaMA, LLaMA-2 및 Vicuna 모델에서 C4와 WikiText2 데이터셋을 사용하여 2.7시간 동안 35GB 메모리로 실행되었으며, 13B 매개변수를 가진 모델에서 최첨단 성능을 달성했습니다. 특히 오류율 (perplexity) 측면에서 뛰어난 성능을 보였으며, 30%에서 50%의 가지치기 비율로도 높은 정확도를 유지했습니다.



### Lightweight Audio Segmentation for Long-form Speech Translation (https://arxiv.org/abs/2406.10549)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이번 연구는 음성 번역 시스템(ST)에서 효율적으로 음성을 세분화하는 새로운 모델을 제안합니다. 새로운 모델은 작은 크기로 높은 번역 품질을 달성하며, ASR-with-punctuation이라는 효과적인 사전 학습 전략을 도입하여 성능 개선을 이루었습니다. 또한, 실시간 번역 품질을 향상시키기 위해 음성 세분화 모델을 ST 시스템에 적절히 통합하는 중요성을 강조합니다.

- **Technical Details**: 음성 세분화 작업을 프레임 단위 분류 작업으로 설정하였고, 모델은 Conformer-M 인코더 아키텍처를 사용합니다. 20초 길이의 입력 오디오를 변환 후, 로그 멜(log-mel) 음향 특징을 출력 벡터로 변환하여 확률값을 계산합니다. 교차 엔트로피 손실(cross-entropy loss)을 사용하여 학습하며, 모델 크기는 27.3M 파라미터로 이전 연구보다 현저히 작습니다. 또한, Inference 시에는 20초 길이로 2초 중첩되게 분할하여 확률 값들을 평균화하여 세그먼트를 만들고, pTHR 등의 알고리즘을 사용합니다.

- **Performance Highlights**: 제안된 세그먼트 모델은 이전 방법들보다 8%에서 14% 적은 크기를 가지면서도 더 높은 번역 품질을 보여줍니다. 이는 다양한 ST 시스템 간의 불일치를 줄임으로써 실현되었으며, 경량화된 특성으로 스트리밍 및 온-디바이스 시나리오에 적합합니다.



### Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning (https://arxiv.org/abs/2406.10522)
- **What's New**: 새로운 멀티모달 선호 데이터를 통해 창의적 과제에서 2억 5천만 개 이상의 인간 평가와 2백 2십만 건 이상의 캡션을 제공합니다. 이는 뉴요커의 주간 만화 캡션 콘테스트를 통해 8년간 수집된 데이터입니다. 이 독특한 데이터셋은 유머러스한 캡션 생성을 위한 멀티모달 대규모 언어 모델과 선호 기반 미세 조정 알고리즘의 개발 및 평가를 지원합니다.

- **Technical Details**: 이 논문은 AI 정렬을 조사하기 위한 데이터셋과 벤치마크를 소개합니다. 데이터셋은 뉴요커 만화 캡션 콘테스트에서 수집된 2억 5천만 개 이상의 인간 평가를 포함합니다. 모델 생성 캡션의 품질을 평가하기 위해 GPT4와 인간 판단을 모두 활용하는 새로운 벤치마크를 제안합니다. 실험 결과 RLHF(Reinforcement Learning from Human Feedback)와 DPO(Direct Preference Optimization)와 같은 현재 미세 조정 방법이 창의적 작업에 한계가 있음을 강조합니다.

- **Performance Highlights**: 최신 모델인 GPT4 및 Claude가 유머러스한 캡션 생성에서 최고 인간 참가자를 능가하지 못하는 한계를 보였습니다. 벤치마크를 통해 SOTA(State-Of-The-Art) 모델들의 성능을 인간 생성 예제와 비교하여 현재 AI 시스템의 강점과 개선할 영역을 식별할 수 있었습니다.



### Reactor Mk.1 performances: MMLU, HumanEval and BBH test results (https://arxiv.org/abs/2406.10515)
- **What's New**: 논문에서는 Reactor Mk.1이라는 ARCs의 대표 대형 언어 모델(Large Language Model)의 성능 결과를 벤치마킹 분석을 통해 제시합니다. 이 모델은 Lychee AI 엔진을 활용하며 100 billion(십억) 미만의 파라미터를 보유하고 있어 효율성과 강력함을 겸비하고 있습니다.

- **Technical Details**: Reactor Mk.1는 MMLU 데이터셋에서 92%, HumanEval 데이터셋에서 91%, BBH 데이터셋에서 88%의 점수를 기록하며 GPT-4o, Claude Opus, Llama 3 등의 모델을 능가했습니다. 이 모델은 어려운 작업 처리 및 추론에 뛰어난 능력을 보유하고 있어 현재 최첨단 AI 기술 분야에서 두각을 나타내고 있습니다.

- **Performance Highlights**: Reactor Mk.1는 MMLU, HumanEval, BBH 데이터셋에서 각각 높은 점수를 기록하며 경쟁 모델들을 능가하였습니다. 특히 어려운 작업 관리와 추론 능력에서 뛰어난 성과를 보이며, 현재 AI 기술 발전의 중요한 솔루션으로 자리잡고 있습니다.



### Articulatory Phonetics Informed Controllable Expressive Speech Synthesis (https://arxiv.org/abs/2406.10514)
- **What's New**: 이번 연구는 발음 음성학(articulatory phonetics) 관점에서 새로운 표현 음성 합성(expressive speech synthesis) 프레임워크를 제안합니다. Glottalization, Tenseness, Resonance(GTR)의 세 가지 차원을 정의하고, 이를 통해 음성 합성을 안내하고자 합니다. 또한, 프로페셔널 성우가 125가지 다른 GTR 조합으로 발음한 20개의 중국어 문장을 포함하는 고품질 음성 데이터셋(GTR-Voice)을 제작했습니다.

- **Technical Details**: 연구팀은 Glottalization, Tenseness, Resonance (GTR)라는 세 가지 기본 차원에서 발음 수준의 표현 음성 합성을 조사했습니다. GTR-Voice 데이터셋은 3.6시간 분량의 음성 오디오로 구성되며, 평균 6초의 2500개의 클립으로 구성됩니다. 이 음성은 상업용 성우에 의해 녹음되었으며, Glottalization 5개, Tenseness 5개, Resonance 7개의 라벨로 분류되었습니다. 이 라벨을 기반으로 총 125개의 서로 다른 GTR 유형이 있습니다. 이를 통해 FastPitch 및 StyleTTS 두 가지 TTS 모델에서 GTR 제어 가능성을 실험했습니다.

- **Performance Highlights**: 적응 모델들은 GTR 차원 각각에 대해 세밀한 제어력을 보여주며, 주관적 평가에서 생성된 음성의 품질과 자연스러움도 훌륭히 평가되었습니다. 이 연구로 인해 GTR 차원에서의 정밀 제어가 가능해졌으며, 이는 중국어와 영어를 포함한 다중 언어 시나리오에서 검증되었습니다.



### Benchmarking Children's ASR with Supervised and Self-supervised Speech Foundation Models (https://arxiv.org/abs/2406.10507)
Comments:
          To appear in Interspeech 2024

- **What's New**: 스피치 파운데이션 모델(Speech Foundation Models, SFMs)이 다양한 스피치 태스크에서 최첨단 결과를 달성했지만, 아동 음성인식(ASR)에서의 성능은 체계적으로 연구되지 않았습니다. 이에 따라, 아동 ASR의 표준 평가를 위한 벤치마크가 없는 상황에서 이 논문은 OGI 및 MyST와 같은 여러 아동 스피치 데이터베이스를 기반으로 한 포괄적인 벤치마크를 제시하며, 다양한 SFM(Whisper, Wav2vec2.0, HuBERT, WavLM)들의 성능을 연구합니다. 또한, 데이터 증가 및 파라미터 효율적인 파인튜닝(PEFT) 방법을 비교하고, 특히 큰 모델에서는 PEFT가 전체 파인튜닝과 같은 성능을 보여주지만 작은 모델에서는 더 나쁜 성능을 보인다는 흥미로운 결과를 발견했습니다. 이와 함께, 증강된 데이터를 사용한 파인튜닝을 안정화하기 위해 PIF(Perturbation Invariant Finetuning) 손실법을 제안합니다.

- **Technical Details**: 논문에서는 Whisper, Wav2vec2.0, HuBERT, WavLM과 같은 다양한 SFMs의 아동 ASR 성능을 평가합니다. 또한 데이터 증가 방법으로는 피치 변조(Pitch Perturbation, PP), 속도 변조(Speed Perturbation, SP), 성대 길이 변조(Vocal Tract Length Perturbation, VTLP), 그리고 SpecAugment(SA)를 사용했습니다. PEFT 방법으로는 Low Rank Adaptation (LoRA), 어댑터 튜닝(Adapter Tuning), 프롬프트 튜닝(Prompt Tuning), 프리픽스 튜닝(Prefix Tuning)을 비교했습니다. 데이터 증가는 Whisper-small 모델을 기준으로 두 개의 증강 발화를 생성했고, LoRA는 각 어텐션 레이어의 쿼리 및 값 관련 파라미터에, 어댑터 튜닝은 인코더와 디코더 각 블록에 레지듀얼 어댑터를, 프롬프트 및 프리픽스 튜닝은 입력 시퀀스와 각 레이어 입력에 프롬프트를 삽입했습니다.

- **Performance Highlights**: 실험은 MyST 및 CSLU OGI 아동 스피치 데이터베이스에서 수행되었으며, MyST는 물리, 지리, 생물과 같은 주제로 가상 튜터링 세션에서 녹음된 약 240시간의 대화형 어린이 스피치 데이터로 구성되어 있습니다. Whisper 모델은 비교적 업데이트 빈도가 낮은 스크립트 방식을 사용했는데, 이로 인해 테스트 샘플이 30초 이상인 경우 결과가 불안정했으며, 이러한 샘플을 제거함으로써 모델 성능을 안정화했습니다.



### Task Facet Learning: A Structured Approach to Prompt Optimization (https://arxiv.org/abs/2406.10504)
- **What's New**: 이 논문에서는 신규 알고리즘인 UniPrompt을 소개하며, 이는 주어진 작업의 기본 설명과 학습 예제를 바탕으로 큰 언어 모델(LLM)을 최적화하기 위한 텍스트 프롬프트를 생성합니다. 기존의 알고리즘 접근 방식이 복잡한 작업을 해결하기 위한 다양한 측면을 포괄하지 못할 수 있다는 점에서 출발하였습니다.

- **Technical Details**: UniPrompt는 프롬프트를 생성할 때 구조적인 접근 방식을 채택합니다. 먼저, 프롬프트가 상대적으로 독립적인 영향을 미치는 느슨하게 결합된 의미 섹션으로 나뉠 수 있다는 점을 발견했습니다. 또한, 입력 공간을 클러스터링하여 다양한 특성 학습을 가능하게 합니다. UniPrompt 알고리즘은 초기 후보 생성을 위한 생성 모델(generative model)과 피드백 메커니즘을 포함하여 섹션별로 후보를 수정해 나가는 절차를 거칩니다. 이에 따라, 다양한 미니배치를 통해 피드백을 받아 각 섹션에 대한 개념적인 설명을 구성합니다.

- **Performance Highlights**: 여러 데이터셋과 실제 작업에 대한 실험 평가에서, UniPrompt가 인간이 조정한 프롬프트 및 최첨단 방법보다 높은 정확성을 달성했습니다. 특히, 기존 방법들이 생성할 수 없는 긴 프롬프트를 생성할 수 있습니다. 예를 들어, 혐오 발언 데이터셋 Ethos에서는 UniPrompt가 94%의 정확도를 기록한 반면, 다음 최고 방법은 82%의 정확도만을 기록했습니다. 또한, 실제 웹 검색 엔진의 의미적 매칭 작업에서 UniPrompt로 생성된 프롬프트는 수작업 프롬프트보다 정확도가 약 5% 증가하여 성능 개선에 기여했습니다.



### TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation (https://arxiv.org/abs/2406.10450)
- **What's New**: 최근 대규모 언어 모델(LLM)을 활용한 차세대 추천 시스템(RecSys)을 발전시키려는 관심이 높아지고 있습니다. 이를 위해 사용자와 아이템을 효율적으로 토큰화(indexing) 하는 것이 필수적입니다. TokenRec라는 새로운 프레임워크를 제안하여 효과적인 ID 토큰화 전략과 효율적인 검색 패러다임을 제공합니다.

- **Technical Details**: TokenRec는 'Masked Vector-Quantized (MQ) Tokenizer'라는 전략을 도입해 협력 필터링을 통해 학습된 사용자의 아이템 표현을 이산 토큰으로 양자화합니다. 이를 통해 고차원의 협력 지식을 LLM에 원활히 통합하고, 새로운 사용자나 아이템에 대해서도 일반화가 가능합니다. 또한 생성적 검색 패러다임을 도입해 LLM의 시간 소모적 오토레그레시브 디코딩 및 빔 서치 프로세스를 제거하여 추론 시간을 크게 줄였습니다.

- **Performance Highlights**: Comprehensive experiments on four widely used real-world datasets validate the effectiveness of TokenRec, demonstrating that it outperforms competitive benchmarks, including both traditional recommender systems and emerging LLM-based recommender systems. 중요한 실험 결과, TokenRec은 기존의 전통적인 추천 시스템과 새로운 LLM 기반 추천 시스템보다 더 나은 성과를 보여주었습니다.



### Efficient Prompting for LLM-based Generative Internet of Things (https://arxiv.org/abs/2406.10382)
Comments:
          13 pages, 11 figures

- **What's New**: 대형 언어 모델 (LLMs)을 사물인터넷 (IoT) 응용 프로그램에 통합하는 방식이 주요 연구 관심사가 되었습니다. 본 연구에서는 로컬 네트워크 설정에서 오픈 소스 LLM을 사용하는 텍스트 기반 생성형 IoT (Generative IoT, GIoT) 시스템을 제안합니다. LLM의 성능 제한을 완화하고 경쟁력 있는 성능을 제공하기 위해 프롬프트 엔지니어링 기법을 적용하고, 맞춤형 프롬프트를 관리하기 위한 프롬프트 관리 모듈과 후처리 모듈을 설계하여 LLM이 생성한 결과를 처리합니다.

- **Technical Details**: 제안된 시스템은 엣지 서버에 오픈 소스 LLM을 배치하여 데이터 프라이버시와 보안을 보장하며 시스템 확장성과 성능을 개선할 수 있습니다. 프롬프트 관리 모듈과 후처리 모듈은 다양한 IoT 장치 요청에 대해 프롬프트를 관리하고 생성된 결과를 처리하는 역할을 합니다. Table-QA (Table Question Answering) 문제를 케이스 스터디로 사용하여 시스템의 효과를 입증합니다. 이 문제는 복잡한 구조와 이질적인 데이터 유형으로 인해 도전적인 과제입니다. 세 단계의 프롬프트 방식 (태스크 계획, 태스크 수행, 태스크 교정)을 제안하여 오픈 소스 LLM의 성능을 향상시키고 새로운 작업으로 쉽게 확장할 수 있음을 보입니다.

- **Performance Highlights**: WikiTableQA와 TabFact 데이터셋에서 광범위한 실험을 통해 제안된 GIoT 시스템과 프롬프트 방식이 경쟁 메서드 대비 뛰어난 성능을 발휘함을 확인했습니다. 실험 결과 우리 방식이 베이스라인 방법을 큰 차이로 능가하며, 최신의 성능을 달성했습니다. 또한, 다른 프롬프트 기법과 오픈 소스 LLM의 성능, 추론 비용을 고려한 포괄적인 분석을 수행하여 LLM 기반 GIoT 시스템에 적합한 LLM과 프롬프트 방법 선택에 대한 가이드라인을 제공합니다.



### From Pixels to Prose: A Large Dataset of Dense Image Captions (https://arxiv.org/abs/2406.10328)
Comments:
          pixelprose 16M dataset

- **What's New**: PixelProse라는 새로운 비전-언어 모델 학습용 데이터셋이 도입되었습니다. 이 데이터셋은 웹에서 자동 생성된 16M(밀리언) 이상의 이미지-텍스트 쌍으로 구성되어 있으며, 기존 웹 스크래핑 데이터셋의 노이즈 문제와 세부 묘사 부족 문제를 개선하기 위해 설계되었습니다.

- **Technical Details**: PixelProse 데이터셋은 Google Gemini 1.0 Pro Vision Model을 사용하여 생성된 정교한 캡션들로 구성되어 있습니다. 데이터셋은 CommonPool, CC12M, RedCaps 등 세 가지 웹 스크랩 데이터베이스에서 다양한 이미지를 포함하여 수집되었습니다. 각 데이터셋에는 텍스트 인식 정확성을 보장하기 위해 여러 필터링 단계와 메타데이터가 포함되어 있습니다.

- **Performance Highlights**: PixelProse는 일반적인 이미지 설명을 제공하며, Vision-Language Model (VLM) 및 diffusion model 훈련에 필요한 다양한 이미지 속성을 포함합니다. 또한, CSAM, PII, 유해성 등의 문제를 철저히 검사하여 데이터 무결성을 보존하고, 교차 해킹을 방지하기 위한 여러 상업용 API를 사용한 추가 검증 과정도 포함하고 있습니다.



### Automatically Labeling $200B Life-Saving Datasets: A Large Clinical Trial Outcome Benchmark (https://arxiv.org/abs/2406.10292)
- **What's New**: 이번 논문에서 제안된 Clinical Trial Outcome (CTO) 데이터셋은 약 479K 개의 임상 시험 데이터를 포함하여 현재까지 공개된 가장 큰 규모의 임상 시험 결과 데이터셋입니다. 여러 약한 지도 라벨(weakly supervised labels)을 통합하여 개별 소스에서 발생하는 노이즈를 최소화하고 사람의 주석이 필요 없도록 설계되었습니다.

- **Technical Details**: CTO 데이터셋은 시험 관련 문서에 대한 대형 언어 모델(LLM) 추론, 뉴스 헤드라인의 감성 분석, 시험 지원자의 주가, 시험 단계 간 연결 및 환자 탈락률과 부작용과 같은 다양한 신호로부터 라벨을 생성하였습니다. 이를 통해 다수의 약한 지도 학습 소스로부터 데이터를 병합하고 불필요한 노이즈를 최소화했습니다.

- **Performance Highlights**: CTO의 라벨은 테스트 분할에서 인간이 주석을 단 임상 시험 결과 라벨과 91 F1 스코어의 전례없는 일치를 보여줍니다. 또한, Phase 3 임상 시험에서는 94 F1 스코어를 기록하여 매우 높은 정밀도의 예측 결과를 제공하였습니다.



### ResearchArena: Benchmarking LLMs' Ability to Collect and Organize Information as Research Agents (https://arxiv.org/abs/2406.10291)
- **What's New**: 이번 연구에서는 대형 언어 모델들(LLMs)이 학술 조사를 수행하는 능력을 측정하기 위한 벤치마크 'ResearchArena'가 개발되었습니다. 이 벤치마크는 학술 조사 과정을 세 단계로 분해하여 평가합니다: 정보 발견(information discovery), 정보 선택(information selection), 및 정보 조직(information organization). 또한, 이 벤치마크는 1,200만 개의 전체 텍스트 학술 논문과 7,900개의 조사 논문으로 구성된 오프라인 환경을 포함합니다.

- **Technical Details**: ResearchArena는 학술 조사 과정의 세 가지 주요 작업을 다룹니다. 첫 번째 작업인 정보 발견(information discovery) 단계에서는 LLM들이 관련 논문을 식별하고 검색하는 능력을 평가합니다. 두 번째 작업, 정보 선택(information selection) 단계는 검색된 논문의 중요성과 관련성을 평가하는 능력을 요구합니다. 세 번째 작업, 정보 조직(information organization) 단계에서는 선택된 논문을 위계적인 지식 마인드맵으로 구성하는 능력을 평가합니다. 이러한 모든 작업은 대규모 학술 데이터셋을 활용하여 엄격히 평가됩니다.

- **Performance Highlights**: 기존의 LLM 기반 방법들은 기본적인 키워드 검색 기법과 비교했을 때 성능이 저하되는 것으로 나타났습니다. 특히, 정보 발견과 정보 선택 단계에서는 LLM들이 전통적인 검색 기법보다 낮은 재현율(recall)과 정밀도(precision)를 보였습니다. 또한, 정보 조직 단계에서도 일관적이고 정확한 지식 구조를 구성하는 데 어려움을 겪는 것으로 나타났습니다. 이러한 결과들은 LLM의 향후 연구를 위한 상당한 개선 기회를 시사합니다.



### Watermarking Language Models with Error Correcting Codes (https://arxiv.org/abs/2406.10281)
- **What's New**: 최근 대형 언어 모델의 발전으로 인해 현실적인 기계 생성 콘텐츠를 생성할 수 있게 되었습니다. 이 논문에서는 사람과 기계가 생성한 텍스트를 구분하기 위해 암호화 기법을 통한 워터마킹(watermarking) 프레임워크를 제안합니다. 이 방법은 오류 수정 코드(error correcting code)를 사용하여 통계적 신호를 인코딩하며, 품질 저하 없이 원래 확률 분포와 비교해도 왜곡이 없습니다.

- **Technical Details**: 본 논문에서는 오류 수정 코드 기반의 Robust Binary Code(RBC) 워터마크를 제안합니다. 출판된 모델 및 명령 맞춤 모델에서 평가한 바에 따르면, 이 워터마크는 편집, 삭제 및 번역에 견딜 수 있도록 설계되었습니다. 또한, 정보 이론적 관점에서 워터마킹을 제공하고, p-value를 생성하기 위한 강력한 통계적 테스트와 이론적 보장을 제공합니다.

- **Performance Highlights**: 실험 결과, RBC 워터마크가 속도와 강력함, 견고성에서 매우 우수한 성능을 보이며, 현재까지 나온 최첨단 기술과 비교해도 경쟁력이 있습니다. 이 워터마크는 왜곡 없이 적용되며, 원래 언어 모델의 품질을 그대로 유지합니다. 또한, 언어 모델 출력의 logits에 간단히 적용할 수 있다는 점에서 사용이 용이한 것이 특징입니다.



### Transferable Embedding Inversion Attack: Uncovering Privacy Risks in Text Embeddings without Model Queries (https://arxiv.org/abs/2406.10280)
Comments:
          Accepted at ACL 2024 Main Conference

- **What's New**: 이 논문은 원본 임베딩 모델에 접근할 수 없는 상황에서의 텍스트 임베딩으로 인한 프라이버시 위험을 조사합니다. 연구진은 서브스티튜트 모델 (surrogate model)을 이용하여 텍스트 임베딩으로부터 민감한 정보를 추론할 수 있는 전송 공격 (transfer attack) 방법을 개발하였습니다. 이 방법은 기존의 모델 접근을 직접 요구하는 방법들보다 더 현실적인 위협 모델입니다.

- **Technical Details**: 연구진은 서브스티튜트 모델을 통해 희생자 모델의 동작을 모방하는 전송 공격을 제안합니다. 이 전송 공격은 두 가지 목표를 가집니다: 첫째, 텍스트 인코더를 훔치려는 시도로, 희생자 모델의 표현을 학습하는 서브스티튜트 모델을 개발합니다. 둘째, 위협 모델 전송 가능성을 통해 서브스티튜트 모델을 공격하여 동일한 방법이 희생자 모델에서도 효과적이도록 합니다. 이를 위해 연구진은 오프더셸프 텍스트 임베딩 모델과 MLP 기반 어댑터를 사용했습니다.

- **Performance Highlights**: 연구 결과, 전송 공격이 기존의 표준 공격 방법보다 40%-50% 더 효과적임을 입증했습니다. 특히, Sentence-BERT, Sentence-T5, OpenAI 텍스트 임베딩 모델을 대상으로 한 실험에서 뛰어난 성능을 보였습니다. 특정 위협 도메인에 대한 프라이버시 위험을 연구하기 위해 MIMIC-III 임상 노트 데이터셋을 사용한 사례 연구에서 민감한 속성(예: 나이, 성별, 질병)을 80%-99%의 정확도로 식별할 수 있음을 확인했습니다.



### Using General Large Language Models to Classify Mathematical Documents (https://arxiv.org/abs/2406.10274)
- **What's New**: 최근 공개된 대형 언어 모델(LLMs)을 사용하여 수학 문서를 분류하는 가능성을 평가하기 위한 초기 탐색 결과에 대해 보고합니다. 자동 분류는 문학 내비게이션을 개선하고 수학적 결과 간의 관계를 식별하는 데 유용할 수 있습니다. 이 실험에서는 arXiv.org의 초록문서를 MSC 2020에 따라 분류했으며, 실험 대상은 제목과 초록만을 사용했습니다.

- **Technical Details**: 수학적 문서 분류 실험은 LLM의 도움으로 이루어졌습니다. 여기에서 사용된 모델은 ChatGPT 3.5와 4, Microsoft Copilot, Google Gemini 등 다양한 최신 AI 서비스입니다. 실험 대상은 최근 arXiv에 제출된 문서들로, 중복된 항목을 제외한 56개의 다채로운 수학 분야 문서에 대한 분류가 수행되었습니다. MSC 2020 수학주제분류 체계를 사용하여 문서의 제목과 초록만으로 분류를 시도했습니다.

- **Performance Highlights**: 샘플의 약 60%에서 LLM은 arXiv에 이미 보고된 기본 분류와 일치하는 분류를 생성했으며, 이 중 절반은 추가적인 기본 분류도 감지되었습니다. 40%에서는 LLM이 다른 분류를 제안했으나, 이러한 경우 대다수는 제공된 분류보다 더 나은 것으로 평가되었습니다. 이를 통해 LLM의 분류가 상당히 효과적이고 신뢰할 수 있음을 보여줍니다.



### Autograding Mathematical Induction Proofs with Natural Language Processing (https://arxiv.org/abs/2406.10268)
- **What's New**: 최근 논문에서는 자연어 처리(NLP) 모델을 활용하여 수학적 증명을 자동으로 평가할 수 있는 새로운 훈련 방법과 모델을 제안하고 있습니다. 이 모델은 기존의 대형 언어 모델과 기타 기계 학습 기술을 결합하여 수학적 증명을 자동으로 채점하며, 이는 학생들이 증명 연습을 더 효과적으로 할 수 있도록 즉각적인 피드백을 제공할 수 있습니다.

- **Technical Details**: 이 논문에서는 수학적 증명 데이터를 사용하여 4가지 다른 대형 언어 모델의 성능을 비교했습니다. 데이터는 증명 귀납법(proof by induction) 문제에서 수집되었으며, 각 모델이 다양한 정도의 만족스러운 성능을 달성했습니다. 또한, 인간 채점자와 비교한 결과, 최상의 채점 모델이 대부분의 인간 채점자보다 더 정확함을 발견했습니다. 이 모델은 수학적 자동 단답형 평가(ASAG)의 관련 문제를 해결하는 데 사용되었습니다.

- **Performance Highlights**: 사용자 연구 결과, 학생들은 자동 채점기에서 제공하는 피드백을 통해 증명을 현저하게 개선할 수 있었지만, 여전히 인간 채점기보다 AI 채점기를 덜 신뢰하는 경향이 있었습니다. 이러한 연구 결과는 향후 자동 채점기의 피드백을 개선하고, 학생들이 AI 자동 채점기를 더 신뢰할 수 있는 방법을 모색하는 방향으로 나아갈 필요가 있음을 시사합니다.



### Large Language Model-empowered multimodal strain sensory system for shape recognition, monitoring, and human interaction of tensegrity (https://arxiv.org/abs/2406.10264)
- **What's New**: 이 논문에서는 6-스트럿(6-strut) 텐스그리티(tensegrity) 시스템에 24개의 다중 모달 변형 센서(multimodal strain sensors)를 통합하여 스마트 텐스그리티(smart tensegrity)를 구현한 사례를 소개합니다. 이 시스템은 깊은 학습 모델(deep learning model)과 대형 언어 모델(large language models)을 활용하여 자가 형태 재구성(self-shape reconstruction)을 실현합니다. 또한, Flask 서버와 GPT-3.5-Turbo 모델을 통합하여 데이터 무선 모니터링과 분석, 설명, 예측, 제안 기능을 제공하며, 이는 iPhone으로 모니터링 데이터를 전송할 수 있습니다.

- **Technical Details**: 이 스마트 텐스그리티 시스템은 전도성 유연 텐던(conductive flexible tendons)과 장단기 메모리 모델(LSTM; long short-term memory model)을 사용하여 외부 센서 없이 자가 형태 재구성을 수행합니다. 또한, Flask 서버와 GPT-3.5-Turbo 모델을 통합하여 스마트 텐스그리티가 iPhone으로 데이터를 전송하고, 이를 분석하여 해석, 예측, 제안을 제공할 수 있습니다. 인간과의 상호작용 시스템도 포함되어 있어 인간이 텐스그리티에 관련된 필요한 정보를 이해하는 데 도움이 됩니다.

- **Performance Highlights**: 이 시스템은 자가 형태 재구성 및 무선 모니터링, 심층 데이터 분석 및 인간 상호작용 기능을 자율적으로 수행할 수 있는 특징이 있습니다. 이를 통해 미래의 탐사 임무, 특히 우주 탐사와 같은 예측할 수 없는 환경에서의 응용 가능성을 보여줍니다.



### AutoSurvey: Large Language Models Can Automatically Write Surveys (https://arxiv.org/abs/2406.10252)
- **What's New**: AutoSurvey는 인공지능(AI) 분야와 같이 빠르게 발전하는 영역에서 종합적인 문헌 조사(literature surveys)를 자동화하기 위한 신속하고 체계적인 방법을 소개합니다. 이 시스템은 정보의 방대함과 복잡성으로 인한 전통적인 조사 문서 작성의 어려움을 극복하고자 합니다. AutoSurvey는 최초 검색 및 개요 작성, 특화된 LLM에 의한 부분 초안 작성, 통합 및 정제, 엄격한 평가와 반복을 통해 이러한 문제를 체계적으로 해결합니다.

- **Technical Details**: AutoSurvey는 두 단계의 병렬 생성 방법을 통해 조사 내용을 효율적으로 생성합니다. 초기 단계에서는 여러 LLMs가 동시에 자세한 개요를 작성하고, 이러한 개요를 종합하여 최종 개요를 만듭니다. 각 섹션은 개요에 따라 병렬로 생성되며, 최종적으로 통합 및 정제 과정을 거쳐 일관된 문서를 완성합니다. 실시간 지식 업데이트는 RAG(Retrieval-Augmented Generation) 접근 방식을 사용하여 최신 연구 논문을 동적으로 반영합니다. 또한 Multi-LLM-as-Judge 전략을 통해 여러 LLMs를 사용하여 생성된 내용을 평가합니다. 이 방법은 학술 기준에 따라 초기 평가 지표를 생성하고, 인간 전문가의 피드백으로 정제하여 신뢰성을 높입니다.

- **Performance Highlights**: AutoSurvey는 최대 64k 토큰 길이의 실험에서 높은 인용 및 내용 품질 점수를 달성했습니다. 64k 토큰에서 인용 품질은  recall 82.25%와 precision 77.41%를 기록했으며, 이는 인간 성과에 근접한 수준입니다. 내용 품질에서는 coverage(4.73), structure(4.33), relevance(4.86)의 점수를 기록하여 인간 성과와 유사한 결과를 보였습니다. 또한, 다양한 길이의 조사에서도 일관된 성과를 유지했으며, 다중 LLM 방식의 평가 결과는 인간 전문가와의 상관관계가 높게 나타났습니다(Spearman's rho = 0.5429).



### A Reality check of the benefits of LLM in business (https://arxiv.org/abs/2406.10249)
- **What's New**: 최근 대형 언어 모델(LLMs)은 전략적 계획, 프로젝트 구현 및 데이터 기반 의사 결정과 같은 다양한 비즈니스 기능에 적응할 수 있게 되었습니다. 이러한 LLMs는 기존 모델과 달리 재훈련 없이 프롬프트 엔지니어링(prompt engineering)을 통해 새로운 도메인에 적응할 수 있습니다. 그러나 편향성, 맥락 이해 및 프롬프트 민감도에 대한 LLMs의 한계는 실제 응용에 대한 준비상태에 대한 우려를 불러일으킵니다. 이 논문은 LLMs의 비즈니스 프로세스에서의 유용성과 준비 상태를 철저히 검토합니다.

- **Technical Details**: LLMs는 수조 개의 단어로 이루어진 인터넷 텍스트를 학습하여 넓은 언어 패턴을 포착할 수 있습니다. 특히, 프롬프트 엔지니어링은 특정 응용프로그램에 맞게 LLMs를 커스터마이징할 수 있게 해주며, 이는 모델의 재훈련이나 추가 데이터 없이도 가능합니다. 이 논문에서는 ChatGPT와 같은 대표적인 LLMs를 다루며, 비즈니스 가치와 한계를 실험 및 평가합니다. 또한, LLMs의 성능을 확인하기 위해 네 가지 접근 가능한 LLM을 사용하여 실제 데이터를 통한 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 접근 가능한 네 가지 LLM 모두 의미 있는 답변을 생성할 수 있었으며, 특정 LLM은 더 세부적인 응답을 제공하는 반면, 다른 LLM은 키워드만 제공하는 것을 확인했습니다. LLMs는 다양한 비즈니스 활동에서 잠재적인 유용성을 나타냈지만 편향성, 맥락 이해 및 프롬프트 민감도와 같은 문제는 여전히 존재합니다. 이러한 실험 결과는 조직이 생성형 AI를 활용할 때 참고할 수 있는 중요한 통찰력을 제공하며, 향후 연구 방향에 대한 귀중한 정보를 제공합니다.



### Towards commands recommender system in BIM authoring tool using transformers (https://arxiv.org/abs/2406.10237)
- **What's New**: 본 연구에서는 건축, 엔지니어링 및 건설(AEC) 부문에서의 BIM(빌딩 정보 모델링) 소프트웨어 사용을 촉진하기 위한 새로운 접근 방법을 제시한다. 사용자 상호작용 데이터를 바탕으로, BIM 소프트웨어 명령(command)을 추천할 수 있는 순차 추천 시스템(sequential recommendation system)의 가능성을 탐구하였다. 이 시스템은 사용자의 이전 상호작용을 바탕으로 다음 명령을 예측하여 실시간 명령 제안을 제공한다.

- **Technical Details**: 연구의 근간이 되는 프레임워크는 대규모 BIM 로그 데이터를 전처리하고, 최신 대형 언어 모델의 트랜스포머 아키텍처(transformer architectures)를 백본(backbone)으로 활용하였다. 이를 통해 BIM 작성 도구인 Vectorworks 내에서 실시간 명령 제안을 가능하게 하는 프로토타입을 구축했다. 특히, BIM 소프트웨어 명령을 추천 가능한 항목으로 취급하여 사용자의 편의성을 극대화하는데 초점을 두었다.

- **Performance Highlights**: 후속 실험을 통해 제안된 모델이 이전 연구를 능가하는 성능을 보였음을 확인하였다. 이로써 순차 추천 시스템이 설계 효율성을 획기적으로 향상시킬 수 있는 잠재력을 갖고 있음을 증명했다.



### Khmer Semantic Search Engine (KSE): Digital Information Access and Document Retrieva (https://arxiv.org/abs/2406.09320)
- **What's New**: 이 연구는 캄보디아에서 처음으로 제안된 Khmer Semantic Search Engine(KSE)을 소개합니다. KSE는 기존의 Khmer 검색 방법을 향상시켜, 고급 시맨틱 매칭 기술을 활용하여 정확한 검색 결과를 제공합니다. 특히, 키워드 사전 기반, 온톨로지 기반 및 랭킹 기반의 세 가지 시맨틱 검색 프레임워크를 제안하여 사용자의 검색 정확성을 극대화합니다.

- **Technical Details**: KSE는 사용자 쿼리에서 의미 있는 키워드를 추출하고, 시맨틱 콘텐츠를 형식적으로 주석 달아 정확한 매칭을 수행합니다. 이 시스템은 자동 및 수동 키워드 추출 도구와 온톨로지 기반 시맨틱 강화 도구를 개발하여 높은 품질의 입력 데이터를 보장합니다. 또한, 기존의 TF-IDF, Word2Vec, BERT 등 다양한 최신 기술들을 통합하여 고성능 검색 결과를 달성합니다.

- **Performance Highlights**: KSE의 성능은 그라운드 트루스 (ground truth) 데이터셋을 사용하여 평가되었으며, 시맨틱 검색어 이해 능력을 통해 검색 정확도를 크게 향상시켰습니다. 이는 사용자가 보다 관련성 높은 문서와 URL을 찾는 데 큰 도움을 줄 수 있음을 보여줍니다.



New uploads on arXiv(cs.IR)

### DiffMM: Multi-Modal Diffusion Model for Recommendation (https://arxiv.org/abs/2406.11781)
- **What's New**: 최근 연구는 데이터 희소성을 극복하기 위해 셀프-슈퍼바이즈드 러닝(self-supervised learning, SSL) 기법을 활용한 개인화 추천 시스템을 제안하였습니다. 그러나 대부분의 기존 연구는 단순한 임의적 증강 방식이나 직관적인 크로스-뷰(cross-view) 정보를 사용하여 멀티 모달 변수와 사용자-아이템 상호작용 모델링 간의 정교한 정렬에 실패했습니다. 이를 해결하기 위해, 새로운 멀티-모달 그래프 확산 모델인 DiffMM을 제안합니다. 이 프레임워크는 모달리티-어웨어 그래프 확산 모델(modality-aware graph diffusion model)과 크로스-모달 대조 학습(cross-modal contrastive learning)을 결합하여 모달리티-어웨어 사용자 표현 학습을 개선합니다.

- **Technical Details**: DiffMM은 그래프 신경망 기반의 협업 필터링(collaborative filtering) 기법을 발전시켜 그래프-구조화 데이터를 활용하여 텍스트, 시각, 음향 특징을 포함한 다양한 모달리티 특성 벡터를 사용하여 사용자-아이템 상호작용 그래프를 풍부하게 만듭니다. 이 모델은 단계별 노이즈 주입 과정과 역변환 과정을 적용하여 모달리티-어웨어 신호 주입 메커니즘을 통해 원래의 사용자-아이템 그래프 구조를 복구합니다.

- **Performance Highlights**: 세 개의 공개 데이터셋에서 실시한 광범위한 실험 결과, DiffMM은 다양한 경쟁 베이스라인 대비 탁월한 성능을 지속적으로 입증하였습니다. 특히, 데이터 희소성 문제와 무작위 증강 방식에서 발생하는 문제를 효과적으로 해결하였습니다. 이 프레임워크는 모달리티 관련 일관성에 의해 안내되는 크로스-모달 대조 학습의 셀프-슈퍼바이즈드 신호로 사용자/아이템 표현을 증강시켰습니다.



### Multi-Layer Ranking with Large Language Models for News Source Recommendation (https://arxiv.org/abs/2406.11745)
Comments:
          Accepted by the SIGIR 2024. arXiv admin note: text overlap with arXiv:2305.04825

- **What's New**: 뉴스 이벤트에 대한 신뢰할 수 있는 정보원을 찾기 위해, 이전에 인용된 발언을 기반으로 신뢰할 수 있는 소스를 식별하는 새로운 전문가 추천 시스템을 제안합니다. 이를 위해 총 23,571개의 인용-화자 쌍으로 구성된 NewsQuote라는 새로운 데이터셋을 만들었습니다. 특히, 대규모 언어 모델(Large Language Models, LLMs)를 활용한 다층 랭킹(Multi-layer Ranking) 프레임워크를 통해 추천 성능을 크게 향상시켰습니다.

- **Technical Details**: 데이터셋 생성은 2019년 11월부터 2020년 8월 사이에 출판된 기사에서 수집된 인용-화자 쌍으로 이루어졌습니다. 뉴스 기사를 문장 단위로 분할하고, 사전 학습된 BERT 기반의 의미 역할 라벨링 모델을 사용하여 인용 트리거 단어를 추출했습니다. 이를 통해 352개의 인용 트리거 단어를 식별하고, 주어와 목적어가 포함된 문장만을 선택했습니다. 수집된 데이터셋은 In-context Learning 기반의 LLM 랭커를 사용하여 다층 랭킹 기반 필터링 메커니즘을 통해 추천 성능을 높였습니다.

- **Performance Highlights**: 실험 결과, 다층 LLM 랭킹을 사용하면 추천 시스템의 예측 품질 및 행동 품질이 크게 향상된 것으로 나타났습니다. 이를 통해 기존의 인기 편향을 효과적으로 완화할 수 있었습니다. 데이터셋은 총 23,571개의 화자-인용 페어와 2,843명의 화자로 구성되어 있으며, 다양한 글로벌 도메인에서 수집된 데이터입니다.



### Graph Neural Re-Ranking via Corpus Graph (https://arxiv.org/abs/2406.11720)
Comments:
          This preprint is the result of work in progress, therefore it should still be considered a draft

- **What's New**: 새로운 그래프 신경망 기반 재배열 방식인 GNRR(Graph Neural Re-Ranking)이 제안되었습니다. 이를 통해 쿼리를 처리하는 동안 문서의 분포를 고려하여 재배열의 품질을 향상시킬 수 있습니다. 주요한 기여는 문서 간의 관계를 모델링한 Corpus Graph를 활용하여 문서 간의 의미적 유사성을 반영하는 점입니다.

- **Technical Details**: GNRR은 각 쿼리가 문서의 분배를 인식할 수 있도록 해주는 Graph Neural Networks(GNNs)에 기반한 파이프라인입니다. 쿼리에 의해 유도된 코퍼스 서브그래프로 문서 간의 관계를 모델링하며, 이는 메시지 전달 규칙(message-passing rule)과 결합됩니다. 실험 결과, GNN이 문서 간 상호작용을 효과적으로 포착하여 다양한 랭킹 지표에서 성능을 개선하는 것으로 나타났습니다.

- **Performance Highlights**: GNRR은 TREC-DL19 데이터셋에서 평균 정밀도(Average Precision)가 기준 모델 대비 5.8% 상대적으로 향상되었습니다. 다양한 랭킹 지표에서 GNN을 활용한 재배열 방식이 기존의 단변량(Univariate) 기법보다 우수한 성능을 보여주었습니다.



### Prompts as Auto-Optimized Training Hyperparameters: Training Best-in-Class IR Models from Scratch with 10 Gold Labels (https://arxiv.org/abs/2406.11706)
- **What's New**: 이번 연구는 1억 개 미만의 파라미터를 가진 소규모 신경 정보 검색 모델을 고작 10개의 고유 관련 라벨로 학습하는 새로운 방법을 개발했습니다. 이 방법은 문서에 대한 질문을 생성하기 위해 언어 모델(LM)을 사용하며, 중요한 단계는 학습 품질에 기반하여 LM 프롬프트를 자동으로 최적화하는 것입니다.

- **Technical Details**: 연구는 BIRCO 벤치마크 기준을 사용하여 실험을 수행하였으며, 우리의 방법으로 학습된 모델이 RankZephyr보다 우수하고, 7B 파라미터 모델인 RankLLama와 경쟁력이 있음을 발견했습니다. 이 모델들은 10만 건 이상의 라벨로 학습되었습니다. 주요 기술적 접근은 PATH(Prompts as Auto-optimized Training Hyperparameters) 방법으로, 프롬프트를 자동으로 최적화하여 합성 데이터셋을 생성하고, 이를 DSPy 프로그래밍 모델로 표현합니다.

- **Performance Highlights**: 평가에서는 10개의 긍정 라벨로 PATH를 적용했을 때 매우 경쟁력 있는 성능을 보였습니다. 특히, 태스크와 LM 전반에 걸쳐 평균 NDCG@10 점수에서 BM25보다 6.0 포인트, 10개의 긍정 라벨로 미세 조정된 LM보다 6.5 포인트, GPT-3.5를 사용한 합성 질문 생성보다 4.5 포인트 더 높은 성과를 보였습니다. 또한, RankLLaMA와 같은 대규모 크로스인코더와 비슷한 수준의 성능을 유지했습니다.



### TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy (https://arxiv.org/abs/2406.11678)
- **What's New**: 본 연구에서는 TourRank라는 새로운 문서 랭킹 방법을 제안합니다. 이는 스포츠 토너먼트 구조에서 영감을 받아 다양한 문서를 효과적으로 랭킹할 수 있도록 설계되었습니다. TourRank는 LLM(Large Language Models)의 입력 길이 제한, 입력 순서에 따른 랭킹 결과의 불균형, 그리고 성능과 비용 간의 균형과 같은 기존 LLM 기반 랭킹 방법의 문제점을 해결하기 위해 도입되었습니다.

- **Technical Details**: TourRank는 다단계 토너먼트 방식을 채택하여 각 문서를 참가자로 간주하고, 여러 단계에서 쿼리에 가장 관련성이 높은 문서를 선택합니다. 각 단계에서 그룹을 형성하고 LLM을 사용하여 그룹 내에서 가장 관련성이 높은 문서를 다음 단계로 진행시킵니다. 이 과정은 병렬화하여 처리 시간을 단축시키고, 첫 번째 검색 모델이 제공하는 초기 문서 순서도 활용하지만 이에 크게 의존하지 않도록 설계되었습니다. 또한, 문서를 여러 라운드의 토너먼트를 통해 점수를 부여하여 최종 점수 기반으로 문서 순위를 결정합니다.

- **Performance Highlights**: TourRank는 TREC DL 19, TREC DL 20, 그리고 BEIR 벤치마크 데이터셋에서 실험을 수행하였으며, 기존의 제로샷 문서 랭킹 방법을 능가하는 성능을 입증했습니다. 특히, Mistral-7B 및 Llama-3-8B와 같은 오픈 소스 모델에서도 최첨단(SOTA) 성능을 달성했습니다. TourRank는 랭킹 성능과 비용 간의 균형을 잘 맞추었으며, 초기 후보 문서 순서에 대한 민감성을 효과적으로 완화하였습니다.



### Making Alice Appear Like Bob: A Probabilistic Preference Obfuscation Method For Implicit Feedback Recommendation Models (https://arxiv.org/abs/2406.11505)
- **What's New**: 이번 연구는 추천 시스템(Recommender Systems, RS)에서 사용자 개인정보 유출 문제를 해결하기 위해 신규 확률적 오프스케이션 방법(SBO, Stereotypicality-Based Obfuscation)을 도입하였습니다. SBO는 기존의 개인정보 보호 기술보다 높은 추천 정확도와 개인정보 보호 간의 균형을 이룹니다.

- **Technical Details**: SBO는 사용자 선호 데이터의 전형성을 줄임으로써 적용되며, 항목의 전형성 점수(Item Stereotypicality Score)와 그룹 성향(IGI, Item Group Inclination)을 기반으로 사용자 전형성을 계산합니다. SBO는 세 가지 대표적인 추천 모델(BPR, MultVAE, LightGCN)과 두 가지 인기 있는 데이터셋(MovieLens-1M, LFM-2B)에서 테스트되었습니다.

- **Performance Highlights**: 실험 결과, SBO는 사용자 개인정보 유출을 줄이면서도 추천 정확도를 유지하거나 오히려 향상시키는 것으로 나타났습니다. 이 방법은 기존의 항목 추가/삭제 방법들보다 효율적이며, 특히 성별 추론 공격에 대해 강력한 방어 성능을 보였습니다.



### Evaluating the Efficacy of Open-Source LLMs in Enterprise-Specific RAG Systems: A Comparative Study of Performance and Scalability (https://arxiv.org/abs/2406.11424)
- **What's New**: 이번 논문은 오픈 소스 대형 언어 모델(LLMs)을 통한 기업 맞춤형 데이터 세트에서의 Retrieval-Augmented Generation(RAG) 작업에 대한 분석을 제공합니다. 웹사이트에서 스크랩된 데이터를 사용하여 다양한 오픈 소스 LLMs의 성능, 접근성, 통합 가능성을 평가합니다. 연구 결과에 따르면 효과적인 임베딩 기술과 결합된 오픈 소스 LLMs는 RAG 시스템의 정확성과 효율성을 크게 향상시킬 수 있다는 것을 보여줍니다.

- **Technical Details**: 논문에서는 데이터 수집, 모델 선택, 시스템 아키텍처, 평가 지표, 실험 절차 등 여러 단계로 구성된 방법론을 설명합니다. 데이터 수집 단계에서는 https://i-venture.org/ 사이트에서 URL을 가져와 텍스트 콘텐츠를 추출했으며, langchain 라이브러리를 사용하여 데이터를 작은 조각으로 나누었습니다. 텍스트 조각에 대한 임베딩(Embeddings)은 Hugging Face 플랫폼에서 가져왔으며, 이를 통해 생성된 임베딩은 FAISS(Face AI Similarity Search) 벡터 데이터베이스에 저장됐습니다. LLMs는 RAG 시스템의 생성을 강화하는 데 사용되었습니다.

- **Performance Highlights**: 연구 결과는 오픈 소스 LLMs와 효과적인 임베딩 기술이 결합될 경우 RAG 시스템의 정확성과 효율성을 크게 향상시킬 수 있다는 것을 입증했습니다. 또한 TopK 매개 변수를 조정하여 다른 LLMs에 대해 상당한 성능 변화를 관찰할 수 있었으며, 이는 특정 기업 환경에서 최적의 성능을 위한 미세 조정의 중요성을 강조합니다.



### Transparency, Privacy, and Fairness in Recommender Systems (https://arxiv.org/abs/2406.11323)
Comments:
          Habilitation (post-doctoral thesis) at Graz University of Technology for the scientific subject Applied Computer Science

- **What's New**: 최신 연구에 따르면 추천 시스템(Recommender Systems)은 신뢰할 수 있는 인공지능(trustworthy AI)을 위해 투명성(transparency), 개인정보 보호(privacy), 공정성(fairness) 등의 개념을 중요한 요소로 받아들여야 함을 강조합니다. 이는 특히 유럽 AI법(European AI Act)과 같은 규정에서 더욱 두드러집니다.

- **Technical Details**: 이 연구는 세 가지 주요 측면에서 추천 시스템을 분석합니다: (i) 투명성과 인지 모델, (ii) 개인정보 보호와 제한된 선호 정보, (iii) 공정성과 인기 편향입니다. 각 측면에 대해 구체적으로, 심리학 이론을 포함한 투명한 디자인 프로세스를 강조해 심리학 정보를 바탕으로 한 추천 시스템(psychology-informed recommender systems)을 제안하고, 차등 비즈니스 데이터 보호(differential privacy)와 협업 필터링(collaborative filtering)에서 정확도와 개인정보 보호 간의 균형을 다룹니다. 더불어, 사용자의 선호 데이터가 제한된 상황에서의 추천 시스템을 설계하였습니다. 마지막으로, 추천 빈도와 아이템의 인기도 사이의 상관관계를 살펴보며 공정한 추천 알고리즘을 연구했습니다.

- **Performance Highlights**: 이 연구에서는 인간 기억 이론의 모델을 통해 태그와 음악 추천 알고리즘을 개발하여 추천 정확도를 향상시켰습니다. 또한, 추천 시스템의 구성 요소가 어떻게 추천 목록을 생성하는지 설명함으로써 투명성 측면에서도 기여할 수 있음을 보였습니다. 사용자 그룹별 인기 편향을 측정한 결과, 특히 여성 사용자들이 알고리즘의 인기 편향 증폭에 더 강하게 영향을 받는 것을 발견했습니다. 그리고 뉴스 기사 추천 분야에서 인기 편향 완화를 위한 연구 결과를 발표했습니다.



### Iterative Utility Judgment Framework via LLMs Inspired by Relevance in Philosophy (https://arxiv.org/abs/2406.11290)
Comments:
          22 pages

- **What's New**: 최근 정보 검색(IR) 분야에서는 주제 관련성(topical relevance)뿐만 아니라 유용성(utility) 평가도 중요하게 다뤄지고 있습니다. 이는 Retrieval-Augmented Generation(RAG)와 같은 다운스트림 작업을 촉진하기 위해 필요합니다. 이 논문에서는 주제 관련성, 해석적 관련성(interpretational relevance), 동기적 관련성(motivational relevance)을 포함한 세 가지 종류의 관련성을 동적으로 통합하는 Iterative utiliTy judgmEnt fraMework(ITEM)을 제안합니다. 이 프레임워크는 LLMs를 활용하여 각 단계의 성능을 크게 향상시킵니다.

- **Technical Details**: ITEM 프레임워크는 세 가지 주요 단계를 결합합니다: 주제 관련성 평가, 유용성 판단, 그리고 답변 생성입니다. Schutz의 철학적 관련성 시스템에서 영감을 받아 이 세 단계가 반복적으로 상호작용하며 성능을 증대시킵니다. 즉, 주제 관련성은 현재의 집중 대상을 형성하고, 해석적 관련성은 과거 경험을 통해 현재 대상을 이해하며, 동기적 관련성은 이를 바탕으로 추가 자료를 획득하여 새로운 경험으로 삼습니다.

- **Performance Highlights**: TREC DL, WebAP, NQ와 같은 다중 등급(grade) 패시지 검색 및 사실 기반 질문 답변(QA) 데이터셋에서 실험을 수행했습니다. 실험 결과, ITEM 프레임워크는 유용성 평가, 주제 관련성 순위, 답변 생성 측면에서 대표적인 기존 방법보다 우수한 성능을 보였습니다. 특히 단일 회차의 유용성 판단 접근법보다도 크게 향상된 결과를 보였습니다.



### Unifying Multimodal Retrieval via Document Screenshot Embedding (https://arxiv.org/abs/2406.11251)
- **What's New**: 다양한 형식과 모달리티(modalities)로 조직된 문서들을 단일 형식으로 처리하는 새로운 검색 패러다임, Document Screenshot Embedding(DSE)를 소개합니다. DSE는 문서 스크린샷을 사용하여 문서의 모든 정보를 보존하면서도 별도의 콘텐츠 추출 전처리 없이 바로 인덱싱할 수 있습니다. Wiki-SS라는 130만 개의 Wikipedia 웹 페이지 스크린샷 데이터셋을 구성하여 이 방법을 평가했습니다.

- **Technical Details**: DSE는 문서 스크린샷을 단일 입력 형식으로 처리하여, 대규모 비전-언어 모델(vision-language model)을 사용해 문서 스크린샷을 고밀도 표현(dense representation)으로 인코딩합니다. 검색 시, 사용자의 쿼리는 언어 모델을 통해 인코딩되어 가장 가까운 문서 임베딩을 찾습니다. 텍스트 중심 환경과 텍스트-이미지 혼합 환경에서 실험을 수행했으며 Wiki-SS와 SlideVQA 데이터셋을 변환하여 대규모 검색 환경을 구축했습니다.

- **Performance Highlights**: 텍스트 중심의 검색 환경에서는, DSE가 기존의 텍스트 기반 검색 BM25보다 top-1 검색 정확도에서 17포인트 우수했습니다. 텍스트-이미지 혼합 환경에서는, DSE가 OCR 기반 검색 방법들보다 nDCG@10에서 15포인트 이상 뛰어났습니다. DSE는 다양한 형식의 문서 검색에 효과적인 새로운 패러다임임을 보여줍니다.



### DELRec: Distilling Sequential Pattern to Enhance LLM-based Recommendation (https://arxiv.org/abs/2406.11156)
- **What's New**: DELRec라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 기존 Sequential Recommendation(SR) 모델에서 추출한 지식(knowledge)을 대형 언어 모델(LLMs)에 적용하여, 사용자의 연속적 추천(sequential recommendation)을 더 효과적으로 수행할 수 있게 합니다.

- **Technical Details**: DELRec는 두 가지 주요 단계로 구성됩니다: 1) SR 모델 패턴 증류(SR Models Pattern Distilling) - 이는 두 가지 잘 설계된 전략을 통해 SR 모델이 보여주는 행동 패턴을 소프트 프롬프트(soft prompts)를 사용하여 추출합니다; 2) LLM 기반 연속 추천(LLMs-based Sequential Recommendation) - 이는 LLMs을 미세 조정하여 증류된 보조 정보를 활용해 더 효과적으로 SR 작업을 수행하도록 합니다.

- **Performance Highlights**: 세 개의 실제 데이터 세트(real datasets)를 사용한 광범위한 실험 결과, DELRec 프레임워크의 유효성을 검증했습니다. 이는 LLMs의 SR 성능을 기존 모델과 비교하여 향상시킵니다.



### ADSNet: Cross-Domain LTV Prediction with an Adaptive Siamese Network in Advertising (https://arxiv.org/abs/2406.10517)
Comments:
          Accepted to KDD 2024

- **What's New**: 광고 플랫폼의 Lifetime Value(LTV) 예측을 개선하기 위해 외부 데이터를 활용하는 방법을 제안합니다. 이를 통해 구매 샘플 수를 늘리고 LTV 예측 모델의 성능을 향상시킬 수 있습니다. 이를 위해 데이터 분포 이동 문제를 해결하기 위해 Adaptive Difference Siamese Network(ADSNet)를 도입합니다. ADSNet는 크로스 도메인 전이 학습(cross-domain transfer learning)을 이용하여 부정적인 전이를 방지합니다. 특정 도메인에 유용한 정보만 학습하고, 노이즈 샘플을 거부하여 부정적인 전이를 피하는 데 도움을 주는 정보 이득 평가 전략을 도입합니다. 또한 도메인 적응 모듈을 설계하여 서로 다른 도메인 간의 분포 거리를 줄이고 표현 공간 분포의 일관성을 높입니다.

- **Technical Details**: ADSNet는 pseudo-siamese 구조를 기반으로 한 정보 이득 평가 전략을 사용하여 타겟 도메인에 유용한 정보를 효과적으로 학습합니다. 이 전략은 부정적인 정보 샘플을 거부하여 부정적인 전이를 방지합니다. 도메인 적응 모듈은 서로 다른 도메인을 연결하여 분포 거리를 줄이고 일관된 표현 공간을 유지합니다. 이 모델은 실제 광고 플랫폼에서 오프라인 실험과 온라인 A/B 테스트를 통해 검증되었습니다.

- **Performance Highlights**: 제안된 ADSNet 방법은 GINI 지수를 2% 향상시켰습니다. 설계된 정보 이득 평가 전략이 부정적인 이득 샘플을 거부하고 모델 성능을 향상시키는 데 중요함을 입증하였습니다. 또한 ADSNet는 롱테일(long-tail) 예측을 크게 개선합니다. 온라인 A/B 테스트에서 online LTV가 3.47%, GMV가 3.89% 증가하였습니다.



### TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation (https://arxiv.org/abs/2406.10450)
- **What's New**: 최근 대규모 언어 모델(LLM)을 활용한 차세대 추천 시스템(RecSys)을 발전시키려는 관심이 높아지고 있습니다. 이를 위해 사용자와 아이템을 효율적으로 토큰화(indexing) 하는 것이 필수적입니다. TokenRec라는 새로운 프레임워크를 제안하여 효과적인 ID 토큰화 전략과 효율적인 검색 패러다임을 제공합니다.

- **Technical Details**: TokenRec는 'Masked Vector-Quantized (MQ) Tokenizer'라는 전략을 도입해 협력 필터링을 통해 학습된 사용자의 아이템 표현을 이산 토큰으로 양자화합니다. 이를 통해 고차원의 협력 지식을 LLM에 원활히 통합하고, 새로운 사용자나 아이템에 대해서도 일반화가 가능합니다. 또한 생성적 검색 패러다임을 도입해 LLM의 시간 소모적 오토레그레시브 디코딩 및 빔 서치 프로세스를 제거하여 추론 시간을 크게 줄였습니다.

- **Performance Highlights**: Comprehensive experiments on four widely used real-world datasets validate the effectiveness of TokenRec, demonstrating that it outperforms competitive benchmarks, including both traditional recommender systems and emerging LLM-based recommender systems. 중요한 실험 결과, TokenRec은 기존의 전통적인 추천 시스템과 새로운 LLM 기반 추천 시스템보다 더 나은 성과를 보여주었습니다.



### Using General Large Language Models to Classify Mathematical Documents (https://arxiv.org/abs/2406.10274)
- **What's New**: 최근 공개된 대형 언어 모델(LLMs)을 사용하여 수학 문서를 분류하는 가능성을 평가하기 위한 초기 탐색 결과에 대해 보고합니다. 자동 분류는 문학 내비게이션을 개선하고 수학적 결과 간의 관계를 식별하는 데 유용할 수 있습니다. 이 실험에서는 arXiv.org의 초록문서를 MSC 2020에 따라 분류했으며, 실험 대상은 제목과 초록만을 사용했습니다.

- **Technical Details**: 수학적 문서 분류 실험은 LLM의 도움으로 이루어졌습니다. 여기에서 사용된 모델은 ChatGPT 3.5와 4, Microsoft Copilot, Google Gemini 등 다양한 최신 AI 서비스입니다. 실험 대상은 최근 arXiv에 제출된 문서들로, 중복된 항목을 제외한 56개의 다채로운 수학 분야 문서에 대한 분류가 수행되었습니다. MSC 2020 수학주제분류 체계를 사용하여 문서의 제목과 초록만으로 분류를 시도했습니다.

- **Performance Highlights**: 샘플의 약 60%에서 LLM은 arXiv에 이미 보고된 기본 분류와 일치하는 분류를 생성했으며, 이 중 절반은 추가적인 기본 분류도 감지되었습니다. 40%에서는 LLM이 다른 분류를 제안했으나, 이러한 경우 대다수는 제공된 분류보다 더 나은 것으로 평가되었습니다. 이를 통해 LLM의 분류가 상당히 효과적이고 신뢰할 수 있음을 보여줍니다.



### Fast solution to the fair ranking problem using the Sinkhorn algorithm (https://arxiv.org/abs/2406.10262)
- **What's New**: 본 논문은 이중 시장구조(예: 온라인 중고장터)에서 구매자에게 개인 맞춤형 상품 순위를 제공하는 추천 시스템이 거래 증진에서 중요한 역할을 한다는 점을 다룹니다. 기존의 Saito와 Joachims(2022)은 공평한 노출을 보장하기 위해 Nash 사회적 복지를 최대화하는 임팩트 기반 공정 랭킹(impact-based fair ranking) 방법을 개발했습니다. 그러나 이 방법은 대규모의 제약된 비선형 최적화 문제를 해결해야 하므로 실질적인 규모의 추천 시스템에 적용하기에는 매우 어렵습니다. 이에 저자들은 빠른 해법을 제안하였습니다.

- **Technical Details**: 저자들은 먼저 공정 랭킹 문제를 비제약 최적화 문제로 변환하고, 최적 운송 문제의 효율적인 해결책으로 잘 알려진 Sinkhorn 알고리즘을 반복 실행하는 Gradient ascent 방법을 설계하였습니다. 이를 통해 임팩트 기반 공정 랭킹 문제를 빠르게 해결할 수 있습니다. 특히, 확률 제약을 만족하는 이중 확률 매트릭스를 설계하여 개별 소비자에게 개인화된 아이템 랭킹을 제공하는 스토캐스틱(stochastic) 랭킹 정책을 사용합니다.

- **Performance Highlights**: 실험 결과, 저자들의 알고리즘은 상업용 최적화 소프트웨어를 사용한 기존의 방법보다 약 1000배 빠르게 고품질의 공정한 랭킹을 제공할 수 있음을 보였습니다. 이는 인공 및 공공 데이터셋을 사용하여 검증되었습니다.



### AutoSurvey: Large Language Models Can Automatically Write Surveys (https://arxiv.org/abs/2406.10252)
- **What's New**: AutoSurvey는 인공지능(AI) 분야와 같이 빠르게 발전하는 영역에서 종합적인 문헌 조사(literature surveys)를 자동화하기 위한 신속하고 체계적인 방법을 소개합니다. 이 시스템은 정보의 방대함과 복잡성으로 인한 전통적인 조사 문서 작성의 어려움을 극복하고자 합니다. AutoSurvey는 최초 검색 및 개요 작성, 특화된 LLM에 의한 부분 초안 작성, 통합 및 정제, 엄격한 평가와 반복을 통해 이러한 문제를 체계적으로 해결합니다.

- **Technical Details**: AutoSurvey는 두 단계의 병렬 생성 방법을 통해 조사 내용을 효율적으로 생성합니다. 초기 단계에서는 여러 LLMs가 동시에 자세한 개요를 작성하고, 이러한 개요를 종합하여 최종 개요를 만듭니다. 각 섹션은 개요에 따라 병렬로 생성되며, 최종적으로 통합 및 정제 과정을 거쳐 일관된 문서를 완성합니다. 실시간 지식 업데이트는 RAG(Retrieval-Augmented Generation) 접근 방식을 사용하여 최신 연구 논문을 동적으로 반영합니다. 또한 Multi-LLM-as-Judge 전략을 통해 여러 LLMs를 사용하여 생성된 내용을 평가합니다. 이 방법은 학술 기준에 따라 초기 평가 지표를 생성하고, 인간 전문가의 피드백으로 정제하여 신뢰성을 높입니다.

- **Performance Highlights**: AutoSurvey는 최대 64k 토큰 길이의 실험에서 높은 인용 및 내용 품질 점수를 달성했습니다. 64k 토큰에서 인용 품질은  recall 82.25%와 precision 77.41%를 기록했으며, 이는 인간 성과에 근접한 수준입니다. 내용 품질에서는 coverage(4.73), structure(4.33), relevance(4.86)의 점수를 기록하여 인간 성과와 유사한 결과를 보였습니다. 또한, 다양한 길이의 조사에서도 일관된 성과를 유지했으며, 다중 LLM 방식의 평가 결과는 인간 전문가와의 상관관계가 높게 나타났습니다(Spearman's rho = 0.5429).



### Robust portfolio optimization for recommender systems considering uncertainty of estimated statistics (https://arxiv.org/abs/2406.10250)
- **What's New**: 이 논문은 추천 항목의 정확성과 다양성 사이의 균형을 맞추기 위해 포트폴리오 최적화 모델을 제안합니다. 특히, 평균-분산 포트폴리오 최적화에 필요한 통계량(기대값, 공분산) 추정 시 발생하는 오차를 해결하기 위해 '로버스트 최적화' 기술을 사용하여 신뢰할 수 있는 솔루션을 도출합니다. 이 모델은 불확실성을 다루기 위해 'Cardinality-based uncertainty sets'를 기반으로 한 혼합 정수 선형 최적화 문제로 축소할 수 있으며, 수학적 최적화 솔버를 통해 정확하게 해결할 수 있습니다.

- **Technical Details**: 제안된 로버스트 포트폴리오 최적화 모델은 통계량 추정의 불확실성을 처리하기 위해 'Cardinality-based uncertainty sets'를 사용합니다. 이 모델은 혼합 정수 선형 최적화 문제로 변환되어 수학적 최적화 솔버로 해결 가능하며, 실험에서는 두 개의 공개된 평점 데이터셋을 사용하여 성과를 확인했습니다. 기존의 평균-분산 포트폴리오 최적화 모델과 비교하여 추천 항목의 정확성과 다양성을 모두 향상했습니다.

- **Performance Highlights**: 실험 결과, 제안된 로버스트 포트폴리오 최적화 모델은 기존의 평균-분산 포트폴리오 최적화 모델에 비해 추천 정확도 뿐만 아니라 추천 다양성에서도 향상을 보였습니다. 이러한 결과는 사용자의 평점에 대한 기대와 공분산의 불확실성을 적절히 처리함으로써, 다양한 요구를 충족하는 고품질의 추천 목록을 생성할 수 있는 잠재력을 보여줍니다. 또한, 다양한 평점 예측 알고리즘의 추천 품질을 개선할 수 있는 가능성도 시사합니다.



### Semantic-Enhanced Relational Metric Learning for Recommender Systems (https://arxiv.org/abs/2406.10246)
- **What's New**: 최근 추천 시스템 분야에서 많은 주목을 받고 있는 관계적 메트릭 학습(relational metric learning) 방법들이 등장했습니다. 이 방법들은 지식 그래프(knowledge graph)에서의 번역 메커니즘(translation mechanism)에서 영감을 얻었으며, 사용자와 아이템 간의 명시적 관계가 부족한 문제를 해결하고자 합니다. 본 논문에서는 이를 극복하기 위해 심층 의미 정보를 통합한 SEMRL(Semantic-Enhanced Relational Metric Learning) 프레임워크를 제안합니다.

- **Technical Details**: SEMRL은 세 가지 주요 모듈로 구성됩니다. 첫 번째 모듈은 HLSTM(Hierarchical Long Short-Term Memory)과 어텐션 메커니즘(attention mechanism)을 사용하여 리뷰로부터 잠재 의미 정보를 추출하는 텍스트 표현 학습 모듈입니다. 두 번째 모듈은 앞서 언급한 의미 신호를 기반으로 관계 유도 과정을 개선하는 관계 유도 모듈입니다. 세 번째 모듈은 사용자의 리뷰와의 유사성을 기반으로 최종적으로 삼중 모델링을 수행하는 관계 메트릭 학습 모듈입니다. 이 모든 모듈은 통합된 프레임워크로 결합되어 엔드 투 엔드(end-to-end) 학습이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, SEMRL은 네 가지 공공 데이터셋에서 여러 최신 방법들과 비교하여 경쟁력 있는 성능을 나타냈습니다. 특히, SEMRL은 추가 의미 신호를 통해 관계 유도 과정을 개선함으로써 사용자와 아이템의 구별 능력을 강화합니다.



### On conceptualisation and an overview of learning path recommender systems in e-learning (https://arxiv.org/abs/2406.10245)
- **What's New**: 최근의 연구 프로젝트에서 e-learning 시스템에 추천 시스템(recommender systems)을 도입하는 새로운 시도가 이루어지고 있습니다. iMath 프로젝트의 일환으로, 개별 학생의 성과와 선호도에 맞춘 개인화된 학습 경로를 개발하는 작업이 진행되고 있습니다. 이 연구는 다양한 추천 시스템을 실험하고, 이를 통해 학생들의 이해와 학습을 돕는 새로운 방법을 모색하고 있습니다.

- **Technical Details**: 이 프로젝트는 고등 교육 수학 과목을 수강하는 학생들을 위한 AI 기반 도구를 개발하는 것이 목표입니다. 이 도구는 학생 개개인의 수준과 필요에 맞춘 개인화된 e-learning 경로를 제시합니다. 프로젝트는 MathE 플랫폼을 통해 구현되며, 여기에는 21개의 수학 주제와 다양한 학습 자료가 포함되어 있습니다. 추천 시스템은 주로 학생의 학습 스타일, 성과, 선호도를 반영한 동적 알고리즘을 사용하여 구현됩니다. NLP(자연어 처리)에서 사용하는 Term-Document 매트릭스와 유사한 Keyword-Question 매트릭스를 활용하여 학습 지표를 분석합니다.

- **Performance Highlights**: iMath 프로젝트를 통해 개발된 추천 시스템은 학생 개개인의 학습 경로를 동적으로 조정함으로써 기존의 일률적인 학습 경로를 탈피할 수 있도록 설계되었습니다. 추천 시스템은 여러 알고리즘(협업 필터링(collaborative filtering), 내용 기반(content-based) 기법 등)을 적용하여 학습 자료와 평가 데이터를 분석하고, 이를 통해 학생의 만족도와 학습 성과를 극대화하려고 합니다.



### GLINT-RU: Gated Lightweight Intelligent Recurrent Units for Sequential Recommender Systems (https://arxiv.org/abs/2406.10244)
- **What's New**: 이번 논문에서는 Sequential Recommender Systems (SRSs)에서 효율적인 추론 속도를 제공하는 새로운 프레임워크인 GLINT-RU를 제안합니다. 이 프레임워크는 dense selective Gated Recurrent Units (GRU) 모듈을 활용하여 추론 시간을 단축하고 GPU 메모리 사용을 최소화합니다. 특히, GRU 모듈과 gated Multi-layer Perceptron (MLP)가 결합되어 사용자-아이템 상호작용 정보를 효과적으로 처리합니다.

- **Technical Details**: GLINT-RU는, dense selective GRU 모듈을 활용하며, 이 모듈은 긴 시간과 짧은 시간의 아이템 의존성을 모두 포착할 수 있습니다. 또한 이 모듈은 추론 시간을 크게 줄이고 GPU 메모리 사용을 최소화합니다. 추가적으로, 글로벌 사용자-아이템 상호작용 정보를 포함한 mixing block과, 정보가 깊게 필터링되는 gated MLP 블록이 통합되었습니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 광범위한 실험 결과, GLINT-RU는 기존의 GRU, Transformer, MLP 및 State Space Model (SSM) 기반의 모델들을 성능 및 추론 속도 면에서 능가하는 것으로 나타났습니다. 이 모델은 새로운 표준을 제시하며, 순차 추천 시스템 분야에서 놀라운 성능을 입증했습니다.



### Predict Click-Through Rates with Deep Interest Network Model in E-commerce Advertising (https://arxiv.org/abs/2406.10239)
Comments:
          Accepted by the 5th International Conference on Information Science, Parallel and Distributed Systems (ISPDS 2024), 2024 IEEE

- **What's New**: 이 논문은 Alibaba의 Taobao 플랫폼 광고 시스템에 적용된 새롭고 향상된 클릭률 예측(CTR Prediction) 모델을 제안한다. 이 연구는 Deep Interest Network(DIN) 모델을 사용하며, 전통적인 딥러닝 접근법과 달리 로컬화된 사용자 행동 활성화에 중점을 두고 맞춤형 광고 타겟팅을 가능하게 한다.

- **Technical Details**: DIN 모델은 광범위한 사용자 행동 데이터를 활용하여 사용자별 맞춤형 광고 타겟팅을 수행한다. 이는 다이나믹하고 다양한 사용자 데이터를 처리하는 능력이 뛰어나 기존의 딥러닝 모델보다 우수하다.

- **Performance Highlights**: 전통적인 모델과 비교하여, 이 방법은 광고 시스템의 효율성을 향상시키고 수익을 증가시키는 데 있어서 탁월한 성능을 보인다.



### Towards commands recommender system in BIM authoring tool using transformers (https://arxiv.org/abs/2406.10237)
- **What's New**: 본 연구에서는 건축, 엔지니어링 및 건설(AEC) 부문에서의 BIM(빌딩 정보 모델링) 소프트웨어 사용을 촉진하기 위한 새로운 접근 방법을 제시한다. 사용자 상호작용 데이터를 바탕으로, BIM 소프트웨어 명령(command)을 추천할 수 있는 순차 추천 시스템(sequential recommendation system)의 가능성을 탐구하였다. 이 시스템은 사용자의 이전 상호작용을 바탕으로 다음 명령을 예측하여 실시간 명령 제안을 제공한다.

- **Technical Details**: 연구의 근간이 되는 프레임워크는 대규모 BIM 로그 데이터를 전처리하고, 최신 대형 언어 모델의 트랜스포머 아키텍처(transformer architectures)를 백본(backbone)으로 활용하였다. 이를 통해 BIM 작성 도구인 Vectorworks 내에서 실시간 명령 제안을 가능하게 하는 프로토타입을 구축했다. 특히, BIM 소프트웨어 명령을 추천 가능한 항목으로 취급하여 사용자의 편의성을 극대화하는데 초점을 두었다.

- **Performance Highlights**: 후속 실험을 통해 제안된 모델이 이전 연구를 능가하는 성능을 보였음을 확인하였다. 이로써 순차 추천 시스템이 설계 효율성을 획기적으로 향상시킬 수 있는 잠재력을 갖고 있음을 증명했다.



### CF Recommender System Based on Ontology and Nonnegative Matrix Factorization (NMF) (https://arxiv.org/abs/2406.10235)
- **What's New**: 이 논문은 추천 시스템에서 발생하는 데이터 부족(data sparsity)과 정확도(accuracy) 문제를 해결하기 위해 새로운 접근법을 제안합니다. 특히, 이 접근법은 차원 축소 방법(dimensional reduction method)에 기반한 협업 필터링(collaborative filtering)의 행렬 분해 알고리즘(matrix factorization algorithm)을 사용하며, 구체적으로는 온톨로지(ontology)와 결합된 비음수 행렬 분해(Nonnegative Matrix Factorization, NMF)를 활용합니다.

- **Technical Details**: 기술적으로, 이 방법은 NMF 알고리즘을 온톨로지와 결합하여 대규모 데이터셋에서 데이터 희소성(sparsity) 문제를 해결하고, 더 정확한 추천을 제공합니다. NMF는 행렬을 양수 값들로만 분해하여 데이터의 구조적인 특성을 잘 반영하며, 온톨로지는 데이터의 의미적 관계를 이용하여 추천의 질을 향상시킵니다. 이러한 접근법은 기존의 고전적 방법들과 비교하여 실시되었습니다.

- **Performance Highlights**: 실험 결과, 이 새로운 접근법은 협업 필터링에서 희소성을 효과적으로 줄이고, 추천 항목의 정확성을 개선시키며, 사용자에게 더 관련성 높은 항목을 추천하는데 성공했습니다. 이는 기존의 전통적인 방법들보다 뛰어난 성능을 보여줍니다.



### Long Code Arena: a Set of Benchmarks for Long-Context Code Models (https://arxiv.org/abs/2406.11612)
Comments:
          54 pages, 4 figures, 22 tables

- **What's New**: Long Code Arena는 코드 처리 작업을 위한 6가지 벤치마크로 구성된 새로운 테스트 세트를 소개합니다. 기존의 단일 파일 또는 메소드 수준의 벤치마크와 달리, Long Code Arena는 프로젝트 전체의 문맥을 요구하는 작업들을 다룹니다.

- **Technical Details**: 이 벤치마크에는 라이브러리 기반 코드 생성, CI 빌드 수리, 프로젝트 레벨 코드 완성, 커밋 메시지 생성, 버그 로컬라이제이션, 모듈 요약 등이 포함됩니다. 각 작업에 대해 수동 검증된 평가 데이터셋, 평가 도구, 그리고 인기 있는 LLM(대규모 언어 모델)에 기초한 베이스라인 솔루션을 제공합니다. 모든 데이터는 오픈 소스 GitHub 저장소에서 수집되었으며, Python 코드가 주로 사용됩니다.

- **Performance Highlights**: Long Code Arena의 각 작업에 대해 모델들은 평균적으로 254개의 Python 파일과 2.5M(characters)의 코드, 그리고 2,242개의 고유한 클래스 및 메서드 이름을 포함하는 라이브러리를 기반으로 코드를 생성해야 합니다. 성능 평가는 주로 ChrF(Chracter F-score)와 API Recall이라는 메트릭을 통해 측정됩니다. 이를 통해 모델이 생성한 코드가 원본 코드와 얼마나 유사한지, 그리고 특정 라이브러리 API 호출을 얼마나 잘 사용하는지를 평가합니다.



### CoSQA+: Enhancing Code Search Dataset with Matching Cod (https://arxiv.org/abs/2406.11589)
Comments:
          11 pages, 4 figures, conference

- **What's New**: 이 논문에서는 자연 언어 쿼리(Natural Language Query)와 일치하는 코드를 검색하는 작업인 시맨틱 코드 검색(Semantic Code Search)의 문제를 다루고 있습니다. 기존의 코드 검색 데이터셋은 비현실적인 쿼리(Unrealistic Queries)를 사용하거나 쿼리와 코드가 맞지 않는 경우(Mismatched Codes)가 많았습니다. 이를 개선하기 위해 고품질 쿼리와 여러 개의 적절한 코드가 매칭된 CoSQA+ 데이터셋을 소개합니다. CoSQA+는 다양한 소스에서 코드를 수집하고, 큰 언어 모델(LLM)을 이용해 쿼리와 코드의 페어를 자동으로 주석(Annotation)하고 필터링하여 생성하였습니다.

- **Technical Details**: CoSQA+는 CoSQA 데이터셋에서 쿼리를 가져오고 StaQC와 CSN 데이터셋에서 코드를 수집하여 여러 모델을 통하여 쿼리와 코드 페어를 생성하였습니다. LLM인 Claude 3 Sonnet를 이용해 후보 페어에 주석을 달고, 정확히 맞는 코드가 없는 경우에는 GPT-4o로 코드를 생성하였습니다. 또한 다중 선택 코드 검색 성능을 평가하기 위해 새로운 지표인 MMRR(Mean Multi-choice Reciprocal Rank)을 도입하였습니다.

- **Performance Highlights**: CoSQA+ 데이터셋의 품질을 평가하기 위한 실험에서, CoSQA+의 쿼리-코드 페어는 CoSQA의 페어보다 62.9% 더 높은 품질로 선택되었습니다. CodeBERT를 CoSQA+에서 파인튜닝한 결과, CSN Python 데이터셋에서 MMRR 0.902를 기록하여 CoSQA에서의 0.850보다 우수한 성능을 보였습니다. 또한, Claude 3 Sonnet를 이용한 자동 주석은 인간 수준의 성능에 가까운 정확도를 보여주었고, MMRR 지표는 다중 선택 코드 검색 성능 평가에 매우 신뢰할만한 것으로 나타났습니다.



### They're All Doctors: Synthesizing Diverse Counterfactuals to Mitigate Associative Bias (https://arxiv.org/abs/2406.11331)
- **What's New**: 이번 연구에서는 CLIP와 같은 Vision Language Models(VLMs)의 불균형 및 불공정한 결과를 개선하기 위해 새로운 프레임워크를 제안했습니다. 이 프레임워크는 텍스트-이미지 모델을 통해 생성된 합성 대체 이미지를 활용하여 다양한 데이터셋을 생성하고, 이를 통해 CLIP를 미세 조정합니다.

- **Technical Details**: 제안된 프레임워크는 기본 합성 이미지 세트를 사용하고, 세그먼테이션(segmentation) 및 인페인팅(inpainting) 모델들을 활용하여 다양한 외모를 가진 인간들을 이미지에 배치합니다. 이러한 방식을 통해 이미지에서 인간의 외모와 맥락을 분리하도록 CLIP를 훈련시킬 수 있으며, 직업이나 상황을 설명하는 요소들이 외모와 관계없이 맥락(예: 배경, 복장, 소품)으로 정의됨을 학습합니다.

- **Performance Highlights**: 우리의 미세 조정 된 CLIP 모델($CF_\alpha$)은 이미지 검색 작업에서 MaxSkew, MinSkew, NDKL와 같은 주요 공정성 지표를 40-66% 향상시켰습니다. 또한, 본래의 CLIP 모델과 호환성을 최대한 유지하며, 사용자가 정확성과 공정성 간의 균형을 설정할 수 있도록 가중치 인터폴레이션(weight interpolation) 기법을 사용하여 제어 가능하게 설계되었습니다.



### ptt5-v2: A Closer Look at Continued Pretraining of T5 Models for the Portuguese Languag (https://arxiv.org/abs/2406.10806)
- **What's New**: 최근 Natural Language Processing (NLP) 분야에서 영어 모델에 비해 다른 언어의 모델 개발이 상대적으로 부족한 가운데, T5 모델을 포르투갈어에 맞게 지속적으로 pretrained 한 새로운 연구가 소개되었습니다. 이 연구는 $	exttt{ptt5-v2}$을 도입하여 포르투갈어 코퍼스를 사용한 T5 모델의 지속적 사전 훈련이 다양한 downstream 작업에 미치는 영향을 분석합니다.

- **Technical Details**: 중심적인 접근 방식으로, Google의 T5 모델(최대 3B 파라미터)을 포르투갈어 텍스트로 지속적으로 사전 훈련했습니다. 여기에는 데이터셋의 품질 필터, 최적화 전략, 다중 에폭(epochs) 사전 훈련과 같은 다양한 프리트레이닝 설정이 포함되었습니다. 실험에서는 포르투갈어 mC4 데이터셋(약 524 GB 텍스트) 및 SentencePiece Unigram 토크나이저(32,000 토큰)를 사용했습니다. 프리트레이닝 접근 방식으로는 span corruption task를 사용했으며, Adafactor optimizer와 constant learning rate(0.001)를 적용했습니다.

- **Performance Highlights**: 세 가지 포르투갈어 다운스트림 작업(ASSIN2 STS, ASSIN2 RTE, TweetSentBR)에 대한 파인튜닝 결과, 후자의 두 작업에서는 state-of-the-art(SOTA) 성능을 달성했습니다. 그러나 프리트레이닝 설정의 변화가 baseline 대비 미미한 영향을 미친다는 것이 주요 발견 중 하나였습니다. $	exttt{ptt5-v2}$ 프리트레인 모델 체크포인트와 파인튜닝된 MonoT5 rerankers는 HuggingFace에서 공개되었습니다.



### HiddenTables & PyQTax: A Cooperative Game and Dataset For TableQA to Ensure Scale and Data Privacy Across a Myriad of Taxonomies (https://arxiv.org/abs/2406.10803)
Comments:
          In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)

- **What's New**: 최근 arxiv에 발표된 'HiddenTables'라는 새로운 협력형 게임에 대한 연구는 대규모 언어 모델(LLM)들이 표 기반 질의응답(Table QA) 과제를 해결하는 데 있어 마주하는 다양한 문제를 해결하려는 시도를 소개하고 있습니다.

- **Technical Details**: HiddenTables는 코드 생성 LLM(Solver)과 평가자(Oracle)의 협력 게임입니다. 여기서 Solver는 오라클이 제공하는 자연언어 구조(Natural Language Schema) 및 지시 사항에 따라 사용자 질문에 대한 코드를 생성하고, 이를 바탕으로 Oracle이 데이터를 보존하면서 답변을 제공합니다. 이 게임은 제한된 문맥 창, 토큰화 패턴과 셀 경계 사이의 불일치, 외부 모델 사용시 생기는 데이터 기밀성 문제 등을 해결하려는 목적으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, HiddenTables는 전체 데이터를 노출하지 않고도 다량의 데이터셋과 상호작용할 수 있는 효율성을 보여주었습니다. 새로운 데이터셋 'PyQTax'를 생성하여 116,671개의 질문-테이블-정답 트립렛을 포함하고, 다양한 질문 분류에 대한 세밀한 분류 및 라벨을 제공합니다. 또한 이 게임을 통해 gpt-3.5-turbo의 효율성을 높였으며, 피드백을 반복적으로 주고받는 과정에서 정확도가 개선되었습니다.



### SparseCL: Sparse Contrastive Learning for Contradiction Retrieva (https://arxiv.org/abs/2406.10746)
- **What's New**: 저자들은 'SparseCL'라 불리는 새로운 접근 방식을 도입하여, 큰 문서 집합에서 쿼리에 모순되는 문서를 효과적으로 검색하는 방법을 제안하였습니다. SparseCL은 문장 임베딩(sentence embeddings)을 활용하여 문장 간 미묘한 모순점을 보존하도록 특별히 훈련되었습니다. 이 접근 방식은 코사인 유사도(cosine similarity)와 희소성 함수(sparsity function)를 결합한 지표를 사용하여 쿼리에 모순되는 문서를 효율적으로 식별하고 검색합니다.

- **Technical Details**: SparseCL 방법은 단어 임베딩의 차이에서 희소성을 유지하도록 문장 임베딩 모델을 훈련시킵니다. 쿼리에 응답할 때, 코사인 유사도와 임베딩의 차이 희소성을 기반으로 각 문서와 쿼리 간의 점수를 계산하고, 가장 높은 점수를 가진 문서를 검색합니다. 희소성의 구체적인 척도로는 Hoyer measure을 사용하였습니다.

- **Performance Highlights**: 이 방법은 Arguana 데이터셋 및 GPT-4를 사용하여 생성된 MSMARCO와 HotpotQA 데이터셋에서 테스트되었습니다. 실험 결과, MSMARCO와 HotpotQA에서 30% 이상의 정확도 향상이 확인되었습니다. 또한, 손상된 데이터 집합을 정리하고 고품질 QA검색 정확도를 복원하는 데에도 성공적으로 적용되었습니다.



### QDA-SQL: Questions Enhanced Dialogue Augmentation for Multi-Turn Text-to-SQL (https://arxiv.org/abs/2406.10593)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문은 다중 턴(Text-to-SQL) 작업에서 큰 언어 모델(LLMs)을 미세 조정하는 과정에서 발생하는 모호하거나 답변할 수 없는 질문을 효과적으로 처리할 수 있는 새로운 데이터 증강 방법, QDA-SQL을 제안합니다. QDA-SQL은 다양한 종류의 다중 턴 Q&A 쌍을 생성하여 LLM의 성능을 향상시킵니다.

- **Technical Details**: QDA-SQL은 기존 데이터 증강 방법에 검증 및 수정 메커니즘을 도입하여 복잡한 다중 턴 Text-to-SQL 작업을 처리합니다. 본 논문에서는 질문 유형 인식 및 샘플 생성 방법을 통해 Text-to-SQL 모델의 신뢰성과 적용 가능성을 크게 향상시키는 새로운 방법론을 소개합니다. 특히, QDA-SQL은 'Chain-of-Thought(CoT)' 기법을 사용해 다양한 Q&A 유형과 주제 간의 관계를 무작위로 결합하여 다방면의 대화 형식을 학습합니다.

- **Performance Highlights**: 실험 결과에 따르면, QDA-SQL을 통해 미세 조정된 모델은 SQL 문장 정확도와 복잡한 다중 턴 Text-to-SQL 작업에서 비정상적인 질문 처리 능력이 향상되었습니다. 이는 다양한 평가 지표를 통해 입증되었으며, QDA-SQL이 텍스트와 SQL 간의 여러 유형의 질문을 효과적으로 처리할 수 있음을 보여줍니다.



### Determination of the Number of Topics Intrinsically: Is It Possible? (https://arxiv.org/abs/2406.10402)
Comments:
          This is the first full draft version of the article. The camera-ready version was accepted at the 11th International Conference on Analysis of Images, Social Networks and Texts (AIST 2023). Presented on September 30, 2023. Expected to be published in the conference proceedings, as part of the Communications in Computer and Information Science series (CCIS, Vol. 1905)

- **What's New**: 이 연구는 데이터셋에서 주제(topic)의 수를 추정하는 다양한 방법들을 비교 분석합니다. 기존의 방법들이 충분히 비교되지 않았고, 주제의 수가 특정한 데이터셋의 절대적인 속성이 아닌 방법과 모델에 따라 달라진다는 것을 강조합니다. 따라서 새로운 접근방식이 필요하며, 이 방향에서의 추가 연구가 제안됩니다.

- **Technical Details**: 주제 모델(topic model)은 unsupervised 텍스트 분석을 위한 통계 모델입니다. 주제 모델은 'word-in-topic' 분포(Φ)와 'topic-in-document' 분포(Θ)라는 두 확률 분포를 통한 훈련을 기반으로 합니다. 이 연구는 내재적 지표(intrinsic metric)를 중심으로 여러 방법을 검토하고, 각 모델과 데이터셋에 따른 실험적 증거를 제공합니다. 주요 방법으로는 hold-out perplexity(holdPerp), 안정성 분석(stability analysis), Jaccard Similarity Index, cophenetic correlation coefficient 등이 있습니다.

- **Performance Highlights**: 기존의 내재적 방법들이 신뢰성과 정확성이 떨어진다는 것을 발견했습니다. 주제의 수(T)는 특정 데이터셋의 고유 속성이 아닌, 사용된 방법과 모델에 의존한다는 결과를 도출했습니다. 이 연구는 수치 지표를 통해 주제 모델을 평가하고, 새로운 지표와 방법론 개발의 필요성을 강조합니다.



### Robustness of Structured Data Extraction from In-plane Rotated Documents using Multi-Modal Large Language Models (LLM) (https://arxiv.org/abs/2406.10295)
Comments:
          20 pages, 6 figures

- **What's New**: 이번 연구는 최신 멀티 모달 대형 언어 모델(LLM)들이 문서의 흔들림(평면 내 회전 또는 skew)에 어떻게 영향을 받는지 조사합니다. 문서 데이터 추출 정확도가 이러한 흔들림으로 인해 얼마나 영향을 받는지 살펴봅니다. 이에 따라 Anthropic Claude V3 Sonnet, GPT-4-Turbo, Llava:v1.6 등의 모델에 대한 실험 결과를 제시합니다.

- **Technical Details**: 문서의 흔들림이 데이터 추출 정확도에 미치는 영향을 조사하기 위해, 다양한 각도의 흔들림을 갖는 합성 샘플 문서에서 특정 엔티티를 추출하는 실험을 진행했습니다. 모델별로 안전한 회전 각도(SIPRA)를 식별하고, 일반적인 skew 탐지 및 수정 메커니즘의 한계와 대안을 제안합니다. 특히, 선제적으로 더 견고한 멀티 모달 아키텍처를 개발하거나 모델의 사전 학습 단계에서 skewing 기법을 통합하는 방법을 고려합니다.

- **Performance Highlights**: 모든 테스트 모델에서 문서의 흔들림이 데이터 추출 정확도에 부정적인 영향을 미쳤으며, 그 영향의 정도는 모델별로 상이했습니다. 또한, 모델의 허위 정보 생성(환각)에도 흔들림이 영향을 미치는 것으로 나타났습니다. 이는 실세계를 반영한 폭넓은 문서 품질 및 조건을 테스트할 필요성을 강조합니다.



### ResearchArena: Benchmarking LLMs' Ability to Collect and Organize Information as Research Agents (https://arxiv.org/abs/2406.10291)
- **What's New**: 이번 연구에서는 대형 언어 모델들(LLMs)이 학술 조사를 수행하는 능력을 측정하기 위한 벤치마크 'ResearchArena'가 개발되었습니다. 이 벤치마크는 학술 조사 과정을 세 단계로 분해하여 평가합니다: 정보 발견(information discovery), 정보 선택(information selection), 및 정보 조직(information organization). 또한, 이 벤치마크는 1,200만 개의 전체 텍스트 학술 논문과 7,900개의 조사 논문으로 구성된 오프라인 환경을 포함합니다.

- **Technical Details**: ResearchArena는 학술 조사 과정의 세 가지 주요 작업을 다룹니다. 첫 번째 작업인 정보 발견(information discovery) 단계에서는 LLM들이 관련 논문을 식별하고 검색하는 능력을 평가합니다. 두 번째 작업, 정보 선택(information selection) 단계는 검색된 논문의 중요성과 관련성을 평가하는 능력을 요구합니다. 세 번째 작업, 정보 조직(information organization) 단계에서는 선택된 논문을 위계적인 지식 마인드맵으로 구성하는 능력을 평가합니다. 이러한 모든 작업은 대규모 학술 데이터셋을 활용하여 엄격히 평가됩니다.

- **Performance Highlights**: 기존의 LLM 기반 방법들은 기본적인 키워드 검색 기법과 비교했을 때 성능이 저하되는 것으로 나타났습니다. 특히, 정보 발견과 정보 선택 단계에서는 LLM들이 전통적인 검색 기법보다 낮은 재현율(recall)과 정밀도(precision)를 보였습니다. 또한, 정보 조직 단계에서도 일관적이고 정확한 지식 구조를 구성하는 데 어려움을 겪는 것으로 나타났습니다. 이러한 결과들은 LLM의 향후 연구를 위한 상당한 개선 기회를 시사합니다.



### VeraCT Scan: Retrieval-Augmented Fake News Detection with Justifiable Reasoning (https://arxiv.org/abs/2406.10289)
- **What's New**: 새로운 논문에서는 가짜 뉴스(Fake News)의 확산 문제에 대응하기 위해 VeraCT Scan이라는 혁신적인 시스템을 소개했습니다. 이 시스템은 뉴스 기사에서 핵심 사실을 추출한 후 인터넷 전역에서 이를 확인하는 검색(Internet-wide search) 과정을 통해 뉴스의 진위 여부를 판별합니다. 결과적으로 투명하고 신뢰할 수 있는 증거와 논리를 제공하여 뉴스의 신뢰성을 높이는 것을 목표로 합니다.

- **Technical Details**: VeraCT Scan은 정보 검색 최적화 기법과 결합된 포괄적인 파이프라인을 갖추고 있습니다. GPT-4 Turbo와 Llama-2 13B를 미세 조정(Fine-tuning)하여 뉴스 내용 이해, 검증, 추론 작업을 수행합니다. 핵심 사실을 다중 수준으로 세분화하고, 각 사실별로 인터넷 검색을 수행하여 관련 정보를 수집한 후, 출처 신뢰도를 고려하여 뉴스의 진위를 판별합니다. 또한, 시스템의 투명성과 신뢰성을 높이기 위해 검증 논리를 생성하고 관리합니다.

- **Performance Highlights**: VeraCT Scan은 여러 가짜 뉴스 검출 데이터셋에서 종합적인 평가를 통해 최고 수준의 성능(State-of-the-art accuracy)을 입증했습니다. 이는 자동 생성 및 미세 조정된 대규모 언어 모델들(LLMs)을 활용하여 높은 정확도의 뉴스 검증 작업을 수행한 결과입니다.



### D\'eveloppement automatique de lexiques pour les concepts \'emergents : une exploration m\'ethodologiqu (https://arxiv.org/abs/2406.10253)
Comments:
          in French language. JADT 2024

- **What's New**: 본 논문은 비기술적 혁신에 중점을 둔 신흥 개념 중심의 어휘집 (lexicon) 개발을 소개합니다. 여러 도메인에 걸쳐 일반화할 수 있는 모델을 확립하기 위해 인간 전문 지식, 통계 분석, 기계 학습 기법을 결합한 4단계 방법론을 소개합니다.

- **Technical Details**: 이 방법론은 주제별 말뭉치(corpus)의 생성, Gold Standard Lexicon(골드 스탠다드 어휘집)의 개발, 교육용 말뭉치(annotation 및 준비), 그리고 새로운 용어를 식별하기 위한 학습 모델의 구현을 포함합니다.

- **Performance Highlights**: 결과는 해당 접근 방식의 견고성과 관련성을 입증하며, 다양한 맥락에서의 적응성과 어휘 연구에의 기여를 강조합니다. 이 개발된 방법론은 개념적 분야에서 적용 가능성을 약속합니다.



### Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction (https://arxiv.org/abs/2406.07979)
Comments:
          Accepted by KDD 2024

- **What's New**: 새로운 연구인 HL-GNN(Heuristic Learning Graph Neural Network)을 발표했습니다. 이는 그래프 내의 로컬과 글로벌 휴리스틱을 통합하여 링크 예측 문제를 해결하는데 초점을 맞추고 있습니다.

- **Technical Details**: HL-GNN은 인접 행렬(adjacency matrix) 곱셈을 통해 로컬과 글로벌 휴리스틱을 통합하는 매트릭스 형식을 제안하고, 이를 바탕으로 효율적인 구현을 위해 intra-layer 전파와 inter-layer 연결을 채택합니다. 덧붙여, GCN보다 낮은 시간 복잡도로 최대 20 계층까지 학습할 수 있습니다.

- **Performance Highlights**: HL-GNN은 Planetoid, Amazon, OGB 데이터셋을 사용한 광범위한 실험에서 최첨단 성능을 보여주었습니다. 주요 하이라이트로는 기존 방법들보다 훨씬 더 우수한 예측 성능을 보여주었고, 휴리스틱 기반 방법들보다 몇 배 더 빠른 속도를 자랑합니다. 또한 소수의 학습 가능한 매개변수만을 요구하며, 학습된 가중치와 일반화된 휴리스틱이 매우 해석 가능하다는 점도 우수한 점입니다.



### Khmer Semantic Search Engine (KSE): Digital Information Access and Document Retrieva (https://arxiv.org/abs/2406.09320)
- **What's New**: 이 연구는 캄보디아에서 처음으로 제안된 Khmer Semantic Search Engine(KSE)을 소개합니다. KSE는 기존의 Khmer 검색 방법을 향상시켜, 고급 시맨틱 매칭 기술을 활용하여 정확한 검색 결과를 제공합니다. 특히, 키워드 사전 기반, 온톨로지 기반 및 랭킹 기반의 세 가지 시맨틱 검색 프레임워크를 제안하여 사용자의 검색 정확성을 극대화합니다.

- **Technical Details**: KSE는 사용자 쿼리에서 의미 있는 키워드를 추출하고, 시맨틱 콘텐츠를 형식적으로 주석 달아 정확한 매칭을 수행합니다. 이 시스템은 자동 및 수동 키워드 추출 도구와 온톨로지 기반 시맨틱 강화 도구를 개발하여 높은 품질의 입력 데이터를 보장합니다. 또한, 기존의 TF-IDF, Word2Vec, BERT 등 다양한 최신 기술들을 통합하여 고성능 검색 결과를 달성합니다.

- **Performance Highlights**: KSE의 성능은 그라운드 트루스 (ground truth) 데이터셋을 사용하여 평가되었으며, 시맨틱 검색어 이해 능력을 통해 검색 정확도를 크게 향상시켰습니다. 이는 사용자가 보다 관련성 높은 문서와 URL을 찾는 데 큰 도움을 줄 수 있음을 보여줍니다.




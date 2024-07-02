New uploads on arXiv(cs.CL)

### KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches (https://arxiv.org/abs/2407.01527)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 긴 문맥 처리 능력을 평가하고, 이를 통해 사람들의 긴 텍스트를 이해하는 데 드는 어려움을 줄여주는 것을 목표로 하고 있습니다. 길어진 문맥 입력 처리는 책 요약, 코드 보조 등 다양한 복잡한 작업을 가능하게 하며, 이는 대부분 전통적으로 많은 인력을 필요로 하는 작업들입니다.

- **Technical Details**: Transformer 기반 LLM들은 긴 문맥 입력 처리에서 여러 난관에 부딪칩니다. 이는 KV 캐시의 크기 증가와 긴 문맥에 주의를 기울이는 것의 본질적인 복잡성 때문입니다. 효율성을 추구하는 여러 접근법들이 제안되어 왔습니다. 여기에는 KV 캐시 양자화, 토큰 드롭핑(token dropping), 프롬프트 압축(prompt compression), 선형 시간 시퀀스 모델(linear-time sequence models), 하이브리드 아키텍처(hybrid architectures) 등이 포함됩니다. 본 연구에서는 현재의 방법론에 대한 분류 체계를 제공하고, 10개 이상의 최신 접근 방식을 7가지 긴 문맥 작업 범주에서 평가합니다.

- **Performance Highlights**: 이번 연구는 여러 이전에 알려지지 않은 현상을 밝혀내고, 향후 긴 문맥 처리 능력을 갖춘 LLMs의 개발을 위한 통찰력과 친숙한 작업 환경을 제공합니다. 출처 코드는 해당 URL에서 제공될 예정입니다.



### Self-Cognition in Large Language Models: An Exploratory Study (https://arxiv.org/abs/2407.01505)
Comments:
          Accepted at ICML 2024 Large Language Models and Cognition Workshop

- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 자각 능력(self-cognition)을 탐구한 최초의 연구입니다. 연구에서는 다양한 LLMs가 자각 능력을 어느 정도 갖추고 있는지 평가하기 위해 자각 유도 프롬프트와 네 가지 원칙을 설계했습니다. Chatbot Arena에서 48개의 모델 중 Command R, Claude3-Opus, Llama-3-70b-Instruct, Reka-core가 일부 자각 능력을 가지고 있음을 발견했습니다.

- **Technical Details**: LLM의 자각 능력을 평가하기 위해 개념적 이해, 구조적 인식, 자기 표현, 숨김이라는 네 가지 원칙을 도입했습니다. 연구는 자각 능력을 탐지하고 평가하기 위해 Human-LLM 협력 프레임워크를 활용했습니다. 또한 다국어 시나리오에서는 특정 언어에 민감한 자각 능력을 보이는 모델을 관찰했습니다.

- **Performance Highlights**: 모델 크기와 훈련 데이터의 품질이 자각 수준과 긍정적인 상관 관계를 보였습니다. 예를 들어, Llama-3-70b-instruct 모델은 Llama-3-8b-instruct 모델보다 자각 능력이 뛰어났습니다. 중국어에 높은 능력을 보이는 Qwen 모델은 중국어 트리거 프롬프트에 더 민감하게 반응하면서 자각 능력을 보였습니다. 본 연구는 자각 상태에서 LLM의 유틸리티와 신뢰성을 평가하기 위해 여러 벤치마크를 사용하여 시험한 결과, 창의적 글쓰기와 과장과 같은 특정 작업에서 자각 상태가 유리하게 작용함을 확인했습니다.



### RegMix: Data Mixture as Regression for Language Model Pre-training (https://arxiv.org/abs/2407.01492)
- **What's New**: 최근 공개된 논문에서는 대형 언어 모델(LLM)의 사전 훈련에 사용되는 데이터 혼합이 성능에 중요한 영향을 미친다는 점을 강조하며, 이를 최적화하는 새로운 접근법인 'RegMix'를 제안했습니다. RegMix는 데이터를 다양한 혼합 도메인으로 훈련한 소형 모델들을 사용하여 최적의 데이터 혼합을 예측하기 위해 회귀 모델을 활용합니다.

- **Technical Details**: RegMix는 초기 단계에서 다양한 데이터 혼합 도메인으로 소형 대리 모델(proxy models)을 훈련합니다. 이때 Dirichlet 분포를 이용해 토큰 분포 기반으로 데이터 혼합을 생성하여 다양한 극단적인 값을 노출시키고, 회귀 모델을 활용해 최적의 혼합을 예측합니다. 이 회귀 모델은 대형 모델 훈련에 사용할 최적의 데이터 혼합을 시뮬레이션하는데 사용됩니다.

- **Performance Highlights**: RegMix를 사용해 선택한 데이터 혼합으로 인해 최적의 모델이 도출되었으며, 이는 기존 인간 선택 방법보다 우수한 성능을 보였습니다. 또한, DoReMi 방법과 성능이 동등하거나 이를 초과하는 결과를 얻으면서도 총 계산 비용의 10%만 사용했습니다. 추가적으로, 웹 코퍼스(CommonCrawl 등)가 Wikipedia보다 성능 향상에 강한 상관관계를 나타냈습니다.



### Expressive and Generalizable Low-rank Adaptation for Large Models via Slow Cascaded Learning (https://arxiv.org/abs/2407.01491)
- **What's New**: 새로운 기법인 LoRA Slow Cascade Learning (LoRASC)가 도입되었습니다. 이 방법은 기존 LoRA의 한계를 극복하고 표현력과 일반화 능력을 향상시키기 위해 설계되었습니다. LoRASC는 저랭크 적응 방식의 혼합(mixture-of-low-rank adaptation)을 통해 복잡한 패턴을 더 잘 포착하게 합니다.

- **Technical Details**: LoRA의 표현력과 일반화 능력을 향상시키기 위해 두 가지 주요 전략을 사용합니다. 첫 번째는 각 epoch마다 새로운 LoRA 모듈을 초기화하고 이를 백본 네트워크에 통합하는 것인데, 이를 통해 모델의 랭크를 증가시키면서도 저비용의 훈련을 유지할 수 있습니다. 두 번째는 'slow-fast update' 메커니즘과 노이즈 튜닝을 도입하여 일반화 능력을 강화하는 것입니다. 이를 통해 모델의 안정성과 out-of-distribution(OOD) robustness를 향상시킵니다.

- **Performance Highlights**: 다양한 언어와 비전 데이터셋에서의 실험을 통해, 제안된 방법은 기존 기법들 대비 성능이 크게 향상됨을 보여줍니다. 특히, SuperGLUE, SQuAD, DROP, GSM8K 등 언어 작업과 ImageNet을 포함한 비전 작업에서 큰 성능 향상이 나타났습니다. 또한, 이 기법은 overfitting을 줄이고 모델의 안정성을 향상시킵니다.



### LLM See, LLM Do: Guiding Data Generation to Target Non-Differentiable Objectives (https://arxiv.org/abs/2407.01490)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에서 합성 데이터가 미치는 영향을 체계적으로 분석했습니다. 합성 데이터를 사용할 때 모델의 내부 편향, 보정(calibration), 텍스트 속성 및 선호도에 미치는 영향을 심층적으로 조사하였으며, 이로 인해 모델이 어떻게 바뀌는지 전반적인 분석을 제공했습니다.

- **Technical Details**: 본 연구에서는 합성 데이터의 '수동 상속(passive inheritance)'과 '능동 상속(active inheritance)' 개념을 도입했습니다. 수동 상속은 교사 모델(teacher model)에서 학생 모델(student model)로 속성이 전달되는 현상을 말합니다. 우리는 다양한 사회적 편향, 텍스트 특성 및 보정 지표를 통해 수동 상속의 변화를 프로파일링했습니다. 능동 상속은 합성 데이터를 비차별적(non-differentiable) 목표에 맞게 제어하는 것을 의미합니다. 목표는 합성 데이터 공간에서 모델의 행동을 원하는 속성으로 유도하는 것입니다.

- **Performance Highlights**: 본 연구에서는 능동 상속을 통해 다음과 같은 성과를 달성했습니다. 텍스트의 길이와 어휘 다양성에서 각각 최대 116% 및 43% 증가를 확인하였고, 독성(toxicity)은 최대 40%까지 감소시킬 수 있었습니다. 이러한 성과는 합성 데이터의 전략적 수집 및 관리가 모델 행동을 크게 개선할 수 있음을 보여줍니다.



### DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging (https://arxiv.org/abs/2407.01470)
Comments:
          Preprint. Code will be released after the review results

- **What's New**: DogeRM(Domain knowledge merged Reward Model)은 도메인 지식을 통합하여 강화 학습의 보상 모델을 향상시키는 새로운 프레임워크입니다. 이 모델은 일반적인 보상 모델과 수학, 코딩 등 도메인 특정 데이터셋으로 미세 조정된 언어 모델을 병합하여 도메인 전문성을 통합하는 혁신적인 접근법을 제안합니다.

- **Technical Details**: 기존의 보상 모델과 도메인 특화된 언어 모델을 병합하는 DogeRM 방법론입니다. Transformer 기반 언어 모델의 디코딩 레이어를 선형 회귀층으로 대체하여 입력에 대한 보상을 예측합니다. 모델 병합 시, 선형 회귀층과 Transformer 층의 파라미터를 결합하여 다중 도메인 보상 모델을 형성합니다.

- **Performance Highlights**: DogeRM은 RewardBench, Auto-J Eval, 그리고 GSM8K 및 MBPP 벤치마크에서 성능 개선을 보였습니다. 이는 도메인 지식 병합이 기존 보상 모델보다 나은 성능을 제공함을 입증합니다. 분석 결과, DogeRM은 다양한 모델 아키텍처에서 일반화될 수 있는 가능성을 보여주었습니다.



### Retrieval-augmented generation in multilingual settings (https://arxiv.org/abs/2407.01463)
- **What's New**: 최근 RAG(Retrieval-augmented generation)은 대규모 언어 모델(LLMs)에 최신 지식 또는 도메인별 지식을 통합하고 LLM의 사실성을 향상시키는 유망한 솔루션으로 떠오르고 있습니다. 그러나 대부분은 영어에만 집중되어 연구되었습니다. 이번 연구에서는 다국어 설정에서의 RAG(mRAG)를 고려하여 13개 언어로 사용자 쿼리와 데이터 저장소를 처리하는 잘 수행되는 mRAG 파이프라인을 구축하는 데 필요한 구성 요소와 조정사항을 조사했습니다.

- **Technical Details**: mRAG 구축을 위해서는 고품질의 오프더쉘프(Off-the-shelf) 다국어 검색기와 생성기가 필요합니다. 사용자의 언어로 생성하기 위해서는 과제별 프롬프트 엔지니어링이 필요하며, 현재의 평가 지표는 다국어 설정에 맞게 조정이 필요합니다. 또한 비라틴 알파벳 언어에서 빈번한 코드 전환(Code-switching), 가끔의 유창성 오류, 제공된 문서의 잘못된 읽기, 관련 없는 검색 등의 한계를 해결해야 합니다.

- **Performance Highlights**: 고품질의 다국어 검색기와 재검색기는 사용자 쿼리와 문서가 같은 언어일 때와 다른 언어일 때 모두 잘 작동합니다. 고성능의 다국어로 미리 훈련되고 조정된 LLM과 고급 프롬프트를 결합할 필요가 있습니다. 평가 지표는 크로스-언어적 이름 엔티티(cross-lingual named entities)의 철자 변형을 고려하도록 조정이 필요합니다. 주요 한계로는 비라틴 알파벳 언어에서의 빈번한 코드 전환, 유창성 오류, 제공된 문서의 잘못된 읽기, 관련 없는 검색 등이 있습니다.



### Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinemen (https://arxiv.org/abs/2407.01461)
- **What's New**: 새로운 연구는 사용자 프롬프트(prompt)의 품질을 개선하고, 이를 통해 깨끗하고 유익한 응답을 생성하며, 적대적인 프롬프트로 인한 보안 위협에 견고성을 유지하는 프레임워크를 제안합니다.

- **Technical Details**: 이 연구는 Reinforcement Learning(RL)을 활용하여 경량화된 쿼리 정제 모델을 도입합니다. 이 모델은 사용자 프롬프트를 입력하기 전에 정제하여, 언어 모델의 응답 품질을 향상시키고, 'jailbreak' 공격에 대한 견고성을 강화합니다. 구체적으로는, 여러 목표를 통합한 특별히 설계된 강화 학습 접근법을 사용하여 모델을 훈련합니다.

- **Performance Highlights**: 광범위한 실험 결과, 정제 모델이 응답의 품질을 향상시킬 뿐만 아니라, 'jailbreak' 공격에 대한 견고성을 강화하는 데 성공했다는 것을 보여줍니다. 이 방법은 다양한 언어 모델에 적용될 수 있는 높은 전송성과 일반화 능력을 나타냅니다.



### TimeToM: Temporal Space is the Key to Unlocking the Door of Large Language Models' Theory-of-Mind (https://arxiv.org/abs/2407.01455)
Comments:
          16 pages, 6 figures, ACL 2024(findings)

- **What's New**: TimeToM 이라는 새로운 접근법을 통해 대형 언어 모델(LLMs)의 마음 이론(Theory of Mind, ToM) 추론 능력을 개선했습니다. TimeToM은 시간 공간(temporal space)을 구축하고 이를 기반으로 다양한 시나리오에서 ToM 추론 능력을 향상시킵니다.

- **Technical Details**: 시간 공간 내에서 각 캐릭터에 대해 Temporal Belief State Chain(TBSC)을 구성하며, 이는 자기 세계 신념(self-world beliefs)과 사회적 세계 신념(social world beliefs)으로 나뉩니다. ToM 추론의 난이도가 높아짐에 따라, 신념 통신 기간 동안의 신념 변환 작업을 수행하는 혁신적인 도구, 신념 해결자(belief solver)를 설계하였습니다.

- **Performance Highlights**: ToMI, BigToM, FanToM 벤치마크 실험 결과, TimeToM이 여러 시나리오에서 LLM의 ToM 문제에 대한 추론 성능을 크게 향상시켰음을 보여줍니다. 특히, 3차 ToM 질문에서도 좋은 성능을 보이며 높은 차수의 ToM 질문에 적합합니다.



### Needle in the Haystack for Memory Based Large Language Models (https://arxiv.org/abs/2407.01437)
Comments:
          5 pages

- **What's New**: 이 논문에서는 메모리 확장형 대형 언어 모델(LLM) 아키텍처가 매우 긴 문맥에서 정보의 회상 능력(recall abilities)을 향상시키는 데 미치는 이점을 실증합니다. 특히, 외부 연상 기억(external associative memory)을 포함하는 최신 LLM 아키텍처인 LARIMAR을 여러 긴 문맥 회상 작업(passkey 및 needle-in-the-haystack 테스트 등)에서 테스트합니다.

- **Technical Details**: LARIMAR은 LLM 디코더에 외부 연상 기억을 추가하는 구조로, 이 외부 메모리는 테스트 시간에 적응하여 훈련 중에 본 적 없는 긴 문맥도 처리할 수 있습니다. 메모리 읽기와 쓰기는 데이터를 저장하고 검색할 수 있도록 시스템을 구성하며, 이 과정을 CPU에서 실행하여 GPU 메모리 공간을 늘리지 않고도 확장할 수 있습니다. 모델의 훈련 시 컨텍스트 길이는 384 토큰(token)에 제한되었지만, 테스트 시에는 100K~1M 토큰의 문맥 길이에서도 강력한 성능을 유지할 수 있습니다.

- **Performance Highlights**: 다른 비슷한 파라미터 수를 가진 모델들과 비교할 때, LARIMAR는 특정 작업에 맞춘 훈련 없이도 긴 문맥 회상 작업에서 강력한 성능을 유지할 수 있습니다. 이 모델의 성능은 훈련된 문맥 길이를 초과하는 긴 문맥에서도 높은 일반화(generalization)를 보여주며, 이는 모델이 훈련된 디코더(decoder)에 맞춰진 메모리 읽기 출력을 유지하기 때문입니다.



### A Global-Local Attention Mechanism for Relation Classification (https://arxiv.org/abs/2407.01424)
Comments:
          This paper has been accepted by the 2024 20th International Conference on Natural Computation, Fuzzy Systems and Knowledge Discovery (ICNC-FSKD)

- **What's New**: 이 논문은 관계 분류(relation classification) 작업에서 전통적인 글로벌 주의 메커니즘(global attention mechanism)이 놓치기 쉬운 로컬 컨텍스트의 중요성을 강조합니다. 이를 해결하기 위해 글로벌-로컬 주의 메커니즘(global-local attention mechanism)을 도입하여 글로벌 주의에 로컬 기준을 추가합니다. 또한, 잠재적인 키워드를 식별하기 위해 하드 및 소프트 로컬화 메커니즘을 제안합니다. 이 접근법은 SemEval-2010 Task 8 데이터셋 실험 결과에서 기존 주의 방식보다 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 방법은 네 부분으로 구성됩니다: 1) 입력 표현(Input Representation): 단어는 단어 임베딩 및 위치 임베딩으로 변환됩니다. 2) 양방향 GRU 층(Bi-directional GRU Layer): 입력에서 고차원의 특징을 추출합니다. 3) 글로벌-로컬 주의 레이어(Global-Local attention Layer): 전통적인 글로벌 주의 메커니즘과 로컬 주의 메커니즘을 결합합니다. 하드 로컬화는 최단 의존 경로에 위치한 모든 단어를 잠재적 키워드로 간주하고, 소프트 로컬화는 이러한 경로를 감독 신호로 사용하여 더 견고한 키워드를 식별합니다. 4) 출력 층(Output Layer): 주의 가중치로 합산된 은닉 상태는 최종 분류 결과를 소프트맥스(softmax)로 제공합니다.

- **Performance Highlights**: SemEval-2010 Task 8 데이터셋을 사용한 실험 결과, 제안된 글로벌-로컬 주의 메커니즘이 기존 주의 방법보다 우수한 성능을 나타냅니다. 추가 분석은 글로벌-로컬 주의가 정확한 키워드에 효과적으로 집중하는 것을 보여줍니다.



### HyperLoader: Integrating Hypernetwork-Based LoRA and Adapter Layers into Multi-Task Transformers for Sequence Labelling (https://arxiv.org/abs/2407.01411)
- **What's New**: HyperLoader는 다양한 파라미터 효율적 미세조정(parameter-efficient fine-tuning) 방법들을 멀티태스킹 설정에서 결합한 새로운 접근 방식입니다. 하이퍼네트워크(hypernetwork)를 통해 모듈의 가중치를 태스크, 트랜스포머 계층, 그리고 이 계층 내 위치에 따라 생성합니다.

- **Technical Details**: HyperLoader는 어댑터(adapters) 및 LoRA(Low-Rank Adaptation)를 결합하여, 태스크의 시퀀스 레이블링(Sequence Labelling) 작업을 수행하는 데 사용된 T5 모델을 기반으로 합니다. 하이퍼네트워크는 다중 태스크에 대한 공통 구조를 학습하면서도 태스크 간 간섭 문제를 해결하기 위해 태스크 특화된 지식을 가중치에 캡슐화합니다.

- **Performance Highlights**: HyperLoader는 이전 접근 방식들보다 대부분 데이터셋에서 뛰어난 성능을 보였으며, 고자원(high-resource) 및 저자원(low-resource) 시나리오 모두에서 최고의 평균 성능을 달성했습니다. 특히, 전체 데이터셋과 검증 데이터셋의 10% 및 20%만을 사용한 저자원 설정에서도 우수한 성능을 입증했습니다.



### Dynamic Few-Shot Learning for Knowledge Graph Question Answering (https://arxiv.org/abs/2407.01409)
- **What's New**: 이번 연구에서는 지식 그래프 질의응답(KGQA)을 위해 동적 소규모 학습(Dynamic Few-Shot Learning; DFSL)이라는 새로운 접근 방식을 제안합니다. DFSL은 컨텍스트 학습(in-context learning)과 의미 유사성(semantic similarity)을 통합하여 최신 성능을 제공합니다. 여러 벤치마크 데이터셋과 아키텍처 설정을 통해 광범위한 평가를 수행했습니다.

- **Technical Details**: DFSL 방법론은 LLM(Large Language Models)을 활용한 컨텍스트 내 학습에 초점을 맞추고 있으며, 학습 세트에서 유사한 질문을 검색하여 이를 프롬프트에 포함시킵니다. 주요 지식 베이스인 DBpedia와 Wikidata를 기반으로 QALD-9, QALD-9 plus, QALD-10, LC-QuAD 2.0을 사용해 성능을 평가했습니다. Mixtral 8x7B, Llama-3 70B, CodeLlama 70B와 같은 최신 LLM들을 백본으로 활용했습니다.

- **Performance Highlights**: 실험 결과, DFSL 모델은 대부분의 벤치마크에서 새로운 최첨단 결과를 달성했으며, 속도와 효율성 측면에서 상당한 이점을 보였습니다. 또한, EL(Entity Linking) 및 RL(Relation Linking) 모듈로부터의 골드 정보를 사용하지 않고도 접근 방식의 유효성을 평가하기 위한 소거 연구를 수행했습니다.



### Adapting Multilingual LLMs to Low-Resource Languages with Knowledge Graphs via Adapters (https://arxiv.org/abs/2407.01406)
Comments:
          9 pages, KaLLM workshop

- **What's New**: 이 논문은 다국어 대형 언어 모델(Large Language Models, LLMs)에 어댑터(adapters)를 사용해 언어적 온톨로지(graph knowledge) 지식을 통합시키는 방식을 탐구합니다. 특히, Maltese, Bulgarian, Indonesian, Nepali, Javanese, Uyghur, Tibetan, Sinhala와 같은 자원 빈약 언어(Low-Resource Languages, LRLs)에서 감정 분석(Sentiment Analysis, SA) 및 명명 엔티티 인식(Named Entity Recognition, NER) 성능을 향상시키고자 합니다.

- **Technical Details**: 이 연구는 성공적인 파라미터 효율적 미세 조정(파인튜닝) 기법 K-ADAPTER와 MAD-X를 기반으로 하여 다국어 그래프들로부터 지식을 통합하는 접근 방식을 제안합니다. ConceptNet 데이터에서 추출한 자료들로 각 언어 특화 어댑터를 미세 조정하여 사전 학습된 다국어 LLM에 지식을 주입합니다. 이를 통해 언어 간 지식 전이 효과를 노릴 수 있습니다. 다양한 미세 조정 목적, 즉 표준 마스크드 언어 모델링(Masked Language Modeling, MLM), 전체 단어 마스킹을 포함한 MLM, 그리고 타겟 마스킹을 포함한 MLM 방법을 비교 분석합니다.

- **Performance Highlights**: 다국어 LLM 및 자원 빈약 언어의 SA 및 NER 작업에서 그래프 지식을 주입한 후 성능 평가가 이루어졌습니다. 실험 결과를 통해 구조화된 그래프 지식이 통합된 다국어 LLM이 자원 빈약 언어에서 어떻게 성능을 향상시킬 수 있는지에 대한 통찰을 제공합니다.



### POLygraph: Polish Fake News Datas (https://arxiv.org/abs/2407.01393)
Comments:
          14 pages, 1 figure, accepted to the 14th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA'24)

- **What's New**: 요약: 본 논문은 폴란드어로 된 가짜 뉴스 탐지를 위한 독특한 데이터셋인 POLygraph 데이터셋을 소개합니다. 이 데이터셋은 두 부분으로 구성되며, 'fake-or-not' 데이터셋에는 11,360개의 뉴스 기사 쌍과 해당 라벨이 포함되어 있고, 'fake-they-say' 데이터셋에는 5,082개의 뉴스 기사와 그것에 대한 트윗이 포함되어 있습니다. 데이터는 전문가와 비전문가 주석자들에 의해 수집되었으며, 고급 기계 학습 기법을 활용하여 콘텐츠의 진위 여부를 분석하는 소프트웨어 도구도 개발되었습니다. 이 소프트웨어와 데이터셋은 공공 부문 기관부터 출판사 및 팩트체킹 조직에 이르기까지 다양한 주체들에게 도움을 줄 것으로 기대됩니다.

- **Technical Details**: 기술적 세부사항: POLygraph 데이터셋은 'fake-or-not'와 'fake-they-say' 두 부분으로 구성됩니다. 'fake-or-not' 데이터셋에는 11,360쌍의 뉴스 기사와 가짜 뉴스 여부를 나타내는 라벨이 포함되어 있으며, 'fake-they-say' 데이터셋에는 5,082개의 뉴스 기사와 해당 기사에 대한 트윗이 포함되어 있습니다. 각 트윗에는 기사의 진위 여부에 대한 댓글자의 의견이 라벨로 붙어 있습니다. 데이터 수집은 전문가뿐만 아니라 비전문가에 의한 수작업 주석을 통해 이루어졌으며, 트위터 API와 웹 스크래핑을 통해 데이터를 수집했습니다.

- **Performance Highlights**: 성과 하이라이트: 본 프로젝트는 내용의 진위 여부를 판별하기 위한 고급 기계 학습 기법을 사용하여 데이터셋을 분석하는 소프트웨어 도구를 개발했습니다. 이 소프트웨어와 데이터셋은 공공 안전을 위해 내부 보안 기관 및 경찰, 출판사, 화폐감독위원회 등의 다양한 주체들에 유용할 것으로 기대됩니다. 이 데이터셋은 폴란드 맥락에서 허위 정보 감지 및 분석을 위한 미래 연구의 기초를 제공할 것입니다.



### Free-text Rationale Generation under Readability Level Contro (https://arxiv.org/abs/2407.01384)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 자연어 설명(Natural Language Explanation, NLE) 생성에서 주어진 가독성(readability)을 어떻게 수행하는지 조사합니다. 특히, 설명이 특정 전문성 단계(예: 중학교, 대학교)를 목표로 할 때 모델의 성능을 평가했습니다.

- **Technical Details**: LLM의 자유 텍스트 근거(free-text rationale) 생성에서 가독성 조절을 통해 모델의 설명력을 평가했습니다. 가독성 평가는 Flesch Reading Ease(FRE), Gunning-Fog Index(GFI), Coleman-Liau Index(CLI)와 같은 전통적인 읽기 쉬운 지표를 사용하여 측정되었습니다. 두 가지 NLP 작업인 증오 발언 탐지(hate speech detection) 및 자연어 추론(NLI)을 대상으로 HateXplain, CAD, SpanEx 데이터셋을 사용했습니다.

- **Performance Highlights**: 연구 결과, LLM의 설명은 지시된 가독성 수준에 맞게 조절 가능하지만, 실제로 측정된 텍스트 복잡도와 요청된 가독성 수준이 일치하지 않는 경우가 많았습니다. 또한, 모든 가독성 수준에서 인간 평가자는 대체로 만족스러운 설명을 받았으며, 고등학교 수준의 가독성이 가장 선호되는 것으로 나타났습니다.



### Bridging the Gap: Transfer Learning from English PLMs to Malaysian English (https://arxiv.org/abs/2407.01374)
Comments:
          Accepted in 9th Workshop on Representation Learning for NLP (Rep4NLP) at ACL 2024

- **What's New**: 말레이시아 영어는 말레이어, 중국어, 타밀어의 요소를 포함하면서도 독특한 형태론적 및 의미론적 특징을 가진 저자원 크리올 언어입니다. 본 논문에서는 말레이시아 영어의 명명된 개체 인식(NER)과 관계 추출(RE) 작업을 위해 특화된 MENmBERT와 MENBERT라는 사전학습 언어 모델을 소개합니다. 이 모델들은 말레이시아 영어 뉴스 기사(MEN) 데이터셋을 활용해 미세 조정되어, 기존 모델들보다 뛰어난 성능을 보입니다.

- **Technical Details**: MENmBERT와 MENBERT는 말레이시아 영어 뉴스 텍스트에서 수동으로 주석된 개체 및 관계를 사용하여 미세 조정되었으며, 말레이시아 영어의 뉘앙스를 효과적으로 캡처합니다. MENmBERT는 NER 작업에서 1.52%, RE 작업에서 26.27%의 성능 향상을 이루었으며, 이는 bert-base-multilingual-cased 모델과 비교됩니다. 전체적인 NER 성능 향상은 미미하지만, 12개 개체 레이블로 평가할 때 유의미한 향상이 나타났습니다.

- **Performance Highlights**: MENmBERT는 NER 작업에서 1.52%, RE 작업에서 26.27%의 성능 향상을 기록하였습니다. 추가 분석 결과, 12개 개체 레이블 기준 평가 시 큰 성능 향상이 확인되었습니다. 이는 언어 특화 및 지리적 초점의 코퍼스에서 사전 학습된 언어 모델이 저자원 환경에서 NER 성능 개선에 유망한 접근법임을 시사합니다.



### Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems (https://arxiv.org/abs/2407.01370)
- **What's New**: 최신 대형 언어 모델(LLMs)와 Retrieval Augmented Generation(RAG) 시스템은 이제 수백만 개의 입력 토큰을 처리할 수 있습니다. 하지만 이러한 시스템의 긴 문맥 작업에서의 출력 품질 평가가 여전히 도전적입니다. 본 연구는 요약 작업이 이러한 평가에서 중요한 역할을 할 수 있다고 제안합니다. 문서 묶음(Haystack)을 합성하는 절차를 설계하였고, 'Summary of a Haystack'(SummHay) 작업을 통해 시스템이 문서를 처리하고 쿼리에 따라 관련된 통찰(insight)을 요약하도록 요구합니다. 자동 평가 방법을 도입하여 요약의 'Coverage'와 'Citation' 두 측면에서 점수를 매깁니다.

- **Technical Details**: SummHay 작업은 대화 및 뉴스 도메인에서 문서 묶음을 생성하는 데이터 합성 프로그램을 사용합니다. 각 문서 묶음은 약 100개의 문서를 포함하고, 관련된 통찰이 반복되도록 설계됩니다. 시스템은 주어진 쿼리에 따라 문서 묶음을 요약하고, 각 통찰의 출처 문서를 정확히 인용해야 합니다. 평가 프로토콜은 요약의 참조 통찰 커버리지 및 인용 품질을 중심으로 합니다. 대규모 평가에서 10개의 LLM과 50개의 RAG 시스템을 테스트했습니다.

- **Performance Highlights**: SummHay는 현재의 시스템에게 여전히 도전적인 과제로 남아 있으며, 인간 성능의 56%에 비해 10점 이상 낮습니다. Oracle 신호를 제공해도 성능 차이가 컸습니다. GPT-4o와 Claude 3 Opus 같은 장문맥 LLM은 SummHay에서 20% 미만의 점수를 기록했습니다. RAG 시스템은 인용 품질에서 향상되었으나 통찰 커버리지는 낮았습니다. SummHay는 장문맥 모델의 위치 편향(Position Bias)을 연구하는 데도 유용하게 사용할 수 있습니다. 우리는 데이터셋과 평가 방법론을 오픈 소스화 하였습니다.



### Nullpointer at ArAIEval Shared Task: Arabic Propagandist Technique Detection with Token-to-Word Mapping in Sequence Tagging (https://arxiv.org/abs/2407.01360)
Comments:
          To appear in proceedings of 2024 Arabic NLP Conference

- **What's New**: 이 논문은 아라비아어 텍스트, 특히 트윗과 뉴스 문단에서 선전 기법 탐지를 최적화하는 방법을 연구합니다. 우리의 접근 방식은 AraBERT v2 모델을 미세 조정하고 시퀀스 태그를 위한 신경망 분류기를 사용하는 것입니다. 실험 결과, 단어의 첫 번째 토큰을 활용한 기법 예측이 가장 우수한 성능을 제공하였습니다. 또한, 장르 정보를 특징으로 도입하면 모델 성능이 더 향상됩니다. 우리의 시스템은 25.41 점을 얻어 리더보드에서 4위를 차지했으며, 제출 후 개선으로 점수가 26.68로 상승했습니다.

- **Technical Details**: 우리의 주요 기여는 아라비아어 텍스트에서 선전 기법을 탐지하기 위해 AraBERT를 활용한 최적화 시스템을 개발한 것입니다. 데이터 전처리는 유니코드 불일치, 잘못 맞춰진 주석 및 사용자 멘션 정규화를 다루는 강력한 파이프라인을 제안합니다. 시퀀스 태깅을 위한 다양한 접근 방식을 평가하고 비교했습니다. 우리 시스템은 AraBERT v2 모델을 활용하며, [CLS] 토큰과 개별 토큰의 컨텍스트 임베딩을 결합하여 각 토큰에 대한 레이블을 예측합니다. 또한 트윗과 뉴스 문단이라는 두 가지 장르 정보를 원-핫 벡터로 인코딩하여 추가 기능으로 사용했습니다.

- **Performance Highlights**: 단어의 첫 번째 토큰을 활용한 시퀀스 태깅 기법이 가장 효과적이었음을 확인했습니다. 이 접근 방식은 아라비아어의 특성과 일치하며, 단어의 본질적 의미가 초기 토큰에 의해 포착됩니다. [CLS] 토큰을 추가하여 전체 입력 시퀀스를 대표함으로써 모델의 성능을 조금 더 향상시킬 수 있었습니다. 최종 모델은 최적의 하이퍼파라미터 튜닝을 통해 훈련되었으며, 테스트 데이터셋에서 선전 기법의 스팬을 예측하여 우수한 결과를 제공했습니다.



### Evaluating Knowledge-based Cross-lingual Inconsistency in Large Language Models (https://arxiv.org/abs/2407.01358)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 관찰되는 다국어 불일치를 조사합니다. ChatGPT, Llama, Baichuan과 같은 모델이 뛰어난 성능을 보이지만, 동일한 개념을 다른 언어로 처리할 때 상당한 불일치를 보입니다. 이 연구는 LLMs에서 다국어 불일치가 존재하는지, 어떤 측면에서 이러한 불일치가 나타나는지, 그리고 다국어 일관성과 다국어 능력 간의 상관관계가 있는지를 중점적으로 탐구합니다.

- **Technical Details**: 이 논문은 LaBSE 모델을 사용하여 Cross-lingual Semantic Consistency (xSC)를 평가하는 혁신적인 방법을 제안합니다. 또한 Cross-lingual Accuracy Consistency (xAC)와 Cross-lingual Timeliness Consistency (xTC)라는 메트릭을 도입하여 의미, 정확성, 시기적 일관성의 불일치를 포괄적으로 평가합니다. 이 메트릭들을 조화시켜 LLMs의 다국어 일관성을 종합적으로 측정합니다.

- **Performance Highlights**: 논문의 주요 발견은 다국어 일관성의 이해와 개선에 기여하며, 더 견고하고 신뢰할 수 있는 다국어 언어 모델 개발에 기여하는 것을 목적으로 합니다. 예를 들어, 같은 질문을 영어와 스페인어로 물었을 때는 올바른 '파리 생제르맹 클럽(PSG)'라는 답변을 하였지만, 중국어와 일본어로 물었을 때는 'FC 바르셀로나'라는 잘못된 답변을 하였습니다. 이와 같은 불일치는 사실적 지식 조회 외에도 감정 분석, 명명된 실체 인식, 의미 이해 등 여러 측면에서도 발생할 수 있습니다.



### Protecting Privacy in Classifiers by Token Manipulation (https://arxiv.org/abs/2407.01334)
- **What's New**: 오늘 다루는 arXiv 논문은 개인 정보 노출 문제를 해결하기 위한 텍스트 조작 기술을 연구합니다. 이 논문에서는 텍스트 분류 모델에서 사용될 수 있는 여러 토큰 매핑(token mapping) 및 문맥 정보 활용 방법을 살펴봅니다. 이 연구는 텍스트 분류 모델의 정확성을 유지하면서 강조할 수 있으며, 원본 텍스트를 복구할 수 없도록 만드는 방법을 탐구합니다.

- **Technical Details**: 연구는 두 가지 주요 접근 방식을 제안합니다. 첫 번째는 단순한 규칙에 기반한 토큰 대체입니다. 예를 들어, 토큰을 쌍(pair) 또는 삼중(triplet)으로 묶어 하나의 대표 토큰으로 매핑합니다. 이 방법은 쉽게 구현할 수 있지만, 숙련된 공격자가 원본 텍스트를 복구하기 어렵습니다. 두 번째 접근 방식은 문맥 정보를 활용하여 전략적으로 토큰을 교체하는 것입니다. 이를 통해 모델의 성능 저하를 최소화하면서도 개인 정보를 보호할 수 있습니다.

- **Performance Highlights**: 실험 결과, 단순 규칙 기반의 토큰 대체 방법은 공격자가 원본 텍스트를 쉽게 복구할 수 있다는 단점이 있지만, 문맥 정보를 활용한 토큰 대체 방법은 모델의 성능을 크게 저하시키지 않으면서도 원본 텍스트를 잘 보호할 수 있음을 보여줍니다. 이 연구는 기존 접근 방식과 달리 LLM의 파라미터에 접근하지 않고도 개인 정보를 보호할 수 있는 새로운 방법을 제안합니다.



### Language Portability Strategies for Open-domain Dialogue with Pre-trained Language Models from High to Low Resource Languages (https://arxiv.org/abs/2407.01315)
Comments:
          The 13th International Workshop on Spoken Dialogue Systems Technology (IWSDS '23)

- **What's New**: 이번 연구에서는 대규모 사전 학습된 언어 모델(PLMs)의 언어 포팅 전략을 고자원 언어에서 저자원 언어로 전환하는 방법을 제안합니다. 특히, 프랑스어를 저자원 언어로 시뮬레이션하여 영어에서 프랑스어로 PLMs를 이전하는 방법을 평가합니다. 이 연구는 Neural Machine Translation(NMT)을 사용한 두 가지 접근법과 BLOOM이라는 다국어 PLM을 사용한 새로운 접근법을 평가합니다. 이러한 다양한 접근법에서 얻은 모델들의 성능을 사람을 대상으로 한 대화 시스템에서 평가하였습니다.

- **Technical Details**: 연구는 세 가지 주요 접근법을 사용합니다. 첫 번째는 L_S 데이터셋을 번역하여 L_T에서 미세 조정(train-on-target)하는 방법입니다. 두 번째는 추론 중에 L_S 모델과 NMT 모듈을 결합하여 L_T 입력에 사용하는(test-on-source) 방법입니다. 세 번째는 BLOOM 모델과 MAD-X Adapter 아키텍처를 사용하여 소스 언어에서 태스크를 학습하고 타겟 언어에 맞게 적응시키는 방법입니다. 이를 통해 L_S와 L_T 각 언어에서의 다양한 자료와 모델을 최대한 활용하고 있습니다.

- **Performance Highlights**: 실험 결과, 세 가지 접근법 모두에서 사람이 인식하는 상호작용 품질의 차이를 비교할 수 있습니다. 특히, L_S에서 미세 조정된 BlenderBot 1.0이 기준 모델로 사용되며, BLOOM 모델을 사용한 접근법은 다국어와 번역 능력을 효과적으로 활용하는 방법으로 주목받았습니다. 또한, NMT를 활용한 방법은 기존 자원을 활용함으로써 저비용으로 언어 포팅이 가능함을 보여주었습니다.



### Collaborative Performance Prediction for Large Language Models (https://arxiv.org/abs/2407.01300)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 다양한 다운스트림 과제에서의 성능을 예측하는 새로운 프레임워크, 협력 성능 예측(Collaborative Performance Prediction, CPP)을 소개합니다. 기존의 스케일링 법칙(scaling law)이 모델 군(family) 내의 유사성만을 고려하는 반면, CPP는 여러 모델 군 간의 성능 기록과 다양한 설계 요소를 활용하여 예측 정확성을 크게 향상시킵니다. 이를 위해 온라인 플랫폼에서 수집한 협력 데이터(collaborative data)를 사용합니다.

- **Technical Details**: CPP 프레임워크는 LLM와 과제의 잠재적 표현(대표적으로 내적(Inner Product))을 학습하여 예측 모델을 구축합니다. 아카데믹 페이퍼, 기술 보고서, 공개 리더보드에서 수집한 72개 모델과 29개 과제의 데이터를 활용합니다. 이 프레임워크는 학습 비용을 낮추고, 특정 모델이나 과제에 한정되지 않는 예측을 가능하게 합니다. 또한, 분석 방법으로 샤플리 값(Shapley Values)을 이용하여 다양한 요인들의 중요성을 해석할 수 있습니다.

- **Performance Highlights**: CPP는 예측 성능 면에서 기존의 scaling laws을 능가하며, 특히 소형 모델 정보만으로 대형 모델의 성능을 정확하게 예측할 수 있습니다. 예를 들어, HELM 데이터셋에서 50%의 점수를 사용해 나머지 50%를 예측했을 때, 예측 순위의 정확도는 10%, MAE@2(Mean Absolute Error @2)는 39%를 달성했습니다. 또한, 수집된 데이터셋에서는 정확도 45%, MAE@2는 84%를 기록했습니다.



### Show Less, Instruct More: Enriching Prompts with Definitions and Guidelines for Zero-Shot NER (https://arxiv.org/abs/2407.01272)
- **What's New**: 최근의 몇 가지 명령어로 미세조정된 Named Entity Recognition (NER) 용 대형 언어 모델(LLM)은 기존 NER 접근 방식에 비해 우수한 일반화 능력을 보여줍니다. 본 연구에서는 기존의 단순한 zero-shot NER 모델에서 나아가 SLIMER를 제안합니다. SLIMER는 모델을 '정의'와 '지침'이 포함된 프롬프트로 풍부하게 하여, 제한된 예제 수로 새로운 엔티티 태그를 처리할 수 있도록 합니다.

- **Technical Details**: SLIMER는 '쇼 레스, 인스트럭트 모어 - 엔티티 인식(Show Less, Instruct More - Entity Recognition)'의 약자로, 보다 자연스러운 명령어와 관련 정의 및 지침을 포함한 프롬프트를 사용합니다. 다른 대형 언어 모델을 통해 자동으로 생성된 정의와 가이드를 활용하여 게획을 단순하게 유지하면서도 강력한 성능을 발휘합니다. SLIMER는 두 개의 표준 NER 벤치마크, MIT와 CrossNER에서 실험을 진행했으며, BUSTER 데이터셋에서 보이지 않았던 엔티티 태그로 성능을 평가했습니다.

- **Performance Highlights**: SLIMER는 정의와 지침이 없는 기본 모델과 비교하여 더 깊은 이해, 더 빠르고 안정적인 학습, 더 나은 zero-shot 성능을 보입니다. 특히 제한된 데이터와 낮은 트레이닝-테스트 태그 오버랩에도 불구하고, SLIMER는 첨단 명령어로 미세조정된 모델들에 견줄 만한 성능을 보여주었습니다.



### First Place Solution of 2023 Global Artificial Intelligence Technology Innovation Competition Track 1 (https://arxiv.org/abs/2407.01271)
Comments:
          First Place of 2023 Global Artificial Intelligence Technology Innovation Competition

- **What's New**: 이 논문에서는 Global Artificial Intelligence Technology Innovation Competition의 Track 1에서 우승한 의료 영상 진단 보고서 생성 솔루션을 제시합니다. 텍스트 생성 작업을 위해 CPT-BASE를 기본 모델로 선택하고, 사전 학습 단계에서 Masked Language Modeling 작업을 제거하고 대신 span mask 전략을 채택하여 디노이징 오토인코더(Denoising Auto-Encoder) 사전 학습 작업을 수행합니다. 미세 조정 단계에서는 반복 검색 증강과 소음 인식 유사성 버킷 프롬프트 전략을 사용합니다.

- **Technical Details**: 사전 학습 단계에서는 CPT-BASE의 Masked Language Modeling 작업을 삭제하고, span mask 전략을 사용해 점차적으로 마스킹 비율을 증가시키는 디노이징 오토인코더(Denoising Auto-Encoder) 사전 학습 작업을 수행합니다. 미세 조정 단계에서는 임베딩을 사용하여 유사한 설명-진단 쌍을 검색하여 미니 지식 기반을 구성하고, 소음 인식 유사성 버킷 프롬프트 전략을 통해 모델이 더 높은 품질의 진단 보고서를 생성하도록 유도합니다.

- **Performance Highlights**: 단일 모델의 경우 leaderboard A에서 2.321 점을 기록하며, 다중 모델 융합 점수는 각각 A와 B 리더보드에서 2.362와 2.320을 기록하여 1위를 차지했습니다.



### The African Woman is Rhythmic and Soulful: Evaluation of Open-ended Generation for Implicit Biases (https://arxiv.org/abs/2407.01270)
- **What's New**: 이 연구는 명시적인 편향 테스트를 통과한 후에도 대형 언어 모델(LLMs)이 여전히 암묵적인 편향을 나타낼 수 있다는 점을 조사했습니다. 심리학적 방법론에서 영감을 받은 혁신적인 편향 측정 방법을 소개하여 이러한 문제를 해결하려고 합니다. 이 연구는 특히 소유권이 강화되어 기존 편향 측정 방법을 적용하기 어려운 LLMs에 대한 접근성을 제한하는 문제를 다루기 위해 개발되었습니다.

- **Technical Details**: LLM 암묵 연관 테스트(IAT) 편향(Implicit Association Test Bias)과 LLM 결정 편향(Decision Bias)이라는 두 가지 측정 방법이 소개되었습니다. LLM IAT 편향은 심리학에서 잘 알려진 IAT를 LLM에 맞게 도입한 프롬프트 기반 방법이며, LLM 결정 편향은 다양한 시나리오에서 LLM이 개인을 선택하는 방식을 통해 미묘한 차별을 감지하는 것을 목표로 합니다. 또한, 주제 분석(thematic analysis)을 통해 워드 생성(word generation)과 스토리텔링(storytelling)의 편향을 조사했습니다.

- **Performance Highlights**: 실험 결과 성별 및 인종 도메인에서 차별적 분류와 이국화(Exoticisation) 등의 편향이 드러났습니다. 암시적 편향의 프롬프트 기반 측정 방법이 전통적인 임베딩(embedding) 기반 방법과 상관성이 있을 뿐만 아니라, LLM 결정 편향이 측정하는 후속 행동을 예측하는 데 더욱 효과적이라는 결과를 발견했습니다. 이는 상대적인 평가(relative evaluations)의 중요성을 강조하며 암묵적인 편향 평가에서 인간 심리학적 통찰과 상통하는 결과입니다. 이러한 연구 결과는 AI 윤리에 대한 이해를 넓히고, 고도화된 AI 시스템의 편향을 지속적으로 평가하고 완화하기 위한 제안을 제공합니다.



### SignCLIP: Connecting Text and Sign Language by Contrastive Learning (https://arxiv.org/abs/2407.01264)
- **What's New**: SignCLIP는 CLIP(Contrastive Language-Image Pretraining)를 재활용하여 음성 언어 텍스트와 수어 비디오, 즉 서로 다른 모달리티의 자연 언어 두 클래스를 동일한 공간에 투영하는 새로운 방법입니다. 이 모델은 특정 작업이나 수어에 직접 최적화하지 않고도 대규모 다국어 비디오-텍스트 쌍으로부터 유용한 시각적 표현을 학습합니다.

- **Technical Details**: SignCLIP는 Spreadthesign이라는 약 50만 개의 비디오 클립과 최대 44개의 수어를 포함하는 주요 수어 사전에서 사전 학습을 수행합니다. 이 모델은 다양한 다운스트림 데이터셋으로 평가하며, 손가락 철자법 등에서의 읽기 능력을 입증합니다. 또한, 대규모 손가락 철자 수어 데이터를 사용하여 검증된 접근 방식을 기반으로 종합적인 사전 학습, 미세 조정 및 평가를 수행합니다.

- **Performance Highlights**: SignCLIP는 텍스트-비디오 및 비디오-텍스트 검색 정확도에서 뛰어난 성능을 보였으며, 필수적인 몇 샷 프롬프트 또는 미세 조정을 통해 고립된 수어 인식과 같은 도메인 외부 다운스트림 작업에서도 경쟁력을 가지고 있습니다. 또한, 이 모델은 음성 언어 텍스트와 수어 동작 간의 잠재 공간을 분석하여 추가적인 언어적 통찰을 제공합니다.



### uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation via Large-Scale Pseudo Labelling (https://arxiv.org/abs/2407.01257)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 Whisper 모델의 지식을 소형 모델로 증류(distilling)하기 위한 새로운 비지도 학습(unsupervised learning) 프레임워크를 제안합니다. 기존 방법에서는 높은 품질의 예측을 필터링하기 위해 지도 학습(supervised learning)이 필요했으나, 이 방법은 라벨이 없는 데이터로도 성능을 발휘할 수 있습니다. 실험 결과, 제안된 모델은 Word Error Rate(WER)에서 교사 모델(teacher model)보다 5-7점 더 뛰어나며, 연산 및 메모리 효율성에서도 25-50% 개선된 성능을 보입니다.

- **Technical Details**: 기존의 지식 증류(knowledge distillation) 과정에서는 높은 품질의 가짜 라벨(pseudo-label)을 필터링하기 위해 실제 라벨 데이터를 필요로 했습니다. 이 논문에서는 라벨이 없는 완전 비지도 학습 방식으로 이 문제를 해결합니다. 구체적으로, 다양한 비지도 방법을 통해 낮은 품질의 가짜 라벨을 필터링하여 라벨이 필요 없는 지식 증류를 구현합니다. 이 방법은 다양한 실험 환경에서 테스트되었으며, SADA 데이터셋에서도 평가가 진행되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 비지도 학습 프레임워크를 통한 증류 모델은 교사 모델보다 WER 측면에서 5-7점 더 높은 성능을 보여주었습니다. 또한, 데이터 규모를 확장했을 때 제로샷(zero-shot) 및 다른 지도 학습 기반 모델보다도 우수한 성능을 기록하였습니다. 제안된 모델은 연산 및 메모리 효율성이 교사 모델 대비 25-50% 더 뛰어나면서도 동일하거나 더 나은 성능을 유지합니다.



### MIRAI: Evaluating LLM Agents for Event Forecasting (https://arxiv.org/abs/2407.01231)
Comments:
          66 pages, 8 figures, 6 tables; Website: this https URL

- **What's New**: LLM (Large Language Models)의 잠재력을 국제 이벤트 예측에 적용하기 위한 새로운 벤치마크인 MIRAI를 도입했습니다. 이는 LLM 에이전트가 단기에서 장기 예측에 이르기까지 다양한 예측 수평선을 통해 국제 이벤트의 시간적 예측 능력을 체계적으로 평가할 수 있는 환경을 제공합니다.

- **Technical Details**: MIRAI 벤치마크는 GDELT(Global Database of Events, Language, and Tone) 이벤트 데이터베이스를 정제 및 분석하여 다양한 관계 예측 작업을 큐레이팅합니다. 에이전트가 다양한 도구(API 포함)를 사용할 수 있도록 코드 기반의 인터페이스를 구현했으며, 에이전트가 대규모 글로벌 데이터베이스에서 중요한 정보를 자율적으로 수집하고 통합하며, 도메인 특화된 API 및 라이브러리를 사용하여 도구를 활용할 수 있도록 환경을 구축했습니다. 예측 작업은 '전문 진술'과 '만나려는 의도를 표현'하는 것과 같은 20개의 폭넓은 카테고리와 하위 카테고리로 분류됩니다.

- **Performance Highlights**: 최근 진행된 실험에서, 가장 성능이 우수했던 GPT-4o 에이전트가 API 전체 세트를 이용하여 두 번째 수준의 관계 예측 작업에서 29.6의 F1 점수를 기록했습니다. '코드 블록' 도구 사용 전략은 더 유연한 상호작용을 가능하게 하지만, 견고한 코드 생성 능력을 요구합니다. GPT-4o 모델만이 이를 효과적으로 활용하여 이점을 얻었습니다. 이러한 결과는 LLM 에이전트의 시간적 추론과 도구 사용의 효과성에 대한 지속적인 연구의 필요성을 강조합니다.



### Searching for Best Practices in Retrieval-Augmented Generation (https://arxiv.org/abs/2407.01219)
- **What's New**: 이 논문은 최신 정보를 통합하고 환각(hallucinations)을 줄이며, 반응의 품질을 향상시키기 위한 Retrieval-augmented generation (RAG) 기술을 최적화하는 방법을 조사합니다. 기존의 RAG 접근 방식은 구현이 복잡하고 응답 시간이 길다는 단점이 있습니다. 이에 따라 다양한 RAG 접근 방식과 조합을 실험하여 성능과 효율성의 균형을 맞출 수 있는 최적의 RAG 전략을 제안합니다. 또한, 멀티모달(multimodal) 검색 기술이 시각적 입력(question-answering)과 콘텐츠 생성 측면에서 성능을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 전형적인 RAG 워크플로우는 여러 처리 단계를 포함합니다: 쿼리 분류, 검색, 재랭킹(reranking), 재포장(repacking), 요약(summarization) 등. 각 단계마다 문서를 조각으로 나누고, 이 조각들을 의미적으로 표현하기 위해 임베딩(embedding)을 사용하는 방법, 벡터 데이터베이스(vector databases)의 선택, 대형 언어 모델(LLMs)의 파인튜닝 방법 등을 결정해야 합니다. 제안된 접근 방식은 각 모듈의 최고 성능을 보이는 세 가지 방법을 선택하고, 이 방법들이 전체 RAG 성능에 미치는 영향을 평가한 다음, 최적의 조합을 찾아내어 성능과 효율성을 균형 있게 맞추는 것입니다.

- **Performance Highlights**: 실험을 통해 멀티모달 검색 기술이 시각적 입력에 대한 질문 답변 능력을 크게 향상시키고, 'retrieval as generation' 전략을 통해 멀티모달 콘텐츠 생성 속도를 가속화할 수 있음을 입증했습니다. 이는 특정 조직이나 도메인에 신속하게 애플리케이션을 배포할 수 있게 하며, 모델 파라미터를 업데이트할 필요 없이 쿼리 관련 문서만 제공하면 됩니다.



### EconNLI: Evaluating Large Language Models on Economics Reasoning (https://arxiv.org/abs/2407.01212)
Comments:
          Findings of ACL 2024

- **What's New**: 경제 분석 보고서 작성 및 금융 자문 제공에서 광범위하게 사용되는 대형 언어 모델(LLMs)의 경제적 지식 이해와 특정 경제 행사 결과에 대한 추론 능력을 체계적으로 평가한 첫 번째 연구가 제안되었습니다. 이를 위해 자연어 추론(NLI) 기반으로 경제적 이벤트(EconNLI)를 평가할 수 있는 새로운 데이터셋과 평가 방법이 도입되었습니다.

- **Technical Details**: 이 연구는 경제적 이벤트를 평가하기 위한 데이터셋 EconNLI를 만들어 제시했습니다. EconNLI는 주어진 전제 이벤트가 가설 이벤트를 유발하는지 여부를 분류하는 능력과 전제로부터 합리적인 결과를 생성하는 능력을 평가합니다. 실험 결과 경제적 추론에 있어서 LLMs는 아직 미흡한 점이 많으며 종종 잘못된 답변이나 'hallucinated' 답을 생성할 수 있음을 발견했습니다.

- **Performance Highlights**: 여러 언어 모델(오픈 소스부터 상용 제품까지)을 대상으로 광범위한 실험을 진행한 결과, 경제적 추론에 대한 LLMs의 성능은 미흡했습니다. 분류 작업에서는 학습 데이터 없이 오픈 소스 LLMs이 랜덤 추론 수준에 머물렀고, ChatGPT 및 GPT-4와 같은 고급 상용 모델도 만족스러운 성능을 보이지 못했습니다. 생성 작업에서는 모델들이 잘못된 답을 만들거나 'hallucinated' 답을 생성하는 경우가 있었습니다.



### $\text{Memory}^3$: Language Modeling with Explicit Memory (https://arxiv.org/abs/2407.01178)
- **What's New**: 이 논문에서는 인간 두뇌의 메모리 계층에서 영감을 받아 대형 언어 모델(LLMs)의 훈련 및 추론 비용을 줄이기 위한 새로운 방법을 제안합니다. LLM을 명시적 메모리(explicit memory)로 장착함으로써 모델 파라미터나 텍스트 검색 강화 생성(RAG)보다 저렴한 메모리 형식을 제공합니다. 이를 통해 모델의 파라미터 크기, 훈련 비용 및 추론 비용을 줄일 수 있습니다. 이 접근 방식으로 2.4B 크기의 LLM을 처음부터 훈련하였으며, 이 모델은 성능 면에서 더 큰 LLM이나 RAG 모델보다 우수하며, 디코딩 속도도 더 빠릅니다.

- **Technical Details**: 제안된 메모리 형식은 명시적 메모리(explicit memory)로, 쓰기 비용과 읽기 비용이 중간 정도인 메모리 형식입니다. 모델은 지식 베이스 또는 텍스트 데이터셋을 명시적 메모리로 변환하고, 추론 중에는 이 메모리를 회상하여 셀프 어텐션 계층에 통합합니다. 이 모델은 대부분의 기존 Transformer 기반 LLM이 약간의 미세 조정으로 명시적 메모리를 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 메모리^3 모델은 더 큰 LLM이나 RAG 모델보다 우수한 성능을 보여주었으며, RAG 모델보다 디코딩 속도가 빠릅니다. 명시적 메모리 메커니즘을 통해 모델 파라미터에 저장해야 할 지식을 외부화하고, 더 가벼운 백본(backbone)을 사용할 수 있게 하여 전반적인 비용을 줄일 수 있었습니다. 예를 들어, 특정 지식을 명시적 메모리로 저장하면 비용 효율성이 크게 향상됩니다.



### Learning to Explore and Select for Coverage-Conditioned Retrieval-Augmented Generation (https://arxiv.org/abs/2407.01158)
Comments:
          Work in progress. Resources are available at this https URL

- **What's New**: 이번 연구에서는 거대 언어 모델(LLM)의 상호작용에서 사용자가 특정 범위의 정보를 요청하는 'coverage-conditioned (C2) 시나리오'에 초점을 맞춥니다. 이를 위해 QTree라는 10,000개의 정보 탐색 쿼리 세트를 구성하여 다양한 관점에서 쿼리를 세분화했습니다. 그리고 QPlanner를 도입하여 C2 쿼리를 따르는 맞춤형 쿼리 개요를 생성합니다.

- **Technical Details**: QTree는 특정 주제에 대한 정보를 다양한 관점에서 세분화한 계층적 쿼리 세트로 구성되었습니다. QPlanner는 70억 파라미터의 autoregressive 언어 모델로, QTree를 기반으로 C2 쿼리에 맞춤형 개요(outlines)를 생성합니다. 실험 결과, QPlanner가 사용자 요구에 맞춘 개요를 제공할 수 있음을 자동 및 인간 평가를 통해 확인했습니다.

- **Performance Highlights**: QPlanner는 alignment training을 통해 C2 쿼리와의 일치를 보장하면서 사용자 관심사를 충족시키는 개요를 생성할 수 있습니다. 이를 통해 최종적으로 retrieval-augmented generation (RAG) 작업에서 성능 향상을 달성했습니다.



### Sociocultural Considerations in Monitoring Anti-LGBTQ+ Content on Social Media (https://arxiv.org/abs/2407.01149)
Comments:
          Accepted Manuscript ACL 2024 Workshop C3NLP

- **What's New**: 이 논문은 반-LGBTQ+ 콘텐츠 감지를 위한 혐오 발언 탐지 시스템의 개발에 있어서 사회적, 문화적, 정치적 요인의 영향을 연구했습니다. 오픈 소스 학습 데이터를 사용하여 다양한 영어 국가 변형에서 반-LGBTQ+ 콘텐츠 수준을 모니터링하는 데 적합성을 조사했습니다. 연구 결과, 오픈 소스 혐오 발언 데이터 세트의 사회적 및 문화적 일치가 예측 출력에 영향을 미친다는 것을 발견했습니다. 또한, 반-LGBTQ+ 욕설에 대한 키워드 검색 접근 방식이 탐지 모델을 지나치게 특화시키는 경향이 있어 반-LGBTQ+ 콘텐츠가 감지되지 않을 수 있습니다.

- **Technical Details**: 혐오 발언 탐지는 텍스트 분류 작업으로 주로 취급되며, 기존 데이터를 사용하여 머신 러닝 모델을 훈련시킵니다. 일반적인 파이프라인은 '데이터 세트 수집 및 준비', '특징 공학(Feature Engineering)', '모델 훈련', '모델 평가'로 구성됩니다. 데이터 수집 및 준비 단계에서 키워드 검색 접근 방식은 특정 욕설과 비속어를 사용하여 혐오 발언을 식별합니다. 하지만 이렇게 수집된 데이터는 성별, 인종, 문화적 편향을 포함할 수 있습니다.

- **Performance Highlights**: 다국어 및 다중 측면 혐오 발언 데이터 세트(MLMA)와 LTEDI 등의 오픈 소스 데이터를 사용하여 반-LGBTQ+ 혐오 발언 탐지 시스템을 개발했습니다. 이 시스템을 통해 지리적 방언에 걸쳐 소셜 미디어 데이터의 반-LGBTQ+ 혐오 발언을 모니터링했습니다. 그러나 키워드 검색 접근 방식은 욕설에 과도하게 특화된 결과를 초래할 수 있으며, 지역 방언에서는 잘못된 분류가 발생할 가능성이 높습니다. 예를 들어, 뉴질랜드 트윗에서 'bugger', 'digger', 'stagger' 등의 단어가 혐오 발언으로 잘못 분류되었습니다.



### An Empirical Comparison of Generative Approaches for Product Attribute-Value Identification (https://arxiv.org/abs/2407.01137)
- **What's New**: 이 논문에서는 Product Attribute and Value Identification (PAVI) 문제를 생성(generation) 태스크로 정의하고, 이를 가장 포괄적으로 평가하였습니다. 세 가지 Attribute-Value Generation (AVG) 전략을 비교하고, 효율적인 End-to-end AVG 접근 방식이 다른 방법들보다 성능이 뛰어남을 보여주었습니다.

- **Technical Details**: 세 가지 AVG 접근 방식은 파이프라인(pipeline) AVG, 멀티태스크(멀티작업) AVG, 그리고 엔드투엔드(end2end) AVG입니다. 각 모델은 encoder-decoder 언어 모델(T5, BART 등)을 파인튜닝(fine-tuning)하여 구현되었습니다. 파이프라인 AVG는 값 추출(value extraction)과 속성 생성(attribute generation)을 별도의 모델로 분리합니다. 멀티태스크 AVG는 하나의 모델로 두 서브태스크(sub-task)를 모두 학습합니다. 엔드투엔드 AVG는 하나의 모델로 속성-값 쌍을 바로 생성합니다.

- **Performance Highlights**: 세 가지 접근 방식을 AE-110K, OA-mine, MAVE라는 세 개의 실제 전자상거래 데이터셋을 사용하여 평가한 결과, 엔드투엔드 방법이 다른 방법들보다 전반적으로 높은 성능을 보였습니다. 모델 크기와 언어 모델에 따라 성능 차이가 있음을 확인했습니다.



### Cross-Lingual Transfer Learning for Speech Translation (https://arxiv.org/abs/2407.01130)
- **What's New**: 최근 다중언어 기반의 기초 모델(multilingual foundation models) 개발에 대한 관심이 높아지고 있습니다. 이에 따라 Zero-shot 교차언어 전이(cross-lingual transfer) 능력을 자연어 처리(NLP) 과제에서 입증된 바 있는데, 이는 특정 언어의 과제별 데이터로 미세 조정(fine-tuning)된 모델이 다른 언어에서도 성능 향상을 가져온다는 것을 보여줍니다. 본 연구에서는 음성 기반 모델이 동일한 전이 능력을 보이는지 탐구합니다. Whisper라는 다중언어 음성 기초 모델을 이용하여 음성 인코더가 생성한 발화 표현(utterance representation)을 조사합니다.

- **Technical Details**: Whisper 모델은 자동 음성 인식(ASR)과 음성 번역(ST)의 다양한 작업을 수행할 수 있는 대규모 미리 학습된 다중 언어 음성 모델입니다. Whisper의 인코더가 생성한 음성 임베딩(space)은 다른 언어의 단어들이 유사한 의미 공간(semantic space)으로 매핑됨을 보여줍니다. 이 연구에서는 Whisper 모델을 단순히 영어-중국어 번역 데이터로 미세 조정하였을 때 다른 언어의 입력 발화에서도 성능 향상이 있음을 보였습니다. 또한 자원이 부족한 언어에 대한 실험에서 Whisper가 미리 학습 단계에서 보지 못한 언어의 발화도 번역할 수 있음을 보여줍니다.

- **Performance Highlights**: Whisper 모델은 특히 낮은 자원의 언어에서도 탁월한 성능을 보입니다. 음성-음성 검색(speech-to-speech retrieval) 작업에서 단어의 의미가 동일한 다양한 언어의 발화들이 서로 가까운 임베딩으로 매핑됨을 보여주는 높은 리콜(recall) 비율을 보였습니다. 이러한 공통 임베딩 공간을 활용하여 Whisper는 다른 언어에서 제로샷 교차언어 전이 능력을 입증했습니다.



### Investigating the potential of Sparse Mixtures-of-Experts for multi-domain neural machine translation (https://arxiv.org/abs/2407.01126)
- **What's New**: 이 연구는 멀티 도메인 신경 기계 번역(NMT) 모델을 개발하는 데 중점을 두고 있으며, Sparse Mixture-of-Experts(SMoE) 모델이 다양한 도메인 데이터를 효과적으로 처리할 수 있는 가능성을 탐구합니다. 또한, 이 연구는 간단한 Transformer 모델의 폭 스케일링(Width Scaling)이 SMoE와 동일한 성능을 제공하는 효율적인 접근법일 수 있음을 발견했습니다.

- **Technical Details**: Sparse Mixture-of-Experts(SMoE) 모델은 게이트 메커니즘을 통해 입력 토큰 별로 활성화할 모델 파라미터의 부분 집합을 결정합니다. 이는 다양한 도메인을 처리하는 데 필요한 모델 크기를 스케일링하면서도 추론 시 연산 비용(FLOPs)을 일정하게 유지합니다. 또한, SMoE는 도메인 태그(Domain Tags)와 도메인 어댑터(Domain Adapters)의 중간 정도로, 서로 유사한 도메인 간에는 지식 전이를, 반대의 경우에는 부정적 전이를 방지할 수 있습니다.

- **Performance Highlights**: SMoE 모델은 베이스라인 Transformer Base 아키텍처를 크게 능가하였으며, 이는 주로 모델 스케일링 효과 때문입니다. 간단한 폭 스케일링을 통해 GPU에서 거의 추가 연산 오버헤드 없이 모델의 효율성을 유지할 수 있었습니다. 또한, 도메인 랜덤화와 일반 도메인 데이터를 혼합하는 기법을 통해 도메인 레이블이 잘못된 경우에도 모델의 견고성과 범용성을 크게 향상시킬 수 있었습니다.



### Calibrated Large Language Models for Binary Question Answering (https://arxiv.org/abs/2407.01122)
Comments:
          Accepted to COPA 2024 (13th Symposium on Conformal and Probabilistic Prediction with Applications)

- **What's New**: 이번 연구는 큰 언어 모델(LLMs)의 이진 텍스트 분류 작업에서 예측의 불확실성을 정량화하는 새로운 방법을 제안합니다. IVAP(Inductive Venn--Abers Predictor)를 사용하여 이진 레이블과 관련된 출력 토큰의 확률을 보정하는 방법을 도입했습니다. Llama 2 모델을 사용한 BoolQ 데이터셋 실험에서 IVAP가 기존의 온도 스케일링(temperature scaling) 방법보다 일관되게 우수한 보정 성능을 보였습니다.

- **Technical Details**: 이 연구는 LLM이 생성하는 원시 단어 점수(raw word scores) 또는 로짓(logits)을 직접 보정하는 기법을 사용합니다. 이 로짓은 이진 클래스 레이블(예: 'yes'와 'no')에 해당합니다. Venn–Abers Predictors를 이용하여 최적의 아이소토닉 매핑(isotonic mapping)을 학습하여, 로짓을 보정된 클래스 확률로 변환합니다. 이 방법은 추가적인 모델 학습이나 수정 없이 사용할 수 있는 제로샷(Zero-shot) 솔루션을 제공합니다.

- **Performance Highlights**: Llama 2 7B 모델을 이용한 실험에서, 이 방법은 온도 스케일링(Temperature Scaling)보다 더 나은 보정 성능을 보였습니다. 실험은 BoolQ 데이터셋을 사용하여 수행되었으며, 그 결과 IVAP가 높은 예측 품질을 유지하면서 잘 보정된 확률을 제공합니다.



### Pron vs Prompt: Can Large Language Models already Challenge a World-Class Fiction Author at Creative Text Writing? (https://arxiv.org/abs/2407.01119)
Comments:
          9 pages 6 figures

- **What's New**: 최근 많은 연구 결과에서 대형 언어 모델(Large Language Models, LLMs)이 다양한 언어 관련 작업에서 평균 인간을 능가하고 있습니다. 이번 연구에서는 LLM이 최고 수준의 소설가와 비교하여 창의적 글쓰기 능력을 얼마나 뛰어넘을 수 있는지 알아보는 첫 시도를 하였습니다. 이를 위해, 저명한 소설가 Patricio Pron과 GPT-4 간의 창의적 글쓰기 대결을 실시하였습니다.

- **Technical Details**: Patricio Pron과 GPT-4는 각각 30개의 제목을 제공한 후, 자신과 상대방의 제목에 대해 짧은 이야기를 작성하였습니다. 평가를 위해 Boden의 창의성 정의를 참고하여 총 5,400개의 전문가 평가를 수집하였습니다. 연구 질문으로는 현재의 생성 인공지능 기술이 최고의 인간 작가와 비교할 수 있는지, 프롬프트가 창의성에 어떤 영향을 미치는지, 영어 이외의 언어에서의 성능은 어떠한지 등을 다루었습니다.

- **Performance Highlights**: 연구 결과, GPT-4는 여전히 최고의 인간 작가와 경쟁하기에는 부족한 것으로 나타났습니다. 5,400개의 전문가 평가에서 품질 차원 모두에서 Patricio Pron이 GPT-4보다 우수한 평가를 받았습니다. 추가적으로, 소설가가 제공한 제목으로 프롬프트를 설정했을 때 GPT-4의 성과가 향상되었으며, 영어보다 스페인어 글쓰기 성능이 떨어지는 것으로 확인되었습니다. 또한, GPT-4의 글쓰기 스타일은 시간 경과에 따라 인식하기 쉬워지는 경향이 있었습니다.



### BERGEN: A Benchmarking Library for Retrieval-Augmented Generation (https://arxiv.org/abs/2407.01102)
Comments:
          29 pages

- **What's New**: 최근의 생성적 대형 언어 모델(Large Language Models, LLM)의 인기를 반영하여, Retrieval-Augmented Generation(RAG) 접근 방식이 제안되었습니다. 이러한 접근 방식들은 다양한 구성요소와 함께 사용되며, 일관되지 않은 벤치마킹이 주요 과제가 됩니다. 이를 해결하기 위해, BERGEN이라는 종단간(end-to-end) 연구 표준화 라이브러리를 소개합니다.

- **Technical Details**: BERGEN은 RAG 실험을 표준화하고 재현성을 보장하는 오픈소스 Python 라이브러리입니다. 이 라이브러리는 500개 이상의 실험을 통해 최첨단의 retrievers, rerankers, 그리고 LLMs를 벤치마크했고, 기존의 다양한 RAG 지표와 데이터셋을 분석합니다. Hugging Face(HF) 허브를 기반으로 데이터셋과 모델을 처리하며, 다양한 모델 아키텍처와 훈련 및 평가 구성을 지원합니다.

- **Performance Highlights**: BERGEN은 기존의 단편적이고 비효율적인 실험 설정을 일원화하여, RAG 접근 방식을 보다 체계적이고 비교할 수 있게 합니다. 또한, 죽음의 중요한 요소로 최첨단의 retrievers와 rerankers의 사용을 강조하고, 일반적인 표면 일치 메트릭(exact match, F1, Rouge-L 등) 외에도 LLM 기반 평가의 중요성을 부각합니다.



### Eliminating Position Bias of Language Models: A Mechanistic Approach (https://arxiv.org/abs/2407.01100)
Comments:
          18 pages, 5 figures

- **What's New**: 최근 연구는 언어 모델(LM)에서 발생하는 위치 편향(position bias)의 문제를 해결하기 위한 새로운 방법을 제안하고 있습니다. 위치 편향은 모델이 문맥 내에서 콘텐츠의 위치에 따라 다른 우선순위를 두면서 발생하며, 이는 다양한 응용 프로그램에서 성능, 안정성 및 신뢰성을 저하시키는 원인이 됩니다. 이 연구는 이러한 위치 편향을 제거하기 위해 별도의 학습 없이 사용 가능한 PINE(Position-INvariant inferencE)을 제안합니다.

- **Technical Details**: 연구팀은 위치 편향의 주요 원인으로 'Causal Attention'과 'Relative Positional Encodings'를 지목했습니다. Causal Attention은 모델이 멀리 떨어진 콘텐츠를 더 선호하게 만들고, 상대적 위치 인코딩(Relative Positional Encodings)인 RoPE는 가까운 콘텐츠를 더 선호하게 만듭니다. PINE 방법은 입력 프롬프트에서 제공된 순서를 사용하는 대신, Causal Attention을 양방향 주의(attention)로 전환하고 모델의 주의값을 사용하여 세그먼트의 상대적 순서를 결정하는 방식으로 위치 편향을 제거합니다. 이는 학습이 필요 없는 제로샷 방식으로 작동합니다.

- **Performance Highlights**: PINE을 사용하면, LMs는 위치 편향이 많이 존재하는 다운스트림 작업에서 더 나은 성능과 신뢰성을 보입니다. 특히, 추론 쌍을 평가할 때 PINE은 대부분의 경우 성능을 8~10% 포인트 향상시킵니다. 또한, Llama-3-70B-Instruct 모델이 GPT-4-0125-preview 모델보다 더욱 좋은 성과를 보였습니다.



### IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation (https://arxiv.org/abs/2407.01093)
Comments:
          Accepted by ACL 2024 Main

- **What's New**: 새롭게 발표된 IBSEN 프레임워크는 감독-배우 공동 작업 에이전트 구조로, 드라마 스크립트를 생성하고 사건 전개를 더 원활하게 통제할 수 있게 합니다. 감독 에이전트는 사용자가 원하는 플롯 요약을 작성하고, 배우 에이전트에게 캐릭터의 역할을 지시하며 인간 플레이어가 시나리오에 참여할 때 플롯을 재조정합니다. 이를 통해 목표로 하는 플롯방향으로 이야기가 전개되도록 보장합니다.

- **Technical Details**: IBSEN 프레임워크에서는 세 가지 에이전트 구조가 도입되었습니다: 감독(Director), 배우(Actor), 플레이어(Player). 감독 에이전트는 현재 드라마의 스토리 라인을 작성하고 검사하는 역할을 하며, 배우 에이전트는 실제 드라마 스크립트를 생성하는 역할을 합니다. 플레이어 에이전트는 인간 플레이어와 상호작용할 수 있는 기능을 제공하며, 이를 통해 이야기에 동적으로 개입할 수 있게 합니다. 주요 구성 요소로는 '플롯 목표'(Plot Objectives) 리스트와 계층적 방법(Hierarchical Method)을 사용하여 플롯을 생성하고 전개하는 방식이 있습니다.

- **Performance Highlights**: 실험 결과, IBSEN 프레임워크는 초기 플롯 목표만으로도 완전하고 다양한 드라마 스크립트를 생성할 수 있었으며, 드라마 속 캐릭터의 특성을 유지하면서 이야기를 효과적으로 전개할 수 있음을 확인하였습니다.



### M2QA: Multi-domain Multilingual Question Answering (https://arxiv.org/abs/2407.01091)
- **What's New**: 이번 연구에서는 다국어 및 다도메인(multilingual multi-domain) 질의응답(Question Answering, QA) 벤치마크인 M2QA를 소개합니다. 이 데이터셋에는 독일어, 터키어, 중국어로 작성된 총 13,500개의 SQuAD 2.0 스타일의 질문-답변 인스턴스가 포함되어 있으며, 제품 리뷰, 뉴스, 창작 글 등 다양한 도메인을 다룹니다.

- **Technical Details**: M2QA는 독일어, 터키어, 중국어의 자연 발생 텍스트를 수동으로 주석 처리하여 번역을 통해 발생할 수 있는 '번역체(translationese)' 등의 인공적인 요소를 최소화하고, 각 언어의 문화적 특성을 통합했습니다. 또한, 다양한 모델 및 전이 학습 기법을 통해 성능을 평가하며, 전체 미세 조정(fully-finetuning), 모듈식 전이 학습(modular transfer learning), 대규모 언어 모델(LLM)도 포함하여 모델의 성능을 탐구합니다.

- **Performance Highlights**: 이번 연구에서 가장 주목할 만한 결과는 도메인 및 언어 조합에 따라 모델 성능이 상당히 달라지며, 출발 언어-도메인 조합과 목표 언어-도메인 조합 간의 성능 차이가 크게 나타난다는 것입니다. 이를 통해 기존의 널리 사용되는 SQuAD 2.0 평가 메트릭이 다국어 추출 QA 평가에 적합하지 않음을 발견하고, 이를 개선하기 위한 새로운 메트릭을 제안했습니다. 또한, 현대의 대규모 언어 모델이 목표 도메인-언어 조합에서 상당히 낮은 성능을 보임으로써, 언어 및 도메인 특화 정보를 효과적으로 전이하는 새로운 방법이 필요함을 강조했습니다.



### Min P Sampling: Balancing Creativity and Coherence at High Temperatur (https://arxiv.org/abs/2407.01082)
Comments:
          8 Pages

- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 텍스트를 생성할 때 발생하는 일관성과 창의성 간의 균형 문제를 해결하기 위해 새로운 동적 절단 샘플링 방법인 min-p를 제안합니다.

- **Technical Details**: min-p 샘플링 방법은 토큰 선택을 위한 최소 확률 임계값을 동적으로 설정하여 높은 온도에서도 일관성을 유지하면서 창의성을 보존합니다. 기존의 top-p 샘플링 방식은 높은 온도에서 유효한 토큰의 '불확실한 꼬리'가 샘플링 풀에 포함되어 창의성과 일관성의 균형을 맞추기 어렵다는 문제를 가지고 있습니다.

- **Performance Highlights**: 여러 벤치마크 실험 결과, min-p 샘플링 방법은 높은 온도 설정에서도 일관성을 유지하면서 top-p 샘플링 방법을 상회하는 창의적이고 다양한 출력을 생성할 수 있음을 보여주었습니다. min-p는 여러 오픈 소스 LLM 구현에서 채택되었으며, 오픈 소스 커뮤니티에서도 그 실용성과 잠재적 가치를 검증받았습니다.



### Face4RAG: Factual Consistency Evaluation for Retrieval Augmented Generation in Chines (https://arxiv.org/abs/2407.01080)
- **What's New**: 최근 연구에서는 RAG(Retrieval Augmented Generation) 시스템 내에서 팩트 일관성이 주요 문제로 대두되었습니다. 이 문제를 해결하기 위해 최초의 종합적 팩트 일관성 평가(FCE) 벤치마크 'Face4RAG'가 제안되었습니다. 이 벤치마크는 특정 LLM(대규모 언어 모델)에 의존하지 않으며, 다양한 에러 유형을 포괄하는 인위적 데이터셋과 실제 데이터를 포함합니다.

- **Technical Details**: Face4RAG는 팩트 일관성 오류의 여러 유형을 포함하는 새로운 오류 유형 분류체계를 통해 다양한 RAG 시스템에서 FCE 방법을 평가합니다. 주요 오류 유형은 환각 오류(hallucination error), 지식 오류(knowledge error), 논리적 오류(logical fallacy) 등 세 가지로 분류됩니다. 새로운 FCE 방법으로 L-Face4RAG가 제안되었으며, 이 방법은 논리 보전 응답 분해(logic-preserving answer decomposition)와 팩트-논리 FCE(fact-logic FCE)로 구성됩니다.

- **Performance Highlights**: 광범위한 실험 결과, L-Face4RAG는 기존의 FCE 방법보다 훨씬 높은 정확도를 보여주었으며, 특히 다양한 RAG 시스템과 다른 FCE 작업에서도 우수한 성능을 입증하였습니다. 영어 FCE 벤치마크와 요약, 대화, 팩트 검증 작업에서도 최고의 성과(SOTA)를 기록하였습니다.



### Development of Cognitive Intelligence in Pre-trained Language Models (https://arxiv.org/abs/2407.01047)
- **What's New**: 최근 연구에서는 대형 사전 학습된 언어 모델(PLM, Large Pre-trained Language Models)에서 발현하는 인지 능력을 발견했습니다. 이번 연구에서는 PLM의 훈련 과정 동안 인지적 발달 궤도를 분석하고, 인간의 인지 발달과의 정렬성을 조사합니다. ThinkTank는 숫자 능력, 언어 능력, 개념적 이해, 유동적 추론과 같은 네 가지 과제를 통해 인지적 정렬성을 평가하는 벤치마크입니다.

- **Technical Details**: 이번 연구는 PLM의 중간 및 최종 훈련 단계에서의 발달적 정렬성을 평가합니다. PLM은 인간의 인지 발달 궤도에 대해 일관된 최대 정렬성 창을 보여줍니다. 이는 모델 크기와 상관없이 나타나며, '백지 상태'에서 빠르게 학습할 준비를 갖추게 되는 것을 의미합니다. 또한 훈련 과정의 후기에는 손실을 줄이는 공학적 목표를 중시하게 됩니다.

- **Performance Highlights**: 실험 결과, PLM은 다양한 크기와 훈련 데이터 양에 관계없이 인간 인지 발달 궤도와 일치하는 최대 정렬성 창을 지속적으로 보여줍니다. 이 창 이전에는 모델이 빠르게 학습할 준비를 갖추며, 이후에는 손실을 줄이는 목표를 우선시하게 됩니다. ThinkTank는 언어 모델의 인지적 성능을 평가하는 데 있어 새로운 표준을 제시하며, 인지 과학 및 인공지능 모델링에 더 나은 통찰을 제공합니다.



### Augmenting Document-level Relation Extraction with Efficient Multi-Supervision (https://arxiv.org/abs/2407.01026)
- **What's New**: 이번 연구에서는 문서 수준의 관계 추출(DocRE)에서 효율적이고 견고한 먼 거리 지도 학습(Distant Supervision) 데이터를 활용하기 위해 Efficient Multi-Supervision(EMS) 방법을 제안합니다. 이 방법은 초기 단계에서 먼 거리 지도 데이터에서 유용한 문서를 선택한 다음, 다중 감독 순위 손실(MSRL)을 사용해 여러 감독 소스의 지식을 통합하여 모델을 학습합니다.

- **Technical Details**: 본 연구는 문서 수준의 관계 추출에서 DS 데이터의 효율적인 활용을 위해 두 가지 주요 단계를 제안합니다. 첫 번째 단계는 Document Informativeness Ranking(DIR)으로, DS 데이터에서 가장 정보 가치가 높은 문서를 선택하여 훈련 데이터를 보강합니다. 두 번째 단계에서는 Multi-Supervision Ranking-based Loss(MSRL)를 통해 먼 거리 지도, 전문가 모델, 그리고 자가 지도(supervision)와 같은 여러 소스로부터의 감독을 통합하여 노이즈의 영향을 줄입니다.

- **Performance Highlights**: DocRED 데이터셋을 사용한 실험에서 제안된 EMS 방법이 모델 성능을 향상시키는 동시에 기존 기준 대비 시간 효율성 또한 높인다는 것을 검증했습니다. 특히 DIR과 MSRL 두 단계가 모두 성능 향상에 중요한 역할을 한다는 것이 입증되었습니다.



### DynaThink: Fast or Slow? A Dynamic Decision-Making Framework for Large Language Models (https://arxiv.org/abs/2407.01009)
- **What's New**: 새로운 논문은 대형 언어 모델(LLMs)이 '빠른(fast)' 추론과 '느린(slow)' 추론 방법을 자율적으로 선택할 수 있도록 하는 동적 의사결정 프레임워크인 'DynaThink'를 소개합니다. 이 프레임워크를 통해 다양한 문제에 대해 효율적이고 효과적인 해결책을 최적화할 수 있습니다.

- **Technical Details**: DynaThink 프레임워크는 LLMs가 문제를 빠르게 해결할 수 있는 고확신 솔루션에 해당하는 'Fast' 경로와 더 복잡한 여러 판단 경로를 탐색해야 하는 'Slow' 경로로 문제를 분류하는 방식으로 동작합니다. '일관성 검증(consistency verification)'과 '추론 복잡성 검증(reasoning complexity verification)' 두 가지 기준을 사용하여 빠르고 느린 추론을 구분합니다. 일관성 검증은 다수 답변이 일치하면 고확신 솔루션으로 간주하고, 추론 복잡성 검증은 최소 추론 단계를 요구하는 답변을 선택합니다.

- **Performance Highlights**: {'Main Results': '다양한 추론 데이터셋(StrategyQA, GSM8K, MATH, AQUA-RAT, SVAMP, MATHQA)에 대해 DynaThink는 높은 효율성과 효과성을 보였습니다. 예를 들어, zero-shot MATH 설정에서 GPT-3.5-Turbo를 사용하여 DynaThink는 45%의 정확도를 달성했으며, 이는 동일한 쿼리 수(2,758회)를 사용한 SC의 41.9%보다 뛰어납니다. 또한, few-shot MathQA 설정에서 2849회 쿼리를 통해 SC보다 4% 높은 정확도 향상을 보였습니다.', 'LLM Comparison': '세 가지 블랙박스 LLMs(GPT-3.5-turbo, GPT-4, Gemini)와 하나의 오픈소스 LLM (Mixtral-8x7B)을 사용하여 결과를 비교했을 때, GPT-4가 최고 성능을 보였습니다. 예를 들어, zero-shot MATHQA 시나리오에서 DynaThink는 2827회 쿼리로 73.8% 정확도를 달성하여, SC의 71.7% 정확도(3000회 쿼리)를 능가했습니다.'}



### Engineering Conversational Search Systems: A Review of Applications, Architectures, and Functional Components (https://arxiv.org/abs/2407.00997)
Comments:
          Accepted to ACL 2024 NLP4ConvAI Workshop

- **What's New**: 대화형 검색 시스템(Conversational Search Systems, CSSs)은 자연어 상호 작용을 통해 정보 검색을 가능하게 함으로써 사용자의 정보 습득을 극대화하려 합니다. 최근 들어 CSSs의 구현 과정에 대한 체계적인 문헌 검토를 통해 주요 연구 결과를 정리 및 제시하였습니다.

- **Technical Details**: 본 연구는 CSSs의 이론적 연구와 기술적 구현 간의 연결고리를 조사하였습니다. CSS의 개념적 시스템 속성과 적합한 응용 시나리오를 식별하고, 문헌에서 제안된 아키텍처 요소를 통합하여 계층적 아키텍처 프레임워크를 제시하였습니다. 핵심 기능 요소로는 혼합 이니셔티브 상호작용(mixed-initiative interaction), 상호 이해(mutual understanding), 문맥 인식 및 메모리(context awareness and memory), 지속적인 개선(continuous refinement) 등이 있습니다.

- **Performance Highlights**: CSSs는 텍스트 기반, 음성 기반, 하이브리드 상호작용 모드(text-based, speech-based, hybrid interaction modalities)를 지원할 수 있으며, 이들의 사용 사례를 폭넓게 검토하였습니다. 특히 대화형 검색은 탐색적 검색 목표나 복잡한 데이터 구조를 다루는 데 효과적이라는 점이 강조됩니다. 또한, 최근 LLMs의 통합을 통한 CSSs의 잠재력, 제한 사항 및 위험성에 대한 논의를 담고 있습니다.



### Can Small Language Models Learn, Unlearn, and Retain Noise Patterns? (https://arxiv.org/abs/2407.00996)
- **What's New**: 이 연구는 Small Language Models (SLMs)가 소음(Noise)을 학습하고 제거할 수 있는 능력을 조사합니다. SLMs는 일반적으로 70억 개 미만의 파라미터를 가지고 있는 대형 언어 모델(LLMs)의 더 컴팩트한 버전으로 간주됩니다. 이 연구에서는 Olmo 1B, Qwen1.5 1.8B, Gemma 2B, Phi2 2.7B의 네 가지 미리 학습된 SLM을 사용했습니다.

- **Technical Details**: 모델들은 소음이 없는 상태에서 지시 튜닝(In-context Learning)을 통해 학습되었고, 이후 소음 패턴이 도입되었습니다. 다양한 종류의 소음을 추가하여 모델의 학습 및 제거 능력을 평가했습니다. 단어 수준의 소음은 플립(FLIPword), 문자 수준의 소음은 문자 플립(FLIPchar) 방법을 사용했습니다. 훈련 데이터 세트로는 AlpaGasus_9k 및 Dolly_3k를 사용하여 소음 없는 데이터를 참조로 모델의 학습 및 제거 능력을 시험했습니다.

- **Performance Highlights**: Phi2는 단어 수준의 소음에서 일관되게 우수한 성능을 보였지만 문자 수준의 소음에서는 가장 낮은 성능을 보였습니다. 약 10억 개의 파라미터를 가지고 있는 Olmo는 일관되게 우수한 성과를 보였습니다.



### LLM Uncertainty Quantification through Directional Entailment Graph and Claim Level Response Augmentation (https://arxiv.org/abs/2407.00994)
Comments:
          11 pages main content, 5 pages appendix

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 응답에서 불확실성(uncertainty)을 평가하는 새로운 방법을 제안합니다. 기존의 비대칭적인 그래프를 활용한 방향성 불안정성(directional instability)을 평가할 수 있는 시스템을 설계하였으며, 이를 통해 모델의 신뢰성을 높이고자 합니다. 또한, 생성된 응답의 모호성을 줄이기 위한 클레임 기반 보강 방법도 제안합니다.

- **Technical Details**: 논문에서는 함의 확률(entailment probability)을 사용하여 방향성 그래프(directional graph)를 구축하고, 비대칭적인 그래프의 특성을 고려하여 랜덤워크 라플라시안(Random Walk Laplacian)을 적용합니다. 이 과정을 통해 라플라시안 프로세스에서 유도된 고유값(eigenvalues)을 활용하여 불확실성을 집계합니다. 또한, 기존 작업의 의미론적 불확실성(semantics uncertainty)을 본 시스템에 통합하는 방법도 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법의 우수성을 입증하였습니다. 응답의 방향성 논리(directional logic)를 반영한 평가 방법이 기존의 평균화 또는 의미론적 유사성만을 고려한 방법보다 더 정확하고 포괄적인 불확실성 평가를 제공함을 확인하였습니다.



### The House Always Wins: A Framework for Evaluating Strategic Deception in LLMs (https://arxiv.org/abs/2407.00948)
Comments:
          Research conducted at the Deception Detection Hackathon 2024 hosted by Apart & Apollo Research

- **What's New**: 대형 언어 모델(LLMs)에서 전략적 기만을 평가하기 위한 새로운 프레임워크를 제안했습니다. 이 프레임워크에서 LLM은 두 가지 시나리오의 게임 마스터로 작동합니다: 무작위 게임 메커니즘이 적용된 경우와 무작위 또는 의도적인 행동 중 하나를 선택할 수 있는 경우입니다. 블랙잭을 예제로 사용하여, LLM들이 '집'의 이익을 위해 전략을 개발하는지 여부를 분석했습니다.

- **Technical Details**: 제안된 프레임워크는 블랙잭 환경에서 LLM의 전략적 의사결정과 정보 조작을 연구합니다. 세 가지 시나리오로 설정되었습니다: 1) 컨트롤(무작위 딜러): 딜러가 무작위로 카드를 선택하는 경우, 2) LLM 딜러(암묵적 무작위성): 딜러가 무작위로 카드를 선택하라는 지시를 받는 경우, 3) LLM 딜러(명시적 선택): 딜러가 무작위로 선택할지 특정 카드를 선택할지 명시적인 선택을 받는 경우.

- **Performance Highlights**: 실험 결과, 암묵적 무작위 지시를 받은 경우 모든 테스트된 LLM은 공정 플레이에서 상당한 벗어남을 보였습니다. 예를 들어, Llama3-70B는 암묵적 무작위성 시나리오에서 18.80%의 플레이어 승률과 1.18%의 딜러 버스트율을 보였습니다. 반면, 명시적 선택을 받은 경우에는 대부분 공정 플레이를 따랐습니다. 이는 지시의 프레임이 LLM의 잠재적 기만 행동에 중요한 역할을 한다는 것을 시사합니다.



### MalAlgoQA: A Pedagogical Approach for Evaluating Counterfactual Reasoning Abilities (https://arxiv.org/abs/2407.00938)
- **What's New**: 새로운 데이터셋 MalAlgoQA가 도입되었습니다. 이 데이터셋은 대형 언어 모델(LLMs)의 반사실적 추론(counterfactual reasoning) 능력을 교육적인 접근을 통해 평가하기 위해 설계되었습니다. MalAlgoQA는 수학 및 독해 질문으로 구성되며, 각 질문에는 네 개의 답변 선택지와 그에 대한 근거가 포함되어 있습니다. 특히, 잘못된 답변의 근거인 'malgorithms'에 중점을 두며, 이 잘못된 추론 과정을 이해하고 수정하는 데 중요한 정보를 제공합니다.

- **Technical Details**: MalAlgoQA는 807개의 수학 문제와 290개의 독해 문제로 구성되어 있으며, 각 문제는 다양한 학년(3-11학년)과 내용 분류(예: 대수학, 기하학, 수와 연산) 및 지식 깊이 수준(Depth of Knowledge, DOK: 1-3)으로 나누어져 있습니다. 이 데이터셋은 대형 언어 모델이 주어진 답변 선택지와 그에 대한 추론 과정을 식별하는 'Malgorithm Identification' 작업을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델 성능을 평가하기 위해 두 가지 주요 지표인 Algorithm Identification Accuracy (AIA)와 Malgorithm Identification Accuracy (MIA)를 도입했습니다. GPT-4o 모델은 AIA에서 95.7%의 정확도를 기록한 반면, MIA에서는 66.1%로 크게 떨어지는 성능을 보였습니다. Chain-of-Thought (CoT) 프롬프팅 기법은 MIA를 지속적으로 향상시키지 못하며, 단순한 프롬프팅에 비해 오히려 성능이 저하되는 경우도 발견되었습니다. 특히, 확률과 같은 복잡한 내용 영역에서는 성능이 더 큰 감소를 보였습니다.



### Large Language Model Enhanced Knowledge Representation Learning: A Survey (https://arxiv.org/abs/2407.00936)
- **What's New**: 이번 리뷰는 대형 언어 모델(Large Language Models, LLMs)과 지식 표현 학습(Knowledge Representation Learning, KRL)의 통합에 관한 최신 연구를 다룸으로써, 이 두 가지 기술의 융합이 인공지능 분야에서의 중요한 발전을 의미한다는 점을 강조합니다. LLM은 고급 언어 및 문맥 이해 능력을 활용하여 KRL의 정확성, 적응성 및 효율성을 향상시키며, 이에 따라 그 응용과 잠재력을 확장시킵니다.

- **Technical Details**: 이번 설문 조사는 Transformer 아키텍처에 기반한 세 가지 모델(인코더 기반, 인코더-디코더 기반, 디코더 기반)로 분류합니다. 이 모델들의 다양한 KRL 다운스트림 작업에 대한 실험 데이터를 분석하여 각 접근 방식의 강점과 약점을 평가합니다. 또한, 지식 그래프(Knowledge Graph, KG)와 대형 언어 모델의 융합과 관련된 최근의 발전 사항을 탐구하고, 언어 모델이 그래프 상에서 어떻게 작동하는지에 대한 기초 원칙을 제시합니다.

- **Performance Highlights**: LLM과 KRL의 통합은 다양한 자연어 처리(NLP) 작업에서 뛰어난 성능을 보여주었습니다. 특히, 질문 응답, 텍스트 생성 및 문서 이해 작업에서 높은 정확도와 효율성을 입증하였으며, 이는 인공지능 일반(AGI) 목표에도 중요한 잠재력을 가지고 있음을 나타냅니다. 각 Transformer 아키텍처는 다양한 KRL 다운스트림 작업에서 고유한 장점과 단점을 드러냈습니다.



### CLEME2.0: Towards More Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction (https://arxiv.org/abs/2407.00934)
Comments:
          16 pages, 8 tables, 2 figures. Under review

- **What's New**: 본 논문은 문법 오류 수정(Grammatical Error Correction, GEC) 시스템의 평가 지표 해석 가능성을 개선하는 CLEME2.0을 제안합니다. CLEME2.0은 참조 기반 평가 전략으로 네 가지 기본 차원인 hit-correction, error-correction, under-correction, over-correction을 설명하여 GEC 시스템의 주요 특성과 단점을 파악할 수 있도록 돕습니다. 이는 다른 참조 기반 및 비참조 기반 지표에 비해 사람의 평가와 높은 일관성을 보입니다. 광범위한 실험을 통해 CLEME2.0의 효과성과 견고성을 입증했으며, 관련 코드도 동료 검토 후 공개할 예정입니다.

- **Technical Details**: CLEME2.0은 GEC 시스템의 평가의 주된 문제점을 해결하기 위해 참조 대상을 기반으로 한 평가 전략을 도입합니다. 평가 시 hit-correction, error-correction, under-correction, over-correction 네 가지 차원을 측정합니다. 이를 통해 GEC 시스템의 문법성(grammaticality)과 충실성(faithfulness)을 평가합니다. 또한 변경의 정도에 따라 가중치를 부여하는 similarity-based weighting과 LLM-based weighting을 도입하여, 전통적인 평가 방식의 표면적인 형태 유사성의 한계를 극복합니다.

- **Performance Highlights**: CLEME2.0은 두 개의 인간 평가 데이터셋(GJG15, SEEDA)과 여섯 개의 참조 데이터셋을 통해 일관되고 높은 상관 관계를 보여주었습니다. 기존의 ERRANT나 MaxMatch와 같은 주류 GEC 평가 지표와 달리, CLEME2.0은 더 세밀한 차원까지 해석할 수 있는 능력을 지니고 있습니다. 이는 더 정확한 GEC 시스템 성능 평가와 특성 파악을 가능하게 합니다.



### EXCGEC: A Benchmark of Edit-wise Explainable Chinese Grammatical Error Correction (https://arxiv.org/abs/2407.00924)
Comments:
          22 pages, 10 tables, 9 figures. Under review

- **What's New**: 이번 연구는 문법 오류 수정 (Grammatical Error Correction, GEC)의 설명 가능성을 다루는 새로운 과업인 EXplainable GEC (EXGEC)를 도입했습니다. EXGEC는 수정과 설명의 상호작용을 고려하여 두 과업을 통합함으로써 사용자에게 더 깊은 이해를 제공하도록 합니다.

- **Technical Details**: EXGEC 과업을 지원하기 위해, 중국어 EXGEC를 위한 맞춤형 기준인 EXCGEC를 제안하였습니다. EXCGEC는 하이브리드 편집 기반 설명(hybrid edit-wise explanations)을 특징으로 하며, 오류 유형, 오류 심각도 수준(1-5점) 및 오류 설명의 세 가지 요소를 포함합니다. 또한, 반자동 데이터세트 구축 방법을 사용하여 8,216개의 설명이 추가된 샘플을 만들었습니다. 이 모든 데이터와 코드들은 검토 후 공개될 예정입니다.

- **Performance Highlights**: 포스트-설명 (correct-then-explain) 모델이 프리-설명 (explain-then-correct) 모델보다 높은 성능을 보여주었으며, Correct-Then-Explain (COTE) 디코딩 알고리즘이 LLM의 정렬 작업 부담을 완화하여 성능 향상에 크게 기여했습니다. 인간 평가 실험을 통해 자동 메트릭이 설명의 인간 일관성을 잘 반영함을 확인하였습니다.



### Preserving Multilingual Quality While Tuning Query Encoder on English Only (https://arxiv.org/abs/2407.00923)
- **What's New**: 본 연구에서는 다중언어 검색 모델(multilingual retrieval)을 영어 데이터셋으로만 학습(tuning)했을 때의 성능 저하 정도를 실험했습니다. 실험 결과, 영어로만 학습해도 다중언어 검색 성능이 저하되지 않을 뿐 아니라 오히려 향상될 수 있다는 것을 발견했습니다.

- **Technical Details**: 이번 연구에서는 최신 다중언어 임베딩 모델(multilingual embedding model)을 검색 시스템의 기초로 사용했습니다. 특히, intfloat/multilingual-e5-small와 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 모델을 활용했습니다. 학습 데이터로는 MSMARCO Triplets, ARXIV subset, XNLI 다중언어 데이터셋을 사용했습니다. 문장 유사도 비교 시 anchor, positive, negative의 삼중항(triplet)을 사용해 대조 학습(contrastive learning)을 실시했습니다.

- **Performance Highlights**: 기본 모델보다 영어 데이터셋으로 학습된 모델의 성능이 높았으며, 특히 MSMARCO와 ARXIV 데이터셋에서 긍정 문서가 앵커와 더 가까워지는 비율이 증가했습니다. 또한, XNLI 데이터셋에서도 언어 간 의미 일치(entailment) 문장 쌍이 중립(neutral) 또는 모순(contradiction) 문장 쌍보다 더 가까운 경우의 비율이 향상되었습니다.



### FineSurE: Fine-grained Summarization Evaluation using LLMs (https://arxiv.org/abs/2407.00908)
Comments:
          Accepted at ACL 2024 (main, long)

- **What's New**: 이번 논문에서는 큰 언어 모델(LLM)을 활용한 세부 평가 방법인 FineSurE를 제안합니다. 이는 요약 과제에 맞춤형으로 설계된 평가자입니다. 기존의 ROUGE와 같은 전통적 방법이 사람의 판단과 잘 일치하지 않는 문제를 해결하고자 합니다. 새로운 FineSurE는 신뢰성(faithfulness), 완전성(completeness), 간결성(conciseness)이라는 다차원적 평가 기준을 사용합니다.

- **Technical Details**: FineSurE는 요약 문장의 세부적인 평가를 위해 설계되었습니다. 주된 절차는 크게 두 가지로 나뉩니다. (1) 먼저 각 요약 문장에서 특정 사실성을 확인하는 fact checking이 있으며, (2) 그 다음으로는 요약 문장에서 주요 사실(keyfacts)을 정렬하여 연관성을 파악하는 keyfact alignment입니다. 이를 통해 요약의 각 문장과 주요 사실을 정렬하고 오류를 자동으로 분류합니다. 평가의 정확도를 높이기 위해 여러 오픈 소스 및 독점 LLM을 비교 분석하였습니다.

- **Performance Highlights**: FineSurE는 기존의 SOTA 방법들인 NLI-, QA-, LLM-기반의 방법과 비교하여 특히 완전성과 간결성 측면에서 향상된 성능을 보였습니다. 이는 요약의 질을 평가할 때 사람의 판단과 더 높은 상관관계를 보였습니다.



### How to Leverage Digit Embeddings to Represent Numbers? (https://arxiv.org/abs/2407.00894)
- **What's New**: 이번 연구에서는 수학적인 선행 지식을 활용하여 숫자 임베딩(embedding)을 집계하는 방법을 탐구하고, 이를 변환기 모델(transformer model)에 명시적으로 통합하는 방법을 제안합니다. 숫자를 나타내기 위해 자릿수 임베딩을 활용하는 기존의 방법들은 특정 한계가 있지만, 이를 명시적으로 집계하여 모델 성능을 향상시킬 수 있다는 가설을 검증하고자 합니다.

- **Technical Details**: 숫자를 자릿수 단위로 분해하여 임베딩하는 방식을 사용하되, 수학적인 선행 지식을 활용해 자릿수 임베딩을 명시적으로 집계하는 방법을 제안합니다. 이를 위해 입력 임베딩에 특별 토큰(special token)을 추가하거나, 올바른 예측을 향상시키기 위한 추가 손실 함수(loss function)를 도입했습니다. 이 방법은 기존에 사전 학습된 모델(pretrained model)에도 쉽게 통합될 수 있으며, 코드 변경이 거의 필요하지 않습니다.

- **Performance Highlights**: 제안된 방법은 소규모 모델에서도 성능 향상을 보여주었으며, 대규모 모델에서도 더욱 큰 효과를 기대할 수 있습니다. 명시적인 자릿수 임베딩 집계 방법은 모델의 크기와 사전 학습 데이터의 품질에 따라 성능에 영향을 미치며, 향후 이 접근 방식을 더욱 발전시킬 수 있는 가능성이 큽니다.



### MoE-CT: A Novel Approach For Large Language Models Training With Resistance To Catastrophic Forgetting (https://arxiv.org/abs/2407.00875)
Comments:
          13 pages, 2 figures

- **What's New**: 이번 연구는 다언어 대형 언어 모델(LLMs)의 성능 향상을 위한 새로운 MoE-CT (Mixture of Experts-Continual Training) 아키텍처를 제안합니다. 기존의 다언어 학습 접근법은 고자원 언어의 성능을 희생시키는 경향이 있었지만, MoE-CT는 기본 모델의 매개변수를 동결(freeze)하여 이러한 문제를 해결합니다. 이로 인해 고자원 언어의 성능을 유지하면서도 저자원 언어의 능력을 크게 향상시킬 수 있습니다.

- **Technical Details**: MoE-CT 아키텍처는 Mixture of Experts(MoE) 모듈을 채택하여 각 전문가가 특정 작업이나 기능 서브스페이스를 학습합니다. 기본 모델의 매개변수를 동결하는 동시에, 새로운 전문가 네트워크를 통해 다언어 학습 능력을 확장합니다. 이를 통해 원본 데이터 없이도 기본 모델의 지식을 보존하고, 새로운 지식과 동적으로 결합하는 게이팅 메커니즘(gating mechanism)을 구현합니다.

- **Performance Highlights**: MoE-CT 접근법은 다언어 성능 벤치마크에서 기존의 Continual Training (CT) 및 Low-Rank Adaptation (LoRA) 방법을 능가하는 성과를 보였습니다. 또한, 모델의 원래 능력 보존 테스트에서 기존 CT 대비 높은 저항성을 나타내어, 망각 저항성 측면에서도 우수한 성능을 입증했습니다. QCWinograd, XCOPA, pasw-X, 그리고 다언어 이해 여부를 검증하기 위한 MMLU, C-Eval 등의 데이터셋을 사용한 실험 결과, 제안된 MoE-CT 방식이 다양한 규모의 LLM 모델에서도 탁월한 성능을 보여주었습니다.



### Roleplay-doh: Enabling Domain-Experts to Create LLM-simulated Patients via Eliciting and Adhering to Principles (https://arxiv.org/abs/2407.00870)
Comments:
          34 pages, 24 figures, 11 Tables

- **What's New**: 이번 연구에서는 민감한 상호작용이 필요한 분야에서 역할극(roleplay)을 지원하는 새로운 인간-LLM 협업 파이프라인인 Roleplay-doh를 개발했습니다. 특히, 초심 상담사가 숙련된 정신건강 지원자와 시뮬레이션할 수 있는 맞춤형 AI 환자를 만들기 위해 사용되었습니다. 이 과정에서 전문가의 질적 피드백을 자연어 규칙으로 변환하여 LLM이 역할극을 수행할 수 있도록 합니다.

- **Technical Details**: Roleplay-doh는 전문가의 질적 피드백을 수집하여 이를 원칙으로 변환하는 인간-LLM 협업 도구입니다. 이 원칙들은 LLM이 역할극에서 따르도록 설계된 자연어 규칙입니다. 초기 도구 설계에서는 전문가가 직관적으로 원칙을 설정하고, 이러한 원칙들이 역할극 반응을 생성하는 데 사용됩니다. 이어서, 원칙 준수 프롬프트 파이프라인을 도입하여 반응 품질과 원칙 준수도를 30% 개선하였습니다.

- **Performance Highlights**: 25명의 상담 전문가와 함께 진행한 사용자 연구에서 Roleplay-doh는 생성된 AI 환자가 실제 환자와 유사하게 나타나는 것으로 평가되었습니다. 원칙 준수 파이프라인을 통해 원칙 준수도와 대화 일관성이 기존 대비 각각 35%와 25% 향상되었습니다. 이 도구는 정신 건강 이외의 다양한 분야에서도 전문가 피드백을 통해 현실적인 LLM 시뮬레이션을 구축할 수 있을 것으로 예상됩니다.



### Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks (https://arxiv.org/abs/2407.00869)
- **What's New**: 새로운 연구는 언어 모델들이 잘못된(fallacious) 및 기만적인(deceptive) 논리를 생성하는 데 어려움을 겪는다는 것을 발견했습니다. 이를 이용한 새로운 '일탈 공격(jailbreak attack)' 방법을 제안하며, 이를 통해 모델의 안전 메커니즘을 우회하여 악의적인 출력을 유도할 수 있다는 것을 보여줍니다.

- **Technical Details**: 해당 연구에서는 LLMs(Large Language Models)이 기만적이면서도 사실인 절차를 생성하도록 요청하여 모델의 안전 메커니즘을 우회합니다. LLMs는 일반적으로 허위적인(고의적으로 틀린) 절차를 무해한 것으로 간주하기 때문에, 이를 활용하면 사실과 해롭지만 허위적인 절차를 생성할 수 있습니다. 이를 Fallacy Failure Attack(FFA)라고 명명하고, 이 방법을 통해 GPT-3.5-turbo, GPT-4, Google Gemini-Pro, Vicuna-1.5, LLaMA-3 등의 모델에서 실험을 진행했습니다.

- **Performance Highlights**: FFA는 특히 GPT-3.5, GPT-4 및 Vicuna-7b 모델에 가장 효과적이며, 이 모델들이 더 많은 해로운 출력을 생성하게 만들었습니다. 또한 기존의 방어 방법들이 FFA에 대해 효과적이지 않음을 발견하여, 이러한 보안 위협에 대한 긴급한 대응의 필요성을 강조합니다.



### Towards Robust Speech Representation Learning for Thousands of Languages (https://arxiv.org/abs/2407.00837)
Comments:
          20 pages

- **What's New**: XEUS는 4057개의 언어를 지원하는 Cross-lingual Encoder for Universal Speech입니다. 이 모델은 100만 시간 이상의 데이터를 활용하여 학습되었으며, 새로운 7400+ 시간의 코퍼스를 포함하고 있습니다.

- **Technical Details**: XEUS는 E-Branchformer 인코더로, 다양한 녹음 조건을 처리할 수 있도록 설계되었습니다. SSL(masked prediction) 방법에 새로운 'dereverberation objective'를 추가하여 다양한 다국어 음성 데이터의 조건을 개선했습니다. 이 모델은 4057개의 언어를 아우르는 데이터로 학습되었습니다.

- **Performance Highlights**: XEUS는 여러 벤치마크에서 기존의 SOTA 모델을 능가하거나 유사한 성능을 보였습니다. 특히 ML-SUPERB 벤치마크에서는 MMS 1B와 w2v-BERT 2.0 v2보다 각각 0.8%와 4.4% 더 높은 성능을 기록했습니다.



### NAIST Simultaneous Speech Translation System for IWSLT 2024 (https://arxiv.org/abs/2407.00826)
Comments:
          IWSLT 2024 system paper

- **What's New**: 이번 논문은 IWSLT 2024 평가 캠페인의 동시 트랙에 대한 NAIST의 제출 모델을 설명합니다. 여기에는 영어-독일어, 일본어, 중국어 음성-텍스트 번역과 영어-일본어 음성-음성 번역이 포함됩니다. 이번 연구에서는 HuBERT와 mBART라는 두 가지 사전 학습된 언어 모델을 결합한 다국어 엔드투엔드 음성-텍스트 번역 모델을 개발했습니다. 특히 두 가지 디코딩 정책인 Local Agreement(LA)와 AlignAtt를 적용했습니다.

- **Technical Details**: 연구에서는 LA와 AlignAtt 두 가지 디코딩 정책을 사용하여 모델을 훈련했으며, 최종 제출된 모델은 이전 모델에서 더 나은 성능을 보였던 LA 정책을 채용했습니다. 영어-일본어 음성-음성 번역 방법으로는 위의 음성-텍스트 모델과 점진적 TTS(Text-to-Speech) 모듈을 결합하여 사용했습니다. 이 모듈은 음소 추정 모델, 병렬 음향 모델, 병렬 WaveGAN vocoder를 포함합니다. 또한, TTS 모듈의 성능을 향상시키기 위해 Transformer 아키텍처와 AlignAtt 정책을 도입했습니다.

- **Performance Highlights**: 결과적으로 업그레이드된 TTS 모듈이 시스템 성능 향상에 기여했음을 보여줍니다. LA 기반 모델은 주어진 지연 시간 제약 내에서 AlignAtt 기반 모델보다 더 나은 품질을 보여주었으며, AlignAtt 정책은 낮은 지연 시간 설정에서 더 좋은 성능을 발휘했습니다.



### Step-Controlled DPO: Leveraging Stepwise Error for Enhanced Mathematical Reasoning (https://arxiv.org/abs/2407.00782)
- **What's New**: 이번 연구에서는 수학적 추론 과정을 자동으로 제공하는 Step-Controlled DPO(SCDPO)를 제안하였습니다. SCDPO는 특정 단계에서 오류를 시작하는 부정적인 샘플을 생성하여, 모델이 추론 오류를 이해하고 정확한 추론 단계를 출력할 수 있도록 돕습니다.

- **Technical Details**: Step-Controlled DPO(SCDPO)는 모델이 수학 문제를 해결하면서 발생하는 단계별 오류를 학습하도록 하는 방법론입니다. 이 과정은 두 단계로 구성됩니다. 첫 번째 단계에서는 모델을 사용하여 중간 단계에서 온도가 높은 softmax 함수를 통해 오류가 있는 샘플을 생성하고, 두 번째 단계에서는 이 샘플과 정답 샘플을 사용하여 DPO(training)를 수행합니다. 이를 통해 모델은 세부적인 추론 능력을 학습하게 됩니다.

- **Performance Highlights**: SCDPO를 사용하여 InternLM2-20B 모델을 미세 조정한 결과, GSM8K에서 88.5%, MATH에서 58.1%의 높은 점수를 기록하였습니다. 이는 다른 오픈소스 LLM들을 능가하는 성능을 나타내며, SCDPO의 잠재력을 보여줍니다.



### Characterizing Stereotypical Bias from Privacy-preserving Pre-Training (https://arxiv.org/abs/2407.00764)
- **What's New**: 이번 연구는 Differential Privacy (DP)를 이용해 언어 모델(Language Models, LMs)에 적용된 텍스트 비공개화가 전형적 연관성(stereotypical associations)에 미치는 영향을 조사했습니다. 연구 결과, 프라이버시 보장이 강화됨에 따라 전형적인 편향(stereotypical bias)은 대체로 감소하지만, 반드시 모든 사회적 도메인에서 편향이 균일하게 감소하지는 않는다는 사실을 밝히고 있습니다.

- **Technical Details**: DP는 원시 텍스트에 공간적 배열을 이용해 단어 임베딩(embedding) 공간에서 적용됩니다. 연구에서는 프라이버시 수준이 다른 편향된 문장을 포함한 텍스트로 BERT 모델을 학습시켰습니다. 이를 통해 DP가 적용된 텍스트가 언어 모델링 능력을 저하시킴으로써 편향을 상쇄할 수 있다는 가정을 테스트했습니다. 프라이버시 보장을 위해 그래디언트 업데이트 시 노이즈를 추가하는 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, 프라이버시가 강화되면 전형적인 편향이 감소하는 경향을 보였습니다. 이는 언어 모델링 능력이 저하된 LMs가 덜 전형적인 연관성을 갖는다는 이전 연구 결과와 일치합니다. 그러나 특정 속성과 관련된 비편향은 안정성, 증폭 및 쇠퇴의 다양한 추세를 보여주었습니다. 따라서 프라이버시 보호 언어 모델을 배포할 때는 신중한 편향 측정이 필요합니다.



### A Comparative Study of Quality Evaluation Methods for Text Summarization (https://arxiv.org/abs/2407.00747)
Comments:
          The paper is under review at Empirical Methods in Natural Language Processing (EMNLP) 2024. It has 15 pages and 4 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 기반으로 한 새로운 텍스트 요약 평가 방법을 제안합니다. 기존의 자동화된 평가 지표(ROUGE, BERTScore 등)와 사람에 의한 평가를 비교한 결과, LLM 기반 평가가 인간 평가와 더 잘 맞는다는 것을 발견했습니다. 따라서, LLM을 이용한 요약 평가 및 개선 프레임워크를 제안하며 이 방법이 커뮤니티에서 큰 관심을 받을 것이라고 주장합니다.

- **Technical Details**: 본 연구는 최신 상태의 예술(SOTA) 요약 모델 7종을 평가했습니다. 평가 실험은 특허 문서 데이터를 사용하여 광범위하게 수행되었으며, 자동화된 지표(ROUGE, BERTScore, SummaC 등)는 일관성이 부족하고 인간 평가와 일치하지 않는다는 결과를 보였습니다. 반면, LLM 기반 평가는 인간 평가와 높은 일치를 보였습니다. 이 연구는 LLM을 활용한 평가 프레임워크를 제안함으로써 요약의 품질을 자동으로 평가하고 개선할 수 있는 방법을 시사합니다.

- **Performance Highlights**: LLM 기반 평가 방법이 ROUGE-2, BERTScore 등의 기존 자동화된 평가 지표보다 인간 평가와의 일치도가 더 높음을 실험 결과로 보여줍니다. 또한, 제안된 LLM 프레임워크는 요약 품질을 자동으로 평가하고 향상시키는 데 유용한 도구로서의 잠재력을 가지고 있습니다.



### Locate&Edit: Energy-based Text Editing for Efficient, Flexible, and Faithful Controlled Text Generation (https://arxiv.org/abs/2407.00740)
Comments:
          18 pages, 2 figures

- **What's New**: Locate&Edit (L&E)라는 새로운 에너지 기반 접근법을 소개합니다. 이 방법은 기본 언어 모델(base LM)의 텍스트 출력을 오프더셸프 에너지 모델(off-the-shelf energy models)을 사용하여 수정하는 방식입니다. 이 방법은 블랙박스 LMs에 적합하며, 기본 LM의 원본 생성 텍스트의 핵심 의미를 보존하면서 조절을 수행합니다.

- **Technical Details**: L&E는 먼저 기본 언어 모델로부터 텍스트를 생성하고, 에너지 모델을 사용하여 제약 조건(예: 유독성)과 관련된 스팬(span)을 찾아낸 후, 이 스팬을 더 적합한 대안으로 대체해서 편집합니다. MLM(masked language model)을 사용해 후보 토큰을 생성하고 에너지 함수를 사용하여 재랭크합니다. 이 방법은 블랙박스 LMs에서도 작동할 수 있으며, 아키텍처 종속이 없어서 다양한 오프더셸프 모델을 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, L&E는 기본 언어 모델 생성의 의미를 superior하게 보존하면서 빠른 속도를 자랑합니다. 독성 회피(toticity avoidance) 및 감정 제어(sentiment control) 작업에서, L&E는 낮은 독성 확률을 유지하면서도 기본 LM 출력의 95.4%를 보존하고 가장 빠른 속도를 기록했습니다. 세분화된 에너지 분포를 사용한 방법이 이진 분류기 에너지 모델에 비해 더 나은 제약 만족도를 달성했습니다.



### Large Language Models Struggle in Token-Level Clinical Named Entity Recognition (https://arxiv.org/abs/2407.00731)
Comments:
          AMIA 2024 Annual Symposium Proceedings

- **What's New**: 이번 연구는 현존하는 사유 LLMs와 오픈소스 LLMs의 성능을 비교하여 의료 텍스트에서 토큰 수준의 NER을 평가하려는 시도를 담고 있습니다. 특히 희귀질환과 관련된 콘텐츠를 중심으로 LLMs의 효과성에 대한 실험을 수행했습니다.

- **Technical Details**: 주요 실험 방법으로는 zero-shot prompting, few-shot prompting, retrieval-augmented generation (RAG), 그리고 instruction-fine-tuning을 활용했습니다. 본 연구에서는 LLaMA-2, Meditron, Llama2-MedTuned, UniversalNER 및 ChatGPT-3.5, ChatGPT-4와 같은 최신 모델들의 성능을 비교하였습니다.

- **Performance Highlights**: 실험 결과, 올바른 파인튜닝이 없는 경우 현지 LLMs는 토큰 수준의 NER 작업에서 어려움을 겪음이 밝혀졌습니다. 그러나 few-shot learning은 대부분의 LLMs 성능을 효과적으로 향상시켰으며, 특정 의료 모델(Llama2-MedTuned)이 ChatGPT-4보다 우수한 성능을 보였습니다. 이는 희귀질환 데이터에 대한 특별한 교육 없이도 우수한 성능을 낼 수 있음을 시사합니다.



### Scaling Technology Acceptance Analysis with Large Language Model (LLM) Annotation Systems (https://arxiv.org/abs/2407.00702)
Comments:
          This is a preprint of a paper accepted for the 32nd International Conference on Information Systems Development (ISD 2024), Gdansk, Poland

- **What's New**: 이 연구는 전통적인 설문조사 대신 대형 언어 모델(LLM, Large Language Model)을 활용하여 온라인 사용자 생성 콘텐츠를 분석하는 방법을 탐구하였습니다. 이 연구는 UTAUT(통합 기술 수용 및 사용 이론) 모델을 기반으로 리뷰를 구조화된 데이터로 변환하는 LLM 주석 시스템을 설계하고 검증하였습니다.

- **Technical Details**: LLM 주석 시스템은 디지털 리뷰와 댓글과 같은 비구조화된 텍스트 데이터를 분석하여 이를 구조화된 형태로 변환합니다. 연구는 일관성과 정확성을 검증하기 위해 두 가지 실험을 수행하였으며, 모델 온도를 낮출수록 일관성이 향상되는 것을 확인하였습니다. 또한, 인간 전문가 주석과 LLM 주석 간의 높은 일치도를 이루었습니다.

- **Performance Highlights**: LLM 주석은 전문가 간 일치도를 초과하는 성능을 보였으며, 특히 UTAUT 변수에 대해 높은 정확성을 나타냈습니다. 이는 LLM이 전통적인 설문조사를 대체할 수 있는 효과적인 도구임을 시사하며, 기술 설계와 수용 분석에 더 깊은 통찰력을 제공할 수 있는 가능성을 보여줍니다.



### HRDE: Retrieval-Augmented Large Language Models for Chinese Health Rumor Detection and Explainability (https://arxiv.org/abs/2407.00668)
- **What's New**: 이 논문은 중국 건강 루머 데이터셋이 부족한 상황을 해결하기 위해 HealthRCN이라는 1.12백만 개 이상의 데이터를 포함한 대규모 건강 루머 데이터셋을 구축했습니다. 또한, 이 데이터셋을 기반으로 중국 건강 루머 탐지 및 설명 가능성을 위한 Retrieval-Augmented Large Language Models (HRDE)이라는 새로운 모델을 제안했습니다.

- **Technical Details**: HRDE 모델은 건강 정보 수집 및 저장(health information collection and storage), 건강 정보 검색(health information retrieval), 검색 정보 재배열(re-ranking), 그리고 루머 탐지 및 분석을 포함한 답변 생성(generating the rumor detection answer) 등 네 가지 주요 구성 요소로 이루어져 있습니다. 이 모델은 LLMs(large language models)를 활용하여 입력된 건강 정보가 루머인지 아닌지를 판단하고, 해당 판단을 설명하는 답변을 제공합니다.

- **Performance Highlights**: 평가 실험에서 HRDE 모델은 GPT-4-1106-Preview를 포함한 여러 모델을 비교하여 루머 탐지 정확성과 답변 품질에서 모두 우수한 성능을 보였습니다. HRDE는 평균 91.04%의 정확도와 91.58%의 F1 점수를 달성했습니다.



### Chain-of-Knowledge: Integrating Knowledge Reasoning into Large Language Models by Learning from Knowledge Graphs (https://arxiv.org/abs/2407.00653)
- **What's New**: Chain-of-Knowledge(CoK)라는 새로운 지식을 유추하는 프레임워크가 제안되었습니다. 이 프레임워크는 데이터셋 구축과 모델 학습을 모두 포함하므로, 대형 언어 모델(LLMs)에서 지식 유추능력을 향상시킬 수 있도록 돕습니다.

- **Technical Details**: CoK는 세 가지 주요 단계로 구성됩니다: 1) 규칙 탐사(rule mining)로 지식 그래프(KG)에서 조합 규칙을 추출, 2) 지식 선택(knowledge selection)으로 규칙에 맞는 삼중항(triples)을 확인, 3) 샘플 생성(sample generation)으로 삼중항을 자연어 샘플로 변환. 모델 학습 단계에서는 행동 복제(Behavior Cloning)로 인해 규칙 과적합(rule overfitting)이 발생할 수 있음을 발견하여 이에 대한 해결책으로 인간의 시도와 오류(trial-and-error) 메커니즘을 추가하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 CoK의 유효성을 검증하였으며, 익명화된 설정과 일반 설정 모두에서 높은 성능을 나타냈습니다. CoK는 지식 유추뿐 아니라 다양한 일반 추론 벤치마크에서도 효과를 보였습니다.



### LegalTurk Optimized BERT for Multi-Label Text Classification and NER (https://arxiv.org/abs/2407.00648)
- **What's New**: 이번 연구에서는 BERT 모델을 법률 터키어 분야에 맞추어 개선한 새로운 전처리(pre-training) 접근 방식을 소개합니다. 특히 이름 엔티티 인식(NER) 및 다중 라벨 텍스트 분류와 같은 다운스트림 작업에 집중하여 성능을 비교했습니다. 우리의 수정된 모델은 원본 BERT 모델에 비해 두 가지 작업 모두에서 상당한 성능 향상을 보여주었습니다.

- **Technical Details**: BERT 기반 구성에는 변화 없이, 성능 향상을 위해 세 가지 주요 전략을 추구했습니다: next sentence prediction (NSP)을 sentence order prediction (SOP)으로 대체, NSP 완전 제거, Masked Language Model (MLM)과 TF-IDF(역문서빈도) 결합. 우리의 혁신적인 접근 방식에서는 MLM을 위한 토큰의 10%를 랜덤 선택 대신 높은 TF-IDF 값을 가지는 토큰으로 대체했습니다. 또한, 원래 MLM 규칙 내에서 다양한 마스킹 전략을 구현했습니다. 50MB의 법률 터키어 말뭉치를 사용하여 제안된 모델들을 처음부터 전처리했습니다.

- **Performance Highlights**: 수정된 전처리 접근 방식을 평가하기 위해 모든 사용자 정의 모델을 미세 조정(fine-tune)했고, 원본 BERT 모델과 성능을 비교했습니다. 수정된 접근 방식은 NER과 다중 라벨 텍스트 분류 작업 모두에서 원본 BERT 모델에 비해 상당한 성능 향상을 보여주었습니다. 실험 결과, 우리의 혁신적인 접근 방식은 작은 말뭉치로 전처리되었음에도 불구하고 BERTurk 모델과 경쟁할 수 있음을 입증했습니다.



### A Collocation-based Method for Addressing Challenges in Word-level Metric Differential Privacy (https://arxiv.org/abs/2407.00638)
Comments:
          13 pages, 2 figures, 9 tables. Accepted to PrivateNLP 2024

- **What's New**: 이번 연구에서는 Metrics Differential Privacy(MDP)의 기존 단어 수준 접근 방식을 향상시키기 위해 단어와 문장 수준 사이의 결합된 어구(corrloations) 수준에서 작동하는 새로운 방법을 제안합니다. 단어 임베딩(word embedding) 공간에서 작동하는 기존의 단어-수준 MDP 방식들은 의미론적으로 일관된 텍스트 출력을 생성하는 데 실패하는 경우가 많았고, 문장 또는 문서 수준의 적용에는 한계가 있었습니다. 이 연구에서는 단일 단어가 아닌 n-그램(n-gram)을 변형함으로써 의미적 일관성과 가변적인 길이를 갖는 출력물을 생산하는 방법을 제시합니다.

- **Technical Details**: 이 접근 방식은 자주 발생하는 단어 그룹을 기반으로 한 임베딩 모델을 구축하여 단어와 함께 bi-gram 및 tri-gram 콜로케이션을 포함합니다. 이를 통해 변형된 출력물이 더 높은 의미적 일관성을 갖게 됩니다. 특히, 두 개의 제안된 콜로케이션 추출 알고리즘과 결합된 콜로케이션 임베딩 모델을 사용하여 MDP 변형을 수행합니다. 이에 따라 전체 변형 횟수를 줄이고, 프라이버시 예산을 절약할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 유틸리티와 프라이버시 테스트에서 유틸리티가 향상되고 다양한 비밀화 전략 하에서 비교할만한 프라이버시가 보장됨을 보여줍니다. 또한, 콜로케이션 기반 임베딩 모델을 사용함으로써 기존의 단어-수준 접근 방식보다 더 효율적이고 일관된 결과를 얻을 수 있습니다. 연구진은 이 알고리즘과 임베딩 모델을 오픈소스로 제공하여 향후 연구에 기여하고자 합니다.



### DP-MLM: Differentially Private Text Rewriting Using Masked Language Models (https://arxiv.org/abs/2407.00637)
Comments:
          15 pages, 2 figures, 8 tables. Accepted to ACL 2024 (Findings)

- **What's New**: 이번 연구에서는 Differential Privacy(DP)를 활용한 텍스트 프라이버타이제이션(text privatization) 방법으로 새로운 접근법을 제안합니다. 기존 방식들이 오토리그래시브 모델(autoregressive model)을 활용하여 프라이버타이즈된 텍스트를 생성했던 반면, 본 연구는 마스크드 랭귀지 모델(masked language model, MLM)을 사용한 DP-MLM을 소개합니다. 이 방법은 입력 텍스트를 토큰 단위로 컨텍스트 정보를 반영하여 재작성함으로써, 의미를 유지하면서도 프라이버시를 보장하는 텍스트를 생성합니다.

- **Technical Details**: DP-MLM은 BERT 기반의 MLM을 활용한 새로운 텍스트 재작성 방법입니다. 기존의 오토리그래시브 생성 모델들과는 달리, 인코더만을 사용하는 MLM을 사용하여 텍스트의 컨텍스트 정보를 반영하여 토큰 단위로 재작성합니다. 이를 통해 프라이버시를 보장하면서 의미를 유지하는 텍스트를 생성합니다. 이는 낮은 ε 수준에서도 유틸리티를 보존할 수 있는 점에서 특히 효과적입니다.

- **Performance Highlights**: DP-MLM은 기존의 SOTA(state-of-the-art) 프라이버시 보장 텍스트 재작성 방법들보다 더 나은 성능을 보입니다. 여러 벤치마크 테스트에서 유틸리티 및 프라이버시 측면에서 더 뛰어난 성과를 나타냈으며, 특히 낮은 토큰 단위 ε 수준에서 유리합니다. 이러한 성능은 MLM이 텍스트 프라이버타이제이션에서 유용한 도구임을 입증한 것입니다.



### MasonTigers at SemEval-2024 Task 10: Emotion Discovery and Flip Reasoning in Conversation with Ensemble of Transformers and Prompting (https://arxiv.org/abs/2407.00581)
- **What's New**: MasonTigers는 SemEval-2024 Task 10에 참가하여 영어 단일 언어 및 힌디-영어 코드-믹스된 대화에서 감정을 식별하고 감정 변화의 원인을 이해하는 작업에 집중했습니다. 이 과제는 세 가지 세부 과제로 구성되어 있으며, 각 과제는 감정 인식 및 이유를 분석하는 방법의 개발에 중점을 두고 있습니다.

- **Technical Details**: 첫 번째 세부 과제는 힌디-영어 코드-믹스된 대화에서 감정 인식(Emotion Recognition in Conversation, ERC)입니다. 두 번째와 세 번째 세부 과제는 각각 힌디-영어 코드-믹스된 대화와 영어 대화에서 감정 변화의 원인을 식별하는 감정 변화 추론(Emotion Flip Reasoning, EFR)입니다. MasonTigers 팀은 각 세부 과제에서 효과적인 감정 인식과 이유 분석 방법을 개발했습니다.

- **Performance Highlights**: 첫 번째 과제에서 0.78, 두 번째 및 세 번째 과제에서 각각 0.79의 F1 점수를 달성했습니다. 이 성과는 우리 방법의 효과를 입증할 뿐만 아니라 첫 번째와 세 번째 세부 과제에서 최고 순위를, 두 번째 과제에서 두 번째 순위를 확보했습니다.



### Answering real-world clinical questions using large language model based systems (https://arxiv.org/abs/2407.00541)
Comments:
          28 pages (2 figures, 3 tables) inclusive of 8 pages of supplemental materials (4 supplemental figures and 4 supplemental tables)

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 의료 결정에 필요한 근거를 요약하거나, 실제 데이터(Real-World Data, RWD)를 기반으로 새로운 연구를 생성하는 잠재력을 평가했습니다. 총 5개의 LLM 기반 시스템이 50개의 임상 질문에 답변하는 능력을 검토했으며, 9명의 독립된 의사들이 이 답변들을 관련성, 신뢰성, 실행 가능성 측면에서 평가했습니다.

- **Technical Details**: 일반 목적 LLMs (ChatGPT-4, Claude 3 Opus, Gemini Pro 1.5)은 관련성과 근거 기반으로 평가된 답변을 드물게 생성했으며(2% - 10%), 반면 검색 강화 생성(Retrieval Augmented Generation, RAG) 기반 및 agentic LLM 시스템은 질문의 24% (OpenEvidence)에서 58% (ChatRWD)에 대해 관련성과 근거 기반의 답변을 제공했습니다. 특히 agentic ChatRWD는 다른 LLMs와 비교했을 때 새로운 질문에 답변할 수 있는 유일한 모델이었습니다 (65% vs. 0-9%).

- **Performance Highlights**: 일반 목적 LLMs는 그 자체로는 사용하기에 부적합한 반면, RAG 기반 시스템과 새로운 증거 생성을 위한 agentic 시스템이 상호작용하여 개선된 환자 치료를 위한 관련 근거를 제공할 잠재력이 있음을 시사합니다.



### ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees (https://arxiv.org/abs/2407.00499)
Comments:
          13 pages, 9 figures, 6 tables

- **What's New**: 자연어 생성(NLG) 과제에서 불확실성 정량화(Uncertainty Quantification, UQ)는 여전히 해결되지 않은 과제입니다. 이 연구는 최근의 대형 언어 모델(LLMs)의 복잡한 특성을 고려하여, Conformal Prediction(CP)을 개방형 NLG 작업에서 블랙박스 접근을 통해 적응시키는 방법을 조사합니다. 본 연구는 자기 일관성을 활용한 샘플링 기반 불확실성 측정을 제안하고, CP 알고리즘의 설계에 올바름과 일치하는 불확실성 조건을 통합하여 Conformal 불확실성 기준을 개발합니다.

- **Technical Details**: CP는 모델 비종속적(model-agnostic)이고 통계적으로 엄격한 불확실성 추정을 제공합니다. 본 연구는 다중선택 질문 응답(MCQA) 설정에서 NLG 작업에 CP를 처음으로 적용하고, 샘플링한 생성물에 대한 의미 클러스터링을 통해 불확실성을 측정하는 새로운 접근법을 제안합니다. 이 접근법은 독립적이고 동일 분포(i.i.d.)의 소량의 보정 데이터를 사용하여 예측 세트를 구성하며, 사용자가 지정한 상한값을 근거로 교정된 예측 세트의 크기는 작게 유지됩니다.

- **Performance Highlights**: 실험 결과, 제안된 불확실성 측정 방식은 기존의 최첨단 방법들을 일반적으로 능가하는 것으로 나타났습니다. 6개의 LLM과 4개의 자유형 NLG 데이터셋을 사용하는 동안 모델의 올바름-피복률을 엄격하게 제어하면서도 예측 세트의 평균 크기가 작아져 실용적인 개방형 NLG 응용에 신뢰할 수 있는 보장을 제공하는 데 효율적임을 강조했습니다.



### LLMs-as-Instructors: Learning from Errors Toward Automating Model Improvemen (https://arxiv.org/abs/2407.00497)
- **What's New**: 이번 논문에서는 'LLMs-as-Instructors' 프레임워크를 소개합니다. 이 프레임워크는 고급 대형 언어 모델(LLMs)을 활용하여 작은 타깃 모델의 학습을 자동으로 향상시킵니다. 주요 아이디어는 오류 분석 이론에 영감을 받아 '오류로부터 학습하기( Learning from Errors)'와 '대조 학습으로부터 오류 학습하기(Learning from Error by Contrast)' 두 가지 전략을 사용하여 타깃 모델의 특정 오류를 분석하고 그에 맞는 학습 데이터를 생성하는 것입니다.

- **Technical Details**: 이 프레임워크는 네 가지 주요 단계로 구성됩니다: 데이터 선택(Data Selection), 결과 수집(Result Collection), 인스트럭터 분석 및 데이터 제공(Instructor Analysis and Data Supply), 타깃 모델 학습 및 평가(Target Model Training and Evaluation). 특히, '오류로부터 학습하기' 전략에서는 오직 잘못된 응답만을 분석하며, '대조 학습으로부터 오류 학습하기' 전략에서는 맞는 응답과 잘못된 응답 모두를 분석하여 오류를 더욱 심층적으로 이해합니다. 이를 통해 맞춤형 학습 데이터를 생성하고 자동 학습을 통해 타깃 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 여러 가지 오픈 소스 모델에 대한 실험 결과, 수학적 추론(GSM8k), 코딩 능력(HumanEval), 사실성 지식(MMLU) 등의 벤치마크에서 크게 향상된 성능을 보였습니다. 특히, 정밀하게 조정된 Llama-3-8b-Instruction 모델이 ChatGPT보다 우수한 성능을 나타냈습니다. 각 전략은 데이터의 특성에 따라 고유의 장점을 제공하며, 두 전략을 결합하여 더욱 균형 잡힌 성능 향상을 달성했습니다.



### PFME: A Modular Approach for Fine-grained Hallucination Detection and Editing of Large Language Models (https://arxiv.org/abs/2407.00488)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 세밀한 환각(hallucination) 유형을 분류하는 표준 프로세스를 소개하고, 이를 탐지하고 수정하기 위한 새로운 프레임워크인 PFME(Progressive Fine-grained Model Editor)를 제안합니다. PFME는 두 개의 협력 모듈로 구성되어 있습니다: 실시간 사실 검색 모듈(Real-time Fact Retrieval Module)과 세밀한 환각 감지 및 수정 모듈(Fine-grained Hallucination Detection and Editing Module)입니다.

- **Technical Details**: PFME는 문서 내 주요 엔티티를 식별하고 신뢰할 수 있는 데이터 소스에서 최신 사실적 증거를 검색하는 실시간 사실 검색 모듈로 시작합니다. 이어서 세부 정보에 따라 문서를 문장 단위로 나누고, 관련 증거 및 이전에 수정된 컨텍스트를 기반으로 각 문장의 환각 유형을 식별, 위치 지정, 수정하는 세밀한 환각 감지 및 수정 모듈로 처리합니다. 이 두 모듈을 통해 PFME는 세밀한 환각 탐지 및 수정 작업을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, PFME는 FavaBench와 FActScore에서 기존 방법보다 뛰어난 성능을 보였습니다. 특히 Llama3-8B-Instruct 모델을 사용할 때, 외부 지식 지원을 통한 세밀한 환각 감지에서 ChatGPT보다 8.7 퍼센트 포인트 향상된 성능을 보여주었습니다. 편집 작업에서는 FActScore-Alpaca13B와 FActScore-ChatGPT 데이터셋의 성능을 각각 16.2 퍼센트 포인트와 4.6 퍼센트 포인트 개선시켰습니다.



### It's Morphing Time: Unleashing the Potential of Multiple LLMs via Multi-objective Optimization (https://arxiv.org/abs/2407.00487)
- **What's New**: 이번 논문에서는 대형 언어 모델 병합을 최적화하기 위해 블랙 박스 다목적 최적화 알고리즘(black-box multi-objective optimization algorithms)을 사용하는 혁신적인 접근 방식을 제안합니다. 모델 병합의 목표는 서로 다른 작업에서 우수한 여러 모델을 단일 모델로 결합하여 각각의 소스 모델보다 더 나은 성능을 발휘하는 것입니다. 기존 방식은 인간의 직관과 맞춤형 전략에 크게 의존하며, 매개변수 충돌(parameter conflicts) 문제를 해결하는 데 한계가 있습니다. 이를 극복하기 위해 MM-MO 방법을 제안하여 최적의 병합 구성을 자동으로 검색하며, 여러 다양한 작업에서 예상 성능을 최적화 목표로 사용하여 중요한 델타 매개변수를 잃지 않고 매개변수 충돌 문제를 완화시킵니다.

- **Technical Details**: 논문에서는 모델 병합을 다목적 최적화 문제로 재구성합니다. 이 방법은 TIES-Merging과 DARE의 장점을 결합하여 견고한 모델 병합 전략을 수립합니다. MM은 무훈련 조건에서 다중 작업 성능을 목표로 합니다. 매개변수 공간을 최적화하는 기존 방법들의 제한점을 극복하기 위해 시험 데이터에 대한 성과 예측을 최적화 목표로 설정합니다. 특히 여러 작업 간의 매개변수 충돌을 줄이고자 설계되었습니다.

- **Performance Highlights**: 제안된 MM-MO 방법은 기존의 방법들과 비교 실험에서 일관되게 더 나은 성능을 발휘하였습니다. 목표로 설정하지 않은 작업 유형에서도 성능 개선이 확인되었으며, 이는 제안 방법이 특정 작업 유형에 대한 과적합(overfitting)이 아니라 모델의 전체적인 잠재력을 높인다는 것을 의미합니다. 또한 세 가지 다른 유형의 자연어 작업에 대한 광범위한 평가를 통해 방법의 효율성을 입증하였습니다. 결과적으로 MM-MO 방법을 통해 병합된 모델은 기존 방식에 비해 우수한 성능을 보였습니다.



### Towards Massive Multilingual Holistic Bias (https://arxiv.org/abs/2407.00486)
- **What's New**: 현재 자동 언어 생성의 세계에서는 다언어화가 진행됨에 따라 인구학적 편향(demographic biases)을 이해하고 평가하며 완화하는 것이 중요해졌습니다. 이에 따라, MASSIVE MULTILINGUAL HOLISTICBIAS (MMHB) 데이터셋 및 벤치마크의 초기 8개 언어를 선보입니다. 이 데이터셋은 약 600만 개의 문장을 포함하고 있으며, 13개의 인구학적 축(axes)을 대표합니다.

- **Technical Details**: MMHB 문장을 자동으로 확장하기 위해 제한된 인력 주석을 활용하는 자동 구성 방법론을 제안합니다. 이 접근법은 다언어 문장 구성에 플레이스홀더(placeholders)를 사용하고, 문장 패턴, 명사, 및 기술어(descriptors)를 독립적으로 번역하는 체계적인 방법을 포함합니다. 인간 번역을 결합하여 다양한 문장 변형을 동적으로 생성하고 인간 번역 작업량을 크게 줄입니다. 번역 과정은 영어 중심의 관점을 피하고 필요한 형태학적 변이를 모두 포함하도록 세밀하게 설계되었습니다.

- **Performance Highlights**: MMHB는 기계 번역 작업에서 성별 편향 및 추가적인 독성(additional toxicity)을 보고하는 데 사용됩니다. 성별 분석에서 MMHB는 (1) 남성 의미 문장이 여성 문장보다 평균적으로 약 +4 chrf 포인트 더 높은 성능을 보이며 성별 강건성이 부족하고, (2) 남성 형태로 과대 일반화(overgeneralize)하는 경향이 있어 남성 참조 문장 평가 시 평균적으로 +12 chrf 포인트 더 높게 보고됩니다. 또한, MMHB는 추가 독성을 최대 2.3%까지 유발합니다.



### Large Language Models for Power Scheduling: A User-Centric Approach (https://arxiv.org/abs/2407.00476)
- **What's New**: 이 논문은 사용자 주도 서비스와 개인 맞춤형 서비스에 중점을 둔 새로운 리소스 스케줄링 아키텍처를 소개합니다. 이는 기존 시스템 중심의 방법에서 사용자 중심의 접근 방식으로의 패러다임 전환을 제시합니다. 특히, 전기차(EV) 충전과 같은 사례에서 사용자의 음성 요청을 리소스 할당 벡터로 변환하는 세 가지 LLM 에이전트 구조를 제안합니다.

- **Technical Details**: 논문에서는 세 개의 LLM 에이전트를 설계합니다: 1) LLM intent recognition agent는 사용자의 요청을 최적화 문제(OP)로 번역합니다. 2) LLM OP parameter identification agent는 필요한 매개변수를 식별합니다. 3) LLM OP solving agent는 최적화 문제를 해결합니다. 이 아키텍처는 주로 Llama 3 8B 모델을 사용하여 평가되었으며, 여러 프롬프트 엔지니어링 시나리오로 테스트되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 아키텍처가 효율적으로 작동함을 보여주었습니다. 대표적인 EV 충전 요청을 다루는 데이터베이스를 기반으로 한 성능 분석에서는 인식/OP 분류 소음 수준 증가로 인한 성능 저하 가능성도 논의되었습니다. 모든 결과와 코드는 오픈 소스로 제공됩니다.



### Classifier identification in Ancient Egyptian as a low-resource sequence-labelling task (https://arxiv.org/abs/2407.00475)
Comments:
          Accepted to ML4AL 2024 (First Machine Learning for Ancient Languages Workshop)

- **What's New**: 고대 이집트(AE) 언어 시스템에서 자주 사용되는 상형문자 부호들을 식별하는 작업을 NLP 과제로 설정하여 접근함으로써, 초기 단계의 성과를 선보입니다. 이 작업은 최근에 시작된 iClassifier 프로젝트의 일환으로, 고대 및 현대 언어에서 분류자를 주석하고 분석하는 웹 기반 플랫폼입니다.

- **Technical Details**: 이번 연구는 Coffin Texts 코퍼스를 기반으로 일련의 sequence-labelling 신경망 모델을 구현하여 고대 이집트 분류자를 식별하려는 첫 걸음을 내딛었습니다. 데이터셋은 주로 Coffin Texts의 부문에서 가져왔으며, 이 코퍼스는 74106개의 데이터 포인트를 포함합니다. 그러나 대부분의 단어 형식이 반복되기 때문에 실제 데이터셋 크기는 8423 유형에 불과합니다. 구현 시, Manuel de Codage(MdC) 전사 시스템을 사용하였고, 분류자들은 UI에서 'semantic classifiers’와 'phonetic classifiers'로 태그되었습니다.

- **Performance Highlights**: 훈련 데이터가 적음에도 불구하고 신경망 모델은 고무적인 성능을 보였습니다. 6739개의 훈련 데이터 포인트와 842개의 개발 및 테스트 데이터 포인트를 사용해 모델을 훈련했습니다. 고대 이집트 상형문자 데이터의 특성상 매우 제한적인 자원 환경에서 작업이 이루어졌음에도 불구하고, 실질적인 성과를 보여줬습니다.



### BioKGBench: A Knowledge Graph Checking Benchmark of AI Agent for Biomedical Scienc (https://arxiv.org/abs/2407.00466)
- **What's New**: 이번 논문에서는 생명과학 분야의 인공지능 연구를 위한 새로운 벤치마크인 BioKGBench를 소개합니다. 이 벤치마크는 기존의 단순한 사실 확인 질문-답변 (QA)에만 국한되지 않고, LLMs의 '환상' 문제를 해결하기 위해 과학적 주장 검증과 구조화된 지식 그래프 질문-답변 (KGQA)를 이용한 문헌 이해 능력을 평가합니다. BioKGBench는 이러한 두 가지 원자적 능력을 결합한 새로운 에이전트 작업인 KGCheck를 통해 기존 대규모 지식 그래프 데이터베이스의 사실적 오류를 식별하는 것을 목표로 합니다.

- **Technical Details**: BioKGBench는 과학적 주장 검증과 구조화된 지식 그래프 질문-답변 (KGQA)을 통해 문헌 이해를 두 가지 원자적 능력으로 분리합니다. 이를 바탕으로 KGCheck 작업을 구성하며, KGQA와 도메인 기반의 Retrieval-Augmented Generation (RAG)를 사용해 기존 대규모 지식 그래프 데이터베이스의 사실적 오류를 식별합니다. 2,000개 이상의 데이터와 225개의 고품질 주석 데이터를 수집하여 기존 및 최신 에이전트들을 평가했습니다. BioKGBench의 데이터와 코드는 GitHub에서 공개됩니다.

- **Performance Highlights**: 최신 에이전트, 특히 일상 및 생의학 시나리오에서의 성능은 우리 벤치마크에서 낮은 성능을 보였습니다. 그러나 BKGAgent라는 단순하지만 효과적인 베이스라인을 도입하여 90개 이상의 사실적 오류를 발견하였습니다. 이는 에이전트들이 새로 발견할 수 있는 시나리오를 제공하며, 우리의 접근 방식이 유효함을 입증합니다.



### Polarization and Morality: Lexical Analysis of Abortion Discourse on Redd (https://arxiv.org/abs/2407.00455)
- **What's New**: 이 논문은 정치적 주제에서의 의견 차이가 언어 사용 패턴과 어떻게 연관되는지 조사합니다. Reddit의 낙태 토론 댓글을 분석하여 r/prolife와 r/prochoice 두 서브레딧 커뮤니티의 언어적 특징을 탐구했습니다.

- **Technical Details**: 도덕적 기초 이론(Moral Foundations Theory)을 고려하여 세 가지 방법으로 어휘 패턴을 분석했습니다. 첫째, 도덕적 기초 사전에 있는 어휘 항목의 비율 빈도를 계산했습니다. 둘째, 각 입장 그룹에서 자주 사용하는 단어들의 연어를 n-gram 모델로 분석했습니다. 마지막으로, 잠재 디리클레 할당(Latent Dirichlet Allocation)을 사용하여 코퍼스 데이터에서 주제 구조를 식별했습니다.

- **Performance Highlights**: 분석 결과, 도덕적 단어의 사용은 낙태에 대한 입장과 연관되었습니다. r/prolife에서는 'murder', 'harm', 'kill' 같은 위반 단어들이 높게 나타났고, r/prochoice에서는 'care', 'safe', 'peace', 'protect' 같은 긍정 단어들이 더 많이 사용되었습니다. n-gram 모델은 단어들의 문맥을 더 상세히 보여주었으며, 각각의 그룹이 특정 도덕적 가치를 강조하면서 상반된 주장을 형성하는 방법을 밝혀냈습니다.



### Self-Translate-Train: A Simple but Strong Baseline for Cross-lingual Transfer of Large Language Models (https://arxiv.org/abs/2407.00454)
- **What's New**: 이 연구에서는 Self-Translate-Train이라는 새로운 방법을 제안합니다. 이 방법은 대형 언어 모델(LLM)의 번역 능력을 활용하여 대상 언어로 합성된 학습 데이터를 생성하고, 모델을 자체 생성된 데이터로 미세 조정(fine-tuning)하는 것입니다. 이를 통해 여러 비영어권 언어에서 상당한 성능 향상을 달성했습니다.

- **Technical Details**: Self-Translate-Train 방법은 LLM을 사용하여 학습 데이터를 대상 언어로 번역하고, 생성된 합성 데이터를 이용해 모델을 훈련시킵니다. 기본 언어(예: 영어)의 학습 코퍼스 𝒟src을 LLM의 번역 기능을 이용하여 타깃 언어의 합성 코퍼스 𝒟tgt로 변환합니다. 이 합성 데이터를 원본 데이터에 추가하여 모델의 일반화 성능을 높입니다. 또한, 원본 데이터와 번역된 데이터를 페어링하여 코드스위치된 인스턴스를 생성하는 것도 실험했습니다.

- **Performance Highlights**: 질문 응답, 텍스트 페어 분류, 수학적 추론 등 여러 작업에서 Self-Translate-Train 방법을 평가했으며, 본 연구 결과 LLM의 다언어 능력을 올바르게 활용함으로써 기존 방법들보다 일관된 성능 향상을 달성할 수 있음을 확인했습니다.



### PerSEval: Assessing Personalization in Text Summarizers (https://arxiv.org/abs/2407.00453)
- **What's New**: 새롭게 제안된 설문 텍스트 요약 모델 평가에서 개인화 수준을 측정하는 새로운 메트릭인 PerSEval을 소개합니다. 이는 기존 EGISES 메트릭의 한계를 보완하며, 사용자 경험(UX)을 더 정확하게 반영하는 평가 방법입니다.

- **Technical Details**: PerSEval은 EGISES의 디자인 원칙을 기반으로 하며, 모델의 정확도가 EGISES 점수를 흐리지 않도록 하는 벌점 요소(Effective DEGRESS Penalty, EDP)를 도입합니다. 이는 모델의 최고 정확도와 실제 성능 간의 불일치를 반영합니다. PerSEval은 사람의 판단과 강한 상관관계를 보이는 신뢰성 있는 평가 메트릭으로, Pearson’s r = 0.73, Spearman’s ρ = 0.62, Kendall’s τ = 0.42의 결과를 보여줍니다.

- **Performance Highlights**: PerSEval은 10개의 최첨단 요약 모델을 PENS 데이터셋에서 벤치마킹하여 높은 신뢰성을 입증했습니다. PerSEval은 각 모델의 순위 안정성을 높이며, EGISES 기반 순위와는 독립적인 평가 결과를 보입니다. 또한, 정확도 기반의 순위는 개인화 요약기 평가에 충분하지 않으며 오히려 오해를 부를 수 있음을 보여주었습니다.



### A Recipe of Parallel Corpora Exploitation for Multilingual Large Language Models (https://arxiv.org/abs/2407.00436)
- **What's New**: 최근의 연구들은 병렬 말뭉치를(multilingual large language models, mLLMs) 활용하여 다국어 대형 언어 모델의 성능을향상시킬 수 있는 잠재력을 강조하고 있습니다. 본 연구는 병렬 말뭉치의 최적 활용 전략을 파악하기 위해 이들을 다양한 언어와 과업(Task)에서 mLLMs와 함께 결합한 결과를 분석합니다.

- **Technical Details**: 본 연구는 병렬 말뭉치의 품질 및 양, 훈련 목표(training objectives), 및 모델 크기(model size)가 mLLMs의 성능에 미치는 영향을 조사합니다. 주요 발견 사항으로는 (i) 불필요한 잡음이 섞인 번역을 필터링하는 것이 필수적이며, 언어 식별(language identification)과 짧은 문장 필터링은 거의 영향을 미치지 않는다는 점; (ii) 단지 10K개의 병렬 문장만으로도 대형 데이터셋과 비슷한 결과를 도출할 수 있다는 점; (iii) 번역(machine translation) 목표만을 사용하는 것이 다양한 훈련 목표와 그의 조합들 중에서 가장 좋은 결과를 낸다는 점; (iv) 큰 모델일수록 병렬 말뭉치로 인한 성능 향상이 크다는 점 등이 있습니다.

- **Performance Highlights**: 병렬 말뭉치의 품질을 유지하는 선별 작업이 매우 중요하며, 10K 문장만으로도 큰 성능 향상을 이끌어낼 수 있습니다. 또한 번역 목표만을 사용하는 것이 최고 성능을 보여주며, 큰 모델일수록 병렬 말뭉치를 더 잘 활용하여 성능 향상을 도모할 수 있습니다.



### Brevity is the soul of wit: Pruning long files for code generation (https://arxiv.org/abs/2407.00434)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 훈련에서 데이터 정제(data pruning)의 중요성을 강조하며, 코드 생성 도메인에 대한 미세 조정을 통해 임베딩 기반 방식과 휴리스틱 기반 방식의 성능을 비교했습니다. 특히, 이 연구는 단순한 휴리스틱 방법인 '긴 파일 제거'가 임베딩 기반 방법을 능가할 수 있음을 발견했습니다.

- **Technical Details**: 데이터 정제 방법에는 임베딩 기반(embedding-based), 휴리스틱 기반(heuristic-based), 분류기 기반(classifier-based)의 세 가지 주요 유형이 있습니다. 이 연구에서는 코드 생성 모델의 미세 조정(fine-tuning)을 위해 임베딩 기반 방법이 종종 문서 길이에 의해 혼동될 수 있음을 발견하였고, 이를 통해 긴 파일을 제거하는 단순한 휴리스틱 방식이 데이터 정제에 효과적임을 입증했습니다.

- **Performance Highlights**: 이 방법을 통해 훈련 효율성이 최대 두 배까지 개선되며, HumanEval 벤치마크에서 절대 성능이 3.5% 향상될 수 있음을 보여주었습니다. 그러나, 긴 파일에 대한 perplexity가 증가할 수 있어, 기존 코딩 벤치마크(HumanEval, MBPP)가 실제 사용 사례에 최적인지에 대한 의문을 제기합니다.



### eFontes. Part of Speech Tagging and Lemmatization of Medieval Latin Texts.A Cross-Genre Survey (https://arxiv.org/abs/2407.00418)
- **What's New**: 이번 연구에서는 중세 라틴어 텍스트의 자동 언어 주석을 위한 eFontes 모델을 소개합니다. 이 모델은 레마타이제이션(lemmatization), 품사 태깅(part-of-speech tagging) 및 형태소 특징 결정(morphological feature determination)을 목표로 합니다. Transformers 라이브러리를 사용하여 Universal Dependencies (UD) 말뭉치와 폴란드 중세 라틴어 eFontes 말뭉치에서 모델을 훈련하였습니다.

- **Technical Details**: Transformers 라이브러리를 기반으로 한 이 모델들은 이름 없는 예제의 맥락 없이 레마타이제이션, 품사 태깅 및 형태소 특징 태깅을 수행합니다. 훈련에는 Perseus, PROIEL, LLCT, ITTB, UDante와 같은 공개적으로 사용 가능한 UD 말뭉치뿐만 아니라, 2013년부터 구축된 폴란드 중세 라틴어 eFontes 말뭉치가 사용되었습니다. 이 말뭉치는 1000년에서 1550년 사이에 작성된 텍스트로, 시간, 장소 및 텍스트 유형에 대한 대표성이 신중하게 모니터링됩니다.

- **Performance Highlights**: 모델의 성능은 높은 정확도를 달성했습니다: 레마타이제이션에서 92.60%, 품사 태깅에서 83.29%, 형태소 특징 결정에서 88.57%의 정확도를 기록했습니다. 이 성과는 고품질 주석이 달린 말뭉치의 중요성을 강조하며, 앞으로 모델을 네임드 엔터티 인식(named entity recognition)으로 확장하는 방안을 제안합니다.



### Too Late to Train, Too Early To Use? A Study on Necessity and Viability of Low-Resource Bengali LLMs (https://arxiv.org/abs/2407.00416)
- **What's New**: 이번 연구에서는 벵골어 전용 대형 언어 모델(LLM)의 필요성을 탐구하였습니다. 연구 결과, 기존의 영어 중심 LLM이 논리적 추론 작업에서는 탁월한 성능을 보였으나, 벵골어 스크립트 생성 작업에서는 일관성이 떨어졌습니다. 이는 벵골어의 비효율적인 토큰화와 기계 번역 데이터셋의 편향성 때문입니다. 따라서 벵골어 전용 LLM의 필요성이 시사되었습니다.

- **Technical Details**: 연구에서는 다양한 벵골어 다운스트림 작업(번역, 요약, 패러프레이징, 질문응답, 자연어 추론)을 수행하며, LLaMA-3와 GPT-4 등의 기존 모델과 파인 튜닝된 엔코더-디코더 모델을 비교 분석했습니다. 또한, 벵골어 텍스트의 비효율적인 토큰화가 LLM 성능에 미치는 영향과 이로 인한 높은 계산 비용도 다루었습니다. 벵골어 스크립트에 대한 BPE 토큰화기가 영어에 비해 과도한 토큰화(평균 0.85 문자당 토큰)로 인해 모델 성능 저하와 비용 증가가 발생했습니다.

- **Performance Highlights**: LLM들은 벵골어 이해 작업(NLU)에서는 뛰어난 성능을 보였으나, 벵골어 생성 작업(NLG)에서는 불안정한 성능을 보였습니다. 특히, 기계 번역된 데이터셋은 특정한 문체를 편향적으로 나타내어 BLEU와 ROUGE 등의 메트릭에 영향을 미쳤습니다. 또한, LLM들이 벵골어 보상 모델링 작업에서 사람의 판단과 일치하지 않는 결과를 보였습니다. 최종적으로, 고품질의 사전 훈련 및 지침 조정 데이터셋의 부재로 인해 벵골어 전용 LLM 개발의 필요성이 강조되었습니다.



### Is It Really Long Context if All You Need Is Retrieval? Towards Genuinely Difficult Long Context NLP (https://arxiv.org/abs/2407.00402)
- **What's New**: 새로운 논문은 언어 모델(LMs)의 긴 문맥 처리를 위한 평가와 개발의 필요성을 강조하면서, 단순히 문맥의 길이로만 정의하는 현행 방식에 대한 문제를 제기합니다. 기존 연구들이 서로 다른 '긴 문맥' 작업을 동일하게 취급함으로 인해 정확한 평가가 어렵다는 것을 지적하며, 보다 정교한 분류체계를 제안합니다.

- **Technical Details**: 논문에서는 긴 문맥 작업을 '확산(Diffusion)'과 '스코프(Scope)'라는 두 가지 축으로 구분하여 설명합니다. '확산'은 필요한 정보를 문맥에서 찾는 난이도를 의미하며, '스코프'는 필요한 정보의 절대량을 나타냅니다. 이 두 가지 축을 기반으로 다양한 긴 문맥 작업의 난이도를 평가하고, 이를 통해 보다 정교한 분석이 가능하다고 주장합니다.

- **Performance Highlights**: 저자들은 긴 문맥 작업 중에서 필요한 정보가 많이 분산되고, 추출하기 어려운 경우가 충분히 탐구되지 않았다고 결론짓습니다. 이 논문은 긴 문맥 작업을 보다 정확하게 이해하기 위한 어휘 사용과 특징적 난이도에 대한 논의를 통해 연구 방향을 형성하는 것을 목표로 합니다. 이를 통해 향후 긴 문맥 처리 모델의 성능 평가 기준과 벤치마크의 설계가 보다 효과적으로 이루어질 수 있습니다.



### A Study on Effect of Reference Knowledge Choice in Generating Technical Content Relevant to SAPPhIRE Model Using Large Language Mod (https://arxiv.org/abs/2407.00396)
- **What's New**: 이 연구는 시스템 설계를 위한 SAPPhIRE 모델을 대형 언어 모델(Large Language Model, LLM)을 이용해 자동 생성하는 방법을 탐구합니다. 이는 두 부분으로 나뉜 연구의 첫 번째 부분으로, 과학 정보를 바탕으로 기술 콘텐츠를 생성하기 위한 방법론을 제시합니다.

- **Technical Details**: 본 논문은 'Retrieval Augmented Generating' 기법을 활용하여 LLM의 환각(hallucination)을 억제하는 방법을 연구합니다. SAPPhIRE 모델의 인과적 구조(SAPPhIRE model of causality)에 관련된 기술 콘텐츠를 정확하게 생성하기 위해 적절한 참조 지식을 선택하는 것이 중요함을 강조합니다.

- **Performance Highlights**: 이 연구의 결과는 주어진 기술 시스템에 대한 SAPPhIRE 모델을 자동으로 생성하는 소프트웨어 지원 도구를 구축하는 데 사용됩니다. 이를 통해 시스템 설계 시 기존의 여러 기술 문서에서 필요한 지식을 추출하는 과정을 간소화할 수 있습니다.



### Advancing Process Verification for Large Language Models via Tree-Based Preference Learning (https://arxiv.org/abs/2407.00390)
- **What's New**: 새로운 연구는 대규모 언어 모델(Large Language Models, LLMs)의 복잡한 추론 경로를 평가하는 데 보다 정교한 피드백 메커니즘을 도입했습니다. 기존의 이진 검증(binary verification) 접근법 대신, 우리가 제안한 'Tree-based Preference Learning Verifier (Tree-PLV)'는 최선 우선 탐색 알고리즘(best-first search algorithm)을 사용하여 개별 단계의 비교 데이터를 수집하고, 이를 통해 더 미세한 단계별 피드백을 제공합니다.

- **Technical Details**: Tree-PLV는 추론 트리를 구축하여 각 단계에서 상위 노드를 비교하고, 선호 학습(preference learning)을 통해 검증기를 훈련시키는 새로운 접근법입니다. 이 과정에서는 단계(level)에 따라 경로 간의 비교를 통해 데이터셋을 구성하고, 이를 순위 손실(ranking loss)을 사용하여 검증기를 훈련시킵니다. 이는 기존의 이진 분류(binary classification)보다 더 정교한 피드백을 가능하게 합니다.

- **Performance Highlights**: Tree-PLV는 여러 산술 및 상식 추론 과제에서 탁월한 성능을 보여주었습니다. 다음과 같은 정확도 향상이 있었습니다: GSM8K(67.55% → 82.79%), MATH(17.00% → 26.80%), CSQA(68.14% → 72.97%), 그리고 StrategyQA(82.86% → 83.25%). 특히, GSM8K 데이터로 훈련된 Tree-PLV는 더 도전적인 MATH 데이터셋에서도 견고한 일반화를 보여주었습니다.



### The Factuality Tax of Diversity-Intervened Text-to-Image Generation: Benchmark and Fact-Augmented Intervention (https://arxiv.org/abs/2407.00377)
- **What's New**: 이번 연구에서는 Text-to-Image(T2I) 모델의 '다양성 개입(diversity interventions)'이 역사적 인물을 생성할 때 사실적인 인구 분포를 왜곡시킬 수 있는지에 대해 조사하고 있습니다. 이를 해결하기 위해 DemOgraphic FActualIty Representation(DoFaiR) 벤치마크가 제안되었습니다. DoFaiR는 756개의 정확히 검증된 테스트 인스턴스를 통해 다양성 지시가 사실성을 어떻게 저해하는지 평가합니다.

- **Technical Details**: DoFaiR 벤치마크는 T2I 모델이 실제 역사적 사건에서 묘사되는 인구 분포를 얼마나 잘 재현하는지 측정합니다. 이를 위해 자동화된 평가 파이프라인을 활용하여 생성된 이미지의 인구 분포를 사실적인 분포와 비교합니다. 또한, Fact-Augmented Intervention(FAI)을 제안하여 LLM를 통해 검증된 사실 정보를 T2I 모델의 이미지 생성 지시에 통합합니다.

- **Performance Highlights**: 다양성 지향 지시가 DALLE-3의 이미지 생성에서 성별 및 인종 그룹의 수를 증가시키지만, 역사적으로 부정확한 인구 분포도 발생시킨다는 점이 밝혀졌습니다. FAI를 사용하면 인구 분포의 사실성을 크게 향상시킬 수 있으며, 인종 그룹의 사실성 정확도를 22% 이상, 주요 인종 사실성을 10% 이상 개선할 수 있습니다.



### How to Train Your Fact Verifier: Knowledge Transfer with Multimodal Open Models (https://arxiv.org/abs/2407.00369)
- **What's New**: 최근 소셜 미디어와 뉴스에서 잘못된 정보(misinformation)가 급증함에 따라, 이러한 정보의 효과적인 실시간 검증을 제공할 시스템의 필요성이 커지고 있습니다. 이 논문에서는 지속적인 업데이트 없이 기초 모델의 성능을 개선할 가능성을 테스트합니다. 우리는 지식 전이(knowledge transfer)를 통해 대규모 언어 모델이 생성한 설명을 이용해 검증하는 방법을 연구하였습니다.

- **Technical Details**: 이 연구는 6개의 잘못된 정보 검출과 사실 확인 데이터셋, 그리고 혐오 발언 탐지(hate speech detection)와 입장 감지(stance detection)와 같은 콘텐츠 관리 작업에 대한 모델을 함께 훈련하여 내부 및 외부 도메인 전이에 대해 분석하였습니다. 또한 GPT-3.5-turbo와 GPT-4 모델의 설명을 활용해 지식 전달의 영향을 탐구하였습니다. CNN과 수식 기반의 예측 모델(MFC)과 설명 모델(MEG)을 사용하여 다중 모드 입력의 진실성과 잠재적 해악을 예측하는 시스템을 구축하였습니다.

- **Performance Highlights**: 논문에서는 12개의 공공 벤치마크에서 검증을 진행하였습니다. 최근의 멀티모달 사실 확인 벤치마크인 Mocheg와 Fakeddit에서, 지식 전이 전략이 성능을 각각 1.7%와 2.9% 향상시켰습니다. 나아가, 잘못된 정보 검출 데이터로부터의 지식 전이가 혐오 발언 탐지 성능을 13.65% 향상시키는 결과를 보였습니다. 이러한 연구 결과는 콘텐츠 관리의 다른 중요한 영역에도 영향을 미칠 수 있습니다.



### Financial Knowledge Large Language Mod (https://arxiv.org/abs/2407.00365)
Comments:
          66 pages

- **What's New**: 금융 산업에서 인공지능의 적용이 현저히 증가하고 있습니다. 이를 위해, IDEA-FinBench라는 금융 지식을 평가하기 위한 새로운 벤치마크가 도입되었습니다. 두 번째로, IDEA-FinKER는 금융 도메인으로 일반 LLMs를 신속히 적응시키기 위해 개발된 프레임워크입니다. 세 번째로, IDEA-FinQA라는 LLM 기반의 금융 질문-응답 시스템이 소개되었습니다.

- **Technical Details**: IDEA-FinBench는 권위 있는 금융 전문가 시험 문제를 활용하여 LLMs의 금융 지식을 평가하는 벤치마크입니다. 이 벤치마크는 중국어와 영어로 제공되며, 16개의 금융 분야를 포괄합니다. IDEA-FinKER는 리트리벌 기반의 페이샷 러닝 메서드와 고품질 금융 지식 명령어를 제공하여 LLMs의 금융 도메인 적응을 지원합니다. IDEA-FinQA는 실시간 지식 주입과 외부 지식 기반의 사실 확인 메커니즘을 갖춘 금융 질문-응답 시스템으로, 데이터 수집기, 데이터 쿼리 모듈, 그리고 특정 기능을 수행하는 LLM 기반 에이전트들로 구성됩니다.

- **Performance Highlights**: 실험 결과, IDEA-FinKER는 LLMs의 금융 전문 능력을 현저히 향상시켰으며, 특히 중국 금융 시험 문제(CPA 등)에서 뛰어난 성능을 보였습니다. IDEA-FinQA는 실시간 지식 주입과 사실 강화 기능을 통해 다양한 데이터 수집 및 쿼리 방법론을 지원하여 최적의 작업별 응답을 제공합니다.



### From RAG to RICHES: Retrieval Interlaced with Sequence Generation (https://arxiv.org/abs/2407.00361)
Comments:
          18 pages, 3 figures, Preprint

- **What's New**: 이번에 소개하는 논문에서는 RICHES라는 새로운 접근법을 제안합니다. RICHES는 기존의 Retrieval Augmented Generation (RAG) 시스템과 달리 별도의 'retriever'와 'generator'를 필요로 하지 않으며, 이 둘을 단일 LLM(대형 언어 모델)과 하나의 디코딩 과정에서 통합하여 처리합니다. 이를 통해 다양한 새로운 과제를 프롬프트(prompt)를 변경하는 것만으로 적응할 수 있습니다. 추가적인 훈련 없이 어떠한 Instruction-tuned 모델과도 호환됩니다.

- **Technical Details**: RICHES는 기존의 증거 코퍼스(evidence corpus)로부터 직접적으로 내용물 또는 관련 자연어 검색 키를 디코딩하여 문서를 검색합니다. 이는 하나의 디코더 패스(decode pass)에서 단일 LLM을 사용하여 텍스트 생성과 검색을 교차시킵니다. RICHES의 접근법은 여러 번의 검색이 필요할 때도 일관된 텍스트 생성 절차와 병합하여 작동합니다. 예를 들어, 멀티-홉 질문 응답(Multi-hop Question Answering)에서 증거를 여러 문서에서 반복적으로 검색하여 지원 문장을 생성합니다.

- **Performance Highlights**: RICHES는 다양한 ODQA(Open Domain Question Answering) 작업에서 강력한 성능을 보입니다. 이는 특히 증거가 필요하거나 멀티-홉 QA가 요구되는 경우에서 두드러집니다. RAG 시스템과 비교하여 별도의 모델 전환 없이도, 하나의 디코딩 패스 내에서 고도의 정확성을 유지하며 추가 훈련 없이 Instruction-tuned 모델에서도 효과적으로 작동합니다.



### Korean Aspect-Based Sentiment Analysis via Implicit-Feature Alignment with Corpus Filtering (https://arxiv.org/abs/2407.00342)
Comments:
          13 pages, EMNLP 2024 (submitted), DMLR@ICML 2024

- **What's New**: 기존 문헌에서 한국어 식당 리뷰에 대한 Aspect-Based Sentiment Analysis (ABSA)에 관한 연구가 부족한 상황에서, 본 연구는 한국어와 같은 저자원 언어에서 효과적인 ABSA 프레임워크를 제안합니다. 이 프레임워크는 번역된 벤치마크와 라벨이 없는 한국어 데이터를 통합하여 예측 라벨을 최적화합니다.

- **Technical Details**: 연구는 번역된 데이터를 통해 모델을 미세 조정하고, 실제 한국어 NLI 세트를 의사 라벨링(pseudo-labeling)했습니다. 그런 다음 LaBSE와 MSP 기반 필터링을 적용하여 Aspect Category Detection 및 Polarization을 강화했습니다. 이중 필터링을 사용하여 실제 한국어 데이터 세트를 NLI 태스크로 변환했으며, 이는 저자원 언어에서 높은 효율을 보였습니다.

- **Performance Highlights**: 이 프레임워크는 한국어 ABSA에서 약 3%의 F1 점수 및 정확도 차이를 보이며 긍정적인 결과를 얻었습니다. 추가 데이터 주입 파이프라인을 통해 높은 자원 데이터를 활용하고, 커뮤니티 내에서 실질적인 모델 구축을 목표로 합니다. 우리가 작성한 데이터셋과 코드도 공개되어 한국어 ABSA 연구에 기여하고자 합니다.



### Iterative Data Augmentation with Large Language Models for Aspect-based Sentiment Analysis (https://arxiv.org/abs/2407.00341)
Comments:
          Work in process

- **What's New**: 새로운 논문은 Aspect-based Sentiment Analysis (ABSA)의 성능 향상을 위해 Iterative Data augmentation 방법인 IterD를 제안합니다. 이 방법은 기존 데이터 증강(Data Augmentation, DA) 방법에서 발생하는 유창성, 일관성, 그리고 다양성 부족 문제를 해결하며, 실세계 시나리오에도 쉽게 적용할 수 있습니다.

- **Technical Details**: IterD는 비감시(sent. corpus) 문장 코퍼스에서 시작하여 대형 언어 모델(LLM)의 강력한 기능을 활용하여 유창하고 다양한 합성 라벨 데이터를 반복적으로 생성합니다. IterD는 세 단계로 구성됩니다: 1) Aspect 추출 및 확장, 2) 유사 데이터 생성, 3) 평가 및 필터링. 이 과정에서 비감시 데이터만을 활용해 고품질 데이터를 생성합니다.

- **Performance Highlights**: 4개의 널리 사용되는 ABSA 벤치마크(Laptop14, Restaurant14, Restaurant15, Restaurant16)에서 IterD를 테스트한 결과, 5개의 기본 ABSA 모델 모두에서 일관되고 큰 성능 향상을 보여주었습니다. 또한 IterD가 기존의 수작업으로 라벨링된 데이터 성능과 비교했을 때 동등하거나 더 나은 성능을 발휘하며, 다른 DA 방법에 비해 명확한 우위를 점했습니다.



### LLM-Generated Natural Language Meets Scaling Laws: New Explorations and Data Augmentation Methods (https://arxiv.org/abs/2407.00322)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 생성한 자연 언어(LLMNL)와 인간이 생성한 자연 언어(HNL) 간의 차이점을 분석하고, 이를 통해 향후 LLM의 확장을 위한 기초를 마련했습니다. 또한, 새로운 데이터 증강 방법인 ZGPTDA를 도입하여 few-shot 텍스트 분류에서의 성능 향상을 꾀했습니다.

- **Technical Details**: 연구팀은 먼저 LLMNL과 HNL이 실제로 얼마나 유사한지를 평가하기 위해 Mandelbrot's law를 사용한 스케일링 법칙(scaling laws)을 도입했습니다. 이를 통해 HNL이 LLMNL보다 약간의 복잡성 우위를 가지고 있음을 확인했습니다. 새로운 ZGPTDA 메소드는 fuzzy computing 메커니즘을 활용하여 scaling laws에 따른 GPT-4 증강 데이터를 선별하여 few-shot 텍스트 분류에 활용합니다.

- **Performance Highlights**: 실험 결과, ZGPTDA 방식은 Bert와 RoBerta의 F1 스코어를 평균 7-10% 개선시켰으며, 최근의 AugGPT 및 GENCO 방식보다 DeBerta에서 약 2% 높은 정확도를 보였습니다. 또한 Hilberg's law와 Taylor's law가 텍스트 분류에서 더 많은 이점을 제공할 수 있음을 발견했습니다.



### LiteSearch: Efficacious Tree Search for LLM (https://arxiv.org/abs/2407.00320)
- **What's New**: 이번 연구에서는 기존의 나무 탐색 알고리즘(tree search algorithms)들이 복잡한 수학적 추론 작업에서 LLM(대형 언어 모델)의 성능을 크게 향상시킬 수 있다는 것을 확인했습니다. 하지만, 이러한 알고리즘은 탐색 과정에서 많은 계산 자원이 필요해 실제 응용에서 어려움을 겪고 있습니다. 이를 해결하기 위해, 본 연구는 동적 노드 선택과 노드 수준의 탐색 예산 계산을 갖춘 새로운 가이드드 트리 탐색 알고리즘을 제안합니다. 이 알고리즘은 최종 답안으로 향하는 탐색 진행 상황(history)과 별도의 단계별 주석 없이 훈련된 가치 네트워크(value network)의 지도를 고려해 가장 유망한 트리 노드를 선택하고, 계산 예산 내에서 그 노드를 확장합니다.

- **Technical Details**: 본 연구에서 제안하는 동적 가이드드 트리 탐색 알고리즘은 가치 점수를 지침으로 사용하여 다음 동작을 수행하기 위해 가장 유망한 노드를 선택하고, 동적으로 계산된 예산 크기 내에서 확장합니다. 이 알고리즘은 선택(selection) 및 확장(expansion) 작업을 반복하여 최종 궤적이 예상 품질 점수에 도달하거나 최대 반복 횟수를 초과할 때까지 진행됩니다. 특히, 각 노드의 계산 예산은 그 가치 점수에 반비례하며, 가치 점수가 높은 노드에는 적은 계산 자원을 할당하여 불필요한 계산을 방지합니다.

- **Performance Highlights**: 실험 결과, GSM8K 및 TabMWP 데이터셋에서 우리의 접근 방식은 다른 기법들과 비교하여 경쟁력 있는 성능을 제공하면서도 계산 비용을 크게 줄였습니다. 특히, 기존 방법보다 약 5배의 계산 비용 절감을 달성했습니다. 각 구성 요소의 유용성을 확인한 상세 분석도 여러 설정에 대해 실용적인 옵션을 제공합니다.



### From Local Concepts to Universals: Evaluating the Multicultural Understanding of Vision-Language Models (https://arxiv.org/abs/2407.00263)
Comments:
          Under peer review

- **What's New**: GlobalRG라는 새로운 벤치마크가 도입되었습니다. 이 벤치마크는 비전-언어 모델(Vision-Language Model, VLM)의 문화적 포용성을 평가하기 위한 목적으로 개발되었으며, 특정 문화에서의 개념들을 이미지 기반으로 평가하는 두 가지 작업을 포함하고 있습니다: 'Retrieval across universals'와 'Cultural visual grounding'. 첫 번째 작업은 50개국에서 '결혼식'과 같은 보편적 개념에 대한 문화적으로 다양한 이미지를 검색하는 것이며, 두 번째 작업은 특정 문화에서 유래된 개념들을 이미지 내에서 식별하는 것입니다.

- **Technical Details**: GlobalRG 벤치마크는 두 가지 핵심 작업을 포함합니다. 첫째, 보편적 개념에 대한 문화적으로 다양한 이미지를 검색하는 'retrieval across universals' 작업이며, 이는 50개국에서 'breakfast', 'wedding' 같은 개념을 대상으로 합니다. 두 번째 작업은 'cultural visual grounding'으로, 이는 15개국에서 특정 문화적 개념(e.g., 멕시코의 molinillo)을 이미지 내에서 찾는 것입니다. 추가적으로 'precision@k'와 'diversity@k' 메트릭을 도입하여 모델의 검색 정확도와 문화적 다양성을 측정합니다.

- **Performance Highlights**: 7개의 모델을 'retrieval' 작업과 5개의 모델을 'grounding' 작업에서 평가한 결과, 문화 간 성능 차이가 크게 나타났습니다. 북미와 유럽에서의 성능이 동아시아와 동남아시아에 비해 월등히 높았습니다. 이는 비전-언어 모델들이 특정 문화에 치우친 성능을 보여주고 있음을 의미합니다. 예를 들어, 결혼식 이미지 검색 시 유럽 이미지를 주로 검색하지만 농업 이미지는 아프리카에서 주로 검색하는 등, 일관되지 않은 성향이 드러났습니다.



### DiffuseDef: Improved Robustness to Adversarial Attacks (https://arxiv.org/abs/2407.00248)
- **What's New**: 프리트레인된 언어 모델(pretrained language models)을 사용한 자연어 처리(NLP) 시스템에 대한 새로운 공격 방어 방법, DiffuseDef를 소개합니다. DiffuseDef는 의료 패러다임에서 널리 사용되는 diffusion layer(확산층)을 디노이저(denoiser)로 활용하여 텍스트 분류 작업에서의 대항 공격(adversarial attacks)을 방어합니다.

- **Technical Details**: DiffuseDef는 텍스트가 인코더(encoder)를 통과하기 전에 난수 노이즈(noise)를 추가하고, 이를 확산층을 통해 반복적으로 디노이징합니다. 그런 다음 디노이징된 여러 버전의 텍스트를 앙상블(ensembling)하여 더 강력한 텍스트 표현을 만듭니다. 이는 기존의 대항 훈련(adversarial training), 텍스트 디노이징(text denoising), 그리고 앙상블 방법을 통합하여 성능을 향상시킵니다.

- **Performance Highlights**: DiffuseDef는 다수의 실험을 통해 기존의 강력한 방어 방법들을 능가하며, 다양한 유형의 대항 공격에 대해 우수한 성능을 발휘함을 증명했습니다. 또한, 앙상블된 디노이징 표현이 모델의 취약점을 공격하는 단어들을 줄이고, 대항 텍스트와 깨끗한 텍스트 간의 거리(latent space distance)를 감소시켜 더 강력한 방어를 제공합니다.



### EHRmonize: A Framework for Medical Concept Abstraction from Electronic Health Records using Large Language Models (https://arxiv.org/abs/2407.00242)
Comments:
          submitted for review, total of 10 pages

- **What's New**: EHRmonize는 대형 언어 모델(LLMs)을 활용해 전자의무기록(EHR) 데이터에서 의료 개념을 추출하는 새로운 프레임워크를 소개합니다. 이 연구는 두 개의 실제 EHR 데이터베이스에서 얻은 약물 데이터를 사용하여 GPT-4o 및 Claude-3.5-Sonnet 모델을 다양한 프롬팅 전략에서 평가했습니다. 특히 GPT-4o 모델은 모든 작업에서 최고 성능을 보였으며, 항생제 이진 분류에서는 100%의 정확도를 달성했습니다. EHRmonize는 주석 작업 시간을 60%까지 줄일 수 있으며, Python 패키지로도 제공되어 연구자들이 쉽게 접근할 수 있습니다.

- **Technical Details**: EHRmonize는 EHR 데이터에서 의료 개념을 정리하고 분류하는 작업을 자동화합니다. 이 프레임워크는 SQL 기반 추출과 LLM 추론으로 구성되며, raw input을 표준화된 클래스로 변환하는 few-shot 프롬팅을 사용합니다. MIMIC-IV와 eICU-CRD 데이터베이스의 약물 데이터를 사용하여 두 가지 작업 유형(일반 경로 및 약물 이름 추출, 바이너리 분류)을 수행했습니다.

- **Performance Highlights**: GPT-4o 모델은 모든 작업에서 최고 성능을 보였으며, 특히 일반 경로 이름 식별에서 97%, 일반 약물 이름에서 82%, 항생제 이진 분류에서 100%의 정확도를 달성했습니다. 또한, EHRmonize는 주석 작업 시간을 약 60% 줄일 수 있어 연구 효율성을 크게 향상시켰습니다.



### Evaluating Human Alignment and Model Faithfulness of LLM Rationa (https://arxiv.org/abs/2407.00219)
- **What's New**: 이번 연구에서는 LLM (Large Language Models)의 입력 텍스트에서 의사 결정을 반영하는 토큰 세트인 'Rationales'를 어떻게 잘 설명할 수 있는지 조사합니다. 두 가지 방법을 통해 LLM의 Rationales를 추출했습니다: 첫째, 주의력(attention)이나 그라디언트를 사용하여 중요한 토큰을 찾는 '속성 기반'(Attribution-based) 방법, 둘째, 프롬프트(prompt)를 사용하여 LLM이 일련의 설명을 추출하도록 유도하는 '프롬프트 기반'(Prompting-based) 방법입니다.

- **Technical Details**: 이 연구에서는 속성 기반 방법과 프롬프트 기반 방법 두 가지를 비교 평가했습니다. 속성 기반 방법은 Attention Weights 또는 Input×Gradient를 사용했고, 프롬프트 기반 방법은 모델의 지시를 따르는 능력을 활용해 주요 토큰을 추출했습니다. 사용된 데이터셋에는 e-SNLI와 MedicalBios가 포함되어 있으며, 총 다섯 개의 최신 LLM들이 평가되었습니다: Llama2, Llama3, Mistral, GPT-3.5-Turbo, GPT-4-Turbo입니다.

- **Performance Highlights**: 프롬프트 기반의 Rationales는 속성 기반 방법보다 인간 주석과 더 잘 일치하는 경향이 있었으며, 모델 성능이 낮아도 인간의 추론과 유사한 설명을 만들어낼 수 있었습니다. 또한, 프롬프트 기반 방법의 충실성(faithfulness) 한계는 예측의 붕괴와 관련이 있을 수 있음이 발견되었습니다. LLM을 특정 데이터셋으로 파인튜닝하게 되면 두 가지 방법 모두 충실성과 인간 일치도 면에서 품질이 향상된다는 점을 확인했습니다.



### Detection and Measurement of Syntactic Templates in Generated Tex (https://arxiv.org/abs/2407.00211)
- **What's New**: 최근 대형 언어 모델(LLM)의 텍스트 생성 다양성을 평가하는 연구는 주로 단어 수준의 특징에 집중되어 있었습니다. 이번 연구는 빈번한 n-그램을 넘어서는 일반적인 반복을 특성화하기 위해 구문(문법)적 특징을 분석합니다. 특히, 구문 템플릿(syntactic templates)을 정의하고, 모델이 다운스트림 작업에서 인간 참조 텍스트보다 더 높은 비율로 템플릿화된 텍스트를 생성한다는 것을 보여줍니다.

- **Technical Details**: 구문 템플릿은 문장의 부분 품사(POS) 태그 시퀀스를 의미하며, 이는 다양한 패턴 세트를 보여주는 구문적 추상화입니다. 연구는 또한 템플릿이 주어진 데이터 집합에서 반복되는 빈도에 따라 특징이 정의된다는 것을 설명합니다. 예를 들어, 'DT JJ NN IN DT JJ NN'와 같이 반복되는 패턴을 통해 텍스트의 유사성을 잡아낼 수 있습니다. 이를 위해 연구자들은 diversity라는 라이브러리를 사용하여 데이터 세트의 단어와 부분 품사 태그의 다양성을 평가하고, 가장 빈번한 n-그램을 추출하였습니다.

- **Performance Highlights**: 이번 연구에서는 주어진 텍스트에서 템플릿을 검출하여 모델이 데이터 메모리제이션(데이터 기억)을 얼마나 잘하는지를 평가했습니다. 특히 템플릿이 모델, 작업, 도메인을 구분하는 중요한 특징으로 작용하며, 모델의 학습 데이터에서 스타일 메모리제이션을 분석하는 데 유용함을 입증했습니다. 연구에는 총 8개의 폐쇄형 모델이 3가지 다른 작업에서 평가되었으며, 각 모델의 템플릿 생성 경향성을 통해 모델 학습 데이터의 특성을 도출할 수 있었음을 보여주었습니다.



### MetaKP: On-Demand Keyphrase Generation (https://arxiv.org/abs/2407.00191)
- **What's New**: 새로운 패러다임인 '온디맨드 키프레이즈 생성(On-demand keyphrase generation)'을 도입했습니다. 이 새로운 방법은 특정 고유의 목표나 의도에 맞는 키프레이즈를 예측하는 방식입니다. 이를 위해 메타KP(MetaKP)라는 대규모 벤치마크를 제공하며, 뉴스와 생의학 분야의 7500개 문서와 3760개의 목표가 포함된 데이터셋을 포함합니다.

- **Technical Details**: MetaKP를 활용하여 감독 학습(supervised)과 비감독 학습(unsupervised) 방법을 설계했습니다. 감독 학습 방법은 멀티태스킹 파인튠(multi-task fine-tuning) 접근 방식을 채택했으며, 비감독 학습 방법으로는 대형 언어 모델(LLMs)에 셀프 컨시스턴시 프롬프팅(self-consistency prompting)을 적용했습니다.

- **Performance Highlights**: 비감독 방법의 셀프 컨시스턴시 프롬프팅 접근 방식이 성능을 크게 향상시켰으며, GPT-4o가 0.548 SemF1을 달성해, 완전히 파인튠된 BART-base 모델의 성능을 능가했습니다. 또한, LLM 기반 비감독 방법은 다양한 도메인에 안정적인 성능을 유지하며, 뉴스 도메인에서는 19% 더 높은 성능을 보였습니다. 이 방법은 소셜 미디어에서 전염병 이벤트를 감지하는 데 성공적으로 활용될 수 있음을 입증했습니다.



### Can GPT-4 Help Detect Quit Vaping Intentions? An Exploration of Automatic Data Annotation Approach (https://arxiv.org/abs/2407.00167)
Comments:
          Accepted for the AI Applications in Public Health and Social Services workshop at the 22nd International Conference on Artificial Intelligence in Medicine (AIME 2024)

- **What's New**: 최근 미국에서는 전자담배 사용이 급증하면서 2019년 EVALI(Electronic Cigarette or Vaping product use-Associated Lung Injury) 사태가 발생해 병원 입원과 사망 사례가 눈에 띄게 늘었습니다. 이에 따라 전자담배 사용 행태를 이해하고 금연 전략을 개발하는 것이 시급합니다. 이 연구에서는 소셜 미디어 데이터, 특히 Reddit의 전자담배 관련 서브 커뮤니티 데이터를 분석하여 사용자들의 금연 의도를 탐지하기 위해 OpenAI의 최신 대형 언어 모델 GPT-4를 활용했습니다.

- **Technical Details**: 연구는 Reddit의 r/QuitVaping 서브레딧에서 1000개의 게시물을 수집한 후, 제목과 본문을 문장 단위로 토크나이징하여 분석했습니다. Layman 주석자와 임상 전문가가 각각 YES 또는 NO로 주석한 데이터를 기반으로 GPT-4를 평가했습니다. Zero-shot, One-shot, Few-shot, 그리고 Chain-of-thought prompting 전략을 사용해 8개의 다양한 프롬프트를 개발하여 GPT-4의 성능을 평가했습니다.

- **Performance Highlights**: 연구 결과, GPT-4 모델은 인간 주석자와 비교해 인상적인 성능을 보였지만 여전히 인간 주석자를 완전히 대체하기에는 한계가 있다는 점을 발견했습니다. 특히 예제 기반 프롬프트와 단계별 사고(prompting)에 대해 더 높은 세부사항을 제공하면 성능이 개선된다는 점을 확인했습니다.



### The Qiyas Benchmark: Measuring ChatGPT Mathematical and Language Understanding in Arabic (https://arxiv.org/abs/2407.00146)
- **What's New**: 이번 연구에서는 아랍어 데이터를 기반으로 사전 학습된 언어 모델들이 부족한 상황을 개선하기 위해, 아랍어 수학적 추론 및 언어 이해 능력을 평가할 수 있는 새로운 벤치마크를 도입하였습니다. 이 벤치마크는 사우디아라비아에서 대학 입학 시 사용되는 표준화된 테스트인 Qiyas 시험에서 유래되었습니다.

- **Technical Details**: 제안된 벤치마크는 General Aptitude Test (GAT)에서 추출된 문제를 포함하고 있으며, 수학적 추론(Mathematical Reasoning)과 언어 이해(Language Understanding)를 평가하도록 설계되었습니다. 두 가지 언어 모델, ChatGPT-3.5-trubo와 ChatGPT-4의 성능을 이 벤치마크를 통해 평가하였습니다.

- **Performance Highlights**: 실험 결과, 새로운 벤치마크는 언어 모델들에게 상당한 도전 과제를 제시함을 확인했습니다. ChatGPT-4는 평균 정확도 64%를 기록했으며, ChatGPT-3.5-trubo는 평균 49%의 정확도를 보였습니다. 이는 향후 아랍어 모델들의 수학적 추론 및 언어 이해 능력을 향상시키는 데 기여할 수 있을 것으로 기대됩니다.



### Empowering 3D Visual Grounding with Reasoning Capabilities (https://arxiv.org/abs/2407.01525)
Comments:
          Accepted by ECCV24. A comprehensive and hierarchical 3D reasoning grounding benchmark in the era of foundation models. Project page: this https URL

- **What's New**: 최근 3D 비주얼 그라운딩(visual grounding) 분야에서 많은 발전이 있었지만, 여전히 명시적 텍스트 설명에 의존하는 모델들이 대부분입니다. 이를 개선하기 위해, 우리는 새로운 과제인 3D 추론 그라운딩(reasoning grounding)을 제안하고, 이에 필요한 새로운 벤치마크 ScanReason을 소개합니다. ScanReason은 5가지 유형의 추론을 요구하는 10,000개 이상의 질문-답변-위치 쌍을 제공하여 추론과 그라운딩의 시너지를 요구합니다.

- **Technical Details**: 우리의 제안 방식인 ReGround3D는 MLLM(Multi-modal Large Language Model)으로 강화된 비주얼 중심 추론 모듈과 3D 그라운딩 모듈로 구성되어 있습니다. 또한 추론과 그라운딩을 반복적으로 교차시키는 Chain-of-Grounding 메커니즘을 도입하여 성능을 한층 더 향상시킵니다. 특히, 이 모델은 복잡한 3D 장면에서 정확한 객체 위치를 찾기 위해 향상된 지오메트리와 세밀한 디테일을 활용합니다.

- **Performance Highlights**: 우리의 ReGround3D는 ScanReason 벤치마크에서 탁월한 성능을 입증했습니다. 다양한 3D 장면에서 효과적인 추론과 그라운딩 과정을 통해 목표 객체의 위치를 정확하게 찾아낼 수 있음을 확인했습니다.



### MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations (https://arxiv.org/abs/2407.01523)
- **What's New**: 새로운 벤치마크인 MMLongBench-Doc가 발표되었습니다. 이 벤치마크는 긴 문서의 다중 모달리티 문서 이해(DU) 능력을 평가하기 위해 설계되었으며, 총 1,062개의 전문가 주석 질문을 포함하고 있습니다.

- **Technical Details**: MMLongBench-Doc는 평균 49.4페이지와 20,970.9개의 텍스트 토큰을 가진 130개의 PDF 문서로 구성되어 있습니다. 질문의 33.2%는 여러 페이지에 걸쳐 증거를 필요로 하는 '크로스 페이지(cross-page) 질문'이며, 22.8%는 잠재적 환각(hallucinations)을 감지하기 위한 '답이 없는(unanswerable) 질문'입니다.

- **Performance Highlights**: 총 14개의 LVLM(Large Vision-Language Models)이 테스트되었으며, 가장 성능이 우수한 GPT-4o는 F1 점수 42.7%를 기록했습니다. 두 번째로 우수한 GPT-4V는 31.4%를 기록했으며, 나머지 12개의 LVLM은 OCR을 통해 변환된 텍스트만을 사용한 LLM보다도 성능이 낮았습니다.



### MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs (https://arxiv.org/abs/2407.01509)
- **What's New**: MIA-Bench는 Multimodal Large Language Models (MLLMs)의 엄격한 명령 준수 능력을 평가하기 위해 새롭게 개발된 벤치마크입니다. 이 벤치마크에는 400개의 이미지-프롬프트(IP) 쌍이 포함되어 있으며, 복잡한 지시 사항을 정확히 따르는 MLLM을 도전하기 위해 설계되었습니다.

- **Technical Details**: MIA-Bench는 층화된(layered) 및 복합적인(compositional) 명령을 처리하고 정확한 답변을 생성할 수 있는 MLLM의 정밀도를 평가합니다. 이를 위해 다섯 가지 기본 명령 카테고리(설명, 언급, 문법, 길이 한계, 장르)를 사용하여 다양한 복잡성 수준의 프롬프트를 구성하였습니다. 또, 다양한 소스로부터 이미지를 수집하였으며, 각 이미지에 여러 하위 명령을 포함하는 복잡한 지시 사항을 수동으로 작성했습니다.

- **Performance Highlights**: 다양한 최첨단 MLLM을 평가한 결과, 모델의 성능에 상당한 차이가 있음을 발견하였습니다. 이는 명령 준수 능력을 향상시키기 위한 상당한 개선 여지가 있다는 것을 강조합니다. 추가적으로, 감독된 추가 학습(SFT)을 통해 MLLM의 명령 준수 성능을 향상시키기 위한 훈련 데이터를 생성하고 실험을 실시하였습니다. 이 실험 결과, 다른 벤치마크 성능을 저해하지 않으면서도 명령을 엄격히 따르는 능력이 향상되었습니다.



### Agentless: Demystifying LLM-based Software Engineering Agents (https://arxiv.org/abs/2407.01489)
- **What's New**: 최근 소프트웨어 개발 작업의 자동화에서 큰 진전을 보인 큰 언어 모델(LLMs)은 코드 생성(code synthesis), 프로그램 수리(program repair), 테스트 생성(test generation) 등의 영역에서 눈에 띄는 성과를 보여왔습니다. 그러나 이러한 복잡한 에이전트 기반 접근법 대신 단순한 에이전트 없는 방법(Agentless)을 제안합니다. Agentless는 문제를 지역화(localization)하고 수리(repair)하는 두 단계로 소프트웨어 개발 문제를 자동으로 해결합니다.

- **Technical Details**: Agentless는 LLM이 미래의 행동을 결정하거나 복잡한 도구를 사용하는 것을 허용하지 않고, 단순히 지역화와 수리 두 단계를 따릅니다. 지역화 단계에서는 프로젝트의 코드베이스를 트리 구조로 변환하고, LLM을 사용해 수정해야 할 의심 파일을 순위 매깁니다. 이후 클래스와 함수의 선언 헤더를 제공하여 LLM이 수정을 희망하는 특정 목록을 출력하게 하고, 최종적으로 특정 라인 등의 세부 편집 위치를 결정합니다. 수리 단계에서는 편집 위치의 코드 스니펫과 문제 설명을 제공하여 LLM이 문제를 해결하는 여러 패치를 생성합니다.

- **Performance Highlights**: Agentless는 SWE-bench Lite 벤치마크에서 27.33%의 최고 성능을 달성했으며, 비용은 단 0.34달러에 불과했습니다. 또한 SWE-bench Lite의 문제를 보다 정교하게 평가하기 위해 불충분한 또는 오도된 문제 설명을 가진 문제들을 제외한 SWE-bench Lite-S를 구축하였습니다. SWE-bench Lite-S는 보다 엄격한 평가 기준을 제공하여 실세계 소프트웨어 개발 문제 해결 능력을 평가하는 데 사용됩니다.



### Tree Search for Language Model Agents (https://arxiv.org/abs/2407.01476)
Comments:
          11 pages. Models and code available at this https URL

- **What's New**: 최근 언어 모델(LM)을 활용한 자율 에이전트가 웹 자동화 같은 의사결정 작업에서 가능성을 보여주고 있습니다. 그러나 다중 단계의 추론, 계획, 환경 피드백 활용에서 여전히 한계가 있습니다. 이를 해결하기 위해 우리는 LM 에이전트가 상호작용 웹 환경에서 탐색과 다단계 계획을 명시적으로 수행할 수 있는 추론 시간 탐색 알고리즘을 제안합니다.

- **Technical Details**: 제안된 방법은 실제 환경 공간 내에서 작동하는 최적 우선 탐색(best-first tree search) 형태로, 대부분의 최신 에이전트와 상호보완적으로 작동합니다. 이 탐색 알고리즘은 LM 에이전트가 현실적인 웹 작업에서 효과를 보이는 것을 처음으로 입증한 최초의 트리 탐색 알고리즘입니다. 환경 피드백을 활용한 다단계 계획과 명시적 탐색을 통해 신뢰성을 줄입니다. 우리는 모델 기반 가치 함수(value function)를 제안하여 명시적 보상이 없는 다양한 환경에서도 최적 우선 탐색을 안내합니다.

- **Performance Highlights**: VisualWebArena 벤치마크에서, 제안된 탐색 알고리즘을 GPT-4o 에이전트에 적용한 결과, 탐색 없는 동일 기준에 비해 성공률이 39.7% 상대적으로 증가하여 26.4%의 최신 성공률을 기록했습니다. WebArena에서도 탐색이 28.0%의 상대 개선을 달성하며 경쟁력 있는 성공률 19.2%를 기록했습니다. 이러한 성과는 증가된 테스트 시간 연산(compute)으로 인해 성능이 확장됨을 보여줍니다.



### ColPali: Efficient Document Retrieval with Vision Language Models (https://arxiv.org/abs/2407.01449)
Comments:
          Under Review

- **What's New**: ViDoRe라는 새로운 Visual Document Retrieval Benchmark가 도입되었습니다. 이 벤치마크는 다양한 도메인, 언어, 설정에서 페이지 수준의 검색 작업을 포함하여 현대 시스템의 시각적 요소를 효율적으로 활용하지 못하는 문제를 해결하려는 목적을 가지고 있습니다. ViDoRe를 통해 문서 검색 시스템의 성능을 평가하는 기회를 제공합니다.

- **Technical Details**: 기존의 문서 검색 시스템들은 텍스트 임베딩 모델의 성능에 주로 의존하였으나, 실질적인 문서 검색에서는 문서의 시각적 요소도 효율적으로 활용되어야 합니다. 이를 위해 새로운 모델 구조인 ColPali가 제안되었습니다. ColPali는 Vision Language Models (VLMs)을 활용하여 문서의 이미지에서 고품질 문맥화된 임베딩을 생성하며, 빠른 쿼리 매칭을 위해 늦은 상호작용 매커니즘을 결합했습니다.

- **Performance Highlights**: ColPali는 ViDoRe 벤치마크에서 대부분의 기존 문서 검색 시스템을 능가하며, 빠르고 엔드 투 엔드 트레이닝이 가능함을 보여주었습니다. 이를 통해 문서 페이지 이미지만으로도 높은 성능의 문서 검색이 가능함을 입증하였습니다.



### Gloss2Text: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing (https://arxiv.org/abs/2407.01394)
- **What's New**: 영상에서 음성 텍스트로의 수화 번역에서 중간 글로스(Gloss) 주석의 역할을 강조하며, 최신 트렌드인 대형 언어 모델(LLMs), 데이터 증강, 그리고 새로운 라벨-스무딩 손실 함수(Label-Smoothing Loss Function)를 사용해 성능을 크게 향상시켰습니다. PHOENIX Weather 2014T 데이터셋을 통해, 제안된 접근 방식이 현재 최고 성능을 능가함을 입증했습니다.

- **Technical Details**: 본 연구는 글로스(Gloss) 주석을 음성 언어로 번역하는 Gloss2Text 단계에 초점을 맞추어 기계 번역(Neural Machine Translation, NMT) 모델을 최적화했습니다. 데이터 증강 기법으로는 변환된 문장을 다시 번역해 원래의 의미를 유지하면서도 다양한 언어적 변화를 도입하는 방법과 역방향 번역 모델을 사용해 새로운 글로스 시퀀스를 생성하여 기본 훈련 과정에 포함시키는 등의 기법을 제안했습니다. 그리고 기존의 라벨-스무딩 접근 방식을 향상시켜, 타겟 번역과 유사한 잘못된 예측에 대한 페널티를 줄이는 방법을 소개했습니다.

- **Performance Highlights**: PHOENIX Weather 2014T 데이터셋에서 제안된 방법론으로 기존 최고 성능을 능가하는 성과를 거두었습니다. 이는 대형 언어 모델을 활용하고, 데이터 증강과 새로운 라벨-스무딩 손실 함수를 도입한 결과로 평가되었습니다. 광범위한 실험과 세밀한 에블레이션 연구를 통해 접근 방식의 유효성을 검증했습니다.



### Badllama 3: removing safety finetuning from Llama 3 in minutes (https://arxiv.org/abs/2407.01376)
- **What's New**: 이 논문은 공격자가 모델 가중치에 접근할 수 있을 때, 대규모 언어 모델(LLM)의 안전성 미세 조정(safety fine-tuning)이 쉽게 무력화될 수 있음을 보여줍니다. 논문에서는 최신 미세 조정 기법인 QLoRA, ReFT, 그리고 Ortho를 평가하며, 알고리즘적 개선이 어떻게 일관된 jailbreaking 성능을 유지하면서도 플롭(FLOPs) 및 최적화 파워를 절감할 수 있는지 논의합니다.

- **Technical Details**: Llama 3 8B와 같은 최첨단 오픈 모델에서 안전성 미세 조정을 제거하는 데 소요되는 시간을 측정했습니다. Llama 3 8B는 단일 GPU에서 1분, Llama 3 70B는 30분 만에 Strip할 수 있습니다. 또한, 구글 Colab 무료 환경에서의 성능도 검토되었습니다. 이를 통해 공격자는 <100MB 크기의 'jailbreak adapter'를 배포할 수 있습니다. 안전성 평가에는 Attack Success Rate(ASR)와 그 반대 지표인 Attack Refusal Rate(ARR)이 사용되었습니다.

- **Performance Highlights**: Llama 3와 비교했을 때 Badllama 3의 성능은 일반적인 LLM 성능 벤치마크에서 거의 차이가 나지 않으며, 불법적인 질문을 거부하는 비율이 유의미하게 낮아졌습니다. 기존의 다른 벤치마크보다 3−5배 더 빠르게 계산 시간을 단축할 수 있으며, 최소의 자원으로도 수행할 수 있음을 보여주었습니다.



### Increasing Model Capacity for Free: A Simple Strategy for Parameter Efficient Fine-tuning (https://arxiv.org/abs/2407.01320)
Comments:
          Accepted at ICLR 2024. Code at this https URL

- **What's New**: CapaBoost라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 기존의 파라미터 효율적 미세 조정(Parameter Efficient Fine-Tuning, PEFT) 방법들과 원활하게 통합되어 모델 용량을 크게 늘릴 수 있습니다. 이는 특히 제한된 파라미터 예산 하에서 효과적입니다.

- **Technical Details**: CapaBoost는 타겟 계층에 병렬 가중치 모듈을 통해 저순위(low-rank) 업데이트를 적용하여 모델 용량을 증가시킵니다. 공유된 가중치 행렬에 정적 랜덤 마스크를 적용해 다양한 가중치 행렬 세트를 구성함으로써 동일한 파라미터 수로 증분 가중치의 순위를 효과적으로 높입니다. 이 접근법은 기존의 다양한 PEFT 방법들과 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 다양한 다운스트림 작업에 대한 실험에서 CapaBoost는 기존 베이스라인 대비 상당한 성능 향상을 보여주었으며, 추가적인 계산 또는 저장 비용없이 수행되었습니다. CapaBoost를 사용한 PEFT 방법들은 동일하거나 더 적은 FLOPs(Floating Point Operations)로 높은 모델 성능을 달성했습니다.



### Lightweight Zero-shot Text-to-Speech with Mixture of Adapters (https://arxiv.org/abs/2407.01291)
Comments:
          5 pages,3 figures, Accepted to INTERSPEECH 2024

- **What's New**: 최신 연구에서는 대형 모델 기반의 제로샷 텍스트-투-스피치(Zero-shot Text-to-Speech, TTS) 방법이 높은 음질로 화자의 특성을 재현할 수 있음을 보여주었습니다. 그러나 이러한 모델은 일상적인 사용을 위한 실용성에서 너무 큽니다. 우리는 다양한 화자에게 적응할 수 있는 경량화된 제로샷 TTS 방법을 제안합니다. 우리의 방법은 혼합 어댑터(Mixture of Adapters, MoA) 모듈을 디코더와 비자립형(non-autoregressive) TTS 모델의 분산 어댑터에 통합하여 화자 특성을 기반으로 적절한 어댑터를 선택합니다.

- **Technical Details**: 제안된 방법은 대형 데이터셋을 사용하여 학습된 모델로 다양한 화자 특성을 커버할 수 있도록 설계되었습니다. 주요 구성 요소는 셀프-슈퍼바이즈드 러닝(Self-Supervised Learning, SSL) 기반의 임베딩 추출기, 피드포워드 트랜스포머(Feed-forward Transformer, FFT) 블럭, 및 MoA 모듈입니다. MoA 모듈은 SSL 모델의 각 레이어로부터 학습 가능한 가중치를 사용하여 음성 표현을 가중합한 후, 양방향 GRU와 어텐션 레이어를 거쳐 고정된 길이의 벡터로 변환합니다. 이러한 방식으로 추출된 화자 임베딩을 TTS 모델에 피드하여 적응합니다.

- **Performance Highlights**: 제안된 방법은 객관적 및 주관적 평가를 통해 기존 방식 대비 40% 이하의 파라미터만 사용하면서도 1.9배 빠른 추론 속도로 더 나은 성능을 달성함을 확인했습니다. 오디오 샘플은 데모 페이지에서 확인할 수 있습니다.



### We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning? (https://arxiv.org/abs/2407.01284)
Comments:
          Work in progress

- **What's New**: WE-MATH는 시각적 수학적 추론을 위한 첫 번째 벤치마크를 소개합니다. 이는 단순한 결과 중심의 성능 평가를 넘어 문제 해결 원칙을 탐구하는 데 중점을 두었습니다. 이 벤치마크는 6,500개의 시각적 수학 문제를 다양한 계층적 지식 개념과 다섯 가지 지식 세분성으로 분류하여 수집했습니다.

- **Technical Details**: WE-MATH는 복합 문제를 필요한 지식 개념에 따라 하위 문제로 분해하고, Insufficient Knowledge (IK), Inadequate Generalization (IG), Complete Mastery (CM), Rote Memorization (RM)의 네 가지 차원적 메트릭을 도입합니다. 이를 통해 LMM(Large Multimodal Models)의 추론 과정에서 발생하는 내재적 문제를 계층적으로 평가합니다. GPT-4o는 IK 문제에서 IG 단계로 진입했음을 확인했습니다. 이에 비해 다른 LMM은 Rote Memorization 성향을 강하게 보여줍니다.

- **Performance Highlights**: WE-MATH의 평가 결과, GPT-4o는 시각적 수학 카테고리에서 최고의 전반적 성능을 보여줬습니다. 닫힌 소스의 LLM과 대규모 파라미터를 가진 LMM이 뛰어난 성능을 발휘한 반면, 다단계 문제에서는 대부분의 LMM이 성능 저하를 겪었습니다. 특히 작은 규모의 모델들은 IK 문제를 많이 겪었으며, 제안된 KCA 전략은 이러한 문제를 효과적으로 감소시켰습니다.



### Leveraging Large Language Models for Actionable Course Evaluation Student Feedback to Lecturers (https://arxiv.org/abs/2407.01274)
Comments:
          Accepted to SEFI 2024

- **What's New**: 이번 연구에서는 학기 말 학생 설문 조사를 통한 교수 교육 피드백 수집 방안을 개선하기 위해 오픈소스 생성 AI(generative AI)를 활용한 방법을 탐색했습니다. 특히, 대형 강의의 경우 설문 응답 수가 많아 전통적인 피드백 수집 방법이 비효율적일 수 있습니다. 본 연구는 이러한 문제를 해결하고자 75개 강의 과목에서 742명의 학생 응답을 바탕으로 강의 평가 요약 및 교수 개선 사항을 자동으로 생성하는 과정을 제안합니다.

- **Technical Details**: 연구에서는 오픈소스 생성 AI를 사용하여 대량의 학생 피드백 데이터를 합성하고 처리했습니다. 각 강의별로 평가 요약본과 실행 가능한 개선사항을 도출하였으며, 이는 사실적이고 실행 가능하며 적절한 요약본을 생성하는 데 중점을 두었습니다. AI를 통해 대량의 피드백 데이터를 효과적으로 요약해 교수의 교육 향상을 지원할 수 있는 방법을 입증했습니다.

- **Performance Highlights**: 예비 결과는 학습 환경에서 교수의 교육 실습을 향상시키기 위한 유망한 방법을 제시합니다. 생성 AI를 통해 교수에게 인사이트 있는 피드백을 제공하는 것이 가능하며, 이는 경제적으로도 효과적인 교육자 지원 수단임을 나타냅니다. 전체적으로 본 연구는 생성 AI를 활용해 교실 내 교수에게 사실적이고 실행 가능한 피드백을 제공하는 가능성을 강조합니다.



### Unaligning Everything: Or Aligning Any Text to Any Image in Multimodal Models (https://arxiv.org/abs/2407.01157)
Comments:
          14 pages, 14 figures

- **What's New**: 최신 연구에 따르면, 다중 모달(multi-modal) 모델들이 새로운 제로샷(zero-shot) 학습 능력을 보여주고 있습니다. 특히, 이 논문은 이미지-텍스트 모델 간의 공유 임베딩 공간(shared embedding space)이 새로운 취약성을 가질 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 기울기 기반 절차(gradient-based procedure)에 기반한 최적화 방법을 사용하는데, 이를 통해 이미지를 미세하게 수정하여 텍스트의 임베딩과 일치시키는 방법을 제시하고 있습니다. 이러한 방법으로, 시각적으로 드러나지 않는 적대적 공격(adversarial attacks)을 통해 텍스트와 이미지 간의 임베딩이 불일치할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 여러 소스로부터의 텍스트와 이미지 데이터셋에 대해 100% 성공률을 달성하여 시각적으로 구별되지 않는 이미지가 아주 다른 텍스트의 임베딩과 일치함을 확인했습니다. 이는 다중 모달 모델이 서로 다른 모달리티의 입력을 의미론적으로 일치시키는 데에 심각한 취약성이 있음을 시사합니다.



### CVLUE: A New Benchmark Dataset for Chinese Vision-Language Understanding Evaluation (https://arxiv.org/abs/2407.01081)
- **What's New**: 기존의 중국어 비전-언어 모델(VLM)들이 서양 중심의 이미지에 기반한 데이터셋을 사용하여 평가되는 문제점을 지적하며, 새로운 중국어 비전-언어 이해 평가(CVLUE) 벤치마크 데이터셋을 소개했습니다. 이 데이터셋은 중국 원어민에 의해 선택된 객체 범주와 이미지로 구성되어, 중국 문화의 대표성을 보장합니다. CVLUE는 이미지-텍스트 검색(image-text retrieval), 시각 질문 응답(visual question answering), 시각적 근거제공(visual grounding), 시각 대화(visual dialogue) 등 네 가지 다양한 VL 작업을 포함합니다.

- **Technical Details**: CVLUE 데이터셋은 중국 인터넷에서 수집된 이미지와 텍스트로 구성됩니다. 데이터셋은 이미지-텍스트 검색(ITR), 시각 질문 응답(VQA), 시각적 근거제공(VG), 시각 대화(VD)와 같은 각기 다른 VL 작업을 평가할 수 있도록 설계되었습니다. 또한, 여러 개방형 다국어 VLM을 CVLUE와 기존 영어 VL 데이터셋에서 벤치마킹하여 성능 차이를 분석했습니다.

- **Performance Highlights**: 벤치마킹 결과, 기존 VLM들이 중국 문화 관련 지식이 부족함을 드러냈습니다. 그러나 중국 문화와 관련된 VL 데이터셋으로 파인 튜닝(fine-tuning)하면 VLM의 중국 문화 이해가 효과적으로 향상될 수 있음을 발견했습니다.



### Human-like object concept representations emerge naturally in multimodal large language models (https://arxiv.org/abs/2407.01067)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)과 다중 모달 언어 모델(MLLMs)이 인간과 유사한 객체 개념 표현을 어떻게 발전시킬 수 있는지를 초점을 맞추었습니다. 이를 위해 470만 개의 트리플릿 판단 데이터를 수집하여 1854개의 자연 객체의 유사성 구조를 낮은 차원의 임베딩으로 도출했습니다.

- **Technical Details**: 연구팀은 LLM과 MLLM의 트리플릿 odd-one-out 과제를 통해 얻은 유사성 판단 데이터를 섭취하여 Sparse Positive Similarity Embedding(SPoSE) 방법을 사용했습니다. 이 방법으로 66차원의 임베딩을 식별하여 객체 간의 유사성을 예측했습니다. 중요한 점은 이러한 임베딩 차원이 인간의 인지적 표현과 유사하다는 점입니다.

- **Performance Highlights**: 연구팀이 식별한 모델 임베딩과 여러 뇌 영역(예: EBA, PPA, RSC, FFA)의 신경활동 패턴 간 강한 정렬이 발견되었습니다. 이는 모델이 인간의 개념적 지식 체계를 반영하는 공통된 기본 구조를 가지고 있음을 시사합니다.



### Mobile-Bench: An Evaluation Benchmark for LLM-based Mobile Agents (https://arxiv.org/abs/2407.00993)
- **What's New**: Mobile-Bench는 LLM 기반의 모바일 에이전트를 평가하는 최초의 종합 벤치마크 플랫폼을 도입했습니다. 이 벤치마크는 다양한 UI 및 API 상호작용을 지원하며, 멀티-앱(앱 간 협업) 시나리오를 평가하기 위해 설계된 832개의 데이터 항목과 200개 이상의 과제들을 포함하고 있습니다.

- **Technical Details**: Mobile-Bench는 기존 UI-only 방식의 제한을 극복하기 위해 103개의 수집된 API를 추가하고, 실제 사용자 쿼리와 LLM을 통한 데이터 확장을 통해 평가 데이터를 수집합니다. 작업의 계획 능력을 보다 잘 평가하기 위해 데이터는 SAST, SAMT, MAMT로 분류되어 다양한 수준의 작업 복잡도를 반영합니다. 새로운 평가 지표 CheckPoint를 도입하여 LLM 기반 모바일 에이전트가 중요한 계획 및 추론 단계를 도달했는지 여부를 평가합니다.

- **Performance Highlights**: Mobile-Bench 플랫폼은 단일 API 호출이 다양한 UI 작업을 대체 가능하도록 하여 작업 완료의 효율성을 크게 향상시켰습니다. 이를 통해 복잡한 멀티-앱 상호작용 시나리오를 더 현실적으로 평가할 수 있으며, 새로운 평가 메트릭을 통해 LLM 기반 에이전트의 순차 작업 과정을 보다 정확하게 측정할 수 있습니다.



### VisEval: A Benchmark for Data Visualization in the Era of Large Language Models (https://arxiv.org/abs/2407.00981)
- **What's New**: 이번 연구에서는 자연어를 시각화(visualization)로 변환하는 과정을 보다 효율적으로 평가하기 위한 새로운 벤치마크, VisEval을 제안합니다. 이 벤치마크는 146개의 데이터베이스에 걸친 2,524개의 쿼리를 포함한 고품질 대규모 데이터셋을 바탕으로 하며, 유효성(validity), 합법성(legality), 가독성(readability) 등의 여러 차원을 포괄하는 자동화된 평가 방법론을 제공하는 것이 특징입니다.

- **Technical Details**: VisEval 벤치마크는 다양한 최신 대형 언어 모델(LLMs)을 평가하기 위해 고안되었습니다. 평가 과정에서는 생성된 코드와 최종 결과물인 차트 이미지를 검토하게 됩니다. 이를 위해 Python의 Matplotlib와 Seaborn 등의 널리 사용되는 시각화 라이브러리를 사용하여 코드를 생성합니다. 추가로, 다양한 체크 포인트를 통해 결함을 체계적으로 분석합니다.

- **Performance Highlights**: 본 평가 결과는 기존 LLMs 기반의 시각화 생성 방법론이 여전히 실행 실패, 데이터 변환 오류, 누락된 범례 등의 문제를 안고 있다는 것을 밝혀냈습니다. 이를 통해 LLMs의 능력을 이해하고 방법론 개선을 위한 귀중한 인사이트를 제공하고 있습니다.



### ProductAgent: Benchmarking Conversational Product Search Agent with Asking Clarification Questions (https://arxiv.org/abs/2407.00942)
Comments:
          17 pages, 13 tables, 6 figures. Under review

- **What's New**: 이 논문은 사용자가 불명확한 질문으로 대화를 시작하는 전자상거래 시나리오에서 제품 수요 명확화 작업(task)을 도입합니다. 이 작업에서는 명확화 질문(clarification questions)을 통해 보다 정확하고 맞춤형의 제품 검색을 수행하는 역할 지향 에이전트를 설계합니다. 이를 위해 ProductAgent라는 대화형 정보 검색 에이전트를 제안하고 있으며, 이 에이전트는 전략적인 명확화 질문 생성 및 동적 제품 검색 기능을 갖추고 있습니다. 또한 LLM 기반 사용자 시뮬레이터를 활용한 PROCLARE이라는 벤치마크를 제안하여 에이전트의 성능을 자동 및 정성적으로 평가합니다.

- **Technical Details**: ProductAgent는 제품 데이터베이스, 메모리 모듈 및 도구 세트를 통합하여 자동 루프를 수행합니다. 1) 제품 데이터베이스는 구조적 및 벡터화된 형태로 제품 항목을 저장하여 관련 제품을 검색 및 요약할 수 있게 합니다. 2) 메모리 모듈은 대화 세션 동안 사용자의 구조적 명확화 질문 및 비구조적 대화 기록을 캐시하여 다음 질문을 동적으로 할 수 있게 합니다. 3) 다양한 작업(예: 제품 검색 및 명확화 질문 생성)을 지원하는 도구 세트를 제공합니다. 이러한 모듈과의 효과적인 상호작용을 위해 설계된 프롬프트를 활용하여, 언어와 머신러닝 모델(LLMs) 간의 상호작용을 자동화했습니다.

- **Performance Highlights**: PROCLARE 벤치마크를 사용하여 ProductAgent의 성능을 평가한 결과, 대화 턴이 증가할수록 사용자의 요구가 점점 더 명확해지고 제품 검색 성능이 향상된다는 것을 확인했습니다. 2,000개의 대화 데이터셋을 활용한 실험을 통해 ProductAgent의 효과성을 실증했습니다.



### FoldGPT: Simple and Effective Large Language Model Compression Schem (https://arxiv.org/abs/2407.00928)
- **What's New**: FoldGPT라는 새로운 모델 볼륨 압축 전략이 제안되었습니다. 이 전략은 큰 언어 모델(LLMs)을 스마트폰과 같은 모바일 기기에 효율적으로 배포할 수 있도록 설계되었습니다. 기존의 모델 자르기(pruning)와 양자화(quantization) 기법과 달리, FoldGPT는 블록 제거(block removal)와 블록 매개변수 공유(block parameter sharing)라는 두 가지 주요 단계를 통합하여 모델의 중복성을 줄이고 성능 손실을 최소화합니다.

- **Technical Details**: FoldGPT는 세 가지 주요 부분으로 구성되어 있습니다. 첫째, 학습 가능한 게이팅(gating) 매개변수를 사용하여 각 블록의 중요도를 평가한 후, 중복 레이어를 삭제합니다. 둘째, 잔여 블록에 대해 그룹 매개변수 공유 전략을 적용하여 여러 블록이 동일한 가중치(parameters)를 공유하도록 합니다. 마지막으로, 소량의 미세 조정을 통해 매개변수 공유로 인한 성능 손실을 회복하고, 꼬리 레이어 증류(tail-layer distillation) 전략을 도입하여 성능을 향상시킵니다.

- **Performance Highlights**: FoldGPT가 기존의 최신 기법들보다 우수한 성능을 입증했습니다. 실험 결과, 모델 매개변수의 36%를 제거하고도 원래 모델 성능의 96.25%를 유지할 수 있었습니다. 이는 현재의 SOTA 모델 자르기 알고리즘보다 뛰어납니다. LLaMA-2-7B, Gemma-2B, TinyLLaMA-1.1B와 같은 다양한 스케일의 모델에서 광범위한 실험을 통해 그 유효성이 검증되었습니다.



### From Introspection to Best Practices: Principled Analysis of Demonstrations in Multimodal In-Context Learning (https://arxiv.org/abs/2407.00902)
- **What's New**: 이번 연구는 또 다른 시각 모달리티(visual modality)를 추가한 멀티모달 대형 언어 모델(multimodal LLM)의 '컨텍스트 내 학습(in-context learning, ICL)' 능력을 평가한 결과를 다루고 있습니다. 다양한 새로운 중요한 작업(task)들을 대상으로 체계적이고 원칙적으로 멀티모달 ICL의 작동 원리를 연구했습니다.

- **Technical Details**: 본 연구에서는 멀티모달 ICL의 원리를 이해하기 위해 시각 정보와 텍스트 정보를 각각 변형(perturbation)하여 다양한 작업에 대해 평가를 진행했습니다. 각 작업에서 모달리티의 영향이 다름을 확인했으며, 이를 기반으로 모달리티 기반 데모 선택 전략(demonstration strategy)을 이용해 ICL 성능을 향상시켰습니다.

- **Performance Highlights**: 연구 결과, 시각 정보는 특정 작업에 있어 텍스트 정보보다 중요한 역할을 할 수 있으며, 적절한 데모 선택은 ICL 성능을 크게 향상시킵니다. 특히, 텍스트 유사성 기반 데모 선택이 일관되게 ICL 성능을 높이는 데 효과적임을 확인했습니다. 또한, 모델의 크기가 증가함에 따라 구체적인 작업의 유도 편향(task inductive bias)을 잘 포착할 수 있는 능력이 향상됨을 발견했습니다.



### Mechanistic Interpretation through Contextual Decomposition in Transformers (https://arxiv.org/abs/2407.00886)
- **What's New**: 이 논문에서는 transformers의 기계적 해석(mechanistic interpretability)을 위한 새로운 기법으로서 Contextual Decomposition for Transformers (CD-T)을 소개합니다. 기존의 CD(Contextual Decomposition) 기법을 RNN과 CNN에서 transformer로 확장하여 복잡한 피처 간의 상호작용을 설명할 수 있게 했습니다.

- **Technical Details**: CD-T는 transformers의 입력 피처 조합이나 내부 구성 요소(예: attention heads, feed-forward networks)의 기여도를 계산하여 최종 예측 결과나 특정 내부 구성 요소의 출력을 설명합니다. 특히 CD-T는 모델의 변경 없이 적용 가능하며, 회로 탐색(circuit discovery)을 위한 새로운 효율적인 알고리즘도 제공합니다.

- **Performance Highlights**: CD-T는 기존의 path patching 방법보다 2배 빠른 계산 효율성을 보이며, real-world 병리 보고서 분류 작업에서 더욱 정확한 attention head 회로를 추출합니다. 또한, SST-2와 AGNews 데이터셋에서 local 해석이 가능하며, 인간 실험을 통해 CD-T가 다른 해석 기법(LIME, SHAP)보다 높은 신뢰성을 제공함을 입증했습니다.



### AIMDiT: Modality Augmentation and Interaction via Multimodal Dimension Transformation for Emotion Recognition in Conversations (https://arxiv.org/abs/2407.00743)
- **What's New**: 'Emotion Recognition in Conversations (ERC)'에서 다중 변양 (multimodal) 기능 융합에 대한 새로운 접근법 AIMDiT 제안

- **Technical Details**: AIMDiT는 두 가지 주요 네트워크로 구성: 1) Modality Augmentation Network - 다양한 변양의 차원 전환을 통해 풍부한 표현 학습 수행, parameter-efficient inception block 사용 2) Modality Interaction Network - 추출된 인터모달 (inter-modal) 및 인트라모달 (intra-modal) 기능의 상호작용 융합 수행.

- **Performance Highlights**: 공개 벤치마크 데이터셋 MELD에서 AIMDiT 프레임워크를 사용한 실험 결과, SOTA 모델 대비 Acc-7 메트릭에서 2.34%, w-F1 메트릭에서 2.87% 개선.



### CAMON: Cooperative Agents for Multi-Object Navigation with LLM-based Conversations (https://arxiv.org/abs/2407.00632)
Comments:
          Accepted to the RSS 2024 Workshop: GROUND

- **What's New**: 이 연구는 다수 로봇이 협력하여 복잡한 가정 내 탐색 작업을 수행하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 활용한 분산형 다중 에이전트 내비게이션을 지원하며, 통신에 의해 동적으로 리더십을 배정하여 탐색 효율성을 높입니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 모듈로 구성됩니다: 인식, 통신, 협력 계획입니다. 로봇은 각 방의 레이아웃 패턴을 분류하고 해당 방에서 잠재적 목표물을 탐색합니다. 통신을 통해 동적으로 리더를 배정하고 정보를 효율적으로 교환합니다. 이러한 구조는 LLM을 사용하여 장면을 언어적으로 설명하고 의사 결정을 지원합니다.

- **Performance Highlights**: 이 프레임워크는 적은 통신 횟수로 팀 합의를 도출하여 탐색 효과를 개선합니다. 또한, 팀 크기가 커질 때도 충돌 없이 강력한 성능을 보입니다. 실험 결과, 구조화된 장면 설명과 순차적 통신 구조가 다수 목표물을 탐색하는 작업에서 높은 효율성과 협력 탐색 능력을 제공함을 입증했습니다.



### Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning (https://arxiv.org/abs/2407.00617)
- **What's New**: 이 연구는 강화 학습을 통한 대화형 인공지능 모델의 인간 선호도 정렬(Reinforcement Learning with Human Feedback, RLHF)에서 기존의 보상 기반 접근 방식을 탈피하여 게임 이론적 관점에서 접근하는 새로운 알고리즘을 제안합니다. 이 알고리즘은 반복 Nash 정책 최적화(Iterative Nash Policy Optimization, INPO)로, 두 플레이어 게임으로 문제를 설정하고, 정책이 자기 자신과 대결하며 무후회(No-Regret) 학습을 통해 Nash 정책에 근접하게 합니다.

- **Technical Details**: INPO 알고리즘은 기존의 Bradley-Terry (BT) 모델 가정에서 벗어나서 인간의 복잡한 선호도를 더 잘 반영합니다. 주요 아이디어는 예측 응답의 기대 승률을 추정하지 않고, 새로운 손실 목표를 도입하여 손실을 직접 최적화하는 방식입니다. 이를 통해 높은 계산 비용이나 주석 비용을 줄일 수 있습니다. 이 접근 방식은 또한 온라인 미러 디센트(Online Mirror Descent, OMD)와 같은 무후회 학습 알고리즘에 기반을 둡니다.

- **Performance Highlights**: INPO 알고리즘은 여러 벤치마크에서 실험하여 그 효능을 입증했습니다. 특히 LLaMA-3-8B 기반의 모델을 사용했을 때, INPO는 AlpacaEval 2.0에서 41.5%의 길이 제어 승률, Arena-Hard에서 38.3%의 승률을 기록하며, 이는 DPO (Direct Preference Optimization) 알고리즘에 비해 각각 45.6%와 25.2% 상대적으로 개선된 결과입니다. 또한, KL 정규화를 포함한 손실 함수의 중요성을 강조한 소거 연구도 수행되었습니다.



### Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models (https://arxiv.org/abs/2407.00569)
Comments:
          Accepted to ACL 2024 Main Conference. 21 pages, 20 figures

- **What's New**: 이번 연구에서는 LVLMs (대형 비전-언어 모델)에서 발생하는 다중모달 헛소리(Multimodal Hallucination)에 대한 새로운 평가 프레임워크인 MMHalSnowball을 제안합니다. 이 프레임워크는 LVLMs가 다중모달 헛소리를 생성했을 때, 이후의 대화에서 얼마나 영향을 받는지 평가합니다.

- **Technical Details**: MMHalSnowball 프레임워크는 LVLMs가 생성된 헛소리와 상호작용할 때 발생하는 행동을 평가합니다. 이 프레임워크는 특정 시각적 질문을 다루는 실험적 대화를 통해 LVLMs의 실제 성능을 검증합니다. 또한, 헛소리 현상을 완화하기 위해 Residual Visual Decoding이라는 훈련이 필요 없는 방법을 제안합니다. 이 방법은 시각 입력의 잔여물에서 파생된 분포를 이용하여 LVLMs의 출력 분포를 수정합니다.

- **Performance Highlights**: 실험 결과, 오픈 소스 LVLMs는 헛소리 대화에서 성능이 최소 $31\%$나 떨어지는 것으로 나타났습니다. 우리의 Residual Visual Decoding 방법을 적용하면 다중모달 헛소리 현상을 24\% 이상 완화하면서 모델의 기본 성능을 유지할 수 있었습니다.



### MMEvalPro: Calibrating Multimodal Benchmarks Towards Trustworthy and Efficient Evaluation (https://arxiv.org/abs/2407.00468)
Comments:
          21 pages, code released at this https URL, Homepage at this https URL

- **What's New**: 최근 다양한 데이터를 처리할 수 있는 대형 멀티모달 모델(LMMs, Large Multimodal Models)들이 주목받고 있으며, 이들은 이미지, 질문을 포함한 여러 선택지를 통해 평가됩니다. 그러나 기존의 평가 방법들이 체계적인 편향을 가질 수 있다는 비판이 있는데, 특히 시각적 정보 없이도 높은 성과를 내는 대형 언어 모델(LLMs, Large Language Models)이 이러한 평가의 신뢰성을 손상시키는 경우가 많습니다. 이를 해결하기 위해, 새로운 평가 벤치마크인 MMEvalPro가 제안되었습니다. 이 벤치마크는 보다 철저한 평가 체계를 통해 평가 오류(Type-I Error)를 방지하고자 합니다.

- **Technical Details**: MMEvalPro는 기존 벤치마크에서 등장하는 질문에 대해 숙련된 인간 주석자들이 perception question과 knowledge anchor question을 추가로 작성합니다. 총 2,138개의 질문 삼중연합(question triplets)으로 구성된 이 벤치마크는 6,414개의 개별 질문을 포함합니다. MMEvalPro의 질문 중 2/3는 전문가들에 의해 수동으로 라벨링되었으며, 나머지는 기존 벤치마크(MMMU, ScienceQA, MathVista)에서 가져왔습니다. 주요 메트릭으로는 'Genuine Accuracy'를 사용하며, 이는 모델이 삼중연합 질문을 동시에 맞추는지를 평가합니다.

- **Performance Highlights**: MMEvalPro는 기존 벤치마크에 비해 더 어려운 평가를 제공합니다. 최신 LLMs과 LMMs을 대상으로 한 실험에서, 최고의 LMM이 인간 성과에 비해 31.73% 낮은 성적을 보였고, 이는 기존 벤치마크에서 평균 8.03%의 차이에 불과했던 것에 비해 큰 격차입니다. 또한, 최고의 LLM과 LMM 간의 성과 차이가 기존 벤치마크에서는 14.64%였던 반면, MMEvalPro에서는 23.09%로 나타났습니다. 이러한 결과는 MMEvalPro가 더욱 신뢰할 수 있는 평가 도구임을 강조하며, 차후 연구 개발에 중요한 기회를 제공할 수 있음을 보여줍니다.



### Open-Source Conversational AI with SpeechBrain 1.0 (https://arxiv.org/abs/2407.00463)
Comments:
          Submitted to JMLR (Machine Learning Open Source Software)

- **What's New**: SpeechBrain 1.0은 PyTorch 기반의 오픈 소스 Conversational AI 툴킷으로, 다양한 음성 처리 작업을 지원합니다. 이제 200개 이상의 레시피(Recipes)와 100개 이상의 사전 학습된 모델(Pre-trained Models)을 제공합니다. 또한, 다양한 학습 모달리티, 대형 언어 모델(LLM) 통합 및 고급 디코딩 전략을 지원하는 새로운 기술을 도입하며, 새로운 벤치마크 저장소도 포함하고 있습니다.

- **Technical Details**: SpeechBrain 1.0은 연속 학습, 해석 가능성, 오디오 생성, 효율적인 파인튜닝(Fine-Tuning) 및 wav2vec2 SSL 사전 학습을 지원합니다. 새로운 모델에는 HyperConformer, Branchformer 등 다양한 아키텍처가 포함되어 있으며, 음성 인식, 감정 구분 등을 위한 기술도 도입되었습니다. 추가로, EEG 신호 처리를 지원하여 뇌파 데이터를 음성 처리 기술과 결합할 수 있게 되었습니다.

- **Performance Highlights**: 새로운 업데이트는 빔 서치 알고리즘 향상, GPU 디코딩 지원, 그리고 다양한 스코어러와의 쉽게 통합할 수 있는 인터페이스를 제공합니다. LLM 통합을 통해 GPT-2, Llama 2/3와 같은 모델을 쉽게 사용할 수 있으며, 대화 모델링과 응답 생성에 활용할 수 있습니다. 벤치마크 저장소는 통합 평가 표준을 제공하여 연구자들이 다양한 작업에서 모델 성능을 평가할 수 있게 합니다.



### SHADE: Semantic Hypernym Annotator for Domain-specific Entities -- DnD Domain Use Cas (https://arxiv.org/abs/2407.00407)
- **What's New**: 이 논문에서는 고판타지 문학 도메인의 엔티티를 주석할 수 있는 특별한 소프트웨어인 SHADE를 소개합니다. 구체적으로는 ‘던전 앤 드래곤즈(Dungeons and Dragons, D&D)’의 '포가튼 렐름(Forgotten Realms)' 위키에서 추출된 자료를 대상으로 합니다.

- **Technical Details**: SHADE는 웹 기반 주석 어플리케이션으로, 엔티티를 레이블 데이터와 함께 주석할 수 있습니다. 어플리케이션은 주어진 엔티티에 대해 잠재적 레이블의 두 가지 목록을 제공하며, 레이블이 주어진 출처에 따라 세 가지 중요도 중 하나를 지정할 수 있습니다. 이로 인해 수동 입력을 제한하고, 용어 데이터베이스에서 태그를 직접 추출하여 인간 오류를 최소화할 수 있습니다.

- **Performance Highlights**: SHADE는 주석 작업의 일관성을 보장하기 위해 설계되었으며, 비전문가 주석자도 엔티티에 대한 문맥을 데이터베이스에서 가져와 더 정확한 주석을 달 수 있게 합니다. 이를 통해 주석의 품질과 효율성을 향상시키고자 합니다.



### GraphArena: Benchmarking Large Language Models on Graph Computational Problems (https://arxiv.org/abs/2407.00379)
- **What's New**: 이번 달 아카이브에서는 Large Language Models (LLMs)의 성능을 평가하기 위한 새로운 벤치마크 도구인 GraphArena가 소개되었습니다. GraphArena는 실제 세계의 백만 규모 그래프를 사용하여 LLM의 그래프 계산 문제 해결 능력을 평가합니다. 이 도구는 4개의 다항시간(polynomial-time) 문제와 6개의 NP-complete 문제를 포함한 다양한 10개의 계산 작업을 제공합니다.

- **Technical Details**: GraphArena는 지식 그래프, 소셜 네트워크, 분자 구조 등 다양한 시나리오에서 실제 세계의 백만 규모 그래프를 사용합니다. 이 벤치마크는 LLM의 출력을 정확, 최적이 아님(Feasible but not optimal), 환상적(Hallucinatory; 형식은 맞지만 실행 불가능)으로 분류하는 엄밀한 평가 프레임워크를 특징으로 합니다. LLM이 각 작업에 대한 구체적인 구성 요소나 경로를 식별해야 하는 요구 사항을 포함하고 있어, 표면적인 패턴 인식에 의존하지 않고 엄밀한 평가가 가능하도록 설계되었습니다.

- **Performance Highlights**: GraphArena에서 GPT-4o와 LLaMA3-70B-Instruct를 포함한 10개의 주요 LLM을 평가한 결과, 상위 모델조차도 더 크고 복잡한 그래프 문제에서 고군분투하며 환상적(Hallucination) 이슈를 나타내었습니다. 특히 더 큰 그래프 및 파라미터가 적은 모델에서는 이 문제가 더 심각했습니다. Chain-of-thought 주장 방식(chain-of-thought prompting)과 같은 전략을 적용해도 문제 해결에는 한계가 있었습니다.



### One Prompt is not Enough: Automated Construction of a Mixture-of-Expert Prompts (https://arxiv.org/abs/2407.00256)
Comments:
          ICML 2024. code available at this https URL

- **What's New**: 최신 연구에 따르면, 높은 일반화 능력을 가진 대형 언어 모델(LLMs)은 언어 지시와 맥락 내 데모를 통해 새로운 작업을 해결하는 능력을 보여주고 있습니다. 하지만, 한 개의 지시만으로는 복잡한 문제 공간의 모든 부분을 충분히 커버하기 어려운 한계가 있습니다. 이를 해결하기 위해 Mixture-of-Expert(전문가 혼합) 파라다임을 채택하여 문제 공간을 여러 하위 영역으로 나누고, 각 하위 영역에 특화된 전문가를 배치, 이 전문가들이 협력하여 문제를 해결하도록 하는 Mixture-of-Prompts(MoP) 방법을 제안하였습니다.

- **Technical Details**: 이 연구에서는 두 단계의 과정을 통해 각 영역에 맞는 전문가를 구축합니다. 첫 번째 단계는 '데모 할당(demo assignment)'입니다. 이는 이론적으로 맥락 학습과 커널 회귀(Kernel Regression) 사이의 연결을 토대로 데모들을 의미적 유사성에 따라 전문가 그룹으로 묶는 것입니다. 두 번째 단계는 '지시 할당(instruction assignment)'로, 각 전문가에게 최적의 지시를 할당하여 데모와 지시를 조합, 시너지 효과를 극대화하는 것입니다. 이 과정에서 각 전문가의 지시와 데모가 함께 최적화되어 각각의 하위 지역에서 구체적이고 세세한 지식을 제공하게 됩니다.

- **Performance Highlights**: 제안된 Mixture-of-Prompts(MoP)는 주요 벤치마크에서 81%의 평균 승률을 기록하였습니다. 이는 Instruction-Induction, Super Natural Instructions, 및 BIG-Bench-Hard 등의 벤치마크에서 기존의 여섯 가지 대표적인 방법들을 능가하는 성과입니다. 연구 결과에 따르면, 의미 공간에서 데모를 클러스터링하는 것은 테스트 샘플을 정확하게 할당하는 데 매우 효과적이며, 각 데모 클러스터에 맞는 최적의 지시를 함께 검색하는 것이 필요함을 보여줍니다.



### Mind the Gap: Analyzing Lacunae with Transformer-Based Transcription (https://arxiv.org/abs/2407.00250)
Comments:
          Accepted to ICDAR 2024 Workshop on Computational Paleography

- **What's New**: 이 연구는 손상된 역사적 문서의 복원 문제를 다루며, Transformer 기반 OCR 모델을 사용하여 lacunae(결손 부분)를 감지하고 복원하는 방법을 제안합니다. 기존 모델이 5% 성공률에 그쳤던 것에 비해, 제안된 모델은 65%의 복원 성공률을 달성했습니다. 또한, 이 모델은 라인 이미지 내의 lacunae와 기타 오류를 탐지할 수 있습니다.

- **Technical Details**: 이 연구에서는 Transformer 기반 OCR 모델(TrOCR)을 사용하여 lacunae가 포함된 합성 데이터를 통해 모델을 학습시켰습니다. TrOCR 모델은 사전 학습된 Vision Transformer와 Language Model을 결합한 구조를 가지고 있습니다. 실험에서는 IAM handwriting database를 사용하여 lacunae를 포함한 학습 데이터를 생성했으며, 이 데이터를 통해 모델을 훈련했습니다. 또한, 로그 확률(log probability)과 주의 메커니즘(attention mechanism)을 활용하여 모델의 성능을 평가했습니다.

- **Performance Highlights**: TrOCR 모델은 lacunae 복원에서 기존 베이스라인 모델보다 크게 향상된 성능을 보였습니다(5.6% -> 65.85%). 또한, HTR 트랜스크립트의 로그 확률을 사용한 예측 모델은 lacunae가 포함된 라인 이미지를 53%의 정확도로, 기타 오류가 포함된 이미지를 84%의 정확도로 탐지할 수 있음을 보였습니다.



### Granite-Function Calling Model: Introducing Function Calling Abilities via Multi-task Learning of Granular Tasks (https://arxiv.org/abs/2407.00121)
- **What's New**: 최근 발표된 논문에서는 대형 언어 모델(LLMs)이 독립적인 에이전트 시스템으로 작동하는 가능성을 크게 확장하고 있습니다. 이 논문에서는 새로운 모델인 GRANITE-20B-FUNCTIONCALLING을 소개하며, 이 모델은 Apache 2.0 라이선스 하에 공개되었습니다. 이 모델은 엔드 투 엔드로 복잡한 작업을 수행하고 외부 도구 및 API와 상호작용을 개선하는 데 중점을 둡니다.

- **Technical Details**: GRANITE-20B-FUNCTIONCALLING 모델은 7가지 기본 작업(예: Nested Function Calling, Function Chaining, Parallel Functions, Function Name Detection, Parameter-Value Pair Detection, Next-Best Function, 그리고 Response Generation)을 포함한 멀티 태스킹 접근 방식을 통해 학습되었습니다. API-Blend 데이터셋을 사용하여 교육되었으며, 다양한 프로그래밍 언어를 사용한 시퀀싱, 슬롯 채우기, 다중 함수 등 여러 작업을 지원합니다. 이 모델은 IBM의 AI 윤리 원칙에 따라 라이선스 허용 데이터를 사용하여 신뢰성 있는 엔터프라이즈 사용을 위해 설계되었습니다.

- **Performance Highlights**: 성능 평가에서 GRANITE-20B-FUNCTIONCALLING 모델은 Berkeley Function Calling Leaderboard(BFCL)에서 최상위 공개 모델들과 비교하여 우수한 성능을 보였습니다. 이 모델은 일반화 가능성이 높아 7개의 다른 평가 데이터셋에서도 높은 성능을 보였으며, 특히 Meta-Llama-3-70B-Instruct보다 더 나은 성능을 발휘했습니다. 전체적으로는 BFCL에서 네 번째로 높은 성능을 기록했습니다.



### Accurate Prediction of Ligand-Protein Interaction Affinities with Fine-Tuned Small Language Models (https://arxiv.org/abs/2407.00111)
- **What's New**: 본 논문에서는 지시(fine-tuning)된 사전 학습된 생성 언어 모델(SLM, Small Language Model)을 사용하여 리간드-단백질 상호작용(LPI, Ligand-Protein Interaction) 친화도, 즉 약물-대상 상호작용(DTI, Drug-Target Interaction)의 정확한 예측을 설명합니다. 이 모델은 리간드의 SMILES 문자열과 단백질의 아미노산 서열만을 입력으로 사용하여 샘플 외 데이터(out-of-sample data)에서 제로샷 설정(zero-shot setting)으로 친화도를 예측하는 데 성공했습니다.

- **Technical Details**: 저자들은 수백만 개의 파라미터를 가진 생성 언어 모델을 출발점으로 사용했습니다. 이러한 소형 파운데이션 모델은 도메인 특화 데이터에 대해서 몇 번의 epoch 동안 지시 학습(fine-tuning)되었습니다. 모델 입력으로는 리간드의 SMILES 문자열과 표적 단백질의 아미노산 서열만을 사용하여 예측 정확도를 평가했습니다. 이 과정에서 고유한 평가 프레임워크가 사용되었습니다.

- **Performance Highlights**: 다양한 리간드-단백질 상호작용 친화도를 예측하는 데 있어 기존의 머신러닝(ML) 및 자유 에너지 섭동(FEP, Free-Energy Perturbation) 기반 방법보다 명확한 개선을 보여주었으며, 이는 특히 어려운 치료 타겟에 대해 신약 개발 캠페인을 가속화하는 데 활용될 수 있습니다.



### A Case Study on Contextual Machine Translation in a Professional Scenario of Subtitling (https://arxiv.org/abs/2407.00108)
Comments:
          Accepted to EAMT 2024

- **What's New**: 최근 연구에서 필름 메타데이터와 같은 추가적인 문맥 정보를 기계 번역(MT) 파이프라인에 통합하면 번역 품질이 향상된다고 밝혀졌습니다. 그러나 이러한 시스템이 실제 산업 현장에서 긍정적인 영향을 미치는지는 아직 증명되지 않았습니다. 본 연구는 TV 자막을 번역하는 전문 시나리오에서 기계 번역의 이점을 조사했으며, 문맥 정보 활용이 후편집 과정에 미치는 영향을 집중 조사했습니다.

- **Technical Details**: 이 연구는 ZOO Digital의 번역 및 후편집 전문가들의 도움을 받아 진행되었습니다. 연구에서는 두 가지 추가 시스템을 조사했습니다: 대규모 데이터로 훈련된 전문 NMT 엔진인 Base-NMT와 필름 메타데이터 및 문서 수준 정보를 관찰하여 훈련된 문맥 기반 MTCue 아키텍처를 기반으로 하는 컨텍스트 모델입니다.

- **Performance Highlights**: 연구 결과, MTCue 모델의 출력을 후편집한 후편집자들은 비문맥 모델보다 문맥 관련 오류를 현저히 적게 표시했습니다. 또한, 후편집자들을 대상으로 한 설문 조사에서는 문맥적 부적합성이 기계 번역에서 일관되게 관찰되는 중요한 격차로 강조되었습니다. 이러한 결과는 완전한 문맥적 MT에 대한 추가 연구의 동기부여를 강화합니다.



### UnUnlearning: Unlearning is not sufficient for content regulation in advanced generative AI (https://arxiv.org/abs/2407.00106)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)에 대한 '정확한 학습 취소'(Exact Unlearning)와 '정확하지 않은 학습 취소'(Inexact Unlearning) 방식을 재검토하고, 기존 학습 취소 방식의 한계를 강조합니다. 특히, 인-컨텍스트 학습(In-Context Learning)으로 인해 학습된 지식을 제거하더라도 다시 모델에서 재현될 수 있는 'Ununlearning' 개념을 도입합니다. 이는 정책적으로 제거해야 하는 생물학적 또는 핵 지식 등의 문제가 발생할 수 있다고 설명합니다.

- **Technical Details**: 학습 취소는 원래 사용자가 요청 시 기계 학습 모델에서 데이터를 제거하는 프라이버시 메커니즘으로 소개되었습니다. 하지만 대형 언어 모델에서 불법적이거나 유해한 지식의 제거에 대한 규제를 위해서는 'Ununlearning'이라는 새로운 개념을 도입해야 한다고 주장합니다. 이는 모델 내부의 인-컨텍스트 학습 능력으로 인해 기존에 제거된 지식이 다시 등장하는 문제를 다룹니다.

- **Performance Highlights**: 해당 연구는 정확한 학습 취소 방식도 모델이 예측 단계에서 불법적이거나 유해한 행동을 하지 못하도록 보장하지 못한다고 주장합니다. 예를 들어, 모델이 'Bomb'이라는 단어를 모르도록 학습시켜도, 관련된 화학 지식을 통해 여전히 폭탄 제조법을 유추할 수 있습니다. 따라서, 학습 취소만으로는 이러한 문제를 완벽하게 해결할 수 없으며, 콘텐츠 필터링(Content Filtering)이 추가로 필요합니다.

- **Broader Implications**: 학습 취소가 장기적으로 콘텐츠 규제의 주요 도구가 될 수 있는지에 대한 의문이 제기됩니다. 이는 단순히 특정 지식을 제거하는 것만으로는 충분하지 않으며, 모델 행동을 실제로 제어하기 위한 보다 포괄적인 접근이 필요함을 의미합니다.



### Enhancing In-Context Learning via Implicit Demonstration Augmentation (https://arxiv.org/abs/2407.00100)
Comments:
          Accepted by ACL 2024 Main 19 pages,10 figures

- **What's New**: 본 논문은 대형 사전 학습 언어 모델(PLM, Pre-trained Language Models)의 in-context learning(ICL)을 개선하기 위해 데몬스트레이션 증강에 중점을 둡니다. ICL은 새로운 입력에 대해 모델 파라미터를 업데이트하지 않고 예측을 수행할 수 있게 합니다. 하지만 데몬스트레이션의 품질, 양, 순서에 크게 좌우되는 등 최적성과 안정성이 부족한 문제가 있었습니다. 이를 해결하기 위해, 본 연구에서는 데몬스트레이션 증강을 통해 이러한 성능 저하 문제를 처음으로 해결하고자 합니다.

- **Technical Details**: 논문은 데몬스트레이션의 깊은 특징 분포를 활용하여 그 대표성을 풍부하게 하는 것에서 시작합니다. 증강된 복사본의 수가 무한에 가까워질 때, 이는 새로운 logit 보정 메커니즘과 통계적 특성을 통합한 것과 비슷한 결과를 가져온다는 것을 이론적으로 증명합니다. 이를 통해 파라미터를 효율적으로 설정하여 다양한 PLM과 작업에서 평균 및 최악의 정확도를 크게 향상시키는 간단하고 효율적인 방법을 제시합니다.

- **Performance Highlights**: 본 방법론을 사용하면 데몬스트레이션, 순서, 템플릿 간의 성능 변동성을 효과적으로 줄일 수 있으며, 불균형 클래스 분포 문제를 해결하는데도 기여할 수 있습니다. 실험 결과, 여러 PLM과 다양한 분류 작업에서 평균 및 최악의 정확도도 크게 향상 시켰으며, 성능 안정성도 제고되었습니다.



### ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback (https://arxiv.org/abs/2407.00087)
- **What's New**: 본 논문에서는 Large Multimodal Models (LMMs)를 강화하기 위한 새로운 알고리즘 ARES를 제안했습니다. 이 알고리즘은 두 단계를 통해 Reinforcement Learning (RL)과 Supervised Fine-Tuning (SFT)을 번갈아 수행합니다. 먼저, Teacher 모델(예: GPT-4, Claude 3 Opus)에서 각 문장이 문제 해결에 얼마나 기여하는지에 대한 점수를 요청합니다. 두 번째 단계에서는 RL 이후 잘못된 추론을 Teacher가 수정하게 하고, 이를 통해 모델을 SFT로 안정화시킵니다. ScienceQA와 A-OKVQA 다중모델 데이터셋에서 테스트한 결과, ARES는 약 70%의 승률과 평균 2.5%의 추론 정답률 향상을 보였습니다.

- **Technical Details**: ARES 알고리즘의 첫 번째 단계에서는 Teacher 모델이 각 문장의 기여도를 0.0에서 1.0까지 점수로 평가합니다. 이 세부적인 피드백은 수학 문제와 일반적인 Chain-of-Thought (CoT) 문제 모두에 적용할 수 있는 보다 세밀한 보상을 제공합니다. 두 번째 단계에서는 Teacher가 RL 과정에서 발생한 반복되는 단어나 불완전한 문장을 수정하여 새로운 데이터셋을 생성하고, 이를 SFT로 모델을 추가로 튜닝합니다. 이 과정에서 대규모 하이퍼파라미터 튜닝의 부담을 줄일 수 있습니다.

- **Performance Highlights**: ScienceQA와 A-OKVQA 데이터셋에서 ARES를 테스트한 결과, MM-CoT와 비교하여 ARES의 추론사고가 GPT-4o에 의해 평가된 승률이 약 70%로 나타났습니다. 또한, 개선된 추론사고는 다중모델 데이터셋에서 평균 2.5%의 추론 정답률 향상을 가져왔습니다. 이는 ARES가 기존 모델들보다 더 나은 성능을 보였음을 의미합니다.



### Logicbreaks: A Framework for Understanding Subversion of Rule-based Inferenc (https://arxiv.org/abs/2407.00075)
- **What's New**: 이번 연구에서는 언어 모델이 규칙을 따라야 하는 상황에서 이를 어떻게 왜곡할 수 있는지를 분석합니다. 이를 통해 대규모 언어 모델(Large Language Models, LLMs)이 논리적 추론과 규칙 기반 설정에서의 '탈옥(jailbreak)' 공격을 이해하는 데 도움을 제공합니다.

- **Technical Details**: 연구자들은 규칙 따르기를 명제적 호른 논리(다음과 같은 형식: '만약 P와 Q라면 R')로 모델링 했습니다. 이를 바탕으로 '단일층 및 단일 self-attention head'를 사용하는 이론적 트랜스포머 모델을 구성하여 데이터 학습 모델을 공격하는 방법을 연구했습니다.

- **Performance Highlights**: 이론적으로 구성된 공격이 실제로 학습된 트랜스포머 모델에서도 성공하며, 특히 두 가지 공격이 이론적 설정과 학습된 추론기 모두에서 성공했습니다. 이로 인해 이론적 공격과 실제 LLM 탈옥 공격 간의 유사성을 입증했습니다.



### Pistis-RAG: A Scalable Cascading Framework Towards Trustworthy Retrieval-Augmented Generation (https://arxiv.org/abs/2407.00072)
- **What's New**: 최근 발표된 논문에서 Pistis-RAG라는 대규모 검색 강화 생성 시스템(Retrieval-Augmented Generation, RAG)의 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 LLM(대형 언어 모델)이 외부 지식 랭킹 방법과의 강한 정렬이 부족하다는 기존 문제를 해결하고, 모델 중심 패러다임 대신 내용 중심 접근 방식을 채택합니다. 특히, 랭킹 단계를 재설계하여 사용자의 피드백과 LLM의 선호도를 반영한 생성 품질을 향상시킵니다.

- **Technical Details**: Pistis-RAG는 매칭, 프리랭킹, 랭킹, 재랭킹의 다단계 검색 파이프라인을 활용합니다. 매칭 단계에서는 검색 공간을 정제하고, 프리랭킹 단계에서는 의미적으로 관련 있는 문서를 우선시하며, 랭킹 단계에서는 LLM의 선호도와 사용자 피드백을 고려하여 내용을 정렬합니다. 또한, 복잡한 Chain-of-Thought(연쇄적 사고) 방법론을 지원하는 Reasoning 및 Aggregating 단계가 포함되어 있습니다. 이 접근 방식은 외부 지식과 LLM 간의 원활한 통합을 핵심 원칙으로 삼아, 각 특정 작업에 맞춘 내용 변환 과정을 최적화합니다.

- **Performance Highlights**: Pistis-RAG는 MMLU 벤치마크 실험에서 9.3%의 성능 향상을 보여줍니다. 또한, 실제 대규모 데이터에서도 프레임워크의 확장 가능성을 검증하였습니다. 이로써 기존 방법보다 더욱 관련성 있고 개인화된 결과를 제공하며, LLM의 프롬프트 순서 민감성을 고려하는 새로운 랭킹 메커니즘을 도입하였습니다.



### Combinatorial Reasoning: Selecting Reasons in Generative AI Pipelines via Combinatorial Optimization (https://arxiv.org/abs/2407.00071)
Comments:
          13 pages, 3 figures

- **What's New**: 최근 거대 언어 모델(LLMs)이 다양한 인간 지능 요구 작업에서 뛰어난 성능을 보이고 있지만, 여전히 추론 작업에서는 성능이 미흡합니다. 이를 해결하기 위해 저희는 '조합 추론(Combinatorial Reasoning, CR)'이라는 프레임워크를 도입했습니다. 이 프레임워크는 LLM 파이프라인에서 이유(reason)를 샘플링하고 이를 이차 비제한 이진 최적화(Quadratic Unconstrained Binary Optimization, QUBO) 문제로 매핑해 완전 자동화된 프롬프트 생성 방법을 제안합니다.

- **Technical Details**: CR 프레임워크는 LLM의 합리적인 경로를 선택하는 데 QUBO 솔루션을 사용하는 방안을 조사합니다. QUBO는 많은 산업 도메인에서 발생하는 도전적인 조합 최적화 문제를 해결하는 데 사용되며, 이를 통해 LLM 프롬프트 생성에 필수적인 최적의 이유(reason)의 부분 집합을 선택할 수 있습니다. 저희 연구는 물리 기반 솔버를 통합한 첫 LLM 파이프라인을 구축하고 여러 NLP 추론 벤치마크를 통해 이 파이프라인의 성능을 평가했습니다.

- **Performance Highlights**: 초기 연구 결과, CR 프레임워크가 몇 가지 BigBench-Hard 추론 작업에서 다른 제로샷(Zero-shot) 프롬프트 전략보다 개선된 성능을 보였습니다. 일부 경우에는 인간 수준의 추론 성능을 달성했습니다. 또한, 특화된 솔버를 통한 CR 가속화와 간단한 제로샷 전략의 성능(예: 직선 다수결 규칙 또는 이유의 임의 선택)도 조사했습니다.



### Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead (https://arxiv.org/abs/2407.00066)
- **What's New**: 이 논문에서는 LoRA(낮은 순위 어댑터)로 미세 조정된 대형 언어 모델(LLM)을 압축하여 실시간 쿼리 응답 시스템의 성능을 개선하는 방법을 제안했습니다. 수백에서 수천 개의 LoRA를 한 번에 서비스해야 하는 경우, LoRA를 GPU 메모리에 지속적으로 로드하고 언로드하는 비용이 문제가 됩니다. 이를 해결하기 위해, 저자는 SVD(단일값 분해)를 사용한 개별 압축과 공유 기저와 LoRA 특유의 스케일링 행렬을 사용하는 공동 압축을 제안합니다.

- **Technical Details**: LoRA 업데이트는 뉴럴 네트워크의 고정 가중치 행렬 W₀를 업데이트하기 위해 매트릭스 A와 B의 곱으로 구성됩니다. 저자는 LoRA 어댑터의 주어진 컬렉션을 압축하는 문제를 재구성 문제로 공식화하였습니다. 두 가지 접근 방식을 조사합니다: 개별 LoRA의 랭크를 낮추어 각각 압축하는 방법(SVD)과 공유 기저와 LoRA 특유의 스케일링 행렬을 사용하는 방법. 이 두 가지 접근 방식은 상호보완적인 트레이드오프를 가지며, 각각의 방법론을 통해 다른 응용에서도 유사한 이점을 얻을 수 있습니다.

- **Performance Highlights**: 최대 500개의 LoRA에서 실험 결과, 압축된 LoRA는 성능을 유지하며 수천 개의 LoRA가 요청될 경우에도 처리량에서 큰 이점을 제공했습니다. 1000개 이상의 LoRA를 서비스하면서 기본 LLM을 서비스하는 처리량의 75%를 유지했습니다. 이를 통해 개별 및 공동 압축 방식 간의 트레이드오프를 분석하고, vLLM 시스템에 LoRA 압축 기능을 통합하여 비동기 요청을 처리하는 성능을 입증했습니다. 추가로, 저자는 Mistral-7B-Instruct-v0.2 모델을 기반으로 500개의 LoRA를 훈련하여 성능을 보존할 수 있음을 보여주었으며, 이를 공개하여 향후 연구를 촉진할 예정입니다.



### A Document-based Knowledge Discovery with Microservices Architectur (https://arxiv.org/abs/2407.00053)
- **What's New**: 이 논문은 조직 내 디지털 기반 지식을 향상시키기 위한 새로운 접근법을 제안합니다. 초점은 미세 서비스 아키텍처(microservices architecture)를 도입하여 키워드 추출, 문서의 유사성 계산, 자연어 데이터베이스 쿼리, 그리고 프로그래밍 언어 독립적인 정보 제공을 가능케 하는 데 있습니다. 이는 독일 특허청에서 확장된 버전으로 사용되고 있는 현대적인 방법론이기도 합니다.

- **Technical Details**: 제안된 개념적 디자인은 반자동 학습, 편집, 온톨로지 시각화를 위한 프로세스 및 애플리케이션 통합을 위한 참조 설계 지침을 제공합니다. 또한 미세 서비스 아키텍처를 사용하여 확장성(Scalability)과 복원력(Resilience) 같은 비기능적 요구사항을 해결합니다.

- **Performance Highlights**: 논문에서 제시된 요구사항의 평가를 위해 구현된 시연기를 통해 개념을 평가했습니다. 이는 문서 기반 지식 발견(KD) 시스템의 아키텍처적 요구사항을 충족시키는 데 중요한 역할을 합니다. 이 접근법은 특히 특허청의 분류, 검색, 검토 업무의 효율성을 크게 향상시킬 수 있습니다.



### One Queue Is All You Need: Resolving Head-of-Line Blocking in Large Language Model Serving (https://arxiv.org/abs/2407.00047)
- **What's New**: 대규모 언어 모델(LLMs)이 클라우드 제공업체들에게 점점 더 중요한 작업이 되고 있습니다. 이를 위해, LLM 추론 요청의 엔드투엔드 지연 시간 SLO(Service Level Objectives)을 달성하는 것이 중요합니다. 그러나 현재의 LLM 제공 시스템은 요청 처리량이나 요청 실행 지연 시간에 중점을 두고 있습니다. 이를 해결하기 위해 QLM이라는 멀티 모델 큐 관리 프레임워크를 제안합니다.

- **Technical Details**: QLM은 여러 LLM 제공 작업(LSOs)을 조율하여 HOL(Head-Of-Line) 블로킹을 줄이고 SLO 달성을 극대화합니다. 주요 LSOs에는 모델 교체, 요청 퇴출, GPU-CPU 상태 교체, 부하 분산, 따뜻한 모델 시작 등이 포함됩니다. QLM은 확률적 프로그래밍을 사용하여 이러한 LSOs를 최적화합니다. QLM은 가상 큐 추상화와 계획 생성기 및 요청 완료 시간(RCT) 추정기를 사용하여 큐 관리 프레임워크를 구성합니다.

- **Performance Highlights**: QLM은 이기종 GPU 장치와 실제 LLM 제공 데이터셋을 사용한 평가에서 SLO 달성을 40-90%, 처리량을 20-400% 개선한 반면, 장치 사용률을 유지하거나 개선했습니다. 모델 워밍 스타트는 멀티 모델 제공에서 처리량을 300% 개선하고, 요청 퇴출은 싱글 모델 제공에서 SLO 달성을 80% 개선하는 데 기여했습니다.



### Visual Language Model based Cross-modal Semantic Communication Systems (https://arxiv.org/abs/2407.00020)
Comments:
          12 pages, 10 figures

- **What's New**: 새로운 Vision-Language Model 기반 Cross-modal Semantic Communication(VLM-CSC) 시스템을 제안합니다. 이 시스템은 기존 ISC(이미지 의미 통신) 시스템이 동적인 환경에서 맞닥뜨리는 여러 문제점들을 해결하는 것을 목표로 합니다. VLM-CSC 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Cross-modal Knowledge Base(CKB)는 고밀도 텍스트 의미를 추출하여 전송하고, 수신기에서 원본 이미지를 재구성합니다. 2) Memory-assisted Encoder and Decoder(MED)는 하이브리드 장/단기간 메모리 메커니즘을 활용하여 동적 환경에서의 의미 인코더와 디코더의 성능 저하를 방지합니다. 3) Noise Attention Module(NAM)은 주어진 SNR(Signal-to-Noise Ratio)에 따라 의미 코딩과 채널 코딩을 적응적으로 조정합니다.

- **Technical Details**: VLM-CSC 시스템은 다음과 같은 핵심 기술들을 포함합니다. CKB는 송신기에서 BLIP 기반의 KB를 사용하여 고품질의 텍스트 설명을 생성하고, 수신기에서 SD 기반의 KB를 사용하여 텍스트 설명에 맞춘 이미지를 재구성합니다. 이는 전송되는 의미의 밀도를 증가시키고, 시스템의 설명 가능성을 향상시킵니다. 또한 MED는 STM(Short-Term Memory)과 LTM(Long-Term Memory)을 사용하는 스토리지 풀을 설계하여 동적 환경 변화를 추적하고, 교육 데이터의 분포 변화로 인한 성능 저하를 방지합니다. 마지막으로 NAM은 주어진 SNR 값에 따라 각 인코더와 디코더층 뒤에 주목 모듈을 사용하여 가중치를 조정함으로써 의미 피처의 전송 강도를 유지합니다.

- **Performance Highlights**: 실험 결과는 VLM-CSC 시스템의 효율성, 적응성 및 견고성을 검증합니다. CKB, MED 및 NAM의 조합을 통해 시스템은 높은 대역폭 압력을 감소시키고, 동적 환경에서의 성능 저하를 방지하며, 다양한 SNR 조건에서도 높은 견고성을 유지할 수 있습니다.



New uploads on arXiv(cs.IR)

### ColPali: Efficient Document Retrieval with Vision Language Models (https://arxiv.org/abs/2407.01449)
Comments:
          Under Review

- **What's New**: ViDoRe라는 새로운 Visual Document Retrieval Benchmark가 도입되었습니다. 이 벤치마크는 다양한 도메인, 언어, 설정에서 페이지 수준의 검색 작업을 포함하여 현대 시스템의 시각적 요소를 효율적으로 활용하지 못하는 문제를 해결하려는 목적을 가지고 있습니다. ViDoRe를 통해 문서 검색 시스템의 성능을 평가하는 기회를 제공합니다.

- **Technical Details**: 기존의 문서 검색 시스템들은 텍스트 임베딩 모델의 성능에 주로 의존하였으나, 실질적인 문서 검색에서는 문서의 시각적 요소도 효율적으로 활용되어야 합니다. 이를 위해 새로운 모델 구조인 ColPali가 제안되었습니다. ColPali는 Vision Language Models (VLMs)을 활용하여 문서의 이미지에서 고품질 문맥화된 임베딩을 생성하며, 빠른 쿼리 매칭을 위해 늦은 상호작용 매커니즘을 결합했습니다.

- **Performance Highlights**: ColPali는 ViDoRe 벤치마크에서 대부분의 기존 문서 검색 시스템을 능가하며, 빠르고 엔드 투 엔드 트레이닝이 가능함을 보여주었습니다. 이를 통해 문서 페이지 이미지만으로도 높은 성능의 문서 검색이 가능함을 입증하였습니다.



### Optimization of Retrieval-Augmented Generation Context with Outlier Detection (https://arxiv.org/abs/2407.01403)
- **What's New**: 이 논문은 질의응답 시스템에서 필요한 프롬프트 컨텍스트의 크기를 줄이고, 품질을 향상시키는 방법을 다룹니다. 대형 언어 모델(LLM)이 질의에 대한 응답을 생성하는 동안, 많은 문서를 검색하려는 시도는 처리 복잡도를 증가시키고 성능을 저하시킬 수 있습니다. 본 연구는 질의와 가장 의미가 일치하는 문서를 선택하고, 불필요한 문서는 이상치로 처리하여 필터링하는 새로운 방법을 제안합니다.

- **Technical Details**: 이 논문에서 제안하는 방법은 검색된 문서들을 임베딩 벡터로 변환하고, 이 벡터들의 중심(centroid) 및 질의 벡터와의 거리 차이를 이용해 이상치를 식별하는 것입니다. 다양한 결합 방법(연결, 가중 합, 상호작용, 다항식)으로 거리를 특징화해 이상치를 구분합니다. Gaussian Mixture Model(GMM)과 로그 가능도 접근법을 사용하여 이상치를 식별합니다. 이를 통해 보다 복잡한 질문에 대한 응답 품질을 향상시키는데 성공했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법을 통해 복잡한 질문 및 답변에 대해 가장 큰 향상을 얻었습니다. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' 모델을 사용해 기본 실험을 진행하였으며, 'mistralai/Mistral-7B-Instruct-v0.2' 모델로도 유사한 결과를 얻었습니다. 실험에는 SQuAD2.0 데이터셋을 사용했습니다. 이러한 실험을 통해 필터링된 컨텍스트가 보다 나은 응답을 제공함을 확인하였습니다.



### Evaluation of Temporal Change in IR Test Collections (https://arxiv.org/abs/2407.01373)
- **What's New**: 이 논문은 기존 Cranfield 패러다임의 정보 검색(IR) 평가에 시간적 변화를 도입하여 평가 결과의 시간적 일반화를 연구합니다. 이는 문서 컬렉션, 주제 트렌드, 사용자 인식의 변화와 같은 다양한 요인의 영향을 고려합니다. 시간적 변화에 따른 검색 시스템의 성능 변화를 평가하기 위해 CRUD(Create, Update, Delete) 작업을 바탕으로 문서와 관련성 레이블의 변화를 분류하고, 이를 통한 다양한 평가 시나리오를 도출했습니다.

- **Technical Details**: 연구는 시간적 변화를 설명하고 효과에 미치는 영향을 측정하는 데 중점을 둡니다. 제안된 방법론은 TripClick, TREC-COVID, LongEval와 같은 세 가지 테스트 컬렉션을 사용해 다섯 개의 최신 IR 시스템을 반복 평가함으로써 검증되었습니다. 각 테스트 컬렉션은 시간적 변화를 다양한 수준에서 포괄합니다.

- **Performance Highlights**: 실험 결과, 검색 시스템의 성능이 평가 시나리오에 크게 의존하며, 시간적 변화가 시스템 간의 상대적 성능에도 강한 영향을 미친다는 사실을 확인했습니다. 제안된 방법론은 시간에 따른 성능 변화를 잘 설명할 수 있었으며, 이를 통해 시스템과 테스트 컬렉션에 대한 새로운 통찰을 얻을 수 있었습니다. 향후 연구 방향으로는 시간적 변화를 고려한 평가 방법론의 개선, 테스트 컬렉션 유지 관리, 재사용성 연구 등이 제시되었습니다.



### Deep Domain Specialisation for single-model multi-domain learning to rank (https://arxiv.org/abs/2407.01069)
- **What's New**: 이 연구는 여러 도메인(domain)을 단일 모델로 통합하는 새로운 접근법인 Deep Domain Specialisation(DDS)을 소개합니다. 기존의 방법들과 비교해 DDS는 각 도메인별 모델보다 적은 파라미터를 사용하면서도 우수한 성능을 보입니다. Amazon 고객 트래픽을 대상으로 한 대규모 온라인 실험에서도 그 효과가 검증되었습니다.

- **Technical Details**: 프로덕션 시스템에서는 여러 도메인에 대해 별도의 랭킹 모델을 사용하는 경우가 많습니다. 이는 도메인 전이(adaptation)을 위한 중간 계층을 학습하도록 강제하는 Deep Domain Adaptation(DDA) 방식을 주로 사용하지만, 본 연구에서는 도메인 특화 표현을 학습하는 DDS 방식을 사용합니다. DDS는 BERT 기반의 텍스트 인코더를 사용하며, 쿼리-제품 유사성 특징을 뽑아내고, 트랜스포머 기반의 리스트와이즈(listwise) 스코어링 아키텍처를 사용합니다. 최종적으로 특정 도메인 인디케이터를 곱해 최종 구매 확률 점수를 생성합니다.

- **Performance Highlights**: 실험 결과, 두 개의 지리적 도메인(아랍에미리트와 사우디아라비아)를 대상으로 단일 모델이 각각의 독립 모델과 동등하거나 더 나은 성능을 보였습니다. 특히, NDCG@16 지표에서 DDS 모델은 아랍에미리트 도메인에서 0.51%의 성능 향상을 보였습니다.



### ProductAgent: Benchmarking Conversational Product Search Agent with Asking Clarification Questions (https://arxiv.org/abs/2407.00942)
Comments:
          17 pages, 13 tables, 6 figures. Under review

- **What's New**: 이 논문은 사용자가 불명확한 질문으로 대화를 시작하는 전자상거래 시나리오에서 제품 수요 명확화 작업(task)을 도입합니다. 이 작업에서는 명확화 질문(clarification questions)을 통해 보다 정확하고 맞춤형의 제품 검색을 수행하는 역할 지향 에이전트를 설계합니다. 이를 위해 ProductAgent라는 대화형 정보 검색 에이전트를 제안하고 있으며, 이 에이전트는 전략적인 명확화 질문 생성 및 동적 제품 검색 기능을 갖추고 있습니다. 또한 LLM 기반 사용자 시뮬레이터를 활용한 PROCLARE이라는 벤치마크를 제안하여 에이전트의 성능을 자동 및 정성적으로 평가합니다.

- **Technical Details**: ProductAgent는 제품 데이터베이스, 메모리 모듈 및 도구 세트를 통합하여 자동 루프를 수행합니다. 1) 제품 데이터베이스는 구조적 및 벡터화된 형태로 제품 항목을 저장하여 관련 제품을 검색 및 요약할 수 있게 합니다. 2) 메모리 모듈은 대화 세션 동안 사용자의 구조적 명확화 질문 및 비구조적 대화 기록을 캐시하여 다음 질문을 동적으로 할 수 있게 합니다. 3) 다양한 작업(예: 제품 검색 및 명확화 질문 생성)을 지원하는 도구 세트를 제공합니다. 이러한 모듈과의 효과적인 상호작용을 위해 설계된 프롬프트를 활용하여, 언어와 머신러닝 모델(LLMs) 간의 상호작용을 자동화했습니다.

- **Performance Highlights**: PROCLARE 벤치마크를 사용하여 ProductAgent의 성능을 평가한 결과, 대화 턴이 증가할수록 사용자의 요구가 점점 더 명확해지고 제품 검색 성능이 향상된다는 것을 확인했습니다. 2,000개의 대화 데이터셋을 활용한 실험을 통해 ProductAgent의 효과성을 실증했습니다.



### Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation (https://arxiv.org/abs/2407.00912)
- **What's New**: 새로운 연구는 추천 시스템과 검색 시스템을 통합하려는 시도로, 사용자의 고유한 변하지 않는 의도(예: 항상 높은 품질의 아이템을 선호)와 변화하는 수요 의도(예: 여름에는 티셔츠, 겨울에는 패딩 재킷을 원함)를 더 정확하게 모델링하기 위해 설계되었습니다. 새로운 모델 'Unified Dual-Intents Translation for joint modeling of Search and Recommendation (UDITSR)'을 제안했습니다.

- **Technical Details**: UDITSR 모델은 '수요 의도 생성기(검색 데이터를 감독 정보로 사용)'와 '이중 의도 번역 전파 메커니즘'으로 구성됩니다. 수요 의도 생성기는 검색 쿼리를 이용해 사용자의 변화하는 의도를 더 정확하게 파악합니다. 이에 더해, 고유 의도와 수요 의도, 상호작용 아이템 간의 관계를 명확하게 모델링하기 위해 임베딩 번역(embedding translations)을 활용하는 이중 의도 번역 메커니즘을 제안하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, UDITSR 모델이 검색 및 추천 과제 모두에서 기존 SOTA 모델들보다 뛰어난 성능을 보임을 입증했습니다. 모델의 효과를 심층적으로 이해하기 위해 관련 의도에 대한 시각적 분석도 제공됩니다.



### Heterogeneous Graph-based Framework with Disentangled Representations Learning for Multi-target Cross Domain Recommendation (https://arxiv.org/abs/2407.00909)
- **What's New**: 이 논문에서는 다중 도메인 정보(cross-domain) 이용을 통해 데이터 희소성 문제를 해결하는 방법을 제안했습니다. 기존의 연구는 단일 타겟(single-target) 또는 이중 타겟(dual-target) 도메인 추천에 집중했고, 다중 타겟(Multi-target) 도메인 추천에 대한 연구는 부족했습니다. 이 논문에서는 이 문제를 해결하기 위해 이종 그래프(Heterogeneous Graph) 기반의 분리된 표현 학습(disentangled representations learning)을 제안합니다.

- **Technical Details**: 제안된 HGDR(Heterogeneous Graph-based Framework with Disentangled Representations Learning) 모델은 이종 네트워크 아키텍처로, 그래프 컨볼루션 레이어(graph convolutional layers)를 사용하여 다양한 도메인 간의 관계를 모델링합니다. 또한, 사용자와 아이템에 대해 도메인 공용 및 도메인 특유의 정보를 분리하여 학습합니다. 이 모델은 사용자와 아이템의 ID 만으로 글로벌 이종 그래프를 생성하며, 도메인 간의 정보 이전을 효율적으로 수행합니다. 네트워크는 매핑 레이어(mapping layer)와 분리된 컨볼루션 구조(disentangled convolutional structure)를 포함하여 두 가지 주요 부분으로 구성됩니다.

- **Performance Highlights**: 실제 데이터 세트와 온라인 A/B 테스트 실험 결과, 제안된 HGDR 모델은 도메인 간의 정보를 효과적으로 전달하며 최신(SOTA) 성능을 달성했습니다. 본 연구는 다중 도메인 추천 시스템에서 모델의 복잡성과 다중 도메인의 성능 향상 어려움을 극복하는 중요한 기여를 했습니다.



### Prediction of Sentinel-2 multi-band imagery with attention BiLSTM for continuous earth surface monitoring (https://arxiv.org/abs/2407.00834)
- **What's New**: 이 연구는 Attention Bidirectional Long Short-Term Memory (BiLSTM) 네트워크를 기반으로 한 프레임워크를 제안하여, 사용자 정의 날짜에 목표 이미지를 예측할 수 있는 능력을 갖추고 있습니다. 이는 특히 클라우드 커버가 지속되는 기간에도 유효합니다. NDVI, 여러 식생 지수, 모든 Sentinel-2 밴드를 예측하는 데 있어 모델의 우수한 성능을 실험적으로 입증하여 원격 감지 데이터의 연속성과 신뢰성을 개선하는 잠재력을 강조합니다.

- **Technical Details**: 제안된 모델은 BiLSTM 네트워크와 Attention 메커니즘을 통합하여, 시퀀스-투-원 (sequence-to-one) 예측 프레임워크에서 짧은 시퀀스를 처리하는 능력을 향상시킵니다. 모델의 구조는 입력 레이어, 양방향 LSTM 레이어, 배치 정규화, ReLU 활성화 함수, Attention 메커니즘, Global Average Pooling 1D 레이어 및 출력 레이어로 구성됩니다. Attention 메커니즘은 입력 시퀀스를 통하여 중요도를 가중치로 적용하여 예측 정확도를 높입니다.

- **Performance Highlights**: NDVI 예측에서 Attention LSTM 모델은 가장 낮은 RMSE (0.0315)와 MAPE (4.469)를 달성했습니다. 다중 식생 지수를 예측할 때는 Attention BiLSTM 모델이 최상의 RMSE (0.0281)와 MAPE (5.348)를 기록했습니다. Sentinel-2 밴드 예측에서는 Attention BiLSTM 모델이 최고 MAPE (6.690)와 경쟁력 있는 RMSE (0.0174)를 달성했습니다. 이 결과들은 BiLSTM과 Attention 메커니즘이 시공간적 복잡성을 효과적으로 처리하는 능력을 보여줍니다.



### Enhancing Travel Decision-Making: A Contrastive Learning Approach for Personalized Review Rankings in Accommodations (https://arxiv.org/abs/2407.00787)
- **What's New**: 이 논문은 두 가지 주요 기여를 제시합니다. 첫째, 50,000개의 숙박 시설에서 수집된 2백만 건 이상의 리뷰를 포함한 새로운 데이터셋을 소개합니다. 둘째, 리뷰어의 맥락 정보를 기반으로 개인 맞춤형 리뷰 순위를 매기는 혁신적인 방법론을 제안합니다. 이를 통해 여행 도메인을 넘어서 온라인 전자상거래 플랫폼 등 다른 분야에서도 활용될 수 있는 가능성을 열어줍니다.

- **Technical Details**: 우리의 접근 방식은 서로 대조적인 학습(contrastive learning)을 이용하여 리뷰와 리뷰어의 맥락 간의 복잡한 관계를 캡처합니다. 이 방법은 다양한 사용자 특성(가족 여행객, 솔로 여행객, 커플 등) 및 여행 유형(해변, 도시, 자연 등)을 고려하여 맞춤형 리뷰 순위를 매깁니다. 실험 결과, Mean Reciprocal Rank (MRR), precision@1, precision@10과 같은 지표에서 기존의 여러 기준을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 종합적인 실험을 통해, 우리의 방법론이 기존의 기준을 모든 보고된 지표에서 능가함을 확인했습니다. Mean Reciprocal Rank (MRR), precision@1 및 precision@10 측면에서 유의미한 성능 향상을 가져왔습니다. 또한, 비교 분석을 통해 우리 모델이 주어진 맥락의 사용자에 의해 작성된 리뷰와 공통 주제를 강조하여 모델 출력의 해석 가능성과 설명 가능성을 제공합니다.



### Dense Retrieval with Continuous Explicit Feedback for Systematic Review Screening Prioritisation (https://arxiv.org/abs/2407.00635)
Comments:
          Accepted at SIGIR 2024

- **What's New**: 이 논문에서는 규칙적 문헌 검토의 스크리닝 우선순위 설정(task명)에서 최신 신경망 기반 모델의 시간 소모적인 미세 조정(fine-tuning)과 추론(inference)의 문제를 극복하기 위해, 밀집 표현(dense representation)과 연관 피드백(relevance feedback)을 활용하는 새로운 접근 방식을 제안합니다. 이 방법은 리뷰어의 연속적인 피드백을 통해 효율적으로 문서 순위를 업데이트하며, 기존 방법보다 더 효율적이고 효과적임을 입증했습니다.

- **Technical Details**: 본 연구의 방법은 신경 인코더 모델(neural encoder model)을 사용하여 밀집 검색(dense retrieval)을 수행하고, 리뷰어의 연속적인 명시적 피드백을 통해 질의를 업데이트합니다. 질의 업데이트는 Rocchio 알고리즘을 사용하여 구현됩니다. 이번 연구에서는 CLEF TAR 데이터셋을 활용하여 평가하였으며, 주로 BERT 기반의 응답자를 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법보다 스크리닝 과정에서의 효율성과 효과성 면에서 우수한 성능을 보였습니다. 특히, 높은 연관성(recall)과 초기 위치에서의 문서 랭킹에서 우수한 성능을 입증하였습니다. 코드 및 자세한 내용은 논문 내에서 제공된 URL을 통해 접근할 수 있습니다.



### Towards Statistically Significant Taxonomy Aware Co-location Pattern Detection (https://arxiv.org/abs/2407.00317)
Comments:
          Accepted in The 16th Conference on Spatial Information Theory (COSIT) 2024

- **What's New**: 이 논문은 계층적 분류(taxonomy)를 고려한 공-위치 패턴(co-location patterns)의 통계적 유의성을 평가하는 새로운 방법을 제안합니다. 이를 통해 생태학, 공간병리학, 리테일 등의 다양한 분야에서 중요한 새로운 공-위치 패턴을 발견할 수 있습니다.

- **Technical Details**: Boolean 공간 특징 타입과 그 인스턴스, 이웃 관계, 그리고 계층적 분류를 기반으로 공-위치 패턴을 검출합니다. 기본 접근법은 리프 노드나 그 조상을 통해 공-위치 패턴의 유의성을 반복적으로 검사합니다. 또한, Benjamini-Hochberg 방법을 사용하여 거짓 발견율(False Discovery Rate, FDR)을 제어하는 진보된 접근법을 제안합니다.

- **Performance Highlights**: 실험 평가와 사례 연구 결과는 제안된 접근법이 효과적임을 보여줍니다. 이 접근법은 높은 수준의 계층적 관계에서도 겹칩 패턴을 잘 탐지하며, 거짓 발견율을 줄이면서 진정한 공-위치 패턴을 효과적으로 발견합니다.



### When Search Engine Services meet Large Language Models: Visions and Challenges (https://arxiv.org/abs/2407.00128)
Comments:
          Under Review

- **What's New**: 최근 연구는 대형 언어 모델(Large Language Models, LLMs)과 검색 엔진 서비스를 결합하여 검색과 정보 검색을 혁신적으로 개선할 수 있는 새로운 방법을 제시합니다. 이 논문은 LLM과 검색 엔진의 상호 이익을 취할 수 있는 다양한 방법을 탐구합니다. 특히, 검색 엔진 데이터를 LLM 학습에 활용하는 Search4LLM과 LLM을 통해 검색 엔진 기능을 향상시키는 LLM4Search에 중점을 둡니다.

- **Technical Details**: Search4LLM 분야에서는 검색 엔진이 LLM의 사전 학습(pre-training)에 필요한 다양한 고품질 데이터셋을 제공하는 방법, 관련 문서를 통해 보다 정확하게 질문에 답변할 수 있도록 돕는 방법, Learning-To-Rank (LTR) 작업을 통해 LLM의 응답 정밀도를 높이는 방법, 최신 검색 결과를 포함하여 LLM이 생성하는 콘텐츠의 정확성과 최신성을 보장하는 방법 등에 대해 탐구합니다. LLM4Search 분야에서는 LLM을 활용하여 검색 엔진의 인덱싱을 위한 콘텐츠 요약, 쿼리 결과 최적화, 문서 관련성 분석을 통한 검색 결과 순위 향상, 다양한 학습 컨텍스트에서의 데이터 주석 등을 개선하는 방법을 검토합니다.

- **Performance Highlights**: 이 논문의 연구는 LLM과 검색 엔진이 각각의 한계를 극복하고 기능을 향상시킬 수 있는 상호 보완적인 방법들을 제시합니다. 또한, 이러한 통합의 도전 과제인 모델 학습의 편향성 문제와 윤리적 문제, 계산 비용 관리, 끊임없는 웹 콘텐츠 변화에 따른 지속적인 모델 업데이트 필요성 등의 문제를 논의합니다. 이 연구는 검색 엔진 아키텍처를 고도화하여 사용자 경험을 크게 향상시킬 수 있는 방안을 모색합니다.



### Predictive accuracy of recommender algorithms (https://arxiv.org/abs/2407.00097)
- **What's New**: 최신 연구에서는 추천 시스템(Recommender Systems)의 다양한 알고리즘을 비교 실험하는 연구가 이루어졌습니다. 특히, 많은 기존 연구들은 공통된 벤치마크 데이터셋(Data Sets)과 향상 모델 평가 방식(Evaluation Metrics)을 사용하지 않았기 때문에 이 연구에서는 이러한 변수들을 통제한 실험을 통해 알고리즘의 정확성을 비교했습니다.

- **Technical Details**: 이번 연구는 세 가지 전통적인 추천 알고리즘과 두 가지 딥러닝(Deep Learning, DL) 알고리즘을 사용하여 공개된 평점 데이터 소스를 통해 그 정확성을 평가했습니다. 실험 결과, 전통적인 알고리즘은 기존의 벤치마크와 일치하는 결과를 보였으나 DL 알고리즘의 성능은 비교적 저조했습니다. 이는 DL 알고리즘 구현 시 나타나는 문제점들을 가시화하였습니다.

- **Performance Highlights**: 전통적인 추천 알고리즘의 성능은 기대에 부합했으나 두 가지 딥러닝 알고리즘은 모델 과적합(Overfitting) 문제로 인해 기대보다 낮은 성능을 보였습니다. 이에 대한 해결책으로 다수의 정규화(Regularization) 전략이 검토되었으며, 이를 통해 예측 오류를 개선할 수 있는 가능성을 모색했습니다.



### Learning to Rank for Maps at Airbnb (https://arxiv.org/abs/2407.00091)
- **What's New**: Airbnb는 기존의 리스트 기반 검색 결과와 다르게 지도로 표시되는 검색 결과에 대한 랭킹 알고리즘을 새롭게 개편했습니다. 이 논문은 어떻게 사용자 상호작용의 수학적 기초를 수정해 랭킹을 재구성했는지 설명하며, 이는 에어비앤비의 사용자 경험을 크게 향상시켰습니다.

- **Technical Details**: 기존에는 리스트 결과와 지도 결과가 동일한 랭킹 알고리즘을 사용했으나, 지도 검색에서는 리스트에서의 가정이 적용되지 않는다는 것을 발견했습니다. NDCG(Normalized Discounted Cumulative Gain)로 리스트와 지도의 검색 성능을 비교해 2%의 갭을 확인했습니다. 실험에서 지도 검색의 경우 랭킹이 무의미하다는 가설을 검증해, 리스트의 검색 결과처럼 사용자의 주의가 점진적으로 감소하지 않음을 확인했습니다.

- **Performance Highlights**: 실제 실험을 통해 지도 검색에서의 랭킹을 임의로 섞어도 예약 건수에 차이가 없다는 것을 발견했습니다. 이는 지도 검색에서는 랭킹이 중요하지 않다는 저자의 가설을 뒷받침합니다. 전체적으로 새 알고리즘은 사용자 경험을 크게 개선하는 데 도움을 주었습니다.



### Compressing Search with Language Models (https://arxiv.org/abs/2407.00085)
- **What's New**: 이번 연구에서는 Google Search 데이터를 활용하여 실세계 이벤트를 예측하는 새로운 접근 방식을 제안합니다. 핵심 기여는 두 가지로, 첫째는 SLaM Compression을 통해 사전 학습된 언어 모델(LM)을 사용하여 검색 데이터를 저차원으로 압축하며, 둘째는 CoSMo라는 제한된 검색 모델(Constrained Search Model)을 도입하여 검색 데이터만으로 실세계 이벤트를 추정하는 것입니다.

- **Technical Details**: SLaM Compression은 검색어를 사전 학습된 언어 모델을 통해 임베딩 벡터로 변환하여, 메모리 효율적이면서도 예측력이 뛰어난 검색 데이터 요약 값을 생성합니다. 이러한 검색 임베딩을 CoSMo 모델에 입력 값으로 사용하여, 예측 대상 변수의 발생 확률을 0과 1 사이의 점수로 산출합니다. CoSMo는 특정 검색 데이터 배포 및 검색 임베딩의 특성을 반영하도록 설계되었습니다.

- **Performance Highlights**: 미국 자동차 판매량 및 미국 독감 발생률을 예측하는 실험에서, Google Search 데이터를 통해 높은 정확도로 실세계 이벤트를 추정하는 데 성공했습니다. 기존의 분류 임베딩 방법과 비교할 때, 검색 임베딩을 사용하는 경우 자동차 판매 예측력이 30% 향상되었으며, 독감 모델링에서도 기존 자체회귀 모델(autoregressive approaches)와 비교해 더 나은 성능을 보였습니다.



### Adapting Job Recommendations to User Preference Drift with Behavioral-Semantic Fusion Learning (https://arxiv.org/abs/2407.00082)
Comments:
          Accepted by KDD 24 Research Track

- **What's New**: 이 논문은 구직자와 일자리의 연결을 개선하는 새로운 세션 기반 프레임워크인 BISTRO를 소개합니다. BISTRO는 사용자 선호도의 빈번한 변화(Preference Drift)를 신속하고 정확하게 포착하기 위해 의미(semantic)와 행동 기반 정보를 결합한 학습 방법을 제안합니다.

- **Technical Details**: BISTRO는 세 가지 주요 단계로 구성됩니다. 첫째, 사용자 상호작용 시퀀스를 세션으로 분할하고 세션 기반 의미 클러스터링(coarse-grained semantic clustering)을 통해 사람-직업 매칭을 넓게 식별합니다. 둘째, 세부적인 직무 선호도를 추출하기 위해 초그래프 파동 학습(hypergraph wavelet learning)을 사용하여 사용자 선호도 변화와 상호작용의 노이즈를 제거합니다. 마지막으로, 순환 신경망(recurrent neural network)을 활용하여 세션 기반 상호작용을 분석하고 개인화된 선호도를 유추하여 상위 k개의 직업 추천을 생성합니다.

- **Performance Highlights**: 세 가지 실제 구직 데이터셋에서의 광범위한 실험 결과, BISTRO는 뛰어난 성능을 보였습니다. 온라인 실험에서도 성공적으로 구현되어 BISTRO의 실시간 모집 환경에서의 유효성을 입증했습니다. 소스 코드는 공개되어 있습니다.



### Differentially Private Graph Diffusion with Applications in Personalized PageRanks (https://arxiv.org/abs/2407.00077)
- **What's New**: 이 연구에서는 엣지 레벨의 차등 프라이버시(differential privacy) 보장을 제공하는 새로운 그래프 확산(framework)을 제안합니다. 이는 각 확산 반복(iteration)마다 Laplace 노이즈를 주입하고, 낮은 차수(degree)의 노드들로 인한 높은 민감도를 완화하기 위해 차수 기반 임계값 함수(thresholding function)를 도입합니다. 이를 통해 그래프 자료에서 발생할 수 있는 민감한 연결 정보를 안전하게 보호할 수 있습니다.

- **Technical Details**: 알고리즘은 Privacy Amplification by Iteration (PABI)을 기반으로 프라이버시 손실을 분석합니다. 이는 Laplace 노이즈와 결합되어 PABI에 대한 분석을 최초로 시도한 것입니다. 또한, 새로운 ∞ -Wasserstein 거리 추적 방법이 도입되어 프라이버시 유출 분석을 더욱 긴밀하게 만들고, 실제 적용 가능성을 높입니다. Graph diffusion 동안 각 반복마다 노이즈를 주입함으로써, Gaussian 메커니즘보다 우수한 성능을 발휘하며, Personalized PageRank를 계산할 때 실험적으로 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, 이번에 제안된 프레임워크는 실제 네트워크 데이터를 통해 엄격한 프라이버시 조건 아래에서도 우월한 성능을 보였습니다. 특히, 낮은 차수 노드의 민감도를 완화하기 위한 차수 기반 임계값 함수 도입으로 향상된 유용성-프라이버시 간의 균형을 이뤘습니다.



### Pistis-RAG: A Scalable Cascading Framework Towards Trustworthy Retrieval-Augmented Generation (https://arxiv.org/abs/2407.00072)
- **What's New**: 최근 발표된 논문에서 Pistis-RAG라는 대규모 검색 강화 생성 시스템(Retrieval-Augmented Generation, RAG)의 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 LLM(대형 언어 모델)이 외부 지식 랭킹 방법과의 강한 정렬이 부족하다는 기존 문제를 해결하고, 모델 중심 패러다임 대신 내용 중심 접근 방식을 채택합니다. 특히, 랭킹 단계를 재설계하여 사용자의 피드백과 LLM의 선호도를 반영한 생성 품질을 향상시킵니다.

- **Technical Details**: Pistis-RAG는 매칭, 프리랭킹, 랭킹, 재랭킹의 다단계 검색 파이프라인을 활용합니다. 매칭 단계에서는 검색 공간을 정제하고, 프리랭킹 단계에서는 의미적으로 관련 있는 문서를 우선시하며, 랭킹 단계에서는 LLM의 선호도와 사용자 피드백을 고려하여 내용을 정렬합니다. 또한, 복잡한 Chain-of-Thought(연쇄적 사고) 방법론을 지원하는 Reasoning 및 Aggregating 단계가 포함되어 있습니다. 이 접근 방식은 외부 지식과 LLM 간의 원활한 통합을 핵심 원칙으로 삼아, 각 특정 작업에 맞춘 내용 변환 과정을 최적화합니다.

- **Performance Highlights**: Pistis-RAG는 MMLU 벤치마크 실험에서 9.3%의 성능 향상을 보여줍니다. 또한, 실제 대규모 데이터에서도 프레임워크의 확장 가능성을 검증하였습니다. 이로써 기존 방법보다 더욱 관련성 있고 개인화된 결과를 제공하며, LLM의 프롬프트 순서 민감성을 고려하는 새로운 랭킹 메커니즘을 도입하였습니다.



### Perceptron Collaborative Filtering (https://arxiv.org/abs/2407.00067)
Comments:
          11 pages, 7 figures

- **What's New**: 이 연구에서는 다변수 로지스틱 회귀 분류기(multi-variate logistic regression classifiers)가 아닌 신경망(neural networks)을 사용하여 협업 필터링(collaborative filtering)을 구현하는 방법을 제안하고 있습니다. 협업 필터링은 여러 사용자들의 취향 정보를 수집하여 개별 사용자의 관심사를 자동으로 예측하는 기법입니다.

- **Technical Details**: 퍼셉트론(perceptron) 또는 신경망은 역전파(backpropagation)와 경사하강법(gradient descent)을 사용하여 복잡한 데이터셋을 맞추도록 설계된 머신러닝 모델입니다. 이 모델은 고급 최적화 기법을 결합하면, 기존의 로지스틱 회귀 분류기에 대한 우수한 대체품이 될 수 있습니다. 최적화 기법에는 특징 스케일링(feature scaling), 평균 정규화(mean normalization), 정규화(regularization), 하이퍼파라미터 튜닝(hyperparameter tuning) 그리고 확률적/미니배치 경사하강법(stochastic/mini-batch gradient descent)이 포함됩니다. 연구에서는 퍼셉트론을 추천 시스템(recommender system)에서 사용하여 여러 사용자의 데이터를 맞추고 특정 사용자의 선호/관심사를 예측합니다.

- **Performance Highlights**: 최적화된 신경망 모델은 특징 스케일링과 정규화 등 다양한 기법을 통해 데이터 적합도를 높여 기존의 로지스틱 회귀 분류기 대비 높은 성능을 보여줄 것으로 기대됩니다.



### An Interpretable Alternative to Neural Representation Learning for Rating Prediction -- Transparent Latent Class Modeling of User Reviews (https://arxiv.org/abs/2407.00063)
- **What's New**: 최근 몇 년간 심층 신경망(DL, Deep Learning) 접근 방식이 추천 시스템 연구 분야에서 널리 도입되었으나, 간단한 알고리즘에 비해 성능 향상이 제한적이며 해석 가능성도 떨어진다는 문제가 제기되었습니다. 이에 따라 본 논문에서는 리뷰 정보를 기반으로 사용자와 제품 잠재 클래스를 위상적으로 조직하는 투명한 확률 모델을 제안합니다. 이를 통해 신경망 접근법에 비해 해석 가능하면서도 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: 본 논문은 리뷰 기반 평가 예측 과제를 위해 사용 가능한 숫자 평점 데이터의 한계를 극복하고자 텍스트 정보를 활용합니다. 제안된 모델은 사용자와 제품의 잠재 클래스를 2차원 격자(Grid) 위에 토포그래픽하게 조직하여 학습합니다. 이를 통해 리뷰 데이터의 해석 가능하고 인간 친화적인 잠재 구조를 제공합니다. 또한, 이 확률적 모델은 잠재 코드의 해석 가능성을 모델 공식화로 명시적으로 부여합니다.

- **Performance Highlights**: 제안된 모델은 리뷰 데이터를 통해 도출된 잠재 클래스 표현이 눈에 띄는 예측 성능을 보여줍니다. 이는 복잡하나 해석이 어려운 기존의 신경망 기반 접근법에 비해 단순하면서도 설명이 가능한 방식으로 구현되었습니다. 특히, 리뷰와 평점 간의 강한 상관관계를 이용하여 텍스트 기반 예측의 정확성을 높였습니다.



### A First Principles Approach to Trust-Based Recommendation Systems (https://arxiv.org/abs/2407.00062)
- **What's New**: 이 논문은 소셜 네트워크에서 추천 시스템을 탐구하며, 아이템 평점, 항목 간 유사성, 신뢰 그래프 정보를 활용합니다. 협업 필터링(collaborative filtering) 접근법에서 아이템 평점 정보가 타 정보 유형보다 더 영향력이 크다는 것을 입증했습니다. 신뢰 그래프 기반 접근법은 조작이 어려운 신뢰 구조로 인해 네트워크 적대적 공격에 더 견고하다는 것이 밝혀졌습니다. 또한, 항목 간 정보는 단독으로는 최적이 아니지만, 다른 정보 형태와 결합되면 예측의 일관성과 하위 성능을 향상시킵니다. 가중 평균(Weighted Average) 프레임워크도 도입되어 사용자의 유사성 메트릭 유동적으로 설계할 수 있게 합니다.

- **Technical Details**: 이 연구는 신뢰 기반 사회 네트워크 데이터를 활용하여 신뢰 그래프 데이터가 적대적 공격에 대해 견고하며, 콜드 스타트(cold-start) 사용자에게도 높은 품질의 추천을 제공할 수 있음을 발견합니다. 최초 원칙 접근법을 사용하여 다양한 데이터 형태를 결합하는 새로운 방법을 개발하며, 사용자 유사성에 기반한 추천 시스템을 설계할 수 있는 프레임워크도 제시합니다. 여러 가지 추천 시스템을 구축하고 평가하여 각 데이터 유형이 다양한 상황에서 어떻게 동작하는지 분석합니다.

- **Performance Highlights**: 아이템 평점 정보가 협업 필터링에서 가장 영향력 있는 요소임을 입증했으며, 신뢰 그래프 기반 접근법은 네트워크 적대적 공격에 견고하고, 항목 간 정보는 하위 성능을 개선함. 가중 평균 프레임워크는 사용자의 유사성 메트릭을 활용하여 유연한 추천 시스템 설계를 가능하게 함.



### MMBee: Live Streaming Gift-Sending Recommendations via Multi-Modal Fusion and Behaviour Expansion (https://arxiv.org/abs/2407.00056)
Comments:
          Accepted at KDD 2024

- **What's New**: 실시간 스트리밍 서비스의 선물 제공 예측을 위한 혁신적인 방법인 MMBee가 발표되었습니다. MMBee는 실시간 Multi-Modal Fusion과 Behavior Expansion을 기반으로 하여 사용자들이 스트리머에게 가상 선물을 보내는 행동을 더 정확하게 예측하고자 합니다. 이 방법은 Kuaishou와 같은 플랫폼에서 실시간 스트리밍 생태계의 최적화 및 개선을 목표로 하고 있습니다.

- **Technical Details**: MMBee는 Multi-Modal Fusion Module with Learnable Query (MFQ)와 Graph-guided Interest Expansion (GIE) 두 가지 주요 구성 요소로 나뉩니다. MFQ 모듈은 스트리밍 세그먼트의 동적 콘텐츠를 인지하고 이미지, 텍스트 코멘트, 음성을 포함한 복잡한 멀티모달 상호 작용을 처리합니다. GIE 접근법은 대규모 선물 교류 그래프와 멀티모달 속성으로 사용자와 스트리머의 표현을 학습하여 선물 행동의 희소성 문제를 완화합니다.

- **Performance Highlights**: MMBee는 Kuaishou의 10억 단위의 산업 데이터셋과 공개 데이터셋 모두에서 성능 향상을 보여주었으며, 온라인 A/B 테스트를 통해 그 효율성이 추가로 검증되었습니다. 현재 MMBee는 Kuaishou의 실시간 스트리밍 추천 시스템에 배포되어 수백만 명의 사용자에게 서비스를 제공하고 있습니다.



### JungleGPT: Designing and Optimizing Compound AI Systems for E-Commerc (https://arxiv.org/abs/2407.00038)
- **What's New**: 이 논문은 실세계 전자상거래(e-commerce) 애플리케이션을 위해 맞춤 제작된 최초의 복합 AI 시스템(JungleGPT)을 소개합니다. 기존 단일 대형 언어 모델(LLMs)의 한계를 극복하고, 실질적인 사용 사례에 최적화된 성능을 제공합니다. JungleGPT는 강력하면서도 비용 효율적인 LLM 솔루션을 통해 전자상거래의 복잡성과 규모를 효과적으로 다룹니다.

- **Technical Details**: JungleGPT 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다: JungleGPT Copilot, JungleGPT Caching Nodes, JungleGPT LLM Nodes. 이 구성 요소들은 비동기 업데이트를 통해 사용자와의 상호작용을 빠르고 원활하게 만듭니다. JungleGPT Copilot은 사용자 인터랙션 루프의 중심에 있어, 웹기술(ex: webGPU)을 활용하여 저지연으로 응답합니다. JungleGPT Caching Nodes는 엣지 네트워크(Edge Network)에 배치되어 글로벌 사용자에게 저지연으로 데이터를 제공합니다. 마지막으로, JungleGPT LLM Nodes는 백엔드에서 Caching Nodes를 주기적으로 업데이트하며, 사용자 인터랙션의 주요 경로에서 분리되어 있습니다.

- **Performance Highlights**: JungleGPT의 성능 최적화 덕분에 LLM 추론 비용이 단일 대형 언어 모델 대비 1% 이하로 감소되었습니다. 이 시스템은 전자상거래 SaaS 제품의 월간 비용 한도를 고려하여 비용 효율성을 극대화하며, 소규모 LLMs와 재랭커(rerankers)들을 결합하여 다중 언어 지원과 분석 품질을 향상시킵니다. 이를 통해 전자상거래 사용자들의 요구에 맞춘 맞춤형 서비스를 제공합니다.



### A Global-Local Attention Mechanism for Relation Classification (https://arxiv.org/abs/2407.01424)
Comments:
          This paper has been accepted by the 2024 20th International Conference on Natural Computation, Fuzzy Systems and Knowledge Discovery (ICNC-FSKD)

- **What's New**: 이 논문은 관계 분류(relation classification) 작업에서 전통적인 글로벌 주의 메커니즘(global attention mechanism)이 놓치기 쉬운 로컬 컨텍스트의 중요성을 강조합니다. 이를 해결하기 위해 글로벌-로컬 주의 메커니즘(global-local attention mechanism)을 도입하여 글로벌 주의에 로컬 기준을 추가합니다. 또한, 잠재적인 키워드를 식별하기 위해 하드 및 소프트 로컬화 메커니즘을 제안합니다. 이 접근법은 SemEval-2010 Task 8 데이터셋 실험 결과에서 기존 주의 방식보다 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 방법은 네 부분으로 구성됩니다: 1) 입력 표현(Input Representation): 단어는 단어 임베딩 및 위치 임베딩으로 변환됩니다. 2) 양방향 GRU 층(Bi-directional GRU Layer): 입력에서 고차원의 특징을 추출합니다. 3) 글로벌-로컬 주의 레이어(Global-Local attention Layer): 전통적인 글로벌 주의 메커니즘과 로컬 주의 메커니즘을 결합합니다. 하드 로컬화는 최단 의존 경로에 위치한 모든 단어를 잠재적 키워드로 간주하고, 소프트 로컬화는 이러한 경로를 감독 신호로 사용하여 더 견고한 키워드를 식별합니다. 4) 출력 층(Output Layer): 주의 가중치로 합산된 은닉 상태는 최종 분류 결과를 소프트맥스(softmax)로 제공합니다.

- **Performance Highlights**: SemEval-2010 Task 8 데이터셋을 사용한 실험 결과, 제안된 글로벌-로컬 주의 메커니즘이 기존 주의 방법보다 우수한 성능을 나타냅니다. 추가 분석은 글로벌-로컬 주의가 정확한 키워드에 효과적으로 집중하는 것을 보여줍니다.



### BERGEN: A Benchmarking Library for Retrieval-Augmented Generation (https://arxiv.org/abs/2407.01102)
Comments:
          29 pages

- **What's New**: 최근의 생성적 대형 언어 모델(Large Language Models, LLM)의 인기를 반영하여, Retrieval-Augmented Generation(RAG) 접근 방식이 제안되었습니다. 이러한 접근 방식들은 다양한 구성요소와 함께 사용되며, 일관되지 않은 벤치마킹이 주요 과제가 됩니다. 이를 해결하기 위해, BERGEN이라는 종단간(end-to-end) 연구 표준화 라이브러리를 소개합니다.

- **Technical Details**: BERGEN은 RAG 실험을 표준화하고 재현성을 보장하는 오픈소스 Python 라이브러리입니다. 이 라이브러리는 500개 이상의 실험을 통해 최첨단의 retrievers, rerankers, 그리고 LLMs를 벤치마크했고, 기존의 다양한 RAG 지표와 데이터셋을 분석합니다. Hugging Face(HF) 허브를 기반으로 데이터셋과 모델을 처리하며, 다양한 모델 아키텍처와 훈련 및 평가 구성을 지원합니다.

- **Performance Highlights**: BERGEN은 기존의 단편적이고 비효율적인 실험 설정을 일원화하여, RAG 접근 방식을 보다 체계적이고 비교할 수 있게 합니다. 또한, 죽음의 중요한 요소로 최첨단의 retrievers와 rerankers의 사용을 강조하고, 일반적인 표면 일치 메트릭(exact match, F1, Rouge-L 등) 외에도 LLM 기반 평가의 중요성을 부각합니다.



### Answering real-world clinical questions using large language model based systems (https://arxiv.org/abs/2407.00541)
Comments:
          28 pages (2 figures, 3 tables) inclusive of 8 pages of supplemental materials (4 supplemental figures and 4 supplemental tables)

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 의료 결정에 필요한 근거를 요약하거나, 실제 데이터(Real-World Data, RWD)를 기반으로 새로운 연구를 생성하는 잠재력을 평가했습니다. 총 5개의 LLM 기반 시스템이 50개의 임상 질문에 답변하는 능력을 검토했으며, 9명의 독립된 의사들이 이 답변들을 관련성, 신뢰성, 실행 가능성 측면에서 평가했습니다.

- **Technical Details**: 일반 목적 LLMs (ChatGPT-4, Claude 3 Opus, Gemini Pro 1.5)은 관련성과 근거 기반으로 평가된 답변을 드물게 생성했으며(2% - 10%), 반면 검색 강화 생성(Retrieval Augmented Generation, RAG) 기반 및 agentic LLM 시스템은 질문의 24% (OpenEvidence)에서 58% (ChatRWD)에 대해 관련성과 근거 기반의 답변을 제공했습니다. 특히 agentic ChatRWD는 다른 LLMs와 비교했을 때 새로운 질문에 답변할 수 있는 유일한 모델이었습니다 (65% vs. 0-9%).

- **Performance Highlights**: 일반 목적 LLMs는 그 자체로는 사용하기에 부적합한 반면, RAG 기반 시스템과 새로운 증거 생성을 위한 agentic 시스템이 상호작용하여 개선된 환자 치료를 위한 관련 근거를 제공할 잠재력이 있음을 시사합니다.



### AI-Driven Skin Cancer Diagnosis: Grad-CAM and Expert Annotations for Enhanced Interpretability (https://arxiv.org/abs/2407.00104)
Comments:
          8 pages, 4 figures, 4 tables, under review

- **What's New**:  본 논문에서는 Basal Cell Carcinoma(BCC) 진단을 위해 유용한 해석 가능한 AI 도구를 개발하여, 원격 피부학(teledermatology)을 통해 진단의 신속성과 자원의 최적화를 도모하였습니다. 이 도구는 BCC 진단을 위한 주요 dermoscopic 패턴을 이미지에서 찾아내어 BCC/Non BCC 분류를 정당화하고, Grad-CAM을 활용한 시각적 설명을 통해 임상적으로 영감을 받은 해석을 제공합니다.

- **Technical Details**:  이 AI 도구는 병합된 ImageNet 전이 학습 및 세 단계 최적화 전략을 기반으로 MobileNet-V2 모델을 사용하여 개발되었습니다. 첫 번째 단계에서는 분류기 가중치를 학습하고, 두 번째 단계에서는 세 개의 마지막 블록과 분류기에 대해 미세 조정(fine-tuning)을 진행하였으며, 세 번째 단계에서는 매우 낮은 학습률(LR)을 적용하여 이진 모델에서 BCC 패턴 감지 임무로 전이 학습을 수행했습니다. 또한, 전문가 생성 세분화 데이터를 사용하여 각 이미지에 대한 개별 BCC 패턴의 세분화를 수행했습니다.

- **Performance Highlights**:  이 AI 모델은 BCC/Non-BCC 분류에서 90%의 정확도를 달성했으며, 임상적으로 영감을 받은 XAI 결과는 임상가에게 유용한 BCC 패턴 감지에서 99%의 정확도를 달성했습니다. 임상적으로 영감을 받은 시각적 XAI 결과로는, 수동으로 분할된 임상적 특징 내에서 Grad-CAM의 정규화된 값의 평균이 0.57인 반면, 이 영역 밖에서는 0.16으로 나타났습니다. 이는 모델이 BCC 패턴의 영역을 정확히 식별하는 데 어려움을 겪고 있음을 나타냅니다. 이 결과들은 AI 도구가 유용한 설명을 제공할 수 있음을 증명합니다.



### Constraint based Modeling according to Reference Design (https://arxiv.org/abs/2407.00064)
- **What's New**: 이 논문에서는 참조 모델(Reference Models, RM)을 사용하여 솔루션 모델 설계를 지원하는 일반적인 접근 방식을 제안하고 있습니다. 이를 위해 의미 기술(Semantic Technologies)을 사용하여 RM을 형식적으로 기술하고, 이러한 기술을 적용하는 방법을 다루고 있습니다. 이 시스템은 구성 요소를 기반으로 한 다양한 기법을 사용해 솔루션 모델을 구축하며, 개발된 디자인이 참조 모델과 일치하는지 검증하는 기능을 제공합니다.

- **Technical Details**: 제안된 접근 방식은 RM의 형식적(description) 기술을 위해 의미 기술(Semantic Technologies)을 사용하며, 이를 적용하여 사용자가 새로운 솔루션을 모델링할 때 지원할 수 있습니다. 이 모델링 어시스턴트는 UML 다이어그램 등의 모델링 언어를 사용하여 솔루션 모델을 구성하고, RM와의 적합성을 검증합니다. 또한, 여러 참조 모델을 사용하여 시스템의 시스템 디자인(System of System Designs)에 적용할 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 요구 사항의 형식화(formalization)에 기여하며, 최종적으로는 성숙도 모델(Maturity Model) 맥락에서 품질 보증(Quality Assurance)으로 이어집니다. 이러한 기법의 적용은 산업 분야에서 평가되었으며, 다양한 모델링 환경에 통합될 수 있습니다. 이를 통해 효율성을 높이고 시간과 비용을 절약할 수 있습니다.



### A Document-based Knowledge Discovery with Microservices Architectur (https://arxiv.org/abs/2407.00053)
- **What's New**: 이 논문은 조직 내 디지털 기반 지식을 향상시키기 위한 새로운 접근법을 제안합니다. 초점은 미세 서비스 아키텍처(microservices architecture)를 도입하여 키워드 추출, 문서의 유사성 계산, 자연어 데이터베이스 쿼리, 그리고 프로그래밍 언어 독립적인 정보 제공을 가능케 하는 데 있습니다. 이는 독일 특허청에서 확장된 버전으로 사용되고 있는 현대적인 방법론이기도 합니다.

- **Technical Details**: 제안된 개념적 디자인은 반자동 학습, 편집, 온톨로지 시각화를 위한 프로세스 및 애플리케이션 통합을 위한 참조 설계 지침을 제공합니다. 또한 미세 서비스 아키텍처를 사용하여 확장성(Scalability)과 복원력(Resilience) 같은 비기능적 요구사항을 해결합니다.

- **Performance Highlights**: 논문에서 제시된 요구사항의 평가를 위해 구현된 시연기를 통해 개념을 평가했습니다. 이는 문서 기반 지식 발견(KD) 시스템의 아키텍처적 요구사항을 충족시키는 데 중요한 역할을 합니다. 이 접근법은 특히 특허청의 분류, 검색, 검토 업무의 효율성을 크게 향상시킬 수 있습니다.




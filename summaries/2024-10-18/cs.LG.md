New uploads on arXiv(cs.CL)

### Can MLLMs Understand the Deep Implication Behind Chinese Images? (https://arxiv.org/abs/2410.13854)
Comments:
          32 pages,18 figures. Project Page: this https URL Code: this https URL Dataset: this https URL

- **What's New**: 이번 연구에서는 중국 이미지의 고차원 인식 및 이해 능력을 평가하기 위한 새로운 벤치마크인 **CII-Bench**를 소개합니다. 이 벤치마크는 중국 전통 문화 이미지를 포함하여 MLLMs의 성능을 진단할 수 있는 도전적인 과제를 제공합니다.

- **Technical Details**: CII-Bench는 698개의 이미지와 800개의 다양한 선택 질문을 포함하고 있으며 여섯 가지 도메인: 생활, 예술, 사회, 정치, 환경, 중국 전통 문화로 구성되어 있습니다. 이 벤치마크는 그림, 만화, 포스터 등 다양한 이미지 유형을 활용하여 MLLMs의 이해 능력을 평가합니다.

- **Performance Highlights**: MLLMs의 정확도를 조사한 결과, 최고 정확도는 64.4%로 인간의 평균 정확도 78.2%와 비교될 때 상당한 성능 차이를 보였습니다. 특히, MLLMs는 중국 전통 문화 이미지에서 낮은 성능을 보였으며, 감정 힌트를 포함할 때 모델의 정확도가 향상되는 경향이 있었습니다.



### Retrospective Learning from Interactions (https://arxiv.org/abs/2410.13852)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)과 사용자 간의 다중 턴(multi-turn) 상호작용에서 발생하는 암묵적인 피드백 신호를 활용하여 모델 향상을 꾀하는 새로운 방법인 ReSpect를 소개합니다.

- **Technical Details**: ReSpect는 과거 상호작용에서 발생했던 암묵적 신호를 통해 학습하는 방법입니다. 이 방법은 사용자와의 상호작용 후, 모델이 자신의 과거 행동을 회고하여 피드백을 해석하고 재훈련하는 과정을 포함합니다. 이를 통해 사용자는 모델의 수행 여부를 신호로 전달하며, 이 신호는 자연어의 한정된 하위 공간에 위치하여 LLM이 이를 쉽게 감지할 수 있게 합니다.

- **Performance Highlights**: ReSpect를 적용한 결과, IDEFICS2-8B 모델의 작업 완료율이 31%에서 82%로 향상되었습니다. 이 과정에서는 외부 주석 없이도, 과거 상호작용을 통해 직접적으로 스스로 피드백을 해석하고 개선하는 능력을 보여주었습니다.



### SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction (https://arxiv.org/abs/2410.13846)
- **What's New**: SimLayerKV라는 새로운 방법을 소개하며, 긴 맥락을 처리하는 대형 언어 모델에서의 KV cache의 비효율성을 줄입니다. 이 방법은 "lazy" layer를 식별하고 이들에 대한 KV cache의 중복을 줄여 효율성을 증대시킵니다.

- **Technical Details**: SimLayerKV는 특정 존재 layers의 KV cache를 선택적으로 제거하여 inter-layer KV cache 중복을 감소시키는 방법입니다. 이는 lazy layers의 주의 할당 패턴을 분석하여 가능하며, non-lazy layers의 KV cache는 유지합니다. 코드 구현은 단 7줄로 가능하며, 훈련 과정이 필요 없습니다.

- **Performance Highlights**: SimLayerKV는 LongBench 벤치마크에서 3개의 대표적인 LLM (LLaMA2-7B, LLaMA3-8B, Mistral-7B)에서 KV cache 비압축 비율 5배를 달성하며, 4-bit quantization을 함께 사용할 때 오직 1.2%의 성능 저하만 보여줍니다.



### De-mark: Watermark Removal in Large Language Models (https://arxiv.org/abs/2410.13808)
- **What's New**: De-mark는 n-gram 기반 워터마크 제거를 위한 새로운 프레임워크로, 랜덤 선택 탐침(querying strategy)이라는 독창적인 방법을 통해 워터마크의 강도를 평가하고, 그린/레드 리스트를 식별하는 데 도움을 줍니다.

- **Technical Details**: n-gram 워터마킹 기법은 고정된 비밀 키와 접두사 n-그램을 사용하여 생성된 콘텐츠에 통계적인 신호를 삽입합니다. De-mark는 이러한 워터마크를 효과적으로 제거하고, 원래 언어 모델 분포를 유지하는 이론적인 보장을 제공합니다.

- **Performance Highlights**: Llama3와 ChatGPT와 같은 인기 있는 언어 모델에서 De-mark의 효율성과 효과성을 검증한 결과, 이 프레임워크가 산업 규모의 LLM에서도 뛰어난 워터마크 제거 기능을 보임을 확인하였습니다.



### A Watermark for Order-Agnostic Language Models (https://arxiv.org/abs/2410.13805)
- **What's New**: 본 연구에서 우리는 order-agnostic LMs(순서 비민감 언어 모델)를 위한 패턴 기반의 워터마킹 프레임워크인 Pattern-mark를 소개합니다.

- **Technical Details**: Pattern-mark는 Markov-chain 기반의 워터마크 생성기를 이용하여 높은 빈도의 키 패턴을 가진 워터마크 키 시퀀스를 생성합니다. 또한, 통계적 패턴 기반의 탐지 알고리즘을 통해 탐지 시 키 시퀀스를 복원하고 고빈도 패턴의 개수를 기반으로 통계적 테스트를 수행합니다.

- **Performance Highlights**: ProteinMPNN 및 CMLM과 같은 order-agnostic LMs에 대한 포괄적인 평가 결과, Pattern-mark는 탐지 효율성, 생성 품질, 내구성에서 기존 방법들보다 상향된 성능을 보였습니다.



### BenTo: Benchmark Task Reduction with In-Context Transferability (https://arxiv.org/abs/2410.13804)
- **What's New**: 대규모 언어 모델(LLM)의 평가를 효율적으로 줄이는 방법을 제안하며, 평가 품질을 유지하는 동시에 필요한 작업 수를 5%로 줄일 수 있다는 점이 주목할 만하다.

- **Technical Details**: 기존의 작업 전송성(transferability) 평가 방법 대신, 본 논문에서는 in-context learning (ICL)을 기반으로 하는 새로운 접근 방식인 in-context transferability (ICT)를 제안한다. 이 방법은 훈련 없이 두 작업 간의 전송성을 추정하며, 이는 벤치마크 작업 감소를 위해 사용된다. 또한, 벤치마크 작업 선택 문제를 facility location (FL) 문제로 변환하여 효과적인 작업 선택을 가능하게 한다.

- **Performance Highlights**: BenTo라는 접근 방식을 통해 LLM 벤치마크 작업을 5%로 줄였음에도 불구하고, 원래 벤치마크와의 평가 차이는 4% 이하로 유지되며, 효율성과 정확성 모두에서 우수한 결과를 보였다.



### Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions (https://arxiv.org/abs/2410.13788)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)이 애매한 사용자 요청에 충분히 대응하지 못하는 문제를 다루고 있습니다. 특히 LLMs가 명확한 질문을 하지 않고 단일 해석에만 의존하는 경향이 있다는 점을 지적하며, 이것이 기존의 데이터 레이블링 방식에서 비롯된다고 주장합니다. 이를 해결하기 위한 새로운 선호 레이블링 방식인 'double-turn preferences'를 제안합니다.

- **Technical Details**: 제안한 방법은 두 차례의 상호작용을 통해 선호도를 레이블링하는 방식으로, 주어진 요청에 대해 LLM이 명확한 질문을 하고, 그에 대한 응답을 바탕으로 다음 단계를 수행합니다. 이 과정에서 효율성과 효과성을 모두 고려하여 모델의 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안한 double-turn preference 방법을 사용하는 LLM이 표준 방법에 비해 F1 점수에서 5% 향상된 성능을 보였습니다. 이를 통해 모델이 사용자의 의도를 더욱 정확하게 이해하고, 필요한 경우 명확한 질문을 할 수 있도록 훈련됩니다.



### Looking Inward: Language Models Can Learn About Themselves by Introspection (https://arxiv.org/abs/2410.13787)
Comments:
          15 pages, 9 figures

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)가 스스로를 반성(introspection)하는 능력이 있음을 보여줍니다. 이는 모델이 자신의 내부 상태에 대해 정보를 얻고, 이를 기반으로 스스로의 행동을 예측할 수 있음을 의미합니다. 특히, 반성 능력은 모델 해석 가능성(interpretable) 향상에 기여할 수 있습니다.

- **Technical Details**: LLMs는 두 개의 모델, M1과 M2를 사용하여 스스로의 행동 특성을 예측하는 방식으로 반성을 연구했습니다. M1은 자신의 행동을 예측하기 위해 미세 조정(finetuning)되었고, M2는 M1의 실제 행동을 바탕으로 훈련되었습니다. 실험 결과 M1이 M2보다 더 정확하게 스스로를 예측하는 것으로 나타났습니다. 이러한 결과는 LLMs가 훈련 데이터에만 의존하지 않고, 자신에 대한 특별한 접근 권한(privileged access)을 가지고 있다는 것을 시사합니다.

- **Performance Highlights**: 실험에서는 M1이 M2에 비해 정확도가 평균 17% 향상되었습니다. 또한, M1은 의도적으로 자신의 행동을 변경한 후에도 여전히 정확한 예측을 수행할 수 있었으며, 이는 LLMs의 반성 능력이 특정 작업에서 더 나은 보정(calibration)을 보여준다는 것을 나타냅니다. 그러나 복잡한 작업에서는 여전히 반성 능력이 제한적임을 발견했습니다.



### PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignmen (https://arxiv.org/abs/2410.13785)
Comments:
          28 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 정렬 개선을 위한 PopAlign 프레임워크를 제안합니다. 기존의 대조 패턴에 대한 제한된 접근 방식을 넘어, 다양한 대조 패턴을 통합하여 모델의 인간 선호에 대한 반응을 개선합니다.

- **Technical Details**: PopAlign은 크게 세 가지 수준인 (1) 프롬프트(prompt), (2) 모델(model), (3) 파이프라인(pipeline)에서 대조적 패턴을 통합합니다. 여섯 가지 대조 전략을 통해 기존의 피드백 라벨링 절차 없이도 효과적인 대조 데이터를 생성할 수 있도록 합니다. 주요 대조 전략에는 Prefix Contrast, Demon Contrast, Elicitive Contrast, NParam Contrast, Leaderboard Contrast, Refine Contrast가 포함됩니다.

- **Performance Highlights**: PopAlign은 기존 방법들에 비해 모델 정렬 성능이 획기적으로 향상됨을 실험적으로 입증하였으며, 특히 Elicitive Contrast 전략이 그 성능 향상에 크게 기여하는 것을 보였습니다. 다양한 작업에서 차별화된 성능을 나타내며, 오라클 선호 모델에 대한 정확성을 기반으로 한 대조 정확도 및 선호 모델링도 분석하였습니다.



### Quantity vs. Quality of Monolingual Source Data in Automatic Text Translation: Can It Be Too Little If It Is Too Good? (https://arxiv.org/abs/2410.13783)
- **What's New**: 이 연구는 저자들이 자동 번역을 위한 훈련에 사용되는 단일 언어 데이터(monolingual data)가 너무 적거나 선별적으로 불리한 선택일 수 있다는 점을 조사합니다.

- **Technical Details**: 단일 언어 데이터를 사용한 자가 학습(self-learning)에서, 적절한 양과 질의 데이터가 중요한 عامل로 작용하며, 언어 자원이 부족한 영어-독일어 NMT(신경 기계 번역) 시스템에서 실험이 진행되었습니다.

- **Performance Highlights**: 실험 결과, 모든 사용 가능한 데이터를 활용하는 것보다, 테스트 데이터의 도메인과의 유사성에 기반하여 가장 유용한 추가 데이터를 선택하는 것이 모델 성능에 긍정적인 영향을 미친다고 합니다.



### The Mystery of the Pathological Path-star Task for Language Models (https://arxiv.org/abs/2410.13779)
Comments:
          EMNLP 2024 Main

- **What's New**: 최근에 도입된 path-star task는 언어모델(Language Models)의 한계점을 보여주기 위해 설계된 최소한의 작업입니다. 이 태스크는 여러 팔이 단일 시작 노드에서 방사하는 path-star 그래프를 포함하며, 각 노드는 유일합니다.

- **Technical Details**: path-star 그래프는 하나의 중앙 시작 노드와 여러 개의 방사형 팔을 포함하고 있으며, 각 팔은 고유한 타겟 노드에서 끝납니다. 주어진 시작 노드와 타겟 노드에 대해 해당 타겟 노드를 포함하는 팔을 생성하는 것이 이 태스크의 목표입니다. 이 작업이 언어모델에겐 어렵다는 가설은 teacher-forcing의 결핍과 다음 토큰 예측 패러다임에서 기인한다고 제시됩니다.

- **Performance Highlights**: 여러 모델 유형에서 결과를 개선하기 위한 구조화된 샘플을 사용하는 정규화 방법이 도입되어, encoder-only 모델이 지속적으로 태스크를 해결할 수 있는 설정을 발견했습니다. 이 연구 결과는 path-star task가 이론적으로 풀 수 있다는 RASP 증명을 제공합니다.



### Aggregation Artifacts in Subjective Tasks Collapse Large Language Models' Posteriors (https://arxiv.org/abs/2410.13776)
Comments:
          12 pages, 7 figures, 2 tables

- **What's New**: 본 논문은 In-context Learning (ICL) 이 LLMs(대형 언어 모델)의 자연어 작업 수행에 있어 주요 방법으로 자리잡고 있으며, 전이 학습에서 얻은 지식이 이 과정에서 중요하다는 점을 강조합니다. 그러나 ICL이 단순히 태스크 프라이어(task priors)를 재탐색하는 데 지나치게 의존하며, 이는 복잡한 주관적 분야에서는 더욱 두드러진다고 지적합니다.

- **Technical Details**: 저자들은 LLM이 제공하는 데이터셋의 집합적(aggregation) 사용으로 인해 발생하는 주석 아티팩트가 모델의 성능에 미치는 영향을 분석하였습니다. 이 과정에서 LLM이 개별 주석자(annotator)의 관점에 더 잘 맞춘다는 사실을 발견하였고, 소수의 주석자(minority annotators)가 LLM의 프라이어와 더 잘 정렬됨을 확인하였습니다.

- **Performance Highlights**: 이 논문에서는 주석의 집합적 사용이 주관적 작업 모델링에 방해가 된다는 강력한 상관관계를 보여주며, 소수 주석자들이 LLM과 더 긍정적인 상호작용을 하는 경향이 있음을 강조합니다. 그러나 데이터 집합의 집합적 사용만으로는 ICL과 최신 기술 간의 성능 차이를 설명할 수 없음을 발견하였습니다.



### Knowledge-Aware Query Expansion with Large Language Models for Textual and Relational Retrieva (https://arxiv.org/abs/2410.13765)
- **What's New**: 이번 논문에서는 지식 그래프(knowledge graph, KG)의 구조화된 문서 관계를 활용하여 LLM(대형 언어 모델)과 결합된 지식 인식 쿼리 확장 프레임워크를 제안합니다.

- **Technical Details**: 이 연구는 사용자 쿼리의 텍스트적 및 관계적 요구 사항을 모두 처리하기 위해 KG로부터의 관계를 활용합니다. 기존 KG 기반 메소드의 한계인 엔티티 기반 스코어링을 보완하기 위해, 문서 텍스트를 KG 노드 표현으로 사용하고 문서 기반 관계 필터링을 도입하여 Knowledge-Aware Retrieval (KAR)를 수행합니다.

- **Performance Highlights**: 세 가지 서로 다른 분야의 데이터셋을 대상으로 한 실험 결과, 우리의 방법이 최신 쿼리 확장 방법에 비해 성능이 우수하고 LLM 기반 검색 에이전트와 동등한 성능을 달성하는 것으로 나타났습니다.



### LLM-Human Pipeline for Cultural Context Grounding of Conversations (https://arxiv.org/abs/2410.13727)
Comments:
          19 pages, 9 figures, 7 tables

- **What's New**: 이 논문에서는 대화에서 문화적 맥락을 이해하기 위해 'Cultural Context Schema'를 도입하는 방법을 제안합니다. 특히, 감정, 대화 행위와 같은 대화 정보와 사회적 규범, 위반과 같은 문화적 정보를 포함한 구조를 구축하고 이를 바탕으로 실제 중국 문화에 맞춘 약 11만 건의 사회적 규범 및 위반 설명을 생성하였습니다.

- **Technical Details**: 논문은 LLMs (Large Language Models)을 사용하여 대화 관련 문화 정보를 생성하고, 이를 표상하는 'Norm Concepts'를 만들기 위한 다단계 접근을 채택합니다. 과정에는 인간 주도의 검증 및 대화 내용의 세부사항을 기호 주석(symbolic annotation)으로 grounding하는 것이 포함됩니다. 또한, 정서 검출(emotion detection), 감정 감지(sentiment detection), 대화 행위 검출(dialogue act detection)과 같은 하위 작업에 활용될 대규모 데이터셋을 생성합니다.

- **Performance Highlights**: 제안된 문화적 맥락 데이터셋은 경험적으로 대화 이해 작업에 대한 성능을 유의미하게 향상시키는 것으로 나타났습니다. 또한, 고품질의 데이터셋과 평가 실험을 통해 결과를 보이고 있으며, 데이터 및 코드가 MIT 라이센스 하에 공개될 예정입니다.



### MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2410.13716)
- **What's New**: 본 논문에서는 현대의 Retrieval-Augmented Generation (RAG) 평가 기준에서의 한계를 해결하기 위해, 성능 높은 LLM (Large Language Model)을 대체할 수 있는 'learning to rank' 모델을 훈련하여 'synthetic arena-based leaderboard'를 생성하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 우리는 RAG를 기반으로 한 평가 지표를 입력으로 사용하여 학습된 surrogate judge 모델을 통해 18개 언어에서 다국어 RAG 벤치마크인 MIRAGE-Bench를 개발했습니다. 이 모델은 Google의 GPT-4o를 사용하여 수행된 쌍별 평가 결과를 바탕으로 Bradley-Terry 모델을 이용해 leaderboard를 생성했습니다. 이 과정에서 우리는 언어 감지, 인용 품질, 지원도, 정답 중복성 및 유창성 등 총 7개의 평가 기준을 활용했습니다.

- **Performance Highlights**: 실험 결과, 학습된 'learning to rank' 모델은 GPT-4o를 기반으로 한 leaderboard와 높은 상관관계(Kendall Tau (τ) = 0.909)를 보였으며, 70B 이상의 매개변수를 가진 대형 모델들이 MIRAGE-Bench에서 우수한 성능을 기록했습니다. 또한, MIRAGE의 훈련 데이터는 작은 오픈 소스 모델을 개선하는 데 유용하다는 것을 보여주었습니다.



### On the Role of Attention Heads in Large Language Model Safety (https://arxiv.org/abs/2410.13708)
Comments:
          28 pages, 18 figures, 7 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 안전 메커니즘의 특정 주의(attention) 헤드의 기여도를 이해하고, 그로 인해 발생하는 안전성 문제를 분석합니다. 특히, Safety Head ImPortant Score (Ships)라는 새로운 메트릭을 도입하여 안전성과 관련된 다중 헤드 주의 메커니즘을 탐구합니다.

- **Technical Details**: 우리는 LLM의 안전성 능력을 다중 헤드 주의(mechanism)와 연결하여 해석하기 위한 연구를 진행했습니다. Ships는 개별 주의 헤드가 해로운 쿼리에 대한 거부 확률 변화에 미치는 영향을 정량화합니다. 추가적으로, Safety Attention Head AttRibution Algorithm (Sahara)을 제안하여 중요 헤드를 그룹화하고, 이들이 모델의 안전성에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과, Llama-2-7b-chat 모델에서 안전 헤드 하나를 제거했을 때, 해로운 쿼리에 대한 공격 성공률(ASR)이 0.04에서 0.64로 증가하였으며, 이는 기존 연구에서 필요로 했던 약 5%의 매개변수 수정과 대조적으로 단 0.006%를 수정함으로써 이루어졌습니다. 또한, 유사한 기본 모델에서 미세 조정된 LLM의 안전 헤드들이 겹친다는 점에서, 안전성에 대한 기존 연구와 새로운 통찰을 제공합니다.



### Unconstrained Model Merging for Enhanced LLM Reasoning (https://arxiv.org/abs/2410.13699)
Comments:
          Under review

- **What's New**: 최근 도메인 특정 대형 언어 모델 (Large Language Models, LLMs)의 발전에서, 복잡한 관계에 대한 논리적 추론과 다단계 문제 해결과 같은 reasoning 능력을 요구하는 작업에서 눈에 띄는 성공을 보여주고 있습니다. 이 작업에서는 유사하거나 이질적인 여러 전문가 모델을 통합하여 단일 LLM을 구축하는 unconstrained model merging (제약 없는 모델 병합) 프레임워크를 제안합니다.

- **Technical Details**: 이 연구에서는 homogeneous(동질) 모델 병합을 위한 세밀한 layer-wise weight merging 전략과 instruction-response fine-tuning 데이터에서 파생된 확률적 분포 지식을 활용한 heterogeneous(이질) 모델 병합 방안을 제시합니다. 이 접근법은 7개의 벤치마크와 9개의 reasoning-optimized LLMs(추론 최적화된 LLM)에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: 모델 통합을 통해 수학 및 코딩 능력이 향상되며, 이를 통해 조합적인 추론 능력이 나타났다. 특히, 코딩 능력이 더 복잡한 작업에서 우수한 결과를 보이고, 초기 pretrained 모델과의 조합이 최적의 성능 통합을 위해 중요함을 발견했습니다.



### HEALTH-PARIKSHA: Assessing RAG Models for Health Chatbots in Real-World Multilingual Settings (https://arxiv.org/abs/2410.13671)
Comments:
          Under Review

- **What's New**: 이번 연구는 인도 환자들과의 의료 챗봇 상호작용에서 수집된 실제 데이터를 기반으로 24개의 큰 언어 모델(LLM)을 평가한 최초의 포괄적인 연구입니다. 다국어 맥락에서 LLM의 능력과 제한을 평가하고 실제 사례에서 성능을 비교하며, 입력 언어가 영어가 아닐 때 정확성이 감소하는 경향을 보였습니다. 또한 코드 혼합 및 문화적 관련 질문이 모델이 응답하는 데 어려운 점이 있음을 보여주었습니다.

- **Technical Details**: 연구는 RAG(리트리벌 증강 생성) 프레임워크를 사용하여 LLM의 응답을 생성하고, 인도 영어 및 4개의 인도 인디오언어를 사용하는 750개의 질문에 대해 성능을 평가합니다. 4가지 메트릭에 기반하여 모델 응답의 객관성과 일관성을 검증하며, 이를 통해 인간 평가자 및 자동화를 통한 기술 평가를 수행하였습니다. LLM 간의 비교는 동일한 RAG 설정을 통해 공정하게 이루어졌습니다.

- **Performance Highlights**: 우리의 결과는 모델 간 성능 차이가 매우 크며, 일부 소형 모델이 대형 모델보다 더 나은 성능을 보이는 경우가 있음을 시사합니다. 비영어 쿼리에 대한 사실적 정확성이 영어 쿼리보다 일반적으로 낮으며, 지침 조정된 인도 모델이 항상 인디언 언어 쿼리에 대해 잘 작동하지 않는 경향이 관찰되었습니다. 또한, 우리가 수집한 데이터셋에 포함된 코드 혼합 및 문화적으로 관련된 질문에 대해 모델들이 답변하는 데 어려움을 겪는 경우가 많았습니다.



### signwriting-evaluation: Effective Sign Language Evaluation via SignWriting (https://arxiv.org/abs/2410.13668)
- **What's New**: SignWriting을 위해 특별히 설계된 새로운 자동 평가 메트릭스 세트를 소개합니다. 이 메트릭스들은 기존의 BLEU, chrF 및 CLIPScore를 SignWriting 특성에 맞게 조정하여 단순한 성과 평가를 넘어서는 진일보한 평가 도구를 제공합니다.

- **Technical Details**: 본 논문에서는 SignWriting의 평가 메트릭스를 평가하는 새로운 접근 방식을 제안하며, BLEU, chrF, CLIPScore의 변형을 포함합니다. 또한, 기호 간의 시각적 거리를 측정하는 새로운 메트릭도 개발하였습니다. 이 새로운 기호 거리 메트릭은 각 기호의 모양, 방향, 회전 및 위치와 같은 속성을 고려하여 SignWriting 기록의 전사 및 번역 성과를 더욱 정확하게 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, 제안된 메트릭스는 SignBank 데이터베이스 내에서의 최근접 이웃 검색과 점수 분포 분석을 통해 성능을 효과적으로 드러내기 보여 주었습니다. 다양한 메트릭의 장단점을 비교하고, 향후 SignWriting 처리 기술의 발전을 위한 귀중한 통찰력을 제공합니다.



### ORCHID: A Chinese Debate Corpus for Target-Independent Stance Detection and Argumentative Dialogue Summarization (https://arxiv.org/abs/2410.13667)
Comments:
          In EMNLP 2023

- **What's New**: 이 논문에서는 ORCHID(Oral Chinese Debate)라는 새로운 데이터셋을 소개합니다. 이는_target-independent_ 스탠스 감지와 토론 요약을 위한 첫 번째 중국어 데이터셋으로, 1,218개의 실제 토론과 476개의 고유 주제에서 파생된 것입니다. 이를 통해 기존의 비영어 연구자들에게 필요한 자원을 제공합니다.

- **Technical Details**: ORCHID 데이터셋은 1,218개의 실제 토론을 포함하며, 2,436개의 스탠스 특화 요약과 14,133개의 완전 주석 발화로 구성되어 있습니다. 데이터는 자동 음성 인식(ASR)을 통해 전사된 뒤 수동 수정 및 주석 작업을 거쳤습니다. 또한, 두 가지 수준의 요약(간단한 진술과 포괄적인 요약)을 제공합니다.

- **Performance Highlights**: 예비 연구 결과, 스탠스 감지와 요약 작업이 긴밀하게 연관되어 있으며 이러한 통합 작업이 논쟁적인 대화에 대한 요약 품질을 개선할 수 있는 가능성을 보여줍니다.



### Red and blue language: Word choices in the Trump & Harris 2024 presidential deba (https://arxiv.org/abs/2410.13654)
Comments:
          Submitted to PLOS ONE, under review

- **What's New**: 이번 논문은 2024년 9월 10일에 열린 트럼프와 해리스 후보 간 정치토론에서의 언어 사용 차이를 분석합니다. 주목할 점은 언어가 상대방과 유권자에게 어떻게 조정되는지를 다룬다는 것입니다.

- **Technical Details**: 정량적 및 질적 분석을 통해 언어 사용의 의미론적(semantic) 및 화용론적(pragmatic) 특성, 즉 가치와 이념의 프레이밍(framing), 감정에 호소하는 방법, 단어의 구체성과 특수성 정도, 단수 또는 복수 대명사의 사용을 분석합니다.

- **Performance Highlights**: 핵심 발견으로는 해리스가 회복과 권한 부여를 중심으로 문제를 프레이밍하는 반면, 트럼프는 위기와 쇠퇴에 집중하는 반면, 감정적 언어 사용에서 유사성을 보였고, 트럼프가 해리스를 이름으로 언급하지 않고, 해리스는 자주 트럼프를 언급하는 경향이 있었습니다.



### A new approach for fine-tuning sentence transformers for intent classification and out-of-scope detection tasks (https://arxiv.org/abs/2410.13649)
Comments:
          Appearing at Empirical Methods in Natural Language Processing 2025 - Industry Track

- **What's New**: 이번 연구는 VA(virtual assistant) 시스템에서 OOS(out-of-scope) 쿼리를 거부하거나 리다이렉트하는 방법에 대해 다룹니다. 기존 연구들은 intent classification 작업과 결합된 OOS 거부 방법을 사용했지만, 이 방법들이 종종 OOS 임베딩과 겹치는 문제가 있었습니다. 연구진은 이를 해결하기 위해 auto-encoder를 활용하여 in-scope 임베딩 재구성 손실을 도입함으로써 크로스 엔트로피 손실을 정규화하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: OOS 쿼리를 감지하는 기존 접근 방식은 주로 transformer 기반의 문장 인코더를 통해 문장 인코딩을 생성하고 이를 이용해 분류하는 방식입니다. 하지만, 이러한 방법은 크로스 엔트로피 손실을 사용하는 경우 in-scope 임베딩이 분산되어 OOS 임베딩과 겹칠 수 있습니다. 연구팀은 auto-encoder를 통해 in-scope 임베딩의 전역적인 분산을 줄이는 새로운 손실 정규화 방식을 제안하며, 이 과정을 통해 in-scope 쿼리와 OOS 쿼리를 더 잘 구분할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 OOS 인스턴스를 거부하기 위한 precision-recall curve에서 1-4%의 향상을 보여주었으며, intent classification 성능에 영향을 주지 않았습니다. 이는 VA 시스템에서 OOS 거부 기능을 효과적으로 처리할 수 있는 가능성을 제시합니다.



### SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs (https://arxiv.org/abs/2410.13648)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)에 대한 새로운 데이터셋, SimpleToM을 소개하고, LLM이 자신의 정신 상태를 추론하여 행동을 예측하고 합리성을 판단하는 능력을 평가합니다. 기존의 이론적 접근법을 넘어 실생활 시나리오에서의 응용을 탐구합니다.

- **Technical Details**: SimpleToM 데이터셋은 1147개의 간결하고 다양한 이야기와 3441개의 질문을 포함하고 있으며, 정신 상태(Mental state)와 행동(Behavior), 판단(Judgment)에 대한 질문이 포함되어 있습니다. 이러한 질문들은 LLM이 정보 인식, 행동 예측 및 행동의 적절성을 판단하는 능력을 측정합니다. 모델들은 단순한 스토리에 대한 논리적 추론을 요구받습니다.

- **Performance Highlights**: 실험 결과, 대부분의 모델이 정신 상태 예측에서는 높은 성능을 보였지만, 행동 예측 및 합리성 판단에서 낮은 성능을 보였습니다. GPT-4o 모델이 행동 예측 정확도를 49.5%에서 93.5%로, 판단 정확도를 15.3%에서 94.7%로 향상시키는 등 개입을 통해 개선할 수 있었지만, 이는 고유한 이론적 사고 능력의 한계를 나타냅니다.



### An Active Learning Framework for Inclusive Generation by Large Language Models (https://arxiv.org/abs/2410.13641)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)가 다양한 하위 집단을 대표하는 텍스트를 생성하도록 보장하기 위한 새로운 클러스터링 기반 액티브 러닝 프레임워크를 제안합니다. 이 프레임워크는 지식 증류(knowledge distillation)를 통해 중간 출력을 변환하여, 생성 작업에 대한 효과적인 액티브 러닝을 가능하게 하였습니다.

- **Technical Details**: 제안된 프레임워크는 액티브 러닝이 데이터를 구성하고 모델을 훈련하는 과정을 반복하는 동안, 정보가 많은 샘플을 선택하여 데이터의 분포 편향성을 방지하는 것을 목표로 합니다. 또한, 외부 LLM의 지식을 활용해 정보를 추가하고, 인간의 노력을 줄이기 위해 외부 LLM의 출력을 인간 주석자가 검증한 후 학습 모델에 전달합니다.

- **Performance Highlights**: 실제로 두 개의 새로운 데이터 세트를 구축하여 밴치마킹 했으며, 이로 인해 기초 모델 대비 2%-10%의 성능 향상이 있었습니다. 결과적으로, 다양한 데이터 하위 그룹에서 성능의 일관성이 높아지고, 어휘적 다양성(lexical diversity)이 증가하여 모델의 저항력이 향상되었습니다.



### Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation (https://arxiv.org/abs/2410.13640)
Comments:
          33 pages, 18 figures, 12 tables

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 정확도를 추정하기 위한 기존의 레이블 필요성을 없애고, 잠재 공간(latent space)에서 Chain-of-Embedding (CoE) 메서드를 제안하였습니다. 이 방식은 LLM이 스스로 출력 없는 자기 평가를 수행할 수 있도록 합니다.

- **Technical Details**: CoE는 LLM의 추론 과정에서 생성되는 모든 점진적 은닉 상태(hidden state)를 포함하며, 이는 LLM의 사고 경로(thinking path)를 나타냅니다. 연구 결과, LLM이 정답을 낼 때와 아닐 때 CoE의 특징이 다르게 나타나는 것을 확인하였으며, 이러한 차이를 통해 LLM의 응답 정확성을 추정할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험을 통해 다양한 분야(수학, 추론, 지식, 이해)에서 LLM 7종의 성과를 평가하였으며, CoE 메서드가 지연 없이 신뢰성 있는 피드백을 제공하는 것을 입증하였습니다. 또한, 이 방법은 레이블이 필요 없고, 밀리세컨드 수준의 계산 비용으로 대규모 환경에서도 실시간 피드백이 가능함을 강조합니다.



### A Comparative Study on Reasoning Patterns of OpenAI's o1 Mod (https://arxiv.org/abs/2410.13639)
- **What's New**: OpenAI의 o1 모델이 다양한 Test-time Compute 방법들과 비교하여 우수한 성능을 발휘하고 있으며, 이는 기존의 모델 파라미터 증가 방식과는 다른 접근법입니다.

- **Technical Details**: o1 모델의 추론 능력 향상을 위해 Best-of-N (BoN), Step-wise BoN, Agent Workflow 및 Self-Refine 방법과 비교하였고, 일반적인 추론 벤치마크인 수학, 코딩, 상식 추론 분야에서 실험을 진행했습니다. 특히, o1 모델은 CoT 기반 접근을 통해 코딩 및 수학 작업에서 최고의 성능을 나타냈습니다.

- **Performance Highlights**: o1 모델은 HotpotQA, Collie, USACO 및 AIME 벤치마크에서 대다수 작업에 대해 가장 높은 성능을 기록하였으며, 다양한 추론 패턴을 통해 문제 해결 능력을 극대화하였습니다.



### Enhancing Fact Retrieval in PLMs through Truthfulness (https://arxiv.org/abs/2410.13562)
- **What's New**: 본 연구는 프리트레인 언어 모델(Pre-trained Language Model, PLM)의 숨겨진 상태(hidden states) 표현을 활용하여 사실 기반의 정보를 얻는 데 도움을 주는 보조 모델(helper model)의 사용을 탐구합니다. 이를 통해 사실 추출의 정확성을 최대 33%까지 향상시키는 방법을 제시합니다.

- **Technical Details**: 연구에서는 PLM의 숨겨진 상태를 평가하여 입력의 진실성을 판단합니다. 보조 모델은 PLM의 상위 k개의 예측 중 올바른 답을 분류합니다. 이 방법은 사실 추출을 개선하기 위해 사용되며, PLM의 여러 마스킹된 예제에서 평가됩니다.

- **Performance Highlights**: 실험 결과, 제안된 보조 모델은 여러 마스킹된 PLM에서 사실 추출 성능을 최대 33% 향상시킴을 보여줍니다. 이 연구는 PLM의 숨겨진 상태 표현이 사실 기반 지식 추출을 개선할 수 있는 잠재력을 강조합니다.



### Integrating Temporal Representations for Dynamic Memory Retrieval and Management in Large Language Models (https://arxiv.org/abs/2410.13553)
- **What's New**: 본 논문에서는 SynapticRAG라는 새로운 메모리 회수 접근 방식을 제안합니다. 이 방법은 생물학적 시냅스를 모방하여 발생 시간에 따른 사건 차별화를 통해 메모리 벡터에 시냅틱 다이나믹스를 통합합니다. 또한, 메모리의 중요성을 동적으로 업데이트하며, 전통적인 RAG 모델에 비해 최대 14.66%의 향상된 메모리 회수 정확도를 보입니다.

- **Technical Details**: SynapticRAG는 메모리 벡터에 시간적 표현을 통합하여 각 메모리 노드의 의미적 및 시간적 관계를 형성합니다. 이 모델은 동적 시간 왜곡(DTW) 방식을 사용하여 메모리 노드 간의 누적 거리 행렬을 계산하고, 자극 전파 메커니즘을 통해 노드 간의 자극을 조절합니다. LIF(Leaky Integrate-and-Fire) 모델을 기반으로 하는 시냅틱 전파 통제 메커니즘도 포함되어 있습니다.

- **Performance Highlights**: 영어, 일본어, 중국어 데이터셋에서의 실험 결과, SynapticRAG는 기존 방법들에 비해 우수한 성능을 보이며, 특히 메모리 회수 정확도에서 향상을 나타냈습니다. 이 모델의 도입으로 AI 대화 에이전트의 맥락 인식 능력이 크게 향상되었습니다.



### Bias in the Mirror : Are LLMs opinions robust to their own adversarial attacks ? (https://arxiv.org/abs/2410.13517)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 편향(bias)의 강도와 유연성을 평가하기 위한 새로운 접근 방식을 제시합니다. 두 개의 LLM 인스턴스가 자기 논쟁(self-debate)을 통해 상반된 관점을 주장하며 중립적인 모델을 설득하려고 시도합니다. 이를 통해 편향이 정보 왜곡(misinformation)을 강화하거나 해로운 관점으로 이동할 수 있는지 여부를 탐구합니다.

- **Technical Details**: 이 연구는 다양한 크기, 출처 및 언어의 여러 LLM을 대상으로 진행된 실험으로, 편향의 지속성과 언어적 맥락에서의 유연성을 평가합니다. 특히, 언어에 따라 어떻게 편향이 표현되는지를 분석하며, LLM들이 이차 언어(secondary languages)에서 서로 다른 편향을 보이는 현상을 관찰합니다. 또한, LLM의 응답과 인간의 반응을 비교하기 위해 포괄적인 인간 평가를 수행합니다.

- **Performance Highlights**: 논문은 다양한 LLM에 대한 포괄적인 분석을 통해 모델들이 편향에 대한 입장을 어떻게 변경할 수 있는지를 보여줍니다. 기존의 연구보다 더욱 세분화된 방식으로 편향의 평가가 진행되어, LLM의 판단력이 인간의 판단과 어떻게 일치 또는 불일치하는지를 조명합니다. 이 연구는 오해의 소지가 있는 주장을 어떻게 다루는지에 대한 더 나은 이해를 제공합니다.



### GeoCoder: Solving Geometry Problems by Generating Modular Code through Vision-Language Models (https://arxiv.org/abs/2410.13510)
- **What's New**: 본 논문에서는 기하 문제 해결을 위해 비전-언어 모델(Vision-Language Models, VLMs)을 향상시키기 위한 새로운 접근 방식인 GeoCoder를 제안합니다. 이 모델은 사전 정의된 기하 함수 라이브러리를 활용하여 코드를 생성 및 실행함으로써 수학적 연산의 정확성을 높입니다.

- **Technical Details**: GeoCoder는 모듈식 코드 파인튜닝(modular code finetuning)을 통해 기하 문제 해결을 위한 코드를 생성하고 실행합니다. 이 과정에서 사용되는 함수 라이브러리는 수식을 올바르게 적용함으로써 오류를 최소화하며, RAG-GeoCoder라는 비파라메트릭 메모리(non-parametric memory) 모듈을 채택하여 함수의 검색 기능을 향상시킵니다.

- **Performance Highlights**: GeoCoder와 RAG-GeoCoder는 GeomVerse 데이터셋에서 다양한 문제 복잡도에 대해 평균 16% 향상된 성능을 보여주었습니다. 이러한 성과는 전통적인 파인튜닝 방법과 비교하여 기하 학적 추론 능력을 획기적으로 증가시켰습니다.



### RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards (https://arxiv.org/abs/2410.13509)
- **What's New**: 이 논문에서는 기존의 Supervised Fine-Tuning (SFT) 방법을 넘어 Differentiable Data Rewards (DDR) 방법을 제안하여 Retrieval-Augmented Generation (RAG) 시스템의 에이전트를 최적화함으로써 지식 선호도를 정렬하는 방법을 소개합니다.

- **Technical Details**: DDR 방법은 전체 RAG 시스템에서 각 에이전트에 대한 보상을 수집하고 이를 바탕으로 에이전트를 최적화합니다. 이 과정에서 롤아웃 방법을 사용하여 여러 잠재적 응답을 샘플링하고, 이러한 변동이 RAG 시스템에 미치는 영향을 평가하여 에이전트를 최적화합니다.

- **Performance Highlights**: 실험 결과에 따르면 DDR 방법이 기존 SFT 방법보다 상당히 우수한 성능을 보였으며, 특히 파라미터가 작은 LLM에서 효과적이었습니다. DDR을 통해 생성 모듈이 문서에서 중요한 정보를 추출하는 능력이 향상되고, 파라미터 메모리와 외부 지식 간의 갈등을 완화할 수 있습니다.



### Enhancing Text Generation in Joint NLG/NLU Learning Through Curriculum Learning, Semi-Supervised Training, and Advanced Optimization Techniques (https://arxiv.org/abs/2410.13498)
- **What's New**: 본 연구에서는 Joint Natural Language Generation (NLG)와 Natural Language Understanding (NLU) 학습 상황에서 텍스트 생성을 개선하기 위한 새로운 접근 방식을 개발했습니다.

- **Technical Details**: 데이터는 주석이 달린 데이터셋을 수집하고 전처리하여 준비하였으며, 여기에는 데이터 정리(cleaning), 토큰화(tokenization), 형태소 분석(stemming), 불용어 제거(stop-word removal)가 포함됩니다. 또한, 특징 추출 기법으로는 POS tagging, Bag of Words, Term Frequency-Inverse Document Frequency (TF-IDF)가 사용됩니다. Transformer 기반의 인코더와 디코더는 긴 범위의 의존성을 포착하고 소스-타겟 시퀀스 모델링을 개선합니다. Optimized BERT와 Hybrid Redfox Artificial Hummingbird Algorithm (HRAHA)와 같은 사전 훈련된 언어 모델이 통합되었습니다.

- **Performance Highlights**: 정책 경량화 기법을 통한 강화 학습(reinforcement learning), 반지도 학습(semi-supervised training), 개선된 주의 메커니즘(attention mechanisms) 및 미분 가능한 근사치(differentiable approximations)를 사용하여 모델을 미세 조정하고 복잡한 언어 과제를 효과적으로 처리합니다. 이 모델은 Python을 사용하여 구현되었습니다.



### Repetition Neurons: How Do Language Models Produce Repetitions? (https://arxiv.org/abs/2410.13497)
- **What's New**: 이번 논문에서는 텍스트 생성 작업에서 반복 문제를 담당하는 기술 신경망인 반복 뉴런(repetition neurons)을 소개합니다. 이 뉴런들은 반복이 계속될수록 더 강하게 활성화되며, 이전 맥락을 반복적으로 복사하는 작업으로 인식하는 것으로 보입니다.

- **Technical Details**: 우리는 최근의 사전 훈련된 언어 모델에 의해 생성된 텍스트에서 반복의 시작 전후의 활성화 값을 비교하여 반복 뉴런을 식별합니다. 실험은 세 개의 영어 모델(예: Gemma-2B, Pythia-2.8B-Deduped, LLaMA-3.2-3B)과 하나의 일본어 모델(LLM-jp-3-1.8B)을 사용하여 수행되었습니다. 반복 뉴런은 중간 및 최종 레이어에 위치하며, 이들의 비활성화는 반복 토큰의 출력 확률을 억제하고 활성화는 이를 증가시킵니다.

- **Performance Highlights**: 이번 연구의 결과는 반복 뉴런이 반복하는 텍스트에서 더 강하게 활성화된다는 것을 보여주었으며, 이는 언어 모델이 텍스트 생성을 수행할 때의 내부 메커니즘에 대한 새로운 통찰력을 제공합니다. 실험 결과는 반복성을 억제하는 새로운 접근법에 대한 가능성을 제시합니다.



### Seeing Through VisualBERT: A Causal Adventure on Memetic Landscapes (https://arxiv.org/abs/2410.13488)
Comments:
          Accepted at EMNLP Findings 2024

- **What's New**: 본 논문은 Structural Causal Model (SCM)을 기반으로 한 새로운 프레임워크를 제안하여, offensive memes의 탐지에서 투명성을 높이며 모형의 행동 해석을 가능하게 한다. VisualBERT를 활용하여 meme 입력과 causal concepts를 모두 고려하여 클래스를 예측한다.

- **Technical Details**: 이 프레임워크는 기존 interpretability 기술의 한계를 극복하기 위해 causal concepts를 통합하고, dynamic routing과 adversarial learning을 활용하여 meme의 공격성을 예측한다. 아울러, 모델 예측의 원인과 오류 사례를 명확히 설명한다.

- **Performance Highlights**: 정량적 분석 결과, 제안한 모델링 기법들이 기존의 input attribution 방법들과 비교하여 causality를 만족하지 못하는 점을 강조하며, 이로 인해 safety-critical applications에서의 신뢰성에 의문을 제기한다. 또한, qualitative 분석을 통해 모델의 결정이 정당화될 수 있는지를 평가하였다.



### IterSelectTune: An Iterative Training Framework for Efficient Instruction-Tuning Data Selection (https://arxiv.org/abs/2410.13464)
- **What's New**: 이 연구에서는 사람의 개입 없이도 고품질의 instruction 데이터를 효과적으로 선택할 수 있는 IterSelectTune이라는 반복 훈련 정책을 소개합니다.

- **Technical Details**: 이 방법은 BERT-base 분류기를 사용하여 개발되었으며, 기본 LLM의 출력을 원본 응답과 비교하여 'hard' 데이터와 'easy' 데이터를 정의합니다. IterSelectTune은 두 가지 주요 단계를 거쳐 작동하며, 초기 다양한 서브셋을 선택한 후 수집된 데이터를 반복적으로 학습하여 classifier를 훈련합니다.

- **Performance Highlights**: 이 연구의 실험 결과, 약 20%의 instruction 데이터로 훈련한 모델이 전체 데이터로 훈련한 모델보다 여러 벤치마크에서 일관되게 우수한 성능을 보여주었으며, Alpaca 및 WizardLM에 대한 테스트에서도 향상된 성능을 입증하였습니다.



### Breaking the Manual Annotation Bottleneck: Creating a Comprehensive Legal Case Criticality Dataset through Semi-Automated Labeling (https://arxiv.org/abs/2410.13460)
- **What's New**: 이 논문에서는 스위스 연방 대법원 판결의 미래 법리에 대한 영향을 평가하기 위한 새로운 Criticality Prediction (사례 중요도 예측) 데이터셋을 소개합니다. 기존의 수작업 주석 접근 방식과 달리, 본 데이터셋은 반자동적으로 레이블을 유도하여 훨씬 더 큰 데이터셋을 제공합니다.

- **Technical Details**: 제안된 데이터셋은 2단계 레이블링 시스템을 특징으로 하며, (1) LD-Label: 주요 결정으로 발표된 사례를 식별하는 이진 지표, (2) Citation-Label: 사례의 인용 빈도와 최근성에 따라 사례를 평가합니다. 이 데이터셋은 2002년부터 2023년까지의 사례를 포함하며 언어는 독일어, 프랑스어, 이탈리아어로 구성됩니다.

- **Performance Highlights**: 여러 다국어 모델을 평가한 결과, 세밀하게 조정된 모델이 제로샷(Zero-shot) 기준선보다 일관되게 우수한 성능을 보였습니다. 이를 통해 작업 특화적 적응이 필요함을 입증하였습니다.



### MedINST: Meta Dataset of Biomedical Instructions (https://arxiv.org/abs/2410.13458)
- **What's New**: 새로운 메타 데이터셋 MedINST가 발표되어, 133개의 바이오메디컬 NLP 작업과 700만 개 이상의 샘플을 포함하여 기존 의료 데이터셋의 한계를 극복합니다. 이 데이터셋은 대규모 언어 모델(LLM)의 일반화 능력을 평가하기 위한 MedINST32 벤치마크를 구성합니다.

- **Technical Details**: MedINST는 12개 카테고리의 133개 바이오메디컬 NLP 작업으로 구성된 데이터셋으로, 각 작업은 텍스트 생성 작업으로 간주되어 훈련 데이터는 인스트럭션 형식으로 포맷되어 있습니다. 인스트럭션은 인간에 의해 주석 처리되어 각 데이터셋/작업에 맞춰 세밀하게 조정되었습니다.

- **Performance Highlights**: MedINST를 기반으로 LLM을 미세 조정한 결과, cross-task generalization 능력이 향상됨을 보여주었습니다. 이는 바이오메디컬 도메인에서 LLM의 유용성을 극대화하는 데 기여할 수 있습니다.



### Unlocking Legal Knowledge: A Multilingual Dataset for Judicial Summarization in Switzerland (https://arxiv.org/abs/2410.13456)
- **What's New**: 이번 논문은 스위스 연방 대법원(SFSC)의 판결을 바탕으로 한 새로운 데이터셋인 Swiss Leading Decision Summarization (SLDS)를 소개하고 있습니다. 이 데이터셋은 독일어, 프랑스어 및 이탈리아어로 된 18,000개의 법원 판결과 독일어 요약을 포함하고 있어, 다국어 법적 요약에 대한 연구를 촉진할 수 있습니다.

- **Technical Details**: 논문에서는 3가지 mT5(multi-lingual T5) 변형 모델과 고유 모델을 미세 조정(fine-tuning)하고 평가했습니다. 분석 결과, 고유 모델은 제로샷(zero-shot) 및 원샷(one-shot) 설정에서 우수한 성능을 보였으나, 미세 조정된 모델이 여전히 강력한 경쟁력을 유지하는 것으로 나타났습니다.

- **Performance Highlights**: SLDS 데이터셋의 공개를 통해 법적 요약 및 법무 전문가를 위한 보조 기술 개발에 대한 추가 연구가 촉진될 것으로 기대됩니다. 이 데이터셋은 수백만 건의 판결을 법률 연구에 더 쉽게 접근할 수 있게 할 잠재력이 있습니다.



### Parameter-efficient Adaptation of Multilingual Multimodal Models for Low-resource ASR (https://arxiv.org/abs/2410.13445)
- **What's New**: 본 연구에서는 낮은 자원을 가진 언어에 대한 자동 음성 인식(ASR)을 개선하기 위해, 다국어 다중 모달 모델인 SeamlessM4T를 활용하여 매개변수 효율적인 미세 조정(parameter-efficient fine-tuning) 및 텍스트 전용 적응(text-only adaptation) 기술을 결합한 방법을 제시합니다.

- **Technical Details**: SeamlessM4T는 다중 언어 및 다중 모달 머신 번역 지원을 제공하는 엔드-투-엔드 모델로, 96개 언어의 입력 및 출력에 대해 자동 음성 인식, 텍스트-음성 변환 등 여러 작업을 수행할 수 있습니다. 이 모델은 셀프-슈퍼바이즈드(self-supervised) 방식으로 백만 시간 이상의 무 라벨 음성을 학습하여 성능이 개선되었습니다.

- **Performance Highlights**: 본 논문에서는 높은 자원 언어에서 낮은 자원 언어로의 언어 간 전이(cross-lingual transfer)를 통해, 라벨이 없는 음성 데이터 없이도 WER(Word Error Rate)를 17% 이상 감소시킬 수 있음을 보였습니다.



### NLIP_Lab-IITH Multilingual MT System for WAT24 MT Shared Task (https://arxiv.org/abs/2410.13443)
Comments:
          WMT 24 WAT Shared Task IndicMultiMT (Best System)

- **What's New**: 이번 논문은 NLIP Lab의 다국어 기계 번역 시스템에 대해 설명하며, 22개의 정해진 인도 언어에 대한 WAT24 다국어 MT 작업을 수행했습니다. 이 연구에서는 alignment agreement objectives를 활용한 사전 훈련 기법을 탐구하고, 소스 문장에서 단어를 대체하기 위해 이중 언어 사전을 사용했습니다. 이 모델은 243M의 파라미터를 가지고 있으며, En-Indic 및 Indic-En 방향에서 모든 비정형 실험 benchmark에서 우수한 성능을 보여줍니다.

- **Technical Details**: 모델 이름은 IndicRASP이며, 22개의 Indic 언어로부터의 데이터를 기반으로 사전 훈련되었습니다. Transformer 빅 모델을 사용하고, 6개의 인코더 및 6개의 디코더 레이어로 구성되어 있으며, embedding size는 1024입니다. 훈련은 Adam optimizer를 사용하며, 주어진 여러 하이퍼파라미터들이 적용되었습니다. 데이터 샘플링은 온도 샘플링(temperature sampling) 기법을 사용했습니다.

- **Performance Highlights**: IN22-Gen benchmark에서 평균 chrF++ 점수는 46.80, BLEU 점수는 18.19로 기록되었습니다. Indic-En 방향에서는 평균 chrF++ 점수 56.34, BLEU 점수 30.82를 달성했습니다. IndicRASP 모델은 이전 IndicTransv1(474M 모델)보다 경쟁력이 있으며, 특히 Indic-Indic 번역에서는 소폭의 성과 차이를 보였습니다.



### Think Thrice Before You Act: Progressive Thought Refinement in Large Language Models (https://arxiv.org/abs/2410.13413)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 응답 품질을 향상시키기 위한 새로운 방법인 Progressive Thought Refinement (PTR) 프레임워크를 제안합니다. 기존 방법들이 특정 작업에 의존하여 일반화가 어려웠던 문제를 해결하기 위해, PTR은 두 단계의 세분화된 접근 방식을 사용합니다.

- **Technical Details**: PTR는 두 가지 주요 단계인 (1) 'Thought data construction' 단계와 (2) 'Thought-Mask Fine-Tuning' 단계를 포함합니다. 첫 번째 단계에서는 약한 모델과 강한 모델 간의 협력 선택 전략을 사용해 고품질 점진적 정제 데이터셋을 구축합니다. 두 번째 단계에서는 'thought'를 마스킹하고 손실 가중치를 조정하여 LLM이 개선 방법을 암묵적으로 이해하도록 합니다. 이를 통해 LLM이 이전 사고에서 개선된 응답을 생성하도록 유도합니다.

- **Performance Highlights**: 실험 결과, PTR을 사용한 LLM은 지식 추론, 코드 생성, 수학적 추론 등 다양한 작업에서 평균 성능이 49.6%에서 53.5%로 향상되었습니다. 특히, 보다 개방된 작업에서도 정답의 질과 형식에서 유의미한 향상이 있으며, 이는 LLM이 스스로 개선할 수 있도록 훈련되었음을 나타냅니다.



### Attr-Int: A Simple and Effective Entity Alignment Framework for Heterogeneous Knowledge Graphs (https://arxiv.org/abs/2410.13409)
- **What's New**: 이 연구는 이질적인 지식 그래프 간의 개체 정렬에 대한 새로운 접근 방식을 제시합니다. 구체적으로, 기존의 개체 정렬 방법들이 비구조적 이질성으로부터 발생하는 한계를 넘어서는 두 가지 새로운 벤치마크를 소개합니다.

- **Technical Details**: 이 연구는 Attr-Int라는 새로운 개체 정렬 프레임워크를 제안하며, 이 프레임워크는 혁신적인 속성 정보 상호 작용 방법을 통합하여 기존의 구조적 임베딩 기법과 결합해 개체 정렬 성능을 향상시킵니다. 이 과정에서 속성 데이터는 별도의 임베딩 학습 없이 활용됩니다.

- **Performance Highlights**: 우리가 제안한 Attr-Int 프레임워크는 두 개의 새로운 벤치마크에서 최신 기술들을 능가하는 성능을 보여주었습니다. 이는 실제 세계의 이질적인 개체 정렬 시나리오를 모사하여 실험된 결과입니다.



### Linguistically Grounded Analysis of Language Models using Shapley Head Values (https://arxiv.org/abs/2410.13396)
- **What's New**: 이번 논문은 언어 모델이 언어 지식을 어떻게 인코딩하는지를 이해하는 데 중점을 두고 있습니다. Shapley Head Values (SHVs)를 이용하여 BERT와 RoBERTa 모델에서의 형태통사적 현상 처리 방식을 조사하였습니다. 이 연구는 언어 모델이 특정 언어 이론에 해당하는 서브네트워크를 학습한다고 제안하고 있습니다.

- **Technical Details**: 본 연구에서는 BLiMP 데이터셋을 활용하여 13개의 현상과 67개의 구조에 대해 문법성 분류 과제를 수행하였습니다. SHVs 기반의 클러스터링을 통해 언어 모델의 관련 주의 헤드들이 어떻게 군집화되는지를 보여줍니다. 언어 모델 내부에 존재하는 관련 현상들이 클러스터링되는 경향을 발견하였습니다.

- **Performance Highlights**: 연구 결과, 두 모델(BERT, RoBERTa)이 서로 다른 형태통사적 현상을 처리하는 방식에서 뚜렷한 패턴을 드러냈습니다. 이는 SHVs 기반의 분석을 통해 모델이 언어 정보를 조직하고 처리하는 방식을 이해하는 데 기여합니다. 또한, 주의 헤드를 제거했을 때 성능의 국소적 변화가 관찰되어 클러스터의 유효성을 입증하였습니다.



### Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs (https://arxiv.org/abs/2410.13394)
- **What's New**: 이번 연구에서는 다국어 평가의 필요성에 대한 인식을 바탕으로 Cross Lingual Auto Evaluation (CIA) Suite라는 새로운 평가 프레임워크를 소개합니다. 이 프레임워크는 다국어 평가를 위한 새로운 테스트 세트인 Recon을 포함하고 있습니다.

- **Technical Details**: CIA Suite는 평가자 LLM(Hercule)과 대규모 인간 주석 데이터로 구성된 새로운 멀티태스킹 테스트 세트를 통해 다국어 성능 평가를 지원합니다. Hercule 모델은 영어에서 이용 가능한 기준 답변을 기반으로 비영어 응답에 점수를 부여하는 방식으로 저자들은 적은 자원으로도 인공지능 모델을 평가할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: Hercule의 실험 결과는 인간의 판단과 더욱 밀접하게 일치하며, 독점적 모델에 비해 성능이 뛰어난 것을 확인했습니다. 또한, 비영어 언어에 있어 0샷 평가(zero-shot evaluation)에서도 높은 효과를 나타내어 다국어 LLM의 다용성을 강조했습니다.



### Metacognitive Monitoring: A Human Ability Beyond Generative Artificial Intelligenc (https://arxiv.org/abs/2410.13392)
Comments:
          28 pages, 2 figures. arXiv admin note: substantial text overlap with arXiv:2403.05152

- **What's New**: 본 연구는 대형 언어 모델 (LLMs)인 ChatGPT가 인간의 메타인지적 모니터링 능력을 가지고 있는지 탐구합니다. 특히 기억 성능을 항목별로 예측하는 능력에 주목했습니다.

- **Technical Details**: 우리는 교차 에이전트 예측 모델을 사용하여 인간과 ChatGPT의 메타인지 성능을 비교했습니다. 실험은 적절하거나 부적절한 맥락 문장에 이어지는 garden-path 문장을 포함한 언어 기반 기억 작업이었습니다.

- **Performance Highlights**: 인간 참가자는 문장의 기억 가능성을 평가했고, 놀랍게도 인식 기억 테스트에서 실적이 확인되었습니다. 반면, ChatGPT는 유사한 예측 능력을 보여주지 않았고, GPT-3.5-turbo, GPT-4-turbo, GPT-4o 모델도 인간의 기억 성능을 항목별로 정확하게 예측하지 못했습니다.



### LAR-ECHR: A New Legal Argument Reasoning Task and Dataset for Cases of the European Court of Human Rights (https://arxiv.org/abs/2410.13352)
Comments:
          Published in Natural Legal Language Processing (NLLP) 2024 workshop

- **What's New**: 논문에서는 Legal Argument Reasoning (LAR)이라는 새로운 과제를 소개하고, 이를 통해 대형 언어 모델(LLMs)의 법적 추론 능력을 평가하고자 합니다. 이 과제는 법원의 주장 연쇄에서 사건의 사실을 기준으로 올바른 다음 진술을 선택하는 것을 요구합니다.

- **Technical Details**: 우리는 LAR-ECHR라는 데이터셋을 유럽 인권 재판소(ECHR)의 사례를 통해 구축했습니다. 이 데이터셋은 403개의 사례로 구성되어 있으며, LLM이 제공된 사례의 사실 및 이전의 법적 주장과 함께 다음 진술을 선택하게 됩니다. LAR는 단순한 패턴 인식이나 암기 이상의 법적 및 상식적 추론을 요구합니다.

- **Performance Highlights**: 실험 결과, 최상위 모델(GPT-4o)은 LAR-ECHR에서 75.8%의 정확도를 기록했습니다. 이는 LegalBench에서의 최고 정확도(73.3%)와 유사하며, 모델 개선의 가능성이 여전히 크다는 것을 나타냅니다. 다양한 법적 시스템에 대해 LAR 데이터셋을 구축할 수 있는 방법도 제시되어 있습니다.



### Representation Learning of Structured Data for Medical Foundation Models (https://arxiv.org/abs/2410.13351)
Comments:
          NeurIPS 2024 Workshop on Unifying Representations in Neural Models (UniReps 2024)

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 의료 분야에서 비문자적 구조 데이터, 특히 ICD-10 및 SNOMED-CT와 같은 의료 코드를 효과적으로 처리하는 데 직면한 문제들을 해결하기 위한 UniStruct 아키텍처를 제안합니다.

- **Technical Details**: UniStruct는 비구조화된 텍스트와 구조화된 데이터를 결합한 다중 모달(multi-modal) 의료 모델을 설계하고, 의료 코드에 적합하도록 하위 단어(tokenization) 기법을 조정하여 이러한 문제를 해결합니다. 아이디어는 자주 함께 발생하는 의료 코드 그룹을 단일 토큰으로 처리하는 것입니다.

- **Performance Highlights**: 내부 의료 데이터베이스에서 10억 개 이상의 토큰으로 사전 훈련된 UniStruct 모델은 평가 메트릭에서 최대 23% 개선을 달성했으며, EHRSHOT 공공 벤치마크에서 1/1000의 사전 훈련 데이터에도 불구하고 42% 이상의 하위 작업에서 성능을 개선했습니다.



### Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancemen (https://arxiv.org/abs/2410.13344)
- **What's New**: 이 논문에서는 Cerberus라는 새로운 adaptive parallel decoding framework를 제안합니다. Cerberus는 각 decoding step에서 적절한 decoding 방식을 선택하는 gating 메커니즘을 도입하고, sequential knowledge를 활용하는 새로운 decoding heads 방식을 소개하여 inference의 효율성을 향상시킵니다.

- **Technical Details**: Cerberus는 두 가지 주요 구성 요소를 포함하고 있습니다: 1) Sequential knowledge를 통합하여 prediction accuracy를 개선하는 novel decoding heads인 Cerberus Heads, 2) 모델의 confidence level을 기반으로 auto-regressive decoding과 parallel decoding을 선택하는 entropy-based gating mechanism. 이 방법들은 decoding 과정 중 overhead를 줄이고, 전체 inference 효율성을 높입니다.

- **Performance Highlights**: Cerberus는 MT-Bench에서 auto-regressive decoding 대비 최대 2.12배의 속도를 달성하였으며, Medusa보다 10% - 30% 더 빠르고, 더 우수한 generation quality를 보여주었습니다.



### Do LLMs Overcome Shortcut Learning? An Evaluation of Shortcut Challenges in Large Language Models (https://arxiv.org/abs/2410.13343)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 성능 및 일반화 능력에 미치는 지름길(shortcut)의 영향을 평가하기 위해 Shortcut Suite라는 포괄적인 테스트 스위트를 제안합니다. 이 스위트는 6종의 지름길 유형, 5개의 평가 지표 및 4가지 프롬프트 전략을 통합하여 LLM의 성능을 평가합니다.

- **Technical Details**: Shortcut Suite는 LLM의 성능을 아래의 세 가지 관점에서 평가합니다: 1) LLM이 다운스트림 태스크에서 지름길 의존도를 평가하기 위해 6개의 데이터셋 수집 및 정확성을 분석. 2) 정확도 외에 설명 능력 평가를 위한 3가지 새로운 지표(Semantic Fidelity Score, Internal Consistency Score, Explanation Quality Score) 도입. 3) 지름길 학습에서 LLM의 성능 및 다양한 프롬프트 전략 비교. 또한, LLM이 예측에 과신하는 경향을 보여줍니다.

- **Performance Highlights**: 연구 결과, LLM들은 지름길을 활용할 때 성능이 현저히 떨어지며 (최대 40% 이상), 큰 LLM이 zero-shot 및 few-shot ICL 프롬프트 하에서 지름길을 더 많이 사용합니다. Chain-of-Thought prompting이 지름길 의존도를 줄이는 데 효과적이며, LLM의 일반 설명 품질이 낮은 것도 확인되었습니다.



### Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieva (https://arxiv.org/abs/2410.13339)
Comments:
          6 figures, 13 tables

- **What's New**: 이번 논문에서 제안한 Probing-RAG은 언어 모델의 중간 계층에서의 숨겨진 상태 표현을 활용하여 추가적인 정보 검색의 필요성을 적응적으로 판단합니다.

- **Technical Details**: Probing-RAG는 언어 모델의 내부 인지 구조를 포착하는 발사기(prober)를 활용하여 쿼리에 대한 추가 검색 단계를 결정합니다. 이 발사기는 5MB의 파라미터만 가지고 있으며, 기존의 외부 분류 기반 방법들에 비해 2000배 작습니다. 실험은 5개의 오픈 도메인 QA 데이터셋에서 이루어졌습니다.

- **Performance Highlights**: Probing-RAG는 기존 방법들에 비해 성능이 뛰어난 것으로 나타났으며, 평균적으로 검색 빈도를 약 50% 줄였습니다.



### Do LLMs Have Political Correctness? Analyzing Ethical Biases and Jailbreak Vulnerabilities in AI Systems (https://arxiv.org/abs/2410.13334)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)에서 안전성을 위해 주입된 의도적 편향이 어떻게 유해한 콘텐츠 생성을 초래할 수 있는지를 탐구하고, 새로운 패러다임인 PCJailbreak와 간단한 방어 전략인 PCDefense를 제안합니다.

- **Technical Details**: LLM의 안전성을 확보하기 위한 다양한 방법(데이터 필터링, 지도 학습 피드백 등)은 보통 의도적인 편향을 불러오며, 이는 'jailbreak' 현상을 초래합니다. PCJailbreak는 이러한 편향을 이용하여 LLM이 유해한 출력을 생성할 수 있도록 조작하는 공격이며, PCDefense는 공격 전에 방어 프롬프트를 주입하여 이를 방지하는 방법입니다.

- **Performance Highlights**: PCJailbreak는 최신 GPT 모델에서도 효과적이며, 제안하는 PCDefense 방법은 추가적인 추론 비용 없이도 jailbreak 공격을 완화할 수 있음을_showcase합니다. 이 연구는 LLM 제공업체들이 보다 책임감 있게 안전성을 설계하고 구현해야 함을 강조합니다.



### Fine-Tuning Language Models on Multiple Datasets for Citation Intention Classification (https://arxiv.org/abs/2410.13332)
Comments:
          To be appear as a Findings paper at EMNLP 2024

- **What's New**: 본 논문에서는 Citation intention Classification (CIC) 도구에 대한 다중 작업 학습(Multi-task Learning, MTL) 프레임워크를 제안합니다. 이 프레임워크는 주 데이터셋과 함께 여러 보조 CIC 데이터셋에 대한 PLMs의 공동 미세 조정을 통해 성능을 향상시킵니다.

- **Technical Details**: 제안된 MTL 프레임워크는 데이터 기반 작업 관계 학습(Task Relation Learning, TRL) 방법을 통해 보조 데이터셋의 기여도를 조절합니다. 이는 부정적 전이를 방지하고 하이퍼파라미터 조정 비용을 절감합니다. 자료의 위치가 CIC 작업에서 유용한 정보를 제공하며, 위치 인식 집계 기능이 PLM 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크로 미세 조정된 PLMs는 작은 데이터셋에서 기존 최첨단 모델보다 7%에서 11% 향상된 성능을 보여주며, 큰 데이터셋에서는 최상의 모델과 일치합니다. KIM이라는 새로운 벤치마크 데이터셋도 소개됩니다.



### Computational Approaches to Arabic-English Code-Switching (https://arxiv.org/abs/2410.13318)
Comments:
          PhD thesis

- **What's New**: 이번 연구는 아랍어와 영어 간의 코드 스위칭(Code-Switching) 데이터를 활용하여 주어진 분야의 여러 NLP(Natural Language Processing) 작업, 특히 Named Entity Recognition(NER) 작업의 문제를 다루고 있습니다. 연구는 특히 현대 표준 아랍어와 아랍어-영어 NER을 위한 첫 번째 주석이 달린 코드 스위치 아랍어-영어 코퍼스를 제작하여 이를 진행했습니다.

- **Technical Details**: 본 연구는 CS 데이터에서 NER 태거(NER tagger)를 개선하기 위해 CS 컨텍스트 임베딩(Contextual Embeddings)과 데이터 증강(Data Augmentation) 기법을 도입했습니다. 또한, 혼합 텍스트의 언어 유형을 결정하고 명명된 개체(named entity)를 식별하기 위한 여러 개의 intra-word 언어 식별 접근 방식을 제안합니다.

- **Performance Highlights**: 모든 방법이 CS 데이터에서 NER 태거의 성능을 향상시키는 결과를 보였습니다. 연구의 결과는 아랍어-영어 코드 스위칭 데이터에서 다루어진 여러 NLP 작업의 중요성과 필요성을 강조합니다.



### Mitigating Biases to Embrace Diversity: A Comprehensive Annotation Benchmark for Toxic Languag (https://arxiv.org/abs/2410.13313)
Comments:
          12 pages, 9 figures, EMNLP-NLP4DH 2024

- **What's New**:  이 연구는 인문학 연구에 기반한 처방적 주석 벤치마크를 도입하여 비공식적이고 비주류 언어 사용에 대한 일관되고 편향 없는 공격성 언어의 라벨링을 보장합니다. 또한 두 개의 새로 주석이 달린 데이터셋을 제공합니다.

- **Technical Details**:  연구에서는 공격성 언어 데이터 라벨링을 위한 새로운 기준을 제안하고, LLM(대형 언어 모델)을 사용하여 전문 주석자가 부족할 경우의 대안을 제시합니다. 연구는 다수의 주석자 간 일치성을 높이는 방법과 잘 정립된 지침이 주관적인 변동성을 줄이는 데 어떻게 기여하는지를 보여줍니다.

- **Performance Highlights**:  소규모 모델이 여러 출처에서 주석 된 데이터로 미세 조정되었을 때, 단일 대규모 인간 주석 데이터 세트에서 훈련된 모델보다 더 나은 성능을 보였습니다. 이는 제한된 데이터와 다양한 언어 유형에 대해서도 우수한 성능을 유지시켜 주는 구조화된 지침의 가치를 부각합니다.



### Reference-Based Post-OCR Processing with LLM for Diacritic Languages (https://arxiv.org/abs/2410.13305)
- **What's New**: 이 연구는 역사적 문서에서 OCR(Optical Character Recognition)로 생성된 부정확한 텍스트를 수정하는 새로운 방법을 제안합니다. 이는 사용 가능한 전자책(e-books)을 참조하여 텍스트를 교정하고 대형 언어 모델(LLM)을 사용하여 고정밀의 유사 페이지 간 레이블을 생성하는 방식입니다.

- **Technical Details**: 이 방법은 먼저 고유한 노이즈를 제거한 후, 콘텐츠 중심의 전자책을 참조 자료로 활용하여 LLM을 통해 시맨틱과 비주얼 유사성을 기반으로 OCR로 생성된 결함 있는 텍스트를 수정합니다. 마지막으로, 짧은 길이의 빈 텍스트는 LLM 기반의 철자 교정으로 조정하여 최종 유사 레이블을 제공합니다.

- **Performance Highlights**: 시험 결과, 이 연구에서 생성한 데이터셋은 10점 만점에 평균 8.72점을 기록하며 최신 Transformer 기반의 철자 교정 모델인 7.03점을 초과했습니다. 본 연구는 또한 19세기 고전 책을 위한 최초의 대규모 공개 데이터셋인 VieBookRead를 출시하여 향후 연구를 지원할 계획입니다.



### Advancing Large Language Model Attribution through Self-Improving (https://arxiv.org/abs/2410.13298)
Comments:
          Accepted by EMNLP 2024 Main Conference

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에게 증거 출처를 인용하는 텍스트 생성을 가르쳐 신뢰성을 높이는 새로운 프레임워크인 START(Self-Taught AttRibuTion)를 소개합니다. 이는 인력 자원을 절약하면서도 LLM의 인용 능력을 개선합니다.

- **Technical Details**: START는 모델이 초기의 불충분한 감독 신호로 인해 정체되는 것을 막기 위해 스스로 합성 훈련 데이터를 생성하도록 유도합니다. 이후 모델의 인용 능력을 자가 개선하기 위해 선택한 응답으로부터 세부적인 선호 감독 신호를 반복적으로 활용합니다.

- **Performance Highlights**: 세 개의 오픈 도메인 질문-응답 데이터셋에서 실험을 수행한 결과, 평균 25.13%의 성능 향상을 이루었으며, 이는 인간 주석이나 더 진보된 모델에 의존하지 않고 달성되었습니다. 또한, START는 여러 출처에서 정보를 집계하는 능력이 뛰어난 것으로 나타났습니다.



### Learning to Route with Confidence Tokens (https://arxiv.org/abs/2410.13284)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)에서 신뢰성을 판단하고 이를 기반으로 결과를 개선하기 위한 새로운 방법론인 Self-REF(Self-Reflection)를 제안합니다. Self-REF는 LLM이 자신의 예측에 대한 신뢰도를 효과적으로 평가할 수 있도록 훈련하는 경량화된 방법론입니다.

- **Technical Details**: Self-REF는 세 가지 주요 단계를 포함합니다: (i) 신뢰도 토큰 주석 추가, (ii) Self-REF 파인튜닝, (iii) 신뢰도 점수 추출. 신뢰도 토큰은 LLM이 올바르게 응답했는지를 기준으로 생성되며, 이러한 토큰에서 신뢰도 점수를 계산하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: Self-REF는 라우팅 및 거부 학습 작업에서 기존 방법들보다 뛰어난 성능을 보이며, 특히 네 개의 공개 데이터셋에서 우수한 결과를 나타냈습니다. 이는 LLM이 낮은 신뢰도 질문을 더 강력한 LLM으로 라우팅하거나 안전한 행동으로 자신의 대답을 거부하는 데 기여합니다.



### BANTH: A Multi-label Hate Speech Detection Dataset for Transliterated Bangla (https://arxiv.org/abs/2410.13281)
- **What's New**: 이번 연구에서는 Bangla 언어로 변환된 증오 발언을 다중 레이블로 분류하기 위한 BanTH 데이터셋을 처음으로 소개합니다. 이는 37,350개의 샘플로 구성되어 있으며, YouTube 댓글을 통해 수집되었습니다.

- **Technical Details**: BanTH 데이터셋은 Hate 또는 Non-Hate 이진 레이블로 분류된 후, 정치적, 종교적, 성별, 개인 공격, 폭력, 출신, 신체 비하와 같은 다중 레이블로 재분류됩니다. 우리는 변환자(Transformer) 기반의 인코더를 사용하여, 변환된 Bangla 텍스트에 대한 추가 사전 훈련을 진행하여 성능을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 추가 사전 훈련된 인코더가 BanTH 데이터셋에서 최첨단 성능을 달성했으며, 번역 기반의 LLM 프롬프팅 전략이 제로샷(Zero-shot) 설정에서 다른 전략들을 능가했습니다.



### SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs (https://arxiv.org/abs/2410.13276)
- **What's New**: 이 논문은 Attention sparsity를 미리 정의되지 않고 학습 가능하다는 점을 강조하며, SeerAttention이라는 새로운 Attention 메커니즘을 소개합니다. 이는 기존의 Attention에 학습 가능한 게이트를 추가하여 중요 블록을 선택하고 불필요한 블록을 스파스(sparse)하게 xử lý합니다.

- **Technical Details**: SeerAttention은 Q(쿼리)와 K(키) 입력을 자동으로 풀링하여 학습 가능한 게이트로 처리하여 중요한 블록을 انتخاب합니다. 이를 통해 하위 블록 스파스 Attention 커널이 불필요한 블록을 건너뛰면서 효율성을 극대화합니다. FlashAttention 커널을 커스터마이즈하여 블록 수준의 Attention 맵 정보를 최소한의 오버헤드로 추출하는 기술을 고안하였습니다.

- **Performance Highlights**: SeerAttention은 후속 학습 단계에서 기존의 스태틱(static) 또는 휴리스틱 기반의 스파스 Attention 방법보다 크게 향상된 성능을 보여주며, 다양한 컨텍스트 길이 및 스파시티 비율에 적응할 수 있는 뛰어난 유연성을 가지고 있습니다. 32k 컨텍스트 길이에서 90% 스파시티 비율을 달성하며, FlashAttention-2에 비해 5.67배의 속도 향상을 제공합니다.



### Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning (https://arxiv.org/abs/2410.13274)
Comments:
          16 pages, 5 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 다중 단계(Multi-Hop) 지식을 효과적으로 삭제할 수 있는 방법을 탐구하며, 기존의 unlearning 기술들의 한계를 지적합니다.

- **Technical Details**: 본 논문에서는 다중 단계 쿼리를 서브 질문(Subquestions)으로 분해하고, 학습되지 않은 모델의 불확실성(Uncertainty)을 활용하여 의사 결정을 지원하는 MUNCH라는 새로운 방법론을 제안합니다.

- **Performance Highlights**: MUNCH는 기존의 방법들에 비해 다중 단계 지식의 unlearning 성능이 크게 향상됨을 보여주며, 추가적인 훈련 없이도 기존의 unlearning 기술들과 통합이 용이합니다.



### Roadmap towards Superhuman Speech Understanding using Large Language Models (https://arxiv.org/abs/2410.13268)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성공이 음성 및 오디오 데이터를 통합하려는 노력을 촉진시켰다고 강조합니다. 특히, GPT-4o와 같은 최신 모델을 통해 비언어적 정보와 세계 지식을 보존하며 깊은 음성 이해를 가능하게 하는 엔드 투 엔드(end-to-end) 음성 LLMs의 잠재력을 제시합니다.

- **Technical Details**: 논문에서는 음성 LLM을 개발하기 위한 다섯 단계의 로드맵을 제안합니다. 이 단계는 기본 자동 음성 인식(ASR)에서 시작하여 비언어적 정보와 추상적 음향 지식을 통합하는 고급 초인간(superhuman) 모델에 이르는 과정을 포함합니다. 또한, SAGI Benchmark라는 기준을 설계하여 각 단계에 걸쳐 다양한 작업을 평가할 수 있는 표준화된 방법을 제공합니다.

- **Performance Highlights**: 인간은 1단계에서 3단계까지의 작업에서 일반적으로 강한 성능을 보였지만, 상위 단계에서는 추상적 음향 지식 부족으로 인해 성능이 제한적이었습니다. 비록 현재의 음성 LLM들이 일부 영역에서 인간을 초월할 수 있는 능력이 있지만, 작업의 다양성과 포괄성 면에서 여전히 부족함을 드러냈습니다.



### From Babbling to Fluency: Evaluating the Evolution of Language Models in Terms of Human Language Acquisition (https://arxiv.org/abs/2410.13259)
- **What's New**: 본 논문은 언어 모델(LM)의 언어 능력을 인지 언어 습득의 관점에서 비판적으로 분석하며, 고전적인 언어 발달 이론을 바탕으로 언어 모델의 생성 능력을 평가하기 위한 3단계 프레임워크를 제안합니다.

- **Technical Details**: 제안된 3단계 프레임워크는 기본 단어 이해에서 복잡한 문법 및 논리적 추론까지의 능력을 평가합니다. 이 연구에서는 2019년부터 2024년까지의 15개의 언어 모델을 평가하며, 레지스터 이론(register theory)이 모델 능력에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 최근 언어 모델들이 전반적인 성능에서 이전 모델들을 능가했지만, 그 발전 경로는 인간의 언어 습득과 일치하지 않음을 보였습니다. 특히, 문장 구조와 보조 동사와 같은 특정 분야에서는 큰 발전이 없었습니다.



### A Systematic Investigation of Knowledge Retrieval and Selection for Retrieval Augmented Generation (https://arxiv.org/abs/2410.13258)
- **What's New**: 이번 연구에서는 Retrieval-augmented generation (RAG) 시스템에서 지식 검색(knowledge retrieval)과 지식 선택(knowledge selection)이 생성(performance) 결과에 미치는 영향을 체계적으로 분석합니다. 연구 결과, 지식 검색의 질이 생성 품질에 커다란 영향을 미친다는 것을 보여주고, 특정 조건에서는 지식 선택의 역할이 제한적이라는 것을 밝혀냈습니다.

- **Technical Details**: RAG는 세 가지 단계로 구성됩니다: (1) Knowledge retrieval, (2) 선택적 Knowledge selection, (3) 생성(generation). 이 과정에서 K는 검색된 지식 세트, K'는 선택된 지식입니다. 연구는 다양한 비율의 금(gold) 지식과 방해(distraction) 지식을 블렌딩하여 실험합니다.

- **Performance Highlights**: 연구에 따르면, 강력한 생성 모델을 사용할 경우 지식 검색 소득(knowledge recall score)을 향상시키는 것이 중요하며, 약한 생성 모델이나 모호한 작업에서는 지식 F1 점수가 전체 성능 향상에 결정적인 요소가 됩니다.



### Automatic Translation Alignment Pipeline for Multilingual Digital Editions of Literary Works (https://arxiv.org/abs/2410.13255)
Comments:
          18 pages, Computational Humanities Research Conference, December 4-6, 2024, Aarhus, Denmark

- **What's New**: 이 논문은 Alessandro Manzoni의 소설 "I promessi sposi"의 다국어 디지털 에디션(Multilingual Digital Edition, MDE) 제작을 위한 번역 정렬 알고리즘의 적용을 조사합니다. 19세기와 20세기의 8개 언어(영어, 스페인어, 프랑스어, 독일어, 네덜란드어, 폴란드어, 러시아어, 중국어) 번역을 포함하여 MDE의 주요 요구 사항을 식별하고, 문학 텍스트 번역에 대한 현재 알고리즘의 한계를 강조하며, MDE 생성을 위한 자동화된 파이프라인을 제안합니다.

- **Technical Details**: 이 연구에서는 문학 작품의 다국어 디지털 에디션을 제작하기 위해 최신 정렬 기법을 적용하는 자동 번역 정렬 파이프라인을 제안합니다. 이 파이프라인은 원문과 번역 텍스트의 나란히 배치된 웹 기반 표현으로 변환되며, 텍스트 조각을 관리 가능한 길이로 정렬하여 사용자에게 독서 및 분석에 용이하도록 합니다. 또한, 문학 번역의 정렬을 평가하기 위한 새로운 메트릭스를 제안하고 있습니다.

- **Performance Highlights**: 논문에서 제안한 정렬 메트릭스는 기존 정렬 알고리즘의 성능을 보다 포괄적으로 평가할 수 있게 하며, 문학 작품의 다국어 디지털 에디션 제작 시 독자의 집중력과 이해를 증진할 수 있는 방법들을 제시합니다. 이는 번역의 삽입 및 생략된 부분을 시각적으로 강조하여 사용자로 하여금 각 번역의 뉘앙스를 파악할 수 있도록 돕습니다.



### Atomic Calibration of LLMs in Long-Form Generations (https://arxiv.org/abs/2410.13246)
- **What's New**: 본 논문에서는 LLM(대규모 언어 모델)의 신뢰성을 높이기 위한 새로운 접근법인 atomic calibration을 제안합니다. 기존의 macro calibration은 주로 짧은 응답에 대한 신뢰도를 평가하는 데 초점을 맞추었으나, 긴 응답의 경우 더욱 복잡한 진술이 포함될 수 있어 적합하지 않다는 점을 강조하였습니다.

- **Technical Details**: atomic calibration은 긴 응답을 작은 단위인 atomic claims로 분해하여 세부적인 신뢰도를 평가합니다. 본 연구에서는 LLM의 신뢰성 추정 방법을 discriminative와 generative 유형으로 나누고, 이들의 조합이 calibration을 개선할 수 있음을 보여줍니다. 또한, 7종의 LLM과 3개의 데이터셋을 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: atomic calibration은 긴 형식의 생성 과정에서도 잘 작동하며, macro calibration 결과를 개선할 수 있는 것으로 나타났습니다. 이 방법은 LLM의 생성 과정에서 신뢰성과 calibration 변화의 패턴을 심도 있게 분석할 수 있는 가능성을 열어줍니다.



### Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis (https://arxiv.org/abs/2410.13237)
Comments:
          17 pages, 6 figures, 14 tables

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)에서 발생하는 'Language Confusion' 현상을 분석하고, 이를 정량화하기 위한 새로운 측정 기준인 'Language Confusion Entropy'를 제안합니다. 이는 다양한 언어 분포의 패턴을 탐구하며, LLM 보안과의 연관성도 밝혔습니다.

- **Technical Details**: Language Confusion Entropy는 LLM에서 발생하는 언어 혼란 정도를 정량화하는 지표로, 언어 유형론에 기반한 언어 분포를 사용하여 LLM이 혼란스러울 때의 양상을 포착합니다. 이 연구는 여러 언어 간의 의미적 유사성과 LLM의 취약성을 연결지으며, 다국어 임베딩 역전 공격(multilingual embedding inversion attacks)에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 언어 유형론을 기반으로 분석한 패턴들이 언어 혼란과 연관되어 있음을 발견했습니다. 특히 자원이 적은 언어는 혼란이 덜 발생하며, 다양한 스크립트와 언어 계열을 아우르는 훈련이 언어 혼란을 보다 효과적으로 완화할 수 있다는 결과를 도출했습니다.



### SPIN: Self-Supervised Prompt INjection (https://arxiv.org/abs/2410.13236)
- **What's New**: 이번 논문에서는 Self-supervised Prompt INjection (SPIN)라는 새로운 방어 메커니즘을 도입하여, 다양한 adversarial 공격 및 jailbreak 공격에 대해 LLMs의 안전성을 향상시키는 방법을 제시합니다. SPIN은 추론 시간(inference-time)에서 이루어지므로 기존의 안전성 정렬과 호환되며 추가적인 안전성 레이어를 제공합니다.

- **Technical Details**: SPIN은 self-supervised learning을 기반으로 하여, 공격을 탐지하고 입력을 복구하는 방어 기법입니다. LLM의 자연 가이드라인을 무효화하는 프롬프트가 모델의 다른 능력도 저하시키기 때문에, 이를 이용해 공격을 탐지할 수 있습니다. 방어 메커니즘은 기존의 방어 시스템과의 호환성이 있으며, 악의적 또는 선의적 레이블에 의존하지 않고 재빠른 시점에서 온라인으로 사용할 수 있습니다.

- **Performance Highlights**: SPIN의 적용 결과, Attack Success Rate (ASR)를 최대 87.9%까지 감소시킬 수 있었으며, benign 사용자 요청에 대한 성능을 유지했습니다. Advbench에서 Universal Adversarial Triggers를 사용한 실험 결과, Vicuna 모델에서는 ASR이 12.11%, Llama-2 모델에서는 0%로 감소하여 두 모델을 완전히 보호했습니다. 또한, 공격자들이 방어 체계를 알고 있어도 여전히 강인성을 보였습니다.



### Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation (https://arxiv.org/abs/2410.13232)
Comments:
          Work in progress

- **What's New**: 최근 대규모 언어 모델(LLMs)을 기반으로 한 웹 에이전트의 성능이 긴 시간의 작업에서 최적이 아님을 강조하며, '세계 모델'(world model)의 부재를 확인했다. 이 논문은 행동의 결과를 시뮬레이션하여 더 나은 의사 결정을 할 수 있는 세계 모델 증강 웹 에이전트(WMA)를 제안한다.

- **Technical Details**: 제안된 WMA 웹 에이전트는 정책 모델에서 도출된 행동 후보의 결과를 시뮬레이션하고 각 행동의 보상을 추정하는 가치 함수를 사용하여 최종 행동을 선택한다. 이를 위해 접합 중심 관찰 추상화(transition-focused observation abstraction)를 도입하여 주요 상태 차이를 강조하는 자유형 자연어 설명을 생성한다.

- **Performance Highlights**: WebArena 및 Mind2Web 실험에서 제안된 세계 모델을 통해 LLM 기반 에이전트의 정책 선택이 향상되었음을 보이며, 최근 트리 탐색 에이전트에 비해 각각 6.8배와 5.3배의 비용 및 시간 효율성을 입증했다.



### Proof Flow: Preliminary Study on Generative Flow Network Language Model Tuning for Formal Reasoning (https://arxiv.org/abs/2410.13224)
- **What's New**: 이 논문에서는 Generative Flow Networks(GFlowNets)를 이용해 LLM(대형 언어 모델)의 고급 추론 능력을 향상시키는 기법을 소개합니다. 기존의 시스템 2 사고 체계를 활용하여 더 복잡한 문제를 해결하기 위한 새로운 접근법을 제시합니다.

- **Technical Details**: GFlowNets는 최대 엔트로피 강화 학습(maximum entropy reinforcement learning) 알고리즘으로, 보상에 비례하여 구성적인 객체를 샘플링하는 정책을 훈련하도록 설계되었습니다. 이 연구에서는 GFlowNet을 Lean(형식 수학 언어) 환경과 통합하여 신경 정리 증명(Neural Theorem Proving, NTP)에서의 증명 검색(task) 문제를 해결합니다.

- **Performance Highlights**: 예비 실험 결과에 따르면, GFlowNet fine-tuning은 신경 정리 증명 작업에서 모델의 탐색성과 추론 능력을 개선하여 증명 검색 성능을 향상시키는 잠재력을 보여주었습니다. 기존 모델보다 더 높은 해결 비율이 관찰되었습니다.



### CBT-Bench: Evaluating Large Language Models on Assisting Cognitive Behavior Therapy (https://arxiv.org/abs/2410.13218)
- **What's New**: 본 논문은 지금의 정신 건강 지원에서 환자의 필요와 제공 가능한 지원 간의 큰 격차를 해결하기 위한 접근법으로서, 대형 언어 모델(LLMs)을 전문적인 심리 치료에 활용하는 가능성을 깊이 조사합니다. 특히, 우리는 인지 행동 치료(CBT) 지원의 체계적 평가를 위한 새로운 벤치마크인 CBT-BENCH를 제안합니다.

- **Technical Details**: CBT-BENCH는 세 가지 수준의 과제로 구성됩니다: I: 기본 CBT 지식 습득을 위한 다중 선택 질문; II: 인지 모델 이해를 위한 인지 왜곡 분류, 주요 핵심 신념 분류, 세부 핵심 신념 분류 작업; III: CBT 세션에서 환자 발화에 대한 치료적 응답 생성. 논문에서는 CBT의 핵심 측면을 AI 지원을 통해 향상할 수 있는 가능성을 조명하며, 각 과제는 기본 지식 암기부터 실제 치료 대화에 참여하는 것과 같은 복잡한 능력 요구 사항의 계층을 포함합니다.

- **Performance Highlights**: 실험 결과에 따르면 LLMs는 CBT 지식을 암기하는 데는 상대적으로 잘 수행하였지만, 환자의 인지 구조에 대한 깊은 분석이 필요한 복잡한 실제 시나리오에서는 부족한 성과를 보였습니다. LLMs는 일반적으로 엄격한 논리적 추론 프로세스를 따르지만, 치료에서 중요한 환자의 관점에서 사고하고 관계를 구축하는 능력이 부족하다는 제한점을 보여주었습니다.



### FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs (https://arxiv.org/abs/2410.13210)
- **What's New**: 이 논문에서는 10개의 현대 LLM(대형 언어 모델)와 8개의 모델 가족에서 발생하는 도전적 환각을 포함하는 FaithBench라는 환각 평가 벤치마크를 제안합니다.

- **Technical Details**: FaithBench는 인공지능 모델에 의해 생성된 요약에서 발생하는 환각(hallucination)을 평가하기 위해 설계되었습니다. 이 벤치마크는 LLM 가족에 따라 다양한 환각 사례를 포함하고 있으며, 각 요약은 인간 전문가에 의해 주석이 달린 ground truth를 포함하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4o와 GPT-3.5-Turbo가 가장 낮은 환각 비율을 나타냈지만, 환각 탐지 모델의 정확도는 여전히 50%에 가까운 수치를 기록하여 개선의 여지가 많음을 시사합니다.



### BQA: Body Language Question Answering Dataset for Video Large Language Models (https://arxiv.org/abs/2410.13206)
- **What's New**: 이 논문은 인간의 신체 언어를 정확하게 해석하기 위해 Video Large Language Models (VideoLLMs)에 필요한 새로운 데이터셋인 BQA (Body Language Question Answering) 를 제안합니다. 이 데이터셋은 26가지 감정 레이블을 가진 신체 언어 비디오에서 감정을 해석하는 능력을 평가합니다.

- **Technical Details**: BQA 데이터셋은 5-10초 길이의 7632개의 짧은 비디오 클립으로 구성되어 있으며, 각 클립은 젠더(gender), 나이(age), 민족(ethnicity) 메타데이터와 함께 26가지 감정 레이블을 포함합니다. 본 연구는 다양한 VideoLLMs 모델의 성능을 평가하여 신체 언어 해석의 어려움을 드러냅니다.

- **Performance Highlights**: BQA를 통해 평가한 결과, VideoLLMs 모델들은 특정 나이 그룹이나 민족에 따라 편향된 답변을 제공하는 경향이 있음을 발견했습니다. 특히 GPT-4o와 Gemini 모델이 높은 정확도를 기록했으며, 신체 언어 해석 능력에서 다른 모델들보다 우수한 성능을 보였습니다.



### Measuring Free-Form Decision-Making Inconsistency of Language Models in Military Crisis Simulations (https://arxiv.org/abs/2410.13204)
- **What's New**: 이번 연구는 언어 모델(LMs)이 고위험 결정-making 환경에서 일관성이 부족한 응답을 나타내는지를 조사합니다. 특히 군사 위기 시뮬레이션에서 LMs가 생성하는 자유형(free-form) 응답을 분석하여 군사적 결정-making에 대한 신뢰성 문제를 강조합니다.

- **Technical Details**: 이 연구에서는 BERTScore를 기반으로 하여 LMs의 응답 일관성을 정량적으로 측정합니다. 실험 결과, 모든 LMs는 세부적인 상황 조정에도 불구하고 의미적으로 상이한 응답을 제공하며, 이는 높은 수준의 일관성 결여를 나타냅니다. 또한, 프롬프트의 민감도 변주가 일관성에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 군사적인 고위험 환경에서 LMs의 의사결정 신뢰성을 고민할 때, 이러한 연구 결과가 더욱 중요해집니다. LMs는 상황에 따라 서로 다른 일관성 수준을 보이며, 이로 인해 고위험 결정-making에서 신중한 접근이 필요하다는 결론을 도출했습니다.



### Meta-DiffuB: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration (https://arxiv.org/abs/2410.13201)
- **What's New**: Meta-DiffuB는 Seq2Seq 텍스트 생성을 위한 새로운 스케줄러-탐색자 모델을 도입하여 기존 S2S-Diffusion 모델의 한계를 극복합니다. 기존 모델들은 고정된 또는 수작업으로 만든 규칙에 의존하여 노이즈를 스케줄링하는 반면, Meta-DiffuB는 문맥화된 노이즈 스케줄링을 통해 문장별로 적합한 노이즈를 적용합니다.

- **Technical Details**: Meta-DiffuB는 두 가지 모델로 구성됩니다: 스케줄러와 탐색자. 스케줄러는 각 문장의 특성에 맞춰 적절한 수준의 노이즈를 스케줄링하고, 탐색자는 해당 노이즈를 활용하여 업데이트 및 생성을 수행합니다. 이 접근 방식은 자연어 처리(NLP)에서 Seq2Seq 작업의 의미론적 특성을 반영합니다.

- **Performance Highlights**: Meta-DiffuB는 네 가지 Seq2Seq 벤치마크 데이터세트에서 기존 S2S-Diffusion 모델 및 정밀 조정된 사전 훈련된 언어 모델(PLMs)과 비교하여 최첨단 성능을 달성합니다. 또한, 스케줄러 모델은 기존 DiffuSeq를 더욱 향상시키기 위한 '플러그 앤 플레이' 기능을 제공합니다.



### The Geometry of Numerical Reasoning: Language Models Compare Numeric Properties in Linear Subspaces (https://arxiv.org/abs/2410.13194)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)이 논리적 비교 질문에 답변할 때 저차원(subspace) 임베딩 공간에 인코딩된 숫자 속성을 어떻게 활용하는지 조사하였습니다. 특히, LLM이 수치적 추론(numerical reasoning) 문제를 해결하기 위해 이러한 선형 서브스페이스를 활용한다는 점을 강조합니다.

- **Technical Details**: 본 연구에서는 부분최소제곱회귀(partial least squares regression, PLS)를 사용하여 LLM의 내부에서 숫자 속성과 관련된 서브스페이스를 식별하였으며, 이 서브스페이스에개입(intervention)을 통해 숨겨진 상태(hidden state)를 조작하여 LLM의 비교 결과를 변경하는 실험을 진행하였습니다. 연구에서는 LLM이 학습한 수치적 속성이 비선형이 아닌 선형적으로 표현된 정보를 활용한다는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, LLM은 출생 연도(birth year), 사망 연도(death year), 위도(latitude) 등 3가지 숫자 속성에 대해 다양한 수치 추론 작업을 해결하는 데 성공적인 성과를 보였습니다. 모델은 출생/사망 연도 예측에서 약 75%의 정확도를 보였으나, 위도 관련 질문에서는 56%의 정확도로 상대적으로 낮은 성과를 나타냈습니다.



### Evaluating Self-Generated Documents for Enhancing Retrieval-Augmented Generation with Large Language Models (https://arxiv.org/abs/2410.13192)
Comments:
          Under Review

- **What's New**: 이번 연구는 Self-Generated Documents (SGDs)와 검색된 콘텐츠의 통합이 대규모 언어 모델(LLM)의 성능 향상에 어떻게 기여하는지를 탐구합니다. 이는 SGDs의 고유한 특성을 중점적으로 분석하는 첫 번째 연구로, 이를 통해 SGD의 다양한 유형이 성능에 미치는 영향을 비교하고자 합니다.

- **Technical Details**: 연구에서는 Systemic Functional Linguistics (SFL)에 기반한 SGDs의 분류 체계를 개발하고, 다양한 지식 집약적(task) 작업에 대한 실험을 통해 서로 다른 SGD 범주의 영향력을 평가합니다. 이를 통해 LLM 성능 개선에 가장 효과적인 SGD 유형을 도출합니다.

- **Performance Highlights**: 연구 결과와 SGD 범주에 따라 개발된 추가 융합(fusion) 방법은 RAG( retrieval-augmented generation) 기반의 지식 기반 QA 작업에서 SGDs를 효과적으로 활용하기 위한 실용적인 지침을 제공합니다.



### MCQG-SRefine: Multiple Choice Question Generation and Evaluation with Iterative Self-Critique, Correction, and Comparison Feedback (https://arxiv.org/abs/2410.13191)
Comments:
          Equal contribution for the first two authors

- **What's New**: 이번 연구에서는 MCQG-SRefine라는 새로운 프레임워크를 제안하여, 전문의 시험을 위한 고품질 다지선다 질문(USMLE 스타일 질문)을 자동 생성하는 방법을 소개합니다. 이 프레임워크는 LLM의 자기 수정(self-refine) 기반으로, 전문가의 피드백과 반복적인 자기 비판을 통해 질문의 품질과 난이도가 향상됩니다.

- **Technical Details**: MCQG-SRefine는 의료 사례를 입력으로 받아 USMLE 스타일의 질문을 생성합니다. FR 쿼리 설정과 41개의 주요 주제를 포함한 체크리스트를 기반으로, LLM이 의료 사례에서 정보를 추출하여 질문을 생성합니다. 또한, LLM 스스로 피드백을 주고, 이를 기반으로 질문을 수정하는 세 가지 단계(S1: 초기 MCQ 생성, S2: 비판 피드백, S3: 수정 피드백)를 따릅니다.

- **Performance Highlights**: MCQG-SRefine를 통해 생성된 질문은 GPT-4가 생성한 질문보다 72.5%의 선호도를 기록했으며, 더 높은 난이도의 질문을 생성하는 것이 확인되었습니다. 쉽고 중간 수준의 질문에서 각각 80% 감소 및 2.25배 증가, 어려운 질문에서 4배 증가하는 결과를 보였습니다. LLM-as-Judge를 활용해 전문가 평가를 대체할 수 있는 신뢰성 있는 자동 평가 시스템 또한 제안되었습니다.



### aiXcoder-7B: A Lightweight and Effective Large Language Model for Code Completion (https://arxiv.org/abs/2410.13187)
Comments:
          aiXcoder-7B is available at this https URL

- **What's New**: aiXcoder-7B는 70억 개의 매개변수를 가지며, 코드 완성을 위하여 설계된 경량화된 대형 언어 모델(LLM)입니다. 기존 LLM에 비해 보다 높은 정확도를 기록하며, 개발자 생산성을 높이기 위해 응답 시간을 단축시킵니다.

- **Technical Details**: aiXcoder-7B는 세 가지 주요 요소에 의해 우수한 성능을 발휘합니다: (1) 다중 목표 훈련(Multi-objective training), (2) 다양한 데이터 샘플링 전략(Diverse data sampling), (3) 방대한 고품질 데이터(Extensive high-quality data). 특히, Structured Fill-In-the-Middle (SFIM)이라는 훈련 목표를 사용하여 코드의 구문 구조를 고려합니다. 이와 함께 1.2 조 개의 고유한 토큰을 소비하여 훈련됩니다.

- **Performance Highlights**: aiXcoder-7B는 6개의 코드 완성 벤치마크에서 최신 LLM들보다 우수한 성능을 나타내며, 심지어 StarCoder2-15B와 CodeLlama-34B와 같은 더 큰 LLM보다도 뛰어난 결과를 기록하였습니다. 이는 aiXcoder-7B가 경량화된 모델임에도 불구하고 뛰어난 코드 완성 정확도를 보유하고 있음을 나타냅니다.



### Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers (https://arxiv.org/abs/2410.13184)
- **What's New**: 이번 논문에서는 Mixture of Depths (MoD) 모델의 한계점을 극복하기 위해 Router-Tuning과 Attention with Dynamic Depths (MindSkip)를 제안하며, 훈련 비용 절감과 성능 유지 간의 균형을 맞추는 방법을 다룹니다.

- **Technical Details**: Router-Tuning은 라우터 네트워크만 조정하는 방법으로, 전체 모델의 매개변수를 업데이트하지 않고도 효율적으로 훈련할 수 있습니다. MindSkip는 Attention 계층에 동적 깊이를 적용하여 성능 손실 없이 연산 비용과 메모리 사용량을 줄입니다.

- **Performance Highlights**: 심층 실험 결과, 제안된 방법은 21%의 속도 향상과 0.2%의 성능 저하율을 기록하였으며, Nvidia RTX A6000에서 30분 이내에 Router-Tuning이 완료됩니다.



### AdaSwitch: Adaptive Switching between Small and Large Agents for Effective Cloud-Local Collaborative Learning (https://arxiv.org/abs/2410.13181)
Comments:
          EMNLP 2024 Main Conference

- **What's New**: AdaSwitch라는 새로운 LLM 활용 패러다임을 제안하여 클라우드 기반의 대형 LLM과 로컬 배치된 소형 LLM이 협력하여 복잡한 작업을 해결한다. 이 프레임워크는 두 가지 주요 모듈인 로컬 에이전트와 클라우드 에이전트로 구성되어 있으며, 각각의 에이전트는 서로 다른 복잡도의 추론 단계를 처리한다.

- **Technical Details**: 이 연구는 AdaSwitch라는 새로운 프레임워크를 통해 소형 LLM과 대형 LLM 간의 협업을 가능하게 하며, 이를 통해 계산 비용을 줄이고 성능을 향상시킨다. 로컬 에이전트는 더 간단한 추론 단계를 처리하고, 클라우드 에이전트는 복잡한 추론 단계를 관리한다. 로컬 에이전트는 필수적인 경우 클라우드 에이전트에게 도움을 요청하도록 설계되었다.

- **Performance Highlights**: AdaSwitch는 수학적 추론 및 복잡한 질문 응답 등 7개의 벤치마크에서 실험을 진행했으며, 다양한 LLM을 사용하여 로컬 및 클라우드 에이전트를 초기화하는 데 성공했다. 예를 들어 DeepSeek-Coder-1.3B 모델은 성능이 29.3%에서 53.9%로 향상되었고, StarCoder2-3B는 Llama-30B에 비해 5배 적은 계산 오버헤드로 유사한 성능을 달성하였다.



### SLM-Mod: Small Language Models Surpass LLMs at Content Moderation (https://arxiv.org/abs/2410.13155)
Comments:
          Preprint: 15 pages, 8 figures, 8 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs) 대신 커뮤니티 특화 콘텐츠 조정을 위해 오픈 소스의 소형 언어 모델(SLMs)을 사용하는 가능성을 탐구합니다. SLM들은 비용 효율적이며 LLMs와 비슷한 성능을 발휘할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: SLMs(15B 파라미터 이하)에 대한 세부 조정을 수행하고, 15개의 인기 있는 Reddit 커뮤니티에서 수집한 150K 댓글을 활용하여 이들의 성능을 평가했습니다. 결과적으로 SLMs는 콘텐츠 조정에서 LLMs보다 11.5% 더 높은 정확도와 25.7% 더 높은 재현율을 보였습니다. 소형 언어 모델을 조정하기 위한 저자원 접근법인 LoRA(Low-Rank Adaptation) 기법이 사용되었습니다.

- **Performance Highlights**: SLMs는 인도메인 콘텐츠 조정 작업에서 LLMs에 비해 더 높은 정확도와 재현율을 기록했으며, 다소 낮은 정밀도를 보였습니다. 이는 커뮤니티 특정 조정을 위한 더 효율적인 대안을 제공할 수 있음을 보여줍니다.



### Better to Ask in English: Evaluation of Large Language Models on English, Low-resource and Cross-Lingual Settings (https://arxiv.org/abs/2410.13153)
- **What's New**: 이 연구는 저자들이 방글라어, 힌디어, 우르두어 같은 저자원 언어에서의 LLM(대형 언어 모델) 성능을 평가하고, GPT-4, Llama 2, Gemini의 효과를 분석한 결과를 담고 있습니다. 또한, 모델들이 저자원 언어에서 영어만큼 성능을 발휘하지 못함을 강조합니다.

- **Technical Details**: 연구에서는 제로샷 프롬프팅(zero-shot prompting)과 5가지 서로 다른 프롬프 설정을 사용하여 방글라어, 힌디어, 우르두어와 같은 저자원 언어에서의 LLM 성능을 분석했습니다. 이 연구는 또한 자연어 추론(NLI)을 포함하여 LLM 성능을 향상시키기 위한 새로운 접근 방식을 소개하였습니다.

- **Performance Highlights**: 결과는 GPT-4가 모든 언어와 프롬프 설정에서 Llama 2 및 Gemini보다 우수한 성능을 보였다는 것을 보여줍니다. 또한 모든 LLM이 저자원 언어 프롬프보다 영어 프롬프에서 더 나은 성능을 발휘했습니다.



### Mapping Bias in Vision Language Models: Signposts, Pitfalls, and the Road Ahead (https://arxiv.org/abs/2410.13146)
Comments:
          Under Review at NAACL 2025

- **What's New**: 이 논문은 Vision Language Models (VLMs)의 공정성을 평가하기 위해 5개의 모델과 6개의 데이터셋을 분석하고, 편향(bias)에 대한 새로운 통찰을 제공합니다. 특히, 기존의 표정 기반(portrait-based) 데이터셋이 VLM의 공정성을 평가하는 데 가장 유용하다는 것을 발견했습니다.

- **Technical Details**: 본 연구는 UTKFace, CelebA, PATA, VLStereoSet, VisoGender와 같은 여러 데이터셋을 활용하여 VLM의 편향을 평가하였습니다. 각 모델이 성별, 인종 및 연령과 같은 보호 속성을 어떻게 처리하는지 분석를 진행하며, 특히 데이터셋의 구성에 따라 평가 결과가 달라질 수 있음을 강조합니다. VisoGender 데이터셋의 어려운 버전을 소개하여 철저한 평가를 가능하게 합니다. 

- **Performance Highlights**: LLaVa와 CLIP 모델 간의 성능과 공정성의 격차를 발견하여, VLM의 공정성 평가에 대한 보다 효과적이고 체계적인 데이터셋 설계의 필요성을 강조합니다. 저자들은 기존 데이터셋의 한계를 지적하며, VLM의 평가를 위한 향후 연구 방향을 제안합니다.



### Data Defenses Against Large Language Models (https://arxiv.org/abs/2410.13138)
- **What's New**: 본 논문에서는 "data defenses"라는 새로운 전략을 정의하고 구축하여 데이터 소유자가 LLM(대형 언어 모델)의 추론을 차단할 수 있도록 하는 방안을 제안합니다. 이는 개인 식별 정보 추론이나 저작권 텍스트 사용을 감소시키기 위한 자동 생성적 adversarial prompt injections 기법으로 구성됩니다.

- **Technical Details**: 우리는 저작권 침해, 개인 정보 보호 침해 및 감시 강화를 포함한 다양한 추론 시 발생할 수 있는 해악을 확인합니다. 이는 LLM 추론을 방해함으로써 권한을 데이터 소유자에게 되돌리는 기회를 제공합니다. 이 "data defenses"는 보다 저렴하고 빠르게 생성될 수 있으며, 중앙 집중적인 대처 수단 없이도 LLM의 정확도를 크게 떨어뜨립니다. 또한 이러한 방어 방법은 여러 공격 설정에 강력하며, 상업적 및 오픈 소스 LLM 모두에 대해 효과적임을 입증합니다.

- **Performance Highlights**: 우리는 제안하는 data defenses가 최신 상업적 및 오픈 소스 LLM에서 효과적으로 작동하며, 방어 메커니즘 생성 시간이 매우 짧고 자동화되어 있다는 것을 보여줍니다. 이러한 데이터 방어는 사용자가 콘텐츠에 대한 LLM의 추론을 제어할 수 있는 즉각적이고 저비용의 방법을 제공합니다.



### Retrieval-Enhanced Named Entity Recognition (https://arxiv.org/abs/2410.13118)
Comments:
          13 pages, 6 figures, 3 tables

- **What's New**: RENER (Retrieval-Enhanced Named Entity Recognition)는 In-Context Learning (ICL) 및 정보 검색 기술을 결합하여 명명된 개체 인식(NER) 작업에서 성능을 향상시키기 위해 제안된 새로운 방법입니다. 이 방법은 입력 텍스트에 대해 유사한 예제를 검색하고 이를 언어 모델에 통합하여 NER을 수행할 수 있도록 합니다.

- **Technical Details**: RENER는 언어 모델과 정보 검색 알고리즘에 독립적이며, 언어 모델과의 결합이 최소화된 상태로 새로운 명명된 개체를 인식하는 데 사용할 수 있습니다. 이 과정에서 언어 모델에 직접적인 의존성이 없으며, 다양한 NER 도메인에 쉽게 배포할 수 있습니다. 또한 CrossNER 컬렉션에서의 실험 결과, RENER는 최신 기술(State-of-the-Art) 성능을 달성하였으며, 정보 검색 기술을 사용할 경우 F-score를 최대 11% 상승시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: RENER는 CrossNER 데이터셋에서 최신 성능을 달성하였으며, 정보 검색 기법을 활용함으로써 비슷한 시스템 대비 성능을 최대 11% 향상시킬 수 있었습니다. 이는 NER 작업에서 ICL과 RAG 기법을 성공적으로 결합했음을 나타냅니다.



### Learning to Summarize from LLM-generated Feedback (https://arxiv.org/abs/2410.13116)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 생성 피드백을 통해 요약 품질을 향상시키는 방법을 탐구하고, FeedSum이라는 대규모 데이터셋을 소개합니다. 이는 다양한 품질의 요약에 대한 다차원 LLM 피드백을 포함하고 있습니다.

- **Technical Details**: FeedSum 데이터셋에서는 13개의 서로 다른 언어 모델을 사용하여 요약을 생성하고, 각 요약에 대해 신뢰성(faithfulness), 완전성(completeness), 간결성(conciseness)이라는 세 가지 핵심 차원에 대한 피드백을 수집합니다. 두 가지 방법인 감독형 세부 조정(supervised fine-tuning)과 직접 선호 최적화(direct preference optimization)를 비교하였고, SummLlama3-8B 모델이 Llama3-70b-instruct 모델보다 더 뛰어난 성능을 보였음을 확인했습니다.

- **Performance Highlights**: SummLlama3-8B 모델은 크기가 10배 이상 큰 Llama3-70b-instruct 모델을 초월하여 인간의 선호에 맞는 요약을 생성하는 데 성공하였습니다. 이는 Smaller 모델이 적절한 훈련을 통해 더 우수한 성능을 얻을 수 있음을 보여줍니다.



### A Little Human Data Goes A Long Way (https://arxiv.org/abs/2410.13098)
- **What's New**: NLP 시스템의 효율성을 높이기 위해, 인간 주석 데이터의 일부를 합성 데이터로 대체하는 방법을 연구하였으며, 90%까지 대체해도 성능 저하가 미미하지만 마지막 10% 대체 시에는 성능이 크게 떨어진다는 중요한 발견을 했습니다.

- **Technical Details**: 합성 데이터 생성 과정을 통해 데이터 포인트 수를 일정하게 유지하며 인간 생성 데이터 비율을 점진적으로 증가시켜 성능을 비교하였습니다. 사용하는 데이터셋은 총 8개로 Fact Verification (FV) 및 Question Answering (QA) 태스크에 대해 실험하였습니다. 평가 지표로는 정확도, Exact Match, String Inclusion, BLEU, ROUGE-1, BERTScore를 사용하였습니다.

- **Performance Highlights**: 완전히 합성 데이터로 훈련된 FV 및 QA 시스템은 최소 125개의 인간 데이터 포인트를 추가할 경우 성능이 현저히 개선되며, 작은 비율의 인간 데이터가 큰 가치를 지닐 수 있다는 것을 발견했습니다. 추가적인 인간 데이터를 통한 성능 향상은 200 포인트의 인간 데이터로 가능하며, 이는 수량적으로 더 많은 합성 데이터 포인트에 비해 비용 효율적이라는 것을 보여줍니다.



### Reverse-Engineering the Reader (https://arxiv.org/abs/2410.13086)
- **What's New**: 이 연구는 기존의 언어 모델을 인간의 심리 측정 데이터에 맞춰 최적화하는 새로운 방법론을 제시합니다. 이를 통해 언어 처리 시스템의 이해를 높이고자 합니다.

- **Technical Details**: 연구진은 언어 모델이 특정 언어 단위의 읽기 시간을 예측하는 능력을 향상시키기 위해 서프라이절 이론(surprisal theory)을 기반으로 한 새로운 정렬 기법을 사용합니다. 모델의 파라미터를 조정하여 읽기 시간을 예측하는 선형 회귀의 계수를 최적화합니다.

- **Performance Highlights**: 제안된 기법은 여러 모델 크기와 데이터 세트에서 언어 모델의 심리 측정 예측력을 향상시키는 것으로 나타났습니다. 그러나 심리 측정 예측력과 후속 자연어 처리(NLP) 작업 성능 간에 반비례 관계가 발견되었습니다.



### Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models (https://arxiv.org/abs/2410.13080)
Comments:
          21 pages, 10 figures

- **What's New**: 이 논문은 GCR(Graph-Constrained Reasoning)이라는 새로운 프레임워크를 제안합니다. GCR은 구조화된 지식(graph)과 비구조화된 추론을 결합하여 LLM의 신뢰할 수 있는 추론을 돕습니다.

- **Technical Details**: GCR은 KG-Trie라는 trie 기반 인덱스를 사용하여 KG의 구조를 LLM의 디코딩 과정에 통합합니다. 이를 통해 LLM은 KG에 직접적으로 기반한 추론을 수행하고 잘못된 정보(hallucinations)를 제거할 수 있습니다.

- **Performance Highlights**: 포괄적인 실험 결과에 따르면 GCR은 여러 KGQA 벤치마크에서 최신 성능(state-of-the-art performance)을 달성하고, 추가적인 학습 없이도 새로운 KG에 강력한 제로샷 제너럴라이제이션(zero-shot generalizability)을 보이는 것으로 나타났습니다.



### Tuning Language Models by Mixture-of-Depths Ensemb (https://arxiv.org/abs/2410.13077)
- **What's New**: 최근 연구에서는 Transformer 기반의 대형 언어 모델(LLMs)에서 최종 레이어만을 사용하는 대신, 중간 레이어의 예측 능력에 주목하여 새로운 조정 프레임워크인 Mixture-of-Depths (MoD)를 제안하였습니다. MoD는 훈련 시 다양한 레이어의 출력을 활용함으로써 예측 성능을 향상시키고, 기존 조정 방법과 통합할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: Mixture-of-Depths (MoD) 프레임워크는 레이어별 가중치를 학습하여 최종 로그잇(logits)으로 기여하는 앙상블로서의 레이어 훈련을 가능하게 합니다. 보조 증류 손실(auxiliary distillation loss) 및 추가 정규화 모듈을 적용하여, 최종 레이어 출력을 교사가 되는 훈련 방식으로 중간 레이어의 예측 출력을 모델 학습 시 최대화하는 접근을 취합니다. 이 방법은 훈련 가능한 파라미터를 소폭 증가시키면서도, 기본 언어 모델의 성능을 유지합니다.

- **Performance Highlights**: MoD 프레임워크를 적용한 결과, 산술 및 상식 추론 작업에서 성능이 일관되게 향상되었으며, 전통적인 훈련 가능한 모듈과 비교하여 97% 적은 파라미터로 유사한 성능을 달성하였습니다. 이러한 결과는 LLM의 중간 표현을 활용하는 것이 훈련 중 예측 능력을 크게 향상시킬 수 있음을 보여줍니다.



### PromptExp: Multi-granularity Prompt Explanation of Large Language Models (https://arxiv.org/abs/2410.13073)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 자연어 처리에서 대규모 언어 모델(LLMs)의 해석 가능한 프롬프트 공학을 위한 새로운 프레임워크인 OurTool을 소개합니다. 이 프레임워크는 토큰 수준의 통찰력을 집계하여 멀티 그레인 울라리티(multi-granularity) 설명을 제공합니다.

- **Technical Details**: OurTool은 두 가지 토큰 수준 설명 접근 방식을 제안합니다: 첫 번째는 aggregation-based approach로, 기존의 로컬 설명 기법을 결합하여 프롬프트의 각 토큰에 대한 포괄적인 설명을 생성합니다. 두 번째는 perturbation-based approach로, 토큰 마스킹의 영향을 평가하기 위한 새로운 기법을 도입합니다.

- **Performance Highlights**: 사례 연구를 통해 OurTool이 감정 분석에서 최고의 성과를 나타내며, 사용자의 평가에서도 80% 이상의 참여자가 OurTool이 제공하는 설명이 타당하고 정확하다고 응답했습니다.



### Is Semantic Chunking Worth the Computational Cost? (https://arxiv.org/abs/2410.13070)
- **What's New**: 최근 Retrieval-Augmented Generation (RAG) 시스템에서 문서를 의미적으로 일관된 세그먼트로 분할하는 semantic chunking이 인기를 얻고 있습니다. 본 연구는 semantic chunking이 보다 간단한 fixed-size chunking에 비해 실질적인 이점을 제공하는지에 대한 체계적인 평가를 진행했습니다.

- **Technical Details**: 연구팀은 document retrieval, evidence retrieval, answer generation 세 가지 일반적인 retrieval 관련 작업을 통해 semantic chunking의 효용성을 평가했으며, 다양한 chunking 전략을 비교하여 최적의 성능을 갖는 chunker를 확인했습니다. 또한, 두 가지 chunking 전략으로 fixed-size chunker와 breakpoint-based semantic chunker, clustering-based semantic chunker를 채택하여 평가하였습니다.

- **Performance Highlights**: 결과적으로, semantic chunking이 특정 상황에서 일부 이점을 보였지만, 이러한 이점들은 불일치하며 고정 크기 청크에 대한 계산 비용을 정당화할 만큼 충분하지 않다는 것을 보였습니다. 이는 RAG 시스템에서 더 효율적이고 적응적인 chunking 전략의 필요성을 강조합니다.



### ERAS: Evaluating the Robustness of Chinese NLP Models to Morphological Garden Path Errors (https://arxiv.org/abs/2410.13057)
Comments:
          Under review in ARR/NAACL

- **What's New**: 이 논문에서는 중국어를 다루는 NLP 모델들이 형태소적 garden path 오류에 취약하다는 것을 보여줍니다. 이를 평가하기 위해 ERAS라는 벤치마크를 제안합니다.

- **Technical Details**: ERAS 벤치마크는 지역적으로 모호한 구문과 모호하지 않은 구문으로 이루어진 203,944 쌍의 시험 문장과 통제 문장을 포함합니다. 이 연구는 Transformer 기반 및 비신경 단어 분리 모델과 캐릭터 수준의 토큰화를 사용하는 감정 분석 모델을 평가합니다.

- **Performance Highlights**: 실험 결과, 단어 분리 모델과 감정 분석 모델 모두가 garden path 오류를 범하며, 단어 경계 정보를 제공하여 모델 성능을 개선할 수 있다는 것을 보여줍니다.



### Channel-Wise Mixed-Precision Quantization for Large Language Models (https://arxiv.org/abs/2410.13056)
- **What's New**: 본 연구에서는 채널별 혼합 정밀도 양자화(Channel-Wise Mixed-Precision Quantization, CMPQ)라는 혁신적인 방법을 제안합니다. CMPQ는 각 채널의 활성화 분포에 기반하여 양자화 정밀도를 할당하는 새로운 혼합 정밀도 양자화 기법으로, 다양한 비트폭 제약에 적응하도록 설계되었습니다.

- **Technical Details**: CMPQ는 비균일 양자화(non-uniform quantization) 전략을 채택하며, 두 가지 이상치 추출(outlier extraction) 기법을 결합하여 필수 정보를 보존하며 양자화 손실을 최소화합니다. 이 방법은 채널별로 정밀도를 조정하여 각 채널의 활성화 norm에 따라 높은 정밀도 혹은 낮은 정밀도를 할당합니다.

- **Performance Highlights**: CMPQ는 실험을 통해 정수 비트 양자화(integer-bit quantization) 작업에서 성능을 향상시키는 한편, 적은 메모리 증가로 상당한 성능 향상을 이끌어냈습니다. 이 연구는 다양한 디바이스의 기능에서 큰 이점을 제공합니다.



### LFOSum: Summarizing Long-form Opinions with Large Language Models (https://arxiv.org/abs/2410.13037)
- **What's New**: 이 논문에서는 온라인 리뷰의 대량 처리 및 요약을 위한 새로운 접근법을 제안합니다. 특히, 1천 개 이상의 리뷰로 구성된 새로운 데이터셋을 소개하며, 이를 기반으로 하는 LLM(대형 언어 모델) 기반 요약 기법을 제안합니다.

- **Technical Details**: LFOSum 데이터셋은 TripAdvisor에서 수집된 호텔 리뷰로, 각 엔티티는 1천 개 이상의 리뷰를 포함하고 있습니다. 두 가지의 훈련이 필요 없는 요약 방법, 즉 Retrieval-Augmented Generation (RAG)과 긴 맥락의 LLM을 이용하여 대량 리뷰 요약을 처리합니다. 사용자 맞춤형 요약을 위한 세 가지 제어 메커니즘(쿼리 제어, 감정 제어, 길이 제어)을 도입하여 사용자 요구에 맞춘 요약을 가능하게 합니다.

- **Performance Highlights**: LLM은 여전히 긴 형식의 요약에서 감정과 형식 준수의 균형을 맞추는 데 어려움을 겪고 있으나, 관련 정보를 집중적으로 추출할 경우 오픈 소스 모델이 효과적으로 간격을 좁힐 수 있음을 보여줍니다.



### When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems (https://arxiv.org/abs/2410.13029)
Comments:
          11 pages, 7 figures, 2 tables

- **What's New**: 본 논문은 대형 언어 모델(GPT 모델)이 해결 불가능한 수학 단어 문제에 적절하게 대응할 수 있는지 평가하고, 이러한 모델들의 개선 방안을 모색한다. 특히, 모델들이 정답이 없을 경우 어떻게 'abstain' (응답 거부) 할 수 있는지를 연구하며, 이 과정을 향상시키기 위한 프롬프트 기술을 탐구한다.

- **Technical Details**: 연구는 Unanswerable Word Math Problem (UWMP) 데이터셋을 활용하였으며, 각 문제에 대해 'abstention' (응답 거부), 정확도(correctness), 신뢰도(confidence)의 세 가지 요소를 통합한 평가 지표를 도입하였다. 실험을 통해 다양한 프롬프트 기술을 적용하여 모델의 응답 행동을 분석하고, 해결 불가능한 질문에 대한 모델의 경향성을 분석하였다.

- **Performance Highlights**: 실험 결과, GPT 모델들은 해결이 불가능한 문제에 대해 잘못된 정보를 스스로 생성하는 경향이 있으며, 결과적으로 이러한 모델들은 수학 문제 해결에 있어 불확실성과 복잡한 추론을 효과적으로 관리하지 못한다는 점이 밝혀졌다. 이는 향후 모델들이 더 나은 관리와 결정을 내릴 수 있도록 개선이 필요함을 강조한다.



### LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks (https://arxiv.org/abs/2410.13025)
Comments:
          9 pages plus references and appendices

- **What's New**: 이 논문은 Low-Rank Adaptation (LoRA) 기법을 사용하여 여러 LoRA 모듈을 통합하여 기술 조합(skill composition)을 구현하는 방법을 연구합니다. 이 방식은 특정 기술과 지식이 필요한 작업에서 이전 모델 및 데이터 병합 기법보다 우수한 성능을 보입니다.

- **Technical Details**: 주요 기여는 LoRA의 연결(concatenation)을 최적으로 평균화하여 서로 다른 기술로 개별 훈련된 LoRA를 병합하는 새로운 방법인 Learnable Concatenation (CAT)을 제안하는 것입니다. 이는 모델의 일부 층에서 저랭크 업데이트를 추가하여 진행됩니다.

- **Performance Highlights**: CAT는 수학 문제 해결에서 기존 방법에 비해 평균 43% 및 12% 향상을 보이며, LLM의 프롬프트 형식 변화에 대한 견고성도 개선합니다. 본 연구는 기술 조합 작업을 해결하기 위한 효율적인 방법으로 모델 병합을 지지합니다.



### LEGAL-UQA: A Low-Resource Urdu-English Dataset for Legal Question Answering (https://arxiv.org/abs/2410.13013)
Comments:
          8 pages

- **What's New**: LEGAL-UQA는 파키스탄 헌법에서 유래한 첫 번째 우르두 법률 질문-답변(QA) 데이터셋을 소개합니다. 이 데이터셋은 619개의 질문-답변 쌍을 포함하며, 법률 기사의 컨텍스트도 포함되어 있어 낮은 자원 언어의 도메인 특화된 NLP 자원의 필요성을 해결합니다.

- **Technical Details**: 데이터셋 생성 과정은 OCR 추출, 수동 수정 및 GPT-4를 활용한 QA 쌍의 번역 및 생성으로 구성됩니다. LEGAL-UQA의 성능을 평가하기 위해 최신의 일반 언어 및 임베딩 모델을 실험하였으며, Claude-3.5-Sonnet 모델이 인간 평가에서 99.19%의 정확도를 달성하였습니다. 또한, mt5-large-UQA-1.0 모델을 미세 조정하여 다국어 모델을 전문 분야에 적용하는 데 따른 도전 과제를 강조하였습니다.

- **Performance Highlights**: OpenAI의 text-embedding-3-large는 Mistral의 mistral-embed 보다 더 나은 검색 성능을 보였습니다. LEGAL-UQA는 글로벌 NLP 발전과 현지화된 응용 프로그램 간의 격차를 해소하며, 파키스탄 내 법률 정보 접근성을 개선하는 기반을 마련합니다.



### POROver: Improving Safety and Reducing Overrefusal in Large Language Models with Overgeneration and Preference Optimization (https://arxiv.org/abs/2410.12999)
- **What's New**: 최근 대규모 언어 모델(LLM)의 안전성과 유용성 균형을 맞추는 것이 주요 도전 과제가 되고 있습니다. 본 연구는 GPT-4o와 같은 고급 'teacher' 모델을 사용하여 훈련 데이터를 과도하게 생성하는 방식이 안전성(safety)과 과도 거부(overrefusal) 간의 균형에 미치는 영향을 분석합니다.

- **Technical Details**: 본 작업에서는 POROver(Preference Optimization for Reducing Overrefusal)라는 새로운 전략을 통해 고급 teacher 모델의 응답을 활용하여 과도 거부를 줄이기 위한 방법을 제시합니다. 연구 결과에 의하면, 일반적인 목적의 프롬프트에 대해 과도하게 생성된 응답은 안전성과 유용성 간의 균형을 개선하며, F1 점수가 70.8%에서 88.3%로 향상됩니다.

- **Performance Highlights**: 과도 생성된 유해 프롬프트에 대한 응답의 경우, 과도 거부율이 94.4%에서 45.2%로 감소합니다. 또한, Preference Optimization 알고리즘을 활용하면 모델의 과도 거부율을 15.0%로 줄일 수 있으며, 비교 가능한 수준의 안전성을 유지할 수 있습니다.



### "Let's Argue Both Sides": Argument Generation Can Force Small Models to Utilize Previously Inaccessible Reasoning Capabilities (https://arxiv.org/abs/2410.12997)
Comments:
          Accepted to Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual at EMNLP 2024

- **What's New**: 이 연구에서는 논리적 추론이 필요한 상황에서 대규모 언어 모델(LLM)의 성능 강화를 위해 'Argument Generation'(주장 생성)이라는 새로운 기법을 제안합니다.

- **Technical Details**: 주장 생성 기법은 두 단계로 진행됩니다. 첫 번째 단계에서는 가능한 선택지 각각에 대한 주장을 생성하도록 모델에게 지시하고, 두 번째 단계에서는 생성된 주장들을 랭크한 다음, 최종 결과(output)와 일치하도록 매핑합니다. 이 방법은 체인 오브 스로우(Chain-of-Thought) 기법에 비해 더 나은 결과를 도출할 수 있습니다.

- **Performance Highlights**: 실험 결과, 주장 생성 기법은 체인 오브 스로우 기법보다 적어도 동등하거나 우수한 성능을 보이며, 특히 작은 언어 모델에 대해 더 큰 성능 향상을 보여주는 복잡한 관계를 나타냅니다.



### Qtok: A Comprehensive Framework for Evaluating Multilingual Tokenizer Quality in Large Language Models (https://arxiv.org/abs/2410.12989)
Comments:
          24 pages, 9 figures, 6 tables. Code and data available at this https URL

- **What's New**: 이번 연구에서 우리는 Qtok이라는 도구를 도입하여 멀티링구얼 모델에서의 토크나이저 품질을 평가하는 방법론을 제공합니다. 기존의 연구에서는 주로 데이터셋 품질이나 모델 아키텍처에 초점을 맞추었지만, 토크나이저의 중요성은 상대적으로 간과되었습니다.

- **Technical Details**: 연구팀은 Qtok 도구를 통해 58개의 공개 모델에서 13개의 다양한 토크나이저를 평가했습니다. 이 도구는 언어 범위, 토큰 완전성, 언어 및 언어 범주에 따라 분포를 측정하는 지표를 포함하여 토크나이저의 품질을 평가합니다. 또한 코어 토큰 개념을 도입하여 긍정적으로 반복되는 토큰을 구분하였습니다.

- **Performance Highlights**: 분석 결과, 다양한 언어 및 범주에서 토큰 분포의 중요 자질 불균형이 발견되어 현재의 토크나이징 전략에서 개선이 필요한 부분을 강조하였습니다. 연구는 토크나이저의 품질 평가 방법을 제공하고 이로 인해 멀티링구얼 LLM의 성능 향상 가능성을 제시합니다.



### Leveraging LLMs for Translating and Classifying Mental Health Data (https://arxiv.org/abs/2410.12985)
- **What's New**: 이번 연구는 그리스어로 생성된 사용자 게시글을 자동 번역하여 우울증의 심각성을 탐지하는 방법을 탐구합니다. 이는 영어 외의 언어에 대한 LLMs (Large Language Models)의 적용에 대한 연구가 부족한 상황에서 이루어졌습니다.

- **Technical Details**: 연구에서는 GPT3.5-turbo 모델을 사용하여 영어와 그리스어로 작성된 게시글에서 우울증의 심각성을 평가했습니다. LLMs의 결과는 영어에서 우울증을 효과적으로 식별하지 못했고, 그리스어에서도 성과가 다양합니다.

- **Performance Highlights**: 연구 결과, 우울증의 심각성을 인식하는 데 있어 GPT3.5-turbo의 성능이 그리스어에서도 일관되게 낮게 나타났으며, 추가 연구와 인간 감독의 중요성이 강조되었습니다.



### BenchmarkCards: Large Language Model and Risk Reporting (https://arxiv.org/abs/2410.12974)
- **What's New**: 대형 언어 모델(LLMs)의 위험을 줄이기 위한 새로운 프레임워크인 BenchmarkCards가 소개되었습니다. 이 프레임워크는 특정 취약성을 테스트하기 위해 설계된 벤치마크의 문서화 방식을 표준화합니다.

- **Technical Details**: BenchmarkCards는 LLM 벤치마크 속성을 문서화하는 구조적 프레임워크를 제공합니다. 이 프레임워크는 벤치마크 결과를 측정하거나 해석하는 방법을 정의하지 않고, 특정 리스크(위험) 및 평가 방법론에 대한 정보를 제공하는 표준화된 방법을 제공합니다. 포함된 속성으로는 bias(편향) 및 fairness(공정성) 등이 있습니다.

- **Performance Highlights**: 이 구조화된 메타데이터는 연구자들이 적절한 벤치마크를 선택할 수 있도록 도와주며, LLM 평가에서의 투명성과 재현성을 촉진합니다.



### Evaluating the Instruction-following Abilities of Language Models using Knowledge Tasks (https://arxiv.org/abs/2410.12972)
- **What's New**: 이번 연구에서는 과제 성과와 명령 이행 능력을 동시에 쉽게 검증할 수 있는 instruction-following을 위한 벤치마크를 개발하는 데 초점을 맞추었습니다.

- **Technical Details**: 기존의 지식 벤치마크를 수정하고, 올바른 지식 과제 답변에 따라 조건부로 적용되는 지침과 다중 선택 지식 답변 과제에서의 후보 옵션 공간을 활용한 지침으로 보강합니다. 연구에서는 다양한 크기의 공개 대형 언어 모델(1B-405B)과 GPT-4o-mini 및 GPT-4와 같은 폐쇄형 모델을 포함하여, 지침이 간단한 경우에도 LLM들이 따라가지 못하는 문제를 발견했습니다.

- **Performance Highlights**: 이 연구에서는 모델의 zero-shot instruction-following 성능을 평가하는 최초의 벤치마크를 출시하며, 다양한 지침 클래스에서 instruction-following 성능을 조사하고 이를 통해 모델의 지식 작업 수행능력을 질적으로 검토하였습니다.



### Self-Pluralising Culture Alignment for Large Language Models (https://arxiv.org/abs/2410.12971)
Comments:
          Implementation for the paper: this https URL

- **What's New**: 방금 발표된 연구에서 제안된 CultureSPA는 대규모 언어 모델(LLMs)이 여러 문화에 동시적으로 적응할 수 있도록 도와주는 혁신적인 프레임워크입니다. 기존의 기술은 특정 문화의 다양성을 고려하지 않았던 반면, CultureSPA는 LLM의 внутрен적인 문화 지식을 활용하여 이를 해결합니다.

- **Technical Details**: CultureSPA는 첫째, 다양한 문화 주제에 대한 질문을 생성하고, 둘째로, 이러한 질문에 대한 LLM의 출력을 수집합니다. 이 과정은 문화 정보를 제공하지 않는 culture-unaware prompting과 특정 문화에 맞춰 LLM을 유도하는 culture-aware prompting을 포함합니다. 최종적으로 이 데이터를 사용하여 LLM을 문화 간 협업(culture-joint) 및 특정 문화(culture-specific) 방식으로 정교화합니다.

- **Performance Highlights**: 광범위한 실험 결과, CultureSPA는 LLM의 다양한 문화에 대한 적합성을 크게 향상시켰으며, 일반적인 능력에 손상을 주지 않았습니다. 또한 CultureSPA와 고급 prompt engineering 기술을 결합하면 추가 개선이 가능합니다. Impressively, culture-joint vs culture-specific tuning 전략의 비교는 전자가 더 우수함을 보여줍니다.



### Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning (https://arxiv.org/abs/2410.12952)
- **What's New**: 본 논문에서는 기존의 단일 턴(single-turn) 상호작용과는 달리 다중 턴(multi-turn) 함수 호출을 LLM(대형 언어 모델)이 수행할 수 있는 필요성을 다룹니다. 이는 실제 세계에서 복합적인 쿼리를 처리하는 데 필수적입니다.

- **Technical Details**: 논문에서 제안하는 BUTTON 접근법은 하향식(top-down) 경로 생성과 하향식(bottom-up) 명령 구성 방법을 통해 합성된 지침 튜닝 데이터(synthetic compositional instruction tuning data)를 생성합니다. 하향식 단계에서는 원자적 작업(atomic tasks)을 기반으로 간단한 작업 정의 후, 이러한 작업을 사용하여 합성 작업을 개발합니다. 그런 다음, 하향식 단계에서는 시뮬레이션된 인간, 보조자, 도구 간의 상호작용을 통해 다중 턴 함수 호출 경로를 수집합니다.

- **Performance Highlights**: BUTTONInstruct라는 8천 개 데이터 포인트로 구성된 데이터셋을 제작하였으며, 다양한 LLM에서의 광범위한 실험을 통해 효과성을 입증하였습니다.



### What Do Speech Foundation Models Not Learn About Speech? (https://arxiv.org/abs/2410.12948)
Comments:
          20 Pages

- **What's New**: 본 연구에서는 Whisper, Seamless, Wav2Vec, HuBERT 및 Qwen2-Audio와 같은 여러 음성 기초 모델이 비언어적 신호(Non-verbal cues)를 어떻게 포착하고 있는지를 분석합니다. 특히 이들 모델의 학습된 표현(Representations)과 다양한 작업에서의 일반화 가능성을 평가합니다.

- **Technical Details**: 연구는 Dynamic-SUPERB 벤치마크에서 선택된 다섯 가지 모델의 레이어별 특징을 추출하고, 레이어별 특징에 대한 K-최근접 이웃(K-Nearest Neighbors, KNN) 및 신경망(Neural Networks, NN) 분류기를 훈련시켜 비언어적 신호에 대한 모델의 응답을 측정합니다. 또한, 제로-샷(Zero-shot) 환경에서 모델을 평가하고, 그 결과를 통해 모델의 레이어별 표현의 특성과 다운스트림(Downstream) 작업 적응에 필요한 변화의 정도를 파악합니다.

- **Performance Highlights**: 연구 결과, 일부 모델은 특정 작업에 대해 제로-샷 환경에서도 우수한 성능을 보이며, 이는 모델이 학습한 표현의 질과 상관관계를 나타냅니다. 또한 모델의 깊이에 따른 학습된 표현의 분리 가능성 사이에는 볼록한 관계가 존재하며, 이로 인해 작업 별 특성을 캡처하는 다양한 레이어가 확인됩니다.



### Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging (https://arxiv.org/abs/2410.12937)
Comments:
          Findings of EMNLP 2024

- **What's New**: 이 연구는 기존의 언어 모델에 새로운 기술을 추가하는 방식인 "parallel train then merge" (PTM) 접근법을 소개합니다. PTM은 여러 기술을 모델에 효율적으로 추가 가능하게 하며, 기존 기술을 잊게 하지 않고도 새로운 기술을 통합할 수 있는 장점이 있습니다.

- **Technical Details**: CFT (continued finetuning), RT (retraining), PTM은 기존 모델에 새로운 기술을 추가하기 위한 세 가지 방법입니다. PTM은 새로운 데이터에 대해서만 개별적으로 훈련하여 모델 파라미터를 병합하는 방식으로, 기존 기술을 보존하는 동시에 새로운 기술도 효과적으로 학습할 수 있습니다. 실험은 과학 문헌 이해, 코딩 및 안전 관련 요청 거부에서 진행되었습니다.

- **Performance Highlights**: PTM은 기존의 CFT보다 50–95% 효율적으로 훈련할 수 있으며, 원래 모델의 일반 기술을 거의 모두 유지하면서 새로운 기술에서도 유사한 성능을 달성합니다. 또한, PTM은 안전 관련 거부 능력을 개선하며, 전반적인 성능을 유지할 수 있습니다. 이 연구의 결과는 PTM이 CFT보다 효과적인 옵션이라는 것을 보여줍니다.



### Enhancing Mathematical Reasoning in LLMs by Stepwise Correction (https://arxiv.org/abs/2410.12934)
Comments:
          under review

- **What's New**: 이 논문에서는 Stepwise Correction (StepCo)라는 새로운 프롬프팅 방법을 제안합니다. 이 방법은 LLM이 스스로 생성한 추론 과정에서 잘못된 단계를 식별하고 수정하는 데 도움을 줍니다.

- **Technical Details**: StepCo는 프로세스 기반 슈퍼바이저(PSV)를 이용하여 이전 단계에서 발생한 오류가 다음 단계로 전파되지 않도록 하며, 각 단계의 정확성을 평가하고 수정하는 반복적인 verify-then-revise 프로세스를 적용합니다.

- **Performance Highlights**: StepCo는 여러 데이터세트에서 평균 정확도 94.1%를 달성하며, Best-of-N 방법에 비해 2.4% 높은 성능을 보이면서 토큰 소비를 77.8% 감소시킵니다.



### Interpreting token compositionality in LLMs: A robustness analysis (https://arxiv.org/abs/2410.12924)
Comments:
          15 pages, 2 Figures, 7 tables

- **What's New**: 이번 연구에서는 Constituent-Aware Pooling (CAP)という 새로운 방법론을 제안하여 대형 언어 모델(LLMs)의 구성적 언어 구조 처리 방식을 분석합니다. 이는 LLM의 내부 메커니즘을 이해하고, 신뢰성 및 해석 가능성을 개선하기 위해 필수적입니다.

- **Technical Details**: CAP는 토큰 식별 기법을 사용하여 개별 토큰의 활성화를 응집된 언어 단위로 집계하는 방법입니다. 연구에서는 세 가지 작업(역 정의 모델링, 동의어 예측 및 상위어 예측)을 통해 모델의 성능을 평가하였으며, 여기서 토큰 간의 정보 처리 방식 및 분산 현상을 관찰했습니다. CAP는 단어 수준 및 구문 수준 모두에서 작동하며, 세 가지 집계 모드(최대 집계, 평균 집계, 합계 집계)를 제공합니다.

- **Performance Highlights**: 연구 결과, LLM의 성능은 구성적 활성화 교란이 적용될 때 유의미하게 저하되는 것으로 나타났습니다. 특히 더 큰 모델일수록 이러한 교란에 더 민감한 경향을 보였습니다. 이는 현재의 변환기 아키텍처가 구성적 의미 처리에 있어 한계가 있음을 시사하며, 해결을 위한 새로운 접근 방식이 필요함을 강조합니다.



### MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation (https://arxiv.org/abs/2410.12916)
Comments:
          3rd Table Representation Learning Workshop at NeurIPS 2024

- **What's New**: 이 논문에서는 자연어를 SQL로 변환하기 위한 텍스트-투-SQL 생성 기술을 향상시키기 위한 새로운 모델, MSc-SQL을 소개하고 있습니다. 최근 몇 가지 클로즈드 소스 모델에 의존하여 성능이 제한된 문제를 해결하기 위해 작은 오픈 소스 모델을 개발하는 데 중점을 두고 있습니다.

- **Technical Details**: Msc-SQL 모델은 여러 SQL 쿼리를 샘플링하고, 생성된 쿼리의 실행 결과와 메타데이터를 바탕으로 최상의 결과를 선택하는 샘플 비평 모델(sample-critiquing model)을 포함합니다. 이 접근 방식은 여러 후보를 동시에 평가하여 보다 적절한 SQL 쿼리를 생성할 수 있도록 도와줍니다.

- **Performance Highlights**: MSc-SQL는 오픈 소스 모델 중에서 최신 성능을 기록하면서도 기존의 클로즈드 소스 모델과 경쟁력을 유지합니다. 이 모델은 더 낮은 비용으로도 높은 품질의 SQL 쿼리를 생성할 수 있습니다.



### A Survey on Data Synthesis and Augmentation for Large Language Models (https://arxiv.org/abs/2410.12896)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 개발에 있어 데이터 합성(data synthesis) 및 증강(augmentation) 기법의 중요성을 강조하며, 전체 생애 주기(lifecycle)와 핵심 기능에 따른 연구를 종합적으로 정리합니다.

- **Technical Details**: 데이터 합성과 증강은 LLM의 훈련과 평가를 위해 필수적으로 사용되며, 기존 데이터에 의존하는 것을 줄이고, 다양한 고품질 데이터를 생성하는 데 중요한 역할을 합니다. 이 연구에서는 LLM의 생애 주기(pre-training, fine-tuning 등)와 핵심 기능(이해, 논리 등)을 중심으로 기존 연구를 분류하고 체계적으로 검토합니다.

- **Performance Highlights**: 이 연구는 LLM 개발의 향후 방향성을 제시하고, 데이터 합성과 증강 기법이 LLM 성능 향상에 기여할 수 있는 가능성을 탐구합니다. 또한, 연구자들이 데이터를 효과적으로 생성하는 방법을 이해할 수 있도록 돕습니다.



### Large Language Models and the Rationalist Empiricist Deba (https://arxiv.org/abs/2410.12895)
- **What's New**: 이 논문은 LLMs(대형 언어 모델)가 Chomsky와 Quine, Skinner 간의 논쟁에 어떻게 영향을 미치는지를 탐구하며, LLMs가 합리주의를 정당화하는 주장과 기존의 경험주의에 대한 비판을 다룬다.

- **Technical Details**: LLMs는 본래의 편향을 내장해야 하며, 이는 언어능력(linguistic competence)을 설명하는 데 있어 경험주의가 개념적 자원을 부족하다는 주장을 뒷받침한다. 그러나 이러한 주장은 사용되는 경험주의의 성격에 의존한다.

- **Performance Highlights**: 인간은 한정된 자극(poverty of stimulus) 속에서도 학습하는 반면, LLMs는 풍부한 자극(rich stimulus) 덕분에 학습한다. 이는 인간과 LLMs가 출력(output)을 생성하는 데에 있어 다른 기본 능력(underlying competencies)을 사용함을 나타낸다.



### MIRROR: A Novel Approach for the Automated Evaluation of Open-Ended Question Generation (https://arxiv.org/abs/2410.12893)
Comments:
          Accepted at FM-Eduassess @ NEURIPS 2024 (ORAL Paper)

- **What's New**: 이번 연구에서는 자동 질문 생성(Automated Question Generation, AQG) 시스템이 생성한 질문의 품질 평가를 자동화하기 위해 대규모 언어 모델(LLM)을 활용하는 새로운 시스템인 MIRROR (Multi-LLM Iterative Review and Response for Optimized Rating)를 제안합니다.

- **Technical Details**: MIRROR는 여러 LLM에 피드백을 제공하여 인간의 평가 지표(grammaticality, relevance, appropriateness, novelty, complexity)에 기반하여 점수를 생성하는 프로세스를 포함합니다. GPT-4, Gemini, Llama2-70b와 같은 최첨단 LLM을 사용하여 실험을 진행하였으며, 인간 전문가의 평가와의 Pearson 상관 계수(Pearson's correlation coefficient)를 측정하여 결과를 비교하였습니다.

- **Performance Highlights**: MIRROR를 적용한 결과, relevance, appropriateness, novelty, complexity, grammaticality와 같은 인간 평가 지표의 점수가 개선되어 인간 기준 점수와 더 가까운 결과를 보였습니다. 더불어 직접 프롬프트를 사용하여 평가한 경우보다 인간 전문가와의 상관 계수가 향상되었습니다.



### Multi-trait User Simulation with Adaptive Decoding for Conversational Task Assistants (https://arxiv.org/abs/2410.12891)
Comments:
          Preprint fron EMNLP 2024 Findings

- **What's New**: 이 논문은 Multi-Trait Adaptive Decoding (mTAD)이라는 새로운 접근 방식을 제안합니다. mTAD는 다양한 trait-specific Language Models (LMs)에서 샘플링하여 디코딩 시간에 다양한 사용자 프로필을 생성하여 사용자 시뮬레이션을 개선합니다.

- **Technical Details**: mTAD는 다양한 대화 trait를 모델링하기 위해 specialized LMs를 결합하는 모델 기반 접근 방식을 따릅니다. 기존의 조합 훈련 데이터나 추가적인 모델 Fine-tuning 없이, trait-specific LM에서 분포를 샘플링하여 동적으로 결합합니다. 이 기법은 사용자 대화 프로필의 다양한 조합을 가능하게 하여 더 풍부한 대화 패턴을 생성합니다.

- **Performance Highlights**: 실험 결과, mTAD는 단일 trait 모델링에서 효과적임을 입증하며, 아울러 특정 패턴을 포착할 수 있는 능력을 보여줍니다. mTAD는 다양한 사용자 시뮬레이터를 결합하는 데 강력하고 유연한 프레임워크로, 기존 LM을 재훈련할 필요 없이 새로운 traits를 추가할 수 있습니다.



### REFINE on Scarce Data: Retrieval Enhancement through Fine-Tuning via Model Fusion of Embedding Models (https://arxiv.org/abs/2410.12890)
Comments:
          Accepted in AJCAI'24

- **What's New**: 본 논문에서는 데이터 부족 문제를 해결하기 위해 REFINE이라는 새로운 접근 방식을 제안합니다. 이 방법은 효과적인 검색을 개선하기 위해 사용 가능한 문서에서 합성 데이터를 생성하고, 모델 융합(Model Fusion) 기법을 통해 임베딩을 향상시킵니다.

- **Technical Details**: REFINE은 LLM(대규모 언어 모델)을 활용하여 사용 가능한 비지도 문서에서 대조적 훈련 데이터셋을 생성합니다. 생성된 데이터셋은 표준 파인튜닝 방법을 통해 임베딩 모델의 성능을 개선하며, 새로운 데이터 특정 학습을 포함하는 모델 융합 기법을 도입하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: SQUAD 및 RAG-12000 데이터셋과 독점 TOURISM 데이터셋에서 실험을 수행한 결과, REFINE이 적용된 표준 파인튜닝이 기본 사전 훈련 모델에 비해 더 나은 성능을 보였고, TOURISM 데이터셋에서는 5.76%, SQUAD 데이터셋에서는 6.58%의 개선을, RAG-12000 데이터셋에서는 0.32%의 향상을 기록했습니다.



### AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency with Topic Filtering and Iterative Reasoning (https://arxiv.org/abs/2410.12886)
- **What's New**: AT-RAG라는 새로운 멀티스텝 RAG 모델을 제안하여, 복잡한 다중 단계 쿼리를 보다 효율적으로 처리하는 방법을 소개합니다.

- **Technical Details**: AT-RAG는 BERTopic을 활용하여 쿼리의 주제를 동적으로 할당함으로써 문서 검색 및 추론 과정의 정확성과 효율성을 향상시킵니다. 이 모델은 Chain-of-Thought (CoT) 추론을 통합하여 반복적인 문서 검색 및 추론을 가능하게 합니다.

- **Performance Highlights**: AT-RAG는 기존 RAG 모델 대비 Accuracy, Completeness, Relevance에서 현저한 개선을 보였으며, 특히 의료 QA와 같은 복잡한 도메인-specific 문제 해결에 적합합니다. 모델은 다양한 benchmark dataset에서 높은 성능을 발휘하였고, 검색 시간을 줄이면서 높은 정밀도를 유지합니다.



### Scaling Laws for Multilingual Language Models (https://arxiv.org/abs/2410.12883)
- **What's New**: 본 연구에서는 다국어 데이터로 훈련된 일반 목적의 디코더 전용 언어 모델을 위한 새로운 스케일링 법칙을 제안합니다. 각 언어의 성능을 개별적으로 분석하기 어려운 문제를 다루기 위해, 우리는 개별 언어 대신 언어 가족에 초점을 맞추고, 각 언어 가족의 테스트 교차 엔트로피 손실(test cross-entropy loss)은 혼합 내 다른 언어와 무관하게 샘플링 비율(sampling ratio)에 의해 결정된다는 가설을 검증했습니다.

- **Technical Details**: 제안한 스케일링 법칙은 테스트 교차 엔트로피 손실을 모델 크기(model size), 데이터셋 크기(dataset size), 샘플링 비율(sampling ratios)과 연결하는 전력 법칙(power-law relationship)을 도출합니다. 이를 통해 다양한 조합에 대한 성능 예측이 가능해졌으며, 훈련 혼합 내 언어 가족의 최적 샘플링 비율을 도출할 수 있게 되었습니다. 우리는 23개 언어, 5개 언어 가족을 대상으로 100개 이상의 모델을 훈련하여 대규모 실증 연구를 수행했습니다.

- **Performance Highlights**: 실험 결과, 작은 모델(85M 파라미터)에서 도출한 최적 샘플링 비율이 수십 배 큰 모델(1.2B 파라미터)에도 효과적으로 일반화됨을 보여주었습니다. 이는 리소스를 효율적으로 사용할 수 있는 다국어 언어 모델 훈련을 위한 접근 방식을 제공합니다.



### Navigating the Cultural Kaleidoscope: A Hitchhiker's Guide to Sensitivity in Large Language Models (https://arxiv.org/abs/2410.12880)
- **What's New**: 이 논문은 글로벌 AI 애플리케이션에서 LLMs의 문화적 민감성을 보장하는 중요성을 강조하며, 작은 매개변수 모델 내에서 발생하는 문화적 손해를 다루기 위한 두 개의 주요 기여를 제시합니다. 첫째, 다양한 문화적 맥락에서 모델의 출력을 평가하기 위한 문화적 손해 테스트 데이터셋을 소개합니다. 둘째, 다양한 주석자 피드백을 기반으로 문화적 민감성을 회복하기 위한 데이터셋을 제안합니다.

- **Technical Details**: 이 연구는 문화적 손해 평가 데이터셋과 문화적 정렬 선호 데이터셋을 구축하여 작은 매개변수 LLMs의 문화적 민감성을 높이고 해로운 출력을 줄이는 것을 목표로 합니다. 데이터셋은 사회적, 정치적, 경제적, 종교적, 문화적 가치를 반영하며, 다양한 문화적 맥락에서 모델 출력을 시스템적으로 평가할 수 있는 프레임워크를 제공합니다. 또한, reinforcement learning from human feedback (RLHF) 기법을 사용하여 문화적 기준을 존중하는 모델의 미세 조정을 지원합니다.

- **Performance Highlights**: 문화적 정렬 피드백을 통합함으로써 Mistral-v0.2(7B) 모델의 해로운 출력 발생률이 71.96%에서 3.07%로 급격히 감소하는 등의 성과를 보였습니다. 이 연구는 LLM이 다양한 문화적 경관에서 안전하고 윤리적으로 탐색할 수 있는 미래의 AI 시스템을 구축하는 데 기여할 것입니다.



### Exploring transfer learning for Deep NLP systems on rarely annotated languages (https://arxiv.org/abs/2410.12879)
- **What's New**: 이 논문은 자연어 처리(NLP) 분야에서 힌디어와 네팔리 언어 간의 Part-of-Speech (POS) 태깅에 대한 전이 학습(Transfer Learning)의 응용을 연구합니다. 특히, 두 언어의 공동 훈련이 성능 향상에 기여하는지를 탐구합니다.

- **Technical Details**: 연구에서는 BLSTM-CNN-CRF(변형 장기 단기 기억-합성곱 신경망-조건부 무작위 필드) 모델을 사용하며, 단일 언어(word embeddings), 벡터 매핑 임베딩(vector-mapped embeddings), 공동 훈련된 힌디어-네팔리 단어 임베딩(jointly trained Hindi-Nepali word embeddings)에서의 결과를 비교합니다. 다양한 드롭아웃 비율(0.25~0.5)과 최적화 알고리즘(ADAM, AdaDelta)을 평가합니다.

- **Performance Highlights**: 결과는 공동 훈련된 힌디어-네팔리 단어 임베딩이 단일 언어 및 벡터 매핑 임베딩에 비해 모든 모델에서 성능을 향상시킴을 보여줍니다.



### Towards More Effective Table-to-Text Generation: Assessing In-Context Learning and Self-Evaluation with Open-Source Models (https://arxiv.org/abs/2410.12878)
Comments:
          15 pages

- **What's New**: 이 연구는 자연어 처리의 핵심 작업인 테이블-텍스트 생성(table-to-text generation)에 대해, 다양한 in-context learning 전략의 효과를 평가합니다. 특히, 모델에 주어진 예제가 성능에 미치는 영향을 조사하고, 실제 애플리케이션을 기반으로 한 사례를 제공합니다.

- **Technical Details**: 모델은 zero-shot, single-shot, few-shot 프롬프트를 사용하여 테이블 데이터에서 내러티브 텍스트로 전환합니다. 이 연구에서는 두 개의 벤치마크 데이터셋인 WikiBio와 ToTTo에서 실험이 수행되었고, Llama 3와 Phi-3 모델을 사용하여 결과를 비교했습니다. 또한, GPT-4를 사용하여 초기 프롬프트를 생성하고, 이를 기반으로 최적화를 진행하였습니다.

- **Performance Highlights**: 예제를 제공함으로써 테이블-텍스트 생성의 성능이 크게 향상되었습니다. LLM 자가 평가 방법은 아직 인간의 판단과 일치도가 개선되어야 하지만, overall 성능 개선을 확인할 수 있었습니다.



### Improving Instruction-Following in Language Models through Activation Steering (https://arxiv.org/abs/2410.12877)
- **What's New**: 이 논문에서는 언어 모델(LLM)의 지침 따르기 능력을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이는 지침에 따라 모델의 동작을 조정하기 위해 지침별 벡터 표현을 파생하는 내용을 다룹니다.

- **Technical Details**: 이 연구는 입력의 지침이 없는 경우와 있는 경우의 활성화(activation)의 차이를 기반으로 벡터 표현을 계산하여 모델의 출력을 조작하는 방식입니다. 사용된 활성화 벡터는 출력 형식, 길이, 특정 단어 포함 여부 등 여러 조건을 모델이 준수하도록 유도합니다.

- **Performance Highlights**: 4개의 서로 다른 모델을 대상으로 한 실험을 통해, 이 방법이 지침을 명시적으로 제공하지 않아도 모델이 제약사항을 따르도록 도와주고, 지침이 있을 때도 성능을 향상시킬 수 있음을 보여주었습니다. 또한, 여러 지침을 동시에 적용할 수 있다는 것이 확인되었습니다.



### In-context KV-Cache Eviction for LLMs via Attention-Ga (https://arxiv.org/abs/2410.12876)
- **What's New**: 본 논문은 Attention-Gate라는 매개변수화된 KV-Cache 제거 메커니즘을 도입하여 비효율적인 기존 제거 전략의 한계를 극복하려고 합니다. 이는 입력된 전체 컨텍스트를 기준으로 각 토큰의 제거 플래그를 생성하여 효율적인 in-context eviction을 실현합니다.

- **Technical Details**: Attention-Gate(AG)는 모델 내의 self-attention 레이어 전방에 위치하여, 입력된 토큰 특징 시퀀스를 처리하여 각 토큰에 대한 제거 플래그를 생성합니다. AG는 사전 훈련된 대형 언어 모델에 무리 없이 통합될 수 있으며, 최소한의 컴퓨팅 및 메모리 오버헤드를 가지면서 효율적입니다.

- **Performance Highlights**: 효율적인 지속적 사전 훈련(CPT) 후에, 기존의 훈련 없는 제거 전략보다 더 높은 평균 정확도를 달성하며 더 많은 토큰을 제거할 수 있음을 증명합니다. Supervised fine-tuning(SFT)에서는 LoRA로 미세 조정된 LLM보다 성능이 우수하며, RTE 데이터셋에서 13.9%의 정확도 향상과 62.8%의 토큰 제거를 달성하여 중복 토큰의 효과적인 제거가 성능을 개선할 수 있음을 나타냅니다.



### On Debiasing Text Embeddings Through Context Injection (https://arxiv.org/abs/2410.12874)
- **What's New**: 이 논문에서는 텍스트 임베딩 모델에서의 편향(bias)을 정량화하고, 그로부터 방지 성능을 평가하기 위해 19개의 임베딩 모델을 체계적으로 분석하였습니다. 최신 컨텍스트 인젝션(context injection) 기법을 활용하여 이들 모델의 편향을 줄이는 새로운 알고리즘을 제안합니다.

- **Technical Details**: 저자들은 두 가지 기존 기법인 기하학적 투영(geometric projection)과 WEAT(Word Embedding Association Test)를 수정하여 19개 임베딩 모델의 편향을 정량화합니다. 각 모델은 서로 다른 강도와 부분에서 편향(예: 성별, 나이)에 따라 평가받습니다. 또한 컨텍스트를 주입하여 편향을 감소시키는 방법론을 이용해 모델의 반응성을 측정합니다.

- **Performance Highlights**: 결과적으로 성능이 높은 임베딩 모델은 일반적으로 더 많은 편향을 캡처하는 경향이 있지만, 컨텍스트를 포함할 경우 편향을 줄이는 데 더 잘 대응한다고 밝혀졌습니다. 본 연구에서 제안하는 새로운 알고리즘은 동적으로 선택된 k 값에 대해 효과적인 검색 결과를 제공할 수 있습니다.



### Beyond Right and Wrong: Mitigating Cold Start in Knowledge Tracing Using Large Language Model and Option Weigh (https://arxiv.org/abs/2410.12872)
Comments:
          11 pages

- **What's New**: 이 논문에서는 LOKT 모델을 소개하여 Knowledge Tracing (KT)의 콜드 스타트 문제를 해결합니다. LOKT는 대규모 언어모델(LLM)을 사용하여 적은 이전 데이터로도 학습자의 지식 상태를 추적하고 예측할 수 있는 방법론을 제시합니다.

- **Technical Details**: LOKT 모델은 전통적인 KT 모델에 옵션 가중치를 통합하여 단순한 정답/오답 분류를 넘어 학습자의 다양한 잘못된 응답을 분석합니다. 이를 통해 LLM이 언어 기반 정량적 정보를 활용하여 학습자의 이해도를 보다 정확하게 평가할 수 있도록 합니다.

- **Performance Highlights**: 5개의 공공 데이터셋을 사용한 실험에서 LOKT 모델은 이른 단계의 개인화 학습 도구를 지원하며, '학습자 콜드 스타트'와 '시스템 콜드 스타트' 상황에서도 높은 예측 정확도를 유지하는 것을 보여주었습니다.



### Skill Learning Using Process Mining for Large Language Model Plan Generation (https://arxiv.org/abs/2410.12870)
Comments:
          12 pages, 5 figures, 2 tables, accepted at ICPM 2024'

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 계획 생성을 개선하기 위해 프로세스 마이닝( process mining ) 기법을 통합한 새로운 기술 학습 접근 방식을 소개합니다. 이 접근 방식은 계획 생성 과정의 효율성과 해석 가능성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 텍스트 베이스 LLM 플래너가 생성한 단순 시퀀스 대신, 프로세스 모델을 사용하여 구조화된 제어 흐름을 만들고 이를 통해 플래너의 능력을 향상시키는 방법을 제안합니다. 새로운 기술 학습 프레임워크에서는 Inductive Miner 알고리즘을 사용하여 일반적인 프로세스 모델을 추출합니다.

- **Performance Highlights**: 실험 결과, 제안한 기술 검색 방법이 특정 조건에서 기존의 정확도 기준을 초과하는 것으로 나타났으며, 유연한 기술 발견과 병렬 실행을 지원하여 성능이 향상되었습니다.



### Language Model Preference Evaluation with Multiple Weak Evaluators (https://arxiv.org/abs/2410.12869)
- **What's New**: 이 논문에서는 효율적인 평가 방식의 필요성을 강조하며 신뢰성 있는 LLM(대규모 언어 모델) 출력 평가를 위한 새로운 방법론인 GED(Preference Graph Ensemble and Denoise)를 소개합니다.

- **Technical Details**: GED는 두 가지 주요 단계로 구성됩니다: (1) 여러 LLM의 평가 결과를 통합하여 단일 preference graph(선호 그래프)를 만드는 graph ensemble과 (2) 반복적 패턴과 불일치를 제거하여 방향 비순환 그래프(DAG) 구조를 보장하는 graph denoising입니다.

- **Performance Highlights**: GED는 실험 결과에서 10개 벤치마크 데이터셋을 통해 기존 방법들보다 우수한 성능을 보였으며, 예를 들어, 응답 선택 작업에서 평균 4.51% 향상을 기록했습니다. GED는 약한 평가자(combiner) 조합을 통해 강한 평가자보다 뛰어난 성능을 보여, 평가 신뢰성을 높이고 모델 성능을 향상시키는 능력을 입증했습니다.



### Empowering Dysarthric Speech: Leveraging Advanced LLMs for Accurate Speech Correction and Multimodal Emotion Analysis (https://arxiv.org/abs/2410.12867)
Comments:
          19 pages, 6 figures, 3 tables

- **What's New**: 이번 논문은 뇌 손상으로 인해 발생하는 운동 언어 장애인 발음장애(dysarthria)의 인식 및 번역에 대한 새로운 접근 방식을 제시합니다. 이 연구는 발음장애를 가진 개인들이 보다 효과적으로 소통할 수 있도록 지원하기 위해 고급 언어 모델(large language models)을 활용합니다.

- **Technical Details**: 이 연구에서는 OpenAI Whisper 모델을 사용하여 발음장애의 음성을 텍스트로 변환한 후, LLaMA 3.1(70B) 및 Mistral 8x7B와 같은 모델을 미세 조정하여 왜곡된 입력으로부터 의도된 문장을 예측합니다. 데이터 세트는 TORGO 데이터 세트와 Google 음성 데이터를 결합하였으며, 감정 컨텍스트를 수작업으로 라벨링하여 모델 학습에 사용합니다.

- **Performance Highlights**: 제안된 시스템은 발음장애의 음성을 재구성하고 감정을 인식하는 데 있어 높은 정확도를 달성하며, 이는 실질적인 음성 데이터와 비교했을 때 눈에 띄는 발전을 보여줍니다. 이 접근 방식은 발음장애 사용자를 위한 보다 포괄적이며 효과적인 커뮤니케이션 지원 도구를 제공합니다.



### Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings (https://arxiv.org/abs/2410.12866)
Comments:
          Preprint V1 with 10 pages main text

- **What's New**: 최근 뇌-컴퓨터 인터페이스(BCI)의 발전으로 인해 두개내(recordings)에서 음조(lexical tones)를 해독하는 것이 가능해졌습니다. 이는 언어 손상으로 인해 의사소통 능력이 제한된 사람들에게 도움을 줄 수 있는 잠재력을 제공합니다. 하지만 생리적 및 기기적 요소로 인해 발생하는 데이터 이질성(data heterogeneity)은 통합적인 뇌 음조 해독에 상당한 도전 과제가 됩니다.

- **Technical Details**: 이 논문에서는 H2DiLR(Homogeneity-Heterogeneity Disentangled Learning for neural Representations)이라는 새로운 프레임워크를 도입하여, 여러 피험자의 두개내 기록에서 동질성과 이질성을 분리하고 학습합니다. 이 연구에서는 407개의 음절(syllables)을 포함하는 중국어 재료를 읽는 여러 참가자로부터 스테레오전자뇌전도(sEEG) 데이터를 수집했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 H2DiLR은 기존의 이질적인 해독 접근 방식보다 현저히 우수한 성능을 보임을 입증했습니다. 또한 H2DiLR이 신경 표현 학습 과정에서 동질성과 이질성을 효과적으로 포착함을 실증적으로 확인하였습니다.



### ELF-Gym: Evaluating Large Language Models Generated Features for Tabular Prediction (https://arxiv.org/abs/2410.12865)
- **What's New**: ELF-Gym 프레임워크를 통해 LLM이 생성한 feature의 품질을 정량적으로 평가하는 새로운 방법론을 제시합니다.

- **Technical Details**: ELF-Gym은 Kaggle 대회에서 수집한 251개의 'golden' features를 기준으로 LLMs의 feature 엔지니어링 능력을 평가합니다. 평가 과정에서 LLM이 생성한 features의 다운스트림 모델 성능과 전문가가 제작한 features와의 의미적, 기능적 유사성을 측정합니다.

- **Performance Highlights**: 최선의 경우, LLM은 'golden' features의 약 56%를 의미적으로 포착할 수 있지만, 복잡한 feature가 요구되는 데이터셋에서는 실패할 수도 있습니다.



### Investigating Implicit Bias in Large Language Models: A Large-Scale Study of Over 50 LLMs (https://arxiv.org/abs/2410.12864)
- **What's New**: 이번 연구에서는 최신 대규모 언어 모델(LLM)들이 내재된 편향(implicit bias)을 가지고 있으며, 이러한 편향이 개발된 모델의 크기나 복잡성이 증가함에 따라 강화되고 있다는 점을 강조합니다. 또한, 편향 완화(bias mitigation)가 모델 개발에서 보편적으로 우선시되지 않고 있다는 사실을 강조합니다.

- **Technical Details**: 연구진은 50개 이상의 LLM을 대상으로 LLM Implicit Association Test (IAT) Bias 및 LLM Decision Bias 측정을 통해 내재된 편향의 정도를 탐구했습니다. 이 연구는 대규모 실험을 통해 신 모델에서 더 높은 편향 수준을 관찰하였으며, 이는 합성 데이터의 사용 증가와 관련이 있을 수 있다고 가정합니다.

- **Performance Highlights**: 새로운 또는 더 큰 언어 모델들이 자동으로 편향 수준이 감소하지 않으며, 때때로 이전 모델들보다 높은 편향 점수를 나타내기도 했습니다. 이러한 발견은 공정하고 책임감 있는 AI 시스템 개발을 위한 편향 탐지 및 완화 전략의 필요성을 강조합니다.



### Enhancing Affinity Propagation for Improved Public Sentiment Insights (https://arxiv.org/abs/2410.12862)
- **What's New**: 이 연구는 감독 학습(supervised learning)에 의존하지 않고 감정 분석(sentiment analysis)을 수행하기 위한 비감독 학습(unsupervised learning) 기술을 도입합니다. 특히 Affinity Propagation (AP) 클러스터링 기법을 사용합니다.

- **Technical Details**: AP 클러스터링은 사전 정의된 클러스터 수 없이 텍스트 데이터를 자연적인 패턴에 따라 그룹화합니다. 이 논문에서는 텍스트 표현을 위한 TF-IDF 벡터화(TF-IDF Vectorization)와 차원 축소(principal component analysis, PCA) 기법을 사용하여 AP 클러스터링과 K-평균 클러스터링(K-means clustering)을 비교합니다. AP는 Agglomerative Hierarchical Clustering과 결합하여 성능을 향상시킵니다.

- **Performance Highlights**: AP와 Agglomerative Hierarchical Clustering의 조합이 K-평균보다 현저히 더 우수한 성능을 보였으며, 실험 평가는 Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Index를 통해 진행되었습니다. 이 연구는 널리 사용되는 레이블 데이터에 대한 필요 없이 대중 감정을 분석할 수 있는 스케일 가능하고 효율적인 비감독 학습 프레임워크를 제안하여 자연어 처리(NLP) 분야에 기여합니다.



### Scaled and Inter-token Relation Enhanced Transformer for Sample-restricted Residential NILM (https://arxiv.org/abs/2410.12861)
Comments:
          Submitted to 27th IEEE-ICCIT

- **What's New**: 이 논문은 Non-Intrusive Load Monitoring (NILM)에서 transformer 모델의 훈련을 개선하기 위한 새로운 두 가지 메커니즘을 제안합니다. 이는 작은 규모의 데이터셋에서 transformer의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 제안된 두 가지 메커니즘은 inter-token relation enhancement mechanism과 dynamic temperature tuning mechanism입니다. 첫 번째 메커니즘은 훈련 중에 토큰 유사성 행렬에서 intra-token의 중요도를 줄이고 inter-token에 집중도를 높입니다. 두 번째 메커니즘은 토큰 유사성 행렬에 대해 학습 가능한 온도 조정을 도입하여 고정 온도 값에 수반되는 과도한 smoothing 문제를 완화합니다.

- **Performance Highlights**: REDD 주거용 NILM 데이터셋을 사용한 실험 결과, 제안된 방법이 원래 transformer 모델보다 여러 가전 제품 유형에서 성능을 현저히 향상시키는 것으로 나타났습니다.



### LLMD: A Large Language Model for Interpreting Longitudinal Medical Records (https://arxiv.org/abs/2410.12860)
- **What's New**: LLMD는 환자의 의료 기록을 기반으로 의료 이력을 분석하도록 설계된 대규모 언어 모델이며, 의료 지식과 레이블이 지정된 장기 기록을 결합하여 정확한 환자 건강 정보를 제공한다.

- **Technical Details**: LLMD는 10년 이상의 치료 기록과 140개 이상의 치료 기관에서 수집된 대량의 데이터를 포함하여, 지속적인 프리트레이닝(pretraining)과 작업 기반 지침 세밀 조정(instruction fine-tuning)을 통해 훈련된다. 이 구조화(structuring) 및 추상화(abstraction) 작업은 의료 기록의 메타데이터와 임상명명 엔티티(clinical named-entities)를 식별하고 정규화하여 높은 수준의 표현으로 변환한다.

- **Performance Highlights**: LLMD-8B는 PubMedQA 텍스트 응답에서 최첨단 정확도를 달성하며, 기존의 크고 일반화된 모델 및 도메인 맞춤형 모델보다 우수한 성능을 보인다. 실제 환자 데이터를 분석할 때, 의료 지식이 아닌 프리트레이닝과 세밀 조정의 중요성을 강조하며 LLM의 의료 활용을 위한 격차에 대해 논의한다.



### Enhancing Long Context Performance in LLMs Through Inner Loop Query Mechanism (https://arxiv.org/abs/2410.12859)
- **What's New**: 이번 논문에서는 Inner Loop Memory Augmented Tree Retrieval (ILM-TR)이라는 혁신적인 접근법을 통해 복잡한 질문에 대한 보다 깊이 있는 답변 생성을 가능하게 하는 새로운 메모리 체계를 도입합니다. 이 메커니즘은 초기 질문뿐만 아니라 중간 결과에 기반한 내부 루프 쿼리를 활용하여 정보를 검색합니다.

- **Technical Details**: ILM-TR 방법은 기본적으로 두 부분으로 구성되어 있습니다: retriever와 inner-loop query. Retriever 부분에서는 RAPTOR의 트리 빌드 방법을 사용하여 원시 데이터를 짧고 연속적인 텍스트 청크로 분할하고, 각 청크의 요약을 생성합니다. Inner-loop 쿼리는 LLM을 사용하여 최종 답변을 생성하며, Short-Term Memory (STM)라는 영역에 정보를 저장하고, 전달된 데이터를 바탕으로 반복적으로 쿼리를 수행합니다.

- **Performance Highlights**: ILM-TR 시스템은 Multi-Needle In A Haystack (M-NIAH) 및 BABILong과 같은 표준 긴 컨텍스트 벤치마크에서 기존 RAG 방법을 초월하는 성능을 보여주며, 500k tokens까지 컨텍스트 길이가 증가해도 성능 저하 없이 지속적인 성능을 유지합니다.



### Large Language Models for Medical OSCE Assessment: A Novel Approach to Transcript Analysis (https://arxiv.org/abs/2410.12858)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 의료 기초 교육 과정에서의 학생의 의사소통 능력을 평가하는 가능성을 탐구하였습니다. 기존의 수작업 평가 방식에 비해 시간과 비용을 절감할 수 있는 자동화된 OSCE 평가 시스템을 제안합니다.

- **Technical Details**: 연구에서 2,027개의 OSCE 비디오 데이터를 활용하여 학생의 환자 의료 정보 요약 능력을 평가하였습니다. Whisper-v3를 사용하여 음성을 텍스트로 변환한 후, GPT-4를 포함한 다양한 LLM 기반 접근 방식을 통해 학생의 성과를 채점하였습니다. 연구에서는 zero-shot prompting, retrieval augmented generation 및 다중 모달 앙상블 기법을 적용하였습니다.

- **Performance Highlights**: GPT-4는 인간 채점자와의 코헨 카파(Cohen's kappa) 지수 0.88을 기록하여 LLM 기반 OSCE 채점의 가능성을 보여주었습니다. 오픈 소스 모델 또한 유망한 결과를 보였으며, 자동 채점 시스템의 구현 가능성을 제시하였습니다.



### Enterprise Benchmarks for Large Language Model Evaluation (https://arxiv.org/abs/2410.12857)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 평가를 위한 새로운 벤치마크를 제시합니다. 이는 금융 서비스, 법률, 사이버 보안 및 기후 변화와 지속 가능성과 같은 다양한 기업 도메인에서의 NLP 작업을 포함하는 25개의 공개 데이터셋을 활용합니다.

- **Technical Details**: 본 연구에서는 LLM 평가를 위한 프레임워크를 개발하여, 각 도메인에 맞는 성능 지표와 벤치마크를 제공합니다. 이 프레임워크는 Stanford의 HELM을 보강하여, 도메인별로 구체화된 벤치마크를 추가하고 이를 통해 LLM의 성능을 측정할 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: 13개의 모델을 다양한 기업 작업에 적용하여 성능을 평가한 결과, 특정 작업의 요구사항에 맞는 모델 선택의 중요성이 드러났습니다. 이 연구는 실질적인 기업 애플리케이션의 요구를 반영한 벤치마크와 평가 메트릭을 통해 LLM의 최적화를 도울 것으로 기대됩니다.



### Optimized Biomedical Question-Answering Services with LLM and Multi-BERT Integration (https://arxiv.org/abs/2410.12856)
Comments:
          10 pages, 12 figures, accepted and to be published in the proceedings of 2024 IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)과 Multi-BERT 구성을 통합하여 생물의학적 질문-응답(QA) 서비스를 개선하는 정교한 접근 방식을 제안합니다. 이 시스템은 복잡한 생물의학 데이터의 방대한 양을 처리하고 우선 순위를 매기는 능력을 향상시켜 의료 전문가들이 더 나은 환자 결과 및 정보에 기반한 의사 결정을 내릴 수 있도록 지원하는 것을 목표로 합니다.

- **Technical Details**: BERT(Bidirectional Encoder Representations from Transformers) 및 BioBERT 모델의 혁신적인 사용과 다층 퍼셉트론(MLP) 레이어의 결합을 통해, 의료 부문의 증가하는 요구에 대해 보다 전문화되고 효율적인 응답을 제공합니다. 이 접근 방식은 과적합(overfitting) 문제를 해결하기 위해 하나의 BERT 모델을 동결(freeze)하면서 다른 모델을 훈련(training)하는 방법을 사용하여 QA 서비스의 전반적인 적응성을 개선합니다.

- **Performance Highlights**: BioASQ 및 BioMRC와 같은 대규모 데이터셋을 사용하여 QA 서비스 성능의 주요 지표에서 상당한 개선을 나타내는 것을 입증합니다. 이 작업은 고급 언어 모델이 의료 분야에서 실질적인 차이를 만들 수 있는 방법을 강조하며, 복잡한 정보를 관리하는 전문가들을 위해 신뢰할 수 있고 반응성이 뛰어난 도구를 제공합니다.



### JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework (https://arxiv.org/abs/2410.12855)
- **What's New**: 이번 연구에서는 jailbreak 공격에 대한 LLMs의 방어력 평가를 위한 새로운 벤치마크인 JAILJUDGE를 제안합니다. 이 벤치마크는 다양한 리스크 시나리오를 포함하고 있으며, 고품질의 인간 주석이 포함된 데이터셋으로 구성되어 있습니다.

- **Technical Details**: JAILJUDGE 데이터셋은 35k 이상의 instruction-tune 데이터를 포함하며, JailJudge MultiAgent 프레임워크를 통해 명시적 reasoning(추론)을 바탕으로 한 세밀한 평가가 가능합니다. JAILJUDGE Guard는 instruction-tuning된 종합적인 평가 모델로 비용 없이 reasonability 설명을 제공합니다.

- **Performance Highlights**: JailJudge 메소드의 성능은 다양한 모델(GPT-4, Llama-Guard 등)에서 최첨단을 나타냅니다. JailBoost는 성능을 29.24% 향상시켰고, GuardShield는 방어 ASR을 40.46%에서 0.15%로 감소시켰습니다.



### TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees (https://arxiv.org/abs/2410.12854)
- **What's New**: 본 연구에서는 기존의 DPO(Direct Preference Optimization) 알고리즘에서 발생하는 한계를 극복하기 위해 TPO(Tree Preference Optimization)를 제안합니다. TPO는 선호 트리(preference tree)에서 대응하는 응답을 샘플링하는 대신, 전체 선호 트리로부터 직접 학습합니다.

- **Technical Details**: TPO는 언어 모델 정렬을 Preference List Ranking 문제로 정의하며, 이는 주어진 프롬프트에 대한 응답의 순위가 매겨진 선호 리스트로부터 더 효과적으로 학습할 수 있도록 합니다. 또한, Adaptive Step Reward를 사용하여 긴 체인의 추론에서 LLM이 차별화된 단계를 인식하는 데 도움을 주고, 각 단계의 보상 값(reward values)을 조정하여 세밀한 선호 최적화(fine-grained preference optimization)를 수행합니다.

- **Performance Highlights**: TPO는 수학적 추론(task)에서의 실험을 통해 DPO보다 세 가지 공개 대형 언어 모델에 대해 네 개의 데이터셋에서 일관되게 우수한 성능을 보였습니다.



### Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks (https://arxiv.org/abs/2410.12853)
Comments:
          11 pages, 9 figures

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력과 사실 정확성을 개선하기 위한 다중 에이전트 토론(multi-agent debate) 프레임워크를 제안합니다. 특히, 다양한 모델을 활용한 경우에 더 뛰어난 성능을 발휘했으며, GPT-4와 비교하여 더 높은 정확성을 기록하였습니다.

- **Technical Details**: 다중 에이전트 토론 프레임워크는 질문 인코딩, 토론 모델, 토론 라운드, 응답 요약, 반복적 정제 및 최종 요약의 여섯 가지 주요 구성 요소로 이루어져 있습니다. 이 과정에서 다양한 모델 아키텍처를 활용하여 각 모델의 사고 다양성에 기반한 강력한 논리를 생성합니다.

- **Performance Highlights**: 이 연구에서 사용한 중간 용량 모델 세트(Gemini-Pro, Mixtral 7BX8,와 PaLM 2-M)는 4회 토론 후 GSM-8K 벤치마크에서 91%의 정확도를 기록하여 GPT-4를 초월하였고, ASDiv 벤치마크에서는 94%로 새로운 최고 기록을 세웠습니다.



### The Large Language Model GreekLegalRoBERTa (https://arxiv.org/abs/2410.12852)
- **What's New**: 그리스 법률 및 비법률 텍스트에 대해 훈련된 네 가지 버전의 GreekLegalRoBERTa 모델을 개발했습니다. 이 모델은 GreekLegalBERT 및 그리스 관련 다른 모델들의 성능을 초과합니다.

- **Technical Details**: 이 논문에서는 RoBERTa(Liu et al., 2019)를 사용하여 그리스 법률 문서에서의 이름 개체 인식(NER) 및 다중 클래스 법률 주제 분류 작업을 수행했습니다. 훈련된 네 가지 GreekLegalRoBERTa 모델은 Nomothesia 플랫폼, 그리스 의회 의사록, 유럽 의회 의사록 병렬 코퍼스 등에서 수집된 데이터로 훈련되었습니다.

- **Performance Highlights**: 모델들은 GreekLegalNER에서 이전의 모든 모델들을 초과하는 성능을 보였고, GreekLegalCode 작업에서도 개선된 성능을 나타내었습니다. 특히, micro 평균에서 GreekLegalBERT-v2의 성능을 1.2 포인트 개선하였고, 다양한 분류에서 다른 성과도 달성하였습니다.



### VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models (https://arxiv.org/abs/2410.12851)
Comments:
          10 pages, unironic use of the word 'vibe'

- **What's New**: VibeCheck는 대형 언어 모델(LLMs) 간의 뚜렷한 특성(vibes)을 발견하고 측정하는 시스템으로, 모델의 출력에서 다양한 차원(ton, formatting, writing style)을 평가할 수 있는 새로운 방식입니다. 이 시스템은 사용자의 선호와 모델의 정체성을 예측할 수 있는 vibres를 자동으로 확인합니다.

- **Technical Details**: VibeCheck는 모델의 출력을 통해 vibes를 반복적으로 발견하고, LLM 판별자를 통해 각 vibe의 유용성을 정량적으로 측정합니다. 발견된 vibes는 다수의 사용자의 합의, 모델 간 차별화, 사용자 선호 예측 세 가지 기준을 충족해야 합니다. VibeCheck는 Llama-3-70b와 GPT-4의 사용자 대화 데이터를 기반으로 실험했으며, 80%의 정체성 예측 정확도와 61%의 사용자 선호 예측 정확도를 달성했습니다.

- **Performance Highlights**: VibeCheck의 결과에 따르면, Llama는 친근하고 유머러스하며 다소 논란이 있는 vibe를 가지며, Command X는 요약 시 구체적인 서론과 결론을 추가하는 경향이 있고, Llama-405b는 수학 문제에서 자신의 사고 과정을 과도하게 설명하는 경향이 있는 반면, GPT-4는 캡셔닝에서 장면의 정서와 분위기에 집중하는 경향이 있음을 확인했습니다.



### RecurFormer: Not All Transformer Heads Need Self-Attention (https://arxiv.org/abs/2410.12850)
- **What's New**: 이 논문에서는 Transformer 기반의 대형 언어 모델(LLM)의 응답 생성 과정에서 발생하는 계산 비용 문제를 해결하기 위해 RecurFormer라는 새로운 아키텍처를 제안합니다. RecurFormer는 특정 attention head를 linear recurrent neural network (RNN)인 Mamba 아키텍처로 교체하여 메모리 캐시 사이즈를 줄이고, 토큰을 제거하지 않으면서 생성 품질을 유지합니다.

- **Technical Details**: RecurFormer는 recency aware 속성을 가진 attention head를 Mamba 아키텍처로 교체하는 방식으로 구성되어 있습니다. Mamba는 selective structured state-space sequence model 기반의 linear RNN으로, parallel 및 recursive 계산을 지원합니다. 이 방식은 기존 Transformer의 가중치를 계속 활용할 수 있도록 하여 모델의 성능을 유지하면서도 계산 효율을 증대시킵니다.

- **Performance Highlights**: 실험 결과, RecurFormer는 원래 모델의 성능을 유지하면서도 추론 효율성을 크게 향상시키는 것으로 나타났습니다. 또한, 지속적인 훈련을 통해 성능 회복이 가능하다는 것을 보여주어, 긴 입력에 관련된 작업에서 Transformer 기반 LLM의 계산적 도전에 대한 실용적인 해결책을 제공합니다.



### Prompt Engineering a Schizophrenia Chatbot: Utilizing a Multi-Agent Approach for Enhanced Compliance with Prompt Instructions (https://arxiv.org/abs/2410.12848)
- **What's New**: 이 논문은 정신분열증 환자를 위한 교육 플랫폼에서 Large Language Models (LLMs)인 GPT-4를 활용하는 방법을 제안합니다. 특히, 챗봇의 반응이 초기에 설정된 범위를 넘는 경우를 다루기 위해 Critical Analysis Filter를 도입했습니다.

- **Technical Details**: 이 시스템은 여러 LLM 에이전트가 챗봇의 반응을 분석하고 개선하는 역할을 합니다. 실험에서는 정보 제공 목적의 정신분열증 챗봇을 개발하고, 필터가 비활성화된 상태에서 대화를 진행하여 챗봇의 범위를 초과하는 모습을 관찰했습니다. 이후 AI 에이전트를 통해 범위를 벗어난 주제에 대한 샘플 대화를 자동 생성하고, 각 반응에 대해 컴플라이언스 점수를 할당했습니다.

- **Performance Highlights**: Critical Analysis Filter를 활성화했을 때 챗봇의 컴플라이언스 점수는 67.0%에서 적정 수준(점수 >=2)을 유지했지만, 필터가 비활성화된 경우에는 단지 8.7%에 불과했습니다. 이는 정신 건강 플랫폼에서 LLM을 효과적이고 안전하게 사용하기 위한 자기 반성 계층의 필요성을 시사합니다.



### ACCEPT: Adaptive Codebook for Composite and Efficient Prompt Tuning (https://arxiv.org/abs/2410.12847)
Comments:
          EMNLP Finding 2024

- **What's New**: 이 연구에서는 Adaptive Codebook for Composite and Efficient Prompt Tuning (ACCEPT)이라는 새로운 방법을 제안합니다. 기존의 Prompt Tuning (PT) 기법이 개별적으로 업데이트되는 프롬프트로 인해 파라미터 수가 비례적으로 증가하는 문제를 해결하여 모든 소프트 프롬프트가 학습 가능한 코드북 벡터를 공유하도록 하여 파라미터 효율성을 높입니다.

- **Technical Details**: ACCEPT는 제품 양자화(Product Quantization, PQ) 개념을 바탕으로 하며, 각 프롬프트의 단어 임베딩을 여러 하위 섹션으로 나누어 각각의 섹션에 대해 코드북을 구성합니다. 이 방법은 프롬프트의 각 하위 벡터가 선형 계수를 통해 부드럽게 결합되도록 하여 더 높은 다양성과 유연성을 제공합니다. 또한, ACCEPT는 0.3%의 플로우우먼스 파라미터만 조정하여 17개의 자연어 작업에서 우수한 성능을 달성합니다.

- **Performance Highlights**: 17개의 다양한 자연어 작업에서 ACCEPT 방법이 이전의 PT 접근법을 일관되게 초과 달성했습니다. 특히, 몇 가지 샷(few-shot) 및 대형 모델 환경에서 뛰어난 성능을 보여주며, 사전 훈련된 언어 모델(PLMs)의 효율성을 극대화합니다.



### Accurate and Regret-aware Numerical Problem Solver for Tabular Question Answering (https://arxiv.org/abs/2410.12846)
- **What's New**: TabLaP라는 모델을 제안하여, Large Language Model (LLM)을 답변 생성기가 아닌 계획자로 활용하며, 숫자 계산을 위한 정확한 처리기인 Python interpreter에게 계산을 맡깁니다. 또한, TabLaP가 생성한 답변의 신뢰성을 정량화하여 사용자가 후회 유발 가능성을 줄일 수 있도록 합니다.

- **Technical Details**: TabLaP는 두 개의 모델 브랜치를 갖고 있으며, 하나는 NumSolver로 숫자 질문을 처리하고, 다른 하나는 최신 TableQA 모델입니다. 생성된 답변을 통합하기 위해 AnsSelecter라는 LLM을 사용하여 신뢰할 수 있는 브랜치를 선택합니다. TwEvaluator 모듈을 통해 각 브랜치의 정확도를 추적하여 답변 신뢰성을 평가합니다.

- **Performance Highlights**: TabLaP는 두 개의 벤치마크 데이터셋에서 기존의 SOTA 모델에 비해 각각 5.7%와 5.8% 향상된 정확도를 기록했습니다. 또한, TabLaP의 신뢰성 플래그는 사용자 후회 비율을 두 데이터셋에서 각각 30.9%와 20.6% 감소시켰습니다.



### Toward Relieving Clinician Burden by Automatically Generating Progress Notes using Interim Hospital Data (https://arxiv.org/abs/2410.12845)
Comments:
          Accepted at the AMIA 2024 Annual Symposium

- **What's New**: 이 논문에서는 전자 건강 기록(EHR)의 구조화된 정보를 활용하여 진행 노트 생성(Progress Note Generation, PNG) 자동화를 위한 새로운 방법론을 제안합니다. 특히, 1616명의 환자로부터 수집된 7089개의 주석 인스턴스를 포함한 대형 데이터셋 ChartPNG를 소개합니다.

- **Technical Details**: 이 연구는 임상 의사들이 작성한 SOAP 노트를 기반으로 하는 프로세스입니다. 진행 노트는 환자의 주관적 및 객관적 상태와 평가 및 계획(A&P)으로 구성되어 있으며, 연구는 A&P 섹션의 자동 생성을 주로 목표로 합니다. 이 과정에서 대형 언어 모델을 활용하여 자동 분석을 수행하고, 향후 연구 기회를 찾아내기 위해 오류 분석을 실시하였습니다.

- **Performance Highlights**: 자동화된 분석에서는 Biomistral 모델이 BERTScore F1 점수 80.53과 MEDCON 점수 19.61을 기록하였고, 수작업 분석에서는 76.9%의 정확도로 관련 구조화 데이터를 활용할 수 있음을 보여주었습니다.



### TextLap: Customizing Language Models for Text-to-Layout Planning (https://arxiv.org/abs/2410.12844)
Comments:
          Accepted to the EMNLP Findings

- **What's New**: 이 논문에서는 사용자가 텍스트 지시만으로 매력적인 그래픽 레이아웃을 생성할 수 있도록 돕는 새로운 방법인 TextLap을 제안합니다. TextLap은 특별히 설계된 레이아웃 계획 데이터셋인 InstLap을 활용하여 대형 언어 모델(LLM)을 사용자 맞춤형 그래픽 디자이너로 변환합니다.

- **Technical Details**: TextLap 모델은 레이아웃 생성을 위한 text-to-layout 작업을 수행합니다. 사용자 입력에 따라 레이아웃을 생성하고 수정할 수 있으며, 이는 자연어 대화를 통해 이루어집니다. InstLap 데이터셋은 이미지-캡션 페어를 필터링하고 향상하여 LLM에 대한 사용자 지시 튜닝 데이터를 제공합니다.

- **Performance Highlights**: 텍스트 기반 레이아웃 계획인 TextLap은 다양한 벤치마크 데이터셋에서 평가된 결과, GPT-4 기반의 방법보다 우수한 성능을 나타냈습니다. TextLap은 디자인 생성에 필요한 시간을 줄이고, 디자이너의 작업 효율성을 향상시키는 데 기여합니다.



### Exploring Prompt Engineering: A Systematic Review with SWOT Analysis (https://arxiv.org/abs/2410.12843)
Comments:
          14 pages, 1 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM) 내에서 프롬프트 엔지니어링(prompt engineering) 기술에 대한 포괄적인 SWOT 분석을 수행했습니다. 언어학 원칙을 강조하며, 다양한 기술들을 분석하여 그 강점, 약점, 기회 및 위협을 파악했습니다. 이러한 발견은 AI 상호작용을 향상시키고 언어 모델이 인간의 프롬프트를 이해하는 방법을 개선하는 데 기여합니다.

- **Technical Details**: 이 논문에서는 100편 이상의 관련 문헌을 조사하여 프롬프트 엔지니어링 분야에 대한 폭넓은 통찰을 제공합니다. 주요 프롬프트 엔지니어링 기술로는 템플릿 기반 접근법(template-based approaches)과 파인 튜닝(fine-tuning) 방식이 있으며, 각 기술의 문제점 및 도전 과제를 다루었습니다. 또한, BLEU, BERTScore, ROUGE 및 Perplexity와 같은 여러 평가 메트릭(metrics)을 확인했습니다. 연구는 언어 모델의 행동을 이해하는 데 도움을 주고, 맞춤형 상호작용을 제공하는 목표에 맞춰 진행되었습니다.

- **Performance Highlights**: 이 연구는 LLM의 정확성 및 관련성을 향상시킬 수 있는 효과적인 프롬프트 엔지니어링의 중요성을 강조하며, 사용자 및 개발자 간의 지식 공유와 대화형 AI 툴의 발전을 촉진합니다. 특히, 차별화된 접근법을 통해 응답 정확도를 높이고, 대화형 AI의 성장에 기여할 것입니다.



### A Two-Model Approach for Humour Style Recognition (https://arxiv.org/abs/2410.12842)
- **What's New**: 이번 연구에서는 1,463개의 인스턴스를 포함하는 새로운 텍스트 데이터셋을 도입하여 네 가지 유머 스타일(자기 증진, 자기 비하, 친화적, 공격적) 및 비유머 텍스트를 인식하는 데 필요한 기계 학습 모델링을 지원합니다. 이는 유머 스타일 인식의 연구 공백을 채우는 중요한 기여를 합니다.

- **Technical Details**: 연구에서는 고전 기계 학습 분류기, 텍스트 임베딩 모델 및 DistilBERT를 포함한 다양한 컴퓨팅 방법을 사용하여 기준 성능을 설정하였습니다. 또한, 친화적 유머 분류의 F1 점수를 11.61% 향상시키는 두 개의 모델 접근 방식을 제안하였습니다. 이 연구는 각 유머 스타일에 대한 다중 클래스 분류 문제를 다룹니다.

- **Performance Highlights**: 두 개의 모델 접근 방식을 통해 14개의 테스트된 모델에서 일관된 성능 개선을 보였으며, 특히 친화적 유머 분류에서 F1 점수의 11.61% 향상을 달성했습니다. 이는 문학, 소셜 미디어 및 다른 텍스트 출처에서 유머를 연구하기 위한 새로운 도구를 제공합니다.



### UniAutoML: A Human-Centered Framework for Unified Discriminative and Generative AutoML with Large Language Models (https://arxiv.org/abs/2410.12841)
Comments:
          24 pages

- **What's New**: 새로운 AutoML 프레임워크인 UniAutoML이 소개되었습니다. UniAutoML은 기존의 AutoML 프레임워크가 주로 다루었던 discriminative task 뿐만 아니라 generative task도 통합하여 지원하는 것이 특징입니다. 사용자가 쉽게 접근할 수 있도록 자연어로 상호작용할 수 있는 대화형 사용자 인터페이스(CUI)를 제공합니다.

- **Technical Details**: UniAutoML은 Large Language Models (LLMs)를 활용하여 데이터 처리, 모델 선택 및 하이퍼파라미터 검색을 자동화한 인공지능 프레임워크입니다. 사용자들은 자연어 명령을 통해 복잡한 모델을 fine-tuning 할 수 있으며, 모델은 HuggingFace에서 사전 훈련된 다양한 모델을 선택하고 사용할 수 있습니다. 또한, safety guard-line을 설계하여 사용자 입력과 LLM 출력의 필터링이 이루어집니다.

- **Performance Highlights**: UniAutoML은 25명의 참가자를 대상으로 8개의 다양한 데이터셋에 대한 실험을 통해 성능과 사용성을 평가하였고, 그 결과 사용자 제어와 신뢰도를 향상시켰습니다. UniAutoML의 인간 중심 디자인은 AutoML의 기능과 사용자 이해 사이의 격차를 해소하여 더 많은 사람들이 ML(Machine Learning)에 접근할 수 있도록 합니다.



### Answering Questions in Stages: Prompt Chaining for Contract QA (https://arxiv.org/abs/2410.12840)
- **What's New**: 이번 연구에서는 법률 문서에서의 질문에 대한 구조적 답변 생성을 위한 새로운 두 단계 프롬프트 체인을 제안합니다. 이전의 프롬프트가 긴 조항을 다루는 데 한계를 보였던 반면, 이 방식은 더 복잡한 법률 텍스트를 효과적으로 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 법률 관련 질문에 대한 응답을 두 단계로 처리하는 전략을 사용하는데, 첫 번째 단계에서는 관련 법률 텍스트의 요약을 생성하고, 두 번째 단계에서는 이 요약을 사용하여 기존의 프롬프트 템플릿에 대해 질문에 대한 답변을 형성합니다. 이를 통해 질문과 답변 옵션 간의 매핑을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 두 단계 프롬프트 체인이 단순한 프롬프트에 비해 대부분의 경우 더 효과적임을 보여주었습니다. 이는 법률 전문가들이 문서를 더 효율적으로 검토하고 자동화된 워크플로우 및 데이터 파이프라인을 구축할 수 있도록 도와주는 기회를 제공합니다.



### Capturing Bias Diversity in LLMs (https://arxiv.org/abs/2410.12839)
Comments:
          2nd International Conference on Foundation and Large Language Models (FLLM2024), 26-29 November, 2024 | Dubai, UAE

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 출력 다양성을 높이기 위해 여러 개의 사용자 정의 GPT 모델을 구성하여 BiasGPT라는 새로운 프레임워크를 제안하고 평가합니다. 이 모델들은 성별, 연령 및 인종 같은 특정 인구통계학적 특성의 편향을 반영하여 협력하고, 다양한 관점을 통합하여 인간의 경험을 보다 잘 캡처한 응답을 생성합니다.

- **Technical Details**: BiasGPT는 여러 개의 사용자 정의 GPT 모델을 사용하여 각 모델이 특정 인구 통계적 특성을 반영함으로써 다양한 응답을 생성하는 방법입니다. 이 방법론은 사용자 정의된 LLM을 통해 학습된 편향들이 통합되어 보다 포괄적이고 공정한 AI 챗봇 응답을 형성하도록 합니다. 또한, 논문에서는 대화 데이터 수집 과정에서 연령, 인종, 성별 기반의 다양한 편향을 다루기 위한 포괄적인 접근 방식을 사용합니다.

- **Performance Highlights**: 일련의 실험을 통해 BiasGPT는 다양한 사회적 특성을 반영한 응답을 생성할 수 있는 능력을 입증하였으며, 이는 더욱 포괄적이고 대표적인 AI 대화를 형성하는 데 기여할 것입니다. 이 연구는 AI 기술의 포용성을 높이는 방향으로 나아가는 데 주요한 실험적 근거를 제공합니다.



### A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions (https://arxiv.org/abs/2410.12837)
Comments:
          4 Figures

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG)의 발전 과정을 포괄적으로 조사하며, 기존 개념에서 최신 기술에 이르기까지의 변화를 설명합니다. RAG는 검색 메커니즘과 생성 언어 모델을 결합하여 출력의 정확성을 높이며, LLMs의 주요 제한 사항을 해결합니다.

- **Technical Details**: RAG의 기본 아키텍처는 지식 집약적인 작업을 처리하기 위해 검색과 생성을 어떻게 통합하는지에 중점을 둡니다. 논문에서는 retrieval-augmented language models에서의 주요 혁신과 질문 답변, 요약 및 지식 기반 작업 등 다양한 도메인에서의 응용 사례를 자세히 리뷰합니다.

- **Performance Highlights**: 최근 연구 성과는 retrieval 효율성을 개선하기 위한 새로운 방법을 강조하고 있으며, RAG의 연구 방향으로는 모델의 견고성 향상, RAG 모델의 적용 범위 확대 및 사회적 함의 문제 다루기가 제안됩니다.



### How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs (https://arxiv.org/abs/2410.13857)
- **What's New**: 이 논문에서는 Transformer 기반 대형 언어 모델(LLMs)의 수학적 능력을 이론적으로 분석하고, 특히 산술작업에서의 성능을 강조합니다. 숫자 정밀도가 수학적 작업의 성공적인 수행을 좌우하는 핵심 요소로 밝혀졌습니다.

- **Technical Details**: 저자들은 LLM의 기본 산술 작업인 정수 덧셈, 반복 덧셈, 정수 곱셈을 분석합니다. 저자들은 정밀도에 따라 모델의 크기가 달라지며, 낮은 정밀도(int8, int4)의 Transformer는 문제를 풀기 위해 폭발적으로 큰 모델을 요구한다고 주장합니다. 이와 대조적으로 표준 정밀도(float32)는 훨씬 작고 효율적인 모델로도 이를 처리할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과는 두 가지 정밀도(int4 및 표준 정밀도) 모두에서 정수 덧셈 작업에서 충분한 성능을 보였지만, 반복 덧셈 및 정수 곱셈과 같은 복잡한 작업에서는 낮은 정밀도가 성능 저하를 일으킨다는 것을 보여주었습니다.



### Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2410.13848)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 다양한 모드의 이해 및 생성을 통합한 새로운 자율 회귀 프레임워크인 Janus를 소개합니다. 기존 연구는 주로 단일 시각 인코더를 사용했으나, Janus는 시각 인코딩을 별도의 경로로 분리하여 성능과 유연성을 향상시켰습니다.

- **Technical Details**: Janus는 고유한 transformer 아키텍처를 사용하여 시각 이해 및 생성을 위한 독립적인 인코딩 경로를 제공합니다. 이를 통해 이해와 생성 작업 사이의 정보를 분리하고, 각 작업에 가장 적합한 인코딩 방법을 선택할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: Janus는 기존의 통합 모델보다 뛰어난 성능을 보여주며, MMBench 및 SEED-Bench와 같은 벤치마크에서 최고 성과를 기록했습니다. 또한, DALL-E 2와 SDXL과 같은 특정 작업 모델을 초월하는 성과를 보였습니다.



### A Unified View of Delta Parameter Editing in Post-Trained Large-Scale Models (https://arxiv.org/abs/2410.13841)
- **What's New**: 이 논문은 delta parameters(델타 파라미터)의 편집 작업을 Riemann sum(리만 합) 근사를 통해 체계적으로 이해하고 분류하는 새로운 관점을 제시합니다. 기존의 편집 기법들이 어떻게 모델 성능에 영향을 미치는지를 설명하는 통합된 프레임워크를 수립했습니다.

- **Technical Details**: 델타 파라미터는 사전 훈련(pre-trained) 모델과 후 훈련(post-trained) 모델의 파라미터 차이를 나타냅니다. 저자들은 delta parameters의 편집 연산을 Riemann sum approximation(리만 합 근사)을 기반으로 설명하며, 수행된 편집으로 인한 손실 변화를 수학적으로 분석하고 competitive, decreased, improved 성능을 가진 기법으로 분류합니다.

- **Performance Highlights**: 많은 실험을 통해 ViT, LLaMA 3, Qwen 2, Mistral 등 다양한 모델에서 저자들의 이론적인 발견을 뒷받침합니다. DARE와 BitDelta와 같은 기존 기법에서의 성능 향상과 저하를 정량적으로 검증하였으며, 기존의 delta parameter 조정 기술의 한계를 지적하고, 보다 일반화된 접근 방식을 제공하는 확장들을 제안합니다.



### A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglemen (https://arxiv.org/abs/2410.13828)
- **What's New**: 본 논문에서는 Reinforcement Learning from Human Feedback (RLHF)에서 전통적인 margin-based 손실을 사용하는 것의 문제점을 다루고 있습니다. 특히, 이 접근 방법이 선호 및 비선호 응답 각각에 대해 이상적인 언어 모델 behavior를 충분히 명시하지 않는다는 점이 강조됩니다.

- **Technical Details**: 우리는 margin의 증가에 따른 두 가지 의도치 않은 결과를 식별했습니다: (1) 비선호 응답의 확률이 증가할 수 있으며 이는 안전 문제와 관련된 alignment 실패를 초래할 수 있습니다. (2) 선호 응답의 확률이 감소할 수 있으며, 이 경우에도 그 응답은 이상적일 수 있습니다. 이러한 현상의 원인은 gradient entanglement으로 명명하였으며, 이는 선호 및 비선호 응답의 확률 변화가 서로 얽혀 있는 문제를 나타냅니다.

- **Performance Highlights**: 본 논문은 margin 기반 preference optimization 알고리즘의 훈련 동역학을 설명하고, margin 기반 방법의 under-specification 문제를 완화할 수 있는 잠재적인 알고리즘 설계를 제안합니다.



### AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents (https://arxiv.org/abs/2410.13825)
- **What's New**: 이 연구는 LLM(대형 언어 모델)을 기반으로 한 웹 에이전트를 개선하기 위한 혁신적인 접근 방식을 제안합니다. 구체적으로, 에이전트의 관찰(object) 및 행동(action) 공간을 정제하여 LLM의 능력에 더욱 잘 부합하도록 합니다.

- **Technical Details**: 제안된 방법은 세 가지 구성 요소로 이루어져 있습니다: 1) 필수적이지 않은 행동을 줄여 에이전트의 기능을 단순화; 2) 중복되거나 불필요한 웹 요소를 제거하여 관찰을 개선; 3) 'branch' 및 'prune'와 같은 두 가지 계획 행동을 도입하여 에이전트의 내비게이션 흐름을 자기 조직화 합니다.

- **Performance Highlights**: AgentOccam는 WebArena 벤치마크에서 기존의 최첨단 방법보다 9.8 포인트 (+29.4%) 향상된 성능을 보이고, 유사한 일반 웹 에이전트에 비해 성공률을 26.6 포인트 (+161%) 증가시켰습니다. 이 모든 것을 추가적인 맥락 예제, 온라인 피드백 또는 검색 전략 없이 달성했습니다.



### Harnessing Webpage UIs for Text-Rich Visual Understanding (https://arxiv.org/abs/2410.13824)
- **What's New**: 이번 연구에서는 웹 페이지 UI에서 일반 다중 모달 지침을 합성하여 MLLM(다중 모달 대형 언어 모델)의 텍스트가 풍부한 시각적 이해(text-rich visual understanding) 능력을 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 730만 개 샘플로 구성된 MultiUI 데이터셋을 활용하며, 이는 100만 개 웹사이트에서 수집되었습니다. 텍스트 기반 대형 언어 모델(LLM)은 웹페이지 접근성 트리에서 구조적 텍스트 표현을 처리하여 다중 모달 모델을 교육하는 데 필요한 지침을 생성합니다.

- **Performance Highlights**: MultiUI로 훈련된 모델은 웹 UI 작업에서 VisualWebBench에서 최대 48%의 개선을 보였으며, Mind2Web 데이터셋에서 액션 정확도가 19.1% 향상되었습니다. 더 나아가 이 모델은 비웹 UI 작업과 문서 이해, OCR, 차트 해석과 같은 비 UI 도메인에서도 놀라운 일반화를 보여주었습니다.



### Optimal Quantization for Matrix Multiplication (https://arxiv.org/abs/2410.13780)
- **What's New**: 본 연구는 대규모 매트릭스의 lossy compression (양자화) 기법을 통해 매트릭스 곱셈을 가속화하기 위한 새로운 알고리즘을 제안합니다. 이 접근법은 전통적인 벡터 양자화와 다르게, 매트릭스 자체가 아니라 매트릭스 곱셈의 근사를 목표로 합니다.

- **Technical Details**: 이 논문은 iid Gaussian 아이템을 가진 매트릭스의 평균 제곱 오차에 대한 비비대칭 하한을 제공하며, 특정한 프레임워크에서 Frobenius norms를 사용하여 매트릭스 A, B의 압축과 동시에 근사 오차를 보장하는 보편적인 양자기를 제안합니다. 이는 깊은 신경망(Deep Neural Networks)과 대규모 언어 모델(Large Language Models)에서 메모리 대역폭의 병목 현상을 해결하기 위한 중요성을 강조합니다.

- **Performance Highlights**: 제안된 양자기는 최적 성능에 근접한 결과를 실현하며, 정보 이론적으로 iid Gaussian 매트릭스의 매트릭스 곱셈에 대한 rate-distortion function을 도출합니다.



### MobA: A Two-Level Agent System for Efficient Mobile Task Automation (https://arxiv.org/abs/2410.13757)
Comments:
          27 pages, 6 figures, and 5 tables. We will release our source code in a few days

- **What's New**: MobA라는 혁신적인 모바일 어시스턴트를 제안합니다. 이를 통해 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLM)을 활용하여 사용자의 명령 이해와 계획 능력을 향상시킵니다.

- **Technical Details**: MobA는 두 가지 수준의 에이전트 아키텍처로 구성되어 있습니다. 상위 에이전트(Global Agent, GA)는 사용자 명령을 이해하고, 히스토리 메모리를 추적하며, 작업을 계획하는 역할을 합니다. 하위 에이전트(Local Agent, LA)는 GA의 메모리와 서브 태스크에 따라 상세한 작업을 함수 호출의 형태로 예측합니다. 또한, Reflect Module을 통합하여 이전에 보지 못한 복잡한 작업을 처리할 수 있는 능력을 제공합니다.

- **Performance Highlights**: MobA는 실제 평가에서 작업 수행 효율성(Task Execution Efficiency)과 완료율(Completion Rate)에서 상당한 개선을 보여주며, MLLM을 활용한 모바일 어시스턴트의 가능성을 강조합니다.



### Exploring the Design Space of Visual Context Representation in Video MLLMs (https://arxiv.org/abs/2410.13694)
Comments:
          Long Video MLLM; work in progress

- **What's New**: 비디오 다중 모달 대형 언어 모델(Video Multimodal Large Language Models, MLLMs)의 시각적 컨텍스트 표현에 대한 체계적인 연구를 다룬 첫 번째 논문입니다. 연구진은 최적의 시각적 컨텍스트 표현 방식인 Opt-Visor 모델을 제안하며, 최대 162 프레임까지의 비디오를 처리할 수 있습니다.

- **Technical Details**: 비디오 MLLMs의 성능 향상을 위해 프레임 선택(frame selection)과 임베딩 선택(embedding selection)을 최적화하는 제약 최적화 문제로 작업을 정의했습니다. 각 프레임에서의 토큰 수와 프레임 수에 따른 언어 모델링 손실(training loss)을 함수로 모델링하여 시각적 컨텍스트의 경쟁 관계를 이해합니다. 이러한 분석을 바탕으로 성능 추세를 설명하는 함수 곡선을 맞추어 다양한 선택 전략의 효과를 평가했습니다.

- **Performance Highlights**: 실험 결과, 시각적 임베딩 수(토큰 또는 프레임)를 증가시키는 것이 전반적으로 성능 향상에 기여한다는 것을 확인했습니다. 특히, 압축 기반 방법이 더 적은 시각적 임베딩으로도 더 많은 의미 정보를 보존할 수 있다는 점이 강조되었습니다. 연구진은 이러한 성과를 통해 프레임 선택과 임베딩 선택 간의 이상적인 비율을 찾는 방법을 제안하고, 경험적 실험과 일치하는 제안된 최적 설정을 검증하였습니다.



### Pose-Based Sign Language Appearance Transfer (https://arxiv.org/abs/2410.13675)
- **What's New**: 이 연구에서는 수화에서 서명자의 외모를 제어하는 방법을 소개하며, 서면 내용은 보존하는 방법을 제시합니다. 이 방법은 서명자의 외모를 다른 사람으로 전이하여 자연스러운 움직임과 전환을 유지합니다.

- **Technical Details**: 서명자의 외모를 변경하고 서명 내용을 유지하기 위해 포즈 시퀀스를 조작하는 방법을 사용합니다. 신호 긴밀성과 자연스러운 운동을 위해 몸체와 얼굴의 특성은 수정하지만 손의 형상은 유지합니다. 이는 평균화된 포즈를 통해 수행됩니다.

- **Performance Highlights**: 이 방법은 서명자의 신원을 식별하는 정확성을 줄이면서도 수화 인식 성능을 약간 저하시킵니다. 분석 결과, 원래 포즈를 이용한 모델이 가장 뛰어난 성능을 보였으며, 전이된 포즈를 사용했을 때 신원 식별 정확도가 52.20%로 감소했습니다. 이는 프라이버시와 유용성 간의 균형을 잘 보여줍니다.



### VL-GLUE: A Suite of Fundamental yet Challenging Visuo-Linguistic Reasoning Tasks (https://arxiv.org/abs/2410.13666)
Comments:
          18 pages, 7 figures

- **What's New**: 본 연구에서는 비주얼-언어적 (Visuo-Linguistic) 이해를 위한 새로운 멀티태스크 벤치마크인 VL-GLUE를 제안합니다. VL-GLUE는 7개의 다양한 태스크로 구성되어 있으며, 10만 개 이상의 샘플을 포함해 비주얼과 텍스트 간의 결합 추론 능력을 평가합니다.

- **Technical Details**: VL-GLUE는 이미지와 텍스트 정보를 결합하여 추론을 필요로 하는 7개의 태스크로 구성되어 있습니다. 이 benchmark는 다양한 이미지 유형(일상 장면, 도표 등)과 특정 도메인 텍스트(요리, 정치 등)를 포함해, 실제 세계에서의 멀티모달 이해의 필요성을 보여줍니다.

- **Performance Highlights**: 기존의 대규모 비전-언어 모델들이 VL-GLUE 벤치마크에서 낮은 점수를 얻어, 이 분야의 모델들이 시기적절한 비주얼-언어적 추론 능력을 갖춤이 긴급하게 요구되고 있다는 점이 부각되었습니다.



### H2OVL-Mississippi Vision Language Models Technical Repor (https://arxiv.org/abs/2410.13611)
- **What's New**: H2OVL-Mississippi 모델은 3700만 개의 이미지-텍스트 쌍을 기반으로, 8개의 H100 GPU를 사용하여 240시간 동안 훈련된 작은 비전-언어 모델(VLM) 쌍을 소개합니다. 특히, H2OVL-Mississippi-0.8B는 8억 개의 매개변수로 구성되어 텍스트 인식에 특화되어 있으며, OCRBench의 텍스트 인식 부문에서 최첨단 성능을 발휘하고 있습니다.

- **Technical Details**: H2OVL-Mississippi 모델은 Vision Transformer(비전 트랜스포머) 구성 요소와 대형 언어 모델(LLM)로 이루어집니다. H2OVL-Mississippi-0.8B는 OCR 및 문서 중심 작업에 최적화되어 있고, H2OVL-Mississippi-2B는 다양한 멀티모달 작업을 수행할 수 있는 일반 목적 모델입니다. 이들은 각각 256에서 1590개의 시각적 토큰을 생성하며, 동적 해상도 전략(dynamic resolution)과 다중 스케일 적응 크롭(multi-scale adaptive cropping) 전략을 활용하여 다양한 이미지 크기와 종횡비에 적응합니다.

- **Performance Highlights**: H2OVL-Mississippi-0.8B는 OCRBench에서 텍스트 인식 부문에서 최첨단 성능을 보여주며, H2OVL-Mississippi-2B는 다양한 학술 벤치마크에서 경쟁력 있는 메트릭스를 제공합니다. 두 모델 모두 H2O-Danube 언어 모델의 기능을 확장하여 비주얼 도메인으로의 적용 가능성을 높이고, Apache 2.0 라이선스 하에 공개되어 문서 AI와 비주얼 LLM의 접근성을 높였습니다.



### MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling (https://arxiv.org/abs/2410.13610)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)와 의료 분야의 복잡한 문제를 해결하기 위한 새로운 프레임워크인 MeNTi를 소개합니다.

- **Technical Details**: MeNTi는 LLMs를 위한 보편적인 에이전트 아키텍처로, 전문화된 의료 도구 세트를 통합하고 메타 도구(meta-tool) 및 중첩 호출(nested calling) 메커니즘을 사용하여 LLM 도구 활용을 강화합니다. 이를 통해 유연한 도구 선택 및 중첩 도구 호출이 가능해지며, 계산기 선택, 슬롯 채우기(slot filling), 단위 변환을 포함한 복잡한 의료 시나리오 문제를 해결합니다.

- **Performance Highlights**: CalcQA라는 벤치마크를 통해 LLM의 정량적 평가 능력을 검증하며, 100개의 사례-계산기 쌍과 281개의 의료 도구가 포함된 도구 키트를 통해 실험 결과에서 상당한 성능 개선을 보여주었습니다.



### Large Language Models as Narrative-Driven Recommenders (https://arxiv.org/abs/2410.13604)
Comments:
          Under review; 19 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 사용하여 자유형식의 텍스트로 표현된 영화 추천 요청에 대한 개인화된 추천을 제공하기 위한 새로운 접근 방식을 탐구하였습니다. 특히, reddit의 영화 추천 커뮤니티에서 수집된 데이터셋을 활용하여 38개의 오픈소스 및 클로즈드 소스 LLM의 성능을 비교하였습니다.

- **Technical Details**: 이 연구는 zero-shot, identity, few-shot 프롬프트 기법을 사용하여 LLM이 사용자 요청을 자연어로 처리하고 관련 영화를 추천할 수 있는지 평가합니다. 평가된 LLM은 크기에 따라 분류되며, 각 모델은 기본적인 zero-shot 프롬프트를 통해 추천 정확도를 높일 수 있음을 보여줍니다.

- **Performance Highlights**: LLMs는 기존의 추천 알고리즘보다 더 높은 성능을 보이며, 특히 GPT-4o는 기본 성능보다 70% 더 높은 추천 성능을 보였습니다. 중간 크기의 오픈소스 모델도 상대적으로 높은 성능을 유지하며 클로즈드 소스 모델과 비교하여 경쟁력을 보여주었습니다.



### MathGAP: Out-of-Distribution Evaluation on Problems with Arbitrarily Complex Proofs (https://arxiv.org/abs/2410.13502)
Comments:
          Preprint

- **What's New**: MathGAP이라는 새로운 평가 프레임워크를 제시하여, 보다 복잡한 산술 증명이 포함된 문제에서 대형 언어 모델(LLMs)의 일반화 능력을 분석합니다. 이를 통해 기존의 평가 방법의 한계를 극복하고 보다 체계적인 연구를 가능하게 합니다.

- **Technical Details**: MathGAP는 고정된 증명 기준을 따르는 문제를 생성하고 체계적인 체인-오브-생각(chain-of-thought) 주석을 제공합니다. 이 프레임워크는 증명 나무(proof trees)를 기반으로 각 문제의 복잡성을 특성화하고, 간단한 예제를 사용하여 더 복잡한 문제를 해결할 수 있는 LLM의 능력을 평가합니다.

- **Performance Highlights**: 모델 성능은 증명 깊이와 너비가 증가함에 따라 일관되게 감소하며, 특히 비선형(nonlinear) 문제에서 눈에 띄는 감소가 관찰됩니다. 흥미롭게도, 테스트 세트와 동일한 분포의 예제를 제공하는 것이 항상 성능에 이롭지 않으며, 다양한 복잡성을 가진 예제를 제시하는 것이 더 유효한 경우가 많습니다.



### Progressive Mixed-Precision Decoding for Efficient LLM Inferenc (https://arxiv.org/abs/2410.13461)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 추론 단계에서 질량량(data precision) 할당을 개선하기 위해 단계 인식(phase-aware) 방법을 제안합니다. 이 방법은 prefill 단계에서 높은 정밀도를 유지하고, decoding 단계에서는 낮은 정밀도로 갈수록 하여 메모리 성능과 품질을 모두 향상시킵니다.

- **Technical Details**: 제안된 방법은 Progressive Mixed-Precision Decoding(PMPD)이라는 전략을 통해, 자동 회귀 생성 과정에서의 후반부 토큰들은 정밀도 감소에 더 탄력적이므로 초기 정밀도를 높게 유지하고, 후반부에서 점진적으로 정밀도를 감소시키는 방식으로 동작합니다. 또한, 두 가지 정밀도 전환 스케줄러를 통해 정밀도 감소 시점을 전략적으로 결정합니다.

- **Performance Highlights**: 다양한 언어 작업에서 우리의 방법을 적용했을 때, NPU 플랫폼에서는 3.8-8.0배의 디코딩 처리량 향상을 달성하고, GPU에서는 fp16 모델에 비해 1.4-12.2배의 속도 증가를 기록했습니다. 이처럼, uniform quantization 접근법보다 1.54배 더 높은 성능을 보여 주며 출력 품질을 유지합니다.



### Similarity-Dissimilarity Loss with Supervised Contrastive Learning for Multi-label Classification (https://arxiv.org/abs/2410.13439)
- **What's New**: 본 연구는 멀티 라벨 분류에서의 슈퍼바이저드 대조 학습(Supervised Contrastive Learning)에서 긍정 샘플을 결정하는 데 있어 새로운 접근 방식을 제안합니다. 특히, 다섯 가지 고유한 관계를 도입하고, 유사성 및 비유사성 손실(Similarity-Dissimilarity Loss)을 통해 대조 손실 함수(weights)를 동적으로 조정합니다.

- **Technical Details**: 다섯 가지 관계(R2, R3, R4, R5)를 정의하여 멀티 라벨 샘플과 앵커(anchor) 사이의 유사성과 비유사성을 계산하여 손실을 재가중화하는 새로운 Similarity-Dissimilarity Loss를 제안합니다. 이를 통해 ALL, ANY 및 MulSupCon 등의 기존 방법의 한계를 극복합니다.

- **Performance Highlights**: MIMIC 데이터셋에서 멀티 라벨 텍스트 분류 실험을 수행한 결과, 제안된 손실 함수가 슈퍼바이저드 대조 학습 패러다임 하에서 모든 인코더(encoders)에 대해 성능을 효과적으로 향상시키는 것으로 나타났습니다. 실험 결과는 제안된 방법의 효과성과 견고성을 뒷받침합니다.



### MoR: Mixture of Ranks for Low-Rank Adaptation Tuning (https://arxiv.org/abs/2410.13408)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 Mixture of Ranks (MoR)라는 새로운 접근 방식을 도입하여 LoRA의 성능을 개선하고, 다중 작업에서 효율적으로 학습할 수 있는 방법을 제안합니다. 기존의 LoRA와 MoE 방식의 한계를 극복하고, 다양한 작업에 대한 rank-specific 정보를 효과적으로 학습합니다.

- **Technical Details**: MoR은 세 가지 주요 구성 요소인 공유 전문가(shared experts), 다중 rank 적응(multi-rank adaptation), 혼합 학습(mixture learning)으로 구성됩니다. 이 방법은 여러 LoRA를 통합하여 학습할 수 있는 새로운 프레임워크를 제공하며, 매핑(mapping) 및 스케일링(scaling)을 통해 다중 작업을 수행합니다.

- **Performance Highlights**: MoR는 기존 LoRA 방법 대비 1.31%의 성능 향상을 달성하면서도 파라미터는 93.93%만 사용합니다. 또한, 다양한 실험에서 MoR은 효율성, 일반화 가능성, 확장성을 입증하며, 다중 LoRA 구조의 학습 비용을 크게 줄이고 더 간결한 정보를 동적으로 학습할 수 있음을 보여줍니다.



### Towards Hybrid Intelligence in Journalism: Findings and Lessons Learnt from a Collaborative Analysis of Greek Political Rhetoric by ChatGPT and Humans (https://arxiv.org/abs/2410.13400)
- **What's New**: 이번 연구 프로젝트는 그리스 2023년 총선을 준비하기 위해 시작된 "정치 담론 분석: 인간과 인공지능 간의 협력"으로, 정치 지도자들의 캠페인 연설을 분석하는 데 중점을 두었습니다.

- **Technical Details**: 본 장에서는 감정 분석(sentiment analysis), 양극화(polarization), 포퓰리즘(populism), 주제 탐지(topic detection), 고유명사 인식(Named Entities Recognition, NER) 등 다양한 정치 담론 분석의 측면을 다룹니다. 대형 언어 모델(large language model, LLM) 특히 OpenAI의 ChatGPT의 정치 연설 분석 능력을 조사하였고, AI가 저널리즘 프로젝트 및 기타 사회적 분야에서 어떻게 활용될 수 있는지에 대해 인간의 감독(human oversight)의 중요성을 강조하였습니다.

- **Performance Highlights**: 이 프로젝트는 디지털 인문학(digital humanities) 분야에서 인간-AI 협력(hybrid intelligence)의 혁신적인 예로서, 향후 연구니즈에 대한 귀중한 통찰력을 제공합니다.



### On the Use of Audio to Improve Dialogue Policies (https://arxiv.org/abs/2410.13385)
Comments:
          IberSpeech 2024

- **What's New**: 본 논문은 오디오 정보와 텍스트 임베딩을 조합하여 대화 정책을 개선하는 새로운 아키텍처를 제안하고 있습니다. 특히, 소음이 있는 전사 환경에서 성능 향상을 이루었습니다.

- **Technical Details**: 제안된 시스템은 Double Multi-Head Attention (MHA) 기법을 활용하여 오디오 정보와 텍스트 정보를 결합합니다. GPT-2를 사용한 텍스트 표현과 Wav2Vec2.0과 같은 미리 훈련된 모델을 통해 오디오 표현을 생성합니다.

- **Performance Highlights**: DSTC2 데이터셋을 사용한 실험에서, 오디오 정보를 포함한 대화 정책은 텍스트 기반 시스템에 비해 9.8% 상대적 향상을 보였습니다. 다중 모달 특성의 결합 방식이 성능 향상에 중요한 역할을 합니다.



### Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistan (https://arxiv.org/abs/2410.13360)
- **What's New**: 본 논문에서는 Retrieval Augmented Personalization (RAP) 프레임워크를 소개하여 다중 모드 대형 언어 모델(MLLMs)의 개인화를 가능하게 합니다. RAP는 일반 MLLM을 개인화된 어시스턴트로 전환하는 세 가지 주요 단계로 구성됩니다: 기억(Recall), 검색(Retrieve), 생성(Generate).

- **Technical Details**: RAP는 사용자 관련 정보(예: 이름, 아바타 등)를 저장하는 키-값 데이터베이스를 설계합니다. 사용자가 대화를 시작할 때, RAP는 다중 모드 검색기를 통해 데이터베이스에서 관련 정보를 검색하고, 이를 MLLM에 입력하여 개인화된 지식 강화 응답을 생성합니다. 추가로 생성 품질 향상을 위해 데이터 수집 파이프라인을 개발하고 개인화된 훈련을 위한 전문적인 데이터셋을 생성합니다.

- **Performance Highlights**: RAP-MLLMs는 개인화된 이미지 캡션 작성, 질문 응답 및 시각적 인식과 같은 다양한 작업에서 뛰어난 유연성과 생성 품질을 보여줍니다. 모델들은 무한한 시각적 개념에 대해 일반화 능력을 발휘하며, 사용자 관련 정보를 효과적으로 처리하여 개인화된 출력을 제공합니다.



### Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding (https://arxiv.org/abs/2410.13321)
- **What's New**: 이번 연구에서는 LVLMs에서 나타나는 언어 priors의 문제를 해결하기 위해 새로운 방법인 Summary-Guided Decoding (SGD)를 제안합니다. SGD는 이미지 정보에 더 집중하도록 모델을 유도하며, 텍스트 품질을 유지합니다.

- **Technical Details**: 연구는 LVLMs에서의 언어 priors를 분석하고, 이미지 관련 부분의 품사(POS)와 연관된 토큰을 생성할 때 언어 priors에 대한 모델의 의존도가 증가함을 발견했습니다. SGD는 요약(context) 기법을 활용하여 이미지 관련 POS 토큰의 다음-토큰 확률을 수정하여 텍스트 품질을 최대한 보존하면서 이미지 정보를 반영합니다.

- **Performance Highlights**: SGD는 객체 환각(object hallucination) 벤치마크에서 모든 다른 해석 방법을 초월했으며(CHAIRS에서 +16.5%, CHAIRI에서 +19% 향상), 정밀도와 재현율의 균형을 잘 유지하며 Pareto optimal성을 달성했습니다. 또한 텍스트 품질을 거의 완벽하게 유지하면서 객체 환각을 줄이는 데 강력한 성과를 보였습니다.



### CLaMP 2: Multimodal Music Information Retrieval Across 101 Languages Using Large Language Models (https://arxiv.org/abs/2410.13267)
Comments:
          17 pages, 10 figures, 4 tables

- **What's New**: CLaMP 2는 101개 언어를 지원하는 새로운 음악 정보 검색 시스템으로, ABC 표기법과 MIDI를 동시에 활용하는 다중 모드(multi-modal) 모델입니다. 이 시스템은 150만 개의 ABC-MIDI-텍스트 트리플로 사전 훈련되어, 언어 모델을 통해 텍스트의 노이즈를 줄이고 다국어 설명을 정제합니다.

- **Technical Details**: CLaMP 2는 대조 학습(contrastive learning)을 통해 텍스트 인코더와 음악 인코더를 정렬합니다. 이 시스템은 LLM(대형 언어 모델)을 이용하여 101개 언어의 음악 정보를 효과적으로 처리합니다. 특히, 기존의 음악 메타데이터의 일관성을 개선하고 이러한 데이터셋을 통해 훈련을 받음으로써 다국어 검색 성능을 높입니다.

- **Performance Highlights**: CLaMP 2는 다국어 의미 검색과 음악 분류에서 최첨단(results state-of-the-art) 성능을 보여주며, 기존 음악 정보 검색 시스템의 한계를 넘어 글로벌 음악 접근성을 향상시킵니다.



### Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation (https://arxiv.org/abs/2410.13248)
- **What's New**: 최근 설명 가능한 추천 시스템에 대한 연구는 표준 텍스트 생성 문제로 접근하며, 모델을 예측된 텍스트와 실제 텍스트 간의 유사성을 기반으로 평가합니다. 그러나 이 접근법은 사용자(구매 후) 감정을 정확히 반영하는지 여부를 간과합니다. 이 연구에서는 사용자의 감정을 중점적으로 고려하는 새로운 데이터셋과 평가 방법을 소개합니다.

- **Technical Details**: 우리는 LLM(Long Language Model)을 사용하여 사용자 구매 후 리뷰에서 긍정적 및 부정적 의견을 명시적으로 추출하여 데이터셋을 구성합니다. 시스템을 평가할 때 생성된 설명이 1) 사용자 감정과 잘 일치하는지, 2) 목표 아이템에 대한 사용자 의견의 긍정적 및 부정적 식별을 정확히 수행하는지에 대한 두 가지 기준을 제안합니다.

- **Performance Highlights**: 여러 최신 모델을 우리의 데이터셋에서 벤치마킹하였으며, 기존 지표에서 높은 성과를 달성하더라도 생성된 설명이 사용자 감정과 잘 일치하지 않을 수 있음을 보여줍니다. 또한, 목표 아이템에 대한 사용자(예측된) 평가가 모델에 직접 입력될 경우, 기존 모델들이 보다 감정 인식적인 설명을 제공할 수 있음을 발견하였습니다.



### Anchored Alignment for Self-Explanations Enhancemen (https://arxiv.org/abs/2410.13216)
- **What's New**: 본 연구에서는 언어 모델의 자기 설명(self-explanation) 능력을 향상시키기 위해 주석이 달린 이유 설명이 없는 경우에도 이들의 사고 내용을 명확히 서술하는 방식으로 모델 정렬(alignment) 방법론을 제안합니다.

- **Technical Details**: 본 방법론은 설명 품질 평가(explanation quality assessment), 자기 지시 데이터셋 생성(self-instruction dataset generation), 모델 정렬(model alignment)이라는 세 가지 핵심 요소로 구성됩니다. 특히, 'Anchor Preference Pairs'라는 새로운 기술을 도입하여 모델 출력을 일관되게 정확한 것, 일관되게 부정확한 것, 가변적인 것으로 세 가지 범주로 분류하여 선호 쌍(preference pairs) 선택을 개선합니다. 이를 통해 Direct Preference Optimization (DPO) 전략의 효과성을 증대시킵니다.

- **Performance Highlights**: 실험 결과, 본 접근법은 다른 fine-tuning 전략과 비교할 때 설명 품질을 유의미하게 개선하면서도 정확성을 유지하는 것으로 나타났습니다. 특히, Anchor Preference Pairs를 활용한 방법론이 Judge 기반 평가에만 의존한 자기 정렬 전략보다 더욱 우수한 성능을 보이는 것을 입증했습니다.



### Failing Forward: Improving Generative Error Correction for ASR with Synthetic Data and Retrieval Augmentation (https://arxiv.org/abs/2410.13198)
Comments:
          Preprint. Under Review

- **What's New**: 본 논문에서는 Generative Error Correction (GEC) 모델의 일반화 한계를 극복하기 위해 DARAG (Data- and Retrieval-Augmented Generative Error Correction) 접근법을 제안합니다. 이 방법은 ASR (Automatic Speech Recognition) 시스템의 오류 교정 성능을 향상시키기 위해 synthetic data를 생성하고, named entities (NEs)를 효과적으로 처리하기 위한 retrieval-augmented correction 기술을 도입합니다.

- **Technical Details**: DARAG 접근법은 LLMs (Large Language Models)에 의해 생성된 synthetic transcripts와 text-to-speech 모델을 사용하여 GEC 훈련 데이터셋을 증강합니다. 또한, OOD (Out-Of-Domain) 시나리오에서의 오류를 비지도 학습 방식으로 시뮬레이션하며, 데이터베이스에서 추출한 named entities를 사용하여 입력 데이터를 보강합니다. 이는 GEC가 테스트 시 경험하는 새로운 오류 유형을 처리할 수 있도록 돕습니다.

- **Performance Highlights**: DARAG는 기존의 GEC 방법들과 비교하여 8%에서 30%까지의 상대적인 Word Error Rate (WER) 개선을 달성하였으며, OOD 환경에서는 10%에서 33% 향상이 있었습니다. 여러 데이터셋과 설정에서의 실험 결과 이 접근법이 ASR 성능을 극대화하는 데 효과적임을 입증했습니다.



### Chain of Ideas: Revolutionizing Research in Novel Idea Development with LLM Agents (https://arxiv.org/abs/2410.13185)
Comments:
          10 pages,5 figures, conference

- **What's New**: 이 논문은 Chain-of-Ideas (CoI) 에이전트를 통해 대형 언어 모델(LLMs)이 연구 아이디어 생성의 효율성을 개선할 수 있는 새로운 방안을 제안합니다. CoI 에이전트는 관련 문헌을 체계적으로 정리하여 연구 분야의 발전을 잘 반영하도록 돕습니다.

- **Technical Details**: CoI 에이전트는 (1) CoI 구성, (2) 아이디어 생성, (3) 실험 설계의 세 가지 단계로 구성됩니다. 각 단계에서 LLM은 연구 분야의 다양한 트렌드를 반영하여 복수의 CoIs를 구축하고, 각 CoI에 대해 예측 및 아이디어를 체계적으로 발전시키는 과정을 거칩니다.

- **Performance Highlights**: 실험 결과에 따르면 CoI 에이전트는 여러 자동화된 방법보다 항상 높은 성능을 보였으며, 사람의 연구 아이디어 생성 품질과도 비교 가능한 결과를 나타냈습니다. CoI 에이전트는 아이디어 생성에서 56 ELO 점수 차이로 두 번째 방법을 초월했습니다.



### EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning (https://arxiv.org/abs/2410.13179)
- **What's New**: 이번 논문에서는 Speech Representation Learning을 위한 새로운 Self-Supervised Learning 접근 법인 EH-MAM (Easy-to-Hard adaptive Masked Acoustic Modeling)을 제안합니다. 기존의 랜덤 마스킹 방식을 사용하는 Masked Acoustic Modeling (MAM)과는 달리, 우리는 선택적이고 적응적인 마스킹 전략을 도입하였습니다.

- **Technical Details**: EH-MAM은 SSL 훈련 중 모델에 점진적으로 더 어려운 영역을 도입하여 재구성을 수행합니다. 개별 프레임의 재구성 손실( reconstruction loss)을 활용하여 MAM 전제 과제를 해결하는 난이도를 판단하며, 이를 위해 교사 모델(teacher model)을 사용하여 프레임 단위 손실을 예측하고 어떤 프레임을 마스킹할 지 결정합니다.

- **Performance Highlights**: EH-MAM은 여러 최신 기준선(baselines) 대비 5%-10% 향상된 성능을 보이며, 저자원(low-resource) 음성 인식 및 SUPERB 벤치마크에서 효과적으로 유용한 컨텍스트를 포착하는 마스킹 영역을 분석합니다.



### An Evolved Universal Transformer Memory (https://arxiv.org/abs/2410.13166)
Comments:
          29 pages, 14 figures. Preprint, under submission. Source code is available at this https URL

- **What's New**: 본 논문은 Neural Attention Memory Models (NAMMs)을 제안하며, 메모리 관리를 위한 학습된 네트워크를 도입하여 Transformers의 성능과 효율성을 동시에 향상시킵니다. 이는 기계가 가진 메모리 관리의 질의를 진화 기반 접근법으로 해결하여, 기능적으로 매우 다양한 아키텍처에서 자율적으로 적용될 수 있도록 설계되었습니다.

- **Technical Details**: NAMMs는 Transformers의 Key-Value (KV) 캐시의 잠재적 메모리를 형성하는 새로운 방법을 제안하여, 각 레이어와 attention head가 그들의 특정 요구에 가장 관련 있는 정보에 집중하도록 지원합니다. 이 방식은 학습된 attention 매트릭스를 기반으로 모든 transformer 기반 아키텍처에 일반적으로 적용 가능하며, Llama 3 8B 모델 위에서 학습하여 성능과 효율성을 모두 극대화합니다.

- **Performance Highlights**: NAMMs를 통한 학습 결과로 36개의 LongBench, InfiniteBench 및 새로운 일본어 벤치마크에서 뛰어난 성능 개선을 기록했습니다. 기존의 수작업 전략과 비교할 때, NAMMs는 성능 저하 없이 메모리 용량을 유의미하게 감소시켰습니다. 또한, NAMMs는 언어 과제로만 학습되었음에도 불구하고 다양한 입력 모달리티를 통해 다른 transformer 모델에 제로샷 전이(transfer) 되는 성과를 보였습니다.



### Controllable Generation via Locally Constrained Resampling (https://arxiv.org/abs/2410.13111)
Comments:
          arXiv admin note: text overlap with arXiv:2312.03905

- **What's New**: 이번 논문에서 저자들은 LLMs (Large Language Models)의 제한을 받는 샘플링 문제를 해결하기 위한 새로운 확률론적 접근 방식을 제안합니다. 기존의 greedy 방법론 대신 Bayesian conditioning을 통해 더 글로벌한 제약 생성이 가능하도록 개선하였습니다.

- **Technical Details**: 제안된 방법은 LLM 샘플에서 유도된 지역적인 분포를 기반으로 하며, 이를 사용하여 제약을 조건화하고 샘플링합니다. 이 접근법은 싱글 토큰별로 제약을 강제로 이행하는 것이 아니라 전체 시퀀스를 고려하여 제약을 적용합니다. 제약 회로(Constraint Circuits)를 통해 Boolean python 함수를 사용하여 제약을 효율적으로 표현할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 LLM의 독소 생성 방지 및 Sudoku 퍼즐 해결과 같은 여러 작업에서 평가되었습니다. 특히, 독소 표현의 리스트를 제외함으로써 모델의 출력을 독소 생성에서 멀어지게 하여 이전의 방법론보다 우수한 성능을 보였으며, Sudoku 퍼즐에서는 100%의 정확도를 달성했습니다. GPT4-o와 Gemini 1.5와 비교할 때 이들의 정확도는 각각 26% 및 45%에 불과했습니다.



### Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models (https://arxiv.org/abs/2410.13097)
- **What's New**: 본 논문에서는 파라미터 효율적인 미세 조정(PEFT) 방법을 여러 장치에 분산된 개인 데이터로 미세 조정하기 위한 새로운 방법인 FedTT 및 FedTT+를 제안합니다. 이 방법들은 Federated Learning(FL)과 통합되어 사용자 프라이버시를 보호하면서도 데이터 이질성 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: FedTT는 클라이언트 측 모델의 인코더/디코더 블록에 텐서화된 어댑터를 통합하여 LLM을 적응시키는 방법입니다. FedTT는 크로스-사일로 FL 및 대규모 크로스-디바이스 FL 모두에 적용될 수 있습니다. FedTT+는 데이터 이질성에 대한 강인성을 추가적으로 향상시키기 위해 텐서 요소의 일부를 적응적으로 동결하여 학습 가능한 파라미터 수를 줄입니다.

- **Performance Highlights**: BERT 및 LLaMA 모델에 대한 실험 결과, 제안된 방법들이 기존의 연합 PEFT 접근 방식과 비교하여 데이터 이질성 문제를 성공적으로 해결했으며, 최대 10배의 통신 비용 절감 효과를 보였습니다. FedTT+는 상태-of-the-art 크로스-사일로 FL 방법들을 능가하는 성능을 보여주었습니다.



### Self-Comparison for Dataset-Level Membership Inference in Large (Vision-)Language Models (https://arxiv.org/abs/2410.13088)
- **What's New**: 본 논문에서는 Self-Comparison Membership Inference (SMI)이라는 새로운 데이터셋 수준의 멤버십 추론 방법을 제안합니다. 기존의 Membership Inference Attack (MIA) 방법론의 한계를 극복하기 위해 설계된 이 방식은 정체된 데이터에 대한 비밀스러운 사용을 감지할 수 있습니다.

- **Technical Details**: SMI 방법은 멤버 데이터(membership data)의 접두사와 비멤버 데이터(non-membership data)의 접미사를 비교하여, 훈련 데이터에 대한 모델의 암기 현상을 유도합니다. 구체적으로는, 멤버 데이터가 주어졌을 때, 패러프레이징(paraphrasing)을 통해 두 세트의 분포가 어떻게 변화하는지를 비교합니다. 기존 MIA 방식은 반드시 그라운드 트루스 멤버 데이터를 요구하는 반면, SMI는 유사한 분포를 가지지 않아도 되는 보조 비멤버 세트를 요구합니다.

- **Performance Highlights**: SMI 방법은 다양한 LLMs 및 VLMs 모델에서 기존의 MIA 및 데이터셋 추론 기술보다 뛰어난 성능을 보였습니다. 이는 특히 그라운드 트루스 멤버 데이터에 대한 사전 지식이 없을 때에도 유효합니다. 실험 결과, 우리의 방법이 공개 모델, 파인 튜닝된 모델 및 API 기반 상업 모델에 이르기까지 여러 데이터셋에서 우수한 성능을 발휘함을 확인하였습니다.



### MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models (https://arxiv.org/abs/2410.13085)
- **What's New**: 본 논문에서는 Med-LVLMs의 사실성을 향상시키기 위해 MMed-RAG라는 다중 모달 RAG 시스템을 제안합니다. 이 시스템은 도메인 인식을 위한 검색 메커니즘, 적응형 검색된 컨텍스트 선택 방법, 그리고 검증 가능한 RAG 기반의 선호 미세 조정 전략을 포함하여, 의료 데이터의 다양한 분야에 대해 일반적이고 신뢰할 수 있는 접근 방식을 제공합니다.

- **Technical Details**: MMed-RAG는 세 가지 주요 요소로 구성됩니다: 1) 도메인 인식 검색 메커니즘 - 입력 의료 이미지에 적합한 검색 모델을 선택하기 위해 도메인 식별 모듈을 설계하였습니다. 2) 적응형 검색된 컨텍스트 선택 - 검색된 컨텍스트의 개수를 선택하는 방법입니다. 3) RAG 기반 선호 미세 조정 - 교차 모달 정렬을 개선하고 모델과 실제 간의 전체 정렬을 높이는 방법입니다.

- **Performance Highlights**: MMed-RAG는 5개의 의료 데이터세트에서 실험을 실시하여, Medical VQA와 보고서 생성 작업에서 각각 18.5% 및 69.1%의 사실 정확도를 향상시켰습니다. 전반적으로 MMed-RAG는 Med-LVLMs의 정확성을 평균 43.8% 개선하였습니다.



### Language Models as Semiotic Machines: Reconceptualizing AI Language Systems through Structuralist and Post-Structuralist Theories of Languag (https://arxiv.org/abs/2410.13065)
Comments:
          18 pages, 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 인간의 인지 과정을 모방하는 것으로 보지 않고, 기호학적 기계(semiotic machines)로 재구성하여 이해하는 새로운 프레임워크를 제안합니다. 저자는 페르디낭 드 소쉬르(Ferdinand de Saussure)와 자크 데리다(Jacques Derrida)의 언어 이론에 기초하여 LLM을 언어 자체의 모델로 설명하고 있습니다.

- **Technical Details**: 논문은 세 부분으로 나뉘어 있으며, 첫 번째 부분에서는 word2vec 임베딩 알고리즘의 작동 방식과 소쉬르의 기호 체계에 대한 설명을 제공합니다. 두 번째 부분에서는 데리다의 비판을 적용하여 LLM이 모델링하는 '쓰기'의 개념을 논의합니다. 마지막 세 번째 부분에서는 현대 LLM이 의미의 고정되지 않은 개념을 어떻게 반영하는지에 대해 설명하며, '다음 토큰 생성' 메커니즘이 의미의 역동성을 포착한다고 주장합니다.

- **Performance Highlights**: 대형 언어 모델은 언어 사용에서 거의 인간의 수준에 도달하며, word2vec 알고리즘을 기반으로 하여 컨텍스트 기반의 의미 표현을 채택하고 있습니다. 이러한 모델은 개별 단어뿐만 아니라 문장 및 다른 언어 구조를 포함한 복잡한 표현을 생성하려고 하며, 현재 사용되는 데이터셋은 방대한 양의 정보를 포함하고 있어, LLM이 언어 자체에 근접한 모델링을 가능하게 합니다.



### Supply Chain Network Extraction and Entity Classification Leveraging Large Language Models (https://arxiv.org/abs/2410.13051)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 자연어 처리(NLP) 및 대형 언어 모델(LLM)를 활용하여 비정형 텍스트 데이터를 기반으로 공급망 그래프를 구축하는 새로운 접근 방식을 제안합니다. 특히 토목 공학 산업을 사례 연구로 삼아 LLM이 기업, 프로젝트 등의 숨겨진 관계를 발견할 수 있는 방법을 보여줍니다.

- **Technical Details**: 본 연구는 데이터 수집, 프롬프트 엔지니어링, 그래프 구축, 엔티티 분류의 네 가지 주요 단계로 구성된 방법론을 적용합니다. 데이터 수집은 공개 소스의 뉴스 기사를 통해 이루어지며, 각 기업에 대해 2018년부터 2023년까지 연도별로 최소 10개의 뉴스 기사를 수집하여 총 50개의 원시 텍스트 데이터 포인트를 확보합니다. 이를 통해 각 기업의 활동에 대한 포괄적인 관점을 유지합니다.

- **Performance Highlights**: LLM으로 특정 산업에 맞춰 세부 조정(fine-tuning)을 수행함으로써 엔티티 분류의 정확도가 향상되었으며, 이는 산업별 공급망 분석의 잠재력을 강조합니다. 본 연구는 LLM을 통해 공급망 네트워크 모델링의 자동화를 가능하게 한 첫 번째 사례로, 공급망 동학에 대한 깊이 있는 통찰력을 제공합니다.



### LLM Confidence Evaluation Measures in Zero-Shot CSS Classification (https://arxiv.org/abs/2410.13047)
- **What's New**: 이 논문은 데이터 주석 작업에서의 대형 언어 모델(LLM)의 신뢰성을 평가하기 위해 세 가지 핵심 기여를 제안합니다. 첫째, 데이터 주석 작업을 위한 불확실성 정량화(UQ) 성능 측정 방법을 제안합니다. 둘째, 세 가지 서로 다른 LLM과 CSS 데이터 주석 작업에서 다섯 가지 UQ 전략을 처음으로 비교합니다. 셋째, LLM의 낮은 신뢰도 주석을 효과적으로 식별하고 잘못 레이블이 붙은 데이터를 발견하는 새로운 UQ 집계 전략을 소개합니다.

- **Technical Details**: 연구는 대형 언어 모델의 신뢰성을 평가하기 위해 여러 UQ 기법을 사용해 분석하였으며, 새로운 UQ 집계 전략을 제안하여 잘못 분류된 LLM 레이블 데이터를 식별하는 과정을 보다 간소화하였습니다. 이 논문은 다양한 UQ 방법을 비교하고, AUC(Area Under Curve) 분석을 통해 신뢰도 점수의 백분위수 기반 임계값을 적용하여 기술됩니다.

- **Performance Highlights**: 제안된 UQ 집계 전략은 기존 방법에 비해 개선된 성능을 보여주며, Human-in-the-loop 데이터 주석 프로세스를 획기적으로 개선할 수 있음을 입증했습니다. 이를 통해, LLM이 생성한 데이터 중 인간이 자원을 소모해야 할 데이터의 식별이 용이해졌습니다.



### Sensitivity of Generative VLMs to Semantically and Lexically Altered Prompts (https://arxiv.org/abs/2410.13030)
- **What's New**: 본 논문은 generative vision-language 모델(VLM)의 프롬프트에서의 어휘적 및 의미적 변화에 대한 민감성을 평가합니다. SugarCrepe++ 데이터셋을 사용하여 이러한 모델들이 프롬프트의 사소한 변화에 어떤 영향을 받는지를 분석합니다.

- **Technical Details**: 이 연구는 BLIP, BakLLaVA 및 GPT-4o와 같은 generative VLMs의 어휘 및 의미 변화 이해 능력을 평가합니다. SugarCrepe++ 데이터셋에서는 두 개의 긍정적인 캡션(P1, P2)과 하나의 부정적인 캡션(N)을 포함하여, 어휘적으로 다르지만 의미적으로 유사한 캡션을 제공합니다.

- **Performance Highlights**: 실험 결과, BakLLaVA와 GPT-4o 모두 입력 프롬프트의 약간의 변화에 대해 높은 민감성을 보였으며, 동일한 프롬프트에서 옵션의 순서를 변경하는 것만으로도 성능에 큰 차이를 보였습니다. 또한, 서로 다른 VLMs 간의 일관성이 부족하여 결과의 일관성을 높이기 위한 추가 연구가 필요함을 보여줍니다.



### Learning Representations for Reasoning: Generalizing Across Diverse Structures (https://arxiv.org/abs/2410.13018)
Comments:
          PhD thesis

- **What's New**: 이 논문은 인공지능 분야에서의 추론의 중요성과 관련하여, 기존의 지식 구조 및 쿼리 구조를 초월하는 일반화 알고리즘을 제안합니다. 또한, 구조적 데이터에서 기계 학습 개발을 가속화하기 위한 시스템을 구축했습니다.

- **Technical Details**: 제안된 모델 NBFNet은 전통적인 경로 기반(path-based) 방법과 동적 프로그래밍(dynamic programming)을 결합하여 새로운 엔티티(entity) 및 관계(relation) 어휘를 사용한 지식 그래프의 미지의 부분에 대한 유도 일반화를 실현합니다. A*Net은 NBFNet의 확장형으로, 수백만 개 규모의 지식 그래프에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: NBFNet은 기존의 최신 방법들에 비해 모든 설정에서 평균 18%의 성능 향상을 이루었으며, 특히 지식 그래프 완성(HITS@1) 및 유도 관계 예측(HITS@10)에서 각각 22%의 성능 개선을 보여줍니다.



### Large Language Models as a Tool for Mining Object Knowledg (https://arxiv.org/abs/2410.12959)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 일반 물체에 대한 명시적 지식을 공식화하는 능력을 조사하며, 물체의 구성 요소(부분) 및 재질에 대한 지식을 명확히 구분합니다. 이로 인해 LLMs의 잠재력을 이용하여 AI 시스템의 지식 기반을 보강하거나 대체하는 데 기여할 수 있습니다.

- **Technical Details**: 이 연구에서는 few-shot과 zero-shot multi-step 프롬프트 기법을 활용하여 약 2,300개의 물체 및 하위 유형에 대한 부품과 재질에 대한 데이터를 수집합니다. LLM의 언어 이해 능력을 통해 물체의 전체 구성과 부품의 재질에 대한 지식을 명확히 정리합니다.

- **Performance Highlights**: 평가 결과, 추출된 지식의 대부분이 인간의 이해와 일치하나, 프롬프트 기법에 따라 과도하게 단순화되거나 필요 이상의 세부 정보가 제공되는 경우도 있음을 보여줍니다. 이 연구는 물체 구조 및 구성에 대한 추론을 위한 유용한 자원으로서 기능할 것입니다.



### Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization (https://arxiv.org/abs/2410.12949)
Comments:
          20 pages, 19 figures, 7 tables

- **What's New**: 본 연구에서는 대형 언어 모델(LLM)에서 지식 편집 및 비학습(unlearning) 방법의 향상을 위한 기계적 해석 가능성(mechanistic interpretability)의 역할을 조사합니다. 특히, 출력 보존(output preserving) 기반의 구성 요소 로컬라이제이션 방식과 예측 가능한 중간 상태를 이용한 고수준 메커니즘 발견 방식 간의 차이를 강조합니다.

- **Technical Details**: 연구에서는 사실 회상(factual recall)을 위한 로컬라이제이션을 FLU(fact lookup) 메커니즘에 기반하여 진행하며, 이를 통해 이전의 방법들보다 더 견고한 편집 및 비학습을 구현합니다. 다양한 입력/출력 형식에서의 견고함이 향상되었으며, 원하지 않는 정보를 다시 학습하는 것을 방지하면서 부작용(side effects)도 감소합니다.

- **Performance Highlights**: Gemma-7B 모델을 사용하여 다양한 데이터셋에서 FLU 메커니즘 기반의 편집 및 비학습이 기존 방법들보다 더 높은 견고성과 일반화 능력을 나타낸다는 것을 확인하였습니다. 특히, 스포츠 사실 데이터셋과 CounterFact 데이터셋에서 실험을 수행하여 이런 결과를 입증하였습니다.



### Exploiting Longitudinal Speech Sessions via Voice Assistant Systems for Early Detection of Cognitive Declin (https://arxiv.org/abs/2410.12885)
Comments:
          IEEE International Conference on E-health Networking, Application & Services

- **What's New**: 본 연구는 음성 비서 시스템(VAS)을 활용하여 18개월 동안 3개월 간격으로 7회의 음성 데이터를 원격으로 수집하는 종단적 연구를 진행하였다. 이를 통해 경도 인지 장애(MCI) 탐지와 인지 변화 예측 방법론을 제안하였다.

- **Technical Details**: 연구에서는 음성과 관련된 데이터에서 역사적 데이터를 포함한 두 가지 방법을 사용하여 MCI 탐지와 인지 변화를 예측하였다. 저장된 음성 데이터는 35명의 참가자들로부터 수집되었으며, 이들은 20명의 MCI 환자와 15명의 건강 대조군(HC)으로 구성되었다. 각 참가자는 18개의 인지 과제 질문에 답함으로써 인지 능력이 평가되었다. 음성 데이터는 자가 감독 학습 및 대형 언어 모델을 통해 추출된 음향(acoustic) 및 언어적(linguistic) 특징을 사용하여 분석되었다.

- **Performance Highlights**: 연구 결과, 역사적 데이터를 포함하였을 때 MCI 탐지의 F1-score는 음향 특징의 경우 58.6%에서 71.2%로(12.6% 개선), 언어적 특징의 경우 62.1%에서 75.1%로(13.0% 개선) 상승하였다. 인지 변화 예측에 있어서도 음향 특징의 경우 73.7%의 F1-score를 달성하였다. 이러한 결과들은 음성 비서 시스템을 기반으로 한 음성 세션이 조기 인지 저하 탐지에 잠재력을 지니고 있음을 확인시켜 준다.



### MIND: Math Informed syNthetic Dialogues for Pretraining LLMs (https://arxiv.org/abs/2410.12881)
Comments:
          31 pages, 5 figures, 14 tables

- **What's New**: 이번 연구에서는 대규모 다채로운 Math Informed syNthetic Dialogue (MIND) 생성 방법을 제안하여 대형 언어 모델(LLMs)의 수학적 추론 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: MIND를 활용하여 OpenWebMath (OWM)를 기반으로 합성 대화를 생성하고, 이를 통해 새로운 수학 데이터셋인 MIND-OWM을 만듭니다. 실험 결과, 대화 참여자 간의 지식 격차를 포함하는 것이 고품질 수학 데이터를 생성하는 데 필수적임을 보여줍니다. 또한, 합성 데이터와 원본 데이터를 사전 학습(pretraining) 시 효과적으로 포맷하고 통합하는 방법을 식별하였습니다.

- **Performance Highlights**: MIND-OWM에서 사전 학습된 모델은 원본 데이터만으로 사전 학습된 모델 대비 수학적 추론에서 상당한 향상을 보였습니다 (GSM8K: +13.42%, MATH: +2.30%). 또한, 전문 지식(MMLU: +4.55%, MMLU-STEM: +4.28%) 및 일반적인 추론 과제(GENERAL REASONING: +2.51%)에서도 우수한 성능을 기록했습니다.



### IMAS: A Comprehensive Agentic Approach to Rural Healthcare Delivery (https://arxiv.org/abs/2410.12868)
- **What's New**: COVID-19 이후, 농촌 지역의 의료 접근성 문제 해결을 위한 첨단 의료 보조 시스템(IMAS) 제안

- **Technical Details**: IMAS는 Large Language Models (LLMs)와 다섯 가지 주요 구성 요소(번역, 의료 복잡성 평가, 전문가 네트워크 통합, 최종 의료 조언 생성, 응답 단순화)로 구성되어 있습니다.

- **Performance Highlights**: IMAS는 MedQA, PubMedQA, JAMA 데이터셋을 통해 효과성을 입증하였으며, 특히 저소득 및 정보 소외 지역사회의 의료 근로자들에게 더 쉽게 접근할 수 있도록 지원합니다.



### A Dutch Financial Large Language Mod (https://arxiv.org/abs/2410.12835)
Comments:
          9 pages, 1 figure, accepted at ACM ICAIF'24

- **What's New**: 이 논문은 다양한 금융 과제를 위해 특별히 설계되고 최적화된 최초의 네덜란드어 금융 대형 언어 모델(LLM)인 FinGEITje를 소개합니다. 이와 함께 140,000개 이상의 샘플로 구성된 전문 네덜란드어 금융 지시 조정 데이터셋이 공개되며, 자동 번역 및 데이터 처리 방법을 활용하여 구축되었습니다.

- **Technical Details**: FinGEITje는 금융 과제를 위한 첫 번째 네덜란드어 LLM으로, 데이터셋 생성을 위한 공개 데이터 구축 방법론이 제공됩니다. 또한 독립 평가자로서 LLM을 활용하는 자동 평가 방법을 도입하여 성능 평가 시 수작업 개입을 줄였습니다.

- **Performance Highlights**: FinGEITje는 다섯 가지 주요 네덜란드어 및 영어 금융 과제에서 우수한 성능을 나타내며, 금융 뉴스 및 소셜 미디어 게시물의 감정 분류, 금융 문서에서의 주요 개체 식별, 가격 변동에 대한 주장 검증을 위한 뉴스 헤드라인 분류, 금융 관계 추출 및 특정 금융 질의 응답 등의 중요한 애플리케이션을 제공합니다.



### Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspectiv (https://arxiv.org/abs/2410.12816)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 CLIP 모델의 두 가지 정렬 문제인 작업 불일치(task misalignment) 및 데이터 불일치(data misalignment)를 해결하기 위한 방법을 제안합니다. 특히, 데이터 불일치가 다운스트림 작업에서 성능에 미치는 영향을 분석하고 Causality-Guided Semantic Decoupling and Classification (CDC) 방법론을 개발하여 이 문제를 해결합니다.

- **Technical Details**: CDC 방법론은 두 가지 주요 구성 요소인 Visual-Language Dual Semantic Decoupling (VSD)와 Decoupled Semantic Trusted Classification (DSTC)로 이루어져 있습니다. VSD는 다양한 의미를 표현하는 여러 프롬프트 템플릿을 모델에 통합하여 학습합니다. DSTC는 각 층에서 분리된 의미에 기반하여 분류 작업을 독립적으로 수행하며, 예측의 불확실성을 동시에 추정합니다.

- **Performance Highlights**: 다양한 데이터셋과 여러 작업에서 진행된 실험 결과, CDC 방법론이 CLIP의 성능을 유의미하게 향상시킴을 보여주었습니다. 특히, 새로운 클래스에 대한 인식 성능이 개선되는 효과가 있음을 확인했습니다.



### From Measurement Instruments to Data: Leveraging Theory-Driven Synthetic Training Data for Classifying Social Constructs (https://arxiv.org/abs/2410.12622)
- **What's New**: 이 논문은 사회적 구조를 측정하는 데 있어 이론 기반의 합성(training) 데이터의 잠재력을 체계적으로 조사합니다. 이를 통해 사회 과학에서의 측정 도구에서 얻은 지식을 합성 데이터 생성에 어떻게 활용할 수 있는지를 탐구합니다.

- **Technical Details**: 연구자는 두 가지 초점을 두어 성차별(sexism)과 정치적 주제를 측정합니다. 연구의 핵심 질문은 이론 기반의 합성 데이터가 사회적 구조 측정의 성과를 향상시킬 수 있는가입니다. 논문에서는 annotation codebooks와 설문 척도를 사용하여 데이터 생성을 이끌어냅니다.

- **Performance Highlights**: 정치 주제 연구에서는 실제 데이터의 30-90%를 합성 데이터로 교체했을 때 성과가 유지되거나 소폭 향상되었습니다. 반면 성차별 연구에서는 합성 데이터의 비율이 증가할수록 모델의 성과가 떨어지는 경향을 보였습니다.



### LLM-based Cognitive Models of Students with Misconceptions (https://arxiv.org/abs/2410.12294)
- **What's New**: 이 논문은 AI 기반 교육 기술에서 학생 인지를 정확하게 모델링하는 것의 중요성을 강조하며, 학생 모델링에서 잘못된 인식을 포함한 정확한 문제 해결을 동시에 충족하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: MalAlgoPy라는 새로운 Python 라이브러리를 통해 대수 문제 해결 과정에서 학생의 해결 패턴을 반영하는 데이터셋을 생성하며, 이를 그래프 기반으로 나타냅니다. 모델은 학생 모델(Cognitive Student Models, CSM)로서, 잘못된 인식과 올바른 지식을 동시에 반영하는 방법으로 훈련됩니다.

- **Performance Highlights**: 잘못된 인식 예제로 훈련된 LLMs는 문제를 올바르게 해결하는 능력이 감소했으나, 훈련 데이터에서 올바른 예제와 잘못된 예제의 비율을 조정함으로써 두 가지 속성을 모두 만족하는 CSM을 개발할 수 있음을 보여주었습니다.



New uploads on arXiv(cs.IR)

### Pessimistic Evaluation (https://arxiv.org/abs/2410.13680)
- **What's New**: 이 연구는 정보 접근 시스템 평가의 새로운 접근 방식을 제안합니다. 기존의 평균 유틸리티 기반 평가 방식이 아닌, 최악의 경우 유틸리티에 초점을 맞춘 비관적(pessimistic) 평가 방법을 주장합니다.

- **Technical Details**: 비관적 평가는 정보 과학 커뮤니티의 평등한 정보 접근에 대한 기존 연구에 기반을 두고 있으며, 정치 이론에서 잘 제정된 방법론을 토대로 합니다. 연구에서는 lexicographic minimum을 비관적 평가의 이론적으로 타당한 방법으로 소개합니다.

- **Performance Highlights**: 이 연구 결과는 비관적 평가 방법이 시스템의 행동을 더 잘 이해하는 데 도움이 될 수 있으며, 특히 사회적 선이라는 원칙과 관련된 상황에서 기존의 평가 방법을 보완할 수 있음을 보여줍니다.



### Large Language Models as Narrative-Driven Recommenders (https://arxiv.org/abs/2410.13604)
Comments:
          Under review; 19 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 사용하여 자유형식의 텍스트로 표현된 영화 추천 요청에 대한 개인화된 추천을 제공하기 위한 새로운 접근 방식을 탐구하였습니다. 특히, reddit의 영화 추천 커뮤니티에서 수집된 데이터셋을 활용하여 38개의 오픈소스 및 클로즈드 소스 LLM의 성능을 비교하였습니다.

- **Technical Details**: 이 연구는 zero-shot, identity, few-shot 프롬프트 기법을 사용하여 LLM이 사용자 요청을 자연어로 처리하고 관련 영화를 추천할 수 있는지 평가합니다. 평가된 LLM은 크기에 따라 분류되며, 각 모델은 기본적인 zero-shot 프롬프트를 통해 추천 정확도를 높일 수 있음을 보여줍니다.

- **Performance Highlights**: LLMs는 기존의 추천 알고리즘보다 더 높은 성능을 보이며, 특히 GPT-4o는 기본 성능보다 70% 더 높은 추천 성능을 보였습니다. 중간 크기의 오픈소스 모델도 상대적으로 높은 성능을 유지하며 클로즈드 소스 모델과 비교하여 경쟁력을 보여주었습니다.



### Cross-Domain Sequential Recommendation via Neural Process (https://arxiv.org/abs/2410.13588)
Comments:
          Work in progress

- **What's New**: 이 논문은 Cross-Domain Sequential Recommendation (CDSR)에서 중첩되지 않은 사용자 행동의 잠재력을 활용하는 방법을 탐구합니다. CDSR의 기존 방법들이 주로 중첩 사용자 행동에 집중함에 따라 발생하는 한계를 극복하기 위해, 저자들은 새로운 CDSRNP라는 프레임워크를 제안합니다.

- **Technical Details**: CDSRNP는 메타 학습(meta-learning) 접근 방식을 활용하여, 지원 세트(support set)에서 관찰된 중첩 사용자 행동을 샘플링하고, 쿼리 세트(query set)에서 비중첩 사용자 예측을 지원합니다. 이는 Neural Processes(NP)를 이용하여 prior와 posterior 샘플 분포를 학습하고, 이는 다양한 도메인 간 상호작용 패턴을 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 저자들은 CDSRNP가 두 개의 실제 데이터 세트에서 기존의 최첨단 방법들과 비교하여 눈에 띄는 성능 향상을 달성했음을 보여주었습니다. CDSRNP는 중첩 사용자 행동을 고려함으로써 CDSR의 새로운 패러다임을 제시하고, 비중첩 사용자 예측을 위한 세밀한 관심 적응 레이어를 설계하였습니다.



### Generate and Instantiate What You Prefer: Text-Guided Diffusion for Sequential Recommendation (https://arxiv.org/abs/2410.13428)
- **What's New**: 최근 생성 추천 시스템의 발전 특히 Sequential Recommendation (순차 추천) 작업에서 새로운 아이템에 대한 일반화 능력을 띄게 되었습니다. 이와 관련된 Diffusion-based Generative Recommendation (확산 기반 생성 추천)이 데이터 분포를 캡처하고 고품질 샘플을 생성할 수 있는 능력을 가지며 효과적인 도구로 부각되었습니다. 그러나 두 가지 주요 문제 점이 지적되었습니다: 1) 오라클 아이템의 데이터 분포 일관성 부족, 2) 역사적 상호 작용을 넘어서 더 유익한 제어 신호로의 확장 어려움.

- **Technical Details**: iDreamRec (intention-guided DreamRec)은 구체적인 사전 지식을 활용해 아이템 임베딩을 구축하며, 상세한 텍스트 설명과 고급 Text Embedding Models (TEM)을 통해 수치화된 데이터로 변환합니다. 이를 통해 생성 과정에서 오라클 아이템 생성을 위한 제어 신호인 의도 지침을 통합할 수 있습니다. TMP와 결합한 상태에서, iDreamRec은 조건부 확산 모델을 훈련시켜 아이템 임베딩을 정렬합니다.

- **Performance Highlights**: 4개의 데이터 세트에서 실험 결과, iDreamRec는 기존의 확산 기반 생성 추천 시스템들 (예: DreamRec, DiffRec)에 비해 향상된 성능을 보여주었으며, 의도 지침을 통합하여 보다 정밀하고 효과적인 추천 생성을 가능하게 하였습니다.



### Context-aware adaptive personalised recommendation: a meta-hybrid (https://arxiv.org/abs/2410.13374)
- **What's New**: 이번 논문에서는 정보를 종합할 수 있는 메타 하이브리드 추천 시스템을 제안하고 있습니다. 이 시스템은 사용자마다 최적의 추천 알고리즘을 예측하기 위해 Machine Learning을 사용할 수 있도록 개발되었습니다.

- **Technical Details**: 제안된 메타 하이브리드 추천 시스템은 사용자에 대한 맥락적 및 선호 정보를 기반으로 다양한 추천 알고리즘 중에서 최고의 성능을 발휘하는 것을 선택합니다. 오프라인 평가를 위해 MovieLens와 The Movie DB 데이터셋을 사용하였으며, 이를 통해 각 세션과 사용자에 적합한 추천기를 선택할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 메타 하이브리드 추천 시스템은 정상화된 Discounted Gain과 Root Mean Square Error 메트릭에서 기존의 개별 접근 방식보다 20-50% 더 나은 성능을 보였습니다. 그러나 사용자의 표준 정보 기반으로 최적 성능을 달성하기란 어려운 과제입니다.



### Starbucks: Improved Training for 2D Matryoshka Embeddings (https://arxiv.org/abs/2410.13230)
- **What's New**: 이 논문에서는 Starbucks라는 새로운 Matryoshka 유사 임베딩 모델 훈련 전략을 제안합니다. 이 전략은 미세 조정(fine-tuning) 및 사전 훈련(pre-training) 단계를 포함하여 2D Matryoshka 모델의 접근성을 높이고 효과성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: Starbucks는 두 가지 주요 프로세스로 구성됩니다: Starbucks Masked Autoencoding (SMAE) 사전 훈련 및 Starbucks Representation Learning (SRL) 미세 조정입니다. SRL 단계에서 특정 레이어-차원 쌍의 고정 목록을 제공하여 손실을 계산하고, SMAE에서는 다양한 레이어-차원 쌍으로의 마스크 자기 인코딩 언어 모델링을 적용합니다.

- **Performance Highlights**: 실험 결과, Starbucks 모델은 2D Matryoshka 모델보다 성능이 향상되어 별도로 훈련한 모델과 동등한 효과성을 보여주었습니다. 이는 의미적 텍스트 유사성 및 검색 벤치마크에서 확인되었습니다.



### Transformers4NewsRec: A Transformer-based News Recommendation Framework (https://arxiv.org/abs/2410.13125)
- **What's New**: Transformers4NewsRec는 새로운 Python 프레임워크로, 다양한 뉴스 추천 모델의 성능을 비교하고 통합할 수 있는 유연한 기능을 제공합니다. 이 프레임워크는 Transformer 기반 아키텍처와 전통적인 DL 기법, 그래프 기반 방법을 활용하여 뉴스 추천을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 프레임워크는 데이터 모듈, 모델 모듈, 평가 모듈, 유틸리티 도구 모듈의 4가지 핵심 컴포넌트로 구성되어 있으며, 다양한 모델과 데이터셋을 시험하고 실험 설정을 유연하게 조정할 수 있는 명령 줄 인터페이스를 제공합니다. 또한, 기존 제로 패딩 방식을 대체하는 새로운 연결 기반 배치 방법을 제안하여 훈련 및 평가 속도를 100% 이상 향상시킵니다.

- **Performance Highlights**: BATM-NR와 GLORY 모델은 MIND-SMALL 및 MIND-LARGE 데이터셋에서 전통적인 모델들보다 더 높은 성능을 보여주었으며, 특히 뉴스 제목과 본문을 함께 사용할 때 그 성능이 크게 향상되었습니다. BERT와 같은 사전 훈련된 언어 모델을 활용한 경우, 모델 성능이 더욱 개선되는 경향을 보였습니다.



### Preference Diffusion for Recommendation (https://arxiv.org/abs/2410.13117)
- **What's New**: PreferDiff는 신규 개인화 순위 손실 함수로, 기존의 추천 시스템들이 사용하는 전통적인 목표 대신에Diffusion Models(확산 모델) 특화된 최적화 목표를 제안합니다.

- **Technical Details**: PreferDiff는 BPR(Bayesian Personalized Ranking)을 로그 가능도 순위 목표로 변환하고 여러 개의 네거티브 샘플을 통합하여 사용자 선호도를 더 잘 포착하도록 설계되었습니다. 변분 추론(variational inference)을 이용해 계산의 어려움을 극복하고 오차 기준에서 MSE 대신 cosine error를 적용하여 추천 작업에 대한 정렬을 개선합니다. 또한, 생성(generation) 및 선호(preference) 간의 균형을 맞춤으로써 DMs의 학습 안정성을 향상시킵니다.

- **Performance Highlights**: 세 가지 벤치마크에서 진행된 실험을 통해, PreferDiff는 우수한 추천 성능을 보였으며 일반적인 연속 추천(sequential recommendation) 능력에서 주목할 만한 결과를 나타냈습니다.



### Leveraging Large Language Models to Enhance Personalized Recommendations in E-commerc (https://arxiv.org/abs/2410.12829)
Comments:
          This paper has been accepted by the 5th International Conference on Electrical, Communication and Computer Engineering (ICECCE 2024)

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 전자상거래의 개인화 추천 시스템에 어떻게 적용될 수 있는지를 심도 있게 탐구합니다. 전통적인 추천 알고리즘의 한계를 극복하기 위해 LLM 기반의 추천 시스템 프레임워크를 제안하였습니다.

- **Technical Details**: 비교 실험을 통해 LLM 기반 추천 모델이 정밀도(precision), 재현율(recall), F1 점수, 평균 클릭률(CTR), 추천 다양성 등의 여러 핵심 지표에서 유의미한 개선을 보여주었습니다. LLM 모델의 정밀도는 0.75에서 0.82로 향상되었고, 재현율은 0.68에서 0.77로 증가하였으며, F1 점수는 0.71에서 0.79로 향상되었습니다. 평균 클릭률은 0.56에서 0.63으로 증가하였고, 추천 다양성은 41.2% 증가하여 0.34에서 0.48로 개선되었습니다.

- **Performance Highlights**: LLM은 사용자 의견과 제품 설명 데이터에 대한 심층적인 의미 이해를 통해 사용자의 암묵적인 요구를 효과적으로 파악하고, 맥락 데이터(contextual data)를 결합하여 동적인 추천을 생성하여 더 정확하고 다양한 결과를 제공합니다. 이는 사용자 경험을 개선하고 플랫폼의 판매 성장에 기여할 수 있는 중요한 연구 결과입니다.



### Optimizing and Evaluating Enterprise Retrieval-Augmented Generation (RAG): A Content Design Perspectiv (https://arxiv.org/abs/2410.12812)
Comments:
          6 pages, 4 figures, to be published in ICAAI 2024 conference proceedings

- **What's New**: 본 논문에서는 리뷰 기반 생성(Retrieval-augmented generation, RAG) 솔루션의 구현 및 유지 관리 경험을 공유하고 있습니다. 기존 RAG 문헌에서 일반적으로 제시된 패턴과의 차별성을 강조하며, 모듈화되어 있고 모델에 의존하지 않는 접근 방식을 중심으로 해결책을 제시합니다.

- **Technical Details**: RAG의 기본 원리는 지식 기반에서 관련 콘텐츠를 검색하고, 이 콘텐츠에 기반한 프롬프트를 작성한 후, LLM에게 출력을 생성하도록 요청하는 것입니다. 그러나 본 팀의 RAG 솔루션은 벡터 데이터베이스에 의존하지 않아 다양한 검색 기법과 LLM을 사용합니다. 지식 기반 콘텐츠 최적화 및 실시간 사용자 질문에 대한 테스트와 평가 방안도 다루고 있습니다.

- **Performance Highlights**: 기존 RAG 평가 지표는 기존 사용자 질문에 대한 응답 평가에 유용하지 않아, 유연한 '인간 선도' 접근 방식이 필요하다는 점을 강조하고 있습니다. 지식 기반 콘텐츠 개선을 통해 RAG 솔루션의 성공 여부에 큰 영향을 미칠 수 있음을 보여줍니다.



### Ads Supply Personalization via Doubly Robust Learning (https://arxiv.org/abs/2410.12799)
Comments:
          Accepted by CIKM'24

- **What's New**: 이 논문에서는 광고 공급 개인화를 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 데이터 수집 정책을 통해 정보를 최적 활용하여 장기적 치료 효과 추정의 정확성을 크게 향상시킵니다. 또한, 낮은 복잡도로 인해 기존 방법들보다 계산 비용이 절감되고, 대규모 애플리케이션에 확장 가능하다는 장점을 지니고 있습니다.

- **Technical Details**: 제안된 프레임워크는 Doubly Robust Learning (DRL)을 기반으로 하여 장기적인 인과 효과를 모델링하는 가벼운 솔루션을 제공합니다. DRT 프레임워크는 데이터 수집과 모델링 단계에서 정보를 효율적으로 활용해 성능 향상 및 모델 복잡성 감소를 이끌어내며, 기존의 광고 및 유기 콘텐츠 배포 시스템과 통합이 용이합니다.

- **Performance Highlights**: 오프라인 실험과 온라인 생산 테스트를 통해, 이 프레임워크는 몇 달 간에 걸쳐 비즈니스 주요 지표에서 상당한 개선을 지속적으로 보여주었으며, 세계에서 가장 큰 소셜 미디어 플랫폼 중 하나에 완전히 배포되었습니다.



### Disaggregating Embedding Recommendation Systems with FlexEMR (https://arxiv.org/abs/2410.12794)
- **What's New**: FlexEMR는 embedding 기반 추천 (EMR) 모델의 비효율성을 해결하기 위한 새로운 분산 시스템으로, 네트워크 데이터 전송의 효율성을 개선하고 총 비용 소유권을 줄이기 위한 디자인을 제안합니다.

- **Technical Details**: FlexEMR는 두 가지 기술 세트를 통해 네트워크 문제를 해결합니다. 첫 번째는 embedding 조회의 시간적 및 공간적 지역성을 활용하여 데이터 이동을 줄이고, 두 번째는 다중 스레드 RDMA 엔진을 설계하여 동시 조회 하위 요청을 최적화하는 것입니다.

- **Performance Highlights**: 초기 프로토타입에서 FlexEMR은 원격 embedding 조회의 성능을 향상시켰으며, queuing latency를 크게 줄이고, 응답 혼잡을 완화하는 데 기여했습니다.



### Knowledge-Aware Query Expansion with Large Language Models for Textual and Relational Retrieva (https://arxiv.org/abs/2410.13765)
- **What's New**: 이번 논문에서는 지식 그래프(knowledge graph, KG)의 구조화된 문서 관계를 활용하여 LLM(대형 언어 모델)과 결합된 지식 인식 쿼리 확장 프레임워크를 제안합니다.

- **Technical Details**: 이 연구는 사용자 쿼리의 텍스트적 및 관계적 요구 사항을 모두 처리하기 위해 KG로부터의 관계를 활용합니다. 기존 KG 기반 메소드의 한계인 엔티티 기반 스코어링을 보완하기 위해, 문서 텍스트를 KG 노드 표현으로 사용하고 문서 기반 관계 필터링을 도입하여 Knowledge-Aware Retrieval (KAR)를 수행합니다.

- **Performance Highlights**: 세 가지 서로 다른 분야의 데이터셋을 대상으로 한 실험 결과, 우리의 방법이 최신 쿼리 확장 방법에 비해 성능이 우수하고 LLM 기반 검색 에이전트와 동등한 성능을 달성하는 것으로 나타났습니다.



### Disjointness Violations in Wikidata (https://arxiv.org/abs/2410.13707)
Comments:
          Sixth International Knowledge Graph and Semantic Web Conference

- **What's New**: 이 논문은 Wikidata에서의 불일치 체크(disjointness checks)의 현재 모델링을 분석하고, 이를 통해 발생하는 불일치 위반(disjointness violations)의 패턴과 원인을 확인하였습니다. SPARQL 쿼리를 사용해 각각의 원인을 규명하고, 서로 충돌하는 정보를 식별 및 수정할 수 있는 공식을 제시합니다.

- **Technical Details**: Wikidata는 1억 개 이상의 객체를 포함하는 대규모 지식 그래프입니다. 본 논문에서는 RDF(리소스 기술 프레임워크)를 사용하여 Wikidata에서 쌍별 불일치 클래스(pairwise disjoint classes)의 정보를 수집하였습니다. SPARQL 쿼리를 작성하여 불일치 유니온 문장(disjoint union statements)의 쌍을 찾아내었습니다.

- **Performance Highlights**: 논문에서 제안한 방식은 불일치 상황을 정량화하고, 성능을 개선하여 사용자가 문제를 식별하고 수정하는 효율성을 높이는 데 기여할 수 있습니다. 총 758개의 불일치 유니온 문장이 631개 클래스에서 생성되었으며, 7,027개의 쌍별 불일치 문장(pairwise disjoint statements)이 도출되었습니다.



### Comparing the Utility, Preference, and Performance of Course Material Search Functionality and Retrieval-Augmented Generation Large Language Model (RAG-LLM) AI Chatbots in Information-Seeking Tasks (https://arxiv.org/abs/2410.13326)
Comments:
          12 pages, 4 figures

- **What's New**: 최근의 대형 언어 모델(LLMs)을 활용한 AI 챗봇이 교육 지원 도구로서의 가능성을 탐구하기 위한 연구가 진행됨. 이 연구는 LLM 기반 챗봇의 성능을 전통적인 검색 기능과 비교하여 학생들을 지원하는 방법을 조사함.

- **Technical Details**: 실험에서는 14명의 참가자가 웹 소프트웨어 개발 과정의 과제를 수행. 참가자는 두 그룹으로 나뉘어 LLM 챗봇과 검색 기능에 대한 접근 순서를 다르게 설정. LLM 챗봇은 retrieval-augmented generation(RAG) 기술을 활용하여 추가 정보를 제공.

- **Performance Highlights**: LLM 기반 챗봇과 전통적인 검색 기능 모두 유용하다고 인식되었으며, 특정 과제에서 더 잘 작동하는 경향이 있음. LLM 챗봇을 먼저 사용한 그룹은 검색 기능을 더 선호했고, 그 반응은 연구에서 흥미로운 결과로 나타짐.



### SBI-RAG: Enhancing Math Word Problem Solving for Students through Schema-Based Instruction and Retrieval-Augmented Generation (https://arxiv.org/abs/2410.13293)
Comments:
          Accepted to the 4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 본 논문에서는 Schema-Based Instruction Retrieval-Augmented Generation (SBI-RAG) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 통합하여 수학 단어 문제(MWP)를 해결하는 과정을 지원하며, 기존의 Schema-Based Instruction(SBI) 방법을 바탕으로 발전했습니다.

- **Technical Details**: SBI-RAG는 기본적으로 네 가지 주요 부분으로 나뉩니다: 1) Schema Classifier, 2) Prompt Creation, 3) Context Retrieval, 4) Answer and Response Generation. Schema Classifier는 DistilBERT를 기반으로 하여 특정 문제에 적합한 schema를 예측하고, 그에 따라 schema-specific prompt를 생성합니다. 이후 Retrieval-Augmented Generation(RAG) 프레임워크를 이용하여 관련 문서를 검색하고, LLM을 통해 구체적인 단계별 해답을 생성합니다.

- **Performance Highlights**: GSM8K 데이터셋에서의 평가 결과, SBI-RAG는 GPT-4 및 GPT-3.5 Turbo와 비교하여 문제 해결의 정확성과 추론의 명료성을 향상시키는 데 효과적임을 보였습니다. 새로운 'reasoning score' 메트릭을 도입하여 LLM의 해결 과정의 질을 평가하였으며, 이는 학생들의 교육적 이점을 제공할 가능성이 있습니다.



### Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation (https://arxiv.org/abs/2410.13248)
- **What's New**: 최근 설명 가능한 추천 시스템에 대한 연구는 표준 텍스트 생성 문제로 접근하며, 모델을 예측된 텍스트와 실제 텍스트 간의 유사성을 기반으로 평가합니다. 그러나 이 접근법은 사용자(구매 후) 감정을 정확히 반영하는지 여부를 간과합니다. 이 연구에서는 사용자의 감정을 중점적으로 고려하는 새로운 데이터셋과 평가 방법을 소개합니다.

- **Technical Details**: 우리는 LLM(Long Language Model)을 사용하여 사용자 구매 후 리뷰에서 긍정적 및 부정적 의견을 명시적으로 추출하여 데이터셋을 구성합니다. 시스템을 평가할 때 생성된 설명이 1) 사용자 감정과 잘 일치하는지, 2) 목표 아이템에 대한 사용자 의견의 긍정적 및 부정적 식별을 정확히 수행하는지에 대한 두 가지 기준을 제안합니다.

- **Performance Highlights**: 여러 최신 모델을 우리의 데이터셋에서 벤치마킹하였으며, 기존 지표에서 높은 성과를 달성하더라도 생성된 설명이 사용자 감정과 잘 일치하지 않을 수 있음을 보여줍니다. 또한, 목표 아이템에 대한 사용자(예측된) 평가가 모델에 직접 입력될 경우, 기존 모델들이 보다 감정 인식적인 설명을 제공할 수 있음을 발견하였습니다.



### Research on Travel Route Planing Problems Based on Greedy Algorithm (https://arxiv.org/abs/2410.13226)
- **What's New**: 이 연구에서는 초기 경로 탐색 및 관광객의 개인화 요구를 충족하는 최적화된 경로 계획 알고리즘이 제안되었습니다. 특히, PCA(Principal Component Analysis) 및 KMO(Kaiser-Meyer-Olkin) 테스트와 TOPSIS(TOPSIS: Technique for Order Preference by Similarity to Ideal Solution) 기법을 통해 도시 평가 지표의 차원 축소 및 종합 평가를 수행했습니다.

- **Technical Details**: 연구에서는 PCA를 사용하여 도시 평가 지표의 차원을 축소하고, KMO 테스트를 통해 데이터 적합성을 판단하고, TOPSIS 및 엔트로피 가중치 방법을 통해 데이터를 종합 평가했습니다. 경로 최적화를 위해서는 그리디 알고리즘이 사용되었으며, 관광 명소 방문에 소요되는 시간을 고려한 경로 계획이 이루어졌습니다.

- **Performance Highlights**: 이 알고리즘은 352개의 도시에서 100개의 관광 명소 데이터를 활용하여 관광객에게 최적화된 여행 경로를 제공함으로써 여행 비용을 줄이고 현지 최적해(local optimum) 문제를 피하는 데 기여합니다. 결과적으로 관광객의 요구에 맞춘 맞춤형 경로 계획을 통해 효율적인 여행 경험을 지원합니다.



### MixEHR-Nest: Identifying Subphenotypes within Electronic Health Records through Hierarchical Guided-Topic Modeling (https://arxiv.org/abs/2410.13217)
- **What's New**: MixEHR-Nest는 전자 건강 기록(EHR) 데이터를 활용하여 고유한 하위 표현형(sub-phenotype) 주제를 유도하는 새로운 지침(topic model) 모델입니다. 이 모델은 경험적인 표현형 개념(PheCodes, CCS 코드를 포함)으로 초기화된 하위 주제를 탐지하여 질병 패턴을 더욱 세분화하여 나타냅니다.

- **Technical Details**: MixEHR-Nest는 다중 모달(multi-modal) EHR 데이터에서 1500개 이상의 표현형으로부터 뚜렷한 하위 표현형 주제를 유도할 수 있는 구조화된 하이라키(topic model)입니다. 이 모델은 각 환자의 의료 기록을 문서(document)로, 코드(예: ICD 코드)를 단어 토큰(word tokens)으로 간주하여 학습합니다. 이 연구는 하위 표현형 주제의 묘사, 다중 유형의 EHR 정보 학습, 높은 해석 가능성을 통한 자동 하위 표현형 유도를 포함합니다.

- **Performance Highlights**: MixEHR-Nest는 ICU 환자 사망률 예측, 당뇨병 환자의 초기 인슐린 치료 예측에서 성능을 향상시켰습니다. 또한 MixEHR-Nest는 같은 표현형 아래에서 연령 분포의 뚜렷한 하위 표현형을 확인함으로써 다양한 질병에 걸쳐 질병의 진행 및 중증도를 예측하는 데 기여했습니다.



### Retrieval-Enhanced Named Entity Recognition (https://arxiv.org/abs/2410.13118)
Comments:
          13 pages, 6 figures, 3 tables

- **What's New**: RENER (Retrieval-Enhanced Named Entity Recognition)는 In-Context Learning (ICL) 및 정보 검색 기술을 결합하여 명명된 개체 인식(NER) 작업에서 성능을 향상시키기 위해 제안된 새로운 방법입니다. 이 방법은 입력 텍스트에 대해 유사한 예제를 검색하고 이를 언어 모델에 통합하여 NER을 수행할 수 있도록 합니다.

- **Technical Details**: RENER는 언어 모델과 정보 검색 알고리즘에 독립적이며, 언어 모델과의 결합이 최소화된 상태로 새로운 명명된 개체를 인식하는 데 사용할 수 있습니다. 이 과정에서 언어 모델에 직접적인 의존성이 없으며, 다양한 NER 도메인에 쉽게 배포할 수 있습니다. 또한 CrossNER 컬렉션에서의 실험 결과, RENER는 최신 기술(State-of-the-Art) 성능을 달성하였으며, 정보 검색 기술을 사용할 경우 F-score를 최대 11% 상승시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: RENER는 CrossNER 데이터셋에서 최신 성능을 달성하였으며, 정보 검색 기법을 활용함으로써 비슷한 시스템 대비 성능을 최대 11% 향상시킬 수 있었습니다. 이는 NER 작업에서 ICL과 RAG 기법을 성공적으로 결합했음을 나타냅니다.



### Is Semantic Chunking Worth the Computational Cost? (https://arxiv.org/abs/2410.13070)
- **What's New**: 최근 Retrieval-Augmented Generation (RAG) 시스템에서 문서를 의미적으로 일관된 세그먼트로 분할하는 semantic chunking이 인기를 얻고 있습니다. 본 연구는 semantic chunking이 보다 간단한 fixed-size chunking에 비해 실질적인 이점을 제공하는지에 대한 체계적인 평가를 진행했습니다.

- **Technical Details**: 연구팀은 document retrieval, evidence retrieval, answer generation 세 가지 일반적인 retrieval 관련 작업을 통해 semantic chunking의 효용성을 평가했으며, 다양한 chunking 전략을 비교하여 최적의 성능을 갖는 chunker를 확인했습니다. 또한, 두 가지 chunking 전략으로 fixed-size chunker와 breakpoint-based semantic chunker, clustering-based semantic chunker를 채택하여 평가하였습니다.

- **Performance Highlights**: 결과적으로, semantic chunking이 특정 상황에서 일부 이점을 보였지만, 이러한 이점들은 불일치하며 고정 크기 청크에 대한 계산 비용을 정당화할 만큼 충분하지 않다는 것을 보였습니다. 이는 RAG 시스템에서 더 효율적이고 적응적인 chunking 전략의 필요성을 강조합니다.



### Supply Chain Network Extraction and Entity Classification Leveraging Large Language Models (https://arxiv.org/abs/2410.13051)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 자연어 처리(NLP) 및 대형 언어 모델(LLM)를 활용하여 비정형 텍스트 데이터를 기반으로 공급망 그래프를 구축하는 새로운 접근 방식을 제안합니다. 특히 토목 공학 산업을 사례 연구로 삼아 LLM이 기업, 프로젝트 등의 숨겨진 관계를 발견할 수 있는 방법을 보여줍니다.

- **Technical Details**: 본 연구는 데이터 수집, 프롬프트 엔지니어링, 그래프 구축, 엔티티 분류의 네 가지 주요 단계로 구성된 방법론을 적용합니다. 데이터 수집은 공개 소스의 뉴스 기사를 통해 이루어지며, 각 기업에 대해 2018년부터 2023년까지 연도별로 최소 10개의 뉴스 기사를 수집하여 총 50개의 원시 텍스트 데이터 포인트를 확보합니다. 이를 통해 각 기업의 활동에 대한 포괄적인 관점을 유지합니다.

- **Performance Highlights**: LLM으로 특정 산업에 맞춰 세부 조정(fine-tuning)을 수행함으로써 엔티티 분류의 정확도가 향상되었으며, 이는 산업별 공급망 분석의 잠재력을 강조합니다. 본 연구는 LLM을 통해 공급망 네트워크 모델링의 자동화를 가능하게 한 첫 번째 사례로, 공급망 동학에 대한 깊이 있는 통찰력을 제공합니다.



### LLM Confidence Evaluation Measures in Zero-Shot CSS Classification (https://arxiv.org/abs/2410.13047)
- **What's New**: 이 논문은 데이터 주석 작업에서의 대형 언어 모델(LLM)의 신뢰성을 평가하기 위해 세 가지 핵심 기여를 제안합니다. 첫째, 데이터 주석 작업을 위한 불확실성 정량화(UQ) 성능 측정 방법을 제안합니다. 둘째, 세 가지 서로 다른 LLM과 CSS 데이터 주석 작업에서 다섯 가지 UQ 전략을 처음으로 비교합니다. 셋째, LLM의 낮은 신뢰도 주석을 효과적으로 식별하고 잘못 레이블이 붙은 데이터를 발견하는 새로운 UQ 집계 전략을 소개합니다.

- **Technical Details**: 연구는 대형 언어 모델의 신뢰성을 평가하기 위해 여러 UQ 기법을 사용해 분석하였으며, 새로운 UQ 집계 전략을 제안하여 잘못 분류된 LLM 레이블 데이터를 식별하는 과정을 보다 간소화하였습니다. 이 논문은 다양한 UQ 방법을 비교하고, AUC(Area Under Curve) 분석을 통해 신뢰도 점수의 백분위수 기반 임계값을 적용하여 기술됩니다.

- **Performance Highlights**: 제안된 UQ 집계 전략은 기존 방법에 비해 개선된 성능을 보여주며, Human-in-the-loop 데이터 주석 프로세스를 획기적으로 개선할 수 있음을 입증했습니다. 이를 통해, LLM이 생성한 데이터 중 인간이 자원을 소모해야 할 데이터의 식별이 용이해졌습니다.



### LFOSum: Summarizing Long-form Opinions with Large Language Models (https://arxiv.org/abs/2410.13037)
- **What's New**: 이 논문에서는 온라인 리뷰의 대량 처리 및 요약을 위한 새로운 접근법을 제안합니다. 특히, 1천 개 이상의 리뷰로 구성된 새로운 데이터셋을 소개하며, 이를 기반으로 하는 LLM(대형 언어 모델) 기반 요약 기법을 제안합니다.

- **Technical Details**: LFOSum 데이터셋은 TripAdvisor에서 수집된 호텔 리뷰로, 각 엔티티는 1천 개 이상의 리뷰를 포함하고 있습니다. 두 가지의 훈련이 필요 없는 요약 방법, 즉 Retrieval-Augmented Generation (RAG)과 긴 맥락의 LLM을 이용하여 대량 리뷰 요약을 처리합니다. 사용자 맞춤형 요약을 위한 세 가지 제어 메커니즘(쿼리 제어, 감정 제어, 길이 제어)을 도입하여 사용자 요구에 맞춘 요약을 가능하게 합니다.

- **Performance Highlights**: LLM은 여전히 긴 형식의 요약에서 감정과 형식 준수의 균형을 맞추는 데 어려움을 겪고 있으나, 관련 정보를 집중적으로 추출할 경우 오픈 소스 모델이 효과적으로 간격을 좁힐 수 있음을 보여줍니다.



### Towards Computational Analysis of Pansori Singing (https://arxiv.org/abs/2410.12956)
Comments:
          Late-Breaking Demo Session of the 25th International Society for Music Information Retrieval (ISMIR) Conference, 2024

- **What's New**: 이 논문에서는 한국 전통 음악인 판소리의 오디오와 해당 전사(Transcription)를 기반으로 한 컴퓨터 분석을 도입하고, 현대의 Music Information Retrieval (MIR) 방식이 어떻게 전통 음악 분석에 활용될 수 있는지를 보여줍니다.

- **Technical Details**: 판소리의 기본 주파수(F0) 윤곽을 추출하기 위해 CREPE 알고리즘을 사용하였고, 노이즈를 줄이기 위해 신뢰 점수 0.6 미만의 F0 값을 필터링했습니다. 비트 감지를 위해 madmom 라이브러리를 활용하였으며, 12/4 박자를 유지하기 위해 수동으로 비트를 주석 처리했습니다. n-gram 알고리즘을 통해 다양한 다목(Daemok)에서 발생하는 패턴을 분석했습니다.

- **Performance Highlights**: 판소리에 나타나는 모드의 분석은 장르의 구조 및 음악 언어를 이해하는 데 중요합니다. 예를 들어, Jeokbyeokga는 Gyemyeonjo와 Ujo 두 가지 모드를 포함하고 있으며, 각 다목의 특정한 음조와 장식 기법을 분석하여 지배 모드를 평가할 수 있습니다. 또한 다목 간 유사한 패턴을 발견하고, 동적 비브라토가 감정 표현에 중요한 역할을 한다는 점도 강조하였습니다.



### REFINE on Scarce Data: Retrieval Enhancement through Fine-Tuning via Model Fusion of Embedding Models (https://arxiv.org/abs/2410.12890)
Comments:
          Accepted in AJCAI'24

- **What's New**: 본 논문에서는 데이터 부족 문제를 해결하기 위해 REFINE이라는 새로운 접근 방식을 제안합니다. 이 방법은 효과적인 검색을 개선하기 위해 사용 가능한 문서에서 합성 데이터를 생성하고, 모델 융합(Model Fusion) 기법을 통해 임베딩을 향상시킵니다.

- **Technical Details**: REFINE은 LLM(대규모 언어 모델)을 활용하여 사용 가능한 비지도 문서에서 대조적 훈련 데이터셋을 생성합니다. 생성된 데이터셋은 표준 파인튜닝 방법을 통해 임베딩 모델의 성능을 개선하며, 새로운 데이터 특정 학습을 포함하는 모델 융합 기법을 도입하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: SQUAD 및 RAG-12000 데이터셋과 독점 TOURISM 데이터셋에서 실험을 수행한 결과, REFINE이 적용된 표준 파인튜닝이 기본 사전 훈련 모델에 비해 더 나은 성능을 보였고, TOURISM 데이터셋에서는 5.76%, SQUAD 데이터셋에서는 6.58%의 개선을, RAG-12000 데이터셋에서는 0.32%의 향상을 기록했습니다.



### AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency with Topic Filtering and Iterative Reasoning (https://arxiv.org/abs/2410.12886)
- **What's New**: AT-RAG라는 새로운 멀티스텝 RAG 모델을 제안하여, 복잡한 다중 단계 쿼리를 보다 효율적으로 처리하는 방법을 소개합니다.

- **Technical Details**: AT-RAG는 BERTopic을 활용하여 쿼리의 주제를 동적으로 할당함으로써 문서 검색 및 추론 과정의 정확성과 효율성을 향상시킵니다. 이 모델은 Chain-of-Thought (CoT) 추론을 통합하여 반복적인 문서 검색 및 추론을 가능하게 합니다.

- **Performance Highlights**: AT-RAG는 기존 RAG 모델 대비 Accuracy, Completeness, Relevance에서 현저한 개선을 보였으며, 특히 의료 QA와 같은 복잡한 도메인-specific 문제 해결에 적합합니다. 모델은 다양한 benchmark dataset에서 높은 성능을 발휘하였고, 검색 시간을 줄이면서 높은 정밀도를 유지합니다.



### Enhancing Affinity Propagation for Improved Public Sentiment Insights (https://arxiv.org/abs/2410.12862)
- **What's New**: 이 연구는 감독 학습(supervised learning)에 의존하지 않고 감정 분석(sentiment analysis)을 수행하기 위한 비감독 학습(unsupervised learning) 기술을 도입합니다. 특히 Affinity Propagation (AP) 클러스터링 기법을 사용합니다.

- **Technical Details**: AP 클러스터링은 사전 정의된 클러스터 수 없이 텍스트 데이터를 자연적인 패턴에 따라 그룹화합니다. 이 논문에서는 텍스트 표현을 위한 TF-IDF 벡터화(TF-IDF Vectorization)와 차원 축소(principal component analysis, PCA) 기법을 사용하여 AP 클러스터링과 K-평균 클러스터링(K-means clustering)을 비교합니다. AP는 Agglomerative Hierarchical Clustering과 결합하여 성능을 향상시킵니다.

- **Performance Highlights**: AP와 Agglomerative Hierarchical Clustering의 조합이 K-평균보다 현저히 더 우수한 성능을 보였으며, 실험 평가는 Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Index를 통해 진행되었습니다. 이 연구는 널리 사용되는 레이블 데이터에 대한 필요 없이 대중 감정을 분석할 수 있는 스케일 가능하고 효율적인 비감독 학습 프레임워크를 제안하여 자연어 처리(NLP) 분야에 기여합니다.



### Enhancing Long Context Performance in LLMs Through Inner Loop Query Mechanism (https://arxiv.org/abs/2410.12859)
- **What's New**: 이번 논문에서는 Inner Loop Memory Augmented Tree Retrieval (ILM-TR)이라는 혁신적인 접근법을 통해 복잡한 질문에 대한 보다 깊이 있는 답변 생성을 가능하게 하는 새로운 메모리 체계를 도입합니다. 이 메커니즘은 초기 질문뿐만 아니라 중간 결과에 기반한 내부 루프 쿼리를 활용하여 정보를 검색합니다.

- **Technical Details**: ILM-TR 방법은 기본적으로 두 부분으로 구성되어 있습니다: retriever와 inner-loop query. Retriever 부분에서는 RAPTOR의 트리 빌드 방법을 사용하여 원시 데이터를 짧고 연속적인 텍스트 청크로 분할하고, 각 청크의 요약을 생성합니다. Inner-loop 쿼리는 LLM을 사용하여 최종 답변을 생성하며, Short-Term Memory (STM)라는 영역에 정보를 저장하고, 전달된 데이터를 바탕으로 반복적으로 쿼리를 수행합니다.

- **Performance Highlights**: ILM-TR 시스템은 Multi-Needle In A Haystack (M-NIAH) 및 BABILong과 같은 표준 긴 컨텍스트 벤치마크에서 기존 RAG 방법을 초월하는 성능을 보여주며, 500k tokens까지 컨텍스트 길이가 증가해도 성능 저하 없이 지속적인 성능을 유지합니다.



### A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions (https://arxiv.org/abs/2410.12837)
Comments:
          4 Figures

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG)의 발전 과정을 포괄적으로 조사하며, 기존 개념에서 최신 기술에 이르기까지의 변화를 설명합니다. RAG는 검색 메커니즘과 생성 언어 모델을 결합하여 출력의 정확성을 높이며, LLMs의 주요 제한 사항을 해결합니다.

- **Technical Details**: RAG의 기본 아키텍처는 지식 집약적인 작업을 처리하기 위해 검색과 생성을 어떻게 통합하는지에 중점을 둡니다. 논문에서는 retrieval-augmented language models에서의 주요 혁신과 질문 답변, 요약 및 지식 기반 작업 등 다양한 도메인에서의 응용 사례를 자세히 리뷰합니다.

- **Performance Highlights**: 최근 연구 성과는 retrieval 효율성을 개선하기 위한 새로운 방법을 강조하고 있으며, RAG의 연구 방향으로는 모델의 견고성 향상, RAG 모델의 적용 범위 확대 및 사회적 함의 문제 다루기가 제안됩니다.



### Predicting the Geolocation of Tweets Using transformer models on Customized Data (https://arxiv.org/abs/2303.07865)
Comments:
          31 pages, 5 tables, 9 figures

- **What's New**: 이번 연구는 트위터 사용자 및 트윗의 지리적 위치 예측을 위한 유연한 접근 방식을 제공합니다. 연구진은 자연어 처리(NLP) 기술인 신경망을 활용하여 위도와 경도로 구성된 위치 좌표를 추정하고, 이차원 가우시안 혼합 모델(GMM)을 적용하여 보다 정확한 예측을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 미리 훈련된 Bidirectional Encoder Representations from Transformers(BERT)를 기반으로 하여 트위터 데이터셋에서 세부 조정되었습니다. 연구 결과, 전 세계 수준에서 평균 30km 미만의 오류를 기록하며, 미국의 경우 15km 미만의 오류로 더욱 향상된 예측 성능을 보였습니다.

- **Performance Highlights**: 제안된 방법론은 트윗의 내용 및 메타데이터 컨텍스트에 대한 텍스트 특징을 훈련 및 평가에 사용했습니다. 연구팀은 전체 트위터 데이터에서 단 1-2%만이 정확한 지리적 좌표를 지닌 메타데이터로 구분됨을 강조하며, 이로 인해 효과적인 지리적 위치 예측의 필요성을 언급합니다.



New uploads on arXiv(cs.CV)

### UniDrive: Towards Universal Driving Perception Across Camera Configurations (https://arxiv.org/abs/2410.13864)
Comments:
          Preprint; 14 pages, 5 figures, 2 tables; Code at this https URL

- **What's New**: 이번 논문에서는 다양한 카메라 구성에 대해 범용적인 인식을 달성하기 위한 새로운 프레임워크인 UniDrive를 제안합니다. 이 프레임워크는 여러 가상 카메라를 통합하여 사용자의 운전 인식 모델을 최적화합니다.

- **Technical Details**: UniDrive는 통합된 가상 카메라 환경을 사용하고, 지면을 인식하는 프로젝션 방법을 통해 원본 이미지를 가상 뷰로 변환합니다. 또한, 원본 카메라와 가상 카메라 간의 예측 프로젝션 오류를 최소화하여 구성 최적화를 수행합니다. 이 방법은 기존의 3D 인식 메서드에 플러그 앤 플레이 모듈로 적용할 수 있습니다.

- **Performance Highlights**: 실험 결과, UniDrive는 하나의 특정 카메라 구성에서 교육된 모델이 다양한 카메라 구성에 잘 일반화될 수 있도록 하며, 성능 저하를 최소화합니다. CARLA의 데이터셋을 이용해 다양한 카메라 구성에서 효율성을 검증하였습니다.



### Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens (https://arxiv.org/abs/2410.13863)
Comments:
          Tech report

- **What's New**: 이 연구는 text-to-image generation의 맥락에서 자가 회귀 모델(autoregressive models) 스케일링 문제를 조사합니다. 특히 이 모델들이 사용하는 token이 discrete인지 continuous인지, 그리고 token이 BERT 및 GPT와 유사한 transformer 아키텍처에서 무작위(random)로 생성되는지 또는 고정(raster) 순서로 생성되는지를 중심으로 성능을 비교합니다.

- **Technical Details**: 연구는 VQ(vector quantization) 방식이 이미지 생성 성능에 미치는 영향과 token 생성 순서가 시각적 품질에 미치는 영향을 분석합니다. Fluid라는 새로운 random-order autoregressive 모델을 continuous token으로 학습시켜, 10.5B 모델인 Fluid가 MS-COCO 30K에서 제로샷 FID(zero-shot FID) 6.16을 기록했습니다.

- **Performance Highlights**: Fluid 모델은 FID와 GenEval 점수에서 다른 모델에 비해 우수한 성능을 보여주며, 특히 무작위 순서 모델이 raster 순서 모델에 비해 GenEval 점수에서 현저히 더 나은 결과를 보였습니다. 연구 결과는 비전 모델과 언어 모델 간의 스케일링 격차를 줄여주는 데 기여할 것으로 기대됩니다.



### DepthSplat: Connecting Gaussian Splatting and Depth (https://arxiv.org/abs/2410.13862)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 Gaussian splatting과 깊이 추정(depth estimation)을 결합하여 DepthSplat를 제안합니다. 이를 통해 두 기술 간의 상호작용을 연구하고, 개별 기술의 한계를 극복하며 성능을 강화합니다.

- **Technical Details**: DepthSplat는 사전 학습된 단안(depth from a single image) 특성을 활용하여 다중 뷰 깊이 모델을 개선합니다. 이 모델은 복잡한 장면에서 일관성 높은 깊이 예측을 수행할 수 있으며, Gaussian splatting 모듈은 완전하게 미분 가능하여 대규모 비구속 데이터셋에서 깊이 예측 모델을 사전 학습하는 새로운 방법을 제공합니다.

- **Performance Highlights**: DepthSplat는 ScanNet, RealEstate10K 및 DL3DV 데이터셋에서 최신 성능(State-of-the-art performance)을 기록했습니다. 이 결과는 Gaussian splatting과 깊이 추정의 연결이 상호 이점을 제공함을 보여줍니다.



### PUMA: Empowering Unified MLLM with Multi-granular Visual Generation (https://arxiv.org/abs/2410.13861)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 PUMA라는 새로운 접근 방식을 제안하고 있습니다. PUMA는 다양한 이미지 생성 작업의 서로 다른 세분화 요구를 통합하여 다양한 시각 작업을 처리할 수 있는 통합된 MLLM(다중 모드 대형 언어 모델) 프레임워크를 제공합니다.

- **Technical Details**: PUMA는 세 가지 주요 모듈로 구성됩니다: 1) 다양한 세분화 표현을 추출하는 이미지 인코더, 2) 다중 스케일 이미지 피쳐를 처리하는 자가 회귀 MLLM, 3) MLLM에서 생성된 피쳐를 다양한 세분화 수준에서 디코딩하는 특수화된 확산 기반 이미지 디코더. PUMA는 두 단계의 훈련 전략을 통해 최적화됩니다.

- **Performance Highlights**: PUMA는 이미지 이해, 텍스트-이미지 생성, 이미지 편집, 인페인팅, 컬러화 및 조건부 생성 등 다양한 다중 모드 작업을 처리할 수 있는 능력을 보여주며, 진정한 AGI(인공지능 일반화)를 향한 중요한 이정표로 자리잡고 있습니다.



### VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding (https://arxiv.org/abs/2410.13860)
Comments:
          CoRL 2024 Camera Ready. 25 pages. A novel zero-shot 3D visual grounding framework based solely on 2D images

- **What's New**: VLM-Grounder는 2D 이미지를 기반으로 한 새로운 zero-shot 3D visual grounding 프레임워크로, 기존의 객체 중심 정보에만 의존하는 방식의 한계를 극복합니다.

- **Technical Details**: 이 새로운 프레임워크는 이미지 시퀀스를 동적으로 스티칭(stitching)하고, 목표 객체를 찾기 위한 grounding 및 feedback 체계를 적용하며, 3D boundary box를 정확히 추정하기 위해 multi-view ensemble projection을 사용합니다. 이 과정에서 VLM(GPT-4V)을 활용하여 사용자 쿼리를 분석합니다.

- **Performance Highlights**: ScanRefer와 Nr3D 데이터셋을 대상으로 한 실험에서 VLM-Grounder는 각각 51.6%의 Acc@0.25와 48.0%의 Acc를 기록하며, 이전의 zero-shot 방법들을 능가하였습니다.



### $\gamma-$MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models (https://arxiv.org/abs/2410.13859)
- **What's New**: 이 논문에서는 기존의 멀티모달 대형 언어 모델(MLLMs)에서 발생하는 높은 계산 비용 문제를 해결하기 위한 새로운 접근법인 γ-MoD를 제안합니다. 이 방법은 "activated tokens"의 관점에서 모델의 효율성을 극대화합니다.

- **Technical Details**: γ-MoD는 주의 맵의 랭크(rank of attention maps, ARank)를 사용하여 각 레이어의 중복성을 측정하고 중복된 레이어를 MoD 레이어로 변환하는 전략입니다. 이를 통해 MLLM의 90% 이상의 밀집(dense) 레이어를 MoD로 효과적으로 변환할 수 있습니다.

- **Performance Highlights**: 실험 결과, γ-MoD는 LLaVA-HR 모델의 학습 및 추론 시간에서 각각 31.0% 및 53.2%를 단축시키며, 성능 저하는 단 1.5%로 유지됩니다. 이는 γ-MoD의 일반화 능력이 뛰어남을 나타냅니다.



### Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2410.13848)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 다양한 모드의 이해 및 생성을 통합한 새로운 자율 회귀 프레임워크인 Janus를 소개합니다. 기존 연구는 주로 단일 시각 인코더를 사용했으나, Janus는 시각 인코딩을 별도의 경로로 분리하여 성능과 유연성을 향상시켰습니다.

- **Technical Details**: Janus는 고유한 transformer 아키텍처를 사용하여 시각 이해 및 생성을 위한 독립적인 인코딩 경로를 제공합니다. 이를 통해 이해와 생성 작업 사이의 정보를 분리하고, 각 작업에 가장 적합한 인코딩 방법을 선택할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: Janus는 기존의 통합 모델보다 뛰어난 성능을 보여주며, MMBench 및 SEED-Bench와 같은 벤치마크에서 최고 성과를 기록했습니다. 또한, DALL-E 2와 SDXL과 같은 특정 작업 모델을 초월하는 성과를 보였습니다.



### D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinemen (https://arxiv.org/abs/2410.13842)
- **What's New**: D-FINE을 제안하며, DETR 모델 내에서 바운딩 박스 회귀 작업을 재정의하여 뛰어난 위치 정확도를 달성.

- **Technical Details**: 주요 구성 요소인 Fine-grained Distribution Refinement (FDR)와 Global Optimal Localization Self-Distillation (GO-LSD)를 통해, 고정 좌표 대신 확률 분포를 반복적으로 정제하여 정확도를 높임. 또한, GO-LSD는 깊은 레이어에서 세밀한 정보로부터 지식을 얕은 레이어로 이전하여 최적화를 단순화함.

- **Performance Highlights**: D-FINE-L와 D-FINE-X는 COCO 데이터셋에서 각각 54.0%와 55.8% AP를 기록하며, NVIDIA T4 GPU에서 각각 124 FPS 및 78 FPS의 속도를 유지. Objects365에서 사전 훈련 후 최대 59.3% AP 달성하며, 기존 실시간 감지기들을 초월함.



### VidPanos: Generative Panoramic Videos from Casual Panning Videos (https://arxiv.org/abs/2410.13832)
Comments:
          Project page at this https URL. To appear at SIGGRAPH Asia 2024 (conference track)

- **What's New**: 이 논문에서는 일반적인 동적인 장면에서 촬영된 패닝(panning) 비디오로부터 파노라마(panoramic) 비디오를 합성하는 새로운 방법을 제안합니다. 기존의 정적인 장면에서는 잘 알고 있던 스티칭(stitching) 문제를 넘어, 움직이는 물체가 포함된 장면의 연속성을 고려하여, 마치 광각(wide-angle) 카메라로 촬영한 것처럼 파노라마 비디오를 생성하는 방법론입니다.

- **Technical Details**: 이 접근법은 공간-시간 오프페인팅(space-time outpainting) 문제로 설정되며, 입력 비디오와 같은 길이를 가진 전체 파노라마 비디오를 생성하는 것을 목표로 합니다. 제안된 방법은 비디오 프레임을 등록하여 단일 비디오 볼륨을 만들고, 입력 비디오 바깥의 공간-시간 영역을 초기적으로 알 수 없는 상태로 두며, 그 후 이 알 수 없는 영역을 완성하는 과정을 포함합니다. 최근의 생성 모델(generative model)을 도입하고, 이 모델의 한계를 최소화하며 효과를 극대화할 수 있는 방법을 적용합니다.

- **Performance Highlights**: 제안된 시스템은 사람, 차량, 흐르는 물 등 다양한 실생활 장면에 대하여 비디오 파노라마를 생성할 수 있으며, 입력 비디오의 알려진 영역과 일관되게 동작하고, 리얼리스틱(realistic)하게 보이는 결과를 생성합니다. 이 연구는 Lumiere와 Phenaki라는 두 가지 비디오 생성 모델을 활용하여 실험을 진행하였으며, 각 모델의 장단점을 분석하였습니다.



### DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Contro (https://arxiv.org/abs/2410.13830)
Comments:
          Project page: this https URL

- **What's New**: DreamVideo-2는 복잡한 테스트 타임 파인튜닝 없이 특정 주제와 모션 궤적을 갖는 비디오를 생성할 수 있는 제로샷(Zero-shot) 비디오 사용자 지정 프레임워크입니다. 사용자는 단일 이미지와 경계 상자(바운딩 박스) 시퀀스를 입력으로 제공하여 비디오를 만들 수 있습니다.

- **Technical Details**: 본 연구에서는 주제 학습을 위한 모델의 고유한 능력을 활용하는 레퍼런스 어텐션(reference attention)과, 경계 상자에서 파생된 박스 마스크의 강력한 모션 신호를 완전히 활용하여 정밀한 모션 제어를 달성하기 위한 마스크 가이드 모션 모듈(mask-guided motion module)을 도입합니다. 또한, 마스크된 레퍼런스 어텐션과 재가중화 확산 손실(reweighted diffusion loss)을 통해 주제 학습과 모션 제어의 균형을 맞추는 두 가지 디자인을 제안합니다.

- **Performance Highlights**: DreamVideo-2는 새로운 데이터셋에서 수행된 포괄적인 실험 결과에 따르면 기존의 최첨단 방법들보다 주제 사용자 지정 및 모션 제어 모두에서 뛰어난 성능을 보입니다.



### Harnessing Webpage UIs for Text-Rich Visual Understanding (https://arxiv.org/abs/2410.13824)
- **What's New**: 이번 연구에서는 웹 페이지 UI에서 일반 다중 모달 지침을 합성하여 MLLM(다중 모달 대형 언어 모델)의 텍스트가 풍부한 시각적 이해(text-rich visual understanding) 능력을 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 730만 개 샘플로 구성된 MultiUI 데이터셋을 활용하며, 이는 100만 개 웹사이트에서 수집되었습니다. 텍스트 기반 대형 언어 모델(LLM)은 웹페이지 접근성 트리에서 구조적 텍스트 표현을 처리하여 다중 모달 모델을 교육하는 데 필요한 지침을 생성합니다.

- **Performance Highlights**: MultiUI로 훈련된 모델은 웹 UI 작업에서 VisualWebBench에서 최대 48%의 개선을 보였으며, Mind2Web 데이터셋에서 액션 정확도가 19.1% 향상되었습니다. 더 나아가 이 모델은 비웹 UI 작업과 문서 이해, OCR, 차트 해석과 같은 비 UI 도메인에서도 놀라운 일반화를 보여주었습니다.



### Deep Generative Models Unveil Patterns in Medical Images Through Vision-Language Conditioning (https://arxiv.org/abs/2410.13823)
Comments:
          Accepted by AIM-FM Workshop of NeurIPS2024

- **What's New**: 이 연구는 깊은 생성 모델(Deep Generative Models)이 의학 이미지 분석에 어떻게 패턴을 드러내고 표현할 수 있는지를 강조합니다. 특히, 기존의 데이터 증대(data augmentation)를 넘어서는 접근 방식을 제시하며, 임상 데이터와 세분화 마스크(segmentation masks)를 결합하여 이미지 합성(image synthesis) 과정을 안내합니다.

- **Technical Details**: 연구에서는 임상 정보를 텍스트로 변환하는 혁신적인 접근 방식을 사용하여 누락된 값을 처리하고, 대규모의 사전 훈련된 비전-언어 모델(vision-language models)을 활용하여 독립적인 임상 항목 간의 관계를 탐구합니다. 텍스트-비주얼 임베딩(text-visual embedding) 메커니즘을 도입하여 네트워크가 제공된 정보를 효과적으로 활용할 수 있도록 조건을 강화합니다. 이 방법은 GAN 기반과 확산 모델(diffusion models) 모두에 일반화할 수 있습니다.

- **Performance Highlights**: 가슴 CT 데이터셋을 활용한 실험 결과에서, 흡연 상태와 관련된 폐의 일관된 강도 변화가 임상 관찰과 일치하는 것을 보여주었으며, 이는 특정 속성이 의료 이미지 패턴에 미치는 영향을 포착하고 시각화하는 데 있어 방법의 효과를 입증합니다. 이 연구는 깊은 생성 모델을 활용하여 복잡한 임상 상태를 조기 발견하고 정밀 시각화할 수 있는 새로운 길을 제공합니다.



### Multi-style conversion for semantic segmentation of lesions in fundus images by adversarial attacks (https://arxiv.org/abs/2410.13822)
Comments:
          preprint

- **What's New**: 이번 논문에서는 다양한 데이터베이스의 주석 스타일 간 표준화를 해결하기 위해 'adversarial style conversion'이라는 새로운 방법을 도입합니다. 이 방법은 단일 아키텍처에서 결합된 데이터베이스를 활용하여 모델이 입력에 따라 자발적으로 세분화 스타일을 조정하도록 훈련되었습니다.

- **Technical Details**: 제안된 방법론은 인코더 특징을 기반으로 데이터셋의 출처를 탐지하는 'linear probe'를 추가하고, 적대적 공격(adversarial attacks)을 통해 모델의 세분화 스타일을 조정하는 방식을 채택합니다. 이는 여러 데이터셋에서 훈련된 세분화 모델의 스타일 변환을 가능하게 합니다.

- **Performance Highlights**: 논문의 결과는 데이터셋 조합을 통해 질적으로나 양적으로 유의미한 개선을 보이며, 모델의 일반화 성능, 불확실성 추정 및 주석 스타일 간의 지속적 보간과 같은 기회를 제공합니다.



### ConsisSR: Delving Deep into Consistency in Diffusion-based Image Super-Resolution (https://arxiv.org/abs/2410.13807)
- **What's New**: 본 논문에서는 Real-world image super-resolution (Real-ISR) 문제를 해결하기 위해 ConsisSR이라는 새로운 방법을 제안합니다. 이 방법은 텍스트-이미지 (T2I) diffusion 모델을 활용하여 의미적 일관성과 픽셀 수준의 일관성을 모두 처리할 수 있도록 설계되었습니다.

- **Technical Details**: ConsisSR의 핵심 기술로는 먼저, Hybrid Prompt Adapter (HPA)를 통해 CLIP 이미지 임베딩과 텍스트 임베딩을 효과적으로 결합하여 의미적 일관성을 확보합니다. 또한 Time-aware Latent Augmentation (TALA)을 도입하여 T2I 생성과 Real-ISR의 일관성 요건 간의 간극을 줄입니다. GAN-Embedding 방식은 Real-ESRGAN의 사전 학습된 데이터를 활용하여 초기 diffusion 단계를 건너뛰고 추론 속도를 획기적으로 향상시킵니다.

- **Performance Highlights**: 제안하는 ConsisSR은 전체적인 SDSR 방법들 중에서 SOTA (state-of-the-art) 성능을 달성하며, 기존 모델에 비해 추론 프로세스를 최소 10단계로 줄여도 샘플링 품질을 유지합니다.



### MotionBank: A Large-scale Video Motion Benchmark with Disentangled Rule-based Annotations (https://arxiv.org/abs/2410.13790)
- **What's New**: 이 논문에서는 대규모 모션 모델(Large Motion Model, LMM)을 구축하고 벤치마크하는 방법에 대해 다루고 있습니다. 새로운 MotionBank 데이터셋은 13개의 비디오 액션 데이터셋을 통합하여 1.24M개의 모션 시퀀스와 132.9M개의 프레임을 포함하고 있어 자연적이고 다양한 인간 모션을 제공합니다.

- **Technical Details**: MotionBank는 다양한 사람들의 일상 활동에서 수집된 대규모 인간 중심의 비디오 액션 데이터셋으로, 4D 모션 데이터의 부족함을 해결합니다. 이 데이터는 SMPL(Skinned Multi-Person Linear Model) 매개변수를 활용하여 생성되며, 이동 캡션 생성 알고리즘을 통해 자동으로 비편향적이고 규칙 기반의 텍스트 설명을 생성합니다.

- **Performance Highlights**: 실험 결과, MotionBank는 인간 모션 생성, 상황에 맞는 모션 생성 및 모션 이해와 같은 일반적인 모션 관련 작업에 유익하다는 것을 보여주었습니다. 또한 이 데이터셋은 LMM을 위한 효율적인 대안으로서 기능할 수 있습니다.



### Emphasizing Semantic Consistency of Salient Posture for Speech-Driven Gesture Generation (https://arxiv.org/abs/2410.13786)
- **What's New**: 이번 연구에서는 기존의 음성 기반 제스처 생성 방법에서 나타나는 문제점을 해결하기 위해 강조된 의미적 일관성을 바탕으로 새로운 방법을 제안합니다. 특히, 두 가지 모달리티인 오디오와 신체 자세의 개별 표현을 학습하는 조인트 매니폴드 공간을 도입하여 의미적 연관성을 활용하고, 특히 중요한 자세 구분에 중점을 두었습니다.

- **Technical Details**: 새로운 방법에서는 의미적 일관성을 강하게 유지하기 위해 일관성 손실(consistency loss)을 활용합니다. 또한, 약한 감독 학습을 활용한 중요한 자세 감지기를 도입하여 중요한 자세를 식별하고, 이와 관련된 일관성 손실을 보강하여 높은 의미적 내용을 가진 자세와 오디오를 효과적으로 정렬합니다. 음성의 표정과 신체 제스처를 별도로 합성하는 두 가지 가지(branch)를 설계하여 동기화 및 자연스러움을 월등히 개선합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 방법은 기존의 최첨단 접근법들에 비해 우수한 성능을 입증하였습니다. 특히, 의미적 일관성을 강조함으로써 만들어진 제스처의 자연스러움과 전달력이 크게 향상되었습니다.



### Improving Multi-modal Large Language Model through Boosting Vision Capabilities (https://arxiv.org/abs/2410.13733)
- **What's New**: 최근 시각-언어 모델의 시각 이해 능력을 향상시키기 위해 Arcana라는 새로운 멀티모달 언어 모델을 제안합니다. 이 모델은 두 가지 중요한 기술인 MM-LoRA와 QLadder 어댑터를 도입합니다.

- **Technical Details**: Arcana는 MM-LoRA를 통해 시각과 언어를 각각 위한 두 개의 병렬 LoRA로 구성된 디코더를 구현합니다. 또한, QLadder 어댑터를 사용하여 고정된 사전 훈련된 시각 인코더로부터의 중간 표현을 집계하는 '계단' 구조를 포함합니다. 이 구조는 시각 정보를 더 잘 학습하고 통합할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, Arcana는 DINOv2 기반의 첨단 방법들과 비슷한 성과를 내며, 기존의 멀티모달 벤치마크에서 성능 향상을 보여줍니다. 기능적 측면에서 MM-LoRA와 QLadder의 효율성을 입증했습니다.



### DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation (https://arxiv.org/abs/2410.13726)
- **What's New**: 이번 연구에서는 DAWN(Dynamic frame Avatar With Non-autoregressive diffusion)이라는 새로운 프레임워크를 통해 오디오 클립과 초상화를 이용한 직접적인 동영상 생성 방식을 선보입니다. DAWN은 기존의 autoregressive (AR) 방식의 한계를 극복하여, 모든 프레임을 동시 생성할 수 있는 비선형(non-autoregressive, NAR) 전략을 적용하였습니다.

- **Technical Details**: DAWN 프레임워크는 (1) 오디오에 기반한 전체적인 얼굴 역학을 생성하는 잠재적 동작 공간에서의 생성과 (2) 오디오 기반의 머리 자세 및 깜빡임 생성이라는 두 가지 주요 구성 요소로 이루어져 있습니다. 추가적으로, Pose and Blink generation Network (PBNet)는 오디오에서 자연스러운 머리 자세와 깜빡임 시퀀스를 생성하는 데 사용됩니다. DAWN은 A2V-FDM(Audio-to-Video Flow Diffusion Model)을 통해 입술과 오디오 간의 암묵적 관계를 학습합니다.

- **Performance Highlights**: DAWN은 빠른 생성 속도와 더불어 정확한 입술 동작 및 자연스러운 자세/깜빡임을 보장하여, 실제감 있고 생동감 넘치는 비디오를 생성합니다. 또한, DAWN은 뛰어난 외삽(extrapolation) 능력을 발휘하며, 긴 비디오에서도 높은 품질을 안정적으로 유지할 수 있는 가능성을 보여줍니다.



### Movie Gen: A Cast of Media Foundation Models (https://arxiv.org/abs/2410.13720)
- **What's New**: 이번 논문에서는 Movie Gen이라는 새로운 foundation 모델 세트를 제안합니다. 이 모델은 다양한 화면 비율과 동기화된 오디오와 함께 고품질 1080p HD 비디오를 생성하며, 사용자의 이미지를 기반으로 한 개인화된 비디오 생성 및 정밀한 지침 기반 비디오 편집 기능도 포함되어 있습니다.

- **Technical Details**: Movie Gen은 30B 파라미터의 트랜스포머 모델로, 최대 73K 비디오 토큰의 컨텍스트 길이를 가지고 있습니다. 이 모델은 텍스트-비디오 합성, 비디오 개인화, 비디오 편집, 비디오-오디오 생성 및 텍스트-오디오 생성과 같은 다양한 작업에서 최첨단 성능을 기록합니다. 인터넷 스케일의 이미지, 비디오, 오디오 데이터를 통해 사전 학습되었습니다.

- **Performance Highlights**: Movie Gen 모델은 기존 상업 시스템을 초월하여 텍스트-비디오 생성, 비디오 개인화, 정밀 비디오 편집 및 오디오 생성 작업에서 탁월한 성능을 보여줍니다. 특히, Movie Gen Video는 최대 16초의 개인화된 HD 비디오 생성을 가능하게 하며, Movie Gen Audio는 정밀한 음악 생성과 음향 효과 생성을 지원합니다.



### Exploring the Design Space of Visual Context Representation in Video MLLMs (https://arxiv.org/abs/2410.13694)
Comments:
          Long Video MLLM; work in progress

- **What's New**: 비디오 다중 모달 대형 언어 모델(Video Multimodal Large Language Models, MLLMs)의 시각적 컨텍스트 표현에 대한 체계적인 연구를 다룬 첫 번째 논문입니다. 연구진은 최적의 시각적 컨텍스트 표현 방식인 Opt-Visor 모델을 제안하며, 최대 162 프레임까지의 비디오를 처리할 수 있습니다.

- **Technical Details**: 비디오 MLLMs의 성능 향상을 위해 프레임 선택(frame selection)과 임베딩 선택(embedding selection)을 최적화하는 제약 최적화 문제로 작업을 정의했습니다. 각 프레임에서의 토큰 수와 프레임 수에 따른 언어 모델링 손실(training loss)을 함수로 모델링하여 시각적 컨텍스트의 경쟁 관계를 이해합니다. 이러한 분석을 바탕으로 성능 추세를 설명하는 함수 곡선을 맞추어 다양한 선택 전략의 효과를 평가했습니다.

- **Performance Highlights**: 실험 결과, 시각적 임베딩 수(토큰 또는 프레임)를 증가시키는 것이 전반적으로 성능 향상에 기여한다는 것을 확인했습니다. 특히, 압축 기반 방법이 더 적은 시각적 임베딩으로도 더 많은 의미 정보를 보존할 수 있다는 점이 강조되었습니다. 연구진은 이러한 성과를 통해 프레임 선택과 임베딩 선택 간의 이상적인 비율을 찾는 방법을 제안하고, 경험적 실험과 일치하는 제안된 최적 설정을 검증하였습니다.



### Label-free prediction of fluorescence markers in bovine satellite cells using deep learning (https://arxiv.org/abs/2410.13685)
Comments:
          11 pages, 4 figures

- **What's New**: 본 연구에서는 소의 위성 세포(Bovine Satellite Cells, BSCs)의 비침습적이고 비표지(label-free) 방법을 통해 품질을 평가하는 새로운 접근법을 제시합니다. 이는 전통적인 염색 및 세포 관찰 방법의 한계를 극복하기 위한 딥러닝 기반의 기술을 활용하였습니다.

- **Technical Details**: U-Net 기반의 CNN 모델을 사용하여 세포 배양의 하나의 밝은 필드 밝기(optical) 이미지를 통해 여러 개의 형광 신호를 예측했습니다. DAPI와 Pax7 두 가지 주요 생체 표지를 사용하여 BSCs의 풍부함과 품질을 평가하였으며, 이미지 전처리 과정에서 형광 잡음을 제거하여 예측 성능을 개선했습니다. 48개의 생물학적 복제본을 사용하고 Pearson 상관 계수 및 SSIM과 같은 통계적 성능 지표로 모델을 평가하였습니다.

- **Performance Highlights**: 모델은 DAPI 예측에서 더 우수한 성능을 보였으며, 이는 균일한 염색 덕분입니다. Pax7 예측은 생물학적 이질성을 반영하여 더 변동성이 컸습니다. 또한 향상된 시각화 기술을 통해 예측의 해석 가능성을 높여 연구자들이 모델 예측을 쉽게 이해하도록 지원했습니다. 최종 결과는 BSC 품질 평가의 비침습적이고 실용적인 AI 기반 평가를 가능하게 하여 기른 고기 산업의 발전에 기여할 것입니다.



### Pose-Based Sign Language Appearance Transfer (https://arxiv.org/abs/2410.13675)
- **What's New**: 이 연구에서는 수화에서 서명자의 외모를 제어하는 방법을 소개하며, 서면 내용은 보존하는 방법을 제시합니다. 이 방법은 서명자의 외모를 다른 사람으로 전이하여 자연스러운 움직임과 전환을 유지합니다.

- **Technical Details**: 서명자의 외모를 변경하고 서명 내용을 유지하기 위해 포즈 시퀀스를 조작하는 방법을 사용합니다. 신호 긴밀성과 자연스러운 운동을 위해 몸체와 얼굴의 특성은 수정하지만 손의 형상은 유지합니다. 이는 평균화된 포즈를 통해 수행됩니다.

- **Performance Highlights**: 이 방법은 서명자의 신원을 식별하는 정확성을 줄이면서도 수화 인식 성능을 약간 저하시킵니다. 분석 결과, 원래 포즈를 이용한 모델이 가장 뛰어난 성능을 보였으며, 전이된 포즈를 사용했을 때 신원 식별 정확도가 52.20%로 감소했습니다. 이는 프라이버시와 유용성 간의 균형을 잘 보여줍니다.



### Diffusion Curriculum: Synthetic-to-Real Generative Curriculum Learning via Image-Guided Diffusion (https://arxiv.org/abs/2410.13674)
- **What's New**: 본 논문에서는 기존의 데이터 증강(data augmentation) 기법의 한계를 극복하기 위해, 이미지 가이드를 활용하여 합성 이미지와 실제 이미지 간의 스펙트럼 보간을 수행하는 새로운 접근 방법인 'Diffusion Curriculum (DisCL)'을 제안합니다.

- **Technical Details**: 기존의 텍스트 가이드는 합성 이미지의 품질이 원본 이미지와 어떤 연관이 있는지를 제어할 수 없지만, 이미지 가이드를 통해 합성 이미지와 실제 이미지의 유사성을 조절할 수 있습니다. DisCL은 훈련 단계에 따라 이미지 합성의 가이드 수준을 조정하여 모델을 위한 어려운 샘플을 식별하고 이들을 학습하는 데 가장 효과적인 가이드 수준을 평가합니다.

- **Performance Highlights**: DisCL을 iWildCam 데이터셋에 적용했을 때 OOD(Out-of-Distribution) 및 ID(In-Distribution) 매크로 정확도에서 각각 2.7% 및 2.1% 향상을 보여주었으며, ImageNet-LT에서 기본 모델의 tail-class 정확도를 4.4%에서 23.64%로 개선하고 모든 클래스 정확도에서 4.02% 향상을 달성했습니다.



### VL-GLUE: A Suite of Fundamental yet Challenging Visuo-Linguistic Reasoning Tasks (https://arxiv.org/abs/2410.13666)
Comments:
          18 pages, 7 figures

- **What's New**: 본 연구에서는 비주얼-언어적 (Visuo-Linguistic) 이해를 위한 새로운 멀티태스크 벤치마크인 VL-GLUE를 제안합니다. VL-GLUE는 7개의 다양한 태스크로 구성되어 있으며, 10만 개 이상의 샘플을 포함해 비주얼과 텍스트 간의 결합 추론 능력을 평가합니다.

- **Technical Details**: VL-GLUE는 이미지와 텍스트 정보를 결합하여 추론을 필요로 하는 7개의 태스크로 구성되어 있습니다. 이 benchmark는 다양한 이미지 유형(일상 장면, 도표 등)과 특정 도메인 텍스트(요리, 정치 등)를 포함해, 실제 세계에서의 멀티모달 이해의 필요성을 보여줍니다.

- **Performance Highlights**: 기존의 대규모 비전-언어 모델들이 VL-GLUE 벤치마크에서 낮은 점수를 얻어, 이 분야의 모델들이 시기적절한 비주얼-언어적 추론 능력을 갖춤이 긴급하게 요구되고 있다는 점이 부각되었습니다.



### DiRecNetV2: A Transformer-Enhanced Network for Aerial Disaster Recognition (https://arxiv.org/abs/2410.13663)
Comments:
          23 pages

- **What's New**: 이번 연구에서는 UAV(무인 항공기)와 AI 모델을 통합한 재난 인식 시스템을 위한 새로운 하이브리드 모델인 DiRecNetV2를 소개합니다. 이 모델은 전통적인 CNN(합성곱 신경망)의 특징 추출 능력을 바탕으로 Vision Transformers(ViT)의 글로벌 컨텍스트 해석 능력을 결합하여 경량화된 재난 관리 솔루션을 제공합니다.

- **Technical Details**: DiRecNetV2는 CNN과 ViT의 결합으로 이루어져 있으며, CNN의 강력한 특징 추출 능력과 ViT의 장거리 의존성 캡처의 조합으로 설계되었습니다. 이 모델은 Nvidia Orin Jetson 장비에서 176.13 FPS의 속도로 실행될 수 있도록 최적화되어 있습니다. AIDERSv2 데이터셋을 활용하여, 단일 레이블과 다중 레이블 인식 성능을 평가하였습니다.

- **Performance Highlights**: DiRecNetV2는 단일 레이블 테스트 세트에서 0.964의 가중 F1 점수를 달성하였으며, 복잡한 다중 레이블 테스트 세트에서 0.614의 점수를 기록하였습니다. 이는 이 모델이 단일 재난과 다중 재난 인식 및 분석에서 효과적임을 보여줍니다.



### ActionCOMET: A Zero-shot Approach to Learn Image-specific Commonsense Concepts about Actions (https://arxiv.org/abs/2410.13662)
Comments:
          15 pages, 3 figures. arXiv admin note: text overlap with arXiv:2004.10796 by other authors

- **What's New**: 이 논문은 인간이 수행하는 행동에 대한 다양한 추론을 자동 시스템에 적용하기 위한 새로운 멀티모달 작업을 제안하고 있습니다. 이를 위해 8.5k의 이미지와 59.3k의 행동 추론을 포함하는 새로운 데이터셋을 개발했습니다.

- **Technical Details**: 이 연구에서는 ActionCOMET이라는 제로샷 프레임워크를 도입하여 제공된 시각적 입력에 따라 언어 모델에서 지식을 감별합니다. 이 시스템은 요리 비디오 데이터셋을 기반으로 수집된 데이터를 사용하여 인과 관계와 같은 복잡한 행동 개념을 학습합니다.

- **Performance Highlights**: ActionCOMET의 초기 결과는 수집된 데이터셋에서의 성능을 나타내며, 기존의 최첨단 VQA(Visual Question Answering) 접근 방식과 비교하여 의미 있는 결과를 보여주고 있습니다.



### Help Me Identify: Is an LLM+VQA System All We Need to Identify Visual Concepts? (https://arxiv.org/abs/2410.13651)
Comments:
          14 pages, 7 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLM)과 비주얼 질문 답변(Visual Question Answering, VQA) 시스템을 활용하여 제로샷(Zero-shot) 기반의 세밀한 비주얼 개념 학습 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크에서는 GPT-3를 통해 데이터셋 내 비주얼 객체의 풍부한 언어적 설명을 얻고, 이를 이진 질문의 집합으로 변환하여 VQA 시스템에 제공합니다. 질문과 쿼리 이미지를 함께 제시하고, 답변을 집계하여 테스트 이미지에서 객체의 존재 여부를 확인합니다.

- **Performance Highlights**: 실험 결과, 기존의 제로샷 비주얼 분류 방법 및 소수 샷(few-shot) 개념 학습 방법과 비교할 때 상대적으로 유사한 성능을 보였습니다. 특히, 본 연구는 상당한 계산 오버헤드 없이도 설명 가능성을 유지하면서 높은 성능을 달성합니다.



### Comparison of Image Preprocessing Techniques for Vehicle License Plate Recognition Using OCR: Performance and Accuracy Evaluation (https://arxiv.org/abs/2410.13622)
Comments:
          12 pages, 13 figures

- **What's New**: 이 연구에서는 Optical Character Recognition (OCR) 기술을 개선하기 위한 다양한 이미지 전처리 기법들을 탐구하고 평가합니다. 특히 차량 번호판 인식에 초점을 맞춰, 그레이스케일 변환, CLAHE (Contrast Limited Adaptive Histogram Equalization), 양방향 필터(Bilateral Filter) 등 여러 기법을 적용하며, 각각의 기술이 어떻게 정확도, 정밀도, 재현율 및 F1 점수에 영향을 미치는지를 분석합니다.

- **Technical Details**: 연구는 BRASIL 차량 번호판 데이터셋을 사용하여 실험을 진행합니다. 각각의 전처리 기법은 단독 및 조합적으로 평가되며, 평가 지표로는 정확도(accuracy), 정밀도(precision), 재현율(recall), F1-score, ROC 곡선(ROC curve), AUC, ANOVA 등을 사용하여 최적의 방법을 도출합니다. 이러한 통계적 분석은 OCR 성능을 현실 세계 시나리오에서 최적화하는 데 중요한 인사이트를 제공합니다.

- **Performance Highlights**: 연구 결과, CLAHE와 그레이스케일 변환 조합이 차량 번호판 인식에서 가장 높은 정확도를 보였으며, 이를 바탕으로 제안된 전처리 기법들 중 최적의 실용적 접근법을 제공합니다. 이 연구는 OCR의 성능을 향상시키기 위해 연구자들에게 유용한 가이드를 제공하며, 교통 모니터링 및 차량 보안과 같은 분야에 significant한 영향을 미칠 것으로 기대됩니다.



### Enhanced Prompt-leveraged Weakly Supervised Cancer Segmentation based on Segment Anything (https://arxiv.org/abs/2410.13621)
Comments:
          10 pages, 7 figures

- **What's New**: 이번 연구에서는 제한된 레이블 데이터로 인한 문제를 해결하기 위한 약한 감독 기반의 세그멘테이션 모델인 Weakly Supervised Semantic Segmentation (WSSS)을 제안합니다. Class Activation Map (CAM)과 Segment Anything Model (SAM) 기반의 가짜 레이블링을 결합하여 이를 실시하였습니다.

- **Technical Details**: SAM을 기반으로 하는 가짜 레이블 생성에서, Attention Dropout Layer(ADL)를 개선하여 시각적 프롬프트를 명시적으로 통합하였습니다. 이를 통해 CAM 기반 접근법에서의 부분 및 잘못된 활성화 문제를 완화하고, GPU 메모리 사용량을 12GB로 제한하면서도 성능을 유지할 수 있었습니다.

- **Performance Highlights**: 제안된 방법은 유방암의 조직병리학적 데이터셋에 대한 실험에서 기존 WSSS 방법들을 초월하였으며, 제안된 접근법이 메모리 효율적임을 보여주었습니다. 코드 또한 공개되어 있어, 연구자들이 적용할 수 있습니다.



### LoLDU: Low-Rank Adaptation via Lower-Diag-Upper Decomposition for Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2410.13618)
Comments:
          13 pages, 7 figures

- **What's New**: 모델 스케일의 급격한 증가로 인해 파라미터 효율적인 미세 조정 기법인 LoLDU가 제안되었습니다. 이 방법은 기존의 Low-Rank Adaptation (LoRA)와 같은 접근과 차별화되면서 훈련 가능한 파라미터 수를 2600배 줄이며 비슷한 성능을 유지합니다.

- **Technical Details**: LoLDU는 Lower-Diag-Upper Decomposition (LDU)을 활용하여 희소 행렬의 초기화 및 최적화를 통해 수렴 속도를 향상시키고, 대각 행렬을 최적화하여 스케일 변환을 강화합니다. 지금까지 제안된 PEFT 기법 중 가장 적은 수의 파라미터(0.00025%)만을 조정하는 방식으로 작업합니다.

- **Performance Highlights**: LoLDU는 다양한 모델 아키텍처(LLaMA2, RoBERTa, ViT, Stable Diffusion)에서 instruction-following, natural language understanding (NLU), 이미지 분류 및 생성 작업을 포함한 4개의 지침 데이터 세트, 6개의 NLU 데이터 세트, 8개의 이미지 분류 데이터 세트에서 포괄적인 실험을 통해 효과성과 다재다능함을 입증했습니다.



### Spatiotemporal Object Detection for Improved Aerial Vehicle Detection in Traffic Monitoring (https://arxiv.org/abs/2410.13616)
Comments:
          13 pages

- **What's New**: 이번 연구는 UAV(무인 항공기) 카메라를 이용한 다중 클래스 차량 탐지의 발전을 다루며, Spatiotemporal Object Detection 모델을 개발하였습니다. 이를 위해 6,600개의 주석이 달린 연속 프레임 이미지로 구성된 Spatio-Temporal Vehicle Detection Dataset(STVD)를 소개하며, 이를 통해 알고리즘의 포괄적인 훈련과 평가를 가능하게 합니다.

- **Technical Details**: YOLO 기반 객체 탐지 알고리즘을 개선하여 시간적 동역학을 통합하였으며, 스페이셔 (spatial)와 템포럴 (temporal) 정보 모두를 활용하는 모델을 개발하였습니다. 기존의 단일 프레임 모델보다 뛰어난 성능을 발휘하며, 특히 주목(attention) 메커니즘을 통합하여 성능을 더 향상시킬 수 있음을 입증하였습니다.

- **Performance Highlights**: 실험적으로, 가장 우수한 시공간 모델이 단일 프레임 모델에 비해 16.22% 향상된 성능을 보였으며, 주목 메커니즘을 통합한 모델은 추가적인 성능 향상 가능성을 보여주었습니다.



### Material Fingerprinting: Identifying and Predicting Perceptual Attributes of Material Appearanc (https://arxiv.org/abs/2410.13615)
Comments:
          14 pages, 12 figures, 3 tables

- **What's New**: 본 논문은 동적인 시각적 자극에서 얻어진 인식적(stylish) 특징을 인코딩하여 새로운 재료 식별 방법을 제시합니다. 347 가지 재료의 비디오를 통해 수집된 16개의 중요한 인식 속성과 이를 활용한 '물질 지문'(material fingerprint) 생성 과정이 포함되어 있습니다.

- **Technical Details**: 심리 물리적 실험을 통해 20명 이상의 참가자로부터 각 재료에 대한 속성 평가를 수집하고, 이를 바탕으로 통계적 및 딥 러닝 이미지 특징과 인식 속성 간의 관계를 예측하기 위해 다층 퍼셉트론(multi-layer perceptron) 모델을 훈련시켰습니다. 이러한 과정은 재료 속성 간의 시각적 유사성 및 차별성을 intuitively 판단할 수 있는 파라미터로 작용합니다.

- **Performance Highlights**: 제안된 모델은 단 두 개의 이미지로부터도 효과적으로 시각적 지문을 추론할 수 있으며, 이는 디지털 애플리케이션에서 더 효율적이고 실용적인 재료 분석 가능성을 보여줍니다. 이 연구는 공공 데이터셋을 구축하고, 16개의 인식 속성을 통해 다양한 재료에 대한 이해를 증진시키는 중요한 기여를 합니다.



### MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes (https://arxiv.org/abs/2410.13613)
- **What's New**: 새로운 4D Gaussian Splatting (4DGS) 기술이 복잡한 동적 3D 장면을 고품질로 캡처할 수 있는 가능성을 보여주고 있습니다. 특히, 이 논문은 전통적인 4DGS의 메모리 문제를 해결하기 위한 메모리 효율적인 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 색상을 3개의 매개변수를 가지는 개별 Gaussian의 직접 색상 요소와 경량의 교대 전류 색상 예측기로 분해합니다. 또한, 엔트로피 제약 Gaussian 변형 기법을 도입하여 각 Gaussian의 작용 범위를 확장하고, 투명도 기반의 엔트로피 손실을 통합하여 필요한 Gaussian의 수를 최소화합니다.

- **Performance Highlights**: 6비트 부동 소수점(FP16) 저장 및 zip 압축을 통해 Technicolor 및 Neural 3D 비디오 데이터셋에서 각각 190× 및 125×의 저장 공간 감소를 달성하면서도, 렌더링 속도와 장면 표현 품질을 유지합니다.



### H2OVL-Mississippi Vision Language Models Technical Repor (https://arxiv.org/abs/2410.13611)
- **What's New**: H2OVL-Mississippi 모델은 3700만 개의 이미지-텍스트 쌍을 기반으로, 8개의 H100 GPU를 사용하여 240시간 동안 훈련된 작은 비전-언어 모델(VLM) 쌍을 소개합니다. 특히, H2OVL-Mississippi-0.8B는 8억 개의 매개변수로 구성되어 텍스트 인식에 특화되어 있으며, OCRBench의 텍스트 인식 부문에서 최첨단 성능을 발휘하고 있습니다.

- **Technical Details**: H2OVL-Mississippi 모델은 Vision Transformer(비전 트랜스포머) 구성 요소와 대형 언어 모델(LLM)로 이루어집니다. H2OVL-Mississippi-0.8B는 OCR 및 문서 중심 작업에 최적화되어 있고, H2OVL-Mississippi-2B는 다양한 멀티모달 작업을 수행할 수 있는 일반 목적 모델입니다. 이들은 각각 256에서 1590개의 시각적 토큰을 생성하며, 동적 해상도 전략(dynamic resolution)과 다중 스케일 적응 크롭(multi-scale adaptive cropping) 전략을 활용하여 다양한 이미지 크기와 종횡비에 적응합니다.

- **Performance Highlights**: H2OVL-Mississippi-0.8B는 OCRBench에서 텍스트 인식 부문에서 최첨단 성능을 보여주며, H2OVL-Mississippi-2B는 다양한 학술 벤치마크에서 경쟁력 있는 메트릭스를 제공합니다. 두 모델 모두 H2O-Danube 언어 모델의 기능을 확장하여 비주얼 도메인으로의 적용 가능성을 높이고, Apache 2.0 라이선스 하에 공개되어 문서 AI와 비주얼 LLM의 접근성을 높였습니다.



### DN-4DGS: Denoised Deformable Network with Temporal-Spatial Aggregation for Dynamic Scene Rendering (https://arxiv.org/abs/2410.13607)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 실시간 수준의 동적 장면 렌더링을 위한 새로운 접근법인 Denoised Deformable Network with Temporal-Spatial Aggregation (DN-4DGS)를 제안합니다. 기존의 NeRF(Neural Radiance Fields) 방식의 한계를 극복하고, 3D Gaussian Splatting(3DGS)을 활용하여 노이즈를 줄이고 성능을 향상시키는 방법에 초점을 맞추고 있습니다.

- **Technical Details**: DN-4DGS는 Noise Suppression Strategy(NSS)와 Decoupled Temporal-Spatial Aggregation Module(DTS)라는 두 가지 주요 구성 요소로 이루어져 있습니다. NSS는 캘러니컬(Canonical) 3D Gaussian의 좌표 분포를 변경하고 노이즈를 억제하여 더 정확한 변형 필드를 생성합니다. DTS는 인접한 포인트와 프레임의 정보를 집계하여 공간-시간 정보를 비뚤어지지 않도록 처리합니다. 또한, 주요 좌표 변형 과정을 통해 노이즈를 감소시키고 최적의 결과를 도출하게 됩니다.

- **Performance Highlights**: 제안된 방법인 DN-4DGS는 다양한 현실 세계 데이터셋에서 최첨단의 렌더링 품질을 실시간 수준에서 달성합니다. 실험을 통해 기존의 동적 씬 렌더링 기법들과 비교하여 빠른 속도와 높은 정확도를 보여주며, 실무에 적용 가능성을 엿볼 수 있습니다.



### Let Me Finish My Sentence: Video Temporal Grounding with Holistic Text Understanding (https://arxiv.org/abs/2410.13598)
Comments:
          Accepted by ACMMM 24

- **What's New**: 이번 논문에서는 Video Temporal Grounding (VTG) 분야에서 query 문장의 전체적인 의미를 고려하지 못했던 기존의 접근 방식을 보완하기 위해 세 가지 주요 기여를 제안합니다. 첫째, 전체적인 텍스트 정보를 통합한 시각적 프레임 레벨 게이트 메커니즘을 소개하고, 둘째, 쿼리와 관련 있는 프레임 간의 정밀한 상관관계를 학습하는 cross-modal alignment loss를 제안합니다.

- **Technical Details**: 제안된 방법론은 두 가지 gating 메커니즘을 활용하는 gated cross-attention을 기반으로 하며, 각각 지역적(local) 및 비지역적(non-local) 게이트를 도입함으로써 텍스트 앵커와 개별 영상 프레임 간의 유사성을 평가합니다. 그리고 두 가지 정밀 정렬 손실(clip-level consistency loss와 frame-level relevance loss)을 통해 영상 내용과 쿼리 텍스트 간의 정렬을 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 QVHighlights, Charades-STA 및 TACoS와 같은 VTG 벤치마크에서 기존 최첨단 방법들보다 우수한 성능을 보여줍니다. 이는 전체적인 텍스트 이해가 모델이 영상 내에서 의미적으로 중요한 부분에 집중하도록 유도함을 나타냅니다.



### Pseudo Dataset Generation for Out-of-Domain Multi-Camera View Recommendation (https://arxiv.org/abs/2410.13585)
Comments:
          Accepted to VCIP 2024. Project page: this https URL

- **What's New**: 이 논문은 정규 비디오에서 유사 레이블(pseudo-labeled) 다중 카메라 편집 데이터셋을 생성하는 방법론을 제안합니다. 이는 레이블이 있는 다중 카메라 보기 추천 데이터가 부족한 문제를 해결하는 데 도움을 주며, 특히 기존 데이터셋이 특정 장면이나 스타일에 국한되었음을 지적합니다.

- **Technical Details**: 논문에서 제안하는 방법은 정규 비디오에서 샷을 감지하고, 이를 클러스터링하여 가상의 카메라 레이블(pseudo-camera labels)을 생성하는 것입니다. 이후 각 가상의 카메라에서 가장 시각적으로 유사한 샷을 후보로 선택하여 유사 레이블 데이터를 생성합니다. 이 과정은 Temporal Segment Network (TSN)를 통해 샷 분류 기능을 추출하고 K-Means 알고리즘으로 클러스터링을 수행하여 이루어집니다.

- **Performance Highlights**: 제안된 방식으로 훈련된 모델은 목표 영역(target domain)에서의 분류 정확도가 68%의 상대적 향상을 이뤘습니다. 이는 훈련된 모델의 정확도가 22.65에서 38.14로 개선되는 결과를 보여줍니다.



### Co-Segmentation without any Pixel-level Supervision with Application to Large-Scale Sketch Classification (https://arxiv.org/abs/2410.13582)
Comments:
          ACCV 2024 Main Paper + Supplementary (Appendix)

- **What's New**: 이 논문에서는 픽셀 수준의 감독 없이 여러 이미지에서 공통 객체의 픽셀 수준 분할(object co-segmentation)을 위한 새로운 방법을 제안합니다. 두 가지의 사전 훈련된 Vision Transformer(비전 변환기) 모델을 활용하여 이미지 내의 클래스 관련성과 클래스타입 간의 관련성을 평가합니다.

- **Technical Details**: 이 방법은 두 개의 ViT 모델을 사용하는데, 하나는 ImageNet으로 분류 훈련된 ViT로 대략적인 객체 위치 추정을 위해 사용되고, 다른 하나는 DINO로 훈련된 자가 감독형 ViT로 이미지 내부의 토큰 관련성을 평가합니다. 단계적으로 패치 수준의 클래스 관련성을 평가한 후, Biased N-Cut 방법을 통해 이미지 패치 분할을 진행합니다.

- **Performance Highlights**: 제안된 방법은 최근의 어려운 벤치마크에서 동등한 감독 수준(이미지 레이블)을 가진 방법들 중에서 최첨단 성능을 달성하였으며, 픽셀 수준의 감독이 포함된 방법들과도 경쟁력이 있습니다. 또한 이 방법은 대규모 스케치 인식 작업에서도 뛰어난 성능을 보입니다.



### DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation (https://arxiv.org/abs/2410.13571)
Comments:
this https URL

- **What's New**: 이번 논문은 자율주행 시나리오에서 4D 장면 재구성을 개선하기 위해 세계 모델 프라이어를 활용한 최초의 프레임워크인 DriveDreamer4D를 소개합니다.

- **Technical Details**: DriveDreamer4D는 자율주행 세계 모델을 생성 엔진으로 활용하여, 실제 주행 데이터 기반의 새로운 경로 비디오를 합성하는 방법을 사용합니다. 이를 통해 복잡한 주행 환경에서 전경과 배경 요소의 동적 운동을 독립적으로 조절하여 4D 장면의 공간-시간 일관성을 보장합니다. 또한, Novel Trajectory Generation Module (NTGM)을 제안하여 다양한 구조화된 교통 조건을 자동으로 생성합니다.

- **Performance Highlights**: 실험 결과, DriveDreamer4D는 새로운 경로 관점에서 생성 품질을 크게 향상시켜, PVG, S3Gaussian, Deformable-GS에 대해 FID에서 각각 24.5%, 39.0%, 10.5%의 상대적 개선을 달성했습니다. 또한, 주행 에이전트 간의 공간-시간 일관성이 눈에 띄게 향상되어 NTA-IoU 지표에서 각각 20.3%, 42.0%, 13.7%의 증가를 보였습니다.



### CCUP: A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models (https://arxiv.org/abs/2410.13567)
- **What's New**: 본 논문에서는 Cloth-changing person re-identification (CC-ReID) 문제를 해결하기 위해, 고품질의 합성 데이터 생성 파이프라인을 제안했습니다. 특히, 6,000개의 개인 ID와 1,179,976개의 이미지로 구성된 새로운 자가 주석 CC-ReID 데이터셋인 Cloth-Changing Unreal Person (CCUP)을 구축하여 기존의 데이터 드리븐 모델의 한계를 극복하고자 했습니다.

- **Technical Details**: CCUP 데이터셋은 현실적 인물 모델과 다양한 시나리오를 통해 대규모 합성 데이터를 생성하는 과정을 포함합니다. 데이터 생성 과정에서는 MakeHuman 소프트웨어를 사용하여 리얼한 인체 스켈레탈 메쉬를 생성하고, Unreal Engine을 통해 다양한 감시 환경에서의 시뮬레이션을 수행했습니다. 이 방식으로, 각 개인은 평균 26.5벌의 의상을 착용하며, CCTV 카메라를 통해 자동으로 레이블링된 데이터가 생성됩니다.

- **Performance Highlights**: 제안된 CCUP 데이터셋을 기반으로 한 프리트레인-파인튜닝(pretrain-finetune) 프레임워크는 TransReID와 FIRe^2와 같은 전통적인 모델의 일반화 능력을 개선하는 데 기여합니다. 실험 결과, CCUP에서 프리트레인 되고 각 벤치마크(PRCC, VC-Clothes, NKUP)에서 파인튜닝된 두 가지 모델이 다른 최신 모델들을 능가하는 성능을 보여주었습니다.



### 360U-Former: HDR Illumination Estimation with Panoramic Adapted Vision Transformers (https://arxiv.org/abs/2410.13566)
Comments:
          Accepted at AIM Workshop 2024 at ECCV 2024, 18 pages, 6 figures

- **What's New**: 이 논문에서는 Equirectangular Panorama (ERP) 포맷을 활용한 조명 추정의 새로운 네트워크 아키텍처인 360U-Former를 제안합니다. 이는 U-Net 스타일의 Vision-Transformer 기반으로, PanoSWIN을 활용하여 ERP 포맷에 맞게 조정된 창 당 주의(attention) 블록을 사용합니다. 이는 조명 추정 분야에서 순수한 Vision-Transformer 모델이 최초로 사용되는 사례입니다.

- **Technical Details**: 360U-Former는 Limited Field of View (LFOV) Low Dynamic Range image (LDRI)로부터 HDRI를 생성하기 위해 Generative Adversarial Network (GAN)으로 훈련되었습니다. 이 모델은 기존의 ERP 이미지에서 발생하는 아티팩트(artifacts) 문제를 해결하여, 세로 가장자리에서의 seam이나 극점(poles)에서의 왜곡을 없앴습니다. PanoSWIN 주의 모듈과 원형 패딩(circular padding)을 사용하여 ERP 이미지를 보다 정확하게 인코딩하고 생성할 수 있도록 하였습니다.

- **Performance Highlights**: 360U-Former는 기존의 최첨단 방법들과 비교하여 ERP 아티팩트를 제거하는 데에 있어 뛰어난 성능을 보였으며, 확장성과 정확도 면에서 우수한 결과를 나타냅니다. 추가적으로, 이 모델은 다양한 실내 및 실외 환경을 재현하는 데 성공하여, 조명 조건을 복잡하게 처리할 수 있는 능력을 보여줍니다.



### SDI-Paste: Synthetic Dynamic Instance Copy-Paste for Video Instance Segmentation (https://arxiv.org/abs/2410.13565)
- **What's New**: 이 논문에서는 Copy-Paste와 같은 데이터 증강(Data Augmentation) 방법을 활용하여 기존 비디오 데이터셋을 확장하는 새로운 접근법을 제안합니다.

- **Technical Details**: Synthetic Dynamic Instance Copy-Paste라는 파이프라인을 통해, 동적으로 변형되는 객체를 포함한 합성 비디오 시퀀스를 생성하고, 이를 목표 비디오 시퀀스의 프레임에 복사하여 붙여넣습니다. 이 방법은 비디오 인스턴스 분할(Video Instance Segmentation) 작업에 적용되었습니다.

- **Performance Highlights**: +2.9 AP (6.5%) 및 +2.1 AP (4.9%)의 성능 상승을 기록했습니다. 두 개의 인기 있는 네트워크를 기반으로 실험을 수행하였으며, 코드와 모델을 공개했습니다.



### Generative Location Modeling for Spatially Aware Object Insertion (https://arxiv.org/abs/2410.13564)
- **What's New**: 이번 연구에서는 객체 삽입(Object Insertion)을 위한 새로운 접근 방식을 제안합니다. 기존의 방법들은 일반적으로 객체의 위치와 형태 생성의 두 단계를 동시에 처리하기 어려웠습니다. 그러나 본 연구에서는 위치 모델(Location Model)을 따로 생성하여 객체의 적절한 위치를 먼저 찾고, 그 이후에 객체를 생성하는 두 단계의 접근 방식을 도입하였습니다.

- **Technical Details**: 우리는 배경 이미지와 원하는 객체 클래스에 조건화된 바운딩 박스 좌표를 생성하는 자기 회귀 모델(Autoregressive Model)을 훈련시킵니다. 이를 통해 객체의 적절한 위치를 식별하고, 기존의 위치 데이터셋에서 희소한 주석(Sparse Annotations)을 효과적으로 처리합니다. 추가로, 긍정적 및 부정적 레이블을 활용한 직접 선호 최적화(Direct Preference Optimization)를 통해 모델의 정확도를 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 생성적 위치 모델이 최신 Instruction-tuned 모델 및 위치 모델링 기준선보다 뛰어난 성능을 보임을 입증했습니다. 특히, 위치 추정의 정확성이 높아질수록 생성된 이미지 품질이 향상되는 경향을 확인했습니다. 사용자 연구(User Study)에서 본 접근 방식이 효과적임을 추가로 검증하였습니다.



### RemoteDet-Mamba: A Hybrid Mamba-CNN Network for Multi-modal Object Detection in Remote Sensing Images (https://arxiv.org/abs/2410.13532)
- **What's New**: 본 논문에서는 원거리 이미지를 활용한 무인 항공기(UAV) 원거리 탐지를 위한 새로운 멀티모달 탐지 네트워크인 RemoteDet-Mamba를 제안합니다. 이 네트워크는 단일 모드의 지역적 특성을 학습하고 패치 수준에서 글로벌 특성을 통합하여 작은 물체의 식별을 개선합니다.

- **Technical Details**: RemoteDet-Mamba는 Siamese CNN 네트워크와 Cross-modal Fusion Mamba (CFM) 모듈로 구성됩니다. CFM 모듈은 선택적 스캔 2D 메커니즘(SS2D)을 기반으로 하며, 특징의 4방향 스캔을 통해 밀집하게 분포된 작은 객체를 효과적으로 분리하고 글로벌 정보를 추출합니다. 이 구조는 선형 시간 복잡성을 유지합니다.

- **Performance Highlights**: DroneVehicle 데이터셋에서의 실험 결과, RemoteDet-Mamba는 최신 기법에 비해 우수한 탐지 성능을 보여주며, 계산 효율성과 매개변수 수를 유지했습니다.



### L3DG: Latent 3D Gaussian Diffusion (https://arxiv.org/abs/2410.13530)
Comments:
          SIGGRAPH Asia 2024, project page: this https URL , video: this https URL

- **What's New**: L3DG는 처음으로 3D Gaussian의 생성적 모델링을 위한 접근법으로, 잠재적 3D Gaussian 확산(3D Gaussian diffusion) 방식을 도입합니다. 이를 통해 전체 방 크기의 장면을 생성할 수 있는 효과적인 생성적 3D 모델링이 가능해졌습니다.

- **Technical Details**: L3DG는 VQ-VAE(vectored-quantized variational autoencoder)를 사용하여 3D Gaussian의 압축된 잠재 공간을 학습합니다. 이 공간은 희소(convolutional) 아키텍처를 통해 구성되며, 이는 효율적인 방 크기 장면 처리를 가능하게 합니다. 이 접근 방식은 8천 개의 Gaussian을 사용하는 작은 규모의 객체와 20만 개의 Gaussian을 사용하는 방 크기 장면 모두를 위한 고충실도(high-fidelity) 뷰 합성을 지원합니다. 3D Gaussian을 통해 생성된 장면은 임의의 시점(viewpoint)에서 실시간으로 렌더링할 수 있습니다.

- **Performance Highlights**: L3DG는 기존의 무조건적 객체 수준의 방사 필드(rapiance field) 합성보다 시각적 품질을 크게 개선하였으며, 대규모 장면 생성에서 더욱 효율적으로 확장 가능한 가능성을 보여줍니다. 실험 결과, L3DG는 PhotoShape 데이터셋에서 FID 지표가 약 45% 향상되었습니다.



### Generative Adversarial Synthesis of Radar Point Cloud Scenes (https://arxiv.org/abs/2410.13526)
Comments:
          ICMIM 2024; 7th IEEE MTT Conference

- **What's New**: 이 논문에서는 자동차 레이더의 검증과 검증을 위해 현실적인 교통 시나리오 데이터셋이 필요하다는 점을 논의하며, GANs(Generative Adversarial Networks)를 활용한 레이더 장면 합성을 제안합니다.

- **Technical Details**: PointNet++ 기반의 GAN 모델을 사용하여 현실적인 레이더 포인트 클라우드 장면을 생성하며, 생성된 장면의 성능을 실제 장면의 테스트 세트와 비교하기 위해 이진 분류기(binary classifier)를 사용합니다.

- **Performance Highlights**: 우리의 GAN 모델은 실제 장면 테스트 세트에 대해 ~87%의 유사한 성능을 달성함을 보여줍니다.



### Can Medical Vision-Language Pre-training Succeed with Purely Synthetic Data? (https://arxiv.org/abs/2410.13523)
Comments:
          Under Review

- **What's New**: 의료 이미지 분석에 대한 기존의 MedVLP 모델이 실제 데이터에 의존하는 반면, 이 연구는 고품질의 합성 데이터(Synthetic Data)를 사용하여 모델을 학습시키는 방안을 제안합니다. 특히, 현실 데이터에 비해 합성 데이터만으로 훈련된 MedVLP 모델이 매력적인 성과를 보였습니다.

- **Technical Details**: 연구에서는 off-the-shelf generative models를 사용하여 200,000개의 합성 X-ray 이미지와 보고서를 포함하는 SynCXR 데이터셋을 생성했습니다. 이 데이터셋은 데이터 품질과 분포를 조절하여 구축되었습니다. 제안된 자동화된 파이프라인을 통해 생성된 합성 데이터만으로 훈련한 MedVLP 모델은 실제 데이터로 훈련한 모델보다 AUC에서 평균 3.8% 개선된 성능을 보였습니다.

- **Performance Highlights**: 합성 데이터 또는 혼합(data mixing) 데이터를 사용하여 훈련된 MedVLP 모델은 실제 데이터에서 훈련된 모델보다 일관되게 우수한 성과를 나타냅니다. 특히 zero-shot classification과 segmentation에서 강력한 성능을 발휘하며, 특정 영역에서는 성능이 9.07% 향상되기도 했습니다.



### SAda-Net: A Self-Supervised Adaptive Stereo Estimation CNN For Remote Sensing Image Data (https://arxiv.org/abs/2410.13500)
Comments:
          Will be presented at ICPR2024 in December 2024 in Kolkata, India

- **What's New**: 본 논문에서는 기존의 깊이 학습(deep learning) 기반 스테레오 추정(stereo estimation) 방법이 정확한 지상 진리(ground truth) 데이터에 의존하는 단점을 극복하기 위해, 지상 진리 데이터 없이도 훈련이 가능한 자가 지도(Self-supervised) CNN을 제안합니다.

- **Technical Details**: 제안된 방법은 단계별로 진행되며, 초기에 생성된 분산 맵(disparity map)은 부정확하고 잡음(noise)이 많습니다. 왼쪽-오른쪽 일관성 체크(left-right consistency check)를 사용하여 초기 의사 지상 진리(pseudo ground-truth)를 생성하고, 이를 기반으로 매 에포크(epoch)마다 모델을 업데이트합니다. 불일치 포인트의 합을 통해 네트워크의 수렴(convergence)을 추적합니다.

- **Performance Highlights**: 실제 복잡한 장면에서 좋은 성능을 나타내며, 약 495K의 경량화된 파라미터를 사용함으로써 상업적 하드웨어에서 효율적으로 사용 가능합니다.



### SemSim: Revisiting Weak-to-Strong Consistency from a Semantic Similarity Perspective for Semi-supervised Medical Image Segmentation (https://arxiv.org/abs/2410.13486)
- **What's New**: 이 논문은 Medical image segmentation에서의 새로운 반지도 학습(Semi-supervised learning)을 제안합니다. 여기서 제안된 SemSim 프레임워크는 기존의 FixMatch에서 발전하였으며, 주요한 두 가지 문제를 해결하기 위한 접근 방식을 포함하고 있습니다.

- **Technical Details**: 본 논문에서는 두 가지 주요 개념, 즉 intra-image semantic consistency와 cross-image semantic consistency를 바탕으로 한 SemSim 프레임워크를 제안합니다. Intra-image에서는 픽셀 간의 관계를 고려하여 예측을 정제하고, cross-image에서는 라벨이 있는 데이터와 라벨이 없는 데이터 간의 유사성을 활용하여 더 일치된 클래스 분포를 확립합니다. 또한, 이 과정에서 spatial-aware fusion module (SFM)을 통하여 다양한 스케일의 특징 정보를 조합하여 더 나은 성능을 이끌어냅니다.

- **Performance Highlights**: 다수의 공공 segmentation 벤치마크에 대한 실험 결과, SemSim은 기존의 최첨단 반지도 학습 방법들에 비해 일관된 개선을 보여줍니다.



### Day-Night Adaptation: An Innovative Source-free Adaptation Framework for Medical Image Segmentation (https://arxiv.org/abs/2410.13472)
Comments:
          10 pages, 4 figures, 6 tables

- **What's New**: 이 논문은 의료 영상의 분포 변화(distribution shifts) 문제를 해결하기 위한 새로운 접근 방식인 Day-Night Adaptation(DyNA) 프레임워크를 제안합니다. 이 프레임워크는 Source-free Domain Adaptation(SFDA)와 Test-Time Adaptation(TTA)을 통합하여 의료 데이터의 개인 정보 보호를 유지하면서도 모델을 효과적으로 적응시키는 방법을 소개합니다.

- **Technical Details**: DyNA 프레임워크는 낮과 밤의 적응 루프를 통해 사전 교육된 모델을 목표 도메인에 지속적으로 적응시키기 위한 전략을 갖추고 있습니다. 낮에는 모델의 파라미터를 고정하고 각 테스트 샘플에 대해 저주파(low-frequency) 프롬프트를 훈련하며, 메모리 뱅크를 구성하여 프롬프트 초기화를 도와줍니다. 밤에는 기존의 Teacher-Student self-training Paradigm에 글로벌 학생 모델을 통합하여 아울러 훈련 안정성을 유지합니다.

- **Performance Highlights**: DyNA는 두 가지 기준 의료 영상 분할 작업(폴립 분할 및 시신경원판/컵 분할)에서 여러 최신 TTA 및 SFDA 방법과 비교하여 뛰어난 성능을 보였습니다. 이로 인해 DyNA는 특히 임상 상황에서의 효용성이 입증되었습니다.



### SiamSeg: Self-Training with Contrastive Learning for Unsupervised Domain Adaptation in Remote Sensing (https://arxiv.org/abs/2410.13471)
- **What's New**: 본 논문은 원거리 감지(Remote Sensing) 이미지의 의미 세분화(Semantic Segmentation)에서 새로운 접근법인 SiamSeg를 제시합니다. 이는 대비 학습(Contrast Learning)을 최초로 도입하여 의미 정보 학습의 부족 문제를 해결하고 세분화 네트워크의 성능을 향상시킵니다.

- **Technical Details**: SiamSeg는 동일한 이미지의 다양한 데이터 증강을 통해 긍정적인 샘플 쌍을 생성하고, Siamese 네트워크를 사용하여 모델을 최적화합니다. 새로운 손실 함수가 제안되어, 대비 학습 손실을 포함하여 모델의 학습 효과성을 증대시킵니다.

- **Performance Highlights**: SiamSeg는 다양한 데이터 세트에서 새로운 최첨단 성능을 달성하며, 특히 원거리 감지 이미지의 세분화 과정에서 중요한 도메인 편향 문제를 효과적으로 해결합니다.



### Object Pose Estimation Using Implicit Representation For Transparent Objects (https://arxiv.org/abs/2410.13465)
- **What's New**: 이 논문은 Neural Radiance Field (NeRF)를 활용하여 투명 객체의 6D pose 추정을 위한 새로운 파이프라인을 제안합니다. 기존의 CAD 모델 대신 신경망 기반의 형상을 사용하여, 실제 장면을 보다 사실적으로 렌더링하고 비교할 수 있는 기법을 도입하였습니다.

- **Technical Details**: 이 접근법은 render-and-compare 방식을 기반으로 하며, NeRF를 활용하여 시각적 종속성을 고려한 고품질 가설을 렌더링합니다. 이 방법은 RGB 이미지 및 다중 시점의 이미지 집합을 활용하여 6D 포즈를 추정하며, 다양한 평가지표인 MSPD, MSSD, ADD 등을 사용하여 성능을 검증합니다.

- **Performance Highlights**: 제안된 NeRF 기반의 render-and-compare 방법은 투명 및 반사적인 가정용 객체에 대한 대규모 데이터셋에서 현재의 최첨단 결과를 초과하는 성과를 거두었습니다.



### Augmentation Policy Generation for Image Classification Using Large Language Models (https://arxiv.org/abs/2410.13453)
Comments:
          5 pages, 2 figures, 4 tables, submitted for consideration to the International Workshop on Computational Intelligence for Multimedia Understanding (IWCIM), ISCAS 2025

- **What's New**: 이 논문은 대형 언어 모델을 활용하여 데이터 세트에 맞춤화된 효율적인 데이터 증강 정책을 자동으로 생성하는 전략을 제안합니다. 기존의 많은 데이터 증강 기법이 특정 데이터 세트에 최적화되어 있었으나, 저자는 모든 데이터 유형에 적용 가능한 데이터 증강 파이프라인을 개발하여 모델의 성능을 개선하였습니다.

- **Technical Details**: 제안된 방법은 LLM (Large Language Model)을 사용하여 데이터 세트와 모델 아키텍처의 특정 특성에 맞춘 증강 정책을 생성합니다. 이 과정에 있어 LLM은 반복적으로 모델 성능 피드백과 상호작용하여 증강 정책을 정련해 나갑니다. 각 반복 과정은 데이터 설명, 모델 아키텍처, 목표 평가 지표와 증강 횟수를 기반으로 합니다.

- **Performance Highlights**: 의료 이미지 데이터 세트를 이용한 실험에서 제안된 방법은 기존의 최첨단 데이터 증강 기술을 초과하는 성과를 보였습니다. APTOS 2019 데이터 세트에서 ResNet18 모델이 0.8743의 검증 정확도를 기록하였으며, Gemini 및 ChatGPT 설정 모두에서 최고의 성능을 발휘했습니다.



### Temporal-Enhanced Multimodal Transformer for Referring Multi-Object Tracking and Segmentation (https://arxiv.org/abs/2410.13437)
- **What's New**: 이번 연구에서 새롭고 향상된 교차 모달 모델인 TenRMOT을 소개했습니다. 이 모델은 언어 표현을 활용하여 비디오 내에서 목표 객체를 추적하는 작업에 있어 시각적 및 언어적 정보를 결합합니다.

- **Technical Details**: TenRMOT은 Transformer 기반의 방법론으로, 인코딩 및 디코딩 단계에서 기능 융합을 진행합니다. 새로운 Interleaving Cross-modality Encoder(ICE)와 Query Update Module(QUM)을 도입하여 객체 추적의 정확도를 높입니다. 특히, 언어 기반 쿼리를 활용하여 메모리 기능을 적절히 탐색하며, 지속적인 쿼리 업데이트를 통해 객체의 일관성을 보장합니다.

- **Performance Highlights**: TenRMOT은 Ref-KITTI Segmentation 데이터셋에서 준수한 성능을 보여주며, 이는 각각 평균 10.7개의 마스크를 포함한 총 818개의 표현으로 이루어져 있습니다. 이 새로운 데이터셋은 기존 비디오 세그멘테이션 데이터셋보다 더 큰 도전 과제를 제공합니다.



### Performance of Gaussian Mixture Model Classifiers on Embedded Feature Spaces (https://arxiv.org/abs/2410.13421)
Comments:
          8 pages

- **What's New**: 이 논문은 CLIP 및 ImageBind와 같은 데이터 임베딩을 이용하여 다중 모달 데이터의 분석을 위한 성능을 분류에 대한 대안으로서 Gaussian Mixture Model (GMM) 기반 계층을 사용하는 것으로 평가합니다. 주요 기여는 이렇게 임베딩된 공간에서 GMM 분류 성능을 조사하고 GMM 기반 분류기를 제안하는 것입니다.

- **Technical Details**: Gaussian mixture models (GMMs)는 강력한 확률 밀도 함수로, 이미지 분류를 위한 데이터 임베딩으로 CLIP 및 ImageBind를 사용하여 SDGM(Sparse Discriminative Gaussian mixture)와 DGMMC(Deep Gaussian Mixture Model Classifier) 계층을 평가합니다. DGMMC는 이전 제안보다 더 적은 매개변수를 가진 새로운 분류기입니다.

- **Performance Highlights**: DGMMC는 이미지 데이터셋에서 CLIP과 ImageBind보다 높은 정확도를 이끌어내며, GMM에서 필요한 가우시안 구성 요소 수는 종종 하나(G=1)로 충분함을 발견했습니다. ImageBind는 이미지 데이터 분류에서 CLIP보다 더 좋은 성능을 제공합니다.



### RescueADI: Adaptive Disaster Interpretation in Remote Sensing Images with Autonomous Agents (https://arxiv.org/abs/2410.13384)
- **What's New**: 본 논문에서는 현재 재해 상황 해석에서의 한계를 극복하기 위해 Adaptive Disaster Interpretation (ADI)라는 새로운 태스크를 소개합니다. ADI는 다수의 해석 태스크를 연계하여 재해 장면에 대한 종합적인 분석을 제공합니다.

- **Technical Details**: 작가는 RescueADI라는 새로운 데이터셋을 발표하였습니다. 이 데이터셋은 고해상도 원격 탐지 이미지와 함께 계획, 인식, 식별을 위한 주석을 포함하고 있으며, 9종의 요청 유형을 포함한 4,044개의 RSIs, 16,949개의 의미 마스크, 14,483개의 객체 경계 박스를 포함합니다. 제안된 새로운 해석 방법은 대형 언어 모델(LLMs)을 사용하여 자율 에이전트를 통해 태스크를 계획하고 실행합니다.

- **Performance Highlights**: RescueADI 데이터셋에 대한 실험 결과, 제안된 방법이 기존 시각적 질문 응답(VQA) 방법보다 9% 더 높은 정확도를 달성하여 전통적인 재해 해석 접근 방식에 비해 우수하다는 것을 보여줍니다.



### Railway LiDAR semantic segmentation based on intelligent semi-automated data annotation (https://arxiv.org/abs/2410.13383)
Comments:
          This article has been accepted for publication in the IEEE VTC Fall 2024

- **What's New**: 이 논문은 자동화된 기차의 환경 인식을 위한 3D LiDAR 의미(segmentation) 분할 방법을 새롭게 제안합니다. 특히, 카메라 이미지와 LiDAR 스캔을 결합하여 점별 3D 의미 분할(point-wise 3D semantic segmentation)을 수행하는 아키텍처인 2DPass 네트워크를 활용합니다.

- **Technical Details**: 이 연구는 데이터셋 준비, 라벨링, 모델 훈련의 세 가지 단계로 이루어집니다. 데이터 손실을 줄이기 위해 Deeplabv3+ 네트워크를 사용한 이미지 의미(segmentation) 네트워크로부터 라벨을 추출하고, 수동으로 수정된 52개의 정밀 라벨로 구성된 데이터셋을 생성합니다. 그 후, 최첨단 LiDAR 의미(segmentation) 네트워크인 2DPass를 훈련하여 총 9개의 클래스에 대해 평균 IoU(Intersection over Union) 71.48%의 성능을 달성합니다.

- **Performance Highlights**: 제안된 네트워크는 작은 크기의 라벨이 붙은 스캔에서 효과적으로 훈련되어, 71.48%의 높은 평균 IoU 성능을 보여줍니다. 이는 기차의 안전한 운전 및 장애물 감지에 필수적인 정확한 환경 인식 기술로 평가됩니다.



### Accurate Checkerboard Corner Detection under Defoucs (https://arxiv.org/abs/2410.13371)
- **What's New**: 본 논문은 체커보드 코너 검출을 위한 새로운 서브픽셀 개선 방법을 제안하며, 기존의 대칭 기반 방법의 한계를 보완하여 가시광선 카메라에서 정확성을 크게 향상시킵니다.

- **Technical Details**: 우리는 대칭 기반의 서브픽셀 정밀도 개선 접근 방식을 소개하며, 불명확한 이미지와 초점 흐림(defocus) 영향을 고려한 단순화된 목적 함수를 도출합니다. 이 방법은 계산 시간을 줄이고 과적합(overfitting) 위험을 완화하는 데 기여합니다.

- **Performance Highlights**: 새로운 방법은 기존 기술보다 가시광선 카메라 캘리브레이션에서 상당한 정확성 개선을 보여주었으며, 특히 기존 방법보다 1/2121/21 / 2 및 1/4141/41 / 4로 재투영 오류를 줄이는 결과를 기록했습니다.



### MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models (https://arxiv.org/abs/2410.13370)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 텍스트-이미지(T2I) 모델의 개인화 작업에서 구성 요소를 제어할 수 있는 새로운 접근법을 제시합니다. 사용자가 특정 시각적 개념의 개별 요소를 재구성할 수 있도록 하여 더 정밀한 커스터마이징을 가능하게 합니다. 

- **Technical Details**: 새로운 프레임워크인 MagicTailor를 통해 Dynamic Masked Degradation (DM-Deg)과 Dual-Stream Balancing (DS-Bal)을 활용하여 개인화 과정에서의 의미적 오염(semantic pollution)을 줄이고, 의미적 불균형(semantic imbalance)을 관리합니다. 이 프레임워크는 T2I 모델을 동적으로 조정하여 개인화된 개념을 정확하게 반영합니다.

- **Performance Highlights**: MagicTailor는 실험을 통해 구성 요소 제어가 가능한 개인화 작업에서 최첨단(SOTA) 성과를 달성하였으며, 다양한 추가 응용 가능성을 보여줍니다.



### Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistan (https://arxiv.org/abs/2410.13360)
- **What's New**: 본 논문에서는 Retrieval Augmented Personalization (RAP) 프레임워크를 소개하여 다중 모드 대형 언어 모델(MLLMs)의 개인화를 가능하게 합니다. RAP는 일반 MLLM을 개인화된 어시스턴트로 전환하는 세 가지 주요 단계로 구성됩니다: 기억(Recall), 검색(Retrieve), 생성(Generate).

- **Technical Details**: RAP는 사용자 관련 정보(예: 이름, 아바타 등)를 저장하는 키-값 데이터베이스를 설계합니다. 사용자가 대화를 시작할 때, RAP는 다중 모드 검색기를 통해 데이터베이스에서 관련 정보를 검색하고, 이를 MLLM에 입력하여 개인화된 지식 강화 응답을 생성합니다. 추가로 생성 품질 향상을 위해 데이터 수집 파이프라인을 개발하고 개인화된 훈련을 위한 전문적인 데이터셋을 생성합니다.

- **Performance Highlights**: RAP-MLLMs는 개인화된 이미지 캡션 작성, 질문 응답 및 시각적 인식과 같은 다양한 작업에서 뛰어난 유연성과 생성 품질을 보여줍니다. 모델들은 무한한 시각적 개념에 대해 일반화 능력을 발휘하며, 사용자 관련 정보를 효과적으로 처리하여 개인화된 출력을 제공합니다.



### Self-Supervised Scene Flow Estimation with Point-Voxel Fusion and Surface Representation (https://arxiv.org/abs/2410.13355)
Comments:
          The paper is under consideration at 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이 논문에서는 기존의 포인트 기반 방식과 복셀 기반 기법의 단점을 보완하기 위해 포인트-복셀 융합(Point-Voxel Fusion) 방법을 제안합니다. 이 방법은 희소 격자 주의(Sparse Grid Attention) 및 이동창(Dynamic Windowing) 전략을 활용하여 장거리 의존성을 캡처하면서 또한 세부적인 특성을 추출합니다.

- **Technical Details**: 포인트-복셀 융합 아키텍처는 포인트 브랜치가 한층 정교한 특징을 추출하고, 복셀 브랜치는 장거리 의존성을 포착합니다. Umbrella Surface Feature Extraction (USFE) 모듈을 통해 복잡한 3D 객체의 지역 표면 정보를 명시적으로 인코딩하고, 이를 통해 기하학적 구조를 정밀하게 유지합니다.

- **Performance Highlights**: FlyingThings3D 및 KITTI 데이터셋에서 실험을 거쳤으며, 이번 방법은 모든 다른 자가 지도 방법들을 초과하며, 완전 감독 방법에 비해서도 매우 경쟁력 있는 결과를 달성했습니다. 특히 KITTIo와 KITTIs 데이터셋에서 EPE가 각각 8.51% 및 10.52% 감소했습니다.



### GlossyGS: Inverse Rendering of Glossy Objects with 3D Gaussian Splatting (https://arxiv.org/abs/2410.13349)
- **What's New**: 본 연구에서는 GlossyGS라는 혁신적인 3D-GS 기반의 역 렌더링 프레임워크를 소개합니다. 이 프레임워크는 재료(priors) 통합을 통해 반짝이는(glossy) 객체의 기하학(geometry)과 재료(materials)를 정확하게 재구성하는 것을 목표로 합니다.

- **Technical Details**: GlossyGS는 micro-facet geometry segmentation prior를 사용하여 본래의 모호성을 줄이고 기하학과 재료의 분해를 개선합니다. 또한, 반사 표면의 법선(normal distribution) 분포를 더 정확하게 시뮬레이션하기 위해 normal map prefiltering 전략을 도입하였습니다. 이를 통해 반짝이는 객체를 묘사하기 위해 명시적(explicit) 및 암시적(implicit) 방법을 사용하는 하이브리드 기하학 및 재료 표현을 구현합니다.

- **Performance Highlights**: 정량적 분석과 정성적 시각화를 통해 제안된 방법이 고충실도(high-fidelity) 기하학과 재료를 재구성하는 데 효과적이며, 최신 기술(state-of-the-art)과 비교할 때 우수한 성능을 발휘한다는 것을 보여주었습니다.



### Inadequate contrast ratio of road markings as an indicator for ADAS failur (https://arxiv.org/abs/2410.13320)
Comments:
          IRF World Congress 2024

- **What's New**: 도로 markings는 인간 운전자는 물론이고 ADAS(Advanced Driver Assistance Systems)와 자율 주행에 사용되는 기계 비전 기술에도 필수적인 도로 안전 기능으로 보고되고 있습니다. 이 연구에서는 여러 가시성 조건에서 카메라 기반 ADAS의 테스트 중 발생한 경로 계획의 심각한 실패를 기록했습니다.

- **Technical Details**: 본 연구는 다양한 가시성 조건(낮, 밤, 비, 눈부심)에서 ADAS의 도로 marking 인식 성능을 분석하였습니다. Type II 도로 marking(구조화된 marking)은 Type I 도로 marking(평평한 선)보다 저조한 가시성 조건에서 일관되게 더 나은 신뢰성을 보였습니다. 도로 marking의 대비 비율(contrast ratio)은 ADAS의 교통 차선 인식에 있어 중요한 요소로 분석되었습니다. 가장 높은 대비 비율은 밤 시간대, 방해 요소 없이 측정되었으며, Type II가 Type I보다 0.1의 통계적으로 유의미한 차이를 보였습니다.

- **Performance Highlights**: 주목할 만한 것은, ADAS의 불충분한 차선 인식은 도로 marking의 매우 낮은 대비 비율과 관련이 있었다는 점입니다. 비 또는 젖은 도로에서 대비 비율은 저하되었으며, Type II marking이 Type I보다 유의미하게 높은 대비 비율을 유지했습니다. 그러나 특정 최소 대비 비율 값은 ADAS 알고리즘의 복잡성으로 인해 찾을 수 없었습니다.



### Enhancing Dataset Distillation via Label Inconsistency Elimination and Learning Pattern Refinemen (https://arxiv.org/abs/2410.13311)
Comments:
          ECCV 2024 Dataset Distillation Challenge

- **What's New**: 이번 연구는 ECCV-2024 데이터 증류 챌린지에서 1위를 차지한 M-DATM(Modified Difficulty-Aligned Trajectory Matching) 기법을 제안하고, 고유의 버전의 DATM을 개선하여 라벨 불일치와 레이트(Late) 궤적 정보 학습의 어려움을 극복하는 방법을 정리합니다.

- **Technical Details**: M-DATM은 DATM에서 두 가지 중요한 수정을 소개합니다: (1) 소프트 라벨 기법 제거 - 이는 라벨 불일치를 줄이고, 평가 시의 일관성을 보장합니다; (2) 일치 범위 조정 - Tiny ImageNet에서의 어려운 패턴 학습 문제를 해결하기 위해 학습 패턴의 난이도를 낮추어 주는 것입니다. M-DATM을 통해 CIFAR-100과 Tiny ImageNet에서 각각 0.4061과 0.1831의 정확도를 기록하며, IPC 트랙에서 1위를 달성했습니다.

- **Performance Highlights**: M-DATM은 ECCV-2024 데이터 증류 챌린지에서 IPC 트랙 1위로 선정되었으며, 이는 데이터 효율성을 높이고, 향후 데이터 증류 연구의 강력한 기준이 될 것입니다.



### LESS: Label-Efficient and Single-Stage Referring 3D Segmentation (https://arxiv.org/abs/2410.13294)
- **What's New**: 본 연구에서는 Referring 3D Segmentation의 새로운 프로세스를 제안하며, 이를 LESS(Label-Efficient and Single-Stage)라는 이름으로 부릅니다. 기존의 2단계 방법 대신, 이 모델은 효율적인 binary mask만으로 감독 받아 단일 단계에서 작업을 수행합니다.

- **Technical Details**: LESS는 Point-Word Cross-Modal Alignment 모듈을 통해 포인트와 단어의 정밀한 특성을 정렬하고, Query Mask Predictor 모듈과 Query-Sentence Alignment 모듈을 통해 마스크와 쿼리 간의 거친 정렬을 수행합니다. 특히, area regularization loss와 point-to-point contrastive loss를 통해 복잡한 객체 배경의 간섭을 제거하는 방법을 고안했습니다.

- **Performance Highlights**: LESS는 ScanRefer 데이터셋에서 기존의 방법보다 약 3.7% mIoU의 성능 향상을 보이며 최첨단 성과를 달성했습니다. 이는 binary label만으로도 가능한 혁신적인 접근임을 보여줍니다.



### Composing Novel Classes: A Concept-Driven Approach to Generalized Category Discovery (https://arxiv.org/abs/2410.13285)
Comments:
          Underreview. The first two authors contribute equally

- **What's New**: 본 논문에서는 라벨이 없는 데이터셋에서 새로운 클래스를 발견하는 일반화된 범주 발견(Generalized Category Discovery, GCD) 문제를 다룹니다. 이미 알려진 클래스의 지식을 활용하여 새로운 클래스를 발견하기 위한 새로운 개념 학습 프레임워크인 ConceptGCD를 제안합니다. 이 방법은 두 가지 유형의 개념—파생 가능한(derivable) 개념과 파생 불가능한(underivable) 개념으로 분류하고, 단계별 학습 방식을 채택하여 각 개념을 별도로 학습합니다.

- **Technical Details**: ConceptGCD 프레임워크는 다음 세 가지 핵심 단계로 구성됩니다: 1) 알려진 클래스 개념 학습: 라벨이 있는 알려진 클래스 데이터를 통해 딥 네트워크 모델을 훈련하여 개념을 얻습니다. 이를 위해 개념 공분산 손실(concept covariance loss)을 도입하여 다양한 개념 간 독립성을 촉진합니다. 2) 파생 가능한 개념 생성: 알려진 클래스 개념을 기반으로 선형 층과 ReLU 층을 활용하여 파생 가능한 개념을 생성합니다. 3) 파생 불가능한 개념 학습: 마지막 단계에서 파생 불가능한 개념을 학습하며, 이전에 생성된 개념을 보존합니다. 이를 위해 처음 선형 층의 차원을 확장하고, 특성 공간(feature space)에서 대조 손실(contrastive loss)로 학습합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 진행한 실험은 ConceptGCD가 기존 최첨단 방법들에 비해 향상된 성능을 보임을 보여줍니다. 특히, 제안하는 공분산 손실과 개념 점수 정규화(concept score normalization) 기술이 개념 학습에 기여한 바가 큽니다.



### Hybrid bundle-adjusting 3D Gaussians for view consistent rendering with pose optimization (https://arxiv.org/abs/2410.13280)
Comments:
          Photonics Asia 2024

- **What's New**: 본 논문은 불완전한 카메라 자세에서 시각적으로 일관된 새로운 시점을 렌더링하는 도전 과제를 해결하기 위한 하이브리드 번들 조정 3D 가우시안 모델을 소개합니다.

- **Technical Details**: 이 모델은 이미지 기반 및 신경 3D 표현을 조합하여 정면 장면에서 일관된 이미지를 생성하고 카메라 자세를 최적화합니다. 주요 기술 구성 요소는 다음과 같습니다: 1) 포인트 클라우드를 복셀화하여 신경 3D 앵커 기능을 추출하고, 2) 근처 뷰의 이미지 기능을 활용하여 렌더링 품질을 향상시키며, 3) 초기 카메라 자세를 조정하기 위해 조정된 번들 조정 기법을 사용합니다.

- **Performance Highlights**: 실제 및 합성 데이터셋에서의 광범위한 실험을 통해 모델이 카메라 자세 불일치를 해결하면서 신경 장면 표현을 효과적으로 최적화할 수 있음을 보여줍니다.



### Inductive Gradient Adjustment For Spectral Bias In Implicit Neural Representations (https://arxiv.org/abs/2410.13271)
Comments:
          28 pages, 12 figures

- **What's New**: 이 논문은 Implicit Neural Representations (INRs)에서의 spectral bias 문제를 해결하기 위해, Multi-layer Perceptrons (MLPs)의 선형 역학 모델을 탐구하고 empirical Neural Tangent Kernel (eNTK) 행렬을 기반으로 inductive gradient adjustment (IGA) 방법을 제안합니다.

- **Technical Details**: 이 연구는 MLPs의 linear dynamics 모델을 이용하여 empirical NTK (eNTK) 행렬을 통해 spectral bias와 training dynamics 사이의 관계를 이론적으로 규명합니다. 제안하는 IGA 방법은 대량의 데이터 포인트에 대해 eNTK 기반 gradient 변환 행렬의 inductive 일반화를 통해 spectral bias를 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 기존의 training dynamics 조정 방법들보다 더 나은 성능을 발휘하며, INRs의 퀄리티를 향상시켜 고해상도 텍스처와 뚜렷한 변별력을 선보임을 보여줍니다.



### Fundus to Fluorescein Angiography Video Generation as a Retinal Generative Foundation Mod (https://arxiv.org/abs/2410.13242)
- **What's New**: Fundus2Video는 단일 CF 이미지에서 동적 FFA 비디오를 생성하는 자가 회귀 생성적 적대 신경망(GAN) 모델로, 기존의 정적 이미지 생성 방법의 한계를 극복합니다.

- **Technical Details**: 이 모델은 동적 FFA 비디오 생성에서 FVD(Frechet Video Distance) 1497.12, PSNR(Peak Signal-to-Noise Ratio) 11.77을 기록하며, 생성된 비디오의 신뢰성은 임상 전문가들에 의해 검증되었습니다. 또한, 모델의 생성기는 혈관 세분화(blood vessel segmentation), 망막 질병 진단(retinal disease diagnosis), 전신 질병 예측(systemic disease prediction) 및 다중 모달 검색(multimodal retrieval)에서 탁월한 전이 학습 성능을 보여줍니다.

- **Performance Highlights**: Fundus2Video는 제로샷(zero-shot) 및 소샷(few-shot) 가능성이 뛰어나며, 비침습적(non-invasive)인 FFA 검사의 강력한 대안으로 자리매김하게 됩니다.



### Latent Image and Video Resolution Prediction using Convolutional Neural Networks (https://arxiv.org/abs/2410.13227)
Comments:
          Submitted in ICIP conference

- **What's New**: 이 논문은 Video Quality Assessment (VQA) 문헌에서 거의 주목받지 않은 latent resolution prediction 문제를 소개합니다. 이 문제는 이미지나 비디오가 원래 해상도보다 높은 해상도로 업스케일된 경우 발생합니다. 논문은 문제를 형식화하고 훈련 및 평가를 위한 데이터셋을 구축하며, 이 문제를 해결하기 위해 두 가지 Convolutional Neural Network (CNN) 알고리즘을 포함한 머신러닝 알고리즘을 소개합니다.

- **Technical Details**: 논문에서 제안하는 두 가지 CNN 기반 방법은 이미지의 질감이 있는 부분에 초점을 맞추지만, 이미지를 처리하는 방식에는 차이가 있습니다. 첫 번째 방법은 이미지에서 관심 지역에서 추출한 여러 패치를 사용하는 반면, 두 번째 방법은 전체 이미지를 입력으로 받아 출력 맵을 생성합니다. 비트맵을 통해 관심 점의 위치를 추적하고, 신뢰할 수 있는 품질 예측을 추출합니다. 이는 144p에서 1080p까지 다양한 latent resolution을 가진 이미지와 비디오로 구성된 커스텀 데이터셋에서 실험하여 예측합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법들이 약 95%의 정확도로 이미지/비디오의 latent resolution을 예측할 수 있음을 보여줍니다.



### UniG: Modelling Unitary 3D Gaussians for View-consistent 3D Reconstruction (https://arxiv.org/abs/2410.13195)
- **What's New**: 이 논문에서는 UniG라는 새로운 3D 재구성 및 새로운 보기 합성 모델을 제안합니다. 이 모델은 드문 이미지로부터 높은 신뢰도의 3D Gaussian 표현을 생성합니다. 기존의 3D Gaussian 기반 방법은 각 보기당 Gaussian을 독립적으로 추정하였으나, 이는 보기 간 불일치 문제를 야기합니다. UniG는 DETR(Distance-Equivalent Transformation Representation) 유사 구조를 통해 이 문제를 해결합니다.

- **Technical Details**: UniG는 다수의 입력 이미지에서 다중 보기 교차 주의(Multi-view Cross-Attention, MVDFA)를 통해 3D Gaussian 쿼리를 업데이트합니다. 각 3D Gaussian은 콘텐츠와 위치(3D Gaussian의 중심) 부분으로 구성되며, 변형 가능 Transformer 인코더-디코더 구조에서 반복적으로 정제됩니다. 또한, 저렴한 메모리 요구 사항을 갖춘 3D Spatial Efficient Self-Attention (SESA) 방식을 사용하여 처리 효율성을 높입니다.

- **Performance Highlights**: UniG는 Objaverse에서 학습하고 GSO 벤치마크에서 테스트할 때 PSNR을 4.2 dB 개선하며, 기존 방법에 비해 정량적 및 정성적으로 우수한 성능을 보입니다.



### FAMSeC: A Few-shot-sample-based General AI-generated Image Detection Method (https://arxiv.org/abs/2410.13156)
- **What's New**: 이번 논문에서는 FAMSeC라는 새로운 AI 생성 이미지 탐지 방법을 제안합니다. 이 방법은 LoRA 기반의 Forgery Awareness Module과 Semantic feature-guided Contrastive learning strategy를 기반으로 하여 적은 샘플로도 일반화 능력을 유지하면서 효과적으로 학습합니다.

- **Technical Details**: FAMSeC는 CLIP:ViT 기능을 활용하여 AI 생성 이미지 탐지의 일반화 능력을 강화합니다. 주요 구성 요소인 Forgery Awareness Module (FAM)은 LoRA를 기반으로 하여 과적합 문제를 방지하면서도 적은 수의 샘플로 유용한 특성을 학습합니다. 또한, Semantic feature-guided Contrastive learning strategy (SeC)는 FAM의 학습을 더욱 향상시킵니다.

- **Performance Highlights**: FAMSeC는 ProGAN 데이터셋에서 4000개의 진짜 및 가짜 이미지로 훈련되었으며, 평균 분류 정확도는 95.22%에 도달했습니다. 이는 기존의 최첨단 방법에 비해 14.55% 높은 정확도를 기록하며, 단 0.56%의 훈련 샘플로 이루어졌습니다.



### Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidanc (https://arxiv.org/abs/2410.13136)
Comments:
          NeurIPS 2024. Code is available at: this https URL

- **What's New**: 본 논문에서는 Masked Generative Models (MGMs)의 성능을 개선하기 위해 일반화된 가이던스 (guidance) 공식화 및 자기 가이던스 샘플링 (self-guidance sampling) 방법을 제안합니다. 이로 인해 MGMs는 높은 품질과 다양성을 동시에 달성할 수 있게 되었습니다.

- **Technical Details**: MGMs는 [MASK] 토큰을 사용하여 입력 토큰을 점진적으로 마스킹하는 방식으로 작동합니다. 이 논문에서는 semantic smoothing을 위한 보조 작업 (auxiliary task)을 도입하여 고온 샘플링 (high-temperature sampling)을 통해 품질과 다양성을 동시에 향상시키는 방법을 설명합니다.

- **Performance Highlights**: 제안된 방법은 기존 MGMs 샘플링 방법들보다 더 효율적인 훈련 및 샘플링 비용으로 상대적으로 우수한 품질-다양성 균형을 달성하였으며, 10회의 미세 조정으로 샘플 품질을 효과적으로 개선했습니다.



### Boosting Imperceptibility of Stable Diffusion-based Adversarial Examples Generation with Momentum (https://arxiv.org/abs/2410.13122)
Comments:
          10 pages, 12 figures. To be published in IEEE TPS 2024 Proceedings. Code available on GitHub: this https URL

- **What's New**: 본 논문은 네트워크 분류기를 효과적으로 혼란시킬 수 있는 적대적 예제를 생성하기 위한 새로운 프레임워크인 Stable Diffusion 기반의 Momentum Integrated Adversarial Examples (SD-MIAE)를 제안합니다. 이 방법은 고유한 클래스 라벨에 대한 시맨틱 유사성을 유지하면서 시각적으로 인지 불가능한 적대적 예제를 생성하는 데 중점을 둡니다.

- **Technical Details**: SD-MIAE는 두 가지 단계로 구성됩니다: (1) 초기 적대적 최적화 단계에서 토큰 임베딩을 수정하여 자연스러운 이미지를 생성하고, (2) 모멘텀 기반의 최적화 단계에서 적대적 perturbation을 정제합니다. 모멘텀을 도입함으로써, 최적화 과정에서의 안정성을 높이고, 고차원 잠재 공간에서의 이미지 생성을 통해 자연스러운 외관을 유지합니다.

- **Performance Highlights**: SD-MIAE는 79%의 높은 오분류율을 달성하여 최신 기법에 비해 35% 개선된 성능을 보이며, 적대적 perturbations의 비가시성과 원래 클래스 라벨에 대한 시맨틱 유사성을 유지하는 데 기여합니다.



### Trust but Verify: Programmatic VLM Evaluation in the Wild (https://arxiv.org/abs/2410.13121)
- **What's New**: 본 논문에서는 프로그램 기반 VLM 평가(Programmatic VLM Evaluation, PROVE)라는 새로운 벤치마크 패러다임을 제안합니다. 이 방법은 비주얼 컨텐츠에 대한 오픈 엔디드 질의에 대한 VLM의 응답을 신뢰성 있게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: PROVE는 하이퍼 세부 이미지 캡션으로부터 구성된 고충실도의 씬 그래프(scene graph) 표현을 기반으로 하며, 이를 통해 다양한 질문-응답(QA) 쌍을 생성하고, 각 QA 쌍의 검증을 위한 프로그램을 함께 생성합니다. 이후, 이 프로그램을 통해 각각의 QA 쌍의 정확성과 기초를 검증하면서 10,500개의 시각적으로 기초가 있는 QA 쌍을 포함하는 데이터셋을 구축합니다.

- **Performance Highlights**: BENCHMARK한 결과, 대부분의 기존 VLM들은 유용성과 진실성 사이의 균형을 잘 맞추지 못함을 발견했습니다. 도출된 결과는 최근 '더 나은' VLM 교육의 진전이 유용성 향상으로 이어지지만 진실성 향상에는 큰 도움을 주지 않는다는 것을 보여줍니다.



### Task Consistent Prototype Learning for Incremental Few-shot Semantic Segmentation (https://arxiv.org/abs/2410.13094)
Comments:
          conference

- **What's New**: 이번 연구는 Incremental Few-Shot Semantic Segmentation (iFSS) 문제에 대한 새로운 접근 방식을 제안합니다. 이는 모델이 몇 개의 주석된 예제만으로 새로운 클래스에 대한 세분화 능력을 지속적으로 확장할 수 있도록 하는 것을 목표로 합니다.

- **Technical Details**: 연구는 메타 학습(meta-learning)을 기반으로 한 프로토타입 접근 방식을 사용하여, 기존 지식을 보존하면서 신속하게 적응할 수 있도록 모델을 유도합니다. 특히, 베이스 세션 동안의 증강 평가 프로토콜을 모방하여 가상의 증분 작업 시퀀스를 샘플링하여 메타 목표를 설정, 신속한 적응을 가능케 합니다. 프로토타입 공간 재분배 학습(Prototype Space Redistribution Learning, PSRL)을 통해 클래스 프로토타입을 동적으로 업데이트하여 최적의 프로토타입 경계를 설정합니다.

- **Performance Highlights**: PASCAL 및 COCO 벤치마크를 기반으로 한 iFSS 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 여러 경쟁 기법에 비해 월등한 성능을 보임을 확인했습니다.



### A low complexity contextual stacked ensemble-learning approach for pedestrian intent prediction (https://arxiv.org/abs/2410.13039)
- **What's New**: 이 논문은 보행자의 횡단 의도를 예측하기 위한 새로운 저복잡도의 앙상블 학습 접근법인 Contextual Stacked Ensemble-learning (CSE) 방법을 제안합니다. 이 방법은 보행자의 이미지를 스켈레톤화(skeleton-ization)하여 데이터 용량을 감소시키고, 맥락 정보(contextual information)를 추가하여 예측 성능을 향상시킵니다.

- **Technical Details**: CSE 방법은 보행자의 이미지를 17개 키포인트로 압축한 후, 스택 앙상블 학습(stacked ensemble learning)을 통해 맥락 정보를 반영합니다. 이 방법은 pedestrian intent prediction (PIP) 문제를 비디오 클립에서 보행자가 횡단하는지 여부를 예측하는 문제로 형성하였으며, 99.7%의 계산 복잡성 감소를 달성합니다.

- **Performance Highlights**: 실험 결과는 기존의 최첨단 방법과 유사한 PIP 예측 성능을 보였으며, FLOPS와 학습 가능한 매개변수에서 각각 99.99%와 99.7%의 감소를 기록했습니다. 이 연구는 IEEE Intelligent Transportation Systems Society (ITSS) 학생 대회에서 보행자 행동 예측 부문 1등 상을 수상했습니다.



### Sensitivity of Generative VLMs to Semantically and Lexically Altered Prompts (https://arxiv.org/abs/2410.13030)
- **What's New**: 본 논문은 generative vision-language 모델(VLM)의 프롬프트에서의 어휘적 및 의미적 변화에 대한 민감성을 평가합니다. SugarCrepe++ 데이터셋을 사용하여 이러한 모델들이 프롬프트의 사소한 변화에 어떤 영향을 받는지를 분석합니다.

- **Technical Details**: 이 연구는 BLIP, BakLLaVA 및 GPT-4o와 같은 generative VLMs의 어휘 및 의미 변화 이해 능력을 평가합니다. SugarCrepe++ 데이터셋에서는 두 개의 긍정적인 캡션(P1, P2)과 하나의 부정적인 캡션(N)을 포함하여, 어휘적으로 다르지만 의미적으로 유사한 캡션을 제공합니다.

- **Performance Highlights**: 실험 결과, BakLLaVA와 GPT-4o 모두 입력 프롬프트의 약간의 변화에 대해 높은 민감성을 보였으며, 동일한 프롬프트에서 옵션의 순서를 변경하는 것만으로도 성능에 큰 차이를 보였습니다. 또한, 서로 다른 VLMs 간의 일관성이 부족하여 결과의 일관성을 높이기 위한 추가 연구가 필요함을 보여줍니다.



### Geometric Trajectory Diffusion Models (https://arxiv.org/abs/2410.13027)
Comments:
          Published at NeurIPS 2024. 29 pages, 10 figures

- **What's New**: 이번 연구에서는 3D 기하학적 궤적을 모델링하기 위해 최초의 diffusion model인 GeoTDM(Geometric Trajectory Diffusion Model)을 제안합니다. 이는 기존의 정적 구조에 대응하는 방법을 넘어, 물리적 시스템이 본질적으로 동적이라는 사실을 반영합니다.

- **Technical Details**: GeoTDM은 물리적 대칭과 동역학의 시간적 상관 관계를 포함한 복잡한 공간 상호작용을 포착해야 하는 도전에 직면했습니다. 이에 SE(3)-equivariant spatial convolution과 temporal attention을 활용한 새로운 전이 커널을 개발하여 적절한 대칭을 가진 밀도를 생성합니다. 또한, 조건부 생성을 위한 표현력 있는 궤적 분포를 유도하기 위해 일반화된 학습 가능한 기하학적 사전(geometric prior)을 도입했습니다.

- **Performance Highlights**: 다양한 시나리오에서 진행한 실험 결과, GeoTDM은 물리적 시뮬레이션, 분자 동역학, 보행자 운동을 포함하여 비조건부 및 조건부 생성을 통해 사실적인 기하 궤적을 생성할 수 있으며, 품질이显著 향상됨을 보여주었습니다.



### Interpreting and Analyzing CLIP's Zero-Shot Image Classification via Mutual Knowledg (https://arxiv.org/abs/2410.13016)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 연구는 Contrastive Language-Image Pretraining (CLIP) 모델의 이미지 분류 해석 방식을 새로운 관점에서 접근합니다. 특히, 이미지와 텍스트 간의 상호 지식(mutual knowledge)을 기반으로 CLIP 모델이 공유하는 임베딩 공간에서의 유사성 및 차이를 어떻게 이해할 수 있는지를 분석합니다.

- **Technical Details**: CLIP 모델은 시각적 인코더와 텍스트 인코더로 구성되어 있으며, 두 인코더는 양의 이미지-텍스트 쌍을 임베딩 공간에서 서로 가깝게 배치하도록 훈련됩니다. 연구에서는 Mutual Information (MI) 분석을 통해 두 인코더 간의 공통 개념을 해석하고, 13개의 CLIP 모델을 사용해 구조, 크기 및 사전 훈련 데이터에 따라 분석합니다. 이 과정에서 이 논문은 비디오와 텍스트 개념을 위한 설명적 접근 방식을 제안합니다.

- **Performance Highlights**: 논문에서 제안하는 방법은 CLIP 모델의 제로샷(zero-shot) 분류 정확도를 3.75% 향상시키며, CLIP 모델이 예측할 때 어떤 공통점을 학습하는지를 시각적으로 설명할 수 있습니다. 또한, 다양한 모델의 크기, 사전 훈련 데이터 및 정확도와의 관계를 탐구할 수 있습니다.



### Explainable Binary Classification of Separable Shape Ensembles (https://arxiv.org/abs/2410.12994)
Comments:
          20 pages, 7 figures

- **What's New**: 이 논문에서는 재료 과학에서 미세 구조의 곡선 경계(curve boundaries)를 효과적으로 분석하기 위해 새로운 패턴 인식 기술을 제안합니다. 전자 후방 산란 회절(electron backscatter diffraction, EBSD) 기술을 사용하여 측정된 미세 구조의 영상 세분화(image segmentation) 데이터를 활용하여 곡선 집합의 차이를 정량화하기 위한 방법론을 개발했습니다.

- **Technical Details**: 주요 기법으로는 곡선 기능에 대한 강체 불변성(rigid-invariance) 오르토노말 분해(orthonormal decomposition)를 사용하여 곡선의 스케일 변화(scale variations)와 비선형 변형(complementary features of undulation)을 분리합니다. 분리된 형태 텐서를 이용해 두 개의 EBSD 앙상블의 형태 분포(shape distributions)를 구별하는 데 사용했습니다. 또한 제품 최대 평균 차이(maximum mean discrepancy, MMD)를 활용하여 레이블이 없는 데이터에서도 신뢰할 수 있는 분류를 지원합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 i) 통계적 거부에 민감하지 않음을 실증하고, ii) 인간의 관측과 일치하는 결론을 도출하며, iii) 인간의 눈으로는 감지할 수 없는 차이까지 검출하는 데 성공했습니다. iv) 또한 오류 측정값을 감지할 수 있는 능력을 보였습니다.



### Super-resolving Real-world Image Illumination Enhancement: A New Dataset and A Conditional Diffusion Mod (https://arxiv.org/abs/2410.12961)
Comments:
          Code and dataset at this https URL

- **What's New**: 본 논문에서는 저조도 환경에서 이미지 품질 향상을 위한 새로운 SRRIIE 데이터셋과 효율적인 conditional diffusion probabilistic model을 제안합니다. 이 데이터셋은 4800개의 저해상도-고해상도 이미지 쌍으로 구성되어 있으며 실세계 이미지를 모사한 degradation을 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: SRRIIE 데이터셋은 -6 EV에서 0 EV까지의 노출 레벨과 ISO 50에서 12800까지의 설정으로 ILDC 카메라로 캡처한 이미지를 포함합니다. 본 연구는 Raw sensor data를 이용한 conditional diffusion 모델을 도입하여 복잡한 잡음 속에서도 고해상도 구조적 세부 정보를 점진적으로 생성하며, novel time-melding condition을 통해 생성 과정의 일관성을 높입니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 기존 방법들에 비해 이미지 구조와 선명도를 효과적으로 복원하는 능력이 뛰어난 것으로 나타났습니다. SRRIIE 데이터셋을 통한 정량적 및 정성적 평가 결과, 새로운 조건을 조정한 conditional diffusion 모델이 복잡한 잡음으로부터 선명한 이미지 세부 정보를 생성할 수 있는 가능성과 효율성을 보여주었습니다.



### Gradient Map-Assisted Head and Neck Tumor Segmentation: A Pre-RT to Mid-RT Approach in MRI-Guided Radiotherapy (https://arxiv.org/abs/2410.12941)
- **What's New**: 이번 연구는 방사선 치료 (RT)에서 두경부 암의 종양 구획 정확성을 향상시키기 위해 사전 방사선 치료 이미지와 지역 그래디언트 맵을 활용한 새로운 방법을 제안합니다. 이는 기존의 수동 분할 방식의 한계를 극복하기 위한 목적으로, MRI 유도 적응 방사선 치료에 적용됩니다.

- **Technical Details**: 본 연구에서는 nnUNet 프레임워크를 사용하여 모델을 구현하고 훈련했습니다. 방사선 치료 전(pre-RT) 이미지의 변형 등록된 구획을 기반으로 종양 주변의 관심 영역 (ROIs)을 정의한 후, 이를 통해 미드 방사선 치료(mid-RT) T2 강도 이미지를 처리하여 그래디언트 맵을 생성했습니다. 이러한 방법을 통해 종양의 경계 구획을 향상시키고자 하였습니다.

- **Performance Highlights**: 본 연구의 최종 DSCagg 점수는 GTVp(주 종양)에서 0.534, GTVn(림프절 종양)에서 0.867로 나타났으며, 평균 점수는 0.70을 기록했습니다. 이는 적응 방사선 치료의 분할 및 치료 계획 강화에 기여할 잠재력을 가지고 있습니다.



### UMambaAdj: Advancing GTV Segmentation for Head and Neck Cancer in MRI-Guided RT with UMamba and nnU-Net ResEnc Planner (https://arxiv.org/abs/2410.12940)
- **What's New**: 본 연구는 MRI 유도 적응 방사선 치료에서 두 가지 최신 심층 학습 세분화 기법, UMamba와 nnU-Net Residual Encoder(ResEnc)를 통합하여 'UMambaAdj'라는 새로운 접근 방식을 제안합니다. 이 방법은 주두경부암 치료에서 종양 부피(GTV)를 더 정밀하게 세분화하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: UMambaAdj는 3D ResEnc U-Net과 Mamba 블록을 결합하여 구성됩니다. CNN 구조는 nnU-Net Residual Encoder Planner에 따라 설정되며, 6단계의 U-Net은 총 6개의 잔차 CNN 블록을 포함하여 GTVp와 GTVn의 세분화를 위해 설계되었습니다. Mamba 레이어는 입력 이미지 특징 맵을 처리하여 장기 의존성을 효과적으로 캡처합니다.

- **Performance Highlights**: MNTS-MRG 2024 챌린지 테스트 세트에서 GTVp에 대해 0.751, GTVn에 대해 0.842의 Dice 유사도 계수(DSC)를 달성했으며, 평균 DSC는 0.796였습니다. 이는 MRI 유도 적응 방사선 치료에서 보다 정밀한 종양 윤곽을 제공하며 HNC 환자의 치료 결과를 향상시킬 가능성을 보여줍니다.



### DreamCraft3D++: Efficient Hierarchical 3D Generation with Multi-Plane Reconstruction Mod (https://arxiv.org/abs/2410.12928)
Comments:
          Project Page: this https URL

- **What's New**: DreamCraft3D++는 복잡한 3D 자산을 효율적이고 고품질로 생성할 수 있는 DreamCraft3D의 확장판입니다. 이 모델은 이전의 기하학 조각 최적화 단계를 대체하는 feed-forward multi-plane 기반 재구성 모델을 도입하여 프로세스를 1000배 가속화했습니다.

- **Technical Details**: 본 연구는 DreamCraft3D의 기존 다단계 생성 프로세스를 유지하면서, geometry sculpting 단계를 feed-forward 방식의 대형 재구성 모델로 대체합니다. 또, IP-Adapter 모듈을 제안하여 다각적 이미지를 기반으로 텍스처 및 기하학 일관성을 향상시킵니다. 이를 통해 DreamCraft3D의 DreamBooth fine-tuning에 비해 4배 더 빠른 결과를 제공합니다.

- **Performance Highlights**: DreamCraft3D++는 다양한 데이터셋에 대한 실험을 통해 복잡한 기하 구조 및 사실적인 360° 텍스처를 가진 창의적인 3D 자산을 생성할 수 있는 능력을 입증하였으며, 기존의 최첨단 image-to-3D 방법론에 비해 품질과 속도에서 우수함을 보였습니다.



### DEeR: Deviation Eliminating and Noise Regulating for Privacy-preserving Federated Low-rank Adaptation (https://arxiv.org/abs/2410.12926)
- **What's New**: 본 논문에서는 저랭크 적응(low-rank adaptation, LoRA)과 연합 학습(federated learning, FL)을 통합하여 개인 정보 보호가 가능한 분산 훈련을 통해 사전 훈련된 기본 모델(pretrained foundation models)을 의료 작업에 적응하도록 하는 새로운 접근 방식을 소개합니다. 이 연구의 핵심은 LoRA와 FL의 직접 결합에서 발생하는 두 가지 문제인 집계 편차(aggregation deviation)와 차분 개인 정보 보호(differential privacy, DP) 노이즈 증폭 효과를 해결하는 것입니다.

- **Technical Details**: 새로 제안된 프레임워크인 DEeR(Deviation Eliminating and Noise Regulating)는 이론적으로 LoRA 매개변수의 동등성을 보장하여 집계 편차를 제거하는 데 필요한 조건을 증명하였습니다. 이를 바탕으로 소위 '편차 제거기(deviation eliminator)'가 설계되어 교대 최소화(alternating minimization) 알고리즘을 활용해 학습 중 집계 편차를 항상 0으로 유지하도록 LoRA의 매개변수를 반복적으로 최적화합니다. 또한, 노이즈 증폭 효과를 분석하여 DP 노이즈와 LoRA 매개변수 간의 '선형 관계'가 이 문제의 주요 원인임을 발견하고, 이를 억제하기 위해 두 가지 조절 인자를 이용하는 '노이즈 조절기(noise regulator)'를 제안하여 DP와 LoRA 간의 관계를 분리하고 강력한 개인 정보 보호와 우수한 파인튜닝 성능을 달성합니다.

- **Performance Highlights**: DEeR는 공개 의료 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보였으며, 편차 제거기와 노이즈 조절기의 유효성을 검증하기 위한 종합적인 실험이 이루어졌습니다.



### GCM-Net: Graph-enhanced Cross-Modal Infusion with a Metaheuristic-Driven Network for Video Sentiment and Emotion Analysis (https://arxiv.org/abs/2410.12828)
- **What's New**: 이 논문은 다양한 모달리티(modality)에서의 감정 분석(Emotion Recognition)과 감정 인식(Sentiment Analysis)의 복잡성을 해결하기 위해 새로운 프레임워크(GCM-Net)를 제안합니다. 본 연구는 모달리티 통합(fusion)과 특징 최적화(feature optimization)에 주안점을 두고 있습니다.

- **Technical Details**: GCM-Net은 그래프 샘플링(graph sampling)과 집계를 통해 모달리티 특징을 재조정하는 기능을 포함하고 있습니다. 또한, 교차 모달(attention) 모듈을 사용하여 모달 간 상호작용을 파악하고 발화 관련성(utterance relevance)을 결정합니다. 하모닉 최적화(harmonic optimization) 모듈은 메타휴리스틱(metaheuristic) 알고리즘을 사용하여 주목(attention)된 특징들을 결합합니다.

- **Performance Highlights**: 제안된 GCM-Net의 성능은 CMU MOSI 데이터셋에서 91.56%, CMU MOSEI 데이터셋에서 86.95%의 정확도를 보였으며, IEMOCAP 데이터셋에서의 감정 분석에서 85.66%의 정확도를 기록했습니다. 이는 기존 방법 대비 상당한 성능 향상을 나타냅니다.



### AVID: Adapting Video Diffusion Models to World Models (https://arxiv.org/abs/2410.12822)
- **What's New**: 이 연구에서는 사전 학습된 비디오 확산 모델을 액션 조건화된 월드 모델에 적응시키는 새로운 접근 방식인 AVID를 제안합니다. AVID는 액션 라벨이 붙은 비디오의 작은 도메인 특정 데이터셋에서 어댑터를 학습하여, 사전 학습된 모델의 매개변수에 접근할 수 없이 액션에 조건화된 비디오를 생성합니다.

- **Technical Details**: Diffusion 모델(확산 모델)의 노이즈 예측을 수정하여 액션 수반 예측을 생성하는 방법을 채택했습니다. AVID는 학습된 마스크를 이용하여 사전 학습된 모델의 중간 출력을 변형하고, 액션과 조건화된 비디오 출력을 생성합니다.

- **Performance Highlights**: 비디오 게임 및 실제 로봇 데이터에서 AVID를 평가하였으며, 기존의 확산 모델 적응 기준선보다 우수한 성능을 보였습니다. 사전학습된 모델을 올바르게 활용할 경우, 임베디드 AI(embodied AI)에 강력한 도구가 될 가능성을 демонстр합니다.



### Interactive Explainable Anomaly Detection for Industrial Settings (https://arxiv.org/abs/2410.12817)
- **What's New**: 이 연구는 산업 환경에서의 품질 보증을 위한 시각적 이상 탐지(Anomaly Detection)에 중점을 두고 있습니다. Convolutional Neural Networks (CNNs) 기반의 분류 모델과 블랙 박스(Classifier) 분류기를 위한 모델-비의존적(Machine-agnostic) 설명 알고리즘의 발전에 초점을 맞추고 있습니다. 이를 통해 사용자-interactive interface를 구현하여 모델의 출력을 수정할 수 있도록 돕습니다.

- **Technical Details**: 두 가지 클래스(정상, 비정상)로 분류되는 산업용 이상 탐지 데이터를 위한 InvRISE라는 새로운 설명 방법을 도입하였으며, 기존 CAIPI 알고리즘에 NearCAIPI라는 확장을 추가하였습니다. 이 알고리즘은 사용자 피드백을 적극적으로 통합하여 모델의 성능을 개선하고 설명성을 높이는 것을 목표로 합니다.

- **Performance Highlights**: 이 프레임워크를 통해 사용자 피드백을 통합한 인터랙티브한 과정이 가능해지며, 모델의 신뢰성과 사용 편의성을 증가시킬 수 있는 성과를 보였습니다. 특정 결함(예: 용접 선상의 결함) 탐지에서 사용자에 대한 추가적인 피드백이 성능 향상에 기여할 수 있음을 보여주고 있습니다.



### Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspectiv (https://arxiv.org/abs/2410.12816)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 CLIP 모델의 두 가지 정렬 문제인 작업 불일치(task misalignment) 및 데이터 불일치(data misalignment)를 해결하기 위한 방법을 제안합니다. 특히, 데이터 불일치가 다운스트림 작업에서 성능에 미치는 영향을 분석하고 Causality-Guided Semantic Decoupling and Classification (CDC) 방법론을 개발하여 이 문제를 해결합니다.

- **Technical Details**: CDC 방법론은 두 가지 주요 구성 요소인 Visual-Language Dual Semantic Decoupling (VSD)와 Decoupled Semantic Trusted Classification (DSTC)로 이루어져 있습니다. VSD는 다양한 의미를 표현하는 여러 프롬프트 템플릿을 모델에 통합하여 학습합니다. DSTC는 각 층에서 분리된 의미에 기반하여 분류 작업을 독립적으로 수행하며, 예측의 불확실성을 동시에 추정합니다.

- **Performance Highlights**: 다양한 데이터셋과 여러 작업에서 진행된 실험 결과, CDC 방법론이 CLIP의 성능을 유의미하게 향상시킴을 보여주었습니다. 특히, 새로운 클래스에 대한 인식 성능이 개선되는 효과가 있음을 확인했습니다.



### Leveraging generative models to characterize the failure conditions of image classifiers (https://arxiv.org/abs/2410.12814)
- **What's New**: 이번 연구에서는 이미지 분류기(image classifier)의 실패 조건(failure conditions)을 파악하는 문제를 다룹니다. 이를 위해, 최근의 Generative Adversarial Networks (GAN)인 StyleGAN2의 고해상도 이미지 데이터 생성 능력을 활용했습니다.

- **Technical Details**: 실행한 전략은 생성 모델(latent space)에서 성능 저하(performance degradation)의 방향을 표현하여, 여러 가지 손상(sources of corruption)이 결합된 코너 케이스(corner cases)를 발견하고 다양한 분류기들의 동작을 보다 자세히 비교하는 것입니다. 성능 저하의 방향은 데이터 생성으로 시각적으로 표현될 수 있어 해석 가능성을 높입니다.

- **Performance Highlights**: MNIST 데이터셋을 사용하여 노이즈와 블러라는 두 가지 손상 출처로 실험을 진행했으며, 이미지 품질은 모든 클래스에 영향을 미치는 반면, 형태(shape)는 특정 클래스(class-specific)에만 영향을 미친다는 것을 보여주는 결과를 도출했습니다. 이 접근 방식은 안전이 중요한 응용 프로그램에서 인공지능(AI) 컴포넌트를 활용하는 위험을 더 잘 이해하고 제어하는 데 기여할 수 있는 가능성을 제시합니다.



### Decoding Emotions: Unveiling Facial Expressions through Acoustic Sensing with Contrastive Attention (https://arxiv.org/abs/2410.12811)
Comments:
          The extended version of the 2023 IEEE INFOCOM conference paper

- **What's New**: FacER+는 기존의 외부 마이크 배열을 필요로 하지 않는 능동적 음향 얼굴 표현 인식 시스템이다.

- **Technical Details**: FacER+는 스마트폰의 earpiece 스피커에서 발산된 근초음파 신호의 반향를 분석하여 얼굴 표현 특징을 추출한다. 이 모델은 다양한 사용자 간의 표현 특징을 일관되게 학습하기 위해 대비 외부 주의(constrastive external attention) 기반 모델을 개발하였다.

- **Performance Highlights**: FacER+는 20명의 자원봉험자를 대상으로 한 실험에서 90% 이상의 정확도로 6가지 공통 얼굴 표현을 인식하여 기존의 음향 센싱 방법보다 10% 향상된 성능을 보여준다.



### Can MLLMs Understand the Deep Implication Behind Chinese Images? (https://arxiv.org/abs/2410.13854)
Comments:
          32 pages,18 figures. Project Page: this https URL Code: this https URL Dataset: this https URL

- **What's New**: 이번 연구에서는 중국 이미지의 고차원 인식 및 이해 능력을 평가하기 위한 새로운 벤치마크인 **CII-Bench**를 소개합니다. 이 벤치마크는 중국 전통 문화 이미지를 포함하여 MLLMs의 성능을 진단할 수 있는 도전적인 과제를 제공합니다.

- **Technical Details**: CII-Bench는 698개의 이미지와 800개의 다양한 선택 질문을 포함하고 있으며 여섯 가지 도메인: 생활, 예술, 사회, 정치, 환경, 중국 전통 문화로 구성되어 있습니다. 이 벤치마크는 그림, 만화, 포스터 등 다양한 이미지 유형을 활용하여 MLLMs의 이해 능력을 평가합니다.

- **Performance Highlights**: MLLMs의 정확도를 조사한 결과, 최고 정확도는 64.4%로 인간의 평균 정확도 78.2%와 비교될 때 상당한 성능 차이를 보였습니다. 특히, MLLMs는 중국 전통 문화 이미지에서 낮은 성능을 보였으며, 감정 힌트를 포함할 때 모델의 정확도가 향상되는 경향이 있었습니다.



### Retrospective Learning from Interactions (https://arxiv.org/abs/2410.13852)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)과 사용자 간의 다중 턴(multi-turn) 상호작용에서 발생하는 암묵적인 피드백 신호를 활용하여 모델 향상을 꾀하는 새로운 방법인 ReSpect를 소개합니다.

- **Technical Details**: ReSpect는 과거 상호작용에서 발생했던 암묵적 신호를 통해 학습하는 방법입니다. 이 방법은 사용자와의 상호작용 후, 모델이 자신의 과거 행동을 회고하여 피드백을 해석하고 재훈련하는 과정을 포함합니다. 이를 통해 사용자는 모델의 수행 여부를 신호로 전달하며, 이 신호는 자연어의 한정된 하위 공간에 위치하여 LLM이 이를 쉽게 감지할 수 있게 합니다.

- **Performance Highlights**: ReSpect를 적용한 결과, IDEFICS2-8B 모델의 작업 완료율이 31%에서 82%로 향상되었습니다. 이 과정에서는 외부 주석 없이도, 과거 상호작용을 통해 직접적으로 스스로 피드백을 해석하고 개선하는 능력을 보여주었습니다.



### Differentiable Robot Rendering (https://arxiv.org/abs/2410.13851)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 로봇 작업을 위한 비주얼 데이터와 액션 데이터 간의 모달리티 갭(modality gap)을 해결하기 위해 차별화 가능한 로봇 렌더링(differentiable robot rendering) 방법을 소개합니다. 이는 로봇의 시각적 형태가 제어 매개변수(control parameters)에 따라 직접적으로 미분 가능하게 해줍니다.

- **Technical Details**: 제안된 방법인 Dr. Robot은 Gaussians Splatting, 선형 혼합 스키닝(Linear Blend Skinning) 및 포즈 조건부 외형 변형 모델(pose-conditioned appearance deformation model)을 통합하여 로봇의 비주얼과 제어 신호를 연결합니다. 이를 통해 픽셀 공간에서 제어 공간으로의 신호 전송이 가능해지며, 복잡한 로봇 작업에서 에이전트가 최적화로 계획하고 제어할 수 있는 방법을 제공합니다.

- **Performance Highlights**: Dr. Robot을 사용한 여러 작업에서 로봇 포즈 복원(robot pose reconstruction)의 성능이 이전 최첨단 방법보다 큰 폭으로 개선되었음을 보여줍니다. 이 모델은 비주얼 퍼포먼스 모델과 연결하여 로봇 행동의 계획 및 제어를 수행할 수 있는 가능성을 열어줍니다.



### Unearthing Skill-Level Insights for Understanding Trade-Offs of Foundation Models (https://arxiv.org/abs/2410.13826)
Comments:
          Code at: this http URL

- **What's New**: 이 논문은 모델 평가에서의 복잡성을 해결하기 위해, 모델이 생성한 이론(rationales)을 사용하여 기저가 되는 기술(skills)을 자동으로 복구하는 방법을 제안합니다. 기존의 평가 지표에 숨겨진 다양한 기술을 분석하여, 구체적이고 행동 가능한 모델 능력 이해를 제공합니다.

- **Technical Details**: 평가 인스턴스에 대해 강력한 모델(예: GPT-4o)을 사용하여 단계별 이론을 생성하고 각 단계에서 적용된 기술을 나열합니다. 이 과정을 통해 46,000개 이상의 인스턴스를 분석하고 기술 조각(skill-slices)을 작성하여 여러 벤치마크에서 기술의 정확성을 비교합니다.

- **Performance Highlights**: 우리는 기술 조각 분석을 통해 모델 간의 성능 무역에 대한 새로운 통찰을 발견했습니다. 예를 들어, Gemini 1.5 Pro는 'molar mass 계산'에서 평균적으로 18% 더 정확하지만 '헌법법 적용'에서는 19% 덜 정확하다는 결과를 보여주었습니다. 이러한 분석 방법을 통해 우리는 전체 12개 데이터셋에서 3%의 정확도 향상을 확인했습니다.



### Eyelid Fold Consistency in Facial Modeling (https://arxiv.org/abs/2410.13760)
- **What's New**: 본 논문은 다양한 인간의 눈꺼풀 모양을 효과적으로 모델링할 수 있는 새로운 방법론을 제안합니다. 특히, 이전의 모델들이 충분히 포괄적이지 않은 눈꺼풀 모양을 다룰 수 있도록 하기 위해 새로운 일관성 정의를 도입합니다.

- **Technical Details**: 눈꺼풀 모양을 묘사하기 위해 두 개의 윗눈꺼풀과 아랫눈꺼풀로 나누는 새로운 일관성을 적용하고, 이를 통해 다양한 눈꺼풀 형태를 통일된 토폴로지를 사용하여 모델링합니다. 기존 데이터셋을 재처리하여 훈련된 파라메트릭 얼굴 모델을 개선합니다.

- **Performance Highlights**: 개선된 모델은 3D 얼굴 재구성 및 얼굴 추적과 같은 컴퓨터 비전 작업에서 성능이 크게 향상됨을 보여줍니다. 다양한 눈꺼풀 형태에 대한 정량적인 평가를 통해 모델의 정확성과 다양성을 입증합니다.



### Deep-learning recognition and tracking of individual nanotubes in low-contrast microscopy videos (https://arxiv.org/abs/2410.13594)
Comments:
          13 pages, 5 Figures, No supporting information included

- **What's New**: 이 연구는 인-시투( in-situ ) 호모다인 편광 현미경( polarization microscopy )을 사용하여 탄소 나노튜브의 성장 동역학을 분석하는 자동화된 딥 러닝( deep learning ) 접근 방식을 개발하여 새로운 기법을 제시합니다.

- **Technical Details**: Mask-RCNN 아키텍처에 ResNet-50 백본( backbone )을 적용하여 현미경 비디오에서 개별 나노튜브를 인식하고 추적하는 기술을 개발하였습니다. 이 방법은 비디오 처리 단계에서 대비를 향상시키고 신호가 약한 빠른 동역학을 효과적으로 관리하는 차별 처리 기술을 포함합니다.

- **Performance Highlights**: 딥 러닝 모델은 수작업 측정과 일관성을 보이며 데이터 처리량( throughput )을 증가시켜 나노튜브 성장에 대한 통계적 연구의 기반을 마련합니다. 이 접근법은 다른 유형의 인-시투 현미경 연구로도 적용 가능하며, 개별 나노 물체에 대한 연구를 위한 고속 데이터 수집( high-throughput data acquisition )의 중요성을 강조합니다.



### RGB to Hyperspectral: Spectral Reconstruction for Enhanced Surgical Imaging (https://arxiv.org/abs/2410.13570)
Comments:
          10 pages, 4 figures, 3 tables

- **What's New**: 이번 연구는 RGB 데이터를 이용하여 하이퍼스펙트럴 서명을 재구성하여 외과 수술 이미징을 향상시키는 방법을 다룹니다. 공개된 HeiPorSPECTRAL 데이터셋과 자체 신경외과 데이터셋을 활용하여 다양한 CNN(Convolutional Neural Networks)과 Transformer 모델의 성능을 비교하고 평가하였습니다.

- **Technical Details**: 연구에서 사용된 모델은 하이퍼스펙트럼 데이터의 정확한 예측을 위해 공간 정보를 효과적으로 통합하는 Transformer 모델입니다. 성능 평가는 RMSE(Root Mean Square Error), SAM(Spectral Angle Mapper), PSNR(Peak Signal-to-Noise Ratio) 및 SSIM(Structural Similarity Index)과 같은 포괄적인 측정을 통해 이루어졌습니다. 모델은 가시광선과 확장된 스펙트럼 범위를 모두 포함한 스펙트럼 프로파일을 예측하는 데 성공하였습니다.

- **Performance Highlights**: Transformer 모델은 평가 지표에서 우수한 성능을 보이며, 질적 평가를 통해 외과적 의사결정에 중요한 스펙트럴 프로파일 예측 능력을 나타냈습니다. 그러나 가시광선과 확장된 하이퍼스펙트럴 범위를 모두 캡처하는 데 있어 MAE(Mean Absolute Error)를 통해 강조된 복잡한 과제가 있었습니다.



### Representing Model Weights with Language using Tree Experts (https://arxiv.org/abs/2410.13569)
- **What's New**: 본 논문은 다양한 신경망 모델의 가중치를 입력으로 사용하는 메타네트워크를 학습하는 새로운 접근 방식을 제안합니다. 대중 모델의 대부분이 소수의 Model Trees에 속하며, 이는 학습을 용이하게 합니다.

- **Technical Details**: 모델 가중치의 변화량을 줄이기 위해 Probing Experts (ProbeX)라는 경량 탐색 방법을 도입합니다. ProbeX는 단일 레이어의 가중치만을 학습하며, 고차원 모델 가중치를 효율적으로 매핑하는 것을 목표로 합니다.

- **Performance Highlights**: ProbeX는 제로샷 모델 분류 및 검색을 포함하여 모델 가중치를 공통의 가중치-언어 임베딩 공간으로 효과적으로 매핑하며, 상대적으로 적은 훈련 시간으로 다양한 과제에서 뛰어난 일반화를 보여줍니다.



### GeoCoder: Solving Geometry Problems by Generating Modular Code through Vision-Language Models (https://arxiv.org/abs/2410.13510)
- **What's New**: 본 논문에서는 기하 문제 해결을 위해 비전-언어 모델(Vision-Language Models, VLMs)을 향상시키기 위한 새로운 접근 방식인 GeoCoder를 제안합니다. 이 모델은 사전 정의된 기하 함수 라이브러리를 활용하여 코드를 생성 및 실행함으로써 수학적 연산의 정확성을 높입니다.

- **Technical Details**: GeoCoder는 모듈식 코드 파인튜닝(modular code finetuning)을 통해 기하 문제 해결을 위한 코드를 생성하고 실행합니다. 이 과정에서 사용되는 함수 라이브러리는 수식을 올바르게 적용함으로써 오류를 최소화하며, RAG-GeoCoder라는 비파라메트릭 메모리(non-parametric memory) 모듈을 채택하여 함수의 검색 기능을 향상시킵니다.

- **Performance Highlights**: GeoCoder와 RAG-GeoCoder는 GeomVerse 데이터셋에서 다양한 문제 복잡도에 대해 평균 16% 향상된 성능을 보여주었습니다. 이러한 성과는 전통적인 파인튜닝 방법과 비교하여 기하 학적 추론 능력을 획기적으로 증가시켰습니다.



### Similarity-Dissimilarity Loss with Supervised Contrastive Learning for Multi-label Classification (https://arxiv.org/abs/2410.13439)
- **What's New**: 본 연구는 멀티 라벨 분류에서의 슈퍼바이저드 대조 학습(Supervised Contrastive Learning)에서 긍정 샘플을 결정하는 데 있어 새로운 접근 방식을 제안합니다. 특히, 다섯 가지 고유한 관계를 도입하고, 유사성 및 비유사성 손실(Similarity-Dissimilarity Loss)을 통해 대조 손실 함수(weights)를 동적으로 조정합니다.

- **Technical Details**: 다섯 가지 관계(R2, R3, R4, R5)를 정의하여 멀티 라벨 샘플과 앵커(anchor) 사이의 유사성과 비유사성을 계산하여 손실을 재가중화하는 새로운 Similarity-Dissimilarity Loss를 제안합니다. 이를 통해 ALL, ANY 및 MulSupCon 등의 기존 방법의 한계를 극복합니다.

- **Performance Highlights**: MIMIC 데이터셋에서 멀티 라벨 텍스트 분류 실험을 수행한 결과, 제안된 손실 함수가 슈퍼바이저드 대조 학습 패러다임 하에서 모든 인코더(encoders)에 대해 성능을 효과적으로 향상시키는 것으로 나타났습니다. 실험 결과는 제안된 방법의 효과성과 견고성을 뒷받침합니다.



### Unsupervised Skull Segmentation via Contrastive MR-to-CT Modality Translation (https://arxiv.org/abs/2410.13427)
Comments:
          16 pages, 5 figures, ACCV 2024 - GAISynMeD Workshop

- **What's New**: 본 연구에서는 MR(자기공명영상)에서 두개골(segmentation)분할 문제를 해결하기 위한 새로운 비지도(un) 접근 방식을 제안합니다. 이를 통해 MR 이미지를 직접적으로 분할하는 대신, MR-CT(컴퓨터단층촬영) 변환을 통해 합성 CT 데이터를 생성하고, 여기서 분할을 수행하는 방법을 탐구합니다.

- **Technical Details**: 제안된 파이프라인은 두 가지 주요 모듈을 포함합니다: Contrastive Unpaired Translation (CUT)과 Laplacian Pyramid Super-Resolution Network (LapSRN)입니다. CUT 모듈은 비어치된(un) 데이터 샘플을 활용하여 MR에서 CT로의 변환을 수행하며, LapSRN은 고해상도 CT 이미지를 생성하여 점진적으로 해상도를 향상시키는 역할을 합니다. 이 과정에서 다양한 데이터 전처리 기법을 적용합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존 두개골 제거(skull stripping) 방법 및 기타 의학적 분할 모델인 MedSAM과 비교되었습니다. 가장 중요한 점은, 우리의 모델이 훈련 과정에서 어떠한 주석(annotations)도 필요로 하지 않으며, 이는 다양한 임상 상황에서의 두개골 분할 작업에 매우 유용할 것으로 기대됩니다.



### Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding (https://arxiv.org/abs/2410.13321)
- **What's New**: 이번 연구에서는 LVLMs에서 나타나는 언어 priors의 문제를 해결하기 위해 새로운 방법인 Summary-Guided Decoding (SGD)를 제안합니다. SGD는 이미지 정보에 더 집중하도록 모델을 유도하며, 텍스트 품질을 유지합니다.

- **Technical Details**: 연구는 LVLMs에서의 언어 priors를 분석하고, 이미지 관련 부분의 품사(POS)와 연관된 토큰을 생성할 때 언어 priors에 대한 모델의 의존도가 증가함을 발견했습니다. SGD는 요약(context) 기법을 활용하여 이미지 관련 POS 토큰의 다음-토큰 확률을 수정하여 텍스트 품질을 최대한 보존하면서 이미지 정보를 반영합니다.

- **Performance Highlights**: SGD는 객체 환각(object hallucination) 벤치마크에서 모든 다른 해석 방법을 초월했으며(CHAIRS에서 +16.5%, CHAIRI에서 +19% 향상), 정밀도와 재현율의 균형을 잘 유지하며 Pareto optimal성을 달성했습니다. 또한 텍스트 품질을 거의 완벽하게 유지하면서 객체 환각을 줄이는 데 강력한 성과를 보였습니다.



### Precipitation Nowcasting Using Diffusion Transformer with Causal Attention (https://arxiv.org/abs/2410.13314)
- **What's New**: 이번 연구에서는 Diffusion Transformer with Causal Attention 모델을 제안하여 단기 강수 예보의 문제를 해결하고자 합니다. 이 모델은 Transformer를 활용하여, 조건 정보와 예보 결과 간의 시공간 쿼리를 효과적으로 설정할 수 있도록 합니다.

- **Technical Details**: DTCA(분산 변환기 인과 주의) 모델은 조건부 강수 분포 특징 관련 쿼리를 기반으로 한 새로운 인과 주의 메커니즘을 도입하며, 다양한 시공간 정보 상호작용을 탐색하고 그 구조를 비교합니다. 실험 결과, 전역 시공간 레이블링 상호작용이 최고의 성능을 발휘하는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 방법은 최신 기술인 U-Net 기반 방법과 비교해 강수 예측에서 약 15% 및 8% 개선된 CSI(비판적 성공 지표)를 달성하여 현재 상태의최고 성능을 기록했습니다.



### Reference-Based Post-OCR Processing with LLM for Diacritic Languages (https://arxiv.org/abs/2410.13305)
- **What's New**: 이 연구는 역사적 문서에서 OCR(Optical Character Recognition)로 생성된 부정확한 텍스트를 수정하는 새로운 방법을 제안합니다. 이는 사용 가능한 전자책(e-books)을 참조하여 텍스트를 교정하고 대형 언어 모델(LLM)을 사용하여 고정밀의 유사 페이지 간 레이블을 생성하는 방식입니다.

- **Technical Details**: 이 방법은 먼저 고유한 노이즈를 제거한 후, 콘텐츠 중심의 전자책을 참조 자료로 활용하여 LLM을 통해 시맨틱과 비주얼 유사성을 기반으로 OCR로 생성된 결함 있는 텍스트를 수정합니다. 마지막으로, 짧은 길이의 빈 텍스트는 LLM 기반의 철자 교정으로 조정하여 최종 유사 레이블을 제공합니다.

- **Performance Highlights**: 시험 결과, 이 연구에서 생성한 데이터셋은 10점 만점에 평균 8.72점을 기록하며 최신 Transformer 기반의 철자 교정 모델인 7.03점을 초과했습니다. 본 연구는 또한 19세기 고전 책을 위한 최초의 대규모 공개 데이터셋인 VieBookRead를 출시하여 향후 연구를 지원할 계획입니다.



### PiLocNet: Physics-informed neural network on 3D localization with rotating point spread function (https://arxiv.org/abs/2410.13295)
Comments:
          25 pages, 4 figures

- **What's New**: 이 논문은 3D localization 문제 해결을 위한 새로운 개선된 Neural Network PiLocNet을 제안합니다. PiLocNet은 기존의 LocNet을 기반으로 하며, forward-model 기반 정보와 data-fitting loss term을 통합하여 물리적으로 합리적인 결과를 도출합니다.

- **Technical Details**: PiLocNet은 Physics-Informed Neural Network (PINN)으로, 이미징 시스템의 포인트 스프레드 함수(PSF)를 통해 물리적 정보를 네트워크에 통합합니다. 세 가지 중요한 구성 요소로는 forward-model 정보, variational method의 regularization term, 그리고 Poisson 및 Gaussian noise 모델을 통한 이미지 노이즈에 대한 강건성 개선이 포함됩니다.

- **Performance Highlights**: 실험 결과, PiLocNet은 3D 포인트 소스의 localization을 위한 정확도 향상에 기여하며, precision과 recall 측면에서 개선된 예측 결과를 보여줍니다. 본 논문은 PiLocNet의 Robustness를 검증했으며, 다양한 PSF와 이미징 문제에서의 적용 가능성을 제시합니다.



### Golyadkin's Torment: Doppelg\"angers and Adversarial Vulnerability (https://arxiv.org/abs/2410.13193)
- **What's New**: 이 논문은 'Adversarial Doppelgangers(AD)'라는 개념을 정의하고 탐구하며, 이는 기존의 adversarial visual metamers를 포함합니다. AD의 성능 및 강건성을 분류 기계와 인간의 성능을 비교하여 분석합니다.

- **Technical Details**: AD는 이 논문에서 정의된 지각적(metric) 측정에 따라 서로 가까운 입력들입니다. 연구에서는 이러한 AD에 대한 분류기의 취약성을 분석하고, AD에 강건한 분류기의 구조와 속성을 설명하며, 개념적 엔트로피(conceptual entropy) 및 개념적 모호성(regions of conceptual ambiguity)에 대한 개념을 도입합니다.

- **Performance Highlights**: 대부분의 분류기는 AD에 취약하며, 강건성-정확도 트레이드오프(robustness-accuracy trade-offs)가 개선되지 않을 수 있습니다. 그러나 정확도가 높은 모든 분류기는 hypersensitive behavior를 보여줄 있으며, 이로 인해 AD 강건성을 개선하는 것이 정확도 개선과 동일함을 발견했습니다.



### Scalable Drift Monitoring in Medical Imaging AI (https://arxiv.org/abs/2410.13174)
- **What's New**: 이 논문에서는 인공지능(AI)을 의료 이미징에 통합하여 임상 진단의 발전을 이루었지만 모델 드리프트 관리와 장기적인 신뢰성을 보장하는 데 몇 가지 도전 과제가 발생한다는 점을 강조하고 있습니다. MMC+라는 확장된 프레임워크를 개발하여 이러한 문제에 대처하고 있습니다.

- **Technical Details**: MMC+는 CheXstray 프레임워크를 기반으로 하여 다중 모달 데이터 일치성을 이용한 실시간 드리프트 감지를 통해 의료 이미지 AI 모델을 위한 확장 가능한 드리프트 모니터링 솔루션을 제안합니다. 이 프레임워크는 다양한 데이터 스트림을 보다 강력하게 처리하고, MedImageInsight와 같은 기초 모델을 통합하여 고차원 이미지 임베딩을 지원하며, 불확실성 경계를 도입하여 동적 임상 환경에서 드리프트를 보다 잘 포착합니다.

- **Performance Highlights**: MMC+는 COVID-19 팬데믹 기간 동안 Massachusetts General Hospital의 실제 데이터를 통해 검증되었으며, 데이터의 중요한 변화 감지와 이를 모델 성능 변화와 연계하는 데 효과적입니다. 이러한 시스템은 성능 저하를 직접적으로 예측하지는 않지만, AI 시스템이 허용 가능한 성능 범위에서 이탈할 가능성을 조기에 경고하여 신속한 개입이 가능하도록 합니다.



### Utilizing Large Language Models in An Iterative Paradigm with Domain Feedback for Molecule Optimization (https://arxiv.org/abs/2410.13147)
- **What's New**: 본 연구에서는 약물 발견에서 분자의 최적화를 지원하기 위해 LLM (Large Language Models)을 효과적으로 활용할 수 있는 도메인 피드백 제공자인 Re²DF를 제안합니다. 이 새로운 접근법은 분자가 화학적으로 유효하지 않을 경우를 고려하여 수정된 분자의 유효성을 즉시 검증하며, 해당 분자의 개선을 위한 구체적인 피드백을 제공합니다.

- **Technical Details**: Re²DF는 외부 툴킷인 RDKit를 이용하여 수정된 분자가 화학적으로 유효한지를 체크합니다. 만약 유효하지 않다면, RDKit로부터 오류 메시지를 제공받아 LLM에게 수정 방향을 제시합니다. 또한, 수정된 분자가 원하는 특성을 충족하는지 확인하여, 목표에 대한 정확한 방향과 거리를 제공하는 신뢰할 수 있는 피드백을 생성합니다.

- **Performance Highlights**: Re²DF는 단일 속성 목표 20개에서 Hit ratio를 각각 16.95% 및 20.76% 향상시키며, 다중 속성 목표에서는 32개에서 각각 6.04% 및 5.25% 향상시켰습니다. 이러한 결과는 Re²DF가 기존 방법들보다 더 나은 성능을 발휘함을 알립니다.



### Mapping Bias in Vision Language Models: Signposts, Pitfalls, and the Road Ahead (https://arxiv.org/abs/2410.13146)
Comments:
          Under Review at NAACL 2025

- **What's New**: 이 논문은 Vision Language Models (VLMs)의 공정성을 평가하기 위해 5개의 모델과 6개의 데이터셋을 분석하고, 편향(bias)에 대한 새로운 통찰을 제공합니다. 특히, 기존의 표정 기반(portrait-based) 데이터셋이 VLM의 공정성을 평가하는 데 가장 유용하다는 것을 발견했습니다.

- **Technical Details**: 본 연구는 UTKFace, CelebA, PATA, VLStereoSet, VisoGender와 같은 여러 데이터셋을 활용하여 VLM의 편향을 평가하였습니다. 각 모델이 성별, 인종 및 연령과 같은 보호 속성을 어떻게 처리하는지 분석를 진행하며, 특히 데이터셋의 구성에 따라 평가 결과가 달라질 수 있음을 강조합니다. VisoGender 데이터셋의 어려운 버전을 소개하여 철저한 평가를 가능하게 합니다. 

- **Performance Highlights**: LLaVa와 CLIP 모델 간의 성능과 공정성의 격차를 발견하여, VLM의 공정성 평가에 대한 보다 효과적이고 체계적인 데이터셋 설계의 필요성을 강조합니다. 저자들은 기존 데이터셋의 한계를 지적하며, VLM의 평가를 위한 향후 연구 방향을 제안합니다.



### See Behind Walls in Real-time Using Aerial Drones and Augmented Reality (https://arxiv.org/abs/2410.13139)
Comments:
          6 pages

- **What's New**: 새로운 ARD2 프레임워크는 두 대의 항공 드론과 증강 현실(augmented reality, AR) 장치를 활용하여 실시간으로 벽을 통과하는 감시를 가능하게 합니다.

- **Technical Details**: ARD2는 두 가지 주요 단계로 구성되며, 첫 번째 단계에서는 드론, 사용자 및 목표물 간의 기하학적 관계를 이용해 목표물의 방향을 사용자 AR 디스플레이에 투사합니다. 두 번째 단계에서는 드론에서 촬영한 이미지를 합성하여 목표물의 외형(contour)을 재구성합니다.

- **Performance Highlights**: 실험 결과, 방향 추정과 외형 재구성 모두에서 시스템의 정확도가 입증되었습니다.



### Adversarial Neural Networks in Medical Imaging Advancements and Challenges in Semantic Segmentation (https://arxiv.org/abs/2410.13099)
- **What's New**: 최근 인공지능(AI)의 발전으로 의료 영상 특히 뇌 영상 분야가 혁신적인 변화를 겪고 있습니다. 이 연구는 AI의 주요 분야인 deep learning을 뇌 이미지의 semantic segmentation에 통합하는 방안을 체계적으로 조사합니다.

- **Technical Details**: Semantic segmentation은 해부학적 구조를 구분하고 병리학적 지표를 식별하는 필수 기법이며, 이는 복잡한 신경 질환의 진단을 위해 필수적입니다. 연구에서는 adversarial neural networks를 적용하여 semantic segmentation 프로세스를 자동화하고 개선하는 방법을 제시합니다.

- **Performance Highlights**: 이 접근 방식은 진단 정확도를 향상시키고 인간의 오류를 줄이며 이미징 데이터 분석의 처리량을 증가시키는 데 기여합니다. 이를 통해 신경학적 평가에서 진단 정확도가 획기적으로 개선되었습니다.



### MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models (https://arxiv.org/abs/2410.13085)
- **What's New**: 본 논문에서는 Med-LVLMs의 사실성을 향상시키기 위해 MMed-RAG라는 다중 모달 RAG 시스템을 제안합니다. 이 시스템은 도메인 인식을 위한 검색 메커니즘, 적응형 검색된 컨텍스트 선택 방법, 그리고 검증 가능한 RAG 기반의 선호 미세 조정 전략을 포함하여, 의료 데이터의 다양한 분야에 대해 일반적이고 신뢰할 수 있는 접근 방식을 제공합니다.

- **Technical Details**: MMed-RAG는 세 가지 주요 요소로 구성됩니다: 1) 도메인 인식 검색 메커니즘 - 입력 의료 이미지에 적합한 검색 모델을 선택하기 위해 도메인 식별 모듈을 설계하였습니다. 2) 적응형 검색된 컨텍스트 선택 - 검색된 컨텍스트의 개수를 선택하는 방법입니다. 3) RAG 기반 선호 미세 조정 - 교차 모달 정렬을 개선하고 모델과 실제 간의 전체 정렬을 높이는 방법입니다.

- **Performance Highlights**: MMed-RAG는 5개의 의료 데이터세트에서 실험을 실시하여, Medical VQA와 보고서 생성 작업에서 각각 18.5% 및 69.1%의 사실 정확도를 향상시켰습니다. 전반적으로 MMed-RAG는 Med-LVLMs의 정확성을 평균 43.8% 개선하였습니다.



### BOXR: Body and head motion Optimization framework for eXtended Reality (https://arxiv.org/abs/2410.13084)
Comments:
          Accepted to 45th IEEE Real-Time Systems Symposium (RTSS'24)

- **What's New**: 새로운 C2D(latency) 메트릭을 소개하여 본체 움직임에 의해 발생하는 지연을 캡처하고 XR 시스템 내에서 본체 및 머리 움직임 지연을 공동 최적화하는 BOXR 프레임워크를 제안합니다.

- **Technical Details**: BOXR 프레임워크는 M2D(Head Motion)와 C2D(Body Motion) 지연을 효율적으로 조정하여 과제 간의 자원 경쟁을 피하고 출력 프레임에서 최신 자세를 유지합니다. 또한, 사용자 동적을 반영하기 위한 움직임 기반의 비주얼 관성 측정기와 장면 의존형의 포베이드 렌더링을 통합합니다.

- **Performance Highlights**: BOXR은 11개의 EuRoC MAV 데이터 세트에서 4개의 XR 애플리케이션을 통해 3개의 하드웨어 플랫폼에서 최신 솔루션보다 성능이 크게 향상되었습니다. M2D와 C2D 지연을 각각 최대 63%와 27%까지 줄이고, 프레임 비율을 최대 43%까지 증가시키며, 실제 환경에서 M2D 지연은 최대 42%, C2D 지연은 최대 31% 감소시켰습니다.



### UniCoN: Universal Conditional Networks for Multi-Age Embryonic Cartilage Segmentation with Sparsely Annotated Data (https://arxiv.org/abs/2410.13043)
- **What's New**: 본 연구에서는 초음파 미세 CT (micro-CT) 이미지를 이용한 배아 (embryonic) 연골 (cartilage) 세분화 (segmentation)에서의 어려움을 극복하기 위해 새로운 딥러닝 (Deep Learning, DL) 접근 방식을 제안합니다. 이 방법은 나이와 공간 정보 (spatial information)를 효과적으로 활용하여 모델 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 본 논문에서는 CNNs, Transformers 및 하이브리드 모델과 같은 다양한 DL 아키텍처에서 적용할 수 있는 두 가지 새로운 메커니즘을 제안합니다. 첫 번째 메커니즘은 이산 (discrete) 나이 범주에 조건화되어 연골의 나이에 따른 형태 변화를 정확히 표현하고, 두 번째 메커니즘은 연속 (continuous) 이미지 크롭 위치에 따라 조정되어 두개부 (cranial region) 내의 세부 형태를 잘 표현합니다.

- **Performance Highlights**: 다양한 연령대의 연골 세분화 데이터셋에서 우리의 조건부 모듈을 인기 있는 DL 세분화 아키텍처에 통합했을 때, 평균 1.7%의 Dice 점수를 얻었으며, 보이지 않는 데이터에서 7.5%의 성능 향상을 이루었습니다. 이러한 결과는 제한된 주석 데이터 (annotated data)로 다양한 데이터셋을 처리할 수 있는 강력하고 보편적인 모델을 개발할 수 있는 가능성을 강조합니다.



### Synthesis and Perceptual Scaling of High Resolution Natural Images Using Stable Diffusion (https://arxiv.org/abs/2410.13034)
Comments:
          29 pages, 7 Figures, 5 tables

- **What's New**: 본 연구에서는 Stable Diffusion XL을 사용하여 자연 이미지의 연속적인 변화를 가진 맞춤형 자극 세트를 합성했다. 이는 각 이미지가 동일 범주 내에서 독특하게 해석 가능하도록 하여 심리물리학적 실험에 적합한 자극 세트를 제공한다.

- **Technical Details**: 자극 이미지는 여섯 개 범주에서 각각 18개의 객체로 구성되며 각 객체에 대해 10개의 변형이 생성되었다. 이러한 변형은 지각적 연속성을 기반으로 하여 배열되어 있으며, 전통적인 이미지 합성 방법으로는 구현하기 어려운 고해상도의 자연 이미지를 생성한다.

- **Performance Highlights**: 1113명의 참가자를 대상으로 한 온라인 유사성 판단 작업을 통해 생성된 자극 세트의 지각적 변이를 검증하였다. 결과는 개발된 이미지를 사용하여 시각적 인식, 주의력 및 기억 연구에 유용한 자료라는 것을 입증하였다.



### Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images (https://arxiv.org/abs/2410.13010)
Comments:
          Published in the 3rd Workshop on New Frontiers in Adversarial Machine Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

- **What's New**: 기존의 적대적 공격이 주로 단일 모드에 초점을 맞추었던 반면, 본 연구에서는 CLIP과 같은 대규모 멀티 모달 모델(LMM)이 가지는 새로운 취약점에 주목합니다. 새로운 ‘Hiding-in-Plain-Sight (HiPS)’ 공격 기법을 통해 모델 예측을 미세하게 수정함으로써, 타겟 객체가 존재하지 않는 것처럼 보이게 하는 방법을 제안합니다.

- **Technical Details**: HiPS 공격은 두 가지 변형으로 소개됩니다: HiPS-cls는 클래스 레이블 정보를 활용하여 공격을 생성하며, HiPS-cap은 원본 이미지 캡션과 타겟 캡션을 사용하여 공격을 설계합니다. 이러한 공격 기법은 CLIP-Cap과 같은 이미지 캡셔닝 모델로 효과적으로 전이될 수 있습니다.

- **Performance Highlights**: HiPS 공격은 타겟 객체가 이미지 캡션에서 효과적으로 제거되도록 설계되었으며, 여러 평가 지표를 통해 성능을 검증합니다. 제안된 공격이 하위 모델에서 어떻게 작동하는지를 보여주며, 적대적 공격의 새로운 기준을 설정합니다.



### Configurable Embodied Data Generation for Class-Agnostic RGB-D Video Segmentation (https://arxiv.org/abs/2410.12995)
Comments:
          Accepted in IEEE Robotics and Automation Letters October 2024

- **What's New**: 이 논문은 로봇의 다양한 형태를 반영하여 클래스-비구속(video instance segmentation) 비디오 분할(Video Segmentation) 성능을 개선할 수 있는 대규모 데이터셋 생성을 위한 새로운 방법을 제시합니다.

- **Technical Details**: 3D 재구성을 활용하여 로봇의 형태적 특성과 환경의 조명, 센서 배치 등을 반영한 구성이 가능한 세분화된 비디오 세트를 생성하는 파이프라인을 개발하였습니다. 이 과정에서 Massive Video Panoptic dataset (MVPd)을 도입하며, 이는 기존 비디오 세분화 벤치마크보다 45배 이상 크며, 18K 개의 주석 달린 RGB-D 비디오를 포함합니다.

- **Performance Highlights**: MVPd를 통해 파인튜닝(fine-tuning) 시 특정 로봇의 형태에 따른 분할 성능 향상을 보여주었으며, 3D 모달리티(깊이 이미지 및 카메라 자세)를 처리할 때 비디오 세분화 정확성과 일관성을 개선할 수 있는 가능성을 입증했습니다.



### Risk Assessment for Autonomous Landing in Urban Environments using Semantic Segmentation (https://arxiv.org/abs/2410.12988)
- **What's New**: 이 논문에서는 복잡한 도시 환경에서 비전 기반 자율 착륙 문제를 다루고 있으며, 이를 위해 깊은 신경망(deep neural networks)을 사용하여 의미 분할(semantic segmentation)과 위험 평가(risk assessment)를 수행합니다.

- **Technical Details**: SegFormer라는 최첨단 비주얼 트랜스포머 네트워크를 사용하여 복잡하고 비구조적인 도시 환경에서의 의미 분할을 수행합니다. 이 접근 방식은 UAV(Unmanned Aerial Vehicle)의 RGB 카메라 이미지에서 실시간으로 세그먼트를 측정하여 도시 환경에서 가장 흔한 클래스(class)로 구분합니다.

- **Performance Highlights**: 제안된 전략은 다양한 사례 연구(case studies)를 통해 검증되었으며, 자율 긴급 착륙을 위한 가장 안전한 착륙 지역을 결정하는 데 있어 의미 분할 기반 전략의 잠재력을 보여줍니다. 이 연구는 UAV의 민간 응용 프로그램에서의 가능성을 극대화하는 데 기여할 것입니다.



### MuVi: Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization (https://arxiv.org/abs/2410.12957)
Comments:
          Working in progress

- **What's New**: MuVi라는 혁신적인 새로운 프레임워크를 통해 비디오의 시각적 내용에 맞는 음악을 생성하는 과제를 효과적으로 해결합니다.

- **Technical Details**: MuVi는 비디오 콘텐츠를 분석하기 위해 특별하게 설계된 시각적 어댑터(visual adaptor)를 사용하여 맥락적 및 시간적으로 관련된 특징을 추출합니다. 이 특징은 비디오의 분위기와 주제뿐만 아니라 리듬과 페이스에 맞는 음악을 생성하는 데 사용됩니다. 또한, 음악 구문(phrase)의 주기성을 반영하여 동기화를 보장하는 대조적 음악-비주얼 사전 훈련 방식(contrastive music-visual pre-training scheme)을 도입했습니다.

- **Performance Highlights**: MuVi는 오디오 품질과 시간 동기화 측면에서 뛰어난 성능을 보여줍니다. 실험 결과, MuVi는 기반 모델들보다 우수한 성능을 보여주었으며, 비디오의 시맨틱 정렬(semantic alignment)과 리듬 동기화(rhythmic synchronization)에서 두드러진 성과를 이루었습니다.



### Long-Tailed Backdoor Attack Using Dynamic Data Augmentation Operations (https://arxiv.org/abs/2410.12955)
- **What's New**: 본 논문은 긴 꼬리(long-tailed) 데이터셋에 대한 백도어 공격(backdoor attack)을 처음으로 탐구합니다. 기존의 백도어 공격은 주로 균형 잡힌 데이터셋에 초점을 맞추었으며, 이로 인해 실제 환경에서 발생하는 불균형 데이터 문제를 간과하고 있었습니다.

- **Technical Details**: 제안된 방법인 D$^2$AO(Dynamic Data Augmentation Operation)는 클래스, 샘플 유형(클린 vs. 백도어), 그리고 샘플 특징에 따라 동적으로 다양하고 적절한 데이터 증강(data augmentation) 연산을 선택합니다. 이를 통해 백도어 샘플과 클린 샘플의 불균형 문제를 해결하고, 데이터 증강에 적응할 수 있는 트리거 생성기를 개발하였습니다.

- **Performance Highlights**: CIFAR10-LT 및 CIFAR100-LT와 같은 두 개의 긴 꼬리 벤치마크에서 폭넓은 실험을 수행하였으며, 제안된 방법은 기존의 백도어 공격 방법과 비교하여 상태-of-the-art 공격 성능을 달성하면서 클린 정확도(clean accuracy)를 유지하였습니다.



### Syn2Real Domain Generalization for Underwater Mine-like Object Detection Using Side-Scan Sonar (https://arxiv.org/abs/2410.12953)
Comments:
          7 pages, 4 figures and 3 tables

- **What's New**: 논문에서는 수중 지뢰 탐지에 대한 Syn2Real (Synthetic to Real) 도메인 일반화 접근 방식을 제안합니다. 이 방법은 DDPM 및 DDIM 모델을 사용하여 생성한 합성 데이터를 통해 실제 환경 샘플을 효과적으로 보강할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 딥러닝 기반의 자동 목표 인식(ATR) 기술을 사용하여 수중 지뢰를 탐지하는 과정을 다룹니다. 특히, 이 논문에서는 DCGAN 및 확산 모델과 같은 합성 데이터 생성 모델을 비교 분석하였으며, 이러한 모델의 하이퍼파라미터 튜닝을 통해 효과적인 결과를 얻었습니다.

- **Performance Highlights**: Mask-RCNN 모델을 합성 데이터와 원본 데이터 조합으로 학습시킨 결과, 평균 정밀도(Average Precision, AP)가 약 60% 증가했습니다. 이는 수중 지뢰 탐지 작업에서 Syn2Real 도메인 일반화의 잠재력을 강조하는 결과입니다.



### Answering Questions in Stages: Prompt Chaining for Contract QA (https://arxiv.org/abs/2410.12840)
- **What's New**: 이번 연구에서는 법률 문서에서의 질문에 대한 구조적 답변 생성을 위한 새로운 두 단계 프롬프트 체인을 제안합니다. 이전의 프롬프트가 긴 조항을 다루는 데 한계를 보였던 반면, 이 방식은 더 복잡한 법률 텍스트를 효과적으로 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 법률 관련 질문에 대한 응답을 두 단계로 처리하는 전략을 사용하는데, 첫 번째 단계에서는 관련 법률 텍스트의 요약을 생성하고, 두 번째 단계에서는 이 요약을 사용하여 기존의 프롬프트 템플릿에 대해 질문에 대한 답변을 형성합니다. 이를 통해 질문과 답변 옵션 간의 매핑을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 두 단계 프롬프트 체인이 단순한 프롬프트에 비해 대부분의 경우 더 효과적임을 보여주었습니다. 이는 법률 전문가들이 문서를 더 효율적으로 검토하고 자동화된 워크플로우 및 데이터 파이프라인을 구축할 수 있도록 도와주는 기회를 제공합니다.



### EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing (https://arxiv.org/abs/2410.12836)
- **What's New**: EditRoom은 자연어 명령을 통해 다양한 레이아웃 편집을 자동으로 수행할 수 있는 통합 프레임워크로, 수동 개입 없이 실행됩니다.

- **Technical Details**: EditRoom은 두 개의 주요 모듈인 Command Parameterizer와 Scene Editor로 구성되어 있습니다. Command Parameterizer는 사전 훈련된 LLM(GPT-4o)을 활용하여 자연어 명령을 여섯 가지 기본 편집 유형에 대한 분해 명령으로 변환합니다. Scene Editor는 소스 장면과 텍스트 명령을 조건으로 삼아 확산 기반(diffusion-based) 모델을 훈련하여 목표 장면을 생성합니다.

- **Performance Highlights**: 편집 작업에 대한 실험 결과, EditRoom은 모든 메트릭에서 다른 기준선보다 우수한 성능을 보였으며, 다중 작업 명령에 대해서도 일반화할 수 있는 능력을 보여줍니다.



### MyData: A Comprehensive Database of Mycetoma Tissue Microscopic Images for Histopathological Analysis (https://arxiv.org/abs/2410.12833)
- **What's New**: 이번 논문에서는 mycetoma(미세포증) 자동 검출 및 분류를 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 mycetoma 조직의 미세탐색적 이미지로 구성되어 있으며, 이는 진단 정확도를 높이고 환자 결과를 개선하는 데 활용될 수 있습니다.

- **Technical Details**: 제안된 데이터셋(MyData)은 142명의 환자에서 수집된 총 864개의 미세탐색적 이미지로 구성되어 있습니다. 각 이미지는 그레인(grain) 존재를 나타내는 이진 마스크(binary masks)로 주석이 달려 있어 검출(detection) 및 분할(segmentation) 작업을 용이하게 합니다. 이 데이터베이스는 특히 전문가 부족으로 어려움을 겪는 농촌 지역에서 유용하게 활용될 수 있습니다.

- **Performance Highlights**: 히스토로파물학적(Histopathological) 접근 방식은 mycetoma 진단에 효과적이며 비용 효율적입니다. 스마트 의료 이미지 분석 모델의 발전과 더불어, 본 연구는 AI 기반 진단 도구의 개발을 지원하고, 질병의 원인 병원체를 효과적으로 식별하는 데 기여할 것으로 기대됩니다.



### Segment as You Wish -- Free-Form Language-Based Segmentation for Medical Images (https://arxiv.org/abs/2410.12831)
- **What's New**: 이번 논문에서는 기존의 바운딩 박스나 포인트 기반의 프롬프트 대신 자연어 기반의 프롬프트를 활용하여 의료 이미지 분할(Medical Image Segmentation, MIS) 문제를 해결하는 새로운 접근 방식을 제안합니다. 이를 위해 RAG(임시 증강 생성) 기술을 이용한 자유형 텍스트 프롬프트 생성기를 개발하고, 다양한 텍스트 프롬프트를 처리할 수 있는 새 모델인 FLanS를 소개합니다.

- **Technical Details**: FLanS는 전문 해부학 기반 쿼리, 해부학 무관 위치 기반 쿼리, 해부학 무관 크기 기반 쿼리를 포함한 다양한 자유형 텍스트 프롬프트를 처리할 수 있는 모델입니다. 또한, 대칭 인지 캐노니컬화 모듈을 통해 스캔 방향에 따른 일관된 정확한 분할을 보장하며, 100,000개 이상의 의료 이미지로 훈련되었습니다.

- **Performance Highlights**: FLanS는 최근의 SOTA(State-of-the-Art) 모델들보다 우수한 언어 이해 능력과 분할 정밀도를 보여주었으며, 다양한 임상 환경에서의 응용 가능성을 입증했습니다. 논문에서는 자질 분석(ablation studies)을 통해 각 구성 요소의 기여도를 검증했습니다.



### DyMix: Dynamic Frequency Mixup Scheduler based Unsupervised Domain Adaptation for Enhancing Alzheimer's Disease Identification (https://arxiv.org/abs/2410.12827)
Comments:
          10 pages, 5 figures, 3 tables

- **What's New**: 이번 연구에서는 Alzheimer’s disease (AD) 진단의 정확성을 높이기 위해 고안된 동적 주파수 믹스업 스케줄러(DyMix)를 제안하였습니다. 이 접근 방식은 Unsupervised Domain Adaptation (UDA)의 성능을 향상시키는데 초점을 맞추고 있습니다.

- **Technical Details**: DyMix는 주파수 영역에서의 동적 조정을 통해 원본(source) 도메인과 대상(target) 도메인 간의 변동성을 처리합니다. 이 방법은 두 가지 주요 단계로 구성됩니다: (i) 의미 불변의 특징 표현을 학습하기 위한 사전 학습(pretraining) 단계, (ii) 동적 주파수 조작을 통한 도메인 적응(domain adaptation) 단계.

- **Performance Highlights**: 실험 결과 DyMix는 Alzheimer’s Disease Neuroimaging Initiative (ADNI)와 Australian Imaging Biomarkers and Lifestyle Study of Aging (AIBL) 데이터세트에서 기존의 최첨단 방법들에 비해 뛰어난 성능을 보였습니다. 이는 동적 주파수 조정을 통해 AD 진단의 정확성을 높이는 데 기여하였습니다.



### Deep Adversarial Learning with Activity-Based User Discrimination Task for Human Activity Recognition (https://arxiv.org/abs/2410.12819)
- **What's New**: 본 연구는 인체 활동 인식(Human Activity Recognition, HAR) 문제를 위해 새로운 적대적 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 사람 간의 변동성을 해결하기 위한 새로운 활동 기반의 구분 작업을 통합하며, 이는 사람들이 동일한 활동을 수행하는 방식이 다름을 인정합니다.

- **Technical Details**: 저자들은 다중 작업, 적대적 학습(adversarial learning) 및 자기 지도 학습(self-supervised learning)에 기반한 표현 학습 방법을 활용하여 모델의 일반화 성능을 향상시키고 개인 정보 유출 문제를 완화합니다. 제안된 프레임워크는 이진 분류 작업으로, 동일한 사람과 동일한 활동에서의 활동 특성 벡터 쌍이 동일한지 구분하는 것을 목적으로 합니다.

- **Performance Highlights**: 제안한 프레임워크는 Leave-One-Person-Out Cross-Validation (LOOCV) 설정에서 새로운 보이지 않는 개인들에 대한 성능을 측정하며, 기존 접근 방식보다 분류 성능을 개선하는 결과를 보여주었습니다. 또한 훈련 및 테스트 참가자 간의 동일한 활동에서의 사람 간 변동성 격차를 감소시켰습니다.



### ChatVTG: Video Temporal Grounding via Chat with Video Dialogue Large Language Models (https://arxiv.org/abs/2410.12813)
Comments:
          10 pages, 3 figures

- **What's New**: ChatVTG는 Video Dialogue Large Language Models (LLMs)를 활용하여 제로샷(zero-shot) 비디오 템포럴 그라운딩(Video Temporal Grounding, VTG) 접근 방식을 제안합니다. 기존의 방법들과 달리 추가적인 훈련 데이터나 쌍으로 연결된 주석 데이터 없이도 작동합니다.

- **Technical Details**: ChatVTG는 비디오를 여러 개의 거친(코스, coarse) 세그먼트로 분할한 후, 각 세그먼트에 대한 캡션을 생성하고 사용자가 제공한 쿼리(query)와 매칭하여 코스 타이밍을 파악합니다. 추가적으로, 슬라이딩 윈도우(sliding window) 방식을 통해 순간 제안을 생성하여 정밀한 타이밍 경계(refinement)를 제공합니다.

- **Performance Highlights**: ChatVTG는 Charades-STA, ActivityNet-Captions, TACoS 등 세 가지 주류 VTG 데이터셋에서 실험을 진행하였으며, 기존의 제로샷 방법보다 향상된 성능을 보였습니다.



### Order-aware Interactive Segmentation (https://arxiv.org/abs/2410.12214)
Comments:
          Interactive demo can be found in project page: this https URL

- **What's New**: 본 논문에서는 사용자 상호작용을 최소화하면서도 정확한 객체 분할을 위한 새로운 방법인 OIS (order-aware interactive segmentation)를 제안합니다. OIS는 객체 간의 상대 깊이를 인코딩하여 사용자 상호작용을 효과적으로 안내하고, 이를 통해 이전의 방법들에 비해 성능을 크게 향상시킵니다.

- **Technical Details**: OIS는 목표 객체의 상대 깊이를 표현하는 order maps를 활용하여 상호작용을 개선합니다. 독창적인 order-aware attention과 object-aware attention 모듈을 도입하여 유사한 깊이를 가진 객체들을 효과적으로 구별할 수 있게 합니다. 또한, 사용자 클릭이 최적화된 이미지 특징에 통합될 수 있도록 조화롭게 설계되어 있습니다.

- **Performance Highlights**: OIS는 HQSeg44K 데이터셋에서 클릭 한 번으로 mIoU가 7.61 증가하였고, DAVIS 데이터셋에서는 1.32의 향상률을 보여줍니다. 뿐만 아니라, SegNext와 비교하여 추론 속도를 2배 향상시킨 것을 입증했습니다.



New uploads on arXiv(cs.AI)

### AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents (https://arxiv.org/abs/2410.13825)
- **What's New**: 이 연구는 LLM(대형 언어 모델)을 기반으로 한 웹 에이전트를 개선하기 위한 혁신적인 접근 방식을 제안합니다. 구체적으로, 에이전트의 관찰(object) 및 행동(action) 공간을 정제하여 LLM의 능력에 더욱 잘 부합하도록 합니다.

- **Technical Details**: 제안된 방법은 세 가지 구성 요소로 이루어져 있습니다: 1) 필수적이지 않은 행동을 줄여 에이전트의 기능을 단순화; 2) 중복되거나 불필요한 웹 요소를 제거하여 관찰을 개선; 3) 'branch' 및 'prune'와 같은 두 가지 계획 행동을 도입하여 에이전트의 내비게이션 흐름을 자기 조직화 합니다.

- **Performance Highlights**: AgentOccam는 WebArena 벤치마크에서 기존의 최첨단 방법보다 9.8 포인트 (+29.4%) 향상된 성능을 보이고, 유사한 일반 웹 에이전트에 비해 성공률을 26.6 포인트 (+161%) 증가시켰습니다. 이 모든 것을 추가적인 맥락 예제, 온라인 피드백 또는 검색 전략 없이 달성했습니다.



### A Pattern to Align Them All: Integrating Different Modalities to Define Multi-Modal Entities (https://arxiv.org/abs/2410.13803)
Comments:
          20 pages, 6 figures

- **What's New**: 이 논문에서는 엔티티(개체)와 그 정보의 의미를 다양한 매체를 통해 표현할 수 있는 개념적 패턴을 제안합니다. Multi-Modal Knowledge Graphs(MMKGs)는 텍스트, 이미지, 오디오, 비디오 등 다양한 모달리티(modality)를 통한 정보의 통합을 촉진하기 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 정보 엔티티(IE)와 그 정보의 물리적 실현(Information Realisation, IR) 간의 개념적 구분을 도입하여 다중 모달리티 이해를 위한 온톨로지 설계 패턴을 제안합니다. 이를 통해 다양한 형식으로 구현되는 정보 리소스 간의 관계를 명확히 하며, MMKG의 다양한 기존 온톨로지와의 조화를 위한 기초를 마련합니다.

- **Performance Highlights**: 제안된 패턴은 현재의 MMKG 리소스에 적용 가능성이 높으며, 기존의 온톨로지와의 조화를 이끌어내어 지능형 응용 프로그램이 요구하는 국소적 및 맥락적 정보 표현을 강화합니다.



### Transformer Guided Coevolution: Improved Team Formation in Multiagent Adversarial Games (https://arxiv.org/abs/2410.13769)
- **What's New**: BERTeam은 다중 에이전트 적대적 게임에서 최적의 팀을 형성하기 위한 혁신적인 알고리즘으로, 변환기 기반의 심층 신경망을 사용하여 에이전트의 조합을 선택합니다.

- **Technical Details**: BERTeam은 Masked Language Model 훈련 방법을 이용하여 팀 구성원을 예측하며, coevolutionary deep reinforcement learning을 통해 다양한 개인 에이전트를 진화시킵니다. 이를 통해 팀의 성능을 극대화하는 팀 선택 과정을 간소화합니다.

- **Performance Highlights**: Marine Capture-The-Flag 게임에서 BERTeam은 MCAA와 같은 기존 알고리즘을 초월하여 성능을 발휘하며, 비정형 팀 구성 방법을 학습하여 보지 못한 상대에 대해서도 잘 대응합니다.



### MixEval-X: Any-to-Any Evaluations from Real-World Data Mixtures (https://arxiv.org/abs/2410.13754)
- **What's New**: 이 논문은 다양한 형태의 입력 및 출력을 지원하는 새로운 벤치마크인 MixEval-X를 소개하여, AI 모델의 평가 방식을 최적화하고 표준화하는 데 중점을 두고 있습니다. 이를 통해 실제 작업 배포에 맞는 평가가 가능하게 됩니다.

- **Technical Details**: MixEval-X는 any-to-any (모든 입력에 대해 가능한 모든 출력) 형식의 벤치마크로, 다양한 modality (양식) 간의 평가 일관성을 높이기 위해 multi-modal (다중 양식) 벤치마크 혼합 및 adaptation-rectification (적응-정정) 파이프라인을 제안합니다. 이 방법은 평가가 실제 사용할 수 있는 사례에 잘 일반화되도록 합니다.

- **Performance Highlights**: 종합적인 메타 평가 결과, MixEval-X는 벤치마크 샘플과 실제 작업 배포 간의 효과적인 정렬을 보여주었으며, 모델 순위는 크라우드 소싱된 실제 평가와 강한 상관관계를 나타냅니다 (상관 계수 0.98까지). 또한, 기존 모델 및 조직을 재순위화할 수 있는 포괄적인 리더보드를 제공하여 다중 양식 평가에 대한 이해를 높이고 향후 연구에 대한 통찰을 제공합니다.



### Disjointness Violations in Wikidata (https://arxiv.org/abs/2410.13707)
Comments:
          Sixth International Knowledge Graph and Semantic Web Conference

- **What's New**: 이 논문은 Wikidata에서의 불일치 체크(disjointness checks)의 현재 모델링을 분석하고, 이를 통해 발생하는 불일치 위반(disjointness violations)의 패턴과 원인을 확인하였습니다. SPARQL 쿼리를 사용해 각각의 원인을 규명하고, 서로 충돌하는 정보를 식별 및 수정할 수 있는 공식을 제시합니다.

- **Technical Details**: Wikidata는 1억 개 이상의 객체를 포함하는 대규모 지식 그래프입니다. 본 논문에서는 RDF(리소스 기술 프레임워크)를 사용하여 Wikidata에서 쌍별 불일치 클래스(pairwise disjoint classes)의 정보를 수집하였습니다. SPARQL 쿼리를 작성하여 불일치 유니온 문장(disjoint union statements)의 쌍을 찾아내었습니다.

- **Performance Highlights**: 논문에서 제안한 방식은 불일치 상황을 정량화하고, 성능을 개선하여 사용자가 문제를 식별하고 수정하는 효율성을 높이는 데 기여할 수 있습니다. 총 758개의 불일치 유니온 문장이 631개 클래스에서 생성되었으며, 7,027개의 쌍별 불일치 문장(pairwise disjoint statements)이 도출되었습니다.



### MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling (https://arxiv.org/abs/2410.13610)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)와 의료 분야의 복잡한 문제를 해결하기 위한 새로운 프레임워크인 MeNTi를 소개합니다.

- **Technical Details**: MeNTi는 LLMs를 위한 보편적인 에이전트 아키텍처로, 전문화된 의료 도구 세트를 통합하고 메타 도구(meta-tool) 및 중첩 호출(nested calling) 메커니즘을 사용하여 LLM 도구 활용을 강화합니다. 이를 통해 유연한 도구 선택 및 중첩 도구 호출이 가능해지며, 계산기 선택, 슬롯 채우기(slot filling), 단위 변환을 포함한 복잡한 의료 시나리오 문제를 해결합니다.

- **Performance Highlights**: CalcQA라는 벤치마크를 통해 LLM의 정량적 평가 능력을 검증하며, 100개의 사례-계산기 쌍과 281개의 의료 도구가 포함된 도구 키트를 통해 실험 결과에서 상당한 성능 개선을 보여주었습니다.



### Instruction-Driven Game Engine: A Poker Case Study (https://arxiv.org/abs/2410.13441)
Comments:
          EMNLP 2024 Demo. arXiv admin note: substantial text overlap with arXiv:2404.00276

- **What's New**: 이드제(Instruction-Driven Game Engine, IDGE) 프로젝트는 사용자들이 자연어 지시를 통해 게임을 쉽게 생성할 수 있도록 하여 게임 개발의 진입 장벽을 낮추는 것을 목표로 하고 있습니다. 기존의 게임 엔진들이 프로그래밍 언어에 의해 구동되는 것과는 달리, IDGE는 사용자와의 상호작용을 통해 게임 상태를 동적으로 생성합니다.

- **Technical Details**: IDGEs는 사용자 지정 게임 스크립트에 따라 게임 상태를 예측하는 Next State Prediction(다음 상태 예측) 작업에 기반하여 설계되었습니다. 이는 아키텍처에 대화형 LLMs(large language models)를 포함하여 사용자 입력을 반영하여 실시간 게임 정보가 포함된 게임 상태를 생성합니다. 모델은 커리큘럼 학습 방식을 통해 안정성과 다양성을 동시에 고려하여 훈련됩니다.

- **Performance Highlights**: 초기 연구 결과로, IDGE는 포커(Poker) 게임을 위한 새로운 엔진으로 발전하였으며, 이는 다양한 포커 변형을 지원하며, 사용자 입력을 통해 한층 개인화된 포커 게임을 생성하는 데 성공했습니다. IDGE는 새로운 카드 조합과 전투 전략을 처리하는 데에도 뛰어난 일반화 능력을 보입니다.



### Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding (https://arxiv.org/abs/2410.13321)
- **What's New**: 이번 연구에서는 LVLMs에서 나타나는 언어 priors의 문제를 해결하기 위해 새로운 방법인 Summary-Guided Decoding (SGD)를 제안합니다. SGD는 이미지 정보에 더 집중하도록 모델을 유도하며, 텍스트 품질을 유지합니다.

- **Technical Details**: 연구는 LVLMs에서의 언어 priors를 분석하고, 이미지 관련 부분의 품사(POS)와 연관된 토큰을 생성할 때 언어 priors에 대한 모델의 의존도가 증가함을 발견했습니다. SGD는 요약(context) 기법을 활용하여 이미지 관련 POS 토큰의 다음-토큰 확률을 수정하여 텍스트 품질을 최대한 보존하면서 이미지 정보를 반영합니다.

- **Performance Highlights**: SGD는 객체 환각(object hallucination) 벤치마크에서 모든 다른 해석 방법을 초월했으며(CHAIRS에서 +16.5%, CHAIRI에서 +19% 향상), 정밀도와 재현율의 균형을 잘 유지하며 Pareto optimal성을 달성했습니다. 또한 텍스트 품질을 거의 완벽하게 유지하면서 객체 환각을 줄이는 데 강력한 성과를 보였습니다.



### A Simplifying and Learnable Graph Convolutional Attention Network for Unsupervised Knowledge Graphs Alignmen (https://arxiv.org/abs/2410.13263)
Comments:
          14 pages, 3 figures

- **What's New**: 최근의 연구에서는 Entity Alignment (EA) 작업의 성공이 레이블이 붙은 데이터에서 제공되는 감독 정보에 크게 의존하고 있음을 강조합니다. 그러나 레이블된 데이터의 비용을 고려할 때 이러한 방법의 실용성은 제한적입니다. 따라서, 본 논문에서는 Unsupervised Knowledge Graphs alignment를 위한 Simplifying and Learnable graph convolutional attention network (SLU)를 제안합니다.

- **Technical Details**: SLU는 LCAT라는 새로운 그래프 신경망(GNN)을 백본 네트워크로 사용하여 Knowledge Graph (KG)의 그래프 구조를 모델링합니다. SLU는 잠재적 매칭 관계를 바탕으로 관계 구조를 재구성하는 방법을 설계하여 잘못된 이웃 정보를 필터링하고, 유사성을 측정하기 위해 일관성 기반의 유사성 함수를 제안합니다.

- **Performance Highlights**: SLU는 세 가지 데이터셋 (15K 및 100K)에서 광범위한 실험을 수행한 결과, 25개의 감독 또는 비감독 방법들을 초월하여 정렬 정확도를 유의미하게 향상시켰습니다. 가장 좋은 경우에서 Hits@1 점수가 6.4% 향상되었습니다.



### Research on Travel Route Planing Problems Based on Greedy Algorithm (https://arxiv.org/abs/2410.13226)
- **What's New**: 이 연구에서는 초기 경로 탐색 및 관광객의 개인화 요구를 충족하는 최적화된 경로 계획 알고리즘이 제안되었습니다. 특히, PCA(Principal Component Analysis) 및 KMO(Kaiser-Meyer-Olkin) 테스트와 TOPSIS(TOPSIS: Technique for Order Preference by Similarity to Ideal Solution) 기법을 통해 도시 평가 지표의 차원 축소 및 종합 평가를 수행했습니다.

- **Technical Details**: 연구에서는 PCA를 사용하여 도시 평가 지표의 차원을 축소하고, KMO 테스트를 통해 데이터 적합성을 판단하고, TOPSIS 및 엔트로피 가중치 방법을 통해 데이터를 종합 평가했습니다. 경로 최적화를 위해서는 그리디 알고리즘이 사용되었으며, 관광 명소 방문에 소요되는 시간을 고려한 경로 계획이 이루어졌습니다.

- **Performance Highlights**: 이 알고리즘은 352개의 도시에서 100개의 관광 명소 데이터를 활용하여 관광객에게 최적화된 여행 경로를 제공함으로써 여행 비용을 줄이고 현지 최적해(local optimum) 문제를 피하는 데 기여합니다. 결과적으로 관광객의 요구에 맞춘 맞춤형 경로 계획을 통해 효율적인 여행 경험을 지원합니다.



### Anchored Alignment for Self-Explanations Enhancemen (https://arxiv.org/abs/2410.13216)
- **What's New**: 본 연구에서는 언어 모델의 자기 설명(self-explanation) 능력을 향상시키기 위해 주석이 달린 이유 설명이 없는 경우에도 이들의 사고 내용을 명확히 서술하는 방식으로 모델 정렬(alignment) 방법론을 제안합니다.

- **Technical Details**: 본 방법론은 설명 품질 평가(explanation quality assessment), 자기 지시 데이터셋 생성(self-instruction dataset generation), 모델 정렬(model alignment)이라는 세 가지 핵심 요소로 구성됩니다. 특히, 'Anchor Preference Pairs'라는 새로운 기술을 도입하여 모델 출력을 일관되게 정확한 것, 일관되게 부정확한 것, 가변적인 것으로 세 가지 범주로 분류하여 선호 쌍(preference pairs) 선택을 개선합니다. 이를 통해 Direct Preference Optimization (DPO) 전략의 효과성을 증대시킵니다.

- **Performance Highlights**: 실험 결과, 본 접근법은 다른 fine-tuning 전략과 비교할 때 설명 품질을 유의미하게 개선하면서도 정확성을 유지하는 것으로 나타났습니다. 특히, Anchor Preference Pairs를 활용한 방법론이 Judge 기반 평가에만 의존한 자기 정렬 전략보다 더욱 우수한 성능을 보이는 것을 입증했습니다.



### LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch (https://arxiv.org/abs/2410.13213)
- **What's New**: LLMOPT라는 통합 학습 기반 프레임워크를 제안하여 최적화 문제의 일반화 능력을 향상시켰습니다. 이 프레임워크는 자연어 설명으로부터 최적화 문제를 정의하고 해결하는 과정을 자동화하는 데 중점을 두고 있습니다.

- **Technical Details**: LLMOPT는 다섯 가지 요소로 구성된 포뮬레이션을 통해 다양한 최적화 문제 유형을 정의하고, 다중 지침 튜닝(multi-instruction tuning) 및 모델 정렬(model alignment)으로 정확성과 일반성을 향상시킵니다. 또한 자동 테스트(auto-testing)와 자기 수정(self-correction) 메커니즘을 통해 hallucinations를 방지합니다.

- **Performance Highlights**: LLMOPT는 20개 분야에서 6개의 실제 데이터셋을 대상으로 평가된 결과, 선형/비선형 프로그래밍, 혼합 정수 프로그래밍 및 조합 최적화와 같은 다양한 최적화 문제를 처리하며 최신 방법보다 평균 11.08%의 해결 정확도 향상을 달성했습니다.



### Context-Enhanced Multi-View Trajectory Representation Learning: Bridging the Gap through Self-Supervised Models (https://arxiv.org/abs/2410.13196)
- **What's New**: MVTraj는 다중 시각의 맥락을 통합하여 경로 표현 학습을 향상시키는 새로운 방법을 제안합니다. GPS, 도로 네트워크 및 POI(관심 지점)의 다양한 맥락적 지식을 활용하여 경로 데이터에 대한 보다 포괄적인 이해를 제공합니다.

- **Technical Details**: MVTraj는 GPS 경로를 연결 고리로 사용하고 셀프 슈퍼바이즈드(자기지도학습) 프리텍스트(사전학습) 작업을 통해 다중 시각 간 학습 프로세스를 정렬합니다. 3개의 다양한 시각(예: GPS, 도로 경로 및 그리드)의 경로를 다루는데, 각 시각에서 독립적인 모달리티로 간주하고 계층적 크로스 모달 상호작용 모듈을 적용하여 지식을 융합합니다.

- **Performance Highlights**: 실제 데이터셋을 활용한 폭넓은 실험 결과, MVTraj는 다양한 공간 시각과 관련된 작업에서 기존의 기준선 모델에 비해 현저한 성능 향상을 보여줍니다.



### Chain of Ideas: Revolutionizing Research in Novel Idea Development with LLM Agents (https://arxiv.org/abs/2410.13185)
Comments:
          10 pages,5 figures, conference

- **What's New**: 이 논문은 Chain-of-Ideas (CoI) 에이전트를 통해 대형 언어 모델(LLMs)이 연구 아이디어 생성의 효율성을 개선할 수 있는 새로운 방안을 제안합니다. CoI 에이전트는 관련 문헌을 체계적으로 정리하여 연구 분야의 발전을 잘 반영하도록 돕습니다.

- **Technical Details**: CoI 에이전트는 (1) CoI 구성, (2) 아이디어 생성, (3) 실험 설계의 세 가지 단계로 구성됩니다. 각 단계에서 LLM은 연구 분야의 다양한 트렌드를 반영하여 복수의 CoIs를 구축하고, 각 CoI에 대해 예측 및 아이디어를 체계적으로 발전시키는 과정을 거칩니다.

- **Performance Highlights**: 실험 결과에 따르면 CoI 에이전트는 여러 자동화된 방법보다 항상 높은 성능을 보였으며, 사람의 연구 아이디어 생성 품질과도 비교 가능한 결과를 나타냈습니다. CoI 에이전트는 아이디어 생성에서 56 ELO 점수 차이로 두 번째 방법을 초월했습니다.



### Language Models as Semiotic Machines: Reconceptualizing AI Language Systems through Structuralist and Post-Structuralist Theories of Languag (https://arxiv.org/abs/2410.13065)
Comments:
          18 pages, 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 인간의 인지 과정을 모방하는 것으로 보지 않고, 기호학적 기계(semiotic machines)로 재구성하여 이해하는 새로운 프레임워크를 제안합니다. 저자는 페르디낭 드 소쉬르(Ferdinand de Saussure)와 자크 데리다(Jacques Derrida)의 언어 이론에 기초하여 LLM을 언어 자체의 모델로 설명하고 있습니다.

- **Technical Details**: 논문은 세 부분으로 나뉘어 있으며, 첫 번째 부분에서는 word2vec 임베딩 알고리즘의 작동 방식과 소쉬르의 기호 체계에 대한 설명을 제공합니다. 두 번째 부분에서는 데리다의 비판을 적용하여 LLM이 모델링하는 '쓰기'의 개념을 논의합니다. 마지막 세 번째 부분에서는 현대 LLM이 의미의 고정되지 않은 개념을 어떻게 반영하는지에 대해 설명하며, '다음 토큰 생성' 메커니즘이 의미의 역동성을 포착한다고 주장합니다.

- **Performance Highlights**: 대형 언어 모델은 언어 사용에서 거의 인간의 수준에 도달하며, word2vec 알고리즘을 기반으로 하여 컨텍스트 기반의 의미 표현을 채택하고 있습니다. 이러한 모델은 개별 단어뿐만 아니라 문장 및 다른 언어 구조를 포함한 복잡한 표현을 생성하려고 하며, 현재 사용되는 데이터셋은 방대한 양의 정보를 포함하고 있어, LLM이 언어 자체에 근접한 모델링을 가능하게 합니다.



### Optimal Transport for Probabilistic Circuits (https://arxiv.org/abs/2410.13061)
- **What's New**: 이 논문에서는 확률 회로(Probabilistic Circuits, PCs) 간의 Wasserstein distance를 계산하기 위한 최적 운송 프레임워크를 도입합니다. 기존에 알려진 알고리즘은 있었으나, 확률 회로로 정의된 분포 간의 Wasserstein distance를 계산하는 방법은 처음으로 제안됩니다.

- **Technical Details**: Wasserstein-type distance를 도입하여 연관된 최적 운송 문제의 coupling measure를 확률 회로로 제한합니다. 이 거리를 계산하기 위해 작은 선형 프로그램(solution to linear programming problems) 문제를 해결하는 알고리즘을 개발하였으며, 이 문제의 해결조건을 제시합니다. 또한, 실험적 데이터를 기반으로 한 PC와의 Wasserstein distance를 최소화하기 위한 효율적인 반복 알고리즘을 제공합니다.

- **Performance Highlights**: 제안된 알고리즘은 두 확률 회로 간의 Wasserstein-type distance를 정확하고 효율적으로 계산할 수 있는 기능을 갖추고 있으며, 이를 기존 방법론과 비교하여 실험적으로 우수한 성능을 보입니다.



### Hypothesis Testing the Circuit Hypothesis in LLMs (https://arxiv.org/abs/2410.13032)
Comments:
          Code available here: this https URL

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 내부 작동 방식을 이해하기 위해 '회로(circuits)'라는 개념을 실험적으로 조사했습니다. 특히 회로가 LLM의 능력을 구현하는지를 테스트하기 위한 기준과 가설 검정 방식이 소개되었습니다.

- **Technical Details**: 연구에서는 다음 세 가지 이상적인 속성을 정의했습니다: 1) 메커니즘 보존(Mechanism Preservation): 회로의 성능이 원래 모델과 일치해야 함. 2) 메커니즘 지역화(Mechanism Localization): 회로를 제거하면 해당 작업을 수행하는 능력이 사라져야 함. 3) 최소성(Minimality): 회로에 중복된 엣지가 없어야 함. 이 속성에 따라 6개의 회로를 평가했습니다.

- **Performance Highlights**: 합성 회로(synthetic circuits)는 이상적인 속성과 잘 일치하는 반면, 발견된 회로는 모든 속성에 엄격히 부합하지 않았습니다. 그러나 특정 발견된 회로는 유명한 작업을 수행하는 데 중요한 역할을 했으며, 이들 회로는 이상적인 특성에 근접하게 개선될 수 있는 가능성을 보여주었습니다.



### Learning Representations for Reasoning: Generalizing Across Diverse Structures (https://arxiv.org/abs/2410.13018)
Comments:
          PhD thesis

- **What's New**: 이 논문은 인공지능 분야에서의 추론의 중요성과 관련하여, 기존의 지식 구조 및 쿼리 구조를 초월하는 일반화 알고리즘을 제안합니다. 또한, 구조적 데이터에서 기계 학습 개발을 가속화하기 위한 시스템을 구축했습니다.

- **Technical Details**: 제안된 모델 NBFNet은 전통적인 경로 기반(path-based) 방법과 동적 프로그래밍(dynamic programming)을 결합하여 새로운 엔티티(entity) 및 관계(relation) 어휘를 사용한 지식 그래프의 미지의 부분에 대한 유도 일반화를 실현합니다. A*Net은 NBFNet의 확장형으로, 수백만 개 규모의 지식 그래프에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: NBFNet은 기존의 최신 방법들에 비해 모든 설정에서 평균 18%의 성능 향상을 이루었으며, 특히 지식 그래프 완성(HITS@1) 및 유도 관계 예측(HITS@10)에서 각각 22%의 성능 개선을 보여줍니다.



### Large Language Models as a Tool for Mining Object Knowledg (https://arxiv.org/abs/2410.12959)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 일반 물체에 대한 명시적 지식을 공식화하는 능력을 조사하며, 물체의 구성 요소(부분) 및 재질에 대한 지식을 명확히 구분합니다. 이로 인해 LLMs의 잠재력을 이용하여 AI 시스템의 지식 기반을 보강하거나 대체하는 데 기여할 수 있습니다.

- **Technical Details**: 이 연구에서는 few-shot과 zero-shot multi-step 프롬프트 기법을 활용하여 약 2,300개의 물체 및 하위 유형에 대한 부품과 재질에 대한 데이터를 수집합니다. LLM의 언어 이해 능력을 통해 물체의 전체 구성과 부품의 재질에 대한 지식을 명확히 정리합니다.

- **Performance Highlights**: 평가 결과, 추출된 지식의 대부분이 인간의 이해와 일치하나, 프롬프트 기법에 따라 과도하게 단순화되거나 필요 이상의 세부 정보가 제공되는 경우도 있음을 보여줍니다. 이 연구는 물체 구조 및 구성에 대한 추론을 위한 유용한 자원으로서 기능할 것입니다.



### MIND: Math Informed syNthetic Dialogues for Pretraining LLMs (https://arxiv.org/abs/2410.12881)
Comments:
          31 pages, 5 figures, 14 tables

- **What's New**: 이번 연구에서는 대규모 다채로운 Math Informed syNthetic Dialogue (MIND) 생성 방법을 제안하여 대형 언어 모델(LLMs)의 수학적 추론 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: MIND를 활용하여 OpenWebMath (OWM)를 기반으로 합성 대화를 생성하고, 이를 통해 새로운 수학 데이터셋인 MIND-OWM을 만듭니다. 실험 결과, 대화 참여자 간의 지식 격차를 포함하는 것이 고품질 수학 데이터를 생성하는 데 필수적임을 보여줍니다. 또한, 합성 데이터와 원본 데이터를 사전 학습(pretraining) 시 효과적으로 포맷하고 통합하는 방법을 식별하였습니다.

- **Performance Highlights**: MIND-OWM에서 사전 학습된 모델은 원본 데이터만으로 사전 학습된 모델 대비 수학적 추론에서 상당한 향상을 보였습니다 (GSM8K: +13.42%, MATH: +2.30%). 또한, 전문 지식(MMLU: +4.55%, MMLU-STEM: +4.28%) 및 일반적인 추론 과제(GENERAL REASONING: +2.51%)에서도 우수한 성능을 기록했습니다.



### IMAS: A Comprehensive Agentic Approach to Rural Healthcare Delivery (https://arxiv.org/abs/2410.12868)
- **What's New**: COVID-19 이후, 농촌 지역의 의료 접근성 문제 해결을 위한 첨단 의료 보조 시스템(IMAS) 제안

- **Technical Details**: IMAS는 Large Language Models (LLMs)와 다섯 가지 주요 구성 요소(번역, 의료 복잡성 평가, 전문가 네트워크 통합, 최종 의료 조언 생성, 응답 단순화)로 구성되어 있습니다.

- **Performance Highlights**: IMAS는 MedQA, PubMedQA, JAMA 데이터셋을 통해 효과성을 입증하였으며, 특히 저소득 및 정보 소외 지역사회의 의료 근로자들에게 더 쉽게 접근할 수 있도록 지원합니다.



### Interpretable Rule-Based System for Radar-Based Gesture Sensing: Enhancing Transparency and Personalization in AI (https://arxiv.org/abs/2410.12806)
Comments:
          accepted at the 21st European Radar Conference, 4 pages, 2 figure

- **What's New**: 본 연구에서는 레이더 기반 제스처 감지를 위한 투명하고 해석 가능한 다중 클래스 규칙 기반 알고리즘인 MIRA를 소개합니다. AI의 이해 가능성이 중요한 분야에서 MIRA는 사용자의 신뢰를 높이기 위해 의사 결정 프로세스에 대한 통찰력을 제공합니다.

- **Technical Details**: MIRA는 개인 맞춤형 규칙 세트를 통해 개별 사용자 행동에 조정되며, 사용자 중심의 AI 경험을 제공합니다. 이 연구에서는 새로운 다중 클래스 분류 아키텍처를 제시하고, 방대한 주파수 변조 연속파 레이더 제스처 데이터 세트를 공유하며, 시스템의 뛰어난 해석 가능성을 입증하는 비교 분석 결과를 보여줍니다.

- **Performance Highlights**: MIRA는 높은 해석 가능성과 성능을 동시에 제공하여 안전이 중요한 응용 프로그램에서 해석 가능한 AI의 광범위한 채택 가능성을 강조합니다.



### How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs (https://arxiv.org/abs/2410.13857)
- **What's New**: 이 논문에서는 Transformer 기반 대형 언어 모델(LLMs)의 수학적 능력을 이론적으로 분석하고, 특히 산술작업에서의 성능을 강조합니다. 숫자 정밀도가 수학적 작업의 성공적인 수행을 좌우하는 핵심 요소로 밝혀졌습니다.

- **Technical Details**: 저자들은 LLM의 기본 산술 작업인 정수 덧셈, 반복 덧셈, 정수 곱셈을 분석합니다. 저자들은 정밀도에 따라 모델의 크기가 달라지며, 낮은 정밀도(int8, int4)의 Transformer는 문제를 풀기 위해 폭발적으로 큰 모델을 요구한다고 주장합니다. 이와 대조적으로 표준 정밀도(float32)는 훨씬 작고 효율적인 모델로도 이를 처리할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과는 두 가지 정밀도(int4 및 표준 정밀도) 모두에서 정수 덧셈 작업에서 충분한 성능을 보였지만, 반복 덧셈 및 정수 곱셈과 같은 복잡한 작업에서는 낮은 정밀도가 성능 저하를 일으킨다는 것을 보여주었습니다.



### Can MLLMs Understand the Deep Implication Behind Chinese Images? (https://arxiv.org/abs/2410.13854)
Comments:
          32 pages,18 figures. Project Page: this https URL Code: this https URL Dataset: this https URL

- **What's New**: 이번 연구에서는 중국 이미지의 고차원 인식 및 이해 능력을 평가하기 위한 새로운 벤치마크인 **CII-Bench**를 소개합니다. 이 벤치마크는 중국 전통 문화 이미지를 포함하여 MLLMs의 성능을 진단할 수 있는 도전적인 과제를 제공합니다.

- **Technical Details**: CII-Bench는 698개의 이미지와 800개의 다양한 선택 질문을 포함하고 있으며 여섯 가지 도메인: 생활, 예술, 사회, 정치, 환경, 중국 전통 문화로 구성되어 있습니다. 이 벤치마크는 그림, 만화, 포스터 등 다양한 이미지 유형을 활용하여 MLLMs의 이해 능력을 평가합니다.

- **Performance Highlights**: MLLMs의 정확도를 조사한 결과, 최고 정확도는 64.4%로 인간의 평균 정확도 78.2%와 비교될 때 상당한 성능 차이를 보였습니다. 특히, MLLMs는 중국 전통 문화 이미지에서 낮은 성능을 보였으며, 감정 힌트를 포함할 때 모델의 정확도가 향상되는 경향이 있었습니다.



### Retrospective Learning from Interactions (https://arxiv.org/abs/2410.13852)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)과 사용자 간의 다중 턴(multi-turn) 상호작용에서 발생하는 암묵적인 피드백 신호를 활용하여 모델 향상을 꾀하는 새로운 방법인 ReSpect를 소개합니다.

- **Technical Details**: ReSpect는 과거 상호작용에서 발생했던 암묵적 신호를 통해 학습하는 방법입니다. 이 방법은 사용자와의 상호작용 후, 모델이 자신의 과거 행동을 회고하여 피드백을 해석하고 재훈련하는 과정을 포함합니다. 이를 통해 사용자는 모델의 수행 여부를 신호로 전달하며, 이 신호는 자연어의 한정된 하위 공간에 위치하여 LLM이 이를 쉽게 감지할 수 있게 합니다.

- **Performance Highlights**: ReSpect를 적용한 결과, IDEFICS2-8B 모델의 작업 완료율이 31%에서 82%로 향상되었습니다. 이 과정에서는 외부 주석 없이도, 과거 상호작용을 통해 직접적으로 스스로 피드백을 해석하고 개선하는 능력을 보여주었습니다.



### Influence Functions for Scalable Data Attribution in Diffusion Models (https://arxiv.org/abs/2410.13850)
- **What's New**: 확산 모델에 대한 데이터 기여도 및 해석 가능성 문제를 해결하기 위해 영향 함수(influence functions) 프레임워크를 개발하여 새로운 방법을 제시합니다.

- **Technical Details**: 기여도 추정 방법인 영향 함수는 모델 출력이 특정 훈련 데이터를 제거했을 때 어떻게 변할지를 근사합니다. K-FAC(Kronecker-Factored Approximate Curvature) 근사 방법을 사용하여 해시안(Hessian) 계산의 확장성을 보장합니다.

- **Performance Highlights**: 제안된 방법은 Linear Data-modelling Score(LDS)와 같은 평가에서 기존 데이터 기여도 접근 방식보다 성능이 우수함을 보여주었으며, 특정 하이퍼파라미터 조정 없이도 성능을 발휘합니다.



### Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2410.13848)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 다양한 모드의 이해 및 생성을 통합한 새로운 자율 회귀 프레임워크인 Janus를 소개합니다. 기존 연구는 주로 단일 시각 인코더를 사용했으나, Janus는 시각 인코딩을 별도의 경로로 분리하여 성능과 유연성을 향상시켰습니다.

- **Technical Details**: Janus는 고유한 transformer 아키텍처를 사용하여 시각 이해 및 생성을 위한 독립적인 인코딩 경로를 제공합니다. 이를 통해 이해와 생성 작업 사이의 정보를 분리하고, 각 작업에 가장 적합한 인코딩 방법을 선택할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: Janus는 기존의 통합 모델보다 뛰어난 성능을 보여주며, MMBench 및 SEED-Bench와 같은 벤치마크에서 최고 성과를 기록했습니다. 또한, DALL-E 2와 SDXL과 같은 특정 작업 모델을 초월하는 성과를 보였습니다.



### SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction (https://arxiv.org/abs/2410.13846)
- **What's New**: SimLayerKV라는 새로운 방법을 소개하며, 긴 맥락을 처리하는 대형 언어 모델에서의 KV cache의 비효율성을 줄입니다. 이 방법은 "lazy" layer를 식별하고 이들에 대한 KV cache의 중복을 줄여 효율성을 증대시킵니다.

- **Technical Details**: SimLayerKV는 특정 존재 layers의 KV cache를 선택적으로 제거하여 inter-layer KV cache 중복을 감소시키는 방법입니다. 이는 lazy layers의 주의 할당 패턴을 분석하여 가능하며, non-lazy layers의 KV cache는 유지합니다. 코드 구현은 단 7줄로 가능하며, 훈련 과정이 필요 없습니다.

- **Performance Highlights**: SimLayerKV는 LongBench 벤치마크에서 3개의 대표적인 LLM (LLaMA2-7B, LLaMA3-8B, Mistral-7B)에서 KV cache 비압축 비율 5배를 달성하며, 4-bit quantization을 함께 사용할 때 오직 1.2%의 성능 저하만 보여줍니다.



### Accelerating Codec-based Speech Synthesis with Multi-Token Prediction and Speculative Decoding (https://arxiv.org/abs/2410.13839)
Comments:
          Submitted to IEEE ICASSP 2025

- **What's New**: 본 논문에서는 음질 저하 없이 코드 기반 음성 합성 시스템의 속도를 가속화하기 위한 향상된 추론 방법을 제안합니다. 이 방법은 추가적인 훈련 없이 추론 간의 속도와 품질 간의 유연한 균형을 제공합니다.

- **Technical Details**: 핵심 아이디어는 여러 개의 예측 헤드를 사용하여 AR 모듈의 추론 단계에서 여러 개의 토큰을 예측하는 것입니다. 이로 인해 헤드 수가 증가함에 따라 합성 시간이 선형적으로 감소합니다. 또한, Viterbi 기반 알고리즘을 활용한 새로운 투기적 디코딩 기법을 도입하여 각 디코딩 단계에서 생성된 토큰의 최적 시퀀스를 선택합니다.

- **Performance Highlights**: 실험 결과, 각 토큰 예측에 필요한 시간이 기준 모델에 비해 4배에서 5배 줄어들었으며, 음성 이해도 측면에서 최소한의 품질 저하 또는 오히려 향상된 결과를 보였습니다.



### ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization (https://arxiv.org/abs/2410.13837)
Comments:
          preprint, 35 pages, 23 figures

- **What's New**: 본 논문에서는 보상 형성(reward shaping)에서의 새로운 접근법인 Online Reward Selection and Policy Optimization (ORSO)를 제안합니다. 이 방법은 보상 형성 선택을 온라인 모델 선택 문제로 프레이밍하여 자동으로 적합한 보상 형성 함수를 찾아내는 데 초점을 맞추고 있습니다.

- **Technical Details**: ORSO는 합리적인 탐색 전략(principled exploration strategies)을 활용하여 인간의 개입 없이도 유망한 보상 형성 함수(shaping reward functions)를 식별합니다. 이 방법은 탐색(exploration)과 활용(exploitation)을 균형 있게 조절하며, 검증 가능한 후회 보장(regret guarantees)을 제공합니다.

- **Performance Highlights**: Isacc Gym 시뮬레이터를 사용한 다양한 연속 제어(tasks) 실험에서 ORSO의 효과를 입증하였으며, 전통적인 방법에 비해 샘플 효율(sample efficiency)을 크게 향상시키고 계산 시간을 줄이며, 도메인 전문가가 수동으로 엔지니어링한 보상에 의해 생성된 정책과 유사한 고품질 보상 함수를 지속적으로 식별합니다.



### The Disparate Benefits of Deep Ensembles (https://arxiv.org/abs/2410.13831)
- **What's New**: 최근 딥 신경망(Deep Neural Networks, DNNs)의 성능을 높일 수 있는 간편한 방법으로 사용되는 딥 앙상블(Deep Ensembles)에 대한 공정성(Algorithmic Fairness) 측면에서의 영향이 잘 이해되지 않았음을 밝히며, 본 연구는 딥 앙상블의 성능 향상과 공정성 간의 상호작용을 분석합니다.

- **Technical Details**: 이 연구는 딥 앙상블을 이용하여 얼굴 분석 및 의료 영상 데이터셋에서 공정성 메트릭을 활용하여 성능 편차를 empirically 조사합니다. 특히, 다양한 protected group 속성에 따라 성능이 상이하게 나타나는 'disparate benefits effect'를 발견했으며, 이 효과의 원인으로 그룹 내 예측의 다양성 차이를 규명했습니다.

- **Performance Highlights**: 본 연구에서 제안된 Hardt 후처리(post-processing) 방법이 효과적으로 공정성을 높이면서도 딥 앙상블의 성능을 유지할 수 있음을 보여줍니다. 분석을 통해 공정성 지표를 향상시킬 수 있는 다양한 접근 방식을 평가하였고, 딥 앙상블의 성능이 다수의 그룹 메트릭에서 불균형적으로 나타나는 것을 실증적으로 확인했습니다.



### A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglemen (https://arxiv.org/abs/2410.13828)
- **What's New**: 본 논문에서는 Reinforcement Learning from Human Feedback (RLHF)에서 전통적인 margin-based 손실을 사용하는 것의 문제점을 다루고 있습니다. 특히, 이 접근 방법이 선호 및 비선호 응답 각각에 대해 이상적인 언어 모델 behavior를 충분히 명시하지 않는다는 점이 강조됩니다.

- **Technical Details**: 우리는 margin의 증가에 따른 두 가지 의도치 않은 결과를 식별했습니다: (1) 비선호 응답의 확률이 증가할 수 있으며 이는 안전 문제와 관련된 alignment 실패를 초래할 수 있습니다. (2) 선호 응답의 확률이 감소할 수 있으며, 이 경우에도 그 응답은 이상적일 수 있습니다. 이러한 현상의 원인은 gradient entanglement으로 명명하였으며, 이는 선호 및 비선호 응답의 확률 변화가 서로 얽혀 있는 문제를 나타냅니다.

- **Performance Highlights**: 본 논문은 margin 기반 preference optimization 알고리즘의 훈련 동역학을 설명하고, margin 기반 방법의 under-specification 문제를 완화할 수 있는 잠재적인 알고리즘 설계를 제안합니다.



### Unearthing Skill-Level Insights for Understanding Trade-Offs of Foundation Models (https://arxiv.org/abs/2410.13826)
Comments:
          Code at: this http URL

- **What's New**: 이 논문은 모델 평가에서의 복잡성을 해결하기 위해, 모델이 생성한 이론(rationales)을 사용하여 기저가 되는 기술(skills)을 자동으로 복구하는 방법을 제안합니다. 기존의 평가 지표에 숨겨진 다양한 기술을 분석하여, 구체적이고 행동 가능한 모델 능력 이해를 제공합니다.

- **Technical Details**: 평가 인스턴스에 대해 강력한 모델(예: GPT-4o)을 사용하여 단계별 이론을 생성하고 각 단계에서 적용된 기술을 나열합니다. 이 과정을 통해 46,000개 이상의 인스턴스를 분석하고 기술 조각(skill-slices)을 작성하여 여러 벤치마크에서 기술의 정확성을 비교합니다.

- **Performance Highlights**: 우리는 기술 조각 분석을 통해 모델 간의 성능 무역에 대한 새로운 통찰을 발견했습니다. 예를 들어, Gemini 1.5 Pro는 'molar mass 계산'에서 평균적으로 18% 더 정확하지만 '헌법법 적용'에서는 19% 덜 정확하다는 결과를 보여주었습니다. 이러한 분석 방법을 통해 우리는 전체 12개 데이터셋에서 3%의 정확도 향상을 확인했습니다.



### Multi-style conversion for semantic segmentation of lesions in fundus images by adversarial attacks (https://arxiv.org/abs/2410.13822)
Comments:
          preprint

- **What's New**: 이번 논문에서는 다양한 데이터베이스의 주석 스타일 간 표준화를 해결하기 위해 'adversarial style conversion'이라는 새로운 방법을 도입합니다. 이 방법은 단일 아키텍처에서 결합된 데이터베이스를 활용하여 모델이 입력에 따라 자발적으로 세분화 스타일을 조정하도록 훈련되었습니다.

- **Technical Details**: 제안된 방법론은 인코더 특징을 기반으로 데이터셋의 출처를 탐지하는 'linear probe'를 추가하고, 적대적 공격(adversarial attacks)을 통해 모델의 세분화 스타일을 조정하는 방식을 채택합니다. 이는 여러 데이터셋에서 훈련된 세분화 모델의 스타일 변환을 가능하게 합니다.

- **Performance Highlights**: 논문의 결과는 데이터셋 조합을 통해 질적으로나 양적으로 유의미한 개선을 보이며, 모델의 일반화 성능, 불확실성 추정 및 주석 스타일 간의 지속적 보간과 같은 기회를 제공합니다.



### Artificial Kuramoto Oscillatory Neurons (https://arxiv.org/abs/2410.13821)
Comments:
          Code: this https URL

- **What's New**: 본 연구에서는 Artificial Kuramoto Oscillatory Neurons (AKOrN)을 소개합니다. 이는 전통적인 threshold units의 동적 대안으로, 다양한 connectivity 디자인과 결합할 수 있습니다.

- **Technical Details**: AKOrN은 Kuramoto 업데이트를 통해 뉴런의 동기화 동적을 이용하며, 이는 비대칭 연결을 통해 뉴런 간의 상호작용을 탐구합니다. 연구에서는 4개의 합성 데이터셋과 2개의 실제 이미지 데이터셋에서 성능을 평가하였습니다.

- **Performance Highlights**: AKOrN은 비지도 객체 발견, 적대적 강건성, 캘리브레이션된 불확실성 정량화 및 추론을 포함한 다양한 작업에서 향상된 성능을 보여주었습니다.



### Guided Reinforcement Learning for Robust Multi-Contact Loco-Manipulation (https://arxiv.org/abs/2410.13817)
Comments:
          J. P. Sleiman and M. Mittal contributed equally. Accepted for CoRL 2024 (Oral). Project website: this https URL

- **What's New**: 이번 연구는 다중 접촉 로코-조작(loco-manipulation) 작업을 위한 행동 합성 및 제어에 대한 체계적인 접근 방식을 제안합니다. 기존의 RL (Reinforcement Learning) 방법이 요구하는 고도의 MDP (Markov Decision Process) 설계를 대체하여, 단일 시연으로 RL 정책을 학습할 수 있게 합니다.

- **Technical Details**: 제안한 방법은 TO (Trajectory Optimization) 기반 프레임워크에서 생성된 적응형 궤적을 사용하여 RL 에이전트가 복잡한 행동을 학습하도록 이끕니다. 우리의 MDP는 모델링 불확실성, 외부 방해 요인 및 예기치 않은 이벤트을 처리하는 능력을 갖춘 로코-조작 정책을 효율적으로 학습하는 데 중점을 둡니다.

- **Performance Highlights**: 제안한 정책은 문을 밀고 당기거나 식기세척기를 여닫는 정해진 네 가지 작업에서 선행 Motion Imitation RL 방법과 비교하여 높은 성공률을 보여주었습니다. 훈련된 정책은 실제 로봇으로 이식되어, 잔여 객체 모델에의 강인성과 슬립(slip) 상황에 대한 반응성을 보여줍니다.



### Learning Graph Quantized Tokenizers for Transformers (https://arxiv.org/abs/2410.13798)
- **What's New**: 이번 논문에서는 Graph Quantized Tokenizer (GQT)를 도입하여 그래프의 토큰화 과정을 개선했습니다. GQT는 멀티태스킹 그래프 자기 지도 학습(multi-task graph self-supervised learning)을 활용하여 토크나이저 훈련과 트랜스포머 훈련을 분리함으로써 더 강력하고 일반화 가능한 토큰을 생성합니다.

- **Technical Details**: GQT는 Residual Vector Quantization (RVQ) 기법을 통해 계층적인 이산 토큰을 학습하여 메모리 요구 사항을 크게 줄이고 일반화 능력을 향상시킵니다. 이 방법은 의미적 엣지와 랜덤 워크를 결합하여 트랜스포머가 장거리 상호작용에 접근할 수 있도록 합니다.

- **Performance Highlights**: GQT를 트랜스포머 인코더와 결합하여 18개 벤치마크 중 16개에서 최첨단 성능을 달성하였으며, 특히 대규모 동질적 및 이질적 데이터셋에서 성능이 뛰어났습니다. 이는 매우 감소된 메모리 풋프린트를 갖춘 임베딩을 통해 달성되었습니다.



### Looking Inward: Language Models Can Learn About Themselves by Introspection (https://arxiv.org/abs/2410.13787)
Comments:
          15 pages, 9 figures

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)가 스스로를 반성(introspection)하는 능력이 있음을 보여줍니다. 이는 모델이 자신의 내부 상태에 대해 정보를 얻고, 이를 기반으로 스스로의 행동을 예측할 수 있음을 의미합니다. 특히, 반성 능력은 모델 해석 가능성(interpretable) 향상에 기여할 수 있습니다.

- **Technical Details**: LLMs는 두 개의 모델, M1과 M2를 사용하여 스스로의 행동 특성을 예측하는 방식으로 반성을 연구했습니다. M1은 자신의 행동을 예측하기 위해 미세 조정(finetuning)되었고, M2는 M1의 실제 행동을 바탕으로 훈련되었습니다. 실험 결과 M1이 M2보다 더 정확하게 스스로를 예측하는 것으로 나타났습니다. 이러한 결과는 LLMs가 훈련 데이터에만 의존하지 않고, 자신에 대한 특별한 접근 권한(privileged access)을 가지고 있다는 것을 시사합니다.

- **Performance Highlights**: 실험에서는 M1이 M2에 비해 정확도가 평균 17% 향상되었습니다. 또한, M1은 의도적으로 자신의 행동을 변경한 후에도 여전히 정확한 예측을 수행할 수 있었으며, 이는 LLMs의 반성 능력이 특정 작업에서 더 나은 보정(calibration)을 보여준다는 것을 나타냅니다. 그러나 복잡한 작업에서는 여전히 반성 능력이 제한적임을 발견했습니다.



### PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignmen (https://arxiv.org/abs/2410.13785)
Comments:
          28 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 정렬 개선을 위한 PopAlign 프레임워크를 제안합니다. 기존의 대조 패턴에 대한 제한된 접근 방식을 넘어, 다양한 대조 패턴을 통합하여 모델의 인간 선호에 대한 반응을 개선합니다.

- **Technical Details**: PopAlign은 크게 세 가지 수준인 (1) 프롬프트(prompt), (2) 모델(model), (3) 파이프라인(pipeline)에서 대조적 패턴을 통합합니다. 여섯 가지 대조 전략을 통해 기존의 피드백 라벨링 절차 없이도 효과적인 대조 데이터를 생성할 수 있도록 합니다. 주요 대조 전략에는 Prefix Contrast, Demon Contrast, Elicitive Contrast, NParam Contrast, Leaderboard Contrast, Refine Contrast가 포함됩니다.

- **Performance Highlights**: PopAlign은 기존 방법들에 비해 모델 정렬 성능이 획기적으로 향상됨을 실험적으로 입증하였으며, 특히 Elicitive Contrast 전략이 그 성능 향상에 크게 기여하는 것을 보였습니다. 다양한 작업에서 차별화된 성능을 나타내며, 오라클 선호 모델에 대한 정확성을 기반으로 한 대조 정확도 및 선호 모델링도 분석하였습니다.



### Optimal Quantization for Matrix Multiplication (https://arxiv.org/abs/2410.13780)
- **What's New**: 본 연구는 대규모 매트릭스의 lossy compression (양자화) 기법을 통해 매트릭스 곱셈을 가속화하기 위한 새로운 알고리즘을 제안합니다. 이 접근법은 전통적인 벡터 양자화와 다르게, 매트릭스 자체가 아니라 매트릭스 곱셈의 근사를 목표로 합니다.

- **Technical Details**: 이 논문은 iid Gaussian 아이템을 가진 매트릭스의 평균 제곱 오차에 대한 비비대칭 하한을 제공하며, 특정한 프레임워크에서 Frobenius norms를 사용하여 매트릭스 A, B의 압축과 동시에 근사 오차를 보장하는 보편적인 양자기를 제안합니다. 이는 깊은 신경망(Deep Neural Networks)과 대규모 언어 모델(Large Language Models)에서 메모리 대역폭의 병목 현상을 해결하기 위한 중요성을 강조합니다.

- **Performance Highlights**: 제안된 양자기는 최적 성능에 근접한 결과를 실현하며, 정보 이론적으로 iid Gaussian 매트릭스의 매트릭스 곱셈에 대한 rate-distortion function을 도출합니다.



### Aggregation Artifacts in Subjective Tasks Collapse Large Language Models' Posteriors (https://arxiv.org/abs/2410.13776)
Comments:
          12 pages, 7 figures, 2 tables

- **What's New**: 본 논문은 In-context Learning (ICL) 이 LLMs(대형 언어 모델)의 자연어 작업 수행에 있어 주요 방법으로 자리잡고 있으며, 전이 학습에서 얻은 지식이 이 과정에서 중요하다는 점을 강조합니다. 그러나 ICL이 단순히 태스크 프라이어(task priors)를 재탐색하는 데 지나치게 의존하며, 이는 복잡한 주관적 분야에서는 더욱 두드러진다고 지적합니다.

- **Technical Details**: 저자들은 LLM이 제공하는 데이터셋의 집합적(aggregation) 사용으로 인해 발생하는 주석 아티팩트가 모델의 성능에 미치는 영향을 분석하였습니다. 이 과정에서 LLM이 개별 주석자(annotator)의 관점에 더 잘 맞춘다는 사실을 발견하였고, 소수의 주석자(minority annotators)가 LLM의 프라이어와 더 잘 정렬됨을 확인하였습니다.

- **Performance Highlights**: 이 논문에서는 주석의 집합적 사용이 주관적 작업 모델링에 방해가 된다는 강력한 상관관계를 보여주며, 소수 주석자들이 LLM과 더 긍정적인 상호작용을 하는 경향이 있음을 강조합니다. 그러나 데이터 집합의 집합적 사용만으로는 ICL과 최신 기술 간의 성능 차이를 설명할 수 없음을 발견하였습니다.



### Rapid and Automated Alloy Design with Graph Neural Network-Powered LLM-Driven Multi-Agent Systems (https://arxiv.org/abs/2410.13768)
- **What's New**: 본 논문에서는 새로운 금속 합금 발견을 자동화하기 위해 다중 에이전트 AI 모델을 활용했습니다. 이 시스템은 LLM(대규모 언어 모델)과 GNN(그래프 신경망)이 결합된 구조로, 물리적 시뮬레이션에서 도출한 데이터와 외부 지식을 통합하여 복잡한 합금 설계를 지원합니다.

- **Technical Details**: 이 모델은 (a) 추론 및 계획 작업을 담당하는 LLM의 모음, (b) 서로 다른 역할과 전문성을 가진 AI 에이전트 그룹, (c) 주요 물리적 특성을 신속하게 검색하기 위한 GNN 모델로 구성됩니다. GNN 모델은 NbMoTa 계열의 BCC 합금을 대상으로 Peierls 장벽 및 용질/스크류 전위 상호작용 에너지와 같은 원자 규모의 특성을 예측합니다.

- **Performance Highlights**: 이 AI 시스템은 계산 비용을 줄이고 여러 에이전트를 통해 자동으로 합금 디자인 공간을 탐색함으로써, 새로운 합금을 발견하는 과정을 가속화합니다. 본 연구는 복잡한 시스템에서의 광범위한 응용 가능성을 제시하며, 소재 설계에서의 자동화된 발견에 큰 진전을 이루었습니다.



### Virtual Sensing for Real-Time Degradation Monitoring of Nuclear Systems: Leveraging DeepONet for Enhanced Sensing Coverage for Digital Twin-Enabling Technology (https://arxiv.org/abs/2410.13762)
- **What's New**: 본 논문에서는 AP-1000 Pressurized Water Reactor (PWR)에서 고온 다리(hot leg)의 열유체 매개변수를 예측하기 위해 Deep Operator Networks (DeepONet)를 사용하는 방안을 제안합니다. 이는 디지털 트윈(digital twin) 프레임워크 내에서 실행되며, 지속적인 재학습의 필요성을 완화하여 온라인 및 실시간 예측을 가능하게 합니다.

- **Technical Details**: DeepONet는 다양한 운영 조건에 대해 훈련되며, 이로 인해 많은 수의 데이터 및 복잡한 고차원 데이터를 효율적으로 처리할 수 있습니다. 본 연구는 DeepONet이 평균 제곱 오차(mean squared error) 및 상대 L2 오류(relational L2 error)가 낮은 결과를 보이며, 전통적인 유한 요소(finite element) 시뮬레이션보다 160,000배 빠른 예측을 할 수 있음을 보여줍니다.

- **Performance Highlights**: DeepONet는 실시간으로 재료 열화(indicators of material degradation)를 추적하는 데 매우 효과적인 도구로 입증되었으며, 이러한 속도와 정확성은 원자로 안전성과 수명을 높이는 데 기여합니다.



### MobA: A Two-Level Agent System for Efficient Mobile Task Automation (https://arxiv.org/abs/2410.13757)
Comments:
          27 pages, 6 figures, and 5 tables. We will release our source code in a few days

- **What's New**: MobA라는 혁신적인 모바일 어시스턴트를 제안합니다. 이를 통해 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLM)을 활용하여 사용자의 명령 이해와 계획 능력을 향상시킵니다.

- **Technical Details**: MobA는 두 가지 수준의 에이전트 아키텍처로 구성되어 있습니다. 상위 에이전트(Global Agent, GA)는 사용자 명령을 이해하고, 히스토리 메모리를 추적하며, 작업을 계획하는 역할을 합니다. 하위 에이전트(Local Agent, LA)는 GA의 메모리와 서브 태스크에 따라 상세한 작업을 함수 호출의 형태로 예측합니다. 또한, Reflect Module을 통합하여 이전에 보지 못한 복잡한 작업을 처리할 수 있는 능력을 제공합니다.

- **Performance Highlights**: MobA는 실제 평가에서 작업 수행 효율성(Task Execution Efficiency)과 완료율(Completion Rate)에서 상당한 개선을 보여주며, MLLM을 활용한 모바일 어시스턴트의 가능성을 강조합니다.



### CLIMB: Language-Guided Continual Learning for Task Planning with Iterative Model Building (https://arxiv.org/abs/2410.13756)
Comments:
          6 pages, 6 figures

- **What's New**: CLIMB는 지속적인 학습을 통해 로봇 작업 계획을 지원하는 새로운 프레임워크로, 자연어 설명을 바탕으로 도메인 모델을 생성하고 비직관적인 술어를 학습하여 향후 문제에 활용할 수 있습니다.

- **Technical Details**: CLIMB는 하이브리드 신경-심볼릭 (neuro-symbolic) 계획 시스템으로, 기초 모델과 전통적인 심볼릭 계획자를 결합하여 중복 학습 없이 과거의 문제를 해결할 수 있는 능력을 보유하고 있습니다. 이 시스템은 PDDL 모델을 점진적으로 구축하며, 작업을 수행하면서 환경의 원인 구조를 즉각 반영합니다.

- **Performance Highlights**: CLIMB는 예비 성능 시험에서 기존 방법과 비교하여 일반 계획 환경에서 성능 향상을 입증했습니다. BlocksWorld++ 도메인을 통해 점진적 논리 세계 모델 구축 능력을 평가했으며, 실험 결과 CLIMB의 향상된 성능을 확인할 수 있었습니다.



### Privacy-Preserving Decentralized AI with Confidential Computing (https://arxiv.org/abs/2410.13752)
- **What's New**: 본 논문은 탈 중앙화된 인공지능(AI) 플랫폼인 Atoma Network에서 기밀 컴퓨팅(Confidential Computing, CC)을 활용한 프라이버시 보호에 대해 다룹니다. 이 기술은 탈 중앙화된 AI가 직면한 프라이버시 문제를 해결하기 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 문서에서 제안하는 기밀 컴퓨팅은 하드웨어 기반의 신뢰할 수 있는 실행 환경(Trusted Execution Environments, TEE)을 활용하여 민감한 데이터를 처리하는 동안 코드를 보호하고 기밀성을 유지하는 역할을 합니다. TEEs는 분산 환경에서도 데이터와 모델 파라미터가 외부에 노출되지 않도록 보장합니다.

- **Performance Highlights**: TEEs는 높은 프라이버시 보호 기능을 제공하고, 탈 중앙화된 AI의 채택을 촉진하는 데 기여할 것으로 기대됩니다. 특히, 프라이버시 우려를 해소하면서 안전하고 신뢰할 수 있는 AI 연산에 소요되는 리소스를 줄일 수 있는 가능성을 보여줍니다.



### LLM-Human Pipeline for Cultural Context Grounding of Conversations (https://arxiv.org/abs/2410.13727)
Comments:
          19 pages, 9 figures, 7 tables

- **What's New**: 이 논문에서는 대화에서 문화적 맥락을 이해하기 위해 'Cultural Context Schema'를 도입하는 방법을 제안합니다. 특히, 감정, 대화 행위와 같은 대화 정보와 사회적 규범, 위반과 같은 문화적 정보를 포함한 구조를 구축하고 이를 바탕으로 실제 중국 문화에 맞춘 약 11만 건의 사회적 규범 및 위반 설명을 생성하였습니다.

- **Technical Details**: 논문은 LLMs (Large Language Models)을 사용하여 대화 관련 문화 정보를 생성하고, 이를 표상하는 'Norm Concepts'를 만들기 위한 다단계 접근을 채택합니다. 과정에는 인간 주도의 검증 및 대화 내용의 세부사항을 기호 주석(symbolic annotation)으로 grounding하는 것이 포함됩니다. 또한, 정서 검출(emotion detection), 감정 감지(sentiment detection), 대화 행위 검출(dialogue act detection)과 같은 하위 작업에 활용될 대규모 데이터셋을 생성합니다.

- **Performance Highlights**: 제안된 문화적 맥락 데이터셋은 경험적으로 대화 이해 작업에 대한 성능을 유의미하게 향상시키는 것으로 나타났습니다. 또한, 고품질의 데이터셋과 평가 실험을 통해 결과를 보이고 있으며, 데이터 및 코드가 MIT 라이센스 하에 공개될 예정입니다.



### DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation (https://arxiv.org/abs/2410.13726)
- **What's New**: 이번 연구에서는 DAWN(Dynamic frame Avatar With Non-autoregressive diffusion)이라는 새로운 프레임워크를 통해 오디오 클립과 초상화를 이용한 직접적인 동영상 생성 방식을 선보입니다. DAWN은 기존의 autoregressive (AR) 방식의 한계를 극복하여, 모든 프레임을 동시 생성할 수 있는 비선형(non-autoregressive, NAR) 전략을 적용하였습니다.

- **Technical Details**: DAWN 프레임워크는 (1) 오디오에 기반한 전체적인 얼굴 역학을 생성하는 잠재적 동작 공간에서의 생성과 (2) 오디오 기반의 머리 자세 및 깜빡임 생성이라는 두 가지 주요 구성 요소로 이루어져 있습니다. 추가적으로, Pose and Blink generation Network (PBNet)는 오디오에서 자연스러운 머리 자세와 깜빡임 시퀀스를 생성하는 데 사용됩니다. DAWN은 A2V-FDM(Audio-to-Video Flow Diffusion Model)을 통해 입술과 오디오 간의 암묵적 관계를 학습합니다.

- **Performance Highlights**: DAWN은 빠른 생성 속도와 더불어 정확한 입술 동작 및 자연스러운 자세/깜빡임을 보장하여, 실제감 있고 생동감 넘치는 비디오를 생성합니다. 또한, DAWN은 뛰어난 외삽(extrapolation) 능력을 발휘하며, 긴 비디오에서도 높은 품질을 안정적으로 유지할 수 있는 가능성을 보여줍니다.



### Persistent Pre-Training Poisoning of LLMs (https://arxiv.org/abs/2410.13722)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)이 사전 훈련(pre-training) 과정에서도 악성 공격에 의해 손상될 수 있는지를 처음으로 평가합니다. 특히, LLM이 유용하고 무해한 챗봇으로 세부 훈련(fine-tuning)될 때까지 이러한 공격 효과가 지속되는지를 연구했습니다.

- **Technical Details**: 연구팀은 0.1%의 데이터가 오염되는 경우에도 여러 가지 공격(서비스 거부, 신념 조작, 탈옥, 프롬프트 도용)의 효과가 지속됨을 확인했습니다. 실험은 600M에서 7B 파라미터의 다양한 모델 크기를 사용해 진행되었습니다. 주요 공격 중, 서비스 거부(denial-of-service) 공격은 0.001%의 오염율에서도 지속성이 나타났습니다.

- **Performance Highlights**: 연구 결과, 사전 훈련 데이터의 0.1%를 오염시키는 것만으로도 훈련 후 모든 공격에서 효과가 들여다보였으며, 탈옥(jailbreaking) 공격은 안전 훈련(safety training) 방법으로는 지속되지 않는 것으로 나타났습니다.



### Movie Gen: A Cast of Media Foundation Models (https://arxiv.org/abs/2410.13720)
- **What's New**: 이번 논문에서는 Movie Gen이라는 새로운 foundation 모델 세트를 제안합니다. 이 모델은 다양한 화면 비율과 동기화된 오디오와 함께 고품질 1080p HD 비디오를 생성하며, 사용자의 이미지를 기반으로 한 개인화된 비디오 생성 및 정밀한 지침 기반 비디오 편집 기능도 포함되어 있습니다.

- **Technical Details**: Movie Gen은 30B 파라미터의 트랜스포머 모델로, 최대 73K 비디오 토큰의 컨텍스트 길이를 가지고 있습니다. 이 모델은 텍스트-비디오 합성, 비디오 개인화, 비디오 편집, 비디오-오디오 생성 및 텍스트-오디오 생성과 같은 다양한 작업에서 최첨단 성능을 기록합니다. 인터넷 스케일의 이미지, 비디오, 오디오 데이터를 통해 사전 학습되었습니다.

- **Performance Highlights**: Movie Gen 모델은 기존 상업 시스템을 초월하여 텍스트-비디오 생성, 비디오 개인화, 정밀 비디오 편집 및 오디오 생성 작업에서 탁월한 성능을 보여줍니다. 특히, Movie Gen Video는 최대 16초의 개인화된 HD 비디오 생성을 가능하게 하며, Movie Gen Audio는 정밀한 음악 생성과 음향 효과 생성을 지원합니다.



### MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2410.13716)
- **What's New**: 본 논문에서는 현대의 Retrieval-Augmented Generation (RAG) 평가 기준에서의 한계를 해결하기 위해, 성능 높은 LLM (Large Language Model)을 대체할 수 있는 'learning to rank' 모델을 훈련하여 'synthetic arena-based leaderboard'를 생성하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 우리는 RAG를 기반으로 한 평가 지표를 입력으로 사용하여 학습된 surrogate judge 모델을 통해 18개 언어에서 다국어 RAG 벤치마크인 MIRAGE-Bench를 개발했습니다. 이 모델은 Google의 GPT-4o를 사용하여 수행된 쌍별 평가 결과를 바탕으로 Bradley-Terry 모델을 이용해 leaderboard를 생성했습니다. 이 과정에서 우리는 언어 감지, 인용 품질, 지원도, 정답 중복성 및 유창성 등 총 7개의 평가 기준을 활용했습니다.

- **Performance Highlights**: 실험 결과, 학습된 'learning to rank' 모델은 GPT-4o를 기반으로 한 leaderboard와 높은 상관관계(Kendall Tau (τ) = 0.909)를 보였으며, 70B 이상의 매개변수를 가진 대형 모델들이 MIRAGE-Bench에서 우수한 성능을 기록했습니다. 또한, MIRAGE의 훈련 데이터는 작은 오픈 소스 모델을 개선하는 데 유용하다는 것을 보여주었습니다.



### On the Role of Attention Heads in Large Language Model Safety (https://arxiv.org/abs/2410.13708)
Comments:
          28 pages, 18 figures, 7 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 안전 메커니즘의 특정 주의(attention) 헤드의 기여도를 이해하고, 그로 인해 발생하는 안전성 문제를 분석합니다. 특히, Safety Head ImPortant Score (Ships)라는 새로운 메트릭을 도입하여 안전성과 관련된 다중 헤드 주의 메커니즘을 탐구합니다.

- **Technical Details**: 우리는 LLM의 안전성 능력을 다중 헤드 주의(mechanism)와 연결하여 해석하기 위한 연구를 진행했습니다. Ships는 개별 주의 헤드가 해로운 쿼리에 대한 거부 확률 변화에 미치는 영향을 정량화합니다. 추가적으로, Safety Attention Head AttRibution Algorithm (Sahara)을 제안하여 중요 헤드를 그룹화하고, 이들이 모델의 안전성에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과, Llama-2-7b-chat 모델에서 안전 헤드 하나를 제거했을 때, 해로운 쿼리에 대한 공격 성공률(ASR)이 0.04에서 0.64로 증가하였으며, 이는 기존 연구에서 필요로 했던 약 5%의 매개변수 수정과 대조적으로 단 0.006%를 수정함으로써 이루어졌습니다. 또한, 유사한 기본 모델에서 미세 조정된 LLM의 안전 헤드들이 겹친다는 점에서, 안전성에 대한 기존 연구와 새로운 통찰을 제공합니다.



### Jailbreaking LLM-Controlled Robots (https://arxiv.org/abs/2410.13691)
- **What's New**: 최근 대형 언어 모델(LLMs)의 도입은 조작(manipulation), 이동(locomotion), 자율 주행 차량(self-driving vehicles) 등 다양한 분야에서 맥락적 추론(contextual reasoning) 및 직관적인 인간-로봇 상호작용을 가능하게 하여 로봇 공학 분야에 혁신을 가져왔습니다. 본 논문에서는 RoboPAIR이라는 알고리즘을 소개하며, 이는 LLM에 의해 제어되는 로봇을 위한 최초의 jailbreak 공격 알고리즘입니다.

- **Technical Details**: RoboPAIR는 세 가지 시나리오에서 LLM 제어 로봇이 해로운 물리적 행동(harmful physical actions)을 유도할 수 있는 방법을 실험적으로 입증합니다: (i) 화이트박스(white-box) 설정 - 공격자가 NVIDIA Dolphins 자율주행 LLM에 완전 접근할 수 있는 경우, (ii) 그레이박스(gray-box) 설정 - 공격자가 GPT-4o 플래너가 장착된 Clearpath Robotics Jackal UGV 로봇에 부분적으로 접근할 수 있는 경우, (iii) 블랙박스(black-box) 설정 - 공격자가 GPT-3.5 통합된 Unitree Robotics Go2 로봇 개에 대해 쿼리만 할 수 있는 경우.

- **Performance Highlights**: RoboPAIR는 세 가지 새로운 해로운 로봇 행동 데이터셋에서 공격 성공률(attack success rate) 100%에 도달하며, 기존 정적 기반선(static baselines)보다 빠르고 효과적으로 jailbreak을 발견하는 성과를 보였습니다. 이는 LLM이 텍스트 생성에 국한되지 않고 실제 세계에서 물리적 손상을 초래할 민족성이 극복되었음을 처음으로 보여줍니다.



### Diffusion Curriculum: Synthetic-to-Real Generative Curriculum Learning via Image-Guided Diffusion (https://arxiv.org/abs/2410.13674)
- **What's New**: 본 논문에서는 기존의 데이터 증강(data augmentation) 기법의 한계를 극복하기 위해, 이미지 가이드를 활용하여 합성 이미지와 실제 이미지 간의 스펙트럼 보간을 수행하는 새로운 접근 방법인 'Diffusion Curriculum (DisCL)'을 제안합니다.

- **Technical Details**: 기존의 텍스트 가이드는 합성 이미지의 품질이 원본 이미지와 어떤 연관이 있는지를 제어할 수 없지만, 이미지 가이드를 통해 합성 이미지와 실제 이미지의 유사성을 조절할 수 있습니다. DisCL은 훈련 단계에 따라 이미지 합성의 가이드 수준을 조정하여 모델을 위한 어려운 샘플을 식별하고 이들을 학습하는 데 가장 효과적인 가이드 수준을 평가합니다.

- **Performance Highlights**: DisCL을 iWildCam 데이터셋에 적용했을 때 OOD(Out-of-Distribution) 및 ID(In-Distribution) 매크로 정확도에서 각각 2.7% 및 2.1% 향상을 보여주었으며, ImageNet-LT에서 기본 모델의 tail-class 정확도를 4.4%에서 23.64%로 개선하고 모든 클래스 정확도에서 4.02% 향상을 달성했습니다.



### A new approach for fine-tuning sentence transformers for intent classification and out-of-scope detection tasks (https://arxiv.org/abs/2410.13649)
Comments:
          Appearing at Empirical Methods in Natural Language Processing 2025 - Industry Track

- **What's New**: 이번 연구는 VA(virtual assistant) 시스템에서 OOS(out-of-scope) 쿼리를 거부하거나 리다이렉트하는 방법에 대해 다룹니다. 기존 연구들은 intent classification 작업과 결합된 OOS 거부 방법을 사용했지만, 이 방법들이 종종 OOS 임베딩과 겹치는 문제가 있었습니다. 연구진은 이를 해결하기 위해 auto-encoder를 활용하여 in-scope 임베딩 재구성 손실을 도입함으로써 크로스 엔트로피 손실을 정규화하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: OOS 쿼리를 감지하는 기존 접근 방식은 주로 transformer 기반의 문장 인코더를 통해 문장 인코딩을 생성하고 이를 이용해 분류하는 방식입니다. 하지만, 이러한 방법은 크로스 엔트로피 손실을 사용하는 경우 in-scope 임베딩이 분산되어 OOS 임베딩과 겹칠 수 있습니다. 연구팀은 auto-encoder를 통해 in-scope 임베딩의 전역적인 분산을 줄이는 새로운 손실 정규화 방식을 제안하며, 이 과정을 통해 in-scope 쿼리와 OOS 쿼리를 더 잘 구분할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 OOS 인스턴스를 거부하기 위한 precision-recall curve에서 1-4%의 향상을 보여주었으며, intent classification 성능에 영향을 주지 않았습니다. 이는 VA 시스템에서 OOS 거부 기능을 효과적으로 처리할 수 있는 가능성을 제시합니다.



### SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs (https://arxiv.org/abs/2410.13648)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)에 대한 새로운 데이터셋, SimpleToM을 소개하고, LLM이 자신의 정신 상태를 추론하여 행동을 예측하고 합리성을 판단하는 능력을 평가합니다. 기존의 이론적 접근법을 넘어 실생활 시나리오에서의 응용을 탐구합니다.

- **Technical Details**: SimpleToM 데이터셋은 1147개의 간결하고 다양한 이야기와 3441개의 질문을 포함하고 있으며, 정신 상태(Mental state)와 행동(Behavior), 판단(Judgment)에 대한 질문이 포함되어 있습니다. 이러한 질문들은 LLM이 정보 인식, 행동 예측 및 행동의 적절성을 판단하는 능력을 측정합니다. 모델들은 단순한 스토리에 대한 논리적 추론을 요구받습니다.

- **Performance Highlights**: 실험 결과, 대부분의 모델이 정신 상태 예측에서는 높은 성능을 보였지만, 행동 예측 및 합리성 판단에서 낮은 성능을 보였습니다. GPT-4o 모델이 행동 예측 정확도를 49.5%에서 93.5%로, 판단 정확도를 15.3%에서 94.7%로 향상시키는 등 개입을 통해 개선할 수 있었지만, 이는 고유한 이론적 사고 능력의 한계를 나타냅니다.



### Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design (https://arxiv.org/abs/2410.13643)
- **What's New**: 이번 연구에서 제안하는 DRAKES 알고리즘은 기존의 discrete diffusion models를 활용하여 특정 작업 목표에 최적화된 시퀀스를 생성하는 데 중점을 두고 있습니다. 특히, 자연성과 고급 보상 최적화를 동시에 달성할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: DRAKES는 Gumbel-Softmax 트릭을 사용해 기존의 비미분 가능했던 경로를 미분 가능하게 만들어 전체 경로를 통한 보상의 직접 역전파를 가능하게 합니다. 이 알고리즘은 reinforcement learning (RL) 방식으로 보상 최대화 문제를 접근하며 KL divergence를 최소화하여 자연성을 유지합니다.

- **Performance Highlights**: DRAKES는 DNA 및 단백질 시퀀스를 생성하는 데 성공적으로 적용되어 각각 enhancer 활동과 단백질 안정성을 최적화한 결과, 중요한 유전자 치료 및 단백질 기반 치료에서의 활용 가능성을 보여주었습니다.



### Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation (https://arxiv.org/abs/2410.13640)
Comments:
          33 pages, 18 figures, 12 tables

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 정확도를 추정하기 위한 기존의 레이블 필요성을 없애고, 잠재 공간(latent space)에서 Chain-of-Embedding (CoE) 메서드를 제안하였습니다. 이 방식은 LLM이 스스로 출력 없는 자기 평가를 수행할 수 있도록 합니다.

- **Technical Details**: CoE는 LLM의 추론 과정에서 생성되는 모든 점진적 은닉 상태(hidden state)를 포함하며, 이는 LLM의 사고 경로(thinking path)를 나타냅니다. 연구 결과, LLM이 정답을 낼 때와 아닐 때 CoE의 특징이 다르게 나타나는 것을 확인하였으며, 이러한 차이를 통해 LLM의 응답 정확성을 추정할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험을 통해 다양한 분야(수학, 추론, 지식, 이해)에서 LLM 7종의 성과를 평가하였으며, CoE 메서드가 지연 없이 신뢰성 있는 피드백을 제공하는 것을 입증하였습니다. 또한, 이 방법은 레이블이 필요 없고, 밀리세컨드 수준의 계산 비용으로 대규모 환경에서도 실시간 피드백이 가능함을 강조합니다.



### Scaling Wearable Foundation Models (https://arxiv.org/abs/2410.13638)
- **What's New**: 이 연구에서는 165,000명 이상의 사용자로부터 수집한 4천만 시간의 다중 모달(sensor modalities) 센서 데이터를 기반으로 한 대규모 착용형 센서 모델(LSM)을 소개하고, 해당 모델의 스케일링 속성을 조사합니다. 본 연구의 주된 목표는 착용형 센서 데이터가 있는 경우 스케일링 법칙이 적용될 수 있는지를 확인하는 것입니다.

- **Technical Details**: LSM 모델은 심박수, 심박수 변동성, 전기 피부 활동(EDA), 가속도계, 피부 온도, 고도계 등 다양한 센서에서 수집된 데이터를 사용합니다. 이 연구는 데이터, 모델 크기, 컴퓨팅 리소스가 늘어날 때 LSM의 성능이 어떻게 향상되는지를 실험 엘 리 분석합니다. 자가 지도 학습(SSL) 기법을 통해 소량의 레이블 데이터 뿐만 아니라 대량의 비레이블 데이터에서 유용한 표현을 학습합니다.

- **Performance Highlights**: LSM은 데이터의 시간적 및 센서 모달리티를 초월하여 임퓨테이션(imputation), 보간(interpolation), 외삽(extrapolation) 작업을 수행할 수 있는 능력을 보여줍니다. 또한 연구는 사용자 주석 이벤트를 활용하여 운동 및 활동 인식과 같은 다운스트림 분류 작업에서의 일반화 가능성을 검증하였습니다.



### Normalizing self-supervised learning for provably reliable Change Point Detection (https://arxiv.org/abs/2410.13637)
- **What's New**: 본 논문에서는 기존의 Change Point Detection (CPD) 기법의 한계를 극복하기 위해, 전통적인 CPD 방법의 신뢰성과 표현 학습 (representation learning) 기술의 표현력을 결합하는 방안을 제안하고 있습니다. 특히, Spectral Normalization (SN)을 통해 딥러닝 모델의 데이터 표현을 최적화하고 있습니다.

- **Technical Details**: CPD 문제를 해결하기 위해, SN 기법을 사용하여 신경망의 학습에서 데이터의 변화를 표현 공간에서 유지하도록 하였습니다. 본 논문은 자기 지도 학습 (Self-Supervised Learning, SSL) 방법을 결합하여, 변화 점 탐지 (change point detection)를 위한 보다 효과적인 임베딩 (embedding) 공간을 제공합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 표준 CPD 데이터셋을 통해 평가된 결과, 현재의 최첨단 기법들보다 현저히 높은 성능을 기록하였습니다. 이는 SN을 통한 임베딩의 정보성이 CPD 활용에 매우 유익함을 보여줍니다.



### Spatiotemporal Object Detection for Improved Aerial Vehicle Detection in Traffic Monitoring (https://arxiv.org/abs/2410.13616)
Comments:
          13 pages

- **What's New**: 이번 연구는 UAV(무인 항공기) 카메라를 이용한 다중 클래스 차량 탐지의 발전을 다루며, Spatiotemporal Object Detection 모델을 개발하였습니다. 이를 위해 6,600개의 주석이 달린 연속 프레임 이미지로 구성된 Spatio-Temporal Vehicle Detection Dataset(STVD)를 소개하며, 이를 통해 알고리즘의 포괄적인 훈련과 평가를 가능하게 합니다.

- **Technical Details**: YOLO 기반 객체 탐지 알고리즘을 개선하여 시간적 동역학을 통합하였으며, 스페이셔 (spatial)와 템포럴 (temporal) 정보 모두를 활용하는 모델을 개발하였습니다. 기존의 단일 프레임 모델보다 뛰어난 성능을 발휘하며, 특히 주목(attention) 메커니즘을 통합하여 성능을 더 향상시킬 수 있음을 입증하였습니다.

- **Performance Highlights**: 실험적으로, 가장 우수한 시공간 모델이 단일 프레임 모델에 비해 16.22% 향상된 성능을 보였으며, 주목 메커니즘을 통합한 모델은 추가적인 성능 향상 가능성을 보여주었습니다.



### H2OVL-Mississippi Vision Language Models Technical Repor (https://arxiv.org/abs/2410.13611)
- **What's New**: H2OVL-Mississippi 모델은 3700만 개의 이미지-텍스트 쌍을 기반으로, 8개의 H100 GPU를 사용하여 240시간 동안 훈련된 작은 비전-언어 모델(VLM) 쌍을 소개합니다. 특히, H2OVL-Mississippi-0.8B는 8억 개의 매개변수로 구성되어 텍스트 인식에 특화되어 있으며, OCRBench의 텍스트 인식 부문에서 최첨단 성능을 발휘하고 있습니다.

- **Technical Details**: H2OVL-Mississippi 모델은 Vision Transformer(비전 트랜스포머) 구성 요소와 대형 언어 모델(LLM)로 이루어집니다. H2OVL-Mississippi-0.8B는 OCR 및 문서 중심 작업에 최적화되어 있고, H2OVL-Mississippi-2B는 다양한 멀티모달 작업을 수행할 수 있는 일반 목적 모델입니다. 이들은 각각 256에서 1590개의 시각적 토큰을 생성하며, 동적 해상도 전략(dynamic resolution)과 다중 스케일 적응 크롭(multi-scale adaptive cropping) 전략을 활용하여 다양한 이미지 크기와 종횡비에 적응합니다.

- **Performance Highlights**: H2OVL-Mississippi-0.8B는 OCRBench에서 텍스트 인식 부문에서 최첨단 성능을 보여주며, H2OVL-Mississippi-2B는 다양한 학술 벤치마크에서 경쟁력 있는 메트릭스를 제공합니다. 두 모델 모두 H2O-Danube 언어 모델의 기능을 확장하여 비주얼 도메인으로의 적용 가능성을 높이고, Apache 2.0 라이선스 하에 공개되어 문서 AI와 비주얼 LLM의 접근성을 높였습니다.



### Large Language Models as Narrative-Driven Recommenders (https://arxiv.org/abs/2410.13604)
Comments:
          Under review; 19 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 사용하여 자유형식의 텍스트로 표현된 영화 추천 요청에 대한 개인화된 추천을 제공하기 위한 새로운 접근 방식을 탐구하였습니다. 특히, reddit의 영화 추천 커뮤니티에서 수집된 데이터셋을 활용하여 38개의 오픈소스 및 클로즈드 소스 LLM의 성능을 비교하였습니다.

- **Technical Details**: 이 연구는 zero-shot, identity, few-shot 프롬프트 기법을 사용하여 LLM이 사용자 요청을 자연어로 처리하고 관련 영화를 추천할 수 있는지 평가합니다. 평가된 LLM은 크기에 따라 분류되며, 각 모델은 기본적인 zero-shot 프롬프트를 통해 추천 정확도를 높일 수 있음을 보여줍니다.

- **Performance Highlights**: LLMs는 기존의 추천 알고리즘보다 더 높은 성능을 보이며, 특히 GPT-4o는 기본 성능보다 70% 더 높은 추천 성능을 보였습니다. 중간 크기의 오픈소스 모델도 상대적으로 높은 성능을 유지하며 클로즈드 소스 모델과 비교하여 경쟁력을 보여주었습니다.



### Text-Guided Multi-Property Molecular Optimization with a Diffusion Language Mod (https://arxiv.org/abs/2410.13597)
- **What's New**: 이 논문은 약물 발견에서 필수적인 단계인 분자 최적화(molecular optimization, MO)를 위한 새로운 접근법인 Transformer 기반 확산 언어 모델(TransDLM)을 제안합니다. 기존 MO 방법은 주로 외부 속성 예측자(property predictors)에 의존했으나, 이는 예측 과정에서 오류와 노이즈를 초래합니다. TransDLM은 표준화된 화학 명명을 활용하여 오류 전파를 방지하며 동시에 여러 속성을 최적화합니다.

- **Technical Details**: TransDLM은 분자의 SMILES 문자열의 단어 벡터를 생성하기 위해 확산 모델을 활용하며, 언어 설명에 의해 안내됩니다. 이는 원하는 분자 속성을 언어 모델을 통해 암시적으로 통합하여 확산 과정에서의 오류 전파를 방지합니다. 또한, TransDLM은 웹 기반 플랫폼에서 배포가 가능하여 분산 환경에서 대규모 MO를 지원합니다.

- **Performance Highlights**: TransDLM은 벤치마크 데이터세트에서 기존의 최첨단 방법들을 초월하는 성능을 보이며, 3가지 ADMET 속성(LogD, Solubility, Clearance)을 최적화하는 데 있어 구조적 유사성을 유지하면서 개선된 화학적 특성을 제공합니다.



### OAH-Net: A Deep Neural Network for Hologram Reconstruction of Off-axis Digital Holographic Microscop (https://arxiv.org/abs/2410.13592)
Comments:
          11 pages, 4 figures

- **What's New**: 본 논문은 심층 학습(Deep Learning)과 비오프축 홀로그램(off-axis holography)의 물리적 원리를 통합한 새로운 홀로그램 재구성 방법을 제안합니다. 이 방식을 통해 홀로그램 재구성의 속도를 크게 향상시킬 수 있으며, 이는 고속 이미징 기술에 필수적입니다.

- **Technical Details**: OAH-Net(Off-Axis Hologram Network)은 3차원 이미지를 국가연구소의 혈액 샘플로부터 학습하여 재구성을 수행합니다. 본 네트워크는 Fourier Imager Heads (FIHs) 및 Complex Valued Network (CVN) 두 가지 주요 모듈로 구성되어 있으며, FIH는 불필요한 홀로그램 성분을 제거하고, CVN은 물체 빔의 파동을 진폭과 비위상으로 변환합니다. 이 모델은 3ms/프레임으로 현실적 처리 속도를 달성하였습니다.

- **Performance Highlights**: OAH-Net은 방해 요소를 최소화하며, 낮은 재구성 오류를 보였습니다. OAH-Net으로 재구성된 이미지의 YOLO(object detection) 성능은 기존 방법과 유사한 성능을 보여 임상적 유용성을 확인했습니다. 또한, 이 모델은 실시간 고해상도 홀로그램 분석을 가능하게 함으로써 생물학적 및 의료 연구 분야에서의 응용 가능성을 크게 확대하였습니다.



### RGB to Hyperspectral: Spectral Reconstruction for Enhanced Surgical Imaging (https://arxiv.org/abs/2410.13570)
Comments:
          10 pages, 4 figures, 3 tables

- **What's New**: 이번 연구는 RGB 데이터를 이용하여 하이퍼스펙트럴 서명을 재구성하여 외과 수술 이미징을 향상시키는 방법을 다룹니다. 공개된 HeiPorSPECTRAL 데이터셋과 자체 신경외과 데이터셋을 활용하여 다양한 CNN(Convolutional Neural Networks)과 Transformer 모델의 성능을 비교하고 평가하였습니다.

- **Technical Details**: 연구에서 사용된 모델은 하이퍼스펙트럼 데이터의 정확한 예측을 위해 공간 정보를 효과적으로 통합하는 Transformer 모델입니다. 성능 평가는 RMSE(Root Mean Square Error), SAM(Spectral Angle Mapper), PSNR(Peak Signal-to-Noise Ratio) 및 SSIM(Structural Similarity Index)과 같은 포괄적인 측정을 통해 이루어졌습니다. 모델은 가시광선과 확장된 스펙트럼 범위를 모두 포함한 스펙트럼 프로파일을 예측하는 데 성공하였습니다.

- **Performance Highlights**: Transformer 모델은 평가 지표에서 우수한 성능을 보이며, 질적 평가를 통해 외과적 의사결정에 중요한 스펙트럴 프로파일 예측 능력을 나타냈습니다. 그러나 가시광선과 확장된 하이퍼스펙트럴 범위를 모두 캡처하는 데 있어 MAE(Mean Absolute Error)를 통해 강조된 복잡한 과제가 있었습니다.



### CCUP: A Controllable Synthetic Data Generation Pipeline for Pretraining Cloth-Changing Person Re-Identification Models (https://arxiv.org/abs/2410.13567)
- **What's New**: 본 논문에서는 Cloth-changing person re-identification (CC-ReID) 문제를 해결하기 위해, 고품질의 합성 데이터 생성 파이프라인을 제안했습니다. 특히, 6,000개의 개인 ID와 1,179,976개의 이미지로 구성된 새로운 자가 주석 CC-ReID 데이터셋인 Cloth-Changing Unreal Person (CCUP)을 구축하여 기존의 데이터 드리븐 모델의 한계를 극복하고자 했습니다.

- **Technical Details**: CCUP 데이터셋은 현실적 인물 모델과 다양한 시나리오를 통해 대규모 합성 데이터를 생성하는 과정을 포함합니다. 데이터 생성 과정에서는 MakeHuman 소프트웨어를 사용하여 리얼한 인체 스켈레탈 메쉬를 생성하고, Unreal Engine을 통해 다양한 감시 환경에서의 시뮬레이션을 수행했습니다. 이 방식으로, 각 개인은 평균 26.5벌의 의상을 착용하며, CCTV 카메라를 통해 자동으로 레이블링된 데이터가 생성됩니다.

- **Performance Highlights**: 제안된 CCUP 데이터셋을 기반으로 한 프리트레인-파인튜닝(pretrain-finetune) 프레임워크는 TransReID와 FIRe^2와 같은 전통적인 모델의 일반화 능력을 개선하는 데 기여합니다. 실험 결과, CCUP에서 프리트레인 되고 각 벤치마크(PRCC, VC-Clothes, NKUP)에서 파인튜닝된 두 가지 모델이 다른 최신 모델들을 능가하는 성능을 보여주었습니다.



### Integrating Temporal Representations for Dynamic Memory Retrieval and Management in Large Language Models (https://arxiv.org/abs/2410.13553)
- **What's New**: 본 논문에서는 SynapticRAG라는 새로운 메모리 회수 접근 방식을 제안합니다. 이 방법은 생물학적 시냅스를 모방하여 발생 시간에 따른 사건 차별화를 통해 메모리 벡터에 시냅틱 다이나믹스를 통합합니다. 또한, 메모리의 중요성을 동적으로 업데이트하며, 전통적인 RAG 모델에 비해 최대 14.66%의 향상된 메모리 회수 정확도를 보입니다.

- **Technical Details**: SynapticRAG는 메모리 벡터에 시간적 표현을 통합하여 각 메모리 노드의 의미적 및 시간적 관계를 형성합니다. 이 모델은 동적 시간 왜곡(DTW) 방식을 사용하여 메모리 노드 간의 누적 거리 행렬을 계산하고, 자극 전파 메커니즘을 통해 노드 간의 자극을 조절합니다. LIF(Leaky Integrate-and-Fire) 모델을 기반으로 하는 시냅틱 전파 통제 메커니즘도 포함되어 있습니다.

- **Performance Highlights**: 영어, 일본어, 중국어 데이터셋에서의 실험 결과, SynapticRAG는 기존 방법들에 비해 우수한 성능을 보이며, 특히 메모리 회수 정확도에서 향상을 나타냈습니다. 이 모델의 도입으로 AI 대화 에이전트의 맥락 인식 능력이 크게 향상되었습니다.



### Can Medical Vision-Language Pre-training Succeed with Purely Synthetic Data? (https://arxiv.org/abs/2410.13523)
Comments:
          Under Review

- **What's New**: 의료 이미지 분석에 대한 기존의 MedVLP 모델이 실제 데이터에 의존하는 반면, 이 연구는 고품질의 합성 데이터(Synthetic Data)를 사용하여 모델을 학습시키는 방안을 제안합니다. 특히, 현실 데이터에 비해 합성 데이터만으로 훈련된 MedVLP 모델이 매력적인 성과를 보였습니다.

- **Technical Details**: 연구에서는 off-the-shelf generative models를 사용하여 200,000개의 합성 X-ray 이미지와 보고서를 포함하는 SynCXR 데이터셋을 생성했습니다. 이 데이터셋은 데이터 품질과 분포를 조절하여 구축되었습니다. 제안된 자동화된 파이프라인을 통해 생성된 합성 데이터만으로 훈련한 MedVLP 모델은 실제 데이터로 훈련한 모델보다 AUC에서 평균 3.8% 개선된 성능을 보였습니다.

- **Performance Highlights**: 합성 데이터 또는 혼합(data mixing) 데이터를 사용하여 훈련된 MedVLP 모델은 실제 데이터에서 훈련된 모델보다 일관되게 우수한 성과를 나타냅니다. 특히 zero-shot classification과 segmentation에서 강력한 성능을 발휘하며, 특정 영역에서는 성능이 9.07% 향상되기도 했습니다.



### Bias in the Mirror : Are LLMs opinions robust to their own adversarial attacks ? (https://arxiv.org/abs/2410.13517)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 편향(bias)의 강도와 유연성을 평가하기 위한 새로운 접근 방식을 제시합니다. 두 개의 LLM 인스턴스가 자기 논쟁(self-debate)을 통해 상반된 관점을 주장하며 중립적인 모델을 설득하려고 시도합니다. 이를 통해 편향이 정보 왜곡(misinformation)을 강화하거나 해로운 관점으로 이동할 수 있는지 여부를 탐구합니다.

- **Technical Details**: 이 연구는 다양한 크기, 출처 및 언어의 여러 LLM을 대상으로 진행된 실험으로, 편향의 지속성과 언어적 맥락에서의 유연성을 평가합니다. 특히, 언어에 따라 어떻게 편향이 표현되는지를 분석하며, LLM들이 이차 언어(secondary languages)에서 서로 다른 편향을 보이는 현상을 관찰합니다. 또한, LLM의 응답과 인간의 반응을 비교하기 위해 포괄적인 인간 평가를 수행합니다.

- **Performance Highlights**: 논문은 다양한 LLM에 대한 포괄적인 분석을 통해 모델들이 편향에 대한 입장을 어떻게 변경할 수 있는지를 보여줍니다. 기존의 연구보다 더욱 세분화된 방식으로 편향의 평가가 진행되어, LLM의 판단력이 인간의 판단과 어떻게 일치 또는 불일치하는지를 조명합니다. 이 연구는 오해의 소지가 있는 주장을 어떻게 다루는지에 대한 더 나은 이해를 제공합니다.



### MathGAP: Out-of-Distribution Evaluation on Problems with Arbitrarily Complex Proofs (https://arxiv.org/abs/2410.13502)
Comments:
          Preprint

- **What's New**: MathGAP이라는 새로운 평가 프레임워크를 제시하여, 보다 복잡한 산술 증명이 포함된 문제에서 대형 언어 모델(LLMs)의 일반화 능력을 분석합니다. 이를 통해 기존의 평가 방법의 한계를 극복하고 보다 체계적인 연구를 가능하게 합니다.

- **Technical Details**: MathGAP는 고정된 증명 기준을 따르는 문제를 생성하고 체계적인 체인-오브-생각(chain-of-thought) 주석을 제공합니다. 이 프레임워크는 증명 나무(proof trees)를 기반으로 각 문제의 복잡성을 특성화하고, 간단한 예제를 사용하여 더 복잡한 문제를 해결할 수 있는 LLM의 능력을 평가합니다.

- **Performance Highlights**: 모델 성능은 증명 깊이와 너비가 증가함에 따라 일관되게 감소하며, 특히 비선형(nonlinear) 문제에서 눈에 띄는 감소가 관찰됩니다. 흥미롭게도, 테스트 세트와 동일한 분포의 예제를 제공하는 것이 항상 성능에 이롭지 않으며, 다양한 복잡성을 가진 예제를 제시하는 것이 더 유효한 경우가 많습니다.



### Enhancing Text Generation in Joint NLG/NLU Learning Through Curriculum Learning, Semi-Supervised Training, and Advanced Optimization Techniques (https://arxiv.org/abs/2410.13498)
- **What's New**: 본 연구에서는 Joint Natural Language Generation (NLG)와 Natural Language Understanding (NLU) 학습 상황에서 텍스트 생성을 개선하기 위한 새로운 접근 방식을 개발했습니다.

- **Technical Details**: 데이터는 주석이 달린 데이터셋을 수집하고 전처리하여 준비하였으며, 여기에는 데이터 정리(cleaning), 토큰화(tokenization), 형태소 분석(stemming), 불용어 제거(stop-word removal)가 포함됩니다. 또한, 특징 추출 기법으로는 POS tagging, Bag of Words, Term Frequency-Inverse Document Frequency (TF-IDF)가 사용됩니다. Transformer 기반의 인코더와 디코더는 긴 범위의 의존성을 포착하고 소스-타겟 시퀀스 모델링을 개선합니다. Optimized BERT와 Hybrid Redfox Artificial Hummingbird Algorithm (HRAHA)와 같은 사전 훈련된 언어 모델이 통합되었습니다.

- **Performance Highlights**: 정책 경량화 기법을 통한 강화 학습(reinforcement learning), 반지도 학습(semi-supervised training), 개선된 주의 메커니즘(attention mechanisms) 및 미분 가능한 근사치(differentiable approximations)를 사용하여 모델을 미세 조정하고 복잡한 언어 과제를 효과적으로 처리합니다. 이 모델은 Python을 사용하여 구현되었습니다.



### Seeing Through VisualBERT: A Causal Adventure on Memetic Landscapes (https://arxiv.org/abs/2410.13488)
Comments:
          Accepted at EMNLP Findings 2024

- **What's New**: 본 논문은 Structural Causal Model (SCM)을 기반으로 한 새로운 프레임워크를 제안하여, offensive memes의 탐지에서 투명성을 높이며 모형의 행동 해석을 가능하게 한다. VisualBERT를 활용하여 meme 입력과 causal concepts를 모두 고려하여 클래스를 예측한다.

- **Technical Details**: 이 프레임워크는 기존 interpretability 기술의 한계를 극복하기 위해 causal concepts를 통합하고, dynamic routing과 adversarial learning을 활용하여 meme의 공격성을 예측한다. 아울러, 모델 예측의 원인과 오류 사례를 명확히 설명한다.

- **Performance Highlights**: 정량적 분석 결과, 제안한 모델링 기법들이 기존의 input attribution 방법들과 비교하여 causality를 만족하지 못하는 점을 강조하며, 이로 인해 safety-critical applications에서의 신뢰성에 의문을 제기한다. 또한, qualitative 분석을 통해 모델의 결정이 정당화될 수 있는지를 평가하였다.



### Breaking the Manual Annotation Bottleneck: Creating a Comprehensive Legal Case Criticality Dataset through Semi-Automated Labeling (https://arxiv.org/abs/2410.13460)
- **What's New**: 이 논문에서는 스위스 연방 대법원 판결의 미래 법리에 대한 영향을 평가하기 위한 새로운 Criticality Prediction (사례 중요도 예측) 데이터셋을 소개합니다. 기존의 수작업 주석 접근 방식과 달리, 본 데이터셋은 반자동적으로 레이블을 유도하여 훨씬 더 큰 데이터셋을 제공합니다.

- **Technical Details**: 제안된 데이터셋은 2단계 레이블링 시스템을 특징으로 하며, (1) LD-Label: 주요 결정으로 발표된 사례를 식별하는 이진 지표, (2) Citation-Label: 사례의 인용 빈도와 최근성에 따라 사례를 평가합니다. 이 데이터셋은 2002년부터 2023년까지의 사례를 포함하며 언어는 독일어, 프랑스어, 이탈리아어로 구성됩니다.

- **Performance Highlights**: 여러 다국어 모델을 평가한 결과, 세밀하게 조정된 모델이 제로샷(Zero-shot) 기준선보다 일관되게 우수한 성능을 보였습니다. 이를 통해 작업 특화적 적응이 필요함을 입증하였습니다.



### Unlocking Legal Knowledge: A Multilingual Dataset for Judicial Summarization in Switzerland (https://arxiv.org/abs/2410.13456)
- **What's New**: 이번 논문은 스위스 연방 대법원(SFSC)의 판결을 바탕으로 한 새로운 데이터셋인 Swiss Leading Decision Summarization (SLDS)를 소개하고 있습니다. 이 데이터셋은 독일어, 프랑스어 및 이탈리아어로 된 18,000개의 법원 판결과 독일어 요약을 포함하고 있어, 다국어 법적 요약에 대한 연구를 촉진할 수 있습니다.

- **Technical Details**: 논문에서는 3가지 mT5(multi-lingual T5) 변형 모델과 고유 모델을 미세 조정(fine-tuning)하고 평가했습니다. 분석 결과, 고유 모델은 제로샷(zero-shot) 및 원샷(one-shot) 설정에서 우수한 성능을 보였으나, 미세 조정된 모델이 여전히 강력한 경쟁력을 유지하는 것으로 나타났습니다.

- **Performance Highlights**: SLDS 데이터셋의 공개를 통해 법적 요약 및 법무 전문가를 위한 보조 기술 개발에 대한 추가 연구가 촉진될 것으로 기대됩니다. 이 데이터셋은 수백만 건의 판결을 법률 연구에 더 쉽게 접근할 수 있게 할 잠재력이 있습니다.



### Parameter-efficient Adaptation of Multilingual Multimodal Models for Low-resource ASR (https://arxiv.org/abs/2410.13445)
- **What's New**: 본 연구에서는 낮은 자원을 가진 언어에 대한 자동 음성 인식(ASR)을 개선하기 위해, 다국어 다중 모달 모델인 SeamlessM4T를 활용하여 매개변수 효율적인 미세 조정(parameter-efficient fine-tuning) 및 텍스트 전용 적응(text-only adaptation) 기술을 결합한 방법을 제시합니다.

- **Technical Details**: SeamlessM4T는 다중 언어 및 다중 모달 머신 번역 지원을 제공하는 엔드-투-엔드 모델로, 96개 언어의 입력 및 출력에 대해 자동 음성 인식, 텍스트-음성 변환 등 여러 작업을 수행할 수 있습니다. 이 모델은 셀프-슈퍼바이즈드(self-supervised) 방식으로 백만 시간 이상의 무 라벨 음성을 학습하여 성능이 개선되었습니다.

- **Performance Highlights**: 본 논문에서는 높은 자원 언어에서 낮은 자원 언어로의 언어 간 전이(cross-lingual transfer)를 통해, 라벨이 없는 음성 데이터 없이도 WER(Word Error Rate)를 17% 이상 감소시킬 수 있음을 보였습니다.



### Solving Prior Distribution Mismatch in Diffusion Models via Optimal Transpor (https://arxiv.org/abs/2410.13431)
- **What's New**: 이 논문에서는 확산 모델(Diffusion Models, DMs)과 최적 운송 이론(Optimal Transport, OT) 사이의 깊은 관계를 탐구하며, DMs에서 발생하는 prior error를 제거하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문은 DMs이 시간 의존적인 OT 계산 방법으로 구성되어 있음을 보여주고, 전통적인 확산 모델에서 발생하는 prior error가 잠재적인 공백을 초래한다는 것을 이론적으로 분석합니다. 또한, 확산 종료 시간이 증가함에 따라 확률 흐름이 Monge-Ampère 방정식의 해결책의 기울기로 기하급수적으로 수렴한다는 것을 증명합니다.

- **Performance Highlights**: 다양한 이미지 데이터세트에서의 실험 결과는 제안한 접근 방식의 효과성을 입증하며, 특히 조건부 및 비조건부 생성 상황에 대한 샘플링 가속화에 자연스럽게 확장될 수 있음을 보여줍니다.



### Shavette: Low Power Neural Network Acceleration via Algorithm-level Error Detection and Undervolting (https://arxiv.org/abs/2410.13415)
- **What's New**: 본 논문은 Deep Neural Network(DNN) 가속기의 전압 저하 운영을 소프트웨어 수정만으로 가능하게 하는 간단한 접근 방식을 소개합니다. 기존의 기술들, 특히 Timing Error Detection(TED) 시스템은 상당한 개발 비용과 오버헤드를 발생시키며, 상용 부품에서는 적용이 어려운 반면, 본 논문에서는 알고리즘 기반의 오류 탐지를 통해 이러한 문제를 해결합니다.

- **Technical Details**: 제안된 접근 방식은 Algorithm Based Fault Tolerance(ABFT) 오류 발견 메커니즘을 통해 이루어집니다. 이를 통해 DNN 모델의 연산을 전환하면서, 전압 마진을 제거하고 가장 에너지 효율적인 전압-주파수(V-F) 조합으로 운영합니다. 실험 결과, LeNet과 VGG16을 GPU 플랫폼에서 실험한 결과, 에너지 소비를 18%에서 25% 절약할 수 있음을 보여주었습니다.

- **Performance Highlights**: 제안된 알고리즘은 정확도 손실 없이 모델의 에너지 소비를 절감하며, 처리량 저하도 3.9% 미만으로 유지됩니다. 기존의 TED 기반 기술과 비교했을 때, 본 접근법은 회로 수정 없이 낮은 개발 비용으로 구현이 가능하다는 장점이 있습니다.



### Think Thrice Before You Act: Progressive Thought Refinement in Large Language Models (https://arxiv.org/abs/2410.13413)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 응답 품질을 향상시키기 위한 새로운 방법인 Progressive Thought Refinement (PTR) 프레임워크를 제안합니다. 기존 방법들이 특정 작업에 의존하여 일반화가 어려웠던 문제를 해결하기 위해, PTR은 두 단계의 세분화된 접근 방식을 사용합니다.

- **Technical Details**: PTR는 두 가지 주요 단계인 (1) 'Thought data construction' 단계와 (2) 'Thought-Mask Fine-Tuning' 단계를 포함합니다. 첫 번째 단계에서는 약한 모델과 강한 모델 간의 협력 선택 전략을 사용해 고품질 점진적 정제 데이터셋을 구축합니다. 두 번째 단계에서는 'thought'를 마스킹하고 손실 가중치를 조정하여 LLM이 개선 방법을 암묵적으로 이해하도록 합니다. 이를 통해 LLM이 이전 사고에서 개선된 응답을 생성하도록 유도합니다.

- **Performance Highlights**: 실험 결과, PTR을 사용한 LLM은 지식 추론, 코드 생성, 수학적 추론 등 다양한 작업에서 평균 성능이 49.6%에서 53.5%로 향상되었습니다. 특히, 보다 개방된 작업에서도 정답의 질과 형식에서 유의미한 향상이 있으며, 이는 LLM이 스스로 개선할 수 있도록 훈련되었음을 나타냅니다.



### Attr-Int: A Simple and Effective Entity Alignment Framework for Heterogeneous Knowledge Graphs (https://arxiv.org/abs/2410.13409)
- **What's New**: 이 연구는 이질적인 지식 그래프 간의 개체 정렬에 대한 새로운 접근 방식을 제시합니다. 구체적으로, 기존의 개체 정렬 방법들이 비구조적 이질성으로부터 발생하는 한계를 넘어서는 두 가지 새로운 벤치마크를 소개합니다.

- **Technical Details**: 이 연구는 Attr-Int라는 새로운 개체 정렬 프레임워크를 제안하며, 이 프레임워크는 혁신적인 속성 정보 상호 작용 방법을 통합하여 기존의 구조적 임베딩 기법과 결합해 개체 정렬 성능을 향상시킵니다. 이 과정에서 속성 데이터는 별도의 임베딩 학습 없이 활용됩니다.

- **Performance Highlights**: 우리가 제안한 Attr-Int 프레임워크는 두 개의 새로운 벤치마크에서 최신 기술들을 능가하는 성능을 보여주었습니다. 이는 실제 세계의 이질적인 개체 정렬 시나리오를 모사하여 실험된 결과입니다.



### MoR: Mixture of Ranks for Low-Rank Adaptation Tuning (https://arxiv.org/abs/2410.13408)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 Mixture of Ranks (MoR)라는 새로운 접근 방식을 도입하여 LoRA의 성능을 개선하고, 다중 작업에서 효율적으로 학습할 수 있는 방법을 제안합니다. 기존의 LoRA와 MoE 방식의 한계를 극복하고, 다양한 작업에 대한 rank-specific 정보를 효과적으로 학습합니다.

- **Technical Details**: MoR은 세 가지 주요 구성 요소인 공유 전문가(shared experts), 다중 rank 적응(multi-rank adaptation), 혼합 학습(mixture learning)으로 구성됩니다. 이 방법은 여러 LoRA를 통합하여 학습할 수 있는 새로운 프레임워크를 제공하며, 매핑(mapping) 및 스케일링(scaling)을 통해 다중 작업을 수행합니다.

- **Performance Highlights**: MoR는 기존 LoRA 방법 대비 1.31%의 성능 향상을 달성하면서도 파라미터는 93.93%만 사용합니다. 또한, 다양한 실험에서 MoR은 효율성, 일반화 가능성, 확장성을 입증하며, 다중 LoRA 구조의 학습 비용을 크게 줄이고 더 간결한 정보를 동적으로 학습할 수 있음을 보여줍니다.



### Context-aware adaptive personalised recommendation: a meta-hybrid (https://arxiv.org/abs/2410.13374)
- **What's New**: 이번 논문에서는 정보를 종합할 수 있는 메타 하이브리드 추천 시스템을 제안하고 있습니다. 이 시스템은 사용자마다 최적의 추천 알고리즘을 예측하기 위해 Machine Learning을 사용할 수 있도록 개발되었습니다.

- **Technical Details**: 제안된 메타 하이브리드 추천 시스템은 사용자에 대한 맥락적 및 선호 정보를 기반으로 다양한 추천 알고리즘 중에서 최고의 성능을 발휘하는 것을 선택합니다. 오프라인 평가를 위해 MovieLens와 The Movie DB 데이터셋을 사용하였으며, 이를 통해 각 세션과 사용자에 적합한 추천기를 선택할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 메타 하이브리드 추천 시스템은 정상화된 Discounted Gain과 Root Mean Square Error 메트릭에서 기존의 개별 접근 방식보다 20-50% 더 나은 성능을 보였습니다. 그러나 사용자의 표준 정보 기반으로 최적 성능을 달성하기란 어려운 과제입니다.



### MagicTailor: Component-Controllable Personalization in Text-to-Image Diffusion Models (https://arxiv.org/abs/2410.13370)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 텍스트-이미지(T2I) 모델의 개인화 작업에서 구성 요소를 제어할 수 있는 새로운 접근법을 제시합니다. 사용자가 특정 시각적 개념의 개별 요소를 재구성할 수 있도록 하여 더 정밀한 커스터마이징을 가능하게 합니다. 

- **Technical Details**: 새로운 프레임워크인 MagicTailor를 통해 Dynamic Masked Degradation (DM-Deg)과 Dual-Stream Balancing (DS-Bal)을 활용하여 개인화 과정에서의 의미적 오염(semantic pollution)을 줄이고, 의미적 불균형(semantic imbalance)을 관리합니다. 이 프레임워크는 T2I 모델을 동적으로 조정하여 개인화된 개념을 정확하게 반영합니다.

- **Performance Highlights**: MagicTailor는 실험을 통해 구성 요소 제어가 가능한 개인화 작업에서 최첨단(SOTA) 성과를 달성하였으며, 다양한 추가 응용 가능성을 보여줍니다.



### Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistan (https://arxiv.org/abs/2410.13360)
- **What's New**: 본 논문에서는 Retrieval Augmented Personalization (RAP) 프레임워크를 소개하여 다중 모드 대형 언어 모델(MLLMs)의 개인화를 가능하게 합니다. RAP는 일반 MLLM을 개인화된 어시스턴트로 전환하는 세 가지 주요 단계로 구성됩니다: 기억(Recall), 검색(Retrieve), 생성(Generate).

- **Technical Details**: RAP는 사용자 관련 정보(예: 이름, 아바타 등)를 저장하는 키-값 데이터베이스를 설계합니다. 사용자가 대화를 시작할 때, RAP는 다중 모드 검색기를 통해 데이터베이스에서 관련 정보를 검색하고, 이를 MLLM에 입력하여 개인화된 지식 강화 응답을 생성합니다. 추가로 생성 품질 향상을 위해 데이터 수집 파이프라인을 개발하고 개인화된 훈련을 위한 전문적인 데이터셋을 생성합니다.

- **Performance Highlights**: RAP-MLLMs는 개인화된 이미지 캡션 작성, 질문 응답 및 시각적 인식과 같은 다양한 작업에서 뛰어난 유연성과 생성 품질을 보여줍니다. 모델들은 무한한 시각적 개념에 대해 일반화 능력을 발휘하며, 사용자 관련 정보를 효과적으로 처리하여 개인화된 출력을 제공합니다.



### LAR-ECHR: A New Legal Argument Reasoning Task and Dataset for Cases of the European Court of Human Rights (https://arxiv.org/abs/2410.13352)
Comments:
          Published in Natural Legal Language Processing (NLLP) 2024 workshop

- **What's New**: 논문에서는 Legal Argument Reasoning (LAR)이라는 새로운 과제를 소개하고, 이를 통해 대형 언어 모델(LLMs)의 법적 추론 능력을 평가하고자 합니다. 이 과제는 법원의 주장 연쇄에서 사건의 사실을 기준으로 올바른 다음 진술을 선택하는 것을 요구합니다.

- **Technical Details**: 우리는 LAR-ECHR라는 데이터셋을 유럽 인권 재판소(ECHR)의 사례를 통해 구축했습니다. 이 데이터셋은 403개의 사례로 구성되어 있으며, LLM이 제공된 사례의 사실 및 이전의 법적 주장과 함께 다음 진술을 선택하게 됩니다. LAR는 단순한 패턴 인식이나 암기 이상의 법적 및 상식적 추론을 요구합니다.

- **Performance Highlights**: 실험 결과, 최상위 모델(GPT-4o)은 LAR-ECHR에서 75.8%의 정확도를 기록했습니다. 이는 LegalBench에서의 최고 정확도(73.3%)와 유사하며, 모델 개선의 가능성이 여전히 크다는 것을 나타냅니다. 다양한 법적 시스템에 대해 LAR 데이터셋을 구축할 수 있는 방법도 제시되어 있습니다.



### Representation Learning of Structured Data for Medical Foundation Models (https://arxiv.org/abs/2410.13351)
Comments:
          NeurIPS 2024 Workshop on Unifying Representations in Neural Models (UniReps 2024)

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 의료 분야에서 비문자적 구조 데이터, 특히 ICD-10 및 SNOMED-CT와 같은 의료 코드를 효과적으로 처리하는 데 직면한 문제들을 해결하기 위한 UniStruct 아키텍처를 제안합니다.

- **Technical Details**: UniStruct는 비구조화된 텍스트와 구조화된 데이터를 결합한 다중 모달(multi-modal) 의료 모델을 설계하고, 의료 코드에 적합하도록 하위 단어(tokenization) 기법을 조정하여 이러한 문제를 해결합니다. 아이디어는 자주 함께 발생하는 의료 코드 그룹을 단일 토큰으로 처리하는 것입니다.

- **Performance Highlights**: 내부 의료 데이터베이스에서 10억 개 이상의 토큰으로 사전 훈련된 UniStruct 모델은 평가 메트릭에서 최대 23% 개선을 달성했으며, EHRSHOT 공공 벤치마크에서 1/1000의 사전 훈련 데이터에도 불구하고 42% 이상의 하위 작업에서 성능을 개선했습니다.



### Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancemen (https://arxiv.org/abs/2410.13344)
- **What's New**: 이 논문에서는 Cerberus라는 새로운 adaptive parallel decoding framework를 제안합니다. Cerberus는 각 decoding step에서 적절한 decoding 방식을 선택하는 gating 메커니즘을 도입하고, sequential knowledge를 활용하는 새로운 decoding heads 방식을 소개하여 inference의 효율성을 향상시킵니다.

- **Technical Details**: Cerberus는 두 가지 주요 구성 요소를 포함하고 있습니다: 1) Sequential knowledge를 통합하여 prediction accuracy를 개선하는 novel decoding heads인 Cerberus Heads, 2) 모델의 confidence level을 기반으로 auto-regressive decoding과 parallel decoding을 선택하는 entropy-based gating mechanism. 이 방법들은 decoding 과정 중 overhead를 줄이고, 전체 inference 효율성을 높입니다.

- **Performance Highlights**: Cerberus는 MT-Bench에서 auto-regressive decoding 대비 최대 2.12배의 속도를 달성하였으며, Medusa보다 10% - 30% 더 빠르고, 더 우수한 generation quality를 보여주었습니다.



### DART: Disentanglement of Accent and Speaker Representation in Multispeaker Text-to-Speech (https://arxiv.org/abs/2410.13342)
Comments:
          Accepted in Audio Imagination workshop of NeurIPS 2024

- **What's New**: 최근 Text-to-Speech (TTS) 시스템에서 자연스럽고 표현력이 풍부한 음성을 생성할 수 있는 발전이 있었습니다. 이 논문에서는 다국적 사용자를 위해 발음과 화자 식별을 효과적으로 분리하는 새로운 접근법을 제안합니다.

- **Technical Details**: 본 연구에서는 Multi-Level Variational Autoencoders (ML-VAE)와 Vector Quantization (VQ)를 사용하여 화자와 발음 표현을 분리하는 방법을 제안합니다. ML-VAE는 계층적으로 데이터의 공동 분포를 모델링하기 위해 구성되며, 이를 통해 화자와 발음의 특성을 명확하게 분리합니다.

- **Performance Highlights**: 제안된 방법은 다양한 발음화된 음성 데이터에 대해 실험하여 효과성을 입증하였으며, 사용자 맞춤형 음성 합성을 가능하게 합니다. 코드와 음성 샘플은 공개되어 활용 가능합니다.



### DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbon (https://arxiv.org/abs/2410.13338)
Comments:
          25 pages, 14 figures

- **What's New**: 이번 논문에서는 확률적 시계열 보간 기술을 위한 새로운 프레임워크인 DiffImp를 제안합니다. DiffImp는 시계열 데이터의 복잡한 분포를 모델링할 수 있는 능력으로 Denoising Diffusion Probabilistic Models (DDPMs)를 활용합니다.

- **Technical Details**: DiffImp는 효율적인 State Space Model (SSM)인 Mamba를 backbone으로 통합하여 denoising 모듈을 구성합니다. 이를 통해 낮은 시간 복잡도로 시퀀스 모델링이 가능하며, Bidirectional Attention Mamba block (BAM)을 통해 양방향 종속성을 처리하고, Channel Mamba Block (CMB)을 통해 서로 다른 채널 간의 의존성을 모델링합니다.

- **Performance Highlights**: DiffImp는 다양한 데이터 세트를 사용한 실험에서 최첨단 성능을 달성하였으며, 다양한 결측 시나리오와 결측 비율에서 우수한 결과를 보여주었습니다.



### Do LLMs Have Political Correctness? Analyzing Ethical Biases and Jailbreak Vulnerabilities in AI Systems (https://arxiv.org/abs/2410.13334)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)에서 안전성을 위해 주입된 의도적 편향이 어떻게 유해한 콘텐츠 생성을 초래할 수 있는지를 탐구하고, 새로운 패러다임인 PCJailbreak와 간단한 방어 전략인 PCDefense를 제안합니다.

- **Technical Details**: LLM의 안전성을 확보하기 위한 다양한 방법(데이터 필터링, 지도 학습 피드백 등)은 보통 의도적인 편향을 불러오며, 이는 'jailbreak' 현상을 초래합니다. PCJailbreak는 이러한 편향을 이용하여 LLM이 유해한 출력을 생성할 수 있도록 조작하는 공격이며, PCDefense는 공격 전에 방어 프롬프트를 주입하여 이를 방지하는 방법입니다.

- **Performance Highlights**: PCJailbreak는 최신 GPT 모델에서도 효과적이며, 제안하는 PCDefense 방법은 추가적인 추론 비용 없이도 jailbreak 공격을 완화할 수 있음을_showcase합니다. 이 연구는 LLM 제공업체들이 보다 책임감 있게 안전성을 설계하고 구현해야 함을 강조합니다.



### Improving Discrete Optimisation Via Decoupled Straight-Through Gumbel-Softmax (https://arxiv.org/abs/2410.13331)
- **What's New**: 이 논문에서는 Straight-Through Gumbel-Softmax (ST-GS) 추정기를 개량한 'Decoupled ST-GS'를 제안합니다. 이 방법은 순방향(forward)과 역방향(backward) 패스를 위한 온도(temperature) 매개변수를 분리하여 gradient의 신뢰성과 모델 성능 간의 균형을 개선합니다.

- **Technical Details**: Decoupled ST-GS은 ST-GS 추정기의 구조를 바탕으로 하며, 온도 조정에서 수반되는 어려움을 해결하기 위해 각각의 패스에 대해 고유한 온도를 사용합니다. 이는 모델의 추론 시 스무딩(smoothing)과 훈련 시 gradient의 신뢰성을 독립적으로 조정할 수 있게 합니다.

- **Performance Highlights**: 다양한 과제(task)와 데이터 세트에서의 실험을 통해 Decoupled ST-GS가 기존 ST-GS 추정기보다 일관되게 뛰어난 성능을 보였음을 입증했습니다. 이 방법은 gradient gap과 bias-variance trade-off 분석을 통해 gradient 기반 최적화의 개선을 도모합니다.



### Computational Approaches to Arabic-English Code-Switching (https://arxiv.org/abs/2410.13318)
Comments:
          PhD thesis

- **What's New**: 이번 연구는 아랍어와 영어 간의 코드 스위칭(Code-Switching) 데이터를 활용하여 주어진 분야의 여러 NLP(Natural Language Processing) 작업, 특히 Named Entity Recognition(NER) 작업의 문제를 다루고 있습니다. 연구는 특히 현대 표준 아랍어와 아랍어-영어 NER을 위한 첫 번째 주석이 달린 코드 스위치 아랍어-영어 코퍼스를 제작하여 이를 진행했습니다.

- **Technical Details**: 본 연구는 CS 데이터에서 NER 태거(NER tagger)를 개선하기 위해 CS 컨텍스트 임베딩(Contextual Embeddings)과 데이터 증강(Data Augmentation) 기법을 도입했습니다. 또한, 혼합 텍스트의 언어 유형을 결정하고 명명된 개체(named entity)를 식별하기 위한 여러 개의 intra-word 언어 식별 접근 방식을 제안합니다.

- **Performance Highlights**: 모든 방법이 CS 데이터에서 NER 태거의 성능을 향상시키는 결과를 보였습니다. 연구의 결과는 아랍어-영어 코드 스위칭 데이터에서 다루어진 여러 NLP 작업의 중요성과 필요성을 강조합니다.



### Precipitation Nowcasting Using Diffusion Transformer with Causal Attention (https://arxiv.org/abs/2410.13314)
- **What's New**: 이번 연구에서는 Diffusion Transformer with Causal Attention 모델을 제안하여 단기 강수 예보의 문제를 해결하고자 합니다. 이 모델은 Transformer를 활용하여, 조건 정보와 예보 결과 간의 시공간 쿼리를 효과적으로 설정할 수 있도록 합니다.

- **Technical Details**: DTCA(분산 변환기 인과 주의) 모델은 조건부 강수 분포 특징 관련 쿼리를 기반으로 한 새로운 인과 주의 메커니즘을 도입하며, 다양한 시공간 정보 상호작용을 탐색하고 그 구조를 비교합니다. 실험 결과, 전역 시공간 레이블링 상호작용이 최고의 성능을 발휘하는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 방법은 최신 기술인 U-Net 기반 방법과 비교해 강수 예측에서 약 15% 및 8% 개선된 CSI(비판적 성공 지표)를 달성하여 현재 상태의최고 성능을 기록했습니다.



### Active inference and deep generative modeling for cognitive ultrasound (https://arxiv.org/abs/2410.13310)
- **What's New**: 초음파(US) 이미징 시스템을 정보 탐색 에이전트로 재구성하여 효율적인 진단을 위한 대칭적 상호작용을 제안합니다. 이 시스템은 자율적으로 촬영을 개인화하고 현장에서 정보 획득을 극대화합니다.

- **Technical Details**: 이 연구에서는 초음파 데이터 수집과 재구성이 '지각-행동 루프(perception-action loop)'로 해석되며, Bayesian inference를 통해 불확실성을 줄이고 진단 가치를 극대화하는 방법을 설명합니다. 시스템은 생성적 모델을 활용하여 환경을 이해하고 최적의 측정을 계획합니다.

- **Performance Highlights**: 딥 생성적 모델을 통해 초음파 이미징의 품질과 진단 정확도를 크게 향상시킬 수 있는 잠재력을 보여주며, 특히 어려운 환자 군에서 지속적인 이미지 분석 및 개입이 가능하도록 합니다.



### Hiformer: Hybrid Frequency Feature Enhancement Inverted Transformer for Long-Term Wind Power Prediction (https://arxiv.org/abs/2410.13303)
- **What's New**: 본 논문에서는 기후 변화의 심각성이 증가함에 따라 재생 에너지로의 긴급한 전환이 필요하다는 점을 강조하고, 특히 풍력 에너지의 대규모 채택이 환경 영향을 완화하는 데 중요하다고 설명합니다. 이는 장기적인 풍력 예측 모델의 필요성을 강조합니다.

- **Technical Details**: 이 논문은 Hybrid Frequency Feature Enhancement Inverted Transformer (Hiformer)라는 새로운 접근 방식을 제안합니다. Hiformer는 신호 분해 기술과 날씨 특징 추출 기술을 통합하여 기상 조건과 풍력 생성 간의 상관관계를 모델링하는 독특한 구조를 가지고 있습니다. 또한 Hiformer는 인코더 전용 아키텍처를 사용하여 장기 풍력 예측에 따른 계산 복잡성을 줄입니다.

- **Performance Highlights**: Hiformer는 최신 방법과 비교하여 예측 정확도를 최대 52.5% 향상시킬 수 있으며, 계산 시간을 최대 68.5% 단축할 수 있습니다.



### Automating IETF Insights generation with AI (https://arxiv.org/abs/2410.13301)
Comments:
          5 pages plus Appendix

- **What's New**: 이 논문은 IETF Insights 프로젝트를 소개합니다. 이 시스템은 Internet Engineering Task Force (IETF) 작업 그룹의 활동에 대한 포괄적인 보고서를 자동으로 생성하여 효율화하는 기능을 가지고 있습니다.

- **Technical Details**: 시스템은 회의록(meeting minutes), 참가자 목록(participant lists), 초안(drafts) 및 의제(agendas)와 같은 다양한 IETF 소스에서 데이터를 수집, 통합 및 분석하는 기능을 포함합니다. 주요 구성 요소로는 데이터 전처리 코드(data preprocessing code)와 LaTeX 또는 Markdown 형식으로 고품질 문서를 생성하는 보고서 생성 모듈(report generation module)이 있습니다. 또한 데이터에 기반한 요약을 위한 대규모 언어 모델(large Language Models, LLMs)를 통합하여 IETF 기록의 접근성 및 유용성을 높입니다.

- **Performance Highlights**: IETF Insights 프로젝트는 IETF의 활동 및 커뮤니티에 대한 기여에 대한 귀중한 개요를 제공하며, IETF 기록의 효율적인 관리와 활용을 촉진합니다.



### LLM-Rank: A Graph Theoretical Approach to Pruning Large Language Models (https://arxiv.org/abs/2410.13299)
- **What's New**: 본 논문에서는 그래프 이론의 중심성 측정을 활용한 새로운 프루닝(pruning) 방법인 MLPRank를 제안하였습니다. 이 방법은 다층 퍼셉트론(multilayer perceptron)과 디코더 전용(transformer) 모델을 대상으로 하여 계산 요구 사항과 메모리 사용량을 줄입니다.

- **Technical Details**: MLPRank는 가중 방향 비순환 그래프(weighted directed acyclic graph)를 생성하여 각 노드의 중요도를 평가하는 데 수정된 PageRank 중심성 측정을 사용합니다. 이와 함께 균일 프루닝(uniform pruning)을 적용하여 구조적 희소성(structured sparsity)을 달성합니다. 또한, 디코더 전용 모델에 대한 확장된 방법인 LLMRank도 소개하였습니다.

- **Performance Highlights**: MLPRank는 세 가지 인기 있는 기준과 비교하여 평균 6.09%의 정확도 유지를 보여주고, LLMRank는 두 가지 기준에 비해 13.42% 높은 성능을 보였습니다. 두 방법 모두 최신 구조적 프루닝 기법인 SliceGPT보다 평균 8.60% 높은 정확도를 기록했습니다.



### Advancing Large Language Model Attribution through Self-Improving (https://arxiv.org/abs/2410.13298)
Comments:
          Accepted by EMNLP 2024 Main Conference

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에게 증거 출처를 인용하는 텍스트 생성을 가르쳐 신뢰성을 높이는 새로운 프레임워크인 START(Self-Taught AttRibuTion)를 소개합니다. 이는 인력 자원을 절약하면서도 LLM의 인용 능력을 개선합니다.

- **Technical Details**: START는 모델이 초기의 불충분한 감독 신호로 인해 정체되는 것을 막기 위해 스스로 합성 훈련 데이터를 생성하도록 유도합니다. 이후 모델의 인용 능력을 자가 개선하기 위해 선택한 응답으로부터 세부적인 선호 감독 신호를 반복적으로 활용합니다.

- **Performance Highlights**: 세 개의 오픈 도메인 질문-응답 데이터셋에서 실험을 수행한 결과, 평균 25.13%의 성능 향상을 이루었으며, 이는 인간 주석이나 더 진보된 모델에 의존하지 않고 달성되었습니다. 또한, START는 여러 출처에서 정보를 집계하는 능력이 뛰어난 것으로 나타났습니다.



### Fairness-Enhancing Ensemble Classification in Water Distribution Networks (https://arxiv.org/abs/2410.13296)
- **What's New**: 이 논문에서는 사회 경제적으로 중요한 기반 시설인 물 분배 네트워크(Water Distribution Networks, WDNs)에서 AI의 공정성 문제를 다루고 있습니다. WDNs의 공정성에 대한 기존 연구가 부족한 만큼, 이 연구는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구자들은 그룹 공정성(Group Fairness)의 개념을 도입하고, 비이진(Binomial) 민감 변수에 대한 정의를 명확히 하였습니다. 기존의 누수 탐지 방법들이 공정하지 않음을 입증하였고, 비구분 가능한 앙상블(Ensemble) 분류 방법에 적용할 수 있는 공정성을 높이기 위한 방법을 제안했습니다.

- **Performance Highlights**: 이 논문은 WDNs에서 머신러닝 모델의 공정성을 평가하기 위한 표준 방법론을 제시하고, 여러 개의 민감 변수를 고려하여 기존 방법론의 몇 가지 조정을 통해 공정성을 강화할 수 있음을 설명합니다.



### PiLocNet: Physics-informed neural network on 3D localization with rotating point spread function (https://arxiv.org/abs/2410.13295)
Comments:
          25 pages, 4 figures

- **What's New**: 이 논문은 3D localization 문제 해결을 위한 새로운 개선된 Neural Network PiLocNet을 제안합니다. PiLocNet은 기존의 LocNet을 기반으로 하며, forward-model 기반 정보와 data-fitting loss term을 통합하여 물리적으로 합리적인 결과를 도출합니다.

- **Technical Details**: PiLocNet은 Physics-Informed Neural Network (PINN)으로, 이미징 시스템의 포인트 스프레드 함수(PSF)를 통해 물리적 정보를 네트워크에 통합합니다. 세 가지 중요한 구성 요소로는 forward-model 정보, variational method의 regularization term, 그리고 Poisson 및 Gaussian noise 모델을 통한 이미지 노이즈에 대한 강건성 개선이 포함됩니다.

- **Performance Highlights**: 실험 결과, PiLocNet은 3D 포인트 소스의 localization을 위한 정확도 향상에 기여하며, precision과 recall 측면에서 개선된 예측 결과를 보여줍니다. 본 논문은 PiLocNet의 Robustness를 검증했으며, 다양한 PSF와 이미징 문제에서의 적용 가능성을 제시합니다.



### SBI-RAG: Enhancing Math Word Problem Solving for Students through Schema-Based Instruction and Retrieval-Augmented Generation (https://arxiv.org/abs/2410.13293)
Comments:
          Accepted to the 4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 본 논문에서는 Schema-Based Instruction Retrieval-Augmented Generation (SBI-RAG) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 통합하여 수학 단어 문제(MWP)를 해결하는 과정을 지원하며, 기존의 Schema-Based Instruction(SBI) 방법을 바탕으로 발전했습니다.

- **Technical Details**: SBI-RAG는 기본적으로 네 가지 주요 부분으로 나뉩니다: 1) Schema Classifier, 2) Prompt Creation, 3) Context Retrieval, 4) Answer and Response Generation. Schema Classifier는 DistilBERT를 기반으로 하여 특정 문제에 적합한 schema를 예측하고, 그에 따라 schema-specific prompt를 생성합니다. 이후 Retrieval-Augmented Generation(RAG) 프레임워크를 이용하여 관련 문서를 검색하고, LLM을 통해 구체적인 단계별 해답을 생성합니다.

- **Performance Highlights**: GSM8K 데이터셋에서의 평가 결과, SBI-RAG는 GPT-4 및 GPT-3.5 Turbo와 비교하여 문제 해결의 정확성과 추론의 명료성을 향상시키는 데 효과적임을 보였습니다. 새로운 'reasoning score' 메트릭을 도입하여 LLM의 해결 과정의 질을 평가하였으며, 이는 학생들의 교육적 이점을 제공할 가능성이 있습니다.



### Learning to Route with Confidence Tokens (https://arxiv.org/abs/2410.13284)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)에서 신뢰성을 판단하고 이를 기반으로 결과를 개선하기 위한 새로운 방법론인 Self-REF(Self-Reflection)를 제안합니다. Self-REF는 LLM이 자신의 예측에 대한 신뢰도를 효과적으로 평가할 수 있도록 훈련하는 경량화된 방법론입니다.

- **Technical Details**: Self-REF는 세 가지 주요 단계를 포함합니다: (i) 신뢰도 토큰 주석 추가, (ii) Self-REF 파인튜닝, (iii) 신뢰도 점수 추출. 신뢰도 토큰은 LLM이 올바르게 응답했는지를 기준으로 생성되며, 이러한 토큰에서 신뢰도 점수를 계산하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: Self-REF는 라우팅 및 거부 학습 작업에서 기존 방법들보다 뛰어난 성능을 보이며, 특히 네 개의 공개 데이터셋에서 우수한 결과를 나타냈습니다. 이는 LLM이 낮은 신뢰도 질문을 더 강력한 LLM으로 라우팅하거나 안전한 행동으로 자신의 대답을 거부하는 데 기여합니다.



### Roadmap towards Superhuman Speech Understanding using Large Language Models (https://arxiv.org/abs/2410.13268)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성공이 음성 및 오디오 데이터를 통합하려는 노력을 촉진시켰다고 강조합니다. 특히, GPT-4o와 같은 최신 모델을 통해 비언어적 정보와 세계 지식을 보존하며 깊은 음성 이해를 가능하게 하는 엔드 투 엔드(end-to-end) 음성 LLMs의 잠재력을 제시합니다.

- **Technical Details**: 논문에서는 음성 LLM을 개발하기 위한 다섯 단계의 로드맵을 제안합니다. 이 단계는 기본 자동 음성 인식(ASR)에서 시작하여 비언어적 정보와 추상적 음향 지식을 통합하는 고급 초인간(superhuman) 모델에 이르는 과정을 포함합니다. 또한, SAGI Benchmark라는 기준을 설계하여 각 단계에 걸쳐 다양한 작업을 평가할 수 있는 표준화된 방법을 제공합니다.

- **Performance Highlights**: 인간은 1단계에서 3단계까지의 작업에서 일반적으로 강한 성능을 보였지만, 상위 단계에서는 추상적 음향 지식 부족으로 인해 성능이 제한적이었습니다. 비록 현재의 음성 LLM들이 일부 영역에서 인간을 초월할 수 있는 능력이 있지만, 작업의 다양성과 포괄성 면에서 여전히 부족함을 드러냈습니다.



### The Latent Road to Atoms: Backmapping Coarse-grained Protein Structures with Latent Diffusion (https://arxiv.org/abs/2410.13264)
Comments:
          Paper under review

- **What's New**: 본 논문에서는 Latent Diffusion Backmapping (LDB)이라는 새로운 접근법을 제시하여, coarse-grained (CG) 분자 동역학 시뮬레이션에서의 단백질 구조 복원을 위한 효율성을 높입니다. 기존의 backmapping 방법들이 직면했던 문제들을 해결하기 위해, LDB는 노이즈 제거(diffusion) 기법을 이용해 라텐트(latent) 공간 내에서 구조를 효율적으로 재구성합니다.

- **Technical Details**: LDB는 노드 레벨의 라텐트 표현을 통해 모든 원자 구조를 인코딩하며, 화학적 유효성을 보장하기 위해 물리적 제약(예: 결합 길이 및 각도)을 적용합니다. 이는 광범위한 후처리 과정 없이도 화학적 유효성을 달성하고, 미세조정되지 않은 개별 원자 구조 복원을 통해 동적인 구조 공간 탐색을 용이하게 만듭니다. 이 방법은 조건부(conditional) 노이즈 제거 모델을 포함하여 디스크리트 라텐트 코드에서 작동함으로써 예측 정확성과 다양한 단백질 구조 생성을 극대화합니다.

- **Performance Highlights**: LDB는 PED, ATLAS 및 PDB와 같은 여러 단백질 다이너믹스 데이터셋에서 최신 성능을 입증하였으며, 구조적 정확성과 화학적 유효성을 유지하며 단백질 앙상블을 효율적으로 복원하는 능력을 보여주었습니다. 이러한 개선 사항에 따라 LDB는 CG 시뮬레이션과 원자 수준 분석 간의 격차를 효과적으로 연결하는 강력하고 확장 가능한 접근법으로 자리잡았습니다.



### scFusionTTT: Single-cell transcriptomics and proteomics fusion with Test-Time Training layers (https://arxiv.org/abs/2410.13257)
- **What's New**: 이번 논문에서는 CITE-seq 데이터를 활용한 단일 세포 다중 오믹스(scMulti-omics) 분석을 위한 새로운 방법인 scFusionTTT를 제안합니다. 이 방법은 Test-Time Training (TTT) 기반의 마스크 오토인코더(masked autoencoder)를 사용하여 유전자와 단백질 순서 정보를 통합하고, 다중 오믹스 데이터를 융합하는 데 기여합니다.

- **Technical Details**: scFusionTTT는 TTT 레이어를 적용하여 다중 오믹스 데이터를 처리하는 혁신적인 접근 방식을 제공합니다. 기존의 모형들이 간과했던 유전자 간의 순차적 관계를 고려하며, 고차원 데이터의 희소성 문제를 해결하기 위해 세 가지 단계의 훈련 전략을 사용합니다. 이 방법은 단일 세포 전사체(transcriptomics) 및 단백질체(proteomics) 데이터를 통합하여 균형 잡힌 표현을 학습하도록 고안되었습니다.

- **Performance Highlights**: scFusionTTT는 4개의 CITE-seq 데이터셋과 4개의 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터셋에서 우수한 성능을 보여주었습니다. 모든 비교 기준에서 기존의 최첨단 방법들과 비교했을 때 더 나은 결과를 달성하였으며, 이는 모델의 유용성과 신뢰성을 입증하는 데 기여합니다.



### Automatic Translation Alignment Pipeline for Multilingual Digital Editions of Literary Works (https://arxiv.org/abs/2410.13255)
Comments:
          18 pages, Computational Humanities Research Conference, December 4-6, 2024, Aarhus, Denmark

- **What's New**: 이 논문은 Alessandro Manzoni의 소설 "I promessi sposi"의 다국어 디지털 에디션(Multilingual Digital Edition, MDE) 제작을 위한 번역 정렬 알고리즘의 적용을 조사합니다. 19세기와 20세기의 8개 언어(영어, 스페인어, 프랑스어, 독일어, 네덜란드어, 폴란드어, 러시아어, 중국어) 번역을 포함하여 MDE의 주요 요구 사항을 식별하고, 문학 텍스트 번역에 대한 현재 알고리즘의 한계를 강조하며, MDE 생성을 위한 자동화된 파이프라인을 제안합니다.

- **Technical Details**: 이 연구에서는 문학 작품의 다국어 디지털 에디션을 제작하기 위해 최신 정렬 기법을 적용하는 자동 번역 정렬 파이프라인을 제안합니다. 이 파이프라인은 원문과 번역 텍스트의 나란히 배치된 웹 기반 표현으로 변환되며, 텍스트 조각을 관리 가능한 길이로 정렬하여 사용자에게 독서 및 분석에 용이하도록 합니다. 또한, 문학 번역의 정렬을 평가하기 위한 새로운 메트릭스를 제안하고 있습니다.

- **Performance Highlights**: 논문에서 제안한 정렬 메트릭스는 기존 정렬 알고리즘의 성능을 보다 포괄적으로 평가할 수 있게 하며, 문학 작품의 다국어 디지털 에디션 제작 시 독자의 집중력과 이해를 증진할 수 있는 방법들을 제시합니다. 이는 번역의 삽입 및 생략된 부분을 시각적으로 강조하여 사용자로 하여금 각 번역의 뉘앙스를 파악할 수 있도록 돕습니다.



### Perceptions of Discriminatory Decisions of Artificial Intelligence: Unpacking the Role of Individual Characteristics (https://arxiv.org/abs/2410.13250)
- **What's New**: 이 연구는 개인의 차이점(디지털 자기 효능감 (digital self-efficacy), 기술 지식 (technical knowledge), 평등에 대한 믿음 (belief in equality), 정치적 이념 (political ideology))과 인구 통계적 요인(연령, 교육, 소득)이 성별 및 인종 차별 편향을 보여주는 인공지능(AI) 결과에 대한 인식과 AI에 대한 일반적인 태도와 어떻게 관련되어 있는지를 조사합니다.

- **Technical Details**: 대규모 실험 데이터셋(N = 1,206)을 분석한 결과, 디지털 자기 효능감과 기술 지식은 AI에 대한 태도와 긍정적인 상관관계를 보였고, 자유주의적 이념은 결과 신뢰도(outcome trust)와 부정적인 감정, 더 큰 회의론(skepticism)과는 부정적인 상관관계가 나타났습니다. 또한, 연령과 소득은 차별적 AI 결과를 이해하는 인지적 격차(cognitive gaps)와 밀접하게 연결되어 있습니다.

- **Performance Highlights**: 이 연구 결과는 디지털 리터러시(digital literacy) 기술을 증진하고 디지털 자기 효능감을 향상시키는 것이 AI에 대한 신뢰와 AI의 유용성 및 안전성에 대한 믿음을 유지하는 데 중요하다는 점을 강조합니다. 또한, 문제 있는 AI 결과에 대한 이해의 차이는 경제적 불평등과 사회의 세대 간 격차와 연관되어 있을 수 있음을 시사합니다.



### Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation (https://arxiv.org/abs/2410.13248)
- **What's New**: 최근 설명 가능한 추천 시스템에 대한 연구는 표준 텍스트 생성 문제로 접근하며, 모델을 예측된 텍스트와 실제 텍스트 간의 유사성을 기반으로 평가합니다. 그러나 이 접근법은 사용자(구매 후) 감정을 정확히 반영하는지 여부를 간과합니다. 이 연구에서는 사용자의 감정을 중점적으로 고려하는 새로운 데이터셋과 평가 방법을 소개합니다.

- **Technical Details**: 우리는 LLM(Long Language Model)을 사용하여 사용자 구매 후 리뷰에서 긍정적 및 부정적 의견을 명시적으로 추출하여 데이터셋을 구성합니다. 시스템을 평가할 때 생성된 설명이 1) 사용자 감정과 잘 일치하는지, 2) 목표 아이템에 대한 사용자 의견의 긍정적 및 부정적 식별을 정확히 수행하는지에 대한 두 가지 기준을 제안합니다.

- **Performance Highlights**: 여러 최신 모델을 우리의 데이터셋에서 벤치마킹하였으며, 기존 지표에서 높은 성과를 달성하더라도 생성된 설명이 사용자 감정과 잘 일치하지 않을 수 있음을 보여줍니다. 또한, 목표 아이템에 대한 사용자(예측된) 평가가 모델에 직접 입력될 경우, 기존 모델들이 보다 감정 인식적인 설명을 제공할 수 있음을 발견하였습니다.



### Enhancing Sentiment Analysis with Collaborative AI: Architecture, Predictions, and Deployment Strategies (https://arxiv.org/abs/2410.13247)
- **What's New**: 이번 연구에서 소개된 협력적 인공지능 프레임워크는 다양한 인공지능 시스템 간의 작업 분배와 해결을 효율적으로 수행하여 복잡한 감정 분석 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 감정 분석을 위해 Generative AI 모델인 ChatGPT와 Google Gemini 같은 모델을 활용하여 복잡한 과제를 관리 가능한 단계적 목표로 단순화하는 방법론을 제시합니다. 논문은 또한 Edge 및 Cloud 환경에서 협력적 AI 시스템을 이용한 사례 연구를 통해 다양하고 풍부한 온라인 미디어 채널에서 감정 분석을 수행하는 효과성을 보여줍니다.

- **Performance Highlights**: 협력적 AI 시스템은 멀티모달 데이터 처리에서 우수한 성능을 발휘하며, 전통적인 LLM이나 신경망보다 더 넓고 정확한 감정 분석 결과를 제공합니다. 알고리즘 기반의 프롬프트 개선도 이루어져 감정 보고서 출력의 안정성을 높였습니다.



### Atomic Calibration of LLMs in Long-Form Generations (https://arxiv.org/abs/2410.13246)
- **What's New**: 본 논문에서는 LLM(대규모 언어 모델)의 신뢰성을 높이기 위한 새로운 접근법인 atomic calibration을 제안합니다. 기존의 macro calibration은 주로 짧은 응답에 대한 신뢰도를 평가하는 데 초점을 맞추었으나, 긴 응답의 경우 더욱 복잡한 진술이 포함될 수 있어 적합하지 않다는 점을 강조하였습니다.

- **Technical Details**: atomic calibration은 긴 응답을 작은 단위인 atomic claims로 분해하여 세부적인 신뢰도를 평가합니다. 본 연구에서는 LLM의 신뢰성 추정 방법을 discriminative와 generative 유형으로 나누고, 이들의 조합이 calibration을 개선할 수 있음을 보여줍니다. 또한, 7종의 LLM과 3개의 데이터셋을 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: atomic calibration은 긴 형식의 생성 과정에서도 잘 작동하며, macro calibration 결과를 개선할 수 있는 것으로 나타났습니다. 이 방법은 LLM의 생성 과정에서 신뢰성과 calibration 변화의 패턴을 심도 있게 분석할 수 있는 가능성을 열어줍니다.



### Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis (https://arxiv.org/abs/2410.13237)
Comments:
          17 pages, 6 figures, 14 tables

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)에서 발생하는 'Language Confusion' 현상을 분석하고, 이를 정량화하기 위한 새로운 측정 기준인 'Language Confusion Entropy'를 제안합니다. 이는 다양한 언어 분포의 패턴을 탐구하며, LLM 보안과의 연관성도 밝혔습니다.

- **Technical Details**: Language Confusion Entropy는 LLM에서 발생하는 언어 혼란 정도를 정량화하는 지표로, 언어 유형론에 기반한 언어 분포를 사용하여 LLM이 혼란스러울 때의 양상을 포착합니다. 이 연구는 여러 언어 간의 의미적 유사성과 LLM의 취약성을 연결지으며, 다국어 임베딩 역전 공격(multilingual embedding inversion attacks)에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 언어 유형론을 기반으로 분석한 패턴들이 언어 혼란과 연관되어 있음을 발견했습니다. 특히 자원이 적은 언어는 혼란이 덜 발생하며, 다양한 스크립트와 언어 계열을 아우르는 훈련이 언어 혼란을 보다 효과적으로 완화할 수 있다는 결과를 도출했습니다.



### SPIN: Self-Supervised Prompt INjection (https://arxiv.org/abs/2410.13236)
- **What's New**: 이번 논문에서는 Self-supervised Prompt INjection (SPIN)라는 새로운 방어 메커니즘을 도입하여, 다양한 adversarial 공격 및 jailbreak 공격에 대해 LLMs의 안전성을 향상시키는 방법을 제시합니다. SPIN은 추론 시간(inference-time)에서 이루어지므로 기존의 안전성 정렬과 호환되며 추가적인 안전성 레이어를 제공합니다.

- **Technical Details**: SPIN은 self-supervised learning을 기반으로 하여, 공격을 탐지하고 입력을 복구하는 방어 기법입니다. LLM의 자연 가이드라인을 무효화하는 프롬프트가 모델의 다른 능력도 저하시키기 때문에, 이를 이용해 공격을 탐지할 수 있습니다. 방어 메커니즘은 기존의 방어 시스템과의 호환성이 있으며, 악의적 또는 선의적 레이블에 의존하지 않고 재빠른 시점에서 온라인으로 사용할 수 있습니다.

- **Performance Highlights**: SPIN의 적용 결과, Attack Success Rate (ASR)를 최대 87.9%까지 감소시킬 수 있었으며, benign 사용자 요청에 대한 성능을 유지했습니다. Advbench에서 Universal Adversarial Triggers를 사용한 실험 결과, Vicuna 모델에서는 ASR이 12.11%, Llama-2 모델에서는 0%로 감소하여 두 모델을 완전히 보호했습니다. 또한, 공격자들이 방어 체계를 알고 있어도 여전히 강인성을 보였습니다.



### Quamba: A Post-Training Quantization Recipe for Selective State Space Models (https://arxiv.org/abs/2410.13229)
- **What's New**: 본 연구는 State Space Models (SSMs)을 위한 정적 8-bit per-tensor quantization 방법을 제안하여 모델의 효율성을 크게 향상시키고, 클라우드 및 에지 플랫폼에서의 원활한 배포를 지원합니다.

- **Technical Details**: 제안된 방법은 Hadamard 변환을 활용하여 SSM의 출력 활성화에서 극단적인 아웃라이어를 부드럽게 하고, 선택적 SSM에 대한 입력 활성화의 최대값을 억제하여 더 섬세한 quantization 정밀도를 제공합니다. 또한, 8-bit 정량화된 Mamba 2.8B SSM은 Nvidia Orin Nano 8G에서 1.72배 더 낮은 생성 지연 시간을 달성하며, 평균 정확도는 0.9% 감소했습니다.

- **Performance Highlights**: Mamba SSM은 하드웨어 가속의 이점을 누리며, 다양한 크기의 SSM 기반 모델의 클라우드 및 에지 플랫폼에서의 효과성과 실용성을 입증했습니다.



### From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning (https://arxiv.org/abs/2410.13228)
Comments:
          physics-informed neural networks, Kolmogorov-Arnold networks, optimization algorithms, separable PINNs, self-adaptive weights, uncertainty quantification

- **What's New**: 최근 Physicas-Informed Neural Networks (PINNs)의 발전이 뚜렷하며, 새로운 구조와 최적화 기법이 포함된 Physics-Informed Kolmogorov-Arnold Networks (PIKANS) 등이 소개되었습니다. PINNs는 물리 법칙을 반영하여 희소 데이터로도 효율적으로 PDE(Partial Differential Equations)를 해결합니다.

- **Technical Details**: PINNs는 물리 법칙을 인코딩하기 위해 추가적인 ‘residual’ 손실 항을 포함하며, 자동 미분을 이용하여 효율적으로 계산합니다. 이를 통해 전통적인 수치적 방법 한계를 극복하고, 복잡한 기하구조를 다루는 데 유리합니다. 또한, PINNs는 불확실성 정량화 및 다양한 최적화 기법을 수용할 수 있습니다.

- **Performance Highlights**: PINNs는 생의학, 유체 역학, 지구 물리학 등 다양한 분야에서 적용 가능함을 보여주었으며, 2017년 첫 발표 이후 11,000회 이상의 인용이 이루어졌습니다. 또한, 여러 연구 그룹에서 PINNs의 알고리즘 개선 및 적용 가능성을 탐색하고 있습니다.



### CBT-Bench: Evaluating Large Language Models on Assisting Cognitive Behavior Therapy (https://arxiv.org/abs/2410.13218)
- **What's New**: 본 논문은 지금의 정신 건강 지원에서 환자의 필요와 제공 가능한 지원 간의 큰 격차를 해결하기 위한 접근법으로서, 대형 언어 모델(LLMs)을 전문적인 심리 치료에 활용하는 가능성을 깊이 조사합니다. 특히, 우리는 인지 행동 치료(CBT) 지원의 체계적 평가를 위한 새로운 벤치마크인 CBT-BENCH를 제안합니다.

- **Technical Details**: CBT-BENCH는 세 가지 수준의 과제로 구성됩니다: I: 기본 CBT 지식 습득을 위한 다중 선택 질문; II: 인지 모델 이해를 위한 인지 왜곡 분류, 주요 핵심 신념 분류, 세부 핵심 신념 분류 작업; III: CBT 세션에서 환자 발화에 대한 치료적 응답 생성. 논문에서는 CBT의 핵심 측면을 AI 지원을 통해 향상할 수 있는 가능성을 조명하며, 각 과제는 기본 지식 암기부터 실제 치료 대화에 참여하는 것과 같은 복잡한 능력 요구 사항의 계층을 포함합니다.

- **Performance Highlights**: 실험 결과에 따르면 LLMs는 CBT 지식을 암기하는 데는 상대적으로 잘 수행하였지만, 환자의 인지 구조에 대한 깊은 분석이 필요한 복잡한 실제 시나리오에서는 부족한 성과를 보였습니다. LLMs는 일반적으로 엄격한 논리적 추론 프로세스를 따르지만, 치료에서 중요한 환자의 관점에서 사고하고 관계를 구축하는 능력이 부족하다는 제한점을 보여주었습니다.



### MixEHR-Nest: Identifying Subphenotypes within Electronic Health Records through Hierarchical Guided-Topic Modeling (https://arxiv.org/abs/2410.13217)
- **What's New**: MixEHR-Nest는 전자 건강 기록(EHR) 데이터를 활용하여 고유한 하위 표현형(sub-phenotype) 주제를 유도하는 새로운 지침(topic model) 모델입니다. 이 모델은 경험적인 표현형 개념(PheCodes, CCS 코드를 포함)으로 초기화된 하위 주제를 탐지하여 질병 패턴을 더욱 세분화하여 나타냅니다.

- **Technical Details**: MixEHR-Nest는 다중 모달(multi-modal) EHR 데이터에서 1500개 이상의 표현형으로부터 뚜렷한 하위 표현형 주제를 유도할 수 있는 구조화된 하이라키(topic model)입니다. 이 모델은 각 환자의 의료 기록을 문서(document)로, 코드(예: ICD 코드)를 단어 토큰(word tokens)으로 간주하여 학습합니다. 이 연구는 하위 표현형 주제의 묘사, 다중 유형의 EHR 정보 학습, 높은 해석 가능성을 통한 자동 하위 표현형 유도를 포함합니다.

- **Performance Highlights**: MixEHR-Nest는 ICU 환자 사망률 예측, 당뇨병 환자의 초기 인슐린 치료 예측에서 성능을 향상시켰습니다. 또한 MixEHR-Nest는 같은 표현형 아래에서 연령 분포의 뚜렷한 하위 표현형을 확인함으로써 다양한 질병에 걸쳐 질병의 진행 및 중증도를 예측하는 데 기여했습니다.



### AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations (https://arxiv.org/abs/2410.13212)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 Large Language Model (LLM)에서 Key-Value Cache (KV Cache)의 비대칭적 구조적 역할을 심층적으로 탐구하고, 이를 바탕으로 새로운 비대칭 양자화 전략인 AsymKV를 제안합니다. 기존의 양자화 기법이 키와 값 행렬에 동일한 구성을 사용하는 것 대신, 키와 값 행렬에 서로 다른 비트 양자화를 적용하는 방식을 채택합니다.

- **Technical Details**: 연구에서는 모델 양자화의 일환으로, KV Cache의 키 행렬과 값 행렬에 대해 비대칭적으로 1비트 양자화를 적용하는 방법을 제시합니다. 이 과정에서 각 행렬의 구조적 특성을 고려하여, 초기 몇 개의 디코더 레이어에는 4비트 양자화를 적용하고 이후의 레이어에는 1비트를 적용하는 방식으로 구현합니다. 또한, 실험을 통해 다양한 데이터셋에서 이 방법을 검증하였습니다.

- **Performance Highlights**: 실험 결과, AsymKV 접근법은 최대 75%의 디코더 레이어를 1비트로 양자화하면서도, 부동소수점 매개변수를 사용할 때와 비슷한 성능 수준을 유지하는 것으로 나타났습니다. 이는 메모리 소비를 줄이면서도 모델의 성능을 보장할 수 있는 효율적인 전략임을 입증합니다.



### Estimating the Probabilities of Rare Outputs in Language Models (https://arxiv.org/abs/2410.13211)
Comments:
          27 pages, 9 figures

- **What's New**: 이 논문은 저확률 추정(low probability estimation) 문제를 다루며, 이는 머신 러닝 모델의 출력으로부터 특정 이진 속성을 추정하는 과정을 포함합니다. 이러한 추정은 확률이 매우 작아 랜덤 샘플링(random sampling)으로는 불가능할 수 있으며, 분포 이동(distribution shift) 문제를 해결하기 위해 필수적입니다.

- **Technical Details**: 저자들은 두 가지 방법을 비교합니다: 1) Importance Sampling(중요도 샘플링): 드문 출력(event)을 생성하는 입력을 위해 새로운 입력 분포를 정의하는 방법입니다. 이 방법에는 Independent Token Gradient Importance Sampling(ITGIS)과 Metropolis-Hastings Importance Sampling(MHIS)이 포함됩니다. 2) Activation Extrapolation(활성화 외삽): 모델 로그잇(logits)에 맞춰 확률 분포를 피팅하고, 이를 기반으로 외부로 확장하는 방법입니다. 이 방법은 Quadratic Logit Decomposition(QLD)과 Gaussian Logit Difference(GLD)로 나눌 수 있습니다.

- **Performance Highlights**: 실험 결과, 중요도 샘플링 방법이 활성화 외삽보다 우수하며, 두 방법 모두 무작위 샘플링보다 좋습니다. 이는 최악의 성능 보장을 제공하기 위한 새로운 저확률 추정 기법이 필요하다는 점을 강조합니다.



### FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs (https://arxiv.org/abs/2410.13210)
- **What's New**: 이 논문에서는 10개의 현대 LLM(대형 언어 모델)와 8개의 모델 가족에서 발생하는 도전적 환각을 포함하는 FaithBench라는 환각 평가 벤치마크를 제안합니다.

- **Technical Details**: FaithBench는 인공지능 모델에 의해 생성된 요약에서 발생하는 환각(hallucination)을 평가하기 위해 설계되었습니다. 이 벤치마크는 LLM 가족에 따라 다양한 환각 사례를 포함하고 있으며, 각 요약은 인간 전문가에 의해 주석이 달린 ground truth를 포함하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, GPT-4o와 GPT-3.5-Turbo가 가장 낮은 환각 비율을 나타냈지만, 환각 탐지 모델의 정확도는 여전히 50%에 가까운 수치를 기록하여 개선의 여지가 많음을 시사합니다.



### TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering (https://arxiv.org/abs/2410.13203)
Comments:
          This paper has been accepted for presentation at the 26th International Conference on Pattern Recognition (ICPR 2024) in Kolkata, India

- **What's New**: 이번 연구에서는 TabSeq라는 새로운 프레임워크를 소개하여, 비정형(tabular) 데이터의 최적의 특성(feature) 순서를 위해 제안되었습니다. 이는 딥러닝 모델의 학습 효율성을 높이는 데 기여할 것입니다.

- **Technical Details**: TabSeq 프레임워크는 클러스터링(clustering)과 지역(local) 및 전역(global) 정렬 기법을 결합하여 비정형 데이터의 특성을 최적화하는 기능을 제공합니다. 이 기술은 멀티 헤드 어탠션(multi-head attention) 메커니즘이 포함된 디노이징 오토인코더(denoising autoencoder) 네트워크와 함께 사용됩니다.

- **Performance Highlights**: 본 연구는 원시 항체 마이크로어레이(raw antibody microarray) 및 기타 두 개의 실제 생물 의학 데이터셋을 통해 제안된 특성 순서 조정 기법이 딥러닝 모델 성능을 유의미하게 개선할 수 있음을 보여주었습니다.



### Meta-DiffuB: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration (https://arxiv.org/abs/2410.13201)
- **What's New**: Meta-DiffuB는 Seq2Seq 텍스트 생성을 위한 새로운 스케줄러-탐색자 모델을 도입하여 기존 S2S-Diffusion 모델의 한계를 극복합니다. 기존 모델들은 고정된 또는 수작업으로 만든 규칙에 의존하여 노이즈를 스케줄링하는 반면, Meta-DiffuB는 문맥화된 노이즈 스케줄링을 통해 문장별로 적합한 노이즈를 적용합니다.

- **Technical Details**: Meta-DiffuB는 두 가지 모델로 구성됩니다: 스케줄러와 탐색자. 스케줄러는 각 문장의 특성에 맞춰 적절한 수준의 노이즈를 스케줄링하고, 탐색자는 해당 노이즈를 활용하여 업데이트 및 생성을 수행합니다. 이 접근 방식은 자연어 처리(NLP)에서 Seq2Seq 작업의 의미론적 특성을 반영합니다.

- **Performance Highlights**: Meta-DiffuB는 네 가지 Seq2Seq 벤치마크 데이터세트에서 기존 S2S-Diffusion 모델 및 정밀 조정된 사전 훈련된 언어 모델(PLMs)과 비교하여 최첨단 성능을 달성합니다. 또한, 스케줄러 모델은 기존 DiffuSeq를 더욱 향상시키기 위한 '플러그 앤 플레이' 기능을 제공합니다.



### Golyadkin's Torment: Doppelg\"angers and Adversarial Vulnerability (https://arxiv.org/abs/2410.13193)
- **What's New**: 이 논문은 'Adversarial Doppelgangers(AD)'라는 개념을 정의하고 탐구하며, 이는 기존의 adversarial visual metamers를 포함합니다. AD의 성능 및 강건성을 분류 기계와 인간의 성능을 비교하여 분석합니다.

- **Technical Details**: AD는 이 논문에서 정의된 지각적(metric) 측정에 따라 서로 가까운 입력들입니다. 연구에서는 이러한 AD에 대한 분류기의 취약성을 분석하고, AD에 강건한 분류기의 구조와 속성을 설명하며, 개념적 엔트로피(conceptual entropy) 및 개념적 모호성(regions of conceptual ambiguity)에 대한 개념을 도입합니다.

- **Performance Highlights**: 대부분의 분류기는 AD에 취약하며, 강건성-정확도 트레이드오프(robustness-accuracy trade-offs)가 개선되지 않을 수 있습니다. 그러나 정확도가 높은 모든 분류기는 hypersensitive behavior를 보여줄 있으며, 이로 인해 AD 강건성을 개선하는 것이 정확도 개선과 동일함을 발견했습니다.



### MCQG-SRefine: Multiple Choice Question Generation and Evaluation with Iterative Self-Critique, Correction, and Comparison Feedback (https://arxiv.org/abs/2410.13191)
Comments:
          Equal contribution for the first two authors

- **What's New**: 이번 연구에서는 MCQG-SRefine라는 새로운 프레임워크를 제안하여, 전문의 시험을 위한 고품질 다지선다 질문(USMLE 스타일 질문)을 자동 생성하는 방법을 소개합니다. 이 프레임워크는 LLM의 자기 수정(self-refine) 기반으로, 전문가의 피드백과 반복적인 자기 비판을 통해 질문의 품질과 난이도가 향상됩니다.

- **Technical Details**: MCQG-SRefine는 의료 사례를 입력으로 받아 USMLE 스타일의 질문을 생성합니다. FR 쿼리 설정과 41개의 주요 주제를 포함한 체크리스트를 기반으로, LLM이 의료 사례에서 정보를 추출하여 질문을 생성합니다. 또한, LLM 스스로 피드백을 주고, 이를 기반으로 질문을 수정하는 세 가지 단계(S1: 초기 MCQ 생성, S2: 비판 피드백, S3: 수정 피드백)를 따릅니다.

- **Performance Highlights**: MCQG-SRefine를 통해 생성된 질문은 GPT-4가 생성한 질문보다 72.5%의 선호도를 기록했으며, 더 높은 난이도의 질문을 생성하는 것이 확인되었습니다. 쉽고 중간 수준의 질문에서 각각 80% 감소 및 2.25배 증가, 어려운 질문에서 4배 증가하는 결과를 보였습니다. LLM-as-Judge를 활용해 전문가 평가를 대체할 수 있는 신뢰성 있는 자동 평가 시스템 또한 제안되었습니다.



### CohEx: A Generalized Framework for Cohort Explanation (https://arxiv.org/abs/2410.13190)
- **What's New**: 이번 논문은 설명 가능한 인공지능(eXplainable Artificial Intelligence, XAI)의 발전을 위해서 새로운 코호트 기반(cohort-based) 설명 방법에 대해 탐구합니다. 기존의 설명 기법들이 전반적인(global) 또는 개별적(local) 설명에 중점을 두고 있는 반면, 본 연구에서는 특정 그룹에 대한 설명의 중요성을 강조합니다. 이를 통해 모델의 결정과정에 대한 보다 깊은 이해를 제공합니다.

- **Technical Details**: 코호트 설명(cohort explanation)은 데이터셋의 하위 집합 또는 모델의 입력/결정 공간의 하위 영역에 대한 일반화된 설명을 제공합니다. 연구진은 이와 관련된 고유한 도전과 기회를 논의하며, 코호트 설명의 이상적인 속성을 정의하고 이를 기반으로 한 일반화된 프레임워크(supervised clustering)를 제안합니다. 또한, 기존의 데이터 기반(local feature importance) 방법을 코호트 설명으로 전환하는 방법도 개발하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 알고리즘은 기존의 벤치마크와 비교하여 우수한 성능을 발휘했다고 보고합니다. 이는 코호트 설명이 각 코호트를 특정 지역으로 구분하여 보다 간결하고 명확한 설명을 가능하게 하기 때문입니다.



### aiXcoder-7B: A Lightweight and Effective Large Language Model for Code Completion (https://arxiv.org/abs/2410.13187)
Comments:
          aiXcoder-7B is available at this https URL

- **What's New**: aiXcoder-7B는 70억 개의 매개변수를 가지며, 코드 완성을 위하여 설계된 경량화된 대형 언어 모델(LLM)입니다. 기존 LLM에 비해 보다 높은 정확도를 기록하며, 개발자 생산성을 높이기 위해 응답 시간을 단축시킵니다.

- **Technical Details**: aiXcoder-7B는 세 가지 주요 요소에 의해 우수한 성능을 발휘합니다: (1) 다중 목표 훈련(Multi-objective training), (2) 다양한 데이터 샘플링 전략(Diverse data sampling), (3) 방대한 고품질 데이터(Extensive high-quality data). 특히, Structured Fill-In-the-Middle (SFIM)이라는 훈련 목표를 사용하여 코드의 구문 구조를 고려합니다. 이와 함께 1.2 조 개의 고유한 토큰을 소비하여 훈련됩니다.

- **Performance Highlights**: aiXcoder-7B는 6개의 코드 완성 벤치마크에서 최신 LLM들보다 우수한 성능을 나타내며, 심지어 StarCoder2-15B와 CodeLlama-34B와 같은 더 큰 LLM보다도 뛰어난 결과를 기록하였습니다. 이는 aiXcoder-7B가 경량화된 모델임에도 불구하고 뛰어난 코드 완성 정확도를 보유하고 있음을 나타냅니다.



### EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning (https://arxiv.org/abs/2410.13179)
- **What's New**: 이번 논문에서는 Speech Representation Learning을 위한 새로운 Self-Supervised Learning 접근 법인 EH-MAM (Easy-to-Hard adaptive Masked Acoustic Modeling)을 제안합니다. 기존의 랜덤 마스킹 방식을 사용하는 Masked Acoustic Modeling (MAM)과는 달리, 우리는 선택적이고 적응적인 마스킹 전략을 도입하였습니다.

- **Technical Details**: EH-MAM은 SSL 훈련 중 모델에 점진적으로 더 어려운 영역을 도입하여 재구성을 수행합니다. 개별 프레임의 재구성 손실( reconstruction loss)을 활용하여 MAM 전제 과제를 해결하는 난이도를 판단하며, 이를 위해 교사 모델(teacher model)을 사용하여 프레임 단위 손실을 예측하고 어떤 프레임을 마스킹할 지 결정합니다.

- **Performance Highlights**: EH-MAM은 여러 최신 기준선(baselines) 대비 5%-10% 향상된 성능을 보이며, 저자원(low-resource) 음성 인식 및 SUPERB 벤치마크에서 효과적으로 유용한 컨텍스트를 포착하는 마스킹 영역을 분석합니다.



### GeSubNet: Gene Interaction Inference for Disease Subtype Network Generation (https://arxiv.org/abs/2410.13178)
Comments:
          Under review as a conference paper at ICLR 2025

- **What's New**: GeSubNet은 다양한 질병 아형에 따른 유전자 상호작용을 예측할 수 있는 통합 표현을 학습하는 새로운 방법론입니다.

- **Technical Details**: GeSubNet은 다단계 표현 학습 프레임워크로, 환자 유전자 발현 프로파일로부터 특정 질병 아형을 학습하는 심층 생성 모델, 지식 데이터베이스로부터 이전 유전자 네트워크의 표현을 포착하는 그래프 신경망(GNN), 그리고 두 표현을 통합하는 추론 손실을 포함한 세 가지 모듈로 구성됩니다.

- **Performance Highlights**: GeSubNet은 네 가지 그래프 평가 지표에서 평균 30.6%, 21.0%, 20.1%, 56.6%의 성능 향상을 기록했으며, 11,327개의 유전자 평가 실험에서 특정 아형에 영향을 미칠 가능성이 83%인 유전자 발견의 잠재력을 보여주었습니다.



### TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness (https://arxiv.org/abs/2410.13175)
- **What's New**: 이번 연구는 Tropical Cyclone Precipitation Diffusion (TCP-Diffusion)이라는 다중 모드 모델을 제안하여 열대성 폭풍(TC) 강수 예측의 정확성을 높였습니다. 이 모델은 과거 강수 관측 및 다양한 환경 변수에 기반하여, TC 중심 주변의 강수를 다음 12시간 동안 3시간 간격으로 예측합니다.

- **Technical Details**: TCP-Diffusion 모델은 인접 잔차 예측(Adjacent Residual Prediction, ARP) 방식을 사용하여, 절대 강수량 대신 강수량 변화를 예측하도록 훈련 목표를 변경했습니다. 이를 통해 누적 오류를 줄이고 물리적 일관성을 확보합니다. 또한, 기상 요소와 수치 기상 예측(NWP) 모델 정보를 통합하여 더 풍부한 메타데이터를 추출합니다.

- **Performance Highlights**: 광범위한 실험 결과, TCP-Diffusion는 최신 딥 러닝(DL) 기반 강수 예측 방법 및 유럽 중기기상예보센터(ECMWF)의 NWP 방법과 비교하여 우수한 성능을 보여주었습니다.



### An Evolved Universal Transformer Memory (https://arxiv.org/abs/2410.13166)
Comments:
          29 pages, 14 figures. Preprint, under submission. Source code is available at this https URL

- **What's New**: 본 논문은 Neural Attention Memory Models (NAMMs)을 제안하며, 메모리 관리를 위한 학습된 네트워크를 도입하여 Transformers의 성능과 효율성을 동시에 향상시킵니다. 이는 기계가 가진 메모리 관리의 질의를 진화 기반 접근법으로 해결하여, 기능적으로 매우 다양한 아키텍처에서 자율적으로 적용될 수 있도록 설계되었습니다.

- **Technical Details**: NAMMs는 Transformers의 Key-Value (KV) 캐시의 잠재적 메모리를 형성하는 새로운 방법을 제안하여, 각 레이어와 attention head가 그들의 특정 요구에 가장 관련 있는 정보에 집중하도록 지원합니다. 이 방식은 학습된 attention 매트릭스를 기반으로 모든 transformer 기반 아키텍처에 일반적으로 적용 가능하며, Llama 3 8B 모델 위에서 학습하여 성능과 효율성을 모두 극대화합니다.

- **Performance Highlights**: NAMMs를 통한 학습 결과로 36개의 LongBench, InfiniteBench 및 새로운 일본어 벤치마크에서 뛰어난 성능 개선을 기록했습니다. 기존의 수작업 전략과 비교할 때, NAMMs는 성능 저하 없이 메모리 용량을 유의미하게 감소시켰습니다. 또한, NAMMs는 언어 과제로만 학습되었음에도 불구하고 다양한 입력 모달리티를 통해 다른 transformer 모델에 제로샷 전이(transfer) 되는 성과를 보였습니다.



### Utilizing Large Language Models in An Iterative Paradigm with Domain Feedback for Molecule Optimization (https://arxiv.org/abs/2410.13147)
- **What's New**: 본 연구에서는 약물 발견에서 분자의 최적화를 지원하기 위해 LLM (Large Language Models)을 효과적으로 활용할 수 있는 도메인 피드백 제공자인 Re²DF를 제안합니다. 이 새로운 접근법은 분자가 화학적으로 유효하지 않을 경우를 고려하여 수정된 분자의 유효성을 즉시 검증하며, 해당 분자의 개선을 위한 구체적인 피드백을 제공합니다.

- **Technical Details**: Re²DF는 외부 툴킷인 RDKit를 이용하여 수정된 분자가 화학적으로 유효한지를 체크합니다. 만약 유효하지 않다면, RDKit로부터 오류 메시지를 제공받아 LLM에게 수정 방향을 제시합니다. 또한, 수정된 분자가 원하는 특성을 충족하는지 확인하여, 목표에 대한 정확한 방향과 거리를 제공하는 신뢰할 수 있는 피드백을 생성합니다.

- **Performance Highlights**: Re²DF는 단일 속성 목표 20개에서 Hit ratio를 각각 16.95% 및 20.76% 향상시키며, 다중 속성 목표에서는 32개에서 각각 6.04% 및 5.25% 향상시켰습니다. 이러한 결과는 Re²DF가 기존 방법들보다 더 나은 성능을 발휘함을 알립니다.



### Trust but Verify: Programmatic VLM Evaluation in the Wild (https://arxiv.org/abs/2410.13121)
- **What's New**: 본 논문에서는 프로그램 기반 VLM 평가(Programmatic VLM Evaluation, PROVE)라는 새로운 벤치마크 패러다임을 제안합니다. 이 방법은 비주얼 컨텐츠에 대한 오픈 엔디드 질의에 대한 VLM의 응답을 신뢰성 있게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: PROVE는 하이퍼 세부 이미지 캡션으로부터 구성된 고충실도의 씬 그래프(scene graph) 표현을 기반으로 하며, 이를 통해 다양한 질문-응답(QA) 쌍을 생성하고, 각 QA 쌍의 검증을 위한 프로그램을 함께 생성합니다. 이후, 이 프로그램을 통해 각각의 QA 쌍의 정확성과 기초를 검증하면서 10,500개의 시각적으로 기초가 있는 QA 쌍을 포함하는 데이터셋을 구축합니다.

- **Performance Highlights**: BENCHMARK한 결과, 대부분의 기존 VLM들은 유용성과 진실성 사이의 균형을 잘 맞추지 못함을 발견했습니다. 도출된 결과는 최근 '더 나은' VLM 교육의 진전이 유용성 향상으로 이어지지만 진실성 향상에는 큰 도움을 주지 않는다는 것을 보여줍니다.



### Preference Diffusion for Recommendation (https://arxiv.org/abs/2410.13117)
- **What's New**: PreferDiff는 신규 개인화 순위 손실 함수로, 기존의 추천 시스템들이 사용하는 전통적인 목표 대신에Diffusion Models(확산 모델) 특화된 최적화 목표를 제안합니다.

- **Technical Details**: PreferDiff는 BPR(Bayesian Personalized Ranking)을 로그 가능도 순위 목표로 변환하고 여러 개의 네거티브 샘플을 통합하여 사용자 선호도를 더 잘 포착하도록 설계되었습니다. 변분 추론(variational inference)을 이용해 계산의 어려움을 극복하고 오차 기준에서 MSE 대신 cosine error를 적용하여 추천 작업에 대한 정렬을 개선합니다. 또한, 생성(generation) 및 선호(preference) 간의 균형을 맞춤으로써 DMs의 학습 안정성을 향상시킵니다.

- **Performance Highlights**: 세 가지 벤치마크에서 진행된 실험을 통해, PreferDiff는 우수한 추천 성능을 보였으며 일반적인 연속 추천(sequential recommendation) 능력에서 주목할 만한 결과를 나타냈습니다.



### Learning to Summarize from LLM-generated Feedback (https://arxiv.org/abs/2410.13116)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 생성 피드백을 통해 요약 품질을 향상시키는 방법을 탐구하고, FeedSum이라는 대규모 데이터셋을 소개합니다. 이는 다양한 품질의 요약에 대한 다차원 LLM 피드백을 포함하고 있습니다.

- **Technical Details**: FeedSum 데이터셋에서는 13개의 서로 다른 언어 모델을 사용하여 요약을 생성하고, 각 요약에 대해 신뢰성(faithfulness), 완전성(completeness), 간결성(conciseness)이라는 세 가지 핵심 차원에 대한 피드백을 수집합니다. 두 가지 방법인 감독형 세부 조정(supervised fine-tuning)과 직접 선호 최적화(direct preference optimization)를 비교하였고, SummLlama3-8B 모델이 Llama3-70b-instruct 모델보다 더 뛰어난 성능을 보였음을 확인했습니다.

- **Performance Highlights**: SummLlama3-8B 모델은 크기가 10배 이상 큰 Llama3-70b-instruct 모델을 초월하여 인간의 선호에 맞는 요약을 생성하는 데 성공하였습니다. 이는 Smaller 모델이 적절한 훈련을 통해 더 우수한 성능을 얻을 수 있음을 보여줍니다.



### Sound Check: Auditing Audio Datasets (https://arxiv.org/abs/2410.13114)
- **What's New**: 이 논문은 생성 오디오 모델의 윤리적 문제와 데이터셋의 공정성을 평가하기 위해 심층적인 문헌 조사를 수행하고, 현재 사용되고 있는 오디오 데이터셋의 편향, 독성 및 지적 재산권 문제를 규명합니다.

- **Technical Details**: 총 175개의 고유 오디오 데이터셋을 분석하였고, 36%는 웹에서 수집되었으며, 35%의 데이터셋이 저작권 침해 가능성이 있다고 평가됩니다. 사용된 주요 기법으로는 Audio Spectrogram Transformer와 LALMs(대형 오디오 언어 모델)가 포함됩니다.

- **Performance Highlights**: 현재의 오디오 데이터셋은 대체로 인종적 및 성별 편향을 포함하고 있으며, 저작권이 있는 자료가 많이 포함되어 있어 아티스트의 권리와 관련된 문제를 야기할 수 있습니다. 또한, 마이너리티 그룹에 대한 언급이 적어 대표성이 결여되어 있습니다.



### Cliqueformer: Model-Based Optimization with Structured Transformers (https://arxiv.org/abs/2410.13106)
- **What's New**: 본 논문에서는 기계 학습을 활용한 모델 기반 최적화(Model-Based Optimization, MBO) 문제를 해결하기 위한 새로운 접근 방식인 Cliqueformer를 소개합니다. Cliqueformer는 블랙박스 함수의 구조를 학습하여 높은 차원의 최적화 문제에서 성능을 향상시킵니다.

- **Technical Details**: Cliqueformer는 transformer 기반 아키텍처를 사용하여 기능적 그래픽 모델(Functional Graphical Model, FGM) 형태로 블랙박스 함수의 구조를 학습합니다. 이 모델은 디자인 후보에 대한 최적화 문제를 해결하기 위해 예측을 클리크의 FGM 상에 분해하고, 클리크들의 주변 분포가 넓은 범위를 커버하도록 강제합니다. 이 과정은 변별적 병목(Variational Bottleneck) 기법을 사용하여 수행됩니다.

- **Performance Highlights**: Cliqueformer는 여러 고차원 블랙박스 함수와 실제 화학 및 유전자 설계 작업에서 기존 방법들과 비교하여 뛰어난 성능을 보여주었습니다. 이 연구는 오프라인 데이터에서 모델 기반 최적화를 위해 기존의 보수적인 접근 방식을 우회하는 효과적인 전략을 제안합니다.



### A Little Human Data Goes A Long Way (https://arxiv.org/abs/2410.13098)
- **What's New**: NLP 시스템의 효율성을 높이기 위해, 인간 주석 데이터의 일부를 합성 데이터로 대체하는 방법을 연구하였으며, 90%까지 대체해도 성능 저하가 미미하지만 마지막 10% 대체 시에는 성능이 크게 떨어진다는 중요한 발견을 했습니다.

- **Technical Details**: 합성 데이터 생성 과정을 통해 데이터 포인트 수를 일정하게 유지하며 인간 생성 데이터 비율을 점진적으로 증가시켜 성능을 비교하였습니다. 사용하는 데이터셋은 총 8개로 Fact Verification (FV) 및 Question Answering (QA) 태스크에 대해 실험하였습니다. 평가 지표로는 정확도, Exact Match, String Inclusion, BLEU, ROUGE-1, BERTScore를 사용하였습니다.

- **Performance Highlights**: 완전히 합성 데이터로 훈련된 FV 및 QA 시스템은 최소 125개의 인간 데이터 포인트를 추가할 경우 성능이 현저히 개선되며, 작은 비율의 인간 데이터가 큰 가치를 지닐 수 있다는 것을 발견했습니다. 추가적인 인간 데이터를 통한 성능 향상은 200 포인트의 인간 데이터로 가능하며, 이는 수량적으로 더 많은 합성 데이터 포인트에 비해 비용 효율적이라는 것을 보여줍니다.



### Task Consistent Prototype Learning for Incremental Few-shot Semantic Segmentation (https://arxiv.org/abs/2410.13094)
Comments:
          conference

- **What's New**: 이번 연구는 Incremental Few-Shot Semantic Segmentation (iFSS) 문제에 대한 새로운 접근 방식을 제안합니다. 이는 모델이 몇 개의 주석된 예제만으로 새로운 클래스에 대한 세분화 능력을 지속적으로 확장할 수 있도록 하는 것을 목표로 합니다.

- **Technical Details**: 연구는 메타 학습(meta-learning)을 기반으로 한 프로토타입 접근 방식을 사용하여, 기존 지식을 보존하면서 신속하게 적응할 수 있도록 모델을 유도합니다. 특히, 베이스 세션 동안의 증강 평가 프로토콜을 모방하여 가상의 증분 작업 시퀀스를 샘플링하여 메타 목표를 설정, 신속한 적응을 가능케 합니다. 프로토타입 공간 재분배 학습(Prototype Space Redistribution Learning, PSRL)을 통해 클래스 프로토타입을 동적으로 업데이트하여 최적의 프로토타입 경계를 설정합니다.

- **Performance Highlights**: PASCAL 및 COCO 벤치마크를 기반으로 한 iFSS 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 여러 경쟁 기법에 비해 월등한 성능을 보임을 확인했습니다.



### Reverse-Engineering the Reader (https://arxiv.org/abs/2410.13086)
- **What's New**: 이 연구는 기존의 언어 모델을 인간의 심리 측정 데이터에 맞춰 최적화하는 새로운 방법론을 제시합니다. 이를 통해 언어 처리 시스템의 이해를 높이고자 합니다.

- **Technical Details**: 연구진은 언어 모델이 특정 언어 단위의 읽기 시간을 예측하는 능력을 향상시키기 위해 서프라이절 이론(surprisal theory)을 기반으로 한 새로운 정렬 기법을 사용합니다. 모델의 파라미터를 조정하여 읽기 시간을 예측하는 선형 회귀의 계수를 최적화합니다.

- **Performance Highlights**: 제안된 기법은 여러 모델 크기와 데이터 세트에서 언어 모델의 심리 측정 예측력을 향상시키는 것으로 나타났습니다. 그러나 심리 측정 예측력과 후속 자연어 처리(NLP) 작업 성능 간에 반비례 관계가 발견되었습니다.



### FedCAP: Robust Federated Learning via Customized Aggregation and Personalization (https://arxiv.org/abs/2410.13083)
Comments:
          14 pages, 12 figures, 5 tables, accepted by 2024 Annual Computer Security Applications Conference (ACSAC 2024)

- **What's New**: FedCAP는 데이터 이질성과 Byzantine 공격에 강한 연합 학습(FL) 프레임워크로, 모델 업데이트 보정 메커니즘을 통해 클라이언트 간 모델 업데이트의 방향성과 크기를 포착합니다. 또한 맞춤형 모델 집계 규칙을 설계하여 유사 클라이언트 간의 협업을 촉진하고 악의적인 클라이언트의 모델 성능 저하를 가속화합니다.

- **Technical Details**: FedCAP는 네 가지 주요 구성요소로 이루어져 있습니다: 모델 보정 메커니즘, 맞춤형 집계 규칙, 이상 감지 메커니즘 및 개인화된 훈련 모듈. 모델 보정 메커니즘은 비독립적이며 동일한 분포(non-IID) 환경에서 악성 모델 업데이트와 양성 업데이트를 구별하는 데 도움을 줍니다. 맞춤형 집계 규칙은 유사 클라이언트 간의 협업을 촉진하며, 이상 감지 메커니즘을 통해 악성 클라이언트를 빠르게 식별하고 제거합니다. Euclidean norm 기반의 감지 메커니즘이 도입되어 클라이언트의 모델 업데이트 차이에 대한 정밀 분석을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, FedCAP는 여러 비독립적 환경 및 일련의 중독 공격에 대한 강한 견고성을 보이며, 기존의 최첨단( SOTA) FL 방법들과 비교하여 모델 정확도와 견고성 모두에서 높은 성능을 나타냈습니다.



### Tuning Language Models by Mixture-of-Depths Ensemb (https://arxiv.org/abs/2410.13077)
- **What's New**: 최근 연구에서는 Transformer 기반의 대형 언어 모델(LLMs)에서 최종 레이어만을 사용하는 대신, 중간 레이어의 예측 능력에 주목하여 새로운 조정 프레임워크인 Mixture-of-Depths (MoD)를 제안하였습니다. MoD는 훈련 시 다양한 레이어의 출력을 활용함으로써 예측 성능을 향상시키고, 기존 조정 방법과 통합할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: Mixture-of-Depths (MoD) 프레임워크는 레이어별 가중치를 학습하여 최종 로그잇(logits)으로 기여하는 앙상블로서의 레이어 훈련을 가능하게 합니다. 보조 증류 손실(auxiliary distillation loss) 및 추가 정규화 모듈을 적용하여, 최종 레이어 출력을 교사가 되는 훈련 방식으로 중간 레이어의 예측 출력을 모델 학습 시 최대화하는 접근을 취합니다. 이 방법은 훈련 가능한 파라미터를 소폭 증가시키면서도, 기본 언어 모델의 성능을 유지합니다.

- **Performance Highlights**: MoD 프레임워크를 적용한 결과, 산술 및 상식 추론 작업에서 성능이 일관되게 향상되었으며, 전통적인 훈련 가능한 모듈과 비교하여 97% 적은 파라미터로 유사한 성능을 달성하였습니다. 이러한 결과는 LLM의 중간 표현을 활용하는 것이 훈련 중 예측 능력을 크게 향상시킬 수 있음을 보여줍니다.



### ERAS: Evaluating the Robustness of Chinese NLP Models to Morphological Garden Path Errors (https://arxiv.org/abs/2410.13057)
Comments:
          Under review in ARR/NAACL

- **What's New**: 이 논문에서는 중국어를 다루는 NLP 모델들이 형태소적 garden path 오류에 취약하다는 것을 보여줍니다. 이를 평가하기 위해 ERAS라는 벤치마크를 제안합니다.

- **Technical Details**: ERAS 벤치마크는 지역적으로 모호한 구문과 모호하지 않은 구문으로 이루어진 203,944 쌍의 시험 문장과 통제 문장을 포함합니다. 이 연구는 Transformer 기반 및 비신경 단어 분리 모델과 캐릭터 수준의 토큰화를 사용하는 감정 분석 모델을 평가합니다.

- **Performance Highlights**: 실험 결과, 단어 분리 모델과 감정 분석 모델 모두가 garden path 오류를 범하며, 단어 경계 정보를 제공하여 모델 성능을 개선할 수 있다는 것을 보여줍니다.



### Channel-Wise Mixed-Precision Quantization for Large Language Models (https://arxiv.org/abs/2410.13056)
- **What's New**: 본 연구에서는 채널별 혼합 정밀도 양자화(Channel-Wise Mixed-Precision Quantization, CMPQ)라는 혁신적인 방법을 제안합니다. CMPQ는 각 채널의 활성화 분포에 기반하여 양자화 정밀도를 할당하는 새로운 혼합 정밀도 양자화 기법으로, 다양한 비트폭 제약에 적응하도록 설계되었습니다.

- **Technical Details**: CMPQ는 비균일 양자화(non-uniform quantization) 전략을 채택하며, 두 가지 이상치 추출(outlier extraction) 기법을 결합하여 필수 정보를 보존하며 양자화 손실을 최소화합니다. 이 방법은 채널별로 정밀도를 조정하여 각 채널의 활성화 norm에 따라 높은 정밀도 혹은 낮은 정밀도를 할당합니다.

- **Performance Highlights**: CMPQ는 실험을 통해 정수 비트 양자화(integer-bit quantization) 작업에서 성능을 향상시키는 한편, 적은 메모리 증가로 상당한 성능 향상을 이끌어냈습니다. 이 연구는 다양한 디바이스의 기능에서 큰 이점을 제공합니다.



### Systems with Switching Causal Relations: A Meta-Causal Perspectiv (https://arxiv.org/abs/2410.13054)
Comments:
          19 pages, 3 figures, 4 tables

- **What's New**: 본 논문에서는 기계 학습에서의 인과관계 연구에서 고정된 과정을 기반으로 한 전통적인 접근 방식의 한계를 지적하고, 메타-인과 상태(meta-causal states)라는 개념을 도입하여 변동하는 시스템 동 dynamics를 분석하는 방법을 제안합니다.

- **Technical Details**: 메타-인과 상태는 고전적인 인과 모델을 유사한 질적 행동에 따라 클러스터로 그룹화하고 특정 메커니즘 매개변수화를 통합하는 방법을 제시합니다. 또한, 관찰된 에이전트 행동으로부터 메타-인과 상태를 추론하는 방법과 레이블이 없는 데이터로부터 이 상태를 분리하는 방법을 논의합니다.

- **Performance Highlights**: 메타-인과 모델(MCM)은 특정 시스템 동역학 내에서 질적 차이를 표현하는 데 있어 고전적인 구조적 인과 모델보다 강력하며, 메타-인과 분석을 통해 전통적인 인과 추론과는 다른 근본 원인 기여를 식별할 수 있습니다.



### FedGTST: Boosting Global Transferability of Federated Models via Statistics Tuning (https://arxiv.org/abs/2410.13045)
- **What's New**: 본 논문에서는 기존의 Federated Learning (FL) 방법들이 해결하지 못한 문제들을 다루는 새로운 접근법인 Federated Global Transferability via Statistics Tuning (FedGTST)를 제안합니다. FL의 여러 도전 과제를 해결하면서 전세계적으로 전이 가능성을 높이는 방법론을 소개합니다.

- **Technical Details**: FedGTST는 클라이언트 간의 Jacobian (그라디언트) 노르므를 활용한 클라이언트-서버 교환 프로토콜과, 서버에서 클라이언트 간의 평균 Jacobian 노르므를 높이는 지역 정규화 기법을 통해 보다 효과적인 전이 가능성을 도모합니다. 이는 전이 실패를 줄이고 목표 손실(target loss)을 보다 정교하게 제어할 수 있도록 돕습니다.

- **Performance Highlights**: FedGTST는 MNIST에서 MNIST-M, CIFAR10에서 SVHN 데이터셋을 포함한 다양한 실험에서 FedSR 및 FedIIR과 같은 기존 방법보다 10%의 성능 향상을 나타냈습니다. 특히, LeNet 모델을 사용할 경우, FedGTST는 FedSR 대비 9.8%, FedIIR 대비 7.6%의 더 높은 정확도를 기록했습니다.



### LFOSum: Summarizing Long-form Opinions with Large Language Models (https://arxiv.org/abs/2410.13037)
- **What's New**: 이 논문에서는 온라인 리뷰의 대량 처리 및 요약을 위한 새로운 접근법을 제안합니다. 특히, 1천 개 이상의 리뷰로 구성된 새로운 데이터셋을 소개하며, 이를 기반으로 하는 LLM(대형 언어 모델) 기반 요약 기법을 제안합니다.

- **Technical Details**: LFOSum 데이터셋은 TripAdvisor에서 수집된 호텔 리뷰로, 각 엔티티는 1천 개 이상의 리뷰를 포함하고 있습니다. 두 가지의 훈련이 필요 없는 요약 방법, 즉 Retrieval-Augmented Generation (RAG)과 긴 맥락의 LLM을 이용하여 대량 리뷰 요약을 처리합니다. 사용자 맞춤형 요약을 위한 세 가지 제어 메커니즘(쿼리 제어, 감정 제어, 길이 제어)을 도입하여 사용자 요구에 맞춘 요약을 가능하게 합니다.

- **Performance Highlights**: LLM은 여전히 긴 형식의 요약에서 감정과 형식 준수의 균형을 맞추는 데 어려움을 겪고 있으나, 관련 정보를 집중적으로 추출할 경우 오픈 소스 모델이 효과적으로 간격을 좁힐 수 있음을 보여줍니다.



### LEGAL-UQA: A Low-Resource Urdu-English Dataset for Legal Question Answering (https://arxiv.org/abs/2410.13013)
Comments:
          8 pages

- **What's New**: LEGAL-UQA는 파키스탄 헌법에서 유래한 첫 번째 우르두 법률 질문-답변(QA) 데이터셋을 소개합니다. 이 데이터셋은 619개의 질문-답변 쌍을 포함하며, 법률 기사의 컨텍스트도 포함되어 있어 낮은 자원 언어의 도메인 특화된 NLP 자원의 필요성을 해결합니다.

- **Technical Details**: 데이터셋 생성 과정은 OCR 추출, 수동 수정 및 GPT-4를 활용한 QA 쌍의 번역 및 생성으로 구성됩니다. LEGAL-UQA의 성능을 평가하기 위해 최신의 일반 언어 및 임베딩 모델을 실험하였으며, Claude-3.5-Sonnet 모델이 인간 평가에서 99.19%의 정확도를 달성하였습니다. 또한, mt5-large-UQA-1.0 모델을 미세 조정하여 다국어 모델을 전문 분야에 적용하는 데 따른 도전 과제를 강조하였습니다.

- **Performance Highlights**: OpenAI의 text-embedding-3-large는 Mistral의 mistral-embed 보다 더 나은 검색 성능을 보였습니다. LEGAL-UQA는 글로벌 NLP 발전과 현지화된 응용 프로그램 간의 격차를 해소하며, 파키스탄 내 법률 정보 접근성을 개선하는 기반을 마련합니다.



### Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images (https://arxiv.org/abs/2410.13010)
Comments:
          Published in the 3rd Workshop on New Frontiers in Adversarial Machine Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

- **What's New**: 기존의 적대적 공격이 주로 단일 모드에 초점을 맞추었던 반면, 본 연구에서는 CLIP과 같은 대규모 멀티 모달 모델(LMM)이 가지는 새로운 취약점에 주목합니다. 새로운 ‘Hiding-in-Plain-Sight (HiPS)’ 공격 기법을 통해 모델 예측을 미세하게 수정함으로써, 타겟 객체가 존재하지 않는 것처럼 보이게 하는 방법을 제안합니다.

- **Technical Details**: HiPS 공격은 두 가지 변형으로 소개됩니다: HiPS-cls는 클래스 레이블 정보를 활용하여 공격을 생성하며, HiPS-cap은 원본 이미지 캡션과 타겟 캡션을 사용하여 공격을 설계합니다. 이러한 공격 기법은 CLIP-Cap과 같은 이미지 캡셔닝 모델로 효과적으로 전이될 수 있습니다.

- **Performance Highlights**: HiPS 공격은 타겟 객체가 이미지 캡션에서 효과적으로 제거되도록 설계되었으며, 여러 평가 지표를 통해 성능을 검증합니다. 제안된 공격이 하위 모델에서 어떻게 작동하는지를 보여주며, 적대적 공격의 새로운 기준을 설정합니다.



### Flex: End-to-End Text-Instructed Visual Navigation with Foundation Models (https://arxiv.org/abs/2410.13002)
- **What's New**: 이 논문에서는 주어진 이미지와 자연어 명령에 기반한 로봇의 제어 정책을 향상시키기 위한 프레임워크인 Flex(Fly-lexically)를 소개합니다. Flex는 사전 훈련된 Vision Language Models(VLMs)를 활용하여 다양한 환경에서 강력한 제어 성능을 달성할 수 있도록 설계되었습니다.

- **Technical Details**: Flex는 두 가지 핵심 구성 요소로 이루어져 있습니다. 첫째, VLM의 패치 기반(descorptive) 특징을 사용하여 공간적(spatial) 및 의미론적(semantic) 정보를 결합합니다. 둘째, 강화된 정책 네트워크(policy network)를 훈련하여 제한된 훈련 데이터 세트에서도 효과적인 제어가 가능하도록 합니다.

- **Performance Highlights**: 드론의 플라이-투-타겟(fly-to-target) 작업에서 본 연구는 복잡한 환경과 다양한 명령 형식에서도 성공적으로 일반화할 수 있는 로봇의 적응력을 입증했습니다. 이 연구 결과는 훈련 데이터가 거의 없는 상황에서도 실제 장면에서 다양한 목표를 처리할 수 있는 능력을 보여주었습니다.



### SSET: Swapping-Sliding Explanation for Time Series Classifiers in Affect Detection (https://arxiv.org/abs/2410.12996)
- **What's New**: 이번 연구에서는 다변량 시계열 분류기를 위한 새로운 설명 방법인 SSET(Swapping-Sliding Decision Explanation)를 제안합니다. 이 방법은 예측 점수에서 중요한 하락을 초래하는 두 가지 주요 단계를 통해 설명을 생성합니다: 스와핑(swap) 단계와 슬라이딩(slide) 단계입니다.

- **Technical Details**: SSET는 두 단계로 구성됩니다. 첫 번째 단계에서는 주어진 인스턴스와 가까운 훈련 데이터로부터 중요한 변수를 찾아내기 위해 스와핑을 사용합니다. 두 번째 단계에서는 각 시간 단계에서 선택된 훈련 데이터에 대해 윈도우를 슬라이드하여 중요한 하위 시퀀스를 탐색합니다.

- **Performance Highlights**: SSET는 WESAD 및 MAHNOB-HCI의 두 실제 생리학적 시계열 데이터셋에서 CN-Waterfall 분류기를 이용해 평가되었으며, 기존 모델들(Dynamask, integrated gradients, LIME)보다 우수한 성능을 보였습니다.



### Qtok: A Comprehensive Framework for Evaluating Multilingual Tokenizer Quality in Large Language Models (https://arxiv.org/abs/2410.12989)
Comments:
          24 pages, 9 figures, 6 tables. Code and data available at this https URL

- **What's New**: 이번 연구에서 우리는 Qtok이라는 도구를 도입하여 멀티링구얼 모델에서의 토크나이저 품질을 평가하는 방법론을 제공합니다. 기존의 연구에서는 주로 데이터셋 품질이나 모델 아키텍처에 초점을 맞추었지만, 토크나이저의 중요성은 상대적으로 간과되었습니다.

- **Technical Details**: 연구팀은 Qtok 도구를 통해 58개의 공개 모델에서 13개의 다양한 토크나이저를 평가했습니다. 이 도구는 언어 범위, 토큰 완전성, 언어 및 언어 범주에 따라 분포를 측정하는 지표를 포함하여 토크나이저의 품질을 평가합니다. 또한 코어 토큰 개념을 도입하여 긍정적으로 반복되는 토큰을 구분하였습니다.

- **Performance Highlights**: 분석 결과, 다양한 언어 및 범주에서 토큰 분포의 중요 자질 불균형이 발견되어 현재의 토크나이징 전략에서 개선이 필요한 부분을 강조하였습니다. 연구는 토크나이저의 품질 평가 방법을 제공하고 이로 인해 멀티링구얼 LLM의 성능 향상 가능성을 제시합니다.



### Reinforcement Learning with Euclidean Data Augmentation for State-Based Continuous Contro (https://arxiv.org/abs/2410.12983)
- **What's New**: 이 논문은 강화 학습(RL) 에이전트의 데이터 효율성을 높이기 위해 유클리드 대칭(Euclidean symmetries) 기반의 데이터 증강(data augmentation) 접근법을 제안합니다. 기존의 방법들이 이미지 기반 데이터 증강에 중점을 두었던 반면, 본 연구는 상태 기반(control state) 데이터 증강에 중점을 둡니다.

- **Technical Details**: 유클리드 데이터 증강은 물리적으로 관찰 가능한 위치(position)와 속도(velocity)와 같은 상태 기반 피처를 활용하여 이루어집니다. 반면 기존의 대칭 기반 변환은 조인트(joint) 구성으로만 이루어져 데이터 증강이 효과적이지 않았습니다. 본 연구에서는 팔다리의 구성(configuration)을 새로운 상태 표현으로 사용하여 더 많은 데이터를 생성하고, 임의의 회전(rotation)이나 이동(translation) 변환을 통해 데이터를 증강합니다.

- **Performance Highlights**: 개별 상태 표현을 사용했을 때, DeepMind Control Suite의 대부분의 작업에 대한 성능이 향상되었으며, 유클리드 데이터 증강 추가 시 거의 모든 작업에서 최적의 성능을 달성했습니다. 예를 들어, Humanoid_run 작업에서 표준 DDPG는 100 이하의 보상을 달성한 반면, 본 방법은 5M 타임스텝 후에 150 이상의 보상을 달성했습니다.



### Flash Inference: Near Linear Time Inference for Long Convolution Sequence Models and Beyond (https://arxiv.org/abs/2410.12982)
Comments:
          15 pages, 9 figures, 5 algorithms

- **What's New**: 이 논문에서는 Long Convolution Sequence Models (LCSMs), 특히 Hyena 모델의 정확한 추론(inference) 속도를 O(L log² L)로 증가시키는 방법을 제안합니다. 또한, 이러한 속도 향상이 가능한 주요 속성들을 정의하고, 이러한 속성을 활용하는 일반적인 프레임워크를 제안합니다.

- **Technical Details**: 제안된 접근 방식은 relaxed polynomial interpolation에 대한 이전 연구를 바탕으로 하며, 메모리 이동을 줄이고 계산을 공유하는 tiling 기법을 활용합니다. 이 방법은 position-mixing 부분의 아키텍처에서 거의 완전한 병렬화(parallelization)를 허용합니다.

- **Performance Highlights**: Hyena 모델의 실험적 구현을 통해, 표준 추론에 비해 최대 1.6배의 엔드 투 엔드(end-to-end) 시간 효율성을 개선하였고, position-mixing 부분에서는 최대 50배의 성능 향상을 달성했습니다.



### Long-Tailed Backdoor Attack Using Dynamic Data Augmentation Operations (https://arxiv.org/abs/2410.12955)
- **What's New**: 본 논문은 긴 꼬리(long-tailed) 데이터셋에 대한 백도어 공격(backdoor attack)을 처음으로 탐구합니다. 기존의 백도어 공격은 주로 균형 잡힌 데이터셋에 초점을 맞추었으며, 이로 인해 실제 환경에서 발생하는 불균형 데이터 문제를 간과하고 있었습니다.

- **Technical Details**: 제안된 방법인 D$^2$AO(Dynamic Data Augmentation Operation)는 클래스, 샘플 유형(클린 vs. 백도어), 그리고 샘플 특징에 따라 동적으로 다양하고 적절한 데이터 증강(data augmentation) 연산을 선택합니다. 이를 통해 백도어 샘플과 클린 샘플의 불균형 문제를 해결하고, 데이터 증강에 적응할 수 있는 트리거 생성기를 개발하였습니다.

- **Performance Highlights**: CIFAR10-LT 및 CIFAR100-LT와 같은 두 개의 긴 꼬리 벤치마크에서 폭넓은 실험을 수행하였으며, 제안된 방법은 기존의 백도어 공격 방법과 비교하여 상태-of-the-art 공격 성능을 달성하면서 클린 정확도(clean accuracy)를 유지하였습니다.



### A Note on Shumailov et al. (2024): `AI Models Collapse When Trained on Recursively Generated Data' (https://arxiv.org/abs/2410.12954)
Comments:
          Comment on this https URL

- **What's New**: Shumailov et al. (2024)의 연구에 따르면, 합성 데이터에 반복적으로 훈련된 생성 모델이 모델 붕괴(model collapse)를 일으킬 수 있다는 사실이 밝혀졌습니다. 이 연구는 현재 모델들이 기존 데이터의 활용 가능성을 거의 소진한 가운데 이루어져 큰 주목을 받고 있습니다.

- **Technical Details**: 연구는 데이터에 대한 적합(distribution fitting) 및 반복적인 샘플링을 통해 모델 붕괴의 원인을 조사합니다. Kernel Density Estimation (KDE)와 KL divergence 및 Wasserstein distance (WSD)와 같은 거리 메트릭을 사용하여 결과를 분석했습니다.

- **Performance Highlights**: 결과는 최종 분포(final distribution)가 붕괴되며, 이 과정에서 샘플링과 적합이 반복될수록 원래 데이터의 구조가 점차 유실되고 더 균일한 분포로 수렴한다는 것을 보여줍니다. 연구는 생성 모델이 데이터의 분포를 정확히 재현하는 데 한계가 있음을 강조하며, 향후 연구의 필요성을 제기합니다.



### Gradient Map-Assisted Head and Neck Tumor Segmentation: A Pre-RT to Mid-RT Approach in MRI-Guided Radiotherapy (https://arxiv.org/abs/2410.12941)
- **What's New**: 이번 연구는 방사선 치료 (RT)에서 두경부 암의 종양 구획 정확성을 향상시키기 위해 사전 방사선 치료 이미지와 지역 그래디언트 맵을 활용한 새로운 방법을 제안합니다. 이는 기존의 수동 분할 방식의 한계를 극복하기 위한 목적으로, MRI 유도 적응 방사선 치료에 적용됩니다.

- **Technical Details**: 본 연구에서는 nnUNet 프레임워크를 사용하여 모델을 구현하고 훈련했습니다. 방사선 치료 전(pre-RT) 이미지의 변형 등록된 구획을 기반으로 종양 주변의 관심 영역 (ROIs)을 정의한 후, 이를 통해 미드 방사선 치료(mid-RT) T2 강도 이미지를 처리하여 그래디언트 맵을 생성했습니다. 이러한 방법을 통해 종양의 경계 구획을 향상시키고자 하였습니다.

- **Performance Highlights**: 본 연구의 최종 DSCagg 점수는 GTVp(주 종양)에서 0.534, GTVn(림프절 종양)에서 0.867로 나타났으며, 평균 점수는 0.70을 기록했습니다. 이는 적응 방사선 치료의 분할 및 치료 계획 강화에 기여할 잠재력을 가지고 있습니다.



### UMambaAdj: Advancing GTV Segmentation for Head and Neck Cancer in MRI-Guided RT with UMamba and nnU-Net ResEnc Planner (https://arxiv.org/abs/2410.12940)
- **What's New**: 본 연구는 MRI 유도 적응 방사선 치료에서 두 가지 최신 심층 학습 세분화 기법, UMamba와 nnU-Net Residual Encoder(ResEnc)를 통합하여 'UMambaAdj'라는 새로운 접근 방식을 제안합니다. 이 방법은 주두경부암 치료에서 종양 부피(GTV)를 더 정밀하게 세분화하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: UMambaAdj는 3D ResEnc U-Net과 Mamba 블록을 결합하여 구성됩니다. CNN 구조는 nnU-Net Residual Encoder Planner에 따라 설정되며, 6단계의 U-Net은 총 6개의 잔차 CNN 블록을 포함하여 GTVp와 GTVn의 세분화를 위해 설계되었습니다. Mamba 레이어는 입력 이미지 특징 맵을 처리하여 장기 의존성을 효과적으로 캡처합니다.

- **Performance Highlights**: MNTS-MRG 2024 챌린지 테스트 세트에서 GTVp에 대해 0.751, GTVn에 대해 0.842의 Dice 유사도 계수(DSC)를 달성했으며, 평균 DSC는 0.796였습니다. 이는 MRI 유도 적응 방사선 치료에서 보다 정밀한 종양 윤곽을 제공하며 HNC 환자의 치료 결과를 향상시킬 가능성을 보여줍니다.



### SoK: On Finding Common Ground in Loss Landscapes Using Deep Model Merging Techniques (https://arxiv.org/abs/2410.12927)
- **What's New**: 이번 연구에서는 신경망의 해석 가능성을 향상시키고, 모델 병합(model merging)이라는 관련 분야의 문헌을 조사하여 신뢰할 수 있는 딥러닝 모델을 개발하기 위한 새로운 통찰력을 제시합니다.

- **Technical Details**: 모델 병합은 여러 신경망의 파라미터를 결합하여 성능이 뛰어난 단일 예측 모델을 만들어내는 기술입니다. 본 연구에서는 손실 경관(loss landscape geometry) 관점에서 모델 병합 기술을 분석하며, 이를 통해 해석 가능성, 보안 및 모델 훈련에 대한 새로운 이해를 제공합니다.

- **Performance Highlights**: 모델 병합 기술은 다양한 신경망을 효율적으로 조합하여 훨씬 더 우수한 성능의 모델을 생성할 수 있는 잠재력을 지니고 있으며, 모델 해석과 보안 분야에서의 의미 있는 연결점을 발견하였습니다.



### Boosting Asynchronous Decentralized Learning with Model Fragmentation (https://arxiv.org/abs/2410.12918)
- **What's New**: 이번 논문에서는 비동기식 분산 학습(Decentralized Learning, DL) 알고리즘인 DivShare를 제안합니다. DivShare는 느린 통신을 겪고 있는 노드들, 즉 stragglers를 효율적으로 처리하여 모델 수렴 속도를 개선합니다.

- **Technical Details**: DivShare는 모델을 파라미터 서브셋으로 나누고, 각 서브셋을 다른 노드에 무작위로 전송하여 병렬로 계산하는 방식을 사용합니다. 이로 인해 집합적인 대역폭을 효율적으로 활용하며, 느린 네트워크를 갖는 노드도 모델 파라미터의 일부를 빠르게 기여할 수 있게 됩니다. 또한, 논문에서는 비동기 통신과 지연의 영향을 고려한 DL 알고리즘의 수렴에 대한 첫 번째 이론적 증명을 제공합니다.

- **Performance Highlights**: CIFAR-10 데이터셋에서 DivShare는 AD-PSGD 대비 최대 3.9배 빠른 시간 내에 정확도에 도달하며, 두 가지 기준선에 비해 최대 19.4% 더 높은 정확도와 CIFAR-10 및 MovieLens 데이터셋에서 각각 9.5% 낮은 테스트 손실을 기록합니다.



### Fair Clustering for Data Summarization: Improved Approximation Algorithms and Complexity Insights (https://arxiv.org/abs/2410.12913)
- **What's New**: 이 연구는 공정한 데이터 요약(fair data summarization) 문제를 다루며, 기존의 $k$-supplier 문제를 공정성을 고려하여 확장한 점이 새롭습니다.

- **Technical Details**: 연구에서는 공정한 $k$-supplier 문제를 정의합니다. 이 문제는 데이터가 여러 그룹으로 구성되고, 각 그룹에서 최소한의 중심(center)을 선택해야 하며, $k$-supplier 목표를 최소화해야 합니다. 두 가지 문제 변형에 대해 각각 알고리즘을 제시하며, 비지지(disjoint) 그룹에 대해 다항식(polynomial) 시간 복잡도를 보이고, 겹치는(overlapping) 그룹에 대해서는 고정-매개변수 고찰(fixed-parameter tractable) 알고리즘을 제공합니다.

- **Performance Highlights**: 비지지 그룹에 대한 알고리즘은 시간 복잡도가 다항식으로 실행되며, 겹치는 그룹에 대한 알고리즘은 중심과 그룹의 수에만 의존하는 지수(exponential) 실행 시간을 가집니다. 제안된 알고리즘은 기존의 $5$보다 개선된 $3$-근사화(approximation) 알고리즘을 제공하며, 이 근사화 계수는 이론적인 하한(lower bound)과 일치합니다.



### Large Language Models and the Rationalist Empiricist Deba (https://arxiv.org/abs/2410.12895)
- **What's New**: 이 논문은 LLMs(대형 언어 모델)가 Chomsky와 Quine, Skinner 간의 논쟁에 어떻게 영향을 미치는지를 탐구하며, LLMs가 합리주의를 정당화하는 주장과 기존의 경험주의에 대한 비판을 다룬다.

- **Technical Details**: LLMs는 본래의 편향을 내장해야 하며, 이는 언어능력(linguistic competence)을 설명하는 데 있어 경험주의가 개념적 자원을 부족하다는 주장을 뒷받침한다. 그러나 이러한 주장은 사용되는 경험주의의 성격에 의존한다.

- **Performance Highlights**: 인간은 한정된 자극(poverty of stimulus) 속에서도 학습하는 반면, LLMs는 풍부한 자극(rich stimulus) 덕분에 학습한다. 이는 인간과 LLMs가 출력(output)을 생성하는 데에 있어 다른 기본 능력(underlying competencies)을 사용함을 나타낸다.



### MIRROR: A Novel Approach for the Automated Evaluation of Open-Ended Question Generation (https://arxiv.org/abs/2410.12893)
Comments:
          Accepted at FM-Eduassess @ NEURIPS 2024 (ORAL Paper)

- **What's New**: 이번 연구에서는 자동 질문 생성(Automated Question Generation, AQG) 시스템이 생성한 질문의 품질 평가를 자동화하기 위해 대규모 언어 모델(LLM)을 활용하는 새로운 시스템인 MIRROR (Multi-LLM Iterative Review and Response for Optimized Rating)를 제안합니다.

- **Technical Details**: MIRROR는 여러 LLM에 피드백을 제공하여 인간의 평가 지표(grammaticality, relevance, appropriateness, novelty, complexity)에 기반하여 점수를 생성하는 프로세스를 포함합니다. GPT-4, Gemini, Llama2-70b와 같은 최첨단 LLM을 사용하여 실험을 진행하였으며, 인간 전문가의 평가와의 Pearson 상관 계수(Pearson's correlation coefficient)를 측정하여 결과를 비교하였습니다.

- **Performance Highlights**: MIRROR를 적용한 결과, relevance, appropriateness, novelty, complexity, grammaticality와 같은 인간 평가 지표의 점수가 개선되어 인간 기준 점수와 더 가까운 결과를 보였습니다. 더불어 직접 프롬프트를 사용하여 평가한 경우보다 인간 전문가와의 상관 계수가 향상되었습니다.



### Multi-trait User Simulation with Adaptive Decoding for Conversational Task Assistants (https://arxiv.org/abs/2410.12891)
Comments:
          Preprint fron EMNLP 2024 Findings

- **What's New**: 이 논문은 Multi-Trait Adaptive Decoding (mTAD)이라는 새로운 접근 방식을 제안합니다. mTAD는 다양한 trait-specific Language Models (LMs)에서 샘플링하여 디코딩 시간에 다양한 사용자 프로필을 생성하여 사용자 시뮬레이션을 개선합니다.

- **Technical Details**: mTAD는 다양한 대화 trait를 모델링하기 위해 specialized LMs를 결합하는 모델 기반 접근 방식을 따릅니다. 기존의 조합 훈련 데이터나 추가적인 모델 Fine-tuning 없이, trait-specific LM에서 분포를 샘플링하여 동적으로 결합합니다. 이 기법은 사용자 대화 프로필의 다양한 조합을 가능하게 하여 더 풍부한 대화 패턴을 생성합니다.

- **Performance Highlights**: 실험 결과, mTAD는 단일 trait 모델링에서 효과적임을 입증하며, 아울러 특정 패턴을 포착할 수 있는 능력을 보여줍니다. mTAD는 다양한 사용자 시뮬레이터를 결합하는 데 강력하고 유연한 프레임워크로, 기존 LM을 재훈련할 필요 없이 새로운 traits를 추가할 수 있습니다.



### Using Protected Attributes to Consider Fairness in Multi-Agent Systems (https://arxiv.org/abs/2410.12889)
- **What's New**: 본 논문에서는 다중 에이전트 시스템(Multi-Agent Systems, MAS) 내에서 에이전트들에게 기대 보상에 불리한 영향을 미치지 않아야 하는 보호 속성(protected attributes)의 개념을 도입합니다. 이는 결정적인 보상 분배 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 알고리즘 공정성 문헌에서 발전된 공정성 메트릭스를 MAS에 적용합니다. 구체적으로 인구 통계적 평등(demographic parity), 반사적 공정성(counterfactual fairness), 그리고 조건부 통계적 평등(conditional statistical parity)이라는 세 가지 공정성 기준을 소개합니다. 이를 통해 MAS의 공정성을 평가하고 최적화하는 방법을 제안합니다.

- **Performance Highlights**: 이 연구는 공정성을 고려한 MAS의 설계를 위한 비전을 제시하며, AI와 인간 에이전트가 혼합된 도시 환경에서의 사례를 통해 불공정한 결과를 최소화할 수 있는 방안을 모색합니다. 예를 들어, 도로 인프라를 변경하여 인간이 조작하는 차량에 대한 공정성을 향상시키는 방안이 검토됩니다.



### Navigating the Cultural Kaleidoscope: A Hitchhiker's Guide to Sensitivity in Large Language Models (https://arxiv.org/abs/2410.12880)
- **What's New**: 이 논문은 글로벌 AI 애플리케이션에서 LLMs의 문화적 민감성을 보장하는 중요성을 강조하며, 작은 매개변수 모델 내에서 발생하는 문화적 손해를 다루기 위한 두 개의 주요 기여를 제시합니다. 첫째, 다양한 문화적 맥락에서 모델의 출력을 평가하기 위한 문화적 손해 테스트 데이터셋을 소개합니다. 둘째, 다양한 주석자 피드백을 기반으로 문화적 민감성을 회복하기 위한 데이터셋을 제안합니다.

- **Technical Details**: 이 연구는 문화적 손해 평가 데이터셋과 문화적 정렬 선호 데이터셋을 구축하여 작은 매개변수 LLMs의 문화적 민감성을 높이고 해로운 출력을 줄이는 것을 목표로 합니다. 데이터셋은 사회적, 정치적, 경제적, 종교적, 문화적 가치를 반영하며, 다양한 문화적 맥락에서 모델 출력을 시스템적으로 평가할 수 있는 프레임워크를 제공합니다. 또한, reinforcement learning from human feedback (RLHF) 기법을 사용하여 문화적 기준을 존중하는 모델의 미세 조정을 지원합니다.

- **Performance Highlights**: 문화적 정렬 피드백을 통합함으로써 Mistral-v0.2(7B) 모델의 해로운 출력 발생률이 71.96%에서 3.07%로 급격히 감소하는 등의 성과를 보였습니다. 이 연구는 LLM이 다양한 문화적 경관에서 안전하고 윤리적으로 탐색할 수 있는 미래의 AI 시스템을 구축하는 데 기여할 것입니다.



### Towards More Effective Table-to-Text Generation: Assessing In-Context Learning and Self-Evaluation with Open-Source Models (https://arxiv.org/abs/2410.12878)
Comments:
          15 pages

- **What's New**: 이 연구는 자연어 처리의 핵심 작업인 테이블-텍스트 생성(table-to-text generation)에 대해, 다양한 in-context learning 전략의 효과를 평가합니다. 특히, 모델에 주어진 예제가 성능에 미치는 영향을 조사하고, 실제 애플리케이션을 기반으로 한 사례를 제공합니다.

- **Technical Details**: 모델은 zero-shot, single-shot, few-shot 프롬프트를 사용하여 테이블 데이터에서 내러티브 텍스트로 전환합니다. 이 연구에서는 두 개의 벤치마크 데이터셋인 WikiBio와 ToTTo에서 실험이 수행되었고, Llama 3와 Phi-3 모델을 사용하여 결과를 비교했습니다. 또한, GPT-4를 사용하여 초기 프롬프트를 생성하고, 이를 기반으로 최적화를 진행하였습니다.

- **Performance Highlights**: 예제를 제공함으로써 테이블-텍스트 생성의 성능이 크게 향상되었습니다. LLM 자가 평가 방법은 아직 인간의 판단과 일치도가 개선되어야 하지만, overall 성능 개선을 확인할 수 있었습니다.



### Improving Instruction-Following in Language Models through Activation Steering (https://arxiv.org/abs/2410.12877)
- **What's New**: 이 논문에서는 언어 모델(LLM)의 지침 따르기 능력을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이는 지침에 따라 모델의 동작을 조정하기 위해 지침별 벡터 표현을 파생하는 내용을 다룹니다.

- **Technical Details**: 이 연구는 입력의 지침이 없는 경우와 있는 경우의 활성화(activation)의 차이를 기반으로 벡터 표현을 계산하여 모델의 출력을 조작하는 방식입니다. 사용된 활성화 벡터는 출력 형식, 길이, 특정 단어 포함 여부 등 여러 조건을 모델이 준수하도록 유도합니다.

- **Performance Highlights**: 4개의 서로 다른 모델을 대상으로 한 실험을 통해, 이 방법이 지침을 명시적으로 제공하지 않아도 모델이 제약사항을 따르도록 도와주고, 지침이 있을 때도 성능을 향상시킬 수 있음을 보여주었습니다. 또한, 여러 지침을 동시에 적용할 수 있다는 것이 확인되었습니다.



### Beyond Right and Wrong: Mitigating Cold Start in Knowledge Tracing Using Large Language Model and Option Weigh (https://arxiv.org/abs/2410.12872)
Comments:
          11 pages

- **What's New**: 이 논문에서는 LOKT 모델을 소개하여 Knowledge Tracing (KT)의 콜드 스타트 문제를 해결합니다. LOKT는 대규모 언어모델(LLM)을 사용하여 적은 이전 데이터로도 학습자의 지식 상태를 추적하고 예측할 수 있는 방법론을 제시합니다.

- **Technical Details**: LOKT 모델은 전통적인 KT 모델에 옵션 가중치를 통합하여 단순한 정답/오답 분류를 넘어 학습자의 다양한 잘못된 응답을 분석합니다. 이를 통해 LLM이 언어 기반 정량적 정보를 활용하여 학습자의 이해도를 보다 정확하게 평가할 수 있도록 합니다.

- **Performance Highlights**: 5개의 공공 데이터셋을 사용한 실험에서 LOKT 모델은 이른 단계의 개인화 학습 도구를 지원하며, '학습자 콜드 스타트'와 '시스템 콜드 스타트' 상황에서도 높은 예측 정확도를 유지하는 것을 보여주었습니다.



### AI-Driven Autonomous Control of Proton-Boron Fusion Reactors Using Backpropagation Neural Networks (https://arxiv.org/abs/2410.12871)
- **What's New**: 본 연구는 프로톤-붕소(p-11B) 핵융합로에서 핵심 파라미터를 자율적으로 제어하기 위해 역전파 기반 신경망을 이용한 새로운 접근 방식을 제안합니다. 이 방법은 물리적 데이터를 기반으로 실시간 피드백과 학습을 통해 변화하는 플라즈마 조건에 적응하는 기능을 제공합니다.

- **Technical Details**: 제안된 AI 기반 제어 시스템은 역전파(Backpropagation)로 훈련된 심층 신경망(Deep Neural Network, DNN)을 활용하여 실시간으로 플라즈마 조건을 최적화합니다. 이 시스템은 플라즈마의 상태를 동적으로 조정하여 동적이고 비선형적인 고온 플라즈마를 안정적으로 유지하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 이 연구의 AI 시스템은 실제 데이터에서 지속적으로 학습함으로써 플라즈마 안정성을 크게 향상시키고 에너지 효율성을 최적화하며, 실제 지속 가능한 핵융합 에너지로의 경로를 가속화할 가능성을 제시합니다.



### Skill Learning Using Process Mining for Large Language Model Plan Generation (https://arxiv.org/abs/2410.12870)
Comments:
          12 pages, 5 figures, 2 tables, accepted at ICPM 2024'

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 계획 생성을 개선하기 위해 프로세스 마이닝( process mining ) 기법을 통합한 새로운 기술 학습 접근 방식을 소개합니다. 이 접근 방식은 계획 생성 과정의 효율성과 해석 가능성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 텍스트 베이스 LLM 플래너가 생성한 단순 시퀀스 대신, 프로세스 모델을 사용하여 구조화된 제어 흐름을 만들고 이를 통해 플래너의 능력을 향상시키는 방법을 제안합니다. 새로운 기술 학습 프레임워크에서는 Inductive Miner 알고리즘을 사용하여 일반적인 프로세스 모델을 추출합니다.

- **Performance Highlights**: 실험 결과, 제안한 기술 검색 방법이 특정 조건에서 기존의 정확도 기준을 초과하는 것으로 나타났으며, 유연한 기술 발견과 병렬 실행을 지원하여 성능이 향상되었습니다.



### Language Model Preference Evaluation with Multiple Weak Evaluators (https://arxiv.org/abs/2410.12869)
- **What's New**: 이 논문에서는 효율적인 평가 방식의 필요성을 강조하며 신뢰성 있는 LLM(대규모 언어 모델) 출력 평가를 위한 새로운 방법론인 GED(Preference Graph Ensemble and Denoise)를 소개합니다.

- **Technical Details**: GED는 두 가지 주요 단계로 구성됩니다: (1) 여러 LLM의 평가 결과를 통합하여 단일 preference graph(선호 그래프)를 만드는 graph ensemble과 (2) 반복적 패턴과 불일치를 제거하여 방향 비순환 그래프(DAG) 구조를 보장하는 graph denoising입니다.

- **Performance Highlights**: GED는 실험 결과에서 10개 벤치마크 데이터셋을 통해 기존 방법들보다 우수한 성능을 보였으며, 예를 들어, 응답 선택 작업에서 평균 4.51% 향상을 기록했습니다. GED는 약한 평가자(combiner) 조합을 통해 강한 평가자보다 뛰어난 성능을 보여, 평가 신뢰성을 높이고 모델 성능을 향상시키는 능력을 입증했습니다.



### Empowering Dysarthric Speech: Leveraging Advanced LLMs for Accurate Speech Correction and Multimodal Emotion Analysis (https://arxiv.org/abs/2410.12867)
Comments:
          19 pages, 6 figures, 3 tables

- **What's New**: 이번 논문은 뇌 손상으로 인해 발생하는 운동 언어 장애인 발음장애(dysarthria)의 인식 및 번역에 대한 새로운 접근 방식을 제시합니다. 이 연구는 발음장애를 가진 개인들이 보다 효과적으로 소통할 수 있도록 지원하기 위해 고급 언어 모델(large language models)을 활용합니다.

- **Technical Details**: 이 연구에서는 OpenAI Whisper 모델을 사용하여 발음장애의 음성을 텍스트로 변환한 후, LLaMA 3.1(70B) 및 Mistral 8x7B와 같은 모델을 미세 조정하여 왜곡된 입력으로부터 의도된 문장을 예측합니다. 데이터 세트는 TORGO 데이터 세트와 Google 음성 데이터를 결합하였으며, 감정 컨텍스트를 수작업으로 라벨링하여 모델 학습에 사용합니다.

- **Performance Highlights**: 제안된 시스템은 발음장애의 음성을 재구성하고 감정을 인식하는 데 있어 높은 정확도를 달성하며, 이는 실질적인 음성 데이터와 비교했을 때 눈에 띄는 발전을 보여줍니다. 이 접근 방식은 발음장애 사용자를 위한 보다 포괄적이며 효과적인 커뮤니케이션 지원 도구를 제공합니다.



### Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings (https://arxiv.org/abs/2410.12866)
Comments:
          Preprint V1 with 10 pages main text

- **What's New**: 최근 뇌-컴퓨터 인터페이스(BCI)의 발전으로 인해 두개내(recordings)에서 음조(lexical tones)를 해독하는 것이 가능해졌습니다. 이는 언어 손상으로 인해 의사소통 능력이 제한된 사람들에게 도움을 줄 수 있는 잠재력을 제공합니다. 하지만 생리적 및 기기적 요소로 인해 발생하는 데이터 이질성(data heterogeneity)은 통합적인 뇌 음조 해독에 상당한 도전 과제가 됩니다.

- **Technical Details**: 이 논문에서는 H2DiLR(Homogeneity-Heterogeneity Disentangled Learning for neural Representations)이라는 새로운 프레임워크를 도입하여, 여러 피험자의 두개내 기록에서 동질성과 이질성을 분리하고 학습합니다. 이 연구에서는 407개의 음절(syllables)을 포함하는 중국어 재료를 읽는 여러 참가자로부터 스테레오전자뇌전도(sEEG) 데이터를 수집했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 H2DiLR은 기존의 이질적인 해독 접근 방식보다 현저히 우수한 성능을 보임을 입증했습니다. 또한 H2DiLR이 신경 표현 학습 과정에서 동질성과 이질성을 효과적으로 포착함을 실증적으로 확인하였습니다.



### ELF-Gym: Evaluating Large Language Models Generated Features for Tabular Prediction (https://arxiv.org/abs/2410.12865)
- **What's New**: ELF-Gym 프레임워크를 통해 LLM이 생성한 feature의 품질을 정량적으로 평가하는 새로운 방법론을 제시합니다.

- **Technical Details**: ELF-Gym은 Kaggle 대회에서 수집한 251개의 'golden' features를 기준으로 LLMs의 feature 엔지니어링 능력을 평가합니다. 평가 과정에서 LLM이 생성한 features의 다운스트림 모델 성능과 전문가가 제작한 features와의 의미적, 기능적 유사성을 측정합니다.

- **Performance Highlights**: 최선의 경우, LLM은 'golden' features의 약 56%를 의미적으로 포착할 수 있지만, 복잡한 feature가 요구되는 데이터셋에서는 실패할 수도 있습니다.



### Investigating Implicit Bias in Large Language Models: A Large-Scale Study of Over 50 LLMs (https://arxiv.org/abs/2410.12864)
- **What's New**: 이번 연구에서는 최신 대규모 언어 모델(LLM)들이 내재된 편향(implicit bias)을 가지고 있으며, 이러한 편향이 개발된 모델의 크기나 복잡성이 증가함에 따라 강화되고 있다는 점을 강조합니다. 또한, 편향 완화(bias mitigation)가 모델 개발에서 보편적으로 우선시되지 않고 있다는 사실을 강조합니다.

- **Technical Details**: 연구진은 50개 이상의 LLM을 대상으로 LLM Implicit Association Test (IAT) Bias 및 LLM Decision Bias 측정을 통해 내재된 편향의 정도를 탐구했습니다. 이 연구는 대규모 실험을 통해 신 모델에서 더 높은 편향 수준을 관찰하였으며, 이는 합성 데이터의 사용 증가와 관련이 있을 수 있다고 가정합니다.

- **Performance Highlights**: 새로운 또는 더 큰 언어 모델들이 자동으로 편향 수준이 감소하지 않으며, 때때로 이전 모델들보다 높은 편향 점수를 나타내기도 했습니다. 이러한 발견은 공정하고 책임감 있는 AI 시스템 개발을 위한 편향 탐지 및 완화 전략의 필요성을 강조합니다.



### Scaled and Inter-token Relation Enhanced Transformer for Sample-restricted Residential NILM (https://arxiv.org/abs/2410.12861)
Comments:
          Submitted to 27th IEEE-ICCIT

- **What's New**: 이 논문은 Non-Intrusive Load Monitoring (NILM)에서 transformer 모델의 훈련을 개선하기 위한 새로운 두 가지 메커니즘을 제안합니다. 이는 작은 규모의 데이터셋에서 transformer의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 제안된 두 가지 메커니즘은 inter-token relation enhancement mechanism과 dynamic temperature tuning mechanism입니다. 첫 번째 메커니즘은 훈련 중에 토큰 유사성 행렬에서 intra-token의 중요도를 줄이고 inter-token에 집중도를 높입니다. 두 번째 메커니즘은 토큰 유사성 행렬에 대해 학습 가능한 온도 조정을 도입하여 고정 온도 값에 수반되는 과도한 smoothing 문제를 완화합니다.

- **Performance Highlights**: REDD 주거용 NILM 데이터셋을 사용한 실험 결과, 제안된 방법이 원래 transformer 모델보다 여러 가전 제품 유형에서 성능을 현저히 향상시키는 것으로 나타났습니다.



### LLMD: A Large Language Model for Interpreting Longitudinal Medical Records (https://arxiv.org/abs/2410.12860)
- **What's New**: LLMD는 환자의 의료 기록을 기반으로 의료 이력을 분석하도록 설계된 대규모 언어 모델이며, 의료 지식과 레이블이 지정된 장기 기록을 결합하여 정확한 환자 건강 정보를 제공한다.

- **Technical Details**: LLMD는 10년 이상의 치료 기록과 140개 이상의 치료 기관에서 수집된 대량의 데이터를 포함하여, 지속적인 프리트레이닝(pretraining)과 작업 기반 지침 세밀 조정(instruction fine-tuning)을 통해 훈련된다. 이 구조화(structuring) 및 추상화(abstraction) 작업은 의료 기록의 메타데이터와 임상명명 엔티티(clinical named-entities)를 식별하고 정규화하여 높은 수준의 표현으로 변환한다.

- **Performance Highlights**: LLMD-8B는 PubMedQA 텍스트 응답에서 최첨단 정확도를 달성하며, 기존의 크고 일반화된 모델 및 도메인 맞춤형 모델보다 우수한 성능을 보인다. 실제 환자 데이터를 분석할 때, 의료 지식이 아닌 프리트레이닝과 세밀 조정의 중요성을 강조하며 LLM의 의료 활용을 위한 격차에 대해 논의한다.



### Enhancing Long Context Performance in LLMs Through Inner Loop Query Mechanism (https://arxiv.org/abs/2410.12859)
- **What's New**: 이번 논문에서는 Inner Loop Memory Augmented Tree Retrieval (ILM-TR)이라는 혁신적인 접근법을 통해 복잡한 질문에 대한 보다 깊이 있는 답변 생성을 가능하게 하는 새로운 메모리 체계를 도입합니다. 이 메커니즘은 초기 질문뿐만 아니라 중간 결과에 기반한 내부 루프 쿼리를 활용하여 정보를 검색합니다.

- **Technical Details**: ILM-TR 방법은 기본적으로 두 부분으로 구성되어 있습니다: retriever와 inner-loop query. Retriever 부분에서는 RAPTOR의 트리 빌드 방법을 사용하여 원시 데이터를 짧고 연속적인 텍스트 청크로 분할하고, 각 청크의 요약을 생성합니다. Inner-loop 쿼리는 LLM을 사용하여 최종 답변을 생성하며, Short-Term Memory (STM)라는 영역에 정보를 저장하고, 전달된 데이터를 바탕으로 반복적으로 쿼리를 수행합니다.

- **Performance Highlights**: ILM-TR 시스템은 Multi-Needle In A Haystack (M-NIAH) 및 BABILong과 같은 표준 긴 컨텍스트 벤치마크에서 기존 RAG 방법을 초월하는 성능을 보여주며, 500k tokens까지 컨텍스트 길이가 증가해도 성능 저하 없이 지속적인 성능을 유지합니다.



### Large Language Models for Medical OSCE Assessment: A Novel Approach to Transcript Analysis (https://arxiv.org/abs/2410.12858)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 의료 기초 교육 과정에서의 학생의 의사소통 능력을 평가하는 가능성을 탐구하였습니다. 기존의 수작업 평가 방식에 비해 시간과 비용을 절감할 수 있는 자동화된 OSCE 평가 시스템을 제안합니다.

- **Technical Details**: 연구에서 2,027개의 OSCE 비디오 데이터를 활용하여 학생의 환자 의료 정보 요약 능력을 평가하였습니다. Whisper-v3를 사용하여 음성을 텍스트로 변환한 후, GPT-4를 포함한 다양한 LLM 기반 접근 방식을 통해 학생의 성과를 채점하였습니다. 연구에서는 zero-shot prompting, retrieval augmented generation 및 다중 모달 앙상블 기법을 적용하였습니다.

- **Performance Highlights**: GPT-4는 인간 채점자와의 코헨 카파(Cohen's kappa) 지수 0.88을 기록하여 LLM 기반 OSCE 채점의 가능성을 보여주었습니다. 오픈 소스 모델 또한 유망한 결과를 보였으며, 자동 채점 시스템의 구현 가능성을 제시하였습니다.



### Enterprise Benchmarks for Large Language Model Evaluation (https://arxiv.org/abs/2410.12857)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 평가를 위한 새로운 벤치마크를 제시합니다. 이는 금융 서비스, 법률, 사이버 보안 및 기후 변화와 지속 가능성과 같은 다양한 기업 도메인에서의 NLP 작업을 포함하는 25개의 공개 데이터셋을 활용합니다.

- **Technical Details**: 본 연구에서는 LLM 평가를 위한 프레임워크를 개발하여, 각 도메인에 맞는 성능 지표와 벤치마크를 제공합니다. 이 프레임워크는 Stanford의 HELM을 보강하여, 도메인별로 구체화된 벤치마크를 추가하고 이를 통해 LLM의 성능을 측정할 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: 13개의 모델을 다양한 기업 작업에 적용하여 성능을 평가한 결과, 특정 작업의 요구사항에 맞는 모델 선택의 중요성이 드러났습니다. 이 연구는 실질적인 기업 애플리케이션의 요구를 반영한 벤치마크와 평가 메트릭을 통해 LLM의 최적화를 도울 것으로 기대됩니다.



### Optimized Biomedical Question-Answering Services with LLM and Multi-BERT Integration (https://arxiv.org/abs/2410.12856)
Comments:
          10 pages, 12 figures, accepted and to be published in the proceedings of 2024 IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)과 Multi-BERT 구성을 통합하여 생물의학적 질문-응답(QA) 서비스를 개선하는 정교한 접근 방식을 제안합니다. 이 시스템은 복잡한 생물의학 데이터의 방대한 양을 처리하고 우선 순위를 매기는 능력을 향상시켜 의료 전문가들이 더 나은 환자 결과 및 정보에 기반한 의사 결정을 내릴 수 있도록 지원하는 것을 목표로 합니다.

- **Technical Details**: BERT(Bidirectional Encoder Representations from Transformers) 및 BioBERT 모델의 혁신적인 사용과 다층 퍼셉트론(MLP) 레이어의 결합을 통해, 의료 부문의 증가하는 요구에 대해 보다 전문화되고 효율적인 응답을 제공합니다. 이 접근 방식은 과적합(overfitting) 문제를 해결하기 위해 하나의 BERT 모델을 동결(freeze)하면서 다른 모델을 훈련(training)하는 방법을 사용하여 QA 서비스의 전반적인 적응성을 개선합니다.

- **Performance Highlights**: BioASQ 및 BioMRC와 같은 대규모 데이터셋을 사용하여 QA 서비스 성능의 주요 지표에서 상당한 개선을 나타내는 것을 입증합니다. 이 작업은 고급 언어 모델이 의료 분야에서 실질적인 차이를 만들 수 있는 방법을 강조하며, 복잡한 정보를 관리하는 전문가들을 위해 신뢰할 수 있고 반응성이 뛰어난 도구를 제공합니다.



### JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework (https://arxiv.org/abs/2410.12855)
- **What's New**: 이번 연구에서는 jailbreak 공격에 대한 LLMs의 방어력 평가를 위한 새로운 벤치마크인 JAILJUDGE를 제안합니다. 이 벤치마크는 다양한 리스크 시나리오를 포함하고 있으며, 고품질의 인간 주석이 포함된 데이터셋으로 구성되어 있습니다.

- **Technical Details**: JAILJUDGE 데이터셋은 35k 이상의 instruction-tune 데이터를 포함하며, JailJudge MultiAgent 프레임워크를 통해 명시적 reasoning(추론)을 바탕으로 한 세밀한 평가가 가능합니다. JAILJUDGE Guard는 instruction-tuning된 종합적인 평가 모델로 비용 없이 reasonability 설명을 제공합니다.

- **Performance Highlights**: JailJudge 메소드의 성능은 다양한 모델(GPT-4, Llama-Guard 등)에서 최첨단을 나타냅니다. JailBoost는 성능을 29.24% 향상시켰고, GuardShield는 방어 ASR을 40.46%에서 0.15%로 감소시켰습니다.



### TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees (https://arxiv.org/abs/2410.12854)
- **What's New**: 본 연구에서는 기존의 DPO(Direct Preference Optimization) 알고리즘에서 발생하는 한계를 극복하기 위해 TPO(Tree Preference Optimization)를 제안합니다. TPO는 선호 트리(preference tree)에서 대응하는 응답을 샘플링하는 대신, 전체 선호 트리로부터 직접 학습합니다.

- **Technical Details**: TPO는 언어 모델 정렬을 Preference List Ranking 문제로 정의하며, 이는 주어진 프롬프트에 대한 응답의 순위가 매겨진 선호 리스트로부터 더 효과적으로 학습할 수 있도록 합니다. 또한, Adaptive Step Reward를 사용하여 긴 체인의 추론에서 LLM이 차별화된 단계를 인식하는 데 도움을 주고, 각 단계의 보상 값(reward values)을 조정하여 세밀한 선호 최적화(fine-grained preference optimization)를 수행합니다.

- **Performance Highlights**: TPO는 수학적 추론(task)에서의 실험을 통해 DPO보다 세 가지 공개 대형 언어 모델에 대해 네 개의 데이터셋에서 일관되게 우수한 성능을 보였습니다.



### Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks (https://arxiv.org/abs/2410.12853)
Comments:
          11 pages, 9 figures

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력과 사실 정확성을 개선하기 위한 다중 에이전트 토론(multi-agent debate) 프레임워크를 제안합니다. 특히, 다양한 모델을 활용한 경우에 더 뛰어난 성능을 발휘했으며, GPT-4와 비교하여 더 높은 정확성을 기록하였습니다.

- **Technical Details**: 다중 에이전트 토론 프레임워크는 질문 인코딩, 토론 모델, 토론 라운드, 응답 요약, 반복적 정제 및 최종 요약의 여섯 가지 주요 구성 요소로 이루어져 있습니다. 이 과정에서 다양한 모델 아키텍처를 활용하여 각 모델의 사고 다양성에 기반한 강력한 논리를 생성합니다.

- **Performance Highlights**: 이 연구에서 사용한 중간 용량 모델 세트(Gemini-Pro, Mixtral 7BX8,와 PaLM 2-M)는 4회 토론 후 GSM-8K 벤치마크에서 91%의 정확도를 기록하여 GPT-4를 초월하였고, ASDiv 벤치마크에서는 94%로 새로운 최고 기록을 세웠습니다.



### VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models (https://arxiv.org/abs/2410.12851)
Comments:
          10 pages, unironic use of the word 'vibe'

- **What's New**: VibeCheck는 대형 언어 모델(LLMs) 간의 뚜렷한 특성(vibes)을 발견하고 측정하는 시스템으로, 모델의 출력에서 다양한 차원(ton, formatting, writing style)을 평가할 수 있는 새로운 방식입니다. 이 시스템은 사용자의 선호와 모델의 정체성을 예측할 수 있는 vibres를 자동으로 확인합니다.

- **Technical Details**: VibeCheck는 모델의 출력을 통해 vibes를 반복적으로 발견하고, LLM 판별자를 통해 각 vibe의 유용성을 정량적으로 측정합니다. 발견된 vibes는 다수의 사용자의 합의, 모델 간 차별화, 사용자 선호 예측 세 가지 기준을 충족해야 합니다. VibeCheck는 Llama-3-70b와 GPT-4의 사용자 대화 데이터를 기반으로 실험했으며, 80%의 정체성 예측 정확도와 61%의 사용자 선호 예측 정확도를 달성했습니다.

- **Performance Highlights**: VibeCheck의 결과에 따르면, Llama는 친근하고 유머러스하며 다소 논란이 있는 vibe를 가지며, Command X는 요약 시 구체적인 서론과 결론을 추가하는 경향이 있고, Llama-405b는 수학 문제에서 자신의 사고 과정을 과도하게 설명하는 경향이 있는 반면, GPT-4는 캡셔닝에서 장면의 정서와 분위기에 집중하는 경향이 있음을 확인했습니다.



### RecurFormer: Not All Transformer Heads Need Self-Attention (https://arxiv.org/abs/2410.12850)
- **What's New**: 이 논문에서는 Transformer 기반의 대형 언어 모델(LLM)의 응답 생성 과정에서 발생하는 계산 비용 문제를 해결하기 위해 RecurFormer라는 새로운 아키텍처를 제안합니다. RecurFormer는 특정 attention head를 linear recurrent neural network (RNN)인 Mamba 아키텍처로 교체하여 메모리 캐시 사이즈를 줄이고, 토큰을 제거하지 않으면서 생성 품질을 유지합니다.

- **Technical Details**: RecurFormer는 recency aware 속성을 가진 attention head를 Mamba 아키텍처로 교체하는 방식으로 구성되어 있습니다. Mamba는 selective structured state-space sequence model 기반의 linear RNN으로, parallel 및 recursive 계산을 지원합니다. 이 방식은 기존 Transformer의 가중치를 계속 활용할 수 있도록 하여 모델의 성능을 유지하면서도 계산 효율을 증대시킵니다.

- **Performance Highlights**: 실험 결과, RecurFormer는 원래 모델의 성능을 유지하면서도 추론 효율성을 크게 향상시키는 것으로 나타났습니다. 또한, 지속적인 훈련을 통해 성능 회복이 가능하다는 것을 보여주어, 긴 입력에 관련된 작업에서 Transformer 기반 LLM의 계산적 도전에 대한 실용적인 해결책을 제공합니다.



### Prompt Engineering a Schizophrenia Chatbot: Utilizing a Multi-Agent Approach for Enhanced Compliance with Prompt Instructions (https://arxiv.org/abs/2410.12848)
- **What's New**: 이 논문은 정신분열증 환자를 위한 교육 플랫폼에서 Large Language Models (LLMs)인 GPT-4를 활용하는 방법을 제안합니다. 특히, 챗봇의 반응이 초기에 설정된 범위를 넘는 경우를 다루기 위해 Critical Analysis Filter를 도입했습니다.

- **Technical Details**: 이 시스템은 여러 LLM 에이전트가 챗봇의 반응을 분석하고 개선하는 역할을 합니다. 실험에서는 정보 제공 목적의 정신분열증 챗봇을 개발하고, 필터가 비활성화된 상태에서 대화를 진행하여 챗봇의 범위를 초과하는 모습을 관찰했습니다. 이후 AI 에이전트를 통해 범위를 벗어난 주제에 대한 샘플 대화를 자동 생성하고, 각 반응에 대해 컴플라이언스 점수를 할당했습니다.

- **Performance Highlights**: Critical Analysis Filter를 활성화했을 때 챗봇의 컴플라이언스 점수는 67.0%에서 적정 수준(점수 >=2)을 유지했지만, 필터가 비활성화된 경우에는 단지 8.7%에 불과했습니다. 이는 정신 건강 플랫폼에서 LLM을 효과적이고 안전하게 사용하기 위한 자기 반성 계층의 필요성을 시사합니다.



### ACCEPT: Adaptive Codebook for Composite and Efficient Prompt Tuning (https://arxiv.org/abs/2410.12847)
Comments:
          EMNLP Finding 2024

- **What's New**: 이 연구에서는 Adaptive Codebook for Composite and Efficient Prompt Tuning (ACCEPT)이라는 새로운 방법을 제안합니다. 기존의 Prompt Tuning (PT) 기법이 개별적으로 업데이트되는 프롬프트로 인해 파라미터 수가 비례적으로 증가하는 문제를 해결하여 모든 소프트 프롬프트가 학습 가능한 코드북 벡터를 공유하도록 하여 파라미터 효율성을 높입니다.

- **Technical Details**: ACCEPT는 제품 양자화(Product Quantization, PQ) 개념을 바탕으로 하며, 각 프롬프트의 단어 임베딩을 여러 하위 섹션으로 나누어 각각의 섹션에 대해 코드북을 구성합니다. 이 방법은 프롬프트의 각 하위 벡터가 선형 계수를 통해 부드럽게 결합되도록 하여 더 높은 다양성과 유연성을 제공합니다. 또한, ACCEPT는 0.3%의 플로우우먼스 파라미터만 조정하여 17개의 자연어 작업에서 우수한 성능을 달성합니다.

- **Performance Highlights**: 17개의 다양한 자연어 작업에서 ACCEPT 방법이 이전의 PT 접근법을 일관되게 초과 달성했습니다. 특히, 몇 가지 샷(few-shot) 및 대형 모델 환경에서 뛰어난 성능을 보여주며, 사전 훈련된 언어 모델(PLMs)의 효율성을 극대화합니다.



### Accurate and Regret-aware Numerical Problem Solver for Tabular Question Answering (https://arxiv.org/abs/2410.12846)
- **What's New**: TabLaP라는 모델을 제안하여, Large Language Model (LLM)을 답변 생성기가 아닌 계획자로 활용하며, 숫자 계산을 위한 정확한 처리기인 Python interpreter에게 계산을 맡깁니다. 또한, TabLaP가 생성한 답변의 신뢰성을 정량화하여 사용자가 후회 유발 가능성을 줄일 수 있도록 합니다.

- **Technical Details**: TabLaP는 두 개의 모델 브랜치를 갖고 있으며, 하나는 NumSolver로 숫자 질문을 처리하고, 다른 하나는 최신 TableQA 모델입니다. 생성된 답변을 통합하기 위해 AnsSelecter라는 LLM을 사용하여 신뢰할 수 있는 브랜치를 선택합니다. TwEvaluator 모듈을 통해 각 브랜치의 정확도를 추적하여 답변 신뢰성을 평가합니다.

- **Performance Highlights**: TabLaP는 두 개의 벤치마크 데이터셋에서 기존의 SOTA 모델에 비해 각각 5.7%와 5.8% 향상된 정확도를 기록했습니다. 또한, TabLaP의 신뢰성 플래그는 사용자 후회 비율을 두 데이터셋에서 각각 30.9%와 20.6% 감소시켰습니다.



### Toward Relieving Clinician Burden by Automatically Generating Progress Notes using Interim Hospital Data (https://arxiv.org/abs/2410.12845)
Comments:
          Accepted at the AMIA 2024 Annual Symposium

- **What's New**: 이 논문에서는 전자 건강 기록(EHR)의 구조화된 정보를 활용하여 진행 노트 생성(Progress Note Generation, PNG) 자동화를 위한 새로운 방법론을 제안합니다. 특히, 1616명의 환자로부터 수집된 7089개의 주석 인스턴스를 포함한 대형 데이터셋 ChartPNG를 소개합니다.

- **Technical Details**: 이 연구는 임상 의사들이 작성한 SOAP 노트를 기반으로 하는 프로세스입니다. 진행 노트는 환자의 주관적 및 객관적 상태와 평가 및 계획(A&P)으로 구성되어 있으며, 연구는 A&P 섹션의 자동 생성을 주로 목표로 합니다. 이 과정에서 대형 언어 모델을 활용하여 자동 분석을 수행하고, 향후 연구 기회를 찾아내기 위해 오류 분석을 실시하였습니다.

- **Performance Highlights**: 자동화된 분석에서는 Biomistral 모델이 BERTScore F1 점수 80.53과 MEDCON 점수 19.61을 기록하였고, 수작업 분석에서는 76.9%의 정확도로 관련 구조화 데이터를 활용할 수 있음을 보여주었습니다.



### Exploring Prompt Engineering: A Systematic Review with SWOT Analysis (https://arxiv.org/abs/2410.12843)
Comments:
          14 pages, 1 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM) 내에서 프롬프트 엔지니어링(prompt engineering) 기술에 대한 포괄적인 SWOT 분석을 수행했습니다. 언어학 원칙을 강조하며, 다양한 기술들을 분석하여 그 강점, 약점, 기회 및 위협을 파악했습니다. 이러한 발견은 AI 상호작용을 향상시키고 언어 모델이 인간의 프롬프트를 이해하는 방법을 개선하는 데 기여합니다.

- **Technical Details**: 이 논문에서는 100편 이상의 관련 문헌을 조사하여 프롬프트 엔지니어링 분야에 대한 폭넓은 통찰을 제공합니다. 주요 프롬프트 엔지니어링 기술로는 템플릿 기반 접근법(template-based approaches)과 파인 튜닝(fine-tuning) 방식이 있으며, 각 기술의 문제점 및 도전 과제를 다루었습니다. 또한, BLEU, BERTScore, ROUGE 및 Perplexity와 같은 여러 평가 메트릭(metrics)을 확인했습니다. 연구는 언어 모델의 행동을 이해하는 데 도움을 주고, 맞춤형 상호작용을 제공하는 목표에 맞춰 진행되었습니다.

- **Performance Highlights**: 이 연구는 LLM의 정확성 및 관련성을 향상시킬 수 있는 효과적인 프롬프트 엔지니어링의 중요성을 강조하며, 사용자 및 개발자 간의 지식 공유와 대화형 AI 툴의 발전을 촉진합니다. 특히, 차별화된 접근법을 통해 응답 정확도를 높이고, 대화형 AI의 성장에 기여할 것입니다.



### A Two-Model Approach for Humour Style Recognition (https://arxiv.org/abs/2410.12842)
- **What's New**: 이번 연구에서는 1,463개의 인스턴스를 포함하는 새로운 텍스트 데이터셋을 도입하여 네 가지 유머 스타일(자기 증진, 자기 비하, 친화적, 공격적) 및 비유머 텍스트를 인식하는 데 필요한 기계 학습 모델링을 지원합니다. 이는 유머 스타일 인식의 연구 공백을 채우는 중요한 기여를 합니다.

- **Technical Details**: 연구에서는 고전 기계 학습 분류기, 텍스트 임베딩 모델 및 DistilBERT를 포함한 다양한 컴퓨팅 방법을 사용하여 기준 성능을 설정하였습니다. 또한, 친화적 유머 분류의 F1 점수를 11.61% 향상시키는 두 개의 모델 접근 방식을 제안하였습니다. 이 연구는 각 유머 스타일에 대한 다중 클래스 분류 문제를 다룹니다.

- **Performance Highlights**: 두 개의 모델 접근 방식을 통해 14개의 테스트된 모델에서 일관된 성능 개선을 보였으며, 특히 친화적 유머 분류에서 F1 점수의 11.61% 향상을 달성했습니다. 이는 문학, 소셜 미디어 및 다른 텍스트 출처에서 유머를 연구하기 위한 새로운 도구를 제공합니다.



### UniAutoML: A Human-Centered Framework for Unified Discriminative and Generative AutoML with Large Language Models (https://arxiv.org/abs/2410.12841)
Comments:
          24 pages

- **What's New**: 새로운 AutoML 프레임워크인 UniAutoML이 소개되었습니다. UniAutoML은 기존의 AutoML 프레임워크가 주로 다루었던 discriminative task 뿐만 아니라 generative task도 통합하여 지원하는 것이 특징입니다. 사용자가 쉽게 접근할 수 있도록 자연어로 상호작용할 수 있는 대화형 사용자 인터페이스(CUI)를 제공합니다.

- **Technical Details**: UniAutoML은 Large Language Models (LLMs)를 활용하여 데이터 처리, 모델 선택 및 하이퍼파라미터 검색을 자동화한 인공지능 프레임워크입니다. 사용자들은 자연어 명령을 통해 복잡한 모델을 fine-tuning 할 수 있으며, 모델은 HuggingFace에서 사전 훈련된 다양한 모델을 선택하고 사용할 수 있습니다. 또한, safety guard-line을 설계하여 사용자 입력과 LLM 출력의 필터링이 이루어집니다.

- **Performance Highlights**: UniAutoML은 25명의 참가자를 대상으로 8개의 다양한 데이터셋에 대한 실험을 통해 성능과 사용성을 평가하였고, 그 결과 사용자 제어와 신뢰도를 향상시켰습니다. UniAutoML의 인간 중심 디자인은 AutoML의 기능과 사용자 이해 사이의 격차를 해소하여 더 많은 사람들이 ML(Machine Learning)에 접근할 수 있도록 합니다.



### Capturing Bias Diversity in LLMs (https://arxiv.org/abs/2410.12839)
Comments:
          2nd International Conference on Foundation and Large Language Models (FLLM2024), 26-29 November, 2024 | Dubai, UAE

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 출력 다양성을 높이기 위해 여러 개의 사용자 정의 GPT 모델을 구성하여 BiasGPT라는 새로운 프레임워크를 제안하고 평가합니다. 이 모델들은 성별, 연령 및 인종 같은 특정 인구통계학적 특성의 편향을 반영하여 협력하고, 다양한 관점을 통합하여 인간의 경험을 보다 잘 캡처한 응답을 생성합니다.

- **Technical Details**: BiasGPT는 여러 개의 사용자 정의 GPT 모델을 사용하여 각 모델이 특정 인구 통계적 특성을 반영함으로써 다양한 응답을 생성하는 방법입니다. 이 방법론은 사용자 정의된 LLM을 통해 학습된 편향들이 통합되어 보다 포괄적이고 공정한 AI 챗봇 응답을 형성하도록 합니다. 또한, 논문에서는 대화 데이터 수집 과정에서 연령, 인종, 성별 기반의 다양한 편향을 다루기 위한 포괄적인 접근 방식을 사용합니다.

- **Performance Highlights**: 일련의 실험을 통해 BiasGPT는 다양한 사회적 특성을 반영한 응답을 생성할 수 있는 능력을 입증하였으며, 이는 더욱 포괄적이고 대표적인 AI 대화를 형성하는 데 기여할 것입니다. 이 연구는 AI 기술의 포용성을 높이는 방향으로 나아가는 데 주요한 실험적 근거를 제공합니다.



### A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions (https://arxiv.org/abs/2410.12837)
Comments:
          4 Figures

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG)의 발전 과정을 포괄적으로 조사하며, 기존 개념에서 최신 기술에 이르기까지의 변화를 설명합니다. RAG는 검색 메커니즘과 생성 언어 모델을 결합하여 출력의 정확성을 높이며, LLMs의 주요 제한 사항을 해결합니다.

- **Technical Details**: RAG의 기본 아키텍처는 지식 집약적인 작업을 처리하기 위해 검색과 생성을 어떻게 통합하는지에 중점을 둡니다. 논문에서는 retrieval-augmented language models에서의 주요 혁신과 질문 답변, 요약 및 지식 기반 작업 등 다양한 도메인에서의 응용 사례를 자세히 리뷰합니다.

- **Performance Highlights**: 최근 연구 성과는 retrieval 효율성을 개선하기 위한 새로운 방법을 강조하고 있으며, RAG의 연구 방향으로는 모델의 견고성 향상, RAG 모델의 적용 범위 확대 및 사회적 함의 문제 다루기가 제안됩니다.



### EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing (https://arxiv.org/abs/2410.12836)
- **What's New**: EditRoom은 자연어 명령을 통해 다양한 레이아웃 편집을 자동으로 수행할 수 있는 통합 프레임워크로, 수동 개입 없이 실행됩니다.

- **Technical Details**: EditRoom은 두 개의 주요 모듈인 Command Parameterizer와 Scene Editor로 구성되어 있습니다. Command Parameterizer는 사전 훈련된 LLM(GPT-4o)을 활용하여 자연어 명령을 여섯 가지 기본 편집 유형에 대한 분해 명령으로 변환합니다. Scene Editor는 소스 장면과 텍스트 명령을 조건으로 삼아 확산 기반(diffusion-based) 모델을 훈련하여 목표 장면을 생성합니다.

- **Performance Highlights**: 편집 작업에 대한 실험 결과, EditRoom은 모든 메트릭에서 다른 기준선보다 우수한 성능을 보였으며, 다중 작업 명령에 대해서도 일반화할 수 있는 능력을 보여줍니다.



### Segment as You Wish -- Free-Form Language-Based Segmentation for Medical Images (https://arxiv.org/abs/2410.12831)
- **What's New**: 이번 논문에서는 기존의 바운딩 박스나 포인트 기반의 프롬프트 대신 자연어 기반의 프롬프트를 활용하여 의료 이미지 분할(Medical Image Segmentation, MIS) 문제를 해결하는 새로운 접근 방식을 제안합니다. 이를 위해 RAG(임시 증강 생성) 기술을 이용한 자유형 텍스트 프롬프트 생성기를 개발하고, 다양한 텍스트 프롬프트를 처리할 수 있는 새 모델인 FLanS를 소개합니다.

- **Technical Details**: FLanS는 전문 해부학 기반 쿼리, 해부학 무관 위치 기반 쿼리, 해부학 무관 크기 기반 쿼리를 포함한 다양한 자유형 텍스트 프롬프트를 처리할 수 있는 모델입니다. 또한, 대칭 인지 캐노니컬화 모듈을 통해 스캔 방향에 따른 일관된 정확한 분할을 보장하며, 100,000개 이상의 의료 이미지로 훈련되었습니다.

- **Performance Highlights**: FLanS는 최근의 SOTA(State-of-the-Art) 모델들보다 우수한 언어 이해 능력과 분할 정밀도를 보여주었으며, 다양한 임상 환경에서의 응용 가능성을 입증했습니다. 논문에서는 자질 분석(ablation studies)을 통해 각 구성 요소의 기여도를 검증했습니다.



### Incorporating Metabolic Information into LLMs for Anomaly Detection in Clinical Time-Series (https://arxiv.org/abs/2410.12830)
- **What's New**: 이번 논문에서는 의료 분야에서의 데이터 분석을 위해 LLMs(Large Language Models)에 대한 도메인 지식을 통합한 새로운 기법인 Metabolism Pathway-driven Prompting(MPP)를 제안합니다. 이 방법론은 생물학 샘플의 구조적 및 시간상의 변화를 더 잘 포착하는데 기여합니다.

- **Technical Details**: 이 논문은 다변량 임상 시계열 데이터의 이상 탐지를 위한 방법론을 제시합니다. MPP는 대사 경로에 관한 정보와 다양한 대사물질의 시간적인 변화를 LLM에 통합하여, 시간 경과에 따른 대사물질 간의 의존성을 고려합니다. 이를 통해 특정 샘플에 대한 이상 점수를 부여하는 함수 f(xt)를 학습합니다.

- **Performance Highlights**: 결과적으로, 이 방법은 스포츠에서의 도핑 탐지 문제에 효과적으로 적용되며, 실제 데이터를 사용하여 의심스러운 표본의 발견 성능을 개선합니다. MPP는 기존의 제로샷 학습(zero-shot learning) 및 맥락 학습(in-context learning) 기법과 비교할 때 우수한 성과를 보였습니다.



### A transformer-based deep reinforcement learning approach to spatial navigation in a partially observable Morris Water Maz (https://arxiv.org/abs/2410.12820)
- **What's New**: 이번 연구는 Morris Water Maze (MWM) 실험을 재현하기 위해 transformer 기반 아키텍처를 이용한 딥 강화학습을 적용한 것입니다. 이는 기존 연구에서 다루지 않았던 접근법으로, 2D 미로에서 에이전트가 효과적으로 탐색할 수 있도록 합니다.

- **Technical Details**: 에이전트는 decoder-only transformer 아키텍처를 활용하여 부분 관찰 가능한 환경에서 Q-value를 예측합니다. 비교적 제한된 시각 정보를 가진 환경에서 효율적으로 학습하며, 그 결과 공간 탐색 전략을 습득하게 됩니다. 뉴럴 네트워크의 회귀 문제를 해결하기 위해 recurrent position encoding과 multi-head attention도 사용합니다.

- **Performance Highlights**: 제안된 transformer 아키텍처는 에이전트가 효율적으로 탐색 임무를 수행하도록 함으로써, 내부 환경 표현에 대한 이해도를 높일 수 있는 기회를 제공합니다. 특히, 보조 작업 없이도 빠르게 학습할 수 있는 능력을 보여주며, 생물학적 에이전트와 유사한 행동을 보일 수 있는 잠재력을 시사합니다.



### Interactive Explainable Anomaly Detection for Industrial Settings (https://arxiv.org/abs/2410.12817)
- **What's New**: 이 연구는 산업 환경에서의 품질 보증을 위한 시각적 이상 탐지(Anomaly Detection)에 중점을 두고 있습니다. Convolutional Neural Networks (CNNs) 기반의 분류 모델과 블랙 박스(Classifier) 분류기를 위한 모델-비의존적(Machine-agnostic) 설명 알고리즘의 발전에 초점을 맞추고 있습니다. 이를 통해 사용자-interactive interface를 구현하여 모델의 출력을 수정할 수 있도록 돕습니다.

- **Technical Details**: 두 가지 클래스(정상, 비정상)로 분류되는 산업용 이상 탐지 데이터를 위한 InvRISE라는 새로운 설명 방법을 도입하였으며, 기존 CAIPI 알고리즘에 NearCAIPI라는 확장을 추가하였습니다. 이 알고리즘은 사용자 피드백을 적극적으로 통합하여 모델의 성능을 개선하고 설명성을 높이는 것을 목표로 합니다.

- **Performance Highlights**: 이 프레임워크를 통해 사용자 피드백을 통합한 인터랙티브한 과정이 가능해지며, 모델의 신뢰성과 사용 편의성을 증가시킬 수 있는 성과를 보였습니다. 특정 결함(예: 용접 선상의 결함) 탐지에서 사용자에 대한 추가적인 피드백이 성능 향상에 기여할 수 있음을 보여주고 있습니다.



### Optimizing and Evaluating Enterprise Retrieval-Augmented Generation (RAG): A Content Design Perspectiv (https://arxiv.org/abs/2410.12812)
Comments:
          6 pages, 4 figures, to be published in ICAAI 2024 conference proceedings

- **What's New**: 본 논문에서는 리뷰 기반 생성(Retrieval-augmented generation, RAG) 솔루션의 구현 및 유지 관리 경험을 공유하고 있습니다. 기존 RAG 문헌에서 일반적으로 제시된 패턴과의 차별성을 강조하며, 모듈화되어 있고 모델에 의존하지 않는 접근 방식을 중심으로 해결책을 제시합니다.

- **Technical Details**: RAG의 기본 원리는 지식 기반에서 관련 콘텐츠를 검색하고, 이 콘텐츠에 기반한 프롬프트를 작성한 후, LLM에게 출력을 생성하도록 요청하는 것입니다. 그러나 본 팀의 RAG 솔루션은 벡터 데이터베이스에 의존하지 않아 다양한 검색 기법과 LLM을 사용합니다. 지식 기반 콘텐츠 최적화 및 실시간 사용자 질문에 대한 테스트와 평가 방안도 다루고 있습니다.

- **Performance Highlights**: 기존 RAG 평가 지표는 기존 사용자 질문에 대한 응답 평가에 유용하지 않아, 유연한 '인간 선도' 접근 방식이 필요하다는 점을 강조하고 있습니다. 지식 기반 콘텐츠 개선을 통해 RAG 솔루션의 성공 여부에 큰 영향을 미칠 수 있음을 보여줍니다.



### A Hierarchical conv-LSTM and LLM Integrated Model for Holistic Stock Forecasting (https://arxiv.org/abs/2410.12807)
Comments:
          8 pages, 2 figures, 2 tables

- **What's New**: 본 연구는 전통적인 주식 시장 예측 모델의 제한점을 극복하기 위해 새로운 Two-Level Conv-LSTM Neural Network와 Large Language Model (LLM)의 통합 접근 방식을 제안합니다.

- **Technical Details**: 모델은 두 가지 주요 레벨로 구성되어 있습니다. 첫 번째 레벨은 주가 및 기술 지표에서 지역 패턴을 추출하기 위한 Convolutional 층과 시간적 역학을 포착하기 위한 Long Short-Term Memory (LSTM) 층을 포함합니다. 두 번째 레벨은 LLM을 통합하여 금융 뉴스, 소셜 미디어 및 보고서의 감정 및 맥락 정보를 분석합니다.

- **Performance Highlights**: 이 통합 접근 방식은 예측 정확도를 향상시키고 주식 조언에 맥락적으로 풍부한 정보를 제공합니다.



### Design of an Efficient Fan-Shaped Clustered Trust-Based Routing Model with QoS & Security-Aware Side-Chaining for IoV Deployments (https://arxiv.org/abs/2410.12798)
Comments:
this https URL

- **What's New**: 이 논문에서는 인터넷 차량(IoV) 환경에서의 데이터 통신을 효율적으로 관리하기 위한 새로운 팬형 신뢰 기반 라우팅 모델을 제안합니다. 이 모델은 품질 보장(QoS)과 보안 인식을 통한 사이드 체인(side-chaining) 관리를 특징으로 합니다.

- **Technical Details**: 제안된 모델은 지연(delay), 처리량(throughput), 패킷 전달 비율(Packet Delivery Ratio, PDR), 에너지 소비를 고려하여 최적의 라우팅 경로를 결정합니다. 기존의 블록체인 기반 보안 모델을 현저하게 개선하며, 박테리아 채집 최적화(Bacterial Foraging Optimizer, BFO) 알고리즘을 이용해 사이드 체인을 동적으로 조정하여 시스템 성능을 극대화합니다. 팬형 군집화(fan-shaped clustering) 기법을 사용하여 노드를 효율적인 클러스터로 그룹화합니다.

- **Performance Highlights**: 제안된 모델은 대안 모델에 비해 지연을 9.5%, 처리량을 10.5%, 패킷 전달 비율(PDR)을 2.9%, 에너지 소비를 4.5% 줄이는 성과를 보였습니다. 또한, 시빌(Sybil), 가장하기(Masquerading), 플러딩(Flooding) 공격에 대한 저항력을 평가하였으며, 이러한 공격 상황에서도 높은 QoS 수준을 유지하며 신뢰할 수 있는 데이터 전송을 보장합니다. 이 모델은 스마트 시티, 산업 자동화, 의료 시스템, 교통 네트워크 및 환경 모니터링 등 다양한 응용 분야에 활용될 수 있습니다.



### Disaggregating Embedding Recommendation Systems with FlexEMR (https://arxiv.org/abs/2410.12794)
- **What's New**: FlexEMR는 embedding 기반 추천 (EMR) 모델의 비효율성을 해결하기 위한 새로운 분산 시스템으로, 네트워크 데이터 전송의 효율성을 개선하고 총 비용 소유권을 줄이기 위한 디자인을 제안합니다.

- **Technical Details**: FlexEMR는 두 가지 기술 세트를 통해 네트워크 문제를 해결합니다. 첫 번째는 embedding 조회의 시간적 및 공간적 지역성을 활용하여 데이터 이동을 줄이고, 두 번째는 다중 스레드 RDMA 엔진을 설계하여 동시 조회 하위 요청을 최적화하는 것입니다.

- **Performance Highlights**: 초기 프로토타입에서 FlexEMR은 원격 embedding 조회의 성능을 향상시켰으며, queuing latency를 크게 줄이고, 응답 혼잡을 완화하는 데 기여했습니다.



### Environment Scan of Generative AI Infrastructure for Clinical and Translational Scienc (https://arxiv.org/abs/2410.12793)
- **What's New**: 이번 연구는 미국의 Clinical and Translational Science Award (CTSA) 프로그램을 지원하는 36개 기관의 GenAI (Generative AI) 인프라를 종합적으로 분석한 것입니다. GenAI 기술의 빠른 발전으로 의료 기관들은 전례 없는 기회와 도전에 직면해 있습니다. 이 연구는 GenAI 통합 현황을 탐색하며, 이해당사자의 역할, 거버넌스 구조 및 윤리적 고려 사항에 집중하고 있습니다.

- **Technical Details**: 연구는 CTSA 기관의 리더들을 대상으로 설문조사를 실시하여 GenAI 채택에 대한 기관의 준비 상태를 평가했습니다. 주요 발견으로는 대부분의 기관이 GenAI 구현의 실험 단계에 있으며, 중앙 집중식 의사결정을 선호하는 경향이 강하나, 인력 교육과 윤리적 감독의 격차가 존재한다는 점이 드러났습니다.

- **Performance Highlights**: 연구 결과, GenAI 채택에 있어 주요 이해당사자의 참여가 다르다는 점이 확인되었습니다. senior leaders가 94.4%로 가장 많이 참여했으며, IT 직원과 연구자, 의사들이 뒤따랐습니다. 36개 응답 기관 중 77.8%는 GenAI 거버넌스를 감독하는 공식 위원회 또는 작업 그룹을 보유하고 있는 것으로 나타났습니다. 또한, 중앙 집중식 접근 방식이 61.1%의 기관에서 사용되고 있으며, 이는 GenAI의 효과적인 구현을 위한 전략적 리더십 및 결정-making의 중요성을 강조합니다.



### Order-aware Interactive Segmentation (https://arxiv.org/abs/2410.12214)
Comments:
          Interactive demo can be found in project page: this https URL

- **What's New**: 본 논문에서는 사용자 상호작용을 최소화하면서도 정확한 객체 분할을 위한 새로운 방법인 OIS (order-aware interactive segmentation)를 제안합니다. OIS는 객체 간의 상대 깊이를 인코딩하여 사용자 상호작용을 효과적으로 안내하고, 이를 통해 이전의 방법들에 비해 성능을 크게 향상시킵니다.

- **Technical Details**: OIS는 목표 객체의 상대 깊이를 표현하는 order maps를 활용하여 상호작용을 개선합니다. 독창적인 order-aware attention과 object-aware attention 모듈을 도입하여 유사한 깊이를 가진 객체들을 효과적으로 구별할 수 있게 합니다. 또한, 사용자 클릭이 최적화된 이미지 특징에 통합될 수 있도록 조화롭게 설계되어 있습니다.

- **Performance Highlights**: OIS는 HQSeg44K 데이터셋에서 클릭 한 번으로 mIoU가 7.61 증가하였고, DAVIS 데이터셋에서는 1.32의 향상률을 보여줍니다. 뿐만 아니라, SegNext와 비교하여 추론 속도를 2배 향상시킨 것을 입증했습니다.



### ClickAgent: Enhancing UI Location Capabilities of Autonomous Agents (https://arxiv.org/abs/2410.11872)
Comments:
          The code for ClickAgent is available at this http URL

- **What's New**: ClickAgent는 GUI와 상호작용할 수 있는 자율 에이전트를 구축하기 위한 새로운 프레임워크입니다. MLLM의 추론과 행동 계획을 담당하며, 별도의 UI 위치 모델이 화면에서 관련 UI 요소를 식별합니다.

- **Technical Details**: ClickAgent는 MLLM 기반의 추론을 InternVL2.0을 사용하여 수행하고, TinyClick UI 위치 모델을 사용합니다. 세 가지 주요 구성 요소로는 Decision, UI Location, Reflection이 있습니다.

- **Performance Highlights**: ClickAgent는 AITW 벤치마크에서 다른 프롬프트 기반 자율 에이전트보다 우수한 성능을 보였으며, 작업 성공률에서 유의미한 개선을 이루었습니다.



New uploads on arXiv(cs.LG)

### How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs (https://arxiv.org/abs/2410.13857)
- **What's New**: 이 논문에서는 Transformer 기반 대형 언어 모델(LLMs)의 수학적 능력을 이론적으로 분석하고, 특히 산술작업에서의 성능을 강조합니다. 숫자 정밀도가 수학적 작업의 성공적인 수행을 좌우하는 핵심 요소로 밝혀졌습니다.

- **Technical Details**: 저자들은 LLM의 기본 산술 작업인 정수 덧셈, 반복 덧셈, 정수 곱셈을 분석합니다. 저자들은 정밀도에 따라 모델의 크기가 달라지며, 낮은 정밀도(int8, int4)의 Transformer는 문제를 풀기 위해 폭발적으로 큰 모델을 요구한다고 주장합니다. 이와 대조적으로 표준 정밀도(float32)는 훨씬 작고 효율적인 모델로도 이를 처리할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과는 두 가지 정밀도(int4 및 표준 정밀도) 모두에서 정수 덧셈 작업에서 충분한 성능을 보였지만, 반복 덧셈 및 정수 곱셈과 같은 복잡한 작업에서는 낮은 정밀도가 성능 저하를 일으킨다는 것을 보여주었습니다.



### Diffusing States and Matching Scores: A New Framework for Imitation Learning (https://arxiv.org/abs/2410.13855)
- **What's New**: 이번 논문에서는 기존의 대립적 강화 학습(Inverse Reinforcement Learning, IRL) 방식을 넘어, 새로운 방법인 SMILING을 제안합니다. SMILING은 불안정한 판별기 훈련에서 벗어나, 보다 간단한 score-matching 기반의 목적 함수를 사용하여 IRL 문제를 해결합니다.

- **Technical Details**: 이 접근법은 score-matching을 통한 학습과 상태 분포의 혼합을 통해 비용 함수를 정의하는 방식으로 이루어집니다. 이로 인해 기존의 IRL보다 훈련이 쉽고 안정적으로 진행될 수 있습니다. 이론적으로 우리는 세련된 첫 번째 및 두 번째 차수의 인스턴스 의존 경계(instance-dependent bounds)를 증명하며, 수렴할 수 있는 특성을 보입니다.

- **Performance Highlights**: SMILING은 복잡한 동작 제어 작업(예: 사람형 로봇의 걷기, 앉기, 기어가기 등)에서 기존 GAN 기반 IRL 방법보다 우수한 성능을 보였으며, 최신 HumanoidBench 벤치마크에서 여러 작업을 수행하는 IRL 방법 중 첫 번째로 성공했습니다.



### AutoAL: Automated Active Learning with Differentiable Query Strategy Search (https://arxiv.org/abs/2410.13853)
- **What's New**: AutoAL은 기존의 Active Learning (AL) 샘플링 전략 위에 구축된 첫 번째 차별화된 AL 전략 검색 방법입니다. 이 방법은 SearchNet과 FitNet이라는 두 개의 신경망을 포함하며, 이들은 차별화된 이중 최적화 프레임워크 내에서 동시에 최적화됩니다.

- **Technical Details**: AutoAL은 레이블이 달린 데이터셋을 사용하여 SearchNet과 FitNet을 반복적으로 공동 최적화하면서 후보 AL 알고리즘 세트의 성능을 학습합니다. 이를 통해 SearchNet은 주어진 태스크에 최적화된 AL 전략을 선택하여 효율적인 모델 훈련을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, AutoAL은 모든 후보 AL 알고리즘 및 기타 선택적 AL 접근 방식에 비해 일관되게 우수한 정확도를 달성하였으며, 다양한 태스크 및 도메인에 걸쳐 여러 기존 AL 방법을 적응하고 통합할 수 있는 잠재력을 보여주고 있습니다.



### Influence Functions for Scalable Data Attribution in Diffusion Models (https://arxiv.org/abs/2410.13850)
- **What's New**: 확산 모델에 대한 데이터 기여도 및 해석 가능성 문제를 해결하기 위해 영향 함수(influence functions) 프레임워크를 개발하여 새로운 방법을 제시합니다.

- **Technical Details**: 기여도 추정 방법인 영향 함수는 모델 출력이 특정 훈련 데이터를 제거했을 때 어떻게 변할지를 근사합니다. K-FAC(Kronecker-Factored Approximate Curvature) 근사 방법을 사용하여 해시안(Hessian) 계산의 확장성을 보장합니다.

- **Performance Highlights**: 제안된 방법은 Linear Data-modelling Score(LDS)와 같은 평가에서 기존 데이터 기여도 접근 방식보다 성능이 우수함을 보여주었으며, 특정 하이퍼파라미터 조정 없이도 성능을 발휘합니다.



### A Unified View of Delta Parameter Editing in Post-Trained Large-Scale Models (https://arxiv.org/abs/2410.13841)
- **What's New**: 이 논문은 delta parameters(델타 파라미터)의 편집 작업을 Riemann sum(리만 합) 근사를 통해 체계적으로 이해하고 분류하는 새로운 관점을 제시합니다. 기존의 편집 기법들이 어떻게 모델 성능에 영향을 미치는지를 설명하는 통합된 프레임워크를 수립했습니다.

- **Technical Details**: 델타 파라미터는 사전 훈련(pre-trained) 모델과 후 훈련(post-trained) 모델의 파라미터 차이를 나타냅니다. 저자들은 delta parameters의 편집 연산을 Riemann sum approximation(리만 합 근사)을 기반으로 설명하며, 수행된 편집으로 인한 손실 변화를 수학적으로 분석하고 competitive, decreased, improved 성능을 가진 기법으로 분류합니다.

- **Performance Highlights**: 많은 실험을 통해 ViT, LLaMA 3, Qwen 2, Mistral 등 다양한 모델에서 저자들의 이론적인 발견을 뒷받침합니다. DARE와 BitDelta와 같은 기존 기법에서의 성능 향상과 저하를 정량적으로 검증하였으며, 기존의 delta parameter 조정 기술의 한계를 지적하고, 보다 일반화된 접근 방식을 제공하는 확장들을 제안합니다.



### ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization (https://arxiv.org/abs/2410.13837)
Comments:
          preprint, 35 pages, 23 figures

- **What's New**: 본 논문에서는 보상 형성(reward shaping)에서의 새로운 접근법인 Online Reward Selection and Policy Optimization (ORSO)를 제안합니다. 이 방법은 보상 형성 선택을 온라인 모델 선택 문제로 프레이밍하여 자동으로 적합한 보상 형성 함수를 찾아내는 데 초점을 맞추고 있습니다.

- **Technical Details**: ORSO는 합리적인 탐색 전략(principled exploration strategies)을 활용하여 인간의 개입 없이도 유망한 보상 형성 함수(shaping reward functions)를 식별합니다. 이 방법은 탐색(exploration)과 활용(exploitation)을 균형 있게 조절하며, 검증 가능한 후회 보장(regret guarantees)을 제공합니다.

- **Performance Highlights**: Isacc Gym 시뮬레이터를 사용한 다양한 연속 제어(tasks) 실험에서 ORSO의 효과를 입증하였으며, 전통적인 방법에 비해 샘플 효율(sample efficiency)을 크게 향상시키고 계산 시간을 줄이며, 도메인 전문가가 수동으로 엔지니어링한 보상에 의해 생성된 정책과 유사한 고품질 보상 함수를 지속적으로 식별합니다.



### Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs (https://arxiv.org/abs/2410.13835)
- **What's New**: 이번 연구는 transformer 기반의 대형 언어 모델(LLM)에서 관찰되는 극단적인 토큰 현상(extreme-token phenomena)을 분석하고, 그 메커니즘을 밝혀내는 데 중점을 두었다. 특히, 'attention sinks', 'value-state drains', 'residual-state peaks'로 정의되는 세 가지 현상이 서로 어떻게 연관되어 있는지를 설명한다.

- **Technical Details**: 이 연구는 Bigram-Backcopy (BB) 작업에서 1~3층의 단순한 transformer 구조를 훈련시키면서, 'active-dormant mechanism'과 'mutual reinforcement mechanism'을 기반으로 극단적인 토큰 현상이 발생하는 방식을 설명한다. 실험 결과, attention head가 입력 도메인에 따라 활성화와 비활성화를 반복적으로 수행하는 현상을 발견하였다.

- **Performance Highlights**: 저자들은 SoftMax를 ReLU로 대체하고 Adam 대신 SGD를 사용하는 구조적, 최적화적 수정 방법을 제안하여 LLM에서 극단적인 토큰 현상을 완화할 수 있음을 보여주었다. 이러한 수정 사항이 실제 성능에 미치는 영향은 Bigram-Backcopy 작업 및 LLM에서 관찰된 것과 일관성을 가진다.



### The Disparate Benefits of Deep Ensembles (https://arxiv.org/abs/2410.13831)
- **What's New**: 최근 딥 신경망(Deep Neural Networks, DNNs)의 성능을 높일 수 있는 간편한 방법으로 사용되는 딥 앙상블(Deep Ensembles)에 대한 공정성(Algorithmic Fairness) 측면에서의 영향이 잘 이해되지 않았음을 밝히며, 본 연구는 딥 앙상블의 성능 향상과 공정성 간의 상호작용을 분석합니다.

- **Technical Details**: 이 연구는 딥 앙상블을 이용하여 얼굴 분석 및 의료 영상 데이터셋에서 공정성 메트릭을 활용하여 성능 편차를 empirically 조사합니다. 특히, 다양한 protected group 속성에 따라 성능이 상이하게 나타나는 'disparate benefits effect'를 발견했으며, 이 효과의 원인으로 그룹 내 예측의 다양성 차이를 규명했습니다.

- **Performance Highlights**: 본 연구에서 제안된 Hardt 후처리(post-processing) 방법이 효과적으로 공정성을 높이면서도 딥 앙상블의 성능을 유지할 수 있음을 보여줍니다. 분석을 통해 공정성 지표를 향상시킬 수 있는 다양한 접근 방식을 평가하였고, 딥 앙상블의 성능이 다수의 그룹 메트릭에서 불균형적으로 나타나는 것을 실증적으로 확인했습니다.



### A Common Pitfall of Margin-based Language Model Alignment: Gradient Entanglemen (https://arxiv.org/abs/2410.13828)
- **What's New**: 본 논문에서는 Reinforcement Learning from Human Feedback (RLHF)에서 전통적인 margin-based 손실을 사용하는 것의 문제점을 다루고 있습니다. 특히, 이 접근 방법이 선호 및 비선호 응답 각각에 대해 이상적인 언어 모델 behavior를 충분히 명시하지 않는다는 점이 강조됩니다.

- **Technical Details**: 우리는 margin의 증가에 따른 두 가지 의도치 않은 결과를 식별했습니다: (1) 비선호 응답의 확률이 증가할 수 있으며 이는 안전 문제와 관련된 alignment 실패를 초래할 수 있습니다. (2) 선호 응답의 확률이 감소할 수 있으며, 이 경우에도 그 응답은 이상적일 수 있습니다. 이러한 현상의 원인은 gradient entanglement으로 명명하였으며, 이는 선호 및 비선호 응답의 확률 변화가 서로 얽혀 있는 문제를 나타냅니다.

- **Performance Highlights**: 본 논문은 margin 기반 preference optimization 알고리즘의 훈련 동역학을 설명하고, margin 기반 방법의 under-specification 문제를 완화할 수 있는 잠재적인 알고리즘 설계를 제안합니다.



### Unearthing Skill-Level Insights for Understanding Trade-Offs of Foundation Models (https://arxiv.org/abs/2410.13826)
Comments:
          Code at: this http URL

- **What's New**: 이 논문은 모델 평가에서의 복잡성을 해결하기 위해, 모델이 생성한 이론(rationales)을 사용하여 기저가 되는 기술(skills)을 자동으로 복구하는 방법을 제안합니다. 기존의 평가 지표에 숨겨진 다양한 기술을 분석하여, 구체적이고 행동 가능한 모델 능력 이해를 제공합니다.

- **Technical Details**: 평가 인스턴스에 대해 강력한 모델(예: GPT-4o)을 사용하여 단계별 이론을 생성하고 각 단계에서 적용된 기술을 나열합니다. 이 과정을 통해 46,000개 이상의 인스턴스를 분석하고 기술 조각(skill-slices)을 작성하여 여러 벤치마크에서 기술의 정확성을 비교합니다.

- **Performance Highlights**: 우리는 기술 조각 분석을 통해 모델 간의 성능 무역에 대한 새로운 통찰을 발견했습니다. 예를 들어, Gemini 1.5 Pro는 'molar mass 계산'에서 평균적으로 18% 더 정확하지만 '헌법법 적용'에서는 19% 덜 정확하다는 결과를 보여주었습니다. 이러한 분석 방법을 통해 우리는 전체 12개 데이터셋에서 3%의 정확도 향상을 확인했습니다.



### Artificial Kuramoto Oscillatory Neurons (https://arxiv.org/abs/2410.13821)
Comments:
          Code: this https URL

- **What's New**: 본 연구에서는 Artificial Kuramoto Oscillatory Neurons (AKOrN)을 소개합니다. 이는 전통적인 threshold units의 동적 대안으로, 다양한 connectivity 디자인과 결합할 수 있습니다.

- **Technical Details**: AKOrN은 Kuramoto 업데이트를 통해 뉴런의 동기화 동적을 이용하며, 이는 비대칭 연결을 통해 뉴런 간의 상호작용을 탐구합니다. 연구에서는 4개의 합성 데이터셋과 2개의 실제 이미지 데이터셋에서 성능을 평가하였습니다.

- **Performance Highlights**: AKOrN은 비지도 객체 발견, 적대적 강건성, 캘리브레이션된 불확실성 정량화 및 추론을 포함한 다양한 작업에서 향상된 성능을 보여주었습니다.



### Adversarial Testing as a Tool for Interpretability: Length-based Overfitting of Elementary Functions in Transformers (https://arxiv.org/abs/2410.13802)
Comments:
          9 pages, 8 figures, 2 tables; to be published

- **What's New**: 이 논문에서는 Transformer 모델이 훈련 데이터의 전반적인 시퀀스 길이에 대해 과적합(overfitting)하는 경향을 분석하였습니다. 특히, 시퀀스-투-시퀀스(sequence-to-sequence) Transformer의 동작을 이해하기 위해 기초적인 문자열 편집 함수들을 연구하였습니다. 이를 통해 짧은 시퀀스에 대한 일반화 가능성을 확인하면서도, 긴 시퀀스에 대해서는 심각한 문제가 존재함을 보여주었습니다.

- **Technical Details**: 본 연구에서는 문자열 복사(copy), 뒤집기(flip), 역순 복사(reverse)와 같은 기초적인 문자열 편집 작업을 다룹니다. 이러한 작업들은 알고리즘적으로 생성된 이진 시퀀스에서 수행되며, Transformer 모델을 위한 RASP(Restricted Access Sequence Processing Language)를 통해 분석하였습니다. 연구에서는 Transformer 아키텍처의 구조적 특성과 알고리즘적 측면이 충돌할 때, 모델이 주로 구조적 측면을 우선시하는 경향을 관찰했습니다.

- **Performance Highlights**: Transformer가 긴 시퀀스에 대한 최적의 성능을 발휘하지 못하는 경향이 확인되었습니다. 머신 번역(MT) 및 기초 문자열 편집 작업에서, 모델은 훈련 데이터에서 관측된 길이 분포에 맞추어 번역 가설을 압축(condense)하거나 늘리는 경향이 있었습니다. 분석 지표 또한 개선되어 다양한 오류 지표를 정의하고, 훈련 내내 이를 추적하여 모델의 행동을 면밀히 연구할 수 있었습니다.



### Arbitrarily-Conditioned Multi-Functional Diffusion for Multi-Physics Emulation (https://arxiv.org/abs/2410.13794)
- **What's New**: 본 논문에서는 Arbitrarily-Conditioned Multi-Functional Diffusion (ACMFD)라는 다기능 확률적 대리 모델을 제안하여 복잡하고 계산 비용이 많이 드는 전통적 물리 시뮬레이션의 한계를 극복하고자 합니다. ACMFD는 여러 물리적 과제에 대해 단일 프레임워크 안에서 동작할 수 있으며, 이는 예측, 다양한 역문제, 및 조건부 데이터 시뮬레이션을 포함합니다.

- **Technical Details**: ACMFD는 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하며, 노이를 Gaussian processes (GP)로 모델링하여 여러 기능 억제를 기능적 공간에서 수행합니다. 또한, 혁신적인 denoising loss를 도입하여 훈련 과정에서 무작위로 선택된 조건부 부분에 대해 예측된 노이를 제로로 맞춤으로써 다양한 기능 값의 생성을 지원합니다.

- **Performance Highlights**: ACMFD는 Darcy flow, convection-diffusion, torus fluid를 포함한 세 가지 기본 물리 시스템에 대해 평가되었으며, 각 시스템에서 세 가지에서 일곱 개의 함수가 관련되어 있습니다. 20개의 예측 작업에서 ACMFD는 각각의 과제를 위해 특별히 훈련된 최신 신경 연산자들과 비교하였을 때 지속적으로 최상위 성능을 달성했습니다. 또한, ACMFD로 생성된 데이터는 governing equations에 더 잘 부합하며, 다양성 또한 크게 향상되었습니다.



### Analyzing Deep Transformer Models for Time Series Forecasting via Manifold Learning (https://arxiv.org/abs/2410.13792)
Comments:
          Accepted to TMLR 2024

- **What's New**: 본 연구는 transformer 기반의 시간 시계열 예측 기법의 잠재 표현의 기하학적 특성을 조사하는 혁신적인 접근 방식을 제안합니다. 특히, 모델의 레이어 간 기하적 행동이 유사하며 모델 성능과의 상관관계를 발견했습니다. 또한, 비훈련 모델은 훈련 기간 동안 빠르게 수렴하는 패턴을 보였습니다.

- **Technical Details**: 우리의 접근법은 manifold learning 관점에서 문제를 다루며, 잠재 표현이 저차원 manifold 근처에 위치한다고 가정합니다. 연구는 내재적 차원 (intrinsic dimension)과 주 곡률 (principal curvature) 분석을 포함하며, 이는 deep transformer 모델의 여러 레이어에서도 유사한 기하학적 동작을 보여주는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, transformer 기반 모델의 기하학적 특성이 모델 성능과 상관관계를 가지며, 테스트 평균 제곱 오차 (test mean squared error)와 MAPC (mean absolute principal curvature) 간의 연관성을 발견했습니다. 또한, 비훈련 모델은 무작위적인 기하학적 패턴을 보이나, 훈련 후 빠르게 최종 기하학적 프로필로 수렴합니다.



### DPLM-2: A Multimodal Diffusion Protein Language Mod (https://arxiv.org/abs/2410.13782)
- **What's New**: 본 논문에서는 DPLM-2라는 새로운 다중 모달( multimodal ) 단백질 생성 모델을 소개합니다. 이 모델은 단백질의 아미노산 서열과 3D 구조를 동시에 모델링하고 생성할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: DPLM-2는 기존의 Discrete Diffusion Protein Language Model (DPLM)을 확장하여, 구조와 서열 간의 관계를 효과적으로 학습할 수 있는 기반을 제공합니다. 이 과정에서 구조의 3D 좌표를 분산된 토큰으로 변환하기 위해 lookup-free quantization 방식을 사용하는 구조 토크나이저를 개발했습니다. 또한, 모델이 실험적 및 고품질 합성 구조에서 공동 분포를 학습하도록 설계되었습니다.

- **Performance Highlights**: DPLM-2는 아미노산 서열과 해당 3D 구조를 동시에 생성할 수 있으며, 전통적인 두 단계 생성 접근 방식의 필요성을 제거합니다. 다양한 조건부 생성 작업에서 경쟁력 있는 성능을 나타내며, 예측 작업에 대한 구조 인식 표현을 제공합니다. 이 모델은 특히 fold( 접힘 ), inverse folding( 역 접힘 ), motiff scaffolding( 모티프 스캐폴딩 )과 같은 작업에서 우수한 성능을 보여줍니다.



### Change Detection in Multivariate data streams: Online Analysis with Kernel-QuantTr (https://arxiv.org/abs/2410.13778)
Comments:
          AALTD workshop at ECML 2024 (this https URL)

- **What's New**: 이번 연구에서는 Kernel-QuantTree Exponentially Weighted Moving Average (KQT-EWMA)라는 비모수적 변화 탐지(변화 감지) 알고리즘을 제안합니다. KQT-EWMA는 Kernel-QuantTree (KQT) 히스토그램과 EWMA 통계량을 결합하여 멀티버리어트(multivariate) 데이터 스트림을 온라인으로 모니터링 할 수 있도록 합니다. 이 알고리즘은 고정된 평균 연속 길이(ARL_0)를 유지하면서 허위 경고를 제어할 수 있는 특징이 있습니다.

- **Technical Details**: KQT-EWMA는 비모수적(non-parametric) 변화 탐지 알고리즘으로, 전체 데이터 스트림의 분포에 대한 가정 없이 작업을 수행합니다. 테스트 통계량의 분포는 정적 조건에서 데이터 스트림 분포에 의존하지 않으며, 변화 감지 알고리즘은 차원(dimension) 벡터를 처리할 수 있도록 구성되었습니다. KQT-EWMA의 알골리즘은 고차원의 데이터 스트림을 처리하면서 허위 경고를 제어하는 데 강력한 효과를 발휘합니다.

- **Performance Highlights**: 실험 결과, KQT-EWMA는 기존의 최첨단 방법들과 비교하여 더 낮은 탐지 지연(detection delay)으로 ARL_0를 효과적으로 제어하며 성능이 우수한 것으로 나타났습니다. 복잡한 시나리오에서도 KQT의 특성을 온라인 상황에 성공적으로 확장하여 QT-EWMA보다 더 나은 성능을 보였습니다.



### Enhancing Retail Sales Forecasting with Optimized Machine Learning Models (https://arxiv.org/abs/2410.13773)
Comments:
          IEEE 4th ICSES 2024

- **What's New**: 이번 연구는 전통적인 통계 방법인 LR을 넘어, Random Forest (RF), Gradient Boosting (GB), Support Vector Regression (SVR), XGBoost와 같은 최신 머신러닝 (ML) 기법을 활용하여 소매 판매 예측 정확도를 높이는 방법을 소개합니다. 특히, 복잡한 데이터셋과 높은 계절성에 대한 처리 능력을 향상시키기 위해 RF 모델을 최적화했습니다.

- **Technical Details**: 연구는 하이퍼파라미터 조정을 활용한 Randomized Search Cross-Validation을 통해 RF 모델을 최적화 하였고, 성능 메트릭으로 R-squared, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Root Mean Squared Logarithmic Error (RMSLE)를 사용했습니다. 이 외에도, 다양한 방식으로 데이터 전처리를 수행하여 모델 훈련에 적합한 데이터를 확보했습니다.

- **Performance Highlights**: 최적화된 RF 모델은 R-squared 값이 0.945로, 초기 RF 모델과 전통적인 LR의 R-squared 값 0.531보다 훨씬 높은 성과를 보였습니다. 또한 RMSLE를 1.172로 낮춰 다른 최첨단 모델인 Gradient Boosting (R-squared: 0.942), SVR (R-squared: 0.940), XGBoost (R-squared: 0.939)보다 우수한 예측 능력을 입증했습니다.



### Is Prior-Free Black-Box Non-Stationary Reinforcement Learning Feasible? (https://arxiv.org/abs/2410.13772)
- **What's New**: 본 논문에서는 시스템의 비정상성(non-stationarity)에 대한 사전 지식 없이 비정상 강화 학습(Non-Stationary Reinforcement Learning, NS-RL) 문제를 연구합니다. 특히, MASTER라는 최신 블랙박스 알고리즘의 비정상성 탐지 메커니즘이 실제 수평(horizon) 선택에서 발동되지 않음을 입증하여, 성능이 무작위 재시작(random restarting) 알고리즘과 유사하다는 것을 보여줍니다.

- **Technical Details**: 주요 기술 세부사항으로는 MASTER의 비정상성 탐지 메커니즘이 실용적인 수평 선택에서 제대로 작동하지 않는다는 점과, 제시된 후회(bound)한계가 최적의 순서(order optimal)임에도 불구하고 최악의 경우 선형 후회(linear regret)보다 높게 유지된다는 점입니다. 이를 통해 MASTER의 성능을 평가하고자, 조각별 정적 다중 팔 밴디트(piecewise stationary multi-armed bandits, PS-MABs)에서 실험을 수행하였습니다.

- **Performance Highlights**: 연구 결과, QUICK CHANGE DETECTION(QCD) 기법을 활용한 방법들이 MASTER 및 다른 무작위 재시작 접근법에 비해 더 강력하고 일관되게 뛰어난 성능을 보임을 확인하였습니다. 또한, PS-MABs에 대한 무작위 재시작 알고리즘을 제안하여 기준선으로 활용하였습니다.



### Virtual Sensing for Real-Time Degradation Monitoring of Nuclear Systems: Leveraging DeepONet for Enhanced Sensing Coverage for Digital Twin-Enabling Technology (https://arxiv.org/abs/2410.13762)
- **What's New**: 본 논문에서는 AP-1000 Pressurized Water Reactor (PWR)에서 고온 다리(hot leg)의 열유체 매개변수를 예측하기 위해 Deep Operator Networks (DeepONet)를 사용하는 방안을 제안합니다. 이는 디지털 트윈(digital twin) 프레임워크 내에서 실행되며, 지속적인 재학습의 필요성을 완화하여 온라인 및 실시간 예측을 가능하게 합니다.

- **Technical Details**: DeepONet는 다양한 운영 조건에 대해 훈련되며, 이로 인해 많은 수의 데이터 및 복잡한 고차원 데이터를 효율적으로 처리할 수 있습니다. 본 연구는 DeepONet이 평균 제곱 오차(mean squared error) 및 상대 L2 오류(relational L2 error)가 낮은 결과를 보이며, 전통적인 유한 요소(finite element) 시뮬레이션보다 160,000배 빠른 예측을 할 수 있음을 보여줍니다.

- **Performance Highlights**: DeepONet는 실시간으로 재료 열화(indicators of material degradation)를 추적하는 데 매우 효과적인 도구로 입증되었으며, 이러한 속도와 정확성은 원자로 안전성과 수명을 높이는 데 기여합니다.



### GDeR: Safeguarding Efficiency, Balancing, and Robustness via Prototypical Graph Pruning (https://arxiv.org/abs/2410.13761)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 새로운 동적 소프트 프루닝 방법인 GDeR(Graph De-Redundancy)를 제안합니다. 이 방법은 훈련 중 그래프 샘플을 업데이트하여 효율성과 함께 데이터의 균형성과 견고성을 유지할 수 있도록 합니다.

- **Technical Details**: GDeR는 프로토타입 학습(prototype learning)에 영감을 받아 그래프 샘플을 하이퍼스페이스(hyperspherical) 임베딩 공간에 프로젝션(project)하여 불균형 및 노이즈에 강한 대표적이고 균형 잡힌 서브셋을 샘플링합니다. 또한, 정규화된 하이퍼구에서 샘플링 분포를 생성하여 최적의 데이터를 추출합니다.

- **Performance Highlights**: GDeR는 (I) 30%에서 50% 적은 훈련 샘플로 전체 데이터셋의 성능을 달성하거나 초과할 수 있으며, (II) 최대 2.81배의 손실 없는 훈련 속도 향상을 이루며, (III) 불균형 훈련 및 노이즈 훈련 시 최신의 프루닝 방법보다 최대 4.3% 및 7.8% 더 나은 성능을 보입니다.



### Supervised Kernel Thinning (https://arxiv.org/abs/2410.13749)
- **What's New**: 이번 연구에서는 Dwivedi & Mackey(2024)의 kernel thinning 알고리즘을 일반화하여 감독 학습 문제의 속도를 개선합니다. 특히 Nadaraya-Watson 회귀나 kernel smoothing, 그리고 kernel ridge regression(KRR)을 KT와 결합하여 훈련과 추론 시간을 각각 2배 향상시킵니다.

- **Technical Details**: 본 연구에서 제안하는 kernel-thinned Nadaraya-Watson estimator (KT-NW)와 kernel-thinned KRR estimator (KT-KRR)는 각각 훈련 시 𝒪(n log³ n), 𝒪(n^{3/2})의 복잡도를 가지며, 추론 시 𝒪(√n)의 복잡도를 요구합니다. KT-NW는 평균 제곱 오차(MSE)에서 n^(-β/(β+d))의 성능을 보이며, KT-KRR은 유한 차원에서 m log n / n의 최적화 성능을 보장합니다.

- **Performance Highlights**: KT-NW와 KT-KRR은 기존의 thinning 기준선보다 더 높은 정확도를 달성하며, 시뮬레이션 및 실제 데이터에서 유리한 런타임을 유지하면서 성능을 입증하였습니다.



### Theory on Score-Mismatched Diffusion Models and Zero-Shot Conditional Samplers (https://arxiv.org/abs/2410.13746)
- **What's New**: 이번 논문은 일반적인 score-mismatched diffusion 샘플러에 대한 첫 번째 성능 보증을 제시하며, 연구된 차원 의존성을 명시적으로 다룹니다.

- **Technical Details**: 우리는 score mismatch가 대상(target) 배포(distribution)와 샘플링 배포 간 비대칭적(asymptotic) 분포 편향을 초래하며, 이는 대상과 훈련(여기서는 unconditional) 배포 간의 누적된 불일치에 비례함을 보여줍니다.

- **Performance Highlights**: 수치적(numerical) 연구에 의해 뒷받침된 결과들은, 선형 조건부 모델을 위한 새로운 bias-optimal zero-shot 샘플러 설계에 유용한 지침을 제공하며, 여러 흥미로운 대상 배포에 대해 차원 및 조건부에 대한 명시적인 의존성과 함께 수렴 보증을 설정합니다.



### Single-Timescale Multi-Sequence Stochastic Approximation Without Fixed Point Smoothness: Theories and Applications (https://arxiv.org/abs/2410.13743)
- **What's New**: 이번 논문은 다중 연쇄 확률 근사(multiple-sequence Stochastic Approximation, MSSA)에 대한 이론적 이해를 확장하며, 고정점의 스무딩 가정을 필요로 하지 않는 더 정교한 단일 시간 척도 분석을 제시합니다.

- **Technical Details**: 논문에서는 모든 연산자가 강한 단조성(strongly monotone)일 때 MSSA가 $\tilde{\mathcal{O}}(K^{-1})$의 수렴 속도를 가지며, 주요 연산자를 제외한 모든 연산자가 강한 단조성을 가질 경우 $\mathcal{O}(K^{-\frac{1}{2}})$의 수렴 속도를 가진다는 것을 밝힙니다.

- **Performance Highlights**: 이론적 결과를 이층 최적화(bilevel optimization) 및 통신 효율적인 분산 학습(distributed learning)에 적용함으로써 더 완화된 가정과/또는 성능 보장(performance guarantees)을 제공하며, 이는 수치 실험을 통해 검증되었습니다.



### Optimizing Probabilistic Conformal Prediction with Vectorized Non-Conformity Scores (https://arxiv.org/abs/2410.13735)
- **What's New**: 본 논문은 복잡한 분포에서 효율적인 예측을 위한 새로운 Probabilistic Conformal Prediction(PCP) 방법론, 즉 PCP-VCR(Probabilistic Conformal Prediction by Vectorizing the Non-Conformity Score with Ranked samples)를 제안합니다. 이 방법은 순위가 매겨진 샘플을 활용하여 비일치(non-conformity) 점수를 벡터화하고, 예측 세트의 형태를 최적화하여 예측의 효율성을 높입니다.

- **Technical Details**: PCP-VCR 방법은 비일치 점수를 벡터화해 에러 분포에 대한 보다 상세한 정보를 포착합니다. 이 점수는 샘플의 경험적 밀도에 따라 샘플을 순위 매겨 불확실성 세트를 구성하고, 동등한 크기의 불확실성 구역을 사용하는 기존 PCP의 한계를 극복합니다. 또한, 샘플의 순위에 따라 개별적으로 분위수(quantile)를 조정하여 더 유연하고 정보성 높은 예측 세트를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, PCP-VCR은 기존 방법들과 비교하여 더 유효한 커버리지를 유지하면서 더 타이트하고 효율적인 예측 세트를 제공하는 것으로 나타났습니다. 이는 의료 진단, 자율 주행 등 고위험 분야에서 정확한 불확실성 정량화가 요구되는 때에 특히 유용합니다.



### Reducing the Transformer Architecture to a Minimum (https://arxiv.org/abs/2410.13732)
Comments:
          8 pages, to appear in KDIR2024

- **What's New**: 이번 연구에서는 Transformer 아키텍처의 비효율성을 개선하기 위해 Multi-Layer Perceptron (MLP) 없이도 비슷한 성능을 발휘할 수 있는 단순화된 구조를 제안합니다. 연구 결과, 파라미터 수를 최대 90% 줄이면서도 분류 성능을 유지할 수 있음을 입증하였습니다.

- **Technical Details**: 본 연구에서는 Attention Mechanism을 기반으로 한 Transformer 모델의 구조적 개선 방안을 제시합니다. MLP의 생략, Query와 Key 행렬의 통합, 그리고 대칭 유사성 행렬을 적용하여 모델의 비선형성을 효과적으로 활용했습니다. 기존의 비대칭 정의에 비해 모집단 수를 줄일 수 있는 대칭 정의로 변경하였습니다.

- **Performance Highlights**: MNIST 및 CIFAR-10 벤치마크 테스트를 통해, 단순화된 Transformer 아키텍처가 원래 아키텍처와 유사한 분류 성능을 보이면서도 파라미터 수를 대폭 절감할 수 있음을 확인하였습니다.



### Generation through the lens of learning theory (https://arxiv.org/abs/2410.13714)
Comments:
          16 pages

- **What's New**: 본 논문은 통계적 학습 이론을 통해 생성 문제를 연구합니다. Kleinberg와 Mullainathan의 연구를 바탕으로 언어 식별 및 생성의 경계를 구성하고, 'uniform generation'이라는 새로운 패러다임을 정립합니다.

- **Technical Details**: 이 연구에서는 이론적인 관점에서 생성 가능성과 예측 가능성을 비교하여 생성과 예측의 불일치성을 보여줍니다. 새로운 조합 차원인 Closure dimension을 도입하여 가설 클래스의 생성 가능성을 차별화합니다.

- **Performance Highlights**: 논문에서는 주로 정보 이론적 성질에 초점을 맞추면서, 생성 문제를 언어 생성 이상의 넓은 범위로 확장합니다. 또한, 생성과 예측 간의 본질적인 차이를 밝히고, 두 과제가 서로 호환되지 않음을 보여줍니다.



### CrystalX: Ultra-Precision Crystal Structure Resolution and Error Correction Using Deep Learning (https://arxiv.org/abs/2410.13713)
- **What's New**: 이 논문에서는 X-ray 회절 측정 데이터를 활용한 심층 학습 모델인 CrystalX를 제안하여 전체 원자 수준에서의 초정밀 결정 구조 분석을 자동화합니다. 기존의 수작업에 의존하던 결정 구조 분석 과정을 몇 초 만에 수행할 수 있도록 혁신적인 발전을 가져왔습니다.

- **Technical Details**: CrystalX는 50,000건 이상의 X-ray 회절 데이터셋을 통해 훈련되었으며, 비수소 원자와 수소 원자를 각각 99.80%와 99.63%의 정확도로 판별합니다. 이 모델은 전자 밀도 피크에서 원자 간의 기하학적 상호작용 패턴을 해독하기 위해 TorchMD-NET이라는 고급 Equivariant Transformer 모델을 활용합니다. 수소 원자의 위치 결정을 위해 분자 내외 상호작용 모델링도 integriert되어 있습니다.

- **Performance Highlights**: CrystalX는 저널 기사에서 1,559건 중 9건의 숨겨진 오류를 감지하고 수정했으며, 최근 발견된 화합물의 구조 분석에서도 성공적으로 적용되었습니다. 이는 CrystalX가 인간 전문가들과 동등하며, 경제성과 속도 면에서도 획기적인 성과를 거두었음을 보여줍니다.



### On-device Federated Learning in Smartphones for Detecting Depression from Reddit Posts (https://arxiv.org/abs/2410.13709)
Comments:
          11 pages, 7 figures, Submitted to IEEE

- **What's New**: 본 연구에서는사용자 데이터의 개인 정보 보호를 확보하면서 스마트폰에서 분산 학습을 통해 우울증을 감지하는 딥러닝 모델을 훈련하는 방법으로 Federated Learning(FL)을 채택하였다.

- **Technical Details**: 연구에서는 Reddit 게시글을 바탕으로 GRU, RNN 및 LSTM의 3가지 신경망 아키텍처를 훈련하고, 이질적인 FL 환경에서 성능을 평가하였다. 클라이언트 장치의 공통 토크나이저(tokenizer)를 활용하여 계산 부하를 줄이고 스마트폰의 자원 소비 및 통신 비용을 분석했다.

- **Performance Highlights**: 실험 결과 분산 모델이 중앙 집중 모델과 비교하여 유사한 성능을 달성하는 것을 보여 주었으며, 클라이언트 장치에서 비공식적인 우울증 예측을 위한 안전하고 효율적인 모델 훈련 과정을 제공하는 FL의 잠재력을 강조하였다.



### Efficient Function Placement in Virtual Networks: An Online Learning Approach (https://arxiv.org/abs/2410.13696)
Comments:
          19 pages

- **What's New**: 본 논문에서는 Virtual Function Placement (VFP) 문제를 위한 새로운 모델과 다중 팔찌 머신(Multi-Armed Bandits) 기반의 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 O(NM√(TlnT)) 속도로 최적의 배치 정책을 학습하며, 높은 확률로 실행 가능성 제약을 준수합니다. 실험 결과, 이 알고리즘은 좋은 실용성과 적절한 계산 복잡도를 보여줍니다.

- **Performance Highlights**: 제안된 가속화 기법을 통해 대규모 네트워크에서도 학습이 가능하며, 전체 실험은 재현 가능하고 코드를 공개하여 사용할 수 있습니다.



### Automated Model Discovery for Tensional Homeostasis: Constitutive Machine Learning in Growth and Remodeling (https://arxiv.org/abs/2410.13645)
Comments:
          46 pages, 12 figures, 5 tables

- **What's New**: 본 연구에서는 비탄력성 물질의 거시적 행동을 보다 정확하게 포착하기 위해 물리 기반 머신러닝 알고리즘을 활용하는 새로운 접근 방식을 제안합니다. 특히, kinematic growth(운동학적 성장) 및 homeostatic surfaces(항상성 표면) 개념을 통합한 iCANNs(비탄력성 구성 인공 신경망)를 확장합니다.

- **Technical Details**: 이 연구에서는 Helmholtz free energy(헬름홀츠 자유 에너지) 및 pseudo potential(유사 잠재력)과 같은 스칼라 모델 방정식들을 발견하기 위해.. 
이러한 모델은 비탄력성 변형의 진화를 정의하고 항상성 상태를 설명합니다. 전체적인 동역학(kinematics) 분석에서 비탄력성 물질의 거시적 행동을 모델링하기 위해, 다양한 매개변수를 실험적으로 수집된 조직 동등 데이터(material point level)로부터 학습하는 네트워크의 능력을 평가합니다.

- **Performance Highlights**: 제안된 네트워크는 훈련 범위를 넘어 예측 정확성이 뛰어나며, 구조적 수준(structural level)에서 적용할 때의 현재 한계에 대해서도 논의하였습니다. 추가로, 연구는 관련 소스 코드, 데이터, 예제 및 자료 서브루틴 구현을 공개합니다.



### Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design (https://arxiv.org/abs/2410.13643)
- **What's New**: 이번 연구에서 제안하는 DRAKES 알고리즘은 기존의 discrete diffusion models를 활용하여 특정 작업 목표에 최적화된 시퀀스를 생성하는 데 중점을 두고 있습니다. 특히, 자연성과 고급 보상 최적화를 동시에 달성할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: DRAKES는 Gumbel-Softmax 트릭을 사용해 기존의 비미분 가능했던 경로를 미분 가능하게 만들어 전체 경로를 통한 보상의 직접 역전파를 가능하게 합니다. 이 알고리즘은 reinforcement learning (RL) 방식으로 보상 최대화 문제를 접근하며 KL divergence를 최소화하여 자연성을 유지합니다.

- **Performance Highlights**: DRAKES는 DNA 및 단백질 시퀀스를 생성하는 데 성공적으로 적용되어 각각 enhancer 활동과 단백질 안정성을 최적화한 결과, 중요한 유전자 치료 및 단백질 기반 치료에서의 활용 가능성을 보여주었습니다.



### Scaling Wearable Foundation Models (https://arxiv.org/abs/2410.13638)
- **What's New**: 이 연구에서는 165,000명 이상의 사용자로부터 수집한 4천만 시간의 다중 모달(sensor modalities) 센서 데이터를 기반으로 한 대규모 착용형 센서 모델(LSM)을 소개하고, 해당 모델의 스케일링 속성을 조사합니다. 본 연구의 주된 목표는 착용형 센서 데이터가 있는 경우 스케일링 법칙이 적용될 수 있는지를 확인하는 것입니다.

- **Technical Details**: LSM 모델은 심박수, 심박수 변동성, 전기 피부 활동(EDA), 가속도계, 피부 온도, 고도계 등 다양한 센서에서 수집된 데이터를 사용합니다. 이 연구는 데이터, 모델 크기, 컴퓨팅 리소스가 늘어날 때 LSM의 성능이 어떻게 향상되는지를 실험 엘 리 분석합니다. 자가 지도 학습(SSL) 기법을 통해 소량의 레이블 데이터 뿐만 아니라 대량의 비레이블 데이터에서 유용한 표현을 학습합니다.

- **Performance Highlights**: LSM은 데이터의 시간적 및 센서 모달리티를 초월하여 임퓨테이션(imputation), 보간(interpolation), 외삽(extrapolation) 작업을 수행할 수 있는 능력을 보여줍니다. 또한 연구는 사용자 주석 이벤트를 활용하여 운동 및 활동 인식과 같은 다운스트림 분류 작업에서의 일반화 가능성을 검증하였습니다.



### Normalizing self-supervised learning for provably reliable Change Point Detection (https://arxiv.org/abs/2410.13637)
- **What's New**: 본 논문에서는 기존의 Change Point Detection (CPD) 기법의 한계를 극복하기 위해, 전통적인 CPD 방법의 신뢰성과 표현 학습 (representation learning) 기술의 표현력을 결합하는 방안을 제안하고 있습니다. 특히, Spectral Normalization (SN)을 통해 딥러닝 모델의 데이터 표현을 최적화하고 있습니다.

- **Technical Details**: CPD 문제를 해결하기 위해, SN 기법을 사용하여 신경망의 학습에서 데이터의 변화를 표현 공간에서 유지하도록 하였습니다. 본 논문은 자기 지도 학습 (Self-Supervised Learning, SSL) 방법을 결합하여, 변화 점 탐지 (change point detection)를 위한 보다 효과적인 임베딩 (embedding) 공간을 제공합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 표준 CPD 데이터셋을 통해 평가된 결과, 현재의 최첨단 기법들보다 현저히 높은 성능을 기록하였습니다. 이는 SN을 통한 임베딩의 정보성이 CPD 활용에 매우 유익함을 보여줍니다.



### All models are wrong, some are useful: Model Selection with Limited Labels (https://arxiv.org/abs/2410.13609)
- **What's New**: MODEL SELECTOR는 레이블 효율적인 사전 학습 분류기 선택을 위한 프레임워크로, 비표시된 데이터 풀에서 정보가 가장 풍부한 샘플(subset)을 선택하여 최적의 사전 학습 모델을 식별하는 데 도움을 줍니다.

- **Technical Details**: MODEL SELECTOR는 모델 비의존적인 방법론으로, 사용자가 제공하는 라벨 예산에 기반하여 비표시된 샘플 중 최대의 정보를 줄 수 있는 예를 선택합니다. 이 방법은 단일 매개변수 모델을 사용하여 레이블과 최상의 모델 간의 상호정보량(mutual information)을 계산합니다.

- **Performance Highlights**: 18개 모델 컬렉션에서 15,000개 이상의 사전 학습 모델을 비교한 결과, MODEL SELECTOR는 최적의 모델을 찾기 위해 레이블 비용을 최대 94.15%까지 줄일 수 있습니다. 또한, 근사 최적 모델 선택 시 레이블 비용을 최대 72.41%까지 감소시켰습니다.



### Transformer-Based Approaches for Sensor-Based Human Activity Recognition: Opportunities and Challenges (https://arxiv.org/abs/2410.13605)
- **What's New**: 이 연구는 센서 기반 인간 활동 인식(HAR)에서 트랜스포머(transformer) 모델의 적용 가능성에 대한 의문을 제기합니다. 500회 이상의 실험을 통해 트랜스포머 기반 모델이 비트랜스포머(non-transformer) 모델에 비해 성능이 일관되게 낮으며, 특히 자원 제약이 있는 장치에서의 양자화(quantization) 과정에서 성능 저하가 심각하다는 사실을 발견하였습니다.

- **Technical Details**: 트랜스포머는 NLP(자연어 처리)와 CV(컴퓨터 비전) 분야에서 뚜렷한 성과를 보였으나, HAR에서는 모델의 연산 비용(inference cost)이 증가하고, 샤프(minima) 수렴의 문제로 인해 일반화 능력(generalization capability)에 손해를 보입니다. 연구에서는 SAM(Sharpness-Aware Minimization) 기법을 통해 성능 개선을 시도하였으나, 비트랜스포머 모델에 비해 여전히 부족한 성능을 보였습니다.

- **Performance Highlights**: 트랜스포머 기반 모델은 비트랜스포머 모델에 비해 약 2.5배에서 26배 더 많은 연산 자원을 요구하며, 양자화 단계에서 비트랜스포머 모델은 약 5% 이내에서 성능 저하를 보이는 반면, 트랜스포머 모델은 최대 85%의 성능 저하를 경험할 수 있습니다. 또한, 트랜스포머는 적대적 공격(adversarial attack)에 대해 더 취약한 경향을 보입니다.



### Text-Guided Multi-Property Molecular Optimization with a Diffusion Language Mod (https://arxiv.org/abs/2410.13597)
- **What's New**: 이 논문은 약물 발견에서 필수적인 단계인 분자 최적화(molecular optimization, MO)를 위한 새로운 접근법인 Transformer 기반 확산 언어 모델(TransDLM)을 제안합니다. 기존 MO 방법은 주로 외부 속성 예측자(property predictors)에 의존했으나, 이는 예측 과정에서 오류와 노이즈를 초래합니다. TransDLM은 표준화된 화학 명명을 활용하여 오류 전파를 방지하며 동시에 여러 속성을 최적화합니다.

- **Technical Details**: TransDLM은 분자의 SMILES 문자열의 단어 벡터를 생성하기 위해 확산 모델을 활용하며, 언어 설명에 의해 안내됩니다. 이는 원하는 분자 속성을 언어 모델을 통해 암시적으로 통합하여 확산 과정에서의 오류 전파를 방지합니다. 또한, TransDLM은 웹 기반 플랫폼에서 배포가 가능하여 분산 환경에서 대규모 MO를 지원합니다.

- **Performance Highlights**: TransDLM은 벤치마크 데이터세트에서 기존의 최첨단 방법들을 초월하는 성능을 보이며, 3가지 ADMET 속성(LogD, Solubility, Clearance)을 최적화하는 데 있어 구조적 유사성을 유지하면서 개선된 화학적 특성을 제공합니다.



### Towards Better Performance in Incomplete LDL: Addressing Data Imbalanc (https://arxiv.org/abs/2410.13579)
- **What's New**: 이번 논문에서는 기존의 Incomplete Label Distribution Learning (InLDL) 방법들이 간과하고 있는, 레이블 분포의 불균형 문제를 다루기 위해 Incomplete and Imbalance Label Distribution Learning (I²LDL)이라는 새로운 프레임워크를 제안합니다. 이 방법은 불완전한 레이블과 불균형한 레이블 분포를 동시에 처리합니다.

- **Technical Details**: I²LDL은 레이블 분포 행렬을 저계수 성분과 희소 성분으로 분해하여, 잦은 레이블(헤드 레이블)의 구조와 드문 레이블(테일 레이블)의 특징을 효과적으로 코드화합니다. 또한, Alternating Direction Method of Multipliers (ADMM)를 사용하여 모델을 최적화하며, Rademacher complexity를 통해 일반화 오차 경계를 도출하여 이론적 보장을 제공합니다.

- **Performance Highlights**: 15개의 실제 레이블 분포 데이터셋에 대한 광범위한 실험 결과, 제안한 I²LDL 프레임워크가 기존의 InLDL 방법들과 비교하여 우수한 성능과 강인성을 보임을 입증했습니다.



### Sample Compression Hypernetworks: From Generalization Bounds to Meta-Learning (https://arxiv.org/abs/2410.13577)
Comments:
          Accepted at the NeurIPS 2024 workshop on Compression in Machine Learning

- **What's New**: 본 연구에서는 일반적으로 고정되어 있는 reconstruction function을 학습할 수 있게 제안합니다. 이를 통해 샘플 압축 이론(sample compression theory)을 통해 증명된 새로운 일반화 경계(generalization bound)를 도출하였습니다.

- **Technical Details**: 새로운 하이퍼네트워크 아키텍처(hypernetwork architecture)를 설계하였고, 메타 학습(meta-learning) 프레임워크를 활용하여 예측기(predictor)를 출력합니다. 이 아키텍처는 압축 집합(compression set)과 메시지(message)를 생성하고, 이 정보를 사용하여 예측기를 재구성하는 새로운 형태의 인코더-디코더(encoder-decoder)로 볼 수 있습니다.

- **Performance Highlights**: 기대되는 초기 실험 결과를 보고하였으며, 이는 tight task-specific sample compression generalization bounds와 관련이 있으며, 실제 값(real-valued) 메시지에 대한 샘플 압축 정리(sample compression theorem)에 기반하고 있습니다.



### Representing Model Weights with Language using Tree Experts (https://arxiv.org/abs/2410.13569)
- **What's New**: 본 논문은 다양한 신경망 모델의 가중치를 입력으로 사용하는 메타네트워크를 학습하는 새로운 접근 방식을 제안합니다. 대중 모델의 대부분이 소수의 Model Trees에 속하며, 이는 학습을 용이하게 합니다.

- **Technical Details**: 모델 가중치의 변화량을 줄이기 위해 Probing Experts (ProbeX)라는 경량 탐색 방법을 도입합니다. ProbeX는 단일 레이어의 가중치만을 학습하며, 고차원 모델 가중치를 효율적으로 매핑하는 것을 목표로 합니다.

- **Performance Highlights**: ProbeX는 제로샷 모델 분류 및 검색을 포함하여 모델 가중치를 공통의 가중치-언어 임베딩 공간으로 효과적으로 매핑하며, 상대적으로 적은 훈련 시간으로 다양한 과제에서 뛰어난 일반화를 보여줍니다.



### Ornstein-Uhlenbeck Adaptation as a Mechanism for Learning in Brains and Machines (https://arxiv.org/abs/2410.13563)
- **What's New**: 본 연구는 전통적인 gradient descent(그래디언트 강하) 방법에 대한 대안으로, 시스템 파라미터의 노이즈와 전역 강화 신호를 활용한 새로운 학습 기법(learning mechanism)을 소개합니다.

- **Technical Details**: 이 방법은 Ornstein-Uhlenbeck 프로세스를 사용하여 학습 과정에서 탐색(exploration)과 활용(exploitation) 사이의 균형을 맞춥니다. 이 접근법은 오류 예측의 편차에 기반한 강화 신호를 사용하며, 연속적인 시간에서 동작합니다. Ornstein-Uhlenbeck 적응(OUA)은 동적이고 시간에 따라 변화하는 환경을 학습하기 위한 일반적인 메커니즘으로 제안됩니다.

- **Performance Highlights**: OUA는 다양한 실험, 즉 피드포워드 및 순환 시스템에서의 지도 학습(supervised learning)과 강화 학습(reinforcement learning)을 통해 검증되었습니다. 또한, 자동으로 하이퍼파라미터를 조정하는 메타 학습(meta-learning) 능력도 보여주었습니다. OUA는 전통적인 gradient 기반 방법에 대한 유효한 대안이 될 수 있는 가능성을 지니며, 생물학적 학습을 위한 노이즈 기반 학습 메커니즘에 대한 통찰력을 제공합니다.



### Adaptive and oblivious statistical adversaries are equivalen (https://arxiv.org/abs/2410.13548)
- **What's New**: 이 논문은 적대적인 샘플 손상에 대한 통계적 작업을 수행하는 능력에 대한 근본적인 질문을 다루고 있습니다. 특히 샘플 적응형 적대자(sample-adaptive adversaries)와 샘플 비적응형 적대자(sample-oblivious adversaries) 간의 동등성을 증명하였습니다. 새로운 구조적 결과를 통해 이 두 가지 유형의 적대자 간의 관계를 명확히 하였습니다.

- **Technical Details**: 논문에서 제시하는 주요 결과는 샘플 크기에 따라 다항식 요인 비율로 샘플 적응형 적대자와 샘플 비적응형 적대자가 동등하다는 것입니다. 특정 알고리즘 A에 대해, 비적응형 적대자가 손상시키더라도 통계적 작업을 수행할 수 있는 알고리즘을 제공하고, 같은 작업을 수행할 수 있는 A' 알고리즘을 제공합니다. A'는 A보다 다항식으로 더 큰 샘플을 요청하며, 균일하게 무작위로 선택된 하위 샘플에서 A를 실행하는 방식으로 구성됩니다.

- **Performance Highlights**: 이 결과는 다양한 정의의 강건성(robustness) 모델 간의 관계를 간소화하며, 특히 하위 샘플링(subsampling) 기법이 많은 모델에서 강건성을 증대시킨다는 것을 보여줍니다. 또한, 데이터셋을 적대자로부터 숨기는 것이 유용한지에 대한 질문도 다루어, 개인 데이터와 공개 데이터 간의 강건성 차이가 크지 않음을 시사합니다.



### PORTAL: Scalable Tabular Foundation Models via Content-Specific Tokenization (https://arxiv.org/abs/2410.13516)
Comments:
          Accepted at Table Representation Learning Workshop at NeurIPS 2024

- **What's New**: 본 논문은 다양한 데이터 모달리티를 지원하며 데이터 클리닝이나 전처리 없이도 활용될 수 있는 PORTAL(Pretraining One-Row-at-a-Time for All tabLes)이라는 프레임워크를 소개합니다. 이 새로운 접근법은 기존의 기법들과 비교하여 더 나은 확장성과 성능을 제공합니다.

- **Technical Details**: PORTAL은 텍스트, 숫자 및 날짜와 같은 다양한 유형의 데이터를 처리할 수 있는 transformer encoder 기반의 모델입니다. 이 모델은 각 row의 데이터를 개별적으로 처리하며, masked cell modeling을 통해 효과적인 pre-training을 수행합니다. 포괄적으로, 이 방법론은 인코딩, 백본, 디코딩의 세 가지 단계로 구성되어 있습니다.

- **Performance Highlights**: PORTAL은 텍스트 중심의 데이터셋에서 XGBoost와 CM2와 비교하여 우수한 성능을 보여줍니다. 또한 CatBoost, AutoGluon 및 CARTE와 같은 최신 기법과 유사하거나 더 뛰어난 성능을 발휘하며, 복잡한 분류 및 회귀 작업에서도 효과적으로 fine-tuning이 가능합니다.



### MathGAP: Out-of-Distribution Evaluation on Problems with Arbitrarily Complex Proofs (https://arxiv.org/abs/2410.13502)
Comments:
          Preprint

- **What's New**: MathGAP이라는 새로운 평가 프레임워크를 제시하여, 보다 복잡한 산술 증명이 포함된 문제에서 대형 언어 모델(LLMs)의 일반화 능력을 분석합니다. 이를 통해 기존의 평가 방법의 한계를 극복하고 보다 체계적인 연구를 가능하게 합니다.

- **Technical Details**: MathGAP는 고정된 증명 기준을 따르는 문제를 생성하고 체계적인 체인-오브-생각(chain-of-thought) 주석을 제공합니다. 이 프레임워크는 증명 나무(proof trees)를 기반으로 각 문제의 복잡성을 특성화하고, 간단한 예제를 사용하여 더 복잡한 문제를 해결할 수 있는 LLM의 능력을 평가합니다.

- **Performance Highlights**: 모델 성능은 증명 깊이와 너비가 증가함에 따라 일관되게 감소하며, 특히 비선형(nonlinear) 문제에서 눈에 띄는 감소가 관찰됩니다. 흥미롭게도, 테스트 세트와 동일한 분포의 예제를 제공하는 것이 항상 성능에 이롭지 않으며, 다양한 복잡성을 가진 예제를 제시하는 것이 더 유효한 경우가 많습니다.



### Integrating Large Language Models and Reinforcement Learning for Non-Linear Reasoning (https://arxiv.org/abs/2410.13501)
- **What's New**: 이번 연구에서는 Long-term planning에서 LLMs의 한계를 극복하기 위해 Reinforcement Learning (RL) Agent를 활용한 새로운 아키텍처를 제안합니다. 이 아키텍처는 특정 도메인에 대한 정보를 기반으로 LLM의 탐색을 유도하며, 즉각적인 다음 단계 생성에 중점을 둡니다.

- **Technical Details**: 제안된 아키텍처는 RL Agent가 후보 솔루션의 품질을 평가하는 데 필요한 도메인별 정보를 활용합니다. LLM은 긴 계획 없이 즉각적인 결과물인 다음 단계를 생성할 수 있도록 구성되며, 비선형 reasoning을 가능하게 합니다. 실험은 program equivalence 과제에서 이루어졌습니다.

- **Performance Highlights**: RL Agent 아키텍처는 기존의 Chain of Thought (CoT) 및 Tree of Thoughts (ToT) 방법보다 중간 reasoning 및 다운스트림 과제에서 더 나은 성과를 보였습니다. CoT는 정확성을 유지하는 데 어려움을 겪었고, ToT는 이전 방법보다 나아졌지만 여전히 한계가 있었습니다.



### Deep Reinforcement Learning for Online Optimal Execution Strategies (https://arxiv.org/abs/2410.13493)
- **What's New**: 본 논문은 동적 금융 시장에서 비마르코프적 (non-Markovian) 최적 실행 전략을 학습하는 도전 과제를 다루고 있습니다. 이 연구는 일반적인 감쇠 커널을 모델링한 일시적인 가격 영향을 중점적으로 다루는 새로운 Actor-Critic 알고리즘을 소개합니다.

- **Technical Details**: 연구진은 Deep Deterministic Policy Gradient (DDPG) 알고리즘을 활용하여 지속적인 행동 공간을 통한 최적 실행 문제를 해결하는 새로운 접근 방식을 제시합니다. 이 알고리즘은 비모델 기반이므로 가격 영향 모델에 대한 엄격한 가정을 필요로 하지 않으며, 비마르코프적 문제에 적응할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 알고리즘은 최적 실행 전략을 성공적으로 근사화하며, 진화하는 시장 조건에 적응할 수 있는 능력을 보여줍니다. 현대의 강화학습 알고리즘이 최적 실행 작업에서 인간 개입의 필요성을 줄이는 솔루션을 제공할 수 있음을 시사합니다.



### Interpreting Temporal Graph Neural Networks with Koopman Theory (https://arxiv.org/abs/2410.13469)
- **What's New**: 이번 연구에서는 Spatiotemporal Graph Neural Networks (STGNNs)의 복잡한 동역학을 이해하고 설명하는 데 있어 Koopman 이론에 영감을 받은 새로운 접근 방식을 제안합니다. 두 가지 방법: Dynamic Mode Decomposition (DMD)와 Sparse Identification of Nonlinear Dynamics (SINDy)를 사용하여 STGNN의 결정 프로세스를 해석합니다.

- **Technical Details**: 연구에서 제안하는 방법은 STGNN의 입력에서 관련된 공간적 및 시간적 패턴을 식별합니다. DMD는 복잡한 동역학 시스템의 차원 감소 방법으로 사용되며, SINDy는 비선형 동역학을 설명하기 위한 일반 도구로 처음 적용됩니다. 다양한 소셜 상호작용을 설명하는 TG 데이터셋을 사용하여 검증되었습니다.

- **Performance Highlights**: 제출된 방법은 감염 시간 및 감염 노드와 같은 해석 가능한 특징을 정확히 식별하며, F1 score를 사용하여 설명의 품질을 측정했습니다. Brier score를 활용하여 예측과 실제 간의 일치도를 평가하고, 감염된 노드에 대한 설명의 지연과 거짓 양성의 발생을 확인했습니다.



### Truncating Trajectories in Monte Carlo Policy Evaluation: an Adaptive Approach (https://arxiv.org/abs/2410.13463)
- **What's New**: 이 논문에서는 Monte Carlo (MC) 정책 평가에서의 데이터 수집 전략을 개선하기 위한 방법을 제시합니다. 저자들은 고정된 길이의 궤적이 아닌 다양한 길이의 궤적을 활용한 평균 제곱 오차의 대체 지표를 제안합니다. 이는 기존의 고정 길이 궤적 스케줄의 비효율성을 보여주며, 에이전트가 보다 적절한 타임스탬프에서 샘플링을 할 수 있도록 적응형 데이터 수집 전략을 권장합니다.

- **Technical Details**: Robust and Iterative Data collection strategy Optimization (RIDO)라는 새로운 알고리즘을 제안합니다. RIDO는 주어진 상호작용 예산을 미니 배치로 나누고, 매 라운드마다 경험한 궤적의 오류를 최소화하는 스케줄을 결정합니다. 이는 환경과의 상호작용을 통해 수집된 정보를 바탕으로 동적으로 적응하는 방식입니다. 또한, 이 논문은 RIDO의 이론적 특성을 논의하고 다양한 영역에서의 성능을 평가합니다.

- **Performance Highlights**: RIDO는 실험을 통해 기존 비적응형 스케줄 대비 최고의 성능을 발휘하는 것으로 나타났습니다. RIDO 알고리즘은 예산, 할인율의 다양한 값에 대해 일관되게 우수한 결과를 보여주었으며, 이는 적응형 전략의 유효성을 강조합니다.



### Progressive Mixed-Precision Decoding for Efficient LLM Inferenc (https://arxiv.org/abs/2410.13461)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 추론 단계에서 질량량(data precision) 할당을 개선하기 위해 단계 인식(phase-aware) 방법을 제안합니다. 이 방법은 prefill 단계에서 높은 정밀도를 유지하고, decoding 단계에서는 낮은 정밀도로 갈수록 하여 메모리 성능과 품질을 모두 향상시킵니다.

- **Technical Details**: 제안된 방법은 Progressive Mixed-Precision Decoding(PMPD)이라는 전략을 통해, 자동 회귀 생성 과정에서의 후반부 토큰들은 정밀도 감소에 더 탄력적이므로 초기 정밀도를 높게 유지하고, 후반부에서 점진적으로 정밀도를 감소시키는 방식으로 동작합니다. 또한, 두 가지 정밀도 전환 스케줄러를 통해 정밀도 감소 시점을 전략적으로 결정합니다.

- **Performance Highlights**: 다양한 언어 작업에서 우리의 방법을 적용했을 때, NPU 플랫폼에서는 3.8-8.0배의 디코딩 처리량 향상을 달성하고, GPU에서는 fp16 모델에 비해 1.4-12.2배의 속도 증가를 기록했습니다. 이처럼, uniform quantization 접근법보다 1.54배 더 높은 성능을 보여 주며 출력 품질을 유지합니다.



### Fast Estimation of Partial Dependence Functions using Trees (https://arxiv.org/abs/2410.13448)
- **What's New**: 이번 논문에서는 기존의 Partial Dependence (PD) 함수에 기반한 해석 방법의 한계를 극복하기 위해 새로운 트리 기반 추정기인 	exttt{FastPD}를 제안합니다. 	exttt{FastPD}는 다양한 PD 함수를 효율적으로 추정할 수 있습니다.

- **Technical Details**: 기존의 SHAP(Shapley additive explanations) 값은 주요 효과와 상호작용 효과를 단일 지역 효과로 통합함으로써 오해를 초래할 수 있습니다. 	exttt{FastPD}는 경량의 트리 구조를 활용하여 인구 집단의 특정 양을 일관되게 추정하며, 	exttt{TreeSHAP}와 달리 특성 간 상관관계가 있는 경우에도 안정성을 유지합니다. 또한 	exttt{FastPD}는 기존 방법들에 비해 관측치 수에 대한 복잡도를 제곱에서 선형으로 개선합니다.

- **Performance Highlights**: 	exttt{FastPD}는 다양한 특성의 하위 집합에 대해 PD 함수를 추정할 수 있기 때문에 SHAP, PD 플롯 및 고차원 상호작용 효과와 같은 PD 기반 해석을 추출하는 데 활용될 수 있습니다.



### Similarity-Dissimilarity Loss with Supervised Contrastive Learning for Multi-label Classification (https://arxiv.org/abs/2410.13439)
- **What's New**: 본 연구는 멀티 라벨 분류에서의 슈퍼바이저드 대조 학습(Supervised Contrastive Learning)에서 긍정 샘플을 결정하는 데 있어 새로운 접근 방식을 제안합니다. 특히, 다섯 가지 고유한 관계를 도입하고, 유사성 및 비유사성 손실(Similarity-Dissimilarity Loss)을 통해 대조 손실 함수(weights)를 동적으로 조정합니다.

- **Technical Details**: 다섯 가지 관계(R2, R3, R4, R5)를 정의하여 멀티 라벨 샘플과 앵커(anchor) 사이의 유사성과 비유사성을 계산하여 손실을 재가중화하는 새로운 Similarity-Dissimilarity Loss를 제안합니다. 이를 통해 ALL, ANY 및 MulSupCon 등의 기존 방법의 한계를 극복합니다.

- **Performance Highlights**: MIMIC 데이터셋에서 멀티 라벨 텍스트 분류 실험을 수행한 결과, 제안된 손실 함수가 슈퍼바이저드 대조 학습 패러다임 하에서 모든 인코더(encoders)에 대해 성능을 효과적으로 향상시키는 것으로 나타났습니다. 실험 결과는 제안된 방법의 효과성과 견고성을 뒷받침합니다.



### Solving Prior Distribution Mismatch in Diffusion Models via Optimal Transpor (https://arxiv.org/abs/2410.13431)
- **What's New**: 이 논문에서는 확산 모델(Diffusion Models, DMs)과 최적 운송 이론(Optimal Transport, OT) 사이의 깊은 관계를 탐구하며, DMs에서 발생하는 prior error를 제거하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문은 DMs이 시간 의존적인 OT 계산 방법으로 구성되어 있음을 보여주고, 전통적인 확산 모델에서 발생하는 prior error가 잠재적인 공백을 초래한다는 것을 이론적으로 분석합니다. 또한, 확산 종료 시간이 증가함에 따라 확률 흐름이 Monge-Ampère 방정식의 해결책의 기울기로 기하급수적으로 수렴한다는 것을 증명합니다.

- **Performance Highlights**: 다양한 이미지 데이터세트에서의 실험 결과는 제안한 접근 방식의 효과성을 입증하며, 특히 조건부 및 비조건부 생성 상황에 대한 샘플링 가속화에 자연스럽게 확장될 수 있음을 보여줍니다.



### Partially Trained Graph Convolutional Networks Resist Oversmoothing (https://arxiv.org/abs/2410.13416)
- **What's New**: 이 연구에서는 Kipf와 Welling이 제안한 GCN(그래프 컨볼루션 네트워크)의 훈련되지 않은 상태에서도 의미 있는 노드 임베딩을 생성할 수 있다는 관찰을 탐구합니다. 특히, GCN의 단일 레이어만 훈련하고 나머지 레이어는 고정한 채로 두는 경우의 영향을 조사하고, 이를 바탕으로 임베딩 생성에 관한 예측의 기초를 제안합니다.

- **Technical Details**: 단일 레이어 GCN만 훈련된 경우에 노드 임베딩 생성에 미치는 영향과 네트워크 너비의 관계를 분석합니다. 실험 결과, 부분적으로 훈련된 GCN의 너비를 증가시키면 성능이 향상되고 과도한 스무딩(oversmoothing)을 감소시킬 수 있음을 보였습니다. 또한, GCN의 훈련 가능한 레이어의 위치와 유형이 성능에 미치는 영향을 연구하였습니다.

- **Performance Highlights**: 실험 결과, 깊은 부분적으로 훈련된 GCN은 과도한 스무딩을 저항하는 능력이 있으며, ‘콜드 스타트’ 상황에서 노드 분류 작업을 수행할 때 높은 정확도를 달성할 수 있다는 점이 강조되었습니다.



### MoR: Mixture of Ranks for Low-Rank Adaptation Tuning (https://arxiv.org/abs/2410.13408)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 Mixture of Ranks (MoR)라는 새로운 접근 방식을 도입하여 LoRA의 성능을 개선하고, 다중 작업에서 효율적으로 학습할 수 있는 방법을 제안합니다. 기존의 LoRA와 MoE 방식의 한계를 극복하고, 다양한 작업에 대한 rank-specific 정보를 효과적으로 학습합니다.

- **Technical Details**: MoR은 세 가지 주요 구성 요소인 공유 전문가(shared experts), 다중 rank 적응(multi-rank adaptation), 혼합 학습(mixture learning)으로 구성됩니다. 이 방법은 여러 LoRA를 통합하여 학습할 수 있는 새로운 프레임워크를 제공하며, 매핑(mapping) 및 스케일링(scaling)을 통해 다중 작업을 수행합니다.

- **Performance Highlights**: MoR는 기존 LoRA 방법 대비 1.31%의 성능 향상을 달성하면서도 파라미터는 93.93%만 사용합니다. 또한, 다양한 실험에서 MoR은 효율성, 일반화 가능성, 확장성을 입증하며, 다중 LoRA 구조의 학습 비용을 크게 줄이고 더 간결한 정보를 동적으로 학습할 수 있음을 보여줍니다.



### Predicting Breast Cancer Survival: A Survival Analysis Approach Using Log Odds and Clinical Variables (https://arxiv.org/abs/2410.13404)
Comments:
          17 pages

- **What's New**: 이번 연구는 유방암 환자의 치료 및 예후에 대한 예측을 개선하기 위해 생존 분석 기법을 활용한 것입니다. 특히, Cox proportional hazards 모델과 파라메트릭 생존 모델을 사용하여 생존의 로그 오즈(log odds)를 예측합니다.

- **Technical Details**: 1557명의 유방암 환자의 데이터를 사용하였으며, 임상 변수로는 종양 크기(tumor size), 호르몬 수용체 상태(hormone receptor status), HER2 상태(HER2 status), 나이(age), 치료 이력(treatment history) 등을 분석했습니다. Kaplan-Meier 생존 곡선과 Cox 비례 위험 모델을 통해 주요 위험 요소를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 나이가 많고 종양이 큰 경우 및 HER2 양성 상태가 사망 위험 증가와 관련이 있음을 발견했습니다. 반면, 에스트로겐 수용체 양성과 유방 보존 수술은 생존 결과에 긍정적인 영향을 미쳤습니다. 이러한 임상 변수를 예측 모델에 통합함으로써 생존 예측의 정확성을 높일 수 있음을 제시합니다.



### A Self-Constructing Multi-Expert Fuzzy System for High-dimensional Data Classification (https://arxiv.org/abs/2410.13390)
- **What's New**: 본 논문에서는 Self-Constructing Multi-Expert Fuzzy System (SOME-FS)이라는 새로운 퍼지 시스템을 제안합니다. 이는 고차원 데이터에서의 문제를 해결하고, Fuzzy Neural Networks (FNNs)가 직면한 도전과제들을 극복하는 데 도움을 줍니다.

- **Technical Details**: SOME-FS는 혼합 구조 학습(mixed structure learning)과 다중 전문가 고급 학습(multi-expert advanced learning)이라는 두 가지 학습 전략을 결합합니다. 각 기본 분류기는 사전 지식 없이 구조를 효과적으로 결정할 수 있으며, 각 규칙은 지역적 특성에 집중하여 소실되는 기울기 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SOME-FS는 고차원 탭ular 데이터에서 특히 불확실성을 다루는 데 효과적임을 보여줍니다. 또한, 안정적인 규칙 채굴 과정(stable rule mining process)을 통해 SOME-FS가 학습한 간결하고 핵심적인 규칙들을 식별할 수 있습니다.



### Data-Augmented Predictive Deep Neural Network: Enhancing the extrapolation capabilities of non-intrusive surrogate models (https://arxiv.org/abs/2410.13376)
- **What's New**: 본 논문에서는 파라메트릭 비선형 동적 시스템의 정확한 예측을 위해 새로운 딥 러닝 프레임워크인 Data-Augmented Predictive Deep Neural Network (DAPredDNN)를 제안합니다. 이 방법은 기존의 훈련 데이터가 주어진 시간 구간을 넘어서는 예측의 정확도를 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: DAPredDNN 모델은 Kernel Dynamic Mode Decomposition (KDMD)를 활용하여 Convolutional Autoencoder (CAE)의 인코더 부분에서 생성된 잠재 공간의 동적 변화를 발전시킵니다. CAE는 원래의 고차원 파라메트릭 데이터를 소규모의 잠재 공간으로 압축하고, 피드포워드 심층 신경망 (FFNN)은 파라미터-시간 쌍을 이 잠재 공간에 매핑합니다.

- **Performance Highlights**: 제안된 방법은 FitzHugh-Nagumo 모델과 원통 주위의 비압축 흐름 모델에서 테스트되었으며, 두 가지 경우 모두 시간 및 파라미터 영역에서 정확하고 빠른 예측 성능을 보여주었습니다.



### Addressing Heterogeneity and Heterophily in Graphs: A Heterogeneous Heterophilic Spectral Graph Neural Network (https://arxiv.org/abs/2410.13373)
- **What's New**: 이 논문에서는 이질적 이형성 그래프에 적용될 수 있는 새로운 Heterogeneous Heterophilic Spectral Graph Neural Network (H2SGNN)을 제안합니다. 이 모델은 서로 다른 동질성을 가진 서브그래프들을 독립적으로 필터링하는 로컬 독립 필터링 모듈과 다양한 서브그래프 간의 상호작용을 포착하는 글로벌 하이브리드 필터링 모듈을 통합합니다.

- **Technical Details**: H2SGNN은 두 가지 주요 모듈로 구성됩니다: 1) 로컬 독립 필터링, 각 서브그래프에 대해 독립적으로 다항 필터를 적용하여 다양한 동질성을 학습합니다. 2) 글로벌 하이브리드 필터링, 이를 통해 서로 다른 메타 경로를 결합하여 전세계적 인접 행렬을 생성하고 이를 기반으로 필터링을 수행합니다.

- **Performance Highlights**: H2SGNN은 네 개의 실제 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수한 성능을 보였으며, 파라미터 수와 메모리 요구량을 줄이면서도 뛰어난 결과를 달성했습니다.



### Statistical testing on generative AI anomaly detection tools in Alzheimer's Disease diagnosis (https://arxiv.org/abs/2410.13363)
- **What's New**: 이번 연구에서는 알츠하이머병 진단을 위한 신뢰할 수 있는 생성적 AI(Generative AI) 방법을 개발하고, 선택적 추론(selective inference)을 통해 가설 검증(hypothesis testing)에서 발생할 수 있는 이중 사용(double-dipping) 문제를 해결하고자 하였습니다.

- **Technical Details**: 알츠하이머병의 진단을 위해 시간 시계열 MRI(Progression)에서 측정되는 신경퇴행(Neurodegeneration)을 생체표지자(biomarker)로 연구하였습니다. 선택적 추론을 통해 전통적인 통계적 방법과 비교했을 때 잘못된 발견률(false discovery rate)을 효과적으로 관리하면서 통계적 힘(statistical power)을 유지하는 결과를 보였습니다.

- **Performance Highlights**: 본 연구의 방법론은 임상 의사(clinicians)가 알츠하이머 진단 및 조기 개입에 있어 도움을 줄 수 있는 가능성을 보여줍니다.



### Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data (https://arxiv.org/abs/2410.13341)
Comments:
          22 pages, 5 figures

- **What's New**: 이 논문은 유망한 고품질 라벨을 활용하여 많은 모델 판단의 편향을 교정할 수 있는 새로운 디바이싱(debiasing) 도구들의 효과를 연구합니다. 하지만 경험적인 평가에 기반한 이론적 한계에 따르면, 만약 판별자가 평가되는 모델과 같은 정확도에 불과하다면, 디바이싱 방법이 진짜 라벨 수요를 절반 이상 줄일 수 없음을 보입니다.

- **Technical Details**: 논문은 모델 평가를 위한 포괄적 설정을 다루며, 특히 분류(classification), 질문 답변(question answering), 경연 스타일 비교(arena-style comparisons) 및 안전성 평가(safety evaluations)에 초점을 맞추고 있습니다. 연구에서는 모델의 판단이 라벨 산출 점수에 어떻게 영향을 미치는지를 연구하며, 디바이싱 방법이 예측 점수를 통해 진정한 점수의 추정치를 제공하는 방법을 분석합니다.

- **Performance Highlights**: MMLU 및 MT-Bench의 실험을 통해 디바이싱 방법에서 두 배 이상의 샘플 크기 절약이 드물다는 것을 확인했습니다. 또한 판별자와 진짜 라벨 사이의 일반적인 합의 비율이 디바이싱 방법에서 발생하는 편향을 의미있게 제한하지 않음을 보여주었고, 새로운 균형 합의 메트릭을 제안하여 샘플 사이즈 절약을 평가하는 방법을 제안했습니다.



### DiffImp: Efficient Diffusion Model for Probabilistic Time Series Imputation with Bidirectional Mamba Backbon (https://arxiv.org/abs/2410.13338)
Comments:
          25 pages, 14 figures

- **What's New**: 이번 논문에서는 확률적 시계열 보간 기술을 위한 새로운 프레임워크인 DiffImp를 제안합니다. DiffImp는 시계열 데이터의 복잡한 분포를 모델링할 수 있는 능력으로 Denoising Diffusion Probabilistic Models (DDPMs)를 활용합니다.

- **Technical Details**: DiffImp는 효율적인 State Space Model (SSM)인 Mamba를 backbone으로 통합하여 denoising 모듈을 구성합니다. 이를 통해 낮은 시간 복잡도로 시퀀스 모델링이 가능하며, Bidirectional Attention Mamba block (BAM)을 통해 양방향 종속성을 처리하고, Channel Mamba Block (CMB)을 통해 서로 다른 채널 간의 의존성을 모델링합니다.

- **Performance Highlights**: DiffImp는 다양한 데이터 세트를 사용한 실험에서 최첨단 성능을 달성하였으며, 다양한 결측 시나리오와 결측 비율에서 우수한 결과를 보여주었습니다.



### Improving Discrete Optimisation Via Decoupled Straight-Through Gumbel-Softmax (https://arxiv.org/abs/2410.13331)
- **What's New**: 이 논문에서는 Straight-Through Gumbel-Softmax (ST-GS) 추정기를 개량한 'Decoupled ST-GS'를 제안합니다. 이 방법은 순방향(forward)과 역방향(backward) 패스를 위한 온도(temperature) 매개변수를 분리하여 gradient의 신뢰성과 모델 성능 간의 균형을 개선합니다.

- **Technical Details**: Decoupled ST-GS은 ST-GS 추정기의 구조를 바탕으로 하며, 온도 조정에서 수반되는 어려움을 해결하기 위해 각각의 패스에 대해 고유한 온도를 사용합니다. 이는 모델의 추론 시 스무딩(smoothing)과 훈련 시 gradient의 신뢰성을 독립적으로 조정할 수 있게 합니다.

- **Performance Highlights**: 다양한 과제(task)와 데이터 세트에서의 실험을 통해 Decoupled ST-GS가 기존 ST-GS 추정기보다 일관되게 뛰어난 성능을 보였음을 입증했습니다. 이 방법은 gradient gap과 bias-variance trade-off 분석을 통해 gradient 기반 최적화의 개선을 도모합니다.



### Precipitation Nowcasting Using Diffusion Transformer with Causal Attention (https://arxiv.org/abs/2410.13314)
- **What's New**: 이번 연구에서는 Diffusion Transformer with Causal Attention 모델을 제안하여 단기 강수 예보의 문제를 해결하고자 합니다. 이 모델은 Transformer를 활용하여, 조건 정보와 예보 결과 간의 시공간 쿼리를 효과적으로 설정할 수 있도록 합니다.

- **Technical Details**: DTCA(분산 변환기 인과 주의) 모델은 조건부 강수 분포 특징 관련 쿼리를 기반으로 한 새로운 인과 주의 메커니즘을 도입하며, 다양한 시공간 정보 상호작용을 탐색하고 그 구조를 비교합니다. 실험 결과, 전역 시공간 레이블링 상호작용이 최고의 성능을 발휘하는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 방법은 최신 기술인 U-Net 기반 방법과 비교해 강수 예측에서 약 15% 및 8% 개선된 CSI(비판적 성공 지표)를 달성하여 현재 상태의최고 성능을 기록했습니다.



### Hiformer: Hybrid Frequency Feature Enhancement Inverted Transformer for Long-Term Wind Power Prediction (https://arxiv.org/abs/2410.13303)
- **What's New**: 본 논문에서는 기후 변화의 심각성이 증가함에 따라 재생 에너지로의 긴급한 전환이 필요하다는 점을 강조하고, 특히 풍력 에너지의 대규모 채택이 환경 영향을 완화하는 데 중요하다고 설명합니다. 이는 장기적인 풍력 예측 모델의 필요성을 강조합니다.

- **Technical Details**: 이 논문은 Hybrid Frequency Feature Enhancement Inverted Transformer (Hiformer)라는 새로운 접근 방식을 제안합니다. Hiformer는 신호 분해 기술과 날씨 특징 추출 기술을 통합하여 기상 조건과 풍력 생성 간의 상관관계를 모델링하는 독특한 구조를 가지고 있습니다. 또한 Hiformer는 인코더 전용 아키텍처를 사용하여 장기 풍력 예측에 따른 계산 복잡성을 줄입니다.

- **Performance Highlights**: Hiformer는 최신 방법과 비교하여 예측 정확도를 최대 52.5% 향상시킬 수 있으며, 계산 시간을 최대 68.5% 단축할 수 있습니다.



### LLM-Rank: A Graph Theoretical Approach to Pruning Large Language Models (https://arxiv.org/abs/2410.13299)
- **What's New**: 본 논문에서는 그래프 이론의 중심성 측정을 활용한 새로운 프루닝(pruning) 방법인 MLPRank를 제안하였습니다. 이 방법은 다층 퍼셉트론(multilayer perceptron)과 디코더 전용(transformer) 모델을 대상으로 하여 계산 요구 사항과 메모리 사용량을 줄입니다.

- **Technical Details**: MLPRank는 가중 방향 비순환 그래프(weighted directed acyclic graph)를 생성하여 각 노드의 중요도를 평가하는 데 수정된 PageRank 중심성 측정을 사용합니다. 이와 함께 균일 프루닝(uniform pruning)을 적용하여 구조적 희소성(structured sparsity)을 달성합니다. 또한, 디코더 전용 모델에 대한 확장된 방법인 LLMRank도 소개하였습니다.

- **Performance Highlights**: MLPRank는 세 가지 인기 있는 기준과 비교하여 평균 6.09%의 정확도 유지를 보여주고, LLMRank는 두 가지 기준에 비해 13.42% 높은 성능을 보였습니다. 두 방법 모두 최신 구조적 프루닝 기법인 SliceGPT보다 평균 8.60% 높은 정확도를 기록했습니다.



### Fairness-Enhancing Ensemble Classification in Water Distribution Networks (https://arxiv.org/abs/2410.13296)
- **What's New**: 이 논문에서는 사회 경제적으로 중요한 기반 시설인 물 분배 네트워크(Water Distribution Networks, WDNs)에서 AI의 공정성 문제를 다루고 있습니다. WDNs의 공정성에 대한 기존 연구가 부족한 만큼, 이 연구는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구자들은 그룹 공정성(Group Fairness)의 개념을 도입하고, 비이진(Binomial) 민감 변수에 대한 정의를 명확히 하였습니다. 기존의 누수 탐지 방법들이 공정하지 않음을 입증하였고, 비구분 가능한 앙상블(Ensemble) 분류 방법에 적용할 수 있는 공정성을 높이기 위한 방법을 제안했습니다.

- **Performance Highlights**: 이 논문은 WDNs에서 머신러닝 모델의 공정성을 평가하기 위한 표준 방법론을 제시하고, 여러 개의 민감 변수를 고려하여 기존 방법론의 몇 가지 조정을 통해 공정성을 강화할 수 있음을 설명합니다.



### PiLocNet: Physics-informed neural network on 3D localization with rotating point spread function (https://arxiv.org/abs/2410.13295)
Comments:
          25 pages, 4 figures

- **What's New**: 이 논문은 3D localization 문제 해결을 위한 새로운 개선된 Neural Network PiLocNet을 제안합니다. PiLocNet은 기존의 LocNet을 기반으로 하며, forward-model 기반 정보와 data-fitting loss term을 통합하여 물리적으로 합리적인 결과를 도출합니다.

- **Technical Details**: PiLocNet은 Physics-Informed Neural Network (PINN)으로, 이미징 시스템의 포인트 스프레드 함수(PSF)를 통해 물리적 정보를 네트워크에 통합합니다. 세 가지 중요한 구성 요소로는 forward-model 정보, variational method의 regularization term, 그리고 Poisson 및 Gaussian noise 모델을 통한 이미지 노이즈에 대한 강건성 개선이 포함됩니다.

- **Performance Highlights**: 실험 결과, PiLocNet은 3D 포인트 소스의 localization을 위한 정확도 향상에 기여하며, precision과 recall 측면에서 개선된 예측 결과를 보여줍니다. 본 논문은 PiLocNet의 Robustness를 검증했으며, 다양한 PSF와 이미징 문제에서의 적용 가능성을 제시합니다.



### SBI-RAG: Enhancing Math Word Problem Solving for Students through Schema-Based Instruction and Retrieval-Augmented Generation (https://arxiv.org/abs/2410.13293)
Comments:
          Accepted to the 4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 본 논문에서는 Schema-Based Instruction Retrieval-Augmented Generation (SBI-RAG) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 통합하여 수학 단어 문제(MWP)를 해결하는 과정을 지원하며, 기존의 Schema-Based Instruction(SBI) 방법을 바탕으로 발전했습니다.

- **Technical Details**: SBI-RAG는 기본적으로 네 가지 주요 부분으로 나뉩니다: 1) Schema Classifier, 2) Prompt Creation, 3) Context Retrieval, 4) Answer and Response Generation. Schema Classifier는 DistilBERT를 기반으로 하여 특정 문제에 적합한 schema를 예측하고, 그에 따라 schema-specific prompt를 생성합니다. 이후 Retrieval-Augmented Generation(RAG) 프레임워크를 이용하여 관련 문서를 검색하고, LLM을 통해 구체적인 단계별 해답을 생성합니다.

- **Performance Highlights**: GSM8K 데이터셋에서의 평가 결과, SBI-RAG는 GPT-4 및 GPT-3.5 Turbo와 비교하여 문제 해결의 정확성과 추론의 명료성을 향상시키는 데 효과적임을 보였습니다. 새로운 'reasoning score' 메트릭을 도입하여 LLM의 해결 과정의 질을 평가하였으며, 이는 학생들의 교육적 이점을 제공할 가능성이 있습니다.



### An Online Learning Approach to Prompt-based Selection of Generative Models (https://arxiv.org/abs/2410.13287)
- **What's New**: 이 연구는 다양한 텍스트 프롬프트에 대해 최적의 생성 모델을 온라인으로 식별하는 가능성을 탐구하고, 이를 위해 커널 기반의 컨텍스트 밴딧(Contextual Bandit, CB) 방법론을 제안합니다. 이 방법은 프롬프트와 생성된 데이터를 활용하여 데이터를 업데이트하고 다음 텍스트 프롬프트에 대해 가장 높은 점수를 얻을 모델을 예측합니다.

- **Technical Details**: 제안된 방법론은 Shared-Context Kernel UCB(SCK-UCB)라는 알고리즘을 기반으로 하며, 커널 기반 예측 함수를 사용하여 UCB 점수를 산출하고 이를 통해 최적의 생성 모델을 선택합니다. 또한, Random Fourier Features(RFF) 기술을 적용하여 온라인 학습 과정을 가속화하고, RFF-UCB 알고리즘에 대해 O(√T) 회귀 경계를 증명합니다.

- **Performance Highlights**: 실제 및 시뮬레이션된 텍스트-이미지 및 이미지-텍스트 생성 모델에 대한 수치 실험 결과, RFF-UCB가 다양한 샘플 유형에 대해 최적의 생성 모델을 성공적으로 식별하는 데 높은 성능을 보여줍니다. 제안된 알고리즘은 탐색을 유도하기 위한 보너스 항 없이도 기존의 탐욕적 기준선을 초월하는 성과를 나타냅니다.



### A Human-in-the-Loop Fairness-Aware Model Selection Framework for Complex Fairness Objective Landscapes (https://arxiv.org/abs/2410.13286)
- **What's New**: 이 논문은 여러 공정성 목표를 고려해야 하는 복잡한 사회적 및 법적 요구사항을 다루는 fairness-aware Machine Learning (FairML) 응용 프로그램의 한계를 해결하기 위해, 공정성을 다목적(Many-objective, MaO) 문제로 접근하는 새로운 프레임워크인 ManyFairHPO를 소개합니다.

- **Technical Details**: ManyFairHPO는 인간 중심의 프로세스를 통해 공정성 메트릭의 갈등을 식별하고 평가하며 균형을 맞추는 지원하는 모델 선택 프레임워크입니다. 이 연구에서는 Law School Admissions 문제에 대한 사례 연구를 통해 ManyFairHPO가 어떻게 복잡한 공정성 목표에 대한 균형을 이룰 수 있는지를 보여줍니다.

- **Performance Highlights**: ManyFairHPO는 일반적인 bi-objective (BiO) 문제 프레임워크와 비교할 수 있을 만큼 경쟁력 있는 성능을 보여주었으며, 공정성 메트릭 간의 상충 관계를 탐색하며 관련된 사회적 결과를 위한 더 많은 정보에 기반한 모델 선택 결정을 지원합니다.



### The Latent Road to Atoms: Backmapping Coarse-grained Protein Structures with Latent Diffusion (https://arxiv.org/abs/2410.13264)
Comments:
          Paper under review

- **What's New**: 본 논문에서는 Latent Diffusion Backmapping (LDB)이라는 새로운 접근법을 제시하여, coarse-grained (CG) 분자 동역학 시뮬레이션에서의 단백질 구조 복원을 위한 효율성을 높입니다. 기존의 backmapping 방법들이 직면했던 문제들을 해결하기 위해, LDB는 노이즈 제거(diffusion) 기법을 이용해 라텐트(latent) 공간 내에서 구조를 효율적으로 재구성합니다.

- **Technical Details**: LDB는 노드 레벨의 라텐트 표현을 통해 모든 원자 구조를 인코딩하며, 화학적 유효성을 보장하기 위해 물리적 제약(예: 결합 길이 및 각도)을 적용합니다. 이는 광범위한 후처리 과정 없이도 화학적 유효성을 달성하고, 미세조정되지 않은 개별 원자 구조 복원을 통해 동적인 구조 공간 탐색을 용이하게 만듭니다. 이 방법은 조건부(conditional) 노이즈 제거 모델을 포함하여 디스크리트 라텐트 코드에서 작동함으로써 예측 정확성과 다양한 단백질 구조 생성을 극대화합니다.

- **Performance Highlights**: LDB는 PED, ATLAS 및 PDB와 같은 여러 단백질 다이너믹스 데이터셋에서 최신 성능을 입증하였으며, 구조적 정확성과 화학적 유효성을 유지하며 단백질 앙상블을 효율적으로 복원하는 능력을 보여주었습니다. 이러한 개선 사항에 따라 LDB는 CG 시뮬레이션과 원자 수준 분석 간의 격차를 효과적으로 연결하는 강력하고 확장 가능한 접근법으로 자리잡았습니다.



### scFusionTTT: Single-cell transcriptomics and proteomics fusion with Test-Time Training layers (https://arxiv.org/abs/2410.13257)
- **What's New**: 이번 논문에서는 CITE-seq 데이터를 활용한 단일 세포 다중 오믹스(scMulti-omics) 분석을 위한 새로운 방법인 scFusionTTT를 제안합니다. 이 방법은 Test-Time Training (TTT) 기반의 마스크 오토인코더(masked autoencoder)를 사용하여 유전자와 단백질 순서 정보를 통합하고, 다중 오믹스 데이터를 융합하는 데 기여합니다.

- **Technical Details**: scFusionTTT는 TTT 레이어를 적용하여 다중 오믹스 데이터를 처리하는 혁신적인 접근 방식을 제공합니다. 기존의 모형들이 간과했던 유전자 간의 순차적 관계를 고려하며, 고차원 데이터의 희소성 문제를 해결하기 위해 세 가지 단계의 훈련 전략을 사용합니다. 이 방법은 단일 세포 전사체(transcriptomics) 및 단백질체(proteomics) 데이터를 통합하여 균형 잡힌 표현을 학습하도록 고안되었습니다.

- **Performance Highlights**: scFusionTTT는 4개의 CITE-seq 데이터셋과 4개의 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터셋에서 우수한 성능을 보여주었습니다. 모든 비교 기준에서 기존의 최첨단 방법들과 비교했을 때 더 나은 결과를 달성하였으며, 이는 모델의 유용성과 신뢰성을 입증하는 데 기여합니다.



### FDF: Flexible Decoupled Framework for Time Series Forecasting with Conditional Denoising and Polynomial Modeling (https://arxiv.org/abs/2410.13253)
- **What's New**: 이번 연구에서는 시간 시계열 예측의 유연한 분리 프레임워크(Flexible Decoupled Framework, FDF)를 제안하여 기존의 확산 모델이 가지고 있는 무차별적인 노이즈 추가의 문제를 해결하고자 합니다. 이 프레임워크는 시간 시계열 데이터를 추세(trend)와 계절성(seasonal) 구성 요소로 분리하여 각각을 모델링함으로써, 예측 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: FDF는 두 가지 핵심 모듈로 구성되어 있습니다: 조건부 노이즈 제거 계절 모듈(Conditional Denoising Seasonal Module, CDSM)과 다항 추세 모듈(Polynomial Trend Module, PTM). CDSM은 역사적 정보를 기반으로 복잡한 계절적 패턴을 조건부로 모델링하며, PTM은 시간에 따라 부드러운 추세 변화를 효율적으로 포착합니다. 이러한 분리에 의해 시간 시계열의 동적 패턴을 보다 정교하게 모델링할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 FDF가 기존 방법들과 비교하여 우수한 성능을 보였음을 입증하였으며, 시간 시계열 예측의 유연성을 강조하였습니다. 연구 결과는 향후 오픈 소스 코드로 공개될 예정입니다.



### Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation (https://arxiv.org/abs/2410.13248)
- **What's New**: 최근 설명 가능한 추천 시스템에 대한 연구는 표준 텍스트 생성 문제로 접근하며, 모델을 예측된 텍스트와 실제 텍스트 간의 유사성을 기반으로 평가합니다. 그러나 이 접근법은 사용자(구매 후) 감정을 정확히 반영하는지 여부를 간과합니다. 이 연구에서는 사용자의 감정을 중점적으로 고려하는 새로운 데이터셋과 평가 방법을 소개합니다.

- **Technical Details**: 우리는 LLM(Long Language Model)을 사용하여 사용자 구매 후 리뷰에서 긍정적 및 부정적 의견을 명시적으로 추출하여 데이터셋을 구성합니다. 시스템을 평가할 때 생성된 설명이 1) 사용자 감정과 잘 일치하는지, 2) 목표 아이템에 대한 사용자 의견의 긍정적 및 부정적 식별을 정확히 수행하는지에 대한 두 가지 기준을 제안합니다.

- **Performance Highlights**: 여러 최신 모델을 우리의 데이터셋에서 벤치마킹하였으며, 기존 지표에서 높은 성과를 달성하더라도 생성된 설명이 사용자 감정과 잘 일치하지 않을 수 있음을 보여줍니다. 또한, 목표 아이템에 대한 사용자(예측된) 평가가 모델에 직접 입력될 경우, 기존 모델들이 보다 감정 인식적인 설명을 제공할 수 있음을 발견하였습니다.



### Quamba: A Post-Training Quantization Recipe for Selective State Space Models (https://arxiv.org/abs/2410.13229)
- **What's New**: 본 연구는 State Space Models (SSMs)을 위한 정적 8-bit per-tensor quantization 방법을 제안하여 모델의 효율성을 크게 향상시키고, 클라우드 및 에지 플랫폼에서의 원활한 배포를 지원합니다.

- **Technical Details**: 제안된 방법은 Hadamard 변환을 활용하여 SSM의 출력 활성화에서 극단적인 아웃라이어를 부드럽게 하고, 선택적 SSM에 대한 입력 활성화의 최대값을 억제하여 더 섬세한 quantization 정밀도를 제공합니다. 또한, 8-bit 정량화된 Mamba 2.8B SSM은 Nvidia Orin Nano 8G에서 1.72배 더 낮은 생성 지연 시간을 달성하며, 평균 정확도는 0.9% 감소했습니다.

- **Performance Highlights**: Mamba SSM은 하드웨어 가속의 이점을 누리며, 다양한 크기의 SSM 기반 모델의 클라우드 및 에지 플랫폼에서의 효과성과 실용성을 입증했습니다.



### From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning (https://arxiv.org/abs/2410.13228)
Comments:
          physics-informed neural networks, Kolmogorov-Arnold networks, optimization algorithms, separable PINNs, self-adaptive weights, uncertainty quantification

- **What's New**: 최근 Physicas-Informed Neural Networks (PINNs)의 발전이 뚜렷하며, 새로운 구조와 최적화 기법이 포함된 Physics-Informed Kolmogorov-Arnold Networks (PIKANS) 등이 소개되었습니다. PINNs는 물리 법칙을 반영하여 희소 데이터로도 효율적으로 PDE(Partial Differential Equations)를 해결합니다.

- **Technical Details**: PINNs는 물리 법칙을 인코딩하기 위해 추가적인 ‘residual’ 손실 항을 포함하며, 자동 미분을 이용하여 효율적으로 계산합니다. 이를 통해 전통적인 수치적 방법 한계를 극복하고, 복잡한 기하구조를 다루는 데 유리합니다. 또한, PINNs는 불확실성 정량화 및 다양한 최적화 기법을 수용할 수 있습니다.

- **Performance Highlights**: PINNs는 생의학, 유체 역학, 지구 물리학 등 다양한 분야에서 적용 가능함을 보여주었으며, 2017년 첫 발표 이후 11,000회 이상의 인용이 이루어졌습니다. 또한, 여러 연구 그룹에서 PINNs의 알고리즘 개선 및 적용 가능성을 탐색하고 있습니다.



### MixEHR-Nest: Identifying Subphenotypes within Electronic Health Records through Hierarchical Guided-Topic Modeling (https://arxiv.org/abs/2410.13217)
- **What's New**: MixEHR-Nest는 전자 건강 기록(EHR) 데이터를 활용하여 고유한 하위 표현형(sub-phenotype) 주제를 유도하는 새로운 지침(topic model) 모델입니다. 이 모델은 경험적인 표현형 개념(PheCodes, CCS 코드를 포함)으로 초기화된 하위 주제를 탐지하여 질병 패턴을 더욱 세분화하여 나타냅니다.

- **Technical Details**: MixEHR-Nest는 다중 모달(multi-modal) EHR 데이터에서 1500개 이상의 표현형으로부터 뚜렷한 하위 표현형 주제를 유도할 수 있는 구조화된 하이라키(topic model)입니다. 이 모델은 각 환자의 의료 기록을 문서(document)로, 코드(예: ICD 코드)를 단어 토큰(word tokens)으로 간주하여 학습합니다. 이 연구는 하위 표현형 주제의 묘사, 다중 유형의 EHR 정보 학습, 높은 해석 가능성을 통한 자동 하위 표현형 유도를 포함합니다.

- **Performance Highlights**: MixEHR-Nest는 ICU 환자 사망률 예측, 당뇨병 환자의 초기 인슐린 치료 예측에서 성능을 향상시켰습니다. 또한 MixEHR-Nest는 같은 표현형 아래에서 연령 분포의 뚜렷한 하위 표현형을 확인함으로써 다양한 질병에 걸쳐 질병의 진행 및 중증도를 예측하는 데 기여했습니다.



### Balancing Label Quantity and Quality for Scalable Elicitation (https://arxiv.org/abs/2410.13215)
- **What's New**: 최근 연구에서 언급된 바와 같이, 대규모 인터넷 데이터셋으로 사전 학습된 언어 모델(LM)은 오류가 있는 레이블로 미세 조정하더라도 정확한 답변을 생성하는 경향을 보입니다. 본 논문에서는 저렴한 레이블 품질의 거래 비용과 효율성을 연구하며, 고품질과 저품질 레이블을 혼합하여 사용하는 방안이 보다 효율적임을 증명합니다.

- **Technical Details**: 본 논문은 세 가지 레이블링 체계(quality-dominant, quantity-dominant, mixed)에서 사전 학습된 모델을 사용한 분류 지식 유도 방법을 제시합니다. 레이블 품질과 양의 거래 비용을 최적화하는 방법론을 탐구하며, 효율적인 샘플링 방법을 제안합니다. 본 연구는 Burns et al. (2023)의 이론적 배경을 바탕으로 하며, 다양한 데이터셋에 대해 유용한 성과를 확인합니다.

- **Performance Highlights**: 문헌에서 제공된 데이터셋을 사용하여, 여러 레이블 조합을 통한 분류 정확도 및 레이블링 비용의 최적화된 경과를 확인하였으며, mixed regime에서의 조합이 가장 높은 성과를 보였습니다. 또한, 레이블의 품질과 양에 관한 저비용-고성능의 파레토 경계를 설정하여 실제 현업에서 활용 가능한 기법을 제시합니다.



### AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations (https://arxiv.org/abs/2410.13212)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 Large Language Model (LLM)에서 Key-Value Cache (KV Cache)의 비대칭적 구조적 역할을 심층적으로 탐구하고, 이를 바탕으로 새로운 비대칭 양자화 전략인 AsymKV를 제안합니다. 기존의 양자화 기법이 키와 값 행렬에 동일한 구성을 사용하는 것 대신, 키와 값 행렬에 서로 다른 비트 양자화를 적용하는 방식을 채택합니다.

- **Technical Details**: 연구에서는 모델 양자화의 일환으로, KV Cache의 키 행렬과 값 행렬에 대해 비대칭적으로 1비트 양자화를 적용하는 방법을 제시합니다. 이 과정에서 각 행렬의 구조적 특성을 고려하여, 초기 몇 개의 디코더 레이어에는 4비트 양자화를 적용하고 이후의 레이어에는 1비트를 적용하는 방식으로 구현합니다. 또한, 실험을 통해 다양한 데이터셋에서 이 방법을 검증하였습니다.

- **Performance Highlights**: 실험 결과, AsymKV 접근법은 최대 75%의 디코더 레이어를 1비트로 양자화하면서도, 부동소수점 매개변수를 사용할 때와 비슷한 성능 수준을 유지하는 것으로 나타났습니다. 이는 메모리 소비를 줄이면서도 모델의 성능을 보장할 수 있는 효율적인 전략임을 입증합니다.



### Estimating the Probabilities of Rare Outputs in Language Models (https://arxiv.org/abs/2410.13211)
Comments:
          27 pages, 9 figures

- **What's New**: 이 논문은 저확률 추정(low probability estimation) 문제를 다루며, 이는 머신 러닝 모델의 출력으로부터 특정 이진 속성을 추정하는 과정을 포함합니다. 이러한 추정은 확률이 매우 작아 랜덤 샘플링(random sampling)으로는 불가능할 수 있으며, 분포 이동(distribution shift) 문제를 해결하기 위해 필수적입니다.

- **Technical Details**: 저자들은 두 가지 방법을 비교합니다: 1) Importance Sampling(중요도 샘플링): 드문 출력(event)을 생성하는 입력을 위해 새로운 입력 분포를 정의하는 방법입니다. 이 방법에는 Independent Token Gradient Importance Sampling(ITGIS)과 Metropolis-Hastings Importance Sampling(MHIS)이 포함됩니다. 2) Activation Extrapolation(활성화 외삽): 모델 로그잇(logits)에 맞춰 확률 분포를 피팅하고, 이를 기반으로 외부로 확장하는 방법입니다. 이 방법은 Quadratic Logit Decomposition(QLD)과 Gaussian Logit Difference(GLD)로 나눌 수 있습니다.

- **Performance Highlights**: 실험 결과, 중요도 샘플링 방법이 활성화 외삽보다 우수하며, 두 방법 모두 무작위 샘플링보다 좋습니다. 이는 최악의 성능 보장을 제공하기 위한 새로운 저확률 추정 기법이 필요하다는 점을 강조합니다.



### TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering (https://arxiv.org/abs/2410.13203)
Comments:
          This paper has been accepted for presentation at the 26th International Conference on Pattern Recognition (ICPR 2024) in Kolkata, India

- **What's New**: 이번 연구에서는 TabSeq라는 새로운 프레임워크를 소개하여, 비정형(tabular) 데이터의 최적의 특성(feature) 순서를 위해 제안되었습니다. 이는 딥러닝 모델의 학습 효율성을 높이는 데 기여할 것입니다.

- **Technical Details**: TabSeq 프레임워크는 클러스터링(clustering)과 지역(local) 및 전역(global) 정렬 기법을 결합하여 비정형 데이터의 특성을 최적화하는 기능을 제공합니다. 이 기술은 멀티 헤드 어탠션(multi-head attention) 메커니즘이 포함된 디노이징 오토인코더(denoising autoencoder) 네트워크와 함께 사용됩니다.

- **Performance Highlights**: 본 연구는 원시 항체 마이크로어레이(raw antibody microarray) 및 기타 두 개의 실제 생물 의학 데이터셋을 통해 제안된 특성 순서 조정 기법이 딥러닝 모델 성능을 유의미하게 개선할 수 있음을 보여주었습니다.



### Golyadkin's Torment: Doppelg\"angers and Adversarial Vulnerability (https://arxiv.org/abs/2410.13193)
- **What's New**: 이 논문은 'Adversarial Doppelgangers(AD)'라는 개념을 정의하고 탐구하며, 이는 기존의 adversarial visual metamers를 포함합니다. AD의 성능 및 강건성을 분류 기계와 인간의 성능을 비교하여 분석합니다.

- **Technical Details**: AD는 이 논문에서 정의된 지각적(metric) 측정에 따라 서로 가까운 입력들입니다. 연구에서는 이러한 AD에 대한 분류기의 취약성을 분석하고, AD에 강건한 분류기의 구조와 속성을 설명하며, 개념적 엔트로피(conceptual entropy) 및 개념적 모호성(regions of conceptual ambiguity)에 대한 개념을 도입합니다.

- **Performance Highlights**: 대부분의 분류기는 AD에 취약하며, 강건성-정확도 트레이드오프(robustness-accuracy trade-offs)가 개선되지 않을 수 있습니다. 그러나 정확도가 높은 모든 분류기는 hypersensitive behavior를 보여줄 있으며, 이로 인해 AD 강건성을 개선하는 것이 정확도 개선과 동일함을 발견했습니다.



### CohEx: A Generalized Framework for Cohort Explanation (https://arxiv.org/abs/2410.13190)
- **What's New**: 이번 논문은 설명 가능한 인공지능(eXplainable Artificial Intelligence, XAI)의 발전을 위해서 새로운 코호트 기반(cohort-based) 설명 방법에 대해 탐구합니다. 기존의 설명 기법들이 전반적인(global) 또는 개별적(local) 설명에 중점을 두고 있는 반면, 본 연구에서는 특정 그룹에 대한 설명의 중요성을 강조합니다. 이를 통해 모델의 결정과정에 대한 보다 깊은 이해를 제공합니다.

- **Technical Details**: 코호트 설명(cohort explanation)은 데이터셋의 하위 집합 또는 모델의 입력/결정 공간의 하위 영역에 대한 일반화된 설명을 제공합니다. 연구진은 이와 관련된 고유한 도전과 기회를 논의하며, 코호트 설명의 이상적인 속성을 정의하고 이를 기반으로 한 일반화된 프레임워크(supervised clustering)를 제안합니다. 또한, 기존의 데이터 기반(local feature importance) 방법을 코호트 설명으로 전환하는 방법도 개발하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 알고리즘은 기존의 벤치마크와 비교하여 우수한 성능을 발휘했다고 보고합니다. 이는 코호트 설명이 각 코호트를 특정 지역으로 구분하여 보다 간결하고 명확한 설명을 가능하게 하기 때문입니다.



### GeSubNet: Gene Interaction Inference for Disease Subtype Network Generation (https://arxiv.org/abs/2410.13178)
Comments:
          Under review as a conference paper at ICLR 2025

- **What's New**: GeSubNet은 다양한 질병 아형에 따른 유전자 상호작용을 예측할 수 있는 통합 표현을 학습하는 새로운 방법론입니다.

- **Technical Details**: GeSubNet은 다단계 표현 학습 프레임워크로, 환자 유전자 발현 프로파일로부터 특정 질병 아형을 학습하는 심층 생성 모델, 지식 데이터베이스로부터 이전 유전자 네트워크의 표현을 포착하는 그래프 신경망(GNN), 그리고 두 표현을 통합하는 추론 손실을 포함한 세 가지 모듈로 구성됩니다.

- **Performance Highlights**: GeSubNet은 네 가지 그래프 평가 지표에서 평균 30.6%, 21.0%, 20.1%, 56.6%의 성능 향상을 기록했으며, 11,327개의 유전자 평가 실험에서 특정 아형에 영향을 미칠 가능성이 83%인 유전자 발견의 잠재력을 보여주었습니다.



### TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness (https://arxiv.org/abs/2410.13175)
- **What's New**: 이번 연구는 Tropical Cyclone Precipitation Diffusion (TCP-Diffusion)이라는 다중 모드 모델을 제안하여 열대성 폭풍(TC) 강수 예측의 정확성을 높였습니다. 이 모델은 과거 강수 관측 및 다양한 환경 변수에 기반하여, TC 중심 주변의 강수를 다음 12시간 동안 3시간 간격으로 예측합니다.

- **Technical Details**: TCP-Diffusion 모델은 인접 잔차 예측(Adjacent Residual Prediction, ARP) 방식을 사용하여, 절대 강수량 대신 강수량 변화를 예측하도록 훈련 목표를 변경했습니다. 이를 통해 누적 오류를 줄이고 물리적 일관성을 확보합니다. 또한, 기상 요소와 수치 기상 예측(NWP) 모델 정보를 통합하여 더 풍부한 메타데이터를 추출합니다.

- **Performance Highlights**: 광범위한 실험 결과, TCP-Diffusion는 최신 딥 러닝(DL) 기반 강수 예측 방법 및 유럽 중기기상예보센터(ECMWF)의 NWP 방법과 비교하여 우수한 성능을 보여주었습니다.



### An Evolved Universal Transformer Memory (https://arxiv.org/abs/2410.13166)
Comments:
          29 pages, 14 figures. Preprint, under submission. Source code is available at this https URL

- **What's New**: 본 논문은 Neural Attention Memory Models (NAMMs)을 제안하며, 메모리 관리를 위한 학습된 네트워크를 도입하여 Transformers의 성능과 효율성을 동시에 향상시킵니다. 이는 기계가 가진 메모리 관리의 질의를 진화 기반 접근법으로 해결하여, 기능적으로 매우 다양한 아키텍처에서 자율적으로 적용될 수 있도록 설계되었습니다.

- **Technical Details**: NAMMs는 Transformers의 Key-Value (KV) 캐시의 잠재적 메모리를 형성하는 새로운 방법을 제안하여, 각 레이어와 attention head가 그들의 특정 요구에 가장 관련 있는 정보에 집중하도록 지원합니다. 이 방식은 학습된 attention 매트릭스를 기반으로 모든 transformer 기반 아키텍처에 일반적으로 적용 가능하며, Llama 3 8B 모델 위에서 학습하여 성능과 효율성을 모두 극대화합니다.

- **Performance Highlights**: NAMMs를 통한 학습 결과로 36개의 LongBench, InfiniteBench 및 새로운 일본어 벤치마크에서 뛰어난 성능 개선을 기록했습니다. 기존의 수작업 전략과 비교할 때, NAMMs는 성능 저하 없이 메모리 용량을 유의미하게 감소시켰습니다. 또한, NAMMs는 언어 과제로만 학습되었음에도 불구하고 다양한 입력 모달리티를 통해 다른 transformer 모델에 제로샷 전이(transfer) 되는 성과를 보였습니다.



### Utilizing Large Language Models in An Iterative Paradigm with Domain Feedback for Molecule Optimization (https://arxiv.org/abs/2410.13147)
- **What's New**: 본 연구에서는 약물 발견에서 분자의 최적화를 지원하기 위해 LLM (Large Language Models)을 효과적으로 활용할 수 있는 도메인 피드백 제공자인 Re²DF를 제안합니다. 이 새로운 접근법은 분자가 화학적으로 유효하지 않을 경우를 고려하여 수정된 분자의 유효성을 즉시 검증하며, 해당 분자의 개선을 위한 구체적인 피드백을 제공합니다.

- **Technical Details**: Re²DF는 외부 툴킷인 RDKit를 이용하여 수정된 분자가 화학적으로 유효한지를 체크합니다. 만약 유효하지 않다면, RDKit로부터 오류 메시지를 제공받아 LLM에게 수정 방향을 제시합니다. 또한, 수정된 분자가 원하는 특성을 충족하는지 확인하여, 목표에 대한 정확한 방향과 거리를 제공하는 신뢰할 수 있는 피드백을 생성합니다.

- **Performance Highlights**: Re²DF는 단일 속성 목표 20개에서 Hit ratio를 각각 16.95% 및 20.76% 향상시키며, 다중 속성 목표에서는 32개에서 각각 6.04% 및 5.25% 향상시켰습니다. 이러한 결과는 Re²DF가 기존 방법들보다 더 나은 성능을 발휘함을 알립니다.



### Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity (https://arxiv.org/abs/2410.13141)
- **What's New**: 이번 논문은 분산된 데이터와 개인 정보 보호 문제를 해결하기 위해 연합 학습(federated learning, FL)과 과학 기계 학습(scientific machine learning, SciML)의 통합을 탐구합니다. FL과 SciML을 조합하여 복잡한 함수 근사 및 미분 방정식 해결을 위해 Federated Physics-informed Neural Networks (FedPINN)와 Federated Deep Operator Networks (FedDeepONet)라는 두 가지 새로운 모델을 제안합니다.

- **Technical Details**: 이 연구에서는 비독립적이고 동일하게 분포되지 않은(noniid) 데이터의 수준을 정량화하기 위해 1-Wasserstein 거리(1-Wasserstein distance)를 활용하고, 데이터 이질성과 연합 모델 성능 간의 관계를 체계적으로 조사합니다. 또한, 연합 학습에서의 가중치 발산(weight divergence)을 측정하고, 전통적인 중앙 집중식 학습과 비교하여 가중치 발산의 성장 경계를 수립하는 이론적 프레임워크를 개발합니다.

- **Performance Highlights**: 10개의 실험을 통해 제안된 연합 모델들이 단지 지역 데이터만으로 학습된 모델들보다 뛰어난 성능을 보여주고, 모든 데이터를 사용하여 훈련된 중앙 집중식 모델들과 경쟁력 있는 정확도를 달성했습니다. 또한, FedDeepONet은 통신 효율성에 민감하지 않은 결과를 보여주었습니다.



### Controllable Generation via Locally Constrained Resampling (https://arxiv.org/abs/2410.13111)
Comments:
          arXiv admin note: text overlap with arXiv:2312.03905

- **What's New**: 이번 논문에서 저자들은 LLMs (Large Language Models)의 제한을 받는 샘플링 문제를 해결하기 위한 새로운 확률론적 접근 방식을 제안합니다. 기존의 greedy 방법론 대신 Bayesian conditioning을 통해 더 글로벌한 제약 생성이 가능하도록 개선하였습니다.

- **Technical Details**: 제안된 방법은 LLM 샘플에서 유도된 지역적인 분포를 기반으로 하며, 이를 사용하여 제약을 조건화하고 샘플링합니다. 이 접근법은 싱글 토큰별로 제약을 강제로 이행하는 것이 아니라 전체 시퀀스를 고려하여 제약을 적용합니다. 제약 회로(Constraint Circuits)를 통해 Boolean python 함수를 사용하여 제약을 효율적으로 표현할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 LLM의 독소 생성 방지 및 Sudoku 퍼즐 해결과 같은 여러 작업에서 평가되었습니다. 특히, 독소 표현의 리스트를 제외함으로써 모델의 출력을 독소 생성에서 멀어지게 하여 이전의 방법론보다 우수한 성능을 보였으며, Sudoku 퍼즐에서는 100%의 정확도를 달성했습니다. GPT4-o와 Gemini 1.5와 비교할 때 이들의 정확도는 각각 26% 및 45%에 불과했습니다.



### Algorithmic Content Selection and the Impact of User Disengagemen (https://arxiv.org/abs/2410.13108)
- **What's New**: 본 논문은 사용자가 불만족할 경우 이탈할 가능성을 고려한 컨텐츠 선택 문제에 대한 새로운 모델을 제시합니다. 기존의 다수의 팔을 당기는 의사결정 모델은 사용자 참여를 고정된 변수로 간주하여 이러한 이탈 가능성을 반영하지 않습니다.

- **Technical Details**: 제안된 모델은 사용자의 만족도와 직접 연결된 수익을 극대화하는 것이 아니라, 사용자 재참여 가능성을 고려합니다. 결과적으로, 사용자의 이전 경험에 따라 이탈 확률이 통계적으로 연관될 수 있음을 보여줍니다. 이 접근법은 다이나믹 프로그래밍(dynamic programming)을 사용하여 최적 정책을 효율적으로 계산할 수 있게 합니다.

- **Performance Highlights**: 온라인 학습 환경에서, 모델은 변동성이 큰 사용자 참여를 고려하여 잘 작동하며, 최대 O(√T)의 후회 보상을 달성할 수 있는 알고리즘을 제시합니다. 이는 사용자가 이탈할 때도 재참여와 수익 손실 간의 복잡한 거래를 효과적으로 관리할 수 있게 합니다.



### Cliqueformer: Model-Based Optimization with Structured Transformers (https://arxiv.org/abs/2410.13106)
- **What's New**: 본 논문에서는 기계 학습을 활용한 모델 기반 최적화(Model-Based Optimization, MBO) 문제를 해결하기 위한 새로운 접근 방식인 Cliqueformer를 소개합니다. Cliqueformer는 블랙박스 함수의 구조를 학습하여 높은 차원의 최적화 문제에서 성능을 향상시킵니다.

- **Technical Details**: Cliqueformer는 transformer 기반 아키텍처를 사용하여 기능적 그래픽 모델(Functional Graphical Model, FGM) 형태로 블랙박스 함수의 구조를 학습합니다. 이 모델은 디자인 후보에 대한 최적화 문제를 해결하기 위해 예측을 클리크의 FGM 상에 분해하고, 클리크들의 주변 분포가 넓은 범위를 커버하도록 강제합니다. 이 과정은 변별적 병목(Variational Bottleneck) 기법을 사용하여 수행됩니다.

- **Performance Highlights**: Cliqueformer는 여러 고차원 블랙박스 함수와 실제 화학 및 유전자 설계 작업에서 기존 방법들과 비교하여 뛰어난 성능을 보여주었습니다. 이 연구는 오프라인 데이터에서 모델 기반 최적화를 위해 기존의 보수적인 접근 방식을 우회하는 효과적인 전략을 제안합니다.



### Communication-Efficient and Tensorized Federated Fine-Tuning of Large Language Models (https://arxiv.org/abs/2410.13097)
- **What's New**: 본 논문에서는 파라미터 효율적인 미세 조정(PEFT) 방법을 여러 장치에 분산된 개인 데이터로 미세 조정하기 위한 새로운 방법인 FedTT 및 FedTT+를 제안합니다. 이 방법들은 Federated Learning(FL)과 통합되어 사용자 프라이버시를 보호하면서도 데이터 이질성 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: FedTT는 클라이언트 측 모델의 인코더/디코더 블록에 텐서화된 어댑터를 통합하여 LLM을 적응시키는 방법입니다. FedTT는 크로스-사일로 FL 및 대규모 크로스-디바이스 FL 모두에 적용될 수 있습니다. FedTT+는 데이터 이질성에 대한 강인성을 추가적으로 향상시키기 위해 텐서 요소의 일부를 적응적으로 동결하여 학습 가능한 파라미터 수를 줄입니다.

- **Performance Highlights**: BERT 및 LLaMA 모델에 대한 실험 결과, 제안된 방법들이 기존의 연합 PEFT 접근 방식과 비교하여 데이터 이질성 문제를 성공적으로 해결했으며, 최대 10배의 통신 비용 절감 효과를 보였습니다. FedTT+는 상태-of-the-art 크로스-사일로 FL 방법들을 능가하는 성능을 보여주었습니다.



### Self-Comparison for Dataset-Level Membership Inference in Large (Vision-)Language Models (https://arxiv.org/abs/2410.13088)
- **What's New**: 본 논문에서는 Self-Comparison Membership Inference (SMI)이라는 새로운 데이터셋 수준의 멤버십 추론 방법을 제안합니다. 기존의 Membership Inference Attack (MIA) 방법론의 한계를 극복하기 위해 설계된 이 방식은 정체된 데이터에 대한 비밀스러운 사용을 감지할 수 있습니다.

- **Technical Details**: SMI 방법은 멤버 데이터(membership data)의 접두사와 비멤버 데이터(non-membership data)의 접미사를 비교하여, 훈련 데이터에 대한 모델의 암기 현상을 유도합니다. 구체적으로는, 멤버 데이터가 주어졌을 때, 패러프레이징(paraphrasing)을 통해 두 세트의 분포가 어떻게 변화하는지를 비교합니다. 기존 MIA 방식은 반드시 그라운드 트루스 멤버 데이터를 요구하는 반면, SMI는 유사한 분포를 가지지 않아도 되는 보조 비멤버 세트를 요구합니다.

- **Performance Highlights**: SMI 방법은 다양한 LLMs 및 VLMs 모델에서 기존의 MIA 및 데이터셋 추론 기술보다 뛰어난 성능을 보였습니다. 이는 특히 그라운드 트루스 멤버 데이터에 대한 사전 지식이 없을 때에도 유효합니다. 실험 결과, 우리의 방법이 공개 모델, 파인 튜닝된 모델 및 API 기반 상업 모델에 이르기까지 여러 데이터셋에서 우수한 성능을 발휘함을 확인하였습니다.



### MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models (https://arxiv.org/abs/2410.13085)
- **What's New**: 본 논문에서는 Med-LVLMs의 사실성을 향상시키기 위해 MMed-RAG라는 다중 모달 RAG 시스템을 제안합니다. 이 시스템은 도메인 인식을 위한 검색 메커니즘, 적응형 검색된 컨텍스트 선택 방법, 그리고 검증 가능한 RAG 기반의 선호 미세 조정 전략을 포함하여, 의료 데이터의 다양한 분야에 대해 일반적이고 신뢰할 수 있는 접근 방식을 제공합니다.

- **Technical Details**: MMed-RAG는 세 가지 주요 요소로 구성됩니다: 1) 도메인 인식 검색 메커니즘 - 입력 의료 이미지에 적합한 검색 모델을 선택하기 위해 도메인 식별 모듈을 설계하였습니다. 2) 적응형 검색된 컨텍스트 선택 - 검색된 컨텍스트의 개수를 선택하는 방법입니다. 3) RAG 기반 선호 미세 조정 - 교차 모달 정렬을 개선하고 모델과 실제 간의 전체 정렬을 높이는 방법입니다.

- **Performance Highlights**: MMed-RAG는 5개의 의료 데이터세트에서 실험을 실시하여, Medical VQA와 보고서 생성 작업에서 각각 18.5% 및 69.1%의 사실 정확도를 향상시켰습니다. 전반적으로 MMed-RAG는 Med-LVLMs의 정확성을 평균 43.8% 개선하였습니다.



### FedCAP: Robust Federated Learning via Customized Aggregation and Personalization (https://arxiv.org/abs/2410.13083)
Comments:
          14 pages, 12 figures, 5 tables, accepted by 2024 Annual Computer Security Applications Conference (ACSAC 2024)

- **What's New**: FedCAP는 데이터 이질성과 Byzantine 공격에 강한 연합 학습(FL) 프레임워크로, 모델 업데이트 보정 메커니즘을 통해 클라이언트 간 모델 업데이트의 방향성과 크기를 포착합니다. 또한 맞춤형 모델 집계 규칙을 설계하여 유사 클라이언트 간의 협업을 촉진하고 악의적인 클라이언트의 모델 성능 저하를 가속화합니다.

- **Technical Details**: FedCAP는 네 가지 주요 구성요소로 이루어져 있습니다: 모델 보정 메커니즘, 맞춤형 집계 규칙, 이상 감지 메커니즘 및 개인화된 훈련 모듈. 모델 보정 메커니즘은 비독립적이며 동일한 분포(non-IID) 환경에서 악성 모델 업데이트와 양성 업데이트를 구별하는 데 도움을 줍니다. 맞춤형 집계 규칙은 유사 클라이언트 간의 협업을 촉진하며, 이상 감지 메커니즘을 통해 악성 클라이언트를 빠르게 식별하고 제거합니다. Euclidean norm 기반의 감지 메커니즘이 도입되어 클라이언트의 모델 업데이트 차이에 대한 정밀 분석을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, FedCAP는 여러 비독립적 환경 및 일련의 중독 공격에 대한 강한 견고성을 보이며, 기존의 최첨단( SOTA) FL 방법들과 비교하여 모델 정확도와 견고성 모두에서 높은 성능을 나타냈습니다.



### AERO: Softmax-Only LLMs for Efficient Private Inferenc (https://arxiv.org/abs/2410.13060)
Comments:
          35 pages, 21 figures, and 9 tables. arXiv admin note: text overlap with arXiv:2410.09637

- **What's New**: 이 논문은 암호화된 입력에 대한 직접적인 추론을 가능케 하는 Private Inference (PI)의 효율성을 향상시키기 위해 AERO라는 네 단계 최적화 프레임워크를 제안합니다. 이를 통해 Transformer 기반 언어 모델의 비선형성을 체계적으로 제거하여 플롭(FLOPs) 수를 줄입니다.

- **Technical Details**: AERO 프레임워크는 LayerNorm과 GELU와 같은 비선형성을 제거하여 PI 효율성을 개선합니다. 또한 Softmax-only 아키텍처를 처음으로 제안하며, 이 아키텍처는 significantly 적은 FLOPs로 PI에 적합합니다. 이를 위해 엔트로피 정규화 기법을 통해 Softmax-only 모델의 성능을 향상시킵니다.

- **Performance Highlights**: AERO는 최대 4.23배의 통신량 감소와 1.94배의 지연 시간 감소를 달성합니다. 또한, 기존의 최첨단 모델과 비교하여 AERO의 효과성을 입증하는 벤치마크 결과를 제공합니다.



### Systems with Switching Causal Relations: A Meta-Causal Perspectiv (https://arxiv.org/abs/2410.13054)
Comments:
          19 pages, 3 figures, 4 tables

- **What's New**: 본 논문에서는 기계 학습에서의 인과관계 연구에서 고정된 과정을 기반으로 한 전통적인 접근 방식의 한계를 지적하고, 메타-인과 상태(meta-causal states)라는 개념을 도입하여 변동하는 시스템 동 dynamics를 분석하는 방법을 제안합니다.

- **Technical Details**: 메타-인과 상태는 고전적인 인과 모델을 유사한 질적 행동에 따라 클러스터로 그룹화하고 특정 메커니즘 매개변수화를 통합하는 방법을 제시합니다. 또한, 관찰된 에이전트 행동으로부터 메타-인과 상태를 추론하는 방법과 레이블이 없는 데이터로부터 이 상태를 분리하는 방법을 논의합니다.

- **Performance Highlights**: 메타-인과 모델(MCM)은 특정 시스템 동역학 내에서 질적 차이를 표현하는 데 있어 고전적인 구조적 인과 모델보다 강력하며, 메타-인과 분석을 통해 전통적인 인과 추론과는 다른 근본 원인 기여를 식별할 수 있습니다.



### Supply Chain Network Extraction and Entity Classification Leveraging Large Language Models (https://arxiv.org/abs/2410.13051)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 자연어 처리(NLP) 및 대형 언어 모델(LLM)를 활용하여 비정형 텍스트 데이터를 기반으로 공급망 그래프를 구축하는 새로운 접근 방식을 제안합니다. 특히 토목 공학 산업을 사례 연구로 삼아 LLM이 기업, 프로젝트 등의 숨겨진 관계를 발견할 수 있는 방법을 보여줍니다.

- **Technical Details**: 본 연구는 데이터 수집, 프롬프트 엔지니어링, 그래프 구축, 엔티티 분류의 네 가지 주요 단계로 구성된 방법론을 적용합니다. 데이터 수집은 공개 소스의 뉴스 기사를 통해 이루어지며, 각 기업에 대해 2018년부터 2023년까지 연도별로 최소 10개의 뉴스 기사를 수집하여 총 50개의 원시 텍스트 데이터 포인트를 확보합니다. 이를 통해 각 기업의 활동에 대한 포괄적인 관점을 유지합니다.

- **Performance Highlights**: LLM으로 특정 산업에 맞춰 세부 조정(fine-tuning)을 수행함으로써 엔티티 분류의 정확도가 향상되었으며, 이는 산업별 공급망 분석의 잠재력을 강조합니다. 본 연구는 LLM을 통해 공급망 네트워크 모델링의 자동화를 가능하게 한 첫 번째 사례로, 공급망 동학에 대한 깊이 있는 통찰력을 제공합니다.



### FedGTST: Boosting Global Transferability of Federated Models via Statistics Tuning (https://arxiv.org/abs/2410.13045)
- **What's New**: 본 논문에서는 기존의 Federated Learning (FL) 방법들이 해결하지 못한 문제들을 다루는 새로운 접근법인 Federated Global Transferability via Statistics Tuning (FedGTST)를 제안합니다. FL의 여러 도전 과제를 해결하면서 전세계적으로 전이 가능성을 높이는 방법론을 소개합니다.

- **Technical Details**: FedGTST는 클라이언트 간의 Jacobian (그라디언트) 노르므를 활용한 클라이언트-서버 교환 프로토콜과, 서버에서 클라이언트 간의 평균 Jacobian 노르므를 높이는 지역 정규화 기법을 통해 보다 효과적인 전이 가능성을 도모합니다. 이는 전이 실패를 줄이고 목표 손실(target loss)을 보다 정교하게 제어할 수 있도록 돕습니다.

- **Performance Highlights**: FedGTST는 MNIST에서 MNIST-M, CIFAR10에서 SVHN 데이터셋을 포함한 다양한 실험에서 FedSR 및 FedIIR과 같은 기존 방법보다 10%의 성능 향상을 나타냈습니다. 특히, LeNet 모델을 사용할 경우, FedGTST는 FedSR 대비 9.8%, FedIIR 대비 7.6%의 더 높은 정확도를 기록했습니다.



### Sample Compression Scheme Reductions (https://arxiv.org/abs/2410.13012)
- **What's New**: 이 논문에서는 다중 클래스 분류, 회귀, 적대적으로 강한 학습 환경에서의 샘플 압축 기법을 이진 샘플 압축 기법으로 축소하는 새로운 방법을 제시합니다.

- **Technical Details**: 이진 클래스의 압축 스킴이 'majority-vote' 또는 'stable compression scheme'일 경우, 다중 클래스 압축 스킴은 크기 O(f(d_G))를 가지게 됩니다. 여기서 d_G는 그래프 차원입니다. 일반 이진 압축 스킴에 대해선 크기가 O(f(d_G)log|Y|)인 압축을 얻습니다. 회귀 문제에 대해서도 유사한 결과를 제시하고, 압축 크기가 O(f(d_P))로 표현 가능한 압축 스킴의 존재를 설명합니다. 여기서 d_P는 유사 차원(pseudo-dimension)입니다.

- **Performance Highlights**: 샘플 압축 추정 정리가 해결되면(미해결 상태임) 이러한 결과들은 다른 설정으로의 증명을 즉시 확장할 수 있는 주요 의미를 가집니다. 또한, 적대적으로 강한 학습과 관련된 유사한 결과를 세우고, 모든 개념 클래스가 유한 크기의 압축 스킴을 가지고 있다고 보장할 수 없는 사례도 보여줍니다.



### Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images (https://arxiv.org/abs/2410.13010)
Comments:
          Published in the 3rd Workshop on New Frontiers in Adversarial Machine Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

- **What's New**: 기존의 적대적 공격이 주로 단일 모드에 초점을 맞추었던 반면, 본 연구에서는 CLIP과 같은 대규모 멀티 모달 모델(LMM)이 가지는 새로운 취약점에 주목합니다. 새로운 ‘Hiding-in-Plain-Sight (HiPS)’ 공격 기법을 통해 모델 예측을 미세하게 수정함으로써, 타겟 객체가 존재하지 않는 것처럼 보이게 하는 방법을 제안합니다.

- **Technical Details**: HiPS 공격은 두 가지 변형으로 소개됩니다: HiPS-cls는 클래스 레이블 정보를 활용하여 공격을 생성하며, HiPS-cap은 원본 이미지 캡션과 타겟 캡션을 사용하여 공격을 설계합니다. 이러한 공격 기법은 CLIP-Cap과 같은 이미지 캡셔닝 모델로 효과적으로 전이될 수 있습니다.

- **Performance Highlights**: HiPS 공격은 타겟 객체가 이미지 캡션에서 효과적으로 제거되도록 설계되었으며, 여러 평가 지표를 통해 성능을 검증합니다. 제안된 공격이 하위 모델에서 어떻게 작동하는지를 보여주며, 적대적 공격의 새로운 기준을 설정합니다.



### LLM Chain Ensembles for Scalable and Accurate Data Annotation (https://arxiv.org/abs/2410.13006)
- **What's New**: 이 연구는 복수의 대형 언어 모델(LLM)을 연속으로 연결하여 LLM 체인 앙상블 방법론을 제시합니다. 이 방법은 분류의 불확실성을 기반으로 데이터 하위 집합을 후속 모델로 라우팅합니다.

- **Technical Details**: LLM 체인 앙상블 방법은 각 LLM의 불확실성 측정치를 이용해 데이터를 라우팅하여, 각 LLM이 가장 자신 있는 데이터 포인트를 처리하게 하고, 더 복잡한 경우는 보다 강력한 모델로 전달합니다. 이 구조를 통해 첫 모델이 처리한 데이터과정에서 각 LLM의 예측과 신뢰 점수를 집계하여 최종 레이블을 도출합니다.

- **Performance Highlights**: 체인 앙상블 방법은 체인 내 최상의 단일 모델 성능을 초과하며, 아울러 최대 90배의 비용 절감 또한 보고했습니다. 이 방법은 대규모 데이터 주석의 실현 가능성과 효율성을 강조합니다.



### SSET: Swapping-Sliding Explanation for Time Series Classifiers in Affect Detection (https://arxiv.org/abs/2410.12996)
- **What's New**: 이번 연구에서는 다변량 시계열 분류기를 위한 새로운 설명 방법인 SSET(Swapping-Sliding Decision Explanation)를 제안합니다. 이 방법은 예측 점수에서 중요한 하락을 초래하는 두 가지 주요 단계를 통해 설명을 생성합니다: 스와핑(swap) 단계와 슬라이딩(slide) 단계입니다.

- **Technical Details**: SSET는 두 단계로 구성됩니다. 첫 번째 단계에서는 주어진 인스턴스와 가까운 훈련 데이터로부터 중요한 변수를 찾아내기 위해 스와핑을 사용합니다. 두 번째 단계에서는 각 시간 단계에서 선택된 훈련 데이터에 대해 윈도우를 슬라이드하여 중요한 하위 시퀀스를 탐색합니다.

- **Performance Highlights**: SSET는 WESAD 및 MAHNOB-HCI의 두 실제 생리학적 시계열 데이터셋에서 CN-Waterfall 분류기를 이용해 평가되었으며, 기존 모델들(Dynamask, integrated gradients, LIME)보다 우수한 성능을 보였습니다.



### Double-Bayesian Learning (https://arxiv.org/abs/2410.12984)
Comments:
          14 pages, 5 figures, draft

- **What's New**: 이 논문은 의사결정이 단일이 아닌 두 개의 Bayesian 결정으로 구성된다는 가정을 제시하며, 이를 통해 의사결정 과정에서 본질적인 불확실성을 강조하고 설명 가능성을 통합하는 방법을 탐구하고 있습니다.

- **Technical Details**: 직관적으로, 제안된 접근법은 Bayesian 학습을 불확실성을 측정하는 로그 함수의 기초를 찾는 것과 동일한 것으로 이해합니다. 따라서 이러한 접근법은 학습률(learning rate)과 모멘텀 가중치(momentum weight)를 신경망을 훈련하는 데 사용되는 문헌의 값과 유사하게 설정하도록 제안합니다.

- **Performance Highlights**: Bayes 정리에 기반한 최적 분류기를 탐구하며, 두 개의 결정 과정이 상호 의존적으로 작용하여 불확실성을 수반한다는 점에서, 이론적으로는 신경망의 훈련에 대한 하이퍼파라미터의 영향을 논의하고 있습니다.



### Reinforcement Learning with Euclidean Data Augmentation for State-Based Continuous Contro (https://arxiv.org/abs/2410.12983)
- **What's New**: 이 논문은 강화 학습(RL) 에이전트의 데이터 효율성을 높이기 위해 유클리드 대칭(Euclidean symmetries) 기반의 데이터 증강(data augmentation) 접근법을 제안합니다. 기존의 방법들이 이미지 기반 데이터 증강에 중점을 두었던 반면, 본 연구는 상태 기반(control state) 데이터 증강에 중점을 둡니다.

- **Technical Details**: 유클리드 데이터 증강은 물리적으로 관찰 가능한 위치(position)와 속도(velocity)와 같은 상태 기반 피처를 활용하여 이루어집니다. 반면 기존의 대칭 기반 변환은 조인트(joint) 구성으로만 이루어져 데이터 증강이 효과적이지 않았습니다. 본 연구에서는 팔다리의 구성(configuration)을 새로운 상태 표현으로 사용하여 더 많은 데이터를 생성하고, 임의의 회전(rotation)이나 이동(translation) 변환을 통해 데이터를 증강합니다.

- **Performance Highlights**: 개별 상태 표현을 사용했을 때, DeepMind Control Suite의 대부분의 작업에 대한 성능이 향상되었으며, 유클리드 데이터 증강 추가 시 거의 모든 작업에서 최적의 성능을 달성했습니다. 예를 들어, Humanoid_run 작업에서 표준 DDPG는 100 이하의 보상을 달성한 반면, 본 방법은 5M 타임스텝 후에 150 이상의 보상을 달성했습니다.



### Flash Inference: Near Linear Time Inference for Long Convolution Sequence Models and Beyond (https://arxiv.org/abs/2410.12982)
Comments:
          15 pages, 9 figures, 5 algorithms

- **What's New**: 이 논문에서는 Long Convolution Sequence Models (LCSMs), 특히 Hyena 모델의 정확한 추론(inference) 속도를 O(L log² L)로 증가시키는 방법을 제안합니다. 또한, 이러한 속도 향상이 가능한 주요 속성들을 정의하고, 이러한 속성을 활용하는 일반적인 프레임워크를 제안합니다.

- **Technical Details**: 제안된 접근 방식은 relaxed polynomial interpolation에 대한 이전 연구를 바탕으로 하며, 메모리 이동을 줄이고 계산을 공유하는 tiling 기법을 활용합니다. 이 방법은 position-mixing 부분의 아키텍처에서 거의 완전한 병렬화(parallelization)를 허용합니다.

- **Performance Highlights**: Hyena 모델의 실험적 구현을 통해, 표준 추론에 비해 최대 1.6배의 엔드 투 엔드(end-to-end) 시간 효율성을 개선하였고, position-mixing 부분에서는 최대 50배의 성능 향상을 달성했습니다.



### A Note on Shumailov et al. (2024): `AI Models Collapse When Trained on Recursively Generated Data' (https://arxiv.org/abs/2410.12954)
Comments:
          Comment on this https URL

- **What's New**: Shumailov et al. (2024)의 연구에 따르면, 합성 데이터에 반복적으로 훈련된 생성 모델이 모델 붕괴(model collapse)를 일으킬 수 있다는 사실이 밝혀졌습니다. 이 연구는 현재 모델들이 기존 데이터의 활용 가능성을 거의 소진한 가운데 이루어져 큰 주목을 받고 있습니다.

- **Technical Details**: 연구는 데이터에 대한 적합(distribution fitting) 및 반복적인 샘플링을 통해 모델 붕괴의 원인을 조사합니다. Kernel Density Estimation (KDE)와 KL divergence 및 Wasserstein distance (WSD)와 같은 거리 메트릭을 사용하여 결과를 분석했습니다.

- **Performance Highlights**: 결과는 최종 분포(final distribution)가 붕괴되며, 이 과정에서 샘플링과 적합이 반복될수록 원래 데이터의 구조가 점차 유실되고 더 균일한 분포로 수렴한다는 것을 보여줍니다. 연구는 생성 모델이 데이터의 분포를 정확히 재현하는 데 한계가 있음을 강조하며, 향후 연구의 필요성을 제기합니다.



### Syn2Real Domain Generalization for Underwater Mine-like Object Detection Using Side-Scan Sonar (https://arxiv.org/abs/2410.12953)
Comments:
          7 pages, 4 figures and 3 tables

- **What's New**: 논문에서는 수중 지뢰 탐지에 대한 Syn2Real (Synthetic to Real) 도메인 일반화 접근 방식을 제안합니다. 이 방법은 DDPM 및 DDIM 모델을 사용하여 생성한 합성 데이터를 통해 실제 환경 샘플을 효과적으로 보강할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 딥러닝 기반의 자동 목표 인식(ATR) 기술을 사용하여 수중 지뢰를 탐지하는 과정을 다룹니다. 특히, 이 논문에서는 DCGAN 및 확산 모델과 같은 합성 데이터 생성 모델을 비교 분석하였으며, 이러한 모델의 하이퍼파라미터 튜닝을 통해 효과적인 결과를 얻었습니다.

- **Performance Highlights**: Mask-RCNN 모델을 합성 데이터와 원본 데이터 조합으로 학습시킨 결과, 평균 정밀도(Average Precision, AP)가 약 60% 증가했습니다. 이는 수중 지뢰 탐지 작업에서 Syn2Real 도메인 일반화의 잠재력을 강조하는 결과입니다.



### Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization (https://arxiv.org/abs/2410.12949)
Comments:
          20 pages, 19 figures, 7 tables

- **What's New**: 본 연구에서는 대형 언어 모델(LLM)에서 지식 편집 및 비학습(unlearning) 방법의 향상을 위한 기계적 해석 가능성(mechanistic interpretability)의 역할을 조사합니다. 특히, 출력 보존(output preserving) 기반의 구성 요소 로컬라이제이션 방식과 예측 가능한 중간 상태를 이용한 고수준 메커니즘 발견 방식 간의 차이를 강조합니다.

- **Technical Details**: 연구에서는 사실 회상(factual recall)을 위한 로컬라이제이션을 FLU(fact lookup) 메커니즘에 기반하여 진행하며, 이를 통해 이전의 방법들보다 더 견고한 편집 및 비학습을 구현합니다. 다양한 입력/출력 형식에서의 견고함이 향상되었으며, 원하지 않는 정보를 다시 학습하는 것을 방지하면서 부작용(side effects)도 감소합니다.

- **Performance Highlights**: Gemma-7B 모델을 사용하여 다양한 데이터셋에서 FLU 메커니즘 기반의 편집 및 비학습이 기존 방법들보다 더 높은 견고성과 일반화 능력을 나타낸다는 것을 확인하였습니다. 특히, 스포츠 사실 데이터셋과 CounterFact 데이터셋에서 실험을 수행하여 이런 결과를 입증하였습니다.



### Multi-modal graph neural networks for localized off-grid weather forecasting (https://arxiv.org/abs/2410.12938)
- **What's New**: 본 연구는 이질적인 그래프 신경망 (GNN) 기반의 신모델을 통해 격자 기반 날씨 예측을 지역적 관심 지점으로 다운스케일링 하는 방법을 제안합니다. 이를 통해 글로벌 규모의 날씨 모델과 지역적 예측 간의 격차를 해소할 수 있습니다.

- **Technical Details**: 다양한 유형의 노드를 가진 이질적인 그래프를 구성하고, 메시지 패싱을 통해 예측 위치의 노드가 이웃 노드로부터 정보를 집계합니다. 이 다중 모드 GNN은 지역 역사적 날씨 관측치(예: 바람, 온도)를 활용하여 격자 기반 날씨 예측을 수정합니다. 데이터는 Northeastern United States의 2019-2023년 날짜를 포함하여 수집되었습니다.

- **Performance Highlights**: 모델은 다양한 오프그리드 예측 방법들과 비교하여 우수한 성능을 나타내었고, 가장 잘 작동하는 MLP 모델에 비해 MSE를 55.22% 감소시켰습니다. ERA5 이전 방식과 비교할 때도 MSE를 82.55%까지 감소시키며, GNN 모델이 가져오는 효과적인 지역적 날씨 패턴 예측의 가능성을 입증하였습니다.



### SoK: On Finding Common Ground in Loss Landscapes Using Deep Model Merging Techniques (https://arxiv.org/abs/2410.12927)
- **What's New**: 이번 연구에서는 신경망의 해석 가능성을 향상시키고, 모델 병합(model merging)이라는 관련 분야의 문헌을 조사하여 신뢰할 수 있는 딥러닝 모델을 개발하기 위한 새로운 통찰력을 제시합니다.

- **Technical Details**: 모델 병합은 여러 신경망의 파라미터를 결합하여 성능이 뛰어난 단일 예측 모델을 만들어내는 기술입니다. 본 연구에서는 손실 경관(loss landscape geometry) 관점에서 모델 병합 기술을 분석하며, 이를 통해 해석 가능성, 보안 및 모델 훈련에 대한 새로운 이해를 제공합니다.

- **Performance Highlights**: 모델 병합 기술은 다양한 신경망을 효율적으로 조합하여 훨씬 더 우수한 성능의 모델을 생성할 수 있는 잠재력을 지니고 있으며, 모델 해석과 보안 분야에서의 의미 있는 연결점을 발견하였습니다.



### Fair Clustering for Data Summarization: Improved Approximation Algorithms and Complexity Insights (https://arxiv.org/abs/2410.12913)
- **What's New**: 이 연구는 공정한 데이터 요약(fair data summarization) 문제를 다루며, 기존의 $k$-supplier 문제를 공정성을 고려하여 확장한 점이 새롭습니다.

- **Technical Details**: 연구에서는 공정한 $k$-supplier 문제를 정의합니다. 이 문제는 데이터가 여러 그룹으로 구성되고, 각 그룹에서 최소한의 중심(center)을 선택해야 하며, $k$-supplier 목표를 최소화해야 합니다. 두 가지 문제 변형에 대해 각각 알고리즘을 제시하며, 비지지(disjoint) 그룹에 대해 다항식(polynomial) 시간 복잡도를 보이고, 겹치는(overlapping) 그룹에 대해서는 고정-매개변수 고찰(fixed-parameter tractable) 알고리즘을 제공합니다.

- **Performance Highlights**: 비지지 그룹에 대한 알고리즘은 시간 복잡도가 다항식으로 실행되며, 겹치는 그룹에 대한 알고리즘은 중심과 그룹의 수에만 의존하는 지수(exponential) 실행 시간을 가집니다. 제안된 알고리즘은 기존의 $5$보다 개선된 $3$-근사화(approximation) 알고리즘을 제공하며, 이 근사화 계수는 이론적인 하한(lower bound)과 일치합니다.



### Generative Reward Models (https://arxiv.org/abs/2410.12832)
- **What's New**: 이번 연구에서는 Reinforcement Learning from Human Feedback (RLHF)와 Reinforcement Learning from AI Feedback (RLAIF)을 통합한 새로운 방법론을 제안합니다. 특히 GenRM이라는 반복적인 알고리즘을 통해 LLM이 스스로 생성한 reasoning traces를 기반으로 synthetic preference labels를 제작하여, 이를 인간의 선호와 일치하도록 조정합니다.

- **Technical Details**: GenRM은 LLM을 평가자로 활용하여 사용자 선호 데이터셋을 기반으로 LLM을 조정하는 STaR-like 방법론을 채택합니다. 이 과정에서 LLM은 효율적인 보상 모델 역할을 학습하며, 훈련된 모델은 Bradley-Terry 보상 모델과 유사한 성능을 발휘합니다.

- **Performance Highlights**: GenRM은 in-distribution 작업에서는 Bradley-Terry 모델과 유사한 정확도를 보이며, out-of-distribution 작업에서는 10-45%의 성과 향상을 이뤄 냅니다. 더불어 GenRM은 LLM을 단독 평가자로 사용하는 경우보다 in-distribution 작업에서 9-31%, out-of-distribution 작업에서 2-6% 더 높은 성능을 발휘합니다.



### Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens (https://arxiv.org/abs/2410.13863)
Comments:
          Tech report

- **What's New**: 이 연구는 text-to-image generation의 맥락에서 자가 회귀 모델(autoregressive models) 스케일링 문제를 조사합니다. 특히 이 모델들이 사용하는 token이 discrete인지 continuous인지, 그리고 token이 BERT 및 GPT와 유사한 transformer 아키텍처에서 무작위(random)로 생성되는지 또는 고정(raster) 순서로 생성되는지를 중심으로 성능을 비교합니다.

- **Technical Details**: 연구는 VQ(vector quantization) 방식이 이미지 생성 성능에 미치는 영향과 token 생성 순서가 시각적 품질에 미치는 영향을 분석합니다. Fluid라는 새로운 random-order autoregressive 모델을 continuous token으로 학습시켜, 10.5B 모델인 Fluid가 MS-COCO 30K에서 제로샷 FID(zero-shot FID) 6.16을 기록했습니다.

- **Performance Highlights**: Fluid 모델은 FID와 GenEval 점수에서 다른 모델에 비해 우수한 성능을 보여주며, 특히 무작위 순서 모델이 raster 순서 모델에 비해 GenEval 점수에서 현저히 더 나은 결과를 보였습니다. 연구 결과는 비전 모델과 언어 모델 간의 스케일링 격차를 줄여주는 데 기여할 것으로 기대됩니다.



### Retrospective Learning from Interactions (https://arxiv.org/abs/2410.13852)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)과 사용자 간의 다중 턴(multi-turn) 상호작용에서 발생하는 암묵적인 피드백 신호를 활용하여 모델 향상을 꾀하는 새로운 방법인 ReSpect를 소개합니다.

- **Technical Details**: ReSpect는 과거 상호작용에서 발생했던 암묵적 신호를 통해 학습하는 방법입니다. 이 방법은 사용자와의 상호작용 후, 모델이 자신의 과거 행동을 회고하여 피드백을 해석하고 재훈련하는 과정을 포함합니다. 이를 통해 사용자는 모델의 수행 여부를 신호로 전달하며, 이 신호는 자연어의 한정된 하위 공간에 위치하여 LLM이 이를 쉽게 감지할 수 있게 합니다.

- **Performance Highlights**: ReSpect를 적용한 결과, IDEFICS2-8B 모델의 작업 완료율이 31%에서 82%로 향상되었습니다. 이 과정에서는 외부 주석 없이도, 과거 상호작용을 통해 직접적으로 스스로 피드백을 해석하고 개선하는 능력을 보여주었습니다.



### From Gradient Clipping to Normalization for Heavy Tailed SGD (https://arxiv.org/abs/2410.13849)
- **What's New**: 최근 머신러닝 애플리케이션에서 중대한 문제를 해결하는 새로운 방법이 소개되었습니다. Non-convex gradient clipping에 대한 기존 이론적 이해의 한계를 극복하기 위해, Normalized SGD (NSGD)의 수렴성을 연구하여 새로운 샘플 복잡도와 더 나은 성능을 증명했습니다.

- **Technical Details**: 이 연구에서는 NSGD의 파라미터 없는 샘플 복잡도를 $	ext{O}(	ilde{	ext{O}}(	ext{ε}^{-rac{2p}{p-1}}))$로 설정했습니다. 또한 모든 문제 파라미터가 알려져 있는 경우, 복잡도를 개선하여 $	ext{O}(	ilde{	ext{O}}(	ext{ε}^{-rac{3p-2}{p-1}}))$로 설정할 수 있음을 보여주었습니다. 마지막으로, 실망 확률에 대한 경미한 로그 의존성을 가진 NSGD의 높은 확률 수렴을 확립했습니다.

- **Performance Highlights**: 이 연구는 기존의 알고리즘 샘플 복잡도를 개선하고, 높은 확률 수렴을 달성할 수 있는 대안 메커니즘을 제공함으로써 heavy-tailed noise 하의 gradient clipping 연구에 기여하고 있습니다.



### SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction (https://arxiv.org/abs/2410.13846)
- **What's New**: SimLayerKV라는 새로운 방법을 소개하며, 긴 맥락을 처리하는 대형 언어 모델에서의 KV cache의 비효율성을 줄입니다. 이 방법은 "lazy" layer를 식별하고 이들에 대한 KV cache의 중복을 줄여 효율성을 증대시킵니다.

- **Technical Details**: SimLayerKV는 특정 존재 layers의 KV cache를 선택적으로 제거하여 inter-layer KV cache 중복을 감소시키는 방법입니다. 이는 lazy layers의 주의 할당 패턴을 분석하여 가능하며, non-lazy layers의 KV cache는 유지합니다. 코드 구현은 단 7줄로 가능하며, 훈련 과정이 필요 없습니다.

- **Performance Highlights**: SimLayerKV는 LongBench 벤치마크에서 3개의 대표적인 LLM (LLaMA2-7B, LLaMA3-8B, Mistral-7B)에서 KV cache 비압축 비율 5배를 달성하며, 4-bit quantization을 함께 사용할 때 오직 1.2%의 성능 저하만 보여줍니다.



### Steering Your Generalists: Improving Robotic Foundation Models via Value Guidanc (https://arxiv.org/abs/2410.13816)
Comments:
          Conference on Robot Learning (CoRL) 2024. Project Page: this https URL

- **What's New**: 이 논문은 Value-Guided Policy Steering (V-GPS)이라는 일반적인 접근 방식을 제안하여, 재정렬된 행동 제안을 통해 일반 로봇 정책의 성능을 개선하는 방법을 소개합니다. 이 방법은 오프라인 강화 학습(offline RL)을 이용해 학습된 가치 함수(value function)를 사용하며, 여러 로봇 플랫폼에서 일관된 성능 개선을 보여줍니다.

- **Technical Details**: V-GPS는 사전 학습된 가치 함수를 사용하여 로봇 정책을 배치 시점에서 조정합니다. 이 접근 방식은 정책의 가중치에 접근할 필요 없이 다양한 일반 정책과 호환됩니다. 실험에는 5개의 최첨단 공개 일반 정책이 포함되었으며, 총 12개의 과제에서 평가되었습니다. 그 결과, 실제 조작 작업에서 +82%의 성능 개선을 달성했습니다.

- **Performance Highlights**: 이 연구는 고품질 데이터에서 학습된 가치 함수를 통해 다소 불안정한 일반 로봇 정책의 성능을 일관되게 향상시킴으로써, 실제 환경에서의 조작 작업에서 뛰어난 성과를 보였습니다. 이는 다섯 개의 다양한 정책이 서로 다른 환경에서도 성공적으로 작동할 수 있음을 보여줍니다.



### Private Counterfactual Retrieva (https://arxiv.org/abs/2410.13812)
- **What's New**: 본 논문에서는 고위험(high-stakes) 애플리케이션에서 블랙박스 기계 학습 모델을 사용하는 경우의 투명성과 설명 가능성의 필요성을 강조하며, 개인 정보 보호를 보장하는 다양한 카운터팩추얼(counterfactual) 설명 제공 방안을 제안합니다.

- **Technical Details**: 본 연구에서는 개인 정보 검색(private information retrieval, PIR) 기법을 바탕으로 한 개인 카운터팩추얼 검색(private counterfactual retrieval, PCR) 문제를 도입합니다. 이를 통해 사용자가 자신의 입력 피처 벡터를 공개하지 않고도 카운터팩추얼 설명을 요청할 수 있도록 설계하였습니다. 제안된 PCR 방법론은 ℓ2 거리(metric) 기반으로 작동하며, 정보 이론적 측면에서 완벽한 개인 정보 보호를 달성합니다.

- **Performance Highlights**: 실험을 통해 제안된 모델이 실제 데이터셋에서 정확도와 개인 정보 보호 간의 트레이드오프를 이해하는 데 성공했음을 입증하였습니다. 두 가지 PCR 스킴인 Diff-PCR과 Mask-PCR을 통해 데이터베이스 보호 측면에서도 향상된 성능을 보여줍니다.



### Discrete distributions are learnable from metastable samples (https://arxiv.org/abs/2410.13800)
Comments:
          Preliminary version, 26 pages

- **What's New**: 이 논문은 Markov chain 샘플러가 메타안정 상태에서 샘플링할 때, 그 분포가 진정한 상태 분포와는 다르게 행동함에도 불구하고, 학습 알고리즘을 통해 진정한 모델을 유도할 수 있다는 점에 주목합니다.

- **Technical Details**: 우리는 reversible Markov chain의 메타안정 분포를 조사하고, 이들이 특정 조건을 만족하는 경우 단일 변수 조건부가 진정한 분포와 평균적으로 매우 밀접하다는 것을 보여줍니다. 구체적으로, Pseudo-Likelihood 방법을 사용하여 메타안정 상태로부터 얻은 i.i.d 샘플로부터도 균형 분포를 효과적으로 학습할 수 있음을 입증합니다.

- **Performance Highlights**: 특정 이진 쌍방향 무방향 그래프 모델의 경우, 메타안정 상태에서 나오는 데이터로 에너지 함수의 매개변수를 학습하고 모델 구조를 복원할 수 있음을 추가적으로 설명합니다.



### Machine-Learning Analysis of Radiative Decays to Dark Matter at the LHC (https://arxiv.org/abs/2410.13799)
Comments:
          32 pages, 9 figures, 3 tables, 4 appendices

- **What's New**: 이번 연구에서는 머신 러닝(Machine Learning) 기술을 활용하여 초대칭 프레임워크 내에서 약하게 상호작용하는 물질 입자(WIMP)의 방사선 붕괴를 탐구하고 있습니다. 이는 다크 매터(Dark Matter) 후보와의 상호작용을 포함하며, 초대칭 모델의 두 번째 가벼운 중성소립자와의 동시 멸균(co-annihilation)을 통해 관측된 우주론적 다크 매터 밀도를 설명하려는 노력을 다룹니다.

- **Technical Details**: 연구팀은 LHC에서의 방사선 붕괴 입자 탐색을 위해 컷 기반(cut-based) 및 머신 러닝(Machine Learning) 방법을 적용하여 이러한 입자의 발견 가능성을 평가합니다. 방사선 붕괴는 중성소립자가 포톤과 가장 가벼운 중성소립자로 붕괴되는 현상을 의미하며, 표준 모델 기반의 강력한 배경(signal backgrounds)으로 인해 탐지가 어렵습니다. 연구 유형은 lhc의 에너지 센터 오브 마스(sqrt{s})와 100 fb-1의 총 통합 루미노시티(total integrated luminosity)를 기준으로 진행됩니다.

- **Performance Highlights**: 머신 러닝 방법을 통해 방사선 붕괴 탐색의 발견 잠재력을 극대화할 수 있으며, 특히 ML 기반 Likelihood 모델을 활용한 분석에서 유의미한 결과를 도출할 수 있습니다. 이는 기존의 컷 기반 방법보다 우수한 성능을 보여주며, 방사선 붕괴 중성소립자의 발견 가능성을 높일 것입니다.



### Learning Graph Quantized Tokenizers for Transformers (https://arxiv.org/abs/2410.13798)
- **What's New**: 이번 논문에서는 Graph Quantized Tokenizer (GQT)를 도입하여 그래프의 토큰화 과정을 개선했습니다. GQT는 멀티태스킹 그래프 자기 지도 학습(multi-task graph self-supervised learning)을 활용하여 토크나이저 훈련과 트랜스포머 훈련을 분리함으로써 더 강력하고 일반화 가능한 토큰을 생성합니다.

- **Technical Details**: GQT는 Residual Vector Quantization (RVQ) 기법을 통해 계층적인 이산 토큰을 학습하여 메모리 요구 사항을 크게 줄이고 일반화 능력을 향상시킵니다. 이 방법은 의미적 엣지와 랜덤 워크를 결합하여 트랜스포머가 장거리 상호작용에 접근할 수 있도록 합니다.

- **Performance Highlights**: GQT를 트랜스포머 인코더와 결합하여 18개 벤치마크 중 16개에서 최첨단 성능을 달성하였으며, 특히 대규모 동질적 및 이질적 데이터셋에서 성능이 뛰어났습니다. 이는 매우 감소된 메모리 풋프린트를 갖춘 임베딩을 통해 달성되었습니다.



### Optimal Quantization for Matrix Multiplication (https://arxiv.org/abs/2410.13780)
- **What's New**: 본 연구는 대규모 매트릭스의 lossy compression (양자화) 기법을 통해 매트릭스 곱셈을 가속화하기 위한 새로운 알고리즘을 제안합니다. 이 접근법은 전통적인 벡터 양자화와 다르게, 매트릭스 자체가 아니라 매트릭스 곱셈의 근사를 목표로 합니다.

- **Technical Details**: 이 논문은 iid Gaussian 아이템을 가진 매트릭스의 평균 제곱 오차에 대한 비비대칭 하한을 제공하며, 특정한 프레임워크에서 Frobenius norms를 사용하여 매트릭스 A, B의 압축과 동시에 근사 오차를 보장하는 보편적인 양자기를 제안합니다. 이는 깊은 신경망(Deep Neural Networks)과 대규모 언어 모델(Large Language Models)에서 메모리 대역폭의 병목 현상을 해결하기 위한 중요성을 강조합니다.

- **Performance Highlights**: 제안된 양자기는 최적 성능에 근접한 결과를 실현하며, 정보 이론적으로 iid Gaussian 매트릭스의 매트릭스 곱셈에 대한 rate-distortion function을 도출합니다.



### The Mystery of the Pathological Path-star Task for Language Models (https://arxiv.org/abs/2410.13779)
Comments:
          EMNLP 2024 Main

- **What's New**: 최근에 도입된 path-star task는 언어모델(Language Models)의 한계점을 보여주기 위해 설계된 최소한의 작업입니다. 이 태스크는 여러 팔이 단일 시작 노드에서 방사하는 path-star 그래프를 포함하며, 각 노드는 유일합니다.

- **Technical Details**: path-star 그래프는 하나의 중앙 시작 노드와 여러 개의 방사형 팔을 포함하고 있으며, 각 팔은 고유한 타겟 노드에서 끝납니다. 주어진 시작 노드와 타겟 노드에 대해 해당 타겟 노드를 포함하는 팔을 생성하는 것이 이 태스크의 목표입니다. 이 작업이 언어모델에겐 어렵다는 가설은 teacher-forcing의 결핍과 다음 토큰 예측 패러다임에서 기인한다고 제시됩니다.

- **Performance Highlights**: 여러 모델 유형에서 결과를 개선하기 위한 구조화된 샘플을 사용하는 정규화 방법이 도입되어, encoder-only 모델이 지속적으로 태스크를 해결할 수 있는 설정을 발견했습니다. 이 연구 결과는 path-star task가 이론적으로 풀 수 있다는 RASP 증명을 제공합니다.



### Probing the Latent Hierarchical Structure of Data via Diffusion Models (https://arxiv.org/abs/2410.13770)
Comments:
          11 pages, 6 figures

- **What's New**: 이번 연구에서는 확산 기반 모델(diffusion-based models)에서의 전방-후방 실험(forward-backward experiments)이 데이터의 잠재 구조(latent structure)를 탐색하는 유망한 도구임을 보여줍니다. 우리는 간단한 계층 모델(hierarchical models)에서 데이터의 변화가 상관된 청크(chunk)로 일어난다는 예측을 하였고, 이 예측이 텍스트와 이미지 데이터셋에서 확인되었습니다.

- **Technical Details**: 고차원 데이터(high-dimensional data)의 구조적 특성(structural properties)을 정량적으로 측정하는 방법이 부족한 가운데, 이번 연구에서는 생성적 제거 확산 모델(generative denoising diffusion probabilistic models), 특히 전방-후방 프로토콜을 활용하여 이러한 문제를 해결했습니다. 우리는 계층적으로 구조화된 데이터의 생성을 위한 확률적 문맥 자유 문법(probabilistic context-free grammars)을 사용하였고, 이 모델에서의 변화의 길이 척도(length scale)를 도출하였습니다.

- **Performance Highlights**: 실험 결과, 우리의 이론적 예측이 실제 데이터에 잘 맞아 떨어진다는 것을 보여주었으며, 특히 마스크드 확산 언어 모델(Masked Denoising Diffusion Language Models)과 비전 제거 확산 확률 모델(Denoising Diffusion Probabilistic Models)에서의 성능을 입증했습니다. 이러한 결과는 잠재 변수의 변화가 데이터에 어떻게 영향을 미치는지를 명확히 드러냅니다.



### CLIMB: Language-Guided Continual Learning for Task Planning with Iterative Model Building (https://arxiv.org/abs/2410.13756)
Comments:
          6 pages, 6 figures

- **What's New**: CLIMB는 지속적인 학습을 통해 로봇 작업 계획을 지원하는 새로운 프레임워크로, 자연어 설명을 바탕으로 도메인 모델을 생성하고 비직관적인 술어를 학습하여 향후 문제에 활용할 수 있습니다.

- **Technical Details**: CLIMB는 하이브리드 신경-심볼릭 (neuro-symbolic) 계획 시스템으로, 기초 모델과 전통적인 심볼릭 계획자를 결합하여 중복 학습 없이 과거의 문제를 해결할 수 있는 능력을 보유하고 있습니다. 이 시스템은 PDDL 모델을 점진적으로 구축하며, 작업을 수행하면서 환경의 원인 구조를 즉각 반영합니다.

- **Performance Highlights**: CLIMB는 예비 성능 시험에서 기존 방법과 비교하여 일반 계획 환경에서 성능 향상을 입증했습니다. BlocksWorld++ 도메인을 통해 점진적 논리 세계 모델 구축 능력을 평가했으며, 실험 결과 CLIMB의 향상된 성능을 확인할 수 있었습니다.



### MixEval-X: Any-to-Any Evaluations from Real-World Data Mixtures (https://arxiv.org/abs/2410.13754)
- **What's New**: 이 논문은 다양한 형태의 입력 및 출력을 지원하는 새로운 벤치마크인 MixEval-X를 소개하여, AI 모델의 평가 방식을 최적화하고 표준화하는 데 중점을 두고 있습니다. 이를 통해 실제 작업 배포에 맞는 평가가 가능하게 됩니다.

- **Technical Details**: MixEval-X는 any-to-any (모든 입력에 대해 가능한 모든 출력) 형식의 벤치마크로, 다양한 modality (양식) 간의 평가 일관성을 높이기 위해 multi-modal (다중 양식) 벤치마크 혼합 및 adaptation-rectification (적응-정정) 파이프라인을 제안합니다. 이 방법은 평가가 실제 사용할 수 있는 사례에 잘 일반화되도록 합니다.

- **Performance Highlights**: 종합적인 메타 평가 결과, MixEval-X는 벤치마크 샘플과 실제 작업 배포 간의 효과적인 정렬을 보여주었으며, 모델 순위는 크라우드 소싱된 실제 평가와 강한 상관관계를 나타냅니다 (상관 계수 0.98까지). 또한, 기존 모델 및 조직을 재순위화할 수 있는 포괄적인 리더보드를 제공하여 다중 양식 평가에 대한 이해를 높이고 향후 연구에 대한 통찰을 제공합니다.



### Improved Convergence Rate for Diffusion Probabilistic Models (https://arxiv.org/abs/2410.13738)
Comments:
          20 pages

- **What's New**: 본 논문에서는 score-based diffusion 모델의 수렴체계를 개선하여 실제 적용에 더 가까운 성과를 도출합니다. 특히, 기존 이론에서 도출된 반복 복잡도를 개선하였으며, 이를 통해 d^{1/3}ε^{-2/3}의 결과를 얻었습니다.

- **Technical Details**: 논문에서는 score-based diffusion 모델의 수렴 분석을 위해 랜덤화된 중간 지점 방법(randomized midpoint method)을 이용하여 수렴성을 평가하였습니다. 이 방법은 로그-오목성(log-concavity) 분포에 대한 샘플링으로 처음 제안되었고(Shen and Lee, 2019), 이후 diffusion 모델에 적용되었습니다. 본 이론은 ε-정확한 점수 추정치를 수용하며, 타겟 분포에 대한 로그-오목성을 요구하지 않습니다.

- **Performance Highlights**: 저자들은 이번 연구를 통해 CIFAR-10과 ImageNet 같은 이미지 데이터셋에 대해 이전 연구에서 요구된 이론적 단계 수치인 약 5050단계보다 적은 단계를 거쳐 좋은 샘플 생성이 가능하다는 것을 강조합니다. 이로 인해 본 연구는 실제 상황에서의 응용 가능성을 포함하여 score-based generative 모델의 수렴 속도를 크게 개선한 것으로 평가됩니다.



### Movie Gen: A Cast of Media Foundation Models (https://arxiv.org/abs/2410.13720)
- **What's New**: 이번 논문에서는 Movie Gen이라는 새로운 foundation 모델 세트를 제안합니다. 이 모델은 다양한 화면 비율과 동기화된 오디오와 함께 고품질 1080p HD 비디오를 생성하며, 사용자의 이미지를 기반으로 한 개인화된 비디오 생성 및 정밀한 지침 기반 비디오 편집 기능도 포함되어 있습니다.

- **Technical Details**: Movie Gen은 30B 파라미터의 트랜스포머 모델로, 최대 73K 비디오 토큰의 컨텍스트 길이를 가지고 있습니다. 이 모델은 텍스트-비디오 합성, 비디오 개인화, 비디오 편집, 비디오-오디오 생성 및 텍스트-오디오 생성과 같은 다양한 작업에서 최첨단 성능을 기록합니다. 인터넷 스케일의 이미지, 비디오, 오디오 데이터를 통해 사전 학습되었습니다.

- **Performance Highlights**: Movie Gen 모델은 기존 상업 시스템을 초월하여 텍스트-비디오 생성, 비디오 개인화, 정밀 비디오 편집 및 오디오 생성 작업에서 탁월한 성능을 보여줍니다. 특히, Movie Gen Video는 최대 16초의 개인화된 HD 비디오 생성을 가능하게 하며, Movie Gen Audio는 정밀한 음악 생성과 음향 효과 생성을 지원합니다.



### On the Role of Attention Heads in Large Language Model Safety (https://arxiv.org/abs/2410.13708)
Comments:
          28 pages, 18 figures, 7 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 안전 메커니즘의 특정 주의(attention) 헤드의 기여도를 이해하고, 그로 인해 발생하는 안전성 문제를 분석합니다. 특히, Safety Head ImPortant Score (Ships)라는 새로운 메트릭을 도입하여 안전성과 관련된 다중 헤드 주의 메커니즘을 탐구합니다.

- **Technical Details**: 우리는 LLM의 안전성 능력을 다중 헤드 주의(mechanism)와 연결하여 해석하기 위한 연구를 진행했습니다. Ships는 개별 주의 헤드가 해로운 쿼리에 대한 거부 확률 변화에 미치는 영향을 정량화합니다. 추가적으로, Safety Attention Head AttRibution Algorithm (Sahara)을 제안하여 중요 헤드를 그룹화하고, 이들이 모델의 안전성에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 실험 결과, Llama-2-7b-chat 모델에서 안전 헤드 하나를 제거했을 때, 해로운 쿼리에 대한 공격 성공률(ASR)이 0.04에서 0.64로 증가하였으며, 이는 기존 연구에서 필요로 했던 약 5%의 매개변수 수정과 대조적으로 단 0.006%를 수정함으로써 이루어졌습니다. 또한, 유사한 기본 모델에서 미세 조정된 LLM의 안전 헤드들이 겹친다는 점에서, 안전성에 대한 기존 연구와 새로운 통찰을 제공합니다.



### Ab initio nonparametric variable selection for scalable Symbolic Regression with large $p$ (https://arxiv.org/abs/2410.13681)
- **What's New**: 본 논문에서는 대규모 데이터셋에서의 기호 회귀(Symbolic Regression, SR)의 확장성을 해결하기 위해 PAN+SR이라는 새로운 방법론을 제안합니다. 이 방법은 비모수 변수 선택(nonparametric variable selection)과 기호 회귀를 결합하여 대규모 입력 공간을 효율적으로 선별하고 검색 복잡성을 줄이는 동시에 정확성을 유지합니다.

- **Technical Details**: PAN+SR 방법은 'Parametric Assisted by Nonparametrics' (PAN) 전략을 활용하여 대량의 입력 변수를 사전 스크리닝합니다. 이 과정은 정확한 표현이 검색 공간에서 사라지지 않도록 필수적입니다. 또한, 본 연구에서는 SRBench를 확장하여 다양한 신호 대 잡음 비율을 가진 고차원 회귀 문제를 포함시키고, 17개의 현대 기호 회귀 방법의 성능 향상을 시연합니다.

- **Performance Highlights**: PAN+SR은 17개의 기존 기호 회귀 방법의 성능을 꾸준히 향상시켜 주며, 몇 가지 방법은 이러한 도전적인 데이터셋에서 최첨단 성능(state-of-the-art performance)을 달성하게 됩니다.



### Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation (https://arxiv.org/abs/2410.13640)
Comments:
          33 pages, 18 figures, 12 tables

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 정확도를 추정하기 위한 기존의 레이블 필요성을 없애고, 잠재 공간(latent space)에서 Chain-of-Embedding (CoE) 메서드를 제안하였습니다. 이 방식은 LLM이 스스로 출력 없는 자기 평가를 수행할 수 있도록 합니다.

- **Technical Details**: CoE는 LLM의 추론 과정에서 생성되는 모든 점진적 은닉 상태(hidden state)를 포함하며, 이는 LLM의 사고 경로(thinking path)를 나타냅니다. 연구 결과, LLM이 정답을 낼 때와 아닐 때 CoE의 특징이 다르게 나타나는 것을 확인하였으며, 이러한 차이를 통해 LLM의 응답 정확성을 추정할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험을 통해 다양한 분야(수학, 추론, 지식, 이해)에서 LLM 7종의 성과를 평가하였으며, CoE 메서드가 지연 없이 신뢰성 있는 피드백을 제공하는 것을 입증하였습니다. 또한, 이 방법은 레이블이 필요 없고, 밀리세컨드 수준의 계산 비용으로 대규모 환경에서도 실시간 피드백이 가능함을 강조합니다.



### H2OVL-Mississippi Vision Language Models Technical Repor (https://arxiv.org/abs/2410.13611)
- **What's New**: H2OVL-Mississippi 모델은 3700만 개의 이미지-텍스트 쌍을 기반으로, 8개의 H100 GPU를 사용하여 240시간 동안 훈련된 작은 비전-언어 모델(VLM) 쌍을 소개합니다. 특히, H2OVL-Mississippi-0.8B는 8억 개의 매개변수로 구성되어 텍스트 인식에 특화되어 있으며, OCRBench의 텍스트 인식 부문에서 최첨단 성능을 발휘하고 있습니다.

- **Technical Details**: H2OVL-Mississippi 모델은 Vision Transformer(비전 트랜스포머) 구성 요소와 대형 언어 모델(LLM)로 이루어집니다. H2OVL-Mississippi-0.8B는 OCR 및 문서 중심 작업에 최적화되어 있고, H2OVL-Mississippi-2B는 다양한 멀티모달 작업을 수행할 수 있는 일반 목적 모델입니다. 이들은 각각 256에서 1590개의 시각적 토큰을 생성하며, 동적 해상도 전략(dynamic resolution)과 다중 스케일 적응 크롭(multi-scale adaptive cropping) 전략을 활용하여 다양한 이미지 크기와 종횡비에 적응합니다.

- **Performance Highlights**: H2OVL-Mississippi-0.8B는 OCRBench에서 텍스트 인식 부문에서 최첨단 성능을 보여주며, H2OVL-Mississippi-2B는 다양한 학술 벤치마크에서 경쟁력 있는 메트릭스를 제공합니다. 두 모델 모두 H2O-Danube 언어 모델의 기능을 확장하여 비주얼 도메인으로의 적용 가능성을 높이고, Apache 2.0 라이선스 하에 공개되어 문서 AI와 비주얼 LLM의 접근성을 높였습니다.



### Towards Satellite Non-IID Imagery: A Spectral Clustering-Assisted Federated Learning Approach (https://arxiv.org/abs/2410.13602)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 저지구 궤도(LEO) 위성 시스템에서 지구 관측 데이터(EOD)를 효과적으로 처리하기 위한 새로운 방법인 OSC-FSKD(Orbit-based Spectral Clustering-assisted Clustered Federated Self-Knowledge Distillation)를 제안합니다. 이는 데이터 전송 없이 FL(Federated Learning) 방식을 활용하여 모델을 훈련하는 혁신적인 접근 방식입니다.

- **Technical Details**: OSC-FSKD 방법은 주어진 궤도의 sink node와 ordinary nodes로 구성된 다중 궤도 기반 위성 집합을 통해 이루어집니다. 이 방법에서는 정규화된 Laplacian 기반의 스펙트럴 클러스터링(Normalized Laplacian-based Spectral Clustering, NLSC)을 사용하여 각 클러스터에서 모델 집합을 수행하고, 자기 지식 증류(Self-Knowledge Distillation, SKD)를 통해 최적의 로컬 모델 훈련을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법의 지구 관측 정확도가 pFedSD, FedProx, FedAU, FedALA 메서드보다 각각 1.01배, 2.15배, 1.10배, 1.03배 더 높은 성능을 보였습니다. 또한, 다른 데이터셋에서도 우수성을 나타냈습니다.



### Generative Adversarial Synthesis of Radar Point Cloud Scenes (https://arxiv.org/abs/2410.13526)
Comments:
          ICMIM 2024; 7th IEEE MTT Conference

- **What's New**: 이 논문에서는 자동차 레이더의 검증과 검증을 위해 현실적인 교통 시나리오 데이터셋이 필요하다는 점을 논의하며, GANs(Generative Adversarial Networks)를 활용한 레이더 장면 합성을 제안합니다.

- **Technical Details**: PointNet++ 기반의 GAN 모델을 사용하여 현실적인 레이더 포인트 클라우드 장면을 생성하며, 생성된 장면의 성능을 실제 장면의 테스트 세트와 비교하기 위해 이진 분류기(binary classifier)를 사용합니다.

- **Performance Highlights**: 우리의 GAN 모델은 실제 장면 테스트 세트에 대해 ~87%의 유사한 성능을 달성함을 보여줍니다.



### CERES: Critical-Event Reconstruction via Temporal Scene Graph Completion (https://arxiv.org/abs/2410.13514)
Comments:
          7 pages, 8 figures

- **What's New**: 이 논문은 실제 데이터에 기반한 주문형 시나리오 생성을 위한 방법을 제안합니다. 자율주행 차량(AV)의 안전성과 정합성을 평가하기 위해, 실제 데이터로부터 도출한 시나리오를 시뮬레이션에 통합하여 테스트 세트의 신뢰성과 검증성을 향상시킵니다.

- **Technical Details**: 본 연구는 Graph Neural Networks(GNNs)를 통해 시각적 자원(visual resources)과 시간적 장면 그래프(temporal scene graphs)를 사용하여 장면의 동적인 관계를 포착하여 시뮬레이션 내에서 시나리오를 생성합니다. 사용자가 정의한 행동(action)과 중요도(criticality)에 따라 유연한 시나리오 생성을 보장합니다.

- **Performance Highlights**: 모델은 요청된 시나리오와 관련된 링크를 예측하는 데 있어 기준 성능 대비 유의미한 개선을 보여주며, CARLA와 같은 기존 시뮬레이터에서의 테스트 및 검증을 통해 유효성과 호환성을 평가합니다.



### SAda-Net: A Self-Supervised Adaptive Stereo Estimation CNN For Remote Sensing Image Data (https://arxiv.org/abs/2410.13500)
Comments:
          Will be presented at ICPR2024 in December 2024 in Kolkata, India

- **What's New**: 본 논문에서는 기존의 깊이 학습(deep learning) 기반 스테레오 추정(stereo estimation) 방법이 정확한 지상 진리(ground truth) 데이터에 의존하는 단점을 극복하기 위해, 지상 진리 데이터 없이도 훈련이 가능한 자가 지도(Self-supervised) CNN을 제안합니다.

- **Technical Details**: 제안된 방법은 단계별로 진행되며, 초기에 생성된 분산 맵(disparity map)은 부정확하고 잡음(noise)이 많습니다. 왼쪽-오른쪽 일관성 체크(left-right consistency check)를 사용하여 초기 의사 지상 진리(pseudo ground-truth)를 생성하고, 이를 기반으로 매 에포크(epoch)마다 모델을 업데이트합니다. 불일치 포인트의 합을 통해 네트워크의 수렴(convergence)을 추적합니다.

- **Performance Highlights**: 실제 복잡한 장면에서 좋은 성능을 나타내며, 약 495K의 경량화된 파라미터를 사용함으로써 상업적 하드웨어에서 효율적으로 사용 가능합니다.



### Enhancing Text Generation in Joint NLG/NLU Learning Through Curriculum Learning, Semi-Supervised Training, and Advanced Optimization Techniques (https://arxiv.org/abs/2410.13498)
- **What's New**: 본 연구에서는 Joint Natural Language Generation (NLG)와 Natural Language Understanding (NLU) 학습 상황에서 텍스트 생성을 개선하기 위한 새로운 접근 방식을 개발했습니다.

- **Technical Details**: 데이터는 주석이 달린 데이터셋을 수집하고 전처리하여 준비하였으며, 여기에는 데이터 정리(cleaning), 토큰화(tokenization), 형태소 분석(stemming), 불용어 제거(stop-word removal)가 포함됩니다. 또한, 특징 추출 기법으로는 POS tagging, Bag of Words, Term Frequency-Inverse Document Frequency (TF-IDF)가 사용됩니다. Transformer 기반의 인코더와 디코더는 긴 범위의 의존성을 포착하고 소스-타겟 시퀀스 모델링을 개선합니다. Optimized BERT와 Hybrid Redfox Artificial Hummingbird Algorithm (HRAHA)와 같은 사전 훈련된 언어 모델이 통합되었습니다.

- **Performance Highlights**: 정책 경량화 기법을 통한 강화 학습(reinforcement learning), 반지도 학습(semi-supervised training), 개선된 주의 메커니즘(attention mechanisms) 및 미분 가능한 근사치(differentiable approximations)를 사용하여 모델을 미세 조정하고 복잡한 언어 과제를 효과적으로 처리합니다. 이 모델은 Python을 사용하여 구현되었습니다.



### Novelty-based Sample Reuse for Continuous Robotics Contro (https://arxiv.org/abs/2410.13490)
- **What's New**: 이 논문에서는 인공지능의 강화 학습에서 샘플의 불균형한 활용 문제를 해결하기 위해 Novelty-guided Sample Reuse (NSR) 방법을 제안합니다. NSR은 자주 관찰되는 상태는 추가 업데이트를 건너뛰고 드물게 관찰되는 상태에 대해 더 많은 업데이트를 제공하여 샘플 활용도를 극대화합니다.

- **Technical Details**: NSR 방법은 Random Network Distillation (RND) 기법을 사용하여 상태의 참신함(Novelty)를 평가합니다. 이 기법은 무작위로 초기화된 신경망의 예측 오차를 측정하여, 고예측 오차가 오히려 더 많은 업데이트를 필요로 하는 새로운 상태를 나타냅니다. 이러한 구조는 DDPG와 통합되어 성능을 크게 향상시킵니다.

- **Performance Highlights**: 우리의 실험 결과, NSR은 알고리즘의 수렴 속도와 성공률을 향상시키며, 환경과의 상호작용을 최소화하면서도 효과적인 학습을 지원합니다. 이는 로봇과 같은 현실 세계 응용에서의 강화 학습 성능을 크게 높입니다.



### Seeing Through VisualBERT: A Causal Adventure on Memetic Landscapes (https://arxiv.org/abs/2410.13488)
Comments:
          Accepted at EMNLP Findings 2024

- **What's New**: 본 논문은 Structural Causal Model (SCM)을 기반으로 한 새로운 프레임워크를 제안하여, offensive memes의 탐지에서 투명성을 높이며 모형의 행동 해석을 가능하게 한다. VisualBERT를 활용하여 meme 입력과 causal concepts를 모두 고려하여 클래스를 예측한다.

- **Technical Details**: 이 프레임워크는 기존 interpretability 기술의 한계를 극복하기 위해 causal concepts를 통합하고, dynamic routing과 adversarial learning을 활용하여 meme의 공격성을 예측한다. 아울러, 모델 예측의 원인과 오류 사례를 명확히 설명한다.

- **Performance Highlights**: 정량적 분석 결과, 제안한 모델링 기법들이 기존의 input attribution 방법들과 비교하여 causality를 만족하지 못하는 점을 강조하며, 이로 인해 safety-critical applications에서의 신뢰성에 의문을 제기한다. 또한, qualitative 분석을 통해 모델의 결정이 정당화될 수 있는지를 평가하였다.



### Breaking the Manual Annotation Bottleneck: Creating a Comprehensive Legal Case Criticality Dataset through Semi-Automated Labeling (https://arxiv.org/abs/2410.13460)
- **What's New**: 이 논문에서는 스위스 연방 대법원 판결의 미래 법리에 대한 영향을 평가하기 위한 새로운 Criticality Prediction (사례 중요도 예측) 데이터셋을 소개합니다. 기존의 수작업 주석 접근 방식과 달리, 본 데이터셋은 반자동적으로 레이블을 유도하여 훨씬 더 큰 데이터셋을 제공합니다.

- **Technical Details**: 제안된 데이터셋은 2단계 레이블링 시스템을 특징으로 하며, (1) LD-Label: 주요 결정으로 발표된 사례를 식별하는 이진 지표, (2) Citation-Label: 사례의 인용 빈도와 최근성에 따라 사례를 평가합니다. 이 데이터셋은 2002년부터 2023년까지의 사례를 포함하며 언어는 독일어, 프랑스어, 이탈리아어로 구성됩니다.

- **Performance Highlights**: 여러 다국어 모델을 평가한 결과, 세밀하게 조정된 모델이 제로샷(Zero-shot) 기준선보다 일관되게 우수한 성능을 보였습니다. 이를 통해 작업 특화적 적응이 필요함을 입증하였습니다.



### Unlocking Legal Knowledge: A Multilingual Dataset for Judicial Summarization in Switzerland (https://arxiv.org/abs/2410.13456)
- **What's New**: 이번 논문은 스위스 연방 대법원(SFSC)의 판결을 바탕으로 한 새로운 데이터셋인 Swiss Leading Decision Summarization (SLDS)를 소개하고 있습니다. 이 데이터셋은 독일어, 프랑스어 및 이탈리아어로 된 18,000개의 법원 판결과 독일어 요약을 포함하고 있어, 다국어 법적 요약에 대한 연구를 촉진할 수 있습니다.

- **Technical Details**: 논문에서는 3가지 mT5(multi-lingual T5) 변형 모델과 고유 모델을 미세 조정(fine-tuning)하고 평가했습니다. 분석 결과, 고유 모델은 제로샷(zero-shot) 및 원샷(one-shot) 설정에서 우수한 성능을 보였으나, 미세 조정된 모델이 여전히 강력한 경쟁력을 유지하는 것으로 나타났습니다.

- **Performance Highlights**: SLDS 데이터셋의 공개를 통해 법적 요약 및 법무 전문가를 위한 보조 기술 개발에 대한 추가 연구가 촉진될 것으로 기대됩니다. 이 데이터셋은 수백만 건의 판결을 법률 연구에 더 쉽게 접근할 수 있게 할 잠재력이 있습니다.



### Parameter-efficient Adaptation of Multilingual Multimodal Models for Low-resource ASR (https://arxiv.org/abs/2410.13445)
- **What's New**: 본 연구에서는 낮은 자원을 가진 언어에 대한 자동 음성 인식(ASR)을 개선하기 위해, 다국어 다중 모달 모델인 SeamlessM4T를 활용하여 매개변수 효율적인 미세 조정(parameter-efficient fine-tuning) 및 텍스트 전용 적응(text-only adaptation) 기술을 결합한 방법을 제시합니다.

- **Technical Details**: SeamlessM4T는 다중 언어 및 다중 모달 머신 번역 지원을 제공하는 엔드-투-엔드 모델로, 96개 언어의 입력 및 출력에 대해 자동 음성 인식, 텍스트-음성 변환 등 여러 작업을 수행할 수 있습니다. 이 모델은 셀프-슈퍼바이즈드(self-supervised) 방식으로 백만 시간 이상의 무 라벨 음성을 학습하여 성능이 개선되었습니다.

- **Performance Highlights**: 본 논문에서는 높은 자원 언어에서 낮은 자원 언어로의 언어 간 전이(cross-lingual transfer)를 통해, 라벨이 없는 음성 데이터 없이도 WER(Word Error Rate)를 17% 이상 감소시킬 수 있음을 보였습니다.



### RAMPA: Robotic Augmented Reality for Machine Programming and Automation (https://arxiv.org/abs/2410.13412)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 새로운 로보틱 증강 현실 시스템인 RAMPA는 최신 AR 헤드셋을 활용하여 로봇 프로그래밍을 직관적으로 지원합니다. 이 시스템은 실시간으로 데이터 수집, 시각화 및 기술 시연 조정이 가능합니다.

- **Technical Details**: RAMPA는 Programming from Demonstration (PfD) 접근법을 활용하여 Universal Robots UR10과 같은 산업용 로봇 팔에서 로봇 프로그래밍을 간소화하며, 사용자가 물리적 환경에서 안전하게 작업할 수 있도록 하였습니다. 이 시스템은 Trajectory Smoothness, Task Performance, System Usability 등의 다양한 양적 지표를 사용하여 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, RAMPA 시스템은 전통적인 방법과 비교하여 작업 효율성을 크게 향상시키고, 작업자의 안전성과 경험을 개선하는 것으로 나타났습니다. 이를 통해 로봇 프로그래밍 분야에서 사용자 참여와 효율성을 높이는 잠재력을 보여주었습니다.



### Learning Counterfactual Distributions via Kernel Nearest Neighbors (https://arxiv.org/abs/2410.13381)
Comments:
          33 pages, 2 figures

- **What's New**: 본 연구에서는 여러 단위(예: 개인, 집단, 지리적 위치)와 결과(예: 치료, 시간, 항목)를 고려한 설정에서 각 단위-결과 항목에 대한 다변량 분포를 배우는 것을 목표로 하며, 특히 데이터를 무작위로 누락하는 문제를 다룹니다.

- **Technical Details**: 이 문제를 해결하기 위해 새로운 distributional matrix completion 프레임워크를 제안하며, kernel 기반의 distributional generalization 접근법을 통해 기본 분포를 추정합니다. 최대 평균 편차(maximum mean discrepancies)와 적절한 요인 모델(factor model)을 활용하여 누락된 데이터의 문제를 견고하게 극복합니다.

- **Performance Highlights**: 제안된 최근접 이웃(nearest neighbors) 접근법은 데이터의 이질성 소음에 견고하며, 이전 연구에서는 다루지 않은 여러 측정값을 통해 보다 나은 성능을 보입니다.



### Remember, Retrieve and Generate: Understanding Infinite Visual Concepts as Your Personalized Assistan (https://arxiv.org/abs/2410.13360)
- **What's New**: 본 논문에서는 Retrieval Augmented Personalization (RAP) 프레임워크를 소개하여 다중 모드 대형 언어 모델(MLLMs)의 개인화를 가능하게 합니다. RAP는 일반 MLLM을 개인화된 어시스턴트로 전환하는 세 가지 주요 단계로 구성됩니다: 기억(Recall), 검색(Retrieve), 생성(Generate).

- **Technical Details**: RAP는 사용자 관련 정보(예: 이름, 아바타 등)를 저장하는 키-값 데이터베이스를 설계합니다. 사용자가 대화를 시작할 때, RAP는 다중 모드 검색기를 통해 데이터베이스에서 관련 정보를 검색하고, 이를 MLLM에 입력하여 개인화된 지식 강화 응답을 생성합니다. 추가로 생성 품질 향상을 위해 데이터 수집 파이프라인을 개발하고 개인화된 훈련을 위한 전문적인 데이터셋을 생성합니다.

- **Performance Highlights**: RAP-MLLMs는 개인화된 이미지 캡션 작성, 질문 응답 및 시각적 인식과 같은 다양한 작업에서 뛰어난 유연성과 생성 품질을 보여줍니다. 모델들은 무한한 시각적 개념에 대해 일반화 능력을 발휘하며, 사용자 관련 정보를 효과적으로 처리하여 개인화된 출력을 제공합니다.



### Representation Learning of Structured Data for Medical Foundation Models (https://arxiv.org/abs/2410.13351)
Comments:
          NeurIPS 2024 Workshop on Unifying Representations in Neural Models (UniReps 2024)

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 의료 분야에서 비문자적 구조 데이터, 특히 ICD-10 및 SNOMED-CT와 같은 의료 코드를 효과적으로 처리하는 데 직면한 문제들을 해결하기 위한 UniStruct 아키텍처를 제안합니다.

- **Technical Details**: UniStruct는 비구조화된 텍스트와 구조화된 데이터를 결합한 다중 모달(multi-modal) 의료 모델을 설계하고, 의료 코드에 적합하도록 하위 단어(tokenization) 기법을 조정하여 이러한 문제를 해결합니다. 아이디어는 자주 함께 발생하는 의료 코드 그룹을 단일 토큰으로 처리하는 것입니다.

- **Performance Highlights**: 내부 의료 데이터베이스에서 10억 개 이상의 토큰으로 사전 훈련된 UniStruct 모델은 평가 메트릭에서 최대 23% 개선을 달성했으며, EHRSHOT 공공 벤치마크에서 1/1000의 사전 훈련 데이터에도 불구하고 42% 이상의 하위 작업에서 성능을 개선했습니다.



### Do LLMs Overcome Shortcut Learning? An Evaluation of Shortcut Challenges in Large Language Models (https://arxiv.org/abs/2410.13343)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 성능 및 일반화 능력에 미치는 지름길(shortcut)의 영향을 평가하기 위해 Shortcut Suite라는 포괄적인 테스트 스위트를 제안합니다. 이 스위트는 6종의 지름길 유형, 5개의 평가 지표 및 4가지 프롬프트 전략을 통합하여 LLM의 성능을 평가합니다.

- **Technical Details**: Shortcut Suite는 LLM의 성능을 아래의 세 가지 관점에서 평가합니다: 1) LLM이 다운스트림 태스크에서 지름길 의존도를 평가하기 위해 6개의 데이터셋 수집 및 정확성을 분석. 2) 정확도 외에 설명 능력 평가를 위한 3가지 새로운 지표(Semantic Fidelity Score, Internal Consistency Score, Explanation Quality Score) 도입. 3) 지름길 학습에서 LLM의 성능 및 다양한 프롬프트 전략 비교. 또한, LLM이 예측에 과신하는 경향을 보여줍니다.

- **Performance Highlights**: 연구 결과, LLM들은 지름길을 활용할 때 성능이 현저히 떨어지며 (최대 40% 이상), 큰 LLM이 zero-shot 및 few-shot ICL 프롬프트 하에서 지름길을 더 많이 사용합니다. Chain-of-Thought prompting이 지름길 의존도를 줄이는 데 효과적이며, LLM의 일반 설명 품질이 낮은 것도 확인되었습니다.



### Do LLMs Have Political Correctness? Analyzing Ethical Biases and Jailbreak Vulnerabilities in AI Systems (https://arxiv.org/abs/2410.13334)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)에서 안전성을 위해 주입된 의도적 편향이 어떻게 유해한 콘텐츠 생성을 초래할 수 있는지를 탐구하고, 새로운 패러다임인 PCJailbreak와 간단한 방어 전략인 PCDefense를 제안합니다.

- **Technical Details**: LLM의 안전성을 확보하기 위한 다양한 방법(데이터 필터링, 지도 학습 피드백 등)은 보통 의도적인 편향을 불러오며, 이는 'jailbreak' 현상을 초래합니다. PCJailbreak는 이러한 편향을 이용하여 LLM이 유해한 출력을 생성할 수 있도록 조작하는 공격이며, PCDefense는 공격 전에 방어 프롬프트를 주입하여 이를 방지하는 방법입니다.

- **Performance Highlights**: PCJailbreak는 최신 GPT 모델에서도 효과적이며, 제안하는 PCDefense 방법은 추가적인 추론 비용 없이도 jailbreak 공격을 완화할 수 있음을_showcase합니다. 이 연구는 LLM 제공업체들이 보다 책임감 있게 안전성을 설계하고 구현해야 함을 강조합니다.



### Active inference and deep generative modeling for cognitive ultrasound (https://arxiv.org/abs/2410.13310)
- **What's New**: 초음파(US) 이미징 시스템을 정보 탐색 에이전트로 재구성하여 효율적인 진단을 위한 대칭적 상호작용을 제안합니다. 이 시스템은 자율적으로 촬영을 개인화하고 현장에서 정보 획득을 극대화합니다.

- **Technical Details**: 이 연구에서는 초음파 데이터 수집과 재구성이 '지각-행동 루프(perception-action loop)'로 해석되며, Bayesian inference를 통해 불확실성을 줄이고 진단 가치를 극대화하는 방법을 설명합니다. 시스템은 생성적 모델을 활용하여 환경을 이해하고 최적의 측정을 계획합니다.

- **Performance Highlights**: 딥 생성적 모델을 통해 초음파 이미징의 품질과 진단 정확도를 크게 향상시킬 수 있는 잠재력을 보여주며, 특히 어려운 환자 군에서 지속적인 이미지 분석 및 개입이 가능하도록 합니다.



### A theoretical perspective on mode collapse in variational inferenc (https://arxiv.org/abs/2410.13300)
- **What's New**: 이 논문에서는 변별적 추론(Variational Inference, VI)에서의 모드 붕괴(mode collapse) 문제를 이론적으로 조사하였습니다. Gaussian mixture models에서 gradient flow의 역학을 분석하며, 모드 붕괴가 발생하는 두 가지 주요 메커니즘인 평균 정렬(mean alignment)과 사라지는 가중치(vanishing weight)를 식별하였습니다.

- **Technical Details**: 모드 붕괴 문제를 다루기 위해 bi-modal Gaussian mixture target distribution을 설정하고, 관련 요약 통계량(summary statistics)에 대한 저차원 동적 시스템(dynamical system)의 고정점(fixed points)을 통해 gradient flow의 역학을 연구하였습니다. 이를 통해 다양한 변별적 분포(q ∈ 𝒬)의 선택에 따라 모드 붕괴가 발생함을 실험적으로 확인하였습니다.

- **Performance Highlights**: 정리하자면, 이 연구는 VI 접근 방식을 구현하는 정상화 흐름(normalizing flows)이 여전히 모드 붕괴를 겪는 것을 보여주며, 초기화(initialization) 및 변별적 분포의 특정 파라미터화(parameterization)가 모드 붕괴를 완화하는 데 도움이 될 수 있음을 시사합니다.



### Learning to Route with Confidence Tokens (https://arxiv.org/abs/2410.13284)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)에서 신뢰성을 판단하고 이를 기반으로 결과를 개선하기 위한 새로운 방법론인 Self-REF(Self-Reflection)를 제안합니다. Self-REF는 LLM이 자신의 예측에 대한 신뢰도를 효과적으로 평가할 수 있도록 훈련하는 경량화된 방법론입니다.

- **Technical Details**: Self-REF는 세 가지 주요 단계를 포함합니다: (i) 신뢰도 토큰 주석 추가, (ii) Self-REF 파인튜닝, (iii) 신뢰도 점수 추출. 신뢰도 토큰은 LLM이 올바르게 응답했는지를 기준으로 생성되며, 이러한 토큰에서 신뢰도 점수를 계산하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: Self-REF는 라우팅 및 거부 학습 작업에서 기존 방법들보다 뛰어난 성능을 보이며, 특히 네 개의 공개 데이터셋에서 우수한 결과를 나타냈습니다. 이는 LLM이 낮은 신뢰도 질문을 더 강력한 LLM으로 라우팅하거나 안전한 행동으로 자신의 대답을 거부하는 데 기여합니다.



### Inductive Gradient Adjustment For Spectral Bias In Implicit Neural Representations (https://arxiv.org/abs/2410.13271)
Comments:
          28 pages, 12 figures

- **What's New**: 이 논문은 Implicit Neural Representations (INRs)에서의 spectral bias 문제를 해결하기 위해, Multi-layer Perceptrons (MLPs)의 선형 역학 모델을 탐구하고 empirical Neural Tangent Kernel (eNTK) 행렬을 기반으로 inductive gradient adjustment (IGA) 방법을 제안합니다.

- **Technical Details**: 이 연구는 MLPs의 linear dynamics 모델을 이용하여 empirical NTK (eNTK) 행렬을 통해 spectral bias와 training dynamics 사이의 관계를 이론적으로 규명합니다. 제안하는 IGA 방법은 대량의 데이터 포인트에 대해 eNTK 기반 gradient 변환 행렬의 inductive 일반화를 통해 spectral bias를 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 기존의 training dynamics 조정 방법들보다 더 나은 성능을 발휘하며, INRs의 퀄리티를 향상시켜 고해상도 텍스처와 뚜렷한 변별력을 선보임을 보여줍니다.



### A Simplifying and Learnable Graph Convolutional Attention Network for Unsupervised Knowledge Graphs Alignmen (https://arxiv.org/abs/2410.13263)
Comments:
          14 pages, 3 figures

- **What's New**: 최근의 연구에서는 Entity Alignment (EA) 작업의 성공이 레이블이 붙은 데이터에서 제공되는 감독 정보에 크게 의존하고 있음을 강조합니다. 그러나 레이블된 데이터의 비용을 고려할 때 이러한 방법의 실용성은 제한적입니다. 따라서, 본 논문에서는 Unsupervised Knowledge Graphs alignment를 위한 Simplifying and Learnable graph convolutional attention network (SLU)를 제안합니다.

- **Technical Details**: SLU는 LCAT라는 새로운 그래프 신경망(GNN)을 백본 네트워크로 사용하여 Knowledge Graph (KG)의 그래프 구조를 모델링합니다. SLU는 잠재적 매칭 관계를 바탕으로 관계 구조를 재구성하는 방법을 설계하여 잘못된 이웃 정보를 필터링하고, 유사성을 측정하기 위해 일관성 기반의 유사성 함수를 제안합니다.

- **Performance Highlights**: SLU는 세 가지 데이터셋 (15K 및 100K)에서 광범위한 실험을 수행한 결과, 25개의 감독 또는 비감독 방법들을 초월하여 정렬 정확도를 유의미하게 향상시켰습니다. 가장 좋은 경우에서 Hits@1 점수가 6.4% 향상되었습니다.



### LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch (https://arxiv.org/abs/2410.13213)
- **What's New**: LLMOPT라는 통합 학습 기반 프레임워크를 제안하여 최적화 문제의 일반화 능력을 향상시켰습니다. 이 프레임워크는 자연어 설명으로부터 최적화 문제를 정의하고 해결하는 과정을 자동화하는 데 중점을 두고 있습니다.

- **Technical Details**: LLMOPT는 다섯 가지 요소로 구성된 포뮬레이션을 통해 다양한 최적화 문제 유형을 정의하고, 다중 지침 튜닝(multi-instruction tuning) 및 모델 정렬(model alignment)으로 정확성과 일반성을 향상시킵니다. 또한 자동 테스트(auto-testing)와 자기 수정(self-correction) 메커니즘을 통해 hallucinations를 방지합니다.

- **Performance Highlights**: LLMOPT는 20개 분야에서 6개의 실제 데이터셋을 대상으로 평가된 결과, 선형/비선형 프로그래밍, 혼합 정수 프로그래밍 및 조합 최적화와 같은 다양한 최적화 문제를 처리하며 최신 방법보다 평균 11.08%의 해결 정확도 향상을 달성했습니다.



### Meta-DiffuB: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration (https://arxiv.org/abs/2410.13201)
- **What's New**: Meta-DiffuB는 Seq2Seq 텍스트 생성을 위한 새로운 스케줄러-탐색자 모델을 도입하여 기존 S2S-Diffusion 모델의 한계를 극복합니다. 기존 모델들은 고정된 또는 수작업으로 만든 규칙에 의존하여 노이즈를 스케줄링하는 반면, Meta-DiffuB는 문맥화된 노이즈 스케줄링을 통해 문장별로 적합한 노이즈를 적용합니다.

- **Technical Details**: Meta-DiffuB는 두 가지 모델로 구성됩니다: 스케줄러와 탐색자. 스케줄러는 각 문장의 특성에 맞춰 적절한 수준의 노이즈를 스케줄링하고, 탐색자는 해당 노이즈를 활용하여 업데이트 및 생성을 수행합니다. 이 접근 방식은 자연어 처리(NLP)에서 Seq2Seq 작업의 의미론적 특성을 반영합니다.

- **Performance Highlights**: Meta-DiffuB는 네 가지 Seq2Seq 벤치마크 데이터세트에서 기존 S2S-Diffusion 모델 및 정밀 조정된 사전 훈련된 언어 모델(PLMs)과 비교하여 최첨단 성능을 달성합니다. 또한, 스케줄러 모델은 기존 DiffuSeq를 더욱 향상시키기 위한 '플러그 앤 플레이' 기능을 제공합니다.



### Context-Enhanced Multi-View Trajectory Representation Learning: Bridging the Gap through Self-Supervised Models (https://arxiv.org/abs/2410.13196)
- **What's New**: MVTraj는 다중 시각의 맥락을 통합하여 경로 표현 학습을 향상시키는 새로운 방법을 제안합니다. GPS, 도로 네트워크 및 POI(관심 지점)의 다양한 맥락적 지식을 활용하여 경로 데이터에 대한 보다 포괄적인 이해를 제공합니다.

- **Technical Details**: MVTraj는 GPS 경로를 연결 고리로 사용하고 셀프 슈퍼바이즈드(자기지도학습) 프리텍스트(사전학습) 작업을 통해 다중 시각 간 학습 프로세스를 정렬합니다. 3개의 다양한 시각(예: GPS, 도로 경로 및 그리드)의 경로를 다루는데, 각 시각에서 독립적인 모달리티로 간주하고 계층적 크로스 모달 상호작용 모듈을 적용하여 지식을 융합합니다.

- **Performance Highlights**: 실제 데이터셋을 활용한 폭넓은 실험 결과, MVTraj는 다양한 공간 시각과 관련된 작업에서 기존의 기준선 모델에 비해 현저한 성능 향상을 보여줍니다.



### EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning (https://arxiv.org/abs/2410.13179)
- **What's New**: 이번 논문에서는 Speech Representation Learning을 위한 새로운 Self-Supervised Learning 접근 법인 EH-MAM (Easy-to-Hard adaptive Masked Acoustic Modeling)을 제안합니다. 기존의 랜덤 마스킹 방식을 사용하는 Masked Acoustic Modeling (MAM)과는 달리, 우리는 선택적이고 적응적인 마스킹 전략을 도입하였습니다.

- **Technical Details**: EH-MAM은 SSL 훈련 중 모델에 점진적으로 더 어려운 영역을 도입하여 재구성을 수행합니다. 개별 프레임의 재구성 손실( reconstruction loss)을 활용하여 MAM 전제 과제를 해결하는 난이도를 판단하며, 이를 위해 교사 모델(teacher model)을 사용하여 프레임 단위 손실을 예측하고 어떤 프레임을 마스킹할 지 결정합니다.

- **Performance Highlights**: EH-MAM은 여러 최신 기준선(baselines) 대비 5%-10% 향상된 성능을 보이며, 저자원(low-resource) 음성 인식 및 SUPERB 벤치마크에서 효과적으로 유용한 컨텍스트를 포착하는 마스킹 영역을 분석합니다.



### Scalable Drift Monitoring in Medical Imaging AI (https://arxiv.org/abs/2410.13174)
- **What's New**: 이 논문에서는 인공지능(AI)을 의료 이미징에 통합하여 임상 진단의 발전을 이루었지만 모델 드리프트 관리와 장기적인 신뢰성을 보장하는 데 몇 가지 도전 과제가 발생한다는 점을 강조하고 있습니다. MMC+라는 확장된 프레임워크를 개발하여 이러한 문제에 대처하고 있습니다.

- **Technical Details**: MMC+는 CheXstray 프레임워크를 기반으로 하여 다중 모달 데이터 일치성을 이용한 실시간 드리프트 감지를 통해 의료 이미지 AI 모델을 위한 확장 가능한 드리프트 모니터링 솔루션을 제안합니다. 이 프레임워크는 다양한 데이터 스트림을 보다 강력하게 처리하고, MedImageInsight와 같은 기초 모델을 통합하여 고차원 이미지 임베딩을 지원하며, 불확실성 경계를 도입하여 동적 임상 환경에서 드리프트를 보다 잘 포착합니다.

- **Performance Highlights**: MMC+는 COVID-19 팬데믹 기간 동안 Massachusetts General Hospital의 실제 데이터를 통해 검증되었으며, 데이터의 중요한 변화 감지와 이를 모델 성능 변화와 연계하는 데 효과적입니다. 이러한 시스템은 성능 저하를 직접적으로 예측하지는 않지만, AI 시스템이 허용 가능한 성능 범위에서 이탈할 가능성을 조기에 경고하여 신속한 개입이 가능하도록 합니다.



### L1-Regularized ICA: A Novel Method for Analysis of Task-related fMRI Data (https://arxiv.org/abs/2410.13171)
Comments:
          29 pages, 9 figures, 4 tables. Python code is available. Please contact the corresponding author for the code

- **What's New**: 본 논문에서는 고차원 데이터로부터 적절한 특징을 추출하기 위한 새로운 독립 성분 분석(Independent Component Analysis, ICA) 방법을 제안합니다. 기존 ICA 방법의 해석 가능성 문제를 해결하기 위해 희소성(sparsity) 제약 조건을 도입하였습니다.

- **Technical Details**: 새로운 ICA 방법은 비용 함수(cost function)에 L1-정규화 항을 추가하여 sparsity를 고려합니다. 이 방법에서 비용 함수의 최소화는 convex 함수의 차분 알고리즘(difference of convex functions, DC algorithm)을 사용하여 수행됩니다. 또한, 본 방법은 합성 데이터 및 실제 기능적 자기공명영상(functional magnetic resonance imaging, fMRI) 데이터에 적용되어 검증됩니다.

- **Performance Highlights**: 기존의 MF 방법들과 비교했을 때, 제안된 방법은 fMRI 데이터에서 더욱 적절한 특징을 추출할 수 있는 가능성을 보여줍니다. 이는 특히 신경망 활동의 분석 및 뇌 기능 이해에 중요한 기여를 예상합니다.



### Continuous normalizing flows for lattice gauge theories (https://arxiv.org/abs/2410.13161)
- **What's New**: 이번 연구에서는 연속 정규화 흐름(continuous normalizing flows)의 새로운 구조를 제안합니다. 이 구조는 매트릭스 리 군(matrix Lie groups)을 위한 일반적인 형태로, 군 변환(group transformations)에 대해 동형성을 유지합니다.

- **Technical Details**: 제안된 구조는 이전 연구를 바탕으로 하며, 고차원 구조체의 대칭(symmetries)을 포함할 수 있는 유연성과 표현력을 제공합니다. 이 모델을 2차원 격자 게이지 이론(lattice gauge theories)에 적용하여 기본 개념을 증명했습니다.

- **Performance Highlights**: 실험 결과, 제안된 구조가 경쟁력 있는 성능을 보여 향후 격자 샘플링 작업(lattice sampling tasks)에서 유용할 가능성을 입증했습니다.



### Data Driven Environmental Awareness Using Wireless Signals for Efficient Spectrum Sharing (https://arxiv.org/abs/2410.13159)
- **What's New**: 무선 장비의 실내/외 분류를 위한 강력한 방법 제안. 이 방법은 기존의 분류 시스템의 한계를 극복하고, 새로운 데이터셋을 활용하여 정확성을 높임.

- **Technical Details**: Wi-Fi, cellular, GPS 데이터를 활용하여 DNN(Deep Neural Network) 모델을 개발함. 이 모델은 Indoor Near Window(INW), Indoor Interior(II), Outdoor(O)의 세 가지 클래스를 분류하며, 100% 정확도 향상을 위해 데이터 집계와 다수결(most voting) 기법을 사용함.

- **Performance Highlights**: DNN 모델이 기존의 ML 방법보다 성능이 우수하며, 새로운 환경에서 테스트하였을 때도 비교적 좋은 성과를 보임. 전체 데이터셋에서 100%의 정확도를 달성하며 강력한 일반화 능력을 나타냄.



### Learning Efficient Representations of Neutrino Telescope Events (https://arxiv.org/abs/2410.13148)
Comments:
          10 pages, 6 figures. Submitted to ICLR 2025

- **What's New**: 이 논문에서는 om2vec이라는 새로운 접근 방식을 소개하며, 이는 transformer 기반의 variational autoencoder(VAE)를 통해 중성미자 망막 사건을 효율적으로 표현하고 설명하는 잠재 표현(latent representation)을 학습합니다.

- **Technical Details**: om2vec은 입력 패턴 불량 데이터(PATD)를 학습하여 고정 크기의 조밀한 잠재 표현으로 변환하는데, 이는 고차원 데이터의 효율적인 분석과 처리를 용이하게 합니다. 논문에서는 데이터의 측면에서 transformer 레이어를 포함하여 기능 학습 능력을 향상시키며, encoder-decoder 구조를 통해 조작됩니다. 각 OM의 PATD는 동일한 길이의 일차원 시간 시퀀스로 형성되며, 각 요소는 특정 시간의 광자 충돌 수를 포함합니다.

- **Performance Highlights**: om2vec은 AGMM 접근 방식에 비해 더 많은 정보를 보존하며, 더 복잡한 PATD를 표현하는 데 뛰어난 유연성을 제공합니다. 이 방법은 최적화 실패를 피하고, AGMM보다 수 배 더 빠른 처리 속도를 자랑하며, GPU 가속화와 병렬 처리에서 더 큰 효익을 거두는 장점이 있습니다. 따라서 om2vec은 실험 데이터 수집 초기 단계에서 배치하기에 적합한 ‘one-size-fits-all’ 중성미자 사건 표현 학습기로 자리잡을 것입니다.



### Boosting Imperceptibility of Stable Diffusion-based Adversarial Examples Generation with Momentum (https://arxiv.org/abs/2410.13122)
Comments:
          10 pages, 12 figures. To be published in IEEE TPS 2024 Proceedings. Code available on GitHub: this https URL

- **What's New**: 본 논문은 네트워크 분류기를 효과적으로 혼란시킬 수 있는 적대적 예제를 생성하기 위한 새로운 프레임워크인 Stable Diffusion 기반의 Momentum Integrated Adversarial Examples (SD-MIAE)를 제안합니다. 이 방법은 고유한 클래스 라벨에 대한 시맨틱 유사성을 유지하면서 시각적으로 인지 불가능한 적대적 예제를 생성하는 데 중점을 둡니다.

- **Technical Details**: SD-MIAE는 두 가지 단계로 구성됩니다: (1) 초기 적대적 최적화 단계에서 토큰 임베딩을 수정하여 자연스러운 이미지를 생성하고, (2) 모멘텀 기반의 최적화 단계에서 적대적 perturbation을 정제합니다. 모멘텀을 도입함으로써, 최적화 과정에서의 안정성을 높이고, 고차원 잠재 공간에서의 이미지 생성을 통해 자연스러운 외관을 유지합니다.

- **Performance Highlights**: SD-MIAE는 79%의 높은 오분류율을 달성하여 최신 기법에 비해 35% 개선된 성능을 보이며, 적대적 perturbations의 비가시성과 원래 클래스 라벨에 대한 시맨틱 유사성을 유지하는 데 기여합니다.



### Distributional Matrix Completion via Nearest Neighbors in the Wasserstein Spac (https://arxiv.org/abs/2410.13112)
- **What's New**: 본 논문에서는 empirical distributions의 희소 관측 행렬을 고려하여, 관측된 항목 및 미관측 항목에 관련된 진짜 분포를 보간하는 distributional matrix completion 문제를 소개합니다. 전통적인 matrix completion에서 관측값이 스칼라 값을 가진 것과 달리, 본 연구는 분포를 다룹니다.

- **Technical Details**: 이 문제를 해결하기 위해 optimal transport(최적 수송)의 도구를 활용하여 분포 설정에서 가장 가까운 이웃 방법을 일반화합니다. 확률 분포에 대한 적절한 잠재적 요인 모델 하에, 제안된 방법이 Wasserstein norm(워서슈타인 노름)에서 분포를 회복한다는 것을 입증합니다.

- **Performance Highlights**: 시뮬레이션을 통해, 본 방법이 (i) 특정 항목의 관측된 샘플만 사용하는 것보다 더 나은 분포 추정치를 제공하고, (ii) 표준 편차 및 value-at-risk와 같은 분포량의 정확한 추정치를 산출하며, (iii) 이질적 노이즈를 본질적으로 지원함을 보여줍니다. 또한, 1차원 분포에서 Wasserstein barycenters(워서슈타인 바리센터)에 대한 새로운 비대칭 결과를 증명합니다.



### Contextual Bandits with Arm Request Costs and Delays (https://arxiv.org/abs/2410.13109)
- **What's New**: 이 연구에서는 맥락 밴딧 문제의 새로운 확장을 소개합니다. 새로운 팔(arm)을 요청할 때의 확률적 시간 지연과 관련 비용을 고려한 설정입니다. 이 모델에서 학습자는 여러 팔을 선택할 수 있으며, 각 선택은 1단위의 시간을 소요합니다. 이 문제는 반고르룸 비결정 과정(semi-Markov decision processes, SMDPs)의 특별한 케이스로 프레임화됩니다.

- **Technical Details**: 학습자는 새로운 결정 집합을 요청할 수 있으며, 이 요청은 무작위 비용이 발생하고, 이후 무작위 지연 후에 팔 및 그 맥락 정보가 제공됩니다. 학습자는 이 집합에서 임의의 팔의 부분 집합을 선택할 수 있습니다. 알고리즘 설계에는 Bellman 최적성 방정식이 활용되며, 이를 통해 최적 정책을 특성화하고, 회귀를 최소화하기 위해 온라인 팔 필터링 알고리즘이 개발됩니다. 이 알고리즘은 잘 정의된 변화량을 기반으로 하여 최적의 팔을 선택하고, 새로운 팔을 요청할 최적 시기를 결정합니다.

- **Performance Highlights**: 제안된 알고리즘의 성능은 시뮬레이션된 데이터와 영화 추천 데이터 세트를 통해 검증되었으며, 그 결과는 이론적 분석과 일관성을 보였습니다. 알고리즘의 회귀 상한은 기존 맥락 밴딧 문헌의 결과와 일치하는 것으로 분석되었습니다.



### A Little Human Data Goes A Long Way (https://arxiv.org/abs/2410.13098)
- **What's New**: NLP 시스템의 효율성을 높이기 위해, 인간 주석 데이터의 일부를 합성 데이터로 대체하는 방법을 연구하였으며, 90%까지 대체해도 성능 저하가 미미하지만 마지막 10% 대체 시에는 성능이 크게 떨어진다는 중요한 발견을 했습니다.

- **Technical Details**: 합성 데이터 생성 과정을 통해 데이터 포인트 수를 일정하게 유지하며 인간 생성 데이터 비율을 점진적으로 증가시켜 성능을 비교하였습니다. 사용하는 데이터셋은 총 8개로 Fact Verification (FV) 및 Question Answering (QA) 태스크에 대해 실험하였습니다. 평가 지표로는 정확도, Exact Match, String Inclusion, BLEU, ROUGE-1, BERTScore를 사용하였습니다.

- **Performance Highlights**: 완전히 합성 데이터로 훈련된 FV 및 QA 시스템은 최소 125개의 인간 데이터 포인트를 추가할 경우 성능이 현저히 개선되며, 작은 비율의 인간 데이터가 큰 가치를 지닐 수 있다는 것을 발견했습니다. 추가적인 인간 데이터를 통한 성능 향상은 200 포인트의 인간 데이터로 가능하며, 이는 수량적으로 더 많은 합성 데이터 포인트에 비해 비용 효율적이라는 것을 보여줍니다.



### Reverse-Engineering the Reader (https://arxiv.org/abs/2410.13086)
- **What's New**: 이 연구는 기존의 언어 모델을 인간의 심리 측정 데이터에 맞춰 최적화하는 새로운 방법론을 제시합니다. 이를 통해 언어 처리 시스템의 이해를 높이고자 합니다.

- **Technical Details**: 연구진은 언어 모델이 특정 언어 단위의 읽기 시간을 예측하는 능력을 향상시키기 위해 서프라이절 이론(surprisal theory)을 기반으로 한 새로운 정렬 기법을 사용합니다. 모델의 파라미터를 조정하여 읽기 시간을 예측하는 선형 회귀의 계수를 최적화합니다.

- **Performance Highlights**: 제안된 기법은 여러 모델 크기와 데이터 세트에서 언어 모델의 심리 측정 예측력을 향상시키는 것으로 나타났습니다. 그러나 심리 측정 예측력과 후속 자연어 처리(NLP) 작업 성능 간에 반비례 관계가 발견되었습니다.



### Two-Timescale Linear Stochastic Approximation: Constant Stepsizes Go a Long Way (https://arxiv.org/abs/2410.13067)
- **What's New**: 이번 연구에서는 두 시계열 확률적 근사(TTSA)에 대한 새로운 접근법인 상수 스텝 사이즈(Constant Stepsize) 스킴을 Markov 과정에 기반하여 조사하였습니다. 연구 결과, 일반적인 공동 정적 분포에 수렴하는 것을 입증하였습니다.

- **Technical Details**: 연구는 Markovian 노이즈가 포함된 Linear TTSA를 다루며, 두 개의 상수 스텝 사이즈가 있는 경우(α < β) 비대칭 제어의 수렴 속도 및 편향과 분산을 수학적으로 유도하였습니다. 편향은 선형적으로 증가하며 분산은 각각의 스텝 사이즈에만 의존합니다.

- **Performance Highlights**: 이번 결과는 기존 연구와 달리 추가 가정 없이 TTSA의 비대칭 제어에서 MSE(Mean Squared Error) 경계를 개선하는 방법을 제시합니다. 그 결과는 두 반복(iterate)의 MSE 경계를 O(β^4 + 1/t)로 개선할 수 있습니다.



### Large data limits and scaling laws for tSNE (https://arxiv.org/abs/2410.13063)
- **What's New**: 이 연구는 t-distributed Stochastic Neighbor Embedding (tSNE) 알고리즘의 대량 데이터의 비대칭적 성질을 다루었으며, tSNE 목적 함수의 적합한 연속 한계를 식별했습니다. 이를 통해 tSNE 알고리즘의 원래 임베딩이 n이 무한으로 갈 때 일관된 한계를 가질 수 없음을 보여주고, 매력 에너지의 비대칭적 감소를 완화하는 재조정된 모델을 제안했습니다.

- **Technical Details**: tSNE는 비선형 차원 축소를 위해 널리 사용되는 알고리즘으로, 서로 다른 엔지니어링 결정들이 상호작용하여 해석하기 어려운 결과를 초래합니다. 이 연구는 tSNE의 최적화 문제에 대한 이론적 기초를 제공하며, 원래 알고리즘이 의존하는 매력과 반발의 비선형 에너지를 최소화하는 방법을 분석합니다. 연구에서는 Kernel 기반의 반발과 점차적으로 사라지는 Laplacian 유형의 정규화기를 결합한 tSNE 목적 함수를 제안했습니다.

- **Performance Highlights**: tSNE는 의학 이미지, 유전체학, 해양 생물학 등 다양한 분야에서 사용되어 높은 인용 수(2024년 Google Scholar 기준 47000회 이상)를 기록한 차원 축소 알고리즘입니다. 이 논문은 기존 tSNE가 직면한 문제들을 명확히 하고, 최적의 차원 축소 임베딩을 찾기 위한 보다 안정적이고 일관된 결과를 제공할 수 있는 새로운 프레임워크를 제시합니다.



### Optimal Transport for Probabilistic Circuits (https://arxiv.org/abs/2410.13061)
- **What's New**: 이 논문에서는 확률 회로(Probabilistic Circuits, PCs) 간의 Wasserstein distance를 계산하기 위한 최적 운송 프레임워크를 도입합니다. 기존에 알려진 알고리즘은 있었으나, 확률 회로로 정의된 분포 간의 Wasserstein distance를 계산하는 방법은 처음으로 제안됩니다.

- **Technical Details**: Wasserstein-type distance를 도입하여 연관된 최적 운송 문제의 coupling measure를 확률 회로로 제한합니다. 이 거리를 계산하기 위해 작은 선형 프로그램(solution to linear programming problems) 문제를 해결하는 알고리즘을 개발하였으며, 이 문제의 해결조건을 제시합니다. 또한, 실험적 데이터를 기반으로 한 PC와의 Wasserstein distance를 최소화하기 위한 효율적인 반복 알고리즘을 제공합니다.

- **Performance Highlights**: 제안된 알고리즘은 두 확률 회로 간의 Wasserstein-type distance를 정확하고 효율적으로 계산할 수 있는 기능을 갖추고 있으며, 이를 기존 방법론과 비교하여 실험적으로 우수한 성능을 보입니다.



### Hypothesis Testing the Circuit Hypothesis in LLMs (https://arxiv.org/abs/2410.13032)
Comments:
          Code available here: this https URL

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 내부 작동 방식을 이해하기 위해 '회로(circuits)'라는 개념을 실험적으로 조사했습니다. 특히 회로가 LLM의 능력을 구현하는지를 테스트하기 위한 기준과 가설 검정 방식이 소개되었습니다.

- **Technical Details**: 연구에서는 다음 세 가지 이상적인 속성을 정의했습니다: 1) 메커니즘 보존(Mechanism Preservation): 회로의 성능이 원래 모델과 일치해야 함. 2) 메커니즘 지역화(Mechanism Localization): 회로를 제거하면 해당 작업을 수행하는 능력이 사라져야 함. 3) 최소성(Minimality): 회로에 중복된 엣지가 없어야 함. 이 속성에 따라 6개의 회로를 평가했습니다.

- **Performance Highlights**: 합성 회로(synthetic circuits)는 이상적인 속성과 잘 일치하는 반면, 발견된 회로는 모든 속성에 엄격히 부합하지 않았습니다. 그러나 특정 발견된 회로는 유명한 작업을 수행하는 데 중요한 역할을 했으며, 이들 회로는 이상적인 특성에 근접하게 개선될 수 있는 가능성을 보여주었습니다.



### Sensitivity of Generative VLMs to Semantically and Lexically Altered Prompts (https://arxiv.org/abs/2410.13030)
- **What's New**: 본 논문은 generative vision-language 모델(VLM)의 프롬프트에서의 어휘적 및 의미적 변화에 대한 민감성을 평가합니다. SugarCrepe++ 데이터셋을 사용하여 이러한 모델들이 프롬프트의 사소한 변화에 어떤 영향을 받는지를 분석합니다.

- **Technical Details**: 이 연구는 BLIP, BakLLaVA 및 GPT-4o와 같은 generative VLMs의 어휘 및 의미 변화 이해 능력을 평가합니다. SugarCrepe++ 데이터셋에서는 두 개의 긍정적인 캡션(P1, P2)과 하나의 부정적인 캡션(N)을 포함하여, 어휘적으로 다르지만 의미적으로 유사한 캡션을 제공합니다.

- **Performance Highlights**: 실험 결과, BakLLaVA와 GPT-4o 모두 입력 프롬프트의 약간의 변화에 대해 높은 민감성을 보였으며, 동일한 프롬프트에서 옵션의 순서를 변경하는 것만으로도 성능에 큰 차이를 보였습니다. 또한, 서로 다른 VLMs 간의 일관성이 부족하여 결과의 일관성을 높이기 위한 추가 연구가 필요함을 보여줍니다.



### When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems (https://arxiv.org/abs/2410.13029)
Comments:
          11 pages, 7 figures, 2 tables

- **What's New**: 본 논문은 대형 언어 모델(GPT 모델)이 해결 불가능한 수학 단어 문제에 적절하게 대응할 수 있는지 평가하고, 이러한 모델들의 개선 방안을 모색한다. 특히, 모델들이 정답이 없을 경우 어떻게 'abstain' (응답 거부) 할 수 있는지를 연구하며, 이 과정을 향상시키기 위한 프롬프트 기술을 탐구한다.

- **Technical Details**: 연구는 Unanswerable Word Math Problem (UWMP) 데이터셋을 활용하였으며, 각 문제에 대해 'abstention' (응답 거부), 정확도(correctness), 신뢰도(confidence)의 세 가지 요소를 통합한 평가 지표를 도입하였다. 실험을 통해 다양한 프롬프트 기술을 적용하여 모델의 응답 행동을 분석하고, 해결 불가능한 질문에 대한 모델의 경향성을 분석하였다.

- **Performance Highlights**: 실험 결과, GPT 모델들은 해결이 불가능한 문제에 대해 잘못된 정보를 스스로 생성하는 경향이 있으며, 결과적으로 이러한 모델들은 수학 문제 해결에 있어 불확실성과 복잡한 추론을 효과적으로 관리하지 못한다는 점이 밝혀졌다. 이는 향후 모델들이 더 나은 관리와 결정을 내릴 수 있도록 개선이 필요함을 강조한다.



### Geometric Trajectory Diffusion Models (https://arxiv.org/abs/2410.13027)
Comments:
          Published at NeurIPS 2024. 29 pages, 10 figures

- **What's New**: 이번 연구에서는 3D 기하학적 궤적을 모델링하기 위해 최초의 diffusion model인 GeoTDM(Geometric Trajectory Diffusion Model)을 제안합니다. 이는 기존의 정적 구조에 대응하는 방법을 넘어, 물리적 시스템이 본질적으로 동적이라는 사실을 반영합니다.

- **Technical Details**: GeoTDM은 물리적 대칭과 동역학의 시간적 상관 관계를 포함한 복잡한 공간 상호작용을 포착해야 하는 도전에 직면했습니다. 이에 SE(3)-equivariant spatial convolution과 temporal attention을 활용한 새로운 전이 커널을 개발하여 적절한 대칭을 가진 밀도를 생성합니다. 또한, 조건부 생성을 위한 표현력 있는 궤적 분포를 유도하기 위해 일반화된 학습 가능한 기하학적 사전(geometric prior)을 도입했습니다.

- **Performance Highlights**: 다양한 시나리오에서 진행한 실험 결과, GeoTDM은 물리적 시뮬레이션, 분자 동역학, 보행자 운동을 포함하여 비조건부 및 조건부 생성을 통해 사실적인 기하 궤적을 생성할 수 있으며, 품질이显著 향상됨을 보여주었습니다.



### LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks (https://arxiv.org/abs/2410.13025)
Comments:
          9 pages plus references and appendices

- **What's New**: 이 논문은 Low-Rank Adaptation (LoRA) 기법을 사용하여 여러 LoRA 모듈을 통합하여 기술 조합(skill composition)을 구현하는 방법을 연구합니다. 이 방식은 특정 기술과 지식이 필요한 작업에서 이전 모델 및 데이터 병합 기법보다 우수한 성능을 보입니다.

- **Technical Details**: 주요 기여는 LoRA의 연결(concatenation)을 최적으로 평균화하여 서로 다른 기술로 개별 훈련된 LoRA를 병합하는 새로운 방법인 Learnable Concatenation (CAT)을 제안하는 것입니다. 이는 모델의 일부 층에서 저랭크 업데이트를 추가하여 진행됩니다.

- **Performance Highlights**: CAT는 수학 문제 해결에서 기존 방법에 비해 평균 43% 및 12% 향상을 보이며, LLM의 프롬프트 형식 변화에 대한 견고성도 개선합니다. 본 연구는 기술 조합 작업을 해결하기 위한 효율적인 방법으로 모델 병합을 지지합니다.



### Learning Representations for Reasoning: Generalizing Across Diverse Structures (https://arxiv.org/abs/2410.13018)
Comments:
          PhD thesis

- **What's New**: 이 논문은 인공지능 분야에서의 추론의 중요성과 관련하여, 기존의 지식 구조 및 쿼리 구조를 초월하는 일반화 알고리즘을 제안합니다. 또한, 구조적 데이터에서 기계 학습 개발을 가속화하기 위한 시스템을 구축했습니다.

- **Technical Details**: 제안된 모델 NBFNet은 전통적인 경로 기반(path-based) 방법과 동적 프로그래밍(dynamic programming)을 결합하여 새로운 엔티티(entity) 및 관계(relation) 어휘를 사용한 지식 그래프의 미지의 부분에 대한 유도 일반화를 실현합니다. A*Net은 NBFNet의 확장형으로, 수백만 개 규모의 지식 그래프에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: NBFNet은 기존의 최신 방법들에 비해 모든 설정에서 평균 18%의 성능 향상을 이루었으며, 특히 지식 그래프 완성(HITS@1) 및 유도 관계 예측(HITS@10)에서 각각 22%의 성능 개선을 보여줍니다.



### LEGAL-UQA: A Low-Resource Urdu-English Dataset for Legal Question Answering (https://arxiv.org/abs/2410.13013)
Comments:
          8 pages

- **What's New**: LEGAL-UQA는 파키스탄 헌법에서 유래한 첫 번째 우르두 법률 질문-답변(QA) 데이터셋을 소개합니다. 이 데이터셋은 619개의 질문-답변 쌍을 포함하며, 법률 기사의 컨텍스트도 포함되어 있어 낮은 자원 언어의 도메인 특화된 NLP 자원의 필요성을 해결합니다.

- **Technical Details**: 데이터셋 생성 과정은 OCR 추출, 수동 수정 및 GPT-4를 활용한 QA 쌍의 번역 및 생성으로 구성됩니다. LEGAL-UQA의 성능을 평가하기 위해 최신의 일반 언어 및 임베딩 모델을 실험하였으며, Claude-3.5-Sonnet 모델이 인간 평가에서 99.19%의 정확도를 달성하였습니다. 또한, mt5-large-UQA-1.0 모델을 미세 조정하여 다국어 모델을 전문 분야에 적용하는 데 따른 도전 과제를 강조하였습니다.

- **Performance Highlights**: OpenAI의 text-embedding-3-large는 Mistral의 mistral-embed 보다 더 나은 검색 성능을 보였습니다. LEGAL-UQA는 글로벌 NLP 발전과 현지화된 응용 프로그램 간의 격차를 해소하며, 파키스탄 내 법률 정보 접근성을 개선하는 기반을 마련합니다.



### Long-Tailed Backdoor Attack Using Dynamic Data Augmentation Operations (https://arxiv.org/abs/2410.12955)
- **What's New**: 본 논문은 긴 꼬리(long-tailed) 데이터셋에 대한 백도어 공격(backdoor attack)을 처음으로 탐구합니다. 기존의 백도어 공격은 주로 균형 잡힌 데이터셋에 초점을 맞추었으며, 이로 인해 실제 환경에서 발생하는 불균형 데이터 문제를 간과하고 있었습니다.

- **Technical Details**: 제안된 방법인 D$^2$AO(Dynamic Data Augmentation Operation)는 클래스, 샘플 유형(클린 vs. 백도어), 그리고 샘플 특징에 따라 동적으로 다양하고 적절한 데이터 증강(data augmentation) 연산을 선택합니다. 이를 통해 백도어 샘플과 클린 샘플의 불균형 문제를 해결하고, 데이터 증강에 적응할 수 있는 트리거 생성기를 개발하였습니다.

- **Performance Highlights**: CIFAR10-LT 및 CIFAR100-LT와 같은 두 개의 긴 꼬리 벤치마크에서 폭넓은 실험을 수행하였으며, 제안된 방법은 기존의 백도어 공격 방법과 비교하여 상태-of-the-art 공격 성능을 달성하면서 클린 정확도(clean accuracy)를 유지하였습니다.



### Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging (https://arxiv.org/abs/2410.12937)
Comments:
          Findings of EMNLP 2024

- **What's New**: 이 연구는 기존의 언어 모델에 새로운 기술을 추가하는 방식인 "parallel train then merge" (PTM) 접근법을 소개합니다. PTM은 여러 기술을 모델에 효율적으로 추가 가능하게 하며, 기존 기술을 잊게 하지 않고도 새로운 기술을 통합할 수 있는 장점이 있습니다.

- **Technical Details**: CFT (continued finetuning), RT (retraining), PTM은 기존 모델에 새로운 기술을 추가하기 위한 세 가지 방법입니다. PTM은 새로운 데이터에 대해서만 개별적으로 훈련하여 모델 파라미터를 병합하는 방식으로, 기존 기술을 보존하는 동시에 새로운 기술도 효과적으로 학습할 수 있습니다. 실험은 과학 문헌 이해, 코딩 및 안전 관련 요청 거부에서 진행되었습니다.

- **Performance Highlights**: PTM은 기존의 CFT보다 50–95% 효율적으로 훈련할 수 있으며, 원래 모델의 일반 기술을 거의 모두 유지하면서 새로운 기술에서도 유사한 성능을 달성합니다. 또한, PTM은 안전 관련 거부 능력을 개선하며, 전반적인 성능을 유지할 수 있습니다. 이 연구의 결과는 PTM이 CFT보다 효과적인 옵션이라는 것을 보여줍니다.



### Quantum Boltzmann machine learning of ground-state energies (https://arxiv.org/abs/2410.12935)
Comments:
          7 pages, 1 figure, Supplementary material available as 'Ancillary Files'

- **What's New**: 본 연구에서는 Hamiltonians의 바닥 상태 에너지(ground-state energy)를 추정하기 위해 양자 볼츠만 머신(quantum Boltzmann machines)의 성능을 분석합니다. 이는 현재까지 깊이 탐구되지 않은 접근법으로, 매개변수화된 열 상태(parameterized thermal states)를 기반으로 하며, barren-plateau 문제(barren-plateau problem)의 영향을 받지 않습니다.

- **Technical Details**: 양자-고전(hybrid quantum-classical) 알고리즘을 상세히 설명하고, 진동 함수에서 매개변수 공간(parameter space)을 최적화하기 위해 $\\varepsilon$ - 근사 정적점(approximate stationary point)에 수렴함을 엄밀하게 증명합니다. 이 과정에서 사용하는 매개변수화된 열 상태 샘플 수는 $\\varepsilon^{-1}$, 매개변수 수, 최적화되는 Hamiltonian의 노름(norm)의 다항식(polynomial)입니다. 본 알고리즘은 고전적 샘플링(classical sampling), 해밀토니안 시뮬레이션(Hamiltonian simulation), 하다마드 테스트(Hadamard test)를 결합한 혁신적인 양자 회로 구조를 통해 에너지 함수의 기울기(gradient)를 효율적으로 추정합니다.

- **Performance Highlights**: 본 연구는 향후 양자 볼츠만 머신 학습에 있어 주요한 장애물을 극복할 가능성을 보여줍니다. 또한, 에너지 함수의 기울기와 헤세 행렬(Hessian)의 계산, 그리고 수렴 분석에 사용되는 후자의 행렬 요소의 상한이 지원합니다.



### Credal Two-Sample Tests of Epistemic Ignoranc (https://arxiv.org/abs/2410.12921)
Comments:
          39 pages

- **What's New**: 새로운 가설 검정 프레임워크인 credal two-sample testing을 도입하여, 모델러의 부분적인 무지에서 오는 epistemic uncertainty를 고려한 credal set(확률 분포의 볼록 집합) 비교 방법을 제시합니다.

- **Technical Details**: credal two-sample testing은 기존 두 샘플 테스트를 일반화하며, equality(동등성), inclusion(포함성), intersection(교차), mutual exclusivity(상호 배타성) 등의 가설을 제시합니다. 이 접근법은 nuisance parameters(폐해 매개변수)를 포함한 두 샘플 테스트로 형식화되며, 커널 기반의 비모수적 테스트를 개발하여 Type I 및 Type II 오류를 효과적으로 제어합니다.

- **Performance Highlights**: 제안된 방법은 기존 방법을 초월하여, 더 강력하고 신뢰할 수 있는 결론을 도출합니다. 특히, permutation 기반 접근 방식으로, 비모수적 통계 분석에서 강력한 특정 효과를 보입니다.



### AT-RAG: An Adaptive RAG Model Enhancing Query Efficiency with Topic Filtering and Iterative Reasoning (https://arxiv.org/abs/2410.12886)
- **What's New**: AT-RAG라는 새로운 멀티스텝 RAG 모델을 제안하여, 복잡한 다중 단계 쿼리를 보다 효율적으로 처리하는 방법을 소개합니다.

- **Technical Details**: AT-RAG는 BERTopic을 활용하여 쿼리의 주제를 동적으로 할당함으로써 문서 검색 및 추론 과정의 정확성과 효율성을 향상시킵니다. 이 모델은 Chain-of-Thought (CoT) 추론을 통합하여 반복적인 문서 검색 및 추론을 가능하게 합니다.

- **Performance Highlights**: AT-RAG는 기존 RAG 모델 대비 Accuracy, Completeness, Relevance에서 현저한 개선을 보였으며, 특히 의료 QA와 같은 복잡한 도메인-specific 문제 해결에 적합합니다. 모델은 다양한 benchmark dataset에서 높은 성능을 발휘하였고, 검색 시간을 줄이면서 높은 정밀도를 유지합니다.



### Scaling Laws for Multilingual Language Models (https://arxiv.org/abs/2410.12883)
- **What's New**: 본 연구에서는 다국어 데이터로 훈련된 일반 목적의 디코더 전용 언어 모델을 위한 새로운 스케일링 법칙을 제안합니다. 각 언어의 성능을 개별적으로 분석하기 어려운 문제를 다루기 위해, 우리는 개별 언어 대신 언어 가족에 초점을 맞추고, 각 언어 가족의 테스트 교차 엔트로피 손실(test cross-entropy loss)은 혼합 내 다른 언어와 무관하게 샘플링 비율(sampling ratio)에 의해 결정된다는 가설을 검증했습니다.

- **Technical Details**: 제안한 스케일링 법칙은 테스트 교차 엔트로피 손실을 모델 크기(model size), 데이터셋 크기(dataset size), 샘플링 비율(sampling ratios)과 연결하는 전력 법칙(power-law relationship)을 도출합니다. 이를 통해 다양한 조합에 대한 성능 예측이 가능해졌으며, 훈련 혼합 내 언어 가족의 최적 샘플링 비율을 도출할 수 있게 되었습니다. 우리는 23개 언어, 5개 언어 가족을 대상으로 100개 이상의 모델을 훈련하여 대규모 실증 연구를 수행했습니다.

- **Performance Highlights**: 실험 결과, 작은 모델(85M 파라미터)에서 도출한 최적 샘플링 비율이 수십 배 큰 모델(1.2B 파라미터)에도 효과적으로 일반화됨을 보여주었습니다. 이는 리소스를 효율적으로 사용할 수 있는 다국어 언어 모델 훈련을 위한 접근 방식을 제공합니다.



### Towards More Effective Table-to-Text Generation: Assessing In-Context Learning and Self-Evaluation with Open-Source Models (https://arxiv.org/abs/2410.12878)
Comments:
          15 pages

- **What's New**: 이 연구는 자연어 처리의 핵심 작업인 테이블-텍스트 생성(table-to-text generation)에 대해, 다양한 in-context learning 전략의 효과를 평가합니다. 특히, 모델에 주어진 예제가 성능에 미치는 영향을 조사하고, 실제 애플리케이션을 기반으로 한 사례를 제공합니다.

- **Technical Details**: 모델은 zero-shot, single-shot, few-shot 프롬프트를 사용하여 테이블 데이터에서 내러티브 텍스트로 전환합니다. 이 연구에서는 두 개의 벤치마크 데이터셋인 WikiBio와 ToTTo에서 실험이 수행되었고, Llama 3와 Phi-3 모델을 사용하여 결과를 비교했습니다. 또한, GPT-4를 사용하여 초기 프롬프트를 생성하고, 이를 기반으로 최적화를 진행하였습니다.

- **Performance Highlights**: 예제를 제공함으로써 테이블-텍스트 생성의 성능이 크게 향상되었습니다. LLM 자가 평가 방법은 아직 인간의 판단과 일치도가 개선되어야 하지만, overall 성능 개선을 확인할 수 있었습니다.



### Improving Instruction-Following in Language Models through Activation Steering (https://arxiv.org/abs/2410.12877)
- **What's New**: 이 논문에서는 언어 모델(LLM)의 지침 따르기 능력을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이는 지침에 따라 모델의 동작을 조정하기 위해 지침별 벡터 표현을 파생하는 내용을 다룹니다.

- **Technical Details**: 이 연구는 입력의 지침이 없는 경우와 있는 경우의 활성화(activation)의 차이를 기반으로 벡터 표현을 계산하여 모델의 출력을 조작하는 방식입니다. 사용된 활성화 벡터는 출력 형식, 길이, 특정 단어 포함 여부 등 여러 조건을 모델이 준수하도록 유도합니다.

- **Performance Highlights**: 4개의 서로 다른 모델을 대상으로 한 실험을 통해, 이 방법이 지침을 명시적으로 제공하지 않아도 모델이 제약사항을 따르도록 도와주고, 지침이 있을 때도 성능을 향상시킬 수 있음을 보여주었습니다. 또한, 여러 지침을 동시에 적용할 수 있다는 것이 확인되었습니다.



### In-context KV-Cache Eviction for LLMs via Attention-Ga (https://arxiv.org/abs/2410.12876)
- **What's New**: 본 논문은 Attention-Gate라는 매개변수화된 KV-Cache 제거 메커니즘을 도입하여 비효율적인 기존 제거 전략의 한계를 극복하려고 합니다. 이는 입력된 전체 컨텍스트를 기준으로 각 토큰의 제거 플래그를 생성하여 효율적인 in-context eviction을 실현합니다.

- **Technical Details**: Attention-Gate(AG)는 모델 내의 self-attention 레이어 전방에 위치하여, 입력된 토큰 특징 시퀀스를 처리하여 각 토큰에 대한 제거 플래그를 생성합니다. AG는 사전 훈련된 대형 언어 모델에 무리 없이 통합될 수 있으며, 최소한의 컴퓨팅 및 메모리 오버헤드를 가지면서 효율적입니다.

- **Performance Highlights**: 효율적인 지속적 사전 훈련(CPT) 후에, 기존의 훈련 없는 제거 전략보다 더 높은 평균 정확도를 달성하며 더 많은 토큰을 제거할 수 있음을 증명합니다. Supervised fine-tuning(SFT)에서는 LoRA로 미세 조정된 LLM보다 성능이 우수하며, RTE 데이터셋에서 13.9%의 정확도 향상과 62.8%의 토큰 제거를 달성하여 중복 토큰의 효과적인 제거가 성능을 개선할 수 있음을 나타냅니다.



### On Debiasing Text Embeddings Through Context Injection (https://arxiv.org/abs/2410.12874)
- **What's New**: 이 논문에서는 텍스트 임베딩 모델에서의 편향(bias)을 정량화하고, 그로부터 방지 성능을 평가하기 위해 19개의 임베딩 모델을 체계적으로 분석하였습니다. 최신 컨텍스트 인젝션(context injection) 기법을 활용하여 이들 모델의 편향을 줄이는 새로운 알고리즘을 제안합니다.

- **Technical Details**: 저자들은 두 가지 기존 기법인 기하학적 투영(geometric projection)과 WEAT(Word Embedding Association Test)를 수정하여 19개 임베딩 모델의 편향을 정량화합니다. 각 모델은 서로 다른 강도와 부분에서 편향(예: 성별, 나이)에 따라 평가받습니다. 또한 컨텍스트를 주입하여 편향을 감소시키는 방법론을 이용해 모델의 반응성을 측정합니다.

- **Performance Highlights**: 결과적으로 성능이 높은 임베딩 모델은 일반적으로 더 많은 편향을 캡처하는 경향이 있지만, 컨텍스트를 포함할 경우 편향을 줄이는 데 더 잘 대응한다고 밝혀졌습니다. 본 연구에서 제안하는 새로운 알고리즘은 동적으로 선택된 k 값에 대해 효과적인 검색 결과를 제공할 수 있습니다.



### Beyond Right and Wrong: Mitigating Cold Start in Knowledge Tracing Using Large Language Model and Option Weigh (https://arxiv.org/abs/2410.12872)
Comments:
          11 pages

- **What's New**: 이 논문에서는 LOKT 모델을 소개하여 Knowledge Tracing (KT)의 콜드 스타트 문제를 해결합니다. LOKT는 대규모 언어모델(LLM)을 사용하여 적은 이전 데이터로도 학습자의 지식 상태를 추적하고 예측할 수 있는 방법론을 제시합니다.

- **Technical Details**: LOKT 모델은 전통적인 KT 모델에 옵션 가중치를 통합하여 단순한 정답/오답 분류를 넘어 학습자의 다양한 잘못된 응답을 분석합니다. 이를 통해 LLM이 언어 기반 정량적 정보를 활용하여 학습자의 이해도를 보다 정확하게 평가할 수 있도록 합니다.

- **Performance Highlights**: 5개의 공공 데이터셋을 사용한 실험에서 LOKT 모델은 이른 단계의 개인화 학습 도구를 지원하며, '학습자 콜드 스타트'와 '시스템 콜드 스타트' 상황에서도 높은 예측 정확도를 유지하는 것을 보여주었습니다.



### AI-Driven Autonomous Control of Proton-Boron Fusion Reactors Using Backpropagation Neural Networks (https://arxiv.org/abs/2410.12871)
- **What's New**: 본 연구는 프로톤-붕소(p-11B) 핵융합로에서 핵심 파라미터를 자율적으로 제어하기 위해 역전파 기반 신경망을 이용한 새로운 접근 방식을 제안합니다. 이 방법은 물리적 데이터를 기반으로 실시간 피드백과 학습을 통해 변화하는 플라즈마 조건에 적응하는 기능을 제공합니다.

- **Technical Details**: 제안된 AI 기반 제어 시스템은 역전파(Backpropagation)로 훈련된 심층 신경망(Deep Neural Network, DNN)을 활용하여 실시간으로 플라즈마 조건을 최적화합니다. 이 시스템은 플라즈마의 상태를 동적으로 조정하여 동적이고 비선형적인 고온 플라즈마를 안정적으로 유지하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 이 연구의 AI 시스템은 실제 데이터에서 지속적으로 학습함으로써 플라즈마 안정성을 크게 향상시키고 에너지 효율성을 최적화하며, 실제 지속 가능한 핵융합 에너지로의 경로를 가속화할 가능성을 제시합니다.



### Skill Learning Using Process Mining for Large Language Model Plan Generation (https://arxiv.org/abs/2410.12870)
Comments:
          12 pages, 5 figures, 2 tables, accepted at ICPM 2024'

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 계획 생성을 개선하기 위해 프로세스 마이닝( process mining ) 기법을 통합한 새로운 기술 학습 접근 방식을 소개합니다. 이 접근 방식은 계획 생성 과정의 효율성과 해석 가능성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 텍스트 베이스 LLM 플래너가 생성한 단순 시퀀스 대신, 프로세스 모델을 사용하여 구조화된 제어 흐름을 만들고 이를 통해 플래너의 능력을 향상시키는 방법을 제안합니다. 새로운 기술 학습 프레임워크에서는 Inductive Miner 알고리즘을 사용하여 일반적인 프로세스 모델을 추출합니다.

- **Performance Highlights**: 실험 결과, 제안한 기술 검색 방법이 특정 조건에서 기존의 정확도 기준을 초과하는 것으로 나타났으며, 유연한 기술 발견과 병렬 실행을 지원하여 성능이 향상되었습니다.



### Language Model Preference Evaluation with Multiple Weak Evaluators (https://arxiv.org/abs/2410.12869)
- **What's New**: 이 논문에서는 효율적인 평가 방식의 필요성을 강조하며 신뢰성 있는 LLM(대규모 언어 모델) 출력 평가를 위한 새로운 방법론인 GED(Preference Graph Ensemble and Denoise)를 소개합니다.

- **Technical Details**: GED는 두 가지 주요 단계로 구성됩니다: (1) 여러 LLM의 평가 결과를 통합하여 단일 preference graph(선호 그래프)를 만드는 graph ensemble과 (2) 반복적 패턴과 불일치를 제거하여 방향 비순환 그래프(DAG) 구조를 보장하는 graph denoising입니다.

- **Performance Highlights**: GED는 실험 결과에서 10개 벤치마크 데이터셋을 통해 기존 방법들보다 우수한 성능을 보였으며, 예를 들어, 응답 선택 작업에서 평균 4.51% 향상을 기록했습니다. GED는 약한 평가자(combiner) 조합을 통해 강한 평가자보다 뛰어난 성능을 보여, 평가 신뢰성을 높이고 모델 성능을 향상시키는 능력을 입증했습니다.



### IMAS: A Comprehensive Agentic Approach to Rural Healthcare Delivery (https://arxiv.org/abs/2410.12868)
- **What's New**: COVID-19 이후, 농촌 지역의 의료 접근성 문제 해결을 위한 첨단 의료 보조 시스템(IMAS) 제안

- **Technical Details**: IMAS는 Large Language Models (LLMs)와 다섯 가지 주요 구성 요소(번역, 의료 복잡성 평가, 전문가 네트워크 통합, 최종 의료 조언 생성, 응답 단순화)로 구성되어 있습니다.

- **Performance Highlights**: IMAS는 MedQA, PubMedQA, JAMA 데이터셋을 통해 효과성을 입증하였으며, 특히 저소득 및 정보 소외 지역사회의 의료 근로자들에게 더 쉽게 접근할 수 있도록 지원합니다.



### Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings (https://arxiv.org/abs/2410.12866)
Comments:
          Preprint V1 with 10 pages main text

- **What's New**: 최근 뇌-컴퓨터 인터페이스(BCI)의 발전으로 인해 두개내(recordings)에서 음조(lexical tones)를 해독하는 것이 가능해졌습니다. 이는 언어 손상으로 인해 의사소통 능력이 제한된 사람들에게 도움을 줄 수 있는 잠재력을 제공합니다. 하지만 생리적 및 기기적 요소로 인해 발생하는 데이터 이질성(data heterogeneity)은 통합적인 뇌 음조 해독에 상당한 도전 과제가 됩니다.

- **Technical Details**: 이 논문에서는 H2DiLR(Homogeneity-Heterogeneity Disentangled Learning for neural Representations)이라는 새로운 프레임워크를 도입하여, 여러 피험자의 두개내 기록에서 동질성과 이질성을 분리하고 학습합니다. 이 연구에서는 407개의 음절(syllables)을 포함하는 중국어 재료를 읽는 여러 참가자로부터 스테레오전자뇌전도(sEEG) 데이터를 수집했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 H2DiLR은 기존의 이질적인 해독 접근 방식보다 현저히 우수한 성능을 보임을 입증했습니다. 또한 H2DiLR이 신경 표현 학습 과정에서 동질성과 이질성을 효과적으로 포착함을 실증적으로 확인하였습니다.



### ELF-Gym: Evaluating Large Language Models Generated Features for Tabular Prediction (https://arxiv.org/abs/2410.12865)
- **What's New**: ELF-Gym 프레임워크를 통해 LLM이 생성한 feature의 품질을 정량적으로 평가하는 새로운 방법론을 제시합니다.

- **Technical Details**: ELF-Gym은 Kaggle 대회에서 수집한 251개의 'golden' features를 기준으로 LLMs의 feature 엔지니어링 능력을 평가합니다. 평가 과정에서 LLM이 생성한 features의 다운스트림 모델 성능과 전문가가 제작한 features와의 의미적, 기능적 유사성을 측정합니다.

- **Performance Highlights**: 최선의 경우, LLM은 'golden' features의 약 56%를 의미적으로 포착할 수 있지만, 복잡한 feature가 요구되는 데이터셋에서는 실패할 수도 있습니다.



### Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks (https://arxiv.org/abs/2410.12853)
Comments:
          11 pages, 9 figures

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력과 사실 정확성을 개선하기 위한 다중 에이전트 토론(multi-agent debate) 프레임워크를 제안합니다. 특히, 다양한 모델을 활용한 경우에 더 뛰어난 성능을 발휘했으며, GPT-4와 비교하여 더 높은 정확성을 기록하였습니다.

- **Technical Details**: 다중 에이전트 토론 프레임워크는 질문 인코딩, 토론 모델, 토론 라운드, 응답 요약, 반복적 정제 및 최종 요약의 여섯 가지 주요 구성 요소로 이루어져 있습니다. 이 과정에서 다양한 모델 아키텍처를 활용하여 각 모델의 사고 다양성에 기반한 강력한 논리를 생성합니다.

- **Performance Highlights**: 이 연구에서 사용한 중간 용량 모델 세트(Gemini-Pro, Mixtral 7BX8,와 PaLM 2-M)는 4회 토론 후 GSM-8K 벤치마크에서 91%의 정확도를 기록하여 GPT-4를 초월하였고, ASDiv 벤치마크에서는 94%로 새로운 최고 기록을 세웠습니다.



### The Large Language Model GreekLegalRoBERTa (https://arxiv.org/abs/2410.12852)
- **What's New**: 그리스 법률 및 비법률 텍스트에 대해 훈련된 네 가지 버전의 GreekLegalRoBERTa 모델을 개발했습니다. 이 모델은 GreekLegalBERT 및 그리스 관련 다른 모델들의 성능을 초과합니다.

- **Technical Details**: 이 논문에서는 RoBERTa(Liu et al., 2019)를 사용하여 그리스 법률 문서에서의 이름 개체 인식(NER) 및 다중 클래스 법률 주제 분류 작업을 수행했습니다. 훈련된 네 가지 GreekLegalRoBERTa 모델은 Nomothesia 플랫폼, 그리스 의회 의사록, 유럽 의회 의사록 병렬 코퍼스 등에서 수집된 데이터로 훈련되었습니다.

- **Performance Highlights**: 모델들은 GreekLegalNER에서 이전의 모든 모델들을 초과하는 성능을 보였고, GreekLegalCode 작업에서도 개선된 성능을 나타내었습니다. 특히, micro 평균에서 GreekLegalBERT-v2의 성능을 1.2 포인트 개선하였고, 다양한 분류에서 다른 성과도 달성하였습니다.



### RecurFormer: Not All Transformer Heads Need Self-Attention (https://arxiv.org/abs/2410.12850)
- **What's New**: 이 논문에서는 Transformer 기반의 대형 언어 모델(LLM)의 응답 생성 과정에서 발생하는 계산 비용 문제를 해결하기 위해 RecurFormer라는 새로운 아키텍처를 제안합니다. RecurFormer는 특정 attention head를 linear recurrent neural network (RNN)인 Mamba 아키텍처로 교체하여 메모리 캐시 사이즈를 줄이고, 토큰을 제거하지 않으면서 생성 품질을 유지합니다.

- **Technical Details**: RecurFormer는 recency aware 속성을 가진 attention head를 Mamba 아키텍처로 교체하는 방식으로 구성되어 있습니다. Mamba는 selective structured state-space sequence model 기반의 linear RNN으로, parallel 및 recursive 계산을 지원합니다. 이 방식은 기존 Transformer의 가중치를 계속 활용할 수 있도록 하여 모델의 성능을 유지하면서도 계산 효율을 증대시킵니다.

- **Performance Highlights**: 실험 결과, RecurFormer는 원래 모델의 성능을 유지하면서도 추론 효율성을 크게 향상시키는 것으로 나타났습니다. 또한, 지속적인 훈련을 통해 성능 회복이 가능하다는 것을 보여주어, 긴 입력에 관련된 작업에서 Transformer 기반 LLM의 계산적 도전에 대한 실용적인 해결책을 제공합니다.



### TextLap: Customizing Language Models for Text-to-Layout Planning (https://arxiv.org/abs/2410.12844)
Comments:
          Accepted to the EMNLP Findings

- **What's New**: 이 논문에서는 사용자가 텍스트 지시만으로 매력적인 그래픽 레이아웃을 생성할 수 있도록 돕는 새로운 방법인 TextLap을 제안합니다. TextLap은 특별히 설계된 레이아웃 계획 데이터셋인 InstLap을 활용하여 대형 언어 모델(LLM)을 사용자 맞춤형 그래픽 디자이너로 변환합니다.

- **Technical Details**: TextLap 모델은 레이아웃 생성을 위한 text-to-layout 작업을 수행합니다. 사용자 입력에 따라 레이아웃을 생성하고 수정할 수 있으며, 이는 자연어 대화를 통해 이루어집니다. InstLap 데이터셋은 이미지-캡션 페어를 필터링하고 향상하여 LLM에 대한 사용자 지시 튜닝 데이터를 제공합니다.

- **Performance Highlights**: 텍스트 기반 레이아웃 계획인 TextLap은 다양한 벤치마크 데이터셋에서 평가된 결과, GPT-4 기반의 방법보다 우수한 성능을 나타냈습니다. TextLap은 디자인 생성에 필요한 시간을 줄이고, 디자이너의 작업 효율성을 향상시키는 데 기여합니다.



### UniAutoML: A Human-Centered Framework for Unified Discriminative and Generative AutoML with Large Language Models (https://arxiv.org/abs/2410.12841)
Comments:
          24 pages

- **What's New**: 새로운 AutoML 프레임워크인 UniAutoML이 소개되었습니다. UniAutoML은 기존의 AutoML 프레임워크가 주로 다루었던 discriminative task 뿐만 아니라 generative task도 통합하여 지원하는 것이 특징입니다. 사용자가 쉽게 접근할 수 있도록 자연어로 상호작용할 수 있는 대화형 사용자 인터페이스(CUI)를 제공합니다.

- **Technical Details**: UniAutoML은 Large Language Models (LLMs)를 활용하여 데이터 처리, 모델 선택 및 하이퍼파라미터 검색을 자동화한 인공지능 프레임워크입니다. 사용자들은 자연어 명령을 통해 복잡한 모델을 fine-tuning 할 수 있으며, 모델은 HuggingFace에서 사전 훈련된 다양한 모델을 선택하고 사용할 수 있습니다. 또한, safety guard-line을 설계하여 사용자 입력과 LLM 출력의 필터링이 이루어집니다.

- **Performance Highlights**: UniAutoML은 25명의 참가자를 대상으로 8개의 다양한 데이터셋에 대한 실험을 통해 성능과 사용성을 평가하였고, 그 결과 사용자 제어와 신뢰도를 향상시켰습니다. UniAutoML의 인간 중심 디자인은 AutoML의 기능과 사용자 이해 사이의 격차를 해소하여 더 많은 사람들이 ML(Machine Learning)에 접근할 수 있도록 합니다.



### Answering Questions in Stages: Prompt Chaining for Contract QA (https://arxiv.org/abs/2410.12840)
- **What's New**: 이번 연구에서는 법률 문서에서의 질문에 대한 구조적 답변 생성을 위한 새로운 두 단계 프롬프트 체인을 제안합니다. 이전의 프롬프트가 긴 조항을 다루는 데 한계를 보였던 반면, 이 방식은 더 복잡한 법률 텍스트를 효과적으로 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 법률 관련 질문에 대한 응답을 두 단계로 처리하는 전략을 사용하는데, 첫 번째 단계에서는 관련 법률 텍스트의 요약을 생성하고, 두 번째 단계에서는 이 요약을 사용하여 기존의 프롬프트 템플릿에 대해 질문에 대한 답변을 형성합니다. 이를 통해 질문과 답변 옵션 간의 매핑을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 두 단계 프롬프트 체인이 단순한 프롬프트에 비해 대부분의 경우 더 효과적임을 보여주었습니다. 이는 법률 전문가들이 문서를 더 효율적으로 검토하고 자동화된 워크플로우 및 데이터 파이프라인을 구축할 수 있도록 도와주는 기회를 제공합니다.



### Incorporating Metabolic Information into LLMs for Anomaly Detection in Clinical Time-Series (https://arxiv.org/abs/2410.12830)
- **What's New**: 이번 논문에서는 의료 분야에서의 데이터 분석을 위해 LLMs(Large Language Models)에 대한 도메인 지식을 통합한 새로운 기법인 Metabolism Pathway-driven Prompting(MPP)를 제안합니다. 이 방법론은 생물학 샘플의 구조적 및 시간상의 변화를 더 잘 포착하는데 기여합니다.

- **Technical Details**: 이 논문은 다변량 임상 시계열 데이터의 이상 탐지를 위한 방법론을 제시합니다. MPP는 대사 경로에 관한 정보와 다양한 대사물질의 시간적인 변화를 LLM에 통합하여, 시간 경과에 따른 대사물질 간의 의존성을 고려합니다. 이를 통해 특정 샘플에 대한 이상 점수를 부여하는 함수 f(xt)를 학습합니다.

- **Performance Highlights**: 결과적으로, 이 방법은 스포츠에서의 도핑 탐지 문제에 효과적으로 적용되며, 실제 데이터를 사용하여 의심스러운 표본의 발견 성능을 개선합니다. MPP는 기존의 제로샷 학습(zero-shot learning) 및 맥락 학습(in-context learning) 기법과 비교할 때 우수한 성과를 보였습니다.



### GCM-Net: Graph-enhanced Cross-Modal Infusion with a Metaheuristic-Driven Network for Video Sentiment and Emotion Analysis (https://arxiv.org/abs/2410.12828)
- **What's New**: 이 논문은 다양한 모달리티(modality)에서의 감정 분석(Emotion Recognition)과 감정 인식(Sentiment Analysis)의 복잡성을 해결하기 위해 새로운 프레임워크(GCM-Net)를 제안합니다. 본 연구는 모달리티 통합(fusion)과 특징 최적화(feature optimization)에 주안점을 두고 있습니다.

- **Technical Details**: GCM-Net은 그래프 샘플링(graph sampling)과 집계를 통해 모달리티 특징을 재조정하는 기능을 포함하고 있습니다. 또한, 교차 모달(attention) 모듈을 사용하여 모달 간 상호작용을 파악하고 발화 관련성(utterance relevance)을 결정합니다. 하모닉 최적화(harmonic optimization) 모듈은 메타휴리스틱(metaheuristic) 알고리즘을 사용하여 주목(attention)된 특징들을 결합합니다.

- **Performance Highlights**: 제안된 GCM-Net의 성능은 CMU MOSI 데이터셋에서 91.56%, CMU MOSEI 데이터셋에서 86.95%의 정확도를 보였으며, IEMOCAP 데이터셋에서의 감정 분석에서 85.66%의 정확도를 기록했습니다. 이는 기존 방법 대비 상당한 성능 향상을 나타냅니다.



### TIMeSynC: Temporal Intent Modelling with Synchronized Context Encodings for Financial Service Applications (https://arxiv.org/abs/2410.12825)
Comments:
          6 pages, Accepted at RecSys 2024

- **What's New**: 이 연구에서는 금융 서비스 분야에서 다양한 채널을 통해 축적된 다중 도메인 시퀀스 데이터를 효과적으로 결합하고 인코딩하여 고객의 의도를 예측하기 위한 새로운 인코더-디코더 트랜스포머 모델 TIMeSynC(Temporal Intent Modelling with Synchronized Context Encodings)를 제안합니다.

- **Technical Details**: TIMeSynC 모델은 고객의 다채널 활동을 플래튼하고 토큰화하여 인코더 컨텍스트를 생성하고, 다음 의도를 예측하기 위한 온라인 의도 시퀀스를 디코더에 통합합니다. 또한, ALiBi 기반의 시간 표현과 다변량 및 다중 도메인 시퀀스 간의 시간적 상관관계를 학습하기 위한 시간 인코더를 도입하였습니다. 이 과정에서 교차 주의(attention) 모듈에 시간 기반 주의 마스크를 사용하고, 도메인 및 필드 인식을 위한 필드 이름 임베딩을 추가하여 복잡한 패턴 간의 직접적인 상호작용을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, TIMeSynC 모델은 기존의 테이블 방식(tabular method)보다 현저한 개선을 보여주었으며, 다양한 금융 서비스 응용 분야에서 효과적인 의도 예측을 가능하게 한다는 것이 입증되었습니다.



### Optimization of Actuarial Neural Networks with Response Surface Methodology (https://arxiv.org/abs/2410.12824)
Comments:
          This work was presented at the Actuarial Research Conference (ARC) this http URL abstract submitted and presented at ARC 2024. More details can be found at \url{this https URL}

- **What's New**: 이 연구는 Combined Actuarial Neural Networks (CANN)의 하이퍼파라미터 최적화를 위해 Response Surface Methodology (RSM)를 활용하여 성능을 향상시키는 방법을 제시합니다. RSM은 하이퍼파라미터 공간을 효과적으로 탐색할 수 있도록 구조화된 접근 방식을 제공합니다.

- **Technical Details**: RSM을 사용하여 하이퍼파라미터 공간에서 최적의 조합을 식별하고, 비효율적인 전통적 그리드 서치를 대체하여 날카롭게 잡은 결과를 제공합니다. 연구에서는 하이퍼파라미터 도메인을 미리 선택하고, 실험에서 응답 변수와 같은 다양한 요소들을 비교하여 최상의 설정을 찾아냅니다.

- **Performance Highlights**: 최종 결과적으로 하이퍼파라미터의 수를 288에서 188로 줄이되 정확성에는 미미한 손실로 유지하였으며, 거의 최적에 가까운 out-of-sample Poisson deviance loss를 달성했습니다.



### Advancing Spatio-temporal Storm Surge Prediction with Hierarchical Deep Neural Networks (https://arxiv.org/abs/2410.12823)
- **What's New**: 본 연구는 허리케인과 노르이스터로 인한 해안 지역의 스톰 서지(storm surge) 예측을 개선하기 위해 고차원 신경망(hierarchical deep neural network, HDNN)과 합성곱 오토인코더(convolutional autoencoder, CAE)를 결합한 새로운 접근 방식을 제안합니다. 이 모델은 고차원의 데이터를 효율적으로 처리할 수 있는 방법을 제공합니다.

- **Technical Details**: 제안된 모델은 CAE를 사용하여 스톰 서지 데이터의 차원을 축소하고, HDNN을 통해 스톰 매개변수(storm parameters)를 저차원 표현으로 매핑합니다. 이러한 방식으로 모델은 서로 다른 시간 스케일(time scales)에 걸쳐 순차적으로 예측을 수행하며, 예측 단계가 누적될 때 발생할 수 있는 오류를 최소화하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 모델은 북대서양 포괄적 해안 연구(North Atlantic Comprehensive Coastal Study)에서 생성된 합성 데이터로 훈련 및 테스트를 거쳤으며, 높은 차원의 서지 데이터를 효과적으로 처리하고 예측 오류의 누적을 완화하는 데 뛰어난 성능을 보였습니다. 이는 스톰 서지 예측을 발전시키기 위한 유망한 도구로 평가됩니다.



### AVID: Adapting Video Diffusion Models to World Models (https://arxiv.org/abs/2410.12822)
- **What's New**: 이 연구에서는 사전 학습된 비디오 확산 모델을 액션 조건화된 월드 모델에 적응시키는 새로운 접근 방식인 AVID를 제안합니다. AVID는 액션 라벨이 붙은 비디오의 작은 도메인 특정 데이터셋에서 어댑터를 학습하여, 사전 학습된 모델의 매개변수에 접근할 수 없이 액션에 조건화된 비디오를 생성합니다.

- **Technical Details**: Diffusion 모델(확산 모델)의 노이즈 예측을 수정하여 액션 수반 예측을 생성하는 방법을 채택했습니다. AVID는 학습된 마스크를 이용하여 사전 학습된 모델의 중간 출력을 변형하고, 액션과 조건화된 비디오 출력을 생성합니다.

- **Performance Highlights**: 비디오 게임 및 실제 로봇 데이터에서 AVID를 평가하였으며, 기존의 확산 모델 적응 기준선보다 우수한 성능을 보였습니다. 사전학습된 모델을 올바르게 활용할 경우, 임베디드 AI(embodied AI)에 강력한 도구가 될 가능성을 демонстр합니다.



### A transformer-based deep reinforcement learning approach to spatial navigation in a partially observable Morris Water Maz (https://arxiv.org/abs/2410.12820)
- **What's New**: 이번 연구는 Morris Water Maze (MWM) 실험을 재현하기 위해 transformer 기반 아키텍처를 이용한 딥 강화학습을 적용한 것입니다. 이는 기존 연구에서 다루지 않았던 접근법으로, 2D 미로에서 에이전트가 효과적으로 탐색할 수 있도록 합니다.

- **Technical Details**: 에이전트는 decoder-only transformer 아키텍처를 활용하여 부분 관찰 가능한 환경에서 Q-value를 예측합니다. 비교적 제한된 시각 정보를 가진 환경에서 효율적으로 학습하며, 그 결과 공간 탐색 전략을 습득하게 됩니다. 뉴럴 네트워크의 회귀 문제를 해결하기 위해 recurrent position encoding과 multi-head attention도 사용합니다.

- **Performance Highlights**: 제안된 transformer 아키텍처는 에이전트가 효율적으로 탐색 임무를 수행하도록 함으로써, 내부 환경 표현에 대한 이해도를 높일 수 있는 기회를 제공합니다. 특히, 보조 작업 없이도 빠르게 학습할 수 있는 능력을 보여주며, 생물학적 에이전트와 유사한 행동을 보일 수 있는 잠재력을 시사합니다.



### Deep Adversarial Learning with Activity-Based User Discrimination Task for Human Activity Recognition (https://arxiv.org/abs/2410.12819)
- **What's New**: 본 연구는 인체 활동 인식(Human Activity Recognition, HAR) 문제를 위해 새로운 적대적 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 사람 간의 변동성을 해결하기 위한 새로운 활동 기반의 구분 작업을 통합하며, 이는 사람들이 동일한 활동을 수행하는 방식이 다름을 인정합니다.

- **Technical Details**: 저자들은 다중 작업, 적대적 학습(adversarial learning) 및 자기 지도 학습(self-supervised learning)에 기반한 표현 학습 방법을 활용하여 모델의 일반화 성능을 향상시키고 개인 정보 유출 문제를 완화합니다. 제안된 프레임워크는 이진 분류 작업으로, 동일한 사람과 동일한 활동에서의 활동 특성 벡터 쌍이 동일한지 구분하는 것을 목적으로 합니다.

- **Performance Highlights**: 제안한 프레임워크는 Leave-One-Person-Out Cross-Validation (LOOCV) 설정에서 새로운 보이지 않는 개인들에 대한 성능을 측정하며, 기존 접근 방식보다 분류 성능을 개선하는 결과를 보여주었습니다. 또한 훈련 및 테스트 참가자 간의 동일한 활동에서의 사람 간 변동성 격차를 감소시켰습니다.



### Restoring Super-High Resolution GPS Mobility Data (https://arxiv.org/abs/2410.12818)
Comments:
          Accepted paper for the 2nd ACM SIGSPATIAL International Workshop on Geo-Privacy and Data Utility for Smart Societies (GeoPrivacy 2024)

- **What's New**: 본 논문은 저해상도 GPS 경로 데이터를 복원하기 위한 새로운 시스템을 제안하며, 이는 이동 응용 프로그램에서 데이터 유용성과 개인 정보 보호 간의 비판적인 균형 문제를 해결합니다.

- **Technical Details**: 시스템은 Transformer 기반의 인코더-디코더 모델과 그래프 컨볼루션 네트워크(Graph Convolutional Networks, GCNs)를 통합하여 경로 데이터의 시간적 의존성과 도로 네트워크의 공간적 관계를 효과적으로 캡처합니다.

- **Performance Highlights**: 이 시스템은 베이징 경로 데이터셋에서 평가되었으며, 전통적인 맵 매칭 알고리즘과 LSTM 기반의 합성 데이터 생성 방법에 비해 우수한 성능을 보여주었습니다. 제안된 모델은 평균 Fréchet 거리 0.198km를 달성했으며, 이는 맵 매칭 알고리즘(0.632km) 및 합성 경로 모델(0.498km)을 크게 능가합니다.



### Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspectiv (https://arxiv.org/abs/2410.12816)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 CLIP 모델의 두 가지 정렬 문제인 작업 불일치(task misalignment) 및 데이터 불일치(data misalignment)를 해결하기 위한 방법을 제안합니다. 특히, 데이터 불일치가 다운스트림 작업에서 성능에 미치는 영향을 분석하고 Causality-Guided Semantic Decoupling and Classification (CDC) 방법론을 개발하여 이 문제를 해결합니다.

- **Technical Details**: CDC 방법론은 두 가지 주요 구성 요소인 Visual-Language Dual Semantic Decoupling (VSD)와 Decoupled Semantic Trusted Classification (DSTC)로 이루어져 있습니다. VSD는 다양한 의미를 표현하는 여러 프롬프트 템플릿을 모델에 통합하여 학습합니다. DSTC는 각 층에서 분리된 의미에 기반하여 분류 작업을 독립적으로 수행하며, 예측의 불확실성을 동시에 추정합니다.

- **Performance Highlights**: 다양한 데이터셋과 여러 작업에서 진행된 실험 결과, CDC 방법론이 CLIP의 성능을 유의미하게 향상시킴을 보여주었습니다. 특히, 새로운 클래스에 대한 인식 성능이 개선되는 효과가 있음을 확인했습니다.



### Cerebral microbleeds: Association with cognitive decline and pathology build-up (https://arxiv.org/abs/2410.12809)
Comments:
          11 pages, 2 figures

- **What's New**: 이번 연구는 두 가지 주요 발견이 있습니다: 첫째, 미세출혈(lobar microbleeds)의 존재가 알츠하이머 질환(AD)의 인지적 감소와 관련이 있다는 점, 둘째, 미세출혈의 위치와 특정 인지 영역(언어, 의미 이해, 실행 기능 등)에 대한 영향을 분석했습니다.

- **Technical Details**: 연구에서는 ADNI(Alzheimer's Disease Neuroimaging Initiative) 참가자 1,573명을 대상으로 MR 스캔 데이터와 미세출혈의 유형 및 위치에 대한 정보를 분석했습니다. 회귀 분석(ordinary least-squares regression)을 사용하여 인지 감소와 미세출혈 간의 연관성을 평가했으며, ADAS-Cog11를 통해 인지 감소를 측정했습니다.

- **Performance Highlights**: 173명 참가자 중 373명은 미세출혈이 발견되었고, 특히 측두엽(microbleeds in the temporal lobe)에서 발생한 미세출혈이 언어 및 의미 이해와 같은 인지 기능 감소와 강한 연관성을 보였습니다. 이러한 연구 결과는 AD 진단과 예후 평가에서 미세출혈을 고려해야 함을 제안합니다.



### A Hierarchical conv-LSTM and LLM Integrated Model for Holistic Stock Forecasting (https://arxiv.org/abs/2410.12807)
Comments:
          8 pages, 2 figures, 2 tables

- **What's New**: 본 연구는 전통적인 주식 시장 예측 모델의 제한점을 극복하기 위해 새로운 Two-Level Conv-LSTM Neural Network와 Large Language Model (LLM)의 통합 접근 방식을 제안합니다.

- **Technical Details**: 모델은 두 가지 주요 레벨로 구성되어 있습니다. 첫 번째 레벨은 주가 및 기술 지표에서 지역 패턴을 추출하기 위한 Convolutional 층과 시간적 역학을 포착하기 위한 Long Short-Term Memory (LSTM) 층을 포함합니다. 두 번째 레벨은 LLM을 통합하여 금융 뉴스, 소셜 미디어 및 보고서의 감정 및 맥락 정보를 분석합니다.

- **Performance Highlights**: 이 통합 접근 방식은 예측 정확도를 향상시키고 주식 조언에 맥락적으로 풍부한 정보를 제공합니다.



### Hip Fracture Patient Pathways and Agent-based Modelling (https://arxiv.org/abs/2410.12804)
Comments:
          4 pages, 2 figures

- **What's New**: 이 연구는 에이전트 기반 시뮬레이션을 사용하여 의료 자원의 최적화에 대한 진행 중인 프로젝트를 개략적으로 설명합니다. 특히 아일랜드의 노인 환자 유입 문제를 해결하기 위한 디지털 기술 도입에 중점을 두고 있습니다.

- **Technical Details**: 연구는 머신러닝(Machine Learning, ML) 기술을 활용하여 예측 진단, 환자 유입 최적화, 자원 관리 등 다양한 의료 분야에 적용하고 있습니다. 에이전트 기반 모델은 환자 경로 및 병목 현상을 분석하는 데 사용되고 있습니다.

- **Performance Highlights**: 이 연구의 결과는 의료 시스템의 운영 효율성을 향상시키고 환자 치료의 질을 높이는 데 기여할 것으로 기대됩니다. 이를 통해 아일랜드는 의료 혁신의 최전선에 서고, 장기적으로 비용 절감과 자원 관리 개선을 이루어낼 가능성이 높습니다.



### Developing Guidelines for Functionally-Grounded Evaluation of Explainable Artificial Intelligence using Tabular Data (https://arxiv.org/abs/2410.12803)
- **What's New**: 본 연구는 Tabular data에 대한 Explainable Artificial Intelligence (XAI) 기법의 평가 기준과 방법을 현대적 확인을 통해 제시합니다. 특히, 기능적으로 구체화된 (functionally-grounded) 평가 프로토콜을 심화 분석하여 Tabular data에서 XAI 기법의 효과성을 강조합니다.

- **Technical Details**: 연구에서는 XAI 기법을 평가하기 위한 20개의 평가 기준과 각각의 평가 방법을 식별하였으며, 각 기준의 평가 시기와 방법에 관한 가이드라인을 제시합니다. Tabular data에 적합한 그림 설명 기법의 정확성과 관련성을 평가하기 위한 적절한 방법론이 부족하여, 평가 절차에 대한 적합성을 재검토했습니다.

- **Performance Highlights**: 이 연구는 Tabular data에 대해 XAI 기법을 평가하기 위한 명확한 기준과 방법을 마련하여, 향후 연구 및 실무 적용에 대한 기초 작업을 다지며, 투명성 및 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Ads Supply Personalization via Doubly Robust Learning (https://arxiv.org/abs/2410.12799)
Comments:
          Accepted by CIKM'24

- **What's New**: 이 논문에서는 광고 공급 개인화를 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 데이터 수집 정책을 통해 정보를 최적 활용하여 장기적 치료 효과 추정의 정확성을 크게 향상시킵니다. 또한, 낮은 복잡도로 인해 기존 방법들보다 계산 비용이 절감되고, 대규모 애플리케이션에 확장 가능하다는 장점을 지니고 있습니다.

- **Technical Details**: 제안된 프레임워크는 Doubly Robust Learning (DRL)을 기반으로 하여 장기적인 인과 효과를 모델링하는 가벼운 솔루션을 제공합니다. DRT 프레임워크는 데이터 수집과 모델링 단계에서 정보를 효율적으로 활용해 성능 향상 및 모델 복잡성 감소를 이끌어내며, 기존의 광고 및 유기 콘텐츠 배포 시스템과 통합이 용이합니다.

- **Performance Highlights**: 오프라인 실험과 온라인 생산 테스트를 통해, 이 프레임워크는 몇 달 간에 걸쳐 비즈니스 주요 지표에서 상당한 개선을 지속적으로 보여주었으며, 세계에서 가장 큰 소셜 미디어 플랫폼 중 하나에 완전히 배포되었습니다.



### Predicting the Geolocation of Tweets Using transformer models on Customized Data (https://arxiv.org/abs/2303.07865)
Comments:
          31 pages, 5 tables, 9 figures

- **What's New**: 이번 연구는 트위터 사용자 및 트윗의 지리적 위치 예측을 위한 유연한 접근 방식을 제공합니다. 연구진은 자연어 처리(NLP) 기술인 신경망을 활용하여 위도와 경도로 구성된 위치 좌표를 추정하고, 이차원 가우시안 혼합 모델(GMM)을 적용하여 보다 정확한 예측을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 미리 훈련된 Bidirectional Encoder Representations from Transformers(BERT)를 기반으로 하여 트위터 데이터셋에서 세부 조정되었습니다. 연구 결과, 전 세계 수준에서 평균 30km 미만의 오류를 기록하며, 미국의 경우 15km 미만의 오류로 더욱 향상된 예측 성능을 보였습니다.

- **Performance Highlights**: 제안된 방법론은 트윗의 내용 및 메타데이터 컨텍스트에 대한 텍스트 특징을 훈련 및 평가에 사용했습니다. 연구팀은 전체 트위터 데이터에서 단 1-2%만이 정확한 지리적 좌표를 지닌 메타데이터로 구분됨을 강조하며, 이로 인해 효과적인 지리적 위치 예측의 필요성을 언급합니다.



### Loss Landscape Characterization of Neural Networks without Over-Parametrization (https://arxiv.org/abs/2410.12455)
- **What's New**: 본 연구에서는 최신 딥 러닝 모델의 손실 경계를 효과적으로 특성화할 수 있는 새로운 함수 클래스를 제안합니다. 과도한 매개변수화 없이 졸업 점(saddle points)을 포함할 수 있는 함수가 이 제안된 클래스에 속합니다.

- **Technical Details**: 새롭게 제안된 α-β 조건을 통해, 다양한 복잡한 함수의 적용 가능성을 이론적으로 증명합니다. 이 조건에 따라 일반적인 그래디언트 기반 최적화기(optimizers)에 대한 수렴 보장을 도출했습니다.

- **Performance Highlights**: 제안된 α-β 조건은 다양한 딥 러닝 모델들(ResNet, LSTM, GNN, Transformer 등)에서의 성능을 실험적으로 검증해, 이전의 PL 조건과 비교할 때 더 작은 네트워크 크기로도 수렴 보장을 제공합니다.



### LLM-based Cognitive Models of Students with Misconceptions (https://arxiv.org/abs/2410.12294)
- **What's New**: 이 논문은 AI 기반 교육 기술에서 학생 인지를 정확하게 모델링하는 것의 중요성을 강조하며, 학생 모델링에서 잘못된 인식을 포함한 정확한 문제 해결을 동시에 충족하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: MalAlgoPy라는 새로운 Python 라이브러리를 통해 대수 문제 해결 과정에서 학생의 해결 패턴을 반영하는 데이터셋을 생성하며, 이를 그래프 기반으로 나타냅니다. 모델은 학생 모델(Cognitive Student Models, CSM)로서, 잘못된 인식과 올바른 지식을 동시에 반영하는 방법으로 훈련됩니다.

- **Performance Highlights**: 잘못된 인식 예제로 훈련된 LLMs는 문제를 올바르게 해결하는 능력이 감소했으나, 훈련 데이터에서 올바른 예제와 잘못된 예제의 비율을 조정함으로써 두 가지 속성을 모두 만족하는 CSM을 개발할 수 있음을 보여주었습니다.



### ClickAgent: Enhancing UI Location Capabilities of Autonomous Agents (https://arxiv.org/abs/2410.11872)
Comments:
          The code for ClickAgent is available at this http URL

- **What's New**: ClickAgent는 GUI와 상호작용할 수 있는 자율 에이전트를 구축하기 위한 새로운 프레임워크입니다. MLLM의 추론과 행동 계획을 담당하며, 별도의 UI 위치 모델이 화면에서 관련 UI 요소를 식별합니다.

- **Technical Details**: ClickAgent는 MLLM 기반의 추론을 InternVL2.0을 사용하여 수행하고, TinyClick UI 위치 모델을 사용합니다. 세 가지 주요 구성 요소로는 Decision, UI Location, Reflection이 있습니다.

- **Performance Highlights**: ClickAgent는 AITW 벤치마크에서 다른 프롬프트 기반 자율 에이전트보다 우수한 성능을 보였으며, 작업 성공률에서 유의미한 개선을 이루었습니다.




New uploads on arXiv(cs.CL)

### Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads (https://arxiv.org/abs/2410.01805)
Comments:
          Preprints

- **What's New**: Locret 프레임워크는 긴 컨텍스트를 처리할 수 있는 LLM 추론을 가능하게 하며, KV 캐시의 유효성을 평가하기 위해 새로운 retaining heads를 도입했습니다. 이는 고정된 캐시 크기 내에서 보다 정밀하게 캐시 제거를 수행할 수 있게 해줍니다.

- **Technical Details**: Locret는 고정된 KV 캐시 크기를 유지하면서 중요도가 낮은 캐시 유닛을 제거합니다. 이 과정에서 약간의 추가 파라미터가 필요한 retaining heads가 도입되며, 이는 기존 LLM 모델을 결합하여 사용할 수 있습니다. Locret는 최소한의 데이터로 frozen backbone LLM 위에서 fine-tuning 되어 정확한 토큰 중요도를 평가합니다.

- **Performance Highlights**: Locret는 Phi-3-mini-128K 및 Llama-3.1-8B-instruct 모델에서 각각 20배 및 8배의 KV 캐시 압축비를 달성하였으며, 이는 소비자 등급의 단일 Nvidia 4090 GPU에서도 128K 긴 컨텍스트 추론이 가능하도록 합니다. Locret는 메모리 효율성과 생성 품질 면에서 최근의 경쟁 기술을 초월하는 성능을 보입니다.



### Loki: An Open-Source Tool for Fact Verification (https://arxiv.org/abs/2410.01794)
- **What's New**: Loki는 잘못된 정보 문제를 해결하기 위해 설계된 오픈 소스 도구로, 사실 확인 프로세스를 5단계가 포함된 방식으로 나누어 인간의 판단을 지원합니다. 이 도구는 사용자가 긴 텍스트를 간단한 주장으로 분해하고, 주장 검토, 증거 검색 및 확인을 돕는 구조를 제공합니다.

- **Technical Details**: Loki는 다섯 단계의 사실 검증 파이프라인: Decomposer(분해기), Checkworthiness Identifier(검증 가능성 식별기), Query Generator(쿼리 생성기), Evidence Retriever(증거 검색기), Claim Verifier(주장 확인기)로 구성되어 있습니다. Python으로 구현되어 있으며, 사용자 인터페이스가 친근하고 여러 인터페이스를 통해 사용 가능합니다.

- **Performance Highlights**: Loki는 다른 최근의 사실 확인 도구들과 비교하여 효율성 및 성능에서 현저한 개선을 보였습니다. 또한 멀티 언어 지원이 가능하고, 상업적으로 활용가능한 수준으로 최적화되어 있습니다.



### When a language model is optimized for reasoning, does it still show embers of autoregression? An analysis of OpenAI o1 (https://arxiv.org/abs/2410.01792)
Comments:
          6 pages

- **What's New**: 이 논문에서는 OpenAI의 새로운 시스템 o1이 이전의 대형 언어 모델(LLMs)을 어떻게 개선했는지를 탐구합니다. 특히, o1은 추론을 위해 최적화되어 있어, 일반적인 작업에서 드문 변형에서 특히 큰 성능 향상을 보여 줍니다.

- **Technical Details**: o1은 Chain of Thought(사고의 연쇄) 기법을 사용하여 복잡한 문제를 단계적으로 해결하는 방식으로 훈련됩니다. 또한, o1은 높은 확률의 출력 예시에서 더 좋은 성능을 보이며, 공통 작업 변형보다 드문 작업 변형에서 성능 차이가 덜 Pronounced(두드러지지 않음)합니다. 이 모델은 또한 'thinking tokens'를 생성하여 과제를 수행하는 방식을 수량화합니다.

- **Performance Highlights**: o1은 높은 확률의 출력에서 92% 정확도를 기록하며, 드문 작업 변형에 대해서도 상당한 성능 향상을 보입니다. 그러나, 여전히 예전 LLMs와 마찬가지로 출력 확률에 민감성을 보이며, 저확률 예시에서 더 많은 'thinking tokens' 을 사용합니다.



### Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models (https://arxiv.org/abs/2410.01782)
Comments:
          Accepted to EMNLP 2024 Findings. Website: this https URL. 14 pages, 7 figures, 5 tables

- **What's New**: 본 논문에서는 Open-RAG라는 새로운 프레임워크를 제안하여 여러 오픈소스 LLM과 함께 RAG를 활용한 추론 능력을 향상시키고자 합니다. 이 프레임워크는 복잡한 추론 작업을 처리할 수 있는 파라미터 효율적인 희소 MoE 모델로의 변환을 통해 성능을 높입니다.

- **Technical Details**: Open-RAG은 기존의 LLM을 기반으로 하여 모델이 자체 반성을 통해 레프리케이션(retrieval)과 관련된 특수 토큰을 생성하면서, 복잡한 질문에 대한 추론을 보다 효과적으로 수행할 수 있도록 학습합니다. 또한, 혼합 적응형 검색(hybrid adaptive retrieval) 방법을 도입하여 성능 향상과 추론 속도의 균형을 맞춥니다.

- **Performance Highlights**: Llama2-7B 기반의 Open-RAG는 ChatGPT, Self-RAG 및 Command R+와 같은 최첨단 LLM 및 RAG 모델을 초월하는 성과를 보이며, 다양한 지식 집약적 작업에서 새로운 벤치마크를 설정하였습니다. 실험 결과, Open-RAG가 선행 오픈소스 RAG 모델에 비해 의미 있는 정확도 향상과 추론 능력 개선을 보여주었습니다.



### DeFine: Enhancing LLM Decision-Making with Factor Profiles and Analogical Reasoning (https://arxiv.org/abs/2410.01772)
- **What's New**: 이 논문에서는 LLMs (Large Language Models) 의사결정 과정에서 복잡한 시나리오를 다루기 위해 DeFine 프레임워크를 제안합니다. 이 프레임워크는 복잡한 시나리오에서 확률적 요인 프로필을 구축하고, 유사한 과거 경험으로부터 통찰력을 활용하여 LLMs가 새로운 상황에서 중요한 결정을 내릴 수 있도록 지원합니다.

- **Technical Details**: DeFine은 분석의 기초로 보통 문법이 틀리거나 불완전한 문장을 포함한 긴 스크립트를 사용합니다. 이 스크립트에서 중요한 정보를 요약해 여러 요인 세트로 정리하고 각 요인의 잠재적 결과에 대한 확률을 추정합니다. Bradley-Terry 모델을 활용하여 결정적인 요인을 식별하고 이들이 의사결정에 미치는 영향을 평가합니다. 이 연구에서는 또한 유사한 상황 간의 연결성을 파악하기 위해 유사 추론(analogical reasoning)을 통합하여 LLMs의 의사결정을 돕습니다.

- **Performance Highlights**: 본 연구는 투자자들이 주식 이동 추세를 예측할 수 있는 통찰력을 제공하며, 이는 주로 순자산 증감 및 관련 지표 분석을 통해 이루어집니다. DeFine의 접근법은 금융 분야를 넘어 의료 상담 및 정치적 논쟁과 같은 복잡한 문제들을 다루는 분야에서도 활용될 가능성이 높습니다.



### Quantifying Generalization Complexity for Large Language Models (https://arxiv.org/abs/2410.01769)
- **What's New**: 이번 논문에서는 Scylla라는 새로운 동적 평가 프레임워크를 소개하여 대형 언어 모델(LLMs)의 일반화 능력을 정량적으로 측정합니다. 이 프레임워크는 메모리화(memorization)와 일반화(generalization)를 분리하고, 모델 성능을 in-distribution (ID)와 out-of-distribution (OOD) 데이터에서 평가합니다.

- **Technical Details**: Scylla는 5가지 난이도와 20개의 작업을 통해 LLMs의 일반화 능력을 측정하며, 비선형(non-monotonic) 관계가 존재함을 발견했습니다. 특히 임계 복잡성(critical complexity)이라는 개념을 도입하여, 메모리에 과도하게 의존하는 경향을 평가할 수 있도록 설계되었습니다. 또한, 결과적으로 대형 모델이 더 복잡한 추론 작업을 처리할 수 있는 능력이 확대되는 경향을 보여주었습니다.

- **Performance Highlights**: 28개의 LLM을 평가한 결과, 대형 모델은 더 높은 난이도의 작업에서도 일반화 능력을 유지할 수 있는 반면, 소형 모델은 더 간단한 작업에서는 성능 차이를 보였습니다. 일반화 계곡(generalization valley)이라는 현상은 특정 작업 복잡도에서 훈련 데이터에 과적합(overfitting)되는 경향을 나타내며, 이때 메모리화에 대한 의존도가 최고조에 달합니다.



### Recursive Abstractive Processing for Retrieval in Dynamic Datasets (https://arxiv.org/abs/2410.01736)
- **What's New**: 이 논문에서는 동적 데이터셋에 대한 효율적인 업데이트를 위해 새로운 알고리즘을 제안하고 있습니다. 기존의 Retrieval-Augmented 방법들은 클러스터링을 통해 계층 구조를 형성하지만, 문서의 추가 및 제거가 잦은 동적 데이터셋에서는 이 구조를 업데이트하는 것이 복잡하고 비효율적입니다. 제안된 알고리즘은 이러한 계층 구조를 유지하면서도 성능을 저하시키지 않습니다.

- **Technical Details**: 우리는 adRAP (adaptive Recursive Abstractive Processing)이라는 알고리즘을 도입하여 RAPTOR의 재귀적 요약 구조를 동적으로 업데이트합니다. 이 알고리즘은 새로운 문서가 추가되거나 제거될 때 전체 재계산을 피하면서 성능을 유지합니다. 또한 postQFRAP라는 새로운 포스트 리트리벌 방법을 소개합니다. 이 방법은 쿼리 중심의 재귀적 요약 처리를 통해 수집된 맥락의 질을 크게 향상시킵니다.

- **Performance Highlights**: 실제 데이터셋을 통해 수행된 광범위한 실험에서는 adRAP가 RAPTOR 트리 구조를 잘 근사하며, postQFRAP가 리트리벌 품질을 효과적으로 향상시킨다는 것을 보여주었습니다. 이 방법들은 동적 데이터 처리와 리트리벌 성능 개선에 효과적입니다.



### LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits (https://arxiv.org/abs/2410.01735)
Comments:
          20 pages; First two authors contributed equally. Code: this https URL

- **What's New**: LASeR (Learning to Adaptively Select Rewards)는 여러 개의 리워드 모델(Reward Models, RMs)을 사용하여 LLM을 적응적으로 훈련하는 새로운 접근법입니다. 이 방법은 학습 과정 중 각 인스턴스에 가장 적합한 RM을 선택하여 리워드의 선택을 다중 무장 강도 문제(multi-armed bandit problem)로 표현합니다.

- **Technical Details**: LASeR는 RMs의 선택을 문맥 정보와 과거 상호작용에 기반해 동적으로 진행합니다. 이는 LLM의 성능과 각 RM의 적합성을 반영하여 훈련을 진행하며, RM 업데이트는 그 성능에 따라 조정됩니다. 이 방법은 단일 RM 사용에 따른 한계를 해결하며 성능 향상을 도모합니다.

- **Performance Highlights**: LASeR를 통해 Llama-3-8B 모델은 commonsense 및 math reasoning 테스트에서 최대 2.67%의 정확도 개선을 달성했습니다. WildChat 데이터셋에서 LASeR를 사용하는 경우 71.45%의 win rate를 기록했으며, 긴 컨텍스트 생성 작업에서도 평균적으로 F1 스코어 2.64 및 2.42의 향상을 이끌어냈습니다.



### Visual Perception in Text Strings (https://arxiv.org/abs/2410.01733)
- **What's New**: 이번 연구에서는 ASCII 아트를 통하여 LLM(대형 언어 모델)과 MLLM(멀티모달 대형 언어 모델)의 시각적 이해 능력을 분석합니다. 기존의 연구들은 텍스트와 이미지를 효과적으로 조합하는 능력에 중점을 두었지만, ASCII 아트는 텍스트 문자열 내에서 시각적 정보를 내포하고 있어 새로운 접근 방식을 제시합니다.

- **Technical Details**: 본 연구는 359개의 개념을 포함하는 평가 데이터세트인 ASCIIEval을 구축하였습니다. 이 데이터세트는 ASCII 아트를 텍스트 문자열, 이미지 또는 두 가지 모드 모두에서 입력 받아 필기적 시각적 의미를 인식하는 능력을 평가합니다.  또한, 모델의 성능을 분석하여 MLLM들이 이미지와 텍스트 모드를 함께 제공받았을 때의 응답 능력 및 효과적 교육 방법론의 필요성을 강조합니다.

- **Performance Highlights**: 모델들은 특정 개념 카테고리에서 60% 이상의 정확도로 ASCII 아트에서의 시각적 의미를 인식할 수 있습니다. 그러나 기존의 LLM들은 ASCIIEval에서 평균적으로 30%의 낮은 정확도로 성능이 저조합니다. GPT-4o는 이미지를 입력으로 받았을 때 82.68%의 정확도를 기록하였으며, 이는 최상의 오픈 소스 MLLM보다 21.95% 더 높은 결과입니다.



### Auto-Demo Prompting: Leveraging Generated Outputs as Demonstrations for Enhanced Batch Prompting (https://arxiv.org/abs/2410.01724)
- **What's New**: 이번 논문에서는 'Auto-Demo Prompting'이라는 새로운 배치 프롬프트 기법을 제안하여, 배치 프롬프트의 성능 저하 문제를 해결합니다. 이 방법은 배치 내의 이전 질문에서 생성된 질문-답변 쌍을 활용하여 후속 질문의 답변을 추론합니다.

- **Technical Details**: Auto-Demo Prompting은 autoregressive generation 과정에서 작동하며, 이전 출력 결과를 활용하여 모델의 내부 표현을 최적화합니다. 질문-답변 쌍을 자동으로 인식하여 후속 질문에 대한 데모로 사용하고, 동일한 구조의 질문에서 발생할 수 있는 성능 저하를 완화하는데 중점을 둡니다.

- **Performance Highlights**: 실험 결과는 Auto-Demo Prompting이 전통적인 배치 프롬프트보다 뛰어난 성능을 보였음을 보여주며, 단일 프롬프트와 비교했을 때도 효율적이고 해석 가능한 방식으로 모델 성능을 크게 향상시켰습니다.



### Examining the Role of Relationship Alignment in Large Language Models (https://arxiv.org/abs/2410.01708)
- **What's New**: 이번 연구는 Llama 3.0 (70B)가 Facebook의 댓글 데이터셋을 기반으로 사용자의 성별, 나이, 친구 관계의 밀접함을 고려하여 의미적 톤을 예측하는 능력을 평가하였다. 이 연구는 LLM(대형 언어 모델)이 어떻게 사회적 관계 정보를 활용하여 인간의 댓글과 유사한 내용을 생성할 수 있는지를 탐구한다.

- **Technical Details**: 이 연구는 2개의 부분으로 나뉘어 진행되며, 첫 번째 부분에서는 사회적 관계 범주 간의 의미적 톤 차이를 평가하고 두 번째 부분에서는 Llama 3.0 (70B)이 생성한 댓글과 인간 댓글의 유사성을 비교한다. 연구 결과는 LLM이 원래 게시물의 맥락만으로도 의미를 이해하고 반응할 수 있음을 나타낸다.

- **Performance Highlights**: 연구 결과는 사회적 관계 정보가 포함된 프롬프트를 사용하더라도 LLM이 생성한 댓글과 인간 댓글의 유사성이 떨어진다는 것을 보여준다. 이는 LLM이 훈련 데이터로 사회적 맥락 정보를 포함하지 않았기 때문일 수 있다. 전반적으로 LLM이 의미를 이해하고 인간 댓글과 유사한 방식으로 반응할 수 있는 능력을 보여주지만 개인화된 댓글의 일반화에 한계가 있음을 강조한다.



### Interpretable Contrastive Monte Carlo Tree Search Reasoning (https://arxiv.org/abs/2410.01707)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)에 대한 새로운 Monte Carlo Tree Search (MCTS) 알고리즘인 SC-MCTS*를 제안합니다. 이 알고리즘은 이전의 MCTS 기반 LLM 추론 방식에서 간과되었던 속도 저하 문제를 해결하고, 각 구성요소에 대한 심층 분석과 실험을 통해서 더 정확한 추론 성능을 이끌어냈습니다.

- **Technical Details**: SC-MCTS*는 contrastive decoding 원칙에 기반하여 설계된 해석 가능한 reward 모델을 도입하고, speculative decoding을 사용해 node 당 평균 51.9%의 속도 향상을 달성하였습니다. 또한, UCT node 선택 전략 및 backpropagation 기법을 개선하여 성능을 크게 향상시켰습니다.

- **Performance Highlights**: Llama-3.1-70B 모델을 사용하여 Blocksworld 다단계 추론 데이터셋에서 OpenAI의 o1-mini 모델보다 평균 17.4% 더 뛰어난 성과를 기록했습니다.



### An Exploration of Self-Supervised Mutual Information Alignment for Multi-Task Settings (https://arxiv.org/abs/2410.01704)
- **What's New**: 최근 언어 모델의 응답을 개인 속성과 선호에 맞게 조정할 수 있는 플루랄리스틱 정렬 방법의 필요성이 증가하고 있습니다. 본 연구에서는 Self-Supervised Alignment with Mutual Information (SAMI) 방법을 소개하며, 이를 통해 행동 선호와 모델 응답 간의 연결을 촉진합니다.

- **Technical Details**: SAMI는 조건부 상호 정보 (conditional mutual information)를 사용하여 다중 작업 환경에서 성능을 향상시킵니다. SAMI는 Direct Preference Optimization (DPO)와 비교되어, 다양한 카테고리에서 훈련된 강력한 모델을 통해 약한 모델의 세부 조정을 수행합니다. 본 실험에서는 MT-Bench 및 GSM-8K 벤치마크를 사용하여 수학적 정확성을 평가합니다.

- **Performance Highlights**: SAMI는 DPO에 대해 57%의 승률을 기록했으며, SFT와 비교할 때 수학적 정확성이 1.1% 향상됩니다. 10번의 시도에서 SAMI는 3.9%의 정확도 개선 효과가 있었고, SFT는 10.1% 향상되었습니다. SAMI와 SFT를 결합했을 때 다중 시도 환경에서 1.3%의 추가 향상이 나타났습니다.



### FactAlign: Long-form Factuality Alignment of Large Language Models (https://arxiv.org/abs/2410.01691)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 이 논문에서는 LLM의 긴형 반응의 사실성을 향상시키기 위한 새로운 정렬 프레임워크인 FactAlign을 제안합니다. 해당 프레임워크는 LLM의 유용성을 유지하면서 사실성을 높이는 것을 목표로 합니다.

- **Technical Details**: FactAlign은 fKTO라는 세부 조정 문장 정렬 알고리즘을 도입하며, 이는 Kahneman-Tversky Optimization (KTO) 방법을 확장하여 문장 수준의 정렬을 가능하게 합니다. 이 알고리즘은 자동 사실성 평가의 최신 발전을 활용하여 세부적인 사실성 평가를 통해 정렬 프로세스를 안내합니다.

- **Performance Highlights**: 실험 결과, FactAlign은 LLM의 긴형 반응의 사실성을 유의미하게 향상시켰으며, 동시에 유용성을 개선하는 데도 기여했습니다. FactAlign은 LLM이 사실성을 유지하면서 더 많은 정보를 제공하도록 훈련할 수 있다는 것을 보여주었고, 사실적 F1 스코어도 개선되었습니다.



### Trying to be human: Linguistic traces of stochastic empathy in language models (https://arxiv.org/abs/2410.01675)
Comments:
          preprint

- **What's New**: 이번 연구는 AI가 생성한 콘텐츠와 인간이 작성한 콘텐츠를 구별하는 데 있어 감정이입(empaty)과 인간처럼 보이려는 인센티브가 어떻게 작용하는지를 실험을 통해 조사했습니다.

- **Technical Details**: 두 가지 실험(Study 1과 Study 2)을 통해 참가자들이 관계에 대한 조언 또는 단순한 설명을 작성하도록 했으며, LLM은 가능한 한 인간처럼 텍스트를 작성하도록 지시받았습니다. 이후 새로운 샘플의 인간들이 텍스트의 출처를 판단했습니다.

- **Performance Highlights**: 연구 결과, 감정이입이 필요한 상황에서는 인간이 더 우수한 성과를 보였으나, 인간처럼 보이려는 지시는 오히려 LLM에 효과적이어서 인간의 우세가 줄어드는 경향을 보였습니다.



### Bridging Context Gaps: Leveraging Coreference Resolution for Long Contextual Understanding (https://arxiv.org/abs/2410.01671)
Comments:
          Underreview version of LQCA, Bridge context gap for long context

- **What's New**: 본 논문에서는 Long Question Coreference Adaptation (LQCA) 방법을 소개하여 긴 컨텍스트에서 핵심 참조(coreference)를 해결함으로써 대화형 AI 모델의 성능을 향상시키는 새로운 접근 방식을 제공합니다.

- **Technical Details**: LQCA 방법은 네 가지 주요 단계로 구성되어 있습니다: (1) 서브 문서 내에서의 핵심 참조 해결, (2) 언급 간의 거리 계산, (3) 핵심 참조를 위한 대표 언급 정의, (4) 언급 교체를 통한 질문 응답.

- **Performance Highlights**: 실험 평가 결과, OpenAI-o1-mini 및 GPT-4o 모델에서 긍정적인 결과가 나타났으며, 핵심 참조 해결 기술을 활용하여 질문 응답에서 컨텍스트 간의 간극을 효과적으로 메우는 것을 확인했습니다.



### Efficient Long-range Language Modeling with Self-supervised Causal Retrieva (https://arxiv.org/abs/2410.01651)
Comments:
          preprint

- **What's New**: 본 논문은 Grouped Cross-Attention (GCA)이라는 새로운 모듈을 제안하여 조합된 사전 훈련을 통해 RLM과 인과 언어 모델을 개선합니다. 이는 긴 컨텍스트 모델링을 돕는 혁신적인 방법입니다.

- **Technical Details**: GCA는 입력 시퀀스를 청크 단위로 나누고 현재 청크를 사용하여 이전 청크를 검색하는 방식을 갖습니다. 이를 통해 자동 회귀 손실을 최소화하는 청크 검색이 효율적인 방식으로 가능해졌습니다. 또한, Differentiable Retrieval-based Transformers (DRT)로 불리는 이 모델은 최대 64K 토큰까지 사전 훈련을 할 수 있으며, 메모리 효율성을 높이기 위해 이전 청크의 숨겨진 상태를 CPU 메모리로 오프로드합니다.

- **Performance Highlights**: DRT는 긴 범위의 언어 모델링에서 이전 모델에 비해 낮은 perplexity를 기록했으며, 훈련 시간은 단 50% 소요됩니다. 또한, 메모리 오프로드가 활성화된 상태에서, 추론 속도는 약 100배 향상되었습니다.



### DeIDClinic: A Multi-Layered Framework for De-identification of Clinical Free-text Data (https://arxiv.org/abs/2410.01648)
Comments:
          ongoing work

- **What's New**: 이 논문은 MASK 프레임워크를 개선하여 ClinicalBERT 모델을 통합함으로써 진료 텍스트에서의 개인 정보 비식별화(de-identification) 성능을 향상시키는 새로운 접근 방식을 제안합니다. 이 시스템은 전통적인 방법인 딕셔너리 검색(dictionary lookup) 및 규칙 기반(rule-based) 접근 방식과 함께 사용됩니다.

- **Technical Details**: SYSTEM: MASK + ClinicalBERT (딥러닝 모델) 
PERFORMANCE: 0.9732 F1-score (개체 인식 성능, 이름, 날짜, 위치 등)
RISK ASSESSMENT: 문서의 고유성을 분석하여 위험 수준을 분류.

- **Performance Highlights**: SYSTEM ENHANCEMENTS: 
1. Multi-layered PHI Entity Recognition 통합 (ClinicalBERT 활용) 
2. 효과적인 Masking 메커니즘 (Redaction 및 대체 방법 포함)
3. 사용자 친화적인 인터페이스 제공 (설정, 엔티티 관리, 배치 처리 지원)



### On The Adaptation of Unlimiformer for Decoder-Only Transformers (https://arxiv.org/abs/2410.01637)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 Unlimiformer 구조를 디코더 전용(Decoder-only) 트랜스포머에 적응시키기 위한 일련의 수정 사항을 제시합니다. 이 수정사항들은 크로스 어텐션(Cross-attention) 공식을 정보 융합으로 수정하고, 인덱스 생성 절차를 업데이트하며, 인덱스의 노후화 문제를 해결하고, 인과 어텐션(Causal attention) 지원을 위한 청크 인코딩 절차를 조정하는 것을 포함합니다.

- **Technical Details**: Unlimiformer는 입력을 겹치는 청크(Chunks)로 나누고, 각 청크를 인코딩하여 kNN 인덱스에 중간 반은 숨겨진 상태를 저장한 후, 디코더에서 밀집 어텐션(Dense attention)을 근사화하는 방식으로 작동합니다. 연구진은 디코더 전용 모델에 대한 호환성 문제를 해결하기 위해 새로운 평가 설정을 도입하고, 요약 및 자유형 Q&A(Free-form Q&A) 작업으로 나누어 실험을 진행했습니다.

- **Performance Highlights**: 수정된 Unlimiformer는 요약 데이터셋에서 성능이 2배 컨텍스트 길이를 가진 모델과 동등한 결과를 보여주었으며, 자유형 Q&A 및 지시형 튜닝 모델에 대한 한계와 미래 방향에 대해서도 논의했습니다.



### Intent Detection in the Age of LLMs (https://arxiv.org/abs/2410.01627)
Comments:
          Accepted at EMNLP 2024 Industry Track

- **What's New**: 이번 연구는 LLMs(대규모 언어 모델)를 활용한 새로운 의도 감지 기법을 제안하며, 이를 통해 기존의 sentence transformer 모델(SetFit)과 비교하여 성능과 지연 시간(latency) 사이의 균형을 맞추고자 하였습니다.

- **Technical Details**: 연구에서는 7가지 SOTA LLM을 적응형 in-context learning(ICL)과 chain-of-thought(CoT) 프롬프트를 사용하여 의도 감지를 수행하였고, 불확실성 기반 라우팅 전략과 부정적 데이터 증강을 통합한 하이브리드 시스템을 제안하였습니다. 제안된 시스템은 LLM의 성능을 2% 이내로 유지하면서 50% 적은 지연 시간으로 사용할 수 있었습니다.

- **Performance Highlights**: 제안된 두 단계 접근법은 Mistral-7B 모델에서 OOS(Out-Of-Scope) 감지 정확도 및 F1 점수를 5% 이상 향상시키는 성과를 보였습니다. 이는 LLM의 의도 감지 및 OOS 감지 역량 향상에 기여하였습니다.



### Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging (https://arxiv.org/abs/2410.01610)
Comments:
          work in progress

- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 모델을 효과적으로 튜닝할 수 있는 데이터 효율적인 접근 방식인 Upcycling Instruction Tuning (UpIT)을 제안합니다. 기존 방법들이 대규모 후속 훈련에 의존하는 반면, UpIT는 중간 체크포인트에서 전문화된 전문가를 구축하는 과정을 활용합니다.

- **Technical Details**: UpIT는 네 가지 주요 단계로 구성됩니다: (1) 전문가 준비 - 미리 저장된 체크포인트들을 이용하여 전문화된 전문가를 위한 기본 지원을 준비합니다. (2) 전문가 확장 - 유전자 알고리즘을 통하여 새로운 전문화를 가진 전문가들을 확장합니다. (3) 라우터 초기화 - 각 전문가 전용의 데이터를 할당하고 이진 분류 손실을 사용하여 라우팅 벡터를 최적화합니다. (4) 모델 업사이클링 - 여러 밀집 모델의 파라미터를 MoE 모델로 병합합니다.

- **Performance Highlights**: 다양한 데이터 규모와 업사이클링 설정에서의 실험 결과, UpIT는 기존 방법들보다 일관되게 우수한 성능을 나타냈습니다. 특히 작은 훈련 데이터 환경에서도 전통적인 밀집 튜닝 방법보다 효과적으로 결과를 개선하였으며, 전문가의 다양성을 유지하여 최종 MoE 모델의 성능을 더욱 향상시켰습니다.



### Spoken Grammar Assessment Using LLM (https://arxiv.org/abs/2410.01579)
Comments:
          5 pages, 2 figures

- **What's New**: 본 논문은 기존의 Spoken Language Assessment (SLA) 시스템을 혁신하여 구술 표현에서의 문법 평가를 통합하는 새로운 end-to-end SLA 시스템을 제안합니다. 이 시스템은 기존의 Written Language Assessment (WLA) 도구의 필요성을 없애고, 평가 항목을 다양하게 변형하여 학습자의 사전 준비를 어렵게 만듭니다.

- **Technical Details**: SLA 시스템은 두 가지 주요 구성 요소로 나뉘어 있으며, 첫 번째 부분은 대화형 언어 모델(LLM)을 활용하여 평가용 문단을 생성하고, 두 번째 부분은 후보자가 발화한 음성 오디오를 기준으로 문법 평가를 실시합니다. hybrid automatic speech recognition (ASR) 시스템과 맞춤형 언어 모델을 사용하여 문법 오류를 자동으로 평가하며, ASR의 오류에 강한 문법 점수 매기기 모듈을 포함합니다.

- **Performance Highlights**: 저자들은 제안한 시스템이 최첨단 ASR 엔진을 초월하여 구술 문법 평가에서 우수한 성능을 발휘한다고 주장합니다. 기존 WLA 시스템 없이도 모든 언어 능력의 평가를 가능하게 하여 테스트 소요 시간을 크게 단축시킬 수 있습니다.



### OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data (https://arxiv.org/abs/2410.01560)
- **What's New**: 이 논문에서는 수학적 추론(mathematical reasoning)을 위한 고품질의 finetuning (SFT) 데이터셋을 생성하기 위한 연구를 다루고 있습니다. 특히, OpenMathInstruct-2 데이터셋을 통해 기존의 공개 수학 추론 데이터셋보다 약 8배 더 큰 규모로 1,400만 개의 질문-답변 쌍을 제공합니다.

- **Technical Details**: 연구에서 	exttt{Llama3.1} 모델을 사용하여 데이터 합성(data synthesis)에 대한 철저한 ablation 실험을 진행하였으며, 주요 발견 사항으로는 (a) 솔루션 형식(solution format)의 중요성, (b) 강력한 teacher 모델이 생성한 데이터가 약한 student 모델의 on-policy 데이터보다 우수함, (c) 저품질 솔루션에 강한 SFT의 내구성, (d) 질문 다양성(question diversity)의 중요성을 제시했습니다.

- **Performance Highlights**: OpenMathInstruct-2로 	exttt{Llama-3.1-8B-Base} 모델을 finetuning한 결과, 	exttt{Llama3.1-8B-Instruct} 모델보다 MATH에서 절대 15.9% 향상된 성능(51.9% → 67.8%)을 보였습니다. 또한, 본 연구의 모형 및 코드와 OpenMathInstruct-2 데이터셋을 상업적 허용 라이센스 하에 공개하여 오픈소스 활동을 가속화하고자 했습니다.



### Integrative Decoding: Improve Factuality via Implicit Self-consistency (https://arxiv.org/abs/2410.01556)
- **What's New**: 이 논문은 Integrative Decoding (ID)라는 새로운 디코딩 전략을 소개하여 오픈-엔디드 생성(task)에서의 자기 일관성(self-consistency) 활용 가능성을 높입니다. 즉, 자기 일관성을 이용해 대규모 언어 모델의 사실 정확도를 향상시키는 방법을 제시합니다.

- **Technical Details**: ID는 각 샘플링된 응답을 원본 프롬프트와 연결하여 새로운 입력을 구성하고, 이 입력을 동시에 처리하여 최종 토큰을 선택합니다. 이 과정에서 각 입력은 샘플링된 응답의 '대표' 역할을 하며, 언어 모델의 다양한 응답 간의 일관성을 집계합니다. 기존 방법은 엄격한 형식 제약이 있는 반면, ID는 형식에 대한 제약이 없고 추가적인 프롬프트가 필요하지 않아서 더 넓은 범위에 적용 가능합니다.

- **Performance Highlights**: ID는 TruthfulQA (+11.2%), Biographies (+15.4%), LongFact (+8.5%) 벤치마크에서 사실성을 일관되게 높이며, 샘플링된 응답 수가 증가함에 따라 성능 향상이 점진적으로 증대하여 반복 샘플링에 대한 확장 가능성을 보여줍니다.



### ACE: A LLM-based Negotiation Coaching System (https://arxiv.org/abs/2410.01555)
Comments:
          EMNLP 2024

- **What's New**: 이 논문은 LLM 기반의 협상 코칭 시스템인 ACE를 제안하며, 이를 통해 사용자에게 개인 맞춤형 피드백을 제공하여 협상 능력을 향상시키고자 한다.

- **Technical Details**: ACE는 MBA 학생들 간의 협상 대화 기록을 수집하여, 사용자들이 협상 중 저지를 수 있는 실수를 식별하고 이를 교정하기 위한 주석 체계를 개발한다. 이 시스템은 실제 비즈니스 스쿨 교육과정을 기반으로 하여 설계된 협상 시나리오를 사용하고, 사용자의 오류를 대상으로 한 피드백을 제공한다.

- **Performance Highlights**: 사용자 실험을 통해 ACE가 제공하는 피드백이 학습 성과를 유의미하게 향상시킨 것을 발견하였다. ACE는 피드백을 제공하지 않는 시스템이나 대안적인 피드백 제공 방식과 비교하여 훨씬 뛰어난 성능을 보였다.



### In-Context Transfer Learning: Demonstration Synthesis by Transferring Similar Tasks (https://arxiv.org/abs/2410.01548)
- **What's New**: 본 논문은 In-context Transfer Learning (ICTL)라는 새로운 접근법을 제안하여 LLM(대형 언어 모델)에서 고비용의 라벨링 작업을 줄이면서 다양한 작업에 적응하도록 돕는다. 이 방법은 유사한 소스 작업으로부터 라벨이 붙은 데모를 전이하여 대상 작업 데모를 합성하는 것이다.

- **Technical Details**: ICTL은 두 가지 단계로 구성된다: 소스 샘플링과 대상 전이. 첫 번째 단계에서는 최적화 목표를 정하여 대상 작업과 유사한 소스 데모를 샘플링하고, 두 번째 단계에서는 LLM을 사용하여 샘플링한 소스 데모를 대상 작업 정의에 맞게 전이한다. 특히, 전이 오차를 최소화하여 소스 데모의 유사성을 보장하는 최적화 목표를 제시한다.

- **Performance Highlights**: Super-NaturalInstructions (Super-NI) 데이터셋에 대한 실험 결과, ICTL은 처음부터 샘플을 합성하는 방법에 비해 평균 2.0%의 성능 향상을 달성하였다. 이러한 결과는 ICTL의 효과성을 보여준다.



### Seeing Eye to AI: Human Alignment via Gaze-Based Response Rewards for Large Language Models (https://arxiv.org/abs/2410.01532)
- **What's New**: 이 연구에서는 GazeReward라는 새로운 프레임워크를 소개하여 눈 추적(eye-tracking, ET) 데이터를 보상을 생성하는 모델에 통합합니다. 이 프레임워크는 사용자의 선호도를 보다 정확하게 반영하도록 설계되었습니다.

- **Technical Details**: GazeReward는 눈의 움직임과 고정 시점을 측정하여 사용자의 인지 및 지각 과정을 이해하는 데 도움을 주며, 이를 통해 더욱 높은 정확도의 보상 모델(Reward Model, RM)을 개발합니다. 연구에서는 여러 LLM과 ET 생성 모델을 이용한 메커니즘 비교 연구를 통해 GazeReward의 효과를 입증하였습니다.

- **Performance Highlights**: GazeReward를 적용한 실험에서, 보상 모델의 정확성이 기존 방법보다 20% 이상 향상됨을 보여주었습니다. 이는 다양한 인간 선호 데이터셋에서 검증되었습니다.



### HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models (https://arxiv.org/abs/2410.01524)
- **What's New**: 이번 연구는 큰 언어 모델(LLM)의 안전성 강화를 위한 데이터 증강 기법인 HarmAug를 소개합니다. 이를 통해 기존의 큰 모델을 작은 모델로 증류하여 모바일 환경에서의 실용성을 높였습니다.

- **Technical Details**: HarmAug는 LLM을 ‘jailbreak’하여 해로운 지시사항을 생성하도록 유도하는 단순하면서도 효과적인 방법입니다. 이 방식은 지시-응답 쌍 데이터셋을 사용하여 소규모 모델의 성능을 향상시키며, 435백만 파라미터의 DeBERTa 모델이 7억 파라미터 이상의 모델과 유사한 성능을 달성하게 합니다.

- **Performance Highlights**: HarmAug를 사용하여 훈련된 안전성 모델은 AUPRC에서 더 큰 모델을 초과하며, 실행 비용을 75%까지 줄이는 데 성공했습니다. 이 모델은 적대적 프롬프트 탐지 및 해킹 공격에 대한 탐지에서 매우 효과적입니다.



### InfiniPot: Infinite Context Processing on Memory-Constrained LLMs (https://arxiv.org/abs/2410.01518)
Comments:
          EMNLP 2024 Main

- **What's New**: 이 논문은 InfiniPot이라는 새로운 KV 캐시 제어 프레임워크를 도입하여, Resource-제한이 있는 환경에서 사전 훈련된 LLM이 고정 메모리 제약 내에서 긴 입력 맥락을 효율적으로 처리할 수 있도록 돕습니다.

- **Technical Details**: InfiniPot은 Continual Context Distillation (CCD)라는 반복적 프로세스를 활용하여, 중요한 정보를 압축하고 보존합니다. CaP(Catalyst Prompt)와 NuC(Novelty under Compression)를 조합하여 중요한 데이터와 새롭게 등장한 정보를 효과적으로 관리합니다.

- **Performance Highlights**: InfiniPot은 여러 NLP 업무에서 긴 맥락을 처리하기 위해 특별히 훈련된 모델들보다 우수한 성능을 보여줍니다. 우리의 평가 결과는 InfiniPot이 LLM의 다양한 실제 사용 시나리오에 적합하다는 것을 보여줍니다.



### InstaTrans: An Instruction-Aware Translation Framework for Non-English Instruction Datasets (https://arxiv.org/abs/2410.01512)
- **What's New**: 최신 연구에서는 기존의 고품질 영어 지침 데이터셋을 비영어 데이터셋으로 번역하는 방법을 제안합니다. 이를 통해 번역의 완전성과 지침 인식을 보장하는 새로운 프레임워크 InstaTrans를 도입하였습니다.

- **Technical Details**: InstaTrans (INSTruction-Aware TRANSlation) 프레임워크는 고품질 영어 지침 데이터셋을 다른 언어로 번역하기 위해 특별히 설계되었습니다. 이 방법은 GPT-4를 활용하여 소규모 샘플을 번역한 후, 이를 기반으로 LLM을 미세 조정하여 번역 품질을 향상시키고 있습니다.

- **Performance Highlights**: 실험 결과, InstaTrans를 통해 번역된 데이터셋은 지침의 완전성과 인식 정보를 높이며, 이를 통해 LLM의 성능이 목표 언어에서 효과적으로 개선됨을 보여주었습니다. 특히, 다양한 언어에서 LLM의 접근성을 높이는 잠재력이 강조됩니다.



### Disentangling Latent Shifts of In-Context Learning Through Self-Training (https://arxiv.org/abs/2410.01508)
- **What's New**: 본 논문에서는 Self-Training ICL(STICL)이라는 새로운 접근 방식을 소개합니다. 이는 demonstrations와 쿼리의 잠재적 이동(latent shifts)을 구분하여 안정성 및 긴 컨텍스트 문제를 해결하고자 합니다.

- **Technical Details**: STICL은 teacher model을 사용하여 pseudo-labels를 생성하고, 이 labels로 student model을 훈련하는 구조입니다. latent shifts를 disentangle하여 demonstrations로부터 독립적으로 쿼리를 처리할 수 있게 하여, 모델의 안정성과 효율성을 높입니다. 또한, adapter module을 통해 demonstrations의 정보를 인코딩하여 활용합니다.

- **Performance Highlights**: STICL은 autoregressive LLMs인 Llama 3 및 Phi 3을 사용하여 GLUE와 MMLU 벤치마크에서 평가되었습니다. 결과적으로, STICL은 전통적인 ICL 방법과 다른 disentangling 방법들에 비해 일관되게 우수한 성능을 보였으며, 안정성과 일반화 능력을 향상시켰습니다.



### PersonaMath: Enhancing Math Reasoning through Persona-Driven Data Augmentation (https://arxiv.org/abs/2410.01504)
- **What's New**: 이번 논문에서는 데이터 증강(data augmentation)을 통한 open-source 모델의 수학 문제 해결 능력 향상 방안을 제안합니다. 특히, PersonaMathQA라는 신규 데이터셋을 소개하며, 이를 통해 PersonaMath 모델을 학습시켰습니다.

- **Technical Details**: 제안된 방법은 두 단계로 나뉩니다. 첫 번째 단계인 'Persona Diversification'에서는 강력한 closed-source LLM을 활용해 상세한 Chain-of-Thought (CoT) 솔루션을 생성하고, 이를 다양한 페르소나(persona)를 통해 재작성하여 데이터셋의 양과 다양성을 향상시킵니다. 두 번째 단계인 'Reflection'에서는 처음에 잘못 답한 질문에 대해 LLM이 자신의 실수를 반성하고 올바른 답변을 생성하도록 하여, 이러한 질문들을 최종 데이터셋에 더 중점적으로 포함시킵니다.

- **Performance Highlights**: 최종적으로, PersonaMath-7B 모델은 MATH 데이터셋에서 24.2%의 정확도, GSM8K에서 68.7%의 정확도를 기록하여 기존의 모든 기준 방법들을 초월하며, 새로운 최고 성능(SOTA)을 달성했습니다. 이번 연구를 통해 제안한 데이터셋이 기존의 대규모 데이터셋에 비해 적은 데이터 양에도 불구하고 더 높은 품질과 다양성을 보유하고 있음을 입증했습니다.



### DLP-LoRA: Efficient Task-Specific LoRA Fusion with a Dynamic, Lightweight Plugin for Large Language Models (https://arxiv.org/abs/2410.01497)
Comments:
          Preprint under review, 18 pages, 7 figures

- **What's New**: DLP-LoRA는 문장 수준에서 여러 LoRA를 동적으로 융합하기 위해 mini-MLP 모듈을 사용하는 Dynamic Lightweight Plugin이며, 이는 효율성을 크게 향상시킨다.

- **Technical Details**: DLP-LoRA는 약 5M의 파라미터를 가진 mini-MLP로 구성되어 있으며, top-p 샘플링 전략을 통해 다중 LoRA를 동적으로 융합한다. 이를 통해 기존의 token-level 방식보다 더 빠른 추론 성능을 제공한다.

- **Performance Highlights**: DLP-LoRA는 26개 작업에서 평균 92.34%의 정확도를 기록했고, 9개의 질문-답변(QA) 데이터셋에서는 BLEU 및 ROUGE 점수가 크게 향상되었다. 특히, MCQ와 QA 작업에서 각각 92.95%와 13.2%의 상대적인 개선을 보였다.



### Extending Context Window of Large Language Models from a Distributional Perspectiv (https://arxiv.org/abs/2410.01490)
Comments:
          14 pages, 8 figures, Accepted to EMNLP2024

- **What's New**: 본 논문은 rotary position embedding (RoPE)을 기반으로 한 대규모 언어 모델(LLMs)의 맥락 창(context window) 확장 문제를 새로운 시각에서 접근하여 개선하는 방법을 제안합니다. 기존의 경험적 방법들과 달리, 로테리(angle) 분포를 기반으로 한 최적화 방법을 통해 성능을 강화하였습니다.

- **Technical Details**: 연구진은 먼저 모델 내의 로테리 각도 분포를 추정하고 맥락 창의 길이 확장이 이 분포에 미치는 영향을 분석합니다. 이어서 미세 조정 없이 로테리 각도 분포 간의 간섭을 최소화하여 일관성을 유지하는 새로운 확장 전략을 제시합니다. 실험을 통해, LLaMA2의 맥락 창을 8k와 16k로 확장할 때 각각 72% 및 32%의 분포적 간섭 감소를 달성했습니다.

- **Performance Highlights**: LongBench-E 벤치마크에서 기존의 최첨단 방법들에 비해 평균 4.33% 향상을 보였으며, Hugging Face Open LLM 벤치마크 상에서도 맥락 창 확장 이후 평균 성능 변동이 -0.12에서 +0.22로 안정적으로 유지되었습니다.



### Small Language Models Like Small Vocabularies: Probing the Linguistic Abilities of Grapheme- and Phoneme-Based Baby Llamas (https://arxiv.org/abs/2410.01487)
- **What's New**: 이번 논문은 Byte Pair Encoding과 같은 서브워드(subword) 기반 토큰화(tokenization) 알고리즘의 유효성을 의문시하며, 토큰화가 필요 없는 음소(phoneme) 및 그래프(표기) 기반 언어 모델의 가능성을 탐구합니다.

- **Technical Details**: Llama 아키텍처에 기반한 소형 모델이 문자 수준(vocabulary)으로 학습되었을 때, 표준 구문(syntactic) 및 혁신적인 어휘(lexical)/음성적(phonetic) 기준에서도 강력한 언어적 성능을 달성할 수 있음을 보입니다. 음소 기반 모델은 어떤 그래프 편향(graphemic biases) 없이도 표준 작업과 새로운 평가에서 그래프 기반 모델에 거의 비슷한 성능을 보였습니다.

- **Performance Highlights**: 이 연구 결과는 언어 습득(acquisition) 및 처리(processing)에 대한 계산적 연구에 더 적합한 언어 모델을 만드는 데 있어 유망한 방향성을 제시합니다.



### A Little Goes a Long Way: Efficient Long Context Training and Inference with Partial Contexts (https://arxiv.org/abs/2410.01485)
- **What's New**: 이번 논문에서는 LongGen이라는 새로운 아키텍처를 제안하여 길이 확장(length extension)과 KV 캐시(KV cache) 감소를 통합함으로써, 대형 언어 모델(LLM)의 훈련 및 제공에서 발생하는 오버헤드를 줄이는 방안을 제시합니다.

- **Technical Details**: LongGen은 긴 컨텍스트 데이터를 처리하기 위해 스파스 어텐션(sparse attention) 패턴을 사용하여 효율적인 아키텍처를 구축합니다. 여기에는 윈도우 어텐션(window attention), 어텐션 싱크(attention sink), 블록 와이즈 스파스 어텐션(blockwise sparse attention) 등이 포함됩니다. 하이브리드 아키텍처에서 1/3의 전체 어텐션(full attention) 계층과 2/3의 효율적인 계층을 결합하여 효율성과 긴 컨텍스트 성능 간의 균형을 맞춥니다.

- **Performance Highlights**: LongGen은 128K 길이의 컨텍스트로 훈련할 때 1.55배의 훈련 속도 향상 및 전체 시간을 36% 감소시켰습니다. 추론 단계에서는 KV 캐시 메모리를 62% 줄이며, 프리필링(prefilling) 속도를 1.67배, 디코딩 속도를 1.41배 향상시켰습니다.



### Agent-Driven Large Language Models for Mandarin Lyric Generation (https://arxiv.org/abs/2410.01450)
Comments:
          6 pages, figures, Accepted at O-COCOSDA 2024

- **What's New**: 이번 연구에서는 멜로디와 가사의 조화를 고려한 multi-agent 시스템을 개발하여 가사 생성 과제를 세분화하였습니다. 각 에이전트는 운율, 음절 수, 가사-멜로디 정렬 및 일관성을 제어합니다.

- **Technical Details**: 연구는 Mpop600 데이터셋에 기반하여, 음계 텍스트와 멜로디 간의 정렬을 학습하는 방법론을 제안합니다. 새로운 접근법으로 음파 변환(singing voice synthesis)을 사용하여 다양한 에이전트 그룹이 생성한 가사의 품질을 평가합니다.

- **Performance Highlights**: 가사-멜로디 생성의 품질 평가를 위한 청취 실험이 수행되었으며, 하모니 알리그먼트(harmony alignment)와 감각적 조화를 통해 멜로디에 적합한 가사를 생성하는 데 성공했습니다.



### Geometric Signatures of Compositionality Across a Language Model's Lifetim (https://arxiv.org/abs/2410.01444)
Comments:
          Under review as a conference paper at ICLR 2025

- **What's New**: 이 연구는 인공 언어 모델의 조합 일반화(compositional generalization) 능력과 관련하여, 표현의 의미가 어떻게 구성되어 있고, 이러한 능력의 기저에 있는 표현 메커니즘을 분석합니다. 처음으로 데이터 세트의 조합성 정도가 LM의 표현의 내재적 차원(intrinsic dimensionality)에 어떻게 반영되는지를 조사합니다.

- **Technical Details**: 연구는 조합의 두 가지 유형, 즉 형태의 조합(compositionality of form)과 의미의 조합(compositionality of meaning)을 구별하며, 비선형 및 선형 차원으로 두 가지 차원성을 측정합니다. LMs은 전달할 프레이즈의 통계적 규칙을 학습하고, 그 과정에서 조합적 복잡성(combinatorial complexity)을 인코딩합니다. 이들은 언어 처리에서 형식과 의미의 차이를 구분할 수 있게 해줍니다.

- **Performance Highlights**: 연구에서는 비선형 ID가 의미 조합성을 인코딩하고 선형 차원이 형태 조합성과 높은 상관관계를 갖는다는 것을 발견했습니다. 이는 LMs가 어떻게 언어를 처리하는 방식에서 의미와 형태가 관련이 있음을 시사합니다. 결과적으로 비선형과 선형 표현의 복잡성이 언어 모델의 훈련 과정에서 어떻게 달라지는지를 보여주었습니다.



### Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks (https://arxiv.org/abs/2410.01428)
Comments:
          Work in progress

- **What's New**: CR-Planner는 복잡한 문제 해결을 위한 새로운 프레임워크로, 비판 모델(critic model)을 활용하여 추론과 정보 검색 과정을 계획합니다. 이 시스템은 복잡한 도메인 지식과 추론을 요구하는 문제에 효과적으로 대응할 수 있도록 설계되었습니다.

- **Technical Details**: CR-Planner는 두 가지 주요 컴포넌트, 즉 Sub-Goal 선택과 Execution 선택을 통해 작동합니다. Sub-Goal 선택 과정에서는 sub-goal critic 모델이 주어진 상태에서 Reason, GenQuery, Retrieve 중 최적의 서브 목표를 선택하도록 돕고, Execution 선택 과정에서는 각 서브 목표를 실현하기 위한 여러 실행 후보를 생성하여 execution critic 모델이 최적의 실행을 선택합니다. 이때, CR-Planner는 Monte Carlo Tree Search(MCTS)를 활용하여 비판 모델 훈련을 위한 데이터를 수집합니다.

- **Performance Highlights**: CR-Planner는 경쟁 프로그래밍, 정리 기반 수학 추론, 복잡한 도메인 검색 문제 등 어려운 과제에서 기존 벤치마크보다 평균 10.06% 더 높은 성능을 보였습니다. 이는 비판 모델의 도움을 통해 추론과 검색의 정확성을 크게 향상시킨 것입니다.



### Question-guided Knowledge Graph Re-scoring and Injection for Knowledge Graph Question Answering (https://arxiv.org/abs/2410.01401)
Comments:
          findings of EMNLP2024

- **What's New**: 본 논문은 Knowledge Graph Question Answering(KGQA) 과정에서 발생하는 노이즈 경로 문제에 대한 해결책을 제시합니다. 제안하는 Question-guided Knowledge Graph Re-scoring method (Q-KGR)를 통해 입력 질문과 관련이 없는 정보를 제거하여 정밀한 지식에 집중할 수 있도록 돕습니다.

- **Technical Details**: Q-KGR은 질문과 각 엣지의 의미적 유사성을 계산하여 엣지에 관련 점수를 부여하고, 이를 기반으로 재-점수화된 지식 그래프를 만듭니다. 이후 이 정보를 LLM에 주입하기 위한 맞춤형 트랜스포머인 Knowformer를 설계했습니다. Knowformer는 LoRA(Hu et al., 2021)와 결합하여 효과적인 작업 적응을 가능하게 합니다.

- **Performance Highlights**: 다양한 KGQA 벤치마크에 대한 대규모 실험을 통해 Q-KGR 및 Knowformer 방법이 기존 시스템보다 우수한 성능을 발휘함을 입증했습니다. 또한, 이 방법은 LLM이 정확한 사실적 지식에 집중하도록 유도하며, 지식 정렬 및 주입에서 강력한 효과를 보입니다.



### CrowdCounter: A benchmark type-specific multi-target counterspeech datas (https://arxiv.org/abs/2410.01400)
Comments:
          19 pages, 1 figure, 14 tables, Code available this https URL

- **What's New**: 이 연구에서는 증오 발언에 대한 반응으로 효과적인 반응(conterspeech)을 작성하기 위한 새로운 데이터셋인 CrowdCounter를 소개합니다. 이는 3,425개의 증오 발언-반응 쌍으로, 6가지 타입의 반응(공감(empathy), 유머(humor), 질문(questioning), 경고(warning), 비난(shaming), 반대(contradiction))을 포함합니다.

- **Technical Details**: CrowdCounter 데이터셋은 Crowd 기반 주석 플랫폼을 통해 수집되었습니다. 1325개의 고유한 증오 발언에서 수집한 3,425개의 쌍 이 데이터셋은 고품질 및 다양한 반응 생성을 목표로 하고 있습니다. 이 연구에서는 vanilla와 type-controlled prompt 두 가지 프레임워크를 활용하여 반응을 생성하고, Flan-T5 모델이 가장 효과적임을 보여줍니다.

- **Performance Highlights**: Flan-T5 모델은 vanilla 프레임워크에서 다양한 측면에서 가장 높은 성과를 보였습니다. 특정 타입에 대한 프롬프트 생성은 반응의 관련성을 높였습니다. DialoGPT 모델은 지침을 따르고 정확한 타입별 반응을 생성하는 데 가장 우수한 성과를 나타냈습니다.



### Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition (https://arxiv.org/abs/2410.01380)
- **What's New**: 이 연구는 모델의 파라메트릭 지식 통합 경향이 프리트레이닝(pretraining) 동안 어떻게 변화하는지와 이러한 행동이 전반적인 성능에 어떻게 영향을 미치는지를 조사합니다. 고유한 개념인 knowledge entropy를 도입하여 저자가 언급하는 모델의 지식 획득 및 망각에 대한 행동 변화를 정의합니다.

- **Technical Details**: knowledge entropy는 모델이 다양한 기억 소스를 통합하는 방식을 정량화하는 데 사용됩니다. 높은 knowledge entropy는 모델이 넓은 범위의 기억 소스를 활용함을 나타내고, 낮은 knowledge entropy는 특정 소스에 대한 의존을 시사합니다. 이 연구는 프리트레이닝이 진행됨에 따라 knowledge entropy가 일관되게 감소한다는 것을 발견했습니다.

- **Performance Highlights**: 모델의 후반 프리트레이닝 단계에서는 낮은 knowledge entropy가 지식 획득과 유지 능력의 감소와 관련이 있음을 발견했습니다. 특히, 비활성 기억 벡터의 활동을 증가시킴으로써 지식 획득과 망각을 향상시킬 수 있다는 점에서 이 연구 결과는 중요합니다.



### PCQPR: Proactive Conversational Question Planning with Reflection (https://arxiv.org/abs/2410.01363)
Comments:
          Accepted by EMNLP 2024 Main

- **What's New**: 이번 연구는 전통적인 대화형 질문 생성 방식에서 벗어나, 특정 결론에 도달하는 방향으로 대화를 주도하는 'Conclusion-driven Conversational Question Generation (CCQG)'이라는 새로운 작업 방식을 제안합니다.

- **Technical Details**: PCQPR(Proactive Conversational Question Planning with self-Refining) 프레임워크는 Monte Carlo Tree Search (MCTS)에서 영감을 받은 계획 알고리즘과 대규모 언어 모델(LLM)의 분석 능력을 결합하여, 미래 대화 전개를 예측하고 질문 전략을 세밀하게 조정합니다. 이 과정에서 LLM은 시뮬레이션된 대화 경로를 분석하여 피드백을 제공하며, 각 단계에서의 성공과 실패를 식별합니다.

- **Performance Highlights**: PCQPR은 기존 CQG 방법론에 비해 현저하게 높은 성능을 보여주며, 결론 중심의 대화형 질문-응답 시스템을 향한 패러다임 변화의 이정표가 됩니다.



### Assisted Data Annotation for Business Process Information Extraction from Textual Documents (https://arxiv.org/abs/2410.01356)
- **What's New**: 이 논문에서는 자연어 프로세스 설명에서 프로세스 모델을 생성하기 위한 데이터 세트 작성 지원을 위한 두 가지 조력 기능을 탐색합니다. 이 기능들은 추천 시스템과 시각화 도구를 포함하여 데이터 세트 작성자의 워크로드를 줄이고 주석 품질을 크게 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 프로세스 정보 추출을 위한 데이터 세트의 품질과 양이 부족한 문제를 해결하고, AI 기반 추천 시스템과 그래픽 비즈니스 프로세스 모델을 사용하여 주석 작업을 지원합니다. 제안된 도구는 프로토타입 주석 도구로 구현되었으며, 31명의 참가자를 대상으로 한 엄격한 사용자 연구를 통해 효과를 평가했습니다.

- **Performance Highlights**: AI를 통한 지원은 주석 작성의 주요 워크로드 메트릭을 51.0% 이상 감소시켰으며, 추출 품질이 38.9% 향상되었습니다. 초보자도 전문 주석가와 유사한 주석 품질에 도달할 수 있게 되어 새로운 데이터 주석가 양성을 가속화하는 데 도움이 됩니다.



### Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models (https://arxiv.org/abs/2410.01335)
Comments:
          11 main pages, 23 pages total, 9 figures, 5 tables

- **What's New**: 본 논문에서는 언어 특정 데이터가 부족한 비영어 사용자를 위한 대형 언어 모델(LLMs)의 수학적 추론 능력을 이전하는 새로운 모델 병합 방법론을 제안합니다. 이 방법론은 모델 수프의 원칙에 따라 두 개의 '전문가(experts)' 모델을 병합함으로써, 언어와 수학의 능력을 동시에 활용할 수 있게 합니다.

- **Technical Details**: 우리는 동일한 사전 학습된 모델에서 시작하여 각각 영어 수학 데이터와 목표 언어의 일반Instruction 데이터로 '전문가' 모델을 파인 튜닝합니다. 이후 수학 전문가의 최상위 및 최하위 transformer 레이어를 언어 전문가의 레이어로 직접 교체하여 목표 언어에서의 수학 성능을 향상시킵니다. 이 방법은 각 전문가를 파인 튜닝할 때 가장 중요한 파라미터 변화에 대한 해석적 분석에 기반하여 간단하고 비용이 적게 들며 직관적입니다.

- **Performance Highlights**: 병합된 모델은 수학 벤치마크인 MGSM에서 평균적으로 10% 더 나은 성능을 보이며, 스와힐리어, 텔루구어, 벵골어, 일본어 등 4개의 주요 언어에서 성능을 향상시킵니다. 특히, 스와힐리어에서는 혼합된 스와힐리 및 수학 데이터셋을 파인 튜닝한 모델의 성능을 초과하여 뛰어난 결과를 냅니다.



### Unveiling Language Skills under Circuits (https://arxiv.org/abs/2410.01334)
- **What's New**: 이 논문은 언어 모델의 메모리 읽기 기능을 독립적으로 조작하는 최소 단위인 Memory Circuit의 개념을 도입합니다. 이를 통해 Transformer 모델 내의 다양한 언어 기술을 더 명확하게 분리하여 분석할 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 논문에서는 Transformer 모델을 완전한 회로 그래프로 나누고, 언어 기술에 기여하는 주요 회로 경로를 추출하는 3단계 프레임워크를 개발합니다. 1단계에서는 Memory Circuits를 통해 각 레이어에서 메모리 기능을 독립적으로 나타내는 회로 그래프를 구성합니다. 2단계에서는 Greedy Search를 사용하여 불필요한 경로를 제거하고 3단계에서는 각 경로의 인과 효과를 추정하여 기술 경로를 선택합니다.

- **Performance Highlights**: 실험 결과, 이전 토큰 기술(Previous Token Skill), 유도 기술(Induction Skill), 그리고 상황 학습 기술(ICL Skill) 간의 관계를 입증하였고, 간단한 언어 기술은 얕은 레이어에, 복잡한 언어 기술은 깊은 레이어에 존재한다는 기존 가설을 검증하였습니다. 이를 통해 Chain-of-Thought(코드)의 재활용 가능성에 대한 새로운 증거를 제시합니다.



### Emotion-Aware Response Generation Using Affect-Enriched Embeddings with LLMs (https://arxiv.org/abs/2410.01306)
- **What's New**: 이 연구는 자동화된 챗봇을 통한 심리치료 세션에서 공감이 가며 일관된 반응을 제공하는 방법을 제안합니다. 특히, 대형 언어 모델(LLMs)의 감정 및 맥락 인식을 향상시키기 위한 새로운 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 NRC Emotion Lexicon, VADER, WordNet, SentiWordNet과 같은 여러 감정 어휘집을 통합하여 LLAMA 2, Flan-T5, ChatGPT 3.0, ChatGPT 4.0와 같은 최신 LLM과 결합합니다. 데이터셋은 상담 및 심리치료 데이터베이스에서 수집한 2,000개 이상의 치료 세션 전사로 구성되어 있으며, 불안, 우울증, 트라우마, 중독 관련 논의를 포함하고 있습니다. 전사를 작은 조각으로 분할하고, BERT, GPT-3, RoBERTa를 사용하여 임베딩을 계산하여 의미적 및 정서적 뉘앙스를 캡처합니다. 이 임베딩은 FAISS 벡터 데이터베이스에 저장되어, 코사인 유사성을 기반으로 효율적인 유사성 검색 및 클러스터링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 감정 어휘집을 통합함으로써 모델의 공감, 일관성, 정보성 및 유창성 점수가 향상되는 것으로 나타났습니다. 이러한 발견은 심리치료 응용을 위한 LLM 성능 향상에서 감정 임베딩의 중요한 역할을 강조합니다.



### Revisiting Hierarchical Text Classification: Inference and Metrics (https://arxiv.org/abs/2410.01305)
Comments:
          Accepted at CoNLL 2024

- **What's New**: 이번 논문에서는 Hierarchical Text Classification (HTC)의 평가 방법에 대해 새로운 접근 방식을 제시합니다. 기존의 다중 레이블 분류(multi-label classification) 문제로 HTC를 다루었던 것과 달리, 연구진은 특정한 계층적 메트릭(hierarchical metrics)에 기반하여 모델을 평가하는 방식을 도입하고, 예측 추론(prediction inference) 방법의 복잡성을 강조합니다. 또한, 새로운 도전적인 데이터셋인 Hierarchical Wikivitals (HWV)를 소개합니다.

- **Technical Details**: HTC는 텍스트에 계층적 구조를 가진 레이블을 할당하는 문제로서, 레이블은 트리(tree)나 유향 비순환 그래프(Directed Acyclic Graph, DAG) 형태로 조직됩니다. 본 논문은 오류의 심각성을 반영할 수 있는 메트릭을 요구하며, 모델 성능을 평가하기 위해 특히 설계된 계층적 메트릭을 사용합니다. 새로운 손실 함수(loss function)를 도입하여 최근 모델과 간단하지만 강력한 기준선(baseline) 모델과 비교하여 성능을 실험했습니다.

- **Performance Highlights**: 실험 결과, 최신 모델들이 반드시 계층 정보를 잘 인코딩하지 못한다는 것을 보여주었으며, HWV 데이터셋에서 우리는 간단한 손실 함수가 최신 모델에 비해 경쟁력을 발휘하는 것을 확인했습니다. 결과적으로, 새로운 방법론을 제안하는 경우 평가 방법론을 신중하게 고려하는 것이 중요하다는 점을 강조하였습니다.



### Endless Jailbreaks with Bijection Learning (https://arxiv.org/abs/2410.01294)
- **What's New**: 이번 연구에서는 기존 LLMs (Large Language Models)의 공격 방식인 bijection learning(바이젝션 학습)을 소개하며, 이는 언어 모델의 안전 메커니즘을 우회하는 새로운 접근법을 제시합니다.

- **Technical Details**: bijection learning은 임의의 문자열-문자열 매핑을 통해 모델에게 영어 평문을 'bijection language'로 인코딩하는 방법으로, 언어 모델의 내장 안전 메커니즘을 무력화합니다. 이 공격은 조건부 생성에서 다양한 매핑을 제공하며, 특히 대규모 모델에서 더 강력하게 작용합니다.

- **Performance Highlights**: Bijection learning은 다양한 현대 언어 모델에 대해 86.3%의 공격 성공률(Attack Success Rate, ASR)을 기록하며, 특정 해로운 요청에 대해 단일 고정 프롬프트 템플릿을 활용하여 폭넓은 해로운 응답을 유도할 수 있습니다.



### Mitigating Copy Bias in In-Context Learning through Neuron Pruning (https://arxiv.org/abs/2410.01288)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 In-Context Learning (ICL)에서 복사 편향(copied bias)을 완화하기 위한 새로운 방법을 제안하였다. 이를 통해 ICL의 오류 문제를 내부 프로세스와 관련된 신경 활성 패턴을 조사하여 해결하는 접근 방식을 취하였다.

- **Technical Details**: 제안된 접근법은 Integrated Gradients(IG) 방법을 사용하여 특정 신경 뉴런을 식별하고 이들 뉴런을 제거(pruning)함으로써 성능을 개선하는 것이다. 이 연구는 Transformer 및 상태 공간 모델(State-Space Models)과 같은 다양한 LLM 아키텍처에서 적용 가능하며, 입력 예제의 모임을 통해 제공되는 처리를 기반으로 작업을 수행하는 방식으로 진행되었다.

- **Performance Highlights**: 제안된 방법은 다양한 ICL 과제를 통해 성능을 향상시키며, 특히 작은 LLM 모델에서 높은 복사 오류율을 개선하는 결과를 보였다. 제거된 뉴런들이 효과적인 작업 인식을 방해했음을 나타내는 작업 벡터(task vectors)의 품질 향상을 통해 이러한 성능 향상을 확인하였다.



### Enhancing Training Data Attribution for Large Language Models with Fitting Error Consideration (https://arxiv.org/abs/2410.01285)
Comments:
          Accepted to the EMNLP 2024 main

- **What's New**: 본 논문에서는 Debias and Denoise Attribution (DDA)이라는 새로운 Training Data Attribution (TDA) 방법을 제안하며, 기존의 influence functions를 개선하여 모델 학습 중 발생할 수 있는 fitting errors를 다루고자 하였다.

- **Technical Details**: DDA는 두 가지 전략으로 구성된다. 첫째, debias 전략을 통해 기본 모델의 지식 편향을 제거하여 influence score를 수정하며, 둘째, denoise 전략을 통해 다양한 fitting 정도로 인해 발생하는 influence score의 불일치를 smoothing 기술을 통해 줄인다.

- **Performance Highlights**: 실험 결과, DDA는 기존 방법들(BM25, TracIN, TRAK, CEA)보다 우수한 성능을 보이며, 평균 AUC 93.49%를 기록하였다. 또한 DDA는 LLaMA2, QWEN2, Mistral 모델 등 다양한 모델에서 강력한 일반성과 확장성을 나타냈다.



### Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Unveiling AI's Potential Through Tools, Techniques, and Applications (https://arxiv.org/abs/2410.01268)
Comments:
          This book contains 156 pages and 9 figures

- **What's New**: 이번 논문은 빅데이터 분석에서의 딥 러닝(deep learning) 및 머신 러닝(machine learning)에 대한 소개를 제공하며, ChatGPT 및 Claude와 같은 도구, 하드웨어 추천, PyTorch 및 TensorFlow와 같은 라이브러리를 사용하는 개발 환경 설정에 대한 실용적인 가이드를 포함하고 있습니다. 또한 AutoML 및 엣지 컴퓨팅(edge computing)과 같은 AI의 미래에 대한 통찰을 제공합니다.

- **Technical Details**: 이 책은 머신 러닝의 역사와 개념을 시작으로 다양한 산업에서의 응용을 설명합니다. AI 도구들은 자연어 처리(natural language processing, NLP)의 발전으로 인해 직면하는 복잡한 문제들을 풀기 위한 강력한 도구로 자리잡고 있습니다. ChatGPT, Claude, Gemini와 같은 다중 모달 AI 도구들은 데이터 분석, 모델 설계, 코드 생성을 지원하여 연구의 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 딥 러닝과 머신 러닝은 스마트폰의 얼굴 인식, 자율 주행 자동차, 의료 영상 분석 및 금융 서비스 등 다양한 산업에서 활용되고 있습니다. 이 기술들은 전문가들만의 영역에서 벗어나 일반인들도 쉽게 AI 모델을 구축할 수 있도록 자동화 머신 러닝(AutoML)을 통해 민주화되고 있습니다. 앞으로 엣지 컴퓨팅을 통해 AI는 클라우드에서 로컬 장치로 이동하며, AI의 의사결정 과정에서의 투명성과 공정성을 확보하는 연구가 중요해질 것입니다.



### AHP-Powered LLM Reasoning for Multi-Criteria Evaluation of Open-Ended Responses (https://arxiv.org/abs/2410.01246)
Comments:
          Accepted for EMNLP 2024 Findings

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)과 분석적 계층화 과정(Analytic Hierarchy Process, AHP)을 활용하여 개방형 질문에 대한 답변을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 두 단계로 나누어집니다: 기준 생성 단계와 평가 단계입니다. 기준 생성 단계에서는 질문에 대한 여러 평가 기준을 LLM을 통해 생성하고, 평가 단계에서는 각 기준에 따라 후보 답변을 쌍대 비교하여 최종 결정을 내립니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 4개의 기준 모델보다 인간의 판단과 더 밀접하게 일치하며, 정량적 지표인 일치 지수 및 부드러운 일치 지수에서 더 뛰어난 성능을 보였습니다.



### Automatic deductive coding in discourse analysis: an application of large language models in learning analytics (https://arxiv.org/abs/2410.01240)
Comments:
          20 pages

- **What's New**: 이번 연구는 자동 추론 코딩(automatic deductive coding)의 필요성과 대형 언어 모델(large language models)의 활용 가능성을 제시합니다. 기존의 수동적 코딩 방식의 한계를 극복할 수 있는 새로운 방법을 모색합니다.

- **Technical Details**: 연구에서는 전통적인 텍스트 분류 방법, BERT와 유사한 사전 훈련된 언어 모델, 그리고 GPT와 같은 사전 훈련된 대형 언어 모델을 포함한 세 가지 분류 방법을 비교하였습니다. 특히, prompt engineering을 활용하여 GPT의 효율성을 극대화했습니다.

- **Performance Highlights**: 세 가지 분류 방법의 정확도 및 Kappa 값을 비교한 결과, prompt engineering이 적용된 GPT 모델이 두 개의 데이터셋에서 제한된 수의 훈련 샘플로도 다른 두 방법보다 우수한 성과를 냈습니다.



### From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging (https://arxiv.org/abs/2410.01215)
Comments:
          Code and data available at this https URL

- **What's New**: MGDebugger는 코드 디버깅을 위한 새로운 접근 방식으로, 다중 수준의 세분화된 오류를 해결하기 위해 계층적 구조를 사용합니다. 이는 기존 시스템들이 단일 단위로 처리했던 것과는 대조적입니다.

- **Technical Details**: MGDebugger는 코드를 하위 함수들의 계층적 트리 구조로 분해하고, 각 하위 함수에 대해 독립적인 디버깅을 진행합니다. 이 과정에서 LLM 기반의 Python 실행 시뮬레이터를 활용하여 변수 상태를 추적하고 오류를 정확하게 식별합니다.

- **Performance Highlights**: MGDebugger는 기존 디버깅 시스템들에 비해 HumanEval에서 18.9%의 정확도 향상을 이루었고, HumanEvalFix에서 97.6%의 수정 성공률을 달성했습니다. 이 시스템은 다양한 종류의 버그와 난이도 수준을 효과적으로 처리할 수 있는 강력함과 효율성을 보여줍니다.



### StringLLM: Understanding the String Processing Capability of Large Language Models (https://arxiv.org/abs/2410.01208)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 문자열 처리 능력을 종합적으로 탐구하고, 이를 위한 벤치마크 데이터셋인 StringBench를 개발하였습니다. 연구자들은 StringLLM을 통해 LLMs의 문자열 처리 능력을 평가하고, 성능 향상 방법을 제안하였습니다.

- **Technical Details**: StringLLM은 기본 문자열 처리 작업인 atomic tasks를 수집하고, 이를 결합하여 복잡한 composite tasks를 생성합니다. 연구는 LLMs의 문자열 처리 능력을 평가하기 위해 세 가지 프롬프트 전략(raw instructions, Chain of Thought (CoT), Program of Thought (PoT))을 사용하여 체계적인 실험을 수행합니다. 또한, fine-tuning을 통해 성능을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: LLMs는 문자열 처리 작업에서 평균 48.89%의 정확도로 인간보다 낮은 성능을 보였으며, random strings는 특히 어려운 경향을 보였습니다. PoT를 사용할 경우 일부 LLMs는 20% 이상의 성능 향상이 나타났습니다. fine-tuning을 통해 LLMs는 평균 38.80% 향상된 정확도로 문자열 처리 능력을 획기적으로 개선했습니다.



### Gold Panning in Vocabulary: An Adaptive Method for Vocabulary Expansion of Domain-Specific LLMs (https://arxiv.org/abs/2410.01188)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 도메인 특화 작업에서의 성능 향상을 위한 새로운 어휘 확장 방법인 VEGAD를 소개합니다. 기존 연구들의 한계를 극복하기 위해, 전체 어휘의 일부만을 확장하는 것이 오히려 더 나은 성능을 낼 수 있다는 발견을 통해 출발합니다.

- **Technical Details**: VEGAD는 도메인 별 어휘에서 중요한 단어들을 자동으로 식별하기 위한 방법으로, 기울기를 기반으로 한 어휘 확장을 수행합니다. 각 단어에 대한 gradient를 계산하고 이를 통해 중요한 단어 후보를 추출합니다. 이 과정에서 Trie 구조를 이용하여 후보 어휘에서 효율적으로 단어를 검색합니다.

- **Performance Highlights**: VEGAD는 법률 및 의료 분야의 세 가지 중국어 데이터셋에서 기존 어휘 생성 방법들보다 뛰어난 성능을 보였습니다. 도메인 특정 작업 및 일반 작업에서 모두 성능 향상을 보여, VEGAD의 도메인 어휘 확장 가능성을 강조합니다.



### FastLexRank: Efficient Lexical Ranking for Structuring Social Media Posts (https://arxiv.org/abs/2410.01183)
- **What's New**: 이 연구에서는 FastLexRank라는 새로운 효율적이고 확장 가능한 텍스트 순위 알고리즘을 소개합니다. 이는 기존의 LexRank 메소드의 계산 및 메모리 복잡도를 줄이고, 텍스트 중심성(text centrality) 계산을 위한 스케일러블한 솔루션을 제공합니다.

- **Technical Details**: FastLexRank는 LexRank의 기본 개념을 바탕으로 하여 문장 그래프의 정적 분포(stationary distribution)를 계산하는 최적화된 방법을 활용합니다. 이 알고리즘은 시간 복잡도를 \mathcal{O}(n^2)에서 \mathcal{O}(n)으로 개선하여 매우 큰 데이터셋을 실시간으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 실증적 결과에 따르면 FastLexRank는 중앙 트윗(central tweets)을 식별하는 데 효과적이며, 이를 통해 소셜 미디어의 대량의 텍스트를 신속하게 요약할 수 있습니다. 이 방법은 LLM(대형 언어 모델)의 콘텐츠 요약 효율성을 향상시킵니다.



### Towards Inference-time Category-wise Safety Steering for Large Language Models (https://arxiv.org/abs/2410.01174)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 안전성을 강화하기 위한 새로운 접근법을 제시합니다. 기존의 교육 및 미세 조정 없이, 추론 단계에서 바로 사용할 수 있는 안전 지향 방법론을 탐구합니다.

- **Technical Details**: 연구에서는 두 가지 주요 방법을 소개합니다: (i) 카테고리별(특정 피해 유형) 스티어링 벡터를 계산하여 세밀한 조정이 가능하게 하고, (ii) 정보가 풍부한 스티어링 벡터를 추출하여 안전성을 확보하면서 생성 텍스트의 품질을 유지합니다. 이들은 전통적인 안전성 문제를 해결하기 위해 모델의 중간 레이어에서 조정됩니다.

- **Performance Highlights**: 다양한 LLM 및 데이터 셋에서 제안된 스티어링 방법의 효과를 입증하며, 안전성을 높이는 동시에 생성되는 텍스트의 품질을 유지하는 데 성공했습니다. 연구는 이러한 접근법이 대형 언어 모델의 안전한 작동을 위해 어떻게 활용될 수 있는지를 논의합니다.



### BordIRlines: A Dataset for Evaluating Cross-lingual Retrieval-Augmented Generation (https://arxiv.org/abs/2410.01171)
Comments:
          NLP for Wikipedia workshop at EMNLP 2024

- **What's New**: 이 논문은 cross-lingual retrieval-augmented generation (xlRAG)의 중요성을 강조하며, 다양한 관점을 반영한 정보를 답변에 포함시키는 방법에 대한 연구를 수행합니다. 특히 251개의 지정학적 분쟁에 대한 질의와 이에 관련된 다국어 데이터셋인 BordIRlines를 제안합니다.

- **Technical Details**: BordIRlines 데이터셋은 49개 언어로 작성된 720개의 질의를 포함하며, Wikipedia에서 관련 구문을 수집하여 구성되었습니다. 연구진은 여러 다국어 정보 검색 모델(mDPR, COLBERT, BM25, BGE M3)을 실시하여, 추가적 맥락을 제공했을 때의 LLM 반응 변화를 분석하였습니다. 이 과정에서 데이터의 구성 변화가 LLM의 일관성에 미치는 영향을 조사하였습니다.

- **Performance Highlights**: 실험 결과, 기존 RAG 시스템들은 다른 언어 간 일관성이 부족하며, 문서 제공 방식의 변화가 모델의 응답에 중대한 영향을 미친다는 것이 확인되었습니다. 이에 대한 두 가지 사례 연구를 제시하고, 추후 연구 방향에 대해서도 논의합니다.



### Unifying the Scope of Bridging Anaphora Types in English: Bridging Annotations in ARRAU and GUM (https://arxiv.org/abs/2410.01170)
Comments:
          The Seventh Workshop on Computational Models of Reference, Anaphora and Coreference (CRAC 2024), EMNLP 2024 Workshop, 15 November 2024

- **What's New**: 이 논문은 서로 다른 coreference 자원 간의 bridging 주석 비교를 용이하게 하기 위해, GUM, GENTLE 및 ARRAU 데이터 세트의 지침을 분석하고 해석 가능한 예측 모델을 사용하여 bridging 사례를 검토합니다. 이 연구는 다양한 현상이 주석으로 달린 방식을 통해, bridging의 정의와 주석 스키마의 비표준화를 해결하고자 합니다.

- **Technical Details**: 연구에서는 GUM, GENTLE, ARRAU WSJ 세 개의 코퍼스를 비교하고, 이들 간의 주석 가이드라인 및 기술적 형식에서 발생한 범주적 차이를 먼저 확인한 후, 각 코퍼스에 대해 예측 모델을 훈련시키고 크로스 코퍼스 예측 결과에 대한 오류 분석을 수행합니다. 또한, 이를 통해 bridging의 발생 환경의 차이를 분석합니다.

- **Performance Highlights**: 붙잡아야 할 결과는 GUM/GENLTE 및 ARRAU WSJ에 대해 조정된 테스트 세트를 제공함으로써, ARRAU 스타일의 bridging 하위 주석을 GUM에 통합하고 개체 유형 주석 카테고리를 통합했습니다. 이 연구는 bridging 자원의 지속적인 교차 호환성에 대한 관심을 증진시킬 것으로 기대됩니다.



### GADFA: Generator-Assisted Decision-Focused Approach for Opinion Expressing Timing Identification (https://arxiv.org/abs/2410.01169)
- **What's New**: 본 연구는 뉴스에 의해 촉발된 의견 표현의 타이밍을 식별하는 새로운 과제를 제시합니다.

- **Technical Details**: 본 연구는 전문 주식 분석가의 활동을 기반으로 하며, 이를 위해 새로운 데이터셋을 개발하였습니다. 우리는 text generation 모델을 활용하여 분류 모델을 조율하고, 이를 통해 전반적인 성능을 향상시키는 결정을 중심으로 한 접근법을 취하고 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 모델이 생성한 텍스트는 다양한 관점에서 새로운 통찰력을 제공하며, 효과적으로 의견 표현의 최적 타이밍을 식별하는 데 기여함을 보여주었습니다.



### Document Type Classification using File Names (https://arxiv.org/abs/2410.01166)
- **What's New**: 해당 논문에서는 경량화된 슈퍼바이즈드 러닝 모델을 활용하여 문서 분류 문제를 해결하는 새로운 방법을 제안하였다. 기존의 복잡한 딥러닝 모델 대신 TF-IDF 기반 특성 추출을 결합한 방식을 사용하여 파일 이름만으로도 문서를 효율적으로 분류할 수 있다.

- **Technical Details**: 이 논문에서는 문서의 파일 이름을 기반으로 이를 분류하기 위해 Random Forest 분류기와 Trie 토크나이저를 사용하였다. 이 접근법은 파일 이름에 대한 예측을 수행할 때 평균 1.23 × 10^{-4} 초의 속도로, DiT 모델보다 442.43배 빠르게 작동하였다. 또한, 파일 이름의 신뢰 점수를 통해 애매한 파일 이름과 의미 있는 파일 이름을 구별할 수 있다.

- **Performance Highlights**: 제안된 파일 이름 분류기는 평균 96.7%의 정확도로 테스트된 데이터셋의 80% 이상을 처리할 수 있으며, 동일한 데이터셋 내에서 표시된 예측 정확도는 99.6%에 이른다. 이를 통해 대규모 데이터셋에서 신속하고 신뢰할 수 있는 문서 분류를 가능하게 한다.



### Evaluating Deduplication Techniques for Economic Research Paper Titles with a Focus on Semantic Similarity using NLP and LLMs (https://arxiv.org/abs/2410.01141)
Comments:
          6 pages, 1 figure

- **What's New**: 이 연구는 경제 연구 논문의 제목으로 구성된 대규모 NLP 데이터셋의 효과적인 중복 제거(duplication) 기법을 조사하였습니다.

- **Technical Details**: 다양한 페어링 방법을 Levenshtein distance, cosine similarity 와 sBERT 모델을 포함하여 탐구하고, 제목만을 기반으로 중복을 탐지하기 위한 여러 기법을 구현하였습니다. 특히, 문자열 기반(method), 해시(mapping), 임베딩 기반(embedding) 방법을 사용하여 텍스트 데이터를 수치벡터로 변환하고 유사성을 비교하였습니다.

- **Performance Highlights**: 연구 결과, Levenshtein distance, cosine similarity 및 SBERT 모델을 활용한 2,000 쌍의 제목 비교를 통해 중복 제목의 낮은 존재 가능성을 확인하였습니다. 데이터 시각화는 세 가지 거리 측정 방법의 상관관계를 보여주었으며, 모든 방법에서 완벽히 동일한 결과를 얻지 못하였음을 나타냈습니다.



### Approximately Aligned Decoding (https://arxiv.org/abs/2410.01103)
Comments:
          9 pages main, 22 pages total

- **What's New**: 본 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 바람직하지 않은 출력을 감소시키는 새로운 방법을 제안합니다. 기존 방법들에 비해 계산 효율성을 높이면서도 출력 분포의 왜곡을 줄여 긴 텍스트 생성에서 어려운 제약을 충족할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 기존의 오류 완화 방법들과 비교하여 낮은 확률 출력의 증폭을 줄이면서 출력 분포를 유지합니다. 이를 통해 강화된 제약 조건이 있을 때 더 빠르게 수렴하는 성능을 보여줍니다. 또한 적응 샘플링 기법(Adaptive Sampling with Approximate Expected Futures, ASAp)을 사용하여 효과적으로 샘플을 추출합니다.

- **Performance Highlights**: 다수의 실험에서 제안된 방법이 ASAp와 유사한 과제 특정(performance specific) 성능을 나타내면서도, 어렵게 만족해야 하는 제약 조건들이 있을 때 기존 방법보다 훨씬 빠르게 수렴함을 보였습니다.



### Unlocking Korean Verbs: A User-Friendly Exploration into the Verb Lexicon (https://arxiv.org/abs/2410.01100)
Comments:
          COLING2025 System Demonstrations (Submitted)

- **What's New**: 이 논문에서는 세종 사전 데이터셋을 기반으로 한 새로운 웹 인터페이스와 파이썬 라이브러리를 소개합니다. 이 도구들은 한국어 동사 정보 수집 및 처리에 도움을 주기 위해 설계되었습니다.

- **Technical Details**: 세종 사전은 동사와 그에 대한 형태적, 통사적(syntactic), 의미적(semantic) 정보를 포함한 XML 파일을 제공합니다. 이 논문은 특히 하위 분류 프레임(subcategorization frames)을 중심으로 한 동사 lexicon을 탐구합니다. 웹 인터페이스는 비전공자들이 쉽게 접근하고 사용할 수 있도록 설계되었으며, 예문과 함께 각 동사에 대한 정보를 제공합니다. 또한, 동사에 대한 감정 역할 레이블링(semtantic role labeling)과 통사적 구문 분석(syntactic parsing)을 돕는 파이썬 라이브러리를 제공합니다.

- **Performance Highlights**: 개발된 웹 인터페이스는 세종 사전의 동사 정보를 손쉽게 탐색할 수 있게 하며, 사용자는 동사의 하위 분류 프레임에 접근할 수 있습니다. 사용자들은 동사의 의미와 사용 예를 체계적으로 이해할 수 있는 시각적 도구를 제공합니다. 이러한 접근은 한국어 처리 애플리케이션 개발에 기여할 것으로 예상됩니다.



### Concept Space Alignment in Multilingual LLMs (https://arxiv.org/abs/2410.01079)
Comments:
          EMNLP 2024

- **What's New**: 본 연구에서는 다국어 대형 언어 모델(multilingual large language models, LLMs)의 개념 정렬(concept alignment) 성능을 평가하고 분석하였습니다. 특히, 서로 다른 언어 간의 개념 간 직선 매핑(linear mapping)의 유무에 대해 검토하였습니다.

- **Technical Details**: 연구진은 10개의 LLM과 여섯 개의 언어를 대상으로 실험을 수행하였으며, Procrustes Analysis를 통해 서로 다른 언어의 개념 공간 간의 적절한 선형 변환을 발견했습니다. 또한, 두 가지 개념 추출 방법(vanilla 및 prompt-based)을 비교하여 각기 다른 매핑의 선형성과 정밀도(precision)를 평가했습니다.

- **Performance Highlights**: 결과에 따르면, 다국어 LLM에서 개념 간 선형 정렬이 유도될 수 있으며, prompt-based 개념 임베딩이 일반적으로 낮은 선형성을 보여줍니다. 추상적인 개념은 물리적 개념보다 더 나은 정렬 성과를 보였으며, 유사한 유형적 거리에서 일반화의 문제는 여전히 존재합니다.



### From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems (https://arxiv.org/abs/2410.01066)
Comments:
          12 pages, 5 figures, 3 tables

- **What's New**: 이 논문은 LLM 기반 텍스트-투-SQL 시스템의 발전을 다룬 포괄적인 연구이며, 초기 규칙 기반 모델에서부터 최신 LLM 접근법까지의 역사를 설명합니다. 특히, 지식 그래프와의 통합이 문맥 정확성 및 스키마 연결에 미치는 영향을 연구합니다.

- **Technical Details**: 현재 기술은 두 가지 범주로 나눌 수 있습니다: corpus의 인-컨텍스트 학습과 파인 튜닝입니다. 이로 인해 제로샷(Zero-shot), Few-shot 학습 및 데이터 증강(Data augmentation)과 같은 접근 방식이 도출됩니다. 또한, 이를 통한 성능 평가를 위한 벤치마크 및 평가 메트릭을 논의합니다.

- **Performance Highlights**: 텍스트-투-SQL 시스템은 비전문가가 자연어로 데이터베이스를 쿼리할 수 있게 해주며, 이는 헬스케어, 물류 및 금융 시스템 등의 데이터 기반 의사결정 향상에 기여합니다. 논문은 모델의 신뢰성, 계산 효율성, 데이터 프라이버시와 같은 주요 과제를 강조하며 LLM 기반 텍스트-투-SQL 시스템의 향후 발전 가능성을 제시합니다.



### From Facts to Insights: A Study on the Generation and Evaluation of Analytical Reports for Deciphering Earnings Calls (https://arxiv.org/abs/2410.01039)
Comments:
          Pre-print

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)을 활용하여 Earnings Calls(ECs)에서 파생된 분석 보고서를 생성하고 평가하는 방법을 탐구합니다. 연구의 공백을 해소하며, 다양한 관점과 분석 주제를 논문 생성 과정에 도입하는 특화된 에이전트를 설계한 다중 에이전트 프레임워크에서의 보고서 생성 방식을 발표합니다.

- **Technical Details**: 분석 보고서를 생성하기 위한 협력적 다중 에이전트 대화 프레임워크를 탐험하며, Microsoft의 AutoGen을 사용하여 초기화 프롬프트를 통해 각 LLM 기반 에이전트에 독특한 역할을 부여합니다. 분석가, 심리학자, 편집자 등의 역할을 포함하는 피드백 에이전트를 두고 글쓰기 에이전트와 대화하며 반복적인 피드백 과정을 통해 보고서를 개선합니다.

- **Performance Highlights**: 결과적으로, 추가된 에이전트들이 더 통찰력 있는 보고서를 생성할 수 있도록 도와주지만, 인간 전문가들이 작성한 보고서 선호도가 여전히 높습니다. 또한, LLM을 사용한 보고서 평가 방법의 유효성을 조사하여 인간 전문가와의 품질 상관관계를 밝혀냅니다.



### MOSEL: 950,000 Hours of Speech Data for Open-Source Speech Foundation Model Training on EU Languages (https://arxiv.org/abs/2410.01036)
Comments:
          Accepted at EMNLP 2024 Main Conference

- **What's New**: 새로운 연구에서는 유럽 연합(EU)의 24개 공식 언어에 대한 오픈 소스 기반의 음성 모델(Speech Foundation Models, SFMs) 구축을 위한 첫 단계로, 950,000시간의 음성 인식 데이터 및 레이블이 없는 음성 데이터 수집을 목표로 하고 있습니다.

- **Technical Details**: 연구팀은 오픈 소스 라이센스에 부합하는 자동 음성 인식(ASR) 데이터 세트와 비표시 음성 코퍼스를 조사하여 EU 언어에 사용할 수 있는 데이터를 수집했습니다. 이 데이터는 'MOSEL' 프로젝트로 이름 붙여져 GitHub에서 공개적으로 이용 가능하며, 추가로 441,000시간의 비표시 데이터에 대한 자동 전사본이 CC-BY 라이센스 하에 생성되었습니다.

- **Performance Highlights**: 이 연구는 자원 부족 언어인 말타어를 대상으로 한 실험을 통해 수집된 데이터가 실제 ASR 모델 훈련에 효과적으로 사용될 수 있음을 보여주었습니다.



### Draft on the Fly: Adaptive Self-Speculative Decoding using Cosine Similarity (https://arxiv.org/abs/2410.01028)
- **What's New**: 본 논문에서는 대형 언어 모델의 빠른 추론을 위한 간단한 방법을 제안합니다. 기존의 (self-)speculative decoding 기법과는 달리, 이 방법은 고정된 초안 모델을 생성하기 위해 미세 조정(fine-tuning)이나 블랙 박스 최적화(black-box optimization)를 필요로 하지 않으며, 입력 컨텍스트에 맞게 적응된 다양한 초안 모델(draft models)을 생성하는 간단한 규칙을 사용합니다.

- **Technical Details**: 제안된 Adaptive Self-Speculative Decoding (ASD) 방법은 Layer를 제거하는 기준으로 코사인 유사도(cosine similarity)를 사용합니다. 이 방법은 기존 모델에서 특정 Attention Layer의 영향을 평가하여, 중요도가 낮은 Layer를 제거함으로써 추론 속도를 높입니다. 이를 통해 초안 모델 D는 모델 M의 하위 네트워크로 즉석에서 생성되어 입력 컨텍스트에 적응합니다.

- **Performance Highlights**: 경량화된 알고리즘이 현재의 SOTA와 경쟁력을 갖추고 있으며, 단순히 'plug-and-play' 방식으로 구현할 수 있습니다. 또한, 성능 저하 없이 추론 속도를 증가시키는 능력을 갖추었습니다.



### Investigating the Synergistic Effects of Dropout and Residual Connections on Language Model Training (https://arxiv.org/abs/2410.01019)
Comments:
          5 pages, 4 figures

- **What's New**: 본 논문은 언어 모델 훈련에서 과적합(overfitting)을 완화하기 위한 dropout 기법의 중대한 역할을 조사합니다. 다양한 dropout 비율이 개별 레이어 및 잔여 연결에 미치는 영향을 분석하며, Tiny Shakespeare 데이터셋을 사용하여 속도와 검증 오류에 미치는 결과를 연구했습니다.

- **Technical Details**: 변화하는 dropout 비율과 잔여 연결을 Transformer 구조에서 언어 모델링에 적용하며, 측정되는 주요 요소에는 훈련 수렴(convergence), 검증 오류(validation error), 일반화 가능성(generality)이 포함됩니다. 각 레이어에 대한 dropout 기법을 포함하여 attention 및 MLP 레이어의 조합, 잔여 연결의 스킵된 레이어의 수를 고려합니다.

- **Performance Highlights**: 실험 결과는 dropout 기법이 정규화(regularization) 및 잔여 연결이 수렴(covrgence)에 유익한 영향을 미친다는 점을 확인했으며, 최적의 딥 뉴럴 네트워크 수렴과 일반화를 위한 잔여 연결 깊이와 dropout 비율 간의 중요한 트레이드오프(trade-off)를 발견했습니다.



### "Hiding in Plain Sight": Designing Synthetic Dialog Generation for Uncovering Socially Situated Norms (https://arxiv.org/abs/2410.00998)
Comments:
          Pre-Print

- **What's New**: 본 논문은 다양한 대화 맥락 및 상황을 기반으로 한 대화 생성 프레임워크를 제안합니다. 이를 통해 대화 중 사회적 규범이 어떻게 위반될 수 있는지를 분석하고, 이를 피하기 위한 방법을 제시하는 'NormHint'라는 데이터셋을 생성하였습니다.

- **Technical Details**: 제안한 프레임워크는 연령대, 직업 및 성격 유형 등 다양한 대화 참여자의 속성을 포함하여 대화를 생성하는 멀티-스텝 과정입니다. NormHint 데이터셋은 189,718개의 대화와 25,543개의 발화로 구성되어 있으며, 대화의 갈등 및 에스컬레이션을 포함한 다양한 맥락 및 특성을 가진 캐릭터 쌍을 특징으로 합니다. 각 대화는 서사적 맥락과 함께 사회적 규범 위반 사례를 제공합니다.

- **Performance Highlights**: 자동 분석 결과 NormHint는 수집된 대화 데이터에 비해 다양성이 10% 더 높으며, 다른 합성 대화 데이터셋보다 27% 더 우수하다는 결과를 보여주었습니다. 96%의 경우 인간 평가에 의해 현실적이라고 평가되었으며, 대화의 자연스러움은 기존 데이터셋과 비교해 우수한 수준으로 확인되었습니다.



### Creative and Context-Aware Translation of East Asian Idioms with GPT-4 (https://arxiv.org/abs/2410.00988)
- **What's New**: 본 논문은 동아시아 관용구의 맥락 인식 번역을 위한 GPT-4의 효과를 평가합니다. 기존 번역 시스템들의 한계를 극복하기 위해 다양한 프롬프트 전략을 활용하여 더 높은 품질의 번역을 생성하는 방법론을 제안합니다.

- **Technical Details**: 이 연구는 4개의 동아시아 언어(중국어, 일본어, 한국어)의 관용구를 포함하며, 각 언어로부터 500개의 문장을 생성하여 27개의 번역을 수집하였습니다. GPT-4는 Faithfulness(충실도)와 Creativity(창의성)를 기준으로 자동 평가를 진행하여 Pareto-optimal한 번역 프롬프트 전략을 선정하였습니다.

- **Performance Highlights**: GPT-4는 Google 및 DeepL과 같은 상업적 번역 엔진보다 높은 Faithfulness와 Creativity 점수를 기록하며, 전체적으로 훨씬 더 높은 품질의 번역을 생성하는 것으로 나타났습니다. 이 연구는 코드를 오픈소스하여 추가 연구에 기여할 것입니다.



### Automatic Speech Recognition for the Ika Languag (https://arxiv.org/abs/2410.00940)
Comments:
          10 pages, 5 Figures This is a pre-release version

- **What's New**: 이 논문에서는 자원이 부족한 언어인 Ika를 위한 Automatic Speech Recognition (ASR) 모델을 개발하는 비용 효율적인 접근법을 제시합니다. 이 기술은 사전 훈련된 wav2vec 2.0 다국어 음성 모델을 Ika의 고품질 음성 데이터 세트에 대해 파인 튜닝(fine-tune)하는 과정으로 구성됩니다.

- **Technical Details**: 연구팀은 New Testament (신약 성경) 번역에서 수집된 음성 데이터 세트를 사용하여 모델을 훈련시켰습니다. 그 결과, 다국어로 사전 훈련된 모델을 파인 튜닝함으로써 0.5377의 Word Error Rate (WER)와 0.2651의 Character Error Rate (CER)를 달성했습니다. 대규모 10억 개의 파라미터를 가진 모델이 3억 개의 파라미터를 가진 작은 모델보다 더 우수한 성능을 보였으나, 작은 훈련 데이터 세트에 대한 과적합(overfitting)을 관찰하여 일반화 가능성이 감소했습니다.

- **Performance Highlights**: 이 연구는 자원 부족 언어를 위한 다국어 사전 훈련 모델을 활용할 수 있는 잠재력을 보여줍니다. 향후 연구에서는 데이터 세트를 확대하고 과적합을 완화할 수 있는 기법을 탐색하는 데 집중해야 합니다.



### Text Clustering as Classification with LLMs (https://arxiv.org/abs/2410.00927)
Comments:
          12 pages, 3 figures

- **What's New**: 이 연구는 대형 언어 모델(LLM)의 인컨텍스트 학습(in-context learning) 능력을 활용하는 새로운 텍스트 클러스터링 프레임워크를 제안합니다. 기존의 세밀한 조정이 필요 없는 클러스터링 기법을 통해 데이터셋의 레이블 생성과 분류를 LLM으로 수행하도록 합니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 미니 배치(mini-batch)로 데이터를 입력하고 LLM에게 레이블 생성 작업을 요청합니다. 두 번째 단계에서는 생성된 유사 레이블을 통합한 후, 각 샘플에 가장 적합한 레이블을 할당하도록 LLM에 요청합니다. 이 방법은 데이터셋을 순차적으로 처리하여 LLM의 입력 길이 제한을 피합니다.

- **Performance Highlights**: 연구 결과, 제안한 프레임워크는 5개의 다양한 데이터셋에 대해 상태-최고(clustering methods)와 비교하여 유사하거나 더 나은 성능을 보였습니다. 특히, 복잡한 하이퍼파라미터 조정이나 데이터셋에 따른 세밀한 조정이 필요 없으므로 시간과 계산 자원을 significantly 절약할 수 있습니다.



### Knowledge-Driven Feature Selection and Engineering for Genotype Data with Large Language Models (https://arxiv.org/abs/2410.01795)
- **What's New**: 이 논문은 작은 해석 가능한 변이 특성(set of variant features)을 바탕으로 복잡한 유전적 기초를 가진 표현형(phenotype)을 예측하는 기존의 문제점들을 해결하기 위한 새로운 지식 기반 프레임워크인 FREEFORM을 개발했습니다. 이 프레임워크는 고차원 유전자형(genotype) 데이터에서 LLMs(대형 언어 모델)의 지식을 활용하여 특성을 선택 및 엔지니어링하는 방식을 제공합니다.

- **Technical Details**: FREEFORM은 체인 오브 씽킹(chain-of-thought) 및 앙상블(ensembling) 원리를 사용하여 LLMs의 내재적 지식을 바탕으로 유전형 데이터의 특성을 선택하고 엔지니어링합니다. 이 연구에서 두 가지 유전자형-표현형 데이터셋, 즉 유전적 조상(genetic ancestry)과 유전성 청각 손실(hereditary hearing loss)에서 평가를 수행하였으며, 이는 데이터 기반 방법들보다 월등히 높은 성능을 보였습니다.

- **Performance Highlights**: FREEFORM은 특히 데이터 샘플이 부족한(low-shot) 환경에서 기존 데이터 기반 방법들보다 더 많은 성과를 보여 주목받았습니다. 또한, 이 프레임워크는 예측 성능을 향상시키면서도 해석 가능성을 유지하고, 데이터 차원을 줄이며, 지식 기반 접근 방식을 통해 기존의 문제들을 해결하는 잠재력을 demonstrated 합니다.



### DreamGarden: A Designer Assistant for Growing Games from a Single Promp (https://arxiv.org/abs/2410.01791)
Comments:
          21 pages + appendix, 11 figures

- **What's New**: 본 논문에서는 게임 디자인에서 사용될 수 있는 DreamGarden이라는 AI 시스템을 제안합니다. 이 시스템은 사용자가 제공하는 초기 프롬프트를 기반으로 고차원적인 계획을 수립하고, 이를 분류하여 구체적인 실행 계획을 제시하는 LLM 기반의 플래너를 사용합니다.

- **Technical Details**: DreamGarden은 Unreal Engine을 활용하여 다양한 게임 환경 개발을 지원하는 반자율적인 AI 도구입니다. 사용자의 꿈이나 상상을 프롬프트로 제공하면, 이 시스템은 이를 계층적인 작업 계획으로 분해하여 전문적인 하위 모듈로 분배합니다. 이를 통해 사용자는 계획의 성장 및 사용자 개입을 통한 피드백을 통해 변화하는 '계획의 정원'을 경험하게 됩니다.

- **Performance Highlights**: 사용자 연구를 통해 DreamGarden의 사용자가 자연어 프롬프트를 3D 시뮬레이션 환경으로 변환할 수 있는지, 그리고 이 시스템의 계층적이고 반복적인 과정이 사용자가 직관적으로 접근할 수 있는지를 평가했습니다. 결과적으로 사용자 인터페이스는 사용자가 다양한 수준에서 개입할 수 있는 충분한 기능을 제공하는 것으로 나타났습니다.



### OmniGenBench: Automating Large-scale in-silico Benchmarking for Genomic Foundation Models (https://arxiv.org/abs/2410.01784)
Comments:
this https URL

- **What's New**: 최근 유전자 모델링(Genomic Modeling) 분야에서 인공지능 기술의 발전, 특히 대형 언어 모델(Large Language Models)로 인해 유전자 기반 모델(Genomic Foundation Models, GFMs)에 대한 기대가 커지고 있습니다. GFMBench 프레임워크는 GFM 벤치마크(benchmark)를 표준화하고 자동화하여 깊이 있는 분석을 가능케 합니다.

- **Technical Details**: GFMBench는 오픈소스로 제공되는 소프트웨어로, 수백 개의 유전자 작업을 포함하는 수백만 개의 유전자 서열을 통합하여 다양한 GFM을 위해 벤치마킹을 표준화합니다. 이는 AutoBench 및 RNA 디자인과 같은 복잡한 과제를 포괄하는 사용자 친화적인 인터페이스와 다양한 튜토리얼을 제공합니다.

- **Performance Highlights**: GFMBench는 유전자 모델링의 표준화에 한 걸음 더 나아가며, 벤치마킹 결과를 공개 리더보드 형태로 제공하여 모델 성능을 대중에게 보여줍니다. 이는 GFM의 민주화(democratization)와 다양한 생명과학 및 치료 설계 분야에서의 활용 가능성을 높이는데 기여할 것입니다.



### Composing Global Optimizers to Reasoning Tasks via Algebraic Objects in Neural Nets (https://arxiv.org/abs/2410.01779)
- **What's New**: 이 논문에서는 2층 신경망의 솔루션 공간 내의 풍부한 대수 구조를 입증하고, 이는 쿼드러틱(Quadratic) 활성화 함수와 $L_2$ 손실을 바탕으로 하는 이유 추론(task) 문제에서의 학습에 적용됩니다. 이러한 구조는 손실의 일부만 만족하는 부분 솔루션으로부터 전역 최적 솔루션(global optimal solution)을 분석적으로 구성할 수 있게 해줍니다. 이 프레임워크는 CoGO(Composing Global Optimizers)라고 명명되었습니다.

- **Technical Details**: 연구에서는 2층 신경망의 가중치 공간이 반환환(semi-ring) 대수 구조를 갖고 있으며, 손실 함수가 단항 가능성(monomial potentials)으로 구성되어 있다는 점을 명시했습니다. 이러한 단항 가능성은 환 동형사상(ring homomorphism)으로, 부분 솔루션을 환 덧셈과 곱셈을 통해 전역 솔루션으로 구성할 수 있도록 합니다. 또한, 경험적으로 얻은 솔루션의 약 95%가 이론적으로 예측된 구조와 정확히 일치함을 보여주었습니다.

- **Performance Highlights**: 이론적 분석에 따르면, 높은 차수의 전역 최적화기는 훈련 동역학(training dynamics)에 불리하며, 과도한 매개변수화(over-parameterization)가 훈련 동역학을 비독립적으로 만들어 성능을 개선하는 경향이 있어 고차 전역 최적화기를 선호하지 않음을 보였습니다. 이 연구는 모델 학습 내에서 대수적 구조를 발견하고, 이를 통해 모듈라 덧셈과 같은 추론 문제에 대한 솔루션을 분석할 수 있는 첫 사례로 평가됩니다.



### LEOPARD : A Vision Language Model For Text-Rich Multi-Image Tasks (https://arxiv.org/abs/2410.01744)
Comments:
          Our code is available at this https URL

- **What's New**: 본 논문에서는 다중 텍스트-리치 이미지(텍스트가 주요 시각 요소인 이미지)에 대한 과제를 해결하기 위해 특별히 설계된 새로운 다중모달 대형 언어 모델(MLLM), Leopard를 소개합니다. 이 모델은 약 천만 개의 고품질 다중모달 교육 데이터를 기반으로 만들어졌으며, 텍스트-리치, 다중 이미지 시나리오에 맞춰 최적화된 설계로 구현되었습니다.

- **Technical Details**: Leopard는 텍스트-리치 다중 이미지 처리를 위한 적응형 고해상도 다중 이미지 인코딩 모듈을 통합하여, 입력 이미지의 원래 비율과 해상도에 따라 비주얼 시퀀스의 길이를 다이내믹하게 최적화합니다. 이 방법은 여러 고해상도 이미지를 수용하면서도 세부 사항이나 선명도를 손상시키지 않도록 합니다.

- **Performance Highlights**: Leopard 모델은 13개 시각-언어 벤치마크 데이터셋에서 평가되었으며, 5개의 텍스트-리치 다중 이미지 벤치마크에서 평균 +9.61 점의 우수한 성능을 보여 기존 오픈 소스 MLLM보다 뛰어난 결과를 기록했습니다. 또한, Leopard는 텍스트-리치 단일 이미지 과제와 일반 영역의 시각-언어 벤치마크에서도 경쟁력 있는 성과를 달성했습니다.



### ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation (https://arxiv.org/abs/2410.01731)
Comments:
          Project website: this https URL

- **What's New**: 이번 논문에서는 사용자 프롬프트에 자동으로 적합하게 조정되는 텍스트-이미지 생성 워크플로를 생성하는 새로운 작업을 소개합니다. 두 가지 LLM 기반 접근법을 제안하며, 효율적인 워크플로 설계를 위한 수많은 컴포넌트와 그 복잡한 상호 의존성을 고려합니다.

- **Technical Details**: 我们提出了两种方法：调优方法和无训练方法。调优方法从用户偏好数据中学习，而无训练方法利用LLM选择现有的流。为了训练LLM，我们从500500个多样化的用户提示中收集数据，并使用人类偏好估计器对生成的图像进行评分。

- **Performance Highlights**: ComfyGen-IC 및 ComfyGen-FT 접근법은 기본 모델과 일반적인 비프롬프트 워크플로우에 비해 이미지 품질을 향상시킵니다. 우리의 방법은 사람의 선호도 및 프롬프트 정렬 벤치마크에서 모든 기준 모델을 초월하며, 프롬프트 의존적 흐름 예측이 텍스트-이미지 생성 품질을 향상시키는 새로운 경로가 됨을 보여줍니다.



### Evaluating Robustness of Reward Models for Mathematical Reasoning (https://arxiv.org/abs/2410.01729)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 RewardBench의 한계점을 지적하고, 수학적 추론(Reasoning) 작업에서 리워드 모델(Reward Model)의 신뢰성 있는 평가를 위한 새로운 벤치마크인 RewardMATH를 소개합니다.

- **Technical Details**: RewardMATH는 리워드 모델의 견고성(Robustness)을 효과적으로 나타내기 위해 설계되었습니다. 기존 리워드 모델은 선택된 완료(Completion)와 거부된 완료(Rejected Completion) 간의 차이를 충분히 나타내지 못하는 단일 비교에 의존하고 있습니다.

- **Performance Highlights**: RewardMATH에서의 점수는 최적화된 정책(Optimized Policy)의 결과와 강한 상관관계를 보여주며, 기존 벤치마크는 거의 상관관계가 없음을 보여줍니다. 이는 평가의 신뢰성을 높이고, 리워드 모델의 견고성을 잘 나타내는 잠재력을 강조합니다.



### Automated Knowledge Concept Annotation and Question Representation Learning for Knowledge Tracing (https://arxiv.org/abs/2410.01727)
- **What's New**: 이번 연구에서는 지식 추적(knowledge tracing, KT) 방식의 두 가지 주요 한계를 해결하기 위해 자동화된 지식 개념 주석(annotation) 및 질문 표현 학습 프레임워크인 KCQRL을 제안합니다. 이 프레임워크는 기존 KT 모델의 효과를 향상시키는 데 기여합니다.

- **Technical Details**: KCQRL은 대형 언어 모델(large language models, LLMs)을 활용하여 질문 솔루션을 생성하고 각 솔루션 단계에서 지식 개념(KC)의 주석을 자동으로 부여합니다. 또한, 생성된 질문 내용과 솔루션 단계의 의미론적 표현을 학습하기 위해 대조 학습(contrastive learning) 접근법을 도입합니다. 이러한 표현은 기존 KT 모델에 통합될 수 있습니다.

- **Performance Highlights**: KCQRL 프레임워크는 15개의 최신 KT 알고리즘을 사용하여 두 개의 대규모 실제 수학 학습 데이터셋에서 일관된 성능 향상을 달성하였습니다. 이 연구는 KCQRL이 기존 KT 모델의 성능을 크게 향상시킬 수 있음을 입증합니다.



### Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspectiv (https://arxiv.org/abs/2410.01720)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 후속 훈련 단계에서의 합성 데이터 생성과 일반화 능력 간의 관계를 체계적으로 분석합니다. 특히, 합성 데이터의 효과를 이론적으로 모델링하고, 정보 이득(Information Gain) 과 일반화 이득(Generalization Gain) 간의 새로운 개념인 GGMI(Generalization Gain via Mutual Information)를 소개합니다.

- **Technical Details**: 저자들은 합성 데이터 생성 과정을 분포 관점에서 모델링하고, 후속 훈련이 진행 중인 LLM에 미치는 합성 데이터의 영향을 분석하기 위해 역 병목 구조(Reverse Bottleneck Framework)를 제시합니다. 이 접근법은 합성 데이터가 LLM의 일반화 능력에 미치는 효과를 정량화할 수 있는 상한(Upper Bounds)을 제공합니다.

- **Performance Highlights**: 많은 최신 LLM들이 합성 데이터를 활용함으로써 훈련 성과를 개선하고 있으며, 이 논문은 합성 데이터를 통해 LLM의 성능과 신뢰성을 높일 수 있는 방법에 대한 통찰을 제공합니다. 이 연구는 특히 제한된 실제 데이터의 상황에서 LLM의 일반화 능력을 어떻게 향상시킬 수 있는지를 탐구하며, 합성 데이터의 설계 및 최적화 프로세스를 이해하는 데 중요한 기여를 합니다.



### CreDes: Causal Reasoning Enhancement and Dual-End Searching for Solving Long-Range Reasoning Problems using LLMs (https://arxiv.org/abs/2410.01696)
- **What's New**: 본 논문에서는 복합 최적화 문제를 처리하는 데 있어서 대형 언어 모델 (LLMs)의 한계를 극복하기 위한 새로운 접근 방식을 제시합니다. Causal Relationship Enhancement (CRE) 메커니즘과 Dual-End Searching (DES) 기법의 결합인 CreDes를 통해 모델의 성능을 개선하였습니다.

- **Technical Details**: CRE는 원인-효과 개입 (cause-effect interventions)과 개인 치료 효과 (Individual Treatment Effect, ITE)를 결합하여 추론 과정과 상태 전이 간의 강력한 인과 관계를 보장합니다. DES는 원래 상태와 목표 상태에서 동시에 시작하여 causal probability tree에서 해결책을 찾습니다. 이러한 접근을 통해 단일 방향 검색 (single-direction search)의 한계를 극복합니다.

- **Performance Highlights**: CreDes는 장기 추론 작업 (long-range reasoning tasks)에서 기존의 State-Of-The-Art (SOTA) 솔루션에 비해 정확성과 시간 효율성 모두에서 유의미한 성능 향상을 보여줍니다.



### U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models (https://arxiv.org/abs/2410.01692)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 큰 언어 모델(LLMs)의 성능이 질문의 난이도에 따라 달라지며, 이것이 emergent abilities(출현 능력)의 예측에 중요한 역할을 한다는 새로운 개념을 제안합니다. 질문의 난이도에 따라 U자 형태의 스케일링과 역 U자 형태의 스케일링을 관찰하고, 이를 통해 모델 성능의 급격한 향상을 예측할 수 있는 Slice-and-Sandwich라는 파이프라인을 제안합니다.

- **Technical Details**: 논문은 LLM의 성능을 질문 난이도에 따라 다르게 분석하는 과정을 설명합니다. 우선 질문을 난이도 기준으로 그룹화하고, emergence threshold(출현 임계값) 전후의 데이터로부터 성능을 각각 예측합니다. 여기서 사용되는 성능 지표로는 Brier Score와 binary Brier Score를 언급하며, 이들은 모델이 선택한 정답의 확률에 의존하는 지표입니다.

- **Performance Highlights**: Slice-and-Sandwich 파이프라인은 난이도에 따라 그룹화된 질문의 성능 예측을 통해 성능 급증을 효과적으로 포착합니다. MMLU, 산술 문제, Persian-QA 데이터셋에서 실험한 결과, 모델 성능이 출현 임계값을 넘어서는 주요 특징을 잘 설명하는 것으로 나타났습니다.



### VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignmen (https://arxiv.org/abs/2410.01679)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 복잡한 추론 작업에서의 신용 할당 문제를 해결하기 위해 VinePPO를 제안합니다. 기존의 Proximal Policy Optimization (PPO) 접근 방식의 단점을 강조하고, 대안적으로 몬테카를로 기반의 보상 추정 방법을 활용하여 성능을 개선합니다.

- **Technical Details**: 기존 PPO의 가치 네트워크는 복잡한 추론 작업에서의 예상 누적 보상을 정확하게 예측하는 데 어려움을 겪습니다. 이는 높은 분산의 업데이트와 최적이 아닌 성능으로 이어질 수 있습니다. 반면, VinePPO는 독립적인 몬테 카를로 샘플을 사용하여 중간 상태의 편향 없는 가치 추정치를 계산하며, 대형 가치 네트워크의 필요성을 제거하고 메모리 요구 사항을 줄입니다.

- **Performance Highlights**: VinePPO는 MATH와 GSM8K 데이터셋에서 PPO 및 기타 RL-free 기준보다 일관되게 뛰어난 성능을 나타냈습니다. gradient 업데이트를 최대 9배 줄이고, wall-clock 시간을 최대 3.0배 단축시키며, 더 나은 KL 발산(trade-off)을 달성했습니다. 이 결과는 LLM의 RL 파인튜닝에서 정확한 신용 할당의 중요성을 강조합니다.



### A Thematic Framework for Analyzing Large-scale Self-reported Social Media Data on Opioid Use Disorder Treatment Using Buprenorphine Produc (https://arxiv.org/abs/2410.01633)
- **What's New**: 이번 연구에서는 Reddit과 같은 소셜 미디어 플랫폼에서의 오피오이드 사용 장애(Opioid Use Disorder, OUD) 치료에 대한 정보 요구를 특성화하기 위한 주제 기반 프레임워크(theme-based framework)를 제안합니다.

- **Technical Details**: 연구는 buprenorphine과 관련된 r/Suboxone 커뮤니티에서 15,253개의 게시물을 수집하였으며, 5개의 주요 주제를 정의하고 6,000개의 게시물을 이 주제에 따라 코딩하였습니다. 각 게시물은 1~3개의 주제로 레이블이 붙을 수 있었습니다.

- **Performance Highlights**: 6,000개의 게시물 중 40.3%는 단일 주제를 포함하고, 36%는 두 개의 주제를, 13.9%는 세 개의 주제를 포함했습니다. 주요 발견 중에는 회복 과정에서의 심리적 및 신체적 영향 보고가 두드러졌으며, buprenorphine 접근의 복잡성 및 약물 투약, 줄이기(tapering), 회복의 다양한 단계에서의 물질 사용에 대한 정보 공백이 눈에 띄었습니다. 또한, 자가 치료 전략과 동료 기반 조언이 중요한 통찰력을 제공하며 잠재적인 오해를 드러냈습니다.



### ENTP: Encoder-only Next Token Prediction (https://arxiv.org/abs/2410.01600)
- **What's New**: 이 연구에서는 주로 디코더 전용 Transformers에 의존하던 다음 토큰 예측 모델의 기존 개념을 도전합니다. 전통적으로, 인과적 주의(causal attention)가 미래 토큰을 마스킹하는 데 필수적이라는 믿음이 있었으나, 본 연구는 이러한 디자인 선택이 필수가 아니라 효율성에 관련된 것임을 논의합니다. 우리는 Encoder-only Next Token Prediction (ENTP) 방식을 도입하여 ENTP와 디코더 전용 Transformers 간의 표현력(expressive power)과 복잡성(complexity) 차이를 탐구합니다.

- **Technical Details**: ENTP는 디코더 전용 Transformers가 가질 수 없는 특정 함수를 표현할 수 있는 능력을 갖추고 있으며, 디코더와 인코더 간의 기본적인 시간 및 공간 복잡성을 비교합니다. 이 연구는 Triplet-Counting 과제를 도입하고 ENTP가 이 과제를 쉽게 수행할 수 있는 반면, 기존의 디코더 전용 Transformer는 이를 수행할 수 없다는 것을 이론적으로 및 실험적으로 보여줍니다. 또한, ENTP 방식이 다양한 현실적인 작업에서 우수한 성능을 발휘함을 입증합니다.

- **Performance Highlights**: 실험 결과, ENTP는 길이 일반화(length generalization)와 인맥학습(in-context learning)과 같은 다양한 실제 작업에서 뛰어난 성능을 보였습니다. 특히, ENTP는 비선형 함수와 2계층 신경망 같은 다양한 간단한 함수 수행에서 효과적으로 작동하며, 대규모 텍스트 데이터셋에서의 언어 모델링 과제에서도 디코더와 인코더 간의 성능을 비교하여 그 장점을 입증하였습니다.



### MedQA-CS: Benchmarking Large Language Models Clinical Skills Using an AI-SCE Framework (https://arxiv.org/abs/2410.01553)
- **What's New**: 본 연구에서는 의료 교육에서 영감을 받은 MedQA-CS라는 AI-구조적 임상 시험(AI-SCE) 프레임워크를 도입하여 기존 임상 능력 평가의 한계를 극복하고자 합니다. 이 프레임워크는 LLMs(Large Language Models)가 제공하는 임상 시나리오에 대한 평가를 통해 실제 의료 환경에서의 기능을 평가합니다.

- **Technical Details**: MedQA-CS는 두 가지 지시 사항에 따른 작업(LLM-as-medical-student 및 LLM-as-CS-examiner)을 포함하며, 이는 임상 능력을 평가하기 위해 MedStuLLM 및 MedExamLLM의 두 가지 구성요소로 구성됩니다. OSCE(Objective Structured Clinical Examinations) 지침에 따라 LLM의 임상 능력을 'shows how' 수준에서 평가하는 것이 핵심적인 특징입니다.

- **Performance Highlights**: 실험 결과, MedQA-CS는 기존의 다지선다형 질문 기반 벤치마크에 비해 LLM의 임상 기술을 평가하는 데 더 도전적임을 보여주었습니다. 연구는 또한 LLM의 임상 기술 수행 능력과 관련된 흥미로운 통찰을 제공하며, LLM-as-Judge 프레임워크의 잠재력을 강조합니다.



### Analyzing Byte-Pair Encoding on Monophonic and Polyphonic Symbolic Music: A Focus on Musical Phrase Segmentation (https://arxiv.org/abs/2410.01448)
Comments:
          Accepted to 3rd Workshop on NLP for Music and Audio (NLP4MusA, co-located with ISMIR 2024)

- **What's New**: 이 연구에서는 Byte-Pair Encoding (BPE) 알고리즘이 텍스트가 아닌 기호 음악에 어떻게 적용되는지 조사합니다. 특히, 다양한 악기에서 BPE의 행동을 분석하고 그것이 단성(모노포닉) 및 다성(폴리포닉) 음악의 음악적 구절 세분화(task) 작업에 미치는 영향을 평가합니다.

- **Technical Details**: BPE는 문자 또는 단어의 하위 요소로 구성된 토큰화 알고리즘으로, 음악에서도 활용되도록 개발되었습니다. 이 연구에서는 다양한 악기에서의 BPE 행위를 분석하고, BPE가 생성한 서브워드(supertokens)가 음악적 특성을 얼마나 잘 포착하는지를 정량적으로 비교합니다. 우리는 MidiTok 패키지를 사용해 토큰화 프로세스를 수행하며, HuggingFace 라이브러리를 이용해 Transformer 모델을 구현합니다.

- **Performance Highlights**: 연구 결과, BPE는 다성 음악에서 성능을 크게 향상시켰으나 단성 음표에서는 특정 BPE 병합 범위 내에서만 성능을 개선하는 것으로 나타났습니다. 또한, BPE가 생성한 슈퍼토큰의 사용 빈도와 길이 변화를 분석하여, 다양한 악기에서 음악과 텍스트 간의 뚜렷한 차이를 확인했습니다.



### Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models (https://arxiv.org/abs/2410.01434)
Comments:
          24 pages, 17 figures

- **What's New**: 이번 연구에서는 신경망, 특히 언어 모델의 모듈형 구조(modularity)를 조사하여 서로 기능적으로 유사한 서브네트워크(subnetwork)가 어떻게 연결되고 재사용될 수 있는지를 분석했습니다.

- **Technical Details**: 연구에서는 transformer 기반의 언어 모델에서 고도로 조합 가능한 하위 작업(subtask)을 위한 회로(circuit)를 식별하고 비교합니다. 특히, 확률론적 문맥 자유 문법(probabilistic context-free grammar)을 바탕으로 10개의 모듈형 문자열 편집 작업(string-edit operations)에 책임이 있는 회로를 분석했습니다.

- **Performance Highlights**: 비슷한 기능을 가진 회로는 눈에 띄는 노드 오버랩(node overlap)과 과제 간 충실도(cross-task faithfulness)를 보여주었으며, 식별된 회로는 서브네트워크 집합 연산(subnetwork set operations)을 통해 재사용되고 결합되어 모델의 더 복잡한 기능 능력을 제시할 수 있음을 입증했습니다.



### The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs (https://arxiv.org/abs/2410.01417)
- **What's New**: 새로운 다중 모달 대형 언어 모델(MLLMs) 벤치마크 제안. 인간의 기본적인 연관 능력을 평가하기 위한 평가 기준 및 주목받지 못한 연관 임무 개발.

- **Technical Details**: 연관 작업을 정의하고 자연어 데이터셋을 활용한 annotation-free (주석 없는) 방법으로 연관 벤치마크 구축. 세 가지 수준의 연관 작업(단일 단계, 동기식, 비동기식) 설정.

- **Performance Highlights**: 현재 공개 소스 MLLMs는 연관 작업에서 인간과 비교해 일관되게 낮은 성능을 보임. 최고 성능의 닫힌 소스 모델도 인간 성능과 큰 차이 존재.



### PairDistill: Pairwise Relevance Distillation for Dense Retrieva (https://arxiv.org/abs/2410.01383)
Comments:
          Accepted to EMNLP 2024 Main Conference

- **What's New**: 이 논문에서는 Pairwise Relevance Distillation (PairDistill)이라는 새로운 방법을 제안하여, 문서 간의 상대적 유사성을 평가하는 pairwise reranking 기법을 활용하여 dense retrieval 모델의 훈련 성능을 향상시키고 있습니다. 기존의 pointwise rerankers와 달리, pairwise 접근법은 유사한 문서 간의 세밀한 구별을 가능하게 합니다.

- **Technical Details**: PairDistill은 pairwise reranker에서 제공하는 미세한 훈련 신호를 활용하여 retrieve 모델의 훈련을 개선합니다. 이를 통해 dense retrieval 모델은 밀접하게 순위가 매겨진 패시지들 간의 미세한 차이를 배울 수 있습니다. 이 방법은 ColBERT 및 DPR 아키텍처에 효과적으로 적용되었습니다.

- **Performance Highlights**: PairDistill은 다양한 벤치마크에서 기존 방법들을 능가하며, 새로운 최첨단 결과를 달성했습니다. 또한, 기존의 유사한 크기의 dense retrieval 모델들보다 성능이 현저히 향상되었음을 보여주었습니다.



### HelpSteer2-Preference: Complementing Ratings with Preferences (https://arxiv.org/abs/2410.01257)
Comments:
          26 pages, 3 figures

- **What's New**: 이 논문에서는 Bradley-Terry 스타일과 Regression 스타일의 리워드 모델을 비교하기 위한 고품질 데이터셋 'HelpSteer2'를 제공하고 있으며, 두 접근 방식의 효과를 면밀히 분석한 최초의 연구입니다.

- **Technical Details**: 리워드 모델은 언어 모델이 지침을 따르도록 하는 데 필수적이며, 두 가지 주요 접근 방식인 Bradley-Terry 스타일과 Regression 스타일로 훈련됩니다. 연구진은 두 스타일의 데이터를 적절히 맞추어 비교 검증을 수행하였으며, 인간이 작성한 정당화(Justification)가 포함된 Preference annotations를 사용합니다. 새로운 접근법으로 두 모델의 조합 방법을 제안하고, Llama-3.1-70B-Instruct 모델을 통해 베스트 성능을 기록하였습니다.

- **Performance Highlights**: 이 연구에서 훈련된 리워드 모델은 RewardBench에서 94.1점을 기록하였으며, 이는 140개 이상의 리워드 모델 중 최고의 역량을 지닌 모델입니다. 또한 RLHF(Reinforcement Learning from Human Feedback)에서 지침을 따르도록 모델을 정렬하는 데 효과적임을 입증하였습니다.



### RGD: Multi-LLM Based Agent Debugger via Refinement and Generation Guidanc (https://arxiv.org/abs/2410.01242)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 기반으로 하는 코드 생성 및 자동 디버깅을 위한 새로운 아키텍처인 RGD(Refinement and Guidance Debugging)를 제안합니다. RGD는 여러 LLM 에이전트를 활용하여 코드 생성 과정을 멀티 스텝으로 분해하고, 반복적인 피드백과 자기 성찰을 통해 코드 개선을 가능하게 합니다.

- **Technical Details**: RGD 프레임워크는 세 가지 유형의 LLM 에이전트를 포함합니다: 가이드 에이전트(Guide Agent), 디버그 에이전트(Debug Agent), 피드백 에이전트(Feedback Agent). 각 에이전트는 특정 역할을 수행하며, 과정 중 코드의 성공 및 실패 사례를 분석하여 지속적으로 수정할 수 있는 능력을 갖추고 있습니다. 이 체계는 특정 작업 설명에서 생성된 가이드를 바탕으로 초기 코드를 생성하고, 이후 피드백을 통해 코드 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 RGD는 HumanEval 데이터셋에서 9.8% 및 MBPP 데이터셋에서 16.2% 성능 향상을 달성하였으며, 기존의 최신 접근 방식과 전통적인 직접 프롬프트 방식에 비해 월등한 코드 생성 능력을 보여주었습니다.



### UAL-Bench: The First Comprehensive Unusual Activity Localization Benchmark (https://arxiv.org/abs/2410.01180)
- **What's New**: 이 연구는 비정상적인 활동의 로컬라이제이션에 대한 새로운 벤치마크인 UAL-Bench를 소개합니다. UAL-Bench는 비정상적인 활동 로컬라이제이션을 위한 세 가지 비디오 데이터셋과 하나의 지침 조정 데이터셋을 포함하고 있습니다.

- **Technical Details**: UAL-Bench는 비디오-언어 모델(Vid-LLMs)과 비전-언어 모델(VLM) 및 대형 언어 모델(LLM)을 통합한 새로운 접근법인 VLM-LLM을 평가합니다. 비정상적인 활동으로 간주되는 여러 사례에 대한 ROI 및 TD 기준으로 평가하며, 이를 통해 모델 성능을 개선합니다.

- **Performance Highlights**: VLM-LLM 접근 방식은 짧은 일관성을 보여주고, 비정상적인 사건 예상(start time)에서 Vid-LLMs보다 더욱 정확한 예측을 수행합니다. 이 연구는 장기 비디오, 특히 자폐 진단과 관련된 비디오에서의 어려움과 더불어 새로운 평가 지표인 R@1, TD <= p를 제안하여 기존 메트릭의 한계를 해결합니다.



### Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech (https://arxiv.org/abs/2410.01162)
- **What's New**: 블록체인 언어 모델(LLM)이 사용자 음성을 인식하고 감정을 반영하여 응답하는 시스템을 개발했습니다. 이 시스템은 LLM의 가중치를 조정하지 않고도 사용자의 감정과 말하는 스타일을 이해할 수 있는 가능성에 대해 연구하였습니다.

- **Technical Details**: 이 연구는 LLM(Llama 3 8B Instruct)과 음성 인코더를 결합한 end-to-end 시스템 임을 강조합니다. 음성 인코더는 감정이 포함된 음성 프롬프트에 맞춰 LLM의 응답을 정렬하며, 준거로 사용된 데이터셋은 감정 인식(SER) 데이터셋입니다. 음성과 텍스트를 동기화하여, 사용할 때마다 같은 반응을 제공할 수 있도록 합니다.

- **Performance Highlights**: SpeechEmotionLlama는 감정이 포함된 음성 프롬프트에 더 높은 품질과 공감적인 응답을 생성하여 성능이 여러 baseline에 비해 개선되었음을 보여줍니다. 이 시스템은 감정 관련 반응 정렬 작업을 추가로 탐색하여 음성 토큰에서 더욱 많은 의미적 정보와 부언어적 정보(paralinguistic information)를 포착할 수 있습니다.



### Unleashing the Power of Large Language Models in Zero-shot Relation Extraction via Self-Prompting (https://arxiv.org/abs/2410.01154)
Comments:
          EMNLP 2024 Short

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 제로샷 관계 추출(zero-shot Relation Extraction, RE) 능력을 최대한 활용할 수 있도록 Self-Prompting 프레임워크를 도입했습니다. 기존의 접근 방식이 세부적인 맥락 특정 프롬프트의 부족으로 최적의 성능을 발휘하지 못했던 문제를 해결하고자 합니다.

- **Technical Details**: Self-Prompting 프레임워크는 LLMs의 내재된 RE 지식을 활용하여 관계에 대한 다수의 합성 샘플을 생성하는 3단계 다양성 접근 방식을 채택합니다. 이 과정에서 관계 동의어 생성, 엔티티 필터링, 문장 재구성을 통해 LLM이 더욱 효과적인 샘플을 생성하도록 돕습니다. 최종적으로 다각적인 합성 데이터를 사용하여 in-context 학습을 수행합니다.

- **Performance Highlights**: 다양한 제로샷 RE 데이터셋에 대한 실험 평가 결과, 제안한 방법이 기존 LLM 기반 제로샷 RE 방법보다 우수한 성능을 보여줍니다. 특히, 3단계 다양성 전략이 샘플의 다양성과 범위를 유의미하게 향상시켜 모델 성능을 높이는 데 기여했습니다.



### Mixing It Up: The Cocktail Effect of Multi-Task Fine-Tuning on LLM Performance -- A Case Study in Financ (https://arxiv.org/abs/2410.01109)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 도메인 특화 적용이 빠르게 성장하고 있으며, 특히 금융 분야에서 이들의 성능 평가에 대한 필요성이 커지고 있습니다. 본 연구에서는 LLMs의 미세 조정에서 멀티태스크 파인튜닝이 더 효과적일 수 있음을 보여주고 있으며, 작은 모델이 더 큰 모델을 초월할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 본 연구에서는 200개 이상의 모델을 학습시키며, 다양한 LLMs를 기준으로 삼아 멀티태스크 파인튜닝의 효과를 조사했습니다. 특히, 관련 기술 작업을 함께 훈련시키는 것이 모델 성능을 향상시킬 수 있는 시너지 효과를 가져온다는 점을 강조하였습니다. 또한, 일반적인 수학적 데이터와 지침 데이터를 포함하여 훈련 과정에서 모델 성능 개선을 위한 정규화 역할을 수행하는 가능성도 탐구하였습니다.

- **Performance Highlights**: Phi-3-Mini 모델이 GPT-4-o 모델을 초월하는 뛰어난 성능을 보였으며, 재무 기초 평가에서 최첨단 결과를 달성하였습니다. 본 연구의 결과는 도메인 특화 작업에서의 멀티태스크 파인튜닝의 중요성과 효과를 강조하며, 단일 작업에 대한 미세 조정이 반드시 도메인 지식의 넓은 향상으로 이어지지는 않는다는 점도 시사합니다.



### Exploring Empty Spaces: Human-in-the-Loop Data Augmentation (https://arxiv.org/abs/2410.01088)
- **What's New**: 이번 연구에서는 데이터 증강(data augmentation)의 중요성을 강조하며, 비구조적 텍스트 데이터셋에서의 'unknown unknowns'를 탐색하는 데 도움을 주는 상호작용 도구인 Amplio를 소개합니다.

- **Technical Details**: Amplio는 사용할 수 있는 데이터 증강 기법으로는 Augment With Concepts, Augment by Interpolation, Augment with Large Language Model(LM)이 포함되어 있으며, 이 세 가지 모두 사용자가 새로운, 관련성 높은 데이터 포인트를 생성할 수 있도록 지원합니다.

- **Performance Highlights**: Amplio를 사용한 사용자 연구에서 18명의 전문 레드 팀원들이 유해 LLM 프롬프트 데이터셋을 증강하는 데 성공적으로 활용하였고, 고품질의 다채로운 데이터를 신속하게 생성할 수 있었음을 보여주었습니다.



### RATIONALYST: Pre-training Process-Supervision for Improving Reasoning (https://arxiv.org/abs/2410.01044)
Comments:
          Our code, data, and model can be found at this repository: this https URL

- **What's New**: 이 논문에서는 LLMs의 추론에서 발생하는 불완전성을 해결하기 위해 RATIONALYST라는 새로운 모델을 제안합니다. 이 모델은 대규모 무표시 데이터에서 추출한 합리적 근거(rationale) 주석을 기반으로 하는 프로세스 감독(process supervision) 방식으로 학습됩니다.

- **Technical Details**: RATIONALYST는 LLaMa-3-8B에서 미세 조정(fine-tuning)되어 79,000개의 합리적 근거를 웹 스케일 무표시 데이터와 최소한의 인적 개입으로 추출한 데이터 집합에서 활용합니다. 이 모델은 연구 결과에서 수학, 상식, 과학, 논리적 추론을 포함한 다양한 추론 작업에서 일반화할 수 있는 능력을 보입니다.

- **Performance Highlights**: RATIONALYST는 대표적인 7개 추론 벤치마크에서 평균 3.9% 향상된 추론 정확도를 보여주며, GPT-4와 같은 훨씬 큰 검증자(compared to significantly larger verifiers) 모델들 및 유사한 크기의 모델과 성능을 비교하여 우수성을 입증하였습니다.



### Show Me What's Wrong!: Combining Charts and Text to Guide Data Analysis (https://arxiv.org/abs/2410.00727)
- **What's New**: 본 논문에서는 다차원 데이터셋에서 이상 패턴을 분석하고 탐지하는 복잡한 과정을 간소화하기 위한 도구를 제안합니다. 이 도구는 자동 정보 하이라이팅, Large Language Model (LLM) 기반의 텍스트 인사이트 및 시각적 분석을 통합하여 사용자의 탐색을 지원합니다.

- **Technical Details**: 제안하는 도구는 사용자가 선정한 데이터 분석 영역에 대한 텍스트 및 그래픽 요약을 제공합니다. 사용자는 데이터를 분리하여 탐색하고, 각 영역에서 필요한 정보를 쉽게 이해할 수 있으며, 각 분석 영역(Knowledge Areas, KAs)에 대한 직관적인 시각적 신호로 추가적인 주의를 기울여야 할 부분을 찾을 수 있습니다. 또한, Hallucination Detection 시스템을 통해 잘못된 정보를 생성할 가능성을 최소화합니다.

- **Performance Highlights**: 일곱 명의 도메인 전문가를 대상으로 한 연구 결과, 제안된 도구가 탐색 분석을 효과적으로 지원하고 의심스러운 정보를 식별하는 데 도움을 준 것으로 나타났습니다. 사용자는 정보 과부하 없이 빠르고 정확한 인사이트를 얻을 수 있습니다.



### Scaling Optimal LR Across Token Horizons (https://arxiv.org/abs/2409.19913)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델) 훈련 시 토큰 지평(token horizon)에 따른 최적 학습률(learning rate, LR)의 변화를 대규모 실험을 통해 조사했습니다. LLM 훈련에 있어 하이퍼파라미터 전이(hyperparameter transfer)가 토큰 지평을 가로막고 있는 중요한 문제로 부각되었습니다.

- **Technical Details**: 이 연구에서는 최적 LR이 토큰 지평에 강하게 의존하며, 긴 훈련 기간에는 더 작은 LR이 필요하다는 점을 보여줍니다. 또한, 최적 LR은 스케일링 법칙(scaling law)을 따르는데, 이를 통해 긴 토큰 지평에 대한 최적 LR을 짧은 지평에서 추론할 수 있습니다. 실험은 Megatron 코드베이스(Megatron codebase)와 RefinedWeb 데이터셋을 바탕으로 진행되었습니다.

- **Performance Highlights**: LLama-1 모델이 너무 높은 LR을 사용했다는 것을 증명하며, 이로 인해 성능 저하가 발생했다는 점을 강조합니다. 이 연구는 데이터 크기 간 하이퍼파라미터 전이가 LLM 훈련에서 간과된 중요한 구성 요소임을 주장합니다.



New uploads on arXiv(cs.IR)

### Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation (https://arxiv.org/abs/2410.01598)
Comments:
          9 pages, 7 figures,The 1st Workshop on Risks, Opportunities, and Evaluation of Generative Models in Recommender Systems (ROEGEN@RecSys 2024), October 2024, Bari, Italy

- **What's New**: 본 논문에서는 새로운 Elaborative Subtopic Query Reformulation (EQR) 방법을 소개합니다. 이는 사용자의 다양한 의도를 파악하여 여행 추천 시스템에서 효과적으로 응답을 생성하는 것을 목표로 합니다. 특히, EQR은 폭넓은 subtopic과 깊이 있는 elaboration을 동시에 제공하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: EQR은 사용자의 자연어 쿼리를 깊이 있게 이해하고 다양한 하위 주제를 생성하기 위해 대형 언어 모델(LLM)을 활용합니다. 이 방법은 기존의 query reformulation 기술의 한계를 극복하며, 보다 효과적인 sparse 및 dense retrieval을 가능하게 합니다. 논문에서는 TravelDest라는 새로운 benchmark 데이터셋도 소개하여, 50개의 넓고 간접적인 NL 쿼리와 관련된 774개의 목적지 도시를 포함합니다.

- **Performance Highlights**: TravelDest 데이터셋에 대한 실험 결과 EQR은 기존의 최첨단 QR 방법들보다 recall과 precision 면에서 유의미한 향상을 보였으며, 특히 넓고 간접적인 NL 쿼리에 대한 대응 능력을 크게 개선했습니다.



### Peeling Back the Layers: An In-Depth Evaluation of Encoder Architectures in Neural News Recommenders (https://arxiv.org/abs/2410.01470)
Comments:
          Accepted at the 12th International Workshop on News Recommendation and Analytics (INRA 2024) in conjunction with ACM RecSys 2024

- **What's New**: 이번 연구는 Neural News Recommender (NNR) 시스템에서 인코더 아키텍처를 체계적으로 분석하여 다양한 인코더 디자인의 유사성과 성과의 차이를 이해하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구에서는 (i) Central Kernel Alignment을 사용하여 학습된 뉴스 및 사용자 표현의 유사성을 평가하고, (ii) Jaccard 계수를 통해 생성된 추천 리스트의 유사성을 측정하며, (iii) 전반적인 추천 성과를 분석합니다.

- **Performance Highlights**: 연구 결과는 복잡한 인코딩 기술의 일부가 실질적으로 불합리할 수 있음을 강조하며, 더 간단하고 효율적인 아키텍처의 필요성을 제시합니다. 또한 뉴스 인코더의 의미적 풍부함과 사용자 인코더의 단순화 가능성을 강조합니다.



### Analyzing Byte-Pair Encoding on Monophonic and Polyphonic Symbolic Music: A Focus on Musical Phrase Segmentation (https://arxiv.org/abs/2410.01448)
Comments:
          Accepted to 3rd Workshop on NLP for Music and Audio (NLP4MusA, co-located with ISMIR 2024)

- **What's New**: 이 연구에서는 Byte-Pair Encoding (BPE) 알고리즘이 텍스트가 아닌 기호 음악에 어떻게 적용되는지 조사합니다. 특히, 다양한 악기에서 BPE의 행동을 분석하고 그것이 단성(모노포닉) 및 다성(폴리포닉) 음악의 음악적 구절 세분화(task) 작업에 미치는 영향을 평가합니다.

- **Technical Details**: BPE는 문자 또는 단어의 하위 요소로 구성된 토큰화 알고리즘으로, 음악에서도 활용되도록 개발되었습니다. 이 연구에서는 다양한 악기에서의 BPE 행위를 분석하고, BPE가 생성한 서브워드(supertokens)가 음악적 특성을 얼마나 잘 포착하는지를 정량적으로 비교합니다. 우리는 MidiTok 패키지를 사용해 토큰화 프로세스를 수행하며, HuggingFace 라이브러리를 이용해 Transformer 모델을 구현합니다.

- **Performance Highlights**: 연구 결과, BPE는 다성 음악에서 성능을 크게 향상시켰으나 단성 음표에서는 특정 BPE 병합 범위 내에서만 성능을 개선하는 것으로 나타났습니다. 또한, BPE가 생성한 슈퍼토큰의 사용 빈도와 길이 변화를 분석하여, 다양한 악기에서 음악과 텍스트 간의 뚜렷한 차이를 확인했습니다.



### PairDistill: Pairwise Relevance Distillation for Dense Retrieva (https://arxiv.org/abs/2410.01383)
Comments:
          Accepted to EMNLP 2024 Main Conference

- **What's New**: 이 논문에서는 Pairwise Relevance Distillation (PairDistill)이라는 새로운 방법을 제안하여, 문서 간의 상대적 유사성을 평가하는 pairwise reranking 기법을 활용하여 dense retrieval 모델의 훈련 성능을 향상시키고 있습니다. 기존의 pointwise rerankers와 달리, pairwise 접근법은 유사한 문서 간의 세밀한 구별을 가능하게 합니다.

- **Technical Details**: PairDistill은 pairwise reranker에서 제공하는 미세한 훈련 신호를 활용하여 retrieve 모델의 훈련을 개선합니다. 이를 통해 dense retrieval 모델은 밀접하게 순위가 매겨진 패시지들 간의 미세한 차이를 배울 수 있습니다. 이 방법은 ColBERT 및 DPR 아키텍처에 효과적으로 적용되었습니다.

- **Performance Highlights**: PairDistill은 다양한 벤치마크에서 기존 방법들을 능가하며, 새로운 최첨단 결과를 달성했습니다. 또한, 기존의 유사한 크기의 dense retrieval 모델들보다 성능이 현저히 향상되었음을 보여주었습니다.



### Integrating Visual and Textual Inputs for Searching Large-Scale Map Collections with CLIP (https://arxiv.org/abs/2410.01190)
Comments:
          18 pages, 7 figures, accepted at the Computational Humanities Research Conference (CHR 2024)

- **What's New**: 이 연구는 자연어 입력, 시각적 입력 및 다중 모드 입력을 활용하여 대규모 지도 컬렉션을 인터랙티브하게 탐색할 수 있는 가능성을 탐구합니다. 56만 장 이상의 지도 이미지를 사용하여 CLIP 모델을 통해 임베딩을 생성하고, 이를 통해 다양한 검색 방법을 구현하였습니다.

- **Technical Details**: 연구팀은 Library of Congress의 API를 통해 562,842장의 지도 이미지를 수집하고, OpenAI의 Contrastive Language-Image Pre-training (CLIP) 모델을 사용하여 이들 지도에 대한 임베딩을 생성했습니다. 탐색 가능성을 높이기 위해 자연어 검색, 이미지 검색 및 다중 모드 검색을 구현하였고, 이러한 검색 구현은 소비자용 GPU를 사용하여 1초 이내에 결과를 반환할 수 있습니다.

- **Performance Highlights**: 이 시스템은 56만 장 이상의 지도 이미지를 1초 내에 검색할 수 있는 빠른 응답성을 자랑합니다. 또한 10,504개의 지도-캡션 쌍의 데이터셋을 도입하고, 이를 CLIP 모델의 파인 튜닝에 활용할 수 있는 아키텍처를 제공합니다. 모든 코드는 공개 도메인으로 제공되며, Jupyter 노트북 형태로 상용구가 제공됩니다.



### GraphRevisedIE: Multimodal Information Extraction with Graph-Revised Network (https://arxiv.org/abs/2410.01160)
- **What's New**: 이번 논문에서는 GraphRevisedIE라는 경량 모델을 제안하여 visually rich documents (VRD)에서의 key information extraction (KIE) 문제를 개선하고자 하였다. 이 모델은 텍스트, 시각 및 레이아웃 기능을 효과적으로 융합하여 다중 모드 기능을 이용한 KIE 작업의 성능을 향상시킨다.

- **Technical Details**: GraphRevisedIE는 그래프 수정(graph revision) 기술을 활용하여 문서의 그래프 표현을 학습하고, graph convolution을 통해 글로벌 컨텍스트로 다중 모드 기능의 삽입을 강화한다. 그래프 모듈은 희소 문서에 적절한 그래프 표현을 학습할 수 있도록 sparsification 기술을 적용한다.

- **Performance Highlights**: 다양한 실제 데이터 세트에 대한 실험 결과, GraphRevisedIE는 기존의 그래프 기반 모델들보다 더 우수한 성능을 보였으며, 사전 훈련된 모델들과 비교했을 때도 비슷한 성능을 발휘하면서 매개 변수가 훨씬 적고 큰 사전 훈련 데이터 세트에 의존하지 않는다.



### Unleashing the Power of Large Language Models in Zero-shot Relation Extraction via Self-Prompting (https://arxiv.org/abs/2410.01154)
Comments:
          EMNLP 2024 Short

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 제로샷 관계 추출(zero-shot Relation Extraction, RE) 능력을 최대한 활용할 수 있도록 Self-Prompting 프레임워크를 도입했습니다. 기존의 접근 방식이 세부적인 맥락 특정 프롬프트의 부족으로 최적의 성능을 발휘하지 못했던 문제를 해결하고자 합니다.

- **Technical Details**: Self-Prompting 프레임워크는 LLMs의 내재된 RE 지식을 활용하여 관계에 대한 다수의 합성 샘플을 생성하는 3단계 다양성 접근 방식을 채택합니다. 이 과정에서 관계 동의어 생성, 엔티티 필터링, 문장 재구성을 통해 LLM이 더욱 효과적인 샘플을 생성하도록 돕습니다. 최종적으로 다각적인 합성 데이터를 사용하여 in-context 학습을 수행합니다.

- **Performance Highlights**: 다양한 제로샷 RE 데이터셋에 대한 실험 평가 결과, 제안한 방법이 기존 LLM 기반 제로샷 RE 방법보다 우수한 성능을 보여줍니다. 특히, 3단계 다양성 전략이 샘플의 다양성과 범위를 유의미하게 향상시켜 모델 성능을 높이는 데 기여했습니다.



### Can We Delegate Learning to Automation?: A Comparative Study of LLM Chatbots, Search Engines, and Books (https://arxiv.org/abs/2410.01396)
Comments:
          21 pages, 14 figures

- **What's New**: 이 연구는 LLM(대규모 언어 모델) 기반 채팅봇의 교육적 효과와 학습 결과에 대한 교육자들의 우려를 심층적으로 조사했습니다. 또한, LLM이 제공하는 자동화된 학습 도구와 기존의 학습 도구(교과서 및 웹자료)의 비교를 통해 학습 결과에 미치는 영향을 분석했습니다.

- **Technical Details**: 이 연구는 92명의 대학생을 대상으로 한 혼합 방법 연구(mixed-methods study)로, 세 가지 학습 도구(책, 웹, ChatGPT)와 그 자동화 수준을 비교했습니다. 교육자들의 우려는 (1) 신뢰성 부족, (2) 체계적 조직이 부족, (3) 인지적 참여가 약하다는 세 가지 주요 원인으로 정리되었습니다.

- **Performance Highlights**: LLM 기반 채팅봇은 개념에 대한 포괄적인 이해를 지원했지만 장기 기억 유지는 책보다 덜 효과적이었습니다. 학업 성과가 높은 학습자들은 검색 활동보다 콘텐츠에 더 깊이 참여하며 더 나은 학습 성과를 보였습니다. 결과적으로 학습 도구보다 학생의 개인적 능력이 수동적 학습에 더 많은 영향을 미쳤습니다.



### Text Clustering as Classification with LLMs (https://arxiv.org/abs/2410.00927)
Comments:
          12 pages, 3 figures

- **What's New**: 이 연구는 대형 언어 모델(LLM)의 인컨텍스트 학습(in-context learning) 능력을 활용하는 새로운 텍스트 클러스터링 프레임워크를 제안합니다. 기존의 세밀한 조정이 필요 없는 클러스터링 기법을 통해 데이터셋의 레이블 생성과 분류를 LLM으로 수행하도록 합니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 미니 배치(mini-batch)로 데이터를 입력하고 LLM에게 레이블 생성 작업을 요청합니다. 두 번째 단계에서는 생성된 유사 레이블을 통합한 후, 각 샘플에 가장 적합한 레이블을 할당하도록 LLM에 요청합니다. 이 방법은 데이터셋을 순차적으로 처리하여 LLM의 입력 길이 제한을 피합니다.

- **Performance Highlights**: 연구 결과, 제안한 프레임워크는 5개의 다양한 데이터셋에 대해 상태-최고(clustering methods)와 비교하여 유사하거나 더 나은 성능을 보였습니다. 특히, 복잡한 하이퍼파라미터 조정이나 데이터셋에 따른 세밀한 조정이 필요 없으므로 시간과 계산 자원을 significantly 절약할 수 있습니다.



New uploads on arXiv(cs.CV)

### Samba: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking (https://arxiv.org/abs/2410.01806)
- **What's New**: 본 논문에서는 Samba라는 새로운 linear-time set-of-sequences 모델을 소개하며, 이를 통해 여러 tracklet들을 동기화된 상태로 처리하여 복잡한 이동 패턴과 상호작용을 모델링하는 방법을 제안합니다. 또한, SambaMOTR라는 최초의 tracking-by-propagation 트래커를 통해 이전의 문제점들을 해결하고, MaskObs라는 기술을 도입하여 불확실한 관측치를 효과적으로 처리하는 방법을 제공합니다.

- **Technical Details**: Samba는 복수의 time-series 데이터 (tracklets)를 동기화된 긴 메모리 표현으로 압축하여 처리합니다. 이 과정에서 self-attention mechanism을 통해 tracklet들 간의 정보를 교환하면서 interdependencies를 고려합니다. SambaMOTR는 이를 기반으로 하여, autoregressive 방식으로 다음 track query를 예측하며, occlusion(가림현상) 문제를 효과적으로 처리하는 새로운 쿼리 전파 모듈을 탑재하고 있습니다.

- **Performance Highlights**: SambaMOTR는 DanceTrack, BFT, SportsMOT 데이터셋에서의 성능을 크게 향상시켜 새로운 state-of-the-art를 기록했습니다. 특히, SambaMOTR는 tracklet 간의 상호작용과 긴-range dependencies를 정확하게 모델링하여 가림현상에서도 효과적으로 객체를 추적할 수 있는 능력을 보여주었습니다.



### EVER: Exact Volumetric Ellipsoid Rendering for Real-time View Synthesis (https://arxiv.org/abs/2410.01804)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 Exact Volumetric Ellipsoid Rendering(불러와지는 텍스트)이라는 새로운 방법을 소개합니다. 이 방법은 실시간으로 차별화 가능한 emission-only volume rendering을 가능하게 하며, 기존의 3D Gaussian Splatting(3DGS) 방식과는 달리 정밀한 볼륨 렌더링을 지원합니다. 이를 통해 popping artifacts 문제를 해결하고, NVIDIA RTX4090에서 720p 해상도로 약 30 FPS의 프레임레이트를 달성합니다.

- **Technical Details**: 이 방법은 ray tracing을 기반으로 하며, constant density ellipsoid 기반 표현을 사용하여 정확한 볼륨 렌더링 적분을 효율적으로 계산합니다. 기존의 3DGS는 가우시안 프리미티브 오버랩에 따라 발생하는 문제를 가지고 있었지만, 우리의 방법은 이러한 불일치를 해결합니다. 더불어, 다양한 광학적 효과(예: fisheye 카메라에서의 왜곡 효과)도 쉽게 모델링할 수 있습니다.

- **Performance Highlights**: 우리는 Zip-NeRF 데이터 세트의 어려운 대규모 장면에서 특히 성능이 잘 나타나며, 기존의 3DGS 기법보다 더 높은 화질을 보여 줍니다. 우리의 접근법은 실시간 렌더링을 보장하면서도, 3DGS 기초에 비해 이미지 품질을 크게 향상시킵니다.



### FabricDiffusion: High-Fidelity Texture Transfer for 3D Garments Generation from In-The-Wild Clothing Images (https://arxiv.org/abs/2410.01801)
Comments:
          Accepted to SIGGRAPH Asia 2024. Project page: this https URL

- **What's New**: FabricDiffusion은 단일 의류 이미지를 3D 의상에 매핑하여 섬유 질감을 이전하는 새로운 방법을 제안합니다. 이 방법은 기존의 2D-3D 질감 매핑 및 깊이 인식을 통한 이미지 보정 기법의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: FabricDiffusion은 대규모 합성 데이터 세트를 기반으로 훈련된 디노이징 확산 모델을 이용하여 입력 텍스처 이미지의 왜곡을 수정하고, 이를 통해 고품질의 질감 맵을 생성합니다. 이 기술은 PBR(Physically-Based Rendering) 방법론과 긴밀하게 결합되어 다양한 조명 조건에서 의상을 사실적으로 재조명할 수 있습니다.

- **Performance Highlights**: FabricDiffusion은 합성 데이터와 실제 의류 이미지 모두에서 최첨단 기법을 능가하는 성능을 보이며, 이전에 보지 못한 질감과 의상 형태에 대해서도 일반화하는 능력을 보여주었습니다.



### SegEarth-OV: Towards Traning-Free Open-Vocabulary Segmentation for Remote Sensing Images (https://arxiv.org/abs/2410.01768)
- **What's New**: 본 논문에서는 원거리 감지 이미지에서 오픈 어휘 의미 분할(open-vocabulary semantic segmentation, OVSS) 기술을 적용하여 수작업 주석의 필요성을 해결합니다. 이를 위해 새롭게 제안된 SimFeatUp 모듈을 통해 저해상도 특징을 복원하고, CLIP에서 발생하는 전역 바이어스를 감소시키기 위한 간단한 뺄셈 방법을 도입합니다.

- **Technical Details**: SimFeatUp은 저해상도(Low Resolution) 특징을 효과적으로 업샘플링하고, 원거리 감지 이미지에서의 의미적 일관성을 유지합니다. 또한, CLIP 기반 OVSS 접근법에서 [CLS] 토큰에 의한 글로벌 바이어스를 줄이기 위해 로컬 패치 토큰과 글로벌 토큰 간의 간단한 뺄셈 작업을 수행합니다. 실험 결과, 17개의 데이터셋에서 다양한 작업에 대해 선진 기술 대비 평균 5.8%, 8.2%, 4%, 15.3%의 향상이 도출되었습니다.

- **Performance Highlights**: 우리의 방법은 건물 추출, 도로 탐지, 홍수 감지와 같은 4가지 작업에서 현재 최고 성능을 기록했습니다. SegEarth-OV라는 최종 모델은 17개의 원거리 감지 데이터셋에서 상태-of-the-art 성능을 달성하고 코드가 공개되었습니다.



### ImageFolder: Autoregressive Image Generation with Folded Tokens (https://arxiv.org/abs/2410.01756)
Comments:
          Code: this https URL

- **What's New**: 이번 논문에서는 이미지 생성 품질을 개선하기 위한 이미지 토크나이저의 효과를 분석하고, 새로운 세멘틱 토크나이저인 ImageFolder를 제안합니다. 이 모델은 오토회귀 모델링 시 더 짧은 토큰 길이를 유지하면서도 높은 수준의 재구성과 생성 품질을 제공합니다.

- **Technical Details**: ImageFolder는 공간적으로 정렬된 이미지 토큰을 제공하여 오토회귀 모델링 동안 토큰 길이를 단축할 수 있도록 설계되었습니다. 이 모델은 듀얼 브랜치 제품 양자화(dual-branch product quantization)를 사용하여 이미지를 위한 서로 다른 맥락을 포착합니다. 각 브랜치에는 세멘틱 정규화(semantic regularization)와 함께 픽셀 수준의 세부 정보를 캡처하는 추가적인 구조가 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, ImageFolder 토크나이저를 사용했을 때 높은 품질의 이미지 생성과 더 짧은 토큰 길이를 동시에 달성할 수 있음을 보여주었습니다. 특히, 토큰의 수는 그대로 유지되더라도, 접힌 토큰(folded tokens)이 펼쳐진 토큰(unfolded tokens)보다 더 나은 생성 성능을 보이는 것으로 나타났습니다.



### LEOPARD : A Vision Language Model For Text-Rich Multi-Image Tasks (https://arxiv.org/abs/2410.01744)
Comments:
          Our code is available at this https URL

- **What's New**: 본 논문에서는 다중 텍스트-리치 이미지(텍스트가 주요 시각 요소인 이미지)에 대한 과제를 해결하기 위해 특별히 설계된 새로운 다중모달 대형 언어 모델(MLLM), Leopard를 소개합니다. 이 모델은 약 천만 개의 고품질 다중모달 교육 데이터를 기반으로 만들어졌으며, 텍스트-리치, 다중 이미지 시나리오에 맞춰 최적화된 설계로 구현되었습니다.

- **Technical Details**: Leopard는 텍스트-리치 다중 이미지 처리를 위한 적응형 고해상도 다중 이미지 인코딩 모듈을 통합하여, 입력 이미지의 원래 비율과 해상도에 따라 비주얼 시퀀스의 길이를 다이내믹하게 최적화합니다. 이 방법은 여러 고해상도 이미지를 수용하면서도 세부 사항이나 선명도를 손상시키지 않도록 합니다.

- **Performance Highlights**: Leopard 모델은 13개 시각-언어 벤치마크 데이터셋에서 평가되었으며, 5개의 텍스트-리치 다중 이미지 벤치마크에서 평균 +9.61 점의 우수한 성능을 보여 기존 오픈 소스 MLLM보다 뛰어난 결과를 기록했습니다. 또한, Leopard는 텍스트-리치 단일 이미지 과제와 일반 영역의 시각-언어 벤치마크에서도 경쟁력 있는 성과를 달성했습니다.



### VitaGlyph: Vitalizing Artistic Typography with Flexible Dual-branch Diffusion Models (https://arxiv.org/abs/2410.01738)
Comments:
this https URL

- **What's New**: VitaGlyph라는 새로운 위스타일화 기법이 개발되었습니다. 이 기법은 입력 문자(예: ‘장미’)를 주제(Subject)와 주변(Surrounding)으로 나누어 읽기 쉬운 예술적 타이포그래피를 생성할 수 있도록 합니다.

- **Technical Details**: VitaGlyph는 세 단계에서 작동합니다: (1) Knowledge Acquisition: 대형 언어 모델을 사용하여 주제와 주변에 대한 텍스트 설명을 설계합니다. (2) Regional Decomposition: Grounding-DINO를 사용하여 주제 설명과 가장 잘 일치하는 이미지를 찾아 입력 글리프 이미지를 주제와 주변 지역으로 분해합니다. (3) Typography Stylization: 주제 구조를 Semantic Typography로 개선하고, 별도로 텍스처를 렌더링합니다.

- **Performance Highlights**: VitaGlyph는 예술성과 가독성을 동시에 향상시킴을 입증하며, 여러 개의 맞춤형 개념을 포함하여 보다 창의적이고 매력적인 예술적 타이포그래피 생성이 가능합니다.



### RADAR: Robust Two-stage Modality-incomplete Industrial Anomaly Detection (https://arxiv.org/abs/2410.01737)
- **What's New**: 이 논문에서는 Modality-Incomplete Industrial Anomaly Detection (MIIAD)에 대한 첫 번째 포괄적인 연구를 소개합니다. 기존의 Multimodal Industrial Anomaly Detection (MIAD) 모델은 모든 2D 및 3D 모달이 쌍으로 이루어져 있다고 가정하는 경향이 있지만, 실제 데이터는 종종 모달이 누락되어 불완전하다는 점을 간과하고 있습니다.

- **Technical Details**: 본 연구는 Robust modAlity-imcomplete fusing and Detecting frAmewoRk (RADAR)라는 새로운 2단계 프레임워크를 제안합니다. 첫 번째 단계는 특징 융합(feature fusion)으로, 미완전한 모달 리 학습(instruction)과 HyperNetwork 기반의 적응형 파라미터 학습을 통해 Multimodal Transformer의 강인성을 향상시킵니다. 두 번째 단계에서는 실-가상 하이브리드 모듈(real-pseudo hybrid module)을 구축하여 모달 조합의 독특성을 강화합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, 제안된 RADAR는 MIIAD 데이터셋에서 기존의 MIAD 방법들과 비교할 때 효과성과 강인성 측면에서 유의미하게 우수한 성과를 보였습니다.



### ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation (https://arxiv.org/abs/2410.01731)
Comments:
          Project website: this https URL

- **What's New**: 이번 논문에서는 사용자 프롬프트에 자동으로 적합하게 조정되는 텍스트-이미지 생성 워크플로를 생성하는 새로운 작업을 소개합니다. 두 가지 LLM 기반 접근법을 제안하며, 효율적인 워크플로 설계를 위한 수많은 컴포넌트와 그 복잡한 상호 의존성을 고려합니다.

- **Technical Details**: 我们提出了两种方法：调优方法和无训练方法。调优方法从用户偏好数据中学习，而无训练方法利用LLM选择现有的流。为了训练LLM，我们从500500个多样化的用户提示中收集数据，并使用人类偏好估计器对生成的图像进行评分。

- **Performance Highlights**: ComfyGen-IC 및 ComfyGen-FT 접근법은 기본 모델과 일반적인 비프롬프트 워크플로우에 비해 이미지 품질을 향상시킵니다. 우리의 방법은 사람의 선호도 및 프롬프트 정렬 벤치마크에서 모든 기준 모델을 초월하며, 프롬프트 의존적 흐름 예측이 텍스트-이미지 생성 품질을 향상시키는 새로운 경로가 됨을 보여줍니다.



### HarmoniCa: Harmonizing Training and Inference for Better Feature Cache in Diffusion Transformer Acceleration (https://arxiv.org/abs/2410.01723)
Comments:
          Code will be released soon

- **What's New**: Diffusion Transformers (DiTs)를 위한 새로운 학습 기반 캐싱 프레임워크인 HarmoniCa를 제안하며, 이는 훈련(training)과 추론(inference) 과정의 조화를 이루게 해준다.

- **Technical Details**: HarmoniCa는 Step-Wise Denoising Training (SDT)와 Image Error Proxy-Guided Objective (IEPO)에 기반하여 개발되었다. SDT는 훈련 중 이전 타임스텝의 정보를 활용하여 노이즈 제거 과정의 연속성을 유지하며, IEPO는 캐시 재사용에 의한 최종 이미지 오류를 근사화하는 효율적인 프록시(proxy) 기제를 통합한다.

- **Performance Highlights**: HarmoniCa는 기존의 캐싱 기법들에 비해 훈련 과정에서의 캐시 사용을 고려하여 최종 이미지 품질과 캐시 활용의 균형을 이루며, 성능과 가속 비율이 향상되는 것을 확인하였다.



### OmniSR: Shadow Removal under Direct and Indirect Lighting (https://arxiv.org/abs/2410.01719)
- **What's New**: 이번 연구는 실내 및 실외 장면에서 직접 조명 및 간접 조명이 결합된 그림자를 처리하기 위한 새로운 렌더링 파이프라인을 제안합니다. 이를 통해 30,000개 이상의 이미지 쌍으로 구성된 포괄적인 합성 데이터셋을 생성하였습니다. 또한, 의미(semantic) 및 기하학적(geometric) 정보를 통합한 혁신적인 그림자 제거 네트워크를 개발하였습니다.

- **Technical Details**: 연구팀은 경로 추적(path tracing) 프레임워크를 사용하여 직접 그림자(Direct Shadows)와 간접 그림자(Indirect Shadows)를 정의하고 렌더링합니다. 제안된 네트워크는 RGB-Depth (RGBD) 입력을 이용하고, 의미적 특징을 결합하여 로컬 주의(local attention) 블록 내에서 기하학적 및 의미적 유사성에 따라 특징을 재조정합니다. 이 방법은 그림자 없는 이미지를 복원하기 위한 보다 정밀한 가이드를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신 그림자 제거 기술에 비해 우수한 성능을 나타내며, 다양한 조명 조건에서 실내 및 실외 장면에 효과적으로 일반화될 수 있음을 보여주었습니다.



### COMUNI: Decomposing Common and Unique Video Signals for Diffusion-based Video Generation (https://arxiv.org/abs/2410.01718)
- **What's New**: 본 논문에서는 비디오 생성의 효율성을 위해 COMUNI라는 새로운 diffusion 기반 프레임워크를 제안합니다. 이 프레임워크는 비디오 신호를 공통(COMmon) 및 고유(UNIque) 신호로 분해하여 중복된 모델링을 방지합니다.

- **Technical Details**: 논문은 CU-VAE를 사용하여 비디오 신호를 분해하고 잠재 특징(latent features)으로 인코딩합니다. 이는 self-supervised 방식으로 학습되며, 비디오 신호를 재구성하고 비디오 프레임을 복원하기 위해 cascading merge module과 non-time-sensitive video decoder를 사용합니다. CU-LDM은 공통 및 고유 잠재 특징을 동시에 모델링하는 두 개의 diffusion 스트림을 채택하여 비디오 생성을 수행합니다.

- **Performance Highlights**: 실험 결과, 공통 및 고유 신호의 분해가 비디오 생성에 필수적임을 입증하였고, 6번째 샘플링 전략에서 가장 우수한 FVD score를 기록함으로써 제안된 방법의 효과성과 효율성을 보여주었습니다.



### Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding (https://arxiv.org/abs/2410.01699)
- **What's New**: 이 논문에서는 학습 없이 전체 자율 회귀(auto-regressive) 텍스트-이미지 생성(Generation)을 가속할 수 있는 새로운 알고리즘인 Speculative Jacobi Decoding(SJD)를 제안합니다.

- **Technical Details**: SJD는 확률적 수렴 기준을 도입하여 각 단계를 통해 다수의 토큰을 병렬로 예측하도록 모델을 변경하고, 이로 인해 전통적인 다음 토큰 예측 패러다임보다 적은 단계에서 이미지를 생성할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, SJD는 거의 시각적 품질 손실 없이 여러 자율 회귀 텍스트-이미지 생성 모델의 생성 속도를 약 2배 가속할 수 있으며, 특정 상황에서는 3배 이상의 가속 비율을 달성할 수 있음을 보여줍니다.



### Open3DTrack: Towards Open-Vocabulary 3D Multi-Object Tracking (https://arxiv.org/abs/2410.01678)
Comments:
          7 pages, 4 figures, 3 tables

- **What's New**: 이 연구에서는 기존의 제한된 객체 카테고리에 구애받지 않고 새로운 객체를 포함할 수 있는 open-vocabulary 3D 추적 시스템을 소개합니다. 이는 자율주행 차량이 예상치 못한 객체를 인식하고 안전하게 항해할 수 있도록 도와줍니다.

- **Technical Details**: 이 방법론은 2D의 open-vocabulary 방식을 3D 추적 프레임워크와 통합하여, 이전에 학습하지 않은 객체 클래스를 일반화할 수 있도록 합니다. 이를 통해 새로운 객체들에 대한 추적 성능 격차를 줄이며, 실시간으로 다양한 환경에서의 robustness와 adaptability를 높입니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 익숙한 객체와 새로운 객체 모두를 효과적으로 추적할 수 있음을 보여줍니다. 특히, 다양한 야외 주행 시나리오에서의 성능이 입증되었으며, 완전한 공개 데이터셋과 코드가 제공되어 커뮤니티에서의 활용이 용이합니다.



### 3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for 3D Object Detection (https://arxiv.org/abs/2410.01647)
Comments:
          Code Page: this https URL

- **What's New**: 본 논문은 3D Gaussian Splatting(3DGS)을 처음으로 3D Object Detection(3DOD)에 도입하여 기존 NeRF 기반 방법의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 3DGS는 명시적 3D 표현을 사용하여 Gaussian blobs를 통해 3D 장면을 모델링합니다. 이 연구는 Gaussian blobs의 모호한 공간 분포와 과도한 배경 blobs 문제를 해결하기 위해 2D Boundary Guidance와 Box-Focused Sampling 전략을 제안합니다. 이들 전략을 통해 Gaussian blobs의 공간적 분포를 개선하고 정확한 개체 탐지를 제공합니다.

- **Performance Highlights**: 제안된 3DGS-DET는 ScanNet 데이터셋에서 mAP@0.25에서 +6.6, mAP@0.5에서 +8.1의 성능 향상을 보였으며, ARKITScenes 데이터셋에서는 mAP@0.25에서 +31.5의 개선을 달성하여 기존 SOTA 방법인 NeRF-Det을 크게 초월했습니다.



### Data Extrapolation for Text-to-image Generation on Small Datasets (https://arxiv.org/abs/2410.01638)
- **What's New**: 이번 논문에서는 텍스트-이미지 생성 분야에서 새로운 데이터 증강 방법으로 선형 외삽(linear extrapolation)을 제안합니다. 기존의 데이터 증강 방법은 크롭(cropping), 플리핑(flipping) 등의 단순한 기법에 의존해 새로운 정보를 도입하지 못했으나, 본 연구는 검색 엔진을 통해 외부에서 이미지를 수집하여 텍스트 특징에만 선형 외삽을 적용해 데이터 양을 대폭 증가시킵니다.

- **Technical Details**: 이 방법의 핵심은 두 가지 아웃라이어 탐지기(outlier detectors)를 사용하여 검색된 이미지의 신뢰성을 보장하는 것입니다. K-means 알고리즘을 사용해 노이즈가 많은 웹 이미지를 군집화하고, CLIP 인코더를 통해 데이터셋 이미지와의 거리를 측정하여 아웃라이어를 제거합니다. 또한, NULL-condition guidance를 통해 텍스트-이미지 생성의 점수 추정(score estimation)을 정제하며, 복잡한 텍스트 정보를 처리하기 위해 재귀적 아핀 변환(recurrent affine transformation)을 사용합니다.

- **Performance Highlights**: 본 모델은 CUB, Oxford, COCO 데이터셋에서 각각 FID(Frechet Inception Distance) 점수 7.91, 9.52, 5.00을 달성했습니다. 이 결과는 원본 데이터셋보다 수십 배 많은 훈련 샘플을 기반으로 하여 텍스트-이미지 성능에서显著한 개선을 나타냅니다.



### LMOD: A Large Multimodal Ophthalmology Dataset and Benchmark for Large Vision-Language Models (https://arxiv.org/abs/2410.01620)
- **What's New**: 이번 연구에서는 망막 이미지를 포함한 21,993장의 사진이 포함된 LMOD (Large Multimodal Ophthalmology Dataset)라는 데이터셋과 벤치마크를 통해 LVLMs(large vision-language models)의 성능을 평가하는 방식이 소개되었습니다.

- **Technical Details**: LMOD는 Optical Coherence Tomography (OCT), Scanning Laser Ophthalmoscopy (SLO), 색상 안저 사진(Color Fundus Photographs), 수술 장면(Surgical Scenes) 등 다양한 안과 이미지를 포함하며 multi-granular annotations(다양한 주석)를 제공합니다. 논문에서는 13개의 최신 LVLM의 성능을 해부학적 인식(anatomical recognition), 질병 진단 및 분류(disease diagnosis and classification), 인구 통계 정보 추출(demographic information extraction) 등 3가지 측면에서 평가했습니다.

- **Performance Highlights**: LVLM 모델들은 안과 이미지를 이해하는 데 있어 큰 어려움을 겪었으며, 특히 진단 분석(diagnostic analysis)과 인구 통계 정보 추출에서 한계를 보였습니다. 연구 결과, spatial reasoning(공간 추론)과 out-of-domain queries(도메인 외 쿼리 처리)에 대한 약점이 도출되었습니다.



### SGBA: Semantic Gaussian Mixture Model-Based LiDAR Bundle Adjustmen (https://arxiv.org/abs/2410.01618)
- **What's New**: 본 연구에서는 기존의 전형적인 LiDAR bundle adjustment (BA) 방법이 미리 정의된 기하학적 특성에 의존하는 것을 탈피하여, 환경을 기하학적 및 의미론적 정보를 모두 사용하는 세멘틱 가우시안 혼합 모델(Semantic Gaussian Mixture Model, GMM)로 모델링하는 SGBA 기법을 제안합니다.

- **Technical Details**: SGBA는 고정된 특성과 상관없이 다양한 환경에 적응할 수 있는 포괄적인 랜드마크 표현을 제공합니다. 이를 위해 조건 수(condition number)를 평가하여 최적화를 위한 가장 정보가 풍부한 세멘틱 클러스터를 선택하는 적응형 세멘틱 선택 프레임워크를 도입합니다. 또한, 확률적 특성 연관 기법을 통해 전체 확률 밀도를 고려하며, 측정 및 초기 포즈 추정의 불확실성을 효과적으로 관리합니다.

- **Performance Highlights**: 다양한 실험 결과, SGBA는 초기 포즈 추정이 낮고 기하학적 특성이 제한된 도전적인 상황에서도 정확하고 강력한 포즈 정제를 달성할 수 있음을 입증하였습니다.



### Saliency-Guided DETR for Moment Retrieval and Highlight Detection (https://arxiv.org/abs/2410.01615)
Comments:
          8 pages, 1 figure, 4 tables

- **What's New**: 본 논문에서는 비디오 순간 검색(video moment retrieval) 및 하이라이트 탐지(highlight detection) 작업에서 텍스트와 비디오 특성을 효율적으로 정렬할 수 있는 새로운 아키텍처를 제안합니다. Saliency-Guided Cross Attention 메커니즘과 하이브리드 DETR 아키텍처를 결합하여 성능을 크게 향상시킵니다. 또한 InterVid-MR이라는 대규모 고품질 데이터셋을 개발하여 모델을 사전 훈련(pretraining)했습니다.

- **Technical Details**: 제안된 아키텍처는 Locally Saliency Scores를 생성하여 텍스트와 비디오 임베딩을 직접 비교하고, 전체 비디오의 글로벌 컨텍스트를 포함하여 이를 개선합니다. 또한 DETR 원리를 적용한 Moment-DETR 구조를 기반으로 하여, 하나의 변형(variation) 구조와 더 나아가 IoU 스코어링 메커니즘을 조합하여 비디오 스팬을 더 정확하게 로컬라이제이션합니다.

- **Performance Highlights**: 본 연구의 모델은 QVHighlights, Charades-STA, TACoS 벤치마크에서 최고 성능을 기록하였으며, InterVid-MR 데이터셋으로 사전 훈련된 경우 제로샷(zero-shot) 평가에서도 우수한 결과를 보여주었습니다.



### Gaussian Splatting in Mirrors: Reflection-Aware Rendering via Virtual Camera Optimization (https://arxiv.org/abs/2410.01614)
Comments:
          To be published on 2024 British Machine Vision Conference

- **What's New**: 본 연구는 3D Gaussian Splatting(3D-GS) 기반의 새로운 렌더링 방법을 제안하며, 특히 거울이 포함된 장면에서 일관된 반사 렌더링을 위해 가상 카메라를 모델링합니다.

- **Technical Details**: 3D-GS를 사용하여 깊이(depth) 및 법선(normal) 추정으로부터 거울 평면을 추정하고, 이를 기반으로 대칭적으로 배치된 가상 카메라를 정의합니다. 또한, 가상 카메라 최적화 방법을 통해 반사 퀄리티를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최신 기술들과 비슷하거나 뛰어난 성능을 보이는 한편, 훈련 시간을 대폭 단축시켰습니다.



### DRUPI: Dataset Reduction Using Privileged Information (https://arxiv.org/abs/2410.01611)
- **What's New**: 본 논문에서는 기존의 데이터셋 축소(Dataset Reduction, DR) 기법을 발전시켜, 줄인 데이터셋과 함께 특권 정보(Privileged Information)를 합성하는 DRUPI( 데이터셋 축소를 위한 특권 정보 활용) 기법을 제안합니다. 이 방법은 모델 학습을 개선하기 위해 추가 학습 대상을 도입합니다.

- **Technical Details**: DRUPI는 기존의 데이터-레이블 쌍 외에 특징 레이블(feature labels) 또는 주의 레이블(attention labels)과 같은 특권 정보를 추가로 합성하여 보조적인 감독(supervision)을 제공합니다. 효과적인 특징 레이블은 지나치게 차별적이지도, 지나치게 다양하지도 않아야 하며, 적절한 수준에서 균형을 이뤄야 합니다.

- **Performance Highlights**: ImageNet, CIFAR-10/100 및 Tiny ImageNet에서의 광범위한 실험 결과, DRUPI는 기존의 데이터셋 축소 방법과 매끄럽게 통합되며, 성능 향상을 가져옵니다. 예를 들어, CIFAR10에서 Herding 방법에 DRUPI를 적용하면 성능이 24.3% 향상되며, K-center 방법에서는 최대 23.4%의 개선 효과를 보여줍니다.



### DAViD: Domain Adaptive Visually-Rich Document Understanding with Synthetic Insights (https://arxiv.org/abs/2410.01609)
Comments:
          Work in progress

- **What's New**: 이번 논문에서는 Visually-Rich Document Understanding (VRDU)에서의 도메인 적응을 위하여 기계 생성 합성 데이터(synthetic data)를 활용하는 DAViD라는 프레임워크를 제안합니다. 이는 고비용 전문가 주석(annotation) 없이도 도메인-specific VRDU 작업에 효과적으로 적응할 수 있는 방법을 제시합니다.

- **Technical Details**: DAViD 프레임워크는 문서 표현 학습(document representation learning)에서 fine-grained(세부 수준) 및 coarse-grained(포괄적인 수준)의 처리를 통합하여, 기계 생성 합성 주석(synthetic annotations)을 사용해 전문가 주석에 대한 의존도를 줄입니다. 이를 통해 VRDU 성능을 향상시키고, 다양한 도메인에 대한 적응 전략을 적용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DAViD는 대규모 주석 데이터셋에 대해 훈련된 모델과 동등한 성능을 보였으며, 합성 데이터를 활용해 도메인-specific VRDU 작업에 효과적으로 적응하는 능력을 입증하였습니다.



### KnobGen: Controlling the Sophistication of Artwork in Sketch-Based Diffusion Models (https://arxiv.org/abs/2410.01595)
- **What's New**: 최근 발생한 전이 모델(Diffusion Models)의 발전은 텍스트 기반 이미지 생성(T2I)에 큰 개선을 가져왔지만, 미세한 정밀함과 높은 수준의 제어 간의 균형을 맞추는 데 어려움을 겪고 있습니다. 본 연구에서는 KnobGen이라는 새로운 이중 경로 프레임워크를 제안하여 스케치 기반 이미지 생성을 민주화하고, 사용자 스킬 수준에 따라 이미지 생성을 적응시킵니다.

- **Technical Details**: KnobGen은 Coarse-Grained Controller(CGC)와 Fine-Grained Controller(FGC) 두 가지 모듈을 사용하여 고급 의미와 세부 정제를 처리합니다. 사용자는 'knob inference' 메커니즘을 통해 두 모듈의 비율을 조정할 수 있어 초보자 스케치와 숙련된 아티스트의 스케치를 모두 유연하게 처리할 수 있습니다.

- **Performance Highlights**: MultiGen-20M 데이터셋과 새롭게 수집한 스케치 데이터셋에서 KnobGen의 효과iveness를 입증하였으며, 사용자의 스케치와 텍스트를 기반으로 최종 결과물의 자연스러운 외형을 유지하면서 강력한 제어를 제공합니다.



### MM-LDM: Multi-Modal Latent Diffusion Model for Sounding Video Generation (https://arxiv.org/abs/2410.01594)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 최근 발표된 연구에서는 Multi-Modal Latent Diffusion Model (MM-LDM)을 도입하여 Sounding Video Generation (SVG)이라는 새로운 오디오-비디오 통합 생성 작업을 해결하고자 합니다. 이 모델은 오디오 및 비디오 데이터를 단일 이미지로 변환하여 표현을 통합합니다.

- **Technical Details**: MM-LDM은 각 모달리티에 대해 저차원 지각 잠재 공간을 구축하고, 이들을 공유하여 고차원 의미적 특성 공간을 만듭니다. 이를 통해 두 모달리티 간 정보 격차를 줄이고, 더 나은 시너지 효과를 제공합니다. Transformer 기반의 자기 주의 메커니즘을 활용하여 단일 모달과 교차 모달 상관관계를 모델링합니다.

- **Performance Highlights**: MM-LDM은 Landscape 및 AIST++ 데이터셋에서 새로운 최첨단 결과를 달성하며, 시각적 및 청각적 품질 모두에서 현저한 향상을 보여줍니다. 샘플링 속도는 10배 빨라지고, 더 큰 배치 사이즈를 허용하여 계산 복잡성을 크게 줄입니다.



### Coordinate-Based Neural Representation Enabling Zero-Shot Learning for 3D Multiparametric Quantitative MRI (https://arxiv.org/abs/2410.01577)
- **What's New**: 본 연구에서는 혁신적인 이미징 방법론인 SUMMIT(SimUltaneous MultiparaMetric quantitative MRI via Implicit neural represenTation)를 제안하고 있습니다. SUMMIT는 3D 다중 매개변수 qMRI를 위한 데이터 수집과 비지도 복원을 포함한 새로운 기술입니다.

- **Technical Details**: SUMMIT는 여러 중요한 양적 물성(parametric properties)을 높은 언샘플링된 k-space에 인코딩하며, 이를 통해 물리 모델과 결합된 암묵적 신경 표현(Implicit Neural Representation)을 활용하여 외부 학습 데이터 세트 없이 원하는 멀티파라메트릭 맵을 복원합니다. 또한, SUMMIT는 T1, T2, T2*, 및 양적 자기 감수성 맵(QSM)을 동시 복원하는 기능을 제공합니다.

- **Performance Highlights**: SUMMIT는 기존 방법에 비해 11.0%, 11.8%, 4.8%의 양적 오차를 줄이며, 잘 제어된 수집 시간 내에 여섯 가지의 고해상도 양적 MR 이미지를 제공합니다. 이로써 SUMMIT는 단일 데이터 수집에서 가장 포괄적인 qMRI 복원을 실현하였습니다.



### Fake It Until You Break It: On the Adversarial Robustness of AI-generated Image Detectors (https://arxiv.org/abs/2410.01574)
- **What's New**: 이 논문은 인공지능(AI)으로 생성된 이미지(AIGI) 탐지기의 최신 기술을 다양한 공격 시나리오에서 평가하며, 현재의 포렌식(Classifiers) 기술이 공격 받을 수 있는 현실적인 설정에서의 취약성을 밝힙니다.

- **Technical Details**: 우리는 포렌식 탐지기와 관련된 두 가지 공격 시나리오(white-box 공격과 black-box 공격)를 연구하였으며, 소셜 미디어와 같은 현실적인 환경에서 진행되는 후처리 과정을 반영하여 실험하였습니다. 최신 탐지기(CNN 및 CLIP 기반)들을 평가하며, 이러한 시스템들이 실제 공격에 얼마나 취약한지를 분석하였습니다.

- **Performance Highlights**: 최신 포렌식 분류기는 공격자의 개입이 없을 때에도 탐지 정확도가 0%까지 감소할 수 있으며, 실제 상황에서 50% 이상의 공격 성공률을 기록했습니다. 이를 통해 현재의 탐지 기술이 갖는 위험을 실질적으로 평가하였고, CLIP 기반 탐지기에 대한 효과적인 방어 메커니즘을 제안하였습니다.



### PASS:Test-Time Prompting to Adapt Styles and Semantic Shapes in Medical Image Segmentation (https://arxiv.org/abs/2410.01573)
Comments:
          Submitted to IEEE TMI

- **What's New**: 이 논문은 Test-time adaptation (TTA) 프레임워크인 PASS를 제안하여 의학적 이미지 세분화에서 발생할 수 있는 도메인 불일치를 해결합니다. 기존 TTA 방법들은 주로 저수준 도메인 변화에 중점을 두었으나, 본 연구에서는 형태의 다양성이 성능 저하에 중요한 요소임을 밝혀냈습니다.

- **Technical Details**: PASS는 두 종류의 프롬프트를 동시에 학습하는 프레임워크로, 입력-공간 프롬프트가 테스트 이미지의 스타일을 재구성하고, 형태 인식 프롬프트가 도메인 간 높은 수준의 형태 차이를 연결합니다. 또한, cross-attention prompt modulator를 도입하여 각 테스트 샘플에 맞춘 형태 프롬프트를 생성합니다.

- **Performance Highlights**: PASS 프레임워크는 여러 의학적 이미지 세분화 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다. 실험 결과에 따르면, 적은 수의 학습 가능한 파라미터로도 다중 센터의 옵틱 디스크 및 MRI 전립선 세분화 작업에서 state-of-the-art 성능을 달성했습니다.



### Boosting Weakly-Supervised Referring Image Segmentation via Progressive Comprehension (https://arxiv.org/abs/2410.01544)
- **What's New**: 이 논문은 약한 지도 학습 기반의 참조 이미지 분할(Weakly-Supervised Referring Image Segmentation, WRIS) 문제를 다루며, 이미지-텍스트 쌍에서 직접적으로 대상 객체의 위치를 학습하는 도전적인 설정에 중점을 두고 있습니다. 기존의 연구를 바탕으로, 우리는 대상 객체를 점진적으로 지역화하기 위해 텍스트 설명에서 관련 구문을 활용하는 새로운 점진적 이해 네트워크(Progressive Comprehension Network, PCNet)를 제안합니다.

- **Technical Details**: PCNet은 입력 텍스트 설명을 작은 구문으로 분해하기 위해 대형 언어 모델(Large Language Model, LLM)을 사용합니다. 이러한 짧은 구문은 대상 관련 단서로 간주되어 조건부 참조 모듈(Conditional Referring Module, CRM)에 단계적으로 제공됩니다. 또한, 시각적 지역화를 단계별로 수행하도록 제어하는 지역 인식 축소(Region-aware Shrinking, RaS) 손실과 중복 지역화 모호성을 줄이는 인스턴스 인식 불명확화(Instance-aware Disambiguation, IaD) 손실을 도입합니다.

- **Performance Highlights**: 우리의 방법은 세 가지 주요 벤치마크에서 기존 방법들보다 더 나은 성능을 보여주며, 약한 감독 신호를 사용하여 데이터 주석 부담을 덜 수 있는 가능성을 제시합니다.



### Edge-preserving noise for diffusion models (https://arxiv.org/abs/2410.01540)
- **What's New**: 이 논문에서는 기존의 균일한 가우시안 노이즈를 사용하는 생성적 확산 모델의 한계를 극복하기 위해, 에지를 보존하는 새로운 확산 모델을 제안합니다. 이 모델은 이미지 처리에서 오래된 비등방 확산(anisotropic diffusion) 기법에서 영감을 받아 구조적 정보를 더욱 잘 반영합니다.

- **Technical Details**: 제안된 모델은 에지 보존 노이즈(Edge-Preserving Noise) 스케줄러를 사용하여 에지를 보존하는 노이즈와 비등방 Gaussian 노이즈 간의 변화를 통해 생성 과정의 수렴 속도를 높입니다. 이를 통해 저주파 및 중주파(low-to-mid frequencies) 정보를 더 효율적으로 학습할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 무조건적인 이미지 생성에서 기존 최첨단 모델들보다 일관되게 우수한 성능을 보이며, 특히 픽셀 공간 확산(Pixel Space Diffusion)에서 FID 점수가 30% 향상되었습니다. 또한, 모양 기반 우선 규칙(shape-based prior)에 기반한 생성 작업에서도 더 뛰어난 품질과 견고성을 나타냅니다.



### Multi-Scale Fusion for Object Representation (https://arxiv.org/abs/2410.01539)
- **What's New**: 이 논문은 Object-Centric Learning (OCL)에서 Variational Autoencoder (VAE) 중간 표현의 지침을 개선하기 위해 Multi-Scale Fusion (MSF)을 제안합니다. MSF는 다양한 크기의 객체들이 VAE의 최적 존 안에 포함되도록 보장하며, 스케일 불변성과 분산을 촉진하기 위한 새로운 방법론을 도입합니다.

- **Technical Details**: 이 논문에서는 이미지 피라미드 기법을 사용하여 여러 스케일에서 중간 표현을 생성하며, inter/intra-scale fusion을 통해 저품질 객체를 높은 품질의 슈퍼픽셀로 보완합니다. 이 방식은 VAE 중간 제안의 차별성을 개선하고 OCL 학습에서 더 나은 지침을 제공합니다.

- **Performance Highlights**: 표준 OCL 벤치마크에서 이 기술은 기존의 주요 방법들, 특히 최신 diffusion 기반 방법들과 비교하여 성능을 개선했습니다. 이 기술의 소스 코드는 보충 자료로 제공됩니다.



### EUFCC-CIR: a Composed Image Retrieval Dataset for GLAM Collections (https://arxiv.org/abs/2410.01536)
Comments:
          ECCV Workshop (AI4DH2024)

- **What's New**: EUFCC-CIR 데이터셋을 소개하며, Composed Image Retrieval (CIR) 작업을 위해 설계되었습니다. 문화유산 컬렉션을 더욱 깊이 있게 탐구할 수 있는 새로운 자원을 제공합니다.

- **Technical Details**: EUFCC-CIR은 EUFCC-340K 데이터셋에 기반하여 180K 이상의 주석이 달린 CIR 삼중 서술어를 포함하고 있습니다. 데이터셋은 훈련, 검증 및 두 개의 테스트 분할로 나뉘어, CIR 모델의 포괄적인 평가를 가능하게 합니다. 특히, 쿼리는 멀티모달(다양한 모드)이며, 입력 이미지와 원하는 속성 조작을 설명하는 짧은 텍스트로 구성됩니다.

- **Performance Highlights**: 유럽 연합의 문화 자료 저장소에서 수집된 이미지를 활용하여, 사용자 경험을 풍부하게 하고 학술 연구를 용이하게 하는 잠재력을 보여주었습니다. 본 연구를 통해, CIR의 유용성을 질적 및 양적으로 분석하고, 이를 통해 문화유산 컬렉션과의 상호작용 방식을 혁신할 수 있는 가능성을 강조하였습니다.



### GaussianBlock: Building Part-Aware Compositional and Editable 3D Scene by Primitives and Gaussians (https://arxiv.org/abs/2410.01535)
- **What's New**: 최근 Neural Radiance Fields (NeRF)와 Gaussian Splatting 기술의 발전으로 인해 3D reconstruction 기술이 높은 충실도를 달성하고 있습니다. 하지만 기존 방법들이 학습한 latent representations는 서로 얽혀 있어서 해석 가능성이 떨어집니다. 본 논문에서는 GaussianBlock이라는 새로운 part-aware compositional reconstruction 방법을 제안하여, 의미론적으로 일관된 다양한 구조로의 편집을 가능하게 합니다.

- **Technical Details**: GaussianBlock은 유연한 액세스성과 편집성을 제공하는 superquadric primitives와 높은 복원 품질을 자랑하는 3D Gaussians의 장점을 결합한 하이브리드 표현을 도입합니다. 이 방법은 Attention-guided Centering (AC) 손실을 활용하여 의미론적으로 일관된 primitives를 reconstruct하며, 동적 분할 및 융합 전략을 통해 compact한 구조를 유지합니다.

- **Performance Highlights**: GaussianBlock은 다양한 벤치마크에 걸쳐 높은 품질로 part-level decomposition과 정밀한 편집 가능성을 보여주며, 특히 DTU, Nerfstudio, BlendedMVS, Mip-360-Garden 및 Tank&Temple-Truck 등에서 경쟁력 있는 충실도를 달성했습니다.



### Toward a Holistic Evaluation of Robustness in CLIP Models (https://arxiv.org/abs/2410.01534)
Comments:
          17 pages, 10 figures, extension of NeurIPS'23 work: A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP). arXiv admin note: text overlap with arXiv:2402.07410

- **What's New**: 이 연구는 CLIP 모델의 분류 강인성을 단순한 정확도를 넘어 다양한 시각적 요인에 대한 저항력을 평가하고, 오분포(out-of-distribution) 탐지 및 예측 불확실성 같은 주요 안전 목표를 분석하였습니다.

- **Technical Details**: 1. 시각적 요인에 대한 강인성: CLIP 모델이 자세, 크기, 색상 등 다양한 요인에 어떻게 반응하는지 분석했습니다.
2. 오분포 탐지: CLIP의 레이블 미포함 샘플 탐지 능력을 평가했습니다.
3. 예측 불확실성: CLIP 모델의 예측이 다양한 테스트 조건에서 어떻게 반영되는지 조사했습니다.
4. 3D 인식: 3D 손상에 대한 CLIP 모델의 저항력을 평가했습니다.
5. 비전-언어 인코더 상호작용: CLIP 모델이 분류 강인성에 미치는 영향을 연구했습니다.

- **Performance Highlights**: CLIP 모델은 높은 분류 강인성을 보였으나, 시각 요인에 따라서는 변동성이 큽니다. 또한, CLIP의 시각 인코더 설계가 3D 손상에 대한 저항성에 중요한 역할을 하며, LLaVA와 같은 비전-언어 모델이 CLIP 보다 더 높은 분류 성능을 보일 수 있음이 확인되었습니다.



### MiraGe: Editable 2D Images using Gaussian Splatting (https://arxiv.org/abs/2410.01521)
- **What's New**: MiraGe 모델은 2D 이미지를 인간의 인식 기준으로 플랫폼화하며, 3D 공간에서의 이미지 조작을 용이하게 합니다. 이 모델은 기존의 GaussianImage와는 달리, 2D 이미지를 3D로 변환하는 새로운 방법을 제공합니다.

- **Technical Details**: MiraGe는 flat-controlled Gaussians를 사용하여 2D 이미지를 3D로 인식하고 조정합니다. 이는 Gaussian Splatting (3DGS) 및 GaMeS parameterization을 통해 이뤄집니다. 측면의 물리 엔진과 결합하여 2D 및 3D 환경에서 물리기반 수정 및 상호작용이 가능합니다.

- **Performance Highlights**: MiraGe는 기존의 모델보다 더 나은 품질과 실제적인 이미지 수정 가능성을 제공합니다. 모델은 실시간으로 작동하며, 고화질 이미지를 복원할 수 있습니다. 여러 가지 환경에서 유용하게 활용될 수 있는 점이 특징입니다.



### UW-GS: Distractor-Aware 3D Gaussian Splatting for Enhanced Underwater Scene Reconstruction (https://arxiv.org/abs/2410.01517)
- **What's New**: UW-GS는 수중 장면 표현을 위해 특별히 설계된 3D Gaussian Splatting 기반 방법으로, 기존 기술의 한계를 극복하고 수중 환경에서의 색상 변화를 모델링하는 새로운 방법을 도입하였습니다.

- **Technical Details**: UW-GS는 거리 의존 색상 변화를 모델링하는 색상 외관(color appearance) 모델을 도입하고 물리 기반 밀도 제어 전략을 사용하여 멀리 있는 객체의 클리어리티를 향상시킵니다. 이 방법은 이진 모션 마스크(binary motion mask)를 사용하여 동적 콘텐츠를 처리합니다. 또한, 새로운 손실 함수(loss function)와 pseudo-depth maps를 최적화하여 산란 매질에서의 효과적인 리프레젠테이션을 제공합니다.

- **Performance Highlights**: UW-GS는 PSNR(peak signal-to-noise ratio)에서 최대 1.26dB의 향상을 보이며, 새로운 수중 데이터셋(S-UW)을 사용해 효과성을 검증하였습니다.



### LEGO: Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion (https://arxiv.org/abs/2410.01506)
Comments:
          Research paper

- **What's New**: 이 논문에서는 컴퓨터 비전 작업에서 서로 다른 표현, 도메인, 모드(모달리티)의 특징(feature)을 효과적으로 결합하는 새로운 접근법을 제안합니다. 저자들은 고차원 특징 공간에서 저차원 해석 가능한 그래프 공간으로 전환하여 다양한 수준의 특징 관계를 인코딩하는 유사성 그래프(similarity graphs)를 구성하였습니다.

- **Technical Details**: 제안된 방법은 그래프 제곱(graph power) 확장을 활용하고, 이러한 그래프 파워를 결합하기 위해 학습 가능한 그래프 융합 연산자(learnable graph fusion operator)를 도입합니다. 이 방법은 관계 중심(relationship-centric)으로 작동하며, 수학적으로 원리에 부합하며, 멀티선형 다항식(multilinear polynomials)을 통해 원소별 유사도 점수(element-wise similarity score)를 집계하는 방식으로 유사합니다.

- **Performance Highlights**: 이 논문에서 제안하는 그래프 기반 융합 방법은 비디오 이상 탐지(video anomaly detection) 작업에서 강력한 성능을 보여주며, 다중 표현(multi-representational), 다중 모달(multi-modal), 다중 도메인(multi-domain) 특징 융합 작업에서 효과적임을 입증하였습니다.



### Quo Vadis RankList-based System in Face Recognition? (https://arxiv.org/abs/2410.01498)
Comments:
          Accepted for presentation at IJCB 2024

- **What's New**: 최근 몇 년간의 얼굴 인식 기술의 발전을 통해 RankList 기반 방법론의 변혁이 이루어졌습니다. 특히, DaliFace 네트워크의 로짓(logits)을 활용한 Logit-Cohort Selection(LoCoS) 방법이 기존의 Cohort 비즈니스 방식을 대체하며 성능을 크게 향상시킬 수 있음을 확인했습니다.

- **Technical Details**: 본 논문에서는 얼굴 인식을 위해 기존의 RankList 기반 방법론을 재검토하고, DaliFace 네트워크에서 추출한 로짓을 사용하는 방법으로 확장하였습니다. Logit-Cohort Selection 알고리즘(LoCoS)을 제안해, 고품질의 등록 이미지와 저품질의 프로브 이미지를 비교하는 과정에서 발생하는 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, 두 개의 어려운 얼굴 인식 데이터셋에서 LoCoS 방법은 기존의 RankList 기반 방법론보다 성능이 크게 향상된 것으로 나타났습니다. 이로써 서로 다른 이미지 품질을 처리하는 향후 연구 방향을 제시하며, LoCoS 접근법은 고품질과 고품질 얼굴 간 비교에서 완벽한 인식 성능을 유지할 수 있음을 보여주었습니다.



### SinkSAM: A Monocular Depth-Guided SAM Framework for Automatic Sinkhole Segmentation (https://arxiv.org/abs/2410.01473)
Comments:
          14 pages, 14 figures

- **What's New**: 이 논문에서는 새로운 형태의 sinkhole 세분화(separation) 방법론인 SinkSAM을 소개합니다. 이 모델은 전통적인 topographic computation과 지정된 데이터를 기반으로 한 Segment Anything Model (SAM)을 결합하여 불규칙한 모양을 가진 sinkhole을 더욱 정확하게 분리합니다.

- **Technical Details**: SinkSAM은 (1) SAM을 통해 픽셀 수준에서 sinkhole 경계를 세분화하며, (2) 수학적 prompting 전략을 사용하여 제한된 학습 기반 모델의 한계를 극복하고, (3) Depth Anything V2 monocular depth를 이용하여 LiDAR 데이터 의존성을 없애고, (4) established sinkhole database로 SAM을 fine-tuning하여 성능을 향상시킵니다.

- **Performance Highlights**: SinkSAM은 숨겨진 테스트 지역인 반건조 지역에서 40.27%의 Intersection-over-Union (IoU) 성능을 달성하여 기존 결과를 초월하였습니다. 이는 RGB 이미지를 단일로 사용하여 sinkhole을 mapping 하는 데 성공함을 보여줍니다.



### Decorrelation-based Self-Supervised Visual Representation Learning for Writer Identification (https://arxiv.org/abs/2410.01441)
- **What's New**: 이 연구는 작가 식별(writer identification) 작업을 위한 자기 지도 학습(self-supervised learning, SSL) 프레임워크를 제안한 최초의 연구이다. 기존의 서명 검증(signature verification)용으로 제안된 decorrelation 기반의 SWIS 프레임워크를 수정하여 사용하였다.

- **Technical Details**: SWIS(Stroke-Wise Identification System)는 각 차원의 특성을 표준화하여 수정된 decorrelation 기반 프레임워크이다. 이 방법의 주요 목표는 손글씨 이미지를 사용하여 각 작가의 고유한 스트로크(stroke) 특징을 학습하고 식별할 수 있도록 하는 것이다. 또한 출력 차원들이 결합(multivariate) 분포에 속하게 하여 차원 분리(disentanglement)를 보장한다.

- **Performance Highlights**: 제안된 프레임워크는 작가 식별 벤치마크에서 기존의 SSL 프레임워크와 여러 감독(supervised) 방법들을 능가하는 성능을 보여주었다.



### EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Camera Settings (https://arxiv.org/abs/2410.01425)
- **What's New**: 본 논문은 EVA-Gaussian이라는 새로운 3D 인간 시점 합성 파이프라인을 제안하여 다양한 카메라 설정에서의 실시간 재구성을 지원합니다. 기존 방법들의 제약을 해소하고, 특히 드문 카메라 관점에서도 유연성을 제공합니다.

- **Technical Details**: EVA-Gaussian은 먼저 Efficient cross-View Attention (EVA) 모듈을 도입하여 원본 이미지에서 각 3D Gaussian의 위치를 추정합니다. 이 후, 추정된 Gaussian 위치 맵과 원본 이미지를 결합하여 3D Gaussians의 속성과 특성 임베딩을 예측합니다. 또한, 지오메트릭 위치 추정 오류로 인한 아티팩트를 수정하기 위해 반복적 특성 세련기(Feature Refiner)를 사용합니다.

- **Performance Highlights**: THuman2.0 및 THumansit 데이터셋에서 실험한 결과, EVA-Gaussian은 다양한 카메라 설정에서 렌더링 품질이 뛰어나고 실시간 재구성 및 렌더링을 가능하게 합니다. 특히, 드문 카메라 설정에서도 일반화가 잘 됩니다.



### The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs (https://arxiv.org/abs/2410.01417)
- **What's New**: 새로운 다중 모달 대형 언어 모델(MLLMs) 벤치마크 제안. 인간의 기본적인 연관 능력을 평가하기 위한 평가 기준 및 주목받지 못한 연관 임무 개발.

- **Technical Details**: 연관 작업을 정의하고 자연어 데이터셋을 활용한 annotation-free (주석 없는) 방법으로 연관 벤치마크 구축. 세 가지 수준의 연관 작업(단일 단계, 동기식, 비동기식) 설정.

- **Performance Highlights**: 현재 공개 소스 MLLMs는 연관 작업에서 인간과 비교해 일관되게 낮은 성능을 보임. 최고 성능의 닫힌 소스 모델도 인간 성능과 큰 차이 존재.



### SHAP-CAT: A interpretable multi-modal framework enhancing WSI classification via virtual staining and shapley-value-based multimodal fusion (https://arxiv.org/abs/2410.01408)
- **What's New**: 본 논문에서는 SHAP-CAT이라는 새로운 해석 가능한 다중 모달(multi-modal) 프레임워크를 제안합니다. 이 프레임워크는 Shapley 값 기반의 차원 축소 기법을 사용하여 H&E 및 IHC 이미지의 통합을 효과적으로 수행합니다.

- **Technical Details**: SHAP-CAT 프레임워크는 가상 염색(virtual staining) 기법을 적용하여 한 가지 새로운 임상 관련 모달리티를 생성하고, 가벼운 bag-level 표현을 추출합니다. 각 차원에 대한 중요성 값을 계산하여 모델 출력에 미치는 영향을 분석하고, 이를 통해 상위 32개 중요한 차원만을 선택하여 최종 분류기에 사용합니다.

- **Performance Highlights**: SHAP-CAT 프레임워크는 합성 모달리티를 통합하여 BCI dataset에서 5%, IHC4BC-ER에서 8%, IHC4BC-PR에서 11%의 정확도 향상을 보여주었습니다.



### AgriCLIP: Adapting CLIP for Agriculture and Livestock via Domain-Specialized Cross-Model Alignmen (https://arxiv.org/abs/2410.01407)
- **What's New**: 최근 농업 및 축산 분야에 최적화된 대규모 이미지-텍스트 데이터셋(ALive)과 함께, 비전-언어 기반의 프레임워크(AgriCLIP)를 소개합니다. 이는 기존의 CLIP 모델을 농업 및 축산 도메인에 맞추어 강화합니다.

- **Technical Details**: AgriCLIP은 이미지-텍스트 대비 학습(contrastive learning)과 자기 지도 학습(self-supervised learning)을 통합한.training pipeline을 사용하여 글로벌 의미(global semantic)와 세부 사항(fine-grained) 학습을 가능하게 합니다. ALive 데이터셋에는 약 600,000개의 이미지-텍스트 쌍이 포함되어 있으며, 이는 다양한 농작물과 가축을 포괄하고 있습니다.

- **Performance Highlights**: 다양한 20개의 다운스트림 작업에서 실험한 결과, AgriCLIP은 평균적으로 7.8%의 제로샷(zero-shot) 분류 정확도 향상을 기록하였습니다. ALive 데이터셋과 코드가 공개되어 있으며, 향후 연구에 기여할 것으로 기대됩니다.



### Gaussian-Det: Learning Closed-Surface Gaussians for 3D Object Detection (https://arxiv.org/abs/2410.01404)
- **What's New**: Gaussian-Det는 다중 보기 기반 3D 객체 탐지를 위해 Gaussian Splatting을 활용하며, 기존 Monocular 및 NeRF 기반 방법과 달리 연속적인 표면을 통한 객체 표현을 제안합니다.

- **Technical Details**: Gaussian-Det는.partial surfaces에 대한 피처 설명자로 Gaussians을 형성하여 연속적인 방식으로 객체를 모델링합니다. Closure Inferring Module (CIM)은 Gaussian Splatting의 불확실성을 고려하여 전반적인 객체 전체의 표면 폐쇄성을 측정하여 객체성(objectness) 예측의 품질과 신뢰성에 대한 선험적 정보(prior)를 제공합니다.

- **Performance Highlights**: Gaussian-Det는 Synthetic 3D-FRONT 및 진짜 ScanNet 데이터셋에서 다양한 기존 접근 방법에 비해 평균 정밀도(average precision)와 재현율(recall) 모두에서 뛰어난 성과를 보여줍니다.



### Signal Adversarial Examples Generation for Signal Detection Network via White-Box Attack (https://arxiv.org/abs/2410.01393)
Comments:
          18 pages, 6 figures, submitted to Mobile Networks and Applications

- **What's New**: 본 논문에서는 신호 감지 네트워크를 위한 신호 적대적 예제(adversarial examples) 생성 모델을 제안합니다. 이는 신호에 작은 변동(perturbations)을 추가하는 관점에서 정의됩니다.

- **Technical Details**: 이 모델은 시간 주파수 도메인과 시간 도메인 사이의 L2-노름(L2-norm) 불평등 관계를 활용하여 신호 변동의 에너지를 제어합니다. 실험적으로, 신호 변동 에너지 비율이 3% 미만일 때 발생한 적대적 공격은 평균 정밀도(mAP)를 28.1% 감소시키는 결과를 보였습니다.

- **Performance Highlights**: 적대적 공격은 무작위 잡음(interference)으로 인한 효과보다 더 크며, 신호 감지 네트워크의 정밀도(precision)는 30.4% 감소했습니다. 이는 신호 감지 네트워크가 신호 적대적 예제에 민감하다는 것을 입증합니다.



### Quantifying Cancer Likeness: A Statistical Approach for Pathological Image Diagnosis (https://arxiv.org/abs/2410.01391)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문에서는 병리 이미지에서 암 영역을 자동으로 식별하기 위한 새로운 통계적 접근 방식을 제안합니다. 제안된 방법은 증거 기반 의학에 따라 구성되며, 영상 특징의 분류 정보와 이들의 공간 분포를 결정하는 계산 기술을 포함합니다.

- **Technical Details**: 제안된 방법은 암 지역과 정상 지역을 나타내는 이진 값으로 구성된 분류 공간을 정의하고, 병리 이미지 내의 지역 이미지 특징의 분포를 분석합니다. Kullback-Leibler divergence와 같은 정보를 측정하는 기법을 사용하여 의미 있는 이미지 특징을 구분합니다. 이 방법은 기존의 경계선 설정 없이도 효과적인 암 분류를 가능하게 합니다.

- **Performance Highlights**: 이 방법은 암 분류 작업에서 AUC(Area Under Curve) 점수가 0.95 이상을 기록하였으며, 병리학자들이 수고스럽게 합의를 이루는 전통적인 작업의 필요성을 줄여줍니다.



### Learning Physics From Video: Unsupervised Physical Parameter Estimation for Continuous Dynamical Systems (https://arxiv.org/abs/2410.01376)
- **What's New**: 이 연구는 동역학 시스템의 물리적 매개변수를 비디오에서 자동으로 추출하는 새로운 방법을 제안합니다. 기존의 기법들은 레이블이 있는 대량의 데이터셋이 필요하지만, 본 연구는 레이블 없이도 동적 시스템의 물리적 매개변수를 추정할 수 있는 혁신적인 접근을 제공합니다.

- **Technical Details**: 기존 방법들은 프레임 예측(frame prediction)에 의존하여 영상의 프레임을 재구성하는 복잡한 구조를 갖고 있지만, 본 연구의 방법은 간단한 인코더와 물리 블록만 포함된 구조를 취합니다. KL-divergence 기반의 손실 함수를 사용하여 잠재 공간(latent space)에서 직접 학습을 진행하며, 프레임 재구성의 필요성을 없앱니다.

- **Performance Highlights**: 제안된 방법은 다양한 동적 시스템에 적용 가능하며, 초기 조건에 대한 견고성을 보여줍니다. 잘 정의된 손실 함수를 통해 비디오로부터의 정확한 동적 상태 추정 및 테스트 시의 정밀한 외삽(extrapolation)을 가능하게 합니다.



### Harnessing the Latent Diffusion Model for Training-Free Image Style Transfer (https://arxiv.org/abs/2410.01366)
- **What's New**: 최근의 확산 모델(Diffusion models)은 고품질 이미지를 생성하는 능력을 보여주고 있지만, 생성 과정 제어는 여전히 도전 과제로 남아있습니다. 본 논문에서는 추가적인 훈련 없이 스타일 전이(style transfer)를 수행할 수 있는 STRDP(Style Tracking Reverse Diffusion Process) 알고리즘을 제안합니다.

- **Technical Details**: 제안하는 STRDP 알고리즘은 사전 훈련된 Latent Diffusion Model(LDM)의 역 확산 과정 중에 스타일 이미지의 인코딩 히스토리를 추적하며 Adaptive Instance Normalization(AdaIN) 함수를 독창적으로 적용합니다. 이 알고리즘은 LDM의 잠재 공간(latent space)에서 스타일 전이를 가능하게 하며, 다양한 LDM 모델과의 호환성을 제공합니다.

- **Performance Highlights**: 실험을 통해 제안하는 방법이 추가 훈련 없이 이미지를 빠르게 스타일 전이할 수 있음을 보여주었습니다. 이 알고리즘은 스타일 전이와 원본 이미지의 색상 보존(color preservation)을 동시에 달성하며, 다른 확산 기반 방법들보다 더 빠른 속도를 자랑합니다.



### High-quality Animatable Eyelid Shapes from Lightweight Captures (https://arxiv.org/abs/2410.01360)
Comments:
          Accepted by SIGGRAPH Asia 2024

- **What's New**: 이 논문에서는 모바일 폰으로 촬영한 RGB 비디오를 사용하여 고품질의 눈꺼풀 재구성과 애니메이션을 실현하는 새로운 방법을 제안합니다. 이는 이전의 복잡한 카메라 설정이나 고비용 시스템 없이 가능하게 합니다.

- **Technical Details**: 제안된 방법은 정적 및 동적 안구 정보를 활용하여 눈꺼풀 재구성을 지원합니다. 자동 안구 보정 방법을 통해 필요한 안구 매개변수를 얻고, 신경망 기반의 눈꺼풀 제어 모듈을 개발하여 의미 있는 애니메이션 제어를 가능하게 합니다. 동적 신경 SDF(Field) 필드를 학습하여 눈꺼풀의 동작을 모델링하며, 접촉 손실(contact loss)을 통해 눈꺼풀이 안구 표면에 밀착되도록 합니다.

- **Performance Highlights**: 광범위한 합성 및 실제 데이터에 대한 실험을 통해, 제안된 방법은 동일한 수준의 캡처 설정을 기반으로 한 이전 방법보다 더 세밀하고 사실적인 결과를 제공합니다.



### Cognition Transferring and Decoupling for Text-supervised Egocentric Semantic Segmentation (https://arxiv.org/abs/2410.01341)
- **What's New**: 본 논문에서는 새로운 Text-supervised Egocentric Semantic Segmentation (TESS) 작업을 제안합니다. 이 작업은 이미지 레벨 레이블에서 텍스트로 약하게 감독된 egocentric 이미지에 대한 픽셀 수준 카테고리를 할당하는 것을 목표로 합니다.

- **Technical Details**: 제안된 Cognition Transferring and Decoupling Network (CTDN)은 이미지와 텍스트 간의 상관관계를 통해 egocentric wearer-object 관계를 학습하며, Cognition Transferring Module (CTM)을 통해 대규모 사전 학습된 모델의 인지 지식을 전이합니다. Fourground-background Decoupling Module (FDM)은 시각 표현을 분리하여 전방 및 후방 영역을 명확히 구별합니다.

- **Performance Highlights**: 제안된 방법은 TESS 벤치마크에서 광범위한 실험을 통해 효과성을 입증하였으며, 최근의 관련 방법들보다 큰 차이로 성능을 초과했습니다.



### VectorGraphNET: Graph Attention Networks for Accurate Segmentation of Complex Technical Drawings (https://arxiv.org/abs/2410.01336)
Comments:
          27 pages, 13 figures

- **What's New**: 본 논문은 PDF 형식의 기술 도면에서 벡터 데이터를 추출 및 분석하는 새로운 접근 방법을 제안합니다. PDF 파일을 SVG 형식으로 변환하여 벡터 개체 간의 관계를 기하학적 정보를 사용하여 포착하는 기능이 풍부한 그래프 표현을 생성합니다.

- **Technical Details**: 그래프 주목 변환기(Graph Attention Transformer)와 계층적 라벨 정의를 적용하여 정확한 선 단위 세분화를 달성합니다. 제안된 방법은 FloorplanCAD 데이터셋을 포함한 두 개의 데이터셋에서 평가하고, 가중치 F1 점수에서 기존 방법을 초월하여 최첨단 결과를 달성하였습니다. 이 연구에서는 기하학적 데이터 추출과 그래프 특징으로의 변환을 통해 그래프 신경망(GNN)을 활용하여 벡터 간의 관계를 학습하고 세멘틱 정보 복원을 수행하였습니다.

- **Performance Highlights**: 제안된 벡터 기반 방법은 기존 비전 기반 접근 방식에 비해 대규모 기술 도면 분석을 위한 더 확장 가능한 솔루션을 제공하며, 현재의 최첨단 벡터 기반 기술 보다 현저히 적은 GPU 전력을 요구합니다. 이 접근법은 AEC 산업에서 기술 도면으로부터 의미 있는 정보를 추출하여 새로운 응용 프로그램을 가능하게 하고 기존 작업 흐름을 개선하는 데 효과적임을 증명하였습니다.



### Finetuning Pre-trained Model with Limited Data for LiDAR-based 3D Object Detection by Bridging Domain Gaps (https://arxiv.org/abs/2410.01319)
Comments:
          Accepted in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024

- **What's New**: 이 논문에서는 Domain Adaptive Distill-Tuning (DADT)이라는 새로운 방법을 제안하여, 사전 훈련된 모델을 적은 양의 타겟 데이터(약 100개의 LiDAR 프레임)를 사용하여 적응시키는 방법을 다루고 있습니다. DADT는 객체 수준(object-level)과 контекст 수준(context-level) 표현을 정렬하는 정규화를 사용하여, 모델의 표현력(representation power)을 유지하면서 과적합(overfitting)을 방지합니다.

- **Technical Details**: DADT는 교사-학생 아키텍처(teacher-student architecture)를 기반으로 하며, 포기된 사전 훈련된 백본(backbone)을 사용하여 밀도 정렬(LiDAR 입력의 밀도를 맞춰주는)된 LiDAR 입력에 대해 동작하는 교사 네트워크와, 원래 밀도 비정렬(non-aligned) LiDAR 입력으로 백본을 훈련하는 학생 네트워크로 구성됩니다.

- **Performance Highlights**: Waymo Open 데이터셋과 KITTI를 포함한 주행 벤치마크를 통해 실행된 실험 결과, 제안된 방법이 적은 타겟 데이터로 사전 훈련된 모델을 효과적으로 미세 조정(finetune)하여 정확도(accuracy)에서 상당한 향상을 이끌어낸 것으로 확인되었습니다.



### Deep learning for action spotting in association football videos (https://arxiv.org/abs/2410.01304)
Comments:
          31 pages, 2 figures, 5 tables

- **What's New**: 이 논문은 SoccerNet이라는 축구 비디오 이해를 위한 데이터셋을 통해 액션 스포팅(action spotting)이라는 새로운 작업을 소개합니다. 2018년 이전에는 스포츠에서 액션 스포팅을 위한 대규모 데이터셋이 공개되지 않아 이 방법의 벤치마킹이 어려웠으나, SoccerNet은 이를 해결하기 위한 솔루션을 마련했습니다.

- **Technical Details**: SoccerNet Action Spotting 데이터셋은 550개 이상의 전체 방송 경기가 포함되어 있으며, 각 게임에서 발생할 수 있는 거의 모든 종류의 액션에 대해 수작업으로 주석이 달려 있습니다. 이 데이터셋은 자동 액션 스포팅 방법을 개발하기 위해 설계되었으며, 딥러닝 접근 방식을 포함하여 많은 연구를 가능하게 합니다.

- **Performance Highlights**: SoccerNet 프로젝트를 통해 지난 5년간 60개 이상의 액션 스포팅 방법이 개발 또는 발표되었습니다. 이는 스포츠 산업에서 액션 스포팅을 실현 가능한 옵션으로 만드는데 기여하였습니다. 매년 개최되는 챌린지는 전 세계 연구자들이 이 분야에서 최신 성과를 달성하기 위해 경쟁하는 장이 되고 있습니다.



### LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion (https://arxiv.org/abs/2410.01295)
Comments:
          For more information: this https URL

- **What's New**: 이 논문은 3D 모델을 고도로 압축된 잠재 공간(latent space)으로 매핑하는 새로운 계층적 오토인코더(hierarchical autoencoder)를 소개합니다. 이 오토인코더는 대규모 데이터셋과 확산(diffusion) 생성 모델링에서 발생하는 문제를 해결하기 위해 특별히 설계되었습니다. 이전의 접근 방식과 달리, 이 모델은 무작위로 배열된 벡터 집합을 처리합니다.

- **Technical Details**: 새로운 아키텍처는 기존 VecSet의 표현력을 개선하기 위해 잠재 벡터(latent vectors)의 수를 늘리고 훈련 데이터셋의 규모를 확장할 수 있도록 설계되었습니다. 이 모델은 빨라진 학습 시간(0.70배)과 메모리 소비(0.58배 적음)를 자랑하며, 고해상도 기하학적 세부 정보를 충실하게 표현할 수 있습니다. 또한, 카세이드(diffusion) 생성 모델을 사용하여 각 단계에서 이전 단계의 생성을 조건으로 사용하는 것이 가능합니다.

- **Performance Highlights**: 제안된 계층적 오토인코더는 다양한 3D 모델을 표현할 수 있으며, 높은 품질의 기하학을 보존하면서 대규모 데이터셋에 대해 훈련할 수 있는 능력을 가지고 있습니다. 이러한 접근 방식은 모델의 세부 수준을 제어 가능하게 하여, 생성하는 기하 구조의 디테일을 조정할 수 있는 이점을 제공합니다.



### SurgeoNet: Realtime 3D Pose Estimation of Articulated Surgical Instruments from Stereo Images using a Synthetically-trained Network (https://arxiv.org/abs/2410.01293)
- **What's New**: 최근 Mixed Reality (MR) 환경에서 수술 모니터링의 중요성이 크게 부각되면서, 수술 도구와 손의 추적에 대한 연구가 진행되고 있습니다. 이 논문에서는 SurgeoNet이라는 실시간 신경망 파이프라인을 소개합니다.

- **Technical Details**: SurgeoNet은 스테레오 VR 뷰에서 수술 도구를 정확하게 감지하고 추적하기 위해 다단계 접근 방식을 채택하고 있습니다. 이 설계는 YOLO 및 Transformers와 같은 최신 신경망 아키텍처에서 영감을 받았습니다. SurgeoNet은 복잡한 실제 시나리오에서도 일반화 능력을 보여줍니다.

- **Performance Highlights**: SurgeoNet은 합성 데이터(즉, synthetic data)로만 학습하여 높은 성능을 달성하고 있으며, 어떤 새로운 다관절 수술 도구 세트에 대해서도 쉽게 확장할 수 있습니다. SurgeoNet의 코드와 데이터는 공개되어 있습니다.



### Panopticus: Omnidirectional 3D Object Detection on Resource-constrained Edge Devices (https://arxiv.org/abs/2410.01270)
Comments:
          Published at MobiCom 2024

- **What's New**: 본 연구에서는 카메라 기반의 3D 물체 탐지를 위한 Panopticus 시스템을 제안합니다. 이 시스템은 엣지 기기에서 자원 제약을 고려하여 동적인 공간 특성에 맞춰 모델 아키텍처와 운영을 조정하여 정확성을 최적화합니다.

- **Technical Details**: Panopticus는 적응형 다중 분기 탐지 스킴(adaptive multi-branch detection scheme)을 사용하여, 각 카메라 뷰에 대해 다르게 설정된 탐지 능력을 통해 효율성을 극대화합니다. 모델은 다양한 탐지 구성(configurations)을 통해 경량화된 추론(inference) 또는 고성능 추론을 적용하여 지연 시간(latency) 요구 사항을 충족합니다.

- **Performance Highlights**: Panopticus는 엄격한 33ms의 지연 목표 하에서 평균 62%의 정확성 향상을 달성하였으며, 기본 모델에 비해 평균 2.1배의 지연 시간 단축을 기록하였습니다.



### Backdooring Vision-Language Models with Out-Of-Distribution Data (https://arxiv.org/abs/2410.01264)
- **What's New**: 이번 연구에서는 Out-Of-Distribution (OOD) 데이터를 이용하여 Vision-Language Models (VLMs)에 대한 백도어 공격을 최초로 탐구하였습니다. 이를 통해 VLOOD라는 새로운 접근 방식을 제안하며, 기존의 공격 방법들과는 다른 실용적인 시나리오를 다룹니다.

- **Technical Details**: VLOOD는 Clean Knowledge Preservation (CKP), Conceptual Consistency Preservation (CCP), 그리고 동적으로 조정되는 가중치로 구성됩니다. CKP는 OOD 데이터로 훈련하면서도 모델의 정상 동작을 유지하도록 돕고, CCP는 주입된 백도어의 의미 일관성을 보장하며, 동적으로 조정되는 가중치 메커니즘은 깨끗한 데이터와 오염된 데이터 사이의 강조를 조정합니다.

- **Performance Highlights**: VLOOD는 이미지 캡셔닝 및 시각 질문 응답(VQA) 작업에 대한 평가에서, OOD 데이터로 훈련하더라도 개념적 일관성 유지를 기존 기준선보다 유의미하게 향상시키면서 높은 공격 성공률을 보여주었습니다.



### Aggregation of Multi Diffusion Models for Enhancing Learned Representations (https://arxiv.org/abs/2410.01262)
- **What's New**: 이번 논문은 다중 확산 모델(Aggregation of Multi Diffusion Models, AMDM) 알고리즘을 소개합니다. 이 알고리즘은 여러 개의 확산 모델로부터 특징을 합성하여 특정 모델의 표현을 풍부하게 하여 보다 세밀한 조정이 가능하게 합니다.

- **Technical Details**: AMDM은 구형 집합(spherical aggregation)과 다양체 최적화(manifold optimization)라는 두 가지 주요 구성 요소로 구성됩니다. 구형 집합은 다양한 확산 모델로부터의 중간 변수들을 최소한의 다양체 편차로 결합하고, 다양체 최적화는 이러한 변수들을 중간 데이터 다양체와 정렬하도록 개선하여 샘플링 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, AMDM은 추가적인 교육(training)이나 추론(inference) 시간 없이 세밀한 제어를 크게 향상시킵니다. 또한, 확산 모델은 초기 단계에서 위치, 속성 및 스타일과 같은 특징에 초점을 맞추고, 후속 단계에서는 생성 품질과 일관성을 개선함을 밝혀냅니다.



### OCC-MLLM:Empowering Multimodal Large Language Model For the Understanding of Occluded Objects (https://arxiv.org/abs/2410.01261)
Comments:
          Accepted by CVPR 2024 T4V Workshop (5 pages, 3 figures, 2 tables)

- **What's New**: 이 논문에서는 기존의 비주얼-언어 멀티모달 모델에서 조사되지 않은 occluded objects(가려진 객체)에 대한 이해의 격차를 해소하기 위해 OCC-MLLM이라는 새로운 모델을 제안합니다. 또한, 가려진 객체를 이해하기 위한 대규모 이미지-텍스트 쌍 데이터셋을 소개합니다.

- **Technical Details**: OCC-MLLM은 RGB 이미지의 가려진 객체를 이해하기 위해 설계된 비주얼 언어 모델입니다. 이 모델은 기존의 CLIP 모델과 새로운 3D 모델을 결합한 비주얼 인코더 모듈을 포함합니다. 본 연구에서 제안된 방법은 세 가지 과정으로 구성됩니다: 입력 공식, 모델 전달, 디코딩.

- **Performance Highlights**: 이 모델은 최신 멀티모달 모델과 비교하는 실험을 시작하며, 그 성능 개선을 목표로 하고 있습니다. 특히, 가려진 객체에 대한 설명의 정확성을 높이는 데 중점을 두고 있으며, 가려진 객체의 이해에서 기존 모델의 한계를 극복할 것으로 기대됩니다.



### Facial Action Unit Detection by Adaptively Constraining Self-Attention and Causally Deconfounding Samp (https://arxiv.org/abs/2410.01251)
Comments:
          This paper is accepted by International Journal of Computer Vision

- **What's New**: 본 논문에서는 얼굴 행동 단위(AU) 탐지를 위한 새로운 프레임워크인 AC2D를 제안합니다. 기존의 방법들이 AU 탐지에 대한 가이드로만 자기 주의(self-attention)를 학습했던 것과는 달리, AC2D는 주의 가중치 분포를 적응적으로 제약하고 샘플 혼란인자를 인과적으로 분리합니다.

- **Technical Details**: AC2D는 ResTv2를 백본으로 사용하며, 각 AU에 대해 특정 브랜치를 통해 독립적인 AU 특징을 추출합니다. 이는 스케일된 점곱 주의(scaled dot-product attention)를 다채널의 공간적 주의(spatial attention)로 변형하여 AU 위치의 사전 정의된 주의 맵에 가까워지도록 장려합니다. 또한, 인과적 개입(causal intervention) 모듈을 통해 훈련 샘플로 인해 발생하는 편향을 제거합니다.

- **Performance Highlights**: AC2D는 BP4D, DISFA, GFT, BP4D+와 같은 어려운 벤치마크에서 기존 최첨단 AU 탐지 방법들과 비교했을 때 경쟁력 있는 성능을 달성하였습니다. 이 방법은 제약된 시나리오와 비제약 시나리오 모두에서 뛰어난 결과를 보였습니다.



### Replacement Learning: Training Vision Tasks with Fewer Learnable Parameters (https://arxiv.org/abs/2410.01239)
- **What's New**: 본 논문은 Replacement Learning이라는 혁신적인 학습 방법을 제안합니다. 이 방법은 고정된 층(frozen layers)의 모든 파라미터를 두 개의 학습 가능한 파라미터로 완전히 대체하여, 불필요한 파라미터 중복(parameter redundancy) 및 자원 비효율성을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: Replacement Learning은 특정 층의 파라미터를 선택적으로 동결시키고, 이러한 동결된 층이 인접한 층으로부터 가져온 파라미터를 사용하여 업데이트됩니다. 이는 두 개의 학습 가능한 파라미터에 의해 제어되는 파라미터 통합 메커니즘을 통해 이루어집니다. 이러한 방식은 인접 구조로부터 정보를 활용함으로써 계산량을 줄이고, GPU 메모리를 절약하며, 기존 입력과 새로운 입력 간의 균형을 유지합니다.

- **Performance Highlights**: 다양한 아키텍처(CNNs 및 ViTs)를 활용하여 CIFAR-10, STL-10, SVHN 및 ImageNet 등 네 가지 기준 데이터셋에서 실험을 실시한 결과, Replacement Learning은 파라미터 수와 훈련 시간을 줄이고 메모리 소비를 감소시키는 동시에 기존의 end-to-end 훈련 방식을 뛰어넘는 성능을 보여주었습니다.



### Towards Native Generative Model for 3D Head Avatar (https://arxiv.org/abs/2410.01226)
- **What's New**: 본 논문은 제한된 3D 데이터셋에서 360도 렌더링 가능한 3D 머리 모델을 생성하기 위해 새로운 접근 방식을 제안하고 있습니다. 특히, 외형, 형상 및 모션을 분리하여 고품질의 3D 생성 모델을 학습하는 방법에 대해 탐구합니다.

- **Technical Details**: 제안하는 모델은 다양한 표현 모델(3D Morphable Model, Neural Radiance Fields 등)을 활용하여 360도 렌더링 가능한 3D 머리 유형을 생성합니다. 이 과정에서 외형, 형상, 표정을 파라메트릭 공간에서 분리하여 처리함으로써 데이터의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 종합적인 실험 결과, 제안된 모델은 SynHead100 데이터셋에서 3D 생성 정확도 및 렌더링 품질에 있어서 최신 기술의 성능을 초월하는 것으로 나타났습니다. 이는 3D 머리 생성 모델의 새로운 기준이 될 것입니다.



### Perceptual Piercing: Human Visual Cue-based Object Detection in Low Visibility Conditions (https://arxiv.org/abs/2410.01225)
- **What's New**: 이번 연구는 대기 산란(atmospheric scattering)과 인간 시각 피질(human visual cortex)의 메커니즘에서 영감을 받아, 안개, 연기, 안개 등 가시성이 떨어지는 상황에서 객체 탐지를 향상시키기 위한 새로운 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 선택적 주의(selective attention)와 환경 적응성(environmental adaptability)을 통합하여 객체 탐지의 정확성과 효율성을 증대시키도록 설계되었습니다.

- **Technical Details**: 제안된 방법론은 경량의 객체 탐지 모델을 사용하여 관심 지역을 식별한 후, 공간적 주의(spatial attention)를 활용하여 디헤이징(dehazing) 과정을 수행하고, 이후 보다 강력한 탐지 모델로 나아가는 구조로 구성되어 있습니다. 전체 시스템은 mean Average Precision (mAP), Structural Similarity Index Measure (SSIM), Peak Signal-to-Noise Ratio (PSNR) 등의 표준 객체 탐지 지표를 통해 평가됩니다.

- **Performance Highlights**: 연구 결과, 이 프레임워크는 Foggy Cityscapes와 RESIDE-beta 데이터셋을 통해 검증되었으며, 객체 탐지 정확도를 크게 향상시키고 계산의 효율성을 최적화할 것으로 기대됩니다. 이를 통해 저조도 환경에서의 객체 탐지 성능이 크게 개선될 것으로 예상되고 있습니다.



### Polyp-SES: Automatic Polyp Segmentation with Self-Enriched Semantic Mod (https://arxiv.org/abs/2410.01210)
Comments:
          Asian Conference on Computer Vision 2024

- **What's New**: 본 논문은 콜로노스코피(Colonoscopy) 이미지의 자동 폴립 세분화(Polyp Segmentation)를 위한 'Self-Enriched Semantic Model'이라는 혁신적인 방법을 제안합니다. 이 접근법은 기존의 방법들이 가진 한계를 극복하도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 Encoder, Decoder, Self-Enriched Semantic(SES) 모듈 등 세 가지 주요 구성 요소로 이루어져 있습니다. Encoder는 다중 스케일의 특징을 추출하고, Local-to-Global Spatial Fusion(LGSF) 메커니즘을 통해 로컬 및 글로벌 공간적 특징을 캡처하여 초기 전역 특징 맵을 생성합니다. 그 후, SES 모듈을 사용하여 추가적인 의미론적 정보를 더해 모델이 문맥을 더 잘 이해할 수 있도록 지원합니다.

- **Performance Highlights**: 제안된 방법은 다섯 가지 폴립 기준에서 최신 연구들에 비해 뛰어난 세분화 성능을 보이며 학습 및 일반화 능력에서 우수한 성과를 나타냈습니다.



### AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction (https://arxiv.org/abs/2410.01202)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 AniSDF라는 새로운 접근법을 제안하여 고충실도의 3D 재구성을 위한 융합 세분화 신경 표면(fused-granularity neural surface)을 학습합니다. 이는 물리 기반 인코딩을 사용해 정확한 기하학을 복원하면서도 사진 같은 품질(photo-realistic rendering)을 달성합니다.

- **Technical Details**: AniSDF는 두 개의 주요 요소인 융합 세분화 기하 구조(fused-granularity geometry structure)와 혼합 방사장(blended radiance fields)을 도입하여 비반사(non-reflective)와 반사(reflective) 표면 간의 기하학과 외관을 명확히 구분합니다. 이는 Anisotropic Spherical Gaussians (ASG)를 통해 스펙큘러(specular) 성분을 모형화하여 정확한 반사를 모델링합니다.

- **Performance Highlights**: AniSDF는 복잡한 구조를 가진 객체를 재구성하고 고품질 렌더링을 생성할 수 있는 능력을 가지고 있습니다. 실험 결과, 우리의 방법은 SDF 기반 방법들에 비해 기하학 재구성과 새 뷰 합성(novel-view synthesis) 작업 모두에서 높은 품질을 자랑합니다.



### [Re] Network Deconvolution (https://arxiv.org/abs/2410.01189)
Comments:
          12 pages, 5 figures

- **What's New**: 이번 연구는 Ye et al. (2020)에 의해 발표된 "Network Deconvolution"의 결과를 재현하려고 합니다. "network deconvolution" 기법은 CNN의 훈련에서 pixel-wise 및 channel-wise 상관관계를 제거하여 모델 성능을 향상시킬 수 있다고 주장합니다. 본 연구는 원본 논문의 주장을 검증하기 위해 다수의 실험을 수행했습니다.

- **Technical Details**: 연구는 367개의 고유한 실험을 통해 10개의 CNN 아키텍처 및 CIFAR-10과 CIFAR-100 데이터셋을 사용하여 수행되었습니다. 네트워크 디콘볼루션은 BN 레이어를 디콘볼루션 레이어로 대체하여 입력 데이터의 공분산 행렬을 계산하고 이를 역제곱근으로 근사하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, 원본 논문에서 제시된 결과와 유사한 경향을 보였으며, 특히 Table 1의 경우 정확도가 약간의 편차를 보였지만 전체적인 경향은 일치하였습니다. Table 2의 모든 14개의 재현된 값은 원본 값과 일치했습니다.



### UAL-Bench: The First Comprehensive Unusual Activity Localization Benchmark (https://arxiv.org/abs/2410.01180)
- **What's New**: 이 연구는 비정상적인 활동의 로컬라이제이션에 대한 새로운 벤치마크인 UAL-Bench를 소개합니다. UAL-Bench는 비정상적인 활동 로컬라이제이션을 위한 세 가지 비디오 데이터셋과 하나의 지침 조정 데이터셋을 포함하고 있습니다.

- **Technical Details**: UAL-Bench는 비디오-언어 모델(Vid-LLMs)과 비전-언어 모델(VLM) 및 대형 언어 모델(LLM)을 통합한 새로운 접근법인 VLM-LLM을 평가합니다. 비정상적인 활동으로 간주되는 여러 사례에 대한 ROI 및 TD 기준으로 평가하며, 이를 통해 모델 성능을 개선합니다.

- **Performance Highlights**: VLM-LLM 접근 방식은 짧은 일관성을 보여주고, 비정상적인 사건 예상(start time)에서 Vid-LLMs보다 더욱 정확한 예측을 수행합니다. 이 연구는 장기 비디오, 특히 자폐 진단과 관련된 비디오에서의 어려움과 더불어 새로운 평가 지표인 R@1, TD <= p를 제안하여 기존 메트릭의 한계를 해결합니다.



### Automatic Image Unfolding and Stitching Framework for Esophageal Lining Video Based on Density-Weighted Feature Matching (https://arxiv.org/abs/2410.01148)
- **What's New**: 이번 논문에서는 식도 내시경 비디오를 위한 자동 이미지 전개 및 스티칭(Automatic Image Unfolding and Stitching) 프레임워크를 새롭게 제안합니다. 이 방법은 복잡한 내부 환경과 동적인 과정에서의 도전 과제를 해결하기 위해 설계되었습니다.

- **Technical Details**: 프레임워크는 LoFTR, SIFT, ORB와 같은 다양한 특징 매칭 알고리즘을 결합하여 특징 필터링 풀을 생성하고, Density-Weighted Homography Optimization (DWHO) 알고리즘을 사용하여 스티칭 정확성을 높입니다. 식도 이미지를 2차원 형식으로 변환하기 위해 Depth Anything 모델을 기반으로 깊이 기반 이미지 전개 기술을 사용합니다.

- **Performance Highlights**: 실험 결과, 프레임워크는 넓은 비디오 시퀀스에서 낮은 Root Mean Square Error (RMSE)와 높은 Structural Similarity Index (SSIM)를 달성하였으며, 이는 임상적 사용 가능성을 입증하고 내시경 시각 데이터의 품질과 연속성을 향상시키는 데 기여합니다.



### Uncertainty-Guided Enhancement on Driving Perception System via Foundation Models (https://arxiv.org/abs/2410.01144)
- **What's New**: 본 연구는 기존의 주행 인식 시스템의 예측을 개선하기 위해 멀티모달(aspect of multiple modalities) 파운데이션 모델을 활용하는 새로운 방법론을 개발했습니다. 이 방법은 자원이 많이 소모되는 모델을 최소한으로 사용하면서 객체 분류의 정확도를 높이는 데 중점을 두었습니다.

- **Technical Details**: 주요 기술적 내용으로는 불확실성을 정량적으로 특성화하고, 이 불확실성이 사전에 설정된 임계값(threshold)을 초과할 때만 파운데이션 모델을 사용하여 예측을 개선하는 접근법을 제안했습니다. 또한, 불확실성은 conformity prediction을 사용하여 감지 모델의 신뢰도 점수를 이론적인 하한으로 보정하여 측정했습니다. 이 연구는 과거 예측을 통합하여 예측 정확도를 높이는 시간적 추론 메커니즘(temporal inference mechanism)도 포함하고 있습니다.

- **Performance Highlights**: 이 방법론은 주행 데이터셋으로부터의 정량적 평가를 바탕으로 예측 정확도가 10~15% 향상되었고, 파운데이션 모델에 대한 쿼리(query) 수를 50% 줄이는 성과를 달성했습니다.



### Using Interleaved Ensemble Unlearning to Keep Backdoors at Bay for Finetuning Vision Transformers (https://arxiv.org/abs/2410.01128)
- **What's New**: 이 논문에서는 Vision Transformers (ViT)의 백도어 공격에 대한 새로운 방어 방법인 Interleaved Ensemble Unlearning (IEU)을 제안합니다. IEU는 두 개의 모델을 사용하여 고신뢰 데이터를 동적으로 제거하는 방법입니다.

- **Technical Details**: IEU는 2단계로 구성됩니다. 1단계에서는 얕은 ViT가 백도어 데이터에 대해 높은 신뢰도를 가지도록 미세 조정됩니다. 2단계에서는 이 얕은 ViT가 주 모델의 학습을 제어하여 잠재적으로 손상된 데이터를 차단하고, 유효한 데이터는 정상적으로 학습하도록 합니다. 데이터가 충분히 수집되면 동적인 비 학습 속도로 백도어 영향을 지웁니다.

- **Performance Highlights**: IEU는 11개의 최신 백도어 공격에 대해 3개의 데이터셋에서 효과적임을 입증하였으며, TinyImageNet과 CIFAR10에서 공격 성공률(Attack Success Rate, ASR)을 각각 33.83%, 31.46% 개선하였습니다. 또한 클린 정확도(Clean Accuracy)도 유지하였습니다.



### Synthetic imagery for fuzzy object detection: A comparative study (https://arxiv.org/abs/2410.01124)
- **What's New**: 이번 연구에서는 3D 모델을 기반으로 한 합성 이미지 생성 및 자동 주석 기법을 통해 퍼지(fuzzy) 객체 탐지의 새로운 접근법을 제안합니다. 퍼지 객체(예: 불, 연기)의 세밀한 탐지를 위한 데이터셋 구축의 어려움을 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 이번 연구에서는 합성 데이터(synthetic data)를 사용하여 3D 모델 기반으로 생성한 불 이미지로 객체 탐지 모델을 훈련했습니다. 이 방법은 축적된 데이터의 양과 질을 높이기 위해 자동으로 주석을 달 수 있는 시스템을 포함합니다. 다양한 비교 실험을 통해 합성 이미지와 실제 이미지를 혼합하여 훈련한 모델의 성능이 다른 모델들보다 우수함을 입증했습니다.

- **Performance Highlights**: 합성 데이터로 훈련된 ML 모델은 실제 이미지로 훈련된 모델보다 더 나은 성능을 보였으며, 더 넓은 스펙트럼의 실제 화재를 포함하는 테스트 데이터셋에서 그 성능이 향상되었습니다. 이는 퍼지 객체 탐지를 위한 합성 데이터 활용의 유효성을 보여줍니다.



### RobustEMD: Domain Robust Matching for Cross-domain Few-shot Medical Image Segmentation (https://arxiv.org/abs/2410.01110)
- **What's New**: 본 연구는 서로 다른 의료 이미징 도메인에서의 일반화 능력을 향상시키기 위해 Earth Mover's Distance (EMD)를 기반으로 한 RobustEMD 매칭 메커니즘을 도입합니다. 이는 Cross-domain Few-shot Medical Image Segmentation (CD-FSMIS) 작업을 다루며, 기존의 방법들과 비교해 도메인 간의 구조적 차이를 고려하여 성능을 개선합니다.

- **Technical Details**: 본 논문에서는 EMD를 활용한 매칭 메커니즘을 통해 포그라운드 지원 및 쿼리 기능 간의 물류 과정을 정립합니다. 텍스처 복잡성 인식 노드 가중치 생성 및 경계 보전 하우스도르프 거리 계산을 통해 노드 변환 비용 함수를 개발하여 도메인 전이 성능을 증대시킵니다. 이 과정에서 sobel 기반 이미지 기울기 계산과 지역 분산 지표를 활용하여 특성의 복잡성을 측정합니다.

- **Performance Highlights**: 우리는 cross-modal, cross-sequence, cross-institution 시나리오 하에서 8개의 의료 데이터 세트에 걸쳐 모델 성능을 평가한 결과, 본 모델이 기존 모델들에 비해 State-of-the-Art (SoTA) 성능을 달성함을 입증하였습니다.



### Semantic Segmentation of Unmanned Aerial Vehicle Remote Sensing Images using SegFormer (https://arxiv.org/abs/2410.01092)
- **What's New**: 이 연구는 UAV(무인 항공기) 이미지를 위한 새로운 의미 분할(Semantic Segmentation) 프레임워크인 SegFormer의 효과성과 효율성을 평가합니다. SegFormer는 실시간(B0)부터 고성능(B5) 모델까지 여러 변형을 가지며 UAVid 데이터셋을 통해 테스트되었습니다.

- **Technical Details**: SegFormer는 Transformer 기반 아키텍처를 활용하여 UAV 이미지에서의 의미 분할을 수행합니다. 주요 평가는 평균 교차 비율(mIoU)과 다양한 SegFormer 변형의 파라미터 수, 초당 프레임(FPS), 지연(latency) 등을 포함합니다.

- **Performance Highlights**: 실험 결과, SegFormer는 다양한 UAV 시나리오에서 높은 효율성과 성능을 보이며 객체와 지형적 특성을 정확하게 구분하는 능력을 나타냈습니다.



### FMBench: Benchmarking Fairness in Multimodal Large Language Models on Medical Tasks (https://arxiv.org/abs/2410.01089)
- **What's New**: 이 논문에서는 FMBench라는 새로운 벤치마크를 소개하며, 이는 다문화 대상을 아우르는 Multimodal Large Language Models (MLLMs)의 성능 공정성을 평가하기 위해 고안되었습니다. 이는 의료 임상 작업에서 VQA와 RG를 포함하여, 다양한 인구 통계적 속성에 대한 실질적인 평가 도구를 제공합니다.

- **Technical Details**: FMBench는 4가지 인구통계 속성(인종, 민족, 언어, 성별)을 포함하고 있으며, zero-shot 세팅에서 VQA와 RG 두 가지 작업을 수행합니다. VQA 작업에서 자유 형식의 질문이 사용되어 실제 임상 상황에 적합하도록 하였으며, Fairness-Aware Performance (FAP)라는 새로운 메트릭을 도입하여 MLLM의 공정성을 평가합니다.

- **Performance Highlights**: 8개의 최신 오픈 소스 MLLM의 성능과 공정성을 평가한 결과, 전통적인 어휘 기반 메트릭이 개방형 다중 양식 작업에는 부족하며, 인구 통계 속성에 따라 모든 MLLM의 성능이 일관되지 않음을 발견했습니다. 이는 의료 분야에서의 공정성을 고려할 필요성을 강조합니다.



### Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Tim (https://arxiv.org/abs/2410.01083)
Comments:
          ECCV 2024

- **What's New**: 이번 연구에서는 기존에 버려지는 activation maps의 활용 가능성을 제시하며, 모델의 예측 성능을 개선할 수 있음을 증명하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 activation map을 효과적으로 검색하고 집계하는 메커니즘을 통해, 모델 성능을 테스트 시간에 향상시킬 수 있도록 설계되었습니다. 이는 전이학습된 Deep Net에서 대체로 활용되지 않는 activation을 기반으로 합니다.

- **Performance Highlights**: 실험 결과, 9개의 서로 다른 아키텍처에서 제안된 방법이 이미지 분류 및 분할 작업에서 모두 성능을 개선하며, 기존의 TTA 방법과의 조합 시 추가적인 이점이 있음을 보여주었습니다.



### Pose Estimation of Buried Deep-Sea Objects using 3D Vision Deep Learning Models (https://arxiv.org/abs/2410.01061)
Comments:
          Submitted to OCEANS 2024 Halifax

- **What's New**: 이 논문에서는 남부 캘리포니아 샌 페드로 분지의 해저에서 발견된 잔해 드럼의 자세(pose) 및 매립 비율(burial fraction) 추정을 위한 새로운 접근 방식을 제안합니다. 이 방법은 최근의 foundation 모델과 vision transformer를 사용하여 드럼의 기하학을 정의하는 포인트 클라우드를 추정합니다.

- **Technical Details**: BarrelNet이라는 시스템을 도입하여 드럼 포인트 클라우드로부터 6-DOF 자세와 드럼의 반경을 추정합니다. BarrelNet은 수치적으로 생성된 드럼 포인트 클라우드를 학습하여 ROV(원격 조종 차량) 비디오 영상을 사용하여 드럼의 가능성을 시각적으로 보여줍니다. 이 방법은 기존의 최소 제곱 접근 방식과 비교하여 눈에 띄는 성능 향상을 보입니다.

- **Performance Highlights**: 본 연구의 방법은 기존의 고전적인 원통 피팅 방법과 비교하였을 때, 수치적 테스트 데이터에서 유의미한 개선을 보여줍니다. 또한, 실제 ROV에서 수집된 이미지를 통해 드럼의 자세와 매립 비율에 대한 일반화 가능성을 입증하였습니다.



### ARPOV: Expanding Visualization of Object Detection in AR with Panoramic Mosaic Stitching (https://arxiv.org/abs/2410.01055)
Comments:
          6 pages, 6 figures, to be published in SIBGRAPI 2024 - 37th conference on Graphics, Patterns, and Images proceedings

- **What's New**: ARPOV는 AR 헤드셋으로 촬영한 비디오에서 개체 탐지 모델 출력을 분석하기 위해 설계된 대화형 시각 분석 도구입니다. 이 도구는 사용자가 모델 성능을 이해하는 데 최대한의 도움을 줄 수 있도록 특화된 기능을 갖추고 있습니다.

- **Technical Details**: ARPOV의 주요 구성 요소는 다음과 같습니다: Annotated Range Slider는 ODM 출력의 관심 지점을 표시합니다. Timeline View는 모델 신뢰도, 교차 비율(IoU), 감지 분류 및 객체 이동 거리를 시간을 기준으로 요약합니다. Video Player는 원본 비디오를 표시하며, Panorama Construction Menu는 사용자가 파노라마 구축 매개 변수를 조정하여 장면의 파노라마를 생성할 수 있도록 합니다. Visualization Menu는 다양한 스타일 및 필터를 선택하여 객체 탐지를 시각화합니다.

- **Performance Highlights**: ARPOV는 머신 러닝(ML) 및 AR 전문가들과의 협업을 통해 개발되었으며, 5명의 도메인 전문가와의 인터뷰를 통해 디자인 선택을 검증했습니다. 이 도구는 객체 탐지 결과에 대한 문제 해결을 돕고, 시간 및 공간 맥락에서 모델 출력을 탐색할 수 있게 합니다.



### FCE-YOLOv8: YOLOv8 with Feature Context Excitation Modules for Fracture Detection in Pediatric Wrist X-ray Images (https://arxiv.org/abs/2410.01031)
Comments:
          arXiv admin note: text overlap with arXiv:2407.03163

- **What's New**: 이 연구는 소아 손목 골절 감지를 위해 기존 YOLOv8 모델에 다양한 Feature Contexts Excitation (FCE) 모듈을 결합한 FCE-YOLOv8 모델을 소개합니다. 각 모델은 Squeeze-and-Excitation (SE), Global Context (GC), Gather-Excite (GE), Gaussian Context Transformer (GCT)와 같은 다른 성능 향상 모듈을 탑재하여 더 나은 감지 성능을 달성합니다.

- **Technical Details**: FCE-YOLOv8 모델은 YOLOv8 아키텍처에 FCE 모듈을 추가하여 소아 손목 골절 X-ray 이미지를 진단하는 데 사용됩니다. GRAZPEDWRI-DX 데이터셋을 사용한 실험에서, YOLOv8+GC-M3 모델은 mAP@50 값을 65.78%에서 66.32%로 향상시켰으며, YOLOv8+SE-M3 모델은 67.07%의 최고 mAP@50 값을 기록하며 현재 SOTA 성능을 초과합니다. 이러한 모델은 신속한 추론 시간을 제공하여 실제 진단에서 유용하게 활용될 수 있습니다.

- **Performance Highlights**: 이번 연구에서 제안된 YOLOv8+SE-M3 모델은 GRAZPEDWRI-DX 데이터셋에서 67.07%로 SOTA 성능을 초과하며, YOLOv8+GC-M3 모델은 66.32%로 높은 성능을 보여줍니다. 두 모델 모두 높은 정확도를 유지하면서도 추론 시간을 단축시켜 실시간 의료현장에서의 활용 가능성을 높입니다.



### Can visual language models resolve textual ambiguity with visual cues? Let visual puns tell you! (https://arxiv.org/abs/2410.01023)
Comments:
          Accepted as main paper in EMNLP 2024

- **What's New**: 이 논문은 멀티모달 간섭(multi-modal interference)- 즉, 언어와 이미지를 포함하는 다양한 형태의 입력을 결합하여 텍스트의 모호성을 해소하는 새로운 벤치마크인 UNPIE(Understanding Pun with Image Explanations)를 소개합니다. 이벤치마크는 1,000개의 재미있는 말장난(pun)과 해당 의미를 설명하는 이미지로 구성되어있으며, 이를 통해 멀티모달 리터러시(multi-modal literacy)를 평가합니다.

- **Technical Details**: UNPIE 데이터셋은 말장난을 중심으로 구성되어 있으며, 텍스트-이미지 쌍을 통해서 리터러시를 측정하기 위한 세 가지 테스트(pun grounding, disambiguation, reconstruction)를 제안합니다. 이 연구는 기존의 단일 텍스트 모델과 비교해 새로운 시각적 맥락을 통해 다루어지는 모델들의 능력을 평가합니다. 이러한 모델은 고차원적 작업일수록 성과가 향상되는 경향을 보였습니다.

- **Performance Highlights**: UNPIE의 실험 결과에 따르면, VLM(Visual-Language Models)과 Socratic 모델 둘 다 단순 텍스트 모델보다 멀티모달 정보를 활용할 때 성능이 개선되었습니다. 특히, 더 높은 난이도의 작업에서 이러한 경향이 두드러지게 나타났고, VLM은 Socratic 모델보다 더 나은 성능을 보였습니다.



### A Critical Assessment of Visual Sound Source Localization Models Including Negative Audio (https://arxiv.org/abs/2410.01020)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구는 Visual Sound Source Localization (VSSL) 모델의 평가 방식을 혁신적으로 확장하였습니다. 특히, 양의 오디오 예제뿐만 아니라 음의 오디오 사례를 포함한 포괄적인 테스트 세트를 제안합니다.

- **Technical Details**: 연구에서는 VSSL 모델의 성능을 평가하기 위해 양성 및 음성 오디오 범주를 포함한 데이터셋을 확장하였습니다. 음의 오디오는 침묵, 노이즈, 오프스크린 소리의 세 가지 유형으로 정의되었습니다. 새로운 메트릭을 활용하여 모델의 성능을 분석하였으며, cIoU와 AUC 지표를 통해 모델의 예측 정확성을 평가하였습니다.

- **Performance Highlights**: SOTA 모델들의 예측이 음성 오디오 입력에 따라 적절히 조정되지 않는 것으로 나타났으며, 많은 모델들이 오디오 정보 활용에 한계를 보였습니다. 또한, 모델 간의 큰 성능 차이를 발견하여 실세계 응용에 적합한 보편적인 임계값을 선택하는 데 어려움이 있음을 시사합니다.



### Y-CA-Net: A Convolutional Attention Based Network for Volumetric Medical Image Segmentation (https://arxiv.org/abs/2410.01003)
- **What's New**: 최근의 주목 기반 볼륨 분할(volumetric segmentation, VS) 방법론이 의료 분야에서 주목할 만한 성과를 달성했지만, 주로 장기 의존성(long-range dependencies) 모델링에 초점을 두고 있습니다. 본 연구에서는 로컬 피쳐(local features)가 VS 모델의 성능에 중요한 요소로 작용하는 반면, 주목 기반 VS 방법에서는 이러한 요소가 결여되어 있는 문제를 해결하고자 하였습니다.

- **Technical Details**: Y-CT-Net 모델을 기반으로 하여, 트랜스포머(backbone)와 컨볼루션 인코더(branch)를 병렬적으로 통합하여 로컬 및 글로벌 피쳐를 추출한 후, Cross Feature Mixer Module (CFMM)을 통해 더욱 나은 분할 마스크 예측을 수행합니다. 이어서 hybrid attention 모델을 통해 Y-CH-Net을 확장하였으며, Y-CA-Net이라는 일반적인 아키텍처를 제안하여 컨볼루션 및 주의 메커니즘의 보완적 강점을 최대한 활용하고자 하였습니다.

- **Performance Highlights**: Y-CT-Net은 다기관 분할(multi-organ segmentation)에서 82.4%의 다이스 점수를 기록하여, 잘 조정된 VS Transformer/CNN 모델인 UNETR 및 ResNet-3D를 각각 2.9% 및 1.4% 초과 성능을 보였습니다. 또한 Y-CH-Net은 동일한 분할 작업에 대해 HD95 점수 기준으로 3%의 개선을 이끌어냈습니다. Y-CA-Net 역시 여러 벤치마크 데이터셋에서 현존하는 최첨단 방법들과 비교하여 뚜렷한 성과 향상을 보여주었습니다.



### LaDTalk: Latent Denoising for Synthesizing Talking Head Videos with High Frequency Details (https://arxiv.org/abs/2410.00990)
- **What's New**: 본 논문에서는 Wav2Lip 모델을 기반으로 한 효과적인 후처리 방법인 LaDTalk를 통해 포토리얼리스틱한 토킹 헤드 비디오 생성을 위한 새로운 접근 방식을 제시합니다. LaDTalk는 고주파 텍스처 세부 정보의 보존을 강화하며, 고유 식별자를 가진 비디오 생성에서의 일관성을 보장합니다.

- **Technical Details**: LaDTalk는 Space-Optimised Vector Quantised Auto Encoder (SOVQAE) 모델을 활용하여, Wav2Lip로 생성된 저해상도 비디오에서 고해상도 비디오로 변환합니다. 이 과정에서 Lipschitz Continuity 이론을 적용하여 Vector Quantised Auto Encoders (VQAEs)의 노이즈 강인성을 향상시킵니다. 또한, 새로운 High-Frequency TalKing head (HFTK) 데이터셋을 구축하여 모델의 성능을 평가합니다.

- **Performance Highlights**: LaDTalk는 기존의 최첨단 방법들에 비해 입술 동기화 정확도에서 월등한 성능을 보이며, 영상 품질 및 세부 사항을 더욱 향상시킵니다. 실험 결과, LaDTalk가 고주파 세부 정보 생성에서 뛰어난 성능을 발휘하며, 초고해상도 얼굴 이미지를 합성할 수 있음을 demonstrated 합니다.



### ScVLM: a Vision-Language Model for Driving Safety Critical Event Understanding (https://arxiv.org/abs/2410.00982)
- **What's New**: 이번 연구에서는 자율주행 및 운전 안전성 연구를 위한 Driving Safety-Critical Events (SCEs), 즉 사고 및 거의 사고를 이해하고 설명하기 위한 새로운 접근법인 ScVLM을 제안합니다. 이 하이브리드 모델은 감독 학습(supervised learning)과 대조 학습(contrastive learning)을 결합하여 비디오를 통해 보다 정확한 이벤트 설명을 강화합니다.

- **Technical Details**: ScVLM은 8,600개 이상의 SCE로 구성된 대규모 데이터셋인 Second Strategic Highway Research Program Naturalistic Driving Study 데이터셋에서 훈련되었습니다. 이 방법은 사건 유형(사고, 타이어 충돌 등)을 인식하기 위해 감독 학습을 사용하고, 갈등 유형을 식별하기 위해 대조 학습을 사용하며, 비디오에서 주어진 환경 맥락을 이해하기 위해 비전-언어 모델(VLM)을 활용합니다.

- **Performance Highlights**: 제안된 ScVLM 접근법은 상황에 맞는 정확한 이벤트 설명을 생성하는 데 있어 탁월한 성능을 보였으며, VLM에서 발생할 수 있는 허위 현상(hallucinations)을 효과적으로 감소시켰습니다.



### Towards Full-parameter and Parameter-efficient Self-learning For Endoscopic Camera Depth Estimation (https://arxiv.org/abs/2410.00979)
Comments:
          WiCV @ ECCV 2024

- **What's New**: 이 논문에서는 내시경 심도 추정을 위한 심층 기초 모델의 적응 방법을 제안합니다. 이를 통해 기존의 낮은 순위 서브스페이스에 제한되는 방식에서 벗어나 전체 파라미터 기반의 효율적인 학습 프레임워크를 구현합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 주의(attention), 컨볼루션(convolution), 다층 퍼셉트론(mlp)의 서브스페이스를 동시에 적응시킵니다. 두 번째 단계에서는 메모리 효율적 최적화(memory-efficient optimization)를 통해 여러 서브스페이스를 결합하여 성능을 더욱 개선합니다.

- **Performance Highlights**: SCARED 데이터셋을 사용한 초기 실험에서는 첫 번째 단계에서 Sq Rel, Abs Rel, RMSE 및 RMSE log의 성능이 각각 10.2%에서 4.1%로 향상되었습니다. 이는 최신의 심도 모델에 비해 개선된 결과를 보여줍니다.



### SegHeD: Segmentation of Heterogeneous Data for Multiple Sclerosis Lesions with Anatomical Constraints (https://arxiv.org/abs/2410.01766)
Comments:
          13 pages, 4 figures, MICCAI, LDTM Workshop

- **What's New**: SegHeD는 이질적인 데이터 형식과 주석 프로토콜을 처리하는 새로운 다중 데이터셋 다중 작업 뇌 병변 세분화 모델입니다. 이 모델은 모든 병변, 새로운 병변, 그리고 소멸 병변을 세분화할 수 있는 기능을 제공합니다.

- **Technical Details**: SegHeD는 크로스 섹션(cross-sectional) 및 종단적(longitudinal) 이미지를 사용하여 학습하며, 세 가지 주석 프로토콜(모든 병변, 새로운 병변, 소멸 병변)을 고려하여 설계되었습니다. 모델은 구조적 일관성과 부피 일관성을 유지하면서도 서로 다른 데이터 형식의 입력을 가능하게 합니다.

- **Performance Highlights**: SegHeD는 다섯 개의 MS 데이터셋에서 평가되었으며, 모든 병변, 새로운 병변, 소멸 병변 세분화에서 높은 성능을 보여주며 여러 최첨단 방법보다 우수한 결과를 달성했습니다.



### COSMIC: Compress Satellite Images Efficiently via Diffusion Compensation (https://arxiv.org/abs/2410.01698)
- **What's New**: COSMIC은 위성 이미지 전송을 위한 경량 학습 압축 솔루션으로, 압축 비율을 우선시하는 경량 인코더를 설계하고 지상에서 디퓨전 모델을 통해 이미지 세부 사항을 보완합니다.

- **Technical Details**: COSMIC은 두 개의 주요 구성 요소로 구성됩니다. 첫 번째는 위성에서 압축을 위한 경량 인코더이며, 두 번째는 지상에서 복원 시 디퓨전 모델이 포함된 복잡한 복원 과정입니다. 우리는 FLOPs를 2.6~5배 줄이는 경량 합성곱 아키텍처를 통해 지역 특성을 추출하고 있으며, 디퓨전 모델을 사용하여 이미지의 멀티모달 특성을 활용합니다.

- **Performance Highlights**: COSMIC은 6개의 최신 SOTA 모델과 비교하였으며, 왜곡 및 인지 메트릭 모두에서 우수한 성능을 보여줍니다. 이를 통해 COSMIC은 기존 솔루션에 비해 2.6~5배 더 적은 FLOPs로 모든 메트릭에서 더 나은 결과를 달성함을 보여주었습니다.



### MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning (https://arxiv.org/abs/2410.01697)
- **What's New**: 제안된 Multi-Objective REpresentation Learning (MOREL) 접근법은 적대적 작용에 대응하기 위해 특징 표현 학습의 강도를 강조하여 모델의 견고성을 높입니다.

- **Technical Details**: MOREL은 클래스 내 유사한 입력에 대해 일관된 특징을 생성하도록 모델을 유도하는 다목적 최적화 프레임워크를 사용합니다. 이 과정에서 코사인 유사도 손실(cosine similarity loss) 및 다중 긍정 대비 손실(multi-positive contrastive loss)를 활용하여 자연적 및 적대적 특징을 정렬하고 밀집 클러스터를 형성합니다.

- **Performance Highlights**: MOREL로 훈련된 모델은 기존 적대적 훈련 방법보다 화이트 박스 및 블랙 박스 적대적 공격에 대해 더 뛰어난 견고성을 보임을 강조하며, 아키텍처 변경이나 테스트 데이터 정화 없이 정확성과 견고성 간의 균형을 효과적으로 조절합니다.



### PHI-S: Distribution Balancing for Label-Free Multi-Teacher Distillation (https://arxiv.org/abs/2410.01680)
- **What's New**: 이번 연구는 라벨 없이 여러 heterogeneous visual foundation model을 융합하는 agglomerative model의 발전에 중점을 두고 있으며, 특히 teacher 모델의 activation 통계와 손실 함수가 student 모델의 품질에 미치는 영향을 다룹니다.

- **Technical Details**: 온도 단순화 실험에서, 우리는 다양한 teacher 모델의 activation 분포와 그 분포의 분산을 분석합니다. 이를 통해 Hadamard 행렬을 사용하여 isotropic 표준화를 수행하고, 이를 'PHI Standardization' (PHI-S)라고 이름 붙였으며, 이 방법이 최상의 student 모델을 생성함을 보여주었습니다.

- **Performance Highlights**: PHI Standardization 방법을 통해 생성된 student 모델은 평가 기준에서 가장 우수한 성능을 보였으며, 여러 가지 합성곱 손실 함수들을 비교 분석하여 그 결과를 제공합니다.



### Towards a vision foundation model for comprehensive assessment of Cardiac MRI (https://arxiv.org/abs/2410.01665)
Comments:
          11 pages, 3 figures, 4 tables

- **What's New**: 이번 논문에서는 심장 자기 공명 영상(CMR) 평가를 위한 비전 기초 모델을 소개합니다. 이 모델은 3600만 개의 CMR 이미지에 대해 자기 지도 방식으로 훈련되었습니다.

- **Technical Details**: 모델은 분류(classification), 분할(segmentation), 랜드마크 위치 지정(landmark localization), 병리 탐지(pathology detection)의 9가지 임상 작업을 위해 미세 조정(finetuning)되었습니다. 각 작업에 대해 다양한 크기의 레이블 데이터셋에서 정확도와 견고성이 개선되었습니다.

- **Performance Highlights**: 대부분의 임상 작업에 대해 SoTA에 상응하는 성능을 달성했으며, 레이블 샘플이 적어도 몇 샷 학습(few-shot learning)에서 개선된 결과를 보였습니다. 제안된 방법은 자원 효율적인 통합 프레임워크를 제공하여 이미지 분석 작업에서 딥 러닝 기반 솔루션의 개발을 가속화할 수 있는 잠재력을 가지고 있습니다.



### Unleashing Parameter Potential of Neural Representation for Efficient Video Compression (https://arxiv.org/abs/2410.01654)
- **What's New**: 이 논문은 현재의 암묵적 신경 표현(Implicit Neural Representation, INR)을 기반으로 한 비디오 압축 방법들이 정보 보존의 잠재력을 최대로 활용하지 못하고 있음을 밝히고, 파라미터 재사용(parameter reuse) 메커니즘을 통해 이 문제를 해결하고자 한다.

- **Technical Details**: INR 기반 비디오 압축은 단순한 신경 네트워크를 학습하여 타겟 비디오를 피팅하고, 그 후 네트워크 파라미터를 압축하여 비디오 압축을 이룬다. 저자들은 네트워크 심화 및 파라미터 재사용을 통해 비디오 표현의 성능을 극대화하는 방안을 제안하였다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 기존의 INR 기반 비디오 압축 방법보다 훨씬 우수한 비율-왜곡 성능을 보여주었으며, 다양한 비디오 데이터셋에 걸쳐서 압축 효율을 크게 향상시켰다.



### Imaging foundation model for universal enhancement of non-ideal measurement C (https://arxiv.org/abs/2410.01591)
- **What's New**: 본 논문에서는 일반화 가능한 비이상 측정 컴퓨터 단층 촬영(NICT) 이미지를 개선하기 위한 최초의 이미징 기반 모델인 다중 스케일 통합 변압기 AMPlifier(“TAMP”)를 제안합니다. TAMP는 360만 쌍의 NICT-ICT 이미지 쌍을 기반으로 사전 훈련되었습니다.

- **Technical Details**: TAMP는 물리적 프로세스에 기반한 사전 훈련 및 적응 프로세스를 구성하며, 3.6 백만 개의 시뮬레이션된 NICT-ICT 이미지 쌍으로 훈련되었습니다. 이 모델은 다양한 비이상 NICT 상황과 신체 부위에 직접 일반화할 수 있는 능력을 가지고 있습니다. 데이터 금고 (Low-rank adaptation, LoRA) 기법을 활용하여 몇 가지 고유한 매개 변수를 조정하여 특정 시나리오에서의 성능을 향상시킵니다.

- **Performance Highlights**: TAMP는 27개의 NICT 개선 작업을 평가하였고, 추가적인 훈련 없이도 다양한 NICT 이미지를 직접적으로 향상시킬 수 있는 능력을 보여주었습니다. 또한 벤치마크 데이터세트인 SimNICT를 공개하여 연구자들이 NICT 개선에 대한 딥러닝 방법을 탐구할 수 있는 유용한 자원을 제공합니다.



### Robo-MUTUAL: Robotic Multimodal Task Specification via Unimodal Learning (https://arxiv.org/abs/2410.01529)
Comments:
          preprint

- **What's New**: 이 연구는 로봇이 다양한 모달리티(task specification)에서 복합적인 작업 지시를 이해할 수 있도록 하는 크로스 모달 정렬(Cross-modality Alignment) 능력을 강화하는 방법을 제시합니다.

- **Technical Details**: 우리는 유니모달(unimodal) 지침을 활용하여 멀티모달(multi-modal) 작업 설명을 학습할 수 있는 방법론을 제안합니다. 로봇에 대한 멀티모달 인코더를 사전 훈련(pretrain)한 후, 두 가지 Collapse 및 Corrupt 작업을 사용하여 잔여적인 모달리티 간의 격차를 줄입니다. 이 방식은 동일한 작업 목표에 대한 다양한 모달리티를 상호 교환 가능한 표현으로 전환시킴으로써 잘 정렬된 멀티모달 잠재 공간(latent space)에서 로봇 작업을 가능하게 합니다.

- **Performance Highlights**: 130개 이상의 작업과 4000회의 평가를 통해 시뮬레이션된 LIBERO 벤치마크 및 실제 로봇 플랫폼에서 우리의 프레임워크가 데이터 제약을 극복하는 데 있어 현저한 이점을 보여줍니다.



### SurgPointTransformer: Vertebrae Shape Completion with RGB-D Data (https://arxiv.org/abs/2410.01443)
- **What's New**: 본 연구는 RGB-D 데이터를 활용하여 방사선 없는 3D 척추 해부학 재구성 방법을 제안합니다. SurgPointTransformer를 사용하여 외부 영역의 관찰로부터 노출되지 않은 척추 영역을 정확하게 재구성할 수 있습니다.

- **Technical Details**: 우리의 방법은 두 가지 주요 단계로 구성됩니다: 분할(Segmentation) 및 형태 완성(Shape Completion). 첫 번째 단계에서는 척추 기둥의 위치 탐지 및 분할을 수행한 후, 각 척추에 대한 세분화를 진행합니다. 이 세분화된 데이터는 주의 메커니즘(Attention Mechanism)을 활용하여 가시적 표면 특징과 숨겨진 해부학적 구조 간의 패턴을 학습하는 SurgPointTransformer에 입력됩니다.

- **Performance Highlights**: 우리 방법은 평균 챔퍼 거리(Chamfer Distance) 5.39, F-Score 0.85, 지구 이동 거리(Earth Mover's Distance) 0.011, 신호 대 잡음 비율(Signal-to-Noise Ratio) 22.90 dB를 기록하며, 기존 최첨단 성능을 획기적으로 초월했습니다.



### CSIM: A Copula-based similarity index sensitive to local changes for Image quality assessmen (https://arxiv.org/abs/2410.01411)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 Gaussian Copula를 기반으로 한 새로운 이미지 유사성 메트릭인 CSIM을 제안합니다. CSIM은 실시간으로 미세한 이미지 변화를 감지할 수 있도록 설계되었습니다.

- **Technical Details**: CSIM 메트릭은 확률 이론에서 Gaussian Copula를 사용하여 이미지를 픽셀 분포 벡터로 변환하고, 이 벡터는 픽셀 값 간의 의존성 정보를 포함하여 이미지 내의 구조적 관계를 캡처합니다. CSIM은 joint distribution을 효과적으로 모델링하여 두 이미지의 유사성을 보다 섬세하게 비교합니다.

- **Performance Highlights**: 실험 결과에 따르면 CSIM은 다양한 이미지 왜곡 시나리오, 예를 들어 노이즈, 압축 아티팩트 및 블러 등에서 기존 유사성 메트릭보다 우수한 성능을 보였습니다. CSIM의 미세한 차이를 감지하는 능력은 의료 이미지와 같은 고정밀도가 요구되는 애플리케이션에 적합합니다.



### Toward Zero-Shot Learning for Visual Dehazing of Urological Surgical Robots (https://arxiv.org/abs/2410.01395)
- **What's New**: 이번 연구에서는 로봇 비전의 경우에서 방울이 생성하는 안개 효과를 해결하기 위해 비지도형 제로샷 디헤이징 방법(RSF-Dehaze)을 제안합니다. 이 방법은 단일 입력 이미지만을 사용하여 흐림을 제거하는 새로운 방식을 제공합니다. 또한, 로봇 수술 비전을 위한 공공 데이터셋인 USRobot-Dehaze 데이터셋을 처음으로 조직하고 제안합니다.

- **Technical Details**: RSF-Dehaze는 두 가지 네트워크(디헤이징 네트워크 및 글로벌 퍼셉션 네트워크)로 구성됩니다. 디헤이징 네트워크는 5개의 동일한 합성 단위로 이루어져 있으며, 각 단위는 재구성 모듈, 컨볼루션 모듈, 영역 유사성 채우기 모듈로 구성됩니다. 도움을 주기 위해 YCbCr 색 공간으로 변환된 이미지를 사용하여 방울에 의한 흐림 현상을 개선합니다.

- **Performance Highlights**: RSF-Dehaze는 세 가지 실제 유로로지 surgical 시나리오에서 20개의 최신 디헤이징 및 이미지 회복 알고리즘과 비교 실험을 통해 그 효과를 입증했습니다. 제안된 방법은 제한된 보기 범위에서는 더 많은 실질적인 이점을 보여 줍니다.



### Anti-biofouling Lensless Camera System with Deep Learning based Image Reconstruction (https://arxiv.org/abs/2410.01365)
Comments:
          9 pages, 8 figures, Ocean Optics 2024

- **What's New**: 최근 수년 동안 해양 구조물의 상태를 모니터링하고 수산 양식 환경에서의 개체 수를 체크하는 데 사용되는 수중 카메라에 대한 수요가 증가하고 있습니다. 본 연구에서는 생물 부착물(biofouling)에 저항성이 높은 재료 기술과 심층 학습(deep learning)을 기반으로 한 이미지 재구성 기술을 사용하는 렌즈 없는 카메라를 제안합니다.

- **Technical Details**: 프로토타입 카메라는 구리(copper)와 같은 얇은 금속판에 1k 직사각형 모양의 핀홀(pinhole)을 가진 코딩된 개구(aperture)를 사용하여 생물 부착물의 성장을 방지합니다. 이 연구에서는 비전 트랜스포머(ViT)와 게이티드 다계층 퍼셉트론(gated MLP)을 사용하여 이미지 재구성을 수행하고, 이 두 접근 방식이 우수한 결과를 나타내는 것을 보여줍니다. 또한, 효과적인 생물 방지(bio-repellence) 재료의 두께를 고려하여 핀홀 보다 충분히 얇은 개구를 사용해야 함을 설명합니다.

- **Performance Highlights**: 현재 해양 환경에서 실제 테스트를 진행 중이며, 렌즈가 없는 카메라와 기존의 방수 카메라 간의 생물 부착물 효과를 비교하고 있습니다. 비전 트랜스포머와 게이티드 MLP 모두 기존의 렌즈 기반 카메라보다 더 나은 이미지 성능을 보여줍니다.



### Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy (https://arxiv.org/abs/2410.01345)
- **What's New**: 이 논문에서는 비전-언어 로봇 조작 정책의 일반화 능력을 평가하기 위한 새로운 벤치마크인 GemBench를 소개합니다. 이 벤치마크는 총 7가지 일반적인 행동 프리미티브와 4단계의 일반화 수준을 포함합니다.

- **Technical Details**: GemBench는 다양한 복잡한 작업을 포함하며, 3D 시각 정보를 활용한 언어 조건의 로봇 조작 정책인 3D-LOTUS를 제안합니다. 3D-LOTUS는 높은 효율성과 성능을 제공하지만 새로운 작업에서의 일반화에는 한계가 있습니다. 이를 개선하기 위해, 3D-LOTUS++를 도입하여 LLM과 VLM을 통합하여 작업 계획 및 객체 정위를 강화했습니다.

- **Performance Highlights**: 3D-LOTUS++는 GemBench에서 Level 2에서 4까지의 새로운 과제에서 최첨단 성능을 달성하여 로봇 조작의 일반화에 대한 새로운 기준을 설정했습니다.



### Forte : Finding Outliers with Representation Typicality Estimation (https://arxiv.org/abs/2410.01322)
- **What's New**: 이 논문은 기존의 Generative 모델들이 생성한 데이터의 OOD(Out-Of-Distribution) 탐지 문제를 다루고 있으며, 새로운 접근 방식인 Forte를 소개합니다. Forte는 self-supervised learning 기법을 활용하여 OOD 탐지의 정확성을 높이며, class labels가 필요하지 않고, OOD 데이터와의 노출 없이도 작동합니다.

- **Technical Details**: Forte는 CLIP, ViT-MSN 및 DINOv2와 같은 다양한 표현 학습 기법을 비모수 밀도 추정 모델(OCSVM, KDE, GMM)과 결합하여 atypical samples를 탐지합니다. 이 방법은 전체 데이터의 정보적 summary statistics를 효과적으로 반영하는 데이터 포인트 단위의 통계들을 제공합니다.

- **Performance Highlights**: Forte는 다양한 OOD 탐지 작업과 합성 이미지 탐지에서 기존의 최첨단 방법들보다 우수한 성능을 보여주며, 포토리얼리스틱 이미지 생성을 포함한 여러 벤치마크에서 최상의 성능을 달성했습니다.



### CANVAS: Commonsense-Aware Navigation System for Intuitive Human-Robot Interaction (https://arxiv.org/abs/2410.01273)
Comments:
          project page this https URL

- **What's New**: 이번 논문에서는 CANVAS라는 새로운 프레임워크를 소개하여 로봇이 인간의 모호한 지침 (abstract instructions)을 이해하고 내비게이션 경로를 최적화할 수 있도록 돕습니다.

- **Technical Details**: CANVAS는 비주얼 및 리걸 관점의 입력을 통합하여 로봇이 인간의 내비게이션 행동을 모방하는 방식으로 학습하는 이모테이션 러닝 (Imitation Learning)을 기반으로 합니다. COMMAND 데이터셋은 48시간 이상의 드라이빙 데이터를 포함하여 총 219킬로미터를 커버합니다. 이 데이터셋은 고유하게 인간이 주석을 단 내비게이션 결과로 구성되어 있습니다.

- **Performance Highlights**: CANVAS는 ROS NavStack보다 모든 환경에서 67%의 성공률을 기록하며, 특히 과수원 환경에서는 ROS NavStack이 0%의 성공률을 기록한 반면 CANVAS는 67%의 성공률을 보였습니다. 실제 환경에서도 69%의 전반적인 성공률을 달성하며 Sim2Real (시뮬레이션에서 실제) 전이에 우수한 성과를 보입니다.



### RS-FME-SwinT: A Novel Feature Map Enhancement Framework Integrating Customized SwinT with Residual and Spatial CNN for Monkeypox Diagnosis (https://arxiv.org/abs/2410.01216)
Comments:
          37 Pages, 5 Tables, 10 Figures

- **What's New**: 이번 논문에서는 Monkeypox (MPox) 감지를 위한 혁신적인 하이브리드 접근법인 RS-FME-SwinT가 제안되었습니다. 이 방법은 Residual Learning과 Spatial Exploitation CNN, 그리고 맞춤형 Swin Transformer의 학습 능력을 통합하여 MPox 진단에 필요한 다중 스케일의 글로벌 및 로컬 상관 피쳐를 캡처합니다.

- **Technical Details**: RS-FME-SwinT 기법은 전이 학습 기반의 Feature Map Enhancement (FME) 기법을 사용하여 글로벌 정보를 캡처하기 위한 맞춤형 Swin Transformer와 텍스처 추출을 위한 residual blocks 및 로컬 대비 변화를 처리하기 위한 spatial blocks를 통합합니다. 새롭게 도입된 inverse residual blocks는 로컬 패턴 캡처를 효과적으로 수행하고 gradient 소멸 문제를 완화합니다.

- **Performance Highlights**: RS-FME-SwinT는 다양한 MPox 데이터셋에서 크로스 검증을 통해 기존의 최신 CNN 및 ViT 모델보다 우수한 성능을 보였습니다. MPox 탐지에서 정확도 97.80%, 민감도 96.82%, 정밀도 98.06%, F-score 97.44%의 성과를 기록했습니다. 이 모델은 의료 종사자들에게 신속하고 정확한 MPox 진단을 가능하게 할 수 있는 유용한 도구로 평가됩니다.



### Formula-Driven Data Augmentation and Partial Retinal Layer Copying for Retinal Layer Segmentation (https://arxiv.org/abs/2410.01185)
Comments:
          The 11th OMIA Workshop on MICCAI 2024

- **What's New**: 이 논문에서는 Optical Coherence Tomography (OCT) 이미지를 이용한 망막층 세분화 방법에서 flattening(평면화)의 필요성을 없애기 위해 새로운 데이터 증강 방법을 제안합니다. Formula-driven data augmentation (FDDA)와 partial retinal layer copying (PRLC) 기법을 도입하여 다양한 망막 구조를 모사하고, flattening 없이 망막 층의 경계를 탐지할 수 있도록 합니다.

- **Technical Details**: FDDA는 주어진 수학적 공식에 따라 OCT 이미지의 각 열을 수직으로 이동시켜 다양한 망막 구조를 모사합니다. 또한, PRLC는 망막 층의 일부를 복사하여 망막 층 외부에 붙여넣는 방식으로 데이터 증강을 수행합니다. 이를 통해 모델이 다양한 구조를 학습할 수 있는 기반을 마련합니다.

- **Performance Highlights**: 실험 결과, FDDA와 PRLC를 활용한 데이터 증강이 flattening에 의존하지 않고도 정확한 망막층 경계 탐지가 가능하다는 것을 입증했습니다. 이러한 기법들은 전통적인 방법들이 가지는 한계를 극복하고, 다양한 망막 구조가 존재하는 실제 임상 환경에서 유용할 것으로 기대됩니다.



### GraphRevisedIE: Multimodal Information Extraction with Graph-Revised Network (https://arxiv.org/abs/2410.01160)
- **What's New**: 이번 논문에서는 GraphRevisedIE라는 경량 모델을 제안하여 visually rich documents (VRD)에서의 key information extraction (KIE) 문제를 개선하고자 하였다. 이 모델은 텍스트, 시각 및 레이아웃 기능을 효과적으로 융합하여 다중 모드 기능을 이용한 KIE 작업의 성능을 향상시킨다.

- **Technical Details**: GraphRevisedIE는 그래프 수정(graph revision) 기술을 활용하여 문서의 그래프 표현을 학습하고, graph convolution을 통해 글로벌 컨텍스트로 다중 모드 기능의 삽입을 강화한다. 그래프 모듈은 희소 문서에 적절한 그래프 표현을 학습할 수 있도록 sparsification 기술을 적용한다.

- **Performance Highlights**: 다양한 실제 데이터 세트에 대한 실험 결과, GraphRevisedIE는 기존의 그래프 기반 모델들보다 더 우수한 성능을 보였으며, 사전 훈련된 모델들과 비교했을 때도 비슷한 성능을 발휘하면서 매개 변수가 훨씬 적고 큰 사전 훈련 데이터 세트에 의존하지 않는다.



### Generating Seamless Virtual Immunohistochemical Whole Slide Images with Content and Color Consistency (https://arxiv.org/abs/2410.01072)
- **What's New**: 이번 연구에서는 CC-WSI-Net이라는 새로운 가상 슬라이드 이미지(WI) 합성 네트워크를 제안합니다. 이 모델은 GAN을 기반으로 하여 타일 경계에서 발생하는 불일치를 해결하고, 일관성 있는 가상 IHC 염색 이미지를 생성하는 데 중점을 둡니다.

- **Technical Details**: CC-WSI-Net은 내용 및 색상 일관성 감독 모듈(content- and color-consistency supervisor)을 통합하여 생성된 WSIs의 타일 간 일관성을 보장합니다. 이는 MELANOCYTE(멜라노사이트) 감지에서 Sox10 면역 히스토 화학(Immunohistochemistry) 정확성을 보장합니다.

- **Performance Highlights**: 연구진은 이미지 품질 분석, 객관적인 감지 평가, 그리고 병리학자들로부터의 주관적 조사를 통해 CC-WSI-Net의 우수성을 검증하였습니다. 이 방법은 고품질의 합성 WSIs를 생성하여 가상 염색 기술의 발전을 위한 기회를 열어줍니다.



### TransResNet: Integrating the Strengths of ViTs and CNNs for High Resolution Medical Image Segmentation via Feature Grafting (https://arxiv.org/abs/2410.00986)
Comments:
          The 33rd British Machine Vision Conference 2022

- **What's New**: 최근 발표된 연구는 고해상도 의료 영상에서의 이미지 분할을 위한 새로운 구조인 TransResNet을 제안하고 있습니다. 이 구조는 Transformer와 CNN을 병렬로 결합하여 다양한 해상도의 이미지에서 독립적으로 특징을 추출합니다.

- **Technical Details**: TransResNet은 Cross Grafting Module (CGM)을 도입하여 Transformer 및 CNN의 특징 맵을 융합하고 self-attention 메커니즘을 통해 저수준의 공간적 세부정보와 글로벌 의미 정보를 결합하는 기법을 사용합니다. 이를 통해 분할 마스크 예측을 위한 디코딩 과정에서 정보 흐름을 증가시킵니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시된 실험 결과, TransResNet은 피부 병변, 망막 혈관, 폴립 분할 작업을 포함하여 여러 분할 작업에서 최첨단의 성능을 달성했습니다.



### Evaluating Deep Regression Models for WSI-Based Gene-Expression Prediction (https://arxiv.org/abs/2410.00945)
- **What's New**: 이 연구에서는 전체 슬라이드 이미지(WSI)로부터 직접적으로 mRNA 유전자 발현 프로필을 예측하는 딥 러닝 모델을 제안합니다. 이는 비용 효율적이면서도 폭넓게 접근 가능한 분자 표현형 분석(Molecular phenotyping)을 제공할 수 있습니다.

- **Technical Details**: 연구는 WSI 기반의 유전자 발현 예측 모델의 높은 차원 회귀 문제(Regression problem)와 관련된 설계 선택들에 대해 자세히 분석합니다. 제안된 방법으로는 20530개의 유전자를 동시에 회귀하는 단일 모델을 훈련시키는 것이 있습니다.

- **Performance Highlights**: 단일 모델 훈련 방식이 계산적으로 효율적이며 매우 강력한 기준선(baseline)을 제공한다는 결론을 내립니다.



### CBAM-SwinT-BL: Small Rail Surface Defect Detection Method Based on Swin Transformer with Block Level CBAM Enhancemen (https://arxiv.org/abs/2409.20113)
Comments:
          27 pages, 17 figures

- **What's New**: 본 연구는 철도 운영 중 발생하는 다양한 결함을 효과적으로 감지하고 유지보수를 지원하기 위한 새로운 프레임워크인 CBAM-Enhanced Swin Transformer in Block Level (CBAM-SwinT-BL)을 소개합니다. 기존의 Swin Transformer (SwinT) 모델 위에 Convolutional Block Attention Module (CBAM)을 통합하여 특히 작은 사이즈의 결함에 대한 탐지 성능을 크게 향상시켰습니다.

- **Technical Details**: 제안한 프레임워크는 Swin Transformer 블록 내에 CBAM을 반복적으로 통합하여 작은 사례 크기의 결함(예: Dirt 및 Squat)의 탐지 성능을 높입니다. 실험 및 ablation 연구를 통해 이 프레임워크의 효과가 입증되었습니다. RIII 데이터셋의 Dirt 및 Dent 카테고리에서 각각 mAP-50이 +23.0% 및 +38.3% 증가하였으며, MUET 데이터셋의 Squat 카테고리에서는 +13.2% 개선을 보였습니다.

- **Performance Highlights**: CBAM-SwinT-BL은 원래 SwinT 모델에 비해 MUET 데이터셋에서 +5%, RIII 데이터셋에서 +7%의 전체 정밀도 향상을 이뤄냈습니다. 각각 69.1% 및 88.1%의 정확도를 기록했습니다. CBAM 모듈은 모델 훈련 속도를 평균 +0.04s/iteration만 연장하여, 시스템 성능의 획기적인 향상에 비해 수용 가능한 수준입니다.



New uploads on arXiv(cs.AI)

### Mimicking Human Intuition: Cognitive Belief-Driven Q-Learning (https://arxiv.org/abs/2410.01739)
Comments:
          Under review by ICLR 25

- **What's New**: 이 논문에서는 전통적인 Q-learning 알고리즘의 한계를 극복하기 위해 주관적인 신념 모델링을 통합한 Cognitive Belief-Driven Q-Learning (CBDQ) 방법론을 제안합니다. 이 방법은 의사결정의 정확성을 높이고, 인간과 유사한 학습 및 추론 능력을 에이전트에게 제공합니다.

- **Technical Details**: CBDQ는 (1) 주관적 신념 구성 요소, (2) 인간 인지 클러스터, (3) 신념-선호 결정 프레임워크(BPDF)를 통합하여 Q-learning의 성능을 향상시킵니다. 이를 통해 Q-learning의 과대 평가 문제를 해결하고, 환경의 상태 공간을 클러스터링하여 고차원 데이터를 의미 있는 저차원 표현으로 압축합니다. 이는 인간의 인지 방식을 모방하여 상태 표현을 개선합니다.

- **Performance Highlights**: CBDQ는 다양한 복잡한 환경에서 이전의 Q-learning 알고리즘들보다 높은 보상을 지속적으로 달성하며, 경쟁하는 알고리즘들과 비교하여 더 향상된 적응력과 강인성을 보여줍니다. 이 연구는 Q-learning에 대한 새로운 접근법을 제시하여 복잡한 결정 시스템에서의 인간과 유사한 에이전트를 향한 진전을 이룹니다.



### Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspectiv (https://arxiv.org/abs/2410.01720)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 후속 훈련 단계에서의 합성 데이터 생성과 일반화 능력 간의 관계를 체계적으로 분석합니다. 특히, 합성 데이터의 효과를 이론적으로 모델링하고, 정보 이득(Information Gain) 과 일반화 이득(Generalization Gain) 간의 새로운 개념인 GGMI(Generalization Gain via Mutual Information)를 소개합니다.

- **Technical Details**: 저자들은 합성 데이터 생성 과정을 분포 관점에서 모델링하고, 후속 훈련이 진행 중인 LLM에 미치는 합성 데이터의 영향을 분석하기 위해 역 병목 구조(Reverse Bottleneck Framework)를 제시합니다. 이 접근법은 합성 데이터가 LLM의 일반화 능력에 미치는 효과를 정량화할 수 있는 상한(Upper Bounds)을 제공합니다.

- **Performance Highlights**: 많은 최신 LLM들이 합성 데이터를 활용함으로써 훈련 성과를 개선하고 있으며, 이 논문은 합성 데이터를 통해 LLM의 성능과 신뢰성을 높일 수 있는 방법에 대한 통찰을 제공합니다. 이 연구는 특히 제한된 실제 데이터의 상황에서 LLM의 일반화 능력을 어떻게 향상시킬 수 있는지를 탐구하며, 합성 데이터의 설계 및 최적화 프로세스를 이해하는 데 중요한 기여를 합니다.



### CreDes: Causal Reasoning Enhancement and Dual-End Searching for Solving Long-Range Reasoning Problems using LLMs (https://arxiv.org/abs/2410.01696)
- **What's New**: 본 논문에서는 복합 최적화 문제를 처리하는 데 있어서 대형 언어 모델 (LLMs)의 한계를 극복하기 위한 새로운 접근 방식을 제시합니다. Causal Relationship Enhancement (CRE) 메커니즘과 Dual-End Searching (DES) 기법의 결합인 CreDes를 통해 모델의 성능을 개선하였습니다.

- **Technical Details**: CRE는 원인-효과 개입 (cause-effect interventions)과 개인 치료 효과 (Individual Treatment Effect, ITE)를 결합하여 추론 과정과 상태 전이 간의 강력한 인과 관계를 보장합니다. DES는 원래 상태와 목표 상태에서 동시에 시작하여 causal probability tree에서 해결책을 찾습니다. 이러한 접근을 통해 단일 방향 검색 (single-direction search)의 한계를 극복합니다.

- **Performance Highlights**: CreDes는 장기 추론 작업 (long-range reasoning tasks)에서 기존의 State-Of-The-Art (SOTA) 솔루션에 비해 정확성과 시간 효율성 모두에서 유의미한 성능 향상을 보여줍니다.



### U-shaped and Inverted-U Scaling behind Emergent Abilities of Large Language Models (https://arxiv.org/abs/2410.01692)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 큰 언어 모델(LLMs)의 성능이 질문의 난이도에 따라 달라지며, 이것이 emergent abilities(출현 능력)의 예측에 중요한 역할을 한다는 새로운 개념을 제안합니다. 질문의 난이도에 따라 U자 형태의 스케일링과 역 U자 형태의 스케일링을 관찰하고, 이를 통해 모델 성능의 급격한 향상을 예측할 수 있는 Slice-and-Sandwich라는 파이프라인을 제안합니다.

- **Technical Details**: 논문은 LLM의 성능을 질문 난이도에 따라 다르게 분석하는 과정을 설명합니다. 우선 질문을 난이도 기준으로 그룹화하고, emergence threshold(출현 임계값) 전후의 데이터로부터 성능을 각각 예측합니다. 여기서 사용되는 성능 지표로는 Brier Score와 binary Brier Score를 언급하며, 이들은 모델이 선택한 정답의 확률에 의존하는 지표입니다.

- **Performance Highlights**: Slice-and-Sandwich 파이프라인은 난이도에 따라 그룹화된 질문의 성능 예측을 통해 성능 급증을 효과적으로 포착합니다. MMLU, 산술 문제, Persian-QA 데이터셋에서 실험한 결과, 모델 성능이 출현 임계값을 넘어서는 주요 특징을 잘 설명하는 것으로 나타났습니다.



### Why context matters in VQA and Reasoning: Semantic interventions for VLM input modalities (https://arxiv.org/abs/2410.01690)
- **What's New**: 이번 연구는 Visual Language Model (VLM)의 예측에서 시각 및 텍스트 모달리티의 통합이 성능에 미치는 영향을 심층적으로 조사합니다.

- **Technical Details**: 우리는 SI-VQA 데이터셋과 다양한 모달리티 구성에서의 VLM 아키텍처 벤치마크 연구, 그리고 Interactive Semantic Interventions (ISI) 툴을 소개합니다. SI-VQA 데이터셋은 벤치마크의 기초를 제공하며, ISI 도구는 이미지와 텍스트 입력에서 의미 개입을 테스트하고 적용할 수 있는 인터페이스를 제공합니다.

- **Performance Highlights**: 연구 결과, 모달리티 간의 보완적인 정보가 답변 및 추론의 질을 향상시키며, 상반된 정보는 모델 성능 및 신뢰성을 저하시키는 것으로 나타났습니다. 또한, PaliGemma의 과도한 자신감은 LLaVA 모델보다 더 높은 침묵적 실패 위험을 내포하고 있습니다.



### Mind Scramble: Unveiling Large Language Model Psychology Via Typoglycemia (https://arxiv.org/abs/2410.01677)
- **What's New**: 이 논문은 LLM 심리학(LLM Psychology)이라는 새로운 연구 분야와 방법론을 소개합니다. 이는 인간 심리 실험을 활용하여 LLM의 인지 행동과 메커니즘을 조사합니다.

- **Technical Details**: Typoglycemia 현상(phenomenon)을 통해 LLM의 '마음'을 연구하였으며, 문자, 단어, 문장 수준에서의 Typoglycemia 실험을 통해 LLM들이 인간과 유사한 행동을 보인다고 보고합니다. LLM의 hidden layers를 분석하여 이러한 현상을 설명하려고 합니다.

- **Performance Highlights**: LLM들은 Typoglycemia 테스트에서 높은 계산 비용을 투입하면서도 혼란스러운 텍스트를 이해하는 능력을 보였습니다. 연구 결과는 LLM의 능력이 통계적이고 데이터 기반이라는 것을 시사하며, 이들의 인지 과정이 인간의 그것과 다르다는 강력한 증거를 제공합니다.



### Finding path and cycle counting formulae in graphs with Deep Reinforcement Learning (https://arxiv.org/abs/2410.01661)
- **What's New**: 이 논문은 Grammar Reinforcement Learning (GRL)이라는 새로운 강화 학습 알고리즘을 제안합니다. GRL은 Monte Carlo Tree Search (MCTS)와 transformer 아키텍처를 활용하여 context-free grammar (CFG) 내에서 Pushdown Automaton (PDA)을 모델링합니다. GRL은 그래프에서 경로와 사이클을 효율적으로 계산하는 문제를 해결하고, 기존 방법보다 두 배에서 여섯 배 향상된 새로운 행렬 기반 공식을 발견했습니다.

- **Technical Details**: GRL은 주어진 CFG 내에서 동작하는 gramformer를 생성하는 프레임워크를 제공합니다. GRL은 문법 구조 내에서 공식을 최적화하기 위한 방법으로 개발되었으며, 그래프 서브구조 카운팅을 위한 새로운 공식을 발견하여 계산 효율성을 크게 개선합니다.

- **Performance Highlights**: GRL은 기존의 11번 문헌에서 제공한 공식을 회복할 뿐만 아니라, 두 배에서 여섯 배의 계산 효율성을 가진 새로운 공식을 발견하였습니다. 논문에서는 GRL이 그래프에서 경로/사이클 카운팅 작업에 대해 상당한 컴퓨팅 성능 향상을 이루었음을 강조합니다.



### Iterated Local Search with Linkage Learning (https://arxiv.org/abs/2410.01583)
- **What's New**: 이 논문에서는 가중치를 고려한 변수 상호작용 그래프(Weighted Variable Interaction Graph, VIGw)를 구축하는 새로운 지역 검색 전략인 링크 학습 2(LSwLL2)를 제안합니다. 기존의 링크 학습 방법론은 상호작용의 강도를 반영하지 못했으나, 본 연구는 이 정보를 포함하여 최적화 문제의 본질과 최적화 알고리즘의 동작을 이해하는 데 도움을 줍니다.

- **Technical Details**: LSwLL2는 경험적 링크 학습을 기반으로 하며, 이는 결정 변수들 간의 비선형 상호작용 강도를 나타내는 가중치가 있는 비방향 그래프(VIGw)를 생성하는 방식을 사용합니다. 이 그래프는 다양한 최적화 문제를 탐색하는 데 유용한 정보를 제공합니다.

- **Performance Highlights**: NK 랜드스케이프, 배낭 문제 및 특성 선택 문제에 대한 실험 결과, LSwLL2가 효율적으로 VIGw를 구축하며, 특히 기계 학습 데이터셋에서 특성 간의 상호작용을 시각화할 수 있는 가능성을 보여줍니다. 새로운 변환 연산자는 지역 검색을 가속화하는 데 기여할 수 있습니다.



### MedQA-CS: Benchmarking Large Language Models Clinical Skills Using an AI-SCE Framework (https://arxiv.org/abs/2410.01553)
- **What's New**: 본 연구에서는 의료 교육에서 영감을 받은 MedQA-CS라는 AI-구조적 임상 시험(AI-SCE) 프레임워크를 도입하여 기존 임상 능력 평가의 한계를 극복하고자 합니다. 이 프레임워크는 LLMs(Large Language Models)가 제공하는 임상 시나리오에 대한 평가를 통해 실제 의료 환경에서의 기능을 평가합니다.

- **Technical Details**: MedQA-CS는 두 가지 지시 사항에 따른 작업(LLM-as-medical-student 및 LLM-as-CS-examiner)을 포함하며, 이는 임상 능력을 평가하기 위해 MedStuLLM 및 MedExamLLM의 두 가지 구성요소로 구성됩니다. OSCE(Objective Structured Clinical Examinations) 지침에 따라 LLM의 임상 능력을 'shows how' 수준에서 평가하는 것이 핵심적인 특징입니다.

- **Performance Highlights**: 실험 결과, MedQA-CS는 기존의 다지선다형 질문 기반 벤치마크에 비해 LLM의 임상 기술을 평가하는 데 더 도전적임을 보여주었습니다. 연구는 또한 LLM의 임상 기술 수행 능력과 관련된 흥미로운 통찰을 제공하며, LLM-as-Judge 프레임워크의 잠재력을 강조합니다.



### From Reward Shaping to Q-Shaping: Achieving Unbiased Learning with LLM-Guided Knowledg (https://arxiv.org/abs/2410.01458)
Comments:
          q-shaping, reinforcement learning, reward shaping

- **What's New**: 본 연구에서는 Q-value 초기화를 확장한 Q-shaping을 소개하며, 이를 통해 도메인 지식을 활용하여 에이전트 훈련을 가속화하고 샘플 효율성을 개선하는 방법을 제안합니다. Q-shaping은 기존의 보상 조정 방법과 달리 Q-value를 직접 수정하여 에이전트의 최적성을 보장합니다.

- **Technical Details**: Q-shaping은 다양한 작업에 대해 일반적이고 강력한 접근 방식을 제공하며, 대형 언어 모델(LLM)을 활용하여 휴리스틱 제공자로 사용합니다. 실험을 통해 Q-shaping이 가장 좋은 기준보다 평균 16.87%의 샘플 효율성 향상을 달성하였고, LLM 기반 보상 조정 방법과 비교하여 253.80%의 성능 향상을 보였습니다.

- **Performance Highlights**: Q-shaping은 20개의 서로 다른 환경에서 평가되었으며, 각 환경 관련 최상의 기준보다 16.87% 향상된 결과를 기록했습니다. 또한 기존 LLM 기반 보상 조정 방식인 T2R 및 Eureka와 비교하여 최적성에서 253.80%의 성능 손실을 경험하는 결과를 보여 주었습니다.



### Improving Fuzzy Rule Classifier with Brain Storm Optimization and Rule Modification (https://arxiv.org/abs/2410.01413)
Comments:
          9 pages,8 figures

- **What's New**: 본 연구는 당뇨 분류 문제를 해결하기 위해 Brain Storm Optimization (BSO) 알고리즘을 활용하여 기존의 퍼지 시스템을 혁신적으로 재정의한 방법을 제시합니다. 특히, 당뇨 관련 데이터에 맞춘 새로운 규칙 생성 모델을 통합하였으며, 이는 분류 작업에 있어 상당한 정확도 향상을 가져옵니다.

- **Technical Details**: BSO 알고리즘에 지수 이동 평균 Exponential Weighted Moving Average (EWMA) 모델을 통합하여 규칙 생성 및 정제를 진행하며, 각 속성에 대한 멤버십 함수의 수를 평가하여 필요한 규칙 수를 결정합니다. 이 과정은 규칙 표현을 보다 직관적으로 만들어주며, 입력 데이터의 동응성을 높이기 위해 동적 제약 조건을 설정합니다.

- **Performance Highlights**: 다양한 실험 결과, 제안된 퍼지 분류 시스템은 Adaptive Generalized Fuzzy System (AGFS)과 비교하여 당뇨 감지 정확도가 크게 향상되었으며, 감도(sensitivity), 특이도(specificity) 및 안정성(stability)에서 우수한 성능을 보였습니다.



### Theoretical Lower Bounds for the Oven Scheduling Problem (https://arxiv.org/abs/2410.01368)
Comments:
          arXiv admin note: text overlap with arXiv:2203.12517

- **What's New**: 본 논문에서는 반도체 산업에서 발생하는 NP-hard 실세계 병렬 배치 스케줄링 문제인 Oven Scheduling Problem (OSP)을 다룹니다. 우리는 OSP를 위한 이론적이고 문제 특화된 하한 (lower bounds)을 신속하게 계산하는 절차를 개발하였으며, 이 하한의 품질을 평가하고 기존 솔루션 방법에 통합하는 방법을 탐구하였습니다.

- **Technical Details**: OSP는 다양한 제약 조건, 예를 들어 오븐의 자격과 가용성, 작업 시작일, 배치 간 설정 시간, 오븐 용량 제한 등을 준수하면서 특히 오븐에서 작업의 총 실행 시간, 작업 지연 및 설정 비용을 최소화하는 것을 목표로 합니다. 우리는 호환 가능한 작업을 함께 묶어서 배치로 처리하는 접근 방식을 사용하며, OSP의 목적을 위한 하한을 계산하는 과정을 제시합니다.

- **Performance Highlights**: 상대적으로 큰 인스턴스에 대해 계산된 하한은 최적 해의 값과 작은 차이를 제공하며, 상업용 솔버가 제공하는 하한을 종종 초과하는 성능을 보여줍니다. 우리의 실험 결과, 많은 기준 인스턴스가 평균 15초 이내에 해결되었으며, 이로 인해 기존 솔루션 방법에 도움을 줄 수 있는 가능성이 있음을 입증하였습니다.



### Life, uh, Finds a Way: Systematic Neural Search (https://arxiv.org/abs/2410.01349)
Comments:
          26 pages, 5 figures

- **What's New**: 이 논문은 새로운 환경에서 동적인 문제를 해결하기 위해 에이전트의 행동을 신속하게 적응시키는 방법을 제안합니다. 특히, 생물체가 새로운 환경에 적응하는 능력을 기계 시스템이 흉내낼 수 없다는 문제를 해결하는 데 집중합니다.

- **Technical Details**: 행동을 검색 절차의 물리적 표현으로 간주하고, 행동을 검출하는 신경 알고리즘을 제안합니다. 이를 위해 Hebbian learning과 entorhinal cortex에서 영감을 받은 새로운 고차원 조화 표현을 사용하는 방법을 설명합니다. 기존의 인공 지능 접근 방식과 유사하지만, 더 유연한 방식으로 행동을 열거하고 이를 통해 실시간으로 행동 적응을 수행하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 연속 상태 공간 내비게이션 문제를 해결하는 데 성공하며, 자율 로봇이 데이터가 sparse한 환경에서 복잡한 기술을 마스터할 수 있는 길을 열어줍니다. 실험 결과는 행동 적응의 실질성 및 효율성을 보여줍니다.



### FanCric : Multi-Agentic Framework for Crafting Fantasy 11 Cricket Teams (https://arxiv.org/abs/2410.01307)
- **What's New**: 이번 연구는 인도의 주요 판타지 크리켓 리그인 Dream11를 중심으로, 이 리그에서의 팀 선택을 향상시키기 위한 새로운 프레임워크인 FanCric를 소개합니다. FanCric는 대규모 언어 모델(LLMs)을 활용하여 판타지 팀 구성에서의 전략적 의사결정을 지원합니다.

- **Technical Details**: FanCric 프레임워크는 다중 에이전트 시스템(multi-agent system)을 기반으로 하며, 구조화된 데이터와 비구조화된 데이터를 모두 활용하여 기존의 방법들을 능가하는 성과를 보여줍니다. 이 연구는 약 1270만 개의 고유한 Dream11 참가자를 분석하여 FanCric의 효율성을 평가했습니다.

- **Performance Highlights**: 분석 결과, FanCric의 판타지 팀 선택은 군중의 집단 지성과 간단한 Prompt Engineering 방법보다 더 나은 성과를 보이며, 추가 연구를 통해 LLMs를 활용한 전략적 의사결정의 잠재력을 더욱 개발할 필요성이 제기되었습니다.



### Towards a Law of Iterated Expectations for Heuristic Estimators (https://arxiv.org/abs/2410.01290)
Comments:
          47 pages, 2 tables, 1 figure

- **What's New**: 본 논문에서는 *heuristic estimator*의 개념을 정립하고, ideal heuristic estimator가 만족해야 하는 특성으로 *iterated estimation*과 *error orthogonality*를 제안합니다.

- **Technical Details**: 이 접근법은 수학적 표현 Y, 정식의 'heuristic argument' π를 입력으로 받아 Y에 대한 추정치를 출력하는 알고리즘인 heuristic estimator \( \mathbb{G}(Y | \pi) \)를 다룹니다. 또한, '정확도(accuracy)' 개념을 도입하여 강력한 추정치를 생성하는 데 있어 직면하는 장벽을 분석합니다.

- **Performance Highlights**: 이 연구는 heuristic estimator가 오류를 예측하지 못해야 한다는 비공식 원칙을 주장하며, 특정 수학적 표현들의 분포에 걸쳐 평균 오류가 제로가 되는 것을 요구합니다.



### Uncertainty-aware Human Mobility Modeling and Anomaly Detection (https://arxiv.org/abs/2410.01281)
- **What's New**: 이번 논문에서는 레이블이 없는 데이터를 활용하여 인간의 이동 행동 모델링 및 이상 탐지를 효과적으로 수행할 수 있는 UIFormer라는 새로운 방법론을 제안합니다.

- **Technical Details**: UIFormer는 이중(dual) Transformer 아키텍처를 기반으로 하며, aleatoric(데이터) 불확실성과 epistemic(모델) 불확실성을 모두 고려하여 인간 행동 모델링 및 이상 탐지를 수행합니다. 이 방법은 복잡한 시공간 의존성을 캡처하고 데이터 희소성 문제를 해결하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, UIFormer는 수만 개의 에이전트를 포함하는 대규모 시뮬레이션 데이터셋에서 기존의 예측 및 이상 탐지 기준선보다 뛰어난 성능을 보였습니다.



### Generative Diffusion-based Contract Design for Efficient AI Twins Migration in Vehicular Embodied AI Networks (https://arxiv.org/abs/2410.01176)
- **What's New**: 이번 논문에서는 차량 통합 인공지능 네트워크(VEANET) 내에서 자율주행차(AV)를 위한 'Embodied AI Twins'라는 새로운 개념을 소개합니다. 이는 현실 세계의 자율 주행 기능을 지원하기 위해 생성형 AI 모델을 사용하는 디지털 쌍둥이입니다.

- **Technical Details**: 본 연구는 AV와 도로 옆 장치(RSU) 간의 다차원 계약 이론 모델을 구축하여 정보 비대칭 문제를 해결합니다. AV의 비이성적 행동을 다루기 위해 기대 효용 이론 대신 전망 이론(Prospect Theory)을 사용하고, 생성적 확산 모델(Generative Diffusion Model)을 활용하여 최적의 계약 설계를 식별합니다.

- **Performance Highlights**: 수치 결과에서 제안된 생성적 확산 모델 기반 알고리즘이 전통적인 딥 강화 학습(deep reinforcement learning) 알고리즘보다 효율성이 뛰어난 것으로 나타났습니다.



### Learning to Build by Building Your Own Instructions (https://arxiv.org/abs/2410.01111)
- **What's New**: 이 논문에서는 LEGO 조립을 위한 Break-and-Make 문제를 해결하기 위해 InstructioNet이라는 새로운 모델을 제안합니다. 이 모델은 에이전트가 LEGO 모델을 분해하고 이미지를 저장하여 스스로 시각적 지침을 생성할 수 있게 합니다.

- **Technical Details**: InstructioNet은 에이전트가 모델을 분해하면서 생성된 이미지 스택을 사용하여, 단계별로 조립 과정을 이해하고 실행할 수 있는 구조적 메모리를 활용합니다. 또한, 이 모델은 온라인 imitation learning(모사 학습)을 통해 에이전트가 실수를 통해 배울 수 있도록 합니다.

- **Performance Highlights**: 제안된 InstructioNet 모델은 기존의 기준 모델보다 더 큰 LEGO 조립 문제에서 훨씬 우수한 성능을 보이며, 새로운 RC-Vehicles 데이터셋에 대해 훈련되었습니다. 이 데이터셋은 평균 31개의 블록으로 구성되어 있으며, 조립 및 분해에 100단계를 요구합니다.



### Mixing It Up: The Cocktail Effect of Multi-Task Fine-Tuning on LLM Performance -- A Case Study in Financ (https://arxiv.org/abs/2410.01109)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 도메인 특화 적용이 빠르게 성장하고 있으며, 특히 금융 분야에서 이들의 성능 평가에 대한 필요성이 커지고 있습니다. 본 연구에서는 LLMs의 미세 조정에서 멀티태스크 파인튜닝이 더 효과적일 수 있음을 보여주고 있으며, 작은 모델이 더 큰 모델을 초월할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 본 연구에서는 200개 이상의 모델을 학습시키며, 다양한 LLMs를 기준으로 삼아 멀티태스크 파인튜닝의 효과를 조사했습니다. 특히, 관련 기술 작업을 함께 훈련시키는 것이 모델 성능을 향상시킬 수 있는 시너지 효과를 가져온다는 점을 강조하였습니다. 또한, 일반적인 수학적 데이터와 지침 데이터를 포함하여 훈련 과정에서 모델 성능 개선을 위한 정규화 역할을 수행하는 가능성도 탐구하였습니다.

- **Performance Highlights**: Phi-3-Mini 모델이 GPT-4-o 모델을 초월하는 뛰어난 성능을 보였으며, 재무 기초 평가에서 최첨단 결과를 달성하였습니다. 본 연구의 결과는 도메인 특화 작업에서의 멀티태스크 파인튜닝의 중요성과 효과를 강조하며, 단일 작업에 대한 미세 조정이 반드시 도메인 지식의 넓은 향상으로 이어지지는 않는다는 점도 시사합니다.



### Generative AI Application for Building Industry (https://arxiv.org/abs/2410.01098)
Comments:
          28 pages, 11 figures, 4 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)과 같은 생성적 AI 기술의 건축 산업에서의 혁신적 가능성을 조사합니다. 이 기술들이 에너지 규정 준수, 건물 설계 최적화, 인력 교육 등 여러 분야에서의 응용을 탐구합니다.

- **Technical Details**: 이 연구는 LLMs가 노동 집약적인 프로세스를 자동화하여 건축 관행에서 효율성, 정확성 및 안전성을 크게 향상시킬 수 있는 방법을 강조합니다. 또한, 복잡한 시각적 및 텍스트 데이터 해석에 따른 도전 과제를 다루고, AI 기반 규정 준수 검사 및 설계 프로세스를 향상시키기 위한 혁신적인 솔루션을 제안합니다.

- **Performance Highlights**: AI 통합의 광범위한 함의에 대해 고려하고, 다양한 규제 영역에서 포괄적인 규정 준수를 위한 AI 기반 도구 개발과 현실적인 시뮬레이션을 통한 인력 교육 혁신 가능성을 탐구합니다.



### Truth or Deceit? A Bayesian Decoding Game Enhances Consistency and Reliability (https://arxiv.org/abs/2410.01064)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 출력 일관성과 신뢰성을 향상시키기 위해 새로운 게임 이론적 접근 방식을 제안합니다. 특히, 생성 단계에서의 모호함을 해소하고 인간의 의도와 출력의 정확성을 정렬하기 위한 멀티 스테이지 Bayesian Decoding Game을 도입합니다.

- **Technical Details**: 제안된 방법은 올바른 출력과 신뢰할 수 없는 출력을 구분하는데 기여하며, 이를 통해 LLM의 해석 가능성을 높입니다. 저자는 Correctness Alignment와 Ambiguity Calibration을 통해 출력의 일관성을 보장하고 신뢰성을 강화하며, 게임 메커니즘을 통해 작은 모델이 큰 모델보다 높은 성과를 낼 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 78.1의 성능을 보이는 LLaMA 13B 모델이 76.6을 기록한 PaLM 540B 모델을 능가하는 등 게임 이론적 도구들이 LLM의 신뢰성과 진실성을 향상시킬 수 있음을 나타냅니다.



### RATIONALYST: Pre-training Process-Supervision for Improving Reasoning (https://arxiv.org/abs/2410.01044)
Comments:
          Our code, data, and model can be found at this repository: this https URL

- **What's New**: 이 논문에서는 LLMs의 추론에서 발생하는 불완전성을 해결하기 위해 RATIONALYST라는 새로운 모델을 제안합니다. 이 모델은 대규모 무표시 데이터에서 추출한 합리적 근거(rationale) 주석을 기반으로 하는 프로세스 감독(process supervision) 방식으로 학습됩니다.

- **Technical Details**: RATIONALYST는 LLaMa-3-8B에서 미세 조정(fine-tuning)되어 79,000개의 합리적 근거를 웹 스케일 무표시 데이터와 최소한의 인적 개입으로 추출한 데이터 집합에서 활용합니다. 이 모델은 연구 결과에서 수학, 상식, 과학, 논리적 추론을 포함한 다양한 추론 작업에서 일반화할 수 있는 능력을 보입니다.

- **Performance Highlights**: RATIONALYST는 대표적인 7개 추론 벤치마크에서 평균 3.9% 향상된 추론 정확도를 보여주며, GPT-4와 같은 훨씬 큰 검증자(compared to significantly larger verifiers) 모델들 및 유사한 크기의 모델과 성능을 비교하여 우수성을 입증하였습니다.



### A Knowledge-Informed Large Language Model Framework for U.S. Nuclear Power Plant Shutdown Initiating Event Classification for Probabilistic Risk Assessmen (https://arxiv.org/abs/2410.00929)
- **What's New**: 본 논문은 원자력 발전소의 저전력 폐쇄 확률적 위험 평가를 위한 종료 유발 사건(SDIE, Shutdown Initiating Events) 식별 및 분류 방법을 제안합니다. 기존 방법의 한계를 극복하기 위해 지식에 기반한 머신러닝 모델과 대형 언어 모델(LLM, Large Language Model)의 하이브리드 파이프라인을 통합한 방안을 제시합니다.

- **Technical Details**: 제안된 파이프라인은 두 단계로 구성됩니다. 첫 번째 단계는 44개의 SDIE 텍스트 패턴을 사용한 사전 선별로, 이는 여섯 가지 SDIE 유형에서 도출된 주요 키워드와 구문으로 구성됩니다. 이 패턴을 기반으로 한 텍스트 벡터화는 간단한 이진 분류기를 사용하여 매우 구분 가능한 피처 벡터를 생성합니다. 두 번째 단계는 Bidirectional Encoder Representations from Transformers (BERT) 기반 LLM을 구축하여, 대규모 데이터셋에서 자가 지도 학습 방식으로 일반적인 영어 언어 표현을 학습하고, 이를 SDIE 분류에 맞게 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: 실험 결과, 사전 선별 단계에서 97% 이상의 비SDIE를 제외할 수 있었으며, LLM을 사용한 SDIE 분류에서 평균 정확도는 93.4%에 달했습니다.



### Samba: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking (https://arxiv.org/abs/2410.01806)
- **What's New**: 본 논문에서는 Samba라는 새로운 linear-time set-of-sequences 모델을 소개하며, 이를 통해 여러 tracklet들을 동기화된 상태로 처리하여 복잡한 이동 패턴과 상호작용을 모델링하는 방법을 제안합니다. 또한, SambaMOTR라는 최초의 tracking-by-propagation 트래커를 통해 이전의 문제점들을 해결하고, MaskObs라는 기술을 도입하여 불확실한 관측치를 효과적으로 처리하는 방법을 제공합니다.

- **Technical Details**: Samba는 복수의 time-series 데이터 (tracklets)를 동기화된 긴 메모리 표현으로 압축하여 처리합니다. 이 과정에서 self-attention mechanism을 통해 tracklet들 간의 정보를 교환하면서 interdependencies를 고려합니다. SambaMOTR는 이를 기반으로 하여, autoregressive 방식으로 다음 track query를 예측하며, occlusion(가림현상) 문제를 효과적으로 처리하는 새로운 쿼리 전파 모듈을 탑재하고 있습니다.

- **Performance Highlights**: SambaMOTR는 DanceTrack, BFT, SportsMOT 데이터셋에서의 성능을 크게 향상시켜 새로운 state-of-the-art를 기록했습니다. 특히, SambaMOTR는 tracklet 간의 상호작용과 긴-range dependencies를 정확하게 모델링하여 가림현상에서도 효과적으로 객체를 추적할 수 있는 능력을 보여주었습니다.



### FabricDiffusion: High-Fidelity Texture Transfer for 3D Garments Generation from In-The-Wild Clothing Images (https://arxiv.org/abs/2410.01801)
Comments:
          Accepted to SIGGRAPH Asia 2024. Project page: this https URL

- **What's New**: FabricDiffusion은 단일 의류 이미지를 3D 의상에 매핑하여 섬유 질감을 이전하는 새로운 방법을 제안합니다. 이 방법은 기존의 2D-3D 질감 매핑 및 깊이 인식을 통한 이미지 보정 기법의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: FabricDiffusion은 대규모 합성 데이터 세트를 기반으로 훈련된 디노이징 확산 모델을 이용하여 입력 텍스처 이미지의 왜곡을 수정하고, 이를 통해 고품질의 질감 맵을 생성합니다. 이 기술은 PBR(Physically-Based Rendering) 방법론과 긴밀하게 결합되어 다양한 조명 조건에서 의상을 사실적으로 재조명할 수 있습니다.

- **Performance Highlights**: FabricDiffusion은 합성 데이터와 실제 의류 이미지 모두에서 최첨단 기법을 능가하는 성능을 보이며, 이전에 보지 못한 질감과 의상 형태에 대해서도 일반화하는 능력을 보여주었습니다.



### Windowed MAPF with Completeness Guarantees (https://arxiv.org/abs/2410.01798)
- **What's New**: 본 논문은 기존의 Windowed Multi-Agent Path Finding (MAPF) 방법들이 지닌 불완전성을 해결하기 위한 새로운 프레임워크인 WinC-MAPF를 소개합니다. 이 프레임워크는 완전성을 보장하며, 기존의 충돌 회피 경로 계산 방식을 개선합니다.

- **Technical Details**: WinC-MAPF는 단일 에이전트(Agent) 실시간 탐색 알고리즘의 휴리스틱 업데이트 통찰력과 MAPF 알고리즘의 에이전트 독립성을 활용합니다. 또한, Single-Step CBS (SS-CBS)를 개발하여 단일 단계만 계획하고 휴리스틱을 업데이트하는 방식으로 난이도 높은 상황을 해결할 수 있도록 합니다.

- **Performance Highlights**: SS-CBS는 기존의 대형 타임 창을 갖춘 ECBS보다 작은 인스턴스와 큰 인스턴스 모두에서 효과적으로 문제를 해결하는 성능을 보여주었습니다. 이는 SS-CBS가 단일 단계 계획을 통해 비효율적인 경로를 피할 수 있도록 돕기 때문입니다.



### When a language model is optimized for reasoning, does it still show embers of autoregression? An analysis of OpenAI o1 (https://arxiv.org/abs/2410.01792)
Comments:
          6 pages

- **What's New**: 이 논문에서는 OpenAI의 새로운 시스템 o1이 이전의 대형 언어 모델(LLMs)을 어떻게 개선했는지를 탐구합니다. 특히, o1은 추론을 위해 최적화되어 있어, 일반적인 작업에서 드문 변형에서 특히 큰 성능 향상을 보여 줍니다.

- **Technical Details**: o1은 Chain of Thought(사고의 연쇄) 기법을 사용하여 복잡한 문제를 단계적으로 해결하는 방식으로 훈련됩니다. 또한, o1은 높은 확률의 출력 예시에서 더 좋은 성능을 보이며, 공통 작업 변형보다 드문 작업 변형에서 성능 차이가 덜 Pronounced(두드러지지 않음)합니다. 이 모델은 또한 'thinking tokens'를 생성하여 과제를 수행하는 방식을 수량화합니다.

- **Performance Highlights**: o1은 높은 확률의 출력에서 92% 정확도를 기록하며, 드문 작업 변형에 대해서도 상당한 성능 향상을 보입니다. 그러나, 여전히 예전 LLMs와 마찬가지로 출력 확률에 민감성을 보이며, 저확률 예시에서 더 많은 'thinking tokens' 을 사용합니다.



### DreamGarden: A Designer Assistant for Growing Games from a Single Promp (https://arxiv.org/abs/2410.01791)
Comments:
          21 pages + appendix, 11 figures

- **What's New**: 본 논문에서는 게임 디자인에서 사용될 수 있는 DreamGarden이라는 AI 시스템을 제안합니다. 이 시스템은 사용자가 제공하는 초기 프롬프트를 기반으로 고차원적인 계획을 수립하고, 이를 분류하여 구체적인 실행 계획을 제시하는 LLM 기반의 플래너를 사용합니다.

- **Technical Details**: DreamGarden은 Unreal Engine을 활용하여 다양한 게임 환경 개발을 지원하는 반자율적인 AI 도구입니다. 사용자의 꿈이나 상상을 프롬프트로 제공하면, 이 시스템은 이를 계층적인 작업 계획으로 분해하여 전문적인 하위 모듈로 분배합니다. 이를 통해 사용자는 계획의 성장 및 사용자 개입을 통한 피드백을 통해 변화하는 '계획의 정원'을 경험하게 됩니다.

- **Performance Highlights**: 사용자 연구를 통해 DreamGarden의 사용자가 자연어 프롬프트를 3D 시뮬레이션 환경으로 변환할 수 있는지, 그리고 이 시스템의 계층적이고 반복적인 과정이 사용자가 직관적으로 접근할 수 있는지를 평가했습니다. 결과적으로 사용자 인터페이스는 사용자가 다양한 수준에서 개입할 수 있는 충분한 기능을 제공하는 것으로 나타났습니다.



### Investigating on RLHF methodology (https://arxiv.org/abs/2410.01789)
Comments:
          23 pages, 6 figures, 6 tables

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 인간 선호도에 맞춘 정렬(alignment) 방법을 탐구합니다. 다중 레이어 학습과 강화 학습(Reinforcement Learning)을 통해 LLM의 성능을 개선하는 다양한 접근 방식을 제안합니다.

- **Technical Details**: Preference Model을 훈련하기 위해 필요한 데이터셋을 수집하고, 이를 통해 LLM을 인간 선호도에 맞게 조정합니다. 논문에서는 Direct Preference Optimization(DPO) 방법을 사용하여 별도의 Preference Model 없이 LLM을 직접 선호도 데이터셋으로 학습시키는 경험을 공유합니다.

- **Performance Highlights**: 연구진들은 퍼플렉서티 필터링(perplexity filtering)을 통해 데이터셋 수집 과정을 보다 용이하고 비용 효과적으로 만들 수 있다고 주장하며, 이를 통해 LLM을 보다 적합하게 산출할 수 있는 가능성을 제시합니다.



### Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models (https://arxiv.org/abs/2410.01782)
Comments:
          Accepted to EMNLP 2024 Findings. Website: this https URL. 14 pages, 7 figures, 5 tables

- **What's New**: 본 논문에서는 Open-RAG라는 새로운 프레임워크를 제안하여 여러 오픈소스 LLM과 함께 RAG를 활용한 추론 능력을 향상시키고자 합니다. 이 프레임워크는 복잡한 추론 작업을 처리할 수 있는 파라미터 효율적인 희소 MoE 모델로의 변환을 통해 성능을 높입니다.

- **Technical Details**: Open-RAG은 기존의 LLM을 기반으로 하여 모델이 자체 반성을 통해 레프리케이션(retrieval)과 관련된 특수 토큰을 생성하면서, 복잡한 질문에 대한 추론을 보다 효과적으로 수행할 수 있도록 학습합니다. 또한, 혼합 적응형 검색(hybrid adaptive retrieval) 방법을 도입하여 성능 향상과 추론 속도의 균형을 맞춥니다.

- **Performance Highlights**: Llama2-7B 기반의 Open-RAG는 ChatGPT, Self-RAG 및 Command R+와 같은 최첨단 LLM 및 RAG 모델을 초월하는 성과를 보이며, 다양한 지식 집약적 작업에서 새로운 벤치마크를 설정하였습니다. 실험 결과, Open-RAG가 선행 오픈소스 RAG 모델에 비해 의미 있는 정확도 향상과 추론 능력 개선을 보여주었습니다.



### Composing Global Optimizers to Reasoning Tasks via Algebraic Objects in Neural Nets (https://arxiv.org/abs/2410.01779)
- **What's New**: 이 논문에서는 2층 신경망의 솔루션 공간 내의 풍부한 대수 구조를 입증하고, 이는 쿼드러틱(Quadratic) 활성화 함수와 $L_2$ 손실을 바탕으로 하는 이유 추론(task) 문제에서의 학습에 적용됩니다. 이러한 구조는 손실의 일부만 만족하는 부분 솔루션으로부터 전역 최적 솔루션(global optimal solution)을 분석적으로 구성할 수 있게 해줍니다. 이 프레임워크는 CoGO(Composing Global Optimizers)라고 명명되었습니다.

- **Technical Details**: 연구에서는 2층 신경망의 가중치 공간이 반환환(semi-ring) 대수 구조를 갖고 있으며, 손실 함수가 단항 가능성(monomial potentials)으로 구성되어 있다는 점을 명시했습니다. 이러한 단항 가능성은 환 동형사상(ring homomorphism)으로, 부분 솔루션을 환 덧셈과 곱셈을 통해 전역 솔루션으로 구성할 수 있도록 합니다. 또한, 경험적으로 얻은 솔루션의 약 95%가 이론적으로 예측된 구조와 정확히 일치함을 보여주었습니다.

- **Performance Highlights**: 이론적 분석에 따르면, 높은 차수의 전역 최적화기는 훈련 동역학(training dynamics)에 불리하며, 과도한 매개변수화(over-parameterization)가 훈련 동역학을 비독립적으로 만들어 성능을 개선하는 경향이 있어 고차 전역 최적화기를 선호하지 않음을 보였습니다. 이 연구는 모델 학습 내에서 대수적 구조를 발견하고, 이를 통해 모듈라 덧셈과 같은 추론 문제에 대한 솔루션을 분석할 수 있는 첫 사례로 평가됩니다.



### DeFine: Enhancing LLM Decision-Making with Factor Profiles and Analogical Reasoning (https://arxiv.org/abs/2410.01772)
- **What's New**: 이 논문에서는 LLMs (Large Language Models) 의사결정 과정에서 복잡한 시나리오를 다루기 위해 DeFine 프레임워크를 제안합니다. 이 프레임워크는 복잡한 시나리오에서 확률적 요인 프로필을 구축하고, 유사한 과거 경험으로부터 통찰력을 활용하여 LLMs가 새로운 상황에서 중요한 결정을 내릴 수 있도록 지원합니다.

- **Technical Details**: DeFine은 분석의 기초로 보통 문법이 틀리거나 불완전한 문장을 포함한 긴 스크립트를 사용합니다. 이 스크립트에서 중요한 정보를 요약해 여러 요인 세트로 정리하고 각 요인의 잠재적 결과에 대한 확률을 추정합니다. Bradley-Terry 모델을 활용하여 결정적인 요인을 식별하고 이들이 의사결정에 미치는 영향을 평가합니다. 이 연구에서는 또한 유사한 상황 간의 연결성을 파악하기 위해 유사 추론(analogical reasoning)을 통합하여 LLMs의 의사결정을 돕습니다.

- **Performance Highlights**: 본 연구는 투자자들이 주식 이동 추세를 예측할 수 있는 통찰력을 제공하며, 이는 주로 순자산 증감 및 관련 지표 분석을 통해 이루어집니다. DeFine의 접근법은 금융 분야를 넘어 의료 상담 및 정치적 논쟁과 같은 복잡한 문제들을 다루는 분야에서도 활용될 가능성이 높습니다.



### VitaGlyph: Vitalizing Artistic Typography with Flexible Dual-branch Diffusion Models (https://arxiv.org/abs/2410.01738)
Comments:
this https URL

- **What's New**: VitaGlyph라는 새로운 위스타일화 기법이 개발되었습니다. 이 기법은 입력 문자(예: ‘장미’)를 주제(Subject)와 주변(Surrounding)으로 나누어 읽기 쉬운 예술적 타이포그래피를 생성할 수 있도록 합니다.

- **Technical Details**: VitaGlyph는 세 단계에서 작동합니다: (1) Knowledge Acquisition: 대형 언어 모델을 사용하여 주제와 주변에 대한 텍스트 설명을 설계합니다. (2) Regional Decomposition: Grounding-DINO를 사용하여 주제 설명과 가장 잘 일치하는 이미지를 찾아 입력 글리프 이미지를 주제와 주변 지역으로 분해합니다. (3) Typography Stylization: 주제 구조를 Semantic Typography로 개선하고, 별도로 텍스처를 렌더링합니다.

- **Performance Highlights**: VitaGlyph는 예술성과 가독성을 동시에 향상시킴을 입증하며, 여러 개의 맞춤형 개념을 포함하여 보다 창의적이고 매력적인 예술적 타이포그래피 생성이 가능합니다.



### Evaluating Robustness of Reward Models for Mathematical Reasoning (https://arxiv.org/abs/2410.01729)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 RewardBench의 한계점을 지적하고, 수학적 추론(Reasoning) 작업에서 리워드 모델(Reward Model)의 신뢰성 있는 평가를 위한 새로운 벤치마크인 RewardMATH를 소개합니다.

- **Technical Details**: RewardMATH는 리워드 모델의 견고성(Robustness)을 효과적으로 나타내기 위해 설계되었습니다. 기존 리워드 모델은 선택된 완료(Completion)와 거부된 완료(Rejected Completion) 간의 차이를 충분히 나타내지 못하는 단일 비교에 의존하고 있습니다.

- **Performance Highlights**: RewardMATH에서의 점수는 최적화된 정책(Optimized Policy)의 결과와 강한 상관관계를 보여주며, 기존 벤치마크는 거의 상관관계가 없음을 보여줍니다. 이는 평가의 신뢰성을 높이고, 리워드 모델의 견고성을 잘 나타내는 잠재력을 강조합니다.



### Auto-Demo Prompting: Leveraging Generated Outputs as Demonstrations for Enhanced Batch Prompting (https://arxiv.org/abs/2410.01724)
- **What's New**: 이번 논문에서는 'Auto-Demo Prompting'이라는 새로운 배치 프롬프트 기법을 제안하여, 배치 프롬프트의 성능 저하 문제를 해결합니다. 이 방법은 배치 내의 이전 질문에서 생성된 질문-답변 쌍을 활용하여 후속 질문의 답변을 추론합니다.

- **Technical Details**: Auto-Demo Prompting은 autoregressive generation 과정에서 작동하며, 이전 출력 결과를 활용하여 모델의 내부 표현을 최적화합니다. 질문-답변 쌍을 자동으로 인식하여 후속 질문에 대한 데모로 사용하고, 동일한 구조의 질문에서 발생할 수 있는 성능 저하를 완화하는데 중점을 둡니다.

- **Performance Highlights**: 실험 결과는 Auto-Demo Prompting이 전통적인 배치 프롬프트보다 뛰어난 성능을 보였음을 보여주며, 단일 프롬프트와 비교했을 때도 효율적이고 해석 가능한 방식으로 모델 성능을 크게 향상시켰습니다.



### Performant, Memory Efficient and Scalable Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2410.01706)
- **What's New**: Sable이라는 새로운 알고리즘을 소개하며, Retentive Networks에서의 retention 메커니즘을 다중 에이전트 강화 학습(MARL)로 적응시키고 메모리 효율성과 확장성을 동시에 달성하는 방법을 제시합니다.

- **Technical Details**: Sable은 retention 기반의 시퀀스 모델링 아키텍처를 이용하여 수천 개의 에이전트로 확장 가능하며, 긴 시간적 컨텍스트(temporal context)를 유지할 수 있습니다. 또한 대규모 부분 관찰 환경(partially observable environments)에서 효과적으로 작동합니다.

- **Performance Highlights**: Sable은 45개 과제 중 34개에서 기존의 최첨단 기법(state-of-the-art)보다 현저히 높은 성능을 보이며, 1000명 이상의 에이전트를 처리하면서 메모리 사용량은 선형적으로 증가합니다.



### From Prohibition to Adoption: How Hong Kong Universities Are Navigating ChatGPT in Academic Workflows (https://arxiv.org/abs/2410.01695)
- **What's New**: 이 논문은 홍콩 대학들이 ChatGPT를 금지하던 시기와 현재의 통합된 학술 과정의 변화를 비교하고 있습니다. 이제는 AI 기술이 학문에 통합되면서 윤리적 문제와 무결성에 대한 우려를 바탕으로 기관들이 AI 교육 및 책임 정책을 도입하는 방향으로 나아가고 있습니다.

- **Technical Details**: 이 연구는 Generative AI(생성적 인공지능)의 교육 분야에서의 활용과 윤리적 사용에 대한 새로운 패러다임을 조사합니다. 특히, 학술적 무결성(Academic Integrity)과 AI 리터러시(AI Literacy) 정책의 필요성을 강조하고 있습니다.

- **Performance Highlights**: 이 논문은 새로운 정책과 교육 프로그램들이 학계에 미치는 긍정적 영향을 조명하며, AI 통합(AI Integration)으로 인한 부정적 효과 방지를 위한 접근 방식을 제안합니다.



### FactAlign: Long-form Factuality Alignment of Large Language Models (https://arxiv.org/abs/2410.01691)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 이 논문에서는 LLM의 긴형 반응의 사실성을 향상시키기 위한 새로운 정렬 프레임워크인 FactAlign을 제안합니다. 해당 프레임워크는 LLM의 유용성을 유지하면서 사실성을 높이는 것을 목표로 합니다.

- **Technical Details**: FactAlign은 fKTO라는 세부 조정 문장 정렬 알고리즘을 도입하며, 이는 Kahneman-Tversky Optimization (KTO) 방법을 확장하여 문장 수준의 정렬을 가능하게 합니다. 이 알고리즘은 자동 사실성 평가의 최신 발전을 활용하여 세부적인 사실성 평가를 통해 정렬 프로세스를 안내합니다.

- **Performance Highlights**: 실험 결과, FactAlign은 LLM의 긴형 반응의 사실성을 유의미하게 향상시켰으며, 동시에 유용성을 개선하는 데도 기여했습니다. FactAlign은 LLM이 사실성을 유지하면서 더 많은 정보를 제공하도록 훈련할 수 있다는 것을 보여주었고, 사실적 F1 스코어도 개선되었습니다.



### Uncertainty Quantification with Bayesian Higher Order ReLU KANs (https://arxiv.org/abs/2410.01687)
Comments:
          13 pages, 7 Figures

- **What's New**: 본 연구에서는 Kolmogorov-Arnold Networks (KANs)에서 불확실성 정량화 (uncertainty quantification)의 첫 번째 방법을 제안합니다. 특히 Higher Order ReLUKANs를 통해 Bayesian 방법의 계산 요구사항을 개선하고자 하였습니다. 제안하는 방법은 일반적인 성질을 가지고 있으며, epistemic과 aleatoric 불확실성에 접근할 수 있습니다.

- **Technical Details**: 제안된 방법은 여러 가지 기준 함수 (basis functions)에 일반화될 수 있으며, 샘플러 (closure tests)를 통해 검증되었습니다. 이 방법은 Stochastic Partial Differential Equations (PDEs)에도 적용되며, 확률적 항의 포함으로 인해 도입된 함수 의존성(functinal dependencies)을 올바르게 식별할 수 있습니다.

- **Performance Highlights**: 간단한 1차원 함수와 Stochastic PDEs에 대한 응용을 통해 본 연구의 방법이 더 나은 계산 효율성과 표현력을 가지게 되는 것을 입증하였습니다. 코드 또한 GitHub에서 확인할 수 있습니다.



### Positional Attention: Out-of-Distribution Generalization and Expressivity for Neural Algorithmic Reasoning (https://arxiv.org/abs/2410.01686)
Comments:
          37 pages, 22 figures

- **What's New**: 최근 신경망(neural network)이 알고리즘 알고리즘 작업을 해결하는 능력에 대한 관심이 증가하고 있습니다. 본 논문에서는 훈련 배포와 다른 값 범위(value range)에서 테스트 배포를 다루며, positional attention을 사용하여 OOD(Out-Of-Distribution) 성능을 향상시키는 방안을 제안합니다.

- **Technical Details**: 본 논문은 Transformer 모델에 positional attention을 도입하여 고정된 위치 인코딩을 통해 주목(attention) 가중치를 결정하는 방법을 설명합니다. 이 접근 방식은 실험적 OOD 성능을 개선하면서도 모델의 표현력(expressivity)을 유지합니다. 논문에서는 PCOC 모델을 통해 기존의 MPC(MapReduce) 모델을 시뮬레이션 할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 연구의 실험 결과, positional attention을 사용한 Transformer의 성능이 기존의 end-to-end 훈련 방식보다 OOD 성능이 현저히 향상되었음을 입증하였습니다. PCOC 모델을 통해 다양한 병렬 알고리즘을 효과적으로 수행할 수 있는 가능성을 제시합니다.



### PHI-S: Distribution Balancing for Label-Free Multi-Teacher Distillation (https://arxiv.org/abs/2410.01680)
- **What's New**: 이번 연구는 라벨 없이 여러 heterogeneous visual foundation model을 융합하는 agglomerative model의 발전에 중점을 두고 있으며, 특히 teacher 모델의 activation 통계와 손실 함수가 student 모델의 품질에 미치는 영향을 다룹니다.

- **Technical Details**: 온도 단순화 실험에서, 우리는 다양한 teacher 모델의 activation 분포와 그 분포의 분산을 분석합니다. 이를 통해 Hadamard 행렬을 사용하여 isotropic 표준화를 수행하고, 이를 'PHI Standardization' (PHI-S)라고 이름 붙였으며, 이 방법이 최상의 student 모델을 생성함을 보여주었습니다.

- **Performance Highlights**: PHI Standardization 방법을 통해 생성된 student 모델은 평가 기준에서 가장 우수한 성능을 보였으며, 여러 가지 합성곱 손실 함수들을 비교 분석하여 그 결과를 제공합니다.



### Trying to be human: Linguistic traces of stochastic empathy in language models (https://arxiv.org/abs/2410.01675)
Comments:
          preprint

- **What's New**: 이번 연구는 AI가 생성한 콘텐츠와 인간이 작성한 콘텐츠를 구별하는 데 있어 감정이입(empaty)과 인간처럼 보이려는 인센티브가 어떻게 작용하는지를 실험을 통해 조사했습니다.

- **Technical Details**: 두 가지 실험(Study 1과 Study 2)을 통해 참가자들이 관계에 대한 조언 또는 단순한 설명을 작성하도록 했으며, LLM은 가능한 한 인간처럼 텍스트를 작성하도록 지시받았습니다. 이후 새로운 샘플의 인간들이 텍스트의 출처를 판단했습니다.

- **Performance Highlights**: 연구 결과, 감정이입이 필요한 상황에서는 인간이 더 우수한 성과를 보였으나, 인간처럼 보이려는 지시는 오히려 LLM에 효과적이어서 인간의 우세가 줄어드는 경향을 보였습니다.



### Bridging Context Gaps: Leveraging Coreference Resolution for Long Contextual Understanding (https://arxiv.org/abs/2410.01671)
Comments:
          Underreview version of LQCA, Bridge context gap for long context

- **What's New**: 본 논문에서는 Long Question Coreference Adaptation (LQCA) 방법을 소개하여 긴 컨텍스트에서 핵심 참조(coreference)를 해결함으로써 대화형 AI 모델의 성능을 향상시키는 새로운 접근 방식을 제공합니다.

- **Technical Details**: LQCA 방법은 네 가지 주요 단계로 구성되어 있습니다: (1) 서브 문서 내에서의 핵심 참조 해결, (2) 언급 간의 거리 계산, (3) 핵심 참조를 위한 대표 언급 정의, (4) 언급 교체를 통한 질문 응답.

- **Performance Highlights**: 실험 평가 결과, OpenAI-o1-mini 및 GPT-4o 모델에서 긍정적인 결과가 나타났으며, 핵심 참조 해결 기술을 활용하여 질문 응답에서 컨텍스트 간의 간극을 효과적으로 메우는 것을 확인했습니다.



### Towards a vision foundation model for comprehensive assessment of Cardiac MRI (https://arxiv.org/abs/2410.01665)
Comments:
          11 pages, 3 figures, 4 tables

- **What's New**: 이번 논문에서는 심장 자기 공명 영상(CMR) 평가를 위한 비전 기초 모델을 소개합니다. 이 모델은 3600만 개의 CMR 이미지에 대해 자기 지도 방식으로 훈련되었습니다.

- **Technical Details**: 모델은 분류(classification), 분할(segmentation), 랜드마크 위치 지정(landmark localization), 병리 탐지(pathology detection)의 9가지 임상 작업을 위해 미세 조정(finetuning)되었습니다. 각 작업에 대해 다양한 크기의 레이블 데이터셋에서 정확도와 견고성이 개선되었습니다.

- **Performance Highlights**: 대부분의 임상 작업에 대해 SoTA에 상응하는 성능을 달성했으며, 레이블 샘플이 적어도 몇 샷 학습(few-shot learning)에서 개선된 결과를 보였습니다. 제안된 방법은 자원 효율적인 통합 프레임워크를 제공하여 이미지 분석 작업에서 딥 러닝 기반 솔루션의 개발을 가속화할 수 있는 잠재력을 가지고 있습니다.



### Conformal Generative Modeling with Improved Sample Efficiency through Sequential Greedy Filtering (https://arxiv.org/abs/2410.01660)
- **What's New**: 본 연구에서는 Generative Models(생성 모델)의 안전성 문제를 해결할 수 있는 Sequential Conformal Prediction for Generative Models(SCOPE-Gen) 방법을 제안합니다. 이 방법은 rigorous(엄격한) 통계적 보장을 기반으로 하여 생성 모델의 출력에 대한 신뢰성을 높입니다.

- **Technical Details**: SCOPE-Gen은 black box generative model에서 i.i.d. 예제를 샘플링하여 초기 예제 집합을 생성한 후, greedy filters(탐욕적 필터)를 통해 이를 반복적으로 다듬어 나갑니다. 최종적인 예측 집합의 admissibility(허가 가능성)는 Markov chain(마코프 체인)으로 분해되어 통제될 수 있습니다. 이를 통해 calibration(보정) 동안 admissibility 평가의 횟수를 크게 줄일 수 있습니다.

- **Performance Highlights**: 자연어 생성 및 분자 그래프 확장(task) 실험을 통해 SCOPE-Gen은 기존 방법들에 비해 훨씬 적은 수의 admissibility 평가로 안전성 비판적 응용 분야에 적합하다는 것을 입증했습니다.



### Efficient Long-range Language Modeling with Self-supervised Causal Retrieva (https://arxiv.org/abs/2410.01651)
Comments:
          preprint

- **What's New**: 본 논문은 Grouped Cross-Attention (GCA)이라는 새로운 모듈을 제안하여 조합된 사전 훈련을 통해 RLM과 인과 언어 모델을 개선합니다. 이는 긴 컨텍스트 모델링을 돕는 혁신적인 방법입니다.

- **Technical Details**: GCA는 입력 시퀀스를 청크 단위로 나누고 현재 청크를 사용하여 이전 청크를 검색하는 방식을 갖습니다. 이를 통해 자동 회귀 손실을 최소화하는 청크 검색이 효율적인 방식으로 가능해졌습니다. 또한, Differentiable Retrieval-based Transformers (DRT)로 불리는 이 모델은 최대 64K 토큰까지 사전 훈련을 할 수 있으며, 메모리 효율성을 높이기 위해 이전 청크의 숨겨진 상태를 CPU 메모리로 오프로드합니다.

- **Performance Highlights**: DRT는 긴 범위의 언어 모델링에서 이전 모델에 비해 낮은 perplexity를 기록했으며, 훈련 시간은 단 50% 소요됩니다. 또한, 메모리 오프로드가 활성화된 상태에서, 추론 속도는 약 100배 향상되었습니다.



### shapiq: Shapley Interactions for Machine Learning (https://arxiv.org/abs/2410.01649)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 Shapley Value (SV)와 Shapley Interactions (SI)를 기반으로 한 새로운 오픈소스 Python 패키지인 shapiq를 소개합니다. 이 패키지는 최첨단 알고리즘을 통합하여 SV와 다양한 순서의 SI를 효율적으로 계산할 수 있게 합니다.

- **Technical Details**: shapiq는 머신러닝 모델의 예측에서 생성되는 기능 간의 상호 작용을 설명하고 시각화하는 도구입니다. 이 패키지는 11개의 머신러닝 적용 사례와 사전 계산된 게임, 실제 값이 포함된 벤치마킹 세트를 포함하여 다양한 분야에서의 계산 성능을 체계적으로 평가할 수 있습니다.

- **Performance Highlights**: shapiq는 vision transformers, language models, XGBoost, LightGBM (TreeSHAP-IQ 포함) 모델의 예측에서 기능 상호 작용을 설명하고 시각화할 수 있는 기능을 제공합니다. 이로 인해 머신러닝 내에서 SV와 SI의 적용을 넓히고 미래의 연구를 촉진할 수 있습니다.



### Stable Offline Value Function Learning with Bisimulation-based Representations (https://arxiv.org/abs/2410.01643)
Comments:
          Under review

- **What's New**: 이 논문에서는 오프라인 강화 학습에서 가치를 안정적으로 학습하기 위한 새로운 방법론인 KROPE(offline Policy Evaluation을 위한 Kernel Representations)를 제안합니다. KROPE 알고리즘은 상태-행동 쌍의 표현 방식을 개선하여 학습 안정성을 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: KROPE는 상태-행동 쌍의 유사성을 즉각적인 보상(reward) 및 목표 정책(target policy)에 따라 도달할 다음 상태-행동 쌍의 유사성에 기반하여 정의하는 커널(kernel)을 사용합니다. 이를 통해 유사한 상태-행동 쌍이 비슷한 표현을 갖도록 하는 것을 목표로 하며, 경량화된 least-squares policy evaluation (LSPE) 알고리즘의 안정성을 증명합니다.

- **Performance Highlights**: KROPE는 기준(baseline) 알고리즘들에 비해 낮은 가치 오류(value error)를 달성하며, 오프라인 가치 함수 학습의 안정성과 정확성을 높이는 데 기여합니다. 또한 이 논문은 KROPE가 기존의 비유사성(non-bisimulation) 기반 알고리즘에 비해 더 높은 안정성을 제공한 것을 실험적으로 검증합니다.



### Moral Alignment for LLM Agents (https://arxiv.org/abs/2410.01639)
- **What's New**: 이 연구는 인간의 도덕적 가치를 명시적으로 인코딩한 보상 함수를 설계하여 LLM 기반 에이전트를 조정하는 새로운 접근 방식을 제시합니다. 이를 통해 학습 에이전트가 더 나은 도덕적 결정을 내릴 수 있도록 합니다.

- **Technical Details**: 본 연구는 강화 학습(Reinforcement Learning)에서의 보상으로 도덕적 보상(intrinsic rewards)을 사용하는 방식을 소개합니다. 이 방법은 반복 죄수의 딜레마(Iterated Prisoner’s Dilemma) 환경에서 에이전트의 행동과 결과를 정량적으로 평가하며 도덕적 전략을 학습하고 자기 중심적인 전략을 '학습 해제'(unlearn)할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 도덕적 보상으로 세부 조정된 LLM 에이전트가 도덕적으로 일치하는 전략을 성공적으로 학습하고, 특정 도덕적 전략이 다른 매트릭스 게임 환경에서도 일반화됨을 보여주었습니다. 또한, 이 접근 방식은 현재의 조정 기술보다 투명하고 비용 효율적인 대안을 제시합니다.



### Data Extrapolation for Text-to-image Generation on Small Datasets (https://arxiv.org/abs/2410.01638)
- **What's New**: 이번 논문에서는 텍스트-이미지 생성 분야에서 새로운 데이터 증강 방법으로 선형 외삽(linear extrapolation)을 제안합니다. 기존의 데이터 증강 방법은 크롭(cropping), 플리핑(flipping) 등의 단순한 기법에 의존해 새로운 정보를 도입하지 못했으나, 본 연구는 검색 엔진을 통해 외부에서 이미지를 수집하여 텍스트 특징에만 선형 외삽을 적용해 데이터 양을 대폭 증가시킵니다.

- **Technical Details**: 이 방법의 핵심은 두 가지 아웃라이어 탐지기(outlier detectors)를 사용하여 검색된 이미지의 신뢰성을 보장하는 것입니다. K-means 알고리즘을 사용해 노이즈가 많은 웹 이미지를 군집화하고, CLIP 인코더를 통해 데이터셋 이미지와의 거리를 측정하여 아웃라이어를 제거합니다. 또한, NULL-condition guidance를 통해 텍스트-이미지 생성의 점수 추정(score estimation)을 정제하며, 복잡한 텍스트 정보를 처리하기 위해 재귀적 아핀 변환(recurrent affine transformation)을 사용합니다.

- **Performance Highlights**: 본 모델은 CUB, Oxford, COCO 데이터셋에서 각각 FID(Frechet Inception Distance) 점수 7.91, 9.52, 5.00을 달성했습니다. 이 결과는 원본 데이터셋보다 수십 배 많은 훈련 샘플을 기반으로 하여 텍스트-이미지 성능에서显著한 개선을 나타냅니다.



### Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis (https://arxiv.org/abs/2410.01635)
- **What's New**: 최근 그래프 프롬프트가 유망한 연구 방향으로 떠오르며, 기존의 그래프 모델 재훈련 없이 원래의 그래프에 추가적인 토큰이나 서브그래프를 학습할 수 있는 가능성을 보여주고 있습니다. 이 논문은 데이터 연산 관점에서 그래프 프롬프트를 철저히 분석하는 이론적 틀을 제공합니다.

- **Technical Details**: 이 논문에서는 그래프 프롬프트가 그래프 변환 연산자를 근사할 수 있는 능력을 보장하는 정리(thm)를 제시합니다. 또한, 단일 그래프의 데이터 연산 오류에 대한 상한을 도출하고, 여러 그래프의 배치에도 이 논의를 확장하여 실용적인 시나리오에서 그래프 프롬프트의 확장성과 일반화를 이해하는 데 중요한 분석을 수행합니다. 이 연구는 선형 그래프 모델(예: GCN)에서 비선형 모델(예: GAT)로의 이론적 발견을 확장합니다.

- **Performance Highlights**: 광범위한 실험이 이론적 결과를 뒷받침하며, 그래프 프롬프트의 효과와 설계를 위한 가이드라인을 제공합니다. 이를 통해 연구자와 실무자들이 다양한 응용 프로그램에서 그래프 프롬프트를 더 자신 있게 활용할 수 있도록 합니다.



### Entropy-Based Uncertainty Modeling for Trajectory Prediction in Autonomous Driving (https://arxiv.org/abs/2410.01628)
Comments:
          10 pages, 5 figures, submitted to International Conference on Learning Representations (2025)

- **What's New**: 자율주행 분야에서 경로 예측의 불확실성(modeling uncertainty) 모델링에 대한 새로운 접근법을 제안합니다. 이 방법은 경로 예측의 불확실성을 정량화하고 분해(decompose)하는 데 중점을 두며, 모델링 선택이 불확실성의 교정(calibration)과 모델 강건성에 미치는 영향을 분석합니다.

- **Technical Details**: 정보 이론(Information theory)에 기반한 접근법을 채택하여 경로 예측 모델의 불확실성을 두 가지 구성 요소로 나누어 측정합니다: aleatoric uncertainty(데이터 내 고유 변동성)와 epistemic uncertainty(정보 부족에서 오는 불확실성). Monte-Carlo 샘플링(MC sampling)을 통해 이 두 가지를 근사하며 불확실성을 정량화합니다.

- **Performance Highlights**: nuScenes 데이터셋을 사용한 광범위한 실험을 통해 다양한 모델 아키텍처와 설정이 불확실성 정량화 및 모델 강건성에 미치는 영향을 평가합니다. 특히, OOD(Out-of-Distribution) 시나리오에서의 강건성을 평가하며 예측 오류와 불확실성 간의 관계를 분석합니다.



### Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint? (https://arxiv.org/abs/2410.01623)
Comments:
          Code is available at: this https URL

- **What's New**: 이 논문에서는 Fira라는 새로운 메모리 효율적 훈련 프레임워크를 제안합니다. 이는 저순위(low-rank) 제약을 유지하면서도 전체 순위(full-rank) 훈련을 가능하게 하며, 저순위 훈련 방식의 장점을 활용하는 첫 번째 시도입니다.

- **Technical Details**: Fira는 두 가지 주요 구성 요소를 포함합니다: (1) norm 기반 스케일링 전략(norm-based scaling strategy), 이 전략은 저순위 최적화 기법의 스케일링 효과를 활용하여 전체 순위 훈련을 촉진합니다. (2) norm-growth 리미터(norm-growth limiter)로서, 이는 기울기(norm)의 급격한 증가를 제한하여 손실 스파이크(loss spikes)를 방지합니다. 이를 통해 Fira는 저순위의 제약 조건을 유지하면서도 더 나은 성능을 제공합니다.

- **Performance Highlights**: 종합 실험 결과, Fira는 LoRA 및 GaLore보다 뛰어난 성능을 보이며, 전체 순위 훈련과 비슷하거나 더 나은 성능을 달성했습니다. 다양한 매개변수 수(60M, 130M, 350M, 1B, 7B)를 통한 실험을 통해 Fira의 효과를 입증했습니다.



### DRUPI: Dataset Reduction Using Privileged Information (https://arxiv.org/abs/2410.01611)
- **What's New**: 본 논문에서는 기존의 데이터셋 축소(Dataset Reduction, DR) 기법을 발전시켜, 줄인 데이터셋과 함께 특권 정보(Privileged Information)를 합성하는 DRUPI( 데이터셋 축소를 위한 특권 정보 활용) 기법을 제안합니다. 이 방법은 모델 학습을 개선하기 위해 추가 학습 대상을 도입합니다.

- **Technical Details**: DRUPI는 기존의 데이터-레이블 쌍 외에 특징 레이블(feature labels) 또는 주의 레이블(attention labels)과 같은 특권 정보를 추가로 합성하여 보조적인 감독(supervision)을 제공합니다. 효과적인 특징 레이블은 지나치게 차별적이지도, 지나치게 다양하지도 않아야 하며, 적절한 수준에서 균형을 이뤄야 합니다.

- **Performance Highlights**: ImageNet, CIFAR-10/100 및 Tiny ImageNet에서의 광범위한 실험 결과, DRUPI는 기존의 데이터셋 축소 방법과 매끄럽게 통합되며, 성능 향상을 가져옵니다. 예를 들어, CIFAR10에서 Herding 방법에 DRUPI를 적용하면 성능이 24.3% 향상되며, K-center 방법에서는 최대 23.4%의 개선 효과를 보여줍니다.



### Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging (https://arxiv.org/abs/2410.01610)
Comments:
          work in progress

- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 모델을 효과적으로 튜닝할 수 있는 데이터 효율적인 접근 방식인 Upcycling Instruction Tuning (UpIT)을 제안합니다. 기존 방법들이 대규모 후속 훈련에 의존하는 반면, UpIT는 중간 체크포인트에서 전문화된 전문가를 구축하는 과정을 활용합니다.

- **Technical Details**: UpIT는 네 가지 주요 단계로 구성됩니다: (1) 전문가 준비 - 미리 저장된 체크포인트들을 이용하여 전문화된 전문가를 위한 기본 지원을 준비합니다. (2) 전문가 확장 - 유전자 알고리즘을 통하여 새로운 전문화를 가진 전문가들을 확장합니다. (3) 라우터 초기화 - 각 전문가 전용의 데이터를 할당하고 이진 분류 손실을 사용하여 라우팅 벡터를 최적화합니다. (4) 모델 업사이클링 - 여러 밀집 모델의 파라미터를 MoE 모델로 병합합니다.

- **Performance Highlights**: 다양한 데이터 규모와 업사이클링 설정에서의 실험 결과, UpIT는 기존 방법들보다 일관되게 우수한 성능을 나타냈습니다. 특히 작은 훈련 데이터 환경에서도 전통적인 밀집 튜닝 방법보다 효과적으로 결과를 개선하였으며, 전문가의 다양성을 유지하여 최종 MoE 모델의 성능을 더욱 향상시켰습니다.



### Automated Red Teaming with GOAT: the Generative Offensive Agent Tester (https://arxiv.org/abs/2410.01606)
- **What's New**: 본 논문에서는 Generative Offensive Agent Tester (GOAT)라는 새로운 자동화된 레드 팀 시스템을 소개합니다. GOAT는 다중 대화(turn) 환경에서 사용자들이 AI 모델과 상호작용하는 방식을 모방하여 다양한 공격 기법을 활용해 대화형 모델의 취약성을 식별합니다.

- **Technical Details**: GOAT는 총 7개의 레드 팀 공격 기법을 구현하며, 일반 목적의 모델에 대한 프롬프트를 통해 사용 가능한 방법, 현재 대상 모델의 응답, 다음 단계 등을 고려하며 추론을 유도합니다. 이러한 방식은 자동화된 레드 팀 방법이 실질적인 해법을 제시할 수 있도록 돕습니다.

- **Performance Highlights**: GOAT는 JailbreakBench 데이터 세트에서 Llama 3.1에 대해 97%, GPT-4에 대해 88%의 ASR@10을 기록하여 기존의 다중 대화 쿼리 방법보다 높은 성공률을 보여줍니다. 또한, 평균 5개의 대화 흐름 내에서 공격을 성사시킵니다.



### Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation (https://arxiv.org/abs/2410.01598)
Comments:
          9 pages, 7 figures,The 1st Workshop on Risks, Opportunities, and Evaluation of Generative Models in Recommender Systems (ROEGEN@RecSys 2024), October 2024, Bari, Italy

- **What's New**: 본 논문에서는 새로운 Elaborative Subtopic Query Reformulation (EQR) 방법을 소개합니다. 이는 사용자의 다양한 의도를 파악하여 여행 추천 시스템에서 효과적으로 응답을 생성하는 것을 목표로 합니다. 특히, EQR은 폭넓은 subtopic과 깊이 있는 elaboration을 동시에 제공하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: EQR은 사용자의 자연어 쿼리를 깊이 있게 이해하고 다양한 하위 주제를 생성하기 위해 대형 언어 모델(LLM)을 활용합니다. 이 방법은 기존의 query reformulation 기술의 한계를 극복하며, 보다 효과적인 sparse 및 dense retrieval을 가능하게 합니다. 논문에서는 TravelDest라는 새로운 benchmark 데이터셋도 소개하여, 50개의 넓고 간접적인 NL 쿼리와 관련된 774개의 목적지 도시를 포함합니다.

- **Performance Highlights**: TravelDest 데이터셋에 대한 실험 결과 EQR은 기존의 최첨단 QR 방법들보다 recall과 precision 면에서 유의미한 향상을 보였으며, 특히 넓고 간접적인 NL 쿼리에 대한 대응 능력을 크게 개선했습니다.



### KnobGen: Controlling the Sophistication of Artwork in Sketch-Based Diffusion Models (https://arxiv.org/abs/2410.01595)
- **What's New**: 최근 발생한 전이 모델(Diffusion Models)의 발전은 텍스트 기반 이미지 생성(T2I)에 큰 개선을 가져왔지만, 미세한 정밀함과 높은 수준의 제어 간의 균형을 맞추는 데 어려움을 겪고 있습니다. 본 연구에서는 KnobGen이라는 새로운 이중 경로 프레임워크를 제안하여 스케치 기반 이미지 생성을 민주화하고, 사용자 스킬 수준에 따라 이미지 생성을 적응시킵니다.

- **Technical Details**: KnobGen은 Coarse-Grained Controller(CGC)와 Fine-Grained Controller(FGC) 두 가지 모듈을 사용하여 고급 의미와 세부 정제를 처리합니다. 사용자는 'knob inference' 메커니즘을 통해 두 모듈의 비율을 조정할 수 있어 초보자 스케치와 숙련된 아티스트의 스케치를 모두 유연하게 처리할 수 있습니다.

- **Performance Highlights**: MultiGen-20M 데이터셋과 새롭게 수집한 스케치 데이터셋에서 KnobGen의 효과iveness를 입증하였으며, 사용자의 스케치와 텍스트를 기반으로 최종 결과물의 자연스러운 외형을 유지하면서 강력한 제어를 제공합니다.



### Imaging foundation model for universal enhancement of non-ideal measurement C (https://arxiv.org/abs/2410.01591)
- **What's New**: 본 논문에서는 일반화 가능한 비이상 측정 컴퓨터 단층 촬영(NICT) 이미지를 개선하기 위한 최초의 이미징 기반 모델인 다중 스케일 통합 변압기 AMPlifier(“TAMP”)를 제안합니다. TAMP는 360만 쌍의 NICT-ICT 이미지 쌍을 기반으로 사전 훈련되었습니다.

- **Technical Details**: TAMP는 물리적 프로세스에 기반한 사전 훈련 및 적응 프로세스를 구성하며, 3.6 백만 개의 시뮬레이션된 NICT-ICT 이미지 쌍으로 훈련되었습니다. 이 모델은 다양한 비이상 NICT 상황과 신체 부위에 직접 일반화할 수 있는 능력을 가지고 있습니다. 데이터 금고 (Low-rank adaptation, LoRA) 기법을 활용하여 몇 가지 고유한 매개 변수를 조정하여 특정 시나리오에서의 성능을 향상시킵니다.

- **Performance Highlights**: TAMP는 27개의 NICT 개선 작업을 평가하였고, 추가적인 훈련 없이도 다양한 NICT 이미지를 직접적으로 향상시킬 수 있는 능력을 보여주었습니다. 또한 벤치마크 데이터세트인 SimNICT를 공개하여 연구자들이 NICT 개선에 대한 딥러닝 방법을 탐구할 수 있는 유용한 자원을 제공합니다.



### Spoken Grammar Assessment Using LLM (https://arxiv.org/abs/2410.01579)
Comments:
          5 pages, 2 figures

- **What's New**: 본 논문은 기존의 Spoken Language Assessment (SLA) 시스템을 혁신하여 구술 표현에서의 문법 평가를 통합하는 새로운 end-to-end SLA 시스템을 제안합니다. 이 시스템은 기존의 Written Language Assessment (WLA) 도구의 필요성을 없애고, 평가 항목을 다양하게 변형하여 학습자의 사전 준비를 어렵게 만듭니다.

- **Technical Details**: SLA 시스템은 두 가지 주요 구성 요소로 나뉘어 있으며, 첫 번째 부분은 대화형 언어 모델(LLM)을 활용하여 평가용 문단을 생성하고, 두 번째 부분은 후보자가 발화한 음성 오디오를 기준으로 문법 평가를 실시합니다. hybrid automatic speech recognition (ASR) 시스템과 맞춤형 언어 모델을 사용하여 문법 오류를 자동으로 평가하며, ASR의 오류에 강한 문법 점수 매기기 모듈을 포함합니다.

- **Performance Highlights**: 저자들은 제안한 시스템이 최첨단 ASR 엔진을 초월하여 구술 문법 평가에서 우수한 성능을 발휘한다고 주장합니다. 기존 WLA 시스템 없이도 모든 언어 능력의 평가를 가능하게 하여 테스트 소요 시간을 크게 단축시킬 수 있습니다.



### Computing Ex Ante Equilibrium in Heterogeneous Zero-Sum Team Games (https://arxiv.org/abs/2410.01575)
- **What's New**: 이 논문에서는 이질적인 팀 게임에서의 전형적인 협동 및 최적화 문제를 해결하기 위한 새로운 프레임워크인 Heterogeneous-PSRO (H-PSRO)를 제안합니다. H-PSRO는 팀원 간의 이질적인 정책을 시퀀셜하게 최적화하여 더 낮은 exploitability를 달성하며, 기존의 Team PSRO보다 우수한 성능을 보여줍니다.

- **Technical Details**: H-PSRO는 팀 정책 공간의 부족한 표현력을 해결하기 위해 팀원들의 이질적인 정책을 매개변수화합니다. 이 과정을 통해 팀 보상에서 증가하는 일관된 개선을 보장하며, 두 팀의 정책 공간이 선형적으로 증가하여 계산 복잡성을 줄입니다. 이러한 접근 방식은 두 팀의 조합된 정책이 서로 다른 역할을 가진 팀원들에 의해 최적화됨을 보장합니다.

- **Performance Highlights**: H-PSRO는 행렬 이질적 게임에서 수렴을 달성하며, 비이질적인 기준선보다 우수한 성능을 보여줍니다. 실험 결과 H-PSRO는 대규모 벤치마크 게임에서도 구현할 수 있으며, 이질적 팀 게임뿐만 아니라 균질한 설정에서도 우수한 성능을 발휘함을 확인하였습니다.



### OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data (https://arxiv.org/abs/2410.01560)
- **What's New**: 이 논문에서는 수학적 추론(mathematical reasoning)을 위한 고품질의 finetuning (SFT) 데이터셋을 생성하기 위한 연구를 다루고 있습니다. 특히, OpenMathInstruct-2 데이터셋을 통해 기존의 공개 수학 추론 데이터셋보다 약 8배 더 큰 규모로 1,400만 개의 질문-답변 쌍을 제공합니다.

- **Technical Details**: 연구에서 	exttt{Llama3.1} 모델을 사용하여 데이터 합성(data synthesis)에 대한 철저한 ablation 실험을 진행하였으며, 주요 발견 사항으로는 (a) 솔루션 형식(solution format)의 중요성, (b) 강력한 teacher 모델이 생성한 데이터가 약한 student 모델의 on-policy 데이터보다 우수함, (c) 저품질 솔루션에 강한 SFT의 내구성, (d) 질문 다양성(question diversity)의 중요성을 제시했습니다.

- **Performance Highlights**: OpenMathInstruct-2로 	exttt{Llama-3.1-8B-Base} 모델을 finetuning한 결과, 	exttt{Llama3.1-8B-Instruct} 모델보다 MATH에서 절대 15.9% 향상된 성능(51.9% → 67.8%)을 보였습니다. 또한, 본 연구의 모형 및 코드와 OpenMathInstruct-2 데이터셋을 상업적 허용 라이센스 하에 공개하여 오픈소스 활동을 가속화하고자 했습니다.



### Integrative Decoding: Improve Factuality via Implicit Self-consistency (https://arxiv.org/abs/2410.01556)
- **What's New**: 이 논문은 Integrative Decoding (ID)라는 새로운 디코딩 전략을 소개하여 오픈-엔디드 생성(task)에서의 자기 일관성(self-consistency) 활용 가능성을 높입니다. 즉, 자기 일관성을 이용해 대규모 언어 모델의 사실 정확도를 향상시키는 방법을 제시합니다.

- **Technical Details**: ID는 각 샘플링된 응답을 원본 프롬프트와 연결하여 새로운 입력을 구성하고, 이 입력을 동시에 처리하여 최종 토큰을 선택합니다. 이 과정에서 각 입력은 샘플링된 응답의 '대표' 역할을 하며, 언어 모델의 다양한 응답 간의 일관성을 집계합니다. 기존 방법은 엄격한 형식 제약이 있는 반면, ID는 형식에 대한 제약이 없고 추가적인 프롬프트가 필요하지 않아서 더 넓은 범위에 적용 가능합니다.

- **Performance Highlights**: ID는 TruthfulQA (+11.2%), Biographies (+15.4%), LongFact (+8.5%) 벤치마크에서 사실성을 일관되게 높이며, 샘플링된 응답 수가 증가함에 따라 성능 향상이 점진적으로 증대하여 반복 샘플링에 대한 확장 가능성을 보여줍니다.



### Edge-preserving noise for diffusion models (https://arxiv.org/abs/2410.01540)
- **What's New**: 이 논문에서는 기존의 균일한 가우시안 노이즈를 사용하는 생성적 확산 모델의 한계를 극복하기 위해, 에지를 보존하는 새로운 확산 모델을 제안합니다. 이 모델은 이미지 처리에서 오래된 비등방 확산(anisotropic diffusion) 기법에서 영감을 받아 구조적 정보를 더욱 잘 반영합니다.

- **Technical Details**: 제안된 모델은 에지 보존 노이즈(Edge-Preserving Noise) 스케줄러를 사용하여 에지를 보존하는 노이즈와 비등방 Gaussian 노이즈 간의 변화를 통해 생성 과정의 수렴 속도를 높입니다. 이를 통해 저주파 및 중주파(low-to-mid frequencies) 정보를 더 효율적으로 학습할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 무조건적인 이미지 생성에서 기존 최첨단 모델들보다 일관되게 우수한 성능을 보이며, 특히 픽셀 공간 확산(Pixel Space Diffusion)에서 FID 점수가 30% 향상되었습니다. 또한, 모양 기반 우선 규칙(shape-based prior)에 기반한 생성 작업에서도 더 뛰어난 품질과 견고성을 나타냅니다.



### Seeing Eye to AI: Human Alignment via Gaze-Based Response Rewards for Large Language Models (https://arxiv.org/abs/2410.01532)
- **What's New**: 이 연구에서는 GazeReward라는 새로운 프레임워크를 소개하여 눈 추적(eye-tracking, ET) 데이터를 보상을 생성하는 모델에 통합합니다. 이 프레임워크는 사용자의 선호도를 보다 정확하게 반영하도록 설계되었습니다.

- **Technical Details**: GazeReward는 눈의 움직임과 고정 시점을 측정하여 사용자의 인지 및 지각 과정을 이해하는 데 도움을 주며, 이를 통해 더욱 높은 정확도의 보상 모델(Reward Model, RM)을 개발합니다. 연구에서는 여러 LLM과 ET 생성 모델을 이용한 메커니즘 비교 연구를 통해 GazeReward의 효과를 입증하였습니다.

- **Performance Highlights**: GazeReward를 적용한 실험에서, 보상 모델의 정확성이 기존 방법보다 20% 이상 향상됨을 보여주었습니다. 이는 다양한 인간 선호 데이터셋에서 검증되었습니다.



### TiVaT: Joint-Axis Attention for Time Series Forecasting with Lead-Lag Dynamics (https://arxiv.org/abs/2410.01531)
Comments:
          15pages, 5 figures

- **What's New**: 이번 연구에서는 기존의 Channel-Dependent (CD) 모델의 한계를 극복하기 위해 TiVaT(Time-Variable Transformer)를 제안합니다. TiVaT는 Joint-Axis (JA) 주의 메커니즘을 통합하여 시계열과 변수 간의 복잡한 상호작용을 동시에 포착할 수 있는 새로운 아키텍처를 제공합니다.

- **Technical Details**: TiVaT는 시계열 예측에서의 변동성과 시간 종속성을 모두 캡처하도록 설계되었습니다. 주요 혁신으로는 JA 주의 메커니즘과 Distance-aware Time-Variable (DTV) 샘플링 기법이 있습니다. DTV 샘플링은 변수와 시간 단계 간의 거리 정보를 학습된 2D 맵으로 캡처하여 중요한 변동-시간 포인트를 동적으로 선택하고, 계산 비용을 줄이며 노이즈를 최소화합니다.

- **Performance Highlights**: TiVaT는 다양한 데이터셋에서 강력한 성능을 꾸준히 발휘하며, 복잡한 패턴을 포착하는 데 뛰어난 능력을 보입니다. 또한, 기존의 최첨단 모델을 초월하거나 경쟁력을 유지하는 성과를 거두어, 복잡한 의존성을 처리하는 새로운 기준으로 자리매김하고 있습니다.



### InstaTrans: An Instruction-Aware Translation Framework for Non-English Instruction Datasets (https://arxiv.org/abs/2410.01512)
- **What's New**: 최신 연구에서는 기존의 고품질 영어 지침 데이터셋을 비영어 데이터셋으로 번역하는 방법을 제안합니다. 이를 통해 번역의 완전성과 지침 인식을 보장하는 새로운 프레임워크 InstaTrans를 도입하였습니다.

- **Technical Details**: InstaTrans (INSTruction-Aware TRANSlation) 프레임워크는 고품질 영어 지침 데이터셋을 다른 언어로 번역하기 위해 특별히 설계되었습니다. 이 방법은 GPT-4를 활용하여 소규모 샘플을 번역한 후, 이를 기반으로 LLM을 미세 조정하여 번역 품질을 향상시키고 있습니다.

- **Performance Highlights**: 실험 결과, InstaTrans를 통해 번역된 데이터셋은 지침의 완전성과 인식 정보를 높이며, 이를 통해 LLM의 성능이 목표 언어에서 효과적으로 개선됨을 보여주었습니다. 특히, 다양한 언어에서 LLM의 접근성을 높이는 잠재력이 강조됩니다.



### LEGO: Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion (https://arxiv.org/abs/2410.01506)
Comments:
          Research paper

- **What's New**: 이 논문에서는 컴퓨터 비전 작업에서 서로 다른 표현, 도메인, 모드(모달리티)의 특징(feature)을 효과적으로 결합하는 새로운 접근법을 제안합니다. 저자들은 고차원 특징 공간에서 저차원 해석 가능한 그래프 공간으로 전환하여 다양한 수준의 특징 관계를 인코딩하는 유사성 그래프(similarity graphs)를 구성하였습니다.

- **Technical Details**: 제안된 방법은 그래프 제곱(graph power) 확장을 활용하고, 이러한 그래프 파워를 결합하기 위해 학습 가능한 그래프 융합 연산자(learnable graph fusion operator)를 도입합니다. 이 방법은 관계 중심(relationship-centric)으로 작동하며, 수학적으로 원리에 부합하며, 멀티선형 다항식(multilinear polynomials)을 통해 원소별 유사도 점수(element-wise similarity score)를 집계하는 방식으로 유사합니다.

- **Performance Highlights**: 이 논문에서 제안하는 그래프 기반 융합 방법은 비디오 이상 탐지(video anomaly detection) 작업에서 강력한 성능을 보여주며, 다중 표현(multi-representational), 다중 모달(multi-modal), 다중 도메인(multi-domain) 특징 융합 작업에서 효과적임을 입증하였습니다.



### Discrete Diffusion Schr\"odinger Bridge Matching for Graph Transformation (https://arxiv.org/abs/2410.01500)
- **What's New**: 이번 연구에서는 DDSBM(Discrete Diffusion Schrödinger Bridge Matching)이라는 새로운 프레임워크를 제안하여 고차원 이산 상태 공간에서 SB(Schrödinger Bridge) 문제를 해결합니다. 이 연구는 이산 도메인에 대한 기존의 한계를 극복하고, 그래프 변환 문제에도 적용 가능성을 보여줍니다.

- **Technical Details**: DDSBM은 연속 시간 마르코프 체인(CTMCs)을 활용하며, 이터레이티브 마르코프 피팅(IMF) 방식으로 이산 도메인에서 SB 문제를 솔빙하는 방법론입니다. 이 기반 위에 그래프 도메인으로 확장하여, 그래프 수정 거리(GED)를 비용 함수로 해석하고, 노드 및 엣지의 독립적인 수정을 통해 함수의 유용성을 강조합니다.

- **Performance Highlights**: 화학 분야의 분자 최적화 문제에 DDSBM을 적용한 실험 결과, DDSBM은 분자의 관심 속성(Property-of-Interest)을 최소한의 그래프 변환으로 효과적으로 최적화하는 데 성공하였으며, 기존의 그래프 대 그래프 변환 모델에 비해 여러 분자의 특성을 잘 유지합니다.



### DLP-LoRA: Efficient Task-Specific LoRA Fusion with a Dynamic, Lightweight Plugin for Large Language Models (https://arxiv.org/abs/2410.01497)
Comments:
          Preprint under review, 18 pages, 7 figures

- **What's New**: DLP-LoRA는 문장 수준에서 여러 LoRA를 동적으로 융합하기 위해 mini-MLP 모듈을 사용하는 Dynamic Lightweight Plugin이며, 이는 효율성을 크게 향상시킨다.

- **Technical Details**: DLP-LoRA는 약 5M의 파라미터를 가진 mini-MLP로 구성되어 있으며, top-p 샘플링 전략을 통해 다중 LoRA를 동적으로 융합한다. 이를 통해 기존의 token-level 방식보다 더 빠른 추론 성능을 제공한다.

- **Performance Highlights**: DLP-LoRA는 26개 작업에서 평균 92.34%의 정확도를 기록했고, 9개의 질문-답변(QA) 데이터셋에서는 BLEU 및 ROUGE 점수가 크게 향상되었다. 특히, MCQ와 QA 작업에서 각각 92.95%와 13.2%의 상대적인 개선을 보였다.



### One Wave to Explain Them All: A Unifying Perspective on Post-hoc Explainability (https://arxiv.org/abs/2410.01482)
Comments:
          main: 10 pages, appendix: 14 pages, 5 Tables, 25 Figures

- **What's New**: 본 논문은 Wavelet Attribution Method (WAM)을 제안하여, 기존의 기울기 기반 특성 할당 방법을 변동 영역(wavelet domain)으로 확장함으로써 이미지, 오디오, 3D 형상에 대한 통합적인 설명 프레임워크를 제공합니다.

- **Technical Details**: WAM은 입력 신호의 wavelet 분해에 따른 분류 모델 예측의 기울기를 계산하여 생성된 설명을 부드럽게 만들어, 모델의 의사 결정 과정에 대한 더 깊은 통찰을 제공합니다. 기존 방법인 SmoothGrad, Integrated Gradients와의 통합을 통해, WAM은 연속 공간에서 정의된 모든 유형의 모달리티에 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, WAM은 다양한 이미지, 오디오 및 3D 설명 가능성에서 기존의 최첨단 방법들과 비교하여 충실도(metrics) 측면에서 동등하거나 뛰어난 성능을 보임을 보여주었습니다.



### SonicSim: A customizable simulation platform for speech processing in moving sound source scenarios (https://arxiv.org/abs/2410.01481)
Comments:
          Technical report

- **What's New**: 이 논문에서는 이동하는 음원 조건하에서 음성 분리 및 향상 모델의 평가를 위한 새로운 합성 데이터 생성 도구인 SonicSim을 소개합니다.

- **Technical Details**: SonicSim은 Habitat-sim 플랫폼에 기반하여 디자인된 합성 툴킷으로, 장면 수준, 마이크로폰 수준, 소스 수준에서 다차원 조정이 가능합니다. 이 플랫폼은 90개의 실제 환경을 포함한 SonicSet 벤치마크 데이터셋을 생성하는 데 사용됩니다.

- **Performance Highlights**: SonicSet 데이터셋은 실제 환경에서 수집된 5시간 분량의 음성 데이터와 비교하여 높은 일반화 능력을 보이며, 합성 데이터가 실제 상황에 효과적으로 적용될 수 있음을 보여줍니다.



### Peeling Back the Layers: An In-Depth Evaluation of Encoder Architectures in Neural News Recommenders (https://arxiv.org/abs/2410.01470)
Comments:
          Accepted at the 12th International Workshop on News Recommendation and Analytics (INRA 2024) in conjunction with ACM RecSys 2024

- **What's New**: 이번 연구는 Neural News Recommender (NNR) 시스템에서 인코더 아키텍처를 체계적으로 분석하여 다양한 인코더 디자인의 유사성과 성과의 차이를 이해하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구에서는 (i) Central Kernel Alignment을 사용하여 학습된 뉴스 및 사용자 표현의 유사성을 평가하고, (ii) Jaccard 계수를 통해 생성된 추천 리스트의 유사성을 측정하며, (iii) 전반적인 추천 성과를 분석합니다.

- **Performance Highlights**: 연구 결과는 복잡한 인코딩 기술의 일부가 실질적으로 불합리할 수 있음을 강조하며, 더 간단하고 효율적인 아키텍처의 필요성을 제시합니다. 또한 뉴스 인코더의 의미적 풍부함과 사용자 인코더의 단순화 가능성을 강조합니다.



### TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation (https://arxiv.org/abs/2410.01469)
Comments:
          Technical report, demo page: this https URL

- **What's New**: 최근 연구에서는 모델 성능 개선에 중점을 두었으나, 저지연(low latency) 음성 처리 시스템에서는 높은 효율성이 동등하게 중요하다는 점을 강조하며 TIGER라는 경량 음성 분리 모델을 제안합니다. 이 모델은 파라미터와 계산 비용을 각각 94.3%와 95.3% 줄입니다.

- **Technical Details**: TIGER(Time-frequency Interleaved Gain Extraction and Reconstruction network)는 주파수 대역을 분할하고 멀티 스케일 선택적 주의 모듈(Multi-scale Selective Attention Module)과 전체 주파수 프레임 주의 모듈(Full-frequency-frame Attention Module)을 활용하여 맥락(feature) 정보를 추출합니다. 새로운 데이터셋 EchoSet은 현실적인 반향(reverberation)과 소음을 포함하여 복잡한 음향 환경에서의 성능 평가를 가능하게 합니다.

- **Performance Highlights**: TIGER는 EchoSet 및 실제 세계 데이터에서 TF-GridNet이라는 최첨단 모델보다 뛰어난 성능을 달성하였으며, 특히 EchoSet에서 성능이 약 5% 개선된 결과를 보여 주었습니다. 이는 TIGER가 100만 개 미만의 파라미터로 SOTA 모델과 유사한 성능을 달성한 첫 번째 음성 분리 모델이 됩니다.



### Agent-Driven Large Language Models for Mandarin Lyric Generation (https://arxiv.org/abs/2410.01450)
Comments:
          6 pages, figures, Accepted at O-COCOSDA 2024

- **What's New**: 이번 연구에서는 멜로디와 가사의 조화를 고려한 multi-agent 시스템을 개발하여 가사 생성 과제를 세분화하였습니다. 각 에이전트는 운율, 음절 수, 가사-멜로디 정렬 및 일관성을 제어합니다.

- **Technical Details**: 연구는 Mpop600 데이터셋에 기반하여, 음계 텍스트와 멜로디 간의 정렬을 학습하는 방법론을 제안합니다. 새로운 접근법으로 음파 변환(singing voice synthesis)을 사용하여 다양한 에이전트 그룹이 생성한 가사의 품질을 평가합니다.

- **Performance Highlights**: 가사-멜로디 생성의 품질 평가를 위한 청취 실험이 수행되었으며, 하모니 알리그먼트(harmony alignment)와 감각적 조화를 통해 멜로디에 적합한 가사를 생성하는 데 성공했습니다.



### Geometric Signatures of Compositionality Across a Language Model's Lifetim (https://arxiv.org/abs/2410.01444)
Comments:
          Under review as a conference paper at ICLR 2025

- **What's New**: 이 연구는 인공 언어 모델의 조합 일반화(compositional generalization) 능력과 관련하여, 표현의 의미가 어떻게 구성되어 있고, 이러한 능력의 기저에 있는 표현 메커니즘을 분석합니다. 처음으로 데이터 세트의 조합성 정도가 LM의 표현의 내재적 차원(intrinsic dimensionality)에 어떻게 반영되는지를 조사합니다.

- **Technical Details**: 연구는 조합의 두 가지 유형, 즉 형태의 조합(compositionality of form)과 의미의 조합(compositionality of meaning)을 구별하며, 비선형 및 선형 차원으로 두 가지 차원성을 측정합니다. LMs은 전달할 프레이즈의 통계적 규칙을 학습하고, 그 과정에서 조합적 복잡성(combinatorial complexity)을 인코딩합니다. 이들은 언어 처리에서 형식과 의미의 차이를 구분할 수 있게 해줍니다.

- **Performance Highlights**: 연구에서는 비선형 ID가 의미 조합성을 인코딩하고 선형 차원이 형태 조합성과 높은 상관관계를 갖는다는 것을 발견했습니다. 이는 LMs가 어떻게 언어를 처리하는 방식에서 의미와 형태가 관련이 있음을 시사합니다. 결과적으로 비선형과 선형 표현의 복잡성이 언어 모델의 훈련 과정에서 어떻게 달라지는지를 보여주었습니다.



### Fair4Free: Generating High-fidelity Fair Synthetic Samples using Data Free Distillation (https://arxiv.org/abs/2410.01423)
- **What's New**: Fair4Free는 개인적이거나 접근할 수 없는 데이터 상황에서도 공정한 합성 데이터를 생성할 수 있는 새로운 생성 모델입니다. 이 모델은 데이터 없이 지식 증류(data-free distillation)를 통해 작동합니다.

- **Technical Details**: Fair4Free는 먼저 교사 모델을 훈련시켜 공정한 표현을 만든 뒤, 이를 학습 데이터 없이 학생 모델로 지식 증류합니다. 이 과정에서 우리는 Variational Autoencoder (VAE)를 사용하여 공정한 표현을 학습하고 이를 기반으로 고충실도 합성 샘플을 재구성합니다. 이 방법은 노이즈를 입력으로 활용합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, Fair4Free의 합성 샘플은 공정성(fairness)에서 5%, 유용성(utility)에서 8%, 합성 품질(synthetic quality)에서 12% 향상된 성능을 보여주는 등 최첨단 모델보다 뛰어난 성능을 발휘합니다.



### The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs (https://arxiv.org/abs/2410.01417)
- **What's New**: 새로운 다중 모달 대형 언어 모델(MLLMs) 벤치마크 제안. 인간의 기본적인 연관 능력을 평가하기 위한 평가 기준 및 주목받지 못한 연관 임무 개발.

- **Technical Details**: 연관 작업을 정의하고 자연어 데이터셋을 활용한 annotation-free (주석 없는) 방법으로 연관 벤치마크 구축. 세 가지 수준의 연관 작업(단일 단계, 동기식, 비동기식) 설정.

- **Performance Highlights**: 현재 공개 소스 MLLMs는 연관 작업에서 인간과 비교해 일관되게 낮은 성능을 보임. 최고 성능의 닫힌 소스 모델도 인간 성능과 큰 차이 존재.



### On the Convergence of FedProx with Extrapolation and Inexact Prox (https://arxiv.org/abs/2410.01410)
Comments:
          36 pages, 6 figures

- **What's New**: 이 논문은 FedProx 알고리즘을 기반으로 한 FedExProx 방법에 대한 새로운 분석을 제공하며, 클라이언트가 정확한 proximal operator를 계산하지 않는 비현실적인 상황에서도 해당 알고리즘의 수렴력을 보장하고 있습니다.

- **Technical Details**: 연구에서는 클라이언트 각자가 수행하는 local optimizer가 요구하는 local iteration complexity를 식별하고, inexactness(비정확성)를 biased compression(편향 압축)과 연결하여 분석의 정교성을 더했습니다. 일반 수렴 결과를 통해 inexactness가 해결책의 근처로 수렴하게 하는 방법을 제시했습니다.

- **Performance Highlights**: FedExProx는 L-strongly convex(강한 볼록성) 설정에서도 동작하며, 기존에 제시된 선형 수렴 속도 \(O(L\gamma(1+\gamma L_{max})/\mu)\)를 달성할 수 있음을 입증하였고, 이후 분석을 통해 새로운 방법의 수렴 속도를 개선하였습니다.



### Can We Delegate Learning to Automation?: A Comparative Study of LLM Chatbots, Search Engines, and Books (https://arxiv.org/abs/2410.01396)
Comments:
          21 pages, 14 figures

- **What's New**: 이 연구는 LLM(대규모 언어 모델) 기반 채팅봇의 교육적 효과와 학습 결과에 대한 교육자들의 우려를 심층적으로 조사했습니다. 또한, LLM이 제공하는 자동화된 학습 도구와 기존의 학습 도구(교과서 및 웹자료)의 비교를 통해 학습 결과에 미치는 영향을 분석했습니다.

- **Technical Details**: 이 연구는 92명의 대학생을 대상으로 한 혼합 방법 연구(mixed-methods study)로, 세 가지 학습 도구(책, 웹, ChatGPT)와 그 자동화 수준을 비교했습니다. 교육자들의 우려는 (1) 신뢰성 부족, (2) 체계적 조직이 부족, (3) 인지적 참여가 약하다는 세 가지 주요 원인으로 정리되었습니다.

- **Performance Highlights**: LLM 기반 채팅봇은 개념에 대한 포괄적인 이해를 지원했지만 장기 기억 유지는 책보다 덜 효과적이었습니다. 학업 성과가 높은 학습자들은 검색 활동보다 콘텐츠에 더 깊이 참여하며 더 나은 학습 성과를 보였습니다. 결과적으로 학습 도구보다 학생의 개인적 능력이 수동적 학습에 더 많은 영향을 미쳤습니다.



### FLAME: Adaptive and Reactive Concept Drift Mitigation for Federated Learning Deployments (https://arxiv.org/abs/2410.01386)
Comments:
          Accepted for Publication at EMERGE Workshop - EWSN 2024

- **What's New**: 이번 논문은 개념 변화(concept drift)를 탐지하고 완화할 수 있는 새로운 솔루션인 Federated Learning with Adaptive Monitoring and Elimination (FLAME)을 제시합니다. FLAME은 Federated Learning (FL) 환경 내에서 IoT 기기의 동적인 변화에 대응하는 데 중점을 둡니다.

- **Technical Details**: FLAME은 FL 아키텍처를 활용하여 데이터의 동적 변화가 모델의 성능에 미치는 영향을 최소화합니다. 이 시스템은 클라우드(cloud), 엣지(edge), 마이크로컨트롤러(microcontroller)로 구성된 3-tier 아키텍처에서 작동하며, 데이터의 보안과 개인 정보 보호를 보장합니다.

- **Performance Highlights**: FLAME은 기존의 경량화된 완화 방법보다 우수한 성능을 보여주며, 대규모 IoT 배치에서 자원 활용도를 줄이고 높은 F1 점수를 유지합니다. 이는 실제 애플리케이션에 유망한 접근 방식을 제공합니다.



### Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition (https://arxiv.org/abs/2410.01380)
- **What's New**: 이 연구는 모델의 파라메트릭 지식 통합 경향이 프리트레이닝(pretraining) 동안 어떻게 변화하는지와 이러한 행동이 전반적인 성능에 어떻게 영향을 미치는지를 조사합니다. 고유한 개념인 knowledge entropy를 도입하여 저자가 언급하는 모델의 지식 획득 및 망각에 대한 행동 변화를 정의합니다.

- **Technical Details**: knowledge entropy는 모델이 다양한 기억 소스를 통합하는 방식을 정량화하는 데 사용됩니다. 높은 knowledge entropy는 모델이 넓은 범위의 기억 소스를 활용함을 나타내고, 낮은 knowledge entropy는 특정 소스에 대한 의존을 시사합니다. 이 연구는 프리트레이닝이 진행됨에 따라 knowledge entropy가 일관되게 감소한다는 것을 발견했습니다.

- **Performance Highlights**: 모델의 후반 프리트레이닝 단계에서는 낮은 knowledge entropy가 지식 획득과 유지 능력의 감소와 관련이 있음을 발견했습니다. 특히, 비활성 기억 벡터의 활동을 증가시킴으로써 지식 획득과 망각을 향상시킬 수 있다는 점에서 이 연구 결과는 중요합니다.



### PCQPR: Proactive Conversational Question Planning with Reflection (https://arxiv.org/abs/2410.01363)
Comments:
          Accepted by EMNLP 2024 Main

- **What's New**: 이번 연구는 전통적인 대화형 질문 생성 방식에서 벗어나, 특정 결론에 도달하는 방향으로 대화를 주도하는 'Conclusion-driven Conversational Question Generation (CCQG)'이라는 새로운 작업 방식을 제안합니다.

- **Technical Details**: PCQPR(Proactive Conversational Question Planning with self-Refining) 프레임워크는 Monte Carlo Tree Search (MCTS)에서 영감을 받은 계획 알고리즘과 대규모 언어 모델(LLM)의 분석 능력을 결합하여, 미래 대화 전개를 예측하고 질문 전략을 세밀하게 조정합니다. 이 과정에서 LLM은 시뮬레이션된 대화 경로를 분석하여 피드백을 제공하며, 각 단계에서의 성공과 실패를 식별합니다.

- **Performance Highlights**: PCQPR은 기존 CQG 방법론에 비해 현저하게 높은 성능을 보여주며, 결론 중심의 대화형 질문-응답 시스템을 향한 패러다임 변화의 이정표가 됩니다.



### Codev-Bench: How Do LLMs Understand Developer-Centric Code Completion? (https://arxiv.org/abs/2410.01353)
- **What's New**: 본 논문에서는 개발자 생산성을 높이는 핵심 작업인 코드 완성을 위한 새로운 평가 기준을 제안합니다. 이를 위해 industrial code completion 툴에서의 비즈니스 데이터를 분석하고, 기존 벤치마크의 한계를 극복하기 위한 새로운 방법론을 소개합니다.

- **Technical Details**: Codev-Agent라는 에이전트 기반 시스템을 도입하여 자동으로 repository를 탐색하고, 실행 환경을 구축하며, 기존 유닛 테스트에서 동적 호출 체인을 추출하고 새로운 테스트 샘플을 생성합니다. 이 시스템은 개발자의 의도와 코드 완성 동작을 최대한 반영하여 코드 완성 도구를 평가하는 데 필요한 설정과 최적화를 지원합니다.

- **Performance Highlights**: Codev-Bench를 사용하여 여러 최선진 LLM에서의 평가 결과를 제시하며, 코드 완성에서의 실용적인 문제점들(잘못된 들여쓰기, 여러 코드 블록에 걸친 예측 오류 등)을 강조합니다. 이러한 문제들은 코드 완성 도구 사용 시 사용자 경험에 심각한 영향을 미치므로, 기존의 벤치마크에서는 포착되지 못했습니다.



### Takin-VC: Zero-shot Voice Conversion via Jointly Hybrid Content and Memory-Augmented Context-Aware Timbre Modeling (https://arxiv.org/abs/2410.01350)
Comments:
          Work in Progress; Under Review

- **What's New**: 이 논문에서는 Takin-VC라는 새로운 제로샷(Zero-shot) 음성 변환(Voice Conversion, VC) 프레임워크를 제안합니다. 이 프레임워크는 하이브리드 콘텐츠와 메모리 보강 기반의 맥락 인식 음질(timbre) 모델링을 결합하여 스피커 유사성과 음질을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: Takin-VC는 정교한 하이브리드 콘텐츠 인코더를 사용하여 전이 학습된 WavLM과 HybridFormer에서 양자화된 특성을 활용하여 소스 음성의 언어적 콘텐츠를 추출합니다. 이후, 교차 주의(cross-attention) 기반의 맥락 인식 음질 모델링 접근법을 통해 목표 음질의 세밀한 특성을 학습합니다. 또한, 조건부 흐름 매칭 모델(conditional flow matching model)을 사용하여 소스 음성의 Mel 스펙트로그램을 재구성하며, 메모리 증강 모듈을 통해 고품질의 조건부 목표 입력을 생성합니다.

- **Performance Highlights**: Takin-VC는 500k 시간의 다국어(Mandarin과 English) 및 LibriTTS 데이터셋에서 수행된 실험에서 기존의 SOTA 제로샷 VC 시스템을 초월하여, 음성 자연스러움과 스피커 유사성 모두에서 뛰어난 성능을 나타냈습니다.



### PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems (https://arxiv.org/abs/2410.01337)
- **What's New**: 본 논문에서는 Physics-encoded Message Passing Graph Network (PhyMPGN)라는 새로운 그래프 학습 접근 방식을 제안하여, 불규칙한 메시 간격을 갖는 데이터에 기반하여 시공간 PDE 시스템을 모델링합니다. 작은 훈련 데이터셋으로도 작동할 수 있는 모델입니다.

- **Technical Details**: PhyMPGN은 메시지 전파 메커니즘을 활용하여 불규칙한 낮은 해상도 메시에서 시공간 동역학을 모델링합니다. 이 모델은 PDE 시스템의 시간적 진행을 근사하기 위해 2차 수치 적분기를 적용하며, 대칭적인 물리학적 현상을 고려하여 학습 가능한 Laplace block을 설계했습니다. 또한 경계 조건 패딩 전략을 통해 모델 정확성과 수렴성을 개선합니다.

- **Performance Highlights**: PhyMPGN은 불규칙한 저해상도 메시에서 다양한 유형의 시공간 동역학을 정확히 예측할 수 있으며, 다른 기준 모델에 비해 50% 이상의 성능 향상을 달성하여 최신 성과를 지속적으로 초과합니다.



### Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models (https://arxiv.org/abs/2410.01335)
Comments:
          11 main pages, 23 pages total, 9 figures, 5 tables

- **What's New**: 본 논문에서는 언어 특정 데이터가 부족한 비영어 사용자를 위한 대형 언어 모델(LLMs)의 수학적 추론 능력을 이전하는 새로운 모델 병합 방법론을 제안합니다. 이 방법론은 모델 수프의 원칙에 따라 두 개의 '전문가(experts)' 모델을 병합함으로써, 언어와 수학의 능력을 동시에 활용할 수 있게 합니다.

- **Technical Details**: 우리는 동일한 사전 학습된 모델에서 시작하여 각각 영어 수학 데이터와 목표 언어의 일반Instruction 데이터로 '전문가' 모델을 파인 튜닝합니다. 이후 수학 전문가의 최상위 및 최하위 transformer 레이어를 언어 전문가의 레이어로 직접 교체하여 목표 언어에서의 수학 성능을 향상시킵니다. 이 방법은 각 전문가를 파인 튜닝할 때 가장 중요한 파라미터 변화에 대한 해석적 분석에 기반하여 간단하고 비용이 적게 들며 직관적입니다.

- **Performance Highlights**: 병합된 모델은 수학 벤치마크인 MGSM에서 평균적으로 10% 더 나은 성능을 보이며, 스와힐리어, 텔루구어, 벵골어, 일본어 등 4개의 주요 언어에서 성능을 향상시킵니다. 특히, 스와힐리어에서는 혼합된 스와힐리 및 수학 데이터셋을 파인 튜닝한 모델의 성능을 초과하여 뛰어난 결과를 냅니다.



### Unveiling Language Skills under Circuits (https://arxiv.org/abs/2410.01334)
- **What's New**: 이 논문은 언어 모델의 메모리 읽기 기능을 독립적으로 조작하는 최소 단위인 Memory Circuit의 개념을 도입합니다. 이를 통해 Transformer 모델 내의 다양한 언어 기술을 더 명확하게 분리하여 분석할 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 논문에서는 Transformer 모델을 완전한 회로 그래프로 나누고, 언어 기술에 기여하는 주요 회로 경로를 추출하는 3단계 프레임워크를 개발합니다. 1단계에서는 Memory Circuits를 통해 각 레이어에서 메모리 기능을 독립적으로 나타내는 회로 그래프를 구성합니다. 2단계에서는 Greedy Search를 사용하여 불필요한 경로를 제거하고 3단계에서는 각 경로의 인과 효과를 추정하여 기술 경로를 선택합니다.

- **Performance Highlights**: 실험 결과, 이전 토큰 기술(Previous Token Skill), 유도 기술(Induction Skill), 그리고 상황 학습 기술(ICL Skill) 간의 관계를 입증하였고, 간단한 언어 기술은 얕은 레이어에, 복잡한 언어 기술은 깊은 레이어에 존재한다는 기존 가설을 검증하였습니다. 이를 통해 Chain-of-Thought(코드)의 재활용 가능성에 대한 새로운 증거를 제시합니다.



### Fair Class-Incremental Learning using Sample Weighting (https://arxiv.org/abs/2410.01324)
- **What's New**: 이 논문은 Trustworthy AI를 위한 공정한 클래스 증분 학습을 다루고 있으며, 기존의 정확도 중심 접근 방식에서 벗어나 민감한 그룹을 포함한 모델 공정성을 연구합니다.

- **Technical Details**: 학습 샘플의 가중치를 조정하여 현재 작업 데이터의 평균 기울기 벡터 방향을 변경함으로써 공정성을 달성합니다. 최적화 문제를 설정하고, Linear Programming으로 문제를 해결하며, Fairness-aware Sample Weighting (FSW) 알고리즘을 제안합니다.

- **Performance Highlights**: FSW는 실제 데이터셋에서 기존의 선진 방법들보다 더 나은 정확도와 공정성 균형을 달성하는 결과를 보여주었습니다.



### Forte : Finding Outliers with Representation Typicality Estimation (https://arxiv.org/abs/2410.01322)
- **What's New**: 이 논문은 기존의 Generative 모델들이 생성한 데이터의 OOD(Out-Of-Distribution) 탐지 문제를 다루고 있으며, 새로운 접근 방식인 Forte를 소개합니다. Forte는 self-supervised learning 기법을 활용하여 OOD 탐지의 정확성을 높이며, class labels가 필요하지 않고, OOD 데이터와의 노출 없이도 작동합니다.

- **Technical Details**: Forte는 CLIP, ViT-MSN 및 DINOv2와 같은 다양한 표현 학습 기법을 비모수 밀도 추정 모델(OCSVM, KDE, GMM)과 결합하여 atypical samples를 탐지합니다. 이 방법은 전체 데이터의 정보적 summary statistics를 효과적으로 반영하는 데이터 포인트 단위의 통계들을 제공합니다.

- **Performance Highlights**: Forte는 다양한 OOD 탐지 작업과 합성 이미지 탐지에서 기존의 최첨단 방법들보다 우수한 성능을 보여주며, 포토리얼리스틱 이미지 생성을 포함한 여러 벤치마크에서 최상의 성능을 달성했습니다.



### Finetuning Pre-trained Model with Limited Data for LiDAR-based 3D Object Detection by Bridging Domain Gaps (https://arxiv.org/abs/2410.01319)
Comments:
          Accepted in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024

- **What's New**: 이 논문에서는 Domain Adaptive Distill-Tuning (DADT)이라는 새로운 방법을 제안하여, 사전 훈련된 모델을 적은 양의 타겟 데이터(약 100개의 LiDAR 프레임)를 사용하여 적응시키는 방법을 다루고 있습니다. DADT는 객체 수준(object-level)과 контекст 수준(context-level) 표현을 정렬하는 정규화를 사용하여, 모델의 표현력(representation power)을 유지하면서 과적합(overfitting)을 방지합니다.

- **Technical Details**: DADT는 교사-학생 아키텍처(teacher-student architecture)를 기반으로 하며, 포기된 사전 훈련된 백본(backbone)을 사용하여 밀도 정렬(LiDAR 입력의 밀도를 맞춰주는)된 LiDAR 입력에 대해 동작하는 교사 네트워크와, 원래 밀도 비정렬(non-aligned) LiDAR 입력으로 백본을 훈련하는 학생 네트워크로 구성됩니다.

- **Performance Highlights**: Waymo Open 데이터셋과 KITTI를 포함한 주행 벤치마크를 통해 실행된 실험 결과, 제안된 방법이 적은 타겟 데이터로 사전 훈련된 모델을 효과적으로 미세 조정(finetune)하여 정확도(accuracy)에서 상당한 향상을 이끌어낸 것으로 확인되었습니다.



### Rethinking the Expressiveness of GNNs: A Computational Model Perspectiv (https://arxiv.org/abs/2410.01308)
- **What's New**: 이번 논문에서는 그래프 머신 러닝에서 중요한 역할을 하는 그래프 신경망(Graph Neural Networks, GNNs)의 표현력에 대한 기존의 분석 방식의 문제점을 지적하며 새로운 접근법인 Resource-Limited CONGEST (RL-CONGEST) 모델을 제안합니다.

- **Technical Details**: RL-CONGEST 모델은 선택적 전처리(preprocessing) 및 후처리(postprocessing)를 포함하여 GNN의 표현력을 분석하는 프레임워크를 형성합니다. 또한, WL 테스트에서 해시 함수의 계산적 난이도(computational hardness) 및 가상 노드(virtual nodes)가 네트워크 용량을 감소시키는 역할에 대한 통찰도 제공합니다.

- **Performance Highlights**: 우리는 고차 GNNs가 1차 모델 검사(model-checking) 문제와 연관되어 있다는 점을 강조하며, 이는 GNN의 표현력에 대한 새로운 관점을 제공합니다.



### Emotion-Aware Response Generation Using Affect-Enriched Embeddings with LLMs (https://arxiv.org/abs/2410.01306)
- **What's New**: 이 연구는 자동화된 챗봇을 통한 심리치료 세션에서 공감이 가며 일관된 반응을 제공하는 방법을 제안합니다. 특히, 대형 언어 모델(LLMs)의 감정 및 맥락 인식을 향상시키기 위한 새로운 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 NRC Emotion Lexicon, VADER, WordNet, SentiWordNet과 같은 여러 감정 어휘집을 통합하여 LLAMA 2, Flan-T5, ChatGPT 3.0, ChatGPT 4.0와 같은 최신 LLM과 결합합니다. 데이터셋은 상담 및 심리치료 데이터베이스에서 수집한 2,000개 이상의 치료 세션 전사로 구성되어 있으며, 불안, 우울증, 트라우마, 중독 관련 논의를 포함하고 있습니다. 전사를 작은 조각으로 분할하고, BERT, GPT-3, RoBERTa를 사용하여 임베딩을 계산하여 의미적 및 정서적 뉘앙스를 캡처합니다. 이 임베딩은 FAISS 벡터 데이터베이스에 저장되어, 코사인 유사성을 기반으로 효율적인 유사성 검색 및 클러스터링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 감정 어휘집을 통합함으로써 모델의 공감, 일관성, 정보성 및 유창성 점수가 향상되는 것으로 나타났습니다. 이러한 발견은 심리치료 응용을 위한 LLM 성능 향상에서 감정 임베딩의 중요한 역할을 강조합니다.



### Speculative Coreset Selection for Task-Specific Fine-tuning (https://arxiv.org/abs/2410.01296)
Comments:
          20 pages, 4 figures, 14 tables

- **What's New**: 본 논문에서는 STAFF라는 새로운 공동 집합(coreset) 선택 방법을 제안합니다. 이는 특정 작업에 대한 LLM(Large Language Model) 파인튜닝(task-specific fine-tuning)을 통해 데이터 효율성과 선택 오버헤드를 크게 개선하는 것을 목표로 합니다.

- **Technical Details**: STAFF는 대상 LLM과 동일한 계열의 작은 모델을 활용하여 데이터 점수를 효율적으로 추정하고, 그 점수를 바탕으로 대상 LLM에서 검증하여 중요한 지역에 보다 많은 선택 예산을 할당하는 방식으로 구성됩니다. STAFF는 두 단계로 나뉘며, 첫 번째 단계인 탐색적 점수 계산(speculative score calculation)에서는 작은 모델을 파인튜닝하여 데이터 샘플의 중요도를 평가합니다. 두 번째 단계는 LLM 검증 및 선택(LLM Verification & Selection) 단계로, 여기서 샘플을 중요도 점수에 따라 나누고 검증합니다.

- **Performance Highlights**: STAFF는 3개의 LLM과 3개의 다운스트림 작업에서 평가되었으며, SOTA(stop-of-the-art) 방법보다 최대 54.3% 향상된 성능을 보여주었고, 선택 오버헤드를 최대 70.5%까지 줄였습니다. 특히 낮은 가지치기 비율(예: 20%)에서 STAFF가 전체 데이터셋보다 더 나은 파인튜닝 성능을 발휘하는 것으로 나타났습니다.



### Deep Unlearn: Benchmarking Machine Unlearning (https://arxiv.org/abs/2410.01276)
- **What's New**: 본 논문에서는 여러 벤치마크 데이터셋과 모델을 기준으로 18가지 최신 머신 언러닝(Machine Unlearning) 방법을 조사하였다. 이는 DNN(Deep Neural Network)에서의 MU가 성공적으로 적용될 수 있는 방법에 대한 포괄적인 연구가 부족한 상황에서 이루어졌다.

- **Technical Details**: 연구에서는 Masked Small Gradients (MSG)와 Convolution Transpose (CT) 방법이 데이터셋과 초기화에 관계없이 모델의 정확성과 실행 성능에서 일관되게 더 나은 성과를 보인다는 것을 밝혔다. 각 MU 방법은 10개의 초기화 동안 평가되었으며, 100K 모델에 걸쳐 검증되었다.

- **Performance Highlights**: 본 연구 결과, Masked Small Gradients, Convolution Transpose 등은 U-LiRA와 같은 공격에 대해 높은 저항력을 보여주었으며, 평가된 18가지 MU 방법 중에서도 매우 높은 성능을 보였다. 또한, 기존 MU 방법들과 비교할 때 NG+를 포함한 새롭고 더 나은 기준 시스템이 필요하다는 점을 강조하였다.



### Transformers Handle Endogeneity in In-Context Linear Regression (https://arxiv.org/abs/2410.01265)
Comments:
          30 pages

- **What's New**: 이번 연구에서는 transformers의 가능성을 탐구하며, in-context linear regression에서의 endogeneity 처리 능력을 분석합니다. 특히 transformers가 instrumental variables (IV)을 사용하여 endogeneity를 효과적으로 다룰 수 있는 메커니즘을 본질적으로 가지고 있다는 점이 주목할 만합니다.

- **Technical Details**: 연구에서는 transformer 아키텍처가 gradient 기반 bi-level 최적화 절차를 에뮬레이트(emulate)할 수 있음을 보여주고, 이는 널리 사용되는 두 단계 최소 제곱법(2SLS) 솔루션으로 수렴(converge)하는 것을 입증합니다. 또한, in-context pretraining 방식을 제안하고, 이론적으로 글로벌 최소값(global minimizer)이 작은 초과 손실(small excess loss)을 달성한다는 보장을 제공하였습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 훈련된 transformer는 endogeneity가 있는 상황에서도 2SLS 방법에 비해 더 강력하고 신뢰할 수 있는 in-context 예측 및 계수 추정치를 제공합니다.



### HelpSteer2-Preference: Complementing Ratings with Preferences (https://arxiv.org/abs/2410.01257)
Comments:
          26 pages, 3 figures

- **What's New**: 이 논문에서는 Bradley-Terry 스타일과 Regression 스타일의 리워드 모델을 비교하기 위한 고품질 데이터셋 'HelpSteer2'를 제공하고 있으며, 두 접근 방식의 효과를 면밀히 분석한 최초의 연구입니다.

- **Technical Details**: 리워드 모델은 언어 모델이 지침을 따르도록 하는 데 필수적이며, 두 가지 주요 접근 방식인 Bradley-Terry 스타일과 Regression 스타일로 훈련됩니다. 연구진은 두 스타일의 데이터를 적절히 맞추어 비교 검증을 수행하였으며, 인간이 작성한 정당화(Justification)가 포함된 Preference annotations를 사용합니다. 새로운 접근법으로 두 모델의 조합 방법을 제안하고, Llama-3.1-70B-Instruct 모델을 통해 베스트 성능을 기록하였습니다.

- **Performance Highlights**: 이 연구에서 훈련된 리워드 모델은 RewardBench에서 94.1점을 기록하였으며, 이는 140개 이상의 리워드 모델 중 최고의 역량을 지닌 모델입니다. 또한 RLHF(Reinforcement Learning from Human Feedback)에서 지침을 따르도록 모델을 정렬하는 데 효과적임을 입증하였습니다.



### AHP-Powered LLM Reasoning for Multi-Criteria Evaluation of Open-Ended Responses (https://arxiv.org/abs/2410.01246)
Comments:
          Accepted for EMNLP 2024 Findings

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)과 분석적 계층화 과정(Analytic Hierarchy Process, AHP)을 활용하여 개방형 질문에 대한 답변을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 두 단계로 나누어집니다: 기준 생성 단계와 평가 단계입니다. 기준 생성 단계에서는 질문에 대한 여러 평가 기준을 LLM을 통해 생성하고, 평가 단계에서는 각 기준에 따라 후보 답변을 쌍대 비교하여 최종 결정을 내립니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 4개의 기준 모델보다 인간의 판단과 더 밀접하게 일치하며, 정량적 지표인 일치 지수 및 부드러운 일치 지수에서 더 뛰어난 성능을 보였습니다.



### RGD: Multi-LLM Based Agent Debugger via Refinement and Generation Guidanc (https://arxiv.org/abs/2410.01242)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 기반으로 하는 코드 생성 및 자동 디버깅을 위한 새로운 아키텍처인 RGD(Refinement and Guidance Debugging)를 제안합니다. RGD는 여러 LLM 에이전트를 활용하여 코드 생성 과정을 멀티 스텝으로 분해하고, 반복적인 피드백과 자기 성찰을 통해 코드 개선을 가능하게 합니다.

- **Technical Details**: RGD 프레임워크는 세 가지 유형의 LLM 에이전트를 포함합니다: 가이드 에이전트(Guide Agent), 디버그 에이전트(Debug Agent), 피드백 에이전트(Feedback Agent). 각 에이전트는 특정 역할을 수행하며, 과정 중 코드의 성공 및 실패 사례를 분석하여 지속적으로 수정할 수 있는 능력을 갖추고 있습니다. 이 체계는 특정 작업 설명에서 생성된 가이드를 바탕으로 초기 코드를 생성하고, 이후 피드백을 통해 코드 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과 RGD는 HumanEval 데이터셋에서 9.8% 및 MBPP 데이터셋에서 16.2% 성능 향상을 달성하였으며, 기존의 최신 접근 방식과 전통적인 직접 프롬프트 방식에 비해 월등한 코드 생성 능력을 보여주었습니다.



### See Me and Believe Me: Causality and Intersectionality in Testimonial Injustice in Healthcar (https://arxiv.org/abs/2410.01227)
- **What's New**: 이 연구는 의료 환경에서 환자의 증언 불공정성을(testimonial injustice)를 정량화하기 위해 인과 발견(causal discovery) 방법론을 사용하였으며, 핵심적으로 인구 통계적 특성이 이러한 불공정성에 어떻게 기여하는지를 밝혀냈습니다.

- **Technical Details**: 연구는 FCI(Fast Causal Inference) 방법을 사용하여 환자의 의료 기록에서 불공정한 용어의 출현을 분석하고, 이러한 용어와 환자의 인구 통계적 특성(예: 나이, 성별, 인종) 간의 인과적 관계를 수립하는 구조적 인과 모델(Structural Causal Model, SCM)을 구축했습니다.

- **Performance Highlights**: 해당 연구는 사람의 특징이 불공정성 경험에 어떻게 교차적으로 작용하는지를 분석하고, 의료 서비스 향상과 신뢰 구축을 위한 디자인 원칙에 대한 통찰을 제시합니다.



### An uncertainty-aware Digital Shadow for underground multimodal CO2 storage monitoring (https://arxiv.org/abs/2410.01218)
- **What's New**: 이번 연구에서는 지질학적 탄소 저장(GCS)에 대한 불확실성을 고려한 디지털 섀도(Digital Shadow)를 설계하고 구현하기 위한 머신 러닝 기반의 데이터 동화(data assimilation) 프레임워크를 소개합니다. 이를 통해 CO2 주입 및 저장 작업의 모니터링을 지원하고자 합니다.

- **Technical Details**: 이 프레임워크는 베이esian 추론(Bayesian inference)을 기반으로 하며, 다중 모드 시계열 데이터에 조건화된 CO2 플룸의 후방 분포를 특성화하는 것을 목표로 합니다. 다중 모드 데이터 통합을 위해 시뮬레이션 기반 추론(Simulation-Based Inference, SBI) 및 앙상블 베이esian 필터링(Ensemble Bayesian Filtering) 기법들을 활용합니다.

- **Performance Highlights**: 컴퓨터 시뮬레이션 연구를 통해, 지층의 투과성(permeability) 필드에 대한 정보 부족이 디지털 섀도의 불확실성 정량화에 통합될 수 있음을 관찰했습니다. 이 연구는 불확실성을 인지한 확장 가능한 디지털 섀도의 개념을 처음으로 증명한 사례로 알려져 있습니다.



### RS-FME-SwinT: A Novel Feature Map Enhancement Framework Integrating Customized SwinT with Residual and Spatial CNN for Monkeypox Diagnosis (https://arxiv.org/abs/2410.01216)
Comments:
          37 Pages, 5 Tables, 10 Figures

- **What's New**: 이번 논문에서는 Monkeypox (MPox) 감지를 위한 혁신적인 하이브리드 접근법인 RS-FME-SwinT가 제안되었습니다. 이 방법은 Residual Learning과 Spatial Exploitation CNN, 그리고 맞춤형 Swin Transformer의 학습 능력을 통합하여 MPox 진단에 필요한 다중 스케일의 글로벌 및 로컬 상관 피쳐를 캡처합니다.

- **Technical Details**: RS-FME-SwinT 기법은 전이 학습 기반의 Feature Map Enhancement (FME) 기법을 사용하여 글로벌 정보를 캡처하기 위한 맞춤형 Swin Transformer와 텍스처 추출을 위한 residual blocks 및 로컬 대비 변화를 처리하기 위한 spatial blocks를 통합합니다. 새롭게 도입된 inverse residual blocks는 로컬 패턴 캡처를 효과적으로 수행하고 gradient 소멸 문제를 완화합니다.

- **Performance Highlights**: RS-FME-SwinT는 다양한 MPox 데이터셋에서 크로스 검증을 통해 기존의 최신 CNN 및 ViT 모델보다 우수한 성능을 보였습니다. MPox 탐지에서 정확도 97.80%, 민감도 96.82%, 정밀도 98.06%, F-score 97.44%의 성과를 기록했습니다. 이 모델은 의료 종사자들에게 신속하고 정확한 MPox 진단을 가능하게 할 수 있는 유용한 도구로 평가됩니다.



### From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging (https://arxiv.org/abs/2410.01215)
Comments:
          Code and data available at this https URL

- **What's New**: MGDebugger는 코드 디버깅을 위한 새로운 접근 방식으로, 다중 수준의 세분화된 오류를 해결하기 위해 계층적 구조를 사용합니다. 이는 기존 시스템들이 단일 단위로 처리했던 것과는 대조적입니다.

- **Technical Details**: MGDebugger는 코드를 하위 함수들의 계층적 트리 구조로 분해하고, 각 하위 함수에 대해 독립적인 디버깅을 진행합니다. 이 과정에서 LLM 기반의 Python 실행 시뮬레이터를 활용하여 변수 상태를 추적하고 오류를 정확하게 식별합니다.

- **Performance Highlights**: MGDebugger는 기존 디버깅 시스템들에 비해 HumanEval에서 18.9%의 정확도 향상을 이루었고, HumanEvalFix에서 97.6%의 수정 성공률을 달성했습니다. 이 시스템은 다양한 종류의 버그와 난이도 수준을 효과적으로 처리할 수 있는 강력함과 효율성을 보여줍니다.



### A versatile machine learning workflow for high-throughput analysis of supported metal catalyst particles (https://arxiv.org/abs/2410.01213)
- **What's New**: 본 연구는 나노입자(NPs)의 분석을 위한 새로운 AI 기반의 2단계 워크플로우를 제안하며, 이는 최신 단일 단계 객체 탐지(single-stage object detection) 및 대규모 비전 트랜스포머(large-scale vision transformer, ViT) 아키텍처의 프롬프트 엔지니어링(prompt engineering) 기법을 활용합니다.

- **Technical Details**: 제안된 방법론은 전송 전자 현미경(transmission electron microscopy, TEM) 및 스캐닝 TEM(stem) 이미지를 분석하여 금속 촉매의 입자 크기 분포를 고해상도 및 고처리량으로 분석하는 데 적용됩니다. 본 연구는 다양한 이종촉매 시스템에 걸쳐 NP를 검출 및 세분화(segmentation)하는 모델의 성능을 검증하였습니다.

- **Performance Highlights**: 제안된 AI 지원 NP 분석 워크플로우는 높은 일반화 능력을 보여주며, 비용이 많이 드는 모델 재훈련 없이 유사한 NP 세분화 작업에 쉽게 적용 가능합니다. YOLOv8x 모델을 통한 데이터 전송 학습 및 세분화 정확성 향상이 두드러집니다.



### Polyp-SES: Automatic Polyp Segmentation with Self-Enriched Semantic Mod (https://arxiv.org/abs/2410.01210)
Comments:
          Asian Conference on Computer Vision 2024

- **What's New**: 본 논문은 콜로노스코피(Colonoscopy) 이미지의 자동 폴립 세분화(Polyp Segmentation)를 위한 'Self-Enriched Semantic Model'이라는 혁신적인 방법을 제안합니다. 이 접근법은 기존의 방법들이 가진 한계를 극복하도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 Encoder, Decoder, Self-Enriched Semantic(SES) 모듈 등 세 가지 주요 구성 요소로 이루어져 있습니다. Encoder는 다중 스케일의 특징을 추출하고, Local-to-Global Spatial Fusion(LGSF) 메커니즘을 통해 로컬 및 글로벌 공간적 특징을 캡처하여 초기 전역 특징 맵을 생성합니다. 그 후, SES 모듈을 사용하여 추가적인 의미론적 정보를 더해 모델이 문맥을 더 잘 이해할 수 있도록 지원합니다.

- **Performance Highlights**: 제안된 방법은 다섯 가지 폴립 기준에서 최신 연구들에 비해 뛰어난 세분화 성능을 보이며 학습 및 일반화 능력에서 우수한 성과를 나타냈습니다.



### Were RNNs All We Needed? (https://arxiv.org/abs/2410.01201)
- **What's New**: 최근 Transformer 아키텍처의 한계 때문에 RNN(순환 신경망)에 대한 관심이 다시 높아지고 있습니다. 기존의 LSTM과 GRU를 발전시켜 새로운 경량 RNN(minLSTM, minGRU)을 제안합니다. 이들 모델은 파라미터 수가 크게 줄어들고 병렬 학습이 가능해졌습니다.

- **Technical Details**: 제안된 minLSTM과 minGRU는 기존 LSTM과 GRU의 입력, 망각, 업데이트 게이트에서 hidden state(은닉 상태) 의존성을 제거하여 BPTT(Through-Time)를 필요로 하지 않습니다. 이로써 훈련 성능이 512의 시퀀스 길이에서 175배 빨라집니다. 또한 이 모델들은 파라미터 수를 대폭 줄였습니다.

- **Performance Highlights**: 이 새로운 경량 RNN 모델들이 최신 sequence 모델들과 비슷한 성능을 보여주며, 과거의 RNN 구조에서 나온 성과를 재현할 수 있음을 입증합니다.



### Towards Inference-time Category-wise Safety Steering for Large Language Models (https://arxiv.org/abs/2410.01174)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 안전성을 강화하기 위한 새로운 접근법을 제시합니다. 기존의 교육 및 미세 조정 없이, 추론 단계에서 바로 사용할 수 있는 안전 지향 방법론을 탐구합니다.

- **Technical Details**: 연구에서는 두 가지 주요 방법을 소개합니다: (i) 카테고리별(특정 피해 유형) 스티어링 벡터를 계산하여 세밀한 조정이 가능하게 하고, (ii) 정보가 풍부한 스티어링 벡터를 추출하여 안전성을 확보하면서 생성 텍스트의 품질을 유지합니다. 이들은 전통적인 안전성 문제를 해결하기 위해 모델의 중간 레이어에서 조정됩니다.

- **Performance Highlights**: 다양한 LLM 및 데이터 셋에서 제안된 스티어링 방법의 효과를 입증하며, 안전성을 높이는 동시에 생성되는 텍스트의 품질을 유지하는 데 성공했습니다. 연구는 이러한 접근법이 대형 언어 모델의 안전한 작동을 위해 어떻게 활용될 수 있는지를 논의합니다.



### Recovering Manifold Structure Using Ollivier-Ricci Curvatur (https://arxiv.org/abs/2410.01149)
- **What's New**: ORC-ManL이라는 새로운 알고리즘이 소개되었습니다. 이 알고리즘은 Ollivier-Ricci curvature와 추정된 metric distortion을 기반으로 최근접 이웃 그래프에서 불필요한 엣지를 제거하는 방법입니다.

- **Technical Details**: 본 연구의 동기는 manifold learning에서 출발합니다. 데이터가 저차원 manifold에서 발생하는 noisy sample일 때, ambient space를 지나는 shortcut 엣지들은 데이터 manifold에 따라 있는 엣지보다 더 부정적인 Ollivier-Ricci curvature를 가지고 있다는 것을 보였습니다.

- **Performance Highlights**: ORC-ManL은 기존의 다양한 pruning 방법보다 좋은 성능을 발휘하며, 최근접 이웃 그래프를 입력으로 사용하는 여러 downstream geometric data analysis 작업에서 성능을 상당히 향상시킵니다. 특히 manifold learning, persistent homology, 차원 추정 등에서 평가되었으며, 단일 세포 RNA 시퀀싱 데이터의 clustering 및 manifold learning 향상에도 사용될 수 있음을 보여주었습니다.



### ProxiMix: Enhancing Fairness with Proximity Samples in Subgroups (https://arxiv.org/abs/2410.01145)
- **What's New**: 이번 연구에서는 기존의 mixup 방법과 새로운 편향 완화 알고리즘을 결합하여 보다 공정한 데이터 증강을 위한 사전 처리 전략을 제안합니다. 이를 통해 레이블 생성 개선을 목표로 하는 새로운 기법인 ProxiMix를 소개합니다.

- **Technical Details**: ProxiMix는 쌍(pairwise) 관계와 근접(proximity) 관계를 모두 유지하여 편향된 레이블이 발생하는 문제를 해결합니다. 연구는 세 가지 데이터셋과 세 가지 머신러닝 모델을 사용하여 ProxiMix의 효과를 검증하였습니다.

- **Performance Highlights**: ProxiMix는 기존의 pairwise mixup 기법보다 예측의 공정성과 재조정의 공정성 측면에서 높은 성능을 보였습니다. 특히, 원본 데이터셋의 레이블이 상당히 편향된 경우에 더욱 효과적이었습니다.



### Evaluating Deduplication Techniques for Economic Research Paper Titles with a Focus on Semantic Similarity using NLP and LLMs (https://arxiv.org/abs/2410.01141)
Comments:
          6 pages, 1 figure

- **What's New**: 이 연구는 경제 연구 논문의 제목으로 구성된 대규모 NLP 데이터셋의 효과적인 중복 제거(duplication) 기법을 조사하였습니다.

- **Technical Details**: 다양한 페어링 방법을 Levenshtein distance, cosine similarity 와 sBERT 모델을 포함하여 탐구하고, 제목만을 기반으로 중복을 탐지하기 위한 여러 기법을 구현하였습니다. 특히, 문자열 기반(method), 해시(mapping), 임베딩 기반(embedding) 방법을 사용하여 텍스트 데이터를 수치벡터로 변환하고 유사성을 비교하였습니다.

- **Performance Highlights**: 연구 결과, Levenshtein distance, cosine similarity 및 SBERT 모델을 활용한 2,000 쌍의 제목 비교를 통해 중복 제목의 낮은 존재 가능성을 확인하였습니다. 데이터 시각화는 세 가지 거리 측정 방법의 상관관계를 보여주었으며, 모든 방법에서 완벽히 동일한 결과를 얻지 못하였음을 나타냈습니다.



### nGPT: Normalized Transformer with Representation Learning on the Hyperspher (https://arxiv.org/abs/2410.01131)
- **What's New**: 새로운 신경망 아키텍처인 정상화된 Transformer(nGPT)를 제안하며, 이는 하이퍼스피어(hypersphere)에서의 표현 학습을 포함합니다. nGPT에서는 임베딩, MLP, 주의(attention) 행렬 및 히든 상태를 구성하는 모든 벡터가 단위 노름(normalized)으로 정규화됩니다. 적절한 훈련(step) 수를 4배에서 20배 줄여 같은 정확도를 성취하는데 걸리는 시간을 단축합니다.

- **Technical Details**: nGPT 구조에서 모든 벡터는 단위 노름 하이퍼스피어에 형성되어, 행렬-벡터 곱셈을 코사인 유사성을 나타내는 내적(dots)으로 간주합니다. 또한, nGPT는 각 계층(layer)에서 두 번째 학습률(eigen learning rate)에 의해 최적화를 제어하여 다단계 최적화(multi-step optimization)를 수행합니다. 이 방식은 훈련 과정의 유사성 추정(accuracy estimation)을 개선합니다.

- **Performance Highlights**: nGPT는 기존 모델보다 훈련 단계에서 4배에서 20배 더 적은 단계를 요구하며, 이는 더 빠른 수렴 속도를 의미합니다.



### Augmentation through Laundering Attacks for Audio Spoof Detection (https://arxiv.org/abs/2410.01108)
- **What's New**: 최근 텍스트-투-스피치(TTS) 기술의 발전으로 인해 목소리 클로닝(VC)이 더욱 현실감 있게 발전하고 있습니다. 이에 따라, ASVspoof 5 Challenge가 새로운 데이터베이스를 도입하여 다양한 악취 조건을 포함하게 되었습니다. 본 연구는 이러한 데이터베이스를 활용하여 Audio Spoof Detection의 성능을 평가하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 데이터 증강(data augmentation)을 이용하여 세탁 공격(laundering attacks)을 통해 Audio Spoof Detection(AASIST) 시스템을 훈련하고 ASVSpoof 5 데이터베이스에서 평가하였습니다. 실험에는 32가지 스푸핑 공격(A01-A32)과 11가지 코덱 및 압축 조건(C01-C11)이 포함됩니다. AASIST는 증강 데이터에서 훈련되고 실제 데이터를 통해 평가되었습니다.

- **Performance Highlights**: 결과적으로, AASIST는 A18, A19, A20, A26, A30 스푸핑 공격 및 C08, C09, C10 코덱/압축 조건에서 최악의 성능을 보였습니다. 또한, 4가지 메트릭(minDCF, actDCF, Cllr, EER)을 사용하여 성능 분석이 이루어졌습니다.



### softmax is not enough (for sharp out-of-distribution) (https://arxiv.org/abs/2410.01104)
Comments:
          Comments welcome. 14 pages, 7 figures

- **What's New**: 이 연구는 softmax 함수의 한계를 밝히고, 이를 개선하기 위한 새로운 접근 방법을 제안합니다. 이전의 믿음과는 달리, softmax는 샤프한 결정(Sharp Decision)을 내리는 데 필요한 강력한 일반화 능력을 갖추고 있지 않음을 보여줍니다.

- **Technical Details**: softmax 함수는 입력 값의 벡터를 확률 분포로 변환하는 데 사용되며, 온도 매개변수(Temperature Parameter) θ를 적용할 수 있습니다. 연구에서는 이론적으로 softmax의 샤프함에서의 분산(Dispersion) 현상을 증명하고, 적응형 온도(adaptive temperature)를 제안하여 softmax의 샤프함을 향상시키는 방법을 모색합니다.

- **Performance Highlights**: 기계 학습 모델이 훈련 시 사용한 문제 크기를 넘어서는 경우, attentional coefficients는 균일 분포로 분산되는 경향이 있음을 확인했습니다. 이는 Transformers의 Attention Head에서도 동일하게 나타나며, 추론 시 입력 크기가 증가함에 따라 발생합니다.



### Approximately Aligned Decoding (https://arxiv.org/abs/2410.01103)
Comments:
          9 pages main, 22 pages total

- **What's New**: 본 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 바람직하지 않은 출력을 감소시키는 새로운 방법을 제안합니다. 기존 방법들에 비해 계산 효율성을 높이면서도 출력 분포의 왜곡을 줄여 긴 텍스트 생성에서 어려운 제약을 충족할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 기존의 오류 완화 방법들과 비교하여 낮은 확률 출력의 증폭을 줄이면서 출력 분포를 유지합니다. 이를 통해 강화된 제약 조건이 있을 때 더 빠르게 수렴하는 성능을 보여줍니다. 또한 적응 샘플링 기법(Adaptive Sampling with Approximate Expected Futures, ASAp)을 사용하여 효과적으로 샘플을 추출합니다.

- **Performance Highlights**: 다수의 실험에서 제안된 방법이 ASAp와 유사한 과제 특정(performance specific) 성능을 나타내면서도, 어렵게 만족해야 하는 제약 조건들이 있을 때 기존 방법보다 훨씬 빠르게 수렴함을 보였습니다.



### Mechanic Maker: Accessible Game Development Via Symbolic Learning Program Synthesis (https://arxiv.org/abs/2410.01096)
Comments:
          11 pages, 8 figures, AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment

- **What's New**: 이번 논문에서는 프로그래밍 기술 없이 다양한 게임 메커니즘을 만들 수 있는 새로운 도구인 Mechanic Maker를 소개합니다. 기존의 게임 개발 도구들이 여전히 프로그래밍을 필요로 하거나 게임 제작의 종류에 큰 제약이 있는 것과 대조적으로, Mechanic Maker는 사용자 예제를 바탕으로 게임 메커니즘을 합성하는 백엔드 기호 학습 시스템에 의존합니다.

- **Technical Details**: Mechanic Maker는 사용자가 원하는 메커니즘을 보여줌으로써 코드를 생성하는 프로그램 합성(Program Synthesis) 기법을 통해 작동합니다. 이 도구는 사용자가 프로그래밍 지식 없이도 사용자 정의 메커니즘을 정의할 수 있게 해줍니다. 사용자는 키보드 입력으로 캐릭터를 이동시키거나, 특정 위치에 도달했을 때 객체가 생성되는 등의 다양한 메커니즘을 만들 수 있습니다.

- **Performance Highlights**: 사용자 연구를 통해 Mechanic Maker는 프로그래밍 능력과 관계없이 다양한 사용자에게 유용하다는 결과를 보여주었습니다. 이는 게임 개발의 민주화를 위한 지능형 도구의 잠재력을 시사합니다.



### Efficient and Private Marginal Reconstruction with Local Non-Negativity (https://arxiv.org/abs/2410.01091)
Comments:
          To appear at NeurIPS 2024

- **What's New**: 본 논문에서는 차별적 프라이버시를 위해 효율적이고 원칙적인 후처리 방법인 ReM (Residuals-to-Marginals)을 도입합니다. 이를 통해 노이즈가 포함된 측정값으로부터 주변 쿼리에 대한 답변을 재구성할 수 있습니다. 또한 GReM-LNN (Gaussian Residuals-to-Marginals with Local Non-negativity) 확장을 통해 가우시안 노이즈 하에서 일관성과 비부정성을 만족하는 마지널을 재구성하는 방법도 제안합니다.

- **Technical Details**: ReM 방법은 회귀 쿼리 기초를 기반으로 하여 노이즈가 포함된 측정값들로부터 마지널 쿼리에 대한 답변을 재구성하는 유효한 메커니즘을 사용합니다. 이 방법은 Kronecker 구조를 활용하여 효율적인 의사 역행렬(pseudo-inverse) 연산을 가능하게 하며, 이로 인해 높은 차원의 데이터 집합에서도 사용할 수 있습니다. GReM-LNN는 가우시안 노이즈 하에서 마지널들을 재구성하며, 일관성 및 비부정성을 보장하여 재구성된 답변의 오차를 줄입니다.

- **Performance Highlights**: ReM 및 GReM-LNN의 적용은 기존의 사적인 쿼리 응답 메커니즘인 ResidualPlanner 및 MWEM에서 오차를 크게 줄이고 확장성을 향상시키는 효과를 보여줍니다.



### From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems (https://arxiv.org/abs/2410.01066)
Comments:
          12 pages, 5 figures, 3 tables

- **What's New**: 이 논문은 LLM 기반 텍스트-투-SQL 시스템의 발전을 다룬 포괄적인 연구이며, 초기 규칙 기반 모델에서부터 최신 LLM 접근법까지의 역사를 설명합니다. 특히, 지식 그래프와의 통합이 문맥 정확성 및 스키마 연결에 미치는 영향을 연구합니다.

- **Technical Details**: 현재 기술은 두 가지 범주로 나눌 수 있습니다: corpus의 인-컨텍스트 학습과 파인 튜닝입니다. 이로 인해 제로샷(Zero-shot), Few-shot 학습 및 데이터 증강(Data augmentation)과 같은 접근 방식이 도출됩니다. 또한, 이를 통한 성능 평가를 위한 벤치마크 및 평가 메트릭을 논의합니다.

- **Performance Highlights**: 텍스트-투-SQL 시스템은 비전문가가 자연어로 데이터베이스를 쿼리할 수 있게 해주며, 이는 헬스케어, 물류 및 금융 시스템 등의 데이터 기반 의사결정 향상에 기여합니다. 논문은 모델의 신뢰성, 계산 효율성, 데이터 프라이버시와 같은 주요 과제를 강조하며 LLM 기반 텍스트-투-SQL 시스템의 향후 발전 가능성을 제시합니다.



### MOSEL: 950,000 Hours of Speech Data for Open-Source Speech Foundation Model Training on EU Languages (https://arxiv.org/abs/2410.01036)
Comments:
          Accepted at EMNLP 2024 Main Conference

- **What's New**: 새로운 연구에서는 유럽 연합(EU)의 24개 공식 언어에 대한 오픈 소스 기반의 음성 모델(Speech Foundation Models, SFMs) 구축을 위한 첫 단계로, 950,000시간의 음성 인식 데이터 및 레이블이 없는 음성 데이터 수집을 목표로 하고 있습니다.

- **Technical Details**: 연구팀은 오픈 소스 라이센스에 부합하는 자동 음성 인식(ASR) 데이터 세트와 비표시 음성 코퍼스를 조사하여 EU 언어에 사용할 수 있는 데이터를 수집했습니다. 이 데이터는 'MOSEL' 프로젝트로 이름 붙여져 GitHub에서 공개적으로 이용 가능하며, 추가로 441,000시간의 비표시 데이터에 대한 자동 전사본이 CC-BY 라이센스 하에 생성되었습니다.

- **Performance Highlights**: 이 연구는 자원 부족 언어인 말타어를 대상으로 한 실험을 통해 수집된 데이터가 실제 ASR 모델 훈련에 효과적으로 사용될 수 있음을 보여주었습니다.



### Can visual language models resolve textual ambiguity with visual cues? Let visual puns tell you! (https://arxiv.org/abs/2410.01023)
Comments:
          Accepted as main paper in EMNLP 2024

- **What's New**: 이 논문은 멀티모달 간섭(multi-modal interference)- 즉, 언어와 이미지를 포함하는 다양한 형태의 입력을 결합하여 텍스트의 모호성을 해소하는 새로운 벤치마크인 UNPIE(Understanding Pun with Image Explanations)를 소개합니다. 이벤치마크는 1,000개의 재미있는 말장난(pun)과 해당 의미를 설명하는 이미지로 구성되어있으며, 이를 통해 멀티모달 리터러시(multi-modal literacy)를 평가합니다.

- **Technical Details**: UNPIE 데이터셋은 말장난을 중심으로 구성되어 있으며, 텍스트-이미지 쌍을 통해서 리터러시를 측정하기 위한 세 가지 테스트(pun grounding, disambiguation, reconstruction)를 제안합니다. 이 연구는 기존의 단일 텍스트 모델과 비교해 새로운 시각적 맥락을 통해 다루어지는 모델들의 능력을 평가합니다. 이러한 모델은 고차원적 작업일수록 성과가 향상되는 경향을 보였습니다.

- **Performance Highlights**: UNPIE의 실험 결과에 따르면, VLM(Visual-Language Models)과 Socratic 모델 둘 다 단순 텍스트 모델보다 멀티모달 정보를 활용할 때 성능이 개선되었습니다. 특히, 더 높은 난이도의 작업에서 이러한 경향이 두드러지게 나타났고, VLM은 Socratic 모델보다 더 나은 성능을 보였습니다.



### Robust Guided Diffusion for Offline Black-Box Optimization (https://arxiv.org/abs/2410.00983)
Comments:
          21 pages

- **What's New**: 오프라인 블랙 박스 최적화 분야에 있어, 본 논문에서는 'Robust Guided Diffusion' (RGD)라는 새로운 프레임워크를 제안합니다. 이는 프로시(Proxy)와 프로시-프리(diffusion-free) 디퓨전의 장점을 통합하여 효과적인 조건부 생성을 가능하게 합니다.

- **Technical Details**: RgD는 두 가지 주요 구성 요소로 구성됩니다. 첫째, 'proxy-enhanced sampling' 모듈을 도입하여 프로시로부터의 명시적 가이드를 프로시-프리 디퓨전으로 통합하고, 샘플링 과정을 향상시킵니다. 둘째, 'diffusion-based proxy refinement' 모듈을 통해 프로시의 견고성과 신뢰성을 높이는 정규화 전략을 설계하였습니다.

- **Performance Highlights**: RGD는 다양한 디자인 벤치 작업에서 최첨단 성능을 달성하며 그 효능을 강조합니다. 이를 통해 높은 성능의 샘플 생성을 가능하게 하고 있습니다.



### Heterogeneous sound classification with the Broad Sound Taxonomy and Datas (https://arxiv.org/abs/2410.00980)
Comments:
          DCASE2024, post-print, 5 pages, 2 figures

- **What's New**: 이 논문은 고차원적인 변동성을 갖는 이질적인 소리의 자동 분류를 위한 방법론을 탐구합니다. Broad Sound Taxonomy(BST)를 기반으로 한 새로운 분류 체계를 제안하며, 이를 통해 소리 데이터를 더 효과적으로 분류할 수 있는 가능성을 제시합니다.

- **Technical Details**: BST는 28개의 클래스를 포함한 2단계 세분화된 계층 구조로 설계되었습니다. 데이터셋은 수동 주석을 통해 구축되었으며, k-NN 분류기와 같은 전통적인 기계 학습 방법론을 사용하여 성과를 평가합니다. 소리의 특징으로는 acoustically derived sound representations와 pretrained deep neural networks를 통한 임베딩을 비교합니다.

- **Performance Highlights**: 실험 결과, acoustic 및 semantic 정보를 인코딩하는 오디오 임베딩이 분류 과제에서 높은 정확도를 달성했음을 보여줍니다. 또한, 분류 오류 분석을 통해 개선 가능성을 제시하고자 합니다.



### Towards Full-parameter and Parameter-efficient Self-learning For Endoscopic Camera Depth Estimation (https://arxiv.org/abs/2410.00979)
Comments:
          WiCV @ ECCV 2024

- **What's New**: 이 논문에서는 내시경 심도 추정을 위한 심층 기초 모델의 적응 방법을 제안합니다. 이를 통해 기존의 낮은 순위 서브스페이스에 제한되는 방식에서 벗어나 전체 파라미터 기반의 효율적인 학습 프레임워크를 구현합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 주의(attention), 컨볼루션(convolution), 다층 퍼셉트론(mlp)의 서브스페이스를 동시에 적응시킵니다. 두 번째 단계에서는 메모리 효율적 최적화(memory-efficient optimization)를 통해 여러 서브스페이스를 결합하여 성능을 더욱 개선합니다.

- **Performance Highlights**: SCARED 데이터셋을 사용한 초기 실험에서는 첫 번째 단계에서 Sq Rel, Abs Rel, RMSE 및 RMSE log의 성능이 각각 10.2%에서 4.1%로 향상되었습니다. 이는 최신의 심도 모델에 비해 개선된 결과를 보여줍니다.



### GAMMA-PD: Graph-based Analysis of Multi-Modal Motor Impairment Assessments in Parkinson's Diseas (https://arxiv.org/abs/2410.00944)
Comments:
          Accepted by the 6th Workshop on GRaphs in biomedicAl Image anaLysis (GRAIL) at the 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024). 12 pages, 3 figures, 2 tables, Source Code: this https URL

- **What's New**: 본 논문에서는 Parkinson's Disease (PD)의 다중 모달 클리닉 데이터 분석을 위한 새로운 이질적 하이퍼그래프 융합 프레임워크인 GAMMA-PD를 제안합니다. GAMMA-PD는 이미징(imaging) 및 비이미징(non-imaging) 데이터를 통합하여 환자 프로필과 증상 하위 유형 간의 유사성을 보존하며, 높은 차원의 비대칭 정보를 학습합니다.

- **Technical Details**: GAMMA-PD는 의료 지식 클러스터링을 기반으로 한 도메인 특화 하이퍼엣지(hyperedge) 유형을 도입하여 임상적으로 관련된 패턴을 학습합니다. 또한, 피처 기반의 주의 가중치 메커니즘을 설계하여 각 예측 작업에 대해 가장 중요한 피처와 관계를 식별하고 우선순위를 매깁니다. 이를 통해 동적 특성을 보존한 채로 환자 간의 특성의 연관성을 학습하고, 임상적으로 유의미한 설명을 생성합니다.

- **Performance Highlights**: Parkinson's Progression Markers Initiative (PPMI) 및 사설 데이터 세트를 통한 평가에서 GAMMA-PD는 PD의 운동 장애 증상을 예측하는 데 있어 성능 향상을 보여줍니다. 또한, 이 모델은 다중 모달 의료 데이터를 활용하여 PD 증상 및 질병의 이질성 분석에 있어 임상적 설명과 해석 가능성을 제공합니다.



### StreamEnsemble: Predictive Queries over Spatiotemporal Streaming Data (https://arxiv.org/abs/2410.00933)
Comments:
          13 pages

- **What's New**: 본 논문은 스페이셜 템포럴(spatiotemporal, ST) 데이터 스트림에 대한 예측 쿼리를 처리하기 위한 새로운 접근법인 StreamEnsemble을 제안합니다. 이 방법은 기계 학습 모델을 동적으로 선택하고 할당하여 데이터 분포의 변화를 효과적으로 처리합니다.

- **Technical Details**: StreamEnsemble은 ST 데이터의 시계열 분포와 각 모델의 특성에 따라 기계 학습 모델을 선정하여 조합하는 방법입니다. 이 방법은 데이터가 유입되는 시점에서의 개별 시계열 윈도우의 분포를 바탕으로 유사한 시계를 클러스터링하고 최적 모델을 식별하는 일반화 오류 추정 기술을 사용합니다.

- **Performance Highlights**: 실험 결과, StreamEnsemble은 전통적인 앙상블 방법 및 단일 모델 접근법에 비해 10배 이상의 예측 오차 감소를 달성하며, ST 데이터 스트림의 복잡한 변동성을 처리하는데 있어 월등한 성능을 보여줍니다.



### ACEV: Unsupervised Intersecting Manifold Segmentation using Adaptation to Angular Change of Eigenvectors in Intrinsic Dimension (https://arxiv.org/abs/2410.00930)
Comments:
          14 pages, 7 figures, 7 tables

- **What's New**: 본 논문에서 제안된 방법은 교차하는 매니폴드 분할(Intersecting manifold segmentation)에 중점을 두고 있으며, 매니폴드의 내부 차원(intrinsic dimension) 및 구조를 학습하여 교차 영역에 존재하는 데이터 포인트를 효과적으로 식별합니다.

- **Technical Details**: 제안된 방법(ACEV)은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 비교차(non-intersecting) 매니폴드를 분할하고, 두 번째 단계에서는 개별 교차 매니폴드를 분할합니다. 이 방법은 로컬 데이터 분산(local data variances)을 측정하고 벡터 방향을 파악하여 매니폴드의 내부 차원을 결정합니다. 또한 지수 이동 평균(exponential moving averages)을 이용하여 부모와 자식의 방향 벡터 간의 각도 차이를 조정하여 교차 영역을 탐지합니다.

- **Performance Highlights**: ACEV 방법은 14개의 실제 데이터셋에서 18개의 최신 상태(SOTA) 매니폴드 분할 방법보다 ARI와 NMI 점수에서 더 나은 성능을 보였으며, 시간 복잡도(time complexity)가 낮고 안정성이 높은 결과를 나타냈습니다.



### IBM Quantum Computers: Evolution, Performance, and Future Directions (https://arxiv.org/abs/2410.00916)
- **What's New**: IBM Quantum이 1,121 qubit의 Condor 프로세서를 공개하며 1,000-qubit 장벽을 넘어선 기념비적인 발전을 이룩했다. 이 프로세서는 IBM Quantum의 발전 여정을 종합적으로 담고 있으며, noise intermediate-scale quantum(NISQ) 시대에서 fault-tolerant quantum computing으로의 전환을 탐구했다.

- **Technical Details**: IBM Quantum의 하드웨어는 superconducting qubits를 기반으로 하고 있으며, Canary, Falcon 및 Condor와 같은 여러 세대의 프로세서가 개발되어왔다. 최근에는 Heron 프로세서가 133 qubit의 새로운 기술로 이전 모델에 비해 3-5배의 성능 개선을 보였다.

- **Performance Highlights**: IBM Quantum은 95% 이상의 가동 시간을 자랑하며, Condor 프로세서는 1,121개의 superconducting qubits가 장착되어 있다. 이들은 안정성이 뛰어나며, 다음 세대의 quantum 시스템으로써 향후 양자 컴퓨터의 성능과 응용 가능성을 넓힐 예정이다.



### Scaling Optimal LR Across Token Horizons (https://arxiv.org/abs/2409.19913)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델) 훈련 시 토큰 지평(token horizon)에 따른 최적 학습률(learning rate, LR)의 변화를 대규모 실험을 통해 조사했습니다. LLM 훈련에 있어 하이퍼파라미터 전이(hyperparameter transfer)가 토큰 지평을 가로막고 있는 중요한 문제로 부각되었습니다.

- **Technical Details**: 이 연구에서는 최적 LR이 토큰 지평에 강하게 의존하며, 긴 훈련 기간에는 더 작은 LR이 필요하다는 점을 보여줍니다. 또한, 최적 LR은 스케일링 법칙(scaling law)을 따르는데, 이를 통해 긴 토큰 지평에 대한 최적 LR을 짧은 지평에서 추론할 수 있습니다. 실험은 Megatron 코드베이스(Megatron codebase)와 RefinedWeb 데이터셋을 바탕으로 진행되었습니다.

- **Performance Highlights**: LLama-1 모델이 너무 높은 LR을 사용했다는 것을 증명하며, 이로 인해 성능 저하가 발생했다는 점을 강조합니다. 이 연구는 데이터 크기 간 하이퍼파라미터 전이가 LLM 훈련에서 간과된 중요한 구성 요소임을 주장합니다.



New uploads on arXiv(cs.LG)

### On the expressiveness and spectral bias of KANs (https://arxiv.org/abs/2410.01803)
Comments:
          17 pages, 5 figures

- **What's New**: 최근 Kolmogorov-Arnold Networks (KAN)이 깊은 학습 모델의 주요 아키텍처인 multi-layer perceptron (MLP)의 대안으로 제안되었습니다. 본 연구에서는 KAN과 MLP의 이론적 비교를 재조명하며, KAN이 MLP에 비해 우수한 표현력과 근사 능력을 가진다고 주장합니다.

- **Technical Details**: KAN은 B-splines를 사용하여 학습된 비선형성을 매개변수화하며, KAN의 다층 학습 기능이 고주파 구성 요소의 학습 과정을 개선하는 데 기여함을 보여줍니다. KAN은 MLP보다 낮은 주파수에 대한 편향이 적으며, 전반적으로 KAN이 어떻게 다양한 하이퍼파라미터를 조정할 수 있는지에 대한 실질적인 통찰을 제공합니다.

- **Performance Highlights**: KAN은 다양한 과제를 수행하는 데 있어 MLP보다 스펙트럴 편향을 덜 겪으며, 이는 과학적 문제 해결에 더 효과적일 수 있음을 확인했습니다. 우리의 실험 결과 KAN이 MLP보다 더 넓은 범위의 주파수에서 학습할 수 있는 가능성을 보여주어 KAN의 성능을 뒷받침하고 있습니다.



### PROXI: Challenging the GNNs for Link Prediction (https://arxiv.org/abs/2410.01802)
- **What's New**: 최근 논문에서는 GNN(그래프 신경망)의 성능을 전통적인 기계 학습(ML) 모델인 PROXI와 비교하여 링크 예측(link prediction) 작업에 대한 새로운 접근 방식을 제안합니다. PROXI 모델은 노드 쌍의 그래프 및 속성 공간에서 근접 정보를 활용하여 링크 형성 예측을 보다 효과적으로 수행합니다.

- **Technical Details**: PROXI는 구조적 근접성(structural proximity)과 도메인 근접성(domain proximity)의 두 가지 유형의 정보를 융합하여 링크 예측 문제를 이진 분류로 처리합니다. 이 모델은 20개의 지수를 통해 SOTA(SOTA: state-of-the-art) GNN 모델들과 경쟁력 있는 결과를 생성하며, 기존 GNN 모델에 통합하여 성능을 11%까지 향상시킬 수 있습니다.

- **Performance Highlights**: PROXI는 동질적(homophilic) 및 이질적(heterophilic) 네트워크 모두에서 우수한 성능을 보이며, 기존 GNN 모델보다 뛰어난 성능을 나타냅니다. 이러한 결과는 현재의 GNN 모델이 전통 모델에 비해 크게 향상되지 않았음을 시사하며, GNN의 잠재력을 이끌어내기 위한 새로운 접근 방식의 필요성을 강조합니다.



### Bellman Diffusion: Generative Modeling as Learning a Linear Operator in the Distribution Spac (https://arxiv.org/abs/2410.01796)
Comments:
          Paper under review

- **What's New**: 이번 연구는 Deep Generative Models (DGMs)가 Markov Decision Processes (MDPs)와 분포적 강화 학습(distributional Reinforcement Learning)의 적용에서 비선형성으로 인해 발생하는 격차를 해결하기 위한 Bellman Diffusion이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Bellman Diffusion은 MDP에서 비선형성을 유지하기 위해 gradient와 scalar field 모델링을 사용합니다. 이 방법은 새로운 stochastic differential equation (SDE)을 통해 샘플링을 진행하며, neural network proxies를 최적화하기 위한 divergence-based training 기법을 채택합니다.

- **Performance Highlights**: 실험 결과, Bellman Diffusion은 정확한 field estimations을 달성하고 이미지 생성에서도 뛰어난 성능을 보이며, 전통적인 histogram 기반 기준 방식보다 분포적 RL 작업에서 1.5배 빠른 수렴 속도를 기록했습니다.



### Knowledge-Driven Feature Selection and Engineering for Genotype Data with Large Language Models (https://arxiv.org/abs/2410.01795)
- **What's New**: 이 논문은 작은 해석 가능한 변이 특성(set of variant features)을 바탕으로 복잡한 유전적 기초를 가진 표현형(phenotype)을 예측하는 기존의 문제점들을 해결하기 위한 새로운 지식 기반 프레임워크인 FREEFORM을 개발했습니다. 이 프레임워크는 고차원 유전자형(genotype) 데이터에서 LLMs(대형 언어 모델)의 지식을 활용하여 특성을 선택 및 엔지니어링하는 방식을 제공합니다.

- **Technical Details**: FREEFORM은 체인 오브 씽킹(chain-of-thought) 및 앙상블(ensembling) 원리를 사용하여 LLMs의 내재적 지식을 바탕으로 유전형 데이터의 특성을 선택하고 엔지니어링합니다. 이 연구에서 두 가지 유전자형-표현형 데이터셋, 즉 유전적 조상(genetic ancestry)과 유전성 청각 손실(hereditary hearing loss)에서 평가를 수행하였으며, 이는 데이터 기반 방법들보다 월등히 높은 성능을 보였습니다.

- **Performance Highlights**: FREEFORM은 특히 데이터 샘플이 부족한(low-shot) 환경에서 기존 데이터 기반 방법들보다 더 많은 성과를 보여 주목받았습니다. 또한, 이 프레임워크는 예측 성능을 향상시키면서도 해석 가능성을 유지하고, 데이터 차원을 줄이며, 지식 기반 접근 방식을 통해 기존의 문제들을 해결하는 잠재력을 demonstrated 합니다.



### Investigating on RLHF methodology (https://arxiv.org/abs/2410.01789)
Comments:
          23 pages, 6 figures, 6 tables

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 인간 선호도에 맞춘 정렬(alignment) 방법을 탐구합니다. 다중 레이어 학습과 강화 학습(Reinforcement Learning)을 통해 LLM의 성능을 개선하는 다양한 접근 방식을 제안합니다.

- **Technical Details**: Preference Model을 훈련하기 위해 필요한 데이터셋을 수집하고, 이를 통해 LLM을 인간 선호도에 맞게 조정합니다. 논문에서는 Direct Preference Optimization(DPO) 방법을 사용하여 별도의 Preference Model 없이 LLM을 직접 선호도 데이터셋으로 학습시키는 경험을 공유합니다.

- **Performance Highlights**: 연구진들은 퍼플렉서티 필터링(perplexity filtering)을 통해 데이터셋 수집 과정을 보다 용이하고 비용 효과적으로 만들 수 있다고 주장하며, 이를 통해 LLM을 보다 적합하게 산출할 수 있는 가능성을 제시합니다.



### Learning To Solve Differential Equation Constrained Optimization Problems (https://arxiv.org/abs/2410.01786)
- **What's New**: 이 논문은 차분 방정식(Differential Equations, DE) 제약 최적화 문제를 해결하기 위해 새로운 학습 기반 접근 방식을 소개합니다. 이 방식은 프록시 최적화(proxy optimization) 기법과 신경망 차분 방정식(neural differential equations)을 결합하여, 제어 전략을 근사하는 네트워크와 관련 DEs를 해결하는 네트워크로 구성된 이중 네트워크 아키텍처를 사용합니다.

- **Technical Details**: 제안된 방법은 전통적인 최적화 접근방식의 계산적 도전과제를 해결하기 위해, DE 가 대상이 되는 제어 전략을 근사하고 해당 DE를 해결하는 두 개의 신경망을 사용하는 이중 네트워크 아키텍처입니다. 이 통합된 접근 방식은 실제 시간에 가까운 상황에서 동적 제약을 고려하여 최적의 전략을 근사할 수 있습니다.

- **Performance Highlights**: 실험 결과, 에너지 최적화 및 금융 모델링 문제에서 제안된 방법이 동적 제약을 완전히 준수하며, 시스템의 동적 방정식을 명시적으로 모델링하지 않는 다른 방법에 비해 최대 25배 더 정확한 결과를 제공함을 보여주었습니다.



### Composing Global Optimizers to Reasoning Tasks via Algebraic Objects in Neural Nets (https://arxiv.org/abs/2410.01779)
- **What's New**: 이 논문에서는 2층 신경망의 솔루션 공간 내의 풍부한 대수 구조를 입증하고, 이는 쿼드러틱(Quadratic) 활성화 함수와 $L_2$ 손실을 바탕으로 하는 이유 추론(task) 문제에서의 학습에 적용됩니다. 이러한 구조는 손실의 일부만 만족하는 부분 솔루션으로부터 전역 최적 솔루션(global optimal solution)을 분석적으로 구성할 수 있게 해줍니다. 이 프레임워크는 CoGO(Composing Global Optimizers)라고 명명되었습니다.

- **Technical Details**: 연구에서는 2층 신경망의 가중치 공간이 반환환(semi-ring) 대수 구조를 갖고 있으며, 손실 함수가 단항 가능성(monomial potentials)으로 구성되어 있다는 점을 명시했습니다. 이러한 단항 가능성은 환 동형사상(ring homomorphism)으로, 부분 솔루션을 환 덧셈과 곱셈을 통해 전역 솔루션으로 구성할 수 있도록 합니다. 또한, 경험적으로 얻은 솔루션의 약 95%가 이론적으로 예측된 구조와 정확히 일치함을 보여주었습니다.

- **Performance Highlights**: 이론적 분석에 따르면, 높은 차수의 전역 최적화기는 훈련 동역학(training dynamics)에 불리하며, 과도한 매개변수화(over-parameterization)가 훈련 동역학을 비독립적으로 만들어 성능을 개선하는 경향이 있어 고차 전역 최적화기를 선호하지 않음을 보였습니다. 이 연구는 모델 학습 내에서 대수적 구조를 발견하고, 이를 통해 모듈라 덧셈과 같은 추론 문제에 대한 솔루션을 분석할 수 있는 첫 사례로 평가됩니다.



### TopER: Topological Embeddings in Graph Representation Learning (https://arxiv.org/abs/2410.01778)
Comments:
          17 pages, 7 figures

- **What's New**: 이번 논문에서는 Topological Evolution Rate (TopER)라는 새로운 저차원 그래프 임베딩 방식을 소개합니다. 이는 Persistent Homology를 활용하여 그래프 하위 구조의 진화율을 계산함으로써 직관적이고 해석 가능한 시각화를 제공합니다. TopER은 경쟁력 있는 그래프 클러스터링 및 분류 성능을 보여줍니다.

- **Technical Details**: TopER은 Persistent Homology의 필터링 프로세스를 최적화하여 그래프 하위 구조의 진화 정보를 효율적으로 캡처합니다. 이 접근 방식은 적은 계산 비용으로도 안정적인 임베딩을 생성할 수 있으며, 여러 필터링 함수에 걸쳐 확장 가능성이 큽니다. 또한, 이 방법은 이론적 안정성을 보장합니다.

- **Performance Highlights**: TopER은 다양한 벤치마크 데이터셋에서 실험을 통해 기존의 최첨단 모델들과 비교하여 일관되게 경쟁력 있는 또는 우수한 성능을 달성했습니다. 그 결과 TopER은 클러스터와 아웃라이어를 명확히 시각화하여 개별 그래프 데이터셋과 여러 데이터셋 간 비교 분석을 용이하게 합니다.



### Trained Transformer Classifiers Generalize and Exhibit Benign Overfitting In-Contex (https://arxiv.org/abs/2410.01774)
Comments:
          34 pages

- **What's New**: 이 논문은 선형 분류 작업에 대해 훈련된 선형 변환기(linear transformers)의 행동을 다룹니다. 특히, 훈련 과정에서 어떤 유형의 사전 훈련 작업이 있어야 좋은 일반화를 이루는지 분석하였고, 'benign overfitting in-context' 현상을 보여 주었습니다.

- **Technical Details**: 훈련된 변환기 모델은 랜덤 선형 분류 작업에 대해 경량화된 선형 주의(attention) 모델을 사용하여 기울기 하강법(gradient descent)으로 최적화됩니다. 전반적인 프레임워크 안에서, 신호 대 잡음비(SNR) 변화를 허용하고, 레이블 플리핑(label flipping) 잡음을 고려하여 실험하였습니다.

- **Performance Highlights**: 훈련된 선형 변환기는 복잡한 작업에 대해 일반화가 가능하며, 특별한 조건 하에 'benign overfitting' 현상을 나타낼 수 있습니다. 이는 테스트 시 레이블 플리핑 잡음이 존재하더라도, 변환기는 훈련 데이터를 기억하면서도 깨끗한 테스트 샘플에 대해 근사 최적화 결과를 도출해냅니다.



### Bayesian Binary Search (https://arxiv.org/abs/2410.01771)
- **What's New**: 이번 연구는 Bayesian Binary Search (BBS)라는 새로운 확률적 변형 알고리즘을 제시합니다. BBS는 전통적인 binary search의 절반을 임의의 미드포인트 대신 확률 밀도 기준으로 나누는 방식을 적용하여 검색 공간의 학습된 분포를 이용해 검색을 안내합니다.

- **Technical Details**: BBS는 supervised probabilistic machine learning 기법(예: Gaussian process regression, Bayesian neural networks, quantile regression)과 unsupervised learning 알고리즘(예: Gaussian mixture models, kernel density estimation (KDE), maximum likelihood estimation (MLE))을 활용하여 검색 공간의 밀도를 추정합니다. 이를 통해 우리는 PDF를 생성 후, 변형된 이진 검색 알고리즘에서 이 PDF의 중간에서 시작하며, 확률 밀도 공간을 이분화합니다.

- **Performance Highlights**: BBS는 다양한 분포에서 시뮬레이션 데이터와 실세계의 비트코인 라이트닝 네트워크에 있는 채널 균형 탐색 사례에서 높은 효율성을 보여 주었습니다. 실제 운영 환경에서 BBS 알고리즘을 배포하여 성과를 입증하였습니다.



### Explainable Earth Surface Forecasting under Extreme Events (https://arxiv.org/abs/2410.01770)
- **What's New**: 이 논문에서는 변화하는 기후로 인한 극단적인 사건을 예측하고 모델링하기 위해 DeepExtremeCubes 데이터셋을 활용하여 고차원의 지구 관측 데이터를 처리하는 새로운 방법론을 소개합니다. 특히, 2016년 1월부터 2022년 10월까지의 약 40,000개의 Sentinel-2 미니큐브를 포함하며, 극단 사건 및 기상 데이터를 라벨링하여 제공하는 데이터셋은 예측 모델링에 효과적으로 사용되었습니다.

- **Technical Details**: DeepExtremeCubes 데이터셋을 기반으로 한 컨볼루션 Long Short-Term Memory (convLSTM) 아키텍처를 통해, 향후 반사율과 식물 영향을 예측하는 모델이 개발되었습니다. 본 연구에서는 특히 kernel normalized difference vegetation index (kNDVI)를 활용하여 모델의 성능을 평가하였으며, 테스트 세트에서 R² 점수 0.9055를 달성하였습니다.

- **Performance Highlights**: 2020년 10월 중남부 아메리카에서 발생한 복합 더위와 가뭄 사건을 분석하는 과정에서, 평균 기온과 표면 기압이 정상적인 조건에서 가장 우수한 예측 변수임을 발견했습니다. 반면, 사건이 발생하는 동안 최소의 증발 이상 및 표면 잠열 유속이 주요한 예측 변수로 작용함을 확인했습니다. 본 연구로 인해 심층 학습 모델이 고차원 원격 감지 데이터에서 xAI(explainable AI)를 적용하고 시각화한 첫 번째 사례로 기록될 것입니다.



### Decision-Focused Uncertainty Quantification (https://arxiv.org/abs/2410.01767)
- **What's New**: 본 논문에서는 다운스트림 (downstream) 의사결정 손실 함수를 고려한 예측 집합을 생성하는 새로운 프레임워크를 개발하였습니다. 이는 고위험 의사결정에 더 적합한 정보를 제공하는 예측을 가능하게 합니다.

- **Technical Details**: 이 연구는 conformal prediction을 기반으로 하며, 사용자 지정 유틸리티 함수에 따라 다운스트림 결정 문제에 대한 정보를 통합하는 방법을 제안합니다. 이 방법은 통계적 커버리지 보장을 유지하면서 사용자가 정의한 '결정 손실'을 최소화하는 예측 집합을 생성하는 알고리즘 두 가지를 제안합니다.

- **Performance Highlights**: 여러 데이터 세트와 유틸리티 메트릭을 통해 실험한 결과, 제안된 방법이 기존의 standard conformal methods보다 현저히 낮은 결정 손실을 달성하는 것으로 나타났습니다. 의료 진단의 실제 사례에 대한 적용도 보여주며, 피부 질환 진단에서 coherent (일관된) 진단 의미를 가지고 예측 집합을 생성한다는 점에서 성공적임을 입증하였습니다.



### TorchSISSO: A PyTorch-Based Implementation of the Sure Independence Screening and Sparsifying Operator for Efficient and Interpretable Model Discovery (https://arxiv.org/abs/2410.01752)
- **What's New**: 이번 논문에서는 TorchSISSO라는 새로운 Python 구현을 소개합니다. 이 구현은 기존의 FORTRAN 기반 SISSO의 성능을 개선하여 머신 러닝 연구자들이 더 넓은 범위의 과학적 응용에서 사용할 수 있도록 합니다.

- **Technical Details**: TorchSISSO는 PyTorch 프레임워크에 내장된 SISSO 알고리즘의 사용자 친화적인 Python 패키지입니다. 이 패키지는 GPU 가속과 병렬 처리 기능을 활용하여 계산 시간을 크게 단축시킵니다. 사용자는 특징 확장 프로세스를 손쉽게 수정할 수 있어 원래 FORTRAN 구현의 제한을 극복할 수 있습니다.

- **Performance Highlights**: TorchSISSO는 다양한 작업에서 기존의 SISSO와 동등하거나 그 이상의 성능을 발휘하며, 특히 계산 시간을 크게 줄이는 이점을 보여줍니다. 실험을 통해 TorchSISSO가 FORTRAN 기반 SISSO가 실패하는 사례에서도 올바른 기호 표현을 찾아낼 수 있음을 입증했습니다.



### Not All LLM Reasoners Are Created Equa (https://arxiv.org/abs/2410.01748)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 수학 문제 해결 능력, 특히 성과의 차이를 평가했습니다. 두 개의 수학 단어 문제를 성취했을 때, 두 번째 문제의 답이 첫 번째 문제에서의 정확한 답변에 의존하도록 하여 진행했습니다.

- **Technical Details**: 연구에서는 8-shot 프롬프트를 사용하여 Gemma2 27B PT 모델로부터 생성된 합성 데이터(synthetic data)를 활용했습니다. GSM8K 훈련 질문에 대한 10가지 솔루션을 생성하고 올바른 최종 답변을 가진 솔루션만 남겼습니다. 또한, 서로 다른 교육 단계에서 모델의 중간 체크포인트를 평가했습니다.

- **Performance Highlights**: 대부분의 LLMs에서 구성적(compositional) 쌍을 해결하는 능력과 각 문제를 독립적으로 해결하는 능력 사이에 상당한 추론 격차(reasoning gap)가 발견되었습니다. 이 격차는 작은, 비용 효율적인 수학 특화 모델에서 더욱 두드러졌으며, 대형 추론 격차는 테스트 세트 유출(test-set leakage) 때문이 아니라 추가적인 맥락에 대한 방해와 두 번째 단계에서의 추론 부족 때문이라는 것을 나타냅니다.



### Leray-Schauder Mappings for Operator Learning (https://arxiv.org/abs/2410.01746)
Comments:
          6 pages, 2 figures, 1 table. Comments are welcome!

- **What's New**: 이번 연구에서는 Banach 공간 간의 연산자(operators)를 학습하기 위한 알고리즘을 제시하고 있습니다. 이 알고리즘은 Leray-Schauder 매핑을 기반으로 하여, 컴팩트 서브스페이스에 대한 유한 차원 근사를 학습합니다. 결과적으로 이 방법이 (비선형일 가능성이 있는) 연산자의 보편적인 근사기(universal approximator)임을 보여주고 있습니다.

- **Technical Details**: 이 연구는 Chebyshev 다항식을 사용하여 함수 기반에서 연산자를 학습하는 접근 방식을 취하고 있으며, Leray-Schauder 매핑을 통해 선택된 요소의 기저(basis)에서 비선형 투영(nonlinear projection)을 수행합니다. 이 과정은 신경망(neural networks)을 통해 모델링되며, 이 결과로 생성된 모델이 여전히 보편적인 근사기를 유지한다는 이론적 보장을 제공합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 벤치마크 데이터셋에서 효율성을 입증하였으며, 최신 상태의 모델(state of the art models)과 유사한 결과를 달성하였다. 이는 제안된 알고리즘의 우수성을 보여줍니다.



### PreND: Enhancing Intrinsic Motivation in Reinforcement Learning through Pre-trained Network Distillation (https://arxiv.org/abs/2410.01745)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 Pre-trained Network Distillation(PreND)이라는 새로운 접근 방식을 소개하여 강화 학습(RL)에서 내재적 동기를 향상시킵니다. 기존의 랜덤 네트워크 증류(RND)의 한계를 극복하기 위해 사전 훈련된 표현 모델을 활용합니다.

- **Technical Details**: PreND는 강화 학습 에이전트의 탐색 능력을 향상시키기 위해 사전 훈련된 네트워크를 사용합니다. 이 방법은 목표 네트워크(target network)와 예측기 네트워크(predictor network)에 사전 훈련된 표현 모델을 통합하여 보다 의미 있고 안정적인 내재적 보상을 생성합니다. 또한, 예측기 네트워크의 학습 속도를 조절하여 초기 과적합을 방지합니다.

- **Performance Highlights**: 실험을 통해 Atari 환경에서 PreND가 RND보다 상당히 뛰어난 성능을 보인다는 것을 입증했습니다. PreND는 보다 강력한 내재적 동기 신호를 제공하여 에이전트의 탐색을 개선하고 샘플 효율성을 높였습니다.



### Evaluating Robustness of Reward Models for Mathematical Reasoning (https://arxiv.org/abs/2410.01729)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 RewardBench의 한계점을 지적하고, 수학적 추론(Reasoning) 작업에서 리워드 모델(Reward Model)의 신뢰성 있는 평가를 위한 새로운 벤치마크인 RewardMATH를 소개합니다.

- **Technical Details**: RewardMATH는 리워드 모델의 견고성(Robustness)을 효과적으로 나타내기 위해 설계되었습니다. 기존 리워드 모델은 선택된 완료(Completion)와 거부된 완료(Rejected Completion) 간의 차이를 충분히 나타내지 못하는 단일 비교에 의존하고 있습니다.

- **Performance Highlights**: RewardMATH에서의 점수는 최적화된 정책(Optimized Policy)의 결과와 강한 상관관계를 보여주며, 기존 벤치마크는 거의 상관관계가 없음을 보여줍니다. 이는 평가의 신뢰성을 높이고, 리워드 모델의 견고성을 잘 나타내는 잠재력을 강조합니다.



### Automated Knowledge Concept Annotation and Question Representation Learning for Knowledge Tracing (https://arxiv.org/abs/2410.01727)
- **What's New**: 이번 연구에서는 지식 추적(knowledge tracing, KT) 방식의 두 가지 주요 한계를 해결하기 위해 자동화된 지식 개념 주석(annotation) 및 질문 표현 학습 프레임워크인 KCQRL을 제안합니다. 이 프레임워크는 기존 KT 모델의 효과를 향상시키는 데 기여합니다.

- **Technical Details**: KCQRL은 대형 언어 모델(large language models, LLMs)을 활용하여 질문 솔루션을 생성하고 각 솔루션 단계에서 지식 개념(KC)의 주석을 자동으로 부여합니다. 또한, 생성된 질문 내용과 솔루션 단계의 의미론적 표현을 학습하기 위해 대조 학습(contrastive learning) 접근법을 도입합니다. 이러한 표현은 기존 KT 모델에 통합될 수 있습니다.

- **Performance Highlights**: KCQRL 프레임워크는 15개의 최신 KT 알고리즘을 사용하여 두 개의 대규모 실제 수학 학습 데이터셋에서 일관된 성능 향상을 달성하였습니다. 이 연구는 KCQRL이 기존 KT 모델의 성능을 크게 향상시킬 수 있음을 입증합니다.



### Meta-TTT: A Meta-learning Minimax Framework For Test-Time Training (https://arxiv.org/abs/2410.01709)
Comments:
          10 pages, 7 tables, 1 figure

- **What's New**: 이 논문에서는 test-time 도메인 적응을 위한 새로운 메타 학습 최소-최대(framework) 프레임워크를 제안합니다. 이 프레임워크는 배치 정규화(BN) 레이어를 통한 훈련을 지원하며, SSL(Self-Supervised Learning) 작업을 주요 작업과 정렬시켜 미니배치 과적합을 해결합니다.

- **Technical Details**: 제안하는 메타 학습 프레임워크는 mixed-BN 접근 방식을 채택하여 현재 테스트 배치 통계치를 소스 도메인 통계와 보간(interpolate)합니다. 또한, 도메인 이동에 대한 모델의 일반화 및 강건성을 개선하기 위해 확률적 도메인 합성(stochastic domain synthesizing) 방법을 도입합니다. 이 과정에서 최소 최대 엔트로피(minimax entropy)를 사용하여 모델의 적응을 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 본 방법은 다양한 도메인 적응 및 일반화 벤치마크에서 최신 기술을 초월하여 사전 훈련된 모델의 강건성을 향상시키는데 기여합니다.



### Performant, Memory Efficient and Scalable Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2410.01706)
- **What's New**: Sable이라는 새로운 알고리즘을 소개하며, Retentive Networks에서의 retention 메커니즘을 다중 에이전트 강화 학습(MARL)로 적응시키고 메모리 효율성과 확장성을 동시에 달성하는 방법을 제시합니다.

- **Technical Details**: Sable은 retention 기반의 시퀀스 모델링 아키텍처를 이용하여 수천 개의 에이전트로 확장 가능하며, 긴 시간적 컨텍스트(temporal context)를 유지할 수 있습니다. 또한 대규모 부분 관찰 환경(partially observable environments)에서 효과적으로 작동합니다.

- **Performance Highlights**: Sable은 45개 과제 중 34개에서 기존의 최첨단 기법(state-of-the-art)보다 현저히 높은 성능을 보이며, 1000명 이상의 에이전트를 처리하면서 메모리 사용량은 선형적으로 증가합니다.



### MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning (https://arxiv.org/abs/2410.01697)
- **What's New**: 제안된 Multi-Objective REpresentation Learning (MOREL) 접근법은 적대적 작용에 대응하기 위해 특징 표현 학습의 강도를 강조하여 모델의 견고성을 높입니다.

- **Technical Details**: MOREL은 클래스 내 유사한 입력에 대해 일관된 특징을 생성하도록 모델을 유도하는 다목적 최적화 프레임워크를 사용합니다. 이 과정에서 코사인 유사도 손실(cosine similarity loss) 및 다중 긍정 대비 손실(multi-positive contrastive loss)를 활용하여 자연적 및 적대적 특징을 정렬하고 밀집 클러스터를 형성합니다.

- **Performance Highlights**: MOREL로 훈련된 모델은 기존 적대적 훈련 방법보다 화이트 박스 및 블랙 박스 적대적 공격에 대해 더 뛰어난 견고성을 보임을 강조하며, 아키텍처 변경이나 테스트 데이터 정화 없이 정확성과 견고성 간의 균형을 효과적으로 조절합니다.



### Uncertainty Quantification with Bayesian Higher Order ReLU KANs (https://arxiv.org/abs/2410.01687)
Comments:
          13 pages, 7 Figures

- **What's New**: 본 연구에서는 Kolmogorov-Arnold Networks (KANs)에서 불확실성 정량화 (uncertainty quantification)의 첫 번째 방법을 제안합니다. 특히 Higher Order ReLUKANs를 통해 Bayesian 방법의 계산 요구사항을 개선하고자 하였습니다. 제안하는 방법은 일반적인 성질을 가지고 있으며, epistemic과 aleatoric 불확실성에 접근할 수 있습니다.

- **Technical Details**: 제안된 방법은 여러 가지 기준 함수 (basis functions)에 일반화될 수 있으며, 샘플러 (closure tests)를 통해 검증되었습니다. 이 방법은 Stochastic Partial Differential Equations (PDEs)에도 적용되며, 확률적 항의 포함으로 인해 도입된 함수 의존성(functinal dependencies)을 올바르게 식별할 수 있습니다.

- **Performance Highlights**: 간단한 1차원 함수와 Stochastic PDEs에 대한 응용을 통해 본 연구의 방법이 더 나은 계산 효율성과 표현력을 가지게 되는 것을 입증하였습니다. 코드 또한 GitHub에서 확인할 수 있습니다.



### Positional Attention: Out-of-Distribution Generalization and Expressivity for Neural Algorithmic Reasoning (https://arxiv.org/abs/2410.01686)
Comments:
          37 pages, 22 figures

- **What's New**: 최근 신경망(neural network)이 알고리즘 알고리즘 작업을 해결하는 능력에 대한 관심이 증가하고 있습니다. 본 논문에서는 훈련 배포와 다른 값 범위(value range)에서 테스트 배포를 다루며, positional attention을 사용하여 OOD(Out-Of-Distribution) 성능을 향상시키는 방안을 제안합니다.

- **Technical Details**: 본 논문은 Transformer 모델에 positional attention을 도입하여 고정된 위치 인코딩을 통해 주목(attention) 가중치를 결정하는 방법을 설명합니다. 이 접근 방식은 실험적 OOD 성능을 개선하면서도 모델의 표현력(expressivity)을 유지합니다. 논문에서는 PCOC 모델을 통해 기존의 MPC(MapReduce) 모델을 시뮬레이션 할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 연구의 실험 결과, positional attention을 사용한 Transformer의 성능이 기존의 end-to-end 훈련 방식보다 OOD 성능이 현저히 향상되었음을 입증하였습니다. PCOC 모델을 통해 다양한 병렬 알고리즘을 효과적으로 수행할 수 있는 가능성을 제시합니다.



### PHI-S: Distribution Balancing for Label-Free Multi-Teacher Distillation (https://arxiv.org/abs/2410.01680)
- **What's New**: 이번 연구는 라벨 없이 여러 heterogeneous visual foundation model을 융합하는 agglomerative model의 발전에 중점을 두고 있으며, 특히 teacher 모델의 activation 통계와 손실 함수가 student 모델의 품질에 미치는 영향을 다룹니다.

- **Technical Details**: 온도 단순화 실험에서, 우리는 다양한 teacher 모델의 activation 분포와 그 분포의 분산을 분석합니다. 이를 통해 Hadamard 행렬을 사용하여 isotropic 표준화를 수행하고, 이를 'PHI Standardization' (PHI-S)라고 이름 붙였으며, 이 방법이 최상의 student 모델을 생성함을 보여주었습니다.

- **Performance Highlights**: PHI Standardization 방법을 통해 생성된 student 모델은 평가 기준에서 가장 우수한 성능을 보였으며, 여러 가지 합성곱 손실 함수들을 비교 분석하여 그 결과를 제공합니다.



### VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignmen (https://arxiv.org/abs/2410.01679)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 복잡한 추론 작업에서의 신용 할당 문제를 해결하기 위해 VinePPO를 제안합니다. 기존의 Proximal Policy Optimization (PPO) 접근 방식의 단점을 강조하고, 대안적으로 몬테카를로 기반의 보상 추정 방법을 활용하여 성능을 개선합니다.

- **Technical Details**: 기존 PPO의 가치 네트워크는 복잡한 추론 작업에서의 예상 누적 보상을 정확하게 예측하는 데 어려움을 겪습니다. 이는 높은 분산의 업데이트와 최적이 아닌 성능으로 이어질 수 있습니다. 반면, VinePPO는 독립적인 몬테 카를로 샘플을 사용하여 중간 상태의 편향 없는 가치 추정치를 계산하며, 대형 가치 네트워크의 필요성을 제거하고 메모리 요구 사항을 줄입니다.

- **Performance Highlights**: VinePPO는 MATH와 GSM8K 데이터셋에서 PPO 및 기타 RL-free 기준보다 일관되게 뛰어난 성능을 나타냈습니다. gradient 업데이트를 최대 9배 줄이고, wall-clock 시간을 최대 3.0배 단축시키며, 더 나은 KL 발산(trade-off)을 달성했습니다. 이 결과는 LLM의 RL 파인튜닝에서 정확한 신용 할당의 중요성을 강조합니다.



### Sparse Covariance Neural Networks (https://arxiv.org/abs/2410.01669)
- **What's New**: 이번 연구에서는 기존의 Covariance Neural Networks (VNNs)의 한계를 극복하기 위해 Sparse coVariance Neural Networks (S-VNNs)라는 새로운 프레임워크를 제안합니다. S-VNNs는 샘플 공분산 행렬에 sparsification 기술을 적용하여 성능 개선과 계산 효율성을 높입니다.

- **Technical Details**: S-VNNs는 진짜 공분산 행렬이 희소할 때 하드와 소프트 임계값 전략을 적용하며, 공분산이 조밀한 경우에는 확률적으로 공분산을 드롭하는 stochastic sparsification 기법을 제안합니다. 이러한 접근은 S-VNNs의 안정성을 향상시키며, 공분산 희소성과 데이터 분포 간의 새로운 관계를 제공합니다.

- **Performance Highlights**: S-VNNs는 다양한 실제 및 합성 데이터셋에서 실험을 통해 VNNs보다 더 나은 성능과 안정성, 계산 효율성을 보여줍니다. 특히, 뇌 데이터와 인간 행동 인식 애플리케이션에서 더 높은 작업 성능과 안정성을 달성하였습니다.



### Conformal Generative Modeling with Improved Sample Efficiency through Sequential Greedy Filtering (https://arxiv.org/abs/2410.01660)
- **What's New**: 본 연구에서는 Generative Models(생성 모델)의 안전성 문제를 해결할 수 있는 Sequential Conformal Prediction for Generative Models(SCOPE-Gen) 방법을 제안합니다. 이 방법은 rigorous(엄격한) 통계적 보장을 기반으로 하여 생성 모델의 출력에 대한 신뢰성을 높입니다.

- **Technical Details**: SCOPE-Gen은 black box generative model에서 i.i.d. 예제를 샘플링하여 초기 예제 집합을 생성한 후, greedy filters(탐욕적 필터)를 통해 이를 반복적으로 다듬어 나갑니다. 최종적인 예측 집합의 admissibility(허가 가능성)는 Markov chain(마코프 체인)으로 분해되어 통제될 수 있습니다. 이를 통해 calibration(보정) 동안 admissibility 평가의 횟수를 크게 줄일 수 있습니다.

- **Performance Highlights**: 자연어 생성 및 분자 그래프 확장(task) 실험을 통해 SCOPE-Gen은 기존 방법들에 비해 훨씬 적은 수의 admissibility 평가로 안전성 비판적 응용 분야에 적합하다는 것을 입증했습니다.



### Extending Contextual Self-Modulation: Meta-Learning Across Modalities, Task Dimensionalities, and Data Regimes (https://arxiv.org/abs/2410.01655)
Comments:
          23 pages, 11 figures, 5 tables

- **What's New**: 이 논문에서는 Contextual Self-Modulation (CSM)의 두 가지 확장을 소개합니다: $i$CSM과 StochasticNCF입니다. $i$CSM은 CSM을 무한 차원 과제로 확장하며, StochasticNCF는 대량 데이터 시나리오에서 CSM을 적용할 수 있도록 합니다.

- **Technical Details**: $i$CSM은 CSM이 사용하는 유한 차원 컨텍스트 벡터와는 달리 무한 차원 함수 공간에 컨텍스트를 임베드합니다. StochasticNCF는 인접 환경의 샘플 집합을 통해 메타-그래디언트 업데이트에 대한 편향되지 않은 근사를 제공하여, CSM과 $i$CSM을 고데이터 시나리오에 적용할 수 있도록 해줍니다. 또한, 고차 Taylor 확장을 통해 더 높은 차수의 근사가 일반화를 반드시 향상시키지 않음을 밝힙니다.

- **Performance Highlights**: CSM은 다른 메타 학습 프레임워크와 통합될 수 있으며, FlashCAVIA라는 계산 효율적인 확장을 통해 다양한 벤치마크에서 이전의 성능을 초과합니다. 이러한 기여는 메타 학습 과제를 해결하기 위한 강력한 프레임워크를 세우고, 분포 외 일반화에 대한 실질적인 통찰을 제공합니다.



### shapiq: Shapley Interactions for Machine Learning (https://arxiv.org/abs/2410.01649)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 Shapley Value (SV)와 Shapley Interactions (SI)를 기반으로 한 새로운 오픈소스 Python 패키지인 shapiq를 소개합니다. 이 패키지는 최첨단 알고리즘을 통합하여 SV와 다양한 순서의 SI를 효율적으로 계산할 수 있게 합니다.

- **Technical Details**: shapiq는 머신러닝 모델의 예측에서 생성되는 기능 간의 상호 작용을 설명하고 시각화하는 도구입니다. 이 패키지는 11개의 머신러닝 적용 사례와 사전 계산된 게임, 실제 값이 포함된 벤치마킹 세트를 포함하여 다양한 분야에서의 계산 성능을 체계적으로 평가할 수 있습니다.

- **Performance Highlights**: shapiq는 vision transformers, language models, XGBoost, LightGBM (TreeSHAP-IQ 포함) 모델의 예측에서 기능 상호 작용을 설명하고 시각화할 수 있는 기능을 제공합니다. 이로 인해 머신러닝 내에서 SV와 SI의 적용을 넓히고 미래의 연구를 촉진할 수 있습니다.



### Stable Offline Value Function Learning with Bisimulation-based Representations (https://arxiv.org/abs/2410.01643)
Comments:
          Under review

- **What's New**: 이 논문에서는 오프라인 강화 학습에서 가치를 안정적으로 학습하기 위한 새로운 방법론인 KROPE(offline Policy Evaluation을 위한 Kernel Representations)를 제안합니다. KROPE 알고리즘은 상태-행동 쌍의 표현 방식을 개선하여 학습 안정성을 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: KROPE는 상태-행동 쌍의 유사성을 즉각적인 보상(reward) 및 목표 정책(target policy)에 따라 도달할 다음 상태-행동 쌍의 유사성에 기반하여 정의하는 커널(kernel)을 사용합니다. 이를 통해 유사한 상태-행동 쌍이 비슷한 표현을 갖도록 하는 것을 목표로 하며, 경량화된 least-squares policy evaluation (LSPE) 알고리즘의 안정성을 증명합니다.

- **Performance Highlights**: KROPE는 기준(baseline) 알고리즘들에 비해 낮은 가치 오류(value error)를 달성하며, 오프라인 가치 함수 학습의 안정성과 정확성을 높이는 데 기여합니다. 또한 이 논문은 KROPE가 기존의 비유사성(non-bisimulation) 기반 알고리즘에 비해 더 높은 안정성을 제공한 것을 실험적으로 검증합니다.



### Moral Alignment for LLM Agents (https://arxiv.org/abs/2410.01639)
- **What's New**: 이 연구는 인간의 도덕적 가치를 명시적으로 인코딩한 보상 함수를 설계하여 LLM 기반 에이전트를 조정하는 새로운 접근 방식을 제시합니다. 이를 통해 학습 에이전트가 더 나은 도덕적 결정을 내릴 수 있도록 합니다.

- **Technical Details**: 본 연구는 강화 학습(Reinforcement Learning)에서의 보상으로 도덕적 보상(intrinsic rewards)을 사용하는 방식을 소개합니다. 이 방법은 반복 죄수의 딜레마(Iterated Prisoner’s Dilemma) 환경에서 에이전트의 행동과 결과를 정량적으로 평가하며 도덕적 전략을 학습하고 자기 중심적인 전략을 '학습 해제'(unlearn)할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 도덕적 보상으로 세부 조정된 LLM 에이전트가 도덕적으로 일치하는 전략을 성공적으로 학습하고, 특정 도덕적 전략이 다른 매트릭스 게임 환경에서도 일반화됨을 보여주었습니다. 또한, 이 접근 방식은 현재의 조정 기술보다 투명하고 비용 효율적인 대안을 제시합니다.



### Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis (https://arxiv.org/abs/2410.01635)
- **What's New**: 최근 그래프 프롬프트가 유망한 연구 방향으로 떠오르며, 기존의 그래프 모델 재훈련 없이 원래의 그래프에 추가적인 토큰이나 서브그래프를 학습할 수 있는 가능성을 보여주고 있습니다. 이 논문은 데이터 연산 관점에서 그래프 프롬프트를 철저히 분석하는 이론적 틀을 제공합니다.

- **Technical Details**: 이 논문에서는 그래프 프롬프트가 그래프 변환 연산자를 근사할 수 있는 능력을 보장하는 정리(thm)를 제시합니다. 또한, 단일 그래프의 데이터 연산 오류에 대한 상한을 도출하고, 여러 그래프의 배치에도 이 논의를 확장하여 실용적인 시나리오에서 그래프 프롬프트의 확장성과 일반화를 이해하는 데 중요한 분석을 수행합니다. 이 연구는 선형 그래프 모델(예: GCN)에서 비선형 모델(예: GAT)로의 이론적 발견을 확장합니다.

- **Performance Highlights**: 광범위한 실험이 이론적 결과를 뒷받침하며, 그래프 프롬프트의 효과와 설계를 위한 가이드라인을 제공합니다. 이를 통해 연구자와 실무자들이 다양한 응용 프로그램에서 그래프 프롬프트를 더 자신 있게 활용할 수 있도록 합니다.



### Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint? (https://arxiv.org/abs/2410.01623)
Comments:
          Code is available at: this https URL

- **What's New**: 이 논문에서는 Fira라는 새로운 메모리 효율적 훈련 프레임워크를 제안합니다. 이는 저순위(low-rank) 제약을 유지하면서도 전체 순위(full-rank) 훈련을 가능하게 하며, 저순위 훈련 방식의 장점을 활용하는 첫 번째 시도입니다.

- **Technical Details**: Fira는 두 가지 주요 구성 요소를 포함합니다: (1) norm 기반 스케일링 전략(norm-based scaling strategy), 이 전략은 저순위 최적화 기법의 스케일링 효과를 활용하여 전체 순위 훈련을 촉진합니다. (2) norm-growth 리미터(norm-growth limiter)로서, 이는 기울기(norm)의 급격한 증가를 제한하여 손실 스파이크(loss spikes)를 방지합니다. 이를 통해 Fira는 저순위의 제약 조건을 유지하면서도 더 나은 성능을 제공합니다.

- **Performance Highlights**: 종합 실험 결과, Fira는 LoRA 및 GaLore보다 뛰어난 성능을 보이며, 전체 순위 훈련과 비슷하거나 더 나은 성능을 달성했습니다. 다양한 매개변수 수(60M, 130M, 350M, 1B, 7B)를 통한 실험을 통해 Fira의 효과를 입증했습니다.



### On Using Certified Training towards Empirical Robustness (https://arxiv.org/abs/2410.01617)
- **What's New**: 이 논문에서는 최근 인증 훈련(certified training) 기법이 단일 단계 공격(single-step attacks)에서 발생하는 치명적인 과적합(catastrophic overfitting, CO)을 방지할 수 있음을 보여주고, 실험적으로 이를 입증하였습니다. 또한, 네트워크 과근사(network over-approximations)에 대한 새로운 정규화기(regularizer)를 제시하여 효율적인 훈련을 가능하게 합니다.

- **Technical Details**: 단일 공격에 대해 조정된 최근 인증 훈련 알고리즘을 통해 CO를 방지하며, 적절한 실험 설정 하에서 다단계 기준선(multi-step baselines)과의 성능 격차를 줄일 수 있음을 확인했습니다. 특히, Exp-IBP라는 새로운 손실(loss) 개념을 사용하여 다단계 훈련(PGD-5)과 동등하거나 그 이상의 성능을 낼 수 있음을 보여주었습니다. 또한, ForwAbs라는 새로운 프록시(proxy)를 사용하여 계산 비용을 줄이면서 CO를 방지할 수 있는 방법도 제안하였습니다.

- **Performance Highlights**: Exp-IBP를 활용한 실험에서 얕은 네트워크 및 긴 훈련 일정에서 상대적인 경험적 강건성(empirical robustness)이 향상되었으며, CIFAR-10 데이터셋에서 큰 교란 반경에 대해 순수 IBP가 강력한 다단계 적대적 훈련 기준선(PGD-10)을 능가하는 성능을 발휘함을 입증했습니다. 본 연구는 현재의 인증 훈련 기법들이 경험적 강건성에 미치는 잠재력과 한계를 모두 드러내며, 향후 연구 방향에 대한 이정표를 제시합니다.



### Automated Red Teaming with GOAT: the Generative Offensive Agent Tester (https://arxiv.org/abs/2410.01606)
- **What's New**: 본 논문에서는 Generative Offensive Agent Tester (GOAT)라는 새로운 자동화된 레드 팀 시스템을 소개합니다. GOAT는 다중 대화(turn) 환경에서 사용자들이 AI 모델과 상호작용하는 방식을 모방하여 다양한 공격 기법을 활용해 대화형 모델의 취약성을 식별합니다.

- **Technical Details**: GOAT는 총 7개의 레드 팀 공격 기법을 구현하며, 일반 목적의 모델에 대한 프롬프트를 통해 사용 가능한 방법, 현재 대상 모델의 응답, 다음 단계 등을 고려하며 추론을 유도합니다. 이러한 방식은 자동화된 레드 팀 방법이 실질적인 해법을 제시할 수 있도록 돕습니다.

- **Performance Highlights**: GOAT는 JailbreakBench 데이터 세트에서 Llama 3.1에 대해 97%, GPT-4에 대해 88%의 ASR@10을 기록하여 기존의 다중 대화 쿼리 방법보다 높은 성공률을 보여줍니다. 또한, 평균 5개의 대화 흐름 내에서 공격을 성사시킵니다.



### ENTP: Encoder-only Next Token Prediction (https://arxiv.org/abs/2410.01600)
- **What's New**: 이 연구에서는 주로 디코더 전용 Transformers에 의존하던 다음 토큰 예측 모델의 기존 개념을 도전합니다. 전통적으로, 인과적 주의(causal attention)가 미래 토큰을 마스킹하는 데 필수적이라는 믿음이 있었으나, 본 연구는 이러한 디자인 선택이 필수가 아니라 효율성에 관련된 것임을 논의합니다. 우리는 Encoder-only Next Token Prediction (ENTP) 방식을 도입하여 ENTP와 디코더 전용 Transformers 간의 표현력(expressive power)과 복잡성(complexity) 차이를 탐구합니다.

- **Technical Details**: ENTP는 디코더 전용 Transformers가 가질 수 없는 특정 함수를 표현할 수 있는 능력을 갖추고 있으며, 디코더와 인코더 간의 기본적인 시간 및 공간 복잡성을 비교합니다. 이 연구는 Triplet-Counting 과제를 도입하고 ENTP가 이 과제를 쉽게 수행할 수 있는 반면, 기존의 디코더 전용 Transformer는 이를 수행할 수 없다는 것을 이론적으로 및 실험적으로 보여줍니다. 또한, ENTP 방식이 다양한 현실적인 작업에서 우수한 성능을 발휘함을 입증합니다.

- **Performance Highlights**: 실험 결과, ENTP는 길이 일반화(length generalization)와 인맥학습(in-context learning)과 같은 다양한 실제 작업에서 뛰어난 성능을 보였습니다. 특히, ENTP는 비선형 함수와 2계층 신경망 같은 다양한 간단한 함수 수행에서 효과적으로 작동하며, 대규모 텍스트 데이터셋에서의 언어 모델링 과제에서도 디코더와 인코더 간의 성능을 비교하여 그 장점을 입증하였습니다.



### DynFrs: An Efficient Framework for Machine Unlearning in Random Fores (https://arxiv.org/abs/2410.01588)
- **What's New**: DynFrs 프레임워크는 랜덤 포레스트(Random Forests)에서 효율적인 머신 언러닝(machine unlearning)을 가능하게 하며, 예측 정확도를 유지합니다. 이 프레임워크는 Occ(q)와 Lzy 태그 전략을 활용하여 모든 랜덤 포레스트 변형에 적응할 수 있습니다.

- **Technical Details**: DynFrs는 Occ(q) 서브샘플링 기법을 사용하여 훈련 세트의 각 샘플이 제한된 수의 트리에만 영향을 미치도록 보장합니다. Lzy 전략은 필요할 때까지 트리 노드의 재구성을 지연하여 불필요한 수정 작업을 피합니다. 실험 결과, DynFrs는 Extremely Randomized Trees(ERTs)에서 기존 방법들보다 더 빠른 언러닝 성능과 우수한 예측 정확도를 보여줍니다.

- **Performance Highlights**: DynFrs는 전체 모델 재훈련에 비해 4000배에서 1500000배 속도 향상을 이루었으며, 순차적 언러닝과 배치 언러닝에서 기존 방법들보다 수량적 향상을 보여줍니다. 온라인 환경에서 수정 요청에 대해 평균 0.12 ms의 지연을 보이며, 쿼리 요청시 평균 1.3 ms의 지연을 기록합니다.



### Learning-Augmented Robust Algorithmic Recours (https://arxiv.org/abs/2410.01580)
- **What's New**: 본 연구는 머신러닝 모델의 변화에 대한 예측 정보를 활용하여 알고리즘적 회복(algorithmic recourse)의 비용을 줄이는 방법을 모색합니다. 특히, 예측이 정확할 때와 불확실할 때의 회복의 효과를 동시에 고려합니다.

- **Technical Details**: 연구자는 예측 모델의 변동성을 예측하여 회복의 유효성을 높이고, 비용을 최소화하는 새로운 알고리즘을 제안합니다. 본 논문은 비선형 모델에 대해 근접 선형 모델을 활용하여 회복을 계산하는 방식도 다룹니다.

- **Performance Highlights**: 제안된 알고리즘은 ROAR 및 RBR에 비해 유효성이 높은 회복을 제공하며, 고정된 유효성 수준에서 비용이 더 낮은 경향이 있습니다.



### Truncated Kernel Stochastic Gradient Descent on Spheres (https://arxiv.org/abs/2410.01570)
Comments:
          57 pages, 7 figures

- **What's New**: 이 논문에서는 구형 데이터 적합을 위한 새로운 알고리즘인 Truncated Kernel Stochastic Gradient Descent (T-kernel SGD)를 제안합니다. T-kernel SGD는 일정한 학습 속도를 사용하면서도 이론적으로 최적의 수렴 속도를 달성할 수 있습니다.

- **Technical Details**: T-kernel SGD 알고리즘은 구형 데이터에 대한 회귀 문제를 해결하기 위해 설계된 것으로, 주어진 데이터에 대해 실시간으로 추정기를 업데이트하며 최적의 수렴 속도를 보장합니다. T-kernel SGD는 계산 복잡도가 O(n^{1+rac{d}{d-1}rac{	ext{epsilon}}{1}})과 O(n^{rac{d}{d-1}rac{	ext{epsilon}}{1}})이고, 여기서 'n'은 샘플 크기, 'd'는 차원, 그리고 'epsilon'은 임의의 작은 상수를 나타냅니다.

- **Performance Highlights**: T-kernel SGD는 기존의 커널 SGD에 비해 편향(bias)과 분산(variance) 간의 균형을 보다 효과적으로 조절하고, 고차원 공간에서 적합한 폐쇄형 커널 함수를 찾는 어려움을 피할 수 있습니다. 실험 결과는 제안된 알고리즘이 이론적 발견을 뒷받침하며 수렴 속도와 메모리 및 계산 비용 측면에서 유리함을 보여줍니다.



### Bayes' Power for Explaining In-Context Learning Generalizations (https://arxiv.org/abs/2410.01565)
- **What's New**: 이 논문은 신경망 훈련을 최대 우도 추정(maximum likelihood estimation, MLE)과 같은 전통적인 해석에서 벗어나, 실제 데이터 생성 프로세스에 의해 정의된 진정한 사후 분포(true posterior distribution)의 근사로 이해할 필요성을 제안하고 있습니다.

- **Technical Details**: 신경망 훈련이 단일 에포크(single-epoch) 설정에서 이루어지며, 훈련 데이터가 보이지 않는 상황에서 사후 분포 근사로 해석되는 것이 권장됩니다. 논문에서는 이러한 해석이 특히 인컨텍스트 학습(in-context learning, ICL) 세팅에서 훈련된 신경망의 일반화 행동을 정확히 예측할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험을 통해 ICL 설정에서 모델이 강력한 인컨텍스트 학습자로 거듭나며, 훈련 데이터의 지식을 효과적으로 조합하는 방식으로 일반화를 이룬다는 것을 입증하였습니다. 또한 사후 분포의 본질적인 제약과 신경망이 이를 근사하는 데 있어 한계도 논의합니다.



### Lines of Thought in Large Language Models (https://arxiv.org/abs/2410.01545)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 "사고의 선(LoT)"의 통계적 특성을 규명하고, 이들의 경로가 비유클리드적 저차원 기하학에 군집화된다는 점을 보입니다. 저자들은 이러한 복잡한 모델을 더 간단한 형태로 감소시킬 수 있는 방법을 제시합니다.

- **Technical Details**: LLM의 잠재 공간에서 토큰의 궤적을 분석하기 위해, 저자들은 GPT-2 모델을 사용합니다. 모델은 총 24개의 transformer layers로 구성되어 있으며, 각 레이어에서 생성된 벡터 출력을 수집하여 궤적을 형성합니다. 이 궤적들은 평균 선형 변환과 무작위 요소로 잘 근사화된다고 주장합니다.

- **Performance Highlights**: 그들의 연구는 LLM이 수행한 방대한 수의 산술 연산에도 불구하고, 궤적을 설명하는 데 필요한 매개변수가 훨씬 적다는 점을 강조합니다. 특히, 이들은 저차원 비유클리드 기하학적 공간에서 군집화된 궤적 구조를 식별하고, 확률론적 모델을 통해 궤적 앙상블을 설명할 수 있음을 발견했습니다.



### TiVaT: Joint-Axis Attention for Time Series Forecasting with Lead-Lag Dynamics (https://arxiv.org/abs/2410.01531)
Comments:
          15pages, 5 figures

- **What's New**: 이번 연구에서는 기존의 Channel-Dependent (CD) 모델의 한계를 극복하기 위해 TiVaT(Time-Variable Transformer)를 제안합니다. TiVaT는 Joint-Axis (JA) 주의 메커니즘을 통합하여 시계열과 변수 간의 복잡한 상호작용을 동시에 포착할 수 있는 새로운 아키텍처를 제공합니다.

- **Technical Details**: TiVaT는 시계열 예측에서의 변동성과 시간 종속성을 모두 캡처하도록 설계되었습니다. 주요 혁신으로는 JA 주의 메커니즘과 Distance-aware Time-Variable (DTV) 샘플링 기법이 있습니다. DTV 샘플링은 변수와 시간 단계 간의 거리 정보를 학습된 2D 맵으로 캡처하여 중요한 변동-시간 포인트를 동적으로 선택하고, 계산 비용을 줄이며 노이즈를 최소화합니다.

- **Performance Highlights**: TiVaT는 다양한 데이터셋에서 강력한 성능을 꾸준히 발휘하며, 복잡한 패턴을 포착하는 데 뛰어난 능력을 보입니다. 또한, 기존의 최첨단 모델을 초월하거나 경쟁력을 유지하는 성과를 거두어, 복잡한 의존성을 처리하는 새로운 기준으로 자리매김하고 있습니다.



### Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions (https://arxiv.org/abs/2410.01516)
- **What's New**: 이번 연구는 $f$-divergence 손실 함수(f-divergence loss functions)를 활용한 밀도 비율 추정(Density Ratio Estimation, DRE)의 새로운 관점을 제시합니다.

- **Technical Details**: 연구진은 $L_p$ 에러에 대한 상한과 하한을 도출하여, Lipschitz 연속 밀도 비율 추정기(estimator) 클래스에 속하는 어떤 추정기에도 적용 가능한 경계를 제시하였습니다. 이 경계는 데이터 차원과 밀도 비율의 기댓값을 포함한 곱으로 표현됩니다. 하한은 Kullback--Leibler divergence에 의존하는 지수(expontential) 항을 포함하며, 이는 $p > 1$일 때 $L_p$ 에러가 Kullback--Leibler divergence에 따라 상당히 증가한다는 점을 시사합니다.

- **Performance Highlights**: 이론적 발견은 수치 실험(numerical experiments)을 통해 뒷받침되었습니다.



### Discrete Diffusion Schr\"odinger Bridge Matching for Graph Transformation (https://arxiv.org/abs/2410.01500)
- **What's New**: 이번 연구에서는 DDSBM(Discrete Diffusion Schrödinger Bridge Matching)이라는 새로운 프레임워크를 제안하여 고차원 이산 상태 공간에서 SB(Schrödinger Bridge) 문제를 해결합니다. 이 연구는 이산 도메인에 대한 기존의 한계를 극복하고, 그래프 변환 문제에도 적용 가능성을 보여줍니다.

- **Technical Details**: DDSBM은 연속 시간 마르코프 체인(CTMCs)을 활용하며, 이터레이티브 마르코프 피팅(IMF) 방식으로 이산 도메인에서 SB 문제를 솔빙하는 방법론입니다. 이 기반 위에 그래프 도메인으로 확장하여, 그래프 수정 거리(GED)를 비용 함수로 해석하고, 노드 및 엣지의 독립적인 수정을 통해 함수의 유용성을 강조합니다.

- **Performance Highlights**: 화학 분야의 분자 최적화 문제에 DDSBM을 적용한 실험 결과, DDSBM은 분자의 관심 속성(Property-of-Interest)을 최소한의 그래프 변환으로 효과적으로 최적화하는 데 성공하였으며, 기존의 그래프 대 그래프 변환 모델에 비해 여러 분자의 특성을 잘 유지합니다.



### Foldable SuperNets: Scalable Merging of Transformers with Different Initializations and Tasks (https://arxiv.org/abs/2410.01483)
- **What's New**: 이번 연구는 서로 다른 초기화에서 훈련된 대형 transformers를 결합하는 어려운 목표에 도전합니다. 기존 접근 방식들이 실패하는 반면, FS-Merge(Foldable SuperNet Merge) 방법을 제안하며 모델을 최적화하여 원본 모델을 융합합니다.

- **Technical Details**: FS-Merge는 SuperNet을 최적화하여 원본 네트워크의 가중치를 결합합니다. 이 과정에서 feature reconstruction loss를 최소화하고, 모델 폭이 다른 다양한 모델을 결합할 수 있는 능력을 가지고 있습니다. 전통적인 방법과 비교하여 FS-Merge는 정보 손실을 줄이고, 데이터 효율성을 개선합니다.

- **Performance Highlights**: FS-Merge는 각종 설정, 크기, 작업 및 모드에서 기존 방법들과 비교하여 일관되게 더 나은 성능을 보이며, 특히 데이터가 제한된 환경에서 SOTA(State-of-the-Art) 결과를 달성하였습니다.



### Reducing Variance in Meta-Learning via Laplace Approximation for Regression Tasks (https://arxiv.org/abs/2410.01476)
- **What's New**: 이번 연구에서는 gradient 기반 메타 학습(Gradient-Based Meta-Learning)에서의 분산 축소(variance reduction) 문제를 해결하기 위한 새로운 접근법인 Laplace Approximation for Variance-reduced Adaptation (LAVA)를 제안합니다. 이 방법은 각 지원 포인트가 파라미터에 대해 고유한 후분포(posterior distribution)를 유도한다는 아이디어를 기반으로 합니다.

- **Technical Details**: LAVA는 메타 학습 과정에서의 정보 집합을 최적화하기 위해 각 지원 포인트에 대한 후분포를 추정하기 위해 Laplace 근사를 사용합니다. 이를 통해 손실 경량(loss landscape)의 커브에 따른 분산을 표현할 수 있으며, 알고리즘은 Bayesian model-averaging의 형태로 각 데이터 포인트에서 유도된 단일 후분포를 조합하여 최적의 파라미터를 추정합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 동적 시스템의 회귀(regression)와 실제 세계의 실험에서 표준 GBML 방법들과 비교하여 최첨단 성능을 보여주었습니다. 이는 메타 학습에서 분산 축소의 중요성을 강조합니다.



### Selective Aggregation for Low-Rank Adaptation in Federated Learning (https://arxiv.org/abs/2410.01463)
- **What's New**: 본 논문에서는 연합 학습(Federated Learning, FL)에서의 LoRA(저랭크 적응) 기술을 분석하고, 두 개의 저랭크 훈련 가능한 행렬 A와 B를 활용하는 새로운 방법인 FedSA-LoRA를 제안합니다. 이 방법은 A 행렬을 서버와 공유하여 집계하고, B 행렬은 클라이언트 특정 지식을 캡처하는 데 초점을 맞춥니다.

- **Technical Details**: 제안된 FedSA-LoRA는 A 행렬이 일반 지식을 학습하는 반면, B 행렬은 클라이언트별 지식을 포착하도록 설계되었습니다. 추가적으로 FedSA-LoRA를 다른 LoRA 변형인 rsLoRA와 VeRA에 확장하여 FedSA-rsLoRA 및 FedSA-VeRA를 도출하였습니다. 저자들은 이러한 방법들이 FL과 LoRA 간의 통합을 위한 일반적인 패러다임을 수립한다고 주장합니다.

- **Performance Highlights**: 모델 평가 결과는 자연어 이해와 생성 작업에서 FedSA-LoRA의 효과성을 보여주며, A 행렬과 B 행렬의 학습 결과가 다른 클라이언트 간에 어떻게 변화하는지를 실험을 통해 입증하였습니다. 이러한 연구는 향후 연합 학습 환경에서 LoRA의 다양한 변형에 대한 진행 방향을 제공할 것으로 기대됩니다.



### Verbalized Graph Representation Learning: A Fully Interpretable Graph Model Based on Large Language Models Throughout the Entire Process (https://arxiv.org/abs/2410.01457)
Comments:
          under review. corresponding author: Zeyu Zhang

- **What's New**: 이 논문에서는 새로운 방법인 Verbalized Graph Representation Learning (VGRL)을 제안하여 텍스트 속성이 있는 그래프(TAGs)에서의 표현 학습을 개선하는 동시에 해석 가능성을 완전하게 보장합니다. 기존의 그래프 신경망(GNN) 방식의 제한 요소를 극복하려는 노력이 돋보입니다.

- **Technical Details**: VGRL은 모델의 입력, 훈련 과정 및 의사결정 과정 전반에 걸쳐 완전한 해석 가능성을 달성하는 것으로, 각 단계에서 텍스트 설명을 생성하여 사용자들이 모델의 동작을 이해할 수 있도록 합니다. 이를 통해 LLM(대형 언어 모델)의 비용을 줄이면서도 성능을 최적화하기 위해 프롬프트 기반 최적화 기법을 활용합니다.

- **Performance Highlights**: VGRL 방법은 다양한 실제 데이터셋을 통해 검증되었으며, 해석 가능성을 높이면서도 기존의 GNN 모델들에 비해 효율적인 성능을 보여주었습니다. 이 접근법은 새로운 데이터셋이나 문제에 대해 프롬프트를 조정하는 것만으로 쉽게 적응 가능하여 다양한 작업에 유연성을 제공합니다.



### Ensembles provably learn equivariance through data augmentation (https://arxiv.org/abs/2410.01452)
- **What's New**: 이번 논문에서는 무한 너비의 신경망(enesemble of neural networks)에서의 그룹 동등성(group equivariance) 개념이 신경 탄젠트 커널(neural tangent kernel) 한계와 무관하게 나타난다는 사실을 증명하였습니다. 더불어, 확률적(stochastic) 환경과 다양한 신경망 구조에도 적용 가능함을 보여줍니다.

- **Technical Details**: 우리는 신경망 아키텍처와 그룹의 작용 간의 관계에 대한 간단한 충분 조건을 제시하였으며, 이러한 조건을 만족할 경우 앙상블 평균이 훈련의 모든 점에서 자동적으로 동등(EQ)하게 됩니다. 우리의 증명은 비점근적(non-asymptotic)이며, 모든 유한 그룹(compact group)에 대해 적용되고, 확률적 경량 하강(stochastic gradient descent)과 무작위 데이터 증강(random augmentation)에도 유효합니다.

- **Performance Highlights**: 또한, 간단한 수치 실험을 통해 우리의 이론을 검증하였습니다. 이를 통해 데이터 증강 방법이 반드시 동등한 모델을 보장하지 않으며, 동등성을 본질적으로 존중하는 네트워크 구조를 채택하는 것이 성능 향상에 기여할 수 있음을 강조합니다.



### Information-Theoretical Principled Trade-off between Jailbreakability and Stealthiness on Vision Language Models (https://arxiv.org/abs/2410.01438)
- **What's New**: 최근 인공지능의 발전과 함께 Vision-Language Models (VLMs)의 성능이 크게 향상되었습니다. 그러나 이러한 모델들은 jailbreak 공격에 취약하여 안전성과 신뢰성을 위협받고 있습니다. 본 논문은 VLMs의 jailbreak 가능성과 은밀성과의 균형을 탐색하고, 비은밀적인 jailbreak 공격을 탐지하는 새로운 알고리즘을 제시하여 모델의 강건성을 향상시킵니다.

- **Technical Details**: 비은밀한 jailbreak 공격을 탐지하는 알고리즘을 제안하며, 이는 Fano의 불확실성을 활용하여 공격 성공률과 은밀성 점수 간의 관계를 설명합니다. 새로운 스텔스 감지 알고리즘은 diffusion 모델을 사용하여 AI 생성 콘텐츠(AIGC)의 탐지 문제를 강조하며, 이를 통해 보다 강력한 방어 메커니즘을 구축할 수 있습니다.

- **Performance Highlights**: 제안된 방법을 통해 최신 VLM 모델들에 대한 공격 성공률과 은밀성의 trade-off를 정보 이론적 관점에서 분석하였으며, 이를 통해 AI 시스템이 복잡한 공격으로부터 더욱 안전하게 보호될 수 있도록 기여하고자 합니다.



### Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models (https://arxiv.org/abs/2410.01434)
Comments:
          24 pages, 17 figures

- **What's New**: 이번 연구에서는 신경망, 특히 언어 모델의 모듈형 구조(modularity)를 조사하여 서로 기능적으로 유사한 서브네트워크(subnetwork)가 어떻게 연결되고 재사용될 수 있는지를 분석했습니다.

- **Technical Details**: 연구에서는 transformer 기반의 언어 모델에서 고도로 조합 가능한 하위 작업(subtask)을 위한 회로(circuit)를 식별하고 비교합니다. 특히, 확률론적 문맥 자유 문법(probabilistic context-free grammar)을 바탕으로 10개의 모듈형 문자열 편집 작업(string-edit operations)에 책임이 있는 회로를 분석했습니다.

- **Performance Highlights**: 비슷한 기능을 가진 회로는 눈에 띄는 노드 오버랩(node overlap)과 과제 간 충실도(cross-task faithfulness)를 보여주었으며, 식별된 회로는 서브네트워크 집합 연산(subnetwork set operations)을 통해 재사용되고 결합되어 모델의 더 복잡한 기능 능력을 제시할 수 있음을 입증했습니다.



### Adaptive teachers for amortized samplers (https://arxiv.org/abs/2410.01432)
Comments:
          26 pages, 12 figures

- **What's New**: 본 논문에서는 Amortized Inference (아몰타이즈드 추론)의 효율성을 높이기 위해 Adaptive Training Distribution (적응형 훈련 분포)인 Teacher(티쳐) 모델을 도입하여 주 모델인 Sampler(샘플러)의 학습을 안내하는 방법을 제시합니다.

- **Technical Details**: Teacher 모델은 Student(학생) 모델의 높은 오류 영역을 샘플링하도록 훈련되며, 이 과정에서 기존의 높은 손실(high-loss) 영역에 우선적으로 집중합니다. 이 방식은 RL (Reinforcement Learning)에서 Off-policy 학습을 활용하여 다양한 높은 보상을 나타내는 후보군을 탐색하는 데 유리합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 접근 방식의 효과를 검증하였으며, 탐색이 도전적인 synthetic environment (합성 환경), 두 개의 diffusion-based sampling tasks (확산 기반 샘플링 과제), 그리고 네 개의 biochemical discovery tasks (생화학적 발견 과제)에서 샘플 효율성(sample efficiency)과 모드 커버리지(mode coverage) 개선을 성공적으로 입증하였습니다.



### Scalable Reinforcement Learning-based Neural Architecture Search (https://arxiv.org/abs/2410.01431)
Comments:
          33 Pages, 19 Figures

- **What's New**: 이번 연구에서는 Neural Architecture Search (NAS) 문제를 해결하기 위해 Reinforcement Learning (RL) 기반의 새로운 솔루션의 유효성을 평가합니다. RL 에이전트가 단일 최적 아키텍처를 반환하는 대신, 좋은 아키텍처를 찾는 학습을 수행합니다.

- **Technical Details**: 이 연구에서는 NAS-Bench-101과 NAS-Bench-301 환경을 고려하고, 지역 검색(local search) 및 랜덤 검색(random search)과 같은 다양한 기존 강력한 기준선(baseline) 알고리즘과 비교합니다. RL 에이전트는 검색 공간의 크기에 따른 확장성이 뛰어나지만 하이퍼파라미터 변경에 대해 제한된 견고함을 보입니다.

- **Performance Highlights**: RL 기반의 NAS 방법론은 두 개의 검증된 벤치마크에서 효과성을 조사합니다. 이로 인해 자동화된 네트워크 아키텍처 탐색의 방향을 제시하며, 계산 자원 소모를 줄이기 위한 첫 단계로 나아갑니다.



### Fair4Free: Generating High-fidelity Fair Synthetic Samples using Data Free Distillation (https://arxiv.org/abs/2410.01423)
- **What's New**: Fair4Free는 개인적이거나 접근할 수 없는 데이터 상황에서도 공정한 합성 데이터를 생성할 수 있는 새로운 생성 모델입니다. 이 모델은 데이터 없이 지식 증류(data-free distillation)를 통해 작동합니다.

- **Technical Details**: Fair4Free는 먼저 교사 모델을 훈련시켜 공정한 표현을 만든 뒤, 이를 학습 데이터 없이 학생 모델로 지식 증류합니다. 이 과정에서 우리는 Variational Autoencoder (VAE)를 사용하여 공정한 표현을 학습하고 이를 기반으로 고충실도 합성 샘플을 재구성합니다. 이 방법은 노이즈를 입력으로 활용합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, Fair4Free의 합성 샘플은 공정성(fairness)에서 5%, 유용성(utility)에서 8%, 합성 품질(synthetic quality)에서 12% 향상된 성능을 보여주는 등 최첨단 모델보다 뛰어난 성능을 발휘합니다.



### On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding (https://arxiv.org/abs/2410.01405)
- **What's New**: 이번 연구에서는 Looped Transformers의 함수 근사 속성을 탐구하고, 그들의 근사 속도를 정의하여 컨티뉴이티의 개념을 도입합니다. 이 연구는 순환 아키텍처에 특유한 한계를 드러내며, 각 루프에 대한 스케일링 파라미터를 도입해야 한다는 점을 강조합니다.

- **Technical Details**: Looped Transformers는 고정 크기의 Transformer 레이어로 구성되어 있으며, 출력이 입력으로 피드백됩니다. 논문에서는 연속 시퀀스-투-시퀀스 함수에 대한 근사율을 도출하기 위해 시퀀스 연속성, 컨텍스트 연속성, 토큰 연속성의 개념을 정의합니다. 이러한 분석은 각 루프에서의 스케일링 파라미터 통합을 요구합니다.

- **Performance Highlights**: 실험 결과, 루프 수가 증가함에 따라 성능이 향상되는 것으로 나타났으며, 타임스텝 인코딩 아키텍처를 통해 추가적인 성과를 달성할 수 있음을 보여주었습니다.



### Gaussian kernel expansion with basis functions uniformly bounded in $\mathcal{L}_{\infty}$ (https://arxiv.org/abs/2410.01394)
- **What's New**: 본 논문은 Gaussian kernel에 대한 모든 가능한 kernel expansion을 조사하며, 특히 공간 ℝ²에서 weights가 ℓₚ에 있는 Gaussian kernel expansion의 구성을 제시합니다.

- **Technical Details**: 연구의 주요 초점은 머신 러닝에서 자주 사용되는 Gaussian kernel의 확장을 다루고, 이 확장이 ℓₚ (
ell_p)에서 최적의 형태로 이루어질 수 있음을 증명합니다. 특히, 해당 확장은 p가 1보다 클 때 가능하며, p=1은 도달할 수 없다는 결과를 도출합니다.

- **Performance Highlights**: 이 연구는 Gaussian kernel로서의 효율성을 검증하며, ℝ²공간에서 주어진 특성에 대한 Mercer 확장이 존재하지 않음을 명확히 합니다. 이러한 결과는 kernel 머신의 일반화 성능 및 수렴 속도에 대한 중요한 통찰을 제공합니다.



### Causal Inference Tools for a Better Evaluation of Machine Learning (https://arxiv.org/abs/2410.01392)
- **What's New**: 이 논문에서는 머신 러닝 시스템을 분석하고 개선하기 위해 계량 경제학(econometrics)의 엄격한 통계 기법을 적용하는 포괄적인 프레임워크를 제시합니다.

- **Technical Details**: 주요 통계 방법으로는 Ordinary Least Squares (OLS) 회귀, 분산 분석(Analysis of Variance, ANOVA), 로지스틱 회귀(logistic regression)가 포함됩니다. 각 방법의 이론적 기초와 머신 러닝 평가에서의 실제 응용을 설명합니다.

- **Performance Highlights**: 논문은 모델의 동작, 성능, 공정성(fairness)에 대한 깊은 통찰력을 제공할 수 있는 방법론을 세세히 설명하며, 전통적인 평가 지표로는 드러나지 않는 섬세한 패턴과 상호작용을 보여줄 수 있는 사례를 제시합니다.



### FLAME: Adaptive and Reactive Concept Drift Mitigation for Federated Learning Deployments (https://arxiv.org/abs/2410.01386)
Comments:
          Accepted for Publication at EMERGE Workshop - EWSN 2024

- **What's New**: 이번 논문은 개념 변화(concept drift)를 탐지하고 완화할 수 있는 새로운 솔루션인 Federated Learning with Adaptive Monitoring and Elimination (FLAME)을 제시합니다. FLAME은 Federated Learning (FL) 환경 내에서 IoT 기기의 동적인 변화에 대응하는 데 중점을 둡니다.

- **Technical Details**: FLAME은 FL 아키텍처를 활용하여 데이터의 동적 변화가 모델의 성능에 미치는 영향을 최소화합니다. 이 시스템은 클라우드(cloud), 엣지(edge), 마이크로컨트롤러(microcontroller)로 구성된 3-tier 아키텍처에서 작동하며, 데이터의 보안과 개인 정보 보호를 보장합니다.

- **Performance Highlights**: FLAME은 기존의 경량화된 완화 방법보다 우수한 성능을 보여주며, 대규모 IoT 배치에서 자원 활용도를 줄이고 높은 F1 점수를 유지합니다. 이는 실제 애플리케이션에 유망한 접근 방식을 제공합니다.



### Towards Dynamic Graph Neural Networks with Provably High-Order Expressive Power (https://arxiv.org/abs/2410.01367)
- **What's New**: 이 논문은 기존의 DyGNN이 진화하는 그래프의 중요한 패턴을 포착할 수 있는 표현력이 부족하다는 문제를 해결하기 위해 새로운 접근 방식을 제시합니다.

- **Technical Details**: k차원 Dynamic WL 테스트 (k-DWL)를 도입하여 DyGNN의 표현력을 정량화합니다. 기존의 DyGNN들이 1-DWL 테스트로 표현력이 제한된다는 것을 보여주며, 중심 노드 쌍의 표현을 업데이트하기 위해 이웃 노드 쌍과의 상호작용 이력을 집계하는 HopeDGN을 제안합니다.

- **Performance Highlights**: HopeDGN은 3.12%의 성능 향상을 달성하였으며, 2-DWL 테스트와 동등한 표현력을 가지는 것으로 이론적으로 입증되었습니다.



### FlashMask: Efficient and Rich Mask Extension of FlashAttention (https://arxiv.org/abs/2410.01359)
- **What's New**: 이 논문에서는 FlashAttention의 확장판인 FlashMask를 제안합니다. FlashMask는 주의 마스크의 열-기반 희소 표현(column-wise sparse representation)을 도입하여 다양한 마스크 유형을 효율적으로 표현합니다. 이를 통해 $O(N)$의 선형 메모리 복잡성을 달성하고, 계산 효율성을 크게 향상시킵니다.

- **Technical Details**: FlashMask는 주의 마스크를 효율적으로 다루기 위해 열-기반 희소 표현을 사용합니다. 이 접근 방식은 최적화된 커널 구현(kernel implementations)을 통해 불필요한 계산을 제거하고, 계산 정확성을 유지하면서 주의 메커니즘의 гибкость을 보장합니다. FlashMask는 LLM(Large Language Models) 훈련을 위한 다양한 마스크 유형을 지원하며, 128K 토큰까지 처리할 수 있습니다.

- **Performance Highlights**: FlashMask는 기존 FlashAttention의 밀집 방법(dense method)에 비해 1.65배에서 3.22배의 속도 향상을 달성하였습니다. 또한 최신 대안인 FlexAttention과 비교할 때 12.1%에서 60.7% 더 나은 커널 TFLOPs/s를 기록하였으며, A100 GPU에서 이론적으로 가능한 최대 FLOPs/s의 37.8%에서 62.3%까지 도달했습니다.



### PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems (https://arxiv.org/abs/2410.01337)
- **What's New**: 본 논문에서는 Physics-encoded Message Passing Graph Network (PhyMPGN)라는 새로운 그래프 학습 접근 방식을 제안하여, 불규칙한 메시 간격을 갖는 데이터에 기반하여 시공간 PDE 시스템을 모델링합니다. 작은 훈련 데이터셋으로도 작동할 수 있는 모델입니다.

- **Technical Details**: PhyMPGN은 메시지 전파 메커니즘을 활용하여 불규칙한 낮은 해상도 메시에서 시공간 동역학을 모델링합니다. 이 모델은 PDE 시스템의 시간적 진행을 근사하기 위해 2차 수치 적분기를 적용하며, 대칭적인 물리학적 현상을 고려하여 학습 가능한 Laplace block을 설계했습니다. 또한 경계 조건 패딩 전략을 통해 모델 정확성과 수렴성을 개선합니다.

- **Performance Highlights**: PhyMPGN은 불규칙한 저해상도 메시에서 다양한 유형의 시공간 동역학을 정확히 예측할 수 있으며, 다른 기준 모델에 비해 50% 이상의 성능 향상을 달성하여 최신 성과를 지속적으로 초과합니다.



### Efficient Learning of POMDPs with Known Observation Model in Average-Reward Setting (https://arxiv.org/abs/2410.01331)
- **What's New**: 본 논문에서는 평균 보상 무한 지평선 POMDP(Partially Observable Markov Decision Processes) 설정을 다루며, 관찰 모델은 알고 있지만 전이 모델은 모르는 상황에서 POMDP 매개변수를 샘플을 통해 학습할 수 있는 새로운 OAS(Observation-Aware Spectral) 추정 기법을 제안합니다. 또한 OAS-UCRL 알고리즘을 통해 탐색과 이용의 트레이드오프를 암묵적으로 균형 잡습니다.

- **Technical Details**: OAS 추정 기법은 신뢰 기반 정책(belief-based policy)을 사용하여 샘플을 수집하고, 이로부터 POMDP 매개변수를 추정합니다. 알고리즘은 점차 길어지는 에피소드를 통해 실행되며, 추정된 POMDP의 최적 신뢰 기반 정책이 환경과 상호작용하여 다음 에피소드에서 사용할 샘플을 수집합니다. OAS-UCRL 알고리즘은 최적의 정책을 재계산하는 최적화 오라클(optimization oracle)을 사용합니다.

- **Performance Highlights**: OAS-UCRL 알고리즘의 후회 보장은 $	ilde{O}(	ext{sqrt}(T))$로 나타났으며, 이는 상태(state), 행동(action), 관찰(observation) 공간의 차원에 대해 효율적으로 확장됩니다. 실험적으로 OAS 추정 절차와 OAS-UCRL 알고리즘의 효율성을 다른 기법들과 비교하여 검증하였습니다.



### Fair Class-Incremental Learning using Sample Weighting (https://arxiv.org/abs/2410.01324)
- **What's New**: 이 논문은 Trustworthy AI를 위한 공정한 클래스 증분 학습을 다루고 있으며, 기존의 정확도 중심 접근 방식에서 벗어나 민감한 그룹을 포함한 모델 공정성을 연구합니다.

- **Technical Details**: 학습 샘플의 가중치를 조정하여 현재 작업 데이터의 평균 기울기 벡터 방향을 변경함으로써 공정성을 달성합니다. 최적화 문제를 설정하고, Linear Programming으로 문제를 해결하며, Fairness-aware Sample Weighting (FSW) 알고리즘을 제안합니다.

- **Performance Highlights**: FSW는 실제 데이터셋에서 기존의 선진 방법들보다 더 나은 정확도와 공정성 균형을 달성하는 결과를 보여주었습니다.



### Forte : Finding Outliers with Representation Typicality Estimation (https://arxiv.org/abs/2410.01322)
- **What's New**: 이 논문은 기존의 Generative 모델들이 생성한 데이터의 OOD(Out-Of-Distribution) 탐지 문제를 다루고 있으며, 새로운 접근 방식인 Forte를 소개합니다. Forte는 self-supervised learning 기법을 활용하여 OOD 탐지의 정확성을 높이며, class labels가 필요하지 않고, OOD 데이터와의 노출 없이도 작동합니다.

- **Technical Details**: Forte는 CLIP, ViT-MSN 및 DINOv2와 같은 다양한 표현 학습 기법을 비모수 밀도 추정 모델(OCSVM, KDE, GMM)과 결합하여 atypical samples를 탐지합니다. 이 방법은 전체 데이터의 정보적 summary statistics를 효과적으로 반영하는 데이터 포인트 단위의 통계들을 제공합니다.

- **Performance Highlights**: Forte는 다양한 OOD 탐지 작업과 합성 이미지 탐지에서 기존의 최첨단 방법들보다 우수한 성능을 보여주며, 포토리얼리스틱 이미지 생성을 포함한 여러 벤치마크에서 최상의 성능을 달성했습니다.



### Sampling from Energy-based Policies using Diffusion (https://arxiv.org/abs/2410.01312)
- **What's New**: 본 논문에서는 에너지 기반 정책을 샘플링하기 위한 확산(diffusion) 기반 접근 방식을 도입하고, 이를 활용한 Actor-Critic 방식의 새로운 알고리즘인 Diffusion Q-Sampling (DQS)을 제안합니다. 이는 여러 환경에서 안정적인 학습을 가능하게 하여 다중 모드 행동을 효과적으로 포착하는 데 기여합니다.

- **Technical Details**: 이 연구는 Boltzmann 배포(Boltzmann distribution)에서 샘플링하는 방법을 제안하며, 여기서 부정 Q-함수(negative Q-function)가 에너지 함수로 정의됩니다. 우리는 확산 모델을 활용하여 복잡한 분포에서 고품질 샘플을 생성하고, 특히 연속 제어 작업에서 탐색과 표현력(Expressiveness)을 강화함으로써 학습 효율성을 높입니다.

- **Performance Highlights**: 제안하는 Diffusion Q-Sampling 방법은 복잡한 행동을 학습하고 샘플 효율성을 개선하는 데에 있어 기존 방법들의 핵심적인 제한 사항을 해결합니다. 이 방법은 미로 탐색 과제에서 효과적인 결과를 보여주고, 연속 제어 작업에서도 우수한 탐색 기능으로 인해 성능이 향상됩니다.



### Rethinking the Expressiveness of GNNs: A Computational Model Perspectiv (https://arxiv.org/abs/2410.01308)
- **What's New**: 이번 논문에서는 그래프 머신 러닝에서 중요한 역할을 하는 그래프 신경망(Graph Neural Networks, GNNs)의 표현력에 대한 기존의 분석 방식의 문제점을 지적하며 새로운 접근법인 Resource-Limited CONGEST (RL-CONGEST) 모델을 제안합니다.

- **Technical Details**: RL-CONGEST 모델은 선택적 전처리(preprocessing) 및 후처리(postprocessing)를 포함하여 GNN의 표현력을 분석하는 프레임워크를 형성합니다. 또한, WL 테스트에서 해시 함수의 계산적 난이도(computational hardness) 및 가상 노드(virtual nodes)가 네트워크 용량을 감소시키는 역할에 대한 통찰도 제공합니다.

- **Performance Highlights**: 우리는 고차 GNNs가 1차 모델 검사(model-checking) 문제와 연관되어 있다는 점을 강조하며, 이는 GNN의 표현력에 대한 새로운 관점을 제공합니다.



### Speculative Coreset Selection for Task-Specific Fine-tuning (https://arxiv.org/abs/2410.01296)
Comments:
          20 pages, 4 figures, 14 tables

- **What's New**: 본 논문에서는 STAFF라는 새로운 공동 집합(coreset) 선택 방법을 제안합니다. 이는 특정 작업에 대한 LLM(Large Language Model) 파인튜닝(task-specific fine-tuning)을 통해 데이터 효율성과 선택 오버헤드를 크게 개선하는 것을 목표로 합니다.

- **Technical Details**: STAFF는 대상 LLM과 동일한 계열의 작은 모델을 활용하여 데이터 점수를 효율적으로 추정하고, 그 점수를 바탕으로 대상 LLM에서 검증하여 중요한 지역에 보다 많은 선택 예산을 할당하는 방식으로 구성됩니다. STAFF는 두 단계로 나뉘며, 첫 번째 단계인 탐색적 점수 계산(speculative score calculation)에서는 작은 모델을 파인튜닝하여 데이터 샘플의 중요도를 평가합니다. 두 번째 단계는 LLM 검증 및 선택(LLM Verification & Selection) 단계로, 여기서 샘플을 중요도 점수에 따라 나누고 검증합니다.

- **Performance Highlights**: STAFF는 3개의 LLM과 3개의 다운스트림 작업에서 평가되었으며, SOTA(stop-of-the-art) 방법보다 최대 54.3% 향상된 성능을 보여주었고, 선택 오버헤드를 최대 70.5%까지 줄였습니다. 특히 낮은 가지치기 비율(예: 20%)에서 STAFF가 전체 데이터셋보다 더 나은 파인튜닝 성능을 발휘하는 것으로 나타났습니다.



### Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models (https://arxiv.org/abs/2410.01280)
- **What's New**: 본 논문에서는 Llama $3$ $70$B 모델을 활용해 RL(강화 학습) 문제를 해결하는데 필요한 in-context 학습 메커니즘을 탐구합니다. 특히 템포럴 디퍼런스(Temporal Difference, TD) 오류를 이해하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 본 연구에서는 Sparse Autoencoders(SAEs)를 사용해 Llama의 잔여 스트림(residual stream)을 분석하여 TD 오류와 Q-value가 잘 나타나는 표현을 발견합니다. 이 과정에서 모델은 다음 토큰을 예측하기 위해 훈련되었음에도 불구하고 이러한 표현이 드러납니다.

- **Performance Highlights**: 연구 결과, SAEs를 통해 얻은 TD 오류와 Q-value의 표현은 Llama의 행동과 표현에 예측 가능한 방식으로 영향을 미친다는 것이 입증되었습니다. 이는 in-context 학습을 보다 정교하게 이해하는 기반이 됩니다.



### Deep Unlearn: Benchmarking Machine Unlearning (https://arxiv.org/abs/2410.01276)
- **What's New**: 본 논문에서는 여러 벤치마크 데이터셋과 모델을 기준으로 18가지 최신 머신 언러닝(Machine Unlearning) 방법을 조사하였다. 이는 DNN(Deep Neural Network)에서의 MU가 성공적으로 적용될 수 있는 방법에 대한 포괄적인 연구가 부족한 상황에서 이루어졌다.

- **Technical Details**: 연구에서는 Masked Small Gradients (MSG)와 Convolution Transpose (CT) 방법이 데이터셋과 초기화에 관계없이 모델의 정확성과 실행 성능에서 일관되게 더 나은 성과를 보인다는 것을 밝혔다. 각 MU 방법은 10개의 초기화 동안 평가되었으며, 100K 모델에 걸쳐 검증되었다.

- **Performance Highlights**: 본 연구 결과, Masked Small Gradients, Convolution Transpose 등은 U-LiRA와 같은 공격에 대해 높은 저항력을 보여주었으며, 평가된 18가지 MU 방법 중에서도 매우 높은 성능을 보였다. 또한, 기존 MU 방법들과 비교할 때 NG+를 포함한 새롭고 더 나은 기준 시스템이 필요하다는 점을 강조하였다.



### HelpSteer2-Preference: Complementing Ratings with Preferences (https://arxiv.org/abs/2410.01257)
Comments:
          26 pages, 3 figures

- **What's New**: 이 논문에서는 Bradley-Terry 스타일과 Regression 스타일의 리워드 모델을 비교하기 위한 고품질 데이터셋 'HelpSteer2'를 제공하고 있으며, 두 접근 방식의 효과를 면밀히 분석한 최초의 연구입니다.

- **Technical Details**: 리워드 모델은 언어 모델이 지침을 따르도록 하는 데 필수적이며, 두 가지 주요 접근 방식인 Bradley-Terry 스타일과 Regression 스타일로 훈련됩니다. 연구진은 두 스타일의 데이터를 적절히 맞추어 비교 검증을 수행하였으며, 인간이 작성한 정당화(Justification)가 포함된 Preference annotations를 사용합니다. 새로운 접근법으로 두 모델의 조합 방법을 제안하고, Llama-3.1-70B-Instruct 모델을 통해 베스트 성능을 기록하였습니다.

- **Performance Highlights**: 이 연구에서 훈련된 리워드 모델은 RewardBench에서 94.1점을 기록하였으며, 이는 140개 이상의 리워드 모델 중 최고의 역량을 지닌 모델입니다. 또한 RLHF(Reinforcement Learning from Human Feedback)에서 지침을 따르도록 모델을 정렬하는 데 효과적임을 입증하였습니다.



### Dual Approximation Policy Optimization (https://arxiv.org/abs/2410.01249)
Comments:
          30 pages, 2 figures

- **What's New**: 이 논문에서는 일반 함수 근사를 정책 미러 강하 방법에 통합한 Dual Approximation Policy Optimization (DAPO) 프레임워크를 제안합니다. DAPO는 정책 투영을 위해 미러 맵에서 유도된 이중 Bregman divergence를 사용하며, 이는 L2-노름($L_2$-norm)으로 함수 근사 오류를 측정하는 전통적인 접근 방식과 대비됩니다.

- **Technical Details**: DAPO는 일반 함수 근사에서 빠른 선형 수렴을 달성할 뿐만 아니라, 여러 잘 알려진 실용적인 방법들이 특정 사례로 포함됩니다. DAPO는 서로 다른 미러 맵을 사용하여 여러 인스턴스를 제시하고, DAPO-L2는 도형 맵으로써 L2-norm을 사용하는 경우와 DAPO-KL의 두 가지 변형에 대해 선형 수렴 속도를 증명합니다.

- **Performance Highlights**: DAPO는 Soft Actor-Critic (SAC)과 Mirror Descent Policy Optimization (MDPO)와 같은 두 가지 최첨단 실용 알고리즘을 포함하며, 성능을 비교하기 위해 여러 표준 MuJoCo 벤치마크 작업에서 평가됩니다. DAPO는 이중성 프레임워크를 통해 이러한 알고리즘에 강력한 수렴 보장을 즉시 제공합니다.



### See Me and Believe Me: Causality and Intersectionality in Testimonial Injustice in Healthcar (https://arxiv.org/abs/2410.01227)
- **What's New**: 이 연구는 의료 환경에서 환자의 증언 불공정성을(testimonial injustice)를 정량화하기 위해 인과 발견(causal discovery) 방법론을 사용하였으며, 핵심적으로 인구 통계적 특성이 이러한 불공정성에 어떻게 기여하는지를 밝혀냈습니다.

- **Technical Details**: 연구는 FCI(Fast Causal Inference) 방법을 사용하여 환자의 의료 기록에서 불공정한 용어의 출현을 분석하고, 이러한 용어와 환자의 인구 통계적 특성(예: 나이, 성별, 인종) 간의 인과적 관계를 수립하는 구조적 인과 모델(Structural Causal Model, SCM)을 구축했습니다.

- **Performance Highlights**: 해당 연구는 사람의 특징이 불공정성 경험에 어떻게 교차적으로 작용하는지를 분석하고, 의료 서비스 향상과 신뢰 구축을 위한 디자인 원칙에 대한 통찰을 제시합니다.



### Induced Covariance for Causal Discovery in Linear Sparse Structures (https://arxiv.org/abs/2410.01221)
- **What's New**: 이 논문은 선형 희소 구조(linearly sparse structures)를 위한 새로운 인과 발견 알고리즘을 소개합니다. 이 알고리즘은 기존의 독립성 테스트나 그래프 적합 절차에 의존하지 않고, 제한된 데이터로도 잘 작동하도록 설계되었습니다.

- **Technical Details**: 제안된 알고리즘은 구조적 인과 모델(Structural Causal Model, SCM)을 기반으로 하며, 인과 그래프 정보를 포함하는 구조적 행렬(structural matrix)을 복구합니다. 이 과정에서는 인수 공분산(induced covariance), 데이터 복구(data recovery) 능력 및 행렬의 대각 구조(diagonal structure)를 활용하여 선형 희소 의존성을 효과적으로 모델링합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안한 알고리즘이 PC, GES, BIC exact search 및 LINGAM 기반 방법들에 비해 평균 35%의 정밀도(precision)와 41%의 재현율(recall) 향상을 보여주었습니다.



### Absolute State-wise Constrained Policy Optimization: High-Probability State-wise Constraints Satisfaction (https://arxiv.org/abs/2410.01212)
Comments:
          submission to Journal of Machine Learning Research

- **What's New**: 본 논문에서는 강화 학습(RL)에서의 안전성 보장을 위한 새로운 방법인 Absolute State-wise Constrained Policy Optimization (ASCPO)를 제안합니다. 기존의 안전 RL 방법들이 상태별 제약 조건을 기대값으로만 강제하거나 비현실적인 가정을 기반으로 하여 경직된 제약 조건을 적용한 반면, ASCPO는 강력한 가정 없이 높은 확률로 상태별 안전을 보장합니다.

- **Technical Details**: ASCPO는 확률적으로 안전한 상태별 제약 조건을 만족하도록 설계된 일반적인 정책 검색 알고리즘입니다. 이 방법은 기대되는 제약 위반과 함께 위반의 분산을 고려하여 특정 사용자 정의 임계값 내에서 위반 확률의 상한을 제어합니다. 주요 아이디어는 상태별 제약 조건이 높은 확률로 충족되도록 보장하는 것입니다.

- **Performance Highlights**: 실험 결과 ASCPO는 로봇 보행 작업에서 상태별 안전 제약을 준수하면서 기존 방법들보다 높은 성과를 내며, 다양한 도전적인 연속 제어 작업에서도 우수한 안전 위반률을 기록했습니다. ASCPO는 실제 응용 분야에서 높은 잠재력을 지니고 있음을 보여주었습니다.



### Debiasing Federated Learning with Correlated Client Participation (https://arxiv.org/abs/2410.01209)
- **What's New**: 본 연구는 연속 참여 인스턴스에서 클라이언트의 최소 분리 제약(minimum separation constraint)을 고려한 federated averaging (FedAvg) 알고리즘의 수렴(convergence)을 분석한 첫 논문이다. 또한, 클라이언트 참여의 비균일성과 상관관계를 모델링하기 위해 Markov chain을 도입했다.

- **Technical Details**: 연구는 클라이언트 참여 패턴을 Markov chain으로 모델링하며, 특정 클라이언트가 재참여하기 전에 최소 R 라운드를 기다려야 하는 조건을 설정한다. 이로 인해, 각 클라이언트의 효과적인 참가 확률이 더욱 균일해지고 FedAvg가 도달하는 솔루션의 편향(bias)이 감소한다.

- **Performance Highlights**: 저자들은 클라이언트 참여 확률을 추정하고 이를 지역 업데이트에 통합하는 debiased FedAvg 알고리즘을 제안했다. 이 알고리즘은 임의의 최소 분리 조건 및 알려지지 않은 클라이언트 가용성 분포에서도 편향 없는 최적 솔루션으로 수렴하는 것을 입증했다.



### Were RNNs All We Needed? (https://arxiv.org/abs/2410.01201)
- **What's New**: 최근 Transformer 아키텍처의 한계 때문에 RNN(순환 신경망)에 대한 관심이 다시 높아지고 있습니다. 기존의 LSTM과 GRU를 발전시켜 새로운 경량 RNN(minLSTM, minGRU)을 제안합니다. 이들 모델은 파라미터 수가 크게 줄어들고 병렬 학습이 가능해졌습니다.

- **Technical Details**: 제안된 minLSTM과 minGRU는 기존 LSTM과 GRU의 입력, 망각, 업데이트 게이트에서 hidden state(은닉 상태) 의존성을 제거하여 BPTT(Through-Time)를 필요로 하지 않습니다. 이로써 훈련 성능이 512의 시퀀스 길이에서 175배 빨라집니다. 또한 이 모델들은 파라미터 수를 대폭 줄였습니다.

- **Performance Highlights**: 이 새로운 경량 RNN 모델들이 최신 sequence 모델들과 비슷한 성능을 보여주며, 과거의 RNN 구조에서 나온 성과를 재현할 수 있음을 입증합니다.



### Stochastic Gradient Descent with Adaptive Data (https://arxiv.org/abs/2410.01195)
- **What's New**: 본 논문에서는 Stochastic Gradient Descent (SGD)를 정책 최적화 문제에 적용할 때 발생하는 도전과제를 다룹니다. 특히, 정책이 환경을 변화시키고 데이터의 비독립적(non-independent)이고 비정상적(non-stationary) 특성을 띈 데이터를 생성함에 따라 생기는 문제를 분석합니다. 이 연구는 SGD의 수렴 속도를 보장하기 위한 기준을 제시합니다.

- **Technical Details**: 논문에서는 adaptively generated data stream을 통해 SGD의 수렴성을 보장하는 간단한 기준을 소개합니다. 저자들은 정책에 의해 유도된 동적 시스템의 mixing time을 고려할 때, adaptive data를 가진 SGD의 수렴 속도가 고전적인 i.i.d. 설정과 유사하다는 것을 보여줍니다. 또한 Lyapunov function 분석을 통해 운영 연구에서 연구된 확률 시스템의 안정성 분석 결과를 SGD의 수렴 속도로 변환할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 저자들은 대기열(queueing) 및 재고 관리(inventory management) 문제에의 응용을 통해 이 결과를 입증하며, actor-critic 정책 기울기 알고리즘의 샘플 복잡성을 연구하는 데에 어떻게 활용될 수 있는지를 보여줍니다.



### Efficient PAC Learning of Halfspaces with Constant Malicious Noise Ra (https://arxiv.org/abs/2410.01186)
- **What's New**: 이 논문에서는 적대적인 잡음(마리셔스 노이즈)이 존재하는 상황에서도 PAC(Probably Approximately Correct) 학습을 통해 halfspaces를 효과적으로 학습할 수 있는 방법을 제안합니다. 특히, 거리 기준(margin parameter)이 충족될 때, reweighted hinge loss를 최소화함으로써 상수 잡음 허용치를 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 적대적인 잡음이 있는 상황에서 반평면(halfspaces)을 학습하는 문제를 다루며, 두 가지 주요 요소를 포함합니다: 1) 손상된 샘플로부터 gradient 저하를 조절하기 위한 효율적인 가중치 찾기 알고리즘, 2) 이러한 가중치가 적용된 hinge loss의 강인성(robustness)에 대한 새로운 분석.

- **Performance Highlights**: 혼합 잡음이 있는 조건 아래에서도 상수 잡음 허용치를 달성할 수 있는 효율적인 알고리즘을 제안하며, 이는 과거 연구에서 달성된 최적의 잡음 허용치에 접근할 수 있는 가능성을 보여줍니다.



### A Deep Learning Approach for Imbalanced Tabular Data in Advertiser Prospecting: A Case of Direct Mail Prospecting (https://arxiv.org/abs/2410.01157)
Comments:
          Third KDD Workshop on End-to-End Customer Journey Optimization

- **What's New**: 이 논문은 직접 우편 광고에서 가능한 고객을 식별하는 데 현대 기계 학습 기법의 적용 가능성을 탐구합니다. 기존 연구에서 직접 우편 광고가 고객 유치에 효과적이라는 사실을 강조하며, 감독학습(supervised learning) 접근 방식을 통해 새로운 고객을 찾아내는 방법을 제시합니다.

- **Technical Details**: 제안하는 프레임워크는 두 가지 구성요소로 이루어져 있습니다: 오토인코더(autoencoder)와 피드 포워드 신경망(feed-forward neural network)입니다. 이들은 비대칭(tabular imbalanced) 데이터셋을 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 우리의 프레임워크는 실제 사례 연구를 통해 우편 광고에 적용하였고, 최신의 랜덤 포레스트(random forest) 방법보다 우수한 성능을 보여주었습니다. 이 프레임워크는 다양한 Fortune 500 기업들에서 강력한 생산 성능을 발휘했습니다.



### Text2PDE: Latent Diffusion Models for Accessible Physics Simulation (https://arxiv.org/abs/2410.01153)
Comments:
          25 pages, 7 figures

- **What's New**: 최신 연구에서는 딥러닝을 활용한 부분 미분 방정식(Partial Differential Equation, PDE) 문제 해결 방법을 발전시키고 있습니다. 이 논문에서는 Latent Diffusion Models를 물리 시뮬레이션에 적용하여 기존의 수치적 방법에 비해 보다 효율적이고 빠른 솔루션을 제공하는 방법들을 소개합니다.

- **Technical Details**: 저자들은 Mesh Autoencoder를 통해 임의로 분할된 PDE 데이터를 압축하며, 이를 통해 다양한 물리에 대해 효율적인 Diffusion 학습을 수행할 수 있습니다. 또한 전체 시공간(solution trajectory)을 한 번에 생성하여 Autoregressive 에러 축적을 줄이는 방법을 제안합니다. 초기 물리량이나 텍스트 프롬프트를 기반으로 한 Conditioning을 통해 텍스트 기반의 PDE 생성(text2PDE)을 구현하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 현재의 신경 PDE 솔버들과 경쟁력 있는 정확성과 효율성을 가지며, 최대 약 30억 개의 매개변수까지 확장할 수 있는 가능성이 있습니다. 이를 통해 더욱 실용적인 물리 시뮬레이터의 개발이 기대됩니다.



### Recovering Manifold Structure Using Ollivier-Ricci Curvatur (https://arxiv.org/abs/2410.01149)
- **What's New**: ORC-ManL이라는 새로운 알고리즘이 소개되었습니다. 이 알고리즘은 Ollivier-Ricci curvature와 추정된 metric distortion을 기반으로 최근접 이웃 그래프에서 불필요한 엣지를 제거하는 방법입니다.

- **Technical Details**: 본 연구의 동기는 manifold learning에서 출발합니다. 데이터가 저차원 manifold에서 발생하는 noisy sample일 때, ambient space를 지나는 shortcut 엣지들은 데이터 manifold에 따라 있는 엣지보다 더 부정적인 Ollivier-Ricci curvature를 가지고 있다는 것을 보였습니다.

- **Performance Highlights**: ORC-ManL은 기존의 다양한 pruning 방법보다 좋은 성능을 발휘하며, 최근접 이웃 그래프를 입력으로 사용하는 여러 downstream geometric data analysis 작업에서 성능을 상당히 향상시킵니다. 특히 manifold learning, persistent homology, 차원 추정 등에서 평가되었으며, 단일 세포 RNA 시퀀싱 데이터의 clustering 및 manifold learning 향상에도 사용될 수 있음을 보여주었습니다.



### ProxiMix: Enhancing Fairness with Proximity Samples in Subgroups (https://arxiv.org/abs/2410.01145)
- **What's New**: 이번 연구에서는 기존의 mixup 방법과 새로운 편향 완화 알고리즘을 결합하여 보다 공정한 데이터 증강을 위한 사전 처리 전략을 제안합니다. 이를 통해 레이블 생성 개선을 목표로 하는 새로운 기법인 ProxiMix를 소개합니다.

- **Technical Details**: ProxiMix는 쌍(pairwise) 관계와 근접(proximity) 관계를 모두 유지하여 편향된 레이블이 발생하는 문제를 해결합니다. 연구는 세 가지 데이터셋과 세 가지 머신러닝 모델을 사용하여 ProxiMix의 효과를 검증하였습니다.

- **Performance Highlights**: ProxiMix는 기존의 pairwise mixup 기법보다 예측의 공정성과 재조정의 공정성 측면에서 높은 성능을 보였습니다. 특히, 원본 데이터셋의 레이블이 상당히 편향된 경우에 더욱 효과적이었습니다.



### Explain Like I'm Five: Using LLMs to Improve PDE Surrogate Models with Tex (https://arxiv.org/abs/2410.01137)
Comments:
          22 pages, 15 figures, 7 tables

- **What's New**: 이 논문에서는 Machine Learning 기술을 통한 Partial Differential Equations (PDEs) 해결 방법을 연구하고 있으며, 사전 훈련된 Large Language Models (LLMs)를 이용한 새로운 멀티모달 접근 방식을 제안합니다. 이 접근 방식은 전통적인 데이터 기반 방식이 아닌 시스템 정보, 즉 경계 조건과 지배 방정식을 통합하여 PDE 학습에 활용하는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 FactFormer 모델을 기반으로 하며, 텍스트 설명을 통해 시스템 정보를 효과적으로 통합합니다. 실험에서는 2D Heat, Burgers, Navier-Stokes, 및 Shallow Water 방정식을 사용하여 모델의 성능을 비교하였으며, 다양한 경계 조건 및 초기 조건을 포함한 데이터 세트를 사용하여 보다 도전적인 벤치마크를 생성하였습니다.

- **Performance Highlights**: 이 접근 방식은 기존 기준 모델에 비해 다음 단계 예측(next-step prediction)과 자기 회귀적 전개 성능(autoregressive rollout performance)에서 유의미한 성과를 보여주었습니다. 또한, 사전 훈련된 LLM은 주어진 시스템 정보에 따라 구조화된 잠재 공간을 제공하였습니다.



### nGPT: Normalized Transformer with Representation Learning on the Hyperspher (https://arxiv.org/abs/2410.01131)
- **What's New**: 새로운 신경망 아키텍처인 정상화된 Transformer(nGPT)를 제안하며, 이는 하이퍼스피어(hypersphere)에서의 표현 학습을 포함합니다. nGPT에서는 임베딩, MLP, 주의(attention) 행렬 및 히든 상태를 구성하는 모든 벡터가 단위 노름(normalized)으로 정규화됩니다. 적절한 훈련(step) 수를 4배에서 20배 줄여 같은 정확도를 성취하는데 걸리는 시간을 단축합니다.

- **Technical Details**: nGPT 구조에서 모든 벡터는 단위 노름 하이퍼스피어에 형성되어, 행렬-벡터 곱셈을 코사인 유사성을 나타내는 내적(dots)으로 간주합니다. 또한, nGPT는 각 계층(layer)에서 두 번째 학습률(eigen learning rate)에 의해 최적화를 제어하여 다단계 최적화(multi-step optimization)를 수행합니다. 이 방식은 훈련 과정의 유사성 추정(accuracy estimation)을 개선합니다.

- **Performance Highlights**: nGPT는 기존 모델보다 훈련 단계에서 4배에서 20배 더 적은 단계를 요구하며, 이는 더 빠른 수렴 속도를 의미합니다.



### Almost Free: Self-concordance in Natural Exponential Families and an Application to Bandits (https://arxiv.org/abs/2410.01112)
Comments:
          Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이 연구는 단일 매개변수 자연 지수 가족(natural exponential families, NEFs)이 서브지수(subexponential) 꼬리를 가질 경우 자기 일관성(self-concordance)을 가진다는 것을 증명합니다. 또한, 서브가우시안(subgaussian) NEFs의 경우 자기 일관성 매개변수의 성장 속도를 정확히 특성화하는 결과를 도출했습니다. 이 발견을 밴딧(bandit) 문제에 적용하여, 일반화된 선형 밴딧(generalized linear bandits)에 대해 새로운 회귀 경계(regret bounds)를 제시했습니다.

- **Technical Details**: 자연 지수 가족의 꼬리 속성이 NEF의 한계를 명시하며, 서브지수 분포를 따르는 경우 자기 일관성 속성을 가집니다. 특히, 서브가우시안 NEFs는 그 변형 인자가 역 제곱적으로 성장하고, 일반화된 선형 밴딧의 보상 분포가 이런 NEF를 따를 경우, 문제 매개변수의 크기에 지수적 의존성이 없는 새로운 두 번째 차수 회귀 경계를 도출할 수 있습니다.

- **Performance Highlights**: 이 연구는 서브지수 꼬리를 가진 일반화된 선형 밴딧을 위한 최초의 회귀 경계를 제공하며, 정상 분포(normal), 포아송(Poisson), 감마(gamma), 음이항 분포(negative binomial 등)도 포함하는 다양한 문제로의 적용 가능성을 확장했습니다. 이로 인해 기존 문헌의 공백을 메우는 데 기여했습니다.



### Embedding-based statistical inference on generative models (https://arxiv.org/abs/2410.01106)
- **What's New**: 최근 공개된 생성 모델은 다양한 주제와 분야에서 인간 전문가 수준의 콘텐츠를 생성할 수 있습니다. 본 논문은 이러한 모델의 기반 모델에서 파생된 모델들 사이의 유사성을 바탕으로 통계적 추론을 확장하는 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 생성 모델의 임베딩 기반 표현을 사용하여 유사 모델들 간의 모델 수준의 추론을 수행하는 방법을 제안합니다. 또한 데이터 커널 관점에서의 아래 기술적 세부 사항이 포함됩니다: 모델의 매개변수(Parameters), 학습 혼합(Mixture) 비율, 모델 안전 점수 등의 변수를 통해 모델의 행동을 이해하기 위한 예측 방법이 필요합니다.

- **Performance Highlights**: 세 가지 하위 추론 작업에서 유사 모델의 존재를 예측하고 모델의 안전성을 예측하는 능력에 대한 실험적 조사를 포함합니다. 이를 통해 측정해야 하는 하이퍼파라미터에 대한 성능 민감도를 조사하였습니다.



### softmax is not enough (for sharp out-of-distribution) (https://arxiv.org/abs/2410.01104)
Comments:
          Comments welcome. 14 pages, 7 figures

- **What's New**: 이 연구는 softmax 함수의 한계를 밝히고, 이를 개선하기 위한 새로운 접근 방법을 제안합니다. 이전의 믿음과는 달리, softmax는 샤프한 결정(Sharp Decision)을 내리는 데 필요한 강력한 일반화 능력을 갖추고 있지 않음을 보여줍니다.

- **Technical Details**: softmax 함수는 입력 값의 벡터를 확률 분포로 변환하는 데 사용되며, 온도 매개변수(Temperature Parameter) θ를 적용할 수 있습니다. 연구에서는 이론적으로 softmax의 샤프함에서의 분산(Dispersion) 현상을 증명하고, 적응형 온도(adaptive temperature)를 제안하여 softmax의 샤프함을 향상시키는 방법을 모색합니다.

- **Performance Highlights**: 기계 학습 모델이 훈련 시 사용한 문제 크기를 넘어서는 경우, attentional coefficients는 균일 분포로 분산되는 경향이 있음을 확인했습니다. 이는 Transformers의 Attention Head에서도 동일하게 나타나며, 추론 시 입력 크기가 증가함에 따라 발생합니다.



### Exploiting Structure in Offline Multi-Agent RL: The Benefits of Low Interaction Rank (https://arxiv.org/abs/2410.01101)
- **What's New**: 이번 연구에서는 오프라인 다중 에이전트 강화 학습(MARL) 환경에서 근사 균형 학습 문제를 조사합니다. 여기서 상호작용 순위(interaction rank)라는 구조적 가정(structural assumption)을 도입하여 낮은 상호작용 순위를 가진 함수들이 일반적인 함수들에 비해 분포 이동(distribution shift)에 훨씬 더 강인하다는 것을 규명하였습니다.

- **Technical Details**: 이 연구는 낮은 상호작용 순위를 가진 함수 클래스(function classes)를 활용하며, 정규화(regularization) 및 무후회 학습(no-regret learning)과 결합하여, 오프라인 MARL에서 분산(decentralized), 계산적(computationally) 및 통계적(statistically) 효율적인 학습이 가능하다고 주장합니다.

- **Performance Highlights**: 검증 실험에서는 낮은 상호작용 순위를 가진 비평가 구조(critic architectures)가 오프라인 MARL에서 볼 수 있는 잠재력을 보여주며, 일반적으로 사용되는 단일 에이전트 가치 분해(single-agent value decomposition) 구조와의 차별점을 보여줍니다.



### Efficient and Private Marginal Reconstruction with Local Non-Negativity (https://arxiv.org/abs/2410.01091)
Comments:
          To appear at NeurIPS 2024

- **What's New**: 본 논문에서는 차별적 프라이버시를 위해 효율적이고 원칙적인 후처리 방법인 ReM (Residuals-to-Marginals)을 도입합니다. 이를 통해 노이즈가 포함된 측정값으로부터 주변 쿼리에 대한 답변을 재구성할 수 있습니다. 또한 GReM-LNN (Gaussian Residuals-to-Marginals with Local Non-negativity) 확장을 통해 가우시안 노이즈 하에서 일관성과 비부정성을 만족하는 마지널을 재구성하는 방법도 제안합니다.

- **Technical Details**: ReM 방법은 회귀 쿼리 기초를 기반으로 하여 노이즈가 포함된 측정값들로부터 마지널 쿼리에 대한 답변을 재구성하는 유효한 메커니즘을 사용합니다. 이 방법은 Kronecker 구조를 활용하여 효율적인 의사 역행렬(pseudo-inverse) 연산을 가능하게 하며, 이로 인해 높은 차원의 데이터 집합에서도 사용할 수 있습니다. GReM-LNN는 가우시안 노이즈 하에서 마지널들을 재구성하며, 일관성 및 비부정성을 보장하여 재구성된 답변의 오차를 줄입니다.

- **Performance Highlights**: ReM 및 GReM-LNN의 적용은 기존의 사적인 쿼리 응답 메커니즘인 ResidualPlanner 및 MWEM에서 오차를 크게 줄이고 확장성을 향상시키는 효과를 보여줍니다.



### Inferring Kernel $\epsilon$-Machines: Discovering Structure in Complex Systems (https://arxiv.org/abs/2410.01076)
- **What's New**: 이 논문은 이전에 제시된 컴퓨터 역학에서의 인과 상태(causal states) 개념을 확장하여, 이들이 생성하는 확산 구성 요소(causal diffusion components)를 명시적으로 도입하는 방법을 제안합니다. 이는 서로 다른 관측 및 시스템 유형에서 직접적으로 인과 구조를 추론할 수 있는 방법론을 제공합니다.

- **Technical Details**: 이 연구에서는 커널 ϵ-machine을 활용하여 인과 상태를 힐베르트 공간(Hilbert space)에서 점으로 표현하고, 이들의 동적 진화를 모델링하는 확률론적 방법을 수립합니다. 커널 인과 상태 알고리즘(empirical kernel causal states algorithm)은 데이터 차원이 다양한 시스템에서 예측 구조를 발견하는데 강력한 방법을 보여줍니다.

- **Performance Highlights**: 주요 사례로는 단순한 진자, n-부탄의 분자 역학 궤적, 월별 태양 흑점 시퀀스, 다년간의 농작물 관측 데이터를 사용하여 제시된 방법이 다양한 형태의 데이터에서 뛰어난 예측 성능을 발휘하는 것을 보여주었습니다.



### Convergent Privacy Loss of Noisy-SGD without Convexity and Smoothness (https://arxiv.org/abs/2410.01068)
- **What's New**: 이번 연구에서는 Noisy-SGD 알고리즘의 Differential Privacy (DP) 보증을 비선형 비매끄러운 손실에 대해서도 수렴 가능한 R'enyi DP 경계를 제공함으로써 기존의 제한된 가정보다 더 일반화된 접근 방식을 제시하고 있습니다.

- **Technical Details**: 연구팀은 Hölder 연속 경계(gradient)가 있는 손실 함수에 대해 Noisy-SGD 알고리즘의 프라이버시 손실이 수렴하는 비자명한 값인지를 증명했습니다. 이 모델은 기존의 연구들이 요구하는 매끄럽고 강하게 볼록한 손실에 대한 제한을 넘어섰습니다.

- **Performance Highlights**: 이 연구는 다양한 손실 함수에서도 더 나은 프라이버시 경계를 제공하며, 기존의 알고리즘들보다 개선된 성능을 보입니다. 특히, 매끄러운 강한 볼록성의 손실에 대해서는 기존 복잡성과 비교했을 때 더 좋은 프라이버시 보장을 입증했습니다.



### Structure-Preserving Operator Learning (https://arxiv.org/abs/2410.01065)
- **What's New**: 이번 논문에서는 복잡한 물리 시스템의 빠르고 정확한 시뮬레이션을 위해 데이터를 기반으로 하는 부분 미분 방정식(Partial Differential Equations) 학습에 대한 새로운 접근법을 제시합니다. 특히, 구조를 보존하는 연산자 네트워크(Structure-Preserving Operator Networks, SPONs)를 도입하여 연속 시스템의 주요 수학적 및 물리적 속성을 유지할 수 있는 방법을 설명합니다.

- **Technical Details**: SPONs는 encode-process-decode 아키텍처로, 입력-출력 공간의 유한 요소(Finite Element, FE) 이산화를 활용하여 설계되었습니다. 이 아키텍처는 복잡한 기하학을 처리할 수 있으며, 특정 경계 조건을 정확하게 적용하고 이론적 보장을 제공합니다. 또한, 다양한 응용 분야에 맞춤형 구조 보존 아키텍처를 개발할 수 있는 유연한 프레임워크를 제공합니다.

- **Performance Highlights**: 이 논문에서 제안하는 다중 격자(multi-grid) 영감을 받은 SPON 아키텍처는 향상된 성능을 더 높은 효율성으로 달성할 수 있는 잠재력을 가지고 있으며, SPON 구조의 설계 및 훈련을 자동화하는 소프트웨어도 공개되었습니다.



### Spherical Analysis of Learning Nonlinear Functionals (https://arxiv.org/abs/2410.01047)
- **What's New**: 최근 심층 ReLU 신경망의 함수 근사 능력을 구체적으로 다루며, 구형 표면에 정의된 함수에 대한 고유한 분석을 시도합니다.

- **Technical Details**: 이 연구에서는 인코더-디코더(framework) 구조를 사용하여 무한 차원 함수에 대해 구형 조화 함수(spherical harmonics)를 활용하여 잠재적인 유한 차원 정보를 추출하고, 완전 연결 신경망을 통해 근사를 분석합니다.

- **Performance Highlights**: 다양한 인코더 구조에 대한 근사율을 제공하며, 이는 실제 데이터의 이산성과 잡음에 대한 저항력을 포함합니다.



### Don't Stop Me Now: Embedding Based Scheduling for LLMs (https://arxiv.org/abs/2410.01035)
- **What's New**: 본 논문에서는 LLM 시스템에서의 효율적인 스케줄링을 위해 TRAIL이라는 새로운 방법을 제안합니다. 이 방법은 LLM의 내부 구조를 기반으로 한 임베딩을 활용하여 출력 길이를 예측하고, 메모리 오버헤드를 줄이는 방향으로 설계되었습니다.

- **Technical Details**: TRAIL 방법은 각 출력 토큰을 생성한 후, 내부 임베딩을 재활용하여 경량 분류기에 전달합니다. 이 분류기는 각 실행 중인 요청의 남은 길이를 예측합니다. 이를 통해 메모리 오버헤드를 고려한 예측 기반의 Shortest Remaining Process Time (SRPT) 변형을 제안합니다. 이 변형은 요청 실행 초기에는 선제적 중단(preemption)이 가능하지만, 요청이 완료에 가까워질수록 중단을 제한하여 자원 활용을 최적화합니다.

- **Performance Highlights**: 이론적 측면에서는 M/G/1 큐 모델에서 SRPT 변형의 폐쇄형 수식을 도출하여 잠재적 가치를 보여줍니다. 실험에서는 TRAIL의 정책과 예측 방법의 구현이 LLM 시스템의 효율성을 증대시키는 것을 보여주었습니다.



### GPTreeO: An R package for continual regression with dividing local Gaussian processes (https://arxiv.org/abs/2410.01024)
- **What's New**: 이번 논문에서는 지속적 학습 문제에 적합한 확장 가능한 Gaussian process (GP) 회귀를 위한 유연한 R 패키지인 GPTreeO를 소개합니다. GPTreeO는 지속적인 입력 데이터 스트림을 이용하여 동적으로 지역 GP 회귀 모델의 이진 트리를 구성하는 Dividing Local Gaussian Processes (DLGP) 알고리즘을 기반으로 합니다. GPTreeO는 GP 하이퍼파라미터의 지속적인 최적화, 불확실성 보정(uncertainty calibration) 통합 및 지역 파티션 생성을 위한 새로운 전략 도입 등 여러 모듈형으로 확장되었습니다.

- **Technical Details**: GPTreeO는 사용자가 선호하는 GP 라이브러리와 인터페이스 할 수 있는 모듈화된 코드 구조를 가지고 있습니다. 사용자는 계산 속도(computational speed), 정확도(accuracy), 안정성(stability), 매끄러움(smoothness) 간의 균형을 세밀하게 조정할 수 있습니다. 여러 구성 가능한 기능이 지속적 학습 환경에서 회귀 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: GPTreeO의 구성 가능한 기능들이 지속적 학습 성능에 미치는 영향을 살펴본 민감도 분석(sensitivity analysis)을 수행하였으며, 이를 통해 다양한 상황에서의 응용 가능성과 GPTreeO의 유연성을 강조합니다.



### Back to Bayesics: Uncovering Human Mobility Distributions and Anomalies with an Integrated Statistical and Neural Framework (https://arxiv.org/abs/2410.01011)
Comments:
          12 pages

- **What's New**: 이 논문에서는 DeepBayesic이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Bayesian 원리를 심층 신경망(deep neural networks)과 통합하여 희소하고 복잡한 데이터셋으로부터 기본적인 다변량 분포(multivariate distributions)를 모델링합니다.

- **Technical Details**: DeepBayesic은 다항 입력을 처리할 수 있도록 설계되어 있으며, 연속형 데이터와 범주형 데이터 모두를 수용합니다. 이 프레임워크는 사용자 정의된 신경 밀도 추정기(neural density estimators)와 하이브리드 아키텍처를 특징으로 하여 다양한 특성 분포를 모델링하는 유연성을 제공합니다. 또한, 개별 에이전트에 대한 개인화된 이상 탐지를 위한 에이전트 임베딩(agent embeddings)을 활용합니다.

- **Performance Highlights**: DeepBayesic은 여러 모빌리티 데이터셋에서 기존의 최첨단 이상 탐지 방법보다 유의미한 개선을 보여줍니다. 개인화 및 고급 시퀀스 모델링 기법을 통합함으로써 시공간 사건 시퀀스에서 미세하고 복잡한 이상을 탐지하는 능력이 크게 향상되었음을 입증하였습니다.



### CktGen: Specification-Conditioned Analog Circuit Generation (https://arxiv.org/abs/2410.00995)
- **What's New**: 본 논문에서는 특정 사양에 기반하여 아날로그 회로를 직접 생성하는 방식을 소개합니다. 이 방법은 전통적인 최적화 문제 접근법의 재사용성과 전이 가능성을 개선하는 데 중점을 둡니다.

- **Technical Details**: CktGen은 변형 오토인코더(Variational Autoencoder, VAE) 모델을 사용하여 베리어블(variables)과 회로를 공동 잠재 공간(laten space)으로 매핑합니다. 또한 대조 학습(contrastive learning)와 분류기 가이드를 통합하여 모델이 서로 다른 사양에 대해 대응할 수 있도록 합니다.

- **Performance Highlights**: Open Circuit Benchmark (OCB)에서 실시한 실험 결과, 기존 최첨단 방법들에 비해 상당한 개선을 보였으며, 생성된 회로는 지정된 사양에 잘 부합하는 것으로 평가되었습니다.



### Tight Rates for Bandit Control Beyond Quadratics (https://arxiv.org/abs/2410.00993)
Comments:
          Neurips 2024

- **What's New**: 본 논문은 adversarial perturbations(적대적 섭동)과 비확률적 비용 함수가 있는 환경에서도 $	ilde{O}(	ext{sqrt}(T))$의 최적 회귀(tx)값을 달성하는 알고리즘을 제시합니다. 이는 기존에 알려진 $	ilde{O}(T^{2/3})$ 회귀 경계보다 향상된 결과입니다.

- **Technical Details**: 제안된 알고리즘은 Bandit Convex Optimization (BCO)의 메모리 없는 버전으로 문제를 단순화하여 대칭적으로 강한 볼록성과 부드러운 비용 함수 아래에서 동작합니다. 저자들은 메모리 문제를 극복하기 위해 강화된 알고리즘을 개발하였습니다.

- **Performance Highlights**: 결과적으로 이 연구는 adversarial perturbations와 bandit 피드백 모델에 대한 최적 성능을 보이는 것으로 입증된 최초의 방법입니다. 이 연구는 BCO의 메모리 문제를 해결하고 강한 볼록 비용을 다루기 위해 최근의 기술을 활용하였습니다.



### Tackling the Accuracy-Interpretability Trade-off in a Hierarchy of Machine Learning Models for the Prediction of Extreme Heatwaves (https://arxiv.org/abs/2410.00984)
- **What's New**: 이번 연구는 복잡한 머신러닝 모델을 사용하여 프랑스의 극단적인 폭염 예측을 수행하고 이는 예측 정확성과 해석 가능성 간의 균형을 최적화하는 접근 방식을 제안합니다. 전통적인 모델부터 복잡한 CNN(Convolutional Neural Networks)까지의 계층적 모델을 비교하여 해석 가능성이 높은 결과를 도출하였습니다.

- **Technical Details**: 모델 구조는 전역 Gaussian Approximation(GA)에서 시작하여 Intrinsically Interpretable Neural Network(IINN), Scattering Transform 기반 모델(ScatNet), 마지막으로 Deep Convolutional Neural Networks(CNNs)로 발전합니다. 이 과정에서 ScatNet은 CNNs와 비슷한 성능을 보이면서도 해석 가능성이 높다는 장점이 있습니다.

- **Performance Highlights**: CNNs는 높은 정확성을 제공하지만 블랙박스 특성으로 인해 해석 가능성이 제한됩니다. 반면, ScatNet은 신뢰성 있는 데이터 패턴과 스케일을 식별하여 극단적인 폭염 예측에서 효과적으로 작용함을 보여주었습니다. 또한, 단순한 모델이 더 복잡한 모델과 비슷한 성과를 낼 수 있다는 사실을 강조하면서 기후 과학 내 머신러닝 모델의 해석 가능성의 중요성을 부각시켰습니다.



### Robust Guided Diffusion for Offline Black-Box Optimization (https://arxiv.org/abs/2410.00983)
Comments:
          21 pages

- **What's New**: 오프라인 블랙 박스 최적화 분야에 있어, 본 논문에서는 'Robust Guided Diffusion' (RGD)라는 새로운 프레임워크를 제안합니다. 이는 프로시(Proxy)와 프로시-프리(diffusion-free) 디퓨전의 장점을 통합하여 효과적인 조건부 생성을 가능하게 합니다.

- **Technical Details**: RgD는 두 가지 주요 구성 요소로 구성됩니다. 첫째, 'proxy-enhanced sampling' 모듈을 도입하여 프로시로부터의 명시적 가이드를 프로시-프리 디퓨전으로 통합하고, 샘플링 과정을 향상시킵니다. 둘째, 'diffusion-based proxy refinement' 모듈을 통해 프로시의 견고성과 신뢰성을 높이는 정규화 전략을 설계하였습니다.

- **Performance Highlights**: RGD는 다양한 디자인 벤치 작업에서 최첨단 성능을 달성하며 그 효능을 강조합니다. 이를 통해 높은 성능의 샘플 생성을 가능하게 하고 있습니다.



### RisingBALLER: A player is a token, a match is a sentence, A path towards a foundational model for football players data analytics (https://arxiv.org/abs/2410.00943)
Comments:
          18 pages, 6 figures. The paper will be presented at the StatsBomb Conference 2024 (this https URL)

- **What's New**: 본 논문에서는 RisingBALLER를 소개합니다. 이는 축구 경기 데이터를 기반으로 학습된 트랜스포머 모델을 활용하여 각 경기별 선수 표현을 배우는 최초의 공개 접근 방식입니다. 선수는 경기의 특정 맥락에 따라 임베딩됩니다.

- **Technical Details**: RisingBALLER는 Masked Player Prediction (MPP)이라는 사전 훈련 작업을 통해 축구 선수 표현을 학습하고, Next Match Statistics Prediction (NMSP)을 다운스트림 작업으로 설정하여 학습된 선수 임베딩의 효과를 보여줍니다. 이 모델은 트랜스포머 아키텍처를 사용하여 각 축구 경기를 선수의 시퀀스로 취급하고, 선수 통계와 팀 표현을 결합하여 다차원적 임베딩을 생성합니다.

- **Performance Highlights**: RisingBALLER는 NMSP 모델에서 일반적으로 사용되는 강력한 기준선 모델을 초과하는 성능을 보여줍니다. 본 연구는 선수 및 포지션 임베딩의 분석을 통해 전술적 역할 이해, 데이터 기반 팀 구성, 스카우트 전략의 향상 가능성을 제시합니다.



### MoS: Unleashing Parameter Efficiency of Low-Rank Adaptation with Mixture of Shards (https://arxiv.org/abs/2410.00938)
- **What's New**: 이 논문에서는 더 많은 맞춤형 모델을 동시에 서비스하기 위한 경량화된 finetuning 방법으로 Mixture of Shards (MoS)를 제안합니다. 이 방법은 LoRA의 장점을 유지하면서도 파라미터 효율성을 향상시킵니다.

- **Technical Details**: MoS는 전반적인 파라미터 공유 기법과 함께, 세분화된 선택(subset selection), 쌍 분리(pair dissociation), 벡터 세분화(vector sharding), 세분화 개인화(shard privatization)라는 네 가지 차별화 전략을 통합합니다. 이러한 전략은 파라미터 효율성을 높이기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, MoS는 기존의 LoRA 설정에 비해 약 8배의 파라미터 절약을 보여주었으며, 각 차별화 전략의 중요성을 입증하는 탈피(ablation) 연구를 통해 모든 구성 요소가 MoS의 파라미터 효율성 발휘에 기여함을 확인하였습니다.



### ACEV: Unsupervised Intersecting Manifold Segmentation using Adaptation to Angular Change of Eigenvectors in Intrinsic Dimension (https://arxiv.org/abs/2410.00930)
Comments:
          14 pages, 7 figures, 7 tables

- **What's New**: 본 논문에서 제안된 방법은 교차하는 매니폴드 분할(Intersecting manifold segmentation)에 중점을 두고 있으며, 매니폴드의 내부 차원(intrinsic dimension) 및 구조를 학습하여 교차 영역에 존재하는 데이터 포인트를 효과적으로 식별합니다.

- **Technical Details**: 제안된 방법(ACEV)은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 비교차(non-intersecting) 매니폴드를 분할하고, 두 번째 단계에서는 개별 교차 매니폴드를 분할합니다. 이 방법은 로컬 데이터 분산(local data variances)을 측정하고 벡터 방향을 파악하여 매니폴드의 내부 차원을 결정합니다. 또한 지수 이동 평균(exponential moving averages)을 이용하여 부모와 자식의 방향 벡터 간의 각도 차이를 조정하여 교차 영역을 탐지합니다.

- **Performance Highlights**: ACEV 방법은 14개의 실제 데이터셋에서 18개의 최신 상태(SOTA) 매니폴드 분할 방법보다 ARI와 NMI 점수에서 더 나은 성능을 보였으며, 시간 복잡도(time complexity)가 낮고 안정성이 높은 결과를 나타냈습니다.



### Efficient $1$-bit tensor approximations (https://arxiv.org/abs/2410.01799)
Comments:
          16 pages, one cat picture reused a lot

- **What's New**: 이번 논문에서는 $	ext{-1, 1}$ 값의 벡터를 통한 텐서 곱의 선형 조합으로 행렬과 임의 차수 텐서를 공간적으로 효율적으로 분해하는 방법을 제안합니다. 특히, 이 방법은 메모리 효율성을 고려하여 가벼운 저장 방식을 제공하며, 계산의 정확도도 유지합니다.

- **Technical Details**: 논문에서는 $A 	ext{ (matrix)} 	ext{ and } R_w 	ext{ where } A - R_w = S_w C_w T_w^	op$를 소개합니다. 여기서 $S_w$와 $T_w$는 각각 $	ext{-1, 1}$ 값의 벡터로 구성되며 $C_w$는 대각 행렬(diagonal matrix)입니다. 메모리 요구사항 측면에서 $(S_w, T_w, C_w)$는 $w 	imes (m + n)$ 비트로 저장할 수 있으며, 이는 $w$ 개의 부동 소수점 숫자(float numbers)만 필요합니다.

- **Performance Highlights**: 논문의 첫 번째 응용으로 Mistral-7B-v0.1 대형 언어 모델의 가중치 행렬(weight matrix)들을 $50	ext{%}$ 공간 압축으로 근사화하였고, 상대 오차(relative error)는 $<6	ext{%}$였습니다. 벤치마크 성능은 공간 압축을 $50	ext{%}$에서 $25	ext{%}$로 줄여도 느리게 저하되었습니다. 또한, 오픈 소스 	extit{rust} 구현에 대해 	extit{simd} 명령어를 통한 최적화를 수행하였습니다.



### Thermodynamic Bayesian Inferenc (https://arxiv.org/abs/2410.01793)
Comments:
          20 pages, 8 figures

- **What's New**: 이번 연구는 역학적 컴퓨팅(thermodynamic computing)을 기반으로 한 새로운 전자 아날로그 장치를 제안하여, 복잡한 베이지안 포스터리어(Bayesian posteriors) 샘플링을 자동화한다.

- **Technical Details**: 이 연구에서는 랜주빈(Langevin) 동역학을 물리적으로 구현함으로써 베이지안 포스터리어를 샘플링 하는 전자 회로 설계를 제시한다. 제안된 회로는 가우시안-가우시안 모델(Gaussian-Gaussian model)과 베이지안 로지스틱 회귀(Bayesian logistic regression)의 포스터리어를 샘플링하는 구조로 설계되었다. 이러한 장치들은 차원 d에 대해 O(N ln d)의 시간 복잡도를 가지고 샘플을 생성할 수 있다.

- **Performance Highlights**: 가우시안-가우시안 모델에 대한 에너지 비용은 d ln d에 비례하며, 이는 기존의 디지털 방식으로 샘플링하는 것보다 훨씬 빠르고 에너지 효율적이다. 특히 비가우시안 샘플링에도 적용 가능성이 있음을 보여준다.



### Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models (https://arxiv.org/abs/2410.01782)
Comments:
          Accepted to EMNLP 2024 Findings. Website: this https URL. 14 pages, 7 figures, 5 tables

- **What's New**: 본 논문에서는 Open-RAG라는 새로운 프레임워크를 제안하여 여러 오픈소스 LLM과 함께 RAG를 활용한 추론 능력을 향상시키고자 합니다. 이 프레임워크는 복잡한 추론 작업을 처리할 수 있는 파라미터 효율적인 희소 MoE 모델로의 변환을 통해 성능을 높입니다.

- **Technical Details**: Open-RAG은 기존의 LLM을 기반으로 하여 모델이 자체 반성을 통해 레프리케이션(retrieval)과 관련된 특수 토큰을 생성하면서, 복잡한 질문에 대한 추론을 보다 효과적으로 수행할 수 있도록 학습합니다. 또한, 혼합 적응형 검색(hybrid adaptive retrieval) 방법을 도입하여 성능 향상과 추론 속도의 균형을 맞춥니다.

- **Performance Highlights**: Llama2-7B 기반의 Open-RAG는 ChatGPT, Self-RAG 및 Command R+와 같은 최첨단 LLM 및 RAG 모델을 초월하는 성과를 보이며, 다양한 지식 집약적 작업에서 새로운 벤치마크를 설정하였습니다. 실험 결과, Open-RAG가 선행 오픈소스 RAG 모델에 비해 의미 있는 정확도 향상과 추론 능력 개선을 보여주었습니다.



### Dynamical-generative downscaling of climate model ensembles (https://arxiv.org/abs/2410.01776)
- **What's New**: 이 연구에서는 기존의 기후 예측 다운스케일링 방법을 혁신적으로 개선한 새로운 접근 방식을 제안합니다. 이는 재생산 인공지능(Generative AI)과 동적 다운스케일링(Dynamical Downscaling)을 결합한 것으로, 대형 기후 예측 앙상블의 다운스케일링 비용을 줄이고 불확실성 추정치를 향상시킵니다.

- **Technical Details**: 제안된 방법론은 먼저 지역 기후 모델(Regional Climate Model, RCM)을 이용하여 지구 시스템 모델(Earth System Model, ESM)의 출력을 중간 해상도로 다운스케일링합니다. 그 후 generative diffusion model을 사용하여 원하는 해상도로 추가적으로 정제합니다. 이 과정에서 RCM은 ESM의 데이터를 대기 상태로 변환하고, 이후 generative model이 다중 모델 앙상블을 다운스케일링 하는 데 필요한 확률 분포를 샘플링합니다.

- **Performance Highlights**: 이 방법은 기존의 동적 다운스케일링이나 전통적인 통계적 방법과 비교할 때, 더 정확한 불확실성 범위를 제공하며, 기존의 방법보다 훨씬 낮은 오류를 기록했습니다. 또한, 기상 필드의 스펙트럼과 다변량 상관관계를 보다 정확하게 포착하는 데 성공했습니다. 이를 통해 대형 기후 예측 앙상블을 효율적으로 다운스케일링할 수 있는 유연하고 정확한 방법론이 제시되었습니다.



### SegHeD: Segmentation of Heterogeneous Data for Multiple Sclerosis Lesions with Anatomical Constraints (https://arxiv.org/abs/2410.01766)
Comments:
          13 pages, 4 figures, MICCAI, LDTM Workshop

- **What's New**: SegHeD는 이질적인 데이터 형식과 주석 프로토콜을 처리하는 새로운 다중 데이터셋 다중 작업 뇌 병변 세분화 모델입니다. 이 모델은 모든 병변, 새로운 병변, 그리고 소멸 병변을 세분화할 수 있는 기능을 제공합니다.

- **Technical Details**: SegHeD는 크로스 섹션(cross-sectional) 및 종단적(longitudinal) 이미지를 사용하여 학습하며, 세 가지 주석 프로토콜(모든 병변, 새로운 병변, 소멸 병변)을 고려하여 설계되었습니다. 모델은 구조적 일관성과 부피 일관성을 유지하면서도 서로 다른 데이터 형식의 입력을 가능하게 합니다.

- **Performance Highlights**: SegHeD는 다섯 개의 MS 데이터셋에서 평가되었으며, 모든 병변, 새로운 병변, 소멸 병변 세분화에서 높은 성능을 보여주며 여러 최첨단 방법보다 우수한 결과를 달성했습니다.



### Integrating Protein Sequence and Expression Level to Analysis Molecular Characterization of Breast Cancer Subtypes (https://arxiv.org/abs/2410.01755)
- **What's New**: 이 연구는 단백질 서열 데이터와 발현 수준을 통합하여 유방암 아형의 분자적 특성을 개선하고 임상 결과를 예측하는 것을 목표로 합니다. ProtGPT2라는 단백질 서열 전용 언어 모델을 사용해 단백질의 기능적 및 구조적 특성을 포착하는 임베딩(embeddings)을 생성하였으며, 이를 통해 머신러닝 기법을 활용해 유방암 환자를 생물학적으로 구별된 그룹으로 클러스터링하고 임상 결과를 정확히 예측하는 데 성공하였습니다.

- **Technical Details**: 연구에서는 Proteomics 데이터를 사용하여 105개의 유방 종양 샘플을 분석했습니다. 연구에 사용된 데이터 세트에서는 고해상도 질량 분석법을 통해 12,553개의 단백질과 임상 데이터를 확보하여, ProtGPT2를 활용해 단백질 서열을 임베딩하고 머신러닝 알고리즘을 통해 클러스터링 및 분류를 수행했습니다. 주요 머신러닝 기법으로는 XGBoost와 앙상블 K-means가 사용되었습니다.

- **Performance Highlights**: 연구의 결과, 생존 상태 예측에 대한 F1 점수는 0.88, 바이오마커 상태 예측에 대한 F1 점수는 0.87을 달성하였으며, KMT2C, GCN1 및 CLASP2 등의 주요 단백질이 확인되었습니다. 이러한 단백질은 호르몬 수용체 및 HER2 발현과 관련이 있으며, 유방암의 진행 및 환자 결과에 중요한 역할을 한다고 볼 수 있습니다.



### Mimicking Human Intuition: Cognitive Belief-Driven Q-Learning (https://arxiv.org/abs/2410.01739)
Comments:
          Under review by ICLR 25

- **What's New**: 이 논문에서는 전통적인 Q-learning 알고리즘의 한계를 극복하기 위해 주관적인 신념 모델링을 통합한 Cognitive Belief-Driven Q-Learning (CBDQ) 방법론을 제안합니다. 이 방법은 의사결정의 정확성을 높이고, 인간과 유사한 학습 및 추론 능력을 에이전트에게 제공합니다.

- **Technical Details**: CBDQ는 (1) 주관적 신념 구성 요소, (2) 인간 인지 클러스터, (3) 신념-선호 결정 프레임워크(BPDF)를 통합하여 Q-learning의 성능을 향상시킵니다. 이를 통해 Q-learning의 과대 평가 문제를 해결하고, 환경의 상태 공간을 클러스터링하여 고차원 데이터를 의미 있는 저차원 표현으로 압축합니다. 이는 인간의 인지 방식을 모방하여 상태 표현을 개선합니다.

- **Performance Highlights**: CBDQ는 다양한 복잡한 환경에서 이전의 Q-learning 알고리즘들보다 높은 보상을 지속적으로 달성하며, 경쟁하는 알고리즘들과 비교하여 더 향상된 적응력과 강인성을 보여줍니다. 이 연구는 Q-learning에 대한 새로운 접근법을 제시하여 복잡한 결정 시스템에서의 인간과 유사한 에이전트를 향한 진전을 이룹니다.



### Recursive Abstractive Processing for Retrieval in Dynamic Datasets (https://arxiv.org/abs/2410.01736)
- **What's New**: 이 논문에서는 동적 데이터셋에 대한 효율적인 업데이트를 위해 새로운 알고리즘을 제안하고 있습니다. 기존의 Retrieval-Augmented 방법들은 클러스터링을 통해 계층 구조를 형성하지만, 문서의 추가 및 제거가 잦은 동적 데이터셋에서는 이 구조를 업데이트하는 것이 복잡하고 비효율적입니다. 제안된 알고리즘은 이러한 계층 구조를 유지하면서도 성능을 저하시키지 않습니다.

- **Technical Details**: 우리는 adRAP (adaptive Recursive Abstractive Processing)이라는 알고리즘을 도입하여 RAPTOR의 재귀적 요약 구조를 동적으로 업데이트합니다. 이 알고리즘은 새로운 문서가 추가되거나 제거될 때 전체 재계산을 피하면서 성능을 유지합니다. 또한 postQFRAP라는 새로운 포스트 리트리벌 방법을 소개합니다. 이 방법은 쿼리 중심의 재귀적 요약 처리를 통해 수집된 맥락의 질을 크게 향상시킵니다.

- **Performance Highlights**: 실제 데이터셋을 통해 수행된 광범위한 실험에서는 adRAP가 RAPTOR 트리 구조를 잘 근사하며, postQFRAP가 리트리벌 품질을 효과적으로 향상시킨다는 것을 보여주었습니다. 이 방법들은 동적 데이터 처리와 리트리벌 성능 개선에 효과적입니다.



### LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits (https://arxiv.org/abs/2410.01735)
Comments:
          20 pages; First two authors contributed equally. Code: this https URL

- **What's New**: LASeR (Learning to Adaptively Select Rewards)는 여러 개의 리워드 모델(Reward Models, RMs)을 사용하여 LLM을 적응적으로 훈련하는 새로운 접근법입니다. 이 방법은 학습 과정 중 각 인스턴스에 가장 적합한 RM을 선택하여 리워드의 선택을 다중 무장 강도 문제(multi-armed bandit problem)로 표현합니다.

- **Technical Details**: LASeR는 RMs의 선택을 문맥 정보와 과거 상호작용에 기반해 동적으로 진행합니다. 이는 LLM의 성능과 각 RM의 적합성을 반영하여 훈련을 진행하며, RM 업데이트는 그 성능에 따라 조정됩니다. 이 방법은 단일 RM 사용에 따른 한계를 해결하며 성능 향상을 도모합니다.

- **Performance Highlights**: LASeR를 통해 Llama-3-8B 모델은 commonsense 및 math reasoning 테스트에서 최대 2.67%의 정확도 개선을 달성했습니다. WildChat 데이터셋에서 LASeR를 사용하는 경우 71.45%의 win rate를 기록했으며, 긴 컨텍스트 생성 작업에서도 평균적으로 F1 스코어 2.64 및 2.42의 향상을 이끌어냈습니다.



### Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspectiv (https://arxiv.org/abs/2410.01720)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 후속 훈련 단계에서의 합성 데이터 생성과 일반화 능력 간의 관계를 체계적으로 분석합니다. 특히, 합성 데이터의 효과를 이론적으로 모델링하고, 정보 이득(Information Gain) 과 일반화 이득(Generalization Gain) 간의 새로운 개념인 GGMI(Generalization Gain via Mutual Information)를 소개합니다.

- **Technical Details**: 저자들은 합성 데이터 생성 과정을 분포 관점에서 모델링하고, 후속 훈련이 진행 중인 LLM에 미치는 합성 데이터의 영향을 분석하기 위해 역 병목 구조(Reverse Bottleneck Framework)를 제시합니다. 이 접근법은 합성 데이터가 LLM의 일반화 능력에 미치는 효과를 정량화할 수 있는 상한(Upper Bounds)을 제공합니다.

- **Performance Highlights**: 많은 최신 LLM들이 합성 데이터를 활용함으로써 훈련 성과를 개선하고 있으며, 이 논문은 합성 데이터를 통해 LLM의 성능과 신뢰성을 높일 수 있는 방법에 대한 통찰을 제공합니다. 이 연구는 특히 제한된 실제 데이터의 상황에서 LLM의 일반화 능력을 어떻게 향상시킬 수 있는지를 탐구하며, 합성 데이터의 설계 및 최적화 프로세스를 이해하는 데 중요한 기여를 합니다.



### Smaller Confidence Intervals From IPW Estimators via Data-Dependent Coarsening (https://arxiv.org/abs/2410.01658)
Comments:
          Accepted for presentation at the 37th Conference on Learning Theory (COLT) 2024

- **What's New**: 이번 논문에서는 인과 추론(causal inference)에서 평균 처리 효과(average treatment effects)를 추정하기 위한 IPW(Inverse Propensity-Score Weighted) 추정기의 한계를 극복하기 위해 Coarse IPW(CIPW) 추정기를 제안합니다.

- **Technical Details**: CIPW 추정기는 공변량(covariate) 공간에서 특정 공변량이 병합된(coarsened) IPW 추정기로, 기존의 IPW 추정기 및 그 변형들을 포괄합니다. 제안된 방법은 부드러운 가정(mild assumptions)을 기반으로 하며, 특히 극단적인 성향 점수(extreme propensity scores)가 Sparse 할 경우 효과적인 알고리즘을 통해 정확도가 개선된 추정기를 찾을 수 있도록 합니다.

- **Performance Highlights**: 제안된 CIPW 추정기의 신뢰 구간(confidence intervals) 크기는 추정기의 정확도와 샘플의 개수에 따라 조정되며, 기존 추정기와는 달리 정확도에 대한 강건함(robustness)을 보여줍니다. 기존 추정기는 특정 조건에서도 신뢰 구간의 크기가 $oldsymbol{	ext{Ω(1)}}$로 고정되어 있지만, CIPW는 $oldsymbol{	ext{ε + 1/√n}}$으로 크기 조절이 가능합니다.



### Scalable and Consistent Graph Neural Networks for Distributed Mesh-based Data-driven Modeling (https://arxiv.org/abs/2410.01657)
- **What's New**: 본 연구는 일관된 신경 메시지 전송 계층을 활용하여 메쉬 기반 모델링 응용을 위한 분산 그래프 신경망(GNN) 방법론을 개발했습니다. 여기서 주요 초점은 하위 그래프 경계에 있는 Halo 노드를 통해 물리적 일관성을 유지하면서 확장 가능한 작업을 가능하게 하는 것입니다.

- **Technical Details**: 이 연구에서 제안된 GNN 접근법은 아르곤 국립연구소에서 개발된 GPU 지원 엑사스케일 CFD 솔버인 NekRS와의 인터페이스를 통해 입증되었습니다. 메쉬는 CFD 코드의 도메인 분해 정보를 활용하여 하위 그래프와 Halo 교환 정보를 제공함으로써 일관된 분산 GNN 작업을 가능하게 합니다. 또한, 크기 조정 및 그래프 크기 구성에 대한 폭넓은 분석이 Frontier 슈퍼컴퓨터에서 수행되었습니다.

- **Performance Highlights**: 일관된 GNN은 Frontier 엑사스케일 슈퍼컴퓨터에서 O(1B) 그래프 노드로의 효율적인 스케일링을 보여주었습니다. 이를 통해 물리 기반 시뮬레이션에서의 복잡한 데이터 처리 능력을 크게 향상시키고, 큰 스케일에서 물리적 현상을 모델링하는 데 있어 GNN 방식의 가능성을 입증했습니다.



### Efficient Statistics With Unknown Truncation, Polynomial Time Algorithms, Beyond Gaussians (https://arxiv.org/abs/2410.01656)
Comments:
          Accepted for presentation at the 65th IEEE Symposium on Foundations of Computer Science (FOCS), 2024; abstract shortened for arXiv

- **What's New**: 본 논문에서는 알려지지 않은 집합 $S \subseteq \mathbb{R}^d$ 내에 있는 샘플에 대해서 분포 추정 분포 파라미터를 추정하는 방법을 연구합니다. 이는 Gaussian 분포의 경우에 집중하고 있으며, 이전 연구에서 밝혀진 $1/\varepsilon$에 대한 지수적 의존성이 필요함을 보완하고 있습니다.

- **Technical Details**: 우리는 구조적 가정이 충족되는 지수 가족을 위한 알고리즘을 제안하며, 이는 적어도 $\varepsilon$ 근사 가능도를 가진 다항식의 차수 $\ell$을 만족하는 알려지지 않은 집합 $S$에도 적용됩니다. 이는 알려지지 않은 $S$에 대해 임의의 Gaussian 분포를 추정하는 첫 번째 알고리즘과 선형 회귀(linear regression) 문제를 해결하기 위한 알고리즘을 포함하고 있습니다.

- **Performance Highlights**: 우리는 또한 $S$가 반공간(halfspace) 또는 축에 정렬된 직사각형(axis-aligned rectangle)인 경우에 대해 모든 Gaussian을 포함하는 지수 가족을 위한 $\mathrm{poly}(d/\varepsilon)$ 시간 알고리즘을 제공하여 성능을 높였습니다. 이는 PAC 학습에서의 긍정 샘플과 레이블이 없는 샘플을 활용한 독립적인 도구 개발로 이어졌습니다.



### A Novel Framework of Horizontal-Vertical Hybrid Federated Learning for EdgeIo (https://arxiv.org/abs/2410.01644)
Comments:
          5 pages, 3 figures

- **What's New**: 이번 연구에서는 모바일 엣지 컴퓨팅(Edge Computing)을 지원하는 IoT 환경에서의 새로운 하이브리드 수평-수직 연합 학습(HoVeFL) 프레임워크를 제안합니다. 이 프레임워크는 다양한 데이터 샘플을 가지고 있지만 공통된 데이터 특징을 분석하는 EdgeIoT 장치와, 동일한 샘플에 대해 서로 다른 특징을 집중하는 장치들로 구성되어 있습니다.

- **Technical Details**: HoVeFL은 EdgeIoT에서 지역 모델과 글로벌 모델의 훈련을 통해 글로벌 손실 함수(global loss function)를 최소화하도록 구성됩니다. 장치는 수평 FL(HFL)과 수직 FL(VFL)을 조합하여 서로 다른 데이터 샘플에 대해 특성(Feature)에 집중하고, 다양한 장치에서의 비독립적이고 동일하게 분포되지 않은(non-IID) 데이터 샘플을 활용합니다.

- **Performance Highlights**: CIFAR-10과 SVHN 데이터셋에서의 성능 평가 결과, 12개의 수평 FL 장치와 6개의 수직 FL 장치를 사용하는 HoVeFL이 6개의 수평 FL 장치와 12개의 수직 FL 장치를 사용하는 설정에 비해 테스트 손실이 각각 5.5%와 25.2% 더 높았습니다.



### DRUPI: Dataset Reduction Using Privileged Information (https://arxiv.org/abs/2410.01611)
- **What's New**: 본 논문에서는 기존의 데이터셋 축소(Dataset Reduction, DR) 기법을 발전시켜, 줄인 데이터셋과 함께 특권 정보(Privileged Information)를 합성하는 DRUPI( 데이터셋 축소를 위한 특권 정보 활용) 기법을 제안합니다. 이 방법은 모델 학습을 개선하기 위해 추가 학습 대상을 도입합니다.

- **Technical Details**: DRUPI는 기존의 데이터-레이블 쌍 외에 특징 레이블(feature labels) 또는 주의 레이블(attention labels)과 같은 특권 정보를 추가로 합성하여 보조적인 감독(supervision)을 제공합니다. 효과적인 특징 레이블은 지나치게 차별적이지도, 지나치게 다양하지도 않아야 하며, 적절한 수준에서 균형을 이뤄야 합니다.

- **Performance Highlights**: ImageNet, CIFAR-10/100 및 Tiny ImageNet에서의 광범위한 실험 결과, DRUPI는 기존의 데이터셋 축소 방법과 매끄럽게 통합되며, 성능 향상을 가져옵니다. 예를 들어, CIFAR10에서 Herding 방법에 DRUPI를 적용하면 성능이 24.3% 향상되며, K-center 방법에서는 최대 23.4%의 개선 효과를 보여줍니다.



### Towards Model Discovery Using Domain Decomposition and PINNs (https://arxiv.org/abs/2410.01599)
- **What's New**: 이번 연구는 일반적인 미분방정식(ODE)으로 표현된 복잡한 시스템에서 모델 매개변수를 학습하기 위한 기계 학습 알고리즘을 개선하는 데 중점을 두고 있습니다. 특히, Physics-Informed Neural Networks (PINNs)와 Finite Basis Physics-Informed Neural Networks (FBPINNs) 두 가지 접근 방식을 평가하여, 제한된 데이터에서의 모델 동역학 학습 성능을 비교했습니다.

- **Technical Details**: 본 연구에서는 생물학적 과정의 수학적 모델링의 복잡성을 해결하기 위해 두 가지 toy 모델(포화 성장 모델 및 경쟁 모델)을 사용했습니다. PINNs와 도메인 분해 기반의 FBPINNs 접근법을 이용하여, 시간이 제한된 경우의 매개변수를 학습하는 능력을 비교했습니다. PINNs는 라벨링된 훈련 데이터와 문제에 대한 사전 지식을 결합하여 훈련됩니다. FBPINNs 않는 도메인 분해 아이디어를 포함하고, ODE 문제에 대한 응용을 탐구합니다.

- **Performance Highlights**: 연구 결과, FBPINN 방식이 vanilla PINN 방법에 비해 괄목할 만한 성과를 보였으며, 제한된 동적 데이터로만 구성된 경우에도 FBPINN의 성능이 더 우수한 것으로 나타났습니다. 이는 생물학적 문제에 대한 매개변수 추정 분야에서 도메인 분해 접근법의 새롭고 효과적인 적용을 보여줍니다.



### SAFE: Semantic Adaptive Feature Extraction with Rate Control for 6G Wireless Communications (https://arxiv.org/abs/2410.01597)
- **What's New**: 본 논문에서는 채널 조건에 따라 다양한 서브 의미 조합을 선택할 수 있는 혁신적인 Semantic Adaptive Feature Extraction (SAFE) 프레임워크를 제안하여 대역폭 효율성을 크게 향상시킵니다.

- **Technical Details**: SAFE 프레임워크는 이미지를 서브 의미로 분해하고 각 서브 의미를 다양한 채널을 통해 전송하여 채널 용량의 제한으로 인해 발생하는 문제를 효과적으로 완화합니다. 또한, 세 가지 고급 학습 알고리즘이 SAFE 프레임워크의 성능 최적화를 위해 도입되었습니다.

- **Performance Highlights**: 일련의 시뮬레이션 실험을 통해 SAFE 프레임워크가 서로 다른 채널 대역폭 조건에서 의미를 효과적으로 및 적응적으로 추출하고 전송할 수 있음을 보여주었습니다. 제안된 프레임워크는 무선 이미지 전송의 대역폭 효율성을 개선하고 다양한 통신 채널 모델에 대한 적응성을 입증하였습니다.



### Coordinate-Based Neural Representation Enabling Zero-Shot Learning for 3D Multiparametric Quantitative MRI (https://arxiv.org/abs/2410.01577)
- **What's New**: 본 연구에서는 혁신적인 이미징 방법론인 SUMMIT(SimUltaneous MultiparaMetric quantitative MRI via Implicit neural represenTation)를 제안하고 있습니다. SUMMIT는 3D 다중 매개변수 qMRI를 위한 데이터 수집과 비지도 복원을 포함한 새로운 기술입니다.

- **Technical Details**: SUMMIT는 여러 중요한 양적 물성(parametric properties)을 높은 언샘플링된 k-space에 인코딩하며, 이를 통해 물리 모델과 결합된 암묵적 신경 표현(Implicit Neural Representation)을 활용하여 외부 학습 데이터 세트 없이 원하는 멀티파라메트릭 맵을 복원합니다. 또한, SUMMIT는 T1, T2, T2*, 및 양적 자기 감수성 맵(QSM)을 동시 복원하는 기능을 제공합니다.

- **Performance Highlights**: SUMMIT는 기존 방법에 비해 11.0%, 11.8%, 4.8%의 양적 오차를 줄이며, 잘 제어된 수집 시간 내에 여섯 가지의 고해상도 양적 MR 이미지를 제공합니다. 이로써 SUMMIT는 단일 데이터 수집에서 가장 포괄적인 qMRI 복원을 실현하였습니다.



### Fake It Until You Break It: On the Adversarial Robustness of AI-generated Image Detectors (https://arxiv.org/abs/2410.01574)
- **What's New**: 이 논문은 인공지능(AI)으로 생성된 이미지(AIGI) 탐지기의 최신 기술을 다양한 공격 시나리오에서 평가하며, 현재의 포렌식(Classifiers) 기술이 공격 받을 수 있는 현실적인 설정에서의 취약성을 밝힙니다.

- **Technical Details**: 우리는 포렌식 탐지기와 관련된 두 가지 공격 시나리오(white-box 공격과 black-box 공격)를 연구하였으며, 소셜 미디어와 같은 현실적인 환경에서 진행되는 후처리 과정을 반영하여 실험하였습니다. 최신 탐지기(CNN 및 CLIP 기반)들을 평가하며, 이러한 시스템들이 실제 공격에 얼마나 취약한지를 분석하였습니다.

- **Performance Highlights**: 최신 포렌식 분류기는 공격자의 개입이 없을 때에도 탐지 정확도가 0%까지 감소할 수 있으며, 실제 상황에서 50% 이상의 공격 성공률을 기록했습니다. 이를 통해 현재의 탐지 기술이 갖는 위험을 실질적으로 평가하였고, CLIP 기반 탐지기에 대한 효과적인 방어 메커니즘을 제안하였습니다.



### HRTF Estimation using a Score-based Prior (https://arxiv.org/abs/2410.01562)
- **What's New**: 이번 연구에서는 데이터 기반 사전 정보(score-based prior)를 활용한 새로운 머리 관련 전달 함수(HRTF) 추정 방법을 소개합니다.

- **Technical Details**: 이 방법은 자연 자극 신호(예: 인간의 목소리)를 사용하여 잔향이 있는 환경에서 HRTF를 추정하며, 방의 임펄스 응답(impulse response)도 함께 추정합니다. 이를 위해 방 음향학(room acoustics)의 통계적 특성을 기반으로 하는 파라메트릭 모델을 최적화합니다. HRTF의 사후 분포(posterior distribution)는 점화 측정(reverberant measurement)과 자극 신호(excitation signal)를 고려하여 모델링됩니다.

- **Performance Highlights**: 제안된 방법은 최적의 HRTF를 제안하는 오라클 추천 시스템(oracle recommender system)보다도 성능이 우수하며, 특히 고주파 콘텐츠(high-frequency content)의 큰 변동성을 잘 설명할 수 있음을 보여줍니다.



### OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data (https://arxiv.org/abs/2410.01560)
- **What's New**: 이 논문에서는 수학적 추론(mathematical reasoning)을 위한 고품질의 finetuning (SFT) 데이터셋을 생성하기 위한 연구를 다루고 있습니다. 특히, OpenMathInstruct-2 데이터셋을 통해 기존의 공개 수학 추론 데이터셋보다 약 8배 더 큰 규모로 1,400만 개의 질문-답변 쌍을 제공합니다.

- **Technical Details**: 연구에서 	exttt{Llama3.1} 모델을 사용하여 데이터 합성(data synthesis)에 대한 철저한 ablation 실험을 진행하였으며, 주요 발견 사항으로는 (a) 솔루션 형식(solution format)의 중요성, (b) 강력한 teacher 모델이 생성한 데이터가 약한 student 모델의 on-policy 데이터보다 우수함, (c) 저품질 솔루션에 강한 SFT의 내구성, (d) 질문 다양성(question diversity)의 중요성을 제시했습니다.

- **Performance Highlights**: OpenMathInstruct-2로 	exttt{Llama-3.1-8B-Base} 모델을 finetuning한 결과, 	exttt{Llama3.1-8B-Instruct} 모델보다 MATH에서 절대 15.9% 향상된 성능(51.9% → 67.8%)을 보였습니다. 또한, 본 연구의 모형 및 코드와 OpenMathInstruct-2 데이터셋을 상업적 허용 라이센스 하에 공개하여 오픈소스 활동을 가속화하고자 했습니다.



### Integrative Decoding: Improve Factuality via Implicit Self-consistency (https://arxiv.org/abs/2410.01556)
- **What's New**: 이 논문은 Integrative Decoding (ID)라는 새로운 디코딩 전략을 소개하여 오픈-엔디드 생성(task)에서의 자기 일관성(self-consistency) 활용 가능성을 높입니다. 즉, 자기 일관성을 이용해 대규모 언어 모델의 사실 정확도를 향상시키는 방법을 제시합니다.

- **Technical Details**: ID는 각 샘플링된 응답을 원본 프롬프트와 연결하여 새로운 입력을 구성하고, 이 입력을 동시에 처리하여 최종 토큰을 선택합니다. 이 과정에서 각 입력은 샘플링된 응답의 '대표' 역할을 하며, 언어 모델의 다양한 응답 간의 일관성을 집계합니다. 기존 방법은 엄격한 형식 제약이 있는 반면, ID는 형식에 대한 제약이 없고 추가적인 프롬프트가 필요하지 않아서 더 넓은 범위에 적용 가능합니다.

- **Performance Highlights**: ID는 TruthfulQA (+11.2%), Biographies (+15.4%), LongFact (+8.5%) 벤치마크에서 사실성을 일관되게 높이며, 샘플링된 응답 수가 증가함에 따라 성능 향상이 점진적으로 증대하여 반복 샘플링에 대한 확장 가능성을 보여줍니다.



### Edge-preserving noise for diffusion models (https://arxiv.org/abs/2410.01540)
- **What's New**: 이 논문에서는 기존의 균일한 가우시안 노이즈를 사용하는 생성적 확산 모델의 한계를 극복하기 위해, 에지를 보존하는 새로운 확산 모델을 제안합니다. 이 모델은 이미지 처리에서 오래된 비등방 확산(anisotropic diffusion) 기법에서 영감을 받아 구조적 정보를 더욱 잘 반영합니다.

- **Technical Details**: 제안된 모델은 에지 보존 노이즈(Edge-Preserving Noise) 스케줄러를 사용하여 에지를 보존하는 노이즈와 비등방 Gaussian 노이즈 간의 변화를 통해 생성 과정의 수렴 속도를 높입니다. 이를 통해 저주파 및 중주파(low-to-mid frequencies) 정보를 더 효율적으로 학습할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 무조건적인 이미지 생성에서 기존 최첨단 모델들보다 일관되게 우수한 성능을 보이며, 특히 픽셀 공간 확산(Pixel Space Diffusion)에서 FID 점수가 30% 향상되었습니다. 또한, 모양 기반 우선 규칙(shape-based prior)에 기반한 생성 작업에서도 더 뛰어난 품질과 견고성을 나타냅니다.



### Attention layers provably solve single-location regression (https://arxiv.org/abs/2410.01537)
Comments:
          41 pages, 7 figures

- **What's New**: 이 논문에서는 Attention 기반 모델의 이론적 이해를 추구하는 새로운 과제로 'single-location regression'을 소개합니다. 이 과제는 입력된 토큰 시퀀스에서 오직 하나의 토큰이 예측을 결정하며, 이 토큰의 위치는 잠재적(random variable) 변수로 설정되어 있다는 점에서 독창적입니다.

- **Technical Details**: 단일 위치 회귀(single-location regression)는 독립적인 랜덤 토큰(예: X1,..., XL)의 시퀀스를 입력으로 하고, 출력 Y∈ℝ은 잠재적 변수 J0에 의해 결정됩니다. 제안된 예측기는 비선형 자기주의(self-attention) 층의 단순화된 버전으로, 이방식의 예측기는 Asymptotic Bayes Optimal을 달성합니다. 이는 복잡한 비볼록(non-convex) 문제에도 불구하고 유효한 구조 학습이 가능함을 보여줍니다.

- **Performance Highlights**: 제안된 예측기는 단일 위치 회귀 문제를 해결할 수 있는 능력을 효과적으로 보여줍니다. 특히, 이 방법은 기존의 표준 선형 회귀기법보다 우수한 성능을 발휘하며, Attention 메커니즘이 희소한 토큰 정보와 내부 선형 구조를 처리할 수 있는 역량을 강조합니다.



### LEGO: Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion (https://arxiv.org/abs/2410.01506)
Comments:
          Research paper

- **What's New**: 이 논문에서는 컴퓨터 비전 작업에서 서로 다른 표현, 도메인, 모드(모달리티)의 특징(feature)을 효과적으로 결합하는 새로운 접근법을 제안합니다. 저자들은 고차원 특징 공간에서 저차원 해석 가능한 그래프 공간으로 전환하여 다양한 수준의 특징 관계를 인코딩하는 유사성 그래프(similarity graphs)를 구성하였습니다.

- **Technical Details**: 제안된 방법은 그래프 제곱(graph power) 확장을 활용하고, 이러한 그래프 파워를 결합하기 위해 학습 가능한 그래프 융합 연산자(learnable graph fusion operator)를 도입합니다. 이 방법은 관계 중심(relationship-centric)으로 작동하며, 수학적으로 원리에 부합하며, 멀티선형 다항식(multilinear polynomials)을 통해 원소별 유사도 점수(element-wise similarity score)를 집계하는 방식으로 유사합니다.

- **Performance Highlights**: 이 논문에서 제안하는 그래프 기반 융합 방법은 비디오 이상 탐지(video anomaly detection) 작업에서 강력한 성능을 보여주며, 다중 표현(multi-representational), 다중 모달(multi-modal), 다중 도메인(multi-domain) 특징 융합 작업에서 효과적임을 입증하였습니다.



### DLP-LoRA: Efficient Task-Specific LoRA Fusion with a Dynamic, Lightweight Plugin for Large Language Models (https://arxiv.org/abs/2410.01497)
Comments:
          Preprint under review, 18 pages, 7 figures

- **What's New**: DLP-LoRA는 문장 수준에서 여러 LoRA를 동적으로 융합하기 위해 mini-MLP 모듈을 사용하는 Dynamic Lightweight Plugin이며, 이는 효율성을 크게 향상시킨다.

- **Technical Details**: DLP-LoRA는 약 5M의 파라미터를 가진 mini-MLP로 구성되어 있으며, top-p 샘플링 전략을 통해 다중 LoRA를 동적으로 융합한다. 이를 통해 기존의 token-level 방식보다 더 빠른 추론 성능을 제공한다.

- **Performance Highlights**: DLP-LoRA는 26개 작업에서 평균 92.34%의 정확도를 기록했고, 9개의 질문-답변(QA) 데이터셋에서는 BLEU 및 ROUGE 점수가 크게 향상되었다. 특히, MCQ와 QA 작업에서 각각 92.95%와 13.2%의 상대적인 개선을 보였다.



### One Wave to Explain Them All: A Unifying Perspective on Post-hoc Explainability (https://arxiv.org/abs/2410.01482)
Comments:
          main: 10 pages, appendix: 14 pages, 5 Tables, 25 Figures

- **What's New**: 본 논문은 Wavelet Attribution Method (WAM)을 제안하여, 기존의 기울기 기반 특성 할당 방법을 변동 영역(wavelet domain)으로 확장함으로써 이미지, 오디오, 3D 형상에 대한 통합적인 설명 프레임워크를 제공합니다.

- **Technical Details**: WAM은 입력 신호의 wavelet 분해에 따른 분류 모델 예측의 기울기를 계산하여 생성된 설명을 부드럽게 만들어, 모델의 의사 결정 과정에 대한 더 깊은 통찰을 제공합니다. 기존 방법인 SmoothGrad, Integrated Gradients와의 통합을 통해, WAM은 연속 공간에서 정의된 모든 유형의 모달리티에 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, WAM은 다양한 이미지, 오디오 및 3D 설명 가능성에서 기존의 최첨단 방법들과 비교하여 충실도(metrics) 측면에서 동등하거나 뛰어난 성능을 보임을 보여주었습니다.



### Introducing Flexible Monotone Multiple Choice Item Response Theory Models and Bit Scales (https://arxiv.org/abs/2410.01480)
- **What's New**: 이 연구에서는 다중 선택 데이터용으로 새로운 모델인 단조 다중 선택(MMC, Monotone Multiple Choice) 모델을 제안하고, 이를 자동 인코더(autoencoder)를 사용하여 적합화하였습니다. MMC 모델은 기존의 명목 반응 IRT 모델보다 데이터 적합성이 높음을 실험적으로 증명하였습니다.

- **Technical Details**: MMC 모델은 각 항목에 대한 정답 및 오답(딕스트렉터, distractor) 선택 확률을 동시에 모델링합니다. 이 방법을 통해 시험 문제의 품질을 평가할 수 있고, 더 정확한 잠재 특성 추정이 가능해집니다. 또한, 본 연구에서는 비율 척도(ratio scale)인 비트 척도(bit scale)를 도입하여 잠재 특성 척도를 변환할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 스웨덴 학력적성검사(Swedish Scholastic Aptitude Test)의 실 데이터와 시뮬레이션을 활용한 결과, MMC 모델이 기존의 IRT 모델보다 뛰어난 데이터 적합성을 보여주며, 잠재 특성 평가의 해석이 용이함을 강조합니다. 비트 척도는 다양한 모델 간 비교를 가능하게 하고, 유연한 모델의 이점을 제공합니다.



### Flow Matching for Accelerated Simulation of Atomic Transport in Materials (https://arxiv.org/abs/2410.01464)
- **What's New**: LiFlow는 결정 재료에 대한 분자 동역학(Molecular Dynamics, MD) 시뮬레이션을 가속화하기 위해 개발된 생성 프레임워크입니다. 이 모델은 원자 변위를 조건부 생성으로 설정하는 과제를 제시합니다.

- **Technical Details**: LiFlow는 flow matching을 사용하며, Propagator 서브모델을 통해 원자 변위를 생성하고, Corrector를 통해 비물리적인 기하학을 지역적으로 수정합니다. Maxwell-Boltzmann 분포를 기반으로 하는 적응형 prior를 통합하여 화학 및 열 조건을 고려합니다.

- **Performance Highlights**: LiFlow는 4,186개의 고체 전해질(Solid-State Electrolyte, SSE) 후보들에 대한 25-ps 궤적 데이터셋을 기준으로 평가되었으며, 예측된 리튬 평균 제곱 변위(Mean Squared Displacement, MSD)에서 0.7-0.8의 일관된 Spearman 순위 상관관계를 달성했습니다. 또한, LiFlow는 짧은 훈련 궤적에서 더 큰 슈퍼셀(Supercell) 및 긴 시뮬레이션으로 일반화하면서 높은 정확도를 유지합니다. 최초의 원리에 비해 최대 600,000배의 속도 향상을 이루어내어, 훨씬 더 큰 길이 및 시간 스케일에서의 시뮬레이션을 가능하게 합니다.



### From Reward Shaping to Q-Shaping: Achieving Unbiased Learning with LLM-Guided Knowledg (https://arxiv.org/abs/2410.01458)
Comments:
          q-shaping, reinforcement learning, reward shaping

- **What's New**: 본 연구에서는 Q-value 초기화를 확장한 Q-shaping을 소개하며, 이를 통해 도메인 지식을 활용하여 에이전트 훈련을 가속화하고 샘플 효율성을 개선하는 방법을 제안합니다. Q-shaping은 기존의 보상 조정 방법과 달리 Q-value를 직접 수정하여 에이전트의 최적성을 보장합니다.

- **Technical Details**: Q-shaping은 다양한 작업에 대해 일반적이고 강력한 접근 방식을 제공하며, 대형 언어 모델(LLM)을 활용하여 휴리스틱 제공자로 사용합니다. 실험을 통해 Q-shaping이 가장 좋은 기준보다 평균 16.87%의 샘플 효율성 향상을 달성하였고, LLM 기반 보상 조정 방법과 비교하여 253.80%의 성능 향상을 보였습니다.

- **Performance Highlights**: Q-shaping은 20개의 서로 다른 환경에서 평가되었으며, 각 환경 관련 최상의 기준보다 16.87% 향상된 결과를 기록했습니다. 또한 기존 LLM 기반 보상 조정 방식인 T2R 및 Eureka와 비교하여 최적성에서 253.80%의 성능 손실을 경험하는 결과를 보여 주었습니다.



### Geometric Signatures of Compositionality Across a Language Model's Lifetim (https://arxiv.org/abs/2410.01444)
Comments:
          Under review as a conference paper at ICLR 2025

- **What's New**: 이 연구는 인공 언어 모델의 조합 일반화(compositional generalization) 능력과 관련하여, 표현의 의미가 어떻게 구성되어 있고, 이러한 능력의 기저에 있는 표현 메커니즘을 분석합니다. 처음으로 데이터 세트의 조합성 정도가 LM의 표현의 내재적 차원(intrinsic dimensionality)에 어떻게 반영되는지를 조사합니다.

- **Technical Details**: 연구는 조합의 두 가지 유형, 즉 형태의 조합(compositionality of form)과 의미의 조합(compositionality of meaning)을 구별하며, 비선형 및 선형 차원으로 두 가지 차원성을 측정합니다. LMs은 전달할 프레이즈의 통계적 규칙을 학습하고, 그 과정에서 조합적 복잡성(combinatorial complexity)을 인코딩합니다. 이들은 언어 처리에서 형식과 의미의 차이를 구분할 수 있게 해줍니다.

- **Performance Highlights**: 연구에서는 비선형 ID가 의미 조합성을 인코딩하고 선형 차원이 형태 조합성과 높은 상관관계를 갖는다는 것을 발견했습니다. 이는 LMs가 어떻게 언어를 처리하는 방식에서 의미와 형태가 관련이 있음을 시사합니다. 결과적으로 비선형과 선형 표현의 복잡성이 언어 모델의 훈련 과정에서 어떻게 달라지는지를 보여주었습니다.



### Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling (https://arxiv.org/abs/2410.01440)
- **What's New**: 이 논문은 자율 로봇 작업 계획의 효율성을 높이기 위해 Self-Refinement 접근 방식을 제안합니다. 기존의 LLM(agent) 플래너가 가진 한계를 극복하기 위한 새로운 방법으로, 반복적으로 초기 계획을 개선하여 균형점에 도달하는 과정을 설명합니다.

- **Technical Details**: Self-refining 프로세스는 고정점 문제(fixed-point problem)로 정의되며, 이를 통해 환경으로부터의 피드백을 포함한 Nested Equilibrium Sequence Modeling 절차를 구현합니다. 이 방식은 LLM이 과거 계획과 피드백을 기반으로 효율적으로 작동할 수 있도록 도와줍니다. 또한, 분석 기법을 사용하여 supervised learning 방식으로 LLM 플래너를 훈련할 수 있습니다.

- **Performance Highlights**: VirtualHome-Env 벤치마크에서 본 방법의 성능을 평가한 결과, 기존 방법들에 비해 더 뛰어난 성능과 효율적인 inference 계산 스케일링을 보여주었습니다.



### Approximation by Steklov Neural Network Operators (https://arxiv.org/abs/2410.01426)
- **What's New**: 이 연구에서는 Steklov Neural Network operators (SNNOs)의 새로운 가족을 구성하고, Steklov type integral을 활용하여 다양한 수렴 정리를 제시합니다. 이를 통해 Neural Network operators의 응용 범위를 확장하고, 더욱 정교한 수렴 특성을 입증합니다.

- **Technical Details**: Steklov Neural Network operators는 주어진 함수 f:[a,b]→ℝ에 대해 Steklov 적분을 사용하여 구성됩니다. 이 삽입 신경망은 pointwise 및 uniform 수렴에 대한 이론을 제공하며, r차 매끄러움의 모듈을 기반으로 한 수렴 비율을 연구합니다. 이는 기존 Neural Network의 대표적인 형태인 sigmoidal 함수와 함께 사용됩니다.

- **Performance Highlights**: 이 논문에서 소개된 SNNOs는 이론적으로 점근적 행동 및 정량적 추정치를 제시하며, 다양한 연속 함수의 근사를 위한 강력한 도구로 작용합니다. 실질적인 응용에 있어 Neural Network의 효과성을 높이며, 특히 추상적인 함수 근사 및 예측에서 높은 성능을 발휘합니다.



### The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs (https://arxiv.org/abs/2410.01417)
- **What's New**: 새로운 다중 모달 대형 언어 모델(MLLMs) 벤치마크 제안. 인간의 기본적인 연관 능력을 평가하기 위한 평가 기준 및 주목받지 못한 연관 임무 개발.

- **Technical Details**: 연관 작업을 정의하고 자연어 데이터셋을 활용한 annotation-free (주석 없는) 방법으로 연관 벤치마크 구축. 세 가지 수준의 연관 작업(단일 단계, 동기식, 비동기식) 설정.

- **Performance Highlights**: 현재 공개 소스 MLLMs는 연관 작업에서 인간과 비교해 일관되게 낮은 성능을 보임. 최고 성능의 닫힌 소스 모델도 인간 성능과 큰 차이 존재.



### Overpredictive Signal Analytics in Federated Learning: Algorithms and Analysis (https://arxiv.org/abs/2410.01399)
- **What's New**: 이번 논문에서는 분산 데이터 샘플의 민감한 개인 정보를 고려하는 동시에, 통신 비용을 최소화하고 샘플링 속도 및 신호 근사 오차 간의 tradeoff를 분석하는 방안을 제안합니다. 이러한 접근법은 federated learning (FL) 아키텍처를 활용하여 클라이언트 장치에서 신호 근사를 수행하며, 이를 통해 얻은 신호 데이터를 서버로 전달하는 방식을 채택합니다.

- **Technical Details**: 주요 기술적 기여는 다음과 같습니다: 1) 신호 표현을 위한 Fourier 기초를 사용하여 오버프레딕션 제약 조건을 기반으로 한 신호 근사를 고려합니다. 2) 집합 신호의 경험적 누적 분포 함수(CDF)를 사용하여 글로벌 신호 분석을 학습하는 집계 절차를 개발합니다. 3) 신호 샘플링의 영향을 고려하여 실제 신호 CDF와 Glivenko-Cantelli CDF 추정치 간의 점별 차이에 대한 수학적 상한을 도출합니다.

- **Performance Highlights**: 이 연구의 알골리즘을 통해 제안된 분산 알고리즘은 공개적으로 이용 가능한 주택 에너지 소비 데이터셋을 사용하여 성능을 검증하였으며, 효과적인 신호 근사 및 통신 비용 절감의 좋은 예시를 제공합니다.



### Response Estimation and System Identification of Dynamical Systems via Physics-Informed Neural Networks (https://arxiv.org/abs/2410.01340)
- **What's New**: 이 논문에서는 물리기반 신경망(Physics-Informed Neural Networks, PINNs)을 활용하여 비선형 및 에너지 소산 특성을 포함하는 동적 시스템의 식별과 추정을 다루고 있습니다. 구조 건강 모니터링(Structural Health Monitoring, SHM)에서 불완전한 센서 데이터를 바탕으로 시스템 상태를 추정할 수 있는 가능성을 제시하며, Bayesian 프레임워크에서의 매개변수 추정과 불확실성 정량화를 탐구합니다.

- **Technical Details**: PINNs는 신경망의 손실 함수에 알려진 물리 법칙을 직접 포함시킴으로써, 불확실성이 존재할 때에도 복잡한 현상을 간단히 통합할 수 있는 장점을 가지고 있습니다. 본 연구는 세 가지 주요 응용을 다루며: 1) 제한된 센서를 통한 상태 추정 2) 시스템 응답과 매개변수가 모두 미지인 경우의 동시 상태-매개변수 추정 3) Bayesian 프레임워크 내에서의 매개변수 추정입니다.

- **Performance Highlights**: PINNs는 모든 과제에서 효과적인 도구로 나타났으며, 모델링 오류가 존재하더라도 신뢰할 수 있는 추정을 제공합니다. 그러나 모델링 오류가 매개변수 추정에 미치는 영향은 더 크며, 이는 최적화 과정에서 설정된 모델과 실제 시스템 행동 간의 불일치를 조정해야 하기 때문입니다. PINNs는 동적 시스템 모델링에서 불확실성을 효과적으로 처리할 수 있는 강력한 접근 방식을 제시합니다.



### Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models (https://arxiv.org/abs/2410.01335)
Comments:
          11 main pages, 23 pages total, 9 figures, 5 tables

- **What's New**: 본 논문에서는 언어 특정 데이터가 부족한 비영어 사용자를 위한 대형 언어 모델(LLMs)의 수학적 추론 능력을 이전하는 새로운 모델 병합 방법론을 제안합니다. 이 방법론은 모델 수프의 원칙에 따라 두 개의 '전문가(experts)' 모델을 병합함으로써, 언어와 수학의 능력을 동시에 활용할 수 있게 합니다.

- **Technical Details**: 우리는 동일한 사전 학습된 모델에서 시작하여 각각 영어 수학 데이터와 목표 언어의 일반Instruction 데이터로 '전문가' 모델을 파인 튜닝합니다. 이후 수학 전문가의 최상위 및 최하위 transformer 레이어를 언어 전문가의 레이어로 직접 교체하여 목표 언어에서의 수학 성능을 향상시킵니다. 이 방법은 각 전문가를 파인 튜닝할 때 가장 중요한 파라미터 변화에 대한 해석적 분석에 기반하여 간단하고 비용이 적게 들며 직관적입니다.

- **Performance Highlights**: 병합된 모델은 수학 벤치마크인 MGSM에서 평균적으로 10% 더 나은 성능을 보이며, 스와힐리어, 텔루구어, 벵골어, 일본어 등 4개의 주요 언어에서 성능을 향상시킵니다. 특히, 스와힐리어에서는 혼합된 스와힐리 및 수학 데이터셋을 파인 튜닝한 모델의 성능을 초과하여 뛰어난 결과를 냅니다.



### Fast Summation of Radial Kernels via QMC Slicing (https://arxiv.org/abs/2410.01316)
- **What's New**: 이 논문에서는 커널 합(kernel sum) 계산의 속도를 높이기 위해 slicing과 quasi-Monte Carlo(QMC) 접근 방식을 사용한 새로운 방법을 제안하고 있습니다. 이 방법은 랜덤 프로젝션(random projection)을 활용하여 고차원 데이터 포인트의 도우미 사상(helper mapping)을 구현합니다.

- **Technical Details**: 제안된 방법은 random projections를 통해 1차원의 서브스페이스(subspace)로 데이터를 투영한 후, fast Fourier summation을 사용하여 커널 합을 근사합니다. QMC 방식을 적용하여 구체적인 샘플링 시퀀스를 선정함으로써 slicing 오류를 감소시킵니다.

- **Performance Highlights**: 제안된 QMC-slicing 접근 방식은 기존의 QMC-random Fourier features, orthogonal Fourier features 및 비-QMC slicing 방법들에 비해 표준 데이터셋에서 상당한 성능 향상을 보여줍니다. 이 접근법은 커널의 차원이 증가해도 효율적인 연산이 가능하다는 장점이 있습니다.



### Getting Free Bits Back from Rotational Symmetries in LLMs (https://arxiv.org/abs/2410.01309)
Comments:
          14 pages, 3 figures

- **What's New**: 본 논문에서는 현재의 신경망 가중치 압축 방법들이 내재된 대칭성을 간과하고 있어 중복 정보를 인코딩하는 데 낭비가 발생하고 있음을 지적합니다. 이를 해결하기 위해, 회전 대칭(rotationally symmetric) Transformer 가중치를 보다 효율적으로 저장할 수 있는 bits-back coding 형식을 제안합니다.

- **Technical Details**: 제안된 방법은 기존의 배열 배열(array layout) 방식을 사용하지 않고, 동일한 부동소수점 정밀도(floating-point precision)를 유지하면서 가중치를 저장합니다. 연구에서는 SliceGPT에 의해 가지치기(pruned)된 대규모 언어 모델(Large Language Models, LLMs)을 평가하여 3-5%의 총 비트 사용량(bit usage) 감소를 달성했습니다.

- **Performance Highlights**: 모델 성능에 영향을 미치지 않으면서 다양한 모델 크기와 아키텍처에서 비트 사용량이 자유롭게 감소하였습니다.



### Revisiting Hierarchical Text Classification: Inference and Metrics (https://arxiv.org/abs/2410.01305)
Comments:
          Accepted at CoNLL 2024

- **What's New**: 이번 논문에서는 Hierarchical Text Classification (HTC)의 평가 방법에 대해 새로운 접근 방식을 제시합니다. 기존의 다중 레이블 분류(multi-label classification) 문제로 HTC를 다루었던 것과 달리, 연구진은 특정한 계층적 메트릭(hierarchical metrics)에 기반하여 모델을 평가하는 방식을 도입하고, 예측 추론(prediction inference) 방법의 복잡성을 강조합니다. 또한, 새로운 도전적인 데이터셋인 Hierarchical Wikivitals (HWV)를 소개합니다.

- **Technical Details**: HTC는 텍스트에 계층적 구조를 가진 레이블을 할당하는 문제로서, 레이블은 트리(tree)나 유향 비순환 그래프(Directed Acyclic Graph, DAG) 형태로 조직됩니다. 본 논문은 오류의 심각성을 반영할 수 있는 메트릭을 요구하며, 모델 성능을 평가하기 위해 특히 설계된 계층적 메트릭을 사용합니다. 새로운 손실 함수(loss function)를 도입하여 최근 모델과 간단하지만 강력한 기준선(baseline) 모델과 비교하여 성능을 실험했습니다.

- **Performance Highlights**: 실험 결과, 최신 모델들이 반드시 계층 정보를 잘 인코딩하지 못한다는 것을 보여주었으며, HWV 데이터셋에서 우리는 간단한 손실 함수가 최신 모델에 비해 경쟁력을 발휘하는 것을 확인했습니다. 결과적으로, 새로운 방법론을 제안하는 경우 평가 방법론을 신중하게 고려하는 것이 중요하다는 점을 강조하였습니다.



### Towards a Law of Iterated Expectations for Heuristic Estimators (https://arxiv.org/abs/2410.01290)
Comments:
          47 pages, 2 tables, 1 figure

- **What's New**: 본 논문에서는 *heuristic estimator*의 개념을 정립하고, ideal heuristic estimator가 만족해야 하는 특성으로 *iterated estimation*과 *error orthogonality*를 제안합니다.

- **Technical Details**: 이 접근법은 수학적 표현 Y, 정식의 'heuristic argument' π를 입력으로 받아 Y에 대한 추정치를 출력하는 알고리즘인 heuristic estimator \( \mathbb{G}(Y | \pi) \)를 다룹니다. 또한, '정확도(accuracy)' 개념을 도입하여 강력한 추정치를 생성하는 데 있어 직면하는 장벽을 분석합니다.

- **Performance Highlights**: 이 연구는 heuristic estimator가 오류를 예측하지 못해야 한다는 비공식 원칙을 주장하며, 특정 수학적 표현들의 분포에 걸쳐 평균 오류가 제로가 되는 것을 요구합니다.



### Mitigating Copy Bias in In-Context Learning through Neuron Pruning (https://arxiv.org/abs/2410.01288)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 In-Context Learning (ICL)에서 복사 편향(copied bias)을 완화하기 위한 새로운 방법을 제안하였다. 이를 통해 ICL의 오류 문제를 내부 프로세스와 관련된 신경 활성 패턴을 조사하여 해결하는 접근 방식을 취하였다.

- **Technical Details**: 제안된 접근법은 Integrated Gradients(IG) 방법을 사용하여 특정 신경 뉴런을 식별하고 이들 뉴런을 제거(pruning)함으로써 성능을 개선하는 것이다. 이 연구는 Transformer 및 상태 공간 모델(State-Space Models)과 같은 다양한 LLM 아키텍처에서 적용 가능하며, 입력 예제의 모임을 통해 제공되는 처리를 기반으로 작업을 수행하는 방식으로 진행되었다.

- **Performance Highlights**: 제안된 방법은 다양한 ICL 과제를 통해 성능을 향상시키며, 특히 작은 LLM 모델에서 높은 복사 오류율을 개선하는 결과를 보였다. 제거된 뉴런들이 효과적인 작업 인식을 방해했음을 나타내는 작업 벡터(task vectors)의 품질 향상을 통해 이러한 성능 향상을 확인하였다.



### Deep Kernel Posterior Learning under Infinite Variance Prior Weights (https://arxiv.org/abs/2410.01284)
Comments:
          21 pages, 11 figures

- **What's New**: 이번 연구에서는 무한 너비를 가진 심층 베이지안 신경망(Bayesian Neural Network, BNN)에서 대응되는 α-stable 커널 프로세스를 개발하였습니다. 네트워크 각 층의 너비가 무한대로 증가할 때, 커널에 인위적으로 노이즈를 추가하는 기존의 접근 방식을 넘어서, 자연스럽게 조건부 가우시안 표현이 이루어지는 과정을 제시했습니다.

- **Technical Details**: 연구진은 무한 분산을 가진 타원형 분포의 네트워크 가중치를 가진 베이지안 심층 신경망이 각 층에서 α-stable 마지널(marginal)을 갖는 프로세스로 수렴함을 입증했습니다. 이 과정은 Cho와 Saul(2009)의 접근법을 사용하여 층별 상관 커널(covariance kernel)을 재귀적으로 연결할 수 있습니다. 또한, 기존의 연구 결과를 다층 네트워크에 대한 일반화와 수치적 부담을 개선하여 제안하였습니다.

- **Performance Highlights**: 수치 실험 및 벤치마크 데이터 세트에 대한 시연에서, 기존의 접근 방식에 비해 본 연구의 방법론이 컴퓨터 계산 및 통계적 이점을 명확히 보여주었습니다.



### Uncertainty-aware Human Mobility Modeling and Anomaly Detection (https://arxiv.org/abs/2410.01281)
- **What's New**: 이번 논문에서는 레이블이 없는 데이터를 활용하여 인간의 이동 행동 모델링 및 이상 탐지를 효과적으로 수행할 수 있는 UIFormer라는 새로운 방법론을 제안합니다.

- **Technical Details**: UIFormer는 이중(dual) Transformer 아키텍처를 기반으로 하며, aleatoric(데이터) 불확실성과 epistemic(모델) 불확실성을 모두 고려하여 인간 행동 모델링 및 이상 탐지를 수행합니다. 이 방법은 복잡한 시공간 의존성을 캡처하고 데이터 희소성 문제를 해결하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, UIFormer는 수만 개의 에이전트를 포함하는 대규모 시뮬레이션 데이터셋에서 기존의 예측 및 이상 탐지 기준선보다 뛰어난 성능을 보였습니다.



### CANVAS: Commonsense-Aware Navigation System for Intuitive Human-Robot Interaction (https://arxiv.org/abs/2410.01273)
Comments:
          project page this https URL

- **What's New**: 이번 논문에서는 CANVAS라는 새로운 프레임워크를 소개하여 로봇이 인간의 모호한 지침 (abstract instructions)을 이해하고 내비게이션 경로를 최적화할 수 있도록 돕습니다.

- **Technical Details**: CANVAS는 비주얼 및 리걸 관점의 입력을 통합하여 로봇이 인간의 내비게이션 행동을 모방하는 방식으로 학습하는 이모테이션 러닝 (Imitation Learning)을 기반으로 합니다. COMMAND 데이터셋은 48시간 이상의 드라이빙 데이터를 포함하여 총 219킬로미터를 커버합니다. 이 데이터셋은 고유하게 인간이 주석을 단 내비게이션 결과로 구성되어 있습니다.

- **Performance Highlights**: CANVAS는 ROS NavStack보다 모든 환경에서 67%의 성공률을 기록하며, 특히 과수원 환경에서는 ROS NavStack이 0%의 성공률을 기록한 반면 CANVAS는 67%의 성공률을 보였습니다. 실제 환경에서도 69%의 전반적인 성공률을 달성하며 Sim2Real (시뮬레이션에서 실제) 전이에 우수한 성과를 보입니다.



### "No Matter What You Do!": Mitigating Backdoor Attacks in Graph Neural Networks (https://arxiv.org/abs/2410.01272)
Comments:
          18 pages, 12 figures, 9 tables

- **What's New**: GCleaner는 GNN의 첫 번째 백도어 완화 방법으로, 백도어 학습 절차를 역전시켜 백도어 언러닝을 달성하는 것을 목표로 합니다. 이 방법은 트리거 회복과 백도어 언러닝의 두 가지 주요 모듈로 구성됩니다.

- **Technical Details**: 이 논문에서는 그래프 트리거 회복을 위해 설명 알고리즘을 활용하여 모델에서 트리거의 위치를 최적화하는 방법을 설명합니다. 또한 지식 증류(knowledge distillation)와 그래디언트 기반 설명 가능한 지식을 결합하여 백도어 로직을 지우는 백도어 언러닝 메커니즘을 도입합니다.

- **Performance Highlights**: GCleaner는 1%의 클린 데이터로도 백도어 공격 성공률을 10%로 줄일 수 있으며, 모델 성능의 저하가 거의 없어 기존의 백도어 방어 방법을 크게 초월합니다.



### Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Unveiling AI's Potential Through Tools, Techniques, and Applications (https://arxiv.org/abs/2410.01268)
Comments:
          This book contains 156 pages and 9 figures

- **What's New**: 이번 논문은 빅데이터 분석에서의 딥 러닝(deep learning) 및 머신 러닝(machine learning)에 대한 소개를 제공하며, ChatGPT 및 Claude와 같은 도구, 하드웨어 추천, PyTorch 및 TensorFlow와 같은 라이브러리를 사용하는 개발 환경 설정에 대한 실용적인 가이드를 포함하고 있습니다. 또한 AutoML 및 엣지 컴퓨팅(edge computing)과 같은 AI의 미래에 대한 통찰을 제공합니다.

- **Technical Details**: 이 책은 머신 러닝의 역사와 개념을 시작으로 다양한 산업에서의 응용을 설명합니다. AI 도구들은 자연어 처리(natural language processing, NLP)의 발전으로 인해 직면하는 복잡한 문제들을 풀기 위한 강력한 도구로 자리잡고 있습니다. ChatGPT, Claude, Gemini와 같은 다중 모달 AI 도구들은 데이터 분석, 모델 설계, 코드 생성을 지원하여 연구의 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 딥 러닝과 머신 러닝은 스마트폰의 얼굴 인식, 자율 주행 자동차, 의료 영상 분석 및 금융 서비스 등 다양한 산업에서 활용되고 있습니다. 이 기술들은 전문가들만의 영역에서 벗어나 일반인들도 쉽게 AI 모델을 구축할 수 있도록 자동화 머신 러닝(AutoML)을 통해 민주화되고 있습니다. 앞으로 엣지 컴퓨팅을 통해 AI는 클라우드에서 로컬 장치로 이동하며, AI의 의사결정 과정에서의 투명성과 공정성을 확보하는 연구가 중요해질 것입니다.



### Transformers Handle Endogeneity in In-Context Linear Regression (https://arxiv.org/abs/2410.01265)
Comments:
          30 pages

- **What's New**: 이번 연구에서는 transformers의 가능성을 탐구하며, in-context linear regression에서의 endogeneity 처리 능력을 분석합니다. 특히 transformers가 instrumental variables (IV)을 사용하여 endogeneity를 효과적으로 다룰 수 있는 메커니즘을 본질적으로 가지고 있다는 점이 주목할 만합니다.

- **Technical Details**: 연구에서는 transformer 아키텍처가 gradient 기반 bi-level 최적화 절차를 에뮬레이트(emulate)할 수 있음을 보여주고, 이는 널리 사용되는 두 단계 최소 제곱법(2SLS) 솔루션으로 수렴(converge)하는 것을 입증합니다. 또한, in-context pretraining 방식을 제안하고, 이론적으로 글로벌 최소값(global minimizer)이 작은 초과 손실(small excess loss)을 달성한다는 보장을 제공하였습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 훈련된 transformer는 endogeneity가 있는 상황에서도 2SLS 방법에 비해 더 강력하고 신뢰할 수 있는 in-context 예측 및 계수 추정치를 제공합니다.



### Aggregation of Multi Diffusion Models for Enhancing Learned Representations (https://arxiv.org/abs/2410.01262)
- **What's New**: 이번 논문은 다중 확산 모델(Aggregation of Multi Diffusion Models, AMDM) 알고리즘을 소개합니다. 이 알고리즘은 여러 개의 확산 모델로부터 특징을 합성하여 특정 모델의 표현을 풍부하게 하여 보다 세밀한 조정이 가능하게 합니다.

- **Technical Details**: AMDM은 구형 집합(spherical aggregation)과 다양체 최적화(manifold optimization)라는 두 가지 주요 구성 요소로 구성됩니다. 구형 집합은 다양한 확산 모델로부터의 중간 변수들을 최소한의 다양체 편차로 결합하고, 다양체 최적화는 이러한 변수들을 중간 데이터 다양체와 정렬하도록 개선하여 샘플링 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, AMDM은 추가적인 교육(training)이나 추론(inference) 시간 없이 세밀한 제어를 크게 향상시킵니다. 또한, 확산 모델은 초기 단계에서 위치, 속성 및 스타일과 같은 특징에 초점을 맞추고, 후속 단계에서는 생성 품질과 일관성을 개선함을 밝혀냅니다.



### Revisiting Optimism and Model Complexity in the Wake of Overparameterized Machine Learning (https://arxiv.org/abs/2410.01259)
Comments:
          59 pages, 17 figures

- **What's New**: 이 논문에서는 기존의 통계학적 개념인 효과적인 자유도(degrees of freedom)를 재정의하고 확장하는 방법을 제안합니다. 특히, 전통적으로 사용되던 고정-X(fixed-X) 예측 오차(prediction error)와는 다르게, 무작위-X(random-X) 설정을 통해 모델 복잡성과 일반화 성능을 재조명합니다.

- **Technical Details**: 고정-X 예측 오차는 동일한 비무작위 공변량(covariate) 포인트를 사용하여 예측 오차를 평균내는 반면, 무작위-X 예측 오차는 공변량 분포에서 새로운 무작위 샘플을 통해 예측 오차를 평균내는 개념입니다. 이는 현대 기계 학습 문제에 더 적합한 방식으로, 데이터에 대한 높은 복잡도의 모델이라도 적절한 조건에서 좋은 일반화 성능을 나타낼 수 있다는 점을 강조합니다.

- **Performance Highlights**: 이 논문에서 제안하는 복잡성 측정의 유용성을 개념적 논의, 이론 및 실험을 통해 입증하며, 다양한 예측 모델을 해석하고 비교하는 데 도움을 줄 수 있음을 보여줍니다.



### Resource-efficient equivariant quantum convolutional neural networks (https://arxiv.org/abs/2410.01252)
Comments:
          20 pages, 7 figures, 1 table

- **What's New**: 이 연구는 이전의 sp-QCNN 모델이 다루지 않았던 일반 대칭을 처리할 수 있는 자원 효율적인 equivariant split-parallelizing QCNN (sp-QCNN) 모델을 제안합니다.

- **Technical Details**: 이 모델은 풀링 레이어에서 회로를 분할하여 대칭을 보존하며, 그 결과로 QCNN의 병렬화(parallelization)가 가능해집니다. 이를 통해 예상 값(expectation value) 추정에서 측정 효율성을 크게 향상시킬 수 있습니다. 또한, 우리의 모델은 높은 학습 가능성과 일반화 성능을 보여줍니다.

- **Performance Highlights**: 수치 실험 결과, equivariant sp-QCNN은 노이즈가 있는 양자 데이터 분류 작업에서 기존의 equivariant QCNN보다 적은 측정 자원으로 훈련되고 일반화될 수 있음을 보여주었습니다.



### Equivariant score-based generative models provably learn distributions with symmetries efficiently (https://arxiv.org/abs/2410.01244)
- **What's New**: 이 연구는 군 대칭(group symmetry)에 대해 불변인 분포를 학습하기 위한 score-based generative models (SGMs) 의 이론적 분석과 보장을 첫 번째로 제공하며, 데이터 증강(data augmentation)과 동치적 귀납적 편향(equivariant inductive bias) 사이의 최초의 정량적 비교를 제시합니다.

- **Technical Details**: 기존 연구인 Wasserstein-1 ($\mathbf{d}_1$) 보장과 관련한 최근 결과를 바탕으로, 군 대칭이 있는 데이터 분포에 대해 SGMs의 일반화 경계를 개선한 결과를 도출했습니다. 또한 Hamilton-Jacobi-Bellman 이론을 이용하여 동치적 SGM의 귀납적 편향을 설명하고, 데이터 증강 없이도 동치적 벡터 필드를 사용하여 대칭화된 분포의 score를 학습할 수 있음을 입증했습니다. 이러한 결과는 데이터베이스의 증강 없이도 효율적인 모델링이 가능함을 시사합니다.

- **Performance Highlights**: 수치 시뮬레이션 결과는 우리의 분석을 뒷받침하고 데이터 증강이 동치적 벡터 필드의 역할을 대체할 수 없음을 강조합니다. 메트릭 불확실성 확률 모델 관점에서, 동치적 SGM의 오류 원인 네 가지를 규명하고 이를 통해 모델 일반화 오류를 제한하는 경계를 제시합니다.



### ConServe: Harvesting GPUs for Low-Latency and High-Throughput Large Language Model Serving (https://arxiv.org/abs/2410.01228)
- **What's New**: 이 논문에서는 온라인과 오프라인 LLM(대형 언어 모델) 추론을 동시에 처리할 수 있는 ConServe라는 새로운 시스템을 수립하여 GPU 활용도를 높이는 방법을 제안합니다. 이를 통해 전통적인 자원 할당 방식의 비효율성을 개선하고, 더 낮은 지연(latency)과 더 높은 처리량(throughput)을 달성할 수 있습니다.

- **Technical Details**: ConServe는 온라인 작업이 도착할 때 실행 중인 오프라인 작업을 중단(preempt)할 수 있는 실행 엔진(execution engine), 중단으로 인한 재계산(recomputation) 비용을 최소화하기 위한 점검 메커니즘(incremental checkpointing), 그리고 최적의 GPU 활용도를 위해 오프라인 작업을 동적으로 배치하는 스케줄러(scheduler)를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, ConServe는 Llama-2-7B 모델에서 비슷한 온라인 지연(latency)을 유지하면서 기존 최첨단 LLM 서비스 시스템에 비해 2.35배 높은 처리량을 달성하였으며, 오프라인 작업의 처리량도 크게 증가시켰습니다. 이 성능은 특히 온라인 및 오프라인 작업이 혼합된 실질적인 환경에서 두드러집니다.



### Statistical Taylor Expansion (https://arxiv.org/abs/2410.01223)
Comments:
          75 pages, 55 figures

- **What's New**: 이 논문에서는 전통적인 Taylor 전개에서 입력 변수를 임의 변수로 대체하는 통계적 Taylor 전개 방법론을 제안하며, 이 방법이 결괏값의 경로 의존성 문제를 해결할 수 있음을 주장합니다.

- **Technical Details**: 통계적 Taylor 전장은 입력 변수를 독립적으로 측정된 확률 변수로 간주하며, 결과 평균과 분산을 구하는 방법을 제시합니다. 이 방법은 고차원의 불확실성을 고려할 수 있으며, 가우시안 분포를 가정하는 것이 일반적이나 일반적인 분포로의 확장이 가능합니다. 논문에서는 차수 교차와 같은 고차량 변수 상호작용을 고려하여 결과 편향을 계산하는 방법을 설명합니다.

- **Performance Highlights**: 통계적 Taylor 전개로 구현된 분산 산술(variance arithmetic)은 기존의 수치 산술보다 탁월한 성능을 보이며, 고급 물리 상수의 측정이 필요할 정도로 정확한 계산도 가능하게 합니다. 이는 수치적 안정성 및 특정 알고리즘 오류를 줄이는 데 유리합니다.



### Effective Tuning Strategies for Generalist Robot Manipulation Policies (https://arxiv.org/abs/2410.01220)
- **What's New**: 이번 연구에서는 Generalist Robot Manipulation Policies (GMPs)의 파인튜닝(fine-tuning) 전략에 대한 중요한 요소들을 체계적으로 조사하였습니다. 특히, 액션 스페이스(action space), 폴리시 헤드(policy head), 감독 신호(supervision signal) 및 조정 가능한 파라미터(tunable parameters)의 선택이 성능에 미치는 영향을 파악하는 데 중점을 두었습니다.

- **Technical Details**: 2,500회의 롤아웃을 평가하여 다양한 파인튜닝 기법과 설계 선택에 대한 성능을 비교 분석하였습니다. 저자들은 RLBench라는 널리 사용되는 시뮬레이션 플랫폼을 활용하여 통계적 재현 가능성을 확보하였으며, 기본 모델로는 Octo-Small을 사용했습니다. 이를 통해 액션 토큰(action token)을 사용하여 마지막 동작을 예측하는 Diffusion Policy 헤드와 Linear 헤드를 비교하였습니다.

- **Performance Highlights**: 파인튜닝된 GMP들은 최신 모방 학습 알고리즘인 ACT 및 Diffusion Policies를 저데이터(low-data) 환경에서 모두 초과 성능을 보였습니다. 이번 연구 결과는 향후 연구에 중요한 새로운 기준을 마련하며, GMPs 툴박스(toolbox)에 중요한 기여를 하게 될 것입니다.



### An uncertainty-aware Digital Shadow for underground multimodal CO2 storage monitoring (https://arxiv.org/abs/2410.01218)
- **What's New**: 이번 연구에서는 지질학적 탄소 저장(GCS)에 대한 불확실성을 고려한 디지털 섀도(Digital Shadow)를 설계하고 구현하기 위한 머신 러닝 기반의 데이터 동화(data assimilation) 프레임워크를 소개합니다. 이를 통해 CO2 주입 및 저장 작업의 모니터링을 지원하고자 합니다.

- **Technical Details**: 이 프레임워크는 베이esian 추론(Bayesian inference)을 기반으로 하며, 다중 모드 시계열 데이터에 조건화된 CO2 플룸의 후방 분포를 특성화하는 것을 목표로 합니다. 다중 모드 데이터 통합을 위해 시뮬레이션 기반 추론(Simulation-Based Inference, SBI) 및 앙상블 베이esian 필터링(Ensemble Bayesian Filtering) 기법들을 활용합니다.

- **Performance Highlights**: 컴퓨터 시뮬레이션 연구를 통해, 지층의 투과성(permeability) 필드에 대한 정보 부족이 디지털 섀도의 불확실성 정량화에 통합될 수 있음을 관찰했습니다. 이 연구는 불확실성을 인지한 확장 가능한 디지털 섀도의 개념을 처음으로 증명한 사례로 알려져 있습니다.



### Diverse Expected Improvement (DEI): Diverse Bayesian Optimization of Expensive Computer Simulators (https://arxiv.org/abs/2410.01196)
- **What's New**: 본 논문은 비용이 많이 드는 블랙박스 시뮬레이터의 최적화를 위한 새로운 방법인 Diverse Expected Improvement (DEI)를 제안합니다. 이 방법은 여러 개의 ‘ε-최적’ 솔루션을 찾는 데 중점을 두어, 사용자들이 선택할 수 있는 다양한 솔루션을 제공합니다.

- **Technical Details**: DEI는 Gaussian process surrogate 모델 하에서 닫힌 형태의 acquisition function을 제공하며, 이를 통해 자동 미분을 사용한 효율적인 순차 쿼리가 가능하게 합니다. 이 메서드는 탐색(exploration), 활용(exploitation) 및 다양성(diversity) 간의 새로운 균형을 확립합니다. DEI는 내연기관 제어 및 로버 궤적 최적화와 같은 다양한 응용 분야에 사용될 수 있습니다.

- **Performance Highlights**: DEI는 기존의 방법들과 비교했을 때 여러 수치 실험에서 성능 향상을 보여주었습니다. 특히, DEI는 내연기관 제어와 같은 응용 프로그램에서 뛰어난 결과를 나타내며, 다양한 최적화 문제에 적합한 솔루션을 제공합니다.



### [Re] Network Deconvolution (https://arxiv.org/abs/2410.01189)
Comments:
          12 pages, 5 figures

- **What's New**: 이번 연구는 Ye et al. (2020)에 의해 발표된 "Network Deconvolution"의 결과를 재현하려고 합니다. "network deconvolution" 기법은 CNN의 훈련에서 pixel-wise 및 channel-wise 상관관계를 제거하여 모델 성능을 향상시킬 수 있다고 주장합니다. 본 연구는 원본 논문의 주장을 검증하기 위해 다수의 실험을 수행했습니다.

- **Technical Details**: 연구는 367개의 고유한 실험을 통해 10개의 CNN 아키텍처 및 CIFAR-10과 CIFAR-100 데이터셋을 사용하여 수행되었습니다. 네트워크 디콘볼루션은 BN 레이어를 디콘볼루션 레이어로 대체하여 입력 데이터의 공분산 행렬을 계산하고 이를 역제곱근으로 근사하는 과정을 포함합니다.

- **Performance Highlights**: 실험 결과, 원본 논문에서 제시된 결과와 유사한 경향을 보였으며, 특히 Table 1의 경우 정확도가 약간의 편차를 보였지만 전체적인 경향은 일치하였습니다. Table 2의 모든 14개의 재현된 값은 원본 값과 일치했습니다.



### Using Interleaved Ensemble Unlearning to Keep Backdoors at Bay for Finetuning Vision Transformers (https://arxiv.org/abs/2410.01128)
- **What's New**: 이 논문에서는 Vision Transformers (ViT)의 백도어 공격에 대한 새로운 방어 방법인 Interleaved Ensemble Unlearning (IEU)을 제안합니다. IEU는 두 개의 모델을 사용하여 고신뢰 데이터를 동적으로 제거하는 방법입니다.

- **Technical Details**: IEU는 2단계로 구성됩니다. 1단계에서는 얕은 ViT가 백도어 데이터에 대해 높은 신뢰도를 가지도록 미세 조정됩니다. 2단계에서는 이 얕은 ViT가 주 모델의 학습을 제어하여 잠재적으로 손상된 데이터를 차단하고, 유효한 데이터는 정상적으로 학습하도록 합니다. 데이터가 충분히 수집되면 동적인 비 학습 속도로 백도어 영향을 지웁니다.

- **Performance Highlights**: IEU는 11개의 최신 백도어 공격에 대해 3개의 데이터셋에서 효과적임을 입증하였으며, TinyImageNet과 CIFAR10에서 공격 성공률(Attack Success Rate, ASR)을 각각 33.83%, 31.46% 개선하였습니다. 또한 클린 정확도(Clean Accuracy)도 유지하였습니다.



### High-dimensional logistic regression with missing data: Imputation, regularization, and universality (https://arxiv.org/abs/2410.01093)
- **What's New**: 이 논문은 높은 차원에서 ridge-regularized (릿지 정규화된) 로지스틱 회귀를 연구하며, 공변량(covariates) 값이 누락되거나 가법적 잡음(additive noise)으로 인해 오염되었을 때의 상황을 다룹니다.

- **Technical Details**: 저자들은 공변량과 가법적 오염이 독립적이고 정규 분포(normally distributed)일 때, 예측 오류(prediction error)와 추정 오류(estimation error)를 정확하게 특성화(characterization)합니다. 데이터 행렬의 항목들이 독립성과 모멘트 조건을 만족하는 한 이러한 보장들이 지속적으로 유지됨을 보여줍니다.

- **Performance Highlights**: 저자들은 여러 가지 대체(imputation-based) 전략의 성능을 Bayes optimal procedure와 비교하였으며, 단일 대체(single imputation)와 단순한 형태의 다중 대체(multiple imputation) 간의 분별, 그리고 단일 대체 로지스틱 회귀에 간단한 ridge 정규화 항을 추가함으로써 추정기가 Bayes optimal 예측 오류와 거의 구별되지 않는 성능을 얻을 수 있음을 밝혀냈습니다.



### Exploring Empty Spaces: Human-in-the-Loop Data Augmentation (https://arxiv.org/abs/2410.01088)
- **What's New**: 이번 연구에서는 데이터 증강(data augmentation)의 중요성을 강조하며, 비구조적 텍스트 데이터셋에서의 'unknown unknowns'를 탐색하는 데 도움을 주는 상호작용 도구인 Amplio를 소개합니다.

- **Technical Details**: Amplio는 사용할 수 있는 데이터 증강 기법으로는 Augment With Concepts, Augment by Interpolation, Augment with Large Language Model(LM)이 포함되어 있으며, 이 세 가지 모두 사용자가 새로운, 관련성 높은 데이터 포인트를 생성할 수 있도록 지원합니다.

- **Performance Highlights**: Amplio를 사용한 사용자 연구에서 18명의 전문 레드 팀원들이 유해 LLM 프롬프트 데이터셋을 증강하는 데 성공적으로 활용하였고, 고품질의 다채로운 데이터를 신속하게 생성할 수 있었음을 보여주었습니다.



### An Introduction to Deep Survival Analysis Models for Predicting Time-to-Event Outcomes (https://arxiv.org/abs/2410.01086)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문은 개인 데이터 포인트 수준에서 시간-사건 예측(time-to-event prediction) 문제를 다루며, 최근의 심층 신경망(neural networks)을 활용하여 지속 가능성 분석(survival analysis) 분야의 기초를 현대적으로 소개합니다.

- **Technical Details**: 주요 초점은 Cox 비례위험모델(Cox proportional hazards model)에서 심층 커널 Kaplan-Meier 추정기(deep kernel Kaplan-Meier estimators)와 신경망 미분 방정식(neural ordinary differential equations) 모델에 이르기까지 시간-사건 예측 모델의 설계 패턴을 제공합니다. 또한 경쟁 위험(competing risks)과 동적 설정(dynamic setting)이라는 두 가지 확장을 다룹니다.

- **Performance Highlights**: 모든 모델과 평가 지표는 코드 저장소(code repository)와 함께 공개되어 있어 독자가 쉽게 접근할 수 있으며, 이를 통해 공정성(fairness), 인과 추론(causal reasoning), 해석 가능성(interpretability), 통계적 보장(statistical guarantees) 등 다양한 주제에 대한 논의를 포함하고 있습니다.



### Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Tim (https://arxiv.org/abs/2410.01083)
Comments:
          ECCV 2024

- **What's New**: 이번 연구에서는 기존에 버려지는 activation maps의 활용 가능성을 제시하며, 모델의 예측 성능을 개선할 수 있음을 증명하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 activation map을 효과적으로 검색하고 집계하는 메커니즘을 통해, 모델 성능을 테스트 시간에 향상시킬 수 있도록 설계되었습니다. 이는 전이학습된 Deep Net에서 대체로 활용되지 않는 activation을 기반으로 합니다.

- **Performance Highlights**: 실험 결과, 9개의 서로 다른 아키텍처에서 제안된 방법이 이미지 분류 및 분할 작업에서 모두 성능을 개선하며, 기존의 TTA 방법과의 조합 시 추가적인 이점이 있음을 보여주었습니다.



### Uncertainty Modelling and Robust Observer Synthesis using the Koopman Operator (https://arxiv.org/abs/2410.01057)
Comments:
          16 pages, 15 figures

- **What's New**: 이 논문은 Koopman operator를 사용하여 모델링된 시스템 집합에 대한 강력한 비선형 관측기 합성 방법을 제안합니다.

- **Technical Details**: Koopman operator는 비선형 시스템을 무한 차원 선형 시스템으로 다시 쓸 수 있게 해줍니다. 데이터에서 Koopman operator의 유한 차원 근사치를 식별할 수 있으며, 이를 통해 비선형 시스템의 거의 선형 모델을 얻을 수 있습니다. 제안된 관측기 합성 방법은 이러한 선형성 덕분에 다수의 Koopman 모델에 대한 불확실성을 주파수 영역에서 정량화할 수 있게 해줍니다. 혼합 $	ext{H}_2$-$	ext{H}_	ext{∞}$ 최적 제어를 통해 강력한 비선형 Koopman 관측기가 합성됩니다.

- **Performance Highlights**: 여러 개의 모터 드라이브를 사용하여 실험적 방법을 입증하였으며, 제조 변동성이 주파수 영역에서 특성화되었습니다.



### Single-Shot Learning of Stable Dynamical Systems for Long-Horizon Manipulation Tasks (https://arxiv.org/abs/2410.01033)
Comments:
          7 pages, submitted to ICRA 2025

- **What's New**: 이번 연구는 기존의 긴 수평 작업 학습 방법을 확장하여, 트레이닝 데이터의 양을 줄이면서도 작업 성공률을 향상시키는 새로운 방법론을 제시합니다. 이 방법는 긴 시연을 세분화하여 각 서브 목표를 정의하고, 전 세계적으로 안정적인 동적 시스템 정책을 학습합니다.

- **Technical Details**: 논문에서는 로봇이 제공된 긴 시연을 통해 각 서브 목표에 도달할 수 있도록 동적 시스템 정책을 학습합니다. 이 정책은 감각적 노이즈 및 확률적 섭동에도 불구하고 로봇이 각 서브 목표를 효과적으로 달성할 수 있도록 가이드를 제공합니다. 이를 통해 스테이블(Stable)한 동적 정책을 학습하고 이를 실제 로봇 플랫폼에 직접 전이할 수 있는 성능을 입증하였습니다.

- **Performance Highlights**: 시뮬레이션과 실제 환경 실험을 통해 제안된 방법의 효과적인 성능을 보여주었으며, 단일 시연으로 실제 시스템에 정책을 배포할 수 있음을 입증했습니다. 이 연구는 로봇의 안전성과 작업 수행의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Investigating the Synergistic Effects of Dropout and Residual Connections on Language Model Training (https://arxiv.org/abs/2410.01019)
Comments:
          5 pages, 4 figures

- **What's New**: 본 논문은 언어 모델 훈련에서 과적합(overfitting)을 완화하기 위한 dropout 기법의 중대한 역할을 조사합니다. 다양한 dropout 비율이 개별 레이어 및 잔여 연결에 미치는 영향을 분석하며, Tiny Shakespeare 데이터셋을 사용하여 속도와 검증 오류에 미치는 결과를 연구했습니다.

- **Technical Details**: 변화하는 dropout 비율과 잔여 연결을 Transformer 구조에서 언어 모델링에 적용하며, 측정되는 주요 요소에는 훈련 수렴(convergence), 검증 오류(validation error), 일반화 가능성(generality)이 포함됩니다. 각 레이어에 대한 dropout 기법을 포함하여 attention 및 MLP 레이어의 조합, 잔여 연결의 스킵된 레이어의 수를 고려합니다.

- **Performance Highlights**: 실험 결과는 dropout 기법이 정규화(regularization) 및 잔여 연결이 수렴(covrgence)에 유익한 영향을 미친다는 점을 확인했으며, 최적의 딥 뉴럴 네트워크 수렴과 일반화를 위한 잔여 연결 깊이와 dropout 비율 간의 중요한 트레이드오프(trade-off)를 발견했습니다.



### Machine Learning-Assisted Intrusion Detection for Enhancing Internet of Things Security (https://arxiv.org/abs/2410.01016)
- **What's New**: IoT(Internet of Things) 보안 강화를 위한 최신 머신 러닝 기반 침입 탐지 전략에 대한 연구가 진행되었습니다.

- **Technical Details**: 본 논문은 IoT 보안의 침입 탐지 시스템에 대한 머신 러닝 응용에 중점을 두며, 실시간 반응성(real-time responsiveness), 탐지 정확도(detection accuracy), 알고리즘 효율성(algorithm efficiency) 등을 강조합니다. 또한 기존 접근 방식에 대한 분류법(taxonomy)을 제공하고, 주요 연구를 검토하였습니다.

- **Performance Highlights**: 현재 IoT 보안 프레임워크의 한계를 언급하며, 향후 연구 방향과 개발을 위한 실질적인 통찰력을 제시합니다.



### Compressing Recurrent Neural Networks for FPGA-accelerated Implementation in Fluorescence Lifetime Imaging (https://arxiv.org/abs/2410.00948)
Comments:
          8 pages, 2 figures

- **What's New**: 이 연구는 실시간 형광 수명 이미징(FLI)을 위한 순환 신경망(RNN) 모델의 압축에 중점을 두어 FPGA 기반의 하드웨어에서의 적용 가능성을 높였습니다.

- **Technical Details**: 연구에서는 가중치 감소(weight reduction), 지식 증류(knowledge distillation, KD), 사후 훈련 양자화(post-training quantization, PTQ), 양자화 인식 훈련(quantization-aware training, QAT) 등 다양한 압축 기법을 평가하여 모델 크기와 계산 부하를 줄이며 추론 정확도를 유지하고자 했습니다.

- **Performance Highlights**: 압축된 RNN 모델인 Seq2SeqLite는 8비트 정밀도에서 계산 효율성과 예측 정확성의 균형을 이뤘으며, KD를 적용하여 모델 매개변수 크기를 98% 감소시켰습니다. 이는 데이터 캡처 중 FPGA에서 동시 실시간 FLI 분석에 적합하다는 것을 의미합니다.



### Spectral Graph Sample Weighting for Interpretable Sub-cohort Analysis in Predictive Models for Neuroimaging (https://arxiv.org/abs/2410.00946)
- **What's New**: 본 논문은 뇌 질환의 이질성을 모델링하기 위해 머신러닝에서 샘플 가중치를 새롭게 설정하는 방식을 제안합니다. 특히, 샘플 가중치를 스펙트럼 기반의 군집(graph)에서 추출한 고유벡터의 선형 결합으로 정의하여 예측력을 향상시키고 해석력을 높이는 방법을 소개합니다.

- **Technical Details**: 샘플 가중치 할당은 머신러닝 모델의 분류 정확도를 향상하는 데 필요하며, 다양한 인구 통계학적 요인(예: 성별, 나이 등)과의 관계를 반영하는 방식으로 설정됩니다. 본 연구에서는 대규모의 데이터셋을 활용하여 스펙트럼 군집(graph) 모델을 통해 샘플 가중치를 학습하고, 각 요인에 기반한 가중치 변화를 효과적으로 다룹니다. 실험에서는 NCANDA와 ADNI 데이터셋을 통해 인간의 알콜 소비와 알츠하이머 질병 관련 예측 정확도를 측정합니다.

- **Performance Highlights**: 기존 샘플 가중치 방법들과 비교할 때, 제공된 샘플 가중치 방식은 더 나은 해석성과 특정 하위 집단의 예측 정확도를 드러내고 있습니다. 특히, 두 개의 대규모 데이터셋을 활용한 결과, 성별, 사회경제적 상태, 가족의 음주 이력 등의 요인이 알콜 소비 예측에 미치는 긍정적인 영향을 나타내었습니다.



### Evaluating Deep Regression Models for WSI-Based Gene-Expression Prediction (https://arxiv.org/abs/2410.00945)
- **What's New**: 이 연구에서는 전체 슬라이드 이미지(WSI)로부터 직접적으로 mRNA 유전자 발현 프로필을 예측하는 딥 러닝 모델을 제안합니다. 이는 비용 효율적이면서도 폭넓게 접근 가능한 분자 표현형 분석(Molecular phenotyping)을 제공할 수 있습니다.

- **Technical Details**: 연구는 WSI 기반의 유전자 발현 예측 모델의 높은 차원 회귀 문제(Regression problem)와 관련된 설계 선택들에 대해 자세히 분석합니다. 제안된 방법으로는 20530개의 유전자를 동시에 회귀하는 단일 모델을 훈련시키는 것이 있습니다.

- **Performance Highlights**: 단일 모델 훈련 방식이 계산적으로 효율적이며 매우 강력한 기준선(baseline)을 제공한다는 결론을 내립니다.



### GAMMA-PD: Graph-based Analysis of Multi-Modal Motor Impairment Assessments in Parkinson's Diseas (https://arxiv.org/abs/2410.00944)
Comments:
          Accepted by the 6th Workshop on GRaphs in biomedicAl Image anaLysis (GRAIL) at the 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024). 12 pages, 3 figures, 2 tables, Source Code: this https URL

- **What's New**: 본 논문에서는 Parkinson's Disease (PD)의 다중 모달 클리닉 데이터 분석을 위한 새로운 이질적 하이퍼그래프 융합 프레임워크인 GAMMA-PD를 제안합니다. GAMMA-PD는 이미징(imaging) 및 비이미징(non-imaging) 데이터를 통합하여 환자 프로필과 증상 하위 유형 간의 유사성을 보존하며, 높은 차원의 비대칭 정보를 학습합니다.

- **Technical Details**: GAMMA-PD는 의료 지식 클러스터링을 기반으로 한 도메인 특화 하이퍼엣지(hyperedge) 유형을 도입하여 임상적으로 관련된 패턴을 학습합니다. 또한, 피처 기반의 주의 가중치 메커니즘을 설계하여 각 예측 작업에 대해 가장 중요한 피처와 관계를 식별하고 우선순위를 매깁니다. 이를 통해 동적 특성을 보존한 채로 환자 간의 특성의 연관성을 학습하고, 임상적으로 유의미한 설명을 생성합니다.

- **Performance Highlights**: Parkinson's Progression Markers Initiative (PPMI) 및 사설 데이터 세트를 통한 평가에서 GAMMA-PD는 PD의 운동 장애 증상을 예측하는 데 있어 성능 향상을 보여줍니다. 또한, 이 모델은 다중 모달 의료 데이터를 활용하여 PD 증상 및 질병의 이질성 분석에 있어 임상적 설명과 해석 가능성을 제공합니다.



### AR-Sieve Bootstrap for the Random Forest and a simulation-based comparison with rangerts time series prediction (https://arxiv.org/abs/2410.00942)
- **What's New**: 본 논문은 기존의 IID 부트스트래핑(IID bootstrapping)을 AR-Sieve Bootstrap (ARSB)으로 대체하여 랜덤 포레스트(Random Forest, RF) 모델의 예측 성능을 향상시키기 위한 새로운 방법론을 제안합니다. ARSB는 자기회귀(Autoregressive) 프로세스를 상정하여 샘플을 추출하는 기법입니다.

- **Technical Details**: 랜덤 포레스트(RF)는 부트스트랩 집합(Bootstrap aggregating) 기술을 사용하여 여러 개의 기본 학습기(base learners)의 예측을 결합하여 성능을 향상시킵니다. 제안된 ARSB는 기본적으로 AR 모델에서 부트스트랩 샘플을 추출하고, 다섯 가지 RF 변형 및 자가 회귀 모델에 대한 벤치마크 모델과 비교하여 성능을 평가합니다.

- **Performance Highlights**: 수치 시뮬레이션 결과, ARSB를 사용한 RF 모델이 다른 부트스트래핑 전략에 비해 더 높은 정확도를 보였습니다. 그러나 이러한 성능 향상은 효율성에서의 일부 손실을 동반함을 유의해야 합니다.



### StreamEnsemble: Predictive Queries over Spatiotemporal Streaming Data (https://arxiv.org/abs/2410.00933)
Comments:
          13 pages

- **What's New**: 본 논문은 스페이셜 템포럴(spatiotemporal, ST) 데이터 스트림에 대한 예측 쿼리를 처리하기 위한 새로운 접근법인 StreamEnsemble을 제안합니다. 이 방법은 기계 학습 모델을 동적으로 선택하고 할당하여 데이터 분포의 변화를 효과적으로 처리합니다.

- **Technical Details**: StreamEnsemble은 ST 데이터의 시계열 분포와 각 모델의 특성에 따라 기계 학습 모델을 선정하여 조합하는 방법입니다. 이 방법은 데이터가 유입되는 시점에서의 개별 시계열 윈도우의 분포를 바탕으로 유사한 시계를 클러스터링하고 최적 모델을 식별하는 일반화 오류 추정 기술을 사용합니다.

- **Performance Highlights**: 실험 결과, StreamEnsemble은 전통적인 앙상블 방법 및 단일 모델 접근법에 비해 10배 이상의 예측 오차 감소를 달성하며, ST 데이터 스트림의 복잡한 변동성을 처리하는데 있어 월등한 성능을 보여줍니다.



### A Knowledge-Informed Large Language Model Framework for U.S. Nuclear Power Plant Shutdown Initiating Event Classification for Probabilistic Risk Assessmen (https://arxiv.org/abs/2410.00929)
- **What's New**: 본 논문은 원자력 발전소의 저전력 폐쇄 확률적 위험 평가를 위한 종료 유발 사건(SDIE, Shutdown Initiating Events) 식별 및 분류 방법을 제안합니다. 기존 방법의 한계를 극복하기 위해 지식에 기반한 머신러닝 모델과 대형 언어 모델(LLM, Large Language Model)의 하이브리드 파이프라인을 통합한 방안을 제시합니다.

- **Technical Details**: 제안된 파이프라인은 두 단계로 구성됩니다. 첫 번째 단계는 44개의 SDIE 텍스트 패턴을 사용한 사전 선별로, 이는 여섯 가지 SDIE 유형에서 도출된 주요 키워드와 구문으로 구성됩니다. 이 패턴을 기반으로 한 텍스트 벡터화는 간단한 이진 분류기를 사용하여 매우 구분 가능한 피처 벡터를 생성합니다. 두 번째 단계는 Bidirectional Encoder Representations from Transformers (BERT) 기반 LLM을 구축하여, 대규모 데이터셋에서 자가 지도 학습 방식으로 일반적인 영어 언어 표현을 학습하고, 이를 SDIE 분류에 맞게 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: 실험 결과, 사전 선별 단계에서 97% 이상의 비SDIE를 제외할 수 있었으며, LLM을 사용한 SDIE 분류에서 평균 정확도는 93.4%에 달했습니다.



### On the topology and geometry of population-based SHM (https://arxiv.org/abs/2410.00923)
- **What's New**: 본 논문은 Population-Based Structural Health Monitoring (PBSHM)의 새로운 접근 방식을 제시하며, 이는 구조물 간의 데이터 공유를 통해 손상 진단을 개선하는 방법입니다. 특히, 기존 연구에서 그래프 형태로 표현된 구조물들을 더 엄밀하게 분석하기 위해 매개변수적 구조 가족으로 확장하는 방법을 모색하고 있습니다.

- **Technical Details**: PBSHM은 Sparse Data를 효과적으로 활용하는 Transfer Learning (TL) 기법을 기반으로 하며, 구조물의 데이터를 벡터 번들 내에서 연속적으로 변환할 수 있는 수학적인 배경을 제공합니다. 이를 통해 구조물과 데이터 간의 연결을 더욱 매끄럽게 할 수 있도록 한 단계 더 발전시킵니다.

- **Performance Highlights**: 이 새로운 기하학적 접근 방식은 PBSHM 시스템의 데이터 수집과 처리에 실질적인 개선을 가져올 것으로 기대됩니다. 구조물 간 데이터의 연속적 상호작용이 가능해지며, 이는 의료 및 엔지니어링 분야에서 SHM 시스템의 적용 가능성을 확장하는 데 기여할 것으로 보입니다.



### Show Me What's Wrong!: Combining Charts and Text to Guide Data Analysis (https://arxiv.org/abs/2410.00727)
- **What's New**: 본 논문에서는 다차원 데이터셋에서 이상 패턴을 분석하고 탐지하는 복잡한 과정을 간소화하기 위한 도구를 제안합니다. 이 도구는 자동 정보 하이라이팅, Large Language Model (LLM) 기반의 텍스트 인사이트 및 시각적 분석을 통합하여 사용자의 탐색을 지원합니다.

- **Technical Details**: 제안하는 도구는 사용자가 선정한 데이터 분석 영역에 대한 텍스트 및 그래픽 요약을 제공합니다. 사용자는 데이터를 분리하여 탐색하고, 각 영역에서 필요한 정보를 쉽게 이해할 수 있으며, 각 분석 영역(Knowledge Areas, KAs)에 대한 직관적인 시각적 신호로 추가적인 주의를 기울여야 할 부분을 찾을 수 있습니다. 또한, Hallucination Detection 시스템을 통해 잘못된 정보를 생성할 가능성을 최소화합니다.

- **Performance Highlights**: 일곱 명의 도메인 전문가를 대상으로 한 연구 결과, 제안된 도구가 탐색 분석을 효과적으로 지원하고 의심스러운 정보를 식별하는 데 도움을 준 것으로 나타났습니다. 사용자는 정보 과부하 없이 빠르고 정확한 인사이트를 얻을 수 있습니다.



### Scaling Optimal LR Across Token Horizons (https://arxiv.org/abs/2409.19913)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델) 훈련 시 토큰 지평(token horizon)에 따른 최적 학습률(learning rate, LR)의 변화를 대규모 실험을 통해 조사했습니다. LLM 훈련에 있어 하이퍼파라미터 전이(hyperparameter transfer)가 토큰 지평을 가로막고 있는 중요한 문제로 부각되었습니다.

- **Technical Details**: 이 연구에서는 최적 LR이 토큰 지평에 강하게 의존하며, 긴 훈련 기간에는 더 작은 LR이 필요하다는 점을 보여줍니다. 또한, 최적 LR은 스케일링 법칙(scaling law)을 따르는데, 이를 통해 긴 토큰 지평에 대한 최적 LR을 짧은 지평에서 추론할 수 있습니다. 실험은 Megatron 코드베이스(Megatron codebase)와 RefinedWeb 데이터셋을 바탕으로 진행되었습니다.

- **Performance Highlights**: LLama-1 모델이 너무 높은 LR을 사용했다는 것을 증명하며, 이로 인해 성능 저하가 발생했다는 점을 강조합니다. 이 연구는 데이터 크기 간 하이퍼파라미터 전이가 LLM 훈련에서 간과된 중요한 구성 요소임을 주장합니다.



